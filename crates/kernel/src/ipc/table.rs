//! [`Ipc`]: servers, sessions, named ports and the requests between them, as plain data in the
//! scheduler's table (kernel spec, "Tables behind a dispatcher": the scheduler table holds
//! "wait lists, events and sessions").
//!
//! The model is Horizon's (kernel spec, "IPC", rule 1): a session has a client end, which
//! handles name ([`Object::Session`]), and a server end, which its server owns; between them is
//! a request in flight and a queue of waiting ones. Which request starts when is the server's
//! [`RequestQueue`]'s to say. The handler code behind a server is not here: it is not plain data,
//! and the dispatcher keeps it (`crate::dispatch`).
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), queueing and close semantics only:
//!
//! - **Sending** (`k_server_session.cpp`, `KServerSession::OnRequest`): a request on a session
//!   whose server end is closed is [`KernelError::SessionClosed`] at once; otherwise it is
//!   added to the session's list and the sender waits. Here it starts at once if the queue says
//!   so ([`Ipc::send`]).
//! - **A request holds the session** (`svc_ipc.cpp`, `SendSyncRequestImpl`: the client session
//!   stays open, through a `KScopedAutoObject`, until the request is answered), so closing the
//!   last handle to a session with a request in flight destroys it only once the request is over.
//! - **Closing the server end** (`KServerSession::Destroy`, `CleanupRequests`): the request in
//!   flight and every waiting one end with [`KernelError::SessionClosed`] ([`Ipc::finish`] with
//!   [`Finish::ServerClosed`]).
//! - **Closing the client end** (`KClientSession::Destroy`, `KServerSession::OnClientClosed`):
//!   the server is told ([`Notice::Closed`]). A synchronous request cannot be pending then,
//!   because it holds the session.
//!
//! An HLE server is told of a session made by connecting to it before it is given a request on
//! it ([`Notice::Accept`]). A session a server makes on itself (a cloned session, a new
//! sub-interface) needs no notice: the server records it when it makes it.

use std::collections::{BTreeMap, VecDeque};

use starquake_trace::{ServiceId, ThreadId};

use super::message::{MessageBuffer, Translated};
use super::port::PortName;
use super::queue::{Admission, QueueError, RequestQueue};
use crate::handle::{Handle, ObjectRefs};
use crate::id::{ObjectId, ServerId};
use crate::object::{Object, ObjectError, Objects};
use crate::result::{KernelError, SvcError};
use crate::slab::Slab;

/// Why the IPC table refused an operation. Only [`IpcError::Kernel`] is a guest's doing; every
/// other variant is a caller's bookkeeping gone wrong.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IpcError {
    /// A Horizon result the guest sees.
    #[error(transparent)]
    Kernel(#[from] KernelError),
    /// The id names no server.
    #[error("{0:?} names no server")]
    UnknownServer(ServerId),
    /// The id names no session.
    #[error("{0:?} names no session")]
    UnknownSession(ObjectId),
    /// A port is already registered under the name.
    #[error("a port is already named {0}")]
    PortTaken(PortName),
    /// A server is already known by the service number.
    #[error("a server is already service {0:?}")]
    ServiceTaken(ServiceId),
    /// The session has no request in flight.
    #[error("session {0:?} has no request in flight")]
    NothingInFlight(ObjectId),
    /// The server table is full.
    #[error("the server table is full")]
    Full,
    /// The request queue refused.
    #[error(transparent)]
    Queue(#[from] QueueError),
    /// The object table refused.
    #[error(transparent)]
    Objects(#[from] ObjectError),
}

impl From<IpcError> for SvcError {
    /// The one place an IPC table failure becomes what a system call returns: a Horizon result
    /// stays one, and every other failure is the kernel's own bookkeeping.
    fn from(error: IpcError) -> Self {
        match error {
            IpcError::Kernel(error) => Self::Kernel(error),
            error => Self::internal(&error),
        }
    }
}

/// Who sent a request, and where its reply goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sender {
    /// The sending guest thread, as the event stream names it.
    ///
    /// PROVISIONAL (walking skeleton, #74): the event stream's id, which is what the dispatcher
    /// is told of its caller. Kernel slice 10 names the scheduler's thread here, so that a
    /// woken sender can be made runnable.
    pub thread: ThreadId,
    /// The core it called from.
    pub core: u8,
    /// Where its message buffer is (its TLS block): the reply is written there.
    pub message_address: u64,
    /// The session handle it passed, for the system call's event.
    pub session_handle: Handle,
}

/// A request: its sender, its message after the kernel's half of it was translated, and the
/// references it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// Who sent it.
    pub sender: Sender,
    /// The message, as the server reads it.
    pub message: Box<MessageBuffer>,
    /// The objects its handles named, each held once for the request.
    pub translated: Translated,
}

/// What a server is told about its sessions, in order, before it is next given a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Notice {
    /// A client connected: open the session with the server's root interface.
    Accept(ObjectId),
    /// The client end is gone: forget the session.
    Closed(ObjectId),
}

/// A request that is over, and what its sender is told.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Done {
    /// The request. Its message holds the reply when `result` is success.
    pub request: Request,
    /// The result of the sender's `SendSyncRequest`.
    pub result: Result<(), KernelError>,
}

/// A request [`Ipc::send`] did not take, handed back so that its holdings can be released.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused {
    /// Why.
    pub error: IpcError,
    /// The request.
    pub request: Box<Request>,
}

/// How a started request ended, as [`Ipc::finish`] is told.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finish {
    /// The server wrote this reply.
    Replied(Box<MessageBuffer>),
    /// The server will reply later: the request stays in flight and the worker is free.
    Deferred,
    /// The server closed its end of the session instead of replying.
    ServerClosed,
    /// The request never reached the server: it fails with this result.
    Failed(KernelError),
}

/// One server.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ServerRow {
    name: String,
    service: ServiceId,
    queue: RequestQueue,
    notices: VecDeque<Notice>,
}

/// One session.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SessionRow {
    server: ServerId,
    /// Whether the server end is open. Once closed, a new request is refused.
    server_open: bool,
    in_flight: Option<Request>,
    waiting: VecDeque<Request>,
}

/// The IPC table. See the module docs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ipc {
    servers: Slab<ServerId, ServerRow>,
    sessions: BTreeMap<ObjectId, SessionRow>,
    ports: BTreeMap<PortName, ServerId>,
    services: BTreeMap<ServiceId, ServerId>,
}

impl Ipc {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a server known as `service`, with one worker ([`super::DEFAULT_WORKERS`]). `name` is
    /// for logs.
    pub fn add_server(
        &mut self,
        name: impl Into<String>,
        service: ServiceId,
    ) -> Result<ServerId, IpcError> {
        if self.services.contains_key(&service) {
            return Err(IpcError::ServiceTaken(service));
        }
        let row = ServerRow {
            name: name.into(),
            service,
            queue: RequestQueue::new(),
            notices: VecDeque::new(),
        };
        let server = self.servers.insert(row).map_err(|_| IpcError::Full)?;
        self.services.insert(service, server);
        Ok(server)
    }

    /// Makes `server` reachable by `ConnectToNamedPort(name)`.
    pub fn name_port(&mut self, name: PortName, server: ServerId) -> Result<(), IpcError> {
        self.server(server)?;
        if self.ports.contains_key(&name) {
            return Err(IpcError::PortTaken(name));
        }
        self.ports.insert(name, server);
        Ok(())
    }

    /// The server a port name leads to.
    #[must_use]
    pub fn port(&self, name: &PortName) -> Option<ServerId> {
        self.ports.get(name).copied()
    }

    /// The server known as `service`.
    #[must_use]
    pub fn service(&self, service: ServiceId) -> Option<ServerId> {
        self.services.get(&service).copied()
    }

    /// A server's name.
    pub fn server_name(&self, server: ServerId) -> Result<&str, IpcError> {
        Ok(&self.server(server)?.name)
    }

    /// A server's service number.
    pub fn server_service(&self, server: ServerId) -> Result<ServiceId, IpcError> {
        Ok(self.server(server)?.service)
    }

    /// The servers, in id order.
    pub fn servers(&self) -> impl Iterator<Item = ServerId> + '_ {
        self.servers.iter().map(|(id, _)| id)
    }

    /// The server's request queue.
    pub fn queue(&self, server: ServerId) -> Result<&RequestQueue, IpcError> {
        Ok(&self.server(server)?.queue)
    }

    fn server(&self, server: ServerId) -> Result<&ServerRow, IpcError> {
        self.servers
            .get(server)
            .map_err(|_| IpcError::UnknownServer(server))
    }

    fn server_mut(&mut self, server: ServerId) -> Result<&mut ServerRow, IpcError> {
        self.servers
            .get_mut(server)
            .map_err(|_| IpcError::UnknownServer(server))
    }

    fn session(&self, session: ObjectId) -> Result<&SessionRow, IpcError> {
        self.sessions
            .get(&session)
            .ok_or(IpcError::UnknownSession(session))
    }

    fn session_mut(&mut self, session: ObjectId) -> Result<&mut SessionRow, IpcError> {
        self.sessions
            .get_mut(&session)
            .ok_or(IpcError::UnknownSession(session))
    }

    /// Whether `session` names a session.
    #[must_use]
    pub fn is_session(&self, session: ObjectId) -> bool {
        self.sessions.contains_key(&session)
    }

    /// The server a session is on.
    pub fn server_of(&self, session: ObjectId) -> Result<ServerId, IpcError> {
        Ok(self.session(session)?.server)
    }

    /// Whether the session's server end is open.
    pub fn is_server_open(&self, session: ObjectId) -> Result<bool, IpcError> {
        Ok(self.session(session)?.server_open)
    }

    /// The request in flight on a session, if any.
    pub fn in_flight(&self, session: ObjectId) -> Result<Option<&Request>, IpcError> {
        Ok(self.session(session)?.in_flight.as_ref())
    }

    /// How many requests wait on a session behind the one in flight.
    pub fn waiting(&self, session: ObjectId) -> Result<usize, IpcError> {
        Ok(self.session(session)?.waiting.len())
    }

    /// Opens a session on `server` and returns its client end, a new object holding one
    /// reference, its creator's (`KAutoObject::Create`). With `notice`, the server is told to
    /// accept it ([`Notice::Accept`]): a session made by connecting to the server.
    pub fn open_session(
        &mut self,
        objects: &mut Objects,
        server: ServerId,
        notice: bool,
    ) -> Result<ObjectId, IpcError> {
        self.server(server)?;
        let session = objects.create(Object::Session)?;
        let row = SessionRow {
            server,
            server_open: true,
            in_flight: None,
            waiting: VecDeque::new(),
        };
        self.sessions.insert(session, row);
        if notice {
            self.server_mut(server)?
                .notices
                .push_back(Notice::Accept(session));
        }
        Ok(session)
    }

    /// Takes back a session [`Ipc::open_session`] just made, whose creator's reference is the
    /// only one and whose server has not been given it: what a system call does when it cannot
    /// give the caller a handle to it. The server's notice to accept it is withdrawn too.
    pub fn discard_session(
        &mut self,
        objects: &mut Objects,
        session: ObjectId,
    ) -> Result<(), IpcError> {
        let server = self.server_of(session)?;
        self.sessions.remove(&session);
        let row = self.server_mut(server)?;
        row.notices
            .retain(|notice| *notice != Notice::Accept(session));
        objects.release(session)?;
        Ok(())
    }

    /// Drops one reference to `object`. When it was the last, and the object was a session's
    /// client end, the session is gone too and its server is told, if its end is still open.
    pub fn release(&mut self, objects: &mut Objects, object: ObjectId) -> Result<(), IpcError> {
        if objects.release(object)?.is_some() {
            self.forget(object)?;
        }
        Ok(())
    }

    /// The object `object` names has lost its last reference (closing a handle released it):
    /// if it was a session's client end, the session goes and its server is told.
    pub fn forget(&mut self, object: ObjectId) -> Result<(), IpcError> {
        let Some(row) = self.sessions.remove(&object) else {
            return Ok(());
        };
        if row.in_flight.is_some() || !row.waiting.is_empty() {
            // A request holds its session, so this cannot happen while one is pending.
            tracing::error!(?object, "a session was destroyed with requests pending");
        }
        let server = self.server_mut(row.server)?;
        server.queue.close(object);
        if row.server_open {
            server.notices.push_back(Notice::Closed(object));
        }
        Ok(())
    }

    /// The notices waiting for `server`, oldest first. The caller gives them to the server's
    /// handler before its next request.
    pub fn take_notices(&mut self, server: ServerId) -> Result<Vec<Notice>, IpcError> {
        Ok(self.server_mut(server)?.notices.drain(..).collect())
    }

    /// Whether any server has notices waiting.
    #[must_use]
    pub fn has_any_notices(&self) -> bool {
        self.servers.iter().any(|(_, row)| !row.notices.is_empty())
    }

    /// Whether `server` has notices waiting.
    pub fn has_notices(&self, server: ServerId) -> Result<bool, IpcError> {
        Ok(!self.server(server)?.notices.is_empty())
    }

    /// A guest thread sent `request` on `session`. The request takes a reference to the
    /// session. [`Admission::Start`]: it is in flight and its handler is to run now.
    /// [`Admission::Queued`]: the sender waits; [`Ipc::start_next`] names the session when its
    /// turn comes.
    ///
    /// [`KernelError::SessionClosed`] if the server end is closed. A request that is refused is
    /// handed back, not taken.
    pub fn send(
        &mut self,
        objects: &mut Objects,
        session: ObjectId,
        request: Request,
    ) -> Result<Admission, Refused> {
        match self.admit(objects, session) {
            Ok(admission) => {
                if let Some(row) = self.sessions.get_mut(&session) {
                    match admission {
                        Admission::Start => row.in_flight = Some(request),
                        Admission::Queued => row.waiting.push_back(request),
                    }
                }
                Ok(admission)
            }
            Err(error) => Err(Refused {
                error,
                request: Box::new(request),
            }),
        }
    }

    /// Adds a notice for `server`: what the dispatcher does when a handler failed and still has
    /// a session the kernel has closed.
    pub fn notify(&mut self, server: ServerId, notice: Notice) -> Result<(), IpcError> {
        self.server_mut(server)?.notices.push_back(notice);
        Ok(())
    }

    /// The queue's and the reference count's half of [`Ipc::send`]: all or nothing.
    fn admit(&mut self, objects: &mut Objects, session: ObjectId) -> Result<Admission, IpcError> {
        let row = self.session(session)?;
        if !row.server_open {
            return Err(KernelError::SessionClosed.into());
        }
        let server = row.server;
        let admission = self.server_mut(server)?.queue.submit(session)?;
        if let Err(error) = objects.open(session) {
            // Undo the queue's count before failing: nothing was taken.
            let queue = &mut self.server_mut(server)?.queue;
            match admission {
                Admission::Start => queue.finish(session)?,
                Admission::Queued => queue.withdraw(session)?,
            }
            return Err(ObjectError::Refs(error).into());
        }
        Ok(admission)
    }

    /// The session of `server` whose oldest waiting request starts now, if one may: that
    /// request is then in flight. Call it after every [`Ipc::finish`] and [`Ipc::complete`]
    /// until it returns `None`.
    pub fn start_next(&mut self, server: ServerId) -> Result<Option<ObjectId>, IpcError> {
        let Some(session) = self.server_mut(server)?.queue.start_next() else {
            return Ok(None);
        };
        let row = self.session_mut(session)?;
        let request = row.waiting.pop_front();
        row.in_flight = request;
        if row.in_flight.is_none() {
            return Err(IpcError::NothingInFlight(session));
        }
        Ok(Some(session))
    }

    /// The running request on `session` ended as `finish` says. Returns the requests that are
    /// over, whose references the caller releases with [`Ipc::release_request`] after telling
    /// their senders: the one in flight, unless it was deferred, and for
    /// [`Finish::ServerClosed`] every waiting one too.
    pub fn finish(&mut self, session: ObjectId, finish: Finish) -> Result<Vec<Done>, IpcError> {
        let server = self.server_of(session)?;
        let queue = &mut self.server_mut(server)?.queue;
        match &finish {
            Finish::Deferred => queue.defer(session)?,
            Finish::ServerClosed => {
                queue.close(session);
            }
            Finish::Replied(_) | Finish::Failed(_) => queue.finish(session)?,
        }
        let row = self.session_mut(session)?;
        let mut done = Vec::new();
        match finish {
            Finish::Deferred => {
                if row.in_flight.is_none() {
                    return Err(IpcError::NothingInFlight(session));
                }
            }
            Finish::Replied(reply) => {
                let mut request = row
                    .in_flight
                    .take()
                    .ok_or(IpcError::NothingInFlight(session))?;
                request.message = reply;
                done.push(Done {
                    request,
                    result: Ok(()),
                });
            }
            Finish::Failed(error) => {
                let request = row
                    .in_flight
                    .take()
                    .ok_or(IpcError::NothingInFlight(session))?;
                done.push(Done {
                    request,
                    result: Err(error),
                });
            }
            Finish::ServerClosed => {
                row.server_open = false;
                let closed = Err(KernelError::SessionClosed);
                let pending = row
                    .in_flight
                    .take()
                    .into_iter()
                    .chain(row.waiting.drain(..));
                done.extend(pending.map(|request| Done {
                    request,
                    result: closed,
                }));
            }
        }
        Ok(done)
    }

    /// The deferred request on `session` was answered with `reply`. Returns it, over; the
    /// caller releases its references with [`Ipc::release_request`].
    pub fn complete(
        &mut self,
        session: ObjectId,
        reply: Box<MessageBuffer>,
    ) -> Result<Done, IpcError> {
        let server = self.server_of(session)?;
        self.server_mut(server)?.queue.complete(session)?;
        let row = self.session_mut(session)?;
        let mut request = row
            .in_flight
            .take()
            .ok_or(IpcError::NothingInFlight(session))?;
        request.message = reply;
        Ok(Done {
            request,
            result: Ok(()),
        })
    }

    /// Releases what a finished request held: its objects, then its session. Either may have
    /// been the last reference.
    pub fn release_request(
        &mut self,
        objects: &mut Objects,
        session: ObjectId,
        request: &Request,
    ) -> Result<(), IpcError> {
        for object in &request.translated.held {
            self.release(objects, *object)?;
        }
        self.release(objects, session)
    }

    /// Releases what a request that was never taken (refused by [`Ipc::send`], or failed in
    /// translation before it) held: its translated objects only.
    pub fn release_translated(
        &mut self,
        objects: &mut Objects,
        translated: &Translated,
    ) -> Result<(), IpcError> {
        for object in &translated.held {
            self.release(objects, *object)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
