//! [`Server`]: one service family's sessions, and the path a request takes through them.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use starquake_ipc::{
    BUFFER_SIZE, Body, CmifRequest, CommandType, IpcError, Message, ReplyBuilder, SessionKind,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_trace::{Env, IpcOutcome, ServiceId};

use super::commands::{Protocol, Session};
use super::context::Context;
use super::defer::ReplyToken;
use super::defer::{Completed, Deferral, Framing, Mailbox, Notify, Pending, Posted, TokenId};
use super::kernel::Kernel;
use super::payload::Writer;
use super::queue::{Admission, QueueError, RequestQueue};
use super::reply::{Moved, Reply};
use super::request::Request;
use super::result::{
    Completion, DEFERRED_REPLY_DROPPED, Fault, NOT_SUPPORTED, OUT_OF_DOMAIN_ENTRIES,
    TARGET_NOT_FOUND, UNKNOWN_COMMAND_ID,
};
use super::snapshot::{Family, Interfaces};
use super::targets::{Domain, TargetError, TargetId, Targets};
use super::trace::{self, Asked};
use super::work::Executor;

/// What a server does with a command its interface does not list
/// (`docs/specs/services.md`, "Unknown commands").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownPolicy {
    /// Log, and reply [`UNKNOWN_COMMAND_ID`]. The default in tests.
    Strict,
    /// Log once per command, and reply success with a zeroed payload. The default when bringing
    /// up a title.
    AutoStub,
}

/// What became of a message given to [`Server::handle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handled {
    /// The reply is in the buffer.
    Replied,
    /// The message closed the session. The buffer is untouched and the session is gone from the
    /// server.
    Closed,
    /// The method kept a reply token: the request stays in flight on its session, the buffer is
    /// untouched, and the caller waits. The reply is written by [`Server::deliver`] once the
    /// token completes.
    Deferred(TokenId),
}

/// A command a server answered by auto-stub.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StubbedCommand {
    /// The interface the command was sent to ([`Session::name`]).
    pub interface: &'static str,
    /// The protocol it came in.
    pub protocol: Protocol,
    /// The command id.
    pub id: u32,
}

/// The most distinct commands a server records (and logs) as auto-stubbed.
///
/// starquake's own bound: command ids are the guest's to choose, so the record must not grow
/// with them. Commands past the bound are still stubbed; they are just not recorded or logged.
pub const MAX_STUBBED_COMMANDS: usize = 4096;

/// The pointer buffer size a server reports unless told otherwise: Eden's default
/// (`eden/src/core/hle/kernel/k_process.h`, `m_pointer_buffer_size = 0x8000`).
pub const DEFAULT_POINTER_BUFFER_SIZE: u16 = 0x8000;

/// Why [`Server`] could not do what the kernel asked. The guest gets no reply from the server in
/// these cases; what it sees instead is for the caller (the kernel's session code) to decide.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ServerError {
    /// The session is not open on this server.
    #[error("session {0:?} is not open on this server")]
    UnknownSession(ObjectId),
    /// The session is already open on this server.
    #[error("session {0:?} is already open on this server")]
    SessionExists(ObjectId),
    /// The server was asked to accept a session and has no root interface to open for it.
    #[error("the server has no root interface to open for a new session")]
    NoRoot,
    /// A request cannot start on the session now: one is in flight on it or senders wait ahead,
    /// or the server has no free worker. The sender should have been queued
    /// ([`Server::submit`]).
    #[error("session {0:?} cannot start a request now")]
    Busy(ObjectId),
    /// The buffer is smaller than a message buffer. Nothing was read from it.
    #[error("message buffer holds {len} bytes, fewer than the {BUFFER_SIZE} a message needs")]
    BufferTooShort {
        /// The buffer's length.
        len: usize,
    },
    /// The message could not be parsed.
    #[error("malformed request: {0}")]
    Malformed(IpcError),
    /// The reply could not be encoded. Unreachable while replies are checked to fit before they
    /// are built; it exists so that a reply that cannot be written is an error, not a panic.
    #[error("reply cannot be written: {0}")]
    Reply(IpcError),
    /// The server's own bookkeeping failed.
    #[error(transparent)]
    Targets(#[from] TargetError),
    /// The request queue refused an operation.
    #[error(transparent)]
    Queue(#[from] QueueError),
}

/// Why a request ends without the reply its command would have built.
pub(super) enum Refusal {
    /// The caller gets a reply holding only this result.
    Result(ResultCode),
    /// The server failed; the caller gets no reply.
    Server(ServerError),
}

impl From<ResultCode> for Refusal {
    fn from(code: ResultCode) -> Self {
        Self::Result(code)
    }
}

impl From<Fault> for Refusal {
    fn from(fault: Fault) -> Self {
        Self::Result(fault.result())
    }
}

impl From<KernelError> for Refusal {
    fn from(error: KernelError) -> Self {
        Self::Result(error.code())
    }
}

impl From<TargetError> for Refusal {
    fn from(error: TargetError) -> Self {
        Self::Server(error.into())
    }
}

impl From<ServerError> for Refusal {
    fn from(error: ServerError) -> Self {
        Self::Server(error)
    }
}

/// How a reply is framed, which follows from how its request was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Frame {
    /// CMIF without a domain header: ordinary sessions, and control requests on any session.
    Cmif,
    /// CMIF with a domain header: the request had one. Holds the session's domain.
    Domain(TargetId),
    /// TIPC, echoing the request's command id.
    Tipc(u16),
}

impl Frame {
    /// A request's reply framing. Eden writes a reply's domain header only when the request
    /// carried one (`eden/src/core/hle/service/ipc_helpers.h`, `ResponseBuilder`:
    /// `mgr->IsDomain() && ctx.HasDomainMessageHeader()`).
    fn of(message: &Message<'_>, target: TargetId) -> Self {
        match (&message.body, message.command_type) {
            (_, CommandType::Tipc(id)) => Self::Tipc(id),
            (
                Body::DomainClose(_)
                | Body::CmifRequest(CmifRequest {
                    domain: Some(_), ..
                }),
                _,
            ) => Self::Domain(target),
            _ => Self::Cmif,
        }
    }

    /// The framing as a deferred request keeps it.
    fn framing(self) -> Framing {
        match self {
            Self::Cmif => Framing::Cmif,
            Self::Domain(_) => Framing::Domain,
            Self::Tipc(command_id) => Framing::Tipc(command_id),
        }
    }

    /// The framing of a deferred request on a session that leads to `target`.
    fn of_deferred(framing: Framing, target: TargetId) -> Self {
        match framing {
            Framing::Cmif => Self::Cmif,
            Framing::Domain => Self::Domain(target),
            Framing::Tipc(command_id) => Self::Tipc(command_id),
        }
    }

    /// The start of a reply with this result.
    pub(super) fn reply(self, result: ResultCode) -> ReplyBuilder {
        match self {
            Self::Cmif => ReplyBuilder::cmif(result.raw()),
            Self::Domain(_) => ReplyBuilder::domain(result.raw()),
            Self::Tipc(command_id) => ReplyBuilder::tipc(command_id, result.raw()),
        }
    }

    /// A reply holding only a result.
    pub(super) fn only(self, result: ResultCode) -> Answer {
        Answer {
            builder: self.reply(result),
            result,
            payload: 0,
            stubbed: false,
        }
    }

    /// A successful reply holding `payload` and nothing else.
    pub(super) fn with_payload(self, payload: &[u8]) -> Answer {
        Answer {
            builder: self.reply(ResultCode::SUCCESS).data(payload),
            result: ResultCode::SUCCESS,
            payload: payload.len(),
            stubbed: false,
        }
    }
}

/// A reply ready to be written, and what its trace event says of it.
pub(super) struct Answer {
    pub(super) builder: ReplyBuilder,
    pub(super) result: ResultCode,
    /// Bytes of payload in the reply.
    pub(super) payload: usize,
    /// Whether an auto-stub wrote it.
    pub(super) stubbed: bool,
}

/// What a request came to.
enum Answered {
    /// A reply to write now.
    Now(Answer),
    /// A reply token: the reply comes later.
    Later(TokenId),
}

/// One command to dispatch.
struct Call<'m> {
    frame: Frame,
    protocol: Protocol,
    command_id: u32,
    payload: &'m [u8],
    message: &'m Message<'m>,
    in_objects: Vec<Option<TargetId>>,
}

/// Makes the root interface of a new session.
type Root = Box<dyn Fn() -> Box<dyn Session> + Send>;

/// One service family's sessions and the interfaces behind them: the Rust side of a server
/// (`docs/specs/services.md`, "Concurrency").
///
/// A method takes `&mut self` and nothing here locks: [`Server::handle`] takes the server
/// exclusively, so one handler of a server runs at a time. Which request that is, is the
/// server's [`RequestQueue`]'s to say ([`Server::submit`], [`Server::start_next`]).
pub struct Server {
    pub(super) name: String,
    service: ServiceId,
    pub(super) policy: UnknownPolicy,
    pub(super) pointer_buffer_size: u16,
    root: Option<Root>,
    pub(super) interfaces: Interfaces,
    pub(super) family: Option<Box<dyn Family>>,
    pub(super) sessions: BTreeMap<ObjectId, TargetId>,
    pub(super) targets: Targets,
    pub(super) queue: RequestQueue,
    /// Deferred requests whose token has not completed, or whose completion has not been taken.
    pub(super) pending: BTreeMap<TokenId, Pending>,
    /// The id of the next reply token.
    pub(super) next_token: u64,
    pub(super) mailbox: Arc<Mailbox>,
    pub(super) notify: Option<Arc<dyn Notify>>,
    /// The environment methods reach the host through; not saved.
    env: Option<Arc<dyn Env>>,
    /// Where methods send work that takes time; not saved.
    executor: Option<Arc<dyn Executor>>,
    stubbed: BTreeSet<StubbedCommand>,
    stub_record_full: bool,
}

impl Server {
    /// A server with no sessions. `name` is the service's name in logs (`"set:sys"`).
    #[must_use]
    pub fn new(name: impl Into<String>, policy: UnknownPolicy) -> Self {
        Self {
            name: name.into(),
            service: ServiceId(0),
            policy,
            pointer_buffer_size: DEFAULT_POINTER_BUFFER_SIZE,
            root: None,
            interfaces: Interfaces::new(),
            family: None,
            sessions: BTreeMap::new(),
            targets: Targets::default(),
            queue: RequestQueue::new(),
            pending: BTreeMap::new(),
            next_token: 0,
            mailbox: Arc::new(Mailbox::new(None)),
            notify: None,
            env: None,
            executor: None,
            stubbed: BTreeSet::new(),
            stub_record_full: false,
        }
    }

    /// Sets the size "query pointer buffer size" reports
    /// (default [`DEFAULT_POINTER_BUFFER_SIZE`]).
    #[must_use]
    pub fn with_pointer_buffer_size(mut self, size: u16) -> Self {
        self.pointer_buffer_size = size;
        self
    }

    /// Sets the number the service is known by: in the server's trace events, and to
    /// [`Kernel::connect`] (default 0).
    #[must_use]
    pub fn with_service_id(mut self, service: ServiceId) -> Self {
        self.service = service;
        self
    }

    /// Sets how the server makes the root interface of a session it is told to
    /// [`Server::accept`]: the service's own interface, new for each session.
    #[must_use]
    pub fn with_root(mut self, root: impl Fn() -> Box<dyn Session> + Send + 'static) -> Self {
        self.root = Some(Box::new(root));
        self
    }

    /// Sets the interface types the server's sessions can be, which a snapshot needs
    /// ([`Server::save`], [`Server::restore`]).
    #[must_use]
    pub fn with_interfaces(mut self, interfaces: Interfaces) -> Self {
        self.interfaces = interfaces;
        self
    }

    /// Sets the family's own state: what all the server's sessions share, reached with
    /// [`Context::family`]. It is saved once in the server's snapshot.
    #[must_use]
    pub fn with_family(mut self, family: impl Family) -> Self {
        self.family = Some(Box::new(family));
        self
    }

    /// Sets who is told when a reply token of this server completes. Tokens taken before this
    /// call are not affected: set it before the server handles requests.
    #[must_use]
    pub fn with_notify(mut self, notify: Arc<dyn Notify>) -> Self {
        self.mailbox = Arc::new(Mailbox::new(Some(Arc::clone(&notify))));
        self.notify = Some(notify);
        self
    }

    /// Sets the injectable environment the server's methods read the host through
    /// ([`Context::env`]). It is how a run reaches its host, not state: a snapshot neither saves
    /// nor replaces it.
    #[must_use]
    pub fn with_env(mut self, env: Arc<dyn Env>) -> Self {
        self.env = Some(env);
        self
    }

    /// Sets the executor the server's methods hand work that takes time to ([`Context::executor`],
    /// [`work`](super::work)). Like the environment, it is not part of a snapshot.
    #[must_use]
    pub fn with_executor(mut self, executor: Arc<dyn Executor>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// The service's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The number the service is known by.
    #[must_use]
    pub fn service_id(&self) -> ServiceId {
        self.service
    }

    /// The policy for unknown commands.
    #[must_use]
    pub fn policy(&self) -> UnknownPolicy {
        self.policy
    }

    /// Number of sessions open on the server.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Number of interface objects alive: session roots, and objects in domains. An interface
    /// shared by several sessions counts once.
    #[must_use]
    pub fn interface_count(&self) -> usize {
        self.targets.interface_count()
    }

    /// The commands answered by auto-stub so far, in order of interface, protocol and id: the
    /// list of what a title needs that nobody has implemented.
    pub fn stubbed(&self) -> impl Iterator<Item = &StubbedCommand> {
        self.stubbed.iter()
    }

    /// The family's state, as the family's own type, if the server has state of that type.
    #[must_use]
    pub fn family<T: Family>(&self) -> Option<&T> {
        let family: &dyn core::any::Any = self.family.as_deref()?;
        family.downcast_ref()
    }

    /// The server's request queue: which sessions have a request in flight, and who waits.
    #[must_use]
    pub fn queue(&self) -> &RequestQueue {
        &self.queue
    }

    /// Opens `session` (a session object the kernel created on this server) with `root` as the
    /// interface behind it.
    pub fn open(&mut self, session: ObjectId, root: Box<dyn Session>) -> Result<(), ServerError> {
        if self.sessions.contains_key(&session) {
            return Err(ServerError::SessionExists(session));
        }
        let target = self.targets.insert_interface(root)?;
        self.sessions.insert(session, target);
        Ok(())
    }

    /// Opens `session`, which a client made by connecting to the service (the `sm:` port, or
    /// [`Kernel::connect`]), with a new root interface ([`Server::with_root`]).
    pub fn accept(&mut self, session: ObjectId) -> Result<(), ServerError> {
        let root = self.root.as_ref().ok_or(ServerError::NoRoot)?;
        let root = root();
        self.open(session, root)
    }

    /// Closes `session`. Interfaces no other session or domain still leads to are dropped, and
    /// so is the session's place in the queue: a request in flight on it is never answered, and
    /// a reply token for it completes into nothing. What the session's waiting senders are told
    /// is the kernel's business.
    pub fn close(&mut self, session: ObjectId) -> Result<(), ServerError> {
        let target = self.sessions.remove(&session);
        let target = target.ok_or(ServerError::UnknownSession(session))?;
        self.queue.close(session);
        self.pending.retain(|_, request| request.session != session);
        Ok(self.targets.release(target)?)
    }

    /// A guest thread sent a request on `session`. [`Admission::Start`] means the kernel gives
    /// the message to [`Server::handle`] now; [`Admission::Queued`] means the sender waits as a
    /// guest thread until [`Server::start_next`] names its session.
    pub fn submit(&mut self, session: ObjectId) -> Result<Admission, ServerError> {
        if !self.sessions.contains_key(&session) {
            return Err(ServerError::UnknownSession(session));
        }
        Ok(self.queue.submit(session)?)
    }

    /// A sender queued on `session` gave up before its request started.
    pub fn withdraw(&mut self, session: ObjectId) -> Result<(), ServerError> {
        Ok(self.queue.withdraw(session)?)
    }

    /// The session whose oldest waiting sender's request starts now, if any may: the kernel
    /// gives that sender's message to [`Server::handle`]. Call it after every
    /// [`Server::handle`], [`Server::deliver`] and [`Server::close`], until it returns `None`.
    pub fn start_next(&mut self) -> Option<ObjectId> {
        self.queue.start_next()
    }

    /// Handles the message a guest thread sent on `session`.
    ///
    /// `buffer` is the thread's message buffer, at least [`BUFFER_SIZE`] bytes, of which the first
    /// [`BUFFER_SIZE`] are used: the request is read from it and, for [`Handled::Replied`], the
    /// reply is written over it. On an error, and for the other two outcomes, the buffer is
    /// untouched.
    ///
    /// The request is one [`Server::submit`] or [`Server::start_next`] said to start. A request
    /// given here without either starts if it may (nothing in flight on its session, no sender
    /// waiting on it, no other session ready, a free worker) and is [`ServerError::Busy`]
    /// otherwise.
    ///
    /// The request has started once its session is found open and the queue lets it start. The
    /// errors from before that leave the queue as it was: [`ServerError::UnknownSession`],
    /// [`ServerError::Busy`], and [`ServerError::Queue`] holding
    /// [`QueueError::TooManyWaiting`]. A request that started is over when this returns,
    /// unless it was deferred: after an error (a buffer that is too short or does not parse, a
    /// reply that cannot be written, the server's own bookkeeping) its worker is free again,
    /// as it is after a reply.
    pub fn handle(
        &mut self,
        kernel: &mut dyn Kernel,
        session: ObjectId,
        buffer: &mut [u8],
    ) -> Result<Handled, ServerError> {
        // A session that is not open has no place in the queue, so there is no worker to free.
        let target = *self
            .sessions
            .get(&session)
            .ok_or(ServerError::UnknownSession(session))?;
        self.admit(session)?;
        let handled = self.run(kernel, session, target, buffer);
        // The request is over or deferred either way: its worker is free again.
        let settled = match handled {
            Ok(Handled::Deferred(_)) => self.queue.defer(session),
            // Closing the session took it out of the queue.
            Ok(Handled::Closed) => Ok(()),
            Ok(Handled::Replied) | Err(_) => self.queue.finish(session),
        };
        let handled = handled?;
        settled?;
        Ok(handled)
    }

    /// Marks the session's request as running, unless it already is.
    fn admit(&mut self, session: ObjectId) -> Result<(), ServerError> {
        if self.queue.is_running(session) {
            return Ok(());
        }
        match self.queue.submit(session)? {
            Admission::Start => Ok(()),
            Admission::Queued => {
                self.queue.withdraw(session)?;
                Err(ServerError::Busy(session))
            }
        }
    }

    /// Parses the message, answers it, and writes the reply and its trace event. The request
    /// has started: every error from here is one [`Server::handle`] frees the worker after.
    fn run(
        &mut self,
        kernel: &mut dyn Kernel,
        session: ObjectId,
        target: TargetId,
        buffer: &mut [u8],
    ) -> Result<Handled, ServerError> {
        if buffer.len() < BUFFER_SIZE {
            return Err(ServerError::BufferTooShort { len: buffer.len() });
        }
        let kind = if self.targets.is_domain(target)? {
            SessionKind::Domain
        } else {
            SessionKind::Session
        };
        let message = Message::parse_request(buffer, kind).map_err(ServerError::Malformed)?;
        if message.command_type.is_close() {
            self.close(session)?;
            return Ok(Handled::Closed);
        }
        let frame = Frame::of(&message, target);
        let asked = Asked::of(&message);
        let answered = match self.answer(kernel, session, target, frame, &message) {
            Ok(answered) => answered,
            Err(Refusal::Result(code)) => Answered::Now(frame.only(code)),
            Err(Refusal::Server(error)) => return Err(error),
        };
        match answered {
            Answered::Now(answer) => {
                self.send(kernel, session, asked, &answer, false, buffer)?;
                Ok(Handled::Replied)
            }
            Answered::Later(token) => {
                let request = Pending {
                    session,
                    framing: frame.framing(),
                    command: asked.command,
                    request_size: asked.request_size,
                };
                self.pending.insert(token, request);
                Ok(Handled::Deferred(token))
            }
        }
    }

    /// Writes a reply into the caller's buffer and records the request's one trace event.
    fn send(
        &self,
        kernel: &mut dyn Kernel,
        session: ObjectId,
        asked: Asked,
        answer: &Answer,
        deferred: bool,
        buffer: &mut [u8],
    ) -> Result<(), ServerError> {
        answer.builder.build(buffer).map_err(ServerError::Reply)?;
        let outcome = match (deferred, answer.stubbed) {
            (true, _) => IpcOutcome::Deferred,
            (false, true) => IpcOutcome::Stubbed,
            (false, false) => IpcOutcome::Done,
        };
        let reply_size = trace::size(answer.payload);
        let event = trace::event(
            self.service,
            session,
            asked,
            reply_size,
            answer.result,
            outcome,
        );
        kernel.record(event);
        Ok(())
    }

    /// The oldest deferred request whose token has completed and whose reply is not written
    /// yet. The kernel finds the thread that waits on [`Completed::session`] and gives its
    /// message buffer to [`Server::deliver`] at once.
    ///
    /// A completion for a request that is no longer pending (its session was closed, or its
    /// method answered at once and left its token behind) is logged and skipped.
    pub fn take_completed(&mut self) -> Option<Completed> {
        // Bounded by the mailbox: every turn takes one completion out of it.
        loop {
            let (token, posted) = self.mailbox.take()?;
            if let Some(pending) = self.pending.remove(&token) {
                return Some(Completed {
                    token,
                    pending,
                    posted,
                });
            }
            tracing::warn!(
                service = %self.name,
                token = token.raw(),
                "a reply token completed for a request that is not pending: ignored"
            );
        }
    }

    /// Writes the reply of a deferred request into its caller's message buffer, and records the
    /// request's trace event. `kernel` speaks for the thread that sent the request, and
    /// `buffer` is that thread's message buffer, as for [`Server::handle`]; it need not still
    /// hold the request.
    ///
    /// The request is no longer in flight afterwards, on an error too: a reply that could not
    /// be written is lost, and what the caller sees is the kernel's to decide.
    pub fn deliver(
        &mut self,
        kernel: &mut dyn Kernel,
        completed: Completed,
        buffer: &mut [u8],
    ) -> Result<(), ServerError> {
        let session = completed.pending.session;
        let delivered = self.write_deferred(kernel, completed, buffer);
        let settled = self.queue.complete(session);
        delivered?;
        Ok(settled?)
    }

    fn write_deferred(
        &mut self,
        kernel: &mut dyn Kernel,
        completed: Completed,
        buffer: &mut [u8],
    ) -> Result<(), ServerError> {
        let Completed {
            pending, posted, ..
        } = completed;
        let session = pending.session;
        if buffer.len() < BUFFER_SIZE {
            return Err(ServerError::BufferTooShort { len: buffer.len() });
        }
        let target = *self
            .sessions
            .get(&session)
            .ok_or(ServerError::UnknownSession(session))?;
        let frame = Frame::of_deferred(pending.framing, target);
        let answer = match posted {
            Posted::Reply(reply) => match self.build_reply(kernel, frame, reply) {
                Ok(answer) => answer,
                Err(Refusal::Result(code)) => frame.only(code),
                Err(Refusal::Server(error)) => return Err(error),
            },
            Posted::Failed(code) => frame.only(code),
            Posted::Dropped => frame.only(DEFERRED_REPLY_DROPPED),
        };
        let asked = Asked {
            command: pending.command,
            request_size: pending.request_size,
        };
        self.send(kernel, session, asked, &answer, true, buffer)
    }

    /// The deferred requests whose completion [`Server::take_completed`] has not taken yet, by
    /// token id, with the session each was sent on.
    pub fn pending(&self) -> impl Iterator<Item = (TokenId, ObjectId)> {
        (self.pending.iter()).map(|(token, request)| (*token, request.session))
    }

    /// A new reply token for a deferred request that is still pending: how the work behind it is
    /// resumed after a [`Server::restore`], which leaves the old tokens leading nowhere. `None`
    /// if no such request is pending.
    ///
    /// Reissuing a token whose first holder is still at work gives the request two tokens; the
    /// first to complete answers it, and the other is ignored and logged.
    pub fn reissue(&self, token: TokenId) -> Option<ReplyToken> {
        let pending = self.pending.contains_key(&token);
        pending.then(|| ReplyToken::new(token, Arc::clone(&self.mailbox)))
    }

    /// Routes a message by its type: control, domain, plain or TIPC.
    fn answer(
        &mut self,
        kernel: &mut dyn Kernel,
        session: ObjectId,
        target: TargetId,
        frame: Frame,
        message: &Message<'_>,
    ) -> Result<Answered, Refusal> {
        let (protocol, command_id, payload, interface, in_objects) =
            match (&message.body, message.command_type) {
                (
                    Body::CmifRequest(body),
                    CommandType::Control | CommandType::ControlWithContext,
                ) => {
                    let answer = self.control(kernel, session, target, body)?;
                    return Ok(Answered::Now(answer));
                }
                (Body::DomainClose(header), _) => {
                    self.close_object(target, header.object_id)?;
                    return Ok(Answered::Now(frame.only(ResultCode::SUCCESS)));
                }
                (
                    Body::CmifRequest(body),
                    CommandType::Request | CommandType::RequestWithContext,
                ) => {
                    let (interface, in_objects) = self.addressee(target, body)?;
                    let id = body.header.command_id;
                    (Protocol::Cmif, id, body.data, interface, in_objects)
                }
                (Body::TipcRequest { data }, CommandType::Tipc(id))
                    if !self.targets.is_domain(target)? =>
                {
                    (Protocol::Tipc, u32::from(id), *data, target, Vec::new())
                }
                _ => return Err(NOT_SUPPORTED.into()),
            };
        let call = Call {
            frame,
            protocol,
            command_id,
            payload,
            message,
            in_objects,
        };
        self.invoke(kernel, interface, call)
    }

    /// The interface a CMIF request is for, and its input objects: the session's own interface,
    /// or in a domain the object the domain header names.
    fn addressee(
        &self,
        target: TargetId,
        body: &CmifRequest<'_>,
    ) -> Result<(TargetId, Vec<Option<TargetId>>), Refusal> {
        let Some(header) = body.domain else {
            return Ok((target, Vec::new()));
        };
        let domain = self.targets.domain(target)?;
        let interface = domain.get(header.object_id).ok_or(TARGET_NOT_FOUND)?;
        let in_objects = body.object_ids.iter().map(|id| domain.get(*id)).collect();
        Ok((interface, in_objects))
    }

    /// Closes a domain object's id. The caller's reply is success whether or not the id named an
    /// object; see the [module docs](super) for how that differs from Eden's
    /// `HandleDomainSyncRequest` (`CloseVirtualHandle`).
    fn close_object(&mut self, domain: TargetId, object_id: u32) -> Result<(), Refusal> {
        let Some(object) = self.targets.domain_mut(domain)?.remove(object_id) else {
            tracing::warn!(service = %self.name, object_id, "close of an empty domain object id");
            return Ok(());
        };
        Ok(self.targets.release(object)?)
    }

    /// Calls the interface's method and turns what it returns into a reply.
    fn invoke(
        &mut self,
        kernel: &mut dyn Kernel,
        interface: TargetId,
        call: Call<'_>,
    ) -> Result<Answered, Refusal> {
        let Call {
            frame,
            protocol,
            command_id,
            ..
        } = call;
        let request = Request::new(
            &*kernel,
            protocol,
            command_id,
            call.payload,
            call.message,
            call.in_objects,
        );
        let mut reply = Reply::default();
        let mut object = self.targets.check_out(interface)?;
        let mut cx = Context {
            kernel: &mut *kernel,
            targets: &mut self.targets,
            family: self.family.as_deref_mut(),
            policy: self.policy,
            deferral: Deferral {
                mailbox: &self.mailbox,
                next_token: &mut self.next_token,
                taken: None,
            },
            env: self.env.as_ref(),
            executor: self.executor.as_ref(),
        };
        let outcome = object.dispatch(protocol, command_id, &mut cx, &request, &mut reply);
        let token = cx.deferral.taken;
        let name = object.name();
        let command_name = object.commands().name(protocol, command_id);
        self.targets.check_in(interface, object)?;
        match (outcome, token) {
            (Some(Ok(Completion::Deferred)), Some(token)) => Ok(Answered::Later(token)),
            (Some(Ok(Completion::Deferred)), None) => {
                tracing::error!(
                    service = %self.name,
                    interface = name,
                    ?protocol,
                    command = command_id,
                    "a method deferred its reply without taking a reply token"
                );
                Err(DEFERRED_REPLY_DROPPED.into())
            }
            (Some(Ok(Completion::Done)), _) => {
                Ok(Answered::Now(self.build_reply(kernel, frame, reply)?))
            }
            (Some(Err(fault)), _) => {
                tracing::warn!(
                    service = %self.name,
                    interface = name,
                    ?protocol,
                    command = command_id,
                    command_name = command_name.unwrap_or_default(),
                    %fault,
                    "command failed"
                );
                Err(fault.into())
            }
            (None, _) => Ok(Answered::Now(
                self.unknown(name, protocol, command_id, frame),
            )),
        }
    }

    /// The reply to a command the interface does not list.
    pub(super) fn unknown(
        &mut self,
        interface: &'static str,
        protocol: Protocol,
        id: u32,
        frame: Frame,
    ) -> Answer {
        let command = StubbedCommand {
            interface,
            protocol,
            id,
        };
        match self.policy {
            UnknownPolicy::Strict => {
                tracing::warn!(service = %self.name, ?command, "unknown command");
                frame.only(UNKNOWN_COMMAND_ID)
            }
            UnknownPolicy::AutoStub => {
                self.record_stub(command);
                // Only the result: every byte after it in the caller's buffer is zero, because
                // building a reply zeroes the buffer first. Nothing of the request is echoed
                // (`eden-nx2/NX2_NOTES.md`, row 96).
                Answer {
                    stubbed: true,
                    ..frame.only(ResultCode::SUCCESS)
                }
            }
        }
    }

    /// Records and logs an auto-stubbed command the first time it is seen.
    fn record_stub(&mut self, command: StubbedCommand) {
        if self.stubbed.contains(&command) {
            return;
        }
        if self.stubbed.len() >= MAX_STUBBED_COMMANDS {
            if !self.stub_record_full {
                self.stub_record_full = true;
                tracing::warn!(
                    service = %self.name,
                    limit = MAX_STUBBED_COMMANDS,
                    "auto-stub record is full: further unknown commands are stubbed unrecorded"
                );
            }
            return;
        }
        tracing::warn!(
            service = %self.name,
            ?command,
            "unknown command auto-stubbed: replying success with a zeroed payload"
        );
        self.stubbed.insert(command);
    }

    /// Builds a successful reply: writes its bytes into the caller's out buffers
    /// ([`Reply::write_buffer`]), adds its handles to the caller's table and opens its new
    /// interfaces, as sessions or as domain objects.
    ///
    /// Nothing is changed unless the whole reply fits the buffer, the domain has room, and the
    /// reply is one the IPC crate can frame (see "Not here yet" in the [module docs](super)).
    /// The buffers are written first, in the order the method added them; a write the kernel
    /// refuses makes the reply its error, with no handle added and no interface opened.
    /// Handles are added in Eden's order, copies and then moves
    /// (`eden/src/core/hle/service/hle_ipc.cpp`, `WriteToOutgoingCommandBuffer`). If the caller's
    /// handle table fills up part-way, the reply is the kernel's error and the handles and
    /// sessions already added stay, as in Eden (`R_TRY(handle_table.Add(...))` there returns
    /// part-way too); new domain objects are only added once every handle is in.
    fn build_reply(
        &mut self,
        kernel: &mut dyn Kernel,
        frame: Frame,
        mut reply: Reply,
    ) -> Result<Answer, Refusal> {
        let writes = reply.take_writes();
        let (payload, copies, moves) = reply.into_parts();
        if !fits(frame, &payload, copies.len(), &moves) {
            tracing::error!(service = %self.name, "reply does not fit a message buffer");
            return Err(NOT_SUPPORTED.into());
        }
        let domain = match frame {
            Frame::Domain(domain) => Some(domain),
            Frame::Cmif | Frame::Tipc(_) => None,
        };
        let sessions = moves.iter().filter(|m| matches!(m, Moved::Session(_)));
        let sessions = sessions.count();
        if let Some(domain) = domain
            && sessions > 0
        {
            if !payload.len().is_multiple_of(size_of::<u32>()) {
                tracing::error!(
                    service = %self.name,
                    payload = payload.len(),
                    "a domain reply cannot yet carry objects after a payload that is not whole words"
                );
                return Err(NOT_SUPPORTED.into());
            }
            if self.targets.domain(domain)?.room() < sessions {
                return Err(OUT_OF_DOMAIN_ENTRIES.into());
            }
        }

        for (buffer, data) in writes {
            if !data.is_empty() {
                kernel.write_memory(buffer.address, &data)?;
            }
        }
        let mut builder = frame.reply(ResultCode::SUCCESS).data(payload.as_bytes());
        for object in copies {
            builder = builder.copy_handle(kernel.insert_handle(object)?.raw());
        }
        let mut objects = Vec::new();
        for moved in moves {
            let handle = match moved {
                Moved::Object(object) => kernel.insert_handle(object)?,
                Moved::Session(session) if domain.is_some() => {
                    objects.push(session);
                    continue;
                }
                Moved::Session(session) => self.open_new(kernel, session)?,
                Moved::Service(service) => kernel.connect(service)?.1,
            };
            builder = builder.move_handle(handle.raw());
        }
        if let Some(domain) = domain {
            for session in objects {
                builder = builder.object_id(self.add_object(domain, session)?);
            }
        }
        Ok(Answer {
            builder,
            result: ResultCode::SUCCESS,
            payload: payload.len(),
            stubbed: false,
        })
    }

    /// Opens a new kernel session on this server that leads to `target`, and returns the caller's
    /// handle to it. The session takes a reference of its own.
    pub(super) fn open_shared(
        &mut self,
        kernel: &mut dyn Kernel,
        target: TargetId,
    ) -> Result<Handle, Refusal> {
        let (session, handle) = kernel.create_session()?;
        if self.sessions.contains_key(&session) {
            return Err(ServerError::SessionExists(session).into());
        }
        self.targets.retain(target)?;
        self.sessions.insert(session, target);
        Ok(handle)
    }

    /// Opens a new kernel session on this server for a new interface, and returns the caller's
    /// handle to it.
    fn open_new(
        &mut self,
        kernel: &mut dyn Kernel,
        interface: Box<dyn Session>,
    ) -> Result<Handle, Refusal> {
        let (session, handle) = kernel.create_session()?;
        self.open(session, interface)?;
        Ok(handle)
    }

    /// Adds a new interface to a domain and returns its object id.
    fn add_object(
        &mut self,
        domain: TargetId,
        interface: Box<dyn Session>,
    ) -> Result<u32, Refusal> {
        let target = self.targets.insert_interface(interface)?;
        if let Some(object_id) = self.targets.domain_mut(domain)?.add(target) {
            return Ok(object_id);
        }
        self.targets.release(target)?;
        Err(OUT_OF_DOMAIN_ENTRIES.into())
    }

    /// Makes `session` a domain whose object [`Domain::ROOT_OBJECT_ID`] is the interface the
    /// session led to, and returns that id.
    pub(super) fn convert_to_domain(
        &mut self,
        session: ObjectId,
        target: TargetId,
    ) -> Result<u32, Refusal> {
        if self.targets.is_domain(target)? {
            return Err(NOT_SUPPORTED.into());
        }
        // The session's reference to its interface becomes the domain's.
        let domain = self.targets.insert_domain(Domain::with_root(target))?;
        self.sessions.insert(session, domain);
        Ok(Domain::ROOT_OBJECT_ID)
    }
}

/// Whether a successful reply with these parts fits a message buffer. Built with placeholder
/// handles and ids, which take the same room as real ones.
fn fits(frame: Frame, payload: &Writer, copies: usize, moves: &[Moved]) -> bool {
    if payload.overflowed() {
        return false;
    }
    let mut probe = frame.reply(ResultCode::SUCCESS).data(payload.as_bytes());
    for _ in 0..copies {
        probe = probe.copy_handle(0);
    }
    for moved in moves {
        probe = match (moved, frame) {
            (Moved::Session(_), Frame::Domain(_)) => probe.object_id(0),
            _ => probe.move_handle(0),
        };
    }
    probe.build(&mut [0; BUFFER_SIZE]).is_ok()
}
