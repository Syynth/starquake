//! [`Server`]: one service family's sessions, and the path a request takes through them.

use std::collections::{BTreeMap, BTreeSet};

use starquake_ipc::{
    BUFFER_SIZE, Body, CmifRequest, CommandType, IpcError, Message, ReplyBuilder, SessionKind,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::{KernelError, ResultCode};

use super::commands::{Protocol, Session};
use super::context::Context;
use super::kernel::Kernel;
use super::payload::Writer;
use super::reply::{Moved, Reply};
use super::request::Request;
use super::result::{
    Completion, Fault, NOT_SUPPORTED, OUT_OF_DOMAIN_ENTRIES, TARGET_NOT_FOUND, UNKNOWN_COMMAND_ID,
};
use super::targets::{Domain, TargetError, TargetId, Targets};

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

    /// A reply holding only a result.
    pub(super) fn reply(self, result: ResultCode) -> ReplyBuilder {
        match self {
            Self::Cmif => ReplyBuilder::cmif(result.raw()),
            Self::Domain(_) => ReplyBuilder::domain(result.raw()),
            Self::Tipc(command_id) => ReplyBuilder::tipc(command_id, result.raw()),
        }
    }
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

/// One service family's sessions and the interfaces behind them: the Rust side of a server
/// (`docs/specs/services.md`, "Concurrency").
///
/// The kernel's session code owns the queueing: it calls [`Server::handle`] for one request at a
/// time, so nothing here locks and a service method takes `&mut self`.
pub struct Server {
    pub(super) name: String,
    pub(super) policy: UnknownPolicy,
    pub(super) pointer_buffer_size: u16,
    pub(super) sessions: BTreeMap<ObjectId, TargetId>,
    pub(super) targets: Targets,
    stubbed: BTreeSet<StubbedCommand>,
    stub_record_full: bool,
}

impl Server {
    /// A server with no sessions. `name` is the service's name in logs (`"set:sys"`).
    #[must_use]
    pub fn new(name: impl Into<String>, policy: UnknownPolicy) -> Self {
        Self {
            name: name.into(),
            policy,
            pointer_buffer_size: DEFAULT_POINTER_BUFFER_SIZE,
            sessions: BTreeMap::new(),
            targets: Targets::default(),
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

    /// The service's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
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

    /// Closes `session`. Interfaces no other session or domain still leads to are dropped.
    pub fn close(&mut self, session: ObjectId) -> Result<(), ServerError> {
        let target = self.sessions.remove(&session);
        let target = target.ok_or(ServerError::UnknownSession(session))?;
        Ok(self.targets.release(target)?)
    }

    /// Handles the message a guest thread sent on `session`.
    ///
    /// `buffer` is the thread's message buffer, at least [`BUFFER_SIZE`] bytes, of which the first
    /// [`BUFFER_SIZE`] are used: the request is read from it and, for [`Handled::Replied`], the
    /// reply is written over it. On an error the buffer is untouched.
    pub fn handle(
        &mut self,
        kernel: &mut dyn Kernel,
        session: ObjectId,
        buffer: &mut [u8],
    ) -> Result<Handled, ServerError> {
        if buffer.len() < BUFFER_SIZE {
            return Err(ServerError::BufferTooShort { len: buffer.len() });
        }
        let target = *self
            .sessions
            .get(&session)
            .ok_or(ServerError::UnknownSession(session))?;
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
        let reply = match self.answer(kernel, session, target, frame, &message) {
            Ok(reply) => reply,
            Err(Refusal::Result(code)) => frame.reply(code),
            Err(Refusal::Server(error)) => return Err(error),
        };
        reply.build(buffer).map_err(ServerError::Reply)?;
        Ok(Handled::Replied)
    }

    /// Routes a message by its type: control, domain, plain or TIPC.
    fn answer(
        &mut self,
        kernel: &mut dyn Kernel,
        session: ObjectId,
        target: TargetId,
        frame: Frame,
        message: &Message<'_>,
    ) -> Result<ReplyBuilder, Refusal> {
        let (protocol, command_id, payload, interface, in_objects) =
            match (&message.body, message.command_type) {
                (
                    Body::CmifRequest(body),
                    CommandType::Control | CommandType::ControlWithContext,
                ) => {
                    return self.control(kernel, session, target, body);
                }
                (Body::DomainClose(header), _) => {
                    self.close_object(target, header.object_id)?;
                    return Ok(frame.reply(ResultCode::SUCCESS));
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
    ) -> Result<ReplyBuilder, Refusal> {
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
        };
        let outcome = object.dispatch(protocol, command_id, &mut cx, &request, &mut reply);
        let name = object.name();
        let command_name = object.commands().name(protocol, command_id);
        self.targets.check_in(interface, object)?;
        match outcome {
            Some(Ok(Completion::Done)) => self.finish(kernel, frame, reply),
            Some(Err(fault)) => {
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
            None => Ok(self.unknown(name, protocol, command_id, frame)),
        }
    }

    /// The reply to a command the interface does not list.
    pub(super) fn unknown(
        &mut self,
        interface: &'static str,
        protocol: Protocol,
        id: u32,
        frame: Frame,
    ) -> ReplyBuilder {
        let command = StubbedCommand {
            interface,
            protocol,
            id,
        };
        match self.policy {
            UnknownPolicy::Strict => {
                tracing::warn!(service = %self.name, ?command, "unknown command");
                frame.reply(UNKNOWN_COMMAND_ID)
            }
            UnknownPolicy::AutoStub => {
                self.record_stub(command);
                // Only the result: every byte after it in the caller's buffer is zero, because
                // building a reply zeroes the buffer first. Nothing of the request is echoed
                // (`eden-nx2/NX2_NOTES.md`, row 96).
                frame.reply(ResultCode::SUCCESS)
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

    /// Builds a successful reply: adds its handles to the caller's table and opens its new
    /// interfaces, as sessions or as domain objects.
    ///
    /// Nothing is changed unless the whole reply fits the buffer, the domain has room, and the
    /// reply is one the IPC crate can frame (see "Not here yet" in the [module docs](super)).
    /// Handles are added in Eden's order, copies and then moves
    /// (`eden/src/core/hle/service/hle_ipc.cpp`, `WriteToOutgoingCommandBuffer`). If the caller's
    /// handle table fills up part-way, the reply is the kernel's error and the handles and
    /// sessions already added stay, as in Eden (`R_TRY(handle_table.Add(...))` there returns
    /// part-way too); new domain objects are only added once every handle is in.
    fn finish(
        &mut self,
        kernel: &mut dyn Kernel,
        frame: Frame,
        reply: Reply,
    ) -> Result<ReplyBuilder, Refusal> {
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
            };
            builder = builder.move_handle(handle.raw());
        }
        if let Some(domain) = domain {
            for session in objects {
                builder = builder.object_id(self.add_object(domain, session)?);
            }
        }
        Ok(builder)
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
