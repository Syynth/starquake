//! The dispatcher's IPC arms, `ConnectToNamedPort` (0x1f) and `SendSyncRequest` (0x21), the
//! handlers it keeps outside the tables, and [`IpcContext`], what a handler reaches the kernel
//! through.
//!
//! Kernel spec, "IPC". `SendSyncRequest` takes the kernel lock three times:
//!
//! 1. To look the session up, copy the message out of the caller's TLS block, translate its
//!    special header ([`crate::ipc::message`]) and queue the request ([`Ipc::send`]). A request
//!    that cannot start leaves its sender waiting ([`Outcome::Waiting`]).
//! 2. Briefly, to take the server's handler out of its slot. The handler then runs **without
//!    the lock**; the calls it makes through [`IpcContext`] take it one at a time. One worker per
//!    server means the slot is never empty when the queue starts a request.
//! 3. To put the handler back, end the request ([`Ipc::finish`]), and start the server's next
//!    ready request, if there is one, on this core: steps 2 and 3 repeat for it. Then every
//!    request that is over is settled: its reply is written into its sender's TLS block, its
//!    references are released, and its events are recorded, the IPC events the handler made
//!    and then the system call's own, in the order the requests ended. The caller's own result
//!    goes into its registers; every other sender that was waiting is woken
//!    ([`Dispatcher::take_woken`]).
//!
//! `ConnectToNamedPort` is ported from Eden `eden/src/core/hle/kernel/svc/svc_port.cpp` @
//! cf654052 (GPL-3.0-or-later), `ConnectToNamedPort`, with the name read as
//! [`PortName::read`] says: a port nothing is registered under is [`KernelError::NotFound`],
//! and a full handle table [`KernelError::OutOfHandles`], in that order. `SendSyncRequest` is
//! `SendSyncRequestImpl` in `eden/src/core/hle/kernel/svc/svc_ipc.cpp` @ cf654052: a handle that
//! names no client session is [`KernelError::InvalidHandle`].
//!
//! Where starquake is stricter than Eden, pinned by the tests named (in `dispatch/ipc/tests.rs`):
//!
//! - A handler's own failure ([`HandlerError`](crate::ipc::HandlerError)) closes the server end of the session, and its
//!   senders see [`KernelError::SessionClosed`], as a console's server framework closes a
//!   session it fails to process (see [`HandlerError`](crate::ipc::HandlerError)). Eden's `ServerManager` logs the
//!   failure and closes the session too. Test: `a_handler_that_fails_closes_the_session`.
//! - A server's handler can reply later ([`Handled::Deferred`]); Eden's HLE servers reply on
//!   their own host threads instead. Test: `a_second_sender_waits_behind_a_deferred_request`.
//!
//! PROVISIONAL (walking skeleton, #74): the dispatcher is told its caller (`Caller`), there is
//! one process, and a woken sender is reported, not made runnable: the scheduler does not know
//! which guest thread is which yet. Kernel slice 10 (the core loop) makes waking real.

use starquake_cpu::GuestRegs;
use starquake_ipc::BUFFER_SIZE;
use starquake_memory::GuestMemory;
use starquake_trace::{
    EnvError, EventKind, GuestTicks, Recorder, SVC_ARG_COUNT, ServiceId, ThreadId,
};

use super::{
    Call, Caller, Dispatcher, Outcome, complete, fault, mismatch, record_point, recorded_args,
    result_word,
};
use crate::handle::{Handle, HandleError, HandleTable};
use crate::id::{ObjectId, ServerId};
use crate::ipc::message::{MessageBuffer, Translated, translate};
use crate::ipc::{
    Admission, Done, Finish, Handled, IpcError, Notice, PortName, Request, Sender, SessionHandler,
};
use crate::memory::AddressSpace;
use crate::object::Object;
use crate::result::{CallResult, KernelError, ResultCode, SvcError, SvcResult};
use crate::svc::{Convention, SvcId};
use crate::tables::Scheduler;

#[cfg(test)]
mod tests;

/// A guest thread whose `SendSyncRequest` ended while it waited, and its result. The reply, when
/// the result is success, is already in its message buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Woken {
    /// The thread, as the event stream names it.
    pub thread: ThreadId,
    /// The core it called from.
    pub core: u8,
    /// What its `SendSyncRequest` returns.
    pub result: ResultCode,
}

/// A request that is over, with the events its handler recorded, waiting to be settled.
struct Finished {
    session: ObjectId,
    done: Done,
    events: Vec<EventKind>,
}

/// The kernel error a guest sees for a handle-table failure, logging the ones that are the
/// kernel's own bookkeeping.
fn handle_error(error: &HandleError) -> KernelError {
    error.kernel_error().unwrap_or_else(|| {
        tracing::error!(%error, "handle table bookkeeping failed during a request");
        KernelError::InvalidState
    })
}

/// The kernel error a handler sees for an IPC table failure, logging the ones that are the
/// kernel's own bookkeeping.
fn ipc_error(error: IpcError) -> KernelError {
    match error {
        IpcError::Kernel(error) => error,
        error => {
            tracing::error!(%error, "IPC bookkeeping failed during a request");
            KernelError::InvalidState
        }
    }
}

/// What a server's handler reaches the kernel through while it handles one request: the
/// request's sender's process (its handle table and memory), the server it runs on, and the
/// event stream. Each call takes the kernel lock for itself; none is held between them.
pub struct IpcContext<'a> {
    dispatcher: &'a Dispatcher,
    memory: &'a dyn GuestMemory,
    server: ServerId,
    process_id: u64,
    translated: Translated,
    events: Vec<EventKind>,
}

impl IpcContext<'_> {
    /// The sender's process id as the guest knows it.
    #[must_use]
    pub fn process_id(&self) -> u64 {
        self.process_id
    }

    /// The object a handle names: one of the request's own handles as it was when the request
    /// was sent (a moved one is no longer in the sender's table), or else whatever the handle
    /// names in the sender's table now.
    pub fn resolve_handle(&self, handle: Handle) -> SvcResult<ObjectId> {
        if let Some(object) = self.translated.object(handle) {
            return Ok(object);
        }
        self.dispatcher
            .with::<(&HandleTable,), _>(|(handles,)| handles.get(handle))
            .map_err(|error| handle_error(&error))
    }

    /// Adds a handle to `object` to the sender's table.
    pub fn insert_handle(&mut self, object: ObjectId) -> SvcResult<Handle> {
        self.dispatcher
            .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
                handles.add(scheduler.objects_mut(), object)
            })
            .map_err(|error| handle_error(&error))
    }

    /// Creates an event. Events are kernel slice 4, which has not landed: always
    /// [`KernelError::NotImplemented`].
    pub fn create_event(&mut self) -> SvcResult<ObjectId> {
        tracing::warn!("a service asked for an event, which the kernel does not have yet");
        Err(KernelError::NotImplemented)
    }

    /// Signals an event. As [`IpcContext::create_event`]: always
    /// [`KernelError::NotImplemented`].
    pub fn signal_event(&mut self, _event: ObjectId) -> SvcResult<()> {
        tracing::warn!("a service signalled an event, which the kernel does not have yet");
        Err(KernelError::NotImplemented)
    }

    /// Opens a session on the server running this request, and adds a handle to it to the
    /// sender's table. The server is not told: it records the session itself. All or nothing.
    pub fn create_session(&mut self) -> SvcResult<(ObjectId, Handle)> {
        let server = self.server;
        self.open_session(server, false)
    }

    /// Opens a session on the server known as `service`, and adds a handle to it to the
    /// sender's table; the server is told to accept it before it is given a request on it.
    /// [`KernelError::NotFound`] for a service no server is known as. All or nothing.
    pub fn connect(&mut self, service: ServiceId) -> SvcResult<(ObjectId, Handle)> {
        let server = self
            .dispatcher
            .with::<(&Scheduler,), _>(|(scheduler,)| scheduler.ipc().service(service));
        let server = server.ok_or(KernelError::NotFound)?;
        self.open_session(server, true)
    }

    fn open_session(&mut self, server: ServerId, notice: bool) -> SvcResult<(ObjectId, Handle)> {
        self.dispatcher
            .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
                open_session_with_handle(handles, scheduler, server, notice)
            })
    }

    /// Copies the sender's memory at `address` into `out`, if the guest may read all of it;
    /// [`KernelError::InvalidCurrentMemory`] otherwise, with nothing copied.
    pub fn read_memory(&self, address: u64, out: &mut [u8]) -> SvcResult<()> {
        let len = u64::try_from(out.len()).map_err(|_| KernelError::InvalidCurrentMemory)?;
        let memory = self.memory;
        self.dispatcher.with::<(&AddressSpace,), _>(|(space,)| {
            if !space.user_can_read(address, len) {
                return Err(KernelError::InvalidCurrentMemory);
            }
            memory
                .read(address, out)
                .map_err(|_| KernelError::InvalidCurrentMemory)
        })
    }

    /// Copies `data` into the sender's memory at `address`, if the guest may write all of it;
    /// [`KernelError::InvalidCurrentMemory`] otherwise, with nothing copied.
    pub fn write_memory(&mut self, address: u64, data: &[u8]) -> SvcResult<()> {
        let len = u64::try_from(data.len()).map_err(|_| KernelError::InvalidCurrentMemory)?;
        let memory = self.memory;
        self.dispatcher.with::<(&AddressSpace,), _>(|(space,)| {
            if !space.user_can_write(address, len) {
                return Err(KernelError::InvalidCurrentMemory);
            }
            memory
                .write(address, data)
                .map_err(|_| KernelError::InvalidCurrentMemory)
        })
    }

    /// Records an event for the request. It is recorded, stamped with the request's sender and
    /// the dispatch's time, when the request is settled, before the system call's own event.
    pub fn record(&mut self, kind: EventKind) {
        self.events.push(kind);
    }
}

/// Opens a session on `server` and gives the caller a handle to it: the session's object then
/// has one reference, the handle's. All or nothing: when the handle cannot be added, the
/// session is taken back.
fn open_session_with_handle(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
    server: ServerId,
    notice: bool,
) -> SvcResult<(ObjectId, Handle)> {
    let (objects, ipc) = scheduler.ipc_mut();
    let session = ipc
        .open_session(objects, server, notice)
        .map_err(ipc_error)?;
    match handles.add(objects, session) {
        Ok(handle) => {
            // The creator's reference; the handle's keeps the session.
            ipc.release(objects, session).map_err(ipc_error)?;
            Ok((session, handle))
        }
        Err(error) => {
            ipc.discard_session(objects, session).map_err(ipc_error)?;
            Err(handle_error(&error))
        }
    }
}

/// How the first step of `SendSyncRequest` ended.
enum Begun {
    /// The request is in flight: run its server's handler.
    Start { server: ServerId, session: ObjectId },
    /// The sender waits behind another request.
    Queued,
}

impl Dispatcher {
    /// Adds an HLE server known as `service` (the number another server's
    /// [`IpcContext::connect`] names it by), reachable by `ConnectToNamedPort` when it has a
    /// `port`, with `handler` behind it. `name` is for logs.
    pub fn add_server(
        &self,
        name: &str,
        service: ServiceId,
        port: Option<PortName>,
        handler: Box<dyn SessionHandler>,
    ) -> Result<ServerId, IpcError> {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        let (scheduler,) = locked.tables();
        let (_, ipc) = scheduler.ipc_mut();
        if let Some(port) = port
            && ipc.port(&port).is_some()
        {
            return Err(IpcError::PortTaken(port));
        }
        let server = ipc.add_server(name, service)?;
        if let Some(port) = port {
            ipc.name_port(port, server)?;
        }
        locked.state.handlers.insert(server, Some(handler));
        Ok(server)
    }

    /// The guest threads woken since the last call, oldest first: senders whose request ended
    /// while another thread's dispatch ran. Each one's result and reply are written; the caller
    /// resumes it.
    pub fn take_woken(&self) -> Vec<Woken> {
        let mut locked = self.lock::<()>();
        std::mem::take(&mut locked.state.woken)
    }

    /// Writes the deferred replies of `server` that are ready, wakes their senders, and starts
    /// the server's next requests. `caller` is the core doing it: its memory is the process's,
    /// and its clock reading stamps the events. The senders are reported by
    /// [`Dispatcher::take_woken`].
    pub fn deliver_replies(
        &self,
        server: ServerId,
        caller: &mut Caller<'_>,
    ) -> Result<(), EnvError> {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        let time = locked.guest_time(caller.env)?;
        let process_id = locked.state.tables.processes.process_id();
        // Bounded by the requests deferred now: each turn answers one of them.
        let deferred = locked
            .state
            .tables
            .scheduler
            .ipc()
            .queue(server)
            .map_or(0, |queue| queue.deferred().count());
        let Some(mut handler) = locked
            .state
            .handlers
            .get_mut(&server)
            .and_then(Option::take)
        else {
            tracing::warn!(
                ?server,
                "deferred replies wait for a server whose handler is busy"
            );
            return Ok(());
        };
        drop(locked);

        let mut finished = Vec::new();
        for _ in 0..deferred {
            let mut cx = self.context(caller.memory, server, process_id, Translated::default());
            let mut message = [0; BUFFER_SIZE];
            let Some((session, written)) = handler.take_reply(&mut cx, &mut message) else {
                break;
            };
            let mut locked = self.lock::<(&mut Scheduler,)>();
            let (scheduler,) = locked.tables();
            let (_, ipc) = scheduler.ipc_mut();
            match written {
                Ok(()) => match ipc.complete(session, Box::new(message)) {
                    Ok(done) => finished.push(Finished {
                        session,
                        done,
                        events: cx.events,
                    }),
                    Err(error) => {
                        tracing::error!(%error, ?session, "a deferred reply has no request");
                    }
                },
                Err(error) => {
                    tracing::error!(%error, ?server, ?session, "closing the session");
                    finished.extend(close_failed(ipc, server, session, cx.events));
                }
            }
        }
        self.put_back(server, handler);
        finished.extend(self.run_ready(server, caller.memory, process_id));
        let woken = self.settle(finished, caller.memory, caller.recorder, time);
        let mut locked = self.lock::<()>();
        locked.state.woken.extend(woken);
        drop(locked);
        self.settle_notices();
        Ok(())
    }

    /// An [`IpcContext`] for a request on `server`.
    fn context<'a>(
        &'a self,
        memory: &'a dyn GuestMemory,
        server: ServerId,
        process_id: u64,
        translated: Translated,
    ) -> IpcContext<'a> {
        IpcContext {
            dispatcher: self,
            memory,
            server,
            process_id,
            translated,
            events: Vec::new(),
        }
    }

    /// Puts a handler back in its server's slot.
    fn put_back(&self, server: ServerId, handler: Box<dyn SessionHandler>) {
        let mut locked = self.lock::<()>();
        locked.state.handlers.insert(server, Some(handler));
    }

    /// Gives every server whose handler is free the notices waiting for it, outside the lock.
    pub(super) fn settle_notices(&self) {
        let servers: Vec<ServerId> = {
            let locked = self.lock::<()>();
            let ipc = locked.state.tables.scheduler.ipc();
            ipc.servers()
                .filter(|server| ipc.has_notices(*server).unwrap_or(false))
                .collect()
        };
        for server in servers {
            let mut locked = self.lock::<(&mut Scheduler,)>();
            let Some(mut handler) = locked
                .state
                .handlers
                .get_mut(&server)
                .and_then(Option::take)
            else {
                // Running: it is given its notices before its next request.
                continue;
            };
            let (scheduler,) = locked.tables();
            let notices = scheduler
                .ipc_mut()
                .1
                .take_notices(server)
                .unwrap_or_default();
            drop(locked);
            apply_notices(&mut *handler, server, notices);
            self.put_back(server, handler);
        }
    }

    /// `ConnectToNamedPort`: the name, the port, a session on its server and a handle to it.
    pub(super) fn connect_to_named_port(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        let mut notices = false;
        let outcome = self.call::<(&mut HandleTable, &AddressSpace, &mut Scheduler)>(
            call,
            |(handles, space, scheduler), call, _| {
                let memory = call.caller.memory;
                let outcome = super::returning(id, convention, call.regs, |[name]| {
                    let name = PortName::read(space, memory, name)?;
                    let server = scheduler.ipc().port(&name).ok_or(KernelError::NotFound)?;
                    let (_, handle) = open_session_with_handle(handles, scheduler, server, true)?;
                    Ok([u64::from(handle.raw())])
                });
                notices = scheduler.ipc().has_any_notices();
                outcome
            },
        );
        // The server accepts the session now, if it is free, rather than at its first request.
        if notices {
            self.settle_notices();
        }
        outcome
    }

    /// `SendSyncRequest`: see the module docs.
    pub(super) fn send_sync_request(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        let mut locked = self.lock::<(&mut HandleTable, &mut Scheduler)>();
        let time = match locked.guest_time(call.caller.env) {
            Ok(time) => time,
            Err(error) => {
                tracing::error!(number = call.number, %error, "environment failed");
                return Outcome::EnvFailed(error);
            }
        };
        let [handle] = match convention.read::<1>(call.regs) {
            Ok(inputs) => inputs,
            Err(error) => {
                let outcome = mismatch(id, error, call.regs);
                record_point(call, time, &outcome);
                return outcome;
            }
        };
        let handle = Handle::from_register(handle);
        let process_id = locked.state.tables.processes.process_id();
        let sender = Sender {
            thread: call.caller.thread,
            core: call.caller.core,
            message_address: call.regs.tpidrro,
            session_handle: handle,
        };
        let (handles, scheduler) = locked.tables();
        let begun = begin(handles, scheduler, call.caller.memory, process_id, sender);
        let (server, session) = match begun {
            Ok(Begun::Start { server, session }) => (server, session),
            Ok(Begun::Queued) => return Outcome::Waiting,
            Err(error) => {
                let outcome = match error {
                    SvcError::Kernel(error) => answer(id, convention, call.regs, error.code()),
                    SvcError::Internal(what) => fault(id, &what, call.regs),
                };
                record_point(call, time, &outcome);
                return outcome;
            }
        };
        drop(locked);

        let mut finished = self.run_request(server, session, call.caller.memory, process_id);
        finished.extend(self.run_ready(server, call.caller.memory, process_id));
        let own = call.caller.thread;
        let mut woken = self.settle(finished, call.caller.memory, call.caller.recorder, time);
        let outcome = match woken.iter().position(|woken| woken.thread == own) {
            Some(index) => {
                let mine = woken.remove(index);
                answer(id, convention, call.regs, mine.result)
            }
            // Deferred: the caller waits for its reply.
            None => Outcome::Waiting,
        };
        let mut locked = self.lock::<()>();
        locked.state.woken.extend(woken);
        drop(locked);
        self.settle_notices();
        outcome
    }

    /// Runs the request in flight on `session`, a session of `server`, through the server's
    /// handler without the lock, and ends it. Returns the requests that are over.
    fn run_request(
        &self,
        server: ServerId,
        session: ObjectId,
        memory: &dyn GuestMemory,
        process_id: u64,
    ) -> Vec<Finished> {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        let (scheduler,) = locked.tables();
        let request = scheduler.ipc().in_flight(session).ok().flatten().cloned();
        let handler = match request {
            Some(_) => locked
                .state
                .handlers
                .get_mut(&server)
                .and_then(Option::take),
            None => None,
        };
        let (scheduler,) = locked.tables();
        let ipc = scheduler.ipc_mut().1;
        let (Some(mut handler), Some(request)) = (handler, request) else {
            // One worker per server keeps the handler in its slot whenever a request starts,
            // and a started request is in flight: the kernel's bookkeeping is broken.
            tracing::error!(
                ?server,
                ?session,
                "a started request has no handler to run it"
            );
            let failed = Finish::Failed(KernelError::NotImplemented);
            return finish(ipc, session, failed, Vec::new());
        };
        let notices = ipc.take_notices(server).unwrap_or_default();
        drop(locked);

        apply_notices(&mut *handler, server, notices);
        let mut cx = self.context(memory, server, process_id, request.translated);
        let mut message: MessageBuffer = *request.message;
        let answered = handler.handle(&mut cx, session, &mut message);

        let mut locked = self.lock::<(&mut Scheduler,)>();
        locked.state.handlers.insert(server, Some(handler));
        let (scheduler,) = locked.tables();
        let ipc = scheduler.ipc_mut().1;
        let ending = match answered {
            Ok(Handled::Replied) => Finish::Replied(Box::new(message)),
            Ok(Handled::Deferred) => Finish::Deferred,
            Ok(Handled::Closed) => Finish::ServerClosed,
            Err(error) => {
                tracing::error!(%error, ?server, ?session, "closing the session");
                return close_failed(ipc, server, session, cx.events);
            }
        };
        finish(ipc, session, ending, cx.events)
    }

    /// Starts and runs the ready requests of `server`, one at a time, until none may start.
    fn run_ready(
        &self,
        server: ServerId,
        memory: &dyn GuestMemory,
        process_id: u64,
    ) -> Vec<Finished> {
        let mut finished = Vec::new();
        // Bounded by the requests waiting: every turn starts one of them.
        loop {
            let next = {
                let mut locked = self.lock::<(&mut Scheduler,)>();
                let (scheduler,) = locked.tables();
                scheduler.ipc_mut().1.start_next(server)
            };
            match next {
                Ok(Some(session)) => {
                    finished.extend(self.run_request(server, session, memory, process_id));
                }
                Ok(None) => return finished,
                Err(error) => {
                    tracing::error!(%error, ?server, "the next request could not start");
                    return finished;
                }
            }
        }
    }

    /// Settles requests that are over, in order: writes each reply into its sender's message
    /// buffer, releases what the request held, and records its events and then its system
    /// call's event, all at `time`. Returns every sender's result, the calling thread's
    /// included.
    fn settle(
        &self,
        finished: Vec<Finished>,
        memory: &dyn GuestMemory,
        recorder: &Recorder,
        time: GuestTicks,
    ) -> Vec<Woken> {
        let mut woken = Vec::new();
        for Finished {
            session,
            done,
            events,
        } in finished
        {
            let Done { request, result } = done;
            let sender = request.sender;
            if result.is_ok()
                && let Err(error) = memory.write(sender.message_address, &request.message[..])
            {
                tracing::error!(
                    ?error,
                    "a reply could not be written to its sender's TLS block"
                );
            }
            let code = result.err().map_or(ResultCode::SUCCESS, KernelError::code);
            let mut locked = self.lock::<(&mut Scheduler,)>();
            let (scheduler,) = locked.tables();
            let (objects, ipc) = scheduler.ipc_mut();
            if let Err(error) = ipc.release_request(objects, session, &request) {
                tracing::error!(%error, ?session, "a finished request could not let go");
            }
            for kind in events {
                recorder.record(time, sender.core, sender.thread, kind);
            }
            record_send(recorder, time, sender, code);
            drop(locked);
            woken.push(Woken {
                thread: sender.thread,
                core: sender.core,
                result: code,
            });
        }
        woken
    }
}

/// The first step of `SendSyncRequest`, under the lock: the session, the message, its special
/// header, and the queue.
fn begin(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
    memory: &dyn GuestMemory,
    process_id: u64,
    sender: Sender,
) -> CallResult<Begun> {
    let session = handles
        .get(sender.session_handle)
        .map_err(|_| KernelError::InvalidHandle)?;
    if scheduler.objects().get(session)? != Object::Session {
        return Err(KernelError::InvalidHandle.into());
    }
    let (objects, ipc) = scheduler.ipc_mut();
    // A closed server end refuses before anything of the message is taken.
    if !ipc.is_server_open(session)? {
        return Err(KernelError::SessionClosed.into());
    }
    let mut message: MessageBuffer = [0; BUFFER_SIZE];
    memory
        .read(sender.message_address, &mut message)
        .map_err(|error| {
            SvcError::Internal(format!(
                "the caller's message buffer at {:#x} cannot be read: {error:?}",
                sender.message_address
            ))
        })?;
    let (translated, result) = translate(&mut message, process_id, handles, objects);
    if let Err(error) = result {
        ipc.release_translated(objects, &translated)?;
        return Err(error.into());
    }
    let request = Request {
        sender,
        message: Box::new(message),
        translated,
    };
    match ipc.send(objects, session, request) {
        Ok(Admission::Start) => Ok(Begun::Start {
            server: ipc.server_of(session)?,
            session,
        }),
        Ok(Admission::Queued) => Ok(Begun::Queued),
        Err(refused) => {
            ipc.release_translated(objects, &refused.request.translated)?;
            Err(refused.error.into())
        }
    }
}

/// Ends a started request and returns what is over, each with `events` for the first.
fn finish(
    ipc: &mut crate::ipc::Ipc,
    session: ObjectId,
    ending: Finish,
    events: Vec<EventKind>,
) -> Vec<Finished> {
    let done = match ipc.finish(session, ending) {
        Ok(done) => done,
        Err(error) => {
            tracing::error!(%error, ?session, "a request could not be ended");
            return Vec::new();
        }
    };
    let mut events = Some(events);
    done.into_iter()
        .map(|done| Finished {
            session,
            done,
            events: events.take().unwrap_or_default(),
        })
        .collect()
}

/// Closes the server end of a session whose handler failed (see [`HandlerError`](crate::ipc::HandlerError)): its requests
/// end with [`KernelError::SessionClosed`], and the handler, which still has the session, is
/// told to forget it.
fn close_failed(
    ipc: &mut crate::ipc::Ipc,
    server: ServerId,
    session: ObjectId,
    events: Vec<EventKind>,
) -> Vec<Finished> {
    if let Err(error) = ipc.notify(server, Notice::Closed(session)) {
        tracing::error!(%error, ?server, "the server cannot be told of a session it lost");
    }
    finish(ipc, session, Finish::ServerClosed, events)
}

/// Gives a handler its notices, in order.
fn apply_notices(handler: &mut dyn SessionHandler, server: ServerId, notices: Vec<Notice>) {
    for notice in notices {
        match notice {
            Notice::Accept(session) => {
                if let Err(error) = handler.accept(session) {
                    tracing::error!(%error, ?server, ?session, "the server could not accept a session");
                }
            }
            Notice::Closed(session) => handler.closed(session),
        }
    }
}

/// Writes a result into `x0` as the call's only output.
fn answer(id: SvcId, convention: &Convention, regs: &mut GuestRegs, result: ResultCode) -> Outcome {
    complete(id, convention, regs, &[result_word(result)])
}

/// Records the `SendSyncRequest` event of a sender woken by someone else's dispatch.
fn record_send(recorder: &Recorder, time: GuestTicks, sender: Sender, result: ResultCode) {
    let mut passed = [0; SVC_ARG_COUNT];
    passed[0] = u64::from(sender.session_handle.raw());
    recorder.record(
        time,
        sender.core,
        sender.thread,
        EventKind::Svc {
            id: u32::from(SvcId::SendSyncRequest.number()),
            args: recorded_args(Some(SvcId::SendSyncRequest), &passed),
            result: starquake_trace::ResultCode(result.raw()),
        },
    );
}
