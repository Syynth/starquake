//! The IPC arms of the dispatcher, driven with a scripted handler, flat guest memory and the
//! fixture's address space: `ConnectToNamedPort`, `SendSyncRequest` with every way a request
//! ends, handle copy and move, deferred replies and the queue behind them, and closing either
//! end of a session.

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use starquake_ipc::{CommandType, RequestBuilder};
use starquake_memory::{FlatMemory, PAGE_SIZE};
use starquake_trace::{Event, FixedClock, LiveEnv, VecSink};

use super::*;
use crate::dispatch::Record;
use crate::handle::{MAX_TABLE_SIZE, ObjectRefs};
use crate::id::{SlabId, ThreadId as KernelThreadId};
use crate::ipc::message::{PROCESS_ID_SIZE, PROCESS_ID_SLOT};
use crate::ipc::{DEFAULT_WORKERS, HandlerError};
use crate::memory::fixture::{self, CODE, CODE_SIZE, DATA, DATA_PAGES};
use crate::process::{Console, ProcessDescription};
use crate::svc::DebugSink;
use crate::tables::FIRST_USER_PROCESS_ID;

/// The guest threads of these tests, distinct so a swap shows.
const A: ThreadId = ThreadId(0xA1);
const B: ThreadId = ThreadId(0xB2);
const C: ThreadId = ThreadId(0xC3);
/// Their message buffers: two pages of the fixture's data, which the guest may read and write,
/// and for C the middle of the third, clear of the port names at its start.
const TLS_A: u64 = DATA;
const TLS_B: u64 = DATA + PAGE_SIZE;
const TLS_C: u64 = NAMES + PAGE_SIZE / 2;
/// Where the tests put port names: the data's last page.
const NAMES: u64 = DATA + 2 * PAGE_SIZE;
/// How long a test lets another thread run into a lock the test expects it to wait on. Only a
/// broken kernel finishes inside it; the assertions after it hold however long it is.
const SETTLE: Duration = Duration::from_millis(200);
/// The end of the data: nothing the guest may touch is mapped after it.
const DATA_END: u64 = DATA + DATA_PAGES * PAGE_SIZE;
/// The service numbers of the two servers.
const FIRST: ServiceId = ServiceId(0x51);
const SECOND: ServiceId = ServiceId(0x52);
/// The command id the scripted handlers echo into their replies.
const COMMAND: u32 = 0x77;

/// What a scripted handler was told.
#[derive(Debug, Default)]
struct Log {
    accepted: Vec<ObjectId>,
    closed: Vec<ObjectId>,
    /// Each request it was given: its session and its message.
    handled: Vec<(ObjectId, MessageBuffer)>,
}

/// Deferred replies that are ready, oldest first: the session, and the byte to fill the reply
/// with, or an error.
type Ready = Arc<Mutex<VecDeque<(ObjectId, Result<u8, HandlerError>)>>>;

type Script = Box<
    dyn FnMut(&mut IpcContext<'_>, ObjectId, &mut [u8]) -> Result<Handled, HandlerError> + Send,
>;

/// What a scripted handler does before each `take_reply`.
type TakeReplyHook = Box<dyn FnMut(&mut IpcContext<'_>) + Send>;

/// What a scripted handler does besides its script: run before its next `accept` (taken, so
/// once) and before each `take_reply`.
#[derive(Default)]
struct Hooks {
    accept: Option<Box<dyn FnOnce() + Send>>,
    take_reply: Option<TakeReplyHook>,
}

/// A handler that does what its script says, and logs what it is told.
struct Scripted {
    log: Arc<Mutex<Log>>,
    script: Script,
    ready: Ready,
    hooks: Arc<Mutex<Hooks>>,
}

impl SessionHandler for Scripted {
    fn accept(&mut self, session: ObjectId) -> Result<(), HandlerError> {
        let hook = lock(&self.hooks).accept.take();
        if let Some(hook) = hook {
            hook();
        }
        lock(&self.log).accepted.push(session);
        Ok(())
    }

    fn closed(&mut self, session: ObjectId) {
        lock(&self.log).closed.push(session);
    }

    fn handle(
        &mut self,
        cx: &mut IpcContext<'_>,
        session: ObjectId,
        message: &mut [u8],
    ) -> Result<Handled, HandlerError> {
        let copy: MessageBuffer = message.try_into().unwrap();
        lock(&self.log).handled.push((session, copy));
        (self.script)(cx, session, message)
    }

    fn take_reply(
        &mut self,
        cx: &mut IpcContext<'_>,
        message: &mut [u8],
    ) -> Option<(ObjectId, Result<(), HandlerError>)> {
        if let Some(hook) = lock(&self.hooks).take_reply.as_mut() {
            hook(cx);
        }
        let (session, reply) = lock(&self.ready).pop_front()?;
        Some((session, reply.map(|fill| message.fill(fill))))
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Replies with every byte `fill`, and records an IPC event naming the session's server.
fn reply_with(fill: u8) -> Script {
    Box::new(move |cx, _, message| {
        message.fill(fill);
        cx.record(ipc_event(COMMAND));
        Ok(Handled::Replied)
    })
}

fn ipc_event(command: u32) -> EventKind {
    EventKind::Ipc {
        session: starquake_trace::SessionId(1),
        service: FIRST,
        command,
        request_size: 0,
        reply_size: 0,
        result: starquake_trace::ResultCode(0),
        outcome: starquake_trace::IpcOutcome::Done,
    }
}

struct NoDebug;

impl DebugSink for NoDebug {
    fn output(&mut self, _: GuestTicks, _: &[u8]) {}
}

/// A dispatcher over the fixture's address space with two servers, `first:` on a port and a
/// second one on none, the guest memory behind it, and the events it records.
struct World {
    dispatcher: Dispatcher,
    memory: FlatMemory,
    env: LiveEnv,
    recorder: Recorder,
    events: Arc<VecSink>,
    first: Server,
    second: Server,
}

/// One server's handles for a test: its id, its log, the script slot, its ready replies and
/// its hooks.
struct Server {
    id: ServerId,
    log: Arc<Mutex<Log>>,
    script: Arc<Mutex<Option<Script>>>,
    ready: Ready,
    hooks: Arc<Mutex<Hooks>>,
}

impl Server {
    fn log(&self) -> std::sync::MutexGuard<'_, Log> {
        lock(&self.log)
    }

    /// Makes the server answer the next requests with `script`.
    fn script(&self, script: Script) {
        *lock(&self.script) = Some(script);
    }
}

/// A handler whose script can be changed between requests through `slot`.
fn swappable(slot: Arc<Mutex<Option<Script>>>) -> Script {
    Box::new(move |cx, session, message| {
        let mut slot = lock(&slot);
        let script = slot.as_mut().expect("a script for the request");
        script(cx, session, message)
    })
}

fn add_server(
    dispatcher: &Dispatcher,
    name: &str,
    service: ServiceId,
    port: Option<PortName>,
) -> Server {
    let log = Arc::new(Mutex::new(Log::default()));
    let script = Arc::new(Mutex::new(None));
    let ready = Arc::new(Mutex::new(VecDeque::new()));
    let hooks = Arc::new(Mutex::new(Hooks::default()));
    let handler = Scripted {
        log: Arc::clone(&log),
        script: swappable(Arc::clone(&script)),
        ready: Arc::clone(&ready),
        hooks: Arc::clone(&hooks),
    };
    let id = dispatcher
        .add_server(name, service, port, Box::new(handler))
        .unwrap();
    Server {
        id,
        log,
        script,
        ready,
        hooks,
    }
}

/// A gate a handler stops at on another core: it says it has arrived, then waits to be let
/// through. The test holds the other ends.
struct Gate {
    arrived: SyncSender<()>,
    through: Receiver<()>,
}

/// The test's ends of a [`Gate`]: wait for the handler to arrive, then let it through.
struct GateKeeper {
    arrived: Receiver<()>,
    through: SyncSender<()>,
}

fn gate() -> (Gate, GateKeeper) {
    let (arrived, arrival) = mpsc::sync_channel(1);
    let (open, through) = mpsc::sync_channel(1);
    (
        Gate { arrived, through },
        GateKeeper {
            arrived: arrival,
            through: open,
        },
    )
}

impl Gate {
    fn pass(&self) {
        self.arrived.send(()).unwrap();
        self.through.recv().unwrap();
    }
}

impl GateKeeper {
    fn wait(&self) {
        self.arrived.recv().unwrap();
    }

    fn open(&self) {
        self.through.send(()).unwrap();
    }
}

/// Each thread's message buffer and core.
fn tls_of(thread: ThreadId) -> u64 {
    match thread {
        A => TLS_A,
        B => TLS_B,
        _ => TLS_C,
    }
}

fn core_of(thread: ThreadId) -> u8 {
    match thread {
        A => 1,
        B => 2,
        _ => 3,
    }
}

impl World {
    fn new() -> Self {
        Self::with_handle_table(i32::from(MAX_TABLE_SIZE))
    }

    fn with_handle_table(size: i32) -> Self {
        let mut description = ProcessDescription::homebrew(Console::Switch2);
        description.handle_table_size = size;
        let dispatcher =
            Dispatcher::new(crate::process::create(&description, fixture::loaded().0).unwrap());
        let first = add_server(
            &dispatcher,
            "first",
            FIRST,
            Some(PortName::new("first:").unwrap()),
        );
        let second = add_server(&dispatcher, "second", SECOND, None);
        let recorder = Recorder::new();
        let events = Arc::new(VecSink::new());
        recorder.add_sink(Arc::clone(&events));
        // The code and one page past it, so a read past the data is refused by the address
        // space, not by the memory.
        let size = usize::try_from(CODE_SIZE + PAGE_SIZE).unwrap();
        Self {
            dispatcher,
            memory: FlatMemory::new(CODE, size),
            env: LiveEnv::new(FixedClock::new(1_000, 1_000), 0),
            recorder,
            events,
            first,
            second,
        }
    }

    fn caller(&self, thread: ThreadId) -> Caller<'_> {
        // The caller needs a mutable debug sink for as long as it lives; these tests never
        // write debug strings, so a leaked zero-sized one is enough.
        let debug: &mut NoDebug = Box::leak(Box::new(NoDebug));
        Caller {
            core: core_of(thread),
            thread,
            env: &self.env,
            memory: &self.memory,
            debug,
            recorder: &self.recorder,
        }
    }

    /// Dispatches `number` from `thread` with `x0` and `x1`, its TLS block at `tls`.
    fn call(&self, thread: ThreadId, number: SvcId, x0: u64, x1: u64) -> (Record, GuestRegs) {
        let mut regs = GuestRegs::default();
        regs.x[0] = x0;
        regs.x[1] = x1;
        regs.tpidrro = tls_of(thread);
        let record = self
            .dispatcher
            .dispatch(number.number(), &mut regs, &mut self.caller(thread));
        (record, regs)
    }

    /// `ConnectToNamedPort` from thread A, with the name written at [`NAMES`] first.
    fn connect(&self, name: &[u8]) -> (Record, GuestRegs) {
        self.memory.write(NAMES, name).unwrap();
        self.call(A, SvcId::ConnectToNamedPort, 0, NAMES)
    }

    /// Connects thread A to `first:` and returns the session handle.
    fn session(&self) -> Handle {
        let (record, regs) = self.connect(b"first:\0");
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!(regs.x[0], 0);
        Handle::from_register(regs.x[1])
    }

    /// Puts `message` in `thread`'s TLS block and sends it on `handle`.
    fn send(
        &self,
        thread: ThreadId,
        handle: Handle,
        message: &RequestBuilder,
    ) -> (Record, GuestRegs) {
        let mut buffer = [0; BUFFER_SIZE];
        message.build(&mut buffer).unwrap();
        self.memory.write(tls_of(thread), &buffer).unwrap();
        self.call(thread, SvcId::SendSyncRequest, u64::from(handle.raw()), 0)
    }

    fn tls(&self, thread: ThreadId) -> MessageBuffer {
        let mut buffer = [0; BUFFER_SIZE];
        self.memory.read(tls_of(thread), &mut buffer).unwrap();
        buffer
    }

    fn close(&self, handle: Handle) -> GuestRegs {
        self.call(A, SvcId::CloseHandle, u64::from(handle.raw()), 0)
            .1
    }

    fn deliver(&self, server: &Server) {
        self.dispatcher
            .deliver_replies(server.id, &mut self.caller(A))
            .unwrap();
    }

    fn events(&self) -> Vec<Event> {
        self.events.events()
    }

    fn with_tables<R>(&self, f: impl FnOnce(&HandleTable, &Scheduler) -> R) -> R {
        self.dispatcher
            .with::<(&HandleTable, &Scheduler), _>(|(handles, scheduler)| f(handles, scheduler))
    }

    /// The session object a handle names.
    fn object(&self, handle: Handle) -> ObjectId {
        self.with_tables(|handles, _| handles.get(handle).unwrap())
    }
}

fn request() -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, COMMAND).data(&[0x5A; 8])
}

fn result(regs: &GuestRegs) -> ResultCode {
    ResultCode::from_raw(u32::try_from(regs.x[0]).unwrap())
}

fn send_event(thread: ThreadId, handle: Handle, result: ResultCode) -> (ThreadId, EventKind) {
    let mut args = [0; SVC_ARG_COUNT];
    args[0] = u64::from(handle.raw());
    (
        thread,
        EventKind::Svc {
            id: u32::from(SvcId::SendSyncRequest.number()),
            args,
            result: starquake_trace::ResultCode(result.raw()),
        },
    )
}

fn threads_and_kinds(events: &[Event]) -> Vec<(ThreadId, EventKind)> {
    events
        .iter()
        .map(|event| (event.thread, event.kind))
        .collect()
}

// ----- ConnectToNamedPort --------------------------------------------------------------------

#[test]
fn connecting_to_a_named_port_gives_a_session_the_server_accepts_at_once() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    world.with_tables(|handles, scheduler| {
        assert_eq!(scheduler.objects().get(session), Ok(Object::Session));
        // The handle holds the only reference.
        assert_eq!(scheduler.objects().refs(session), 1);
        assert_eq!(scheduler.ipc().server_of(session), Ok(world.first.id));
        assert_eq!(handles.len(), 1);
        assert!(!scheduler.ipc().has_any_notices());
    });
    assert_eq!(world.first.log().accepted, [session]);
    assert_eq!(world.second.log().accepted, Vec::<ObjectId>::new());
}

#[test]
fn a_port_nothing_is_registered_under_is_not_found() {
    let world = World::new();
    for name in [&b"second\0"[..], b"first\0", b"\0"] {
        let (record, regs) = world.connect(name);
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!(result(&regs), KernelError::NotFound.code());
        // The handle output is written as zero, as for any failing call.
        assert_eq!(regs.x[1], 0);
    }
    world.with_tables(|handles, scheduler| {
        assert!(handles.is_empty());
        assert!(scheduler.objects().is_empty());
    });
}

#[test]
fn connecting_many_times_never_runs_out_of_sessions() {
    // Eden's `KClientPort::CreateSession` can refuse with `ResultLimitReached` or
    // `ResultOutOfSessions`; with no session limits, only the handle table runs out.
    let world = World::new();
    for _ in 0..MAX_TABLE_SIZE {
        world.session();
    }
    let (_, regs) = world.connect(b"first:\0");
    assert_eq!(result(&regs), KernelError::OutOfHandles.code());
    assert_eq!(
        world.first.log().accepted.len(),
        usize::from(MAX_TABLE_SIZE)
    );
}

#[test]
fn a_bad_name_is_refused_before_the_port_is_looked_up() {
    let world = World::new();
    // Twelve bytes and no terminator.
    let (_, regs) = world.connect(b"first:aaaaaa");
    assert_eq!(result(&regs), KernelError::OutOfRange.code());
    // A name that runs off the end of what the guest may read.
    world.memory.write(DATA_END - 2, b"fi").unwrap();
    let (_, regs) = world.call(A, SvcId::ConnectToNamedPort, 0, DATA_END - 2);
    assert_eq!(result(&regs), KernelError::InvalidPointer.code());
    assert_eq!(world.first.log().accepted, Vec::<ObjectId>::new());
}

#[test]
fn a_full_handle_table_gets_no_session_and_the_server_hears_of_none() {
    let world = World::with_handle_table(1);
    world.session();
    let (_, regs) = world.connect(b"first:\0");
    assert_eq!(result(&regs), KernelError::OutOfHandles.code());
    world.with_tables(|handles, scheduler| {
        assert_eq!(handles.len(), 1);
        // The one session left is the first connection's.
        assert_eq!(scheduler.objects().len(), 1);
        assert!(!scheduler.ipc().has_any_notices());
    });
    assert_eq!(world.first.log().accepted.len(), 1);
}

// ----- SendSyncRequest -----------------------------------------------------------------------

#[test]
fn a_reply_is_written_over_the_request_and_recorded_after_its_ipc_events() {
    let world = World::new();
    let handle = world.session();
    world.first.script(reply_with(0xC3));
    let (record, regs) = world.send(A, handle, &request());
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    assert_eq!(world.tls(A), [0xC3; BUFFER_SIZE]);
    // The handler saw the request as sent, on the session the handle names.
    let mut sent = [0; BUFFER_SIZE];
    request().build(&mut sent).unwrap();
    assert_eq!(world.first.log().handled, [(world.object(handle), sent)]);
    let events = world.events();
    assert_eq!(
        threads_and_kinds(&events[1..]),
        [
            (A, ipc_event(COMMAND)),
            send_event(A, handle, ResultCode::SUCCESS)
        ]
    );
    // Both stamped with the caller's core and the dispatch's one clock reading.
    assert_eq!((events[1].core, events[2].core), (1, 1));
    assert_eq!(events[1].time, events[2].time);
    // The request let go of its session.
    let session = world.object(handle);
    world.with_tables(|_, scheduler| assert_eq!(scheduler.objects().refs(session), 1));
}

#[test]
fn a_handle_that_names_no_session_is_an_invalid_handle_and_reaches_no_server() {
    let world = World::new();
    world.first.script(reply_with(0));
    let thread = world
        .dispatcher
        .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
            let thread = scheduler
                .objects_mut()
                .create(Object::Thread(KernelThreadId::from_parts(0, 1)));
            let thread = thread.unwrap();
            let handle = handles.add(scheduler.objects_mut(), thread).unwrap();
            scheduler.objects_mut().close(thread).unwrap();
            handle
        });
    for handle in [
        Handle::INVALID,
        Handle::CURRENT_PROCESS,
        Handle::from_raw(0x1_0042),
        thread,
    ] {
        let (record, regs) = world.send(A, handle, &request());
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!(
            result(&regs),
            KernelError::InvalidHandle.code(),
            "{handle:?}"
        );
    }
    assert_eq!(
        world.first.log().handled,
        Vec::<(ObjectId, MessageBuffer)>::new()
    );
    // Each was recorded with its result.
    let last = world.events().last().map(|event| event.kind);
    assert_eq!(
        last,
        Some(send_event(A, thread, KernelError::InvalidHandle.code()).1)
    );
}

#[test]
fn the_process_id_slot_holds_the_senders_process_id() {
    let world = World::new();
    let handle = world.session();
    world.first.script(Box::new(|cx, _, message| {
        assert_eq!(cx.process_id(), FIRST_USER_PROCESS_ID);
        message.fill(0);
        Ok(Handled::Replied)
    }));
    world.send(A, handle, &request().send_pid(0xDEAD_BEEF));
    let (_, seen) = world.first.log().handled[0];
    let slot = PROCESS_ID_SLOT..PROCESS_ID_SLOT + PROCESS_ID_SIZE;
    assert_eq!(seen[slot], FIRST_USER_PROCESS_ID.to_le_bytes());
}

#[test]
fn a_copied_handle_stays_with_the_sender_and_a_moved_one_goes_to_the_request() {
    let world = World::new();
    let handle = world.session();
    let copied = world.session();
    let moved = world.session();
    let (copied_object, moved_object) = (world.object(copied), world.object(moved));
    let resolved = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&resolved);
    world.first.script(Box::new(move |cx, _, message| {
        lock(&seen).push((cx.resolve_handle(copied), cx.resolve_handle(moved)));
        message.fill(0);
        Ok(Handled::Replied)
    }));
    let (_, regs) = world.send(
        A,
        handle,
        &request().copy_handle(copied.raw()).move_handle(moved.raw()),
    );
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    // The handler resolved both, the moved one although it had left the sender's table.
    assert_eq!(*lock(&resolved), [(Ok(copied_object), Ok(moved_object))]);
    world.with_tables(|handles, scheduler| {
        assert_eq!(handles.get(copied), Ok(copied_object));
        assert!(handles.get(moved).is_err());
        assert_eq!(scheduler.objects().refs(copied_object), 1);
        // The request held the moved session's last reference, and let it go.
        assert!(!scheduler.ipc().is_session(moved_object));
    });
    // The server hears that the moved session is gone.
    assert_eq!(world.first.log().closed, [moved_object]);
}

#[test]
fn a_bad_handle_in_the_message_fails_the_request_before_the_server_sees_it() {
    let world = World::new();
    let handle = world.session();
    let moved = world.session();
    let (session, moved_object) = (world.object(handle), world.object(moved));
    world.first.script(reply_with(0));
    let stale = Handle::from_raw(0x7_0123);
    let (_, regs) = world.send(
        A,
        handle,
        &request().copy_handle(stale.raw()).move_handle(moved.raw()),
    );
    assert_eq!(result(&regs), KernelError::InvalidHandle.code());
    assert_eq!(
        world.first.log().handled,
        Vec::<(ObjectId, MessageBuffer)>::new()
    );
    // The moved handle left the table anyway, and with it its session, whose server is told.
    world.with_tables(|handles, scheduler| {
        assert!(handles.get(moved).is_err());
        assert!(!scheduler.ipc().is_session(moved_object));
        assert!(scheduler.ipc().is_session(session));
    });
    assert_eq!(world.first.log().closed, [moved_object]);
}

#[test]
fn a_request_that_moves_its_own_sessions_last_handle_after_a_bad_one_destroys_the_session() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    world.first.script(reply_with(0));
    let stale = Handle::from_raw(0x7_0123);
    let (_, regs) = world.send(
        A,
        handle,
        &request().copy_handle(stale.raw()).move_handle(handle.raw()),
    );
    assert_eq!(result(&regs), KernelError::InvalidHandle.code());
    world.with_tables(|handles, scheduler| {
        assert!(handles.is_empty());
        assert!(scheduler.objects().is_empty());
        assert!(!scheduler.ipc().is_session(session));
        assert!(!scheduler.ipc().has_any_notices());
    });
    assert_eq!(world.first.log().closed, [session]);
}

#[test]
fn a_close_request_ends_with_session_closed_and_the_session_stays_closed() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    world.first.script(Box::new(|_, _, _| Ok(Handled::Closed)));
    let (_, regs) = world.send(A, handle, &RequestBuilder::close());
    assert_eq!(result(&regs), KernelError::SessionClosed.code());
    // The 0xf601 the reference emulator returns for it.
    assert_eq!(KernelError::SessionClosed.code().raw(), 0xf601);
    // Nothing was written over the request.
    let mut sent = [0; BUFFER_SIZE];
    RequestBuilder::close().build(&mut sent).unwrap();
    assert_eq!(world.tls(A), sent);
    // A later request is refused without reaching the server.
    world.first.script(reply_with(0));
    let (_, regs) = world.send(A, handle, &request());
    assert_eq!(result(&regs), KernelError::SessionClosed.code());
    assert_eq!(world.first.log().handled.len(), 1);
    // Closing the handle destroys the session; the server, which closed it, is not told.
    assert_eq!(result(&world.close(handle)), ResultCode::SUCCESS);
    world.with_tables(|_, scheduler| assert!(!scheduler.ipc().is_session(session)));
    assert_eq!(world.first.log().closed, Vec::<ObjectId>::new());
}

#[test]
fn a_session_whose_server_end_closed_takes_nothing_of_a_new_message() {
    let world = World::new();
    let handle = world.session();
    let moved = world.session();
    world.first.script(Box::new(|_, _, _| Ok(Handled::Closed)));
    world.send(A, handle, &RequestBuilder::close());
    // Both faults at once: the session is closed, and the message moves a handle. The closed
    // session is reported, and the handle stays with the sender (Horizon refuses the request
    // before it reads the message: `KServerSession::OnRequest`).
    let (_, regs) = world.send(A, handle, &request().move_handle(moved.raw()));
    assert_eq!(result(&regs), KernelError::SessionClosed.code());
    let session = world.object(moved);
    world.with_tables(|handles, scheduler| {
        assert_eq!(handles.get(moved), Ok(session));
        assert_eq!(scheduler.objects().refs(session), 1);
    });
}

#[test]
fn a_handler_that_fails_closes_the_session() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Err(HandlerError("broken".to_owned()))));
    let (_, regs) = world.send(A, handle, &request());
    assert_eq!(result(&regs), KernelError::SessionClosed.code());
    // The handler, which still had the session, was told to forget it.
    assert_eq!(world.first.log().closed, [session]);
}

#[test]
fn closing_the_last_handle_tells_the_server_and_one_of_two_does_not() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    let second = world
        .dispatcher
        .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
            handles.add(scheduler.objects_mut(), session).unwrap()
        });
    world.close(handle);
    assert_eq!(world.first.log().closed, Vec::<ObjectId>::new());
    world.close(second);
    assert_eq!(world.first.log().closed, [session]);
    world.with_tables(|handles, scheduler| {
        assert!(handles.is_empty());
        assert!(scheduler.objects().is_empty());
        assert!(!scheduler.ipc().is_session(session));
    });
}

#[test]
fn exiting_the_process_closes_its_sessions_and_the_server_is_told() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    assert_eq!(world.first.log().closed, Vec::<ObjectId>::new());
    let (record, _) = world.call(A, SvcId::ExitProcess, 0, 0);
    assert_eq!(record.outcome, Outcome::ProcessExited);
    assert_eq!(world.first.log().closed, [session]);
    world.with_tables(|handles, scheduler| {
        assert!(handles.is_empty());
        assert!(scheduler.objects().is_empty());
        assert!(!scheduler.ipc().is_session(session));
    });
}

// ----- What a handler reaches the kernel through ---------------------------------------------

#[test]
fn a_handler_runs_without_the_lock_and_can_hand_out_sessions_on_either_server() {
    let world = World::new();
    let handle = world.session();
    let made = Arc::new(Mutex::new(Vec::new()));
    let keep = Arc::clone(&made);
    world.first.script(Box::new(move |cx, _, message| {
        // Each of these takes the kernel lock: they would deadlock if the handler held it.
        let own = cx.create_session().unwrap();
        let other = cx.connect(SECOND).unwrap();
        assert_eq!(cx.connect(ServiceId(0x99)), Err(KernelError::NotFound));
        assert_eq!(cx.resolve_handle(own.1), Ok(own.0));
        lock(&keep).extend([own, other]);
        message.fill(0);
        Ok(Handled::Replied)
    }));
    world.send(A, handle, &request());
    let made = lock(&made).clone();
    assert_eq!(made.len(), 2, "{made:?}");
    let ((own, own_handle), (other, other_handle)) = (made[0], made[1]);
    world.with_tables(|handles, scheduler| {
        assert_eq!(handles.get(own_handle), Ok(own));
        assert_eq!(handles.get(other_handle), Ok(other));
        assert_eq!(scheduler.ipc().server_of(own), Ok(world.first.id));
        assert_eq!(scheduler.ipc().server_of(other), Ok(world.second.id));
        assert_eq!(scheduler.objects().refs(other), 1);
    });
    // The second server accepted the session it was connected to; the first was not told of
    // the session it made itself.
    assert_eq!(world.second.log().accepted, [other]);
    assert_eq!(world.first.log().accepted, [world.object(handle)]);
}

#[test]
fn a_handler_reads_and_writes_only_what_the_guest_may() {
    let world = World::new();
    let handle = world.session();
    world.memory.write(CODE, &[0x11; 4]).unwrap();
    world.first.script(Box::new(|cx, _, message| {
        let mut bytes = [0; 4];
        // Text is readable but not writable.
        assert_eq!(cx.read_memory(CODE, &mut bytes), Ok(()));
        assert_eq!(bytes, [0x11; 4]);
        assert_eq!(
            cx.write_memory(CODE, &bytes),
            Err(KernelError::InvalidCurrentMemory)
        );
        // Data is both, up to its last byte exactly.
        assert_eq!(cx.write_memory(DATA_END - 4, &[0x22; 4]), Ok(()));
        assert_eq!(
            cx.write_memory(DATA_END - 3, &[0x22; 4]),
            Err(KernelError::InvalidCurrentMemory)
        );
        assert_eq!(
            cx.read_memory(DATA_END - 3, &mut bytes),
            Err(KernelError::InvalidCurrentMemory)
        );
        assert_eq!(
            cx.read_memory(u64::MAX - 1, &mut bytes),
            Err(KernelError::InvalidCurrentMemory)
        );
        message.fill(0);
        Ok(Handled::Replied)
    }));
    let (_, regs) = world.send(A, handle, &request());
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    let mut tail = [0; 4];
    world.memory.read(DATA_END - 4, &mut tail).unwrap();
    assert_eq!(tail, [0x22; 4]);
    // The refused write past the end changed nothing.
    let mut past = [0; 1];
    world.memory.read(DATA_END, &mut past).unwrap();
    assert_eq!(past, [0]);
}

// ----- Deferred replies and the queue --------------------------------------------------------

#[test]
fn a_second_sender_waits_behind_a_deferred_request() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    let before = world.events().len();
    let (record, regs) = world.send(A, handle, &request());
    assert_eq!(record.outcome, Outcome::Waiting);
    // Nothing written, nothing recorded yet.
    assert_eq!(regs.x[0], u64::from(handle.raw()));
    assert_eq!(world.events().len(), before);

    // Thread B sends on the same session: it waits too, and the server does not see it.
    world.first.script(reply_with(0xB0));
    let (record, _) = world.send(B, handle, &request());
    assert_eq!(record.outcome, Outcome::Waiting);
    assert_eq!(world.first.log().handled.len(), 1);
    world.with_tables(|_, scheduler| {
        assert_eq!(scheduler.ipc().waiting(session), Ok(1));
        assert_eq!(scheduler.objects().refs(session), 3);
    });
    assert_eq!(world.dispatcher.take_woken(), Vec::<Woken>::new());

    // A's reply is ready: A wakes with it, and B's request runs and is answered at once.
    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    world.deliver(&world.first);
    assert_eq!(
        world.dispatcher.take_woken(),
        [
            Woken {
                thread: A,
                core: 1,
                result: ResultCode::SUCCESS
            },
            Woken {
                thread: B,
                core: 2,
                result: ResultCode::SUCCESS
            },
        ]
    );
    assert_eq!(world.tls(A), [0xA0; BUFFER_SIZE]);
    assert_eq!(world.tls(B), [0xB0; BUFFER_SIZE]);
    assert_eq!(world.first.log().handled.len(), 2);
    assert_eq!(
        threads_and_kinds(&world.events()[before..]),
        [
            send_event(A, handle, ResultCode::SUCCESS),
            (B, ipc_event(COMMAND)),
            send_event(B, handle, ResultCode::SUCCESS),
        ]
    );
    world.with_tables(|_, scheduler| {
        assert_eq!(scheduler.objects().refs(session), 1);
        assert_eq!(scheduler.ipc().queue(world.first.id).unwrap().running(), 0);
    });
    assert_eq!(world.dispatcher.take_woken(), Vec::<Woken>::new());
    assert_eq!(DEFAULT_WORKERS, 1);
}

#[test]
fn a_deferred_request_frees_the_server_for_another_session() {
    let world = World::new();
    let first = world.session();
    let other = world.session();
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, first, &request());
    world.first.script(reply_with(0x0B));
    let (record, regs) = world.send(B, other, &request());
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    assert_eq!(world.tls(B), [0x0B; BUFFER_SIZE]);
}

#[test]
fn closing_the_client_end_of_a_deferred_request_waits_for_its_reply() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    // Another thread closes the handle while A waits: the request keeps the session.
    world.call(B, SvcId::CloseHandle, u64::from(handle.raw()), 0);
    world.with_tables(|_, scheduler| assert!(scheduler.ipc().is_session(session)));
    assert_eq!(world.first.log().closed, Vec::<ObjectId>::new());
    lock(&world.first.ready).push_back((session, Ok(0x5E)));
    world.deliver(&world.first);
    assert_eq!(world.dispatcher.take_woken()[0].result, ResultCode::SUCCESS);
    world.with_tables(|_, scheduler| assert!(!scheduler.ipc().is_session(session)));
    assert_eq!(world.first.log().closed, [session]);
}

#[test]
fn closing_the_server_end_wakes_the_waiting_senders_with_session_closed() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    world.send(B, handle, &request());
    // The deferred reply cannot be written: the server end closes.
    lock(&world.first.ready).push_back((session, Err(HandlerError("lost".to_owned()))));
    world.deliver(&world.first);
    let closed = KernelError::SessionClosed.code();
    let woken: Vec<_> = world
        .dispatcher
        .take_woken()
        .iter()
        .map(|woken| (woken.thread, woken.result))
        .collect();
    assert_eq!(woken, [(A, closed), (B, closed)]);
    // B's request never reached the server, which was told to forget the session.
    assert_eq!(world.first.log().handled.len(), 1);
    assert_eq!(world.first.log().closed, [session]);
}

#[test]
fn delivering_when_nothing_is_ready_changes_nothing() {
    let world = World::new();
    let before = world.events().len();
    world.deliver(&world.first);
    assert_eq!(world.dispatcher.take_woken(), Vec::<Woken>::new());
    assert_eq!(world.events().len(), before);
}

#[test]
fn a_failed_deferred_reply_records_its_events_with_the_first_request_it_ends() {
    const LOST: u32 = 0x99;
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    let before = world.events().len();
    world.send(A, handle, &request());
    world.send(B, handle, &request());
    // The handler records an event while it fails to write A's reply.
    lock(&world.first.hooks).take_reply = Some(Box::new(|cx| cx.record(ipc_event(LOST))));
    lock(&world.first.ready).push_back((session, Err(HandlerError("lost".to_owned()))));
    world.deliver(&world.first);
    let closed = KernelError::SessionClosed.code();
    // The event is A's, recorded before A's system call; B's request never ran.
    assert_eq!(
        threads_and_kinds(&world.events()[before..]),
        [
            (A, ipc_event(LOST)),
            send_event(A, handle, closed),
            send_event(B, handle, closed),
        ]
    );
}

#[test]
fn a_request_that_ends_in_an_error_leaves_the_message_as_it_was_sent() {
    // The process id slot is the one part of the message the kernel rewrites: a request that
    // does not succeed leaves the sender's own value in it.
    const SENT_PID: u64 = 0x0123_4567_89AB_CDEF;
    let world = World::new();
    let closing = world.session();
    let failing = world.session();
    let message = request().send_pid(SENT_PID);
    let mut sent = [0; BUFFER_SIZE];
    message.build(&mut sent).unwrap();
    assert_ne!(SENT_PID, FIRST_USER_PROCESS_ID);

    world.first.script(Box::new(|_, _, _| Ok(Handled::Closed)));
    let (_, regs) = world.send(A, closing, &message);
    assert_eq!(result(&regs), KernelError::SessionClosed.code());
    assert_eq!(world.tls(A), sent);

    world
        .first
        .script(Box::new(|_, _, _| Err(HandlerError("broken".to_owned()))));
    let (_, regs) = world.send(B, failing, &message);
    assert_eq!(result(&regs), KernelError::SessionClosed.code());
    assert_eq!(world.tls(B), sent);
    // Both handlers saw the kernel's id in the slot.
    let slot = PROCESS_ID_SLOT..PROCESS_ID_SLOT + PROCESS_ID_SIZE;
    for (_, seen) in &world.first.log().handled {
        assert_eq!(seen[slot.clone()], FIRST_USER_PROCESS_ID.to_le_bytes());
    }
}

#[test]
fn a_session_connected_while_a_reply_is_delivered_is_accepted_before_the_next_request_runs() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    // B waits on the same session; when its request runs, it notes what the server accepted.
    let accepted_then = Arc::new(Mutex::new(Vec::new()));
    let note = Arc::clone(&accepted_then);
    let log = Arc::clone(&world.first.log);
    world.first.script(Box::new(move |_, _, message| {
        *lock(&note) = lock(&log).accepted.clone();
        message.fill(0xB0);
        Ok(Handled::Replied)
    }));
    world.send(B, handle, &request());
    // Writing A's reply connects a new session on the same server.
    let connected = Arc::new(Mutex::new(None));
    let keep = Arc::clone(&connected);
    lock(&world.first.hooks).take_reply = Some(Box::new(move |cx| {
        if lock(&keep).is_none() {
            *lock(&keep) = Some(cx.connect(FIRST).unwrap().0);
        }
    }));
    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    world.deliver(&world.first);
    let new = lock(&connected).unwrap();
    assert_eq!(world.first.log().accepted, [session, new]);
    assert_eq!(*lock(&accepted_then), [session, new]);
    let woken: Vec<_> = world
        .dispatcher
        .take_woken()
        .iter()
        .map(|woken| (woken.thread, woken.result))
        .collect();
    assert_eq!(woken, [(A, ResultCode::SUCCESS), (B, ResultCode::SUCCESS)]);
}

#[test]
fn a_session_the_server_has_not_accepted_yet_is_accepted_before_a_request_on_it_runs() {
    // `ConnectToNamedPort` queues the server's notice under the lock and gives it to the server
    // under the lock again, later: another core can send on the new handle in between. The
    // test opens the session as the first half does, and sends at once.
    let world = World::new();
    let (session, handle) = world
        .dispatcher
        .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
            open_session_with_handle(handles, scheduler, world.first.id, true)
        })
        .unwrap();
    let accepted_then = Arc::new(Mutex::new(Vec::new()));
    let note = Arc::clone(&accepted_then);
    let log = Arc::clone(&world.first.log);
    world.first.script(Box::new(move |_, _, message| {
        *lock(&note) = lock(&log).accepted.clone();
        message.fill(0);
        Ok(Handled::Replied)
    }));
    let (_, regs) = world.send(B, handle, &request());
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    assert_eq!(*lock(&accepted_then), [session]);
    assert_eq!(world.first.log().accepted, [session]);
}

// ----- Several cores at once -----------------------------------------------------------------

#[test]
fn a_request_sent_while_another_core_has_the_server_accept_a_session_is_answered() {
    let world = World::new();
    let handle = world.session();
    world.first.script(reply_with(0xB0));
    let (gate, keeper) = gate();
    lock(&world.first.hooks).accept = Some(Box::new(move || gate.pass()));
    let world = &world;
    std::thread::scope(|scope| {
        // A connects, and the server stops while it accepts the new session.
        let connecting = scope.spawn(|| world.connect(b"first:\0"));
        keeper.wait();
        // B sends on the existing session meanwhile.
        let (sent_tx, sent_rx) = mpsc::channel();
        let sending = scope.spawn(move || {
            let sent = world.send(B, handle, &request());
            sent_tx.send(()).unwrap();
            sent
        });
        // A kernel that runs B now has no handler to run it with; one that is right makes B
        // wait for the server.
        let _ = sent_rx.recv_timeout(SETTLE);
        keeper.open();
        let (record, regs) = sending.join().unwrap();
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!(result(&regs), ResultCode::SUCCESS);
        assert_eq!(world.tls(B), [0xB0; BUFFER_SIZE]);
        let (_, regs) = connecting.join().unwrap();
        assert_eq!(result(&regs), ResultCode::SUCCESS);
    });
    assert_eq!(world.first.log().accepted.len(), 2);
    assert_eq!(world.first.log().handled.len(), 1);
}

#[test]
fn a_session_connected_while_another_core_has_the_server_accept_one_is_accepted_too() {
    let world = World::new();
    let (gate, keeper) = gate();
    lock(&world.first.hooks).accept = Some(Box::new(move || gate.pass()));
    let world = &world;
    std::thread::scope(|scope| {
        let connecting = scope.spawn(|| world.connect(b"first:\0"));
        keeper.wait();
        let (sent_tx, sent_rx) = mpsc::channel();
        let second = scope.spawn(move || {
            let connected = world.call(B, SvcId::ConnectToNamedPort, 0, NAMES);
            sent_tx.send(()).unwrap();
            connected
        });
        let _ = sent_rx.recv_timeout(SETTLE);
        keeper.open();
        for joined in [connecting.join().unwrap(), second.join().unwrap()] {
            assert_eq!(result(&joined.1), ResultCode::SUCCESS);
        }
    });
    // The second session's notice did not wait for the server's next activity.
    assert_eq!(world.first.log().accepted.len(), 2);
    world.with_tables(|_, scheduler| assert!(!scheduler.ipc().has_any_notices()));
}

#[test]
fn requests_sent_while_another_core_delivers_a_reply_wait_and_are_woken() {
    let world = World::new();
    let deferred = world.session();
    let (b_session, c_session) = (world.session(), world.session());
    let deferred_object = world.object(deferred);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    let (record, _) = world.send(A, deferred, &request());
    assert_eq!(record.outcome, Outcome::Waiting);
    world.first.script(reply_with(0xBC));
    let (gate, keeper) = gate();
    let gate = Mutex::new(Some(gate));
    lock(&world.first.hooks).take_reply = Some(Box::new(move |_| {
        if let Some(gate) = lock(&gate).take() {
            gate.pass();
        }
    }));
    lock(&world.first.ready).push_back((deferred_object, Ok(0xA0)));
    std::thread::scope(|scope| {
        // Another core writes A's reply, and the handler stops in the middle of it.
        let delivering = scope.spawn(|| world.deliver(&world.first));
        keeper.wait();
        // B and C send on two other sessions of the same server meanwhile: both wait, B for
        // the handler, C behind B for the server's one worker.
        let (b_record, _) = world.send(B, b_session, &request());
        let (c_record, _) = world.send(C, c_session, &request());
        keeper.open();
        delivering.join().unwrap();
        assert_eq!(
            (b_record.outcome, c_record.outcome),
            (Outcome::Waiting, Outcome::Waiting)
        );
    });
    let woken: Vec<_> = world
        .dispatcher
        .take_woken()
        .iter()
        .map(|woken| (woken.thread, woken.result))
        .collect();
    assert_eq!(
        woken,
        [
            (A, ResultCode::SUCCESS),
            (B, ResultCode::SUCCESS),
            (C, ResultCode::SUCCESS),
        ]
    );
    assert_eq!(world.tls(A), [0xA0; BUFFER_SIZE]);
    assert_eq!(world.tls(B), [0xBC; BUFFER_SIZE]);
    assert_eq!(world.tls(C), [0xBC; BUFFER_SIZE]);
    world.with_tables(|_, scheduler| {
        assert_eq!(scheduler.ipc().queue(world.first.id).unwrap().running(), 0);
    });
}

#[test]
fn a_request_on_a_second_session_waits_while_another_core_runs_the_handler() {
    let world = World::new();
    let (first, other) = (world.session(), world.session());
    let first_object = world.object(first);
    let (gate, keeper) = gate();
    world.first.script(Box::new(move |_, session, message| {
        if session == first_object {
            gate.pass();
        }
        message.fill(0xEE);
        Ok(Handled::Replied)
    }));
    std::thread::scope(|scope| {
        let sending = scope.spawn(|| world.send(A, first, &request()));
        keeper.wait();
        let (b_record, _) = world.send(B, other, &request());
        keeper.open();
        assert_eq!(b_record.outcome, Outcome::Waiting);
        let (record, regs) = sending.join().unwrap();
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!(result(&regs), ResultCode::SUCCESS);
    });
    // A's core ran B's request after A's, and B is reported woken with its reply written.
    assert_eq!(
        world.dispatcher.take_woken(),
        [Woken {
            thread: B,
            core: core_of(B),
            result: ResultCode::SUCCESS
        }]
    );
    assert_eq!(world.tls(B), [0xEE; BUFFER_SIZE]);
}
