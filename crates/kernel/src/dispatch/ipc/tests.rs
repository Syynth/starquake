//! The IPC arms of the dispatcher, driven with a scripted handler, flat guest memory and the
//! fixture's address space: `ConnectToNamedPort`, `SendSyncRequest` with every way a request
//! ends, handle copy and move, deferred replies and the queue behind them, and closing either
//! end of a session.

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use starquake_ipc::{CommandType, ReplyBuilder, RequestBuilder};
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

mod rollback;

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
        interface: starquake_trace::InterfaceName::lossy("IScripted"),
        protocol: starquake_trace::IpcProtocol::Cmif,
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

/// How long either side of a [`Gate`] waits for the other before the test fails, so that a
/// broken kernel cannot hang the gate.
const GATE_LIMIT: Duration = Duration::from_secs(30);

impl Gate {
    fn pass(&self) {
        self.arrived.send(()).unwrap();
        self.through
            .recv_timeout(GATE_LIMIT)
            .expect("the test opens the gate");
    }
}

impl GateKeeper {
    fn wait(&self) {
        self.arrived
            .recv_timeout(GATE_LIMIT)
            .expect("the handler arrives at the gate");
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
        Self::with_process(crate::process::create(&description, fixture::loaded().0).unwrap())
    }

    /// A world around the tables of a process already made.
    fn with_process(tables: Tables) -> Self {
        let dispatcher = Dispatcher::new(tables);
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

    /// Delivers `server`'s ready replies as the timer thread does, with A's core as the caller,
    /// and returns the processes the delivery ended.
    fn deliver(&self, server: &Server) -> Vec<Ended> {
        self.dispatcher
            .deliver_replies(server.id, &mut self.caller(A))
            .unwrap()
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

/// Writes a successful reply that copies `handles` to the sender, as a server gives the handles
/// it placed: the sender keeps exactly the placed handles its reply names
/// (`dispatch/ipc/placed.rs`).
fn name_in_reply(message: &mut [u8], handles: &[Handle]) {
    let reply = handles.iter().fold(ReplyBuilder::cmif(0), |reply, handle| {
        reply.copy_handle(handle.raw())
    });
    reply.build(message).unwrap();
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
        assert_eq!(scheduler.objects_besides_processes(), 0);
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
        assert_eq!(scheduler.objects_besides_processes(), 1);
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

/// What a handler resolves each of a request's handles to (`IpcContext::resolve_handle`), and
/// the reply it gives: every byte zero.
fn resolving(handles: Vec<Handle>, seen: Arc<Mutex<Vec<SvcResult<ObjectId>>>>) -> Script {
    Box::new(move |cx, _, message| {
        let resolved = handles.iter().map(|handle| cx.resolve_handle(*handle));
        lock(&seen).extend(resolved);
        message.fill(0);
        Ok(Handled::Replied)
    })
}

/// The process `object` is the object of, or `None` for any other kind of object.
fn process_named(scheduler: &Scheduler, object: ObjectId) -> Option<crate::id::ProcessId> {
    if let Ok(Object::Process(process)) = scheduler.objects().get(object) {
        Some(process)
    } else {
        None
    }
}

/// A world whose process runs a user thread of its own on `thread`'s core, as the core loop
/// runs its main thread, and that thread's object.
fn world_running_a_user_thread(thread: ThreadId) -> (World, ObjectId) {
    let mut tables = crate::process::create(
        &ProcessDescription::homebrew(Console::Switch2),
        fixture::loaded().0,
    )
    .unwrap();
    let own = tables.processes.own();
    let object = run_user_thread(&mut tables, own, thread);
    (World::with_process(tables), object)
}

/// Starts a user thread of `process` on `thread`'s core and has the core run it; returns its
/// object.
fn run_user_thread(tables: &mut Tables, process: ProcessId, thread: ThreadId) -> ObjectId {
    let core = crate::sched::CoreId::new(core_of(thread)).unwrap();
    let spec = crate::sched::UserThread {
        entry: CODE,
        argument: 0,
        stack_top: DATA_END,
        priority: crate::sched::Priority::new(44).unwrap(),
        // Virtual core n is physical core n for the cores these tests use
        // (`VIRTUAL_TO_PHYSICAL_CORE`).
        virtual_core: core.index(),
    };
    let scheduler = &mut tables.scheduler;
    let (id, object) = scheduler
        .create_user_thread(&mut tables.address_space, process, &spec)
        .unwrap();
    scheduler.start(id).unwrap();
    scheduler.update(GuestTicks(0), None, &mut Quiet).unwrap();
    assert_eq!(scheduler.schedule(core, &mut Quiet), Ok(Some(id)));
    object
}

/// A world of two processes: the tables' own, which runs a user thread on A's core, and a
/// second one, which runs a user thread on B's core. Returns the world and the two processes,
/// A's first.
fn world_of_two_processes() -> (World, ProcessId, ProcessId) {
    let description = ProcessDescription::homebrew(Console::Switch2);
    let mut tables = crate::process::create(&description, fixture::loaded().0).unwrap();
    let own = tables.processes.own();
    let other = tables.processes.create(&description).unwrap();
    tables.scheduler.add_process(other).unwrap();
    run_user_thread(&mut tables, own, A);
    run_user_thread(&mut tables, other, B);
    (World::with_process(tables), own, other)
}

/// Whether `process` has ended, and how many of its threads have not.
fn process_state(world: &World, process: ProcessId) -> (bool, usize) {
    world.with_tables(|_, scheduler| {
        let record = scheduler.process(process).unwrap();
        let running = record
            .threads()
            .filter(|thread| {
                scheduler.thread(*thread).unwrap().state() != crate::sched::ThreadState::Terminated
            })
            .count();
        (
            record.state() == crate::process::ProcessState::Terminated,
            running,
        )
    })
}

#[test]
fn a_copied_current_process_reaches_the_server_as_the_senders_process_and_is_let_go() {
    let world = World::new();
    let handle = world.session();
    let seen = Arc::new(Mutex::new(Vec::new()));
    world
        .first
        .script(resolving(vec![Handle::CURRENT_PROCESS], Arc::clone(&seen)));
    // As libnx's `_appletGetSessionProxy` sends `OpenApplicationProxy`: the process id and
    // the current-process pseudo-handle.
    let (record, regs) = world.send(
        A,
        handle,
        &request()
            .send_pid(0)
            .copy_handle(Handle::CURRENT_PROCESS.raw()),
    );
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    let object = lock(&seen)[0].unwrap();
    world.with_tables(|handles, scheduler| {
        let process = process_named(scheduler, object).unwrap();
        assert_eq!(scheduler.process(process).unwrap().object(), object);
        // The request's reference went when it ended; the process's own stays.
        assert_eq!(scheduler.objects().refs(object), 1);
        // The pseudo-handle took no entry in the table.
        assert_eq!(handles.len(), 1);
    });
}

#[test]
fn a_copied_current_thread_reaches_the_server_as_the_calling_thread() {
    let (world, thread) = world_running_a_user_thread(A);
    let handle = world.session();
    let seen = Arc::new(Mutex::new(Vec::new()));
    world.first.script(resolving(
        vec![Handle::CURRENT_THREAD, Handle::CURRENT_PROCESS],
        Arc::clone(&seen),
    ));
    let thread_refs = world.with_tables(|_, scheduler| scheduler.objects().refs(thread));
    let (record, regs) = world.send(
        A,
        handle,
        &request()
            .copy_handle(Handle::CURRENT_THREAD.raw())
            .copy_handle(Handle::CURRENT_PROCESS.raw()),
    );
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    let seen = lock(&seen).clone();
    assert_eq!(seen[0], Ok(thread));
    world.with_tables(|_, scheduler| {
        let process = process_named(scheduler, seen[1].unwrap()).unwrap();
        assert_eq!(seen[1], Ok(scheduler.process(process).unwrap().object()));
        assert_eq!(scheduler.objects().refs(thread), thread_refs);
    });
}

#[test]
fn a_copied_current_thread_names_the_thread_on_the_senders_own_core() {
    // A user thread runs on A's core; B sends from its own core, which runs nothing.
    let (world, thread) = world_running_a_user_thread(A);
    assert_ne!(core_of(A), core_of(B));
    let handle = world.session();
    let seen = Arc::new(Mutex::new(Vec::new()));
    world
        .first
        .script(resolving(vec![Handle::CURRENT_THREAD], Arc::clone(&seen)));
    let (record, regs) = world.send(
        B,
        handle,
        &request().copy_handle(Handle::CURRENT_THREAD.raw()),
    );
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(result(&regs), KernelError::InvalidHandle.code());
    assert_eq!(
        world.first.log().handled,
        Vec::<(ObjectId, MessageBuffer)>::new()
    );
    // The same request from A's core reaches the server as A's core's thread.
    let (record, regs) = world.send(
        A,
        handle,
        &request().copy_handle(Handle::CURRENT_THREAD.raw()),
    );
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    assert_eq!(*lock(&seen), [Ok(thread)]);
}

#[test]
fn a_request_holding_current_process_that_fails_on_a_later_bad_handle_lets_the_process_go() {
    let tables = crate::process::create(
        &ProcessDescription::homebrew(Console::Switch2),
        fixture::loaded().0,
    )
    .unwrap();
    let own = tables.processes.own();
    let process = tables.scheduler.process(own).unwrap().object();
    let world = World::with_process(tables);
    let handle = world.session();
    world.first.script(reply_with(0));
    assert_eq!(
        world.with_tables(|_, scheduler| scheduler.objects().refs(process)),
        1
    );
    let stale = Handle::from_raw(0x7_0123);
    let (record, regs) = world.send(
        A,
        handle,
        &request()
            .copy_handle(Handle::CURRENT_PROCESS.raw())
            .copy_handle(stale.raw()),
    );
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(result(&regs), KernelError::InvalidHandle.code());
    assert_eq!(
        world.first.log().handled,
        Vec::<(ObjectId, MessageBuffer)>::new()
    );
    // The reference the request held went with it; the process's own stays.
    world.with_tables(|handles, scheduler| {
        assert_eq!(scheduler.objects().refs(process), 1);
        assert_eq!(handles.len(), 1);
    });
}

#[test]
fn a_copied_current_thread_from_a_core_whose_thread_has_no_object_is_an_invalid_handle() {
    let world = World::new();
    let handle = world.session();
    // A core with no scheduler thread, then one running a thread of no process.
    for running in [false, true] {
        if running {
            run_thread_for(&world, A);
        }
        let (record, regs) = world.send(
            A,
            handle,
            &request().copy_handle(Handle::CURRENT_THREAD.raw()),
        );
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!(
            result(&regs),
            KernelError::InvalidHandle.code(),
            "running: {running}"
        );
    }
    assert_eq!(
        world.first.log().handled,
        Vec::<(ObjectId, MessageBuffer)>::new()
    );
}

#[test]
fn a_moved_current_process_is_an_invalid_handle_and_reaches_no_server() {
    let world = World::new();
    let handle = world.session();
    let before = world.with_tables(|_, scheduler| scheduler.objects().len());
    let (record, regs) = world.send(
        A,
        handle,
        &request().move_handle(Handle::CURRENT_PROCESS.raw()),
    );
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(result(&regs), KernelError::InvalidHandle.code());
    assert_eq!(
        world.first.log().handled,
        Vec::<(ObjectId, MessageBuffer)>::new()
    );
    // Nothing was taken or let go: the session's handle and the objects are as they were.
    world.with_tables(|handles, scheduler| {
        assert_eq!(handles.len(), 1);
        assert_eq!(scheduler.objects().len(), before);
        assert_eq!(scheduler.objects_besides_processes(), 1);
    });
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
        assert_eq!(scheduler.objects_besides_processes(), 0);
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
        assert_eq!(scheduler.objects_besides_processes(), 0);
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
        assert_eq!(scheduler.objects_besides_processes(), 0);
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
        name_in_reply(message, &[own.1, other.1]);
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

// ----- Senders the scheduler runs (kernel slice 10) -------------------------------------------

use crate::sched::ThreadState;

/// Signals nobody listens to: these tests switch cores by hand.
struct Quiet;

impl crate::sched::CoreSignals for Quiet {
    fn interrupt(&mut self, _: crate::sched::CoreId) {}

    fn wake(&mut self, _: crate::sched::CoreId) {}
}

/// Makes `thread`'s core run a scheduler thread of its own (one of no process), as the core
/// loop does, and returns it.
fn run_thread_for(world: &World, thread: ThreadId) -> KernelThreadId {
    let core = crate::sched::CoreId::new(core_of(thread)).unwrap();
    world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| {
            let priority = crate::sched::Priority::new(44).unwrap();
            let id = scheduler.create_thread(priority, core).unwrap();
            scheduler.start(id).unwrap();
            scheduler.update(GuestTicks(0), None, &mut Quiet).unwrap();
            assert_eq!(scheduler.schedule(core, &mut Quiet), Ok(Some(id)));
            id
        })
}

/// The run state of a scheduler thread, and whether it waits for a reply on `session`.
fn reply_state(world: &World, thread: KernelThreadId, session: ObjectId) -> (ThreadState, bool) {
    world.with_tables(|_, scheduler| {
        let waits = matches!(
            scheduler.sync().wait(thread),
            Some(crate::sync::Wait::Reply { session: waited, .. }) if *waited == session
        );
        (scheduler.thread(thread).unwrap().state(), waits)
    })
}

/// A thread's saved `x0`.
fn saved_x0(world: &World, thread: KernelThreadId) -> u64 {
    world.with_tables(|_, scheduler| scheduler.thread(thread).unwrap().context().x[0])
}

#[test]
fn senders_that_wait_are_blocked_threads_and_the_reply_makes_them_runnable() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    let (a, b) = (run_thread_for(&world, A), run_thread_for(&world, B));
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    // A's request is deferred, B's waits behind it: both wait for their replies.
    assert_eq!(
        world.send(A, handle, &request()).0.outcome,
        Outcome::Waiting
    );
    world.first.script(reply_with(0xB0));
    assert_eq!(
        world.send(B, handle, &request()).0.outcome,
        Outcome::Waiting
    );
    for thread in [a, b] {
        assert_eq!(
            reply_state(&world, thread, session),
            (ThreadState::Waiting, true)
        );
        // The context saved when the request was sent still has the handle in `x0`.
        assert_eq!(saved_x0(&world, thread), u64::from(handle.raw()));
    }
    // The context is the registers of the call: its message buffer is where the reply goes.
    world.with_tables(|_, scheduler| {
        assert_eq!(scheduler.thread(b).unwrap().context().tpidrro, TLS_B);
    });

    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    world.deliver(&world.first);
    for thread in [a, b] {
        assert_eq!(
            reply_state(&world, thread, session),
            (ThreadState::Runnable, false)
        );
        // The result is in the saved `x0`, where a switch loads it from.
        assert_eq!(
            saved_x0(&world, thread),
            u64::from(ResultCode::SUCCESS.raw())
        );
    }
    assert_eq!(world.tls(A), [0xA0; BUFFER_SIZE]);
    assert_eq!(world.tls(B), [0xB0; BUFFER_SIZE]);
    // They are still reported, as before.
    assert_eq!(world.dispatcher.take_woken().len(), 2);
}

#[test]
fn a_sender_answered_at_once_never_waits() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    let a = run_thread_for(&world, A);
    world.first.script(reply_with(0xA1));
    let (record, regs) = world.send(A, handle, &request());
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    assert_eq!(
        reply_state(&world, a, session),
        (ThreadState::Runnable, false)
    );
    assert_eq!(world.tls(A), [0xA1; BUFFER_SIZE]);
}

#[test]
fn a_thread_that_ends_while_its_request_waits_leaves_the_queue() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    let (_, b) = (run_thread_for(&world, A), run_thread_for(&world, B));
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    world.first.script(reply_with(0xB0));
    world.send(B, handle, &request());
    world.with_tables(|_, scheduler| {
        assert_eq!(scheduler.ipc().waiting(session), Ok(1));
        // The handle's reference and each request's.
        assert_eq!(scheduler.objects().refs(session), 3);
    });

    world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| scheduler.terminate_thread(b))
        .unwrap();
    world.with_tables(|_, scheduler| {
        assert_eq!(scheduler.ipc().waiting(session), Ok(0));
        assert_eq!(scheduler.objects().refs(session), 2);
        assert_eq!(scheduler.sync().wait(b), None);
    });

    // A's reply comes; B's request is gone, so the server never sees it.
    let b_buffer = world.tls(B);
    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    world.deliver(&world.first);
    assert_eq!(world.first.log().handled.len(), 1);
    assert_eq!(world.tls(B), b_buffer);
    world.with_tables(|_, scheduler| {
        assert_eq!(scheduler.objects().refs(session), 1);
        assert_eq!(scheduler.ipc().queue(world.first.id).unwrap().running(), 0);
    });
}

#[test]
fn a_thread_that_ends_while_its_request_is_in_flight_gets_no_reply_and_no_event() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    let a = run_thread_for(&world, A);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| scheduler.terminate_thread(a))
        .unwrap();
    // The request is with its handler, so it stays until the handler answers.
    world.with_tables(|_, scheduler| {
        assert_eq!(scheduler.objects().refs(session), 2);
    });
    let before = world.events().len();
    let a_buffer = world.tls(A);
    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    world.deliver(&world.first);
    assert_eq!(world.tls(A), a_buffer, "no reply for an ended thread");
    assert_eq!(world.events().len(), before, "no event for an ended thread");
    assert_eq!(world.dispatcher.take_woken(), Vec::<Woken>::new());
    world.with_tables(|_, scheduler| {
        assert_eq!(scheduler.objects().refs(session), 1);
        assert_eq!(
            scheduler.thread(a).unwrap().state(),
            ThreadState::Terminated
        );
    });
}

#[test]
fn a_sender_that_ends_while_its_handler_runs_stays_ended_when_the_request_is_deferred() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    let a = run_thread_for(&world, A);
    let (gate, keeper) = gate();
    world.first.script(Box::new(move |_, _, _| {
        gate.pass();
        Ok(Handled::Deferred)
    }));
    std::thread::scope(|scope| {
        let sending = scope.spawn(|| world.send(A, handle, &request()));
        keeper.wait();
        // A ends while its handler runs without the lock; the handler then defers.
        world
            .dispatcher
            .with::<(&mut Scheduler,), _>(|(scheduler,)| scheduler.terminate_thread(a))
            .unwrap();
        keeper.open();
        assert_eq!(sending.join().unwrap().0.outcome, Outcome::Waiting);
    });
    assert_eq!(
        reply_state(&world, a, session),
        (ThreadState::Terminated, false)
    );
    // The reply comes: nothing is written, recorded or woken, and A stays ended.
    let a_buffer = world.tls(A);
    let before = world.events().len();
    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    world.deliver(&world.first);
    assert_eq!(
        reply_state(&world, a, session),
        (ThreadState::Terminated, false)
    );
    assert_eq!(world.tls(A), a_buffer);
    assert_eq!(world.events().len(), before);
    assert_eq!(world.dispatcher.take_woken(), Vec::<Woken>::new());
    world.with_tables(|_, scheduler| {
        assert_eq!(scheduler.ipc().queue(world.first.id).unwrap().running(), 0);
    });
}

#[test]
fn replies_said_to_be_ready_while_another_core_runs_the_handler_are_delivered() {
    let world = World::new();
    let (deferred, other) = (world.session(), world.session());
    let deferred_object = world.object(deferred);
    let a = run_thread_for(&world, A);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    assert_eq!(
        world.send(A, deferred, &request()).0.outcome,
        Outcome::Waiting
    );
    let (gate, keeper) = gate();
    world.first.script(Box::new(move |_, _, message| {
        gate.pass();
        message.fill(0xB0);
        Ok(Handled::Replied)
    }));
    std::thread::scope(|scope| {
        // B's core runs its request through the handler, which stops in the middle of it.
        let sending = scope.spawn(|| world.send(B, other, &request()));
        keeper.wait();
        // A's reply is said to be ready meanwhile: the handler is out, so the delivery is
        // owed, and A still waits.
        lock(&world.first.ready).push_back((deferred_object, Ok(0xA0)));
        world.deliver(&world.first);
        assert_eq!(
            reply_state(&world, a, deferred_object),
            (ThreadState::Waiting, true)
        );
        keeper.open();
        let (record, regs) = sending.join().unwrap();
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!(result(&regs), ResultCode::SUCCESS);
    });
    // B's core delivered A's reply when it put the handler back.
    assert_eq!(
        reply_state(&world, a, deferred_object),
        (ThreadState::Runnable, false)
    );
    assert_eq!(world.tls(A), [0xA0; BUFFER_SIZE]);
    assert_eq!(world.tls(B), [0xB0; BUFFER_SIZE]);
    assert_eq!(
        world.dispatcher.take_woken(),
        [Woken {
            thread: A,
            core: core_of(A),
            result: ResultCode::SUCCESS
        }]
    );
    assert!(lock(&world.first.ready).is_empty());
}

// ----- Whether a run is stalled (`Dispatcher::is_stalled`) ------------------------------------

#[test]
fn a_run_whose_only_thread_waits_for_a_deferred_reply_is_not_stalled() {
    // The sender is a scheduler thread: it waits for its reply, and the request is deferred.
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    let a = run_thread_for(&world, A);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    assert_eq!(
        reply_state(&world, a, session),
        (ThreadState::Waiting, true)
    );
    world.with_tables(|_, scheduler| {
        assert!(!scheduler.has_runnable_thread());
        assert_eq!(scheduler.sync().next_deadline(), None);
    });
    assert!(!world.dispatcher.is_stalled());

    // The sender is no scheduler thread, so nothing waits for a reply: the deferred request
    // alone keeps the run from being stalled.
    let world = World::new();
    assert!(world.dispatcher.is_stalled());
    let handle = world.session();
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    world.with_tables(|_, scheduler| assert!(!scheduler.sync().has_reply_wait()));
    assert!(!world.dispatcher.is_stalled());
}

/// What [`Dispatcher::settle_and_end`] does with the requests a delivery of the first server's
/// replies leaves over, in a world whose process runs A, whose deferred request is answered:
/// how many times it takes the lock, and the processes it ends. `end` is whether the handler
/// asks to end the process as it gives the reply. Every other request in flight (`waiting`
/// senders queued behind A's) is answered by the same delivery.
fn settling_takes_the_lock(end: bool, waiting: &[ThreadId]) -> (u64, Vec<Ended>) {
    let (world, _) = world_running_a_user_thread(A);
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    world.first.script(reply_with(0xBC));
    for sender in waiting {
        world.send(*sender, handle, &request());
    }
    if end {
        lock(&world.first.hooks).take_reply = Some(Box::new(ask_to_end));
    }
    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    // The first half of a delivery (`deliver_replies`), by hand.
    let server = world.first.id;
    let process_id = world
        .dispatcher
        .lock::<(&mut Scheduler,)>()
        .state
        .tables
        .processes
        .process_id();
    world
        .dispatcher
        .lock::<(&mut Scheduler,)>()
        .state
        .owed
        .insert(server);
    let finished = world
        .dispatcher
        .deliver_owed(server, &world.memory, process_id);
    assert_eq!(finished.len(), 1 + waiting.len());
    let before = world.dispatcher.lock_waits().lock.acquisitions;
    let (mine, ended) = world.dispatcher.settle_and_end(
        finished,
        None,
        &world.memory,
        &world.recorder,
        GuestTicks(0),
    );
    let taken = world.dispatcher.lock_waits().lock.acquisitions - before;
    assert_eq!(mine, None);
    assert_eq!(world.tls(A), [0xA0; BUFFER_SIZE]);
    (taken, ended)
}

#[test]
fn replies_and_the_end_they_ask_for_are_settled_under_one_hold_of_the_lock() {
    // Each request with no end asked is settled under a lock of its own, and the woken are
    // reported under one more.
    for waiting in [&[][..], &[B][..], &[B, C][..]] {
        let count = u64::try_from(waiting.len()).unwrap();
        assert_eq!(
            settling_takes_the_lock(false, waiting),
            (count + 2, Vec::new())
        );
        // With an end asked, everything (every reply, the reports and the end) is one hold, so
        // no core can run a sender of the ending process between its reply and the end.
        let (taken, ended) = settling_takes_the_lock(true, waiting);
        assert_eq!((taken, ended.len()), (1, 1));
    }
}

#[test]
fn a_run_whose_process_a_delivery_ended_is_not_stalled() {
    // What a core that parks after the timer thread's delivery ended the process, and before
    // the timer thread stops the run, sees: nothing could run or end a wait, but the run is
    // over, not deadlocked.
    let (world, thread) = world_running_a_user_thread(A);
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    lock(&world.first.hooks).take_reply = Some(Box::new(ask_to_end));
    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    assert_eq!(world.deliver(&world.first).len(), 1);
    assert_eq!(holdings(&world, thread), (0, 1, 0, true));
    world.with_tables(|_, scheduler| {
        assert!(!scheduler.has_runnable_thread());
        assert_eq!(scheduler.sync().next_deadline(), None);
        assert!(!scheduler.sync().has_reply_wait());
        let ipc = scheduler.ipc();
        assert!(ipc.servers().all(|server| {
            ipc.queue(server)
                .is_ok_and(|queue| queue.deferred().next().is_none())
        }));
    });
    assert!(!world.dispatcher.is_stalled());
}

#[test]
fn a_run_is_not_stalled_while_a_handler_is_out_of_its_slot() {
    // The sender is no scheduler thread: nothing is runnable or waits, and nothing is
    // deferred, but the handler running on another thread may still end something.
    let world = World::new();
    let handle = world.session();
    let (gate, keeper) = gate();
    world.first.script(Box::new(move |_, _, message| {
        gate.pass();
        message.fill(0xA0);
        Ok(Handled::Replied)
    }));
    assert!(world.dispatcher.is_stalled());
    std::thread::scope(|scope| {
        let sending = scope.spawn(|| world.send(A, handle, &request()));
        keeper.wait();
        assert!(!world.dispatcher.is_stalled());
        keeper.open();
        assert_eq!(sending.join().unwrap().0.outcome, Outcome::Completed);
    });
    assert!(world.dispatcher.is_stalled());
}

#[test]
fn a_run_is_not_stalled_while_a_thread_waits_for_a_reply_nothing_defers() {
    // The window between a delivery completing a deferred request (it is deferred no more,
    // and the handler is back in its slot) and settling it, which wakes the sender.
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    let a = run_thread_for(&world, A);
    world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| {
            let origin = WaitOrigin {
                thread: A,
                core: core_of(A),
                args: [0; SVC_ARG_COUNT],
            };
            scheduler.wait_for_reply(a, session, origin)
        })
        .unwrap();
    world.with_tables(|_, scheduler| {
        assert!(!scheduler.has_runnable_thread());
        assert_eq!(scheduler.sync().next_deadline(), None);
        assert!(
            scheduler
                .ipc()
                .queue(world.first.id)
                .unwrap()
                .deferred()
                .next()
                .is_none()
        );
    });
    assert!(!world.dispatcher.is_stalled());
}

#[test]
fn a_run_is_not_stalled_while_a_delivery_runs_a_request_queued_behind_the_deferred_one() {
    let world = World::new();
    let handle = world.session();
    let session = world.object(handle);
    let (a, b) = (run_thread_for(&world, A), run_thread_for(&world, B));
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    let (gate, keeper) = gate();
    world.first.script(Box::new(move |_, _, message| {
        gate.pass();
        message.fill(0xB0);
        Ok(Handled::Replied)
    }));
    assert_eq!(
        world.send(B, handle, &request()).0.outcome,
        Outcome::Waiting
    );
    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    std::thread::scope(|scope| {
        // A thread that is no core delivers A's reply, then runs B's request through the
        // handler, which stops in it.
        let delivering = scope.spawn(|| world.deliver(&world.first));
        keeper.wait();
        world.with_tables(|_, scheduler| {
            // A's request is over and B's is running: nothing is deferred, nothing is
            // runnable and no wait has a timer, but both still wait for their replies.
            let queue = scheduler.ipc().queue(world.first.id).unwrap();
            assert!(queue.deferred().next().is_none());
            assert!(!scheduler.has_runnable_thread());
            assert_eq!(scheduler.sync().next_deadline(), None);
        });
        assert!(!world.dispatcher.is_stalled());
        keeper.open();
        delivering.join().unwrap();
    });
    for thread in [a, b] {
        assert_eq!(
            reply_state(&world, thread, session),
            (ThreadState::Runnable, false)
        );
    }
    assert_eq!(world.tls(A), [0xA0; BUFFER_SIZE]);
    assert_eq!(world.tls(B), [0xB0; BUFFER_SIZE]);
}

// ----- Clearing a server's event, and ending the sender's process ----------------------------

/// Whether the event `event` is an end of is signalled.
fn signalled(world: &World, event: ObjectId) -> bool {
    world.with_tables(|_, scheduler| {
        let readable = match scheduler.objects().get(event).unwrap() {
            Object::Event(event) => event.readable(),
            Object::ReadableEvent(_) => event,
            other => unreachable!("{event:?} is {other:?}, not an event"),
        };
        match scheduler.objects().get(readable).unwrap() {
            Object::ReadableEvent(state) => state.is_signalled(),
            other => unreachable!("{readable:?} is {other:?}"),
        }
    })
}

/// A script that runs `body` on the context, keeps what it returns in `kept`, and replies
/// with zeros.
fn step<T: Send + 'static>(
    kept: &Arc<Mutex<Vec<T>>>,
    body: impl Fn(&mut IpcContext<'_>) -> T + Send + 'static,
) -> Script {
    let kept = Arc::clone(kept);
    Box::new(move |cx, _, message| {
        let value = body(cx);
        lock(&kept).push(value);
        message.fill(0);
        Ok(Handled::Replied)
    })
}

/// A hook that does nothing.
struct Deaf;

impl EventHook for Deaf {
    fn signalled(&self, _: ObjectId) {}
}

#[test]
fn a_handler_clears_an_event_it_made_whether_or_not_it_is_signalled() {
    let world = World::new();
    let handle = world.session();
    let made = Arc::new(Mutex::new(Vec::new()));
    world.first.script(step(&made, |cx| {
        let event = cx.create_event().unwrap();
        cx.signal_event(event).unwrap();
        event
    }));
    world.send(A, handle, &request());
    let event = lock(&made)[0];
    assert!(signalled(&world, event));

    // Signalled: cleared. Not signalled: stays so, and the clear still succeeds.
    let cleared = Arc::new(Mutex::new(Vec::new()));
    world
        .first
        .script(step(&cleared, move |cx| cx.clear_event(event)));
    world.send(A, handle, &request());
    assert!(!signalled(&world, event));
    world.send(A, handle, &request());
    assert!(!signalled(&world, event));
    assert_eq!(*lock(&cleared), [Ok(()), Ok(())]);
}

#[test]
fn a_handler_clears_either_end_of_a_hooked_event_it_made() {
    let world = World::new();
    let handle = world.session();
    let made = Arc::new(Mutex::new(Vec::new()));
    world.first.script(step(&made, |cx| {
        cx.create_hooked_event(Arc::new(Deaf)).unwrap()
    }));
    world.send(A, handle, &request());
    let (writable, readable) = lock(&made)[0];
    for end in [writable, readable] {
        let cleared = Arc::new(Mutex::new(Vec::new()));
        world.first.script(step(&cleared, move |cx| {
            cx.signal_event(readable).unwrap();
            cx.clear_event(end)
        }));
        world.send(A, handle, &request());
        assert_eq!(*lock(&cleared), [Ok(())], "{end:?}");
        assert!(!signalled(&world, readable), "{end:?}");
    }
}

#[test]
fn clearing_an_event_the_server_did_not_make_is_an_invalid_handle_and_changes_nothing() {
    let world = World::new();
    let handle = world.session();

    // An event the second server made and signalled, on a session the first one hands out.
    let connected = Arc::new(Mutex::new(Vec::new()));
    let keep = Arc::clone(&connected);
    world.first.script(Box::new(move |cx, _, message| {
        let (_, handle) = cx.connect(SECOND).unwrap();
        lock(&keep).push(handle);
        name_in_reply(message, &[handle]);
        Ok(Handled::Replied)
    }));
    world.send(A, handle, &request());
    let second = lock(&connected)[0];
    let made = Arc::new(Mutex::new(Vec::new()));
    world.second.script(step(&made, |cx| {
        let event = cx.create_event().unwrap();
        cx.signal_event(event).unwrap();
        event
    }));
    world.send(A, second, &request());
    let theirs = lock(&made)[0];

    // A guest's event, both ends, signalled.
    let (_, regs) = world.call(A, SvcId::CreateEvent, 0, 0);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    let (writable, readable) = (regs.x[1], regs.x[2]);
    let (_, signal) = world.call(A, SvcId::SignalEvent, writable, 0);
    assert_eq!(result(&signal), ResultCode::SUCCESS);
    let guest = [writable, readable].map(|raw| world.object(Handle::from_register(raw)));

    // And an object that is no event at all: the session.
    let session = world.object(handle);
    let ids = [theirs, guest[0], guest[1], session];
    let refused = Arc::new(Mutex::new(Vec::new()));
    world
        .first
        .script(step(&refused, move |cx| ids.map(|id| cx.clear_event(id))));
    world.send(A, handle, &request());
    assert_eq!(*lock(&refused), [[Err(KernelError::InvalidHandle); 4]]);
    assert!(signalled(&world, theirs));
    assert!(signalled(&world, guest[1]));

    // The server that made it may clear it: the check is the running server's.
    let cleared = Arc::new(Mutex::new(Vec::new()));
    world
        .second
        .script(step(&cleared, move |cx| cx.clear_event(theirs)));
    world.send(A, second, &request());
    assert_eq!(*lock(&cleared), [Ok(())]);
    assert!(!signalled(&world, theirs));
}

/// A `take_reply` hook that asks to end the sender's process as the reply is given.
fn ask_to_end(cx: &mut IpcContext<'_>) {
    cx.end_process();
}

/// Replies with every byte `fill`, records an IPC event, and asks to end the process.
fn reply_and_end(fill: u8) -> Script {
    Box::new(move |cx, _, message| {
        message.fill(fill);
        cx.record(ipc_event(COMMAND));
        cx.end_process();
        cx.end_process();
        Ok(Handled::Replied)
    })
}

/// The process `thread` (a thread object) belongs to.
fn process_of(scheduler: &Scheduler, thread: ObjectId) -> ProcessId {
    let Object::Thread(id) = scheduler.objects().get(thread).unwrap() else {
        unreachable!("{thread:?} is not a thread");
    };
    scheduler.thread(id).unwrap().process().unwrap()
}

/// What the process `thread` (a thread object) belongs to holds: its handles, the objects
/// besides each process's own, and its threads that have not ended; and whether it has ended.
fn holdings(world: &World, thread: ObjectId) -> (usize, usize, usize, bool) {
    world.with_tables(|handles, scheduler| {
        let process = process_of(scheduler, thread);
        let record = scheduler.process(process).unwrap();
        let running = record
            .threads()
            .filter(|thread| {
                scheduler.thread(*thread).unwrap().state() != crate::sched::ThreadState::Terminated
            })
            .count();
        (
            handles.len(),
            scheduler.objects_besides_processes(),
            running,
            record.state() == crate::process::ProcessState::Terminated,
        )
    })
}

#[test]
fn a_handler_that_ends_the_process_replies_first_and_then_releases_everything() {
    let (world, thread) = world_running_a_user_thread(A);
    let handle = world.session();
    let session = world.object(handle);
    // The thread and the session, each held once: the session by its handle, the thread by
    // its creator's reference.
    assert_eq!(holdings(&world, thread), (1, 2, 1, false));
    world.first.script(reply_and_end(0xE0));
    let before = world.events().len();
    let (record, regs) = world.send(A, handle, &request());

    assert_eq!(record.outcome, Outcome::ProcessExited);
    // The reply reached the sender, with its result, and the request was recorded: a sender
    // whose thread had already ended would have got neither.
    assert_eq!(world.tls(A), [0xE0; BUFFER_SIZE]);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    assert_eq!(
        threads_and_kinds(&world.events()[before..]),
        [
            (A, ipc_event(COMMAND)),
            send_event(A, handle, ResultCode::SUCCESS),
        ]
    );
    // Then everything went, as `ExitProcess` lets it go: the session's server was told.
    // The one object left is the thread's, which the test's own reference (its creator's,
    // `world_running_a_user_thread`) still holds; letting it go leaves nothing.
    assert_eq!(holdings(&world, thread), (0, 1, 0, true));
    world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| scheduler.release_objects(&[thread]))
        .unwrap();
    world.with_tables(|_, scheduler| assert_eq!(scheduler.objects_besides_processes(), 0));
    assert_eq!(world.first.log().closed, [session]);
    world.with_tables(|_, scheduler| assert!(!scheduler.ipc().is_session(session)));
}

#[test]
fn an_end_asked_by_a_handler_that_defers_is_not_honoured() {
    let (world, thread) = world_running_a_user_thread(A);
    let handle = world.session();
    let session = world.object(handle);
    world.first.script(Box::new(|cx, _, _| {
        cx.end_process();
        Ok(Handled::Deferred)
    }));
    let (record, _) = world.send(A, handle, &request());
    assert_eq!(record.outcome, Outcome::Waiting);
    assert_eq!(holdings(&world, thread), (1, 2, 1, false));

    // Its reply, when it comes, ends nothing either.
    lock(&world.first.ready).push_back((session, Ok(0xD0)));
    assert_eq!(world.deliver(&world.first), []);
    assert_eq!(world.tls(A), [0xD0; BUFFER_SIZE]);
    assert_eq!(holdings(&world, thread), (1, 2, 1, false));
}

#[test]
fn an_end_asked_by_a_request_a_delivery_runs_is_honoured_after_the_replies() {
    let (world, thread) = world_running_a_user_thread(A);
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    // B's request waits behind A's, and asks to end the process when it runs. B's core runs
    // no scheduler thread, so B sends for the tables' own process.
    world.first.script(reply_and_end(0xB0));
    assert_eq!(
        world.send(B, handle, &request()).0.outcome,
        Outcome::Waiting
    );
    let own = world.with_tables(|_, scheduler| process_of(scheduler, thread));

    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    let before = world.events().len();
    let ended = world.deliver(&world.first);
    assert_eq!(
        ended,
        [Ended {
            process: own,
            thread: B,
            core: core_of(B)
        }]
    );
    // Both replies reached their senders and were recorded, A's first, before the end.
    assert_eq!(world.tls(A), [0xA0; BUFFER_SIZE]);
    assert_eq!(world.tls(B), [0xB0; BUFFER_SIZE]);
    assert_eq!(
        threads_and_kinds(&world.events()[before..]),
        [
            send_event(A, handle, ResultCode::SUCCESS),
            (B, ipc_event(COMMAND)),
            send_event(B, handle, ResultCode::SUCCESS),
        ]
    );
    // Then the process let everything go, as `ExitProcess` lets it go.
    assert_eq!(holdings(&world, thread), (0, 1, 0, true));
    assert_eq!(world.first.log().closed, [session]);
    world.with_tables(|_, scheduler| assert!(!scheduler.ipc().is_session(session)));
}

#[test]
fn an_end_asked_while_a_delivery_takes_a_deferred_reply_is_honoured_after_it() {
    let (world, thread) = world_running_a_user_thread(A);
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    assert_eq!(
        world.send(A, handle, &request()).0.outcome,
        Outcome::Waiting
    );
    let own = world.with_tables(|_, scheduler| process_of(scheduler, thread));
    // The handler asks while it gives the reply, as a service that defers `Exit` would.
    lock(&world.first.hooks).take_reply = Some(Box::new(|cx| {
        cx.end_process();
        cx.record(ipc_event(COMMAND));
    }));
    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    let before = world.events().len();
    assert_eq!(
        world.deliver(&world.first),
        [Ended {
            process: own,
            thread: A,
            core: core_of(A)
        }]
    );
    assert_eq!(world.tls(A), [0xA0; BUFFER_SIZE]);
    assert_eq!(
        threads_and_kinds(&world.events()[before..]),
        [
            (A, ipc_event(COMMAND)),
            send_event(A, handle, ResultCode::SUCCESS),
        ]
    );
    assert_eq!(holdings(&world, thread), (0, 1, 0, true));
    assert_eq!(world.first.log().closed, [session]);
}

#[test]
fn a_delivery_ends_the_process_of_the_request_that_asked_and_no_other() {
    let (world, own, other) = world_of_two_processes();
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    assert_eq!(
        world.send(A, handle, &request()).0.outcome,
        Outcome::Waiting
    );
    // B, a thread of the other process, asks to end its process from behind A's request.
    world.first.script(reply_and_end(0xB0));
    assert_eq!(
        world.send(B, handle, &request()).0.outcome,
        Outcome::Waiting
    );
    assert_eq!(process_state(&world, own), (false, 1));
    assert_eq!(process_state(&world, other), (false, 1));

    // The delivery is made on A's core, on behalf of B's process.
    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    assert_eq!(
        world.deliver(&world.first),
        [Ended {
            process: other,
            thread: B,
            core: core_of(B)
        }]
    );
    assert_eq!(world.tls(A), [0xA0; BUFFER_SIZE]);
    assert_eq!(world.tls(B), [0xB0; BUFFER_SIZE]);
    assert_eq!(process_state(&world, other), (true, 0));
    assert_eq!(process_state(&world, own), (false, 1));
}

#[test]
fn a_delivery_ends_a_process_two_requests_asked_for_once_naming_the_first() {
    let (world, thread) = world_running_a_user_thread(A);
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    // B's request and then C's wait behind A's, and both ask to end the process.
    world.first.script(reply_and_end(0xBC));
    for sender in [B, C] {
        assert_eq!(
            world.send(sender, handle, &request()).0.outcome,
            Outcome::Waiting
        );
    }
    let own = world.with_tables(|_, scheduler| process_of(scheduler, thread));
    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    assert_eq!(
        world.deliver(&world.first),
        [Ended {
            process: own,
            thread: B,
            core: core_of(B)
        }]
    );
    assert_eq!(world.tls(B), [0xBC; BUFFER_SIZE]);
    assert_eq!(world.tls(C), [0xBC; BUFFER_SIZE]);
    assert_eq!(holdings(&world, thread), (0, 1, 0, true));
}

#[test]
fn an_event_noted_for_no_server_is_let_go_again() {
    let world = World::new();
    let nowhere = ServerId::from_parts(9, 9);
    let noted = world.first.id;
    world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| {
            let before = scheduler.objects().len();
            let (writable, readable) = scheduler.create_event().unwrap();
            assert_eq!(
                note_server_events(scheduler, nowhere, &[writable, readable]),
                Err(SvcError::internal(&IpcError::UnknownServer(nowhere)))
            );
            assert_eq!(scheduler.objects().len(), before);
            let readable = scheduler.create_readable_event().unwrap();
            note_server_events(scheduler, noted, &[readable]).unwrap();
            assert!(scheduler.ipc().is_server_event(noted, readable));
            assert_eq!(scheduler.objects().len(), before + 1);
        });
}

/// Runs `script` for one request from A, whose process runs a user thread, and checks that the
/// end the script asked for was honoured although the request ended without a reply: the
/// dispatch ends in [`Outcome::ProcessExited`], A's `x0` holds [`KernelError::SessionClosed`],
/// the session is gone, and the process let everything go. `told` is whether the handler is
/// told the session closed: a failed handler is, one that closed it itself is not.
fn assert_an_end_is_honoured_on_a_closed_session(script: Script, told: bool) {
    let (world, thread) = world_running_a_user_thread(A);
    let handle = world.session();
    let session = world.object(handle);
    assert_eq!(holdings(&world, thread), (1, 2, 1, false));
    world.first.script(script);
    let (record, regs) = world.send(A, handle, &request());
    assert_eq!(record.outcome, Outcome::ProcessExited);
    assert_eq!(result(&regs), KernelError::SessionClosed.code());
    let closed: &[ObjectId] = if told { &[session] } else { &[] };
    assert_eq!(world.first.log().closed, closed);
    assert_eq!(holdings(&world, thread), (0, 1, 0, true));
    world.with_tables(|_, scheduler| assert!(!scheduler.ipc().is_session(session)));
}

#[test]
fn an_end_asked_by_a_handler_that_then_fails_is_honoured() {
    assert_an_end_is_honoured_on_a_closed_session(
        Box::new(|cx, _, _| {
            cx.end_process();
            Err(HandlerError("broken".to_owned()))
        }),
        true,
    );
}

#[test]
fn an_end_asked_by_a_handler_that_closes_the_session_is_honoured() {
    assert_an_end_is_honoured_on_a_closed_session(
        Box::new(|cx, _, _| {
            cx.end_process();
            Ok(Handled::Closed)
        }),
        false,
    );
}

#[test]
fn an_end_asked_by_a_request_the_dispatch_putting_the_handler_back_runs_is_honoured() {
    let world = World::new();
    let deferred = world.session();
    let other = world.session();
    let (deferred_object, other_object) = (world.object(deferred), world.object(other));
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    assert_eq!(
        world.send(A, deferred, &request()).0.outcome,
        Outcome::Waiting
    );
    // B's request waits behind A's deferred one and asks to end the process when it runs; C's,
    // on another session, stops in its handler until the test lets it through.
    let (gate, keeper) = gate();
    world.first.script(Box::new(move |cx, session, message| {
        if session == other_object {
            gate.pass();
            message.fill(0xC0);
        } else {
            message.fill(0xB0);
            cx.end_process();
        }
        Ok(Handled::Replied)
    }));
    assert_eq!(
        world.send(B, deferred, &request()).0.outcome,
        Outcome::Waiting
    );
    std::thread::scope(|scope| {
        let sending = scope.spawn(|| world.send(C, other, &request()));
        keeper.wait();
        // A's reply is ready while C's core has the handler out: the delivery is owed, and
        // C's dispatch makes it when it puts the handler back, then runs B's request.
        let a_buffer = world.tls(A);
        lock(&world.first.ready).push_back((deferred_object, Ok(0xA0)));
        world.deliver(&world.first);
        assert_eq!(world.tls(A), a_buffer, "the delivery waits for the handler");
        keeper.open();
        let (record, regs) = sending.join().unwrap();
        // That dispatch is a guest's, so the end B's handler asked for is honoured there.
        assert_eq!(record.outcome, Outcome::ProcessExited);
        assert_eq!(result(&regs), ResultCode::SUCCESS);
    });
    assert_eq!(world.tls(A), [0xA0; BUFFER_SIZE]);
    assert_eq!(world.tls(B), [0xB0; BUFFER_SIZE]);
    assert_eq!(world.tls(C), [0xC0; BUFFER_SIZE]);
    world.with_tables(|handles, scheduler| {
        assert!(handles.is_empty());
        assert!(!scheduler.ipc().is_session(deferred_object));
        assert!(!scheduler.ipc().is_session(other_object));
    });
}

#[test]
fn a_dispatch_that_ends_another_process_returns_its_own_result() {
    let (world, own, other) = world_of_two_processes();
    let deferred = world.session();
    let unrelated = world.session();
    let (deferred_object, unrelated_object) = (world.object(deferred), world.object(unrelated));
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    assert_eq!(
        world.send(A, deferred, &request()).0.outcome,
        Outcome::Waiting
    );
    // B, a thread of the other process, asks to end it from behind A's deferred request; C,
    // whose core runs no thread of a process and so sends for the tables' own, stops in its
    // handler on another session until the test lets it through.
    let (gate, keeper) = gate();
    world.first.script(Box::new(move |cx, session, message| {
        if session == unrelated_object {
            gate.pass();
            message.fill(0xC0);
        } else {
            message.fill(0xB0);
            cx.end_process();
        }
        Ok(Handled::Replied)
    }));
    assert_eq!(
        world.send(B, deferred, &request()).0.outcome,
        Outcome::Waiting
    );
    std::thread::scope(|scope| {
        let sending = scope.spawn(|| world.send(C, unrelated, &request()));
        keeper.wait();
        // Owed while C's core has the handler out: this delivery ends nothing itself.
        lock(&world.first.ready).push_back((deferred_object, Ok(0xA0)));
        assert_eq!(world.deliver(&world.first), []);
        keeper.open();
        let (record, regs) = sending.join().unwrap();
        // C's dispatch ran B's request and ended B's process, not its own.
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!(result(&regs), ResultCode::SUCCESS);
    });
    assert_eq!(world.tls(A), [0xA0; BUFFER_SIZE]);
    assert_eq!(world.tls(B), [0xB0; BUFFER_SIZE]);
    assert_eq!(world.tls(C), [0xC0; BUFFER_SIZE]);
    assert_eq!(process_state(&world, other), (true, 0));
    assert_eq!(process_state(&world, own), (false, 1));
}

#[test]
fn ending_a_process_that_has_ended_already_fails_and_reports_nothing() {
    let (world, thread) = world_running_a_user_thread(A);
    let process = world.with_tables(|_, scheduler| process_of(scheduler, thread));
    let (record, _) = world.call(A, SvcId::ExitProcess, 0, 0);
    assert_eq!(record.outcome, Outcome::ProcessExited);
    drop(world.dispatcher.take_woken());
    let before = world.events().len();
    let mut locked = world
        .dispatcher
        .lock::<(&mut HandleTable, &mut Scheduler)>();
    assert!(!terminate_asked(
        &mut locked,
        process,
        &world.recorder,
        GuestTicks(0)
    ));
    drop(locked);
    assert_eq!(world.events().len(), before);
    assert_eq!(world.dispatcher.take_woken(), []);
}

#[test]
fn a_delivery_does_not_report_an_end_asked_for_a_process_that_ended_meanwhile() {
    // A's request is deferred and still in flight, holding its session, when the process ends
    // by `ExitProcess` (from C, a host thread). The handler then asks to end the process as it
    // gives the reply: there is nothing left to end, and nothing is reported.
    let (world, thread) = world_running_a_user_thread(A);
    let handle = world.session();
    let session = world.object(handle);
    world
        .first
        .script(Box::new(|_, _, _| Ok(Handled::Deferred)));
    world.send(A, handle, &request());
    run_thread_for(&world, C);
    let (record, _) = world.call(C, SvcId::ExitProcess, 0, 0);
    assert_eq!(record.outcome, Outcome::ProcessExited);
    assert!(holdings(&world, thread).3);
    let asked = Arc::new(Mutex::new(0));
    let counted = Arc::clone(&asked);
    lock(&world.first.hooks).take_reply = Some(Box::new(move |cx| {
        cx.end_process();
        *lock(&counted) += 1;
    }));
    lock(&world.first.ready).push_back((session, Ok(0xA0)));
    assert_eq!(world.deliver(&world.first), []);
    // The handler was asked for the reply, so the end was asked for.
    assert_eq!(*lock(&asked), 1);
}

/// A world whose process runs a user thread on A's core, with a thread of no process on B's
/// core (a host thread, which outlives the process) waiting for that user thread to end
/// (`WaitSynchronization` on a handle to it, with no timeout).
fn world_with_a_host_thread_waiting_for_the_process() -> World {
    let (world, thread) = world_running_a_user_thread(A);
    run_thread_for(&world, B);
    let handle = world
        .dispatcher
        .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
            handles.add(scheduler.objects_mut(), thread).unwrap()
        });
    world
        .memory
        .write(NAMES, &handle.raw().to_le_bytes())
        .unwrap();
    let mut regs = GuestRegs::default();
    regs.x[1] = NAMES;
    regs.x[2] = 1;
    // No timeout: -1.
    regs.x[3] = u64::MAX;
    regs.tpidrro = TLS_B;
    let record = world.dispatcher.dispatch(
        SvcId::WaitSynchronization.number(),
        &mut regs,
        &mut world.caller(B),
    );
    assert_eq!(record.outcome, Outcome::Waiting);
    world
}

/// What `end` (a dispatch that ends the process) leaves B, the host thread waiting for the
/// process's user thread, as soon as it returns: B's events, and the threads reported woken.
fn what_an_end_leaves_the_waiter(end: impl FnOnce(&World) -> Outcome) -> (Vec<Event>, Vec<Woken>) {
    let world = world_with_a_host_thread_waiting_for_the_process();
    let before = world.events().len();
    assert_eq!(end(&world), Outcome::ProcessExited);
    let events = world.events()[before..]
        .iter()
        .filter(|event| event.thread == B)
        .copied()
        .collect();
    (events, world.dispatcher.take_woken())
}

#[test]
fn a_handler_ending_the_process_wakes_its_threads_waiters_as_exit_process_does() {
    let by_handler = what_an_end_leaves_the_waiter(|world| {
        let handle = world.session();
        world.first.script(reply_and_end(0xE0));
        world.send(A, handle, &request()).0.outcome
    });
    let by_exit =
        what_an_end_leaves_the_waiter(|world| world.call(A, SvcId::ExitProcess, 0, 0).0.outcome);
    // The waiter's wait ended as the thread did, and is recorded and reported in the call that
    // ended the process, not at some later call.
    assert_eq!(
        by_handler.1,
        [Woken {
            thread: B,
            core: core_of(B),
            result: ResultCode::SUCCESS
        }]
    );
    assert!(
        matches!(
            by_handler.0.as_slice(),
            [Event { kind: EventKind::Svc { id, result, .. }, .. }]
                if *id == u32::from(SvcId::WaitSynchronization.number()) && result.0 == 0
        ),
        "{:?}",
        by_handler.0
    );
    assert_eq!(
        by_handler
            .0
            .iter()
            .map(|event| (event.core, event.kind))
            .collect::<Vec<_>>(),
        by_exit
            .0
            .iter()
            .map(|event| (event.core, event.kind))
            .collect::<Vec<_>>()
    );
    assert_eq!(by_handler.1, by_exit.1);
}

// ----- Reading the handlers back ---------------------------------------------------------------

/// A handler as `inspect_handlers` showed it: its server's number and its log.
type Shown = (ServiceId, Arc<Mutex<Log>>);

/// Every handler `inspect_handlers` shows, and how many it did not.
fn shown(world: &World) -> (Vec<Shown>, usize) {
    let mut shown = Vec::new();
    let out = world.dispatcher.inspect_handlers(|service, handler| {
        let handler: &dyn core::any::Any = handler;
        let scripted = handler.downcast_ref::<Scripted>().unwrap();
        shown.push((service, Arc::clone(&scripted.log)));
    });
    (shown, out)
}

#[test]
fn every_handler_in_its_slot_is_shown_with_the_number_its_server_is_known_by() {
    let world = World::new();
    let (handlers, out) = shown(&world);
    assert_eq!(out, 0);
    // In the order of the kernel's server ids, each with its own server's number.
    assert!(world.first.id < world.second.id);
    let services: Vec<ServiceId> = handlers.iter().map(|(service, _)| *service).collect();
    assert_eq!(services, [FIRST, SECOND]);
    assert!(Arc::ptr_eq(&handlers[0].1, &world.first.log));
    assert!(Arc::ptr_eq(&handlers[1].1, &world.second.log));
}

#[test]
fn a_handler_out_of_its_slot_is_not_shown_but_counted() {
    let world = World::new();
    let handle = world.session();
    let (gate, keeper) = gate();
    world.first.script(Box::new(move |_, _, message| {
        gate.pass();
        message.fill(0xA0);
        Ok(Handled::Replied)
    }));
    let world = &world;
    std::thread::scope(|scope| {
        // A's request has the first server's handler out of its slot until the gate opens.
        let sending = scope.spawn(move || world.send(A, handle, &request()));
        keeper.wait();
        let (handlers, out) = shown(world);
        assert_eq!(out, 1);
        assert_eq!(handlers.len(), 1);
        assert_eq!(handlers[0].0, SECOND);
        assert!(Arc::ptr_eq(&handlers[0].1, &world.second.log));
        keeper.open();
        let (_, regs) = sending.join().unwrap();
        assert_eq!(result(&regs), ResultCode::SUCCESS);
    });
    // Back in its slot, it is shown again.
    let (handlers, out) = shown(world);
    assert_eq!((handlers.len(), out), (2, 0));
}
