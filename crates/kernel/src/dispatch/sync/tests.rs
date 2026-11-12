//! The synchronisation arms of the dispatcher, driven with guest registers, flat guest memory,
//! the fixture's address space and two guest threads running on two cores: each call's errors
//! in Eden's order, the immediate returns, a wait ended by another thread's call, a host
//! thread's signal, a timer and a server's handler, and where each one's event is recorded.

use std::sync::{Arc, Mutex, PoisonError};

use starquake_cpu::GuestRegs;
use starquake_ipc::{BUFFER_SIZE, CommandType, RequestBuilder};
use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE};
use starquake_trace::{
    EnvError, Event, EventKind, FixedClock, GuestTicks, HostNanos, LiveEnv, Recorder, ReplayEnv,
    SVC_ARG_COUNT, ServiceId, ThreadId, VecSink,
};

use super::*;
use crate::dispatch::{Caller, Record, Tables, Woken};
use crate::handle::{HandleTable, MAX_TABLE_SIZE, ObjectRefs};
use crate::id::{ProcessId, SlabId, ThreadId as KernelThreadId};
use crate::ipc::{Handled, HandlerError, IpcContext, SessionHandler};
use crate::memory::fixture::{self, CODE, CODE_SIZE, DATA, DATA_PAGES};
use crate::object::Object;
use crate::result::{KernelError, ResultCode};
use crate::sched::{CoreId, CoreSignals, Priority};
use crate::svc::DebugSink;
use crate::sync::{ARGUMENT_HANDLE_COUNT_MAX, TIMEOUT_SLACK_NS};
use crate::tables::Processes;
use crate::time::CounterRate;

/// The two guest threads, as the event stream names them, and the cores they run on.
const A: ThreadId = ThreadId(0xA1);
const B: ThreadId = ThreadId(0xB2);
const CORE_A: u8 = 0;
const CORE_B: u8 = 1;
/// Where the tests put handle arrays: the first page of the fixture's data.
const ARRAY: u64 = DATA;
/// B's message buffer, for the request a server's handler signals from.
const TLS_B: u64 = DATA + PAGE_SIZE;
/// The end of the data: the guest may read nothing after it, though the flat memory goes on.
const DATA_END: u64 = DATA + DATA_PAGES * PAGE_SIZE;
/// The clock: its first reading, and how far it moves at each reading.
const CLOCK_START_NS: u64 = 1_000_000;
const CLOCK_STEP_NS: u64 = 1_000;
/// The register a test fills with garbage before a call, to see that the call writes it.
const STALE: u64 = 0x5555_5555_5555_5555;
/// The index -1 as `x1` holds it: four bytes, zero-extended.
const MINUS_ONE: u64 = 0xFFFF_FFFF;
/// Eden's application limit on events (`LimitableResource::EventCountMax`, 900, in
/// `eden/src/core/hle/kernel/k_resource_limit.cpp` @ cf654052): what `CreateEvent` would stop
/// at there.
const EDEN_EVENT_COUNT_MAX: usize = 900;

struct NoDebug;

impl DebugSink for NoDebug {
    fn output(&mut self, _: GuestTicks, _: &[u8]) {}
}

struct NoSignals;

impl CoreSignals for NoSignals {
    fn interrupt(&mut self, _: CoreId) {}

    fn wake(&mut self, _: CoreId) {}
}

fn core(index: u8) -> CoreId {
    CoreId::new(index).unwrap()
}

fn word(error: KernelError) -> u64 {
    u64::from(error.code().raw())
}

/// The dispatcher, guest memory, clock and event stream of one test, with thread A running on
/// core 0 and thread B on core 1.
struct World {
    dispatcher: Dispatcher,
    memory: FlatMemory,
    env: LiveEnv,
    recorder: Recorder,
    events: Arc<VecSink>,
    /// The scheduler's names for A and B.
    a: KernelThreadId,
    b: KernelThreadId,
}

impl World {
    fn new() -> Self {
        let mut tables = Tables {
            handles: HandleTable::new(i32::from(MAX_TABLE_SIZE)).unwrap(),
            address_space: fixture::loaded().0,
            scheduler: Scheduler::default(),
            processes: Processes::new(CounterRate::SWITCH_2),
        };
        let mut start = |on: u8| {
            let scheduler = &mut tables.scheduler;
            let thread = scheduler
                .create_thread(Priority::new(30).unwrap(), core(on))
                .unwrap();
            scheduler.start(thread).unwrap();
            scheduler
                .update(GuestTicks(0), None, &mut NoSignals)
                .unwrap();
            assert_eq!(
                scheduler.schedule(core(on), &mut NoSignals),
                Ok(Some(thread))
            );
            thread
        };
        let a = start(CORE_A);
        let b = start(CORE_B);
        let recorder = Recorder::new();
        let events = Arc::new(VecSink::new());
        recorder.add_sink(Arc::clone(&events));
        // A page of flat memory past the data that the address space does not map.
        let size = usize::try_from(CODE_SIZE + PAGE_SIZE).unwrap();
        Self {
            dispatcher: Dispatcher::new(tables),
            memory: FlatMemory::new(CODE, size),
            env: LiveEnv::new(FixedClock::new(CLOCK_START_NS, CLOCK_STEP_NS), 0),
            recorder,
            events,
            a,
            b,
        }
    }

    fn caller(&self, thread: ThreadId) -> Caller<'_> {
        let debug: &mut NoDebug = Box::leak(Box::new(NoDebug));
        Caller {
            core: if thread == A { CORE_A } else { CORE_B },
            thread,
            env: &self.env,
            memory: &self.memory,
            debug,
            recorder: &self.recorder,
        }
    }

    /// `id` from `thread` with `x0`-`x3` as given and every other register stale.
    fn call(&self, thread: ThreadId, id: SvcId, x: [u64; 4]) -> (Record, GuestRegs) {
        let mut regs = GuestRegs {
            x: [STALE; 31],
            tpidrro: TLS_B,
            ..GuestRegs::default()
        };
        regs.x[..4].copy_from_slice(&x);
        let record = self
            .dispatcher
            .dispatch(id.number(), &mut regs, &mut self.caller(thread));
        (record, regs)
    }

    /// `CreateEvent` from A: the writable and readable handles.
    fn create_event(&self) -> (Handle, Handle) {
        let (record, regs) = self.call(A, SvcId::CreateEvent, [0; 4]);
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!(regs.x[0], 0);
        (
            Handle::from_register(regs.x[1]),
            Handle::from_register(regs.x[2]),
        )
    }

    /// Writes `handles` at [`ARRAY`] and makes `WaitSynchronization` from `thread`.
    fn wait(&self, thread: ThreadId, handles: &[Handle], timeout_ns: i64) -> (Record, GuestRegs) {
        let bytes: Vec<u8> = handles
            .iter()
            .flat_map(|handle| handle.raw().to_le_bytes())
            .collect();
        self.memory.write(ARRAY, &bytes).unwrap();
        let count = u64::try_from(handles.len()).unwrap();
        self.call(
            thread,
            SvcId::WaitSynchronization,
            [STALE, ARRAY, count, timeout_ns.cast_unsigned()],
        )
    }

    /// The object a handle of the process names.
    fn object(&self, handle: Handle) -> ObjectId {
        self.dispatcher
            .with::<(&HandleTable,), _>(|(handles,)| handles.get(handle))
            .unwrap()
    }

    fn refs(&self, object: ObjectId) -> u32 {
        self.dispatcher
            .with::<(&Scheduler,), _>(|(scheduler,)| scheduler.objects().refs(object))
    }

    fn handles(&self) -> usize {
        self.dispatcher
            .with::<(&HandleTable,), _>(|(handles,)| handles.len())
    }

    fn objects(&self) -> usize {
        self.dispatcher
            .with::<(&Scheduler,), _>(|(scheduler,)| scheduler.objects().len())
    }

    /// A handle to thread `thread`'s object.
    fn thread_handle(&self, thread: KernelThreadId) -> Handle {
        self.dispatcher
            .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
                let objects = scheduler.objects_mut();
                let object = objects.create(Object::Thread(thread)).unwrap();
                let handle = handles.add(objects, object).unwrap();
                objects.close(object).unwrap();
                handle
            })
    }

    fn state(&self, thread: KernelThreadId) -> crate::sched::ThreadState {
        self.dispatcher
            .with::<(&Scheduler,), _>(|(scheduler,)| scheduler.thread(thread).unwrap().state())
    }

    fn context(&self, thread: KernelThreadId) -> GuestRegs {
        self.dispatcher.with::<(&Scheduler,), _>(|(scheduler,)| {
            scheduler.thread(thread).unwrap().context().clone()
        })
    }

    fn events(&self) -> Vec<Event> {
        self.events.events()
    }
}

fn result(regs: &GuestRegs) -> u64 {
    regs.x[0]
}

/// The `WaitSynchronization` event of `thread` on `core`, as the event stream carries it.
fn wait_event(array: u64, count: u64, timeout_ns: i64, result: ResultCode) -> EventKind {
    let mut args = [0; SVC_ARG_COUNT];
    args[1] = array;
    args[2] = count;
    args[3] = timeout_ns.cast_unsigned();
    EventKind::Svc {
        id: u32::from(SvcId::WaitSynchronization.number()),
        args,
        result: starquake_trace::ResultCode(result.raw()),
    }
}

/// `(thread, core, kind)` of each event.
fn who_and_what(events: &[Event]) -> Vec<(ThreadId, u8, EventKind)> {
    events
        .iter()
        .map(|event| (event.thread, event.core, event.kind))
        .collect()
}

// ----- CreateEvent ----------------------------------------------------------------------------

#[test]
fn create_event_returns_both_handles_and_each_holds_its_ends_only_reference() {
    let world = World::new();
    let (writable, readable) = world.create_event();
    assert_ne!(writable, readable);
    let (w, r) = (world.object(writable), world.object(readable));
    assert!(matches!(
        world.dispatcher.with::<(&Scheduler,), _>(|(s,)| s.objects().get(w)),
        Ok(Object::Event(event)) if event.readable() == r
    ));
    assert_eq!((world.refs(w), world.refs(r)), (1, 1));
    assert_eq!((world.handles(), world.objects()), (2, 2));
}

#[test]
fn creating_many_events_never_reaches_a_limit() {
    // Divergence: Eden refuses with `ResultLimitReached` past its resource limit. Here events
    // run out only with the handle table. Keep each event's readable end, closing its writable
    // one, so every live event holds one handle.
    let world = World::new();
    let reserved = world.handles();
    let room = usize::from(MAX_TABLE_SIZE) - reserved;
    assert!(room - 1 > EDEN_EVENT_COUNT_MAX);
    for _ in 0..room - 1 {
        let (writable, _) = world.create_event();
        let (_, regs) = world.call(A, SvcId::CloseHandle, [u64::from(writable.raw()), 0, 0, 0]);
        assert_eq!(result(&regs), 0);
    }
    // One handle left: the readable end can't be added, and the writable end's goes again.
    let objects = world.objects();
    let (record, regs) = world.call(A, SvcId::CreateEvent, [0; 4]);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(regs.x[..3], [word(KernelError::OutOfHandles), 0, 0]);
    assert_eq!(
        (world.handles(), world.objects()),
        (usize::from(MAX_TABLE_SIZE) - 1, objects)
    );
}

// ----- SignalEvent, ClearEvent and ResetSignal -----------------------------------------------

#[test]
fn each_event_call_takes_the_ends_eden_takes_and_refuses_the_rest() {
    let world = World::new();
    let (writable, readable) = world.create_event();
    let thread = world.thread_handle(world.a);
    let invalid = word(KernelError::InvalidHandle);
    let state = word(KernelError::InvalidState);
    let reset_unsignalled = |handle: Handle| {
        world
            .call(A, SvcId::ResetSignal, [u64::from(handle.raw()), 0, 0, 0])
            .1
            .x[0]
    };
    // (handle, SignalEvent, ClearEvent, ResetSignal on an unsignalled event), the results.
    let cases = [
        (writable, 0, 0, invalid),
        (readable, invalid, 0, state),
        (thread, invalid, invalid, invalid),
        (Handle::CURRENT_THREAD, invalid, invalid, invalid),
        // `GetObject<KProcess>` takes the process pseudo-handle: a process that is never
        // signalled cannot be reset.
        (Handle::CURRENT_PROCESS, invalid, invalid, state),
        (Handle::INVALID, invalid, invalid, invalid),
    ];
    for (handle, signal, clear, reset) in cases {
        // Each case starts unsignalled; ClearEvent sees to that.
        world.call(A, SvcId::ClearEvent, [u64::from(readable.raw()), 0, 0, 0]);
        assert_eq!(reset_unsignalled(handle), reset, "reset {handle:?}");
        let x0 = u64::from(handle.raw()) | 0xFFFF_FFFF_0000_0000;
        let (_, regs) = world.call(A, SvcId::SignalEvent, [x0, 0, 0, 0]);
        assert_eq!(result(&regs), signal, "signal {handle:?}");
        let (_, regs) = world.call(A, SvcId::ClearEvent, [x0, 0, 0, 0]);
        assert_eq!(result(&regs), clear, "clear {handle:?}");
    }
    // A signalled event resets once.
    world.call(A, SvcId::SignalEvent, [u64::from(writable.raw()), 0, 0, 0]);
    assert_eq!(reset_unsignalled(readable), 0);
    assert_eq!(reset_unsignalled(readable), state);
    // A process object by handle, like the pseudo-handle.
    let process = world
        .dispatcher
        .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
            let objects = scheduler.objects_mut();
            let object = objects
                .create(Object::Process(ProcessId::from_parts(0, 1)))
                .unwrap();
            handles.add(objects, object).unwrap()
        });
    assert_eq!(reset_unsignalled(process), state);
}

// ----- WaitSynchronization: the checks before a wait ------------------------------------------

#[test]
fn the_handle_count_is_checked_first_at_both_edges() {
    let world = World::new();
    let (writable, readable) = world.create_event();
    world.call(A, SvcId::SignalEvent, [u64::from(writable.raw()), 0, 0, 0]);
    let max = u64::try_from(ARGUMENT_HANDLE_COUNT_MAX).unwrap();
    // The array is unreadable too: the count is reported first.
    for count in [
        max + 1,
        u64::from(u32::MAX),
        u64::from((-1_i32).cast_unsigned()),
    ] {
        let (_, regs) = world.call(A, SvcId::WaitSynchronization, [STALE, DATA_END, count, 0]);
        assert_eq!(
            regs.x[..2],
            [word(KernelError::OutOfRange), 0],
            "{count:#x}"
        );
    }
    // Exactly the maximum is taken: the same signalled handle every time.
    let all = vec![readable; ARGUMENT_HANDLE_COUNT_MAX];
    let (_, regs) = world.wait(A, &all, -1);
    assert_eq!(regs.x[..2], [0, 0]);
    // Only the count's low four bytes are the argument.
    let (_, regs) = world.call(
        A,
        SvcId::WaitSynchronization,
        [STALE, ARRAY, 0xFFFF_FFFF_0000_0001, 0],
    );
    assert_eq!(regs.x[..2], [0, 0]);
}

#[test]
fn the_array_must_lie_wholly_in_memory_the_guest_may_read() {
    let world = World::new();
    let (writable, readable) = world.create_event();
    world.call(A, SvcId::SignalEvent, [u64::from(writable.raw()), 0, 0, 0]);
    let bytes = readable.raw().to_le_bytes();
    let size = u64::try_from(sync::HANDLE_SIZE).unwrap();
    // Two handles whose second ends exactly at the end of the data: read.
    let last = DATA_END - 2 * size;
    world.memory.write(last, &bytes).unwrap();
    world.memory.write(last + size, &bytes).unwrap();
    let (_, regs) = world.call(A, SvcId::WaitSynchronization, [STALE, last, 2, 0]);
    assert_eq!(regs.x[..2], [0, 0]);
    // One byte further: the second handle's last byte is past it.
    let (_, regs) = world.call(A, SvcId::WaitSynchronization, [STALE, last + 1, 2, 0]);
    assert_eq!(regs.x[..2], [word(KernelError::InvalidPointer), 0]);
    // An address whose sum with the array's size overflows.
    let (_, regs) = world.call(A, SvcId::WaitSynchronization, [STALE, u64::MAX - 3, 2, 0]);
    assert_eq!(regs.x[..2], [word(KernelError::InvalidPointer), 0]);
    // An unreadable array with an invalid handle in its readable part: the pointer is
    // reported, before any handle is looked at.
    world
        .memory
        .write(DATA_END - 2, &Handle::INVALID.raw().to_le_bytes())
        .unwrap();
    let (_, regs) = world.call(A, SvcId::WaitSynchronization, [STALE, DATA_END - 2, 1, 0]);
    assert_eq!(regs.x[..2], [word(KernelError::InvalidPointer), 0]);
    // No handles: the address is not read at all.
    let (_, regs) = world.call(A, SvcId::WaitSynchronization, [STALE, u64::MAX, 0, 0]);
    assert_eq!(regs.x[..2], [word(KernelError::TimedOut), 0]);
}

#[test]
fn a_handle_array_the_guest_may_not_read_is_an_invalid_pointer() {
    // Divergence: the flat memory past the data reads fine (as Eden's `ReadBlock` would for a
    // mapped page), but the address space gives the guest no access there.
    let world = World::new();
    let (_, readable) = world.create_event();
    world
        .memory
        .write(DATA_END, &readable.raw().to_le_bytes())
        .unwrap();
    let mut probe = [0; 4];
    assert!(world.memory.read(DATA_END, &mut probe).is_ok());
    let (_, regs) = world.call(A, SvcId::WaitSynchronization, [STALE, DATA_END, 1, 0]);
    assert_eq!(regs.x[..2], [word(KernelError::InvalidPointer), 0]);
}

#[test]
fn every_handle_must_name_a_synchronisation_object_and_none_stays_referenced() {
    let world = World::new();
    let (writable, readable) = world.create_event();
    let r = world.object(readable);
    let thread = world.thread_handle(world.b);
    for refused in [
        writable,
        Handle::CURRENT_THREAD,
        Handle::CURRENT_PROCESS,
        Handle::INVALID,
    ] {
        // The good handles before the bad one were opened, then closed again. A zero timeout
        // as well: the handles are checked first.
        let (_, regs) = world.wait(A, &[readable, thread, refused], 0);
        assert_eq!(
            regs.x[..2],
            [word(KernelError::InvalidHandle), 0],
            "{refused:?}"
        );
        assert_eq!(world.refs(r), 1);
    }
    // A thread is a synchronisation object.
    let (_, regs) = world.wait(A, &[thread, readable], 0);
    assert_eq!(regs.x[..2], [word(KernelError::TimedOut), 0]);
}

// ----- WaitSynchronization: returning at once -------------------------------------------------

#[test]
fn a_signalled_object_returns_its_index_at_once_and_is_recorded_then() {
    let world = World::new();
    let (_, first) = world.create_event();
    let (second_writable, second) = world.create_event();
    world.call(
        A,
        SvcId::SignalEvent,
        [u64::from(second_writable.raw()), 0, 0, 0],
    );
    let before = world.events().len();
    let (record, regs) = world.wait(A, &[first, second], -1);
    assert_eq!(record.outcome, Outcome::Completed);
    // x2 and x3 are inputs; nothing past x1 is written.
    assert_eq!(regs.x[..5], [0, 1, 2, u64::MAX, STALE]);
    let events = world.events();
    assert_eq!(
        who_and_what(&events[before..]),
        [(A, CORE_A, wait_event(ARRAY, 2, -1, ResultCode::SUCCESS))]
    );
    assert_eq!(world.dispatcher.take_woken(), []);
}

#[test]
fn a_zero_timeout_and_a_pending_cancel_return_index_zero_not_minus_one() {
    // `Wait`'s early exits return before `*out_index` is assigned: the wrapper's zero.
    let world = World::new();
    let (_, readable) = world.create_event();
    let (_, regs) = world.wait(A, &[readable], 0);
    assert_eq!(regs.x[..2], [word(KernelError::TimedOut), 0]);
    let (_, regs) = world.call(
        A,
        SvcId::CancelSynchronization,
        [u64::from(Handle::CURRENT_THREAD.raw()), 0, 0, 0],
    );
    assert_eq!(result(&regs), 0);
    let (_, regs) = world.wait(A, &[readable], -1);
    assert_eq!(regs.x[..2], [word(KernelError::Cancelled), 0]);
}

// ----- WaitSynchronization: waiting ------------------------------------------------------------

#[test]
fn a_wait_writes_nothing_and_is_recorded_when_another_threads_signal_ends_it() {
    let world = World::new();
    let (writable, readable) = world.create_event();
    let before = world.events().len();
    let (record, regs) = world.wait(A, &[readable], -1);
    assert_eq!(record.outcome, Outcome::Waiting);
    assert_eq!(regs.x[..4], [STALE, ARRAY, 1, u64::MAX]);
    assert_eq!(world.events().len(), before);
    assert_eq!(world.state(world.a), crate::sched::ThreadState::Waiting);
    // Its registers are its saved context.
    assert_eq!(world.context(world.a), regs);

    let (record, regs) = world.call(B, SvcId::SignalEvent, [u64::from(writable.raw()), 0, 0, 0]);
    assert_eq!((record.outcome, result(&regs)), (Outcome::Completed, 0));
    let events = world.events();
    let mut signal_args = [0; SVC_ARG_COUNT];
    signal_args[0] = u64::from(writable.raw());
    assert_eq!(
        who_and_what(&events[before..]),
        [
            (
                B,
                CORE_B,
                EventKind::Svc {
                    id: u32::from(SvcId::SignalEvent.number()),
                    args: signal_args,
                    result: starquake_trace::ResultCode(0),
                }
            ),
            (A, CORE_A, wait_event(ARRAY, 1, -1, ResultCode::SUCCESS)),
        ]
    );
    // Both stamped with the signal's clock reading.
    assert_eq!(events[before].time, events[before + 1].time);
    assert_eq!(
        world.dispatcher.take_woken(),
        [Woken {
            thread: A,
            core: CORE_A,
            result: ResultCode::SUCCESS,
        }]
    );
    let context = world.context(world.a);
    assert_eq!(context.x[..3], [0, 0, 1]);
    assert_eq!(world.state(world.a), crate::sched::ThreadState::Runnable);
}

#[test]
fn cancel_synchronization_ends_another_threads_wait_with_index_minus_one() {
    let world = World::new();
    let (_, readable) = world.create_event();
    let a = world.thread_handle(world.a);
    world.wait(A, &[readable], -1);
    let (_, regs) = world.call(
        B,
        SvcId::CancelSynchronization,
        [u64::from(a.raw()), 0, 0, 0],
    );
    assert_eq!(result(&regs), 0);
    assert_eq!(
        world.context(world.a).x[..2],
        [word(KernelError::Cancelled), MINUS_ONE]
    );
    let events = world.events();
    assert_eq!(
        who_and_what(&events[events.len() - 1..]),
        [(
            A,
            CORE_A,
            wait_event(ARRAY, 1, -1, KernelError::Cancelled.code())
        )]
    );
    assert_eq!(
        world.dispatcher.take_woken(),
        [Woken {
            thread: A,
            core: CORE_A,
            result: KernelError::Cancelled.code(),
        }]
    );
    // What GetObject<KThread> refuses.
    let (writable, _) = world.create_event();
    for refused in [writable, readable, Handle::CURRENT_PROCESS, Handle::INVALID] {
        let (_, regs) = world.call(
            B,
            SvcId::CancelSynchronization,
            [u64::from(refused.raw()), 0, 0, 0],
        );
        assert_eq!(
            result(&regs),
            word(KernelError::InvalidHandle),
            "{refused:?}"
        );
    }
}

#[test]
fn a_timeout_is_fired_by_the_timer_thread_and_recorded_then() {
    let world = World::new();
    let (_, readable) = world.create_event();
    // The wait's dispatch reads the clock once (reading 0, after the CreateEvent's): its
    // deadline is one past the next reading, which the first `fire_timers` gets.
    let timeout = i64::try_from(CLOCK_STEP_NS + 1 - TIMEOUT_SLACK_NS).unwrap();
    let (record, _) = world.wait(A, &[readable], timeout);
    assert_eq!(record.outcome, Outcome::Waiting);
    let waited_at = CLOCK_START_NS + CLOCK_STEP_NS;
    let deadline = HostNanos(waited_at + CLOCK_STEP_NS + 1);
    assert_eq!(
        world.dispatcher.fire_timers(&world.env, &world.recorder),
        Ok(Some(deadline))
    );
    assert_eq!(world.state(world.a), crate::sched::ThreadState::Waiting);
    let before = world.events().len();
    assert_eq!(
        world.dispatcher.fire_timers(&world.env, &world.recorder),
        Ok(None)
    );
    assert_eq!(
        world.context(world.a).x[..2],
        [word(KernelError::TimedOut), MINUS_ONE]
    );
    let events = world.events();
    assert_eq!(
        who_and_what(&events[before..]),
        [(
            A,
            CORE_A,
            wait_event(ARRAY, 1, timeout, KernelError::TimedOut.code())
        )]
    );
    assert_eq!(
        events[before].time,
        CounterRate::SWITCH_2.ticks_at(HostNanos(waited_at + 2 * CLOCK_STEP_NS))
    );
    assert_eq!(
        world.dispatcher.take_woken(),
        [Woken {
            thread: A,
            core: CORE_A,
            result: KernelError::TimedOut.code(),
        }]
    );
}

#[test]
fn a_woken_thread_is_reported_with_the_core_it_waited_on() {
    let world = World::new();
    let (writable, readable) = world.create_event();
    let (record, _) = world.wait(B, &[readable], -1);
    assert_eq!(record.outcome, Outcome::Waiting);
    let (_, regs) = world.call(A, SvcId::SignalEvent, [u64::from(writable.raw()), 0, 0, 0]);
    assert_eq!(result(&regs), 0);
    assert_eq!(
        world.dispatcher.take_woken(),
        [Woken {
            thread: B,
            core: CORE_B,
            result: ResultCode::SUCCESS,
        }]
    );
    let events = world.events();
    assert_eq!(
        who_and_what(&events[events.len() - 1..]),
        [(B, CORE_B, wait_event(ARRAY, 1, -1, ResultCode::SUCCESS))]
    );
    assert_eq!(world.state(world.b), crate::sched::ThreadState::Runnable);
}

// ----- Waking from outside a system call -----------------------------------------------------

#[test]
fn a_host_thread_signals_an_event_and_the_waiter_is_recorded() {
    let world = World::new();
    let (_, readable) = world.create_event();
    world.wait(A, &[readable], -1);
    let before = world.events().len();
    let r = world.object(readable);
    assert_eq!(
        world
            .dispatcher
            .signal_event(r, &world.env, &world.recorder),
        Ok(())
    );
    assert_eq!(
        who_and_what(&world.events()[before..]),
        [(A, CORE_A, wait_event(ARRAY, 1, -1, ResultCode::SUCCESS))]
    );
    assert_eq!(
        world.dispatcher.take_woken(),
        [Woken {
            thread: A,
            core: CORE_A,
            result: ResultCode::SUCCESS,
        }]
    );
    // An id that names no event.
    let thread = world.object(world.thread_handle(world.b));
    assert_eq!(
        world
            .dispatcher
            .signal_event(thread, &world.env, &world.recorder),
        Err(WakeError::Call(SvcError::Kernel(
            KernelError::InvalidHandle
        )))
    );
    // A clock that fails: nothing is signalled.
    world.call(A, SvcId::ClearEvent, [u64::from(readable.raw()), 0, 0, 0]);
    let replay = ReplayEnv::new(Vec::new());
    assert!(matches!(
        world.dispatcher.signal_event(r, &replay, &world.recorder),
        Err(WakeError::Env(EnvError::Exhausted { .. }))
    ));
    assert!(matches!(
        world.dispatcher.fire_timers(&replay, &world.recorder),
        Err(WakeError::Env(EnvError::Exhausted { .. }))
    ));
    let (_, regs) = world.call(A, SvcId::ResetSignal, [u64::from(readable.raw()), 0, 0, 0]);
    assert_eq!(result(&regs), word(KernelError::InvalidState));
}

#[test]
fn a_wake_made_without_the_recorder_is_recorded_before_the_woken_threads_next_call() {
    let world = World::new();
    let (_, readable) = world.create_event();
    world.wait(A, &[readable], -1);
    let r = world.object(readable);
    // Signalled with no recorder at hand, as a server's handler does.
    world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| scheduler.signal_object(r))
        .unwrap();
    let before = world.events().len();
    world.call(A, SvcId::GetSystemTick, [0; 4]);
    let kinds: Vec<_> = who_and_what(&world.events()[before..])
        .into_iter()
        .map(|(thread, _, kind)| (thread, kind))
        .collect();
    assert_eq!(kinds[0], (A, wait_event(ARRAY, 1, -1, ResultCode::SUCCESS)));
    assert!(
        matches!(&kinds[1], (A, EventKind::Svc { id, .. }) if *id == u32::from(SvcId::GetSystemTick.number()))
    );
    assert_eq!(kinds.len(), 2);
}

/// A server that creates an event at its first request and signals `target` at every request.
struct Signaller {
    target: ObjectId,
    created: Arc<Mutex<Vec<ObjectId>>>,
}

impl SessionHandler for Signaller {
    fn accept(&mut self, _: ObjectId) -> Result<(), HandlerError> {
        Ok(())
    }

    fn closed(&mut self, _: ObjectId) {}

    fn handle(
        &mut self,
        cx: &mut IpcContext<'_>,
        _: ObjectId,
        message: &mut [u8],
    ) -> Result<Handled, HandlerError> {
        let created = cx.create_event().unwrap();
        self.created
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(created);
        assert_eq!(cx.signal_event(self.target), Ok(()));
        assert_eq!(
            cx.signal_event(ObjectId::from_parts(u32::MAX, 0)),
            Err(KernelError::InvalidHandle)
        );
        message.fill(0);
        Ok(Handled::Replied)
    }

    fn take_reply(
        &mut self,
        _: &mut IpcContext<'_>,
        _: &mut [u8],
    ) -> Option<(ObjectId, Result<(), HandlerError>)> {
        None
    }
}

#[test]
fn a_servers_handler_signals_an_event_and_the_waiter_is_recorded_when_the_request_ends() {
    let world = World::new();
    let (_, readable) = world.create_event();
    let created = Arc::new(Mutex::new(Vec::new()));
    let handler = Signaller {
        target: world.object(readable),
        created: Arc::clone(&created),
    };
    let server = world
        .dispatcher
        .add_server("signaller", ServiceId(0x77), None, Box::new(handler))
        .unwrap();
    let session = world
        .dispatcher
        .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
            let (objects, ipc) = scheduler.ipc_mut();
            let session = ipc.open_session(objects, server, true).unwrap();
            let handle = handles.add(objects, session).unwrap();
            objects.close(session).unwrap();
            handle
        });
    world.wait(A, &[readable], -1);
    let mut message = [0; BUFFER_SIZE];
    RequestBuilder::cmif(CommandType::Request, 1)
        .build(&mut message)
        .unwrap();
    world.memory.write(TLS_B, &message).unwrap();
    let before = world.events().len();
    let (record, regs) = world.call(
        B,
        SvcId::SendSyncRequest,
        [u64::from(session.raw()), 0, 0, 0],
    );
    assert_eq!((record.outcome, result(&regs)), (Outcome::Completed, 0));
    let events = who_and_what(&world.events()[before..]);
    let last = events.last().unwrap();
    assert_eq!(
        last,
        &(A, CORE_A, wait_event(ARRAY, 1, -1, ResultCode::SUCCESS))
    );
    assert_eq!(events[events.len() - 2].0, B);
    assert_eq!(world.dispatcher.take_woken().len(), 1);
    assert_eq!(world.context(world.a).x[..2], [0, 0]);
    // The event the handler made is a readable end, held by the server alone.
    let created = created
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    assert_eq!(created.len(), 1);
    assert_eq!(world.refs(created[0]), 1);
    assert!(matches!(
        world
            .dispatcher
            .with::<(&Scheduler,), _>(|(s,)| s.objects().get(created[0])),
        Ok(Object::ReadableEvent(_))
    ));
}

/// A server that defers every request, and signals `target` when it delivers the reply.
struct DeferredSignaller {
    target: ObjectId,
    pending: Vec<ObjectId>,
}

impl SessionHandler for DeferredSignaller {
    fn accept(&mut self, _: ObjectId) -> Result<(), HandlerError> {
        Ok(())
    }

    fn closed(&mut self, _: ObjectId) {}

    fn handle(
        &mut self,
        _: &mut IpcContext<'_>,
        session: ObjectId,
        _: &mut [u8],
    ) -> Result<Handled, HandlerError> {
        self.pending.push(session);
        Ok(Handled::Deferred)
    }

    fn take_reply(
        &mut self,
        cx: &mut IpcContext<'_>,
        message: &mut [u8],
    ) -> Option<(ObjectId, Result<(), HandlerError>)> {
        let session = self.pending.pop()?;
        assert_eq!(cx.signal_event(self.target), Ok(()));
        message.fill(0);
        Some((session, Ok(())))
    }
}

#[test]
fn a_signal_made_while_delivering_a_deferred_reply_is_recorded_when_the_delivery_ends() {
    let world = World::new();
    let (_, readable) = world.create_event();
    let handler = DeferredSignaller {
        target: world.object(readable),
        pending: Vec::new(),
    };
    let server = world
        .dispatcher
        .add_server("deferred", ServiceId(0x78), None, Box::new(handler))
        .unwrap();
    let session = world
        .dispatcher
        .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
            let (objects, ipc) = scheduler.ipc_mut();
            let session = ipc.open_session(objects, server, true).unwrap();
            let handle = handles.add(objects, session).unwrap();
            objects.close(session).unwrap();
            handle
        });
    world.wait(A, &[readable], -1);
    let mut message = [0; BUFFER_SIZE];
    RequestBuilder::cmif(CommandType::Request, 1)
        .build(&mut message)
        .unwrap();
    world.memory.write(TLS_B, &message).unwrap();
    let (record, _) = world.call(
        B,
        SvcId::SendSyncRequest,
        [u64::from(session.raw()), 0, 0, 0],
    );
    assert_eq!(record.outcome, Outcome::Waiting);
    let before = world.events().len();
    world
        .dispatcher
        .deliver_replies(server, &mut world.caller(B))
        .unwrap();
    let events = who_and_what(&world.events()[before..]);
    assert_eq!(
        events.last(),
        Some(&(A, CORE_A, wait_event(ARRAY, 1, -1, ResultCode::SUCCESS)))
    );
    let woken: Vec<ThreadId> = world
        .dispatcher
        .take_woken()
        .iter()
        .map(|woken| woken.thread)
        .collect();
    assert_eq!(woken, [B, A]);
}
