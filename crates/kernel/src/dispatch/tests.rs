//! The dispatcher driven with a fake register file, environment, memory and debug sink.

use std::sync::mpsc;
use std::sync::{Arc, Barrier, Mutex, PoisonError};

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_memory::FlatMemory;
use starquake_trace::{
    EnvCall, EnvRecord, Event, EventKind, FixedClock, HostNanos, LiveEnv, Recorder, ReplayEnv,
    ThreadId, VecSink,
};

use super::*;
use crate::handle::{MAX_TABLE_SIZE, ObjectRefs};
use crate::id::{ProcessId, SlabId};
use crate::object::Object;
use crate::sched::{CoreSignals, Priority, SchedError};
use crate::svc::{Arg, Kind};
use crate::time::{CounterRate, NANOS_PER_SECOND};

/// Above 4 GiB, so an address truncated to 32 bits would miss it.
const BASE: u64 = 0x80_0000_0000;
/// The counter rate of the process in these tests: the Switch 2 rate, so a conversion at the
/// Switch 1 rate (the only rate the kernel had before it kept guest time) shows.
const RATE: CounterRate = CounterRate::SWITCH_2;
/// The first clock reading, in nanoseconds (what the `Env` reports): a whole number of seconds,
/// so the expected guest ticks are `CLOCK_SECONDS * RATE.hz()` with no rounding.
const CLOCK_SECONDS: u64 = 1_000_000_007;
const CLOCK_START_NS: u64 = CLOCK_SECONDS * NANOS_PER_SECOND;
/// How far the clock moves between readings: one second, so each dispatch's guest time is a
/// whole second of ticks after the one before.
const CLOCK_STEP_NS: u64 = NANOS_PER_SECOND;

/// The thread every test call is made by: distinct from the core numbers used and from zero, so
/// a swapped or dropped field shows in the recorded event.
const THREAD: ThreadId = ThreadId(0x7EAD);

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
const PROPTEST_SEED: u64 = 0x4449_5350_4154_4348;

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 256,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(PROPTEST_SEED),
        failure_persistence: None,
        ..Config::default()
    }
}

/// A live environment whose clock reads `CLOCK_START_NS` first, then `CLOCK_STEP_NS` more each
/// time.
fn env() -> LiveEnv {
    LiveEnv::new(FixedClock::new(CLOCK_START_NS, CLOCK_STEP_NS), 0)
}

/// The guest time of the `n`th dispatch (from zero) against [`env`].
fn time_of(n: u64) -> GuestTicks {
    RATE.ticks_at(HostNanos(CLOCK_START_NS + n * CLOCK_STEP_NS))
}

/// Fresh tables for a process at `rate`.
fn tables_at(rate: CounterRate) -> Tables {
    Tables {
        handles: HandleTable::new(i32::from(MAX_TABLE_SIZE)).expect("a maximum-size handle table"),
        address_space: AddressSpace::default(),
        scheduler: Scheduler::default(),
        processes: Processes::new(rate),
    }
}

/// Fresh tables. The one place the tests build them.
fn tables() -> Tables {
    tables_at(RATE)
}

/// Collects debug strings with the time each was output at.
#[derive(Default)]
struct Collect(Vec<(GuestTicks, Vec<u8>)>);

impl Collect {
    fn messages(&self) -> Vec<Vec<u8>> {
        self.0.iter().map(|(_, message)| message.clone()).collect()
    }
}

impl DebugSink for Collect {
    fn output(&mut self, time: GuestTicks, message: &[u8]) {
        self.0.push((time, message.to_vec()));
    }
}

struct NoSignals;

impl CoreSignals for NoSignals {
    fn interrupt(&mut self, _: CoreId) {}

    fn wake(&mut self, _: CoreId) {}
}

/// Registers that all differ, so a stray write or a read from the wrong register shows up.
fn regs() -> GuestRegs {
    let mut regs = GuestRegs::default();
    for (i, x) in (0u64..).zip(regs.x.iter_mut()) {
        *x = 0x1111_1111_1111_1111_u64.wrapping_mul(i + 1);
    }
    regs
}

fn not_implemented_word() -> u64 {
    u64::from(KernelError::NotImplemented.code().raw())
}

fn word(error: KernelError) -> u64 {
    u64::from(error.code().raw())
}

/// Runs one call on core `core` against fresh tables and the given memory.
fn dispatch_one(
    number: u16,
    regs: &mut GuestRegs,
    core: u8,
    memory: &FlatMemory,
    sink: &mut Collect,
) -> Record {
    dispatch_recorded(number, regs, core, memory, sink, &Recorder::new())
}

/// As [`dispatch_one`], recording into `recorder`.
fn dispatch_recorded(
    number: u16,
    regs: &mut GuestRegs,
    core: u8,
    memory: &FlatMemory,
    sink: &mut Collect,
    recorder: &Recorder,
) -> Record {
    let dispatcher = Dispatcher::new(tables());
    let env = env();
    let mut caller = Caller {
        core,
        thread: THREAD,
        env: &env,
        memory,
        debug: sink,
        recorder,
    };
    dispatcher.dispatch(number, regs, &mut caller)
}

/// Runs one call on `dispatcher` from core `core`, with a fresh environment, no guest memory
/// worth reading and a recorder that keeps the events.
fn dispatch_on(
    dispatcher: &Dispatcher,
    number: u16,
    regs: &mut GuestRegs,
    core: u8,
) -> (Record, Vec<Event>) {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let env = env();
    let (recorder, events) = recorder();
    let mut caller = Caller {
        core,
        thread: THREAD,
        env: &env,
        memory: &memory,
        debug: &mut sink,
        recorder: &recorder,
    };
    let record = dispatcher.dispatch(number, regs, &mut caller);
    (record, events.events())
}

/// A recorder with one sink that keeps every event.
fn recorder() -> (Recorder, Arc<VecSink>) {
    let recorder = Recorder::new();
    let sink = Arc::new(VecSink::new());
    recorder.add_sink(Arc::clone(&sink));
    (recorder, sink)
}

/// The event a call must have been recorded as, with `args` as the event carries them. `time`
/// is the dispatch's guest time.
fn svc_event(
    seq: u64,
    time: GuestTicks,
    core: u8,
    number: u16,
    args: [u64; SVC_ARG_COUNT],
    result: ResultCode,
) -> Event {
    Event {
        seq,
        time,
        core,
        thread: THREAD,
        kind: EventKind::Svc {
            id: u32::from(number),
            args,
            result: starquake_trace::ResultCode(result.raw()),
        },
    }
}

/// The kind of event a call with one argument in `x<register>` is recorded as.
fn svc_kind(id: SvcId, register: usize, argument: u64, result: ResultCode) -> EventKind {
    let mut args = [0; SVC_ARG_COUNT];
    args[register] = argument;
    EventKind::Svc {
        id: u32::from(id.number()),
        args,
        result: starquake_trace::ResultCode(result.raw()),
    }
}

fn kinds(events: &[Event]) -> Vec<EventKind> {
    events.iter().map(|event| event.kind).collect()
}

fn assert_untouched_from(regs: &GuestRegs, before: &GuestRegs, first: usize) {
    assert_eq!(regs.x[first..], before.x[first..]);
    assert_eq!(
        (regs.sp, regs.pc, regs.pstate, regs.tpidr, regs.tpidrro),
        (
            before.sp,
            before.pc,
            before.pstate,
            before.tpidr,
            before.tpidrro
        )
    );
}

fn core_id(index: u8) -> CoreId {
    CoreId::new(index).unwrap()
}

/// Creates a thread of priority `priority` bound to `core` in `tables`, with a handle to it
/// that holds the object's only reference, as a creating system call leaves it.
fn thread_with_handle(
    tables: &mut Tables,
    priority: u8,
    core: u8,
) -> (crate::id::ThreadId, Handle) {
    let thread = tables
        .scheduler
        .create_thread(Priority::new(priority).unwrap(), core_id(core))
        .unwrap();
    let objects = tables.scheduler.objects_mut();
    let object = objects.create(Object::Thread(thread)).unwrap();
    let handle = tables.handles.add(objects, object).unwrap();
    objects.close(object).unwrap();
    (thread, handle)
}

/// Makes `thread` the thread `core` is running.
fn run_on(tables: &mut Tables, thread: crate::id::ThreadId, core: u8) {
    tables.scheduler.start(thread).unwrap();
    tables
        .scheduler
        .update(GuestTicks(0), None, &mut NoSignals)
        .unwrap();
    assert_eq!(
        tables.scheduler.schedule(core_id(core), &mut NoSignals),
        Ok(Some(thread))
    );
}

// --- Guest time -----------------------------------------------------------------------------

#[test]
fn get_system_tick_writes_the_dispatchs_guest_time_to_x0() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let mut regs = regs();
    let before = regs.clone();
    let record = dispatch_one(
        SvcId::GetSystemTick.number(),
        &mut regs,
        0,
        &memory,
        &mut sink,
    );
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(record.id, Some(SvcId::GetSystemTick));
    // The clock reports nanoseconds; the guest gets ticks at the process's rate, not the raw
    // reading. A full 64-bit value (int64_t), not truncated to a 32-bit word.
    let ticks = CLOCK_SECONDS * RATE.hz();
    assert_ne!(ticks >> 32, 0);
    assert_eq!(regs.x[0], ticks);
    assert_eq!(GuestTicks(ticks), time_of(0));
    assert_untouched_from(&regs, &before, 1);
}

#[test]
fn guest_time_follows_the_rate_of_the_process() {
    // One clock, two processes: each reads its own rate's worth of ticks, and its events are
    // stamped with the same number.
    for rate in [CounterRate::SWITCH_1, CounterRate::SWITCH_2] {
        let dispatcher = Dispatcher::new(tables_at(rate));
        let mut regs = regs();
        let (record, events) =
            dispatch_on(&dispatcher, SvcId::GetSystemTick.number(), &mut regs, 0);
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!(regs.x[0], CLOCK_SECONDS * rate.hz());
        assert_eq!(
            events.iter().map(|event| event.time).collect::<Vec<_>>(),
            [GuestTicks(CLOCK_SECONDS * rate.hz())]
        );
    }
    assert_ne!(CounterRate::SWITCH_1.hz(), CounterRate::SWITCH_2.hz());
}

#[test]
fn every_dispatch_reads_the_clock_once_and_stamps_its_event_with_that_reading() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let dispatcher = Dispatcher::new(tables());
    let env = env();
    let (recorder, events) = recorder();
    let mut caller = Caller {
        core: 2,
        thread: THREAD,
        env: &env,
        memory: &memory,
        debug: &mut sink,
        recorder: &recorder,
    };
    // An implemented call that returns the time, an unlisted number, a planned call, one that
    // touches tables, and the first again.
    let numbers = [
        SvcId::GetSystemTick.number(),
        u16::MAX,
        SvcId::SetHeapSize.number(),
        SvcId::CloseHandle.number(),
        SvcId::GetSystemTick.number(),
    ];
    let mut ticks = Vec::new();
    for number in numbers {
        let mut regs = regs();
        let before = regs.clone();
        let record = dispatcher.dispatch(number, &mut regs, &mut caller);
        // The record holds x0-x7 from before the call, not the outputs written over them.
        assert_eq!(record.args[..], before.x[..SVC_ARG_COUNT]);
        assert_ne!(regs.x[0], before.x[0]);
        assert_eq!((record.core, record.number), (2, number));
        if number == SvcId::GetSystemTick.number() {
            ticks.push(GuestTicks(regs.x[0]));
        }
    }
    // One event per call, numbered in call order, each stamped with its own dispatch's time.
    let stamped: Vec<(u64, GuestTicks)> = events
        .events()
        .iter()
        .map(|event| (event.seq, event.time))
        .collect();
    let expected: Vec<(u64, GuestTicks)> = (0u64..).zip((0u64..5).map(time_of)).collect();
    assert_eq!(stamped, expected);
    // What `GetSystemTick` returned is the time its own event carries.
    assert_eq!(ticks, [time_of(0), time_of(4)]);
    // One clock reading per dispatch, in order, whatever the call.
    let readings: Vec<EnvRecord> = (0u64..5)
        .map(|n| EnvRecord::Clock(HostNanos(CLOCK_START_NS + n * CLOCK_STEP_NS)))
        .collect();
    assert_eq!(env.recording(), readings);
    let _tables: Tables = dispatcher.into_tables();
}

/// How many threads dispatch at once in
/// `event_times_follow_the_kernels_order_when_threads_race_to_dispatch`.
const RACING_THREADS: u8 = 4;
/// How many calls each of them makes.
const CALLS_PER_THREAD: u64 = 500;

#[test]
fn event_times_follow_the_kernels_order_when_threads_race_to_dispatch() {
    // Threads share one dispatcher, one environment and one recorder. The clock is read under
    // the kernel lock, so the reading a dispatch gets and the `seq` its event gets are taken in
    // the same order: sorted by `seq`, the times are the clock's readings in order. A reading
    // taken before the lock could be overtaken on the way to it, and two events would be out of
    // order.
    let dispatcher = Dispatcher::new(tables());
    let env = env();
    let (recorder, events) = recorder();
    let start = Barrier::new(usize::from(RACING_THREADS));
    std::thread::scope(|scope| {
        for core in 0..RACING_THREADS {
            let (dispatcher, env, recorder, start) = (&dispatcher, &env, &recorder, &start);
            scope.spawn(move || {
                let memory = FlatMemory::new(BASE, 0x10);
                let mut sink = Collect::default();
                let mut caller = Caller {
                    core,
                    thread: THREAD,
                    env,
                    memory: &memory,
                    debug: &mut sink,
                    recorder,
                };
                start.wait();
                for _ in 0..CALLS_PER_THREAD {
                    let mut regs = GuestRegs::default();
                    let number = SvcId::GetSystemTick.number();
                    let record = dispatcher.dispatch(number, &mut regs, &mut caller);
                    assert_eq!(record.outcome, Outcome::Completed);
                }
            });
        }
    });
    let mut events = events.events();
    events.sort_by_key(|event| event.seq);
    let calls = u64::from(RACING_THREADS) * CALLS_PER_THREAD;
    assert_eq!(u64::try_from(events.len()), Ok(calls));
    // The first event whose time is not the reading its place in the order implies; each
    // reading is a whole second of ticks after the one before, so this is also "strictly
    // increasing".
    let out_of_order = (0u64..)
        .zip(&events)
        .find(|(n, event)| event.time != time_of(*n))
        .map(|(n, event)| (n, event.time, time_of(n)));
    assert_eq!(out_of_order, None);
    // And the environment's recording is those readings, in that order.
    let readings: Vec<EnvRecord> = (0..calls)
        .map(|n| EnvRecord::Clock(HostNanos(CLOCK_START_NS + n * CLOCK_STEP_NS)))
        .collect();
    assert_eq!(env.recording(), readings);
}

#[test]
fn a_dispatch_that_waits_for_the_lock_reads_the_clock_only_once_it_holds_it() {
    let (held, is_held) = mpsc::channel();
    let (waiting, is_waiting) = mpsc::channel();
    let dispatcher = Dispatcher::new(tables()).with_lock_timer(StepTimer::new(waiting));
    let env = env();
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let recorder = Recorder::new();
    let mut regs = regs();
    std::thread::scope(|scope| {
        let (dispatcher, env) = (&dispatcher, &env);
        scope.spawn(move || {
            dispatcher.with::<(&mut Scheduler,), _>(|_| {
                held.send(()).unwrap();
                // The dispatch below has found the lock held: the lock timer is read for the
                // first time only then.
                is_waiting.recv().unwrap();
                // It cannot go on until this closure returns, and it has not read the clock.
                assert_eq!(env.recording(), []);
            });
        });
        is_held.recv().unwrap();
        let mut caller = Caller {
            core: 0,
            thread: THREAD,
            env,
            memory: &memory,
            debug: &mut sink,
            recorder: &recorder,
        };
        let record = dispatcher.dispatch(SvcId::GetSystemTick.number(), &mut regs, &mut caller);
        assert_eq!(record.outcome, Outcome::Completed);
    });
    assert_eq!(GuestTicks(regs.x[0]), time_of(0));
    assert_eq!(
        env.recording(),
        [EnvRecord::Clock(HostNanos(CLOCK_START_NS))]
    );
}

#[test]
fn a_replay_of_the_clock_gives_the_same_ticks_and_event_times() {
    let live = env();
    let (live_recorder, live_events) = recorder();
    let (replay_recorder, replay_events) = recorder();
    let numbers = [
        SvcId::GetSystemTick.number(),
        SvcId::OutputDebugString.number(),
        SvcId::GetSystemTick.number(),
    ];
    let run = |env: &dyn Env, recorder: &Recorder| {
        let memory = FlatMemory::new(BASE, 0x10);
        let mut sink = Collect::default();
        let dispatcher = Dispatcher::new(tables());
        let mut caller = Caller {
            core: 0,
            thread: THREAD,
            env,
            memory: &memory,
            debug: &mut sink,
            recorder,
        };
        numbers
            .iter()
            .map(|&number| {
                let mut regs = GuestRegs::default();
                dispatcher.dispatch(number, &mut regs, &mut caller);
                regs.x[0]
            })
            .collect::<Vec<u64>>()
    };
    let first = run(&live, &live_recorder);
    let replay = ReplayEnv::new(live.recording());
    let second = run(&replay, &replay_recorder);
    assert_eq!(first, second);
    assert_eq!(first[0], time_of(0).0);
    assert_eq!(first[2], time_of(2).0);
    assert_eq!(live_events.events(), replay_events.events());
    assert_eq!(replay.consumed(), numbers.len());
}

#[test]
fn a_failing_environment_ends_the_call_without_writing_registers() {
    // A replay with no clock reading left: the dispatch has no time to give the call or its
    // event. Whatever the call is, it does not run.
    for number in [
        SvcId::GetSystemTick.number(),
        SvcId::OutputDebugString.number(),
        SvcId::SetHeapSize.number(),
        u16::MAX,
    ] {
        let memory = FlatMemory::new(BASE, 0x100);
        memory.write(BASE, b"line").unwrap();
        let mut sink = Collect::default();
        let dispatcher = Dispatcher::new(tables());
        let env = ReplayEnv::new(Vec::new());
        let (recorder, events) = recorder();
        let mut caller = Caller {
            core: 0,
            thread: THREAD,
            env: &env,
            memory: &memory,
            debug: &mut sink,
            recorder: &recorder,
        };
        let mut regs = regs();
        regs.x[0] = BASE;
        regs.x[1] = 4;
        let before = regs.clone();
        let record = dispatcher.dispatch(number, &mut regs, &mut caller);
        assert_eq!(
            record.outcome,
            Outcome::EnvFailed(EnvError::Exhausted {
                index: 0,
                requested: EnvCall::Clock
            }),
            "svc {number:#x}"
        );
        assert_untouched_from(&regs, &before, 0);
        // The call never returned to the guest, so it has no result and no event, and it did
        // nothing: no debug string was output.
        assert_eq!(events.events(), []);
        assert_eq!(sink.messages(), Vec::<Vec<u8>>::new());
    }
}

// --- The calls that touch no table ----------------------------------------------------------

#[test]
fn get_current_processor_number_writes_the_core_zero_extended() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    for core in [0, 3, 5] {
        let mut regs = regs();
        regs.x[0] = u64::MAX;
        let before = regs.clone();
        let record = dispatch_one(
            SvcId::GetCurrentProcessorNumber.number(),
            &mut regs,
            core,
            &memory,
            &mut sink,
        );
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!(record.core, core);
        assert_eq!(regs.x[0], u64::from(core));
        assert_untouched_from(&regs, &before, 1);
    }
}

#[test]
fn output_debug_string_decodes_x0_and_x1_and_writes_success() {
    let memory = FlatMemory::new(BASE, 0x100);
    memory.write(BASE + 0x10, b"hello from the guest").unwrap();
    let mut sink = Collect::default();
    let mut regs = regs();
    regs.x[0] = BASE + 0x10;
    regs.x[1] = 5;
    let before = regs.clone();
    let record = dispatch_one(
        SvcId::OutputDebugString.number(),
        &mut regs,
        0,
        &memory,
        &mut sink,
    );
    assert_eq!(record.outcome, Outcome::Completed);
    // The sink is told the dispatch's guest time: the time the call's own event carries.
    assert_eq!(sink.0, [(time_of(0), b"hello".to_vec())]);
    assert_eq!(regs.x[0], u64::from(ResultCode::SUCCESS.raw()));
    // Only x0 is an output; x1 still holds the length.
    assert_untouched_from(&regs, &before, 1);
}

#[test]
fn output_debug_string_reports_its_error_in_x0() {
    let memory = FlatMemory::new(BASE, 0x100);
    let mut sink = Collect::default();
    let mut regs = regs();
    regs.x[0] = BASE + 0xff;
    regs.x[1] = 2; // one byte past the end of mapped memory
    let record = dispatch_one(
        SvcId::OutputDebugString.number(),
        &mut regs,
        0,
        &memory,
        &mut sink,
    );
    // The call ran; its error result is an output like any other.
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(regs.x[0], word(KernelError::InvalidCurrentMemory));
    assert_eq!(sink.messages(), Vec::<Vec<u8>>::new());
}

#[test]
fn exit_process_ends_the_process_and_writes_no_register() {
    // Eden's `SvcWrap_ExitProcess64` takes no argument and writes no output.
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let mut regs = regs();
    let before = regs.clone();
    let record = dispatch_one(
        SvcId::ExitProcess.number(),
        &mut regs,
        4,
        &memory,
        &mut sink,
    );
    assert_eq!(record.outcome, Outcome::ProcessExited);
    assert_eq!(record.id, Some(SvcId::ExitProcess));
    assert_untouched_from(&regs, &before, 0);
}

// --- The calls that take tables -------------------------------------------------------------

#[test]
fn close_handle_takes_the_handle_table_and_the_scheduler_and_closes_the_handle() {
    let mut tables = tables();
    let (_, handle) = thread_with_handle(&mut tables, 40, 0);
    let (_, kept) = thread_with_handle(&mut tables, 41, 0);
    let object = tables.handles.get(handle).unwrap();
    let dispatcher = Dispatcher::new(tables);
    let id = SvcId::CloseHandle;

    let mut regs = regs();
    // The handle is the low word of x0; the upper half is not part of it.
    regs.x[0] = 0xdead_beef_0000_0000 | u64::from(handle.raw());
    let before = regs.clone();
    let (record, events) = dispatch_on(&dispatcher, id.number(), &mut regs, 1);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(regs.x[0], u64::from(ResultCode::SUCCESS.raw()));
    assert_untouched_from(&regs, &before, 1);
    assert_eq!(
        kinds(&events),
        [svc_kind(
            id,
            0,
            u64::from(handle.raw()),
            ResultCode::SUCCESS
        )]
    );

    // Closing it again finds nothing: the first call changed both tables.
    let mut regs = GuestRegs::default();
    regs.x[0] = u64::from(handle.raw());
    let (record, _) = dispatch_on(&dispatcher, id.number(), &mut regs, 1);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(regs.x[0], word(KernelError::InvalidHandle));

    let tables = dispatcher.into_tables();
    assert_eq!(tables.handles.len(), 1);
    assert!(tables.handles.get(kept).is_ok());
    // The handle held the object's last reference, so the object is gone too.
    assert_eq!(tables.scheduler.objects().refs(object), 0);
    assert_eq!(tables.scheduler.objects().len(), 1);
}

#[test]
fn get_thread_priority_reads_the_scheduler_through_a_handle_in_x1() {
    let mut tables = tables();
    let (_, handle) = thread_with_handle(&mut tables, 37, 2);
    let dispatcher = Dispatcher::new(tables);
    let id = SvcId::GetThreadPriority;

    let mut regs = regs();
    // Eden's wrapper: `thread_handle:Handle:x1 | ret:Result:x0 out_priority:int32_t:x1`.
    regs.x[1] = 0xffff_ffff_0000_0000 | u64::from(handle.raw());
    let before = regs.clone();
    let (record, events) = dispatch_on(&dispatcher, id.number(), &mut regs, 4);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!((regs.x[0], regs.x[1]), (0, 37));
    assert_untouched_from(&regs, &before, 2);
    assert_eq!(
        kinds(&events),
        [svc_kind(
            id,
            1,
            u64::from(handle.raw()),
            ResultCode::SUCCESS
        )]
    );
    // A query: both tables are as they were.
    let tables = dispatcher.into_tables();
    assert_eq!(
        (tables.handles.len(), tables.scheduler.objects().len()),
        (1, 1)
    );
}

#[test]
fn the_current_thread_pseudo_handle_is_the_thread_the_calling_core_runs() {
    let mut tables = tables();
    let (first, _) = thread_with_handle(&mut tables, 30, 0);
    let (second, _) = thread_with_handle(&mut tables, 44, 3);
    run_on(&mut tables, first, 0);
    run_on(&mut tables, second, 3);
    let dispatcher = Dispatcher::new(tables);
    for (core, priority) in [(0, 30), (3, 44)] {
        let mut regs = regs();
        regs.x[1] = u64::from(Handle::CURRENT_THREAD.raw());
        let (record, _) = dispatch_on(
            &dispatcher,
            SvcId::GetThreadPriority.number(),
            &mut regs,
            core,
        );
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!((regs.x[0], regs.x[1]), (0, priority), "core {core}");
    }
}

#[test]
fn a_failing_call_writes_its_result_and_zeroes_its_other_outputs() {
    // Eden's `SvcWrap_GetThreadPriority64` writes `ret` to x0 and the value-initialised
    // `out_priority` to x1 whether or not the call succeeded.
    let mut tables = tables();
    let process = tables
        .scheduler
        .objects_mut()
        .create(Object::Process(ProcessId::from_parts(0, 1)))
        .unwrap();
    let process_handle = tables
        .handles
        .add(tables.scheduler.objects_mut(), process)
        .unwrap();
    let dispatcher = Dispatcher::new(tables);
    let id = SvcId::GetThreadPriority;
    for handle in [
        Handle::INVALID,
        Handle::CURRENT_PROCESS,
        process_handle,
        Handle::from_raw(0x1234_5678),
    ] {
        let mut regs = regs();
        regs.x[1] = u64::from(handle.raw());
        let before = regs.clone();
        let (record, events) = dispatch_on(&dispatcher, id.number(), &mut regs, 0);
        assert_eq!(record.outcome, Outcome::Completed, "{handle:?}");
        assert_eq!(
            (regs.x[0], regs.x[1]),
            (word(KernelError::InvalidHandle), 0),
            "{handle:?}"
        );
        assert_untouched_from(&regs, &before, 2);
        assert_eq!(
            kinds(&events),
            [svc_kind(
                id,
                1,
                u64::from(handle.raw()),
                KernelError::InvalidHandle.code()
            )]
        );
    }
}

#[test]
fn an_internal_failure_is_not_implemented_to_the_guest() {
    // The scheduler has not been told what any core runs, so "the calling thread" has no
    // answer. That is the kernel's own state, not the guest's mistake: the guest sees what it
    // sees for a call that isn't implemented, with nothing but x0 written.
    let dispatcher = Dispatcher::new(tables());
    let id = SvcId::GetThreadPriority;
    // Core 200 is no core at all (a `CoreId` can't hold it); core 2 exists and runs nothing.
    for core in [2, 200] {
        let mut regs = regs();
        regs.x[1] = u64::from(Handle::CURRENT_THREAD.raw());
        let before = regs.clone();
        let (record, events) = dispatch_on(&dispatcher, id.number(), &mut regs, core);
        assert_eq!(record.outcome, Outcome::NotImplemented, "core {core}");
        assert_eq!(regs.x[0], not_implemented_word());
        assert_untouched_from(&regs, &before, 1);
        assert_eq!(
            kinds(&events),
            [svc_kind(
                id,
                1,
                u64::from(Handle::CURRENT_THREAD.raw()),
                KernelError::NotImplemented.code()
            )]
        );
    }
    // The same through the helper every `Result`-returning arm uses, with the failure spelled
    // out: an internal failure writes x0 only.
    let convention = id.convention().unwrap();
    let mut regs = regs();
    let before = regs.clone();
    let internal = SchedError::NoCurrentThread { core: 2 };
    assert_eq!(
        returning(
            id,
            convention,
            &mut regs,
            |[_handle]| -> CallResult<[u64; 1]> { Err(internal.into()) }
        ),
        Outcome::NotImplemented
    );
    assert_eq!(regs.x[0], not_implemented_word());
    assert_untouched_from(&regs, &before, 1);
}

// --- Declared tables ------------------------------------------------------------------------

#[test]
fn a_table_set_declares_exactly_the_tables_it_names_and_how() {
    let shared = Some(Access::Shared);
    let exclusive = Some(Access::Exclusive);
    let declared = |handles, address_space, scheduler| Declared {
        handles,
        address_space,
        scheduler,
    };
    assert_eq!(<() as TableSet>::DECLARED, Declared::NONE);
    assert_eq!(Declared::NONE, declared(None, None, None));
    assert_eq!(
        <(&HandleTable,) as TableSet>::DECLARED,
        declared(shared, None, None)
    );
    assert_eq!(
        <(&mut HandleTable,) as TableSet>::DECLARED,
        declared(exclusive, None, None)
    );
    assert_eq!(
        <(&AddressSpace,) as TableSet>::DECLARED,
        declared(None, shared, None)
    );
    assert_eq!(
        <(&mut AddressSpace,) as TableSet>::DECLARED,
        declared(None, exclusive, None)
    );
    assert_eq!(
        <(&Scheduler,) as TableSet>::DECLARED,
        declared(None, None, shared)
    );
    assert_eq!(
        <(&mut Scheduler,) as TableSet>::DECLARED,
        declared(None, None, exclusive)
    );
    assert_eq!(
        <(&HandleTable, &mut AddressSpace) as TableSet>::DECLARED,
        declared(shared, exclusive, None)
    );
    assert_eq!(
        <(&mut HandleTable, &Scheduler) as TableSet>::DECLARED,
        declared(exclusive, None, shared)
    );
    assert_eq!(
        <(&mut AddressSpace, &mut Scheduler) as TableSet>::DECLARED,
        declared(None, exclusive, exclusive)
    );
    assert_eq!(
        <(&HandleTable, &mut AddressSpace, &Scheduler) as TableSet>::DECLARED,
        declared(shared, exclusive, shared)
    );
    assert_eq!(
        <(&mut HandleTable, &AddressSpace, &mut Scheduler) as TableSet>::DECLARED,
        declared(exclusive, shared, exclusive)
    );
}

#[test]
fn with_hands_over_the_declared_tables_and_keeps_what_was_done_to_them() {
    let dispatcher = Dispatcher::new(tables());
    // Two tables at once, both exclusive: disjoint borrows of one locked struct.
    let handle =
        dispatcher.with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
            let thread = scheduler
                .create_thread(Priority::new(33).unwrap(), core_id(1))
                .unwrap();
            let object = scheduler
                .objects_mut()
                .create(Object::Thread(thread))
                .unwrap();
            handles.add(scheduler.objects_mut(), object).unwrap()
        });
    // A shared pair sees what the exclusive pair left.
    let priority = dispatcher.with::<(&HandleTable, &Scheduler), _>(|(handles, scheduler)| {
        thread::get_thread_priority(scheduler, handles, core_id(0), handle)
    });
    assert_eq!(priority, Ok(33));
    // One table alone, and none.
    assert_eq!(
        dispatcher.with::<(&HandleTable,), _>(|(handles,)| handles.len()),
        1
    );
    assert_eq!(dispatcher.with::<(), _>(|()| 7), 7);
    // And a dispatch sees the same tables.
    let mut regs = GuestRegs::default();
    regs.x[1] = u64::from(handle.raw());
    let (record, _) = dispatch_on(&dispatcher, SvcId::GetThreadPriority.number(), &mut regs, 0);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!((regs.x[0], regs.x[1]), (0, 33));
}

// --- Lock waits -----------------------------------------------------------------------------

/// A lock timer that moves [`STEP`] forward at every reading, and tells a channel the first
/// time it is read: the dispatcher reads it only when it has found the lock held.
struct StepTimer {
    now: Mutex<Duration>,
    waiting: Mutex<Option<mpsc::Sender<()>>>,
}

/// How far [`StepTimer`] moves per reading: the wait every contended acquisition measures.
const STEP: Duration = Duration::from_micros(250);

impl StepTimer {
    fn new(waiting: mpsc::Sender<()>) -> Self {
        Self {
            now: Mutex::new(Duration::from_secs(5)),
            waiting: Mutex::new(Some(waiting)),
        }
    }
}

impl LockTimer for StepTimer {
    fn now(&self) -> Duration {
        let mut now = self.now.lock().unwrap_or_else(PoisonError::into_inner);
        let reading = *now;
        *now += STEP;
        let first = self
            .waiting
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(waiting) = first {
            waiting.send(()).unwrap();
        }
        reading
    }
}

fn waits(acquisitions: u64, exclusive: u64, contended: u64, waited: Duration) -> Waits {
    Waits {
        acquisitions,
        exclusive,
        contended,
        waited,
    }
}

#[test]
fn every_acquisition_is_counted_under_the_tables_its_call_declared() {
    let dispatcher = Dispatcher::new(tables());
    assert_eq!(dispatcher.lock_waits(), LockWaits::default());
    // No table; the handle table and the scheduler exclusively; both shared; an unlisted number
    // (no table).
    for number in [
        SvcId::GetSystemTick.number(),
        SvcId::CloseHandle.number(),
        SvcId::GetThreadPriority.number(),
        u16::MAX,
    ] {
        dispatch_on(&dispatcher, number, &mut GuestRegs::default(), 0);
    }
    dispatcher.with::<(&AddressSpace, &mut Scheduler), _>(|_| ());
    // Reading the report is not an acquisition: a second reading is the same.
    let report = dispatcher.lock_waits();
    assert_eq!(dispatcher.lock_waits(), report);
    assert_eq!(
        report,
        LockWaits {
            timed: false,
            // Five acquisitions; two of them declared a table exclusively.
            lock: waits(5, 2, 0, Duration::ZERO),
            handles: waits(2, 1, 0, Duration::ZERO),
            address_space: waits(1, 0, 0, Duration::ZERO),
            scheduler: waits(3, 2, 0, Duration::ZERO),
        }
    );
}

#[test]
fn a_wait_for_a_held_lock_is_timed_and_charged_to_each_declared_table() {
    let (held, is_held) = mpsc::channel();
    let (waiting, is_waiting) = mpsc::channel();
    let dispatcher = Dispatcher::new(tables()).with_lock_timer(StepTimer::new(waiting));
    let shared = &dispatcher;
    std::thread::scope(|scope| {
        // Holds the lock until the dispatch below is known to have found it held: the timer is
        // read for the first time only then.
        scope.spawn(move || {
            shared.with::<(&mut Scheduler,), _>(|_| {
                held.send(()).unwrap();
                is_waiting.recv().unwrap();
            });
        });
        is_held.recv().unwrap();
        let mut regs = GuestRegs::default();
        let (record, _) = dispatch_on(&dispatcher, SvcId::CloseHandle.number(), &mut regs, 0);
        assert_eq!(record.outcome, Outcome::Completed);
    });
    assert_eq!(
        dispatcher.lock_waits(),
        LockWaits {
            timed: true,
            // The holder and the dispatch; only the dispatch waited, for one timer step.
            lock: waits(2, 2, 1, STEP),
            // `CloseHandle` declares the handle table and the scheduler, so its wait is under
            // both; the holder declared the scheduler only, and did not wait.
            handles: waits(1, 1, 1, STEP),
            address_space: Waits::default(),
            scheduler: waits(2, 2, 1, STEP),
        }
    );
}

#[test]
fn a_poisoned_lock_is_taken_without_counting_a_wait() {
    let dispatcher = Dispatcher::new(tables());
    // A panic while the tables are held poisons the lock.
    let panicked = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                dispatcher.with::<(&mut Scheduler,), _>(|_| {
                    panic!("a panic that poisons the dispatcher's lock")
                });
            })
            .join()
    });
    assert!(panicked.is_err());
    assert!(dispatcher.state.is_poisoned());
    // The next dispatch takes it all the same, and it did not have to wait for it.
    let mut regs = regs();
    let (record, events) = dispatch_on(&dispatcher, SvcId::GetSystemTick.number(), &mut regs, 0);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(GuestTicks(regs.x[0]), time_of(0));
    assert_eq!(events.len(), 1);
    assert_eq!(
        dispatcher.lock_waits(),
        LockWaits {
            timed: false,
            // The panicking holder, which declared the scheduler exclusively, and the dispatch,
            // which declared no table. Neither found the lock held.
            lock: waits(2, 1, 0, Duration::ZERO),
            handles: Waits::default(),
            address_space: Waits::default(),
            scheduler: waits(1, 1, 0, Duration::ZERO),
        }
    );
}

#[test]
fn without_a_timer_a_wait_measures_as_zero() {
    let dispatcher = Dispatcher::new(tables());
    assert_eq!(dispatcher.timer_now(), Duration::ZERO);
    assert!(!dispatcher.lock_waits().timed);
    // The same contended acquisition the timed test counts, as the report would hold it.
    let waited = dispatcher
        .timer_now()
        .saturating_sub(dispatcher.timer_now());
    let mut report = LockWaits::default();
    report.note(
        <(&mut HandleTable, &mut Scheduler) as TableSet>::DECLARED,
        Some(waited),
    );
    assert_eq!(report.lock, waits(1, 1, 1, Duration::ZERO));
    assert_eq!(report.handles, waits(1, 1, 1, Duration::ZERO));
    assert_eq!(report.scheduler, waits(1, 1, 1, Duration::ZERO));
    assert_eq!(report.address_space, Waits::default());
}

#[test]
fn wait_counters_saturate_instead_of_wrapping() {
    let full = Waits {
        acquisitions: u64::MAX,
        exclusive: u64::MAX,
        contended: u64::MAX,
        waited: Duration::MAX,
    };
    let mut noted = full;
    noted.note(Access::Exclusive, Some(STEP));
    assert_eq!(noted, full);
}

// --- Unknown and unimplemented calls --------------------------------------------------------

#[test]
fn unknown_numbers_return_not_implemented_and_touch_only_x0() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let unlisted: Vec<u16> = (0..=u16::MAX)
        .filter(|&n| SvcId::from_number(n).is_none())
        .collect();
    // 0 (below the first call), the gap after GetResourceLimitPeakValue, the first number past
    // the last call, and the top.
    let gap = SvcId::GetResourceLimitPeakValue.number() + 1;
    let past_last = SvcId::ALL.last().unwrap().number() + 1;
    for number in [0, gap, past_last, u16::MAX] {
        assert!(unlisted.contains(&number), "{number:#x} is listed");
        let mut regs = regs();
        let before = regs.clone();
        let record = dispatch_one(number, &mut regs, 1, &memory, &mut sink);
        let mut args = [0; SVC_ARG_COUNT];
        args.copy_from_slice(&before.x[..SVC_ARG_COUNT]);
        assert_eq!(
            record,
            Record {
                core: 1,
                number,
                id: None,
                args,
                outcome: Outcome::NotImplemented,
            }
        );
        assert_eq!(regs.x[0], not_implemented_word());
        assert_untouched_from(&regs, &before, 1);
    }
}

#[test]
fn every_unlisted_number_is_handled_without_panicking() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let dispatcher = Dispatcher::new(tables());
    let env = env();
    let recorder = Recorder::new();
    let mut caller = Caller {
        core: 0,
        thread: THREAD,
        env: &env,
        memory: &memory,
        debug: &mut sink,
        recorder: &recorder,
    };
    for number in (0..=u16::MAX).filter(|&n| SvcId::from_number(n).is_none()) {
        let mut regs = regs();
        let record = dispatcher.dispatch(number, &mut regs, &mut caller);
        assert_eq!(record.outcome, Outcome::NotImplemented, "{number:#x}");
        assert_eq!(regs.x[0], not_implemented_word(), "{number:#x}");
    }
}

proptest! {
    #![proptest_config(config())]

    /// Any svc number with any register values, against an `Env`, memory and sink that don't
    /// panic themselves, completes the dispatch and returns a record for that number.
    #[test]
    fn any_number_and_registers_dispatch_without_panicking(
        number in any::<u16>(),
        core in any::<u8>(),
        x in prop::array::uniform8(any::<u64>()),
    ) {
        let memory = FlatMemory::new(BASE, 0x10);
        let mut sink = Collect::default();
        let mut regs = GuestRegs::default();
        regs.x[..x.len()].copy_from_slice(&x);
        let record = dispatch_one(number, &mut regs, core, &memory, &mut sink);
        prop_assert_eq!(record.number, number);
        prop_assert_eq!(record.core, core);
        prop_assert_eq!(record.args, x);
    }
}

#[test]
fn a_planned_call_returns_not_implemented_and_leaves_its_outputs_alone() {
    // SetHeapSize's convention outputs x0 (result) and x1 (address). Divergence from Eden's
    // wrapper, which would write x1 too: only x0 changes. A deferred call with a convention
    // (light IPC, which outputs x0-x7) behaves the same.
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    for (id, status) in [
        (SvcId::SetHeapSize, Status::Planned),
        (SvcId::SetGpuMemoryAttribute, Status::Planned),
        (SvcId::ExitThread, Status::Planned),
        (SvcId::SendSyncRequestLight, Status::Deferred),
    ] {
        assert_eq!(id.status(), status);
        assert!(id.convention().is_some());
        let mut regs = regs();
        let before = regs.clone();
        let record = dispatch_one(id.number(), &mut regs, 0, &memory, &mut sink);
        assert_eq!(record.outcome, Outcome::NotImplemented, "{}", id.name());
        assert_eq!(record.id, Some(id));
        assert_eq!(regs.x[0], not_implemented_word(), "{}", id.name());
        assert_untouched_from(&regs, &before, 1);
    }
}

#[test]
fn a_not_planned_call_returns_not_implemented() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let id = SvcId::CallSecureMonitor;
    assert_eq!(id.status(), Status::NotPlanned);
    let mut regs = regs();
    let record = dispatch_one(id.number(), &mut regs, 0, &memory, &mut sink);
    assert_eq!(record.outcome, Outcome::NotImplemented);
    assert_eq!(regs.x[0], not_implemented_word());
}

#[test]
fn the_dispatcher_runs_exactly_the_calls_marked_implemented() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    for &id in SvcId::ALL {
        // Zeroed registers: OutputDebugString then has length 0 and succeeds, and the calls
        // that take a handle are given the invalid handle, which is a result, not a failure
        // to run.
        let mut regs = GuestRegs::default();
        let record = dispatch_one(id.number(), &mut regs, 0, &memory, &mut sink);
        let expected = match (id.status(), id) {
            (Status::Implemented, SvcId::ExitProcess) => Outcome::ProcessExited,
            (Status::Implemented, _) => Outcome::Completed,
            _ => Outcome::NotImplemented,
        };
        assert_eq!(
            record.outcome,
            expected,
            "{} ({:#x})",
            id.name(),
            id.number()
        );
    }
}

// --- The recording point --------------------------------------------------------------------

#[test]
fn every_call_that_returns_is_recorded_once_with_its_decoded_arguments_and_result() {
    let memory = FlatMemory::new(BASE, 0x100);
    memory.write(BASE, b"line").unwrap();
    let not_implemented = KernelError::NotImplemented.code();
    // Every register holds something before each call (see `regs`), as a guest's registers do:
    // what an earlier call left behind must not reach the event.
    let stale = regs();
    let raw = |x0: u64, x1: u64| {
        let mut all = [0; SVC_ARG_COUNT];
        all.copy_from_slice(&stale.x[..SVC_ARG_COUNT]);
        all[0] = x0;
        all[1] = x1;
        all
    };
    // A handle is 32 bits: the upper half of its register is not part of the argument.
    let handle_register = 0xdead_beef_0000_8001;
    // (number, x0, x1, the arguments the event must carry, the result it must carry)
    let calls = [
        // Two arguments and a `Result` output: success, then an error (one byte past the end
        // of mapped memory).
        (
            SvcId::OutputDebugString.number(),
            BASE,
            4,
            [BASE, 4, 0, 0, 0, 0, 0, 0],
            ResultCode::SUCCESS,
        ),
        (
            SvcId::OutputDebugString.number(),
            BASE + 0xff,
            2,
            [BASE + 0xff, 2, 0, 0, 0, 0, 0, 0],
            KernelError::InvalidCurrentMemory.code(),
        ),
        // No arguments. Outputs that are not a `Result` (ticks, a core number) are not
        // recorded as one.
        (
            SvcId::GetSystemTick.number(),
            1,
            2,
            [0; SVC_ARG_COUNT],
            ResultCode::SUCCESS,
        ),
        (
            SvcId::GetCurrentProcessorNumber.number(),
            3,
            4,
            [0; SVC_ARG_COUNT],
            ResultCode::SUCCESS,
        ),
        // A void call with no arguments.
        (
            SvcId::ExitProcess.number(),
            5,
            6,
            [0; SVC_ARG_COUNT],
            ResultCode::SUCCESS,
        ),
        // Planned calls are decoded too. SetHeapSize's one argument is in x1, not x0.
        (
            SvcId::SetHeapSize.number(),
            7,
            8,
            [0, 8, 0, 0, 0, 0, 0, 0],
            not_implemented,
        ),
        // A call that takes tables and fails: no handle in the fresh table has this value.
        (
            SvcId::CloseHandle.number(),
            handle_register,
            9,
            [0x8001, 0, 0, 0, 0, 0, 0, 0],
            KernelError::InvalidHandle.code(),
        ),
        // A listed call with no convention entry yet, and an unlisted number: nothing says
        // which registers are arguments, so all eight are kept as passed.
        (
            SvcId::CreateSession.number(),
            10,
            11,
            raw(10, 11),
            not_implemented,
        ),
        (u16::MAX, 12, 13, raw(12, 13), not_implemented),
    ];
    assert_eq!(SvcId::CreateSession.convention(), None);
    for (number, x0, x1, args, result) in calls {
        let (recorder, events) = recorder();
        let mut sink = Collect::default();
        let mut regs = stale.clone();
        regs.x[0] = x0;
        regs.x[1] = x1;
        let record = dispatch_recorded(number, &mut regs, 3, &memory, &mut sink, &recorder);
        assert_eq!(
            events.events(),
            [svc_event(0, time_of(0), 3, number, args, result)],
            "svc {number:#x}"
        );
        // The dispatch's own record still has the registers as passed.
        assert_eq!(record.args, raw(x0, x1), "svc {number:#x}");
    }
}

#[test]
fn a_call_that_disagrees_with_its_convention_returns_not_implemented() {
    static ONE_OUTPUT: &[Arg] = &[Arg {
        name: "ret",
        reg: 0,
        kind: Kind::Result,
    }];
    let convention = Convention {
        inputs: &[],
        outputs: ONE_OUTPUT,
    };
    let id = SvcId::GetSystemTick;
    let mut regs = regs();
    let before = regs.clone();
    assert_eq!(
        complete(id, &convention, &mut regs, &[1, 2]),
        Outcome::NotImplemented
    );
    assert_eq!(regs.x[0], not_implemented_word());
    assert_untouched_from(&regs, &before, 1);

    // The same through `returning`: one input too many, one output too many, and more outputs
    // than there are registers.
    let reads_one_too_many = returning(
        id,
        &convention,
        &mut regs,
        |[_]: [u64; 1]| -> CallResult<[u64; 0]> { Ok([]) },
    );
    assert_eq!(reads_one_too_many, Outcome::NotImplemented);
    let writes_one_too_many = returning(
        id,
        &convention,
        &mut regs,
        |[]: [u64; 0]| -> CallResult<[u64; 1]> { Ok([7]) },
    );
    assert_eq!(writes_one_too_many, Outcome::NotImplemented);
    let more_than_the_registers = returning(
        id,
        &convention,
        &mut regs,
        |[]: [u64; 0]| -> CallResult<[u64; SVC_ARG_COUNT]> { Ok([7; SVC_ARG_COUNT]) },
    );
    assert_eq!(more_than_the_registers, Outcome::NotImplemented);
    assert_eq!(regs.x[0], not_implemented_word());
    assert_untouched_from(&regs, &before, 1);

    // And one that agrees: the result, and nothing else.
    let agrees = returning(
        id,
        &convention,
        &mut regs,
        |[]: [u64; 0]| -> CallResult<[u64; 0]> { Err(KernelError::Busy.into()) },
    );
    assert_eq!(agrees, Outcome::Completed);
    assert_eq!(regs.x[0], word(KernelError::Busy));
    assert_untouched_from(&regs, &before, 1);
}

#[test]
fn the_dispatcher_can_be_shared_between_core_threads() {
    fn shared<T: Send + Sync>() {}
    shared::<Dispatcher>();
    assert_eq!(
        format!("{:?}", Dispatcher::new(tables())),
        "Dispatcher { timed: false, .. }"
    );
}
