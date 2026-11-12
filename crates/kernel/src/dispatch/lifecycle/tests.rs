//! The lifecycle calls' dispatch arms: which registers each reads and writes, and how each
//! dispatch ends.

use starquake_cpu::GuestRegs;
use starquake_memory::{FlatMemory, PAGE_SIZE};
use starquake_trace::{
    EventKind, FixedClock, GuestTicks, LiveEnv, Recorder, ThreadId as EventThread,
};

use super::*;
use crate::dispatch::{Caller, Record};
use crate::memory::{GUARD_PAGES, fixture};
use crate::process::{
    Console, IDEAL_CORE_DONT_CARE, IDEAL_CORE_USE_PROCESS_VALUE, MainThread, ProcessDescription,
    THREAD_LOCAL_REGION_SIZE,
};
use crate::result::KernelError;
use crate::sched::{CoreId, CoreSignals, FIRST_THREAD_ID, ThreadState};
use crate::svc::DebugSink;
use crate::tables::FIRST_USER_PROCESS_ID;

const ENTRY: u64 = 0x8000_0040;
const STACK_TOP: u64 = 0x10_8000_0000;

struct NoSignals;

impl CoreSignals for NoSignals {
    fn interrupt(&mut self, _: CoreId) {}

    fn wake(&mut self, _: CoreId) {}
}

struct NoDebug;

impl DebugSink for NoDebug {
    fn output(&mut self, _: GuestTicks, _: &[u8]) {}
}

fn core(index: u8) -> CoreId {
    CoreId::new(index).unwrap()
}

/// A dispatcher over a homebrew process whose main thread runs on core 0, and memory over its
/// first thread-local page.
struct World {
    dispatcher: Dispatcher,
    memory: FlatMemory,
    main: MainThread,
    env: LiveEnv,
    recorder: Recorder,
}

impl World {
    fn new() -> Self {
        let mut tables = crate::process::create(
            &ProcessDescription::homebrew(Console::Switch1),
            fixture::loaded().0,
        )
        .unwrap();
        let first = tables.address_space.regions().kernel_map.start + GUARD_PAGES * PAGE_SIZE;
        let memory = FlatMemory::new(first, usize::try_from(PAGE_SIZE).unwrap());
        let main = crate::process::run(&mut tables, &memory, ENTRY, STACK_TOP).unwrap();
        let world = Self {
            dispatcher: Dispatcher::new(tables),
            memory,
            main,
            env: LiveEnv::new(FixedClock::new(1_000, 1_000), 0),
            recorder: Recorder::new(),
        };
        world.schedule(core(0), Some(main.thread));
        world
    }

    fn schedule(&self, on: CoreId, expected: Option<crate::id::ThreadId>) {
        self.dispatcher
            .with::<(&mut Scheduler,), _>(|(scheduler,)| {
                scheduler
                    .update(GuestTicks(0), None, &mut NoSignals)
                    .unwrap();
                assert_eq!(scheduler.schedule(on, &mut NoSignals), Ok(expected));
            });
    }

    /// Dispatches `id` on core `on` with `regs`.
    fn call(&self, on: u8, id: SvcId, regs: &mut GuestRegs) -> Record {
        let mut debug = NoDebug;
        let mut caller = Caller {
            core: on,
            thread: EventThread(1),
            env: &self.env,
            memory: &self.memory,
            debug: &mut debug,
            recorder: &self.recorder,
        };
        self.dispatcher.dispatch(id.number(), regs, &mut caller)
    }

    fn with_tables<R>(&self, f: impl FnOnce(&HandleTable, &Scheduler) -> R) -> R {
        self.dispatcher
            .with::<(&HandleTable, &Scheduler), _>(|(handles, scheduler)| f(handles, scheduler))
    }
}

/// Registers that all differ, so a stray write or a read from the wrong register shows.
fn regs() -> GuestRegs {
    let mut regs = GuestRegs::default();
    for (i, x) in (0_u64..).zip(regs.x.iter_mut()) {
        *x = 0x1111_1111_1111_1111_u64.wrapping_mul(i + 1);
    }
    regs
}

/// A 4-byte argument in a register whose upper half holds something else: the call must read
/// only the low four bytes.
fn word(value: i32) -> u64 {
    0xDEAD_BEEF_0000_0000 | u64::from(value.cast_unsigned())
}

fn result(error: KernelError) -> u64 {
    u64::from(error.code().raw())
}

#[test]
fn create_thread_reads_x1_to_x5_and_returns_the_handle_in_x1() {
    let world = World::new();
    let mut regs = regs();
    let (entry, argument, stack_top) = (0x8000_2000, 0x1234_5678, 0x10_9000_0000);
    regs.x[1] = entry;
    regs.x[2] = argument;
    regs.x[3] = stack_top;
    regs.x[4] = word(44);
    regs.x[5] = word(IDEAL_CORE_USE_PROCESS_VALUE);
    let before = regs.clone();
    let record = world.call(0, SvcId::CreateThread, &mut regs);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(regs.x[0], 0);
    let handle = Handle::from_register(regs.x[1]);
    assert_eq!(regs.x[2..], before.x[2..]);
    world.with_tables(|handles, scheduler| {
        let thread = thread::thread_from_handle(scheduler, handles, core(0), handle).unwrap();
        let row = scheduler.thread(thread).unwrap();
        assert_eq!(row.state(), ThreadState::Initialized);
        let context = row.context();
        assert_eq!(
            (context.pc, context.x[0], context.sp),
            (entry, argument, stack_top)
        );
        assert_eq!(row.priority().get(), 44);
        assert_eq!(row.virtual_core_mask(), (0, 1));
    });
    // A failure writes its result and zeroes the handle.
    let mut failing = self::regs();
    failing.x[4] = word(64);
    failing.x[5] = word(0);
    let record = world.call(0, SvcId::CreateThread, &mut failing);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(
        (failing.x[0], failing.x[1]),
        (result(KernelError::InvalidPriority), 0)
    );
}

#[test]
fn start_thread_reads_x0() {
    let world = World::new();
    let mut create = regs();
    create.x[4] = word(44);
    create.x[5] = word(1);
    world.call(0, SvcId::CreateThread, &mut create);
    let handle = create.x[1];
    let mut regs = regs();
    regs.x[0] = handle;
    let before = regs.clone();
    assert_eq!(
        world.call(0, SvcId::StartThread, &mut regs).outcome,
        Outcome::Completed
    );
    assert_eq!(regs.x[0], 0);
    assert_eq!(regs.x[1..], before.x[1..]);
    regs.x[0] = handle;
    world.call(0, SvcId::StartThread, &mut regs);
    assert_eq!(regs.x[0], result(KernelError::InvalidState));
}

#[test]
fn exit_thread_ends_the_thread_or_with_the_last_one_the_process() {
    let world = World::new();
    let mut create = regs();
    create.x[4] = word(44);
    create.x[5] = word(1);
    world.call(0, SvcId::CreateThread, &mut create);
    let mut start = regs();
    start.x[0] = create.x[1];
    world.call(0, SvcId::StartThread, &mut start);
    let second = world.with_tables(|handles, scheduler| {
        let handle = Handle::from_register(create.x[1]);
        thread::thread_from_handle(scheduler, handles, core(0), handle).unwrap()
    });
    world.schedule(core(1), Some(second));
    let mut regs = regs();
    let before = regs.clone();
    let record = world.call(1, SvcId::ExitThread, &mut regs);
    assert_eq!(record.outcome, Outcome::ThreadExited);
    assert_eq!(regs, before, "ExitThread writes no register");
    let record = world.call(0, SvcId::ExitThread, &mut regs);
    assert_eq!(record.outcome, Outcome::ProcessExited);
    assert_eq!(regs, before);
    world.with_tables(|handles, _| assert!(handles.is_empty()));
}

#[test]
fn sleep_thread_writes_no_register() {
    let world = World::new();
    for ns in [0_i64, -1, -2, -3] {
        let mut regs = regs();
        regs.x[0] = ns.cast_unsigned();
        let before = regs.clone();
        let record = world.call(0, SvcId::SleepThread, &mut regs);
        assert_eq!(record.outcome, Outcome::Completed, "{ns}");
        assert_eq!(regs, before, "{ns}");
    }
}

#[test]
fn a_positive_sleep_waits_and_its_event_is_recorded_when_its_timer_ends_it() {
    let sink = std::sync::Arc::new(starquake_trace::VecSink::new());
    let world = World::new();
    world.recorder.add_sink(std::sync::Arc::clone(&sink));
    let mut regs = regs();
    regs.x[0] = 5;
    let before = regs.clone();
    let record = world.call(0, SvcId::SleepThread, &mut regs);
    assert_eq!(record.outcome, Outcome::Waiting);
    assert_eq!(regs, before, "a sleep writes no register");
    assert!(sink.events().is_empty(), "nothing is recorded yet");
    world.with_tables(|_, scheduler| {
        let main = scheduler.thread(world.main.thread).unwrap();
        assert_eq!(main.state(), ThreadState::Waiting);
        assert_eq!(main.context(), &before);
    });
    // The clock steps 1,000 ns a reading, far past the 5 ns sleep: the next firing ends it.
    world
        .dispatcher
        .fire_timers(&world.env, &world.recorder)
        .unwrap();
    let events: Vec<_> = sink.events().into_iter().map(|event| event.kind).collect();
    let mut args = [0; starquake_trace::SVC_ARG_COUNT];
    args[0] = 5;
    assert_eq!(
        events,
        vec![EventKind::Svc {
            id: u32::from(SvcId::SleepThread.number()),
            args,
            result: starquake_trace::ResultCode(0),
        }]
    );
    world.with_tables(|_, scheduler| {
        let main = scheduler.thread(world.main.thread).unwrap();
        assert_eq!(main.state(), ThreadState::Runnable);
        assert_eq!(main.context(), &before);
    });
}

#[test]
fn the_core_mask_calls_read_and_write_their_registers() {
    let world = World::new();
    let mut set = regs();
    set.x[0] = u64::from(Handle::CURRENT_THREAD.raw());
    set.x[1] = word(IDEAL_CORE_DONT_CARE);
    set.x[2] = 0b101;
    let before = set.clone();
    assert_eq!(
        world.call(0, SvcId::SetThreadCoreMask, &mut set).outcome,
        Outcome::Completed
    );
    assert_eq!(set.x[0], 0);
    assert_eq!(set.x[1..], before.x[1..]);
    let mut get = regs();
    get.x[2] = u64::from(Handle::CURRENT_THREAD.raw());
    world.call(0, SvcId::GetThreadCoreMask, &mut get);
    // The ideal core is an `int32_t`: -1, written zero-extended.
    assert_eq!(
        (get.x[0], get.x[1], get.x[2]),
        (0, u64::from(IDEAL_CORE_DONT_CARE.cast_unsigned()), 0b101)
    );
}

#[test]
fn set_thread_priority_reads_x0_and_x1() {
    let world = World::new();
    let mut regs = regs();
    regs.x[0] = u64::from(Handle::CURRENT_THREAD.raw());
    regs.x[1] = word(30);
    world.call(0, SvcId::SetThreadPriority, &mut regs);
    assert_eq!(regs.x[0], 0);
    let mut get = regs.clone();
    get.x[1] = u64::from(Handle::CURRENT_THREAD.raw());
    world.call(0, SvcId::GetThreadPriority, &mut get);
    assert_eq!((get.x[0], get.x[1]), (0, 30));
}

#[test]
fn the_id_calls_return_in_x1() {
    let world = World::new();
    let mut thread = regs();
    thread.x[1] = u64::from(world.main.handle.raw());
    world.call(0, SvcId::GetThreadId, &mut thread);
    assert_eq!((thread.x[0], thread.x[1]), (0, FIRST_THREAD_ID));
    let mut process = regs();
    process.x[1] = u64::from(Handle::CURRENT_PROCESS.raw());
    world.call(0, SvcId::GetProcessId, &mut process);
    assert_eq!((process.x[0], process.x[1]), (0, FIRST_USER_PROCESS_ID));
    let mut bad = regs();
    bad.x[1] = u64::from(Handle::CURRENT_THREAD.raw());
    world.call(0, SvcId::GetProcessId, &mut bad);
    assert_eq!(
        (bad.x[0], bad.x[1]),
        (result(KernelError::InvalidHandle), 0)
    );
}

#[test]
fn a_call_from_a_core_running_nothing_is_not_implemented() {
    let world = World::new();
    for id in [SvcId::ExitThread, SvcId::SleepThread, SvcId::CreateThread] {
        let mut regs = regs();
        // SleepThread's yield (zero) acts on the calling core's thread; CreateThread ignores x0.
        regs.x[0] = 0;
        let record = world.call(5, id, &mut regs);
        assert_eq!(record.outcome, Outcome::NotImplemented, "{}", id.name());
        assert_eq!(
            regs.x[0],
            result(KernelError::NotImplemented),
            "{}",
            id.name()
        );
    }
    // The second thread-local region was never taken.
    world.with_tables(|_, scheduler| {
        let tls = scheduler.thread(world.main.thread).unwrap().tls_address();
        let own = scheduler.process(
            scheduler
                .thread(world.main.thread)
                .unwrap()
                .process()
                .unwrap(),
        );
        assert_eq!(own.unwrap().tls().used(), 1);
        assert_eq!(tls % THREAD_LOCAL_REGION_SIZE, 0);
    });
}

#[test]
fn exiting_twice_is_not_implemented_the_second_time() {
    let world = World::new();
    let mut regs = regs();
    assert_eq!(
        world.call(0, SvcId::ExitProcess, &mut regs).outcome,
        Outcome::ProcessExited
    );
    let before = regs.clone();
    assert_eq!(
        world.call(0, SvcId::ExitProcess, &mut regs).outcome,
        Outcome::NotImplemented
    );
    assert_eq!(regs.x[0], result(KernelError::NotImplemented));
    assert_eq!(regs.x[1..], before.x[1..]);
}
