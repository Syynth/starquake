//! The core loop's half of the dispatcher, driven against a homebrew process whose main thread
//! has started but runs nowhere yet, with a fake for the core signals.

use starquake_cpu::GuestRegs;
use starquake_memory::{FlatMemory, PAGE_SIZE};
use starquake_trace::{FixedClock, GuestTicks, LiveEnv, Recorder, ThreadId as EventThread};

use super::*;
use crate::dispatch::{Caller, Outcome, Record};
use crate::memory::{GUARD_PAGES, fixture};
use crate::process::{Console, MainThread, ProcessDescription};
use crate::sched::{PREEMPTION_PRIORITIES, Priority};
use crate::svc::{DebugSink, SvcId};

const ENTRY: u64 = 0x8000_0040;
const STACK_TOP: u64 = 0x10_8000_0000;

/// What the core loop was asked to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signal {
    Interrupt(u8),
    Wake(u8),
}

#[derive(Default)]
struct Signals {
    log: Vec<Signal>,
}

impl CoreSignals for Signals {
    fn interrupt(&mut self, core: CoreId) {
        self.log.push(Signal::Interrupt(core.index()));
    }

    fn wake(&mut self, core: CoreId) {
        self.log.push(Signal::Wake(core.index()));
    }
}

struct NoDebug;

impl DebugSink for NoDebug {
    fn output(&mut self, _: GuestTicks, _: &[u8]) {}
}

fn core(index: u8) -> CoreId {
    CoreId::new(index).unwrap()
}

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
        Self {
            dispatcher: Dispatcher::new(tables),
            memory,
            main,
            env: LiveEnv::new(FixedClock::new(1_000, 1_000), 0),
            recorder: Recorder::new(),
        }
    }

    /// Dispatches `id` on core `on` with `regs`, as the thread named 1.
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

    fn switch(&self, on: u8, save: Option<&GuestRegs>) -> (Switch, Vec<Signal>) {
        let mut signals = Signals::default();
        let switch = self
            .dispatcher
            .switch(core(on), save, &mut signals)
            .unwrap();
        (switch, signals.log)
    }

    fn scheduler<R>(&self, f: impl FnOnce(&mut Scheduler) -> R) -> R {
        self.dispatcher
            .with::<(&mut Scheduler,), _>(|(scheduler,)| f(scheduler))
    }

    /// A started thread of no process at `priority` on core `on`.
    fn spawn(&self, priority: u8, on: u8) -> ThreadId {
        self.scheduler(|scheduler| {
            let id = scheduler
                .create_thread(Priority::new(priority).unwrap(), core(on))
                .unwrap();
            scheduler.start(id).unwrap();
            id
        })
    }

    fn context(&self, thread: ThreadId) -> GuestRegs {
        self.scheduler(|scheduler| scheduler.thread(thread).unwrap().context().clone())
    }

    /// Runs the main thread on core 0 and returns its registers.
    fn run_main(&self) -> GuestRegs {
        let loaded = load(self.switch(0, None).0);
        assert_eq!(loaded.thread, self.main.thread);
        loaded.context
    }
}

/// The thread a switch loads; a test fails if it loads none.
fn load(switch: Switch) -> Loaded {
    let loaded = match switch {
        Switch::Load(loaded) => Some(*loaded),
        Switch::Stay | Switch::Park => None,
    };
    loaded.expect("the core loads a thread")
}

/// Registers that differ from any a thread starts with.
fn marked(mark: u64) -> GuestRegs {
    let mut regs = GuestRegs::default();
    regs.x[19] = mark;
    regs.pc = 0x8000_0000 + mark;
    regs
}

#[test]
fn a_core_loads_the_thread_chosen_for_it_then_stays_on_it() {
    let world = World::new();
    let (switch, signals) = world.switch(0, None);
    let loaded = load(switch);
    assert_eq!(loaded.thread, world.main.thread);
    assert_eq!(loaded.guest_id, EventThread(crate::sched::FIRST_THREAD_ID));
    assert_eq!(loaded.context, world.context(world.main.thread));
    assert_eq!(loaded.context.pc, ENTRY);
    // The caller is not signalled; no other core's thread changed.
    assert_eq!(signals, vec![]);
    // Nothing changed since: it stays, and what it would have saved is ignored.
    let before = world.context(world.main.thread);
    assert_eq!(world.switch(0, Some(&marked(1))).0, Switch::Stay);
    assert_eq!(world.context(world.main.thread), before);
}

#[test]
fn a_core_with_nothing_to_run_parks() {
    let world = World::new();
    assert_eq!(world.switch(1, None).0, Switch::Park);
    assert_eq!(world.switch(1, None).0, Switch::Park);
}

#[test]
fn switching_away_saves_the_registers_the_core_gives_and_loads_the_next_thread() {
    let world = World::new();
    world.run_main();
    let main_priority = world.scheduler(|scheduler| {
        scheduler
            .thread(world.main.thread)
            .unwrap()
            .priority()
            .get()
    });
    let second = world.spawn(main_priority, 0);
    // The main thread yields: the second one is next on its level.
    world.scheduler(|scheduler| scheduler.yield_without_core_migration(core(0)).unwrap());
    let running = marked(7);
    let (switch, _) = world.switch(0, Some(&running));
    let loaded = load(switch);
    assert_eq!(loaded.thread, second);
    assert_eq!(world.context(world.main.thread), running);
}

#[test]
fn a_thread_that_began_a_wait_keeps_the_context_its_wait_saved() {
    let world = World::new();
    world.run_main();
    let mut regs = marked(3);
    regs.x[0] = 5;
    let record = world.call(0, SvcId::SleepThread, &mut regs);
    assert_eq!(record.outcome, Outcome::Waiting);
    // A core that wrongly passes newer registers does not overwrite the saved ones.
    let (switch, _) = world.switch(0, Some(&marked(4)));
    assert_eq!(switch, Switch::Park);
    assert_eq!(world.context(world.main.thread), regs);
}

#[test]
fn a_woken_thread_more_urgent_than_a_cores_thread_interrupts_that_core() {
    let world = World::new();
    world.run_main();
    // A more urgent thread of core 0 that sleeps on core 1's behalf: it runs on core 1 first.
    let urgent = world.spawn(20, 1);
    let loaded = load(world.switch(1, None).0);
    assert_eq!(loaded.thread, urgent);
    let mut regs = marked(5);
    regs.x[0] = 5;
    assert_eq!(
        world.call(1, SvcId::SleepThread, &mut regs).outcome,
        Outcome::Waiting
    );
    assert_eq!(world.switch(1, None).0, Switch::Park);
    // Move it to core 0, whose thread is less urgent, while it sleeps.
    world.scheduler(|scheduler| {
        scheduler
            .set_core_mask(
                urgent,
                crate::sched::IdealCore::Core(core(0)),
                AffinityMask::single(core(0)),
            )
            .unwrap();
    });
    // The timer ends the sleep; telling the cores interrupts core 0, which runs guest code.
    world
        .dispatcher
        .fire_timers(&world.env, &world.recorder)
        .unwrap();
    let mut signals = Signals::default();
    let changed = world.dispatcher.reschedule(&mut signals).unwrap();
    assert_eq!(changed, AffinityMask::single(core(0)));
    assert_eq!(signals.log, vec![Signal::Interrupt(0)]);
    let (switch, _) = world.switch(0, Some(&marked(6)));
    let loaded = load(switch);
    assert_eq!(loaded.thread, urgent);
    assert_eq!(loaded.context, regs);
    assert_eq!(world.context(world.main.thread), marked(6));
}

#[test]
fn reschedule_with_nothing_changed_signals_nobody() {
    let world = World::new();
    world.run_main();
    let mut signals = Signals::default();
    assert_eq!(
        world.dispatcher.reschedule(&mut signals).unwrap(),
        AffinityMask::EMPTY
    );
    assert_eq!(signals.log, vec![]);
}

#[test]
fn preemption_takes_turns_between_threads_at_a_cores_preemption_level() {
    let world = World::new();
    let level = PREEMPTION_PRIORITIES[1].get();
    let first = world.spawn(level, 1);
    let second = world.spawn(level, 1);
    let loaded = load(world.switch(1, None).0);
    assert_eq!(loaded.thread, first);
    for expected in [second, first] {
        let mut signals = Signals::default();
        let changed = world.dispatcher.preempt(&mut signals).unwrap();
        assert!(changed.contains(core(1)));
        assert!(signals.log.contains(&Signal::Interrupt(1)));
        let loaded = load(world.switch(1, Some(&marked(9))).0);
        assert_eq!(loaded.thread, expected);
    }
}

#[test]
fn a_process_is_stalled_only_when_nothing_can_run_or_end_a_wait() {
    let world = World::new();
    // The main thread is runnable.
    assert!(!world.dispatcher.is_stalled());
    world.run_main();
    // It sleeps with a timer: time will end the wait.
    let mut regs = marked(1);
    regs.x[0] = 5;
    world.call(0, SvcId::SleepThread, &mut regs);
    assert!(!world.dispatcher.is_stalled());
    world
        .dispatcher
        .fire_timers(&world.env, &world.recorder)
        .unwrap();
    assert!(!world.dispatcher.is_stalled());
    // It sleeps with no timer: nothing can end it.
    world.switch(0, None);
    let mut forever = marked(2);
    forever.x[0] = i64::MAX.cast_unsigned();
    assert_eq!(
        world.call(0, SvcId::SleepThread, &mut forever).outcome,
        Outcome::Waiting
    );
    assert!(world.dispatcher.is_stalled());
}
