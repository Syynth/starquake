//! Tests for the scheduler: scenarios for each yield and for migration, with the outcome taken
//! from Eden's `k_scheduler.cpp` (cited per test), the thread operations and their errors, the
//! listed divergences, and a property test of the scheduling invariants under random
//! operations.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use super::*;
use crate::id::SlabId;
use crate::sched::{HIGHEST_THREAD_PRIORITY, LOWEST_THREAD_PRIORITY, PRIORITY_COUNT, SuspendType};
use crate::slab::SlabError;

/// What the fake core loop was asked to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signal {
    Interrupt(u8),
    Wake(u8),
}

/// A [`CoreSignals`] that records every call.
#[derive(Default)]
struct Fake {
    log: Vec<Signal>,
}

impl CoreSignals for Fake {
    fn interrupt(&mut self, core: CoreId) {
        self.log.push(Signal::Interrupt(core.index()));
    }

    fn wake(&mut self, core: CoreId) {
        self.log.push(Signal::Wake(core.index()));
    }
}

fn core(index: u8) -> CoreId {
    CoreId::new(index).unwrap()
}

fn prio(value: u8) -> Priority {
    Priority::new(value).unwrap()
}

fn mask(cores: &[u8]) -> AffinityMask {
    AffinityMask::from_bits(cores.iter().fold(0, |bits, &c| bits | core(c).bit()))
}

/// A started thread at `priority` on `ideal`, allowed on `allowed`.
fn spawn(s: &mut Scheduler, priority: u8, ideal: u8, allowed: &[u8]) -> ThreadId {
    let id = s.create_thread(prio(priority), core(ideal)).unwrap();
    if allowed != [ideal] {
        s.set_core_mask(id, IdealCore::Core(core(ideal)), mask(allowed))
            .unwrap();
    }
    s.start(id).unwrap();
    id
}

/// Recomputes every core's thread at `now`, then lets every core switch until none needs to.
fn settle_at(s: &mut Scheduler, now: u64) -> Fake {
    let mut fake = Fake::default();
    s.update(GuestTicks(now), None, &mut fake).unwrap();
    for _ in 0..=s.core_count() {
        let mut any = false;
        for c in 0..s.core_count() {
            if s.needs_scheduling(core(c)).unwrap() {
                any = true;
                s.schedule(core(c), &mut fake).unwrap();
            }
        }
        if !any {
            break;
        }
    }
    fake
}

fn settle(s: &mut Scheduler) -> Fake {
    settle_at(s, 0)
}

fn highest(s: &Scheduler, c: u8) -> Option<ThreadId> {
    s.highest(core(c)).unwrap()
}

fn kernel(error: KernelError) -> SchedError {
    SchedError::Kernel(error)
}

// -------------------------------------------------------------------------------------------
// Construction and thread operations
// -------------------------------------------------------------------------------------------

#[test]
fn core_count_is_configuration_within_bounds() {
    assert_eq!(Scheduler::default().core_count(), APPLICATION_CORE_COUNT);
    assert_eq!(Scheduler::new(MAX_CORES).unwrap().core_count(), MAX_CORES);
    assert_eq!(Scheduler::new(1).unwrap().core_count(), 1);
    assert_eq!(Scheduler::new(0), Err(SchedError::CoreCount { count: 0 }));
    assert_eq!(
        Scheduler::new(MAX_CORES + 1),
        Err(SchedError::CoreCount {
            count: MAX_CORES + 1
        })
    );
}

#[test]
fn every_core_starts_needing_scheduling_and_idles() {
    let mut s = Scheduler::new(2).unwrap();
    assert!(s.needs_scheduling(core(0)).unwrap());
    let fake = settle(&mut s);
    assert_eq!(fake.log, vec![]);
    assert_eq!(s.current(core(0)).unwrap(), None);
    assert!(!s.needs_scheduling(core(1)).unwrap());
}

#[test]
fn priority_is_bounded_by_the_svc_constants() {
    assert_eq!(
        Priority::new(HIGHEST_THREAD_PRIORITY),
        Some(Priority::HIGHEST)
    );
    assert_eq!(
        Priority::new(LOWEST_THREAD_PRIORITY),
        Some(Priority::LOWEST)
    );
    assert_eq!(Priority::new(LOWEST_THREAD_PRIORITY + 1), None);
    assert_eq!(usize::from(Priority::LOWEST.get()) + 1, PRIORITY_COUNT);
}

#[test]
fn create_thread_sets_initialize_fields() {
    let mut s = Scheduler::new(4).unwrap();
    let id = s.create_thread(prio(44), core(3)).unwrap();
    let t = s.thread(id).unwrap();
    assert_eq!(t.state(), ThreadState::Initialized);
    assert_eq!(t.priority(), prio(44));
    assert_eq!(t.base_priority(), prio(44));
    assert_eq!(t.affinity(), mask(&[3]));
    assert_eq!(t.ideal_core(), Some(core(3)));
    assert_eq!(t.active_core(), Some(core(3)));
    assert_eq!(t.current_core(), core(3));
    assert_eq!(t.suspend_flags(), 0);
    assert_eq!(t.last_scheduled_tick(), GuestTicks(0));
    assert_eq!(t.context(), &GuestRegs::default());
    // Not runnable yet, so no core picks it.
    settle(&mut s);
    assert_eq!(highest(&s, 3), None);
}

#[test]
fn create_thread_rejects_a_core_the_scheduler_lacks() {
    let mut s = Scheduler::new(4).unwrap();
    assert_eq!(
        s.create_thread(prio(1), core(4)),
        Err(kernel(KernelError::InvalidCoreId))
    );
}

#[test]
fn a_stale_thread_id_is_a_slab_error() {
    let mut s = Scheduler::new(1).unwrap();
    let id = s.create_thread(prio(1), core(0)).unwrap();
    let stale = ThreadId::from_parts(id.index(), id.generation() + 1);
    let expected = SchedError::Slab(SlabError::Stale {
        index: stale.index(),
        generation: stale.generation(),
    });
    assert_eq!(s.thread(stale).err(), Some(expected));
    assert_eq!(s.start(stale), Err(expected));
}

#[test]
fn start_runs_an_initialized_thread_once() {
    let mut s = Scheduler::new(1).unwrap();
    let id = s.create_thread(prio(10), core(0)).unwrap();
    s.start(id).unwrap();
    assert_eq!(s.start(id), Err(kernel(KernelError::InvalidState)));
    let fake = settle(&mut s);
    assert_eq!(fake.log, vec![Signal::Wake(0)]);
    assert_eq!(s.current(core(0)).unwrap(), Some(id));
}

#[test]
fn context_is_saved_per_thread() {
    let mut s = Scheduler::new(1).unwrap();
    let a = s.create_thread(prio(10), core(0)).unwrap();
    let b = s.create_thread(prio(10), core(0)).unwrap();
    s.context_mut(a).unwrap().x[0] = 0x1234;
    s.context_mut(b).unwrap().pc = 0x8000;
    assert_eq!(s.thread(a).unwrap().context().x[0], 0x1234);
    assert_eq!(s.thread(a).unwrap().context().pc, 0);
    assert_eq!(s.thread(b).unwrap().context().pc, 0x8000);
}

#[test]
fn waiting_leaves_the_queues_and_runnable_rejoins() {
    let mut s = Scheduler::new(1).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    let b = spawn(&mut s, 20, 0, &[0]);
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(a));
    s.set_state(a, ThreadState::Waiting).unwrap();
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(b));
    s.set_state(a, ThreadState::Runnable).unwrap();
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(a));
}

#[test]
fn suspend_types_are_independent() {
    let mut s = Scheduler::new(1).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    s.request_suspend(a, SuspendType::Process).unwrap();
    s.request_suspend(a, SuspendType::Debug).unwrap();
    assert_eq!(
        s.thread(a).unwrap().suspend_flags(),
        SuspendType::Process.bit() | SuspendType::Debug.bit()
    );
    settle(&mut s);
    assert_eq!(highest(&s, 0), None);
    s.resume(a, SuspendType::Process).unwrap();
    settle(&mut s);
    assert_eq!(highest(&s, 0), None, "still debug-suspended");
    s.resume(a, SuspendType::Debug).unwrap();
    settle(&mut s);
    assert_eq!(highest(&s, 0), Some(a));
}

#[test]
fn suspend_bits_are_distinct() {
    let all = SuspendType::ALL.iter().fold(0_u8, |bits, kind| {
        assert_eq!(bits & kind.bit(), 0);
        bits | kind.bit()
    });
    assert_eq!(all.count_ones(), 6);
}

#[test]
fn set_activity_pauses_and_resumes() {
    // k_thread.cpp SetActivity: state must be Waiting or Runnable; pausing requires not
    // paused, resuming requires paused; all three failures are ResultInvalidState.
    let mut s = Scheduler::new(1).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    assert_eq!(
        s.set_activity(a, ThreadActivity::Runnable),
        Err(kernel(KernelError::InvalidState))
    );
    s.set_activity(a, ThreadActivity::Paused).unwrap();
    assert!(
        s.thread(a)
            .unwrap()
            .is_suspend_requested(SuspendType::Thread)
    );
    assert_eq!(
        s.set_activity(a, ThreadActivity::Paused),
        Err(kernel(KernelError::InvalidState))
    );
    settle(&mut s);
    assert_eq!(highest(&s, 0), None);
    s.set_activity(a, ThreadActivity::Runnable).unwrap();
    settle(&mut s);
    assert_eq!(highest(&s, 0), Some(a));

    // A waiting thread can be paused too.
    s.set_state(a, ThreadState::Waiting).unwrap();
    s.set_activity(a, ThreadActivity::Paused).unwrap();
    s.set_state(a, ThreadState::Runnable).unwrap();
    settle(&mut s);
    assert_eq!(highest(&s, 0), None, "runnable but paused");

    for state in [ThreadState::Initialized, ThreadState::Terminated] {
        let b = s.create_thread(prio(10), core(0)).unwrap();
        s.set_state(b, state).unwrap();
        assert_eq!(
            s.set_activity(b, ThreadActivity::Paused),
            Err(kernel(KernelError::InvalidState))
        );
    }
}

#[test]
fn raising_a_thread_on_another_core_interrupts_that_core() {
    let mut s = Scheduler::new(2).unwrap();
    let a = spawn(&mut s, 30, 1, &[1]);
    let b = spawn(&mut s, 40, 1, &[1]);
    settle(&mut s);
    assert_eq!(s.current(core(1)).unwrap(), Some(a));
    s.set_base_priority(core(0), b, prio(20)).unwrap();
    assert_eq!(s.thread(b).unwrap().priority(), prio(20));
    let mut fake = Fake::default();
    let changed = s.update(GuestTicks(5), Some(core(0)), &mut fake).unwrap();
    assert_eq!(changed, mask(&[1]));
    assert_eq!(fake.log, vec![Signal::Interrupt(1)]);
    assert_eq!(highest(&s, 1), Some(b));
    // The thread it displaced records when it stopped being the choice.
    assert_eq!(s.thread(a).unwrap().last_scheduled_tick(), GuestTicks(5));
}

#[test]
fn update_skips_the_caller_and_wakes_parked_cores() {
    let mut s = Scheduler::new(3).unwrap();
    settle(&mut s);
    let _a = spawn(&mut s, 10, 0, &[0]);
    let _b = spawn(&mut s, 10, 2, &[2]);
    let mut fake = Fake::default();
    let changed = s.update(GuestTicks(1), Some(core(0)), &mut fake).unwrap();
    assert_eq!(changed, mask(&[0, 2]));
    assert_eq!(fake.log, vec![Signal::Wake(2)]);
    assert!(s.needs_scheduling(core(0)).unwrap());
    // Nothing changed since: no signals, nothing to report.
    let mut fake = Fake::default();
    assert_eq!(
        s.update(GuestTicks(2), None, &mut fake).unwrap(),
        AffinityMask::EMPTY
    );
    assert_eq!(fake.log, vec![]);
}

#[test]
fn a_running_thread_lowered_to_a_busy_level_keeps_running() {
    // OnThreadPriorityChanged passes is_running = (thread == current thread), and
    // ChangePriority pushes a running thread to the front.
    let mut s = Scheduler::new(1).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    let b = spawn(&mut s, 20, 0, &[0]);
    settle(&mut s);
    s.set_base_priority(core(0), a, prio(20)).unwrap();
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(a));

    // Not the caller's current thread: it goes to the back.
    let c = spawn(&mut s, 30, 0, &[0]);
    s.set_base_priority(core(0), c, prio(20)).unwrap();
    s.set_base_priority(core(0), a, prio(25)).unwrap();
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(b));
    s.set_state(b, ThreadState::Waiting).unwrap();
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(c));
}

#[test]
fn set_core_mask_moves_the_active_core() {
    // k_thread.cpp SetCoreMask: when the active core leaves the mask, the thread moves to the
    // ideal core, or with none to the highest core in the mask.
    let mut s = Scheduler::new(4).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    s.set_core_mask(a, IdealCore::Core(core(1)), mask(&[1, 2]))
        .unwrap();
    assert_eq!(s.thread(a).unwrap().active_core(), Some(core(1)));
    assert_eq!(s.thread(a).unwrap().ideal_core(), Some(core(1)));
    settle(&mut s);
    assert_eq!(highest(&s, 1), Some(a));

    s.set_core_mask(a, IdealCore::DontCare, mask(&[0, 3]))
        .unwrap();
    assert_eq!(s.thread(a).unwrap().active_core(), Some(core(3)));
    assert_eq!(s.thread(a).unwrap().ideal_core(), None);
    settle(&mut s);
    assert_eq!(highest(&s, 3), Some(a));
    assert_eq!(highest(&s, 1), None);

    // Active core still allowed: it stays, even if the ideal core changes.
    s.set_core_mask(a, IdealCore::Core(core(0)), mask(&[0, 3]))
        .unwrap();
    assert_eq!(s.thread(a).unwrap().active_core(), Some(core(3)));
    assert_eq!(s.thread(a).unwrap().ideal_core(), Some(core(0)));

    // NoUpdate keeps the ideal core, which must be in the new mask.
    s.set_core_mask(a, IdealCore::NoUpdate, mask(&[0, 2]))
        .unwrap();
    assert_eq!(s.thread(a).unwrap().ideal_core(), Some(core(0)));
    assert_eq!(s.thread(a).unwrap().active_core(), Some(core(0)));
    assert_eq!(
        s.set_core_mask(a, IdealCore::NoUpdate, mask(&[2])),
        Err(kernel(KernelError::InvalidCombination))
    );
}

#[test]
fn set_core_mask_rejects_what_eden_asserts() {
    let mut s = Scheduler::new(4).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    let before = s.thread(a).unwrap().clone();
    assert_eq!(
        s.set_core_mask(a, IdealCore::DontCare, AffinityMask::EMPTY),
        Err(kernel(KernelError::InvalidCombination))
    );
    // Core 4 is the first core this scheduler lacks.
    assert_eq!(
        s.set_core_mask(a, IdealCore::DontCare, mask(&[0, 4])),
        Err(kernel(KernelError::InvalidCoreId))
    );
    assert_eq!(
        s.set_core_mask(a, IdealCore::Core(core(2)), mask(&[0, 1])),
        Err(kernel(KernelError::InvalidCombination))
    );
    // svcSetThreadCoreMask: an ideal core that is not a valid core id is InvalidCoreId (the
    // `!IsValidVirtualCoreId` arm), even though it is also outside the mask.
    assert_eq!(
        s.set_core_mask(a, IdealCore::Core(core(4)), mask(&[0])),
        Err(kernel(KernelError::InvalidCoreId))
    );
    assert_eq!(s.thread(a).unwrap(), &before, "failures change nothing");
}

#[test]
fn no_update_with_dont_care_ideal_tests_bit_63() {
    let mut s = Scheduler::new(MAX_CORES).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    s.set_core_mask(a, IdealCore::DontCare, mask(&[0, 1]))
        .unwrap();
    assert_eq!(
        s.set_core_mask(a, IdealCore::NoUpdate, mask(&[0, 1, 2])),
        Err(kernel(KernelError::InvalidCombination))
    );
    let top = MAX_CORES - 1;
    s.set_core_mask(a, IdealCore::NoUpdate, mask(&[0, top]))
        .unwrap();
    assert_eq!(s.thread(a).unwrap().ideal_core(), None);
}

// -------------------------------------------------------------------------------------------
// Migration (k_scheduler.cpp, UpdateHighestPriorityThreadsImpl)
// -------------------------------------------------------------------------------------------

#[test]
fn an_idle_core_takes_a_suggested_thread_that_is_not_on_top() {
    // Lines 257-276: the suggested thread isn't the top thread on its core, and that top
    // thread may be migrated away from, so the suggestion moves to the idle core.
    let mut s = Scheduler::new(2).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    let b = spawn(&mut s, 20, 0, &[0, 1]);
    let fake = settle(&mut s);
    assert_eq!(highest(&s, 0), Some(a));
    assert_eq!(highest(&s, 1), Some(b));
    assert_eq!(s.thread(b).unwrap().active_core(), Some(core(1)));
    assert_eq!(s.current(core(1)).unwrap(), Some(b));
    assert_eq!(fake.log, vec![Signal::Wake(0), Signal::Wake(1)]);
}

#[test]
fn an_idle_core_takes_a_candidate_core_top_thread_when_that_core_has_another() {
    // Lines 284-306: every suggestion tops its core, so the idle core takes a candidate core's
    // top thread if that core has a next thread to run.
    let mut s = Scheduler::new(2).unwrap();
    let a = spawn(&mut s, 10, 0, &[0, 1]);
    let b = spawn(&mut s, 20, 0, &[0]);
    settle(&mut s);
    assert_eq!(highest(&s, 0), Some(b));
    assert_eq!(highest(&s, 1), Some(a));
    assert_eq!(s.thread(a).unwrap().active_core(), Some(core(1)));
}

#[test]
fn a_top_thread_alone_on_its_core_is_not_taken() {
    let mut s = Scheduler::new(2).unwrap();
    let a = spawn(&mut s, 10, 0, &[0, 1]);
    settle(&mut s);
    assert_eq!(highest(&s, 0), Some(a));
    assert_eq!(highest(&s, 1), None);
}

#[test]
fn migration_stops_at_a_core_whose_top_thread_is_too_urgent() {
    // Lines 264-268: `top_thread->GetPriority() < HighestCoreMigrationAllowedPriority` breaks.
    let allowed = HIGHEST_CORE_MIGRATION_ALLOWED_PRIORITY.get();
    let mut s = Scheduler::new(2).unwrap();
    let _top = spawn(&mut s, allowed - 1, 0, &[0]);
    let b = spawn(&mut s, 20, 0, &[0, 1]);
    settle(&mut s);
    assert_eq!(highest(&s, 1), None);
    assert_eq!(s.thread(b).unwrap().active_core(), Some(core(0)));

    // Exactly at the limit, migration is allowed.
    let mut s = Scheduler::new(2).unwrap();
    let _top = spawn(&mut s, allowed, 0, &[0]);
    let b = spawn(&mut s, 20, 0, &[0, 1]);
    settle(&mut s);
    assert_eq!(highest(&s, 1), Some(b));
}

#[test]
fn a_core_waits_for_a_thread_still_running_elsewhere() {
    let mut s = Scheduler::new(2).unwrap();
    let a = spawn(&mut s, 10, 0, &[0, 1]);
    let mut fake = Fake::default();
    s.update(GuestTicks(0), None, &mut fake).unwrap();
    assert_eq!(s.schedule(core(0), &mut fake).unwrap(), Some(a));

    // Move `a` to core 1 while core 0 still runs it.
    s.set_core_mask(a, IdealCore::Core(core(1)), mask(&[1]))
        .unwrap();
    let mut fake = Fake::default();
    s.update(GuestTicks(1), None, &mut fake).unwrap();
    assert_eq!(fake.log, vec![Signal::Interrupt(0), Signal::Wake(1)]);
    assert_eq!(s.running_on(a), Some(core(0)));

    // Core 1 gets there first: it parks, still needing scheduling.
    assert_eq!(s.schedule(core(1), &mut fake).unwrap(), None);
    assert!(s.needs_scheduling(core(1)).unwrap());

    // Core 0 lets `a` go and wakes core 1, which then runs it.
    let mut fake = Fake::default();
    assert_eq!(s.schedule(core(0), &mut fake).unwrap(), None);
    assert_eq!(fake.log, vec![Signal::Wake(1)]);
    assert_eq!(s.schedule(core(1), &mut fake).unwrap(), Some(a));
    assert_eq!(s.running_on(a), Some(core(1)));
    assert_eq!(s.thread(a).unwrap().current_core(), core(1));
}

// -------------------------------------------------------------------------------------------
// Yields (k_scheduler.cpp, YieldWithoutCoreMigration / YieldWithCoreMigration /
// YieldToAnyThread)
// -------------------------------------------------------------------------------------------

#[test]
fn yield_needs_a_current_thread_on_a_real_core() {
    let mut s = Scheduler::new(2).unwrap();
    assert_eq!(
        s.yield_without_core_migration(core(1)),
        Err(SchedError::NoCurrentThread { core: 1 })
    );
    assert_eq!(
        s.yield_to_any_thread(core(2)),
        Err(SchedError::NoSuchCore { core: 2, count: 2 })
    );
    let mut fake = Fake::default();
    assert_eq!(
        s.schedule(core(2), &mut fake),
        Err(SchedError::NoSuchCore { core: 2, count: 2 })
    );
}

#[test]
fn yield_without_migration_rotates_the_level() {
    // Lines 698-711: MoveToScheduledBack; the next thread differs, so an update is needed.
    let mut s = Scheduler::new(1).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    let b = spawn(&mut s, 10, 0, &[0]);
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(a));
    s.yield_without_core_migration(core(0)).unwrap();
    assert!(s.update_needed);
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(b));
}

#[test]
fn a_fruitless_yield_is_remembered_until_something_is_scheduled() {
    // Lines 686-689 and 707-711: alone at its level, the thread is its own next thread, so it
    // records the scheduled count and the next yield returns at once.
    let mut s = Scheduler::new(1).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    let _lower = spawn(&mut s, 11, 0, &[0]);
    settle(&mut s);
    s.yield_without_core_migration(core(0)).unwrap();
    assert!(!s.update_needed);
    let count = s.scheduled_count;
    assert_eq!(s.thread(a).unwrap().yield_schedule_count, Some(count));
    s.yield_without_core_migration(core(0)).unwrap();
    assert_eq!(s.scheduled_count, count, "redundant yield did no work");
}

#[test]
fn yield_count_is_one_for_the_table() {
    // A change to any thread (here one that will never share a level with `a`) makes the next
    // yield do work again. In Eden only a thread of the same process would.
    let mut s = Scheduler::new(2).unwrap();
    let _a = spawn(&mut s, 10, 0, &[0]);
    settle(&mut s);
    s.yield_without_core_migration(core(0)).unwrap();
    let count = s.scheduled_count;
    let other = s.create_thread(prio(50), core(1)).unwrap();
    s.start(other).unwrap();
    assert_ne!(s.scheduled_count, count);
    let before = s.scheduled_count;
    s.yield_without_core_migration(core(0)).unwrap();
    assert_eq!(s.scheduled_count, before + 1, "the yield ran");
}

#[test]
fn yield_with_migration_pulls_an_equal_suggestion_to_the_front() {
    // Lines 746-779: the suggestion isn't running on its core and is as urgent as the yielding
    // thread, and the yielding thread is its own next thread, so it migrates to the front.
    let mut s = Scheduler::new(2).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    let _c = spawn(&mut s, 5, 1, &[1]);
    let d = spawn(&mut s, 10, 1, &[0, 1]);
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(a));
    s.yield_with_core_migration(core(0)).unwrap();
    assert_eq!(s.thread(d).unwrap().active_core(), Some(core(0)));
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(d));
}

#[test]
fn yield_with_migration_ignores_a_less_urgent_suggestion() {
    // Lines 762-768: a suggestion less urgent than the yielding thread ends the search.
    let mut s = Scheduler::new(2).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    let _c = spawn(&mut s, 5, 1, &[1]);
    let d = spawn(&mut s, 11, 1, &[0, 1]);
    settle(&mut s);
    s.yield_with_core_migration(core(0)).unwrap();
    assert_eq!(s.thread(d).unwrap().active_core(), Some(core(1)));
    assert!(!s.update_needed);
    assert_eq!(
        s.thread(a).unwrap().yield_schedule_count,
        Some(s.scheduled_count)
    );
}

#[test]
fn yield_with_migration_prefers_a_next_thread_that_waited_longer() {
    // Lines 762-765: at equal priority, if the next thread on this core stopped being chosen
    // earlier than the suggestion did, keep the next thread.
    let mut s = Scheduler::new(2).unwrap();
    let _a = spawn(&mut s, 10, 0, &[0]);
    let n = spawn(&mut s, 10, 0, &[0]);
    let _c = spawn(&mut s, 5, 1, &[1]);
    let d = spawn(&mut s, 10, 1, &[0, 1]);
    settle(&mut s);
    s.threads.get_mut(n).unwrap().last_scheduled_tick = GuestTicks(1);
    s.threads.get_mut(d).unwrap().last_scheduled_tick = GuestTicks(2);
    s.yield_with_core_migration(core(0)).unwrap();
    assert_eq!(s.thread(d).unwrap().active_core(), Some(core(1)));
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(n));

    // With the ticks the other way round, the suggestion migrates.
    let mut s = Scheduler::new(2).unwrap();
    let _a = spawn(&mut s, 10, 0, &[0]);
    let n = spawn(&mut s, 10, 0, &[0]);
    let _c = spawn(&mut s, 5, 1, &[1]);
    let d = spawn(&mut s, 10, 1, &[0, 1]);
    settle(&mut s);
    s.threads.get_mut(n).unwrap().last_scheduled_tick = GuestTicks(2);
    s.threads.get_mut(d).unwrap().last_scheduled_tick = GuestTicks(1);
    s.yield_with_core_migration(core(0)).unwrap();
    assert_eq!(s.thread(d).unwrap().active_core(), Some(core(0)));
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(d));
}

#[test]
fn yield_with_migration_rechecks_when_the_suggestions_core_is_too_urgent() {
    // Lines 770-784: the suggestion's core runs a thread above the migration limit, so nothing
    // moves, and the yield count is not recorded (`recheck`).
    let allowed = HIGHEST_CORE_MIGRATION_ALLOWED_PRIORITY.get();
    let mut s = Scheduler::new(2).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    let _c = spawn(&mut s, allowed - 1, 1, &[1]);
    let d = spawn(&mut s, 10, 1, &[0, 1]);
    settle(&mut s);
    s.yield_with_core_migration(core(0)).unwrap();
    assert_eq!(s.thread(d).unwrap().active_core(), Some(core(1)));
    assert!(!s.update_needed);
    assert_eq!(s.thread(a).unwrap().yield_schedule_count, None);
}

#[test]
fn yield_to_any_thread_gives_the_core_to_a_suggestion() {
    // Lines 830-866: the yielding thread loses its core; the core has nothing scheduled, so the
    // first suggestion that doesn't top its own core migrates in.
    let mut s = Scheduler::new(2).unwrap();
    let a = spawn(&mut s, 10, 0, &[0, 1]);
    let b = spawn(&mut s, 10, 1, &[1]);
    // As urgent as `a`, and suggested to core 0 before `a` is.
    let c = spawn(&mut s, 10, 1, &[0, 1]);
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(a));
    assert_eq!(s.current(core(1)).unwrap(), Some(b));
    s.yield_to_any_thread(core(0)).unwrap();
    assert_eq!(s.thread(a).unwrap().active_core(), None);
    assert_eq!(s.thread(c).unwrap().active_core(), Some(core(0)));
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(c));
    assert_eq!(s.current(core(1)).unwrap(), Some(b));
}

#[test]
fn yield_to_any_thread_alone_takes_its_core_back() {
    // Lines 863-871: the only suggestion is the yielding thread itself, which migrates back,
    // and the yield is recorded as fruitless.
    let mut s = Scheduler::new(2).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    settle(&mut s);
    s.yield_to_any_thread(core(0)).unwrap();
    assert_eq!(s.thread(a).unwrap().active_core(), Some(core(0)));
    assert!(!s.update_needed);
    assert_eq!(
        s.thread(a).unwrap().yield_schedule_count,
        Some(s.scheduled_count)
    );
}

#[test]
fn yield_to_any_thread_with_work_left_on_the_core_just_updates() {
    // Lines 872-875: something else is scheduled on the core, so no migration is tried.
    let mut s = Scheduler::new(1).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    let b = spawn(&mut s, 30, 0, &[0]);
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(a));
    s.yield_to_any_thread(core(0)).unwrap();
    assert!(s.update_needed);
    settle(&mut s);
    assert_eq!(s.current(core(0)).unwrap(), Some(b));
    // `a` is only suggested now, and core 0 is busy.
    assert_eq!(s.thread(a).unwrap().active_core(), None);
}

#[test]
fn yield_without_active_core_is_a_no_op() {
    let mut s = Scheduler::new(1).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    let _b = spawn(&mut s, 30, 0, &[0]);
    settle(&mut s);
    s.yield_to_any_thread(core(0)).unwrap();
    assert_eq!(s.thread(a).unwrap().active_core(), None);
    // Still current on core 0 until it reschedules.
    let snapshot = s.clone();
    s.yield_without_core_migration(core(0)).unwrap();
    s.yield_with_core_migration(core(0)).unwrap();
    s.yield_to_any_thread(core(0)).unwrap();
    assert_eq!(s, snapshot);
}

#[test]
fn a_yield_by_a_thread_that_is_not_runnable_does_nothing() {
    let mut s = Scheduler::new(1).unwrap();
    let a = spawn(&mut s, 10, 0, &[0]);
    let _b = spawn(&mut s, 10, 0, &[0]);
    settle(&mut s);
    s.set_state(a, ThreadState::Waiting).unwrap();
    let snapshot = s.clone();
    s.yield_without_core_migration(core(0)).unwrap();
    s.yield_with_core_migration(core(0)).unwrap();
    s.yield_to_any_thread(core(0)).unwrap();
    assert_eq!(s, snapshot);
}

// -------------------------------------------------------------------------------------------
// Invariants under random operations
// -------------------------------------------------------------------------------------------

const PROP_CORES: u8 = 4;
const PROP_THREADS: usize = 10;
/// Fixed seed so a run is reproducible; failures are not persisted to disk.
const PROPTEST_SEED: u64 = 0x5343_4845_4455_4C45;

#[derive(Clone, Debug)]
enum Op {
    Create {
        priority: u8,
        core: u8,
    },
    Start(usize),
    SetState(usize, ThreadState),
    Suspend(usize, usize),
    Resume(usize, usize),
    Activity(usize, bool),
    SetPriority {
        caller: u8,
        thread: usize,
        priority: u8,
    },
    SetCoreMask {
        thread: usize,
        ideal: u8,
        mask: u64,
    },
    YieldPlain(u8),
    YieldMigrate(u8),
    YieldAny(u8),
    Schedule(u8),
}

fn priority_strategy() -> impl Strategy<Value = u8> {
    // Bias towards a few levels so threads share levels and hit the migration limit.
    prop_oneof![
        0_u8..4,
        8_u8..11,
        HIGHEST_THREAD_PRIORITY..=LOWEST_THREAD_PRIORITY
    ]
}

fn op_strategy() -> impl Strategy<Value = Op> {
    let t = 0..PROP_THREADS;
    let c = 0..PROP_CORES;
    let state = prop_oneof![
        Just(ThreadState::Waiting),
        Just(ThreadState::Runnable),
        Just(ThreadState::Terminated),
    ];
    prop_oneof![
        3 => (priority_strategy(), c.clone()).prop_map(|(priority, core)| Op::Create { priority, core }),
        3 => t.clone().prop_map(Op::Start),
        2 => (t.clone(), state).prop_map(|(i, s)| Op::SetState(i, s)),
        1 => (t.clone(), 0..SuspendType::ALL.len()).prop_map(|(i, k)| Op::Suspend(i, k)),
        1 => (t.clone(), 0..SuspendType::ALL.len()).prop_map(|(i, k)| Op::Resume(i, k)),
        1 => (t.clone(), any::<bool>()).prop_map(|(i, p)| Op::Activity(i, p)),
        2 => (c.clone(), t.clone(), priority_strategy())
            .prop_map(|(caller, thread, priority)| Op::SetPriority { caller, thread, priority }),
        2 => (t, 0..PROP_CORES + 2, 0_u64..(1 << PROP_CORES))
            .prop_map(|(thread, ideal, mask)| Op::SetCoreMask { thread, ideal, mask }),
        2 => c.clone().prop_map(Op::YieldPlain),
        2 => c.clone().prop_map(Op::YieldMigrate),
        2 => c.clone().prop_map(Op::YieldAny),
        3 => c.prop_map(Op::Schedule),
    ]
}

/// Errors a random operation may legitimately produce.
fn allowed(result: Result<(), SchedError>) {
    assert!(
        matches!(
            result,
            Ok(())
                | Err(SchedError::Kernel(
                    KernelError::InvalidState
                        | KernelError::InvalidCombination
                        | KernelError::InvalidCoreId,
                ) | SchedError::NoCurrentThread { .. })
        ),
        "unexpected result {result:?}"
    );
}

fn apply(s: &mut Scheduler, ids: &mut Vec<ThreadId>, op: &Op, fake: &mut Fake) {
    let pick = |ids: &[ThreadId], i: usize| ids.get(i % ids.len().max(1)).copied();
    match *op {
        Op::Create { priority, core: c } => {
            if ids.len() < PROP_THREADS {
                ids.push(s.create_thread(prio(priority), core(c)).unwrap());
            }
        }
        Op::Start(i) => {
            if let Some(id) = pick(ids, i) {
                allowed(s.start(id));
            }
        }
        Op::SetState(i, state) => {
            if let Some(id) = pick(ids, i) {
                allowed(s.set_state(id, state));
            }
        }
        Op::Suspend(i, k) => {
            if let Some(id) = pick(ids, i) {
                allowed(s.request_suspend(id, SuspendType::ALL[k]));
            }
        }
        Op::Resume(i, k) => {
            if let Some(id) = pick(ids, i) {
                allowed(s.resume(id, SuspendType::ALL[k]));
            }
        }
        Op::Activity(i, pause) => {
            if let Some(id) = pick(ids, i) {
                let activity = if pause {
                    ThreadActivity::Paused
                } else {
                    ThreadActivity::Runnable
                };
                allowed(s.set_activity(id, activity));
            }
        }
        Op::SetPriority {
            caller,
            thread,
            priority,
        } => {
            if let Some(id) = pick(ids, thread) {
                allowed(s.set_base_priority(core(caller), id, prio(priority)));
            }
        }
        Op::SetCoreMask {
            thread,
            ideal,
            mask,
        } => {
            if let Some(id) = pick(ids, thread) {
                let ideal = match ideal {
                    i if i < PROP_CORES => IdealCore::Core(core(i)),
                    i if i == PROP_CORES => IdealCore::DontCare,
                    _ => IdealCore::NoUpdate,
                };
                allowed(s.set_core_mask(id, ideal, AffinityMask::from_bits(mask)));
            }
        }
        Op::YieldPlain(c) => allowed(s.yield_without_core_migration(core(c))),
        Op::YieldMigrate(c) => allowed(s.yield_with_core_migration(core(c))),
        Op::YieldAny(c) => allowed(s.yield_to_any_thread(core(c))),
        Op::Schedule(c) => {
            s.schedule(core(c), fake).unwrap();
        }
    }
}

/// Every list holds exactly the threads it should, each once, and the level bitmaps agree.
fn check_queues(s: &Scheduler) {
    let live: Vec<(ThreadId, &Thread)> = s.threads.iter().collect();
    for c in 0..s.core_count() {
        for level in 0..=LOWEST_THREAD_PRIORITY {
            let p = prio(level);
            let mut scheduled = s.queue.scheduled_list(&s.threads, p, core(c)).unwrap();
            let mut suggested = s.queue.suggested_list(&s.threads, p, core(c)).unwrap();
            assert_eq!(
                s.queue.is_available(true, p, core(c)),
                !scheduled.is_empty()
            );
            assert_eq!(
                s.queue.is_available(false, p, core(c)),
                !suggested.is_empty()
            );
            let mut want_scheduled: Vec<ThreadId> = live
                .iter()
                .filter(|(_, t)| t.is_queued() && t.priority == p && t.active_core == Some(core(c)))
                .map(|(id, _)| *id)
                .collect();
            let mut want_suggested: Vec<ThreadId> = live
                .iter()
                .filter(|(_, t)| {
                    t.is_queued()
                        && t.priority == p
                        && t.affinity.contains(core(c))
                        && t.active_core != Some(core(c))
                })
                .map(|(id, _)| *id)
                .collect();
            for list in [
                &mut scheduled,
                &mut suggested,
                &mut want_scheduled,
                &mut want_suggested,
            ] {
                list.sort();
            }
            assert_eq!(
                scheduled, want_scheduled,
                "scheduled core {c} priority {level}"
            );
            assert_eq!(
                suggested, want_suggested,
                "suggested core {c} priority {level}"
            );
        }
    }
}

/// `UpdateHighestPriorityThreadsImpl`'s migration walk, checked on the settled state: walking an
/// idle core's suggested threads in queue order (most urgent level first, then list order),
/// skipping those that are their own core's choice, the first thread that is not its core's
/// choice would have been migrated here unless that core's choice is more urgent than
/// [`HIGHEST_CORE_MIGRATION_ALLOWED_PRIORITY`] (`k_scheduler.cpp`, the `break` before
/// `SetActiveCore`). A walk that runs out of threads has nothing to assert: the fallback that
/// moves a candidate core's top thread depends on what else those cores can run.
fn check_idle_core_walk(s: &Scheduler, chosen: &[Option<ThreadId>], idle: CoreId) {
    for level in 0..=LOWEST_THREAD_PRIORITY {
        for id in s
            .queue
            .suggested_list(&s.threads, prio(level), idle)
            .unwrap()
        {
            let active = s.thread(id).unwrap().active_core;
            let top = active.and_then(|c| chosen[usize::from(c.index())]);
            if top == Some(id) {
                continue;
            }
            assert!(
                top.is_some(),
                "idle core {idle:?} left {id:?} unmigrated with no top thread to block it"
            );
            let blocker = top.unwrap();
            assert!(
                s.thread(blocker).unwrap().priority < HIGHEST_CORE_MIGRATION_ALLOWED_PRIORITY,
                "idle core {idle:?} skipped {id:?}, whose core's choice {blocker:?} is not too urgent to leave"
            );
            return;
        }
    }
}

/// The acceptance invariants after an update: each core's choice is its most urgent scheduled
/// thread and is allowed there, no thread is chosen twice, and a core is idle only when no
/// allowed thread could be migrated to it.
fn check_choices(s: &Scheduler) {
    let live: Vec<(ThreadId, &Thread)> = s.threads.iter().collect();
    for (_, t) in &live {
        if let Some(active) = t.active_core {
            assert!(t.affinity.contains(active), "active core outside the mask");
        }
    }
    let chosen: Vec<Option<ThreadId>> = (0..s.core_count()).map(|c| highest(s, c)).collect();
    for c in 0..s.core_count() {
        let here = core(c);
        if let Some(id) = chosen[usize::from(c)] {
            let t = s.thread(id).unwrap();
            assert!(t.is_queued(), "core {c} chose a thread that can't run");
            assert!(
                t.affinity.contains(here),
                "core {c} chose a thread not allowed there"
            );
            assert_eq!(t.active_core, Some(here));
            for (_, other) in &live {
                if other.is_queued() && other.active_core == Some(here) {
                    assert!(
                        t.priority <= other.priority,
                        "core {c} skipped a more urgent thread"
                    );
                }
            }
            let times = chosen.iter().filter(|&&x| x == Some(id)).count();
            assert_eq!(times, 1, "a thread chosen by two cores");
        } else {
            for (_, other) in &live {
                if other.is_queued() {
                    assert_ne!(other.active_core, Some(here), "core {c} idle with work");
                }
            }
            check_idle_core_walk(s, &chosen, here);
        }
    }
}

/// After every core has switched: each runs its choice, and no thread runs on two cores.
fn check_running(s: &Scheduler) {
    for c in 0..s.core_count() {
        assert_eq!(
            s.current(core(c)).unwrap(),
            highest(s, c),
            "core {c} runs its choice"
        );
        assert!(!s.needs_scheduling(core(c)).unwrap());
    }
    let running: Vec<ThreadId> = (0..s.core_count())
        .filter_map(|c| s.current(core(c)).unwrap())
        .collect();
    let mut unique = running.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), running.len(), "a thread runs on two cores");
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 256,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(PROPTEST_SEED),
        failure_persistence: None,
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn invariants_hold_under_random_operations(ops in prop::collection::vec(op_strategy(), 1..80)) {
        let mut s = Scheduler::new(PROP_CORES).unwrap();
        let mut ids = Vec::new();
        for (step, op) in ops.iter().enumerate() {
            let mut fake = Fake::default();
            apply(&mut s, &mut ids, op, &mut fake);
            check_queues(&s);
            let now = GuestTicks(u64::try_from(step).unwrap() + 1);
            let changed = s.update(now, None, &mut fake).unwrap();
            for c in changed.iter() {
                let signalled = fake.log.contains(&Signal::Interrupt(c.index()))
                    || fake.log.contains(&Signal::Wake(c.index()));
                prop_assert!(signalled, "core {} changed but was not signalled", c.index());
            }
            check_queues(&s);
            check_choices(&s);
            // Every so often, let all cores catch up.
            if step % 7 == 6 {
                settle_at(&mut s, now.0);
                check_running(&s);
            }
        }
        settle_at(&mut s, u64::try_from(ops.len()).unwrap() + 1);
        check_queues(&s);
        check_choices(&s);
        check_running(&s);
    }
}
