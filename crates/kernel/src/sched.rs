//! Threads, priority queues and the scheduler: the thread half of the [`Scheduler`] table
//! (kernel spec, "Scheduler"; port slice 3).
//!
//! Ported decision for decision from Eden @ cf654052 (GPL-3.0-or-later):
//! `eden/src/core/hle/kernel/k_priority_queue.h` (the private `queue` module),
//! `k_scheduler.{h,cpp}` ([`Scheduler`]), the scheduler-facing parts of `k_thread.{h,cpp}`
//! ([`Thread`]) and `k_affinity_mask.h` ([`AffinityMask`]). Eden's structure is not ported:
//! threads are rows in a [`Slab`](crate::slab::Slab) named by [`ThreadId`](crate::id::ThreadId),
//! the queues link rows by id, and nothing locks. The dispatcher holds the kernel lock around a
//! whole system call and calls [`Scheduler::update`] before releasing it, which is where Eden's
//! `KAbstractSchedulerLock::Unlock` recomputes each core's thread and interrupts the others.
//!
//! # How the cores use it
//!
//! A core thread calls [`Scheduler::schedule`] when [`Scheduler::needs_scheduling`] says so, and
//! runs the thread it returns (or parks if it returns `None`). The scheduler reaches the other
//! cores through [`CoreSignals`]: `interrupt` for a core running guest code, `wake` for a parked
//! one. The calling core is not signalled; it checks [`Scheduler::needs_scheduling`] itself, as
//! Eden's `EnableScheduling` reschedules the current core directly and only interrupts the
//! others (`RescheduleOtherCores`).
//!
//! # Built here, and not yet
//!
//! Built: the object table ([`Scheduler::objects`], see [`crate::object`]); thread state (run
//! state, suspend requests, priority, affinity, ideal core, saved
//! context as a [`GuestRegs`](starquake_cpu::GuestRegs) placeholder), the scheduled and
//! suggested queues over 64 priorities, choosing each core's thread and migrating suggested
//! threads to idle cores, the three yields, base-priority and core-mask changes, thread
//! activity (pause and resume), and the switch bookkeeping a core does when it changes threads.
//!
//! Left for later slices, each with the Eden code that needs it:
//! - Priority inheritance (`RestorePriority` over held locks, `m_num_kernel_waiters` in
//!   `TrySuspend` and `UpdateHighestPriorityThreadsImpl`): slice 5. Here a thread's priority is
//!   always its base priority, and `TrySuspend` always suspends.
//! - Pinned threads (`GetPinnedThread`, `m_num_core_migration_disables`, the original affinity
//!   saved while migration is disabled, `m_suspend_allowed_flags`): not in the spec's slices yet.
//! - The waits in `SetCoreMask` and `SetActivity` after the scheduler lock is released (until
//!   the thread is no longer running on a core it may not use, or stops running): the system
//!   call layer, using [`Scheduler::running_on`].
//! - Virtual-to-physical core translation (`VirtualToPhysicalCoreMap`) and the process core
//!   mask checks of `svcSetThreadCoreMask`: the system call layer. Cores here are physical.
//! - `RotateScheduledQueue` (time slicing), `Exit`, `Terminate`: slices 8 and 10.
//! - Serialising the table for snapshots (observability spec): a follow-up, because a
//!   deserialised queue must be validated link by link.
//!
//! # Intentional divergences from Eden
//!
//! Each is pinned by the named test in `sched/scheduler/tests.rs`.
//! - Eden counts `IncrementScheduledCount` per process (`KProcess::m_schedule_count`) and
//!   compares it with a thread's yield count. There is one count for the whole table here; with
//!   one process the two are the same. Test: `yield_count_is_one_for_the_table`.
//! - A yield by a thread that has no active core (possible only between `YieldToAnyThread` and
//!   the core's next [`Scheduler::schedule`]) does nothing. Eden indexes the queues with core
//!   `-1` there, which its `ASSERT(IsValidCore(core))` rejects. Test: `yield_without_active_core_is_a_no_op`.
//! - Arguments Eden asserts on are errors: a priority outside 0..=63 cannot be built, a core the
//!   scheduler doesn't have is [`KernelError::InvalidCoreId`] (in an affinity mask, and as the
//!   ideal core of `set_core_mask`), an empty core mask [`KernelError::InvalidCombination`] (the
//!   checks `svcSetThreadCoreMask` makes first).
//!   Test: `set_core_mask_rejects_what_eden_asserts`.
//! - `SetCoreMask` with `IdealCoreNoUpdate` when the ideal core is "don't care" (-1): Eden
//!   evaluates `1ULL << -1`, which C++ leaves undefined. Here it tests bit 63, what AArch64's
//!   variable shift (count taken mod 64) computes. A guess; a hardware test of
//!   `svcSetThreadCoreMask(handle, -3, mask)` after `(handle, -1, mask)` would confirm it.
//!   Test: `no_update_with_dont_care_ideal_tests_bit_63`.
//! - Two cores choosing the same thread: Eden's core spins on the thread's context lock until the
//!   other core unloads it. Here [`Scheduler::schedule`] returns `None` (park) and keeps the core
//!   marked as needing scheduling, and the core that lets the thread go wakes it.
//!   Test: `a_core_waits_for_a_thread_still_running_elsewhere`.

mod cores;
mod queue;
mod scheduler;
mod thread;

use thiserror::Error;

pub use cores::{
    APPLICATION_CORE_COUNT, AffinityMask, CoreId, CoreIdOutOfRange, CoreSignals, MAX_CORES,
};
pub use scheduler::{FIRST_THREAD_ID, Scheduler, ThreadExit, UserThread};
pub use thread::{
    HIGHEST_THREAD_PRIORITY, LOWEST_THREAD_PRIORITY, PRIORITY_COUNT, Priority, SuspendType, Thread,
    ThreadState,
};

use crate::id::ProcessId;
use crate::result::{KernelError, SvcError};
use crate::slab::{SlabError, StoredIdStale};

/// Threads more urgent than this are never migrated off the core they top: Eden
/// `global_scheduler_context.h`, `HighestCoreMigrationAllowedPriority = 2`.
pub const HIGHEST_CORE_MIGRATION_ALLOWED_PRIORITY: Priority = match Priority::new(2) {
    Some(priority) => priority,
    // Unreachable: 2 is a valid priority. A `const` can't unwrap, so the match needs an arm.
    None => Priority::HIGHEST,
};

/// The ideal-core argument of [`Scheduler::set_core_mask`]: Eden's `IdealCoreNoUpdate` (-3),
/// `IdealCoreDontCare` (-1) and a core id (`svc_types.h`). `IdealCoreUseProcessValue` (-2) is
/// resolved by the system call before it reaches the scheduler, as Eden's `SetThreadCoreMask`
/// does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IdealCore {
    /// Keep the current ideal core; it must be in the new mask.
    NoUpdate,
    /// No preference.
    DontCare,
    /// This core, which must be in the new mask.
    Core(CoreId),
}

/// The argument of [`Scheduler::set_activity`]: Eden's `Svc::ThreadActivity` (`svc_types.h`:
/// `Runnable = 0`, `Paused = 1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ThreadActivity {
    /// Resume a paused thread.
    Runnable,
    /// Pause the thread.
    Paused,
}

/// Why a scheduler operation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SchedError {
    /// A result the guest sees, with Eden's code for the same condition.
    #[error(transparent)]
    Kernel(#[from] KernelError),
    /// The thread id names no live thread, or the thread table is full.
    #[error(transparent)]
    Slab(#[from] SlabError),
    /// A core index the scheduler doesn't have. The core loop passes its own index, so this is a
    /// host bug, not a guest error.
    #[error("core {core} is not one of the scheduler's {count} cores")]
    NoSuchCore {
        /// The core asked for.
        core: u8,
        /// How many cores the scheduler has.
        count: u8,
    },
    /// A core count of zero or above [`MAX_CORES`].
    #[error("a scheduler needs 1 to {MAX_CORES} cores, not {count}")]
    CoreCount {
        /// The count asked for.
        count: u8,
    },
    /// A yield came from a core with no current thread.
    #[error("core {core} has no current thread")]
    NoCurrentThread {
        /// The calling core.
        core: u8,
    },
    /// The queue links are inconsistent. Unreachable through the public API; it exists so that
    /// damaged bookkeeping fails instead of panicking.
    #[error("scheduler bookkeeping is inconsistent")]
    Corrupt,
    /// A process id the scheduler has no row for: the kernel created the process without
    /// telling the scheduler, or named the wrong one.
    #[error("the scheduler has no process {0:?}")]
    UnknownProcess(ProcessId),
}

impl SchedError {
    /// The Horizon result this error is to a guest, if it has one: only
    /// [`SchedError::Kernel`] does. Every other case is the kernel or the core loop misusing the
    /// table (a stale thread id, a core that does not exist, damaged links), which no guest
    /// input reaches.
    #[must_use]
    pub const fn kernel_error(&self) -> Option<KernelError> {
        match self {
            Self::Kernel(error) => Some(*error),
            Self::Slab(_)
            | Self::NoSuchCore { .. }
            | Self::CoreCount { .. }
            | Self::NoCurrentThread { .. }
            | Self::Corrupt
            | Self::UnknownProcess(_) => None,
        }
    }
}

impl From<SchedError> for SvcError {
    /// The one place a scheduler failure becomes what a system call returns: its Horizon result
    /// where it has one ([`SchedError::kernel_error`]), an internal failure otherwise.
    fn from(error: SchedError) -> Self {
        match error.kernel_error() {
            Some(result) => Self::Kernel(result),
            None => Self::internal(&error),
        }
    }
}

impl From<StoredIdStale> for SchedError {
    /// A stale id in a queue link or a core's state is damaged bookkeeping.
    fn from(_: StoredIdStale) -> Self {
        Self::Corrupt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant, with what a guest sees for it.
    fn cases() -> Vec<(SchedError, Option<KernelError>)> {
        vec![
            (
                SchedError::Kernel(KernelError::InvalidState),
                Some(KernelError::InvalidState),
            ),
            (
                SchedError::Kernel(KernelError::InvalidCoreId),
                Some(KernelError::InvalidCoreId),
            ),
            (
                SchedError::Slab(SlabError::Stale {
                    index: 3,
                    generation: 2,
                }),
                None,
            ),
            (SchedError::NoSuchCore { core: 9, count: 6 }, None),
            (SchedError::CoreCount { count: 0 }, None),
            (SchedError::NoCurrentThread { core: 1 }, None),
            (SchedError::Corrupt, None),
            (
                SchedError::UnknownProcess(crate::id::SlabId::from_parts(2, 7)),
                None,
            ),
        ]
    }

    #[test]
    fn only_a_kernel_result_reaches_the_guest_and_the_rest_are_internal() {
        for (error, expected) in cases() {
            assert_eq!(error.kernel_error(), expected, "{error}");
            let call = SvcError::from(error);
            match expected {
                Some(result) => assert_eq!(call, SvcError::Kernel(result)),
                None => assert_eq!(call, SvcError::Internal(error.to_string())),
            }
        }
    }

    #[test]
    fn a_stale_stored_id_is_damaged_bookkeeping() {
        assert_eq!(SchedError::from(StoredIdStale), SchedError::Corrupt);
        assert_eq!(
            SvcError::from(SchedError::from(StoredIdStale)),
            SvcError::Internal("scheduler bookkeeping is inconsistent".to_owned())
        );
    }
}
