//! A thread's row in the scheduler's thread table: run state, suspend requests, priorities,
//! affinity, the cores it is bound to and running on, its links in the priority queues, and its
//! saved register context.
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_thread.h` @ cf654052 (GPL-3.0-or-later): the
//! fields `KThread` keeps for the scheduler (`m_thread_state`, `m_suspend_request_flags`,
//! `m_priority`, `m_base_priority`, `m_physical_affinity_mask`, `m_physical_ideal_core_id`,
//! `m_core_id`, `m_current_core_id`, `m_last_scheduled_tick`, `m_schedule_count`, and the
//! per-core `QueueEntry` array). Fields for later slices (wait queue, held locks, TLS, the
//! owning process) are added by those slices.

use starquake_cpu::GuestRegs;
use starquake_trace::GuestTicks;

use super::cores::{AffinityMask, CoreId};
use crate::id::ThreadId;

/// The highest (most urgent) thread priority: Eden `svc_types.h`, `HighestThreadPriority = 0`.
pub const HIGHEST_THREAD_PRIORITY: u8 = 0;

/// The lowest thread priority: Eden `svc_types.h`, `LowestThreadPriority = 63`.
pub const LOWEST_THREAD_PRIORITY: u8 = 63;

/// The number of priority levels the queues hold (`k_priority_queue.h`,
/// `NumPriority = LowestPriority - HighestPriority + 1`).
pub const PRIORITY_COUNT: usize = (LOWEST_THREAD_PRIORITY - HIGHEST_THREAD_PRIORITY + 1) as usize;

/// A thread priority, from [`HIGHEST_THREAD_PRIORITY`] to [`LOWEST_THREAD_PRIORITY`]. A smaller
/// number is more urgent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(u8);

impl Priority {
    /// The most urgent priority.
    pub const HIGHEST: Self = Self(HIGHEST_THREAD_PRIORITY);
    /// The least urgent priority.
    pub const LOWEST: Self = Self(LOWEST_THREAD_PRIORITY);

    /// The priority with this value, or `None` if it is outside the range Eden's
    /// `KThread::Initialize` and `SetBasePriority` assert.
    #[must_use]
    pub const fn new(value: u8) -> Option<Self> {
        if value <= LOWEST_THREAD_PRIORITY {
            Some(Self(value))
        } else {
            None
        }
    }

    /// The numeric value.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// The priority as an index into the queue levels.
    pub(super) const fn level(self) -> usize {
        (self.0 - HIGHEST_THREAD_PRIORITY) as usize
    }
}

/// A thread's run state without its suspend flags: Eden's `ThreadState` values below
/// `SuspendShift` (`k_thread.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ThreadState {
    /// Created, not yet started.
    Initialized,
    /// Blocked on something.
    Waiting,
    /// Ready to run (and in the priority queues, unless suspended).
    Runnable,
    /// Finished.
    Terminated,
}

/// Why a thread is suspended: Eden's `SuspendType` (`k_thread.h`). A thread is suspended while
/// any request is outstanding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SuspendType {
    /// Its process is suspended.
    Process,
    /// `SetThreadActivity(Paused)`.
    Thread,
    /// A debugger stopped it.
    Debug,
    /// A backtrace is being taken.
    Backtrace,
    /// Held at initialisation.
    Init,
    /// The system suspended it.
    System,
}

impl SuspendType {
    /// Every suspend type, in Eden's order.
    pub const ALL: [Self; 6] = [
        Self::Process,
        Self::Thread,
        Self::Debug,
        Self::Backtrace,
        Self::Init,
        Self::System,
    ];

    /// This type's bit in [`Thread::suspend_flags`]. Eden keeps the same bits shifted up by
    /// `ThreadState::SuspendShift` (`1 << (SuspendShift + type)`); the shift only exists there
    /// because the flags share a word with the run state, so it is left out here.
    #[must_use]
    pub const fn bit(self) -> u8 {
        1 << (self as u8)
    }
}

/// A thread's place in one core's queue at one priority: the neighbours before and after it.
/// Eden's `KThread::QueueEntry`; one per core, shared by the scheduled and suggested queues
/// because a thread is in at most one of them per core.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct QueueLink {
    pub(super) prev: Option<ThreadId>,
    pub(super) next: Option<ThreadId>,
}

/// One row of the thread table. Read it with [`Scheduler::thread`](super::Scheduler::thread);
/// it changes only through the scheduler's operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Thread {
    pub(super) state: ThreadState,
    pub(super) suspend_flags: u8,
    pub(super) priority: Priority,
    pub(super) base_priority: Priority,
    pub(super) affinity: AffinityMask,
    pub(super) ideal_core: Option<CoreId>,
    pub(super) active_core: Option<CoreId>,
    pub(super) current_core: CoreId,
    pub(super) last_scheduled_tick: GuestTicks,
    pub(super) yield_schedule_count: Option<u64>,
    pub(super) links: Vec<QueueLink>,
    pub(super) context: GuestRegs,
}

impl Thread {
    /// The run state, without suspension.
    #[must_use]
    pub fn state(&self) -> ThreadState {
        self.state
    }

    /// The outstanding suspend requests, one [`SuspendType::bit`] each.
    #[must_use]
    pub fn suspend_flags(&self) -> u8 {
        self.suspend_flags
    }

    /// Whether a suspend of this type is requested (`KThread::IsSuspendRequested(type)`).
    #[must_use]
    pub fn is_suspend_requested(&self, kind: SuspendType) -> bool {
        self.suspend_flags & kind.bit() != 0
    }

    /// The effective priority (`GetPriority`).
    #[must_use]
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// The priority the thread asked for (`GetBasePriority`). Equal to [`Thread::priority`]
    /// until priority inheritance (slice 5) raises the effective one.
    #[must_use]
    pub fn base_priority(&self) -> Priority {
        self.base_priority
    }

    /// The cores the thread may run on.
    #[must_use]
    pub fn affinity(&self) -> AffinityMask {
        self.affinity
    }

    /// The preferred core, or `None` for "don't care" (`IdealCoreDontCare`).
    #[must_use]
    pub fn ideal_core(&self) -> Option<CoreId> {
        self.ideal_core
    }

    /// The core whose scheduled queue holds the thread (`GetActiveCore`), or `None` when it is
    /// only suggested to cores (after `YieldToAnyThread`).
    #[must_use]
    pub fn active_core(&self) -> Option<CoreId> {
        self.active_core
    }

    /// The core the thread last ran on (`GetCurrentCore`).
    #[must_use]
    pub fn current_core(&self) -> CoreId {
        self.current_core
    }

    /// When the thread last stopped being some core's choice (`GetLastScheduledTick`).
    #[must_use]
    pub fn last_scheduled_tick(&self) -> GuestTicks {
        self.last_scheduled_tick
    }

    /// The saved register context: the state a core loads when it switches to this thread.
    #[must_use]
    pub fn context(&self) -> &GuestRegs {
        &self.context
    }

    /// Whether the thread belongs in the priority queues: runnable with no suspend in effect.
    /// Eden tests `GetRawState() == ThreadState::Runnable`, the run state and suspend flags in
    /// one word.
    pub(super) fn is_queued(&self) -> bool {
        self.state == ThreadState::Runnable && self.suspend_flags == 0
    }
}
