//! What changes about a process while it runs: its state, its threads, how many of them are
//! running, its thread-local pages and its scheduled count. The rows live in the scheduler
//! table ([`crate::sched::Scheduler::process`]): Horizon changes each of these under its
//! scheduler lock (`KProcess::ChangeState`, `CreateThreadLocalRegion`,
//! `IncrementScheduledCount`, `eden/src/core/hle/kernel/k_process.cpp` @ cf654052).

use std::collections::BTreeSet;

use super::tls::ThreadLocalPages;
use crate::id::ThreadId;

/// A process's state: the part of Eden's `KProcess::State` a process without a debugger goes
/// through (`Created`, `Running`, `Terminated`; `Terminating` is never seen from outside a call
/// here, because termination finishes inside the call that starts it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProcessState {
    /// Created, its main thread not yet started.
    Created,
    /// Its main thread was started.
    Running,
    /// Ended: every thread terminated and its handles closed.
    Terminated,
}

/// One process's row in the scheduler table. See the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessRecord {
    pub(crate) state: ProcessState,
    /// Threads started and not yet ended (`m_num_running_threads`).
    pub(crate) running_threads: u32,
    /// Every thread of the process that still has a row (`m_thread_list`).
    pub(crate) threads: BTreeSet<ThreadId>,
    pub(crate) tls: ThreadLocalPages,
    /// `m_schedule_count`: what its threads' yields compare with.
    pub(crate) scheduled_count: u64,
}

impl ProcessRecord {
    /// A newly created process.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: ProcessState::Created,
            running_threads: 0,
            threads: BTreeSet::new(),
            tls: ThreadLocalPages::new(),
            scheduled_count: 0,
        }
    }

    /// Its state.
    #[must_use]
    pub const fn state(&self) -> ProcessState {
        self.state
    }

    /// How many of its threads were started and have not ended.
    #[must_use]
    pub const fn running_threads(&self) -> u32 {
        self.running_threads
    }

    /// Its threads, in id order.
    pub fn threads(&self) -> impl Iterator<Item = ThreadId> + '_ {
        self.threads.iter().copied()
    }

    /// Its thread-local pages.
    #[must_use]
    pub const fn tls(&self) -> &ThreadLocalPages {
        &self.tls
    }

    /// Its scheduled count (`GetScheduledCount`).
    #[must_use]
    pub const fn scheduled_count(&self) -> u64 {
        self.scheduled_count
    }
}

impl Default for ProcessRecord {
    fn default() -> Self {
        Self::new()
    }
}
