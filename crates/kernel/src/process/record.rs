//! What changes about a process while it runs: its state, its threads, how many of them are
//! running, its thread-local pages and its scheduled count. The rows live in the scheduler
//! table ([`crate::sched::Scheduler::process`]): Horizon changes each of these under its
//! scheduler lock (`KProcess::ChangeState`, `CreateThreadLocalRegion`,
//! `IncrementScheduledCount`, `eden/src/core/hle/kernel/k_process.cpp` @ cf654052).
//!
//! The row also keeps the process's random entropy (`KProcess::m_entropy`, which `GetInfo`'s
//! `RandomEntropy` id reads). Horizon draws it in `KProcess::Initialize` (`GenerateRandom(
//! m_entropy)`, `k_process.cpp` @ cf654052, line 239) and never changes it. Here it is drawn
//! from the environment the first time `GetInfo` asks for it, because creating a process has no
//! route to the environment's random numbers ([`crate::svc::info`] lists the divergence), and
//! never changes after that.

use std::collections::BTreeSet;

use super::tls::ThreadLocalPages;
use crate::id::{ObjectId, ThreadId};

/// How many words of random entropy a process has: `std::array<u64, 4> m_entropy`
/// (`eden/src/core/hle/kernel/k_process.h` @ cf654052, line 103; the same in
/// `eden-pr4588/src/core/hle/kernel/k_process.h` @ e5fb3b97, so the same for a Switch 2
/// program).
pub const ENTROPY_COUNT: usize = 4;

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
    /// The process's row in the object table, whose one reference this row holds.
    pub(crate) object: ObjectId,
    pub(crate) state: ProcessState,
    /// Threads started and not yet ended (`m_num_running_threads`).
    pub(crate) running_threads: u32,
    /// Every thread of the process that still has a row (`m_thread_list`).
    pub(crate) threads: BTreeSet<ThreadId>,
    pub(crate) tls: ThreadLocalPages,
    /// `m_schedule_count`: what its threads' yields compare with.
    pub(crate) scheduled_count: u64,
    /// `m_entropy`, once drawn (see the module docs).
    pub(crate) entropy: Option<[u64; ENTROPY_COUNT]>,
}

impl ProcessRecord {
    /// A newly created process, whose object is `object`.
    #[must_use]
    pub fn new(object: ObjectId) -> Self {
        Self {
            object,
            state: ProcessState::Created,
            running_threads: 0,
            threads: BTreeSet::new(),
            tls: ThreadLocalPages::new(),
            scheduled_count: 0,
            entropy: None,
        }
    }

    /// Its row in the object table ([`crate::object::Object::Process`]): what a copied
    /// [`crate::handle::Handle::CURRENT_PROCESS`] names in an IPC request.
    #[must_use]
    pub const fn object(&self) -> ObjectId {
        self.object
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

    /// Its random entropy (`GetRandomEntropy`), or `None` while it has not been drawn
    /// ([`crate::sched::Scheduler::set_entropy`]).
    #[must_use]
    pub const fn entropy(&self) -> Option<[u64; ENTROPY_COUNT]> {
        self.entropy
    }
}
