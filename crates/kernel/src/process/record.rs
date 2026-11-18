//! What changes about a process while it runs: its state, its threads, how many of them are
//! running, its thread-local pages and its scheduled count. The rows live in the scheduler
//! table ([`crate::sched::Scheduler::process`]): Horizon changes each of these under its
//! scheduler lock (`KProcess::ChangeState`, `CreateThreadLocalRegion`,
//! `IncrementScheduledCount`, `eden/src/core/hle/kernel/k_process.cpp` @ cf654052).
//!
//! The row holds one reference to the process's resource limit (`KProcess::m_resource_limit`,
//! opened in `KProcess::Initialize` and closed in `Finalize`; [`crate::resource_limit`]), which
//! is never closed, as process rows are never removed.
//!
//! The row also keeps the process's random entropy (`KProcess::m_entropy`, which `GetInfo`'s
//! `RandomEntropy` id reads). Horizon draws it in `KProcess::Initialize` (`GenerateRandom(
//! m_entropy)`, `k_process.cpp` @ cf654052, line 239) and never changes it. Here
//! [`crate::process::create_with_env`] draws it from the environment as it creates the process,
//! and it never changes after that. A process made without an environment
//! ([`crate::process::create`]) has none ([`crate::svc::info`] lists what a `RandomEntropy`
//! call of it answers).

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
    /// Its resource limit's row in the object table ([`crate::object::Object::ResourceLimit`]),
    /// one reference to which this row holds (`KProcess::m_resource_limit`).
    pub(crate) resource_limit: ObjectId,
    pub(crate) state: ProcessState,
    /// Threads started and not yet ended (`m_num_running_threads`).
    pub(crate) running_threads: u32,
    /// Every thread of the process that still has a row (`m_thread_list`).
    pub(crate) threads: BTreeSet<ThreadId>,
    pub(crate) tls: ThreadLocalPages,
    /// `m_schedule_count`: what its threads' yields compare with.
    pub(crate) scheduled_count: u64,
    /// `m_entropy`, drawn at creation, or `None` (see the module docs).
    pub(crate) entropy: Option<[u64; ENTROPY_COUNT]>,
}

impl ProcessRecord {
    /// A newly created process, whose object is `object` and whose resource limit's object is
    /// `resource_limit`.
    #[must_use]
    pub fn new(object: ObjectId, resource_limit: ObjectId) -> Self {
        Self {
            object,
            resource_limit,
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

    /// Its resource limit's row in the object table: what `GetInfo`'s `ResourceLimit` id adds a
    /// handle to ([`crate::resource_limit`]).
    #[must_use]
    pub const fn resource_limit(&self) -> ObjectId {
        self.resource_limit
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

    /// How many of its threads still have a row: the current value of its resource limit's
    /// thread count ([`crate::resource_limit`]).
    #[must_use]
    pub fn thread_count(&self) -> usize {
        self.threads.len()
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

    /// Its random entropy (`GetRandomEntropy`), drawn as the process was created
    /// ([`crate::process::create_with_env`]), or `None` for a process made without an
    /// environment ([`crate::process::create`]).
    #[must_use]
    pub const fn entropy(&self) -> Option<[u64; ENTROPY_COUNT]> {
        self.entropy
    }
}
