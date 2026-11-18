//! A process's resource limit (Horizon's `KResourceLimit`): what a handle from `GetInfo`'s
//! `ResourceLimit` id names, what `GetResourceLimitLimitValue` and
//! `GetResourceLimitCurrentValue` answer from, what `GetInfo`'s `FreeThreadCount` is derived
//! from, and what refuses a thread past the limit.
//!
//! Ported from Eden (GPL-3.0-or-later), paths relative to the reference folder:
//!
//! - `eden/src/core/hle/kernel/k_resource_limit.cpp` @ cf654052: `CreateResourceLimitForProcess`
//!   (the limit values, below), `GetLimitValue`, `GetCurrentValue`, `GetFreeValue` and
//!   `Reserve` (the thread check, [`Values::reserve_one`]).
//! - `eden/src/core/hle/kernel/svc_types.h` @ cf654052: `enum class LimitableResource`, the
//!   numbers of [`LimitableResource`], and `IsValidResourceType` (`type < Count`,
//!   `k_resource_limit.h`): [`LimitableResource::from_raw`].
//! - `eden/src/core/hle/kernel/k_process.cpp` @ cf654052, `LoadFromMetadata`: every process Eden
//!   loads gets one (`CreateResourceLimitForProcess(system, physical_memory_size)`, the size of
//!   the process's pool); here [`crate::sched::Scheduler::add_process`] makes it beside the
//!   process's own object, and the process's row holds its one reference
//!   ([`crate::process::ProcessRecord::resource_limit`]), as `KProcess::m_resource_limit` does.
//!
//! # The limits
//!
//! Each is `CreateResourceLimitForProcess`'s `SetLimitValue` (the same in
//! `eden-pr4588/src/core/hle/kernel/k_resource_limit.cpp` @ e5fb3b97):
//!
//! | Resource | Limit | Eden |
//! |---|---|---|
//! | [`LimitableResource::PhysicalMemory`] | the process's pool, [`MemoryLimit::limit`] | `SetLimitValue(PhysicalMemoryMax, physical_memory_size)`, `MemoryManager().GetSize(pool)` |
//! | [`LimitableResource::ThreadCount`] | [`THREAD_COUNT_LIMIT`] (800) | `SetLimitValue(ThreadCountMax, 800)` |
//! | [`LimitableResource::EventCount`] | [`EVENT_COUNT_LIMIT`] (900) | `SetLimitValue(EventCountMax, 900)` |
//! | [`LimitableResource::TransferMemoryCount`] | [`TRANSFER_MEMORY_COUNT_LIMIT`] (200) | `SetLimitValue(TransferMemoryCountMax, 200)` |
//! | [`LimitableResource::SessionCount`] | [`SESSION_COUNT_LIMIT`] (1133) | `SetLimitValue(SessionCountMax, 1133)` |
//!
//! Eden's own comment calls these "the system defaults", lower for service processes; for an
//! application they are a guess Eden makes too.
//!
//! # No counts of its own
//!
//! The object ([`ResourceLimit`]) names the process it belongs to and holds nothing else. Each
//! value is read from the table that already keeps the fact, so the limit, `GetInfo` and the
//! checks cannot disagree, and a snapshot has no second count to validate:
//!
//! - **Physical memory**: the owner's address space's [`MemoryLimit`], the count every
//!   reservation of code, heap, stack and `MapPhysicalMemory` goes through, and which `GetInfo`
//!   6 (`TotalMemorySize`) and 7 (`UsedMemorySize`) read. The current value is what is reserved
//!   ([`MemoryLimit::reserved`]), which is `GetInfo` 7's used size (code, normal memory, the
//!   main thread's stack and the system resource's charge). The system resource is charged to
//!   this same count when the address space is made
//!   ([`crate::memory::AddressSpace::with_system_resource`], as Eden's
//!   `KSecureSystemResource::Initialize` reserves `PhysicalMemoryMax` from the process's limit),
//!   so the current value holds it with no count of its own here. Test (in
//!   `svc/info/tests/process_ids.rs`):
//!   `the_total_and_used_memory_count_the_system_resource_as_eden_does`.
//! - **Threads**: the owner's threads that still have a row
//!   ([`crate::process::ProcessRecord::threads`]). Eden's current value counts a thread from its
//!   reservation (`CreateThread`, `KProcess::Run`) to its destruction
//!   (`KThread::PostDestroy`'s `Release(ThreadCountMax, 1, hint)`), which is the life of a row
//!   here. A thread is refused when the count has reached the limit
//!   ([`crate::sched::Scheduler::create_user_thread`], [`Values::reserve_one`]).
//!
//! # Divergences from Eden
//!
//! - **Events, transfer memories and sessions are not counted.** Eden reserves one of each from
//!   the creating process's limit (`CreateEvent`, `CreateTransferMemory`, `ConnectToNamedPort`
//!   and the services' own sessions) and refuses past it with `ResultLimitReached`. Here their
//!   current value is zero ([`UNCOUNTED_CURRENT_VALUE`]) and nothing is refused; their limits are
//!   still Eden's. Tests: `events_transfer_memories_and_sessions_are_not_counted`, and the
//!   calls that would be refused past the limits, `creating_many_events_never_reaches_a_limit`
//!   (`dispatch/sync/tests.rs`), `more_transfer_memories_than_edens_limit_are_all_lent`
//!   (`svc/transfer_memory/tests.rs`) and `connecting_many_times_never_runs_out_of_sessions`
//!   (`dispatch/ipc/tests.rs`).
//! - **A thread past the limit is refused at once.** Eden's `CreateThread` waits up to 100 ms
//!   (`KResourceLimit::Reserve` with a timeout) when the count is at the limit but an exited
//!   thread's release is pending (`m_current_hints` below `m_current_values`: a thread that has
//!   called `ExitThread` and whose last handle is still open). Here no release hint is kept, so
//!   the call fails with `ResultLimitReached` without waiting; in Eden it succeeds only if
//!   another thread closes that last handle within the 100 ms. Test (in
//!   `svc/thread/limit_tests.rs`): `an_ended_thread_with_an_open_handle_still_counts_and_nothing_waits`.
//! - **Only a process's own limit exists.** `CreateResourceLimit` and `SetResourceLimitLimitValue`
//!   (and `GetResourceLimitPeakValue`, which Eden does not implement) are not built: nothing on
//!   the path to a title uses them. So no limit is shared by two processes, and the limit values
//!   never change. Adding them means storing the limit values in the object (and its snapshot,
//!   with a version bump) instead of taking Eden's constants.

use crate::id::ProcessId;
use crate::memory::MemoryLimit;
use crate::result::KernelError;

/// What a resource limit limits (`Svc::LimitableResource`, `eden/src/core/hle/kernel/svc_types.h`
/// @ cf654052). The number of each is [`LimitableResource::number`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LimitableResource {
    /// `PhysicalMemoryMax` (0): bytes of physical memory.
    PhysicalMemory,
    /// `ThreadCountMax` (1): threads.
    ThreadCount,
    /// `EventCountMax` (2): events.
    EventCount,
    /// `TransferMemoryCountMax` (3): transfer memories.
    TransferMemoryCount,
    /// `SessionCountMax` (4): sessions.
    SessionCount,
}

impl LimitableResource {
    /// Every resource, in number order.
    pub const ALL: [Self; 5] = [
        Self::PhysicalMemory,
        Self::ThreadCount,
        Self::EventCount,
        Self::TransferMemoryCount,
        Self::SessionCount,
    ];

    /// The resource's number, as a system call passes it.
    #[must_use]
    pub const fn number(self) -> u32 {
        match self {
            Self::PhysicalMemory => 0,
            Self::ThreadCount => 1,
            Self::EventCount => 2,
            Self::TransferMemoryCount => 3,
            Self::SessionCount => 4,
        }
    }

    /// The resource numbered `raw`, or `None` for a number Eden's `IsValidResourceType`
    /// refuses (`type < LimitableResource::Count`, `Count` being 5).
    #[must_use]
    pub fn from_raw(raw: u32) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|resource| resource.number() == raw)
    }
}

/// The `ThreadCountMax` limit of every process's resource limit:
/// `SetLimitValue(LimitableResource::ThreadCountMax, 800)` in `CreateResourceLimitForProcess`
/// (see the module docs).
pub const THREAD_COUNT_LIMIT: u64 = 800;

/// The `EventCountMax` limit: `SetLimitValue(LimitableResource::EventCountMax, 900)`.
pub const EVENT_COUNT_LIMIT: u64 = 900;

/// The `TransferMemoryCountMax` limit:
/// `SetLimitValue(LimitableResource::TransferMemoryCountMax, 200)`.
pub const TRANSFER_MEMORY_COUNT_LIMIT: u64 = 200;

/// The `SessionCountMax` limit: `SetLimitValue(LimitableResource::SessionCountMax, 1133)`.
pub const SESSION_COUNT_LIMIT: u64 = 1133;

/// The current value of the resources nothing here counts yet (events, transfer memories and
/// sessions; see the module docs, "Divergences from Eden").
pub const UNCOUNTED_CURRENT_VALUE: u64 = 0;

/// A resource limit object: what [`crate::object::Object::ResourceLimit`] holds. It names the
/// process it belongs to; every value is read from that process's tables ([`Usage`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceLimit {
    owner: ProcessId,
}

impl ResourceLimit {
    /// The resource limit of `owner` (`CreateResourceLimitForProcess`). The constructor is
    /// public, so any code, in this crate or another, may make one; the kernel makes one per
    /// process, in [`crate::sched::Scheduler::add_process`], and the object table's restore
    /// remakes the ones a snapshot holds after checking each against its owner's row. A limit
    /// placed in the object table by any other path is one its owner's row does not name:
    /// saving or restoring the table refuses it
    /// ([`crate::object::snapshot::ObjectFault::ResourceLimit`]).
    #[must_use]
    pub const fn of(owner: ProcessId) -> Self {
        Self { owner }
    }

    /// The process it belongs to.
    #[must_use]
    pub const fn owner(self) -> ProcessId {
        self.owner
    }
}

/// What the owner of a resource limit holds that the limit counts (see the module docs, "No
/// counts of its own").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage<'a> {
    /// The owner's address space's physical memory count.
    pub memory: &'a MemoryLimit,
    /// How many of the owner's threads still have a row.
    pub threads: usize,
}

/// One resource's limit and current value (`GetLimitValue`, `GetCurrentValue`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Values {
    /// The most that may be held.
    pub limit: u64,
    /// What is held now.
    pub current: u64,
}

impl Values {
    /// The values of `resource` for an owner holding `usage`: the limit from the module docs'
    /// table, the current value from what the owner holds.
    #[must_use]
    pub fn of(resource: LimitableResource, usage: Usage<'_>) -> Self {
        match resource {
            LimitableResource::PhysicalMemory => Self {
                limit: usage.memory.limit(),
                current: usage.memory.reserved(),
            },
            LimitableResource::ThreadCount => Self::threads(usage.threads),
            LimitableResource::EventCount => Self::uncounted(EVENT_COUNT_LIMIT),
            LimitableResource::TransferMemoryCount => Self::uncounted(TRANSFER_MEMORY_COUNT_LIMIT),
            LimitableResource::SessionCount => Self::uncounted(SESSION_COUNT_LIMIT),
        }
    }

    /// The thread values of an owner with `threads` threads that still have a row:
    /// [`THREAD_COUNT_LIMIT`], and the count.
    #[must_use]
    pub fn threads(threads: usize) -> Self {
        Self {
            limit: THREAD_COUNT_LIMIT,
            // A count no `u64` holds is past any limit.
            current: u64::try_from(threads).unwrap_or(u64::MAX),
        }
    }

    const fn uncounted(limit: u64) -> Self {
        Self {
            limit,
            current: UNCOUNTED_CURRENT_VALUE,
        }
    }

    /// What can still be reserved (`GetFreeValue`, `limit - current`): zero for a current value
    /// past the limit, which Eden asserts never happens and which no live operation here leaves
    /// (the thread check refuses first, and a memory reservation never passes its limit).
    #[must_use]
    pub const fn free(self) -> u64 {
        self.limit.saturating_sub(self.current)
    }

    /// Whether one more fits: `KResourceLimit::Reserve(which, 1)` without its wait (see the
    /// module docs). [`KernelError::LimitReached`], the result `CreateThread` and
    /// `KProcess::Run` give a failed reservation (`R_UNLESS(thread_reservation.Succeeded(),
    /// ResultLimitReached)`), when the current value has reached the limit
    /// (`m_current_values[index] + value <= m_limit_values[index]` is false) or one more would
    /// overflow.
    pub fn reserve_one(self) -> Result<(), KernelError> {
        match self.current.checked_add(1) {
            Some(total) if total <= self.limit => Ok(()),
            _ => Err(KernelError::LimitReached),
        }
    }
}

#[cfg(test)]
mod tests;
