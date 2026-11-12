//! The [`Scheduler`] table: the thread rows, the priority queues, and each core's choice.
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later):
//! - `eden/src/core/hle/kernel/k_scheduler.cpp`: `UpdateHighestPriorityThread(s)Impl`,
//!   `OnThreadStateChanged`, `OnThreadPriorityChanged`, `OnThreadAffinityMaskChanged`, the three
//!   yields, `ScheduleImpl`/`SwitchThread` (bookkeeping only) and `RescheduleCores`.
//! - `eden/src/core/hle/kernel/k_scheduler_lock.h`: `Unlock`, as [`Scheduler::update`].
//! - `eden/src/core/hle/kernel/k_thread.cpp`: `Initialize` (scheduler fields), `Run`,
//!   `SetState`, `RequestSuspend`, `Resume`, `TrySuspend`, `UpdateState`, `SetActivity`,
//!   `SetBasePriority`, `RestorePriority` (without inheritance) and `SetCoreMask`.
//!
//! Every method that takes a [`ThreadId`] looks the row up once to check it, then works through
//! ids: Eden's `KThread*` becomes an id, and `thread->Field()` a slab lookup. An id a caller
//! passed in is looked up with `Slab::get` (stale is the caller's error, [`SchedError::Slab`]);
//! an id read back out of a queue or a core's state with `Slab::stored` (stale is damaged
//! bookkeeping, [`SchedError::Corrupt`]). See the module docs of `sched` for what is deferred
//! and where starquake diverges.

use std::collections::BTreeMap;

use starquake_cpu::GuestRegs;
use starquake_trace::GuestTicks;

use super::cores::{APPLICATION_CORE_COUNT, AffinityMask, CoreId, CoreSignals, MAX_CORES};
use super::queue::{PriorityQueue, Threads, Walk};
use super::thread::{Priority, QueueLink, SuspendType, Thread, ThreadState};
use super::{HIGHEST_CORE_MIGRATION_ALLOWED_PRIORITY, IdealCore, SchedError, ThreadActivity};
use crate::id::{ProcessId, ThreadId};
use crate::ipc::Ipc;
use crate::object::Objects;
use crate::process::ProcessRecord;
use crate::result::KernelError;

mod lifecycle;

pub use lifecycle::{ThreadExit, UserThread};

/// The id the first thread gets: Eden's `next_thread_id{1}` (`eden/src/core/hle/kernel/kernel.cpp`
/// @ cf654052).
pub const FIRST_THREAD_ID: u64 = 1;

/// One core's scheduling state: the scheduler's `SchedulingState` and current thread
/// (`k_scheduler.h`).
#[derive(Clone, Debug, PartialEq, Eq)]
struct CoreState {
    /// The thread the core should run (`highest_priority_thread`); `None` means idle.
    highest: Option<ThreadId>,
    /// The thread the core is running (`m_current_thread`); `None` means idle or parked.
    current: Option<ThreadId>,
    /// Whether the core must call [`Scheduler::schedule`] (`needs_scheduling`).
    needs_scheduling: bool,
}

impl CoreState {
    /// Eden's `KScheduler` constructor sets `needs_scheduling`; there is no thread yet.
    const fn new() -> Self {
        Self {
            highest: None,
            current: None,
            needs_scheduling: true,
        }
    }
}

/// Threads and their scheduling: the kernel's scheduler table. See the `sched` module docs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scheduler {
    /// What handles name (kernel spec: events and sessions live in this table; a thread's
    /// object row names a row of `threads`).
    objects: Objects,
    /// Servers, sessions and named ports (kernel spec: sessions live in this table).
    ipc: Ipc,
    threads: Threads,
    queue: PriorityQueue,
    cores: Vec<CoreState>,
    /// `GlobalSchedulerContext::m_scheduler_update_needed`.
    update_needed: bool,
    /// `KProcess::m_schedule_count` for the threads that belong to no process; each process
    /// keeps its own ([`ProcessRecord::scheduled_count`]).
    scheduled_count: u64,
    /// What changes about each process while it runs: its state, threads and thread-local
    /// pages (kernel slice 8). Its fixed facts are in [`crate::tables::Processes`].
    processes: BTreeMap<ProcessId, ProcessRecord>,
    /// The id the next thread gets (`KernelCore::CreateNewThreadID`, from 1).
    next_thread_id: u64,
}

impl Default for Scheduler {
    /// A scheduler with [`APPLICATION_CORE_COUNT`] cores.
    fn default() -> Self {
        Self::build(APPLICATION_CORE_COUNT)
    }
}

impl Scheduler {
    fn build(core_count: u8) -> Self {
        Self {
            objects: Objects::new(),
            ipc: Ipc::new(),
            threads: Threads::new(),
            queue: PriorityQueue::new(core_count),
            cores: vec![CoreState::new(); usize::from(core_count)],
            update_needed: false,
            scheduled_count: 0,
            processes: BTreeMap::new(),
            next_thread_id: FIRST_THREAD_ID,
        }
    }

    /// A scheduler with `core_count` cores, 1 to [`MAX_CORES`] (kernel spec, rule 5: cores are
    /// configuration).
    pub fn new(core_count: u8) -> Result<Self, SchedError> {
        if core_count == 0 || core_count > MAX_CORES {
            return Err(SchedError::CoreCount { count: core_count });
        }
        Ok(Self::build(core_count))
    }

    /// The number of cores.
    #[must_use]
    pub fn core_count(&self) -> u8 {
        // `build` only ever gets a u8, so the length fits.
        u8::try_from(self.cores.len()).unwrap_or(MAX_CORES)
    }

    fn core(&self, core: CoreId) -> Result<&CoreState, SchedError> {
        self.cores
            .get(usize::from(core.index()))
            .ok_or(SchedError::NoSuchCore {
                core: core.index(),
                count: self.core_count(),
            })
    }

    fn core_mut(&mut self, core: CoreId) -> Result<&mut CoreState, SchedError> {
        let count = self.core_count();
        self.cores
            .get_mut(usize::from(core.index()))
            .ok_or(SchedError::NoSuchCore {
                core: core.index(),
                count,
            })
    }

    /// Every core this scheduler has, lowest first.
    fn all_cores(&self) -> impl Iterator<Item = CoreId> + use<> {
        (0..self.core_count()).filter_map(CoreId::new)
    }

    /// The mask of every core this scheduler has.
    fn core_mask(&self) -> AffinityMask {
        AffinityMask::from_bits(self.all_cores().fold(0, |bits, core| bits | core.bit()))
    }

    /// The object table: what handles name, with their reference counts.
    #[must_use]
    pub fn objects(&self) -> &Objects {
        &self.objects
    }

    /// The object table, to create objects in and to open and close references through.
    pub fn objects_mut(&mut self) -> &mut Objects {
        &mut self.objects
    }

    /// The IPC table: servers, sessions and named ports.
    #[must_use]
    pub fn ipc(&self) -> &Ipc {
        &self.ipc
    }

    /// The object table and the IPC table together, for the operations that change both (a
    /// session is an object, and a request holds references).
    pub fn ipc_mut(&mut self) -> (&mut Objects, &mut Ipc) {
        (&mut self.objects, &mut self.ipc)
    }

    /// The thread `id` names.
    pub fn thread(&self, id: ThreadId) -> Result<&Thread, SchedError> {
        Ok(self.threads.get(id)?)
    }

    fn thread_mut(&mut self, id: ThreadId) -> Result<&mut Thread, SchedError> {
        Ok(self.threads.get_mut(id)?)
    }

    /// A thread's saved register context, to write a result into or load on a switch.
    pub fn context_mut(&mut self, id: ThreadId) -> Result<&mut GuestRegs, SchedError> {
        Ok(&mut self.thread_mut(id)?.context)
    }

    /// The thread a core should run (`highest_priority_thread`), as of the last
    /// [`Scheduler::update`].
    pub fn highest(&self, core: CoreId) -> Result<Option<ThreadId>, SchedError> {
        Ok(self.core(core)?.highest)
    }

    /// The thread a core is running (`GetSchedulerCurrentThread`).
    pub fn current(&self, core: CoreId) -> Result<Option<ThreadId>, SchedError> {
        Ok(self.core(core)?.current)
    }

    /// Whether a core must call [`Scheduler::schedule`].
    pub fn needs_scheduling(&self, core: CoreId) -> Result<bool, SchedError> {
        Ok(self.core(core)?.needs_scheduling)
    }

    /// The core a thread is running on, if any: the loop over every core's current thread in
    /// Eden's `SetCoreMask`.
    #[must_use]
    pub fn running_on(&self, id: ThreadId) -> Option<CoreId> {
        self.all_cores()
            .find(|&core| self.core(core).is_ok_and(|state| state.current == Some(id)))
    }

    // ---------------------------------------------------------------------------------------
    // Thread lifecycle and state (k_thread.cpp)
    // ---------------------------------------------------------------------------------------

    /// Creates a thread in the [`ThreadState::Initialized`] state, bound to `core` at
    /// `priority`: the scheduler fields of `KThread::Initialize` (ideal and active core
    /// `core`, affinity `{core}`, no suspend requests, never scheduled).
    pub fn create_thread(
        &mut self,
        priority: Priority,
        core: CoreId,
    ) -> Result<ThreadId, SchedError> {
        self.insert_thread(priority, core, None)
    }

    /// Mints the next guest thread id (`CreateNewThreadID`). Ids are never reused; running out
    /// of 64 bits is a table that cannot grow.
    fn next_thread_id(&mut self) -> Result<u64, SchedError> {
        let id = self.next_thread_id;
        self.next_thread_id = id.checked_add(1).ok_or(SchedError::Corrupt)?;
        Ok(id)
    }

    /// Inserts a thread row for [`Scheduler::create_thread`] and the lifecycle's user threads:
    /// the scheduler fields of `KThread::Initialize`, bound to the physical `core`, with the
    /// virtual ideal core and mask `virtual_core` (the physical core when it is `None`).
    fn insert_thread(
        &mut self,
        priority: Priority,
        core: CoreId,
        virtual_core: Option<u8>,
    ) -> Result<ThreadId, SchedError> {
        if !self.core_mask().contains(core) {
            return Err(KernelError::InvalidCoreId.into());
        }
        let virtual_core = virtual_core.unwrap_or(core.index());
        let thread_id = self.next_thread_id()?;
        let thread = Thread {
            state: ThreadState::Initialized,
            suspend_flags: 0,
            priority,
            base_priority: priority,
            affinity: AffinityMask::single(core),
            ideal_core: Some(core),
            active_core: Some(core),
            current_core: core,
            last_scheduled_tick: GuestTicks(0),
            yield_schedule_count: None,
            links: vec![QueueLink::default(); self.cores.len()],
            context: GuestRegs::default(),
            process: None,
            object: None,
            holds_itself: false,
            guest_id: thread_id,
            tls_address: 0,
            virtual_ideal_core: i32::from(virtual_core),
            virtual_affinity: 1_u64.checked_shl(u32::from(virtual_core)).unwrap_or(0),
        };
        Ok(self.threads.insert(thread)?)
    }

    /// Starts a created thread (`KThread::Run`): it must be [`ThreadState::Initialized`], or
    /// the result is [`KernelError::InvalidState`]. A thread of a process counts as one of its
    /// running threads (`IncrementRunningThreadCount`), and a thread with an object row takes a
    /// reference to it (`this->Open`), which it holds until it ends.
    pub fn start(&mut self, id: ThreadId) -> Result<(), SchedError> {
        let thread = self.thread(id)?;
        if thread.state != ThreadState::Initialized {
            return Err(KernelError::InvalidState.into());
        }
        let (process, object) = (thread.process, thread.object);
        if let Some(process) = process {
            let record = self.record_mut(process)?;
            record.running_threads = record
                .running_threads
                .checked_add(1)
                .ok_or(SchedError::Corrupt)?;
        }
        if let Some(object) = object {
            crate::handle::ObjectRefs::open(&mut self.objects, object)
                .map_err(|_| SchedError::Corrupt)?;
            self.thread_mut(id)?.holds_itself = true;
        }
        self.set_state(id, ThreadState::Runnable)
    }

    /// Sets a thread's run state, keeping its suspend requests (`KThread::SetState`). Later
    /// slices block and wake threads through this.
    pub fn set_state(&mut self, id: ThreadId, state: ThreadState) -> Result<(), SchedError> {
        let thread = self.thread_mut(id)?;
        let was_queued = thread.is_queued();
        let old_state = thread.state;
        thread.state = state;
        if old_state != state {
            self.on_thread_state_changed(id, was_queued)?;
        }
        Ok(())
    }

    /// Adds a suspend request (`KThread::RequestSuspend`, then `TrySuspend`, which always
    /// suspends until slice 5 adds kernel waiters).
    pub fn request_suspend(&mut self, id: ThreadId, kind: SuspendType) -> Result<(), SchedError> {
        let thread = self.thread_mut(id)?;
        let flags = thread.suspend_flags | kind.bit();
        self.update_suspend_flags(id, flags)
    }

    /// Withdraws a suspend request (`KThread::Resume`).
    pub fn resume(&mut self, id: ThreadId, kind: SuspendType) -> Result<(), SchedError> {
        let thread = self.thread_mut(id)?;
        let flags = thread.suspend_flags & !kind.bit();
        self.update_suspend_flags(id, flags)
    }

    /// `KThread::UpdateState`: put the requested flags into the state and tell the scheduler
    /// if that changed it.
    fn update_suspend_flags(&mut self, id: ThreadId, flags: u8) -> Result<(), SchedError> {
        let thread = self.thread_mut(id)?;
        let was_queued = thread.is_queued();
        let old_flags = thread.suspend_flags;
        thread.suspend_flags = flags;
        if old_flags != flags {
            self.on_thread_state_changed(id, was_queued)?;
        }
        Ok(())
    }

    /// Pauses or resumes a thread (`KThread::SetActivity`, the part under the scheduler lock).
    ///
    /// The thread must be [`ThreadState::Waiting`] or [`ThreadState::Runnable`]; pausing a
    /// paused thread or resuming one that isn't paused is also [`KernelError::InvalidState`].
    pub fn set_activity(
        &mut self,
        id: ThreadId,
        activity: ThreadActivity,
    ) -> Result<(), SchedError> {
        let thread = self.thread(id)?;
        if !matches!(thread.state, ThreadState::Waiting | ThreadState::Runnable) {
            return Err(KernelError::InvalidState.into());
        }
        let paused = thread.is_suspend_requested(SuspendType::Thread);
        match activity {
            ThreadActivity::Paused if paused => Err(KernelError::InvalidState.into()),
            ThreadActivity::Paused => self.request_suspend(id, SuspendType::Thread),
            ThreadActivity::Runnable if !paused => Err(KernelError::InvalidState.into()),
            ThreadActivity::Runnable => self.resume(id, SuspendType::Thread),
        }
    }

    /// Sets a thread's base priority (`KThread::SetBasePriority`). `caller` is the core making
    /// the call: if `id` is that core's current thread it stays at the front of its new level
    /// (`OnThreadPriorityChanged` passes `thread == GetCurrentThreadPointer()`).
    pub fn set_base_priority(
        &mut self,
        caller: CoreId,
        id: ThreadId,
        priority: Priority,
    ) -> Result<(), SchedError> {
        self.core(caller)?;
        self.thread_mut(id)?.base_priority = priority;
        self.restore_priority(caller, id)
    }

    /// `KThread::RestorePriority` for one thread with no held locks: the effective priority
    /// becomes the base priority. Inheritance (slice 5) walks lock owners from here.
    fn restore_priority(&mut self, caller: CoreId, id: ThreadId) -> Result<(), SchedError> {
        let thread = self.thread_mut(id)?;
        let new_priority = thread.base_priority;
        if new_priority == thread.priority {
            return Ok(());
        }
        let old_priority = thread.priority;
        thread.priority = new_priority;
        self.on_thread_priority_changed(caller, id, old_priority)
    }

    /// Sets a thread's ideal core and affinity mask (`KThread::SetCoreMask`, the part under the
    /// scheduler lock, with migration enabled).
    ///
    /// Errors, before anything changes: an empty mask is [`KernelError::InvalidCombination`];
    /// a mask with a core the scheduler doesn't have is [`KernelError::InvalidCoreId`], and so is
    /// an [`IdealCore::Core`] the scheduler doesn't have (checked before the mask); an ideal core
    /// the scheduler has but the mask lacks, or [`IdealCore::NoUpdate`] when the current ideal
    /// core is outside the mask, is [`KernelError::InvalidCombination`].
    ///
    /// If the thread's active core leaves its mask, it moves to the ideal core, or with no ideal
    /// core to the highest core in the mask.
    pub fn set_core_mask(
        &mut self,
        id: ThreadId,
        ideal: IdealCore,
        mask: AffinityMask,
    ) -> Result<(), SchedError> {
        // svcSetThreadCoreMask's checks (svc_thread.cpp): mask within the allowed cores
        // (InvalidCoreId), non-empty (InvalidCombination), ideal core a valid core id
        // (`IsValidVirtualCoreId`, else InvalidCoreId), then in the mask (InvalidCombination).
        if mask.bits() & !self.core_mask().bits() != 0 {
            return Err(KernelError::InvalidCoreId.into());
        }
        if mask.is_empty() {
            return Err(KernelError::InvalidCombination.into());
        }
        let thread = self.thread(id)?;
        let new_ideal = match ideal {
            IdealCore::Core(core) => {
                if self.core_mask().bits() & core.bit() == 0 {
                    return Err(KernelError::InvalidCoreId.into());
                }
                if !mask.contains(core) {
                    return Err(KernelError::InvalidCombination.into());
                }
                Some(core)
            }
            IdealCore::DontCare => None,
            IdealCore::NoUpdate => {
                // `1ULL << -1` for a "don't care" ideal core: bit 63 (a listed guess).
                let tested = thread
                    .ideal_core
                    .or(CoreId::new(MAX_CORES - 1))
                    .map_or(0, CoreId::bit);
                if mask.bits() & tested == 0 {
                    return Err(KernelError::InvalidCombination.into());
                }
                thread.ideal_core
            }
        };

        self.apply_core_mask(id, new_ideal, mask)
    }

    /// Gives a thread a new physical ideal core and affinity mask, already checked
    /// (`KThread::SetCoreMask`, the part with migration enabled): if its active core leaves the
    /// mask, it moves to the ideal core, or with no ideal core to the highest core in the mask.
    fn apply_core_mask(
        &mut self,
        id: ThreadId,
        new_ideal: Option<CoreId>,
        mask: AffinityMask,
    ) -> Result<(), SchedError> {
        let thread = self.thread_mut(id)?;
        let old_mask = thread.affinity;
        thread.ideal_core = new_ideal;
        thread.affinity = mask;
        if mask != old_mask {
            let active_core = thread.active_core;
            if let Some(active) = active_core
                && !mask.contains(active)
            {
                thread.active_core = new_ideal.or_else(|| mask.highest());
            }
            self.on_thread_affinity_mask_changed(id, old_mask, active_core)?;
        }
        Ok(())
    }

    // ---------------------------------------------------------------------------------------
    // Notifications (k_scheduler.cpp, OnThread*Changed)
    // ---------------------------------------------------------------------------------------

    /// `IncrementScheduledCount(thread)`: the count of the thread's process, or the table's own
    /// for a thread that belongs to no process.
    fn count_scheduled(&mut self, thread: ThreadId) -> Result<(), SchedError> {
        let count = match self.threads.stored(thread)?.process {
            Some(process) => &mut self.record_mut(process)?.scheduled_count,
            None => &mut self.scheduled_count,
        };
        // A counter compared only for equality; wrapping keeps it total.
        *count = count.wrapping_add(1);
        Ok(())
    }

    /// The scheduled count a thread's yields are compared with (`GetScheduledCount` of its
    /// process).
    fn scheduled_count_of(&self, thread: ThreadId) -> Result<u64, SchedError> {
        Ok(match self.threads.stored(thread)?.process {
            Some(process) => self.record(process)?.scheduled_count,
            None => self.scheduled_count,
        })
    }

    /// `KScheduler::OnThreadStateChanged`, given whether the thread was queued before.
    fn on_thread_state_changed(
        &mut self,
        id: ThreadId,
        was_queued: bool,
    ) -> Result<(), SchedError> {
        let now_queued = self.thread(id)?.is_queued();
        if was_queued && !now_queued {
            self.queue.remove(&mut self.threads, id)?;
        } else if !was_queued && now_queued {
            self.queue.push_back(&mut self.threads, id)?;
        } else {
            return Ok(());
        }
        self.count_scheduled(id)?;
        self.update_needed = true;
        Ok(())
    }

    /// `KScheduler::OnThreadPriorityChanged`.
    fn on_thread_priority_changed(
        &mut self,
        caller: CoreId,
        id: ThreadId,
        old_priority: Priority,
    ) -> Result<(), SchedError> {
        if self.thread(id)?.is_queued() {
            let is_running = self.core(caller)?.current == Some(id);
            self.queue
                .change_priority(&mut self.threads, old_priority, is_running, id)?;
            self.count_scheduled(id)?;
            self.update_needed = true;
        }
        Ok(())
    }

    /// `KScheduler::OnThreadAffinityMaskChanged`.
    fn on_thread_affinity_mask_changed(
        &mut self,
        id: ThreadId,
        old_affinity: AffinityMask,
        old_core: Option<CoreId>,
    ) -> Result<(), SchedError> {
        if self.thread(id)?.is_queued() {
            self.queue
                .change_affinity_mask(&mut self.threads, old_core, old_affinity, id)?;
            self.count_scheduled(id)?;
            self.update_needed = true;
        }
        Ok(())
    }

    // ---------------------------------------------------------------------------------------
    // Choosing each core's thread (k_scheduler.cpp, k_scheduler_lock.h)
    // ---------------------------------------------------------------------------------------

    /// Ends a batch of changes, as releasing Eden's scheduler lock does
    /// (`KAbstractSchedulerLock::Unlock`): if anything changed, recompute each core's thread
    /// (migrating suggested threads to idle cores), then signal every core whose thread changed
    /// except `caller`. Returns the cores whose thread changed, `caller` included.
    ///
    /// `now` stamps the threads that stop being a core's choice (Eden stamps them with
    /// `CoreTiming().GetClockTicks()`, the guest counter). The scheduler only compares stamps
    /// with each other, in the yields, so any clock that never runs backwards works; the caller
    /// passes the dispatch's guest time, converted from the environment's clock by
    /// [`CounterRate::ticks_at`](crate::time::CounterRate::ticks_at), the kernel's one conversion
    /// point.
    pub fn update(
        &mut self,
        now: GuestTicks,
        caller: Option<CoreId>,
        signals: &mut dyn CoreSignals,
    ) -> Result<AffinityMask, SchedError> {
        let changed = if self.update_needed {
            self.update_highest_priority_threads(now)?
        } else {
            AffinityMask::EMPTY
        };
        // RescheduleOtherCores / RescheduleCores.
        for core in changed.iter() {
            if Some(core) == caller {
                continue;
            }
            if self.core(core)?.current.is_some() {
                signals.interrupt(core);
            } else {
                signals.wake(core);
            }
        }
        Ok(changed)
    }

    /// `KScheduler::UpdateHighestPriorityThread`: record a core's new choice; returns the core's
    /// bit if it changed.
    fn set_highest(
        &mut self,
        core: CoreId,
        thread: Option<ThreadId>,
        now: GuestTicks,
    ) -> Result<u64, SchedError> {
        let previous = self.core(core)?.highest;
        if previous == thread {
            return Ok(0);
        }
        if let Some(previous) = previous {
            self.count_scheduled(previous)?;
            self.threads.stored_mut(previous)?.last_scheduled_tick = now;
        }
        let state = self.core_mut(core)?;
        state.highest = thread;
        state.needs_scheduling = true;
        Ok(core.bit())
    }

    /// The active core of a thread that a queue walk reached.
    fn active_core_of(&self, id: ThreadId) -> Result<Option<CoreId>, SchedError> {
        Ok(self.threads.stored(id)?.active_core)
    }

    /// The priority of a thread that a queue walk reached.
    fn priority_of(&self, id: ThreadId) -> Result<Priority, SchedError> {
        Ok(self.threads.stored(id)?.priority)
    }

    /// Makes `core` the active core of `id`, which was `from` (`SetActiveCore` then
    /// `ChangeCore`).
    fn migrate(
        &mut self,
        id: ThreadId,
        from: Option<CoreId>,
        to: Option<CoreId>,
        to_front: bool,
    ) -> Result<(), SchedError> {
        self.threads.stored_mut(id)?.active_core = to;
        self.queue
            .change_core(&mut self.threads, from, id, to_front)
    }

    /// `KScheduler::UpdateHighestPriorityThreadsImpl`, without the pinned-thread check.
    fn update_highest_priority_threads(
        &mut self,
        now: GuestTicks,
    ) -> Result<AffinityMask, SchedError> {
        self.update_needed = false;
        let mut changed = 0_u64;
        let mut idle = 0_u64;
        let mut top: Vec<Option<ThreadId>> = vec![None; self.cores.len()];
        let top_of = |top: &[Option<ThreadId>], core: CoreId| {
            top.get(usize::from(core.index()))
                .copied()
                .ok_or(SchedError::Corrupt)
        };
        let set_top = |top: &mut [Option<ThreadId>], core: CoreId, thread: Option<ThreadId>| {
            let slot = top
                .get_mut(usize::from(core.index()))
                .ok_or(SchedError::Corrupt)?;
            *slot = thread;
            Ok::<(), SchedError>(())
        };

        // Each core's scheduled front.
        for core in self.all_cores() {
            let front = self.queue.scheduled_front(core)?;
            if front.is_none() {
                idle |= core.bit();
            }
            set_top(&mut top, core, front)?;
            changed |= self.set_highest(core, front, now)?;
        }

        // Migrate suggested threads to each idle core in turn.
        for core in AffinityMask::from_bits(idle).iter() {
            let Some(first) = self.queue.suggested_front(core)? else {
                continue;
            };
            let mut candidates: Vec<CoreId> = Vec::new();
            let mut suggested = Some(first);
            let mut walk = Walk::new(&self.threads);
            while let Some(thread) = suggested {
                walk.step()?;
                let suggested_core = self.active_core_of(thread)?;
                let top_thread = match suggested_core {
                    Some(c) => top_of(&top, c)?,
                    None => None,
                };
                if top_thread != Some(thread) {
                    // Don't migrate away from a core whose top thread is too urgent.
                    if let Some(top_thread) = top_thread
                        && self.priority_of(top_thread)? < HIGHEST_CORE_MIGRATION_ALLOWED_PRIORITY
                    {
                        break;
                    }
                    // The suggested thread isn't its core's top thread: migrate it.
                    self.migrate(thread, suggested_core, Some(core), false)?;
                    set_top(&mut top, core, Some(thread))?;
                    changed |= self.set_highest(core, Some(thread), now)?;
                    break;
                }
                // It tops its own core: a candidate if nothing else can be moved.
                if let Some(c) = suggested_core {
                    candidates.push(c);
                }
                suggested = self.queue.suggested_next(&self.threads, core, thread)?;
            }

            // No suggestion could move; try moving a candidate core's top thread, if that core
            // has another thread to run instead.
            if suggested.is_none() {
                for candidate in candidates {
                    let Some(moving) = top_of(&top, candidate)? else {
                        continue;
                    };
                    let Some(next) = self
                        .queue
                        .scheduled_next(&self.threads, candidate, moving)?
                    else {
                        continue;
                    };
                    set_top(&mut top, candidate, Some(next))?;
                    changed |= self.set_highest(candidate, Some(next), now)?;
                    self.migrate(moving, Some(candidate), Some(core), false)?;
                    set_top(&mut top, core, Some(moving))?;
                    changed |= self.set_highest(core, Some(moving), now)?;
                    break;
                }
            }
        }
        Ok(AffinityMask::from_bits(changed))
    }

    /// Switches a core to the thread [`Scheduler::update`] chose for it and returns it (`None`:
    /// park). The bookkeeping of Eden's `ScheduleImpl` and `SwitchThread`; the caller saves the
    /// old thread's registers and loads the new one's.
    ///
    /// A thread can only run on one core: if the chosen thread is still current on another
    /// core, this core parks and stays marked as needing scheduling, and the other core wakes it
    /// through `signals` when it lets the thread go.
    pub fn schedule(
        &mut self,
        core: CoreId,
        signals: &mut dyn CoreSignals,
    ) -> Result<Option<ThreadId>, SchedError> {
        let state = self.core_mut(core)?;
        state.needs_scheduling = false;
        let next = state.highest;
        let previous = state.current;
        if next == previous {
            return Ok(previous);
        }

        // Unload the current thread, then wake any core that was waiting for it.
        state.current = None;
        if let Some(previous) = previous {
            for other in self.all_cores() {
                let waiting = self.core(other)?;
                if other != core && waiting.highest == Some(previous) && waiting.current.is_none() {
                    signals.wake(other);
                }
            }
        }

        let Some(next) = next else {
            return Ok(None);
        };
        if self.running_on(next).is_some() {
            // Eden's core spins on the thread's context lock here.
            self.core_mut(core)?.needs_scheduling = true;
            return Ok(None);
        }
        self.thread_mut(next)?.current_core = core;
        self.core_mut(core)?.current = Some(next);
        Ok(Some(next))
    }

    // ---------------------------------------------------------------------------------------
    // Yields (k_scheduler.cpp)
    // ---------------------------------------------------------------------------------------

    /// The calling core's current thread, which every yield acts on.
    fn yielding_thread(&self, core: CoreId) -> Result<ThreadId, SchedError> {
        self.core(core)?
            .current
            .ok_or(SchedError::NoCurrentThread { core: core.index() })
    }

    /// Whether a yield can return at once: the thread's yield count matches the scheduled
    /// count, so nothing changed since its last fruitless yield.
    fn yield_is_redundant(&self, id: ThreadId) -> Result<bool, SchedError> {
        Ok(self.thread(id)?.yield_schedule_count == Some(self.scheduled_count_of(id)?))
    }

    fn note_fruitless_yield(&mut self, id: ThreadId) -> Result<(), SchedError> {
        let count = self.scheduled_count_of(id)?;
        self.thread_mut(id)?.yield_schedule_count = Some(count);
        Ok(())
    }

    /// `svcSleepThread(0)`: `KScheduler::YieldWithoutCoreMigration`. The calling core's current
    /// thread goes to the back of its priority level on its core.
    pub fn yield_without_core_migration(&mut self, core: CoreId) -> Result<(), SchedError> {
        let current = self.yielding_thread(core)?;
        if self.yield_is_redundant(current)? {
            return Ok(());
        }
        let thread = self.thread(current)?;
        if !thread.is_queued() {
            return Ok(());
        }
        let Some(active) = thread.active_core else {
            return Ok(());
        };
        let next = self
            .queue
            .move_to_scheduled_back(&mut self.threads, current, active)?;
        self.count_scheduled(current)?;
        if next == Some(current) {
            self.note_fruitless_yield(current)?;
        } else {
            self.update_needed = true;
        }
        Ok(())
    }

    /// `svcSleepThread(-1)`: `KScheduler::YieldWithCoreMigration`. Like
    /// [`Scheduler::yield_without_core_migration`], then pull the first suggested thread that
    /// isn't running elsewhere and is at least as urgent onto this core, at the front.
    pub fn yield_with_core_migration(&mut self, core: CoreId) -> Result<(), SchedError> {
        let current = self.yielding_thread(core)?;
        if self.yield_is_redundant(current)? {
            return Ok(());
        }
        let thread = self.thread(current)?;
        if !thread.is_queued() {
            return Ok(());
        }
        let Some(core_id) = thread.active_core else {
            return Ok(());
        };
        let current_priority = thread.priority;

        let next = self
            .queue
            .move_to_scheduled_back(&mut self.threads, current, core_id)?;
        self.count_scheduled(current)?;

        let mut recheck = false;
        let mut suggested = self.queue.suggested_front(core_id)?;
        let mut walk = Walk::new(&self.threads);
        while let Some(candidate) = suggested {
            walk.step()?;
            let suggested_core = self.active_core_of(candidate)?;
            let running = match suggested_core {
                Some(c) => self.core(c)?.highest,
                None => None,
            };
            if running != Some(candidate) {
                // Prefer the next thread to a less urgent suggestion, or to an equally urgent one
                // that has waited less.
                let candidate_priority = self.priority_of(candidate)?;
                let next_waited_longer = match next {
                    Some(next) if next != current => {
                        self.thread(next)?.last_scheduled_tick
                            < self.thread(candidate)?.last_scheduled_tick
                    }
                    _ => false,
                };
                if candidate_priority > current_priority
                    || (candidate_priority == current_priority && next_waited_longer)
                {
                    suggested = None;
                    break;
                }
                let allowed = match running {
                    None => true,
                    Some(running) => {
                        self.priority_of(running)? >= HIGHEST_CORE_MIGRATION_ALLOWED_PRIORITY
                    }
                };
                if allowed {
                    self.migrate(candidate, suggested_core, Some(core_id), true)?;
                    self.count_scheduled(candidate)?;
                    break;
                }
                // Couldn't migrate now; a later yield should look again.
                recheck = true;
            }
            suggested = self
                .queue
                .suggested_next(&self.threads, core_id, candidate)?;
        }

        if suggested.is_some() || next != Some(current) {
            self.update_needed = true;
        } else if !recheck {
            self.note_fruitless_yield(current)?;
        }
        Ok(())
    }

    /// `svcSleepThread(-2)`: `KScheduler::YieldToAnyThread`. The calling core's current thread
    /// gives up its core (it stays suggested to every core in its mask); if that leaves the
    /// core with nothing scheduled, the first suggested thread that doesn't top its own core
    /// migrates in.
    pub fn yield_to_any_thread(&mut self, core: CoreId) -> Result<(), SchedError> {
        let current = self.yielding_thread(core)?;
        if self.yield_is_redundant(current)? {
            return Ok(());
        }
        let thread = self.thread(current)?;
        if !thread.is_queued() {
            return Ok(());
        }
        let Some(core_id) = thread.active_core else {
            return Ok(());
        };

        self.migrate(current, Some(core_id), None, false)?;
        self.count_scheduled(current)?;

        if self.queue.scheduled_front(core_id)?.is_some() {
            self.update_needed = true;
            return Ok(());
        }

        let mut suggested = self.queue.suggested_front(core_id)?;
        let mut walk = Walk::new(&self.threads);
        while let Some(candidate) = suggested {
            walk.step()?;
            let suggested_core = self.active_core_of(candidate)?;
            let top = match suggested_core {
                Some(c) => self.queue.scheduled_front(c)?,
                None => None,
            };
            if top != Some(candidate) {
                let allowed = match top {
                    None => true,
                    Some(top) => self.priority_of(top)? >= HIGHEST_CORE_MIGRATION_ALLOWED_PRIORITY,
                };
                if allowed {
                    self.migrate(candidate, suggested_core, Some(core_id), false)?;
                    self.count_scheduled(candidate)?;
                }
                // Whether or not it moved, there was a candidate.
                break;
            }
            suggested = self
                .queue
                .suggested_next(&self.threads, core_id, candidate)?;
        }

        if suggested == Some(current) {
            self.note_fruitless_yield(current)?;
        } else {
            self.update_needed = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
