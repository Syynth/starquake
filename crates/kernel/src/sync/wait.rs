//! Threads waiting: wait lists by object id, wait records, the timer queue, pending cancels, and
//! the one place a wait ends, for every kind of wait ([`Wait`]).
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later):
//! - `eden/src/core/hle/kernel/k_synchronization_object.cpp`: `KSynchronizationObject::Wait`
//!   ([`Scheduler::wait_synchronization`]), `NotifyAvailable`
//!   ([`Scheduler::notify_available`]), and `ThreadQueueImplForKSynchronizationObjectWait`'s
//!   `NotifyAvailable` and `CancelWait` (`Scheduler::end_wait`).
//! - `eden/src/core/hle/kernel/k_synchronization_object.h`: `LinkNode` appends to the tail of an
//!   object's list, so a list is in the order its threads began to wait, and `NotifyAvailable`
//!   walks it from the head.
//! - `eden/src/core/hle/kernel/k_thread.cpp`: `WaitCancel`
//!   ([`Scheduler::cancel_synchronization`]) and `OnTimer` ([`Scheduler::expire`]).
//! - `eden/src/core/hle/kernel/k_hardware_timer_base.h`: `DoInterruptTaskImpl`, which handles
//!   every task whose time is not after the current time ([`Scheduler::take_due_timers`]).
//! - `eden/src/core/hle/kernel/svc/svc_synchronization.cpp`: the timeout conversion
//!   ([`Timeout::from_nanos`]).
//! - `eden/src/core/hle/kernel/k_address_arbiter.cpp` and `k_condition_variable.cpp`: the
//!   `CancelWait` of `ThreadQueueImplForKAddressArbiter`,
//!   `ThreadQueueImplForKConditionVariableWaitForAddress` and
//!   `ThreadQueueImplForKConditionVariableWaitConditionVariable`: a wait that ends other than by
//!   its own signal leaves the tree it is in and the lock it waits on (`Scheduler::unlink_wait`).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use starquake_cpu::GuestRegs;
use starquake_trace::{HostNanos, SVC_ARG_COUNT};

use super::lock::LockTable;
use super::tree::{ProcessKey, WaitTree};
use crate::id::{ObjectId, ThreadId};
use crate::object::Object;
use crate::result::{CallResult, KernelError, ResultCode, SvcError};
use crate::sched::{CoreId, SchedError, Scheduler, ThreadState};
use crate::svc::SvcId;

/// The most handles one `WaitSynchronization` takes: Eden's `Svc::ArgumentHandleCountMax`
/// (`eden/src/core/hle/kernel/svc_common.h` @ cf654052, `0x40`).
pub const ARGUMENT_HANDLE_COUNT_MAX: usize = 0x40;

/// What `WaitSynchronization` adds to a positive timeout besides the current time: Eden's
/// `ticks += 2` (`eden/src/core/hle/kernel/svc/svc_synchronization.cpp` @ cf654052), in the
/// hardware timer's unit, nanoseconds. `WaitForAddress` and `WaitProcessWideKeyAtomic` add the
/// same (`offset_tick + 2` in `svc_address_arbiter.cpp` and `svc_condition_variable.cpp`).
pub const TIMEOUT_SLACK_NS: u64 = 2;

/// The index a wait that no object ended reports: Eden's `SetSyncedIndex(-1)` before
/// `BeginWait`, which a timeout or a cancel leaves in place.
pub(super) const NO_INDEX: i32 = -1;

/// When a wait ends if no object ends it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Timeout {
    /// At once: a timeout of zero (`timeout == 0` in `KSynchronizationObject::Wait`), which
    /// returns [`KernelError::TimedOut`] instead of waiting.
    Poll,
    /// When the environment's clock reaches this reading.
    At(HostNanos),
    /// Never: a negative timeout.
    Never,
}

impl Timeout {
    /// The timeout of a `WaitSynchronization` made at `now` with `timeout_ns`, as Eden's
    /// `WaitSynchronization` converts it: a positive timeout ends at `now + timeout_ns +`
    /// [`TIMEOUT_SLACK_NS`], zero is [`Timeout::Poll`], and a negative one is [`Timeout::Never`].
    /// `WaitForAddress` and `WaitProcessWideKeyAtomic` convert theirs the same way.
    ///
    /// Eden keeps the deadline as an `s64`, so a sum above `i64::MAX` turns negative there and
    /// the wait has no timer (`WaitForAddress` and `WaitProcessWideKeyAtomic` set it to
    /// `i64::MAX` instead, a deadline no clock reaches). Here such a sum is [`Timeout::Never`]
    /// for all three, and so is one that overflows `u64`, which Eden would wrap (a clock reading
    /// past 2^63 ns, about 292 years, is needed for that).
    #[must_use]
    pub fn from_nanos(now: HostNanos, timeout_ns: i64) -> Self {
        if timeout_ns == 0 {
            return Self::Poll;
        }
        let Ok(timeout) = u64::try_from(timeout_ns) else {
            return Self::Never;
        };
        let latest = i64::MAX.cast_unsigned();
        match now
            .0
            .checked_add(timeout)
            .and_then(|sum| sum.checked_add(TIMEOUT_SLACK_NS))
        {
            Some(deadline) if deadline <= latest => Self::At(HostNanos(deadline)),
            _ => Self::Never,
        }
    }

    /// The deadline a wait with this timeout registers, or `None` for no timer; `Err` with
    /// [`KernelError::TimedOut`] for [`Timeout::Poll`], which does not wait.
    pub(super) fn deadline(self) -> Result<Option<HostNanos>, KernelError> {
        match self {
            Self::Poll => Err(KernelError::TimedOut),
            Self::At(deadline) => Ok(Some(deadline)),
            Self::Never => Ok(None),
        }
    }
}

/// Names one wait of one thread, so that a timer entry for an earlier wait is told apart from
/// the thread's current one. Every wait the table begins gets the next token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaitToken(u64);

/// Who a wait is for, as the event stream names it: what the dispatcher records the call's
/// event with when the wait ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaitOrigin {
    /// The waiting guest thread, as the event stream names it.
    pub thread: starquake_trace::ThreadId,
    /// The core it called from.
    pub core: u8,
    /// The call's decoded arguments, as its event carries them.
    pub args: [u64; SVC_ARG_COUNT],
}

/// The thread making a call that may wait: the core it runs on, its registers (saved as its
/// context if it waits) and who its event is recorded for.
#[derive(Clone, Copy, Debug)]
pub struct WaitCall<'a> {
    /// The calling core, whose current thread waits.
    pub core: CoreId,
    /// The caller's registers.
    pub regs: &'a GuestRegs,
    /// Who the call's event is recorded for.
    pub origin: WaitOrigin,
}

/// What a blocked thread waits on (kernel spec, "Waiting and waking"). The reply wait of an IPC
/// request is the IPC table's (`crate::ipc`), not a record here: a request ends it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Wait {
    /// A `WaitSynchronization`: any of `objects` being signalled ends it.
    Sync {
        /// The objects, in the order the guest listed their handles. The wait holds one
        /// reference to each.
        objects: Vec<ObjectId>,
        /// This wait's token.
        token: WaitToken,
        /// When its timer expires, if it has one.
        deadline: Option<HostNanos>,
        /// Who to record the call's event for when it ends.
        origin: WaitOrigin,
    },
    /// A `WaitForAddress`: in the process's address-arbiter tree under `key` until a
    /// `SignalToAddress` takes it out ([`crate::sync::arbiter`]).
    Address {
        /// The process and the address.
        key: ProcessKey,
        /// This wait's token.
        token: WaitToken,
        /// When its timer expires, if it has one.
        deadline: Option<HostNanos>,
        /// Who to record the call's event for when it ends.
        origin: WaitOrigin,
    },
    /// A `WaitProcessWideKeyAtomic`: in the process's condition-variable tree under `key` until
    /// a `SignalProcessWideKey` takes it out, then, if the lock at `lock` is held, waiting on
    /// its owner as `ArbitrateLock` does ([`crate::sync::condvar`]). The timer runs through
    /// both.
    CondVar {
        /// The process and the condition-variable key.
        key: ProcessKey,
        /// The lock's address.
        lock: u64,
        /// This wait's token.
        token: WaitToken,
        /// When its timer expires, if it has one.
        deadline: Option<HostNanos>,
        /// Who to record the call's event for when it ends.
        origin: WaitOrigin,
    },
    /// An `ArbitrateLock`: waiting on the owner of the lock at `lock` ([`crate::sync::lock`]).
    /// It has no timer.
    Lock {
        /// The lock's address.
        lock: u64,
        /// The owner's object, which the wait holds a reference to, as Eden's
        /// `KConditionVariable::WaitForAddress` keeps the reference `GetObjectWithoutPseudoHandle`
        /// opened until the wait ends.
        owner: ObjectId,
        /// This wait's token.
        token: WaitToken,
        /// Who to record the call's event for when it ends.
        origin: WaitOrigin,
    },
    /// A `SleepThread` with a positive time (`KThread::Sleep`): only its timer ends it, and the
    /// call returns nothing (`svcSleepThread` is void, and Eden notes that Nintendo does not
    /// check the sleep's result), so its event records success.
    Sleep {
        /// This wait's token.
        token: WaitToken,
        /// When it ends: `None` for a time so long that the deadline does not fit, which Eden
        /// turns into `i64::MAX`, a deadline no clock reaches.
        deadline: Option<HostNanos>,
        /// Who to record the call's event for when it ends.
        origin: WaitOrigin,
    },
}

impl Wait {
    /// This wait's token.
    #[must_use]
    pub const fn token(&self) -> WaitToken {
        match self {
            Self::Sync { token, .. }
            | Self::Address { token, .. }
            | Self::CondVar { token, .. }
            | Self::Lock { token, .. }
            | Self::Sleep { token, .. } => *token,
        }
    }

    /// When its timer expires, if it has one.
    #[must_use]
    pub const fn deadline(&self) -> Option<HostNanos> {
        match self {
            Self::Sync { deadline, .. }
            | Self::Address { deadline, .. }
            | Self::CondVar { deadline, .. }
            | Self::Sleep { deadline, .. } => *deadline,
            Self::Lock { .. } => None,
        }
    }

    /// Who the wait is for.
    #[must_use]
    pub const fn origin(&self) -> WaitOrigin {
        match self {
            Self::Sync { origin, .. }
            | Self::Address { origin, .. }
            | Self::CondVar { origin, .. }
            | Self::Lock { origin, .. }
            | Self::Sleep { origin, .. } => *origin,
        }
    }

    /// The system call the wait is in: its result is written with that call's register
    /// convention, and the call's event is recorded when the wait ends.
    #[must_use]
    pub const fn call(&self) -> SvcId {
        match self {
            Self::Sync { .. } => SvcId::WaitSynchronization,
            Self::Address { .. } => SvcId::WaitForAddress,
            Self::CondVar { .. } => SvcId::WaitProcessWideKeyAtomic,
            Self::Lock { .. } => SvcId::ArbitrateLock,
            Self::Sleep { .. } => SvcId::SleepThread,
        }
    }

    /// The object references the wait holds, released when it ends.
    fn references(&self) -> &[ObjectId] {
        match self {
            Self::Sync { objects, .. } => objects,
            Self::Lock { owner, .. } => std::slice::from_ref(owner),
            Self::Address { .. } | Self::CondVar { .. } | Self::Sleep { .. } => &[],
        }
    }

    /// The tree the wait puts its thread in, and the key: what a priority change takes the
    /// thread out of and puts it back into.
    pub(super) const fn tree(&self) -> Option<(Tree, ProcessKey)> {
        match self {
            Self::Address { key, .. } => Some((Tree::Arbiter, *key)),
            Self::CondVar { key, .. } => Some((Tree::CondVar, *key)),
            Self::Sync { .. } | Self::Lock { .. } | Self::Sleep { .. } => None,
        }
    }
}

/// One of the two kinds of [`WaitTree`] (Eden's `ConditionVariableThreadTree`, which
/// `m_condvar_tree` points to for both).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Tree {
    /// The address-arbiter tree.
    Arbiter,
    /// The condition-variable tree.
    CondVar,
}

/// A timeout in the timer queue: the deadline, the wait it is for, and its thread. Ordered by
/// deadline, then by the order the waits began.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimerEntry {
    /// When it expires: a reading of the environment's clock.
    pub deadline: HostNanos,
    /// The wait it ends.
    pub token: WaitToken,
    /// The waiting thread.
    pub thread: ThreadId,
}

/// A wait that ended, for the dispatcher to record and report: the thread is runnable and its
/// result (and for `WaitSynchronization` its index) is already in its saved registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Wake {
    /// The thread, as the scheduler names it.
    pub thread: ThreadId,
    /// Who the wait was for, as the event stream names it.
    pub origin: WaitOrigin,
    /// The system call the wait was in ([`Wait::call`]): the event is recorded for it.
    pub call: SvcId,
    /// What the call returns in `x0`.
    pub result: ResultCode,
    /// What a `WaitSynchronization` returns in `x1`: the index of the object that ended the
    /// wait, or -1. The other calls return only a result; for them it is -1 and written nowhere.
    pub index: i32,
}

/// How [`Scheduler::wait_synchronization`] ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WaitBegun {
    /// An object was already signalled: the call succeeds at once with its index, the first
    /// signalled one in the list.
    Signalled(usize),
    /// The call ends at once with this result: [`KernelError::TimedOut`] for a zero timeout,
    /// [`KernelError::Cancelled`] for a pending cancel. Eden's `Wait` returns before it assigns
    /// the index, so the index output stays as the wrapper initialised it, zero.
    Ended(KernelError),
    /// The thread waits. Its registers are saved as its context, and the wake writes the
    /// result there.
    Waiting(ThreadId),
}

/// The scheduler's synchronisation rows: wait lists by object id, each waiting thread's
/// [`Wait`], the timer queue, pending cancels, the address-arbiter and condition-variable
/// trees, the locks threads hold and wait on, and the wakes the dispatcher has yet to record.
/// See the module docs of [`crate::sync`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SyncTable {
    /// Each object's waiting threads, in the order they began to wait. A thread waiting on the
    /// same object twice is listed twice. An object with no waiters has no entry.
    lists: BTreeMap<ObjectId, VecDeque<ThreadId>>,
    waits: BTreeMap<ThreadId, Wait>,
    timers: BTreeSet<TimerEntry>,
    /// Threads with a cancel pending (`m_wait_cancelled`).
    cancelled: BTreeSet<ThreadId>,
    wakes: Vec<Wake>,
    next_token: u64,
    /// Every process's address-arbiter tree (`KProcess::m_address_arbiter`).
    pub(super) arbiter: WaitTree,
    /// Every process's condition-variable tree (`KProcess::m_cond_var`).
    pub(super) condvars: WaitTree,
    /// Held locks, lock waiters and address keys.
    pub(super) locks: LockTable,
}

impl SyncTable {
    /// The threads waiting on `object`, in the order they began to wait.
    pub fn waiters(&self, object: ObjectId) -> impl Iterator<Item = ThreadId> + '_ {
        self.lists.get(&object).into_iter().flatten().copied()
    }

    /// What `thread` waits on, if it is in a wait this table keeps.
    #[must_use]
    pub fn wait(&self, thread: ThreadId) -> Option<&Wait> {
        self.waits.get(&thread)
    }

    /// Whether a cancel is pending for `thread`: its next `WaitSynchronization` that would wait
    /// returns [`KernelError::Cancelled`] instead.
    #[must_use]
    pub fn is_cancel_pending(&self, thread: ThreadId) -> bool {
        self.cancelled.contains(&thread)
    }

    /// The earliest deadline in the timer queue: when the timer thread should next call
    /// [`Scheduler::take_due_timers`].
    #[must_use]
    pub fn next_deadline(&self) -> Option<HostNanos> {
        self.timers.first().map(|entry| entry.deadline)
    }

    /// The entries in the timer queue, earliest first.
    pub fn timers(&self) -> impl Iterator<Item = TimerEntry> + '_ {
        self.timers.iter().copied()
    }

    /// The address-arbiter trees of every process: the threads in a `WaitForAddress`.
    #[must_use]
    pub fn arbiter(&self) -> &WaitTree {
        &self.arbiter
    }

    /// The condition-variable trees of every process: the threads in a
    /// `WaitProcessWideKeyAtomic` that no signal has taken out yet.
    #[must_use]
    pub fn condition_variables(&self) -> &WaitTree {
        &self.condvars
    }

    /// The locks threads hold and wait on.
    #[must_use]
    pub fn locks(&self) -> &LockTable {
        &self.locks
    }

    /// The tree `which` names.
    pub(super) fn tree_mut(&mut self, which: Tree) -> &mut WaitTree {
        match which {
            Tree::Arbiter => &mut self.arbiter,
            Tree::CondVar => &mut self.condvars,
        }
    }

    /// The tree `which` names, to read.
    pub(super) fn tree(&self, which: Tree) -> &WaitTree {
        match which {
            Tree::Arbiter => &self.arbiter,
            Tree::CondVar => &self.condvars,
        }
    }

    /// How many objects have a wait list. Test support for the rule that an object with no
    /// waiters has no entry.
    #[cfg(test)]
    pub(super) fn list_count(&self) -> usize {
        self.lists.len()
    }

    /// Removes `thread`'s wait record, takes it off every wait list and tree it is on and out of
    /// the timer queue, and returns the record. A lock it waits on is the scheduler's to leave
    /// (`Scheduler::unlink_wait`), because leaving one changes priorities.
    fn detach(&mut self, thread: ThreadId) -> Option<Wait> {
        let wait = self.waits.remove(&thread)?;
        if let Wait::Sync { objects, .. } = &wait {
            for object in objects {
                if let Some(list) = self.lists.get_mut(object) {
                    list.retain(|waiter| *waiter != thread);
                    if list.is_empty() {
                        self.lists.remove(object);
                    }
                }
            }
        }
        if let Some((which, key)) = wait.tree() {
            self.tree_mut(which).remove(key, thread);
        }
        if let Some(deadline) = wait.deadline() {
            self.timers.remove(&TimerEntry {
                deadline,
                token: wait.token(),
                thread,
            });
        }
        Some(wait)
    }
}

/// Whether threads can wait on `object`: Eden's classes derived from `KSynchronizationObject`
/// that starquake has. `KReadableEvent` derives from it directly (`k_readable_event.h`), and
/// `KThread` and `KProcess` through `KWorkerTask` (`k_worker_task.h`). `KEvent` (`k_event.h`) and
/// `KClientSession` (`k_client_session.h`) derive from `KAutoObject` only, so they are not, and
/// nor are `KSharedMemory` (`k_shared_memory.h`) and `KTransferMemory` (`k_transfer_memory.h`).
#[must_use]
pub const fn is_synchronization_object(object: Object) -> bool {
    match object {
        Object::ReadableEvent(_) | Object::Thread(_) | Object::Process(_) => true,
        Object::Event(_)
        | Object::Session
        | Object::SharedMemory(_)
        | Object::TransferMemory(_) => false,
    }
}

impl Scheduler {
    /// The synchronisation rows: wait lists, wait records, the timer queue, pending cancels,
    /// the arbiter and condition-variable trees, and the locks.
    #[must_use]
    pub fn sync(&self) -> &SyncTable {
        self.sync_table()
    }

    /// Drops one reference to `object` through the one teardown path the object table has
    /// ([`Scheduler::close_object`]: the row goes with its last reference, a session's server
    /// is told and a thread is destroyed).
    pub(crate) fn release_object(&mut self, object: ObjectId) -> CallResult<()> {
        self.close_object(object)
    }

    /// Drops one reference to each of `objects`, all of them even when one fails; the first
    /// failure is returned.
    pub(crate) fn release_objects(&mut self, objects: &[ObjectId]) -> CallResult<()> {
        let mut result = Ok(());
        for object in objects {
            let released = self.release_object(*object);
            if result.is_ok() {
                result = released;
            }
        }
        result
    }

    /// Whether the synchronisation object `object` is signalled (`IsSignaled`): a readable
    /// event that is signalled; a thread whose state is [`ThreadState::Terminated`]; a process
    /// never (see the module docs of [`crate::sync`]). Anything else is not a synchronisation
    /// object, which only the kernel's own bookkeeping can ask about.
    pub fn is_signalled(&self, object: ObjectId) -> CallResult<bool> {
        match self.objects().get(object)? {
            Object::ReadableEvent(event) => Ok(event.is_signalled()),
            Object::Thread(thread) => Ok(self.thread(thread)?.state() == ThreadState::Terminated),
            Object::Process(_) => Ok(false),
            other @ (Object::Event(_)
            | Object::Session
            | Object::SharedMemory(_)
            | Object::TransferMemory(_)) => Err(SvcError::Internal(format!(
                "{object:?} is {other:?}, not a synchronisation object"
            ))),
        }
    }

    /// `KSynchronizationObject::Wait` for the thread `core` is running, which makes the call
    /// with `regs` and is recorded as `origin`. Takes over one reference to each of `objects`,
    /// which must be synchronisation objects: they are released at once unless the thread
    /// waits, and when its wait ends otherwise.
    ///
    /// In Eden's order: an object that is already signalled ends the call with the first such
    /// index; then a zero timeout ends it with [`KernelError::TimedOut`]; then a pending cancel
    /// ends it with [`KernelError::Cancelled`] and is cleared. Otherwise the thread is put on
    /// every object's wait list, its timer (if any) in the timer queue, its registers saved as
    /// its context, and it waits ([`ThreadState::Waiting`]). Only the cancel check and the
    /// wait need the calling thread, so a core running no thread the scheduler knows of is an
    /// internal failure only then ([`SchedError::NoCurrentThread`]).
    pub fn wait_synchronization(
        &mut self,
        core: CoreId,
        objects: &[ObjectId],
        timeout: Timeout,
        regs: &GuestRegs,
        origin: WaitOrigin,
    ) -> CallResult<WaitBegun> {
        let begun = self.begin_wait(core, objects, timeout, regs, origin);
        if matches!(begun, Ok(WaitBegun::Waiting(_))) {
            return begun;
        }
        // The call is over (or failed before the thread waited): Eden's `SCOPE_EXIT` closes
        // the references here.
        let released = self.release_objects(objects);
        let begun = begun?;
        released?;
        Ok(begun)
    }

    fn begin_wait(
        &mut self,
        core: CoreId,
        objects: &[ObjectId],
        timeout: Timeout,
        regs: &GuestRegs,
        origin: WaitOrigin,
    ) -> CallResult<WaitBegun> {
        for (index, object) in objects.iter().enumerate() {
            if self.is_signalled(*object)? {
                return Ok(WaitBegun::Signalled(index));
            }
        }
        let deadline = match timeout.deadline() {
            Ok(deadline) => deadline,
            Err(error) => return Ok(WaitBegun::Ended(error)),
        };
        let thread = self.calling_thread(core)?;
        if self.sync_table_mut().cancelled.remove(&thread) {
            return Ok(WaitBegun::Ended(KernelError::Cancelled));
        }
        self.block(thread, regs, None, |token| Wait::Sync {
            objects: objects.to_vec(),
            token,
            deadline,
            origin,
        })?;
        Ok(WaitBegun::Waiting(thread))
    }

    /// The thread `core` runs: the thread a call that may wait is for. A core running no thread
    /// the scheduler knows of is an internal failure ([`SchedError::NoCurrentThread`]).
    pub(crate) fn calling_thread(&self, core: CoreId) -> CallResult<ThreadId> {
        Ok(self
            .current(core)?
            .ok_or(SchedError::NoCurrentThread { core: core.index() })?)
    }

    /// Fails unless `thread` is in no wait: a thread that is running can't be.
    pub(super) fn ensure_not_waiting(&self, thread: ThreadId) -> CallResult<()> {
        if self.sync_table().waits.contains_key(&thread) {
            return Err(SvcError::Internal(format!(
                "{thread:?} began a wait while it was waiting"
            )));
        }
        Ok(())
    }

    /// Makes `thread` wait (`BeginWait`): the wait `make` builds from its new token is linked
    /// where it belongs (an object's wait lists, or the arbiter or condition-variable tree at
    /// the thread's priority), its timer (if any) goes in the timer queue, `regs` are saved as
    /// the thread's context, and it becomes [`ThreadState::Waiting`]. A lock it waits on was
    /// joined before (`AddWaiter` comes before `BeginWait` in Eden too). If anything fails, the
    /// thread leaves everything it joined, the lock included (`caller` is the core whose
    /// priority bookkeeping that is, if any), so nothing names a thread that is not waiting.
    /// The references the wait would hold stay the caller's to release. A thread that is in a
    /// wait already is an internal failure, before anything changes.
    pub(super) fn block(
        &mut self,
        thread: ThreadId,
        regs: &GuestRegs,
        caller: Option<CoreId>,
        make: impl FnOnce(WaitToken) -> Wait,
    ) -> CallResult<()> {
        self.ensure_not_waiting(thread)?;
        let blocked = self.try_block(thread, regs, make);
        if let Err(error) = blocked {
            // Undo the links; the wait's references are the caller's.
            let (_, unlinked) = self.unlink_wait(thread, caller);
            if let Err(undo) = unlinked {
                tracing::error!(%undo, "a wait that failed to begin could not be undone");
            }
            return Err(error);
        }
        Ok(())
    }

    fn try_block(
        &mut self,
        thread: ThreadId,
        regs: &GuestRegs,
        make: impl FnOnce(WaitToken) -> Wait,
    ) -> CallResult<()> {
        // Everything that can fail before the table changes.
        let token = self.sync_table().next_token;
        let next_token = token
            .checked_add(1)
            .ok_or_else(|| SvcError::Internal("the wait tokens ran out".to_owned()))?;
        let wait = make(WaitToken(token));
        let place = match wait.tree() {
            Some((which, key)) => Some((which, key, self.tree_place(which, key, thread)?)),
            None => None,
        };
        *self.context_mut(thread)? = regs.clone();

        let table = self.sync_table_mut();
        table.next_token = next_token;
        if let Wait::Sync { objects, .. } = &wait {
            for object in objects {
                table.lists.entry(*object).or_default().push_back(thread);
            }
        }
        if let Some((which, key, index)) = place {
            table.tree_mut(which).insert_at(key, index, thread);
        }
        if let Some(deadline) = wait.deadline() {
            table.timers.insert(TimerEntry {
                deadline,
                token: wait.token(),
                thread,
            });
        }
        table.waits.insert(thread, wait);
        self.set_state(thread, ThreadState::Waiting)?;
        Ok(())
    }

    /// Where `thread` goes among `key`'s threads in the tree `which`: after every thread whose
    /// priority is not less urgent (`ConditionVariableComparator` returns 1 for a tie).
    pub(super) fn tree_place(
        &self,
        which: Tree,
        key: ProcessKey,
        thread: ThreadId,
    ) -> Result<usize, SchedError> {
        let priority = self.thread(thread)?.priority();
        let members = self.sync_table().tree(which).waiters(key);
        for (index, member) in members.iter().enumerate() {
            if self.thread(*member)?.priority() > priority {
                return Ok(index);
            }
        }
        Ok(members.len())
    }

    /// Takes `thread` out of its wait without ending it (the queue-specific half of Eden's
    /// `CancelWait`): its record goes, it leaves every wait list and tree and the timer queue
    /// (`SyncTable::detach`), and it leaves the lock it waits on, if any (`RemoveWaiter`, which
    /// may lower the owner's inherited priority; `caller` is the core making the call, for the
    /// scheduler's front-of-queue rule). Its address key is forgotten. Returns the record and
    /// whether leaving the lock succeeded.
    pub(super) fn unlink_wait(
        &mut self,
        thread: ThreadId,
        caller: Option<CoreId>,
    ) -> (Option<Wait>, CallResult<()>) {
        let wait = self.sync_table_mut().detach(thread);
        let left = match self.sync_table().locks.owner_of(thread) {
            Some(owner) => self.remove_waiter(caller, owner, thread),
            None => Ok(()),
        };
        self.sync_table_mut().locks.forget_address_key(thread);
        (wait, left.map_err(SvcError::from))
    }

    /// Ends `thread`'s wait with `result` and `index`, as a wait that no core's call ended (a
    /// timer, a signal from a host thread).
    fn end_wait(&mut self, thread: ThreadId, result: ResultCode, index: i32) -> CallResult<()> {
        self.end_wait_from(None, thread, result, index)
    }

    /// Ends `thread`'s wait with `result` and `index`: the one place a wait ends. The thread
    /// leaves every wait list, tree, lock and the timer queue ([`Scheduler::unlink_wait`]), the
    /// result is written into its saved `x0` (and for `WaitSynchronization` the index into
    /// `x1`) with its call's convention, it is made runnable, the references its wait held are
    /// released, and a [`Wake`] is queued. Each step runs even if an earlier one failed; the
    /// first failure is returned.
    pub(super) fn end_wait_from(
        &mut self,
        caller: Option<CoreId>,
        thread: ThreadId,
        result: ResultCode,
        index: i32,
    ) -> CallResult<()> {
        let (wait, unlinked) = self.unlink_wait(thread, caller);
        let Some(wait) = wait else {
            unlinked?;
            return Err(SvcError::Internal(format!("{thread:?} is not waiting")));
        };
        let call = wait.call();
        let written = self.write_wait_result(thread, call, result, index);
        let state = self.set_state(thread, ThreadState::Runnable);
        let released = self.release_objects(wait.references());
        self.sync_table_mut().wakes.push(Wake {
            thread,
            origin: wait.origin(),
            call,
            result,
            index,
        });
        unlinked?;
        written?;
        state?;
        released
    }

    /// Writes a wait's outputs into the thread's saved registers, with the convention the
    /// dispatcher writes `call` with when it returns at once: the result, and for
    /// `WaitSynchronization` the index after it.
    fn write_wait_result(
        &mut self,
        thread: ThreadId,
        call: SvcId,
        result: ResultCode,
        index: i32,
    ) -> CallResult<()> {
        let convention = call.convention().ok_or_else(|| {
            SvcError::Internal(format!("{} has no register convention", call.name()))
        })?;
        if convention.outputs.is_empty() {
            // `SleepThread` returns nothing: its saved registers stay as they were.
            return Ok(());
        }
        let result = u64::from(result.raw());
        let with_index = [result, u64::from(index.cast_unsigned())];
        let outputs: &[u64] = if call == SvcId::WaitSynchronization {
            &with_index
        } else {
            &with_index[..1]
        };
        convention
            .write(self.context_mut(thread)?, outputs)
            .map_err(|error| SvcError::internal(&error))
    }

    /// Wakes the threads waiting on `object` if it is signalled
    /// (`KSynchronizationObject::NotifyAvailable`), in the order they began to wait. Each ends
    /// its wait with success and the index of `object` in its list (the first, if listed more
    /// than once). The object stays signalled: an event wakes every waiter.
    pub fn notify_available(&mut self, object: ObjectId) -> CallResult<()> {
        if !self.is_signalled(object)? {
            return Ok(());
        }
        let waiting: Vec<ThreadId> = self.sync_table().waiters(object).collect();
        let mut result = Ok(());
        for thread in waiting {
            // A thread listed twice was woken at its first entry.
            let Some(Wait::Sync { objects, .. }) = self.sync_table().waits.get(&thread) else {
                continue;
            };
            let index = objects
                .iter()
                .position(|listed| *listed == object)
                .and_then(|index| i32::try_from(index).ok());
            // Every waiter is woken even if one cannot be; the first failure is returned.
            let woken = match index {
                Some(index) => self.end_wait(thread, ResultCode::SUCCESS, index),
                None => Err(SvcError::Internal(format!(
                    "{thread:?} is listed on {object:?}, which it does not wait on"
                ))),
            };
            if result.is_ok() {
                result = woken;
            }
        }
        result
    }

    /// Forgets `thread`'s synchronisation state as it terminates: a wait it is in ends with no
    /// result written and no [`Wake`] queued (the thread will not run again); it leaves the
    /// tree and the lock it waits on; the references that wait held are released; and a pending
    /// cancel goes. Eden cancels a terminating thread's wait with `ResultTerminationRequested`
    /// and lets the thread run to its exit (`KThread::RequestTerminate`); termination here ends
    /// the thread in the call instead ([`Scheduler::terminate_thread`]). The locks the thread
    /// holds keep their waiters until it is destroyed (`Scheduler::release_held_locks`, as
    /// Eden's `KThread::Finalize`). Each step runs even if an earlier one failed.
    pub(crate) fn forget_waits(&mut self, thread: ThreadId) -> CallResult<()> {
        self.sync_table_mut().cancelled.remove(&thread);
        let (wait, unlinked) = self.unlink_wait(thread, None);
        let released = match wait {
            Some(wait) => self.release_objects(wait.references()),
            None => Ok(()),
        };
        unlinked?;
        released
    }

    /// Cancels `thread`'s wait (`KThread::WaitCancel`): a thread in a synchronisation wait
    /// ends it with [`KernelError::Cancelled`] and index -1. Any other thread gets a pending
    /// cancel, which its next `WaitSynchronization` that would wait consumes: Eden cancels only
    /// a waiting thread that is `m_cancellable`, which only `KSynchronizationObject::Wait` sets,
    /// and notes `m_wait_cancelled` otherwise, so a thread in an arbiter, condition-variable or
    /// lock wait keeps waiting.
    pub fn cancel_synchronization(&mut self, thread: ThreadId) -> CallResult<()> {
        self.thread(thread)?;
        if matches!(
            self.sync_table().waits.get(&thread),
            Some(Wait::Sync { .. })
        ) {
            self.sync_table_mut().cancelled.remove(&thread);
            self.end_wait(thread, KernelError::Cancelled.code(), NO_INDEX)
        } else {
            self.sync_table_mut().cancelled.insert(thread);
            Ok(())
        }
    }

    /// Takes every entry whose deadline is not after `now` out of the timer queue, earliest
    /// first, without ending their waits: [`Scheduler::expire`] does that. Eden's
    /// `DoInterruptTaskImpl` keeps a task whose time is after the current time.
    pub fn take_due_timers(&mut self, now: HostNanos) -> Vec<TimerEntry> {
        let timers = &mut self.sync_table_mut().timers;
        let mut due = Vec::new();
        while let Some(first) = timers.first()
            && first.deadline <= now
        {
            due.extend(timers.pop_first());
        }
        due
    }

    /// Expires a timer entry (`KThread::OnTimer`): if its thread is still in the wait the
    /// entry was made for, the wait ends with [`KernelError::TimedOut`] (and index -1), or for
    /// a sleep with success (see [`Wait::Sleep`]), and the result is `true`. A late entry (the
    /// wait already ended, or the thread is in a later wait) is ignored: `false`.
    pub fn expire(&mut self, entry: TimerEntry) -> CallResult<bool> {
        let result = match self.sync_table().waits.get(&entry.thread) {
            Some(Wait::Sleep { token, .. }) if *token == entry.token => ResultCode::SUCCESS,
            Some(wait) if wait.token() == entry.token => KernelError::TimedOut.code(),
            _ => return Ok(false),
        };
        self.end_wait(entry.thread, result, NO_INDEX)?;
        Ok(true)
    }

    /// `KThread::Sleep` for the thread `core` runs, which makes the call with `regs` and is
    /// recorded as `origin`: the thread waits until `timeout` ([`Wait::Sleep`]). A
    /// [`Timeout::Poll`], which no positive time converts to, does not wait (`None`); a
    /// [`Timeout::Never`] waits with no timer, as Eden's `i64::MAX` deadline does.
    pub fn sleep(&mut self, call: WaitCall<'_>, timeout: Timeout) -> CallResult<Option<ThreadId>> {
        let Ok(deadline) = timeout.deadline() else {
            return Ok(None);
        };
        let thread = self.calling_thread(call.core)?;
        self.block(thread, call.regs, Some(call.core), |token| Wait::Sleep {
            token,
            deadline,
            origin: call.origin,
        })?;
        Ok(Some(thread))
    }

    /// Takes and expires every entry due at `now` ([`Scheduler::take_due_timers`], then
    /// [`Scheduler::expire`] each), and returns how many waits that ended. Every taken entry is
    /// expired even if one fails, since a taken entry is no longer in the queue and nothing else
    /// would end its wait; the first failure is returned.
    pub fn fire_timers(&mut self, now: HostNanos) -> CallResult<usize> {
        let mut ended = 0_usize;
        let mut result = Ok(());
        for entry in self.take_due_timers(now) {
            match self.expire(entry) {
                Ok(true) => ended = ended.saturating_add(1),
                Ok(false) => {}
                Err(error) => {
                    if result.is_ok() {
                        result = Err(error);
                    }
                }
            }
        }
        result.map(|()| ended)
    }

    /// The waits that ended since the last call, in the order they ended. The dispatcher
    /// records each one's event and reports its thread woken.
    pub fn take_wakes(&mut self) -> Vec<Wake> {
        std::mem::take(&mut self.sync_table_mut().wakes)
    }
}
