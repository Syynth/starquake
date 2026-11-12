//! Threads waiting on synchronisation objects: wait lists by object id, wait records, the timer
//! queue, pending cancels, and the one place a wait ends.
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

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use starquake_cpu::GuestRegs;
use starquake_trace::{HostNanos, SVC_ARG_COUNT};

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
/// hardware timer's unit, nanoseconds.
pub const TIMEOUT_SLACK_NS: u64 = 2;

/// The index a wait that no object ended reports: Eden's `SetSyncedIndex(-1)` before
/// `BeginWait`, which a timeout or a cancel leaves in place.
const NO_INDEX: i32 = -1;

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
    ///
    /// Eden keeps the deadline as an `s64`, so a sum above `i64::MAX` turns negative there and
    /// the wait has no timer. Here such a sum is [`Timeout::Never`] too, and so is one that
    /// overflows `u64`, which Eden would wrap (a clock reading past 2^63 ns, about 292 years,
    /// is needed for that).
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
}

/// Names one wait of one thread, so that a timer entry for an earlier wait is told apart from
/// the thread's current one. Every wait the table begins gets the next token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaitToken(u64);

/// Who a wait is for, as the event stream names it: what the dispatcher records the
/// `WaitSynchronization` event with when the wait ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaitOrigin {
    /// The waiting guest thread, as the event stream names it.
    pub thread: starquake_trace::ThreadId,
    /// The core it called from.
    pub core: u8,
    /// The call's decoded arguments, as its event carries them.
    pub args: [u64; SVC_ARG_COUNT],
}

/// What a blocked thread waits on (kernel spec, "Waiting and waking"). Address-arbiter,
/// condition-variable, sleep and reply waits are added by the slices that build them.
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
/// result and index are already in its saved registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Wake {
    /// The thread, as the scheduler names it.
    pub thread: ThreadId,
    /// Who the wait was for, as the event stream names it.
    pub origin: WaitOrigin,
    /// What its `WaitSynchronization` returns in `x0`.
    pub result: ResultCode,
    /// What it returns in `x1`: the index of the object that ended the wait, or -1.
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
/// [`Wait`], the timer queue, pending cancels, and the wakes the dispatcher has yet to record.
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

    /// Removes `thread`'s wait record, takes it off every wait list and out of the timer queue,
    /// and returns the record.
    fn detach(&mut self, thread: ThreadId) -> Option<Wait> {
        let wait = self.waits.remove(&thread)?;
        let Wait::Sync {
            objects,
            token,
            deadline,
            ..
        } = &wait;
        for object in objects {
            if let Some(list) = self.lists.get_mut(object) {
                list.retain(|waiter| *waiter != thread);
                if list.is_empty() {
                    self.lists.remove(object);
                }
            }
        }
        if let Some(deadline) = *deadline {
            self.timers.remove(&TimerEntry {
                deadline,
                token: *token,
                thread,
            });
        }
        Some(wait)
    }
}

/// Whether threads can wait on `object`: Eden's classes derived from `KSynchronizationObject`
/// that starquake has. `KReadableEvent` derives from it directly (`k_readable_event.h`), and
/// `KThread` and `KProcess` through `KWorkerTask` (`k_worker_task.h`). `KEvent` (`k_event.h`) and
/// `KClientSession` (`k_client_session.h`) derive from `KAutoObject` only, so they are not.
#[must_use]
pub const fn is_synchronization_object(object: Object) -> bool {
    match object {
        Object::ReadableEvent(_) | Object::Thread(_) | Object::Process(_) => true,
        Object::Event(_) | Object::Session => false,
    }
}

impl Scheduler {
    /// The synchronisation rows: wait lists, wait records, the timer queue and pending cancels.
    #[must_use]
    pub fn sync(&self) -> &SyncTable {
        self.sync_table()
    }

    /// Drops one reference to `object` through the one teardown path the object table has
    /// ([`crate::ipc::Ipc::release`]: the row goes with its last reference, and a session's
    /// server is told).
    pub(crate) fn release_object(&mut self, object: ObjectId) -> CallResult<()> {
        let (objects, ipc) = self.ipc_mut();
        ipc.release(objects, object)?;
        Ok(())
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
            other @ (Object::Event(_) | Object::Session) => Err(SvcError::Internal(format!(
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
        let deadline = match timeout {
            Timeout::Poll => return Ok(WaitBegun::Ended(KernelError::TimedOut)),
            Timeout::At(deadline) => Some(deadline),
            Timeout::Never => None,
        };
        let thread = self
            .current(core)?
            .ok_or(SchedError::NoCurrentThread { core: core.index() })?;
        if self.sync_table_mut().cancelled.remove(&thread) {
            return Ok(WaitBegun::Ended(KernelError::Cancelled));
        }
        if self.sync_table().waits.contains_key(&thread) {
            return Err(SvcError::Internal(format!(
                "{thread:?} began a wait while it was waiting"
            )));
        }

        // Everything that can fail before the table changes.
        let token = self.sync_table().next_token;
        let next_token = token
            .checked_add(1)
            .ok_or_else(|| SvcError::Internal("the wait tokens ran out".to_owned()))?;
        let token = WaitToken(token);
        *self.context_mut(thread)? = regs.clone();

        let table = self.sync_table_mut();
        table.next_token = next_token;
        for object in objects {
            table.lists.entry(*object).or_default().push_back(thread);
        }
        if let Some(deadline) = deadline {
            table.timers.insert(TimerEntry {
                deadline,
                token,
                thread,
            });
        }
        table.waits.insert(
            thread,
            Wait::Sync {
                objects: objects.to_vec(),
                token,
                deadline,
                origin,
            },
        );
        if let Err(error) = self.set_state(thread, ThreadState::Waiting) {
            // Undo the links, so nothing names a thread that is not waiting.
            self.sync_table_mut().detach(thread);
            return Err(error.into());
        }
        Ok(WaitBegun::Waiting(thread))
    }

    /// Ends `thread`'s wait with `result` and `index`: the one place a wait ends. The thread
    /// leaves every wait list and the timer queue, the result and index are written into its
    /// saved `x0` and `x1` (`WaitSynchronization`'s convention), it is made runnable, the
    /// references its wait held are released, and a [`Wake`] is queued. Each step runs even if
    /// an earlier one failed; the first failure is returned.
    fn end_wait(&mut self, thread: ThreadId, result: ResultCode, index: i32) -> CallResult<()> {
        let Some(Wait::Sync {
            objects, origin, ..
        }) = self.sync_table_mut().detach(thread)
        else {
            return Err(SvcError::Internal(format!("{thread:?} is not waiting")));
        };
        let written = self.write_wait_result(thread, result, index);
        let state = self.set_state(thread, ThreadState::Runnable);
        let released = self.release_objects(&objects);
        self.sync_table_mut().wakes.push(Wake {
            thread,
            origin,
            result,
            index,
        });
        written?;
        state?;
        released
    }

    /// Writes a wait's outputs into the thread's saved registers, with the convention the
    /// dispatcher writes a call that returns at once with.
    fn write_wait_result(
        &mut self,
        thread: ThreadId,
        result: ResultCode,
        index: i32,
    ) -> CallResult<()> {
        let convention = SvcId::WaitSynchronization.convention().ok_or_else(|| {
            SvcError::Internal("WaitSynchronization has no register convention".to_owned())
        })?;
        let outputs = [u64::from(result.raw()), u64::from(index.cast_unsigned())];
        convention
            .write(self.context_mut(thread)?, &outputs)
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

    /// Cancels `thread`'s wait (`KThread::WaitCancel`): a thread in a synchronisation wait
    /// ends it with [`KernelError::Cancelled`] and index -1; any other thread gets a pending
    /// cancel, which its next `WaitSynchronization` that would wait consumes.
    pub fn cancel_synchronization(&mut self, thread: ThreadId) -> CallResult<()> {
        self.thread(thread)?;
        if self.sync_table().waits.contains_key(&thread) {
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
    /// entry was made for, the wait ends with [`KernelError::TimedOut`] and index -1, and the
    /// result is `true`. A late entry (the wait already ended, or the thread is in a later
    /// wait) is ignored: `false`.
    pub fn expire(&mut self, entry: TimerEntry) -> CallResult<bool> {
        match self.sync_table().waits.get(&entry.thread) {
            Some(Wait::Sync { token, .. }) if *token == entry.token => {
                self.end_wait(entry.thread, KernelError::TimedOut.code(), NO_INDEX)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Takes and expires every entry due at `now` ([`Scheduler::take_due_timers`], then
    /// [`Scheduler::expire`] each), and returns how many waits that ended.
    pub fn fire_timers(&mut self, now: HostNanos) -> CallResult<usize> {
        let mut ended = 0_usize;
        for entry in self.take_due_timers(now) {
            if self.expire(entry)? {
                ended = ended.saturating_add(1);
            }
        }
        Ok(ended)
    }

    /// The waits that ended since the last call, in the order they ended. The dispatcher
    /// records each one's `WaitSynchronization` event and reports its thread woken.
    pub fn take_wakes(&mut self) -> Vec<Wake> {
        std::mem::take(&mut self.sync_table_mut().wakes)
    }
}
