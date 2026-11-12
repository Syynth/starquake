//! Synchronisation: events and threads waiting on objects (kernel spec, "Waiting and waking";
//! port slice 4), and the address arbiter, condition variables and lock arbitration with
//! priority inheritance (port slice 5).
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), paths relative to the reference folder:
//! - `eden/src/core/hle/kernel/k_synchronization_object.{h,cpp}`: `Wait` (the checks before a
//!   thread waits, then linking it to every object), `NotifyAvailable` and the wait queue's
//!   `NotifyAvailable` and `CancelWait` ([`wait`]).
//! - `eden/src/core/hle/kernel/k_event.{h,cpp}` and `k_readable_event.{h,cpp}`: an event's two
//!   ends, `Signal`, `Clear` and `Reset` ([`event`]).
//! - `eden/src/core/hle/kernel/k_thread.cpp` @ cf654052: `WaitCancel`, `BeginWait`, `OnTimer`;
//!   `k_thread_queue.cpp`: `EndWait` and `CancelWait`; `k_hardware_timer_base.h`: a task fires
//!   once the time reaches its deadline (`task_time > cur_time` waits).
//! - `eden/src/core/hle/kernel/k_address_arbiter.{h,cpp}` ([`arbiter`]),
//!   `k_condition_variable.{h,cpp}` ([`condvar`], and `WaitForAddress`/`SignalToAddress` in
//!   [`lock`]), and the waiter lists and `RestorePriority` of `k_thread.{h,cpp}` ([`lock`]);
//!   `k_thread.h`'s `ConditionVariableComparator` ([`tree`]). User memory is reached as
//!   [`words`] says, with the guest's access checked as Atmosphère checks it (facts only).
//!
//! # Every kind of wait goes one way
//!
//! A `WaitForAddress`, a `WaitProcessWideKeyAtomic` and an `ArbitrateLock` wait the way a
//! `WaitSynchronization` does: a [`Wait`] record ([`Wait::Address`], [`Wait::CondVar`],
//! [`Wait::Lock`]) in the [`SyncTable`], the registers saved as the thread's context, a timer in
//! the same queue (a lock wait has none), and one place a wait ends, which leaves every list,
//! tree and lock the thread is in, writes the result with the call's own register convention
//! and queues a [`Wake`] naming the call. The address-arbiter and condition-variable trees and
//! the locks are scheduler rows too ([`SyncTable::arbiter`], [`SyncTable::condition_variables`],
//! [`SyncTable::locks`]), under the scheduler's lock like everything else here.
//!
//! # Slice 5: intentional divergences from Eden
//!
//! Each is pinned by the named test.
//! - **User memory is checked** (Atmosphère's `ReadFromUser`, `WriteToUser` and
//!   `CanAccessAtomic`; Eden checks nothing and reads unmapped memory as zero): a word the guest
//!   may not read is [`KernelError::InvalidCurrentMemory`](crate::result::KernelError) where
//!   each call reads it, and the atomic updates need a word it may read and write. Tests:
//!   `sync/arbiter/tests.rs`,
//!   `a_word_the_guest_may_not_access_is_invalid_current_memory_before_the_comparison`;
//!   `sync/condvar/tests.rs`,
//!   `a_lock_word_the_guest_may_not_access_atomically_ends_the_wait_with_invalid_current_memory`;
//!   `sync/lock/tests.rs`,
//!   `unlock_of_an_unwritable_word_fails_and_ends_the_next_owners_wait_with_that_result`;
//!   `svc/arbitration/tests.rs`, `a_read_only_word_reads_but_takes_no_write_or_atomic_update`
//!   and `memory_past_the_data_takes_nothing_from_its_first_word`.
//! - **A waiter that terminates leaves at once,** as slice 4's do (below): it leaves the tree
//!   or the lock it waits on, and an owner it lent its priority to gives it back, with no result
//!   written and nothing reported. Eden cancels the wait with `ResultTerminationRequested`.
//!   Tests: `sync/lock/tests.rs`, `a_terminated_lock_waiter_leaves_the_lock_and_releases_the_owner`;
//!   `sync/arbiter/tests.rs`, `a_terminated_address_waiter_leaves_the_tree`.
//! - **A priority change that no core's call makes puts the thread at the back of its new
//!   level.** A timer that ends a condition-variable waiter's wait on a lock owner, and a
//!   terminating waiter, change the owner's priority with no calling core, so no thread counts
//!   as running (`KScheduler::OnThreadPriorityChanged`'s `thread == GetCurrentThreadPointer()`).
//!   Eden asks the core that handles the timer, whose current thread may be the owner, which
//!   then keeps the front of its level. Test: `sync/condvar/tests.rs`,
//!   `a_timer_restoring_an_owners_priority_puts_it_behind_its_equals`.
//! - **`SignalAndModifyByWaitingCountIfEqual` with `count <= 0` and threads waiting writes
//!   `value - 1`,** as Atmosphère does (`libmesosphere/source/kern_k_address_arbiter.cpp`
//!   @ 4e3f3703, fact only), not Eden's `value - 2` (Eden, `eden-nx2` and `eden-pr4588` agree
//!   on `- 2`). The guest sees the word: it is the semaphore's count. Atmosphère is followed as
//!   it is for the user-memory checks above; a hardware trace would settle it
//!   ([`arbiter::modified_value`]). Tests: `sync/arbiter/tests.rs`,
//!   `the_modified_value_is_edens_except_minus_one_when_count_up_to_zero_wakes_waiters` and
//!   `signal_and_modify_by_waiting_count_writes_the_new_value_and_wakes`.
//!
//! # Slice 5: matching Eden where it is easy to miss
//!
//! - Threads of one priority wait in the order they came, in the trees and on a lock (the
//!   comparators return 1 for a tie). Tests: `sync/lock/tests.rs`,
//!   `waiters_of_one_priority_get_the_lock_in_the_order_they_came`; `sync/arbiter/tests.rs`,
//!   `waiters_of_one_priority_are_woken_in_the_order_they_came`.
//! - A thread's held locks are listed most recently added first (`AddHeldLock`'s
//!   `push_front`), whether it began holding the lock when a thread first waited on it or got
//!   it handed on by an unlock, and a destroyed owner ends its waiters in that order. Tests:
//!   `sync/lock/tests.rs`, `a_destroyed_owner_ends_the_waiters_of_its_most_recently_added_lock_first`
//!   and `a_lock_handed_on_goes_to_the_front_of_the_new_owners_list`.
//! - `SignalAndModifyByWaitingCountIfEqual` with `count > 0` counts waiters as Eden's loop does,
//!   which writes `value - 1` (all of them woken) only when no more than `count` wait.
//!   Atmosphère's loop pre-increments where Eden's post-increments; they stop at different
//!   counts but agree on that condition. Test: `sync/arbiter/tests.rs`,
//!   `the_modified_value_is_edens_except_minus_one_when_count_up_to_zero_wakes_waiters`.
//! - `DecrementAndWaitIfLessThan` decrements before the zero-timeout check, and a
//!   `WaitProcessWideKeyAtomic` with a zero timeout lets go of the lock first. Tests:
//!   `sync/arbiter/tests.rs`, `a_zero_timeout_times_out_after_the_comparison_and_keeps_the_decrement`;
//!   `sync/condvar/tests.rs`,
//!   `a_zero_timeout_lets_go_of_the_lock_and_flags_the_key_before_it_times_out`.
//! - A condition-variable waiter that a signal hands to a lock owner keeps its timer, and is
//!   still in `WaitProcessWideKeyAtomic` when it gets the lock. Test: `sync/condvar/tests.rs`,
//!   `a_timer_ends_the_wait_in_the_tree_and_while_it_waits_on_the_owner`.
//! - Only a `WaitSynchronization` can be cancelled; `CancelSynchronization` on a thread in any
//!   other wait leaves a pending cancel (Eden's `m_cancellable`). Tests:
//!   `cancelling_a_lock_waiter_only_leaves_a_pending_cancel` and its arbiter and
//!   condition-variable twins.
//!
//! # Slice 5: not ported, and guesses
//!
//! - `IsTerminationRequested` is not checked before these waits either (below).
//! - The read-modify-write of user words is not atomic against guest code on another core
//!   ([`words`] says why and where that would change).
//! - Kernel light locks (`KLightLock`, `KLightConditionVariable`) are not ported, so nothing
//!   makes a kernel waiter outside this module's tests; [`lock`] counts them already.
//!
//! # Where synchronisation objects sit in the lock order
//!
//! Every row this module keeps is in the [`Scheduler`](crate::sched::Scheduler) table: an
//! event's state is its row in the object table, and the wait lists, wait records, timer queue
//! and pending cancels are the scheduler's [`SyncTable`]. So synchronisation objects are under
//! the scheduler's lock, the last of the dispatcher's three (handle table, address space,
//! scheduler), as Horizon keeps them under its scheduler lock (`KScopedSchedulerLock` in
//! `KReadableEvent::Signal`, `KSynchronizationObject::NotifyAvailable`). A call that resolves
//! handles to objects takes the handle table first; `WaitSynchronization` also reads its handle
//! array through the address space, second. Nothing here takes a lock.
//!
//! # Waiting and waking
//!
//! A thread that waits records what it waits on ([`Wait::Sync`]) in the [`SyncTable`], and its
//! registers are saved as its context. It holds a reference to each object while it waits
//! (kernel spec, "Reference counts are explicit (handles and in-flight waits hold them)"), so
//! closing the last handle to an object never destroys it under a waiter. The waker (a signal,
//! a cancel, an expired timer) ends the wait in one place, `end_wait`: the thread leaves every
//! wait list, its timer entry goes, the result and index are written into its saved `x0` and
//! `x1` with `WaitSynchronization`'s register convention, the thread is made runnable through
//! [`Scheduler::set_state`](crate::sched::Scheduler::set_state), its references are released,
//! and a [`Wake`] is queued for the dispatcher to record and report.
//!
//! Timeouts are entries in the timer queue, a deadline on the environment's clock
//! ([`HostNanos`](starquake_trace::HostNanos), the unit Eden's hardware timer counts in) and the
//! wait's [`WaitToken`]. An entry is removed when its wait ends some other way, as Eden's
//! `KThreadQueue::EndWait` cancels the timer task; an entry that was already taken out of the
//! queue when the wait ended is ignored when it fires, because the thread's current wait (if
//! any) has a different token.
//!
//! # Intentional divergences from Eden
//!
//! Each is pinned by the named test in `sync/tests.rs` or `dispatch/sync/tests.rs`.
//! - **No resource limit.** `CreateEvent` reserves `LimitableResource::EventCountMax` first and
//!   fails with `ResultLimitReached` when it can't. Resource limits are a later slice (kernel
//!   spec, "Port plan"); PROVISIONAL. Test: `creating_many_events_never_reaches_a_limit`.
//! - **An event's two ends are two rows that don't reference each other.** Eden's `KEvent`
//!   holds its `KReadableEvent` inline and the readable end holds a reference to its parent;
//!   when the readable end is destroyed the parent notes it (`m_readable_event_destroyed`) and
//!   `Signal` and `Clear` then succeed doing nothing. Here the writable end keeps the readable
//!   end's id, and an id whose row is gone is the same note: signalling and clearing succeed
//!   doing nothing. What a guest sees is the same. Test:
//!   `signalling_after_the_readable_end_is_closed_succeeds_and_does_nothing`.
//! - **Events a service makes have no writable end.** Eden's services hold a `KEvent` and give
//!   the guest its readable end. Here a service holds the readable end's id (and its creator's
//!   reference, for as long as the service lives) and signals it through that id
//!   ([`Scheduler::signal_object`]): the writable end of a service's event is something only the
//!   service can reach, so it is not made an object. Test:
//!   `a_service_event_is_a_readable_end_the_service_signals`.
//! - **Processes are never signalled.** Eden signals a process when its state changes
//!   (`m_is_signaled`); here process state changes notify no one. PROVISIONAL. A thread is
//!   signalled when its state is `Terminated`, and its waiters wake as it terminates
//!   ([`Scheduler::terminate_thread`](crate::sched::Scheduler::terminate_thread)), as Eden's
//!   `KThread::FinishTermination` sets `m_signaled` and calls `NotifyAvailable`. Waiting on
//!   either is allowed, as in Eden. Tests:
//!   `a_thread_object_is_signalled_once_terminated_and_a_process_never`, and in
//!   `sched/scheduler/lifecycle/tests.rs`,
//!   `a_thread_that_terminates_wakes_its_waiters_and_the_last_wait_reference_destroys_it`.
//! - **A thread that terminates while it waits leaves its wait at once,** with no result
//!   written and nothing reported (`Scheduler::forget_waits`),
//!   and a pending cancel goes with it. Eden cancels the wait with
//!   `ResultTerminationRequested` and the thread runs on to its exit; termination here ends
//!   the thread in the call (`sched/scheduler/lifecycle.rs`). Test (in
//!   `sched/scheduler/lifecycle/tests.rs`):
//!   `a_thread_that_terminates_while_waiting_leaves_every_wait_list`.
//! - **References are released when the wait ends,** not when the thread next runs (Eden's
//!   `SCOPE_EXIT` in `WaitSynchronization` closes them after `Wait` returns). Nothing a guest
//!   sees depends on when an object it no longer has a handle to is destroyed. Test:
//!   `closing_the_last_handle_while_waiting_keeps_the_object_until_the_wait_ends`.
//!
//! # Matching Eden where it is easy to miss
//!
//! - **Timer entries with the same deadline expire in the order their waits began,** as in
//!   Eden. `KTimerTask::Compare` (`eden/src/core/hle/kernel/k_timer_task.h`) returns -1 only
//!   when the left task's time is earlier and 1 otherwise, never 0, so a tie compares greater;
//!   `RB_INSERT` (`eden/src/common/tree.h`) sends `comp > 0` to the right, so a task registered
//!   later sits right of an earlier one with the same time; and `DoInterruptTaskImpl`
//!   (`k_hardware_timer_base.h`) takes the leftmost task first. A wait registers its timer task
//!   as it begins, before the scheduler lock is released (`~KScopedSchedulerLockAndSleep` in
//!   `k_scoped_scheduler_lock_and_sleep.h`, around `KSynchronizationObject::Wait`), so tasks are
//!   registered in the order waits began. Here the queue is ordered by deadline, then by
//!   [`WaitToken`], which grows with every wait begun. Test:
//!   `timers_with_the_same_deadline_expire_in_the_order_their_waits_began`.
//!
//! # Not ported yet
//!
//! - **`IsTerminationRequested`** is not checked before a thread waits: kernel slice 8 ends a
//!   thread in the call that asks for its termination (`sched/scheduler/lifecycle.rs`), so no
//!   thread that was asked to terminate runs again to wait. The check (and
//!   `ResultTerminationRequested`) comes with termination that waits for other cores (kernel
//!   slice 10). PROVISIONAL.
//! - **`ResultSessionClosed` is not mapped to success** (`R_SUCCEED_IF(res ==
//!   ResultSessionClosed)` in `svc_synchronization.cpp`): no object here ends a wait with it.
//!   Only server sessions do (`KServerSession`), and they are deferred (kernel spec, "IPC");
//!   they bring the mapping with them.

pub mod arbiter;
pub mod condvar;
pub mod event;
pub mod lock;
pub mod tree;
pub mod wait;
pub mod words;

#[cfg(test)]
pub(crate) mod testing;
#[cfg(test)]
mod tests;

pub use arbiter::{ArbitrationType, SignalType, modified_value};
pub use condvar::{HAS_NO_WAITER_FLAG, HAS_WAITER_FLAG};
pub use event::{Event, ReadableEvent};
pub use lock::{AddressKey, Arbitrated, HANDLE_WAIT_MASK, HeldLock, LockTable, NO_OWNER};
pub use tree::{ProcessKey, WaitTree};
pub use wait::{
    ARGUMENT_HANDLE_COUNT_MAX, SyncTable, TIMEOUT_SLACK_NS, Timeout, TimerEntry, Wait, WaitBegun,
    WaitCall, WaitOrigin, WaitToken, Wake, is_synchronization_object,
};
pub use words::{GuestWords, UserWords, WORD_SIZE};
