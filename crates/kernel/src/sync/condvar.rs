//! Condition variables: `WaitProcessWideKeyAtomic` and `SignalProcessWideKey`.
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_condition_variable.{h,cpp}` @ cf654052
//! (GPL-3.0-or-later): `KConditionVariable::Wait`
//! ([`Scheduler::wait_condition_variable`]), `Signal` and `SignalImpl`
//! ([`Scheduler::signal_condition_variable`]). The flags written to the condition-variable key
//! are Eden's `HasWaiterFlag = 1` and `HasNoWaiterFlag = 0`, local constants of those two
//! functions.
//!
//! A waiting thread is in its process's condition-variable tree until a signal takes it out;
//! then it takes the lock it gave up, or, if another thread holds the lock, waits on that owner
//! as an `ArbitrateLock` does, with its timer still running. Either way it is still in its
//! `WaitProcessWideKeyAtomic` ([`Wait::CondVar`]), and that call returns when it has the lock.

use super::lock::{AddressKey, Arbitrated, HANDLE_WAIT_MASK, NO_OWNER};
use super::tree::ProcessKey;
use super::wait::{NO_INDEX, Timeout, Tree, Wait, WaitCall};
use super::words::{UserWords, update_lock};
use crate::handle::{Handle, HandleTable};
use crate::id::ThreadId;
use crate::result::{CallResult, KernelError, ResultCode, SvcError};
use crate::sched::{CoreId, SchedError, Scheduler};

/// What `WaitProcessWideKeyAtomic` writes to the condition-variable key: a thread waits on it
/// (Eden's `HasWaiterFlag`, a local constant of `KConditionVariable::Wait`).
pub const HAS_WAITER_FLAG: u32 = 1;

/// What `SignalProcessWideKey` writes to the condition-variable key when no thread is left
/// waiting on it (Eden's `HasNoWaiterFlag`, a local constant of `KConditionVariable::Signal`).
pub const HAS_NO_WAITER_FLAG: u32 = 0;

impl Scheduler {
    /// `WaitProcessWideKeyAtomic` for the thread `call.core` runs (`KConditionVariable::Wait`),
    /// after the system call's checks and with `cv_key` aligned down to a word. In the
    /// reference's order:
    /// - The thread lets go of the lock at `address`: its most urgent waiter is the next owner
    ///   and its wait ends with success ([`Scheduler::remove_waiter_by_key`]).
    /// - [`HAS_WAITER_FLAG`] is written to `cv_key`; a failed write is ignored.
    /// - The lock word becomes the next owner's tag, with [`HANDLE_WAIT_MASK`] if threads still
    ///   wait, or zero: [`KernelError::InvalidCurrentMemory`] if the guest may not write it.
    /// - A zero timeout: [`KernelError::TimedOut`], with the lock let go.
    /// - The thread waits in the process's condition-variable tree under `cv_key`, at its
    ///   priority, with the lock's address and `tag` as its address key.
    pub fn wait_condition_variable(
        &mut self,
        words: &dyn UserWords,
        call: WaitCall<'_>,
        address: u64,
        cv_key: u64,
        tag: u32,
        timeout: Timeout,
    ) -> CallResult<Arbitrated> {
        let thread = self.calling_thread(call.core)?;
        self.ensure_not_waiting(thread)?;
        let caller = Some(call.core);
        let (next, has_waiters) =
            self.remove_waiter_by_key(caller, thread, AddressKey::user(address))?;
        let mut next_value = NO_OWNER;
        if let Some(next) = next {
            let (_, next_tag) = self
                .sync_table()
                .locks
                .address_key(next)
                .ok_or(SchedError::Corrupt)?;
            next_value = if has_waiters {
                next_tag | HANDLE_WAIT_MASK
            } else {
                next_tag
            };
            self.end_wait_from(caller, next, ResultCode::SUCCESS, NO_INDEX)?;
        }
        // Eden ignores whether this write succeeds.
        words.write(cv_key, HAS_WAITER_FLAG);
        if !words.write(address, next_value) {
            return Err(KernelError::InvalidCurrentMemory.into());
        }
        let deadline = timeout.deadline()?;
        let key = ProcessKey {
            process: self.thread(thread)?.process(),
            address: cv_key,
        };
        self.sync_table_mut()
            .locks
            .set_address_key(thread, AddressKey::user(address), tag);
        self.block(thread, call.regs, caller, |token| Wait::CondVar {
            key,
            lock: address,
            token,
            deadline,
            origin: call.origin,
        })?;
        Ok(Arbitrated::Waiting(thread))
    }

    /// `SignalProcessWideKey` for the thread `core` runs (`KConditionVariable::Signal`), with
    /// `cv_key` aligned down to a word: the first `count` threads waiting on `cv_key` in the
    /// process's tree (all of them for `count <= 0`), most urgent first, each leave the tree and
    /// try to take their lock (`SignalImpl`, below). Then, if no thread is left waiting on
    /// `cv_key`, [`HAS_NO_WAITER_FLAG`] is written to it; a failed write is ignored.
    ///
    /// Taking the lock (`KConditionVariable::SignalImpl`): the lock word is updated atomically, to the thread's tag if it was
    /// zero and with [`HANDLE_WAIT_MASK`] set otherwise. A word the guest may not access
    /// atomically ends the thread's wait with [`KernelError::InvalidCurrentMemory`]; a zero word
    /// (no owner) ends it with success. Otherwise the word names the owner's handle (the mask
    /// cleared), which is looked up in `handles`, the signalling process's table: a thread
    /// there gets this one as a waiter (inheriting its priority if it is more urgent); anything
    /// else ends the wait with [`KernelError::InvalidState`].
    pub fn signal_condition_variable(
        &mut self,
        handles: &HandleTable,
        words: &dyn UserWords,
        core: CoreId,
        cv_key: u64,
        count: i32,
    ) -> CallResult<()> {
        let thread = self.calling_thread(core)?;
        let key = ProcessKey {
            process: self.thread(thread)?.process(),
            address: cv_key,
        };
        let caller = Some(core);
        let mut woken = 0_i32;
        let mut result = Ok(());
        while count <= 0 || woken < count {
            let Some(waiter) = self
                .sync_table_mut()
                .tree_mut(Tree::CondVar)
                .take_front(key)
            else {
                break;
            };
            let signalled = self.take_lock_for(handles, words, caller, waiter);
            if result.is_ok() {
                result = signalled;
            }
            woken = woken.saturating_add(1);
        }
        if self.sync_table().condvars.waiters(key).is_empty() {
            // Eden ignores whether this write succeeds.
            words.write(cv_key, HAS_NO_WAITER_FLAG);
        }
        result
    }

    /// `KConditionVariable::SignalImpl` for `waiter`, which a signal took out of the tree.
    fn take_lock_for(
        &mut self,
        handles: &HandleTable,
        words: &dyn UserWords,
        caller: Option<CoreId>,
        waiter: ThreadId,
    ) -> CallResult<()> {
        let (lock, tag) = self
            .sync_table()
            .locks
            .address_key(waiter)
            .ok_or(SchedError::Corrupt)?;
        let Some(previous) = update_lock(words, lock.address, tag, HANDLE_WAIT_MASK) else {
            return self.end_wait_from(
                caller,
                waiter,
                KernelError::InvalidCurrentMemory.code(),
                NO_INDEX,
            );
        };
        if previous == NO_OWNER {
            return self.end_wait_from(caller, waiter, ResultCode::SUCCESS, NO_INDEX);
        }
        let owner_handle = Handle::from_raw(previous & !HANDLE_WAIT_MASK);
        match self.thread_object(handles, owner_handle) {
            Ok((_, owner)) => self.add_waiter(caller, owner, waiter).map_err(Into::into),
            Err(SvcError::Kernel(KernelError::InvalidHandle)) => {
                self.end_wait_from(caller, waiter, KernelError::InvalidState.code(), NO_INDEX)
            }
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests;
