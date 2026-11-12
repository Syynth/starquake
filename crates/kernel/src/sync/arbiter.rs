//! The address arbiter: `WaitForAddress` and `SignalToAddress`, every arbitration and signal
//! type.
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_address_arbiter.{h,cpp}` @ cf654052
//! (GPL-3.0-or-later): `WaitIfLessThan` and `WaitIfEqual` ([`Scheduler::wait_for_address`]),
//! `Signal`, `SignalAndIncrementIfEqual` and `SignalAndModifyByWaitingCountIfEqual`
//! ([`Scheduler::signal_to_address`]). The type values are Eden's `Svc::ArbitrationType` and
//! `Svc::SignalType` (`eden/src/core/hle/kernel/svc_types.h`).
//!
//! # The seam for per-bucket locks
//!
//! The kernel spec ("Tables behind a dispatcher", step 3) wants the arbiter's wait lists
//! sharded by address later, so the hot `WaitForAddress`/`SignalToAddress` pair holds the
//! scheduler lock only for the final wake-up. Callers reach the arbiter through two methods
//! only, [`Scheduler::wait_for_address`] and [`Scheduler::signal_to_address`], and its state is
//! one [`WaitTree`](super::WaitTree) keyed by process and address
//! ([`SyncTable::arbiter`](super::SyncTable::arbiter)), read and changed nowhere else but the
//! one place a wait ends. Sharding moves that tree behind per-bucket locks and changes these two
//! methods' bodies, not their callers (`crate::svc::arbitration`).

use super::lock::Arbitrated;
use super::tree::ProcessKey;
use super::wait::{NO_INDEX, Timeout, Tree, Wait, WaitCall};
use super::words::{UserWords, decrement_if_less_than, update_if_equal};
use crate::id::ThreadId;
use crate::result::{CallResult, KernelError, ResultCode};
use crate::sched::{CoreId, Scheduler};

/// How a `WaitForAddress` compares the word: Eden's `Svc::ArbitrationType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArbitrationType {
    /// Wait while the word is less than the value (`WaitIfLessThan = 0`).
    WaitIfLessThan,
    /// Decrement the word if it is less than the value, and then wait
    /// (`DecrementAndWaitIfLessThan = 1`).
    DecrementAndWaitIfLessThan,
    /// Wait while the word equals the value (`WaitIfEqual = 2`).
    WaitIfEqual,
}

impl ArbitrationType {
    /// The type `raw` encodes, or `None` for a value Eden's `IsValidArbitrationType` rejects.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::WaitIfLessThan),
            1 => Some(Self::DecrementAndWaitIfLessThan),
            2 => Some(Self::WaitIfEqual),
            _ => None,
        }
    }

    /// The type's value.
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::WaitIfLessThan => 0,
            Self::DecrementAndWaitIfLessThan => 1,
            Self::WaitIfEqual => 2,
        }
    }
}

/// What a `SignalToAddress` does to the word first: Eden's `Svc::SignalType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SignalType {
    /// Nothing (`Signal = 0`).
    Signal,
    /// Increment it if it equals the value (`SignalAndIncrementIfEqual = 1`).
    SignalAndIncrementIfEqual,
    /// If it equals the value, change it by how many threads wait
    /// (`SignalAndModifyByWaitingCountIfEqual = 2`).
    SignalAndModifyByWaitingCountIfEqual,
}

impl SignalType {
    /// The type `raw` encodes, or `None` for a value Eden's `IsValidSignalType` rejects.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Signal),
            1 => Some(Self::SignalAndIncrementIfEqual),
            2 => Some(Self::SignalAndModifyByWaitingCountIfEqual),
            _ => None,
        }
    }

    /// The type's value.
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::Signal => 0,
            Self::SignalAndIncrementIfEqual => 1,
            Self::SignalAndModifyByWaitingCountIfEqual => 2,
        }
    }
}

/// The new word `SignalAndModifyByWaitingCountIfEqual` writes for `value` when `waiting`
/// threads wait on the address and `count` are to be woken (`count <= 0`: all of them), as
/// Eden computes it: with no waiter, `value + 1`; when every waiter is woken, `value - 2` for
/// `count <= 0` and `value - 1` otherwise; when some stay, `value`. The arithmetic wraps, as
/// the reference's `s32` arithmetic does on a two's-complement machine.
///
/// Eden counts the waiters after the first with `if (tmp_num_waiters++ >= count) break;`,
/// which stops at `count + 1` of them, and wakes them all only when that count is below
/// `count`: that is, when `waiting <= count`.
#[must_use]
pub fn modified_value(value: i32, waiting: usize, count: i32) -> i32 {
    if waiting == 0 {
        return value.wrapping_add(1);
    }
    let Ok(count) = usize::try_from(count) else {
        return value.wrapping_sub(2);
    };
    if count == 0 {
        return value.wrapping_sub(2);
    }
    if waiting <= count {
        value.wrapping_sub(1)
    } else {
        value
    }
}

impl Scheduler {
    /// The arbiter key of `thread`'s process and `address`.
    fn process_key(&self, thread: ThreadId, address: u64) -> CallResult<ProcessKey> {
        Ok(ProcessKey {
            process: self.thread(thread)?.process(),
            address,
        })
    }

    /// `WaitForAddress` for the thread `call.core` runs, after the system call's checks of its
    /// arguments (`KAddressArbiter::WaitIfLessThan` and `WaitIfEqual`). In the reference's order:
    /// - The word at `address` is read as an `s32`, and for
    ///   [`ArbitrationType::DecrementAndWaitIfLessThan`] decremented in place if it is less than
    ///   `value`, before any other check: [`KernelError::InvalidCurrentMemory`] if the guest may
    ///   not read it, or for the decrement may not access it atomically.
    /// - Unless the word is less than `value` (or for [`ArbitrationType::WaitIfEqual`] equal to
    ///   it): [`KernelError::InvalidState`].
    /// - A zero timeout: [`KernelError::TimedOut`] (the decrement stays done).
    /// - The thread waits in the process's arbiter tree, at its priority, until a
    ///   [`Scheduler::signal_to_address`] on the address (success) or its timer
    ///   ([`KernelError::TimedOut`]).
    pub fn wait_for_address(
        &mut self,
        words: &dyn UserWords,
        call: WaitCall<'_>,
        address: u64,
        kind: ArbitrationType,
        value: i32,
        timeout: Timeout,
    ) -> CallResult<Arbitrated> {
        let thread = self.calling_thread(call.core)?;
        let user_value = match kind {
            ArbitrationType::DecrementAndWaitIfLessThan => {
                decrement_if_less_than(words, address, value)
            }
            ArbitrationType::WaitIfLessThan | ArbitrationType::WaitIfEqual => {
                words.read(address).map(u32::cast_signed)
            }
        }
        .ok_or(KernelError::InvalidCurrentMemory)?;
        let waits = match kind {
            ArbitrationType::WaitIfLessThan | ArbitrationType::DecrementAndWaitIfLessThan => {
                user_value < value
            }
            ArbitrationType::WaitIfEqual => user_value == value,
        };
        if !waits {
            return Err(KernelError::InvalidState.into());
        }
        let deadline = timeout.deadline()?;
        let key = self.process_key(thread, address)?;
        self.block(thread, call.regs, Some(call.core), |token| Wait::Address {
            key,
            token,
            deadline,
            origin: call.origin,
        })?;
        Ok(Arbitrated::Waiting(thread))
    }

    /// `SignalToAddress` for the thread `core` runs, after the system call's checks of its
    /// arguments (`KAddressArbiter::SignalToAddress`):
    /// - [`SignalType::Signal`] wakes the threads waiting on `address`.
    /// - [`SignalType::SignalAndIncrementIfEqual`] first increments the word if it equals
    ///   `value`: [`KernelError::InvalidCurrentMemory`] if the guest may not access it
    ///   atomically, [`KernelError::InvalidState`] (and no wake) if it differs.
    /// - [`SignalType::SignalAndModifyByWaitingCountIfEqual`] first works out the new word
    ///   ([`modified_value`]) and writes it if it equals `value`, atomically when it changes
    ///   and as a plain read when it does not; the same two errors.
    ///
    /// The threads woken are the first `count` waiting on the address in the process's arbiter
    /// tree (every one for `count <= 0`), most urgent first, each with success.
    pub fn signal_to_address(
        &mut self,
        words: &dyn UserWords,
        core: CoreId,
        address: u64,
        kind: SignalType,
        value: i32,
        count: i32,
    ) -> CallResult<()> {
        let thread = self.calling_thread(core)?;
        let key = self.process_key(thread, address)?;
        let user_value = match kind {
            SignalType::Signal => None,
            SignalType::SignalAndIncrementIfEqual => Some(
                update_if_equal(words, address, value, value.wrapping_add(1))
                    .ok_or(KernelError::InvalidCurrentMemory)?,
            ),
            SignalType::SignalAndModifyByWaitingCountIfEqual => {
                let waiting = self.sync_table().arbiter.waiters(key).len();
                let new_value = modified_value(value, waiting, count);
                let read = if new_value == value {
                    words.read(address).map(u32::cast_signed)
                } else {
                    update_if_equal(words, address, value, new_value)
                };
                Some(read.ok_or(KernelError::InvalidCurrentMemory)?)
            }
        };
        if user_value.is_some_and(|user_value| user_value != value) {
            return Err(KernelError::InvalidState.into());
        }
        self.wake_tree(Tree::Arbiter, key, count, Some(core))
    }

    /// Ends the waits of the first `count` threads in `which` under `key` (all of them for
    /// `count <= 0`) with success, taking each out of the tree first. Every thread is woken
    /// even if one cannot be; the first failure is returned.
    fn wake_tree(
        &mut self,
        which: Tree,
        key: ProcessKey,
        count: i32,
        caller: Option<CoreId>,
    ) -> CallResult<()> {
        let mut woken = 0_i32;
        let mut result = Ok(());
        while count <= 0 || woken < count {
            let Some(thread) = self.sync_table_mut().tree_mut(which).take_front(key) else {
                break;
            };
            let ended = self.end_wait_from(caller, thread, ResultCode::SUCCESS, NO_INDEX);
            if result.is_ok() {
                result = ended;
            }
            woken = woken.saturating_add(1);
        }
        result
    }
}

#[cfg(test)]
mod tests;
