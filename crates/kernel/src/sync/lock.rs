//! Lock arbitration with priority inheritance: `ArbitrateLock` and `ArbitrateUnlock`, the locks
//! each thread holds and the threads waiting on them, and the walk that passes a waiter's
//! priority up a chain of owners and takes it back.
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), paths relative to the reference folder:
//! - `eden/src/core/hle/kernel/k_condition_variable.cpp`: `KConditionVariable::WaitForAddress`
//!   ([`Scheduler::arbitrate_lock`]) and `SignalToAddress` ([`Scheduler::arbitrate_unlock`]).
//! - `eden/src/core/hle/kernel/k_thread.cpp`: `AddWaiterImpl`, `RemoveWaiterImpl`, `AddWaiter`,
//!   `RemoveWaiter`, `RemoveWaiterByKey`, `RestorePriority`, `AddHeldLock`, `FindHeldLock`, and
//!   the part of `Finalize` that cancels the waiters of the locks a destroyed thread still holds
//!   ([`Scheduler::release_held_locks`]).
//! - `eden/src/core/hle/kernel/k_thread.h`: `LockWithPriorityInheritanceInfo` (a held lock: its
//!   address key and its waiters by priority, `LockWithPriorityInheritanceComparator` putting a
//!   tie after the threads already there) and `m_num_kernel_waiters`.
//! - `eden/src/core/hle/kernel/svc_common.h`: `HandleWaitMask = 1u << 30`.
//!
//! # How the rows map Eden's
//!
//! A thread's `m_held_lock_info_list`, `m_waiting_lock_info`, `m_address_key`,
//! `m_address_key_value`, `m_is_kernel_address_key` and `m_num_kernel_waiters` are rows of the
//! [`LockTable`], keyed by thread id. A held lock ([`HeldLock`]) has no owner field: it is in its
//! owner's list, and each waiter's row names the owner instead (`GetLockOwner`), so moving a
//! lock to a new owner (`RemoveWaiterByKey`) rewrites its waiters' rows.
//!
//! # Kernel waiters
//!
//! A kernel address key is what Horizon's own light locks (`KLightLock`) wait with; a thread
//! with kernel waiters is not suspended (`TrySuspend`, [`Scheduler::request_suspend`]). Nothing
//! in starquake takes a kernel light lock yet, so only this module's tests make kernel waiters;
//! the counting is ported so that one can be added without touching it.

use std::collections::BTreeMap;

use crate::handle::{Handle, HandleTable, ObjectRefs};
use crate::id::{ObjectId, ThreadId};
use crate::object::Object;
use crate::result::{CallResult, KernelError, ResultCode, SvcError};
use crate::sched::{CoreId, PRIORITY_COUNT, SchedError, Scheduler};

use super::wait::{NO_INDEX, Wait, WaitCall};
use super::words::UserWords;

/// The bit a lock word carries when threads wait on it: Eden's `Svc::HandleWaitMask`
/// (`eden/src/core/hle/kernel/svc_common.h` @ cf654052, `1u << 30`).
pub const HANDLE_WAIT_MASK: u32 = 1 << 30;

/// A lock word with no owner: Eden's `Svc::InvalidHandle` (`svc_common.h`, `Handle(0)`), the
/// value `SignalImpl` compares the previous tag with.
pub const NO_OWNER: u32 = 0;

/// What a lock is keyed by (`m_address_key` and `m_is_kernel_address_key`): its address, and
/// whether a kernel light lock (rather than a user lock) waits with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AddressKey {
    /// The lock's address.
    pub address: u64,
    /// Whether it is a kernel address key (`SetKernelAddressKey`).
    pub kernel: bool,
}

impl AddressKey {
    /// A user lock at `address` (`SetUserAddressKey`).
    #[must_use]
    pub const fn user(address: u64) -> Self {
        Self {
            address,
            kernel: false,
        }
    }
}

/// A lock a thread holds that other threads wait on (`LockWithPriorityInheritanceInfo`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldLock {
    key: AddressKey,
    /// Most urgent first; within a priority, in the order they were added.
    waiters: Vec<ThreadId>,
}

impl HeldLock {
    /// The lock's key.
    #[must_use]
    pub const fn key(&self) -> AddressKey {
        self.key
    }

    /// The threads waiting on it, most urgent first (`GetHighestPriorityWaiter` is the first).
    #[must_use]
    pub fn waiters(&self) -> &[ThreadId] {
        &self.waiters
    }
}

/// The locks threads hold and wait on. See the module docs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LockTable {
    /// Each owner's held locks, most recently added first (`push_front`). An owner with none
    /// has no entry.
    held: BTreeMap<ThreadId, Vec<HeldLock>>,
    /// Each waiter's lock owner (`GetLockOwner`).
    owners: BTreeMap<ThreadId, ThreadId>,
    /// Each thread's address key and the tag it writes when it gets the lock.
    keys: BTreeMap<ThreadId, (AddressKey, u32)>,
    /// Each thread's count of kernel waiters, when not zero.
    kernel_waiters: BTreeMap<ThreadId, u32>,
}

impl LockTable {
    /// The locks `owner` holds that threads wait on, most recently added first.
    #[must_use]
    pub fn held(&self, owner: ThreadId) -> &[HeldLock] {
        self.held.get(&owner).map_or(&[], Vec::as_slice)
    }

    /// The owner of the lock `waiter` waits on, if it waits on one (`GetLockOwner`).
    #[must_use]
    pub fn owner_of(&self, waiter: ThreadId) -> Option<ThreadId> {
        self.owners.get(&waiter).copied()
    }

    /// The lock `thread` waits for (or will, once a condition variable is signalled) and the
    /// tag it writes into the lock word when it gets it (`GetAddressKeyValue`).
    #[must_use]
    pub fn address_key(&self, thread: ThreadId) -> Option<(AddressKey, u32)> {
        self.keys.get(&thread).copied()
    }

    /// How many threads wait on `thread`'s locks with a kernel address key
    /// (`GetNumKernelWaiters`).
    #[must_use]
    pub fn kernel_waiters(&self, thread: ThreadId) -> u32 {
        self.kernel_waiters.get(&thread).copied().unwrap_or(0)
    }

    /// How many threads hold locks that are waited on: the walk of
    /// [`Scheduler::restore_priority_chain`] moves only between them.
    fn owner_count(&self) -> usize {
        self.held.len()
    }

    /// Sets `thread`'s address key and tag (`SetUserAddressKey`, `SetKernelAddressKey`,
    /// `SetConditionVariable`).
    pub(crate) fn set_address_key(&mut self, thread: ThreadId, key: AddressKey, tag: u32) {
        self.keys.insert(thread, (key, tag));
    }

    /// Forgets `thread`'s address key: its wait is over.
    pub(crate) fn forget_address_key(&mut self, thread: ThreadId) {
        self.keys.remove(&thread);
    }

    fn change_kernel_waiters(
        &mut self,
        thread: ThreadId,
        add: u32,
        remove: u32,
    ) -> Result<(), SchedError> {
        let count = self
            .kernel_waiters(thread)
            .checked_add(add)
            .and_then(|count| count.checked_sub(remove))
            .ok_or(SchedError::Corrupt)?;
        if count == 0 {
            self.kernel_waiters.remove(&thread);
        } else {
            self.kernel_waiters.insert(thread, count);
        }
        Ok(())
    }

    fn lock_index(&self, owner: ThreadId, key: AddressKey) -> Option<usize> {
        self.held(owner).iter().position(|lock| lock.key == key)
    }
}

/// How a call that may wait began: the call ended at once, successfully, or the thread waits.
/// The calls' failures are their `Err`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arbitrated {
    /// The call succeeded without waiting.
    Done,
    /// The thread waits; the wake writes its result into its saved registers.
    Waiting(ThreadId),
}

impl Scheduler {
    /// Where `thread` goes among `waiters`: after every thread whose priority is not less
    /// urgent (`LockWithPriorityInheritanceComparator` returns 1 for a tie).
    fn lock_place(&self, waiters: &[ThreadId], thread: ThreadId) -> Result<usize, SchedError> {
        let priority = self.thread(thread)?.priority();
        for (index, waiter) in waiters.iter().enumerate() {
            if self.thread(*waiter)?.priority() > priority {
                return Ok(index);
            }
        }
        Ok(waiters.len())
    }

    /// `KThread::AddWaiterImpl`: `waiter` joins the lock `owner` holds under its address key,
    /// which `owner` starts holding (at the front of its list) if it did not, at its place by
    /// priority. A kernel key counts a kernel waiter and asks for a scheduler update.
    pub(crate) fn add_waiter_impl(
        &mut self,
        owner: ThreadId,
        waiter: ThreadId,
    ) -> Result<(), SchedError> {
        let (key, _) = self
            .sync_table()
            .locks
            .address_key(waiter)
            .ok_or(SchedError::Corrupt)?;
        if key.kernel {
            self.sync_table_mut()
                .locks
                .change_kernel_waiters(owner, 1, 0)?;
            self.note_update_needed();
        }
        let index = self.sync_table().locks.lock_index(owner, key);
        let place = match index {
            Some(index) => {
                let lock = self
                    .sync_table()
                    .locks
                    .held(owner)
                    .get(index)
                    .ok_or(SchedError::Corrupt)?;
                self.lock_place(&lock.waiters, waiter)?
            }
            None => 0,
        };
        let locks = &mut self.sync_table_mut().locks;
        let held = locks.held.entry(owner).or_default();
        match index.and_then(|index| held.get_mut(index)) {
            Some(lock) => lock.waiters.insert(place, waiter),
            None => held.insert(
                0,
                HeldLock {
                    key,
                    waiters: vec![waiter],
                },
            ),
        }
        locks.owners.insert(waiter, owner);
        Ok(())
    }

    /// `KThread::RemoveWaiterImpl`: `waiter` leaves the lock it waits on, which `owner` holds;
    /// a lock left with no waiters is no longer held. A kernel key uncounts a kernel waiter and
    /// asks for a scheduler update.
    pub(crate) fn remove_waiter_impl(
        &mut self,
        owner: ThreadId,
        waiter: ThreadId,
    ) -> Result<(), SchedError> {
        let (key, _) = self
            .sync_table()
            .locks
            .address_key(waiter)
            .ok_or(SchedError::Corrupt)?;
        if key.kernel {
            self.sync_table_mut()
                .locks
                .change_kernel_waiters(owner, 0, 1)?;
            self.note_update_needed();
        }
        let locks = &mut self.sync_table_mut().locks;
        if locks.owners.get(&waiter) != Some(&owner) {
            return Err(SchedError::Corrupt);
        }
        let index = locks.lock_index(owner, key).ok_or(SchedError::Corrupt)?;
        let held = locks.held.get_mut(&owner).ok_or(SchedError::Corrupt)?;
        let lock = held.get_mut(index).ok_or(SchedError::Corrupt)?;
        lock.waiters.retain(|listed| *listed != waiter);
        if lock.waiters.is_empty() {
            held.remove(index);
            if held.is_empty() {
                locks.held.remove(&owner);
            }
        }
        locks.owners.remove(&waiter);
        Ok(())
    }

    /// `KThread::AddWaiter`: [`Scheduler::add_waiter_impl`], then, if `waiter` is more urgent
    /// than `owner`, `owner` inherits its priority ([`Scheduler::restore_priority_chain`]).
    /// `caller` is the core making the call, if a core is.
    pub(crate) fn add_waiter(
        &mut self,
        caller: Option<CoreId>,
        owner: ThreadId,
        waiter: ThreadId,
    ) -> Result<(), SchedError> {
        self.add_waiter_impl(owner, waiter)?;
        if self.thread(waiter)?.priority() < self.thread(owner)?.priority() {
            self.restore_priority_chain(caller, owner)?;
        }
        Ok(())
    }

    /// `KThread::RemoveWaiter`: [`Scheduler::remove_waiter_impl`], then, if `owner` runs at
    /// `waiter`'s priority and that is above its base priority (it may have inherited it from
    /// `waiter`), its priority is worked out again.
    pub(crate) fn remove_waiter(
        &mut self,
        caller: Option<CoreId>,
        owner: ThreadId,
        waiter: ThreadId,
    ) -> Result<(), SchedError> {
        self.remove_waiter_impl(owner, waiter)?;
        let owner_row = self.thread(owner)?;
        if owner_row.priority() == self.thread(waiter)?.priority()
            && owner_row.priority() < owner_row.base_priority()
        {
            self.restore_priority_chain(caller, owner)?;
        }
        Ok(())
    }

    /// `KThread::RemoveWaiterByKey`: `owner` lets go of the lock it holds under `key`. Its most
    /// urgent waiter is the next owner and takes the lock's other waiters with it (as a lock it
    /// holds, at the front of its list); the result is that thread and whether any waiters
    /// remain. A lock `owner` does not hold (no thread waits on it) is `(None, false)`. Kernel
    /// waiters move with the lock. If `owner` runs at the next owner's priority and that is
    /// above its base priority, its priority is worked out again.
    pub(crate) fn remove_waiter_by_key(
        &mut self,
        caller: Option<CoreId>,
        owner: ThreadId,
        key: AddressKey,
    ) -> Result<(Option<ThreadId>, bool), SchedError> {
        let locks = &mut self.sync_table_mut().locks;
        let Some(index) = locks.lock_index(owner, key) else {
            return Ok((None, false));
        };
        let held = locks.held.get_mut(&owner).ok_or(SchedError::Corrupt)?;
        if index >= held.len() {
            return Err(SchedError::Corrupt);
        }
        let lock = held.remove(index);
        if held.is_empty() {
            locks.held.remove(&owner);
        }
        let waiting = u32::try_from(lock.waiters.len()).map_err(|_| SchedError::Corrupt)?;
        if key.kernel {
            locks.change_kernel_waiters(owner, 0, waiting)?;
        }
        let Some((&next, rest)) = lock.waiters.split_first() else {
            return Err(SchedError::Corrupt);
        };
        locks.owners.remove(&next);
        let has_waiters = !rest.is_empty();
        if has_waiters {
            for waiter in rest {
                locks.owners.insert(*waiter, next);
            }
            if key.kernel {
                locks.change_kernel_waiters(next, waiting.saturating_sub(1), 0)?;
            }
            locks.held.entry(next).or_default().insert(
                0,
                HeldLock {
                    key,
                    waiters: rest.to_vec(),
                },
            );
        }
        if key.kernel {
            self.note_update_needed();
        }
        let owner_row = self.thread(owner)?;
        if owner_row.priority() == self.thread(next)?.priority()
            && owner_row.priority() < owner_row.base_priority()
        {
            // The next owner needs no restore: it was the lock's most urgent waiter.
            self.restore_priority_chain(caller, owner)?;
        }
        Ok((Some(next), has_waiters))
    }

    /// `KThread::RestorePriority`: `thread`'s priority becomes the most urgent of its base
    /// priority and the first waiter of each lock it holds; if that changed it, the thread
    /// takes its new place in the lock it waits on and in the tree it is in, the scheduler's
    /// queues are told, and the walk goes on to the owner of the lock it waits on.
    ///
    /// Each step moves every priority in one direction (more or less urgent), so the walk ends
    /// even around a cycle of threads waiting on each other's locks. It is also bounded by the
    /// structure it walks: more than [`PRIORITY_COUNT`] steps per lock owner, plus one, is
    /// damaged bookkeeping ([`SchedError::Corrupt`]).
    pub(crate) fn restore_priority_chain(
        &mut self,
        caller: Option<CoreId>,
        start: ThreadId,
    ) -> Result<(), SchedError> {
        let bound = self
            .sync_table()
            .locks
            .owner_count()
            .saturating_add(1)
            .saturating_mul(PRIORITY_COUNT);
        let mut steps = 0_usize;
        let mut next = Some(start);
        while let Some(thread) = next {
            steps = steps.saturating_add(1);
            if steps > bound {
                return Err(SchedError::Corrupt);
            }
            let row = self.thread(thread)?;
            let current = row.priority();
            let mut new_priority = row.base_priority();
            for lock in self.sync_table().locks.held(thread) {
                let first = lock.waiters.first().ok_or(SchedError::Corrupt)?;
                new_priority = new_priority.min(self.thread(*first)?.priority());
            }
            if new_priority == current {
                return Ok(());
            }
            let owner = self.sync_table().locks.owner_of(thread);
            if let Some(owner) = owner {
                self.remove_waiter_impl(owner, thread)?;
            }
            let tree = self
                .sync_table()
                .wait(thread)
                .and_then(Wait::tree)
                .filter(|(which, key)| self.sync_table().tree(*which).contains(*key, thread));
            if let Some((which, key)) = tree {
                self.sync_table_mut().tree_mut(which).remove(key, thread);
            }
            self.set_effective_priority(caller, thread, new_priority)?;
            if let Some((which, key)) = tree {
                let place = self.tree_place(which, key, thread)?;
                self.sync_table_mut()
                    .tree_mut(which)
                    .insert_at(key, place, thread);
            }
            if let Some(owner) = owner {
                self.add_waiter_impl(owner, thread)?;
            }
            next = owner;
        }
        Ok(())
    }

    /// The part of `KThread::Finalize` that lets go of the locks a destroyed thread still
    /// holds: each lock's waiters, most urgent first, leave it and end their waits with
    /// [`KernelError::InvalidState`], lock by lock from the front of its list. Every waiter is
    /// ended even if one fails; the first failure is returned.
    pub(crate) fn release_held_locks(&mut self, thread: ThreadId) -> CallResult<()> {
        let locks = &mut self.sync_table_mut().locks;
        let held = locks.held.remove(&thread).unwrap_or_default();
        locks.kernel_waiters.remove(&thread);
        let mut result = Ok(());
        for lock in held {
            for waiter in lock.waiters {
                self.sync_table_mut().locks.owners.remove(&waiter);
                let ended =
                    self.end_wait_from(None, waiter, KernelError::InvalidState.code(), NO_INDEX);
                if result.is_ok() {
                    result = ended;
                }
            }
        }
        result
    }

    /// `ArbitrateLock` for the thread `call.core` runs (`KConditionVariable::WaitForAddress`),
    /// after the system call's checks of the address. In the reference's order:
    /// - The lock word at `address` is read: [`KernelError::InvalidCurrentMemory`] if the guest
    ///   may not read it.
    /// - Unless it is `owner_handle` with [`HANDLE_WAIT_MASK`] set, the lock changed hands
    ///   already: [`Arbitrated::Done`].
    /// - `owner_handle` must name a thread in `handles` (`GetObjectWithoutPseudoHandle`, so not
    ///   a pseudo-handle): [`KernelError::InvalidHandle`] otherwise.
    /// - The thread waits on the owner with its address key and `tag`, and the owner inherits
    ///   its priority if it is more urgent. The wait holds a reference to the owner's object.
    pub fn arbitrate_lock(
        &mut self,
        handles: &HandleTable,
        words: &dyn UserWords,
        call: WaitCall<'_>,
        owner_handle: Handle,
        address: u64,
        tag: u32,
    ) -> CallResult<Arbitrated> {
        let thread = self.calling_thread(call.core)?;
        let test_tag = words
            .read(address)
            .ok_or(KernelError::InvalidCurrentMemory)?;
        if test_tag != owner_handle.raw() | HANDLE_WAIT_MASK {
            return Ok(Arbitrated::Done);
        }
        let (object, owner) = self.thread_object(handles, owner_handle)?;
        self.ensure_not_waiting(thread)?;
        self.objects_mut()
            .open(object)
            .map_err(|error| SvcError::internal(&error))?;
        let caller = Some(call.core);
        self.sync_table_mut()
            .locks
            .set_address_key(thread, AddressKey::user(address), tag);
        let blocked = self
            .add_waiter(caller, owner, thread)
            .map_err(SvcError::from)
            .and_then(|()| {
                self.block(thread, call.regs, caller, |token| Wait::Lock {
                    lock: address,
                    owner: object,
                    token,
                    origin: call.origin,
                })
            });
        if let Err(error) = blocked {
            // `block` undid itself; a failed `add_waiter` is undone here. Both leave nothing.
            let (_, unlinked) = self.unlink_wait(thread, caller);
            let released = self.release_object(object);
            if let Err(undo) = unlinked.and(released) {
                tracing::error!(%undo, "an ArbitrateLock that failed to wait could not be undone");
            }
            return Err(error);
        }
        Ok(Arbitrated::Waiting(thread))
    }

    /// The thread `handle` names in `handles` (`GetObjectWithoutPseudoHandle<KThread>`), with
    /// its object: [`KernelError::InvalidHandle`] for a pseudo-handle, a handle the table does
    /// not have, or one that names anything but a thread.
    pub(crate) fn thread_object(
        &self,
        handles: &HandleTable,
        handle: Handle,
    ) -> CallResult<(ObjectId, ThreadId)> {
        let object = handles
            .get(handle)
            .map_err(|_| KernelError::InvalidHandle)?;
        match self.objects().get(object)? {
            Object::Thread(thread) => Ok((object, thread)),
            Object::Process(_) | Object::Session | Object::Event(_) | Object::ReadableEvent(_) => {
                Err(KernelError::InvalidHandle.into())
            }
        }
    }

    /// `ArbitrateUnlock` for the thread `core` runs (`KConditionVariable::SignalToAddress`),
    /// after the system call's checks of the address: the thread lets go of the lock at
    /// `address` ([`Scheduler::remove_waiter_by_key`]), and the lock word becomes the next
    /// owner's tag, with [`HANDLE_WAIT_MASK`] if threads still wait, or zero with no next owner.
    /// A word the guest may not write is [`KernelError::InvalidCurrentMemory`], and the next
    /// owner's wait ends with that result too; otherwise it ends with success.
    pub fn arbitrate_unlock(
        &mut self,
        words: &dyn UserWords,
        core: CoreId,
        address: u64,
    ) -> CallResult<()> {
        let owner = self.calling_thread(core)?;
        let caller = Some(core);
        let (next, has_waiters) =
            self.remove_waiter_by_key(caller, owner, AddressKey::user(address))?;
        let next_value = match next {
            Some(next) => {
                let (_, tag) = self
                    .sync_table()
                    .locks
                    .address_key(next)
                    .ok_or(SchedError::Corrupt)?;
                if has_waiters {
                    tag | HANDLE_WAIT_MASK
                } else {
                    tag
                }
            }
            None => NO_OWNER,
        };
        let written = words.write(address, next_value);
        let result = if written {
            ResultCode::SUCCESS
        } else {
            KernelError::InvalidCurrentMemory.code()
        };
        if let Some(next) = next {
            self.end_wait_from(caller, next, result, NO_INDEX)?;
        }
        if written {
            Ok(())
        } else {
            Err(KernelError::InvalidCurrentMemory.into())
        }
    }
}

#[cfg(test)]
mod tests;
