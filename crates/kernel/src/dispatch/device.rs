//! The device-share lock's token: how the graphics server takes a lock on the sender's memory
//! during a request, and how the timer thread releases it later
//! (`docs/specs/graphics-server.md`, "The model": the owner's ruling of 2026-11-17).
//!
//! No reference source is ported: Horizon's `KPageTableBase::LockForMapDeviceAddressSpace` and
//! `UnlockForDeviceAddressSpace` take a bare range from a caller in the right process. The
//! token is the spec's design, because the thread that learns the GPU is done is a host thread
//! with no current process.
//!
//! # Making a token
//!
//! [`IpcContext::lock_device_share`](super::IpcContext::lock_device_share) locks the sender's
//! range ([`AddressSpace::lock_device_share`]) and returns a [`DeviceShareToken`] naming the
//! process and that exact range. [`Dispatcher::reclaim_device_share`] makes one from what a
//! snapshot saved, only after the checks below pass for it: a restored token is checked as a
//! live one is, and the snapshot's releases of one range are counted against the locks the
//! kernel holds on it.
//!
//! Reclaiming is the one way outside tests to get a token from raw numbers (a process and a
//! range), and it is meant only for a snapshot restore (`starquake_gsv::completion`, "A token
//! held across a snapshot"). It is `pub` because gsv's restore calls it; the kernel cannot tell
//! a restore from any other caller holding a [`Dispatcher`]. What it does guarantee: it makes a
//! token only for a range on which the kernel holds more live locks than the caller says it
//! has already reclaimed (`earlier`), so a restore that counts as gsv's does never gets more
//! tokens for a range than the kernel has locks on it. A caller that passes a wrong count can
//! get a token for a lock that another token already names; unlocking it then releases that
//! lock, as unlocking the other token would.
//!
//! # Checking a token
//!
//! [`Dispatcher::unlock_device_share`] (the timer thread's path, and the request's through
//! [`IpcContext::unlock_device_share`](super::IpcContext::unlock_device_share)) and
//! [`Dispatcher::reclaim_device_share`] check, in this order, and answer the first failure:
//!
//! 1. [`DeviceShareError::WrongProcess`] unless the token names the process whose address
//!    space the dispatcher holds ([`Processes::own`](crate::tables::Processes::own)). One
//!    process has an address space today (`Tables`); a second brings its own, and this check
//!    becomes a lookup.
//! 2. [`DeviceShareError::ProcessEnded`] when that process has ended
//!    ([`ProcessState::Terminated`]) or its scheduler row is gone.
//! 3. [`DeviceShareError::NotLocked`] unless a live lock was taken on exactly the token's range
//!    ([`AddressSpace::device_locks`]): one page short or one page long is refused, and so is
//!    a second unlock of a token whose one lock the first released. A reclaim is refused unless
//!    more live locks are held on the range than the `earlier` releases of it the snapshot
//!    already reclaimed: with N locks, N releases of the range are taken back and the next one
//!    is refused (`as_many_reclaims_as_locks_are_taken_back_and_one_more_is_refused`).
//!
//! An unlock then releases the lock ([`AddressSpace::unlock_device_share`]), and its state
//! checks' refusal is [`DeviceShareError::Refused`]. Nothing changes on any refusal. Tests:
//! `dispatch/device/tests.rs`, one per refusal, and
//! `the_process_is_checked_before_the_range`, which has both faults at once.
//!
//! # When the process ends
//!
//! Ending a process releases every device-share lock its address space still holds
//! ([`release_ended`]): each live lock on each exact range ([`AddressSpace::device_share_locks`])
//! goes through [`AddressSpace::unlock_device_share`], the path a token's unlock takes, once per
//! lock, ranges in ascending order. It runs where the transfer-memory unlocks and the
//! shared-memory give-back of teardown run: when the dispatcher's lock is released after a call
//! that ended the process (`ExitProcess`, the last thread's `ExitThread`, or a service's
//! `IpcContext::end_process`), whichever call it was (`Locked`'s `Drop`, in `dispatch.rs`). In
//! teardown's order it sits after the transfer-memory unlocks and before the shared-memory
//! give-back, so every unlock of the process's pages comes before any page is given back.
//! A device-share release makes no call to the translation (it changes only the blocks'
//! device counts and attributes): shared memory cannot be device-locked, and a range lent as
//! transfer memory cannot be (and the other way round), so the three never meet on one page,
//! and the translation's calls are still `#224`'s order
//! (`exiting_the_process_releases_device_locks_and_keeps_the_unlock_and_give_back_order`, in
//! `dispatch/memory_objects/tests.rs`).
//!
//! Then a token of the process is refused as [`DeviceShareError::ProcessEnded`], before its
//! range is looked at, so a release that was queued before the process ended and is drained
//! after it releases nothing a second time
//! (`ending_the_process_releases_every_lock_once_and_a_late_unlock_is_refused`). While the
//! process lives, the check leaves its locks alone (`a_live_process_keeps_its_locks_across_calls`).
//!
//! No reference releases them this way: Horizon and Eden finalize the whole page table when the
//! process goes (`eden/src/core/hle/kernel/k_process.cpp` @ cf654052, `KProcess::Finalize`:
//! `m_page_table.Finalize()`), and the device counts go with the blocks. The address space here
//! is not finalized when its process ends (a snapshot or a test can still read it), so the
//! locks are released on their own. An unlock refused here is the kernel's own bookkeeping
//! going wrong (a range a lock was counted on that its blocks do not show locked): it is logged
//! as an error, the rest of that range's locks are left, and the next range's are released all
//! the same. That branch is log-only, and no
//! test reaches it, because a sound block map never refuses it.
//!
//! # What is recorded
//!
//! Nothing here. An unlock changes only the address space's counts and attributes, which a
//! guest sees only through a later system call; when the timer thread makes it is the place of
//! the release in gsv's completion queue, which is drained on one thread in one order
//! (`starquake_gsv::completion`). Whether that place gets a note of its own is gsv's to decide.

use super::Dispatcher;
use crate::id::ProcessId;
use crate::memory::{AddressSpace, DeviceShareToken};
use crate::process::ProcessState;
use crate::result::{CallResult, KernelError, SvcError};
use crate::tables::Scheduler;

#[cfg(test)]
mod tests;

/// Why the kernel refused a [`DeviceShareToken`], to unlock it or to take it back from a
/// snapshot (module docs, "Checking a token", for the order).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DeviceShareError {
    /// The token names a process whose address space the kernel does not hold.
    #[error("the device-share token names a process whose memory this kernel does not hold")]
    WrongProcess,
    /// The token's process has ended.
    #[error("the device-share token's process has ended")]
    ProcessEnded,
    /// No live lock was taken on exactly the token's range.
    #[error("no device-share lock is held on exactly {size:#x} bytes at {address:#x}")]
    NotLocked {
        /// Where the token's range starts.
        address: u64,
        /// How many bytes it holds.
        size: u64,
    },
    /// The address space's state checks refused the unlock.
    #[error("the address space refused the device-share unlock: {0}")]
    Refused(SvcError),
}

impl DeviceShareError {
    /// The result a server sees for the refusal, when it unlocks during a request
    /// ([`IpcContext::unlock_device_share`](super::IpcContext::unlock_device_share)):
    /// [`KernelError::InvalidCurrentMemory`], what the address space answers a range that is
    /// not locked, for the token's own refusals (starquake's choice: Horizon has no token), and
    /// the address space's own result otherwise.
    pub(super) fn into_svc_error(self) -> SvcError {
        match self {
            Self::WrongProcess | Self::ProcessEnded | Self::NotLocked { .. } => {
                KernelError::InvalidCurrentMemory.into()
            }
            Self::Refused(error) => error,
        }
    }
}

/// Checks `token` against the process the tables belong to, `own`, and the live locks on its
/// range, of which `earlier` are already spoken for (module docs, "Checking a token"): zero for
/// an unlock, and the count of the snapshot's earlier releases of the range for a reclaim.
fn check(
    own: ProcessId,
    space: &AddressSpace,
    scheduler: &Scheduler,
    token: DeviceShareToken,
    earlier: u32,
) -> Result<(), DeviceShareError> {
    if token.process() != own {
        return Err(DeviceShareError::WrongProcess);
    }
    if has_ended(own, scheduler) {
        return Err(DeviceShareError::ProcessEnded);
    }
    if space.device_locks(token.address(), token.size()) <= earlier {
        return Err(DeviceShareError::NotLocked {
            address: token.address(),
            size: token.size(),
        });
    }
    Ok(())
}

/// Whether `process` has ended ([`ProcessState::Terminated`]) or its scheduler row is gone: what
/// [`DeviceShareError::ProcessEnded`] means, and when [`release_ended`] releases.
fn has_ended(process: ProcessId, scheduler: &Scheduler) -> bool {
    scheduler
        .process(process)
        .map_or(true, |record| record.state() == ProcessState::Terminated)
}

/// Releases every device-share lock `space` holds if `own`, the process it belongs to, has
/// ended, each through [`AddressSpace::unlock_device_share`] (module docs, "When the process
/// ends"); does nothing while it lives.
pub(super) fn release_ended(space: &mut AddressSpace, scheduler: &Scheduler, own: ProcessId) {
    if !has_ended(own, scheduler) {
        return;
    }
    // Taken out first: each unlock changes the counts. As many entries as ranges locked.
    let held: Vec<(u64, u64, u32)> = space.device_share_locks().collect();
    for (address, size, count) in held {
        for _ in 0..count {
            if let Err(error) = space.unlock_device_share(address, size) {
                tracing::error!(
                    %error,
                    address,
                    size,
                    "releasing an ended process's device-share lock failed"
                );
                break;
            }
        }
    }
}

impl Dispatcher {
    /// Locks `size` bytes at `address` of the process the tables belong to and returns its
    /// token ([`AddressSpace::lock_device_share`], whose errors these are).
    pub(super) fn lock_device_share(
        &self,
        address: u64,
        size: u64,
    ) -> CallResult<DeviceShareToken> {
        let mut locked = self.lock::<(&mut AddressSpace,)>();
        let own = locked.state.tables.processes.own();
        let (space,) = locked.tables();
        space.lock_device_share(address, size)?;
        Ok(DeviceShareToken::new(own, address, size))
    }

    /// Releases the device-share lock `token` names, after checking it (module docs,
    /// "Checking a token"). The timer thread's path: thread-safe, as every dispatcher
    /// operation is, and it names no guest thread. Nothing changes on a refusal.
    pub fn unlock_device_share(&self, token: DeviceShareToken) -> Result<(), DeviceShareError> {
        let mut locked = self.lock::<(&mut AddressSpace, &Scheduler)>();
        let own = locked.state.tables.processes.own();
        let (space, scheduler) = locked.tables();
        check(own, space, scheduler, token, 0)?;
        space
            .unlock_device_share(token.address(), token.size())
            .map_err(DeviceShareError::Refused)
    }

    /// The token for a lock on `size` bytes at `address` of `process`, as a snapshot saved it,
    /// if the checks a live token passes pass for it and the kernel holds more live locks on
    /// that range than `earlier`, the number of the snapshot's releases of the same range that
    /// were reclaimed before this one (module docs, "Checking a token"). For a snapshot restore
    /// only (module docs, "Making a token"). It releases nothing; unlocking it checks it again.
    pub fn reclaim_device_share(
        &self,
        process: ProcessId,
        address: u64,
        size: u64,
        earlier: u32,
    ) -> Result<DeviceShareToken, DeviceShareError> {
        let token = DeviceShareToken::new(process, address, size);
        let mut locked = self.lock::<(&AddressSpace, &Scheduler)>();
        let own = locked.state.tables.processes.own();
        let (space, scheduler) = locked.tables();
        check(own, space, scheduler, token, earlier)?;
        Ok(token)
    }
}
