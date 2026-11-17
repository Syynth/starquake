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
//! A lock is not released when its process ends: the token is then refused, and the address
//! space goes with the process.
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
    let live = scheduler
        .process(own)
        .is_ok_and(|record| record.state() != ProcessState::Terminated);
    if !live {
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
