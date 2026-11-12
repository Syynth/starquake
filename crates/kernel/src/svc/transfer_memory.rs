//! `CreateTransferMemory` (0x15), and unlocking the owner's pages when a transfer memory goes.
//!
//! Ported from Eden (GPL-3.0-or-later), paths relative to the reference folder:
//!
//! - `eden/src/core/hle/kernel/svc/svc_transfer_memory.cpp` @ cf654052: `CreateTransferMemory`
//!   (with `IsValidTransferMemoryPermission`).
//! - `eden/src/core/hle/kernel/k_transfer_memory.cpp` @ cf654052: `KTransferMemory::Initialize`
//!   (lock the owner's pages) and `Finalize` (unlock them, unless the receiver mapped them).
//!
//! The page-table steps are [`AddressSpace::lock_for_transfer_memory`] and
//! [`AddressSpace::unlock_for_transfer_memory`].
//!
//! # Results, in the order they are checked
//!
//! | Check | Result |
//! |---|---|
//! | address is page aligned | `InvalidAddress` |
//! | size is page aligned and not zero | `InvalidSize` |
//! | `address + size` does not wrap | `InvalidCurrentMemory` |
//! | permission is none, read, or read-write | `InvalidNewMemoryPermission` |
//! | the object table has room | `OutOfResource` |
//! | the range lies inside the address space (`Contains`) | `InvalidCurrentMemory` |
//! | the range can be transferred, is read-write, and has no attributes | `InvalidCurrentMemory` |
//! | the handle table has room | `OutOfHandles` |
//!
//! When the handle cannot be added, the transfer memory goes with the creator's reference
//! (Eden's `SCOPE_EXIT`), and its pages are unlocked before the call returns.
//!
//! # Divergences from Eden
//!
//! - **No resource limit.** Eden reserves one `TransferMemoryCountMax` from the process's
//!   resource limit first (`ResultLimitReached`; Eden's limit is 200,
//!   `CreateResourceLimitForProcess` in `eden/src/core/hle/kernel/k_resource_limit.cpp`).
//!   Starquake's resource limits are a later slice, so this call never fails that way.
//!   PROVISIONAL. Test: `more_transfer_memories_than_edens_limit_are_all_lent`.
//! - **The receiver never maps the pages.** `MapTransferMemory` (0x51) and
//!   `UnmapTransferMemory` (0x52) are server-side calls, deferred with the rest (kernel spec);
//!   an HLE server reads and writes the owner's pages where they are, through
//!   [`IpcContext::read_transfer_memory`](crate::dispatch::IpcContext::read_transfer_memory)
//!   and [`IpcContext::write_transfer_memory`](crate::dispatch::IpcContext::write_transfer_memory),
//!   as Eden's HLE services read `GetSourceAddress`. So `Finalize`'s "unless mapped" never
//!   applies: a locked transfer memory always unlocks its pages when it goes. Tests:
//!   `a_server_reads_and_writes_a_transfer_memory_it_is_sent_within_its_bounds` (in
//!   `dispatch/memory_objects/tests.rs`),
//!   `while_lent_the_owner_cannot_change_the_pages_and_closing_the_handle_gives_them_back`.
//! - **A failed unlock is an internal failure, and the other unlocks still run**
//!   ([`unlock_released`]); as the kernel lock is let go it is logged and the run goes on. Eden
//!   asserts it succeeds. Both mean the kernel's own bookkeeping is broken. Test:
//!   `unlocking_goes_on_past_a_failure_and_reports_the_first`.

use super::memory::check_range;
use crate::handle::{Handle, HandleTable};
use crate::id::{ObjectId, ProcessId};
use crate::memory::{AddressSpace, SvcPermission, TransferMemory, is_transfer_memory_permission};
use crate::object::Object;
use crate::result::{CallResult, KernelError, SvcError};
use crate::tables::Scheduler;

/// Lends `size` bytes of the caller's memory at `address` to a new transfer memory, of which the
/// caller keeps `permission` (the raw value the guest passed), and returns a handle to it.
/// `owner` is the calling process.
pub fn create_transfer_memory(
    handles: &mut HandleTable,
    space: &mut AddressSpace,
    scheduler: &mut Scheduler,
    owner: ProcessId,
    address: u64,
    size: u64,
    permission: u32,
) -> CallResult<Handle> {
    // Validate the size.
    check_range(address, size)?;
    // Validate the permissions.
    let permission = SvcPermission::from_raw(permission)
        .filter(|permission| is_transfer_memory_permission(*permission))
        .ok_or(KernelError::InvalidNewMemoryPermission)?;
    // Create the transfer memory.
    let unlocked = TransferMemory::new(owner, address, size, permission);
    let object = scheduler
        .objects_mut()
        .create(Object::TransferMemory(unlocked))?;
    let added = lock_and_add(handles, space, scheduler, object, unlocked);
    // Ensure the only reference is in the handle table when we're done. When the handle could
    // not be added this is the last reference, and the pages are unlocked at once.
    let closed = scheduler.close_object(object);
    let unlocks = unlock_released(scheduler, space, owner);
    let handle = added?;
    closed?;
    unlocks?;
    Ok(handle)
}

/// The steps between creating the object and closing the creator's reference.
fn lock_and_add(
    handles: &mut HandleTable,
    space: &mut AddressSpace,
    scheduler: &mut Scheduler,
    object: ObjectId,
    unlocked: TransferMemory,
) -> CallResult<Handle> {
    let (address, size) = (unlocked.address(), unlocked.size());
    // Ensure that the region is in range.
    if !space.regions().contains(address, size) {
        return Err(KernelError::InvalidCurrentMemory.into());
    }
    // Initialize the transfer memory: lock the memory.
    space.lock_for_transfer_memory(address, size, unlocked.owner_permission())?;
    let locked = Object::TransferMemory(unlocked.with_pages_locked());
    scheduler.objects_mut().replace(object, locked)?;
    // Add the transfer memory to the handle table.
    Ok(handles.add(scheduler.objects_mut(), object)?)
}

/// Unlocks the pages of every transfer memory that went while they were locked
/// ([`crate::object::Objects::take_unlocks`]), oldest first: `KTransferMemory::Finalize` for
/// each. `own` is the process whose address space `space` is. All are unlocked even when one
/// fails, and the first failure is returned: every failure is the kernel's own bookkeeping (a
/// transfer memory of another process, whose address space is not this one, or pages that are
/// no longer as the lock left them).
pub fn unlock_released(
    scheduler: &mut Scheduler,
    space: &mut AddressSpace,
    own: ProcessId,
) -> CallResult<()> {
    let mut result = Ok(());
    for transfer in scheduler.objects_mut().take_unlocks() {
        let unlocked = if transfer.owner() == own {
            space.unlock_for_transfer_memory(transfer.address(), transfer.size())
        } else {
            Err(SvcError::Internal(format!(
                "a transfer memory of {:?} went in the tables of {own:?}",
                transfer.owner()
            )))
        };
        if result.is_ok() {
            result = unlocked;
        }
    }
    result
}

#[cfg(test)]
mod tests;
