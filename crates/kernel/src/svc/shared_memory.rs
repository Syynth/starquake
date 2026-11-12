//! `MapSharedMemory` (0x13) and `UnmapSharedMemory` (0x14).
//!
//! Ported from Eden (GPL-3.0-or-later), paths relative to the reference folder:
//!
//! - `eden/src/core/hle/kernel/svc/svc_shared_memory.cpp` @ cf654052: `MapSharedMemory` (with
//!   `IsValidSharedMemoryPermission`) and `UnmapSharedMemory`.
//! - `eden/src/core/hle/kernel/k_shared_memory.cpp` @ cf654052: `KSharedMemory::Map` and
//!   `Unmap`, which the calls end in.
//! - `eden/src/core/hle/kernel/k_process.cpp` @ cf654052: `KProcess::AddSharedMemory` and
//!   `RemoveSharedMemory`, which open a reference to the shared memory for each mapping and
//!   close it when the mapping goes (or when the mapping fails, `ON_RESULT_FAILURE`).
//!
//! The page-table steps are [`AddressSpace::map_shared`] and [`AddressSpace::unmap_shared`].
//!
//! # Results, in the order they are checked
//!
//! | Call | Check | Result |
//! |---|---|---|
//! | `MapSharedMemory` | address is page aligned | `InvalidAddress` |
//! | | size is page aligned and not zero | `InvalidSize` |
//! | | `address + size` does not wrap | `InvalidCurrentMemory` |
//! | | permission is read or read-write | `InvalidNewMemoryPermission` |
//! | | the handle names a shared memory | `InvalidHandle` |
//! | | the range may hold shared memory (`CanContain`) | `InvalidMemoryRegion` |
//! | | size is the shared memory's size | `InvalidSize` |
//! | | permission is the shared memory's user permission | `InvalidNewMemoryPermission` |
//! | | the range is free | `InvalidCurrentMemory` |
//! | `UnmapSharedMemory` | address, size and wrap, as above | as above |
//! | | the handle names a shared memory | `InvalidHandle` |
//! | | the range may hold shared memory | `InvalidMemoryRegion` |
//! | | size is the shared memory's size | `InvalidSize` |
//! | | the range is shared memory with no attributes, and is a mapping of this shared memory that starts at `address` | `InvalidCurrentMemory` |
//!
//! Eden's `KProcess::AddSharedMemory` can also fail, with `ResultOutOfResource`, when it has no
//! room for the process's record of the shared memory; here the record is a reference in the
//! object table and the address space's map of mappings, which have no such limit (a reference
//! count that would overflow is an internal failure).
//!
//! Divergences from Eden are listed in [`crate::memory::shared`] and
//! [`crate::memory::space`] ("No page groups"), each with its test.

use super::memory::check_range;
use crate::handle::{Handle, HandleTable, ObjectRefs};
use crate::id::ObjectId;
use crate::memory::{
    AddressSpace, MemoryState, SharedMemory, SvcPermission, is_shared_memory_permission,
};
use crate::object::Object;
use crate::result::{CallResult, KernelError, SvcError};
use crate::tables::Scheduler;

/// The shared memory `handle` names in the caller's table (`GetObject<KSharedMemory>`, which
/// refuses both pseudo-handles), or [`KernelError::InvalidHandle`].
fn shared_memory_of(
    handles: &HandleTable,
    scheduler: &Scheduler,
    handle: Handle,
) -> CallResult<(ObjectId, SharedMemory)> {
    let object = handles
        .get(handle)
        .map_err(|_| KernelError::InvalidHandle)?;
    match scheduler.objects().get(object)? {
        Object::SharedMemory(memory) => Ok((object, memory)),
        _ => Err(KernelError::InvalidHandle.into()),
    }
}

/// The checks both calls open with, and the shared memory `handle` names, mapped or unmapped
/// at `address`, which must be a range that may hold shared memory.
fn target(
    handles: &HandleTable,
    space: &AddressSpace,
    scheduler: &Scheduler,
    handle: Handle,
    address: u64,
    size: u64,
) -> CallResult<(ObjectId, SharedMemory)> {
    // Get the shared memory.
    let (object, memory) = shared_memory_of(handles, scheduler, handle)?;
    // Verify that the mapping is in range.
    if !space.can_contain(address, size, MemoryState::Shared) {
        return Err(KernelError::InvalidMemoryRegion.into());
    }
    Ok((object, memory))
}

/// Maps the shared memory `handle` names at `address` in the caller's address space, with
/// `permission` (the raw value the guest passed). `size` must be the shared memory's size.
pub fn map_shared_memory(
    handles: &HandleTable,
    space: &mut AddressSpace,
    scheduler: &mut Scheduler,
    handle: Handle,
    address: u64,
    size: u64,
    permission: u32,
) -> CallResult<()> {
    // Validate the address/size.
    check_range(address, size)?;
    // Validate the permission.
    let permission = SvcPermission::from_raw(permission)
        .filter(|permission| is_shared_memory_permission(*permission))
        .ok_or(KernelError::InvalidNewMemoryPermission)?;
    let (object, memory) = target(handles, space, scheduler, handle, address, size)?;
    // Add the shared memory to the process: the mapping holds a reference.
    scheduler
        .objects_mut()
        .open(object)
        .map_err(|error| SvcError::internal(&error))?;
    let mapped = map(space, object, memory, address, size, permission);
    if mapped.is_err() {
        // Ensure that we clean up the shared memory if we fail to map it. The handle still
        // holds a reference, so this is not the last.
        scheduler.close_object(object)?;
    }
    mapped
}

/// `KSharedMemory::Map`: the size and permission checks, then the page table.
fn map(
    space: &mut AddressSpace,
    object: ObjectId,
    memory: SharedMemory,
    address: u64,
    size: u64,
    permission: SvcPermission,
) -> CallResult<()> {
    // Validate the size.
    if memory.size() != size {
        return Err(KernelError::InvalidSize.into());
    }
    // Validate the permission. No process owns a shared memory here (see
    // `crate::memory::shared`), so the test permission is always the user permission.
    if permission != memory.user_permission() {
        return Err(KernelError::InvalidNewMemoryPermission.into());
    }
    space.map_shared(address, object, memory, permission)
}

/// Unmaps the shared memory `handle` names from `address` in the caller's address space, where
/// [`map_shared_memory`] mapped it. `size` must be the shared memory's size.
pub fn unmap_shared_memory(
    handles: &HandleTable,
    space: &mut AddressSpace,
    scheduler: &mut Scheduler,
    handle: Handle,
    address: u64,
    size: u64,
) -> CallResult<()> {
    // Validate the address/size.
    check_range(address, size)?;
    let (object, memory) = target(handles, space, scheduler, handle, address, size)?;
    // Validate the size.
    if memory.size() != size {
        return Err(KernelError::InvalidSize.into());
    }
    // Unmap the shared memory.
    space.unmap_shared(address, size, object)?;
    // Remove the shared memory from the process: the mapping's reference goes.
    scheduler.close_object(object)
}

#[cfg(test)]
mod tests;
