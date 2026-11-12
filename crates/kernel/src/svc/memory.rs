//! The memory system calls of the calling process: `SetHeapSize` (0x01), `SetMemoryPermission`
//! (0x02), `SetMemoryAttribute` (0x03) and `QueryMemory` (0x06).
//!
//! Each validates its arguments as Horizon does, in Horizon's order, and then calls the
//! [`AddressSpace`] operation of the same name. Ported from Eden (GPL-3.0-or-later), paths
//! relative to the reference folder:
//!
//! - `eden/src/core/hle/kernel/svc/svc_physical_memory.cpp` @ cf654052, `SetHeapSize`.
//! - `eden/src/core/hle/kernel/svc/svc_memory.cpp` @ cf654052, `SetMemoryPermission`
//!   (with `IsValidSetMemoryPermission`) and `SetMemoryAttribute`.
//! - `eden/src/core/hle/kernel/svc/svc_query_memory.cpp` @ cf654052, `QueryMemory` and
//!   `QueryProcessMemory`.
//! - The constants: `HeapSizeAlignment` (`eden/src/core/hle/kernel/svc_common.h`) and
//!   `MainMemorySizeMax` (`eden/src/core/hle/kernel/k_memory_layout.h`).
//!
//! # Results, in the order they are checked
//!
//! | Call | Check | Result |
//! |---|---|---|
//! | `SetHeapSize` | size is a multiple of [`HEAP_SIZE_ALIGNMENT`] | `InvalidSize` |
//! | | size is below [`MAIN_MEMORY_SIZE_MAX`], or below the heap region's size if that is larger | `InvalidSize` |
//! | | size fits the heap region and the largest heap | `OutOfMemory` |
//! | | the pool has room for the growth | `LimitReached` |
//! | | the memory given up is plain heap, the memory taken is free | `InvalidCurrentMemory` |
//! | `SetMemoryPermission` | address is page aligned | `InvalidAddress` |
//! | | size is page aligned and not zero | `InvalidSize` |
//! | | `address + size` does not wrap | `InvalidCurrentMemory` |
//! | | permission is none, read, or read-write | `InvalidNewMemoryPermission` |
//! | | the range lies inside the address space | `InvalidCurrentMemory` |
//! | | the range can be reprotected and has no attributes | `InvalidCurrentMemory` |
//! | `SetMemoryAttribute` | address, size and wrap, as above | as above |
//! | | the attribute lies inside the mask | `InvalidCombination` |
//! | | mask and attribute lie inside `Uncached \| PermissionLocked` | `InvalidCombination` |
//! | | `PermissionLocked` is set if it is masked | `InvalidCombination` |
//! | | the range lies inside the address space | `InvalidCurrentMemory` |
//! | | the range's state allows the change | `InvalidCurrentMemory` |
//! | `QueryMemory` | the result can be written to the caller's memory | `InvalidPointer` |
//!
//! # Divergences from Eden
//!
//! - **`SetHeapSize`'s upper bound** is PR #4588's
//!   (`eden-pr4588/src/core/hle/kernel/svc/svc_physical_memory.cpp` @ e5fb3b97):
//!   `size < max(MainMemorySizeMax, heap region size)`. Eden has `size < MainMemorySizeMax`
//!   alone. The two agree while the heap region is 8 GiB, which it is for a console with up
//!   to 8 GiB of memory; with more, a Switch 2 heap may exceed 8 GiB. Test:
//!   `the_heap_size_bound_is_the_larger_of_eight_gib_and_the_heap_region`.
//! - **`QueryMemory` with a result pointer that cannot be written** returns
//!   [`KernelError::InvalidPointer`]. Eden's `WriteBlock` logs and the call succeeds. The
//!   result is Horizon's: its user-pointer copy fails with `svc::ResultInvalidPointer`
//!   (`atmosphere/libraries/libmesosphere/include/mesosphere/svc/kern_svc_k_user_pointer.hpp`
//!   @ 4e3f3703, fact only). Horizon checks the pointer before it queries; here the query
//!   comes first, which no guest can tell apart, since the query cannot fail. Test:
//!   `query_memory_to_a_pointer_that_cannot_be_written_is_invalid_pointer`.
//! - **`QueryMemory` takes no handle**: Eden resolves the current-process pseudo-handle and
//!   queries that process, which is the caller.
//!
//! The Switch 2 attribute `GpuAccessible` (`svc 0x80`, and bit 5 of what `QueryMemory`
//! reports) is not built; it lands with that call.

use starquake_memory::{GuestMemory, PAGE_SIZE};

use crate::memory::{AddressSpace, MemoryAttribute, SvcPermission};
use crate::result::{CallResult, KernelError, SvcResult};

/// What a heap's size must be a multiple of (`HeapSizeAlignment`): 2 MiB.
pub const HEAP_SIZE_ALIGNMENT: u64 = 2 << 20;

/// The size a heap must stay below, unless the heap region is larger (`MainMemorySizeMax`):
/// 8 GiB.
pub const MAIN_MEMORY_SIZE_MAX: u64 = 8 << 30;

/// The value of `PageInfo::flags` that `QueryMemory` returns: always zero
/// (`KPageTableBase::QueryInfoImpl`: `out_page->flags = 0`).
pub const PAGE_INFO_FLAGS: u32 = 0;

/// The checks `SetMemoryPermission` and `SetMemoryAttribute` share: a page-aligned address, a
/// page-aligned size that is not zero, and a range that does not wrap.
fn check_range(address: u64, size: u64) -> SvcResult<()> {
    if !address.is_multiple_of(PAGE_SIZE) {
        return Err(KernelError::InvalidAddress);
    }
    if !size.is_multiple_of(PAGE_SIZE) || size == 0 {
        return Err(KernelError::InvalidSize);
    }
    // `address < address + size`, which is false when the sum wraps, also to exactly zero.
    if address.checked_add(size).is_none() {
        return Err(KernelError::InvalidCurrentMemory);
    }
    Ok(())
}

/// Sets the size of the calling process's heap and returns the heap's address. It can both
/// extend and shrink the heap.
pub fn set_heap_size(space: &mut AddressSpace, size: u64) -> CallResult<u64> {
    // Validate size.
    if !size.is_multiple_of(HEAP_SIZE_ALIGNMENT) {
        return Err(KernelError::InvalidSize.into());
    }
    if size >= MAIN_MEMORY_SIZE_MAX.max(space.regions().heap.size()) {
        return Err(KernelError::InvalidSize.into());
    }
    // Set the heap size.
    space.set_heap_size(size)
}

/// Changes what the calling process may do with `size` bytes of its data at `address`.
/// `permission` is the raw value the guest passed.
pub fn set_memory_permission(
    space: &mut AddressSpace,
    address: u64,
    size: u64,
    permission: u32,
) -> CallResult<()> {
    // Validate address / size.
    check_range(address, size)?;
    // Validate the permission: none, read, or read and write.
    let permission = SvcPermission::from_raw(permission)
        .filter(|permission| *permission != SvcPermission::ReadExecute)
        .ok_or(KernelError::InvalidNewMemoryPermission)?;
    // Validate that the region is in range for the current process.
    if !space.regions().contains(address, size) {
        return Err(KernelError::InvalidCurrentMemory.into());
    }
    space.set_memory_permission(address, size, permission)
}

/// Replaces the attributes in `mask` with `attribute` on `size` bytes of the calling process's
/// memory at `address`. Both are the raw values the guest passed.
pub fn set_memory_attribute(
    space: &mut AddressSpace,
    address: u64,
    size: u64,
    mask: u32,
    attribute: u32,
) -> CallResult<()> {
    // Validate address / size.
    check_range(address, size)?;
    // Validate the attribute and mask.
    let supported = MemoryAttribute::SET_MASK.svc();
    if mask | attribute != mask || mask | attribute | supported != supported {
        return Err(KernelError::InvalidCombination.into());
    }
    // Check that permission locked is either being set or not masked.
    let lock = MemoryAttribute::PERMISSION_LOCKED.svc();
    if mask & lock != attribute & lock {
        return Err(KernelError::InvalidCombination.into());
    }
    // Validate that the region is in range for the current process.
    if !space.regions().contains(address, size) {
        return Err(KernelError::InvalidCurrentMemory.into());
    }
    // Both fit eight bits: they lie inside the supported mask.
    let narrow = |bits: u32| {
        u8::try_from(bits)
            .map(MemoryAttribute::from_bits)
            .map_err(|_| KernelError::InvalidCombination)
    };
    space.set_memory_attribute(address, size, narrow(mask)?, narrow(attribute)?)
}

/// Writes what the calling process's address space holds at `address` to `out_memory_info` in
/// `memory`, the caller's memory, and returns the page info word. Any address can be queried:
/// one above the address space is in a block of inaccessible memory.
pub fn query_memory(
    space: &AddressSpace,
    memory: &dyn GuestMemory,
    out_memory_info: u64,
    address: u64,
) -> CallResult<u32> {
    let info = space.query_info(address)?.svc_info();
    memory
        .write(out_memory_info, &info.to_bytes())
        .map_err(|_| KernelError::InvalidPointer)?;
    Ok(PAGE_INFO_FLAGS)
}

#[cfg(test)]
mod tests;
