//! The memory system calls of the calling process: `SetHeapSize` (0x01), `SetMemoryPermission`
//! (0x02), `SetMemoryAttribute` (0x03), `MapMemory` (0x04), `UnmapMemory` (0x05),
//! `QueryMemory` (0x06), and Switch 2's `SetGpuMemoryAttribute` (0x80).
//!
//! Each validates its arguments as Horizon does, in Horizon's order, and then calls the
//! [`AddressSpace`] operation of the same name. Ported from Eden (GPL-3.0-or-later), paths
//! relative to the reference folder:
//!
//! - `eden/src/core/hle/kernel/svc/svc_physical_memory.cpp` @ cf654052, `SetHeapSize`.
//! - `eden/src/core/hle/kernel/svc/svc_memory.cpp` @ cf654052, `SetMemoryPermission`
//!   (with `IsValidSetMemoryPermission`), `SetMemoryAttribute`, and `MapMemory` and
//!   `UnmapMemory` with `MapUnmapMemorySanityChecks` and `IsValidAddressRange`. Eden-NX2's and
//!   PR #4588's copies of the last four are Eden's, line for line
//!   (`eden-nx2/src/core/hle/kernel/svc/svc_memory.cpp` @ 50fa86bb,
//!   `eden-pr4588/src/core/hle/kernel/svc/svc_memory.cpp` @ e5fb3b97).
//! - `eden/src/core/hle/kernel/svc/svc_query_memory.cpp` @ cf654052, `QueryMemory` and
//!   `QueryProcessMemory`.
//! - The constants: `HeapSizeAlignment` (`eden/src/core/hle/kernel/svc_common.h`) and
//!   `MainMemorySizeMax` (`eden/src/core/hle/kernel/k_memory_layout.h`).
//! - `SetGpuMemoryAttribute`: Eden PR #4588's
//!   (`eden-pr4588/src/core/hle/kernel/svc/svc_memory.cpp` @ e5fb3b97), checked against
//!   Eden-NX2's (`eden-nx2/src/core/hle/kernel/svc.cpp` @ 50fa86bb, the `imm == 0x80` case of
//!   `Call`). Both then call the page table's `SetMemoryAttribute`, whose Switch 2 form is in
//!   [`AddressSpace::set_memory_attribute`]. The call is a Switch 2 one (scoping doc, "Process,
//!   memory, kernel": "New SVC: `svc 0x80`"); what a console does is known only from these two
//!   ports and from the SDK callers they captured, so its checks are theirs.
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
//! | `MapMemory`, `UnmapMemory` | the destination is page aligned | `InvalidAddress` |
//! | | the source is page aligned | `InvalidAddress` (Eden: `InvalidSize`) |
//! | | size is not zero | `InvalidSize` |
//! | | size is page aligned | `InvalidSize` |
//! | | `dst + size` does not wrap, then `src + size` | `InvalidCurrentMemory` |
//! | | the source lies inside the address space | `InvalidCurrentMemory` |
//! | | the destination may hold stack memory (not in Eden) | `InvalidMemoryRegion` |
//! | | then the page table's checks (`memory::space::alias`) | |
//! | `QueryMemory` | the guest may write every byte of the result: each block the 0x28 bytes touch has user write permission | `InvalidPointer` |
//! | | the result can be written to the caller's memory | `InvalidPointer` |
//! | `SetGpuMemoryAttribute` | the process was built for Switch 2 (PR: `UsesPAuthReturns`) | `NotImplemented` |
//! | | address, size and wrap, as `SetMemoryPermission` | as above |
//! | | the attribute lies inside the mask, and both inside `GpuAccessible` | `InvalidCombination` |
//! | | the range lies inside the address space | `InvalidCurrentMemory` |
//! | | the range's state allows the change (`FlagCanChangeAttribute`) and its attributes are settable or `DeviceShared` | `InvalidCurrentMemory` |
//!
//! # Divergences from Eden
//!
//! - **`SetHeapSize`'s upper bound** is PR #4588's
//!   (`eden-pr4588/src/core/hle/kernel/svc/svc_physical_memory.cpp` @ e5fb3b97):
//!   `size < max(MainMemorySizeMax, heap region size)`. Eden has `size < MainMemorySizeMax`
//!   alone. The two agree while the heap region is 8 GiB, which it is for a console with up
//!   to 8 GiB of memory; with more, a Switch 2 heap may exceed 8 GiB. Test:
//!   `the_heap_size_bound_is_the_larger_of_eight_gib_and_the_heap_region`.
//! - **`QueryMemory` with a result pointer the guest may not write** returns
//!   [`KernelError::InvalidPointer`]: when any block the result's
//!   [`SvcMemoryInfo::SIZE`] bytes touch lacks
//!   [`USER_WRITE`](crate::memory::MemoryPermission::USER_WRITE) (free
//!   memory, text, read-only data, a page made read-only or no access, or a range that wraps
//!   or reaches above the address space), checked with [`AddressSpace::user_can_write`],
//!   and when the caller's memory refuses the write. Eden's `WriteBlock` logs and the call
//!   succeeds. The result is Horizon's: its user-pointer copy goes through the guest's own
//!   permissions and fails with `svc::ResultInvalidPointer`
//!   (`atmosphere/libraries/libmesosphere/include/mesosphere/svc/kern_svc_k_user_pointer.hpp`
//!   @ 4e3f3703, fact only). The permission check is here because the caller's memory
//!   (core's guest memory) writes as the host, whatever the guest's permissions. Horizon
//!   checks the pointer before it queries; here the query comes first, which no guest can
//!   tell apart, since the query cannot fail. Tests:
//!   `query_memory_to_a_pointer_that_cannot_be_written_is_invalid_pointer`,
//!   `query_memory_to_memory_the_guest_may_not_write_is_invalid_pointer`,
//!   `query_memory_to_a_heap_page_made_inaccessible_is_invalid_pointer`.
//! - **`QueryMemory` takes no handle**: Eden resolves the current-process pseudo-handle and
//!   queries that process, which is the caller.
//!
//! - **`MapMemory` and `UnmapMemory` with a source that is not page aligned** return
//!   [`KernelError::InvalidAddress`]; Eden returns `ResultInvalidSize`
//!   (`MapUnmapMemorySanityChecks`: "Source address is not aligned to 4KB" then
//!   `R_THROW(ResultInvalidSize)`). The result is Horizon's: it checks both addresses for
//!   alignment with `svc::ResultInvalidAddress`
//!   (`atmosphere/libraries/libmesosphere/source/svc/kern_svc_memory.cpp` @ 4e3f3703,
//!   `MapMemory` and `UnmapMemory`, fact only). Test:
//!   `a_source_that_is_not_page_aligned_is_invalid_address_not_eden_s_invalid_size`.
//! - **`MapMemory` and `UnmapMemory` check that the stack region can hold the destination**
//!   (`CanContain(dst_address, size, KMemoryState::Stack)`), last of the argument checks, with
//!   [`KernelError::InvalidMemoryRegion`]. Eden's `MapUnmapMemorySanityChecks` has no such
//!   check, so in Eden a free destination anywhere in the address space is mapped as stack
//!   memory. The check and its result are Horizon's (the same Atmosphère file, fact only).
//!   Test: `the_destination_must_be_one_the_stack_region_can_hold`.
//! - **The order of the two wrap checks** is Eden's (destination, then source); Horizon checks
//!   the source first. Both give `InvalidCurrentMemory`, so no guest can tell them apart.
//!
//! # Where the two Switch 2 references differ on `SetGpuMemoryAttribute`
//!
//! PR #4588 is followed, and Eden-NX2 differs in two places, each pinned by a test:
//!
//! - **A process not built for Switch 2.** PR returns `ResultNotImplemented` before any other
//!   check (`R_UNLESS(process.UsesPAuthReturns(), ResultNotImplemented)`); Eden-NX2 runs the
//!   call for any process. Test:
//!   `set_gpu_memory_attribute_is_not_implemented_for_a_switch_1_process`.
//! - **A range that wraps and a combination that is wrong, at once.** PR checks the wrap
//!   (`ResultInvalidCurrentMemory`) before the combination (`ResultInvalidCombination`), as
//!   `SetMemoryAttribute` does; Eden-NX2 checks the combination first and the wrap together
//!   with `Contains`. Test: `set_gpu_memory_attribute_checks_the_wrap_before_the_combination`.
//!
//! `QueryMemory` reports the attribute as bit 5 of the block's attributes, as it reports the
//! others (`KMemoryInfo::GetSvcMemoryInfo` copies them).

use starquake_memory::{GuestMemory, PAGE_SIZE};

use crate::memory::{AddressSpace, MemoryAttribute, MemoryState, SvcMemoryInfo, SvcPermission};
use crate::result::{CallResult, KernelError, SvcResult};

/// What a heap's size must be a multiple of (`HeapSizeAlignment`): 2 MiB.
pub const HEAP_SIZE_ALIGNMENT: u64 = 2 << 20;

/// The size a heap must stay below, unless the heap region is larger (`MainMemorySizeMax`):
/// 8 GiB.
pub const MAIN_MEMORY_SIZE_MAX: u64 = 8 << 30;

/// The value of `PageInfo::flags` that `QueryMemory` returns: always zero
/// (`KPageTableBase::QueryInfoImpl`: `out_page->flags = 0`).
pub const PAGE_INFO_FLAGS: u32 = 0;

/// The size of what `QueryMemory` writes, as an address offset.
const RESULT_SIZE: u64 = SvcMemoryInfo::SIZE as u64;

/// The first two checks of every call that takes a range of pages: a page-aligned address
/// ([`KernelError::InvalidAddress`]), then a page-aligned size that is not zero
/// ([`KernelError::InvalidSize`]). What a call checks next, the wrap, has a different result in
/// different calls.
pub(super) fn check_pages(address: u64, size: u64) -> SvcResult<()> {
    if !address.is_multiple_of(PAGE_SIZE) {
        return Err(KernelError::InvalidAddress);
    }
    if !size.is_multiple_of(PAGE_SIZE) || size == 0 {
        return Err(KernelError::InvalidSize);
    }
    Ok(())
}

/// The checks `SetMemoryPermission` and `SetMemoryAttribute` share, and the shared and
/// transfer memory calls open with: a page-aligned address ([`KernelError::InvalidAddress`]), a
/// page-aligned size that is not zero ([`KernelError::InvalidSize`]), and a range that does not
/// wrap ([`KernelError::InvalidCurrentMemory`]), in that order.
pub(super) fn check_range(address: u64, size: u64) -> SvcResult<()> {
    check_pages(address, size)?;
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
    let supported = MemoryAttribute::SET_MEMORY_ATTRIBUTE_SUPPORTED.svc();
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

/// Replaces the GPU-accessible attribute in `mask` with `attribute` on `size` bytes of the
/// calling process's memory at `address` (`SetGpuMemoryAttribute`, Switch 2's `svc 0x80`): what
/// the SDK does to memory before it registers it with the GPU, and undoes when it destroys it.
/// `switch_2` says whether the calling process was built for Switch 2. Both bit sets are the raw
/// values the guest passed. See the module docs for the checks and their order.
pub fn set_gpu_memory_attribute(
    space: &mut AddressSpace,
    switch_2: bool,
    address: u64,
    size: u64,
    mask: u32,
    attribute: u32,
) -> CallResult<()> {
    if !switch_2 {
        return Err(KernelError::NotImplemented.into());
    }
    // Validate address / size.
    check_range(address, size)?;
    // Validate the attribute and mask: only the GPU-accessible bit.
    let supported = MemoryAttribute::GPU_ACCESSIBLE.svc();
    if attribute & !mask != 0 || (mask | attribute) & !supported != 0 {
        return Err(KernelError::InvalidCombination.into());
    }
    // Validate that the region is in range for the current process.
    if !space.regions().contains(address, size) {
        return Err(KernelError::InvalidCurrentMemory.into());
    }
    // Both fit eight bits: they lie inside the supported bit.
    let narrow = |bits: u32| {
        u8::try_from(bits)
            .map(MemoryAttribute::from_bits)
            .map_err(|_| KernelError::InvalidCombination)
    };
    space.set_memory_attribute(address, size, narrow(mask)?, narrow(attribute)?)
}

/// `MapUnmapMemorySanityChecks`, with Horizon's result for an unaligned source and its stack
/// region check (module docs, "Divergences"): the checks `MapMemory` and `UnmapMemory` make of
/// their arguments, in the module docs' order, before the page table's.
fn map_unmap_memory_sanity_checks(
    space: &AddressSpace,
    dst: u64,
    src: u64,
    size: u64,
) -> SvcResult<()> {
    if !dst.is_multiple_of(PAGE_SIZE) {
        return Err(KernelError::InvalidAddress);
    }
    if !src.is_multiple_of(PAGE_SIZE) {
        return Err(KernelError::InvalidAddress);
    }
    if size == 0 {
        return Err(KernelError::InvalidSize);
    }
    if !size.is_multiple_of(PAGE_SIZE) {
        return Err(KernelError::InvalidSize);
    }
    // `IsValidAddressRange`: `address + size > address`, false when the sum wraps (to exactly
    // zero too).
    if dst.checked_add(size).is_none() {
        return Err(KernelError::InvalidCurrentMemory);
    }
    if src.checked_add(size).is_none() {
        return Err(KernelError::InvalidCurrentMemory);
    }
    if !space.regions().contains(src, size) {
        return Err(KernelError::InvalidCurrentMemory);
    }
    if !space.can_contain(dst, size, MemoryState::Stack) {
        return Err(KernelError::InvalidMemoryRegion);
    }
    Ok(())
}

/// Maps the `size` bytes of the calling process's memory at `src` a second time at `dst`, in
/// its stack region, and takes them from the guest at `src` until [`unmap_memory`]
/// (`MapMemory`). See the module docs for the checks and their order, and
/// [`AddressSpace::map_memory`] for the page table's.
pub fn map_memory(space: &mut AddressSpace, dst: u64, src: u64, size: u64) -> CallResult<()> {
    map_unmap_memory_sanity_checks(space, dst, src, size)?;
    space.map_memory(dst, src, size)
}

/// Unmaps the alias of `size` bytes at `dst` that [`map_memory`] made of `src`, and gives the
/// source back to the guest (`UnmapMemory`). See the module docs for the checks and their
/// order, and [`AddressSpace::unmap_memory`] for the page table's.
pub fn unmap_memory(space: &mut AddressSpace, dst: u64, src: u64, size: u64) -> CallResult<()> {
    map_unmap_memory_sanity_checks(space, dst, src, size)?;
    space.unmap_memory(dst, src, size)
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
    // The guest's permissions, which `memory`'s host writes do not check.
    if !space.user_can_write(out_memory_info, RESULT_SIZE) {
        return Err(KernelError::InvalidPointer.into());
    }
    memory
        .write(out_memory_info, &info.to_bytes())
        .map_err(|_| KernelError::InvalidPointer)?;
    Ok(PAGE_INFO_FLAGS)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod alias_tests;
