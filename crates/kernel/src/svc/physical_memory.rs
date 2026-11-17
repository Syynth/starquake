//! `MapPhysicalMemory` (0x2C) and `UnmapPhysicalMemory` (0x2D): memory the calling process maps
//! in its alias region for its own use, and gives back.
//!
//! Ported from Eden (GPL-3.0-or-later), paths relative to the reference folder:
//! `eden/src/core/hle/kernel/svc/svc_physical_memory.cpp` @ cf654052, `MapPhysicalMemory` and
//! `UnmapPhysicalMemory`, which check the arguments and then call the page table's functions of
//! the same names ([`AddressSpace::map_physical_memory`] and
//! [`AddressSpace::unmap_physical_memory`]). Eden-NX2's copy
//! (`eden-nx2/src/core/hle/kernel/svc/svc_physical_memory.cpp` @ 50fa86bb) adds only a log
//! line around the two calls' 64-bit wrappers, and PR #4588's
//! (`eden-pr4588/src/core/hle/kernel/svc/svc_physical_memory.cpp` @ e5fb3b97) changes only
//! `SetHeapSize`; neither changes these checks. The scoping doc has no row on these calls;
//! what Switch 2 changes is the size of the alias region ([`crate::memory::layout`]).
//!
//! # Results, in the order they are checked
//!
//! Both calls make the same checks:
//!
//! | Check | Result |
//! |---|---|
//! | address is page aligned (`Common::IsAligned(addr, YUZU_PAGESIZE)`) | `InvalidAddress` |
//! | size is page aligned | `InvalidSize` |
//! | size is not zero | `InvalidSize` |
//! | `addr < addr + size`: the range does not wrap, also to exactly zero | `InvalidMemoryRegion` |
//! | the process has a system resource (`GetTotalSystemResourceSize() != 0`) | `InvalidState` |
//! | the range lies inside the address space (`Contains`) | `InvalidMemoryRegion` |
//! | the range lies inside the alias region (`IsInAliasRegion`) | `InvalidMemoryRegion` |
//!
//! Then `MapPhysicalMemory` fails with `LimitReached` when the pool has no room for the pages
//! of the range that are free, and `UnmapPhysicalMemory` with `InvalidCurrentMemory` when a
//! block of the range is neither free nor `Normal` with no attributes
//! ([`AddressSpace::map_physical_memory`], [`AddressSpace::unmap_physical_memory`]).
//!
//! The page-table half checks the alias region again, with the same result
//! ([`AddressSpace::map_physical_memory`]; its module's "Divergences"), and in Eden too
//! `IsInAliasRegion` includes `Contains`. So which of the three checks refuses a range outside
//! the alias region is not observable from the result; the tests
//! (`each_check_refuses_at_its_edge_and_passes_just_inside_it`) pin the result at each edge.
//!
//! The system resource is the one `GetInfo` 16 reports
//! ([`secure_resource_size`](crate::svc::info::secure_resource_size): the size the program's
//! metadata asks for, in whole pages); the dispatcher reads it from the process facts.
//!
//! The 32-bit wrappers (`MapPhysicalMemory64From32`) are not built: only 64-bit programs run
//! (`crate::memory::layout`).

use super::memory::check_pages;
use crate::memory::AddressSpace;
use crate::result::{CallResult, KernelError, SvcResult};

/// The checks both calls make before they reach the page table, in Eden's order (module docs).
/// `system_resource_size` is the calling process's (`GetTotalSystemResourceSize`).
fn check(
    space: &AddressSpace,
    system_resource_size: u64,
    address: u64,
    size: u64,
) -> SvcResult<()> {
    // The address's alignment, then the size's, then the size is not zero.
    check_pages(address, size)?;
    // `addr < addr + size`, which is false when the sum wraps, also to exactly zero.
    if address.checked_add(size).is_none() {
        return Err(KernelError::InvalidMemoryRegion);
    }
    if system_resource_size == 0 {
        return Err(KernelError::InvalidState);
    }
    if !space.regions().contains(address, size) {
        return Err(KernelError::InvalidMemoryRegion);
    }
    if !space.is_in_alias_region(address, size) {
        return Err(KernelError::InvalidMemoryRegion);
    }
    Ok(())
}

/// Maps memory over the free pages of the `size` bytes at `address` in the calling process's
/// alias region. `system_resource_size` is the process's system resource size (module docs).
pub fn map_physical_memory(
    space: &mut AddressSpace,
    system_resource_size: u64,
    address: u64,
    size: u64,
) -> CallResult<()> {
    check(space, system_resource_size, address, size)?;
    space.map_physical_memory(address, size)
}

/// Unmaps the memory [`map_physical_memory`] mapped in the `size` bytes at `address` of the
/// calling process's alias region. `system_resource_size` is the process's system resource
/// size (module docs).
pub fn unmap_physical_memory(
    space: &mut AddressSpace,
    system_resource_size: u64,
    address: u64,
    size: u64,
) -> CallResult<()> {
    check(space, system_resource_size, address, size)?;
    space.unmap_physical_memory(address, size)
}

#[cfg(test)]
mod tests;
