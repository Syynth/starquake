//! The page-table half of `MapPhysicalMemory` and `UnmapPhysicalMemory`: what each does to the
//! blocks, the translation and the memory count, and what each refuses.
//!
//! "Eden" below is `eden/src/core/hle/kernel/k_page_table_base.cpp` @ cf654052,
//! `KPageTableBase::MapPhysicalMemory` and `UnmapPhysicalMemory`.

use starquake_memory::Perms;

use super::*;
use crate::memory::fixture::loaded;
use crate::memory::testing::{MapperCall, MapperLog};

/// What the tests read of a block: where it is, how many pages, its state, permission,
/// attributes and merge barriers.
type Row = (
    u64,
    u64,
    MemoryState,
    MemoryPermission,
    MemoryAttribute,
    DisableMerge,
);

/// The fixture's loaded address space, and the address of page `index` of its alias region.
fn alias_space() -> (AddressSpace, MapperLog, impl Fn(u64) -> u64) {
    let (space, log, _) = loaded();
    let alias = space.regions().alias.start;
    (space, log, move |index: u64| alias + index * PAGE_SIZE)
}

/// The blocks that `from..to` touches, each cut to that range: a block that starts below
/// `from` is shown from `from`, one that ends past `to` up to `to`.
fn rows(space: &AddressSpace, from: u64, to: u64) -> Vec<Row> {
    space
        .blocks()
        .filter(|block| block.address() < to && from <= block.last())
        .map(|block| {
            let start = block.address().max(from);
            let end = block.end().map_or(to, |end| end.min(to));
            (
                start,
                (end - start) / PAGE_SIZE,
                block.state(),
                block.permission(),
                block.attribute(),
                block.disable_merge(),
            )
        })
        .collect()
}

/// A row of mapped physical memory with no barrier.
fn physical(address: u64, pages: u64) -> Row {
    (
        address,
        pages,
        MemoryState::Normal,
        MemoryPermission::USER_READ_WRITE,
        MemoryAttribute::NONE,
        DisableMerge::NONE,
    )
}

/// A row of free memory with no barrier.
fn free(address: u64, pages: u64) -> Row {
    (
        address,
        pages,
        MemoryState::Free,
        MemoryPermission::NONE,
        MemoryAttribute::NONE,
        DisableMerge::NONE,
    )
}

/// A row of IPC memory as [`ipc`] maps it.
fn ipc_row(address: u64, pages: u64) -> Row {
    (
        address,
        pages,
        MemoryState::Ipc,
        MemoryPermission::USER_READ_WRITE,
        MemoryAttribute::NONE,
        DisableMerge::NORMAL,
    )
}

/// Maps `pages` pages of IPC memory at `address`: memory of another state than `Normal` that
/// may lie in the alias region.
fn ipc(space: &mut AddressSpace, address: u64, pages: u64) {
    space
        .map_pages(
            address,
            pages,
            MemoryState::Ipc,
            MemoryPermission::USER_READ_WRITE,
        )
        .unwrap();
}

fn map_call(address: u64, pages: u64) -> MapperCall {
    MapperCall::Map {
        address,
        size: pages * PAGE_SIZE,
        perms: Perms::ReadWrite,
    }
}

fn unmap_call(address: u64, pages: u64) -> MapperCall {
    MapperCall::Unmap {
        address,
        size: pages * PAGE_SIZE,
    }
}

/// Everything about an address space the two calls could change, for "nothing changed".
fn state(space: &AddressSpace) -> (Vec<MemoryBlock>, MemoryLimit, u64) {
    (
        space.blocks().copied().collect(),
        *space.limit(),
        space.mapped_physical_memory_size(),
    )
}

const LIMIT_REACHED: CallResult<()> = Err(SvcError::Kernel(KernelError::LimitReached));
const INVALID_CURRENT: CallResult<()> = Err(SvcError::Kernel(KernelError::InvalidCurrentMemory));
const INVALID_REGION: CallResult<()> = Err(SvcError::Kernel(KernelError::InvalidMemoryRegion));

// ----- MapPhysicalMemory ---------------------------------------------------------------------

#[test]
fn mapping_physical_memory_maps_free_pages_read_write_and_counts_them() {
    let (mut space, log, page) = alias_space();
    // A neighbour that starts exactly at the range's end.
    ipc(&mut space, page(6), 2);
    drop(log.take());
    let reserved = space.limit().reserved();
    let used = space.used_user_physical_memory_size();
    let total = space.total_user_physical_memory_size();

    assert_eq!(space.map_physical_memory(page(2), 4 * PAGE_SIZE), Ok(()));
    assert_eq!(log.take(), [map_call(page(2), 4)]);
    assert_eq!(
        rows(&space, page(0), page(8)),
        [free(page(0), 2), physical(page(2), 4), ipc_row(page(6), 2)]
    );
    // The memory is reserved and counted as normal memory (`GetNormalMemorySize`), so in the
    // used memory `GetInfo` 7 and 22 report; the total (6) stays what it was, since what is
    // used and what is free move together.
    let mapped = 4 * PAGE_SIZE;
    assert_eq!(space.mapped_physical_memory_size(), mapped);
    assert_eq!(space.limit().reserved(), reserved + mapped);
    assert_eq!(space.normal_memory_size(), space.heap_size() + mapped);
    assert_eq!(space.used_user_physical_memory_size(), used + mapped);
    assert_eq!(space.total_user_physical_memory_size(), total);
    assert!(space.blocks.check_state());
}

#[test]
fn the_heap_and_physical_memory_both_count_as_normal_memory() {
    let (mut space, _log, page) = alias_space();
    let heap = 0x20_0000;
    space.set_heap_size(heap).unwrap();
    space.map_physical_memory(page(0), 3 * PAGE_SIZE).unwrap();
    assert_eq!(space.normal_memory_size(), heap + 3 * PAGE_SIZE);
    // Shrinking the heap leaves the physical memory counted, and the reverse.
    space.set_heap_size(0).unwrap();
    assert_eq!(space.normal_memory_size(), 3 * PAGE_SIZE);
    space.set_heap_size(heap).unwrap();
    space.unmap_physical_memory(page(0), 3 * PAGE_SIZE).unwrap();
    assert_eq!(space.normal_memory_size(), heap);
}

#[test]
fn a_partly_mapped_range_maps_only_its_free_pages_and_reserves_only_them() {
    let (mut space, log, page) = alias_space();
    // Pages 2-3 are physical memory already and page 5 is IPC memory: both count as mapped,
    // whatever their state (Eden: `info.GetState() != KMemoryState::Free`).
    space.map_physical_memory(page(2), 2 * PAGE_SIZE).unwrap();
    ipc(&mut space, page(5), 1);
    drop(log.take());
    let reserved = space.limit().reserved();
    let mapped = space.mapped_physical_memory_size();

    assert_eq!(space.map_physical_memory(page(0), 8 * PAGE_SIZE), Ok(()));
    assert_eq!(
        log.take(),
        [
            map_call(page(0), 2),
            map_call(page(4), 1),
            map_call(page(6), 2)
        ]
    );
    // Five pages were free.
    assert_eq!(space.limit().reserved(), reserved + 5 * PAGE_SIZE);
    assert_eq!(
        space.mapped_physical_memory_size(),
        mapped + 5 * PAGE_SIZE
    );
    // The new pages have no barrier, so they merge with the physical memory beside them; the
    // IPC page keeps its barrier, and its own state.
    assert_eq!(
        rows(&space, page(0), page(8)),
        [
            physical(page(0), 5),
            ipc_row(page(5), 1),
            physical(page(6), 2)
        ]
    );
    assert!(space.blocks.check_state());
}

#[test]
fn a_range_with_no_free_page_maps_nothing_and_reserves_nothing() {
    let (mut space, log, page) = alias_space();
    space.map_physical_memory(page(0), 2 * PAGE_SIZE).unwrap();
    ipc(&mut space, page(2), 1);
    drop(log.take());
    let before = state(&space);
    // Mapped already, in one state or two.
    for (first, pages) in [(0, 2), (0, 3), (2, 1), (1, 1)] {
        assert_eq!(
            space.map_physical_memory(page(first), pages * PAGE_SIZE),
            Ok(()),
            "{first} {pages}"
        );
    }
    assert_eq!(log.take(), []);
    assert_eq!(state(&space), before);
}

#[test]
fn mapping_at_the_alias_regions_start_sets_no_barrier() {
    // Eden passes `DisableMergeAttribute::Normal` to set when the range starts at the alias
    // region's start, but `UpdateIfMatch` updates every block as no head, so it is not set.
    let (mut space, _log, page) = alias_space();
    space.map_physical_memory(page(0), 2 * PAGE_SIZE).unwrap();
    assert_eq!(
        rows(&space, page(0), page(4)),
        [physical(page(0), 2), free(page(2), 2)]
    );
    assert_eq!(
        space.query_info(page(0)).unwrap().disable_merge(),
        DisableMerge::NONE
    );
}

#[test]
fn the_pool_refuses_a_map_one_page_past_its_free_memory_and_takes_one_exactly_at_it() {
    let (mut space, log, page) = alias_space();
    // One page mapped first: a range over it needs only its free pages.
    space.map_physical_memory(page(0), PAGE_SIZE).unwrap();
    drop(log.take());
    let free = space.limit().free();
    assert!(free.is_multiple_of(PAGE_SIZE));
    // The range, from the mapped page, lies inside the alias region.
    assert!(space.regions().alias.holds(page(0), free + 2 * PAGE_SIZE));
    let before = state(&space);

    // Free pages one page past the free memory: refused, with nothing changed.
    assert_eq!(
        space.map_physical_memory(page(0), free + 2 * PAGE_SIZE),
        LIMIT_REACHED
    );
    assert_eq!(log.take(), []);
    assert_eq!(state(&space), before);

    // Exactly the free memory: taken whole.
    assert_eq!(
        space.map_physical_memory(page(0), free + PAGE_SIZE),
        Ok(())
    );
    assert_eq!(log.take(), [map_call(page(1), free / PAGE_SIZE)]);
    assert_eq!(space.limit().free(), 0);
    assert_eq!(space.mapped_physical_memory_size(), free + PAGE_SIZE);
    // And now one page more is refused.
    let end = page(1) + free;
    assert_eq!(space.map_physical_memory(end, PAGE_SIZE), LIMIT_REACHED);
}

#[test]
fn the_page_table_half_checks_pages_then_the_alias_region() {
    let (mut space, log, page) = alias_space();
    let alias = space.regions().alias;
    let before = state(&space);
    let both = |space: &mut AddressSpace, address: u64, size: u64| {
        (
            space.map_physical_memory(address, size),
            space.unmap_physical_memory(address, size),
        )
    };
    // Not whole pages inside the address space: `InvalidCurrentMemory`, also for a range that
    // is outside the alias region too (the pages are checked first).
    for (address, size) in [
        (page(0) + 1, PAGE_SIZE),
        (page(0), PAGE_SIZE + 1),
        (page(0), 0),
        (alias.start - PAGE_SIZE + 1, PAGE_SIZE),
        (u64::MAX - PAGE_SIZE + 1, PAGE_SIZE),
    ] {
        assert_eq!(
            both(&mut space, address, size),
            (INVALID_CURRENT, INVALID_CURRENT),
            "{address:#x} {size:#x}"
        );
    }
    // Whole pages, but not all inside the alias region: one page below it, one page past it,
    // or outside it altogether (the heap region).
    for (address, size) in [
        (alias.start - PAGE_SIZE, 2 * PAGE_SIZE),
        (alias.end - PAGE_SIZE, 2 * PAGE_SIZE),
        (alias.end, PAGE_SIZE),
        (space.regions().heap.start, PAGE_SIZE),
    ] {
        assert_eq!(
            both(&mut space, address, size),
            (INVALID_REGION, INVALID_REGION),
            "{address:#x} {size:#x}"
        );
    }
    assert_eq!(log.take(), []);
    assert_eq!(state(&space), before);
    // The alias region's first and last pages are inside it.
    for address in [alias.start, alias.end - PAGE_SIZE] {
        assert_eq!(both(&mut space, address, PAGE_SIZE), (Ok(()), Ok(())));
    }
    assert_eq!(state(&space), before);
}

#[test]
fn the_alias_region_check_is_exact_at_both_ends() {
    let (space, _log, _) = alias_space();
    let alias = space.regions().alias;
    let inside = |address, size| space.is_in_alias_region(address, size);
    assert!(inside(alias.start, alias.size()));
    assert!(inside(alias.start, 1));
    assert!(inside(alias.end - 1, 1));
    assert!(!inside(alias.start - 1, 1));
    assert!(!inside(alias.end, 1));
    assert!(!inside(alias.start, alias.size() + 1));
    assert!(!inside(alias.start, 0));
    assert!(!inside(alias.start, u64::MAX));
}

// ----- UnmapPhysicalMemory -------------------------------------------------------------------

#[test]
fn unmapping_frees_only_what_was_mapped_and_gives_it_back() {
    let (mut space, log, page) = alias_space();
    space.map_physical_memory(page(2), 2 * PAGE_SIZE).unwrap();
    space.map_physical_memory(page(6), 2 * PAGE_SIZE).unwrap();
    // A neighbour that starts exactly at the range's end.
    space.map_physical_memory(page(10), 2 * PAGE_SIZE).unwrap();
    drop(log.take());
    let reserved = space.limit().reserved();
    let used = space.used_user_physical_memory_size();

    assert_eq!(space.unmap_physical_memory(page(0), 10 * PAGE_SIZE), Ok(()));
    assert_eq!(
        log.take(),
        [unmap_call(page(2), 2), unmap_call(page(6), 2)]
    );
    assert_eq!(
        rows(&space, page(0), page(12)),
        [free(page(0), 10), physical(page(10), 2)]
    );
    let freed = 4 * PAGE_SIZE;
    assert_eq!(space.limit().reserved(), reserved - freed);
    assert_eq!(space.mapped_physical_memory_size(), 2 * PAGE_SIZE);
    assert_eq!(space.used_user_physical_memory_size(), used - freed);
    assert!(space.blocks.check_state());
}

#[test]
fn unmapping_part_of_a_mapping_splits_it() {
    let (mut space, log, page) = alias_space();
    space.map_physical_memory(page(0), 6 * PAGE_SIZE).unwrap();
    drop(log.take());
    assert_eq!(space.unmap_physical_memory(page(2), 2 * PAGE_SIZE), Ok(()));
    assert_eq!(log.take(), [unmap_call(page(2), 2)]);
    assert_eq!(
        rows(&space, page(0), page(6)),
        [physical(page(0), 2), free(page(2), 2), physical(page(4), 2)]
    );
    assert_eq!(space.mapped_physical_memory_size(), 4 * PAGE_SIZE);
}

#[test]
fn unmapping_a_range_with_nothing_mapped_does_nothing() {
    let (mut space, log, page) = alias_space();
    let before = state(&space);
    assert_eq!(space.unmap_physical_memory(page(0), 4 * PAGE_SIZE), Ok(()));
    assert_eq!(log.take(), []);
    assert_eq!(state(&space), before);
}

#[test]
fn unmapping_is_refused_unless_every_block_is_free_or_plain_normal() {
    // Eden: `R_UNLESS(is_normal || is_free, ResultInvalidCurrentMemory)`, where `is_normal` is
    // the state `Normal` with no attribute at all.
    let (mut space, log, page) = alias_space();
    space.map_physical_memory(page(0), 4 * PAGE_SIZE).unwrap();
    ipc(&mut space, page(6), 1);
    // Page 1 uncached (a settable attribute); page 2 device shared (a counted one).
    let uncached = MemoryAttribute::UNCACHED;
    space
        .set_memory_attribute(page(1), PAGE_SIZE, uncached, uncached)
        .unwrap();
    space.lock_device_share(page(2), PAGE_SIZE).unwrap();
    drop(log.take());
    let before = state(&space);
    // Each refused range holds one block that is neither, and mapped pages besides.
    for (first, pages) in [(0, 2), (2, 2), (3, 4), (6, 1)] {
        assert_eq!(
            space.unmap_physical_memory(page(first), pages * PAGE_SIZE),
            INVALID_CURRENT,
            "{first} {pages}"
        );
    }
    assert_eq!(log.take(), []);
    assert_eq!(state(&space), before);
    // Page 3, reprotected, is still plain `Normal`: the permission is not checked.
    space
        .set_memory_permission(page(3), PAGE_SIZE, SvcPermission::Read)
        .unwrap();
    drop(log.take());
    assert_eq!(space.unmap_physical_memory(page(3), 3 * PAGE_SIZE), Ok(()));
    assert_eq!(log.take(), [unmap_call(page(3), 1)]);
}

#[test]
fn unmapping_from_the_alias_regions_start_clears_a_head_barrier() {
    // Eden clears `Normal` from the barriers when the range's first block is `Normal` and
    // starts both the range and the alias region. `MapPhysicalMemory` never sets that barrier
    // (`mapping_at_the_alias_regions_start_sets_no_barrier`), so the block is made here as a
    // mapping with it, to see the clear: mapped and counted, then remade in the blocks alone,
    // from free, with the barrier at its head.
    let barrier_at_start = || {
        let (mut space, log, page) = alias_space();
        space.map_physical_memory(page(0), 2 * PAGE_SIZE).unwrap();
        for (properties, set) in [(FREE, DisableMerge::NONE), (PHYSICAL, DisableMerge::NORMAL)] {
            space
                .blocks
                .update(
                    page(0),
                    2,
                    properties.state,
                    properties.permission,
                    properties.attribute,
                    set,
                    DisableMerge::NONE,
                )
                .unwrap();
        }
        assert_eq!(
            space.query_info(page(0)).unwrap().disable_merge(),
            DisableMerge::NORMAL
        );
        drop(log.take());
        (space, page)
    };
    // From the alias region's start: the barrier goes, and the freed pages merge with the free
    // memory above.
    let (mut space, page) = barrier_at_start();
    space.unmap_physical_memory(page(0), 2 * PAGE_SIZE).unwrap();
    assert_eq!(space.query_info(page(0)).unwrap().disable_merge(), DisableMerge::NONE);
    assert!(space.query_info(page(0)).unwrap().pages() > 2);
    // From one page in, the first block starts below the range: the barrier stays.
    let (mut space, page) = barrier_at_start();
    space.unmap_physical_memory(page(1), PAGE_SIZE).unwrap();
    assert_eq!(
        rows(&space, page(0), page(1)),
        [(
            page(0),
            1,
            MemoryState::Normal,
            MemoryPermission::USER_READ_WRITE,
            MemoryAttribute::NONE,
            DisableMerge::NORMAL
        )]
    );
    // From the start but over a range whose first block is free: nothing to clear.
    let (mut space, log, page) = alias_space();
    space.map_physical_memory(page(1), PAGE_SIZE).unwrap();
    drop(log.take());
    space.unmap_physical_memory(page(0), 2 * PAGE_SIZE).unwrap();
    assert_eq!(log.take(), [unmap_call(page(1), 1)]);
}

// ----- A translation that refuses partway ----------------------------------------------------

/// Pages 0-1, 3-4 and 6-7 free, with IPC memory at 2 and 5: a map of pages 0-7 is three
/// `Mapper` calls.
fn three_free_runs() -> (AddressSpace, MapperLog, impl Fn(u64) -> u64) {
    let (mut space, log, page) = alias_space();
    ipc(&mut space, page(2), 1);
    ipc(&mut space, page(5), 1);
    drop(log.take());
    (space, log, page)
}

#[test]
fn a_map_refused_partway_unmaps_what_it_mapped_and_releases_the_memory() {
    let (mut space, log, page) = three_free_runs();
    let before = state(&space);
    // The third map is refused; the undo's two unmaps are accepted.
    log.refuse_calls(&[2]);
    assert_eq!(
        space.map_physical_memory(page(0), 8 * PAGE_SIZE),
        Err(SvcError::Internal(format!(
            "the translation refused to map 0x2000 bytes at {:#x}: the test told the mapper to \
             refuse",
            page(6)
        )))
    );
    assert_eq!(
        log.take(),
        [
            map_call(page(0), 2),
            map_call(page(3), 2),
            unmap_call(page(0), 2),
            unmap_call(page(3), 2)
        ]
    );
    assert_eq!(state(&space), before);
}

#[test]
fn a_piece_whose_undo_fails_too_stays_mapped_and_counted() {
    let (mut space, log, page) = three_free_runs();
    let reserved = space.limit().reserved();
    // The third map is refused, and so is the undo of the first.
    log.refuse_calls(&[2, 3]);
    assert_eq!(
        space.map_physical_memory(page(0), 8 * PAGE_SIZE),
        Err(SvcError::Internal(format!(
            "the translation refused to map 0x2000 bytes at {:#x}: the test told the mapper to \
             refuse; undoing the map of 0x2000 bytes at {:#x} failed too, so they stay mapped: \
             the test told the mapper to refuse",
            page(6),
            page(0)
        )))
    );
    assert_eq!(
        log.take(),
        [
            map_call(page(0), 2),
            map_call(page(3), 2),
            unmap_call(page(3), 2)
        ]
    );
    // Pages 0-1 are what the translation still maps: mapped physical memory, reserved.
    assert_eq!(
        rows(&space, page(0), page(8)),
        [
            physical(page(0), 2),
            ipc_row(page(2), 1),
            free(page(3), 2),
            ipc_row(page(5), 1),
            free(page(6), 2)
        ]
    );
    assert_eq!(space.mapped_physical_memory_size(), 2 * PAGE_SIZE);
    assert_eq!(space.limit().reserved(), reserved + 2 * PAGE_SIZE);
    assert!(space.blocks.check_state());
    // And they can be unmapped as any other physical memory.
    assert_eq!(space.unmap_physical_memory(page(0), 2 * PAGE_SIZE), Ok(()));
    assert_eq!(space.limit().reserved(), reserved);
    assert_eq!(space.mapped_physical_memory_size(), 0);
}

#[test]
fn an_unmap_refused_partway_frees_what_it_unmapped() {
    let (mut space, log, page) = alias_space();
    for first in [0, 3, 6] {
        space.map_physical_memory(page(first), 2 * PAGE_SIZE).unwrap();
    }
    drop(log.take());
    let reserved = space.limit().reserved();
    // The second unmap is refused.
    log.refuse_calls(&[1]);
    assert_eq!(
        space.unmap_physical_memory(page(0), 8 * PAGE_SIZE),
        Err(SvcError::Internal(format!(
            "the translation refused to unmap 0x2000 bytes at {:#x}: the test told the mapper \
             to refuse",
            page(3)
        )))
    );
    assert_eq!(log.take(), [unmap_call(page(0), 2)]);
    // Pages 0-1 are gone from the translation, so from the blocks and the count too.
    assert_eq!(
        rows(&space, page(0), page(8)),
        [
            free(page(0), 3),
            physical(page(3), 2),
            free(page(5), 1),
            physical(page(6), 2)
        ]
    );
    assert_eq!(space.mapped_physical_memory_size(), 4 * PAGE_SIZE);
    assert_eq!(space.limit().reserved(), reserved - 2 * PAGE_SIZE);
    assert!(space.blocks.check_state());
}
