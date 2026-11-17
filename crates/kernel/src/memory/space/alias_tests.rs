//! The page-table half of `MapMemory` and `UnmapMemory`: what each does to the blocks, the
//! translation and the alias record, what each refuses and in what order, and how each undoes
//! a refused translation.
//!
//! "Eden" below is `eden/src/core/hle/kernel/k_page_table_base.cpp` @ cf654052,
//! `KPageTableBase::MapMemory` and `UnmapMemory`.

use starquake_memory::Perms;

use super::alias::Aliases;
use super::snapshot::SnapshotError;
use super::*;
use crate::memory::fixture::loaded;
use crate::memory::testing::{MapperCall, MapperLog, RecordingMapper};
use crate::svc::memory::HEAP_SIZE_ALIGNMENT;

/// What `MapMemory` makes of the source (`new_src_perm`).
const LOCKED_SOURCE: MemoryPermission = MemoryPermission::from_bits(
    MemoryPermission::KERNEL_READ.bits() | MemoryPermission::NOT_MAPPED.bits(),
);

fn invalid<T>(error: KernelError) -> CallResult<T> {
    Err(SvcError::Kernel(error))
}

/// The fixture's loaded address space with a heap of [`HEAP_SIZE_ALIGNMENT`] bytes, the log
/// emptied, the address of heap page `n`, and of page `n` of the free top of the stack region
/// (page 0 is 64 pages below the region's end).
struct Rig {
    space: AddressSpace,
    log: MapperLog,
    heap: u64,
    stack: u64,
}

impl Rig {
    fn new() -> Self {
        let (mut space, log, _) = loaded();
        let heap = space.set_heap_size(HEAP_SIZE_ALIGNMENT).unwrap();
        drop(log.take());
        let stack = space.regions().stack.end - 64 * PAGE_SIZE;
        Self {
            space,
            log,
            heap,
            stack,
        }
    }

    fn src(&self, n: u64) -> u64 {
        self.heap + n * PAGE_SIZE
    }

    fn dst(&self, n: u64) -> u64 {
        self.stack + n * PAGE_SIZE
    }

    fn blocks(&self) -> Vec<MemoryBlock> {
        self.space.blocks().copied().collect()
    }

    fn block(&self, address: u64) -> MemoryBlock {
        self.space.query_info(address).unwrap()
    }

    fn runs(&self) -> Vec<(u64, u64, u64)> {
        self.space.aliases.runs()
    }
}

fn pages(n: u64) -> u64 {
    n * PAGE_SIZE
}

// ----- What a passing call does -----------------------------------------------------------------

#[test]
fn map_memory_locks_the_source_and_maps_a_stack_alias_of_its_pages() {
    let mut rig = Rig::new();
    let (src, dst) = (rig.src(1), rig.dst(0));
    assert_eq!(rig.space.map_memory(dst, src, pages(2)), Ok(()));
    // Eden's order: reprotect the source, then map the alias.
    assert_eq!(
        rig.log.take(),
        [
            MapperCall::Protect {
                address: src,
                size: pages(2),
                perms: Perms::None
            },
            MapperCall::MapAlias {
                address: dst,
                source: src,
                size: pages(2),
                perms: Perms::ReadWrite
            },
        ]
    );
    let source = rig.block(src);
    assert_eq!(
        (
            source.address(),
            source.pages(),
            source.state(),
            source.permission(),
            source.attribute(),
            source.disable_merge()
        ),
        (
            src,
            2,
            MemoryState::Normal,
            LOCKED_SOURCE,
            MemoryAttribute::LOCKED,
            DisableMerge::LOCKED
        )
    );
    let alias = rig.block(dst);
    assert_eq!(
        (
            alias.address(),
            alias.pages(),
            alias.state(),
            alias.permission(),
            alias.attribute(),
            alias.disable_merge()
        ),
        (
            dst,
            2,
            MemoryState::Stack,
            MemoryPermission::USER_READ_WRITE,
            MemoryAttribute::NONE,
            DisableMerge::NORMAL
        )
    );
    // The heap page before and after the source are untouched.
    for neighbour in [rig.src(0), rig.src(3)] {
        let block = rig.block(neighbour);
        assert_eq!(block.permission(), MemoryPermission::USER_READ_WRITE);
        assert_eq!(block.attribute(), MemoryAttribute::NONE);
    }
    assert_eq!(rig.block(rig.dst(2)).state(), MemoryState::Free);
    assert_eq!(rig.runs(), [(dst, src, pages(2))]);
}

#[test]
fn unmap_memory_gives_back_the_blocks_as_they_were() {
    let mut rig = Rig::new();
    let before = rig.blocks();
    let (src, dst) = (rig.src(1), rig.dst(0));
    rig.space.map_memory(dst, src, pages(2)).unwrap();
    drop(rig.log.take());
    assert_eq!(rig.space.unmap_memory(dst, src, pages(2)), Ok(()));
    assert_eq!(
        rig.log.take(),
        [
            MapperCall::Unmap {
                address: dst,
                size: pages(2)
            },
            MapperCall::Protect {
                address: src,
                size: pages(2),
                perms: Perms::ReadWrite
            },
        ]
    );
    // The barriers the map set are cleared, so the blocks merge back.
    assert_eq!(rig.blocks(), before);
    assert_eq!(rig.runs(), []);
    // And the source can be aliased again.
    assert_eq!(rig.space.map_memory(dst, src, pages(2)), Ok(()));
}

// ----- What the page table refuses, and in what order --------------------------------------------

#[test]
fn the_page_table_half_checks_pages_then_the_stack_region() {
    let mut rig = Rig::new();
    let before = rig.blocks();
    let (src, dst) = (rig.src(0), rig.dst(0));
    let heap_free = rig.space.regions().heap.start + HEAP_SIZE_ALIGNMENT;
    let end = rig.space.regions().address_space.end;
    for call in [AddressSpace::map_memory, AddressSpace::unmap_memory] {
        // Not whole pages, or not inside the address space: the source, then the destination.
        for (d, s, size) in [
            (dst, src + 1, PAGE_SIZE),
            (dst + 1, src, PAGE_SIZE),
            (dst, src, PAGE_SIZE + 1),
            (dst, end, PAGE_SIZE),
            (end, src, PAGE_SIZE),
        ] {
            assert_eq!(
                call(&mut rig.space, d, s, size),
                invalid(KernelError::InvalidCurrentMemory),
                "{d:#x} {s:#x} {size:#x}"
            );
        }
        // A destination the stack region cannot hold: in the heap region, free.
        assert_eq!(
            call(&mut rig.space, heap_free, src, PAGE_SIZE),
            invalid(KernelError::InvalidMemoryRegion)
        );
        // Both faults at once, a source outside the address space and a destination outside the
        // stack region: the pages are checked first.
        assert_eq!(
            call(&mut rig.space, heap_free, end, PAGE_SIZE),
            invalid(KernelError::InvalidCurrentMemory)
        );
        // The region before the states: a source that is not heap at all (free) and a
        // destination outside the stack region.
        assert_eq!(
            call(&mut rig.space, heap_free, heap_free + PAGE_SIZE, PAGE_SIZE),
            invalid(KernelError::InvalidMemoryRegion)
        );
    }
    assert_eq!(rig.blocks(), before);
    assert_eq!(rig.log.take(), []);
}

#[test]
fn the_destination_may_end_exactly_at_the_stack_regions_end_and_not_a_page_past_it() {
    let mut rig = Rig::new();
    let stack_end = rig.space.regions().stack.end;
    let src = rig.src(0);
    assert_eq!(
        rig.space.map_memory(stack_end - PAGE_SIZE, src, pages(2)),
        invalid(KernelError::InvalidMemoryRegion)
    );
    assert_eq!(
        rig.space.map_memory(stack_end - pages(2), src, pages(2)),
        Ok(())
    );
}

#[test]
fn the_source_must_be_plain_read_write_memory_that_can_be_aliased() {
    let mut rig = Rig::new();
    let dst = rig.dst(0);
    let heap_end = rig.heap + HEAP_SIZE_ALIGNMENT;
    // Free memory (past the heap), the main thread's stack (no `CanAlias`), a heap page made
    // read-only, and a heap page that is uncached.
    let stack_bottom = rig.space.regions().stack.start + GUARD_PAGES * PAGE_SIZE;
    rig.space
        .set_memory_permission(rig.src(4), PAGE_SIZE, SvcPermission::Read)
        .unwrap();
    rig.space
        .set_memory_attribute(
            rig.src(6),
            PAGE_SIZE,
            MemoryAttribute::UNCACHED,
            MemoryAttribute::UNCACHED,
        )
        .unwrap();
    let before = rig.blocks();
    drop(rig.log.take());
    for source in [heap_end, stack_bottom, rig.src(4), rig.src(6)] {
        assert_eq!(
            rig.space.map_memory(dst, source, PAGE_SIZE),
            invalid(KernelError::InvalidCurrentMemory),
            "{source:#x}"
        );
    }
    // A source that runs one page past the heap's end is refused; one that ends exactly there
    // is taken.
    assert_eq!(
        rig.space.map_memory(dst, heap_end - PAGE_SIZE, pages(2)),
        invalid(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(rig.blocks(), before);
    assert_eq!(rig.log.take(), []);
    assert_eq!(
        rig.space.map_memory(dst, heap_end - pages(2), pages(2)),
        Ok(())
    );
    // A source that is aliased already is locked, and cannot be aliased again.
    assert_eq!(
        rig.space
            .map_memory(rig.dst(8), heap_end - PAGE_SIZE, PAGE_SIZE),
        invalid(KernelError::InvalidCurrentMemory)
    );
}

#[test]
fn module_data_can_be_aliased_too() {
    // `CodeData` has `FlagCanAlias` (`FlagsData`).
    let mut rig = Rig::new();
    let data = crate::memory::fixture::DATA;
    assert_eq!(rig.space.map_memory(rig.dst(0), data, PAGE_SIZE), Ok(()));
    assert_eq!(rig.block(data).state(), MemoryState::CodeData);
    assert_eq!(rig.block(data).attribute(), MemoryAttribute::LOCKED);
}

#[test]
fn the_destination_must_be_free_and_a_neighbour_at_its_end_is_untouched() {
    let mut rig = Rig::new();
    // An alias whose first page is exactly where the next one would end.
    rig.space
        .map_memory(rig.dst(2), rig.src(2), PAGE_SIZE)
        .unwrap();
    let neighbour = rig.block(rig.dst(2));
    let before = rig.blocks();
    drop(rig.log.take());
    // Overlapping it by one page is refused, and changes nothing.
    assert_eq!(
        rig.space.map_memory(rig.dst(1), rig.src(0), pages(2)),
        invalid(KernelError::InvalidCurrentMemory)
    );
    // The main thread's stack is not free either.
    let stack_bottom = rig.space.regions().stack.start + GUARD_PAGES * PAGE_SIZE;
    assert_eq!(
        rig.space.map_memory(stack_bottom, rig.src(0), PAGE_SIZE),
        invalid(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(rig.blocks(), before);
    assert_eq!(rig.log.take(), []);
    // Ending exactly at it is taken; the neighbour stays as it was, through the map and the
    // unmap.
    assert_eq!(
        rig.space.map_memory(rig.dst(0), rig.src(0), pages(2)),
        Ok(())
    );
    assert_eq!(rig.block(rig.dst(2)), neighbour);
    assert_eq!(
        rig.space.unmap_memory(rig.dst(0), rig.src(0), pages(2)),
        Ok(())
    );
    assert_eq!(rig.block(rig.dst(2)), neighbour);
    assert_eq!(rig.runs(), [(rig.dst(2), rig.src(2), PAGE_SIZE)]);
}

#[test]
fn unmapping_what_was_never_mapped_or_a_mismatched_pair_gives_the_references_errors() {
    let mut rig = Rig::new();
    rig.space
        .map_memory(rig.dst(0), rig.src(0), PAGE_SIZE)
        .unwrap();
    rig.space
        .map_memory(rig.dst(4), rig.src(4), PAGE_SIZE)
        .unwrap();
    let before = rig.blocks();
    drop(rig.log.take());
    let stack_bottom = rig.space.regions().stack.start + GUARD_PAGES * PAGE_SIZE;
    for (dst, src, error, what) in [
        // Never mapped: a plain heap source fails the source's state check.
        (
            rig.dst(8),
            rig.src(8),
            KernelError::InvalidCurrentMemory,
            "never mapped",
        ),
        // A free destination fails the destination's state check.
        (
            rig.dst(8),
            rig.src(0),
            KernelError::InvalidCurrentMemory,
            "free destination",
        ),
        // Both ranges pass their state checks, but the destination maps another source
        // (`IsValidPageGroup` fails: `ResultInvalidMemoryRegion`).
        (
            rig.dst(4),
            rig.src(0),
            KernelError::InvalidMemoryRegion,
            "mismatched source",
        ),
        (
            rig.dst(0),
            rig.src(4),
            KernelError::InvalidMemoryRegion,
            "mismatched destination",
        ),
        // Stack memory that is no alias: the main thread's stack.
        (
            stack_bottom,
            rig.src(0),
            KernelError::InvalidMemoryRegion,
            "the main thread's stack",
        ),
        // Two faults at once, a plain heap source and a destination that is another source's
        // alias: the state check comes before the page group.
        (
            rig.dst(4),
            rig.src(8),
            KernelError::InvalidCurrentMemory,
            "state before page group",
        ),
    ] {
        assert_eq!(
            rig.space.unmap_memory(dst, src, PAGE_SIZE),
            invalid(error),
            "{what}"
        );
    }
    assert_eq!(rig.blocks(), before);
    assert_eq!(rig.log.take(), []);
    assert_eq!(
        rig.runs(),
        [
            (rig.dst(0), rig.src(0), PAGE_SIZE),
            (rig.dst(4), rig.src(4), PAGE_SIZE)
        ]
    );
}

#[test]
fn unmapping_needs_each_page_to_alias_the_source_page_at_its_offset() {
    // Two aliases made one page at a time, side by side, in order: unmapped together.
    let mut rig = Rig::new();
    rig.space
        .map_memory(rig.dst(0), rig.src(0), PAGE_SIZE)
        .unwrap();
    rig.space
        .map_memory(rig.dst(1), rig.src(1), PAGE_SIZE)
        .unwrap();
    assert_eq!(
        rig.space.unmap_memory(rig.dst(0), rig.src(0), pages(2)),
        Ok(())
    );
    assert_eq!(rig.runs(), []);

    // Side by side but swapped: each page is a source page, not the one at its offset.
    rig.space
        .map_memory(rig.dst(0), rig.src(1), PAGE_SIZE)
        .unwrap();
    rig.space
        .map_memory(rig.dst(1), rig.src(0), PAGE_SIZE)
        .unwrap();
    assert_eq!(
        rig.space.unmap_memory(rig.dst(0), rig.src(0), pages(2)),
        invalid(KernelError::InvalidMemoryRegion)
    );
    // Each on its own is fine.
    assert_eq!(
        rig.space.unmap_memory(rig.dst(1), rig.src(0), PAGE_SIZE),
        Ok(())
    );
    assert_eq!(
        rig.space.unmap_memory(rig.dst(0), rig.src(1), PAGE_SIZE),
        Ok(())
    );

    // Part of an alias: its last page, then its first, each with its own source page.
    rig.space
        .map_memory(rig.dst(0), rig.src(0), pages(3))
        .unwrap();
    assert_eq!(
        rig.space.unmap_memory(rig.dst(2), rig.src(2), PAGE_SIZE),
        Ok(())
    );
    assert_eq!(rig.runs(), [(rig.dst(0), rig.src(0), pages(2))]);
    // The middle page with the wrong source page is refused.
    assert_eq!(
        rig.space.unmap_memory(rig.dst(1), rig.src(0), PAGE_SIZE),
        invalid(KernelError::InvalidMemoryRegion)
    );
    assert_eq!(
        rig.space.unmap_memory(rig.dst(0), rig.src(0), PAGE_SIZE),
        Ok(())
    );
    assert_eq!(rig.runs(), [(rig.dst(1), rig.src(1), PAGE_SIZE)]);
    assert_eq!(
        rig.space.unmap_memory(rig.dst(1), rig.src(1), PAGE_SIZE),
        Ok(())
    );
    assert_eq!(rig.runs(), []);
}

#[test]
fn the_alias_record_matches_page_by_page_and_cuts_runs_at_both_ends() {
    let p = PAGE_SIZE;
    let mut aliases = Aliases::default();
    aliases.insert(10 * p, 100 * p, 4 * p);
    aliases.insert(14 * p, 104 * p, p);
    // Inside one run, across two runs that line up, and at the exact ends.
    assert!(aliases.matches(10 * p, 100 * p, 4 * p));
    assert!(aliases.matches(11 * p, 101 * p, 4 * p));
    assert!(aliases.matches(13 * p, 103 * p, p));
    // One page before the first run, or past the last, or off by a page in the source.
    assert!(!aliases.matches(9 * p, 99 * p, 2 * p));
    assert!(!aliases.matches(14 * p, 104 * p, 2 * p));
    assert!(!aliases.matches(10 * p, 101 * p, p));
    assert!(!aliases.matches(10 * p, 99 * p, p));
    // No bytes is never an alias (Eden: an empty page group is invalid), and a range or a
    // source that would wrap is none either.
    assert!(!aliases.matches(10 * p, 100 * p, 0));
    assert!(!aliases.matches(u64::MAX - p + 1, 100 * p, 2 * p));
    let mut wrapping = Aliases::default();
    wrapping.insert(10 * p, u64::MAX - p + 1, 2 * p);
    assert!(wrapping.matches(10 * p, u64::MAX - p + 1, p));
    assert!(!wrapping.matches(11 * p, 0, p));

    // Cutting the middle of a run leaves both ends, with their source pages.
    aliases.remove(11 * p, 2 * p);
    assert_eq!(
        aliases.runs(),
        [
            (10 * p, 100 * p, p),
            (13 * p, 103 * p, p),
            (14 * p, 104 * p, p)
        ]
    );
    // Cutting across two runs, and nothing at all.
    aliases.remove(13 * p, 2 * p);
    aliases.remove(20 * p, p);
    assert_eq!(aliases.runs(), [(10 * p, 100 * p, p)]);
    assert_eq!(aliases.first(), Some((10 * p, p)));
}

// ----- A refusing translation ---------------------------------------------------------------------

#[test]
fn a_refused_reprotect_of_the_source_changes_nothing() {
    let mut rig = Rig::new();
    let before = rig.blocks();
    rig.log.refuse_calls(&[0]);
    let refused = rig.space.map_memory(rig.dst(0), rig.src(0), PAGE_SIZE);
    assert!(matches!(refused, Err(SvcError::Internal(_))), "{refused:?}");
    assert_eq!(rig.log.take(), []);
    assert_eq!(rig.blocks(), before);
    assert_eq!(rig.runs(), []);
}

#[test]
fn a_refused_alias_reprotects_the_source_back_and_changes_nothing() {
    let mut rig = Rig::new();
    let before = rig.blocks();
    let (src, dst) = (rig.src(0), rig.dst(0));
    rig.log.refuse_calls(&[1]);
    assert_eq!(
        rig.space.map_memory(dst, src, PAGE_SIZE),
        Err(SvcError::Internal(format!(
            "the translation refused to map {PAGE_SIZE:#x} bytes at {dst:#x}: the test told \
             the mapper to refuse"
        )))
    );
    assert_eq!(
        rig.log.take(),
        [
            MapperCall::Protect {
                address: src,
                size: PAGE_SIZE,
                perms: Perms::None
            },
            MapperCall::Protect {
                address: src,
                size: PAGE_SIZE,
                perms: Perms::ReadWrite
            },
        ]
    );
    assert_eq!(rig.blocks(), before);
    assert_eq!(rig.runs(), []);
    // The next attempt succeeds.
    assert_eq!(rig.space.map_memory(dst, src, PAGE_SIZE), Ok(()));
}

#[test]
fn a_refused_undo_of_the_source_leaves_it_without_access_and_unlocked() {
    let mut rig = Rig::new();
    let (src, dst) = (rig.src(1), rig.dst(0));
    rig.log.refuse_calls(&[1, 2]);
    let refused = rig.space.map_memory(dst, src, PAGE_SIZE);
    assert!(
        matches!(&refused, Err(SvcError::Internal(text)) if text.contains("reprotecting the source back failed too")),
        "{refused:?}"
    );
    // The translation holds the source with no access: so do the blocks, with no lock and no
    // alias.
    let source = rig.block(src);
    assert_eq!(
        (
            source.address(),
            source.pages(),
            source.permission(),
            source.attribute()
        ),
        (src, 1, LOCKED_SOURCE, MemoryAttribute::NONE)
    );
    assert_eq!(rig.block(dst).state(), MemoryState::Free);
    assert_eq!(rig.runs(), []);
    // Its neighbours are as they were, and the guest can give it its access back.
    assert_eq!(
        rig.block(rig.src(0)).permission(),
        MemoryPermission::USER_READ_WRITE
    );
    assert_eq!(
        rig.space
            .set_memory_permission(src, PAGE_SIZE, SvcPermission::ReadWrite),
        Ok(())
    );
}

#[test]
fn a_refused_unmap_of_the_alias_changes_nothing() {
    let mut rig = Rig::new();
    let (src, dst) = (rig.src(0), rig.dst(0));
    rig.space.map_memory(dst, src, PAGE_SIZE).unwrap();
    let before = rig.blocks();
    drop(rig.log.take());
    rig.log.refuse_calls(&[0]);
    assert!(rig.space.unmap_memory(dst, src, PAGE_SIZE).is_err());
    assert_eq!(rig.log.take(), []);
    assert_eq!(rig.blocks(), before);
    assert_eq!(rig.runs(), [(dst, src, PAGE_SIZE)]);
}

#[test]
fn a_refused_reprotect_maps_the_alias_again_and_changes_nothing() {
    let mut rig = Rig::new();
    let (src, dst) = (rig.src(0), rig.dst(0));
    rig.space.map_memory(dst, src, PAGE_SIZE).unwrap();
    let before = rig.blocks();
    drop(rig.log.take());
    rig.log.refuse_calls(&[1]);
    assert_eq!(
        rig.space.unmap_memory(dst, src, PAGE_SIZE),
        Err(SvcError::Internal(format!(
            "the translation refused to reprotect {PAGE_SIZE:#x} bytes at {src:#x}: the test \
             told the mapper to refuse"
        )))
    );
    assert_eq!(
        rig.log.take(),
        [
            MapperCall::Unmap {
                address: dst,
                size: PAGE_SIZE
            },
            MapperCall::MapAlias {
                address: dst,
                source: src,
                size: PAGE_SIZE,
                perms: Perms::ReadWrite
            },
        ]
    );
    assert_eq!(rig.blocks(), before);
    assert_eq!(rig.runs(), [(dst, src, PAGE_SIZE)]);
    assert_eq!(rig.space.unmap_memory(dst, src, PAGE_SIZE), Ok(()));
}

#[test]
fn a_refused_remap_leaves_the_destination_free_and_the_source_locked() {
    let mut rig = Rig::new();
    let (src, dst) = (rig.src(0), rig.dst(0));
    rig.space.map_memory(dst, src, pages(2)).unwrap();
    // A neighbouring alias that starts exactly at the end, which must stay.
    rig.space
        .map_memory(rig.dst(2), rig.src(2), PAGE_SIZE)
        .unwrap();
    drop(rig.log.take());
    rig.log.refuse_calls(&[1, 2]);
    let refused = rig.space.unmap_memory(dst, src, pages(2));
    assert!(
        matches!(&refused, Err(SvcError::Internal(text)) if text.contains("mapping the alias again failed too")),
        "{refused:?}"
    );
    assert_eq!(rig.block(dst).state(), MemoryState::Free);
    assert_eq!(rig.block(dst).end(), Some(rig.dst(2)));
    let source = rig.block(src);
    assert_eq!(
        (source.permission(), source.attribute()),
        (LOCKED_SOURCE, MemoryAttribute::LOCKED)
    );
    assert_eq!(rig.runs(), [(rig.dst(2), rig.src(2), PAGE_SIZE)]);
    assert_eq!(rig.block(rig.dst(2)).state(), MemoryState::Stack);
}

// ----- Snapshots ----------------------------------------------------------------------------------

#[test]
fn an_address_space_with_an_alias_is_not_saved() {
    let mut rig = Rig::new();
    rig.space
        .map_memory(rig.dst(4), rig.src(4), PAGE_SIZE)
        .unwrap();
    rig.space
        .map_memory(rig.dst(0), rig.src(0), pages(2))
        .unwrap();
    // The alias with the lowest address is named.
    let refused = rig.space.save();
    assert_eq!(
        refused,
        Err(SnapshotError::Aliased {
            address: rig.dst(0),
            size: pages(2)
        })
    );
    assert_eq!(
        refused.unwrap_err().to_string(),
        format!(
            "the address space has {:#x} bytes at {:#x} aliased by MapMemory, which a snapshot \
             cannot save yet",
            pages(2),
            rig.dst(0)
        )
    );
    // Once the aliases are gone it saves, and what it restores can alias again.
    rig.space
        .unmap_memory(rig.dst(0), rig.src(0), pages(2))
        .unwrap();
    assert!(rig.space.save().is_err());
    rig.space
        .unmap_memory(rig.dst(4), rig.src(4), PAGE_SIZE)
        .unwrap();
    let bytes = rig.space.save().unwrap();
    let (mapper, _log) = RecordingMapper::new();
    let mut restored = AddressSpace::restore(&bytes, Box::new(mapper), |_| None).unwrap();
    assert_eq!(
        restored.map_memory(rig.dst(0), rig.src(0), PAGE_SIZE),
        Ok(())
    );
}

#[test]
fn a_restored_alias_shape_is_not_an_alias() {
    // Blocks shaped like an aliased pair, made without `MapMemory`: heap lent to a transfer
    // memory with no access for its owner (`Normal`, `KernelRead | NotMapped`, `Locked`), and
    // a stack page.
    let mut rig = Rig::new();
    let (src, dst) = (rig.src(0), rig.dst(0));
    rig.space
        .lock_for_transfer_memory(src, PAGE_SIZE, SvcPermission::None)
        .unwrap();
    rig.space
        .map_pages(
            dst,
            1,
            MemoryState::Stack,
            MemoryPermission::USER_READ_WRITE,
        )
        .unwrap();
    let bytes = rig.space.save().unwrap();
    let (mapper, _log) = RecordingMapper::new();
    let mut restored = AddressSpace::restore(&bytes, Box::new(mapper), |_| None).unwrap();
    assert_eq!(
        restored.query_info(src).unwrap().permission(),
        LOCKED_SOURCE
    );
    assert_eq!(
        restored.unmap_memory(dst, src, PAGE_SIZE),
        invalid(KernelError::InvalidMemoryRegion)
    );
}
