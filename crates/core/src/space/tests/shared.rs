//! Shared memory in the address space over flat guest-physical memory: runs of guest RAM handed
//! out whole, mapped at once wherever the kernel maps them, with the permission it asks for,
//! and never handed out again.

use starquake_memory::stage1::VA_SPACE_END;

use super::*;

/// `pages` pages of guest RAM from RAM page `n` on.
fn ram(n: u64, pages: u64) -> BackingRange {
    BackingRange::new(RAM_BASE + n * PAGE_SIZE, pages * PAGE_SIZE).unwrap()
}

/// All of the tests' guest RAM, in bytes.
const ALL_RAM: u64 = RAM_PAGES * PAGE_SIZE;

/// The span of one level-1 descriptor with a 4 KiB granule: 1 GiB (Arm DDI 0487, "Translation
/// table level"; `starquake_memory::stage1` keeps its shifts private). Mappings this far apart
/// each need tables of their own below the root.
const LEVEL_1_SPAN: u64 = 1 << 30;

// ----- Handing out ------------------------------------------------------------------------------

#[test]
fn shared_pages_are_one_run_of_ram_taken_in_turn_with_touched_pages() {
    let space = space();
    space.map(LOW, PAGE_SIZE, Perms::ReadWrite).unwrap();
    let tables = space.tables.table_pages_in_use();
    assert_eq!(space.allocate_shared(2 * PAGE_SIZE), Ok(ram(0, 2)));
    // Handing out maps nothing: no table entry, no table page, no flush.
    assert_eq!(space.tables.table_pages_in_use(), tables);
    assert_eq!(space.tables().take_pending_flushes(), []);
    // A touched page gets the RAM after it, and the next shared memory the RAM after that.
    assert_eq!(space.touch(LOW, Access::Write), Touch::Backed);
    assert_eq!(space.tables.walk(LOW), Ok(page(2, Perms::ReadWrite)));
    assert_eq!(space.allocate_shared(PAGE_SIZE), Ok(ram(3, 1)));
    assert_eq!(space.shadow().next, RAM_BASE + 4 * PAGE_SIZE);
    assert_eq!(
        space.shadow().shared.iter().collect::<Vec<_>>(),
        [
            (&RAM_BASE, &(RAM_BASE + 2 * PAGE_SIZE)),
            (&(RAM_BASE + 3 * PAGE_SIZE), &(RAM_BASE + 4 * PAGE_SIZE))
        ]
    );
}

#[test]
fn shared_pages_read_as_zero_because_no_page_of_ram_is_handed_out_twice() {
    let space = space();
    // A page that held something and was unmapped does not come back as shared memory.
    space.map(LOW, PAGE_SIZE, Perms::ReadWrite).unwrap();
    space.write(LOW, b"used").unwrap();
    space.unmap(LOW, PAGE_SIZE).unwrap();
    let pages = space.allocate_shared(PAGE_SIZE).unwrap();
    assert_eq!(pages, ram(1, 1));
    assert_eq!(ram_bytes::<4>(&space, 0, 0), *b"used");
    assert_eq!(ram_bytes::<4>(&space, 1, 0), [0; 4]);
}

#[test]
fn a_shared_memory_must_be_whole_pages_and_not_empty_and_that_is_checked_before_room() {
    let space = space();
    let before = snapshot(&space);
    assert_eq!(
        space.allocate_shared(0),
        Err(SpaceError::Map(MapError::Empty))
    );
    assert_eq!(
        space.allocate_shared(PAGE_SIZE + 1),
        Err(SpaceError::Map(MapError::MisalignedSize {
            size: PAGE_SIZE + 1
        }))
    );
    // Both faults at once, a misaligned size and more than RAM holds: the size is reported.
    assert_eq!(
        space.allocate_shared(ALL_RAM + 1),
        Err(SpaceError::Map(MapError::MisalignedSize { size: ALL_RAM + 1 }))
    );
    assert_eq!(snapshot(&space), before);
}

#[test]
fn an_allocation_that_exhausts_ram_by_exactly_one_page_is_refused_and_changes_nothing() {
    let space = space();
    let before = snapshot(&space);
    // One page more than all of RAM.
    assert_eq!(
        space.allocate_shared(ALL_RAM + PAGE_SIZE),
        Err(SpaceError::OutOfRam {
            size: ALL_RAM + PAGE_SIZE,
            available: ALL_RAM
        })
    );
    // A size so large that the end of the run would pass the end of the 64-bit space.
    let largest = u64::MAX - (PAGE_SIZE - 1);
    assert_eq!(
        space.allocate_shared(largest),
        Err(SpaceError::OutOfRam {
            size: largest,
            available: ALL_RAM
        })
    );
    assert_eq!(snapshot(&space), before);

    // With one page touched, all of RAM is one page too many, and all but that page fits.
    space.map(LOW, PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(space.touch(LOW, Access::Read), Touch::Backed);
    let touched = snapshot(&space);
    assert_eq!(
        space.allocate_shared(ALL_RAM),
        Err(SpaceError::OutOfRam {
            size: ALL_RAM,
            available: ALL_RAM - PAGE_SIZE
        })
    );
    assert_eq!(snapshot(&space), touched);
    assert_eq!(
        space.allocate_shared(ALL_RAM - PAGE_SIZE),
        Ok(ram(1, RAM_PAGES - 1))
    );
    // RAM is used up: a page more of either kind is refused.
    let none_left = SpaceError::OutOfRam {
        size: PAGE_SIZE,
        available: 0,
    };
    assert_eq!(space.allocate_shared(PAGE_SIZE), Err(none_left));
    space.map(FAR, PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(space.touch(FAR, Access::Read), Touch::Failed(none_left));
}

// ----- Mapping ------------------------------------------------------------------------------------

#[test]
fn a_shared_mapping_backs_every_page_at_once_with_the_permission_asked_for() {
    for perms in [Perms::Read, Perms::ReadWrite] {
        let space = space();
        let pages = space.allocate_shared(3 * PAGE_SIZE).unwrap();
        let next = space.shadow().next;
        assert_eq!(space.map_shared(LOW, pages, perms), Ok(()));
        for index in 0..3 {
            assert_eq!(
                space.tables.walk(LOW + index * PAGE_SIZE),
                Ok(page(index, perms)),
                "{perms:?}"
            );
        }
        assert!(unmapped_at(&space, LOW + 3 * PAGE_SIZE));
        assert!(unmapped_at(&space, LOW - PAGE_SIZE));
        // Every page is backed already, so a fault there is never for want of backing, and no
        // RAM is taken for it; mapping owes no flush.
        for access in [Access::Read, Access::Write, Access::Execute] {
            assert_eq!(space.touch(LOW, access), Touch::Refused, "{perms:?}");
        }
        assert_eq!(space.shadow().next, next);
        assert_eq!(space.tables().take_pending_flushes(), []);
    }
}

#[test]
fn a_shared_memory_of_one_page_is_reached_by_the_host_through_its_pages_and_its_mapping() {
    let space = Arc::new(space());
    let mut mapper = SpaceMapper(Arc::clone(&space));
    let shared = mapper.allocate_shared(PAGE_SIZE).unwrap();
    assert_eq!(shared.backing(), ram(0, 1));
    assert_eq!(
        mapper.map_shared(LOW, shared.backing(), Perms::ReadWrite),
        Ok(())
    );
    // The service writes the last bytes of the page; the guest's mapping reads them.
    shared.write(PAGE_SIZE - 2, &[0xA1, 0xB2]).unwrap();
    let mut seen = [0; 2];
    space.read(LOW + PAGE_SIZE - 2, &mut seen).unwrap();
    assert_eq!(seen, [0xA1, 0xB2]);
    assert_eq!(ram_bytes::<2>(&space, 0, PAGE_SIZE - 2), [0xA1, 0xB2]);
    // A write through the mapping reaches the service.
    space.write(LOW + 5, &[0xC3]).unwrap();
    let mut byte = [0];
    shared.read(5, &mut byte).unwrap();
    assert_eq!(byte, [0xC3]);
    // The pages reach no further than the page.
    assert_eq!(
        shared.read(PAGE_SIZE - 1, &mut seen),
        Err(Fault { address: PAGE_SIZE })
    );
}

#[test]
fn a_shared_mapping_ending_where_a_neighbour_starts_leaves_the_neighbour_alone() {
    let space = space();
    let neighbour = LOW + 2 * PAGE_SIZE;
    space.map(neighbour, PAGE_SIZE, Perms::ReadWrite).unwrap();
    space.write(neighbour, b"next").unwrap();
    let pages = space.allocate_shared(2 * PAGE_SIZE).unwrap();
    let before = snapshot(&space);
    // One page higher and the range overlaps the neighbour: refused at its first page.
    assert_eq!(
        space.map_shared(LOW + PAGE_SIZE, pages, Perms::Read),
        Err(SpaceError::Map(MapError::AlreadyMapped { va: neighbour }))
    );
    assert_eq!(snapshot(&space), before);
    // Ending exactly where the neighbour starts is fine.
    assert_eq!(space.map_shared(LOW, pages, Perms::Read), Ok(()));
    let neighbours_page = page(0, Perms::ReadWrite);
    assert_eq!(space.tables.walk(neighbour), Ok(neighbours_page));
    // Unmapping the shared memory leaves the neighbour's mapping, entry and bytes.
    assert_eq!(space.unmap(LOW, 2 * PAGE_SIZE), Ok(()));
    assert_eq!(space.tables.walk(neighbour), Ok(neighbours_page));
    assert_eq!(
        space.shadow().mapping(neighbour),
        Some((
            neighbour,
            Mapped {
                end: neighbour + PAGE_SIZE,
                perms: Perms::ReadWrite
            }
        ))
    );
    let mut buf = [0; 4];
    space.read(neighbour, &mut buf).unwrap();
    assert_eq!(&buf, b"next");
}

#[test]
fn a_shared_mapping_may_end_at_the_end_of_the_address_space_and_not_a_page_past_it() {
    let space = space();
    let pages = space.allocate_shared(2 * PAGE_SIZE).unwrap();
    let before = snapshot(&space);
    let last = VA_SPACE_END - 2 * PAGE_SIZE;
    assert_eq!(
        space.map_shared(last + PAGE_SIZE, pages, Perms::Read),
        Err(SpaceError::Map(MapError::VirtualRange {
            va: last + PAGE_SIZE,
            size: 2 * PAGE_SIZE
        }))
    );
    assert_eq!(snapshot(&space), before);
    assert_eq!(space.map_shared(last, pages, Perms::Read), Ok(()));
    assert_eq!(
        space.tables.walk(VA_SPACE_END - PAGE_SIZE),
        Ok(page(1, Perms::Read))
    );
}

#[test]
fn pages_that_are_not_one_shared_memory_are_refused_and_change_nothing() {
    let space = space();
    let pages = space.allocate_shared(2 * PAGE_SIZE).unwrap();
    space.map(FAR, PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(space.touch(FAR, Access::Read), Touch::Backed);
    let before = snapshot(&space);
    let not_shared = [
        // A page a touch backed.
        ram(2, 1),
        // Part of the shared memory: its first page, its last, and a run past its end.
        ram(0, 1),
        ram(1, 1),
        ram(0, 3),
        // RAM never handed out, and the table pool.
        ram(5, 1),
        BackingRange::new(POOL_BASE, PAGE_SIZE).unwrap(),
    ];
    for backing in not_shared {
        assert_eq!(
            space.map_shared(LOW, backing, Perms::ReadWrite),
            Err(SpaceError::NotShared {
                base: backing.base(),
                size: backing.size()
            }),
            "{backing:x?}"
        );
        assert_eq!(snapshot(&space), before, "{backing:x?}");
    }
    assert_eq!(space.map_shared(LOW, pages, Perms::ReadWrite), Ok(()));
}

#[test]
fn a_shared_mapping_checks_the_range_then_the_pages_then_what_is_mapped() {
    let space = space();
    let pages = space.allocate_shared(PAGE_SIZE).unwrap();
    space.map(LOW, PAGE_SIZE, Perms::ReadWrite).unwrap();
    let foreign = ram(4, 1);
    let before = snapshot(&space);
    // A misaligned address and pages that are not a shared memory: the address.
    assert_eq!(
        space.map_shared(LOW + 1, foreign, Perms::Read),
        Err(SpaceError::Map(MapError::MisalignedAddress { address: LOW + 1 }))
    );
    // Pages that are not a shared memory, at a place that is taken: the pages.
    assert_eq!(
        space.map_shared(LOW, foreign, Perms::Read),
        Err(SpaceError::NotShared {
            base: foreign.base(),
            size: foreign.size()
        })
    );
    // The shared memory at a place that is taken.
    assert_eq!(
        space.map_shared(LOW, pages, Perms::Read),
        Err(SpaceError::Map(MapError::AlreadyMapped { va: LOW }))
    );
    assert_eq!(snapshot(&space), before);
    assert_eq!(
        SpaceError::NotShared {
            base: foreign.base(),
            size: foreign.size()
        }
        .to_string(),
        format!(
            "the {:#x} bytes of guest RAM at {:#x} are not a shared memory this address space \
             handed out",
            foreign.size(),
            foreign.base()
        )
    );
}

#[test]
fn a_shared_mapping_with_no_room_for_its_tables_is_refused_and_changes_nothing() {
    let space = space();
    let pages = space.allocate_shared(PAGE_SIZE).unwrap();
    // One mapping in each 1 GiB slot needs new tables each time, until the pool runs out.
    let mut slot = 0;
    let refusal = loop {
        let before = snapshot(&space);
        let free = space.tables.pool_pages() - space.tables.table_pages_in_use();
        match space.map_shared(slot * LEVEL_1_SPAN, pages, Perms::Read) {
            Ok(()) => slot += 1,
            Err(error) => {
                assert_eq!(snapshot(&space), before);
                assert!(unmapped_at(&space, slot * LEVEL_1_SPAN));
                // A level-2 and a level-3 table were needed and fewer were left.
                assert!(free < 2, "{free} table pages were free");
                break error;
            }
        }
    };
    assert_eq!(refusal, SpaceError::Map(MapError::OutOfTablePages));
    assert!(slot > 0);
    // The mappings made before the refusal are whole.
    for made in 0..slot {
        assert_eq!(
            space.tables.walk(made * LEVEL_1_SPAN),
            Ok(page(0, Perms::Read))
        );
    }
}

// ----- After mapping ----------------------------------------------------------------------------

#[test]
fn one_shared_memory_mapped_at_two_places_is_the_same_pages() {
    let space = space();
    let pages = space.allocate_shared(2 * PAGE_SIZE).unwrap();
    space.map_shared(LOW, pages, Perms::Read).unwrap();
    space.map_shared(FAR, pages, Perms::ReadWrite).unwrap();
    space.write(FAR + PAGE_SIZE + 8, b"both").unwrap();
    let mut seen = [0; 4];
    space.read(LOW + PAGE_SIZE + 8, &mut seen).unwrap();
    assert_eq!(&seen, b"both");
    assert_eq!(space.tables.walk(LOW + PAGE_SIZE), Ok(page(1, Perms::Read)));
    assert_eq!(
        space.tables.walk(FAR + PAGE_SIZE),
        Ok(page(1, Perms::ReadWrite))
    );
}

#[test]
fn unmapping_a_shared_mapping_keeps_its_pages_and_their_bytes_and_never_hands_them_out_again() {
    let space = Arc::new(space());
    let mut mapper = SpaceMapper(Arc::clone(&space));
    let shared = mapper.allocate_shared(2 * PAGE_SIZE).unwrap();
    mapper
        .map_shared(LOW, shared.backing(), Perms::ReadWrite)
        .unwrap();
    shared.write(PAGE_SIZE + 4, b"kept").unwrap();
    assert_eq!(mapper.unmap(LOW, 2 * PAGE_SIZE), Ok(()));
    // Guest and host lose the mapping, and its table entries are owed a flush.
    assert!(unmapped_at(&space, LOW) && unmapped_at(&space, LOW + PAGE_SIZE));
    let mut buf = [0; 4];
    assert_eq!(
        space.read(LOW + PAGE_SIZE + 4, &mut buf),
        Err(Fault {
            address: LOW + PAGE_SIZE + 4
        })
    );
    assert_eq!(space.touch(LOW, Access::Read), Touch::Refused);
    let flushes = space.tables().take_pending_flushes();
    assert_eq!(flushes.len(), 1);
    assert_eq!((flushes[0].va, flushes[0].size), (LOW, 2 * PAGE_SIZE));
    // The pages keep what they hold, for the service and for a new mapping of them.
    shared.read(PAGE_SIZE + 4, &mut buf).unwrap();
    assert_eq!(&buf, b"kept");
    mapper.map_shared(FAR, shared.backing(), Perms::Read).unwrap();
    space.read(FAR + PAGE_SIZE + 4, &mut buf).unwrap();
    assert_eq!(&buf, b"kept");
    // RAM handed out later is past them, also after every flush has run.
    mapper.map(LOW, PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(space.touch(LOW, Access::Write), Touch::Backed);
    assert_eq!(space.tables.walk(LOW), Ok(page(2, Perms::ReadWrite)));
    assert_eq!(
        mapper.allocate_shared(PAGE_SIZE).map(|pages| pages.backing()),
        Ok(ram(3, 1))
    );
}

#[test]
fn protecting_a_shared_mapping_changes_its_entries_and_keeps_its_pages() {
    let space = space();
    let pages = space.allocate_shared(2 * PAGE_SIZE).unwrap();
    space.map_shared(LOW, pages, Perms::ReadWrite).unwrap();
    space.write(LOW, b"same").unwrap();
    assert_eq!(space.protect(LOW, PAGE_SIZE, Perms::Read), Ok(()));
    assert_eq!(space.tables.walk(LOW), Ok(page(0, Perms::Read)));
    // The page past the range is left as it was.
    assert_eq!(
        space.tables.walk(LOW + PAGE_SIZE),
        Ok(page(1, Perms::ReadWrite))
    );
    let flushes = space.tables().take_pending_flushes();
    assert_eq!(flushes.len(), 1);
    assert_eq!((flushes[0].va, flushes[0].size), (LOW, PAGE_SIZE));
    assert_eq!(ram_bytes::<4>(&space, 0, 0), *b"same");
}

#[test]
fn the_kernels_mapper_reports_a_shared_memory_refusal_in_words() {
    let space = Arc::new(space());
    let mut mapper = SpaceMapper(Arc::clone(&space));
    assert_eq!(
        mapper.allocate_shared(ALL_RAM + PAGE_SIZE).map(|pages| pages.backing()),
        Err(MapperError(format!(
            "mapping {:#x} bytes needs more guest RAM than the {ALL_RAM:#x} bytes left",
            ALL_RAM + PAGE_SIZE
        )))
    );
    assert_eq!(
        mapper.map_shared(LOW, ram(0, 1), Perms::Read),
        Err(MapperError(format!(
            "the {PAGE_SIZE:#x} bytes of guest RAM at {RAM_BASE:#x} are not a shared memory this \
             address space handed out"
        )))
    );
}
