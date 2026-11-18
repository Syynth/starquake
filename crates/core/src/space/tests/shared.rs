//! Shared memory in the address space over flat guest-physical memory: runs of guest RAM handed
//! out whole, mapped at once wherever the kernel maps them, with the permission it asks for,
//! and handed out again only once given back and every flush has run.

use std::sync::atomic::{AtomicBool, Ordering};

use starquake_kernel::memory::testing::RecordingMapper;
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
        Err(SpaceError::Map(MapError::MisalignedSize {
            size: ALL_RAM + 1
        }))
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
    // A size of zero needs no room, and is refused as empty even now.
    assert_eq!(
        space.allocate_shared(0),
        Err(SpaceError::Map(MapError::Empty))
    );
}

// ----- Mapping ------------------------------------------------------------------------------------

#[test]
fn a_shared_mapping_backs_every_page_at_once_with_the_permission_asked_for() {
    for perms in [Perms::Read, Perms::ReadWrite] {
        let space = space();
        let pages = space.allocate_shared(3 * PAGE_SIZE).unwrap();
        let next = space.shadow().next;
        assert_eq!(space.map_shared(LOW, pages, perms), Ok(()));
        // The shadow, which `protect`, `unmap` and `touch` read, records the permission too.
        assert_eq!(
            space.shadow().mapping(LOW),
            Some((
                LOW,
                Mapped {
                    end: LOW + 3 * PAGE_SIZE,
                    perms
                }
            )),
            "{perms:?}"
        );
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

/// The shadow's check alone refuses an overlap on the last page: the neighbour there is
/// untouched, so it has no table entry for the tables to refuse.
#[test]
fn a_shared_mapping_whose_last_page_is_an_untouched_neighbour_is_refused_there() {
    let space = space();
    let neighbour = LOW + PAGE_SIZE;
    space.map(neighbour, PAGE_SIZE, Perms::ReadWrite).unwrap();
    let pages = space.allocate_shared(2 * PAGE_SIZE).unwrap();
    let before = snapshot(&space);
    assert_eq!(
        space.map_shared(LOW, pages, Perms::Read),
        Err(SpaceError::Map(MapError::AlreadyMapped { va: neighbour }))
    );
    assert_eq!(snapshot(&space), before);
    assert!(unmapped_at(&space, LOW));
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
        Err(SpaceError::Map(MapError::MisalignedAddress {
            address: LOW + 1
        }))
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

/// A touch or a host write that the tables refuse takes no RAM: the address space moves past
/// a page only once its table entry is made.
#[test]
fn a_page_the_tables_have_no_room_for_takes_no_ram_from_a_touch_or_a_host_write() {
    let space = space();
    let pages = space.allocate_shared(PAGE_SIZE).unwrap();
    // One mapping in each 1 GiB slot needs new tables each time, until the pool runs out.
    let mut slot = 0;
    while space
        .map_shared(slot * LEVEL_1_SPAN, pages, Perms::Read)
        .is_ok()
    {
        slot += 1;
    }
    let fresh = slot * LEVEL_1_SPAN;
    space.map(fresh, PAGE_SIZE, Perms::ReadWrite).unwrap();
    let before = snapshot(&space);
    assert_eq!(
        space.touch(fresh, Access::Write),
        Touch::Failed(SpaceError::Map(MapError::OutOfTablePages))
    );
    assert_eq!(snapshot(&space), before);
    assert_eq!(space.write(fresh, b"x"), Err(Fault { address: fresh }));
    assert_eq!(snapshot(&space), before);
    assert!(unmapped_at(&space, fresh));
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
    mapper
        .map_shared(FAR, shared.backing(), Perms::Read)
        .unwrap();
    space.read(FAR + PAGE_SIZE + 4, &mut buf).unwrap();
    assert_eq!(&buf, b"kept");
    // RAM handed out later is past them, also after every flush has run.
    mapper.map(LOW, PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(space.touch(LOW, Access::Write), Touch::Backed);
    assert_eq!(space.tables.walk(LOW), Ok(page(2, Perms::ReadWrite)));
    assert_eq!(
        mapper
            .allocate_shared(PAGE_SIZE)
            .map(|pages| pages.backing()),
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
        mapper
            .allocate_shared(ALL_RAM + PAGE_SIZE)
            .map(|pages| pages.backing()),
        Err(MapperError::OutOfMemory(format!(
            "mapping {:#x} bytes needs more guest RAM than the {ALL_RAM:#x} bytes left",
            ALL_RAM + PAGE_SIZE
        )))
    );
    assert_eq!(
        mapper.map_shared(LOW, ram(0, 1), Perms::Read),
        Err(MapperError::Refused(format!(
            "the {PAGE_SIZE:#x} bytes of guest RAM at {RAM_BASE:#x} are not a shared memory this \
             address space handed out"
        )))
    );
    // Giving back: pages that are not a run handed out, and pages still mapped.
    assert_eq!(
        mapper.release_shared(ram(0, 1)),
        Err(MapperError::Refused(format!(
            "the {PAGE_SIZE:#x} bytes of guest RAM at {RAM_BASE:#x} are not a shared memory this \
             address space handed out"
        )))
    );
    let shared = mapper.allocate_shared(PAGE_SIZE).unwrap();
    mapper
        .map_shared(LOW, shared.backing(), Perms::Read)
        .unwrap();
    assert_eq!(
        mapper.release_shared(shared.backing()),
        Err(MapperError::Refused(format!(
            "the {PAGE_SIZE:#x} bytes of guest RAM at {RAM_BASE:#x} are still mapped, at \
             {LOW:#x} among other places"
        )))
    );
}

// ----- Giving back ------------------------------------------------------------------------------

#[test]
fn given_back_pages_wait_for_every_flush_and_come_back_zeroed() {
    let space = Arc::new(space());
    let mut mapper = SpaceMapper(Arc::clone(&space));
    let shared = mapper.allocate_shared(2 * PAGE_SIZE).unwrap();
    assert_eq!(shared.backing(), ram(0, 2));
    mapper
        .map_shared(LOW, shared.backing(), Perms::ReadWrite)
        .unwrap();
    space.write(LOW + PAGE_SIZE + 8, b"held").unwrap();
    mapper.unmap(LOW, 2 * PAGE_SIZE).unwrap();
    assert!(space.tables().has_pending_flushes());
    assert_eq!(mapper.release_shared(shared.backing()), Ok(()));
    // While the unmapping's flush is owed, a vCPU may still reach the pages: a shared memory
    // of their size comes from RAM never handed out, and they keep what they held.
    assert_eq!(space.allocate_shared(2 * PAGE_SIZE), Ok(ram(2, 2)));
    assert_eq!(ram_bytes::<4>(&space, 1, 8), *b"held");
    // Once the flush is handed out, they are handed out again, zeroed.
    assert_eq!(space.tables().take_pending_flushes().len(), 1);
    assert_eq!(space.allocate_shared(2 * PAGE_SIZE), Ok(ram(0, 2)));
    assert_eq!(ram_bytes::<4>(&space, 1, 8), [0; 4]);
    // Reuse took nothing from the RAM never handed out.
    assert_eq!(space.shadow().next, RAM_BASE + 4 * PAGE_SIZE);
    assert_eq!(
        space.shadow().shared.iter().collect::<Vec<_>>(),
        [
            (&RAM_BASE, &(RAM_BASE + 2 * PAGE_SIZE)),
            (&(RAM_BASE + 2 * PAGE_SIZE), &(RAM_BASE + 4 * PAGE_SIZE))
        ]
    );
}

#[test]
fn a_run_given_back_with_no_flush_owed_is_reusable_at_the_next_allocation() {
    let space = space();
    let pages = space.allocate_shared(PAGE_SIZE).unwrap();
    // Never mapped, so no flush is owed for it, and none for anything else.
    assert!(!space.tables().has_pending_flushes());
    assert_eq!(space.release_shared(pages), Ok(()));
    assert_eq!(space.allocate_shared(PAGE_SIZE), Ok(ram(0, 1)));
}

#[test]
fn given_back_runs_are_reused_first_in_address_order_and_split() {
    let space = space();
    let three = space.allocate_shared(3 * PAGE_SIZE).unwrap();
    let kept = space.allocate_shared(PAGE_SIZE).unwrap();
    let one = space.allocate_shared(PAGE_SIZE).unwrap();
    assert_eq!((three, kept, one), (ram(0, 3), ram(3, 1), ram(4, 1)));
    // Given back highest first: reuse goes by address, not by the order they came back.
    space.release_shared(one).unwrap();
    space.release_shared(three).unwrap();
    // One page: the first run in address order that holds it (a best fit would take the run
    // of exactly one page), whose rest stays reusable.
    assert_eq!(space.allocate_shared(PAGE_SIZE), Ok(ram(0, 1)));
    assert_eq!(
        space.shadow().reusable.iter().collect::<Vec<_>>(),
        [
            (&(RAM_BASE + PAGE_SIZE), &(RAM_BASE + 3 * PAGE_SIZE)),
            (&(RAM_BASE + 4 * PAGE_SIZE), &(RAM_BASE + 5 * PAGE_SIZE))
        ]
    );
    // Two pages: the rest, which holds exactly two.
    assert_eq!(space.allocate_shared(2 * PAGE_SIZE), Ok(ram(1, 2)));
    // Two more: no run holds them, so they come from RAM never handed out.
    assert_eq!(space.allocate_shared(2 * PAGE_SIZE), Ok(ram(5, 2)));
    assert_eq!(space.allocate_shared(PAGE_SIZE), Ok(ram(4, 1)));
    assert_eq!(space.shadow().reusable.len(), 0);
}

#[test]
fn running_out_counts_the_largest_reusable_run_as_available() {
    let space = space();
    let two = space.allocate_shared(2 * PAGE_SIZE).unwrap();
    space.allocate_shared(ALL_RAM - 2 * PAGE_SIZE).unwrap();
    space.release_shared(two).unwrap();
    // RAM never handed out is used up; the run given back holds two pages.
    let before = snapshot(&space);
    assert_eq!(
        space.allocate_shared(3 * PAGE_SIZE),
        Err(SpaceError::OutOfRam {
            size: 3 * PAGE_SIZE,
            available: 2 * PAGE_SIZE
        })
    );
    // The refusal made the run reusable (no flush is owed), and handed nothing out.
    let promoted = snapshot(&space);
    assert_eq!((promoted.0, &promoted.3), (before.0, &before.3));
    assert_eq!(promoted.4, []);
    assert_eq!(promoted.5, [(RAM_BASE, RAM_BASE + 2 * PAGE_SIZE)]);
    // Exactly the run fits, and then nothing is left.
    assert_eq!(space.allocate_shared(2 * PAGE_SIZE), Ok(ram(0, 2)));
    assert_eq!(
        space.allocate_shared(PAGE_SIZE),
        Err(SpaceError::OutOfRam {
            size: PAGE_SIZE,
            available: 0
        })
    );
    // The size is still checked before the room, and a reusable run does not change that.
    space.release_shared(ram(0, 2)).unwrap();
    assert_eq!(
        space.allocate_shared(PAGE_SIZE + 1),
        Err(SpaceError::Map(MapError::MisalignedSize {
            size: PAGE_SIZE + 1
        }))
    );
    assert_eq!(
        space.allocate_shared(0),
        Err(SpaceError::Map(MapError::Empty))
    );
}

#[test]
fn a_release_is_refused_unless_the_pages_are_a_run_handed_out_and_unmapped() {
    let space = space();
    let pages = space.allocate_shared(2 * PAGE_SIZE).unwrap();
    // A neighbour whose pages start exactly where these end, mapped throughout.
    let neighbour = space.allocate_shared(PAGE_SIZE).unwrap();
    assert_eq!(neighbour.base(), pages.end());
    space.map_shared(FAR, pages, Perms::Read).unwrap();
    space.map_shared(LOW, pages, Perms::ReadWrite).unwrap();
    space
        .map_shared(LOW + 2 * PAGE_SIZE, neighbour, Perms::ReadWrite)
        .unwrap();
    let before = snapshot(&space);
    let not_shared = |range: BackingRange| {
        Err(SpaceError::NotShared {
            base: range.base(),
            size: range.size(),
        })
    };
    // Part of the run, and a run never handed out: not one run. The part is mapped too, and
    // that is checked second.
    assert_eq!(space.release_shared(ram(0, 1)), not_shared(ram(0, 1)));
    assert_eq!(space.release_shared(ram(1, 1)), not_shared(ram(1, 1)));
    assert_eq!(space.release_shared(ram(6, 1)), not_shared(ram(6, 1)));
    // The run, mapped at two places: the lowest is named.
    let still_mapped = |va| {
        Err(SpaceError::StillMapped {
            base: RAM_BASE,
            size: 2 * PAGE_SIZE,
            va,
        })
    };
    assert_eq!(space.release_shared(pages), still_mapped(LOW));
    assert_eq!(snapshot(&space), before);
    space.unmap(LOW, 2 * PAGE_SIZE).unwrap();
    assert_eq!(space.release_shared(pages), still_mapped(FAR));
    // Unmapped everywhere: given back, though the neighbour is still mapped.
    space.unmap(FAR, 2 * PAGE_SIZE).unwrap();
    assert_eq!(space.release_shared(pages), Ok(()));
    assert_eq!(space.shadow().released, [pages]);
    // A second time: no longer a run handed out.
    assert_eq!(space.release_shared(pages), not_shared(pages));
    assert_eq!(space.shadow().released, [pages]);
}

/// Guest-physical memory whose RAM can be told to refuse writes: the zeroing write's failure.
#[derive(Debug)]
struct RefusingRam {
    inner: FlatMemory,
    refuse: AtomicBool,
}

impl GuestMemory for RefusingRam {
    fn read(&self, pa: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.inner.read(pa, buf)
    }

    fn write(&self, pa: u64, data: &[u8]) -> Result<(), Fault> {
        if pa >= RAM_BASE && self.refuse.load(Ordering::SeqCst) {
            return Err(Fault { address: pa });
        }
        self.inner.write(pa, data)
    }
}

#[test]
fn a_run_whose_zeroing_fails_is_not_handed_out_and_stays_reusable() {
    let pages = usize::try_from((POOL_PAGES + RAM_PAGES) * PAGE_SIZE).unwrap();
    let memory = RefusingRam {
        inner: FlatMemory::new(POOL_BASE, pages),
        refuse: AtomicBool::new(false),
    };
    let pool = BackingRange::new(POOL_BASE, POOL_PAGES * PAGE_SIZE).unwrap();
    let space = Arc::new(AddressSpace::new(memory, ram(0, RAM_PAGES), pool).unwrap());
    let mut mapper = SpaceMapper(Arc::clone(&space));
    let shared = mapper.allocate_shared(2 * PAGE_SIZE).unwrap();
    mapper.release_shared(shared.backing()).unwrap();
    // Made reusable by a first allocation, which the run cannot hold.
    assert_eq!(space.allocate_shared(3 * PAGE_SIZE), Ok(ram(2, 3)));
    let before = snapshot(&space);
    space.tables.memory().refuse.store(true, Ordering::SeqCst);
    let fault = Fault { address: RAM_BASE };
    assert_eq!(
        space.allocate_shared(2 * PAGE_SIZE),
        Err(SpaceError::Zeroing(fault))
    );
    assert_eq!(snapshot(&space), before);
    // The kernel hears of it as a refusal, in words.
    assert_eq!(
        mapper
            .allocate_shared(PAGE_SIZE)
            .map(|pages| pages.backing()),
        Err(MapperError::Refused(format!(
            "guest RAM refused the write that zeroes pages handed out again: {fault}"
        )))
    );
    assert_eq!(snapshot(&space), before);
    space.tables.memory().refuse.store(false, Ordering::SeqCst);
    assert_eq!(space.allocate_shared(2 * PAGE_SIZE), Ok(ram(0, 2)));
}

/// What each step of [`give_back_scenario`] came to: `Ok`, or which kind of refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Came {
    Ok,
    Refused,
    OutOfMemory,
}

fn came<T>(result: &Result<T, MapperError>) -> Came {
    match result {
        Ok(_) => Came::Ok,
        Err(MapperError::Refused(_)) => Came::Refused,
        Err(MapperError::OutOfMemory(_)) => Came::OutOfMemory,
    }
}

/// One shared memory made, mapped, given back too early, unmapped, given back twice, and a
/// shared memory no translation has room for; and the pages of one of its size made after.
fn give_back_scenario(mapper: &mut dyn Mapper) -> (Vec<Came>, BackingRange, BackingRange) {
    let pages = mapper.allocate_shared(PAGE_SIZE).unwrap().backing();
    let larger = BackingRange::new(pages.base(), 2 * PAGE_SIZE).unwrap();
    let steps = vec![
        came(&mapper.map_shared(LOW, pages, Perms::ReadWrite)),
        came(&mapper.release_shared(pages)),
        came(&mapper.release_shared(larger)),
        came(&mapper.unmap(LOW, PAGE_SIZE)),
        came(&mapper.release_shared(pages)),
        came(&mapper.release_shared(pages)),
        came(&mapper.allocate_shared(u64::MAX - (PAGE_SIZE - 1))),
    ];
    (
        steps,
        pages,
        mapper.allocate_shared(PAGE_SIZE).unwrap().backing(),
    )
}

#[test]
fn the_recording_mapper_gives_back_as_this_one_does_but_never_hands_out_again() {
    let space = Arc::new(space());
    let mut real = SpaceMapper(Arc::clone(&space));
    // This one: a flush is owed for the unmapping until the tables hand it out, so the pages
    // made after the scenario come from RAM never handed out; once it has run, they come back.
    let (real_steps, real_pages, real_after) = give_back_scenario(&mut real);
    let (mut recording, _log) = RecordingMapper::new();
    let (recorded_steps, recorded_pages, recorded_after) = give_back_scenario(&mut recording);
    let expected = [
        Came::Ok,
        Came::Refused,
        Came::Refused,
        Came::Ok,
        Came::Ok,
        Came::Refused,
        Came::OutOfMemory,
    ];
    assert_eq!(real_steps, expected);
    assert_eq!(recorded_steps, expected);
    assert_ne!(real_after, real_pages);
    space.tables().take_pending_flushes();
    assert_eq!(
        real.allocate_shared(PAGE_SIZE).map(|pages| pages.backing()),
        Ok(real_pages)
    );
    // The difference the recording mapper lists: it never hands given-back pages out again.
    assert_ne!(recorded_after, recorded_pages);
    assert_eq!(
        recording
            .allocate_shared(PAGE_SIZE)
            .map(|pages| pages.backing().base()),
        Ok(recorded_after.end())
    );
}
