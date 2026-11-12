//! The address space over flat guest-physical memory, checked with the software table walker:
//! what guest code would see, and what host code reads and writes.

use starquake_memory::FlatMemory;
use starquake_memory::stage1::{FlushRange, Resolved, WalkError};

use super::*;

/// Guest-physical layout of the tests: a table pool, then RAM, in one flat memory. Nothing
/// here is a virtual address the tests map.
const POOL_BASE: u64 = 0x4_0000_0000;
const POOL_PAGES: u64 = 16;
const RAM_BASE: u64 = POOL_BASE + POOL_PAGES * PAGE_SIZE;
const RAM_PAGES: u64 = 8;

/// Virtual addresses: two neighbouring pages and one far away.
const LOW: u64 = 0x8000_0000;
const FAR: u64 = 0x10_0000_0000;

fn space() -> AddressSpace<FlatMemory> {
    let pages = usize::try_from((POOL_PAGES + RAM_PAGES) * PAGE_SIZE).unwrap();
    let physical = FlatMemory::new(POOL_BASE, pages);
    let pool = BackingRange::new(POOL_BASE, POOL_PAGES * PAGE_SIZE).unwrap();
    let ram = BackingRange::new(RAM_BASE, RAM_PAGES * PAGE_SIZE).unwrap();
    AddressSpace::new(physical, ram, pool).unwrap()
}

/// What guest RAM page `n` holds at `offset`, read behind the address space's back.
fn ram_bytes<const N: usize>(space: &AddressSpace<FlatMemory>, n: u64, offset: u64) -> [u8; N] {
    let mut bytes = [0; N];
    space
        .tables
        .memory()
        .read(RAM_BASE + n * PAGE_SIZE + offset, &mut bytes)
        .unwrap();
    bytes
}

/// What the table walker (and so guest code) finds at a backed page: RAM page `n` with
/// `perms`.
fn page(n: u64, perms: Perms) -> Resolved {
    Resolved {
        pa: RAM_BASE + n * PAGE_SIZE,
        perms,
        level: 3,
    }
}

fn unmapped_at(space: &AddressSpace<FlatMemory>, va: u64) -> bool {
    matches!(space.tables.walk(va), Err(WalkError::Unmapped { .. }))
}

/// What [`snapshot`] takes: the next page of RAM, the mappings, the backed pages, and how
/// many table pages are in use.
type Snapshot = (u64, Vec<(u64, Mapped)>, Vec<(u64, u64)>, u64);

/// Everything the tests can see of an address space, for "nothing changed".
fn snapshot(space: &AddressSpace<FlatMemory>) -> Snapshot {
    let shadow = space.shadow();
    (
        shadow.next,
        shadow.mapped.iter().map(|(k, v)| (*k, *v)).collect(),
        shadow.backed.iter().map(|(k, v)| (*k, *v)).collect(),
        space.tables.table_pages_in_use(),
    )
}

// ----- The walk over an access -------------------------------------------------------------------

#[test]
fn a_walk_hands_out_each_piece_with_its_offset_and_length() {
    // Three places of 4, 4 and 100 bytes at addresses 10, 14 and 18; an access of 11 bytes
    // from address 12 takes two bytes of the first, the second whole, and five of the third.
    let resolve = |at: u64| match at {
        10..14 => Some(('a', 14 - at)),
        14..18 => Some(('b', 18 - at)),
        18..118 => Some(('c', 118 - at)),
        _ => None,
    };
    let mut seen = Vec::new();
    let walked = pieces(12, 11, resolve, |place, done, len| {
        seen.push((place, done, len));
        Ok(())
    });
    assert_eq!(walked, Ok(()));
    assert_eq!(seen, [('a', 0, 2), ('b', 2, 4), ('c', 6, 5)]);
    // An access of no bytes asks nothing.
    assert_eq!(
        pieces(0, 0, |_| None::<((), u64)>, |(), _, _| Ok(())),
        Ok(())
    );
    // It stops at the first address that is not mapped, and at the piece that fails.
    assert_eq!(
        pieces(116, 4, resolve, |_, _, _| Ok(())),
        Err(Fault { address: 118 })
    );
    assert_eq!(
        pieces(12, 11, resolve, |place, _, _| {
            if place == 'b' {
                Err(Fault { address: 0 })
            } else {
                Ok(())
            }
        }),
        Err(Fault { address: 14 })
    );
    // An address that overflows is not mapped.
    assert_eq!(
        pieces(u64::MAX, 2, |_| Some(((), 1)), |(), _, _| Ok(())),
        Err(Fault { address: u64::MAX })
    );
}

/// The fix for the walk that never ended: a resolver that says "mapped, zero bytes here" made
/// the old loop ask about the same address for ever (`done += 0`). It is a fault there now.
#[test]
fn a_piece_of_no_bytes_is_a_fault_not_an_endless_walk() {
    let calls = std::cell::Cell::new(0_u32);
    let stuck = |at: u64| {
        calls.set(calls.get().saturating_add(1));
        // Mapped up to address 4, with nothing at address 4 and no end to it.
        Some(((), 4_u64.saturating_sub(at)))
    };
    let mut delivered = Vec::new();
    let walked = pieces(2, 8, stuck, |(), done, len| {
        // Bounded, so that the walk this test is about cannot fill the host's memory.
        if delivered.len() < 4 {
            delivered.push((done, len));
        }
        Ok(())
    });
    assert_eq!(walked, Err(Fault { address: 4 }));
    // The two bytes before it were handed out, and the resolver was asked exactly twice.
    assert_eq!(delivered, [(0, 2)]);
    assert_eq!(calls.get(), 2);
}

// ----- Mapping and backing -----------------------------------------------------------------------

#[test]
fn a_mapping_has_no_table_entry_and_no_ram_until_it_is_touched() {
    let space = space();
    space.map(LOW, 2 * PAGE_SIZE, Perms::ReadWrite).unwrap();
    space.map(FAR, PAGE_SIZE, Perms::Read).unwrap();
    let tables_before = space.tables.table_pages_in_use();
    assert!(unmapped_at(&space, LOW) && unmapped_at(&space, FAR));
    assert_eq!(space.shadow().next, RAM_BASE);

    // Touched out of address order: each page gets the next page of RAM, in the order of
    // the touches, with its mapping's permissions.
    assert_eq!(
        space.touch(LOW + PAGE_SIZE + 0x123, Access::Write),
        Touch::Backed
    );
    assert_eq!(space.touch(FAR + 0xfff, Access::Read), Touch::Backed);
    assert_eq!(space.touch(LOW, Access::Read), Touch::Backed);
    let walked = |va: u64| space.tables.walk(va);
    assert_eq!(walked(LOW + PAGE_SIZE), Ok(page(0, Perms::ReadWrite)));
    assert_eq!(walked(FAR), Ok(page(1, Perms::Read)));
    assert_eq!(walked(LOW), Ok(page(2, Perms::ReadWrite)));
    assert!(space.tables.table_pages_in_use() > tables_before);
    // The page between the neighbours and the far region is not mapped.
    assert_eq!(
        walked(LOW + 2 * PAGE_SIZE),
        Err(WalkError::Unmapped {
            va: LOW + 2 * PAGE_SIZE,
            level: 3
        })
    );
    assert_eq!(space.root().pa(), POOL_BASE);
    // Mapping needs no flush.
    assert_eq!(space.tables().take_pending_flushes(), []);
}

#[test]
fn a_touch_backs_a_page_only_when_backing_is_all_the_access_lacks() {
    let space = space();
    space.map(LOW, PAGE_SIZE, Perms::Read).unwrap();
    space
        .map(LOW + PAGE_SIZE, PAGE_SIZE, Perms::ReadExecute)
        .unwrap();
    space.map(FAR, PAGE_SIZE, Perms::None).unwrap();
    let before = snapshot(&space);
    let refused = [
        // An access the mapping does not allow.
        (LOW, Access::Write),
        (LOW, Access::Execute),
        (LOW + PAGE_SIZE, Access::Write),
        (FAR, Access::Read),
        (FAR, Access::Write),
        (FAR, Access::Execute),
        // Nothing mapped: just below, just above, far away.
        (LOW - 1, Access::Read),
        (LOW + 2 * PAGE_SIZE, Access::Read),
        (0, Access::Read),
        (u64::MAX, Access::Write),
    ];
    for (address, access) in refused {
        assert_eq!(
            space.touch(address, access),
            Touch::Refused,
            "{address:#x} {access:?}"
        );
    }
    assert_eq!(snapshot(&space), before);
    // Allowed accesses back the page, at the first and the last byte of it.
    assert_eq!(space.touch(LOW, Access::Read), Touch::Backed);
    assert_eq!(
        space.touch(LOW + 2 * PAGE_SIZE - 1, Access::Execute),
        Touch::Backed
    );
    // A page that is backed already faulted for another reason: nothing to do.
    assert_eq!(space.touch(LOW, Access::Read), Touch::Refused);
    assert_eq!(space.shadow().next, RAM_BASE + 2 * PAGE_SIZE);
}

#[test]
fn ram_runs_out_exactly_when_it_is_used_up() {
    let space = space();
    space
        .map(FAR, (RAM_PAGES + 2) * PAGE_SIZE, Perms::ReadWrite)
        .unwrap();
    // Exactly all of RAM can be touched.
    for index in 0..RAM_PAGES {
        assert_eq!(
            space.touch(FAR + index * PAGE_SIZE, Access::Read),
            Touch::Backed
        );
    }
    let out_of_ram = SpaceError::OutOfRam {
        size: PAGE_SIZE,
        available: 0,
    };
    let before = snapshot(&space);
    assert_eq!(
        space.touch(FAR + RAM_PAGES * PAGE_SIZE, Access::Read),
        Touch::Failed(out_of_ram)
    );
    // A host write that needs a page of RAM faults at that page and writes nothing, also
    // into the page before it, which is backed.
    let across = FAR + RAM_PAGES * PAGE_SIZE - 2;
    assert_eq!(
        space.write(across, b"1234"),
        Err(Fault {
            address: FAR + RAM_PAGES * PAGE_SIZE
        })
    );
    assert_eq!(ram_bytes::<2>(&space, RAM_PAGES - 1, PAGE_SIZE - 2), [0, 0]);
    assert_eq!(snapshot(&space), before);
    // Reading the unbacked pages needs no RAM.
    let mut buf = [0xee; 4];
    assert_eq!(space.read(across, &mut buf), Ok(()));
    assert_eq!(buf, [0; 4]);
    assert_eq!(
        out_of_ram.to_string(),
        "mapping 0x1000 bytes needs more guest RAM than the 0x0 bytes left"
    );
}

#[test]
fn a_range_that_is_not_pages_of_the_address_space_is_refused_and_changes_nothing() {
    let space = space();
    space.map(LOW, PAGE_SIZE, Perms::Read).unwrap();
    let before = snapshot(&space);
    let top = VA_SPACE_END;
    let refused = [
        (
            FAR + 1,
            PAGE_SIZE,
            MapError::MisalignedAddress { address: FAR + 1 },
        ),
        (
            FAR,
            PAGE_SIZE + 1,
            MapError::MisalignedSize {
                size: PAGE_SIZE + 1,
            },
        ),
        (FAR, 0, MapError::Empty),
        // Ending one page above what the tables translate, starting there, and wrapping.
        (
            top - PAGE_SIZE,
            2 * PAGE_SIZE,
            MapError::VirtualRange {
                va: top - PAGE_SIZE,
                size: 2 * PAGE_SIZE,
            },
        ),
        (
            top,
            PAGE_SIZE,
            MapError::VirtualRange {
                va: top,
                size: PAGE_SIZE,
            },
        ),
        (
            u64::MAX - PAGE_SIZE + 1,
            PAGE_SIZE,
            MapError::VirtualRange {
                va: u64::MAX - PAGE_SIZE + 1,
                size: PAGE_SIZE,
            },
        ),
    ];
    for (va, size, error) in refused {
        let error = Err(SpaceError::Map(error));
        assert_eq!(space.map(va, size, Perms::Read), error, "{va:#x}+{size:#x}");
        assert_eq!(space.unmap(va, size), error);
        assert_eq!(space.protect(va, size, Perms::Read), error);
    }
    assert_eq!(snapshot(&space), before);
    // The last page the tables translate can be mapped.
    assert_eq!(space.map(top - PAGE_SIZE, PAGE_SIZE, Perms::Read), Ok(()));
}

#[test]
fn a_mapping_over_a_mapped_page_is_refused_at_the_lowest_such_page() {
    let space = space();
    space
        .map(LOW + 2 * PAGE_SIZE, 2 * PAGE_SIZE, Perms::Read)
        .unwrap();
    let before = snapshot(&space);
    let cases = [
        // Starting inside it, ending inside it, covering it, and exactly it.
        (LOW + 3 * PAGE_SIZE, 2, LOW + 3 * PAGE_SIZE),
        (LOW, 3, LOW + 2 * PAGE_SIZE),
        (LOW, 8, LOW + 2 * PAGE_SIZE),
        (LOW + 2 * PAGE_SIZE, 2, LOW + 2 * PAGE_SIZE),
    ];
    for (va, pages, taken) in cases {
        assert_eq!(
            space.map(va, pages * PAGE_SIZE, Perms::ReadWrite),
            Err(SpaceError::Map(MapError::AlreadyMapped { va: taken })),
            "{va:#x}"
        );
    }
    assert_eq!(snapshot(&space), before);
    assert_eq!(
        SpaceError::Map(MapError::AlreadyMapped { va: LOW }).to_string(),
        "the guest's translation tables refused a mapping: 0x80000000 is already mapped"
    );
    // Right below and right above it there is room.
    assert_eq!(space.map(LOW, 2 * PAGE_SIZE, Perms::Read), Ok(()));
    assert_eq!(
        space.map(LOW + 4 * PAGE_SIZE, PAGE_SIZE, Perms::Read),
        Ok(())
    );
}

// ----- Reprotecting and unmapping ------------------------------------------------------------------

#[test]
fn protect_changes_backed_pages_at_once_and_unbacked_ones_when_they_are_touched() {
    let space = space();
    space.map(LOW, 4 * PAGE_SIZE, Perms::ReadWrite).unwrap();
    // Pages 0 and 2 are backed; 1 and 3 are not.
    assert_eq!(space.touch(LOW, Access::Write), Touch::Backed);
    assert_eq!(
        space.touch(LOW + 2 * PAGE_SIZE, Access::Write),
        Touch::Backed
    );
    // Pages 1 to 3 become read-only: inside the mapping, across backed and unbacked pages.
    assert_eq!(
        space.protect(LOW + PAGE_SIZE, 3 * PAGE_SIZE, Perms::Read),
        Ok(())
    );
    let walked = |va: u64| space.tables.walk(va);
    assert_eq!(walked(LOW), Ok(page(0, Perms::ReadWrite)));
    assert_eq!(walked(LOW + 2 * PAGE_SIZE), Ok(page(1, Perms::Read)));
    // Only the backed page had a table entry to change, so only it is owed a flush.
    assert_eq!(
        space.tables().take_pending_flushes(),
        [FlushRange {
            va: LOW + 2 * PAGE_SIZE,
            size: PAGE_SIZE,
            leaf_only: true
        }]
    );
    // A store to a read-only page is refused whether or not the page is backed; a load backs
    // the unbacked one, read-only.
    assert_eq!(space.touch(LOW + PAGE_SIZE, Access::Write), Touch::Refused);
    assert_eq!(
        space.touch(LOW + 2 * PAGE_SIZE, Access::Write),
        Touch::Refused
    );
    assert_eq!(space.touch(LOW + PAGE_SIZE, Access::Read), Touch::Backed);
    assert_eq!(walked(LOW + PAGE_SIZE), Ok(page(2, Perms::Read)));
    // The first page kept its permission: a store there is still not for want of backing.
    assert_eq!(
        space.shadow().mapping(LOW).unwrap().1.perms,
        Perms::ReadWrite
    );

    // Back to read-write over two mappings at once (the read-write page and the read-only
    // run): what the pages hold is kept.
    space.write(LOW + PAGE_SIZE, b"kept").unwrap();
    assert_eq!(space.protect(LOW, 4 * PAGE_SIZE, Perms::ReadWrite), Ok(()));
    assert_eq!(walked(LOW + PAGE_SIZE), Ok(page(2, Perms::ReadWrite)));
    assert_eq!(&ram_bytes::<4>(&space, 2, 0), b"kept");
    assert_eq!(
        space.touch(LOW + 3 * PAGE_SIZE, Access::Write),
        Touch::Backed
    );
}

#[test]
fn unmap_removes_table_entries_and_a_new_mapping_there_reads_as_zero() {
    let space = space();
    space.map(LOW, 4 * PAGE_SIZE, Perms::ReadWrite).unwrap();
    space.write(LOW + PAGE_SIZE, b"gone").unwrap();
    space.write(LOW + 3 * PAGE_SIZE, b"stay").unwrap();
    assert_eq!(
        space.tables.walk(LOW + PAGE_SIZE),
        Ok(page(0, Perms::ReadWrite))
    );
    // The middle two pages go: one backed, one not.
    assert_eq!(space.unmap(LOW + PAGE_SIZE, 2 * PAGE_SIZE), Ok(()));
    assert!(unmapped_at(&space, LOW + PAGE_SIZE));
    assert_eq!(space.touch(LOW + PAGE_SIZE, Access::Read), Touch::Refused);
    assert_eq!(
        space.touch(LOW + 2 * PAGE_SIZE, Access::Read),
        Touch::Refused
    );
    let mut buf = [0; 4];
    assert_eq!(
        space.read(LOW + PAGE_SIZE, &mut buf),
        Err(Fault {
            address: LOW + PAGE_SIZE
        })
    );
    // The table entry that went is owed a flush; the unbacked page had none.
    let flushes = space.tables().take_pending_flushes();
    assert_eq!(flushes.len(), 1);
    assert_eq!(
        (flushes[0].va, flushes[0].size),
        (LOW + PAGE_SIZE, PAGE_SIZE)
    );
    // The pages on either side are still mapped, with what they held.
    assert_eq!(space.read(LOW + 3 * PAGE_SIZE, &mut buf), Ok(()));
    assert_eq!(&buf, b"stay");
    assert_eq!(space.touch(LOW, Access::Write), Touch::Backed);

    // Mapped again, the page reads as zero for the host and for the guest: it gets RAM that
    // was never used, not the page that held "gone".
    assert_eq!(space.map(LOW + PAGE_SIZE, PAGE_SIZE, Perms::Read), Ok(()));
    assert_eq!(space.read(LOW + PAGE_SIZE, &mut buf), Ok(()));
    assert_eq!(buf, [0; 4]);
    assert_eq!(space.touch(LOW + PAGE_SIZE, Access::Read), Touch::Backed);
    let Resolved { pa, .. } = space.tables.walk(LOW + PAGE_SIZE).unwrap();
    assert_ne!(pa, RAM_BASE);
    let mut fresh = [0xee; 4];
    space.tables.memory().read(pa, &mut fresh).unwrap();
    assert_eq!(fresh, [0; 4]);
    assert_eq!(&ram_bytes::<4>(&space, 0, 0), b"gone");
}

#[test]
fn unmap_and_protect_need_every_page_mapped_and_say_which_is_not() {
    let space = space();
    space.map(LOW, PAGE_SIZE, Perms::ReadWrite).unwrap();
    space
        .map(LOW + 2 * PAGE_SIZE, PAGE_SIZE, Perms::ReadWrite)
        .unwrap();
    assert_eq!(space.touch(LOW, Access::Read), Touch::Backed);
    let before = snapshot(&space);
    let cases = [
        // The hole between the two, reached from below and named exactly; the page above;
        // a range that starts unmapped.
        (LOW, 3, LOW + PAGE_SIZE),
        (LOW + 2 * PAGE_SIZE, 2, LOW + 3 * PAGE_SIZE),
        (LOW - PAGE_SIZE, 2, LOW - PAGE_SIZE),
        (FAR, 1, FAR),
    ];
    for (va, pages, hole) in cases {
        let error = Err(SpaceError::Map(MapError::NotMapped { va: hole }));
        assert_eq!(space.unmap(va, pages * PAGE_SIZE), error, "{va:#x}");
        assert_eq!(space.protect(va, pages * PAGE_SIZE, Perms::Read), error);
    }
    assert_eq!(snapshot(&space), before);
    assert_eq!(space.tables().take_pending_flushes(), []);
    // Each mapping on its own can be unmapped, the unbacked one without a flush.
    assert_eq!(space.unmap(LOW + 2 * PAGE_SIZE, PAGE_SIZE), Ok(()));
    assert_eq!(space.tables().take_pending_flushes(), []);
    assert_eq!(space.unmap(LOW, PAGE_SIZE), Ok(()));
    assert_eq!(space.tables().take_pending_flushes().len(), 1);
    assert!(space.shadow().mapped.is_empty() && space.shadow().backed.is_empty());
}

// ----- Host access -------------------------------------------------------------------------------

#[test]
fn host_access_goes_through_the_shadow_whatever_the_guest_may_do() {
    let space = space();
    // Neither page is writable by guest code; the host writes both. They are neighbours in
    // virtual addresses, separate mappings, and the wrong way round in RAM.
    space.map(LOW + PAGE_SIZE, PAGE_SIZE, Perms::None).unwrap();
    space.map(LOW, PAGE_SIZE, Perms::ReadExecute).unwrap();
    space.write(LOW + PAGE_SIZE, b"x").unwrap();
    let across = LOW + PAGE_SIZE - 3;
    space.write(across, b"abcdefgh").unwrap();
    // The first three bytes end the lower page (RAM page 1), the rest start the upper
    // (RAM page 0).
    assert_eq!(&ram_bytes::<3>(&space, 1, PAGE_SIZE - 3), b"abc");
    assert_eq!(&ram_bytes::<5>(&space, 0, 0), b"defgh");
    let mut back = [0; 8];
    space.read(across, &mut back).unwrap();
    assert_eq!(&back, b"abcdefgh");
    // The pages the host wrote are backed with their mappings' permissions, for the guest.
    assert_eq!(space.tables.walk(LOW), Ok(page(1, Perms::ReadExecute)));
    assert_eq!(space.tables.walk(LOW + PAGE_SIZE), Ok(page(0, Perms::None)));
    // A virtual address is not a physical one: the same bytes are not at the virtual address
    // in the physical memory (which does not even reach that far).
    assert!(space.tables.memory().read(across, &mut back).is_err());
}

#[test]
fn a_page_that_is_mapped_and_not_backed_reads_as_zero_and_a_read_does_not_back_it() {
    let space = space();
    space.map(LOW, 3 * PAGE_SIZE, Perms::ReadWrite).unwrap();
    space.write(LOW + PAGE_SIZE, &[0xaa; 4]).unwrap();
    let before = snapshot(&space);
    // From the unbacked first page, over the backed second, into the unbacked third.
    let len = usize::try_from(PAGE_SIZE).unwrap() + 4;
    let mut buf = vec![0xee; len];
    assert_eq!(space.read(LOW + PAGE_SIZE - 2, &mut buf), Ok(()));
    assert_eq!(buf[..2], [0, 0]);
    assert_eq!(buf[2..6], [0xaa; 4]);
    assert!(buf[6..].iter().all(|byte| *byte == 0));
    assert_eq!(snapshot(&space), before);
    assert!(unmapped_at(&space, LOW) && unmapped_at(&space, LOW + 2 * PAGE_SIZE));
}

#[test]
fn an_access_that_leaves_what_is_mapped_faults_there_and_does_nothing() {
    let space = space();
    space.map(LOW, 2 * PAGE_SIZE, Perms::ReadWrite).unwrap();
    let end = LOW + 2 * PAGE_SIZE;
    space.write(end - 4, b"keep").unwrap();
    let before = snapshot(&space);

    // Starts mapped, ends one byte past: the fault is at the first unmapped address, and
    // the mapped part was neither written nor read, nor was the unbacked first page backed.
    assert_eq!(space.write(end - 4, b"12345"), Err(Fault { address: end }));
    assert_eq!(
        space.write(LOW, &vec![1; usize::try_from(2 * PAGE_SIZE).unwrap() + 1]),
        Err(Fault { address: end })
    );
    assert_eq!(&ram_bytes::<4>(&space, 0, PAGE_SIZE - 4), b"keep");
    let mut buf = [0xee; 5];
    assert_eq!(space.read(end - 4, &mut buf), Err(Fault { address: end }));
    assert_eq!(buf, [0xee; 5]);
    // Ending exactly at the end is inside.
    assert_eq!(space.read(end - 4, &mut buf[..4]), Ok(()));
    assert_eq!(&buf[..4], b"keep");

    // Starts unmapped: below, above, and where adding the length overflows.
    for address in [LOW - 1, end, 0, u64::MAX] {
        assert_eq!(
            space.read(address, &mut buf),
            Err(Fault { address }),
            "{address:#x}"
        );
        assert_eq!(
            space.write(address, &buf),
            Err(Fault { address }),
            "{address:#x}"
        );
    }
    // An access of no bytes touches nothing, so it cannot fault.
    assert_eq!(space.read(end, &mut []), Ok(()));
    assert_eq!(space.write(u64::MAX, &[]), Ok(()));
    assert_eq!(snapshot(&space), before);
}

// ----- Construction and the kernel's side ----------------------------------------------------------

#[test]
fn a_pool_the_memory_does_not_hold_is_refused() {
    let physical = FlatMemory::new(POOL_BASE, usize::try_from(PAGE_SIZE).unwrap());
    let pool = BackingRange::new(POOL_BASE, 2 * PAGE_SIZE).unwrap();
    let ram = BackingRange::new(RAM_BASE, PAGE_SIZE).unwrap();
    let error = AddressSpace::new(physical, ram, pool).unwrap_err();
    assert_eq!(
        error,
        SpaceError::Map(MapError::Memory(Fault {
            address: POOL_BASE + PAGE_SIZE
        }))
    );
}

#[test]
fn the_kernels_mapper_drives_the_address_space_and_reports_refusals_in_words() {
    let space = Arc::new(space());
    let mut mapper = SpaceMapper(Arc::clone(&space));
    assert_eq!(mapper.map(LOW, 2 * PAGE_SIZE, Perms::ReadWrite), Ok(()));
    assert_eq!(space.touch(LOW, Access::Write), Touch::Backed);
    assert_eq!(mapper.protect(LOW, PAGE_SIZE, Perms::Read), Ok(()));
    assert_eq!(space.tables.walk(LOW), Ok(page(0, Perms::Read)));
    assert_eq!(mapper.unmap(LOW, 2 * PAGE_SIZE), Ok(()));
    assert!(unmapped_at(&space, LOW));
    assert!(space.shadow().mapped.is_empty());

    let refusal = |text: &str| Err(MapperError(text.to_owned()));
    assert_eq!(mapper.map(LOW, PAGE_SIZE, Perms::Read), Ok(()));
    assert_eq!(
        mapper.map(LOW, PAGE_SIZE, Perms::Read),
        refusal("the guest's translation tables refused a mapping: 0x80000000 is already mapped")
    );
    assert_eq!(
        mapper.unmap(FAR, PAGE_SIZE),
        refusal("the guest's translation tables refused a mapping: 0x1000000000 is not mapped")
    );
    assert_eq!(
        mapper.protect(FAR + 1, PAGE_SIZE, Perms::Read),
        refusal(
            "the guest's translation tables refused a mapping: address 0x1000000001 is not \
             page aligned"
        )
    );
}
