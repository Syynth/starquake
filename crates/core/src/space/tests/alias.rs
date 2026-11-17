//! `MapMemory`'s aliases in the address space over flat guest-physical memory: a second mapping
//! of the source's RAM, backed at once, whose bytes are the source's both ways, and whose
//! unmap leaves the source as it was.

use super::*;

/// A source of two pages, mapped read-write, and where its alias goes: far from it, in another
/// 1 GiB slot of the tables.
const SOURCE: u64 = LOW;
const ALIAS: u64 = FAR;
const TWO: u64 = 2 * PAGE_SIZE;

/// The span of one level-2 descriptor with a 4 KiB granule: 2 MiB (Arm DDI 0487, "Translation
/// table level"). A page past such a boundary needs a level-3 table of its own.
const LEVEL_2_SPAN: u64 = 1 << 21;

/// The span of one level-1 descriptor: 1 GiB.
const LEVEL_1_SPAN: u64 = 1 << 30;

/// RAM page `n`, as one page of backing.
fn ram_page(n: u64) -> u64 {
    RAM_BASE + n * PAGE_SIZE
}

#[test]
fn an_alias_backs_every_source_page_and_maps_the_same_ram_at_once() {
    let space = space();
    space.map(SOURCE, TWO, Perms::ReadWrite).unwrap();
    // The second page is backed already (RAM page 0); the first is not.
    space.write(SOURCE + PAGE_SIZE, b"two").unwrap();
    assert_eq!(space.map_alias(ALIAS, SOURCE, TWO, Perms::ReadWrite), Ok(()));
    // The first source page got the next RAM page, with the source's permission.
    assert_eq!(space.tables.walk(SOURCE), Ok(page(1, Perms::ReadWrite)));
    assert_eq!(
        space.tables.walk(SOURCE + PAGE_SIZE),
        Ok(page(0, Perms::ReadWrite))
    );
    // The alias maps the same RAM, page by page, with the permission asked for.
    assert_eq!(space.tables.walk(ALIAS), Ok(page(1, Perms::ReadWrite)));
    assert_eq!(
        space.tables.walk(ALIAS + PAGE_SIZE),
        Ok(page(0, Perms::ReadWrite))
    );
    assert!(unmapped_at(&space, ALIAS + TWO));
    assert!(unmapped_at(&space, ALIAS - PAGE_SIZE));
    assert_eq!(
        space.shadow().mapping(ALIAS),
        Some((
            ALIAS,
            Mapped {
                end: ALIAS + TWO,
                perms: Perms::ReadWrite
            }
        ))
    );
    assert_eq!(space.shadow().next, ram_page(2));
    // Every alias page is backed, so a fault there is never for want of backing; mapping owes
    // no flush.
    for access in [Access::Read, Access::Write, Access::Execute] {
        assert_eq!(space.touch(ALIAS, access), Touch::Refused);
    }
    assert_eq!(space.tables().take_pending_flushes(), []);
}

#[test]
fn bytes_written_through_one_are_read_through_the_other_and_outlive_the_alias() {
    let space = space();
    space.map(SOURCE, TWO, Perms::ReadWrite).unwrap();
    space.write(SOURCE + 3, b"src").unwrap();
    space.map_alias(ALIAS, SOURCE, TWO, Perms::ReadWrite).unwrap();
    let mut seen = [0; 3];
    space.read(ALIAS + 3, &mut seen).unwrap();
    assert_eq!(&seen, b"src");
    // A write that crosses the page boundary of the alias lands in both source pages.
    space.write(ALIAS + PAGE_SIZE - 2, b"dsta").unwrap();
    let mut both = [0; 4];
    space.read(SOURCE + PAGE_SIZE - 2, &mut both).unwrap();
    assert_eq!(&both, b"dsta");
    // Unmapping the alias leaves the source's RAM and what it holds.
    assert_eq!(space.unmap(ALIAS, TWO), Ok(()));
    assert!(unmapped_at(&space, ALIAS));
    space.read(SOURCE + PAGE_SIZE - 2, &mut both).unwrap();
    assert_eq!(&both, b"dsta");
    assert_eq!(space.tables.walk(SOURCE), Ok(page(0, Perms::ReadWrite)));
    assert_eq!(space.read(ALIAS, &mut seen), Err(Fault { address: ALIAS }));
}

#[test]
fn a_source_with_no_access_keeps_it_and_its_alias_has_the_permission_asked_for() {
    // As `MapMemory` does: the source loses the guest's access first, then gets its alias.
    let space = space();
    space.map(SOURCE, PAGE_SIZE, Perms::ReadWrite).unwrap();
    space.protect(SOURCE, PAGE_SIZE, Perms::None).unwrap();
    space
        .map_alias(ALIAS, SOURCE, PAGE_SIZE, Perms::ReadWrite)
        .unwrap();
    assert_eq!(space.tables.walk(SOURCE), Ok(page(0, Perms::None)));
    assert_eq!(space.tables.walk(ALIAS), Ok(page(0, Perms::ReadWrite)));
    // And `UnmapMemory`'s order: the alias goes, then the source gets its access back.
    space.unmap(ALIAS, PAGE_SIZE).unwrap();
    space.protect(SOURCE, PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(space.tables.walk(SOURCE), Ok(page(0, Perms::ReadWrite)));
}

#[test]
fn an_alias_is_refused_with_its_checks_in_order_and_maps_nothing() {
    let space = space();
    space.map(SOURCE, TWO, Perms::ReadWrite).unwrap();
    space.map(ALIAS + TWO, PAGE_SIZE, Perms::ReadWrite).unwrap();
    let before = snapshot(&space);
    // The ranges: the alias's, then the source's.
    assert_eq!(
        space.map_alias(ALIAS + 1, SOURCE + 1, TWO, Perms::ReadWrite),
        Err(SpaceError::Map(MapError::MisalignedAddress { address: ALIAS + 1 }))
    );
    assert_eq!(
        space.map_alias(ALIAS, SOURCE + 1, TWO, Perms::ReadWrite),
        Err(SpaceError::Map(MapError::MisalignedAddress {
            address: SOURCE + 1
        }))
    );
    assert_eq!(
        space.map_alias(ALIAS, SOURCE, 0, Perms::ReadWrite),
        Err(SpaceError::Map(MapError::Empty))
    );
    // The source must be mapped, all of it: one page past it is not, whatever the alias.
    assert_eq!(
        space.map_alias(ALIAS + TWO, SOURCE, TWO + PAGE_SIZE, Perms::ReadWrite),
        Err(SpaceError::Map(MapError::NotMapped { va: SOURCE + TWO }))
    );
    // The alias's range must be free: a mapping that starts exactly at its last page is in
    // the way, and one that starts exactly at its end is not.
    assert_eq!(
        space.map_alias(ALIAS + PAGE_SIZE, SOURCE, TWO, Perms::ReadWrite),
        Err(SpaceError::Map(MapError::AlreadyMapped { va: ALIAS + TWO }))
    );
    assert_eq!(snapshot(&space), before);
    assert_eq!(space.map_alias(ALIAS, SOURCE, TWO, Perms::ReadWrite), Ok(()));
    assert_eq!(
        space.shadow().mapping(ALIAS + TWO),
        Some((
            ALIAS + TWO,
            Mapped {
                end: ALIAS + TWO + PAGE_SIZE,
                perms: Perms::ReadWrite
            }
        ))
    );
    assert!(unmapped_at(&space, ALIAS + TWO));
}

#[test]
fn an_alias_whose_source_needs_more_ram_than_is_left_is_refused_before_backing_any() {
    let space = space();
    let all_ram = RAM_PAGES * PAGE_SIZE;
    // A source one page larger than RAM, with nothing backed: every page needs RAM.
    space
        .map(SOURCE, all_ram + PAGE_SIZE, Perms::ReadWrite)
        .unwrap();
    let before = snapshot(&space);
    assert_eq!(
        space.map_alias(ALIAS, SOURCE, all_ram + PAGE_SIZE, Perms::ReadWrite),
        Err(SpaceError::OutOfRam {
            size: all_ram + PAGE_SIZE,
            available: all_ram
        })
    );
    assert_eq!(snapshot(&space), before);
    // Exactly as much as is left fits.
    assert_eq!(
        space.map_alias(ALIAS, SOURCE, all_ram, Perms::ReadWrite),
        Ok(())
    );
    assert_eq!(space.shadow().next, ram_page(RAM_PAGES));
}

#[test]
fn an_alias_the_tables_refuse_part_way_unmaps_what_it_mapped() {
    let space = space();
    space.map(SOURCE, TWO, Perms::ReadWrite).unwrap();
    // Back the source first, so that the refusal is the alias's tables'.
    space.write(SOURCE, &[1]).unwrap();
    space.write(SOURCE + PAGE_SIZE, &[2]).unwrap();
    // The alias's first page is the last of a 2 MiB block whose level-3 table exists (a page
    // below it is backed), and its second page is the first of the next block, which needs a
    // table of its own.
    let alias = ALIAS + LEVEL_2_SPAN - PAGE_SIZE;
    space
        .map(alias - PAGE_SIZE, PAGE_SIZE, Perms::ReadWrite)
        .unwrap();
    assert_eq!(space.touch(alias - PAGE_SIZE, Access::Read), Touch::Backed);
    // Use the table pool up: a mapping in each new 1 GiB slot needs two tables, then one in
    // a new 2 MiB block of the alias's slot needs one.
    let pages = space.allocate_shared(PAGE_SIZE).unwrap();
    let mut slot = 16;
    while space.tables.pool_pages() - space.tables.table_pages_in_use() >= 2 {
        space
            .map_shared(slot * LEVEL_1_SPAN, pages, Perms::Read)
            .unwrap();
        slot += 1;
    }
    if space.tables.pool_pages() > space.tables.table_pages_in_use() {
        space
            .map_shared(ALIAS + 4 * LEVEL_2_SPAN, pages, Perms::Read)
            .unwrap();
    }
    assert_eq!(space.tables.pool_pages(), space.tables.table_pages_in_use());
    drop(space.tables().take_pending_flushes());
    let before = snapshot(&space);
    assert_eq!(
        space.map_alias(alias, SOURCE, TWO, Perms::ReadWrite),
        Err(SpaceError::Map(MapError::OutOfTablePages))
    );
    // The first alias page was mapped and is gone again, and its unmap owes a flush.
    assert!(unmapped_at(&space, alias));
    assert_eq!(snapshot(&space), before);
    assert_eq!(
        space.tables().take_pending_flushes(),
        [FlushRange {
            va: alias,
            size: PAGE_SIZE,
            leaf_only: true
        }]
    );
}

#[test]
fn a_reprotect_before_a_refused_alias_still_owes_its_flush() {
    // `MapMemory` reprotects the source, then asks for the alias; when the alias is refused
    // the flush the reprotect owes is still there for the core loop to run after the call.
    let space = space();
    space.map(SOURCE, PAGE_SIZE, Perms::ReadWrite).unwrap();
    space.write(SOURCE, &[7]).unwrap();
    space.map(ALIAS, PAGE_SIZE, Perms::ReadWrite).unwrap();
    space.protect(SOURCE, PAGE_SIZE, Perms::None).unwrap();
    assert_eq!(
        space.map_alias(ALIAS, SOURCE, PAGE_SIZE, Perms::ReadWrite),
        Err(SpaceError::Map(MapError::AlreadyMapped { va: ALIAS }))
    );
    assert_eq!(
        space.tables().take_pending_flushes(),
        [FlushRange {
            va: SOURCE,
            size: PAGE_SIZE,
            leaf_only: true
        }]
    );
}

#[test]
fn the_kernel_reaches_the_alias_through_the_mapper() {
    let space = Arc::new(space());
    let mut mapper = SpaceMapper(Arc::clone(&space));
    mapper.map(SOURCE, PAGE_SIZE, Perms::ReadWrite).unwrap();
    space.write(SOURCE, b"k").unwrap();
    assert_eq!(
        mapper.map_alias(ALIAS, SOURCE, PAGE_SIZE, Perms::ReadWrite),
        Ok(())
    );
    let mut byte = [0];
    space.read(ALIAS, &mut byte).unwrap();
    assert_eq!(&byte, b"k");
    // A refusal is the address space's error, in words.
    assert_eq!(
        mapper.map_alias(ALIAS, SOURCE, PAGE_SIZE, Perms::ReadWrite),
        Err(MapperError(
            SpaceError::Map(MapError::AlreadyMapped { va: ALIAS }).to_string()
        ))
    );
}
