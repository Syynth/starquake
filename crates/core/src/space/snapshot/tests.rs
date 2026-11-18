//! The guest-pages snapshot: its layout, the round trip with the kernel's address space through
//! a new address space, every check in its order with each limit's edge, and the save-side
//! checks on a corrupted shadow.

use starquake_kernel::id::SlabId;
use starquake_kernel::memory::{
    MemoryArrangement, ProcessMemory, RegionOffsets, SharedPages, SvcPermission,
};
use starquake_memory::stage1::WalkError;
use starquake_memory::{Access, FlatMemory, MapError, Perms};

use super::*;
use crate::space::{Shadow, Touch};

/// Guest-physical layout: a table pool, then RAM, in one flat memory.
const POOL_BASE: u64 = 0x4_0000_0000;
const POOL_PAGES: u64 = 64;
const RAM_BASE: u64 = POOL_BASE + POOL_PAGES * PAGE_SIZE;
const RAM_PAGES: u64 = 32;
const RAM_END: u64 = RAM_BASE + RAM_PAGES * PAGE_SIZE;

/// Virtual addresses for the tests without the kernel: two neighbouring pages and one far away.
const LOW: u64 = 0x8000_0000;
const FAR: u64 = 0x10_0000_0000;

/// The kernel fixture's code address and size (`crates/kernel/src/memory/fixture.rs`), and a
/// free place for shared memory inside the alias-code region: as `space/tests/kernel.rs` has.
const CODE: u64 = 0x8000_0000;
const CODE_SIZE: u64 = 7 * PAGE_SIZE;
const SHARED: u64 = 0x4000_0000;

/// The id the tests' mapped shared memory has in a made-up object table.
fn mapped_id() -> ObjectId {
    ObjectId::from_parts(3, 5)
}

/// An address space over flat memory with `pool_pages` of table pool and `ram_pages` of RAM,
/// of which the flat memory holds the first `held_pages`.
fn rig_with(pool_pages: u64, ram_pages: u64, held_pages: u64) -> AddressSpace<FlatMemory> {
    let ram_base = POOL_BASE + pool_pages * PAGE_SIZE;
    let bytes = usize::try_from((pool_pages + held_pages) * PAGE_SIZE).unwrap();
    let physical = FlatMemory::new(POOL_BASE, bytes);
    let pool = BackingRange::new(POOL_BASE, pool_pages * PAGE_SIZE).unwrap();
    let ram = BackingRange::new(ram_base, ram_pages * PAGE_SIZE).unwrap();
    AddressSpace::new(physical, ram, pool).unwrap()
}

fn rig() -> AddressSpace<FlatMemory> {
    rig_with(POOL_PAGES, RAM_PAGES, RAM_PAGES)
}

fn ram() -> BackingRange {
    BackingRange::new(RAM_BASE, RAM_PAGES * PAGE_SIZE).unwrap()
}

/// The fixture's program.
fn process() -> ProcessMemory {
    ProcessMemory {
        code_address: CODE,
        code_size: CODE_SIZE,
        arrangement: MemoryArrangement::FourGb,
        alias_region_extra: false,
        offsets: RegionOffsets::default(),
    }
}

/// A page of `byte`, as a page's contents.
fn filled(byte: u8) -> Contents {
    Contents::Bytes(PageBytes(vec![byte; PAGE_STEP]))
}

/// A snapshot of nothing but `next`.
fn empty(next: u64) -> Body {
    Body {
        next,
        shared: Vec::new(),
        free: Vec::new(),
        pages: Vec::new(),
    }
}

fn bytes_of(body: &Body) -> Vec<u8> {
    encode(FORMAT, body).unwrap()
}

/// What `begin_restore` makes of `body` on a new rig.
fn begin(body: &Body) -> Result<(), PagesError> {
    rig().begin_restore(&bytes_of(body))
}

/// Everything a restore sets: next, mappings, backed pages, shared runs, the given-back runs
/// (waiting or reusable, together), aliased pages and pages waiting for a mapping.
type State = (
    u64,
    Vec<(u64, u64, Perms)>,
    Vec<(u64, u64)>,
    Vec<(u64, u64)>,
    Vec<(u64, u64)>,
    Vec<u64>,
    Vec<(u64, u64)>,
);

fn state<M: GuestMemory>(space: &AddressSpace<M>) -> State {
    let shadow = space.shadow();
    let mut free: Vec<(u64, u64)> = shadow
        .released
        .iter()
        .map(|run| (run.base(), run.end()))
        .chain(shadow.reusable.iter().map(|(b, e)| (*b, *e)))
        .collect();
    free.sort_unstable();
    // Mappings side by side with one permission are one: the kernel's blocks merge where the
    // calls that made them did not.
    let mut mapped: Vec<(u64, u64, Perms)> = Vec::new();
    for (start, m) in &shadow.mapped {
        match mapped.last_mut() {
            Some((_, end, perms)) if *end == *start && *perms == m.perms => *end = m.end,
            _ => mapped.push((*start, m.end, m.perms)),
        }
    }
    (
        shadow.next,
        mapped,
        shadow.backed.iter().map(|(v, p)| (*v, *p)).collect(),
        shadow.shared.iter().map(|(b, e)| (*b, *e)).collect(),
        free,
        shadow.aliased.iter().copied().collect(),
        shadow.pending.iter().map(|(v, p)| (*v, *p)).collect(),
    )
}

/// Every byte of guest RAM, read behind the address space's back.
fn ram_bytes(space: &AddressSpace<FlatMemory>) -> Vec<u8> {
    let mut bytes = vec![0; usize::try_from(RAM_PAGES * PAGE_SIZE).unwrap()];
    space.tables.memory().read(RAM_BASE, &mut bytes).unwrap();
    bytes
}

fn new_state() -> State {
    state(&rig())
}

// ----- The layout --------------------------------------------------------------------------------

#[test]
fn the_body_layout_is_pinned() {
    // Every variant and list, each value distinct and not zero; the page bytes are short, since
    // the layout does not depend on their length (the checks refuse it, not the encoding).
    let body = Body {
        next: 0x2345,
        shared: vec![SharedRun {
            base: 0x11,
            pages: vec![Contents::Zero, Contents::Bytes(PageBytes(vec![0xAA, 0xBB]))],
        }],
        free: vec![FreeRun {
            base: 0x22,
            size: 0x33,
        }],
        pages: vec![
            SavedPage {
                va: 0x44,
                pa: 0x55,
                contents: Contents::Bytes(PageBytes(vec![0xCC])),
            },
            SavedPage {
                va: 0x66,
                pa: 0x77,
                contents: Contents::Zero,
            },
        ],
    };
    let expected: &[u8] = &[
        b'S', b'Q', b'G', b'P', // magic
        0x01, // version
        0xC5, 0x46, // next, 0x2345 as a varint
        0x01, // one shared run
        0x11, // its base
        0x02, // two pages
        0x00, // zero
        0x01, 0x02, 0xAA, 0xBB, // bytes: tag, count, bytes
        0x01, // one given-back run
        0x22, 0x33, // base, size
        0x02, // two private pages
        0x44, 0x55, 0x01, 0x01, 0xCC, // va, pa, bytes
        0x66, 0x77, 0x00, // va, pa, zero
    ];
    assert_eq!(bytes_of(&body), expected);
    assert_eq!(decode::<Body>(FORMAT, expected), Ok(body));
}

#[test]
fn the_magic_and_version_are_the_documented_ones() {
    assert_eq!(&PAGES_MAGIC, b"SQGP");
    assert_eq!(PAGES_VERSION, 1);
}

#[test]
fn another_version_and_another_magic_are_refused_by_the_header() {
    let mut bytes = bytes_of(&empty(RAM_BASE));
    bytes[4] = 2;
    assert_eq!(
        rig().begin_restore(&bytes),
        Err(PagesError::Format(FormatError::UnsupportedVersion {
            name: "guest pages",
            found: 2,
            supported: PAGES_VERSION,
        }))
    );
    let mut bytes = bytes_of(&empty(RAM_BASE));
    bytes[0] = b'X';
    assert_eq!(
        rig().begin_restore(&bytes),
        Err(PagesError::Format(FormatError::BadMagic {
            name: "guest pages"
        }))
    );
}

// ----- The round trip ----------------------------------------------------------------------------

/// A process the tests built through the kernel, over the real address space.
struct Scene {
    space: Arc<AddressSpace<FlatMemory>>,
    kernel: KernelSpace,
    heap: u64,
    alias: u64,
    physical: u64,
    /// The mapped shared memory's pages, and the one only a service holds.
    mapped: SharedPages,
    held: SharedPages,
    /// The run of the shared memory given back.
    given: BackingRange,
}

/// A synthetic guest: code loaded and made executable, a stack, four pages of heap (the guest
/// stores to the first, loads from the second, aliases the third at the top of the stack
/// region and writes through the alias, and leaves the fourth untouched), physical memory written by the host, a shared memory
/// mapped and written by the service and the guest, one only the service holds, and one given
/// back after it.
fn scene() -> Scene {
    let space = Arc::new(rig());
    let mut kernel =
        KernelSpace::new(&process(), Box::new(SpaceMapper(Arc::clone(&space)))).unwrap();
    kernel
        .set_process_memory_permission(CODE, PAGE_SIZE, SvcPermission::ReadExecute)
        .unwrap();
    space.write(CODE, &[0x1F, 0x20, 0x03, 0xD5]).unwrap();
    kernel.create_main_thread_stack(PAGE_SIZE).unwrap();
    let heap = kernel.set_heap_size(4 * PAGE_SIZE).unwrap();
    assert_eq!(space.touch(heap, Access::Write), Touch::Backed);
    space
        .write(heap + 8, &0x0123_4567_u32.to_le_bytes())
        .unwrap();
    assert_eq!(space.touch(heap + PAGE_SIZE, Access::Read), Touch::Backed);
    let alias = kernel.regions().stack.end - 16 * PAGE_SIZE;
    kernel
        .map_memory(alias, heap + 2 * PAGE_SIZE, PAGE_SIZE)
        .unwrap();
    space.write(alias + 16, b"through the alias").unwrap();
    let physical = kernel.regions().alias.start;
    kernel.map_physical_memory(physical, PAGE_SIZE).unwrap();
    space.write(physical + 0x40, b"physical").unwrap();
    let mapped = kernel.allocate_shared(2 * PAGE_SIZE).unwrap();
    mapped.write(PAGE_SIZE + 4, b"service").unwrap();
    let memory = SharedMemory::new(mapped.backing(), SvcPermission::ReadWrite).unwrap();
    kernel
        .map_shared(SHARED, mapped_id(), memory, SvcPermission::ReadWrite)
        .unwrap();
    space.write(SHARED + 32, b"guest").unwrap();
    let held = kernel.allocate_shared(PAGE_SIZE).unwrap();
    held.write(0, b"held").unwrap();
    // Given back last, so that no later shared memory takes it.
    let given = kernel.allocate_shared(PAGE_SIZE).unwrap().backing();
    kernel.release_shared(given).unwrap();
    Scene {
        space,
        kernel,
        heap,
        alias,
        physical,
        mapped,
        held,
        given,
    }
}

/// The shared memory the scene mapped, for a restore's object table.
fn objects(scene: &Scene) -> impl FnMut(ObjectId) -> Option<SharedMemory> {
    let backing = scene.mapped.backing();
    move |id| {
        (id == mapped_id()).then(|| SharedMemory::new(backing, SvcPermission::ReadWrite).unwrap())
    }
}

/// The virtual pages whose translation a round trip compares: every backed page, every page
/// the scene names, and one past each.
fn pages_to_walk(scene: &Scene) -> Vec<u64> {
    let mut pages: Vec<u64> = scene.space.shadow().backed.keys().copied().collect();
    for page in [CODE, scene.heap, scene.alias, scene.physical, SHARED] {
        pages.extend([page, page + PAGE_SIZE, page + 4 * PAGE_SIZE]);
    }
    pages
}

#[test]
fn a_synthetic_guest_saves_and_restores_into_a_new_address_space() {
    let scene = scene();
    let pages = scene.space.save_pages().unwrap();
    let blocks = scene.kernel.save().unwrap();

    let restored = Arc::new(rig());
    let mut kernel = restore(&restored, &pages, &blocks, objects(&scene)).unwrap();

    // The touched pages, the shared mappings and the RAM behind them are what they were.
    assert_eq!(state(&restored), state(&scene.space));
    assert_eq!(ram_bytes(&restored), ram_bytes(&scene.space));
    for page in pages_to_walk(&scene) {
        assert_eq!(
            restored.tables.walk(page),
            scene.space.tables.walk(page),
            "{page:#x}"
        );
    }
    // Guest code would read what it wrote, through each kind of page.
    let read = |at: u64, len: usize| {
        let mut bytes = vec![0; len];
        restored.read(at, &mut bytes).unwrap();
        bytes
    };
    assert_eq!(read(scene.heap + 8, 4), 0x0123_4567_u32.to_le_bytes());
    assert_eq!(read(scene.alias + 16, 17), b"through the alias");
    assert_eq!(
        read(scene.heap + 2 * PAGE_SIZE + 16, 17),
        b"through the alias"
    );
    assert_eq!(read(scene.physical + 0x40, 8), b"physical");
    assert_eq!(read(SHARED + 32, 5), b"guest");
    assert_eq!(read(SHARED + PAGE_SIZE + 4, 7), b"service");

    // The restored kernel drives the restored space: the heap grows onto RAM never handed out,
    // the alias goes and its source is the guest's again, and the next shared memory of one
    // page takes the run given back, zeroed.
    let next = restored.shadow().next;
    let heap = kernel.set_heap_size(5 * PAGE_SIZE).unwrap();
    assert_eq!(
        restored.touch(heap + 4 * PAGE_SIZE, Access::Write),
        Touch::Backed
    );
    assert_eq!(
        restored.shadow().backed.get(&(heap + 4 * PAGE_SIZE)),
        Some(&next)
    );
    kernel
        .unmap_memory(scene.alias, scene.heap + 2 * PAGE_SIZE, PAGE_SIZE)
        .unwrap();
    assert!(!restored.shadow().aliased.contains(&scene.alias));
    let again = kernel.allocate_shared(PAGE_SIZE).unwrap();
    assert_eq!(again.backing(), scene.given);
    let mut first = [0xFF; 4];
    again.read(0, &mut first).unwrap();
    assert_eq!(first, [0; 4]);
    // The page a service holds kept its bytes, at the same guest-physical address.
    let mut held = [0; 4];
    restored
        .tables
        .memory()
        .read(scene.held.backing().base(), &mut held)
        .unwrap();
    assert_eq!(&held, b"held");
}

#[test]
fn a_restored_space_saves_the_same_bytes() {
    let scene = scene();
    let pages = scene.space.save_pages().unwrap();
    let blocks = scene.kernel.save().unwrap();
    let restored = Arc::new(rig());
    let kernel = restore(&restored, &pages, &blocks, objects(&scene)).unwrap();
    assert_eq!(kernel.save().unwrap(), blocks);
    assert_eq!(restored.save_pages().unwrap(), pages);
}

#[test]
fn what_is_saved_is_each_page_of_ram_once_and_a_read_only_page_as_zero() {
    let scene = scene();
    let body: Body = decode(FORMAT, &scene.space.save_pages().unwrap()).unwrap();
    let shadow = scene.space.shadow();
    // The private pages: neither the alias nor the shared mapping's pages, which share RAM with
    // other pages.
    let vas: Vec<u64> = body.pages.iter().map(|page| page.va).collect();
    let expected: Vec<u64> = shadow
        .backed
        .keys()
        .copied()
        .filter(|va| *va != scene.alias && !(SHARED..SHARED + 2 * PAGE_SIZE).contains(va))
        .collect();
    assert_eq!(vas, expected);
    assert!(vas.contains(&(scene.heap + 2 * PAGE_SIZE)));
    // The page only loaded from is zero; the page stored to is bytes.
    let contents = |va: u64| {
        &body
            .pages
            .iter()
            .find(|page| page.va == va)
            .unwrap()
            .contents
    };
    assert_eq!(contents(scene.heap + PAGE_SIZE), &Contents::Zero);
    assert!(matches!(contents(scene.heap), Contents::Bytes(_)));
    // Both live shared runs with their pages; the given-back run without.
    let runs: Vec<(u64, usize)> = body
        .shared
        .iter()
        .map(|run| (run.base, run.pages.len()))
        .collect();
    assert_eq!(
        runs,
        [
            (scene.mapped.backing().base(), 2),
            (scene.held.backing().base(), 1)
        ]
    );
    // Each shared page was written (the guest wrote the first, the service the second).
    assert!(
        body.shared
            .iter()
            .flat_map(|run| &run.pages)
            .all(|page| matches!(page, Contents::Bytes(_)))
    );
    assert_eq!(
        body.free,
        [FreeRun {
            base: scene.given.base(),
            size: scene.given.size(),
        }]
    );
    assert_eq!(body.next, shadow.next);
}

#[test]
fn a_run_given_back_while_a_flush_was_owed_comes_back_reusable() {
    let space = rig();
    let run = space.allocate_shared(PAGE_SIZE).unwrap();
    space.map(LOW, PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(space.touch(LOW, Access::Write), Touch::Backed);
    space.unmap(LOW, PAGE_SIZE).unwrap();
    assert!(space.tables.has_pending_flushes());
    space.release_shared(run).unwrap();
    assert_eq!(space.shadow().released, [run]);

    let restored = rig();
    restored
        .begin_restore(&space.save_pages().unwrap())
        .unwrap();
    restored.finish_restore().unwrap();
    // Waiting in the original, reusable here: the next shared memory takes it.
    assert!(restored.shadow().released.is_empty());
    assert_eq!(
        restored.shadow().reusable.iter().collect::<Vec<_>>(),
        [(&run.base(), &run.end())]
    );
    assert_eq!(restored.allocate_shared(PAGE_SIZE), Ok(run));
    // The original hands out new RAM while the flush is owed.
    assert_ne!(space.allocate_shared(PAGE_SIZE), Ok(run));
}

#[test]
fn an_alias_of_restored_pages_needs_no_new_ram() {
    // Every page of RAM handed out: the alias's source pages have their RAM from the snapshot,
    // so mapping the alias asks for none.
    let space = rig_with(POOL_PAGES, 2, 2);
    space.map(LOW, 2 * PAGE_SIZE, Perms::ReadWrite).unwrap();
    space.write(LOW, &[1]).unwrap();
    space.write(LOW + PAGE_SIZE, &[2]).unwrap();
    let full = space.shadow().next;
    assert_eq!(full, space.ram.end());

    let restored = rig_with(POOL_PAGES, 2, 2);
    restored
        .begin_restore(&space.save_pages().unwrap())
        .unwrap();
    restored.map(LOW, 2 * PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(
        restored.map_alias(FAR, LOW, 2 * PAGE_SIZE, Perms::ReadWrite),
        Ok(())
    );
    restored.finish_restore().unwrap();
    let mut byte = [0];
    restored.read(FAR + PAGE_SIZE, &mut byte).unwrap();
    assert_eq!(byte, [2]);
    assert_eq!(restored.shadow().next, full);
}

#[test]
fn a_shared_mapping_whose_run_is_not_saved_is_refused() {
    let scene = scene();
    let mut body: Body = decode(FORMAT, &scene.space.save_pages().unwrap()).unwrap();
    // Drop the mapped shared memory's run; the RAM stays handed out.
    body.shared.remove(0);
    let blocks = scene.kernel.save().unwrap();
    let restored = Arc::new(rig());
    let error = restore(&restored, &bytes_of(&body), &blocks, objects(&scene)).unwrap_err();
    let base = scene.mapped.backing().base();
    assert_eq!(
        error,
        RestoreError::Space(SnapshotError::Translation {
            address: SHARED,
            size: 2 * PAGE_SIZE,
            error: starquake_kernel::memory::MapperError::Refused(format!(
                "the {:#x} bytes of guest RAM at {base:#x} are not a shared memory this address \
                 space handed out",
                2 * PAGE_SIZE
            )),
            stuck: 0,
        })
    );
}

#[test]
fn a_page_outside_the_saved_blocks_is_refused_and_one_at_their_last_page_is_not() {
    let scene = scene();
    let mut body: Body = decode(FORMAT, &scene.space.save_pages().unwrap()).unwrap();
    let blocks = scene.kernel.save().unwrap();
    // The heap's fourth and last page was never touched: a page there is inside the blocks. The
    // page right after the heap is not.
    let spare = scene.space.shadow().next;
    let add = |body: &mut Body, va: u64| {
        body.next = spare + PAGE_SIZE;
        body.pages.push(SavedPage {
            va,
            pa: spare,
            contents: filled(0x5A),
        });
        body.pages.sort_by_key(|page| page.va);
    };
    let heap_end = scene.heap + 4 * PAGE_SIZE;
    let last = heap_end - PAGE_SIZE;
    let mut inside = body.clone();
    add(&mut inside, last);
    let restored = Arc::new(rig());
    restore(&restored, &bytes_of(&inside), &blocks, objects(&scene)).unwrap();
    let mut byte = [0];
    restored.read(last, &mut byte).unwrap();
    assert_eq!(byte, [0x5A]);

    add(&mut body, heap_end);
    let restored = Arc::new(rig());
    assert_eq!(
        restore(&restored, &bytes_of(&body), &blocks, objects(&scene)).err(),
        Some(RestoreError::Pages(PagesError::Outside { va: heap_end }))
    );
}

// ----- Finishing ---------------------------------------------------------------------------------

/// A snapshot of private pages at `vas`, in order, each with its own page of RAM from the
/// rig's RAM.
fn private(vas: &[u64]) -> Body {
    private_at(RAM_BASE, vas)
}

/// [`private`], for RAM that starts at `ram_base`.
fn private_at(ram_base: u64, vas: &[u64]) -> Body {
    let pages: Vec<SavedPage> = (ram_base..)
        .step_by(PAGE_STEP)
        .zip(vas)
        .zip(1..)
        .map(|((pa, va), byte)| SavedPage {
            va: *va,
            pa,
            contents: filled(byte),
        })
        .collect();
    Body {
        next: ram_base + u64::try_from(pages.len()).unwrap() * PAGE_SIZE,
        pages,
        ..empty(ram_base)
    }
}

#[test]
fn a_restored_page_gets_its_mappings_permission_and_its_ram() {
    let space = rig();
    space.begin_restore(&bytes_of(&private(&[LOW]))).unwrap();
    // Waiting, it has no table entry.
    assert!(matches!(
        space.tables.walk(LOW),
        Err(WalkError::Unmapped { .. })
    ));
    space.map(LOW, PAGE_SIZE, Perms::Read).unwrap();
    space.finish_restore().unwrap();
    assert_eq!(
        space.tables.walk(LOW).map(|r| (r.pa, r.perms)),
        Ok((RAM_BASE, Perms::Read))
    );
    assert!(space.shadow().pending.is_empty());
    let mut byte = [0];
    space.read(LOW + 7, &mut byte).unwrap();
    assert_eq!(byte, [1]);
}

#[test]
fn a_page_where_a_mapping_ends_is_refused_and_the_page_before_is_not() {
    // The mapping is two pages: its last page is accepted, the page that starts at its end is
    // refused.
    let space = rig();
    space
        .begin_restore(&bytes_of(&private(&[LOW + PAGE_SIZE])))
        .unwrap();
    space.map(LOW, 2 * PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(space.finish_restore(), Ok(()));

    let space = rig();
    space
        .begin_restore(&bytes_of(&private(&[LOW + 2 * PAGE_SIZE])))
        .unwrap();
    space.map(LOW, 2 * PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(
        space.finish_restore(),
        Err(PagesError::Outside {
            va: LOW + 2 * PAGE_SIZE
        })
    );
}

#[test]
fn the_first_page_outside_in_address_order_is_reported_and_nothing_is_mapped() {
    let space = rig();
    space
        .begin_restore(&bytes_of(&private(&[LOW, LOW + PAGE_SIZE, FAR])))
        .unwrap();
    space.map(LOW, PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(
        space.finish_restore(),
        Err(PagesError::Outside {
            va: LOW + PAGE_SIZE
        })
    );
    assert!(matches!(
        space.tables.walk(LOW),
        Err(WalkError::Unmapped { .. })
    ));
}

#[test]
fn a_page_a_shared_mapping_holds_is_taken() {
    let mut body = private(&[LOW + PAGE_SIZE]);
    body.next += PAGE_SIZE;
    let run = body.next - PAGE_SIZE;
    body.shared.push(SharedRun {
        base: run,
        pages: vec![Contents::Zero],
    });
    let space = rig();
    space.begin_restore(&bytes_of(&body)).unwrap();
    let run = BackingRange::new(run, PAGE_SIZE).unwrap();
    space.map_shared(LOW + PAGE_SIZE, run, Perms::Read).unwrap();
    assert_eq!(
        space.finish_restore(),
        Err(PagesError::Taken {
            va: LOW + PAGE_SIZE
        })
    );
}

#[test]
fn a_table_refusal_unmaps_the_pages_mapped_before_it() {
    // A pool of three pages: the root and the two tables below it for the first page; the far
    // page needs two more.
    let space = rig_with(3, RAM_PAGES, RAM_PAGES);
    let body = private_at(space.ram.base(), &[LOW, FAR]);
    space.begin_restore(&bytes_of(&body)).unwrap();
    space.map(LOW, PAGE_SIZE, Perms::ReadWrite).unwrap();
    space.map(FAR, PAGE_SIZE, Perms::ReadWrite).unwrap();
    assert_eq!(
        space.finish_restore(),
        Err(PagesError::Translation {
            va: FAR,
            error: SpaceError::Map(MapError::OutOfTablePages),
        })
    );
    assert!(matches!(
        space.tables.walk(LOW),
        Err(WalkError::Unmapped { .. })
    ));
    assert!(!space.shadow().backed.contains_key(&LOW));
}

// ----- The checks, in order ----------------------------------------------------------------------

#[test]
fn each_part_of_a_used_address_space_makes_it_not_new() {
    type Spoil = fn(&mut Shadow);
    let spoils: [(&str, Spoil); 8] = [
        ("next", |s| s.next += PAGE_SIZE),
        ("mapped", |s| {
            s.mapped.insert(
                LOW,
                crate::space::Mapped {
                    end: LOW + PAGE_SIZE,
                    perms: Perms::Read,
                },
            );
        }),
        ("backed", |s| {
            s.backed.insert(LOW, RAM_BASE);
        }),
        ("shared", |s| {
            s.shared.insert(RAM_BASE, RAM_BASE + PAGE_SIZE);
        }),
        ("released", |s| {
            s.released
                .push(BackingRange::new(RAM_BASE, PAGE_SIZE).unwrap());
        }),
        ("reusable", |s| {
            s.reusable.insert(RAM_BASE, RAM_BASE + PAGE_SIZE);
        }),
        ("aliased", |s| {
            s.aliased.insert(LOW);
        }),
        ("pending", |s| {
            s.pending.insert(LOW, RAM_BASE);
        }),
    ];
    let bytes = bytes_of(&empty(RAM_BASE));
    for (name, spoil) in spoils {
        let space = rig();
        spoil(&mut space.shadow());
        let before = state(&space);
        assert_eq!(
            space.begin_restore(&bytes),
            Err(PagesError::NotFresh),
            "{name}"
        );
        assert_eq!(state(&space), before, "{name}");
    }
    assert_eq!(rig().begin_restore(&bytes), Ok(()));
}

#[test]
fn a_snapshot_exactly_at_the_bound_is_decoded_and_one_byte_more_is_refused() {
    let limit = usize::try_from(max_snapshot_bytes(ram())).unwrap();
    let mut bytes = bytes_of(&empty(RAM_BASE));
    let body = bytes.len();
    bytes.resize(limit, 0);
    // At the bound: decoded, and refused for what follows the body.
    assert_eq!(
        rig().begin_restore(&bytes),
        Err(PagesError::Format(FormatError::TrailingBytes {
            name: "guest pages",
            count: limit - body,
        }))
    );
    bytes.push(0);
    assert_eq!(
        rig().begin_restore(&bytes),
        Err(PagesError::TooLarge {
            size: u64::try_from(limit + 1).unwrap(),
            limit: u64::try_from(limit).unwrap(),
        })
    );
}

#[test]
fn the_bound_is_the_largest_entry_for_each_page_of_ram() {
    // Eden has no snapshot; the figures are postcard's: a varint of a u64 is at most ten bytes,
    // of a u32 five, and a page's byte count (0x1000) two.
    assert_eq!(MAX_VARINT_U64, 10);
    assert_eq!(MAX_HEADER, 4 + 5);
    assert_eq!(MAX_CONTENTS, 1 + 2 + PAGE_SIZE);
    assert_eq!(varint_len(0x7F), 1);
    assert_eq!(varint_len(0x80), 2);
    assert_eq!(varint_len(u64::MAX), MAX_VARINT_U64);
    let one = BackingRange::new(RAM_BASE, PAGE_SIZE).unwrap();
    assert_eq!(max_snapshot_bytes(one), 9 + 40 + 20 + 3 + PAGE_SIZE);
    assert_eq!(
        max_snapshot_bytes(ram()),
        9 + 40 + RAM_PAGES * (20 + 3 + PAGE_SIZE)
    );
}

#[test]
fn a_snapshot_of_every_page_of_ram_is_within_the_bound() {
    // Every page of RAM holding bytes, each kind of entry: private pages near the top of the
    // address space (the largest addresses a page can have), and a shared run of one page.
    let space = rig();
    let top = VA_SPACE_END - (RAM_PAGES - 1) * PAGE_SIZE;
    space
        .map(top, (RAM_PAGES - 1) * PAGE_SIZE, Perms::ReadWrite)
        .unwrap();
    for page in 0..RAM_PAGES - 1 {
        space.write(top + page * PAGE_SIZE, &[0xEE; 8]).unwrap();
    }
    let run = space.allocate_shared(PAGE_SIZE).unwrap();
    space.tables.memory().write(run.base(), &[0xDD]).unwrap();
    assert_eq!(space.shadow().next, RAM_END);
    let bytes = space.save_pages().unwrap();
    assert!(
        u64::try_from(bytes.len()).unwrap() <= max_snapshot_bytes(ram()),
        "{} bytes",
        bytes.len()
    );
    // And it restores.
    let restored = rig();
    restored.begin_restore(&bytes).unwrap();
    restored
        .map(top, (RAM_PAGES - 1) * PAGE_SIZE, Perms::ReadWrite)
        .unwrap();
    restored.finish_restore().unwrap();
    assert_eq!(ram_bytes(&restored), ram_bytes(&space));
}

#[test]
fn next_must_be_a_page_of_ram_up_to_its_end() {
    let cases = [
        (RAM_BASE, Ok(())),
        (RAM_BASE + 1, Err(PagesError::Next { next: RAM_BASE + 1 })),
        (
            RAM_BASE - PAGE_SIZE,
            Err(PagesError::Next {
                next: RAM_BASE - PAGE_SIZE,
            }),
        ),
        (RAM_END, Ok(())),
        (
            RAM_END + PAGE_SIZE,
            Err(PagesError::PastRam {
                next: RAM_END + PAGE_SIZE,
                end: RAM_END,
            }),
        ),
    ];
    for (next, expected) in cases {
        assert_eq!(begin(&empty(next)), expected, "{next:#x}");
    }
}

/// A snapshot with shared runs `runs` (each base and page count, pages zero) and `next` at the
/// end of RAM.
fn with_shared(runs: &[(u64, usize)]) -> Body {
    Body {
        shared: runs
            .iter()
            .map(|(base, pages)| SharedRun {
                base: *base,
                pages: vec![Contents::Zero; *pages],
            })
            .collect(),
        ..empty(RAM_END)
    }
}

fn shared_fault(index: usize, fault: RunFault) -> Result<(), PagesError> {
    Err(PagesError::Shared { index, fault })
}

#[test]
fn each_shared_run_is_checked_in_order_with_each_edge() {
    let cases: [(&str, Body, Result<(), PagesError>); 9] = [
        (
            "misaligned",
            with_shared(&[(RAM_BASE + 1, 1)]),
            shared_fault(0, RunFault::Shape),
        ),
        (
            "no pages",
            with_shared(&[(RAM_BASE, 0)]),
            shared_fault(0, RunFault::Shape),
        ),
        (
            "wraps",
            with_shared(&[(u64::MAX - PAGE_SIZE + 1, 1)]),
            shared_fault(0, RunFault::Shape),
        ),
        (
            "below RAM",
            with_shared(&[(RAM_BASE - PAGE_SIZE, 1)]),
            shared_fault(0, RunFault::Ram),
        ),
        (
            "ends at next",
            with_shared(&[(RAM_END - PAGE_SIZE, 1)]),
            Ok(()),
        ),
        (
            "ends past next",
            Body {
                next: RAM_END - PAGE_SIZE,
                ..with_shared(&[(RAM_END - PAGE_SIZE, 1)])
            },
            shared_fault(0, RunFault::Ram),
        ),
        (
            "side by side",
            with_shared(&[(RAM_BASE, 1), (RAM_BASE + PAGE_SIZE, 1)]),
            Ok(()),
        ),
        (
            "overlaps the one before",
            with_shared(&[(RAM_BASE, 2), (RAM_BASE + PAGE_SIZE, 1)]),
            shared_fault(1, RunFault::Order),
        ),
        (
            "before the one before",
            with_shared(&[(RAM_BASE + PAGE_SIZE, 1), (RAM_BASE, 1)]),
            shared_fault(1, RunFault::Order),
        ),
    ];
    for (name, body, expected) in cases {
        assert_eq!(begin(&body), expected, "{name}");
    }
}

#[test]
fn a_shared_pages_bytes_must_be_exactly_a_page() {
    for (len, expected) in [
        (PAGE_STEP - 1, shared_fault(0, RunFault::Length)),
        (PAGE_STEP, Ok(())),
        (PAGE_STEP + 1, shared_fault(0, RunFault::Length)),
    ] {
        let mut body = with_shared(&[(RAM_BASE, 2)]);
        body.shared[0].pages[1] = Contents::Bytes(PageBytes(vec![7; len]));
        assert_eq!(begin(&body), expected, "{len}");
    }
}

#[test]
fn a_runs_checks_come_in_their_order_and_the_first_run_first() {
    // Shape before RAM: misaligned and below RAM.
    assert_eq!(
        begin(&with_shared(&[(RAM_BASE - PAGE_SIZE + 1, 1)])),
        shared_fault(0, RunFault::Shape)
    );
    // RAM before order: below RAM and before the one before.
    assert_eq!(
        begin(&with_shared(&[(RAM_BASE, 1), (RAM_BASE - PAGE_SIZE, 1)])),
        shared_fault(1, RunFault::Ram)
    );
    // Order before length.
    let mut body = with_shared(&[(RAM_BASE, 2), (RAM_BASE + PAGE_SIZE, 1)]);
    body.shared[1].pages[0] = Contents::Bytes(PageBytes(vec![1]));
    assert_eq!(begin(&body), shared_fault(1, RunFault::Order));
    // Two faulty runs: the first is reported.
    assert_eq!(
        begin(&with_shared(&[(RAM_BASE + 1, 1), (RAM_BASE, 0)])),
        shared_fault(0, RunFault::Shape)
    );
    // `next` before the runs: both wrong.
    let body = Body {
        next: RAM_END + PAGE_SIZE,
        ..with_shared(&[(RAM_BASE + 1, 1)])
    };
    assert_eq!(
        begin(&body),
        Err(PagesError::PastRam {
            next: RAM_END + PAGE_SIZE,
            end: RAM_END
        })
    );
}

/// A snapshot with given-back runs `runs` (each base and size), one shared run of one page at
/// `RAM_BASE + 4 pages`, and `next` at the end of RAM.
fn with_free(runs: &[(u64, u64)]) -> Body {
    Body {
        free: runs
            .iter()
            .map(|(base, size)| FreeRun {
                base: *base,
                size: *size,
            })
            .collect(),
        ..with_shared(&[(SHARED_RUN, 1)])
    }
}

/// Where [`with_free`]'s shared run lies.
const SHARED_RUN: u64 = RAM_BASE + 4 * PAGE_SIZE;

fn free_fault(index: usize, fault: RunFault) -> Result<(), PagesError> {
    Err(PagesError::Free { index, fault })
}

#[test]
fn each_given_back_run_is_checked_in_order_with_each_edge() {
    let cases: [(&str, Body, Result<(), PagesError>); 12] = [
        (
            "misaligned base",
            with_free(&[(RAM_BASE + 1, PAGE_SIZE)]),
            free_fault(0, RunFault::Shape),
        ),
        (
            "misaligned size",
            with_free(&[(RAM_BASE, PAGE_SIZE + 1)]),
            free_fault(0, RunFault::Shape),
        ),
        (
            "empty",
            with_free(&[(RAM_BASE, 0)]),
            free_fault(0, RunFault::Shape),
        ),
        (
            "wraps",
            with_free(&[(u64::MAX - PAGE_SIZE + 1, PAGE_SIZE)]),
            free_fault(0, RunFault::Shape),
        ),
        (
            "below RAM",
            with_free(&[(RAM_BASE - PAGE_SIZE, PAGE_SIZE)]),
            free_fault(0, RunFault::Ram),
        ),
        (
            "ends at next",
            with_free(&[(RAM_END - PAGE_SIZE, PAGE_SIZE)]),
            Ok(()),
        ),
        (
            "ends past next",
            with_free(&[(RAM_END - PAGE_SIZE, 2 * PAGE_SIZE)]),
            free_fault(0, RunFault::Ram),
        ),
        (
            "overlaps the one before",
            with_free(&[(RAM_BASE, 2 * PAGE_SIZE), (RAM_BASE + PAGE_SIZE, PAGE_SIZE)]),
            free_fault(1, RunFault::Order),
        ),
        (
            "ends where the shared run starts",
            with_free(&[(SHARED_RUN - PAGE_SIZE, PAGE_SIZE)]),
            Ok(()),
        ),
        (
            "starts where the shared run ends",
            with_free(&[(SHARED_RUN + PAGE_SIZE, PAGE_SIZE)]),
            Ok(()),
        ),
        (
            "holds the shared run's page",
            with_free(&[(SHARED_RUN - PAGE_SIZE, 2 * PAGE_SIZE)]),
            free_fault(0, RunFault::Shared),
        ),
        (
            "order before shared",
            with_free(&[(SHARED_RUN + PAGE_SIZE, PAGE_SIZE), (SHARED_RUN, PAGE_SIZE)]),
            free_fault(1, RunFault::Order),
        ),
    ];
    for (name, body, expected) in cases {
        assert_eq!(begin(&body), expected, "{name}");
    }
    // The runs before the given-back ones: both faulty.
    let mut body = with_free(&[(RAM_BASE, 0)]);
    body.shared[0].base += 1;
    assert_eq!(begin(&body), shared_fault(0, RunFault::Shape));
}

/// A snapshot of `pages` (each va, pa and contents), a shared run of one page at
/// [`SHARED_RUN`], a given-back run of one page after it, and `next` at the end of RAM.
fn with_pages(pages: &[(u64, u64, Contents)]) -> Body {
    Body {
        pages: pages
            .iter()
            .map(|(va, pa, contents)| SavedPage {
                va: *va,
                pa: *pa,
                contents: contents.clone(),
            })
            .collect(),
        ..with_free(&[(SHARED_RUN + PAGE_SIZE, PAGE_SIZE)])
    }
}

fn page_fault(index: usize, fault: PageFault) -> Result<(), PagesError> {
    Err(PagesError::Page { index, fault })
}

#[test]
fn each_private_page_is_checked_in_order_with_each_edge() {
    let z = || Contents::Zero;
    let top = VA_SPACE_END - PAGE_SIZE;
    let cases: [(&str, Body, Result<(), PagesError>); 15] = [
        (
            "misaligned va",
            with_pages(&[(LOW + 1, RAM_BASE, z())]),
            page_fault(0, PageFault::Shape),
        ),
        (
            "misaligned pa",
            with_pages(&[(LOW, RAM_BASE + 1, z())]),
            page_fault(0, PageFault::Shape),
        ),
        ("the last page", with_pages(&[(top, RAM_BASE, z())]), Ok(())),
        (
            "past the address space",
            with_pages(&[(VA_SPACE_END, RAM_BASE, z())]),
            page_fault(0, PageFault::Shape),
        ),
        (
            "wraps",
            with_pages(&[(u64::MAX - PAGE_SIZE + 1, RAM_BASE, z())]),
            page_fault(0, PageFault::Shape),
        ),
        (
            "same va",
            with_pages(&[(LOW, RAM_BASE, z()), (LOW, RAM_BASE + PAGE_SIZE, z())]),
            page_fault(1, PageFault::Order),
        ),
        (
            "lower va",
            with_pages(&[(FAR, RAM_BASE, z()), (LOW, RAM_BASE + PAGE_SIZE, z())]),
            page_fault(1, PageFault::Order),
        ),
        (
            "below RAM",
            with_pages(&[(LOW, RAM_BASE - PAGE_SIZE, z())]),
            page_fault(0, PageFault::Ram),
        ),
        (
            "the last page handed out",
            with_pages(&[(LOW, RAM_END - PAGE_SIZE, z())]),
            Ok(()),
        ),
        (
            "past next",
            Body {
                next: RAM_END - PAGE_SIZE,
                ..with_pages(&[(LOW, RAM_END - PAGE_SIZE, z())])
            },
            page_fault(0, PageFault::Ram),
        ),
        (
            "in the shared run",
            with_pages(&[(LOW, SHARED_RUN, z())]),
            page_fault(0, PageFault::Run),
        ),
        (
            "in the given-back run",
            with_pages(&[(LOW, SHARED_RUN + PAGE_SIZE, z())]),
            page_fault(0, PageFault::Run),
        ),
        (
            "just past the runs",
            with_pages(&[(LOW, SHARED_RUN + 2 * PAGE_SIZE, z())]),
            Ok(()),
        ),
        (
            "an earlier page's RAM",
            with_pages(&[(LOW, RAM_BASE, z()), (FAR, RAM_BASE, z())]),
            page_fault(1, PageFault::Duplicate),
        ),
        (
            "short bytes",
            with_pages(&[(LOW, RAM_BASE, Contents::Bytes(PageBytes(vec![1; 8])))]),
            page_fault(0, PageFault::Length),
        ),
    ];
    for (name, body, expected) in cases {
        assert_eq!(begin(&body), expected, "{name}");
    }
}

#[test]
fn a_pages_checks_come_in_their_order_and_the_first_page_first() {
    let short = || Contents::Bytes(PageBytes(vec![1]));
    let z = || Contents::Zero;
    let both = [
        // Shape before order.
        (
            with_pages(&[(FAR, RAM_BASE, z()), (LOW + 1, RAM_BASE + PAGE_SIZE, z())]),
            page_fault(1, PageFault::Shape),
        ),
        // Order before RAM.
        (
            with_pages(&[(FAR, RAM_BASE, z()), (LOW, RAM_END, z())]),
            page_fault(1, PageFault::Order),
        ),
        // The first page first: both in a run, the second also the first's RAM.
        (
            with_pages(&[(LOW, SHARED_RUN, z()), (FAR, SHARED_RUN, z())]),
            page_fault(0, PageFault::Run),
        ),
        // Duplicate before length.
        (
            with_pages(&[(LOW, RAM_BASE, z()), (FAR, RAM_BASE, short())]),
            page_fault(1, PageFault::Duplicate),
        ),
        // RAM before length.
        (
            with_pages(&[(LOW, RAM_END, short())]),
            page_fault(0, PageFault::Ram),
        ),
        // Two faulty pages: the first is reported.
        (
            with_pages(&[(LOW, RAM_BASE, short()), (FAR + 1, RAM_BASE, z())]),
            page_fault(0, PageFault::Length),
        ),
        // The runs before the pages.
        (
            Body {
                free: vec![FreeRun {
                    base: RAM_BASE,
                    size: 0,
                }],
                ..with_pages(&[(LOW + 1, RAM_BASE, z())])
            },
            free_fault(0, RunFault::Shape),
        ),
    ];
    for (body, expected) in both {
        assert_eq!(begin(&body), expected);
    }
}

#[test]
fn the_length_comes_before_the_header_and_the_space_before_the_length() {
    let space = rig();
    space.shadow().next += PAGE_SIZE;
    let too_long = vec![b'X'; usize::try_from(max_snapshot_bytes(ram())).unwrap() + 1];
    assert_eq!(space.begin_restore(&too_long), Err(PagesError::NotFresh));
    assert!(matches!(
        rig().begin_restore(&too_long),
        Err(PagesError::TooLarge { .. })
    ));
    // The encoding before the body's checks: a body with a bad `next` and trailing bytes.
    let mut bytes = bytes_of(&empty(RAM_BASE + 1));
    bytes.push(0);
    assert_eq!(
        rig().begin_restore(&bytes),
        Err(PagesError::Format(FormatError::TrailingBytes {
            name: "guest pages",
            count: 1
        }))
    );
}

#[test]
fn a_refused_restore_leaves_the_space_new_and_the_next_one_succeeds() {
    let space = rig();
    let mut body = private(&[LOW]);
    body.pages[0].va += 1;
    assert!(space.begin_restore(&bytes_of(&body)).is_err());
    assert_eq!(state(&space), new_state());
    assert_eq!(ram_bytes(&space), ram_bytes(&rig()));
    assert_eq!(space.begin_restore(&bytes_of(&private(&[LOW]))), Ok(()));
}

#[test]
fn a_write_ram_refuses_zeroes_the_pages_written_and_leaves_the_space_new() {
    // RAM of four pages, of which the flat memory holds two: the third page cannot be written.
    let space = rig_with(POOL_PAGES, 4, 2);
    let base = space.ram.base();
    let body = Body {
        next: base + 3 * PAGE_SIZE,
        shared: vec![SharedRun {
            base,
            pages: vec![filled(1)],
        }],
        free: Vec::new(),
        pages: vec![
            SavedPage {
                va: LOW,
                pa: base + PAGE_SIZE,
                contents: filled(2),
            },
            SavedPage {
                va: FAR,
                pa: base + 2 * PAGE_SIZE,
                contents: filled(3),
            },
        ],
    };
    let fresh = state(&space);
    assert_eq!(
        space.begin_restore(&bytes_of(&body)),
        Err(PagesError::Ram(Fault {
            address: base + 2 * PAGE_SIZE
        }))
    );
    assert_eq!(state(&space), fresh);
    let mut written = vec![0xFF; 2 * PAGE_STEP];
    space.tables.memory().read(base, &mut written).unwrap();
    assert!(written.iter().all(|byte| *byte == 0));
    // A snapshot that fits is restored.
    body_without_last(&space, body);
}

/// Restores `body` without its last page into `space`, and checks the bytes arrived.
fn body_without_last(space: &AddressSpace<FlatMemory>, mut body: Body) {
    body.pages.pop();
    body.next -= PAGE_SIZE;
    assert_eq!(space.begin_restore(&bytes_of(&body)), Ok(()));
    let mut byte = [0];
    space
        .tables
        .memory()
        .read(space.ram.base() + PAGE_SIZE, &mut byte)
        .unwrap();
    assert_eq!(byte, [2]);
}

// ----- The save-side checks ----------------------------------------------------------------------

#[test]
fn a_save_part_way_through_a_restore_is_refused() {
    let space = rig();
    space.begin_restore(&bytes_of(&private(&[LOW]))).unwrap();
    assert_eq!(space.save_pages(), Err(PagesError::Unfinished));
    space.map(LOW, PAGE_SIZE, Perms::Read).unwrap();
    space.finish_restore().unwrap();
    assert!(space.save_pages().is_ok());
}

#[test]
fn a_save_refuses_an_alias_of_ram_it_does_not_keep() {
    let space = rig();
    space.map(LOW, PAGE_SIZE, Perms::ReadWrite).unwrap();
    space.write(LOW, &[1]).unwrap();
    space
        .map_alias(FAR, LOW, PAGE_SIZE, Perms::ReadWrite)
        .unwrap();
    assert!(space.save_pages().is_ok());
    // The source's page forgotten: the alias maps RAM no saved page holds.
    space.shadow().backed.remove(&LOW);
    assert_eq!(space.save_pages(), Err(PagesError::Alias { va: FAR }));
    // An alias page with no backing at all.
    let space = rig();
    space.shadow().aliased.insert(FAR);
    assert_eq!(space.save_pages(), Err(PagesError::Alias { va: FAR }));
}

#[test]
fn a_save_refuses_a_page_whose_ram_was_given_back() {
    let space = rig();
    let run = space.allocate_shared(PAGE_SIZE).unwrap();
    space.release_shared(run).unwrap();
    space.map(LOW, PAGE_SIZE, Perms::Read).unwrap();
    // Corrupt: a backed page whose RAM is the given-back run.
    space.shadow().backed.insert(LOW, run.base());
    assert_eq!(
        space.save_pages(),
        Err(PagesError::Page {
            index: 0,
            fault: PageFault::Run
        })
    );
}

#[test]
fn a_save_refuses_ram_handed_out_past_the_end() {
    let space = rig();
    space.shadow().next = RAM_END + PAGE_SIZE;
    assert_eq!(
        space.save_pages(),
        Err(PagesError::PastRam {
            next: RAM_END + PAGE_SIZE,
            end: RAM_END
        })
    );
}

#[test]
fn every_error_reads_as_plain_words() {
    let cases = [
        (
            PagesError::NotFresh,
            "guest pages can only be restored into a new address space",
        ),
        (
            PagesError::TooLarge { size: 10, limit: 9 },
            "the guest pages snapshot is 10 bytes, more than the 9 bytes one of this guest RAM \
             can take",
        ),
        (
            PagesError::Next { next: 0x1001 },
            "the end of the guest RAM handed out, 0x1001, is not a page of guest RAM",
        ),
        (
            PagesError::PastRam {
                next: 0x3000,
                end: 0x2000,
            },
            "the snapshot hands out guest RAM up to 0x3000, past the end of guest RAM at 0x2000",
        ),
        (
            PagesError::Shared {
                index: 0,
                fault: RunFault::Length,
            },
            "shared memory run 0 has a page whose bytes are not a page long",
        ),
        (
            PagesError::Free {
                index: 1,
                fault: RunFault::Shared,
            },
            "given-back run 1 shares a page with a shared memory's run",
        ),
        (
            PagesError::Page {
                index: 2,
                fault: PageFault::Duplicate,
            },
            "saved page 2 has the RAM of an earlier page",
        ),
        (
            PagesError::Ram(Fault { address: 0x10 }),
            "guest RAM refused an access while its pages were saved or restored: guest memory \
             fault at 0x10",
        ),
        (
            PagesError::Outside { va: 0x4000 },
            "the restored page at 0x4000 lies outside every mapping the blocks made",
        ),
        (
            PagesError::Taken { va: 0x4000 },
            "the restored page at 0x4000 was mapped already, with other RAM",
        ),
        (
            PagesError::Translation {
                va: 0x4000,
                error: SpaceError::Map(MapError::OutOfTablePages),
            },
            "the restored page at 0x4000 could not be mapped: the guest's translation tables \
             refused a mapping: out of translation table pages",
        ),
        (
            PagesError::Unfinished,
            "the address space is part-way through a restore",
        ),
        (
            PagesError::Alias { va: 0x5000 },
            "the alias page at 0x5000 maps RAM no saved page holds",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
    for (fault, text) in [
        (RunFault::Shape, "is not a run of whole pages"),
        (RunFault::Ram, "lies outside the guest RAM handed out"),
        (
            RunFault::Order,
            "starts before the end of the one before it",
        ),
    ] {
        assert_eq!(fault.to_string(), text);
    }
    for (fault, text) in [
        (PageFault::Shape, "is not a page of the address space"),
        (PageFault::Order, "is not above the one before it"),
        (PageFault::Ram, "has RAM outside the guest RAM handed out"),
        (PageFault::Run, "has RAM inside a run"),
        (PageFault::Length, "has bytes that are not a page long"),
    ] {
        assert_eq!(fault.to_string(), text);
    }
}
