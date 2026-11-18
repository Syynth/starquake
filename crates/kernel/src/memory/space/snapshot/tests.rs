//! The address space snapshot: a round trip of a space with every kind of block the kernel
//! makes, the restored space in use, the layout pinned by golden bytes, one fault case per
//! check with its exact error, pairs of faults for the documented order, and generated round
//! trips of live address spaces.

use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use super::*;
use crate::id::SlabId;
use crate::memory::fixture::{CODE, DATA, RODATA, SHARED, loaded, shared_memory};
use crate::memory::layout::LayoutError;
use crate::memory::state::SvcPermission;
use starquake_memory::Perms;

use crate::memory::testing::{MapperCall, MapperLog, RecordingMapper};
use crate::object::Objects;
use crate::result::{CallResult, KernelError, SvcError};
use crate::snapshot::FormatError;

mod alias;
mod attribute;

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
const PROPTEST_SEED: u64 = 0x5351_4153_534E_4150;

fn config(cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(PROPTEST_SEED),
        failure_persistence: None,
        ..Config::default()
    }
}

/// The shared memory each object id names, as a restored object table would give it.
type Resolver = BTreeMap<ObjectId, SharedMemory>;

/// An address space with every kind of block the kernel makes, and what restoring it needs.
struct Rich {
    space: AddressSpace,
    log: MapperLog,
    objects: Resolver,
    /// The two shared memories, in the order they are mapped.
    shared: [ObjectId; 2],
    heap: u64,
    alias: u64,
    /// The first of the free pages near the stack region's top that `MapMemory` aliases go to
    /// ([`stack_top`]).
    stack: u64,
}

/// Page `n` of the 64 pages below the stack region's top: free in the fixture's loaded space,
/// whose main thread's stack is at the region's bottom.
fn stack_top(space: &AddressSpace, n: u64) -> u64 {
    page(space.regions().stack.end - 64 * PAGE_SIZE, n)
}

/// Page `index` from `start`.
const fn page(start: u64, index: u64) -> u64 {
    start + index * PAGE_SIZE
}

/// Where the rich space's generated code goes: above its two shared memory mappings.
const GENERATED: u64 = SHARED + 8 * PAGE_SIZE;

/// The fixture's loaded address space with: 16 pages of heap; mapped physical memory in two
/// runs at the alias region's start; device-share locks taken twice on heap pages 2-3, once on
/// heap pages 3-5 (overlapping the first) and once on the first alias page; heap pages 8-9
/// GPU accessible and page 9 uncached too; heap pages 12-13 lent to a transfer memory; two
/// shared memory mappings side by side; a thread-local page; two pages of generated code; the
/// data segment's first page permission locked; and two `MapMemory` aliases, heap page 6 at
/// stack page 48 ([`stack_top`]), device-share locked once there, and heap pages 10-11 at
/// stack pages 50-51. The mapper log is empty.
fn rich() -> Rich {
    let (mut space, log, _) = loaded();
    let heap = space.regions().heap.start;
    let alias = space.regions().alias.start;
    let stack = stack_top(&space, 0);
    space.set_heap_size(16 * PAGE_SIZE).unwrap();
    space.map_physical_memory(alias, 2 * PAGE_SIZE).unwrap();
    space
        .map_physical_memory(page(alias, 4), PAGE_SIZE)
        .unwrap();
    for (address, pages) in [(page(heap, 2), 2), (page(heap, 2), 2), (page(heap, 3), 3)] {
        space.lock_device_share(address, pages * PAGE_SIZE).unwrap();
    }
    space.lock_device_share(alias, PAGE_SIZE).unwrap();
    let gpu = MemoryAttribute::GPU_ACCESSIBLE;
    space
        .set_memory_attribute(page(heap, 8), 2 * PAGE_SIZE, gpu, gpu)
        .unwrap();
    let uncached = MemoryAttribute::UNCACHED;
    space
        .set_memory_attribute(page(heap, 9), PAGE_SIZE, uncached, uncached)
        .unwrap();
    space
        .lock_for_transfer_memory(page(heap, 12), 2 * PAGE_SIZE, SvcPermission::Read)
        .unwrap();
    let mut table = Objects::new();
    let mut objects = Resolver::new();
    let mut shared = [None; 2];
    for (slot, (address, pages, permission)) in shared.iter_mut().zip([
        (SHARED, 2, SvcPermission::ReadWrite),
        (page(SHARED, 2), 1, SvcPermission::Read),
    ]) {
        let (object, pages) = shared_memory(&mut space, &mut table, pages, permission);
        let memory = SharedMemory::new(pages.backing(), permission).unwrap();
        space
            .map_shared(address, object, memory, permission)
            .unwrap();
        objects.insert(object, memory);
        *slot = Some(object);
    }
    space
        .map_pages_in_region(
            1,
            MemoryState::ThreadLocal,
            MemoryPermission::USER_READ_WRITE,
        )
        .unwrap();
    space
        .map_pages(
            GENERATED,
            2,
            MemoryState::GeneratedCode,
            MemoryPermission::USER_READ_EXECUTE,
        )
        .unwrap();
    let locked = MemoryAttribute::PERMISSION_LOCKED;
    space
        .set_memory_attribute(DATA, PAGE_SIZE, locked, locked)
        .unwrap();
    space
        .map_memory(page(stack, 48), page(heap, 6), PAGE_SIZE)
        .unwrap();
    space.lock_device_share(page(stack, 48), PAGE_SIZE).unwrap();
    space
        .map_memory(page(stack, 50), page(heap, 10), 2 * PAGE_SIZE)
        .unwrap();
    drop(log.take());
    Rich {
        space,
        log,
        objects,
        shared: shared.map(Option::unwrap),
        heap,
        alias,
        stack,
    }
}

/// `bytes` restored over a new recording mapper, with `objects` resolving shared memory.
fn restore_with(
    bytes: &[u8],
    objects: &Resolver,
) -> (Result<AddressSpace, SnapshotError>, MapperLog) {
    let (mapper, log) = RecordingMapper::new();
    let restored = AddressSpace::restore(bytes, Box::new(mapper), |object| {
        objects.get(&object).copied()
    });
    (restored, log)
}

/// The calls a restore makes of its mapper for `body`'s blocks: each block that is not free,
/// in address order, without the pages an alias covers (found page by page here, not by the
/// restore's walk), then each alias in address order.
fn expected_calls(body: &Body, objects: &Resolver) -> Vec<MapperCall> {
    let backing: BTreeMap<u64, _> = body
        .shared
        .iter()
        .map(|mapping| (mapping.address, objects[&mapping.object].backing()))
        .collect();
    let aliased: BTreeSet<u64> = body
        .aliases
        .iter()
        .flat_map(|alias| (0..alias.size / PAGE_SIZE).map(|n| page(alias.address, n)))
        .collect();
    let mut calls = Vec::new();
    for block in body
        .blocks
        .iter()
        .filter(|block| block.state != MemoryState::Free.word())
    {
        let address = block.address;
        let perms = MemoryPermission::from_bits(block.permission)
            .guest()
            .unwrap();
        if let Some(&backing) = backing.get(&address) {
            calls.push(MapperCall::MapShared {
                address,
                backing,
                perms,
            });
            continue;
        }
        // Each run of pages no alias covers.
        let mut run: Option<(u64, u64)> = None;
        for n in 0..block.pages {
            let at = page(address, n);
            match run {
                _ if aliased.contains(&at) => {
                    if let Some((address, size)) = run.take() {
                        calls.push(MapperCall::Map {
                            address,
                            size,
                            perms,
                        });
                    }
                }
                Some((_, ref mut size)) => *size += PAGE_SIZE,
                None => run = Some((at, PAGE_SIZE)),
            }
        }
        if let Some((address, size)) = run {
            calls.push(MapperCall::Map {
                address,
                size,
                perms,
            });
        }
    }
    calls.extend(body.aliases.iter().map(|alias| MapperCall::MapAlias {
        address: alias.address,
        source: alias.source,
        size: alias.size,
        perms: Perms::ReadWrite,
    }));
    calls
}

/// The index of the saved block that starts at `address`.
fn at(body: &Body, address: u64) -> usize {
    body.blocks
        .iter()
        .position(|block| block.address == address)
        .unwrap()
}

// ----- Round trip ----------------------------------------------------------------------------

#[test]
fn a_rich_space_saves_and_restores_as_it_was() {
    let rich = rich();
    let bytes = rich.space.save().unwrap();
    assert_eq!(bytes.get(..4), Some(&ADDRESS_SPACE_MAGIC[..]));
    let (restored, log) = restore_with(&bytes, &rich.objects);
    let restored = restored.unwrap();
    let body = Body::of(&rich.space);
    assert_eq!(Body::of(&restored), body);
    // Every kind of block the kernel makes is in it.
    let states: BTreeSet<u32> = body.blocks.iter().map(|block| block.state).collect();
    for state in [
        MemoryState::Free,
        MemoryState::Code,
        MemoryState::CodeData,
        MemoryState::Normal,
        MemoryState::Shared,
        MemoryState::Stack,
        MemoryState::ThreadLocal,
        MemoryState::GeneratedCode,
    ] {
        assert!(states.contains(&state.word()), "no {} block", state.name());
    }
    // The counts the address space keeps beside its blocks.
    let (heap, alias) = (rich.heap, rich.alias);
    assert_eq!(restored.device_locks(page(heap, 2), 2 * PAGE_SIZE), 2);
    assert_eq!(restored.device_locks(page(heap, 3), 3 * PAGE_SIZE), 1);
    assert_eq!(restored.device_locks(alias, PAGE_SIZE), 1);
    assert_eq!(restored.mapped_physical_memory_size(), 3 * PAGE_SIZE);
    assert_eq!(restored.heap_size(), 16 * PAGE_SIZE);
    assert_eq!(restored.max_heap_size(), rich.space.max_heap_size());
    assert_eq!(restored.limit(), rich.space.limit());
    assert_eq!(
        restored.max_process_memory(),
        rich.space.max_process_memory()
    );
    assert_eq!(
        restored.total_user_physical_memory_size(),
        rich.space.total_user_physical_memory_size()
    );
    assert_eq!(restored.regions(), rich.space.regions());
    // The new translation was given every block that is not free, in address order.
    assert_eq!(log.take(), expected_calls(&body, &rich.objects));
}

/// One step a test takes on two address spaces: what it is, what it does, and its result.
type Step = (
    &'static str,
    Box<dyn Fn(&mut AddressSpace) -> CallResult<()>>,
    CallResult<()>,
);

#[test]
fn the_restored_space_is_used_as_the_original_is() {
    let mut rich = rich();
    let bytes = rich.space.save().unwrap();
    let (restored, restored_log) = restore_with(&bytes, &rich.objects);
    let mut restored = restored.unwrap();
    drop(restored_log.take());
    let (heap, alias, stack) = (rich.heap, rich.alias, rich.stack);
    let [first, second] = rich.shared;
    let invalid: CallResult<()> = Err(SvcError::Kernel(KernelError::InvalidCurrentMemory));
    // The same steps on both, each with the result it should have.
    let mut steps: Vec<Step> = vec![
        (
            "unlock the twice-locked range",
            Box::new(move |s| s.unlock_device_share(page(heap, 2), 2 * PAGE_SIZE)),
            Ok(()),
        ),
        (
            "unlock it again",
            Box::new(move |s| s.unlock_device_share(page(heap, 2), 2 * PAGE_SIZE)),
            Ok(()),
        ),
        (
            "a third unlock has no lock left",
            Box::new(move |s| s.unlock_device_share(page(heap, 2), 2 * PAGE_SIZE)),
            invalid.clone(),
        ),
        (
            "map more physical memory beside what is there",
            Box::new(move |s| s.map_physical_memory(page(alias, 2), 2 * PAGE_SIZE)),
            Ok(()),
        ),
        (
            "unmapping physical memory under a device lock is refused",
            Box::new(move |s| s.unmap_physical_memory(alias, 5 * PAGE_SIZE)),
            invalid.clone(),
        ),
        (
            "unmap the physical memory past the lock",
            Box::new(move |s| s.unmap_physical_memory(page(alias, 1), 4 * PAGE_SIZE)),
            Ok(()),
        ),
        (
            "unmap the second shared memory",
            Box::new(move |s| s.unmap_shared(page(SHARED, 2), PAGE_SIZE, second)),
            Ok(()),
        ),
        (
            "the first shared memory is not mapped where the second was",
            Box::new(move |s| s.unmap_shared(page(SHARED, 2), PAGE_SIZE, first)),
            invalid.clone(),
        ),
        (
            "take back the lent pages",
            Box::new(move |s| s.unlock_for_transfer_memory(page(heap, 12), 2 * PAGE_SIZE)),
            Ok(()),
        ),
    ];
    steps.extend(alias_steps(heap, stack));
    steps.extend::<[Step; 2]>([
        (
            "grow the heap",
            Box::new(|s| s.set_heap_size(32 * PAGE_SIZE).map(drop)),
            Ok(()),
        ),
        (
            "lock a heap range again",
            Box::new(move |s| s.lock_device_share(page(heap, 20), 4 * PAGE_SIZE)),
            Ok(()),
        ),
    ]);
    for (what, step, expected) in &steps {
        assert_eq!(step(&mut rich.space), *expected, "original: {what}");
        assert_eq!(step(&mut restored), *expected, "restored: {what}");
        assert_eq!(Body::of(&restored), Body::of(&rich.space), "after: {what}");
        for address in [
            page(heap, 2),
            page(alias, 1),
            page(SHARED, 2),
            page(heap, 12),
            page(heap, 6),
            page(heap, 11),
            page(stack, 48),
            page(stack, 51),
        ] {
            assert_eq!(
                restored.query_info(address).unwrap(),
                rich.space.query_info(address).unwrap(),
                "query of {address:#x} after: {what}"
            );
        }
        assert_eq!(
            restored.aliases.runs(),
            rich.space.aliases.runs(),
            "aliases after: {what}"
        );
    }
    // Both translations were asked the same things.
    let calls = rich.log.take();
    assert_ne!(calls.len(), 0);
    assert_eq!(restored_log.take(), calls);
}

/// The steps of `the_restored_space_is_used_as_the_original_is` on the rich space's aliases.
fn alias_steps(heap: u64, stack: u64) -> Vec<Step> {
    let invalid: CallResult<()> = Err(SvcError::Kernel(KernelError::InvalidCurrentMemory));
    vec![
        (
            "unmapping the alias a device holds is refused",
            Box::new(move |s| s.unmap_memory(page(stack, 48), page(heap, 6), PAGE_SIZE)),
            invalid,
        ),
        (
            "unmapping the second alias as the first's source is refused",
            Box::new(move |s| s.unmap_memory(page(stack, 50), page(heap, 6), PAGE_SIZE)),
            Err(SvcError::Kernel(KernelError::InvalidMemoryRegion)),
        ),
        (
            "unmap the second alias's last page",
            Box::new(move |s| s.unmap_memory(page(stack, 51), page(heap, 11), PAGE_SIZE)),
            Ok(()),
        ),
        (
            "the device lets go of the first alias",
            Box::new(move |s| s.unlock_device_share(page(stack, 48), PAGE_SIZE)),
            Ok(()),
        ),
        (
            "unmap the first alias",
            Box::new(move |s| s.unmap_memory(page(stack, 48), page(heap, 6), PAGE_SIZE)),
            Ok(()),
        ),
        (
            "alias the first source again, elsewhere",
            Box::new(move |s| s.map_memory(page(stack, 56), page(heap, 6), PAGE_SIZE)),
            Ok(()),
        ),
    ]
}

#[test]
fn saving_a_restored_space_gives_the_same_bytes() {
    let rich = rich();
    let bytes = rich.space.save().unwrap();
    let restored = restore_with(&bytes, &rich.objects).0.unwrap();
    assert_eq!(restored.save().unwrap(), bytes);
}

#[test]
fn a_new_space_with_nothing_mapped_but_its_code_round_trips() {
    let (space, _log) = crate::memory::fixture::created();
    let bytes = space.save().unwrap();
    let restored = restore_with(&bytes, &Resolver::new()).0.unwrap();
    assert_eq!(Body::of(&restored), Body::of(&space));
    assert_eq!(restored.max_heap_size(), 0);
}

// ----- Layout --------------------------------------------------------------------------------

/// A body with one block of every memory state, each field of each block a distinct value that
/// is not zero (except the `Free` state's word, which is zero), and every other field distinct
/// and not zero.
fn golden_body() -> Body {
    let blocks = MemoryState::ALL
        .iter()
        .zip(1_u8..)
        .map(|(state, n)| {
            let wide = u16::from(n);
            SavedBlock {
                address: u64::from(n) << 16,
                pages: 0x20 + u64::from(n),
                state: state.word(),
                permission: 0x40 + n,
                original_permission: 0x60 + n,
                attribute: 0x80 + n,
                ipc_lock_count: 0x100 + wide,
                device_use_count: 0x200 + wide,
                ipc_disable_merge_count: 0x300 + wide,
                device_disable_merge_left_count: 0x400 + wide,
                device_disable_merge_right_count: 0x500 + wide,
                disable_merge: 0xa0 + n,
            }
        })
        .collect();
    Body {
        request: SavedRequest {
            code_address: 0x1_1000,
            code_size: 0x2000,
            arrangement: arrangement_tag(MemoryArrangement::TwelveGb),
            alias_region_extra: true,
            offsets: [0x3_0000, 0x4_0000, 0x5_0000, 0x6_0000],
        },
        regions: SavedRegions {
            address_space: (0x11, 0x12),
            code: (0x13, 0x14),
            alias: (0x15, 0x16),
            heap: (0x17, 0x18),
            stack: (0x19, 0x1a),
            kernel_map: (0x1b, 0x1c),
            alias_region_extra_size: 0x1d,
        },
        heap_size: 0x21,
        max_heap_size: 0x22,
        reserved: 0x23,
        main_thread_stack_size: 0x24,
        mapped_physical_memory_size: 0x25,
        blocks,
        shared: vec![SavedShared {
            address: 0x26,
            size: 0x27,
            object: crate::id::ObjectId::from_parts(0x28, 0x29),
        }],
        locks: vec![SavedLock {
            address: 0x2a,
            size: 0x2b,
            count: 0x2c,
        }],
        aliases: vec![SavedAlias {
            address: 0x2d,
            source: 0x2e,
            size: 0x2f,
        }],
    }
}

/// The golden body's encoding, written out by field (`postcard`: integers above `u8` as
/// little-endian base-128 varints, `u8` and `bool` as one byte, arrays and tuples as their
/// items, a `Vec` as a varint length and its items).
fn golden_bytes() -> Vec<u8> {
    let mut bytes = vec![
        // The request: code address 0x1_1000, code size 0x2000, arrangement tag 2, the alias
        // extra, the four offsets.
        0x80,
        0xa0,
        0x04,
        0x80,
        0x40,
        2,
        1,
        0x80,
        0x80,
        0x0c,
        0x80,
        0x80,
        0x10,
        0x80,
        0x80,
        0x14,
        0x80,
        0x80,
        0x18,
        // The regions, each start and end, then the alias extra size.
        0x11,
        0x12,
        0x13,
        0x14,
        0x15,
        0x16,
        0x17,
        0x18,
        0x19,
        0x1a,
        0x1b,
        0x1c,
        0x1d,
        // The heap, largest heap, reservation, stack and mapped physical memory.
        0x21,
        0x22,
        0x23,
        0x24,
        0x25,
        // How many blocks.
        u8::try_from(MemoryState::ALL.len()).unwrap(),
    ];
    for (state, n) in MemoryState::ALL.iter().zip(1_u8..) {
        // Address `n << 16`: three varint bytes.
        bytes.extend([0x80, 0x80, 0x04 * n]);
        // Pages: one byte below 0x80.
        bytes.push(0x20 + n);
        bytes.extend(varint(u64::from(state.word())));
        bytes.extend([0x40 + n, 0x60 + n, 0x80 + n]);
        for base in [0x100_u16, 0x200, 0x300, 0x400, 0x500] {
            bytes.extend(varint(u64::from(base + u16::from(n))));
        }
        bytes.push(0xa0 + n);
    }
    // One shared mapping (address, size, object index and generation), one lock (address,
    // size, count), one alias (address, source, size).
    bytes.extend([
        1, 0x26, 0x27, 0x28, 0x29, 1, 0x2a, 0x2b, 0x2c, 1, 0x2d, 0x2e, 0x2f,
    ]);
    bytes
}

/// A little-endian base-128 varint: what `postcard` writes for an integer wider than a byte.
fn varint(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let low = u8::try_from(value & 0x7f).unwrap();
        value >>= 7;
        if value == 0 {
            bytes.push(low);
            return bytes;
        }
        bytes.push(low | 0x80);
    }
}

#[test]
fn the_varint_helper_matches_postcard() {
    for value in [
        0,
        1,
        0x7f,
        0x80,
        0x3fff,
        0x4000,
        u64::from(u32::MAX),
        u64::MAX,
    ] {
        assert_eq!(varint(value), postcard::to_allocvec(&value).unwrap());
    }
}

#[test]
fn the_body_layout_is_pinned() {
    let body = golden_body();
    let bytes = postcard::to_allocvec(&body).unwrap();
    assert_eq!(bytes, golden_bytes());
    // The whole snapshot is the header, then those bytes.
    let snapshot = encode(FORMAT, &body).unwrap();
    let mut header = ADDRESS_SPACE_MAGIC.to_vec();
    header.extend(varint(u64::from(ADDRESS_SPACE_VERSION)));
    assert_eq!(snapshot.get(..header.len()), Some(&header[..]));
    assert_eq!(snapshot.get(header.len()..), Some(&bytes[..]));
    // Decoding gives the body back.
    assert_eq!(decode::<Body>(FORMAT, &snapshot), Ok(body));
}

#[test]
fn a_hand_built_body_decodes_field_by_field() {
    // The golden bytes, built by hand above rather than by the encoder, decode to the golden
    // body: each field read from its own place.
    let mut snapshot = ADDRESS_SPACE_MAGIC.to_vec();
    snapshot.extend(varint(u64::from(ADDRESS_SPACE_VERSION)));
    snapshot.extend(golden_bytes());
    assert_eq!(decode::<Body>(FORMAT, &snapshot), Ok(golden_body()));
}

#[test]
fn the_arrangement_tags_are_pinned() {
    assert_eq!(arrangement_tag(MemoryArrangement::FourGb), 1);
    assert_eq!(arrangement_tag(MemoryArrangement::TwelveGb), 2);
    for (arrangement, tag) in ARRANGEMENT_TAGS {
        assert_eq!(arrangement_of(tag), Ok(arrangement));
        assert_ne!(tag, 0);
    }
    for tag in [0, 3, u8::MAX] {
        assert_eq!(arrangement_of(tag), Err(SnapshotError::Arrangement { tag }));
    }
    // The tag is the request's third field: after the code address and size (one byte each
    // here).
    let request = SavedRequest {
        code_address: 0x10,
        code_size: 0x20,
        arrangement: arrangement_tag(MemoryArrangement::FourGb),
        alias_region_extra: false,
        offsets: [0x30, 0x40, 0x50, 0x60],
    };
    assert_eq!(
        postcard::to_allocvec(&request).unwrap(),
        [0x10, 0x20, 1, 0, 0x30, 0x40, 0x50, 0x60]
    );
}

/// Any body: every field drawn on its own, so a read that takes another field's place shows.
fn any_body() -> impl Strategy<Value = Body> {
    let request = (
        any::<u64>(),
        any::<u64>(),
        any::<u8>(),
        any::<bool>(),
        any::<[u64; 4]>(),
    )
        .prop_map(
            |(code_address, code_size, arrangement, alias_region_extra, offsets)| SavedRequest {
                code_address,
                code_size,
                arrangement,
                alias_region_extra,
                offsets,
            },
        );
    let regions =
        (any::<[u64; 12]>(), any::<u64>()).prop_map(|(r, alias_region_extra_size)| SavedRegions {
            address_space: (r[0], r[1]),
            code: (r[2], r[3]),
            alias: (r[4], r[5]),
            heap: (r[6], r[7]),
            stack: (r[8], r[9]),
            kernel_map: (r[10], r[11]),
            alias_region_extra_size,
        });
    let block = (
        (any::<u64>(), any::<u64>(), any::<u32>()),
        any::<[u8; 4]>(),
        any::<[u16; 5]>(),
    )
        .prop_map(|((address, pages, state), bits, counts)| SavedBlock {
            address,
            pages,
            state,
            permission: bits[0],
            original_permission: bits[1],
            attribute: bits[2],
            ipc_lock_count: counts[0],
            device_use_count: counts[1],
            ipc_disable_merge_count: counts[2],
            device_disable_merge_left_count: counts[3],
            device_disable_merge_right_count: counts[4],
            disable_merge: bits[3],
        });
    let shared = (any::<u64>(), any::<u64>(), any::<u32>(), any::<u32>()).prop_map(
        |(address, size, index, generation)| SavedShared {
            address,
            size,
            object: ObjectId::from_parts(index, generation),
        },
    );
    let lock =
        (any::<u64>(), any::<u64>(), any::<u32>()).prop_map(|(address, size, count)| SavedLock {
            address,
            size,
            count,
        });
    let alias = any::<[u64; 3]>().prop_map(|[address, source, size]| SavedAlias {
        address,
        source,
        size,
    });
    (
        request,
        regions,
        any::<[u64; 5]>(),
        prop::collection::vec(block, 0..4),
        prop::collection::vec(shared, 0..3),
        prop::collection::vec(lock, 0..3),
        prop::collection::vec(alias, 0..3),
    )
        .prop_map(
            |(request, regions, sizes, blocks, shared, locks, aliases)| Body {
                request,
                regions,
                heap_size: sizes[0],
                max_heap_size: sizes[1],
                reserved: sizes[2],
                main_thread_stack_size: sizes[3],
                mapped_physical_memory_size: sizes[4],
                blocks,
                shared,
                locks,
                aliases,
            },
        )
}

proptest! {
    #![proptest_config(config(256))]

    /// Any body encodes and decodes to itself.
    #[test]
    fn any_body_round_trips(body in any_body()) {
        let bytes = encode(FORMAT, &body).unwrap();
        prop_assert_eq!(decode::<Body>(FORMAT, &bytes), Ok(body));
    }

    /// Any bytes after a good header are refused or restored, never a panic.
    #[test]
    fn any_bytes_are_refused_or_restored_without_a_panic(
        tail in prop::collection::vec(any::<u8>(), 0..256)
    ) {
        let mut bytes = ADDRESS_SPACE_MAGIC.to_vec();
        bytes.extend(varint(u64::from(ADDRESS_SPACE_VERSION)));
        bytes.extend(tail);
        // Random bytes are almost never a whole, sound address space; either outcome is fine
        // as long as there is one.
        let (restored, _log) = restore_with(&bytes, &Resolver::new());
        drop(restored);
    }
}

// ----- One fault per check, in the documented order -------------------------------------------

/// What a fault case can change: the body, the shared memory the caller resolves, and the new
/// translation's log.
struct Case {
    body: Body,
    objects: Resolver,
    /// The new translation's calls to refuse, counted from zero.
    refuse: Vec<usize>,
}

impl Case {
    fn new(rich: &Rich) -> Self {
        Self {
            body: Body::of(&rich.space),
            objects: rich.objects.clone(),
            refuse: Vec::new(),
        }
    }

    /// The case restored, and the new translation's calls.
    fn restore(&self) -> (Result<AddressSpace, SnapshotError>, Vec<MapperCall>) {
        let (mapper, log) = RecordingMapper::new();
        log.refuse_calls(&self.refuse);
        let bytes = encode(FORMAT, &self.body).unwrap();
        let restored = AddressSpace::restore(&bytes, Box::new(mapper), |object| {
            self.objects.get(&object).copied()
        });
        (restored, log.take())
    }

    fn error(&self) -> SnapshotError {
        self.restore().0.unwrap_err()
    }
}

/// One fault: it changes a case and says which error that case gives.
type Fault = fn(&mut Case, &Rich) -> SnapshotError;

/// Inserts a lock and keeps the locks in order; returns its index.
fn insert_lock(body: &mut Body, address: u64, size: u64) -> usize {
    body.locks.push(SavedLock {
        address,
        size,
        count: 1,
    });
    body.locks.sort_by_key(|lock| (lock.address, lock.size));
    body.locks
        .iter()
        .position(|lock| (lock.address, lock.size) == (address, size))
        .unwrap()
}

/// Makes the block at `address` device shared with the counts and barriers of one lock taken
/// exactly over it, so that it passes the block checks alone.
fn device_share_alone(body: &mut Body, address: u64) {
    let index = at(body, address);
    let block = &mut body.blocks[index];
    block.attribute |= MemoryAttribute::DEVICE_SHARED.bits();
    block.device_use_count = 1;
    block.device_disable_merge_left_count = 1;
    block.device_disable_merge_right_count = 1;
    block.disable_merge |= (DisableMerge::DEVICE_LEFT | DisableMerge::DEVICE_RIGHT).bits();
}

/// `fault` in the rich space's second alias.
const fn alias_fault(fault: AliasFault) -> SnapshotError {
    SnapshotError::Alias { index: 1, fault }
}

/// The pool the rich space draws from.
const fn pool() -> u64 {
    MemoryArrangement::FourGb.application_pool_size()
}

/// The faults, in the order the module docs give the checks, each named.
#[expect(
    clippy::too_many_lines,
    reason = "one fault per documented check, in the documented order, reads best as one list"
)]
fn faults() -> Vec<(&'static str, Fault)> {
    vec![
        ("arrangement", |case, _| {
            case.body.request.arrangement = 0;
            SnapshotError::Arrangement { tag: 0 }
        }),
        ("layout", |case, _| {
            case.body.request.code_size += 1;
            SnapshotError::Layout(LayoutError::BadCode {
                address: CODE,
                size: case.body.request.code_size,
            })
        }),
        ("regions", |case, _| {
            case.body.regions.alias.0 += PAGE_SIZE;
            SnapshotError::Regions
        }),
        ("stack", |case, _| {
            case.body.main_thread_stack_size += 1;
            SnapshotError::Stack {
                size: case.body.main_thread_stack_size,
            }
        }),
        ("largest heap", |case, _| {
            case.body.max_heap_size -= PAGE_SIZE;
            SnapshotError::MaxHeap {
                size: case.body.max_heap_size,
            }
        }),
        ("heap", |case, _| {
            case.body.heap_size += 1;
            SnapshotError::Heap {
                size: case.body.heap_size,
            }
        }),
        ("over the limit", |case, _| {
            case.body.reserved = pool() + 1;
            SnapshotError::OverLimit {
                reserved: pool() + 1,
                limit: pool(),
            }
        }),
        ("reservation", |case, _| {
            let used = case.body.reserved;
            case.body.reserved -= PAGE_SIZE;
            SnapshotError::Reservation {
                reserved: used - PAGE_SIZE,
                used,
            }
        }),
        ("a block", |case, _| {
            case.body.blocks[0].pages = 0;
            SnapshotError::Block {
                index: 0,
                fault: BlockFault::Shape,
            }
        }),
        ("short", |case, _| {
            let last = case.body.blocks.pop().unwrap();
            SnapshotError::Short { end: last.address }
        }),
        ("code range", |case, _| {
            let text = at(&case.body, CODE);
            case.body.blocks[text].state = MemoryState::AliasCode.word();
            SnapshotError::CodeRange
        }),
        ("heap range", |case, _| {
            case.body.heap_size += PAGE_SIZE;
            case.body.reserved += PAGE_SIZE;
            SnapshotError::HeapRange
        }),
        ("physical", |case, _| {
            let saved = case.body.mapped_physical_memory_size;
            case.body.mapped_physical_memory_size += PAGE_SIZE;
            case.body.reserved += PAGE_SIZE;
            SnapshotError::Physical {
                saved: saved + PAGE_SIZE,
                mapped: saved,
            }
        }),
        ("shared order", |case, _| {
            case.body.shared.swap(0, 1);
            SnapshotError::Shared {
                index: 1,
                fault: SharedFault::Order,
            }
        }),
        ("shared block", |case, _| {
            case.body.shared[0].size += PAGE_SIZE;
            SnapshotError::Shared {
                index: 0,
                fault: SharedFault::NotABlock,
            }
        }),
        ("unmapped", |case, _| {
            let second = case.body.shared.pop().unwrap();
            SnapshotError::Unmapped {
                block: at(&case.body, second.address),
            }
        }),
        ("lock count", |case, _| {
            case.body.locks[0].count = 0;
            SnapshotError::Lock {
                index: 0,
                fault: LockFault::Count,
            }
        }),
        ("lock order", |case, _| {
            case.body.locks.swap(0, 1);
            SnapshotError::Lock {
                index: 1,
                fault: LockFault::Order,
            }
        }),
        ("lock range", |case, _| {
            case.body.locks[0].address += 1;
            SnapshotError::Lock {
                index: 0,
                fault: LockFault::Range,
            }
        }),
        ("lock alignment", |case, _| {
            // Inside the three pages of text: whole pages, but no block starts there.
            let index = insert_lock(&mut case.body, page(CODE, 1), PAGE_SIZE);
            SnapshotError::Lock {
                index,
                fault: LockFault::Unaligned,
            }
        }),
        ("lock not shared", |case, _| {
            // Exactly the read-only data's block, which is not device shared.
            let index = insert_lock(&mut case.body, RODATA, PAGE_SIZE);
            SnapshotError::Lock {
                index,
                fault: LockFault::NotShared,
            }
        }),
        ("lock state", |case, rich| {
            // The thread-local page, device shared with the counts of one lock exactly over it,
            // and that lock: its state cannot be device mapped.
            let tls = thread_local(rich);
            device_share_alone(&mut case.body, tls);
            let index = insert_lock(&mut case.body, tls, PAGE_SIZE);
            SnapshotError::Lock {
                index,
                fault: LockFault::State,
            }
        }),
        ("device counts", |case, _| {
            case.body.locks[0].count += 1;
            let first = case.body.locks[0].address;
            SnapshotError::DeviceCounts {
                block: at(&case.body, first),
            }
        }),
        // Each alias fault is on the second alias, so that a pair of them is reported by the
        // order of the checks of one alias, not by which alias comes first.
        ("alias shape", |case, _| {
            case.body.aliases[1].size += 1;
            alias_fault(AliasFault::Shape)
        }),
        ("alias order", |case, _| {
            case.body.aliases[1].address = case.body.aliases[0].address;
            alias_fault(AliasFault::Order)
        }),
        ("alias region", |case, rich| {
            // Its two pages across the stack region's end.
            case.body.aliases[1].address = rich.space.regions().stack.end - PAGE_SIZE;
            alias_fault(AliasFault::Region)
        }),
        ("alias overlap", |case, _| {
            case.body.aliases[1].source = case.body.aliases[1].address;
            alias_fault(AliasFault::Overlap)
        }),
        ("alias source", |case, rich| {
            // Heap page 7, which is not aliased.
            case.body.aliases[1].source = page(rich.heap, 7);
            alias_fault(AliasFault::Source)
        }),
        ("alias destination", |case, rich| {
            // Free stack pages above the second alias.
            case.body.aliases[1].address = page(rich.stack, 56);
            alias_fault(AliasFault::Destination)
        }),
        ("alias source taken", |case, _| {
            // The first alias takes the second's first source page.
            case.body.aliases[0].source = case.body.aliases[1].source;
            alias_fault(AliasFault::SourceTaken)
        }),
        ("shared missing", |case, rich| {
            case.objects.remove(&rich.shared[0]);
            SnapshotError::Shared {
                index: 0,
                fault: SharedFault::Missing,
            }
        }),
        ("shared size", |case, rich| {
            let other = case.objects[&rich.shared[0]];
            case.objects.insert(rich.shared[1], other);
            SnapshotError::Shared {
                index: 1,
                fault: SharedFault::Size,
            }
        }),
        ("shared permission", |case, _| {
            // The read-only shared memory's block made read-write.
            let second = case.body.shared[1].address;
            let index = at(&case.body, second);
            case.body.blocks[index].permission = MemoryPermission::USER_READ_WRITE.bits();
            SnapshotError::Shared {
                index: 1,
                fault: SharedFault::Permission,
            }
        }),
        ("translation", |case, _| {
            case.refuse = vec![2];
            let third = case
                .body
                .blocks
                .iter()
                .filter(|block| block.state != MemoryState::Free.word())
                .nth(2)
                .unwrap();
            SnapshotError::Translation {
                address: third.address,
                size: third.pages * PAGE_SIZE,
                error: MapperError("the test told the mapper to refuse".into()),
                stuck: 0,
            }
        }),
    ]
}

#[test]
fn the_rich_space_passes_every_check() {
    let rich = rich();
    let case = Case::new(&rich);
    let (restored, calls) = case.restore();
    assert!(restored.is_ok());
    assert_eq!(calls, expected_calls(&case.body, &rich.objects));
}

#[test]
fn each_fault_alone_is_its_own_error() {
    let rich = rich();
    for (name, fault) in faults() {
        let mut case = Case::new(&rich);
        let expected = fault(&mut case, &rich);
        assert_eq!(case.error(), expected, "{name}");
    }
}

#[test]
fn with_two_faults_the_earlier_check_is_reported() {
    // Each pair of neighbouring checks, the later fault applied first so that it cannot undo
    // the earlier one: the earlier check's error is the one reported.
    let rich = rich();
    let faults = faults();
    for pair in faults.windows(2) {
        let [(earlier_name, earlier), (later_name, later)] = pair else {
            continue;
        };
        let mut alone = Case::new(&rich);
        let expected = earlier(&mut alone, &rich);
        let mut both = Case::new(&rich);
        let later_error = later(&mut both, &rich);
        earlier(&mut both, &rich);
        assert_ne!(expected, later_error, "{earlier_name} and {later_name}");
        assert_eq!(both.error(), expected, "{earlier_name} before {later_name}");
    }
}

#[test]
fn a_refusal_before_the_translation_asks_the_new_mapper_nothing() {
    let rich = rich();
    // The last check before the translation, with the translation's fault present too.
    let mut case = Case::new(&rich);
    case.refuse = vec![0];
    let index = at(&case.body, case.body.shared[1].address);
    case.body.blocks[index].permission = MemoryPermission::USER_READ_WRITE.bits();
    let (restored, calls) = case.restore();
    assert_eq!(
        restored.unwrap_err(),
        SnapshotError::Shared {
            index: 1,
            fault: SharedFault::Permission
        }
    );
    assert_eq!(calls, []);
}

/// The unmap that undoes `call`, a mapping a restore made.
fn undo(call: MapperCall) -> MapperCall {
    match call {
        MapperCall::Map { address, size, .. } | MapperCall::MapAlias { address, size, .. } => {
            MapperCall::Unmap { address, size }
        }
        MapperCall::MapShared {
            address, backing, ..
        } => MapperCall::Unmap {
            address,
            size: backing.size(),
        },
        other => unreachable!("a restore makes only mappings, not {other:?}"),
    }
}

#[test]
fn a_refused_mapping_unmaps_what_was_mapped_in_address_order() {
    let rich = rich();
    let mut case = Case::new(&rich);
    case.refuse = vec![2];
    let (restored, calls) = case.restore();
    assert!(matches!(
        restored,
        Err(SnapshotError::Translation { stuck: 0, .. })
    ));
    let made = expected_calls(&case.body, &rich.objects);
    assert_eq!(
        calls,
        [made[0], made[1], undo(made[0]), undo(made[1])],
        "two mapped, the third refused, the two unmapped in order"
    );
}

#[test]
fn an_unmap_refused_in_the_undo_counts_its_bytes_as_stuck() {
    let rich = rich();
    let mut case = Case::new(&rich);
    // The third mapping, then the first undo.
    case.refuse = vec![2, 3];
    let (restored, calls) = case.restore();
    let made = expected_calls(&case.body, &rich.objects);
    let MapperCall::Unmap { size, .. } = undo(made[0]) else {
        unreachable!("an undo is an unmap");
    };
    let Err(SnapshotError::Translation { stuck, .. }) = restored else {
        unreachable!("not a translation refusal");
    };
    assert_eq!(stuck, size);
    // The second is still unmapped after the first's unmap was refused.
    assert_eq!(calls, [made[0], made[1], undo(made[1])]);
}

// ----- The checks of one block, in their order ------------------------------------------------

/// Where the rich space's thread-local page is.
fn thread_local(rich: &Rich) -> u64 {
    rich.space
        .blocks()
        .find(|block| block.state() == MemoryState::ThreadLocal)
        .unwrap()
        .address()
}

/// A change to one saved block.
type BlockChange = Box<dyn Fn(&mut SavedBlock)>;

/// The user's read, write and execute with the kernel's read and write: every user bit, with
/// the kernel bits a conversion would give it, but no system call permission is all three.
const READ_WRITE_EXECUTE: u8 =
    MemoryPermission::USER_READ_WRITE.bits() | MemoryPermission::USER_EXECUTE.bits();

/// The user's write bit alone: a permission with no translation.
const WRITE_ONLY: u8 = MemoryPermission::USER_WRITE.bits() & MemoryPermission::USER_MASK.bits();

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one case per check of a block, in the documented order, reads best as one list"
)]
fn each_check_of_one_block_has_its_own_fault() {
    let rich = rich();
    let tls = thread_local(&rich);
    let shared = MemoryAttribute::DEVICE_SHARED.bits();
    let cases: Vec<(&str, BlockChange, BlockFault)> = vec![
        ("unaligned", Box::new(|b| b.address += 1), BlockFault::Shape),
        ("no pages", Box::new(|b| b.pages = 0), BlockFault::Shape),
        (
            "an end past 2^64",
            Box::new(|b| b.pages = u64::MAX / PAGE_SIZE),
            BlockFault::Shape,
        ),
        (
            "starting a page early",
            Box::new(|b| {
                b.address -= PAGE_SIZE;
                b.pages += 1;
            }),
            BlockFault::Overlap,
        ),
        (
            "starting a page late",
            Box::new(|b| b.address += PAGE_SIZE),
            BlockFault::Gap,
        ),
        (
            "an unknown state",
            Box::new(|b| b.state = 0xdead_beef),
            BlockFault::State { word: 0xdead_beef },
        ),
        (
            "every unknown attribute bit",
            Box::new(|b| b.attribute |= !KNOWN_ATTRIBUTE_BITS),
            BlockFault::Bits,
        ),
        (
            "every unknown permission bit",
            Box::new(|b| b.permission |= !KNOWN_PERMISSION_BITS),
            BlockFault::Bits,
        ),
        (
            "an IPC lock count",
            Box::new(|b| b.ipc_lock_count = 1),
            BlockFault::Ipc,
        ),
        (
            "an IPC merge count",
            Box::new(|b| b.ipc_disable_merge_count = 1),
            BlockFault::Ipc,
        ),
        (
            "an original permission",
            Box::new(|b| b.original_permission = MemoryPermission::USER_READ.bits()),
            BlockFault::Ipc,
        ),
        (
            "the IPC-locked attribute",
            Box::new(|b| b.attribute |= MemoryAttribute::IPC_LOCKED.bits()),
            BlockFault::Ipc,
        ),
        (
            "the IPC-left barrier",
            Box::new(|b| b.disable_merge |= DisableMerge::IPC_LEFT.bits()),
            BlockFault::Ipc,
        ),
        (
            "device shared with no use count",
            Box::new(move |b| b.attribute |= shared),
            BlockFault::DeviceCounts,
        ),
        (
            "a use count without device shared",
            Box::new(|b| b.device_use_count = 1),
            BlockFault::DeviceCounts,
        ),
        (
            "a left barrier with no count",
            Box::new(move |b| {
                b.attribute |= shared;
                b.device_use_count = 1;
                b.disable_merge |= DisableMerge::DEVICE_LEFT.bits();
            }),
            BlockFault::DeviceCounts,
        ),
        (
            "a right count with no barrier",
            Box::new(move |b| {
                b.attribute |= shared;
                b.device_use_count = 1;
                b.device_disable_merge_right_count = 1;
            }),
            BlockFault::DeviceCounts,
        ),
        (
            "a left count above the use count",
            Box::new(move |b| {
                b.attribute |= shared;
                b.device_use_count = 1;
                b.device_disable_merge_left_count = 2;
                b.disable_merge |= DisableMerge::DEVICE_LEFT.bits();
            }),
            BlockFault::DeviceCounts,
        ),
        (
            "a right count above the use count",
            Box::new(move |b| {
                b.attribute |= shared;
                b.device_use_count = 1;
                b.device_disable_merge_right_count = 2;
                b.disable_merge |= DisableMerge::DEVICE_RIGHT.bits();
            }),
            BlockFault::DeviceCounts,
        ),
        (
            "write without read",
            Box::new(|b| b.permission = WRITE_ONLY),
            BlockFault::Translation,
        ),
        (
            "read, write and execute, which no system call permission converts to",
            Box::new(|b| b.permission = READ_WRITE_EXECUTE),
            BlockFault::Permission,
        ),
        (
            "read and write without the kernel's write",
            Box::new(|b| {
                b.permission =
                    (MemoryPermission::USER_READ_WRITE & !MemoryPermission::KERNEL_WRITE).bits();
            }),
            BlockFault::Permission,
        ),
        (
            "uncached thread-local memory",
            Box::new(|b| b.attribute |= MemoryAttribute::UNCACHED.bits()),
            BlockFault::ChangedAttribute,
        ),
        (
            "GPU-accessible thread-local memory",
            Box::new(|b| b.attribute |= MemoryAttribute::GPU_ACCESSIBLE.bits()),
            BlockFault::ChangedAttribute,
        ),
        (
            "permission-locked thread-local memory",
            Box::new(|b| b.attribute |= MemoryAttribute::PERMISSION_LOCKED.bits()),
            BlockFault::PermissionLock,
        ),
        (
            "thread-local memory as a stack",
            Box::new(|b| b.state = MemoryState::Stack.word()),
            BlockFault::Region,
        ),
        (
            "inaccessible memory, which has no region",
            Box::new(|b| b.state = MemoryState::Inaccessible.word()),
            BlockFault::Region,
        ),
    ];
    for (name, change, fault) in cases {
        let mut case = Case::new(&rich);
        let index = at(&case.body, tls);
        change(&mut case.body.blocks[index]);
        assert_eq!(
            case.error(),
            SnapshotError::Block { index, fault },
            "{name}"
        );
    }
    // A free block (the first) with anything at all.
    let free: Vec<(&str, BlockChange)> = vec![
        (
            "a permission",
            Box::new(|b| b.permission = MemoryPermission::KERNEL_READ.bits()),
        ),
        (
            "an attribute",
            Box::new(|b| b.attribute = MemoryAttribute::UNCACHED.bits()),
        ),
        (
            "an original permission",
            Box::new(|b| b.original_permission = MemoryPermission::KERNEL_READ.bits()),
        ),
        ("a device count", Box::new(|b| b.device_use_count = 1)),
        (
            "a left count",
            Box::new(|b| b.device_disable_merge_left_count = 1),
        ),
    ];
    for (name, change) in free {
        let mut case = Case::new(&rich);
        assert_eq!(case.body.blocks[0].state, MemoryState::Free.word());
        change(&mut case.body.blocks[0]);
        assert_eq!(
            case.error(),
            SnapshotError::Block {
                index: 0,
                fault: BlockFault::Free
            },
            "a free block with {name}"
        );
    }
    // Past the end: the last block runs one page on.
    let mut case = Case::new(&rich);
    let last = case.body.blocks.len() - 1;
    case.body.blocks[last].pages += 1;
    assert_eq!(
        case.error(),
        SnapshotError::Block {
            index: last,
            fault: BlockFault::PastEnd
        }
    );
}

#[test]
fn a_block_that_could_merge_with_the_one_before_is_refused() {
    // The stack's block cut in two halves that are alike in everything a merge compares, the
    // stack's barrier left with the lower half as a split leaves it.
    let rich = rich();
    let mut case = Case::new(&rich);
    let stack = case
        .body
        .blocks
        .iter()
        .position(|block| block.state == MemoryState::Stack.word())
        .unwrap();
    let mut upper = case.body.blocks[stack];
    let half = upper.pages / 2;
    case.body.blocks[stack].pages = half;
    upper.address += half * PAGE_SIZE;
    upper.pages -= half;
    upper.disable_merge = 0;
    case.body.blocks.insert(stack + 1, upper);
    assert_eq!(
        case.error(),
        SnapshotError::Block {
            index: stack + 1,
            fault: BlockFault::Mergeable
        }
    );
    // With a barrier at the upper half's start the two are not mergeable: the split passes the
    // block checks (and fails later, since the stack is one mapping of the region it was).
    upper.disable_merge = DisableMerge::NORMAL.bits();
    case.body.blocks[stack + 1] = upper;
    assert!(!matches!(
        case.restore().0,
        Err(SnapshotError::Block { .. })
    ));
}

#[test]
fn a_block_with_ipc_lock_state_is_refused() {
    // Module docs, "Divergences": stricter than the reference, which can have IPC locks. The
    // heap's first block, with an IPC lock and its attribute as `LockForIpc` leaves them.
    let rich = rich();
    let mut case = Case::new(&rich);
    let index = at(&case.body, rich.heap);
    let block = &mut case.body.blocks[index];
    block.ipc_lock_count = 1;
    block.attribute |= MemoryAttribute::IPC_LOCKED.bits();
    block.original_permission = block.permission;
    assert_eq!(
        case.error(),
        SnapshotError::Block {
            index,
            fault: BlockFault::Ipc
        }
    );
}

#[test]
fn of_two_faults_in_one_block_the_earlier_check_is_reported() {
    let rich = rich();
    let tls = thread_local(&rich);
    let pairs: Vec<(&str, BlockChange, BlockFault)> = vec![
        (
            "no pages and a page early",
            Box::new(|b| {
                b.pages = 0;
                b.address -= PAGE_SIZE;
            }),
            BlockFault::Shape,
        ),
        (
            "a page early and an unknown state",
            Box::new(|b| {
                b.address -= PAGE_SIZE;
                b.pages += 1;
                b.state = 0xdead_beef;
            }),
            BlockFault::Overlap,
        ),
        (
            "a page late and an unknown state",
            Box::new(|b| {
                b.address += PAGE_SIZE;
                b.state = 0xdead_beef;
            }),
            BlockFault::Gap,
        ),
        (
            "an unknown state and an IPC lock",
            Box::new(|b| {
                b.state = 0xdead_beef;
                b.ipc_lock_count = 1;
            }),
            BlockFault::State { word: 0xdead_beef },
        ),
        (
            "an unknown state and an unknown attribute bit",
            Box::new(|b| {
                b.state = 0xdead_beef;
                b.attribute |= !KNOWN_ATTRIBUTE_BITS;
            }),
            BlockFault::State { word: 0xdead_beef },
        ),
        (
            "an unknown permission bit and an IPC lock",
            Box::new(|b| {
                b.permission |= !KNOWN_PERMISSION_BITS;
                b.ipc_lock_count = 1;
            }),
            BlockFault::Bits,
        ),
        (
            "an IPC lock and a use count without device shared",
            Box::new(|b| {
                b.ipc_lock_count = 1;
                b.device_use_count = 1;
            }),
            BlockFault::Ipc,
        ),
        (
            "bad device counts and write without read",
            Box::new(|b| {
                b.device_use_count = 1;
                b.permission = WRITE_ONLY;
            }),
            BlockFault::DeviceCounts,
        ),
        (
            "write without read and the wrong region",
            Box::new(|b| {
                b.permission = WRITE_ONLY;
                b.state = MemoryState::Stack.word();
            }),
            BlockFault::Translation,
        ),
        (
            "a permission the state never has and the wrong region",
            Box::new(|b| {
                b.permission = READ_WRITE_EXECUTE;
                b.state = MemoryState::Stack.word();
            }),
            BlockFault::Permission,
        ),
    ];
    for (name, change, fault) in pairs {
        let mut case = Case::new(&rich);
        let index = at(&case.body, tls);
        change(&mut case.body.blocks[index]);
        assert_eq!(
            case.error(),
            SnapshotError::Block { index, fault },
            "{name}"
        );
    }
}

#[test]
fn of_two_faults_on_the_first_last_or_two_blocks_the_earlier_check_is_reported() {
    let rich = rich();
    let tls = thread_local(&rich);
    // Unknown bits before free, on the first block (free).
    let mut case = Case::new(&rich);
    case.body.blocks[0].attribute = !KNOWN_ATTRIBUTE_BITS;
    assert_eq!(
        case.error(),
        SnapshotError::Block {
            index: 0,
            fault: BlockFault::Bits
        }
    );
    // Free before IPC, on the first block (free).
    let mut case = Case::new(&rich);
    case.body.blocks[0].permission = MemoryPermission::KERNEL_READ.bits();
    case.body.blocks[0].ipc_lock_count = 1;
    assert_eq!(
        case.error(),
        SnapshotError::Block {
            index: 0,
            fault: BlockFault::Free
        }
    );
    // Past the end before the state word, on the last block.
    let mut case = Case::new(&rich);
    let last = case.body.blocks.len() - 1;
    case.body.blocks[last].pages += 1;
    case.body.blocks[last].state = 0xdead_beef;
    assert_eq!(
        case.error(),
        SnapshotError::Block {
            index: last,
            fault: BlockFault::PastEnd
        }
    );
    // An earlier block's fault before a later block's.
    let mut case = Case::new(&rich);
    let index = at(&case.body, tls);
    case.body.blocks[index].ipc_lock_count = 1;
    case.body.blocks[0].permission = MemoryPermission::KERNEL_READ.bits();
    assert_eq!(
        case.error(),
        SnapshotError::Block {
            index: 0,
            fault: BlockFault::Free
        }
    );
}

// ----- Edges of the counts ---------------------------------------------------------------------

/// The rich case with the stack, largest heap and heap as given, and the reservation made to
/// add up.
fn counts_case(rich: &Rich, stack: u64, max_heap: u64, heap: u64) -> Case {
    let mut case = Case::new(rich);
    let body = &mut case.body;
    let others = body.reserved - body.heap_size - body.main_thread_stack_size;
    body.main_thread_stack_size = stack;
    body.max_heap_size = max_heap;
    body.heap_size = heap;
    body.reserved = others + heap + stack;
    case
}

#[test]
fn the_stack_and_the_code_may_fill_the_largest_process_memory_and_no_more() {
    let rich = rich();
    let max = rich.space.max_process_memory();
    let code = rich.space.created_from.code_size;
    // Exactly full: the stack check passes, and the reservation is past the pool.
    let full = counts_case(&rich, max - code, 0, 0);
    assert!(matches!(full.error(), SnapshotError::OverLimit { .. }));
    // A page more does not fit.
    let over = counts_case(&rich, max - code + PAGE_SIZE, 0, 0);
    assert_eq!(
        over.error(),
        SnapshotError::Stack {
            size: max - code + PAGE_SIZE
        }
    );
}

#[test]
fn the_largest_heap_is_what_the_stack_left_or_zero_with_no_stack() {
    let rich = rich();
    let max = rich.space.max_process_memory();
    let code = rich.space.created_from.code_size;
    let stack = rich.space.main_thread_stack_size;
    let heap = rich.space.heap_size();
    let left = max - code - stack;
    assert_eq!(rich.space.max_heap_size(), left);
    let is_max_heap = |case: &Case| matches!(case.restore().0, Err(SnapshotError::MaxHeap { .. }));
    // With a stack: exactly what it left, not zero, not a page either side.
    for wrong in [0, left - PAGE_SIZE, left + PAGE_SIZE] {
        assert_eq!(
            counts_case(&rich, stack, wrong, heap).error(),
            SnapshotError::MaxHeap { size: wrong },
            "{wrong:#x}"
        );
    }
    assert!(!is_max_heap(&counts_case(&rich, stack, left, heap)));
    // With no stack: zero (the stack was never made) or what the code left (a stack of no
    // bytes was made); not a page less than that.
    assert!(!is_max_heap(&counts_case(&rich, 0, 0, 0)));
    assert!(!is_max_heap(&counts_case(&rich, 0, max - code, heap)));
    assert_eq!(
        counts_case(&rich, 0, max - code - PAGE_SIZE, heap).error(),
        SnapshotError::MaxHeap {
            size: max - code - PAGE_SIZE
        }
    );
}

#[test]
fn the_heap_size_must_be_exactly_the_heap_regions_normal_memory() {
    // A page more than the blocks hold, and a page less (heap memory past the heap's end).
    let rich = rich();
    let heap = rich.space.heap_size();
    let stack = rich.space.main_thread_stack_size;
    let max_heap = rich.space.max_heap_size();
    for size in [heap + PAGE_SIZE, heap - PAGE_SIZE] {
        assert_eq!(
            counts_case(&rich, stack, max_heap, size).error(),
            SnapshotError::HeapRange,
            "{size:#x}"
        );
    }
    assert!(
        counts_case(&rich, stack, max_heap, heap)
            .restore()
            .0
            .is_ok()
    );
}

#[test]
fn the_heap_may_be_the_largest_heap_and_no_more() {
    let rich = rich();
    let stack = rich.space.main_thread_stack_size;
    let max_heap = rich.space.max_heap_size();
    // Exactly the largest heap passes the heap check (and is past the pool).
    let full = counts_case(&rich, stack, max_heap, max_heap);
    assert!(matches!(full.error(), SnapshotError::OverLimit { .. }));
    let over = counts_case(&rich, stack, max_heap, max_heap + PAGE_SIZE);
    assert_eq!(
        over.error(),
        SnapshotError::Heap {
            size: max_heap + PAGE_SIZE
        }
    );
}

#[test]
fn the_reservation_may_be_the_whole_pool_and_no_more() {
    let rich = rich();
    let mut case = Case::new(&rich);
    let used = case.body.reserved;
    case.body.reserved = pool();
    assert_eq!(
        case.error(),
        SnapshotError::Reservation {
            reserved: pool(),
            used
        }
    );
    case.body.reserved = pool() + 1;
    assert_eq!(
        case.error(),
        SnapshotError::OverLimit {
            reserved: pool() + 1,
            limit: pool()
        }
    );
}

#[test]
fn the_mapped_physical_memory_must_be_the_alias_regions_normal_bytes_exactly() {
    let rich = rich();
    let mapped = rich.space.mapped_physical_memory_size();
    for saved in [mapped - PAGE_SIZE, mapped + PAGE_SIZE] {
        let mut case = Case::new(&rich);
        case.body.mapped_physical_memory_size = saved;
        case.body.reserved = case.body.reserved - mapped + saved;
        assert_eq!(
            case.error(),
            SnapshotError::Physical { saved, mapped },
            "{saved:#x}"
        );
    }
}

// ----- Device-share locks ---------------------------------------------------------------------

#[test]
fn each_live_lock_counts_once_against_its_blocks() {
    // The heap range locked twice is saved with a count of 2. One fewer or one more, or no
    // entry for it, is a block whose use count the locks do not add up to.
    let rich = rich();
    let body = Body::of(&rich.space);
    let range = (page(rich.heap, 2), 2 * PAGE_SIZE);
    let twice = body
        .locks
        .iter()
        .position(|lock| (lock.address, lock.size) == range)
        .unwrap();
    assert_eq!(body.locks[twice].count, 2);
    let block = at(&body, range.0);
    for count in [1, 3] {
        let mut case = Case::new(&rich);
        case.body.locks[twice].count = count;
        assert_eq!(
            case.error(),
            SnapshotError::DeviceCounts { block },
            "{count}"
        );
    }
    let mut case = Case::new(&rich);
    case.body.locks.remove(twice);
    assert_eq!(case.error(), SnapshotError::DeviceCounts { block });
}

#[test]
fn a_lock_must_end_where_a_block_ends() {
    // The twice-locked heap range ends where its last block ends. A page longer ends inside
    // the block after it (heap pages 4-5, locked once by the overlapping lock), and is refused.
    let rich = rich();
    let mut case = Case::new(&rich);
    let range = (page(rich.heap, 2), 2 * PAGE_SIZE);
    let index = case
        .body
        .locks
        .iter()
        .position(|lock| (lock.address, lock.size) == range)
        .unwrap();
    let next = at(&case.body, page(rich.heap, 4));
    assert_eq!(case.body.blocks[next].pages, 2);
    case.body.locks[index].size += PAGE_SIZE;
    assert_eq!(
        case.error(),
        SnapshotError::Lock {
            index,
            fault: LockFault::Unaligned
        }
    );
    // Ending where that block ends is aligned again (and the counts no longer add up).
    case.body.locks[index].size += PAGE_SIZE;
    assert!(matches!(case.error(), SnapshotError::DeviceCounts { .. }));
}

#[test]
fn a_lock_outside_the_address_space_or_wrapping_is_out_of_range() {
    let rich = rich();
    let end = rich.space.regions().address_space.end;
    for (address, size) in [
        (end - PAGE_SIZE, 2 * PAGE_SIZE),
        (end, PAGE_SIZE),
        (u64::MAX - PAGE_SIZE + 1, PAGE_SIZE),
        (rich.heap, 0),
    ] {
        let mut case = Case::new(&rich);
        let index = insert_lock(&mut case.body, address, size);
        assert_eq!(
            case.error(),
            SnapshotError::Lock {
                index,
                fault: LockFault::Range
            },
            "{address:#x} + {size:#x}"
        );
    }
    // The last page of the address space is in range (and is not a block of its own).
    let mut case = Case::new(&rich);
    let index = insert_lock(&mut case.body, end - PAGE_SIZE, PAGE_SIZE);
    assert_eq!(
        case.error(),
        SnapshotError::Lock {
            index,
            fault: LockFault::Unaligned
        }
    );
}

// ----- Saving checks too ----------------------------------------------------------------------

#[test]
fn saving_a_space_whose_bookkeeping_disagrees_is_refused() {
    // A lock count the blocks do not hold: a kernel bug, which save reports.
    let mut rich = rich();
    let key = (page(rich.heap, 3), 3 * PAGE_SIZE);
    rich.space.device_locks.insert(key, 2);
    let block = at(&Body::of(&rich.space), key.0);
    assert_eq!(
        rich.space.save(),
        Err(SnapshotError::DeviceCounts { block })
    );
    // Mapped physical memory that the reservation does not hold.
    let mut other = self::rich();
    other.space.mapped_physical_memory_size += PAGE_SIZE;
    assert!(matches!(
        other.space.save(),
        Err(SnapshotError::Reservation { .. })
    ));
}

#[test]
fn a_snapshot_of_another_version_is_refused_before_its_body_is_read() {
    let rich = rich();
    let mut bytes = rich.space.save().unwrap();
    let version_at = ADDRESS_SPACE_MAGIC.len();
    bytes[version_at] += 1;
    // A body that would not decode either.
    bytes.truncate(version_at + 2);
    assert_eq!(
        restore_with(&bytes, &rich.objects).0.unwrap_err(),
        SnapshotError::Format(FormatError::UnsupportedVersion {
            name: "address space",
            found: ADDRESS_SPACE_VERSION + 1,
            supported: ADDRESS_SPACE_VERSION
        })
    );
}

#[test]
fn trailing_bytes_are_refused() {
    let rich = rich();
    let mut bytes = rich.space.save().unwrap();
    bytes.push(0);
    assert_eq!(
        restore_with(&bytes, &rich.objects).0.unwrap_err(),
        SnapshotError::Format(FormatError::TrailingBytes {
            name: "address space",
            count: 1
        })
    );
}

#[test]
fn the_errors_read_as_plain_words() {
    assert_eq!(
        SnapshotError::Block {
            index: 3,
            fault: BlockFault::Overlap
        }
        .to_string(),
        "block 3 of the address space snapshot overlaps the one before it"
    );
    assert_eq!(
        SnapshotError::Lock {
            index: 1,
            fault: LockFault::NotShared
        }
        .to_string(),
        "device-share lock 1 of the address space snapshot covers a block that is not device shared"
    );
    assert_eq!(
        SnapshotError::Physical {
            saved: 0x3000,
            mapped: 0x2000
        }
        .to_string(),
        "the address space snapshot counts 0x3000 bytes of mapped physical memory, but its blocks hold 0x2000"
    );
    assert_eq!(
        SnapshotError::Translation {
            address: 0x4000_0000,
            size: 0x2000,
            error: MapperError("no".into()),
            stuck: 0x1000
        }
        .to_string(),
        "the new translation refused to map 0x2000 bytes at 0x40000000 (no); 0x1000 bytes it had mapped could not be unmapped"
    );
}

// ----- Generated round trips of live address spaces -------------------------------------------

/// One operation on a live address space, its operands page numbers from the start of the
/// region it works in.
#[derive(Debug, Clone)]
enum Op {
    Heap(u64),
    MapPhysical(u64, u64),
    UnmapPhysical(u64, u64),
    LockHeap(u64, u64),
    LockAlias(u64, u64),
    /// Unlocks the live lock this picks.
    Unlock(usize),
    Attribute(u64, u64, u8),
    Protect(u64, u64, bool),
    Lend(u64, u64),
    TakeBack(u64, u64),
    /// `MapMemory` of `n` heap pages from heap page `src` to stack page `dst` ([`stack_top`]).
    Alias {
        dst: u64,
        src: u64,
        n: u64,
    },
    /// `UnmapMemory` of up to `n` pages from page `skip` of the live alias run this picks,
    /// with the source pages it maps.
    Unalias {
        pick: usize,
        skip: u64,
        n: u64,
    },
    /// A device-share lock on stack pages ([`stack_top`]), where aliases go.
    LockStack(u64, u64),
}

fn any_op() -> impl Strategy<Value = Op> {
    let range = || (0_u64..12, 1_u64..5);
    prop_oneof![
        (0_u64..12, range()).prop_map(|(dst, (src, n))| Op::Alias { dst, src, n }),
        (any::<usize>(), 0_u64..4, 1_u64..4).prop_map(|(pick, skip, n)| Op::Unalias {
            pick,
            skip,
            n
        }),
        range().prop_map(|(at, n)| Op::LockStack(at, n)),
        (0_u64..16).prop_map(Op::Heap),
        range().prop_map(|(at, n)| Op::MapPhysical(at, n)),
        range().prop_map(|(at, n)| Op::UnmapPhysical(at, n)),
        range().prop_map(|(at, n)| Op::LockHeap(at, n)),
        range().prop_map(|(at, n)| Op::LockAlias(at, n)),
        any::<usize>().prop_map(Op::Unlock),
        (range(), 0_u8..4).prop_map(|((at, n), which)| Op::Attribute(at, n, which)),
        (range(), any::<bool>()).prop_map(|((at, n), read_only)| Op::Protect(at, n, read_only)),
        range().prop_map(|(at, n)| Op::Lend(at, n)),
        range().prop_map(|(at, n)| Op::TakeBack(at, n)),
    ]
}

/// Applies `op`, and says whether it was taken.
fn apply(space: &mut AddressSpace, op: &Op) -> bool {
    let heap = space.regions().heap.start;
    let alias = space.regions().alias.start;
    let bytes = |pages: u64| pages * PAGE_SIZE;
    let result = match *op {
        Op::Heap(pages) => space.set_heap_size(bytes(pages)).map(drop),
        Op::MapPhysical(at, n) => space.map_physical_memory(page(alias, at), bytes(n)),
        Op::UnmapPhysical(at, n) => space.unmap_physical_memory(page(alias, at), bytes(n)),
        Op::LockHeap(at, n) => space.lock_device_share(page(heap, at), bytes(n)),
        Op::LockAlias(at, n) => space.lock_device_share(page(alias, at), bytes(n)),
        Op::Unlock(pick) => {
            let live: Vec<(u64, u64)> = space.device_locks.keys().copied().collect();
            match live.get(pick % live.len().max(1)) {
                Some(&(address, size)) => space.unlock_device_share(address, size),
                None => Ok(()),
            }
        }
        Op::Attribute(at, n, which) => {
            let mask = match which {
                0 => MemoryAttribute::UNCACHED,
                1 => MemoryAttribute::GPU_ACCESSIBLE,
                2 => MemoryAttribute::PERMISSION_LOCKED,
                _ => MemoryAttribute::UNCACHED | MemoryAttribute::GPU_ACCESSIBLE,
            };
            space.set_memory_attribute(page(heap, at), bytes(n), mask, mask)
        }
        Op::Protect(at, n, read_only) => {
            let permission = if read_only {
                SvcPermission::Read
            } else {
                SvcPermission::ReadWrite
            };
            space.set_memory_permission(page(heap, at), bytes(n), permission)
        }
        Op::Lend(at, n) => {
            space.lock_for_transfer_memory(page(heap, at), bytes(n), SvcPermission::None)
        }
        Op::TakeBack(at, n) => {
            // The kernel takes back only a range a transfer memory locked, never an alias's
            // source (module docs, "Why each relation holds live": the aliases), which this
            // call would take as well, since it does not test the permission.
            let range = Region {
                start: page(heap, at),
                end: page(heap, at + n),
            };
            let a_source = space.aliases.runs().iter().any(|&(_, source, size)| {
                range.overlaps(Region {
                    start: source,
                    end: source + size,
                })
            });
            if a_source {
                return false;
            }
            space.unlock_for_transfer_memory(page(heap, at), bytes(n))
        }
        Op::Alias { dst, src, n } => {
            let dst = stack_top(space, dst);
            space.map_memory(dst, page(heap, src), bytes(n))
        }
        Op::Unalias { pick, skip, n } => {
            let runs = space.aliases.runs();
            match runs.get(pick % runs.len().max(1)) {
                Some(&(address, source, size)) => {
                    let pages = size / PAGE_SIZE;
                    let skip = skip % pages;
                    let n = n.min(pages - skip);
                    space.unmap_memory(page(address, skip), page(source, skip), bytes(n))
                }
                None => Ok(()),
            }
        }
        Op::LockStack(at, n) => space.lock_device_share(stack_top(space, at), bytes(n)),
    };
    result.is_ok()
}

proptest! {
    #![proptest_config(config(128))]

    /// Random operations, each of which may be refused, on the fixture's loaded address space;
    /// after each, the space saves and restores as it was, and the new translation is given
    /// each block that is not free.
    #[test]
    fn a_space_built_by_random_operations_saves_and_restores_as_it_was(
        ops in prop::collection::vec(any_op(), 1..24)
    ) {
        let (mut space, _log, _) = loaded();
        for op in &ops {
            apply(&mut space, op);
            let bytes = space.save();
            prop_assert!(bytes.is_ok(), "save after {:?}: {:?}", op, bytes);
            let bytes = bytes.unwrap_or_default();
            let (restored, log) = restore_with(&bytes, &Resolver::new());
            prop_assert!(restored.is_ok(), "restore after {:?}: {:?}", op, restored.as_ref().err());
            let body = Body::of(&space);
            if let Ok(restored) = restored {
                prop_assert_eq!(Body::of(&restored), body.clone());
            }
            prop_assert_eq!(log.take(), expected_calls(&body, &Resolver::new()));
        }
    }
}

#[test]
fn the_random_operations_reach_nested_and_overlapping_locks() {
    // A fixed sequence of the generated kind, so the property test's operations are known to
    // reach what the device counts check: nested, overlapping and neighbouring locks, locks on
    // mapped physical memory, and an unlock of one of them.
    let (mut space, _log, _) = loaded();
    for op in [
        Op::Heap(12),
        Op::LockHeap(2, 4),
        Op::LockHeap(2, 4),
        Op::LockHeap(4, 4),
        Op::LockHeap(8, 2),
        Op::MapPhysical(0, 4),
        Op::LockAlias(1, 2),
        Op::Attribute(3, 2, 1),
        Op::Unlock(1),
    ] {
        assert!(apply(&mut space, &op), "{op:?}");
    }
    assert_eq!(space.device_locks.len(), 4);
    let bytes = space.save().unwrap();
    let mut restored = restore_with(&bytes, &Resolver::new()).0.unwrap();
    assert_eq!(Body::of(&restored), Body::of(&space));
    // Unlocking every live lock on the restored space leaves no device-shared block.
    let live: Vec<((u64, u64), u32)> = restored
        .device_locks
        .iter()
        .map(|(&key, &count)| (key, count))
        .collect();
    for ((address, size), count) in live {
        for _ in 0..count {
            restored.unlock_device_share(address, size).unwrap();
        }
    }
    assert!(restored.blocks().all(|block| block.device_use_count() == 0
        && !block.attribute().contains(MemoryAttribute::DEVICE_SHARED)));
}

// ----- Checks the mutation run asked for -------------------------------------------------------

#[test]
fn the_code_must_have_pages() {
    let rich = rich();
    let mut case = Case::new(&rich);
    case.body.request.code_size = 0;
    assert_eq!(
        case.error(),
        SnapshotError::Layout(LayoutError::BadCode {
            address: CODE,
            size: 0
        })
    );
}

#[test]
fn normal_memory_outside_the_heap_and_alias_regions_is_out_of_its_region() {
    // The thread-local page made `Normal`: in the kernel-map region, which is neither.
    let rich = rich();
    let mut case = Case::new(&rich);
    let index = at(&case.body, thread_local(&rich));
    case.body.blocks[index].state = MemoryState::Normal.word();
    assert_eq!(
        case.error(),
        SnapshotError::Block {
            index,
            fault: BlockFault::Region
        }
    );
}

#[test]
fn the_heap_must_start_at_the_heap_regions_start() {
    // The heap's first two pages freed and the heap two pages shorter: the heap region holds as
    // many `Normal` bytes as the heap is long, but not from its start.
    let rich = rich();
    let mut case = Case::new(&rich);
    let first = at(&case.body, rich.heap);
    let block = &mut case.body.blocks[first];
    assert_eq!(block.pages, 2);
    assert_eq!(block.attribute, 0);
    block.state = MemoryState::Free.word();
    block.permission = MemoryPermission::NONE.bits();
    case.body.heap_size -= 2 * PAGE_SIZE;
    case.body.reserved -= 2 * PAGE_SIZE;
    assert_eq!(case.error(), SnapshotError::HeapRange);
}

#[test]
fn a_mapping_over_a_block_of_another_state_is_not_a_shared_block() {
    // The second mapping moved onto the generated code, which has two pages like the mapping
    // is made to have: the right size, the wrong state.
    let rich = rich();
    let mut case = Case::new(&rich);
    case.body.shared[1].address = GENERATED;
    case.body.shared[1].size = 2 * PAGE_SIZE;
    assert_eq!(
        case.error(),
        SnapshotError::Shared {
            index: 1,
            fault: SharedFault::NotABlock
        }
    );
}

#[test]
fn a_lock_range_saved_twice_is_out_of_order() {
    let rich = rich();
    let mut case = Case::new(&rich);
    let first = case.body.locks[0];
    case.body.locks.insert(1, first);
    assert_eq!(
        case.error(),
        SnapshotError::Lock {
            index: 1,
            fault: LockFault::Order
        }
    );
}

/// A change to one saved block, as a plain function.
type SavedChange = fn(&mut SavedBlock);

#[test]
fn each_device_count_is_checked_against_the_locks_on_its_own() {
    // Heap pages 3-5 are locked once, overlapping the twice-locked pages 2-3: page 3 is used by
    // three locks, starts one and ends two; pages 4-5 are used by one and end it. Each change
    // keeps the block's own checks passing, so only the sums can catch it.
    let rich = rich();
    let middle = page(rich.heap, 3);
    let last = page(rich.heap, 4);
    let body = Body::of(&rich.space);
    let counts = |address| {
        let block = body.blocks[at(&body, address)];
        (
            block.device_use_count,
            block.device_disable_merge_left_count,
            block.device_disable_merge_right_count,
        )
    };
    assert_eq!(counts(middle), (3, 1, 2));
    assert_eq!(counts(last), (1, 0, 1));
    let changes: [(&str, u64, SavedChange); 3] = [
        ("a use count", last, |b| b.device_use_count = 2),
        ("a left count", middle, |b| {
            b.device_disable_merge_left_count = 2;
        }),
        ("a right count", middle, |b| {
            b.device_disable_merge_right_count = 1;
        }),
    ];
    for (name, address, change) in changes {
        let mut case = Case::new(&rich);
        let block = at(&case.body, address);
        change(&mut case.body.blocks[block]);
        assert_eq!(
            case.error(),
            SnapshotError::DeviceCounts { block },
            "{name}"
        );
    }
}

// ----- Review fixes: bits, permissions, lock states and the edges mutations found ---------------

#[test]
fn unknown_permission_and_attribute_bits_are_refused_each_alone() {
    // The heap's first block, with each bit starquake does not name, one at a time.
    let rich = rich();
    let unknown: Vec<(&str, u8, u8)> = (0..u8::BITS)
        .map(|bit| 1_u8 << bit)
        .flat_map(|bit| {
            let permission = (bit & !KNOWN_PERMISSION_BITS != 0).then_some(("permission", bit, 0));
            let attribute = (bit & !KNOWN_ATTRIBUTE_BITS != 0).then_some(("attribute", 0, bit));
            permission.into_iter().chain(attribute)
        })
        .collect();
    // Bits 5 (Eden's `KernelExecute`) and 7 of the permission; bits 6 and 7 of the attributes.
    assert_eq!(unknown.len(), 4);
    for (what, permission, attribute) in unknown {
        let mut case = Case::new(&rich);
        let index = at(&case.body, rich.heap);
        case.body.blocks[index].permission |= permission;
        case.body.blocks[index].attribute |= attribute;
        assert_eq!(
            case.error(),
            SnapshotError::Block {
                index,
                fault: BlockFault::Bits
            },
            "{what} bit {:#x}",
            permission | attribute
        );
    }
}

#[test]
fn a_block_has_only_a_permission_the_kernel_gives_its_state() {
    let rich = rich();
    let tls = thread_local(&rich);
    let read_write = MemoryPermission::from_svc(SvcPermission::ReadWrite).bits();
    let cases: [(&str, u64, u8); 4] = [
        (
            "heap memory read, write and execute",
            rich.heap,
            READ_WRITE_EXECUTE,
        ),
        ("code read, write and execute", CODE, READ_WRITE_EXECUTE),
        // `SetProcessMemoryPermission` would have made it `CodeData`.
        ("code read and write", CODE, read_write),
        (
            "thread-local memory read, write and execute",
            tls,
            READ_WRITE_EXECUTE,
        ),
    ];
    for (name, address, permission) in cases {
        let mut case = Case::new(&rich);
        let index = at(&case.body, address);
        case.body.blocks[index].permission = permission;
        assert_eq!(
            case.error(),
            SnapshotError::Block {
                index,
                fault: BlockFault::Permission
            },
            "{name}"
        );
    }
}

#[test]
fn every_converted_permission_is_accepted_where_the_state_is_not_code() {
    // The heap's first block (`Normal`) with each of the four conversions, read and execute
    // included: module docs, "Decision: no per-state permission table" (only the system call
    // refuses execute on heap memory). Each restores, and keeps its permission.
    let rich = rich();
    for svc in SvcPermission::ALL {
        let permission = MemoryPermission::from_svc(svc);
        let mut case = Case::new(&rich);
        let index = at(&case.body, rich.heap);
        case.body.blocks[index].permission = permission.bits();
        let restored = case.restore().0.unwrap();
        assert_eq!(
            restored.query_info(rich.heap).unwrap().permission(),
            permission,
            "{svc:?}"
        );
    }
    // Code takes every conversion that is not writable. (Read-only text would merge with the
    // read-only data above it, so only the permission check's own verdict is compared.)
    for svc in SvcPermission::ALL {
        let permission = MemoryPermission::from_svc(svc);
        let mut case = Case::new(&rich);
        let index = at(&case.body, CODE);
        case.body.blocks[index].permission = permission.bits();
        let writable = permission.contains(MemoryPermission::USER_WRITE);
        let refused = case.restore().0.err()
            == Some(SnapshotError::Block {
                index,
                fault: BlockFault::Permission,
            });
        assert_eq!(refused, writable, "{svc:?}");
    }
}

#[test]
fn a_lock_over_a_block_whose_state_cannot_be_device_mapped_is_refused() {
    // The thread-local page device shared as one lock over it leaves it, and that lock: live,
    // `lock_device_share` refuses the page, and `unlock_device_share` could never release it.
    let rich = rich();
    let tls = thread_local(&rich);
    let mut case = Case::new(&rich);
    device_share_alone(&mut case.body, tls);
    let index = insert_lock(&mut case.body, tls, PAGE_SIZE);
    assert_eq!(
        case.error(),
        SnapshotError::Lock {
            index,
            fault: LockFault::State
        }
    );
    // What the restore refuses: the live space refuses to lock the page at all.
    let mut space = rich.space;
    assert_eq!(
        space.lock_device_share(tls, PAGE_SIZE),
        Err(SvcError::Kernel(KernelError::InvalidCurrentMemory))
    );
}

#[test]
fn a_lock_over_memory_that_cannot_be_device_mapped_aligned_is_refused() {
    // The first alias page, locked once, made IPC memory without the device flag: device
    // mappable and reference counted, but not aligned (what `lock_device_share` tests). It is
    // no longer mapped physical memory, so the counts follow.
    let rich = rich();
    let state = MemoryState::NonDeviceIpc;
    assert_ne!(state.word() & flag::CAN_DEVICE_MAP, 0);
    assert_ne!(state.word() & flag::REFERENCE_COUNTED, 0);
    assert_eq!(state.word() & flag::CAN_ALIGNED_DEVICE_MAP, 0);
    let mut case = Case::new(&rich);
    let block = at(&case.body, rich.alias);
    assert_eq!(case.body.blocks[block].pages, 1);
    case.body.blocks[block].state = state.word();
    case.body.mapped_physical_memory_size -= PAGE_SIZE;
    case.body.reserved -= PAGE_SIZE;
    let index = case
        .body
        .locks
        .iter()
        .position(|lock| (lock.address, lock.size) == (rich.alias, PAGE_SIZE))
        .unwrap();
    assert_eq!(
        case.error(),
        SnapshotError::Lock {
            index,
            fault: LockFault::State
        }
    );
}

#[test]
fn every_state_a_lock_can_be_taken_on_can_be_unlocked() {
    // Module docs, "The device-share locks' states": the unlock's flag adds nothing to the
    // lock's for the states there are, so check 7's test of it cannot be told apart from
    // the lock's alone by any block. This pins that it stays so.
    for &state in MemoryState::ALL {
        let word = state.word();
        if word & DEVICE_SHARE_LOCK_STATE == DEVICE_SHARE_LOCK_STATE {
            assert_eq!(
                word & DEVICE_SHARE_UNLOCK_STATE,
                DEVICE_SHARE_UNLOCK_STATE,
                "{}",
                state.name()
            );
        }
    }
}

#[test]
fn a_lock_over_a_block_neither_shared_nor_lockable_is_reported_as_not_shared() {
    // The thread-local page as it is (not device shared, a state with no device mapping), and a
    // lock over it: both faults, and the earlier check's is reported.
    let rich = rich();
    let tls = thread_local(&rich);
    let mut case = Case::new(&rich);
    let index = insert_lock(&mut case.body, tls, PAGE_SIZE);
    assert_eq!(
        case.error(),
        SnapshotError::Lock {
            index,
            fault: LockFault::NotShared
        }
    );
}

#[test]
fn the_lock_state_check_takes_every_block_of_the_range() {
    // The generated code's two pages split into a page of transfer memory (whose state can be
    // device mapped) and a page of generated code (whose state cannot), in either order, both
    // device shared as one lock over the two leaves them, and that lock: the block that cannot
    // be device mapped is refused whether it is the range's first or its last.
    let rich = rich();
    let lockable = MemoryState::Transferred;
    let not_lockable = MemoryState::GeneratedCode;
    let flags = DEVICE_SHARE_LOCK_STATE | DEVICE_SHARE_UNLOCK_STATE;
    assert_eq!(lockable.word() & flags, flags);
    assert_ne!(not_lockable.word() & flags, flags);
    for (states, expected) in [
        ([lockable, lockable], None),
        ([lockable, not_lockable], Some(LockFault::State)),
        ([not_lockable, lockable], Some(LockFault::State)),
        // Device mappable, aligned too, but not reference counted.
        ([lockable, MemoryState::IoMemory], Some(LockFault::State)),
    ] {
        let mut case = Case::new(&rich);
        let index = at(&case.body, GENERATED);
        let mut upper = case.body.blocks[index];
        assert_eq!(upper.pages, 2);
        upper.address += PAGE_SIZE;
        upper.pages = 1;
        case.body.blocks[index].pages = 1;
        case.body.blocks.insert(index + 1, upper);
        let barriers = [DisableMerge::DEVICE_LEFT, DisableMerge::DEVICE_RIGHT];
        for (at, (state, barrier)) in [index, index + 1]
            .into_iter()
            .zip(states.into_iter().zip(barriers))
        {
            let block = &mut case.body.blocks[at];
            block.state = state.word();
            block.attribute |= MemoryAttribute::DEVICE_SHARED.bits();
            block.device_use_count = 1;
            block.disable_merge |= barrier.bits();
            if at == index {
                block.device_disable_merge_left_count = 1;
            } else {
                block.device_disable_merge_right_count = 1;
            }
        }
        let lock = insert_lock(&mut case.body, GENERATED, 2 * PAGE_SIZE);
        let expected = expected.map(|fault| SnapshotError::Lock { index: lock, fault });
        assert_eq!(case.restore().0.err(), expected, "{states:?}");
    }
}

#[test]
fn a_shared_mapping_must_have_its_shared_memorys_permission() {
    // The read-only shared memory's block saved read-write: live, `MapSharedMemory` refuses
    // any permission but the user permission.
    let rich = rich();
    let mut case = Case::new(&rich);
    let index = at(&case.body, case.body.shared[1].address);
    assert_eq!(
        case.body.blocks[index].permission,
        MemoryPermission::from_svc(SvcPermission::Read).bits()
    );
    case.body.blocks[index].permission = MemoryPermission::USER_READ_WRITE.bits();
    assert_eq!(
        case.error(),
        SnapshotError::Shared {
            index: 1,
            fault: SharedFault::Permission
        }
    );
}

#[test]
fn a_shared_memory_smaller_than_its_mapping_is_the_wrong_size() {
    // The first mapping (two pages) resolved to the one-page object, which is also read-only
    // where the mapping is read-write: the size is checked first.
    let rich = rich();
    let mut case = Case::new(&rich);
    let small = case.objects[&rich.shared[1]];
    assert!(small.size() < case.body.shared[0].size);
    case.objects.insert(rich.shared[0], small);
    assert_eq!(
        case.error(),
        SnapshotError::Shared {
            index: 0,
            fault: SharedFault::Size
        }
    );
}

#[test]
fn a_shared_mapping_saved_twice_is_out_of_order() {
    let rich = rich();
    let mut case = Case::new(&rich);
    let first = case.body.shared[0];
    case.body.shared.insert(1, first);
    assert_eq!(
        case.error(),
        SnapshotError::Shared {
            index: 1,
            fault: SharedFault::Order
        }
    );
}

#[test]
fn normal_memory_partly_outside_the_alias_region_is_out_of_its_region() {
    // The first alias block's start moved one page down into the free block below it: every
    // page but its first is in the alias region.
    let rich = rich();
    let mut case = Case::new(&rich);
    let index = at(&case.body, rich.alias);
    let below = index - 1;
    assert_eq!(case.body.blocks[below].state, MemoryState::Free.word());
    assert!(case.body.blocks[below].pages > 1);
    case.body.blocks[below].pages -= 1;
    case.body.blocks[index].address -= PAGE_SIZE;
    case.body.blocks[index].pages += 1;
    assert_eq!(
        case.error(),
        SnapshotError::Block {
            index,
            fault: BlockFault::Region
        }
    );
}

#[test]
fn a_lock_whose_size_is_not_whole_pages_is_out_of_range() {
    // A page-aligned address, a size one byte over a page.
    let rich = rich();
    let mut case = Case::new(&rich);
    let index = insert_lock(&mut case.body, RODATA, PAGE_SIZE + 1);
    assert_eq!(
        case.error(),
        SnapshotError::Lock {
            index,
            fault: LockFault::Range
        }
    );
}

#[test]
fn the_codes_last_block_must_be_code_too() {
    // The block that ends where the code ends, made alias code data: in the code region, but
    // not the code.
    let rich = rich();
    let mut case = Case::new(&rich);
    let code_end = case.body.request.code_address + case.body.request.code_size;
    let last = case
        .body
        .blocks
        .iter()
        .position(|block| block.address + block.pages * PAGE_SIZE == code_end)
        .unwrap();
    assert_ne!(case.body.blocks[last].address, CODE);
    case.body.blocks[last].state = MemoryState::AliasCodeData.word();
    assert_eq!(case.error(), SnapshotError::CodeRange);
}

#[test]
fn the_new_errors_read_as_plain_words() {
    assert_eq!(
        SnapshotError::Block {
            index: 2,
            fault: BlockFault::Bits
        }
        .to_string(),
        "block 2 of the address space snapshot has a permission or attribute bit no memory block has"
    );
    assert_eq!(
        SnapshotError::Block {
            index: 2,
            fault: BlockFault::Permission
        }
        .to_string(),
        "block 2 of the address space snapshot has a permission the kernel never gives its state"
    );
    assert_eq!(
        SnapshotError::Block {
            index: 0,
            fault: BlockFault::ChangedAttribute
        }
        .to_string(),
        "block 0 of the address space snapshot is uncached or GPU accessible, which its memory state does not allow"
    );
    assert_eq!(
        SnapshotError::Block {
            index: 0,
            fault: BlockFault::PermissionLock
        }
        .to_string(),
        "block 0 of the address space snapshot is permission locked, which its memory state does not allow"
    );
    assert_eq!(
        SnapshotError::Shared {
            index: 1,
            fault: SharedFault::Permission
        }
        .to_string(),
        "shared memory mapping 1 of the address space snapshot is mapped with a permission other than its shared memory's"
    );
    assert_eq!(
        SnapshotError::Lock {
            index: 0,
            fault: LockFault::State
        }
        .to_string(),
        "device-share lock 0 of the address space snapshot covers a block whose memory state cannot be shared with a device"
    );
    let alias = |index, fault| SnapshotError::Alias { index, fault }.to_string();
    for (fault, words) in [
        (
            AliasFault::Shape,
            "alias 1 of the address space snapshot is not whole pages inside the address space",
        ),
        (
            AliasFault::Order,
            "alias 1 of the address space snapshot is out of order or overlaps the one before it",
        ),
        (
            AliasFault::Region,
            "alias 1 of the address space snapshot is outside the stack region",
        ),
        (
            AliasFault::Overlap,
            "alias 1 of the address space snapshot has a source that overlaps its destination",
        ),
        (
            AliasFault::Source,
            "alias 1 of the address space snapshot has a source that is not memory MapMemory locked",
        ),
        (
            AliasFault::Destination,
            "alias 1 of the address space snapshot has a destination that is not the stack memory MapMemory makes",
        ),
        (
            AliasFault::SourceTaken,
            "alias 1 of the address space snapshot has a source that an alias before it maps too",
        ),
    ] {
        assert_eq!(alias(1, fault), words);
    }
}

#[test]
fn an_unknown_merge_barrier_bit_restores_as_it_is() {
    // Module docs, "Not checked here": the barriers other than the IPC and device ones are not
    // checked, so a bit `DisableMerge` does not name (bit 7) restores and is kept.
    let unknown = 1_u8 << 7;
    let all_named = (DisableMerge::NORMAL
        | DisableMerge::DEVICE_LEFT
        | DisableMerge::IPC_LEFT
        | DisableMerge::LOCKED
        | DisableMerge::DEVICE_RIGHT)
        .bits();
    assert_eq!(all_named & unknown, 0);
    let rich = rich();
    let tls = thread_local(&rich);
    let mut case = Case::new(&rich);
    let index = at(&case.body, tls);
    case.body.blocks[index].disable_merge |= unknown;
    let restored = case.restore().0.unwrap();
    let block = restored.query_info(tls).unwrap();
    assert_eq!(block.disable_merge().bits() & unknown, unknown);
}
