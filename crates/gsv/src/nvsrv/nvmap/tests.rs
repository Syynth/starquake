//! nvmap's ioctls on byte fixtures, and its saved table.
//!
//! Every argument is laid out by hand, word by word, from Eden's structs
//! (`eden/src/core/hle/service/nvdrv/devices/nvmap.h` @ cf654052, `IocCreateParams` and the rest,
//! with their `static_assert`ed sizes), and every ioctl number is written as libnx sends it
//! (`libnx/nx/source/nvidia/ioctl/nvmap.c` @ b156b477: `_NV_IOWR(0x01, cmd, data)`, both
//! directions, the struct's size).

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use super::*;

/// `_NV_IOWR(0x01, 0x01, data)` with an 8-byte `data`.
const CREATE: u32 = 0xC008_0101;
/// `_NV_IOWR(0x01, 0x03, data)`, 8 bytes.
const FROM_ID: u32 = 0xC008_0103;
/// `_NV_IOWR(0x01, 0x04, data)`, 32 bytes.
const ALLOC: u32 = 0xC020_0104;
/// `_NV_IOWR(0x01, 0x05, data)`, 24 bytes.
const FREE: u32 = 0xC018_0105;
/// `_NV_IOWR(0x01, 0x09, data)`, 12 bytes.
const PARAM: u32 = 0xC00C_0109;
/// `_NV_IOWR(0x01, 0x0E, data)`, 8 bytes.
const GET_ID: u32 = 0xC008_010E;

/// Little-endian words as bytes.
fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// The low and high words of a `u64`.
fn halves(value: u64) -> [u32; 2] {
    let bytes = value.to_le_bytes();
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    [word(0), word(4)]
}

/// Runs the ioctl `raw` with `input`; it must be one nvmap knows.
fn call(map: &mut NvMap, raw: u32, input: &[u8]) -> (NvResult, Vec<u8>) {
    map.ioctl(Ioctl::from_raw(raw), input)
        .expect("an ioctl nvmap knows")
}

/// `IocCreateParams`: size, then the handle the caller left there.
fn create_args(size: u32, handle: u32) -> Vec<u8> {
    le(&[size, handle])
}

/// Creates a handle of `size` bytes; its id.
fn create(map: &mut NvMap, size: u32) -> u32 {
    let (result, output) = call(map, CREATE, &create_args(size, 0));
    assert_eq!(result, NvResult::Success);
    u32::from_le_bytes(output[4..8].try_into().unwrap())
}

/// `IocAllocParams` (32 bytes): handle, heap mask, flags, alignment, the kind byte and seven
/// bytes of padding, the address.
fn alloc_args(
    handle: u32,
    heap_mask: u32,
    flags: u32,
    align: u32,
    kind: u8,
    address: u64,
) -> Vec<u8> {
    let [low, high] = halves(address);
    le(&[
        handle,
        heap_mask,
        flags,
        align,
        u32::from(kind),
        0,
        low,
        high,
    ])
}

/// Allocates `handle` at `address` with `align`, kind 0, cached.
fn alloc(map: &mut NvMap, handle: u32, align: u32, address: u64) -> NvResult {
    call(map, ALLOC, &alloc_args(handle, 0, 0, align, 0, address)).0
}

/// `IocFreeParams` (24 bytes) as the caller sends it: the handle, four bytes of padding, and
/// the outputs as the caller left them.
fn free_args(handle: u32, stale: u32) -> Vec<u8> {
    le(&[handle, stale, stale, stale, stale, stale])
}

/// `IocParamParams`: handle, parameter, the result as the caller left it.
fn param_args(handle: u32, param: u32, stale: u32) -> Vec<u8> {
    le(&[handle, param, stale])
}

/// The answer to parameter `param` of `handle`.
fn param_of(map: &mut NvMap, handle: u32, param: u32) -> (NvResult, u32) {
    let (result, output) = call(map, PARAM, &param_args(handle, param, 0xEEEE_EEEE));
    (
        result,
        u32::from_le_bytes(output[8..12].try_into().unwrap()),
    )
}

/// What a page is, as the tests write it (Eden's `YUZU_PAGESIZE`, 4 KiB).
const PAGE: u32 = 0x1000;

#[test]
fn the_ioctl_numbers_are_libnx_s() {
    let cases = [
        (CREATE, command::CREATE, layout::CREATE_BYTES),
        (FROM_ID, command::FROM_ID, layout::FROM_ID_BYTES),
        (ALLOC, command::ALLOC, layout::ALLOC_BYTES),
        (FREE, command::FREE, layout::FREE_BYTES),
        (PARAM, command::PARAM, layout::PARAM_BYTES),
        (GET_ID, command::GET_ID, layout::GET_ID_BYTES),
    ];
    for (raw, command, bytes) in cases {
        let ioctl = Ioctl::from_raw(raw);
        assert_eq!(ioctl.group(), command::GROUP, "{raw:#x}");
        assert_eq!(ioctl.command(), command, "{raw:#x}");
        assert_eq!(usize::try_from(ioctl.size()).unwrap(), bytes, "{raw:#x}");
    }
    assert_eq!(u64::from(PAGE), PAGE_SIZE);
}

/// Each struct's fields are where Eden's struct puts them, and the struct's size is its
/// `static_assert`ed one.
#[test]
fn the_layouts_are_eden_s() {
    use layout::*;
    assert_eq!((CREATE_SIZE, CREATE_HANDLE, CREATE_BYTES), (0, 4, 8));
    assert_eq!((FROM_ID_ID, FROM_ID_HANDLE, FROM_ID_BYTES), (0, 4, 8));
    assert_eq!(
        (
            ALLOC_HANDLE,
            ALLOC_HEAP_MASK,
            ALLOC_FLAGS,
            ALLOC_ALIGN,
            ALLOC_KIND,
            ALLOC_ADDRESS,
            ALLOC_BYTES
        ),
        (0, 4, 8, 12, 16, 24, 32)
    );
    assert_eq!(
        (FREE_HANDLE, FREE_ADDRESS, FREE_SIZE, FREE_FLAGS, FREE_BYTES),
        (0, 8, 16, 20, 24)
    );
    assert_eq!(
        (PARAM_HANDLE, PARAM_PARAM, PARAM_RESULT, PARAM_BYTES),
        (0, 4, 8, 12)
    );
    assert_eq!((GET_ID_ID, GET_ID_HANDLE, GET_ID_BYTES), (0, 4, 8));
}

// ----- IocCreate -------------------------------------------------------------------------------

#[test]
fn create_hands_out_ids_four_apart_and_rounds_to_pages() {
    let mut map = NvMap::new();
    // The output is the struct with the id written over the caller's stale handle field.
    let (result, output) = call(&mut map, CREATE, &create_args(1, 0x7777_7777));
    assert_eq!((result, output), (NvResult::Success, create_args(1, 4)));
    assert_eq!(create(&mut map, PAGE), 8);
    assert_eq!(create(&mut map, PAGE + 1), 12);
    let sizes = [(4, 1, PAGE), (8, PAGE, PAGE), (12, PAGE + 1, 2 * PAGE)];
    for (id, created, rounded) in sizes {
        let handle = map.handle(id).unwrap();
        assert_eq!(handle.created_size(), created);
        assert_eq!(handle.size(), u64::from(rounded));
        assert_eq!(handle.dupes(), 1);
        assert_eq!(handle.allocation(), None);
    }
    assert_eq!(map.len(), 3);
    assert!(!map.is_empty());
    assert!(NvMap::default().is_empty());
}

/// Eden rounds the size up as a `u32`, so a size within a page of the top wraps to 0 and is
/// refused like 0 itself; the last size that does not wrap is accepted.
#[test]
fn create_refuses_sizes_that_round_to_nothing() {
    let largest = u32::try_from(MAX_HANDLE_SIZE).unwrap();
    assert_eq!(largest, 0xFFFF_F000);
    let mut map = NvMap::new();
    for size in [0, largest + 1, u32::MAX] {
        let (result, output) = call(&mut map, CREATE, &create_args(size, 0x7777_7777));
        // Refused, and the struct goes back as it came.
        assert_eq!(
            (result, output),
            (NvResult::BadValue, create_args(size, 0x7777_7777))
        );
    }
    assert!(map.is_empty());
    let id = create(&mut map, largest);
    assert_eq!(map.handle(id).unwrap().size(), MAX_HANDLE_SIZE);
    assert_eq!(create(&mut map, 1), id + HANDLE_ID_INCREMENT);
}

/// A table of `count` unallocated one-page handles with ids 4, 8, ...
fn table_of(count: usize) -> NvMap {
    let handle = NvMapHandle {
        size: PAGE,
        dupes: 1,
        allocation: None,
    };
    let ids = (1..=u32::try_from(count).unwrap()).map(|n| n * HANDLE_ID_INCREMENT);
    let handles: BTreeMap<u32, NvMapHandle> = ids.map(|id| (id, handle)).collect();
    let next_id = (u64::try_from(count).unwrap() + 1) * u64::from(HANDLE_ID_INCREMENT);
    NvMap {
        handles,
        next_id,
        ..NvMap::new()
    }
}

#[test]
fn create_refuses_a_full_table_and_spent_ids() {
    // One below the limit: the last handle fits.
    let mut map = table_of(MAX_HANDLES - 1);
    create(&mut map, 1);
    assert_eq!(map.len(), MAX_HANDLES);
    // At the limit: refused, nothing made.
    let (result, output) = call(&mut map, CREATE, &create_args(1, 0x7777_7777));
    assert_eq!(
        (result, output),
        (NvResult::InsufficientMemory, create_args(1, 0x7777_7777))
    );
    assert_eq!(map.len(), MAX_HANDLES);
    // A bad size is reported before a full table.
    assert_eq!(
        call(&mut map, CREATE, &create_args(0, 0)).0,
        NvResult::BadValue
    );

    // The last id is handed out; after it, ids are spent.
    let mut map = NvMap {
        next_id: HANDLE_ID_END - u64::from(HANDLE_ID_INCREMENT),
        ..NvMap::new()
    };
    assert_eq!(create(&mut map, 1), 0xFFFF_FFFC);
    assert_eq!(map.next_id, HANDLE_ID_END);
    let (result, output) = call(&mut map, CREATE, &create_args(1, 0x7777_7777));
    assert_eq!(
        (result, output),
        (NvResult::InsufficientMemory, create_args(1, 0x7777_7777))
    );
    assert_eq!(map.len(), 1);
    assert_eq!(map.next_id, HANDLE_ID_END);
}

// ----- IocAlloc --------------------------------------------------------------------------------

#[test]
fn alloc_records_the_handle_s_allocation() {
    let mut map = NvMap::new();
    let id = create(&mut map, 3 * PAGE);
    let input = alloc_args(
        id,
        0x5151_5151,
        MAP_UNCACHED | 0x20,
        0x4000,
        0x7E,
        0x0000_0051_2345_6000,
    );
    let (result, output) = call(&mut map, ALLOC, &input);
    // Nothing in the struct changes: the alignment is above a page already.
    assert_eq!((result, output), (NvResult::Success, input));
    let allocation = *map.handle(id).unwrap().allocation().unwrap();
    assert!(allocation.uncached());
    assert_eq!(allocation.align(), 0x4000);
    assert_eq!(allocation.kind(), 0x7E);
    assert_eq!(allocation.address(), 0x0000_0051_2345_6000);
    // A second allocation of the same handle is refused, and the first stands.
    let again = alloc_args(id, 0, 0, 0x1_0000, 1, 0x2000);
    assert_eq!(
        call(&mut map, ALLOC, &again),
        (NvResult::InsufficientMemory, again)
    );
    assert_eq!(map.handle(id).unwrap().allocation(), Some(&allocation));
}

/// The alignment is checked, then raised to a page and written back, before the handle is
/// looked up: so a refusal for the handle still hands the raised alignment back.
#[test]
fn alloc_checks_the_alignment_then_raises_it_before_the_handle() {
    let mut map = NvMap::new();
    let id = create(&mut map, PAGE);
    // Handle 0 comes first: the alignment is left as sent (0 would have been raised).
    let zero = alloc_args(0, 0, 0, 0, 0, 0x1000);
    assert_eq!(call(&mut map, ALLOC, &zero), (NvResult::BadValue, zero));
    // An alignment that is not a power of two: refused as sent.
    for align in [3, 0x1800, 0xFFFF_FFFF] {
        let bad = alloc_args(id, 0, 0, align, 0, 0x1000);
        assert_eq!(
            call(&mut map, ALLOC, &bad),
            (NvResult::BadValue, bad),
            "{align:#x}"
        );
    }
    // An unknown handle: refused, with the alignment raised in the struct.
    let unknown = alloc_args(id + HANDLE_ID_INCREMENT, 0, 0, 0, 0, 0x1000);
    let raised = alloc_args(id + HANDLE_ID_INCREMENT, 0, 0, PAGE, 0, 0x1000);
    assert_eq!(
        call(&mut map, ALLOC, &unknown),
        (NvResult::BadValue, raised)
    );
    assert_eq!(map.handle(id).unwrap().allocation(), None);
}

/// 0 and every power of two below a page become a page; a page and above are kept.
#[test]
fn the_alignment_is_at_least_a_page() {
    for (sent, kept) in [
        (0, PAGE),
        (1, PAGE),
        (PAGE / 2, PAGE),
        (PAGE, PAGE),
        (PAGE * 2, PAGE * 2),
        (1 << 31, 1 << 31),
    ] {
        let mut map = NvMap::new();
        let id = create(&mut map, PAGE);
        let (result, output) = call(&mut map, ALLOC, &alloc_args(id, 0, 0, sent, 0, 0x1000));
        assert_eq!(result, NvResult::Success, "{sent:#x}");
        assert_eq!(output, alloc_args(id, 0, 0, kept, 0, 0x1000), "{sent:#x}");
        assert_eq!(map.handle(id).unwrap().allocation().unwrap().align(), kept);
    }
}

/// The handle's page-rounded size from its address must end below 2^64: a range ending at
/// 2^64 - 1 fits, and one a byte further, ending exactly at 2^64, is refused and nothing is
/// stored (a starquake check; Eden stores it).
#[test]
fn an_allocation_past_the_address_space_is_refused() {
    let mut map = NvMap::new();
    let id = create(&mut map, PAGE + 1);
    let size = map.handle(id).unwrap().size();
    let last = u64::MAX - size;
    let past = alloc_args(id, 0, 0, 0, 0, last + 1);
    let raised = alloc_args(id, 0, 0, PAGE, 0, last + 1);
    assert_eq!(call(&mut map, ALLOC, &past), (NvResult::BadValue, raised));
    assert_eq!(map.handle(id).unwrap().allocation(), None);
    assert_eq!(alloc(&mut map, id, 0, last), NvResult::Success);
    assert_eq!(
        map.handle(id).unwrap().allocation().unwrap().address(),
        last
    );
}

/// Eden allocates a handle with no address (and logs that it cannot map it), so this does too.
#[test]
fn an_allocation_with_no_address_is_accepted() {
    let mut map = NvMap::new();
    let id = create(&mut map, PAGE);
    assert_eq!(alloc(&mut map, id, 0, 0), NvResult::Success);
    assert_eq!(map.handle(id).unwrap().allocation().unwrap().address(), 0);
}

// ----- IocFromId -------------------------------------------------------------------------------

#[test]
fn from_id_adds_a_reference_to_an_allocated_handle_only() {
    let mut map = NvMap::new();
    let id = create(&mut map, PAGE);
    // Id 0, an id with no handle, and an unallocated handle: refused as sent.
    for asked in [0, id + HANDLE_ID_INCREMENT, id] {
        let args = le(&[asked, 0x7777_7777]);
        assert_eq!(
            call(&mut map, FROM_ID, &args),
            (NvResult::BadValue, args),
            "{asked}"
        );
    }
    assert_eq!(map.handle(id).unwrap().dupes(), 1);
    alloc(&mut map, id, 0, 0x1000);
    let (result, output) = call(&mut map, FROM_ID, &le(&[id, 0x7777_7777]));
    assert_eq!((result, output), (NvResult::Success, le(&[id, id])));
    assert_eq!(map.handle(id).unwrap().dupes(), 2);
}

#[test]
fn from_id_stops_at_the_duplicate_limit() {
    let mut map = NvMap::new();
    let id = create(&mut map, PAGE);
    alloc(&mut map, id, 0, 0x1000);
    map.handles.get_mut(&id).unwrap().dupes = MAX_DUPES - 1;
    assert_eq!(call(&mut map, FROM_ID, &le(&[id, 0])).0, NvResult::Success);
    assert_eq!(map.handle(id).unwrap().dupes(), MAX_DUPES);
    let args = le(&[id, 0x7777_7777]);
    assert_eq!(call(&mut map, FROM_ID, &args), (NvResult::BadValue, args));
    assert_eq!(map.handle(id).unwrap().dupes(), MAX_DUPES);
    assert_eq!(MAX_DUPES, 0x7FFF_FFFF);
}

// ----- IocFree ---------------------------------------------------------------------------------

#[test]
fn free_of_nothing_succeeds_and_writes_nothing() {
    let mut map = NvMap::new();
    let id = create(&mut map, PAGE);
    for handle in [0, id + HANDLE_ID_INCREMENT] {
        let args = free_args(handle, 0x7777_7777);
        assert_eq!(
            call(&mut map, FREE, &args),
            (NvResult::Success, args),
            "{handle}"
        );
    }
    assert_eq!(map.len(), 1);
}

/// Each free drops one reference and reports the handle; the last one removes it.
#[test]
fn free_drops_a_reference_and_the_last_removes_the_handle() {
    let mut map = NvMap::new();
    let id = create(&mut map, PAGE + 1);
    let address = 0x0000_0051_2345_6000;
    let input = alloc_args(id, 0, MAP_UNCACHED, 0, 0x2A, address);
    assert_eq!(call(&mut map, ALLOC, &input).0, NvResult::Success);
    call(&mut map, FROM_ID, &le(&[id, 0]));
    let [low, high] = halves(address);
    // Handle, the padding as sent, the address, the page-rounded size, the uncached bit.
    let reported = le(&[id, 0x7777_7777, low, high, 2 * PAGE, MAP_UNCACHED]);
    assert_eq!(
        call(&mut map, FREE, &free_args(id, 0x7777_7777)),
        (NvResult::Success, reported.clone())
    );
    assert_eq!(map.handle(id).unwrap().dupes(), 1);
    assert_eq!(
        call(&mut map, FREE, &free_args(id, 0x7777_7777)),
        (NvResult::Success, reported)
    );
    assert_eq!(map.handle(id), None);
    // Gone: a third free finds nothing and writes nothing.
    let args = free_args(id, 0x7777_7777);
    assert_eq!(call(&mut map, FREE, &args), (NvResult::Success, args));
}

/// An unallocated handle reports address 0 and no flags; a cached allocation reports flags 0
/// whatever other bits it was allocated with.
#[test]
fn free_reports_only_the_uncached_bit() {
    let mut map = NvMap::new();
    let unallocated = create(&mut map, PAGE);
    assert_eq!(
        call(&mut map, FREE, &free_args(unallocated, 0x7777_7777)),
        (
            NvResult::Success,
            le(&[unallocated, 0x7777_7777, 0, 0, PAGE, 0])
        )
    );
    let cached = create(&mut map, PAGE);
    let input = alloc_args(cached, 0, !MAP_UNCACHED, 0, 0, 0x3000);
    assert_eq!(call(&mut map, ALLOC, &input).0, NvResult::Success);
    assert!(!map.handle(cached).unwrap().allocation().unwrap().uncached());
    assert_eq!(
        call(&mut map, FREE, &free_args(cached, 0x7777_7777)),
        (
            NvResult::Success,
            le(&[cached, 0x7777_7777, 0x3000, 0, PAGE, 0])
        )
    );
}

// ----- IocParam and IocGetId -------------------------------------------------------------------

#[test]
fn param_answers_each_parameter_before_and_after_allocation() {
    let mut map = NvMap::new();
    let id = create(&mut map, PAGE + 3);
    let before = [
        (param::SIZE, PAGE + 3),
        (param::ALIGNMENT, 0),
        (param::BASE, 0xFFFF_FFEA),
        (param::HEAP, 0),
        (param::KIND, 0),
        (param::IS_SHARED_MEM_MAPPED, 0),
    ];
    for (asked, answer) in before {
        assert_eq!(
            param_of(&mut map, id, asked),
            (NvResult::Success, answer),
            "{asked}"
        );
    }
    let input = alloc_args(id, 0, 0, 0x8000, 0xC3, 0x5000);
    assert_eq!(call(&mut map, ALLOC, &input).0, NvResult::Success);
    let after = [
        (param::SIZE, PAGE + 3),
        (param::ALIGNMENT, 0x8000),
        (param::BASE, 0xFFFF_FFEA),
        (param::HEAP, 0x4000_0000),
        (param::KIND, 0xC3),
        (param::IS_SHARED_MEM_MAPPED, 0),
    ];
    for (asked, answer) in after {
        assert_eq!(
            param_of(&mut map, id, asked),
            (NvResult::Success, answer),
            "{asked}"
        );
    }
    assert_eq!((BASE_ANSWER, HEAP_ANSWER), (0xFFFF_FFEA, 0x4000_0000));
}

#[test]
fn param_refuses_an_unknown_handle_or_parameter() {
    let mut map = NvMap::new();
    let id = create(&mut map, PAGE);
    let cases = [
        (0, param::SIZE),
        (id + HANDLE_ID_INCREMENT, param::SIZE),
        (id, 0),
        (id, param::IS_SHARED_MEM_MAPPED + 1),
        (id, u32::MAX),
    ];
    for (handle, asked) in cases {
        let args = param_args(handle, asked, 0x7777_7777);
        assert_eq!(
            call(&mut map, PARAM, &args),
            (NvResult::BadValue, args),
            "{handle} {asked}"
        );
    }
}

#[test]
fn get_id_answers_the_handle_s_own_id() {
    let mut map = NvMap::new();
    let id = create(&mut map, PAGE);
    assert_eq!(
        call(&mut map, GET_ID, &le(&[0x7777_7777, id])),
        (NvResult::Success, le(&[id, id]))
    );
    let zero = le(&[0x7777_7777, 0]);
    assert_eq!(call(&mut map, GET_ID, &zero), (NvResult::BadValue, zero));
    let unknown = le(&[0x7777_7777, id + HANDLE_ID_INCREMENT]);
    assert_eq!(
        call(&mut map, GET_ID, &unknown),
        (NvResult::AccessDenied, unknown)
    );
}

// ----- Dispatch --------------------------------------------------------------------------------

/// Only group 1's six commands are nvmap's; any other group or command is unknown.
#[test]
fn every_other_group_and_command_is_unknown() {
    let known = [
        command::CREATE,
        command::FROM_ID,
        command::ALLOC,
        command::FREE,
        command::PARAM,
        command::GET_ID,
    ];
    let mut map = NvMap::new();
    for group in 0..=0xFF_u32 {
        for cmd in 0..=0xFF_u32 {
            let raw = 0xC000_0000 | group << 8 | cmd;
            let answer = map.ioctl(Ioctl::from_raw(raw), &[]);
            let expected = group == command::GROUP && known.contains(&cmd);
            assert_eq!(answer.is_some(), expected, "{raw:#x}");
        }
    }
}

/// The size and direction bits are not looked at: a create with neither direction and size 0
/// creates, and one with a size other than the struct's too.
#[test]
fn the_size_and_direction_bits_are_not_checked() {
    let mut map = NvMap::new();
    let (result, output) = call(&mut map, 0x0000_0101, &create_args(PAGE, 0));
    assert_eq!((result, output), (NvResult::Success, create_args(PAGE, 4)));
    let (result, output) = call(&mut map, 0x8123_0101, &create_args(PAGE, 0));
    assert_eq!((result, output), (NvResult::Success, create_args(PAGE, 8)));
}

/// A short argument is read as zeros past its end, a long one is cut to the struct.
#[test]
fn a_short_argument_reads_zeros_and_a_long_one_is_cut() {
    let mut map = NvMap::new();
    // Only the size's first two bytes: 0x1001. The handle field comes back as the id.
    let (result, output) = call(&mut map, CREATE, &[0x01, 0x10]);
    assert_eq!(
        (result, output),
        (NvResult::Success, create_args(0x1001, 4))
    );
    // Nothing at all: size 0.
    assert_eq!(
        call(&mut map, CREATE, &[]),
        (NvResult::BadValue, create_args(0, 0))
    );
    // Twelve bytes: the last four are not part of the struct and do not come back.
    let mut long = create_args(PAGE, 0);
    long.extend([0xAB; 4]);
    assert_eq!(
        call(&mut map, CREATE, &long),
        (NvResult::Success, create_args(PAGE, 8))
    );
}

fn config() -> ProptestConfig {
    Config {
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x5EED_0187),
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(config())]

    /// An allocation's fields, laid out by hand and each read back through another ioctl (kind
    /// and alignment through `IocParam`, address and flags through `IocFree`): swapping any two
    /// of `IocAlloc`'s reads shows, since the fields' values are drawn apart.
    #[test]
    fn an_allocation_reads_each_field_from_its_own_place(
        heap_mask: u32,
        uncached: bool,
        other_flags in any::<u32>().prop_map(|flags| flags & !MAP_UNCACHED),
        align_shift in 13..=31_u32,
        kind in 1..=u8::MAX,
        address in 1..=u64::MAX - u64::from(PAGE),
    ) {
        let mut map = NvMap::new();
        let id = create(&mut map, PAGE);
        let flags = other_flags | if uncached { MAP_UNCACHED } else { 0 };
        let align = 1 << align_shift;
        let input = alloc_args(id, heap_mask, flags, align, kind, address);
        prop_assert_eq!(call(&mut map, ALLOC, &input), (NvResult::Success, input));
        prop_assert_eq!(param_of(&mut map, id, param::KIND), (NvResult::Success, u32::from(kind)));
        prop_assert_eq!(param_of(&mut map, id, param::ALIGNMENT), (NvResult::Success, align));
        let [low, high] = halves(address);
        let flags = if uncached { MAP_UNCACHED } else { 0 };
        prop_assert_eq!(
            call(&mut map, FREE, &free_args(id, 0)),
            (NvResult::Success, le(&[id, 0, low, high, PAGE, flags]))
        );
    }
}

// ----- The saved table -------------------------------------------------------------------------

/// The saved layout of a table, as `postcard` writes it: the handle count, then each id, created
/// size, reference count and allocation (an `Option` tag, then uncached, alignment, kind and
/// address), then the next id. Two handles, one allocated and one not, every value distinct and
/// non-zero where the table allows it.
#[test]
fn the_saved_table_s_layout_is_pinned() {
    let map = NvMap {
        handles: BTreeMap::from([
            (
                4,
                NvMapHandle {
                    size: 0x11,
                    dupes: 0x12,
                    allocation: Some(Allocation {
                        uncached: true,
                        align: 0x1000,
                        kind: 0x13,
                        address: 0x14,
                    }),
                },
            ),
            (
                8,
                NvMapHandle {
                    size: 0x21,
                    dupes: 1,
                    allocation: None,
                },
            ),
        ]),
        next_id: 0x3C,
        ..NvMap::new()
    };
    let bytes = postcard::to_allocvec(&map).unwrap();
    assert_eq!(
        bytes,
        [
            2, // two handles
            4, 0x11, 0x12, 1, 1, 0x80, 0x20, 0x13,
            0x14, // id 4: size, dupes, Some(true, 0x1000 as a varint, kind, address)
            8, 0x21, 1, 0,    // id 8: size, dupes, None
            0x3C, // the next id
        ]
    );
    assert_eq!(postcard::from_bytes::<NvMap>(&bytes).unwrap(), map);
}

/// A table that is valid, to break one field at a time: two handles, one allocated with two
/// references, and the next id past both.
fn valid() -> NvMap {
    let mut map = NvMap::new();
    let first = create(&mut map, PAGE + 1);
    create(&mut map, PAGE);
    alloc(&mut map, first, 0, 0x1000);
    call(&mut map, FROM_ID, &le(&[first, 0]));
    // The allocation's lock step, which a request would take to the kernel (a lock at address 0
    // is refused there), so the table holds no lock: all a saved table can say.
    assert_eq!(map.take_steps().len(), 1);
    map
}

/// `map` saved and read back.
fn reload(map: &NvMap) -> Result<NvMap, NvMapStateError> {
    NvMap::try_from(SavedNvMap {
        handles: map.handles.clone(),
        next_id: map.next_id,
    })
}

/// Handle `id` of `map`, to change.
fn handle_of(map: &mut NvMap, id: u32) -> &mut NvMapHandle {
    map.handles.get_mut(&id).unwrap()
}

/// The allocation of handle 4 of [`valid`]'s table, to change.
fn allocation_of(map: &mut NvMap) -> &mut Allocation {
    handle_of(map, 4).allocation.as_mut().unwrap()
}

/// [`valid`]'s table with `change` made.
fn valid_with(change: impl Fn(&mut NvMap)) -> NvMap {
    let mut map = valid();
    change(&mut map);
    map
}

/// [`valid`]'s table with a copy of handle 8 at `id`.
fn valid_with_copy_at(id: u32) -> NvMap {
    valid_with(|map| {
        let handle = map.handles[&8];
        map.handles.insert(id, handle);
    })
}

/// Each saved table no sequence of ioctls reaches, and the error it is refused with.
fn unreachable_tables() -> Vec<(NvMap, NvMapStateError)> {
    use NvMapStateError::{
        Align, Dupes, NextId, NotAnId, NotIssued, PastTheEnd, Size, TooMany, UnallocatedDuplicated,
    };
    let past = u64::MAX - 0x1FFF;
    vec![
        (valid_with(|map| map.next_id = 0), NextId { next: 0 }),
        (valid_with(|map| map.next_id = 13), NextId { next: 13 }),
        (
            valid_with(|map| map.next_id = HANDLE_ID_END + 4),
            NextId {
                next: HANDLE_ID_END + 4,
            },
        ),
        (
            table_of(MAX_HANDLES + 1),
            TooMany {
                count: MAX_HANDLES + 1,
            },
        ),
        (valid_with_copy_at(6), NotAnId { id: 6 }),
        (valid_with_copy_at(0), NotAnId { id: 0 }),
        (valid_with_copy_at(12), NotIssued { id: 12 }),
        (
            valid_with(|map| handle_of(map, 8).size = 0),
            Size { id: 8, size: 0 },
        ),
        (
            valid_with(|map| handle_of(map, 8).size = 0xFFFF_F001),
            Size {
                id: 8,
                size: 0xFFFF_F001,
            },
        ),
        (
            valid_with(|map| handle_of(map, 4).dupes = 0),
            Dupes { id: 4, dupes: 0 },
        ),
        (
            valid_with(|map| handle_of(map, 4).dupes = MAX_DUPES + 1),
            Dupes {
                id: 4,
                dupes: MAX_DUPES + 1,
            },
        ),
        (
            valid_with(|map| handle_of(map, 8).dupes = 2),
            UnallocatedDuplicated { id: 8, dupes: 2 },
        ),
        (
            valid_with(|map| allocation_of(map).align = 0x800),
            Align {
                id: 4,
                align: 0x800,
            },
        ),
        (
            valid_with(|map| allocation_of(map).align = 0x3000),
            Align {
                id: 4,
                align: 0x3000,
            },
        ),
        (
            valid_with(|map| allocation_of(map).address = past),
            PastTheEnd {
                id: 4,
                address: past,
            },
        ),
        // Two faults: a bad next id and a bad handle; the table's own check comes first.
        (
            valid_with(|map| {
                map.next_id = 2;
                handle_of(map, 8).size = 0;
            }),
            NextId { next: 2 },
        ),
        // Two faults in two handles: the lower id's comes first.
        (
            valid_with(|map| {
                handle_of(map, 4).dupes = 0;
                handle_of(map, 8).size = 0;
            }),
            Dupes { id: 4, dupes: 0 },
        ),
        // Two faults in one handle: its size is checked before its references.
        (
            valid_with(|map| {
                handle_of(map, 8).size = 0;
                handle_of(map, 8).dupes = 0;
            }),
            Size { id: 8, size: 0 },
        ),
    ]
}

#[test]
fn a_saved_table_no_ioctl_reaches_is_refused() {
    for (map, error) in unreachable_tables() {
        assert_eq!(reload(&map), Err(error), "{error}");
        // Through `postcard`, as a snapshot reads it: refused.
        let bytes = postcard::to_allocvec(&map).unwrap();
        assert!(postcard::from_bytes::<NvMap>(&bytes).is_err(), "{error}");
    }
}

/// Each relation's edge, on the side that is reachable: accepted as it is.
#[test]
fn a_saved_table_at_each_edge_is_accepted() {
    let cases = [
        NvMap::new(),
        valid(),
        valid_with(|map| map.next_id = HANDLE_ID_END),
        table_of(MAX_HANDLES),
        valid_with(|map| handle_of(map, 8).size = 0xFFFF_F000),
        valid_with(|map| handle_of(map, 8).size = 1),
        valid_with(|map| handle_of(map, 4).dupes = MAX_DUPES),
        valid_with(|map| allocation_of(map).align = 1 << 31),
        valid_with(|map| allocation_of(map).align = PAGE),
        valid_with(|map| allocation_of(map).address = u64::MAX - 0x2000),
    ];
    for map in cases {
        assert_eq!(reload(&map).as_ref(), Ok(&map));
        let bytes = postcard::to_allocvec(&map).unwrap();
        assert_eq!(postcard::from_bytes::<NvMap>(&bytes).unwrap(), map);
    }
}

#[test]
fn a_refused_table_is_said_in_plain_words() {
    let said = [
        (
            NvMapStateError::NextId { next: 13 },
            "the next nvmap handle id 0xd is not one handle ids can reach",
        ),
        (
            NvMapStateError::TooMany { count: 7 },
            "the nvmap table holds 7 handles, more than it can",
        ),
        (
            NvMapStateError::NotAnId { id: 6 },
            "nvmap handle id 0x6 is not a handle id",
        ),
        (
            NvMapStateError::NotIssued { id: 12 },
            "nvmap handle id 0xc was never handed out",
        ),
        (
            NvMapStateError::Size { id: 8, size: 0 },
            "nvmap handle 0x8 has size 0x0, which no handle can have",
        ),
        (
            NvMapStateError::Dupes { id: 4, dupes: 0 },
            "nvmap handle 0x4 holds 0 references",
        ),
        (
            NvMapStateError::UnallocatedDuplicated { id: 8, dupes: 2 },
            "unallocated nvmap handle 0x8 holds 2 references",
        ),
        (
            NvMapStateError::Align {
                id: 4,
                align: 0x800,
            },
            "nvmap handle 0x4 is aligned to 0x800, which no allocation can be",
        ),
        (
            NvMapStateError::PastTheEnd {
                id: 4,
                address: 0xFFFF_FFFF_FFFF_E000,
            },
            "nvmap handle 0x4 at 0xffffffffffffe000 reaches the end of the address space",
        ),
    ];
    for (error, words) in said {
        assert_eq!(error.to_string(), words);
    }
}

// ----- The device-share lock -------------------------------------------------------------------

/// The process the tests' tokens name: not the first slot or generation.
fn locker() -> starquake_kernel::id::ProcessId {
    use starquake_kernel::id::SlabId;
    starquake_kernel::id::ProcessId::from_parts(2, 3)
}

/// The token a lock of `size` bytes at `address` returns.
fn token(address: u64, size: u64) -> DeviceShareToken {
    starquake_kernel::memory::testing::device_share_token(locker(), address, size)
}

/// Creates a handle of `size` bytes, allocates it at `address` and keeps the lock the request
/// would take; its id and the lock's token.
fn locked(map: &mut NvMap, size: u32, address: u64) -> (u32, DeviceShareToken) {
    let id = create(map, size);
    assert_eq!(alloc(map, id, 0, address), NvResult::Success);
    let held = token(address, map.handle(id).unwrap().size());
    assert_eq!(map.take_steps().len(), 1);
    map.keep_lock(id, held).unwrap();
    (id, held)
}

/// An allocation leaves one lock step over its address and page-rounded size; a create, an
/// allocation refused at each of its checks, a parameter and an id leave none (module docs,
/// "The device-share lock").
#[test]
fn alloc_leaves_one_lock_step_and_a_refused_alloc_none() {
    let mut map = NvMap::new();
    let id = create(&mut map, PAGE + 1);
    assert_eq!(map.take_steps(), []);
    let address = 0x0000_0051_2345_6000;
    // Refused: handle 0, a bad alignment, no such handle; then the one that succeeds.
    assert_eq!(alloc(&mut map, 0, 0, address), NvResult::BadValue);
    assert_eq!(alloc(&mut map, id, 3, address), NvResult::BadValue);
    assert_eq!(
        alloc(&mut map, id + HANDLE_ID_INCREMENT, 0, address),
        NvResult::BadValue
    );
    assert_eq!(map.take_steps(), []);
    assert_eq!(alloc(&mut map, id, 0, address), NvResult::Success);
    let size = map.handle(id).unwrap().size();
    assert_eq!(size, 2 * u64::from(PAGE));
    assert_eq!(
        map.take_steps(),
        [DeviceShareStep::Lock {
            handle: id,
            address,
            size,
        }]
    );
    // Taken out: none are left. Allocated already, and past the end: refused, no step.
    assert_eq!(map.take_steps(), []);
    assert_eq!(
        alloc(&mut map, id, 0, address),
        NvResult::InsufficientMemory
    );
    let other = create(&mut map, PAGE);
    assert_eq!(alloc(&mut map, other, 0, u64::MAX), NvResult::BadValue);
    param_of(&mut map, id, param::SIZE);
    call(&mut map, GET_ID, &le(&[0, id]));
    assert_eq!(map.take_steps(), []);
    // No lock is held until one is kept.
    assert_eq!(map.lock(id), None);
}

/// A lock is kept only by the handle it was asked for, allocated over exactly the token's
/// range, holding none yet; anything else gives the token back.
#[test]
fn keep_lock_takes_only_a_token_for_the_allocation_s_exact_range() {
    let mut map = NvMap::new();
    let id = create(&mut map, PAGE);
    let unallocated = create(&mut map, PAGE);
    let address = 0x5000_0000;
    alloc(&mut map, id, 0, address);
    let page = u64::from(PAGE);
    let exact = token(address, page);
    let refused = [
        (id, token(address + page, page)),
        (id, token(address - page, page)),
        (id, token(address, 2 * page)),
        (unallocated, token(address, page)),
        (unallocated + HANDLE_ID_INCREMENT, exact),
    ];
    for (handle, offered) in refused {
        assert_eq!(map.keep_lock(handle, offered), Err(offered), "{handle:#x}");
    }
    assert_eq!(map.lock(id), None);
    assert_eq!(map.keep_lock(id, exact), Ok(()));
    assert_eq!(map.lock(id), Some(exact));
    // A handle holds one lock: a second is given back, the first kept.
    assert_eq!(map.keep_lock(id, exact), Err(exact));
    assert_eq!(map.lock(id), Some(exact));
    assert_eq!(map.lock(unallocated), None);
}

/// `IocFree` leaves the release of the handle's lock with its last reference, and none while a
/// reference is left or when the handle holds no lock.
#[test]
fn free_leaves_the_release_with_the_last_reference_only() {
    let mut map = NvMap::new();
    let (id, held) = locked(&mut map, PAGE, 0x5000_0000);
    call(&mut map, FROM_ID, &le(&[id, 0]));
    call(&mut map, FREE, &free_args(id, 0));
    assert_eq!(map.take_steps(), []);
    assert_eq!(map.lock(id), Some(held));
    call(&mut map, FREE, &free_args(id, 0));
    assert_eq!(map.take_steps(), [DeviceShareStep::Unlock(held)]);
    assert_eq!(map.lock(id), None);
    // A handle allocated with no lock kept (the kernel refused it) releases nothing.
    let bare = create(&mut map, PAGE);
    alloc(&mut map, bare, 0, 0x6000_0000);
    map.take_steps();
    call(&mut map, FREE, &free_args(bare, 0));
    assert_eq!(map.take_steps(), []);
}

/// `CloseSession`: one reference of each handle goes, as one `IocFree` of each would; the
/// handles that went give their locks, and a duplicated one stays with its lock and the rest of
/// its references (module docs, "When the process's last session closes").
#[test]
fn close_session_frees_one_reference_of_each_handle_and_gives_the_locks_of_those_that_went() {
    let mut map = NvMap::new();
    let (lone, lone_lock) = locked(&mut map, PAGE, 0x5000_0000);
    let (twice, twice_lock) = locked(&mut map, PAGE, 0x6000_0000);
    call(&mut map, FROM_ID, &le(&[twice, 0]));
    let unallocated = create(&mut map, PAGE);
    assert_eq!(map.close_session(), [lone_lock]);
    assert_eq!(map.handle(lone), None);
    assert_eq!(map.handle(unallocated), None);
    assert_eq!(map.handle(twice).unwrap().dupes(), 1);
    assert_eq!(map.lock(twice), Some(twice_lock));
    // The ioctls left nothing for a request to take: the caller releases what it was given.
    assert_eq!(map.take_steps(), []);
    // A second close takes the last reference.
    assert_eq!(map.close_session(), [twice_lock]);
    assert!(map.is_empty());
    assert_eq!(map.close_session(), []);
}

/// The handles go in ascending id order, whatever order they were allocated and locked in.
#[test]
fn close_session_frees_in_ascending_id_order() {
    let mut map = NvMap::new();
    let ids: Vec<u32> = (0..3).map(|_| create(&mut map, PAGE)).collect();
    let mut tokens = BTreeMap::new();
    // Allocated and locked last id first.
    for (n, &id) in ids.iter().enumerate().rev() {
        let address = 0x5000_0000 + u64::try_from(n).unwrap() * 0x10_0000;
        alloc(&mut map, id, 0, address);
        map.take_steps();
        let held = token(address, u64::from(PAGE));
        map.keep_lock(id, held).unwrap();
        tokens.insert(id, held);
    }
    let in_order: Vec<DeviceShareToken> = ids.iter().map(|id| tokens[id]).collect();
    assert_eq!(map.close_session(), in_order);
}

/// A table's locks are the run's: saved and read back, it holds its handles and no lock
/// (module docs, "Where this differs from Eden").
#[test]
fn a_restored_table_holds_no_locks() {
    let mut map = NvMap::new();
    let (id, held) = locked(&mut map, PAGE, 0x5000_0000);
    assert_eq!(map.lock(id), Some(held));
    let bytes = postcard::to_allocvec(&map).unwrap();
    let restored = postcard::from_bytes::<NvMap>(&bytes).unwrap();
    assert_eq!(restored.handle(id), map.handle(id));
    assert_eq!(restored.lock(id), None);
}
