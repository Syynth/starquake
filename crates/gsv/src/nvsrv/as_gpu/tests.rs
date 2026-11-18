//! nvhost-as-gpu's views, regions and shim, driven through the nvsrv state over a real GPU
//! memory, and through the server beside the gpu session over one gsv shared state.
//!
//! Arguments are laid out by hand, word by word, from PR's structs
//! (`eden-pr4588/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.h` @ e5fb3b97:
//! `IoctlAllocAsEx` is flags, `as_fd`, big page size, reserved, then start, end and split as
//! `u64`s; `IoctlGetVaRegions` is a `u64` address, a `u32` size, a `u32`, then two 0x18-byte
//! `VaRegion`s; `IoctlSyncpointShimInfo` a `u64` address and two `u32`s), and the ioctl numbers
//! as NvRm writes them (`super::super::tests::AS_ALLOC_AS_EX` and its neighbours).

use std::sync::Arc;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_gpu::interface::{SYNCPOINT_COUNT, Target};
use starquake_kernel::result::KernelError;
use starquake_services::framework::UnknownPolicy;

use super::super::tests::{
    AS_ALLOC_AS_EX, AS_GET_VA_REGIONS, AS_SHIM_INFO, Direct, FrontCall, Rig, STALE, le, ok,
};
use super::*;
use crate::nvsrv::{Device, IoctlVariant, NvsrvState, NvsrvStateError};
use crate::testing::halves;

/// `AllocAsEx`'s 40 bytes: flags, `as_fd`, big page size, reserved, start, end, split.
fn alloc(flags: u32, as_fd: i32, big: u32, reserved: u32, range: [u64; 3]) -> Vec<u8> {
    let [start, end, split] = range.map(halves);
    let mut words = vec![flags, as_fd.cast_unsigned(), big, reserved];
    words.extend(start);
    words.extend(end);
    words.extend(split);
    le(&words)
}

/// A parentless `AllocAsEx` with flags 0: big page size, start, end, split.
fn plain(big: u32, start: u64, end: u64, split: u64) -> Vec<u8> {
    alloc(0, 0, big, 0, [start, end, split])
}

/// `GetVARegions`'s 0x40 bytes as sent: a buffer address and size the ioctl ignores and
/// overwrites, and stale regions.
fn va_request() -> Vec<u8> {
    let mut input = le(&[0x1111_2222, 0x3333_4444, 0x30, 0x5555_6666]);
    input.resize(0x40, 0x77);
    input
}

/// A region as `VaRegion` lays it out: offset (`u64`), page size, padding, pages (`u64`).
fn region(offset: u64, page_size: u32, pages: u64) -> Vec<u8> {
    let mut words = halves(offset).to_vec();
    words.extend([page_size, 0]);
    words.extend(halves(pages));
    le(&words)
}

/// `GetVARegions`'s struct back: the buffer address as sent, the size, the reserved word as
/// sent, then the two regions.
fn va_reply(size: u32, regions: &[Vec<u8>; 2]) -> Vec<u8> {
    let mut bytes = le(&[0x1111_2222, 0x3333_4444, size, 0x5555_6666]);
    bytes.extend(&regions[0]);
    bytes.extend(&regions[1]);
    bytes
}

/// The shim's struct back: address, stride, count.
fn shim_reply(address: u64, stride: u32, count: u32) -> Vec<u8> {
    let mut words = halves(address).to_vec();
    words.extend([stride, count]);
    le(&words)
}

/// A stale shim request: every field non-zero.
fn shim_request() -> Vec<u8> {
    le(&[0xAAAA_AAAA, 0xBBBB_BBBB, 0xCCCC, 0xDDDD])
}

/// A zero region, 0x18 bytes.
fn zero_region() -> Vec<u8> {
    vec![0; 0x18]
}

/// A direct state with one as-gpu fd open (fd 1).
fn opened() -> (Direct, i32) {
    let mut direct = Direct::new();
    let fd = direct.open(Device::NvhostAsGpu);
    (direct, fd)
}

/// The NVN window's ends and the native layout's end, written out (row 25; PR
/// `GetGpuAddressSpaceLayout(true)`).
const WINDOW_START: u64 = 0x800_0000;
const WINDOW_END: u64 = 0xC0_0000_0000;
const NATIVE_END: u64 = 1 << 40;

// ----- AllocAsEx ranges --------------------------------------------------------------------------

/// The spec's offline test, "`AllocAsEx` ranges": each kind of request, and the regions and shim
/// the fd then reports, written out by hand.
#[test]
fn alloc_as_ex_ranges() {
    struct Case {
        what: &'static str,
        request: Vec<u8>,
        size: u32,
        regions: [Vec<u8>; 2],
        shim: u64,
    }
    let cases = [
        Case {
            what: "a start of 0: PR's native layout, big pages 0x20000",
            request: plain(0, 0, 0, 0),
            size: 0x30,
            regions: [
                region(WINDOW_START, 0x1000, 0xBFF_8000),
                region(WINDOW_END, 0x2_0000, 0x20_0000),
            ],
            shim: WINDOW_END - 0x1000,
        },
        Case {
            what: "row 25 with as_fd 0: flags 5, no big pages, split 0",
            request: alloc(5, 0, 0, 0, [WINDOW_START, WINDOW_END, 0]),
            size: 0x18,
            regions: [region(WINDOW_START, 0x1000, 0xBFF_8000), zero_region()],
            shim: WINDOW_END - 0x1000,
        },
        Case {
            what: "explicit ranges, a 4 KiB request taken as 64 KiB big pages",
            request: plain(0x1000, 0x10_0000, 0x100_0000, 0x40_0000),
            size: 0x30,
            regions: [
                region(0x10_0000, 0x1000, 0x300),
                region(0x40_0000, 0x1_0000, 0xC0),
            ],
            shim: 0x3F_F000,
        },
        Case {
            what: "a split inside a big page: the big region starts at the page below",
            request: plain(0x1_0000, 0x10_0000, 0x100_0000, 0x40_1000),
            size: 0x30,
            regions: [
                region(0x10_0000, 0x1000, 0x301),
                region(0x40_0000, 0x1_0000, 0xC0),
            ],
            shim: 0x40_0000,
        },
        Case {
            what: "start equal to split: no small pages",
            request: plain(0x2_0000, 0x40_0000, 0x80_0000, 0x40_0000),
            size: 0x30,
            regions: [
                region(0x40_0000, 0x1000, 0),
                region(0x40_0000, 0x2_0000, 0x20),
            ],
            shim: 0x3F_F000,
        },
    ];
    for case in cases {
        let (mut direct, fd) = opened();
        let answer = direct.ioctl(fd, AS_ALLOC_AS_EX, &case.request);
        assert_eq!(answer.result, NvResult::Success, "{}", case.what);
        assert_eq!(answer.output, case.request, "{}", case.what);
        let answer = direct.ioctl(fd, AS_GET_VA_REGIONS, &va_request());
        assert_eq!(answer.result, NvResult::Success, "{}", case.what);
        assert_eq!(
            answer.output,
            va_reply(case.size, &case.regions),
            "{}",
            case.what
        );
        let answer = direct.ioctl(fd, AS_SHIM_INFO, &shim_request());
        assert_eq!(answer.result, NvResult::Success, "{}", case.what);
        let expected = shim_reply(case.shim, 4, SYNCPOINT_COUNT);
        assert_eq!(answer.output, expected, "{}", case.what);
        // A new address space, the first, given to the GPU's front.
        assert_eq!(
            direct.front.calls(),
            [FrontCall::Create(AsId(1))],
            "{}",
            case.what
        );
    }
}

/// A parentless request with split 0 takes small pages to its end and has no big pages
/// (module docs, "Conflicts"); with a split it keeps it.
#[test]
fn split_0_without_a_parent_takes_small_pages_to_the_end() {
    let (mut direct, fd) = opened();
    let request = plain(0x2_0000, 0x10_0000, 0x200_0000, 0);
    assert_eq!(
        direct.ioctl(fd, AS_ALLOC_AS_EX, &request).result,
        NvResult::Success
    );
    let view = direct.state.as_gpu().view(1).unwrap();
    let expected = Layout {
        start: 0x10_0000,
        split: 0x200_0000,
        end: 0x200_0000,
        big_page_size: 0x2_0000,
    };
    assert_eq!(view.layout, expected);
    // The same request with a split keeps it.
    let fd = direct.open(Device::NvhostAsGpu);
    let request = plain(0x2_0000, 0x10_0000, 0x200_0000, 0x100_0000);
    assert_eq!(
        direct.ioctl(fd, AS_ALLOC_AS_EX, &request).result,
        NvResult::Success
    );
    let view = direct.state.as_gpu().view(2).unwrap();
    assert_eq!(view.layout.split, 0x100_0000);
}

/// The big page size word: 0 keeps the default, 0x1000 becomes 0x10000, 0x10000 and 0x20000
/// stay, and a size that is not a power of two or not among the supported is `BadValue`.
#[test]
fn big_page_sizes_are_pr_s() {
    let cases = [
        (0, Some(0x2_0000)),
        (0x1000, Some(0x1_0000)),
        (0x1_0000, Some(0x1_0000)),
        (0x2_0000, Some(0x2_0000)),
        (0x3_0000, None),
        (0x4_0000, None),
        (0x8000, None),
        (0x2000, None),
        (u32::MAX, None),
    ];
    for (requested, kept) in cases {
        let (mut direct, fd) = opened();
        let request = plain(requested, 0x10_0000, 0x100_0000, 0x40_0000);
        let answer = direct.ioctl(fd, AS_ALLOC_AS_EX, &request);
        let view = direct.state.as_gpu().view(1);
        if let Some(size) = kept {
            assert_eq!(answer.result, NvResult::Success, "{requested:#x}");
            assert_eq!(view.unwrap().layout.big_page_size, size, "{requested:#x}");
        } else {
            assert_eq!(answer.result, NvResult::BadValue, "{requested:#x}");
            assert_eq!(view, None, "{requested:#x}");
            assert_eq!(direct.front.calls(), [], "{requested:#x}");
        }
    }
}

/// Ranges out of order or not page aligned are `BadValue`, nothing made; equal ends are the
/// accepted edge.
#[test]
fn a_layout_out_of_order_is_a_bad_value() {
    let cases = [
        // (start, end, split, accepted)
        (0x10_0000, 0x100_0000, 0x100_0000, true),
        (0x10_0000, 0x100_0000, 0x10_0000, true),
        (0x10_0000, 0x100_0000, 0x10_0000 - 0x1000, false),
        (0x10_0000, 0x100_0000, 0x100_0000 + 0x1000, false),
        (0x10_0800, 0x100_0000, 0x40_0000, false),
        (0x10_0000, 0x100_0800, 0x40_0000, false),
        (0x10_0000, 0x100_0000, 0x40_0800, false),
    ];
    for (start, end, split, accepted) in cases {
        let (mut direct, fd) = opened();
        let answer = direct.ioctl(fd, AS_ALLOC_AS_EX, &plain(0, start, end, split));
        let expected = if accepted {
            NvResult::Success
        } else {
            NvResult::BadValue
        };
        assert_eq!(answer.result, expected, "{start:#x} {end:#x} {split:#x}");
        assert_eq!(direct.state.as_gpu().view(1).is_some(), accepted);
        assert_eq!(
            direct.memory.address_spaces().count(),
            usize::from(accepted)
        );
    }
}

/// An end at [`VA_LIMIT`] is accepted, a page more is `BadValue`.
#[test]
fn a_layout_past_the_va_limit_is_a_bad_value() {
    assert_eq!(VA_LIMIT, NATIVE_END);
    for (end, expected) in [
        (VA_LIMIT, NvResult::Success),
        (VA_LIMIT + 0x1000, NvResult::BadValue),
        (u64::MAX - 0xFFF, NvResult::BadValue),
    ] {
        let (mut direct, fd) = opened();
        let answer = direct.ioctl(fd, AS_ALLOC_AS_EX, &plain(0, 0x10_0000, end, 0x40_0000));
        assert_eq!(answer.result, expected, "{end:#x}");
    }
}

/// A second `AllocAsEx` on the same fd is `InvalidState`, before anything else is looked at: a
/// bad parent and a bad big page size at once answer it, and the view is unchanged.
#[test]
fn a_second_alloc_as_ex_is_an_invalid_state() {
    let (mut direct, fd) = opened();
    let first = plain(0, 0x10_0000, 0x100_0000, 0x40_0000);
    assert_eq!(
        direct.ioctl(fd, AS_ALLOC_AS_EX, &first).result,
        NvResult::Success
    );
    let view = direct.state.as_gpu().view(1);
    let both = alloc(1, -1, 0x3_0000, 0, [0x10_0000, 0x100_0000, 0x40_0000]);
    for request in [first, both] {
        let answer = direct.ioctl(fd, AS_ALLOC_AS_EX, &request);
        assert_eq!(answer.result, NvResult::InvalidState);
    }
    assert_eq!(direct.state.as_gpu().view(1), view);
    assert_eq!(direct.memory.address_spaces().count(), 1);
}

// ----- Parents ---------------------------------------------------------------------------------

/// PR's own cases for `IsSupportedNativeAddressSpaceView` ("Native address space views
/// validate both small-page request formats"), against the native layout.
#[test]
fn supported_views_are_pr_s() {
    let parent = NATIVE_LAYOUT;
    let end = 0xA0_0000_0000;
    let request = |start, split, end| Requested { start, split, end };
    let start = parent.start;
    assert!(is_supported_view(
        5,
        0x1000,
        0,
        request(start, 0, end),
        &parent
    ));
    assert!(is_supported_view(
        0,
        0x1000,
        0,
        request(start, end, end),
        &parent
    ));
    let refused = [
        (1, 0x1000, 0, request(start, end, end)),
        (0, 0x1_0000, 0, request(start, end, end)),
        (0, 0x1000, 1, request(start, end, end)),
        (0, 0x1000, 0, request(start, 0, end)),
        (5, 0x1000, 0, request(start, end, end)),
        (0, 0x1000, 0, request(0, end, end)),
        (0, 0x1000, 0, request(end, end, end)),
        (0, 0x1000, 0, request(start, parent.end, parent.end)),
        (0, 0x1000, 0, request(start + 1, end, end)),
        (0, 0x1000, 0, request(start, end + 1, end + 1)),
    ];
    for (flags, page_size, reserved, requested) in refused {
        assert!(
            !is_supported_view(flags, page_size, reserved, requested, &parent),
            "{flags} {page_size:#x} {reserved} {requested:?}"
        );
    }
    // The edges this port adds: the start exactly at the parent's, the end exactly at its split
    // (accepted), a page past the split (refused).
    assert!(is_supported_view(
        0,
        0x1000,
        0,
        request(start, parent.split, parent.split),
        &parent
    ));
    let past = parent.split + 0x1000;
    assert!(!is_supported_view(
        0,
        0x1000,
        0,
        request(start, past, past),
        &parent
    ));
    assert!(!is_supported_view(
        0,
        0x1000,
        0,
        request(start - 0x1000, end, end),
        &parent
    ));
}

/// A request naming a gpu-33 address space (one gsv holds that no `AllocAsEx` made) is a view of
/// it: no new address space, the native layout as its parent, its split at its end, 64 KiB big
/// pages and none in its range; its shim lands in the parent's table.
#[test]
fn a_view_shares_its_parent_s_address_space() {
    let (mut direct, fd) = opened();
    let parent = direct.memory.create_address_space(&direct.front).unwrap();
    let as_fd = i32::try_from(parent.0).unwrap();
    let end = 0xA0_0000_0000;
    let request = alloc(5, as_fd, 0x1000, 0, [WINDOW_START, end, 0]);
    assert_eq!(
        direct.ioctl(fd, AS_ALLOC_AS_EX, &request).result,
        NvResult::Success
    );
    let view = direct.state.as_gpu().view(1).unwrap();
    let expected = Layout {
        start: WINDOW_START,
        split: end,
        end,
        big_page_size: 0x1_0000,
    };
    assert_eq!(
        view,
        View {
            space: parent,
            layout: expected
        }
    );
    assert_eq!(direct.state.as_gpu().made(parent), None);
    assert_eq!(direct.front.calls(), [FrontCall::Create(parent)]);
    let answer = direct.ioctl(fd, AS_GET_VA_REGIONS, &va_request());
    let regions = [region(WINDOW_START, 0x1000, 0x9FF_8000), zero_region()];
    assert_eq!(answer.output, va_reply(0x18, &regions));
    let answer = direct.ioctl(fd, AS_SHIM_INFO, &shim_request());
    assert_eq!(answer.output, shim_reply(end - 0x1000, 4, SYNCPOINT_COUNT));
    let shown = direct.memory.mappings(parent);
    assert!(
        shown
            .iter()
            .any(|m| m.target == Target::SyncpointShim && m.range.start() == end - 0x1000),
        "{shown:?}"
    );
}

/// A view of a gpu-33 space is checked against the native layout: an end at its split is the
/// accepted edge, a page past it `BadParameter`; flags 0 with the split at the end is the other
/// format PR accepts.
#[test]
fn a_view_of_a_gpu_session_space_uses_the_native_layout() {
    for (flags, end, split, expected) in [
        (5, WINDOW_END, 0, NvResult::Success),
        (5, WINDOW_END + 0x1000, 0, NvResult::BadParameter),
        (0, WINDOW_END, WINDOW_END, NvResult::Success),
        (0, WINDOW_END, 0, NvResult::BadParameter),
    ] {
        let (mut direct, fd) = opened();
        let parent = direct.memory.create_address_space(&direct.front).unwrap();
        let as_fd = i32::try_from(parent.0).unwrap();
        let request = alloc(flags, as_fd, 0x1000, 0, [WINDOW_START, end, split]);
        let answer = direct.ioctl(fd, AS_ALLOC_AS_EX, &request);
        assert_eq!(answer.result, expected, "{flags} {end:#x} {split:#x}");
    }
}

/// A view of a space an earlier parentless `AllocAsEx` made is checked against that space's
/// layout, not the native one.
#[test]
fn a_view_of_a_made_space_uses_its_layout() {
    let (mut direct, first) = opened();
    let made = plain(0, 0x10_0000, 0x100_0000, 0x80_0000);
    assert_eq!(
        direct.ioctl(first, AS_ALLOC_AS_EX, &made).result,
        NvResult::Success
    );
    let space = direct.state.as_gpu().view(1).unwrap().space;
    let as_fd = i32::try_from(space.0).unwrap();
    let second = direct.open(Device::NvhostAsGpu);
    // Inside its small pages, to its split exactly: accepted; a page past: refused, though the
    // native layout would take it.
    let past = alloc(5, as_fd, 0x1000, 0, [0x10_0000, 0x80_1000, 0]);
    let refused = direct.ioctl(second, AS_ALLOC_AS_EX, &past);
    assert_eq!(refused.result, NvResult::BadParameter);
    let edge = alloc(5, as_fd, 0x1000, 0, [0x10_0000, 0x80_0000, 0]);
    assert_eq!(
        direct.ioctl(second, AS_ALLOC_AS_EX, &edge).result,
        NvResult::Success
    );
    assert_eq!(direct.state.as_gpu().view(2).unwrap().space, space);
    assert_eq!(direct.memory.address_spaces().count(), 1);
}

/// `as_fd` that names nothing gsv holds, or is negative, is `BadParameter` for a request that
/// is otherwise a view PR supports, and nothing is made.
#[test]
fn a_parent_gsv_does_not_hold_is_a_bad_parameter() {
    for as_fd in [-1, i32::MIN, 1, 2, i32::MAX] {
        let (mut direct, fd) = opened();
        let request = alloc(5, as_fd, 0x1000, 0, [WINDOW_START, 0xA0_0000_0000, 0]);
        let answer = direct.ioctl(fd, AS_ALLOC_AS_EX, &request);
        assert_eq!(answer.result, NvResult::BadParameter, "{as_fd}");
        assert_eq!(direct.state.as_gpu().view(1), None);
        assert_eq!(direct.front.calls(), []);
    }
    // The id exists once gsv makes it.
    let (mut direct, fd) = opened();
    direct.memory.create_address_space(&direct.front).unwrap();
    let request = alloc(5, 1, 0x1000, 0, [WINDOW_START, 0xA0_0000_0000, 0]);
    assert_eq!(
        direct.ioctl(fd, AS_ALLOC_AS_EX, &request).result,
        NvResult::Success
    );
}

/// A bad layout is reported before a full table of address spaces (both at once). (A bad big
/// page size and a bad layout both answer `BadValue`, so their order cannot be seen.)
#[test]
fn a_bad_layout_comes_before_a_full_table() {
    let (mut direct, fd) = opened();
    for _ in 0..MAX_ADDRESS_SPACES {
        direct.memory.create_address_space(&direct.front).unwrap();
    }
    let bad_and_full = plain(0, 0x10_0000, 0x100_0000, 0x200_0000);
    let answer = direct.ioctl(fd, AS_ALLOC_AS_EX, &bad_and_full);
    assert_eq!(answer.result, NvResult::BadValue);
    let full = plain(0, 0x10_0000, 0x100_0000, 0x40_0000);
    let answer = direct.ioctl(fd, AS_ALLOC_AS_EX, &full);
    assert_eq!(answer.result, NvResult::InsufficientMemory);
}

/// Parentless views count against the address spaces gpu 33 makes: with one fewer than
/// [`MAX_ADDRESS_SPACES`] made, the next is made; at the limit, `InsufficientMemory`, no view.
/// A view of an existing space still works at the limit.
#[test]
fn address_spaces_are_bounded_with_gpu_33_s() {
    let (mut direct, fd) = opened();
    for _ in 0..MAX_ADDRESS_SPACES - 1 {
        direct.memory.create_address_space(&direct.front).unwrap();
    }
    let request = plain(0, 0x10_0000, 0x100_0000, 0x40_0000);
    assert_eq!(
        direct.ioctl(fd, AS_ALLOC_AS_EX, &request).result,
        NvResult::Success
    );
    let made = AsId(u32::try_from(MAX_ADDRESS_SPACES).unwrap());
    assert_eq!(direct.state.as_gpu().view(1).unwrap().space, made);
    let fd = direct.open(Device::NvhostAsGpu);
    let answer = direct.ioctl(fd, AS_ALLOC_AS_EX, &request);
    assert_eq!(answer.result, NvResult::InsufficientMemory);
    assert_eq!(direct.state.as_gpu().view(2), None);
    assert_eq!(direct.memory.address_spaces().count(), MAX_ADDRESS_SPACES);
    let view = alloc(5, 1, 0x1000, 0, [WINDOW_START, 0xA0_0000_0000, 0]);
    assert_eq!(
        direct.ioctl(fd, AS_ALLOC_AS_EX, &view).result,
        NvResult::Success
    );
}

// ----- Closing -----------------------------------------------------------------------------------

/// The layout [`own_space`] asks for: small pages from 1 MiB to 4 MiB, big pages to 16 MiB.
const OWN: [u64; 3] = [0x10_0000, 0x100_0000, 0x40_0000];

/// A parentless `AllocAsEx` on a new fd, which must succeed: the fd and its new space.
fn own_space(direct: &mut Direct) -> (i32, AsId) {
    let fd = direct.open(Device::NvhostAsGpu);
    let [start, end, split] = OWN;
    let request = plain(0, start, end, split);
    assert_eq!(
        direct.ioctl(fd, AS_ALLOC_AS_EX, &request).result,
        NvResult::Success
    );
    (
        fd,
        direct
            .state
            .as_gpu()
            .view(fd.cast_unsigned())
            .unwrap()
            .space,
    )
}

/// A flags-5 view of `space` on a new fd, inside [`OWN`]'s small pages, or, for a space gpu 33
/// made, inside the native layout's: the fd, and the result.
fn view_of(direct: &mut Direct, space: AsId, native: bool) -> (i32, NvResult) {
    let fd = direct.open(Device::NvhostAsGpu);
    let range = if native {
        [WINDOW_START, 0xA0_0000_0000, 0]
    } else {
        [OWN[0], OWN[2], 0]
    };
    let as_fd = i32::try_from(space.0).unwrap();
    let request = alloc(5, as_fd, 0x1000, 0, range);
    (fd, direct.ioctl(fd, AS_ALLOC_AS_EX, &request).result)
}

/// The address spaces gsv holds.
fn spaces(direct: &Direct) -> Vec<AsId> {
    direct.memory.address_spaces().collect()
}

/// Every update the front was sent.
fn updates(direct: &Direct) -> Vec<FrontCall> {
    let calls = direct.front.calls();
    calls
        .into_iter()
        .filter(|call| matches!(call, FrontCall::Update(..)))
        .collect()
}

/// A space held by a creator and a child view: closing either first keeps it (the view that is
/// left still answers its shim, and nothing is sent to the GPU); closing the other destroys it,
/// its layout is forgotten and its shim's page unmapped. A new fd starts with no view.
#[test]
fn the_last_close_destroys_the_space_and_the_one_before_keeps_it() {
    for creator_first in [true, false] {
        let mut direct = Direct::new();
        let (creator, space) = own_space(&mut direct);
        let (child, result) = view_of(&mut direct, space, false);
        assert_eq!(result, NvResult::Success);
        let shim = OWN[2] - SMALL_PAGE_SIZE;
        let answer = direct.ioctl(child, AS_SHIM_INFO, &shim_request());
        assert_eq!(answer.output, shim_reply(shim, 4, SYNCPOINT_COUNT));
        let (first, last) = if creator_first {
            (creator, child)
        } else {
            (child, creator)
        };

        assert_eq!(direct.close(first), NvResult::Success);
        assert_eq!(spaces(&direct), [space], "creator first: {creator_first}");
        assert!(direct.state.as_gpu().made(space).is_some());
        assert_eq!(updates(&direct), []);
        let answer = direct.ioctl(last, AS_SHIM_INFO, &shim_request());
        assert_eq!(
            (answer.result, answer.output),
            (NvResult::Success, shim_reply(shim, 4, SYNCPOINT_COUNT))
        );

        assert_eq!(direct.close(last), NvResult::Success);
        assert_eq!(spaces(&direct), []);
        assert_eq!(direct.state.as_gpu().made(space), None);
        assert_eq!(direct.state.as_gpu().fds().count(), 0);
        let range = starquake_memory::PageRange::new(shim, SMALL_PAGE_SIZE).unwrap();
        assert_eq!(
            updates(&direct),
            [FrontCall::Update(
                space,
                starquake_gpu::interface::VmUpdate::Unmap { range },
                None
            )]
        );
        let fd = direct.open(Device::NvhostAsGpu);
        let answer = direct.ioctl(fd, AS_GET_VA_REGIONS, &va_request());
        assert_eq!(answer.result, NvResult::BadValue);
    }
}

/// Opening, allocating and closing more times than there are address spaces never runs out,
/// with no other space held and with every other slot held by gpu 33; each space takes a new
/// id.
#[test]
fn opening_allocating_and_closing_never_runs_out() {
    for held in [0, MAX_ADDRESS_SPACES - 1] {
        let mut direct = Direct::new();
        for _ in 0..held {
            direct.memory.create_address_space(&direct.front).unwrap();
        }
        let rounds = 2 * MAX_ADDRESS_SPACES + 1;
        for round in 0..rounds {
            let (fd, space) = own_space(&mut direct);
            let id = u32::try_from(held + round).unwrap() + FIRST_ADDRESS_SPACE.0;
            assert_eq!(space, AsId(id), "held {held}, round {round}");
            assert_eq!(spaces(&direct).len(), held + 1);
            assert_eq!(direct.close(fd), NvResult::Success);
            assert_eq!(spaces(&direct).len(), held);
        }
    }
}

/// gpu 33 and nvhost-as-gpu share [`MAX_ADDRESS_SPACES`], whichever path made the spaces below
/// it: one below the limit, either path makes the last; at the limit both refuse; the last
/// close of an as-gpu space frees a slot that either path then takes.
#[test]
fn gpu_33_and_as_gpu_share_the_slots_at_the_limit() {
    for as_gpu_below in [false, true] {
        let mut direct = Direct::new();
        let mut fds = Vec::new();
        for _ in 0..MAX_ADDRESS_SPACES - 1 {
            if as_gpu_below {
                fds.push(own_space(&mut direct).0);
            } else {
                direct.memory.create_address_space(&direct.front).unwrap();
            }
        }
        // One below the limit: the last slot, by the other path than the ones below.
        let last = if as_gpu_below {
            direct
                .memory
                .create_address_space(&direct.front)
                .map(|_| ())
        } else {
            fds.push(own_space(&mut direct).0);
            Ok(())
        };
        assert_eq!(last, Ok(()));
        assert_eq!(spaces(&direct).len(), MAX_ADDRESS_SPACES);
        // At the limit: both refuse.
        assert_eq!(
            direct.memory.create_address_space(&direct.front),
            Err(KernelError::OutOfResource)
        );
        let refused = direct.open(Device::NvhostAsGpu);
        let [start, end, split] = OWN;
        let request = plain(0, start, end, split);
        let answer = direct.ioctl(refused, AS_ALLOC_AS_EX, &request);
        assert_eq!(answer.result, NvResult::InsufficientMemory);
        // An as-gpu space's last close frees one slot: gpu 33 takes it, and then both refuse.
        let fd = fds.pop().unwrap();
        assert_eq!(direct.close(fd), NvResult::Success);
        assert_eq!(spaces(&direct).len(), MAX_ADDRESS_SPACES - 1);
        direct.memory.create_address_space(&direct.front).unwrap();
        let answer = direct.ioctl(refused, AS_ALLOC_AS_EX, &request);
        assert_eq!(answer.result, NvResult::InsufficientMemory);
        assert_eq!(
            direct.memory.create_address_space(&direct.front),
            Err(KernelError::OutOfResource)
        );
        // When another frees one, as-gpu takes it.
        if let Some(fd) = fds.pop() {
            assert_eq!(direct.close(fd), NvResult::Success);
            let answer = direct.ioctl(refused, AS_ALLOC_AS_EX, &request);
            assert_eq!(answer.result, NvResult::Success);
            assert_eq!(spaces(&direct).len(), MAX_ADDRESS_SPACES);
        }
    }
}

/// A view of a space gpu 33 made does not hold it alone: gpu 33 does, so closing the view keeps
/// it, and sends nothing.
#[test]
fn closing_a_view_of_a_gpu_33_space_keeps_it() {
    let mut direct = Direct::new();
    let space = direct.memory.create_address_space(&direct.front).unwrap();
    let (fd, result) = view_of(&mut direct, space, true);
    assert_eq!(result, NvResult::Success);
    assert_eq!(direct.close(fd), NvResult::Success);
    assert_eq!(spaces(&direct), [space]);
    assert_eq!(updates(&direct), []);
}

/// An `AllocAsEx` naming as its parent the id of a space its last close destroyed is
/// `BadParameter`, with no view and no space made, and nothing sent to the GPU.
#[test]
fn a_view_of_a_destroyed_space_is_a_bad_parameter() {
    let mut direct = Direct::new();
    let (creator, space) = own_space(&mut direct);
    assert_eq!(direct.close(creator), NvResult::Success);
    assert_eq!(spaces(&direct), []);
    let sent = updates(&direct);
    let (fd, result) = view_of(&mut direct, space, false);
    assert_eq!(result, NvResult::BadParameter);
    assert_eq!(direct.state.as_gpu().view(fd.cast_unsigned()), None);
    assert_eq!(spaces(&direct), []);
    assert_eq!(updates(&direct), sent);
}

/// After its creator closes, a space a child view still holds can be named as a parent (here
/// `as_fd` names the space, not the closed fd; module docs, "Where this differs"). The state
/// the close leaves, the space held by the child alone, survives `postcard` and is accepted
/// against gsv's memory.
#[test]
fn a_child_keeps_the_space_nameable_after_its_creator_closes() {
    let mut direct = Direct::new();
    let (creator, space) = own_space(&mut direct);
    let (_child, result) = view_of(&mut direct, space, false);
    assert_eq!(result, NvResult::Success);
    assert_eq!(direct.close(creator), NvResult::Success);
    let state = direct.state.as_gpu().clone();
    let bytes = postcard::to_allocvec(&state).unwrap();
    let restored = postcard::from_bytes::<AsGpu>(&bytes).unwrap();
    assert_eq!(restored, state);
    assert_eq!(restored.check_spaces(&direct.memory), Ok(()));
    let (second, result) = view_of(&mut direct, space, false);
    assert_eq!(result, NvResult::Success);
    assert_eq!(
        direct
            .state
            .as_gpu()
            .view(second.cast_unsigned())
            .unwrap()
            .space,
        space
    );
}

/// An nvsrv state saved before a destroy names a space gsv no longer holds: checked against the
/// memory after the destroy it is refused, and the state after the destroy is accepted. The
/// made spaces are checked before the views, so a state with both faults reports the made
/// space.
#[test]
fn a_state_saved_before_a_destroy_is_refused_against_the_memory_after_it() {
    let mut direct = Direct::new();
    let native = direct.memory.create_address_space(&direct.front).unwrap();
    let (on_native, result) = view_of(&mut direct, native, true);
    assert_eq!(result, NvResult::Success);
    let (creator, space) = own_space(&mut direct);
    let before = direct.state.as_gpu().clone();
    assert_eq!(before.check_spaces(&direct.memory), Ok(()));

    assert_eq!(direct.close(creator), NvResult::Success);
    assert_eq!(direct.state.as_gpu().check_spaces(&direct.memory), Ok(()));
    assert_eq!(
        before.check_spaces(&direct.memory),
        Err(AsGpuStateError::MadeSpaceNotHeld { space: space.0 })
    );
    // A view of a space gsv does not hold, alone and beside the made space's fault.
    let fd = on_native.cast_unsigned();
    let empty = GpuMemory::new();
    let views_only =
        AsGpu::try_from(saved(&[(fd, direct.state.as_gpu().view(fd).unwrap())], &[])).unwrap();
    assert_eq!(
        views_only.check_spaces(&empty),
        Err(AsGpuStateError::ViewSpaceNotHeld {
            fd,
            space: native.0
        })
    );
    assert_eq!(
        before.check_spaces(&empty),
        Err(AsGpuStateError::MadeSpaceNotHeld { space: space.0 })
    );
}

/// `check_spaces` walks the made spaces in id order: of two that gsv does not hold, the lower
/// is reported.
#[test]
fn check_spaces_reports_the_lowest_unheld_made_space() {
    let creator = |id| View {
        space: AsId(id),
        layout: MADE,
    };
    let state = AsGpu::try_from(saved(
        &[(1, creator(4)), (2, creator(3))],
        &[(AsId(3), MADE), (AsId(4), MADE)],
    ))
    .unwrap();
    assert_eq!(
        state.check_spaces(&GpuMemory::new()),
        Err(AsGpuStateError::MadeSpaceNotHeld { space: 3 })
    );
}

/// `check_spaces` walks the views in fd order: of two views of spaces gsv does not hold, the
/// lower fd's is reported, though its space has the higher id.
#[test]
fn check_spaces_reports_the_lowest_fd_whose_space_is_unheld() {
    let native = |id| View {
        space: AsId(id),
        layout: child_of(&NATIVE_LAYOUT),
    };
    let state = AsGpu::try_from(saved(&[(1, native(6)), (2, native(5))], &[])).unwrap();
    assert_eq!(
        state.check_spaces(&GpuMemory::new()),
        Err(AsGpuStateError::ViewSpaceNotHeld { fd: 1, space: 6 })
    );
}

// ----- GetVARegions and the shim -----------------------------------------------------------------

/// Before `AllocAsEx`: `GetVARegions` is `BadValue` and its struct comes back as sent; the shim
/// is `NotSupported` with every field 0.
#[test]
fn without_a_view_regions_and_shim_are_refused() {
    let (mut direct, fd) = opened();
    let answer = direct.ioctl(fd, AS_GET_VA_REGIONS, &va_request());
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::BadValue, va_request())
    );
    let answer = direct.ioctl(fd, AS_SHIM_INFO, &shim_request());
    assert_eq!(answer.result, NvResult::NotSupported);
    assert_eq!(answer.output, shim_reply(0, 0, 0));
}

/// The shim is on the page below the split, placed once: a second call answers the same and
/// makes no new mapping.
#[test]
fn the_shim_is_on_the_page_below_the_split() {
    let (mut direct, fd) = opened();
    let request = plain(0, 0x10_0000, 0x100_0000, 0x40_0000);
    assert_eq!(
        direct.ioctl(fd, AS_ALLOC_AS_EX, &request).result,
        NvResult::Success
    );
    let first = direct.ioctl(fd, AS_SHIM_INFO, &shim_request());
    assert_eq!(first.output, shim_reply(0x3F_F000, 4, SYNCPOINT_COUNT));
    assert_eq!(direct.memory.shim(AsId(1)), Some(0x3F_F000));
    let shown = direct.memory.mappings(AsId(1));
    let second = direct.ioctl(fd, AS_SHIM_INFO, &shim_request());
    assert_eq!(second, first);
    assert_eq!(direct.memory.mappings(AsId(1)), shown);
    let shims = shown
        .iter()
        .filter(|m| m.target == Target::SyncpointShim)
        .count();
    assert_eq!(shims, 1);
}

/// A split at the start leaves the small-page region empty, and the shim goes on the page below
/// the start, outside every region the fd reports, as PR's `va_range_split - YUZU_PAGESIZE`
/// places it (`AllocateFixed`, no range check); for a start of one page, at GPU address 0.
#[test]
fn a_split_at_the_start_puts_the_shim_below_the_view() {
    for start in [0x10_0000, SMALL_PAGE_SIZE] {
        let (mut direct, fd) = opened();
        let end = 0x100_0000;
        let request = plain(0, start, end, start);
        assert_eq!(
            direct.ioctl(fd, AS_ALLOC_AS_EX, &request).result,
            NvResult::Success
        );
        let regions = direct.ioctl(fd, AS_GET_VA_REGIONS, &va_request());
        assert_eq!(regions.result, NvResult::Success);
        // The big pages start at the split rounded down to a big page.
        let big_page = u64::from(DEFAULT_BIG_PAGE_SIZE);
        let small_word = u32::try_from(SMALL_PAGE_SIZE).unwrap();
        let small = region(start, small_word, 0);
        let big = region(
            start / big_page * big_page,
            DEFAULT_BIG_PAGE_SIZE,
            end / big_page - start / big_page,
        );
        let size = u32::try_from(2 * REGION_BYTES).unwrap();
        assert_eq!(regions.output, va_reply(size, &[small, big]));
        let below = start - SMALL_PAGE_SIZE;
        let answer = direct.ioctl(fd, AS_SHIM_INFO, &shim_request());
        assert_eq!(answer.result, NvResult::Success, "{start:#x}");
        assert_eq!(
            answer.output,
            shim_reply(below, SHIM_STRIDE_BYTES, SYNCPOINT_COUNT)
        );
        assert_eq!(direct.memory.shim(AsId(1)), Some(below));
        assert!(below < start, "{start:#x}");
    }
}

/// Two views of one address space: the first to ask places the shim, the second reports the
/// same page, not its own split's.
#[test]
fn a_second_view_reports_the_same_shim() {
    let mut direct = Direct::new();
    let parent = direct.memory.create_address_space(&direct.front).unwrap();
    let as_fd = i32::try_from(parent.0).unwrap();
    let a = direct.open(Device::NvhostAsGpu);
    let b = direct.open(Device::NvhostAsGpu);
    let first = alloc(5, as_fd, 0x1000, 0, [WINDOW_START, 0xA0_0000_0000, 0]);
    let second = alloc(5, as_fd, 0x1000, 0, [WINDOW_START, 0x90_0000_0000, 0]);
    assert_eq!(
        direct.ioctl(a, AS_ALLOC_AS_EX, &first).result,
        NvResult::Success
    );
    assert_eq!(
        direct.ioctl(b, AS_ALLOC_AS_EX, &second).result,
        NvResult::Success
    );
    let placed = shim_reply(0x9F_FFFF_F000, 4, SYNCPOINT_COUNT);
    assert_eq!(
        direct.ioctl(a, AS_SHIM_INFO, &shim_request()).output,
        placed
    );
    assert_eq!(
        direct.ioctl(b, AS_SHIM_INFO, &shim_request()).output,
        placed
    );
    // The other way round: b first.
    let mut direct = Direct::new();
    direct.memory.create_address_space(&direct.front).unwrap();
    let a = direct.open(Device::NvhostAsGpu);
    let b = direct.open(Device::NvhostAsGpu);
    direct.ioctl(a, AS_ALLOC_AS_EX, &first);
    direct.ioctl(b, AS_ALLOC_AS_EX, &second);
    let placed = shim_reply(0x8F_FFFF_F000, 4, SYNCPOINT_COUNT);
    assert_eq!(
        direct.ioctl(b, AS_SHIM_INFO, &shim_request()).output,
        placed
    );
    assert_eq!(
        direct.ioctl(a, AS_SHIM_INFO, &shim_request()).output,
        placed
    );
}

/// `GetVARegions` through `Ioctl3`: out buffer 1 gets as many regions as the buffer size says
/// and it holds, then zero regions up to what it holds of two; never more than two.
#[test]
fn va_regions_through_ioctl3() {
    let two = plain(0x1_0000, 0x10_0000, 0x100_0000, 0x40_0000);
    let one = plain(0x1_0000, 0x10_0000, 0x100_0000, 0);
    let small = region(0x10_0000, 0x1000, 0x300);
    let big = region(0x40_0000, 0x1_0000, 0xC0);
    let small_to_end = region(0x10_0000, 0x1000, 0xF00);
    let cases: [(&[u8], usize, Vec<u8>); 8] = [
        (&two, 0, vec![]),
        (&two, 0x17, vec![]),
        (&two, 0x18, small.clone()),
        (&two, 0x30, [small.clone(), big.clone()].concat()),
        (&two, 0x50, [small, big].concat()),
        (&one, 0x18, small_to_end.clone()),
        (&one, 0x30, [small_to_end.clone(), zero_region()].concat()),
        (&one, 0x48, [small_to_end, zero_region()].concat()),
    ];
    for (request, inline_len, expected) in cases {
        let (mut direct, fd) = opened();
        direct.ioctl(fd, AS_ALLOC_AS_EX, request);
        let sizes = (0x40, inline_len);
        let answer = direct.call(
            UnknownPolicy::Strict,
            IoctlVariant::Ioctl3,
            fd,
            AS_GET_VA_REGIONS,
            &va_request(),
            sizes,
        );
        assert_eq!(answer.result, NvResult::Success, "{inline_len:#x}");
        assert_eq!(answer.inline, expected, "{inline_len:#x}");
    }
    // Without a view: zero regions, as many as it holds of two.
    let (mut direct, fd) = opened();
    let answer = direct.call(
        UnknownPolicy::Strict,
        IoctlVariant::Ioctl3,
        fd,
        AS_GET_VA_REGIONS,
        &va_request(),
        (0x40, 0x48),
    );
    assert_eq!(answer.result, NvResult::BadValue);
    assert_eq!(answer.inline, vec![0; 0x30]);
}

/// Every other ioctl, on every variant, goes to the policy: `AllocAsEx` and the shim on
/// `Ioctl3`, `GetVARegions` on `Ioctl2`, and the commands not built (module docs, "Not built").
#[test]
fn other_ioctls_go_to_the_policy() {
    let (mut direct, fd) = opened();
    let unknown = [
        (IoctlVariant::Ioctl3, AS_ALLOC_AS_EX),
        (IoctlVariant::Ioctl3, AS_SHIM_INFO),
        (IoctlVariant::Ioctl2, AS_GET_VA_REGIONS),
        // BindChannel, AllocateSpace, FreeSpace, UnmapBuffer, MapBufferEx, ModifyKind, native
        // remap, Remap (PR's `nvhost_as_gpu::Ioctl1` cases).
        (IoctlVariant::Ioctl, 0xC004_4101),
        (IoctlVariant::Ioctl, 0xC018_4102),
        (IoctlVariant::Ioctl, 0xC010_4103),
        (IoctlVariant::Ioctl, 0xC008_4105),
        (IoctlVariant::Ioctl, 0xC028_4106),
        (IoctlVariant::Ioctl, 0xC020_410C),
        (IoctlVariant::Ioctl, 0xC028_410D),
        (IoctlVariant::Ioctl, 0xC014_4114),
        // AllocAsEx's command in another group.
        (IoctlVariant::Ioctl, 0xC028_4709),
    ];
    for (variant, raw) in unknown {
        let input = plain(0, 0x10_0000, 0x100_0000, 0x40_0000);
        let answer = direct.call(
            UnknownPolicy::Strict,
            variant,
            fd,
            raw,
            &input,
            (0x40, 0x40),
        );
        assert_eq!(
            answer.result,
            NvResult::NotImplemented,
            "{variant:?} {raw:#x}"
        );
    }
    assert_eq!(direct.state.as_gpu().view(1), None);
}

/// Each field of `AllocAsEx` is read from its own place: generated layouts with every field
/// distinct are recorded as sent, so swapping any two reads fails. Flags are random (unread
/// without a parent), `as_fd` 0, so a swap of the two names a parent and is refused.
#[test]
fn alloc_as_ex_reads_each_field_from_its_place() {
    let config = Config {
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x0188_A5A5),
        cases: 128,
        ..Config::default()
    };
    let pages = VA_LIMIT >> 12;
    let strategy = (
        1..pages / 4,
        1..pages / 4,
        1..pages / 4,
        prop::sample::select(vec![0x1_0000_u32, 0x2_0000]),
        1..u32::MAX,
    );
    proptest!(config, |((a, b, c, big, flags) in strategy)| {
        let start = a << 12;
        let split = (a + b) << 12;
        let end = (a + b + c) << 12;
        let (mut direct, fd) = opened();
        let request = alloc(flags, 0, big, 0, [start, end, split]);
        prop_assert_eq!(direct.ioctl(fd, AS_ALLOC_AS_EX, &request).result, NvResult::Success);
        let layout = direct.state.as_gpu().view(1).unwrap().layout;
        prop_assert_eq!(layout, Layout { start, split, end, big_page_size: big });
    });
}

// ----- Through the server, beside the gpu session -------------------------------------------------

/// One gsv shared state under the gpu server and nvsrv: gpu 33's address space is the parent an
/// nvsrv view names, and a parentless `AllocAsEx` takes the next id of the same sequence, which
/// the gpu worker's jobs then name.
#[test]
fn gpu_33_and_nvsrv_share_address_spaces() {
    use crate::gpu::{command::CREATE_ADDRESS_SPACE, gpu_server};
    use crate::testing::tipc_type;
    use starquake_ipc::BUFFER_SIZE;
    use starquake_services::framework::Handled;
    use starquake_trace::ServiceId;

    let mut rig = Rig::new();
    let as_front = super::super::tests::as_front(&rig.front);
    let mut gpu = gpu_server(Arc::clone(&rig.shared), as_front, UnknownPolicy::Strict)
        .with_service_id(ServiceId(0x0749));
    let session = rig.kernel.new_object();
    gpu.accept(session).unwrap();
    // gpu 33 as PR checks it: header word 0x31 (16 + 33), one raw word, the word 0.
    let mut buffer = vec![STALE; BUFFER_SIZE];
    buffer[..12].copy_from_slice(&le(&[0x31, 1, 0]));
    assert_eq!(
        gpu.handle(&mut rig.kernel, session, &mut buffer),
        Ok(Handled::Replied)
    );
    assert_eq!(
        buffer[..16],
        le(&[tipc_type(CREATE_ADDRESS_SPACE), 2, 0, 1])
    );

    let view = rig.open_device(Device::NvhostAsGpu);
    let request = alloc(5, 1, 0x1000, 0, [WINDOW_START, 0xA0_0000_0000, 0]);
    let (reply, out) = rig.ioctl(view, AS_ALLOC_AS_EX, &request, request.len());
    assert_eq!((reply, out), (ok(&[0]), request));
    let (reply, out) = rig.ioctl(view, AS_SHIM_INFO, &shim_request(), 16);
    assert_eq!(reply, ok(&[0]));
    assert_eq!(out, shim_reply(0x9F_FFFF_F000, 4, SYNCPOINT_COUNT));
    let shim = rig.shared.lock().gpu().shim(AsId(1));
    assert_eq!(shim, Some(0x9F_FFFF_F000));

    let own = rig.open_device(Device::NvhostAsGpu);
    let request = plain(0, 0, 0, 0);
    let (reply, _) = rig.ioctl(own, AS_ALLOC_AS_EX, &request, request.len());
    assert_eq!(reply, ok(&[0]));
    assert_eq!(rig.state().as_gpu().view(2).unwrap().space, AsId(2));
    let spaces: Vec<AsId> = rig.shared.lock().gpu().address_spaces().collect();
    assert_eq!(spaces, [AsId(1), AsId(2)]);
    assert_eq!(
        rig.front.calls(),
        [FrontCall::Create(AsId(1)), FrontCall::Create(AsId(2))]
    );
    // gpu 33 again: the next id after nvsrv's.
    let mut buffer = vec![STALE; BUFFER_SIZE];
    buffer[..12].copy_from_slice(&le(&[0x31, 1, 0]));
    gpu.handle(&mut rig.kernel, session, &mut buffer).unwrap();
    assert_eq!(
        buffer[..16],
        le(&[tipc_type(CREATE_ADDRESS_SPACE), 2, 0, 3])
    );
}

/// `Ioctl3` `GetVARegions` through the server: the regions in out buffer 1, its tail left as it
/// was.
#[test]
fn va_regions_through_the_server() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::NvhostAsGpu);
    let request = plain(0x1_0000, 0x10_0000, 0x100_0000, 0x40_0000);
    rig.ioctl(fd, AS_ALLOC_AS_EX, &request, request.len());
    let (reply, out, inline) = rig.ioctl3(fd, AS_GET_VA_REGIONS, &va_request(), (0x40, 0x40));
    assert_eq!(reply, ok(&[0]));
    let regions = [
        region(0x10_0000, 0x1000, 0x300),
        region(0x40_0000, 0x1_0000, 0xC0),
    ];
    assert_eq!(out, va_reply(0x30, &regions));
    let mut expected = regions.concat();
    expected.resize(0x40, STALE);
    assert_eq!(inline, expected);
}

// ----- Saved state -------------------------------------------------------------------------------

/// A layout `AllocAsEx` makes.
const MADE: Layout = Layout {
    start: 0x10_0000,
    split: 0x40_0000,
    end: 0x100_0000,
    big_page_size: 0x2_0000,
};

/// A view of `parent`, as a flags-5 request makes it.
const fn child_of(parent: &Layout) -> Layout {
    Layout {
        start: parent.start,
        split: parent.split,
        end: parent.split,
        big_page_size: 0x1_0000,
    }
}

fn saved(views: &[(u32, View)], made: &[(AsId, Layout)]) -> SavedAsGpu {
    SavedAsGpu {
        views: views.iter().copied().collect(),
        made: made.iter().copied().collect(),
        bound: BTreeMap::new(),
    }
}

/// Each saved state `AllocAsEx` cannot reach, and the error it is refused with, first fault
/// first.
#[test]
fn a_saved_state_alloc_as_ex_cannot_reach_is_refused() {
    let space = |id: u32| AsId(id);
    let too_many: Vec<(AsId, Layout)> = (1..=u32::try_from(MAX_ADDRESS_SPACES).unwrap() + 1)
        .map(|id| (space(id), MADE))
        .collect();
    let bad = Layout { start: 0, ..MADE };
    let view = |id, layout| View {
        space: space(id),
        layout,
    };
    let cases = [
        (
            saved(&[], &too_many),
            AsGpuStateError::TooManySpaces {
                count: MAX_ADDRESS_SPACES + 1,
            },
        ),
        (
            saved(&[], &[(space(3), bad)]),
            AsGpuStateError::MadeLayout { space: 3 },
        ),
        (
            saved(
                &[],
                &[(
                    space(3),
                    Layout {
                        big_page_size: 0x3_0000,
                        ..MADE
                    },
                )],
            ),
            AsGpuStateError::MadeLayout { space: 3 },
        ),
        // A view of a made space with neither its layout nor a view of it.
        (
            saved(
                &[(
                    7,
                    view(
                        3,
                        Layout {
                            end: 0x200_0000,
                            ..MADE
                        },
                    ),
                )],
                &[(space(3), MADE)],
            ),
            AsGpuStateError::ViewLayout { fd: 7 },
        ),
        // A view of a gpu-33 space that is not a view of the native layout.
        (
            saved(&[(7, view(4, MADE))], &[]),
            AsGpuStateError::ViewLayout { fd: 7 },
        ),
        // Two faults: the count before a layout, a made layout before a view.
        (
            {
                let mut both = too_many.clone();
                both[0].1 = bad;
                saved(&[], &both)
            },
            AsGpuStateError::TooManySpaces {
                count: MAX_ADDRESS_SPACES + 1,
            },
        ),
        (
            saved(&[(7, view(4, MADE))], &[(space(3), bad)]),
            AsGpuStateError::MadeLayout { space: 3 },
        ),
    ];
    let more = unissued_and_creators().into_iter().chain(unheld());
    for (state, error) in cases.into_iter().chain(more) {
        assert_eq!(AsGpu::try_from(state), Err(error), "{error}");
    }
}

/// The address-space id and creator cases of
/// `a_saved_state_alloc_as_ex_cannot_reach_is_refused`, first fault first.
fn unissued_and_creators() -> Vec<(SavedAsGpu, AsGpuStateError)> {
    let space = |id: u32| AsId(id);
    let bad = Layout { start: 0, ..MADE };
    let below_first = AsId(FIRST_ADDRESS_SPACE.0 - 1);
    let view = |id, layout| View {
        space: space(id),
        layout,
    };
    vec![
        // An id below the first gsv hands out: as a made space, and as a view's space (whose
        // layout would otherwise pass as a view of a gpu-33 space).
        (
            saved(&[], &[(below_first, MADE)]),
            AsGpuStateError::MadeSpaceNotIssued {
                space: below_first.0,
            },
        ),
        (
            saved(&[(7, view(below_first.0, child_of(&NATIVE_LAYOUT)))], &[]),
            AsGpuStateError::ViewSpaceNotIssued {
                fd: 7,
                space: below_first.0,
            },
        ),
        // Two faults: an unissued made id before its layout, before an unissued view id.
        (
            saved(
                &[(7, view(below_first.0, child_of(&NATIVE_LAYOUT)))],
                &[(below_first, bad)],
            ),
            AsGpuStateError::MadeSpaceNotIssued {
                space: below_first.0,
            },
        ),
        // An unissued view id before a made layout.
        (
            saved(
                &[(7, view(below_first.0, child_of(&NATIVE_LAYOUT)))],
                &[(space(3), bad)],
            ),
            AsGpuStateError::ViewSpaceNotIssued {
                fd: 7,
                space: below_first.0,
            },
        ),
        // Two fds with the creator's own layout, which is not a view of itself: only the fd whose
        // `AllocAsEx` made the space has it.
        (
            saved(
                &[(1, view(3, MADE)), (2, view(3, MADE))],
                &[(space(3), MADE)],
            ),
            AsGpuStateError::TwoCreators {
                space: 3,
                first: 1,
                second: 2,
            },
        ),
        // A view layout fault before two creators.
        (
            saved(
                &[
                    (1, view(3, MADE)),
                    (2, view(3, MADE)),
                    (
                        9,
                        view(
                            3,
                            Layout {
                                end: 0x200_0000,
                                ..MADE
                            },
                        ),
                    ),
                ],
                &[(space(3), MADE)],
            ),
            AsGpuStateError::ViewLayout { fd: 9 },
        ),
    ]
}

/// The unheld-space cases of `a_saved_state_alloc_as_ex_cannot_reach_is_refused`: a made space
/// no view holds (only a destroy leaves one), alone, beside a held one, and after the faults
/// checked before it.
fn unheld() -> Vec<(SavedAsGpu, AsGpuStateError)> {
    let space = |id: u32| AsId(id);
    let view = |id, layout| View {
        space: space(id),
        layout,
    };
    vec![
        (
            saved(&[], &[(space(3), MADE)]),
            AsGpuStateError::MadeSpaceUnheld { space: 3 },
        ),
        // Two unheld: the lower id is reported.
        (
            saved(&[], &[(space(3), MADE), (space(4), MADE)]),
            AsGpuStateError::MadeSpaceUnheld { space: 3 },
        ),
        (
            saved(
                &[(1, view(4, MADE)), (2, view(5, child_of(&MADE)))],
                &[(space(3), MADE), (space(4), MADE), (space(5), MADE)],
            ),
            AsGpuStateError::MadeSpaceUnheld { space: 3 },
        ),
        (
            saved(
                &[(1, view(3, MADE)), (2, view(3, MADE))],
                &[(space(3), MADE), (space(5), MADE)],
            ),
            AsGpuStateError::TwoCreators {
                space: 3,
                first: 1,
                second: 2,
            },
        ),
        (
            saved(
                &[(
                    9,
                    view(
                        4,
                        Layout {
                            end: 0x200_0000,
                            ..MADE
                        },
                    ),
                )],
                &[(space(3), MADE), (space(4), MADE)],
            ),
            AsGpuStateError::ViewLayout { fd: 9 },
        ),
    ]
}

/// Each relation's reachable side is accepted: the most made spaces, a creator's view, a view of
/// a made space, a made space held by a child view alone, a view of a gpu-33 space; and the
/// state survives `postcard`.
#[test]
fn a_saved_state_at_each_edge_is_accepted() {
    let all: Vec<(AsId, Layout)> = (1..=u32::try_from(MAX_ADDRESS_SPACES).unwrap())
        .map(|id| (AsId(id), MADE))
        .collect();
    let view = |space, layout| View {
        space: AsId(space),
        layout,
    };
    let views = [
        (3, view(1, MADE)),
        (5, view(1, child_of(&MADE))),
        (6, view(20, child_of(&NATIVE_LAYOUT))),
        // A made space held only by a child view, as its creator's close leaves it
        // (`a_child_keeps_the_space_nameable_after_its_creator_closes`).
        (7, view(2, child_of(&MADE))),
    ];
    // Every other made space held by its creator's fd.
    let creators =
        (3..=u32::try_from(MAX_ADDRESS_SPACES).unwrap()).map(|id| (10 + id, view(id, MADE)));
    let views: Vec<(u32, View)> = views.into_iter().chain(creators).collect();
    let state = AsGpu::try_from(saved(&views, &all)).unwrap();
    let bytes = postcard::to_allocvec(&state).unwrap();
    assert_eq!(postcard::from_bytes::<AsGpu>(&bytes).unwrap(), state);
    // The first id gsv hands out is accepted, as a made space and as a view's.
    let first = [(FIRST_ADDRESS_SPACE, MADE)];
    let on_first = [(
        2,
        View {
            space: FIRST_ADDRESS_SPACE,
            layout: MADE,
        },
    )];
    assert!(AsGpu::try_from(saved(&on_first, &first)).is_ok());
    let native_first = [(
        2,
        View {
            space: FIRST_ADDRESS_SPACE,
            layout: child_of(&NATIVE_LAYOUT),
        },
    )];
    assert!(AsGpu::try_from(saved(&native_first, &[])).is_ok());
    // A creator whose layout is a view of itself (split at the end, 4 KiB big pages asked for)
    // shares it with any number of views; one creator of each of two spaces is accepted too.
    let own = child_of(&MADE);
    assert!(own.is_view_of(&own));
    let shared = [
        (
            1,
            View {
                space: AsId(3),
                layout: own,
            },
        ),
        (
            2,
            View {
                space: AsId(3),
                layout: own,
            },
        ),
        (
            4,
            View {
                space: AsId(5),
                layout: MADE,
            },
        ),
    ];
    assert!(AsGpu::try_from(saved(&shared, &[(AsId(3), own), (AsId(5), MADE)])).is_ok());
    // Through postcard, a refused state is refused.
    let refused = AsGpu {
        views: BTreeMap::from([(
            6,
            View {
                space: AsId(20),
                layout: MADE,
            },
        )]),
        made: BTreeMap::new(),
        bound: BTreeMap::new(),
    };
    let bytes = postcard::to_allocvec(&refused).unwrap();
    assert!(postcard::from_bytes::<AsGpu>(&bytes).is_err());
}

#[test]
fn a_refused_state_is_said_in_plain_words() {
    let said = [
        (
            AsGpuStateError::TooManySpaces { count: 9 },
            "9 address spaces were made through nvhost-as-gpu, more than can be",
        ),
        (
            AsGpuStateError::MadeLayout { space: 3 },
            "address space 3 has a layout nvhost-as-gpu cannot make",
        ),
        (
            AsGpuStateError::ViewLayout { fd: 7 },
            "fd 7 has a view its address space cannot have",
        ),
        (
            AsGpuStateError::MadeSpaceNotIssued { space: 4 },
            "address space 4 was made through nvhost-as-gpu but is not an id gsv hands out",
        ),
        (
            AsGpuStateError::ViewSpaceNotIssued { fd: 6, space: 5 },
            "fd 6 has a view of address space 5, which is not an id gsv hands out",
        ),
        (
            AsGpuStateError::TwoCreators {
                space: 3,
                first: 1,
                second: 2,
            },
            "fds 1 and 2 both hold the layout address space 3 was made with",
        ),
        (
            AsGpuStateError::MadeSpaceUnheld { space: 3 },
            "address space 3 was made through nvhost-as-gpu but no fd holds it",
        ),
        (
            AsGpuStateError::MadeSpaceNotHeld { space: 4 },
            "address space 4 was made through nvhost-as-gpu but gsv does not hold it",
        ),
        (
            AsGpuStateError::ViewSpaceNotHeld { fd: 6, space: 5 },
            "fd 6 has a view of address space 5, which gsv does not hold",
        ),
    ];
    for (error, words) in said {
        assert_eq!(error.to_string(), words);
    }
    assert_eq!(
        NvsrvStateError::NotAnAddressSpace { fd: 3 }.to_string(),
        "fd 3 has an address-space view but is not an open nvhost-as-gpu"
    );
}

/// The nvsrv state checks a view's fd against its files: a view on an fd that is not an open
/// as-gpu is refused, before the groups.
#[test]
fn a_saved_view_must_fit_the_files() {
    let (mut direct, fd) = opened();
    let tsg = direct.open(Device::NvhostTsgGpu);
    let request = plain(0, 0x10_0000, 0x100_0000, 0x40_0000);
    assert_eq!(
        direct.ioctl(fd, AS_ALLOC_AS_EX, &request).result,
        NvResult::Success
    );
    let timeslice = le(&[100]);
    let set = direct.ioctl(tsg, crate::nvsrv::tsg_gpu::SET_TIMESLICE, &timeslice);
    assert_eq!(set.result, NvResult::Success);
    let state = direct.state;
    let check = |state: &NvsrvState| NvsrvState::try_from(super::super::tests::saved(state));
    assert_eq!(check(&state).as_ref(), Ok(&state));
    let fd = u32::try_from(fd).unwrap();
    let tsg = u32::try_from(tsg).unwrap();
    for device in [Device::Nvmap, Device::NvhostGpu, Device::NvhostTsgGpu] {
        let mut changed = state.clone();
        changed.files.insert(fd, device);
        assert_eq!(
            check(&changed),
            Err(NvsrvStateError::NotAnAddressSpace { fd })
        );
    }
    let mut closed = state.clone();
    closed.files.remove(&fd);
    assert_eq!(
        check(&closed),
        Err(NvsrvStateError::NotAnAddressSpace { fd })
    );
    // Both a view and a group on the wrong files: the view is reported.
    let mut both = state;
    both.files.insert(fd, Device::Nvmap);
    both.files.insert(tsg, Device::Nvmap);
    assert_eq!(check(&both), Err(NvsrvStateError::NotAnAddressSpace { fd }));
}

/// What a restore accepts as a view of a parent: each of [`Layout::is_view_of`]'s conditions
/// failing alone is refused, and the edges are accepted.
#[test]
fn a_view_of_a_parent_is_checked_condition_by_condition() {
    let parent = MADE;
    let view = child_of(&parent);
    assert!(view.is_view_of(&parent));
    let page = SMALL_PAGE_SIZE;
    let accepted = [
        Layout {
            end: parent.split,
            split: parent.split,
            ..view
        },
        Layout {
            start: parent.split - page,
            ..view
        },
    ];
    for layout in accepted {
        assert!(layout.is_view_of(&parent), "{layout:?}");
    }
    let refused = [
        Layout {
            split: view.end - page,
            ..view
        },
        Layout {
            big_page_size: 0x2_0000,
            ..view
        },
        Layout {
            start: parent.start - page,
            ..view
        },
        Layout {
            start: view.end,
            ..view
        },
        Layout {
            end: parent.split + page,
            split: parent.split + page,
            ..view
        },
        Layout {
            start: parent.start + 0x800,
            ..view
        },
        Layout {
            end: parent.split - 0x800,
            split: parent.split - 0x800,
            ..view
        },
    ];
    for layout in refused {
        assert!(!layout.is_view_of(&parent), "{layout:?}");
    }
}

/// A server saved with a view and a timeslice group restores, over the same gsv state, and
/// carries on: the view answers its regions and shim, the group unbinds its channel, and the
/// next `AllocAsEx` makes the next address space.
#[test]
fn a_restored_server_carries_on_with_views_and_groups() {
    use super::super::tests::{as_front, cmif};
    use super::super::{CLOSE, nvsrv_server};

    let mut rig = Rig::new();
    let fd = rig.open_device(Device::NvhostAsGpu);
    let tsg = rig.open_device(Device::NvhostTsgGpu);
    let channel = rig.open_device(Device::NvhostGpu);
    let request = plain(0x1_0000, 0x10_0000, 0x100_0000, 0x40_0000);
    rig.ioctl(fd, AS_ALLOC_AS_EX, &request, request.len());
    let bind = le(&[channel.cast_unsigned()]);
    let (reply, _) = rig.ioctl(tsg, crate::nvsrv::tsg_gpu::BIND_CHANNEL, &bind, 4);
    assert_eq!(reply, ok(&[0]));
    let saved = rig.server.save().unwrap();
    let state = rig.state().clone();

    let restored = nvsrv_server(
        Arc::clone(&rig.shared),
        as_front(&rig.front),
        UnknownPolicy::Strict,
    );
    rig.server = restored.with_service_id(rig.server.service_id());
    rig.server.restore(&saved).unwrap();
    assert_eq!(rig.state(), &state);
    let (reply, out) = rig.ioctl(fd, AS_GET_VA_REGIONS, &va_request(), 0x40);
    assert_eq!(reply, ok(&[0]));
    let regions = [
        region(0x10_0000, 0x1000, 0x300),
        region(0x40_0000, 0x1_0000, 0xC0),
    ];
    assert_eq!(out, va_reply(0x30, &regions));
    let (reply, out) = rig.ioctl(fd, AS_SHIM_INFO, &shim_request(), 16);
    assert_eq!(
        (reply, out),
        (ok(&[0]), shim_reply(0x3F_F000, 4, SYNCPOINT_COUNT))
    );
    let unbind = crate::nvsrv::tsg_gpu::UNBIND_CHANNEL;
    let (reply, _) = rig.ioctl(tsg, unbind, &bind, 4);
    assert_eq!(reply, ok(&[0]));
    let other = rig.open_device(Device::NvhostAsGpu);
    assert_eq!(other, channel + 1);
    let (reply, _) = rig.ioctl(other, AS_ALLOC_AS_EX, &request, request.len());
    assert_eq!(reply, ok(&[0]));
    assert_eq!(
        rig.state()
            .as_gpu()
            .view(other.cast_unsigned())
            .unwrap()
            .space,
        AsId(2)
    );
    // `Close` through the server destroys the restored view's space in gsv's shared state, its
    // last fd, and keeps the other.
    rig.send(&cmif(CLOSE, &fd.to_le_bytes()));
    assert_eq!(rig.state().as_gpu().view(fd.cast_unsigned()), None);
    let spaces: Vec<AsId> = rig.shared.lock().gpu().address_spaces().collect();
    assert_eq!(spaces, [AsId(2)]);
    assert_eq!(rig.state().as_gpu().made(AsId(1)), None);
}

/// A view whose address space gsv does not hold (a restore whose nvsrv and gsv states disagree,
/// module docs of `crate::nvsrv`) answers its shim with `BadValue` and zero fields, and nothing
/// is placed; its regions still answer from its layout.
#[test]
fn a_view_of_a_space_gsv_does_not_hold_answers_bad_value() {
    let (mut direct, fd) = opened();
    let request = plain(0, 0x10_0000, 0x100_0000, 0x40_0000);
    assert_eq!(
        direct.ioctl(fd, AS_ALLOC_AS_EX, &request).result,
        NvResult::Success
    );
    // The same nvsrv state over a GPU memory that holds nothing.
    let mut other = Direct::new();
    other.state = direct.state.clone();
    let answer = other.ioctl(fd, AS_SHIM_INFO, &shim_request());
    assert_eq!(
        (answer.result, answer.output),
        (NvResult::BadValue, shim_reply(0, 0, 0))
    );
    assert_eq!(other.memory.address_spaces().count(), 0);
    let answer = other.ioctl(fd, AS_GET_VA_REGIONS, &va_request());
    assert_eq!(answer.result, NvResult::Success);
}
