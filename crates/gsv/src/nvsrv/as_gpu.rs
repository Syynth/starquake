//! `/dev/nvhost-as-gpu`: address-space views over gsv's GPU address spaces, the VA regions they
//! report, and the syncpoint shim (`docs/specs/graphics-server.md`, "Channels and submission:
//! nvsrv", the device table's as-gpu row, and "The model": "Address spaces (gpu 33, and nvsrv's
//! nvhost-as-gpu opens): one object reachable from both paths by id"; port slice 10).
//!
//! Ported from Eden PR #4588 (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project) @
//! e5fb3b97: `eden-pr4588/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.cpp` and
//! `nvhost_as_gpu.h` (`AllocAsEx`, `GetVARegionsImpl`, `GetVARegions1`, `GetVARegions3`,
//! `GetNativeSyncpointShim`, `GetSyncpointShimInfo`, the argument structs) and
//! `gpu_address_space_layout.h` (`GetGpuAddressSpaceLayout`, `GetGpuAddressSpaceBigPageSize`,
//! `IsSupportedNativeAddressSpaceView`); and from Eden-NX2 (GPL-3.0-or-later) @ 50fa86bb,
//! `eden-nx2/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.cpp` (`AllocAsEx`'s split-0 case,
//! `Nx2GetSyncpointShimInfo`). "Row N" is a row of `eden-nx2/NX2_NOTES.md` @ 50fa86bb.
//!
//! # The model
//!
//! An as-gpu fd starts with nothing. `AllocAsEx` gives it a **view**: an address space of gsv's
//! GPU memory ([`crate::gpu::GpuMemory`], the one the gpu session's 33 and worker jobs use, under
//! gsv's shared lock) and a [`Layout`], the ranges the fd reports. Either:
//!
//! - the request names a **parent** in its `as_fd` word: the view shares that address space,
//!   and its small-page region ends where the request's range ends (PR: `vm.va_range_split =
//!   params.va_range_end`, "GPU mappings must resolve through the same page table"); or
//! - it names none (`as_fd` 0): a new address space is made, through the same
//!   [`GpuMemory::create_address_space`](crate::gpu::GpuMemory::create_address_space) gpu 33
//!   calls, so it counts against the same [`MAX_ADDRESS_SPACES`] and takes the next id of the
//!   same sequence.
//!
//! **What `as_fd` names.** In PR it is an nvdrv fd, because PR's gpu 33 opens an as-gpu fd and
//! answers that fd as the address-space id (`IGpuSession::CreateAddressSpace`). So the id gpu 33
//! answers is what NvRm passes. Here gpu 33's id is gsv's own ([`AsId`]), not an nvsrv fd, so
//! `as_fd` is read as an [`AsId`]: it names the address space gpu 33 (or an earlier parentless
//! `AllocAsEx`) made, checked against what gsv holds. A title that named another as-gpu *fd*
//! as the parent would find a different object here, or none; nothing shows NvRm doing that
//! (a **guess**: a trace of `AllocAsEx`'s `as_fd` against gpu 33's reply settles it).
//!
//! # The ioctls
//!
//! Group `'A'` ([`GROUP`]). Each argument is a fixed struct ([`super::ioctl::Fixed`]), given back
//! whole whatever the result, every field the ioctl does not set as the caller sent it.
//!
//! | Variant | Cmd | Ioctl | Argument (offset: field) |
//! |---|---|---|---|
//! | `Ioctl` | 0x9 | `AllocAsEx` | 0: flags, 4: `as_fd` (`s32`), 8: big page size, 12: reserved, 16: start, 24: end, 32: split (the three `u64`) |
//! | `Ioctl`, `Ioctl3` | 0x8 | `GetVARegions` | 0: buffer address (`u64`), 8: buffer size, 12: reserved, 16: two regions of 0x18 bytes (offset `u64`, page size, padding, pages `u64`); `Ioctl3` also writes the regions to out buffer 1 |
//! | `Ioctl` | 0xB | `GetSyncpointShimInfo` | 0: address (`u64`), 8: stride, 12: count |
//!
//! **`AllocAsEx`**, PR's checks in its order, the first that fails answering:
//!
//! 1. the fd has a view already: `InvalidState`;
//! 2. with a parent: `as_fd` negative, no address space of that id, or the request not a view
//!    PR supports ([`is_supported_view`]: flags 5 with split 0, or flags 0 with split equal to
//!    the end; page size 0x1000; reserved 0; a non-empty range inside the parent's small-page
//!    region; start and end page aligned): `BadParameter`;
//! 3. the big page size after PR's mapping of 0x1000 to 0x10000 ([`big_page_size`]) not 0, not
//!    a power of two or not one of [`SUPPORTED_BIG_PAGE_SIZES`]: `BadValue`;
//! 4. (starquake's, see "Where this differs") the layout not page aligned, its ranges out of
//!    order, or its end past [`VA_LIMIT`]: `BadValue`;
//! 5. no parent, and gsv holds [`MAX_ADDRESS_SPACES`] already: `InsufficientMemory` (a guess:
//!    PR opens a new GMMU and has no limit).
//!
//! The layout: a start of 0 takes PR's native layout ([`NATIVE_LAYOUT`]'s ranges, PR
//! `GetGpuAddressSpaceLayout(true)` when `!params.va_range_start`); otherwise the request's
//! start, split and end. With a parent, the split becomes the end. Without one and with a split
//! of 0, the split also becomes the end (a **conflict**, below). The big page size is the
//! request's (0 keeps [`DEFAULT_BIG_PAGE_SIZE`]).
//!
//! **`GetVARegions`**: no view, `BadValue`. Otherwise the buffer size and two regions, as PR's
//! `GetVARegionsImpl`: the small pages from the start to the split, the big pages from the split
//! (rounded down to a big page) to the end; when there are no big pages the second region is
//! zero and the buffer size one region ([`regions`]). `Ioctl3` writes as many regions to out
//! buffer 1 as the buffer size says and out buffer 1 holds, and zero regions after them up to
//! what it holds of two (Eden `WrapFixedInlOut` writes every whole region the buffer holds, zero
//! past the ones set); on a refusal, those are all zero.
//!
//! **`GetSyncpointShimInfo`**: no view, `NotSupported`, and the three fields 0 (PR's out values
//! are written whatever the result). Otherwise the shim's page: the page below the view's split
//! (PR `vm.va_range_split - VM::YUZU_PAGESIZE`), placed in the address space the first time
//! ([`GpuMemory::map_syncpoint_shim`](crate::gpu::GpuMemory::map_syncpoint_shim)); the stride 4
//! and the count [`SYNCPOINT_COUNT`] (192). An address space has one shim: a view whose address
//! space has one already reports that one's page.
//!
//! Every other as-gpu ioctl goes to the server's unknown-ioctl policy (see "Not built").
//!
//! # Conflicts
//!
//! Each is picked by decision 8 and pinned by the test named, in `as_gpu/tests.rs`. The spec's
//! verification item 1 does not list them yet; the open question with each settles it.
//!
//! - **`AllocAsEx` with no parent and split 0** (row 25's request, if NvRm sends `as_fd` 0):
//!   PR keeps split 0, which leaves its small-page region empty and its shim below address 0;
//!   NX2 makes the split the end and adds 4 GiB of big pages past it. Here the split becomes
//!   the end, as both do, and the big-page region stays empty, as PR's views have it (the leg
//!   "the one a title ran on": NX2 ran on row 25; its extra 4 GiB only kept its own big-page
//!   allocator valid, which nothing here has). Open question: does NvRm's `AllocAsEx` carry
//!   gpu 33's id in `as_fd`? (`split_0_without_a_parent_takes_small_pages_to_the_end`.)
//! - **The shim's address**: PR puts it on the page below the split, NX2 on the page below its
//!   end. PR's is used: both are reported to the title (row 46), and PR's keeps it inside the
//!   view's small-page region (`the_shim_is_on_the_page_below_the_split`).
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `as_gpu/tests.rs` unless another file is given:
//!
//! | Case | PR | Here |
//! |---|---|---|
//! | `as_fd` | an nvdrv fd | a gsv address-space id (see "The model") (`a_view_shares_its_parent_s_address_space`) |
//! | `as_fd` naming the fd itself | `BadParameter` | cannot happen: the fd has no address space before its `AllocAsEx` |
//! | The parent's layout | the parent fd's own | the layout an earlier parentless `AllocAsEx` gave the space, or [`NATIVE_LAYOUT`] (what PR's gpu 33 asks for) for a space gpu 33 made (`a_view_of_a_gpu_session_space_uses_the_native_layout`) |
//! | A layout out of order, not page aligned, or past [`VA_LIMIT`] | accepted; its allocators then hold truncated 32-bit page numbers | `BadValue`, nothing made (`a_layout_out_of_order_is_a_bad_value`, `a_layout_past_the_va_limit_is_a_bad_value`) |
//! | More address spaces than [`MAX_ADDRESS_SPACES`] | no limit | `InsufficientMemory`, shared with gpu 33's count (`address_spaces_are_bounded_with_gpu_33_s`) |
//! | Closing the fd | frees its address space when nothing else holds it | the view goes; the address space stays in gsv, as gpu 33's do (gsv has no destroy of an address space), so a title that opens, allocates and closes reaches [`MAX_ADDRESS_SPACES`] (`closing_keeps_the_address_space`) |
//! | Two views of one address space asking for the shim | each places its own (the GMMU keeps the last) | the first places it, the second reports the same page (`a_second_view_reports_the_same_shim`) |
//! | `Ioctl3` `GetVARegions` with out buffer 1 holding more than two regions | every whole region it holds written, zero past the two | the first two only (`va_regions_through_ioctl3`) |
//!
//! # Not built
//!
//! `BindChannel` (0x1, which needs nvhost-gpu's channels, port slice 11), `AllocateSpace` (0x2),
//! `FreeSpace` (0x3), `UnmapBuffer` (0x5), `MapBufferEx` (0x6), `ModifyKind` (0xC), PR's native
//! remap (0xD) and `Remap` (0x14). Row 30 (observed): Switch 2 never maps through nvhost-as-gpu;
//! row 149 (observed): NvRm's mapping modify reaches gsv as gpu 212, which the gpu worker
//! serves. `ModifyKind` and `Remap` are in the spec's device table from PR alone (read); they
//! need an ordered job with no notification event in gsv's GPU memory, and `Remap` the
//! allocations `AllocateSpace` makes. They go to the unknown-ioctl policy until a title shows
//! them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::SYNCPOINT_COUNT;
use starquake_memory::PAGE_SIZE;

use super::ctrl_gpu;
use super::ioctl::Fixed;
use super::{GpuAccess, IoctlAnswer, IoctlCall, IoctlVariant, NvResult};
use crate::gpu::AsId;
use crate::gpu::memory::{MAX_ADDRESS_SPACES, VA_WINDOW_END, VA_WINDOW_START};

#[cfg(test)]
mod tests;

/// as-gpu's ioctl group, `'A'` (`nvhost_as_gpu::Ioctl1`: `case 'A':`).
pub const GROUP: u32 = 0x41;

/// as-gpu's commands in the group (`nvhost_as_gpu::Ioctl1` and `Ioctl3`).
pub mod command {
    /// `GetVARegions` (`case 0x8:`), on `Ioctl` and `Ioctl3`.
    pub const GET_VA_REGIONS: u32 = 0x8;
    /// `AllocAsEx` (`case 0x9:`).
    pub const ALLOC_AS_EX: u32 = 0x9;
    /// `GetSyncpointShimInfo` (`case 0xb:`).
    pub const GET_SYNCPOINT_SHIM_INFO: u32 = 0xB;
}

/// Where each field of each argument struct is, and each struct's size (`nvhost_as_gpu.h`:
/// `IoctlAllocAsEx`, `IoctlGetVaRegions`, `VaRegion`, `IoctlSyncpointShimInfo` and their
/// `static_assert`ed sizes; little-endian `u32`s unless noted).
pub mod layout {
    /// `IoctlAllocAsEx`: 40 bytes.
    pub const ALLOC_AS_EX_BYTES: usize = 40;
    /// `flags`.
    pub const ALLOC_FLAGS: usize = 0;
    /// `as_fd`, an `s32`.
    pub const ALLOC_AS_FD: usize = 4;
    /// `big_page_size`.
    pub const ALLOC_BIG_PAGE_SIZE: usize = 8;
    /// `reserved`.
    pub const ALLOC_RESERVED: usize = 12;
    /// `va_range_start`, a `u64`.
    pub const ALLOC_START: usize = 16;
    /// `va_range_end`, a `u64`.
    pub const ALLOC_END: usize = 24;
    /// `va_range_split`, a `u64`.
    pub const ALLOC_SPLIT: usize = 32;

    /// `IoctlGetVaRegions`: 16 bytes and two regions.
    pub const VA_REGIONS_BYTES: usize = 16 + 2 * REGION_BYTES;
    /// `buf_addr`, a `u64`, which nothing reads.
    pub const VA_BUF_ADDR: usize = 0;
    /// `buf_size`, set to the regions' size.
    pub const VA_BUF_SIZE: usize = 8;
    /// The first region.
    pub const VA_REGION_0: usize = 16;
    /// The second region.
    pub const VA_REGION_1: usize = VA_REGION_0 + REGION_BYTES;

    /// `VaRegion`: 0x18 bytes.
    pub const REGION_BYTES: usize = 0x18;
    /// `offset`, a `u64`.
    pub const REGION_OFFSET: usize = 0;
    /// `page_size`.
    pub const REGION_PAGE_SIZE: usize = 8;
    /// `pages`, a `u64` (after four bytes of padding).
    pub const REGION_PAGES: usize = 16;

    /// `IoctlSyncpointShimInfo`: 16 bytes.
    pub const SHIM_BYTES: usize = 16;
    /// `address`, a `u64`.
    pub const SHIM_ADDRESS: usize = 0;
    /// `stride`.
    pub const SHIM_STRIDE: usize = 8;
    /// `count`.
    pub const SHIM_COUNT: usize = 12;
}

use layout::{
    ALLOC_AS_EX_BYTES, ALLOC_AS_FD, ALLOC_BIG_PAGE_SIZE, ALLOC_END, ALLOC_FLAGS, ALLOC_RESERVED,
    ALLOC_SPLIT, ALLOC_START, REGION_BYTES, REGION_OFFSET, REGION_PAGE_SIZE, REGION_PAGES,
    SHIM_ADDRESS, SHIM_BYTES, SHIM_COUNT, SHIM_STRIDE, VA_BUF_SIZE, VA_REGION_0, VA_REGION_1,
    VA_REGIONS_BYTES,
};

/// The small page: PR's `VM::YUZU_PAGESIZE` (0x1000), the memory crate's page.
pub const SMALL_PAGE_SIZE: u64 = PAGE_SIZE;

/// The big page sizes `AllocAsEx` accepts, as a mask (PR `VM::SUPPORTED_BIG_PAGE_SIZES =
/// 0x30000`: 64 KiB and 128 KiB).
pub const SUPPORTED_BIG_PAGE_SIZES: u32 = 0x3_0000;

/// The big page size when a request gives none (PR `VM::DEFAULT_BIG_PAGE_SIZE = 0x20000`), and
/// the one PR's gpu 33 asks for (`IGpuSession::CreateAddressSpace`: `big_page_size{0x20000}`).
pub const DEFAULT_BIG_PAGE_SIZE: u32 = 0x2_0000;

/// What a request for 4 KiB big pages gets instead: PR `GetGpuAddressSpaceBigPageSize`
/// (`extended && requested == 0x1000 ? 0x10000 : requested`), "a separate large-page pool".
pub const NATIVE_BIG_PAGE_SIZE: u32 = 0x1_0000;

/// One past the last GPU address a layout may reach: the 40-bit GPU address space the
/// characteristics block reports ([`ctrl_gpu::value::VA_BITS`]); PR "Both layouts fit within the
/// GPU's reported 40-bit virtual address width", and its native layout ends there.
pub const VA_LIMIT: u64 = 1 << ctrl_gpu::value::VA_BITS;

/// `AllocAsEx`'s flags for a native small-pages view with split 0 (PR
/// `IsSupportedNativeAddressSpaceView`: `flags == 5 && requested.split == 0`; row 25: "flags 5").
pub const NATIVE_VIEW_FLAGS: u32 = 5;

/// `AllocAsEx`'s flags for a view whose split is its end (PR: `flags == 0 && requested.split ==
/// requested.end`).
pub const LEGACY_VIEW_FLAGS: u32 = 0;

/// The shim's stride: one `u32` per syncpoint (PR `stride = sizeof(u32)`).
pub const SHIM_STRIDE_BYTES: u32 = 4;

/// The ranges an as-gpu fd reports: small pages from `start` to `split`, big pages of
/// `big_page_size` from `split` to `end` (PR's `VM`: `va_range_start`, `va_range_split`,
/// `va_range_end`, `big_page_size`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    /// The first GPU address.
    pub start: u64,
    /// Where the big pages start.
    pub split: u64,
    /// One past the last GPU address.
    pub end: u64,
    /// The big page size.
    pub big_page_size: u32,
}

/// PR's native layout (`GetGpuAddressSpaceLayout(true)`: start `0x8000000`, split `3ULL << 38`,
/// end `1ULL << 40`), with the big page size PR's gpu 33 asks for. Its start and split are NVN's
/// window ([`VA_WINDOW_START`], [`VA_WINDOW_END`]; row 25), its end [`VA_LIMIT`].
pub const NATIVE_LAYOUT: Layout = Layout {
    start: VA_WINDOW_START,
    split: VA_WINDOW_END,
    end: VA_LIMIT,
    big_page_size: DEFAULT_BIG_PAGE_SIZE,
};

impl Layout {
    /// Whether the layout is one `AllocAsEx` can make: page-aligned ranges in order, a start that
    /// is not 0, an end within [`VA_LIMIT`], and a supported big page size.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.start != 0
            && self.start.is_multiple_of(SMALL_PAGE_SIZE)
            && self.split.is_multiple_of(SMALL_PAGE_SIZE)
            && self.end.is_multiple_of(SMALL_PAGE_SIZE)
            && self.start <= self.split
            && self.split <= self.end
            && self.end <= VA_LIMIT
            && is_supported_big_page_size(self.big_page_size)
    }

    /// Whether the layout is one an `AllocAsEx` with `parent`'s address space as its parent can
    /// make: [`is_supported_view`]'s range checks, the split at the end, and the big page size
    /// a 4 KiB request gets.
    #[must_use]
    pub const fn is_view_of(&self, parent: &Self) -> bool {
        self.split == self.end
            && self.big_page_size == NATIVE_BIG_PAGE_SIZE
            && self.start >= parent.start
            && self.start < self.end
            && self.end <= parent.split
            && self.start.is_multiple_of(SMALL_PAGE_SIZE)
            && self.end.is_multiple_of(SMALL_PAGE_SIZE)
    }
}

/// Whether `size` is a big page size `AllocAsEx` keeps: a power of two among
/// [`SUPPORTED_BIG_PAGE_SIZES`] (PR: `std::has_single_bit` and `& VM::SUPPORTED_BIG_PAGE_SIZES`).
#[must_use]
pub const fn is_supported_big_page_size(size: u32) -> bool {
    size.is_power_of_two() && size & SUPPORTED_BIG_PAGE_SIZES != 0
}

/// The big page size a request's word gives (PR `GetGpuAddressSpaceBigPageSize(true, ..)`).
#[must_use]
pub const fn big_page_size(requested: u32) -> u32 {
    if requested == 0x1000 {
        NATIVE_BIG_PAGE_SIZE
    } else {
        requested
    }
}

/// The ranges an `AllocAsEx` asks for: start, split, end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Requested {
    /// `va_range_start`.
    pub start: u64,
    /// `va_range_split`.
    pub split: u64,
    /// `va_range_end`.
    pub end: u64,
}

/// PR's `IsSupportedNativeAddressSpaceView`, check for check: small pages only (flags 5 with
/// split 0, or flags 0 with split at the end), page size 0x1000, reserved 0, the start inside
/// the parent and below the end, the end within the parent's split, both page aligned.
#[must_use]
pub fn is_supported_view(
    flags: u32,
    page_size: u32,
    reserved: u32,
    requested: Requested,
    parent: &Layout,
) -> bool {
    let small_pages_only = (flags == NATIVE_VIEW_FLAGS && requested.split == 0)
        || (flags == LEGACY_VIEW_FLAGS && requested.split == requested.end);
    small_pages_only
        && u64::from(page_size) == SMALL_PAGE_SIZE
        && reserved == 0
        && requested.start >= parent.start
        && requested.start < requested.end
        && requested.end <= parent.split
        && requested.start.is_multiple_of(SMALL_PAGE_SIZE)
        && requested.end.is_multiple_of(SMALL_PAGE_SIZE)
}

/// One `VaRegion`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Region {
    /// Its first GPU address.
    pub offset: u64,
    /// Its page size.
    pub page_size: u32,
    /// How many pages it holds.
    pub pages: u64,
}

impl Region {
    /// The region as `VaRegion`'s 0x18 bytes.
    #[must_use]
    pub fn to_bytes(self) -> [u8; REGION_BYTES] {
        let mut bytes = Fixed::<REGION_BYTES>::from_input(&[]);
        bytes.set_u64::<REGION_OFFSET>(self.offset);
        bytes.set_u32::<REGION_PAGE_SIZE>(self.page_size);
        bytes.set_u64::<REGION_PAGES>(self.pages);
        *bytes.bytes()
    }
}

/// The two regions `layout` reports and how many count (PR `GetVARegionsImpl` over its two
/// allocators: the small pages' from `start >> 12` to `split >> 12`, the big pages' from
/// `split >> bits` to `end >> bits`; with no big pages, the second is zero and only one counts).
#[must_use]
pub fn regions(layout: &Layout) -> ([Region; 2], usize) {
    let small_shift = SMALL_PAGE_SIZE.trailing_zeros();
    let small = Region {
        offset: (layout.start >> small_shift) << small_shift,
        page_size: small_page_word(),
        pages: (layout.split >> small_shift).saturating_sub(layout.start >> small_shift),
    };
    let big_shift = layout.big_page_size.trailing_zeros();
    let big = Region {
        offset: (layout.split >> big_shift) << big_shift,
        page_size: layout.big_page_size,
        pages: (layout.end >> big_shift).saturating_sub(layout.split >> big_shift),
    };
    if big.pages == 0 {
        ([small, Region::default()], 1)
    } else {
        ([small, big], 2)
    }
}

/// [`SMALL_PAGE_SIZE`] as the `u32` a region holds.
fn small_page_word() -> u32 {
    u32::try_from(SMALL_PAGE_SIZE).unwrap_or_default()
}

/// An as-gpu fd's view: the address space and the layout it reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct View {
    /// gsv's address space.
    pub space: AsId,
    /// What the fd reports.
    pub layout: Layout,
}

/// Every as-gpu fd's view, and the layouts of the address spaces parentless `AllocAsEx`s made.
/// Plain data, saved with the nvsrv state; deserialising goes through `SavedAsGpu`, which
/// refuses a state `AllocAsEx` cannot reach ([`AsGpuStateError`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SavedAsGpu")]
pub struct AsGpu {
    views: BTreeMap<u32, View>,
    made: BTreeMap<AsId, Layout>,
}

/// The serialised form of [`AsGpu`], field for field.
#[derive(Deserialize)]
struct SavedAsGpu {
    views: BTreeMap<u32, View>,
    made: BTreeMap<AsId, Layout>,
}

/// Why a saved [`AsGpu`] could not be read back. The checks run in this order, the first that
/// fails reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AsGpuStateError {
    /// More address spaces made than gsv holds.
    #[error("{count} address spaces were made through nvhost-as-gpu, more than can be")]
    TooManySpaces {
        /// How many.
        count: usize,
    },
    /// A made address space's layout is not one `AllocAsEx` makes.
    #[error("address space {space} has a layout nvhost-as-gpu cannot make")]
    MadeLayout {
        /// The address space's id.
        space: u32,
    },
    /// A view's layout is neither its address space's own nor a view of it.
    #[error("fd {fd} has a view its address space cannot have")]
    ViewLayout {
        /// The fd.
        fd: u32,
    },
}

impl TryFrom<SavedAsGpu> for AsGpu {
    type Error = AsGpuStateError;

    fn try_from(saved: SavedAsGpu) -> Result<Self, Self::Error> {
        let SavedAsGpu { views, made } = saved;
        if made.len() > MAX_ADDRESS_SPACES {
            return Err(AsGpuStateError::TooManySpaces { count: made.len() });
        }
        if let Some(space) = made.iter().find(|(_, layout)| !layout.is_valid()) {
            return Err(AsGpuStateError::MadeLayout { space: space.0.0 });
        }
        let fits = |view: &View| match made.get(&view.space) {
            Some(own) if *own == view.layout => true,
            Some(own) => view.layout.is_view_of(own),
            None => view.layout.is_view_of(&NATIVE_LAYOUT),
        };
        if let Some((&fd, _)) = views.iter().find(|(_, view)| !fits(view)) {
            return Err(AsGpuStateError::ViewLayout { fd });
        }
        Ok(Self { views, made })
    }
}

impl AsGpu {
    /// No views.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The view of fd `fd`, if its `AllocAsEx` made one.
    #[must_use]
    pub fn view(&self, fd: u32) -> Option<View> {
        self.views.get(&fd).copied()
    }

    /// The fds that have views, in order.
    pub fn fds(&self) -> impl Iterator<Item = u32> + '_ {
        self.views.keys().copied()
    }

    /// The layout a parentless `AllocAsEx` gave address space `space`, if one made it.
    #[must_use]
    pub fn made(&self, space: AsId) -> Option<Layout> {
        self.made.get(&space).copied()
    }

    /// Fd `fd` was closed: its view goes; its address space stays (module docs).
    pub(super) fn close(&mut self, fd: u32) {
        self.views.remove(&fd);
    }

    /// An ioctl on as-gpu fd `fd`, or `None` for one the device does not know (module docs,
    /// "The ioctls").
    pub(super) fn ioctl(
        &mut self,
        fd: u32,
        call: &IoctlCall<'_>,
        mut gpu: GpuAccess<'_>,
    ) -> Option<IoctlAnswer> {
        let ioctl = call.ioctl;
        if ioctl.group() != GROUP {
            return None;
        }
        match (call.variant, ioctl.command()) {
            (IoctlVariant::Ioctl, command::ALLOC_AS_EX) => {
                let arg = Fixed::<ALLOC_AS_EX_BYTES>::from_input(call.input);
                let result = self.alloc_as_ex(fd, &arg, &mut gpu);
                Some(IoctlAnswer::with(result, arg.into_output()))
            }
            (IoctlVariant::Ioctl, command::GET_VA_REGIONS) => Some(self.va_regions(fd, call, false)),
            (IoctlVariant::Ioctl3, command::GET_VA_REGIONS) => Some(self.va_regions(fd, call, true)),
            (IoctlVariant::Ioctl, command::GET_SYNCPOINT_SHIM_INFO) => {
                let mut arg = Fixed::<SHIM_BYTES>::from_input(call.input);
                let result = self.shim(fd, &mut arg, &mut gpu);
                Some(IoctlAnswer::with(result, arg.into_output()))
            }
            _ => None,
        }
    }

    /// `AllocAsEx` (module docs, "The ioctls"). The argument comes back as it was sent.
    fn alloc_as_ex(
        &mut self,
        fd: u32,
        arg: &Fixed<ALLOC_AS_EX_BYTES>,
        gpu: &mut GpuAccess<'_>,
    ) -> NvResult {
        if self.views.contains_key(&fd) {
            return NvResult::InvalidState;
        }
        let flags = arg.u32_at::<ALLOC_FLAGS>();
        let as_fd = arg.u32_at::<ALLOC_AS_FD>().cast_signed();
        let requested_big = arg.u32_at::<ALLOC_BIG_PAGE_SIZE>();
        let reserved = arg.u32_at::<ALLOC_RESERVED>();
        let requested = Requested {
            start: arg.u64_at::<ALLOC_START>(),
            split: arg.u64_at::<ALLOC_SPLIT>(),
            end: arg.u64_at::<ALLOC_END>(),
        };
        let parent = if as_fd == 0 {
            None
        } else {
            let Ok(id) = u32::try_from(as_fd) else {
                return NvResult::BadParameter;
            };
            let space = AsId(id);
            if gpu.memory.table(space).is_none() {
                return NvResult::BadParameter;
            }
            let parent = self.made.get(&space).copied().unwrap_or(NATIVE_LAYOUT);
            if !is_supported_view(flags, requested_big, reserved, requested, &parent) {
                return NvResult::BadParameter;
            }
            Some(space)
        };
        let big = match big_page_size(requested_big) {
            0 => DEFAULT_BIG_PAGE_SIZE,
            big if is_supported_big_page_size(big) => big,
            _ => return NvResult::BadValue,
        };
        let mut layout = if requested.start == 0 {
            Layout {
                big_page_size: big,
                ..NATIVE_LAYOUT
            }
        } else {
            Layout {
                start: requested.start,
                split: requested.split,
                end: requested.end,
                big_page_size: big,
            }
        };
        // With a parent: PR's `vm.va_range_split = params.va_range_end`. Without one and with
        // split 0: conflict (module docs), the split is the end too.
        if parent.is_some() || (requested.start != 0 && requested.split == 0) {
            layout.split = layout.end;
        }
        if !layout.is_valid() {
            return NvResult::BadValue;
        }
        let space = if let Some(space) = parent {
            space
        } else {
            match gpu.memory.create_address_space(gpu.front) {
                Ok(space) => {
                    self.made.insert(space, layout);
                    space
                }
                Err(error) => {
                    tracing::warn!(%error, "nvsrv: no room for another GPU address space");
                    return NvResult::InsufficientMemory;
                }
            }
        };
        self.views.insert(fd, View { space, layout });
        NvResult::Success
    }

    /// `GetVARegions` through `Ioctl` or, if `inline`, `Ioctl3` (module docs, "The ioctls").
    fn va_regions(&self, fd: u32, call: &IoctlCall<'_>, inline: bool) -> IoctlAnswer {
        let mut arg = Fixed::<VA_REGIONS_BYTES>::from_input(call.input);
        let room = (call.inline_out_len / REGION_BYTES).min(2);
        let Some(view) = self.views.get(&fd) else {
            let zeros = if inline {
                vec![0; room * REGION_BYTES]
            } else {
                Vec::new()
            };
            return IoctlAnswer {
                result: NvResult::BadValue,
                output: arg.into_output(),
                inline: zeros,
            };
        };
        let (found, count) = regions(&view.layout);
        let [first, second] = found;
        let size = count * REGION_BYTES;
        arg.set_u32::<VA_BUF_SIZE>(u32::try_from(size).unwrap_or_default());
        arg.set_u64::<{ VA_REGION_0 + REGION_OFFSET }>(first.offset);
        arg.set_u32::<{ VA_REGION_0 + REGION_PAGE_SIZE }>(first.page_size);
        arg.set_u32::<{ VA_REGION_0 + REGION_PAGE_SIZE + 4 }>(0);
        arg.set_u64::<{ VA_REGION_0 + REGION_PAGES }>(first.pages);
        arg.set_u64::<{ VA_REGION_1 + REGION_OFFSET }>(second.offset);
        arg.set_u32::<{ VA_REGION_1 + REGION_PAGE_SIZE }>(second.page_size);
        arg.set_u32::<{ VA_REGION_1 + REGION_PAGE_SIZE + 4 }>(0);
        arg.set_u64::<{ VA_REGION_1 + REGION_PAGES }>(second.pages);
        let mut written = Vec::new();
        if inline {
            for (index, region) in found.into_iter().enumerate().take(room) {
                let region = if index < count {
                    region
                } else {
                    Region::default()
                };
                written.extend(region.to_bytes());
            }
        }
        IoctlAnswer {
            result: NvResult::Success,
            output: arg.into_output(),
            inline: written,
        }
    }

    /// `GetSyncpointShimInfo` (module docs, "The ioctls").
    fn shim(&self, fd: u32, arg: &mut Fixed<SHIM_BYTES>, gpu: &mut GpuAccess<'_>) -> NvResult {
        let (address, result) = match self.views.get(&fd) {
            None => (None, NvResult::NotSupported),
            Some(view) => match shim_page(view, gpu) {
                Ok(address) => (Some(address), NvResult::Success),
                Err(result) => (None, result),
            },
        };
        let (address, stride, count) =
            address.map_or((0, 0, 0), |at| (at, SHIM_STRIDE_BYTES, SYNCPOINT_COUNT));
        arg.set_u64::<SHIM_ADDRESS>(address);
        arg.set_u32::<SHIM_STRIDE>(stride);
        arg.set_u32::<SHIM_COUNT>(count);
        result
    }
}

/// The shim's page in `view`'s address space: the one placed already, or the page below the
/// view's split, placed now. `BadValue` if gsv does not hold the address space or refuses the
/// page: neither can follow from requests (`AllocAsEx` checked both), only from a restore whose
/// nvsrv and gsv states disagree.
fn shim_page(view: &View, gpu: &mut GpuAccess<'_>) -> Result<u64, NvResult> {
    if let Some(placed) = gpu.memory.shim(view.space) {
        return Ok(placed);
    }
    let address = view
        .layout
        .split
        .checked_sub(SMALL_PAGE_SIZE)
        .ok_or(NvResult::BadValue)?;
    gpu.memory
        .map_syncpoint_shim(view.space, address)
        .map_err(|error| {
            tracing::warn!(%error, space = view.space.0, "nvsrv: the shim could not be placed");
            NvResult::BadValue
        })?;
    Ok(address)
}
