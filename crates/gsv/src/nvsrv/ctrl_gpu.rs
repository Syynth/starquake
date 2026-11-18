//! `/dev/nvhost-ctrl-gpu`: the GPU's characteristics block, which Switch 2's NvRm reads before
//! it opens anything else, the replies that describe the GPU's layout (Z-cull, TPC masks, the
//! virtual SM map), and the device's two events (`docs/specs/graphics-server.md`, "Channels and
//! submission: nvsrv", the device table's ctrl-gpu row and the paragraph after the table; port
//! slice 10).
//!
//! Ported from Eden PR #4588 (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project) @
//! e5fb3b97: `eden-pr4588/src/core/hle/service/nvdrv/devices/native_gpu_characteristics.h`
//! (`MakeNativeGpuCharacteristics`, `NativeGpuSubcontextCount`, the two block sizes) and
//! `nvhost_ctrl_gpu.cpp` (`nvhost_ctrl_gpu::Ioctl1`'s `0x1` and `0x2` cases, `ZCullGetCtxSize`,
//! `ZCullGetInfo`; `nvhost_ctrl_gpu::Ioctl3`, the `0xc0e04705` and `0xc0e84705` case, the `0x6`
//! case and `GetTPCMasks3`, the `0x13` case (`VSMS_MAPPING`); the constructor and `QueryEvent`),
//! over the Switch 1 block both build on, Eden @ cf654052
//! `eden/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.h` (`IoctlGpuCharacteristics`, its
//! fields and offsets; `IoctlZcullGetCtxSize`, `IoctlNvgpuGpuZcullGetInfoArgs`,
//! `IoctlGpuGetTpcMasksArgs`) and `nvhost_ctrl_gpu.cpp` (`GetCharacteristics3`, the values), and
//! `eden/src/core/hle/service/nvdrv/devices/ioctl_serialization.h` (`WrapFixed`,
//! `WrapFixedInlOut`). "Row N" is a row of `eden-nx2/NX2_NOTES.md` @ 50fa86bb; "NX2" is
//! `eden-nx2/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.cpp` @ 50fa86bb
//! (`ApplyNx2Characteristics`, `ApplyNx2CharacteristicsTail`, and its `Ioctl1`, `Ioctl3`,
//! `ZCullGetCtxSize`, `ZCullGetInfo`, `GetTPCMasks3` and `QueryEvent`, compared with PR's for
//! each ioctl below). libnx (`libnx/nx/source/nvidia/ioctl/nvhost-ctrl-gpu.c` @ b156b477) sends
//! the two Z-cull numbers: `_NV_IOR(0x47, 0x01, ..)` and `_NV_IOR(0x47, 0x02, ..)`.
//!
//! # The ioctls
//!
//! | Variant | Ioctl | Cmd | What |
//! |---|---|---|---|
//! | `Ioctl3` | [`CHARACTERISTICS`], [`CHARACTERISTICS_EXTENDED`] | 0x5 | the block (below) |
//! | `Ioctl` | any number with group `'G'` and command [`ZCULL_GET_CTX_SIZE`] | 0x1 | `ZCullGetCtxSize` |
//! | `Ioctl` | any number with group `'G'` and command [`ZCULL_GET_INFO`] | 0x2 | `ZCullGetInfo` |
//! | `Ioctl3` | any number with group `'G'` and command [`GET_TPC_MASKS`] | 0x6 | `GetTPCMasks3` |
//! | `Ioctl3` | group `'G'` and command [`VSMS_MAPPING`], only [`VSMS_MAPPING_IOCTL`] answered | 0x13 | `VSMS_MAPPING` (PR's only) |
//!
//! The commands are matched on the group and the command alone, as PR's `switch (command.group)`
//! then `switch (command.cmd)` does, except where PR checks the whole number (the block's two
//! and `VSMS_MAPPING`). The title's run sends `0x80044701`, `0x80284702`, `0xc0184706` and
//! `0xc0084713` (issue #300's table).
//!
//! **`GetCharacteristics`.** `Ioctl3` (12) with ioctl [`CHARACTERISTICS`] (0xD0-byte block) or
//! [`CHARACTERISTICS_EXTENDED`] (0xD8): group `'G'` ([`GROUP`]), command 5, both directions,
//! PR's two exact numbers. The argument is a 16-byte header (`u64` size at +0, `u64` buffer
//! address at +8) followed by the block: [`HEADER_BYTES`] + the block's size, the "packet".
//!
//! *Checks*, PR's one: the in buffer or out buffer 0 shorter than the packet, or out buffer 1
//! (the inline output) shorter than the block, is `BadParameter`, and nothing is written.
//!
//! *Answer.* Out buffer 0 gets the packet: the block's size as a `u64` at +0, the caller's
//! buffer address (in bytes 8 to 16) as it sent it, then the block. Out buffer 1 gets the block.
//! (PR zeroes both buffers first and copies the header, then writes the size over its first
//! eight bytes.)
//!
//! **`ZCullGetCtxSize`** (`WrapFixed` over `IoctlZcullGetCtxSize`, 4 bytes): `Success`, and the
//! struct back with its size [`zcull::CTX_SIZE`]. **`ZCullGetInfo`** (`WrapFixed` over
//! `IoctlNvgpuGpuZcullGetInfoArgs`, 40 bytes, [`zcull::INFO_BYTES`]): `Success`, and the ten
//! words of [`zcull::info`]. Neither reads its argument: a short or long in buffer changes
//! nothing (`WrapFixed` reads what there is and fills the rest with zeros, all overwritten).
//!
//! **`GetTPCMasks3`** (`WrapFixedInlOut` over `IoctlGpuGetTpcMasksArgs`, 24 bytes, and `u32`
//! inline items): the struct as sent (zeros past a short input); if its mask buffer size (+0) is
//! not zero, its mask (+16) is set to the GPC's TPC mask ([`topology::TPC_MASK`]). Then the
//! struct goes back whole, and out buffer 1's first word gets the struct's mask, whether set or
//! as sent, if out buffer 1 holds a whole word (`tpc_mask.front() = params.tcp_mask`, and
//! `WrapGeneric` makes `inline_output.size() / 4` items). `Success` always.
//!
//! **`VSMS_MAPPING`** (PR's only, `Ioctl3`): the number other than [`VSMS_MAPPING_IOCTL`], an in
//! buffer or out buffer 0 other than exactly [`VSMS_ARGUMENT_BYTES`], or an out buffer 1 shorter
//! than two bytes per SM ([`topology::SM_COUNT`]), is `BadParameter`, and nothing is written (PR
//! checks all four in one condition, so their order is not observable). Otherwise out buffer 0
//! gets the argument as sent and out buffer 1 one `{gpc, tpc}` byte pair per virtual SM
//! ([`topology::vsm_location`]), nothing past them. `Success`.
//!
//! Every other ctrl-gpu ioctl, on every variant, goes to the server's unknown-ioctl policy
//! (see "Not built").
//!
//! # The GPU's layout, one source
//!
//! The replies above describe the same GPU the block does: one GPC ([`value::NUM_GPC`]) of two
//! TPCs ([`value::TPC_PER_GPC`]), each of two SMs ([`topology::SMS_PER_TPC`]: PR's
//! `VSMS_MAPPING` comment, "The GPU reports one GPC with two TPCs of two SMs each (see
//! GetCharacteristics1/GetTPCMasks3)", read). [`topology`] derives the TPC mask, the SM count and
//! the SM map from those counts, and [`zcull::info`] derives its per-TPC and per-GPC fields from
//! them, so the replies cannot disagree with the block. The derivations reproduce the
//! references' constants for the block's counts (PR and NX2 both answer `tcp_mask = 3`, four SM
//! pairs `{0, i / 2}` in PR, and the Z-cull values below), each pinned by its golden test, and
//! `the_layout_replies_agree_with_the_block` checks them against the counts read back from the
//! block's bytes. **Guess:** which Z-cull fields scale with the TPC count (the two width
//! alignments, 16 pixels per TPC) and which with the GPC count (the region byte multiplier and
//! the two header sizes, per GPC) is not in any pinned reference: it is read from the field
//! names and the references' values for one GPC and two TPCs. A console's reply, or NvRm's use
//! of the fields in a trace, would confirm it; for the block's counts it changes no byte.
//!
//! # Events
//!
//! `QueryEvent` on a ctrl-gpu fd answers id 1 ([`event::ERROR_NOTIFIER`]) and id 2
//! ([`event::UNKNOWN`]), PR's and NX2's `QueryEvent` alike (`error_notifier_event`,
//! `unknown_event`), with a copied handle and `Success`, the same event each time it is asked;
//! any other id is `BadParameter`, as before. The two events are made when the fd is opened, as
//! the spec says ("ctrl-gpu events made at open") and as both references' constructors make them
//! (`events_interface.CreateEvent("CtrlGpuErrorNotifier")`, `("CtrlGpuUnknownEvent")`): nvsrv's
//! `Open` makes them before it opens the fd ([`CtrlGpu`], [`super::NvsrvState::open`]). Nothing
//! signals them, here as in both references. They are saved with nvsrv's state and dropped when
//! the fd closes; the kernel events themselves are not freed (see "Where this differs"). The
//! owner's decision of 2026-11-17 deferred these two until a title sent one ("Defer as-gpu
//! `ModifyKind` and `Remap`, and ctrl-gpu `QueryEvent` 1 and 2, until a title sends them");
//! the title of issue #300 sends `QueryEvent` 2.
//!
//! # The block
//!
//! [`characteristics`] builds it, as PR's `MakeNativeGpuCharacteristics` does: the first 0x70
//! bytes of Eden's Switch 1 block (`GetCharacteristics3`'s values, at `IoctlGpuCharacteristics`'s
//! offsets), then PR's changes, then the nine interface levels at +0x70 and the subcontext count
//! at +0xB4; everything else is zero. Offsets are in [`layout`], values in [`value`].
//!
//! | Offset | Field | 0xD0 | 0xD8 | Source |
//! |---|---|---|---|---|
//! | 0x00 | architecture | 0x170 | 0x170 | row 24 (observed): chip 0x17E; PR |
//! | 0x04 | implementation | 0xE | 0xE | row 24; PR |
//! | 0x08–0x27 | revision, GPCs, L2 size, video memory, TPCs per GPC, bus type | Eden's | Eden's | PR copies the Switch 1 prefix |
//! | 0x28 | big page size | 0x1000 | 0x1000 | PR: "normal mappings support 4 KiB CPU/GPU aliases" |
//! | 0x2C–0x33 | compression page size, PDE coverage bits | Eden's | Eden's | PR copies them |
//! | 0x34 | available big page sizes | 0x11000 | 0x11000 | PR: Eden's 0x30000 masked to 0x10000, or 0x1000 |
//! | 0x38 | GPC mask | Eden's | Eden's | PR copies it |
//! | 0x3C, 0x40 | SM and SPA versions | 0x807 | 0x808 | PR |
//! | 0x44–0x4F | warp count, VA bits (0x28: 40), reserved | Eden's | Eden's | PR copies them |
//! | 0x50 | flags | Eden's 0x55, and bits 8, 22, 26, 34, 35, 40–43 | the same | 8 and 22: row 45; 26: row 46; 34: row 149 (observed); 35 and 40–43: PR (read) |
//! | 0x58–0x6F | 2D, 3D, compute, GPFIFO, inline-to-memory, DMA copy classes | Eden's (Maxwell) | Eden's | PR copies them |
//! | 0x70 | nine `s16` interface levels | 0x1C, 0xB, -1, 0x1D, 0x14, 0, 0, 0, 0 | the fourth 0x1F | values PR's; the level at +0x78 at least 6, NX2's comment in `ApplyNx2Characteristics` (row 24 says more, see "Conflicts") |
//! | 0xB4 | subcontexts per TSG (`u32`) | 64 | 64 | row 45 (observed); PR `NativeGpuSubcontextCount` |
//!
//! The 0xD0 block is pinned byte for byte by `the_block_s_golden_bytes`, and each listed bit by
//! `every_listed_flag_bit_is_set` (in `ctrl_gpu/tests.rs`).
//!
//! # Conflicts
//!
//! NX2 answers the same ioctl from Eden's Switch 1 path with its own changes
//! (`ApplyNx2Characteristics`): Ampere classes at +0x5C to +0x6C (0xC797, 0xC7C0, 0xC56F,
//! 0xC7B5), six TPCs per GPC, SM version 0x808 in the 0xD0 block, the chip name `"ga10f"` at
//! +0x90, the Switch 1 page sizes at +0x28 and +0x34, all nine interface levels 0x40, and
//! neither bit 35 nor bits 40–43. The spec takes PR's whole layout ("full layout read (PR
//! `native_gpu_characteristics.h`)"), by decision 8's leg for Nvidia's headers and readings that
//! cite the system module for layouts: every one of those bytes is PR's here, and the golden
//! bytes pin them. Open question, for the spec's verification item 1 (it does not list these
//! yet): which of these fields does NvRm or NVN read, and does a title behave differently with
//! NX2's values? A trace of the reads that follow the ioctl settles it.
//!
//! **The interface levels.** Row 24 (observed) says the `s16` levels "at 0x70..0x80 must be >= 6
//! etc.", all of them. NX2's own comment in `ApplyNx2Characteristics` is narrower: NvRm
//! "requires e.g. [0x78] >= 6 and enables features above 0xa, 0x1f and 0x26". PR's levels,
//! used here, meet the narrower reading (0x14 at +0x78, pinned by
//! `levels_and_subcontexts_are_where_the_sdk_reads_them`) but not row 24's: -1 at +0x74 and 0
//! from +0x7A to +0x80 are below 6. And PR's channel level at +0x76 in the 0xD0 block, 0x1D, is
//! below NX2's 0x1F feature threshold (the 0xD8 block's 0x1F is not above it either; NX2 does
//! not say which level each threshold is for), where NX2 reports 0x40 everywhere. Open question: which levels does NvRm read, and which features
//! does it gate on each? A trace of NvRm's reads of the block, or a title that fails with PR's
//! levels and runs with NX2's, settles it.
//!
//! **`VSMS_MAPPING`.** PR answers it (above); NX2's `Ioctl3` answers `case 0x13` with
//! `NotImplemented` ("(STUBBED)", before its Switch 2 stub could answer it). PR's is used, by
//! decision 8's leg for a reading of the system module's meaning (PR's comment says what the SDK
//! sends and treats as failure) and because PR's answer contains NX2's refusal. Open question:
//! does NvRm go on without the map? A trace of the title's next request settles it. Pinned by
//! `vsms_mapping_s_golden_bytes`.
//!
//! The other ioctls and the events are the same in PR and NX2 (`ZCullGetCtxSize`,
//! `ZCullGetInfo`, `GetTPCMasks3`, `QueryEvent`, the constructor), and Eden's too. NX2's TPC mask
//! (3, two TPCs) disagrees with NX2's own block (six TPCs per GPC); here both follow PR's block.
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `ctrl_gpu/tests.rs` unless another file is given:
//!
//! | Case | PR | Here |
//! |---|---|---|
//! | Out buffer 0 past the packet, out buffer 1 past the block | zeroed | left as they were (`only_the_packet_and_the_block_are_written`), as nvsrv writes every answer (`crate::nvsrv`, "Where this differs from the references") |
//! | A refused request's buffers | written with whatever the server's scratch buffers held | nothing written (`a_short_buffer_is_a_bad_parameter`) |
//! | The bootstrap's GPU profile (`ApplyNativeGpuProfile` in `nvdrv_interface.cpp`'s `Ioctl3`, which sets the SM versions and the fourth level from root 4's flags) | applied | not applied: nvsrv does not see the root's bootstrap yet (see "Not built"); a title bootstrapped with flags 6 or 7 reads 0x807 from the 0xD0 block (`the_block_s_golden_bytes`) |
//! | `GetTPCMasks3`'s out buffer 1 past its first word | zeroed, whole words (`WrapGeneric` copies `inline_output.size() / 4` items, all zero but the first) | left as it was: only the mask word is written, so a title cannot make the server build a reply as large as its buffer (`tpc_masks_write_one_word`) |
//! | A `VSMS_MAPPING` refusal's buffers | written with whatever the server's scratch buffers held | nothing written (`vsms_mapping_refuses_each_size_and_number`) |
//! | A ctrl-gpu fd's kernel events when it closes | freed (`~nvhost_ctrl_gpu`: `events_interface.FreeEvent` of each) | not freed: the services framework has no call that frees an event, so each open and close of ctrl-gpu leaves the two kernel events the server made, with no bound over the run, as nvhost-gpu's channels do (`super::gpu`, "Where this differs"; `a_closed_ctrl_gpu_s_events_are_not_freed`) |
//! | A ctrl-gpu `Open` whose second event cannot be made | opened, the device holding a null `unknown_event` (`ServiceContext::CreateEvent` in `kernel_helpers.cpp` logs and returns null when the event cannot be reserved or made; `nvhost_ctrl_gpu`'s constructor keeps what it returns) | `Open` fails with the kernel's error and uses no fd; the first event, made already, stays in the kernel with no owner, since the services framework has no call that frees an event (`a_ctrl_gpu_open_whose_second_event_fails_leaves_the_first`, in `nvsrv/tests.rs`) |
//!
//! # What is saved
//!
//! The events of each open ctrl-gpu fd ([`CtrlGpu`]) are saved with nvsrv's state. Restoring
//! checks the fds one by one in fd order, the first failing reported ([`CtrlGpuStateError`]);
//! within an fd, that its two events are not one event before that neither is a lower fd's
//! (`a_saved_ctrl_gpu_table_open_cannot_make_is_refused`). Then, with the open files
//! ([`super::NvsrvStateError`]), it checks that each fd with events is an open ctrl-gpu and that
//! each open ctrl-gpu has its events (made at open, so none lacks them). `Open` refuses a pair
//! that fails the first two checks ([`super::NvsrvState::open`]), so whatever is saved restores.
//!
//! Not checked here: that an event is one the kernel holds, and that a ctrl-gpu event is not
//! also an nvhost-gpu channel's event (the kernel makes each event fresh, so neither happens in a
//! run; a forged snapshot that shares one makes the two devices' queries hand out one event,
//! which nothing signals).
//!
//! # Not built
//!
//! - The Switch 1 forms of `GetCharacteristics` (`Ioctl1` command 5, and `Ioctl3` command 5
//!   with any other number), `GetTPCMasks1` (`Ioctl1` command 6), the ZBC ioctls, `FlushL2`,
//!   `GetGpuTime`, `GetActiveSlotMask`, `PmuGetGpuLoad`, and NX2's `GetEngineInfo`: no Switch 2
//!   row needs them yet. They go to the unknown-ioctl policy.
//! - The bootstrap's GPU profile (above): it needs root 4's flags carried to nvsrv, which is the
//!   wiring slice's (16).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use starquake_kernel::id::ObjectId;

use super::gpu::Queried;
use super::ioctl::{Fixed, Ioctl};
use super::{IoctlAnswer, IoctlCall, NvResult};

#[cfg(test)]
mod tests;

/// ctrl-gpu's ioctl group, `'G'` (`nvhost_ctrl_gpu::Ioctl3`: `case 'G':`).
pub const GROUP: u32 = 0x47;

/// `ZCullGetCtxSize`'s command in the group (`Ioctl1`: `case 0x1:`).
pub const ZCULL_GET_CTX_SIZE: u32 = 0x1;

/// `ZCullGetInfo`'s command (`Ioctl1`: `case 0x2:`).
pub const ZCULL_GET_INFO: u32 = 0x2;

/// `GetCharacteristics`'s command in the group (`case 0x5:`).
pub const GET_CHARACTERISTICS: u32 = 0x5;

/// `GetTPCMasks3`'s command (`Ioctl3`: `case 0x6:`).
pub const GET_TPC_MASKS: u32 = 0x6;

/// `VSMS_MAPPING`'s command (`Ioctl3`: `case 0x13:`; PR's comment names it
/// `NVGPU_GPU_IOCTL_VSMS_MAPPING`).
pub const VSMS_MAPPING: u32 = 0x13;

/// The one `VSMS_MAPPING` number PR answers: `command.raw != 0xc0084713` is `BadParameter`
/// (both directions, an 8-byte argument).
pub const VSMS_MAPPING_IOCTL: u32 = 0xC008_4713;

/// `VSMS_MAPPING`'s argument, which PR requires the in buffer and out buffer 0 to be exactly
/// (`input.size() != 8 || output.size() != 8`): "the 8-byte argument struct".
pub const VSMS_ARGUMENT_BYTES: usize = 8;

/// Bytes of out buffer 1 per virtual SM: a `{gpc_index, tpc_index}` byte pair (PR's comment, and
/// `inline_output.size() < sm_count * 2`).
pub const VSM_PAIR_BYTES: usize = 2;

/// `IoctlGpuGetTpcMasksArgs`, 24 bytes (`static_assert(sizeof(IoctlGpuGetTpcMasksArgs) == 24)`):
/// `u32 mask_buffer_size`, a padding word, `u64 mask_buffer_address`, `u32 tcp_mask`, a padding
/// word.
pub mod tpc_masks {
    /// The struct's size.
    pub const BYTES: usize = 24;
    /// `mask_buffer_size`.
    pub const BUFFER_SIZE: usize = 0;
    /// `mask_buffer_address`, a `u64` nothing reads.
    pub const BUFFER_ADDRESS: usize = 8;
    /// `tcp_mask`.
    pub const MASK: usize = 16;
    /// The bytes of one inline item, a `u32` (`std::span<u32> tpc_mask`).
    pub const ITEM_BYTES: usize = 4;
}

/// The GPU's layout as the block reports it, and what the layout replies derive from it (module
/// docs, "The GPU's layout, one source").
pub mod topology {
    use super::value::{NUM_GPC, TPC_PER_GPC};

    /// SMs per TPC: PR's `VSMS_MAPPING` comment, "one GPC with two TPCs of two SMs each" (read).
    pub const SMS_PER_TPC: u32 = 2;

    /// Every TPC of every GPC.
    pub const TPC_COUNT: u32 = NUM_GPC * TPC_PER_GPC;

    /// Every SM: PR's `constexpr size_t sm_count = 4`.
    pub const SM_COUNT: u32 = TPC_COUNT * SMS_PER_TPC;

    /// One GPC's TPC mask, a bit per TPC: PR's and NX2's `params.tcp_mask = 3`.
    pub const TPC_MASK: u32 = (1 << TPC_PER_GPC) - 1;

    /// The GPC and TPC of virtual SM `vsm`, SMs numbered TPC by TPC and GPC by GPC: PR's
    /// `inline_output[i * 2] = 0` and `inline_output[i * 2 + 1] = i / 2` for its one GPC.
    #[must_use]
    pub const fn vsm_location(vsm: u32) -> (u32, u32) {
        let per_gpc = TPC_PER_GPC * SMS_PER_TPC;
        (vsm / per_gpc, (vsm / SMS_PER_TPC) % TPC_PER_GPC)
    }
}

/// `ZCullGetCtxSize` and `ZCullGetInfo` (Eden's `ZCullGetCtxSize` and `ZCullGetInfo`, the same in
/// PR and NX2).
pub mod zcull {
    use super::topology::TPC_COUNT;
    use super::value::NUM_GPC;

    /// `IoctlZcullGetCtxSize`: one `u32`.
    pub const CTX_SIZE_BYTES: usize = 4;
    /// `params.size = 0x1`.
    pub const CTX_SIZE: u32 = 0x1;

    /// `IoctlNvgpuGpuZcullGetInfoArgs`: ten `u32`s, 40 bytes.
    pub const INFO_BYTES: usize = 40;

    /// The pixels of width alignment per TPC: Eden's `width_align_pixels = 0x20` and
    /// `subregion_width_align_pixels = 0x20` over two TPCs (the scaling is a **guess**, module
    /// docs).
    pub const WIDTH_ALIGN_PER_TPC: u32 = 0x10;
    /// `height_align_pixels = 0x20`.
    pub const HEIGHT_ALIGN_PIXELS: u32 = 0x20;
    /// `pixel_squares_by_aliquots = 0x400`.
    pub const PIXEL_SQUARES_BY_ALIQUOTS: u32 = 0x400;
    /// `aliquot_total = 0x800`.
    pub const ALIQUOT_TOTAL: u32 = 0x800;
    /// `region_byte_multiplier = 0x20` for one GPC (per GPC: a **guess**).
    pub const REGION_BYTE_MULTIPLIER_PER_GPC: u32 = 0x20;
    /// `region_header_size = 0x20` for one GPC (per GPC: a **guess**).
    pub const REGION_HEADER_PER_GPC: u32 = 0x20;
    /// `subregion_header_size = 0xc0` for one GPC (per GPC: a **guess**).
    pub const SUBREGION_HEADER_PER_GPC: u32 = 0xC0;
    /// `subregion_height_align_pixels = 0x40`.
    pub const SUBREGION_HEIGHT_ALIGN_PIXELS: u32 = 0x40;
    /// `subregion_count = 0x10`.
    pub const SUBREGION_COUNT: u32 = 0x10;

    /// The ten words, in the struct's order: `width_align_pixels`, `height_align_pixels`,
    /// `pixel_squares_by_aliquots`, `aliquot_total`, `region_byte_multiplier`,
    /// `region_header_size`, `subregion_header_size`, `subregion_width_align_pixels`,
    /// `subregion_height_align_pixels`, `subregion_count`.
    #[must_use]
    pub const fn info() -> [u32; 10] {
        [
            WIDTH_ALIGN_PER_TPC * TPC_COUNT,
            HEIGHT_ALIGN_PIXELS,
            PIXEL_SQUARES_BY_ALIQUOTS,
            ALIQUOT_TOTAL,
            REGION_BYTE_MULTIPLIER_PER_GPC * NUM_GPC,
            REGION_HEADER_PER_GPC * NUM_GPC,
            SUBREGION_HEADER_PER_GPC * NUM_GPC,
            WIDTH_ALIGN_PER_TPC * TPC_COUNT,
            SUBREGION_HEIGHT_ALIGN_PIXELS,
            SUBREGION_COUNT,
        ]
    }
}

/// The events `QueryEvent` answers on a ctrl-gpu fd (PR's and NX2's `QueryEvent`).
pub mod event {
    /// `case 1: return error_notifier_event;`.
    pub const ERROR_NOTIFIER: u32 = 1;
    /// `case 2: return unknown_event;`.
    pub const UNKNOWN: u32 = 2;
}

/// `GetCharacteristics` for the 0xD0-byte block: PR's `command.raw == 0xc0e04705` (both
/// directions, a 0xE0-byte argument: the header and the block).
pub const CHARACTERISTICS: u32 = 0xC0E0_4705;

/// `GetCharacteristics` for the 0xD8-byte block: PR's `command.raw == 0xc0e84705`.
pub const CHARACTERISTICS_EXTENDED: u32 = 0xC0E8_4705;

/// The header before the block in the argument: a `u64` size and a `u64` buffer address (Eden
/// `IoctlCharacteristics`; PR `packet_size = 0x10 + native_size`).
pub const HEADER_BYTES: usize = 0x10;

/// The block's size (PR `NativeGpuCharacteristicsSize = 0xd0`; row 24, observed).
pub const BLOCK_BYTES: usize = 0xD0;

/// The extended block's size (PR `ExtendedNativeGpuCharacteristicsSize = 0xd8`).
pub const EXTENDED_BLOCK_BYTES: usize = 0xD8;

/// Subcontexts (VEIDs) per timeslice group: PR `NativeGpuSubcontextCount = 64`; row 45
/// (observed: "64 subcontexts at characteristics +0xb4"). The TSG device bounds a binding's
/// subcontext id by it too ([`super::tsg_gpu`]).
pub const SUBCONTEXT_COUNT: u32 = 64;

/// Which block a request asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockSize {
    /// The 0xD0-byte block ([`CHARACTERISTICS`]).
    Native,
    /// The 0xD8-byte block ([`CHARACTERISTICS_EXTENDED`]): PR's newer shader profile.
    Extended,
}

impl BlockSize {
    /// The block size `ioctl` asks for, if it is one of PR's two numbers.
    #[must_use]
    pub const fn of(ioctl: Ioctl) -> Option<Self> {
        match ioctl.raw() {
            CHARACTERISTICS => Some(Self::Native),
            CHARACTERISTICS_EXTENDED => Some(Self::Extended),
            _ => None,
        }
    }

    /// The block's size in bytes.
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            Self::Native => BLOCK_BYTES,
            Self::Extended => EXTENDED_BLOCK_BYTES,
        }
    }
}

/// Where each field is in the block (Eden `IoctlGpuCharacteristics` for +0x00 to +0x70, PR
/// `MakeNativeGpuCharacteristics` for the rest).
pub mod layout {
    /// `arch`.
    pub const ARCH: usize = 0x00;
    /// `impl`.
    pub const IMPL: usize = 0x04;
    /// `rev`.
    pub const REV: usize = 0x08;
    /// `num_gpc`.
    pub const NUM_GPC: usize = 0x0C;
    /// `l2_cache_size`, a `u64`.
    pub const L2_CACHE_SIZE: usize = 0x10;
    /// `on_board_video_memory_size`, a `u64`.
    pub const VIDEO_MEMORY_SIZE: usize = 0x18;
    /// `num_tpc_per_gpc`.
    pub const TPC_PER_GPC: usize = 0x20;
    /// `bus_type`.
    pub const BUS_TYPE: usize = 0x24;
    /// `big_page_size`: PR's preferred page size.
    pub const BIG_PAGE_SIZE: usize = 0x28;
    /// `compression_page_size`.
    pub const COMPRESSION_PAGE_SIZE: usize = 0x2C;
    /// `pde_coverage_bit_count`.
    pub const PDE_COVERAGE_BITS: usize = 0x30;
    /// `available_big_page_sizes`.
    pub const AVAILABLE_BIG_PAGE_SIZES: usize = 0x34;
    /// `gpc_mask`.
    pub const GPC_MASK: usize = 0x38;
    /// `sm_arch_sm_version`.
    pub const SM_VERSION: usize = 0x3C;
    /// `sm_arch_spa_version`.
    pub const SPA_VERSION: usize = 0x40;
    /// `sm_arch_warp_count`.
    pub const WARP_COUNT: usize = 0x44;
    /// `gpu_va_bit_count`.
    pub const VA_BITS: usize = 0x48;
    /// `flags`, a `u64`.
    pub const FLAGS: usize = 0x50;
    /// `twod_class`.
    pub const TWOD_CLASS: usize = 0x58;
    /// `threed_class`.
    pub const THREED_CLASS: usize = 0x5C;
    /// `compute_class`.
    pub const COMPUTE_CLASS: usize = 0x60;
    /// `gpfifo_class`.
    pub const GPFIFO_CLASS: usize = 0x64;
    /// `inline_to_memory_class`.
    pub const INLINE_TO_MEMORY_CLASS: usize = 0x68;
    /// `dma_copy_class`.
    pub const DMA_COPY_CLASS: usize = 0x6C;
    /// The first of the nine `s16` interface levels (PR: `result.data() + 0x70 + i * 2`); where
    /// Eden's Switch 1 block has its topology fields.
    pub const LEVELS: usize = 0x70;
    /// The channel interface level, the fourth (PR `ApplyNativeGpuProfile`'s `0x76`).
    pub const CHANNEL_LEVEL: usize = 0x76;
    /// The subcontext count, a `u32` (PR `result.data() + 0xb4`).
    pub const SUBCONTEXTS: usize = 0xB4;
}

/// The values the block holds (Eden `GetCharacteristics3` and PR `MakeNativeGpuCharacteristics`).
pub mod value {
    /// The architecture: PR `native_arch{0x170}`; row 24 (chip 0x17E, the architecture plus the
    /// implementation).
    pub const ARCH: u32 = 0x170;
    /// The implementation: PR `native_impl{0xe}`.
    pub const IMPL: u32 = 0xE;
    /// Eden `params.gc.rev = 0xa1`.
    pub const REV: u32 = 0xA1;
    /// Eden `num_gpc = 0x1`.
    pub const NUM_GPC: u32 = 0x1;
    /// Eden `l2_cache_size = 0x40000`.
    pub const L2_CACHE_SIZE: u64 = 0x4_0000;
    /// Eden `num_tpc_per_gpc = 0x2`.
    pub const TPC_PER_GPC: u32 = 0x2;
    /// Eden `bus_type = 0x20`.
    pub const BUS_TYPE: u32 = 0x20;
    /// PR `preferred_page_size{0x1000}`.
    pub const BIG_PAGE_SIZE: u32 = 0x1000;
    /// Eden `compression_page_size = 0x20000`.
    pub const COMPRESSION_PAGE_SIZE: u32 = 0x2_0000;
    /// Eden `pde_coverage_bit_count = 0x1B`.
    pub const PDE_COVERAGE_BITS: u32 = 0x1B;
    /// Eden `available_big_page_sizes = 0x30000`, before PR's mask.
    pub const SWITCH_1_BIG_PAGE_SIZES: u32 = 0x3_0000;
    /// The part of it PR keeps (`& 0x10000`).
    pub const KEPT_BIG_PAGE_SIZES: u32 = 0x1_0000;
    /// Eden `gpc_mask = 0x1`.
    pub const GPC_MASK: u32 = 0x1;
    /// PR `native_sm_version`: 0x807 for the 0xD0 block.
    pub const SM_VERSION: u32 = 0x807;
    /// PR `native_sm_version`: 0x808 for the 0xD8 block.
    pub const EXTENDED_SM_VERSION: u32 = 0x808;
    /// Eden `sm_arch_warp_count = 0x80`.
    pub const WARP_COUNT: u32 = 0x80;
    /// Eden `gpu_va_bit_count = 0x28`: a 40-bit GPU address space (nvhost-as-gpu bounds its
    /// ranges by it, [`super::super::as_gpu::VA_LIMIT`]).
    pub const VA_BITS: u32 = 0x28;
    /// Eden `flags = 0x55`, before PR's bits.
    pub const SWITCH_1_FLAGS: u64 = 0x55;
    /// Eden `twod_class = 0x902D`.
    pub const TWOD_CLASS: u32 = 0x902D;
    /// Eden `threed_class = 0xB197`.
    pub const THREED_CLASS: u32 = 0xB197;
    /// Eden `compute_class = 0xB1C0`.
    pub const COMPUTE_CLASS: u32 = 0xB1C0;
    /// Eden `gpfifo_class = 0xB06F`.
    pub const GPFIFO_CLASS: u32 = 0xB06F;
    /// Eden `inline_to_memory_class = 0xA140`.
    pub const INLINE_TO_MEMORY_CLASS: u32 = 0xA140;
    /// Eden `dma_copy_class = 0xB0B5`.
    pub const DMA_COPY_CLASS: u32 = 0xB0B5;
    /// The nine interface levels of the 0xD0 block, in order (PR `levels`): GPU control 0x1C,
    /// TSG 0xB, the debugger -1 (absent), channels 0x1D, address spaces 0x14, then four zeros.
    pub const LEVELS: [i16; 9] = [0x1C, 0xB, -1, 0x1D, 0x14, 0, 0, 0, 0];
    /// The channel level of the 0xD8 block (PR: `Size == ExtendedNativeGpuCharacteristicsSize ?
    /// 0x1f : 0x1d`).
    pub const EXTENDED_CHANNEL_LEVEL: i16 = 0x1F;
}

/// The flag bits PR sets over Eden's (`flags | (1ULL << 8) | (1ULL << 22) | (1ULL << 26) |
/// (1ULL << 34) | (1ULL << 35) | (0xfULL << 40)`).
pub mod flag {
    /// Bit 8, timeslice groups (row 45, observed; NX2: `NVGPU_GPU_FLAGS_SUPPORT_TSG`).
    pub const TSG: u64 = 1 << 8;
    /// Bit 22, TSG subcontexts (row 45; NX2: `NVGPU_GPU_FLAGS_SUPPORT_TSG_SUBCONTEXTS`).
    pub const TSG_SUBCONTEXTS: u64 = 1 << 22;
    /// Bit 26, the syncpoint shim (row 46, observed; NX2:
    /// `NVGPU_GPU_FLAGS_SUPPORT_SYNCPOINT_ADDRESS`).
    pub const SYNCPOINT_ADDRESS: u64 = 1 << 26;
    /// Bit 34, mapping modify (row 149, observed: without it compressible textures get GPU
    /// address 0; PR: "The native SDK gates `ModifyKind` (AS ioctl 12) on capability bit 34").
    pub const MAPPING_MODIFY: u64 = 1 << 34;
    /// Bit 35: PR sets it with no comment of its own (read; its meaning is not known).
    pub const BIT_35: u64 = 1 << 35;
    /// Bits 40 to 43: PR "Native SDK class enumeration filters 2D/3D/compute/inline families
    /// with bits 40..43" (read).
    pub const CLASS_FAMILIES: u64 = 0xF << 40;
    /// Every bit PR adds.
    pub const ADDED: u64 =
        TSG | TSG_SUBCONTEXTS | SYNCPOINT_ADDRESS | MAPPING_MODIFY | BIT_35 | CLASS_FAMILIES;
}

/// The largest block, which [`characteristics`] fills and then cuts to the size asked for.
type Block = Fixed<EXTENDED_BLOCK_BYTES>;

/// The block of `size` (module docs, "The block").
#[must_use]
pub fn characteristics(size: BlockSize) -> Vec<u8> {
    let mut block = Block::from_input(&[]);
    block.set_u32::<{ layout::ARCH }>(value::ARCH);
    block.set_u32::<{ layout::IMPL }>(value::IMPL);
    block.set_u32::<{ layout::REV }>(value::REV);
    block.set_u32::<{ layout::NUM_GPC }>(value::NUM_GPC);
    block.set_u64::<{ layout::L2_CACHE_SIZE }>(value::L2_CACHE_SIZE);
    block.set_u64::<{ layout::VIDEO_MEMORY_SIZE }>(0);
    block.set_u32::<{ layout::TPC_PER_GPC }>(value::TPC_PER_GPC);
    block.set_u32::<{ layout::BUS_TYPE }>(value::BUS_TYPE);
    block.set_u32::<{ layout::BIG_PAGE_SIZE }>(value::BIG_PAGE_SIZE);
    block.set_u32::<{ layout::COMPRESSION_PAGE_SIZE }>(value::COMPRESSION_PAGE_SIZE);
    block.set_u32::<{ layout::PDE_COVERAGE_BITS }>(value::PDE_COVERAGE_BITS);
    block.set_u32::<{ layout::AVAILABLE_BIG_PAGE_SIZES }>(
        (value::SWITCH_1_BIG_PAGE_SIZES & value::KEPT_BIG_PAGE_SIZES) | value::BIG_PAGE_SIZE,
    );
    block.set_u32::<{ layout::GPC_MASK }>(value::GPC_MASK);
    let sm_version = match size {
        BlockSize::Native => value::SM_VERSION,
        BlockSize::Extended => value::EXTENDED_SM_VERSION,
    };
    block.set_u32::<{ layout::SM_VERSION }>(sm_version);
    block.set_u32::<{ layout::SPA_VERSION }>(sm_version);
    block.set_u32::<{ layout::WARP_COUNT }>(value::WARP_COUNT);
    block.set_u32::<{ layout::VA_BITS }>(value::VA_BITS);
    block.set_u64::<{ layout::FLAGS }>(value::SWITCH_1_FLAGS | flag::ADDED);
    block.set_u32::<{ layout::TWOD_CLASS }>(value::TWOD_CLASS);
    block.set_u32::<{ layout::THREED_CLASS }>(value::THREED_CLASS);
    block.set_u32::<{ layout::COMPUTE_CLASS }>(value::COMPUTE_CLASS);
    block.set_u32::<{ layout::GPFIFO_CLASS }>(value::GPFIFO_CLASS);
    block.set_u32::<{ layout::INLINE_TO_MEMORY_CLASS }>(value::INLINE_TO_MEMORY_CLASS);
    block.set_u32::<{ layout::DMA_COPY_CLASS }>(value::DMA_COPY_CLASS);
    let [l0, l1, l2, l3, l4, l5, l6, l7, l8] = value::LEVELS.map(i16::cast_unsigned);
    block.set_u16::<{ layout::LEVELS }>(l0);
    block.set_u16::<{ layout::LEVELS + 2 }>(l1);
    block.set_u16::<{ layout::LEVELS + 4 }>(l2);
    block.set_u16::<{ layout::LEVELS + 6 }>(l3);
    block.set_u16::<{ layout::LEVELS + 8 }>(l4);
    block.set_u16::<{ layout::LEVELS + 10 }>(l5);
    block.set_u16::<{ layout::LEVELS + 12 }>(l6);
    block.set_u16::<{ layout::LEVELS + 14 }>(l7);
    block.set_u16::<{ layout::LEVELS + 16 }>(l8);
    if size == BlockSize::Extended {
        block.set_u16::<{ layout::CHANNEL_LEVEL }>(value::EXTENDED_CHANNEL_LEVEL.cast_unsigned());
    }
    block.set_u32::<{ layout::SUBCONTEXTS }>(SUBCONTEXT_COUNT);
    let mut bytes = block.into_output();
    bytes.truncate(size.bytes());
    bytes
}

/// ctrl-gpu's answer to `call` on `Ioctl`, or `None` for an ioctl it does not know (module docs,
/// "The ioctls").
pub(super) fn ioctl1(call: &IoctlCall<'_>) -> Option<IoctlAnswer> {
    if call.ioctl.group() != GROUP {
        return None;
    }
    // Each sets every field of its struct, so the input is not read (module docs).
    let output = match call.ioctl.command() {
        ZCULL_GET_CTX_SIZE => zcull::CTX_SIZE.to_le_bytes().to_vec(),
        ZCULL_GET_INFO => zcull::info()
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
        _ => return None,
    };
    Some(IoctlAnswer::with(NvResult::Success, output))
}

/// ctrl-gpu's answer to `call` on `Ioctl3`, or `None` for an ioctl it does not know (module
/// docs, "The ioctls").
pub(super) fn ioctl3(call: &IoctlCall<'_>) -> Option<IoctlAnswer> {
    if call.ioctl.group() == GROUP {
        match call.ioctl.command() {
            GET_TPC_MASKS => return Some(tpc_masks(call)),
            VSMS_MAPPING => return Some(vsms_mapping(call)),
            _ => {}
        }
    }
    characteristics_answer(call)
}

/// `GetTPCMasks3` (module docs, "The ioctls").
fn tpc_masks(call: &IoctlCall<'_>) -> IoctlAnswer {
    let mut arg = Fixed::<{ tpc_masks::BYTES }>::from_input(call.input);
    if arg.u32_at::<{ tpc_masks::BUFFER_SIZE }>() != 0 {
        arg.set_u32::<{ tpc_masks::MASK }>(topology::TPC_MASK);
    }
    let inline = if call.inline_out_len >= tpc_masks::ITEM_BYTES {
        arg.u32_at::<{ tpc_masks::MASK }>().to_le_bytes().to_vec()
    } else {
        Vec::new()
    };
    IoctlAnswer {
        result: NvResult::Success,
        output: arg.into_output(),
        inline,
    }
}

/// `VSMS_MAPPING` (module docs, "The ioctls").
fn vsms_mapping(call: &IoctlCall<'_>) -> IoctlAnswer {
    let sms = usize::try_from(topology::SM_COUNT).unwrap_or(usize::MAX);
    let pairs_bytes = sms.saturating_mul(VSM_PAIR_BYTES);
    if call.ioctl.raw() != VSMS_MAPPING_IOCTL
        || call.input.len() != VSMS_ARGUMENT_BYTES
        || call.out_len != VSMS_ARGUMENT_BYTES
        || call.inline_out_len < pairs_bytes
    {
        tracing::warn!(
            ioctl = call.ioctl.raw(),
            input = call.input.len(),
            output = call.out_len,
            inline = call.inline_out_len,
            "nvsrv: a VSMS mapping request ctrl-gpu does not take"
        );
        return IoctlAnswer::only(NvResult::BadParameter);
    }
    let inline = (0..topology::SM_COUNT)
        .flat_map(|vsm| {
            let (gpc, tpc) = topology::vsm_location(vsm);
            [gpc, tpc].map(|index| u8::try_from(index).unwrap_or(u8::MAX))
        })
        .collect();
    IoctlAnswer {
        result: NvResult::Success,
        output: call.input.to_vec(),
        inline,
    }
}

/// `GetCharacteristics` on `Ioctl3`, or `None` for a number other than PR's two (module docs,
/// "The ioctls").
fn characteristics_answer(call: &IoctlCall<'_>) -> Option<IoctlAnswer> {
    let size = BlockSize::of(call.ioctl)?;
    let block_bytes = size.bytes();
    let packet = HEADER_BYTES + block_bytes;
    if call.input.len() < packet || call.out_len < packet || call.inline_out_len < block_bytes {
        tracing::warn!(
            input = call.input.len(),
            output = call.out_len,
            inline = call.inline_out_len,
            "nvsrv: characteristics buffers too short"
        );
        return Some(IoctlAnswer::only(NvResult::BadParameter));
    }
    let block = characteristics(size);
    let mut output = Vec::with_capacity(packet);
    output.extend(u64::try_from(block_bytes).unwrap_or_default().to_le_bytes());
    output.extend(call.input.get(8..HEADER_BYTES).unwrap_or_default());
    output.extend(&block);
    Some(IoctlAnswer {
        result: NvResult::Success,
        output,
        inline: block,
    })
}

/// One ctrl-gpu fd's two events, made when it was opened (module docs, "Events").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CtrlGpuEvents {
    /// Event 1 ([`event::ERROR_NOTIFIER`]).
    pub error_notifier: ObjectId,
    /// Event 2 ([`event::UNKNOWN`]).
    pub unknown: ObjectId,
}

impl CtrlGpuEvents {
    /// The event `QueryEvent` id `id` answers, or `None` for an id ctrl-gpu has no event for.
    #[must_use]
    pub const fn get(self, id: u32) -> Option<ObjectId> {
        match id {
            event::ERROR_NOTIFIER => Some(self.error_notifier),
            event::UNKNOWN => Some(self.unknown),
            _ => None,
        }
    }

    /// Both events, in id order.
    const fn both(self) -> [ObjectId; 2] {
        [self.error_notifier, self.unknown]
    }
}

/// The saved form of [`CtrlGpuEvents`], field for field; [`CtrlGpu`]'s restore checks it.
#[derive(Deserialize)]
struct SavedCtrlGpuEvents {
    error_notifier: ObjectId,
    unknown: ObjectId,
}

/// Every open ctrl-gpu fd's events. Plain data, saved with the nvsrv state; deserialising goes
/// through `SavedCtrlGpu`, which refuses a table `Open` cannot make ([`CtrlGpuStateError`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SavedCtrlGpu")]
pub struct CtrlGpu {
    events: BTreeMap<u32, CtrlGpuEvents>,
    /// Which fd each event is, an index of `events` kept beside it so that a check costs a
    /// lookup, not a walk of the table; not saved, rebuilt on restore.
    #[serde(skip)]
    owners: BTreeMap<ObjectId, u32>,
}

/// The saved form of [`CtrlGpu`].
#[derive(Deserialize)]
struct SavedCtrlGpu {
    events: BTreeMap<u32, SavedCtrlGpuEvents>,
}

/// Why a ctrl-gpu fd's events could not be kept (module docs, "What is saved"): checked, in this
/// order, when `Open` keeps a new fd's pair and when a saved table is read back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CtrlGpuStateError {
    /// The fd's two events are one event.
    #[error("ctrl-gpu fd {fd} has one event as both its events")]
    SameEvent {
        /// The fd.
        fd: u32,
    },
    /// An event already another ctrl-gpu fd's.
    #[error("ctrl-gpu fds {first} and {second} share an event")]
    SharedEvent {
        /// The fd that has it already (in a saved table, the lower).
        first: u32,
        /// The fd that has it too.
        second: u32,
    },
}

impl TryFrom<SavedCtrlGpu> for CtrlGpu {
    type Error = CtrlGpuStateError;

    fn try_from(saved: SavedCtrlGpu) -> Result<Self, Self::Error> {
        let mut table = Self::default();
        for (fd, saved) in saved.events {
            let events = CtrlGpuEvents {
                error_notifier: saved.error_notifier,
                unknown: saved.unknown,
            };
            table.keep(fd, events)?;
        }
        Ok(table)
    }
}

impl CtrlGpu {
    /// No fds.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The fds with events, in order.
    pub fn fds(&self) -> impl Iterator<Item = u32> + '_ {
        self.events.keys().copied()
    }

    /// The events of fd `fd`, if it is a ctrl-gpu fd that is open.
    #[must_use]
    pub fn events(&self, fd: u32) -> Option<CtrlGpuEvents> {
        self.events.get(&fd).copied()
    }

    /// The events `fd`'s pair may not use: its own two must differ, and neither may be another
    /// fd's (module docs, "What is saved"). Nothing is changed.
    pub fn check(&self, fd: u32, events: CtrlGpuEvents) -> Result<(), CtrlGpuStateError> {
        if events.error_notifier == events.unknown {
            return Err(CtrlGpuStateError::SameEvent { fd });
        }
        match events
            .both()
            .iter()
            .find_map(|event| self.owners.get(event))
        {
            Some(&first) => Err(CtrlGpuStateError::SharedEvent { first, second: fd }),
            None => Ok(()),
        }
    }

    /// Keeps `events` as fd `fd`'s, if [`check`](Self::check) allows them; nothing changes
    /// otherwise.
    pub(super) fn keep(&mut self, fd: u32, events: CtrlGpuEvents) -> Result<(), CtrlGpuStateError> {
        self.check(fd, events)?;
        for event in events.both() {
            self.owners.insert(event, fd);
        }
        self.events.insert(fd, events);
        Ok(())
    }

    /// `QueryEvent` of event `id` on fd `fd` (module docs, "Events"): the event, or
    /// [`Queried::Missing`] for an id ctrl-gpu has no event for, or an fd without events.
    #[must_use]
    pub fn event(&self, fd: u32, id: u32) -> Queried {
        self.events(fd)
            .and_then(|events| events.get(id))
            .map_or(Queried::Missing, Queried::Event)
    }

    /// Fd `fd` was closed: its events go (the kernel events are not freed, module docs).
    pub(super) fn close(&mut self, fd: u32) {
        if let Some(events) = self.events.remove(&fd) {
            for event in events.both() {
                self.owners.remove(&event);
            }
        }
    }
}
