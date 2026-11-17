//! `/dev/nvhost-ctrl-gpu`: the GPU's characteristics block, which Switch 2's NvRm reads before
//! it opens anything else (`docs/specs/graphics-server.md`, "Channels and submission: nvsrv",
//! the device table's ctrl-gpu row; port slice 10).
//!
//! Ported from Eden PR #4588 (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project) @
//! e5fb3b97: `eden-pr4588/src/core/hle/service/nvdrv/devices/native_gpu_characteristics.h`
//! (`MakeNativeGpuCharacteristics`, `NativeGpuSubcontextCount`, the two block sizes) and
//! `nvhost_ctrl_gpu.cpp` (`nvhost_ctrl_gpu::Ioctl3`, the `0xc0e04705` and `0xc0e84705` case),
//! over the Switch 1 block both build on, Eden @ cf654052
//! `eden/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.h` (`IoctlGpuCharacteristics`, its
//! fields and offsets) and `nvhost_ctrl_gpu.cpp` (`GetCharacteristics3`, the values). "Row N"
//! is a row of `eden-nx2/NX2_NOTES.md` @ 50fa86bb; "NX2" is
//! `eden-nx2/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.cpp` @ 50fa86bb
//! (`ApplyNx2Characteristics`, `ApplyNx2CharacteristicsTail`).
//!
//! # The ioctl
//!
//! `Ioctl3` (12) with ioctl [`CHARACTERISTICS`] (0xD0-byte block) or [`CHARACTERISTICS_EXTENDED`]
//! (0xD8): group `'G'` ([`GROUP`]), command 5, both directions, PR's two exact numbers. The
//! argument is a 16-byte header (`u64` size at +0, `u64` buffer address at +8) followed by the
//! block: [`HEADER_BYTES`] + the block's size, the "packet".
//!
//! **Checks**, PR's one: the in buffer or out buffer 0 shorter than the packet, or out buffer 1
//! (the inline output) shorter than the block, is `BadParameter`, and nothing is written.
//!
//! **Answer.** Out buffer 0 gets the packet: the block's size as a `u64` at +0, the caller's
//! buffer address (in bytes 8 to 16) as it sent it, then the block. Out buffer 1 gets the block.
//! (PR zeroes both buffers first and copies the header, then writes the size over its first
//! eight bytes.)
//!
//! Every other ctrl-gpu ioctl, on every variant, goes to the server's unknown-ioctl policy
//! (see "Not built").
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
//! | 0x70 | nine `s16` interface levels | 0x1C, 0xB, -1, 0x1D, 0x14, 0, 0, 0, 0 | the fourth 0x1F | row 24 (observed: "must be >= 6"); values PR's |
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
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `ctrl_gpu/tests.rs` unless another file is given:
//!
//! | Case | PR | Here |
//! |---|---|---|
//! | Out buffer 0 past the packet, out buffer 1 past the block | zeroed | left as they were (`only_the_packet_and_the_block_are_written`), as nvsrv writes every answer (`crate::nvsrv`, "Where this differs from the references") |
//! | A refused request's buffers | written with whatever the server's scratch buffers held | nothing written (`a_short_buffer_is_a_bad_parameter`) |
//! | The bootstrap's GPU profile (`ApplyNativeGpuProfile` in `nvdrv_interface.cpp`'s `Ioctl3`, which sets the SM versions and the fourth level from root 4's flags) | applied | not applied: nvsrv does not see the root's bootstrap yet (see "Not built"); a title bootstrapped with flags 6 or 7 reads 0x807 from the 0xD0 block (`the_block_s_golden_bytes`) |
//!
//! # Not built
//!
//! - The Switch 1 forms of `GetCharacteristics` (`Ioctl1` command 5, and `Ioctl3` command 5
//!   with any other number), `GetTPCMasks`, the ZBC and ZCULL ioctls, `FlushL2`, `GetGpuTime`,
//!   PR's `VSMS_MAPPING`, NX2's `GetEngineInfo`, and ctrl-gpu's two events (`QueryEvent` 1 and
//!   2): no Switch 2 row needs them yet. They go to the unknown-ioctl policy, and `QueryEvent`
//!   answers `BadParameter` as for any device without events.
//! - The bootstrap's GPU profile (above): it needs root 4's flags carried to nvsrv, which is the
//!   wiring slice's (16).

use super::ioctl::{Fixed, Ioctl};
use super::{IoctlAnswer, IoctlCall, NvResult};

#[cfg(test)]
mod tests;

/// ctrl-gpu's ioctl group, `'G'` (`nvhost_ctrl_gpu::Ioctl3`: `case 'G':`).
pub const GROUP: u32 = 0x47;

/// `GetCharacteristics`'s command in the group (`case 0x5:`).
pub const GET_CHARACTERISTICS: u32 = 0x5;

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

/// ctrl-gpu's answer to `call` on `Ioctl3`, or `None` for an ioctl it does not know (module
/// docs, "The ioctl").
pub(super) fn ioctl3(call: &IoctlCall<'_>) -> Option<IoctlAnswer> {
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
