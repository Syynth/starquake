//! The characteristics block, byte for byte, and `GetCharacteristics` through the server.
//!
//! The golden bytes are written out by hand from the cited layouts: Eden's
//! `IoctlGpuCharacteristics` offsets and `GetCharacteristics3` values
//! (`eden/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.h` and `.cpp` @ cf654052) with PR's
//! changes (`eden-pr4588/src/core/hle/service/nvdrv/devices/native_gpu_characteristics.h` @
//! e5fb3b97), never from the block builder.

use starquake_services::framework::UnknownPolicy;

use super::super::tests::{Rig, STALE, le, ok};
use super::*;
use crate::nvsrv::Device;

/// The 0xD0 block as 52 little-endian words.
const GOLDEN: [u32; 52] = [
    // 0x00: architecture 0x170, implementation 0xE (chip 0x17E, row 24).
    0x170,
    0xE, //
    // 0x08: revision 0xA1, one GPC; 0x10: L2 0x40000 (u64); 0x18: no video memory (u64).
    0xA1,
    0x1,
    0x4_0000,
    0,
    0,
    0, //
    // 0x20: two TPCs per GPC, bus type 0x20; 0x28: PR's big page size 0x1000.
    0x2,
    0x20,
    0x1000, //
    // 0x2C: compression page 0x20000; 0x30: PDE coverage 0x1B; 0x34: big page sizes 0x11000.
    0x2_0000,
    0x1B,
    0x1_1000, //
    // 0x38: GPC mask 1; 0x3C, 0x40: SM and SPA versions 0x807.
    0x1,
    0x807,
    0x807, //
    // 0x44: warp count 0x80; 0x48: 40 VA bits; 0x4C: reserved.
    0x80,
    0x28,
    0, //
    // 0x50: flags, Eden's 0x55 and bits 8, 22, 26 (low word); 34, 35, 40-43 (high word).
    0x0440_0155,
    0x0000_0F0C, //
    // 0x58: 2D, 3D, compute, GPFIFO, inline-to-memory, DMA copy classes (Eden's).
    0x902D,
    0xB197,
    0xB1C0,
    0xB06F,
    0xA140,
    0xB0B5, //
    // 0x70: levels 0x1C, 0xB | -1, 0x1D | 0x14, 0 | 0, 0 | 0, 0 (two s16 per word).
    0x000B_001C,
    0x001D_FFFF,
    0x0000_0014,
    0,
    0, //
    // 0x84 to 0xB4: zero.
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0, //
    // 0xB4: 64 subcontexts; 0xB8 to 0xD0: zero.
    64,
    0,
    0,
    0,
    0,
    0,
    0,
];

/// The block for the 0xD0 request is the golden bytes, every one.
#[test]
fn the_block_s_golden_bytes() {
    let block = characteristics(BlockSize::Native);
    assert_eq!(block.len(), BLOCK_BYTES);
    assert_eq!(block, le(&GOLDEN));
}

/// The 0xD8 block is the 0xD0 one but for the SM and SPA versions (0x808), the channel level
/// (0x1F) and eight zero bytes after it (PR's "Extended native GPU characteristics select the
/// newer shader profile").
#[test]
fn the_extended_block_differs_only_where_pr_says() {
    let mut expected = le(&GOLDEN);
    expected[0x3C..0x40].copy_from_slice(&0x808_u32.to_le_bytes());
    expected[0x40..0x44].copy_from_slice(&0x808_u32.to_le_bytes());
    expected[0x76..0x78].copy_from_slice(&0x1F_u16.to_le_bytes());
    expected.extend([0; 8]);
    let block = characteristics(BlockSize::Extended);
    assert_eq!(block.len(), EXTENDED_BLOCK_BYTES);
    assert_eq!(block, expected);
}

/// Every flag bit the spec lists (8, 22, 26, 34, 35, 40 to 43) is set, Eden's 0x55 stays, and no
/// other bit is.
#[test]
fn every_listed_flag_bit_is_set() {
    let block = characteristics(BlockSize::Native);
    let flags = u64::from_le_bytes(block[layout::FLAGS..layout::FLAGS + 8].try_into().unwrap());
    let listed = [8, 22, 26, 34, 35, 40, 41, 42, 43];
    for bit in listed {
        assert_ne!(flags & (1 << bit), 0, "bit {bit}");
    }
    let listed_mask: u64 = listed.iter().map(|bit| 1 << bit).sum();
    assert_eq!(flag::ADDED, listed_mask);
    assert_eq!(flags, value::SWITCH_1_FLAGS | listed_mask);
    // The named bits are the listed ones.
    assert_eq!(
        [
            flag::TSG,
            flag::TSG_SUBCONTEXTS,
            flag::SYNCPOINT_ADDRESS,
            flag::MAPPING_MODIFY,
            flag::BIT_35,
        ],
        [1 << 8, 1 << 22, 1 << 26, 1 << 34, 1 << 35]
    );
    assert_eq!(flag::CLASS_FAMILIES, 0xF << 40);
}

/// The nine levels read as `s16`s, and the subcontext count, at their offsets.
#[test]
fn levels_and_subcontexts_are_where_the_sdk_reads_them() {
    let block = characteristics(BlockSize::Native);
    let level = |index: usize| {
        let at = layout::LEVELS + 2 * index;
        i16::from_le_bytes([block[at], block[at + 1]])
    };
    let levels: Vec<i16> = (0..9).map(level).collect();
    assert_eq!(levels, [0x1C, 0xB, -1, 0x1D, 0x14, 0, 0, 0, 0]);
    // Row 24: NvRm requires the level at +0x78 to be at least 6.
    assert!(level(4) >= 6);
    let at = layout::SUBCONTEXTS;
    let count = u32::from_le_bytes(block[at..at + 4].try_into().unwrap());
    assert_eq!(count, SUBCONTEXT_COUNT);
    assert_eq!(layout::CHANNEL_LEVEL, layout::LEVELS + 6);
}

/// The two numbers are PR's, and decode as group 'G', command 5, both directions, with the
/// header and the block as the size.
#[test]
fn the_ioctl_numbers_are_pr_s() {
    for (raw, size) in [
        (CHARACTERISTICS, BlockSize::Native),
        (CHARACTERISTICS_EXTENDED, BlockSize::Extended),
    ] {
        let ioctl = Ioctl::from_raw(raw);
        assert_eq!(BlockSize::of(ioctl), Some(size));
        assert_eq!(ioctl.group(), GROUP);
        assert_eq!(ioctl.command(), GET_CHARACTERISTICS);
        assert!(ioctl.is_in() && ioctl.is_out());
        let declared = usize::try_from(ioctl.size()).unwrap();
        assert_eq!(declared, HEADER_BYTES + size.bytes());
    }
    assert_eq!(
        [CHARACTERISTICS, CHARACTERISTICS_EXTENDED],
        [0xC0E0_4705, 0xC0E8_4705]
    );
    // Group 'G' command 5 with another size (the Switch 1 form) is not one of them.
    assert_eq!(BlockSize::of(Ioctl::from_raw(0xC0B0_4705)), None);
}

/// The argument as NvRm sends it: the header (a size and a buffer address), then room for the
/// block, `extra` bytes more.
fn request(size: BlockSize, extra: usize) -> Vec<u8> {
    let mut input = 0xFFFF_u64.to_le_bytes().to_vec();
    input.extend(0x1122_3344_5566_7788_u64.to_le_bytes());
    input.resize(HEADER_BYTES + size.bytes() + extra, 0x3C);
    input
}

/// Through the server: out buffer 0 gets the block's size, the caller's buffer address and the
/// block; out buffer 1 the block; with buffers exactly the packet's and the block's size, and
/// with larger ones, whose tails are left as they were.
#[test]
fn only_the_packet_and_the_block_are_written() {
    for size in [BlockSize::Native, BlockSize::Extended] {
        let raw = match size {
            BlockSize::Native => CHARACTERISTICS,
            BlockSize::Extended => CHARACTERISTICS_EXTENDED,
        };
        let block = characteristics(size);
        let packet = HEADER_BYTES + size.bytes();
        for extra in [0, 0x20] {
            let mut rig = Rig::new();
            let fd = rig.open_device(Device::NvhostCtrlGpu);
            let input = request(size, extra);
            let lens = (packet + extra, size.bytes() + extra);
            let (reply, out, inline) = rig.ioctl3(fd, raw, &input, lens);
            assert_eq!(reply, ok(&[NvResult::Success.raw()]), "{size:?} {extra}");
            let mut expected = u64::try_from(size.bytes()).unwrap().to_le_bytes().to_vec();
            expected.extend(0x1122_3344_5566_7788_u64.to_le_bytes());
            expected.extend(&block);
            expected.resize(packet + extra, STALE);
            assert_eq!(out, expected, "{size:?} {extra}");
            let mut expected = block.clone();
            expected.resize(size.bytes() + extra, STALE);
            assert_eq!(inline, expected, "{size:?} {extra}");
        }
    }
}

/// One byte short in the in buffer, out buffer 0 or out buffer 1: `BadParameter`, and neither
/// out buffer is written. All three exactly long enough is the accepted edge
/// (`only_the_packet_and_the_block_are_written`).
#[test]
fn a_short_buffer_is_a_bad_parameter() {
    let size = BlockSize::Native;
    let packet = HEADER_BYTES + size.bytes();
    let cases = [
        (packet - 1, packet, size.bytes()),
        (packet, packet - 1, size.bytes()),
        (packet, packet, size.bytes() - 1),
    ];
    for (input_len, out_len, inline_len) in cases {
        let mut rig = Rig::new();
        let fd = rig.open_device(Device::NvhostCtrlGpu);
        let mut input = request(size, 0);
        input.truncate(input_len);
        let (reply, out, inline) = rig.ioctl3(fd, CHARACTERISTICS, &input, (out_len, inline_len));
        let refused = ok(&[NvResult::BadParameter.raw()]);
        assert_eq!(reply, refused, "{input_len} {out_len} {inline_len}");
        assert_eq!(out, vec![STALE; out_len]);
        assert_eq!(inline, vec![STALE; inline_len]);
    }
}

/// The ioctl is `Ioctl3`'s only: on `Ioctl`, and the Switch 1 number on `Ioctl3`, it goes to the
/// policy (strict: `NotImplemented`, nothing written).
#[test]
fn other_forms_go_to_the_policy() {
    let mut rig = Rig::with(UnknownPolicy::Strict);
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let input = request(BlockSize::Native, 0);
    let not_implemented = ok(&[NvResult::NotImplemented.raw()]);
    let (reply, out) = rig.ioctl(fd, CHARACTERISTICS, &input, 0xE0);
    assert_eq!((reply, out), (not_implemented.clone(), vec![STALE; 0xE0]));
    let (reply, out, inline) = rig.ioctl3(fd, 0xC0B0_4705, &input, (0xE0, 0xD0));
    assert_eq!(reply, not_implemented);
    assert_eq!((out, inline), (vec![STALE; 0xE0], vec![STALE; 0xD0]));
}
