//! The characteristics block, byte for byte, and `GetCharacteristics` through the server.
//!
//! The golden bytes are written out by hand from the cited layouts: Eden's
//! `IoctlGpuCharacteristics` offsets and `GetCharacteristics3` values
//! (`eden/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.h` and `.cpp` @ cf654052) with PR's
//! changes (`eden-pr4588/src/core/hle/service/nvdrv/devices/native_gpu_characteristics.h` @
//! e5fb3b97), never from the block builder.

use std::sync::Arc;

use starquake_ipc::{Message, SessionKind};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::SlabId;
use starquake_services::framework::{Kernel, UnknownPolicy};

use super::super::command::{CLOSE, QUERY_EVENT};
use super::super::tests::{Answer, Rig, STALE, answer, as_front, cmif, le, ok};
use super::*;
use crate::nvsrv::{Device, NvsrvFamily, nvsrv_server};

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
    // NX2's comment in `ApplyNx2Characteristics`: NvRm requires the level at +0x78 to be at
    // least 6. (Row 24 says every level from 0x70 to 0x80; the module docs' "Conflicts".)
    assert_eq!(layout::LEVELS + 2 * 4, 0x78);
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

// ----- The layout ioctls -----------------------------------------------------------------------

/// The numbers the title of issue #300 sends (`0x80044701`, `0x80284702`, `0xc0184706`,
/// `0xc0084713`), and the constants they decode to, against the references' values written out.
#[test]
fn the_layout_numbers_decode_to_their_commands() {
    let cases = [
        (0x8004_4701, ZCULL_GET_CTX_SIZE, zcull::CTX_SIZE_BYTES),
        (0x8028_4702, ZCULL_GET_INFO, zcull::INFO_BYTES),
        (0xC018_4706, GET_TPC_MASKS, tpc_masks::BYTES),
        (VSMS_MAPPING_IOCTL, VSMS_MAPPING, VSMS_ARGUMENT_BYTES),
    ];
    for (raw, command, size) in cases {
        let ioctl = Ioctl::from_raw(raw);
        assert_eq!(ioctl.group(), GROUP, "{raw:#x}");
        assert_eq!(ioctl.command(), command, "{raw:#x}");
        assert_eq!(usize::try_from(ioctl.size()).unwrap(), size, "{raw:#x}");
        assert!(ioctl.is_out(), "{raw:#x}");
    }
    assert_eq!(VSMS_MAPPING_IOCTL, 0xC008_4713);
    assert_eq!(
        [
            GROUP,
            ZCULL_GET_CTX_SIZE,
            ZCULL_GET_INFO,
            GET_TPC_MASKS,
            VSMS_MAPPING
        ],
        [0x47, 0x1, 0x2, 0x6, 0x13]
    );
    assert_eq!(
        [
            zcull::CTX_SIZE_BYTES,
            zcull::INFO_BYTES,
            tpc_masks::BYTES,
            tpc_masks::BUFFER_SIZE,
            tpc_masks::BUFFER_ADDRESS,
            tpc_masks::MASK,
            tpc_masks::ITEM_BYTES,
            VSMS_ARGUMENT_BYTES,
            VSM_PAIR_BYTES,
        ],
        [4, 40, 24, 0, 8, 16, 4, 8, 2]
    );
    assert_eq!(topology::SMS_PER_TPC, 2);
    assert_eq!([event::ERROR_NOTIFIER, event::UNKNOWN], [1, 2]);
}

/// `ZCullGetCtxSize`: Eden's `size = 0x1`, whatever was sent; an out buffer longer than the
/// struct keeps its tail, and a shorter one gets what fits.
#[test]
fn zcull_get_ctx_size_s_golden_bytes() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let raw = 0x8004_4701;
    for input in [vec![], le(&[0xAABB_CCDD]), le(&[0xAABB_CCDD, 0x1122_3344])] {
        let (reply, out) = rig.ioctl(fd, raw, &input, 4);
        assert_eq!((reply, out), (ok(&[0]), le(&[1])), "{input:x?}");
    }
    let (_, out) = rig.ioctl(fd, raw, &[], 6);
    assert_eq!(out, [1, 0, 0, 0, STALE, STALE]);
    let (_, out) = rig.ioctl(fd, raw, &[], 3);
    assert_eq!(out, [1, 0, 0]);
}

/// `ZCullGetInfo`: Eden's ten words (the same in PR and NX2), written out, whatever was sent.
#[test]
fn zcull_get_info_s_golden_bytes() {
    let golden = le(&[0x20, 0x20, 0x400, 0x800, 0x20, 0x20, 0xC0, 0x20, 0x40, 0x10]);
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let raw = 0x8028_4702;
    for input in [vec![], vec![0x5A; 40]] {
        let (reply, out) = rig.ioctl(fd, raw, &input, 40);
        assert_eq!((reply, out), (ok(&[0]), golden.clone()));
    }
    let (_, out) = rig.ioctl(fd, raw, &[], 44);
    let mut expected = golden.clone();
    expected.extend([STALE; 4]);
    assert_eq!(out, expected);
    let (_, out) = rig.ioctl(fd, raw, &[], 39);
    assert_eq!(out, golden[..39]);
}

/// `IoctlGpuGetTpcMasksArgs` with distinct values: buffer size `size`, padding 0x11, address
/// `0x0102_0304_0506_0708`, mask 0xEE, padding 0x22.
fn tpc_request(size: u32) -> Vec<u8> {
    le(&[size, 0x11, 0x0506_0708, 0x0102_0304, 0xEE, 0x22])
}

/// `GetTPCMasks3`: with a mask buffer, the mask becomes 3 (PR and NX2: `tcp_mask = 3`) and is
/// out buffer 1's first word; with none, the mask comes back as sent, and is out buffer 1's
/// first word all the same; every other field as sent.
#[test]
fn tpc_masks_golden_bytes() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let raw = 0xC018_4706;
    let (reply, out, inline) = rig.ioctl3(fd, raw, &tpc_request(4), (24, 4));
    assert_eq!(reply, ok(&[0]));
    assert_eq!(out, le(&[4, 0x11, 0x0506_0708, 0x0102_0304, 3, 0x22]));
    assert_eq!(inline, le(&[3]));
    // A size that differs only above its low byte is still not zero.
    let (_, out, inline) = rig.ioctl3(fd, raw, &tpc_request(0x100), (24, 4));
    assert_eq!(out, le(&[0x100, 0x11, 0x0506_0708, 0x0102_0304, 3, 0x22]));
    assert_eq!(inline, le(&[3]));
    let (reply, out, inline) = rig.ioctl3(fd, raw, &tpc_request(0), (24, 4));
    assert_eq!(reply, ok(&[0]));
    assert_eq!(out, tpc_request(0));
    assert_eq!(inline, le(&[0xEE]));
    // A struct cut after the size: the rest is zero, the mask set.
    let (_, out, inline) = rig.ioctl3(fd, raw, &le(&[4]), (24, 4));
    assert_eq!(out, le(&[4, 0, 0, 0, 3, 0]));
    assert_eq!(inline, le(&[3]));
}

/// Out buffer 1 gets the mask word only if it holds a whole word, and nothing past it (a listed
/// divergence: PR zeroes every whole word after it).
#[test]
fn tpc_masks_write_one_word() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let raw = 0xC018_4706;
    let (reply, _, inline) = rig.ioctl3(fd, raw, &tpc_request(4), (24, 3));
    assert_eq!((reply, inline), (ok(&[0]), vec![STALE; 3]));
    let (_, _, inline) = rig.ioctl3(fd, raw, &tpc_request(4), (24, 4));
    assert_eq!(inline, le(&[3]));
    let (_, _, inline) = rig.ioctl3(fd, raw, &tpc_request(4), (24, 12));
    let mut expected = le(&[3]);
    expected.resize(12, STALE);
    assert_eq!(inline, expected);
}

/// `VSMS_MAPPING`: out buffer 0 gets the argument as sent, out buffer 1 PR's four `{gpc, tpc}`
/// pairs written out, and nothing past them.
#[test]
fn vsms_mapping_s_golden_bytes() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let input = le(&[0x0403_0201, 0x0807_0605]);
    let pairs = vec![0, 0, 0, 0, 0, 1, 0, 1];
    let (reply, out, inline) = rig.ioctl3(fd, VSMS_MAPPING_IOCTL, &input, (8, 8));
    assert_eq!(
        (reply, out, inline),
        (ok(&[0]), input.clone(), pairs.clone())
    );
    let (_, _, inline) = rig.ioctl3(fd, VSMS_MAPPING_IOCTL, &input, (8, 11));
    let mut expected = pairs;
    expected.extend([STALE; 3]);
    assert_eq!(inline, expected);
}

/// Each `VSMS_MAPPING` check refuses at its edge with `BadParameter` and writes nothing: an in
/// buffer or out buffer 0 one byte either side of 8, out buffer 1 one byte short of two per SM
/// (and empty), and numbers with command 0x13 but PR's (another size, only the out bit, another
/// group's 0x13 does not reach here). The accepted edges are in `vsms_mapping_s_golden_bytes`.
#[test]
fn vsms_mapping_refuses_each_size_and_number() {
    let pairs = usize::try_from(topology::SM_COUNT).unwrap() * VSM_PAIR_BYTES;
    let cases = [
        (VSMS_MAPPING_IOCTL, 7, 8, pairs),
        (VSMS_MAPPING_IOCTL, 9, 8, pairs),
        (VSMS_MAPPING_IOCTL, 8, 7, pairs),
        (VSMS_MAPPING_IOCTL, 8, 9, pairs),
        (VSMS_MAPPING_IOCTL, 8, 8, pairs - 1),
        (VSMS_MAPPING_IOCTL, 8, 8, 0),
        (0xC010_4713, 8, 8, pairs),
        (0x8008_4713, 8, 8, pairs),
    ];
    for (raw, input_len, out_len, inline_len) in cases {
        let mut rig = Rig::new();
        let fd = rig.open_device(Device::NvhostCtrlGpu);
        let input = vec![0x3C; input_len];
        let (reply, out, inline) = rig.ioctl3(fd, raw, &input, (out_len, inline_len));
        assert_eq!(
            reply,
            ok(&[NvResult::BadParameter.raw()]),
            "{raw:#x} {input_len} {out_len} {inline_len}"
        );
        assert_eq!(out, vec![STALE; out_len]);
        assert_eq!(inline, vec![STALE; inline_len]);
    }
}

/// The Z-cull ioctls are `Ioctl`'s, the TPC masks and the SM map `Ioctl3`'s; on another variant,
/// or in another group, each goes to the policy (strict: `NotImplemented`, nothing written).
#[test]
fn the_layout_ioctls_on_other_variants_go_to_the_policy() {
    let mut rig = Rig::with(UnknownPolicy::Strict);
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let not_implemented = ok(&[NvResult::NotImplemented.raw()]);
    for raw in [0x8004_4701, 0x8028_4702] {
        let (reply, out, inline) = rig.ioctl3(fd, raw, &[0; 40], (40, 40));
        assert_eq!(reply, not_implemented, "{raw:#x}");
        assert_eq!((out, inline), (vec![STALE; 40], vec![STALE; 40]));
    }
    for raw in [0xC018_4706, VSMS_MAPPING_IOCTL] {
        let (reply, out) = rig.ioctl(fd, raw, &[0; 24], 24);
        assert_eq!((reply, out), (not_implemented.clone(), vec![STALE; 24]));
    }
    // Command 1 in group 'H', and ZBC's command 3 in 'G', on `Ioctl`.
    for raw in [0x8004_4801, 0xC02C_4703] {
        let (reply, out) = rig.ioctl(fd, raw, &[0; 4], 4);
        assert_eq!((reply, out), (not_implemented.clone(), vec![STALE; 4]));
    }
    // TPC masks and the SM map in group 'H' on `Ioctl3`.
    for raw in [0xC018_4806, 0xC008_4813] {
        let (reply, _, inline) = rig.ioctl3(fd, raw, &tpc_request(4), (24, 8));
        assert_eq!((reply, inline), (not_implemented.clone(), vec![STALE; 8]));
    }
}

/// The layout replies agree with the block (module docs, "The GPU's layout, one source"): the GPC
/// and TPC counts read back from the block's bytes give the TPC mask, the SM count (the exact
/// out buffer 1 `VSMS_MAPPING` takes), the SM map and the Z-cull fields that scale with them.
#[test]
fn the_layout_replies_agree_with_the_block() {
    let block = characteristics(BlockSize::Native);
    let word = |at: usize| u32::from_le_bytes(block[at..at + 4].try_into().unwrap());
    let gpcs = word(layout::NUM_GPC);
    let tpcs_per_gpc = word(layout::TPC_PER_GPC);
    assert_eq!(word(layout::GPC_MASK), (1 << gpcs) - 1);
    let tpcs = gpcs * tpcs_per_gpc;
    let sms = tpcs * topology::SMS_PER_TPC;

    let mut rig = Rig::new();
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let (_, _, inline) = rig.ioctl3(fd, 0xC018_4706, &tpc_request(4), (24, 4));
    assert_eq!(inline, le(&[(1 << tpcs_per_gpc) - 1]));

    let pairs = usize::try_from(sms).unwrap() * VSM_PAIR_BYTES;
    let input = [0; VSMS_ARGUMENT_BYTES];
    let (reply, _, map) = rig.ioctl3(fd, VSMS_MAPPING_IOCTL, &input, (8, pairs));
    assert_eq!(reply, ok(&[0]));
    let (reply, _, _) = rig.ioctl3(fd, VSMS_MAPPING_IOCTL, &input, (8, pairs - 1));
    assert_eq!(reply, ok(&[NvResult::BadParameter.raw()]));
    // Every TPC of every GPC appears once per SM, in order.
    let mut expected = Vec::new();
    for gpc in 0..gpcs {
        for tpc in 0..tpcs_per_gpc {
            for _ in 0..topology::SMS_PER_TPC {
                expected.extend([u8::try_from(gpc).unwrap(), u8::try_from(tpc).unwrap()]);
            }
        }
    }
    assert_eq!(map, expected);

    let (_, info) = rig.ioctl(fd, 0x8028_4702, &[], 40);
    let info: Vec<u32> = info
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    // Width alignments per TPC; region multiplier and header sizes per GPC (a guess, module docs).
    assert_eq!(info[0], zcull::WIDTH_ALIGN_PER_TPC * tpcs);
    assert_eq!(info[7], zcull::WIDTH_ALIGN_PER_TPC * tpcs);
    assert_eq!(info[4], zcull::REGION_BYTE_MULTIPLIER_PER_GPC * gpcs);
    assert_eq!(info[5], zcull::REGION_HEADER_PER_GPC * gpcs);
    assert_eq!(info[6], zcull::SUBREGION_HEADER_PER_GPC * gpcs);
}

/// [`topology::vsm_location`] numbers SMs TPC by TPC and GPC by GPC, PR's `{0, i / 2}` for its
/// GPC; past the first GPC (which this GPU does not have) the GPC counts up.
#[test]
fn vsm_locations_walk_tpcs_then_gpcs() {
    let per_gpc = value::TPC_PER_GPC * topology::SMS_PER_TPC;
    for vsm in 0..topology::SM_COUNT {
        assert_eq!(topology::vsm_location(vsm), (0, vsm / 2), "{vsm}");
    }
    assert_eq!(topology::vsm_location(per_gpc), (1, 0));
    assert_eq!(topology::vsm_location(per_gpc + 3), (1, 1));
}

// ----- Events ----------------------------------------------------------------------------------

/// `QueryEvent` on `fd` for `id`: the reply and the handles it copied.
fn query(rig: &mut Rig, fd: i32, id: u32) -> (Answer, Vec<u32>) {
    let mut data = fd.to_le_bytes().to_vec();
    data.extend(id.to_le_bytes());
    let buffer = rig.send(&cmif(QUERY_EVENT, &data));
    let handles = Message::parse_reply(&buffer, SessionKind::Session)
        .unwrap()
        .copy_handles;
    (answer(&buffer, 1), handles)
}

/// The object behind a copied handle.
fn object(rig: &Rig, handle: u32) -> ObjectId {
    rig.kernel.resolve_handle(Handle::from_raw(handle)).unwrap()
}

/// Events 1 and 2 hand out the two events made at open, the same each time, each with
/// `Success`; any other id is `BadParameter` with no handle. A second ctrl-gpu fd has its own.
#[test]
fn query_event_1_and_2_hand_out_the_events_made_at_open() {
    let mut rig = Rig::new();
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let made = rig.state().ctrl_gpu().events(fd.cast_unsigned()).unwrap();
    let success = ok(&[NvResult::Success.raw()]);
    for _ in 0..2 {
        let (reply, handles) = query(&mut rig, fd, event::ERROR_NOTIFIER);
        assert_eq!(reply, success);
        assert_eq!(handles.len(), 1);
        assert_eq!(object(&rig, handles[0]), made.error_notifier);
        let (reply, handles) = query(&mut rig, fd, event::UNKNOWN);
        assert_eq!(reply, success);
        assert_eq!(handles.len(), 1);
        assert_eq!(object(&rig, handles[0]), made.unknown);
    }
    // Asking makes nothing new.
    assert_eq!(
        rig.state().ctrl_gpu().events(fd.cast_unsigned()),
        Some(made)
    );
    for id in [0, 3, 7, 0x101, u32::MAX] {
        let (reply, handles) = query(&mut rig, fd, id);
        assert_eq!(reply, ok(&[NvResult::BadParameter.raw()]), "{id}");
        assert_eq!(handles, Vec::<u32>::new());
    }
    // Another fd's events are its own.
    let other = rig.open_device(Device::NvhostCtrlGpu);
    let (_, handles) = query(&mut rig, other, event::ERROR_NOTIFIER);
    let theirs = object(&rig, handles[0]);
    assert_ne!(theirs, made.error_notifier);
    assert_ne!(theirs, made.unknown);
    let kept = rig
        .state()
        .ctrl_gpu()
        .events(other.cast_unsigned())
        .unwrap();
    assert_eq!(kept.error_notifier, theirs);
}

/// Closing a ctrl-gpu fd drops its events: a query on it is the closed fd's `NotImplemented`.
/// The kernel events are not freed (a listed divergence): two cycles of open and close leave four
/// distinct events, each still held by the kernel.
#[test]
fn a_closed_ctrl_gpu_s_events_are_not_freed() {
    let mut rig = Rig::new();
    let mut made = Vec::new();
    for _ in 0..2 {
        let fd = rig.open_device(Device::NvhostCtrlGpu);
        let events = rig.state().ctrl_gpu().events(fd.cast_unsigned()).unwrap();
        made.extend([events.error_notifier, events.unknown]);
        let reply = answer(&rig.send(&cmif(CLOSE, &fd.to_le_bytes())), 1);
        assert_eq!(reply, ok(&[0]));
        assert_eq!(rig.state().ctrl_gpu().events(fd.cast_unsigned()), None);
        let (reply, handles) = query(&mut rig, fd, event::ERROR_NOTIFIER);
        assert_eq!(reply, ok(&[NvResult::NotImplemented.raw()]));
        assert_eq!(handles, Vec::<u32>::new());
    }
    let distinct: std::collections::BTreeSet<_> = made.iter().collect();
    assert_eq!(distinct.len(), 4);
    for event in made {
        assert_eq!(rig.kernel.signals(event), Some(0), "{event:?}");
    }
    assert_eq!(rig.state().ctrl_gpu().fds().count(), 0);
}

/// After a whole-server snapshot and a restore into a new server, `QueryEvent` 1 and 2 hand out
/// the same events as before.
#[test]
fn a_restored_ctrl_gpu_hands_out_its_events() {
    let mut rig = Rig::new();
    rig.open_device(Device::Nvmap);
    let fd = rig.open_device(Device::NvhostCtrlGpu);
    let made = rig.state().ctrl_gpu().events(fd.cast_unsigned()).unwrap();
    let state = rig.state().clone();
    let saved = rig.server.save().unwrap();
    let family = NvsrvFamily::new(Arc::clone(&rig.shared), as_front(&rig.front));
    rig.server = nvsrv_server(
        Arc::clone(&rig.shared),
        as_front(&rig.front),
        UnknownPolicy::Strict,
    )
    .with_family(family)
    .with_service_id(rig.server.service_id());
    rig.server.restore(&saved).unwrap();
    assert_eq!(rig.state(), &state);
    let (reply, handles) = query(&mut rig, fd, event::UNKNOWN);
    assert_eq!(reply, ok(&[NvResult::Success.raw()]));
    assert_eq!(object(&rig, handles[0]), made.unknown);
    let (_, handles) = query(&mut rig, fd, event::ERROR_NOTIFIER);
    assert_eq!(object(&rig, handles[0]), made.error_notifier);
}

// ----- The saved events ------------------------------------------------------------------------

/// Made-up event pairs keyed by fd, from `(fd, index, index)` triples, generation 7.
fn saved_table(pairs: &[(u32, u32, u32)]) -> SavedCtrlGpu {
    let id = |index: u32| ObjectId::from_parts(index, 7);
    SavedCtrlGpu {
        events: pairs
            .iter()
            .map(|&(fd, first, second)| {
                let events = SavedCtrlGpuEvents {
                    error_notifier: id(first),
                    unknown: id(second),
                };
                (fd, events)
            })
            .collect(),
    }
}

/// A saved table `Open` cannot make is refused: an fd's two events one event; an event two fds
/// share, either of the pair, the lower fd first. Both faults on one fd: the one event is
/// reported. Events that differ only in their generation are distinct.
#[test]
fn a_saved_ctrl_gpu_table_open_cannot_make_is_refused() {
    let shared = CtrlGpuStateError::SharedEvent {
        first: 2,
        second: 4,
    };
    let refused = [
        (vec![(3, 5, 5)], CtrlGpuStateError::SameEvent { fd: 3 }),
        (vec![(2, 5, 6), (4, 5, 8)], shared),
        (vec![(2, 5, 6), (4, 8, 6)], shared),
        (vec![(2, 5, 6), (4, 6, 5)], shared),
        // Both on fd 4: its one event is fd 2's too.
        (
            vec![(2, 5, 6), (4, 6, 6)],
            CtrlGpuStateError::SameEvent { fd: 4 },
        ),
        // The first fault in fd order: fd 3's.
        (
            vec![(2, 5, 6), (3, 9, 9), (4, 5, 8)],
            CtrlGpuStateError::SameEvent { fd: 3 },
        ),
    ];
    for (pairs, error) in refused {
        assert_eq!(
            CtrlGpu::try_from(saved_table(&pairs)),
            Err(error),
            "{pairs:?}"
        );
    }
    let table = CtrlGpu::try_from(saved_table(&[(2, 5, 6), (4, 8, 9)])).unwrap();
    assert_eq!(table.fds().collect::<Vec<_>>(), [2, 4]);
    // The same indexes, another generation: other events.
    let mut saved = saved_table(&[(2, 5, 6)]);
    saved.events.insert(
        4,
        SavedCtrlGpuEvents {
            error_notifier: ObjectId::from_parts(5, 8),
            unknown: ObjectId::from_parts(6, 8),
        },
    );
    assert!(CtrlGpu::try_from(saved).is_ok());
    // Through the bytes: one fd 2, events (5, 7) and (6, 7); with the second made the first,
    // the bytes are a decode error.
    let one = CtrlGpu::try_from(saved_table(&[(2, 5, 6)])).unwrap();
    let bytes = postcard::to_allocvec(&one).unwrap();
    assert_eq!(bytes, [1, 2, 5, 7, 6, 7]);
    assert_eq!(postcard::from_bytes::<CtrlGpu>(&bytes).unwrap(), one);
    assert!(postcard::from_bytes::<CtrlGpu>(&[1, 2, 5, 7, 5, 7]).is_err());
    assert_eq!(
        CtrlGpuStateError::SameEvent { fd: 3 }.to_string(),
        "ctrl-gpu fd 3 has one event as both its events"
    );
    assert_eq!(shared.to_string(), "ctrl-gpu fds 2 and 4 share an event");
}

/// A restored table checks new pairs as `Open` does: a pair sharing one of its events is
/// refused, and closing an fd frees its events for another (the index follows the table).
#[test]
fn a_restored_table_checks_new_pairs_against_its_events() {
    let mut table = CtrlGpu::try_from(saved_table(&[(2, 5, 6)])).unwrap();
    let id = |index: u32| ObjectId::from_parts(index, 7);
    let sharing = CtrlGpuEvents {
        error_notifier: id(9),
        unknown: id(6),
    };
    assert_eq!(
        table.check(3, sharing),
        Err(CtrlGpuStateError::SharedEvent {
            first: 2,
            second: 3
        })
    );
    table.close(2);
    assert_eq!(table.check(3, sharing), Ok(()));
    table.keep(3, sharing).unwrap();
    assert_eq!(table.event(3, event::UNKNOWN), Queried::Event(id(6)));
    assert_eq!(table.event(3, 3), Queried::Missing);
    assert_eq!(table.event(2, event::ERROR_NOTIFIER), Queried::Missing);
}
