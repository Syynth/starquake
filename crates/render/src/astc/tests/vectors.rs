//! Blocks laid out by hand from the ASTC specification, with the texels its formulas give.
//!
//! Section, table and figure numbers are those of `KHR_texture_compression_astc_hdr`. Each
//! expected value is derived in a comment from the specification's own arithmetic.

use super::{
    CEM_AT, COLOR_AT, DUAL_PLANE, ENDS, GRID_4X4, MULTI_CEM_AT, MULTI_COLOR_AT, PARTITION_INDEX_AT,
    PARTITIONS_AT, WEIGHTS_0_TO_2, WEIGHTS_0_TO_3, WEIGHTS_0_TO_4, WEIGHTS_0_TO_31, astc_formats,
    byte_values, decode_one, grey, legal, plain_weights, stream_bit, texel_count, two_bit_block,
};
use crate::astc::ERROR_COLOR;
use crate::format::PixelFormat;
use PixelFormat as F;

/// A luminance ramp from 0 to 255 at the four two-bit weights.
///
/// Two-bit weights unquantize by bit replication to 6 bits and then `if (T > 32) T += 1`
/// (C.2.17): 0, 21, 42 + 1, 63 + 1. With endpoints 0 and 255 widened to 0 and 65535, C.2.19
/// gives `C = floor((65535 * i + 32) / 64)` = 0, 21504, 44031, 65535, and `C / 65536` as a
/// byte, `floor(255 * C / 65536 + 0.5)`, is 0, 84, 171, 255.
const RAMP: [u8; 4] = [0, 84, 171, 255];

/// Mode 0 (luminance, direct: `e0 = (v0, v0, v0, 0xFF)`, `e1 = (v1, v1, v1, 0xFF)`) with every
/// two-bit weight value.
#[test]
fn luminance_with_two_bit_weights() {
    let weights = [0, 1, 2, 3, 3, 2, 1, 0, 0, 0, 3, 3, 1, 2, 1, 2];
    let block = two_bit_block(0, &[0, 255], &weights);
    let expected: Vec<_> = weights
        .iter()
        .map(|weight| grey(RAMP[*weight as usize]))
        .collect();
    assert_eq!(legal(block, F::Astc2d4x4Unorm), expected);
    // With the sRGB widening (`(C << 8) | 0x80`, top 8 bits of the result) the same block
    // gives the same bytes: C = 128, 21548, 43988, 65408, whose top bytes are 0, 84, 171, 255.
    assert_eq!(legal(block, F::Astc2d4x4Srgb), expected);
}

/// Weights stored as trits (range 0..2), which unquantize to 0, 32 and 63 + 1 (table C.2.25).
///
/// Sixteen weights are three packed blocks of five and one of one. With no extra bits a
/// packed block is its eight bits T7..T0 (figure C.5), and C.2.12's decoding gives:
/// T = 1 (C = 1): t0 = 1; T = 2 (C = 2): t0 = 2; T = 0x80: t4 = 1. The last block has only
/// T1 T0; T = 2 again gives t0 = 2.
#[test]
fn luminance_with_trit_weights() {
    let weights = stream_bit(0)       // first block, T0: weight 0 is trit 1
        | stream_bit(8 + 1)           // second block, T1: weight 5 is trit 2
        | stream_bit(16 + 7)          // third block, T7: weight 14 is trit 1
        | stream_bit(24 + 1); //         last block, T1: weight 15 is trit 2
    let block = GRID_4X4 | WEIGHTS_0_TO_2 | byte_values(&[0, 255], COLOR_AT) | weights;
    // Weight 32 with endpoints 0 and 255: C = floor((65535 * 32 + 32) / 64) = 32768, exactly
    // half of 65536; 255 * 0.5 + 0.5 = 128. Weight 64: C = 65535, which is 1.0.
    let (o, h, f) = (grey(0), grey(128), grey(255));
    let expected = [h, o, o, o, o, f, o, o, o, o, o, o, o, o, h, f];
    assert_eq!(legal(block, F::Astc2d4x4Unorm), expected);
}

/// Weights stored as quints (range 0..4), which unquantize to 0, 16, 32, 47 + 1 and 63 + 1
/// (table C.2.25).
///
/// Sixteen weights are five packed blocks of three and one of one. With no extra bits a
/// packed block is its seven bits Q6..Q0 (figure C.6). By C.2.12, for Q below 5 (Q[2:1] not
/// 11, C[2:0] not 101): q0 = Q. Q = 0x48: q2 = Q[6:5] = 2, C = Q[4:0] = 8, q1 = C[4:3] = 1,
/// q0 = 0.
#[test]
fn luminance_with_quint_weights() {
    let weights = stream_bit(0)                     // block 0, Q = 1: weight 0 is quint 1
        | stream_bit(7 + 1)                         // block 1, Q = 2: weight 3 is quint 2
        | stream_bit(14) | stream_bit(14 + 1)       // block 2, Q = 3: weight 6 is quint 3
        | stream_bit(21 + 2)                        // block 3, Q = 4: weight 9 is quint 4
        | stream_bit(28 + 3) | stream_bit(28 + 6)   // block 4, Q = 0x48: weights 13, 14 are 1, 2
        | stream_bit(35 + 2); //                       last block, Q = 4: weight 15 is quint 4
    let block = GRID_4X4 | WEIGHTS_0_TO_4 | byte_values(&[0, 255], COLOR_AT) | weights;
    // C = floor((65535 * i + 32) / 64) for i = 16, 32, 48, 64: 16384, 32768, 49151, 65535;
    // as bytes 64, 128, 191, 255.
    let level = [grey(0), grey(64), grey(128), grey(191), grey(255)];
    // Each texel's quint: weight 0, 16, 32, 48 or 64.
    let quints = [1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 0, 1, 2, 4];
    let expected = quints.map(|quint: usize| level[quint]);
    assert_eq!(legal(block, F::Astc2d4x4Unorm), expected);
}

/// Five-bit weights (range 0..31), on a grid the size of the footprint and on one smaller
/// than it (weight infill, C.2.18).
#[test]
fn five_bit_weights_and_infill() {
    let stored: Vec<u32> = (0..16).map(|i| 2 * i).collect();
    let block =
        GRID_4X4 | WEIGHTS_0_TO_31 | byte_values(&[0, 255], COLOR_AT) | plain_weights(&stored, 5);
    // Five bits replicate to six as (v << 1) | (v >> 4), then values above 32 gain 1: the
    // stored 0, 2, .., 30 become 0, 4, 8, 12, 16, 20, 24, 28, 34, 38, 42, 46, 50, 54, 58, 62.
    // On a 4 by 4 footprint the infill is the identity (Ds = 342: gs = 0, 16, 32, 48), so
    // each texel is floor(255 * floor((65535 * i + 32) / 64) / 65536 + 0.5):
    let on_4x4 = [
        0, 16, 32, 48, 64, 80, 96, 112, 135, 151, 167, 183, 199, 215, 231, 247,
    ];
    assert_eq!(legal(block, F::Astc2d4x4Unorm), on_4x4.map(grey).to_vec());
    // On an 8 by 8 footprint, Ds = Dt = floor((1024 + 4) / 7) = 146, and each texel's weight
    // is the bilinear mix of four stored weights that C.2.18 spells out. The first row's
    // weights come to 0, 2, 4, 5, 7, 9, 10, 12 and the last texel's to 62.
    #[rustfmt::skip]
    let on_8x8 = [
        0, 8, 16, 20, 28, 36, 40, 48,
        28, 36, 44, 48, 56, 64, 68, 76,
        56, 64, 72, 76, 84, 92, 96, 104,
        88, 92, 100, 108, 112, 120, 128, 135,
        112, 120, 128, 135, 139, 147, 155, 159,
        143, 151, 159, 163, 171, 179, 183, 191,
        171, 179, 187, 191, 199, 207, 211, 219,
        199, 207, 215, 219, 227, 235, 239, 247,
    ];
    assert_eq!(legal(block, F::Astc2d8x8Unorm), on_8x8.map(grey).to_vec());
}

/// Two weight planes (C.2.20): the colour component selector picks the one channel that
/// uses the second plane. Mode 4 is luminance and alpha, direct: `e0 = (v0, v0, v0, v2)`,
/// `e1 = (v1, v1, v1, v3)`.
#[test]
fn dual_plane_selects_one_channel() {
    // Each grid point stores plane 0's weight, then plane 1's (C.2.16). Plane 0 cycles
    // through the four weights; plane 1 is 3 (64) for the first eight texels and 0 after.
    let mut weights = Vec::new();
    for texel in 0..16 {
        weights.push(texel % 4);
        weights.push(if texel < 8 { 3 } else { 0 });
    }
    // 64 weight bits; the selector sits directly below them, at bits 63..62 (C.2.20).
    let block = |selector: u128| {
        GRID_4X4
            | WEIGHTS_0_TO_3
            | DUAL_PLANE
            | (4 << CEM_AT)
            | byte_values(&[0, 255, 255, 0], COLOR_AT)
            | (selector << 62)
            | plain_weights(&weights, 2)
    };
    // Luminance runs 0 to 255 and alpha 255 to 0. On plane 0 a channel follows the ramp; on
    // plane 1 it is at its second endpoint for the first eight texels, its first after.
    let ramp = |texel: usize| RAMP[texel % 4];
    let falling = |texel: usize| RAMP[3 - texel % 4];
    let step_up = |texel: usize| if texel < 8 { 255 } else { 0 };
    let step_down = |texel: usize| if texel < 8 { 0 } else { 255 };
    let expected = |selector: usize| -> Vec<[u8; 4]> {
        (0..16)
            .map(|texel| {
                let mut color = [ramp(texel), ramp(texel), ramp(texel), falling(texel)];
                color[selector] = if selector == 3 {
                    step_down(texel)
                } else {
                    step_up(texel)
                };
                color
            })
            .collect()
    };
    for (selector, bits) in [(0, 0), (1, 1), (2, 2), (3, 3)] {
        assert_eq!(
            legal(block(bits), F::Astc2d4x4Unorm),
            expected(selector),
            "selector {selector}"
        );
    }
}

/// An endpoint mode, its 8-bit values, and the two endpoints they give.
type EndpointCase = (u128, &'static [u8], [u8; 4], [u8; 4]);

/// Each LDR endpoint mode of C.2.14, with weights that give the first endpoint at even texels
/// and the second at odd ones. The values are 8 bits each, so the endpoints are C.2.14's
/// arithmetic on the bytes written here.
#[test]
fn ldr_endpoint_modes() {
    let cases: [EndpointCase; 13] = [
        // Mode 1, luminance base and offset: L0 = (v0 >> 2) | (v1 & 0xC0), L1 = L0 + (v1 &
        // 0x3F). (75 >> 2) | (213 & 0xC0) = 18 | 192 = 210; 210 + 21 = 231.
        (1, &[75, 213], grey(210), grey(231)),
        // The same with L1 = 255 + 63 clamped to 255.
        (1, &[255, 255], grey(255), grey(255)),
        // Mode 4, luminance and alpha, direct.
        (4, &[10, 20, 30, 40], [10, 10, 10, 30], [20, 20, 20, 40]),
        // Mode 5, luminance and alpha, base and offset. bit_transfer_signed(v1, v0) with
        // (140, 200): v0 = (200 >> 1) | (140 & 0x80) = 228, v1 = (140 >> 1) & 0x3F = 6.
        // With (126, 10): v2 = 5, v3 = 63, which has bit 5 set, so 63 - 64 = -1.
        // e0 = (228, 228, 228, 5), e1 = (234, 234, 234, 4).
        (
            5,
            &[200, 140, 10, 126],
            [228, 228, 228, 5],
            [234, 234, 234, 4],
        ),
        // The same clamped: (0xBF, 255) gives v0 = 255, v1 = 31, and 286 clamps to 255;
        // (0x40, 1) gives v2 = 0, v3 = 0x20 - 0x40 = -32, and -32 clamps to 0.
        (
            5,
            &[255, 0xBF, 1, 0x40],
            [255, 255, 255, 0],
            [255, 255, 255, 0],
        ),
        // Mode 6, RGB base and scale: e0 = (v0 * v3 >> 8, ..), e1 = (v0, v1, v2).
        (
            6,
            &[200, 100, 50, 128],
            [100, 50, 25, 255],
            [200, 100, 50, 255],
        ),
        // Mode 8, RGB direct, with s1 = 120 >= s0 = 90: no swap.
        (
            8,
            &[10, 20, 30, 40, 50, 60],
            [10, 30, 50, 255],
            [20, 40, 60, 255],
        ),
        // The same with s1 < s0: the endpoints swap and blue_contract applies,
        // ((r + b) >> 1, (g + b) >> 1, b): (10, 30, 50) -> (30, 40, 50), (20, 40, 60) ->
        // (40, 50, 60).
        (
            8,
            &[20, 10, 40, 30, 60, 50],
            [30, 40, 50, 255],
            [40, 50, 60, 255],
        ),
        // Mode 9, RGB base and offset. bit_transfer_signed gives (v0, v1) = (50, 8) from
        // (100, 16), (v2, v3) = (178, 8) from (101, 144), and (v4, v5) = (3, -2) from
        // (6, 124). The offsets sum to 14 >= 0: e0 = (50, 178, 3), e1 = (58, 186, 1).
        (
            9,
            &[100, 16, 101, 144, 6, 124],
            [50, 178, 3, 255],
            [58, 186, 1, 255],
        ),
        // The same with every offset -2 (from 124) on bases 50, 30, 10 (from 100, 60, 20):
        // the sum is negative, so e0 = blue_contract(48, 28, 8) = (28, 18, 8) and
        // e1 = blue_contract(50, 30, 10) = (30, 20, 10).
        (
            9,
            &[100, 124, 60, 124, 20, 124],
            [28, 18, 8, 255],
            [30, 20, 10, 255],
        ),
        // Mode 10, RGB base and scale with two alphas.
        (
            10,
            &[200, 100, 50, 128, 7, 250],
            [100, 50, 25, 7],
            [200, 100, 50, 250],
        ),
        // Mode 12, RGBA direct, swapped: e0 = blue_contract(v1, v3, v5, v7), e1 =
        // blue_contract(v0, v2, v4, v6).
        (
            12,
            &[20, 10, 40, 30, 60, 50, 70, 80],
            [30, 40, 50, 80],
            [40, 50, 60, 70],
        ),
        // Mode 13, RGBA base and offset, as mode 9's first case with alpha (228, 6) from
        // (200, 140): e0 alpha 228, e1 alpha 234.
        (
            13,
            &[100, 16, 101, 144, 6, 124, 200, 140],
            [50, 178, 3, 228],
            [58, 186, 1, 234],
        ),
    ];
    for (mode, values, e0, e1) in cases {
        let texels = legal(two_bit_block(mode, values, &ENDS), F::Astc2d4x4Unorm);
        let expected: Vec<_> = (0..16).map(|i| if i % 2 == 0 { e0 } else { e1 }).collect();
        assert_eq!(texels, expected, "mode {mode} with {values:?}");
    }
}

/// Mode 12 without the swap, and mode 13 with it: the alpha values travel with their colours.
#[test]
fn rgba_modes_keep_alpha_with_its_endpoint() {
    let texels = legal(
        two_bit_block(12, &[10, 20, 30, 40, 50, 60, 70, 80], &ENDS),
        F::Astc2d4x4Unorm,
    );
    assert_eq!(texels[0], [10, 30, 50, 70]);
    assert_eq!(texels[1], [20, 40, 60, 80]);
    // Mode 13 with every colour offset -2 (see mode 9) and alpha (228, 6): the swap gives
    // e0 = blue_contract(48, 28, 8, 228 + 6), e1 = blue_contract(50, 30, 10, 228).
    let texels = legal(
        two_bit_block(13, &[100, 124, 60, 124, 20, 124, 200, 140], &ENDS),
        F::Astc2d4x4Unorm,
    );
    assert_eq!(texels[0], [28, 18, 8, 234]);
    assert_eq!(texels[1], [30, 20, 10, 228]);
}

/// A multi-partition block whose partitions share one endpoint mode (table C.2.11, selector
/// 00), with flat colours 10, 20, 30, 40 for partitions 0 to 3, so each texel shows its
/// partition. The patterns are C.2.21's `select_partition` worked through for these seeds.
#[test]
fn partition_patterns() {
    // Colour data after the 29 configuration bits: 128 - 29 - 32 = 67 bits, enough for
    // eight byte values.
    let block = |partitions: usize, seed: u128| {
        let count_field = [0, 1, 2, 3][partitions - 1];
        GRID_4X4
            | WEIGHTS_0_TO_3
            | (count_field << PARTITIONS_AT)
            | (seed << PARTITION_INDEX_AT)
            | byte_values(
                &[10, 10, 20, 20, 30, 30, 40, 40][..2 * partitions],
                MULTI_COLOR_AT,
            )
    };
    let shown = |texels: Vec<[u8; 4]>, width: usize| -> Vec<String> {
        texels
            .chunks(width)
            .map(|row| {
                row.iter()
                    .map(|texel| {
                        assert_eq!(*texel, grey(texel[0]));
                        char::from(b'0' + texel[0] / 10 - 1)
                    })
                    .collect()
            })
            .collect()
    };
    // 4 by 4 is a small block (fewer than 31 texels), so x and y are doubled.
    assert_eq!(
        shown(legal(block(2, 613), F::Astc2d4x4Unorm), 4),
        ["1111", "0000", "0000", "0000"]
    );
    assert_eq!(
        shown(legal(block(3, 613), F::Astc2d4x4Unorm), 4),
        ["2101", "2201", "2101", "1101"]
    );
    assert_eq!(
        shown(legal(block(4, 613), F::Astc2d4x4Unorm), 4),
        ["3223", "0000", "3233", "2111"]
    );
    // 6 by 5 has 30 texels and is small; 6 by 6 has 36 and is not.
    assert_eq!(
        shown(legal(block(2, 999), F::Astc2d6x5Unorm), 6),
        ["000111", "100111", "100111", "100111", "110111"]
    );
    assert_eq!(
        shown(legal(block(2, 999), F::Astc2d6x6Unorm), 6),
        ["000000", "000000", "100000", "100000", "110000", "110000"]
    );
    assert_eq!(
        shown(legal(block(3, 613), F::Astc2d8x8Unorm), 8),
        [
            "22110210", "22212110", "22220111", "22200111", "22100111", "21100111", "11100111",
            "11000111"
        ]
    );
    assert_eq!(
        shown(legal(block(4, 1023), F::Astc2d12x12Unorm), 12),
        [
            "333333111333",
            "333333311333",
            "333333311333",
            "333333331333",
            "222222221230",
            "222222333000",
            "223333311000",
            "333330111000",
            "333000111000",
            "300000011000",
            "000000001000",
            "000000000022"
        ]
    );
}

/// Two partitions with their own endpoints, sharing mode 0: partition 0 runs 0 to 255,
/// partition 1 runs 255 to 0, and every weight is 1 (21 of 64).
#[test]
fn two_partitions_have_their_own_endpoints() {
    let block = GRID_4X4
        | WEIGHTS_0_TO_3
        | (1 << PARTITIONS_AT)
        | (613 << PARTITION_INDEX_AT)
        | byte_values(&[0, 255, 255, 0], MULTI_COLOR_AT)
        | plain_weights(&[1; 16], 2);
    // Partition 0: the ramp's 84. Partition 1: C = floor((65535 * 43 + 32) / 64) = 44031,
    // the ramp's 171. Seed 613 puts the first row in partition 1 (see `partition_patterns`).
    let mut expected = vec![grey(84); 16];
    expected[..4].fill(grey(171));
    assert_eq!(legal(block, F::Astc2d4x4Unorm), expected);
}

/// Two partitions with different endpoint modes (figure C.4). The selector 01 means classes 0
/// or 1; C0 = 0 and M0 = 0 make partition 0 mode 0, and C1 = 1 with M1 = 0 makes partition 1
/// mode 4. M1's two bits are directly below the weight data.
#[test]
fn two_partitions_with_different_modes() {
    let selector = 0b01;
    let c1 = 1 << 3; // figure C.4: bit 26 of the block, bit 3 of the field at bit 23
    let block = GRID_4X4
        | WEIGHTS_0_TO_3
        | (1 << PARTITIONS_AT)
        | (613 << PARTITION_INDEX_AT)
        | ((selector | c1) << MULTI_CEM_AT)
        // Partition 0: luminance 0 to 255. Partition 1: luminance 255 to 0, alpha 40 to 200.
        // 128 - 29 - 2 - 32 = 65 colour bits hold the six values as bytes.
        | byte_values(&[0, 255, 255, 0, 40, 200], MULTI_COLOR_AT)
        | plain_weights(&ENDS, 2);
    let texels = legal(block, F::Astc2d4x4Unorm);
    let (dark, light) = (grey(0), grey(255));
    let (first, second) = ([255, 255, 255, 40], [0, 0, 0, 200]);
    #[rustfmt::skip]
    let expected = [
        first, second, first, second,
        dark, light, dark, light,
        dark, light, dark, light,
        dark, light, dark, light,
    ];
    assert_eq!(texels, expected);
}

/// C.2.19: with sRGB conversion the red, green and blue endpoints are widened with 0x80 and
/// the result is the top 8 bits; alpha is always widened by replication.
#[test]
fn srgb_interpolation_differs_from_linear() {
    // Mode 4 with luminance and alpha both running 2 to 255, and the trit weights of
    // `luminance_with_trit_weights`: weight 32 at texels 0 and 14, 64 at 5 and 15, 0 elsewhere.
    let weights = stream_bit(0) | stream_bit(9) | stream_bit(23) | stream_bit(25);
    let block = GRID_4X4
        | WEIGHTS_0_TO_2
        | (4 << CEM_AT)
        | byte_values(&[2, 255, 2, 255], COLOR_AT)
        | weights;
    // Linear, weight 32: C0 = 0x0202 = 514, C1 = 65535;
    // C = floor((514 * 32 + 65535 * 32 + 32) / 64) = 33025;
    // floor(255 * 33025 / 65536 + 0.5) = floor(128.99..) = 128.
    let linear = legal(block, F::Astc2d4x4Unorm);
    assert_eq!(linear[0], [128, 128, 128, 128]);
    assert_eq!(linear[1], [2, 2, 2, 2]);
    assert_eq!(linear[5], [255, 255, 255, 255]);
    // sRGB, weight 32: C0 = 0x0280 = 640, C1 = 0xFF80 = 65408;
    // C = floor((640 * 32 + 65408 * 32 + 32) / 64) = 33024 = 0x8100; top byte 0x81 = 129.
    // Alpha stays on the linear path: 128.
    let srgb = legal(block, F::Astc2d4x4Srgb);
    assert_eq!(srgb[0], [129, 129, 129, 128]);
    assert_eq!(srgb[14], [129, 129, 129, 128]);
    assert_eq!(srgb[1], [2, 2, 2, 2]);
    assert_eq!(srgb[5], [255, 255, 255, 255]);
    assert_eq!(srgb[15], [255, 255, 255, 255]);
}

/// HDR endpoint modes in a linear format are decoded and brought to bytes: the FP16 value
/// C.2.19 gives, clamped to 0..1, times 255, rounded half up.
#[test]
fn hdr_endpoint_modes_in_a_linear_format() {
    let ramp = [0, 1, 2, 3, 3, 2, 1, 0, 0, 0, 3, 3, 1, 2, 1, 2];
    // Mode 2, HDR luminance, large range: v1 >= v0, so y0 = 0x70 << 4 = 0x700 and
    // y1 = 0x78 << 4 = 0x780, alpha 0x780. By C.2.19 a 12-bit value v at weight 0 is
    // C = v << 4: 0x7000 has E = 14, M = 0, so Cf = 0x3800 = 0.5; 0x7800 gives 0x3C00 = 1.0.
    // In between (weights 21 and 43) C = 0x7220 and 0x7460: M = 544 and 1120, Mt = 4M - 512,
    // Cf = 0x3910 and 0x3A70, which are 0.6289.. and 0.8007..: 161 and 205 as bytes.
    let texels = legal(two_bit_block(2, &[0x70, 0x78], &ramp), F::Astc2d4x4Unorm);
    let levels = [128, 161, 205, 255];
    let expected: Vec<_> = ramp.iter().map(|w| grey(levels[*w as usize])).collect();
    assert_eq!(texels, expected);

    // Mode 3, HDR luminance, small range, with v0's top bit set: y0 = ((v1 & 0xE0) << 4) |
    // ((v0 & 0x7F) << 2) = 0x600 | 0x100 = 0x700 and d = (v1 & 0x1F) << 2 = 0x7C, so
    // y1 = 0x77C. Cf = 0x3800, 0x3905, 0x3A5A, 0x3BD8: 128, 160, 202, 250.
    let texels = legal(two_bit_block(3, &[0xC0, 0x7F], &ramp), F::Astc2d4x4Unorm);
    let levels = [128, 160, 202, 250];
    let expected: Vec<_> = ramp.iter().map(|w| grey(levels[*w as usize])).collect();
    assert_eq!(texels, expected);

    // Mode 11, HDR RGB, with the top bits of v4 and v5 both set (major component 3): the
    // direct form, e0 = (v0 << 4, v2 << 4, (v4 & 0x7F) << 5), e1 = (v1 << 4, v3 << 4,
    // (v5 & 0x7F) << 5), alpha 0x780. 0x780 is 1.0, 0x700 is 0.5, 0 is 0.
    let rgb = [0x78, 0x70, 0x70, 0x78, 0x80, 0xBC];
    let texels = legal(two_bit_block(11, &rgb, &ENDS), F::Astc2d4x4Unorm);
    assert_eq!(texels[0], [255, 128, 0, 255]);
    assert_eq!(texels[1], [128, 255, 255, 255]);

    // Mode 14 adds LDR alphas v6 and v7, interpolated as bytes.
    let mut values = rgb.to_vec();
    values.extend([40, 200]);
    let texels = legal(two_bit_block(14, &values, &ENDS), F::Astc2d4x4Unorm);
    assert_eq!(texels[0], [255, 128, 0, 40]);
    assert_eq!(texels[1], [128, 255, 255, 200]);

    // Mode 15 adds HDR alphas; with the top bits of v6 and v7 set (mode 3) they are direct:
    // (v6 & 0x7F) << 5 = 0x780 and (v7 & 0x7F) << 5 = 0x700.
    let mut values = rgb.to_vec();
    values.extend([0x80 | 0x3C, 0x80 | 0x38]);
    let texels = legal(two_bit_block(15, &values, &ENDS), F::Astc2d4x4Unorm);
    assert_eq!(texels[0], [255, 128, 0, 255]);
    assert_eq!(texels[1], [128, 255, 255, 128]);
}

/// In an sRGB format the decoder is in the specification's LDR mode: the texels of a
/// partition with an HDR endpoint mode are the error colour, and the block is still a legal
/// encoding (C.2.24's closing note).
#[test]
fn hdr_in_an_srgb_format_is_the_error_colour() {
    // Each HDR mode with the number of values it takes (table C.2.17).
    for (mode, count) in [(2, 2), (3, 2), (7, 4), (11, 6), (14, 8), (15, 8)] {
        let values = [0x70, 0x78, 0x70, 0x78, 0x80, 0xBC, 40, 200];
        let block = two_bit_block(mode, &values[..count], &ENDS);
        assert_eq!(
            legal(block, F::Astc2d4x4Srgb),
            vec![ERROR_COLOR; 16],
            "mode {mode}"
        );
    }
    // Only the HDR partition's texels: partition 0 is mode 0 (LDR), partition 1 is mode 7
    // (class 1, M1 = 3, whose two bits are directly below the 32 weight bits).
    let block = GRID_4X4
        | WEIGHTS_0_TO_3
        | (1 << PARTITIONS_AT)
        | (613 << PARTITION_INDEX_AT)
        | ((0b01 | (1 << 3)) << MULTI_CEM_AT)
        | (0b11 << 94)
        | byte_values(&[9, 9, 0, 0, 0, 0], MULTI_COLOR_AT);
    let texels = legal(block, F::Astc2d4x4Srgb);
    let mut expected = vec![grey(9); 16];
    expected[..4].fill(ERROR_COLOR);
    assert_eq!(texels, expected);
}

/// Void-extent blocks (C.2.23, figure C.7): bits 8..0 are 111111100, bit 9 says HDR, bits 11
/// and 10 are 1, four 13-bit coordinates follow, and the colour is in the top 64 bits as
/// red, green, blue, alpha.
const VOID_EXTENT: u128 = 0x1FC | (0b11 << 10);
/// All four coordinates all ones: no extent.
const NO_EXTENT: u128 = ((1 << 52) - 1) << 12;
const HDR: u128 = 1 << 9;

fn color16(rgba: [u16; 4]) -> u128 {
    (u128::from(rgba[0]) << 64)
        | (u128::from(rgba[1]) << 80)
        | (u128::from(rgba[2]) << 96)
        | (u128::from(rgba[3]) << 112)
}

/// An LDR void-extent block is one colour, the top 8 bits of each UNORM16 component, on every
/// footprint and in both encodings.
#[test]
fn ldr_void_extent() {
    let block = VOID_EXTENT | NO_EXTENT | color16([0x1234, 0x00FF, 0xFF00, 0x8081]);
    for format in astc_formats() {
        assert_eq!(
            legal(block, format),
            vec![[0x12, 0x00, 0xFF, 0x80]; texel_count(format)],
            "{format}"
        );
    }
}

/// An HDR void-extent block holds FP16 values. In a linear format each is clamped to 0..1 and
/// rounded to a byte; a NaN becomes 0.
#[test]
fn hdr_void_extent_in_a_linear_format() {
    let cases: [([u16; 4], [u8; 4]); 4] = [
        // 1.0, 0.5, 0.0 and +infinity.
        ([0x3C00, 0x3800, 0x0000, 0x7C00], [255, 128, 0, 255]),
        // -infinity, a NaN, -1.0 and 0.25 (63.75 + 0.5 rounds to 64).
        ([0xFC00, 0x7E00, 0xBC00, 0x3400], [0, 0, 0, 64]),
        // Just under 1.0 (254.87.. + 0.5), the smallest subnormal, -0.0 and the largest
        // finite value.
        ([0x3BFF, 0x0001, 0x8000, 0x7BFF], [255, 0, 0, 255]),
        // 2^-8 (0.996.. + 0.5 rounds to 1), just under it (0.995.. + 0.5 rounds to 1 too),
        // 2^-9 (0.498.. + 0.5 rounds to 0) and a negative NaN.
        ([0x1C00, 0x1BFF, 0x1800, 0xFE00], [1, 1, 0, 0]),
    ];
    for (halves, bytes) in cases {
        let block = VOID_EXTENT | HDR | NO_EXTENT | color16(halves);
        for format in [F::Astc2d4x4Unorm, F::Astc2d12x12Unorm] {
            assert_eq!(
                legal(block, format),
                vec![bytes; texel_count(format)],
                "{halves:x?}"
            );
        }
    }
}

/// An HDR void-extent block in an sRGB format: the error colour (C.2.23), though the
/// encoding is legal.
#[test]
fn hdr_void_extent_in_an_srgb_format() {
    let block = VOID_EXTENT | HDR | NO_EXTENT | color16([0x3C00, 0x3800, 0, 0x3C00]);
    assert_eq!(legal(block, F::Astc2d6x6Srgb), vec![ERROR_COLOR; 36]);
}

/// Colour bits past the end of the colour values are not read (C.2.12; module docs, "Where
/// this differs from Eden"), and the colour values of a block can be trits.
///
/// One partition with mode 12 (RGBA, direct: `e0 = (v0, v2, v4, v6)` and
/// `e1 = (v1, v3, v5, v7)` when `v1 + v3 + v5 >= v0 + v2 + v4`), on the 12 by 12 footprint
/// with a weight grid 4 wide and 11 high of two-bit weights. Table C.2.8's third row (bits
/// 3..2 = 10) is a grid A+2 wide and B+8 high: A = 2 in bits 6..5, B = 3 in bits 8..7. The 44
/// weights take 88 bits, which leaves 128 - 17 - 88 = 23 bits of colour data. Eight values
/// fit them in the range 0..5 at most (a trit and a bit each, 8 + ceil(8 * 8 / 5) = 21 bits;
/// the next range, 0..7, takes 24), so colour bits 21 and 22, block bits 38 and 39, are
/// unused.
///
/// Range 0..5 unquantizes to 0, 51, 102 for bit 0 with trit 0, 1, 2, and to 255, 204, 153
/// for bit 1 (C.2.13: A is the bit nine times, C = 204, `T = trit * C`, `T ^= A`,
/// `(A & 0x80) | (T >> 2)`).
///
/// The first trit block holds v0 to v4 with every trit 0 and the bits 0, 1, 1, 1, 1 (m0 at
/// colour bit 0, m1 at 3, m2 at 6, m3 at 8, m4 at 11): 0, 255, 255, 255, 255. The second
/// starts at colour bit 13 and holds v5 to v7: m0 = 1 at 13, T[4:0] = 11100 (T2 at 17, T3 at
/// 18, T4 at 20), m1 = 0 at 16 and m2 = 1 at 19. Its T[7:5] are past the end of the values,
/// so they are 0: T[4:2] = 111 gives C = { T[7:5], T[1:0] } = 00000 and trits 0, 0, 0, which
/// makes v5 = 255, v6 = 0 and v7 = 255. So `e0 = (0, 255, 255, 0)` and
/// `e1 = (255, 255, 255, 255)`.
///
/// T5 would be colour bit 22, an unused bit. A decoder that read it as set would get
/// C = 00100, the trits 0, 1, 0, v6 = 51 and `e0 = (0, 255, 255, 51)`.
#[test]
fn unused_colour_bits_are_not_read() {
    const GRID_4X11: u128 = (3 << 7) | (2 << 5) | (0b10 << 2);
    const RGBA_DIRECT: u128 = 12;
    let first = (1 << 3) | (1 << 6) | (1 << 8) | (1 << 11);
    let second = (1 << 13) | (1 << 17) | (1 << 18) | (1 << 19) | (1 << 20);
    let colors: u128 = (first | second) << COLOR_AT;
    let block = GRID_4X11 | WEIGHTS_0_TO_3 | (RGBA_DIRECT << CEM_AT) | colors;
    let unused: [u128; 2] = [1 << (COLOR_AT + 21), 1 << (COLOR_AT + 22)];
    for extra in [0, unused[0], unused[1], unused[0] | unused[1]] {
        // Every weight 0: the first endpoint throughout.
        assert_eq!(
            legal(block | extra, F::Astc2d12x12Unorm),
            vec![[0, 255, 255, 0]; 144],
            "{extra:#x}"
        );
        // Every weight 3, which is 64 after unquantization and infill: the second.
        assert_eq!(
            legal(
                block | extra | plain_weights(&[3; 44], 2),
                F::Astc2d12x12Unorm
            ),
            vec![[255; 4]; 144],
            "{extra:#x}"
        );
    }
}

/// C.2.23 makes a low coordinate at or above its high one illegal. The LDR profile lets a
/// decoder skip the check, and this one does, as Eden does; for the HDR block below that is
/// Eden's behaviour only (module docs).
#[test]
fn void_extent_coordinates_are_not_checked() {
    // Minimum S = 5, maximum S = 3, minimum T = 7, maximum T = 7.
    let coordinates: u128 = (5 << 12) | (3 << 25) | (7 << 38) | (7 << 51);
    let block = VOID_EXTENT | coordinates | color16([0xFFFF, 0, 0x8000, 0xFFFF]);
    assert_eq!(
        legal(block, F::Astc2d5x5Unorm),
        vec![[0xFF, 0, 0x80, 0xFF]; 25]
    );
    let (legal, _) = decode_one(block | HDR, F::Astc2d5x5Unorm);
    assert!(legal);
}
