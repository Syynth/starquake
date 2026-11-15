//! Endpoint unquantization and the endpoint modes, against values worked by hand from the
//! specification's tables and procedures (C.2.13 to C.2.15).

use super::{
    Endpoints, bit_transfer_signed, blue_contract, decode, hdr_alpha, hdr_base_scale, hdr_direct,
    unquantize_color, value_count,
};
use crate::astc::integer::{Digit, Encoding, Value};

mod hdr_tables;

const fn encoding(digit: Digit, bits: u32) -> Encoding {
    Encoding { digit, bits }
}

/// Table C.2.17's value counts.
#[test]
fn values_per_mode() {
    let counts = [2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8, 8];
    for (mode, count) in (0..).zip(counts) {
        assert_eq!(value_count(mode), count, "mode {mode}");
    }
}

/// The two smallest colour ranges, by table C.2.16's arithmetic: `T = D * C + B; T ^= A;
/// T = (A & 0x80) | (T >> 2)`, with A the lowest bit nine times over.
#[test]
fn colour_unquantization_by_hand() {
    // 0..5: a trit and one bit, B = 0, C = 204. Bit 0: 0, 204 >> 2, 408 >> 2. Bit 1 (A = 511):
    // 0x80 | (511 >> 2), 0x80 | ((204 ^ 511) >> 2), 0x80 | ((408 ^ 511) >> 2).
    let six = [
        (0, 0, 0),
        (1, 0, 51),
        (2, 0, 102),
        (0, 1, 255),
        (1, 1, 204),
        (2, 1, 153),
    ];
    for (digit, bits, expected) in six {
        assert_eq!(
            unquantize_color(encoding(Digit::Trit, 1), Value { bits, digit }),
            Some(expected),
            "trit {digit}, bit {bits}"
        );
    }
    // 0..9: a quint and one bit, B = 0, C = 113.
    let ten = [
        (0, 0, 0),
        (1, 0, 28),
        (2, 0, 56),
        (3, 0, 84),
        (4, 0, 113),
        (0, 1, 255),
        (1, 1, 227),
        (2, 1, 199),
        (3, 1, 171),
        (4, 1, 142),
    ];
    for (digit, bits, expected) in ten {
        assert_eq!(
            unquantize_color(encoding(Digit::Quint, 1), Value { bits, digit }),
            Some(expected),
            "quint {digit}, bit {bits}"
        );
    }
    // 0..11: a trit and two bits "ba", B = b000b0bb0, C = 93. Value 0b11 with trit 1:
    // A = 511, B = 0b100010110 = 278, T = (93 + 278) ^ 511 = 140, 0x80 | 35 = 163.
    assert_eq!(
        unquantize_color(encoding(Digit::Trit, 2), Value { bits: 3, digit: 1 }),
        Some(163)
    );
    // 0..19: a quint and two bits, B = b0000bb00, C = 54. Value 0b10 with quint 4:
    // A = 0, B = 0b100001100 = 268, T = 216 + 268 = 484, 484 >> 2 = 121.
    assert_eq!(
        unquantize_color(encoding(Digit::Quint, 2), Value { bits: 2, digit: 4 }),
        Some(121)
    );
    // Plain bits replicate from the top: 3 bits "101" become "10110110".
    assert_eq!(
        unquantize_color(encoding(Digit::None, 3), Value { bits: 5, digit: 0 }),
        Some(0b1011_0110)
    );
    assert_eq!(
        unquantize_color(
            encoding(Digit::None, 8),
            Value {
                bits: 0xA7,
                digit: 0
            }
        ),
        Some(0xA7)
    );
}

/// Every colour range unquantizes to that many distinct values, from 0 to 255, mirrored about
/// the middle (v and 255 - v), and spread evenly: the i-th smallest is within 1.5 of
/// `255 * i / (count - 1)` (the bit manipulation is an approximation of that scaling; the
/// furthest it strays is 1.08, in the range 0..39).
#[test]
fn every_colour_range_is_an_even_spread() {
    let mut ranges = Vec::new();
    for bits in 1..=8 {
        ranges.push((encoding(Digit::None, bits), 1));
    }
    for bits in 1..=6 {
        ranges.push((encoding(Digit::Trit, bits), 3));
    }
    for bits in 1..=5 {
        ranges.push((encoding(Digit::Quint, bits), 5));
    }
    assert_eq!(ranges.len(), 19);
    for (encoding, digits) in ranges {
        let mut values = Vec::new();
        for digit in 0..digits {
            for bits in 0..(1 << encoding.bits) {
                values.push(unquantize_color(encoding, Value { bits, digit }).expect("a range"));
            }
        }
        values.sort_unstable();
        let count = values.len();
        assert_eq!(
            count,
            usize::try_from(digits << encoding.bits).expect("small")
        );
        assert_eq!((values[0], values[count - 1]), (0, 255), "{encoding:?}");
        for (index, value) in values.iter().enumerate() {
            assert_eq!(values[count - 1 - index], 255 - value, "{encoding:?}");
            if let Some(next) = values.get(index + 1) {
                assert!(value < next, "{encoding:?}: {value} twice");
            }
            let index = i32::try_from(index).expect("small");
            let steps = i32::try_from(count - 1).expect("small");
            // |value - 255 * index / steps| <= 1.5, without fractions.
            assert!(
                2 * (value * steps - 255 * index).abs() <= 3 * steps,
                "{encoding:?}: value {index} is {value}"
            );
        }
    }
}

/// Encodings no colour value has (ranges below 0..5, or wider than the table) give `None`.
#[test]
fn encodings_colours_never_have() {
    let value = Value { bits: 0, digit: 0 };
    for encoding in [
        encoding(Digit::None, 0),
        encoding(Digit::None, 9),
        encoding(Digit::Trit, 0),
        encoding(Digit::Trit, 7),
        encoding(Digit::Quint, 0),
        encoding(Digit::Quint, 6),
    ] {
        assert_eq!(unquantize_color(encoding, value), None, "{encoding:?}");
    }
}

#[test]
fn bit_transfer_and_blue_contract() {
    // (a, b) -> (a', b'): b' = (b >> 1) | (a & 0x80); a' = (a >> 1) & 0x3F, less 64 when its
    // bit 5 is set.
    assert_eq!(bit_transfer_signed(140, 200), (6, 228));
    assert_eq!(bit_transfer_signed(126, 10), (-1, 5));
    assert_eq!(bit_transfer_signed(0, 0), (0, 0));
    assert_eq!(bit_transfer_signed(255, 255), (-1, 255));
    assert_eq!(bit_transfer_signed(0x40, 1), (-32, 0));
    assert_eq!(bit_transfer_signed(0x3E, 0xFF), (31, 127));
    assert_eq!(bit_transfer_signed(0x80, 0), (0, 0x80));
    assert_eq!(blue_contract(10, 30, 50, 7), [30, 40, 50, 7]);
    assert_eq!(blue_contract(1, 2, 2, 0), [1, 2, 2, 0]);
}

const fn ldr(e0: [u32; 4], e1: [u32; 4]) -> Endpoints {
    Endpoints {
        e0,
        e1,
        hdr_rgb: false,
        hdr_alpha: false,
    }
}

const fn hdr(e0: [u32; 4], e1: [u32; 4], hdr_alpha: bool) -> Endpoints {
    Endpoints {
        e0,
        e1,
        hdr_rgb: true,
        hdr_alpha,
    }
}

/// The LDR modes whose arithmetic the whole-block vectors also work through, here at the
/// endpoint level, plus the clamp of mode 1.
#[test]
fn ldr_modes() {
    assert_eq!(
        decode(0, &[7, 200]),
        Some(ldr([7, 7, 7, 255], [200, 200, 200, 255]))
    );
    assert_eq!(
        decode(1, &[75, 213]),
        Some(ldr([210, 210, 210, 255], [231, 231, 231, 255]))
    );
    assert_eq!(decode(1, &[255, 255]), Some(ldr([255; 4], [255; 4])));
    assert_eq!(
        decode(6, &[200, 100, 50, 128]),
        Some(ldr([100, 50, 25, 255], [200, 100, 50, 255]))
    );
    // Mode 8 exactly at the swap's threshold: equal sums do not swap.
    assert_eq!(
        decode(8, &[10, 10, 20, 20, 30, 30]),
        Some(ldr([10, 20, 30, 255], [10, 20, 30, 255]))
    );
    assert_eq!(
        decode(8, &[11, 10, 20, 20, 30, 30]),
        Some(ldr([20, 25, 30, 255], [20, 25, 30, 255]))
    );
    // Mode 9 exactly at its threshold: offsets summing to 0 do not swap, to -1 do.
    // (v, 0) transfers to an offset of 0; (v, 126) to -1.
    assert_eq!(
        decode(9, &[100, 0, 60, 0, 20, 0]),
        Some(ldr([50, 30, 10, 255], [50, 30, 10, 255]))
    );
    assert_eq!(
        decode(9, &[100, 126, 60, 0, 20, 0]),
        Some(ldr([29, 20, 10, 255], [30, 20, 10, 255]))
    );
    // Mode 13 at the same threshold, with alpha (200, 0) transferring to (100, 0).
    assert_eq!(
        decode(13, &[100, 0, 60, 0, 20, 0, 200, 0]),
        Some(ldr([50, 30, 10, 100], [50, 30, 10, 100]))
    );
    assert_eq!(
        decode(13, &[100, 126, 60, 0, 20, 0, 200, 2]),
        Some(ldr([29, 20, 10, 101], [30, 20, 10, 100]))
    );
}

/// HDR mode 2 (luminance, large range) and mode 3 (small range), both branches of each
/// (C.2.15). Alpha is 0x780, which is 1.0.
#[test]
fn hdr_luminance_modes() {
    // v1 >= v0: y0 = v0 << 4, y1 = v1 << 4.
    assert_eq!(
        decode(2, &[0x70, 0x78]),
        Some(hdr([0x700, 0x700, 0x700, 0x780], [0x780; 4], true))
    );
    assert_eq!(
        decode(2, &[0x70, 0x70]),
        Some(hdr(
            [0x700, 0x700, 0x700, 0x780],
            [0x700, 0x700, 0x700, 0x780],
            true
        ))
    );
    // v1 < v0: y0 = (v1 << 4) + 8, y1 = (v0 << 4) - 8.
    assert_eq!(
        decode(2, &[116, 23]),
        Some(hdr([376, 376, 376, 0x780], [1848, 1848, 1848, 0x780], true))
    );
    // Mode 3 with v0's top bit set: y0 = ((v1 & 0xE0) << 4) | ((v0 & 0x7F) << 2),
    // d = (v1 & 0x1F) << 2.
    assert_eq!(
        decode(3, &[0xC0, 0x7F]),
        Some(hdr(
            [0x700, 0x700, 0x700, 0x780],
            [0x77C, 0x77C, 0x77C, 0x780],
            true
        ))
    );
    // With it clear: y0 = ((v1 & 0xF0) << 4) | ((v0 & 0x7F) << 1), d = (v1 & 0x0F) << 1.
    assert_eq!(
        decode(3, &[0x45, 0xA7]),
        Some(hdr(
            [0xA8A, 0xA8A, 0xA8A, 0x780],
            [0xA98, 0xA98, 0xA98, 0x780],
            true
        ))
    );
    // y1 clamps at 0xFFF: y0 = 0xF00 | 0xFE, d = 30.
    assert_eq!(
        decode(3, &[0x7F, 0xFF]),
        Some(hdr(
            [0xFFE, 0xFFE, 0xFFE, 0x780],
            [0xFFF, 0xFFF, 0xFFF, 0x780],
            true
        ))
    );
}

/// HDR mode 7 (RGB, base and scale), worked through C.2.15's procedure and table C.2.20.
#[test]
fn hdr_base_scale_by_hand() {
    // Mode 5 (modeval 0xF: v0's top two bits and the top bits of v1 and v2): 7 bits each.
    // red = 0x2A | X4 << 6 = 0x6A, green = 0x03 | X0 << 6 = 0x43, blue = 0x01 | X3 << 5 =
    // 0x21, scale = 0x02 | X6 << 5 = 0x22; all shifted left 5; no differences in mode 5.
    // e1 = (3392, 2144, 1056); e0 = e1 - 1088, with blue clamped at 0.
    assert_eq!(
        hdr_base_scale(0xEA, 0xC3, 0xA1, 0xA2),
        ([2304, 1056, 0], [3392, 2144, 1056])
    );
    // Mode 0 with major component 1 (modeval 0b0100: the top bit of v1): red has 11 bits,
    // 0x15 | X4 << 6 | X0 << 9 = 0x255; green = 0x0A, blue = 0x04, scale = 0x08; all shifted
    // left 1: 1194, 20, 8, 16. Green and blue are differences from red: 1174 and 1186. The
    // major component swaps red and green: e1 = (1174, 1194, 1186), e0 = e1 - 16.
    assert_eq!(
        hdr_base_scale(0x15, 0xCA, 0x04, 0x88),
        ([1158, 1178, 1170], [1174, 1194, 1186])
    );
    // Mode 3 (modeval 3), from a block of the cross-check: red = 12, green = 19 | X1 << 5 =
    // 51, blue = 0, scale = 6 | X6 << 5 | X5 << 6 = 102; shifted left 3: 96, 408, 0, 816.
    // green = 96 - 408 = -312, which clamps to 0; blue = 96 - 0. e0 is all below 0.
    assert_eq!(hdr_base_scale(204, 51, 0, 102), ([0, 0, 0], [96, 0, 96]));
    // Through `decode`, with alpha 1.0.
    assert_eq!(
        decode(7, &[0xEA, 0xC3, 0xA1, 0xA2]),
        Some(hdr([2304, 1056, 0, 0x780], [3392, 2144, 1056, 0x780], true))
    );
}

/// HDR mode 11 (RGB, direct), worked through C.2.15's procedure and table C.2.23, and modes
/// 14 and 15, which take their colours from it.
#[test]
fn hdr_direct_by_hand() {
    // The top bits of v4 and v5 both set: the values are the colours' top bits.
    assert_eq!(
        hdr_direct([0x78, 0x70, 0x70, 0x78, 0x80, 0xBC]),
        ([0x780, 0x700, 0], [0x700, 0x780, 0x780])
    );
    assert_eq!(
        hdr_direct([0x12, 0x34, 0x56, 0x78, 0x80 | 0x38, 0x80 | 0x01]),
        ([0x120, 0x560, 0x700], [0x340, 0x780, 0x020])
    );
    // Mode 4 (the top bit of v3), major component 0: a has 11 bits, b 8, c 6, d 5.
    // a = 0x80 | X4 << 9 = 0x280 (X4 is bit 5 of v4), b0 = 8, b1 = 4, c = 0x10,
    // d0 = 0b00011 = 3, d1 = 0b11110 = -2; all shifted left 1.
    // e1 = (a, a - b0, a - b1) = (1280, 1264, 1272);
    // e0 = (a - c, a - b0 - c - d0, a - b1 - c - d1) = (1248, 1226, 1244).
    // d0 is five bits wide here: bit 5 of v4 is X4 and belongs to a, not to d0. (Eden's
    // `SignExtend` keeps it, giving d0 = 35; module docs.)
    assert_eq!(
        hdr_direct([0x80, 0x10, 0x08, 0x84, 0x23, 0x1E]),
        ([1248, 1226, 1244], [1280, 1264, 1272])
    );
    // The same with major component 2 (the top bit of v5): red and blue swap in both.
    assert_eq!(
        hdr_direct([0x80, 0x10, 0x08, 0x84, 0x23, 0x9E]),
        ([1244, 1226, 1248], [1272, 1264, 1280])
    );
    // And major component 1 (the top bit of v4): red and green swap.
    assert_eq!(
        hdr_direct([0x80, 0x10, 0x08, 0x84, 0xA3, 0x1E]),
        ([1226, 1248, 1244], [1264, 1280, 1272])
    );
    // Mode 0 (no mode bits): a has 9 bits, b 7, c 6, d 7, shifted left 3.
    // a = 0x40 | a8 << 8 = 0x140 (bit 6 of v1), c = 0x05, b0 = 0x02 | X0 << 6 = 0x42,
    // b1 = 0x03, d0 = 0x7F = -1 (seven bits), d1 = 0x01.
    // e1 = (2560, 2560 - 528, 2560 - 24); e0 = (2560 - 40, 2032 - 40 + 8, 2536 - 40 - 8).
    assert_eq!(
        hdr_direct([0x40, 0x45, 0x42, 0x03, 0x7F, 0x01]),
        ([2520, 2000, 2488], [2560, 2032, 2536])
    );
    // Mode 11 has alpha 1.0; mode 14 has byte alphas; mode 15 has HDR alphas.
    let values = [0x78, 0x70, 0x70, 0x78, 0x80, 0xBC, 40, 200];
    assert_eq!(
        decode(11, &values[..6]),
        Some(hdr(
            [0x780, 0x700, 0, 0x780],
            [0x700, 0x780, 0x780, 0x780],
            true
        ))
    );
    assert_eq!(
        decode(14, &values),
        Some(hdr(
            [0x780, 0x700, 0, 40],
            [0x700, 0x780, 0x780, 200],
            false
        ))
    );
    assert_eq!(
        decode(15, &[0x78, 0x70, 0x70, 0x78, 0x80, 0xBC, 0xBC, 0xB8]),
        Some(hdr(
            [0x780, 0x700, 0, 0x780],
            [0x700, 0x780, 0x780, 0x700],
            true
        ))
    );
}

/// HDR mode 15's alphas (table C.2.24): the top bit of v6 and of v7 pick one of four forms.
#[test]
fn hdr_alpha_by_hand() {
    // Both set: direct, (v & 0x7F) << 5.
    assert_eq!(hdr_alpha(0x80 | 0x3C, 0x80 | 0x38), [0x780, 0x700]);
    // Neither: a = v6 | ((v7 << 1) & 0x780), a signed 6-bit delta from v7, both shifted left
    // 4. (0x30, 0x41): a = 0x30 | 0x80 = 0xB0, delta = 1: 2816 and 2832.
    assert_eq!(hdr_alpha(0x30, 0x41), [2816, 2832]);
    // A delta of -1 (0x3F).
    assert_eq!(hdr_alpha(0x30, 0x3F), [768, 752]);
    // The second alpha clamps at both ends: 0 - 512 and 4080 + 496.
    assert_eq!(hdr_alpha(0x00, 0x20), [0, 0]);
    assert_eq!(hdr_alpha(0x7F, 0x5F), [4080, 4095]);
    // Only v6's: a takes two bits from v7, the delta is 5 bits, shifted left 3.
    // (0x10, 0x23): a = 0x10 | ((0x23 << 2) & 0x780) = 0x90, delta = 3: 1152 and 1176.
    assert_eq!(hdr_alpha(0x80 | 0x10, 0x23), [1152, 1176]);
    // Only v7's: a takes three bits, the delta is 4 bits, shifted left 2.
    // (0x10, 0x23): a = 0x10 | ((0x23 << 3) & 0x780) = 0x110, delta = 3: 1088 and 1100.
    assert_eq!(hdr_alpha(0x10, 0x80 | 0x23), [1088, 1100]);
}

/// A mode given the wrong number of values, or a mode past 15, is `None`.
#[test]
fn wrong_value_counts() {
    let values = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    for mode in 0..16 {
        let count = value_count(mode);
        assert!(decode(mode, &values[..count]).is_some(), "mode {mode}");
        assert_eq!(decode(mode, &values[..count - 1]), None, "mode {mode}");
        assert_eq!(decode(mode, &values[..=count]), None, "mode {mode}");
    }
    assert_eq!(decode(16, &values[..2]), None);
    assert_eq!(decode(16, &values[..8]), None);
}

/// Every HDR mode 7 selection (six modes, three major components) against the table.
#[test]
fn hdr_base_scale_table() {
    assert_eq!(hdr_tables::BASE_SCALE.len(), 48);
    for (v, e0, e1) in hdr_tables::BASE_SCALE {
        assert_eq!(hdr_base_scale(v[0], v[1], v[2], v[3]), (e0, e1), "{v:x?}");
    }
}

/// Every HDR mode 11 mode with every major component against the table.
#[test]
fn hdr_direct_table() {
    assert_eq!(hdr_tables::DIRECT.len(), 72);
    for (v, e0, e1) in hdr_tables::DIRECT {
        assert_eq!(hdr_direct(v), (e0, e1), "{v:x?}");
    }
}

/// Every form of HDR mode 15's alphas against the table.
#[test]
fn hdr_alpha_table() {
    assert_eq!(hdr_tables::ALPHA.len(), 24);
    for (v, alphas) in hdr_tables::ALPHA {
        assert_eq!(hdr_alpha(v[0], v[1]), alphas, "{v:x?}");
    }
}
