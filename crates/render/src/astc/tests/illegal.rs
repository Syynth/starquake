//! The illegal encodings of C.2.24, each with a legal neighbour exactly at its limit.
//!
//! Every block here is also given to the second decoder by `decode_one`, which checks that it
//! agrees on which are legal.

use super::{
    CEM_AT, DUAL_PLANE, GRID_4X4, MULTI_CEM_AT, PARTITIONS_AT, WEIGHTS_0_TO_1, WEIGHTS_0_TO_2,
    WEIGHTS_0_TO_3, WEIGHTS_0_TO_5, assert_illegal, astc_formats, legal,
};
use crate::format::PixelFormat;
use PixelFormat as F;

// Table C.2.8's rows, by the grid size each gives. `a` and `b` are the two-bit fields A and B.

/// Row 1: a grid B+4 wide and A+2 high. The weight range goes in bits 1..0 and 4.
const fn wide_4(a: u128, b: u128) -> u128 {
    (b << 7) | (a << 5)
}

/// Row 2: a grid B+8 wide and A+2 high. The weight range goes in bits 1..0 and 4.
const fn wide_8(a: u128, b: u128) -> u128 {
    (b << 7) | (a << 5) | (0b01 << 2)
}

/// Row 6: a grid 12 wide and A+2 high. The weight range goes in bits 3..2 and 4.
const fn wide_12(a: u128) -> u128 {
    a << 5
}

/// Row 10: a grid A+6 wide and B+6 high, with one weight plane and the low-precision ranges.
/// The weight range goes in bits 3..2 and 4.
const fn from_6(a: u128, b: u128) -> u128 {
    (b << 9) | (1 << 8) | (a << 5)
}

// Weight ranges for rows 6 to 10, where R2 R1 are bits 3..2 (rows 1 to 5 have them in bits
// 1..0, as the constants in the parent module).

/// Range 0..1 (R = 010): one bit a weight.
const HIGH_ROWS_0_TO_1: u128 = 0b01 << 2;
/// Range 0..3 (R = 100): two bits a weight.
const HIGH_ROWS_0_TO_3: u128 = 0b10 << 2;

/// The H bit with R = 110: range 0..23, one trit and three bits a weight.
const WEIGHTS_0_TO_23: u128 = (1 << 9) | 0b11;

#[test]
fn reserved_block_modes() {
    for format in astc_formats() {
        // Bits 3..0 all zero (the table's last row), whatever the rest holds.
        assert_illegal(0, format);
        assert_illegal(0x7F0, format);
        assert_illegal(u128::MAX << 4, format);
        // Bits 8..6 all one with bits 1..0 zero, and not the void-extent pattern.
        assert_illegal(0x1C4, format);
        assert_illegal(0x1F8, format);
        assert_illegal(0x7F4 | (u128::MAX << 12), format);
    }
}

#[test]
fn void_extent_reserved_bits() {
    let void_extent = 0x1FC | (u128::MAX << 12);
    for format in astc_formats() {
        for hdr in [0, 1 << 9] {
            legal(void_extent | hdr | (0b11 << 10), format);
            assert_illegal(void_extent | hdr | (0b01 << 10), format);
            assert_illegal(void_extent | hdr | (0b10 << 10), format);
            assert_illegal(void_extent | hdr, format);
        }
    }
}

/// "The size of the weight grid exceeds the size of the block footprint in any dimension."
#[test]
fn weight_grid_larger_than_the_footprint() {
    // 20 weights of two bits: 40 weight bits either way.
    let five_by_four = wide_4(2, 1) | WEIGHTS_0_TO_3;
    let four_by_five = wide_4(3, 0) | WEIGHTS_0_TO_3;
    legal(GRID_4X4 | WEIGHTS_0_TO_3, F::Astc2d4x4Unorm);
    assert_illegal(five_by_four, F::Astc2d4x4Unorm);
    assert_illegal(four_by_five, F::Astc2d4x4Unorm);
    legal(five_by_four, F::Astc2d5x4Unorm);
    assert_illegal(four_by_five, F::Astc2d5x4Unorm);
    legal(five_by_four, F::Astc2d5x5Unorm);
    legal(four_by_five, F::Astc2d5x5Unorm);
    // A grid 12 wide and 2 high of one-bit weights (24 bits) fits only the 12-wide footprints.
    let twelve_by_two = wide_12(0) | HIGH_ROWS_0_TO_1;
    for format in astc_formats() {
        let fits = super::footprint(format).width() == 12;
        if fits {
            legal(twelve_by_two, format);
        } else {
            assert_illegal(twelve_by_two, format);
        }
    }
}

/// "A block mode has been specified that would require more than 64 weights total."
#[test]
fn more_than_64_weights() {
    // One plane: 8 by 8 is 64 one-bit weights; 9 by 9 is 81 (81 bits, within 24..96).
    legal(from_6(2, 2) | HIGH_ROWS_0_TO_1, F::Astc2d8x8Unorm);
    legal(from_6(2, 2) | HIGH_ROWS_0_TO_1, F::Astc2d12x12Unorm);
    assert_illegal(from_6(3, 3) | HIGH_ROWS_0_TO_1, F::Astc2d12x12Unorm);
    assert_illegal(from_6(3, 3) | HIGH_ROWS_0_TO_1, F::Astc2d10x10Unorm);
    // Two planes: 8 by 4 is 2 * 32 = 64 weights; 11 by 3 is 2 * 33 = 66 (66 bits).
    legal(
        wide_8(2, 0) | WEIGHTS_0_TO_1 | DUAL_PLANE,
        F::Astc2d12x12Unorm,
    );
    assert_illegal(
        wide_8(1, 3) | WEIGHTS_0_TO_1 | DUAL_PLANE,
        F::Astc2d12x12Unorm,
    );
}

/// "... more than 96 bits for integer sequence encoding of the weight grid", "... fewer than
/// 24 bits ...". C.2.22: a weight of `t` trits, `q` quints and `b` bits costs
/// `ceil(8 n t / 5) + ceil(7 n q / 3) + n b` bits for `n` weights.
#[test]
fn weight_bits_out_of_range() {
    // 96: 8 by 6 two-bit weights. 97: 7 by 3 weights of a trit and three bits,
    // ceil(8 * 21 / 5) + 21 * 3 = 34 + 63.
    legal(from_6(2, 0) | HIGH_ROWS_0_TO_3, F::Astc2d8x6Unorm);
    assert_illegal(wide_4(1, 3) | WEIGHTS_0_TO_23, F::Astc2d8x8Unorm);
    // 24: 12 by 2 one-bit weights. 23: 7 by 2 trits, ceil(8 * 14 / 5).
    legal(wide_12(0) | HIGH_ROWS_0_TO_1, F::Astc2d12x10Unorm);
    assert_illegal(wide_4(0, 3) | WEIGHTS_0_TO_2, F::Astc2d8x8Unorm);
    // Well under: 4 by 2 one-bit weights are 8 bits.
    assert_illegal(wide_4(0, 0) | WEIGHTS_0_TO_1, F::Astc2d4x4Unorm);
    // Well over: 12 by 5 two-bit weights are 120 bits (and 60 weights, within 64).
    assert_illegal(wide_12(3) | HIGH_ROWS_0_TO_3, F::Astc2d12x12Unorm);
}

/// "Dual weight mode is enabled for a block with 4 partitions."
#[test]
fn dual_plane_with_four_partitions() {
    let dual = GRID_4X4 | WEIGHTS_0_TO_3 | DUAL_PLANE;
    for format in [F::Astc2d4x4Unorm, F::Astc2d8x8Srgb] {
        // Three partitions of mode 0: six values in 128 - 29 - 2 - 64 = 33 bits.
        legal(dual | (2 << PARTITIONS_AT), format);
        assert_illegal(dual | (3 << PARTITIONS_AT), format);
    }
    // Four partitions with one plane are fine.
    legal(
        GRID_4X4 | WEIGHTS_0_TO_3 | (3 << PARTITIONS_AT),
        F::Astc2d4x4Unorm,
    );
}

/// "Color endpoint modes have been specified such that the color integer sequence encoding
/// would require more than 18 integers."
#[test]
fn too_many_colour_values_is_illegal() {
    // A 4 by 4 grid of two-bit weights leaves 128 - 29 - 32 = 67 colour bits, more than
    // ceil(13 * 24 / 5) = 63, so only the count decides here.
    let base = GRID_4X4 | WEIGHTS_0_TO_3;
    let same_mode = |partitions: u128, mode: u128| {
        base | ((partitions - 1) << PARTITIONS_AT) | ((mode << 2) << MULTI_CEM_AT)
    };
    // Mode 8 has six values and mode 12 eight (table C.2.17).
    legal(same_mode(3, 8), F::Astc2d4x4Unorm); // 18
    assert_illegal(same_mode(3, 12), F::Astc2d4x4Unorm); // 24
    legal(same_mode(4, 4), F::Astc2d4x4Unorm); // 16
    assert_illegal(same_mode(4, 8), F::Astc2d4x4Unorm); // 24
    // Four partitions of classes 1 or 2 (selector 10): two of class 2 and two of class 1 are
    // 6 + 6 + 4 + 4 = 20 values in 128 - 37 - 32 = 59 bits (ceil(13 * 20 / 5) = 52). One of
    // class 2 is 6 + 4 + 4 + 4 = 18.
    let mixed = |class_bits: u128| {
        base | (3 << PARTITIONS_AT) | ((0b10 | (class_bits << 2)) << MULTI_CEM_AT)
    };
    assert_illegal(mixed(0b0011), F::Astc2d4x4Unorm);
    legal(mixed(0b0001), F::Astc2d4x4Unorm);
}

/// "The number of bits available for color endpoint encoding after all the other fields have
/// been counted is less than ceil(13C/5) where C is the number of color endpoint integers."
#[test]
fn too_few_colour_bits_is_illegal() {
    // One partition of mode 12: eight values need ceil(13 * 8 / 5) = 21 bits.
    // 9 by 5 two-bit weights: 128 - 17 - 90 = 21 bits. Legal, exactly.
    legal(
        wide_8(3, 1) | WEIGHTS_0_TO_3 | (12 << CEM_AT),
        F::Astc2d10x5Unorm,
    );
    // 7 by 5 weights of a trit and a bit: ceil(8 * 35 / 5) + 35 = 91 weight bits, which
    // leave 20.
    assert_illegal(
        wide_4(3, 3) | WEIGHTS_0_TO_5 | (12 << CEM_AT),
        F::Astc2d10x5Unorm,
    );
    // Three partitions of mode 0: six values need 16 bits. 6 by 7 two-bit weights leave
    // 128 - 29 - 84 = 15; 8 by 5 leave 19.
    let three = 2 << PARTITIONS_AT;
    assert_illegal(from_6(0, 1) | HIGH_ROWS_0_TO_3 | three, F::Astc2d8x8Unorm);
    legal(wide_8(3, 0) | WEIGHTS_0_TO_3 | three, F::Astc2d8x8Unorm);
    // Two partitions of mode 12 with 96 weight bits: 16 values in 3 bits.
    assert_illegal(
        from_6(2, 0) | HIGH_ROWS_0_TO_3 | (1 << PARTITIONS_AT) | ((12 << 2) << MULTI_CEM_AT),
        F::Astc2d8x8Unorm,
    );
}

/// Every one of these is the error colour in every encoding, and stays so with any other
/// bits set: the checks do not depend on the colour or weight data.
#[test]
fn illegal_encodings_decode_to_the_error_colour() {
    let illegal_on_8x8 = [
        0,
        0x1C4,
        wide_12(0) | HIGH_ROWS_0_TO_1,
        wide_4(1, 3) | WEIGHTS_0_TO_23,
        wide_4(0, 3) | WEIGHTS_0_TO_2,
        GRID_4X4 | WEIGHTS_0_TO_3 | DUAL_PLANE | (3 << PARTITIONS_AT),
    ];
    for block in illegal_on_8x8 {
        for fill in [
            0,
            u128::MAX << 13,
            0x5A5A_5A5A_5A5A_5A5A_5A5A_5A5A_5A5A_4000,
        ] {
            let block = block | (fill & !0x1FFF) | (block & 0x1FFF);
            assert_illegal(block, F::Astc2d8x8Unorm);
            assert_illegal(block, F::Astc2d8x8Srgb);
        }
    }
}
