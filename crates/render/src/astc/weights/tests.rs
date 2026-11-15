//! Weight unquantization, infill and partition selection, against values worked by hand from
//! the specification (C.2.17, C.2.18, C.2.21).

use super::{MAX_WEIGHTS, infill, select_partition, unquantize_weight};
use crate::astc::Footprint;
use crate::astc::integer::{Digit, Encoding, Value};
use crate::astc::mode::WeightGrid;
use crate::format::{Compression, PixelFormat};

const fn encoding(digit: Digit, bits: u32) -> Encoding {
    Encoding { digit, bits }
}

fn all_values(encoding: Encoding, digits: u32) -> Vec<u32> {
    let mut values = Vec::new();
    for digit in 0..digits {
        for bits in 0..(1 << encoding.bits) {
            values.push(unquantize_weight(encoding, Value { bits, digit }).expect("a range"));
        }
    }
    values
}

/// Tables C.2.25 and C.2.26 by hand, each followed by `if (T > 32) T += 1`.
#[test]
fn weight_unquantization_by_hand() {
    // Digits alone: 0, 32, 63 and 0, 16, 32, 47, 63.
    assert_eq!(all_values(encoding(Digit::Trit, 0), 3), [0, 32, 64]);
    assert_eq!(
        all_values(encoding(Digit::Quint, 0), 5),
        [0, 16, 32, 48, 64]
    );
    // Plain bits replicate to six: 1 bit, 2 bits, 3 bits ("abc" becomes "abcabc").
    assert_eq!(all_values(encoding(Digit::None, 1), 1), [0, 64]);
    assert_eq!(all_values(encoding(Digit::None, 2), 1), [0, 21, 43, 64]);
    assert_eq!(
        all_values(encoding(Digit::None, 3), 1),
        [0, 9, 18, 27, 37, 46, 55, 64]
    );
    // A trit and one bit (0..5): B = 0, C = 50, A the bit seven times over.
    // Bit 0: 0, 50 >> 2, 100 >> 2. Bit 1: 0x20 | (127 >> 2) = 63, 0x20 | ((50 ^ 127) >> 2) = 51,
    // 0x20 | ((100 ^ 127) >> 2) = 38; each above 32 gains 1.
    assert_eq!(
        all_values(encoding(Digit::Trit, 1), 3),
        [0, 64, 12, 52, 25, 39]
    );
    // A quint and one bit (0..9): C = 28.
    assert_eq!(
        all_values(encoding(Digit::Quint, 1), 5),
        [0, 64, 7, 57, 14, 50, 21, 43, 28, 36]
    );
    // A trit and two bits "ba" (0..11): B = b000b0b, C = 23. Value 0b10 with trit 2:
    // A = 0, B = 0b1000101 = 69, T = 46 + 69 = 115, 115 >> 2 = 28.
    assert_eq!(
        unquantize_weight(encoding(Digit::Trit, 2), Value { bits: 2, digit: 2 }),
        Some(28)
    );
    // A quint and two bits (0..19): B = b0000b0, C = 13. Value 0b11 with quint 1:
    // A = 127, B = 0b1000010 = 66, T = (13 + 66) ^ 127 = 48, 0x20 | 12 = 44, and 45.
    assert_eq!(
        unquantize_weight(encoding(Digit::Quint, 2), Value { bits: 3, digit: 1 }),
        Some(45)
    );
    // A trit and three bits "cba" (0..23): B = cb000cb, C = 11. Value 0b101 with trit 1:
    // A = 127, cb = 0b10, B = 0b1000010 = 66, T = (11 + 66) ^ 127 = 50, 0x20 | 12 = 44, and 45.
    assert_eq!(
        unquantize_weight(encoding(Digit::Trit, 3), Value { bits: 5, digit: 1 }),
        Some(45)
    );
}

/// Every weight range of table C.2.7 unquantizes to that many distinct values, from 0 to 64,
/// mirrored about the middle (w and 64 - w), and spread evenly: the i-th smallest is within
/// 1.5 of `64 * i / (count - 1)` (the furthest it strays is 1.09, in the range 0..11).
#[test]
fn every_weight_range_is_an_even_spread() {
    let ranges = [
        (encoding(Digit::None, 1), 1, 2),
        (encoding(Digit::Trit, 0), 3, 3),
        (encoding(Digit::None, 2), 1, 4),
        (encoding(Digit::Quint, 0), 5, 5),
        (encoding(Digit::Trit, 1), 3, 6),
        (encoding(Digit::None, 3), 1, 8),
        (encoding(Digit::Quint, 1), 5, 10),
        (encoding(Digit::Trit, 2), 3, 12),
        (encoding(Digit::None, 4), 1, 16),
        (encoding(Digit::Quint, 2), 5, 20),
        (encoding(Digit::Trit, 3), 3, 24),
        (encoding(Digit::None, 5), 1, 32),
    ];
    for (encoding, digits, count) in ranges {
        let mut values = all_values(encoding, digits);
        values.sort_unstable();
        assert_eq!(values.len(), count, "{encoding:?}");
        assert_eq!((values[0], values[count - 1]), (0, 64), "{encoding:?}");
        for (index, value) in values.iter().enumerate() {
            assert_eq!(values[count - 1 - index], 64 - value, "{encoding:?}");
            if let Some(next) = values.get(index + 1) {
                assert!(value < next, "{encoding:?}: {value} twice");
            }
            let value = i32::try_from(*value).expect("small");
            let index = i32::try_from(index).expect("small");
            let steps = i32::try_from(count - 1).expect("small");
            assert!(
                2 * (value * steps - 64 * index).abs() <= 3 * steps,
                "{encoding:?}: value {index} is {value}"
            );
        }
    }
}

/// Encodings no weight has, and digits past a digit-only table, give `None`.
#[test]
fn encodings_weights_never_have() {
    let value = Value { bits: 0, digit: 0 };
    for encoding in [
        encoding(Digit::None, 0),
        encoding(Digit::None, 6),
        encoding(Digit::Trit, 4),
        encoding(Digit::Quint, 3),
    ] {
        assert_eq!(unquantize_weight(encoding, value), None, "{encoding:?}");
    }
    let digit = |digit| Value { bits: 0, digit };
    assert_eq!(
        unquantize_weight(encoding(Digit::Trit, 0), digit(2)),
        Some(64)
    );
    assert_eq!(unquantize_weight(encoding(Digit::Trit, 0), digit(3)), None);
    assert_eq!(
        unquantize_weight(encoding(Digit::Quint, 0), digit(4)),
        Some(64)
    );
    assert_eq!(unquantize_weight(encoding(Digit::Quint, 0), digit(5)), None);
}

fn footprints() -> Vec<Footprint> {
    PixelFormat::ALL
        .into_iter()
        .filter(|format| format.compression() == Compression::Astc && !format.is_srgb())
        .map(|format| Footprint::of(format).expect("ASTC"))
        .collect()
}

const fn grid(width: u32, height: u32, dual_plane: bool) -> WeightGrid {
    WeightGrid {
        width,
        height,
        dual_plane,
        max_weight: 31,
    }
}

/// Stored weights 0, 1, 2, .. for `count` grid points, one plane.
fn counting(count: u32) -> [u32; MAX_WEIGHTS] {
    let mut stored = [0; MAX_WEIGHTS];
    for (weight, value) in stored.iter_mut().zip(0..count) {
        *weight = value;
    }
    stored
}

/// The fourteen footprints of the console's formats.
#[test]
fn the_footprints() {
    let sizes: Vec<_> = footprints()
        .iter()
        .map(|f| (f.width(), f.height()))
        .collect();
    let mut sorted = sizes.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        [
            (4, 4),
            (5, 4),
            (5, 5),
            (6, 5),
            (6, 6),
            (8, 5),
            (8, 6),
            (8, 8),
            (10, 5),
            (10, 6),
            (10, 8),
            (10, 10),
            (12, 10),
            (12, 12)
        ]
    );
}

/// A grid the size of the footprint is used as it is: C.2.18's scale factors bring each
/// texel onto its own grid point with no fraction.
#[test]
fn a_full_size_grid_is_not_interpolated() {
    for footprint in footprints() {
        let (width, height) = (footprint.width(), footprint.height());
        // At most 64 weights are stored, so the largest footprints cannot have a full grid.
        if width * height > 64 {
            continue;
        }
        let weights = infill(
            &grid(width, height, false),
            &counting(width * height),
            footprint,
        );
        let expected: Vec<u32> = (0..width * height).collect();
        assert_eq!(
            weights[0][..expected.len()],
            expected,
            "{width} by {height}"
        );
    }
}

/// C.2.18 worked through for a 4 by 4 grid on an 8 by 8 footprint (Ds = Dt = 146), with the
/// grid's weights 0, 4, 8, .., 60.
#[test]
fn infill_by_hand() {
    let mut stored = [0; MAX_WEIGHTS];
    for (weight, index) in stored.iter_mut().zip(0..16) {
        *weight = 4 * index;
    }
    let footprint = Footprint::of(PixelFormat::Astc2d8x8Unorm).expect("ASTC");
    let weights = infill(&grid(4, 4, false), &stored, footprint);
    // Texel (1, 0): cs = 146, gs = (146 * 3 + 32) >> 6 = 7, so js = 0 and fs = 7; gt = 0.
    // w11 = (7 * 0 + 8) >> 4 = 0, w10 = 0, w01 = 7, w00 = 9:
    // (0 * 9 + 4 * 7 + 8) >> 4 = 2.
    #[rustfmt::skip]
    let expected = [
        0, 2, 4, 5, 7, 9, 10, 12,
        7, 9, 11, 12, 14, 16, 17, 19,
        14, 16, 18, 19, 21, 23, 24, 26,
        21, 23, 25, 26, 28, 30, 31, 33,
        27, 29, 31, 32, 34, 36, 37, 39,
        34, 36, 38, 39, 41, 43, 44, 46,
        41, 43, 45, 46, 48, 50, 51, 53,
        48, 50, 52, 53, 55, 57, 58, 60,
    ];
    assert_eq!(weights[0][..64], expected);
    // The second plane is untouched with one plane, and so is everything past the footprint.
    assert!(weights[1].iter().all(|w| *w == 0));
    assert!(weights[0][64..].iter().all(|w| *w == 0));
}

/// Whatever the grid, the footprint's four corner texels take the grid's four corner weights
/// exactly, and no texel's weight is outside the range of the stored ones.
#[test]
fn corners_are_exact_and_weights_stay_in_range() {
    for footprint in footprints() {
        for width in 2..=footprint.width() {
            for height in 2..=footprint.height() {
                let points = width * height;
                if points > 64 {
                    continue;
                }
                // Distinct weights from 1 to `points`, so a zero would show a point read
                // from outside the grid.
                let mut stored = [0; MAX_WEIGHTS];
                for (weight, value) in stored.iter_mut().zip(1..=points) {
                    *weight = value;
                }
                let weights = infill(&grid(width, height, false), &stored, footprint);
                let (bw, bh) = (footprint.width(), footprint.height());
                let at = |x: u32, y: u32| weights[0][(y * bw + x) as usize];
                let what = format!("{width} by {height} on {bw} by {bh}");
                assert_eq!(at(0, 0), 1, "{what}");
                assert_eq!(at(bw - 1, 0), width, "{what}");
                assert_eq!(at(0, bh - 1), points - width + 1, "{what}");
                assert_eq!(at(bw - 1, bh - 1), points, "{what}");
                for y in 0..bh {
                    for x in 0..bw {
                        assert!((1..=points).contains(&at(x, y)), "{what}: ({x}, {y})");
                    }
                }
                // With every stored weight at the top of the range, every texel's is too.
                let full = infill(&grid(width, height, false), &[64; MAX_WEIGHTS], footprint);
                assert!(
                    full[0][..(bw * bh) as usize].iter().all(|w| *w == 64),
                    "{what}"
                );
            }
        }
    }
}

/// With two planes, each grid point stores plane 0's weight and then plane 1's (C.2.16).
#[test]
fn two_planes_are_stored_together() {
    let footprint = Footprint::of(PixelFormat::Astc2d4x4Unorm).expect("ASTC");
    let mut stored = [0; MAX_WEIGHTS];
    for (index, weight) in stored.iter_mut().take(32).enumerate() {
        let point = u32::try_from(index / 2).expect("small");
        *weight = if index % 2 == 0 { point } else { 40 + point };
    }
    let weights = infill(&grid(4, 4, true), &stored, footprint);
    let first: Vec<u32> = (0..16).collect();
    let second: Vec<u32> = (40..56).collect();
    assert_eq!(weights[0][..16], first);
    assert_eq!(weights[1][..16], second);
}

fn pattern(seed: u32, partitions: u32, width: u32, height: u32, small: bool) -> Vec<String> {
    (0..height)
        .map(|y| {
            (0..width)
                .map(|x| {
                    let partition = select_partition(seed, x, y, partitions, small);
                    char::from_digit(u32::try_from(partition).expect("small"), 10).expect("a digit")
                })
                .collect()
        })
        .collect()
}

/// C.2.21's `select_partition` worked through for a few seeds, partition counts and block
/// sizes (the same patterns the whole-block vectors show through colours).
#[test]
fn partition_patterns_by_hand() {
    assert_eq!(
        pattern(613, 2, 4, 4, true),
        ["1111", "0000", "0000", "0000"]
    );
    assert_eq!(
        pattern(613, 3, 4, 4, true),
        ["2101", "2201", "2101", "1101"]
    );
    assert_eq!(
        pattern(613, 4, 4, 4, true),
        ["3223", "0000", "3233", "2111"]
    );
    assert_eq!(pattern(5, 2, 4, 4, true), ["1110", "1100", "1000", "1000"]);
    assert_eq!(pattern(5, 4, 4, 4, true), ["3330", "3302", "3022", "0022"]);
    // The same seed with and without the small-block doubling.
    assert_eq!(
        pattern(999, 2, 6, 5, true),
        ["000111", "100111", "100111", "100111", "110111"]
    );
    assert_eq!(
        pattern(999, 2, 6, 6, false),
        ["000000", "000000", "100000", "100000", "110000", "110000"]
    );
    assert_eq!(
        pattern(857, 4, 8, 6, false),
        [
            "22222333", "22222333", "22222333", "22222333", "22222330", "22222330"
        ]
    );
}

/// One partition is partition 0 whatever the seed, and a texel's partition is always below
/// the partition count.
#[test]
fn partitions_stay_below_the_count() {
    for seed in 0..1024 {
        assert_eq!(select_partition(seed, 7, 3, 1, false), 0);
        assert_eq!(select_partition(seed, 7, 3, 0, false), 0);
        for partitions in 2..=4_u32 {
            for small in [false, true] {
                for y in [0, 5, 11] {
                    for x in [0, 6, 11] {
                        let partition = select_partition(seed, x, y, partitions, small);
                        assert!(partition < partitions as usize, "seed {seed}");
                    }
                }
            }
        }
    }
}
