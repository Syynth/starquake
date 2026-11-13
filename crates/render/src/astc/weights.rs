//! Texel weights: unquantizing a block's stored weights (C.2.17), spreading the weight grid
//! over the footprint (C.2.18), and the partition a texel belongs to (C.2.21).
//!
//! Ported from Eden's `astc.cpp` (`UnquantizeTexelWeight`, `UnquantizeTexelWeights`, `hash52`,
//! `SelectPartition`); see the parent module for the attribution.

use super::integer::{Digit, Encoding, Value, replicate};
use super::mode::WeightGrid;
use super::{Footprint, MAX_TEXELS};

/// The most weights a block may store (C.2.24).
pub(super) const MAX_WEIGHTS: usize = 64;

/// Unquantizes one stored weight to 0..64 (C.2.17, tables C.2.25 and C.2.26), or `None` for an
/// encoding weights never have.
pub(super) fn unquantize_weight(encoding: Encoding, value: Value) -> Option<u32> {
    let bits = value.bits;
    // The table's A: the lowest bit, seven times over.
    let fill = if bits & 1 == 0 { 0 } else { 0x7F };
    // The bits above the lowest: "b" or "cb" in table C.2.26's B column.
    let x = bits >> 1;
    let (b, c) = match (encoding.digit, encoding.bits) {
        (Digit::None, 1..=5) => return Some(widen(replicate(bits, encoding.bits, 6))),
        (Digit::Trit, 0) => return [0, 32, 63].get(value.digit as usize).copied().map(widen),
        (Digit::Quint, 0) => {
            return [0, 16, 32, 47, 63]
                .get(value.digit as usize)
                .copied()
                .map(widen);
        }
        (Digit::Trit, 1) => (0, 50),
        (Digit::Trit, 2) => ((x << 6) | (x << 2) | x, 23),
        (Digit::Trit, 3) => ((x << 5) | x, 11),
        (Digit::Quint, 1) => (0, 28),
        (Digit::Quint, 2) => ((x << 6) | (x << 1), 13),
        _ => return None,
    };
    let mixed = (value.digit * c + b) ^ fill;
    Some(widen((fill & 0x20) | (mixed >> 2)))
}

/// The last step of C.2.17: from 0..63 to 0..64, `if (T > 32) { T += 1; }`.
const fn widen(weight: u32) -> u32 {
    if weight > 32 { weight + 1 } else { weight }
}

/// The weight of every texel of the footprint, for each plane, from the unquantized weights
/// of the grid (C.2.18). `stored` holds the grid in raster order, with both planes' weights
/// together at each point when there are two (C.2.16).
pub(super) fn infill(
    grid: &WeightGrid,
    stored: &[u32; MAX_WEIGHTS],
    footprint: Footprint,
) -> [[u32; MAX_TEXELS]; 2] {
    let planes = if grid.dual_plane { 2 } else { 1 };
    let ds = (1024 + footprint.width / 2) / (footprint.width - 1);
    let dt = (1024 + footprint.height / 2) / (footprint.height - 1);
    let points = grid.width * grid.height;
    let mut weights = [[0; MAX_TEXELS]; 2];
    for (plane, plane_weights) in (0..planes).zip(&mut weights) {
        // A grid point's weight on this plane; a point outside the grid counts as 0 (it is
        // only ever asked for with a factor of 0).
        let at = |point: u32| {
            if point < points {
                stored
                    .get((point * planes + plane) as usize)
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            }
        };
        let texels = (0..footprint.height).flat_map(|t| (0..footprint.width).map(move |s| (s, t)));
        for ((s, t), weight) in texels.zip(plane_weights) {
            let gs = (ds * s * (grid.width - 1) + 32) >> 6;
            let gt = (dt * t * (grid.height - 1) + 32) >> 6;
            let (js, fs) = (gs >> 4, gs & 0xF);
            let (jt, ft) = (gt >> 4, gt & 0xF);
            let v0 = js + jt * grid.width;
            let w11 = (fs * ft + 8) >> 4;
            let w10 = ft - w11;
            let w01 = fs - w11;
            let w00 = 16 + w11 - fs - ft;
            *weight = (at(v0) * w00
                + at(v0 + 1) * w01
                + at(v0 + grid.width) * w10
                + at(v0 + grid.width + 1) * w11
                + 8)
                >> 4;
        }
    }
    weights
}

/// C.2.21's `hash52`.
const fn hash52(seed: u32) -> u32 {
    let mut p = seed;
    p ^= p >> 15;
    p = p.wrapping_sub(p << 17);
    p = p.wrapping_add(p << 7);
    p = p.wrapping_add(p << 4);
    p ^= p >> 5;
    p = p.wrapping_add(p << 16);
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    p
}

/// C.2.21's `select_partition` for a 2D block (z is 0): the partition, below `partitions`, of
/// the texel at (`x`, `y`). `seed` is the block's 10-bit partition index, and `small_block`
/// whether the footprint has fewer than 31 texels.
pub(super) fn select_partition(
    seed: u32,
    x: u32,
    y: u32,
    partitions: u32,
    small_block: bool,
) -> usize {
    if partitions <= 1 {
        return 0;
    }
    let (x, y) = if small_block {
        (x << 1, y << 1)
    } else {
        (x, y)
    };
    let seed = seed + (partitions - 1) * 1024;
    let rnum = hash52(seed);
    // The specification's seed1 to seed8 (the terms in z, seed9 to seed12, vanish in 2D).
    // Each is four bits of the hash, squared as a byte (at most 225) and shifted down.
    let (sh1, sh2) = if seed & 1 == 0 {
        (
            if partitions == 3 { 6 } else { 5 },
            if seed & 2 == 0 { 5 } else { 4 },
        )
    } else {
        (
            if seed & 2 == 0 { 5 } else { 4 },
            if partitions == 3 { 6 } else { 5 },
        )
    };
    let term = |index: u32, shift: u32| {
        let nibble = (rnum >> (4 * index)) & 0xF;
        (nibble * nibble) >> shift
    };
    // The specification's a, b, c and d: one score for each partition, the last two zero
    // when there are fewer partitions.
    let score = [
        (term(0, sh1) * x + term(1, sh2) * y + (rnum >> 14)) & 0x3F,
        (term(2, sh1) * x + term(3, sh2) * y + (rnum >> 10)) & 0x3F,
        if partitions < 3 {
            0
        } else {
            (term(4, sh1) * x + term(5, sh2) * y + (rnum >> 6)) & 0x3F
        },
        if partitions < 4 {
            0
        } else {
            (term(6, sh1) * x + term(7, sh2) * y + (rnum >> 2)) & 0x3F
        },
    ];
    if score[0] >= score[1] && score[0] >= score[2] && score[0] >= score[3] {
        0
    } else if score[1] >= score[2] && score[1] >= score[3] {
        1
    } else if score[2] >= score[3] {
        2
    } else {
        3
    }
}

#[cfg(test)]
mod tests;
