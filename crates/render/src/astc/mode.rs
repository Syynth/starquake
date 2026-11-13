//! The block mode: the first eleven bits of a block, which give the size of its weight grid,
//! the range of its weights and whether it has two weight planes (C.2.10).
//!
//! Ported from Eden's `astc.cpp` (`DecodeBlockInfo`, `TexelWeightParams`); see the parent
//! module for the attribution.

use super::integer::Encoding;

/// Bits in the block mode field.
pub(super) const MODE_BITS: u32 = 11;

/// Bits 8..0 of a void-extent block: 111111100 (table C.2.8).
const VOID_EXTENT_MASK: u32 = 0x1FF;
const VOID_EXTENT: u32 = 0x1FC;
/// Bit 9 of a void-extent block: its colour is FP16 rather than UNORM16 (C.2.23).
const VOID_EXTENT_HDR: u32 = 1 << 9;

/// The H bit: the high-precision weight ranges (table C.2.7).
const HIGH_PRECISION: u32 = 1 << 9;
/// The D bit: two weight planes.
const DUAL_PLANE: u32 = 1 << 10;

/// Table C.2.7's weight ranges, by R - 2: the largest weight of each, without and with the H
/// bit.
const LOW_PRECISION_MAX: [u32; 6] = [1, 2, 3, 4, 5, 7];
const HIGH_PRECISION_MAX: [u32; 6] = [9, 11, 15, 19, 23, 31];

/// The weight grid a block mode describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WeightGrid {
    /// Weights across.
    pub(super) width: u32,
    /// Weights down.
    pub(super) height: u32,
    /// Whether each grid point has two weights.
    pub(super) dual_plane: bool,
    /// The largest stored weight value.
    pub(super) max_weight: u32,
}

impl WeightGrid {
    /// Weights stored: one or two for each grid point. At most 12 * 12 * 2.
    pub(super) const fn weight_count(&self) -> u32 {
        self.width * self.height * if self.dual_plane { 2 } else { 1 }
    }

    /// How the weights are stored.
    pub(super) const fn encoding(&self) -> Encoding {
        Encoding::for_max(self.max_weight)
    }
}

/// What a block mode says the block is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BlockMode {
    /// One colour for the whole block (C.2.23).
    VoidExtent {
        /// Whether the colour is FP16.
        hdr: bool,
    },
    /// Endpoints and a grid of weights.
    Weights(WeightGrid),
}

/// Decodes the block mode in the low eleven bits of `mode` (table C.2.8), or `None` for a
/// reserved one.
pub(super) fn block_mode(mode: u32) -> Option<BlockMode> {
    if mode & VOID_EXTENT_MASK == VOID_EXTENT {
        return Some(BlockMode::VoidExtent {
            hdr: mode & VOID_EXTENT_HDR != 0,
        });
    }
    // Reserved: bits 8..6 all one with bits 1..0 zero. (The table's other reserved row, bits
    // 3..0 all zero, is an R of 0 or 1, which the range lookup below refuses.)
    if mode & 0x1C3 == 0x1C0 {
        return None;
    }
    let a = (mode >> 5) & 0x3;
    let b = (mode >> 7) & 0x3;
    let r0 = (mode >> 4) & 0x1;
    // R2 and R1 are bits 1..0 in the table's first five rows, where they cannot both be zero,
    // and bits 3..2 in the rest.
    let (width, height, r, planes_and_precision) = if mode & 0x3 != 0 {
        let r = ((mode & 0x3) << 1) | r0;
        let (width, height) = match (mode >> 2) & 0x3 {
            0 => (b + 4, a + 2),
            1 => (b + 8, a + 2),
            2 => (a + 2, b + 8),
            _ if mode & 0x100 == 0 => (a + 2, (b & 0x1) + 6),
            _ => ((b & 0x1) + 2, a + 2),
        };
        (width, height, r, true)
    } else {
        let r = (((mode >> 2) & 0x3) << 1) | r0;
        match b {
            0 => (12, a + 2, r, true),
            1 => (a + 2, 12, r, true),
            // The one row with neither a D nor an H bit: bits 10..9 are B.
            2 => (a + 6, ((mode >> 9) & 0x3) + 6, r, false),
            _ if mode & 0x20 == 0 => (6, 10, r, true),
            _ => (10, 6, r, true),
        }
    };
    let dual_plane = planes_and_precision && mode & DUAL_PLANE != 0;
    let ranges = if planes_and_precision && mode & HIGH_PRECISION != 0 {
        HIGH_PRECISION_MAX
    } else {
        LOW_PRECISION_MAX
    };
    // Table C.2.7: an R of 0 or 1 is invalid. That happens exactly when bits 3..0 are all
    // zero: R's top two bits are then bits 3..2.
    let max_weight = *ranges.get(usize::try_from(r.checked_sub(2)?).ok()?)?;
    Some(BlockMode::Weights(WeightGrid {
        width,
        height,
        dual_plane,
        max_weight,
    }))
}

#[cfg(test)]
mod tests {
    use super::{BlockMode, WeightGrid, block_mode};
    use crate::astc::integer::{Digit, Encoding};

    fn grid(mode: u32) -> WeightGrid {
        match block_mode(mode) {
            Some(BlockMode::Weights(grid)) => grid,
            other => unreachable!("{mode:#x} is {other:?}"),
        }
    }

    const fn weights(width: u32, height: u32, dual_plane: bool, max_weight: u32) -> WeightGrid {
        WeightGrid {
            width,
            height,
            dual_plane,
            max_weight,
        }
    }

    /// A block mode from its fields as table C.2.8 lays them out: bits 10..9, bits 8..7, bits
    /// 6..5, bit 4, bits 3..2 and bits 1..0.
    const fn row(top: u32, b: u32, a: u32, r0: u32, mid: u32, low: u32) -> u32 {
        (top << 9) | (b << 7) | (a << 5) | (r0 << 4) | (mid << 2) | low
    }

    /// One mode from each row of table C.2.8, with A, B and R chosen so that no two fields
    /// hold the same value, written out bit by bit.
    #[test]
    fn each_row_of_the_table() {
        // Row 1: D H B B A A R0 0 0 R2 R1, width B+4, height A+2.
        // B = 3, A = 1, R = 101 (R2 R1 = 10, R0 = 1): range 0..4.
        assert_eq!(
            grid(row(0b00, 0b11, 0b01, 1, 0b00, 0b10)),
            weights(7, 3, false, 4)
        );
        // Row 2: .. R0 0 1 R2 R1, width B+8, height A+2. B = 1, A = 2, R = 011: 0..2.
        assert_eq!(
            grid(row(0b00, 0b01, 0b10, 1, 0b01, 0b01)),
            weights(9, 4, false, 2)
        );
        // Row 3: .. R0 1 0 R2 R1, width A+2, height B+8. B = 2, A = 3, R = 110: 0..5.
        assert_eq!(
            grid(row(0b00, 0b10, 0b11, 0, 0b10, 0b11)),
            weights(5, 10, false, 5)
        );
        // Row 4: D H 0 B A A R0 1 1 R2 R1, width A+2, height B+6. B = 1, A = 2, R = 010: 0..1.
        assert_eq!(
            grid(row(0b00, 0b01, 0b10, 0, 0b11, 0b01)),
            weights(4, 7, false, 1)
        );
        // Row 5: D H 1 B A A R0 1 1 R2 R1, width B+2, height A+2. B = 1, A = 0, R = 111: 0..7.
        assert_eq!(
            grid(row(0b00, 0b11, 0b00, 1, 0b11, 0b11)),
            weights(3, 2, false, 7)
        );
        // Row 6: D H 0 0 A A R0 R2 R1 0 0, width 12, height A+2. A = 1, R = 100: 0..3.
        assert_eq!(
            grid(row(0b00, 0b00, 0b01, 0, 0b10, 0b00)),
            weights(12, 3, false, 3)
        );
        // Row 7: D H 0 1 A A R0 R2 R1 0 0, width A+2, height 12. A = 3, R = 011: 0..2.
        assert_eq!(
            grid(row(0b00, 0b01, 0b11, 1, 0b01, 0b00)),
            weights(5, 12, false, 2)
        );
        // Row 8: D H 1 1 0 0 R0 R2 R1 0 0, width 6, height 10. R = 101: 0..4.
        assert_eq!(
            grid(row(0b00, 0b11, 0b00, 1, 0b10, 0b00)),
            weights(6, 10, false, 4)
        );
        // Row 9: D H 1 1 0 1 R0 R2 R1 0 0, width 10, height 6. R = 010: 0..1.
        assert_eq!(
            grid(row(0b00, 0b11, 0b01, 0, 0b01, 0b00)),
            weights(10, 6, false, 1)
        );
        // Row 10: B B 1 0 A A R0 R2 R1 0 0, width A+6, height B+6. B = 3, A = 1, R = 110.
        assert_eq!(
            grid(row(0b11, 0b10, 0b01, 0, 0b11, 0b00)),
            weights(7, 9, false, 5)
        );
    }

    /// The D and H bits: two planes, and the high-precision ranges of table C.2.7.
    #[test]
    fn dual_plane_and_precision_bits() {
        let base = row(0b00, 0b00, 0b10, 0, 0b00, 0b10); // row 1: 4 by 4, R = 100
        assert_eq!(grid(base), weights(4, 4, false, 3));
        assert_eq!(grid(base | (1 << 10)), weights(4, 4, true, 3));
        assert_eq!(grid(base | (1 << 9)), weights(4, 4, false, 15));
        assert_eq!(grid(base | (0b11 << 9)), weights(4, 4, true, 15));
        // In row 10 those bits are B: neither a second plane nor the high-precision ranges.
        assert_eq!(
            grid(row(0b11, 0b10, 0b00, 0, 0b10, 0b00)),
            weights(6, 9, false, 3)
        );
        // Rows 6 to 9 have both.
        assert_eq!(
            grid(row(0b11, 0b00, 0b00, 0, 0b10, 0b00)),
            weights(12, 2, true, 15)
        );
        assert_eq!(
            grid(row(0b10, 0b11, 0b01, 0, 0b10, 0b00)),
            weights(10, 6, true, 3)
        );
    }

    /// Table C.2.7: every R with and without H, in a row of each kind (R2 R1 in bits 1..0,
    /// and in bits 3..2).
    #[test]
    fn every_weight_range() {
        let ranges = [
            (0b010, 1, 9),
            (0b011, 2, 11),
            (0b100, 3, 15),
            (0b101, 4, 19),
            (0b110, 5, 23),
            (0b111, 7, 31),
        ];
        for (r, low, high) in ranges {
            let (r21, r0) = (r >> 1, r & 1);
            let first_rows = r21 | (r0 << 4) | (2 << 5); // row 1, 4 by 4
            let later_rows = (r21 << 2) | (r0 << 4); // row 6, 12 by 2
            assert_eq!(grid(first_rows).max_weight, low, "{r:#b}");
            assert_eq!(grid(first_rows | (1 << 9)).max_weight, high, "{r:#b}");
            assert_eq!(grid(later_rows).max_weight, low, "{r:#b}");
            assert_eq!(grid(later_rows | (1 << 9)).max_weight, high, "{r:#b}");
        }
    }

    #[test]
    fn void_extent_and_reserved_modes() {
        assert_eq!(
            block_mode(0x1FC),
            Some(BlockMode::VoidExtent { hdr: false })
        );
        assert_eq!(
            block_mode(0x1FC | (1 << 9)),
            Some(BlockMode::VoidExtent { hdr: true })
        );
        // Bit 10 is not part of the pattern.
        assert_eq!(
            block_mode(0x1FC | (1 << 10)),
            Some(BlockMode::VoidExtent { hdr: false })
        );
        // Every mode with bits 3..0 zero is reserved; so is every mode with bits 8..6 one
        // and bits 1..0 zero, except the void-extent pattern.
        for mode in 0..0x800 {
            let low_four_clear =
                [0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70].contains(&(mode & 0x7F));
            let reserved = low_four_clear || (mode & 0x1C3 == 0x1C0 && mode & 0x1FF != 0x1FC);
            assert_eq!(block_mode(mode).is_none(), reserved, "{mode:#x}");
        }
        // One bit away from each reserved pattern is a mode.
        assert!(block_mode(0b0001).is_some());
        assert!(block_mode(0b0100).is_some());
        assert!(block_mode(0x1C0 | 0x1).is_some());
        assert!(block_mode(0x0C0 | 0x4).is_some());
    }

    /// Every mode that is not reserved gives a grid of 2 to 12 each way, with a weight range
    /// of table C.2.7.
    #[test]
    fn every_mode_is_in_range() {
        for mode in 0..0x800 {
            if let Some(BlockMode::Weights(grid)) = block_mode(mode) {
                assert!((2..=12).contains(&grid.width), "{mode:#x}");
                assert!((2..=12).contains(&grid.height), "{mode:#x}");
                assert!(
                    [1, 2, 3, 4, 5, 7, 9, 11, 15, 19, 23, 31].contains(&grid.max_weight),
                    "{mode:#x}"
                );
            }
        }
    }

    #[test]
    fn weight_counts_and_encodings() {
        assert_eq!(weights(4, 4, false, 3).weight_count(), 16);
        assert_eq!(weights(12, 12, true, 1).weight_count(), 288);
        assert_eq!(weights(3, 5, true, 1).weight_count(), 30);
        assert_eq!(
            weights(4, 4, false, 11).encoding(),
            Encoding {
                digit: Digit::Trit,
                bits: 2
            }
        );
    }
}
