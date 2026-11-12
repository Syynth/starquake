//! The Switch 2 GOB derivation (`docs/specs/gpu-protocol.md`, "Block-linear layout"): Eden's GOB
//! swizzle is Mesa's Tegra colour GOB, and Eden-NX2's Switch 2 tweak (swap byte-offset bits 5
//! and 6) turns it into Mesa's Turing colour GOB, bit for bit.
//!
//! The oracles below are Eden's code, ported for this test (GPL-3.0-or-later, like this
//! project):
//!
//! - `swizzle_offset`: `SwizzleOffset` in
//!   `eden-nx2/src/video_core/host_shaders/astc_decoder.comp` @ 50fa86bb (line 1397).
//! - `native_gob_offset`: `NativeGobOffset` in `eden-nx2/src/video_core/textures/decoders.cpp`
//!   @ 50fa86bb (blocker-log row 69).
//! - `eden_level_offset`: the offset arithmetic of `SwizzleImpl` in the same `decoders.cpp`,
//!   with `pdep` and the masks `SWIZZLE_X_BITS` and `SWIZZLE_Y_BITS` from `decoders.h`. Eden
//!   only has tiles one GOB wide, so it checks tilings with `x_log2` 0.

/// In a `cfg(test)` module so the helpers may unwrap like the tests do.
#[cfg(test)]
mod derivation {
    use starquake_layout::{
        Extent4D, GOB_HEIGHT, GOB_LINE_WIDTH_B, GOB_SECTOR_B, GOB_SIZE_B, GOB_WIDTH_B, GobType,
        Offset4D, Tiling, block_linear_offset,
    };

    /// Eden's `SwizzleOffset(pos)`: byte offset bits 0–3 = x bits 0–3, bit 4 = y bit 0, bit 5 = x
    /// bit 4, bits 6–7 = y bits 1–2, bit 8 = x bit 5.
    fn swizzle_offset(x: u32, y: u32) -> u32 {
        ((x & 32) << 3) | ((y & 6) << 5) | ((x & 16) << 1) | ((y & 1) << 4) | (x & 15)
    }

    /// Eden-NX2's `NativeGobOffset(offset, true)`: swaps offset bits 5 and 6.
    fn native_gob_offset(offset: u32) -> u32 {
        let difference = ((offset >> 5) ^ (offset >> 6)) & 1;
        offset ^ (difference << 5) ^ (difference << 6)
    }

    /// Eden's `pdep<mask>(value)`: deposits the low bits of `value` into the set bits of `mask`.
    fn pdep(mask: u32, value: u32) -> u32 {
        let mut result = 0;
        let mut m = mask;
        let mut bit = 1u32;
        while m != 0 {
            if value & bit != 0 {
                result |= m.isolate_lowest_one();
            }
            m &= m - 1;
            bit = bit.wrapping_add(bit);
        }
        result
    }

    /// `SWIZZLE_X_BITS` and `SWIZZLE_Y_BITS` (`eden-nx2/src/video_core/textures/decoders.h`).
    const SWIZZLE_X_BITS: u32 = 0b1_0010_1111;
    const SWIZZLE_Y_BITS: u32 = 0b0_1101_0000;

    /// The tiled offset `SwizzleImpl` computes for byte (`x`, `y`, `z`) of a level `stride` bytes
    /// wide and `height` rows tall, with `block_height` and `block_depth` as log2 GOB counts.
    fn eden_level_offset(
        stride: u32,
        height: u32,
        block_height: u32,
        block_depth: u32,
        (x, y, z): (u32, u32, u32),
        native_tiling: bool,
    ) -> u32 {
        let size_shift = GOB_SIZE_B.ilog2();
        let width_shift = GOB_WIDTH_B.ilog2();
        let height_shift = GOB_HEIGHT.ilog2();
        let gobs_in_x = stride.div_ceil(1 << width_shift);
        let block_size = gobs_in_x << (size_shift + block_height + block_depth);
        let slice_size = height.div_ceil(1 << (block_height + height_shift)) * block_size;
        let block_height_mask = (1 << block_height) - 1;
        let block_depth_mask = (1 << block_depth) - 1;
        let x_shift = size_shift + block_height + block_depth;

        let offset_z = (z >> block_depth) * slice_size
            + ((z & block_depth_mask) << (size_shift + block_height));
        let block_y = y >> height_shift;
        let offset_y =
            (block_y >> block_height) * block_size + ((block_y & block_height_mask) << size_shift);
        let offset_x = (x >> width_shift) << x_shift;
        let swizzled = pdep(SWIZZLE_X_BITS, x) | pdep(SWIZZLE_Y_BITS, y);
        let offset = offset_z + offset_y + offset_x + swizzled;
        if native_tiling {
            native_gob_offset(offset)
        } else {
            offset
        }
    }

    fn every_gob_byte() -> impl Iterator<Item = (u32, u32)> {
        (0..GOB_HEIGHT).flat_map(|y| (0..GOB_WIDTH_B).map(move |x| (x, y)))
    }

    #[test]
    fn eden_swizzle_is_the_tegra_colour_gob() {
        for (x, y) in every_gob_byte() {
            assert_eq!(
                GobType::TegraColor.byte_offset(x, y),
                Ok(swizzle_offset(x, y)),
                "byte ({x}, {y})"
            );
        }
    }

    #[test]
    fn switch2_native_swizzle_is_the_turing_colour_gob() {
        for (x, y) in every_gob_byte() {
            assert_eq!(
                GobType::TuringColor2d.byte_offset(x, y),
                Ok(native_gob_offset(swizzle_offset(x, y))),
                "byte ({x}, {y})"
            );
        }
    }

    #[test]
    fn the_two_gobs_really_differ() {
        // Guards the test above against a table that ignores the GOB kind.
        let differing = every_gob_byte()
            .filter(|&(x, y)| {
                GobType::TegraColor.byte_offset(x, y) != GobType::TuringColor2d.byte_offset(x, y)
            })
            .count();
        // Eden's offset bits 5 and 6 hold x bit 4 and y bit 1, so the swap moves a byte exactly when
        // those two bits differ: half the GOB.
        assert_eq!(differing, (GOB_SIZE_B / 2) as usize);
    }

    /// The sector numbers the spec derives: Eden's `x4 + 2·y1 + 4·y2 + 8·x5` and Eden-NX2's
    /// `y1 + 2·x4 + 4·y2 + 8·x5` (bit `n` of x or y written `xn` or `yn`).
    #[test]
    fn sector_formulas_match_the_spec() {
        let bit = |value: u32, n: u32| (value >> n) & 1;
        for (x, y) in every_gob_byte() {
            let tegra_sector = bit(x, 4) + 2 * bit(y, 1) + 4 * bit(y, 2) + 8 * bit(x, 5);
            let turing_sector = bit(y, 1) + 2 * bit(x, 4) + 4 * bit(y, 2) + 8 * bit(x, 5);
            let in_sector = (y % 2) * GOB_LINE_WIDTH_B + x % GOB_LINE_WIDTH_B;
            assert_eq!(
                GobType::TegraColor.byte_offset(x, y),
                Ok(tegra_sector * GOB_SECTOR_B + in_sector)
            );
            assert_eq!(
                GobType::TuringColor2d.byte_offset(x, y),
                Ok(turing_sector * GOB_SECTOR_B + in_sector)
            );
        }
    }

    /// Whole-level offsets against Eden's `SwizzleImpl`, for several block heights and depths and
    /// level extents that are not whole tiles.
    #[test]
    fn level_offsets_match_eden_swizzle_impl() {
        let cases = [
            // (width in bytes, height, depth, block_height, block_depth)
            (64, 8, 1, 0, 0),
            (200, 37, 1, 2, 0),
            (130, 70, 3, 3, 1),
            (64, 300, 1, 5, 0),
            (100, 9, 9, 0, 2),
            (256, 16, 5, 1, 3),
        ];
        for (width, height, depth, block_height, block_depth) in cases {
            let level = Extent4D::new(width, height, depth, 1);
            for (gob, native) in [(GobType::TegraColor, false), (GobType::TuringColor2d, true)] {
                let tiling = Tiling::new(gob, 0, block_height, block_depth).unwrap();
                for z in 0..depth {
                    for y in 0..height {
                        for x in 0..width {
                            let expected = eden_level_offset(
                                width,
                                height,
                                u32::from(block_height),
                                u32::from(block_depth),
                                (x, y, z),
                                native,
                            );
                            assert_eq!(
                                block_linear_offset(&tiling, level, Offset4D::new(x, y, z, 0)),
                                Ok(u64::from(expected)),
                                "{gob:?} {width}x{height}x{depth} bh {block_height} bd {block_depth} at ({x}, {y}, {z})"
                            );
                        }
                    }
                }
            }
        }
    }
}
