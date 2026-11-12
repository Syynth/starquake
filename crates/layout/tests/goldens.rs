//! Hand-computed offsets and layouts.
//!
//! The sector grids are transcribed from the `GOBType` documentation in
//! `mesa-nouveau/src/nouveau/nil/tiling.rs` @ 52238762, which is independent of the line lists
//! the crate's tables are ported from (`nil/copy.rs`). Every other expectation is worked by
//! hand in the comment above it.

use starquake_layout::{
    Extent4D, Format, GOB_HEIGHT, GOB_LINE_WIDTH_B, GOB_SECTOR_B, GOB_SIZE_B, GOB_WIDTH_B, GobType,
    Image, ImageDesc, ImageDim, ImageLevel, LINEAR_DEFAULT_ALIGN_B, LINEAR_PITCH_ALIGN_B,
    LayoutError, MIN_IMAGE_ALIGN_B, Offset4D, SampleLayout, Tiling, block_linear_offset,
};

/// Mesa's sector grids, `[row][column]`: four columns of 16-byte lines, four rows of two lines.
const TEGRA_SECTORS: [[u32; 4]; 4] = [[0, 1, 8, 9], [2, 3, 10, 11], [4, 5, 12, 13], [6, 7, 14, 15]];
const TURING_SECTORS: [[u32; 4]; 4] =
    [[0, 2, 8, 10], [1, 3, 9, 11], [4, 6, 12, 14], [5, 7, 13, 15]];

fn check_sector_grid(gob: GobType, grid: &[[u32; 4]; 4]) {
    let lines_per_sector = GOB_SECTOR_B / GOB_LINE_WIDTH_B;
    for (row, sectors) in (0u32..).zip(grid) {
        for (column, &sector) in (0u32..).zip(sectors) {
            for line in 0..lines_per_sector {
                for byte in 0..GOB_LINE_WIDTH_B {
                    let x = column * GOB_LINE_WIDTH_B + byte;
                    let y = row * lines_per_sector + line;
                    assert_eq!(
                        gob.byte_offset(x, y),
                        Ok(sector * GOB_SECTOR_B + line * GOB_LINE_WIDTH_B + byte),
                        "{gob:?} byte ({x}, {y})"
                    );
                }
            }
        }
    }
}

#[test]
fn tegra_gob_matches_mesa_sector_grid() {
    check_sector_grid(GobType::TegraColor, &TEGRA_SECTORS);
}

#[test]
fn turing_gob_matches_mesa_sector_grid() {
    check_sector_grid(GobType::TuringColor2d, &TURING_SECTORS);
}

#[test]
fn every_gob_is_a_permutation_of_its_bytes() {
    for gob in [GobType::TegraColor, GobType::TuringColor2d] {
        let mut seen = vec![false; GOB_SIZE_B as usize];
        for y in 0..GOB_HEIGHT {
            for x in 0..GOB_WIDTH_B {
                let offset = gob.byte_offset(x, y).unwrap() as usize;
                assert!(!seen[offset], "{gob:?} offset {offset} twice");
                seen[offset] = true;
            }
        }
    }
}

#[test]
fn gob_extents() {
    assert_eq!(GobType::Linear.extent_b(), Extent4D::new(1, 1, 1, 1));
    for gob in [
        GobType::FermiZs,
        GobType::TegraColor,
        GobType::TuringColor2d,
    ] {
        assert_eq!(gob.extent_b(), Extent4D::new(GOB_WIDTH_B, GOB_HEIGHT, 1, 1));
    }
}

#[test]
fn byte_offset_edges_and_unsupported_kinds() {
    let last_x = GOB_WIDTH_B - 1;
    let last_y = GOB_HEIGHT - 1;
    assert!(GobType::TuringColor2d.byte_offset(last_x, last_y).is_ok());
    assert_eq!(
        GobType::TuringColor2d.byte_offset(GOB_WIDTH_B, 0),
        Err(LayoutError::OutsideGob {
            x: GOB_WIDTH_B,
            y: 0
        })
    );
    assert_eq!(
        GobType::TuringColor2d.byte_offset(0, GOB_HEIGHT),
        Err(LayoutError::OutsideGob {
            x: 0,
            y: GOB_HEIGHT
        })
    );
    // An unsupported kind is reported before an outside position.
    for gob in [GobType::Linear, GobType::FermiZs] {
        assert_eq!(gob.byte_offset(0, 0), Err(LayoutError::UnsupportedGob(gob)));
        assert_eq!(
            gob.byte_offset(GOB_WIDTH_B, GOB_HEIGHT),
            Err(LayoutError::UnsupportedGob(gob))
        );
    }
}

fn tiling(gob: GobType, x: u8, y: u8, z: u8) -> Tiling {
    Tiling::new(gob, x, y, z).unwrap()
}

/// Turing GOB, tiles 1 by 2 GOBs (64 by 16 bytes), a level 128 by 32 bytes.
/// Byte (70, 9): tile column 1, row 0 → tile 1 → 1 * 1024 = 1024. In the tile (6, 9): GOB row
/// 1 → 512. In the GOB (6, 1): Turing line at y 1, x 0 is 0x10 → 0x10 + 6 = 22.
/// 1024 + 512 + 22 = 1558.
#[test]
fn offset_in_a_two_gob_tall_tile() {
    let offset = block_linear_offset(
        &tiling(GobType::TuringColor2d, 0, 1, 0),
        Extent4D::new(128, 32, 1, 1),
        Offset4D::new(70, 9, 0, 0),
    );
    assert_eq!(offset, Ok(1558));
}

/// Tegra GOB, tiles 2 by 1 GOBs (128 by 8 bytes), a level 256 by 16 bytes.
/// Byte (200, 13): tile column 1, row 1 → tile 1 * 2 + 1 = 3 → 3 * 1024 = 3072. In the tile
/// (72, 5): GOB column 1 (GOBs run x first inside a tile) → 512. In the GOB (8, 5): Tegra line
/// at y 5, x 0 is 0x90 → 0x90 + 8 = 152. 3072 + 512 + 152 = 3736.
#[test]
fn offset_in_a_two_gob_wide_tile() {
    let offset = block_linear_offset(
        &tiling(GobType::TegraColor, 1, 0, 0),
        Extent4D::new(256, 16, 1, 1),
        Offset4D::new(200, 13, 0, 0),
    );
    assert_eq!(offset, Ok(3736));
}

/// Turing GOB, tiles 1 by 1 by 2 GOBs (64 by 8 by 2), a level 64 by 8 by 4 bytes.
/// Byte (17, 2, 2): tile layer 1 → 1024. In the tile z 0 → 0. In the GOB (17, 2): Turing line at
/// y 2, x 16 is 0x60 → 0x60 + 1 = 97. Total 1121. At z 3 the slice inside the tile is 1, one GOB
/// further: 1024 + 512 + 97 = 1633.
#[test]
fn offset_in_a_two_slice_tile() {
    let tiling = tiling(GobType::TuringColor2d, 0, 0, 1);
    let level = Extent4D::new(64, 8, 4, 1);
    assert_eq!(
        block_linear_offset(&tiling, level, Offset4D::new(17, 2, 2, 0)),
        Ok(1121)
    );
    assert_eq!(
        block_linear_offset(&tiling, level, Offset4D::new(17, 2, 3, 0)),
        Ok(1633)
    );
}

#[test]
fn offset_bounds_on_both_sides() {
    let tiling = tiling(GobType::TuringColor2d, 0, 0, 0);
    let level = Extent4D::new(100, 10, 2, 1);
    assert!(block_linear_offset(&tiling, level, Offset4D::new(99, 9, 1, 0)).is_ok());
    for position in [
        Offset4D::new(100, 0, 0, 0),
        Offset4D::new(0, 10, 0, 0),
        Offset4D::new(0, 0, 2, 0),
    ] {
        assert_eq!(
            block_linear_offset(&tiling, level, position),
            Err(LayoutError::RegionOutOfBounds)
        );
    }
}

#[test]
fn offset_error_order() {
    let level = Extent4D::new(64, 8, 1, 1);
    let outside = Offset4D::new(64, 0, 0, 0);
    let fermi = tiling(GobType::FermiZs, 0, 0, 0);
    // A layer is reported before an outside position, which is reported before the GOB kind.
    assert_eq!(
        block_linear_offset(&fermi, level, Offset4D::new(64, 0, 0, 1)),
        Err(LayoutError::RegionHasLayers)
    );
    assert_eq!(
        block_linear_offset(&fermi, level, outside),
        Err(LayoutError::RegionOutOfBounds)
    );
    assert_eq!(
        block_linear_offset(&fermi, level, Offset4D::new(0, 0, 0, 0)),
        Err(LayoutError::UnsupportedGob(GobType::FermiZs))
    );
    assert_eq!(
        block_linear_offset(&Tiling::LINEAR, level, Offset4D::new(0, 0, 0, 0)),
        Err(LayoutError::UnsupportedGob(GobType::Linear))
    );
}

#[test]
fn offset_of_a_level_too_large_for_64_bits() {
    let tiling = tiling(GobType::TuringColor2d, 0, 0, 0);
    let level = Extent4D::new(
        u32::MAX - (GOB_WIDTH_B - 1),
        u32::MAX - (GOB_HEIGHT - 1),
        u32::MAX,
        1,
    );
    assert_eq!(
        block_linear_offset(&tiling, level, Offset4D::new(0, 0, 0, 0)),
        Err(LayoutError::Overflow)
    );
}

fn rgba8() -> Format {
    Format::uncompressed(4).unwrap()
}

fn desc_2d(width: u32, height: u32, layers: u32, levels: u32, tiling: Tiling) -> ImageDesc {
    ImageDesc {
        dim: ImageDim::D2,
        format: rgba8(),
        extent_px: Extent4D::new(width, height, 1, layers),
        levels,
        sample_layout: SampleLayout::S1x1,
        tiling,
        pitch_b: None,
    }
}

fn level(offset_b: u64, tiling: Tiling, row_stride_b: u32) -> ImageLevel {
    ImageLevel {
        offset_b,
        tiling,
        row_stride_b,
    }
}

/// RGBA8, 100 by 50, 4 levels, 2 layers, Turing tiles 1 by 4 GOBs (64 by 32 bytes).
///
/// - Level 0: 400 by 50 bytes. 7 GOBs tall → ceil(log2 7) = 3 ≥ 2, tiling kept. Padded to 448
///   by 64 = 28672 bytes at offset 0.
/// - Level 1: 50 by 25 px = 200 by 25 bytes. 4 GOBs tall → 2, kept. Padded 256 by 32 = 8192 at
///   28672.
/// - Level 2: 25 by 12 px = 100 by 12. 2 GOBs tall → 1: tiling 1 by 2 GOBs, the mip tail
///   starts here. Padded 128 by 16 = 2048 at 36864.
/// - Level 3: 12 by 6 px = 48 by 6. 1 GOB → 0: one GOB. Padded 64 by 8 = 512 at 38912.
///
/// A layer is 39424 bytes; rounded to a level-0 tile (2048) the stride is 40960. Two layers are
/// 81920, already a multiple of the 4096 alignment (max(2048, 4096)).
#[test]
fn mipmapped_array_layout() {
    let requested = tiling(GobType::TuringColor2d, 0, 2, 0);
    let image = Image::new(&desc_2d(100, 50, 2, 4, requested)).unwrap();
    assert_eq!(
        image.levels(),
        &[
            level(0, requested, 448),
            level(28672, requested, 256),
            level(36864, tiling(GobType::TuringColor2d, 0, 1, 0), 128),
            level(38912, tiling(GobType::TuringColor2d, 0, 0, 0), 64),
        ]
    );
    assert_eq!(image.array_stride_b(), 40960);
    assert_eq!(image.align_b(), MIN_IMAGE_ALIGN_B);
    assert_eq!(image.size_b(), 81920);
    assert_eq!(image.mip_tail_first_level(), Some(2));
    assert_eq!(image.mip_tail_offset_b(), Ok(36864));
    assert_eq!(image.mip_tail_size_b(), Ok(40960 - 36864));
    assert_eq!(image.level_layer_offset_b(3, 1), Ok(40960 + 38912));
    assert_eq!(image.level_layer_size_b(2), Ok(2048));
    assert_eq!(image.level_size_b(0), Ok(40960 + 28672));
    assert_eq!(image.level_extent_px(2), Ok(Extent4D::new(25, 12, 1, 2)));
    assert_eq!(image.level_extent_b(2), Ok(Extent4D::new(100, 12, 1, 2)));
    assert_eq!(image.dim(), ImageDim::D2);
    assert_eq!(image.format(), rgba8());
    assert_eq!(image.extent_px(), Extent4D::new(100, 50, 1, 2));
    assert_eq!(image.sample_layout(), SampleLayout::S1x1);
}

/// R8, 3D 64 by 8 by 5, 2 levels, Turing tiles 1 by 1 by 4 GOBs (64 by 8 by 4).
///
/// - Level 0: 64 by 8 by 5 bytes; 5 slices → ceil(log2 5) = 3 ≥ 2, kept. Padded depth 8:
///   64 * 8 * 8 = 4096 at 0.
/// - Level 1: 32 by 4 by 2; narrower than a tile, so x stays 0; 2 slices → 1: tiles 1 by 1 by 2,
///   the mip tail starts here. Padded 64 by 8 by 2 = 1024 at 4096.
///
/// A layer is 5120; a level-0 tile is 2048, so the stride is 6144, and the size rounds to 8192.
/// Slice 3 of level 0 is the fourth GOB of the first tile layer: 3 * 512 = 1536. Slice 4 starts
/// the second tile layer: 1 tile * 2048 = 2048.
#[test]
fn three_d_layout_and_slice_offsets() {
    let requested = tiling(GobType::TuringColor2d, 0, 0, 2);
    let image = Image::new(&ImageDesc {
        dim: ImageDim::D3,
        format: Format::uncompressed(1).unwrap(),
        extent_px: Extent4D::new(64, 8, 5, 1),
        levels: 2,
        sample_layout: SampleLayout::S1x1,
        tiling: requested,
        pitch_b: None,
    })
    .unwrap();
    assert_eq!(
        image.levels(),
        &[
            level(0, requested, 64),
            level(4096, tiling(GobType::TuringColor2d, 0, 0, 1), 64),
        ]
    );
    assert_eq!(image.array_stride_b(), 6144);
    assert_eq!(image.size_b(), 8192);
    assert_eq!(image.mip_tail_first_level(), Some(1));
    assert_eq!(image.level_z_offset_b(0, 3), Ok(1536));
    assert_eq!(image.level_z_offset_b(0, 4), Ok(2048));
    // The slice offsets agree with the block-linear offset of the slice's first byte.
    for z in 0..5 {
        assert_eq!(
            image.level_z_offset_b(0, z),
            block_linear_offset(
                &requested,
                Extent4D::new(64, 8, 5, 1),
                Offset4D::new(0, 0, z, 0)
            )
        );
    }
    assert_eq!(image.level_z_offset_b(1, 1), Ok(512));
    assert_eq!(
        image.level_z_offset_b(0, 5),
        Err(LayoutError::SliceOutOfRange { z: 5, depth: 5 })
    );
    assert_eq!(
        image.level_z_offset_b(1, 2),
        Err(LayoutError::SliceOutOfRange { z: 2, depth: 2 })
    );
}

/// ASTC 8 by 8 (16-byte blocks), 100 by 100 px: 13 by 13 blocks = 208 bytes by 13 rows. With
/// tiles 1 by 16 GOBs requested, level 0 is 2 GOBs tall → tiles 1 by 2 GOBs (64 by 16), so the
/// mip tail starts at level 0 (a difference from Mesa, which asserts a tail starts above level
/// 0). Padded 256 by 16 = 4096 bytes; size 4096.
#[test]
fn compressed_format_and_a_tail_from_level_0() {
    let image = Image::new(&ImageDesc {
        format: Format::new(16, 8, 8, 1).unwrap(),
        ..desc_2d(100, 100, 1, 1, tiling(GobType::TuringColor2d, 0, 4, 0))
    })
    .unwrap();
    assert_eq!(
        image.levels(),
        &[level(0, tiling(GobType::TuringColor2d, 0, 1, 0), 256)]
    );
    assert_eq!(image.level_extent_b(0), Ok(Extent4D::new(208, 13, 1, 1)));
    assert_eq!(image.size_b(), 4096);
    assert_eq!(image.mip_tail_first_level(), Some(0));
    assert_eq!(image.mip_tail_offset_b(), Ok(0));
    assert_eq!(image.mip_tail_size_b(), Ok(4096));
}

/// RGBA8, 16 by 16 px, 4 samples in a 2 by 2 grid: 32 by 32 samples = 128 by 32 bytes; tiles 1
/// by 2 GOBs kept (4 GOBs tall). Padded 128 by 32 = 4096; no mip tail.
#[test]
fn multisampled_layout() {
    let requested = tiling(GobType::TuringColor2d, 0, 1, 0);
    let image = Image::new(&ImageDesc {
        sample_layout: SampleLayout::S2x2,
        ..desc_2d(16, 16, 1, 1, requested)
    })
    .unwrap();
    assert_eq!(image.levels(), &[level(0, requested, 128)]);
    assert_eq!(image.size_b(), 4096);
    assert_eq!(image.mip_tail_first_level(), None);
    assert_eq!(image.mip_tail_offset_b(), Err(LayoutError::NoMipTail));
    assert_eq!(image.mip_tail_size_b(), Err(LayoutError::NoMipTail));
}

/// Tiles 2 GOBs wide, level 0 only 100 bytes wide: Mesa's clamp drops the width to 1 GOB.
#[test]
fn a_narrow_level_drops_the_tile_width() {
    let image = Image::new(&desc_2d(
        25,
        8,
        1,
        1,
        tiling(GobType::TuringColor2d, 1, 0, 0),
    ))
    .unwrap();
    assert_eq!(
        image.levels()[0].tiling,
        tiling(GobType::TuringColor2d, 0, 0, 0)
    );
    let image = Image::new(&desc_2d(
        32,
        8,
        1,
        1,
        tiling(GobType::TuringColor2d, 1, 0, 0),
    ))
    .unwrap();
    assert_eq!(
        image.levels()[0].tiling,
        tiling(GobType::TuringColor2d, 1, 0, 0)
    );
}

#[test]
fn clamp_follows_mesa() {
    let wide_deep = tiling(GobType::TuringColor2d, 1, 0, 1);
    // Mesa drops the width when any dimension is smaller than a tile, depth included.
    assert_eq!(
        wide_deep.clamp(Extent4D::new(256, 8, 1, 1)),
        Ok(tiling(GobType::TuringColor2d, 0, 0, 0))
    );
    assert_eq!(wide_deep.clamp(Extent4D::new(256, 8, 2, 1)), Ok(wide_deep));
    // Height shrinks to the power of two of GOBs covering the level: 17 rows are 3 GOBs → 2.
    let tall = tiling(GobType::TegraColor, 0, 5, 0);
    assert_eq!(
        tall.clamp(Extent4D::new(64, 17, 1, 1)),
        Ok(tiling(GobType::TegraColor, 0, 2, 0))
    );
    // At exactly 4 GOBs (32 rows) the height is 2; one more row needs 3.
    assert_eq!(
        tall.clamp(Extent4D::new(64, 4 * GOB_HEIGHT, 1, 1)),
        Ok(tiling(GobType::TegraColor, 0, 2, 0))
    );
    assert_eq!(
        tall.clamp(Extent4D::new(64, 4 * GOB_HEIGHT + 1, 1, 1)),
        Ok(tiling(GobType::TegraColor, 0, 3, 0))
    );
    assert_eq!(
        Tiling::LINEAR.clamp(Extent4D::new(1, 1, 1, 1)),
        Ok(Tiling::LINEAR)
    );
}

#[test]
fn tiling_accessors_and_sizes() {
    let t = tiling(GobType::TegraColor, 1, 2, 3);
    assert_eq!(t.gob_type(), GobType::TegraColor);
    assert_eq!((t.x_log2(), t.y_log2(), t.z_log2()), (1, 2, 3));
    assert!(t.is_tiled());
    assert!(!Tiling::LINEAR.is_tiled());
    assert_eq!(Tiling::default(), Tiling::LINEAR);
    assert_eq!(
        t.extent_b(),
        Extent4D::new(GOB_WIDTH_B << 1, GOB_HEIGHT << 2, 1 << 3, 1)
    );
    assert_eq!(t.size_b(), GOB_SIZE_B << (1 + 2 + 3));
    assert_eq!(Tiling::LINEAR.size_b(), 1);
    assert_eq!(Tiling::new(GobType::Linear, 0, 0, 0), Ok(Tiling::LINEAR));
}

#[test]
fn linear_tilings_cannot_have_a_block() {
    for (x, y, z) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
        assert_eq!(
            Tiling::new(GobType::Linear, x, y, z),
            Err(LayoutError::LinearWithBlock)
        );
    }
}

/// A tile's byte size must fit in 32 bits: 2^(9 + x + y + z) for a 512-byte GOB, so the log2s
/// may add up to 31 - 9 = 22.
#[test]
fn block_size_limit_on_both_sides() {
    let max_sum = u8::try_from(31 - GOB_SIZE_B.ilog2()).unwrap();
    for split in [
        (max_sum, 0, 0),
        (0, max_sum, 0),
        (0, 0, max_sum),
        (10, 6, 6),
    ] {
        let (x, y, z) = split;
        assert!(
            Tiling::new(GobType::TuringColor2d, x, y, z).is_ok(),
            "{split:?}"
        );
    }
    for split in [
        (max_sum + 1, 0, 0),
        (0, max_sum + 1, 0),
        (0, 0, max_sum + 1),
        (10, 7, 6),
    ] {
        let (x, y, z) = split;
        assert_eq!(
            Tiling::new(GobType::TuringColor2d, x, y, z),
            Err(LayoutError::BlockTooLarge {
                x_log2: x,
                y_log2: y,
                z_log2: z
            }),
            "{split:?}"
        );
    }
    // A shift past 32 bits fails before any product.
    assert_eq!(
        Tiling::new(GobType::TuringColor2d, 0, 0, 200),
        Err(LayoutError::BlockTooLarge {
            x_log2: 0,
            y_log2: 0,
            z_log2: 200
        })
    );
}

/// RGBA8 30 by 4: a row is 120 bytes. With no pitch it rounds to 128 (alignment 128): 512
/// bytes. With pitch 160 (a multiple of 32): 640 bytes, alignment 32, and the level's bytes are
/// three full rows plus one row of data: 160 * 3 + 120 = 600.
#[test]
fn pitch_linear_layouts() {
    let image = Image::new(&desc_2d(30, 4, 1, 1, Tiling::LINEAR)).unwrap();
    assert_eq!(image.levels(), &[level(0, Tiling::LINEAR, 128)]);
    assert_eq!(image.align_b(), LINEAR_DEFAULT_ALIGN_B);
    assert_eq!(image.size_b(), 512);
    assert_eq!(image.array_stride_b(), 0);
    assert_eq!(image.mip_tail_first_level(), None);

    let image = Image::new(&ImageDesc {
        pitch_b: Some(160),
        ..desc_2d(30, 4, 1, 1, Tiling::LINEAR)
    })
    .unwrap();
    assert_eq!(image.levels(), &[level(0, Tiling::LINEAR, 160)]);
    assert_eq!(image.align_b(), LINEAR_PITCH_ALIGN_B);
    assert_eq!(image.size_b(), 640);
    assert_eq!(image.level_layer_size_b(0), Ok(600));
    assert_eq!(image.level_size_b(0), Ok(600));
    assert_eq!(image.level_z_offset_b(0, 0), Ok(0));
}

/// RGBA8 24 by 1: a row is 96 bytes, a multiple of 32. Pitch 96 is the smallest allowed; 64 is
/// too small; 100 is not a multiple of 32.
#[test]
fn pitch_checks_at_their_edges() {
    let with_pitch = |pitch| {
        Image::new(&ImageDesc {
            pitch_b: Some(pitch),
            ..desc_2d(24, 1, 1, 1, Tiling::LINEAR)
        })
    };
    assert!(with_pitch(96).is_ok());
    assert_eq!(
        with_pitch(96 - LINEAR_PITCH_ALIGN_B),
        Err(LayoutError::PitchTooSmall { pitch: 64, row: 96 })
    );
    assert_eq!(
        with_pitch(100),
        Err(LayoutError::PitchNotAligned {
            pitch: 100,
            align: LINEAR_PITCH_ALIGN_B
        })
    );
    // An unaligned pitch is reported before a small one.
    assert_eq!(
        with_pitch(33),
        Err(LayoutError::PitchNotAligned {
            pitch: 33,
            align: LINEAR_PITCH_ALIGN_B
        })
    );
    assert_eq!(
        Image::new(&ImageDesc {
            pitch_b: Some(128),
            ..desc_2d(24, 1, 1, 1, tiling(GobType::TuringColor2d, 0, 0, 0))
        }),
        Err(LayoutError::PitchOnBlockLinear)
    );
}

/// The image Eden-NX2 hit in blocker-log row 87: 8192 by 8192, 68 layers of RGBA8, about 18 GB.
/// It lays out without overflow; each layer is 8192 * 4 * 8192 bytes = 256 MiB.
#[test]
fn a_huge_virtual_texture_lays_out() {
    let image = Image::new(&desc_2d(
        8192,
        8192,
        68,
        1,
        tiling(GobType::TuringColor2d, 0, 4, 0),
    ))
    .unwrap();
    let layer = 8192 * 4 * 8192;
    assert_eq!(image.array_stride_b(), layer);
    assert_eq!(image.size_b(), layer * 68);
    assert!(image.size_b() > u64::from(u32::MAX));
}

#[test]
fn every_sample_layout() {
    let cases = [
        (SampleLayout::S1x1, 1, (1, 1)),
        (SampleLayout::S2x1, 2, (2, 1)),
        (SampleLayout::S2x1D3d, 2, (2, 1)),
        (SampleLayout::S2x2, 4, (2, 2)),
        (SampleLayout::S4x2, 8, (4, 2)),
        (SampleLayout::S4x2D3d, 8, (4, 2)),
        (SampleLayout::S4x4, 16, (4, 4)),
    ];
    for (layout, samples, (grid_w, grid_h)) in cases {
        assert_eq!(layout.samples(), samples, "{layout:?}");
        // The grid shows in the byte extent of a 3 by 5 RGBA8 image.
        assert_eq!(
            Extent4D::new(3, 5, 1, 1).to_b(rgba8(), layout),
            Ok(Extent4D::new(3 * grid_w * 4, 5 * grid_h, 1, 1)),
            "{layout:?}"
        );
    }
    let chosen = [
        (1, SampleLayout::S1x1),
        (2, SampleLayout::S2x1D3d),
        (4, SampleLayout::S2x2),
        (8, SampleLayout::S4x2D3d),
        (16, SampleLayout::S4x4),
    ];
    for (samples, layout) in chosen {
        assert_eq!(SampleLayout::from_samples(samples), Ok(layout));
    }
    for samples in [0, 3, 5, 32] {
        assert_eq!(
            SampleLayout::from_samples(samples),
            Err(LayoutError::InvalidSampleCount(samples))
        );
    }
}

/// Pixel offsets round down to the element; pixel extents round up (ASTC 8 by 8, 16 bytes).
#[test]
fn pixel_to_byte_conversions() {
    let astc = Format::new(16, 8, 8, 1).unwrap();
    assert_eq!(
        Offset4D::new(15, 16, 2, 3).to_b(astc, SampleLayout::S1x1),
        Ok(Offset4D::new(16, 2, 2, 3))
    );
    assert_eq!(
        Extent4D::new(9, 16, 1, 1).to_b(astc, SampleLayout::S1x1),
        Ok(Extent4D::new(32, 2, 1, 1))
    );
    assert_eq!(
        Offset4D::new(u32::MAX, 0, 0, 0).to_b(rgba8(), SampleLayout::S1x1),
        Err(LayoutError::Overflow)
    );
    assert_eq!(
        Extent4D::new(u32::MAX, 1, 1, 1).to_b(rgba8(), SampleLayout::S1x1),
        Err(LayoutError::Overflow)
    );
}
