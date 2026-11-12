//! Tiling and untiling regions: where each byte goes, what is left alone, and every refusal.

use starquake_layout::{
    Extent4D, GOB_LINE_WIDTH_B, GobType, LayoutError, LinearLayout, Offset4D, Tiling,
    block_linear_offset, linear_to_tiled, tiled_to_linear,
};

fn tiling(gob: GobType, x: u8, y: u8, z: u8) -> Tiling {
    Tiling::new(gob, x, y, z).unwrap()
}

/// Bytes in a level padded to whole tiles.
fn padded_size(tiling: &Tiling, level: Extent4D<starquake_layout::Bytes>) -> usize {
    let tile = tiling.extent_b();
    (level.width.next_multiple_of(tile.width) as usize)
        * (level.height.next_multiple_of(tile.height) as usize)
        * (level.depth.next_multiple_of(tile.depth) as usize)
}

/// A pattern where every byte of a buffer below 2^24 bytes differs from its neighbours.
fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 7 + i / 251) as u8).collect()
}

/// Untiling a whole level puts each linear byte at exactly `block_linear_offset` of the tiled
/// level, for a level that is not whole tiles in any dimension.
#[test]
fn untile_reads_each_byte_from_its_block_linear_offset() {
    let tiling = tiling(GobType::TuringColor2d, 1, 1, 1);
    let level = Extent4D::new(150, 21, 3, 1);
    let tiled = pattern(padded_size(&tiling, level));
    let layout = LinearLayout {
        row_stride_b: 150,
        plane_stride_b: 150 * 21,
    };
    let mut linear = vec![0; 150 * 21 * 3];
    tiled_to_linear(
        &mut linear,
        layout,
        &tiled,
        level,
        Offset4D::new(0, 0, 0, 0),
        level,
        &tiling,
    )
    .unwrap();
    for z in 0..3 {
        for y in 0..21 {
            for x in 0..150 {
                let at = block_linear_offset(&tiling, level, Offset4D::new(x, y, z, 0)).unwrap();
                let index = (z * 150 * 21 + y * 150 + x) as usize;
                assert_eq!(linear[index], tiled[at as usize], "byte ({x}, {y}, {z})");
            }
        }
    }
}

/// Tiling a sub-region writes exactly the region's bytes, from the linear slice laid out with
/// padding strides, and leaves every other byte of the level as it was. The region starts and
/// ends inside 16-byte lines, so pieces shorter than a line are exercised on both sides.
#[test]
fn tile_a_region_leaves_its_neighbours_alone() {
    let tiling = tiling(GobType::TegraColor, 0, 2, 0);
    let level = Extent4D::new(200, 40, 1, 1);
    let offset = Offset4D::new(GOB_LINE_WIDTH_B * 2 + 3, 5, 0, 0);
    let extent = Extent4D::new(70, 20, 1, 1);
    let layout = LinearLayout {
        row_stride_b: 80,
        plane_stride_b: 80 * 20,
    };
    let linear = pattern(layout.plane_stride_b);
    let before = vec![0xEE; padded_size(&tiling, level)];
    let mut tiled = before.clone();
    linear_to_tiled(&mut tiled, level, &linear, layout, offset, extent, &tiling).unwrap();

    let mut expected = before;
    for y in 0..extent.height {
        for x in 0..extent.width {
            let at = block_linear_offset(
                &tiling,
                level,
                Offset4D::new(offset.x + x, offset.y + y, 0, 0),
            )
            .unwrap();
            expected[at as usize] = linear[(y as usize) * layout.row_stride_b + x as usize];
        }
    }
    assert_eq!(tiled, expected);
}

/// Untiling a region writes only the region's bytes of the linear slice: the padding between
/// rows keeps its old value.
#[test]
fn untile_a_region_leaves_linear_padding_alone() {
    let tiling = tiling(GobType::TuringColor2d, 0, 0, 0);
    let level = Extent4D::new(64, 8, 1, 1);
    let tiled = pattern(padded_size(&tiling, level));
    let layout = LinearLayout {
        row_stride_b: 20,
        plane_stride_b: 0,
    };
    let mut linear = vec![0xEE; 20 * 3];
    tiled_to_linear(
        &mut linear,
        layout,
        &tiled,
        level,
        Offset4D::new(10, 2, 0, 0),
        Extent4D::new(15, 3, 1, 1),
        &tiling,
    )
    .unwrap();
    for y in 0..3u32 {
        for x in 0..20u32 {
            let value = linear[(y * 20 + x) as usize];
            if x < 15 {
                let at = block_linear_offset(&tiling, level, Offset4D::new(10 + x, 2 + y, 0, 0))
                    .unwrap();
                assert_eq!(value, tiled[at as usize]);
            } else {
                assert_eq!(value, 0xEE, "padding at ({x}, {y})");
            }
        }
    }
}

#[test]
fn an_empty_region_copies_nothing() {
    let tiling = tiling(GobType::TuringColor2d, 0, 0, 0);
    let level = Extent4D::new(64, 8, 1, 1);
    let mut tiled = vec![0xEE; padded_size(&tiling, level)];
    let layout = LinearLayout {
        row_stride_b: 64,
        plane_stride_b: 512,
    };
    for extent in [
        Extent4D::new(0, 8, 1, 1),
        Extent4D::new(64, 0, 1, 1),
        Extent4D::new(64, 8, 0, 1),
    ] {
        // An empty region needs no linear bytes at all.
        linear_to_tiled(
            &mut tiled,
            level,
            &[],
            layout,
            Offset4D::new(0, 0, 0, 0),
            extent,
            &tiling,
        )
        .unwrap();
        assert!(tiled.iter().all(|&byte| byte == 0xEE));
    }
}

struct Fixture {
    tiling: Tiling,
    level: Extent4D<starquake_layout::Bytes>,
    tiled: Vec<u8>,
    layout: LinearLayout,
    linear: Vec<u8>,
}

/// A 100 by 10 by 2 level and a whole-level linear buffer.
fn fixture() -> Fixture {
    let tiling = tiling(GobType::TuringColor2d, 0, 0, 0);
    let level = Extent4D::new(100, 10, 2, 1);
    Fixture {
        tiling,
        level,
        tiled: vec![0; padded_size(&tiling, level)],
        layout: LinearLayout {
            row_stride_b: 100,
            plane_stride_b: 1000,
        },
        linear: vec![0; 2000],
    }
}

/// Runs both directions with the same arguments and checks they agree on the error.
fn both_directions(
    f: &mut Fixture,
    tiling: &Tiling,
    tiled_len: usize,
    linear_len: usize,
    offset: Offset4D<starquake_layout::Bytes>,
    extent: Extent4D<starquake_layout::Bytes>,
) -> Result<(), LayoutError> {
    let to_linear = tiled_to_linear(
        &mut f.linear[..linear_len],
        f.layout,
        &f.tiled[..tiled_len],
        f.level,
        offset,
        extent,
        tiling,
    );
    let to_tiled = linear_to_tiled(
        &mut f.tiled[..tiled_len],
        f.level,
        &f.linear[..linear_len],
        f.layout,
        offset,
        extent,
        tiling,
    );
    assert_eq!(to_linear, to_tiled);
    to_linear
}

#[test]
fn region_must_be_one_layer_at_layer_0() {
    let mut f = fixture();
    let (t, tl, ll) = (f.tiling, f.tiled.len(), f.linear.len());
    let origin = Offset4D::new(0, 0, 0, 0);
    let one = Extent4D::new(1, 1, 1, 1);
    assert_eq!(both_directions(&mut f, &t, tl, ll, origin, one), Ok(()));
    assert_eq!(
        both_directions(&mut f, &t, tl, ll, Offset4D::new(0, 0, 0, 1), one),
        Err(LayoutError::RegionHasLayers)
    );
    for layers in [0, 2] {
        assert_eq!(
            both_directions(&mut f, &t, tl, ll, origin, Extent4D::new(1, 1, 1, layers)),
            Err(LayoutError::RegionHasLayers)
        );
    }
}

/// The region may end exactly at the level's edge in each dimension, not one past it, and an
/// end past 32 bits is out of bounds.
#[test]
fn region_bounds_on_both_sides() {
    let mut f = fixture();
    let (t, tl, ll) = (f.tiling, f.tiled.len(), f.linear.len());
    let level = f.level;
    assert_eq!(
        both_directions(&mut f, &t, tl, ll, Offset4D::new(0, 0, 0, 0), level),
        Ok(())
    );
    let inside = [
        (Offset4D::new(99, 0, 0, 0), Extent4D::new(1, 1, 1, 1)),
        (Offset4D::new(0, 9, 0, 0), Extent4D::new(1, 1, 1, 1)),
        (Offset4D::new(0, 0, 1, 0), Extent4D::new(1, 1, 1, 1)),
    ];
    for (offset, extent) in inside {
        assert_eq!(both_directions(&mut f, &t, tl, ll, offset, extent), Ok(()));
    }
    let outside = [
        (Offset4D::new(99, 0, 0, 0), Extent4D::new(2, 1, 1, 1)),
        (Offset4D::new(0, 9, 0, 0), Extent4D::new(1, 2, 1, 1)),
        (Offset4D::new(0, 0, 1, 0), Extent4D::new(1, 1, 2, 1)),
        (Offset4D::new(1, 0, 0, 0), Extent4D::new(u32::MAX, 1, 1, 1)),
        (Offset4D::new(0, u32::MAX, 0, 0), Extent4D::new(1, 1, 1, 1)),
    ];
    for (offset, extent) in outside {
        assert_eq!(
            both_directions(&mut f, &t, tl, ll, offset, extent),
            Err(LayoutError::RegionOutOfBounds),
            "{offset:?} {extent:?}"
        );
    }
}

#[test]
fn unsupported_gobs_are_refused() {
    let mut f = fixture();
    let (tl, ll, level) = (f.tiled.len(), f.linear.len(), f.level);
    for (t, gob) in [
        (Tiling::LINEAR, GobType::Linear),
        (tiling(GobType::FermiZs, 0, 0, 0), GobType::FermiZs),
    ] {
        assert_eq!(
            both_directions(&mut f, &t, tl, ll, Offset4D::new(0, 0, 0, 0), level),
            Err(LayoutError::UnsupportedGob(gob))
        );
    }
}

/// The tiled slice must hold the whole padded level: 128 by 16 by 2 = 4096 bytes here.
#[test]
fn tiled_buffer_size_on_both_sides() {
    let mut f = fixture();
    let (t, ll) = (f.tiling, f.linear.len());
    let needed = f.tiled.len();
    assert_eq!(needed, 4096);
    let origin = Offset4D::new(0, 0, 0, 0);
    let one = Extent4D::new(1, 1, 1, 1);
    assert_eq!(both_directions(&mut f, &t, needed, ll, origin, one), Ok(()));
    assert_eq!(
        both_directions(&mut f, &t, needed - 1, ll, origin, one),
        Err(LayoutError::TiledTooSmall {
            needed: needed as u64,
            len: needed - 1
        })
    );
}

/// The linear slice must reach the region's last byte: for 30 by 4 by 2 at strides 100 and
/// 1000, (2 - 1) * 1000 + (4 - 1) * 100 + 30 = 1330 bytes.
#[test]
fn linear_buffer_size_on_both_sides() {
    let mut f = fixture();
    let (t, tl) = (f.tiling, f.tiled.len());
    let origin = Offset4D::new(0, 0, 0, 0);
    let extent = Extent4D::new(30, 4, 2, 1);
    assert_eq!(
        both_directions(&mut f, &t, tl, 1330, origin, extent),
        Ok(())
    );
    assert_eq!(
        both_directions(&mut f, &t, tl, 1329, origin, extent),
        Err(LayoutError::LinearTooSmall {
            needed: 1330,
            len: 1329
        })
    );
}

/// Strides so large that the region's last byte is past 64 bits.
#[test]
fn linear_extent_overflow() {
    let tiling = tiling(GobType::TuringColor2d, 0, 0, 0);
    let level = Extent4D::new(64, 8, 3, 1);
    let mut tiled = vec![0; padded_size(&tiling, level)];
    let layout = LinearLayout {
        row_stride_b: usize::MAX,
        plane_stride_b: usize::MAX,
    };
    assert_eq!(
        linear_to_tiled(
            &mut tiled,
            level,
            &[],
            layout,
            Offset4D::new(0, 0, 0, 0),
            Extent4D::new(1, 2, 3, 1),
            &tiling,
        ),
        Err(LayoutError::Overflow)
    );
}

/// With two faults at once, the earlier check in the documented order wins.
#[test]
fn copy_check_order() {
    let mut f = fixture();
    let t = f.tiling;
    let fermi = tiling(GobType::FermiZs, 0, 0, 0);
    let origin = Offset4D::new(0, 0, 0, 0);
    let one = Extent4D::new(1, 1, 1, 1);
    let past_end = Extent4D::new(101, 1, 1, 1);
    // Layers before bounds.
    assert_eq!(
        both_directions(&mut f, &t, 4096, 2000, Offset4D::new(0, 0, 0, 1), past_end),
        Err(LayoutError::RegionHasLayers)
    );
    // Bounds before the GOB kind.
    assert_eq!(
        both_directions(&mut f, &fermi, 4096, 2000, origin, past_end),
        Err(LayoutError::RegionOutOfBounds)
    );
    // The GOB kind before the tiled size.
    assert_eq!(
        both_directions(&mut f, &fermi, 0, 2000, origin, one),
        Err(LayoutError::UnsupportedGob(GobType::FermiZs))
    );
    // The tiled size before the linear size.
    assert_eq!(
        both_directions(&mut f, &t, 0, 0, origin, one),
        Err(LayoutError::TiledTooSmall {
            needed: 4096,
            len: 0
        })
    );
}

#[test]
fn a_refused_copy_writes_nothing() {
    let mut f = fixture();
    f.linear.fill(0xAA);
    let tiled_before = f.tiled.clone();
    let result = linear_to_tiled(
        &mut f.tiled,
        f.level,
        &f.linear[..10],
        f.layout,
        Offset4D::new(0, 0, 0, 0),
        Extent4D::new(30, 4, 1, 1),
        &f.tiling,
    );
    assert_eq!(
        result,
        Err(LayoutError::LinearTooSmall {
            needed: 330,
            len: 10
        })
    );
    assert_eq!(f.tiled, tiled_before);
}
