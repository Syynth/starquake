//! The shared surface walker and the pending-write list.

use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, RngSeed};
use starquake_layout::LayoutError;

use super::super::surface::{
    BlockFields, BlockLinearDesc, CopySide, MAX_BLOCK_LOG2, ONE_GOB, Region, Surface,
};
use super::super::{CopyError, MAX_PENDING_RANGES};
use super::*;
use crate::classes::{cl902d, cla140, clc7b5, clc7c0, clc797};

#[test]
fn every_class_defines_the_same_largest_block() {
    assert_eq!(MAX_BLOCK_LOG2, 5);
    assert_eq!(ONE_GOB, 0);
    for top in [
        clc7b5::SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS,
        clc7b5::SET_DST_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS,
        clc7b5::SET_SRC_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS,
        clc7b5::SET_SRC_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS,
        cla140::SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS,
        cla140::SET_DST_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS,
        clc797::SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS,
        clc797::SET_DST_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS,
        clc7c0::SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS,
        clc7c0::SET_DST_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS,
        cl902d::SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS,
        cl902d::SET_DST_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS,
        cl902d::SET_SRC_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS,
        cl902d::SET_SRC_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS,
    ] {
        assert_eq!(top, MAX_BLOCK_LOG2);
    }
    for one in [
        clc7b5::SET_SRC_BLOCK_SIZE_WIDTH_ONE_GOB,
        cla140::SET_DST_BLOCK_SIZE_WIDTH_ONE_GOB,
        clc797::SET_DST_BLOCK_SIZE_WIDTH_ONE_GOB,
        clc7c0::SET_DST_BLOCK_SIZE_WIDTH_ONE_GOB,
    ] {
        assert_eq!(one, ONE_GOB);
    }
}

fn desc(
    width_b: u64,
    height: u32,
    depth: u32,
    bh: u32,
    bd: u32,
    origin: (u64, u32),
) -> BlockLinearDesc {
    BlockLinearDesc {
        block: BlockFields {
            width: ONE_GOB,
            height: bh,
            depth: bd,
        },
        width_b,
        height,
        depth,
        origin_x_b: origin.0,
        origin_y: origin.1,
    }
}

#[test]
fn a_width_or_origin_past_32_bits_is_an_overflow() {
    let overflow = Err(CopyError::Layout {
        side: CopySide::Source,
        error: LayoutError::Overflow,
    });
    let big = u64::from(u32::MAX) + 1;
    assert_eq!(
        Surface::block_linear(CopySide::Source, 0, &desc(big, 1, 1, 0, 0, (0, 0))),
        overflow
    );
    assert_eq!(
        Surface::block_linear(CopySide::Source, 0, &desc(64, 1, 1, 0, 0, (big, 0))),
        overflow
    );
    assert!(
        Surface::block_linear(
            CopySide::Source,
            0,
            &desc(big - 1, 1, 1, 0, 0, (big - 1, 0))
        )
        .is_ok()
    );
}

#[test]
fn a_cover_past_the_address_space_is_an_extent_error() {
    let surface = Surface::block_linear(
        CopySide::Destination,
        u64::MAX - 64 * 8 + 1,
        &desc(64, 8, 1, 0, 0, (0, 0)),
    )
    .unwrap();
    let region = Region { width: 1, lines: 1 };
    // The padded surface is 512 bytes: ending exactly at 2^64 is not representable as an end.
    assert_eq!(
        surface.cover(region),
        Err(CopyError::Extent {
            side: CopySide::Destination
        })
    );
    let surface = Surface::block_linear(
        CopySide::Destination,
        u64::MAX - 64 * 8,
        &desc(64, 8, 1, 0, 0, (0, 0)),
    )
    .unwrap();
    assert_eq!(surface.cover(region), Ok(Some(u64::MAX - 512..u64::MAX)));
    // A pitch line ending exactly at u64::MAX fits; one more byte does not.
    let pitch = Surface::pitch(CopySide::Source, u64::MAX - 0x20, 0x10);
    assert_eq!(
        pitch.cover(Region {
            width: 0x10,
            lines: 2
        }),
        Ok(Some(u64::MAX - 0x20..u64::MAX))
    );
    assert_eq!(
        pitch.cover(Region {
            width: 0x11,
            lines: 2
        }),
        Err(CopyError::Extent {
            side: CopySide::Source
        })
    );
    // An empty region has no cover, whatever its surface.
    assert_eq!(pitch.cover(Region { width: 0, lines: 2 }), Ok(None));
    assert_eq!(pitch.cover(Region { width: 2, lines: 0 }), Ok(None));
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        rng_seed: RngSeed::Fixed(0x5355_5246),
        ..ProptestConfig::default()
    })]

    /// Every byte a block-linear region reaches lies inside its cover, and each run maps byte
    /// for byte to the layout crate's offsets.
    #[test]
    fn every_run_lies_in_the_cover_and_matches_the_layout_crate(
        width in 1u32..300,
        height in 1u32..40,
        depth in 1u32..4,
        bh in 0u32..=3,
        bd in 0u32..=2,
        ox in 0u32..50,
        oy in 0u32..20,
        w in 1u32..100,
        lines in 1u32..60,
    ) {
        let width = width + ox + w;
        let height = height + oy;
        let mem = memory();
        let surface = Surface::block_linear(
            CopySide::Source,
            GPU_BASE,
            &desc(u64::from(width), height, depth, bh, bd, (u64::from(ox), oy)),
        )
        .unwrap();
        let per = height - oy;
        let region = Region { width: u64::from(w), lines };
        let cover = surface.cover(region);
        if (lines - 1) / per >= depth {
            prop_assert_eq!(
                cover,
                Err(CopyError::Layout { side: CopySide::Source, error: LayoutError::RegionOutOfBounds })
            );
            return Ok(());
        }
        let cover = cover.unwrap().unwrap();
        let bl = Bl { width, height, depth, bh: u8::try_from(bh).unwrap(), bd: u8::try_from(bd).unwrap() };
        prop_assert_eq!(cover.clone(), GPU_BASE..GPU_BASE + bl.size());
        for line in 0..lines {
            let mut buf = vec![0; w as usize];
            surface.read(&mem, line, 0..u64::from(w), &mut buf).unwrap();
            for x in 0..w {
                let offset = starquake_layout::block_linear_offset(
                    &bl.tiling(),
                    bl.extent(),
                    Offset4D::new(ox + x, oy + line % per, line / per, 0),
                )
                .unwrap();
                prop_assert!(GPU_BASE + offset < cover.end);
                prop_assert_eq!(buf[x as usize], read(&mem, GPU_BASE + offset, 1)[0]);
            }
        }
    }
}

// --- PendingWrites ---

#[test]
fn pending_writes_merge_overlapping_and_touching_ranges() {
    // Overlap on either side, touching on either side, containment, and a disjoint range.
    assert_eq!(pending_after(&[10..20, 15..25]), vec![10..25]);
    assert_eq!(pending_after(&[15..25, 10..20]), vec![10..25]);
    assert_eq!(pending_after(&[10..20, 20..30]), vec![10..30]);
    assert_eq!(pending_after(&[20..30, 10..20]), vec![10..30]);
    assert_eq!(pending_after(&[10..40, 20..30]), vec![10..40]);
    assert_eq!(pending_after(&[20..30, 10..40]), vec![10..40]);
    // One byte apart stays apart.
    assert_eq!(pending_after(&[10..20, 21..30]), vec![10..20, 21..30]);
    // A range bridging several merges them all.
    assert_eq!(
        pending_after(&[0..5, 10..15, 20..25, 30..35, 4..21]),
        vec![0..25, 30..35]
    );
    // Out-of-order pushes come out sorted.
    assert_eq!(
        pending_after(&[50..60, 0..5, 30..31]),
        vec![0..5, 30..31, 50..60]
    );
    // An empty range is ignored.
    assert_eq!(
        pending_after(&[5..5, Range { start: 7, end: 3 }]),
        Vec::<Range<u64>>::new()
    );
}

#[test]
fn pending_writes_collapse_past_the_cap() {
    // MAX_PENDING_RANGES disjoint ranges stay; one more collapses everything into one cover.
    let ranges: Vec<Range<u64>> = (0..MAX_PENDING_RANGES as u64)
        .map(|i| i * 10..i * 10 + 5)
        .collect();
    assert_eq!(pending_after(&ranges), ranges);
    let mut more = ranges.clone();
    more.push(1_000_000..1_000_001);
    assert_eq!(pending_after(&more), vec![0..1_000_001]);
    // A range that merges at the cap does not collapse.
    let mut merging = ranges.clone();
    merging.push(5..10);
    assert_eq!(pending_after(&merging).len(), MAX_PENDING_RANGES - 1);
}

#[test]
fn flush_hands_over_every_range_in_order_and_empties_the_list() {
    let mut pending = PendingWrites::new();
    pending.add(30..40);
    pending.add(0..10);
    let mut sink = Recorder::default();
    pending.flush(&mut sink);
    assert_eq!(
        sink.events,
        vec![Event::Invalidate(0..10), Event::Invalidate(30..40)]
    );
    assert!(pending.is_empty());
    pending.flush(&mut sink);
    assert_eq!(sink.events.len(), 2);
}

#[test]
fn region_bytes_cannot_overflow() {
    assert_eq!(
        Region {
            width: u64::MAX,
            lines: u32::MAX
        }
        .bytes(),
        u128::from(u64::MAX) * u128::from(u32::MAX)
    );
    assert!(Region { width: 0, lines: 3 }.is_empty());
    assert!(Region { width: 3, lines: 0 }.is_empty());
    assert!(!Region { width: 1, lines: 1 }.is_empty());
}
