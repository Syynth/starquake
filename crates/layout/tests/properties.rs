//! Generated tests: tile-then-untile round trips over generated images, and no panic or
//! unchecked overflow for any description, tiling or copy.

/// In a `cfg(test)` module so the helpers may unwrap like the tests do.
#[cfg(test)]
mod properties {
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
    use starquake_layout::{
        Bytes, Extent4D, Format, GOB_SIZE_B, GobType, Image, ImageDesc, ImageDim, LayoutError,
        LinearLayout, MAX_LEVELS, Offset4D, SampleLayout, Tiling, linear_to_tiled, tiled_to_linear,
    };

    /// Fixed seed so a run is reproducible; failures are not persisted to disk.
    fn config(cases: u32) -> ProptestConfig {
        ProptestConfig {
            cases,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(0x4C41_594F_5554_4E49),
            failure_persistence: None,
            ..Config::default()
        }
    }

    fn colour_gob() -> impl Strategy<Value = GobType> {
        prop_oneof![Just(GobType::TegraColor), Just(GobType::TuringColor2d)]
    }

    fn any_gob() -> impl Strategy<Value = GobType> {
        prop_oneof![
            Just(GobType::Linear),
            Just(GobType::FermiZs),
            Just(GobType::TegraColor),
            Just(GobType::TuringColor2d),
        ]
    }

    fn any_sample_layout() -> impl Strategy<Value = SampleLayout> {
        prop_oneof![
            Just(SampleLayout::S1x1),
            Just(SampleLayout::S2x1),
            Just(SampleLayout::S2x1D3d),
            Just(SampleLayout::S2x2),
            Just(SampleLayout::S4x2),
            Just(SampleLayout::S4x2D3d),
            Just(SampleLayout::S4x4),
        ]
    }

    /// Half small numbers, half any `u32`, so both the accepted and the overflowing paths run.
    fn mostly_small() -> impl Strategy<Value = u32> {
        prop_oneof![0u32..=70, any::<u32>()]
    }

    fn any_dim() -> impl Strategy<Value = ImageDim> {
        prop_oneof![Just(ImageDim::D1), Just(ImageDim::D2), Just(ImageDim::D3)]
    }

    /// Small block-linear images that `Image::new` accepts: a format of 1 to 16 bytes per element
    /// with blocks up to 12 by 12, any dimension, up to 4 levels and 3 layers.
    fn small_image() -> impl Strategy<Value = ImageDesc> {
        (
            any_dim(),
            1u32..=16,
            prop_oneof![
                Just((1u32, 1u32)),
                Just((4, 4)),
                Just((8, 6)),
                Just((12, 12))
            ],
            (1u32..=90, 1u32..=40, 1u32..=6, 1u32..=3),
            1u32..=4,
            (colour_gob(), 0u8..=2, 0u8..=4, 0u8..=3),
        )
            .prop_map(
                |(dim, el_size, (block_w, block_h), (w, h, d, layers), levels, (gob, x, y, z))| {
                    let (h, d, layers) = match dim {
                        ImageDim::D1 => (1, 1, layers),
                        ImageDim::D2 => (h, 1, layers),
                        ImageDim::D3 => (h, d, 1),
                    };
                    let block_h = if dim == ImageDim::D1 { 1 } else { block_h };
                    ImageDesc {
                        dim,
                        format: Format::new(el_size, block_w, block_h, 1).unwrap(),
                        extent_px: Extent4D::new(w, h, d, layers),
                        levels,
                        sample_layout: SampleLayout::S1x1,
                        tiling: Tiling::new(gob, x, y, z).unwrap(),
                        pitch_b: None,
                    }
                },
            )
    }

    /// A pattern of bytes from a seed, distinct from its neighbours.
    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| {
                i.to_le_bytes()[0].wrapping_mul(13).wrapping_add(seed) ^ (i >> 8).to_le_bytes()[0]
            })
            .collect()
    }

    fn tight(extent: Extent4D<Bytes>) -> LinearLayout {
        LinearLayout {
            row_stride_b: extent.width as usize,
            plane_stride_b: extent.width as usize * extent.height as usize,
        }
    }

    proptest! {
        #![proptest_config(config(256))]

        /// Every level of every layer, tiled into the image's buffer at its offset and untiled back,
        /// comes back unchanged; and every level lies inside the image's size.
        #[test]
        fn whole_image_round_trip(desc in small_image(), seed in any::<u8>()) {
            let image = Image::new(&desc).unwrap();
            let size = usize::try_from(image.size_b()).unwrap();
            let mut tiled = vec![0u8; size];
            let mut originals = Vec::new();
            for layer in 0..desc.extent_px.array_len {
                for level in 0..desc.levels {
                    let offset = image.level_layer_offset_b(level, layer).unwrap();
                    let layer_size = image.level_layer_size_b(level).unwrap();
                    prop_assert!(offset + layer_size <= image.size_b());
                    let mut extent = image.level_extent_b(level).unwrap();
                    extent.array_len = 1;
                    let linear = pattern(
                        extent.width as usize * extent.height as usize * extent.depth as usize,
                        seed ^ level.to_le_bytes()[0] ^ (layer.to_le_bytes()[0] << 4),
                    );
                    let start = usize::try_from(offset).unwrap();
                    let tiling = image.levels()[level as usize].tiling;
                    linear_to_tiled(
                        &mut tiled[start..],
                        extent,
                        &linear,
                        tight(extent),
                        Offset4D::new(0, 0, 0, 0),
                        extent,
                        &tiling,
                    )
                    .unwrap();
                    originals.push((start, extent, tiling, linear));
                }
            }
            // Untile only after every level is written, so a level that overlapped another would
            // show as a mismatch.
            for (start, extent, tiling, linear) in originals {
                let mut back = vec![0u8; linear.len()];
                tiled_to_linear(
                    &mut back,
                    tight(extent),
                    &tiled[start..],
                    extent,
                    Offset4D::new(0, 0, 0, 0),
                    extent,
                    &tiling,
                )
                .unwrap();
                prop_assert_eq!(back, linear);
            }
        }

        /// A random sub-region of a level, with padded linear strides, round-trips.
        #[test]
        fn region_round_trip(
            gob in colour_gob(),
            (x_log2, y_log2, z_log2) in (0u8..=2, 0u8..=4, 0u8..=2),
            (w, h, d) in (1u32..=200, 1u32..=40, 1u32..=5),
            fractions in (0u32..1000, 0u32..1000, 0u32..1000, 0u32..1000, 0u32..1000, 0u32..1000),
            (row_pad, plane_pad) in (0usize..=17, 0usize..=33),
            seed in any::<u8>(),
        ) {
            let tiling = Tiling::new(gob, x_log2, y_log2, z_log2).unwrap();
            let level = Extent4D::new(w, h, d, 1);
            // A start in 0..size and a length in 1..=size - start, from two numbers in
            // 0..1000.
            let pick = |size: u32, start: u32, len: u32| {
                let start = u32::try_from(u64::from(size) * u64::from(start) / 1000).unwrap();
                let len = 1 + u32::try_from(u64::from(size - start - 1) * u64::from(len) / 999).unwrap();
                (start, len)
            };
            let (x, rw) = pick(w, fractions.0, fractions.1);
            let (y, rh) = pick(h, fractions.2, fractions.3);
            let (z, rd) = pick(d, fractions.4, fractions.5);
            let row = rw as usize + row_pad;
            let plane = row * rh as usize + plane_pad;
            let layout = LinearLayout { row_stride_b: row, plane_stride_b: plane };
            let linear = pattern(plane * rd as usize, seed);
            let tile = tiling.extent_b();
            let tiled_len = w.next_multiple_of(tile.width) as usize
                * h.next_multiple_of(tile.height) as usize
                * d.next_multiple_of(tile.depth) as usize;
            let mut tiled = vec![0u8; tiled_len];
            let offset = Offset4D::new(x, y, z, 0);
            let extent = Extent4D::new(rw, rh, rd, 1);
            linear_to_tiled(&mut tiled, level, &linear, layout, offset, extent, &tiling).unwrap();
            let mut back = vec![0u8; linear.len()];
            tiled_to_linear(&mut back, layout, &tiled, level, offset, extent, &tiling).unwrap();
            for zz in 0..rd as usize {
                for yy in 0..rh as usize {
                    let start = zz * plane + yy * row;
                    prop_assert_eq!(&back[start..start + rw as usize], &linear[start..start + rw as usize]);
                }
            }
            // The tiled level holds exactly the region's bytes and zeros elsewhere.
            let written = tiled.iter().filter(|&&b| b != 0).count();
            let nonzero_in_region = (0..rd as usize)
                .flat_map(|zz| (0..rh as usize).map(move |yy| (zz, yy)))
                .map(|(zz, yy)| {
                    let start = zz * plane + yy * row;
                    linear[start..start + rw as usize].iter().filter(|&&b| b != 0).count()
                })
                .sum::<usize>();
            prop_assert_eq!(written, nonzero_in_region);
        }
    }

    proptest! {
        #![proptest_config(config(2048))]

        /// `Image::new` and every query return a value or an error for any description, however
        /// large; a description it accepts keeps every level inside the image.
        #[test]
        fn image_never_panics(
            dim in any_dim(),
            (el, bw, bh, bd) in (mostly_small(), mostly_small(), mostly_small(), mostly_small()),
            (w, h, d, a) in (mostly_small(), mostly_small(), mostly_small(), mostly_small()),
            levels in 0u32..=MAX_LEVELS + 2,
            sample_layout in any_sample_layout(),
            (gob, x, y, z) in (any_gob(), 0u8..=12, 0u8..=12, 0u8..=12),
            pitch in proptest::option::of(mostly_small()),
            (level, layer, slice) in (0u32..=MAX_LEVELS, mostly_small(), mostly_small()),
        ) {
            let Ok(format) = Format::new(el, bw, bh, bd) else { return Ok(()) };
            let Ok(tiling) = Tiling::new(gob, x, y, z) else { return Ok(()) };
            let desc = ImageDesc {
                dim,
                format,
                extent_px: Extent4D::new(w, h, d, a),
                levels,
                sample_layout,
                tiling,
                pitch_b: pitch,
            };
            if let Ok(image) = Image::new(&desc) {
                for level in 0..levels {
                    let end = image.level_layer_offset_b(level, 0).unwrap()
                        + image.level_layer_size_b(level).unwrap();
                    prop_assert!(end <= image.size_b());
                }
                let _ = image.level_layer_offset_b(level, layer);
                let _ = image.level_size_b(level);
                let _ = image.level_z_offset_b(level, slice);
                let _ = image.mip_tail_size_b();
            }
        }

        /// `Tiling::new` accepts exactly the tilings whose block fits in 32 bits.
        #[test]
        fn tiling_accepts_exactly_what_fits(gob in any_gob(), x in any::<u8>(), y in any::<u8>(), z in any::<u8>()) {
            let result = Tiling::new(gob, x, y, z);
            let fits = u32::from(x) + u32::from(y) + u32::from(z) + GOB_SIZE_B.ilog2() <= 31;
            match gob {
                GobType::Linear if (x, y, z) == (0, 0, 0) => prop_assert_eq!(result, Ok(Tiling::LINEAR)),
                GobType::Linear => prop_assert_eq!(result, Err(LayoutError::LinearWithBlock)),
                _ if fits => {
                    let tiling = result.unwrap();
                    prop_assert_eq!(u64::from(tiling.size_b()), 1u64 << (GOB_SIZE_B.ilog2() + u32::from(x) + u32::from(y) + u32::from(z)));
                }
                _ => prop_assert_eq!(result, Err(LayoutError::BlockTooLarge { x_log2: x, y_log2: y, z_log2: z })),
            }
        }

        /// A copy returns a value or an error for any region, strides and buffer sizes.
        #[test]
        fn copies_never_panic(
            (gob, x_log2, y_log2, z_log2) in (any_gob(), 0u8..=8, 0u8..=8, 0u8..=8),
            level in (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()),
            offset in (any::<u32>(), any::<u32>(), any::<u32>(), 0u32..=1),
            extent in (any::<u32>(), any::<u32>(), any::<u32>(), 0u32..=2),
            small in (0u32..=130, 0u32..=20, 0u32..=3),
            strides in (any::<usize>(), any::<usize>()),
            (tiled_len, linear_len) in (0usize..=8192, 0usize..=4096),
        ) {
            let Ok(tiling) = Tiling::new(gob, x_log2, y_log2, z_log2) else { return Ok(()) };
            // Half the cases use a small level, so the size checks and the copy loop run too.
            let level = if level.3 % 2 == 0 {
                Extent4D::new(small.0, small.1, small.2, 1)
            } else {
                Extent4D::new(level.0, level.1, level.2, level.3)
            };
            let offset = Offset4D::new(offset.0 % 200, offset.1 % 30, offset.2 % 4, offset.3);
            let extent = Extent4D::new(extent.0 % 200, extent.1 % 30, extent.2 % 4, extent.3);
            let layout = LinearLayout { row_stride_b: strides.0 % 300, plane_stride_b: strides.1 % 5000 };
            let mut tiled = vec![0u8; tiled_len];
            let mut linear = vec![0u8; linear_len];
            let _ = linear_to_tiled(&mut tiled, level, &linear, layout, offset, extent, &tiling);
            let _ = tiled_to_linear(&mut linear, layout, &tiled, level, offset, extent, &tiling);
        }
    }
}
