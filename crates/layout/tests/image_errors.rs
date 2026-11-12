//! Every description `Image::new` refuses, the order it checks them in, the range checks of the
//! level queries, and the plain-words message of every error.

use starquake_layout::{
    Extent4D, Format, GobType, Image, ImageDesc, ImageDim, LayoutError, MAX_LEVELS, SampleLayout,
    Tiling,
};

fn turing() -> Tiling {
    Tiling::new(GobType::TuringColor2d, 0, 1, 0).unwrap()
}

fn desc(dim: ImageDim, extent: (u32, u32, u32, u32)) -> ImageDesc {
    ImageDesc {
        dim,
        format: Format::uncompressed(4).unwrap(),
        extent_px: Extent4D::new(extent.0, extent.1, extent.2, extent.3),
        levels: 1,
        sample_layout: SampleLayout::S1x1,
        tiling: turing(),
        pitch_b: None,
    }
}

#[test]
fn zero_extent_in_each_dimension() {
    for extent in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
        assert_eq!(
            Image::new(&desc(ImageDim::D3, extent)),
            Err(LayoutError::ZeroExtent),
            "{extent:?}"
        );
    }
}

#[test]
fn level_count_on_both_sides() {
    let with_levels = |levels| {
        Image::new(&ImageDesc {
            levels,
            ..desc(ImageDim::D2, (64, 64, 1, 1))
        })
    };
    assert!(with_levels(1).is_ok());
    let full = with_levels(MAX_LEVELS).unwrap();
    assert_eq!(full.levels().len(), MAX_LEVELS as usize);
    for levels in [0, MAX_LEVELS + 1] {
        assert_eq!(
            with_levels(levels),
            Err(LayoutError::InvalidLevelCount {
                levels,
                max: MAX_LEVELS
            })
        );
    }
}

#[test]
fn dimension_rules() {
    assert!(Image::new(&desc(ImageDim::D1, (64, 1, 1, 3))).is_ok());
    for extent in [(64, 2, 1, 1), (64, 1, 2, 1)] {
        assert_eq!(
            Image::new(&desc(ImageDim::D1, extent)),
            Err(LayoutError::Invalid1d)
        );
    }
    assert_eq!(
        Image::new(&ImageDesc {
            sample_layout: SampleLayout::S2x1,
            ..desc(ImageDim::D1, (64, 1, 1, 1))
        }),
        Err(LayoutError::Invalid1d)
    );

    assert!(Image::new(&desc(ImageDim::D2, (64, 64, 1, 6))).is_ok());
    assert_eq!(
        Image::new(&desc(ImageDim::D2, (64, 64, 2, 1))),
        Err(LayoutError::Invalid2d)
    );

    assert!(Image::new(&desc(ImageDim::D3, (64, 64, 4, 1))).is_ok());
    assert_eq!(
        Image::new(&desc(ImageDim::D3, (64, 64, 4, 2))),
        Err(LayoutError::Invalid3d)
    );
    assert_eq!(
        Image::new(&ImageDesc {
            sample_layout: SampleLayout::S2x2,
            ..desc(ImageDim::D3, (64, 64, 4, 1))
        }),
        Err(LayoutError::Invalid3d)
    );
}

#[test]
fn multisampled_images_have_one_level() {
    let msaa = |levels| {
        Image::new(&ImageDesc {
            sample_layout: SampleLayout::S4x2D3d,
            levels,
            ..desc(ImageDim::D2, (64, 64, 1, 1))
        })
    };
    assert!(msaa(1).is_ok());
    assert_eq!(msaa(2), Err(LayoutError::MultisampleMipmaps));
}

#[test]
fn pitch_linear_rules() {
    let linear = |dim, extent, levels, sample_layout| {
        Image::new(&ImageDesc {
            tiling: Tiling::LINEAR,
            levels,
            sample_layout,
            ..desc(dim, extent)
        })
    };
    assert!(linear(ImageDim::D2, (64, 64, 1, 1), 1, SampleLayout::S1x1).is_ok());
    let refused = [
        linear(ImageDim::D1, (64, 1, 1, 1), 1, SampleLayout::S1x1),
        linear(ImageDim::D3, (64, 64, 2, 1), 1, SampleLayout::S1x1),
        linear(ImageDim::D2, (64, 64, 1, 2), 1, SampleLayout::S1x1),
        linear(ImageDim::D2, (64, 64, 1, 1), 2, SampleLayout::S1x1),
        linear(ImageDim::D2, (64, 64, 1, 1), 1, SampleLayout::S2x2),
    ];
    for result in refused {
        assert_eq!(result, Err(LayoutError::InvalidLinear));
    }
}

#[test]
fn image_check_order() {
    // A zero extent before a bad level count.
    assert_eq!(
        Image::new(&ImageDesc {
            levels: 0,
            ..desc(ImageDim::D2, (0, 1, 1, 1))
        }),
        Err(LayoutError::ZeroExtent)
    );
    // A bad level count before the dimension's rule.
    assert_eq!(
        Image::new(&ImageDesc {
            levels: 0,
            ..desc(ImageDim::D2, (1, 1, 2, 1))
        }),
        Err(LayoutError::InvalidLevelCount {
            levels: 0,
            max: MAX_LEVELS
        })
    );
    // The dimension's rule before the multisample rule.
    assert_eq!(
        Image::new(&ImageDesc {
            levels: 2,
            sample_layout: SampleLayout::S2x2,
            ..desc(ImageDim::D2, (1, 1, 2, 1))
        }),
        Err(LayoutError::Invalid2d)
    );
    // The multisample rule before the pitch-linear rules.
    assert_eq!(
        Image::new(&ImageDesc {
            levels: 2,
            sample_layout: SampleLayout::S2x2,
            tiling: Tiling::LINEAR,
            ..desc(ImageDim::D2, (1, 1, 1, 2))
        }),
        Err(LayoutError::MultisampleMipmaps)
    );
    // The pitch-linear shape before the pitch.
    assert_eq!(
        Image::new(&ImageDesc {
            tiling: Tiling::LINEAR,
            pitch_b: Some(33),
            ..desc(ImageDim::D2, (1, 1, 1, 2))
        }),
        Err(LayoutError::InvalidLinear)
    );
}

/// 1D, R8, 2^32 - 64 wide: the row pads to exactly 2^32 - 64 bytes. One byte wider and padding
/// to a whole GOB width overflows 32 bits.
#[test]
fn padding_overflow_on_both_sides() {
    let wide = |width| {
        Image::new(&ImageDesc {
            format: Format::uncompressed(1).unwrap(),
            tiling: Tiling::new(GobType::TuringColor2d, 0, 0, 0).unwrap(),
            ..desc(ImageDim::D1, (width, 1, 1, 1))
        })
    };
    let largest = u32::MAX - (starquake_layout::GOB_WIDTH_B - 1);
    let image = wide(largest).unwrap();
    assert_eq!(image.levels()[0].row_stride_b, largest);
    assert_eq!(wide(largest + 1), Err(LayoutError::Overflow));
}

/// A 3D level whose padded size is past 64 bits.
#[test]
fn layer_size_overflow() {
    let huge = 1 << 31;
    assert_eq!(
        Image::new(&ImageDesc {
            format: Format::uncompressed(1).unwrap(),
            ..desc(ImageDim::D3, (huge, huge, huge, 1))
        }),
        Err(LayoutError::Overflow)
    );
}

#[test]
fn level_and_layer_ranges() {
    let image = Image::new(&ImageDesc {
        levels: 3,
        ..desc(ImageDim::D2, (64, 64, 1, 4))
    })
    .unwrap();
    assert!(image.level_layer_offset_b(2, 3).is_ok());
    assert_eq!(
        image.level_layer_offset_b(3, 0),
        Err(LayoutError::LevelOutOfRange {
            level: 3,
            levels: 3
        })
    );
    assert_eq!(
        image.level_layer_offset_b(0, 4),
        Err(LayoutError::LayerOutOfRange {
            layer: 4,
            layers: 4
        })
    );
    // The level is checked before the layer.
    assert_eq!(
        image.level_layer_offset_b(3, 4),
        Err(LayoutError::LevelOutOfRange {
            level: 3,
            levels: 3
        })
    );
    let out_of_range = Err(LayoutError::LevelOutOfRange {
        level: 3,
        levels: 3,
    });
    assert_eq!(image.level_extent_px(3).map(|_| ()), out_of_range);
    assert_eq!(image.level_extent_b(3).map(|_| ()), out_of_range);
    assert_eq!(image.level_layer_size_b(3).map(|_| ()), out_of_range);
    assert_eq!(image.level_size_b(3).map(|_| ()), out_of_range);
    assert_eq!(image.level_z_offset_b(3, 0).map(|_| ()), out_of_range);
    assert_eq!(
        image.level_extent_px(u32::MAX).map(|_| ()),
        Err(LayoutError::LevelOutOfRange {
            level: u32::MAX,
            levels: 3
        })
    );
}

#[test]
fn every_message_is_plain_words() {
    let cases = [
        (LayoutError::Overflow, "a layout size or offset overflows"),
        (
            LayoutError::ZeroExtent,
            "an image extent has a zero dimension",
        ),
        (
            LayoutError::InvalidFormat,
            "a format's element size and block dimensions must be non-zero",
        ),
        (
            LayoutError::InvalidSampleCount(3),
            "3 samples per pixel is not a supported sample count",
        ),
        (
            LayoutError::InvalidLevelCount {
                levels: 17,
                max: 16,
            },
            "17 levels is outside the supported range of 1 to 16",
        ),
        (
            LayoutError::Invalid1d,
            "a 1D image must have height 1, depth 1 and one sample",
        ),
        (LayoutError::Invalid2d, "a 2D image must have depth 1"),
        (
            LayoutError::Invalid3d,
            "a 3D image must have one layer and one sample",
        ),
        (
            LayoutError::MultisampleMipmaps,
            "a multisampled image must have exactly one level",
        ),
        (
            LayoutError::InvalidLinear,
            "a pitch-linear image must be 2D with one level, one layer and one sample",
        ),
        (
            LayoutError::PitchNotAligned {
                pitch: 33,
                align: 32,
            },
            "pitch 33 is not a multiple of 32 bytes",
        ),
        (
            LayoutError::PitchTooSmall { pitch: 64, row: 96 },
            "pitch 64 is smaller than a row of 96 bytes",
        ),
        (
            LayoutError::PitchOnBlockLinear,
            "a pitch can only be given for a pitch-linear image",
        ),
        (
            LayoutError::LinearWithBlock,
            "a linear tiling cannot have a block size",
        ),
        (
            LayoutError::BlockTooLarge {
                x_log2: 1,
                y_log2: 2,
                z_log2: 30,
            },
            "a block of 2^1 by 2^2 by 2^30 GOBs is too large",
        ),
        (
            LayoutError::UnsupportedGob(GobType::FermiZs),
            "the Fermi depth-stencil GOB kind has no known byte order",
        ),
        (
            LayoutError::UnsupportedGob(GobType::Linear),
            "the linear GOB kind has no known byte order",
        ),
        (
            LayoutError::OutsideGob { x: 64, y: 0 },
            "byte (64, 0) is outside a GOB",
        ),
        (
            LayoutError::LevelOutOfRange {
                level: 3,
                levels: 3,
            },
            "level 3 is out of range for an image with 3 levels",
        ),
        (
            LayoutError::LayerOutOfRange {
                layer: 4,
                layers: 4,
            },
            "layer 4 is out of range for an image with 4 layers",
        ),
        (
            LayoutError::SliceOutOfRange { z: 5, depth: 5 },
            "slice 5 is out of range for a level of depth 5",
        ),
        (LayoutError::NoMipTail, "the image has no mip tail"),
        (
            LayoutError::RegionHasLayers,
            "a region or position must be in layer 0 of one layer",
        ),
        (
            LayoutError::RegionOutOfBounds,
            "the region does not fit in the level",
        ),
        (
            LayoutError::TiledTooSmall {
                needed: 4096,
                len: 10,
            },
            "the tiled buffer holds 10 bytes, the level needs 4096",
        ),
        (
            LayoutError::LinearTooSmall {
                needed: 330,
                len: 10,
            },
            "the linear buffer holds 10 bytes, the region needs 330",
        ),
    ];
    for (error, message) in cases {
        assert_eq!(error.to_string(), message);
    }
    assert_eq!(GobType::TegraColor.name(), "Tegra colour");
    assert_eq!(GobType::TuringColor2d.name(), "Turing colour 2D");
}
