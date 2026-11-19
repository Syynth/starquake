//! Tests of the texture header decode and the format table.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};
use starquake_layout::{GobType, ImageDim, SampleLayout};
use starquake_render::PixelFormat;
use starquake_render::resource::{Swizzle, SwizzleSource, ViewDimension};

use super::super::format::{FLOAT, SINT, SNORM, UINT, UNORM};
use super::super::{
    CUBE_FACES, FORMAT_TABLE, HEADER_WORDS, HeaderError, ImageInfo, MemoryLayout, TextureHeader,
    ViewInfo, gob_for, pixel_format,
};
use super::{Build, bl_header};
use crate::classes::clc797tex as tex;

// --- Hand-built headers, one per header form ---

#[test]
fn a_hand_built_block_linear_header_decodes_every_field() {
    // Word by word from the header's bit ranges (`clc797tex.h`):
    // 0: COMPONENTS A8B8G8R8 (0x08); R/G/B/A_DATA_TYPE NUM_UNORM (2) at 7, 10, 13, 16;
    //    X_SOURCE IN_B (4) at 19, Y IN_G (3) at 22, Z IN_R (2) at 25, W IN_ONE_FLOAT (7) at 28.
    // 1: ADDRESS_BITS31TO9 (address bits 9..31 in place) plus GOB_DEPTH_OFFSET = 3 at bits 5..6.
    // 2: ADDRESS_BITS48TO32 = 0x12; HEADER_VERSION SELECT_BLOCKLINEAR (3) at 21.
    // 3: GOBS_PER_BLOCK_HEIGHT SIXTEEN_GOBS (4) at 3; MAX_MIP_LEVEL 2 at 28.
    // 4: WIDTH_MINUS_ONE 299; TEXTURE_TYPE TWO_D_ARRAY (5) at 23.
    // 5: HEIGHT_MINUS_ONE 199; DEPTH_MINUS_ONE 3 at 16.
    // 7: RES_VIEW_MIN_MIP_LEVEL 1; RES_VIEW_MAX_MIP_LEVEL 2 at 4.
    let words = [
        0x74E2_4908,
        0x3456_7A60,
        0x0060_0012,
        0x2000_0020,
        0x0280_012B,
        0x0003_00C7,
        0,
        0x0000_0021,
    ];
    assert_eq!(
        TextureHeader::decode(&words),
        Ok(TextureHeader {
            image: ImageInfo {
                address: 0x12_3456_7A00,
                format: PixelFormat::A8B8G8R8Unorm,
                dim: ImageDim::D2,
                width: 300,
                height: 200,
                depth: 1,
                layers: 4,
                levels: 3,
                samples: SampleLayout::S1x1,
                layout: MemoryLayout::BlockLinear {
                    gob: GobType::TuringColor2d,
                    block_height_log2: 4,
                    block_depth_log2: 0,
                },
            },
            view: ViewInfo {
                dimension: ViewDimension::D2Array,
                base_mip: 1,
                mip_count: 2,
                swizzle: Swizzle {
                    r: SwizzleSource::B,
                    g: SwizzleSource::G,
                    b: SwizzleSource::R,
                    a: SwizzleSource::One,
                },
            },
        })
    );
}

#[test]
fn a_hand_built_colour_key_block_linear_header_views_every_level() {
    // As above with HEADER_VERSION SELECT_BLOCKLINEAR_COLOR_KEY (4), and word 7 the colour key:
    // the form has no view levels and no multisample mode.
    let words = [
        0x74E2_4908,
        0x3456_7A60,
        0x0080_0012,
        0x2000_0020,
        0x0280_012B,
        0x0003_00C7,
        0,
        0xDEAD_BEEF,
    ];
    let header = TextureHeader::decode(&words).unwrap();
    assert_eq!(header.image.samples, SampleLayout::S1x1);
    assert_eq!((header.view.base_mip, header.view.mip_count), (0, 3));
    assert_eq!(header.image.address, 0x12_3456_7A00);
    assert_eq!((header.image.width, header.image.height), (300, 200));
}

#[test]
fn a_hand_built_pitch_header_decodes_every_field() {
    // 0: COMPONENTS R8 (0x1D), NUM_UNORM types, sources IN_R, IN_G, IN_B, IN_A.
    // 1: ADDRESS_BITS31TO5 (address bits 5..31 in place) plus 0x1F in the reserved bits 0..4.
    // 2: ADDRESS_BITS48TO32 = 1; HEADER_VERSION SELECT_PITCH (2) at 21.
    // 3: PITCH_BITS20TO5 = 0x22.
    // 4: WIDTH_MINUS_ONE 999; PITCH_BIT21 at 17; TEXTURE_TYPE TWO_D (1) at 23.
    // 5: HEIGHT_MINUS_ONE 9.
    let words = [
        0x58D2_491D,
        0x0000_013F,
        0x0040_0001,
        0x0000_0022,
        0x0082_03E7,
        0x0000_0009,
        0,
        0,
    ];
    assert_eq!(
        TextureHeader::decode(&words),
        Ok(TextureHeader {
            image: ImageInfo {
                address: 0x1_0000_0120,
                format: PixelFormat::R8Unorm,
                dim: ImageDim::D2,
                width: 1000,
                height: 10,
                depth: 1,
                layers: 1,
                levels: 1,
                samples: SampleLayout::S1x1,
                layout: MemoryLayout::Pitch { pitch_b: 0x20_0440 },
            },
            view: ViewInfo {
                dimension: ViewDimension::D2,
                base_mip: 0,
                mip_count: 1,
                swizzle: Swizzle::IDENTITY,
            },
        })
    );
}

#[test]
fn a_hand_built_colour_key_pitch_header_decodes_its_pitch() {
    // As above with HEADER_VERSION SELECT_PITCH_COLOR_KEY (1) and a colour key in word 7.
    let words = [
        0x58D2_491D,
        0x0000_013F,
        0x0020_0001,
        0x0000_0022,
        0x0082_03E7,
        0x0000_0009,
        0,
        0x1234_5678,
    ];
    let header = TextureHeader::decode(&words).unwrap();
    assert_eq!(
        header.image.layout,
        MemoryLayout::Pitch { pitch_b: 0x20_0440 }
    );
    assert_eq!(header.image.address, 0x1_0000_0120);
    assert_eq!((header.view.base_mip, header.view.mip_count), (0, 1));
}

// --- Divergences from Eden ---

#[test]
fn the_address_skips_the_low_reserved_bits() {
    // Every bit below the low address field set: none of them is an address bit.
    let words = bl_header(0)
        .set(tex::TEXHEAD_BL_RESERVED1Y, 0x1F)
        .set(tex::TEXHEAD_BL_GOB_DEPTH_OFFSET, 3)
        .set(tex::TEXHEAD_BL_RESERVED1X, 3)
        .set(tex::TEXHEAD_BL_ADDRESS_BITS31TO9, 1)
        .words();
    assert_eq!(TextureHeader::decode(&words).unwrap().image.address, 1 << 9);
    // The high field has 17 bits (Eden reads 16): bit 48 is an address bit.
    let words = bl_header(0)
        .set(tex::TEXHEAD_BL_ADDRESS_BITS48TO32, 0x1_0000)
        .words();
    assert_eq!(
        TextureHeader::decode(&words).unwrap().image.address,
        1 << 48
    );
    let words = Build::new()
        .set(
            tex::TEXHEAD_PITCH_HEADER_VERSION,
            tex::TEXHEAD_PITCH_HEADER_VERSION_SELECT_PITCH,
        )
        .set(
            tex::TEXHEAD_PITCH_COMPONENTS,
            tex::TEXHEAD_PITCH_COMPONENTS_SIZES_R8,
        )
        .types(tex::TEXHEAD_PITCH_R_DATA_TYPE.low(), UNORM)
        .identity(tex::TEXHEAD_PITCH_X_SOURCE.low())
        .set(
            tex::TEXHEAD_PITCH_TEXTURE_TYPE,
            tex::TEXHEAD_PITCH_TEXTURE_TYPE_TWO_D,
        )
        .set(tex::TEXHEAD_PITCH_RESERVED1A, 0x1F)
        .set(tex::TEXHEAD_PITCH_ADDRESS_BITS31TO5, 1)
        .words();
    assert_eq!(TextureHeader::decode(&words).unwrap().image.address, 1 << 5);
}

#[test]
fn width_and_height_take_their_top_bits() {
    let words = bl_header(0)
        .set(tex::TEXHEAD_BL_WIDTH_MINUS_ONE, 0x1_FFFF)
        .set(tex::TEXHEAD_BL_HEIGHT_MINUS_ONE, 0xFFFF)
        .set(tex::TEXHEAD_BL_HEIGHT_MINUS_ONE_BIT16, 1)
        .words();
    let image = TextureHeader::decode(&words).unwrap().image;
    assert_eq!((image.width, image.height), (1 << 17, 1 << 17));
    // Only the top bit set: a value that differs from Eden's 16-bit read only above the low 16.
    let words = bl_header(0)
        .set(tex::TEXHEAD_BL_WIDTH_MINUS_ONE, 0x1_0000)
        .set(tex::TEXHEAD_BL_HEIGHT_MINUS_ONE, 0)
        .set(tex::TEXHEAD_BL_HEIGHT_MINUS_ONE_BIT16, 1)
        .words();
    let image = TextureHeader::decode(&words).unwrap().image;
    assert_eq!((image.width, image.height), (0x1_0001, 0x1_0001));
    // A block-linear depth has 15 bits; a pitch header's bit 145 is the pitch, not the depth.
    let words = bl_header(0)
        .set(
            tex::TEXHEAD_BL_TEXTURE_TYPE,
            tex::TEXHEAD_BL_TEXTURE_TYPE_THREE_D,
        )
        .set(tex::TEXHEAD_BL_DEPTH_MINUS_ONE, 1)
        .set(tex::TEXHEAD_BL_DEPTH_MINUS_ONE_BIT14, 1)
        .words();
    assert_eq!(TextureHeader::decode(&words).unwrap().image.depth, 0x4002);
}

#[test]
fn both_ones_become_one_and_each_source_maps() {
    let cases = [
        (tex::TEXHEAD_BL_X_SOURCE_IN_ZERO, SwizzleSource::Zero),
        (tex::TEXHEAD_BL_X_SOURCE_IN_R, SwizzleSource::R),
        (tex::TEXHEAD_BL_X_SOURCE_IN_G, SwizzleSource::G),
        (tex::TEXHEAD_BL_X_SOURCE_IN_B, SwizzleSource::B),
        (tex::TEXHEAD_BL_X_SOURCE_IN_A, SwizzleSource::A),
        (tex::TEXHEAD_BL_X_SOURCE_IN_ONE_INT, SwizzleSource::One),
        (tex::TEXHEAD_BL_X_SOURCE_IN_ONE_FLOAT, SwizzleSource::One),
    ];
    for (value, source) in cases {
        // Each component on its own, the others distinct, so a swapped read fails.
        let words = bl_header(0).set(tex::TEXHEAD_BL_Z_SOURCE, value).words();
        let swizzle = TextureHeader::decode(&words).unwrap().view.swizzle;
        assert_eq!(
            swizzle,
            Swizzle {
                b: source,
                ..Swizzle::IDENTITY
            },
            "{value}"
        );
    }
    // The header defines no source 1.
    for (field, component) in [
        (tex::TEXHEAD_BL_X_SOURCE, 'r'),
        (tex::TEXHEAD_BL_Y_SOURCE, 'g'),
        (tex::TEXHEAD_BL_Z_SOURCE, 'b'),
        (tex::TEXHEAD_BL_W_SOURCE, 'a'),
    ] {
        let words = bl_header(0).set(field, 1).words();
        assert_eq!(
            TextureHeader::decode(&words),
            Err(HeaderError::InvalidSwizzle {
                component,
                value: 1
            })
        );
    }
}

#[test]
fn the_swizzle_reads_each_component_from_its_own_field() {
    let words = bl_header(0)
        .set(tex::TEXHEAD_BL_X_SOURCE, tex::TEXHEAD_BL_X_SOURCE_IN_A)
        .set(tex::TEXHEAD_BL_Y_SOURCE, tex::TEXHEAD_BL_Y_SOURCE_IN_ZERO)
        .set(tex::TEXHEAD_BL_Z_SOURCE, tex::TEXHEAD_BL_Z_SOURCE_IN_G)
        .set(tex::TEXHEAD_BL_W_SOURCE, tex::TEXHEAD_BL_W_SOURCE_IN_R)
        .words();
    assert_eq!(
        TextureHeader::decode(&words).unwrap().view.swizzle,
        Swizzle {
            r: SwizzleSource::A,
            g: SwizzleSource::Zero,
            b: SwizzleSource::G,
            a: SwizzleSource::R,
        }
    );
}

#[test]
fn depth_formats_take_the_guessed_fermi_zs_gob() {
    let words = bl_header(0)
        .set(
            tex::TEXHEAD_BL_COMPONENTS,
            tex::TEXHEAD_BL_COMPONENTS_SIZES_ZF32,
        )
        .types(tex::TEXHEAD_BL_R_DATA_TYPE.low(), FLOAT)
        .words();
    let image = TextureHeader::decode(&words).unwrap().image;
    assert_eq!(image.format, PixelFormat::D32Float);
    assert!(matches!(
        image.layout,
        MemoryLayout::BlockLinear {
            gob: GobType::FermiZs,
            ..
        }
    ));
    // Every aspect: colour is the Switch 2 colour GOB (row 69), the rest the guess.
    assert_eq!(gob_for(PixelFormat::R8Unorm), GobType::TuringColor2d);
    assert_eq!(gob_for(PixelFormat::D16Unorm), GobType::FermiZs);
    assert_eq!(gob_for(PixelFormat::S8Uint), GobType::FermiZs);
    assert_eq!(gob_for(PixelFormat::D24UnormS8Uint), GobType::FermiZs);
}

// --- Texture types, sample modes, blocks and levels ---

#[test]
fn every_texture_type_gives_its_shape() {
    use tex::*;
    // (type, depth field + 1) -> (dim, layers, depth, view)
    type Shape = (ImageDim, u32, u32, ViewDimension);
    let cases: [(u32, u32, Option<Shape>); 18] = [
        (
            TEXHEAD_BL_TEXTURE_TYPE_ONE_D,
            1,
            Some((ImageDim::D1, 1, 1, ViewDimension::D1)),
        ),
        (
            TEXHEAD_BL_TEXTURE_TYPE_ONE_D,
            3,
            Some((ImageDim::D1, 3, 1, ViewDimension::D1Array)),
        ),
        (
            TEXHEAD_BL_TEXTURE_TYPE_TWO_D,
            1,
            Some((ImageDim::D2, 1, 1, ViewDimension::D2)),
        ),
        (
            TEXHEAD_BL_TEXTURE_TYPE_TWO_D,
            3,
            Some((ImageDim::D2, 3, 1, ViewDimension::D2Array)),
        ),
        (
            TEXHEAD_BL_TEXTURE_TYPE_THREE_D,
            5,
            Some((ImageDim::D3, 1, 5, ViewDimension::D3)),
        ),
        (
            TEXHEAD_BL_TEXTURE_TYPE_CUBEMAP,
            1,
            Some((ImageDim::D2, CUBE_FACES, 1, ViewDimension::Cube)),
        ),
        (
            TEXHEAD_BL_TEXTURE_TYPE_CUBEMAP,
            2,
            Some((ImageDim::D2, 2 * CUBE_FACES, 1, ViewDimension::CubeArray)),
        ),
        (
            TEXHEAD_BL_TEXTURE_TYPE_ONE_D_ARRAY,
            1,
            Some((ImageDim::D1, 1, 1, ViewDimension::D1Array)),
        ),
        (
            TEXHEAD_BL_TEXTURE_TYPE_TWO_D_ARRAY,
            1,
            Some((ImageDim::D2, 1, 1, ViewDimension::D2Array)),
        ),
        (
            TEXHEAD_BL_TEXTURE_TYPE_TWO_D_ARRAY,
            7,
            Some((ImageDim::D2, 7, 1, ViewDimension::D2Array)),
        ),
        (TEXHEAD_BL_TEXTURE_TYPE_ONE_D_BUFFER, 1, None),
        (
            TEXHEAD_BL_TEXTURE_TYPE_TWO_D_NO_MIPMAP,
            1,
            Some((ImageDim::D2, 1, 1, ViewDimension::D2)),
        ),
        // Not promoted to an array (Eden's `ImageInfo`): one layer, whatever the depth.
        (
            TEXHEAD_BL_TEXTURE_TYPE_TWO_D_NO_MIPMAP,
            2,
            Some((ImageDim::D2, 1, 1, ViewDimension::D2)),
        ),
        (
            TEXHEAD_BL_TEXTURE_TYPE_CUBEMAP_ARRAY,
            1,
            Some((ImageDim::D2, CUBE_FACES, 1, ViewDimension::CubeArray)),
        ),
        (
            TEXHEAD_BL_TEXTURE_TYPE_CUBEMAP_ARRAY,
            3,
            Some((ImageDim::D2, 3 * CUBE_FACES, 1, ViewDimension::CubeArray)),
        ),
        (TEXHEAD_BL_TEXTURE_TYPE_HTEX_TWOD, 1, None),
        (TEXHEAD_BL_TEXTURE_TYPE_HTEX_THREE_D, 1, None),
        (TEXHEAD_BL_TEXTURE_TYPE_HTEX_TWOD_ARRAY, 1, None),
    ];
    for (texture_type, depth, expected) in cases {
        let words = bl_header(0)
            .set(TEXHEAD_BL_TEXTURE_TYPE, texture_type)
            .set(TEXHEAD_BL_HEIGHT_MINUS_ONE, 15)
            .set(TEXHEAD_BL_DEPTH_MINUS_ONE, depth - 1)
            .words();
        let got = TextureHeader::decode(&words)
            .map(|h| (h.image.dim, h.image.layers, h.image.depth, h.view.dimension));
        match expected {
            Some(shape) => assert_eq!(got, Ok(shape), "type {texture_type} depth {depth}"),
            None => assert_eq!(
                got,
                Err(HeaderError::UnsupportedTextureType { texture_type }),
                "type {texture_type}"
            ),
        }
    }
}

#[test]
fn a_2d_no_mipmap_header_is_one_layer_whatever_its_depth() {
    use tex::*;
    // Eden promotes 1D, 2D and cube types to arrays when the depth is above 1, but not
    // `Texture2DNoMipmap`, which asserts a depth of 1 and takes one layer (`eden/src/video_core/
    // texture_cache/image_info.cpp` and `image_view_info.cpp` @ cf654052).
    let shape = |texture_type, depth: u32| {
        TextureHeader::decode(
            &bl_header(0)
                .set(TEXHEAD_BL_TEXTURE_TYPE, texture_type)
                .set(TEXHEAD_BL_DEPTH_MINUS_ONE, depth - 1)
                .words(),
        )
        .map(|h| (h.image.dim, h.image.layers, h.image.depth, h.view.dimension))
    };
    for depth in [1, 2, 7] {
        assert_eq!(
            shape(TEXHEAD_BL_TEXTURE_TYPE_TWO_D_NO_MIPMAP, depth),
            Ok((ImageDim::D2, 1, 1, ViewDimension::D2)),
            "{depth}"
        );
    }
    // A plain 2D header of the same depth is an array.
    assert_eq!(
        shape(TEXHEAD_BL_TEXTURE_TYPE_TWO_D, 7),
        Ok((ImageDim::D2, 7, 1, ViewDimension::D2Array))
    );
}

#[test]
fn a_1d_image_is_one_high_and_undefined_types_are_refused() {
    use tex::*;
    // A 1D image's height is 1 whatever the header says.
    let words = bl_header(0)
        .set(TEXHEAD_BL_TEXTURE_TYPE, TEXHEAD_BL_TEXTURE_TYPE_ONE_D)
        .set(TEXHEAD_BL_HEIGHT_MINUS_ONE, 15)
        .words();
    assert_eq!(TextureHeader::decode(&words).unwrap().image.height, 1);
    // The undefined values.
    for texture_type in [9, 0xC, 0xD, 0xF] {
        let words = bl_header(0)
            .set(TEXHEAD_BL_TEXTURE_TYPE, texture_type)
            .words();
        assert_eq!(
            TextureHeader::decode(&words),
            Err(HeaderError::UnsupportedTextureType { texture_type })
        );
    }
}

#[test]
fn every_sample_mode_gives_its_grid() {
    use tex::*;
    let cases = [
        (
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_1X1,
            Some(SampleLayout::S1x1),
        ),
        (
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X1,
            Some(SampleLayout::S2x1),
        ),
        (
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X2,
            Some(SampleLayout::S2x2),
        ),
        (
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2,
            Some(SampleLayout::S4x2),
        ),
        (
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2_D3D,
            Some(SampleLayout::S4x2D3d),
        ),
        (
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X1_D3D,
            Some(SampleLayout::S2x1D3d),
        ),
        (
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X4,
            Some(SampleLayout::S4x4),
        ),
        (7, None),
        (
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X2_VC_4,
            Some(SampleLayout::S2x2),
        ),
        (
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X2_VC_12,
            Some(SampleLayout::S2x2),
        ),
        (
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2_VC_8,
            Some(SampleLayout::S4x2),
        ),
        (
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2_VC_24,
            Some(SampleLayout::S4x2),
        ),
        (12, None),
        (13, None),
        (14, None),
        (15, None),
    ];
    for (mode, expected) in cases {
        let words = bl_header(0)
            .set(TEXHEAD_BL_MULTI_SAMPLE_COUNT, mode)
            .words();
        let got = TextureHeader::decode(&words).map(|h| h.image.samples);
        assert_eq!(
            got,
            expected.ok_or(HeaderError::InvalidSampleMode { mode }),
            "mode {mode}"
        );
    }
}

#[test]
fn the_mode_literals_are_the_headers() {
    // Nvidia's values written out (`clc797tex.h`), against the constants the decode matches.
    use tex::*;
    assert_eq!(
        [
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_1X1,
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X1,
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X2,
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2,
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2_D3D,
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X1_D3D,
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X4,
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X2_VC_4,
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X2_VC_12,
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2_VC_8,
            TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2_VC_24,
        ],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    );
    assert_eq!(
        [
            TEXHEAD_BL_HEADER_VERSION_SELECT_ONE_D_BUFFER,
            TEXHEAD_BL_HEADER_VERSION_SELECT_PITCH_COLOR_KEY,
            TEXHEAD_BL_HEADER_VERSION_SELECT_PITCH,
            TEXHEAD_BL_HEADER_VERSION_SELECT_BLOCKLINEAR,
            TEXHEAD_BL_HEADER_VERSION_SELECT_BLOCKLINEAR_COLOR_KEY,
        ],
        [0, 1, 2, 3, 4]
    );
    assert_eq!(
        [
            TEXHEAD_BL_X_SOURCE_IN_ZERO,
            TEXHEAD_BL_X_SOURCE_IN_R,
            TEXHEAD_BL_X_SOURCE_IN_G,
            TEXHEAD_BL_X_SOURCE_IN_B,
            TEXHEAD_BL_X_SOURCE_IN_A,
            TEXHEAD_BL_X_SOURCE_IN_ONE_INT,
            TEXHEAD_BL_X_SOURCE_IN_ONE_FLOAT,
        ],
        [0, 2, 3, 4, 5, 6, 7]
    );
    // Eden's `TextureType` (`textures/texture.h`): 1D 0, 2D 1, 3D 2, cube 3, 1D array 4,
    // 2D array 5, 1D buffer 6, 2D no mipmap 7, cube array 8.
    assert_eq!(
        [
            TEXHEAD_BL_TEXTURE_TYPE_ONE_D,
            TEXHEAD_BL_TEXTURE_TYPE_TWO_D,
            TEXHEAD_BL_TEXTURE_TYPE_THREE_D,
            TEXHEAD_BL_TEXTURE_TYPE_CUBEMAP,
            TEXHEAD_BL_TEXTURE_TYPE_ONE_D_ARRAY,
            TEXHEAD_BL_TEXTURE_TYPE_TWO_D_ARRAY,
            TEXHEAD_BL_TEXTURE_TYPE_ONE_D_BUFFER,
            TEXHEAD_BL_TEXTURE_TYPE_TWO_D_NO_MIPMAP,
            TEXHEAD_BL_TEXTURE_TYPE_CUBEMAP_ARRAY,
        ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(CUBE_FACES, 6);
    assert_eq!(HEADER_WORDS, 8);
}

#[test]
fn every_header_version_selects_its_form_or_is_refused() {
    use tex::*;
    for version in 0..8 {
        let words = bl_header(0).set(TEXHEAD_BL_HEADER_VERSION, version).words();
        let got = TextureHeader::decode(&words);
        match version {
            TEXHEAD_BL_HEADER_VERSION_SELECT_BLOCKLINEAR
            | TEXHEAD_BL_HEADER_VERSION_SELECT_BLOCKLINEAR_COLOR_KEY => assert!(
                matches!(got, Ok(h) if matches!(h.image.layout, MemoryLayout::BlockLinear { .. })),
                "{version}"
            ),
            TEXHEAD_BL_HEADER_VERSION_SELECT_PITCH
            | TEXHEAD_BL_HEADER_VERSION_SELECT_PITCH_COLOR_KEY => assert!(
                matches!(got, Ok(h) if matches!(h.image.layout, MemoryLayout::Pitch { .. })),
                "{version}"
            ),
            _ => assert_eq!(got, Err(HeaderError::UnsupportedVersion { version })),
        }
    }
}

#[test]
fn the_block_size_is_checked_at_each_edge() {
    use tex::*;
    let block = |w, h, d| {
        TextureHeader::decode(
            &bl_header(0)
                .set(TEXHEAD_BL_GOBS_PER_BLOCK_WIDTH, w)
                .set(TEXHEAD_BL_GOBS_PER_BLOCK_HEIGHT, h)
                .set(TEXHEAD_BL_GOBS_PER_BLOCK_DEPTH, d)
                .words(),
        )
        .map(|h| h.image.layout)
    };
    let max_h = TEXHEAD_BL_GOBS_PER_BLOCK_HEIGHT_THIRTYTWO_GOBS;
    let max_d = TEXHEAD_BL_GOBS_PER_BLOCK_DEPTH_THIRTYTWO_GOBS;
    assert_eq!(
        block(0, max_h, max_d),
        Ok(MemoryLayout::BlockLinear {
            gob: GobType::TuringColor2d,
            block_height_log2: 5,
            block_depth_log2: 5,
        })
    );
    // Height and depth distinct, so a swapped read fails.
    assert_eq!(
        block(0, 2, 1),
        Ok(MemoryLayout::BlockLinear {
            gob: GobType::TuringColor2d,
            block_height_log2: 2,
            block_depth_log2: 1,
        })
    );
    for (w, h, d) in [(1, 0, 0), (0, max_h + 1, 0), (0, 0, max_d + 1), (7, 7, 7)] {
        assert_eq!(
            block(w, h, d),
            Err(HeaderError::InvalidBlock {
                width: w,
                height: h,
                depth: d
            }),
            "{w} {h} {d}"
        );
    }
}

#[test]
fn the_view_level_range_is_checked_at_each_edge() {
    use tex::*;
    let levels = |max_mip, min, max| {
        TextureHeader::decode(
            &bl_header(0)
                .set(TEXHEAD_BL_MAX_MIP_LEVEL, max_mip)
                .set(TEXHEAD_BL_RES_VIEW_MIN_MIP_LEVEL, min)
                .set(TEXHEAD_BL_RES_VIEW_MAX_MIP_LEVEL, max)
                .words(),
        )
        .map(|h| (h.image.levels, h.view.base_mip, h.view.mip_count))
    };
    assert_eq!(levels(3, 0, 3), Ok((4, 0, 4)));
    assert_eq!(levels(3, 2, 2), Ok((4, 2, 1)));
    assert_eq!(levels(3, 1, 2), Ok((4, 1, 2)));
    assert_eq!(levels(15, 15, 15), Ok((16, 15, 1)));
    assert_eq!(
        levels(3, 0, 4),
        Err(HeaderError::InvalidLevelRange {
            min: 0,
            max: 4,
            levels: 4
        })
    );
    assert_eq!(
        levels(3, 3, 2),
        Err(HeaderError::InvalidLevelRange {
            min: 3,
            max: 2,
            levels: 4
        })
    );
}

#[test]
fn srgb_selects_the_srgb_format() {
    let words = bl_header(0)
        .set(tex::TEXHEAD_BL_S_R_G_B_CONVERSION, 1)
        .words();
    assert_eq!(
        TextureHeader::decode(&words).unwrap().image.format,
        PixelFormat::A8B8G8R8Srgb
    );
}

// --- Errors ---

#[test]
fn each_refusal_reports_its_own_error() {
    use tex::*;
    let ok = bl_header(0);
    assert!(TextureHeader::decode(&ok.words()).is_ok());
    assert_eq!(
        TextureHeader::decode(
            &ok.set(
                TEXHEAD_BL_HEADER_VERSION,
                TEXHEAD_BL_HEADER_VERSION_SELECT_ONE_D_BUFFER
            )
            .words()
        ),
        Err(HeaderError::UnsupportedVersion { version: 0 })
    );
    assert_eq!(
        TextureHeader::decode(
            &ok.types(TEXHEAD_BL_R_DATA_TYPE.low(), SINT)
                .set(TEXHEAD_BL_S_R_G_B_CONVERSION, 1)
                .words()
        ),
        Err(HeaderError::UnknownFormat {
            components: TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8,
            types: [SINT; 4],
            srgb: true
        })
    );
    assert_eq!(
        TextureHeader::decode(&ok.set(TEXHEAD_BL_W_SOURCE, 1).words()),
        Err(HeaderError::InvalidSwizzle {
            component: 'a',
            value: 1
        })
    );
    assert_eq!(
        TextureHeader::decode(
            &ok.set(
                TEXHEAD_BL_TEXTURE_TYPE,
                TEXHEAD_BL_TEXTURE_TYPE_ONE_D_BUFFER
            )
            .words()
        ),
        Err(HeaderError::UnsupportedTextureType { texture_type: 6 })
    );
}

#[test]
fn an_unknown_format_is_an_error_not_eden_s_fallback() {
    // FORCE_FP16 types are in no row; nor is an sRGB R8.
    assert_eq!(
        pixel_format(tex::TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8, [5; 4], false),
        None
    );
    assert_eq!(
        pixel_format(tex::TEXHEAD_BL_COMPONENTS_SIZES_R8, [UNORM; 4], true),
        None
    );
    assert_eq!(
        pixel_format(tex::TEXHEAD_BL_COMPONENTS_SIZES_INVALID, [UNORM; 4], false),
        None
    );
    let words = bl_header(0)
        .set(
            tex::TEXHEAD_BL_COMPONENTS,
            tex::TEXHEAD_BL_COMPONENTS_SIZES_R1,
        )
        .words();
    assert_eq!(
        TextureHeader::decode(&words),
        Err(HeaderError::UnknownFormat {
            components: tex::TEXHEAD_BL_COMPONENTS_SIZES_R1,
            types: [UNORM; 4],
            srgb: false
        })
    );
}

#[test]
fn checks_run_in_their_documented_order() {
    use tex::*;
    // Every fault at once, then each removed in the documented order: version, format,
    // swizzle, texture type, block, sample mode, level range.
    let all = bl_header(0)
        .set(TEXHEAD_BL_COMPONENTS, TEXHEAD_BL_COMPONENTS_SIZES_R1)
        .set(TEXHEAD_BL_X_SOURCE, 1)
        .set(
            TEXHEAD_BL_TEXTURE_TYPE,
            TEXHEAD_BL_TEXTURE_TYPE_ONE_D_BUFFER,
        )
        .set(TEXHEAD_BL_GOBS_PER_BLOCK_WIDTH, 1)
        .set(TEXHEAD_BL_MULTI_SAMPLE_COUNT, 7)
        .set(TEXHEAD_BL_RES_VIEW_MIN_MIP_LEVEL, 1);
    let decode = |b: Build| TextureHeader::decode(&b.words());
    let b = all.set(TEXHEAD_BL_HEADER_VERSION, 5);
    assert_eq!(
        decode(b),
        Err(HeaderError::UnsupportedVersion { version: 5 })
    );
    let b = all;
    assert!(matches!(decode(b), Err(HeaderError::UnknownFormat { .. })));
    let b = b.set(TEXHEAD_BL_COMPONENTS, TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8);
    assert!(matches!(decode(b), Err(HeaderError::InvalidSwizzle { .. })));
    let b = b.set(TEXHEAD_BL_X_SOURCE, TEXHEAD_BL_X_SOURCE_IN_R);
    assert!(matches!(
        decode(b),
        Err(HeaderError::UnsupportedTextureType { .. })
    ));
    let b = b.set(TEXHEAD_BL_TEXTURE_TYPE, TEXHEAD_BL_TEXTURE_TYPE_TWO_D);
    assert!(matches!(decode(b), Err(HeaderError::InvalidBlock { .. })));
    let b = b.set(TEXHEAD_BL_GOBS_PER_BLOCK_WIDTH, 0);
    assert!(matches!(
        decode(b),
        Err(HeaderError::InvalidSampleMode { .. })
    ));
    let b = b.set(TEXHEAD_BL_MULTI_SAMPLE_COUNT, 0);
    assert!(matches!(
        decode(b),
        Err(HeaderError::InvalidLevelRange { .. })
    ));
    let b = b.set(TEXHEAD_BL_RES_VIEW_MIN_MIP_LEVEL, 0);
    assert!(decode(b).is_ok());
}

#[test]
fn error_messages_are_plain_words() {
    assert_eq!(
        HeaderError::UnsupportedVersion { version: 0 }.to_string(),
        "texture header version 0 is not supported"
    );
    assert_eq!(
        HeaderError::InvalidSwizzle {
            component: 'g',
            value: 1
        }
        .to_string(),
        "texture swizzle source 1 for component g is not defined"
    );
    assert_eq!(
        HeaderError::InvalidLevelRange {
            min: 2,
            max: 1,
            levels: 4
        }
        .to_string(),
        "texture view levels 2 to 1 do not fit an image of 4 levels"
    );
    assert_eq!(
        HeaderError::InvalidBlock {
            width: 1,
            height: 0,
            depth: 0
        }
        .to_string(),
        "texture block of 1, 0 and 0 (log2 GOBs) is not one the header defines"
    );
    assert_eq!(
        HeaderError::UnknownFormat {
            components: 0x1d,
            types: [1, 2, 3, 4],
            srgb: true
        }
        .to_string(),
        "texture format 0x1d with data types 1, 2, 3 and 4 (red to alpha) and sRGB conversion on \
         is not known"
    );
    assert_eq!(
        HeaderError::UnknownFormat {
            components: 0,
            types: [0; 4],
            srgb: false
        }
        .to_string(),
        "texture format 0x0 with data types 0, 0, 0 and 0 (red to alpha) and sRGB conversion off \
         is not known"
    );
    assert_eq!(
        HeaderError::UnsupportedTextureType { texture_type: 6 }.to_string(),
        "texture type 6 is not supported"
    );
    assert_eq!(
        HeaderError::InvalidSampleMode { mode: 7 }.to_string(),
        "texture multisample mode 7 is not defined"
    );
}

// --- The format table ---

#[test]
fn eden_format_values_match_the_header() {
    use tex::*;
    // Eden's `TextureFormat` and `ComponentType` literals (`eden/src/video_core/textures/
    // texture.h` @ cf654052), each against the header constant of the same meaning: two
    // independent sources for every value the table names.
    let formats = [
        (TEXHEAD_BL_COMPONENTS_SIZES_R32_G32_B32_A32, 0x01),
        (TEXHEAD_BL_COMPONENTS_SIZES_R32_G32_B32, 0x02),
        (TEXHEAD_BL_COMPONENTS_SIZES_R16_G16_B16_A16, 0x03),
        (TEXHEAD_BL_COMPONENTS_SIZES_R32_G32, 0x04),
        (TEXHEAD_BL_COMPONENTS_SIZES_R32_B24G8, 0x05),
        (TEXHEAD_BL_COMPONENTS_SIZES_ETC2_RGB, 0x06),
        (TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8, 0x08),
        (TEXHEAD_BL_COMPONENTS_SIZES_A2B10G10R10, 0x09),
        (TEXHEAD_BL_COMPONENTS_SIZES_ETC2_RGB_PTA, 0x0a),
        (TEXHEAD_BL_COMPONENTS_SIZES_ETC2_RGBA, 0x0b),
        (TEXHEAD_BL_COMPONENTS_SIZES_R16_G16, 0x0c),
        (TEXHEAD_BL_COMPONENTS_SIZES_G24R8, 0x0e),
        (TEXHEAD_BL_COMPONENTS_SIZES_R32, 0x0f),
        (TEXHEAD_BL_COMPONENTS_SIZES_BC6H_SF16, 0x10),
        (TEXHEAD_BL_COMPONENTS_SIZES_BC6H_UF16, 0x11),
        (TEXHEAD_BL_COMPONENTS_SIZES_A4B4G4R4, 0x12),
        (TEXHEAD_BL_COMPONENTS_SIZES_A5B5G5R1, 0x13),
        (TEXHEAD_BL_COMPONENTS_SIZES_A1B5G5R5, 0x14),
        (TEXHEAD_BL_COMPONENTS_SIZES_B5G6R5, 0x15),
        (TEXHEAD_BL_COMPONENTS_SIZES_BC7U, 0x17),
        (TEXHEAD_BL_COMPONENTS_SIZES_G8R8, 0x18),
        (TEXHEAD_BL_COMPONENTS_SIZES_EAC, 0x19),
        (TEXHEAD_BL_COMPONENTS_SIZES_EACX2, 0x1a),
        (TEXHEAD_BL_COMPONENTS_SIZES_R16, 0x1b),
        (TEXHEAD_BL_COMPONENTS_SIZES_R8, 0x1d),
        (TEXHEAD_BL_COMPONENTS_SIZES_G4R4, 0x1e),
        (TEXHEAD_BL_COMPONENTS_SIZES_E5B9G9R9_SHAREDEXP, 0x20),
        (TEXHEAD_BL_COMPONENTS_SIZES_BF10GF11RF11, 0x21),
        (TEXHEAD_BL_COMPONENTS_SIZES_DXT1, 0x24),
        (TEXHEAD_BL_COMPONENTS_SIZES_DXT23, 0x25),
        (TEXHEAD_BL_COMPONENTS_SIZES_DXT45, 0x26),
        (TEXHEAD_BL_COMPONENTS_SIZES_DXN1, 0x27),
        (TEXHEAD_BL_COMPONENTS_SIZES_DXN2, 0x28),
        (TEXHEAD_BL_COMPONENTS_SIZES_Z24S8, 0x29),
        (TEXHEAD_BL_COMPONENTS_SIZES_X8Z24, 0x2a),
        (TEXHEAD_BL_COMPONENTS_SIZES_S8Z24, 0x2b),
        (TEXHEAD_BL_COMPONENTS_SIZES_ZF32, 0x2f),
        (TEXHEAD_BL_COMPONENTS_SIZES_ZF32_X24S8, 0x30),
        (TEXHEAD_BL_COMPONENTS_SIZES_Z16, 0x3a),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_4X4, 0x40),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_5X5, 0x41),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_6X6, 0x42),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_8X8, 0x44),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_10X10, 0x45),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_12X12, 0x46),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_5X4, 0x50),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_6X5, 0x51),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_8X6, 0x52),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_10X8, 0x53),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_12X10, 0x54),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_8X5, 0x55),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_10X5, 0x56),
        (TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_10X6, 0x57),
    ];
    for (header, eden) in formats {
        assert_eq!(header, eden, "{eden:#x}");
    }
    // Every format value in the table is one of those.
    for &(components, ..) in FORMAT_TABLE {
        assert!(
            formats.iter().any(|&(h, _)| h == components),
            "{components:#x}"
        );
    }
    assert_eq!([SNORM, UNORM, SINT, UINT, FLOAT], [1, 2, 3, 4, 7]);
}

#[test]
fn the_table_has_every_eden_row_and_each_row_is_found() {
    // Eden's 111 `case`s and the one Eden-NX2 adds.
    assert_eq!(FORMAT_TABLE.len(), 112);
    for &(components, types, srgb, format) in FORMAT_TABLE {
        assert_eq!(pixel_format(components, types, srgb), Some(format));
    }
}

#[test]
fn hand_written_rows_from_eden_s_table() {
    // Literal values (Eden's `Hash(...)` cases), not read through the table's own constants.
    let cases = [
        (0x08, [2, 2, 2, 2], false, PixelFormat::A8B8G8R8Unorm),
        (0x08, [2, 2, 2, 2], true, PixelFormat::A8B8G8R8Srgb),
        (0x08, [3, 3, 3, 3], false, PixelFormat::A8B8G8R8Sint),
        (0x08, [4, 4, 4, 4], false, PixelFormat::A8B8G8R8Uint),
        (0x18, [1, 1, 1, 1], false, PixelFormat::R8G8Snorm),
        (0x03, [7, 7, 7, 7], false, PixelFormat::R16G16B16A16Float),
        (0x2f, [7, 4, 4, 4], false, PixelFormat::D32Float),
        (0x2f, [7, 7, 7, 7], false, PixelFormat::D32Float),
        (0x29, [4, 2, 2, 2], false, PixelFormat::S8UintD24Unorm),
        (0x29, [4, 2, 4, 4], false, PixelFormat::S8UintD24Unorm),
        (0x0e, [4, 2, 2, 2], false, PixelFormat::S8UintD24Unorm),
        (0x2b, [2, 4, 4, 4], false, PixelFormat::D24UnormS8Uint),
        // Eden-NX2's added row (`eden-nx2/src/video_core/texture_cache/format_lookup_table.cpp`
        // @ 50fa86bb, line 160).
        (0x2b, [2, 4, 2, 2], false, PixelFormat::D24UnormS8Uint),
        (0x30, [7, 4, 2, 2], false, PixelFormat::D32FloatS8Uint),
        (0x05, [7, 4, 2, 2], false, PixelFormat::D32FloatS8Uint),
        (0x24, [2, 2, 2, 2], true, PixelFormat::Bc1RgbaSrgb),
        (0x27, [1, 1, 1, 1], false, PixelFormat::Bc4Snorm),
        (0x10, [7, 7, 7, 7], false, PixelFormat::Bc6hSfloat),
        (0x11, [7, 7, 7, 7], false, PixelFormat::Bc6hUfloat),
        (0x1a, [1, 1, 1, 1], false, PixelFormat::EacR11G11Snorm),
        (0x44, [2, 2, 2, 2], true, PixelFormat::Astc2d8x8Srgb),
        (0x51, [2, 2, 2, 2], false, PixelFormat::Astc2d6x5Unorm),
        (0x21, [7, 7, 7, 7], false, PixelFormat::B10G11R11Float),
        (0x20, [7, 7, 7, 7], false, PixelFormat::E5B9G9R9Float),
    ];
    for (components, types, srgb, format) in cases {
        assert_eq!(
            pixel_format(components, types, srgb),
            Some(format),
            "{components:#x} {types:?} {srgb}"
        );
    }
    // A depth-stencil row is not reached by another type order.
    assert_eq!(pixel_format(0x29, [2, 4, 2, 2], false), None);
}

// --- Generated round trips ---

proptest! {
    #![proptest_config(Config {
        cases: 256,
        rng_seed: RngSeed::Fixed(0x7e17_0328),
        ..Config::default()
    })]

    /// Every field built at a distinct value decodes to itself, so two swapped reads fail.
    #[test]
    fn a_built_block_linear_header_decodes_to_what_was_built(
        address in 0u64..(1 << 49),
        width in 1u32..=(1 << 17),
        height in 1u32..=(1 << 17),
        layers in 1u32..=(1 << 15),
        max_mip in 0u32..16,
        block_h in 0u32..=5,
        block_d in 0u32..=5,
        row in 0usize..FORMAT_TABLE.len(),
        swizzle in proptest::array::uniform4(prop::sample::select(vec![0u32, 2, 3, 4, 5, 6, 7])),
        mins in (0u32..16, 0u32..16),
        mode in prop::sample::select(vec![0u32, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]),
    ) {
        use tex::*;
        let (components, types, srgb, format) = FORMAT_TABLE[row];
        let address = address & !0x1FF;
        let (a, b) = mins;
        let (min, max) = (a.min(b).min(max_mip), a.max(b).min(max_mip));
        let mut build = Build::new()
            .set(TEXHEAD_BL_HEADER_VERSION, TEXHEAD_BL_HEADER_VERSION_SELECT_BLOCKLINEAR)
            .set(TEXHEAD_BL_COMPONENTS, components)
            .set(TEXHEAD_BL_S_R_G_B_CONVERSION, u32::from(srgb))
            .set(TEXHEAD_BL_ADDRESS_BITS31TO9, u32::try_from((address >> 9) & 0x7F_FFFF).unwrap())
            .set(TEXHEAD_BL_ADDRESS_BITS48TO32, u32::try_from(address >> 32).unwrap())
            .set(TEXHEAD_BL_WIDTH_MINUS_ONE, width - 1)
            .set(TEXHEAD_BL_HEIGHT_MINUS_ONE, (height - 1) & 0xFFFF)
            .set(TEXHEAD_BL_HEIGHT_MINUS_ONE_BIT16, (height - 1) >> 16)
            .set(TEXHEAD_BL_DEPTH_MINUS_ONE, (layers - 1) & 0x3FFF)
            .set(TEXHEAD_BL_DEPTH_MINUS_ONE_BIT14, (layers - 1) >> 14)
            .set(TEXHEAD_BL_TEXTURE_TYPE, TEXHEAD_BL_TEXTURE_TYPE_TWO_D_ARRAY)
            .set(TEXHEAD_BL_MAX_MIP_LEVEL, max_mip)
            .set(TEXHEAD_BL_GOBS_PER_BLOCK_HEIGHT, block_h)
            .set(TEXHEAD_BL_GOBS_PER_BLOCK_DEPTH, block_d)
            .set(TEXHEAD_BL_RES_VIEW_MIN_MIP_LEVEL, min)
            .set(TEXHEAD_BL_RES_VIEW_MAX_MIP_LEVEL, max)
            .set(TEXHEAD_BL_MULTI_SAMPLE_COUNT, mode);
        for (field, value) in [
            TEXHEAD_BL_R_DATA_TYPE,
            TEXHEAD_BL_G_DATA_TYPE,
            TEXHEAD_BL_B_DATA_TYPE,
            TEXHEAD_BL_A_DATA_TYPE,
        ].into_iter().zip(types) {
            build = build.set(field, value);
        }
        for (field, value) in [
            TEXHEAD_BL_X_SOURCE,
            TEXHEAD_BL_Y_SOURCE,
            TEXHEAD_BL_Z_SOURCE,
            TEXHEAD_BL_W_SOURCE,
        ].into_iter().zip(swizzle) {
            build = build.set(field, value);
        }
        let header = TextureHeader::decode(&build.words()).unwrap();
        prop_assert_eq!(header.image.address, address);
        prop_assert_eq!(header.image.format, format);
        prop_assert_eq!((header.image.width, header.image.height), (width, height));
        prop_assert_eq!((header.image.layers, header.image.depth), (layers, 1));
        prop_assert_eq!(header.image.levels, max_mip + 1);
        prop_assert_eq!((header.view.base_mip, header.view.mip_count), (min, max - min + 1));
        prop_assert_eq!(
            header.image.layout,
            MemoryLayout::BlockLinear {
                gob: gob_for(format),
                block_height_log2: u8::try_from(block_h).unwrap(),
                block_depth_log2: u8::try_from(block_d).unwrap(),
            }
        );
        let source = |v: u32| match v {
            0 => SwizzleSource::Zero,
            2 => SwizzleSource::R,
            3 => SwizzleSource::G,
            4 => SwizzleSource::B,
            5 => SwizzleSource::A,
            _ => SwizzleSource::One,
        };
        prop_assert_eq!(
            header.view.swizzle,
            Swizzle {
                r: source(swizzle[0]),
                g: source(swizzle[1]),
                b: source(swizzle[2]),
                a: source(swizzle[3]),
            }
        );
        let grid = match mode {
            0 => SampleLayout::S1x1,
            1 => SampleLayout::S2x1,
            2 | 8 | 9 => SampleLayout::S2x2,
            3 | 10 | 11 => SampleLayout::S4x2,
            4 => SampleLayout::S4x2D3d,
            5 => SampleLayout::S2x1D3d,
            _ => SampleLayout::S4x4,
        };
        prop_assert_eq!(header.image.samples, grid);
    }

    /// A pitch header's address, pitch, width and height decode to what was built.
    #[test]
    fn a_built_pitch_header_decodes_to_what_was_built(
        address in 0u64..(1 << 49),
        pitch in 0u32..(1 << 17),
        width in 1u32..=(1 << 17),
        height in 1u32..=(1 << 17),
    ) {
        use tex::*;
        let address = address & !0x1F;
        let pitch_b = pitch << 5;
        let words = Build::new()
            .set(TEXHEAD_PITCH_HEADER_VERSION, TEXHEAD_PITCH_HEADER_VERSION_SELECT_PITCH)
            .set(TEXHEAD_PITCH_COMPONENTS, TEXHEAD_PITCH_COMPONENTS_SIZES_R8)
            .types(TEXHEAD_PITCH_R_DATA_TYPE.low(), UNORM)
            .identity(TEXHEAD_PITCH_X_SOURCE.low())
            .set(TEXHEAD_PITCH_TEXTURE_TYPE, TEXHEAD_PITCH_TEXTURE_TYPE_TWO_D)
            .set(TEXHEAD_PITCH_ADDRESS_BITS31TO5, u32::try_from((address >> 5) & 0x7FF_FFFF).unwrap())
            .set(TEXHEAD_PITCH_ADDRESS_BITS48TO32, u32::try_from(address >> 32).unwrap())
            .set(TEXHEAD_PITCH_PITCH_BITS20TO5, pitch & 0xFFFF)
            .set(TEXHEAD_PITCH_PITCH_BIT21, pitch >> 16)
            .set(TEXHEAD_PITCH_WIDTH_MINUS_ONE, width - 1)
            .set(TEXHEAD_PITCH_HEIGHT_MINUS_ONE, (height - 1) & 0xFFFF)
            .set(TEXHEAD_PITCH_HEIGHT_MINUS_ONE_BIT16, (height - 1) >> 16)
            .words();
        let header = TextureHeader::decode(&words).unwrap();
        prop_assert_eq!(header.image.address, address);
        prop_assert_eq!(header.image.layout, MemoryLayout::Pitch { pitch_b });
        prop_assert_eq!((header.image.width, header.image.height), (width, height));
    }

    /// Any eight words decode or are refused; nothing panics.
    #[test]
    fn any_header_decodes_or_is_refused(words in proptest::array::uniform8(any::<u32>())) {
        let _ = TextureHeader::decode(&words);
    }
}
