//! The format table checked against Mesa's `nil_formats.csv` and Eden's predicates.

use super::{Compression, FormatAspect, NumericType, PIXEL_FORMAT_COUNT, PixelFormat};
use std::collections::BTreeMap;

/// Mesa's NVIDIA format table (`mesa-nouveau/src/nouveau/nil/nil_formats.csv` @ 52238762, MIT).
const NIL_FORMATS: &str = include_str!("nil_formats.csv");

/// One row of Mesa's table: the columns this test reads.
#[derive(Debug)]
struct MesaRow {
    tcs: String,
    ty: String,
    support: String,
}

fn mesa_rows() -> BTreeMap<String, MesaRow> {
    let mut rows = BTreeMap::new();
    for line in NIL_FORMATS.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        assert!(cols.len() >= 7, "short row: {line}");
        let previous = rows.insert(
            cols[0].to_owned(),
            MesaRow {
                tcs: cols[2].to_owned(),
                ty: cols[3].to_owned(),
                support: cols[6].to_owned(),
            },
        );
        assert!(previous.is_none(), "duplicate Mesa row {}", cols[0]);
    }
    rows
}

/// Each Eden format's Mesa `PIPE_FORMAT` row. Packed formats are named MSB-first by Eden and
/// Vulkan (`eden/src/video_core/renderer_vulkan/maxwell_to_vk.cpp` @ cf654052 maps each to a
/// `VK_FORMAT_*_PACK*`) and LSB-first by Mesa's pipe formats, so `A8B8G8R8_UNORM` is Mesa's
/// `R8G8B8A8_UNORM`. BC, ETC2 and EAC use Mesa's DXT, RGTC, BPTC and ETC2 names.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per format; the match is exhaustive, so a new format must be mapped here"
)]
fn mesa_name(format: PixelFormat) -> Option<&'static str> {
    use PixelFormat as P;
    Some(match format {
        P::A8B8G8R8Unorm => "R8G8B8A8_UNORM",
        P::A8B8G8R8Snorm => "R8G8B8A8_SNORM",
        P::A8B8G8R8Sint => "R8G8B8A8_SINT",
        P::A8B8G8R8Uint => "R8G8B8A8_UINT",
        P::R5G6B5Unorm => "B5G6R5_UNORM",
        P::B5G6R5Unorm => "R5G6B5_UNORM",
        P::A1R5G5B5Unorm => "B5G5R5A1_UNORM",
        P::A2B10G10R10Unorm => "R10G10B10A2_UNORM",
        P::A2B10G10R10Uint => "R10G10B10A2_UINT",
        P::A2R10G10B10Unorm => "B10G10R10A2_UNORM",
        P::A1B5G5R5Unorm => "R5G5B5A1_UNORM",
        P::A5B5G5R1Unorm => "A1B5G5R5_UNORM",
        P::R8Unorm => "R8_UNORM",
        P::R8Snorm => "R8_SNORM",
        P::R8Sint => "R8_SINT",
        P::R8Uint => "R8_UINT",
        P::R16G16B16A16Float => "R16G16B16A16_FLOAT",
        P::R16G16B16A16Unorm => "R16G16B16A16_UNORM",
        P::R16G16B16A16Snorm => "R16G16B16A16_SNORM",
        P::R16G16B16A16Sint => "R16G16B16A16_SINT",
        P::R16G16B16A16Uint => "R16G16B16A16_UINT",
        P::B10G11R11Float => "R11G11B10_FLOAT",
        P::R32G32B32A32Uint => "R32G32B32A32_UINT",
        P::Bc1RgbaUnorm => "DXT1_RGBA",
        P::Bc2Unorm => "DXT3_RGBA",
        P::Bc3Unorm => "DXT5_RGBA",
        P::Bc4Unorm => "RGTC1_UNORM",
        P::Bc4Snorm => "RGTC1_SNORM",
        P::Bc5Unorm => "RGTC2_UNORM",
        P::Bc5Snorm => "RGTC2_SNORM",
        P::Bc7Unorm => "BPTC_RGBA_UNORM",
        P::Bc6hUfloat => "BPTC_RGB_UFLOAT",
        P::Bc6hSfloat => "BPTC_RGB_FLOAT",
        P::Astc2d4x4Unorm => "ASTC_4x4",
        P::B8G8R8A8Unorm => "B8G8R8A8_UNORM",
        P::R32G32B32A32Float => "R32G32B32A32_FLOAT",
        P::R32G32B32A32Sint => "R32G32B32A32_SINT",
        P::R32G32Float => "R32G32_FLOAT",
        P::R32G32Sint => "R32G32_SINT",
        P::R32Float => "R32_FLOAT",
        P::R16Float => "R16_FLOAT",
        P::R16Unorm => "R16_UNORM",
        P::R16Snorm => "R16_SNORM",
        P::R16Uint => "R16_UINT",
        P::R16Sint => "R16_SINT",
        P::R16G16Unorm => "R16G16_UNORM",
        P::R16G16Float => "R16G16_FLOAT",
        P::R16G16Uint => "R16G16_UINT",
        P::R16G16Sint => "R16G16_SINT",
        P::R16G16Snorm => "R16G16_SNORM",
        P::R32G32B32Float => "R32G32B32_FLOAT",
        P::A8B8G8R8Srgb => "R8G8B8A8_SRGB",
        P::R8G8Unorm => "R8G8_UNORM",
        P::R8G8Snorm => "R8G8_SNORM",
        P::R8G8Sint => "R8G8_SINT",
        P::R8G8Uint => "R8G8_UINT",
        P::R32G32Uint => "R32G32_UINT",
        P::R16G16B16X16Float => "R16G16B16X16_FLOAT",
        P::R32Uint => "R32_UINT",
        P::R32Sint => "R32_SINT",
        P::Astc2d8x8Unorm => "ASTC_8x8",
        P::Astc2d8x5Unorm => "ASTC_8x5",
        P::Astc2d5x4Unorm => "ASTC_5x4",
        P::B8G8R8A8Srgb => "B8G8R8A8_SRGB",
        P::Bc1RgbaSrgb => "DXT1_SRGBA",
        P::Bc2Srgb => "DXT3_SRGBA",
        P::Bc3Srgb => "DXT5_SRGBA",
        P::Bc7Srgb => "BPTC_SRGBA",
        P::A4B4G4R4Unorm => "R4G4B4A4_UNORM",
        P::G4R4Unorm => return None,
        P::Astc2d4x4Srgb => "ASTC_4x4_SRGB",
        P::Astc2d8x8Srgb => "ASTC_8x8_SRGB",
        P::Astc2d8x5Srgb => "ASTC_8x5_SRGB",
        P::Astc2d5x4Srgb => "ASTC_5x4_SRGB",
        P::Astc2d5x5Unorm => "ASTC_5x5",
        P::Astc2d5x5Srgb => "ASTC_5x5_SRGB",
        P::Astc2d10x8Unorm => "ASTC_10x8",
        P::Astc2d10x8Srgb => "ASTC_10x8_SRGB",
        P::Astc2d6x6Unorm => "ASTC_6x6",
        P::Astc2d6x6Srgb => "ASTC_6x6_SRGB",
        P::Astc2d10x6Unorm => "ASTC_10x6",
        P::Astc2d10x6Srgb => "ASTC_10x6_SRGB",
        P::Astc2d10x5Unorm => "ASTC_10x5",
        P::Astc2d10x5Srgb => "ASTC_10x5_SRGB",
        P::Astc2d10x10Unorm => "ASTC_10x10",
        P::Astc2d10x10Srgb => "ASTC_10x10_SRGB",
        P::Astc2d12x10Unorm => "ASTC_12x10",
        P::Astc2d12x10Srgb => "ASTC_12x10_SRGB",
        P::Astc2d12x12Unorm => "ASTC_12x12",
        P::Astc2d12x12Srgb => "ASTC_12x12_SRGB",
        P::Astc2d8x6Unorm => "ASTC_8x6",
        P::Astc2d8x6Srgb => "ASTC_8x6_SRGB",
        P::Astc2d6x5Unorm => "ASTC_6x5",
        P::Astc2d6x5Srgb => "ASTC_6x5_SRGB",
        P::E5B9G9R9Float => "R9G9B9E5_FLOAT",
        P::Etc2RgbUnorm => "ETC2_RGB8",
        P::Etc2RgbaUnorm => "ETC2_RGBA8",
        P::Etc2RgbPtaUnorm => "ETC2_RGB8A1",
        P::Etc2RgbSrgb => "ETC2_SRGB8",
        P::Etc2RgbaSrgb => "ETC2_SRGBA8",
        P::Etc2RgbPtaSrgb => "ETC2_SRGB8A1",
        P::EacR11Unorm => "ETC2_R11_UNORM",
        P::EacR11Snorm => "ETC2_R11_SNORM",
        P::EacR11G11Unorm => "ETC2_RG11_UNORM",
        P::EacR11G11Snorm => "ETC2_RG11_SNORM",
        P::D32Float => "Z32_FLOAT",
        P::D16Unorm => "Z16_UNORM",
        P::X8D24Unorm => "Z24X8_UNORM",
        P::S8Uint => "S8_UINT",
        P::D24UnormS8Uint => "Z24_UNORM_S8_UINT",
        P::S8UintD24Unorm => "S8_UINT_Z24_UNORM",
        P::D32FloatS8Uint => "Z32_FLOAT_S8X24_UINT",
    })
}

/// Formats with no Mesa row, and why. Exact: a format listed here that gains a mapping fails.
const NO_MESA_ROW: &[(PixelFormat, &str)] = &[(
    PixelFormat::G4R4Unorm,
    "Mesa's table has no two-channel 4-bit red-green format (its R4A4 and A4R4 rows share the \
     G4R4 component sizes but carry red and alpha)",
)];

/// Block width, block height and bits per block of Mesa's compressed component layouts (the
/// `tcs` column). These are facts of the compression formats themselves (the Khronos Data
/// Format Specification 1.3, sections "BC1" to "BC7", "ETC2" and "ASTC"): BC1, BC4, ETC2 RGB,
/// ETC2 punch-through and EAC R11 blocks are 8 bytes; the others 16 bytes; all are 4x4 except
/// ASTC, whose footprint is in its name.
fn compressed_layout(tcs: &str) -> Option<(u32, u32, u32)> {
    let four_by_four = |bits| Some((4, 4, bits));
    match tcs {
        "DXT1" | "DXN1" | "ETC2_RGB" | "ETC2_RGB_PTA" | "EAC" => four_by_four(64),
        "DXT23" | "DXT45" | "DXN2" | "BC7U" | "BC6H_SF16" | "BC6H_UF16" | "ETC2_RGBA" | "EACX2" => {
            four_by_four(128)
        }
        _ => {
            let footprint = tcs.strip_prefix("ASTC_2D_")?;
            let (w, h) = footprint.split_once('X')?;
            Some((w.parse().ok()?, h.parse().ok()?, 128))
        }
    }
}

/// Bits per texel of an uncompressed component layout: the sum of the widths in its name
/// (`A8B8G8R8` is 32, `ZF32_X24S8` is 64, `E5B9G9R9_SHAREDEXP` is 32).
fn uncompressed_bits(tcs: &str) -> u32 {
    let mut total = 0;
    let mut current = 0;
    for c in tcs.chars() {
        if let Some(d) = c.to_digit(10) {
            current = current * 10 + d;
        } else {
            total += current;
            current = 0;
        }
    }
    total + current
}

fn mesa_letter(numeric: NumericType) -> char {
    match numeric {
        NumericType::Unorm => 'N',
        NumericType::Snorm => 'S',
        NumericType::Uint => 'U',
        NumericType::Sint => 'I',
        NumericType::Float => 'F',
    }
}

#[test]
fn list_matches_eden_count() {
    // `PIXEL_FORMAT_LIST` in `eden/src/video_core/surface.h` @ cf654052 has 112 entries; the
    // array type of `ALL` ties the variant count to the constant at compile time.
    assert_eq!(PixelFormat::ALL.len(), PIXEL_FORMAT_COUNT);
    for (i, format) in PixelFormat::ALL.iter().enumerate() {
        assert_eq!(format.index(), i, "{format}");
    }
}

#[test]
fn names_are_unique() {
    let mut names: Vec<&str> = PixelFormat::ALL.iter().map(|f| f.name()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), PIXEL_FORMAT_COUNT);
}

#[test]
fn eden_aspect_ranges() {
    // Eden's `GetFormatType` splits the enum at `MaxColorFormat = D32_FLOAT`,
    // `MaxDepthFormat = S8_UINT` and `MaxStencilFormat = D24_UNORM_S8_UINT`.
    for format in PixelFormat::ALL {
        let expected = if format < PixelFormat::D32Float {
            FormatAspect::Color
        } else if format < PixelFormat::S8Uint {
            FormatAspect::Depth
        } else if format < PixelFormat::D24UnormS8Uint {
            FormatAspect::Stencil
        } else {
            FormatAspect::DepthStencil
        };
        assert_eq!(format.aspect(), expected, "{format}");
        assert_eq!(
            format.has_depth(),
            matches!(expected, FormatAspect::Depth | FormatAspect::DepthStencil)
        );
        assert_eq!(
            format.has_stencil(),
            matches!(expected, FormatAspect::Stencil | FormatAspect::DepthStencil)
        );
    }
}

fn names_where(pred: impl Fn(PixelFormat) -> bool) -> Vec<&'static str> {
    let mut v: Vec<&str> = PixelFormat::ALL
        .into_iter()
        .filter(|f| pred(*f))
        .map(PixelFormat::name)
        .collect();
    v.sort_unstable();
    v
}

fn sorted(list: &[&'static str]) -> Vec<&'static str> {
    let mut v = list.to_vec();
    v.sort_unstable();
    v
}

#[test]
fn srgb_set_is_edens() {
    // `IsPixelFormatSRGB`, `eden/src/video_core/surface.cpp` @ cf654052.
    let eden = [
        "A8B8G8R8_SRGB",
        "B8G8R8A8_SRGB",
        "BC1_RGBA_SRGB",
        "BC2_SRGB",
        "BC3_SRGB",
        "BC7_SRGB",
        "ETC2_RGB_SRGB",
        "ETC2_RGBA_SRGB",
        "ETC2_RGB_PTA_SRGB",
        "ASTC_2D_4X4_SRGB",
        "ASTC_2D_8X8_SRGB",
        "ASTC_2D_8X5_SRGB",
        "ASTC_2D_5X4_SRGB",
        "ASTC_2D_5X5_SRGB",
        "ASTC_2D_10X6_SRGB",
        "ASTC_2D_10X8_SRGB",
        "ASTC_2D_6X6_SRGB",
        "ASTC_2D_10X5_SRGB",
        "ASTC_2D_10X10_SRGB",
        "ASTC_2D_12X12_SRGB",
        "ASTC_2D_12X10_SRGB",
        "ASTC_2D_8X6_SRGB",
        "ASTC_2D_6X5_SRGB",
    ];
    assert_eq!(names_where(PixelFormat::is_srgb), sorted(&eden));
}

#[test]
fn integer_sets_are_edens() {
    // `IsPixelFormatInteger` and `IsPixelFormatSignedInteger`, `surface.cpp` @ cf654052.
    let integer = [
        "A8B8G8R8_SINT",
        "A8B8G8R8_UINT",
        "A2B10G10R10_UINT",
        "R8_SINT",
        "R8_UINT",
        "R16G16B16A16_SINT",
        "R16G16B16A16_UINT",
        "R32G32B32A32_UINT",
        "R32G32B32A32_SINT",
        "R32G32_SINT",
        "R16_UINT",
        "R16_SINT",
        "R16G16_UINT",
        "R16G16_SINT",
        "R8G8_SINT",
        "R8G8_UINT",
        "R32G32_UINT",
        "R32_UINT",
        "R32_SINT",
    ];
    let signed = [
        "A8B8G8R8_SINT",
        "R8_SINT",
        "R16G16B16A16_SINT",
        "R32G32B32A32_SINT",
        "R32G32_SINT",
        "R16_SINT",
        "R16G16_SINT",
        "R8G8_SINT",
        "R32_SINT",
    ];
    assert_eq!(names_where(PixelFormat::is_integer), sorted(&integer));
    assert_eq!(names_where(PixelFormat::is_signed_integer), sorted(&signed));
}

#[test]
fn compression_sets_are_edens() {
    // `IsPixelFormatBCn` and `IsPixelFormatETC2` (`surface.cpp` @ cf654052) list these exactly;
    // `IsPixelFormatASTC` lists every `ASTC_2D_*` entry.
    let bc = [
        "BC1_RGBA_UNORM",
        "BC2_UNORM",
        "BC3_UNORM",
        "BC4_UNORM",
        "BC4_SNORM",
        "BC5_UNORM",
        "BC5_SNORM",
        "BC1_RGBA_SRGB",
        "BC2_SRGB",
        "BC3_SRGB",
        "BC7_UNORM",
        "BC6H_UFLOAT",
        "BC6H_SFLOAT",
        "BC7_SRGB",
    ];
    let etc2 = [
        "ETC2_RGB_UNORM",
        "ETC2_RGBA_UNORM",
        "ETC2_RGB_PTA_UNORM",
        "ETC2_RGB_SRGB",
        "ETC2_RGBA_SRGB",
        "ETC2_RGB_PTA_SRGB",
        "EAC_R11_UNORM",
        "EAC_R11_SNORM",
        "EAC_R11G11_UNORM",
        "EAC_R11G11_SNORM",
    ];
    assert_eq!(
        names_where(|f| f.compression() == Compression::Bc),
        sorted(&bc)
    );
    assert_eq!(
        names_where(|f| f.compression() == Compression::Etc2),
        sorted(&etc2)
    );
    let astc = names_where(|f| f.compression() == Compression::Astc);
    assert_eq!(astc.len(), 28);
    assert!(astc.iter().all(|n| n.starts_with("ASTC_2D_")));
    assert_eq!(
        names_where(|f| f.name().starts_with("ASTC_2D_")),
        astc,
        "every ASTC name is in the ASTC family"
    );
    for format in PixelFormat::ALL {
        assert_eq!(
            format.is_compressed(),
            format.block_width() > 1 || format.block_height() > 1,
            "{format}"
        );
    }
}

#[test]
fn table_matches_mesa() {
    let rows = mesa_rows();
    for format in PixelFormat::ALL {
        let Some(name) = mesa_name(format) else {
            assert!(
                NO_MESA_ROW.iter().any(|(f, _)| *f == format),
                "{format} has no Mesa mapping and no listed reason"
            );
            continue;
        };
        assert!(
            !NO_MESA_ROW.iter().any(|(f, _)| *f == format),
            "{format} is listed as having no Mesa row but maps to {name}"
        );
        assert!(rows.contains_key(name), "{format}: no Mesa row {name}");
        let row = &rows[name];

        let (bw, bh, bits) =
            compressed_layout(&row.tcs).unwrap_or_else(|| (1, 1, uncompressed_bits(&row.tcs)));
        assert_eq!(
            (
                format.block_width(),
                format.block_height(),
                format.bits_per_block()
            ),
            (bw, bh, bits),
            "{format} against Mesa {name} ({})",
            row.tcs
        );
        assert_eq!(format.bytes_per_block() * 8, format.bits_per_block());

        assert_eq!(
            format.is_srgb(),
            name.contains("SRGB"),
            "{format} sRGB against Mesa {name}"
        );

        let is_zs = row.support.contains('Z');
        assert_eq!(
            format.aspect() != FormatAspect::Color,
            is_zs,
            "{format} aspect against Mesa {name} ({})",
            row.support
        );

        let letter = mesa_letter(format.numeric());
        if is_zs {
            // Mesa lists one letter per stored component (depth, stencil and padding); the
            // depth (or stencil-only) type must be among them, and a stencil channel is `U`.
            assert!(row.ty.contains(letter), "{format} type against Mesa {name}");
            if format.has_stencil() {
                assert!(row.ty.contains('U'), "{format} stencil against Mesa {name}");
            }
        } else {
            assert!(
                row.ty.chars().all(|c| c == letter),
                "{format} type {letter} against Mesa {name} ({})",
                row.ty
            );
        }
    }
}

#[test]
fn test_helpers_read_layouts() {
    assert_eq!(uncompressed_bits("A8B8G8R8"), 32);
    assert_eq!(uncompressed_bits("ZF32_X24S8"), 64);
    assert_eq!(uncompressed_bits("R32_G32_B32"), 96);
    assert_eq!(compressed_layout("ASTC_2D_12X10"), Some((12, 10, 128)));
    assert_eq!(compressed_layout("DXT1"), Some((4, 4, 64)));
    assert_eq!(compressed_layout("R8"), None);
}
