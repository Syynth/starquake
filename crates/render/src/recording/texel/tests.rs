//! Clear encodings, computed by hand from each format's bit layout, with distinct values per
//! channel so a swapped channel shows.

use super::{encode_clear, f16_exact};
use crate::command::ClearValue;
use crate::format::{Compression, FormatAspect, PixelFormat};

fn float(c: [f32; 4]) -> ClearValue {
    ClearValue::Float(c)
}

#[test]
fn unorm_layouts() {
    // Red 1, green 0, blue 1, alpha 0: every channel at a distinct, exact value.
    let v = float([1.0, 0.0, 1.0, 0.0]);
    for (format, bytes) in [
        // R in bits 0..8, G 8..16, B 16..24, A 24..32.
        (PixelFormat::A8B8G8R8Unorm, vec![0xff, 0x00, 0xff, 0x00]),
        // B in bits 0..8, R in 16..24.
        (PixelFormat::B8G8R8A8Unorm, vec![0xff, 0x00, 0xff, 0x00]),
        // R5G6B5 PACK16: B bits 0..5, G 5..11, R 11..16 -> 0xf81f.
        (PixelFormat::R5G6B5Unorm, vec![0x1f, 0xf8]),
        // B5G6R5 PACK16: R bits 0..5, G 5..11, B 11..16 -> 0xf81f.
        (PixelFormat::B5G6R5Unorm, vec![0x1f, 0xf8]),
        // A1R5G5B5: B 0..5, G 5..10, R 10..15, A 15 -> 0x7c1f.
        (PixelFormat::A1R5G5B5Unorm, vec![0x1f, 0x7c]),
        // A2B10G10R10: R 0..10, G 10..20, B 20..30, A 30..32 -> 0x3ff003ff.
        (PixelFormat::A2B10G10R10Unorm, vec![0xff, 0x03, 0xf0, 0x3f]),
        // A2R10G10B10: B 0..10, G 10..20, R 20..30 -> 0x3ff003ff as well.
        (PixelFormat::A2R10G10B10Unorm, vec![0xff, 0x03, 0xf0, 0x3f]),
        // A4B4G4R4: R 0..4, G 4..8, B 8..12, A 12..16 -> 0x0f0f.
        (PixelFormat::A4B4G4R4Unorm, vec![0x0f, 0x0f]),
        (PixelFormat::R8Unorm, vec![0xff]),
        (PixelFormat::R8G8Unorm, vec![0xff, 0x00]),
        (PixelFormat::R16Unorm, vec![0xff, 0xff]),
        (PixelFormat::R16G16Unorm, vec![0xff, 0xff, 0x00, 0x00]),
        (
            PixelFormat::R16G16B16A16Unorm,
            vec![0xff, 0xff, 0, 0, 0xff, 0xff, 0, 0],
        ),
    ] {
        assert_eq!(encode_clear(format, v), Some(bytes), "{format}");
    }
    // Alpha alone, to see where it lands.
    let alpha = float([0.0, 0.0, 0.0, 1.0]);
    for (format, bytes) in [
        (PixelFormat::A8B8G8R8Unorm, vec![0, 0, 0, 0xff]),
        (PixelFormat::B8G8R8A8Unorm, vec![0, 0, 0, 0xff]),
        (PixelFormat::A1R5G5B5Unorm, vec![0x00, 0x80]),
        (PixelFormat::A2B10G10R10Unorm, vec![0, 0, 0, 0xc0]),
        (PixelFormat::A4B4G4R4Unorm, vec![0x00, 0xf0]),
    ] {
        assert_eq!(encode_clear(format, alpha), Some(bytes), "{format}");
    }
    // Green alone on the 5-6-5 formats: bits 5..11 in both -> 0x07e0.
    let green = float([0.0, 1.0, 0.0, 0.0]);
    for format in [PixelFormat::R5G6B5Unorm, PixelFormat::B5G6R5Unorm] {
        assert_eq!(encode_clear(format, green), Some(vec![0xe0, 0x07]));
    }
    // Red alone tells R5G6B5 and B5G6R5 apart: red is the top field in one, the bottom in the
    // other.
    let red = float([1.0, 0.0, 0.0, 0.0]);
    assert_eq!(
        encode_clear(PixelFormat::R5G6B5Unorm, red),
        Some(vec![0x00, 0xf8])
    );
    assert_eq!(
        encode_clear(PixelFormat::B5G6R5Unorm, red),
        Some(vec![0x1f, 0x00])
    );
    assert_eq!(
        encode_clear(PixelFormat::A2R10G10B10Unorm, red),
        Some(vec![0x00, 0x00, 0xf0, 0x3f])
    );
    assert_eq!(
        encode_clear(PixelFormat::B8G8R8A8Unorm, red),
        Some(vec![0, 0, 0xff, 0])
    );
}

#[test]
fn normalised_values_clamp_and_only_endpoints_are_exact() {
    let f = PixelFormat::A8B8G8R8Unorm;
    assert_eq!(
        encode_clear(f, float([2.0, -3.0, f32::INFINITY, f32::NEG_INFINITY])),
        Some(vec![0xff, 0, 0xff, 0])
    );
    for between in [0.5, 1.0 / 255.0, f32::MIN_POSITIVE, 0.999_999_9] {
        assert_eq!(
            encode_clear(f, float([between, 0.0, 0.0, 0.0])),
            None,
            "{between}"
        );
    }
    assert_eq!(encode_clear(f, float([f32::NAN, 0.0, 0.0, 0.0])), None);
    // One inexact channel makes the whole texel unknown, whichever channel it is.
    for i in 0..4 {
        let mut c = [1.0; 4];
        c[i] = 0.25;
        assert_eq!(encode_clear(f, float(c)), None, "channel {i}");
    }
}

#[test]
fn snorm_endpoints() {
    // -1 is -127 (0x81) on 8 bits, not -128; 1 is 127.
    assert_eq!(
        encode_clear(PixelFormat::A8B8G8R8Snorm, float([-1.0, 1.0, 0.0, -5.0])),
        Some(vec![0x81, 0x7f, 0x00, 0x81])
    );
    assert_eq!(
        encode_clear(PixelFormat::R16Snorm, float([-1.0, 0.0, 0.0, 0.0])),
        Some(vec![0x01, 0x80])
    );
    assert_eq!(
        encode_clear(PixelFormat::R16G16Snorm, float([1.0, -0.0, 0.0, 0.0])),
        Some(vec![0xff, 0x7f, 0x00, 0x00])
    );
    for between in [0.5, 0.25, -0.25, f32::MIN_POSITIVE] {
        assert_eq!(
            encode_clear(PixelFormat::R8Snorm, float([between, 0.0, 0.0, 0.0])),
            None,
            "{between}"
        );
    }
    assert_eq!(
        encode_clear(PixelFormat::R8Snorm, float([f32::NAN, 0.0, 0.0, 0.0])),
        None
    );
}

#[test]
fn srgb_colour_channels_are_exact_only_at_the_ends() {
    let f = PixelFormat::A8B8G8R8Srgb;
    assert_eq!(
        encode_clear(f, float([1.0, 0.0, 1.0, 1.0])),
        Some(vec![0xff, 0, 0xff, 0xff])
    );
    assert_eq!(encode_clear(f, float([0.5, 0.0, 0.0, 1.0])), None);
    assert_eq!(
        encode_clear(PixelFormat::B8G8R8A8Srgb, float([1.0, 0.0, 0.0, 0.0])),
        Some(vec![0, 0, 0xff, 0])
    );
}

#[test]
fn float_formats() {
    // 1.5 = 0x3fc00000 (binary32), 0x3e00 (binary16); -2 = 0xc0000000, 0xc000.
    let v = float([1.5, -2.0, 0.0, 65504.0]);
    assert_eq!(
        encode_clear(PixelFormat::R32Float, v),
        Some(vec![0x00, 0x00, 0xc0, 0x3f])
    );
    assert_eq!(
        encode_clear(PixelFormat::R32G32Float, v),
        Some(vec![0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0x00, 0xc0])
    );
    assert_eq!(
        encode_clear(PixelFormat::R32G32B32Float, v),
        Some(vec![0, 0, 0xc0, 0x3f, 0, 0, 0, 0xc0, 0, 0, 0, 0])
    );
    assert_eq!(
        encode_clear(PixelFormat::R32G32B32A32Float, v),
        Some(vec![
            0, 0, 0xc0, 0x3f, 0, 0, 0, 0xc0, 0, 0, 0, 0, 0, 0xe0, 0x7f, 0x47
        ])
    );
    assert_eq!(
        encode_clear(PixelFormat::R16Float, v),
        Some(vec![0x00, 0x3e])
    );
    assert_eq!(
        encode_clear(PixelFormat::R16G16Float, v),
        Some(vec![0x00, 0x3e, 0x00, 0xc0])
    );
    // 65504 is binary16's largest finite value, 0x7bff.
    assert_eq!(
        encode_clear(PixelFormat::R16G16B16A16Float, v),
        Some(vec![0x00, 0x3e, 0x00, 0xc0, 0x00, 0x00, 0xff, 0x7b])
    );
    // NaN and subnormals are unknown in binary32.
    for bad in [f32::NAN, f32::from_bits(1)] {
        assert_eq!(
            encode_clear(PixelFormat::R32Float, float([bad, 0.0, 0.0, 0.0])),
            None
        );
    }
    assert_eq!(
        encode_clear(PixelFormat::R32Float, float([f32::INFINITY, 0.0, 0.0, 0.0])),
        Some(f32::INFINITY.to_bits().to_le_bytes().to_vec())
    );
}

#[test]
fn binary16_exactness() {
    for (v, bits) in [
        (1.0, Some(0x3c00)),
        (-2.0, Some(0xc000)),
        (0.5, Some(0x3800)),
        (0.0, Some(0x0000)),
        (-0.0, Some(0x8000)),
        (65504.0, Some(0x7bff)),
        (65520.0, None),
        (1.0 + 2.0_f32.powi(-10), Some(0x3c01)),
        (1.0 + 2.0_f32.powi(-11), None),
        // The smallest normal binary16 number, and half of it (subnormal: left out).
        (2.0_f32.powi(-14), Some(0x0400)),
        (2.0_f32.powi(-15), None),
        // The largest exponent binary16 has, and one past it.
        (2.0_f32.powi(15), Some(0x7800)),
        (2.0_f32.powi(16), None),
        (f32::INFINITY, Some(0x7c00)),
        (f32::NEG_INFINITY, Some(0xfc00)),
        (f32::NAN, None),
        (f32::from_bits(1), None),
    ] {
        assert_eq!(f16_exact(v), bits, "{v}");
    }
}

#[test]
fn integer_formats_and_their_ranges() {
    let u = ClearValue::Uint([1, 2, 3, 255]);
    assert_eq!(
        encode_clear(PixelFormat::A8B8G8R8Uint, u),
        Some(vec![1, 2, 3, 255])
    );
    assert_eq!(
        encode_clear(PixelFormat::A8B8G8R8Uint, ClearValue::Uint([1, 2, 3, 256])),
        None
    );
    // A2B10G10R10_UINT: R 0..10, G 10..20, B 20..30, A 30..32.
    // 1 | 2 << 10 | 3 << 20 | 3 << 30 = 0xc0300801.
    assert_eq!(
        encode_clear(PixelFormat::A2B10G10R10Uint, ClearValue::Uint([1, 2, 3, 3])),
        Some(vec![0x01, 0x08, 0x30, 0xc0])
    );
    assert_eq!(
        encode_clear(
            PixelFormat::A2B10G10R10Uint,
            ClearValue::Uint([1023, 0, 0, 4])
        ),
        None,
        "alpha has 2 bits"
    );
    assert_eq!(
        encode_clear(PixelFormat::R16Uint, ClearValue::Uint([65535, 7, 7, 7])),
        Some(vec![0xff, 0xff])
    );
    assert_eq!(
        encode_clear(PixelFormat::R16Uint, ClearValue::Uint([65536, 0, 0, 0])),
        None
    );
    assert_eq!(
        encode_clear(
            PixelFormat::R32G32B32A32Uint,
            ClearValue::Uint([u32::MAX, 1, 2, 3])
        ),
        Some(vec![
            0xff, 0xff, 0xff, 0xff, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0
        ])
    );
    // Signed: -128..=127 on 8 bits.
    assert_eq!(
        encode_clear(PixelFormat::R8G8Sint, ClearValue::Sint([-128, 127, 0, 0])),
        Some(vec![0x80, 0x7f])
    );
    for out in [-129, 128] {
        assert_eq!(
            encode_clear(PixelFormat::R8Sint, ClearValue::Sint([out, 0, 0, 0])),
            None,
            "{out}"
        );
    }
    assert_eq!(
        encode_clear(
            PixelFormat::R16G16Sint,
            ClearValue::Sint([-1, -32768, 0, 0])
        ),
        Some(vec![0xff, 0xff, 0x00, 0x80])
    );
    assert_eq!(
        encode_clear(PixelFormat::R32Sint, ClearValue::Sint([i32::MIN, 0, 0, 0])),
        Some(i32::MIN.to_le_bytes().to_vec())
    );
}

#[test]
fn depth_and_stencil() {
    assert_eq!(
        encode_clear(PixelFormat::D32Float, ClearValue::Depth(0.25)),
        Some(0.25_f32.to_le_bytes().to_vec())
    );
    assert_eq!(
        encode_clear(PixelFormat::D16Unorm, ClearValue::Depth(1.0)),
        Some(vec![0xff, 0xff])
    );
    assert_eq!(
        encode_clear(PixelFormat::D16Unorm, ClearValue::Depth(0.5)),
        None
    );
    for out in [-0.5, 1.5, f32::NAN] {
        assert_eq!(
            encode_clear(PixelFormat::D32Float, ClearValue::Depth(out)),
            None
        );
        assert_eq!(
            encode_clear(PixelFormat::D16Unorm, ClearValue::Depth(out)),
            None
        );
    }
    assert_eq!(
        encode_clear(PixelFormat::S8Uint, ClearValue::Stencil(255)),
        Some(vec![0xff])
    );
    assert_eq!(
        encode_clear(PixelFormat::S8Uint, ClearValue::Stencil(256)),
        None
    );
}

#[test]
fn a_value_of_the_wrong_kind_encodes_nothing() {
    assert_eq!(
        encode_clear(PixelFormat::A8B8G8R8Uint, float([1.0; 4])),
        None
    );
    assert_eq!(
        encode_clear(PixelFormat::A8B8G8R8Unorm, ClearValue::Uint([1; 4])),
        None
    );
    assert_eq!(
        encode_clear(PixelFormat::R8Sint, ClearValue::Uint([1; 4])),
        None
    );
    assert_eq!(
        encode_clear(PixelFormat::D32Float, ClearValue::Stencil(1)),
        None
    );
    assert_eq!(
        encode_clear(PixelFormat::S8Uint, ClearValue::Depth(1.0)),
        None
    );
    assert_eq!(
        encode_clear(PixelFormat::R32Float, ClearValue::Depth(1.0)),
        None,
        "depth into a colour format"
    );
}

/// The formats the module leaves out, listed once: every other uncompressed format is modelled.
const LEFT_OUT: [PixelFormat; 7] = [
    PixelFormat::A1B5G5R5Unorm,
    PixelFormat::A5B5G5R1Unorm,
    PixelFormat::G4R4Unorm,
    PixelFormat::R16G16B16X16Float,
    PixelFormat::B10G11R11Float,
    PixelFormat::E5B9G9R9Float,
    PixelFormat::X8D24Unorm,
];

#[test]
fn every_uncompressed_single_aspect_format_but_the_listed_ones_is_modelled() {
    for format in PixelFormat::ALL {
        let value = match format.aspect() {
            FormatAspect::DepthStencil => ClearValue::DepthStencil {
                depth: 1.0,
                stencil: 1,
            },
            FormatAspect::Depth => ClearValue::Depth(1.0),
            FormatAspect::Stencil => ClearValue::Stencil(1),
            FormatAspect::Color if format.is_signed_integer() => ClearValue::Sint([0; 4]),
            FormatAspect::Color if format.is_integer() => ClearValue::Uint([0; 4]),
            FormatAspect::Color => float([0.0; 4]),
        };
        let modelled = format.compression() == Compression::Uncompressed
            && format.aspect() != FormatAspect::DepthStencil
            && !LEFT_OUT.contains(&format);
        let got = encode_clear(format, value);
        assert_eq!(got.is_some(), modelled, "{format}");
        if let Some(bytes) = got {
            assert_eq!(
                bytes.len(),
                usize::try_from(format.bytes_per_block()).expect("small"),
                "{format}"
            );
        }
    }
}
