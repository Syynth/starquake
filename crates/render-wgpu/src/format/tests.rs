//! The format map checked against wgpu's own description of each format.

use super::texture_format;
use starquake_render::PixelFormat;
use starquake_render::format::{Compression, FormatAspect, NumericType};
use wgpu::TextureFormat;

/// What wgpu's `Debug` name says about a format's channel type, the one property neither
/// `TextureFormat::sample_type` (snorm and unorm both sample as float) nor anything else wgpu
/// exposes separates.
fn numeric_from_name(name: &str) -> Option<NumericType> {
    // Order matters: "Ufloat" before "Float", "UnormSrgb" contains "Unorm".
    [
        ("Ufloat", NumericType::Float),
        ("Float", NumericType::Float),
        ("Unorm", NumericType::Unorm),
        ("Snorm", NumericType::Snorm),
        ("Uint", NumericType::Uint),
        ("Sint", NumericType::Sint),
    ]
    .into_iter()
    .find(|(suffix, _)| name.contains(suffix))
    .map(|(_, numeric)| numeric)
}

#[test]
fn every_entry_stores_the_same_blocks_as_its_guest_format() {
    for format in PixelFormat::ALL {
        let Some(host) = texture_format(format) else {
            continue;
        };
        let name = format.name();
        assert_eq!(
            host.block_dimensions(),
            (format.block_width(), format.block_height()),
            "{name}: block size"
        );
        let aspect = match format.aspect() {
            FormatAspect::Color | FormatAspect::DepthStencil => None,
            FormatAspect::Depth => Some(wgpu::TextureAspect::DepthOnly),
            FormatAspect::Stencil => Some(wgpu::TextureAspect::StencilOnly),
        };
        if format.aspect() != FormatAspect::DepthStencil {
            assert_eq!(
                host.block_copy_size(aspect),
                Some(format.bytes_per_block()),
                "{name}: bytes per block"
            );
        }
        assert_eq!(host.is_srgb(), format.is_srgb(), "{name}: sRGB");
        assert_eq!(host.has_depth_aspect(), format.has_depth(), "{name}: depth");
        assert_eq!(
            host.has_stencil_aspect(),
            format.has_stencil(),
            "{name}: stencil"
        );
        assert_eq!(
            host.is_compressed(),
            format.compression() != Compression::Uncompressed,
            "{name}: compression"
        );
        assert_eq!(
            host.is_bcn(),
            format.compression() == Compression::Bc,
            "{name}: BC"
        );
        assert_eq!(
            host.is_astc(),
            format.compression() == Compression::Astc,
            "{name}: ASTC"
        );
        let debug = format!("{host:?}");
        if format.aspect() == FormatAspect::Color {
            assert_eq!(
                numeric_from_name(&debug),
                Some(format.numeric()),
                "{name}: channel type of {debug}"
            );
            // Channel order: Eden's B8G8R8A8 formats are wgpu's Bgra8 ones, and nothing else is.
            assert_eq!(
                debug.starts_with("Bgra"),
                name.starts_with("B8G8R8A8"),
                "{name}: channel order of {debug}"
            );
        }
    }
}

#[test]
fn hand_checked_entries() {
    // Written out from Eden's table (maxwell_to_vk.cpp) and wgpu-hal's Vulkan map (conv.rs):
    // the guest format, Eden's Vulkan format, and the wgpu format wgpu-hal maps to it.
    let cases = [
        (PixelFormat::A8B8G8R8Unorm, TextureFormat::Rgba8Unorm), // A8B8G8R8_UNORM_PACK32
        (PixelFormat::B8G8R8A8Srgb, TextureFormat::Bgra8UnormSrgb), // B8G8R8A8_SRGB
        (PixelFormat::A2B10G10R10Uint, TextureFormat::Rgb10a2Uint), // A2B10G10R10_UINT_PACK32
        (PixelFormat::B10G11R11Float, TextureFormat::Rg11b10Ufloat), // B10G11R11_UFLOAT_PACK32
        (PixelFormat::E5B9G9R9Float, TextureFormat::Rgb9e5Ufloat), // E5B9G9R9_UFLOAT_PACK32
        (PixelFormat::Bc6hSfloat, TextureFormat::Bc6hRgbFloat),  // BC6H_SFLOAT_BLOCK
        (
            PixelFormat::Etc2RgbPtaSrgb,
            TextureFormat::Etc2Rgb8A1UnormSrgb,
        ), // ETC2_R8G8B8A1_SRGB
        (PixelFormat::EacR11G11Snorm, TextureFormat::EacRg11Snorm), // EAC_R11G11_SNORM_BLOCK
        (
            PixelFormat::D32FloatS8Uint,
            TextureFormat::Depth32FloatStencil8,
        ), // D32_SFLOAT_S8_UINT
        (
            PixelFormat::Astc2d10x6Srgb,
            TextureFormat::Astc {
                block: wgpu::AstcBlock::B10x6,
                channel: wgpu::AstcChannel::UnormSrgb,
            },
        ), // ASTC_10x6_SRGB_BLOCK
    ];
    for (guest, host) in cases {
        assert_eq!(texture_format(guest), Some(host), "{}", guest.name());
    }
}

#[test]
fn formats_without_an_exact_host_form_have_none() {
    let none = [
        PixelFormat::R5G6B5Unorm,
        PixelFormat::B5G6R5Unorm,
        PixelFormat::A1R5G5B5Unorm,
        PixelFormat::A2R10G10B10Unorm,
        PixelFormat::A1B5G5R5Unorm,
        PixelFormat::A5B5G5R1Unorm,
        PixelFormat::R32G32B32Float,
        PixelFormat::R16G16B16X16Float,
        PixelFormat::A4B4G4R4Unorm,
        PixelFormat::G4R4Unorm,
        PixelFormat::X8D24Unorm,
        PixelFormat::D24UnormS8Uint,
        PixelFormat::S8UintD24Unorm,
    ];
    for format in PixelFormat::ALL {
        assert_eq!(
            texture_format(format).is_none(),
            none.contains(&format),
            "{}",
            format.name()
        );
    }
}

#[test]
fn no_two_guest_formats_share_a_host_format() {
    let mut seen = std::collections::BTreeMap::new();
    for format in PixelFormat::ALL {
        if let Some(host) = texture_format(format) {
            let key = format!("{host:?}");
            assert_eq!(seen.insert(key.clone(), format), None, "{key}");
        }
    }
}
