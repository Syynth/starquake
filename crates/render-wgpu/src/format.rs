//! Which wgpu texture format holds each [`PixelFormat`] natively, byte for byte.
//!
//! The map goes through the Vulkan format Eden gives each guest format
//! (`eden/src/video_core/renderer_vulkan/maxwell_to_vk.cpp` @ cf654052, `SURFACE_FORMAT_ELEM`
//! rows) and the wgpu format that names the same Vulkan format (wgpu-hal 30.0.2,
//! `src/vulkan/conv.rs`, `map_texture_format`). A guest format has a wgpu format here only when
//! the two store the same bytes in the same order and mean the same channels; every other guest
//! format has none, so the device reports no support for it ([`crate::caps`]) and the shared
//! fallback choice (`starquake_render::host_format`) picks its stand-in. Those, with the reason:
//!
//! - no wgpu format at all: `R5G6B5_UNORM`, `B5G6R5_UNORM`, `A1R5G5B5_UNORM`,
//!   `A2R10G10B10_UNORM`, `A1B5G5R5_UNORM`, `A5B5G5R1_UNORM`, `R32G32B32_FLOAT`,
//!   `A4B4G4R4_UNORM`, `G4R4_UNORM`, `X8_D24_UNORM`, `S8_UINT_D24_UNORM`;
//! - `D24_UNORM_S8_UINT`: wgpu's `Depth24PlusStencil8` promises at least 24 bits of depth, not
//!   24-bit unorm (on Apple GPUs it is 32-bit float), so it is not this format;
//! - `R16G16B16X16_FLOAT`: Eden gives it Vulkan's `R16G16B16A16_SFLOAT`, but sampling that reads
//!   the fourth channel from memory where the guest's format reads one; the fallback's stand-in
//!   (with its conversion) is used instead.
//!
//! `format::tests` checks every entry against wgpu's own description of the format (block size,
//! bytes per block, sRGB, channel type, channel order), which is a second source.

use starquake_render::PixelFormat;
use wgpu::{AstcBlock, AstcChannel, TextureFormat as T};

/// An ASTC format of `block`, sRGB when `srgb`.
const fn astc(block: AstcBlock, srgb: bool) -> T {
    T::Astc {
        block,
        channel: if srgb {
            AstcChannel::UnormSrgb
        } else {
            AstcChannel::Unorm
        },
    }
}

/// The wgpu format that holds `format` natively, or `None` (module docs).
#[expect(
    clippy::too_many_lines,
    reason = "one arm per guest format keeps the whole table in one place"
)]
pub const fn texture_format(format: PixelFormat) -> Option<wgpu::TextureFormat> {
    use PixelFormat as P;
    Some(match format {
        P::A8B8G8R8Unorm => T::Rgba8Unorm,
        P::A8B8G8R8Snorm => T::Rgba8Snorm,
        P::A8B8G8R8Sint => T::Rgba8Sint,
        P::A8B8G8R8Uint => T::Rgba8Uint,
        P::A2B10G10R10Unorm => T::Rgb10a2Unorm,
        P::A2B10G10R10Uint => T::Rgb10a2Uint,
        P::R8Unorm => T::R8Unorm,
        P::R8Snorm => T::R8Snorm,
        P::R8Sint => T::R8Sint,
        P::R8Uint => T::R8Uint,
        P::R16G16B16A16Float => T::Rgba16Float,
        P::R16G16B16A16Unorm => T::Rgba16Unorm,
        P::R16G16B16A16Snorm => T::Rgba16Snorm,
        P::R16G16B16A16Sint => T::Rgba16Sint,
        P::R16G16B16A16Uint => T::Rgba16Uint,
        P::B10G11R11Float => T::Rg11b10Ufloat,
        P::R32G32B32A32Uint => T::Rgba32Uint,
        P::Bc1RgbaUnorm => T::Bc1RgbaUnorm,
        P::Bc2Unorm => T::Bc2RgbaUnorm,
        P::Bc3Unorm => T::Bc3RgbaUnorm,
        P::Bc4Unorm => T::Bc4RUnorm,
        P::Bc4Snorm => T::Bc4RSnorm,
        P::Bc5Unorm => T::Bc5RgUnorm,
        P::Bc5Snorm => T::Bc5RgSnorm,
        P::Bc7Unorm => T::Bc7RgbaUnorm,
        P::Bc6hUfloat => T::Bc6hRgbUfloat,
        P::Bc6hSfloat => T::Bc6hRgbFloat,
        P::Astc2d4x4Unorm => astc(AstcBlock::B4x4, false),
        P::B8G8R8A8Unorm => T::Bgra8Unorm,
        P::R32G32B32A32Float => T::Rgba32Float,
        P::R32G32B32A32Sint => T::Rgba32Sint,
        P::R32G32Float => T::Rg32Float,
        P::R32G32Sint => T::Rg32Sint,
        P::R32Float => T::R32Float,
        P::R16Float => T::R16Float,
        P::R16Unorm => T::R16Unorm,
        P::R16Snorm => T::R16Snorm,
        P::R16Uint => T::R16Uint,
        P::R16Sint => T::R16Sint,
        P::R16G16Unorm => T::Rg16Unorm,
        P::R16G16Float => T::Rg16Float,
        P::R16G16Uint => T::Rg16Uint,
        P::R16G16Sint => T::Rg16Sint,
        P::R16G16Snorm => T::Rg16Snorm,
        P::A8B8G8R8Srgb => T::Rgba8UnormSrgb,
        P::R8G8Unorm => T::Rg8Unorm,
        P::R8G8Snorm => T::Rg8Snorm,
        P::R8G8Sint => T::Rg8Sint,
        P::R8G8Uint => T::Rg8Uint,
        P::R32G32Uint => T::Rg32Uint,
        P::R32Uint => T::R32Uint,
        P::R32Sint => T::R32Sint,
        P::Astc2d8x8Unorm => astc(AstcBlock::B8x8, false),
        P::Astc2d8x5Unorm => astc(AstcBlock::B8x5, false),
        P::Astc2d5x4Unorm => astc(AstcBlock::B5x4, false),
        P::B8G8R8A8Srgb => T::Bgra8UnormSrgb,
        P::Bc1RgbaSrgb => T::Bc1RgbaUnormSrgb,
        P::Bc2Srgb => T::Bc2RgbaUnormSrgb,
        P::Bc3Srgb => T::Bc3RgbaUnormSrgb,
        P::Bc7Srgb => T::Bc7RgbaUnormSrgb,
        P::Astc2d4x4Srgb => astc(AstcBlock::B4x4, true),
        P::Astc2d8x8Srgb => astc(AstcBlock::B8x8, true),
        P::Astc2d8x5Srgb => astc(AstcBlock::B8x5, true),
        P::Astc2d5x4Srgb => astc(AstcBlock::B5x4, true),
        P::Astc2d5x5Unorm => astc(AstcBlock::B5x5, false),
        P::Astc2d5x5Srgb => astc(AstcBlock::B5x5, true),
        P::Astc2d10x8Unorm => astc(AstcBlock::B10x8, false),
        P::Astc2d10x8Srgb => astc(AstcBlock::B10x8, true),
        P::Astc2d6x6Unorm => astc(AstcBlock::B6x6, false),
        P::Astc2d6x6Srgb => astc(AstcBlock::B6x6, true),
        P::Astc2d10x6Unorm => astc(AstcBlock::B10x6, false),
        P::Astc2d10x6Srgb => astc(AstcBlock::B10x6, true),
        P::Astc2d10x5Unorm => astc(AstcBlock::B10x5, false),
        P::Astc2d10x5Srgb => astc(AstcBlock::B10x5, true),
        P::Astc2d10x10Unorm => astc(AstcBlock::B10x10, false),
        P::Astc2d10x10Srgb => astc(AstcBlock::B10x10, true),
        P::Astc2d12x10Unorm => astc(AstcBlock::B12x10, false),
        P::Astc2d12x10Srgb => astc(AstcBlock::B12x10, true),
        P::Astc2d12x12Unorm => astc(AstcBlock::B12x12, false),
        P::Astc2d12x12Srgb => astc(AstcBlock::B12x12, true),
        P::Astc2d8x6Unorm => astc(AstcBlock::B8x6, false),
        P::Astc2d8x6Srgb => astc(AstcBlock::B8x6, true),
        P::Astc2d6x5Unorm => astc(AstcBlock::B6x5, false),
        P::Astc2d6x5Srgb => astc(AstcBlock::B6x5, true),
        P::E5B9G9R9Float => T::Rgb9e5Ufloat,
        P::Etc2RgbUnorm => T::Etc2Rgb8Unorm,
        P::Etc2RgbaUnorm => T::Etc2Rgba8Unorm,
        P::Etc2RgbPtaUnorm => T::Etc2Rgb8A1Unorm,
        P::Etc2RgbSrgb => T::Etc2Rgb8UnormSrgb,
        P::Etc2RgbaSrgb => T::Etc2Rgba8UnormSrgb,
        P::Etc2RgbPtaSrgb => T::Etc2Rgb8A1UnormSrgb,
        P::EacR11Unorm => T::EacR11Unorm,
        P::EacR11Snorm => T::EacR11Snorm,
        P::EacR11G11Unorm => T::EacRg11Unorm,
        P::EacR11G11Snorm => T::EacRg11Snorm,
        P::D32Float => T::Depth32Float,
        P::D16Unorm => T::Depth16Unorm,
        P::S8Uint => T::Stencil8,
        P::D32FloatS8Uint => T::Depth32FloatStencil8,
        P::R5G6B5Unorm
        | P::B5G6R5Unorm
        | P::A1R5G5B5Unorm
        | P::A2R10G10B10Unorm
        | P::A1B5G5R5Unorm
        | P::A5B5G5R1Unorm
        | P::R32G32B32Float
        | P::R16G16B16X16Float
        | P::A4B4G4R4Unorm
        | P::G4R4Unorm
        | P::X8D24Unorm
        | P::D24UnormS8Uint
        | P::S8UintD24Unorm => return None,
    })
}

#[cfg(test)]
mod tests;
