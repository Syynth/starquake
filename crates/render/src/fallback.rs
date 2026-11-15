//! Format fallbacks: which host format a guest format gets when the device lacks it
//! (docs/specs/render-backend.md, "Formats"; slice R2).
//!
//! Each backend reports, per [`PixelFormat`], what its device can do with the format natively
//! ([`FormatTable`]). [`host_format`] is the one pure function that chooses from that report,
//! so the choice is the same for every backend and is tested once, with synthetic tables.
//!
//! # The choice
//!
//! Given a guest format and the [`FormatFeatures`] its texture needs
//! ([`features_for_usage`], plus [`FormatFeatures::FILTER`] or [`FormatFeatures::BLEND`] when
//! the caller knows it will filter or blend):
//!
//! 1. the guest format itself, when the device has every needed feature for it;
//! 2. otherwise the first of the guest format's stand-ins, in the order below, that has every
//!    needed feature, with the [`Conversion`] its data needs where it crosses guest memory;
//! 3. otherwise [`RenderError::FormatUnsupported`], naming the guest format and the first
//!    needed feature (in the order sampling, filtering, render targets, blending, storage
//!    images, copies) that the device lacks for it.
//!
//! # The stand-ins
//!
//! The spec's table of known conversions, with its third row ("formats the host can't render
//! to or blend: a wider format") made concrete by Eden's list of alternatives:
//! `GetFormatAlternatives` in `eden/src/video_core/vulkan_common/vulkan_device.cpp` and
//! `SurfaceFormat` in `eden/src/video_core/renderer_vulkan/maxwell_to_vk.cpp` (its ASTC rule,
//! and the rows of its table that give a guest format another format on every device), both
//! @ cf654052 (GPL-3.0-or-later), from which the lists below are ported.
//!
//! | Guest format | Stand-ins, in order | Conversion | Source |
//! |---|---|---|---|
//! | Every ASTC format (all block sizes) | `A8B8G8R8_UNORM`, or `A8B8G8R8_SRGB` for an sRGB format | [`Conversion::AstcDecode`] | The spec's first row; Eden's `SurfaceFormat` |
//! | `D24_UNORM_S8_UINT` | `D32_FLOAT_S8_UINT` | [`Conversion::Texel`] | The spec's second row; Eden's `DEPTH24_UNORM_STENCIL8_UINT` |
//! | `S8_UINT_D24_UNORM` | `D24_UNORM_S8_UINT`, `D32_FLOAT_S8_UINT` | [`Conversion::Texel`] | Eden's `SurfaceFormat` maps it to Vulkan's D24S8 ("emulated"), then `DEPTH24_UNORM_STENCIL8_UINT` |
//! | `S8_UINT` | `D24_UNORM_S8_UINT`, `D32_FLOAT_S8_UINT` | [`Conversion::Texel`] | Eden's `STENCIL8_UINT` |
//! | `X8_D24_UNORM` | `D32_FLOAT` | [`Conversion::Texel`] | Eden's `DEPTH24_UNORM_DONTCARE8` |
//! | `B5G6R5_UNORM` | `R5G6B5_UNORM` | [`Conversion::Texel`] | Eden's `B5G6R5_UNORM_PACK16` |
//! | `R32G32B32_FLOAT` | `R32G32B32A32_FLOAT` | [`Conversion::Texel`] | Eden's `VK_FORMAT_R32G32B32_SFLOAT` |
//! | `B10G11R11_FLOAT` | `R16G16B16A16_FLOAT` | [`Conversion::Texel`] | Eden's `B10G11R11_UFLOAT_PACK32` |
//! | `A1B5G5R5_UNORM` | `A1R5G5B5_UNORM` | [`Conversion::Texel`] | Eden's `SurfaceFormat` maps it to Vulkan's `A1R5G5B5_UNORM_PACK16` ("flipped with swizzle") |
//! | `R16G16B16X16_FLOAT` | `R16G16B16A16_FLOAT` | [`Conversion::Texel`] | Eden's `SurfaceFormat` maps it to Vulkan's `R16G16B16A16_SFLOAT` |
//!
//! Every other format has no stand-in: the device has it, or the texture cannot be made.
//!
//! # Where this differs from Eden
//!
//! Each names the test in `fallback::tests` that pins it.
//!
//! - **Stricter: nothing fits.** Eden logs an error and uses the wanted format anyway; here it
//!   is an error the GPU crate counts. `the_error_names_the_first_missing_feature`,
//!   `every_combination`.
//! - **Stricter: no narrower or lossy stand-in.** Eden's lists end with `D16_UNORM` for
//!   `X8_D24_UNORM` (16 bits of depth for 24) and `A8B8G8R8_SRGB` for `B10G11R11_FLOAT` (8-bit
//!   sRGB for unsigned floats), and offer `R8_UNORM` for `G4R4_UNORM` (one channel for two).
//!   None can hold the guest's values, so none is offered. The spec's row says "a wider
//!   format"; the rule here is that a stand-in can hold the guest's values, which one of the
//!   same width can (`R5G6B5_UNORM` for `B5G6R5_UNORM`). `where_this_differs_from_eden`,
//!   and `a_stand_in_can_hold_its_guest` for every stand-in.
//! - **Not expressible.** Eden's alternatives that [`PixelFormat`] has no entry for are left
//!   out (`D16_UNORM_S8_UINT`; `R4G4B4A4_UNORM` for `A4B4G4R4_UNORM`), as is the Vulkan format
//!   its `SurfaceFormat` gives `A5B5G5R1_UNORM` on every device (`R5G5B5A1_UNORM_PACK16`,
//!   "specially swizzled"), and its entries for vertex formats, which are not texture formats.
//!   `where_this_differs_from_eden`.
//! - **Per report, not on every device.** Eden's `SurfaceFormat` gives `S8_UINT_D24_UNORM`,
//!   `A1B5G5R5_UNORM` and `R16G16B16X16_FLOAT` another format whatever the device has. Here
//!   that format is a stand-in, used only when the device's report lacks the guest format.
//!   `every_combination` (a device that has the guest format uses it).
//! - **Per format, not per device.** Eden decodes every ASTC format unless the device has all
//!   of them (`ComputeIsOptimalAstcSupported`). Here each format is decided on its own report.
//!   `astc_is_decided_per_format`.
//! - **Not ported: recompression, BC, ETC2 and EAC.** Eden can recompress decoded ASTC to BC1
//!   or BC3 (the spec defers that, decision 6), and decodes BC formats and ETC2 and EAC
//!   formats on devices without them (`IsOptimalBcnSupported`, `IsOptimalEtc2Supported`); the
//!   spec's table has no row for either and no decoder for them is planned, so a BC, ETC2 or
//!   EAC format the device lacks is an error. `where_this_differs_from_eden`.
//! - **Conversion instead of a view swizzle.** Eden fixes the channel order of `B5G6R5_UNORM`
//!   and `A1B5G5R5_UNORM` with the view's swizzle; a device without view swizzles cannot, so
//!   the data is converted. `every_combination` (the conversion of each stand-in).

use crate::caps::{FormatFeatures, FormatTable};
use crate::error::RenderError;
use crate::format::{Compression, PixelFormat};
use crate::resource::TextureUsage;
use crate::validate::USAGE_FEATURES;

/// What happens to a texture's data where it crosses guest memory (uploads and readbacks) when
/// its host format is not the guest's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Conversion {
    /// The host format is the guest's: the bytes cross as they are.
    None,
    /// ASTC blocks are decoded into RGBA8 texels on upload: on the GPU where it can (slice
    /// R10), with the CPU decoder (slice R3) as the fallback and the oracle. One way only:
    /// there is no encoder, so reading such a texture back into guest memory is not supported.
    AstcDecode,
    /// Texels are converted one by one on upload, and back on readback.
    Texel,
}

/// The host format a guest format is realised as, and how its data is converted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HostFormat {
    /// The format the host texture is created with.
    pub format: PixelFormat,
    /// What its data needs where it crosses guest memory. With [`Conversion::AstcDecode`]
    /// that is uploads only: such a texture cannot be read back.
    pub conversion: Conversion,
}

/// The format features a texture with `usage` needs of its format: sampling for a sampled
/// texture, storage images for a storage texture, render targets for a render target, and
/// copies for either copy usage. Filtering and blending are not usages: a caller that knows it
/// will filter or blend adds [`FormatFeatures::FILTER`] or [`FormatFeatures::BLEND`] itself.
pub fn features_for_usage(usage: TextureUsage) -> FormatFeatures {
    USAGE_FEATURES
        .into_iter()
        .filter(|(flag, _)| usage.contains(*flag))
        .fold(FormatFeatures::EMPTY, |features, (_, feature)| {
            features | feature
        })
}

/// The formats that can stand in for `guest`, most preferred first, and the conversion their
/// data needs (module docs, "The stand-ins").
fn stand_ins(guest: PixelFormat) -> (&'static [PixelFormat], Conversion) {
    use PixelFormat as F;
    if guest.compression() == Compression::Astc {
        let rgba8: &[PixelFormat] = if guest.is_srgb() {
            &[F::A8B8G8R8Srgb]
        } else {
            &[F::A8B8G8R8Unorm]
        };
        return (rgba8, Conversion::AstcDecode);
    }
    let formats: &[PixelFormat] = match guest {
        F::D24UnormS8Uint => &[F::D32FloatS8Uint],
        F::S8UintD24Unorm | F::S8Uint => &[F::D24UnormS8Uint, F::D32FloatS8Uint],
        F::X8D24Unorm => &[F::D32Float],
        F::B5G6R5Unorm => &[F::R5G6B5Unorm],
        F::R32G32B32Float => &[F::R32G32B32A32Float],
        F::B10G11R11Float | F::R16G16B16X16Float => &[F::R16G16B16A16Float],
        F::A1B5G5R5Unorm => &[F::A1R5G5B5Unorm],
        _ => &[],
    };
    (formats, Conversion::Texel)
}

/// Chooses the host format for `guest` on a device with `formats`, for a texture that needs
/// `needed` of its format (module docs).
pub fn host_format(
    formats: &FormatTable,
    guest: PixelFormat,
    needed: FormatFeatures,
) -> Result<HostFormat, RenderError> {
    let missing = needed.difference(formats.get(guest));
    // The first needed feature the device lacks for the guest format, in bit order.
    let Some((_, first_missing)) = FormatFeatures::NAMED
        .iter()
        .find(|(_, feature)| missing.contains(*feature))
    else {
        return Ok(HostFormat {
            format: guest,
            conversion: Conversion::None,
        });
    };
    let (candidates, conversion) = stand_ins(guest);
    candidates
        .iter()
        .find(|candidate| formats.get(**candidate).contains(needed))
        .map(|candidate| HostFormat {
            format: *candidate,
            conversion,
        })
        .ok_or(RenderError::FormatUnsupported {
            format: guest,
            usage: first_missing.in_words(),
        })
}

#[cfg(test)]
mod tests;
