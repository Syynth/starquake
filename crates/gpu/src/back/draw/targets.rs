//! Render targets: which colour and depth surfaces a draw renders into, their formats, and the
//! images the texture cache makes for them (gpu-protocol spec, "3D: from state to a draw" step
//! 3, "render targets").
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`; Eden is GPL-3.0-or-later.
//!
//! - Formats: Eden's `PixelFormatFromRenderTargetFormat` and `PixelFormatFromDepthFormat`
//!   (`eden/src/video_core/surface.cpp` @ cf654052), keyed here on the header's values
//!   (`SET_COLOR_TARGET_FORMAT_V_*`, `SET_ZT_FORMAT_V_*` in `classes/clc797.h`), which equal
//!   Eden's `RenderTargetFormat` and `DepthFormat` numbers (`eden/src/video_core/gpu.h`). A
//!   format Eden does not map (its `default:` arm falls back to `A8B8G8R8_UNORM` with a log) is
//!   refused here ([`DrawError::TargetFormat`]); a listed divergence, pinned by
//!   `tests::target_formats_eden_does_not_map_are_refused`.
//! - Images: Eden's `ImageInfo(const RenderTargetConfig&, MsaaMode)` and
//!   `ImageInfo(const Zeta&, const ZetaSize&, MsaaMode)` (`texture_cache/image_info.cpp`): a
//!   pitch colour target's width register is its pitch in bytes; a block-linear one is an array
//!   of `THIRD_DIMENSION` layers, or a 3D image that deep when `THIRD_DIMENSION_CONTROL` says so;
//!   a depth target's layers follow `SET_ZT_SIZE_C_CONTROL`. One level, the samples of
//!   `SET_ANTI_ALIAS` (Eden's `NumSamples`, the grids of `samples_helper.h`).
//! - Which targets: `SET_CT_SELECT`'s count and its map from output to colour target (Eden's
//!   `rt_control.Map`); the depth target when `SET_ZT_SELECT`'s count is not zero.
//!
//! # Guesses and divergences
//!
//! - A block-linear target whose `THIRD_DIMENSION` (or a depth target's array size) is 0 is one
//!   layer here; Eden takes 0 layers. A **guess** that a title never means "no layers".
//!   `tests::a_zero_layer_target_is_one_layer`.
//! - A depth target is always block-linear: `clc797.h`'s `SET_ZT_*` methods have no layout
//!   field, where Eden's `Zeta::tile_mode` has a pitch flag.
//! - A pitch colour target is single-sampled: Eden's pitch branch returns before it reads the
//!   samples.

use starquake_layout::{ImageDim, SampleLayout};
use starquake_render::PixelFormat;
use starquake_render::caps::GUEST_COLOR_TARGETS;
use starquake_render::pipeline::TargetFormats;

use super::DrawError;
use crate::back::state::{COLOR_TARGETS, Mirror, ThreeD};
use crate::back::texture::{ImageInfo, MemoryLayout, gob_for};
use crate::classes::clc797;

/// The host format of a colour target format value; `None` for `DISABLED`.
pub fn color_format(value: u32) -> Result<Option<PixelFormat>, DrawError> {
    use PixelFormat as P;
    use clc797 as c;
    Ok(Some(match value {
        c::SET_COLOR_TARGET_FORMAT_V_DISABLED => return Ok(None),
        c::SET_COLOR_TARGET_FORMAT_V_RF32_GF32_BF32_AF32
        | c::SET_COLOR_TARGET_FORMAT_V_RF32_GF32_BF32_X32 => P::R32G32B32A32Float,
        c::SET_COLOR_TARGET_FORMAT_V_RS32_GS32_BS32_AS32
        | c::SET_COLOR_TARGET_FORMAT_V_RS32_GS32_BS32_X32 => P::R32G32B32A32Sint,
        c::SET_COLOR_TARGET_FORMAT_V_RU32_GU32_BU32_AU32
        | c::SET_COLOR_TARGET_FORMAT_V_RU32_GU32_BU32_X32 => P::R32G32B32A32Uint,
        c::SET_COLOR_TARGET_FORMAT_V_R16_G16_B16_A16 => P::R16G16B16A16Unorm,
        c::SET_COLOR_TARGET_FORMAT_V_RN16_GN16_BN16_AN16 => P::R16G16B16A16Snorm,
        c::SET_COLOR_TARGET_FORMAT_V_RS16_GS16_BS16_AS16 => P::R16G16B16A16Sint,
        c::SET_COLOR_TARGET_FORMAT_V_RU16_GU16_BU16_AU16 => P::R16G16B16A16Uint,
        c::SET_COLOR_TARGET_FORMAT_V_RF16_GF16_BF16_AF16 => P::R16G16B16A16Float,
        c::SET_COLOR_TARGET_FORMAT_V_RF32_GF32 => P::R32G32Float,
        c::SET_COLOR_TARGET_FORMAT_V_RS32_GS32 => P::R32G32Sint,
        c::SET_COLOR_TARGET_FORMAT_V_RU32_GU32 => P::R32G32Uint,
        c::SET_COLOR_TARGET_FORMAT_V_RF16_GF16_BF16_X16 => P::R16G16B16X16Float,
        c::SET_COLOR_TARGET_FORMAT_V_A8R8G8B8 | c::SET_COLOR_TARGET_FORMAT_V_X8R8G8B8 => {
            P::B8G8R8A8Unorm
        }
        c::SET_COLOR_TARGET_FORMAT_V_A8RL8GL8BL8 | c::SET_COLOR_TARGET_FORMAT_V_X8RL8GL8BL8 => {
            P::B8G8R8A8Srgb
        }
        c::SET_COLOR_TARGET_FORMAT_V_A2B10G10R10 => P::A2B10G10R10Unorm,
        c::SET_COLOR_TARGET_FORMAT_V_AU2BU10GU10RU10 => P::A2B10G10R10Uint,
        c::SET_COLOR_TARGET_FORMAT_V_A2R10G10B10 => P::A2R10G10B10Unorm,
        c::SET_COLOR_TARGET_FORMAT_V_A8B8G8R8 | c::SET_COLOR_TARGET_FORMAT_V_X8B8G8R8 => {
            P::A8B8G8R8Unorm
        }
        c::SET_COLOR_TARGET_FORMAT_V_A8BL8GL8RL8 | c::SET_COLOR_TARGET_FORMAT_V_X8BL8GL8RL8 => {
            P::A8B8G8R8Srgb
        }
        c::SET_COLOR_TARGET_FORMAT_V_AN8BN8GN8RN8 => P::A8B8G8R8Snorm,
        c::SET_COLOR_TARGET_FORMAT_V_AS8BS8GS8RS8 => P::A8B8G8R8Sint,
        c::SET_COLOR_TARGET_FORMAT_V_AU8BU8GU8RU8 => P::A8B8G8R8Uint,
        c::SET_COLOR_TARGET_FORMAT_V_R16_G16 => P::R16G16Unorm,
        c::SET_COLOR_TARGET_FORMAT_V_RN16_GN16 => P::R16G16Snorm,
        c::SET_COLOR_TARGET_FORMAT_V_RS16_GS16 => P::R16G16Sint,
        c::SET_COLOR_TARGET_FORMAT_V_RU16_GU16 => P::R16G16Uint,
        c::SET_COLOR_TARGET_FORMAT_V_RF16_GF16 => P::R16G16Float,
        c::SET_COLOR_TARGET_FORMAT_V_BF10GF11RF11 => P::B10G11R11Float,
        c::SET_COLOR_TARGET_FORMAT_V_RS32 => P::R32Sint,
        c::SET_COLOR_TARGET_FORMAT_V_RU32 => P::R32Uint,
        c::SET_COLOR_TARGET_FORMAT_V_RF32 => P::R32Float,
        c::SET_COLOR_TARGET_FORMAT_V_R5G6B5 => P::R5G6B5Unorm,
        c::SET_COLOR_TARGET_FORMAT_V_A1R5G5B5 | c::SET_COLOR_TARGET_FORMAT_V_X1R5G5B5 => {
            P::A1R5G5B5Unorm
        }
        c::SET_COLOR_TARGET_FORMAT_V_G8R8 => P::R8G8Unorm,
        c::SET_COLOR_TARGET_FORMAT_V_GN8RN8 => P::R8G8Snorm,
        c::SET_COLOR_TARGET_FORMAT_V_GS8RS8 => P::R8G8Sint,
        c::SET_COLOR_TARGET_FORMAT_V_GU8RU8 => P::R8G8Uint,
        c::SET_COLOR_TARGET_FORMAT_V_R16 => P::R16Unorm,
        c::SET_COLOR_TARGET_FORMAT_V_RN16 => P::R16Snorm,
        c::SET_COLOR_TARGET_FORMAT_V_RS16 => P::R16Sint,
        c::SET_COLOR_TARGET_FORMAT_V_RU16 => P::R16Uint,
        c::SET_COLOR_TARGET_FORMAT_V_RF16 => P::R16Float,
        c::SET_COLOR_TARGET_FORMAT_V_R8 => P::R8Unorm,
        c::SET_COLOR_TARGET_FORMAT_V_RN8 => P::R8Snorm,
        c::SET_COLOR_TARGET_FORMAT_V_RS8 => P::R8Sint,
        c::SET_COLOR_TARGET_FORMAT_V_RU8 => P::R8Uint,
        other => return Err(DrawError::TargetFormat { value: other }),
    }))
}

/// The host format of a depth target format value.
pub fn depth_format(value: u32) -> Result<PixelFormat, DrawError> {
    use PixelFormat as P;
    use clc797 as c;
    Ok(match value {
        c::SET_ZT_FORMAT_V_Z24S8 => P::S8UintD24Unorm,
        c::SET_ZT_FORMAT_V_S8Z24 => P::D24UnormS8Uint,
        c::SET_ZT_FORMAT_V_ZF32 => P::D32Float,
        c::SET_ZT_FORMAT_V_Z16 => P::D16Unorm,
        c::SET_ZT_FORMAT_V_S8 => P::S8Uint,
        c::SET_ZT_FORMAT_V_ZF32_X24S8 => P::D32FloatS8Uint,
        c::SET_ZT_FORMAT_V_X8Z24 => P::X8D24Unorm,
        other => return Err(DrawError::TargetFormat { value: other }),
    })
}

/// The sample grid of `SET_ANTI_ALIAS`'s mode (Eden's `NumSamples`; a virtual-coverage mode
/// takes its plain grid, as the texture header decode does).
pub fn sample_layout(mode: u32) -> Result<SampleLayout, DrawError> {
    use clc797 as c;
    Ok(match mode {
        c::SET_ANTI_ALIAS_SAMPLES_MODE_1X1 => SampleLayout::S1x1,
        c::SET_ANTI_ALIAS_SAMPLES_MODE_2X1 => SampleLayout::S2x1,
        c::SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D => SampleLayout::S2x1D3d,
        c::SET_ANTI_ALIAS_SAMPLES_MODE_2X2
        | c::SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_4
        | c::SET_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12 => SampleLayout::S2x2,
        c::SET_ANTI_ALIAS_SAMPLES_MODE_4X2
        | c::SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_8
        | c::SET_ANTI_ALIAS_SAMPLES_MODE_4X2_VC_24 => SampleLayout::S4x2,
        c::SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D => SampleLayout::S4x2D3d,
        c::SET_ANTI_ALIAS_SAMPLES_MODE_4X4 => SampleLayout::S4x4,
        other => return Err(DrawError::SampleMode { mode: other }),
    })
}

/// Log2 of a block size in GOBs from the `ONE_GOB` to `THIRTYTWO_GOBS` values every block field
/// shares (0 to 5).
fn block_log2(value: u32) -> u8 {
    // The getters accept only the six named values, 0 to 5.
    u8::try_from(value).unwrap_or(0)
}

/// The colour targets, depth target and formats of the next draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Targets {
    /// The image at each output, by output index.
    pub colors: [Option<ImageInfo>; GUEST_COLOR_TARGETS],
    /// The depth target's image.
    pub depth: Option<ImageInfo>,
    /// The formats and sample count the pipeline renders to.
    pub formats: TargetFormats,
}

/// The targets of the next draw (module docs).
pub fn targets(mirror: &Mirror) -> Result<Targets, DrawError> {
    let three_d = mirror.three_d()?;
    let samples = sample_layout(
        mirror
            .reg(&clc797::SET_ANTI_ALIAS, &[])?
            .field(clc797::SET_ANTI_ALIAS_SAMPLES)?,
    )?;
    let select = three_d.ct_select();
    let count = select.target_count()?.min(COLOR_TARGETS);
    let mut colors = [None; GUEST_COLOR_TARGETS];
    let mut formats = [None; GUEST_COLOR_TARGETS];
    for (output, (color, format)) in (0..count).zip(colors.iter_mut().zip(formats.iter_mut())) {
        let index = select.target(output)?;
        if let Some(info) = color_target(three_d, index, samples)? {
            *format = Some(info.format);
            *color = Some(info);
        }
    }
    let depth = if three_d.zt().target_count()? == 0 {
        None
    } else {
        Some(depth_target(three_d, samples)?)
    };
    Ok(Targets {
        colors,
        depth,
        formats: TargetFormats {
            colors: formats,
            depth_stencil: depth.map(|info| info.format),
            sample_count: samples.samples(),
        },
    })
}

/// Colour target `index`'s image, or `None` when its format is `DISABLED`.
pub(crate) fn color_target(
    three_d: ThreeD<'_>,
    index: u32,
    samples: SampleLayout,
) -> Result<Option<ImageInfo>, DrawError> {
    let ct = three_d.color_target(index)?;
    let Some(format) = color_format(ct.format()?.value)? else {
        return Ok(None);
    };
    let base = ImageInfo {
        address: ct.address()?,
        format,
        dim: ImageDim::D2,
        width: ct.width()?,
        height: ct.height()?,
        depth: 1,
        layers: 1,
        levels: 1,
        samples,
        layout: MemoryLayout::Pitch { pitch_b: 0 },
    };
    if ct.layout()?.value == clc797::SET_COLOR_TARGET_MEMORY_LAYOUT_PITCH {
        let pitch_b = base.width;
        return Ok(Some(ImageInfo {
            width: pitch_b / format.bytes_per_block().max(1),
            layout: MemoryLayout::Pitch { pitch_b },
            samples: SampleLayout::S1x1,
            ..base
        }));
    }
    let third = ct.third_dimension()?.max(1);
    let is_3d = ct.third_dimension_control()?.value
        == clc797::SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_DEPTH_SIZE;
    Ok(Some(ImageInfo {
        dim: if is_3d { ImageDim::D3 } else { ImageDim::D2 },
        depth: if is_3d { third } else { 1 },
        layers: if is_3d { 1 } else { third },
        layout: MemoryLayout::BlockLinear {
            gob: gob_for(format),
            block_height_log2: block_log2(ct.block_height()?.value),
            block_depth_log2: block_log2(ct.block_depth()?.value),
        },
        ..base
    }))
}

/// The depth target's image.
pub(crate) fn depth_target(
    three_d: ThreeD<'_>,
    samples: SampleLayout,
) -> Result<ImageInfo, DrawError> {
    let zt = three_d.zt();
    let format = depth_format(zt.format()?.value)?;
    let width = zt.width()?;
    let layers = if zt.size_control()?.value == clc797::SET_ZT_SIZE_C_CONTROL_ARRAY_SIZE_IS_ONE {
        1
    } else {
        zt.third_dimension()?.max(1)
    };
    Ok(ImageInfo {
        address: zt.address()?,
        format,
        dim: ImageDim::D2,
        width,
        height: zt.height()?,
        depth: 1,
        layers,
        levels: 1,
        samples,
        layout: MemoryLayout::BlockLinear {
            gob: gob_for(format),
            block_height_log2: block_log2(zt.block_height()?.value),
            block_depth_log2: block_log2(zt.block_depth()?.value),
        },
    })
}
