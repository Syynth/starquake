//! `Caps` from what the wgpu adapter reports (docs/specs/render-backend.md, slice R5 and
//! verification 1).
//!
//! [`device_features`] picks the optional wgpu features the backend asks the device for (those
//! of [`WANTED_FEATURES`] the adapter has), and [`caps`] turns the device's features, limits and
//! per-format support into the backend-neutral [`Caps`]. Both are pure, so they are tested with
//! made-up adapters; the backend calls them with the real one.
//!
//! What each part of `Caps` comes from:
//!
//! - **Limits**: the wgpu limit of the same meaning, except three. `max_buffer_size` is wgpu's
//!   rounded down to a multiple of [`COPY_BUFFER_ALIGNMENT`] (host buffers are made that much
//!   larger, [`crate::rules::host_buffer_size`]) and capped at the backend's memory limit (a
//!   buffer the budget cannot hold is refused before anything is allocated; this backend's
//!   choice). `max_staging_size` is the configured staging limit, capped at `max_buffer_size`.
//!   `max_viewports` is 1: a wgpu render pass has one viewport (`RenderPass::set_viewport`).
//!   `max_sampler_anisotropy` is 16, the most wgpu-core passes on (`Device::create_sampler`
//!   clamps `anisotropy_clamp` to 16, "per the wgpu-hal interface", wgpu-core 30.0.2
//!   `src/device/resource.rs`).
//! - **Features**: clamp-to-border, polygon modes line and point, depth clamping
//!   (`DEPTH_CLIP_CONTROL`, which gives `unclipped_depth`) and dual-source blending where the
//!   device has the wgpu feature; multi-draw indirect where the device has indirect execution
//!   (`DownlevelFlags::INDIRECT_EXECUTION`, which `RenderPass::multi_draw_indirect` needs; wgpu 30
//!   has no feature flag for it). Never: logic ops, mirror-clamp-to-edge, min/max sampler
//!   reduction, a −1..1 depth range, turning primitive restart off and untyped fragment outputs
//!   (none is in wgpu 30; render-backend decision 1's table), and the two robust-access features
//!   (wgpu bounds-checks shaders itself, but does not promise "read zero, drop writes"; claiming
//!   less is the safe side).
//! - **Stages**: none beyond vertex, fragment and compute (wgpu has no geometry or tessellation
//!   stage).
//! - **Dynamic state**: viewport, scissor, blend constant and stencil reference, the four wgpu
//!   sets per pass (render-backend, "Commands").
//! - **Sample counts**: one sample, and the counts the device supports for `A8B8G8R8_UNORM`
//!   (wgpu's `Rgba8Unorm`, a format every device can render). wgpu decides sample counts per
//!   format, so each texture's own format is checked too ([`crate::rules`]).
//! - **View swizzle**: `false`; wgpu has none (render-backend, "Textures, views, samplers,
//!   render targets").
//! - **Formats**: for a guest format with a wgpu form ([`crate::format::texture_format`]) whose
//!   required features the device has, what wgpu reports for that form ([`format_features`]);
//!   nothing for every other format.
//!
//! # Where `Caps` claims more than the backend does
//!
//! [`FormatFeatures::COPY`] reads "copied to and from, uploaded and read back", and is reported
//! wherever wgpu allows both copy usages, which it does for every depth and stencil format. But
//! wgpu copies one aspect at a time, and does not copy into `Depth32Float` from a buffer, so
//! the backend moves depth and stencil texels to and from buffers only where wgpu can
//! (`crate::rules::check_buffer_texture`): `D16_UNORM` and `S8_UINT` both ways, `D32_FLOAT` out
//! only (texture-to-buffer copies and readbacks), each covering the level's width and height.
//! These keep `COPY` while the backend refuses:
//!
//! - an upload or buffer-to-texture copy into `D32_FLOAT`;
//! - every upload, buffer copy and readback of `D32_FLOAT_S8_UINT`, which is also the format the
//!   shared fallback picks for `D24_UNORM_S8_UINT` (with a texel conversion where it crosses
//!   guest memory), so the spec's "converted on upload and readback" fails at `submit` for it;
//! - a copy between a guest buffer and a depth or stencil texture of several rows that are not
//!   a multiple of 256 bytes, and one covering part of the level's width or height.
//!
//! Moving two-aspect texels needs the copy to name an aspect, which the backend-neutral copy
//! commands do not, or a conversion pass the backend does not have yet. Whether `Caps` should
//! drop `COPY` for these formats, or the commands gain an aspect, is an open question for the
//! owner; until then each refusal is a named `RenderError::Unsupported` from `submit`
//! (`rules::tests::depth_and_stencil_buffer_copies_follow_wgpu_in_order`).

use crate::format::texture_format;
use starquake_render::caps::{DynamicState, StageSupport};
use starquake_render::{
    Caps, Features, FormatFeatures, FormatTable, Limits, PixelFormat, SampleCounts,
};
use wgpu::{TextureFormatFeatureFlags, TextureUsages};

/// wgpu's alignment for buffer copies and buffer writes, in bytes (`wgpu::COPY_BUFFER_ALIGNMENT`).
pub const COPY_BUFFER_ALIGNMENT: u64 = wgpu::COPY_BUFFER_ALIGNMENT;

/// The most anisotropy wgpu-core passes on to a sampler (module docs).
pub const WGPU_MAX_ANISOTROPY: u32 = 16;

/// Viewports a wgpu render pass has (module docs).
pub const WGPU_VIEWPORTS: u32 = 1;

/// The optional wgpu features the backend uses when the adapter has them: the texture
/// compression families, 16-bit normalised formats, `D32_FLOAT_S8_UINT`, compressed 3D
/// textures, the adapter's own format support (not only WebGPU's guaranteed set), the
/// features [`caps`] reports, and the shader features guest shaders may need (slice R6):
/// subgroups, f16 and i64 arithmetic, and binding arrays of textures with non-uniform indexing
/// (`docs/specs/shader-pipeline.md`, the E1 table). wgpu validates each module against the
/// device's features, so a shader that needs one the device lacks fails its pipeline.
pub const WANTED_FEATURES: wgpu::Features = wgpu::Features::TEXTURE_COMPRESSION_BC
    .union(wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D)
    .union(wgpu::Features::TEXTURE_COMPRESSION_ETC2)
    .union(wgpu::Features::TEXTURE_COMPRESSION_ASTC)
    .union(wgpu::Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D)
    .union(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
    .union(wgpu::Features::DEPTH32FLOAT_STENCIL8)
    .union(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
    .union(wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER)
    .union(wgpu::Features::POLYGON_MODE_LINE)
    .union(wgpu::Features::POLYGON_MODE_POINT)
    .union(wgpu::Features::DEPTH_CLIP_CONTROL)
    .union(wgpu::Features::DUAL_SOURCE_BLENDING)
    .union(wgpu::Features::SUBGROUP)
    .union(wgpu::Features::SHADER_F16)
    .union(wgpu::Features::SHADER_INT64)
    .union(wgpu::Features::TEXTURE_BINDING_ARRAY)
    .union(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

/// The features to ask the device for: those of [`WANTED_FEATURES`] the adapter has.
pub fn device_features(adapter: wgpu::Features) -> wgpu::Features {
    adapter & WANTED_FEATURES
}

/// Byte limits the backend is configured with, which bound what `Caps` allows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteLimits {
    /// The memory the backend may use (`budget().allowed`); no single buffer may be larger.
    pub memory: u64,
    /// The largest staging slice one call may ask for.
    pub staging: u64,
}

/// What a device can do with a format, in this crate's words: sampled, filtered, rendered to,
/// blended, used as a storage image, and copied (both copy usages).
pub fn format_features(features: &wgpu::TextureFormatFeatures) -> FormatFeatures {
    let usages = features.allowed_usages;
    let flags = features.flags;
    let mut out = FormatFeatures::EMPTY;
    for (has, feature) in [
        (
            usages.contains(TextureUsages::TEXTURE_BINDING),
            FormatFeatures::SAMPLE,
        ),
        (
            flags.contains(TextureFormatFeatureFlags::FILTERABLE),
            FormatFeatures::FILTER,
        ),
        (
            usages.contains(TextureUsages::RENDER_ATTACHMENT),
            FormatFeatures::RENDER,
        ),
        (
            flags.contains(TextureFormatFeatureFlags::BLENDABLE),
            FormatFeatures::BLEND,
        ),
        (
            usages.contains(TextureUsages::STORAGE_BINDING),
            FormatFeatures::STORAGE,
        ),
        (
            usages.contains(TextureUsages::COPY_SRC | TextureUsages::COPY_DST),
            FormatFeatures::COPY,
        ),
    ] {
        if has {
            out = out.union(feature);
        }
    }
    out
}

/// The sample counts wgpu's format flags allow: one, and each `MULTISAMPLE_X*` flag set.
pub fn sample_counts(flags: TextureFormatFeatureFlags) -> SampleCounts {
    let mut counts = SampleCounts::X1;
    for (flag, count) in [
        (TextureFormatFeatureFlags::MULTISAMPLE_X2, SampleCounts::X2),
        (TextureFormatFeatureFlags::MULTISAMPLE_X4, SampleCounts::X4),
        (TextureFormatFeatureFlags::MULTISAMPLE_X8, SampleCounts::X8),
        (
            TextureFormatFeatureFlags::MULTISAMPLE_X16,
            SampleCounts::X16,
        ),
    ] {
        if flags.contains(flag) {
            counts = counts.union(count);
        }
    }
    counts
}

/// What a device can do with each guest format: its [`FormatTable`] and, per format, the sample
/// counts and the wgpu usages it allows. A format with no wgpu form, or whose wgpu form needs a
/// feature the device lacks, can do nothing, has only one sample and allows no usage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatSupport {
    /// What each format can be used for.
    pub table: FormatTable,
    /// The sample counts of each format, by [`PixelFormat::index`].
    pub samples: Vec<SampleCounts>,
    /// The wgpu usages each format allows, by [`PixelFormat::index`].
    pub usages: Vec<TextureUsages>,
}

impl FormatSupport {
    /// Asks `describe` about the wgpu form of every guest format a device with
    /// `device_features` can use.
    pub fn new(
        device_features: wgpu::Features,
        describe: impl Fn(wgpu::TextureFormat) -> wgpu::TextureFormatFeatures,
    ) -> Self {
        let mut table = FormatTable::uniform(FormatFeatures::EMPTY);
        let mut samples = vec![SampleCounts::X1; PixelFormat::ALL.len()];
        let mut usages = vec![TextureUsages::empty(); PixelFormat::ALL.len()];
        for format in PixelFormat::ALL {
            let Some(host) = texture_format(format) else {
                continue;
            };
            if !device_features.contains(host.required_features()) {
                continue;
            }
            let described = describe(host);
            table.set(format, format_features(&described));
            if let Some(slot) = samples.get_mut(format.index()) {
                *slot = sample_counts(described.flags);
            }
            if let Some(slot) = usages.get_mut(format.index()) {
                *slot = described.allowed_usages;
            }
        }
        Self {
            table,
            samples,
            usages,
        }
    }

    /// The wgpu usages `format` allows.
    pub fn usages(&self, format: PixelFormat) -> TextureUsages {
        self.usages
            .get(format.index())
            .copied()
            .unwrap_or(TextureUsages::empty())
    }

    /// The sample counts `format` allows.
    pub fn samples(&self, format: PixelFormat) -> SampleCounts {
        self.samples
            .get(format.index())
            .copied()
            .unwrap_or(SampleCounts::X1)
    }
}

/// The `Caps` of a device with `features`, `limits` and `downlevel` flags and `formats`, for a
/// backend configured with `bytes` (module docs).
pub fn caps(
    features: wgpu::Features,
    limits: &wgpu::Limits,
    downlevel: wgpu::DownlevelFlags,
    formats: &FormatSupport,
    bytes: ByteLimits,
) -> Caps {
    let aligned_buffer = limits.max_buffer_size - limits.max_buffer_size % COPY_BUFFER_ALIGNMENT;
    let max_buffer_size = aligned_buffer.min(bytes.memory);
    let limits = Limits {
        max_buffer_size,
        max_staging_size: bytes.staging.min(max_buffer_size),
        max_texture_dimension_1d: limits.max_texture_dimension_1d,
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
        max_texture_dimension_3d: limits.max_texture_dimension_3d,
        max_texture_array_layers: limits.max_texture_array_layers,
        max_sampler_anisotropy: WGPU_MAX_ANISOTROPY,
        max_vertex_attributes: limits.max_vertex_attributes,
        max_vertex_buffers: limits.max_vertex_buffers,
        max_vertex_buffer_stride: limits.max_vertex_buffer_array_stride,
        max_color_attachments: limits.max_color_attachments,
        max_viewports: WGPU_VIEWPORTS,
        max_bind_groups: limits.max_bind_groups,
        max_bindings_per_bind_group: limits.max_bindings_per_bind_group,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        max_uniform_buffers_per_stage: limits.max_uniform_buffers_per_shader_stage,
        max_storage_buffers_per_stage: limits.max_storage_buffers_per_shader_stage,
        max_sampled_textures_per_stage: limits.max_sampled_textures_per_shader_stage,
        max_storage_textures_per_stage: limits.max_storage_textures_per_shader_stage,
        max_samplers_per_stage: limits.max_samplers_per_shader_stage,
    };
    let mut ours = Features::EMPTY;
    for (has, feature) in [
        (
            features.contains(wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER),
            Features::CLAMP_TO_BORDER,
        ),
        (
            features.contains(wgpu::Features::POLYGON_MODE_LINE),
            Features::POLYGON_MODE_LINE,
        ),
        (
            features.contains(wgpu::Features::POLYGON_MODE_POINT),
            Features::POLYGON_MODE_POINT,
        ),
        (
            features.contains(wgpu::Features::DEPTH_CLIP_CONTROL),
            Features::DEPTH_CLAMP,
        ),
        (
            features.contains(wgpu::Features::DUAL_SOURCE_BLENDING),
            Features::DUAL_SOURCE_BLEND,
        ),
        (
            downlevel.contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION),
            Features::MULTI_DRAW_INDIRECT,
        ),
    ] {
        if has {
            ours = ours.union(feature);
        }
    }
    Caps {
        limits,
        features: ours,
        stages: StageSupport::EMPTY,
        dynamic_state: DynamicState::VIEWPORT
            .union(DynamicState::SCISSOR)
            .union(DynamicState::BLEND_CONSTANT)
            .union(DynamicState::STENCIL_REFERENCE),
        sample_counts: formats.samples(PixelFormat::A8B8G8R8Unorm),
        view_swizzle: false,
        formats: formats.table.clone(),
    }
}

#[cfg(test)]
mod tests;
