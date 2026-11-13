//! `Caps`: what the host device can do, fixed for the backend's lifetime.
//!
//! Each backend fills one from its device (the wgpu backend from the adapter's features and
//! limits, slice R5); the validation in [`crate::validate`] checks every description against it
//! before a host object exists (docs/specs/render-backend.md, "Errors and untrusted input"). The
//! GPU crate reads it too: [`Caps::view_swizzle`] decides whether a texture's swizzle goes into
//! the view or into the shader variant key, and [`Caps::dynamic_state`] decides what the pipeline
//! key leaves out (render-backend, "Commands" and "Textures, views, samplers, render targets").
//!
//! The guest GPU's own limits are constants here, so a description never exceeds the guest's
//! register file even on a device that allows more. Where a validation limit is "the smaller of
//! the device's and the guest's", the function that applies it says so.

use crate::flags::flag_set;
use crate::format::{PIXEL_FORMAT_COUNT, PixelFormat};

/// Colour render targets the guest's 3D engine has (`NumRenderTargets`,
/// `eden/src/video_core/engines/maxwell_3d.h` @ cf654052).
pub const GUEST_COLOR_TARGETS: usize = 8;

/// Vertex attributes the guest's 3D engine has (`NumVertexAttributes`, `maxwell_3d.h`
/// @ cf654052).
pub const GUEST_VERTEX_ATTRIBUTES: u32 = 32;

/// Vertex buffers (vertex streams) the guest's 3D engine has (`NumVertexArrays`, `maxwell_3d.h`
/// @ cf654052).
pub const GUEST_VERTEX_BUFFERS: u32 = 32;

/// Viewports (and scissors) the guest's 3D engine has (`NumViewports`, `maxwell_3d.h`
/// @ cf654052).
pub const GUEST_VIEWPORTS: u32 = 16;

/// Numeric limits of the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Largest buffer, in bytes.
    pub max_buffer_size: u64,
    /// Largest staging slice one `staging` call may ask for, in bytes. A backend sets it from its
    /// memory budget; it bounds what a hostile guest can make the host allocate at once.
    pub max_staging_size: u64,
    /// Largest width of a 1D texture.
    pub max_texture_dimension_1d: u32,
    /// Largest width and height of a 2D or cube texture.
    pub max_texture_dimension_2d: u32,
    /// Largest width, height and depth of a 3D texture.
    pub max_texture_dimension_3d: u32,
    /// Most layers of an array texture (for a cube array, six per cube).
    pub max_texture_array_layers: u32,
    /// Largest sampler anisotropy (1 means none).
    pub max_sampler_anisotropy: u32,
    /// Most vertex attributes in a pipeline.
    pub max_vertex_attributes: u32,
    /// Most vertex buffers in a pipeline or a draw.
    pub max_vertex_buffers: u32,
    /// Largest vertex buffer stride, in bytes.
    pub max_vertex_buffer_stride: u32,
    /// Most colour attachments bound at once.
    pub max_color_attachments: u32,
    /// Most viewports (and scissors) in a draw.
    pub max_viewports: u32,
    /// Most bind groups (descriptor sets) a pipeline uses.
    pub max_bind_groups: u32,
    /// Most bindings in one bind group (largest binding number, plus one).
    pub max_bindings_per_bind_group: u32,
    /// Most workgroups of a dispatch in each dimension.
    pub max_compute_workgroups_per_dimension: u32,
    /// Most uniform buffers one shader uses (wgpu-types 30.0.1 `limits.rs`,
    /// `max_uniform_buffers_per_shader_stage`, 12 by default). The guest has 18 constant
    /// buffers per stage (`MaxConstBuffers`, `maxwell_3d.h` @ cf654052), more than wgpu's
    /// default, so the shader crate's mapping of them meets this limit.
    pub max_uniform_buffers_per_stage: u32,
    /// Most storage buffers one shader uses, read-only or writable
    /// (`max_storage_buffers_per_shader_stage`, 8 by default in wgpu).
    pub max_storage_buffers_per_stage: u32,
    /// Most sampled textures one shader uses (`max_sampled_textures_per_shader_stage`, 16 by
    /// default in wgpu).
    pub max_sampled_textures_per_stage: u32,
    /// Most storage textures one shader uses (`max_storage_textures_per_shader_stage`, 4 by
    /// default in wgpu).
    pub max_storage_textures_per_stage: u32,
    /// Most samplers one shader uses, comparison samplers included
    /// (`max_samplers_per_shader_stage`, 16 by default in wgpu).
    pub max_samplers_per_stage: u32,
}

flag_set! {
    /// What a device can do with one [`PixelFormat`] natively (render-backend, "Formats").
    FormatFeatures {
        /// Sampled in a shader.
        SAMPLE = 0,
        /// Sampled with linear filtering.
        FILTER = 1,
        /// Used as a render target (colour, or depth/stencil).
        RENDER = 2,
        /// Blended into as a colour target.
        BLEND = 3,
        /// Used as a storage image.
        STORAGE = 4,
        /// Copied to and from, uploaded and read back.
        COPY = 5,
    }
}

flag_set! {
    /// Optional device features the validation checks.
    Features {
        /// The clamp-to-border sampler address mode.
        CLAMP_TO_BORDER = 0,
        /// Drawing polygons as lines.
        POLYGON_MODE_LINE = 1,
        /// Drawing polygons as points.
        POLYGON_MODE_POINT = 2,
        /// Depth clamping instead of clipping.
        DEPTH_CLAMP = 3,
        /// Colour logic operations.
        LOGIC_OP = 4,
        /// Blend factors that read a second fragment output.
        DUAL_SOURCE_BLEND = 5,
        /// Indirect draws that read more than one draw's parameters.
        MULTI_DRAW_INDIRECT = 6,
        /// Out-of-bounds buffer accesses are defined (read zero, writes dropped). Not checked
        /// here: it is passed to the shader crate, which picks bounds checks from it
        /// (render-backend, "Errors and untrusted input").
        ROBUST_BUFFER_ACCESS = 7,
        /// Out-of-bounds image accesses are defined. Passed on like `ROBUST_BUFFER_ACCESS`.
        ROBUST_IMAGE_ACCESS = 8,
        /// The mirror-once, clamp-to-edge sampler address mode.
        MIRROR_CLAMP_TO_EDGE = 9,
        /// Samplers that return the minimum or maximum of the texels a filter reads instead of
        /// their weighted average (Vulkan's `samplerFilterMinmax`). wgpu 30 has no sampler
        /// reduction mode, so a wgpu device leaves this out.
        SAMPLER_MIN_MAX_REDUCTION = 10,
        /// A clip-space depth range of -1..1 as pipeline state (Vulkan's
        /// `VK_EXT_depth_clip_control`). wgpu and Metal have no such switch, so a wgpu device
        /// leaves this out, and [`crate::emulation::emulation`] then asks for the vertex
        /// shader's rewrite.
        DEPTH_RANGE_MINUS_ONE_TO_ONE = 11,
        /// Primitive restart can be turned off: an index equal to the restart value is drawn
        /// as a vertex. wgpu restarts every indexed strip, and Metal and MoltenVK always
        /// restart, so those devices leave this out.
        PRIMITIVE_RESTART_OFF = 12,
        /// A fragment output's type need not match its colour target's: a float output may
        /// feed an integer target. MoltenVK needs them to match; that wgpu's pipeline
        /// validation does too is a guess, settled in slice R7, so until then a wgpu device
        /// leaves this out.
        UNTYPED_FRAGMENT_OUTPUTS = 13,
    }
}

flag_set! {
    /// Optional shader stages the device can run. Vertex, fragment and compute always can.
    StageSupport {
        /// Geometry shaders.
        GEOMETRY = 0,
        /// Tessellation control and evaluation shaders.
        TESSELLATION = 1,
    }
}

flag_set! {
    /// Draw state the device sets per draw rather than per pipeline. The GPU crate keys
    /// pipelines only on what is not here (render-backend, "Commands"). Under wgpu only the
    /// viewport, scissor, blend constant and stencil reference are dynamic.
    DynamicState {
        /// Viewports.
        VIEWPORT = 0,
        /// Scissor rectangles.
        SCISSOR = 1,
        /// The blend constant colour.
        BLEND_CONSTANT = 2,
        /// The stencil reference value.
        STENCIL_REFERENCE = 3,
        /// The stencil compare and write masks.
        STENCIL_MASKS = 4,
        /// Depth bias.
        DEPTH_BIAS = 5,
        /// Line width.
        LINE_WIDTH = 6,
    }
}

flag_set! {
    /// Sample counts the device supports for textures and render targets.
    SampleCounts {
        /// One sample (always supported by a real device).
        X1 = 0,
        /// Two samples.
        X2 = 1,
        /// Four samples.
        X4 = 2,
        /// Eight samples.
        X8 = 3,
        /// Sixteen samples.
        X16 = 4,
    }
}

impl FormatFeatures {
    /// What an error calls the feature ([`crate::RenderError::FormatUnsupported`]'s `usage`).
    /// A set that is not exactly one feature has no name of its own.
    pub(crate) const fn in_words(self) -> &'static str {
        match self {
            Self::SAMPLE => "sampling",
            Self::FILTER => "filtering",
            Self::RENDER => "render targets",
            Self::BLEND => "blending",
            Self::STORAGE => "storage images",
            Self::COPY => "copies",
            _ => "several uses",
        }
    }
}

impl SampleCounts {
    /// The flag for `count` samples, or `None` when `count` is not 1, 2, 4, 8 or 16.
    pub const fn from_count(count: u32) -> Option<Self> {
        match count {
            1 => Some(Self::X1),
            2 => Some(Self::X2),
            4 => Some(Self::X4),
            8 => Some(Self::X8),
            16 => Some(Self::X16),
            _ => None,
        }
    }

    /// Whether `count` samples is in the set.
    pub const fn supports(self, count: u32) -> bool {
        match Self::from_count(count) {
            Some(flag) => self.contains(flag),
            None => false,
        }
    }
}

/// What the device can do with each [`PixelFormat`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatTable {
    features: [FormatFeatures; PIXEL_FORMAT_COUNT],
}

impl FormatTable {
    /// A table giving every format the same features.
    pub const fn uniform(features: FormatFeatures) -> Self {
        Self {
            features: [features; PIXEL_FORMAT_COUNT],
        }
    }

    /// What the device can do with `format`.
    pub const fn get(&self, format: PixelFormat) -> FormatFeatures {
        self.features[format.index()]
    }

    /// Records what the device can do with `format`.
    pub const fn set(&mut self, format: PixelFormat, features: FormatFeatures) {
        self.features[format.index()] = features;
    }
}

/// What the host device can do. Fixed for the backend's lifetime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Caps {
    /// Numeric limits.
    pub limits: Limits,
    /// Optional features.
    pub features: Features,
    /// Optional shader stages.
    pub stages: StageSupport,
    /// Draw state set per draw.
    pub dynamic_state: DynamicState,
    /// Supported sample counts.
    pub sample_counts: SampleCounts,
    /// Whether a view can remap a texture's components. `false` under wgpu: the backend then
    /// refuses a non-identity swizzle, and the GPU crate folds each binding's swizzle into the
    /// shader variant key instead (render-backend, "Textures, views, samplers, render targets").
    pub view_swizzle: bool,
    /// Per-format support.
    pub formats: FormatTable,
}

#[cfg(test)]
mod tests {
    use super::{FormatFeatures, FormatTable, SampleCounts};
    use crate::format::PixelFormat;

    #[test]
    fn sample_count_flags() {
        for (count, flag) in [
            (1, SampleCounts::X1),
            (2, SampleCounts::X2),
            (4, SampleCounts::X4),
            (8, SampleCounts::X8),
            (16, SampleCounts::X16),
        ] {
            assert_eq!(SampleCounts::from_count(count), Some(flag));
            assert!(flag.supports(count));
            assert!(!SampleCounts::ALL.difference(flag).supports(count));
        }
        for count in [0, 3, 5, 6, 7, 9, 15, 17, 32, u32::MAX] {
            assert_eq!(SampleCounts::from_count(count), None, "{count}");
            assert!(!SampleCounts::ALL.supports(count));
        }
    }

    #[test]
    fn format_table_set_and_get() {
        let mut table = FormatTable::uniform(FormatFeatures::SAMPLE);
        table.set(PixelFormat::R8Unorm, FormatFeatures::ALL);
        assert_eq!(table.get(PixelFormat::R8Unorm), FormatFeatures::ALL);
        assert_eq!(table.get(PixelFormat::R8Snorm), FormatFeatures::SAMPLE);
        assert_eq!(
            table.get(PixelFormat::D32FloatS8Uint),
            FormatFeatures::SAMPLE
        );
    }
}
