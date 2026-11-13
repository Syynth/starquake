//! `emulation()`: which guest fixed-function features the shaders must emulate because the host
//! lacks them (docs/specs/render-backend.md, "Fixed-function state the host lacks"; slice R2).
//!
//! A pure function of [`Caps`] and the guest state in [`EmulationState`]. It is the one place
//! the list is decided: the GPU crate copies the [`EmulationFlags`] it returns into the shader
//! variant key and into [`crate::GraphicsPipelineDesc::emulation`], and the shader crate only
//! reads the copied fields (review guide, M13).
//!
//! # The table
//!
//! One flag per row of the spec's table that is emulated in shaders or indices. Eden paths are
//! under `eden/src/` @ cf654052 (GPL-3.0-or-later); the conditions are ported from the checks
//! cited.
//!
//! | Flag | Set when | Reference |
//! |---|---|---|
//! | [`EmulationFlags::ALPHA_TEST`] | the alpha test is on with a comparison other than "always", on every host | wgpu, Vulkan and Metal have no alpha test. Eden: `video_core/renderer_vulkan/fixed_pipeline_state.cpp` stores "always" when the test is off, and `shader_recompiler/backend/spirv/emit_spirv_special.cpp`, `AlphaTest`, emits nothing for "always" |
//! | [`EmulationFlags::DEPTH_RANGE_REWRITE`] | clip-space depth is -1..1 and the device lacks [`Features::DEPTH_RANGE_MINUS_ONE_TO_ONE`] | Eden: `emit_spirv_special.cpp`, `convert_depth_mode && !support_native_ndc`, where `support_native_ndc` is `IsExtDepthClipControlSupported()` (`video_core/renderer_vulkan/vk_pipeline_cache.cpp`) |
//! | [`EmulationFlags::PRIMITIVE_RESTART_REWRITE`] | primitive restart is off and the device lacks [`Features::PRIMITIVE_RESTART_OFF`] | Eden: `video_core/renderer_vulkan/vk_graphics_pipeline.cpp`, "MoltenVK/Metal always has primitive restart enabled and cannot disable it"; wgpu restarts every indexed strip (the spec's decision 1 table) |
//! | [`EmulationFlags::ATTRIBUTE_FETCH`] | the guest uses more vertex attributes than [`Limits::max_vertex_attributes`] | Eden: `video_core/vulkan_common/vulkan_device.cpp` caps MoltenVK at 16 ("MVK driver breaks when using more than 16 vertex attributes/bindings"); the guest has 32 |
//! | [`EmulationFlags::TYPED_OUTPUTS`] | a colour target has an integer format and the device lacks [`Features::UNTYPED_FRAGMENT_OUTPUTS`] | Eden: `vk_pipeline_cache.cpp` fills `color_output_types` from the target formats on MoltenVK only |
//!
//! Rows of the spec's table that are not flags, and so not decided here: quads and line loops
//! (the backend rewrites indices), logic operations and geometry shaders (the draw is skipped or
//! drawn without, and counted; [`Features::LOGIC_OP`] and [`crate::caps::StageSupport`] say
//! whether the device has them), the per-texture swizzle ([`Caps::view_swizzle`]) and transform
//! feedback (deferred).
//!
//! # Where this differs from Eden
//!
//! - Eden does not rewrite indices for primitive restart: at the line cited it turns restart on
//!   for MoltenVK whatever the guest asks, and its Vulkan renderer has no rewrite. The flag
//!   follows the spec's row instead.
//! - Eden has no attribute fetch from a buffer: it only lowers MoltenVK's attribute limit to 16.
//!   The flag follows the spec's row.
//! - Eden types every colour output on MoltenVK, float included. A float output is what a
//!   shader has without the flag, so only an integer target sets it.
//!
//! # What the primitive restart flag does not look at
//!
//! Whether a draw can meet the restart value at all (only an indexed draw can, and under wgpu
//! only a strip) is for whoever rewrites the indices: the pipeline's topology and the draw's
//! index buffer are in front of it. The flag says only that the host cannot be told to ignore
//! the value.
//!
//! [`Limits::max_vertex_attributes`]: crate::caps::Limits::max_vertex_attributes

use crate::caps::{Caps, Features, GUEST_COLOR_TARGETS};
use crate::format::PixelFormat;
use crate::pipeline::EmulationFlags;
use crate::resource::CompareFunction;

/// The range of clip-space depth the guest's vertex shaders write (the guest's `DepthMode`,
/// `eden/src/video_core/engines/maxwell_3d.h` @ cf654052).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DepthRange {
    /// 0 at the near plane to 1 at the far plane: what wgpu, Vulkan and Metal use.
    ZeroToOne,
    /// -1 at the near plane to 1 at the far plane: OpenGL's convention.
    MinusOneToOne,
}

/// The guest state [`emulation`] reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EmulationState {
    /// The alpha test's comparison, or `None` when the test is off.
    pub alpha_test: Option<CompareFunction>,
    /// The clip-space depth range.
    pub depth_range: DepthRange,
    /// Whether an index equal to the restart value starts a new primitive.
    pub primitive_restart: bool,
    /// How many vertex attributes the guest's vertex layout uses, before any are moved to a
    /// storage buffer.
    pub vertex_attributes: u32,
    /// The colour targets' formats, by the guest's target index.
    pub color_targets: [Option<PixelFormat>; GUEST_COLOR_TARGETS],
}

/// What the shaders (and, for primitive restart, the indices) must emulate for `state` on a
/// device with `caps` (module docs).
pub fn emulation(caps: &Caps, state: &EmulationState) -> EmulationFlags {
    let lacks = |feature| !caps.features.contains(feature);
    let alpha_test = state
        .alpha_test
        .is_some_and(|comparison| comparison != CompareFunction::Always);
    let depth_range_rewrite = state.depth_range == DepthRange::MinusOneToOne
        && lacks(Features::DEPTH_RANGE_MINUS_ONE_TO_ONE);
    let primitive_restart_rewrite =
        !state.primitive_restart && lacks(Features::PRIMITIVE_RESTART_OFF);
    let attribute_fetch = state.vertex_attributes > caps.limits.max_vertex_attributes;
    let typed_outputs = state
        .color_targets
        .iter()
        .flatten()
        .any(|format| format.is_integer())
        && lacks(Features::UNTYPED_FRAGMENT_OUTPUTS);

    [
        (alpha_test, EmulationFlags::ALPHA_TEST),
        (depth_range_rewrite, EmulationFlags::DEPTH_RANGE_REWRITE),
        (
            primitive_restart_rewrite,
            EmulationFlags::PRIMITIVE_RESTART_REWRITE,
        ),
        (attribute_fetch, EmulationFlags::ATTRIBUTE_FETCH),
        (typed_outputs, EmulationFlags::TYPED_OUTPUTS),
    ]
    .into_iter()
    .filter(|(needed, _)| *needed)
    .fold(EmulationFlags::EMPTY, |flags, (_, flag)| flags | flag)
}

#[cfg(test)]
mod tests;
