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
//! | [`EmulationFlags::FLIPPED_VIEWPORT`] | the draw's first viewport flips y or reverses depth ([`ViewportFlip`]) and the device lacks [`Features::FLIPPED_VIEWPORT`] | Not from Eden, which hands Vulkan the flipped viewport as it is (`GetViewportState`, `video_core/renderer_vulkan/vk_rasterizer.cpp`); wgpu-core 30.0.2's `set_viewport` refuses it, so the vertex shader flips instead (render-backend's table; decided by the owner, 2026-11-19) |
//!
//! Rows of the spec's table that are not flags, and so not decided here: quads and line loops
//! (the backend rewrites indices), logic operations and geometry shaders (the draw is skipped or
//! drawn without, and counted; [`Features::LOGIC_OP`] and [`crate::caps::StageSupport`] say
//! whether the device has them), the per-texture swizzle ([`Caps::view_swizzle`]) and transform
//! feedback (deferred).
//!
//! # Where this differs from Eden
//!
//! Each names the test in `emulation::tests` that pins it.
//!
//! - Eden does not rewrite indices for primitive restart: at the line cited it turns restart on
//!   for MoltenVK whatever the guest asks, and its Vulkan renderer has no rewrite. The flag
//!   follows the spec's row instead. `each_flag_follows_its_rows` (the restart rows).
//! - Eden has no attribute fetch from a buffer: it only lowers MoltenVK's attribute limit to 16.
//!   The flag follows the spec's row. `each_flag_follows_its_rows` (the attribute rows, at each
//!   limit and one past it).
//! - Eden types every colour output on MoltenVK, float included. A float output is what a
//!   shader has without the flag, so only an integer target sets it.
//!   `each_flag_follows_its_rows` (the output rows), `an_integer_target_in_any_slot_is_seen`.
//!
//! # The flipped viewport
//!
//! A guest viewport whose height is negative, or whose depth range is reversed (the near
//! plane's depth above the far plane's), is one wgpu refuses. With
//! [`EmulationFlags::FLIPPED_VIEWPORT`] the pipeline's vertex shader and the backend share the
//! work so the draw covers the same pixels and, for a depth range inside 0..=1, the same depths
//! (the exact condition is below):
//!
//! - **The backend** sets [`host_viewport`]: a negative height `h` at `y` becomes `-h` at
//!   `y + h` (the same rows), and a reversed range is swapped.
//! - **The vertex shader** (shader-pipeline slice 17 lowers it) negates clip-space y when
//!   [`ViewportFlip::flip_y`] is set, and mirrors clip-space z when
//!   [`ViewportFlip::reverse_depth`] is set. The mirror depends on the range clip-space z covers
//!   once the shader is done with it ([`clip_depth_range`]): `w - z` when that is 0..w (the
//!   guest's 0..1 depth mode, or its -1..1 mode after the [`EmulationFlags::DEPTH_RANGE_REWRITE`]
//!   rewrite, which yields a 0..w z), and `-z` when it is -w..w (the guest's -1..1 depth mode
//!   on a device with [`Features::DEPTH_RANGE_MINUS_ONE_TO_ONE`], where no rewrite is asked
//!   for). The GPU crate carries the flips and the clip range in the driver constants.
//!
//! Window y is affine in clip-space y, `y + (1 + s * y_ndc) / 2 * h` for the host's sign `s`,
//! so `y + h` and `-h` with `y_ndc` negated give the same value. With a 0..w clip range, depth
//! is `near + z / w * (far - near)`, `far + (w - z) / w * (near - far)` is the same value, and
//! `0 <= w - z <= w` clips exactly where `0 <= z <= w` does. With a -w..w clip range, depth is
//! `near + (z / w + 1) / 2 * (far - near)`, `far + (-z / w + 1) / 2 * (near - far)` is the
//! same value, and `-w <= -z <= w` clips exactly where `-w <= z <= w` does; `w - z` there would
//! shift every depth and clip `z = -w`. Pinned by
//! `tests::a_flipped_viewport_lands_on_the_guests_pixels_and_depths` (both clip ranges) and
//! `tests::a_native_minus_one_to_one_range_is_mirrored_about_zero` (the device with the -1..1
//! feature and without [`Features::FLIPPED_VIEWPORT`]).
//!
//! Only the first viewport is read: a wgpu render pass has one, and the driver constants carry
//! one flip. A host with several viewports and without [`Features::FLIPPED_VIEWPORT`] would
//! need a flip per viewport; no such backend exists. A negative width (a negative x scale) is not
//! flipped: Eden hands one to Vulkan as it is, which allows only the height to be negative, and
//! the backend still refuses it.
//!
//! ## A depth outside 0..=1: an open owner question
//!
//! The approved spec's row (render-backend, "Fixed-function state the host lacks") reads: "Guest
//! y-flip (negative viewport height); reversed or out-of-range depth range | wgpu refuses a
//! negative viewport height and a depth range that is reversed or outside 0..1 | An
//! `EmulationFlags` field and a driver constant: the vertex shader flips y and remaps depth".
//! This module does not remap an out-of-range depth: the GPU crate clamps both depths to 0..=1
//! before the backend sees them, as Eden does without `VK_EXT_depth_range_unrestricted`
//! (`GetViewportState`, `eden/src/video_core/renderer_vulkan/vk_rasterizer.cpp`), so
//! [`ViewportFlip`] never sees one from the GPU crate and a backend still refuses one from
//! elsewhere. A clamp and a remap give different depths: the clamp compresses the part of the
//! range past 0..=1 onto its edge, where a remap would keep the guest's depths and clip what
//! falls outside. Which to keep is **open for the owner**, with two options: (1) keep Eden's
//! clamp, and narrow the spec row to the reversed range; or (2) remap in the vertex shader (a
//! z scale and offset from the guest's range, which the driver constants' position words leave
//! room for) and clip there, as the row says. Until the owner rules, this is a departure from
//! the row, pinned by the backend's own test that a flipped pipeline still refuses a depth
//! outside 0..=1 (`starquake_render_wgpu`'s
//! `draw::tests::a_flipped_viewport_is_set_positive_only_for_a_flipped_pipeline`).
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
use crate::command::Viewport;
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
    /// How the draw's first viewport flips (module docs, "The flipped viewport").
    pub viewport: ViewportFlip,
}

/// How a guest viewport departs from one every host can set (module docs, "The flipped
/// viewport").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ViewportFlip {
    /// The height is negative.
    pub flip_y: bool,
    /// The depth range is reversed: the near plane's depth is above the far plane's.
    pub reverse_depth: bool,
}

impl ViewportFlip {
    /// No flip: what [`Default`] gives.
    pub const NONE: Self = Self {
        flip_y: false,
        reverse_depth: false,
    };

    /// How `viewport` flips. A height of `-0.0` or NaN, and depths that are equal or NaN, are
    /// not flips.
    pub fn of(viewport: &Viewport) -> Self {
        Self {
            flip_y: viewport.height < 0.0,
            reverse_depth: viewport.min_depth > viewport.max_depth,
        }
    }

    /// Whether anything is flipped.
    pub const fn any(self) -> bool {
        self.flip_y || self.reverse_depth
    }
}

/// The viewport a backend sets for `guest` when the pipeline has
/// [`EmulationFlags::FLIPPED_VIEWPORT`]: a negative height made positive over the same rows, and
/// a reversed depth range swapped; every other field as it is (module docs, "The flipped
/// viewport"). The arithmetic is `f32`'s, so a result past the host's range (a `y + height`
/// that overflows to infinity) is left for the backend's own viewport check to refuse.
pub fn host_viewport(guest: &Viewport) -> Viewport {
    let flip = ViewportFlip::of(guest);
    let mut host = *guest;
    if flip.flip_y {
        host.y = guest.y + guest.height;
        host.height = -guest.height;
    }
    if flip.reverse_depth {
        host.min_depth = guest.max_depth;
        host.max_depth = guest.min_depth;
    }
    host
}

/// The range clip-space z covers when the vertex shader hands the position on, for a guest
/// whose depth mode is `depth_range` under a pipeline with `flags`: the guest's own range, except
/// that [`EmulationFlags::DEPTH_RANGE_REWRITE`] rewrites -1..1 to 0..1. The depth mirror of
/// [`EmulationFlags::FLIPPED_VIEWPORT`] is `w - z` for 0..1 and `-z` for -1..1 (module docs,
/// "The flipped viewport").
pub fn clip_depth_range(depth_range: DepthRange, flags: EmulationFlags) -> DepthRange {
    if flags.contains(EmulationFlags::DEPTH_RANGE_REWRITE) {
        DepthRange::ZeroToOne
    } else {
        depth_range
    }
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
    let flipped_viewport = state.viewport.any() && lacks(Features::FLIPPED_VIEWPORT);

    [
        (alpha_test, EmulationFlags::ALPHA_TEST),
        (depth_range_rewrite, EmulationFlags::DEPTH_RANGE_REWRITE),
        (
            primitive_restart_rewrite,
            EmulationFlags::PRIMITIVE_RESTART_REWRITE,
        ),
        (attribute_fetch, EmulationFlags::ATTRIBUTE_FETCH),
        (typed_outputs, EmulationFlags::TYPED_OUTPUTS),
        (flipped_viewport, EmulationFlags::FLIPPED_VIEWPORT),
    ]
    .into_iter()
    .filter(|(needed, _)| *needed)
    .fold(EmulationFlags::EMPTY, |flags, (_, flag)| flags | flag)
}

#[cfg(test)]
mod tests;
