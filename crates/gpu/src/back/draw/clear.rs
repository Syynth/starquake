//! Clears: the 3D engine's `CLEAR_SURFACE` turned into the render backend's
//! [`Command::Clear`], or into a draw for a clear a clear command cannot express (gpu-protocol
//! spec, "Clears" and slice G24; decided by the owner, 2026-11-19, ruling 9).
//!
//! The executor hands every word written to `CLEAR_SURFACE` to [`super::Draws::method`], which
//! clears through [`Clears`]. A clear reads the register file the way a draw does: the target
//! images through G18's decoders ([`super::targets`]), scissor 0
//! ([`super::fixed::guest_scissor`]), and the clear values, through G10's mirror
//! ([`Mirror::reg`]).
//!
//! # A clear, step by step
//!
//! This is Eden-NX2's `RasterizerVulkan::Clear`, which Eden's `DrawManager::Clear` calls once
//! `ShouldExecute` passes, with a layer count of 1. Steps 1 to 5 make its checks in its order;
//! the checks it does not make (a slice or layer past the target, a NaN depth value, a failed
//! pipeline) are listed under "Divergences from Eden-NX2" and "Guesses".
//!
//! 1. **Conditional render**: a clear while the guest disabled rendering does nothing
//!    ([`ClearSkip::RenderDisabled`]), as `DrawManager::Clear` tests `ShouldExecute` first.
//! 2. **What is asked** ([`ClearSurface`]): colour when any of `R_ENABLE` to `A_ENABLE` is set,
//!    depth for `Z_ENABLE`, stencil for `STENCIL_ENABLE` (Eden's `use_color`, `use_depth`,
//!    `use_stencil`). A word asking for none does nothing ([`Cleared::NOTHING`]).
//! 3. **The area**: with `SET_CLEAR_SURFACE_CONTROL`'s `USE_SCISSOR0` (Eden's
//!    `clear_control.use_scissor`), scissor 0 as Eden's `GetScissorState(regs, 0)` gives it;
//!    otherwise the whole target. Then it is clamped to the render area, as Eden's
//!    `clamp_rect_to_render_area`: the smallest width and height of colour targets 0 to
//!    `SET_CT_SELECT`'s count, not through its map, whose format is not `DISABLED`, and of the
//!    depth target when `SET_ZT_SELECT`'s count is not 0. Those are the attachments of Eden's
//!    framebuffer (`UpdateRenderTargets` fills it with `FindColorBuffer(index)` for each index,
//!    which gives no view at or past the count), whose `Framebuffer::RenderArea` is their minimum
//!    (`CreateFramebuffer` in `vk_texture_cache.cpp`); so the area never extends past the
//!    cleared colour target, which is one of them (`tests::the_render_area_is_over_the_unmapped_colour_targets`).
//!    Eden's area is also bounded by `SET_SURFACE_CLIP`, which is not read here (under
//!    "Divergences"). An empty area does nothing ([`ClearSkip::EmptyArea`]).
//! 4. **Colour** ([`ClearSurface::target`] is `MRT_SELECT`, Eden's `clear_surface.RT`): colour
//!    target `MRT_SELECT` itself, not through `SET_CT_SELECT`'s map, when it is below
//!    `SET_CT_SELECT`'s count and its format is not `DISABLED` (Eden's `FindColorBuffer(index)`
//!    and `HasAspectColorBit`); otherwise nothing ([`ClearSkip::NotBound`]). The value is
//!    `SET_COLOR_CLEAR_VALUE(0..3)` as floats, converted for an integer format by Eden's
//!    formula ([`color_value`]). All four channels: a [`Command::Clear`], unless the area is part
//!    of a multisampled target ("Partial clears", below); some: a draw ([`Part::Drawn`], below),
//!    as Eden's `blit_image.ClearColor`.
//! 5. **Depth and stencil**: the depth target G18 binds (`SET_ZT_SELECT`'s count not 0, Eden's
//!    `FindDepthBuffer`), the aspects asked that its format has (Eden's `HasAspectDepthBit`,
//!    `HasAspectStencilBit`; none is [`ClearSkip::NoAspect`]). Depth is `SET_Z_CLEAR_VALUE`,
//!    stencil `SET_STENCIL_CLEAR_VALUE`. A stencil clear whose `SET_STENCIL_MASK` is neither
//!    `0xff` nor 0 is a draw writing through that mask, as Eden's `blit_image.ClearDepthStencil`
//!    (with depth too when depth is asked); otherwise a [`Command::Clear`] when the area covers
//!    the depth target, and a draw when it is part of it ("Partial clears", below).
//!
//! Colour comes before depth and stencil in the list, as in Eden.
//!
//! The clear of one target and layer is [`Command::Clear`] of a view of that layer
//! ([`crate::back::texture::TextureCache::render_target_layer`]): the target's own view when
//! it has one layer, so clears and draws share it. The rectangle is `None` when the clamped area
//! covers the target.
//! Resolving a target marks it GPU-modified and uploads it first if guest memory is newer, as
//! for a draw (`back::texture`, "Uses and uploads").
//!
//! # Partial clears
//!
//! The wgpu render backend clears a depth, stencil or multisampled view only whole: a clear
//! with a rectangle that does not cover such a view fails the whole list
//! (`render-wgpu`'s `rules::check_clear_area` and its `WHOLE_CLEAR` message). So a clear whose
//! area is part of a depth target, or part of a multisampled colour target, is drawn (below)
//! instead of being a [`Command::Clear`] with a rectangle; the pixels are the same. Only a
//! single-sampled colour target is cleared in part by a [`Command::Clear`] with a rectangle.
//! `tests::golden_scissored_depth`, `tests::golden_depth_larger_than_the_render_area`,
//! `tests::golden_scissored_msaa`.
//!
//! # The draw-based clear
//!
//! The render backend's [`ClearDesc`] has no write mask, so a masked clear, and a partial clear
//! of a target a clear command clears only whole, is drawn with the render crate's clear shader
//! ([`HostShader::Clear`], ported from the Eden shaders that `blit_image` draws with): one
//! triangle covering the target, the clear area as the scissor, the value in a 16-byte uniform
//! buffer, the pipeline of the target's own sample count, and
//!
//! - for colour, the target alone at colour slot 0, written through the channels asked
//!   ([`ColorWrites`]), by the fragment entry point of the target's kind (float, signed or
//!   unsigned integer);
//! - for depth and stencil, the depth target alone, and depth written with the test `ALWAYS`
//!   only when depth is asked (Eden's `FindOrEmplaceClearStencilPipeline`). When stencil is
//!   asked, the stencil test is `ALWAYS` with `REPLACE` on pass, the reference the clear value,
//!   the write mask `SET_STENCIL_MASK` when it is partial and `0xff` otherwise (a full clear
//!   writes every bit), and the read mask [`STENCIL_READ_MASK`]; when it is not, the pipeline
//!   has no stencil state, so the stencil is kept. The fragment stage writes the depth value
//!   when the format has depth; a stencil-only target has no fragment stage
//!   ([`host_shaders::clear::FRAGMENT_DEPTH`] says why).
//!   `tests::a_partial_area_depth_stencil_draw_writes_only_the_aspects_asked`.
//!
//! The draw sets its own render targets ([`Command::SetRenderTargets`]), so the next guest draw
//! sets the guest's again. Its pipelines are made through the backend and kept, at most
//! [`MAX_CLEAR_PIPELINES`] of them (a **guess** bounding what a guest can make the back hold;
//! past it a masked clear fails with [`DrawError::ClearPipelinesFull`]). A pipeline that failed
//! to build skips its clears ([`ClearSkip::PipelineFailed`]). Every host object here is
//! destroyed when the draw path is dropped, after its last submitted fence.
//!
//! # The spec's offline test
//!
//! gpu-protocol's slice table gives G24's test as "a whole-target clear of colour, depth and
//! stencil each, and a masked and a scissored clear that become draws". Here a scissored clear
//! of a single-sampled colour target with every channel is a [`Command::Clear`] with a
//! rectangle, since the render backend's clear takes one ("colour, depth, stencil; whole or
//! rect", render-backend, "Commands") and the wgpu backend clears part of such a view, as the
//! spec's "Clears" paragraph allows ("a draw when the clear is masked or scissored in a way a
//! render-pass clear cannot express") and issue #428 asks ("a clear rectangle where the backend
//! can"). A masked clear is a draw, and so is a scissored or otherwise partial clear of a depth,
//! stencil or multisampled target ("Partial clears"). An **owner question**: whether the
//! table's wording should follow. `tests::golden_scissored`, `tests::golden_masked`,
//! `tests::golden_scissored_depth`, `tests::golden_depth_larger_than_the_render_area`,
//! `tests::golden_scissored_msaa`.
//!
//! # Errors and what they leave
//!
//! Every check that can skip a part comes before the targets are resolved, except a failed
//! pipeline ([`ClearSkip::PipelineFailed`]), which is known only once the part's target is
//! resolved and its pipeline looked up. A part whose target, pipeline or staging memory cannot be
//! had fails the clear ([`DrawError`]) before any clear or draw command is appended, so a clear
//! appends its commands whole or not at all (`tests::a_failed_part_appends_no_part_of_the_clear`).
//! What resolving did stays: uploads the texture cache appended, a target marked GPU-modified, a
//! pipeline made, as for a draw (`back::draw`, "Errors and what they leave").
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`.
//!
//! - `eden-nx2/src/video_core/engines/draw_manager.cpp` @ 50fa86bb (`DrawManager::Clear`,
//!   `ProcessMethodCall`'s `clear_surface` case) and
//!   `eden-nx2/src/video_core/renderer_vulkan/vk_rasterizer.cpp` @ 50fa86bb
//!   (`RasterizerVulkan::Clear`, `GetScissorState`), GPL-3.0-or-later.
//! - `eden-nx2/src/video_core/renderer_vulkan/blit_image.cpp` @ 50fa86bb (`ClearColor`,
//!   `ClearDepthStencil`, `FindOrEmplaceClearStencilPipeline`), `vk_texture_cache.cpp`
//!   (`Framebuffer::CreateFramebuffer`'s render area) and `texture_cache/texture_cache.h`
//!   (`FindColorBuffer`, `FindDepthBuffer`), GPL-3.0-or-later.
//! - `eden-nx2/src/video_core/surface.cpp` @ 50fa86bb (`PixelComponentSizeBitsInteger`), for
//!   [`integer_component_bits`].
//! - Registers, from `classes/clc797.h`
//!   (`mesa-nouveau/src/nouveau/headers/nvidia/classes/` @ 52238762, MIT): `CLEAR_SURFACE`
//!   (0x19d0: `Z_ENABLE` 0:0, `STENCIL_ENABLE` 1:1, `R_ENABLE` to `A_ENABLE` 2:2 to 5:5,
//!   `MRT_SELECT` 9:6, `RT_ARRAY_INDEX` 25:10), `SET_CLEAR_SURFACE_CONTROL` (0x10f8:
//!   `RESPECT_STENCIL_MASK` 0:0, `USE_CLEAR_RECT` 4:4, `USE_SCISSOR0` 8:8,
//!   `USE_VIEWPORT_CLIP0` 12:12), `SET_COLOR_CLEAR_VALUE(i)` (0x0d80 + 4i), `SET_Z_CLEAR_VALUE`
//!   (0x0d90, a float in Mesa's `METHOD_IS_FLOAT`), `SET_STENCIL_CLEAR_VALUE` (0x0da0, `V`
//!   7:0), `SET_STENCIL_MASK` (0x139c, `V` 7:0), `SET_STENCIL_FUNC_MASK` (0x1398, `V` 7:0; not
//!   read, under "Divergences"), `SET_SCISSOR_*(0)` (0x0e00 to 0x0e08). Eden's `Maxwell3D::Regs` places the same registers
//!   at the same offsets (`ASSERT_REG_POSITION` in `eden-nx2/src/video_core/engines/maxwell_3d.h`)
//!   and its `ClearSurface` and `ClearControl` bit fields match. Pinned against these literals
//!   by `tests::clear_registers_are_where_the_header_puts_them`.
//!
//! # Divergences from Eden-NX2
//!
//! Each is pinned by the test named, in `clear/tests.rs`.
//!
//! - **One layer, always.** Eden (both pins) defers a whole, unmasked clear at layer 0 with no
//!   render pass open into the next render pass's load operation (`scheduler.DeferColorClear`,
//!   realised by `Scheduler::RealizeDeferredClear` in `vk_scheduler.cpp`), which clears every
//!   layer of a layered framebuffer; otherwise it clears the one layer `RT_ARRAY_INDEX` names
//!   (`ClearAttachments` with `VkClearRect`'s `baseArrayLayer`, `layerCount` 1). Here a clear is
//!   always of that one layer, as the `ClearAttachments` path.
//!   `a_clear_of_layer_zero_of_a_layered_target_clears_that_layer_alone`.
//! - **A target at GPU address 0 is cleared like any other**, as G18 draws to it
//!   ([`super::targets`]); Eden's `FindColorBuffer` and `FindDepthBuffer` give no view for
//!   address 0, so it clears nothing there. `a_target_at_address_zero_is_cleared_like_any_other`.
//! - **A layer past the target's is skipped** ([`ClearSkip::Layer`]), where Eden hands Vulkan a
//!   layer the framebuffer does not have. `a_layer_past_the_targets_is_skipped`.
//! - **A colour target that is not bound is skipped** ([`ClearSkip::NotBound`]). Eden's
//!   `HasAspectColorBit` reads `rt_map` for it, which holds a stale or zero attachment index.
//!   `a_colour_target_that_is_not_bound_is_skipped`.
//! - **A reversed scissor is empty.** Eden's `GetScissorState` stores `max - min` in a `VkRect2D`
//!   extent, so a scissor whose end is below its start becomes a huge extent and clears from its
//!   start to the edge of the render area; here such a span is empty, as G18's scissors
//!   ([`super::fixed`]). `a_reversed_scissor_clears_nothing`.
//! - **A masked colour clear writes through a write mask**, where Eden blends with the mask as
//!   the blend constant (`ClearColor`'s `SetBlendConstants`), which wgpu cannot do for an
//!   integer target; for a float target the pixels are the same.
//!   `a_masked_colour_clear_is_a_draw_through_the_mask`.
//! - **The masked colour draw writes the converted value** for an integer target, the same value
//!   a full clear writes; Eden's colour clear shader writes the floats as they are.
//!   `a_masked_integer_clear_draws_the_converted_value`.
//! - **A depth clear value is clamped to 0 to 1, and NaN is skipped** ([`ClearSkip::DepthNotANumber`]).
//!   Eden hands any value to Vulkan, which requires 0 to 1; the render backend refuses a value
//!   outside it (render-backend, "Errors and untrusted input", decided by the owner,
//!   2026-11-19), which would lose the whole list.
//!   `a_depth_value_outside_zero_to_one_is_clamped_and_nan_is_skipped`.
//! - **Out-of-range integer conversions saturate.** Eden's [`color_value`] formula converts a
//!   float out of the integer's range (or NaN) to `u32` or `s32`, which C++ leaves undefined;
//!   Rust's conversion saturates and makes NaN 0. `integer_conversions_saturate`.
//! - **The stencil value and masks are their `7:0` fields.** Eden reads the whole words; the
//!   bits above 7 change nothing on an 8-bit stencil, except that Eden's partial-mask test sees
//!   them (a mask word of `0x1ff` is partial there and full here; Eden's draw then writes
//!   through `u8(0x1ff)`, all eight bits, so the pixels agree). Decision 10 lets the header
//!   settle the width. `stencil_registers_are_their_low_byte`.
//! - **`SET_STENCIL_FUNC_MASK` is not read.** Eden's masked stencil clear sets it as the
//!   pipeline's compare mask and keys its pipelines on it (`BlitDepthStencilPipelineKey`'s
//!   `stencil_compare_mask`); with the compare `ALWAYS` the mask reads no bit, so here the read
//!   mask is the fixed [`STENCIL_READ_MASK`] and a title's stencil-test state cannot multiply
//!   clear pipelines toward [`MAX_CLEAR_PIPELINES`]. The pixels are the same.
//!   `masked_stencil_clears_under_any_func_mask_share_one_pipeline`.
//! - **A partial clear of a depth, stencil or multisampled target is a draw**, where Eden clears
//!   the rectangle with `ClearAttachments` ("Partial clears"); the pixels are the same.
//!   `golden_scissored_depth`, `golden_depth_larger_than_the_render_area`,
//!   `golden_scissored_msaa`.
//! - **`SET_SURFACE_CLIP` does not bound the render area.** Eden's framebuffer starts its render
//!   area at the surface clip's width and height (`render_targets.size` in
//!   `UpdateRenderTargets`) before taking the minimum over its attachments; here only the
//!   targets bound it, so a surface clip smaller than the targets clears more than Eden.
//!   `the_surface_clip_does_not_bound_the_clear`.
//!
//! # Not read
//!
//! - `SET_CLEAR_SURFACE_CONTROL`'s `RESPECT_STENCIL_MASK`, `USE_CLEAR_RECT` and
//!   `USE_VIEWPORT_CLIP0`, and `SET_CLEAR_RECT_HORIZONTAL` and `_VERTICAL`: Eden reads only
//!   `use_scissor`. A partial stencil mask is always respected and a mask of 0 is a full clear,
//!   as Eden's `stencil_partial` test has it. A **guess** that NVN2's clears do not depend on
//!   the three bits (a title clearing through the clear rect would be bounded by scissor 0 or
//!   by nothing here); a trace of NVN2's `SET_CLEAR_SURFACE_CONTROL` writes would confirm.
//!   `the_unread_control_bits_change_nothing`.
//! - Eden's `IsFullClear`, which lets a whole-target clear skip uploading the target first:
//!   here the target is uploaded if it must be and then cleared, which costs an upload and
//!   gives the same pixels.
//! - Eden's resolution rescaling, its occlusion query segment (`query_cache.NotifySegment`) and
//!   its viewport update, none of which change what a clear writes here.
//!
//! # Guesses
//!
//! - **Integer clear values** ([`color_value`]): Eden-NX2 converts `SET_COLOR_CLEAR_VALUE`'s
//!   floats with `u32(f32(bits << 1) * value)` and `s32(f32((bits - 1) << 1) * (value - 0.5))`.
//!   Mesa's parser lists only `SET_Z_CLEAR_VALUE` as a float, which suggests the colour words
//!   are raw bits for an integer target. Decision 10(c) follows the observed behaviour where the
//!   header explains nothing; an **owner question** until a trace of an integer clear settles
//!   it. `integer_clear_values_follow_edens_formula`.
//! - **A 3D target deeper than one slice is skipped** ([`ClearSkip::VolumeTarget`]): which
//!   slice `RT_ARRAY_INDEX` names, and how the backend would clear one slice, is not built.

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::Arc;

use starquake_layout::ImageDim;
use starquake_render::caps::GUEST_COLOR_TARGETS;
use starquake_render::command::{
    Binding, BindingResource, BufferSlice, ClearDesc, ClearValue, CopyRegion, DrawDesc, DrawParams,
    DynamicValues, Rect, RenderTargets, Viewport,
};
use starquake_render::format::FormatAspect;
use starquake_render::host_shaders::{self, OUTPUT_LOCATION};
use starquake_render::pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorWrites, CullMode, DepthBias,
    DepthStencilState, FrontFace, PolygonMode, RasterState, Stages, StencilFace, StencilOperation,
    StencilState, TargetFormats, Topology, VertexLayout,
};
use starquake_render::resource::CompareFunction;
use starquake_render::shader::{
    InterfaceVariable, Interpolation, ScalarKind, ShaderDesc, ShaderInterface, ShaderSource,
    ShaderStage,
};
use starquake_render::{
    Backend, BufferDesc, BufferId, BufferUsage, Command, CommandList, EmulationFlags, FenceValue,
    GraphicsPipelineDesc, HostShader, PipelineId, PipelineStatus, PixelFormat, ResourceId,
    ShaderId, StagingSlice, ViewId,
};

use super::{DrawContext, DrawError, fixed, targets};
use crate::back::state::{COLOR_TARGETS, Mirror};
use crate::back::texture::ImageInfo;
use crate::classes::{Method, clc797};
use crate::memory::GpuMemory;

/// The methods whose words start a clear: the executor drains CPU writes and flushes inline
/// writes before them, as before a draw (`back::exec`, "Before each draw, dispatch and copy").
pub const CLEAR_METHODS: [&Method; 1] = [&clc797::CLEAR_SURFACE];

/// The most pipelines the draw-based clear keeps per draw path (module docs; a **guess**).
pub const MAX_CLEAR_PIPELINES: usize = 256;

/// The value of a full `SET_STENCIL_MASK`: Eden's `stencil_partial` test compares with `0xFF`.
pub const FULL_STENCIL_MASK: u32 = 0xff;

/// The draw-based stencil clear's read mask: its compare is `ALWAYS`, which reads no bit, so the
/// mask is fixed rather than `SET_STENCIL_FUNC_MASK` (module docs, "Divergences").
pub const STENCIL_READ_MASK: u32 = FULL_STENCIL_MASK;

/// The clear shader's parameter block, in bytes, as an array length.
const PARAMS_LEN: usize = host_shaders::clear::PARAMS_SIZE as usize;

/// One word written to `CLEAR_SURFACE`, decoded (module docs, step 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClearSurface {
    /// `Z_ENABLE`.
    pub depth: bool,
    /// `STENCIL_ENABLE`.
    pub stencil: bool,
    /// `R_ENABLE` to `A_ENABLE`.
    pub color: ColorWrites,
    /// `MRT_SELECT`: the colour target.
    pub target: u32,
    /// `RT_ARRAY_INDEX`: the layer.
    pub layer: u32,
}

/// The colour channels of `CLEAR_SURFACE` and their write bits, in Eden's `color_mask` order.
const CHANNELS: [(crate::field::Field, ColorWrites); 4] = [
    (clc797::CLEAR_SURFACE_R_ENABLE, ColorWrites::RED),
    (clc797::CLEAR_SURFACE_G_ENABLE, ColorWrites::GREEN),
    (clc797::CLEAR_SURFACE_B_ENABLE, ColorWrites::BLUE),
    (clc797::CLEAR_SURFACE_A_ENABLE, ColorWrites::ALPHA),
];

impl ClearSurface {
    /// The fields of `word`, through the header's bit positions.
    pub const fn decode(word: u32) -> Self {
        let mut color = ColorWrites::EMPTY;
        let mut i = 0;
        while i < CHANNELS.len() {
            if CHANNELS[i].0.is_set(word) {
                color = color.union(CHANNELS[i].1);
            }
            i += 1;
        }
        Self {
            depth: clc797::CLEAR_SURFACE_Z_ENABLE.is_set(word),
            stencil: clc797::CLEAR_SURFACE_STENCIL_ENABLE.is_set(word),
            color,
            target: clc797::CLEAR_SURFACE_MRT_SELECT.get(word),
            layer: clc797::CLEAR_SURFACE_RT_ARRAY_INDEX.get(word),
        }
    }

    /// The word with these fields, through the same bit positions; `None` when the target or
    /// the layer does not fit its field.
    pub fn encode(&self) -> Option<u32> {
        let mut word = clc797::CLEAR_SURFACE_Z_ENABLE.insert(0, u32::from(self.depth))?;
        word = clc797::CLEAR_SURFACE_STENCIL_ENABLE.insert(word, u32::from(self.stencil))?;
        for (field, bit) in CHANNELS {
            word = field.insert(word, u32::from(self.color.contains(bit)))?;
        }
        word = clc797::CLEAR_SURFACE_MRT_SELECT.insert(word, self.target)?;
        clc797::CLEAR_SURFACE_RT_ARRAY_INDEX.insert(word, self.layer)
    }
}

/// `SET_CLEAR_SURFACE_CONTROL`, decoded through the header's bit positions. Only
/// [`ClearControl::use_scissor0`] changes a clear (module docs, "Not read").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ClearControl(u32);

impl ClearControl {
    /// The register's word.
    pub const fn from_word(word: u32) -> Self {
        Self(word)
    }

    /// The register's word.
    pub const fn word(self) -> u32 {
        self.0
    }

    /// `RESPECT_STENCIL_MASK` (not read).
    pub const fn respect_stencil_mask(self) -> bool {
        clc797::SET_CLEAR_SURFACE_CONTROL_RESPECT_STENCIL_MASK.is_set(self.0)
    }

    /// `USE_CLEAR_RECT` (not read).
    pub const fn use_clear_rect(self) -> bool {
        clc797::SET_CLEAR_SURFACE_CONTROL_USE_CLEAR_RECT.is_set(self.0)
    }

    /// `USE_SCISSOR0`: the clear is bounded by scissor 0.
    pub const fn use_scissor0(self) -> bool {
        clc797::SET_CLEAR_SURFACE_CONTROL_USE_SCISSOR0.is_set(self.0)
    }

    /// `USE_VIEWPORT_CLIP0` (not read).
    pub const fn use_viewport_clip0(self) -> bool {
        clc797::SET_CLEAR_SURFACE_CONTROL_USE_VIEWPORT_CLIP0.is_set(self.0)
    }
}

/// Why a part of a clear was not done, though nothing was wrong with the state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClearSkip {
    /// Conditional rendering turned it off.
    RenderDisabled,
    /// The clear area, clamped to the render area, is empty.
    EmptyArea,
    /// The colour target is not below `SET_CT_SELECT`'s count or its format is `DISABLED`, or
    /// no depth target is bound.
    NotBound,
    /// Neither aspect asked (depth, stencil) is in the depth target's format.
    NoAspect,
    /// `RT_ARRAY_INDEX` is at or past the target's layers.
    Layer {
        /// The layer asked for.
        layer: u32,
        /// The target's layers.
        layers: u32,
    },
    /// The target is a 3D image of more than one slice (module docs, "Guesses").
    VolumeTarget,
    /// `SET_Z_CLEAR_VALUE` is NaN.
    DepthNotANumber,
    /// The draw-based clear's pipeline failed to build.
    PipelineFailed,
}

/// What became of one part (colour, or depth and stencil) of a clear.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// A [`Command::Clear`] was appended.
    Cleared,
    /// A draw was appended (module docs, "The draw-based clear").
    Drawn,
    /// It was skipped.
    Skipped(ClearSkip),
}

/// What one `CLEAR_SURFACE` word did: each part asked, or `None` for a part not asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cleared {
    /// The colour part.
    pub color: Option<Part>,
    /// The depth and stencil part.
    pub depth_stencil: Option<Part>,
}

impl Cleared {
    /// A word that asked for nothing.
    pub const NOTHING: Self = Self {
        color: None,
        depth_stencil: None,
    };

    /// Whether a part was drawn, which sets render targets of its own.
    pub fn drew(&self) -> bool {
        self.color == Some(Part::Drawn) || self.depth_stencil == Some(Part::Drawn)
    }

    /// Every part asked, skipped for `why`.
    fn skipped(surface: &ClearSurface, why: ClearSkip) -> Self {
        Self {
            color: (!surface.color.is_empty()).then_some(Part::Skipped(why)),
            depth_stencil: (surface.depth || surface.stencil).then_some(Part::Skipped(why)),
        }
    }
}

/// Counts of what clears did, for counters and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClearCounters {
    /// Words written to `CLEAR_SURFACE`.
    pub words: u64,
    /// Parts done with a [`Command::Clear`].
    pub cleared: u64,
    /// Parts done with a draw.
    pub drawn: u64,
    /// Parts skipped ([`ClearSkip`]).
    pub skipped: u64,
    /// Words that asked for nothing.
    pub nothing: u64,
    /// Words that failed with a [`DrawError`].
    pub failed: u64,
}

impl ClearCounters {
    /// Counts one word's outcome.
    fn count(&mut self, outcome: &Result<Cleared, DrawError>) {
        let bump = |n: &mut u64| *n = n.saturating_add(1);
        bump(&mut self.words);
        match outcome {
            Ok(cleared) if *cleared == Cleared::NOTHING => bump(&mut self.nothing),
            Ok(cleared) => {
                for part in [cleared.color, cleared.depth_stencil].into_iter().flatten() {
                    match part {
                        Part::Cleared => bump(&mut self.cleared),
                        Part::Drawn => bump(&mut self.drawn),
                        Part::Skipped(_) => bump(&mut self.skipped),
                    }
                }
            }
            Err(_) => bump(&mut self.failed),
        }
    }
}

/// The colour clear value for a target of `format` from `SET_COLOR_CLEAR_VALUE(0..3)` read as
/// floats (module docs, step 4): the floats for a float or normalised format, else Eden's
/// conversion by [`integer_component_bits`]: `u32(f32(bits << 1) * value)` for an unsigned
/// format and `s32(f32((bits - 1) << 1) * (value - 0.5))` for a signed one (a **guess**, module
/// docs, "Guesses"). An integer format Eden's table does not list would convert with 0 bits,
/// which no colour target reaches: every integer format [`super::targets::color_format`] gives
/// is in the table (`tests::every_integer_colour_target_format_has_a_component_width`). Were one
/// to, it would diverge from Eden for a signed format: the scale here saturates to 0, where
/// Eden's `(int_size - 1) << 1` gives -2.
pub fn color_value(format: PixelFormat, values: [f32; 4]) -> ClearValue {
    if !format.is_integer() {
        return ClearValue::Float(values);
    }
    let bits = integer_component_bits(format).unwrap_or(0);
    if format.is_signed_integer() {
        let scale = f32::from(bits.saturating_sub(1).saturating_mul(2));
        ClearValue::Sint(values.map(|v| saturating_i32(scale * (v - 0.5))))
    } else {
        let scale = f32::from(bits.saturating_mul(2));
        ClearValue::Uint(values.map(|v| saturating_u32(scale * v)))
    }
}

/// `value` as a `u32`: saturating, NaN to 0 (module docs, "Divergences").
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Rust's float-to-integer conversion saturates, which is the behaviour documented"
)]
fn saturating_u32(value: f32) -> u32 {
    value as u32
}

/// `value` as an `i32`: saturating, NaN to 0 (module docs, "Divergences").
#[expect(
    clippy::cast_possible_truncation,
    reason = "Rust's float-to-integer conversion saturates, which is the behaviour documented"
)]
fn saturating_i32(value: f32) -> i32 {
    value as i32
}

/// The bits of one component of an integer colour format, as Eden-NX2's
/// `PixelComponentSizeBitsInteger` lists them (`eden-nx2/src/video_core/surface.cpp` @
/// 50fa86bb); `None` for a format it does not list.
pub fn integer_component_bits(format: PixelFormat) -> Option<u8> {
    use PixelFormat as P;
    Some(match format {
        P::A8B8G8R8Sint | P::A8B8G8R8Uint | P::R8Sint | P::R8Uint | P::R8G8Sint | P::R8G8Uint => 8,
        P::A2B10G10R10Uint => 10,
        P::R16G16B16A16Sint
        | P::R16G16B16A16Uint
        | P::R16Uint
        | P::R16Sint
        | P::R16G16Uint
        | P::R16G16Sint => 16,
        P::R32G32B32A32Uint
        | P::R32G32B32A32Sint
        | P::R32G32Sint
        | P::R32G32Uint
        | P::R32Uint
        | P::R32Sint => 32,
        _ => return None,
    })
}

/// `rect` clamped to a render area of `width` by `height`, as Eden's `clamp_rect_to_render_area`;
/// `None` when nothing is left (an empty extent, a start at or past the area's end, or an empty
/// area).
pub fn clamp_to_area(rect: Rect, width: u32, height: u32) -> Option<Rect> {
    let axis = |start: u32, extent: u32, limit: u32| -> Option<(u32, u32)> {
        if extent == 0 || start >= limit {
            return None;
        }
        let end = u64::from(start)
            .saturating_add(u64::from(extent))
            .min(u64::from(limit));
        let extent = u32::try_from(end - u64::from(start)).ok()?;
        Some((start, extent))
    };
    let (x, w) = axis(rect.x, rect.width, width)?;
    let (y, h) = axis(rect.y, rect.height, height)?;
    Some(Rect {
        x,
        y,
        width: w,
        height: h,
    })
}

/// The clear values of the register file.
#[derive(Clone, Copy, Debug)]
struct Values {
    color: [f32; 4],
    depth: f32,
    stencil: u32,
    stencil_mask: u32,
}

impl Values {
    fn read(mirror: &Mirror) -> Result<Self, DrawError> {
        let mut color = [0.0; 4];
        for (index, slot) in (0u32..).zip(color.iter_mut()) {
            *slot = mirror
                .reg(&clc797::SET_COLOR_CLEAR_VALUE, &[index])?
                .float();
        }
        Ok(Self {
            color,
            depth: mirror.reg(&clc797::SET_Z_CLEAR_VALUE, &[])?.float(),
            stencil: mirror
                .reg(&clc797::SET_STENCIL_CLEAR_VALUE, &[])?
                .field(clc797::SET_STENCIL_CLEAR_VALUE_V)?,
            stencil_mask: mirror
                .reg(&clc797::SET_STENCIL_MASK, &[])?
                .field(clc797::SET_STENCIL_MASK_V)?,
        })
    }
}

/// The key of a draw-based clear's pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PipelineKey {
    /// A colour clear of one target through `mask` (a [`ColorWrites`] bit set).
    Color {
        format: PixelFormat,
        mask: u32,
        samples: u32,
    },
    /// A depth and stencil clear: depth written when `depth`, stencil through the write mask
    /// `stencil` when it is `Some`. The stencil read mask is always [`STENCIL_READ_MASK`].
    DepthStencil {
        format: PixelFormat,
        depth: bool,
        stencil: Option<u32>,
        samples: u32,
    },
}

/// The clear shader's entry points (the vertex one, then the fragment ones by kind).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Entry {
    Vertex,
    Float,
    Sint,
    Uint,
    Depth,
}

impl Entry {
    const COUNT: usize = 5;

    fn index(self) -> usize {
        match self {
            Self::Vertex => 0,
            Self::Float => 1,
            Self::Sint => 2,
            Self::Uint => 3,
            Self::Depth => 4,
        }
    }

    fn name(self) -> &'static str {
        use host_shaders::clear as c;
        match self {
            Self::Vertex => c::VERTEX,
            Self::Float => c::FRAGMENT_COLOR_FLOAT,
            Self::Sint => c::FRAGMENT_COLOR_SINT,
            Self::Uint => c::FRAGMENT_COLOR_UINT,
            Self::Depth => c::FRAGMENT_DEPTH,
        }
    }

    /// The colour entry point for a target of `format`.
    fn for_color(format: PixelFormat) -> Self {
        if !format.is_integer() {
            Self::Float
        } else if format.is_signed_integer() {
            Self::Sint
        } else {
            Self::Uint
        }
    }

    fn interface(self) -> ShaderInterface {
        let output = |scalar| InterfaceVariable {
            location: OUTPUT_LOCATION,
            scalar,
            components: 4,
            interpolation: Interpolation::Perspective,
        };
        let (resources, outputs) = match self {
            Self::Vertex => (Vec::new(), Vec::new()),
            Self::Float => (
                host_shaders::clear::RESOURCES.to_vec(),
                vec![output(ScalarKind::Float)],
            ),
            Self::Sint => (
                host_shaders::clear::RESOURCES.to_vec(),
                vec![output(ScalarKind::Sint)],
            ),
            Self::Uint => (
                host_shaders::clear::RESOURCES.to_vec(),
                vec![output(ScalarKind::Uint)],
            ),
            Self::Depth => (host_shaders::clear::RESOURCES.to_vec(), Vec::new()),
        };
        ShaderInterface {
            resources,
            inputs: Vec::new(),
            outputs,
        }
    }
}

/// The host objects of the draw-based clear (module docs), made on first use.
pub(crate) struct Clears {
    backend: Arc<dyn Backend>,
    shaders: [Option<ShaderId>; Entry::COUNT],
    /// Made in this order, destroyed in this order.
    made_shaders: Vec<ShaderId>,
    params: Option<BufferId>,
    pipelines: HashMap<PipelineKey, PipelineId>,
    /// Made in this order, destroyed in this order.
    made_pipelines: Vec<PipelineId>,
    pipeline_limit: usize,
    counters: ClearCounters,
}

impl core::fmt::Debug for Clears {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Clears")
            .field("pipelines", &self.made_pipelines.len())
            .field("counters", &self.counters)
            .finish_non_exhaustive()
    }
}

/// How one part is done, once every check passed and its target is resolved.
enum Plan {
    Clear(ClearDesc),
    Draw {
        staging: StagingSlice,
        targets: RenderTargets,
        desc: Box<DrawDesc>,
    },
}

/// What every part of one clear shares.
struct Common {
    /// The clear area, clamped to the render area.
    area: Rect,
    /// The clear values.
    values: Values,
}

/// What a part needs to know of its target.
struct Target {
    info: ImageInfo,
    layer: u32,
}

impl Target {
    /// The checks every part makes of its target (module docs): a volume deeper than one slice,
    /// a layer past the target's.
    fn check(info: ImageInfo, layer: u32) -> Result<Self, ClearSkip> {
        if info.dim == ImageDim::D3 && info.depth > 1 {
            return Err(ClearSkip::VolumeTarget);
        }
        // A 3D target is one layer (`targets::color_target`), so this bounds its one slice too.
        let layers = info.layers;
        if layer >= layers {
            return Err(ClearSkip::Layer { layer, layers });
        }
        Ok(Self { info, layer })
    }

    /// `area` as a clear command's rectangle: `None` when it covers the target.
    fn rect(&self, area: Rect) -> Option<Rect> {
        let whole = Rect {
            x: 0,
            y: 0,
            width: self.info.width,
            height: self.info.height,
        };
        (area != whole).then_some(area)
    }

    /// The target's samples per pixel.
    fn samples(&self) -> u32 {
        self.info.samples.samples()
    }

    /// Whether a [`Command::Clear`] of `area` can clear it: the area covers it, or it is a
    /// single-sampled colour target (module docs, "Partial clears").
    fn clearable(&self, area: Rect) -> bool {
        self.rect(area).is_none()
            || (self.info.format.aspect() == FormatAspect::Color && self.samples() == 1)
    }
}

impl Clears {
    /// No host objects yet; they are made through `backend`.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self {
            backend,
            shaders: [None; Entry::COUNT],
            made_shaders: Vec::new(),
            params: None,
            pipelines: HashMap::new(),
            made_pipelines: Vec::new(),
            pipeline_limit: MAX_CLEAR_PIPELINES,
            counters: ClearCounters::default(),
        }
    }

    /// What clears have done so far.
    pub fn counters(&self) -> ClearCounters {
        self.counters
    }

    /// The pipelines the draw-based clear holds.
    pub fn pipelines(&self) -> usize {
        self.made_pipelines.len()
    }

    /// One word `word` written to `CLEAR_SURFACE`, after `mirror` stored it (module docs).
    /// `execute` is conditional rendering's answer; `last_targets` the render targets the draw
    /// path last set, which a draw-based clear replaces with its own.
    pub(crate) fn clear<M: GpuMemory + ?Sized>(
        &mut self,
        ctx: &mut DrawContext<'_, M>,
        mirror: &Mirror,
        word: u32,
        execute: bool,
        last_targets: &mut Option<RenderTargets>,
    ) -> Result<Cleared, DrawError> {
        let outcome = self.run(ctx, mirror, word, execute, last_targets);
        self.counters.count(&outcome);
        if let Err(error) = &outcome {
            tracing::debug!(%error, "a clear could not be done");
        }
        outcome
    }

    fn run<M: GpuMemory + ?Sized>(
        &mut self,
        ctx: &mut DrawContext<'_, M>,
        mirror: &Mirror,
        word: u32,
        execute: bool,
        last_targets: &mut Option<RenderTargets>,
    ) -> Result<Cleared, DrawError> {
        let surface = ClearSurface::decode(word);
        // Step 1.
        if !execute {
            return Ok(Cleared::skipped(&surface, ClearSkip::RenderDisabled));
        }
        // Step 2.
        let color_asked = !surface.color.is_empty();
        let ds_asked = surface.depth || surface.stencil;
        if !color_asked && !ds_asked {
            return Ok(Cleared::NOTHING);
        }
        // Step 3.
        let three_d = mirror.three_d()?;
        let bound = Framebuffer::read(mirror)?;
        let control =
            ClearControl::from_word(mirror.reg(&clc797::SET_CLEAR_SURFACE_CONTROL, &[])?.word());
        let scissor = if control.use_scissor0() {
            fixed::guest_scissor(three_d, 0)?
        } else {
            WHOLE
        };
        let (width, height) = bound.render_area();
        let Some(area) = clamp_to_area(scissor, width, height) else {
            return Ok(Cleared::skipped(&surface, ClearSkip::EmptyArea));
        };
        let values = Values::read(mirror)?;

        // Step 4: the checks, then the target.
        let color = if color_asked {
            Some(match bound.color_target(&surface) {
                Ok(target) => {
                    let value = color_value(target.info.format, values.color);
                    Ok((target, value))
                }
                Err(skip) => Err(skip),
            })
        } else {
            None
        };
        // Step 5: the checks.
        let depth_stencil = if ds_asked {
            Some(depth_stencil_part(&bound, &surface, &values))
        } else {
            None
        };

        // Every check passed or skipped; resolve and plan, appending no clear or draw.
        let common = Common { area, values };
        let mut plans: Vec<Plan> = Vec::with_capacity(2);
        let outcome = Cleared {
            color: match color {
                None => None,
                Some(Err(skip)) => Some(Part::Skipped(skip)),
                Some(Ok((target, value))) => Some(self.plan_color(
                    ctx,
                    &mut plans,
                    &target,
                    value,
                    surface.color,
                    &common,
                )?),
            },
            depth_stencil: match depth_stencil {
                None => None,
                Some(Err(skip)) => Some(Part::Skipped(skip)),
                Some(Ok(part)) => Some(self.plan_depth_stencil(ctx, &mut plans, &part, &common)?),
            },
        };

        // Nothing below can fail.
        for plan in plans {
            append(ctx.list, plan, self.params, last_targets);
        }
        Ok(outcome)
    }

    /// Resolves the colour part's target and plans its clear or draw.
    fn plan_color<M: GpuMemory + ?Sized>(
        &mut self,
        ctx: &mut DrawContext<'_, M>,
        plans: &mut Vec<Plan>,
        target: &Target,
        value: ClearValue,
        mask: ColorWrites,
        common: &Common,
    ) -> Result<Part, DrawError> {
        let view = ctx
            .textures
            .render_target_layer(ctx.memory, &target.info, target.layer, ctx.list)?
            .view;
        if mask == ColorWrites::ALL && target.clearable(common.area) {
            plans.push(Plan::Clear(ClearDesc {
                target: view,
                value,
                rect: target.rect(common.area),
            }));
            return Ok(Part::Cleared);
        }
        let key = PipelineKey::Color {
            format: target.info.format,
            mask: mask.bits(),
            samples: target.samples(),
        };
        self.push_draw(plans, &key, target, view, value_bits(value), common)
    }

    /// Resolves the depth and stencil part's target and plans its clear or draw.
    fn plan_depth_stencil<M: GpuMemory + ?Sized>(
        &mut self,
        ctx: &mut DrawContext<'_, M>,
        plans: &mut Vec<Plan>,
        part: &DepthStencilPart,
        common: &Common,
    ) -> Result<Part, DrawError> {
        let target = &part.target;
        let view = ctx
            .textures
            .render_target_layer(ctx.memory, &target.info, target.layer, ctx.list)?
            .view;
        if !part.partial && target.clearable(common.area) {
            plans.push(Plan::Clear(ClearDesc {
                target: view,
                value: part.value(),
                rect: target.rect(common.area),
            }));
            return Ok(Part::Cleared);
        }
        // A partial mask writes through it; an unmasked stencil (a mask of `0xff` or 0) writes
        // every bit, as a clear command would.
        let stencil = part.stencil.map(|_| {
            if part.partial {
                common.values.stencil_mask
            } else {
                FULL_STENCIL_MASK
            }
        });
        let key = PipelineKey::DepthStencil {
            format: target.info.format,
            depth: part.depth.is_some(),
            stencil,
            samples: target.samples(),
        };
        let bits = [part.depth.unwrap_or(0.0).to_bits(), 0, 0, 0];
        self.push_draw(plans, &key, target, view, bits, common)
    }

    /// Plans a draw-based clear ([`Clears::draw_plan`]) and says what became of the part.
    fn push_draw(
        &mut self,
        plans: &mut Vec<Plan>,
        key: &PipelineKey,
        target: &Target,
        view: ViewId,
        bits: [u32; 4],
        common: &Common,
    ) -> Result<Part, DrawError> {
        Ok(match self.draw_plan(key, target, view, bits, common)? {
            Some(plan) => {
                plans.push(plan);
                Part::Drawn
            }
            None => Part::Skipped(ClearSkip::PipelineFailed),
        })
    }

    /// A draw-based clear of `target` through `view` (module docs): its pipeline (made if new),
    /// the parameter block holding `bits` in staging memory, and the draw; `None` when the
    /// pipeline failed to build.
    fn draw_plan(
        &mut self,
        key: &PipelineKey,
        target: &Target,
        view: ViewId,
        bits: [u32; 4],
        common: &Common,
    ) -> Result<Option<Plan>, DrawError> {
        let values = &common.values;
        let pipeline = self.pipeline(key)?;
        if self.backend.pipeline_status(pipeline) == PipelineStatus::Failed {
            return Ok(None);
        }
        let buffer = self.params_buffer()?;
        let mut staging = self
            .backend
            .staging(u64::from(host_shaders::clear::PARAMS_SIZE))
            .map_err(DrawError::Render)?;
        let mut bytes = [0u8; PARAMS_LEN];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(bits) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        let Some(slot) = staging.bytes_mut().get_mut(..PARAMS_LEN) else {
            return Err(DrawError::Internal(
                "a staging slice is shorter than asked for",
            ));
        };
        slot.copy_from_slice(&bytes);
        let (targets, bindings, reference, write_mask) = match *key {
            PipelineKey::Color { .. } => {
                let mut colors = [None; GUEST_COLOR_TARGETS];
                colors[0] = Some(view);
                (
                    RenderTargets {
                        colors,
                        depth_stencil: None,
                    },
                    vec![params_binding(buffer)],
                    0,
                    0,
                )
            }
            PipelineKey::DepthStencil {
                format, stencil, ..
            } => (
                RenderTargets {
                    colors: [None; GUEST_COLOR_TARGETS],
                    depth_stencil: Some(view),
                },
                if format.has_depth() {
                    vec![params_binding(buffer)]
                } else {
                    Vec::new()
                },
                values.stencil,
                stencil.unwrap_or(0),
            ),
        };
        #[expect(
            clippy::cast_precision_loss,
            reason = "a target's extent is at most a 32-bit count; the viewport is a float"
        )]
        let viewport = Viewport {
            x: 0.0,
            y: 0.0,
            width: target.info.width as f32,
            height: target.info.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let desc = DrawDesc {
            pipeline,
            bindings,
            vertex_buffers: Vec::new(),
            index_buffer: None,
            dynamic: DynamicValues {
                viewports: vec![viewport],
                scissors: vec![common.area],
                blend_constant: [0.0; 4],
                stencil_reference: reference,
                stencil_read_mask: STENCIL_READ_MASK,
                stencil_write_mask: write_mask,
                depth_bias: DepthBias::default(),
                line_width: 1.0,
            },
            params: DrawParams::Direct {
                vertex_count: host_shaders::clear::VERTEX_COUNT,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
        };
        Ok(Some(Plan::Draw {
            staging,
            targets,
            desc: Box::new(desc),
        }))
    }

    /// The pipeline of `key`, made if new.
    fn pipeline(&mut self, key: &PipelineKey) -> Result<PipelineId, DrawError> {
        if let Some(&id) = self.pipelines.get(key) {
            return Ok(id);
        }
        if self.made_pipelines.len() >= self.pipeline_limit {
            return Err(DrawError::ClearPipelinesFull {
                limit: self.pipeline_limit,
            });
        }
        let vertex = self.shader(Entry::Vertex)?;
        let desc = match *key {
            PipelineKey::Color {
                format,
                mask,
                samples,
            } => {
                let fragment = self.shader(Entry::for_color(format))?;
                let mut blend = [DISABLED_BLEND; GUEST_COLOR_TARGETS];
                blend[0].write_mask = ColorWrites::from_bits(mask).unwrap_or(ColorWrites::EMPTY);
                let mut colors = [None; GUEST_COLOR_TARGETS];
                colors[0] = Some(format);
                pipeline_desc(
                    vertex,
                    Some(fragment),
                    DepthStencilState {
                        depth_test: None,
                        depth_write: false,
                        stencil: None,
                    },
                    blend,
                    TargetFormats {
                        colors,
                        depth_stencil: None,
                        sample_count: samples,
                    },
                )
            }
            PipelineKey::DepthStencil {
                format,
                depth,
                stencil,
                samples,
            } => {
                let fragment = if format.has_depth() {
                    Some(self.shader(Entry::Depth)?)
                } else {
                    None
                };
                let face = StencilFace {
                    compare: CompareFunction::Always,
                    fail: StencilOperation::Keep,
                    depth_fail: StencilOperation::Keep,
                    pass: StencilOperation::Replace,
                };
                pipeline_desc(
                    vertex,
                    fragment,
                    DepthStencilState {
                        depth_test: depth.then_some(CompareFunction::Always),
                        depth_write: depth,
                        stencil: stencil.map(|write_mask| StencilState {
                            front: face,
                            back: face,
                            read_mask: STENCIL_READ_MASK,
                            write_mask,
                        }),
                    },
                    [DISABLED_BLEND; GUEST_COLOR_TARGETS],
                    TargetFormats {
                        colors: [None; GUEST_COLOR_TARGETS],
                        depth_stencil: Some(format),
                        sample_count: samples,
                    },
                )
            }
        };
        let id = self
            .backend
            .create_graphics_pipeline(desc)
            .map_err(DrawError::Render)?;
        self.pipelines.insert(*key, id);
        self.made_pipelines.push(id);
        Ok(id)
    }

    /// The clear shader's entry point `entry`, made if new.
    fn shader(&mut self, entry: Entry) -> Result<ShaderId, DrawError> {
        if let Some(Some(id)) = self.shaders.get(entry.index()) {
            return Ok(*id);
        }
        let (module, info) = clear_module()?;
        let stage = if entry == Entry::Vertex {
            ShaderStage::Vertex
        } else {
            ShaderStage::Fragment
        };
        let id = self
            .backend
            .create_shader(ShaderDesc {
                stage,
                entry: entry.name().to_owned(),
                source: ShaderSource::Naga {
                    module: Box::new(module),
                    info: Box::new(info),
                },
                interface: entry.interface(),
                origin: None,
            })
            .map_err(DrawError::Render)?;
        if let Some(slot) = self.shaders.get_mut(entry.index()) {
            *slot = Some(id);
        }
        self.made_shaders.push(id);
        Ok(id)
    }

    /// The parameter block's buffer, made on first use.
    fn params_buffer(&mut self) -> Result<BufferId, DrawError> {
        if let Some(id) = self.params {
            return Ok(id);
        }
        let id = self
            .backend
            .create_buffer(&BufferDesc {
                size: u64::from(host_shaders::clear::PARAMS_SIZE),
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            })
            .map_err(DrawError::Render)?;
        self.params = Some(id);
        Ok(id)
    }

    /// Destroys every host object made here after `fence`: the pipelines, then the shaders,
    /// then the parameter buffer, each in the order made.
    pub fn destroy_all(&mut self, fence: FenceValue) {
        for id in self.made_pipelines.drain(..) {
            self.backend.destroy(ResourceId::Pipeline(id), fence);
        }
        self.pipelines.clear();
        for id in self.made_shaders.drain(..) {
            self.backend.destroy(ResourceId::Shader(id), fence);
        }
        self.shaders = [None; Entry::COUNT];
        if let Some(id) = self.params.take() {
            self.backend.destroy(ResourceId::Buffer(id), fence);
        }
    }
}

/// A scissor that does not bound the clear: Eden's `default_scissor`, the largest `i32` in each
/// extent.
const WHOLE: Rect = Rect {
    x: 0,
    y: 0,
    width: i32::MAX.unsigned_abs(),
    height: i32::MAX.unsigned_abs(),
};

/// A colour slot that blends nothing and writes nothing.
const DISABLED_BLEND: BlendState = BlendState {
    enabled: false,
    color: NO_BLEND,
    alpha: NO_BLEND,
    write_mask: ColorWrites::EMPTY,
};

/// The equation of a slot whose blending is off.
const NO_BLEND: BlendComponent = BlendComponent {
    src: BlendFactor::One,
    dst: BlendFactor::Zero,
    op: BlendOperation::Add,
};

/// The targets a clear's render area spans (module docs, step 3), as Eden's framebuffer holds
/// them: colour targets 0 to `SET_CT_SELECT`'s count, not through its map, and the depth
/// target.
struct Framebuffer {
    /// Colour target `i`'s image, for `i` below the count; `None` past it or when `DISABLED`.
    colors: [Option<ImageInfo>; GUEST_COLOR_TARGETS],
    /// The depth target's image, when `SET_ZT_SELECT`'s count is not 0.
    depth: Option<ImageInfo>,
}

impl Framebuffer {
    fn read(mirror: &Mirror) -> Result<Self, DrawError> {
        let three_d = mirror.three_d()?;
        let samples = targets::sample_layout(
            mirror
                .reg(&clc797::SET_ANTI_ALIAS, &[])?
                .field(clc797::SET_ANTI_ALIAS_SAMPLES)?,
        )?;
        let count = three_d.ct_select().target_count()?.min(COLOR_TARGETS);
        let mut colors = [None; GUEST_COLOR_TARGETS];
        for (index, slot) in (0..count).zip(colors.iter_mut()) {
            *slot = targets::color_target(three_d, index, samples)?;
        }
        let depth = if three_d.zt().target_count()? == 0 {
            None
        } else {
            Some(targets::depth_target(three_d, samples)?)
        };
        Ok(Self { colors, depth })
    }

    /// The render area: the smallest width and height of the targets, or the largest values
    /// when there is none.
    fn render_area(&self) -> (u32, u32) {
        self.colors
            .iter()
            .flatten()
            .chain(self.depth.iter())
            .fold((u32::MAX, u32::MAX), |(w, h), info| {
                (w.min(info.width), h.min(info.height))
            })
    }

    /// The colour target `surface` names, or why it is skipped (module docs, step 4).
    fn color_target(&self, surface: &ClearSurface) -> Result<Target, ClearSkip> {
        let info = usize::try_from(surface.target)
            .ok()
            .and_then(|index| self.colors.get(index).copied().flatten())
            .ok_or(ClearSkip::NotBound)?;
        Target::check(info, surface.layer)
    }
}

/// The depth and stencil part's target, aspects and values, once its checks passed.
struct DepthStencilPart {
    target: Target,
    /// The depth value, when depth is asked and the format has it.
    depth: Option<f32>,
    /// The stencil value, when stencil is asked and the format has it.
    stencil: Option<u32>,
    /// Whether stencil is written through a partial mask (a draw).
    partial: bool,
}

impl DepthStencilPart {
    fn value(&self) -> ClearValue {
        match (self.depth, self.stencil) {
            (Some(depth), Some(stencil)) => ClearValue::DepthStencil { depth, stencil },
            (Some(depth), None) => ClearValue::Depth(depth),
            (None, Some(stencil)) => ClearValue::Stencil(stencil),
            // `depth_stencil_part` refuses a part with neither.
            (None, None) => ClearValue::Depth(0.0),
        }
    }
}

/// The depth and stencil part's checks (module docs, step 5), in order: a bound depth target,
/// an aspect, the depth value, the target's slices and layers.
fn depth_stencil_part(
    bound: &Framebuffer,
    surface: &ClearSurface,
    values: &Values,
) -> Result<DepthStencilPart, ClearSkip> {
    let info = bound.depth.ok_or(ClearSkip::NotBound)?;
    let format = info.format;
    let use_depth = surface.depth && format.has_depth();
    let use_stencil = surface.stencil && format.has_stencil();
    if !use_depth && !use_stencil {
        return Err(ClearSkip::NoAspect);
    }
    let depth = if use_depth {
        if values.depth.is_nan() {
            return Err(ClearSkip::DepthNotANumber);
        }
        Some(values.depth.clamp(0.0, 1.0))
    } else {
        None
    };
    let target = Target::check(info, surface.layer)?;
    let partial =
        use_stencil && values.stencil_mask != FULL_STENCIL_MASK && values.stencil_mask != 0;
    Ok(DepthStencilPart {
        target,
        depth,
        stencil: use_stencil.then_some(values.stencil),
        partial,
    })
}

/// The four 32-bit patterns of a colour value, as the clear shader takes them.
fn value_bits(value: ClearValue) -> [u32; 4] {
    match value {
        ClearValue::Float(c) => c.map(f32::to_bits),
        ClearValue::Sint(c) => c.map(i32::cast_unsigned),
        ClearValue::Uint(c) => c,
        ClearValue::Depth(d) | ClearValue::DepthStencil { depth: d, .. } => [d.to_bits(), 0, 0, 0],
        ClearValue::Stencil(s) => [s, 0, 0, 0],
    }
}

/// The parameter block's binding.
fn params_binding(buffer: BufferId) -> Binding {
    Binding {
        group: host_shaders::clear::GROUP,
        binding: host_shaders::clear::PARAMS_BINDING,
        resource: BindingResource::Buffer(BufferSlice {
            buffer,
            offset: 0,
            size: u64::from(host_shaders::clear::PARAMS_SIZE),
        }),
    }
}

/// A draw-based clear's pipeline: no vertex input, one triangle list, no culling.
fn pipeline_desc(
    vertex: ShaderId,
    fragment: Option<ShaderId>,
    depth_stencil: DepthStencilState,
    blend: [BlendState; GUEST_COLOR_TARGETS],
    targets: TargetFormats,
) -> GraphicsPipelineDesc {
    GraphicsPipelineDesc {
        stages: Stages {
            vertex,
            tess_control: None,
            tess_eval: None,
            geometry: None,
            fragment,
        },
        vertex_input: VertexLayout::default(),
        topology: Topology::Triangles,
        primitive_restart: false,
        raster: RasterState {
            cull: CullMode::None,
            front_face: FrontFace::Ccw,
            polygon_mode: PolygonMode::Fill,
            depth_clamp: false,
            depth_bias: DepthBias::default(),
            line_width: 1.0,
        },
        depth_stencil,
        blend,
        logic_op: None,
        targets,
        emulation: EmulationFlags::EMPTY,
    }
}

/// The render crate's clear shader as a validated naga module, parsed from its WGSL.
fn clear_module() -> Result<(naga::Module, naga::valid::ModuleInfo), DrawError> {
    let module = naga::front::wgsl::parse_str(HostShader::Clear.source()).map_err(|e| {
        tracing::error!("the clear shader did not parse: {e}");
        DrawError::Internal("the clear shader did not parse")
    })?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|e| {
        tracing::error!("the clear shader did not validate: {e}");
        DrawError::Internal("the clear shader did not validate")
    })?;
    Ok((module, info))
}

/// Appends one planned part to `list`. A draw also sets its render targets and uploads its
/// parameter block first, and becomes the draw path's last targets.
fn append(
    list: &mut CommandList,
    plan: Plan,
    params: Option<BufferId>,
    last_targets: &mut Option<RenderTargets>,
) {
    match plan {
        Plan::Clear(desc) => list.push(Command::Clear(desc)),
        Plan::Draw {
            staging,
            targets,
            desc,
        } => {
            if let Some(buffer) = params {
                list.push(Command::Upload {
                    from: staging,
                    to: CopyRegion::Buffer(BufferSlice {
                        buffer,
                        offset: 0,
                        size: u64::from(host_shaders::clear::PARAMS_SIZE),
                    }),
                });
            }
            list.push(Command::SetRenderTargets(targets));
            *last_targets = Some(targets);
            list.push(Command::Draw(*desc));
        }
    }
}
