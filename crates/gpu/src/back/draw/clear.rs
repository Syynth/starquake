//! Clears: the 3D engine's `CLEAR_SURFACE` turned into the render backend's
//! [`Command::Clear`], with the write masks the guest asks for (gpu-protocol spec, "Clears" and
//! slice G24; decided by the owner, 2026-11-19, rulings 9 and 6 of the two "Owner rulings"
//! entries).
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
//! the checks it does not make (a slice or layer past the target, a NaN depth value) are listed
//! under "Divergences from Eden-NX2" and "Guesses".
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
//!    formula ([`color_value`]). The clear's colour mask is the channels asked
//!    ([`ClearDesc::color_mask`]), all four or some, as Eden's `blit_image.ClearColor` writes
//!    through them.
//! 5. **Depth and stencil**: the depth target G18 binds (`SET_ZT_SELECT`'s count not 0, Eden's
//!    `FindDepthBuffer`), the aspects asked that its format has (Eden's `HasAspectDepthBit`,
//!    `HasAspectStencilBit`; none is [`ClearSkip::NoAspect`]). Depth is `SET_Z_CLEAR_VALUE`,
//!    stencil `SET_STENCIL_CLEAR_VALUE`. A stencil clear whose `SET_STENCIL_MASK` is neither
//!    `0xff` nor 0 writes through that mask ([`ClearDesc::stencil_mask`]), as Eden's
//!    `blit_image.ClearDepthStencil`; otherwise it writes every stencil bit, as Eden's
//!    `ClearAttachments` does (a mask of 0 is a full clear, as Eden's `stencil_partial` test
//!    has it).
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
//! # Masks and partial areas are the render backend's
//!
//! Every part that is not skipped is one [`Command::Clear`], whatever its masks and area:
//! [`ClearDesc`] carries the colour channels and stencil bits a clear writes, and the render
//! backend draws a clear its host API cannot perform as one (a partial mask, or a rectangle
//! over a view it clears only whole, such as a depth, stencil or multisampled view under wgpu)
//! with its own clear helper shader (render-backend, "Clears carry write masks"; decided by the
//! owner, 2026-11-19). So this module draws no clear, makes no host object, and does not parse
//! the render crate's `clear.wgsl`; a clear sets no render targets, and the next guest draw
//! keeps the targets the draw path last set. `tests::golden_masked`,
//! `tests::golden_masked_stencil`, `tests::golden_scissored_depth`,
//! `tests::golden_scissored_msaa`.
//!
//! The masks are set for every part, read or not: a colour clear writes every stencil bit and
//! a depth and stencil clear every colour channel, which the backend does not read (the
//! [`ClearDesc`] docs), so an unmasked part is [`ClearDesc::unmasked`]'s masks
//! (`tests::an_unmasked_part_carries_full_masks`).
//!
//! # The spec's offline test
//!
//! gpu-protocol's slice table gives G24's test as "a whole-target clear of colour, depth and
//! stencil each, and a masked and a scissored clear", and since the owner's ruling a masked
//! clear is "a `Command::Clear` with its write masks" (the table's G24 row). Here a masked
//! clear and a scissored clear are each one [`Command::Clear`]: `tests::golden_masked`,
//! `tests::golden_masked_stencil` and `tests::golden_scissored`.
//!
//! # Errors and what they leave
//!
//! Every check that can skip a part comes before the targets are resolved. A part whose target
//! cannot be had fails the clear ([`DrawError`]) before any clear command is appended, so a
//! clear appends its commands whole or not at all
//! (`tests::a_failed_part_appends_no_part_of_the_clear`). What resolving did stays: uploads the
//! texture cache appended and a target marked GPU-modified, as for a draw (`back::draw`,
//! "Errors and what they leave").
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
//! - **A masked colour clear carries a write mask**, where Eden draws with the mask as the blend
//!   constant (`ClearColor`'s `SetBlendConstants`), which wgpu cannot do for an integer target;
//!   the render backend writes through the mask, and for a float target the pixels are the
//!   same. `a_masked_colour_clear_writes_through_its_mask`.
//! - **A masked colour clear's value is the converted value** for an integer target, the same
//!   value a full clear writes; Eden's colour clear shader writes the floats as they are.
//!   `a_masked_integer_clear_writes_the_converted_value`.
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
//!   `stencil_compare_mask`); its compare is `ALWAYS`, which reads no bit, and a
//!   [`ClearDesc`] has no compare mask, so a title's stencil-test state changes nothing in a
//!   clear. The pixels are the same. `the_stencil_func_mask_changes_nothing_in_a_clear`.
//! - **`SET_SURFACE_CLIP` does not bound the render area.** Eden's framebuffer starts its render
//!   area at the surface clip's width and height (`render_targets.size` in
//!   `UpdateRenderTargets`) before taking the minimum over its attachments; here only the
//!   targets bound it, so a surface clip smaller than the targets clears more than Eden (a
//!   listed divergence the owner kept, ruling 6 of 2026-11-19).
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
//!   are raw bits for an integer target. Eden's formula stays a **guess** until a trace of a
//!   title clearing an integer target settles it (gpu-protocol, "Clears"; the owner's ruling 6
//!   of 2026-11-19). `integer_clear_values_follow_edens_formula`.
//! - **A 3D target deeper than one slice is skipped** ([`ClearSkip::VolumeTarget`]): which
//!   slice `RT_ARRAY_INDEX` names, and how the backend would clear one slice, is not built.

#[cfg(test)]
mod tests;

use starquake_layout::ImageDim;
use starquake_render::caps::GUEST_COLOR_TARGETS;
use starquake_render::command::{ALL_STENCIL_BITS, ClearDesc, ClearValue, Rect};
use starquake_render::pipeline::ColorWrites;
use starquake_render::{Command, PixelFormat};

use super::{DrawContext, DrawError, fixed, targets};
use crate::back::state::{COLOR_TARGETS, Mirror};
use crate::back::texture::ImageInfo;
use crate::classes::{Method, clc797};
use crate::memory::GpuMemory;

/// The methods whose words start a clear: the executor drains CPU writes and flushes inline
/// writes before them, as before a draw (`back::exec`, "Before each draw, dispatch and copy").
pub const CLEAR_METHODS: [&Method; 1] = [&clc797::CLEAR_SURFACE];

/// The value of a full `SET_STENCIL_MASK`: Eden's `stencil_partial` test compares with `0xFF`.
pub const FULL_STENCIL_MASK: u32 = 0xff;

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
}

/// What became of one part (colour, or depth and stencil) of a clear.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// A [`Command::Clear`] was appended.
    Cleared,
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
    stencil_mask: u8,
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
            stencil_mask: u8::try_from(
                mirror
                    .reg(&clc797::SET_STENCIL_MASK, &[])?
                    .field(clc797::SET_STENCIL_MASK_V)?,
            )
            .map_err(|_| DrawError::Internal("an 8-bit register field held more than 8 bits"))?,
        })
    }
}

/// The clear path (module docs): what clears have done.
#[derive(Debug, Default)]
pub(crate) struct Clears {
    counters: ClearCounters,
}

/// What every part of one clear shares.
struct Common {
    /// The clear area, clamped to the render area.
    area: Rect,
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
}

impl Clears {
    /// Nothing done yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// What clears have done so far.
    pub fn counters(&self) -> ClearCounters {
        self.counters
    }

    /// One word `word` written to `CLEAR_SURFACE`, after `mirror` stored it (module docs).
    /// `execute` is conditional rendering's answer.
    pub(crate) fn clear<M: GpuMemory + ?Sized>(
        &mut self,
        ctx: &mut DrawContext<'_, M>,
        mirror: &Mirror,
        word: u32,
        execute: bool,
    ) -> Result<Cleared, DrawError> {
        let outcome = run(ctx, mirror, word, execute);
        self.counters.count(&outcome);
        if let Err(error) = &outcome {
            tracing::debug!(%error, "a clear could not be done");
        }
        outcome
    }
}

/// [`Clears::clear`]'s steps (module docs).
fn run<M: GpuMemory + ?Sized>(
    ctx: &mut DrawContext<'_, M>,
    mirror: &Mirror,
    word: u32,
    execute: bool,
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

    // Step 4: the checks.
    let color = color_asked.then(|| {
        bound.color_target(&surface).map(|target| {
            let value = color_value(target.info.format, values.color);
            (target, value)
        })
    });
    // Step 5: the checks.
    let depth_stencil = ds_asked.then(|| depth_stencil_part(&bound, &surface, &values));

    // Every check passed or skipped; resolve the targets, appending no clear.
    let common = Common { area };
    let mut clears: Vec<ClearDesc> = Vec::with_capacity(2);
    let outcome = Cleared {
        color: match color {
            None => None,
            Some(Err(skip)) => Some(Part::Skipped(skip)),
            Some(Ok((target, value))) => Some(plan_color(
                ctx,
                &mut clears,
                &target,
                value,
                surface.color,
                &common,
            )?),
        },
        depth_stencil: match depth_stencil {
            None => None,
            Some(Err(skip)) => Some(Part::Skipped(skip)),
            Some(Ok(part)) => Some(plan_depth_stencil(ctx, &mut clears, &part, &common)?),
        },
    };

    // Nothing below can fail.
    for clear in clears {
        ctx.list.push(Command::Clear(clear));
    }
    Ok(outcome)
}

/// Resolves the colour part's target and plans its clear through `mask`.
fn plan_color<M: GpuMemory + ?Sized>(
    ctx: &mut DrawContext<'_, M>,
    clears: &mut Vec<ClearDesc>,
    target: &Target,
    value: ClearValue,
    mask: ColorWrites,
    common: &Common,
) -> Result<Part, DrawError> {
    let view = ctx
        .textures
        .render_target_layer(ctx.memory, &target.info, target.layer, ctx.list)?
        .view;
    clears.push(ClearDesc {
        color_mask: mask,
        ..ClearDesc::unmasked(view, value, target.rect(common.area))
    });
    Ok(Part::Cleared)
}

/// Resolves the depth and stencil part's target and plans its clear: a partial stencil mask
/// writes through it, and an unmasked stencil (a mask of `0xff` or 0) writes every bit.
fn plan_depth_stencil<M: GpuMemory + ?Sized>(
    ctx: &mut DrawContext<'_, M>,
    clears: &mut Vec<ClearDesc>,
    part: &DepthStencilPart,
    common: &Common,
) -> Result<Part, DrawError> {
    let target = &part.target;
    let view = ctx
        .textures
        .render_target_layer(ctx.memory, &target.info, target.layer, ctx.list)?
        .view;
    clears.push(ClearDesc {
        stencil_mask: part.stencil_mask,
        ..ClearDesc::unmasked(view, part.value(), target.rect(common.area))
    });
    Ok(Part::Cleared)
}

/// A scissor that does not bound the clear: Eden's `default_scissor`, the largest `i32` in each
/// extent.
const WHOLE: Rect = Rect {
    x: 0,
    y: 0,
    width: i32::MAX.unsigned_abs(),
    height: i32::MAX.unsigned_abs(),
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

/// The depth and stencil part's target, aspects, values and stencil mask, once its checks
/// passed.
struct DepthStencilPart {
    target: Target,
    /// The depth value, when depth is asked and the format has it.
    depth: Option<f32>,
    /// The stencil value, when stencil is asked and the format has it.
    stencil: Option<u32>,
    /// The stencil bits written ([`ClearDesc::stencil_mask`]).
    stencil_mask: u8,
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
    Ok(DepthStencilPart {
        target,
        depth,
        stencil: use_stencil.then_some(values.stencil),
        stencil_mask: stencil_write_mask(use_stencil, values.stencil_mask),
    })
}

/// The stencil bits a depth and stencil clear writes (module docs, step 5): a partial
/// `SET_STENCIL_MASK` (neither [`FULL_STENCIL_MASK`] nor 0, Eden's `stencil_partial`) when
/// stencil is written, and every bit otherwise.
fn stencil_write_mask(use_stencil: bool, mask: u8) -> u8 {
    let partial = use_stencil && u32::from(mask) != FULL_STENCIL_MASK && mask != 0;
    if partial { mask } else { ALL_STENCIL_BITS }
}
