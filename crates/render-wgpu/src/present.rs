//! What window presentation decides, as pure functions (docs/specs/render-backend.md, "Present
//! targets" and "Presenting a guest framebuffer"; slice R8). The render thread
//! ([`crate::window`]) does the wgpu calls; every choice it makes is one of these, so each is
//! tested for every input combination with no GPU.
//!
//! For policy only, these follow Eden's Vulkan presenter (`eden/src/video_core/renderer_vulkan/`
//! @ cf654052: `vk_swapchain.cpp`, `vk_present_manager.cpp`, `vk_blit_screen.cpp`,
//! `present/layer.cpp`, `present/present_push_constants.h`), and Eden's window layout
//! (`eden/src/core/frontend/framebuffer_layout.cpp`); the code is new. Where they differ:
//!
//! - **Present mode** ([`present_mode`]): the spec's order, mailbox, then immediate, then FIFO.
//!   Eden reaches the same order from its default setting (`ChooseSwapPresentMode`: with FIFO
//!   asked for, mailbox if the surface has it, else immediate, else FIFO), but also honours a
//!   user's choice and FIFO-relaxed, which starquake has no setting for. A surface that lists
//!   none of the three is refused here, where Eden would ask for FIFO regardless; wgpu says FIFO
//!   is always listed, so this refusal is a guard, not a case seen. Pinned by
//!   `present::tests::the_present_mode_for_every_set_of_modes`.
//! - **Surface format** ([`surface_format`]): Eden's rule (`ChooseSwapSurfaceFormat`):
//!   `B8G8R8A8_UNORM` if listed, else the first format listed. Eden also checks the colour space
//!   is sRGB non-linear; wgpu's listed formats are those its automatic colour space accepts
//!   (`SurfaceCapabilities::formats`), so no check is needed. An empty list is refused.
//! - **Alpha** ([`alpha_mode`]): Eden's `ChooseAlphaFlags`: opaque, else inherit. Where the
//!   surface has neither, Eden logs and asks for opaque anyway; this refuses the surface,
//!   because wgpu would refuse that configuration (`wgpu-core` 30.0.2,
//!   `device/surface_config.rs`, `UnsupportedAlphaMode`). Pinned by
//!   `present::tests::the_alpha_mode_for_every_set_of_modes`.
//! - **The window layout** ([`fit`]): the largest rectangle with the shown image's aspect ratio
//!   that fits the window, centred, as Eden's `DefaultFrameLayout` and `MaxRectangle` place the
//!   screen; the rest of the window is cleared to black. Eden takes the aspect ratio from a
//!   setting (16:9 by default); starquake has no such setting and uses the presented
//!   rectangle's own, turned by the transform. Eden scales in `float` and rounds; this computes
//!   in whole numbers, rounding half up, so the result never depends on float rounding. Pinned
//!   by `present::tests::fit_letterboxes_and_pillarboxes`.
//! - **Filtering** ([`filter`]): linear, as Eden's default scaling filter, except that an image
//!   shown at its own size is sampled with the nearest filter (both give the same pixels at the
//!   texel centres a 1:1 blit samples, in exact arithmetic; the nearest filter gives them in a
//!   GPU's arithmetic too), and a format the device cannot filter is always sampled with the
//!   nearest filter (`host_shaders::present::TEXTURE_BINDING`'s note).
//! - **Transforms** ([`blit_params`]): Eden's `NormalizeCrop` swaps the crop's left and right for
//!   a horizontal flip and its top and bottom for a vertical one, and leaves rotations
//!   unimplemented. Rotations here turn the image clockwise by the named angle, as
//!   `PresentTransform` defines them; which corner of the source lands where is this module's
//!   choice (a guess at what a title expects, to be checked against a title that rotates),
//!   pinned by `present::tests::each_transform_puts_each_source_corner_in_its_place`.
//! - **The matrix** ([`blit_params`]): Eden's orthographic matrix (`MakeOrthographicMatrix`)
//!   with its y row negated, as `present.wgsl`'s header says the backend must, because WGSL's
//!   clip space has y = -1 at the bottom where Vulkan's has it at the top.
//!
//! # The frame queue
//!
//! The newest frame wins ("Present targets"): [`frame_outcome`] shows a present only when it is
//! the last in its list and no list queued after it presents too; every other present is
//! counted as dropped. The render thread blits the shown one into the window at its place in
//! the list, so a command after it that writes the image cannot change what is shown ("Submit").
//! What the window then does with a shown frame (a mailbox swapchain may replace it before the
//! display reads it) wgpu does not report, so `PresentOutcome::Shown` means "handed to the
//! window", not "seen".

use crate::plan::TexelBox;
use starquake_render::command::{PresentTransform, Rect};
use starquake_render::host_shaders::present::{
    MATRIX_OFFSET, PARAMS_SIZE, VERTEX_COUNT, VERTEX_POSITION_OFFSET, VERTEX_STRIDE,
    VERTEX_TEX_COORD_OFFSET, VERTICES_OFFSET,
};
use starquake_render::{PresentOutcome, RenderError};

/// Bytes in the present blit's parameter block (`host_shaders::present::PARAMS_SIZE`).
pub const PARAMS_LEN: usize = PARAMS_SIZE as usize;

/// The present mode for a surface that lists `supported`: mailbox, else immediate, else FIFO
/// ("Present targets"), or `None` when it lists none of them (module docs).
pub fn present_mode(supported: &[wgpu::PresentMode]) -> Option<wgpu::PresentMode> {
    [
        wgpu::PresentMode::Mailbox,
        wgpu::PresentMode::Immediate,
        wgpu::PresentMode::Fifo,
    ]
    .into_iter()
    .find(|mode| supported.contains(mode))
}

/// The surface format among `supported`: `Bgra8Unorm`, else the first (Eden's rule, module
/// docs), or `None` when there is none.
pub fn surface_format(supported: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
    if supported.contains(&wgpu::TextureFormat::Bgra8Unorm) {
        Some(wgpu::TextureFormat::Bgra8Unorm)
    } else {
        supported.first().copied()
    }
}

/// The alpha mode among `supported`: opaque, else inherit (Eden's rule), or `None` (module
/// docs).
pub fn alpha_mode(supported: &[wgpu::CompositeAlphaMode]) -> Option<wgpu::CompositeAlphaMode> {
    [
        wgpu::CompositeAlphaMode::Opaque,
        wgpu::CompositeAlphaMode::Inherit,
    ]
    .into_iter()
    .find(|mode| supported.contains(mode))
}

/// The adapter cannot present to the window, or the surface lists no format, present mode or
/// alpha mode the backend uses, or cannot be rendered to ([`surface_choice`]).
pub const SURFACE_UNSUPPORTED: &str = "the GPU adapter cannot present to this window";

/// How a window's surface is configured, chosen from what it lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceChoice {
    /// [`surface_format`]'s choice.
    pub format: wgpu::TextureFormat,
    /// [`present_mode`]'s choice.
    pub present_mode: wgpu::PresentMode,
    /// [`alpha_mode`]'s choice.
    pub alpha_mode: wgpu::CompositeAlphaMode,
}

/// The configuration for a surface whose capabilities are `caps`: refused with
/// [`SURFACE_UNSUPPORTED`] when any of the three choices has nothing to choose from, or the
/// surface cannot be a render attachment (wgpu says it always can; a guard, as for FIFO).
pub fn surface_choice(caps: &wgpu::SurfaceCapabilities) -> Result<SurfaceChoice, RenderError> {
    let refused = RenderError::Unsupported(SURFACE_UNSUPPORTED);
    if !caps.usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
        return Err(refused);
    }
    Ok(SurfaceChoice {
        format: surface_format(&caps.formats).ok_or(refused.clone())?,
        present_mode: present_mode(&caps.present_modes).ok_or(refused.clone())?,
        alpha_mode: alpha_mode(&caps.alpha_modes).ok_or(refused)?,
    })
}

/// The surface's frame latency hint: wgpu's default (`Surface::get_default_config`). A guess,
/// to measure: wgpu notes that under mailbox on Metal and Vulkan, 2 limits the frame rate to
/// twice the display's (`wgpu-types-30.0.2/src/surface.rs`, `desired_maximum_frame_latency`),
/// which is more than the guest's 60 Hz on any display.
pub const FRAME_LATENCY: u32 = 2;

/// The formats views of a target in `format` may take besides `format` itself: its sRGB
/// counterpart, when it has one. The blit writes through the counterpart when the presented
/// view's sRGB-ness differs from the target's ([`write_format`]).
pub fn view_formats(format: wgpu::TextureFormat) -> Vec<wgpu::TextureFormat> {
    let other = write_format(format, !format.is_srgb());
    if other == format {
        Vec::new()
    } else {
        vec![other]
    }
}

/// The format the blit writes a target in `format` as: its sRGB form when the presented view is
/// sRGB, its linear form otherwise. With both sides alike, a pixel is decoded and encoded
/// again by the same curve, so an 8-bit pixel reaches the window as the guest wrote it. A
/// target format with no sRGB counterpart is written as it is.
pub fn write_format(target: wgpu::TextureFormat, srgb: bool) -> wgpu::TextureFormat {
    if srgb {
        target.add_srgb_suffix()
    } else {
        target.remove_srgb_suffix()
    }
}

/// The size to configure a surface at for a window of `window` physical pixels: `None` when
/// either side is zero (nothing can be shown; the frame is dropped), else each side clamped to
/// `max_dimension`, the device's largest 2D texture, which wgpu refuses to go past (`wgpu-core`
/// 30.0.2, `device/surface_config.rs`, `TooLarge`). A clamped surface is scaled to the window by
/// the platform (wgpu's note on `SurfaceConfiguration::width`).
pub fn surface_size(window: (u32, u32), max_dimension: u32) -> Option<(u32, u32)> {
    let (width, height) = window;
    if width == 0 || height == 0 {
        return None;
    }
    Some((width.min(max_dimension), height.min(max_dimension)))
}

/// Where in the target the image goes, in pixels from the target's top-left corner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenRect {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

/// The size an image of `rect` is shown at before scaling: its own, or turned on its side by a
/// quarter turn.
pub fn shown_size(rect: Rect, transform: PresentTransform) -> (u32, u32) {
    match transform {
        PresentTransform::Rotate90 | PresentTransform::Rotate270 => (rect.height, rect.width),
        PresentTransform::Identity
        | PresentTransform::FlipHorizontal
        | PresentTransform::FlipVertical
        | PresentTransform::Rotate180 => (rect.width, rect.height),
    }
}

/// The largest rectangle of `shown`'s aspect ratio that fits in `target`, centred (module
/// docs); `None` when either has a zero side. The scaled side is rounded half up and is at
/// least one pixel.
pub fn fit(shown: (u32, u32), target: (u32, u32)) -> Option<ScreenRect> {
    let (sw, sh) = (u64::from(shown.0), u64::from(shown.1));
    let (tw, th) = (u64::from(target.0), u64::from(target.1));
    if sw == 0 || sh == 0 || tw == 0 || th == 0 {
        return None;
    }
    // Products of two u32s fit in u64; so do the sums below (each under 2^64 - 2^32).
    let (width, height) = if tw * sh <= th * sw {
        // The target is relatively narrower: full width.
        (tw, ((tw * sh + sw / 2) / sw).max(1))
    } else {
        (((th * sw + sh / 2) / sh).max(1), th)
    };
    let side = |v: u64| u32::try_from(v).ok();
    Some(ScreenRect {
        x: side((tw - width) / 2)?,
        y: side((th - height) / 2)?,
        width: side(width)?,
        height: side(height)?,
    })
}

/// The filter the blit samples with (module docs): nearest when the image is shown at its own
/// size or the format cannot be filtered, linear otherwise.
pub fn filter(shown: (u32, u32), screen: ScreenRect, filterable: bool) -> wgpu::FilterMode {
    if !filterable || (screen.width, screen.height) == shown {
        wgpu::FilterMode::Nearest
    } else {
        wgpu::FilterMode::Linear
    }
}

/// The texture coordinates of the presented rectangle within the texture the blit samples,
/// which holds `source` (the rectangle rounded out to whole blocks, from its top-left corner):
/// from 0 at the left and top to `right` and `bottom`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crop {
    /// The right edge, 0 to 1.
    pub right: f32,
    /// The bottom edge, 0 to 1.
    pub bottom: f32,
}

impl Crop {
    /// The crop of `rect` in the block-rounded `source` that starts at its corner.
    pub fn new(rect: Rect, source: TexelBox) -> Self {
        Self {
            right: ratio(rect.width, source.extent.width),
            bottom: ratio(rect.height, source.extent.height),
        }
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "sizes here are window and texture sides, at most the device's 2D texture limit \
              (16384 under wgpu on Metal), inside f32's exact range of 2^24; a larger one \
              rounds, which moves a vertex or a coordinate by a fraction of a pixel"
)]
fn float(v: u32) -> f32 {
    v as f32
}

/// `part / whole`, or 0 for an empty whole.
fn ratio(part: u32, whole: u32) -> f32 {
    if whole == 0 {
        0.0
    } else {
        float(part) / float(whole)
    }
}

/// The texture coordinate each corner of the screen rectangle samples, in the blit's vertex
/// order (top-left, top-right, bottom-left, bottom-right, Eden's `SetVertexData`), for
/// `transform` (module docs).
pub fn corners(crop: Crop, transform: PresentTransform) -> [[f32; 2]; 4] {
    let (l, t, r, b) = (0.0, 0.0, crop.right, crop.bottom);
    match transform {
        PresentTransform::Identity => [[l, t], [r, t], [l, b], [r, b]],
        PresentTransform::FlipHorizontal => [[r, t], [l, t], [r, b], [l, b]],
        PresentTransform::FlipVertical => [[l, b], [r, b], [l, t], [r, t]],
        PresentTransform::Rotate90 => [[l, b], [l, t], [r, b], [r, t]],
        PresentTransform::Rotate180 => [[r, b], [l, b], [r, t], [l, t]],
        PresentTransform::Rotate270 => [[r, t], [r, b], [l, t], [l, b]],
    }
}

/// The present blit's parameter block (`present.wgsl`'s `PresentParams`) for a target of
/// `target` pixels showing `crop` of the sampled texture in `screen`, turned by `transform`.
/// Little-endian `f32`s at the offsets `host_shaders::present` names: the matrix maps pixel
/// (0, 0) to the top-left of WGSL's clip space (module docs), and the vertices are the screen
/// rectangle's corners in pixels with their texture coordinates.
pub fn blit_params(
    target: (u32, u32),
    screen: ScreenRect,
    crop: Crop,
    transform: PresentTransform,
) -> [u8; PARAMS_LEN] {
    let mut out = [0; PARAMS_LEN];
    let mut put = |offset: u32, value: f32| {
        let at = usize::try_from(offset).unwrap_or(usize::MAX);
        if let Some(slot) = out.get_mut(at..at.saturating_add(4)) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
    };
    // Column-major: column 0 scales x, column 1 scales y (negated), column 3 translates.
    let matrix = [
        [2.0 / float(target.0), 0.0, 0.0, 0.0],
        [0.0, -2.0 / float(target.1), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0, 1.0],
    ];
    for (c, column) in (0u32..).zip(matrix) {
        for (r, value) in (0u32..).zip(column) {
            put(MATRIX_OFFSET + (c * 4 + r) * 4, value);
        }
    }
    let (x, y) = (float(screen.x), float(screen.y));
    let (w, h) = (float(screen.width), float(screen.height));
    let positions = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]];
    let tex = corners(crop, transform);
    for (i, (position, tex)) in (0..VERTEX_COUNT).zip(positions.into_iter().zip(tex)) {
        let vertex = VERTICES_OFFSET + i * VERTEX_STRIDE;
        put(vertex + VERTEX_POSITION_OFFSET, position[0]);
        put(vertex + VERTEX_POSITION_OFFSET + 4, position[1]);
        put(vertex + VERTEX_TEX_COORD_OFFSET, tex[0]);
        put(vertex + VERTEX_TEX_COORD_OFFSET + 4, tex[1]);
    }
    out
}

/// Whether a present to a window is shown or dropped (the frame queue, module docs): shown only
/// when it is the last present of its list and no list queued after it presents.
pub fn frame_outcome(last_in_list: bool, newer_queued: bool) -> PresentOutcome {
    if last_in_list && !newer_queued {
        PresentOutcome::Shown
    } else {
        PresentOutcome::Dropped
    }
}

/// What `Surface::get_current_texture` answered (`wgpu::CurrentSurfaceTexture`, without the
/// texture).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Acquired {
    /// A texture, as configured.
    Success,
    /// A texture that no longer matches the surface.
    Suboptimal,
    /// No texture in time.
    Timeout,
    /// The window is hidden.
    Occluded,
    /// The configuration no longer fits the surface.
    Outdated,
    /// The surface is gone.
    Lost,
    /// wgpu reported a validation error (counted as a host error by the device's handler).
    Validation,
}

/// What the render thread does after an [`Acquired`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquireAction {
    /// Draw into the texture and present it; reconfigure before the next frame when `reconfigure`.
    Show {
        /// Whether the surface must be configured again before the next frame.
        reconfigure: bool,
    },
    /// Configure the surface again and ask once more.
    Retry,
    /// Drop this frame and keep the surface.
    Drop,
    /// Drop this frame and let the surface go: later presents are dropped until the frontend
    /// sets a window target again (a lost surface must be made again from the window, which on
    /// macOS only the main thread may do).
    Forget,
}

/// The action for `acquired`, where `retried` says whether this is the second ask for this
/// frame. Wgpu's own guidance per answer (`wgpu-30.0.2/src/api/surface_texture.rs`): use a
/// suboptimal texture and reconfigure; skip on a timeout or an occluded window; reconfigure and
/// try again when outdated (once here, so a surface that stays outdated drops the frame instead
/// of looping); make the surface again when lost.
pub fn acquire_action(acquired: Acquired, retried: bool) -> AcquireAction {
    match acquired {
        Acquired::Success => AcquireAction::Show { reconfigure: false },
        Acquired::Suboptimal => AcquireAction::Show { reconfigure: true },
        Acquired::Outdated if !retried => AcquireAction::Retry,
        Acquired::Outdated | Acquired::Timeout | Acquired::Occluded | Acquired::Validation => {
            AcquireAction::Drop
        }
        Acquired::Lost => AcquireAction::Forget,
    }
}

/// Whether a rectangle at `origin` starts on a block of a format whose blocks are `block`
/// texels: the window blit copies from there in whole blocks, and wgpu refuses a copy of a
/// compressed format that starts inside a block (`wgpu-core` 30.0.2, `command/transfer.rs`,
/// `UnalignedCopyOriginX` and `UnalignedCopyOriginY`). The shared present checks allow such a
/// rectangle, so a present of one to a window is dropped (`crate::window`). A zero block side
/// (no format has one) counts as unaligned.
pub fn starts_on_block(origin: (u32, u32), block: (u32, u32)) -> bool {
    origin.0.checked_rem(block.0) == Some(0) && origin.1.checked_rem(block.1) == Some(0)
}

/// Whether a window surface may be made on this thread: always, except where the platform
/// attaches the window's layer, which must happen on the main thread (macOS; wgpu panics
/// otherwise, `wgpu-30.0.2/src/api/surface.rs`, `SurfaceTarget`).
pub fn may_make_surface(needs_main_thread: bool, on_main_thread: bool) -> bool {
    !needs_main_thread || on_main_thread
}

#[cfg(test)]
mod tests;
