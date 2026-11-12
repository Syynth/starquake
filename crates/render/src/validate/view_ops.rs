//! Checks of the commands that act on views: clears, blits and presents.
//!
//! A backend looks up the texture and view descriptions the command's view ids name and passes
//! them here, as it does for copies ([`super::Resolved`]). Each function returns the first
//! problem it finds, in the order its documentation lists.
//!
//! What each command touches (this crate's choice, which every backend follows):
//!
//! - a clear writes the view's base mip level, every layer of the view (every slice of a 3D
//!   texture's level), within its rectangle or the whole level;
//! - a blit reads the source view's base mip level at its base layer and writes the destination
//!   view's base mip level at its base layer (slice 0 of a 3D texture);
//! - a present reads the view's base mip level at its base layer.
//!
//! Rules not taken from the device's limits, with where each comes from:
//!
//! - A cleared texture needs the render-target usage: guest clears are 3D-engine clears of bound
//!   render targets (this crate's choice).
//! - A compressed texture cannot be cleared (Vulkan's `vkCmdClearColorImage` refuses compressed
//!   images; guess: from memory of the Vulkan specification, to be confirmed by slice R5).
//! - A clear value suits its view's format: a float colour for normalised and float formats, an
//!   unsigned or signed integer colour for those integer formats, depth for depth formats,
//!   stencil for stencil formats, and depth, stencil or both for depth-stencil formats.
//! - A depth clear value (`ClearValue::Depth`, or the depth of `ClearValue::DepthStencil`) lies
//!   in `0.0..=1.0`, and NaN is refused (Vulkan's VUID-VkClearDepthStencilValue-depth-00022,
//!   without `VK_EXT_depth_range_unrestricted`; guess: from memory of the Vulkan specification,
//!   to be confirmed by slice R5). `-0.0` is accepted: it compares equal to `0.0`, so it lies
//!   within the range (this crate's reading of the rule).
//! - Blit and present sources and blit destinations are single-sampled (Vulkan's
//!   `vkCmdBlitImage` rule; guess, as above); blit sides store the same aspect, are both integer
//!   or both not, of the same signedness, and a depth or stencil blit has identical formats and
//!   the nearest filter (Vulkan's blit rules; guess, as above).
//! - A blit destination is not compressed: compressed formats lack Vulkan's `BLIT_DST` format
//!   feature, and a render-pass blit (wgpu) cannot target a BC, ETC2 or ASTC texture (guess:
//!   from memory of both, to be confirmed by slice R5). A compressed source is allowed.
//! - An integer colour blit uses the nearest filter: Vulkan's
//!   VUID-vkCmdBlitImage-filter-02001 needs the linear-filter format feature, which integer
//!   formats lack, and wgpu cannot filter integer textures (guess: from memory, as above).
//! - A presented view stores colour that is not integer (this crate's choice: the present blit
//!   samples the view as float into a unorm swapchain).
//! - Every rectangle is non-empty and lies within the view's base mip level.

use super::validate_rect;
use crate::command::{BlitDesc, ClearDesc, ClearValue, PresentDesc, Rect};
use crate::error::RenderError;
use crate::format::{FormatAspect, NumericType, PixelFormat};
use crate::resource::{Extent3d, FilterMode, TextureDesc, TextureUsage, ViewDesc};

/// A view with the description of the texture it views.
#[derive(Clone, Copy, Debug)]
pub struct ResolvedView<'a> {
    /// The viewed texture's description.
    pub texture: &'a TextureDesc,
    /// The view's description.
    pub view: &'a ViewDesc,
}

impl ResolvedView<'_> {
    /// The size of the view's base mip level.
    pub fn base_extent(&self) -> Result<Extent3d, RenderError> {
        self.texture
            .mip_extent(self.view.base_mip)
            .ok_or(RenderError::NotBelow {
                what: "mip level",
                index: self.view.base_mip.into(),
                count: self.texture.mip_levels.into(),
            })
    }

    fn rect(&self, rect: &Rect) -> Result<(), RenderError> {
        if rect.width == 0 || rect.height == 0 {
            return Err(RenderError::Invalid("a rectangle must not be empty"));
        }
        let e = self.base_extent()?;
        validate_rect(e.width, e.height, rect)
    }

    fn usage(
        &self,
        needed: TextureUsage,
        resource: &'static str,
        usage: &'static str,
    ) -> Result<(), RenderError> {
        if self.texture.usage.contains(needed) {
            Ok(())
        } else {
            Err(RenderError::MissingUsage { resource, usage })
        }
    }

    fn single_sampled(&self, what: &'static str) -> Result<(), RenderError> {
        if self.texture.sample_count == 1 {
            Ok(())
        } else {
            Err(RenderError::Invalid(what))
        }
    }
}

/// Whether `value` suits a view of `format`.
fn value_suits(format: PixelFormat, value: ClearValue) -> bool {
    match (format.aspect(), value) {
        (FormatAspect::Color, ClearValue::Float(_)) => matches!(
            format.numeric(),
            NumericType::Unorm | NumericType::Snorm | NumericType::Float
        ),
        (FormatAspect::Color, ClearValue::Uint(_)) => format.numeric() == NumericType::Uint,
        (FormatAspect::Color, ClearValue::Sint(_)) => format.numeric() == NumericType::Sint,
        (FormatAspect::Depth, ClearValue::Depth(_))
        | (FormatAspect::Stencil, ClearValue::Stencil(_))
        | (
            FormatAspect::DepthStencil,
            ClearValue::Depth(_) | ClearValue::Stencil(_) | ClearValue::DepthStencil { .. },
        ) => true,
        _ => false,
    }
}

/// Whether a clear value's depth, if it has one, lies in `0.0..=1.0` (module docs). NaN does
/// not; `-0.0` does.
fn depth_in_range(value: ClearValue) -> bool {
    match value {
        ClearValue::Depth(depth) | ClearValue::DepthStencil { depth, .. } => {
            (0.0..=1.0).contains(&depth)
        }
        ClearValue::Float(_)
        | ClearValue::Uint(_)
        | ClearValue::Sint(_)
        | ClearValue::Stencil(_) => true,
    }
}

/// Checks a clear, in the order: the texture's render-target usage, a compressed format, the
/// value against the view's format, the depth value's range, the rectangle.
pub fn validate_clear(target: ResolvedView<'_>, clear: &ClearDesc) -> Result<(), RenderError> {
    target.usage(
        TextureUsage::RENDER_TARGET,
        "cleared texture",
        "render target",
    )?;
    let format = target.view.format;
    if format.is_compressed() {
        return Err(RenderError::Invalid(
            "a compressed texture cannot be cleared",
        ));
    }
    if !value_suits(format, clear.value) {
        return Err(RenderError::Invalid(
            "the clear value does not suit the view's format",
        ));
    }
    if !depth_in_range(clear.value) {
        return Err(RenderError::Invalid(
            "a depth clear value must be between 0 and 1",
        ));
    }
    match &clear.rect {
        Some(rect) => target.rect(rect),
        None => target.base_extent().map(|_| ()),
    }
}

/// Checks a blit, in the order: source and destination usages, sample counts, aspects, integer
/// kinds, depth and stencil rules, a compressed destination, the filter of an integer blit, the
/// source rectangle, the destination rectangle.
pub fn validate_blit(
    src: ResolvedView<'_>,
    dst: ResolvedView<'_>,
    blit: &BlitDesc,
) -> Result<(), RenderError> {
    src.usage(TextureUsage::COPY_SRC, "blit source texture", "copy source")?;
    dst.usage(
        TextureUsage::COPY_DST,
        "blit destination texture",
        "copy destination",
    )?;
    src.single_sampled("a blit source must have one sample")?;
    dst.single_sampled("a blit destination must have one sample")?;
    let (s, d) = (src.view.format, dst.view.format);
    if s.aspect() != d.aspect() {
        return Err(RenderError::Invalid(
            "a blit needs formats with the same aspect",
        ));
    }
    if s.is_integer() != d.is_integer() || s.is_signed_integer() != d.is_signed_integer() {
        return Err(RenderError::Invalid(
            "a blit needs formats of the same integer kind",
        ));
    }
    if s.aspect() != FormatAspect::Color {
        if s != d {
            return Err(RenderError::Invalid(
                "a depth or stencil blit needs identical formats",
            ));
        }
        if blit.filter != FilterMode::Nearest {
            return Err(RenderError::Invalid(
                "a depth or stencil blit needs the nearest filter",
            ));
        }
    }
    if d.is_compressed() {
        return Err(RenderError::Invalid(
            "a blit destination must not be compressed",
        ));
    }
    if s.is_integer() && blit.filter != FilterMode::Nearest {
        return Err(RenderError::Invalid(
            "an integer blit needs the nearest filter",
        ));
    }
    src.rect(&blit.src_rect)?;
    dst.rect(&blit.dst_rect)
}

/// Checks a present, in the order: the texture's copy-source usage, its sample count, the
/// view's aspect, an integer format, the rectangle.
pub fn validate_present(src: ResolvedView<'_>, present: &PresentDesc) -> Result<(), RenderError> {
    src.usage(TextureUsage::COPY_SRC, "presented texture", "copy source")?;
    src.single_sampled("a presented texture must have one sample")?;
    if src.view.format.aspect() != FormatAspect::Color {
        return Err(RenderError::Invalid("a presented view must store colour"));
    }
    if src.view.format.is_integer() {
        return Err(RenderError::Invalid(
            "a presented view must not store integers",
        ));
    }
    src.rect(&present.src_rect)
}

#[cfg(test)]
mod tests;
