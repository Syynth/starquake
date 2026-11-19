//! A present's image (the executor's module docs, "Presents"): the [`ImageDesc`] gsv latched,
//! as the texture cache's [`ImageInfo`] and the header [`TextureCache::texture`] takes.
//!
//! The image is what vi2 learns from a native image's preference words (graphics-server, "What
//! vi2 needs from the image"; row 48 of `eden-nx2/NX2_NOTES.md`): a block-linear 2D image of
//! `width` by `height` pixels, one level and one layer, single-sampled, of `2^block_height_log2`
//! GOBs per block in height and one in depth, in the format the word names: 1 is RGBA8 and 4 is
//! RGB10A2 (the same section). Which of the render crate's formats those are is a **guess**:
//! [`PixelFormat::A8B8G8R8Unorm`] and [`PixelFormat::A2B10G10R10Unorm`], the formats Nvidia names
//! with red in the low bits, as the colour targets the guest renders its frames into use. A
//! trace of a presented frame's colour target format confirms it. The GOB is the colour GOB
//! ([`gob_for`]), as for any colour image.
//!
//! The row pitch (`pitch`, in 64-byte GOBs) is not read: a block-linear image's rows are as
//! wide as its width in GOBs, and the texture cache's identity for a block-linear image has no
//! pitch. A pitch wider than the width (padding) would place rows where this image does not; a
//! **guess** that NVN2 presents images whose pitch is their width rounded up to a GOB.
//!
//! [`TextureCache::texture`]: crate::back::texture::TextureCache::texture

use starquake_layout::{ImageDim, SampleLayout};
use starquake_render::PixelFormat;
use starquake_render::resource::{Swizzle, ViewDimension};

use super::ExecError;
use crate::back::texture::{ImageInfo, MemoryLayout, TextureHeader, ViewInfo, gob_for};
use crate::interface::ImageDesc;

/// The format word for RGBA8 (graphics-server, "What vi2 needs from the image").
pub const PRESENT_FORMAT_RGBA8: u32 = 1;

/// The format word for RGB10A2 (the same section).
pub const PRESENT_FORMAT_RGB10A2: u32 = 4;

/// The render crate's format of a present's format word (module docs).
pub fn present_format(format: u32) -> Result<PixelFormat, ExecError> {
    match format {
        PRESENT_FORMAT_RGBA8 => Ok(PixelFormat::A8B8G8R8Unorm),
        PRESENT_FORMAT_RGB10A2 => Ok(PixelFormat::A2B10G10R10Unorm),
        format => Err(ExecError::PresentFormat { format }),
    }
}

/// The image a present shows (module docs). A block height the layout crate cannot hold is
/// refused by the texture cache when it is made.
pub fn image(desc: &ImageDesc) -> Result<ImageInfo, ExecError> {
    let format = present_format(desc.format)?;
    Ok(ImageInfo {
        address: desc.address,
        format,
        dim: ImageDim::D2,
        width: desc.width,
        height: desc.height,
        depth: 1,
        layers: 1,
        levels: 1,
        samples: SampleLayout::S1x1,
        layout: MemoryLayout::BlockLinear {
            gob: gob_for(format),
            block_height_log2: u8::try_from(desc.block_height_log2).unwrap_or(u8::MAX),
            block_depth_log2: 0,
        },
    })
}

/// The header of `image` read whole: a 2D view of its level with the identity swizzle.
pub fn header(image: ImageInfo) -> TextureHeader {
    TextureHeader {
        image,
        view: ViewInfo {
            dimension: ViewDimension::D2,
            base_mip: 0,
            mip_count: 1,
            swizzle: Swizzle::IDENTITY,
        },
    }
}
