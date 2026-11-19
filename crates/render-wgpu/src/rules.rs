//! What wgpu refuses beyond the shared checks (`starquake_render::validate`), and the host
//! shapes the backend gives wgpu.
//!
//! The shared checks follow Vulkan's rules and say where wgpu may refuse more. These are those
//! extra refusals, each taken from wgpu-core 30.0.2's own check (cited by function) so that a
//! description the backend accepts is never refused later by wgpu on the render thread, where
//! the trait could not report it. Each refusal is a [`RenderError::Unsupported`] with one of the
//! messages below; the tests name each.
//!
//! Textures ([`check_texture`], in this order; wgpu-core `Device::create_texture_inner` and
//! `conv::check_texture_dimension_size`):
//!
//! 1. a 1D texture has one layer and one mip level, is not compressed and stores colour, and is
//!    not a render target;
//! 2. a depth or stencil texture is 2D (a cube is a 2D array to wgpu);
//! 3. a compressed 3D texture needs the device's sliced-3D feature for its family (BC, ASTC);
//!    ETC2 and EAC are never 3D;
//! 4. a multisampled texture has one layer, is a render target, is not a storage image, and its
//!    own format allows the sample count (wgpu decides per format; `Caps::sample_counts` is
//!    `A8B8G8R8_UNORM`'s, [`crate::caps`]);
//! 5. the host size fits the device: a compressed texture is made with its width and height
//!    rounded up to whole blocks (wgpu needs whole blocks; the guest need not have them), and
//!    that size must still be within the device's limit.
//!
//! Views ([`check_view`]; wgpu-core `Device::create_texture_view` and `create_texture_inner`'s
//! view-format check): no 1D array views (wgpu has none); a view's format is its texture's or
//! that format's sRGB counterpart, and only on a device that can reinterpret
//! (`DownlevelFlags::VIEW_FORMATS`); a multisampled texture's view is 2D.
//!
//! Commands (wgpu-core `command/transfer.rs` and `device/queue.rs`):
//!
//! - [`check_buffer_range`]: a buffer range written by an upload or copied from or to by a
//!   buffer-to-buffer copy starts and ends on a [`COPY_BUFFER_ALIGNMENT`] boundary
//!   (`Queue::write_buffer`, `copy_buffer_to_buffer`). Readbacks of buffers need no alignment:
//!   the backend reads the aligned range around them.
//! - [`check_buffer_copy`]: a buffer-to-buffer copy is between two buffers, even when the
//!   ranges do not overlap (`copy_buffer_to_buffer`, `TransferError::SameSourceDestinationBuffer`).
//! - [`check_buffer_texture`], in this order, for a copy between a buffer and a texture, an
//!   upload into a texture and a readback of a texture (`validate_texture_copy_src_format`,
//!   `validate_texture_copy_dst_format`, `validate_texture_buffer_copy` and
//!   `validate_texture_copy_range`):
//!   1. a colour texture's buffer side starts on a multiple of the format's block size
//!      (`UnalignedBufferOffset`), and nothing else below applies to it;
//!   2. a depth or stencil texture's texels move only for a single-aspect format (`D16_UNORM`,
//!      `D32_FLOAT`, `S8_UINT`): this backend names no aspect in a copy, and wgpu copies one
//!      aspect at a time (`CopyAspectNotOne`);
//!   3. nothing is copied into a `D32_FLOAT` texture (wgpu's `Depth32Float` is never a copy
//!      destination, `CopyToForbiddenTextureFormat`);
//!   4. the device can copy depth and stencil texels to and from buffers
//!      (`DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES`);
//!   5. the buffer side starts on a multiple of [`DEPTH_COPY_OFFSET_ALIGNMENT`] bytes, wgpu's
//!      offset alignment for every depth-stencil format;
//!   6. the texture side covers the whole width and height of its mip level, with any layers
//!      (`UnsupportedPartialTransfer`).
//!
//!   And [`check_depth_buffer_rows`], for a copy between a guest buffer and a depth or stencil
//!   texture (not an upload or a readback): its rows are a multiple of 256 bytes
//!   (`COPY_BYTES_PER_ROW_ALIGNMENT`) or it has one row of texels in each layer. Guest buffer
//!   data is tightly packed, so wgpu's 256-byte row stride fits it only then, and the one copy
//!   per row a colour copy falls back to is a partial copy wgpu refuses for these formats.
//!   An upload (`Queue::write_texture` needs no stride alignment) and a readback (read with
//!   padded rows, [`check_read_size`]) have no such limit.
//!
//!   Two-aspect texels (`D32_FLOAT_S8_UINT`, the format the fallback picks for
//!   `D24_UNORM_S8_UINT`) are refused by 2; [`crate::caps`] says what that leaves `Caps`
//!   claiming.
//! - [`check_texture_copy`], in this order: a texture-to-texture copy is between formats that
//!   differ at most in sRGB; within one mip level of one texture, a 1D, 2D or cube copy is
//!   between layer ranges that do not overlap, whatever the regions' x and y, and a 3D copy is
//!   between regions whose first slices differ (`validate_copy_within_same_texture`, to which
//!   wgpu passes the copy's layer count for a 2D texture and 1 for a 1D or 3D one; the shared
//!   checks have already refused regions that share a texel); and a copy of a depth, stencil
//!   or multisampled texture covers the whole width and height of the mip level, with any
//!   layers (`validate_texture_copy_range`, whose z check never asks for the whole extent).
//! - [`check_read_size`]: a readback of a texture reads into a buffer of its own in one copy,
//!   with each row padded to wgpu's 256-byte row stride (`COPY_BYTES_PER_ROW_ALIGNMENT`), and
//!   that buffer must fit `Caps::limits.max_buffer_size`; a bigger one is refused. A frame dump
//!   whose read would not fit is not refused: its pixels are reported unknown
//!   ([`crate::plan`]).
//! - [`check_clear_area`]: a clear of part of a multisampled or depth or stencil view is
//!   refused; the backend clears part of a view by clearing a scratch texture and copying it in,
//!   which those copies forbid.
//! - [`check_sampler`]: a border colour other than transparent black, opaque black or opaque
//!   white is refused when an address mode clamps to the border (wgpu's `SamplerBorderColor`).

use crate::caps::{COPY_BUFFER_ALIGNMENT, FormatSupport};
use crate::format::texture_format;
use starquake_render::command::{Rect, TextureRegion};
use starquake_render::format::{Compression, FormatAspect};
use starquake_render::resource::{
    AddressMode, BorderColor, Extent3d, TextureDimension, ViewDimension,
};
use starquake_render::{
    BufferId, Caps, PixelFormat, RenderError, SamplerDesc, TextureDesc, TextureUsage, ViewDesc,
};

/// A 1D texture with more than one layer.
pub const ONE_D_LAYERS: &str = "a 1D texture has one layer under wgpu";
/// A 1D texture with more than one mip level.
pub const ONE_D_MIPS: &str = "a 1D texture has one mip level under wgpu";
/// A 1D texture of a compressed, depth or stencil format.
pub const ONE_D_FORMAT: &str = "a 1D texture stores uncompressed colour under wgpu";
/// A 1D render target.
pub const ONE_D_TARGET: &str = "a 1D texture cannot be a render target under wgpu";
/// A depth or stencil texture that is not 2D.
pub const DEPTH_DIMENSION: &str = "a depth or stencil texture is 2D under wgpu";
/// A compressed 3D texture the device cannot make.
pub const COMPRESSED_3D: &str = "this device cannot make a 3D texture of this compressed format";
/// A multisampled texture with more than one layer.
pub const MULTISAMPLED_LAYERS: &str = "a multisampled texture has one layer under wgpu";
/// A multisampled texture without the render-target usage.
pub const MULTISAMPLED_TARGET: &str = "a multisampled texture must be a render target under wgpu";
/// A multisampled storage texture.
pub const MULTISAMPLED_STORAGE: &str =
    "a multisampled texture cannot be a storage image under wgpu";
/// A compressed texture whose size, rounded up to whole blocks, is past the device's limit.
pub const PADDED_SIZE: &str = "a compressed texture rounded up to whole blocks is too large";
/// A 1D array view.
pub const ONE_D_ARRAY_VIEW: &str = "wgpu has no 1D array views";
/// A view whose format differs from its texture's by more than sRGB.
pub const VIEW_FORMAT: &str = "a wgpu view's format is its texture's or its sRGB counterpart";
/// A view of a multisampled texture that is not 2D.
pub const MULTISAMPLED_VIEW: &str = "a view of a multisampled texture is 2D under wgpu";
/// A buffer range off the copy alignment.
pub const BUFFER_ALIGNMENT: &str =
    "a buffer range the wgpu backend writes or copies starts and ends on a 4-byte boundary";
/// A buffer copy of a two-aspect depth-stencil texture's texels.
pub const DEPTH_STENCIL_BUFFER_COPY: &str =
    "the wgpu backend does not copy depth-stencil texels to or from buffers yet";
/// An upload or buffer copy into a 32-bit float depth texture.
pub const DEPTH32_DESTINATION: &str =
    "wgpu cannot copy into a 32-bit float depth texture from a buffer";
/// A buffer copy of depth or stencil texels on a device that cannot make one.
pub const DEPTH_COPIES: &str = "this device cannot copy depth or stencil texels to or from buffers";
/// A buffer side of a depth or stencil copy that is not on a 4-byte boundary.
pub const DEPTH_OFFSET: &str =
    "a buffer copied to or from a depth or stencil texture starts on a 4-byte boundary";
/// A buffer side of a texture copy that is not on a block boundary.
pub const BLOCK_OFFSET: &str =
    "a buffer copied to or from a texture starts on a multiple of the texel block size";
/// A texture copy between formats that differ by more than sRGB.
pub const COPY_FORMATS: &str = "a wgpu texture copy is between formats that differ at most in sRGB";
/// A buffer-to-buffer copy within one buffer.
pub const SAME_BUFFER: &str = "a wgpu buffer copy is between two buffers";
/// A texture-to-texture copy within one mip level of one texture whose layer ranges overlap (for
/// a 3D texture, whose first slices are equal).
pub const SAME_LEVEL: &str =
    "a wgpu copy within one texture is between different mip levels or layers";
/// A copy of part of the width or height of a depth, stencil or multisampled texture's level.
pub const WHOLE_COPY: &str = "a wgpu copy of a depth, stencil or multisampled texture covers the whole width and height of its mip level";
/// A partial clear of a depth, stencil or multisampled view.
pub const WHOLE_CLEAR: &str =
    "the wgpu backend clears depth, stencil and multisampled views only whole";
/// A border colour wgpu cannot express.
pub const BORDER_COLOR: &str =
    "wgpu's border colours are transparent black, opaque black and opaque white only";

fn unsupported(message: &'static str) -> RenderError {
    RenderError::Unsupported(message)
}

/// The wgpu texture shape the backend makes for a checked description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostTexture {
    /// The wgpu dimension (a cube is a 2D array).
    pub dimension: wgpu::TextureDimension,
    /// The size, with a compressed format's width and height rounded up to whole blocks.
    pub size: Extent3d,
}

/// Checks the rules wgpu adds for `desc`, in the order of the module docs, and returns the
/// host shape. `desc` has passed `validate_texture` against `caps`.
pub fn check_texture(
    caps: &Caps,
    formats: &FormatSupport,
    features: wgpu::Features,
    desc: &TextureDesc,
) -> Result<HostTexture, RenderError> {
    let format = desc.format;
    let color = format.aspect() == FormatAspect::Color;
    let compression = format.compression();
    if desc.dimension == TextureDimension::D1 {
        if desc.size.depth_or_layers != 1 {
            return Err(unsupported(ONE_D_LAYERS));
        }
        if desc.mip_levels != 1 {
            return Err(unsupported(ONE_D_MIPS));
        }
        if !color || compression != Compression::Uncompressed {
            return Err(unsupported(ONE_D_FORMAT));
        }
        if desc.usage.contains(TextureUsage::RENDER_TARGET) {
            return Err(unsupported(ONE_D_TARGET));
        }
    }
    let flat = matches!(
        desc.dimension,
        TextureDimension::D2 | TextureDimension::Cube
    );
    if !color && !flat {
        return Err(unsupported(DEPTH_DIMENSION));
    }
    if desc.dimension == TextureDimension::D3 {
        let needed = match compression {
            Compression::Uncompressed => wgpu::Features::empty(),
            Compression::Bc => wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D,
            Compression::Astc => wgpu::Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D,
            Compression::Etc2 => return Err(unsupported(COMPRESSED_3D)),
        };
        if !features.contains(needed) {
            return Err(unsupported(COMPRESSED_3D));
        }
    }
    if desc.sample_count > 1 {
        if desc.size.depth_or_layers != 1 {
            return Err(unsupported(MULTISAMPLED_LAYERS));
        }
        if !desc.usage.contains(TextureUsage::RENDER_TARGET) {
            return Err(unsupported(MULTISAMPLED_TARGET));
        }
        if desc.usage.contains(TextureUsage::STORAGE) {
            return Err(unsupported(MULTISAMPLED_STORAGE));
        }
        if !formats.samples(format).supports(desc.sample_count) {
            return Err(RenderError::SampleCount(desc.sample_count));
        }
    }
    let (dimension, limit) = match desc.dimension {
        TextureDimension::D1 => (
            wgpu::TextureDimension::D1,
            caps.limits.max_texture_dimension_1d,
        ),
        TextureDimension::D2 | TextureDimension::Cube => (
            wgpu::TextureDimension::D2,
            caps.limits.max_texture_dimension_2d,
        ),
        TextureDimension::D3 => (
            wgpu::TextureDimension::D3,
            caps.limits.max_texture_dimension_3d,
        ),
    };
    let pad = |len: u32, block: u32| -> Result<u32, RenderError> {
        len.checked_next_multiple_of(block)
            .filter(|padded| *padded <= limit)
            .ok_or(unsupported(PADDED_SIZE))
    };
    Ok(HostTexture {
        dimension,
        size: Extent3d {
            width: pad(desc.size.width, format.block_width())?,
            height: pad(desc.size.height, format.block_height())?,
            depth_or_layers: desc.size.depth_or_layers,
        },
    })
}

/// Checks the rules wgpu adds for a view `view` of `texture`; both have passed `validate_view`.
/// `reinterpret` is whether the device can view a texture in another format
/// (`DownlevelFlags::VIEW_FORMATS`).
pub fn check_view(
    texture: &TextureDesc,
    view: &ViewDesc,
    reinterpret: bool,
) -> Result<(), RenderError> {
    if view.dimension == ViewDimension::D1Array {
        return Err(unsupported(ONE_D_ARRAY_VIEW));
    }
    if view.format != texture.format {
        let base = |f: PixelFormat| texture_format(f).map(|t| t.remove_srgb_suffix());
        let same = base(view.format).is_some() && base(view.format) == base(texture.format);
        if !same || !reinterpret {
            return Err(unsupported(VIEW_FORMAT));
        }
    }
    if texture.sample_count > 1 && view.dimension != ViewDimension::D2 {
        return Err(unsupported(MULTISAMPLED_VIEW));
    }
    Ok(())
}

/// Checks that a buffer range written by an upload or a buffer-to-buffer copy is aligned.
pub fn check_buffer_range(offset: u64, size: u64) -> Result<(), RenderError> {
    if offset.is_multiple_of(COPY_BUFFER_ALIGNMENT) && size.is_multiple_of(COPY_BUFFER_ALIGNMENT) {
        Ok(())
    } else {
        Err(unsupported(BUFFER_ALIGNMENT))
    }
}

/// Checks a buffer-to-buffer copy from `src` to `dst` beyond the alignment of each range.
pub fn check_buffer_copy(src: BufferId, dst: BufferId) -> Result<(), RenderError> {
    if src == dst {
        Err(unsupported(SAME_BUFFER))
    } else {
        Ok(())
    }
}

/// Which way a copy between a buffer and a texture moves texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexelFlow {
    /// From a buffer or staging memory into the texture (an upload or a buffer-to-texture copy).
    IntoTexture,
    /// From the texture into a buffer (a texture-to-buffer copy or a readback).
    OutOfTexture,
}

/// A copy between a guest buffer and a depth or stencil texture of several rows that are not a
/// multiple of 256 bytes.
pub const DEPTH_ROWS: &str = "a wgpu copy between a buffer and a depth or stencil texture has rows of a multiple of 256 bytes or one row";

/// Checks the rows of a copy between a guest buffer and a texture of `format`: `rows` rows of
/// `row_bytes` bytes in each layer (module docs). Colour formats always pass.
pub fn check_depth_buffer_rows(
    format: PixelFormat,
    row_bytes: u32,
    rows: u32,
) -> Result<(), RenderError> {
    let color = format.aspect() == FormatAspect::Color;
    if color || rows <= 1 || row_bytes.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) {
        Ok(())
    } else {
        Err(unsupported(DEPTH_ROWS))
    }
}

/// The buffer offset alignment of a depth or stencil copy: wgpu-core 30.0.2
/// `validate_texture_buffer_copy` uses 4 for every depth-stencil format, whatever its texel
/// size.
pub const DEPTH_COPY_OFFSET_ALIGNMENT: u64 = 4;

/// Checks the texture side of a buffer copy, an upload or a readback of `region` of `desc`,
/// whose buffer side starts at `buffer_offset` (an upload's staging bytes start at 0), in the
/// order of the module docs. `depth_copies` is whether the device can copy depth and stencil
/// texels to and from buffers (`DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES`).
pub fn check_buffer_texture(
    desc: &TextureDesc,
    region: &TextureRegion,
    buffer_offset: u64,
    flow: TexelFlow,
    depth_copies: bool,
) -> Result<(), RenderError> {
    let format = desc.format;
    if format.aspect() == FormatAspect::Color {
        return if buffer_offset.is_multiple_of(format.bytes_per_block().into()) {
            Ok(())
        } else {
            Err(unsupported(BLOCK_OFFSET))
        };
    }
    if !matches!(
        texture_format(format),
        Some(
            wgpu::TextureFormat::Depth16Unorm
                | wgpu::TextureFormat::Depth32Float
                | wgpu::TextureFormat::Stencil8
        )
    ) {
        return Err(unsupported(DEPTH_STENCIL_BUFFER_COPY));
    }
    if flow == TexelFlow::IntoTexture && format == PixelFormat::D32Float {
        return Err(unsupported(DEPTH32_DESTINATION));
    }
    if !depth_copies {
        return Err(unsupported(DEPTH_COPIES));
    }
    if !buffer_offset.is_multiple_of(DEPTH_COPY_OFFSET_ALIGNMENT) {
        return Err(unsupported(DEPTH_OFFSET));
    }
    if !whole_width_and_height(desc, region) {
        return Err(unsupported(WHOLE_COPY));
    }
    Ok(())
}

/// Whether `region` covers the whole width and height of its mip level of `desc`, with any
/// layers or slices (wgpu's partial-copy check skips z, `validate_texture_copy_range`). A mip
/// level the description lacks is never covered.
fn whole_width_and_height(desc: &TextureDesc, region: &TextureRegion) -> bool {
    let Some(level) = desc.mip_extent(region.mip) else {
        return false;
    };
    region.origin.x == 0
        && region.origin.y == 0
        && region.extent.width == level.width
        && region.extent.height == level.height
}

/// A texture readback whose rows, each padded to wgpu's 256-byte row stride, take more bytes
/// than the largest buffer.
pub const READ_TOO_LARGE: &str =
    "a texture readback with its rows padded to 256 bytes is larger than the largest buffer";

/// The row stride and the size in bytes of a read of `rows` rows of `row_bytes` bytes in each
/// of `images` images, with every row padded to `COPY_BYTES_PER_ROW_ALIGNMENT` so that one wgpu
/// copy reads them all (`crate::thread`), or `None` when either overflows.
pub fn padded_read(row_bytes: u32, rows: u32, images: u32) -> Option<(u32, u64)> {
    let stride = row_bytes.checked_next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)?;
    let size = u64::from(stride)
        .checked_mul(rows.into())?
        .checked_mul(images.into())?;
    Some((stride, size))
}

/// Checks that a texture readback of `rows` rows of `row_bytes` bytes in each of `images`
/// images, padded as [`padded_read`] pads it, fits in a buffer of `max_buffer_size` bytes
/// (`Caps::limits`). The read buffer is a wgpu buffer like any other, and the padding is what
/// bounds its size: at most 255 bytes more per row than the packed bytes.
pub fn check_read_size(
    row_bytes: u32,
    rows: u32,
    images: u32,
    max_buffer_size: u64,
) -> Result<(), RenderError> {
    match padded_read(row_bytes, rows, images) {
        Some((_, size)) if size <= max_buffer_size => Ok(()),
        _ => Err(unsupported(READ_TOO_LARGE)),
    }
}

/// Checks a texture-to-texture copy that has passed `validate_copy`.
pub fn check_texture_copy(
    src: &TextureDesc,
    src_region: &TextureRegion,
    dst: &TextureDesc,
    dst_region: &TextureRegion,
) -> Result<(), RenderError> {
    let base = |f: PixelFormat| texture_format(f).map(|t| t.remove_srgb_suffix());
    if base(src.format).is_none() || base(src.format) != base(dst.format) {
        return Err(unsupported(COPY_FORMATS));
    }
    if src_region.texture == dst_region.texture && src_region.mip == dst_region.mip {
        let (s, d) = (src_region.origin.z, dst_region.origin.z);
        let overlap = if src.dimension == TextureDimension::D3 {
            // wgpu passes a 3D copy a layer count of 1: only an equal first slice is refused.
            s == d
        } else {
            // Each side's own range (the shared checks make the two extents equal).
            let end = |start: u32, r: &TextureRegion| {
                u64::from(start) + u64::from(r.extent.depth_or_layers)
            };
            u64::from(s) < end(d, dst_region) && u64::from(d) < end(s, src_region)
        };
        if overlap {
            return Err(unsupported(SAME_LEVEL));
        }
    }
    let partial =
        !whole_width_and_height(src, src_region) || !whole_width_and_height(dst, dst_region);
    let restricted = src.format.aspect() != FormatAspect::Color || src.sample_count > 1;
    if restricted && partial {
        return Err(unsupported(WHOLE_COPY));
    }
    Ok(())
}

/// Checks a clear of `rect` (`None`: the whole level) in a view of `texture` whose format is
/// `format` and whose base level is `width` by `height`.
pub fn check_clear_area(
    texture: &TextureDesc,
    format: PixelFormat,
    rect: Option<&Rect>,
    width: u32,
    height: u32,
) -> Result<(), RenderError> {
    let whole = rect.is_none_or(|r| r.x == 0 && r.y == 0 && r.width == width && r.height == height);
    let restricted = format.aspect() != FormatAspect::Color || texture.sample_count > 1;
    if restricted && !whole {
        return Err(unsupported(WHOLE_CLEAR));
    }
    Ok(())
}

/// The wgpu border colour for `color`, if wgpu has one.
pub fn border_color(color: BorderColor) -> Option<wgpu::SamplerBorderColor> {
    match color {
        BorderColor::TransparentBlack | BorderColor::Custom([0.0, 0.0, 0.0, 0.0]) => {
            Some(wgpu::SamplerBorderColor::TransparentBlack)
        }
        BorderColor::OpaqueBlack | BorderColor::Custom([0.0, 0.0, 0.0, 1.0]) => {
            Some(wgpu::SamplerBorderColor::OpaqueBlack)
        }
        BorderColor::OpaqueWhite | BorderColor::Custom([1.0, 1.0, 1.0, 1.0]) => {
            Some(wgpu::SamplerBorderColor::OpaqueWhite)
        }
        BorderColor::Custom(_) => None,
    }
}

/// Checks a sampler that has passed `validate_sampler`.
pub fn check_sampler(desc: &SamplerDesc) -> Result<(), RenderError> {
    let clamps = desc.address.contains(&AddressMode::ClampToBorder);
    if clamps && border_color(desc.border_color).is_none() {
        return Err(unsupported(BORDER_COLOR));
    }
    Ok(())
}

/// The size of the host buffer for a guest buffer of `size` bytes: rounded up to the copy
/// alignment, so that a readback can read the aligned range around any guest range. `None`
/// when it overflows (it cannot for a size `Caps` allows: `max_buffer_size` is a multiple of
/// the alignment).
pub fn host_buffer_size(size: u64) -> Option<u64> {
    size.checked_next_multiple_of(COPY_BUFFER_ALIGNMENT)
}

#[cfg(test)]
mod tests;
