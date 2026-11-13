//! A CPU decoder for ASTC-compressed textures (docs/specs/render-backend.md, "ASTC"; slice R3).
//!
//! Hosts whose GPU has no native ASTC get RGBA8 instead (the format fallback choice, slice
//! R2). The decode
//! normally runs on the GPU (slice R10); this decoder is the fallback for it and the oracle it
//! is checked against. It decodes the 2D blocks of the console's fourteen footprints into
//! RGBA8 texels: bytes in the order red, green, blue, alpha, which is the memory layout of
//! [`PixelFormat::A8B8G8R8Unorm`] and [`PixelFormat::A8B8G8R8Srgb`].
//!
//! # Sources
//!
//! Ported from Eden's `eden/src/video_core/textures/astc.cpp` @ cf654052 (GPL-3.0-or-later,
//! Copyright 2026 Eden Emulator Project), which is itself derived from `FasTC` (Apache-2.0,
//! Copyright 2016 The University of North Carolina at Chapel Hill;
//! <http://gamma.cs.unc.edu/FasTC/>). Section numbers (C.2.x) are those of the ASTC
//! specification as published in the Khronos extension `KHR_texture_compression_astc_hdr`,
//! which is what the tests' hand-encoded blocks are computed from.
//!
//! # Untrusted input
//!
//! A block is 16 bytes of guest data, and every size in it (the weight grid, the weight and
//! colour ranges, the partition count, the endpoint modes) is read from those bytes. Each is
//! checked against the specification's limits before it sizes anything, every loop is bounded
//! by a constant of the format (at most 144 texels, 64 weights, 18 colour values), and no
//! input panics: `astc::tests::any_block_decodes_without_panicking` feeds it arbitrary bytes.
//! An illegal block decodes to the error colour, [`ERROR_COLOR`]. [`decode`] also checks the
//! image's sizes, in arithmetic that cannot overflow unnoticed, before it reads or writes.
//!
//! # Where this follows the specification rather than Eden
//!
//! Each is pinned by the test named.
//!
//! - **The error colour is opaque magenta** (C.2.5), where Eden writes transparent black
//!   (`FillError`). `illegal_encodings_decode_to_the_error_colour`.
//! - **Two more illegal encodings are recognised** (C.2.24): colour endpoint modes that need
//!   more than 18 values, and fewer colour bits than `ceil(13 * values / 5)`. Eden decodes
//!   both from whatever bits there are. `too_many_colour_values_is_illegal`,
//!   `too_few_colour_bits_is_illegal`.
//! - **sRGB formats use the sRGB interpolation** (C.2.19): endpoints are widened with `0x80`
//!   rather than by replication and the result is the top 8 bits, for red, green and blue.
//!   Eden has one path for both encodings. `srgb_interpolation_differs_from_linear`.
//! - **An sRGB format is decoded in the specification's LDR mode** (C.2.5, C.2.24): the texels
//!   of a partition with an HDR endpoint mode, and an HDR void-extent block, get the error
//!   colour. `hdr_in_an_srgb_format_is_the_error_colour`.
//! - **An HDR void-extent block in a linear format is decoded** like every other HDR value,
//!   where Eden gives its error colour although it decodes HDR endpoints.
//!   `hdr_void_extent_in_a_linear_format`.
//! - **HDR mode 11's `d0` and `d1` are as wide as table C.2.23 says** (5, 6 or 7 bits by the
//!   mode). Eden's `SignExtend` keeps the bits of `v4` and `v5` above the field, which in the
//!   narrower modes are the extra bits X2 to X5 and belong to other fields: a block with one
//!   of them set decodes to other colours there. `endpoints::tests::hdr_direct_by_hand`.
//!
//! # Where this follows Eden
//!
//! - **HDR endpoint modes in a linear format are decoded**, and each value is clamped to
//!   0..1 and rounded to a byte, since the stand-in texture is RGBA8. That the console's GPU
//!   decodes HDR blocks in its UNORM ASTC formats is a guess (Eden does; a hardware test
//!   would settle it); in the specification's LDR mode they would be the error colour.
//! - **A linear LDR value is the FP16-mode result rounded to a byte**: `C / 65536` (C.2.19)
//!   times 255, rounded half up, computed in integers.
//! - **A void-extent block's coordinates are not checked** (C.2.23 makes a low coordinate at
//!   or above its high one illegal; the LDR profile, C.2.25, lets a decoder skip the check, and
//!   Eden skips it). `void_extent_coordinates_are_not_checked`.
//! - **An LDR void-extent colour is the top 8 bits** of each 16-bit component.
//!
//! # How it is checked
//!
//! `astc::tests` has blocks laid out by hand from the specification's figures, one block for
//! each illegal encoding with a legal neighbour at the limit, tables computed from the
//! specification's procedures by a separate program (every trit and quint code, every colour
//! and weight value, the HDR endpoint modes), and a comparison of every decoded block with a
//! second decoder, whose limits `astc::tests::oracle` states.
//!
//! # What is not here
//!
//! 3D blocks (the console's formats are 2D; a texture's depth or layers are decoded slice by
//! slice, as Eden does), and recompression to BC (deferred: the spec's decision 6).

use crate::error::RenderError;
use crate::format::{ColorEncoding, Compression, PixelFormat};
use crate::resource::Extent3d;
use crate::validate::packed_size;

mod block;
mod endpoints;
mod integer;
mod mode;
mod weights;

/// Bytes in one ASTC block, whatever its footprint (C.2.9).
pub const BLOCK_BYTES: usize = 16;

/// Texels in the largest footprint, 12 by 12.
pub const MAX_TEXELS: usize = 144;

/// The colour of every texel of an illegal block: opaque magenta (C.2.5), as red, green, blue,
/// alpha.
pub const ERROR_COLOR: [u8; 4] = [0xFF, 0x00, 0xFF, 0xFF];

/// The size in texels of the area one block covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Footprint {
    width: u32,
    height: u32,
}

impl Footprint {
    /// The footprint of an ASTC format, or `None` for any other format.
    pub const fn of(format: PixelFormat) -> Option<Self> {
        match format.compression() {
            Compression::Astc => Some(Self {
                width: format.block_width(),
                height: format.block_height(),
            }),
            Compression::Uncompressed | Compression::Bc | Compression::Etc2 => None,
        }
    }

    /// Texels across.
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Texels down.
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Texels in the footprint: at most [`MAX_TEXELS`].
    const fn texel_count(self) -> usize {
        (self.width * self.height) as usize
    }
}

/// Decodes one block into `texels`, row after row: texel (x, y) of the block is entry
/// `y * footprint.width() + x`, as red, green, blue, alpha. Entries past the footprint's
/// last texel are left as they were.
///
/// Returns whether the block is a legal encoding. An illegal block's texels are all
/// [`ERROR_COLOR`]. A legal block can still hold error-coloured texels: those of a partition
/// with an HDR endpoint mode, when `encoding` is sRGB (module docs).
pub fn decode_block(
    block: &[u8; BLOCK_BYTES],
    footprint: Footprint,
    encoding: ColorEncoding,
    texels: &mut [[u8; 4]; MAX_TEXELS],
) -> bool {
    let Some(texels) = texels.get_mut(..footprint.texel_count()) else {
        return false;
    };
    let legal = block::decode(u128::from_le_bytes(*block), footprint, encoding, texels).is_some();
    if !legal {
        texels.fill(ERROR_COLOR);
    }
    legal
}

/// A length as the 64-bit number sizes are compared in.
fn len_u64(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

/// Decodes a whole ASTC image into RGBA8.
///
/// `data` holds the blocks of `extent` texels of `format`, tightly packed
/// ([`crate::validate::packed_size`]): rows of blocks, row after row, then slice or layer
/// after slice. `out` receives `extent.width * extent.height * 4` bytes for each slice or
/// layer, row after row. A block that reaches past the image's right or bottom edge is
/// cropped.
///
/// Returns the number of illegal blocks, each decoded to [`ERROR_COLOR`], for the caller to
/// count. Errors: `format` is not ASTC; `data` or `out` is not exactly the size the extent
/// needs; or a size does not fit in 64 bits.
pub fn decode(
    format: PixelFormat,
    extent: Extent3d,
    data: &[u8],
    out: &mut [u8],
) -> Result<u64, RenderError> {
    let footprint = Footprint::of(format).ok_or(RenderError::Invalid("the format is not ASTC"))?;
    // Both sizes are worked out, in checked arithmetic, before either length is compared.
    let data_size = packed_size(format, extent)?;
    let out_size = packed_size(PixelFormat::A8B8G8R8Unorm, extent)?;
    if len_u64(data.len()) != data_size {
        return Err(RenderError::SizeMismatch {
            what: "ASTC data",
            expected: data_size,
            actual: len_u64(data.len()),
        });
    }
    if len_u64(out.len()) != out_size {
        return Err(RenderError::SizeMismatch {
            what: "ASTC decode output",
            expected: out_size,
            actual: len_u64(out.len()),
        });
    }
    if out.is_empty() {
        return Ok(0);
    }
    let encoding = if format.is_srgb() {
        ColorEncoding::Srgb
    } else {
        ColorEncoding::Linear
    };

    // `out` holds width * height * 4 bytes a layer and is not empty, so the width, the height
    // and every product below fit in `usize`.
    let (width, height) = (extent.width as usize, extent.height as usize);
    let (block_width, block_height) = (footprint.width as usize, footprint.height as usize);
    let columns = width.div_ceil(block_width);
    let blocks_per_layer = columns * height.div_ceil(block_height);
    let layer_bytes = width * height * 4;

    let mut illegal = 0;
    let mut texels = [ERROR_COLOR; MAX_TEXELS];
    let (blocks, _) = data.as_chunks::<BLOCK_BYTES>();
    for (index, block) in blocks.iter().enumerate() {
        illegal += u64::from(!decode_block(block, footprint, encoding, &mut texels));
        let (layer, in_layer) = (index / blocks_per_layer, index % blocks_per_layer);
        let (x, y) = (
            (in_layer % columns) * block_width,
            (in_layer / columns) * block_height,
        );
        // The part of the block that is inside the image.
        let (shown_width, shown_height) =
            (block_width.min(width - x), block_height.min(height - y));
        for row in 0..shown_height {
            let source = &texels[row * block_width..row * block_width + shown_width];
            let start = layer * layer_bytes + ((y + row) * width + x) * 4;
            let (target, _) = out[start..start + shown_width * 4].as_chunks_mut::<4>();
            for (bytes, texel) in target.iter_mut().zip(source) {
                *bytes = *texel;
            }
        }
    }
    Ok(illegal)
}

#[cfg(test)]
mod tests;
