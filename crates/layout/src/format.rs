//! What the layout needs to know of a texel format: the size of one element and the block of
//! samples it covers.
//!
//! Ported from `mesa-nouveau/src/nouveau/nil/format.rs` @ 52238762 (MIT, Copyright © 2024
//! Collabora, Ltd.; the licence text is in this crate's `lib.rs`), `el_size_B` and `el_extent_sa`
//! only. Mesa's `Format` wraps a Gallium `pipe_format` and reads both numbers from Mesa's format
//! description table (`block.bits / 8` and `block.width`/`height`/`depth`); this crate has no
//! format table, so the caller (the GPU crate, decoding a texture header) supplies the numbers.
//! Mesa's support queries (texturing, rendering, storage) and depth/stencil predicates are not
//! ported: they choose driver behaviour, not layout.

use crate::error::{LayoutError, Result};
use crate::extent::Extent4D;
use crate::extent::units::Samples;

/// The layout-relevant part of a texel format.
///
/// One element is one texel of an uncompressed format, or one compressed block (for example a
/// 16-byte ASTC block covering 8 by 8 texels).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Format {
    el_size_b: u32,
    block_width: u32,
    block_height: u32,
    block_depth: u32,
}

impl Format {
    /// A format whose elements are `el_size_b` bytes and cover `block_width` by
    /// `block_height` by `block_depth` samples.
    ///
    /// Every number must be non-zero ([`LayoutError::InvalidFormat`] otherwise); Mesa's format
    /// table never holds a zero, so it has no such check.
    pub fn new(
        el_size_b: u32,
        block_width: u32,
        block_height: u32,
        block_depth: u32,
    ) -> Result<Self> {
        if el_size_b == 0 || block_width == 0 || block_height == 0 || block_depth == 0 {
            return Err(LayoutError::InvalidFormat);
        }
        Ok(Self {
            el_size_b,
            block_width,
            block_height,
            block_depth,
        })
    }

    /// An uncompressed format of `el_size_b` bytes per texel (a 1 by 1 by 1 block).
    pub fn uncompressed(el_size_b: u32) -> Result<Self> {
        Self::new(el_size_b, 1, 1, 1)
    }

    /// Bytes per element: Mesa's `el_size_B`.
    pub fn el_size_b(&self) -> u32 {
        self.el_size_b
    }

    /// The samples one element covers: Mesa's `el_extent_sa`.
    pub(crate) fn el_extent_sa(&self) -> Extent4D<Samples> {
        Extent4D::new(self.block_width, self.block_height, self.block_depth, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_zero_field_is_rejected() {
        assert_eq!(Format::new(0, 1, 1, 1), Err(LayoutError::InvalidFormat));
        assert_eq!(Format::new(1, 0, 1, 1), Err(LayoutError::InvalidFormat));
        assert_eq!(Format::new(1, 1, 0, 1), Err(LayoutError::InvalidFormat));
        assert_eq!(Format::new(1, 1, 1, 0), Err(LayoutError::InvalidFormat));
        assert_eq!(Format::uncompressed(0), Err(LayoutError::InvalidFormat));
    }

    #[test]
    fn fields_reach_their_accessors() {
        let format = Format::new(16, 8, 6, 2).unwrap();
        assert_eq!(format.el_size_b(), 16);
        assert_eq!(format.el_extent_sa(), Extent4D::new(8, 6, 2, 1));
        let plain = Format::uncompressed(4).unwrap();
        assert_eq!(plain.el_size_b(), 4);
        assert_eq!(plain.el_extent_sa(), Extent4D::new(1, 1, 1, 1));
    }
}
