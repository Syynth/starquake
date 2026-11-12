//! Extents and offsets tagged with their unit, and the conversions between units.
//!
//! Ported from `mesa-nouveau/src/nouveau/nil/extent.rs` @ 52238762 (MIT, Copyright © 2024
//! Collabora, Ltd.; the licence text is in this crate's `lib.rs`). Every multiplication, addition
//! and alignment is checked: Mesa's `u32` arithmetic wraps or panics on a huge extent, these
//! return [`LayoutError::Overflow`] instead. A unit is a marker type, as in Mesa, so a byte
//! extent cannot be passed where a pixel extent is meant.
//!
//! The units, from the outside in: pixels (what an image's size is given in), samples (pixels
//! times the sample layout's grid), elements (samples divided by the format's block, so one
//! element is one compressed block or one uncompressed texel) and bytes (elements times the
//! element size, in x only). GOBs and tiles are byte extents divided by the GOB or tile extent.

use core::marker::PhantomData;

use crate::error::{LayoutError, Result};
use crate::format::Format;
use crate::image::SampleLayout;
use crate::tiling::{GobType, Tiling};

/// The unit markers for [`Extent4D`] and [`Offset4D`].
pub mod units {
    /// Pixels: the unit an image's size is given in.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Pixels;

    /// Samples: pixels times the sample layout's grid.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(crate) struct Samples;

    /// Elements: one compressed block, or one texel of an uncompressed format.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(crate) struct Elements;

    /// Bytes in x; rows in y; slices in z.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Bytes;

    /// Tiles (Mesa's name for NVIDIA's blocks of GOBs).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(crate) struct Tiles;

    /// GOBs.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(crate) struct Gobs;
}

use units::{Bytes, Elements, Gobs, Pixels, Samples, Tiles};

/// Rounds `a / b` up; `None` when `b` is zero.
pub(crate) fn checked_div_ceil(a: u32, b: u32) -> Option<u32> {
    let q = a.checked_div(b)?;
    if a.checked_rem(b)? == 0 {
        Some(q)
    } else {
        q.checked_add(1)
    }
}

/// `max(1, value >> level)`, Mesa's `Minify` for `u32` (`nil/lib.rs`). A shift of 32 or more
/// gives 1, as the value has no bits left.
pub(crate) fn minify(value: u32, level: u32) -> u32 {
    value.checked_shr(level).unwrap_or(0).max(1)
}

/// `ceil(log2(value))`, with 0 for 0 and 1: Mesa's `ILog2Ceil` for `u32` (`nil/lib.rs`).
pub(crate) fn ilog2_ceil(value: u32) -> u32 {
    if value <= 1 {
        0
    } else {
        // `value - 1` is at least 1 here, so `ilog2` cannot see zero.
        (value - 1).ilog2() + 1
    }
}

fn overflow<T>(value: Option<T>) -> Result<T> {
    value.ok_or(LayoutError::Overflow)
}

/// A width, height, depth and layer count in unit `U`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Extent4D<U> {
    /// Width in `U`.
    pub width: u32,
    /// Height in `U`.
    pub height: u32,
    /// Depth in `U`.
    pub depth: u32,
    /// Number of array layers.
    pub array_len: u32,
    unit: PhantomData<U>,
}

// Copy whatever the unit: a derive would require the marker `U` to be `Copy` at every use.
impl<U> Clone for Extent4D<U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<U> Copy for Extent4D<U> {}

impl<U> Extent4D<U> {
    /// An extent of `width` by `height` by `depth`, with `array_len` layers.
    pub const fn new(width: u32, height: u32, depth: u32, array_len: u32) -> Self {
        Self {
            width,
            height,
            depth,
            array_len,
            unit: PhantomData,
        }
    }

    /// Whether any dimension is zero.
    pub(crate) fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.depth == 0 || self.array_len == 0
    }

    /// Rounds each dimension up to a multiple of `alignment`'s.
    pub(crate) fn align(self, alignment: &Self) -> Result<Self> {
        Ok(Self::new(
            overflow(self.width.checked_next_multiple_of(alignment.width))?,
            overflow(self.height.checked_next_multiple_of(alignment.height))?,
            overflow(self.depth.checked_next_multiple_of(alignment.depth))?,
            overflow(self.array_len.checked_next_multiple_of(alignment.array_len))?,
        ))
    }

    fn mul<V>(self, other: Extent4D<V>) -> Result<Extent4D<V>> {
        Ok(Extent4D::new(
            overflow(self.width.checked_mul(other.width))?,
            overflow(self.height.checked_mul(other.height))?,
            overflow(self.depth.checked_mul(other.depth))?,
            overflow(self.array_len.checked_mul(other.array_len))?,
        ))
    }

    fn div_ceil<V>(self, other: Self) -> Result<Extent4D<V>> {
        Ok(Extent4D::new(
            overflow(checked_div_ceil(self.width, other.width))?,
            overflow(checked_div_ceil(self.height, other.height))?,
            overflow(checked_div_ceil(self.depth, other.depth))?,
            overflow(checked_div_ceil(self.array_len, other.array_len))?,
        ))
    }
}

impl Extent4D<Pixels> {
    /// Pixels to samples: multiplies by the sample layout's grid.
    pub(crate) fn to_sa(self, sample_layout: SampleLayout) -> Result<Extent4D<Samples>> {
        self.mul(sample_layout.px_extent_sa())
    }

    /// Pixels to elements: samples divided by the format's block, rounded up.
    pub(crate) fn to_el(
        self,
        format: Format,
        sample_layout: SampleLayout,
    ) -> Result<Extent4D<Elements>> {
        self.to_sa(sample_layout)?.div_ceil(format.el_extent_sa())
    }

    /// Pixels to bytes: the width in bytes, the height in rows of elements, the depth in
    /// slices of elements, and the layer count unchanged.
    pub fn to_b(self, format: Format, sample_layout: SampleLayout) -> Result<Extent4D<Bytes>> {
        self.to_el(format, sample_layout)?.to_b(format)
    }

    /// Pixels to tiles of `tiling`, rounded up.
    pub(crate) fn to_tl(
        self,
        tiling: &Tiling,
        format: Format,
        sample_layout: SampleLayout,
    ) -> Result<Extent4D<Tiles>> {
        self.to_b(format, sample_layout)?
            .div_ceil(tiling.extent_b())
    }

    /// The extent of mip level `level`: each of width, height and depth halved `level` times,
    /// never below 1; the layer count unchanged.
    pub(crate) fn minify(self, level: u32) -> Self {
        Self::new(
            minify(self.width, level),
            minify(self.height, level),
            minify(self.depth, level),
            self.array_len,
        )
    }
}

impl Extent4D<Elements> {
    /// Elements to bytes: the width times the element size.
    pub(crate) fn to_b(self, format: Format) -> Result<Extent4D<Bytes>> {
        Ok(Extent4D::new(
            overflow(self.width.checked_mul(format.el_size_b()))?,
            self.height,
            self.depth,
            self.array_len,
        ))
    }
}

impl Extent4D<Bytes> {
    /// Bytes in one layer: width times height times depth. The layer count is not used: Mesa's
    /// `size_B` asserts it is 1, and every caller here passes a one-layer extent.
    pub(crate) fn size_b(&self) -> Result<u64> {
        let area = overflow(u64::from(self.width).checked_mul(u64::from(self.height)))?;
        overflow(area.checked_mul(u64::from(self.depth)))
    }

    /// Bytes to GOBs of `gob_type`, rounded up.
    pub(crate) fn to_gob(self, gob_type: GobType) -> Result<Extent4D<Gobs>> {
        self.div_ceil(gob_type.extent_b())
    }
}

/// An x, y, z position and a layer in unit `U`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Offset4D<U> {
    /// x in `U`.
    pub x: u32,
    /// y in `U`.
    pub y: u32,
    /// z in `U`.
    pub z: u32,
    /// Array layer.
    pub a: u32,
    unit: PhantomData<U>,
}

impl<U> Clone for Offset4D<U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<U> Copy for Offset4D<U> {}

impl<U> Offset4D<U> {
    /// The position (`x`, `y`, `z`) in layer `a`.
    pub const fn new(x: u32, y: u32, z: u32, a: u32) -> Self {
        Self {
            x,
            y,
            z,
            a,
            unit: PhantomData,
        }
    }

    fn mul<V>(self, other: Extent4D<V>) -> Result<Offset4D<V>> {
        Ok(Offset4D::new(
            overflow(self.x.checked_mul(other.width))?,
            overflow(self.y.checked_mul(other.height))?,
            overflow(self.z.checked_mul(other.depth))?,
            overflow(self.a.checked_mul(other.array_len))?,
        ))
    }

    fn div_floor<V>(self, other: Extent4D<U>) -> Result<Offset4D<V>> {
        Ok(Offset4D::new(
            overflow(self.x.checked_div(other.width))?,
            overflow(self.y.checked_div(other.height))?,
            overflow(self.z.checked_div(other.depth))?,
            overflow(self.a.checked_div(other.array_len))?,
        ))
    }

    /// The position one `extent` further on: the exclusive end of a region starting here.
    pub(crate) fn checked_add(self, extent: Extent4D<U>) -> Result<Self> {
        Ok(Self::new(
            overflow(self.x.checked_add(extent.width))?,
            overflow(self.y.checked_add(extent.height))?,
            overflow(self.z.checked_add(extent.depth))?,
            overflow(self.a.checked_add(extent.array_len))?,
        ))
    }
}

impl Offset4D<Pixels> {
    /// Pixels to elements: samples divided by the format's block, rounded down (a position
    /// inside a compressed block belongs to that block).
    pub(crate) fn to_el(
        self,
        format: Format,
        sample_layout: SampleLayout,
    ) -> Result<Offset4D<Elements>> {
        self.mul(sample_layout.px_extent_sa())?
            .div_floor(format.el_extent_sa())
    }

    /// Pixels to bytes: x in bytes, y in rows of elements, z in slices of elements.
    pub fn to_b(self, format: Format, sample_layout: SampleLayout) -> Result<Offset4D<Bytes>> {
        self.to_el(format, sample_layout)?.to_b(format)
    }
}

impl Offset4D<Elements> {
    /// Elements to bytes: x times the element size.
    pub(crate) fn to_b(self, format: Format) -> Result<Offset4D<Bytes>> {
        Ok(Offset4D::new(
            overflow(self.x.checked_mul(format.el_size_b()))?,
            self.y,
            self.z,
            self.a,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div_ceil_edges() {
        assert_eq!(checked_div_ceil(0, 8), Some(0));
        assert_eq!(checked_div_ceil(8, 8), Some(1));
        assert_eq!(checked_div_ceil(9, 8), Some(2));
        assert_eq!(checked_div_ceil(u32::MAX, 1), Some(u32::MAX));
        assert_eq!(checked_div_ceil(u32::MAX, 2), Some(1 << 31));
        assert_eq!(checked_div_ceil(1, 0), None);
    }

    #[test]
    fn minify_halves_down_to_one() {
        assert_eq!(minify(100, 0), 100);
        assert_eq!(minify(100, 1), 50);
        assert_eq!(minify(100, 6), 1);
        assert_eq!(minify(100, 7), 1);
        assert_eq!(minify(u32::MAX, 31), 1);
        assert_eq!(minify(u32::MAX, 32), 1);
        assert_eq!(minify(1, u32::MAX), 1);
    }

    #[test]
    fn ilog2_ceil_matches_mesa() {
        let cases = [
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 3),
            (8, 3),
            (9, 4),
        ];
        for (value, expected) in cases {
            assert_eq!(ilog2_ceil(value), expected, "{value}");
        }
        assert_eq!(ilog2_ceil(1 << 31), 31);
        assert_eq!(ilog2_ceil((1 << 31) + 1), 32);
        assert_eq!(ilog2_ceil(u32::MAX), 32);
    }

    #[test]
    fn align_and_size() {
        let extent: Extent4D<Bytes> = Extent4D::new(65, 9, 3, 1);
        let aligned = extent.align(&Extent4D::new(64, 8, 2, 1));
        assert_eq!(aligned, Ok(Extent4D::new(128, 16, 4, 1)));
        assert_eq!(aligned.and_then(|e| e.size_b()), Ok(128 * 16 * 4));
        let huge: Extent4D<Bytes> = Extent4D::new(u32::MAX, 1, 1, 1);
        assert_eq!(
            huge.align(&Extent4D::new(64, 1, 1, 1)),
            Err(LayoutError::Overflow)
        );
        let cube: Extent4D<Bytes> = Extent4D::new(u32::MAX, u32::MAX, u32::MAX, 1);
        assert_eq!(cube.size_b(), Err(LayoutError::Overflow));
        assert!(Extent4D::<Bytes>::new(0, 1, 1, 1).is_empty());
        assert!(Extent4D::<Bytes>::new(1, 1, 1, 0).is_empty());
        assert!(!Extent4D::<Bytes>::new(1, 1, 1, 1).is_empty());
    }

    #[test]
    fn offset_end_overflow() {
        let start: Offset4D<Bytes> = Offset4D::new(u32::MAX, 0, 0, 0);
        assert_eq!(
            start.checked_add(Extent4D::new(0, 1, 1, 1)),
            Ok(Offset4D::new(u32::MAX, 1, 1, 1))
        );
        assert_eq!(
            start.checked_add(Extent4D::new(1, 1, 1, 1)),
            Err(LayoutError::Overflow)
        );
    }
}
