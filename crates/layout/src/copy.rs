//! Copies between a block-linear level and linear memory: tiling and untiling a region.
//!
//! Ported from `mesa-nouveau/src/nouveau/nil/copy.rs` @ 52238762 (MIT, Copyright © 2024 Valve
//! Corp. and Collabora, Ltd.; the licence text is in this crate's `lib.rs`):
//! `nil_copy_linear_to_tiled` and `nil_copy_tiled_to_linear`, with the same arguments, over
//! byte slices instead of raw pointers.
//!
//! Mesa walks the region tile by tile, GOB by GOB and line by line, copying whole GOBs without
//! bounds checks where it can. This walks it row by row in pieces that end at a 16-byte line
//! boundary ([`GOB_LINE_WIDTH_B`]), since a line is the largest run that is contiguous in both
//! layouts for every GOB kind here, and finds each piece's tiled offset with
//! [`crate::block_linear_offset`]'s arithmetic. The bytes moved are the same; every index is
//! checked.
//!
//! # Differences from Mesa
//!
//! Mesa checks none of its inputs; here each is an error, in this order:
//!
//! 1. [`LayoutError::RegionHasLayers`]: the region must be one layer at layer 0 (Mesa
//!    `debug_assert`s `start.a == 0` and `end.a == 1`). A copy works on one layer; the caller
//!    passes that layer's level bytes ([`crate::Image::level_layer_offset_b`]).
//! 2. [`LayoutError::RegionOutOfBounds`]: the region must lie inside the level, including when
//!    its end does not fit in 32 bits.
//! 3. [`LayoutError::UnsupportedGob`]: `Linear` and `FermiZS` (Mesa panics).
//! 4. [`LayoutError::Overflow`]: the padded level's size does not fit in 64 bits.
//! 5. [`LayoutError::TiledTooSmall`]: the tiled slice must hold the whole level padded to
//!    whole tiles.
//! 6. [`LayoutError::LinearTooSmall`]: the linear slice must reach the region's last byte,
//!    `(depth - 1) * plane_stride + (height - 1) * row_stride + width`.
//!
//! A region with a zero width, height or depth copies nothing once those checks pass.

use core::ops::Range;

use crate::error::{LayoutError, Result};
use crate::extent::units::Bytes;
use crate::extent::{Extent4D, Offset4D};
use crate::tiling::{GOB_LINE_WIDTH_B, Strides, Tiling};

/// How the linear side of a copy is laid out: the region's first byte is the slice's first
/// byte, and rows and slices follow at these strides (Mesa: `LinearPointer`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LinearLayout {
    /// Bytes from one row to the next.
    pub row_stride_b: usize,
    /// Bytes from one depth slice to the next.
    pub plane_stride_b: usize,
}

/// A checked copy: the strides of the level and where the region is.
struct Plan<'a> {
    tiling: &'a Tiling,
    strides: Strides,
    linear: LinearLayout,
    offset: Offset4D<Bytes>,
    extent: Extent4D<Bytes>,
}

fn to_usize(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| LayoutError::Overflow)
}

fn checked_range(start: u64, len: u32) -> Result<Range<usize>> {
    let start = to_usize(start)?;
    let len = usize::try_from(len).map_err(|_| LayoutError::Overflow)?;
    let end = start.checked_add(len).ok_or(LayoutError::Overflow)?;
    Ok(start..end)
}

impl<'a> Plan<'a> {
    /// Checks a copy's arguments in the order the module docs give.
    fn new(
        tiling: &'a Tiling,
        level_extent_b: Extent4D<Bytes>,
        tiled_len: usize,
        linear: LinearLayout,
        linear_len: usize,
        offset: Offset4D<Bytes>,
        extent: Extent4D<Bytes>,
    ) -> Result<Self> {
        if offset.a != 0 || extent.array_len != 1 {
            return Err(LayoutError::RegionHasLayers);
        }
        let end = offset
            .checked_add(extent)
            .map_err(|_| LayoutError::RegionOutOfBounds)?;
        if end.x > level_extent_b.width
            || end.y > level_extent_b.height
            || end.z > level_extent_b.depth
        {
            return Err(LayoutError::RegionOutOfBounds);
        }
        tiling.gob_type().lines()?;
        let strides = Strides::new(tiling, level_extent_b)?;
        if u64::try_from(tiled_len).is_ok_and(|len| len < strides.level_size()) {
            return Err(LayoutError::TiledTooSmall {
                needed: strides.level_size(),
                len: tiled_len,
            });
        }
        let needed = if extent.is_empty() {
            0
        } else {
            let last = |count: u32, stride: usize| {
                u64::try_from(stride)
                    .ok()
                    .and_then(|stride| stride.checked_mul(u64::from(count - 1)))
            };
            last(extent.depth, linear.plane_stride_b)
                .zip(last(extent.height, linear.row_stride_b))
                .and_then(|(planes, rows)| planes.checked_add(rows))
                .and_then(|before| before.checked_add(u64::from(extent.width)))
                .ok_or(LayoutError::Overflow)?
        };
        if u64::try_from(linear_len).is_ok_and(|len| len < needed) {
            return Err(LayoutError::LinearTooSmall {
                needed,
                len: linear_len,
            });
        }
        Ok(Self {
            tiling,
            strides,
            linear,
            offset,
            extent,
        })
    }

    /// Calls `f` with the tiled and linear byte ranges of each piece of the region, row by row,
    /// each piece ending at a line boundary of the level or at the region's edge.
    fn for_each_piece(
        &self,
        mut f: impl FnMut(Range<usize>, Range<usize>) -> Result<()>,
    ) -> Result<()> {
        if self.extent.is_empty() {
            return Ok(());
        }
        for z in 0..self.extent.depth {
            for y in 0..self.extent.height {
                let row_start = usize::try_from(z)
                    .ok()
                    .and_then(|z| z.checked_mul(self.linear.plane_stride_b))
                    .zip(
                        usize::try_from(y)
                            .ok()
                            .and_then(|y| y.checked_mul(self.linear.row_stride_b)),
                    )
                    .and_then(|(plane, row)| plane.checked_add(row))
                    .ok_or(LayoutError::Overflow)?;
                let mut x = 0;
                while x < self.extent.width {
                    // `Plan::new` checked the region's end fits in the level, so these sums of
                    // a start and a position inside the region cannot overflow.
                    let level_x = self.offset.x + x;
                    let to_line_end = GOB_LINE_WIDTH_B - level_x % GOB_LINE_WIDTH_B;
                    let len = to_line_end.min(self.extent.width - x);
                    let position = Offset4D::new(level_x, self.offset.y + y, self.offset.z + z, 0);
                    let tiled = self.strides.offset(self.tiling, position)?;
                    let linear = u64::try_from(row_start)
                        .ok()
                        .and_then(|start| start.checked_add(u64::from(x)))
                        .ok_or(LayoutError::Overflow)?;
                    f(checked_range(tiled, len)?, checked_range(linear, len)?)?;
                    x += len;
                }
            }
        }
        Ok(())
    }
}

/// Copies a region of a block-linear level into linear memory: untiling (Mesa:
/// `nil_copy_tiled_to_linear`).
///
/// `tiled` holds the level (one layer of it) from its first byte, `level_extent_b` is the
/// level's unpadded extent in bytes ([`crate::Image::level_extent_b`]) and `tiling` the level's
/// tiling ([`crate::ImageLevel::tiling`]). The region starts at `offset_b` and spans
/// `extent_b`, both in bytes; its first byte lands at `linear[0]`. Every check runs before the
/// first byte is written (see the module docs for the checks and their order).
pub fn tiled_to_linear(
    linear: &mut [u8],
    linear_layout: LinearLayout,
    tiled: &[u8],
    level_extent_b: Extent4D<Bytes>,
    offset_b: Offset4D<Bytes>,
    extent_b: Extent4D<Bytes>,
    tiling: &Tiling,
) -> Result<()> {
    let plan = Plan::new(
        tiling,
        level_extent_b,
        tiled.len(),
        linear_layout,
        linear.len(),
        offset_b,
        extent_b,
    )?;
    plan.for_each_piece(|tiled_range, linear_range| {
        let source = tiled.get(tiled_range).ok_or(LayoutError::Overflow)?;
        let target = linear.get_mut(linear_range).ok_or(LayoutError::Overflow)?;
        target.copy_from_slice(source);
        Ok(())
    })
}

/// Copies a region of linear memory into a block-linear level: tiling (Mesa:
/// `nil_copy_linear_to_tiled`).
///
/// The arguments mean what they mean for [`tiled_to_linear`], with the direction reversed.
/// Bytes of the level outside the region are left as they were.
pub fn linear_to_tiled(
    tiled: &mut [u8],
    level_extent_b: Extent4D<Bytes>,
    linear: &[u8],
    linear_layout: LinearLayout,
    offset_b: Offset4D<Bytes>,
    extent_b: Extent4D<Bytes>,
    tiling: &Tiling,
) -> Result<()> {
    let plan = Plan::new(
        tiling,
        level_extent_b,
        tiled.len(),
        linear_layout,
        linear.len(),
        offset_b,
        extent_b,
    )?;
    plan.for_each_piece(|tiled_range, linear_range| {
        let source = linear.get(linear_range).ok_or(LayoutError::Overflow)?;
        let target = tiled.get_mut(tiled_range).ok_or(LayoutError::Overflow)?;
        target.copy_from_slice(source);
        Ok(())
    })
}
