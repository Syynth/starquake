//! Moving one level of one layer between its guest layout and the packed rows a render backend
//! takes (`command`'s module docs: rows of whole blocks with no padding, then slice after slice).
//!
//! Block-linear levels go through [`starquake_layout::tiled_to_linear`] and
//! [`starquake_layout::linear_to_tiled`] (decision 5: the layout crate does all detiling). A
//! pitch-linear level (one level of one layer, the layout crate's rule) is rows at the pitch, so
//! it is copied row by row here; there is nothing to detile.

use core::ops::Range;

use starquake_layout::{
    Bytes, Extent4D, Image, LayoutError, LinearLayout, Offset4D, Pixels, Tiling, linear_to_tiled,
    tiled_to_linear,
};

/// The shape of one level, for moving it.
#[derive(Clone, Copy, Debug)]
pub(super) struct LevelShape {
    /// The level's extent in pixels: what an upload region names.
    pub extent_px: Extent4D<Pixels>,
    /// Its extent in bytes: a packed row is `width` bytes, there are `height` rows of blocks per
    /// slice and `depth` slices.
    pub extent_b: Extent4D<Bytes>,
    /// The level's tiling.
    pub tiling: Tiling,
    /// Bytes from one guest row to the next (the pitch, for a pitch-linear level).
    pub row_stride_b: u32,
    /// Bytes of the level in one layer of guest memory.
    pub guest_len: u64,
}

impl LevelShape {
    /// The shape of level `level` of `image`.
    pub fn of(image: &Image, level: u32) -> Result<Self, LayoutError> {
        let info = usize::try_from(level)
            .ok()
            .and_then(|i| image.levels().get(i))
            .ok_or(LayoutError::LevelOutOfRange {
                level,
                levels: u32::try_from(image.levels().len()).unwrap_or(u32::MAX),
            })?;
        let mut extent_b = image.level_extent_b(level)?;
        extent_b.array_len = 1;
        let mut extent_px = image.level_extent_px(level)?;
        extent_px.array_len = 1;
        Ok(Self {
            extent_px,
            extent_b,
            tiling: info.tiling,
            row_stride_b: info.row_stride_b,
            guest_len: image.level_layer_size_b(level)?,
        })
    }

    /// Bytes of one packed row.
    pub fn row_b(&self) -> u64 {
        u64::from(self.extent_b.width)
    }

    /// Bytes of the whole level of one layer, packed.
    pub fn packed_len(&self) -> Result<u64, LayoutError> {
        self.row_b()
            .checked_mul(u64::from(self.extent_b.height))
            .and_then(|slice| slice.checked_mul(u64::from(self.extent_b.depth)))
            .ok_or(LayoutError::Overflow)
    }

    fn linear(&self, rows: u32) -> Result<LinearLayout, LayoutError> {
        let row_stride_b =
            usize::try_from(self.extent_b.width).map_err(|_| LayoutError::Overflow)?;
        let plane_stride_b = row_stride_b
            .checked_mul(usize::try_from(rows).map_err(|_| LayoutError::Overflow)?)
            .ok_or(LayoutError::Overflow)?;
        Ok(LinearLayout {
            row_stride_b,
            plane_stride_b,
        })
    }

    /// The guest byte range of packed row `y` of a pitch-linear level.
    fn pitch_row(&self, y: u32) -> Result<Range<usize>, LayoutError> {
        let start = u64::from(y)
            .checked_mul(u64::from(self.row_stride_b))
            .ok_or(LayoutError::Overflow)?;
        let end = start
            .checked_add(self.row_b())
            .ok_or(LayoutError::Overflow)?;
        let start = usize::try_from(start).map_err(|_| LayoutError::Overflow)?;
        let end = usize::try_from(end).map_err(|_| LayoutError::Overflow)?;
        Ok(start..end)
    }

    /// Fills `out` with packed rows `y0..y0 + rows` of slice `z`, read from the level's guest
    /// bytes `guest`.
    pub fn read_rows(
        &self,
        guest: &[u8],
        z: u32,
        y0: u32,
        rows: u32,
        out: &mut [u8],
    ) -> Result<(), LayoutError> {
        if self.tiling.is_tiled() {
            let extent = Extent4D::new(self.extent_b.width, rows, 1, 1);
            return tiled_to_linear(
                out,
                self.linear(rows)?,
                guest,
                self.extent_b,
                Offset4D::new(0, y0, z, 0),
                extent,
                &self.tiling,
            );
        }
        let row = usize::try_from(self.row_b()).map_err(|_| LayoutError::Overflow)?;
        for (i, packed) in (0..rows).zip(out.chunks_exact_mut(row.max(1))) {
            let y = y0.checked_add(i).ok_or(LayoutError::Overflow)?;
            let source = guest
                .get(self.pitch_row(y)?)
                .ok_or(LayoutError::RegionOutOfBounds)?;
            packed.copy_from_slice(source);
        }
        Ok(())
    }

    /// Writes the whole packed level `packed` (every row of every slice) into the level's guest
    /// bytes `guest`, leaving the padding as it was.
    pub fn write_level(&self, packed: &[u8], guest: &mut [u8]) -> Result<(), LayoutError> {
        if self.tiling.is_tiled() {
            return linear_to_tiled(
                guest,
                self.extent_b,
                packed,
                self.linear(self.extent_b.height)?,
                Offset4D::new(0, 0, 0, 0),
                self.extent_b,
                &self.tiling,
            );
        }
        let row = usize::try_from(self.row_b()).map_err(|_| LayoutError::Overflow)?;
        let rows = packed
            .chunks_exact(row.max(1))
            .take(usize::try_from(self.extent_b.height).map_err(|_| LayoutError::Overflow)?);
        for (y, source) in (0..self.extent_b.height).zip(rows) {
            let target = guest
                .get_mut(self.pitch_row(y)?)
                .ok_or(LayoutError::RegionOutOfBounds)?;
            target.copy_from_slice(source);
        }
        Ok(())
    }
}
