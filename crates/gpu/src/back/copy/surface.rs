//! One side of a copy: a pitch-linear or block-linear surface in GPU memory, walked line by line
//! in runs that are contiguous in GPU memory. The DMA copy, the inline upload and the 2D blit
//! all read and write through it, so the three agree on every layout question.
//!
//! A copy's region is `width` bytes by `lines` lines. On a pitch surface, line `l` starts at
//! `address + l * pitch` (the pitch may be negative for the DMA class). On a block-linear surface
//! the region starts at the surface's origin, and line `l` is row `origin_y + l % per` of depth
//! slice `l / per`, where `per = height - origin_y`: lines past the surface's last row continue
//! in the next slice, as Eden-NX2's `SwizzleSubrectImpl` and `UnswizzleSubrectImpl` do
//! (`eden-nx2/src/video_core/textures/decoders.cpp` @ 50fa86bb, the `unprocessed_lines` loop).
//! Every byte offset in a block-linear surface comes from [`starquake_layout::block_linear_offset`]
//! (decision 5: the layout crate does all detiling), one run per 16-byte GOB line
//! ([`GOB_LINE_WIDTH_B`]), the largest piece contiguous in both layouts.
//!
//! The GOB is [`GobType::TuringColor2d`], the Switch 2 colour GOB (row 69; the layout crate's
//! `tiling` docs). Whether depth and stencil surfaces use another GOB is a **guess** left open by
//! the spec (verification item 9, "the depth GOB kind"): a depth target dump settles it. The
//! copy classes carry no memory kind, so every block-linear side uses the colour GOB.

use core::ops::Range;

use starquake_layout::{
    Bytes, Extent4D, GOB_LINE_WIDTH_B, GobType, LayoutError, Offset4D, Tiling, block_linear_offset,
};
use starquake_memory::PAGE_SIZE;

use super::CopyError;
use crate::memory::GpuMemory;

/// Which side of a copy a surface or an error is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CopySide {
    /// What is read.
    Source,
    /// What is written.
    Destination,
}

impl core::fmt::Display for CopySide {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Source => "source",
            Self::Destination => "destination",
        })
    }
}

/// The block size fields of a block-linear surface, as log2 of the number of GOBs, before
/// validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockFields {
    /// `BLOCK_SIZE_WIDTH`: only `ONE_GOB` (0) is defined in any of the classes.
    pub width: u32,
    /// `BLOCK_SIZE_HEIGHT`: `ONE_GOB` (0) to `THIRTYTWO_GOBS` (5).
    pub height: u32,
    /// `BLOCK_SIZE_DEPTH`: `ONE_GOB` (0) to `THIRTYTWO_GOBS` (5).
    pub depth: u32,
}

/// The largest block height or depth any copy class defines: `THIRTYTWO_GOBS`, 5, in
/// `clc7b5.h`, `cla140.h`, `clc797.h`, `clc7c0.h` and `cl902d.h` alike (pinned by
/// `tests::surface::every_class_defines_the_same_largest_block`).
pub(crate) const MAX_BLOCK_LOG2: u32 = crate::classes::clc7b5::SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS;

/// The only block width any copy class defines: `ONE_GOB`, 0.
pub(crate) const ONE_GOB: u32 = crate::classes::clc7b5::SET_DST_BLOCK_SIZE_WIDTH_ONE_GOB;

/// What a block-linear surface's registers say, before validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockLinearDesc {
    /// The block size fields.
    pub block: BlockFields,
    /// The width in bytes.
    pub width_b: u64,
    /// The height in rows.
    pub height: u32,
    /// The depth in slices.
    pub depth: u32,
    /// The region's first byte in x.
    pub origin_x_b: u64,
    /// The region's first row.
    pub origin_y: u32,
}

/// A validated block-linear surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlockLinear {
    tiling: Tiling,
    extent: Extent4D<Bytes>,
    origin_x: u32,
    origin_y: u32,
}

/// How a surface's bytes are arranged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// Line `l` starts at `address + l * pitch`.
    Pitch { pitch: i64 },
    /// Block-linear, through the layout crate.
    BlockLinear(BlockLinear),
}

/// One side of a copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Surface {
    side: CopySide,
    address: u64,
    shape: Shape,
}

/// A copy's region: `width` bytes in each of `lines` lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    /// Bytes per line.
    pub width: u64,
    /// Lines.
    pub lines: u32,
}

impl Region {
    /// Whether the region holds no bytes.
    pub fn is_empty(self) -> bool {
        self.width == 0 || self.lines == 0
    }

    /// The region's size in bytes, which cannot overflow a `u128`.
    pub fn bytes(self) -> u128 {
        u128::from(self.width) * u128::from(self.lines)
    }
}

/// log2 of one block dimension, checked against [`MAX_BLOCK_LOG2`].
fn block_log2(value: u32, error: CopyError) -> Result<u8, CopyError> {
    if value > MAX_BLOCK_LOG2 {
        return Err(error);
    }
    u8::try_from(value).map_err(|_| error)
}

impl Surface {
    /// A pitch-linear surface whose line `l` starts at `address + l * pitch`.
    pub fn pitch(side: CopySide, address: u64, pitch: i64) -> Self {
        Self {
            side,
            address,
            shape: Shape::Pitch { pitch },
        }
    }

    /// A block-linear surface, checked in this order: the block width
    /// ([`CopyError::BlockWidth`]), height and depth ([`CopyError::BlockHeight`],
    /// [`CopyError::BlockDepth`]), then the width and origin in bytes, which must fit 32 bits as
    /// the layout crate counts them ([`CopyError::Layout`] with [`LayoutError::Overflow`]).
    pub fn block_linear(
        side: CopySide,
        address: u64,
        desc: &BlockLinearDesc,
    ) -> Result<Self, CopyError> {
        if desc.block.width != ONE_GOB {
            return Err(CopyError::BlockWidth {
                side,
                value: desc.block.width,
            });
        }
        let height = block_log2(
            desc.block.height,
            CopyError::BlockHeight {
                side,
                value: desc.block.height,
            },
        )?;
        let depth = block_log2(
            desc.block.depth,
            CopyError::BlockDepth {
                side,
                value: desc.block.depth,
            },
        )?;
        let layout = |error| CopyError::Layout { side, error };
        let tiling = Tiling::new(GobType::TuringColor2d, 0, height, depth).map_err(layout)?;
        let width = u32::try_from(desc.width_b).map_err(|_| layout(LayoutError::Overflow))?;
        let origin_x =
            u32::try_from(desc.origin_x_b).map_err(|_| layout(LayoutError::Overflow))?;
        Ok(Self {
            side,
            address,
            shape: Shape::BlockLinear(BlockLinear {
                tiling,
                extent: Extent4D::new(width, desc.height, desc.depth, 1),
                origin_x,
                origin_y: desc.origin_y,
            }),
        })
    }

    fn layout_error(&self, error: LayoutError) -> CopyError {
        CopyError::Layout {
            side: self.side,
            error,
        }
    }

    fn extent_error(&self) -> CopyError {
        CopyError::Extent { side: self.side }
    }

    /// Checks that `region` lies inside the surface and that every address it reaches is
    /// representable, and returns the GPU range that covers it (`None` for an empty region).
    ///
    /// For a pitch surface the cover runs from the lowest line start to the end of the highest
    /// line ([`CopyError::Extent`] when either is outside the 64-bit address space). For a
    /// block-linear surface the region must lie inside the surface ([`CopyError::Layout`] with
    /// [`LayoutError::RegionOutOfBounds`]: its x end past the width, its first row at or past
    /// the height, or its last line in a slice at or past the depth), and the cover is the
    /// whole surface padded to whole blocks, from its address ([`CopyError::Extent`] when that
    /// end is not representable).
    pub fn cover(&self, region: Region) -> Result<Option<Range<u64>>, CopyError> {
        if region.is_empty() {
            return Ok(None);
        }
        match self.shape {
            Shape::Pitch { pitch } => {
                let first = self.line_start(0, pitch)?;
                let last = self.line_start(region.lines - 1, pitch)?;
                let (low, high) = if first <= last {
                    (first, last)
                } else {
                    (last, first)
                };
                let end = high
                    .checked_add(region.width)
                    .ok_or_else(|| self.extent_error())?;
                Ok(Some(low..end))
            }
            Shape::BlockLinear(surface) => {
                let out = || self.layout_error(LayoutError::RegionOutOfBounds);
                let x_end = u64::from(surface.origin_x)
                    .checked_add(region.width)
                    .ok_or_else(out)?;
                if x_end > u64::from(surface.extent.width) {
                    return Err(out());
                }
                let (z, _) = surface.position(region.lines - 1).ok_or_else(out)?;
                if z >= surface.extent.depth {
                    return Err(out());
                }
                let size = padded_size(&surface).ok_or_else(|| self.extent_error())?;
                let end = self
                    .address
                    .checked_add(size)
                    .ok_or_else(|| self.extent_error())?;
                Ok(Some(self.address..end))
            }
        }
    }

    /// The GPU address of line `line` of a pitch surface.
    fn line_start(&self, line: u32, pitch: i64) -> Result<u64, CopyError> {
        let start = i128::from(self.address) + i128::from(line) * i128::from(pitch);
        u64::try_from(start).map_err(|_| self.extent_error())
    }

    /// Calls `f` with the GPU address of each run of bytes `x` (relative to the region's first
    /// byte) of line `line`, and the run's range within `x`, in increasing x. The region must
    /// have passed [`Surface::cover`].
    fn runs(
        &self,
        line: u32,
        x: Range<u64>,
        mut f: impl FnMut(u64, Range<usize>) -> Result<(), CopyError>,
    ) -> Result<(), CopyError> {
        if x.is_empty() {
            return Ok(());
        }
        let len = usize::try_from(x.end - x.start).map_err(|_| self.extent_error())?;
        match self.shape {
            Shape::Pitch { pitch } => {
                let va = self
                    .line_start(line, pitch)?
                    .checked_add(x.start)
                    .ok_or_else(|| self.extent_error())?;
                f(va, 0..len)
            }
            Shape::BlockLinear(surface) => {
                let out = || self.layout_error(LayoutError::RegionOutOfBounds);
                let (z, y) = surface.position(line).ok_or_else(out)?;
                let start = u64::from(surface.origin_x)
                    .checked_add(x.start)
                    .and_then(|start| u32::try_from(start).ok())
                    .ok_or_else(out)?;
                let mut done = 0usize;
                while done < len {
                    let sx = u32::try_from(done)
                        .ok()
                        .and_then(|done| start.checked_add(done))
                        .ok_or_else(out)?;
                    let to_line_end = GOB_LINE_WIDTH_B - sx % GOB_LINE_WIDTH_B;
                    let step = usize::try_from(to_line_end).map_or(len - done, |step| {
                        step.min(len - done)
                    });
                    let offset = block_linear_offset(
                        &surface.tiling,
                        surface.extent,
                        Offset4D::new(sx, y, z, 0),
                    )
                    .map_err(|error| self.layout_error(error))?;
                    let va = self
                        .address
                        .checked_add(offset)
                        .ok_or_else(|| self.extent_error())?;
                    f(va, done..done + step)?;
                    done += step;
                }
                Ok(())
            }
        }
    }

    /// Reads bytes `x` of line `line` into `buf` (`buf.len()` must be the length of `x`).
    pub fn read<M: GpuMemory + ?Sized>(
        &self,
        memory: &M,
        line: u32,
        x: Range<u64>,
        buf: &mut [u8],
    ) -> Result<(), CopyError> {
        self.runs(line, x, |va, range| {
            let target = buf.get_mut(range).ok_or_else(|| self.extent_error())?;
            memory.read(va, target).map_err(|fault| CopyError::Fault {
                side: self.side,
                fault,
            })
        })
    }

    /// Writes `buf` to bytes `x` of line `line`, and calls `written` with each run written.
    pub fn write<M: GpuMemory + ?Sized>(
        &self,
        memory: &M,
        line: u32,
        x: Range<u64>,
        buf: &[u8],
        mut written: impl FnMut(Range<u64>),
    ) -> Result<(), CopyError> {
        self.runs(line, x, |va, range| {
            let len = range.len();
            let source = buf.get(range).ok_or_else(|| self.extent_error())?;
            memory.write(va, source).map_err(|fault| CopyError::Fault {
                side: self.side,
                fault,
            })?;
            // `va + len` was checked when the run was made: it lies inside the cover.
            written(va..va.saturating_add(len as u64));
            Ok(())
        })
    }

    /// Checks, before anything is written, that every page bytes `x` of each line in `lines`
    /// reach is mapped, by reading one byte of each: the run's first byte, then the first byte
    /// of each later page it reaches. The first that faults is [`CopyError::Fault`] on this side,
    /// naming the first unmapped byte the copy itself would have reached.
    pub fn probe<M: GpuMemory + ?Sized>(
        &self,
        memory: &M,
        lines: Range<u32>,
        x: Range<u64>,
    ) -> Result<(), CopyError> {
        let mut last_page = None;
        let mut byte = [0u8; 1];
        for line in lines {
            self.runs(line, x.clone(), |va, range| {
                let end = va.saturating_add(range.len() as u64);
                let mut at = va;
                while at < end {
                    let page = at - at % PAGE_SIZE;
                    if last_page != Some(page) {
                        memory
                            .read(at, &mut byte)
                            .map_err(|fault| CopyError::Fault {
                                side: self.side,
                                fault,
                            })?;
                        last_page = Some(page);
                    }
                    at = page.saturating_add(PAGE_SIZE);
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}

impl BlockLinear {
    /// The depth slice and row of line `line` of the region, or `None` when the origin is at
    /// or past the height.
    fn position(&self, line: u32) -> Option<(u32, u32)> {
        let per = self.extent.height.checked_sub(self.origin_y)?;
        if per == 0 {
            return None;
        }
        Some((line / per, self.origin_y + line % per))
    }
}

/// The bytes of a block-linear surface padded to whole blocks: the width, height and depth each
/// rounded up to the block's, multiplied (the layout crate's padded level size, `Strides` in
/// `starquake-layout`'s `tiling.rs`). `None` when it does not fit 64 bits.
fn padded_size(surface: &BlockLinear) -> Option<u64> {
    let block = surface.tiling.extent_b();
    let pad = |value: u32, to: u32| u64::from(value).checked_next_multiple_of(u64::from(to));
    pad(surface.extent.width, block.width)?
        .checked_mul(pad(surface.extent.height, block.height)?)?
        .checked_mul(pad(surface.extent.depth, block.depth)?)
}
