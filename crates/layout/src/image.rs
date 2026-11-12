//! Image layout: where each level and layer of an image starts, its tiling, the mip tail, and
//! the image's total size.
//!
//! Ported from `mesa-nouveau/src/nouveau/nil/image.rs` @ 52238762 (MIT, Copyright © 2024
//! Collabora, Ltd.; the licence text is in this crate's `lib.rs`): `SampleLayout`, `ImageDim`,
//! `Image::new`, `new_linear`, `new_tiled` and the level queries.
//!
//! An image is an array of layers; each layer holds every mip level in turn, level 0 first. A
//! block-linear level is padded to whole tiles of its own tiling (the requested tiling clamped to
//! the level, as the hardware does at each level), and the layer stride is the layer's size
//! rounded up to a whole level-0 tile.
//!
//! # Differences from Mesa
//!
//! - The caller gives the tiling ([`ImageDesc::tiling`]) and, for a pitch-linear image, the
//!   pitch; Mesa chooses them (`Tiling::choose`, DRM modifiers, sparse residency, video usage),
//!   and none of that is ported. Neither are PTE kinds, the tile mode word, Kepler's 256-byte
//!   linear alignment (no Ampere class needs it), or the view helpers (`image_for_level`,
//!   `level_as_uncompressed`, `_3d_level_as_2d_array`, `msaa_as_samples`).
//! - A pitch-linear image is chosen by [`Tiling::LINEAR`] rather than a usage bit.
//! - Every Mesa `assert!` on the description is an error instead: [`LayoutError::Invalid1d`],
//!   [`LayoutError::Invalid2d`], [`LayoutError::Invalid3d`], [`LayoutError::InvalidLinear`],
//!   [`LayoutError::MultisampleMipmaps`] (Mesa asserts it in `level_extent_px` for any level but
//!   0), [`LayoutError::InvalidSampleCount`] (Mesa's `SampleLayout::Invalid`, which panics when
//!   used) and the level and layer range checks. A zero extent ([`LayoutError::ZeroExtent`]) and
//!   a level count outside 1 to [`MAX_LEVELS`] ([`LayoutError::InvalidLevelCount`]; Mesa indexes
//!   its 16-entry array out of bounds) are refused too.
//! - Pitch checks are stricter: a pitch must be a multiple of [`LINEAR_PITCH_ALIGN_B`] in every
//!   build (Mesa: `debug_assert!`), must hold one row of the image
//!   ([`LayoutError::PitchTooSmall`]; Mesa does not check), and cannot be given for a
//!   block-linear image ([`LayoutError::PitchOnBlockLinear`]; Mesa asserts it equals every
//!   level's padded width).
//! - The mip tail is computed for every block-linear image, as Mesa computes it for a sparse one:
//!   it starts at the first level whose clamped tiling differs from the requested one. Mesa
//!   leaves it 0 for other images, meaning none, as only sparse binding uses it. Here "none" is
//!   `None`, and a tail starting at level 0 (the requested tiling is larger than level 0) is
//!   `Some(0)`, whose offset is 0; Mesa asserts the first tail level is above 0.
//! - Sizes and offsets are `u64` with checked arithmetic throughout (Mesa computes
//!   `level_z_offset_B` and `level_depth_stride_B` in `u32`, and returns the mip tail size as
//!   `u32`).

use crate::error::{LayoutError, Result};
use crate::extent::Extent4D;
use crate::extent::units::{Bytes, Pixels, Samples};
use crate::format::Format;
use crate::tiling::Tiling;

/// The most mip levels an image can have (Mesa: `MAX_LEVELS`).
pub const MAX_LEVELS: u32 = 16;

/// The alignment a given pitch must have, in bytes (Mesa `new_linear`: "The texture headers can
/// handle as low as 32B-aligned").
pub const LINEAR_PITCH_ALIGN_B: u32 = 32;

/// The alignment of a pitch this crate chooses, in bytes (Mesa `new_linear`: "require 128B so
/// that we can render to the image without workarounds"). It is also a pitch-linear image's
/// alignment when no pitch is given.
pub const LINEAR_DEFAULT_ALIGN_B: u32 = 128;

/// The smallest alignment of a block-linear image, in bytes (Mesa `new_tiled`:
/// `max(align_B, 4096)`). The image's size is rounded up to its alignment. Mesa's allocation
/// rule; whether NVN2 rounds the same way is a **guess**, which graphics-server slice 8's size
/// comparison will confirm.
pub const MIN_IMAGE_ALIGN_B: u32 = 4096;

/// An image's dimensionality (Mesa: `ImageDim`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageDim {
    /// One-dimensional: height and depth are 1.
    D1,
    /// Two-dimensional: depth is 1.
    D2,
    /// Three-dimensional: one layer.
    D3,
}

/// How a pixel's samples are laid out as a grid of samples (Mesa: `SampleLayout`).
///
/// The `D3d` variants have the same grid as their plain twins and differ only in where each
/// sample sits inside a pixel, which does not affect layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleLayout {
    /// One sample.
    S1x1,
    /// Two samples side by side.
    S2x1,
    /// Two samples side by side, Direct3D positions.
    S2x1D3d,
    /// Four samples, 2 by 2.
    S2x2,
    /// Eight samples, 4 by 2.
    S4x2,
    /// Eight samples, 4 by 2, Direct3D positions.
    S4x2D3d,
    /// Sixteen samples, 4 by 4.
    S4x4,
}

impl SampleLayout {
    /// The layout Mesa picks for `samples` samples per pixel (`choose_sample_layout`): 1, 2, 4,
    /// 8 or 16; any other count is [`LayoutError::InvalidSampleCount`].
    pub fn from_samples(samples: u32) -> Result<Self> {
        match samples {
            1 => Ok(SampleLayout::S1x1),
            2 => Ok(SampleLayout::S2x1D3d),
            4 => Ok(SampleLayout::S2x2),
            8 => Ok(SampleLayout::S4x2D3d),
            16 => Ok(SampleLayout::S4x4),
            other => Err(LayoutError::InvalidSampleCount(other)),
        }
    }

    /// Samples per pixel.
    pub fn samples(self) -> u32 {
        let grid = self.px_extent_sa();
        grid.width * grid.height
    }

    /// The grid of samples one pixel covers (Mesa: `px_extent_sa`).
    pub(crate) fn px_extent_sa(self) -> Extent4D<Samples> {
        match self {
            SampleLayout::S1x1 => Extent4D::new(1, 1, 1, 1),
            SampleLayout::S2x1 | SampleLayout::S2x1D3d => Extent4D::new(2, 1, 1, 1),
            SampleLayout::S2x2 => Extent4D::new(2, 2, 1, 1),
            SampleLayout::S4x2 | SampleLayout::S4x2D3d => Extent4D::new(4, 2, 1, 1),
            SampleLayout::S4x4 => Extent4D::new(4, 4, 1, 1),
        }
    }
}

/// What an image is: everything its layout depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageDesc {
    /// 1D, 2D or 3D.
    pub dim: ImageDim,
    /// The element size and block.
    pub format: Format,
    /// Level 0's width, height and depth in pixels, and the layer count.
    pub extent_px: Extent4D<Pixels>,
    /// Mip levels, 1 to [`MAX_LEVELS`].
    pub levels: u32,
    /// The sample layout.
    pub sample_layout: SampleLayout,
    /// The requested tiling: [`Tiling::LINEAR`] for a pitch-linear image.
    pub tiling: Tiling,
    /// A pitch-linear image's pitch in bytes, or `None` to round a row up to
    /// [`LINEAR_DEFAULT_ALIGN_B`]. Must be `None` for a block-linear image.
    pub pitch_b: Option<u32>,
}

/// Where one level sits in each layer, and how it is tiled (Mesa: `ImageLevel`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageLevel {
    /// Offset of the level from the start of the layer, in bytes.
    pub offset_b: u64,
    /// The level's tiling: the requested tiling clamped to the level.
    pub tiling: Tiling,
    /// Bytes from one row of elements to the next: the padded width of a block-linear level, or
    /// the pitch of a pitch-linear one.
    pub row_stride_b: u32,
}

/// The computed layout of an image (Mesa: `Image`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Image {
    dim: ImageDim,
    format: Format,
    extent_px: Extent4D<Pixels>,
    sample_layout: SampleLayout,
    levels: Vec<ImageLevel>,
    mip_tail_first_level: Option<u32>,
    array_stride_b: u64,
    align_b: u32,
    size_b: u64,
}

fn overflow<T>(value: Option<T>) -> Result<T> {
    value.ok_or(LayoutError::Overflow)
}

impl Image {
    /// Lays out the image `desc` describes (Mesa: `Image::new`).
    ///
    /// The checks run in this order: [`LayoutError::ZeroExtent`],
    /// [`LayoutError::InvalidLevelCount`], the dimension's rule ([`LayoutError::Invalid1d`],
    /// [`LayoutError::Invalid2d`] or [`LayoutError::Invalid3d`]),
    /// [`LayoutError::MultisampleMipmaps`], then the pitch-linear or block-linear rules; any
    /// size that does not fit is [`LayoutError::Overflow`].
    pub fn new(desc: &ImageDesc) -> Result<Self> {
        if desc.extent_px.is_empty() {
            return Err(LayoutError::ZeroExtent);
        }
        if desc.levels == 0 || desc.levels > MAX_LEVELS {
            return Err(LayoutError::InvalidLevelCount {
                levels: desc.levels,
                max: MAX_LEVELS,
            });
        }
        let single_sample = desc.sample_layout == SampleLayout::S1x1;
        let extent = desc.extent_px;
        match desc.dim {
            ImageDim::D1 if extent.height != 1 || extent.depth != 1 || !single_sample => {
                return Err(LayoutError::Invalid1d);
            }
            ImageDim::D2 if extent.depth != 1 => return Err(LayoutError::Invalid2d),
            ImageDim::D3 if extent.array_len != 1 || !single_sample => {
                return Err(LayoutError::Invalid3d);
            }
            ImageDim::D1 | ImageDim::D2 | ImageDim::D3 => {}
        }
        if !single_sample && desc.levels != 1 {
            return Err(LayoutError::MultisampleMipmaps);
        }
        if desc.tiling.is_tiled() {
            Self::new_tiled(desc)
        } else {
            Self::new_linear(desc)
        }
    }

    /// Mesa's `new_linear`: one level, one row stride.
    fn new_linear(desc: &ImageDesc) -> Result<Self> {
        if desc.dim != ImageDim::D2
            || desc.extent_px.array_len != 1
            || desc.levels != 1
            || desc.sample_layout != SampleLayout::S1x1
        {
            return Err(LayoutError::InvalidLinear);
        }
        let extent_b = desc.extent_px.to_b(desc.format, desc.sample_layout)?;
        let (align_b, row_stride_b) = match desc.pitch_b {
            Some(pitch) => {
                if pitch % LINEAR_PITCH_ALIGN_B != 0 {
                    return Err(LayoutError::PitchNotAligned {
                        pitch,
                        align: LINEAR_PITCH_ALIGN_B,
                    });
                }
                if pitch < extent_b.width {
                    return Err(LayoutError::PitchTooSmall {
                        pitch,
                        row: extent_b.width,
                    });
                }
                (LINEAR_PITCH_ALIGN_B, pitch)
            }
            None => (
                LINEAR_DEFAULT_ALIGN_B,
                overflow(
                    extent_b
                        .width
                        .checked_next_multiple_of(LINEAR_DEFAULT_ALIGN_B),
                )?,
            ),
        };
        let size_b = u64::from(row_stride_b) * u64::from(extent_b.height);
        Ok(Self {
            dim: desc.dim,
            format: desc.format,
            extent_px: desc.extent_px,
            sample_layout: desc.sample_layout,
            levels: vec![ImageLevel {
                offset_b: 0,
                tiling: Tiling::LINEAR,
                row_stride_b,
            }],
            mip_tail_first_level: None,
            array_stride_b: 0,
            align_b,
            size_b,
        })
    }

    /// Mesa's `new_tiled`, with the tiling given.
    fn new_tiled(desc: &ImageDesc) -> Result<Self> {
        if desc.pitch_b.is_some() {
            return Err(LayoutError::PitchOnBlockLinear);
        }
        let mut levels = Vec::new();
        let mut mip_tail_first_level = None;
        let mut layer_size_b: u64 = 0;
        for level in 0..desc.levels {
            let mut level_extent_b = desc
                .extent_px
                .minify(level)
                .to_b(desc.format, desc.sample_layout)?;
            // "For the purposes of computing the size of a miplevel, we don't care about
            // arrays."
            level_extent_b.array_len = 1;
            let level_tiling = desc.tiling.clamp(level_extent_b)?;
            if level_tiling != desc.tiling && mip_tail_first_level.is_none() {
                mip_tail_first_level = Some(level);
            }
            let padded = level_extent_b.align(&level_tiling.extent_b())?;
            levels.push(ImageLevel {
                offset_b: layer_size_b,
                tiling: level_tiling,
                row_stride_b: padded.width,
            });
            layer_size_b = overflow(layer_size_b.checked_add(padded.size_b()?))?;
        }
        // "The array stride has to be aligned to the size of a level 0 tile", using level 0's
        // clamped tiling, which is the one programmed into the hardware.
        let level0_tile_b = levels.first().map_or(1, |level| level.tiling.size_b());
        let array_stride_b =
            overflow(layer_size_b.checked_next_multiple_of(u64::from(level0_tile_b)))?;
        let align_b = level0_tile_b.max(MIN_IMAGE_ALIGN_B);
        let size_b = overflow(
            array_stride_b
                .checked_mul(u64::from(desc.extent_px.array_len))
                .and_then(|size| size.checked_next_multiple_of(u64::from(align_b))),
        )?;
        Ok(Self {
            dim: desc.dim,
            format: desc.format,
            extent_px: desc.extent_px,
            sample_layout: desc.sample_layout,
            levels,
            mip_tail_first_level,
            array_stride_b,
            align_b,
            size_b,
        })
    }

    /// 1D, 2D or 3D.
    pub fn dim(&self) -> ImageDim {
        self.dim
    }

    /// The format.
    pub fn format(&self) -> Format {
        self.format
    }

    /// Level 0's extent in pixels, and the layer count.
    pub fn extent_px(&self) -> Extent4D<Pixels> {
        self.extent_px
    }

    /// The sample layout.
    pub fn sample_layout(&self) -> SampleLayout {
        self.sample_layout
    }

    /// Every level, level 0 first.
    pub fn levels(&self) -> &[ImageLevel] {
        &self.levels
    }

    /// The first level of the mip tail, or `None` if every level keeps the requested tiling.
    pub fn mip_tail_first_level(&self) -> Option<u32> {
        self.mip_tail_first_level
    }

    /// Bytes from one layer to the next (0 for a pitch-linear image, which has one layer).
    pub fn array_stride_b(&self) -> u64 {
        self.array_stride_b
    }

    /// The image's alignment in bytes.
    pub fn align_b(&self) -> u32 {
        self.align_b
    }

    /// The image's size in bytes.
    pub fn size_b(&self) -> u64 {
        self.size_b
    }

    fn level(&self, level: u32) -> Result<&ImageLevel> {
        usize::try_from(level)
            .ok()
            .and_then(|index| self.levels.get(index))
            .ok_or(LayoutError::LevelOutOfRange {
                level,
                levels: self.level_count(),
            })
    }

    fn level_count(&self) -> u32 {
        // `new` allows at most `MAX_LEVELS` levels.
        u32::try_from(self.levels.len()).unwrap_or(MAX_LEVELS)
    }

    /// Level `level`'s extent in pixels (Mesa: `level_extent_px`).
    pub fn level_extent_px(&self, level: u32) -> Result<Extent4D<Pixels>> {
        self.level(level)?;
        Ok(self.extent_px.minify(level))
    }

    /// Level `level`'s extent in bytes, unpadded (Mesa: `level_extent_B`): what a copy takes as
    /// its level extent.
    pub fn level_extent_b(&self, level: u32) -> Result<Extent4D<Bytes>> {
        self.level_extent_px(level)?
            .to_b(self.format, self.sample_layout)
    }

    /// Where level `level` of layer `layer` starts, in bytes from the start of the image (Mesa:
    /// `level_layer_offset_B`).
    pub fn level_layer_offset_b(&self, level: u32, layer: u32) -> Result<u64> {
        let offset_b = self.level(level)?.offset_b;
        if layer >= self.extent_px.array_len {
            return Err(LayoutError::LayerOutOfRange {
                layer,
                layers: self.extent_px.array_len,
            });
        }
        overflow(
            u64::from(layer)
                .checked_mul(self.array_stride_b)
                .and_then(|layer_offset| layer_offset.checked_add(offset_b)),
        )
    }

    /// Bytes of level `level` in one layer (Mesa: `level_layer_size_B`): the level padded to
    /// whole tiles, or for a pitch-linear image every row but the last at the pitch plus the
    /// last row's bytes.
    pub fn level_layer_size_b(&self, level: u32) -> Result<u64> {
        let info = self.level(level)?;
        let mut extent_b = self.level_extent_b(level)?;
        extent_b.array_len = 1;
        if info.tiling.is_tiled() {
            extent_b.align(&info.tiling.extent_b())?.size_b()
        } else {
            // `new` refuses a zero extent, so the height is at least 1.
            let rows_before_last = u64::from(extent_b.height.saturating_sub(1));
            overflow(
                rows_before_last
                    .checked_mul(u64::from(info.row_stride_b))
                    .and_then(|size| size.checked_add(u64::from(extent_b.width))),
            )
        }
    }

    /// Bytes from the start of level `level` in layer 0 to the end of it in the last layer
    /// (Mesa: `level_size_B`).
    pub fn level_size_b(&self, level: u32) -> Result<u64> {
        let layer_size_b = self.level_layer_size_b(level)?;
        // `new` refuses a zero layer count.
        let layers_before_last = u64::from(self.extent_px.array_len.saturating_sub(1));
        overflow(
            layers_before_last
                .checked_mul(self.array_stride_b)
                .and_then(|size| size.checked_add(layer_size_b)),
        )
    }

    /// Where depth slice `z` of level `level` starts, in bytes from the start of the level
    /// (Mesa: `level_z_offset_B`): whole layers of tiles below it, plus `z`'s place inside its
    /// tile. [`LayoutError::SliceOutOfRange`] when `z` is not below the level's depth.
    pub fn level_z_offset_b(&self, level: u32, z: u32) -> Result<u64> {
        let tiling = self.level(level)?.tiling;
        let extent_px = self.level_extent_px(level)?;
        if z >= extent_px.depth {
            return Err(LayoutError::SliceOutOfRange {
                z,
                depth: extent_px.depth,
            });
        }
        // A tile is 2^z_log2 slices deep (a GOB is one slice), so these are Mesa's
        // `z >> z_log2` and `z & ((1 << z_log2) - 1)`.
        let tile = tiling.extent_b();
        let z_tile = overflow(z.checked_div(tile.depth))?;
        let z_in_tile = overflow(z.checked_rem(tile.depth))?;
        let extent_tl = extent_px.to_tl(&tiling, self.format, self.sample_layout)?;
        let tiles_below = u64::from(extent_tl.width)
            .checked_mul(u64::from(extent_tl.height))
            .and_then(|plane| plane.checked_mul(u64::from(z_tile)))
            .and_then(|tiles| tiles.checked_mul(u64::from(tiling.size_b())));
        let in_tile = u64::from(tile.width)
            .checked_mul(u64::from(tile.height))
            .and_then(|plane| plane.checked_mul(u64::from(z_in_tile)));
        overflow(
            tiles_below
                .zip(in_tile)
                .and_then(|(below, inside)| below.checked_add(inside)),
        )
    }

    /// Where the mip tail starts, in bytes from the start of a layer (Mesa:
    /// `mip_tail_offset_B`); [`LayoutError::NoMipTail`] when the image has none.
    pub fn mip_tail_offset_b(&self) -> Result<u64> {
        let first = self.mip_tail_first_level.ok_or(LayoutError::NoMipTail)?;
        Ok(self.level(first)?.offset_b)
    }

    /// Bytes from the start of the mip tail to the end of the layer stride (Mesa:
    /// `mip_tail_size_B`); [`LayoutError::NoMipTail`] when the image has none.
    pub fn mip_tail_size_b(&self) -> Result<u64> {
        let offset = self.mip_tail_offset_b()?;
        overflow(self.array_stride_b.checked_sub(offset))
    }
}
