//! GOB kinds, tilings, and the byte offset of a position in a block-linear level.
//!
//! Ported from `mesa-nouveau/src/nouveau/nil/tiling.rs` (`GOBType`, `Tiling`) and the offset
//! arithmetic of `nil/copy.rs` (`BlockPointer` and the `for_each_gob_line` tables) @ 52238762
//! (MIT, Copyright © 2024 Collabora, Ltd., and for `copy.rs` Copyright © 2024 Valve Corp. and
//! Collabora, Ltd.; the licence text is in this crate's `lib.rs`).
//!
//! # The three tiers
//!
//! As Mesa's `Tiling` documentation describes: a level is a grid of tiles (NVIDIA's "blocks"),
//! ordered x fastest, then y, then z. A tile is a grid of GOBs ("groups of bytes"), 2^`x_log2` by
//! 2^`y_log2` by 2^`z_log2` of them, ordered the same way. A GOB is 64 bytes by 8 rows
//! ([`GOB_WIDTH_B`], [`GOB_HEIGHT`]), made of 16-byte lines ([`GOB_LINE_WIDTH_B`]) whose order
//! inside the GOB depends on the [`GobType`].
//!
//! # The Switch 2 colour GOB
//!
//! Eden's GOB (`SwizzleOffset` in `eden-nx2/src/video_core/host_shaders/astc_decoder.comp` @
//! 50fa86bb) is [`GobType::TegraColor`]; Eden-NX2's Switch 2 tweak, which swaps byte-offset bits
//! 5 and 6 (`NativeGobOffset` in `eden-nx2/src/video_core/textures/decoders.cpp`; blocker-log
//! row 69), gives exactly [`GobType::TuringColor2d`], the GOB Mesa picks for colour images on
//! Turing and later (`GOBType::choose`). `tests/gob_derivation.rs` checks both bit for bit
//! (`docs/specs/gpu-protocol.md`, "Block-linear layout"). Depth and stencil use
//! [`GobType::FermiZs`] on Turing in Mesa; whether NVN2 does is a **guess** (confirm with a
//! depth target dump), and no reference gives that GOB's byte order, so offsets and copies refuse
//! it.
//!
//! # Differences from Mesa
//!
//! - Only the GOB kinds a Tegra or Turing-class GPU uses are here: `Linear`, `FermiZS`,
//!   `TegraColor` and `TuringColor2D`. Mesa's `FermiColor` (discrete Fermi to Volta) and the
//!   Blackwell kinds are not ported; each is one more line table and variant if ever needed.
//! - The tiling is the caller's: the console's texture header and render-target registers give
//!   the block size, and the memory kind gives the GOB. Mesa's `Tiling::choose` (the proprietary
//!   driver's heuristic) and `Tiling::sparse` are not ported, nor is `GOBType::choose`.
//! - [`Tiling::new`] refuses a linear tiling with a block size ([`LayoutError::LinearWithBlock`];
//!   Mesa only `debug_assert`s it in `is_tiled`) and a block whose byte size does not fit in 32
//!   bits ([`LayoutError::BlockTooLarge`]; Mesa's shifts and `size_B` would overflow).
//! - [`block_linear_offset`] and the copies return [`LayoutError::UnsupportedGob`] for `Linear`
//!   and `FermiZS`, where Mesa's copies panic ("Unsupported GOB type").

use crate::error::{LayoutError, Result};
use crate::extent::units::Bytes;
use crate::extent::{Extent4D, Offset4D, ilog2_ceil};

/// Width of a GOB in bytes (Mesa: `GOBType::extent_B`, 64 by 8 by 1).
pub const GOB_WIDTH_B: u32 = 64;
/// Height of a GOB in rows (Mesa: `GOBType::extent_B`).
pub const GOB_HEIGHT: u32 = 8;
/// Bytes in one GOB.
pub const GOB_SIZE_B: u32 = GOB_WIDTH_B * GOB_HEIGHT;
/// Width of one contiguous line inside a GOB, in bytes: the `LINE_WIDTH_B` of Mesa's
/// `CopyGOBTegra` and `CopyGOBTuring2D`.
pub const GOB_LINE_WIDTH_B: u32 = 16;
/// Bytes in one sector of a GOB: two lines, one above the other (Mesa's `GOBType` docs, "Each
/// Sector is 32 Bytes").
pub const GOB_SECTOR_B: u32 = 2 * GOB_LINE_WIDTH_B;

/// Lines in one row of a GOB.
const LINES_PER_ROW: usize = (GOB_WIDTH_B / GOB_LINE_WIDTH_B) as usize;
/// Lines in one half of a GOB: Mesa's tables list the left 32 bytes and repeat them, shifted by
/// 32 bytes in x and 256 bytes in the GOB, for the right half (`for i in 0..2`).
const LINES_PER_HALF: usize = LINES_PER_ROW / 2 * GOB_HEIGHT as usize;
/// The x shift from the left half of a GOB to the right half.
const HALF_GOB_WIDTH_B: u32 = GOB_WIDTH_B / 2;
/// The byte shift from the left half of a GOB to the right half.
const HALF_GOB_SIZE_B: u32 = GOB_SIZE_B / 2;

/// One line of a GOB's left half: (byte offset in the GOB, x in bytes, y).
type HalfLine = (u32, u32, u32);

/// `CopyGOBTegra::for_each_gob_line` (`nil/copy.rs`), the `i = 0` half.
const TEGRA_HALF: [HalfLine; LINES_PER_HALF] = [
    (0x00, 0, 0),
    (0x10, 0, 1),
    (0x20, 16, 0),
    (0x30, 16, 1),
    (0x40, 0, 2),
    (0x50, 0, 3),
    (0x60, 16, 2),
    (0x70, 16, 3),
    (0x80, 0, 4),
    (0x90, 0, 5),
    (0xa0, 16, 4),
    (0xb0, 16, 5),
    (0xc0, 0, 6),
    (0xd0, 0, 7),
    (0xe0, 16, 6),
    (0xf0, 16, 7),
];

/// `CopyGOBTuring2D::for_each_gob_line` (`nil/copy.rs`), the `i = 0` half.
const TURING_2D_HALF: [HalfLine; LINES_PER_HALF] = [
    (0x00, 0, 0),
    (0x10, 0, 1),
    (0x20, 0, 2),
    (0x30, 0, 3),
    (0x40, 16, 0),
    (0x50, 16, 1),
    (0x60, 16, 2),
    (0x70, 16, 3),
    (0x80, 0, 4),
    (0x90, 0, 5),
    (0xa0, 0, 6),
    (0xb0, 0, 7),
    (0xc0, 16, 4),
    (0xd0, 16, 5),
    (0xe0, 16, 6),
    (0xf0, 16, 7),
];

/// Line offsets by position: `[y][x / GOB_LINE_WIDTH_B]` is the byte offset in the GOB of the
/// line starting at that position.
type LineOffsets = [[u32; LINES_PER_ROW]; GOB_HEIGHT as usize];

/// Turns a half-GOB line list into offsets by position, adding the right half as Mesa's
/// `for i in 0..2` loop does. An entry outside the GOB fails the build (const evaluation).
const fn line_offsets(half: &[HalfLine; LINES_PER_HALF]) -> LineOffsets {
    let mut table = [[0; LINES_PER_ROW]; GOB_HEIGHT as usize];
    let mut i = 0;
    while i < 2 {
        let mut j = 0;
        while j < LINES_PER_HALF {
            let (offset, x, y) = half[j];
            let x = x + i * HALF_GOB_WIDTH_B;
            table[y as usize][(x / GOB_LINE_WIDTH_B) as usize] = offset + i * HALF_GOB_SIZE_B;
            j += 1;
        }
        i += 1;
    }
    table
}

const TEGRA_LINES: LineOffsets = line_offsets(&TEGRA_HALF);
const TURING_2D_LINES: LineOffsets = line_offsets(&TURING_2D_HALF);

/// How bytes are arranged inside one GOB (Mesa: `GOBType`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GobType {
    /// Not tiled: the image is pitch-linear, and a "GOB" is one byte.
    #[default]
    Linear,
    /// Mesa's GOB for every depth and stencil surface. Its byte order is not in any reference.
    FermiZs,
    /// The Tegra colour GOB (Switch 1, and Eden's only GOB). Sectors of 32 bytes, numbered in
    /// a 4 by 4 grid, each sector 16 bytes wide and 2 rows tall (Mesa's table):
    ///
    /// | | | | |
    /// |---|---|---|---|
    /// | 0 | 1 | 8 | 9 |
    /// | 2 | 3 | 10 | 11 |
    /// | 4 | 5 | 12 | 13 |
    /// | 6 | 7 | 14 | 15 |
    TegraColor,
    /// The Turing colour GOB, which the Switch 2 uses for colour (see the module docs):
    ///
    /// | | | | |
    /// |---|---|---|---|
    /// | 0 | 2 | 8 | 10 |
    /// | 1 | 3 | 9 | 11 |
    /// | 4 | 6 | 12 | 14 |
    /// | 5 | 7 | 13 | 15 |
    TuringColor2d,
}

impl GobType {
    /// The GOB's extent in bytes: 1 by 1 by 1 for [`GobType::Linear`], 64 by 8 by 1 for the
    /// others (Mesa: `GOBType::extent_B`).
    pub fn extent_b(self) -> Extent4D<Bytes> {
        match self {
            GobType::Linear => Extent4D::new(1, 1, 1, 1),
            GobType::FermiZs | GobType::TegraColor | GobType::TuringColor2d => {
                Extent4D::new(GOB_WIDTH_B, GOB_HEIGHT, 1, 1)
            }
        }
    }

    /// A plain name for messages.
    pub fn name(self) -> &'static str {
        match self {
            GobType::Linear => "linear",
            GobType::FermiZs => "Fermi depth-stencil",
            GobType::TegraColor => "Tegra colour",
            GobType::TuringColor2d => "Turing colour 2D",
        }
    }

    pub(crate) fn lines(self) -> Result<&'static LineOffsets> {
        match self {
            GobType::TegraColor => Ok(&TEGRA_LINES),
            GobType::TuringColor2d => Ok(&TURING_2D_LINES),
            GobType::Linear | GobType::FermiZs => Err(LayoutError::UnsupportedGob(self)),
        }
    }

    /// The byte offset inside one GOB of the byte at (`x`, `y`), `x` in bytes.
    ///
    /// [`LayoutError::UnsupportedGob`] for a kind with no known byte order (checked first), then
    /// [`LayoutError::OutsideGob`] for `x` of [`GOB_WIDTH_B`] or more or `y` of [`GOB_HEIGHT`] or
    /// more.
    pub fn byte_offset(self, x: u32, y: u32) -> Result<u32> {
        let lines = self.lines()?;
        let outside = LayoutError::OutsideGob { x, y };
        if x >= GOB_WIDTH_B || y >= GOB_HEIGHT {
            return Err(outside);
        }
        let row = usize::try_from(y).map_err(|_| outside)?;
        let column = usize::try_from(x / GOB_LINE_WIDTH_B).map_err(|_| outside)?;
        let line = lines
            .get(row)
            .and_then(|row| row.get(column))
            .ok_or(outside)?;
        Ok(line + x % GOB_LINE_WIDTH_B)
    }
}

/// Shifts `value` left by `shift`, or `None` if a set bit would be lost.
fn shl_exact(value: u32, shift: u8) -> Option<u32> {
    let shifted = value.checked_shl(u32::from(shift))?;
    (shifted >> shift == value).then_some(shifted)
}

/// A level's tiling: the GOB kind and the tile ("block") size in GOBs (Mesa: `Tiling`).
///
/// Built only by [`Tiling::new`] and [`Tiling::clamp`], so the tile's byte size always fits in
/// 32 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tiling {
    gob_type: GobType,
    x_log2: u8,
    y_log2: u8,
    z_log2: u8,
    extent_b: Extent4D<Bytes>,
}

impl Default for Tiling {
    fn default() -> Self {
        Self::LINEAR
    }
}

impl Tiling {
    /// The tiling of a pitch-linear image (Mesa: `Tiling::default()`).
    pub const LINEAR: Tiling = Tiling {
        gob_type: GobType::Linear,
        x_log2: 0,
        y_log2: 0,
        z_log2: 0,
        extent_b: Extent4D::new(1, 1, 1, 1),
    };

    /// A tiling of `gob_type` GOBs in tiles of 2^`x_log2` by 2^`y_log2` by 2^`z_log2` GOBs.
    ///
    /// [`LayoutError::LinearWithBlock`] for a linear tiling with any non-zero size;
    /// [`LayoutError::BlockTooLarge`] when the tile's extent or byte size does not fit in 32
    /// bits.
    pub fn new(gob_type: GobType, x_log2: u8, y_log2: u8, z_log2: u8) -> Result<Self> {
        if gob_type == GobType::Linear {
            if (x_log2, y_log2, z_log2) != (0, 0, 0) {
                return Err(LayoutError::LinearWithBlock);
            }
            return Ok(Self::LINEAR);
        }
        let too_large = LayoutError::BlockTooLarge {
            x_log2,
            y_log2,
            z_log2,
        };
        let gob = gob_type.extent_b();
        let width = shl_exact(gob.width, x_log2).ok_or(too_large)?;
        let height = shl_exact(gob.height, y_log2).ok_or(too_large)?;
        let depth = shl_exact(gob.depth, z_log2).ok_or(too_large)?;
        width
            .checked_mul(height)
            .and_then(|area| area.checked_mul(depth))
            .ok_or(too_large)?;
        Ok(Self {
            gob_type,
            x_log2,
            y_log2,
            z_log2,
            extent_b: Extent4D::new(width, height, depth, 1),
        })
    }

    /// The GOB kind.
    pub fn gob_type(&self) -> GobType {
        self.gob_type
    }

    /// log2 of the tile width in GOBs.
    pub fn x_log2(&self) -> u8 {
        self.x_log2
    }

    /// log2 of the tile height in GOBs (the texture header's "block height").
    pub fn y_log2(&self) -> u8 {
        self.y_log2
    }

    /// log2 of the tile depth in GOBs (the texture header's "block depth").
    pub fn z_log2(&self) -> u8 {
        self.z_log2
    }

    /// Whether the tiling is block-linear (Mesa: `is_tiled`).
    pub fn is_tiled(&self) -> bool {
        self.gob_type != GobType::Linear
    }

    /// One tile's extent in bytes (Mesa: `Tiling::extent_B`).
    pub fn extent_b(&self) -> Extent4D<Bytes> {
        self.extent_b
    }

    /// Bytes in one tile (Mesa: `Tiling::size_B`).
    pub fn size_b(&self) -> u32 {
        // `new` checked this product, and `clamp` only makes the tile smaller.
        self.extent_b.width * self.extent_b.height * self.extent_b.depth
    }

    /// The tiling the hardware uses for a level of `extent_b` bytes (Mesa: `Tiling::clamp`,
    /// "done by the hardware at each LOD"): the tile width drops to one GOB when the level is
    /// smaller than one tile in any dimension, and the height and depth shrink to the smallest
    /// power of two of GOBs that covers the level. A linear tiling is returned unchanged.
    pub fn clamp(&self, extent_b: Extent4D<Bytes>) -> Result<Self> {
        if !self.is_tiled() {
            return Ok(*self);
        }
        let tile = self.extent_b();
        let x_log2 = if extent_b.width < tile.width
            || extent_b.height < tile.height
            || extent_b.depth < tile.depth
        {
            0
        } else {
            self.x_log2
        };
        let extent_gob = extent_b.to_gob(self.gob_type)?;
        // `ilog2_ceil` of a `u32` is at most 32, so these fit in a `u8`.
        let ceil_h =
            u8::try_from(ilog2_ceil(extent_gob.height)).map_err(|_| LayoutError::Overflow)?;
        let ceil_d =
            u8::try_from(ilog2_ceil(extent_gob.depth)).map_err(|_| LayoutError::Overflow)?;
        Self::new(
            self.gob_type,
            x_log2,
            self.y_log2.min(ceil_h),
            self.z_log2.min(ceil_d),
        )
    }
}

/// The byte offset, from the start of a level, of byte `offset` of that level.
///
/// `level_extent_b` is the level's extent in bytes (unaligned; its layer count is not used, as
/// a level's layers are separate, see [`crate::Image::level_layer_offset_b`]) and `tiling` the
/// level's own tiling ([`crate::ImageLevel::tiling`]). The arithmetic is Mesa's `BlockPointer`
/// (`nil/copy.rs`), applied twice, tiles in the level and GOBs in the tile, plus the GOB's line
/// table.
///
/// Errors, in the order checked: [`LayoutError::RegionHasLayers`] when `offset.a` is not 0,
/// [`LayoutError::RegionOutOfBounds`] when the position is outside the level,
/// [`LayoutError::UnsupportedGob`] for a GOB with no known byte order, and
/// [`LayoutError::Overflow`] when the aligned level is larger than 64 bits can count.
pub fn block_linear_offset(
    tiling: &Tiling,
    level_extent_b: Extent4D<Bytes>,
    offset: Offset4D<Bytes>,
) -> Result<u64> {
    if offset.a != 0 {
        return Err(LayoutError::RegionHasLayers);
    }
    if offset.x >= level_extent_b.width
        || offset.y >= level_extent_b.height
        || offset.z >= level_extent_b.depth
    {
        return Err(LayoutError::RegionOutOfBounds);
    }
    tiling.gob_type.lines()?;
    let level = Strides::new(tiling, level_extent_b)?;
    level.offset(tiling, offset)
}

/// The multipliers of Mesa's `BlockPointer` for one level, in `u64`.
pub(crate) struct Strides {
    /// Tile height times depth: bytes per byte of x at tile granularity.
    x_mul: u64,
    /// Aligned level width times tile depth.
    y_mul: u64,
    /// Aligned level width times aligned level height.
    z_mul: u64,
    /// Bytes in the whole aligned level.
    level_size: u64,
}

impl Strides {
    /// The multipliers for a level of `level_extent_b` bytes tiled by `tiling`.
    pub(crate) fn new(tiling: &Tiling, level_extent_b: Extent4D<Bytes>) -> Result<Self> {
        let tile = tiling.extent_b();
        let mut level = level_extent_b;
        level.array_len = 1;
        let level = level.align(&tile)?;
        let width = u64::from(level.width);
        let height = u64::from(level.height);
        let x_mul = u64::from(tile.height) * u64::from(tile.depth);
        let y_mul = width * u64::from(tile.depth);
        let z_mul = width * height;
        let level_size = z_mul
            .checked_mul(u64::from(level.depth))
            .ok_or(LayoutError::Overflow)?;
        Ok(Self {
            x_mul,
            y_mul,
            z_mul,
            level_size,
        })
    }

    /// Bytes in the whole aligned level.
    pub(crate) fn level_size(&self) -> u64 {
        self.level_size
    }

    /// The offset of `offset`, which the caller has checked is inside the level and in a
    /// supported GOB.
    ///
    /// Every term is a part of a position inside the aligned level times a stride of that
    /// level, and their sum is less than [`Strides::level_size`], which `new` computed with
    /// checked arithmetic; so the unchecked `u64` sums and products here cannot overflow.
    pub(crate) fn offset(&self, tiling: &Tiling, offset: Offset4D<Bytes>) -> Result<u64> {
        let tile = tiling.extent_b();
        // Tile extents are powers of two, so masking finds the tile's origin.
        let tile_x = offset.x & !(tile.width - 1);
        let tile_y = offset.y & !(tile.height - 1);
        let tile_z = offset.z & !(tile.depth - 1);
        let tile_offset = u64::from(tile_z) * self.z_mul
            + u64::from(tile_y) * self.y_mul
            + u64::from(tile_x) * self.x_mul;

        // GOBs inside the tile: `BlockPointer::new(tile, GOB_EXTENT_B, tile_extent_B)`.
        let in_tile_x = offset.x - tile_x;
        let in_tile_y = offset.y - tile_y;
        let in_tile_z = offset.z - tile_z;
        let gob_x = in_tile_x & !(GOB_WIDTH_B - 1);
        let gob_y = in_tile_y & !(GOB_HEIGHT - 1);
        let gob_offset = u64::from(in_tile_z) * u64::from(tile.width) * u64::from(tile.height)
            + u64::from(gob_y) * u64::from(tile.width)
            + u64::from(gob_x) * u64::from(GOB_HEIGHT);

        let in_gob = tiling
            .gob_type
            .byte_offset(in_tile_x - gob_x, in_tile_y - gob_y)?;
        Ok(tile_offset + gob_offset + u64::from(in_gob))
    }
}
