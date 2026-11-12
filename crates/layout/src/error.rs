//! The error type shared by every part of the layout code.
//!
//! Every input this crate takes (an image description, a tiling, a copy region, the buffers of a
//! copy) may come from guest memory, so each condition Mesa's `nil` asserts or leaves unchecked
//! is one of these variants instead of a panic.

use crate::tiling::GobType;

/// Result alias for this crate.
pub type Result<T, E = LayoutError> = core::result::Result<T, E>;

/// Why a layout could not be computed or a copy could not be made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LayoutError {
    /// A size, offset or extent does not fit the integer type that holds it.
    #[error("a layout size or offset overflows")]
    Overflow,
    /// An image extent has a zero dimension.
    #[error("an image extent has a zero dimension")]
    ZeroExtent,
    /// A format's element size or block dimension is zero.
    #[error("a format's element size and block dimensions must be non-zero")]
    InvalidFormat,
    /// A sample count that no sample layout has.
    #[error("{0} samples per pixel is not a supported sample count")]
    InvalidSampleCount(u32),
    /// An image's level count is zero or more than [`crate::MAX_LEVELS`].
    #[error("{levels} levels is outside the supported range of 1 to {max}")]
    InvalidLevelCount {
        /// The level count asked for.
        levels: u32,
        /// The largest level count, [`crate::MAX_LEVELS`].
        max: u32,
    },
    /// A 1D image with a height, depth or sample count other than 1.
    #[error("a 1D image must have height 1, depth 1 and one sample")]
    Invalid1d,
    /// A 2D image with a depth other than 1.
    #[error("a 2D image must have depth 1")]
    Invalid2d,
    /// A 3D image with more than one layer or more than one sample.
    #[error("a 3D image must have one layer and one sample")]
    Invalid3d,
    /// A multisampled image with more than one level.
    #[error("a multisampled image must have exactly one level")]
    MultisampleMipmaps,
    /// A pitch-linear image that is not 2D with one level, one layer and one sample.
    #[error("a pitch-linear image must be 2D with one level, one layer and one sample")]
    InvalidLinear,
    /// A pitch that is not a multiple of [`crate::LINEAR_PITCH_ALIGN_B`].
    #[error("pitch {pitch} is not a multiple of {align} bytes")]
    PitchNotAligned {
        /// The pitch asked for, in bytes.
        pitch: u32,
        /// The required alignment, in bytes.
        align: u32,
    },
    /// A pitch smaller than one row of the image.
    #[error("pitch {pitch} is smaller than a row of {row} bytes")]
    PitchTooSmall {
        /// The pitch asked for, in bytes.
        pitch: u32,
        /// The bytes in one row of the image.
        row: u32,
    },
    /// A pitch given for a block-linear image.
    #[error("a pitch can only be given for a pitch-linear image")]
    PitchOnBlockLinear,
    /// A linear tiling with a non-zero block dimension.
    #[error("a linear tiling cannot have a block size")]
    LinearWithBlock,
    /// A tiling whose block does not fit in 32 bits.
    #[error("a block of 2^{x_log2} by 2^{y_log2} by 2^{z_log2} GOBs is too large")]
    BlockTooLarge {
        /// log2 of the block width in GOBs.
        x_log2: u8,
        /// log2 of the block height in GOBs.
        y_log2: u8,
        /// log2 of the block depth in GOBs.
        z_log2: u8,
    },
    /// A GOB kind whose byte order inside the GOB no reference gives.
    #[error("the {} GOB kind has no known byte order", .0.name())]
    UnsupportedGob(GobType),
    /// A byte position outside one GOB.
    #[error("byte ({x}, {y}) is outside a GOB")]
    OutsideGob {
        /// The x position, in bytes.
        x: u32,
        /// The y position, in rows.
        y: u32,
    },
    /// A level index at or past the image's level count.
    #[error("level {level} is out of range for an image with {levels} levels")]
    LevelOutOfRange {
        /// The level asked for.
        level: u32,
        /// The image's level count.
        levels: u32,
    },
    /// A layer index at or past the image's layer count.
    #[error("layer {layer} is out of range for an image with {layers} layers")]
    LayerOutOfRange {
        /// The layer asked for.
        layer: u32,
        /// The image's layer count.
        layers: u32,
    },
    /// A depth slice at or past the level's depth.
    #[error("slice {z} is out of range for a level of depth {depth}")]
    SliceOutOfRange {
        /// The slice asked for.
        z: u32,
        /// The level's depth.
        depth: u32,
    },
    /// The image has no mip tail.
    #[error("the image has no mip tail")]
    NoMipTail,
    /// A copy region or offset with a layer other than layer 0 of one layer.
    #[error("a copy region must be one layer starting at layer 0")]
    RegionHasLayers,
    /// A copy region or offset that reaches past the level.
    #[error("the region does not fit in the level")]
    RegionOutOfBounds,
    /// The tiled buffer is smaller than the level it holds.
    #[error("the tiled buffer holds {len} bytes, the level needs {needed}")]
    TiledTooSmall {
        /// Bytes the level needs.
        needed: u64,
        /// Bytes the buffer holds.
        len: usize,
    },
    /// The linear buffer is smaller than the region.
    #[error("the linear buffer holds {len} bytes, the region needs {needed}")]
    LinearTooSmall {
        /// Bytes the region needs.
        needed: u64,
        /// Bytes the buffer holds.
        len: usize,
    },
}
