//! `PixelFormat`: the host-neutral list of pixel formats the guest can name, with block sizes and
//! properties.
//!
//! The list and its order are Eden's `PIXEL_FORMAT_LIST` (`eden/src/video_core/surface.h`
//! @ cf654052, GPL-3.0-or-later): 112 entries (105 colour, 3 depth, 1 stencil, 3 depth-stencil),
//! each with its block width, block height and bits per block as Eden states them. The
//! properties (aspect, numeric type, sRGB encoding, compression family) follow Eden's predicates in
//! `eden/src/video_core/surface.cpp` @ cf654052 (`GetFormatType`, `IsPixelFormatSRGB`,
//! `IsPixelFormatInteger`, `IsPixelFormatSignedInteger`, `IsPixelFormatASTC`, `IsPixelFormatBCn`,
//! `IsPixelFormatETC2`); the tests pin each of those sets.
//!
//! Every entry is cross-checked in the tests against Mesa's NVIDIA format table
//! (`mesa-nouveau/src/nouveau/nil/nil_formats.csv` @ 52238762, MIT, Copyright 2024 Collabora
//! Ltd., copied into this folder as the test's data with only its line endings changed to LF): block size, bits per block, numeric
//! type, sRGB and depth/stencil aspect. One entry has no Mesa row (`G4R4_UNORM`; the test lists
//! it, with the reason).
//!
//! Decoding the guest's texture header fields into a `PixelFormat` is the GPU crate's job, not
//! this one's (docs/specs/render-backend.md, "Formats").
//!
//! Note: the render spec says "Eden's 116 entries"; `PIXEL_FORMAT_LIST` at the pinned commit has
//! 112 (`format::tests::list_matches_eden_count`).

use serde::{Deserialize, Serialize};

/// Which parts of a pixel a format stores (Eden's `SurfaceType`, `surface.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FormatAspect {
    /// Colour channels.
    Color,
    /// Depth only.
    Depth,
    /// Stencil only.
    Stencil,
    /// Depth and stencil.
    DepthStencil,
}

/// The numeric type of a format's channels. For depth-stencil formats it is the depth channel's
/// type (the stencil channel is always [`NumericType::Uint`]); for stencil-only formats it is
/// [`NumericType::Uint`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericType {
    /// Unsigned normalised to 0..1.
    Unorm,
    /// Signed normalised to -1..1.
    Snorm,
    /// Unsigned integer.
    Uint,
    /// Signed integer.
    Sint,
    /// Floating point (including the unsigned packed floats and BC6H).
    Float,
}

/// How a colour format's values are encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorEncoding {
    /// Linear values.
    Linear,
    /// sRGB-encoded colour channels (alpha stays linear).
    Srgb,
}

/// The block-compression family of a format, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Compression {
    /// One pixel per block.
    Uncompressed,
    /// BC1 to BC7 (Eden's `IsPixelFormatBCn`).
    Bc,
    /// ASTC (Eden's `IsPixelFormatASTC`).
    Astc,
    /// ETC2 and EAC (Eden's `IsPixelFormatETC2`, which includes the EAC formats).
    Etc2,
}

/// Everything this crate knows about one [`PixelFormat`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FormatInfo {
    /// Texels per block, horizontally.
    pub block_width: u32,
    /// Texels per block, vertically.
    pub block_height: u32,
    /// Bits per block.
    pub bits_per_block: u32,
    /// What the format stores.
    pub aspect: FormatAspect,
    /// The channels' numeric type.
    pub numeric: NumericType,
    /// Linear or sRGB.
    pub encoding: ColorEncoding,
    /// Block-compression family.
    pub compression: Compression,
}

/// Defines [`PixelFormat`], its list and its table from one set of rows, in Eden's order, the way
/// Eden's `PIXEL_FORMAT_LIST` X-macro does.
macro_rules! pixel_formats {
    ($( $variant:ident $eden:ident $bw:literal $bh:literal $bits:literal
        $aspect:ident $numeric:ident $encoding:ident $compression:ident ),* $(,)?) => {
        /// A pixel format the guest can name (Eden's `PixelFormat`, `surface.h`). Each variant's
        /// documentation gives Eden's name, which [`PixelFormat::name`] also returns.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub enum PixelFormat {
            $(
                #[doc = concat!("Eden's `", stringify!($eden), "`.")]
                $variant,
            )*
        }

        impl PixelFormat {
            /// Every format, in Eden's order.
            pub const ALL: [Self; PIXEL_FORMAT_COUNT] = [$(Self::$variant),*];

            /// Eden's name for the format, which is also how errors print it.
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($eden),)*
                }
            }

            /// The format's block size and properties.
            pub const fn info(self) -> FormatInfo {
                match self {
                    $(Self::$variant => FormatInfo {
                        block_width: $bw,
                        block_height: $bh,
                        bits_per_block: $bits,
                        aspect: FormatAspect::$aspect,
                        numeric: NumericType::$numeric,
                        encoding: ColorEncoding::$encoding,
                        compression: Compression::$compression,
                    },)*
                }
            }
        }
    };
}

/// Number of [`PixelFormat`]s (the length of Eden's `PIXEL_FORMAT_LIST` at the pinned commit).
pub const PIXEL_FORMAT_COUNT: usize = 112;

pixel_formats! {
    A8B8G8R8Unorm A8B8G8R8_UNORM 1 1 32 Color Unorm Linear Uncompressed,
    A8B8G8R8Snorm A8B8G8R8_SNORM 1 1 32 Color Snorm Linear Uncompressed,
    A8B8G8R8Sint A8B8G8R8_SINT 1 1 32 Color Sint Linear Uncompressed,
    A8B8G8R8Uint A8B8G8R8_UINT 1 1 32 Color Uint Linear Uncompressed,
    R5G6B5Unorm R5G6B5_UNORM 1 1 16 Color Unorm Linear Uncompressed,
    B5G6R5Unorm B5G6R5_UNORM 1 1 16 Color Unorm Linear Uncompressed,
    A1R5G5B5Unorm A1R5G5B5_UNORM 1 1 16 Color Unorm Linear Uncompressed,
    A2B10G10R10Unorm A2B10G10R10_UNORM 1 1 32 Color Unorm Linear Uncompressed,
    A2B10G10R10Uint A2B10G10R10_UINT 1 1 32 Color Uint Linear Uncompressed,
    A2R10G10B10Unorm A2R10G10B10_UNORM 1 1 32 Color Unorm Linear Uncompressed,
    A1B5G5R5Unorm A1B5G5R5_UNORM 1 1 16 Color Unorm Linear Uncompressed,
    A5B5G5R1Unorm A5B5G5R1_UNORM 1 1 16 Color Unorm Linear Uncompressed,
    R8Unorm R8_UNORM 1 1 8 Color Unorm Linear Uncompressed,
    R8Snorm R8_SNORM 1 1 8 Color Snorm Linear Uncompressed,
    R8Sint R8_SINT 1 1 8 Color Sint Linear Uncompressed,
    R8Uint R8_UINT 1 1 8 Color Uint Linear Uncompressed,
    R16G16B16A16Float R16G16B16A16_FLOAT 1 1 64 Color Float Linear Uncompressed,
    R16G16B16A16Unorm R16G16B16A16_UNORM 1 1 64 Color Unorm Linear Uncompressed,
    R16G16B16A16Snorm R16G16B16A16_SNORM 1 1 64 Color Snorm Linear Uncompressed,
    R16G16B16A16Sint R16G16B16A16_SINT 1 1 64 Color Sint Linear Uncompressed,
    R16G16B16A16Uint R16G16B16A16_UINT 1 1 64 Color Uint Linear Uncompressed,
    B10G11R11Float B10G11R11_FLOAT 1 1 32 Color Float Linear Uncompressed,
    R32G32B32A32Uint R32G32B32A32_UINT 1 1 128 Color Uint Linear Uncompressed,
    Bc1RgbaUnorm BC1_RGBA_UNORM 4 4 64 Color Unorm Linear Bc,
    Bc2Unorm BC2_UNORM 4 4 128 Color Unorm Linear Bc,
    Bc3Unorm BC3_UNORM 4 4 128 Color Unorm Linear Bc,
    Bc4Unorm BC4_UNORM 4 4 64 Color Unorm Linear Bc,
    Bc4Snorm BC4_SNORM 4 4 64 Color Snorm Linear Bc,
    Bc5Unorm BC5_UNORM 4 4 128 Color Unorm Linear Bc,
    Bc5Snorm BC5_SNORM 4 4 128 Color Snorm Linear Bc,
    Bc7Unorm BC7_UNORM 4 4 128 Color Unorm Linear Bc,
    Bc6hUfloat BC6H_UFLOAT 4 4 128 Color Float Linear Bc,
    Bc6hSfloat BC6H_SFLOAT 4 4 128 Color Float Linear Bc,
    Astc2d4x4Unorm ASTC_2D_4X4_UNORM 4 4 128 Color Unorm Linear Astc,
    B8G8R8A8Unorm B8G8R8A8_UNORM 1 1 32 Color Unorm Linear Uncompressed,
    R32G32B32A32Float R32G32B32A32_FLOAT 1 1 128 Color Float Linear Uncompressed,
    R32G32B32A32Sint R32G32B32A32_SINT 1 1 128 Color Sint Linear Uncompressed,
    R32G32Float R32G32_FLOAT 1 1 64 Color Float Linear Uncompressed,
    R32G32Sint R32G32_SINT 1 1 64 Color Sint Linear Uncompressed,
    R32Float R32_FLOAT 1 1 32 Color Float Linear Uncompressed,
    R16Float R16_FLOAT 1 1 16 Color Float Linear Uncompressed,
    R16Unorm R16_UNORM 1 1 16 Color Unorm Linear Uncompressed,
    R16Snorm R16_SNORM 1 1 16 Color Snorm Linear Uncompressed,
    R16Uint R16_UINT 1 1 16 Color Uint Linear Uncompressed,
    R16Sint R16_SINT 1 1 16 Color Sint Linear Uncompressed,
    R16G16Unorm R16G16_UNORM 1 1 32 Color Unorm Linear Uncompressed,
    R16G16Float R16G16_FLOAT 1 1 32 Color Float Linear Uncompressed,
    R16G16Uint R16G16_UINT 1 1 32 Color Uint Linear Uncompressed,
    R16G16Sint R16G16_SINT 1 1 32 Color Sint Linear Uncompressed,
    R16G16Snorm R16G16_SNORM 1 1 32 Color Snorm Linear Uncompressed,
    R32G32B32Float R32G32B32_FLOAT 1 1 96 Color Float Linear Uncompressed,
    A8B8G8R8Srgb A8B8G8R8_SRGB 1 1 32 Color Unorm Srgb Uncompressed,
    R8G8Unorm R8G8_UNORM 1 1 16 Color Unorm Linear Uncompressed,
    R8G8Snorm R8G8_SNORM 1 1 16 Color Snorm Linear Uncompressed,
    R8G8Sint R8G8_SINT 1 1 16 Color Sint Linear Uncompressed,
    R8G8Uint R8G8_UINT 1 1 16 Color Uint Linear Uncompressed,
    R32G32Uint R32G32_UINT 1 1 64 Color Uint Linear Uncompressed,
    R16G16B16X16Float R16G16B16X16_FLOAT 1 1 64 Color Float Linear Uncompressed,
    R32Uint R32_UINT 1 1 32 Color Uint Linear Uncompressed,
    R32Sint R32_SINT 1 1 32 Color Sint Linear Uncompressed,
    Astc2d8x8Unorm ASTC_2D_8X8_UNORM 8 8 128 Color Unorm Linear Astc,
    Astc2d8x5Unorm ASTC_2D_8X5_UNORM 8 5 128 Color Unorm Linear Astc,
    Astc2d5x4Unorm ASTC_2D_5X4_UNORM 5 4 128 Color Unorm Linear Astc,
    B8G8R8A8Srgb B8G8R8A8_SRGB 1 1 32 Color Unorm Srgb Uncompressed,
    Bc1RgbaSrgb BC1_RGBA_SRGB 4 4 64 Color Unorm Srgb Bc,
    Bc2Srgb BC2_SRGB 4 4 128 Color Unorm Srgb Bc,
    Bc3Srgb BC3_SRGB 4 4 128 Color Unorm Srgb Bc,
    Bc7Srgb BC7_SRGB 4 4 128 Color Unorm Srgb Bc,
    A4B4G4R4Unorm A4B4G4R4_UNORM 1 1 16 Color Unorm Linear Uncompressed,
    G4R4Unorm G4R4_UNORM 1 1 8 Color Unorm Linear Uncompressed,
    Astc2d4x4Srgb ASTC_2D_4X4_SRGB 4 4 128 Color Unorm Srgb Astc,
    Astc2d8x8Srgb ASTC_2D_8X8_SRGB 8 8 128 Color Unorm Srgb Astc,
    Astc2d8x5Srgb ASTC_2D_8X5_SRGB 8 5 128 Color Unorm Srgb Astc,
    Astc2d5x4Srgb ASTC_2D_5X4_SRGB 5 4 128 Color Unorm Srgb Astc,
    Astc2d5x5Unorm ASTC_2D_5X5_UNORM 5 5 128 Color Unorm Linear Astc,
    Astc2d5x5Srgb ASTC_2D_5X5_SRGB 5 5 128 Color Unorm Srgb Astc,
    Astc2d10x8Unorm ASTC_2D_10X8_UNORM 10 8 128 Color Unorm Linear Astc,
    Astc2d10x8Srgb ASTC_2D_10X8_SRGB 10 8 128 Color Unorm Srgb Astc,
    Astc2d6x6Unorm ASTC_2D_6X6_UNORM 6 6 128 Color Unorm Linear Astc,
    Astc2d6x6Srgb ASTC_2D_6X6_SRGB 6 6 128 Color Unorm Srgb Astc,
    Astc2d10x6Unorm ASTC_2D_10X6_UNORM 10 6 128 Color Unorm Linear Astc,
    Astc2d10x6Srgb ASTC_2D_10X6_SRGB 10 6 128 Color Unorm Srgb Astc,
    Astc2d10x5Unorm ASTC_2D_10X5_UNORM 10 5 128 Color Unorm Linear Astc,
    Astc2d10x5Srgb ASTC_2D_10X5_SRGB 10 5 128 Color Unorm Srgb Astc,
    Astc2d10x10Unorm ASTC_2D_10X10_UNORM 10 10 128 Color Unorm Linear Astc,
    Astc2d10x10Srgb ASTC_2D_10X10_SRGB 10 10 128 Color Unorm Srgb Astc,
    Astc2d12x10Unorm ASTC_2D_12X10_UNORM 12 10 128 Color Unorm Linear Astc,
    Astc2d12x10Srgb ASTC_2D_12X10_SRGB 12 10 128 Color Unorm Srgb Astc,
    Astc2d12x12Unorm ASTC_2D_12X12_UNORM 12 12 128 Color Unorm Linear Astc,
    Astc2d12x12Srgb ASTC_2D_12X12_SRGB 12 12 128 Color Unorm Srgb Astc,
    Astc2d8x6Unorm ASTC_2D_8X6_UNORM 8 6 128 Color Unorm Linear Astc,
    Astc2d8x6Srgb ASTC_2D_8X6_SRGB 8 6 128 Color Unorm Srgb Astc,
    Astc2d6x5Unorm ASTC_2D_6X5_UNORM 6 5 128 Color Unorm Linear Astc,
    Astc2d6x5Srgb ASTC_2D_6X5_SRGB 6 5 128 Color Unorm Srgb Astc,
    E5B9G9R9Float E5B9G9R9_FLOAT 1 1 32 Color Float Linear Uncompressed,
    Etc2RgbUnorm ETC2_RGB_UNORM 4 4 64 Color Unorm Linear Etc2,
    Etc2RgbaUnorm ETC2_RGBA_UNORM 4 4 128 Color Unorm Linear Etc2,
    Etc2RgbPtaUnorm ETC2_RGB_PTA_UNORM 4 4 64 Color Unorm Linear Etc2,
    Etc2RgbSrgb ETC2_RGB_SRGB 4 4 64 Color Unorm Srgb Etc2,
    Etc2RgbaSrgb ETC2_RGBA_SRGB 4 4 128 Color Unorm Srgb Etc2,
    Etc2RgbPtaSrgb ETC2_RGB_PTA_SRGB 4 4 64 Color Unorm Srgb Etc2,
    EacR11Unorm EAC_R11_UNORM 4 4 64 Color Unorm Linear Etc2,
    EacR11Snorm EAC_R11_SNORM 4 4 64 Color Snorm Linear Etc2,
    EacR11G11Unorm EAC_R11G11_UNORM 4 4 128 Color Unorm Linear Etc2,
    EacR11G11Snorm EAC_R11G11_SNORM 4 4 128 Color Snorm Linear Etc2,
    D32Float D32_FLOAT 1 1 32 Depth Float Linear Uncompressed,
    D16Unorm D16_UNORM 1 1 16 Depth Unorm Linear Uncompressed,
    X8D24Unorm X8_D24_UNORM 1 1 32 Depth Unorm Linear Uncompressed,
    S8Uint S8_UINT 1 1 8 Stencil Uint Linear Uncompressed,
    D24UnormS8Uint D24_UNORM_S8_UINT 1 1 32 DepthStencil Unorm Linear Uncompressed,
    S8UintD24Unorm S8_UINT_D24_UNORM 1 1 32 DepthStencil Unorm Linear Uncompressed,
    D32FloatS8Uint D32_FLOAT_S8_UINT 1 1 64 DepthStencil Float Linear Uncompressed,
}

impl PixelFormat {
    /// Texels per block, horizontally (Eden's `DefaultBlockWidth`).
    pub const fn block_width(self) -> u32 {
        self.info().block_width
    }

    /// Texels per block, vertically (Eden's `DefaultBlockHeight`).
    pub const fn block_height(self) -> u32 {
        self.info().block_height
    }

    /// Bits per block (Eden's `BitsPerBlock`).
    pub const fn bits_per_block(self) -> u32 {
        self.info().bits_per_block
    }

    /// Bytes per block (Eden's `BytesPerBlock`: bits divided by 8; every entry is a whole number
    /// of bytes).
    pub const fn bytes_per_block(self) -> u32 {
        self.info().bits_per_block / u8::BITS
    }

    /// What the format stores (Eden's `GetFormatType`).
    pub const fn aspect(self) -> FormatAspect {
        self.info().aspect
    }

    /// The channels' numeric type.
    pub const fn numeric(self) -> NumericType {
        self.info().numeric
    }

    /// Whether the colour channels are sRGB-encoded (Eden's `IsPixelFormatSRGB`).
    pub const fn is_srgb(self) -> bool {
        matches!(self.info().encoding, ColorEncoding::Srgb)
    }

    /// The block-compression family.
    pub const fn compression(self) -> Compression {
        self.info().compression
    }

    /// Whether a block holds more than one texel.
    pub const fn is_compressed(self) -> bool {
        !matches!(self.info().compression, Compression::Uncompressed)
    }

    /// Whether this is an integer colour format (Eden's `IsPixelFormatInteger`).
    pub const fn is_integer(self) -> bool {
        matches!(self.info().aspect, FormatAspect::Color)
            && matches!(self.info().numeric, NumericType::Uint | NumericType::Sint)
    }

    /// Whether this is a signed integer colour format (Eden's `IsPixelFormatSignedInteger`).
    pub const fn is_signed_integer(self) -> bool {
        matches!(self.info().aspect, FormatAspect::Color)
            && matches!(self.info().numeric, NumericType::Sint)
    }

    /// Whether the format has a depth channel.
    pub const fn has_depth(self) -> bool {
        matches!(
            self.info().aspect,
            FormatAspect::Depth | FormatAspect::DepthStencil
        )
    }

    /// Whether the format has a stencil channel.
    pub const fn has_stencil(self) -> bool {
        matches!(
            self.info().aspect,
            FormatAspect::Stencil | FormatAspect::DepthStencil
        )
    }

    /// The format's position in [`PixelFormat::ALL`], for per-format tables.
    pub const fn index(self) -> usize {
        self as usize
    }
}

impl core::fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests;
