//! A texture header's format fields to a [`PixelFormat`].
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`.
//!
//! - The table is ported from `PixelFormatFromTextureInfo` in
//!   `eden/src/video_core/texture_cache/format_lookup_table.cpp` @ cf654052 (Copyright 2026 Eden
//!   Emulator Project, GPL-3.0-or-later; Copyright 2019 yuzu Emulator Project,
//!   GPL-2.0-or-later): every `case` of its switch, in its order, one row of [`FORMAT_TABLE`] each (111 rows). Eden's
//!   `TextureFormat` and `ComponentType` values are the header's `COMPONENTS` and `*_DATA_TYPE`
//!   values (`eden/src/video_core/textures/texture.h`), so the rows name the header's constants,
//!   read through [`crate::classes::clc797tex`]. Seven of Eden's names differ from the header's
//!   (`Z32` is `ZF32`, `Z32_X24S8` is `ZF32_X24S8`, `R32B24G8` is `R32_B24G8`, `E5B9G9R9` is
//!   `E5B9G9R9_SHAREDEXP`, `B10G11R11` is `BF10GF11RF11`, `BC6H_S16` is `BC6H_SF16`, `BC6H_U16`
//!   is `BC6H_UF16`); `tests::header::eden_format_values_match_the_header` pins every value the
//!   table names against Eden's literal.
//! - Eden's `Hash(format, component, is_srgb)` form gives all four channels the same type; its
//!   five-argument form names each. The rows spell all four out.
//!
//! # Divergence
//!
//! Eden logs an unknown combination and returns `A8B8G8R8_UNORM`. Here an unknown combination is
//! `None`, which the header decode reports as [`super::HeaderError::UnknownFormat`], so a texture
//! is never read as a format it is not. Pinned by
//! `tests::header::an_unknown_format_is_an_error_not_eden_s_fallback`.

use starquake_render::PixelFormat as P;

use crate::classes::clc797tex as tex;

/// `ComponentType::SNORM` (the header's `NUM_SNORM`).
pub(super) const SNORM: u32 = tex::TEXHEAD_BL_R_DATA_TYPE_NUM_SNORM;
/// `ComponentType::UNORM` (the header's `NUM_UNORM`).
pub(super) const UNORM: u32 = tex::TEXHEAD_BL_R_DATA_TYPE_NUM_UNORM;
/// `ComponentType::SINT` (the header's `NUM_SINT`).
pub(super) const SINT: u32 = tex::TEXHEAD_BL_R_DATA_TYPE_NUM_SINT;
/// `ComponentType::UINT` (the header's `NUM_UINT`).
pub(super) const UINT: u32 = tex::TEXHEAD_BL_R_DATA_TYPE_NUM_UINT;
/// `ComponentType::FLOAT` (the header's `NUM_FLOAT`).
pub(super) const FLOAT: u32 = tex::TEXHEAD_BL_R_DATA_TYPE_NUM_FLOAT;

/// One row of Eden's table: `COMPONENTS`, the red, green, blue and alpha data types, the
/// `S_R_G_B_CONVERSION` bit, and the format they name.
pub type FormatRow = (u32, [u32; 4], bool, P);

/// Eden's table, in its order (module docs).
pub const FORMAT_TABLE: &[FormatRow] = &[
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::A8B8G8R8Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8,
        [SNORM, SNORM, SNORM, SNORM],
        false,
        P::A8B8G8R8Snorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8,
        [UINT, UINT, UINT, UINT],
        false,
        P::A8B8G8R8Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8,
        [SINT, SINT, SINT, SINT],
        false,
        P::A8B8G8R8Sint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::A8B8G8R8Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_B5G6R5,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::B5G6R5Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_A2B10G10R10,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::A2B10G10R10Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_A2B10G10R10,
        [UINT, UINT, UINT, UINT],
        false,
        P::A2B10G10R10Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_A1B5G5R5,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::A1B5G5R5Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_A4B4G4R4,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::A4B4G4R4Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_G4R4,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::G4R4Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_A5B5G5R1,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::A5B5G5R1Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R8,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::R8Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R8,
        [SNORM, SNORM, SNORM, SNORM],
        false,
        P::R8Snorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R8,
        [UINT, UINT, UINT, UINT],
        false,
        P::R8Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R8,
        [SINT, SINT, SINT, SINT],
        false,
        P::R8Sint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_G8R8,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::R8G8Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_G8R8,
        [SNORM, SNORM, SNORM, SNORM],
        false,
        P::R8G8Snorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_G8R8,
        [UINT, UINT, UINT, UINT],
        false,
        P::R8G8Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_G8R8,
        [SINT, SINT, SINT, SINT],
        false,
        P::R8G8Sint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16_G16_B16_A16,
        [FLOAT, FLOAT, FLOAT, FLOAT],
        false,
        P::R16G16B16A16Float,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16_G16_B16_A16,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::R16G16B16A16Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16_G16_B16_A16,
        [SNORM, SNORM, SNORM, SNORM],
        false,
        P::R16G16B16A16Snorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16_G16_B16_A16,
        [UINT, UINT, UINT, UINT],
        false,
        P::R16G16B16A16Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16_G16_B16_A16,
        [SINT, SINT, SINT, SINT],
        false,
        P::R16G16B16A16Sint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16_G16,
        [FLOAT, FLOAT, FLOAT, FLOAT],
        false,
        P::R16G16Float,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16_G16,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::R16G16Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16_G16,
        [SNORM, SNORM, SNORM, SNORM],
        false,
        P::R16G16Snorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16_G16,
        [UINT, UINT, UINT, UINT],
        false,
        P::R16G16Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16_G16,
        [SINT, SINT, SINT, SINT],
        false,
        P::R16G16Sint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16,
        [FLOAT, FLOAT, FLOAT, FLOAT],
        false,
        P::R16Float,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::R16Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16,
        [SNORM, SNORM, SNORM, SNORM],
        false,
        P::R16Snorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16,
        [UINT, UINT, UINT, UINT],
        false,
        P::R16Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R16,
        [SINT, SINT, SINT, SINT],
        false,
        P::R16Sint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_BF10GF11RF11,
        [FLOAT, FLOAT, FLOAT, FLOAT],
        false,
        P::B10G11R11Float,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R32_G32_B32_A32,
        [FLOAT, FLOAT, FLOAT, FLOAT],
        false,
        P::R32G32B32A32Float,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R32_G32_B32_A32,
        [UINT, UINT, UINT, UINT],
        false,
        P::R32G32B32A32Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R32_G32_B32_A32,
        [SINT, SINT, SINT, SINT],
        false,
        P::R32G32B32A32Sint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R32_G32_B32,
        [FLOAT, FLOAT, FLOAT, FLOAT],
        false,
        P::R32G32B32Float,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R32_G32,
        [FLOAT, FLOAT, FLOAT, FLOAT],
        false,
        P::R32G32Float,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R32_G32,
        [UINT, UINT, UINT, UINT],
        false,
        P::R32G32Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R32_G32,
        [SINT, SINT, SINT, SINT],
        false,
        P::R32G32Sint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R32,
        [FLOAT, FLOAT, FLOAT, FLOAT],
        false,
        P::R32Float,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R32,
        [UINT, UINT, UINT, UINT],
        false,
        P::R32Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R32,
        [SINT, SINT, SINT, SINT],
        false,
        P::R32Sint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_E5B9G9R9_SHAREDEXP,
        [FLOAT, FLOAT, FLOAT, FLOAT],
        false,
        P::E5B9G9R9Float,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ZF32,
        [FLOAT, FLOAT, FLOAT, FLOAT],
        false,
        P::D32Float,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ZF32,
        [FLOAT, UINT, UINT, UINT],
        false,
        P::D32Float,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_Z16,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::D16Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_Z16,
        [UNORM, UINT, UINT, UINT],
        false,
        P::D16Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_X8Z24,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::X8D24Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_X8Z24,
        [UNORM, UINT, UINT, UINT],
        false,
        P::X8D24Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_Z24S8,
        [UINT, UNORM, UNORM, UNORM],
        false,
        P::S8UintD24Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_Z24S8,
        [UINT, UNORM, UINT, UINT],
        false,
        P::S8UintD24Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_G24R8,
        [UINT, UNORM, UNORM, UNORM],
        false,
        P::S8UintD24Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_S8Z24,
        [UNORM, UINT, UINT, UINT],
        false,
        P::D24UnormS8Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ZF32_X24S8,
        [FLOAT, UINT, UNORM, UNORM],
        false,
        P::D32FloatS8Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_R32_B24G8,
        [FLOAT, UINT, UNORM, UNORM],
        false,
        P::D32FloatS8Uint,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_DXT1,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Bc1RgbaUnorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_DXT1,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Bc1RgbaSrgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_DXT23,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Bc2Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_DXT23,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Bc2Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_DXT45,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Bc3Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_DXT45,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Bc3Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_DXN1,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Bc4Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_DXN1,
        [SNORM, SNORM, SNORM, SNORM],
        false,
        P::Bc4Snorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_DXN2,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Bc5Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_DXN2,
        [SNORM, SNORM, SNORM, SNORM],
        false,
        P::Bc5Snorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_BC7U,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Bc7Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_BC7U,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Bc7Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_BC6H_SF16,
        [FLOAT, FLOAT, FLOAT, FLOAT],
        false,
        P::Bc6hSfloat,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_BC6H_UF16,
        [FLOAT, FLOAT, FLOAT, FLOAT],
        false,
        P::Bc6hUfloat,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ETC2_RGB,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Etc2RgbUnorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ETC2_RGB_PTA,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Etc2RgbPtaUnorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ETC2_RGBA,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Etc2RgbaUnorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ETC2_RGB,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Etc2RgbSrgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ETC2_RGB_PTA,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Etc2RgbPtaSrgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ETC2_RGBA,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Etc2RgbaSrgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_EAC,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::EacR11Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_EAC,
        [SNORM, SNORM, SNORM, SNORM],
        false,
        P::EacR11Snorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_EACX2,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::EacR11G11Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_EACX2,
        [SNORM, SNORM, SNORM, SNORM],
        false,
        P::EacR11G11Snorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_4X4,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d4x4Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_4X4,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d4x4Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_5X4,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d5x4Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_5X4,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d5x4Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_5X5,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d5x5Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_5X5,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d5x5Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_8X8,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d8x8Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_8X8,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d8x8Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_8X5,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d8x5Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_8X5,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d8x5Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_10X8,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d10x8Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_10X8,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d10x8Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_6X6,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d6x6Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_6X6,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d6x6Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_10X6,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d10x6Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_10X6,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d10x6Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_10X5,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d10x5Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_10X5,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d10x5Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_10X10,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d10x10Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_10X10,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d10x10Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_12X10,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d12x10Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_12X10,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d12x10Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_12X12,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d12x12Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_12X12,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d12x12Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_8X6,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d8x6Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_8X6,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d8x6Srgb,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_6X5,
        [UNORM, UNORM, UNORM, UNORM],
        false,
        P::Astc2d6x5Unorm,
    ),
    (
        tex::TEXHEAD_BL_COMPONENTS_SIZES_ASTC_2D_6X5,
        [UNORM, UNORM, UNORM, UNORM],
        true,
        P::Astc2d6x5Srgb,
    ),
];

/// The pixel format of a header whose `COMPONENTS` field is `components`, whose red, green,
/// blue and alpha data types are `types`, and whose `S_R_G_B_CONVERSION` bit is `srgb`; `None`
/// for a combination Eden's table does not list (module docs, "Divergence").
pub fn pixel_format(components: u32, types: [u32; 4], srgb: bool) -> Option<P> {
    FORMAT_TABLE
        .iter()
        .find(|&&(c, t, s, _)| c == components && t == types && s == srgb)
        .map(|&(_, _, _, format)| format)
}
