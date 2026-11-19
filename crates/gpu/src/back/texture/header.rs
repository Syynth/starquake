//! Decoding a texture header (`clc797tex.h`) into the image it describes and the view a shader
//! reads it through.
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`.
//!
//! - Every field is read through [`crate::classes::clc797tex`], generated from
//!   `mesa-nouveau/src/nouveau/headers/nvidia/classes/clc797tex.h` @ 52238762 (MIT), in the
//!   layout `HEADER_VERSION` selects: `TEXHEAD_BL_*` (block-linear), `TEXHEAD_BLCK_*`
//!   (block-linear with colour key), `TEXHEAD_PITCH_*` (pitch) or `TEXHEAD_PITCHCK_*` (pitch with
//!   colour key). `HEADER_VERSION` sits at the same bits in all of them.
//! - What the fields mean for the image and the view follows Eden's `ImageInfo(const TICEntry&)`
//!   (`eden/src/video_core/texture_cache/image_info.cpp` @ cf654052) and
//!   `ImageViewInfo(const TICEntry&, s32)` (`image_view_info.cpp`): the level count is
//!   `MAX_MIP_LEVEL + 1`; a 1D, 2D or cube header whose depth is above 1 is an array of that many
//!   layers (six per cube), but a 2D no-mipmap header is one layer whatever its depth (Eden's
//!   depth promotion in `ImageInfo` leaves `Texture2DNoMipmap` out, and its case asserts
//!   `Depth() == 1` and takes one layer; here the depth is ignored without a log, pinned by
//!   `tests::header::a_2d_no_mipmap_header_is_one_layer_whatever_its_depth`); the view's levels are `RES_VIEW_MIN_MIP_LEVEL` to
//!   `RES_VIEW_MAX_MIP_LEVEL`; the multisample modes give the sample grids of Eden's
//!   `NumSamplesX` and `NumSamplesY` (`samples_helper.h`), the virtual-coverage modes those of
//!   their plain grids.
//! - The format is [`super::format::pixel_format`] (Eden's lookup table, with Eden-NX2's added
//!   row).
//!
//! The GOB of a block-linear image is the Switch 2 colour GOB, Mesa's `TuringColor2d` (row 69 of
//! `eden-nx2/NX2_NOTES.md`; the layout crate's `tiling` docs), for colour formats, and
//! `FermiZs` for depth and stencil formats. The second is the gpu-protocol spec's **guess**
//! ("Block-linear layout": Mesa uses `FermiZS` for depth and stencil on Turing; whether NVN2 does
//! is not in the notes), its verification item 9; a depth target dump settles it. Pinned by
//! `tests::header::depth_formats_take_the_guessed_fermi_zs_gob`.
//!
//! # Checks
//!
//! [`TextureHeader::decode`] reports the first failing check, in this order: the header version
//! ([`HeaderError::UnsupportedVersion`]), the format ([`HeaderError::UnknownFormat`]), the
//! swizzle, red to alpha ([`HeaderError::InvalidSwizzle`]), the texture type
//! ([`HeaderError::UnsupportedTextureType`]), the block size of a block-linear header
//! ([`HeaderError::InvalidBlock`]), the multisample mode ([`HeaderError::InvalidSampleMode`]) and
//! the view's level range ([`HeaderError::InvalidLevelRange`]). Pinned by
//! `tests::header::checks_run_in_their_documented_order`. Extents are not checked here: the
//! texture cache lays the image out and checks it against the device and the budget.
//!
//! # Divergences from Eden
//!
//! Eden reads its own `TICEntry` layout, which differs from Nvidia's header; the header wins
//! (decision log 2026-11-12, "Twenty-one graphics decisions", item 2: Nvidia's headers settle
//! layouts). Each is pinned by a test in `tests::header`:
//!
//! - **Address.** Eden takes the whole second word as address bits 0–31. In a block-linear
//!   header its low 9 bits are `RESERVED1Y`, `GOB_DEPTH_OFFSET` and `RESERVED1X`, and in a pitch
//!   header its low 5 are reserved; here they are not address bits, and Eden's 16-bit high
//!   address is the header's 17-bit `ADDRESS_BITS48TO32`
//!   (`the_address_skips_the_low_reserved_bits`).
//! - **Base layer.** Eden reads a base layer from bits the header calls `ADDRESS_BITS48TO32` bit
//!   16, `RESERVED_ADDRESS`, `RESOURCE_VIEW_COHERENCY_HASH` bit 3 and `RESERVED2A`, and from
//!   `WIDTH_MINUS_ONE` bit 16 and the two bits above it. The header has no base layer, so there is
//!   none here, and those bits keep their header meaning (`width_and_height_take_their_top_bits`).
//! - **Width and height.** Eden reads 16 bits of each; the header has 17 (`WIDTH_MINUS_ONE`, and
//!   `HEIGHT_MINUS_ONE` with `HEIGHT_MINUS_ONE_BIT16`), and a block-linear depth has 15
//!   (`DEPTH_MINUS_ONE` with `DEPTH_MINUS_ONE_BIT14`).
//! - **Refusals.** Eden asserts, logs or falls back where this returns an error: an unknown
//!   format (Eden reads `A8B8G8R8_UNORM`), an undefined swizzle source, a 1D-buffer header or
//!   texture type (Eden's buffer path; buffer textures are not this slice's), the `HTEX` types, an
//!   undefined multisample mode, and a view level range whose minimum is above its maximum or
//!   whose maximum is past the image's last level (Eden computes a level count from them
//!   unchecked) (`each_refusal_reports_its_own_error`).
//! - **Swizzle.** `IN_ONE_INT` and `IN_ONE_FLOAT` both become [`SwizzleSource::One`]: the render
//!   crate's swizzle has one constant one, which a view or shader reads in the view's own numeric
//!   type (`both_ones_become_one_and_each_source_maps`).
//! - **Not read.** `TILE_WIDTH_IN_GOBS` (sparse tile spacing), `GOB3D`, `GOB_DEPTH_OFFSET`,
//!   `PACK_COMPONENTS`, `DEPTH_TEXTURE`, `NORMALIZED_COORDS`, the sampling hints (`LOD_*`,
//!   `ANISO_*`, `TRILIN_OPT`, `MIP_LOD_BIAS`, `MIN_LOD_CLAMP`, `SECTOR_PROMOTION`,
//!   `BORDER_SIZE`), the colour key and `RESOURCE_VIEW_COHERENCY_HASH`.

use starquake_layout::{GobType, ImageDim, SampleLayout, Tiling};
use starquake_render::PixelFormat;
use starquake_render::format::FormatAspect;
use starquake_render::resource::{Swizzle, SwizzleSource, ViewDimension};
use thiserror::Error;

use super::format::pixel_format;
use crate::classes::MwField;
use crate::classes::clc797tex as tex;

/// Words in a texture header (`clc797tex.h`'s fields end at bit 255).
pub const HEADER_WORDS: usize = 8;

/// Faces of a cube: the layers one cube takes.
pub const CUBE_FACES: u32 = 6;

/// How an image's bytes are laid out in guest memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    /// Block-linear, in blocks of one GOB across, `2^block_height_log2` GOBs down and
    /// `2^block_depth_log2` deep, of `gob`.
    BlockLinear {
        /// The GOB kind.
        gob: GobType,
        /// Log2 of the block's height in GOBs.
        block_height_log2: u8,
        /// Log2 of the block's depth in GOBs.
        block_depth_log2: u8,
    },
    /// Pitch-linear: rows `pitch_b` bytes apart.
    Pitch {
        /// Bytes from one row to the next.
        pitch_b: u32,
    },
}

impl MemoryLayout {
    /// The layout crate's tiling for this layout, and the pitch it takes.
    pub fn tiling(self) -> Result<(Tiling, Option<u32>), starquake_layout::LayoutError> {
        match self {
            Self::BlockLinear {
                gob,
                block_height_log2,
                block_depth_log2,
            } => Ok((
                Tiling::new(gob, 0, block_height_log2, block_depth_log2)?,
                None,
            )),
            Self::Pitch { pitch_b } => Ok((Tiling::LINEAR, Some(pitch_b))),
        }
    }
}

/// An image in guest memory: everything its layout and its host texture depend on. Two uses of
/// the same `ImageInfo` are one image in the texture cache.
///
/// Built by [`TextureHeader::decode`] for a sampled texture, or by the caller from the render
/// target registers. Nothing here is checked until the texture cache lays the image out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageInfo {
    /// The GPU address of its first byte.
    pub address: u64,
    /// Its texel format.
    pub format: PixelFormat,
    /// 1D, 2D or 3D. A cube is a 2D image of six layers per cube.
    pub dim: ImageDim,
    /// Level 0's width in pixels.
    pub width: u32,
    /// Level 0's height in pixels (1 for 1D).
    pub height: u32,
    /// Level 0's depth in slices (1 unless 3D).
    pub depth: u32,
    /// Layers (1 for 3D).
    pub layers: u32,
    /// Mip levels.
    pub levels: u32,
    /// The sample grid of one pixel.
    pub samples: SampleLayout,
    /// Block-linear or pitch.
    pub layout: MemoryLayout,
}

/// The part of a header that says how a shader reads the image: not part of its identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewInfo {
    /// How the view presents the image.
    pub dimension: ViewDimension,
    /// The first level the view reads.
    pub base_mip: u32,
    /// How many levels it reads.
    pub mip_count: u32,
    /// The header's component swizzle (`X_SOURCE` to `W_SOURCE`).
    pub swizzle: Swizzle,
}

/// A decoded texture header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHeader {
    /// The image.
    pub image: ImageInfo,
    /// The view of it.
    pub view: ViewInfo,
}

/// Why a texture header could not be decoded (module docs, "Checks", for their order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum HeaderError {
    /// `HEADER_VERSION` is the 1D-buffer form or a value the header does not define.
    #[error("texture header version {version} is not supported")]
    UnsupportedVersion {
        /// The `HEADER_VERSION` field.
        version: u32,
    },
    /// The format fields name no format of the table ([`super::format::FORMAT_TABLE`]).
    #[error(
        "texture format {components:#x} with data types {} (red to alpha) and sRGB conversion {} \
         is not known",
        data_types(.types),
        on_off(*.srgb)
    )]
    UnknownFormat {
        /// `COMPONENTS`.
        components: u32,
        /// `R_DATA_TYPE` to `A_DATA_TYPE`.
        types: [u32; 4],
        /// `S_R_G_B_CONVERSION`.
        srgb: bool,
    },
    /// A swizzle field holds a value the header does not define.
    #[error("texture swizzle source {value} for component {component} is not defined")]
    InvalidSwizzle {
        /// `r`, `g`, `b` or `a`: the component whose source it is (`X_SOURCE` is red).
        component: char,
        /// The field.
        value: u32,
    },
    /// `TEXTURE_TYPE` is a 1D buffer, an `HTEX` type, or a value the header does not define.
    #[error("texture type {texture_type} is not supported")]
    UnsupportedTextureType {
        /// The `TEXTURE_TYPE` field.
        texture_type: u32,
    },
    /// A block-linear header's block is wider than one GOB, or taller or deeper than 32 GOBs.
    #[error(
        "texture block of {width}, {height} and {depth} (log2 GOBs) is not one the header defines"
    )]
    InvalidBlock {
        /// `GOBS_PER_BLOCK_WIDTH`.
        width: u32,
        /// `GOBS_PER_BLOCK_HEIGHT`.
        height: u32,
        /// `GOBS_PER_BLOCK_DEPTH`.
        depth: u32,
    },
    /// `MULTI_SAMPLE_COUNT` is a value the header does not define.
    #[error("texture multisample mode {mode} is not defined")]
    InvalidSampleMode {
        /// The field.
        mode: u32,
    },
    /// The view's levels start above where they end, or end past the image's last level.
    #[error("texture view levels {min} to {max} do not fit an image of {levels} levels")]
    InvalidLevelRange {
        /// `RES_VIEW_MIN_MIP_LEVEL`.
        min: u32,
        /// `RES_VIEW_MAX_MIP_LEVEL`.
        max: u32,
        /// `MAX_MIP_LEVEL + 1`.
        levels: u32,
    },
}

/// Four data types in plain words: `1, 2, 3 and 4`.
fn data_types(types: &[u32; 4]) -> String {
    let [r, g, b, a] = types;
    format!("{r}, {g}, {b} and {a}")
}

/// `on` or `off`.
fn on_off(on: bool) -> &'static str {
    if on { "on" } else { "off" }
}

/// Where one header form keeps the fields the decode reads.
struct Fields {
    components: MwField,
    types: [MwField; 4],
    sources: [MwField; 4],
    /// The low address field and the address bit it starts at.
    address_low: (MwField, u32),
    address_high: MwField,
    width_minus_one: MwField,
    height_minus_one: MwField,
    height_bit16: MwField,
    depth_minus_one: MwField,
    /// Block-linear forms only.
    depth_bit14: Option<MwField>,
    max_mip_level: MwField,
    srgb: MwField,
    texture_type: MwField,
    /// Block-linear forms: `GOBS_PER_BLOCK_WIDTH`, `_HEIGHT`, `_DEPTH`.
    block: Option<[MwField; 3]>,
    /// Pitch forms: `PITCH_BITS20TO5` and `PITCH_BIT21`.
    pitch: Option<(MwField, MwField)>,
    /// Forms without a colour key: `MULTI_SAMPLE_COUNT`.
    samples: Option<MwField>,
    /// Forms without a colour key: `RES_VIEW_MIN_MIP_LEVEL` and `_MAX_`.
    view_levels: Option<(MwField, MwField)>,
}

/// Address bit 9: where a block-linear header's low address field starts (`ADDRESS_BITS31TO9`).
const BL_ADDRESS_SHIFT: u32 = 9;
/// Address bit 5: where a pitch header's low address field starts (`ADDRESS_BITS31TO5`).
const PITCH_ADDRESS_SHIFT: u32 = 5;
/// Address bit 32: where `ADDRESS_BITS48TO32` starts.
const ADDRESS_HIGH_SHIFT: u32 = 32;
/// Pitch bit 5: where `PITCH_BITS20TO5` starts.
const PITCH_LOW_SHIFT: u32 = 5;
/// Pitch bit 21: `PITCH_BIT21`.
const PITCH_HIGH_SHIFT: u32 = 21;
/// Height bit 16: `HEIGHT_MINUS_ONE_BIT16`.
const HEIGHT_HIGH_SHIFT: u32 = 16;
/// Depth bit 14: `DEPTH_MINUS_ONE_BIT14`.
const DEPTH_HIGH_SHIFT: u32 = 14;

const BL: Fields = Fields {
    components: tex::TEXHEAD_BL_COMPONENTS,
    types: [
        tex::TEXHEAD_BL_R_DATA_TYPE,
        tex::TEXHEAD_BL_G_DATA_TYPE,
        tex::TEXHEAD_BL_B_DATA_TYPE,
        tex::TEXHEAD_BL_A_DATA_TYPE,
    ],
    sources: [
        tex::TEXHEAD_BL_X_SOURCE,
        tex::TEXHEAD_BL_Y_SOURCE,
        tex::TEXHEAD_BL_Z_SOURCE,
        tex::TEXHEAD_BL_W_SOURCE,
    ],
    address_low: (tex::TEXHEAD_BL_ADDRESS_BITS31TO9, BL_ADDRESS_SHIFT),
    address_high: tex::TEXHEAD_BL_ADDRESS_BITS48TO32,
    width_minus_one: tex::TEXHEAD_BL_WIDTH_MINUS_ONE,
    height_minus_one: tex::TEXHEAD_BL_HEIGHT_MINUS_ONE,
    height_bit16: tex::TEXHEAD_BL_HEIGHT_MINUS_ONE_BIT16,
    depth_minus_one: tex::TEXHEAD_BL_DEPTH_MINUS_ONE,
    depth_bit14: Some(tex::TEXHEAD_BL_DEPTH_MINUS_ONE_BIT14),
    max_mip_level: tex::TEXHEAD_BL_MAX_MIP_LEVEL,
    srgb: tex::TEXHEAD_BL_S_R_G_B_CONVERSION,
    texture_type: tex::TEXHEAD_BL_TEXTURE_TYPE,
    block: Some([
        tex::TEXHEAD_BL_GOBS_PER_BLOCK_WIDTH,
        tex::TEXHEAD_BL_GOBS_PER_BLOCK_HEIGHT,
        tex::TEXHEAD_BL_GOBS_PER_BLOCK_DEPTH,
    ]),
    pitch: None,
    samples: Some(tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT),
    view_levels: Some((
        tex::TEXHEAD_BL_RES_VIEW_MIN_MIP_LEVEL,
        tex::TEXHEAD_BL_RES_VIEW_MAX_MIP_LEVEL,
    )),
};

const BLCK: Fields = Fields {
    components: tex::TEXHEAD_BLCK_COMPONENTS,
    types: [
        tex::TEXHEAD_BLCK_R_DATA_TYPE,
        tex::TEXHEAD_BLCK_G_DATA_TYPE,
        tex::TEXHEAD_BLCK_B_DATA_TYPE,
        tex::TEXHEAD_BLCK_A_DATA_TYPE,
    ],
    sources: [
        tex::TEXHEAD_BLCK_X_SOURCE,
        tex::TEXHEAD_BLCK_Y_SOURCE,
        tex::TEXHEAD_BLCK_Z_SOURCE,
        tex::TEXHEAD_BLCK_W_SOURCE,
    ],
    address_low: (tex::TEXHEAD_BLCK_ADDRESS_BITS31TO9, BL_ADDRESS_SHIFT),
    address_high: tex::TEXHEAD_BLCK_ADDRESS_BITS48TO32,
    width_minus_one: tex::TEXHEAD_BLCK_WIDTH_MINUS_ONE,
    height_minus_one: tex::TEXHEAD_BLCK_HEIGHT_MINUS_ONE,
    height_bit16: tex::TEXHEAD_BLCK_HEIGHT_MINUS_ONE_BIT16,
    depth_minus_one: tex::TEXHEAD_BLCK_DEPTH_MINUS_ONE,
    depth_bit14: Some(tex::TEXHEAD_BLCK_DEPTH_MINUS_ONE_BIT14),
    max_mip_level: tex::TEXHEAD_BLCK_MAX_MIP_LEVEL,
    srgb: tex::TEXHEAD_BLCK_S_R_G_B_CONVERSION,
    texture_type: tex::TEXHEAD_BLCK_TEXTURE_TYPE,
    block: Some([
        tex::TEXHEAD_BLCK_GOBS_PER_BLOCK_WIDTH,
        tex::TEXHEAD_BLCK_GOBS_PER_BLOCK_HEIGHT,
        tex::TEXHEAD_BLCK_GOBS_PER_BLOCK_DEPTH,
    ]),
    pitch: None,
    samples: None,
    view_levels: None,
};

const PITCH: Fields = Fields {
    components: tex::TEXHEAD_PITCH_COMPONENTS,
    types: [
        tex::TEXHEAD_PITCH_R_DATA_TYPE,
        tex::TEXHEAD_PITCH_G_DATA_TYPE,
        tex::TEXHEAD_PITCH_B_DATA_TYPE,
        tex::TEXHEAD_PITCH_A_DATA_TYPE,
    ],
    sources: [
        tex::TEXHEAD_PITCH_X_SOURCE,
        tex::TEXHEAD_PITCH_Y_SOURCE,
        tex::TEXHEAD_PITCH_Z_SOURCE,
        tex::TEXHEAD_PITCH_W_SOURCE,
    ],
    address_low: (tex::TEXHEAD_PITCH_ADDRESS_BITS31TO5, PITCH_ADDRESS_SHIFT),
    address_high: tex::TEXHEAD_PITCH_ADDRESS_BITS48TO32,
    width_minus_one: tex::TEXHEAD_PITCH_WIDTH_MINUS_ONE,
    height_minus_one: tex::TEXHEAD_PITCH_HEIGHT_MINUS_ONE,
    height_bit16: tex::TEXHEAD_PITCH_HEIGHT_MINUS_ONE_BIT16,
    depth_minus_one: tex::TEXHEAD_PITCH_DEPTH_MINUS_ONE,
    depth_bit14: None,
    max_mip_level: tex::TEXHEAD_PITCH_MAX_MIP_LEVEL,
    srgb: tex::TEXHEAD_PITCH_S_R_G_B_CONVERSION,
    texture_type: tex::TEXHEAD_PITCH_TEXTURE_TYPE,
    block: None,
    pitch: Some((
        tex::TEXHEAD_PITCH_PITCH_BITS20TO5,
        tex::TEXHEAD_PITCH_PITCH_BIT21,
    )),
    samples: Some(tex::TEXHEAD_PITCH_MULTI_SAMPLE_COUNT),
    view_levels: Some((
        tex::TEXHEAD_PITCH_RES_VIEW_MIN_MIP_LEVEL,
        tex::TEXHEAD_PITCH_RES_VIEW_MAX_MIP_LEVEL,
    )),
};

const PITCHCK: Fields = Fields {
    components: tex::TEXHEAD_PITCHCK_COMPONENTS,
    types: [
        tex::TEXHEAD_PITCHCK_R_DATA_TYPE,
        tex::TEXHEAD_PITCHCK_G_DATA_TYPE,
        tex::TEXHEAD_PITCHCK_B_DATA_TYPE,
        tex::TEXHEAD_PITCHCK_A_DATA_TYPE,
    ],
    sources: [
        tex::TEXHEAD_PITCHCK_X_SOURCE,
        tex::TEXHEAD_PITCHCK_Y_SOURCE,
        tex::TEXHEAD_PITCHCK_Z_SOURCE,
        tex::TEXHEAD_PITCHCK_W_SOURCE,
    ],
    address_low: (tex::TEXHEAD_PITCHCK_ADDRESS_BITS31TO5, PITCH_ADDRESS_SHIFT),
    address_high: tex::TEXHEAD_PITCHCK_ADDRESS_BITS48TO32,
    width_minus_one: tex::TEXHEAD_PITCHCK_WIDTH_MINUS_ONE,
    height_minus_one: tex::TEXHEAD_PITCHCK_HEIGHT_MINUS_ONE,
    height_bit16: tex::TEXHEAD_PITCHCK_HEIGHT_MINUS_ONE_BIT16,
    depth_minus_one: tex::TEXHEAD_PITCHCK_DEPTH_MINUS_ONE,
    depth_bit14: None,
    max_mip_level: tex::TEXHEAD_PITCHCK_MAX_MIP_LEVEL,
    srgb: tex::TEXHEAD_PITCHCK_S_R_G_B_CONVERSION,
    texture_type: tex::TEXHEAD_PITCHCK_TEXTURE_TYPE,
    block: None,
    pitch: Some((
        tex::TEXHEAD_PITCHCK_PITCH_BITS20TO5,
        tex::TEXHEAD_PITCHCK_PITCH_BIT21,
    )),
    samples: None,
    view_levels: None,
};

/// The field's value; every field of the tables above lies inside a header.
fn get(field: MwField, words: &[u32; HEADER_WORDS]) -> u32 {
    field.get(words).unwrap_or(0)
}

/// The view source a swizzle field names (`*_SOURCE_IN_*`), `None` for an undefined value.
fn swizzle_source(value: u32) -> Option<SwizzleSource> {
    Some(match value {
        tex::TEXHEAD_BL_X_SOURCE_IN_ZERO => SwizzleSource::Zero,
        tex::TEXHEAD_BL_X_SOURCE_IN_R => SwizzleSource::R,
        tex::TEXHEAD_BL_X_SOURCE_IN_G => SwizzleSource::G,
        tex::TEXHEAD_BL_X_SOURCE_IN_B => SwizzleSource::B,
        tex::TEXHEAD_BL_X_SOURCE_IN_A => SwizzleSource::A,
        tex::TEXHEAD_BL_X_SOURCE_IN_ONE_INT | tex::TEXHEAD_BL_X_SOURCE_IN_ONE_FLOAT => {
            SwizzleSource::One
        }
        _ => return None,
    })
}

/// The sample grid of a `MULTI_SAMPLE_COUNT` mode, `None` for an undefined one.
fn sample_layout(mode: u32) -> Option<SampleLayout> {
    Some(match mode {
        tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_1X1 => SampleLayout::S1x1,
        tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X1 => SampleLayout::S2x1,
        tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X1_D3D => SampleLayout::S2x1D3d,
        tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X2
        | tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X2_VC_4
        | tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_2X2_VC_12 => SampleLayout::S2x2,
        tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2
        | tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2_VC_8
        | tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2_VC_24 => SampleLayout::S4x2,
        tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X2_D3D => SampleLayout::S4x2D3d,
        tex::TEXHEAD_BL_MULTI_SAMPLE_COUNT_MODE_4X4 => SampleLayout::S4x4,
        _ => return None,
    })
}

/// The block-linear GOB of `format` (module docs: the depth one is a guess).
pub fn gob_for(format: PixelFormat) -> GobType {
    match format.aspect() {
        FormatAspect::Color => GobType::TuringColor2d,
        FormatAspect::Depth | FormatAspect::Stencil | FormatAspect::DepthStencil => {
            GobType::FermiZs
        }
    }
}

/// The image dimension, layer count, depth and view dimension of a texture type, for a header
/// whose `depth` is `depth`; `None` for a type this decode does not support.
fn shape(texture_type: u32, depth: u32) -> Option<(ImageDim, u32, u32, ViewDimension)> {
    let arrayed = depth > 1;
    Some(match texture_type {
        tex::TEXHEAD_BL_TEXTURE_TYPE_ONE_D if !arrayed => (ImageDim::D1, 1, 1, ViewDimension::D1),
        tex::TEXHEAD_BL_TEXTURE_TYPE_ONE_D | tex::TEXHEAD_BL_TEXTURE_TYPE_ONE_D_ARRAY => {
            (ImageDim::D1, depth, 1, ViewDimension::D1Array)
        }
        // A 2D no-mipmap header is never an array, as in Eden (module docs).
        tex::TEXHEAD_BL_TEXTURE_TYPE_TWO_D_NO_MIPMAP => (ImageDim::D2, 1, 1, ViewDimension::D2),
        tex::TEXHEAD_BL_TEXTURE_TYPE_TWO_D if !arrayed => (ImageDim::D2, 1, 1, ViewDimension::D2),
        tex::TEXHEAD_BL_TEXTURE_TYPE_TWO_D | tex::TEXHEAD_BL_TEXTURE_TYPE_TWO_D_ARRAY => {
            (ImageDim::D2, depth, 1, ViewDimension::D2Array)
        }
        tex::TEXHEAD_BL_TEXTURE_TYPE_THREE_D => (ImageDim::D3, 1, depth, ViewDimension::D3),
        tex::TEXHEAD_BL_TEXTURE_TYPE_CUBEMAP if !arrayed => {
            (ImageDim::D2, CUBE_FACES, 1, ViewDimension::Cube)
        }
        tex::TEXHEAD_BL_TEXTURE_TYPE_CUBEMAP | tex::TEXHEAD_BL_TEXTURE_TYPE_CUBEMAP_ARRAY => (
            ImageDim::D2,
            // At most 2^15 cubes of six layers: well inside 32 bits.
            depth.saturating_mul(CUBE_FACES),
            1,
            ViewDimension::CubeArray,
        ),
        _ => return None,
    })
}

impl Fields {
    /// `X_SOURCE` to `W_SOURCE`, checked red to alpha.
    fn swizzle(&self, words: &[u32; HEADER_WORDS]) -> Result<Swizzle, HeaderError> {
        let mut sources = [SwizzleSource::Zero; 4];
        for ((source, field), component) in sources
            .iter_mut()
            .zip(self.sources)
            .zip(['r', 'g', 'b', 'a'])
        {
            let value = get(field, words);
            *source =
                swizzle_source(value).ok_or(HeaderError::InvalidSwizzle { component, value })?;
        }
        let [red, green, blue, alpha] = sources;
        Ok(Swizzle {
            r: red,
            g: green,
            b: blue,
            a: alpha,
        })
    }

    /// The block size of a block-linear form, or the pitch of a pitch form.
    fn layout(
        &self,
        words: &[u32; HEADER_WORDS],
        format: PixelFormat,
    ) -> Result<MemoryLayout, HeaderError> {
        let Some([width, height, depth]) = self.block else {
            let (low, high) = self
                .pitch
                .map_or((0, 0), |(low, high)| (get(low, words), get(high, words)));
            return Ok(MemoryLayout::Pitch {
                pitch_b: low << PITCH_LOW_SHIFT | high << PITCH_HIGH_SHIFT,
            });
        };
        let (width_log2, height_log2, depth_log2) =
            (get(width, words), get(height, words), get(depth, words));
        let invalid = HeaderError::InvalidBlock {
            width: width_log2,
            height: height_log2,
            depth: depth_log2,
        };
        if width_log2 != tex::TEXHEAD_BL_GOBS_PER_BLOCK_WIDTH_ONE_GOB
            || height_log2 > tex::TEXHEAD_BL_GOBS_PER_BLOCK_HEIGHT_THIRTYTWO_GOBS
            || depth_log2 > tex::TEXHEAD_BL_GOBS_PER_BLOCK_DEPTH_THIRTYTWO_GOBS
        {
            return Err(invalid);
        }
        Ok(MemoryLayout::BlockLinear {
            gob: gob_for(format),
            block_height_log2: u8::try_from(height_log2).map_err(|_| invalid)?,
            block_depth_log2: u8::try_from(depth_log2).map_err(|_| invalid)?,
        })
    }

    /// The level count, and the view's first level and level count.
    fn levels(&self, words: &[u32; HEADER_WORDS]) -> Result<(u32, u32, u32), HeaderError> {
        let levels = get(self.max_mip_level, words).saturating_add(1);
        let Some((min, max)) = self.view_levels else {
            return Ok((levels, 0, levels));
        };
        let (min, max) = (get(min, words), get(max, words));
        if min > max || max >= levels {
            return Err(HeaderError::InvalidLevelRange { min, max, levels });
        }
        Ok((levels, min, max - min + 1))
    }
}

impl TextureHeader {
    /// Decodes a header's eight words (module docs, "Checks", for the order of the errors).
    pub fn decode(words: &[u32; HEADER_WORDS]) -> Result<Self, HeaderError> {
        let version = get(tex::TEXHEAD_BL_HEADER_VERSION, words);
        let fields = match version {
            tex::TEXHEAD_BL_HEADER_VERSION_SELECT_BLOCKLINEAR => &BL,
            tex::TEXHEAD_BL_HEADER_VERSION_SELECT_BLOCKLINEAR_COLOR_KEY => &BLCK,
            tex::TEXHEAD_BL_HEADER_VERSION_SELECT_PITCH => &PITCH,
            tex::TEXHEAD_BL_HEADER_VERSION_SELECT_PITCH_COLOR_KEY => &PITCHCK,
            _ => return Err(HeaderError::UnsupportedVersion { version }),
        };
        let field = |f: MwField| get(f, words);

        let components = field(fields.components);
        let types = fields.types.map(field);
        let srgb = field(fields.srgb) != 0;
        let format = pixel_format(components, types, srgb).ok_or(HeaderError::UnknownFormat {
            components,
            types,
            srgb,
        })?;

        let swizzle = fields.swizzle(words)?;

        let width = field(fields.width_minus_one).saturating_add(1);
        let height = (field(fields.height_minus_one)
            | field(fields.height_bit16) << HEIGHT_HIGH_SHIFT)
            .saturating_add(1);
        let depth_bit14 = fields.depth_bit14.map_or(0, field);
        let depth =
            (field(fields.depth_minus_one) | depth_bit14 << DEPTH_HIGH_SHIFT).saturating_add(1);

        let texture_type = field(fields.texture_type);
        let (dim, layers, depth, dimension) = shape(texture_type, depth)
            .ok_or(HeaderError::UnsupportedTextureType { texture_type })?;
        let height = if dim == ImageDim::D1 { 1 } else { height };

        let layout = fields.layout(words, format)?;
        let samples = match fields.samples {
            Some(f) => {
                let mode = field(f);
                sample_layout(mode).ok_or(HeaderError::InvalidSampleMode { mode })?
            }
            None => SampleLayout::S1x1,
        };
        let (levels, base_mip, mip_count) = fields.levels(words)?;

        let (low, low_shift) = fields.address_low;
        let address = u64::from(field(low)) << low_shift
            | u64::from(field(fields.address_high)) << ADDRESS_HIGH_SHIFT;

        Ok(Self {
            image: ImageInfo {
                address,
                format,
                dim,
                width,
                height,
                depth,
                layers,
                levels,
                samples,
                layout,
            },
            view: ViewInfo {
                dimension,
                base_mip,
                mip_count,
                swizzle,
            },
        })
    }
}
