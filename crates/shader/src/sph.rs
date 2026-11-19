//! The shader program header (SPH) in front of every graphics program: version 4, 128 bytes,
//! parsed into a typed [`ProgramHeader`] (`docs/specs/shader-pipeline.md`, "Where programs come
//! from" and port slice 2).
//!
//! The header says which stage a program is for, which attributes and system values it reads
//! and writes, how a pixel program interpolates each input component, which colour targets it
//! writes, and its local memory size; it also carries the geometry and tessellation fields
//! (spec, "Deferred, not ruled out": "the SPH parser reads their fields"). Slice 15 (inputs and
//! outputs) and the local and shared memory pass (spec, "Passes needed for correctness", items
//! 5 and 6) read it.
//!
//! # Sources
//!
//! - The layout: `mesa-nouveau/src/nouveau/headers/nvidia/classes/clc797sph.h` @ 52238762 (MIT,
//!   Copyright (c) 2003-2021 NVIDIA CORPORATION & AFFILIATES), field by field in the private
//!   `layout` module, which names each define. Two header types share the first five words: type
//!   1 for vertex, tessellation and geometry programs, type 2 for pixel programs.
//! - Cross-checked with NAK's writer, `mesa-nouveau/src/nouveau/compiler/nak/sph.rs` @ 52238762
//!   (MIT, Copyright 2023 Collabora, Ltd.), and Eden's reader,
//!   `eden/src/shader_recompiler/program_header.h` @ cf654052, an 80-byte layout (version 3, the
//!   first 20 words): every field Eden names is at the position read here.
//! - The size, 128 bytes ([`HEADER_BYTES`]), is cited, not measured: NAK writes
//!   `SPHV4_SHADER_HEADER_SIZE` = 32 words for SM 7.3 and later, and PR #4588 skips 128 bytes in
//!   front of a graphics program (`eden-pr4588/src/video_core/shader_environment.cpp`,
//!   `InitializeNativeCode(u32 source_header_size = 128, ..)`). Verification E3 (spec, "What to
//!   verify before building", item 3) confirms it on a homebrew or local shader; until then it
//!   is the sources' value.
//! - Byte order: the header is 32 little-endian words, bit `n` being bit `n % 32` of word
//!   `n / 32`, the convention of `clc797sph.h`'s `MW(high:low)` ranges and of NAK's
//!   `BitMutView` over its `[u32; 32]`. Eden reads the bytes into a struct of `u32` bit fields,
//!   which is the same on a little-endian host.
//!
//! # Checks, in order
//!
//! [`ProgramHeader::from_program`] refuses fewer than [`HEADER_BYTES`] bytes
//! ([`HeaderError::Short`]); then [`ProgramHeader::from_words`] checks, in this order:
//!
//! 1. the version is [`ProgramHeader::VERSION`] ([`HeaderError::Version`]), before anything
//!    else is read;
//! 2. the header type is 1 or 2 ([`HeaderError::SphType`]);
//! 3. the shader type is one of 1 to 5 ([`HeaderError::ShaderType`]);
//! 4. the header type matches the shader type: type 2 for a pixel program, type 1 for the others
//!    ([`HeaderError::TypeMismatch`]);
//! 5. the output topology is 0 (none), 1, 6 or 7 ([`HeaderError::OutputTopology`]).
//!
//! Every other field takes any value its width holds: they are counts, masks, flags and
//! interpolation modes (all four two-bit values are modes). The test
//! `the_checks_run_in_the_documented_order` has each pair of faults present at once.
//!
//! # Bits not read
//!
//! The parser reads every field `clc797sph.h` names except `RESERVED_COMMON_A` and the padding.
//! It reads `RESERVED_COMMON_B` (bits 26 and 27) and `RESERVED_COMMON_C` (bits 96 to 119) as
//! the version 3 fields at those positions, and the unnamed bits 24 and 25 as NAK's (the private
//! `layout` module cites each). It ignores exactly these bits, whatever they hold
//! (`ignored_bits_are_exactly_the_unnamed_ones` flips every bit of the header to check it):
//!
//! - both types: 21 to 23 (22 is `RESERVED_COMMON_A`; 21 and 23 are unnamed), 124 to 127 and
//!   148 to 151 (unnamed in version 4; version 3's `RESERVED_COMMON_B` and `RESERVED_COMMON_C`);
//! - type 1: bit 187 of each map (347, 587, 827: unnamed, between the fog coordinate and the
//!   tessellation point) and the padding, 880 to 1023;
//! - type 2: 475 (the same unnamed bit), 612 to 639 (unnamed), and the padding, 872 to 1023.
//!
//! None of them is refused when set. NAK writes into two of these ranges on every SM: a
//! tessellation control program's per-patch attribute count goes into bits 124 to 127 (low
//! half) and 148 to 151 (high half) as well as its own field (`set_per_patch_attribute_count`),
//! so a set reserved bit is not evidence of a malformed header.
//!
//! # Where this differs from the references
//!
//! - Eden checks nothing and reads the layout its pipeline slot names (`ProgramHeader`'s `vtg`
//!   or `ps` union member, chosen by the stage). This parser refuses an unknown version, header
//!   type, shader type or output topology, and a header type that does not match the shader type
//!   (above); pinned by `each_check_refuses_with_its_exact_error`. Whether a header matches its
//!   pipeline slot is the caller's question ([`ProgramHeader::stage`]).
//! - The output topology is checked for every stage, though Eden reads it for geometry programs
//!   only (`eden/src/shader_recompiler/frontend/maxwell/translate_program.cpp`); pinned by
//!   `output_topology_is_checked_for_every_stage`. Should a title carry another value outside
//!   geometry programs, the check moves to geometry headers.
//! - Eden's header is 80 bytes; the words after the twentieth (the type 1 header's third map,
//!   the type 2 header's `_EXT` fields) are read here and not by Eden.
//! - **Conflict:** the local memory size. Eden composes it as the low field plus the high field
//!   shifted by 24 (`ProgramHeader::LocalMemorySize`); NAK's writer stores bits 32 to 55 of the
//!   size in the high field (`set_shader_local_memory_size`: `(size >> 32) & 0xffffff`), so the
//!   two disagree once the high field is non-zero, at 16 MiB and above. Eden's reading wins
//!   here, since PR #4588 runs Switch 2 titles on it ([`Common::local_memory_size`]); pinned by
//!   `local_memory_size_follows_eden`. Verification E3 can settle it with a large-local-memory
//!   shader.

#[cfg(feature = "test-support")]
pub mod encode;
mod layout;
#[cfg(test)]
mod tests;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::translate::{COLOR_TARGETS, ColorOutputs, GuestStage};
use layout::{
    BARYCENTRIC_CELL, BARYCENTRIC_W_CELL, BARYCENTRICS, CLIP_DISTANCES, COLOR_EXT, COMPONENTS,
    DOES_FP64, DOES_GLOBAL_STORE, DOES_LOAD_OR_STORE, FOG_COORDINATE, GENERIC_ATTRIBUTES,
    GENERIC_EXT, GEOMETRY_PASSTHROUGH, HeaderField, IMAP, IMAP_CONSTANT, IMAP_PERSPECTIVE,
    IMAP_UNUSED, INSTANCE_ID, ISBE_SPACE_SHARING, KILLS_PIXELS, LOCAL_MEMORY_CRS_SIZE,
    LOCAL_MEMORY_HIGH_SIZE, LOCAL_MEMORY_LOW_SIZE, MAX_OUTPUT_VERTEX_COUNT, MRT_ENABLE, OMAP,
    OMAP_DEPTH, OMAP_SAMPLE_MASK, OUTPUT_TOPOLOGY, OUTPUT_TOPOLOGY_LINESTRIP, OUTPUT_TOPOLOGY_NONE,
    OUTPUT_TOPOLOGY_POINTLIST, OUTPUT_TOPOLOGY_TRIANGLESTRIP, PER_PATCH_ATTRIBUTE_COUNT,
    POINT_SIZE, POINT_SPRITE_S, POINT_SPRITE_T, POSITION, PRIMITIVE_ID, PS_COLOR, PS_COLORS,
    PS_GENERIC, PS_SYSTEM_VALUES_C, PS_SYSTEM_VALUES_D, PS_TARGETS, PS_TEXTURE,
    READS_INTERLOCK_TICKET, RT_ARRAY_INDEX, SASS_VERSION, SHADER_TYPE, SHADER_TYPE_GEOMETRY,
    SHADER_TYPE_PIXEL, SHADER_TYPE_TESSELLATION, SHADER_TYPE_TESSELLATION_INIT, SHADER_TYPE_VERTEX,
    SPH_TYPE, SPH_TYPE_PS, SPH_TYPE_VTG, STORE_REQ_END, STORE_REQ_START, STREAM_OUT_MASK,
    SYSTEM_VALUES_A, SYSTEM_VALUES_A_EXT, SYSTEM_VALUES_B_EXT, SYSTEM_VALUES_C_EXT,
    SYSTEM_VALUES_D_EXT, TESSELLATION_POINT_U, TESSELLATION_POINT_V, TEXTURE_COORDINATES,
    TEXTURE_EXT, THREADS_PER_INPUT_PRIMITIVE, USES_INNER_COVERAGE, VERSION, VERTEX_ID,
    VIEWPORT_INDEX, VIEWPORT_MASK_CELL, VIEWPORT_MASK2_CELL, VMAP, VPRS_TABLE_INDEX, VTG_COLOR,
    VTG_COLORS, VTG_GENERIC, VTG_SYSTEM_VALUES_C, VTG_SYSTEM_VALUES_D, VTG_TEXTURE,
};

/// How many 32-bit words the header has (NAK `SPHV4_SHADER_HEADER_SIZE`).
pub const HEADER_WORDS: usize = 32;

/// How many bytes the header takes in front of a graphics program's first instruction.
pub const HEADER_BYTES: usize = HEADER_WORDS * 4;

/// Why a header was refused. The variants are listed in the order the parser checks them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum HeaderError {
    /// The program is shorter than the header.
    #[error("a shader program header needs {needed} bytes, but the program has {len}")]
    Short {
        /// How many bytes the program has.
        len: usize,
        /// [`HEADER_BYTES`].
        needed: usize,
    },
    /// The header is not version 4.
    #[error(
        "shader program header version {version} is not supported; only version {supported} is"
    )]
    Version {
        /// The version field.
        version: u32,
        /// [`ProgramHeader::VERSION`].
        supported: u32,
    },
    /// The header type is neither 1 nor 2.
    #[error(
        "shader program header type {value} is neither 1 (vertex, tessellation or geometry) nor 2 (pixel)"
    )]
    SphType {
        /// The header type field.
        value: u32,
    },
    /// The shader type is not one of the five stages.
    #[error("shader type {value} is not one of 1 (vertex) to 5 (pixel)")]
    ShaderType {
        /// The shader type field.
        value: u32,
    },
    /// The header type does not suit the shader type.
    #[error("a {layout} header cannot describe a {stage} program")]
    TypeMismatch {
        /// The layout the header type names.
        layout: Layout,
        /// The stage the shader type names.
        stage: GuestStage,
    },
    /// The output topology is not one the header defines.
    #[error(
        "output topology {value} is not 0 (none), 1 (point list), 6 (line strip) or 7 (triangle strip)"
    )]
    OutputTopology {
        /// The output topology field.
        value: u32,
    },
}

/// The two layouts of a header, named by its header type field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Type 1: vertex, tessellation and geometry programs.
    Vtg,
    /// Type 2: pixel programs.
    Pixel,
}

impl Display for Layout {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Vtg => "vertex, tessellation and geometry",
            Self::Pixel => "pixel",
        })
    }
}

/// A geometry program's output primitive (`OUTPUT_TOPOLOGY_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutputTopology {
    /// `POINTLIST`.
    PointList,
    /// `LINESTRIP`.
    LineStrip,
    /// `TRIANGLESTRIP`.
    TriangleStrip,
}

/// How a pixel program interpolates one input component (`GENERIC_IMAP_X_*`; NAK and Eden call
/// it `PixelImap`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PixelImap {
    /// Not read.
    #[default]
    Unused,
    /// Flat: one value for the whole primitive.
    Constant,
    /// Perspective-correct.
    Perspective,
    /// Linear in screen space.
    ScreenLinear,
}

/// Four values, one per component: x, y, z, w for an attribute or position, s, t, r, q for a
/// texture coordinate, red, green, blue, alpha for a colour, in that order in the header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Components<T> {
    /// The first component.
    pub x: T,
    /// The second component.
    pub y: T,
    /// The third component.
    pub z: T,
    /// The fourth component.
    pub w: T,
}

/// Which components are used.
pub type ComponentMask = Components<bool>;

impl ComponentMask {
    /// The mask as four bits, `x` in bit 0 (the header's order).
    pub fn bits(self) -> u8 {
        u8::from(self.x) | u8::from(self.y) << 1 | u8::from(self.z) << 2 | u8::from(self.w) << 3
    }
}

/// The fields every header has, in its first five words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors the header's one-bit fields one for one"
)]
pub struct Common {
    /// `MRT_ENABLE` (NAK sets it for every pixel program, `encode_header`).
    pub mrt_enable: bool,
    /// `KILLS_PIXELS`.
    pub kills_pixels: bool,
    /// `DOES_GLOBAL_STORE`.
    pub does_global_store: bool,
    /// `SASS_VERSION`, four bits.
    pub sass_version: u8,
    /// Bit 24: geometry pass-through (NAK, Eden).
    pub geometry_passthrough: bool,
    /// Bit 25: ISBE space sharing (NAK).
    pub isbe_space_sharing: bool,
    /// Bit 26: version 3's `DOES_LOAD_OR_STORE`.
    pub does_load_or_store: bool,
    /// Bit 27: version 3's `DOES_FP64`.
    pub does_fp64: bool,
    /// `STREAM_OUT_MASK`, four bits.
    pub stream_out_mask: u8,
    /// `SHADER_LOCAL_MEMORY_LOW_SIZE`, 24 bits.
    pub local_memory_low_size: u32,
    /// `PER_PATCH_ATTRIBUTE_COUNT`.
    pub per_patch_attribute_count: u8,
    /// `SHADER_LOCAL_MEMORY_HIGH_SIZE`, 24 bits.
    pub local_memory_high_size: u32,
    /// `THREADS_PER_INPUT_PRIMITIVE`.
    pub threads_per_input_primitive: u8,
    /// Bits 96 to 119: version 3's `SHADER_LOCAL_MEMORY_CRS_SIZE`, 24 bits.
    pub local_memory_crs_size: u32,
    /// `OUTPUT_TOPOLOGY`; `None` for the value 0.
    pub output_topology: Option<OutputTopology>,
    /// `MAX_OUTPUT_VERTEX_COUNT`, 12 bits.
    pub max_output_vertex_count: u16,
    /// `STORE_REQ_START`.
    pub store_req_start: u8,
    /// `STORE_REQ_END`.
    pub store_req_end: u8,
}

impl Common {
    /// The local memory size in bytes, as Eden composes it: the low field plus the high field
    /// shifted by 24 (`ProgramHeader::LocalMemorySize`). At most 48 bits. NAK's writer places
    /// the high field at bit 32 instead (module docs, "Conflict"). PR #4588 adds
    /// [`Self::local_memory_crs_size`] to it; that choice is the local memory pass's.
    pub fn local_memory_size(&self) -> u64 {
        // Both fields are at most 24 bits, so this is at most 48 bits.
        u64::from(self.local_memory_low_size) | u64::from(self.local_memory_high_size) << 24
    }
}

/// System values A and B of a map: the bits from the map's base through `POSITION_W`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors the header's one-bit fields one for one"
)]
pub struct SystemValuesAb {
    /// The 23 bits below `VPRS_TABLE_INDEX`, which `clc797sph.h` does not name (version 3
    /// names tessellation inputs there); kept as they are.
    pub system_values_a: u32,
    /// `VPRS_TABLE_INDEX`.
    pub vprs_table_index: bool,
    /// `PRIMITIVE_ID`.
    pub primitive_id: bool,
    /// `RT_ARRAY_INDEX`: the render target layer.
    pub rt_array_index: bool,
    /// `VIEWPORT_INDEX`.
    pub viewport_index: bool,
    /// `POINT_SIZE`.
    pub point_size: bool,
    /// `POSITION_X` to `POSITION_W`.
    pub position: ComponentMask,
}

/// System values C of a map: clip distances and the fixed-function values after them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors the header's one-bit fields one for one"
)]
pub struct SystemValuesC {
    /// `CLIP_DISTANCE0` to `CLIP_DISTANCE7`, distance `i` in bit `i`.
    pub clip_distances: u8,
    /// `POINT_SPRITE_S`.
    pub point_sprite_s: bool,
    /// `POINT_SPRITE_T`.
    pub point_sprite_t: bool,
    /// `FOG_COORDINATE`.
    pub fog_coordinate: bool,
    /// `TESSELLATION_EVALUATION_POINT_U`.
    pub tessellation_point_u: bool,
    /// `TESSELLATION_EVALUATION_POINT_V`.
    pub tessellation_point_v: bool,
    /// `INSTANCE_ID`.
    pub instance_id: bool,
    /// `VERTEX_ID`.
    pub vertex_id: bool,
}

/// System values D of a map: a flag per value in a vertex map and in the pixel header's
/// extension, an interpolation mode per value in the pixel header's input map.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SystemValuesD<T> {
    /// `VIEWPORT_MASK`.
    pub viewport_mask: T,
    /// `VIEWPORT_MASK2`.
    pub viewport_mask2: T,
    /// `BARYCENTRIC0` to `BARYCENTRIC2`.
    pub barycentric: [T; BARYCENTRICS],
    /// `BARYCENTRIC0_W` to `BARYCENTRIC2_W`.
    pub barycentric_w: [T; BARYCENTRICS],
}

/// One attribute map of a vertex, tessellation or geometry header: which values the program
/// reads (input map), writes (output map), or holds in the map `clc797sph.h` calls VMAP.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VtgMap {
    /// System values A and B.
    pub system_values_ab: SystemValuesAb,
    /// Each generic attribute's components (`GENERIC_IMAP_X(i)` to `_W(i)`).
    pub generic: [ComponentMask; GENERIC_ATTRIBUTES],
    /// Front diffuse, front specular, back diffuse and back specular colours.
    pub color: [ComponentMask; VTG_COLORS],
    /// System values C.
    pub system_values_c: SystemValuesC,
    /// Each fixed-function texture coordinate's components (`TEX_IMAP_S(i)` to `_Q(i)`).
    pub texture: [ComponentMask; TEXTURE_COORDINATES],
    /// System values D.
    pub system_values_d: SystemValuesD<bool>,
}

/// The stage of a type 1 header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VtgStage {
    /// `SHADER_TYPE_VERTEX`.
    Vertex,
    /// `SHADER_TYPE_TESSELLATION_INIT`: tessellation control.
    TessellationInit,
    /// `SHADER_TYPE_TESSELLATION`: tessellation evaluation.
    Tessellation,
    /// `SHADER_TYPE_GEOMETRY`.
    Geometry,
}

/// The rest of a type 1 header: its stage and three maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VtgIo {
    /// The stage.
    pub stage: VtgStage,
    /// What the program reads (`IMAP_*`).
    pub input: VtgMap,
    /// What the program writes (`OMAP_*`).
    pub output: VtgMap,
    /// The map `clc797sph.h` calls VMAP (`VMAP_*`). No reference states what it means; it is
    /// read and kept.
    pub vmap: VtgMap,
}

/// The rest of a type 2 (pixel) header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors the header's one-bit fields one for one"
)]
pub struct PixelIo {
    /// System values A and B of the input map.
    pub system_values_ab: SystemValuesAb,
    /// How each component of each generic attribute is interpolated.
    pub generic: [Components<PixelImap>; GENERIC_ATTRIBUTES],
    /// How the diffuse and specular colours are interpolated.
    pub color: [Components<PixelImap>; PS_COLORS],
    /// System values C of the input map.
    pub system_values_c: SystemValuesC,
    /// How each fixed-function texture coordinate is interpolated.
    pub texture: [Components<PixelImap>; TEXTURE_COORDINATES],
    /// How system values D are interpolated.
    pub system_values_d: SystemValuesD<PixelImap>,
    /// Which components of each colour target the program writes (`OMAP_RED(i)` to
    /// `OMAP_ALPHA(i)`).
    pub targets: [ComponentMask; COLOR_TARGETS],
    /// `OMAP_SAMPLE_MASK`: the program writes the sample mask.
    pub writes_sample_mask: bool,
    /// `OMAP_DEPTH`: the program writes depth.
    pub writes_depth: bool,
    /// `READS_INTERLOCK_TICKET`.
    pub reads_interlock_ticket: bool,
    /// `USES_INNER_COVERAGE`.
    pub uses_inner_coverage: bool,
    /// `IMAP_SYSTEM_VALUES_A_EXT`, 24 bits, bit `i` for index `i`.
    pub system_values_a_ext: u32,
    /// `IMAP_SYSTEM_VALUES_B_EXT`, bit `i` for index `i`.
    pub system_values_b_ext: u8,
    /// `GENERIC_IMAP_X_EXT(i)` to `_W_EXT(i)` (NAK writes its per-vertex inputs here).
    pub generic_ext: [ComponentMask; GENERIC_ATTRIBUTES],
    /// `IMAP_COLOR_DIFFUSE_*_EXT` and `IMAP_COLOR_SPECULAR_*_EXT`.
    pub color_ext: [ComponentMask; PS_COLORS],
    /// `IMAP_SYSTEM_VALUES_C_EXT`, bit `i` for index `i`.
    pub system_values_c_ext: u16,
    /// `TEX_IMAP_S_EXT(i)` to `_Q_EXT(i)`.
    pub texture_ext: [ComponentMask; TEXTURE_COORDINATES],
    /// `IMAP_VIEWPORT_MASK_EXT` to `IMAP_BARYCENTRIC2_W_EXT`.
    pub system_values_d_ext: SystemValuesD<bool>,
}

impl PixelIo {
    /// The colour components this program writes, as the translator takes them.
    pub fn color_outputs(&self) -> ColorOutputs {
        ColorOutputs {
            targets: self.targets.map(ComponentMask::bits),
        }
    }
}

/// What follows the common fields: one of the two layouts.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Io {
    /// A type 1 header: a vertex, tessellation or geometry program.
    Vtg(Box<VtgIo>),
    /// A type 2 header: a pixel program.
    Pixel(Box<PixelIo>),
}

/// A parsed SPH v4 header. Every combination of field values is a header some program could
/// carry; the header type and shader type fields are not stored, as [`Io`] determines both.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProgramHeader {
    /// The fields both layouts have.
    pub common: Common,
    /// The stage and its maps.
    pub io: Io,
}

impl ProgramHeader {
    /// The only header version this parser reads (`clc797sph.h` is "Version 4"; NAK writes
    /// version 4 for SM 7.3 and later, `ShaderProgramHeader::new`).
    pub const VERSION: u32 = 4;

    /// Reads the header in front of a graphics program and returns it with the bytes after it
    /// (the program's instructions). Bytes past the header are not looked at.
    pub fn from_program(program: &[u8]) -> Result<(Self, &[u8]), HeaderError> {
        let Some((header, code)) = program.split_first_chunk::<HEADER_BYTES>() else {
            return Err(HeaderError::Short {
                len: program.len(),
                needed: HEADER_BYTES,
            });
        };
        let (chunks, _) = header.as_chunks::<4>();
        let mut words = [0; HEADER_WORDS];
        for (word, chunk) in words.iter_mut().zip(chunks) {
            *word = u32::from_le_bytes(*chunk);
        }
        Ok((Self::from_words(&words)?, code))
    }

    /// Parses the header's 32 words, in the order of checks the module docs give.
    pub fn from_words(words: &[u32; HEADER_WORDS]) -> Result<Self, HeaderError> {
        let read = Reader { words };
        let version = read.field(VERSION);
        if version != Self::VERSION {
            return Err(HeaderError::Version {
                version,
                supported: Self::VERSION,
            });
        }
        let layout = match read.field(SPH_TYPE) {
            SPH_TYPE_VTG => Layout::Vtg,
            SPH_TYPE_PS => Layout::Pixel,
            value => return Err(HeaderError::SphType { value }),
        };
        let stage = match read.field(SHADER_TYPE) {
            SHADER_TYPE_VERTEX => GuestStage::Vertex,
            SHADER_TYPE_TESSELLATION_INIT => GuestStage::TessellationInit,
            SHADER_TYPE_TESSELLATION => GuestStage::Tessellation,
            SHADER_TYPE_GEOMETRY => GuestStage::Geometry,
            SHADER_TYPE_PIXEL => GuestStage::Pixel,
            value => return Err(HeaderError::ShaderType { value }),
        };
        let vtg_stage = match (layout, stage) {
            (Layout::Vtg, GuestStage::Vertex) => Some(VtgStage::Vertex),
            (Layout::Vtg, GuestStage::TessellationInit) => Some(VtgStage::TessellationInit),
            (Layout::Vtg, GuestStage::Tessellation) => Some(VtgStage::Tessellation),
            (Layout::Vtg, GuestStage::Geometry) => Some(VtgStage::Geometry),
            (Layout::Pixel, GuestStage::Pixel) => None,
            _ => return Err(HeaderError::TypeMismatch { layout, stage }),
        };
        let common = read.common()?;
        let io = match vtg_stage {
            Some(stage) => Io::Vtg(Box::new(VtgIo {
                stage,
                input: read.vtg_map(IMAP),
                output: read.vtg_map(OMAP),
                vmap: read.vtg_map(VMAP),
            })),
            None => Io::Pixel(Box::new(read.pixel_io())),
        };
        Ok(Self { common, io })
    }

    /// The stage the header's shader type names: [`GuestStage::Vertex`],
    /// [`GuestStage::TessellationInit`], [`GuestStage::Tessellation`], [`GuestStage::Geometry`]
    /// or [`GuestStage::Pixel`] (never a compute or cull-before-fetch stage: the header has no
    /// such shader type).
    pub fn stage(&self) -> GuestStage {
        match &self.io {
            Io::Vtg(io) => io.stage.guest_stage(),
            Io::Pixel(_) => GuestStage::Pixel,
        }
    }

    /// The header's layout.
    pub fn layout(&self) -> Layout {
        match self.io {
            Io::Vtg(_) => Layout::Vtg,
            Io::Pixel(_) => Layout::Pixel,
        }
    }
}

impl VtgStage {
    /// The guest stage this is.
    pub const fn guest_stage(self) -> GuestStage {
        match self {
            Self::Vertex => GuestStage::Vertex,
            Self::TessellationInit => GuestStage::TessellationInit,
            Self::Tessellation => GuestStage::Tessellation,
            Self::Geometry => GuestStage::Geometry,
        }
    }
}

/// A value of `width` bits read from the header and kept in the low bits.
pub(crate) trait Cell: Copy + Default {
    /// How many bits one cell takes.
    const WIDTH: u32;
    /// The cell holding these bits; only the low [`Cell::WIDTH`] bits are given.
    fn from_bits(bits: u32) -> Self;
    /// The cell's bits, for the test encoder.
    #[cfg(feature = "test-support")]
    fn to_bits(self) -> u32;
}

impl Cell for bool {
    const WIDTH: u32 = 1;
    fn from_bits(bits: u32) -> Self {
        bits != 0
    }
    #[cfg(feature = "test-support")]
    fn to_bits(self) -> u32 {
        u32::from(self)
    }
}

impl Cell for PixelImap {
    const WIDTH: u32 = 2;
    fn from_bits(bits: u32) -> Self {
        match bits {
            IMAP_UNUSED => Self::Unused,
            IMAP_CONSTANT => Self::Constant,
            IMAP_PERSPECTIVE => Self::Perspective,
            // Only `IMAP_SCREEN_LINEAR` is left in two bits.
            _ => Self::ScreenLinear,
        }
    }
    #[cfg(feature = "test-support")]
    fn to_bits(self) -> u32 {
        match self {
            Self::Unused => IMAP_UNUSED,
            Self::Constant => IMAP_CONSTANT,
            Self::Perspective => IMAP_PERSPECTIVE,
            Self::ScreenLinear => layout::IMAP_SCREEN_LINEAR,
        }
    }
}

/// Where cell `index` of a run of `T` cells starting at bit `start` is.
pub(crate) const fn cell<T: Cell>(start: u32, index: u32) -> HeaderField {
    HeaderField::new(
        start.saturating_add(index.saturating_mul(T::WIDTH)),
        T::WIDTH,
    )
}

/// Where entry `index` of a run of four-cell entries starting at bit `start` begins.
pub(crate) const fn entry<T: Cell>(start: u32, index: u32) -> u32 {
    start.saturating_add(index.saturating_mul(COMPONENTS).saturating_mul(T::WIDTH))
}

/// Narrows a field of at most eight bits.
fn byte(value: u32) -> u8 {
    // Cannot fail: every caller's field is at most eight bits wide.
    u8::try_from(value).unwrap_or(u8::MAX)
}

/// The header's words, read field by field.
struct Reader<'a> {
    words: &'a [u32; HEADER_WORDS],
}

impl Reader<'_> {
    fn field(&self, field: HeaderField) -> u32 {
        field.read(self.words)
    }

    fn flag(&self, field: HeaderField) -> bool {
        self.field(field) != 0
    }

    fn cell<T: Cell>(&self, start: u32, index: u32) -> T {
        T::from_bits(self.field(cell::<T>(start, index)))
    }

    fn components<T: Cell>(&self, start: u32) -> Components<T> {
        Components {
            x: self.cell(start, 0),
            y: self.cell(start, 1),
            z: self.cell(start, 2),
            w: self.cell(start, 3),
        }
    }

    fn run<T: Cell, const N: usize>(&self, start: u32) -> [Components<T>; N] {
        let mut run = [Components::<T>::default(); N];
        for (index, entry_value) in (0_u32..).zip(run.iter_mut()) {
            *entry_value = self.components(entry::<T>(start, index));
        }
        run
    }

    fn common(&self) -> Result<Common, HeaderError> {
        let output_topology = match self.field(OUTPUT_TOPOLOGY) {
            OUTPUT_TOPOLOGY_NONE => None,
            OUTPUT_TOPOLOGY_POINTLIST => Some(OutputTopology::PointList),
            OUTPUT_TOPOLOGY_LINESTRIP => Some(OutputTopology::LineStrip),
            OUTPUT_TOPOLOGY_TRIANGLESTRIP => Some(OutputTopology::TriangleStrip),
            value => return Err(HeaderError::OutputTopology { value }),
        };
        Ok(Common {
            mrt_enable: self.flag(MRT_ENABLE),
            kills_pixels: self.flag(KILLS_PIXELS),
            does_global_store: self.flag(DOES_GLOBAL_STORE),
            sass_version: byte(self.field(SASS_VERSION)),
            geometry_passthrough: self.flag(GEOMETRY_PASSTHROUGH),
            isbe_space_sharing: self.flag(ISBE_SPACE_SHARING),
            does_load_or_store: self.flag(DOES_LOAD_OR_STORE),
            does_fp64: self.flag(DOES_FP64),
            stream_out_mask: byte(self.field(STREAM_OUT_MASK)),
            local_memory_low_size: self.field(LOCAL_MEMORY_LOW_SIZE),
            per_patch_attribute_count: byte(self.field(PER_PATCH_ATTRIBUTE_COUNT)),
            local_memory_high_size: self.field(LOCAL_MEMORY_HIGH_SIZE),
            threads_per_input_primitive: byte(self.field(THREADS_PER_INPUT_PRIMITIVE)),
            local_memory_crs_size: self.field(LOCAL_MEMORY_CRS_SIZE),
            output_topology,
            // Cannot fail: the field is 12 bits wide.
            max_output_vertex_count: u16::try_from(self.field(MAX_OUTPUT_VERTEX_COUNT))
                .unwrap_or(u16::MAX),
            store_req_start: byte(self.field(STORE_REQ_START)),
            store_req_end: byte(self.field(STORE_REQ_END)),
        })
    }

    fn system_values_ab(&self, base: u32) -> SystemValuesAb {
        SystemValuesAb {
            system_values_a: self.field(SYSTEM_VALUES_A.at(base)),
            vprs_table_index: self.flag(VPRS_TABLE_INDEX.at(base)),
            primitive_id: self.flag(PRIMITIVE_ID.at(base)),
            rt_array_index: self.flag(RT_ARRAY_INDEX.at(base)),
            viewport_index: self.flag(VIEWPORT_INDEX.at(base)),
            point_size: self.flag(POINT_SIZE.at(base)),
            position: self.components(base.saturating_add(POSITION)),
        }
    }

    fn system_values_c(&self, start: u32) -> SystemValuesC {
        SystemValuesC {
            clip_distances: byte(self.field(CLIP_DISTANCES.at(start))),
            point_sprite_s: self.flag(POINT_SPRITE_S.at(start)),
            point_sprite_t: self.flag(POINT_SPRITE_T.at(start)),
            fog_coordinate: self.flag(FOG_COORDINATE.at(start)),
            tessellation_point_u: self.flag(TESSELLATION_POINT_U.at(start)),
            tessellation_point_v: self.flag(TESSELLATION_POINT_V.at(start)),
            instance_id: self.flag(INSTANCE_ID.at(start)),
            vertex_id: self.flag(VERTEX_ID.at(start)),
        }
    }

    fn system_values_d<T: Cell>(&self, start: u32) -> SystemValuesD<T> {
        let barycentrics = |first: u32| -> [T; BARYCENTRICS] {
            let mut values = [T::default(); BARYCENTRICS];
            for (index, value) in (first..).zip(values.iter_mut()) {
                *value = self.cell(start, index);
            }
            values
        };
        SystemValuesD {
            viewport_mask: self.cell(start, VIEWPORT_MASK_CELL),
            viewport_mask2: self.cell(start, VIEWPORT_MASK2_CELL),
            barycentric: barycentrics(BARYCENTRIC_CELL),
            barycentric_w: barycentrics(BARYCENTRIC_W_CELL),
        }
    }

    fn vtg_map(&self, base: u32) -> VtgMap {
        VtgMap {
            system_values_ab: self.system_values_ab(base),
            generic: self.run::<bool, GENERIC_ATTRIBUTES>(base.saturating_add(VTG_GENERIC)),
            color: self.run::<bool, VTG_COLORS>(base.saturating_add(VTG_COLOR)),
            system_values_c: self.system_values_c(base.saturating_add(VTG_SYSTEM_VALUES_C)),
            texture: self.run::<bool, TEXTURE_COORDINATES>(base.saturating_add(VTG_TEXTURE)),
            system_values_d: self.system_values_d(base.saturating_add(VTG_SYSTEM_VALUES_D)),
        }
    }

    fn pixel_io(&self) -> PixelIo {
        PixelIo {
            system_values_ab: self.system_values_ab(IMAP),
            generic: self.run::<PixelImap, GENERIC_ATTRIBUTES>(PS_GENERIC),
            color: self.run::<PixelImap, PS_COLORS>(PS_COLOR),
            system_values_c: self.system_values_c(PS_SYSTEM_VALUES_C),
            texture: self.run::<PixelImap, TEXTURE_COORDINATES>(PS_TEXTURE),
            system_values_d: self.system_values_d(PS_SYSTEM_VALUES_D),
            targets: self.run::<bool, COLOR_TARGETS>(PS_TARGETS),
            writes_sample_mask: self.flag(OMAP_SAMPLE_MASK),
            writes_depth: self.flag(OMAP_DEPTH),
            reads_interlock_ticket: self.flag(READS_INTERLOCK_TICKET),
            uses_inner_coverage: self.flag(USES_INNER_COVERAGE),
            system_values_a_ext: self.field(SYSTEM_VALUES_A_EXT),
            system_values_b_ext: byte(self.field(SYSTEM_VALUES_B_EXT)),
            generic_ext: self.run::<bool, GENERIC_ATTRIBUTES>(GENERIC_EXT),
            color_ext: self.run::<bool, PS_COLORS>(COLOR_EXT),
            // Cannot fail: the field is 16 bits wide.
            system_values_c_ext: u16::try_from(self.field(SYSTEM_VALUES_C_EXT)).unwrap_or(u16::MAX),
            texture_ext: self.run::<bool, TEXTURE_COORDINATES>(TEXTURE_EXT),
            system_values_d_ext: self.system_values_d(SYSTEM_VALUES_D_EXT),
        }
    }
}
