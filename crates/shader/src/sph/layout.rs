//! Bit positions of the SPH v4 header, written once for the parser and the test encoder.
//!
//! Ported from `mesa-nouveau/src/nouveau/headers/nvidia/classes/clc797sph.h` @ 52238762 (MIT,
//! Copyright (c) 2003-2021 NVIDIA CORPORATION & AFFILIATES; the licence text is the MIT text in
//! `LICENSES/`). Each constant names the `NVC797_SPHV4T1_*` or `NVC797_SPHV4T2_*` define it
//! comes from; the header writes a field as `MW(high:low)`, an inclusive bit range over the
//! header read as one 1024-bit little-endian number (bit `n` is bit `n % 32` of word `n / 32`).
//! Five common fields that `clc797sph.h` leaves reserved come from the version 3 header
//! (`mesa-nouveau/src/nouveau/headers/nvidia/classes/cla097sph.h`, `NVA097_SPHV3T1_*`) or from
//! NAK's writer (`mesa-nouveau/src/nouveau/compiler/nak/sph.rs`), which writes them for SM 7.3
//! and later too; each says so.
//!
//! The three attribute maps of a vertex, tessellation or geometry header (input, output and the
//! map `clc797sph.h` calls VMAP) share one layout at three bases, so their fields are written
//! relative to the map's base. Runs of components (an attribute's four, a colour's four) are
//! back to back: component `c` of entry `e` of a run of cells `w` bits wide starting at `s` is
//! at `s + (e * 4 + c) * w`, as every `(i)` define of the header lays them out.

use super::HEADER_WORDS;

/// How many bits the header has: [`HEADER_WORDS`] 32-bit words.
#[cfg(test)]
pub(crate) const HEADER_BITS: u32 = 1024;

/// A field of the header: `width` bits starting at bit `start` (`MW(start + width - 1:start)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct HeaderField {
    /// The first bit.
    pub(crate) start: u32,
    /// How many bits; at most 32.
    pub(crate) width: u32,
}

impl HeaderField {
    /// The field of `width` bits starting at `start`.
    pub(crate) const fn new(start: u32, width: u32) -> Self {
        Self { start, width }
    }

    /// A one-bit field.
    pub(crate) const fn bit(start: u32) -> Self {
        Self::new(start, 1)
    }

    /// This field moved up by `base` bits: a map field placed at its map's base.
    pub(crate) const fn at(self, base: u32) -> Self {
        Self::new(self.start.saturating_add(base), self.width)
    }

    /// The first bit past the field.
    #[cfg(test)]
    pub(crate) const fn end(self) -> u32 {
        self.start.saturating_add(self.width)
    }

    /// The field's value in its low bits: the ones of a `width`-bit number (32 bits for a width
    /// of 32 or more).
    pub(crate) const fn ones(self) -> u32 {
        match u32::MAX.checked_shr(32_u32.saturating_sub(self.width)) {
            Some(ones) if self.width != 0 => ones,
            _ => 0,
        }
    }

    /// The field's value in `words`. A field may cross a word boundary; bits past the header
    /// read as zero.
    pub(crate) fn read(self, words: &[u32; HEADER_WORDS]) -> u32 {
        let index = usize::try_from(self.start / 32).unwrap_or(usize::MAX);
        let shift = self.start % 32;
        let low = words
            .get(index)
            .copied()
            .unwrap_or(0)
            .checked_shr(shift)
            .unwrap_or(0);
        let high = match shift {
            0 => 0,
            _ => words
                .get(index.saturating_add(1))
                .copied()
                .unwrap_or(0)
                .checked_shl(32 - shift)
                .unwrap_or(0),
        };
        (low | high) & self.ones()
    }
}

// The common fields, the same in both header types (`NVC797_SPHV4T1_*` and `NVC797_SPHV4T2_*`
// give each the same range).

/// `SPH_TYPE`, `MW(4:0)`: which layout the rest of the header has.
pub(crate) const SPH_TYPE: HeaderField = HeaderField::new(0, 5);
/// `SPH_TYPE_TYPE_01_VTG`: a vertex, tessellation or geometry header.
pub(crate) const SPH_TYPE_VTG: u32 = 1;
/// `SPH_TYPE_TYPE_02_PS`: a pixel header.
pub(crate) const SPH_TYPE_PS: u32 = 2;
/// `VERSION`, `MW(9:5)`.
pub(crate) const VERSION: HeaderField = HeaderField::new(5, 5);
/// `SHADER_TYPE`, `MW(13:10)`.
pub(crate) const SHADER_TYPE: HeaderField = HeaderField::new(10, 4);
/// `SHADER_TYPE_VERTEX`.
pub(crate) const SHADER_TYPE_VERTEX: u32 = 1;
/// `SHADER_TYPE_TESSELLATION_INIT`.
pub(crate) const SHADER_TYPE_TESSELLATION_INIT: u32 = 2;
/// `SHADER_TYPE_TESSELLATION`.
pub(crate) const SHADER_TYPE_TESSELLATION: u32 = 3;
/// `SHADER_TYPE_GEOMETRY`.
pub(crate) const SHADER_TYPE_GEOMETRY: u32 = 4;
/// `SHADER_TYPE_PIXEL`.
pub(crate) const SHADER_TYPE_PIXEL: u32 = 5;
/// `MRT_ENABLE`, `MW(14:14)`.
pub(crate) const MRT_ENABLE: HeaderField = HeaderField::bit(14);
/// `KILLS_PIXELS`, `MW(15:15)`.
pub(crate) const KILLS_PIXELS: HeaderField = HeaderField::bit(15);
/// `DOES_GLOBAL_STORE`, `MW(16:16)`.
pub(crate) const DOES_GLOBAL_STORE: HeaderField = HeaderField::bit(16);
/// `SASS_VERSION`, `MW(20:17)`.
pub(crate) const SASS_VERSION: HeaderField = HeaderField::new(17, 4);
/// Bit 24, unnamed in `clc797sph.h`: geometry pass-through (NAK `set_gs_passthrough_enable`:
/// `set_bit(24, ..)`; Eden `ProgramHeader::common0.geometry_passthrough`, `BitField<24, 1>`).
pub(crate) const GEOMETRY_PASSTHROUGH: HeaderField = HeaderField::bit(24);
/// Bit 25, unnamed in `clc797sph.h`: ISBE space sharing (NAK `set_isbe_space_sharing_enable`:
/// `set_bit(25, ..)`).
pub(crate) const ISBE_SPACE_SHARING: HeaderField = HeaderField::bit(25);
/// Bit 26, half of `RESERVED_COMMON_B` (`MW(27:26)`) in `clc797sph.h`: version 3's
/// `NVA097_SPHV3T1_DOES_LOAD_OR_STORE`, `MW(26:26)`, which NAK writes on every SM
/// (`set_does_load_or_store`).
pub(crate) const DOES_LOAD_OR_STORE: HeaderField = HeaderField::bit(26);
/// Bit 27, the other half of `RESERVED_COMMON_B`: version 3's `NVA097_SPHV3T1_DOES_FP64`,
/// `MW(27:27)` (NAK `set_does_fp64`).
pub(crate) const DOES_FP64: HeaderField = HeaderField::bit(27);
/// `STREAM_OUT_MASK`, `MW(31:28)`.
pub(crate) const STREAM_OUT_MASK: HeaderField = HeaderField::new(28, 4);
/// `SHADER_LOCAL_MEMORY_LOW_SIZE`, `MW(55:32)`.
pub(crate) const LOCAL_MEMORY_LOW_SIZE: HeaderField = HeaderField::new(32, 24);
/// `PER_PATCH_ATTRIBUTE_COUNT`, `MW(63:56)`.
pub(crate) const PER_PATCH_ATTRIBUTE_COUNT: HeaderField = HeaderField::new(56, 8);
/// `SHADER_LOCAL_MEMORY_HIGH_SIZE`, `MW(87:64)`.
pub(crate) const LOCAL_MEMORY_HIGH_SIZE: HeaderField = HeaderField::new(64, 24);
/// `THREADS_PER_INPUT_PRIMITIVE`, `MW(95:88)`.
pub(crate) const THREADS_PER_INPUT_PRIMITIVE: HeaderField = HeaderField::new(88, 8);
/// `RESERVED_COMMON_C`, `MW(119:96)`, in `clc797sph.h`: version 3's
/// `NVA097_SPHV3T1_SHADER_LOCAL_MEMORY_CRS_SIZE`, which NAK writes on every SM
/// (`set_shader_local_memory_crs_size`) and PR #4588 adds to the local memory size
/// (`eden-pr4588/src/video_core/shader_environment.cpp`, `GraphicsEnvironment`).
pub(crate) const LOCAL_MEMORY_CRS_SIZE: HeaderField = HeaderField::new(96, 24);
/// `OUTPUT_TOPOLOGY`, `MW(123:120)`.
pub(crate) const OUTPUT_TOPOLOGY: HeaderField = HeaderField::new(120, 4);
/// No output topology: the value a header that is not a geometry header carries. Not named by
/// `clc797sph.h`; NAK's header starts as zeros and only a geometry program sets the field
/// (`ShaderProgramHeader::new`, `encode_header`).
pub(crate) const OUTPUT_TOPOLOGY_NONE: u32 = 0;
/// `OUTPUT_TOPOLOGY_POINTLIST`.
pub(crate) const OUTPUT_TOPOLOGY_POINTLIST: u32 = 1;
/// `OUTPUT_TOPOLOGY_LINESTRIP`.
pub(crate) const OUTPUT_TOPOLOGY_LINESTRIP: u32 = 6;
/// `OUTPUT_TOPOLOGY_TRIANGLESTRIP`.
pub(crate) const OUTPUT_TOPOLOGY_TRIANGLESTRIP: u32 = 7;
/// `MAX_OUTPUT_VERTEX_COUNT`, `MW(139:128)`.
pub(crate) const MAX_OUTPUT_VERTEX_COUNT: HeaderField = HeaderField::new(128, 12);
/// `STORE_REQ_START`, `MW(147:140)`.
pub(crate) const STORE_REQ_START: HeaderField = HeaderField::new(140, 8);
/// `STORE_REQ_END`, `MW(159:152)`.
pub(crate) const STORE_REQ_END: HeaderField = HeaderField::new(152, 8);

// The attribute maps. Both header types start their input map at bit 160; a vertex,
// tessellation or geometry header has two more.

/// Where the input map starts: the system values below `IMAP_VPRS_TABLE_INDEX` (`MW(183:183)`)
/// that NAK writes as one 32-bit value from bit 160 (`imap_system_values_ab`: `160..192`).
pub(crate) const IMAP: u32 = 160;
/// Where a vertex, tessellation or geometry header's output map starts (`OMAP_VPRS_TABLE_INDEX`
/// is `MW(423:423)`, 23 bits in; NAK `omap_system_values_ab`: `400..432`).
pub(crate) const OMAP: u32 = 400;
/// Where the map `clc797sph.h` calls VMAP starts (`VMAP_VPRS_TABLE_INDEX`, `MW(663:663)`).
pub(crate) const VMAP: u32 = 640;

/// System values A: the 23 bits below `VPRS_TABLE_INDEX`, which `clc797sph.h` does not name
/// (version 3 names tessellation inputs there, `NVA097_SPHV3T1_IMAP_TESSELLATION_LOD_LEFT`,
/// `MW(164:164)`, and on). Relative to the map's base.
pub(crate) const SYSTEM_VALUES_A: HeaderField = HeaderField::new(0, 23);
/// `IMAP_VPRS_TABLE_INDEX`, `MW(183:183)` (and `OMAP_`, `VMAP_` at 423, 663).
pub(crate) const VPRS_TABLE_INDEX: HeaderField = HeaderField::bit(23);
/// `IMAP_PRIMITIVE_ID`, `MW(184:184)`.
pub(crate) const PRIMITIVE_ID: HeaderField = HeaderField::bit(24);
/// `IMAP_RT_ARRAY_INDEX`, `MW(185:185)`.
pub(crate) const RT_ARRAY_INDEX: HeaderField = HeaderField::bit(25);
/// `IMAP_VIEWPORT_INDEX`, `MW(186:186)`.
pub(crate) const VIEWPORT_INDEX: HeaderField = HeaderField::bit(26);
/// `IMAP_POINT_SIZE`, `MW(187:187)`.
pub(crate) const POINT_SIZE: HeaderField = HeaderField::bit(27);
/// `IMAP_POSITION_X` to `IMAP_POSITION_W`, `MW(188:188)` to `MW(191:191)`: four one-bit cells.
pub(crate) const POSITION: u32 = 28;
/// How many bits system values A and B take from the map's base: through `IMAP_POSITION_W`.
#[cfg(test)]
pub(crate) const SYSTEM_VALUES_AB_BITS: u32 = 32;

/// How many generic attributes a map has: `GENERIC_IMAP_X(i)` runs from bit 192 to the colours
/// at 320 in a vertex header (four bits each) and to 448 in a pixel header (eight bits each).
/// Eden agrees (`ProgramHeader::ps.imap_generic_vector[32]`).
pub(crate) const GENERIC_ATTRIBUTES: usize = 32;
/// How many fixed-function texture coordinates a map has: `TEX_IMAP_S(i)` runs from bit 352 to
/// `IMAP_VIEWPORT_MASK` at 392 (four bits each). Eden agrees (`ImapFixedFncTexture[10]`).
pub(crate) const TEXTURE_COORDINATES: usize = 10;

/// `GENERIC_IMAP_X(0)`, `MW(192:192)`, relative to the input map's base: a vertex map's
/// generic attributes, four one-bit cells each.
pub(crate) const VTG_GENERIC: u32 = 32;
/// `IMAP_COLOR_FRONT_DIFFUSE_RED`, `MW(320:320)`: front diffuse, front specular, back diffuse,
/// back specular, four one-bit cells (red, green, blue, alpha) each.
pub(crate) const VTG_COLOR: u32 = 160;
/// How many colours a vertex map has.
pub(crate) const VTG_COLORS: usize = 4;
/// `IMAP_CLIP_DISTANCE0`, `MW(336:336)`: system values C.
pub(crate) const VTG_SYSTEM_VALUES_C: u32 = 176;
/// `TEX_IMAP_S(0)`, `MW(352:352)`.
pub(crate) const VTG_TEXTURE: u32 = 192;
/// `IMAP_VIEWPORT_MASK`, `MW(392:392)`: system values D, eight one-bit cells.
pub(crate) const VTG_SYSTEM_VALUES_D: u32 = 232;
/// How many bits one vertex map takes: from its base to the next map (400 - 160).
#[cfg(test)]
pub(crate) const VTG_MAP_BITS: u32 = 240;
/// `PADDING(0)`, `MW(880:880)`: a vertex header's padding, to the end of the header.
#[cfg(test)]
pub(crate) const VTG_PADDING: u32 = 880;

// System values C, relative to their start (`IMAP_CLIP_DISTANCE0`: bit 336 in a vertex
// header, 464 in a pixel header).

/// `IMAP_CLIP_DISTANCE0` to `IMAP_CLIP_DISTANCE7`, `MW(336:336)` to `MW(343:343)`.
pub(crate) const CLIP_DISTANCES: HeaderField = HeaderField::new(0, 8);
/// `IMAP_POINT_SPRITE_S`, `MW(344:344)`.
pub(crate) const POINT_SPRITE_S: HeaderField = HeaderField::bit(8);
/// `IMAP_POINT_SPRITE_T`, `MW(345:345)`.
pub(crate) const POINT_SPRITE_T: HeaderField = HeaderField::bit(9);
/// `IMAP_FOG_COORDINATE`, `MW(346:346)`. Bit 11 (347) is unnamed.
pub(crate) const FOG_COORDINATE: HeaderField = HeaderField::bit(10);
/// `IMAP_TESSELLATION_EVALUATION_POINT_U`, `MW(348:348)`.
pub(crate) const TESSELLATION_POINT_U: HeaderField = HeaderField::bit(12);
/// `IMAP_TESSELLATION_EVALUATION_POINT_V`, `MW(349:349)`.
pub(crate) const TESSELLATION_POINT_V: HeaderField = HeaderField::bit(13);
/// `IMAP_INSTANCE_ID`, `MW(350:350)`.
pub(crate) const INSTANCE_ID: HeaderField = HeaderField::bit(14);
/// `IMAP_VERTEX_ID`, `MW(351:351)`.
pub(crate) const VERTEX_ID: HeaderField = HeaderField::bit(15);

// System values D, as cell indices from their start (`IMAP_VIEWPORT_MASK`): one bit per cell in
// a vertex map and in the pixel header's extension, two in the pixel header's input map.

/// `IMAP_VIEWPORT_MASK`.
pub(crate) const VIEWPORT_MASK_CELL: u32 = 0;
/// `IMAP_VIEWPORT_MASK2`.
pub(crate) const VIEWPORT_MASK2_CELL: u32 = 1;
/// `IMAP_BARYCENTRIC0` to `IMAP_BARYCENTRIC2`.
pub(crate) const BARYCENTRIC_CELL: u32 = 2;
/// `IMAP_BARYCENTRIC0_W` to `IMAP_BARYCENTRIC2_W`.
pub(crate) const BARYCENTRIC_W_CELL: u32 = 5;
/// How many barycentric cells of each kind there are.
pub(crate) const BARYCENTRICS: usize = 3;

// The pixel header (`NVC797_SPHV4T2_*`), absolute positions.

/// `GENERIC_IMAP_X(0)`, `MW(193:192)`: two-bit interpolation cells.
pub(crate) const PS_GENERIC: u32 = 192;
/// `IMAP_COLOR_DIFFUSE_RED`, `MW(449:448)`: diffuse, then specular.
pub(crate) const PS_COLOR: u32 = 448;
/// How many colours a pixel header's input map has.
pub(crate) const PS_COLORS: usize = 2;
/// `IMAP_CLIP_DISTANCE0`, `MW(464:464)`: system values C. Bit 475 is unnamed.
pub(crate) const PS_SYSTEM_VALUES_C: u32 = 464;
/// `TEX_IMAP_S(0)`, `MW(481:480)`.
pub(crate) const PS_TEXTURE: u32 = 480;
/// `IMAP_VIEWPORT_MASK`, `MW(561:560)`: system values D, two-bit cells.
pub(crate) const PS_SYSTEM_VALUES_D: u32 = 560;
/// `OMAP_RED(0)`, `MW(576:576)`: the colour targets, four one-bit cells (red, green, blue,
/// alpha) each, [`crate::translate::COLOR_TARGETS`] of them (to `OMAP_SAMPLE_MASK` at 608).
pub(crate) const PS_TARGETS: u32 = 576;
/// `OMAP_SAMPLE_MASK`, `MW(608:608)`.
pub(crate) const OMAP_SAMPLE_MASK: HeaderField = HeaderField::bit(608);
/// `OMAP_DEPTH`, `MW(609:609)`.
pub(crate) const OMAP_DEPTH: HeaderField = HeaderField::bit(609);
/// `READS_INTERLOCK_TICKET`, `MW(610:610)`.
pub(crate) const READS_INTERLOCK_TICKET: HeaderField = HeaderField::bit(610);
/// `USES_INNER_COVERAGE`, `MW(611:611)`. Bits 612 to 639 are unnamed.
pub(crate) const USES_INNER_COVERAGE: HeaderField = HeaderField::bit(611);
/// `IMAP_SYSTEM_VALUES_A_EXT(i)`, `MW(640:640)` and the 23 bits after it.
pub(crate) const SYSTEM_VALUES_A_EXT: HeaderField = HeaderField::new(640, 24);
/// `IMAP_SYSTEM_VALUES_B_EXT(i)`, `MW(664:664)` and the 7 bits after it.
pub(crate) const SYSTEM_VALUES_B_EXT: HeaderField = HeaderField::new(664, 8);
/// `GENERIC_IMAP_X_EXT(0)`, `MW(672:672)`: one-bit cells (NAK writes its per-vertex inputs
/// here, `set_pervertex_imap_vector`: `672..800`).
pub(crate) const GENERIC_EXT: u32 = 672;
/// `IMAP_COLOR_DIFFUSE_RED_EXT`, `MW(800:800)`: diffuse, then specular.
pub(crate) const COLOR_EXT: u32 = 800;
/// `IMAP_SYSTEM_VALUES_C_EXT(i)`, `MW(808:808)` and the 15 bits after it.
pub(crate) const SYSTEM_VALUES_C_EXT: HeaderField = HeaderField::new(808, 16);
/// `TEX_IMAP_S_EXT(0)`, `MW(824:824)`.
pub(crate) const TEXTURE_EXT: u32 = 824;
/// `IMAP_VIEWPORT_MASK_EXT`, `MW(864:864)`: system values D, one-bit cells.
pub(crate) const SYSTEM_VALUES_D_EXT: u32 = 864;
/// `PADDING_C(0)`, `MW(872:872)`: a pixel header's padding, to the end of the header.
#[cfg(test)]
pub(crate) const PS_PADDING: u32 = 872;

/// How many cells a run of components has per entry (x, y, z, w; s, t, r, q; red, green, blue,
/// alpha).
pub(crate) const COMPONENTS: u32 = 4;

/// The interpolation values of a pixel header's two-bit cells (`GENERIC_IMAP_X_UNUSED` to
/// `GENERIC_IMAP_X_SCREEN_LINEAR`).
pub(crate) const IMAP_UNUSED: u32 = 0;
/// `GENERIC_IMAP_X_CONSTANT`.
pub(crate) const IMAP_CONSTANT: u32 = 1;
/// `GENERIC_IMAP_X_PERSPECTIVE`.
pub(crate) const IMAP_PERSPECTIVE: u32 = 2;
/// `GENERIC_IMAP_X_SCREEN_LINEAR`. The parser takes it as the only two-bit value left.
#[cfg(any(test, feature = "test-support"))]
pub(crate) const IMAP_SCREEN_LINEAR: u32 = 3;
