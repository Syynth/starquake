//! SPH tests: every position against `clc797sph.h` written out, golden words for both layouts
//! with distinct non-zero values, hand-built headers per format feature, every refusal with its
//! exact error and message, the order of checks with two faults at once, the bits the parser
//! ignores (every bit of the header flipped), and generated round trips.

use proptest::prelude::*;

use super::encode::{EncodeError, encode, encode_bytes};
use super::layout::*;
use super::*;
use crate::testing::config;

/// The bits the parser ignores in both layouts, written out from `clc797sph.h` (unnamed or
/// `RESERVED_COMMON_A`) and `cla097sph.h` (`RESERVED_COMMON_B`, `RESERVED_COMMON_C`).
const COMMON_IGNORED: [(u32, u32); 3] = [(21, 23), (124, 127), (148, 151)];
/// The bits a type 1 header's parser ignores beyond the common ones (inclusive ranges).
const VTG_IGNORED: [(u32, u32); 4] = [(347, 347), (587, 587), (827, 827), (880, 1023)];
/// The bits a type 2 header's parser ignores beyond the common ones (inclusive ranges).
const PS_IGNORED: [(u32, u32); 3] = [(475, 475), (612, 639), (872, 1023)];

fn ignored(layout: Layout, bit: u32) -> bool {
    let extra: &[(u32, u32)] = match layout {
        Layout::Vtg => &VTG_IGNORED,
        Layout::Pixel => &PS_IGNORED,
    };
    COMMON_IGNORED
        .iter()
        .chain(extra)
        .any(|&(first, last)| (first..=last).contains(&bit))
}

/// `words` with the bits the parser ignores cleared.
fn without_ignored(layout: Layout, mut words: [u32; HEADER_WORDS]) -> [u32; HEADER_WORDS] {
    for bit in 0..HEADER_BITS {
        if ignored(layout, bit) {
            flip_bit(&mut words, bit, false);
        }
    }
    words
}

fn flip_bit(words: &mut [u32; HEADER_WORDS], bit: u32, set: bool) {
    let word = &mut words[usize::try_from(bit / 32).unwrap()];
    let mask = 1 << (bit % 32);
    if set {
        *word |= mask;
    } else {
        *word &= !mask;
    }
}

fn toggle(words: &[u32; HEADER_WORDS], bit: u32) -> [u32; HEADER_WORDS] {
    let mut words = *words;
    words[usize::try_from(bit / 32).unwrap()] ^= 1 << (bit % 32);
    words
}

/// Word 0 of a header with this header type and shader type, version 4; hand-built from the
/// literal positions (`SPH_TYPE` `MW(4:0)`, `VERSION` `MW(9:5)`, `SHADER_TYPE` `MW(13:10)`).
fn word0(sph_type: u32, shader_type: u32) -> u32 {
    sph_type | 4 << 5 | shader_type << 10
}

fn mask(bits: u8) -> ComponentMask {
    Components {
        x: bits & 1 != 0,
        y: bits & 2 != 0,
        z: bits & 4 != 0,
        w: bits & 8 != 0,
    }
}

const MODES: [PixelImap; 3] = [
    PixelImap::Constant,
    PixelImap::Perspective,
    PixelImap::ScreenLinear,
];

/// A run of masks, each entry distinct from its neighbours and non-zero.
fn masks<const N: usize>(seed: u8) -> [ComponentMask; N] {
    let mut run = [ComponentMask::default(); N];
    for (index, entry) in (0_u8..).zip(run.iter_mut()) {
        *entry = mask((seed.wrapping_add(index)) % 15 + 1);
    }
    run
}

/// A run of interpolation entries, every component non-zero and neighbours different.
fn imaps<const N: usize>(seed: usize) -> [Components<PixelImap>; N] {
    let mut run = [Components::<PixelImap>::default(); N];
    for (index, entry) in run.iter_mut().enumerate() {
        let at = |offset: usize| MODES[(seed + index + offset) % 3];
        *entry = Components {
            x: at(0),
            y: at(1),
            z: at(2),
            w: at(4),
        };
    }
    run
}

fn golden_common() -> Common {
    Common {
        mrt_enable: true,
        kills_pixels: true,
        does_global_store: true,
        sass_version: 0xb,
        geometry_passthrough: true,
        isbe_space_sharing: true,
        does_load_or_store: true,
        does_fp64: true,
        stream_out_mask: 0x5,
        local_memory_low_size: 0x00a1_b2c3,
        per_patch_attribute_count: 0x5c,
        local_memory_high_size: 0x0000_0d1e,
        threads_per_input_primitive: 0x6d,
        local_memory_crs_size: 0x002f_3a4b,
        output_topology: Some(OutputTopology::TriangleStrip),
        max_output_vertex_count: 0x7e1,
        store_req_start: 0x8f,
        store_req_end: 0x91,
    }
}

fn golden_ab(system_values_a: u32, position: u8) -> SystemValuesAb {
    SystemValuesAb {
        system_values_a,
        vprs_table_index: true,
        primitive_id: true,
        rt_array_index: true,
        viewport_index: true,
        point_size: true,
        position: mask(position),
    }
}

fn golden_c(clip_distances: u8) -> SystemValuesC {
    SystemValuesC {
        clip_distances,
        point_sprite_s: true,
        point_sprite_t: true,
        fog_coordinate: true,
        tessellation_point_u: true,
        tessellation_point_v: true,
        instance_id: true,
        vertex_id: true,
    }
}

fn golden_d_flags() -> SystemValuesD<bool> {
    SystemValuesD {
        viewport_mask: true,
        viewport_mask2: true,
        barycentric: [true; BARYCENTRICS],
        barycentric_w: [true; BARYCENTRICS],
    }
}

fn golden_map(seed: u8, system_values_a: u32, position: u8, clip_distances: u8) -> VtgMap {
    VtgMap {
        system_values_ab: golden_ab(system_values_a, position),
        generic: masks(seed),
        color: masks(seed.wrapping_add(3)),
        system_values_c: golden_c(clip_distances),
        texture: masks(seed.wrapping_add(7)),
        system_values_d: golden_d_flags(),
    }
}

/// A geometry header with a distinct non-zero value in every multi-bit field and every flag
/// set.
fn golden_vtg() -> ProgramHeader {
    ProgramHeader {
        common: golden_common(),
        io: Io::Vtg(Box::new(VtgIo {
            stage: VtgStage::Geometry,
            input: golden_map(0, 0x0012_3456, 0xd, 0xa5),
            output: golden_map(5, 0x0023_4567, 0xb, 0x5a),
            vmap: golden_map(10, 0x0034_5678, 0x7, 0xc3),
        })),
    }
}

/// A pixel header with a distinct non-zero value in every multi-bit field and every flag set.
fn golden_pixel() -> ProgramHeader {
    ProgramHeader {
        common: Common {
            output_topology: Some(OutputTopology::LineStrip),
            ..golden_common()
        },
        io: Io::Pixel(Box::new(PixelIo {
            system_values_ab: golden_ab(0x0045_6789, 0xe),
            generic: imaps(0),
            color: imaps(1),
            system_values_c: golden_c(0x96),
            texture: imaps(2),
            system_values_d: SystemValuesD {
                viewport_mask: PixelImap::Constant,
                viewport_mask2: PixelImap::Perspective,
                barycentric: [
                    PixelImap::ScreenLinear,
                    PixelImap::Constant,
                    PixelImap::Perspective,
                ],
                barycentric_w: [
                    PixelImap::Constant,
                    PixelImap::ScreenLinear,
                    PixelImap::Perspective,
                ],
            },
            targets: masks(2),
            writes_sample_mask: true,
            writes_depth: true,
            reads_interlock_ticket: true,
            uses_inner_coverage: true,
            system_values_a_ext: 0x00ab_cdef,
            system_values_b_ext: 0x9c,
            generic_ext: masks(4),
            color_ext: masks(9),
            system_values_c_ext: 0xbeef,
            texture_ext: masks(11),
            system_values_d_ext: golden_d_flags(),
        })),
    }
}

/// `golden_vtg`'s words, written out (checked by hand for words 0 and 3 to 6: header type 1,
/// version 4, shader type 4, the flags, `SASS_VERSION` 0xb, the stream-out mask 5; the CRS size
/// and topology 7; the vertex count 0x7e1 and the two store fields; system values A 0x123456 with
/// the five flags and position 0xd; generic masks 1 to 8).
const GOLDEN_VTG_WORDS: [u32; HEADER_WORDS] = [
    0x5f17_d081,
    0x5ca1_b2c3,
    0x6d00_0d1e,
    0x072f_3a4b,
    0x9108_f7e1,
    0xdf92_3456,
    0x8765_4321,
    0x1fed_cba9,
    0x9876_5432,
    0x21fe_dcba,
    0xf7a5_7654,
    0xfedc_ba98,
    0x4567_ff21,
    0x9876_bfa3,
    0x21fe_dcba,
    0xa987_6543,
    0x321f_edcb,
    0xcba9_7654,
    0x1fed_f75a,
    0xff76_5432,
    0x7fb4_5678,
    0x321f_edcb,
    0xba98_7654,
    0x4321_fedc,
    0xcba9_8765,
    0xf7c3_21fe,
    0xa987_6543,
    0x0000_ffcb,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
];

/// `golden_pixel`'s words, written out (checked by hand for words 6, 18, 19 and 20: generic
/// attribute 0 as constant, perspective, screen-linear, perspective (0xb9); targets 3 to 10; the
/// four output flags; system values A and B of the extension).
const GOLDEN_PIXEL_WORDS: [u32; HEADER_WORDS] = [
    0x5f17_d482,
    0x5ca1_b2c3,
    0x6d00_0d1e,
    0x062f_3a4b,
    0x9108_f7e1,
    0xefc5_6789,
    0xb967_deb9,
    0xdeb9_67de,
    0x67de_b967,
    0xb967_deb9,
    0xdeb9_67de,
    0x67de_b967,
    0xb967_deb9,
    0xdeb9_67de,
    0xf796_67de,
    0x67de_b967,
    0xb967_deb9,
    0xb679_67de,
    0xa987_6543,
    0x0000_000f,
    0x9cab_cdef,
    0xcba9_8765,
    0x5432_1fed,
    0xdcba_9876,
    0x6543_21fe,
    0xdcbe_efba,
    0x6543_21fe,
    0x0000_00ff,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
];

fn parse(words: &[u32; HEADER_WORDS]) -> Result<ProgramHeader, HeaderError> {
    ProgramHeader::from_words(words)
}

fn pixel(header: &ProgramHeader) -> &PixelIo {
    match &header.io {
        Io::Pixel(io) => Some(io),
        Io::Vtg(_) => None,
    }
    .expect("a pixel header")
}

fn pixel_mut(header: &mut ProgramHeader) -> &mut PixelIo {
    match &mut header.io {
        Io::Pixel(io) => Some(io),
        Io::Vtg(_) => None,
    }
    .expect("a pixel header")
}

fn vtg_mut(header: &mut ProgramHeader) -> &mut VtgIo {
    match &mut header.io {
        Io::Vtg(io) => Some(io),
        Io::Pixel(_) => None,
    }
    .expect("a vertex header")
}

fn vtg(stage: VtgStage, input: VtgMap, output: VtgMap, vmap: VtgMap) -> Io {
    Io::Vtg(Box::new(VtgIo {
        stage,
        input,
        output,
        vmap,
    }))
}

#[test]
fn golden_words_cover_every_field() {
    for (header, golden) in [
        (golden_vtg(), GOLDEN_VTG_WORDS),
        (golden_pixel(), GOLDEN_PIXEL_WORDS),
    ] {
        assert_eq!(encode(&header), Ok(golden));
        assert_eq!(parse(&golden), Ok(header));
    }
}

#[test]
fn bytes_are_little_endian_words_and_the_code_follows() {
    for (header, golden) in [
        (golden_vtg(), GOLDEN_VTG_WORDS),
        (golden_pixel(), GOLDEN_PIXEL_WORDS),
    ] {
        let bytes = encode_bytes(&header).unwrap();
        let expected: Vec<u8> = golden.iter().flat_map(|word| word.to_le_bytes()).collect();
        assert_eq!(bytes.to_vec(), expected);
        let mut program = bytes.to_vec();
        program.extend_from_slice(&[0xde, 0xad]);
        let (parsed, code) = ProgramHeader::from_program(&program).unwrap();
        assert_eq!(parsed, header);
        assert_eq!(code, &[0xde, 0xad]);
    }
}

/// Every common position against `clc797sph.h` (and the five common fields against
/// `cla097sph.h` and NAK), written out as the define's `MW(high:low)` gives it; the shipping
/// constants are only on the left.
#[test]
fn layout_matches_the_reference_literals() {
    let field = HeaderField::new;
    let common = [
        (SPH_TYPE, field(0, 5)),
        (VERSION, field(5, 5)),
        (SHADER_TYPE, field(10, 4)),
        (MRT_ENABLE, field(14, 1)),
        (KILLS_PIXELS, field(15, 1)),
        (DOES_GLOBAL_STORE, field(16, 1)),
        (SASS_VERSION, field(17, 4)),
        (GEOMETRY_PASSTHROUGH, field(24, 1)),
        (ISBE_SPACE_SHARING, field(25, 1)),
        (DOES_LOAD_OR_STORE, field(26, 1)),
        (DOES_FP64, field(27, 1)),
        (STREAM_OUT_MASK, field(28, 4)),
        (LOCAL_MEMORY_LOW_SIZE, field(32, 24)),
        (PER_PATCH_ATTRIBUTE_COUNT, field(56, 8)),
        (LOCAL_MEMORY_HIGH_SIZE, field(64, 24)),
        (THREADS_PER_INPUT_PRIMITIVE, field(88, 8)),
        (LOCAL_MEMORY_CRS_SIZE, field(96, 24)),
        (OUTPUT_TOPOLOGY, field(120, 4)),
        (MAX_OUTPUT_VERTEX_COUNT, field(128, 12)),
        (STORE_REQ_START, field(140, 8)),
        (STORE_REQ_END, field(152, 8)),
        (OMAP_SAMPLE_MASK, field(608, 1)),
        (OMAP_DEPTH, field(609, 1)),
        (READS_INTERLOCK_TICKET, field(610, 1)),
        (USES_INNER_COVERAGE, field(611, 1)),
        (SYSTEM_VALUES_A_EXT, field(640, 24)),
        (SYSTEM_VALUES_B_EXT, field(664, 8)),
        (SYSTEM_VALUES_C_EXT, field(808, 16)),
    ];
    for (shipping, reference) in common {
        assert_eq!(shipping, reference);
    }
    assert_eq!(
        [SPH_TYPE_VTG, SPH_TYPE_PS, ProgramHeader::VERSION],
        [1, 2, 4]
    );
    assert_eq!(
        [
            SHADER_TYPE_VERTEX,
            SHADER_TYPE_TESSELLATION_INIT,
            SHADER_TYPE_TESSELLATION,
            SHADER_TYPE_GEOMETRY,
            SHADER_TYPE_PIXEL,
        ],
        [1, 2, 3, 4, 5]
    );
    assert_eq!(
        [
            OUTPUT_TOPOLOGY_NONE,
            OUTPUT_TOPOLOGY_POINTLIST,
            OUTPUT_TOPOLOGY_LINESTRIP,
            OUTPUT_TOPOLOGY_TRIANGLESTRIP,
        ],
        [0, 1, 6, 7]
    );
    assert_eq!(
        [
            IMAP_UNUSED,
            IMAP_CONSTANT,
            IMAP_PERSPECTIVE,
            IMAP_SCREEN_LINEAR
        ],
        [0, 1, 2, 3]
    );
    assert_eq!((HEADER_WORDS, HEADER_BYTES), (32, 128));
}

/// The type 1 header's three maps against `clc797sph.h`, written out.
#[test]
fn vtg_layout_matches_the_reference_literals() {
    let field = HeaderField::new;
    // The three type 1 maps: `IMAP_*`, `OMAP_*`, `VMAP_*` of the same name, 240 bits apart.
    for (base, offset) in [(IMAP, 0), (OMAP, 240), (VMAP, 480)] {
        let at = |bit: u32| bit + offset;
        assert_eq!(SYSTEM_VALUES_A.at(base), field(at(160), 23));
        assert_eq!(VPRS_TABLE_INDEX.at(base), field(at(183), 1));
        assert_eq!(PRIMITIVE_ID.at(base), field(at(184), 1));
        assert_eq!(RT_ARRAY_INDEX.at(base), field(at(185), 1));
        assert_eq!(VIEWPORT_INDEX.at(base), field(at(186), 1));
        assert_eq!(POINT_SIZE.at(base), field(at(187), 1));
        assert_eq!(cell::<bool>(base + POSITION, 0), field(at(188), 1));
        assert_eq!(cell::<bool>(base + POSITION, 3), field(at(191), 1));
        // `GENERIC_IMAP_X(0)` and `GENERIC_IMAP_W(31)`: 192 and 195 + 31 * 4.
        assert_eq!(
            cell::<bool>(entry::<bool>(base + VTG_GENERIC, 0), 0),
            field(at(192), 1)
        );
        assert_eq!(
            cell::<bool>(entry::<bool>(base + VTG_GENERIC, 31), 3),
            field(at(319), 1)
        );
        // `IMAP_COLOR_FRONT_DIFFUSE_RED`, `IMAP_COLOR_BACK_DIFFUSE_RED` and
        // `IMAP_COLOR_BACK_SPECULAR_ALPHA`.
        assert_eq!(
            cell::<bool>(entry::<bool>(base + VTG_COLOR, 0), 0),
            field(at(320), 1)
        );
        assert_eq!(
            cell::<bool>(entry::<bool>(base + VTG_COLOR, 2), 0),
            field(at(328), 1)
        );
        assert_eq!(
            cell::<bool>(entry::<bool>(base + VTG_COLOR, 3), 3),
            field(at(335), 1)
        );
        let c = base + VTG_SYSTEM_VALUES_C;
        assert_eq!(CLIP_DISTANCES.at(c), field(at(336), 8));
        assert_eq!(POINT_SPRITE_S.at(c), field(at(344), 1));
        assert_eq!(POINT_SPRITE_T.at(c), field(at(345), 1));
        assert_eq!(FOG_COORDINATE.at(c), field(at(346), 1));
        assert_eq!(TESSELLATION_POINT_U.at(c), field(at(348), 1));
        assert_eq!(TESSELLATION_POINT_V.at(c), field(at(349), 1));
        assert_eq!(INSTANCE_ID.at(c), field(at(350), 1));
        assert_eq!(VERTEX_ID.at(c), field(at(351), 1));
        // `TEX_IMAP_S(0)` and `TEX_IMAP_Q(9)`: 352 and 355 + 9 * 4.
        assert_eq!(
            cell::<bool>(entry::<bool>(base + VTG_TEXTURE, 0), 0),
            field(at(352), 1)
        );
        assert_eq!(
            cell::<bool>(entry::<bool>(base + VTG_TEXTURE, 9), 3),
            field(at(391), 1)
        );
        let d = base + VTG_SYSTEM_VALUES_D;
        assert_eq!(cell::<bool>(d, VIEWPORT_MASK_CELL), field(at(392), 1));
        assert_eq!(cell::<bool>(d, VIEWPORT_MASK2_CELL), field(at(393), 1));
        assert_eq!(cell::<bool>(d, BARYCENTRIC_CELL), field(at(394), 1));
        assert_eq!(cell::<bool>(d, BARYCENTRIC_CELL + 2), field(at(396), 1));
        assert_eq!(cell::<bool>(d, BARYCENTRIC_W_CELL), field(at(397), 1));
        assert_eq!(cell::<bool>(d, BARYCENTRIC_W_CELL + 2), field(at(399), 1));
    }
    assert_eq!(VTG_PADDING, 880);
}

/// The type 2 header's positions against `clc797sph.h`, written out.
#[test]
fn pixel_layout_matches_the_reference_literals() {
    let field = HeaderField::new;
    assert_eq!(VPRS_TABLE_INDEX.at(IMAP), field(183, 1));
    let ps = |start: u32, entry_index: u32, component: u32| {
        cell::<PixelImap>(entry::<PixelImap>(start, entry_index), component)
    };
    // `GENERIC_IMAP_X(0)` `MW(193:192)`, `GENERIC_IMAP_Y(0)` `MW(195:194)`,
    // `GENERIC_IMAP_W(31)` `MW((199+31*8):(198+31*8))`.
    assert_eq!(ps(PS_GENERIC, 0, 0), field(192, 2));
    assert_eq!(ps(PS_GENERIC, 0, 1), field(194, 2));
    assert_eq!(ps(PS_GENERIC, 31, 3), field(446, 2));
    // `IMAP_COLOR_DIFFUSE_RED` `MW(449:448)`, `IMAP_COLOR_SPECULAR_RED` `MW(457:456)`,
    // `IMAP_COLOR_SPECULAR_ALPHA` `MW(463:462)`.
    assert_eq!(ps(PS_COLOR, 0, 0), field(448, 2));
    assert_eq!(ps(PS_COLOR, 1, 0), field(456, 2));
    assert_eq!(ps(PS_COLOR, 1, 3), field(462, 2));
    assert_eq!(CLIP_DISTANCES.at(PS_SYSTEM_VALUES_C), field(464, 8));
    assert_eq!(POINT_SPRITE_S.at(PS_SYSTEM_VALUES_C), field(472, 1));
    assert_eq!(FOG_COORDINATE.at(PS_SYSTEM_VALUES_C), field(474, 1));
    assert_eq!(TESSELLATION_POINT_U.at(PS_SYSTEM_VALUES_C), field(476, 1));
    assert_eq!(VERTEX_ID.at(PS_SYSTEM_VALUES_C), field(479, 1));
    // `TEX_IMAP_S(0)` `MW(481:480)`, `TEX_IMAP_Q(9)` `MW((487+72):(486+72))`.
    assert_eq!(ps(PS_TEXTURE, 0, 0), field(480, 2));
    assert_eq!(ps(PS_TEXTURE, 9, 3), field(558, 2));
    // `IMAP_VIEWPORT_MASK` `MW(561:560)` to `IMAP_BARYCENTRIC2_W` `MW(575:574)`.
    let d = PS_SYSTEM_VALUES_D;
    assert_eq!(cell::<PixelImap>(d, VIEWPORT_MASK_CELL), field(560, 2));
    assert_eq!(cell::<PixelImap>(d, VIEWPORT_MASK2_CELL), field(562, 2));
    assert_eq!(cell::<PixelImap>(d, BARYCENTRIC_CELL), field(564, 2));
    assert_eq!(cell::<PixelImap>(d, BARYCENTRIC_W_CELL), field(570, 2));
    assert_eq!(cell::<PixelImap>(d, BARYCENTRIC_W_CELL + 2), field(574, 2));
    // `OMAP_RED(0)` and `OMAP_ALPHA(7)`: 576 and 579 + 7 * 4.
    assert_eq!(cell::<bool>(entry::<bool>(PS_TARGETS, 0), 0), field(576, 1));
    assert_eq!(cell::<bool>(entry::<bool>(PS_TARGETS, 7), 3), field(607, 1));
    // `GENERIC_IMAP_X_EXT(0)`, `GENERIC_IMAP_W_EXT(31)`, the colours, `TEX_IMAP_Q_EXT(9)`.
    assert_eq!(
        cell::<bool>(entry::<bool>(GENERIC_EXT, 0), 0),
        field(672, 1)
    );
    assert_eq!(
        cell::<bool>(entry::<bool>(GENERIC_EXT, 31), 3),
        field(799, 1)
    );
    assert_eq!(cell::<bool>(entry::<bool>(COLOR_EXT, 0), 0), field(800, 1));
    assert_eq!(cell::<bool>(entry::<bool>(COLOR_EXT, 1), 3), field(807, 1));
    assert_eq!(
        cell::<bool>(entry::<bool>(TEXTURE_EXT, 0), 0),
        field(824, 1)
    );
    assert_eq!(
        cell::<bool>(entry::<bool>(TEXTURE_EXT, 9), 3),
        field(863, 1)
    );
    assert_eq!(
        cell::<bool>(SYSTEM_VALUES_D_EXT, VIEWPORT_MASK_CELL),
        field(864, 1)
    );
    assert_eq!(
        cell::<bool>(SYSTEM_VALUES_D_EXT, BARYCENTRIC_W_CELL + 2),
        field(871, 1)
    );
    assert_eq!(PS_PADDING, 872);
}

/// Each run of entries ends exactly where the next field starts, so the counts
/// (`GENERIC_ATTRIBUTES`, `TEXTURE_COORDINATES`, the colours, `COLOR_TARGETS`) follow from the
/// header's positions.
#[test]
fn runs_end_where_the_next_field_starts() {
    let run_end = |start: u32, count: usize, width: u32| {
        start + u32::try_from(count).unwrap() * COMPONENTS * width
    };
    assert_eq!(run_end(VTG_GENERIC, GENERIC_ATTRIBUTES, 1), VTG_COLOR);
    assert_eq!(run_end(VTG_COLOR, VTG_COLORS, 1), VTG_SYSTEM_VALUES_C);
    assert_eq!(VTG_SYSTEM_VALUES_C + VERTEX_ID.end(), VTG_TEXTURE);
    assert_eq!(
        run_end(VTG_TEXTURE, TEXTURE_COORDINATES, 1),
        VTG_SYSTEM_VALUES_D
    );
    assert_eq!(
        cell::<bool>(VTG_SYSTEM_VALUES_D, BARYCENTRIC_W_CELL + 2).end(),
        VTG_MAP_BITS
    );
    assert_eq!(POSITION + COMPONENTS, SYSTEM_VALUES_AB_BITS);
    assert_eq!(SYSTEM_VALUES_AB_BITS, VTG_GENERIC);
    assert_eq!(SYSTEM_VALUES_A.end(), VPRS_TABLE_INDEX.start);
    assert_eq!(IMAP + VTG_MAP_BITS, OMAP);
    assert_eq!(OMAP + VTG_MAP_BITS, VMAP);
    assert_eq!(VMAP + VTG_MAP_BITS, VTG_PADDING);

    assert_eq!(IMAP + VTG_GENERIC, PS_GENERIC);
    assert_eq!(run_end(PS_GENERIC, GENERIC_ATTRIBUTES, 2), PS_COLOR);
    assert_eq!(run_end(PS_COLOR, PS_COLORS, 2), PS_SYSTEM_VALUES_C);
    assert_eq!(PS_SYSTEM_VALUES_C + VERTEX_ID.end(), PS_TEXTURE);
    assert_eq!(
        run_end(PS_TEXTURE, TEXTURE_COORDINATES, 2),
        PS_SYSTEM_VALUES_D
    );
    assert_eq!(
        cell::<PixelImap>(PS_SYSTEM_VALUES_D, BARYCENTRIC_W_CELL + 2).end(),
        PS_TARGETS
    );
    assert_eq!(
        run_end(PS_TARGETS, COLOR_TARGETS, 1),
        OMAP_SAMPLE_MASK.start
    );
    assert_eq!(SYSTEM_VALUES_A_EXT.end(), SYSTEM_VALUES_B_EXT.start);
    assert_eq!(SYSTEM_VALUES_B_EXT.end(), GENERIC_EXT);
    assert_eq!(run_end(GENERIC_EXT, GENERIC_ATTRIBUTES, 1), COLOR_EXT);
    assert_eq!(run_end(COLOR_EXT, PS_COLORS, 1), SYSTEM_VALUES_C_EXT.start);
    assert_eq!(SYSTEM_VALUES_C_EXT.end(), TEXTURE_EXT);
    assert_eq!(
        run_end(TEXTURE_EXT, TEXTURE_COORDINATES, 1),
        SYSTEM_VALUES_D_EXT
    );
    assert_eq!(
        cell::<bool>(SYSTEM_VALUES_D_EXT, BARYCENTRIC_W_CELL + 2).end(),
        PS_PADDING
    );
    assert_eq!(HEADER_BITS, u32::try_from(HEADER_WORDS).unwrap() * 32);
}

/// Every bit of the header, flipped one at a time in valid headers of each layout (the golden
/// ones and nearly empty ones): the parse is the same exactly for the bits the module docs list
/// as ignored, and differs (another header, or a refusal) for every other bit.
#[test]
fn ignored_bits_are_exactly_the_unnamed_ones() {
    let empty = |sph_type: u32, shader_type: u32| {
        let mut words = [0; HEADER_WORDS];
        words[0] = word0(sph_type, shader_type);
        words
    };
    for (layout, base) in [
        (Layout::Vtg, GOLDEN_VTG_WORDS),
        (Layout::Vtg, empty(1, 1)),
        (Layout::Pixel, GOLDEN_PIXEL_WORDS),
        (Layout::Pixel, empty(2, 5)),
    ] {
        let parsed = parse(&base);
        assert!(parsed.is_ok());
        for bit in 0..HEADER_BITS {
            let flipped = parse(&toggle(&base, bit));
            if ignored(layout, bit) {
                assert_eq!(flipped, parsed, "bit {bit} should be ignored");
            } else {
                assert_ne!(flipped, parsed, "bit {bit} should be read");
            }
        }
    }
}

/// One hand-built header per shader type, from literal positions only.
#[test]
fn hand_built_headers_of_each_shader_type() {
    let empty = VtgMap::default();
    for (shader_type, stage) in [
        (1, VtgStage::Vertex),
        (2, VtgStage::TessellationInit),
        (3, VtgStage::Tessellation),
        (4, VtgStage::Geometry),
    ] {
        let mut words = [0; HEADER_WORDS];
        words[0] = word0(1, shader_type);
        let header = parse(&words).unwrap();
        assert_eq!(header.io, vtg(stage, empty, empty, empty));
        assert_eq!(header.common, Common::default());
        assert_eq!(header.layout(), Layout::Vtg);
    }
    let mut words = [0; HEADER_WORDS];
    words[0] = word0(2, 5);
    let header = parse(&words).unwrap();
    assert_eq!(header.io, Io::Pixel(Box::default()));
    assert_eq!(header.layout(), Layout::Pixel);
}

#[test]
fn stage_names_each_shader_type() {
    let cases = [
        (1, 1, GuestStage::Vertex),
        (1, 2, GuestStage::TessellationInit),
        (1, 3, GuestStage::Tessellation),
        (1, 4, GuestStage::Geometry),
        (2, 5, GuestStage::Pixel),
    ];
    for (sph_type, shader_type, stage) in cases {
        let mut words = [0; HEADER_WORDS];
        words[0] = word0(sph_type, shader_type);
        assert_eq!(parse(&words).unwrap().stage(), stage);
    }
}

/// One hand-built pixel header per interpolation mode: generic attribute 0's x (bits 192..194)
/// and attribute 1's w (bits 206..208) carry the mode, nothing else is set.
#[test]
fn hand_built_headers_of_each_interpolation_mode() {
    for (bits, mode) in [
        (0, PixelImap::Unused),
        (1, PixelImap::Constant),
        (2, PixelImap::Perspective),
        (3, PixelImap::ScreenLinear),
    ] {
        let mut words = [0; HEADER_WORDS];
        words[0] = word0(2, 5);
        words[6] = bits | bits << 14;
        let header = parse(&words).unwrap();
        let io = pixel(&header);
        let mut expected = PixelIo::default();
        expected.generic[0].x = mode;
        expected.generic[1].w = mode;
        assert_eq!(*io, expected);
    }
}

/// A hand-built pixel header writing target 0's red, all of target 1, target 2's green only,
/// target 3's blue only and target 7's alpha (word 18 is bits 576..608, four per target), plus
/// the sample mask and depth (bits 608, 609). Green and blue apart pin the component order of
/// `ComponentMask::bits`.
#[test]
fn hand_built_header_with_colour_targets() {
    let mut words = [0; HEADER_WORDS];
    words[0] = word0(2, 5);
    words[18] = 0x8000_42f1;
    words[19] = 0b11;
    let header = parse(&words).unwrap();
    let io = pixel(&header);
    let mut expected = PixelIo::default();
    expected.targets[0] = mask(0b0001);
    expected.targets[1] = mask(0b1111);
    expected.targets[2] = mask(0b0010);
    expected.targets[3] = mask(0b0100);
    expected.targets[7] = mask(0b1000);
    expected.writes_sample_mask = true;
    expected.writes_depth = true;
    assert_eq!(*io, expected);
    assert_eq!(
        io.color_outputs(),
        ColorOutputs {
            targets: [0x1, 0xf, 0x2, 0x4, 0, 0, 0, 0x8]
        }
    );
}

/// A hand-built vertex header with local memory: the low size in bits 32..56 and the high size
/// in bits 64..88, each with its neighbour field set beside it (the per-patch count at 56, the
/// thread count at 88) so a wrong width shows, and the CRS size in bits 96..120.
#[test]
fn hand_built_header_with_local_memory() {
    let mut words = [0; HEADER_WORDS];
    words[0] = word0(1, 1);
    words[1] = 0x12ab_cdef;
    words[2] = 0x3400_0003;
    words[3] = 0x0000_0040;
    let header = parse(&words).unwrap();
    assert_eq!(
        header.common,
        Common {
            local_memory_low_size: 0x00ab_cdef,
            per_patch_attribute_count: 0x12,
            local_memory_high_size: 0x0000_0003,
            threads_per_input_primitive: 0x34,
            local_memory_crs_size: 0x40,
            ..Common::default()
        }
    );
}

/// The high field counts from bit 24 of the size, as Eden composes it, not from bit 32 as
/// NAK's writer stores it (module docs, "Conflict").
#[test]
fn local_memory_size_follows_eden() {
    let common = Common {
        local_memory_low_size: 0x00ab_cdef,
        local_memory_high_size: 0x0000_0003,
        ..Common::default()
    };
    assert_eq!(common.local_memory_size(), 0x03ab_cdef);
    let widest = Common {
        local_memory_low_size: LOCAL_MEMORY_LOW_SIZE.ones(),
        local_memory_high_size: LOCAL_MEMORY_HIGH_SIZE.ones(),
        ..Common::default()
    };
    assert_eq!(widest.local_memory_size(), 0xffff_ffff_ffff);
    // The type does not hold the fields to their widths: a caller-built `Common` reaches 56
    // bits (the docs of `local_memory_size`).
    let caller_built = Common {
        local_memory_low_size: u32::MAX,
        local_memory_high_size: u32::MAX,
        ..Common::default()
    };
    assert_eq!(caller_built.local_memory_size(), 0x00ff_ffff_ffff_ffff);
}

/// A hand-built geometry header: pass-through, the topology, vertex count and thread count, and
/// one bit in each of the three maps (generic attribute 0's x at 192, 432 and 672).
#[test]
fn hand_built_geometry_header_with_three_maps() {
    let mut words = [0; HEADER_WORDS];
    words[0] = word0(1, 4) | 1 << 24;
    words[2] = 0x0300_0000;
    words[3] = 0x0700_0000;
    words[4] = 0x0000_0004;
    words[6] = 1;
    words[13] = 1 << 16;
    words[21] = 1;
    let header = parse(&words).unwrap();
    assert_eq!(
        header.common,
        Common {
            geometry_passthrough: true,
            threads_per_input_primitive: 3,
            output_topology: Some(OutputTopology::TriangleStrip),
            max_output_vertex_count: 4,
            ..Common::default()
        }
    );
    let mut one = VtgMap::default();
    one.generic[0].x = true;
    assert_eq!(header.io, vtg(VtgStage::Geometry, one, one, one));
}

#[test]
fn short_programs_are_refused_at_the_exact_edge() {
    for len in [0, 1, HEADER_BYTES - 1] {
        assert_eq!(
            ProgramHeader::from_program(&vec![0; len]),
            Err(HeaderError::Short {
                len,
                needed: HEADER_BYTES
            })
        );
    }
    let exact = encode_bytes(&golden_pixel()).unwrap();
    let (header, code) = ProgramHeader::from_program(&exact).unwrap();
    assert_eq!(header, golden_pixel());
    assert_eq!(code.len(), 0);
    // One byte short of a valid header is refused, not read with the missing byte as zero.
    assert_eq!(
        ProgramHeader::from_program(&exact[..HEADER_BYTES - 1]),
        Err(HeaderError::Short {
            len: HEADER_BYTES - 1,
            needed: HEADER_BYTES
        })
    );
}

#[test]
fn each_check_refuses_with_its_exact_error() {
    let with = |word: usize, value: u32| {
        let mut words = [0; HEADER_WORDS];
        words[0] = word0(1, 1);
        words[word] = value;
        parse(&words)
    };
    // The version, one either side of 4 and the extremes.
    for version in [0, 3, 5, 31] {
        assert_eq!(
            with(0, 1 | version << 5 | 1 << 10),
            Err(HeaderError::Version {
                version,
                supported: 4
            })
        );
    }
    assert!(with(0, word0(1, 1)).is_ok());
    for value in [0, 3, 31] {
        assert_eq!(
            with(0, word0(value, 1)),
            Err(HeaderError::SphType { value })
        );
    }
    for value in [0, 6, 15] {
        assert_eq!(
            with(0, word0(1, value)),
            Err(HeaderError::ShaderType { value })
        );
    }
    for (shader_type, stage) in [
        (1, GuestStage::Vertex),
        (2, GuestStage::TessellationInit),
        (3, GuestStage::Tessellation),
        (4, GuestStage::Geometry),
    ] {
        assert_eq!(
            with(0, word0(2, shader_type)),
            Err(HeaderError::TypeMismatch {
                layout: Layout::Pixel,
                stage
            })
        );
    }
    assert_eq!(
        with(0, word0(1, 5)),
        Err(HeaderError::TypeMismatch {
            layout: Layout::Vtg,
            stage: GuestStage::Pixel
        })
    );
    for value in [2, 3, 4, 5, 8, 15] {
        assert_eq!(
            with(3, value << 24),
            Err(HeaderError::OutputTopology { value })
        );
    }
    for value in [0, 1, 6, 7] {
        assert!(with(3, value << 24).is_ok());
    }
}

#[test]
fn output_topology_is_checked_for_every_stage() {
    for (sph_type, shader_type) in [(1, 1), (1, 2), (1, 3), (1, 4), (2, 5)] {
        let mut words = [0; HEADER_WORDS];
        words[0] = word0(sph_type, shader_type);
        words[3] = 2 << 24;
        assert_eq!(parse(&words), Err(HeaderError::OutputTopology { value: 2 }));
    }
}

/// Every pair of checks whose faults can occur together, with both faults present: the earlier
/// one is reported. A mismatch needs a valid header type and shader type, so it has no case with
/// either of their faults.
#[test]
fn the_checks_run_in_the_documented_order() {
    let bad_topology = 2 << 24;
    let version_3 = HeaderError::Version {
        version: 3,
        supported: 4,
    };
    let cases = [
        // Version and header type both wrong.
        (3 << 5 | 1 << 10, 0, version_3),
        // Version and shader type both wrong.
        (1 | 3 << 5 | 6 << 10, 0, version_3),
        // Version wrong and the header type not matching the shader type.
        (2 | 3 << 5 | 1 << 10, 0, version_3),
        // Version and output topology both wrong.
        (1 | 3 << 5 | 1 << 10, bad_topology, version_3),
        // Header type and output topology both wrong.
        (word0(3, 1), bad_topology, HeaderError::SphType { value: 3 }),
        // Header type and shader type both wrong.
        (word0(3, 6), 0, HeaderError::SphType { value: 3 }),
        // Shader type and output topology both wrong.
        (
            word0(1, 6),
            bad_topology,
            HeaderError::ShaderType { value: 6 },
        ),
        // Mismatch and output topology both wrong.
        (
            word0(2, 1),
            bad_topology,
            HeaderError::TypeMismatch {
                layout: Layout::Pixel,
                stage: GuestStage::Vertex,
            },
        ),
    ];
    for (first, fourth, error) in cases {
        let mut words = [0; HEADER_WORDS];
        words[0] = first;
        words[3] = fourth;
        assert_eq!(parse(&words), Err(error));
    }
    // Short is checked before the version: 127 bytes of a header with a wrong version.
    let mut bytes = vec![0; HEADER_BYTES - 1];
    bytes[0] = 0x3f;
    assert_eq!(
        ProgramHeader::from_program(&bytes),
        Err(HeaderError::Short {
            len: HEADER_BYTES - 1,
            needed: HEADER_BYTES
        })
    );
}

#[test]
fn errors_read_as_plain_words() {
    let cases = [
        (
            HeaderError::Short {
                len: 0,
                needed: HEADER_BYTES,
            },
            "a shader program header needs 128 bytes, but the program has 0",
        ),
        (
            HeaderError::Short {
                len: 1,
                needed: HEADER_BYTES,
            },
            "a shader program header needs 128 bytes, but the program has 1",
        ),
        (
            HeaderError::Version {
                version: 3,
                supported: 4,
            },
            "shader program header version 3 is not supported; only version 4 is",
        ),
        (
            HeaderError::SphType { value: 0 },
            "shader program header type 0 is neither 1 (vertex, tessellation or geometry) nor 2 (pixel)",
        ),
        (
            HeaderError::ShaderType { value: 6 },
            "shader type 6 is not one of 1 (vertex) to 5 (pixel)",
        ),
        (
            HeaderError::TypeMismatch {
                layout: Layout::Pixel,
                stage: GuestStage::Geometry,
            },
            "a pixel header cannot describe a geometry program",
        ),
        (
            HeaderError::TypeMismatch {
                layout: Layout::Vtg,
                stage: GuestStage::Pixel,
            },
            "a vertex, tessellation and geometry header cannot describe a pixel program",
        ),
        (
            HeaderError::OutputTopology { value: 2 },
            "output topology 2 is not 0 (none), 1 (point list), 6 (line strip) or 7 (triangle strip)",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
}

/// The encoder refuses a value wider than its field, at the exact edge, for every field whose
/// type holds more bits than the field.
#[test]
fn the_encoder_refuses_values_wider_than_their_field() {
    type Set = fn(&mut ProgramHeader, u32);
    let cases: [(HeaderField, ProgramHeader, Set); 9] = [
        (SASS_VERSION, golden_vtg(), |h, v| {
            h.common.sass_version = u8::try_from(v).unwrap();
        }),
        (STREAM_OUT_MASK, golden_vtg(), |h, v| {
            h.common.stream_out_mask = u8::try_from(v).unwrap();
        }),
        (LOCAL_MEMORY_LOW_SIZE, golden_vtg(), |h, v| {
            h.common.local_memory_low_size = v;
        }),
        (LOCAL_MEMORY_HIGH_SIZE, golden_vtg(), |h, v| {
            h.common.local_memory_high_size = v;
        }),
        (LOCAL_MEMORY_CRS_SIZE, golden_vtg(), |h, v| {
            h.common.local_memory_crs_size = v;
        }),
        (MAX_OUTPUT_VERTEX_COUNT, golden_vtg(), |h, v| {
            h.common.max_output_vertex_count = u16::try_from(v).unwrap();
        }),
        (SYSTEM_VALUES_A.at(OMAP), golden_vtg(), |h, v| {
            vtg_mut(h).output.system_values_ab.system_values_a = v;
        }),
        (SYSTEM_VALUES_A.at(IMAP), golden_pixel(), |h, v| {
            pixel_mut(h).system_values_ab.system_values_a = v;
        }),
        (SYSTEM_VALUES_A_EXT, golden_pixel(), |h, v| {
            pixel_mut(h).system_values_a_ext = v;
        }),
    ];
    for (field, base, set) in cases {
        let max = field.ones();
        let mut fits = base.clone();
        set(&mut fits, max);
        let words = encode(&fits).unwrap();
        assert_eq!(field.read(&words), max);
        assert_eq!(parse(&words), Ok(fits));
        let mut wide = base;
        set(&mut wide, max + 1);
        assert_eq!(
            encode(&wide),
            Err(EncodeError::TooWide {
                start: field.start,
                width: field.width,
                value: max + 1
            })
        );
    }
    assert_eq!(
        EncodeError::TooWide {
            start: 17,
            width: 4,
            value: 0x10
        }
        .to_string(),
        "the value 0x10 does not fit the 4-bit field at bit 17"
    );
}

#[test]
fn a_field_may_cross_a_word_boundary() {
    let mut words = [0; HEADER_WORDS];
    words[0] = 0x8000_0000;
    words[1] = 0x0000_0005;
    assert_eq!(HeaderField::new(30, 5).read(&words), 0b1_0110);
    assert_eq!(HeaderField::new(31, 1).read(&words), 1);
    assert_eq!(HeaderField::new(32, 32).read(&words), 5);
    // Past the end reads as zero rather than panicking.
    assert_eq!(
        HeaderField::new(HEADER_BITS - 1, 2).read(&[u32::MAX; HEADER_WORDS]),
        1
    );
    assert_eq!(
        HeaderField::new(HEADER_BITS, 4).read(&[u32::MAX; HEADER_WORDS]),
        0
    );
    assert_eq!(HeaderField::new(0, 0).ones(), 0);
    assert_eq!(HeaderField::new(0, 1).ones(), 1);
    assert_eq!(HeaderField::new(0, 31).ones(), 0x7fff_ffff);
    assert_eq!(HeaderField::new(0, 32).ones(), u32::MAX);
}

/// The index of the word that holds `field`, which must not cross a word boundary.
fn word_of(field: HeaderField) -> usize {
    assert_eq!(field.start / 32, (field.start + field.width - 1) / 32);
    usize::try_from(field.start / 32).unwrap()
}

/// `field`'s bits within its word.
fn mask_in_word(field: HeaderField) -> u32 {
    field.ones() << (field.start % 32)
}

/// Words with a valid header type, shader type and topology forced into arbitrary ones.
fn valid_words() -> impl Strategy<Value = [u32; HEADER_WORDS]> {
    let word0_fields = [SPH_TYPE, VERSION, SHADER_TYPE];
    assert!(word0_fields.iter().all(|&field| word_of(field) == 0));
    let word0_mask = word0_fields
        .iter()
        .fold(0, |mask, &field| mask | mask_in_word(field));
    let topology_word = word_of(OUTPUT_TOPOLOGY);
    (
        any::<[u32; HEADER_WORDS]>(),
        0_usize..5,
        prop::sample::select(vec![0_u32, 1, 6, 7]),
    )
        .prop_map(move |(mut words, stage, topology)| {
            let (sph_type, shader_type) = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 5)][stage];
            words[0] = (words[0] & !word0_mask) | word0(sph_type, shader_type);
            words[topology_word] = (words[topology_word] & !mask_in_word(OUTPUT_TOPOLOGY))
                | topology << (OUTPUT_TOPOLOGY.start % 32);
            words
        })
}

proptest! {
    #![proptest_config(config())]

    /// Parse, then encode, then parse: the encoder gives the words back without the ignored
    /// bits, and the parser gives the header back. A swapped pair of reads in the parser
    /// changes the header it builds, so the encoder writes the two values to each other's
    /// places and the words differ.
    #[test]
    fn headers_round_trip(words in valid_words()) {
        let header = parse(&words).unwrap();
        let encoded = encode(&header).unwrap();
        prop_assert_eq!(encoded, without_ignored(header.layout(), words));
        prop_assert_eq!(parse(&encoded), Ok(header));
    }

    #[test]
    fn arbitrary_words_never_panic(words in any::<[u32; HEADER_WORDS]>()) {
        let _ = parse(&words);
    }

    #[test]
    fn arbitrary_programs_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..300)) {
        match ProgramHeader::from_program(&bytes) {
            Ok((_, code)) => prop_assert_eq!(code.len(), bytes.len() - HEADER_BYTES),
            Err(HeaderError::Short { len, .. }) => prop_assert!(len < HEADER_BYTES),
            Err(_) => prop_assert!(bytes.len() >= HEADER_BYTES),
        }
    }
}
