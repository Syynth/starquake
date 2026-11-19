//! Tests of slice 6f, the texture and surface group (`op::texture`): NAK's expected text for the
//! bindless texture instructions, one hand-built word per form and per modifier bit, every
//! dimension, mode and type in hand-built words, the handle forms at the edges of their fields,
//! the values and bits the group refuses, the divergences from PR #4588's table, the encoder's
//! refusals, and generated round trips of the group alone.

use proptest::prelude::*;
use proptest::strategy::Union;

use super::super::op::texture::{self, *};
use super::*;
use crate::asm::{EncodeError, assemble};

/// A hand-built word's opcode and fields, and its expected text.
type Case = (u128, &'static [(u32, u128)], &'static str);

/// The error for `value` in the field at `bit` of `width` bits.
fn unknown(opcode: u16, bit: u32, width: u32, value: u64) -> DecodeError {
    DecodeError::UnknownValue {
        opcode,
        bit,
        width,
        value,
    }
}

/// The operands NAK's `test_texture` gives every case: destinations `r0` and `r2`, sources `r1`
/// and `r3`, the fault predicate `p0` and the mask `0x7`.
fn nak_operands() -> ([Gpr; 2], [Gpr; 2], Pred, ChannelMask) {
    (
        [r(0), r(2)],
        [r(1), r(3)],
        p(0),
        ChannelMask::new(0x7).expect("four bits"),
    )
}

/// `test_texture`'s `TEX`, `TLD` and `TLD4` cases with `.scr` on or off (`scr`).
fn nak_sample_cases(scalar: bool, scr: &str) -> Vec<(Op, String)> {
    let (dsts, srcs, fault, mask) = nak_operands();
    let lod_modes = [
        (TexLodMode::Auto, ""),
        (TexLodMode::Zero, ".lz"),
        (TexLodMode::Lod, ".ll"),
        (TexLodMode::Bias, ".lb"),
        (TexLodMode::Clamp, ".lc"),
        (TexLodMode::BiasClamp, ".lb.lc"),
    ];
    let offset_modes = [
        (TexOffsetMode::None, ""),
        (TexOffsetMode::AddOffI, ".aoffi"),
        (TexOffsetMode::PerPx, ".ptp"),
    ];
    let mut cases: Vec<(Op, String)> = Vec::new();
    for (lod_mode, lod) in lod_modes {
        cases.push((
            Op::Tex(OpTex {
                dsts,
                fault,
                tex: TexRef::Bindless,
                srcs,
                dim: TexDim::D2,
                lod_mode,
                ndv: false,
                z_cmpr: false,
                offset: false,
                f16: false,
                eviction: MemEvictionPriority::First,
                nodep: true,
                channel_mask: mask,
                scalar,
            }),
            format!("tex{scr}.b{lod}.ef.nodep p0, r2, r0, r1, r3, 2d, 0x7;"),
        ));
        if matches!(lod_mode, TexLodMode::Zero | TexLodMode::Lod) {
            cases.push((
                Op::Tld(OpTld {
                    dsts,
                    fault,
                    tex: TexRef::Bindless,
                    srcs,
                    dim: TexDim::D2,
                    is_ms: false,
                    clamp: false,
                    lod_mode,
                    offset: false,
                    f16: false,
                    eviction: MemEvictionPriority::First,
                    nodep: true,
                    channel_mask: mask,
                    scalar,
                }),
                format!("tld{scr}.b{lod}.ef.nodep p0, r2, r0, r1, r3, 2d, 0x7;"),
            ));
        }
    }
    for (offset_mode, offset) in offset_modes {
        cases.push((
            Op::Tld4(OpTld4 {
                dsts,
                fault,
                tex: TexRef::Bindless,
                srcs,
                dim: TexDim::D2,
                comp: TexChannel::G,
                offset_mode,
                z_cmpr: false,
                eviction: MemEvictionPriority::First,
                nodep: true,
                channel_mask: mask,
                scalar,
            }),
            format!("tld4{scr}.g.b{offset}.ef.nodep p0, r2, r0, r1, r3, 2d, 0x7;"),
        ));
    }
    cases
}

/// `test_texture`'s `TMML`, `TXD` and `TXQ` cases.
fn nak_query_cases() -> Vec<(Op, String)> {
    let (dsts, srcs, fault, mask) = nak_operands();
    let mut cases: Vec<(Op, String)> = Vec::new();
    cases.push((
        Op::Tmml(OpTmml {
            dsts,
            tex: TexRef::Bindless,
            srcs,
            dim: TexDim::D2,
            ndv: false,
            nodep: true,
            channel_mask: mask,
        }),
        "tmml.b.lod.nodep r2, r0, r1, r3, 2d, 0x7;".into(),
    ));
    cases.push((
        Op::Txd(OpTxd {
            dsts,
            fault,
            tex: TexRef::Bindless,
            srcs,
            dim: TexDim::D2,
            offset: false,
            eviction: MemEvictionPriority::First,
            nodep: true,
            channel_mask: mask,
        }),
        "txd.b.ef.nodep p0, r2, r0, r1, r3, 2d, 0x7;".into(),
    ));
    for (query, name) in [
        (TexQuery::Dimension, "dimension"),
        (TexQuery::TextureType, "texture_type"),
        (TexQuery::SamplerPos, "sampler_pos"),
    ] {
        cases.push((
            Op::Txq(OpTxq {
                dsts,
                tex: TexRef::Bindless,
                src: r(1),
                query,
                nodep: true,
                channel_mask: mask,
            }),
            format!("txq.b.nodep r2, r0, r1, tex_header_{name}, 0x7;"),
        ));
    }
    cases
}

/// NAK's `test_texture`, ported as a fixture table (`nvdisasm_tests.rs` @ 52238762): the
/// bindless forms with `r0` and `r2` as destinations, `r1` and `r3` as sources, `p0` as the
/// fault predicate, `2d`, the mask `0x7`, `.ef` and `.nodep`; `TEX` with every level-of-detail
/// mode and `TLD` with the explicit ones, scalar and not; `TLD4` gathering green with every
/// offset mode; `TMML`, `TXD`, and `TXQ` with every query. SM 8.6 has every case (the test
/// skips only below SM 7.0, and `BiasClamp` only from SM 10.0).
#[test]
fn texture_text_matches_nak_nvdisasm_cases() {
    let cases: Vec<(Op, String)> = [
        nak_sample_cases(false, ""),
        nak_sample_cases(true, ".scr"),
        nak_query_cases(),
    ]
    .concat();
    // Per scalar setting: six `TEX`, two `TLD` and three `TLD4`; then `TMML`, `TXD` and three
    // `TXQ`.
    assert_eq!(cases.len(), 2 * (6 + 2 + 3) + 1 + 1 + 3);
    for (op, expected) in cases {
        assert_eq!(round_text(Instr::new(op)), expected);
    }
}

/// One hand-built word per form of the group, in form order, and its text. Within each word,
/// every field set above 1 has a value no other field of the word has, but for `RZ` (0xff),
/// which two register fields of a word may share, so a swapped read of two fields fails
/// (`form_cases_give_each_field_a_distinct_value` checks this of the table).
const FORM_CASES: [Case; 22] = [
    // TEX, constant buffer: dst 8, a 9 << 24, b 10 << 32, words 0x10 << 40 (0x40 bytes),
    // index 12 << 54, .scr 60, dim 4 (array_1d) << 61, dst2 11 << 64, mask 6 << 72, .aoffi 76,
    // .ndv 77, .dc 78, .f16 79, fault 5 << 81, evict 2 (.el) << 84, lod 3 (.ll) << 87,
    // .nodep 90.
    (
        0xb60,
        &[
            (16, 8),
            (24, 9),
            (32, 10),
            (40, 0x10),
            (54, 12),
            (60, 1),
            (61, 4),
            (64, 11),
            (72, 6),
            (76, 1),
            (77, 1),
            (78, 1),
            (79, 1),
            (81, 5),
            (84, 2),
            (87, 3),
            (90, 1),
        ],
        "tex.scr.ll.aoffi.ndv.dc.f16.el.nodep p5, r11, r8, r9, r10, c[0xc][0x40], array_1d, 0x6;",
    ),
    // TEX.B: .B 59, dim 7 (array_cube), dst2 RZ, mask 0xf, fault 3, evict 1 (normal).
    (
        0x361,
        &[
            (16, 6),
            (24, 8),
            (32, 9),
            (59, 1),
            (61, 7),
            (64, 0xff),
            (72, 0xf),
            (81, 3),
            (84, 1),
        ],
        "tex.b p3, rz, r6, r8, r9, array_cube, 0xf;",
    ),
    // TEX.UR: ur9 at 40, bit 91, dim 5 (array_2d), lod 1 (.lz), .f16, evict 0 (.ef).
    (
        0xf60,
        &[
            (16, 10),
            (24, 11),
            (32, 12),
            (40, 9),
            (61, 5),
            (64, 13),
            (72, 1),
            (79, 1),
            (81, 0),
            (87, 1),
            (91, 1),
        ],
        "tex.lz.f16.ef p0, r13, r10, r11, r12, ur9, array_2d, 0x1;",
    ),
    // TLD, constant buffer at both field ends: index 31, words 0x3fff (0xfffc bytes);
    // dim 2 (3d), .aoffi, .ms 78, .f16, fault 4, evict 5 (.na), lod 3, .scr, .nodep.
    (
        0xb66,
        &[
            (16, 8),
            (24, 9),
            (32, 10),
            (40, 0x3fff),
            (54, 0x1f),
            (60, 1),
            (61, 2),
            (64, 11),
            (72, 7),
            (76, 1),
            (78, 1),
            (79, 1),
            (81, 4),
            (84, 5),
            (87, 3),
            (90, 1),
        ],
        "tld.scr.ll.aoffi.ms.f16.na.nodep p4, r11, r8, r9, r10, c[0x1f][0xfffc], 3d, 0x7;",
    ),
    // TLD.B: dim 0 (1d), lod 1, evict 3 (.lu), b RZ, dst2 RZ.
    (
        0x367,
        &[
            (16, 5),
            (24, 6),
            (32, 0xff),
            (59, 1),
            (64, 0xff),
            (72, 1),
            (81, 2),
            (84, 3),
            (87, 1),
        ],
        "tld.b.lz.lu p2, rz, r5, r6, rz, 1d, 0x1;",
    ),
    // TLD.UR: ur0, .cl 77, dim 1, fault 2, evict 4 (.eu), lod 3.
    (
        0xf66,
        &[
            (16, 8),
            (24, 9),
            (32, 10),
            (40, 0),
            (61, 1),
            (64, 11),
            (72, 0xc),
            (77, 1),
            (81, 2),
            (84, 4),
            (87, 3),
            (91, 1),
        ],
        "tld.ll.cl.eu p2, r11, r8, r9, r10, ur0, 2d, 0xc;",
    ),
    // TLD4, constant buffer: index 12, words 13 (0x34 bytes), offset mode 2 (.ptp), .dc,
    // comp 3 (.a), dim 7 (array_cube), fault 4, evict 5 (.na), .scr, .nodep.
    (
        0xb63,
        &[
            (16, 8),
            (24, 9),
            (32, 10),
            (40, 13),
            (54, 12),
            (60, 1),
            (61, 7),
            (64, 11),
            (72, 0xf),
            (76, 2),
            (78, 1),
            (81, 4),
            (84, 5),
            (87, 3),
            (90, 1),
        ],
        "tld4.scr.a.ptp.dc.na.nodep p4, r11, r8, r9, r10, c[0xc][0x34], array_cube, 0xf;",
    ),
    // TLD4.B: comp 1 (.g), offset mode 1 (.aoffi), evict 0.
    (
        0x364,
        &[
            (16, 20),
            (24, 21),
            (32, 22),
            (59, 1),
            (61, 1),
            (64, 23),
            (72, 3),
            (76, 1),
            (81, 0),
            (87, 1),
        ],
        "tld4.g.b.aoffi.ef p0, r23, r20, r21, r22, 2d, 0x3;",
    ),
    // TLD4.UR: ur62, comp 0 (.r), dim 3 (cube), evict 1.
    (
        0xf63,
        &[
            (16, 8),
            (24, 9),
            (32, 10),
            (40, 62),
            (61, 3),
            (64, 0xff),
            (72, 1),
            (81, 7),
            (84, 1),
            (91, 1),
        ],
        "tld4.r pt, rz, r8, r9, r10, ur62, cube, 0x1;",
    ),
    // TMML, constant buffer: index 3, words 0x20 (0x80), dim 1, mask 0xc, .ndv 77, .nodep.
    (
        0xb69,
        &[
            (16, 8),
            (24, 9),
            (32, 10),
            (40, 0x20),
            (54, 3),
            (61, 1),
            (64, 11),
            (72, 0xc),
            (77, 1),
            (90, 1),
        ],
        "tmml.lod.ndv.nodep r11, r8, r9, r10, c[0x3][0x80], 2d, 0xc;",
    ),
    // TMML.B: dim 4 (array_1d).
    (
        0x36a,
        &[
            (16, 5),
            (24, 6),
            (32, 7),
            (59, 1),
            (61, 4),
            (64, 0xff),
            (72, 1),
        ],
        "tmml.b.lod rz, r5, r6, r7, array_1d, 0x1;",
    ),
    // TXD, constant buffer: index 4, words 0x12 (0x48), dim 5, .aoffi 76, fault 6, evict 2.
    (
        0xb6c,
        &[
            (16, 8),
            (24, 9),
            (32, 10),
            (40, 0x12),
            (54, 4),
            (61, 5),
            (64, 11),
            (72, 0xf),
            (76, 1),
            (81, 6),
            (84, 2),
            (90, 1),
        ],
        "txd.aoffi.el.nodep p6, r11, r8, r9, r10, c[0x4][0x48], array_2d, 0xf;",
    ),
    // TXD.B: dim 1, fault PT, evict 1.
    (
        0x36d,
        &[
            (16, 8),
            (24, 9),
            (32, 10),
            (59, 1),
            (61, 1),
            (64, 11),
            (72, 3),
            (81, 7),
            (84, 1),
        ],
        "txd.b pt, r11, r8, r9, r10, 2d, 0x3;",
    ),
    // TXQ, constant buffer: index 0, words 0x5a (0x168), query 2 at 62, dst2 RZ, mask 7.
    (
        0xb6f,
        &[
            (16, 8),
            (24, 9),
            (40, 0x5a),
            (62, 2),
            (64, 0xff),
            (72, 7),
            (90, 1),
        ],
        "txq.nodep rz, r8, r9, c[0x0][0x168], tex_header_sampler_pos, 0x7;",
    ),
    // TXQ.B: query 1.
    (
        0x370,
        &[(16, 3), (24, 4), (59, 1), (62, 1), (64, 5), (72, 1)],
        "txq.b r5, r3, r4, tex_header_texture_type, 0x1;",
    ),
    // SULD.P: dst 8, coord 9, dim 3 (2d) at 61, handle 10 at 64, mask 0xf, order 7
    // (.strong.gpu) at 77, fault 0, evict 1.
    (
        0x998,
        &[
            (16, 8),
            (24, 9),
            (61, 3),
            (64, 10),
            (72, 0xf),
            (77, 7),
            (81, 0),
            (84, 1),
        ],
        "suld.p.2d.strong.gpu p0, r8, [r9], r10, 0xf;",
    ),
    // SULD.D: dim 5 (3d), type 6 (.128) at 73, order 10 (.strong.sys), fault PT, evict 0.
    (
        0x99a,
        &[
            (16, 8),
            (24, 9),
            (61, 5),
            (64, 11),
            (73, 6),
            (77, 0xa),
            (81, 7),
            (84, 0),
        ],
        "suld.d.3d.128.strong.sys.ef pt, r8, [r9], r11;",
    ),
    // SUST.P: coord 8, data 9, dim 1 (1d_buffer), handle 10, mask 1, order 4 (.constant),
    // evict 2.
    (
        0x99c,
        &[
            (24, 8),
            (32, 9),
            (61, 1),
            (64, 10),
            (72, 1),
            (77, 4),
            (84, 2),
        ],
        "sust.p.1d_buffer.constant.el [r8], r9, r10, 0x1;",
    ),
    // SUST.D: dim 2 (array_1d), type 3 (.s16), order 5 (.strong.sm), evict 4 (.eu).
    (
        0x99e,
        &[
            (24, 8),
            (32, 9),
            (61, 2),
            (64, 10),
            (73, 3),
            (77, 5),
            (84, 4),
        ],
        "sust.d.array_1d.s16.strong.sm.eu [r8], r9, r10;",
    ),
    // SUATOM: dst 9, coord 10, data 11, dim 4, handle 12, .ba 72, type 5 (.s64), order 0
    // (weak, printed as nothing), fault 1, evict 1, operation 8 (.exch) at 87.
    (
        0x394,
        &[
            (16, 9),
            (24, 10),
            (32, 11),
            (61, 4),
            (64, 12),
            (72, 1),
            (73, 5),
            (81, 1),
            (84, 1),
            (87, 8),
        ],
        "suatom.p.array_2d.exch.s64.ba p1, r9, [r10], r11, r12;",
    ),
    // SUATOM.CAS: type 2 (.64), order 7, fault 4, evict 0.
    (
        0x396,
        &[
            (16, 9),
            (24, 10),
            (32, 11),
            (61, 3),
            (64, 12),
            (73, 2),
            (77, 7),
            (81, 4),
        ],
        "suatom.p.2d.cas.64.strong.gpu.ef p4, r9, [r10], r11, r12;",
    ),
    // SURED: dst RZ (as NAK writes), dim 0, type 3 (.f32.ftz.rn), order 10, fault PT, evict 1,
    // operation 6 (.or).
    (
        0x3a0,
        &[
            (16, 0xff),
            (24, 12),
            (32, 13),
            (64, 14),
            (73, 3),
            (77, 0xa),
            (81, 7),
            (84, 1),
            (87, 6),
        ],
        "sured.p.1d.or.f32.ftz.rn.strong.sys pt, rz, [r12], r13, r14;",
    ),
];

/// Every word of [`FORM_CASES`] gives each field it sets above 1 a value no other field of the
/// word has (`RZ`, 0xff, aside), so a read of one field where another belongs changes the text.
#[test]
fn form_cases_give_each_field_a_distinct_value() {
    for (opcode, fields, _) in FORM_CASES {
        for (i, &(bit, value)) in fields.iter().enumerate() {
            if value <= 1 || value == u128::from(Gpr::RZ.index()) {
                continue;
            }
            for &(other_bit, other) in &fields[i + 1..] {
                assert_ne!(
                    value, other,
                    "{opcode:#x}: the fields at {bit} and {other_bit} share a value"
                );
            }
        }
    }
}

/// One hand-built word per form of the group ([`FORM_CASES`]), and its text (unconfirmed: the
/// module docs say which parts `nvdisasm` has confirmed).
#[test]
fn hand_built_words_of_each_form_decode() {
    let cases = FORM_CASES;
    // One case per form of the group, in form order.
    let opcodes: Vec<u128> = cases.iter().map(|&(opcode, _, _)| opcode).collect();
    let forms: Vec<u128> = texture::FORMS
        .iter()
        .map(|form| u128::from(form.opcode))
        .collect();
    assert_eq!(opcodes, forms);
    for (opcode, fields, expected) in cases {
        let word = word(opcode, fields);
        assert_eq!(text(word), expected, "{opcode:#x}");
        // The assembler gives the hand-built word back.
        let instr = decode(word).expect("decodes");
        assert_eq!(assemble(&instr), Ok(word), "{opcode:#x}");
    }
}

/// `TEX.B r1, r2, r3, 2d`, mask 1, fault `PT`, eviction normal.
const TEX_B: &[(u32, u128)] = &[
    (16, 1),
    (24, 2),
    (32, 3),
    (59, 1),
    (61, 1),
    (64, 0xff),
    (72, 1),
    (81, 7),
    (84, 1),
];
/// `TEX` with the handle `c[0x0][0x0]`, for `.f16`.
const TEX_CB: &[(u32, u128)] = &[
    (16, 1),
    (24, 2),
    (32, 3),
    (61, 1),
    (64, 0xff),
    (72, 1),
    (81, 7),
    (84, 1),
];
/// `TEX.B` with the eviction priority 0 (`.ef`).
const TEX_B_EF: &[(u32, u128)] = &[
    (16, 1),
    (24, 2),
    (32, 3),
    (59, 1),
    (61, 1),
    (64, 0xff),
    (72, 1),
    (81, 7),
];
/// `TLD.UR` with `ur4`, for `.cl`, `.ms` and `.f16`.
const TLD_UR: &[(u32, u128)] = &[
    (16, 1),
    (24, 2),
    (32, 3),
    (40, 4),
    (61, 1),
    (64, 0xff),
    (72, 1),
    (81, 7),
    (84, 1),
    (91, 1),
];
/// `TLD4.R.B` and `TXD.B`, which have the same fields at these values.
const TLD4_B: &[(u32, u128)] = TEX_B;
/// `TMML.B`.
const TMML_B: &[(u32, u128)] = &[
    (16, 1),
    (24, 2),
    (32, 3),
    (59, 1),
    (61, 1),
    (64, 0xff),
    (72, 1),
];
/// `TXQ.B`.
const TXQ_B: &[(u32, u128)] = &[(16, 1), (24, 2), (59, 1), (64, 0xff), (72, 1)];
/// `SUATOM.ADD.U32.WEAK`, fault `PT`, eviction normal.
const SUATOM_PLAIN: &[(u32, u128)] = &[(16, 1), (24, 2), (32, 3), (64, 4), (81, 7), (84, 1)];

/// A plain word's opcode and fields, one more field, and the expected text.
type ModifierCase = (u128, &'static [(u32, u128)], (u32, u128), &'static str);

/// The modifier cases: each plain word with nothing more (`(0, 0)`), then each modifier bit or
/// each bit of a mode field alone.
const MODIFIER_CASES: &[ModifierCase] = &[
    (0x361, TEX_B, (0, 0), "tex.b pt, rz, r1, r2, r3, 2d, 0x1;"),
    (
        0x361,
        TEX_B,
        (60, 1),
        "tex.scr.b pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x361,
        TEX_B,
        (76, 1),
        "tex.b.aoffi pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x361,
        TEX_B,
        (77, 1),
        "tex.b.ndv pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x361,
        TEX_B,
        (78, 1),
        "tex.b.dc pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x361,
        TEX_B,
        (90, 1),
        "tex.b.nodep pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    // The eviction priority 1 plus one bit: 3 (.lu), 5 (.na); and 0 (.ef).
    (
        0x361,
        TEX_B,
        (85, 1),
        "tex.b.lu pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x361,
        TEX_B,
        (86, 1),
        "tex.b.na pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x361,
        TEX_B_EF,
        (0, 0),
        "tex.b.ef pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    // Each bit of the level-of-detail mode alone: 1 (.lz), 2 (.lb), 4 (.lc).
    (
        0x361,
        TEX_B,
        (87, 1),
        "tex.b.lz pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x361,
        TEX_B,
        (88, 1),
        "tex.b.lb pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x361,
        TEX_B,
        (89, 1),
        "tex.b.lc pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0xb60,
        TEX_CB,
        (0, 0),
        "tex pt, rz, r1, r2, r3, c[0x0][0x0], 2d, 0x1;",
    ),
    (
        0xb60,
        TEX_CB,
        (79, 1),
        "tex.f16 pt, rz, r1, r2, r3, c[0x0][0x0], 2d, 0x1;",
    ),
    (
        0xf66,
        TLD_UR,
        (0, 0),
        "tld pt, rz, r1, r2, r3, ur4, 2d, 0x1;",
    ),
    (
        0xf66,
        TLD_UR,
        (77, 1),
        "tld.cl pt, rz, r1, r2, r3, ur4, 2d, 0x1;",
    ),
    (
        0xf66,
        TLD_UR,
        (78, 1),
        "tld.ms pt, rz, r1, r2, r3, ur4, 2d, 0x1;",
    ),
    (
        0xf66,
        TLD_UR,
        (79, 1),
        "tld.f16 pt, rz, r1, r2, r3, ur4, 2d, 0x1;",
    ),
    (
        0xf66,
        TLD_UR,
        (76, 1),
        "tld.aoffi pt, rz, r1, r2, r3, ur4, 2d, 0x1;",
    ),
    (
        0xf66,
        TLD_UR,
        (60, 1),
        "tld.scr pt, rz, r1, r2, r3, ur4, 2d, 0x1;",
    ),
    (
        0x364,
        TLD4_B,
        (0, 0),
        "tld4.r.b pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x364,
        TLD4_B,
        (76, 1),
        "tld4.r.b.aoffi pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x364,
        TLD4_B,
        (77, 1),
        "tld4.r.b.ptp pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x364,
        TLD4_B,
        (78, 1),
        "tld4.r.b.dc pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x364,
        TLD4_B,
        (90, 1),
        "tld4.r.b.nodep pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    // Each bit of the component alone: 1 (.g), 2 (.b).
    (
        0x364,
        TLD4_B,
        (87, 1),
        "tld4.g.b pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x364,
        TLD4_B,
        (88, 1),
        "tld4.b.b pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x364,
        TLD4_B,
        (60, 1),
        "tld4.scr.r.b pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (0x36a, TMML_B, (0, 0), "tmml.b.lod rz, r1, r2, r3, 2d, 0x1;"),
    (
        0x36a,
        TMML_B,
        (77, 1),
        "tmml.b.lod.ndv rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x36a,
        TMML_B,
        (90, 1),
        "tmml.b.lod.nodep rz, r1, r2, r3, 2d, 0x1;",
    ),
    (0x36d, TLD4_B, (0, 0), "txd.b pt, rz, r1, r2, r3, 2d, 0x1;"),
    (
        0x36d,
        TLD4_B,
        (76, 1),
        "txd.b.aoffi pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x36d,
        TLD4_B,
        (90, 1),
        "txd.b.nodep pt, rz, r1, r2, r3, 2d, 0x1;",
    ),
    (
        0x370,
        TXQ_B,
        (0, 0),
        "txq.b rz, r1, r2, tex_header_dimension, 0x1;",
    ),
    (
        0x370,
        TXQ_B,
        (90, 1),
        "txq.b.nodep rz, r1, r2, tex_header_dimension, 0x1;",
    ),
    (
        0x394,
        SUATOM_PLAIN,
        (0, 0),
        "suatom.p.1d.add pt, r1, [r2], r3, r4;",
    ),
    (
        0x394,
        SUATOM_PLAIN,
        (72, 1),
        "suatom.p.1d.add.ba pt, r1, [r2], r3, r4;",
    ),
    (
        0x3a0,
        SUATOM_PLAIN,
        (72, 1),
        "sured.p.1d.add.ba pt, r1, [r2], r3, r4;",
    ),
];

/// Each modifier bit and each bit of a mode field, set alone in a hand-built word, changes the
/// text the way its field says ([`MODIFIER_CASES`]): the plain word of each instruction, then
/// one bit at a time.
#[test]
fn hand_built_words_pin_each_modifier_bit() {
    for &(opcode, base, extra, expected) in MODIFIER_CASES {
        let mut fields = base.to_vec();
        fields.push(extra);
        assert_eq!(
            text(word(opcode, &fields)),
            expected,
            "{opcode:#x} {extra:?}"
        );
    }
}

/// The registers of the group, each moved alone to a value no other register field has: the
/// destinations, sources, fault predicate and uniform handle of the texture instructions, and
/// the destination, coordinates, data and handle of the surface ones.
#[test]
fn hand_built_words_pin_each_register_field() {
    // Every register field 0 (r0, p0, ur0) but the one each case sets.
    let tex = |fields: &[(u32, u128)]| {
        let base = [(61, 1), (72, 1), (84, 1), (91, 1)];
        let all: Vec<(u32, u128)> = base.iter().chain(fields).copied().collect();
        text(word(0xf60, &all))
    };
    let tex_cases = [
        ((16, 9), "tex p0, r0, r9, r0, r0, ur0, 2d, 0x1;"),
        ((24, 9), "tex p0, r0, r0, r9, r0, ur0, 2d, 0x1;"),
        ((32, 9), "tex p0, r0, r0, r0, r9, ur0, 2d, 0x1;"),
        ((64, 9), "tex p0, r9, r0, r0, r0, ur0, 2d, 0x1;"),
        ((81, 3), "tex p3, r0, r0, r0, r0, ur0, 2d, 0x1;"),
        ((40, 63), "tex p0, r0, r0, r0, r0, urz, 2d, 0x1;"),
    ];
    for (field, expected) in tex_cases {
        assert_eq!(tex(&[field]), expected, "{field:?}");
    }
    let suatom = |fields: &[(u32, u128)]| {
        let base = [(84, 1)];
        let all: Vec<(u32, u128)> = base.iter().chain(fields).copied().collect();
        text(word(0x394, &all))
    };
    let suatom_cases = [
        ((16, 9), "suatom.p.1d.add p0, r9, [r0], r0, r0;"),
        ((24, 9), "suatom.p.1d.add p0, r0, [r9], r0, r0;"),
        ((32, 9), "suatom.p.1d.add p0, r0, [r0], r9, r0;"),
        ((64, 9), "suatom.p.1d.add p0, r0, [r0], r0, r9;"),
        ((81, 2), "suatom.p.1d.add p2, r0, [r0], r0, r0;"),
    ];
    for (field, expected) in suatom_cases {
        assert_eq!(suatom(&[field]), expected, "{field:?}");
    }
}

/// Every texture dimension in a hand-built `TEX.B` word and every image dimension in a
/// hand-built `SULD.D` word, each decoding to the dimension of its value, with NAK's encodings
/// written out here.
#[test]
fn every_dimension_decodes() {
    let tex_dims = [
        (0, TexDim::D1, "1d"),
        (4, TexDim::Array1D, "array_1d"),
        (1, TexDim::D2, "2d"),
        (5, TexDim::Array2D, "array_2d"),
        (2, TexDim::D3, "3d"),
        (3, TexDim::Cube, "cube"),
        (7, TexDim::ArrayCube, "array_cube"),
    ];
    assert_eq!(tex_dims.len(), TexDim::ALL.len());
    let tex_dim = |word| match decode(word).map(|instr| instr.op) {
        Ok(Op::Tex(tex)) => Some(tex.dim),
        _ => None,
    };
    for (bits, dim, name) in tex_dims {
        let word = word(0x361, &[(59, 1), (61, bits), (81, 7), (84, 1)]);
        assert_eq!(tex_dim(word), Some(dim));
        assert_eq!(
            text(word),
            format!("tex.b pt, r0, r0, r0, r0, {name}, 0x0;")
        );
    }
    let image_dims = [
        (0, ImageDim::D1, "1d"),
        (1, ImageDim::Buffer1D, "1d_buffer"),
        (2, ImageDim::Array1D, "array_1d"),
        (3, ImageDim::D2, "2d"),
        (4, ImageDim::Array2D, "array_2d"),
        (5, ImageDim::D3, "3d"),
    ];
    assert_eq!(image_dims.len(), ImageDim::ALL.len());
    let image_dim = |word| match decode(word).map(|instr| instr.op) {
        Ok(Op::SuLd(suld)) => Some(suld.dim),
        _ => None,
    };
    for (bits, dim, name) in image_dims {
        let word = word(0x99a, &[(61, bits), (81, 7), (84, 1)]);
        assert_eq!(image_dim(word), Some(dim));
        assert_eq!(text(word), format!("suld.d.{name}.u8 pt, r0, [r0], r0;"));
    }
}

/// Every value of the texture instructions' other enumerated fields, in hand-built words: the
/// level-of-detail modes, eviction priorities, `TLD4`'s components and offset modes, and
/// `TXQ`'s queries.
#[test]
fn every_texture_mode_decodes() {
    let tex = |field: (u32, u128)| text(word(0x361, &[(59, 1), (81, 7), (84, 1), field]));
    for (bits, text) in [
        (0, ""),
        (1, ".lz"),
        (2, ".lb"),
        (3, ".ll"),
        (4, ".lc"),
        (5, ".lb.lc"),
    ] {
        assert_eq!(
            tex((87, bits)),
            format!("tex.b{text} pt, r0, r0, r0, r0, 1d, 0x0;")
        );
    }
    for (bits, text) in [
        (0, ".ef"),
        (1, ""),
        (2, ".el"),
        (3, ".lu"),
        (4, ".eu"),
        (5, ".na"),
    ] {
        let word = word(0x361, &[(59, 1), (81, 7), (84, bits)]);
        assert_eq!(
            text_of(word),
            format!("tex.b{text} pt, r0, r0, r0, r0, 1d, 0x0;")
        );
    }
    let tld4 = |fields: &[(u32, u128)]| {
        let all: Vec<(u32, u128)> = [(59, 1), (81, 7), (84, 1)]
            .iter()
            .chain(fields)
            .copied()
            .collect();
        text(word(0x364, &all))
    };
    for (bits, comp) in [(0, ".r"), (1, ".g"), (2, ".b"), (3, ".a")] {
        assert_eq!(
            tld4(&[(87, bits)]),
            format!("tld4{comp}.b pt, r0, r0, r0, r0, 1d, 0x0;")
        );
    }
    for (bits, mode) in [(0, ""), (1, ".aoffi"), (2, ".ptp")] {
        assert_eq!(
            tld4(&[(76, bits)]),
            format!("tld4.r.b{mode} pt, r0, r0, r0, r0, 1d, 0x0;")
        );
    }
    for (bits, query) in [(0, "dimension"), (1, "texture_type"), (2, "sampler_pos")] {
        assert_eq!(
            text(word(0x370, &[(59, 1), (62, bits)])),
            format!("txq.b r0, r0, r0, tex_header_{query}, 0x0;")
        );
    }
}

/// Every value of the surface instructions' enumerated fields, in hand-built words: memory
/// types and orders, atomic types, and the atomic operations of `SUATOM` and `SURED`.
#[test]
fn every_surface_type_order_and_operation_decodes() {
    let suld = |fields: &[(u32, u128)]| {
        let all: Vec<(u32, u128)> = [(81, 7), (84, 1)].iter().chain(fields).copied().collect();
        text(word(0x99a, &all))
    };
    for (bits, mem_type) in [
        (0, ".u8"),
        (1, ".s8"),
        (2, ".u16"),
        (3, ".s16"),
        (4, ""),
        (5, ".64"),
        (6, ".128"),
    ] {
        assert_eq!(
            suld(&[(73, bits)]),
            format!("suld.d.1d{mem_type} pt, r0, [r0], r0;")
        );
    }
    for (bits, order) in [
        (0x0, ""),
        (0x4, ".constant"),
        (0x5, ".strong.sm"),
        (0x7, ".strong.gpu"),
        (0xa, ".strong.sys"),
    ] {
        assert_eq!(
            suld(&[(77, bits)]),
            format!("suld.d.1d.u8{order} pt, r0, [r0], r0;")
        );
    }
    let suatom = |opcode: u128, fields: &[(u32, u128)]| {
        let all: Vec<(u32, u128)> = [(81, 7), (84, 1)].iter().chain(fields).copied().collect();
        text(word(opcode, &all))
    };
    for (bits, atom_type) in [
        (0, ""),
        (1, ".s32"),
        (2, ".64"),
        (3, ".f32.ftz.rn"),
        (4, ".f16x2.rn"),
        (5, ".s64"),
        (6, ".f64.rn"),
    ] {
        assert_eq!(
            suatom(0x394, &[(73, bits)]),
            format!("suatom.p.1d.add{atom_type} pt, r0, [r0], r0, r0;")
        );
    }
    let ops = [
        (0, ".add"),
        (1, ".min"),
        (2, ".max"),
        (3, ".inc"),
        (4, ".dec"),
        (5, ".and"),
        (6, ".or"),
        (7, ".xor"),
        (8, ".exch"),
    ];
    for (bits, op) in ops {
        assert_eq!(
            suatom(0x394, &[(87, bits)]),
            format!("suatom.p.1d{op} pt, r0, [r0], r0, r0;")
        );
        // `SURED`'s three bits hold the reductions, 0 to 7.
        if bits < 8 {
            assert_eq!(
                suatom(0x3a0, &[(87, bits)]),
                format!("sured.p.1d{op} pt, r0, [r0], r0, r0;")
            );
        }
    }
}

/// Decodes and prints `word`, as [`text`] does (a name that does not shadow the loop's).
fn text_of(word: Instruction) -> String {
    text(word)
}

/// The handle fields at both edges: buffer 0 and 31, word offsets 0 and `0x3fff` (bytes 0 and
/// `0xfffc`), uniform registers 0 and `URZ`; and the assembler at and one past each edge.
#[test]
fn handles_at_the_edges_of_their_fields() {
    let last_index = (1_u8 << TEX_CBUF_INDEX.width) - 1;
    let last_offset = ((1_u16 << TEX_CBUF_OFFSET.width) - 1) * TEX_CBUF_UNIT;
    assert_eq!((last_index, last_offset), (31, 0xfffc));
    let txq = |tex| {
        Instr::new(Op::Txq(OpTxq {
            dsts: [r(1), Gpr::RZ],
            tex,
            src: r(2),
            query: TexQuery::Dimension,
            nodep: false,
            channel_mask: ChannelMask::RGBA,
        }))
    };
    let cbuf = |index, offset| TexRef::CBuf(TexCBufRef { index, offset });
    let cases = [
        (
            cbuf(0, 0),
            "txq rz, r1, r2, c[0x0][0x0], tex_header_dimension, 0xf;",
        ),
        (
            cbuf(last_index, last_offset),
            "txq rz, r1, r2, c[0x1f][0xfffc], tex_header_dimension, 0xf;",
        ),
    ];
    for (tex, expected) in cases {
        assert_eq!(round_text(txq(tex)), expected);
    }
    // The same edges in hand-built words.
    assert_eq!(
        text(word(
            0xb6f,
            &[
                (16, 1),
                (24, 2),
                (40, 0x3fff),
                (54, 31),
                (64, 0xff),
                (72, 0xf)
            ]
        )),
        "txq rz, r1, r2, c[0x1f][0xfffc], tex_header_dimension, 0xf;"
    );
    let tex = |handle| {
        Instr::new(Op::Tex(OpTex {
            dsts: [r(1), Gpr::RZ],
            fault: Pred::PT,
            tex: TexRef::Uniform(handle),
            srcs: [r(2), r(3)],
            dim: TexDim::D2,
            lod_mode: TexLodMode::Auto,
            ndv: false,
            z_cmpr: false,
            offset: false,
            f16: false,
            eviction: MemEvictionPriority::Normal,
            nodep: false,
            channel_mask: ChannelMask::RGBA,
            scalar: false,
        }))
    };
    assert_eq!(
        round_text(tex(ur(0))),
        "tex pt, rz, r1, r2, r3, ur0, 2d, 0xf;"
    );
    assert_eq!(
        round_text(tex(UniformReg::URZ)),
        "tex pt, rz, r1, r2, r3, urz, 2d, 0xf;"
    );
    // One past the buffer index; an offset one byte either side of a word.
    assert_eq!(
        assemble(&txq(cbuf(last_index + 1, 0))),
        Err(EncodeError::TooWide {
            what: "texture handle buffer",
            value: u64::from(last_index) + 1,
            bits: TEX_CBUF_INDEX.width
        })
    );
    for offset in [1, last_offset - 1, last_offset + 1] {
        assert_eq!(
            assemble(&txq(cbuf(0, offset))),
            Err(EncodeError::Misaligned {
                what: "texture handle offset",
                value: u64::from(offset),
                align: u64::from(TEX_CBUF_UNIT)
            })
        );
    }
    assert_eq!(
        assemble(&txq(cbuf(0, 2))).map_err(|error| error.to_string()),
        Err("texture handle offset 0x2 is not a multiple of 4".into())
    );
}

/// The values the texture instructions refuse, each alone, one below each decoding, and two
/// refused values at once, where the lower field is the one named.
#[test]
fn values_the_texture_instructions_refuse() {
    let tex = |fields: &[(u32, u128)]| {
        let all: Vec<(u32, u128)> = [(59, 1)].iter().chain(fields).copied().collect();
        decode(word(0x361, &all))
    };
    assert_eq!(tex(&[(61, 6)]), Err(unknown(0x361, 61, 3, 6)));
    assert!(tex(&[(61, 7)]).is_ok());
    assert!(tex(&[(61, 5)]).is_ok());
    for value in [6, 7] {
        assert_eq!(
            tex(&[(84, value)]),
            Err(unknown(0x361, 84, 3, value.try_into().expect("small")))
        );
        assert_eq!(
            tex(&[(87, value)]),
            Err(unknown(0x361, 87, 3, value.try_into().expect("small")))
        );
    }
    assert!(tex(&[(84, 5), (87, 5)]).is_ok());
    // Two refused at once: the dimension before the eviction priority before the mode.
    assert_eq!(
        tex(&[(61, 6), (84, 6), (87, 6)]),
        Err(unknown(0x361, 61, 3, 6))
    );
    assert_eq!(tex(&[(84, 7), (87, 6)]), Err(unknown(0x361, 84, 3, 7)));
    // TLD, through its constant-buffer form: the same three fields, in the same order.
    let tld = |fields: &[(u32, u128)]| decode(word(0xb66, fields));
    assert_eq!(tld(&[(87, 7)]), Err(unknown(0xb66, 87, 3, 7)));
    assert_eq!(tld(&[(84, 6), (87, 7)]), Err(unknown(0xb66, 84, 3, 6)));
    assert_eq!(tld(&[(61, 6), (84, 6)]), Err(unknown(0xb66, 61, 3, 6)));
    // TLD4: the dimension, then the offset mode, then the eviction priority.
    let tld4 = |fields: &[(u32, u128)]| decode(word(0xb63, fields));
    assert_eq!(tld4(&[(76, 3)]), Err(unknown(0xb63, 76, 2, 3)));
    assert!(tld4(&[(76, 2)]).is_ok());
    assert_eq!(tld4(&[(61, 6), (76, 3)]), Err(unknown(0xb63, 61, 3, 6)));
    assert_eq!(tld4(&[(76, 3), (84, 6)]), Err(unknown(0xb63, 76, 2, 3)));
    assert_eq!(tld4(&[(84, 6)]), Err(unknown(0xb63, 84, 3, 6)));
    // TMML: the dimension.
    assert_eq!(
        decode(word(0xb69, &[(61, 6)])),
        Err(unknown(0xb69, 61, 3, 6))
    );
    // TXD: the dimension before the eviction priority.
    let txd = |fields: &[(u32, u128)]| decode(word(0xb6c, fields));
    assert_eq!(txd(&[(84, 7)]), Err(unknown(0xb6c, 84, 3, 7)));
    assert_eq!(txd(&[(61, 6), (84, 7)]), Err(unknown(0xb6c, 61, 3, 6)));
    // TXQ: the query.
    assert_eq!(
        decode(word(0xb6f, &[(62, 3)])),
        Err(unknown(0xb6f, 62, 2, 3))
    );
    assert!(decode(word(0xb6f, &[(62, 2)])).is_ok());
    assert_eq!(
        tex(&[(61, 6)]).map_err(|error| error.to_string()),
        Err(
            "opcode 0x361 has the value 6 in its 3-bit field at bit 61, which its form does not \
             define"
                .into()
        )
    );
}

/// The values the surface instructions refuse, each alone, one below each decoding, and two
/// refused values at once, where the lower field is the one named.
#[test]
fn values_the_surface_instructions_refuse() {
    // SULD.D: the dimension, then the memory type, then the order, then the eviction priority.
    let suld = |fields: &[(u32, u128)]| decode(word(0x99a, fields));
    for value in [6, 7] {
        assert_eq!(
            suld(&[(61, value)]),
            Err(unknown(0x99a, 61, 3, value.try_into().expect("small")))
        );
    }
    assert!(suld(&[(61, 5)]).is_ok());
    assert_eq!(suld(&[(73, 7)]), Err(unknown(0x99a, 73, 3, 7)));
    assert!(suld(&[(73, 6)]).is_ok());
    assert_eq!(suld(&[(84, 6)]), Err(unknown(0x99a, 84, 3, 6)));
    assert_eq!(suld(&[(61, 6), (73, 7)]), Err(unknown(0x99a, 61, 3, 6)));
    assert_eq!(suld(&[(73, 7), (77, 1)]), Err(unknown(0x99a, 73, 3, 7)));
    assert_eq!(suld(&[(77, 1), (84, 6)]), Err(unknown(0x99a, 77, 4, 1)));
    // Every memory order: the five NAK encodes decode, the other eleven are refused.
    let orders = [0x0, 0x4, 0x5, 0x7, 0xa];
    for value in 0..16_u64 {
        let result = suld(&[(77, value.into())]);
        if orders.contains(&value) {
            assert!(result.is_ok(), "{value:#x}");
        } else {
            assert_eq!(result, Err(unknown(0x99a, 77, 4, value)), "{value:#x}");
        }
    }
    // SULD.P and SUST.P read a mask, not a memory type: any four bits.
    for mask in 0..16 {
        assert!(decode(word(0x998, &[(72, mask)])).is_ok());
        assert!(decode(word(0x99c, &[(72, mask)])).is_ok());
    }
    // SUST.D: the dimension, then the memory type, then the order, then the eviction
    // priority, each alone and with the next one at once.
    let sust = |fields: &[(u32, u128)]| decode(word(0x99e, fields));
    assert_eq!(sust(&[(61, 6)]), Err(unknown(0x99e, 61, 3, 6)));
    assert_eq!(sust(&[(73, 7)]), Err(unknown(0x99e, 73, 3, 7)));
    assert_eq!(sust(&[(77, 1)]), Err(unknown(0x99e, 77, 4, 1)));
    assert_eq!(sust(&[(84, 6)]), Err(unknown(0x99e, 84, 3, 6)));
    assert_eq!(sust(&[(61, 6), (73, 7)]), Err(unknown(0x99e, 61, 3, 6)));
    assert_eq!(sust(&[(73, 7), (77, 1)]), Err(unknown(0x99e, 73, 3, 7)));
    assert_eq!(sust(&[(77, 1), (84, 6)]), Err(unknown(0x99e, 77, 4, 1)));
    // SUST.P, which has no memory type: the dimension, then the order, then the eviction
    // priority.
    let sust_p = |fields: &[(u32, u128)]| decode(word(0x99c, fields));
    assert_eq!(sust_p(&[(61, 6), (77, 1)]), Err(unknown(0x99c, 61, 3, 6)));
    assert_eq!(sust_p(&[(77, 1), (84, 6)]), Err(unknown(0x99c, 77, 4, 1)));
    // SUATOM: the dimension, the type, the order, the eviction priority, then the operation.
    let suatom = |fields: &[(u32, u128)]| decode(word(0x394, fields));
    assert_eq!(suatom(&[(73, 7)]), Err(unknown(0x394, 73, 3, 7)));
    assert!(suatom(&[(73, 6)]).is_ok());
    for value in 9..16_u64 {
        assert_eq!(
            suatom(&[(87, value.into())]),
            Err(unknown(0x394, 87, 4, value))
        );
    }
    assert!(suatom(&[(87, 8)]).is_ok());
    assert_eq!(suatom(&[(84, 6), (87, 9)]), Err(unknown(0x394, 84, 3, 6)));
    assert_eq!(suatom(&[(77, 1), (84, 6)]), Err(unknown(0x394, 77, 4, 1)));
    assert_eq!(suatom(&[(73, 7), (77, 1)]), Err(unknown(0x394, 73, 3, 7)));
    assert_eq!(suatom(&[(61, 7), (73, 7)]), Err(unknown(0x394, 61, 3, 7)));
    assert_eq!(
        decode(word(0x396, &[(73, 7)])),
        Err(unknown(0x396, 73, 3, 7))
    );
    assert_eq!(
        decode(word(0x3a0, &[(61, 6)])),
        Err(unknown(0x3a0, 61, 3, 6))
    );
    assert!(suld(&[(84, 5)]).is_ok());
}

/// The `.B` forms require bit 59 and the `.UR` forms bit 91, naming it when clear; the
/// constant-buffer forms have neither (bit 59 is outside their masks).
#[test]
fn handle_forms_require_their_fixed_bit() {
    for opcodes in [
        TEX_OPCODES,
        TLD_OPCODES,
        TLD4_OPCODES,
        TMML_OPCODES,
        TXD_OPCODES,
        TXQ_OPCODES,
    ] {
        let bindless = opcodes.bindless;
        assert_eq!(
            decode(word(bindless.into(), &[])),
            Err(DecodeError::MissingBit {
                opcode: bindless,
                bit: 59
            })
        );
        assert!(decode(word(bindless.into(), &[(59, 1)])).is_ok());
        let cbuf = opcodes.cbuf;
        assert!(decode(word(cbuf.into(), &[])).is_ok());
        assert_eq!(
            decode(word(cbuf.into(), &[(59, 1)])),
            Err(DecodeError::UnknownBit {
                opcode: cbuf,
                bit: 59
            })
        );
        if let Some(uniform) = opcodes.uniform {
            assert_eq!(
                decode(word(uniform.into(), &[])),
                Err(DecodeError::MissingBit {
                    opcode: uniform,
                    bit: 91
                })
            );
            assert!(decode(word(uniform.into(), &[(91, 1)])).is_ok());
            // The handle is six bits: 46 is outside the mask.
            assert_eq!(
                decode(word(uniform.into(), &[(46, 1), (91, 1)])),
                Err(DecodeError::UnknownBit {
                    opcode: uniform,
                    bit: 46
                })
            );
        }
    }
}

/// `.F16` (bit 79) decodes in every form of `TEX` and `TLD` and `.CL` (bit 77) in every form of
/// `TLD`, the forms adding them where PR #4588's rows leave them out (`TEX.B`'s bit 79, named by
/// PR #4588's `Visitor::TEX`; `TLD`'s constant-buffer and `.B` bit 77 and `TLD.B`'s bit 79,
/// named by NAK's `OpTld` comments). `TLD4`, whose rows have no bit 79 and which neither
/// reference names a `.F16` for, refuses it (module docs). Each decoded word prints the flag and
/// re-assembles to itself.
#[test]
fn f16_and_clamp_in_every_form_that_names_them() {
    let handle_bits = |opcode: u16| -> Vec<(u32, u128)> {
        let fixed = lookup(opcode).expect("a form").fixed;
        (0..128)
            .filter(|bit| fixed & (1 << bit) != 0)
            .map(|bit| (bit, 1))
            .collect()
    };
    let with = |opcode: u16, bit: u32| {
        let mut fields = handle_bits(opcode);
        fields.push((bit, 1));
        let word = word(opcode.into(), &fields);
        (word, decode(word))
    };
    for (opcode, bit, flag) in [
        (0xb60, 79, ".f16"),
        (0x361, 79, ".f16"),
        (0xf60, 79, ".f16"),
        (0xb66, 79, ".f16"),
        (0x367, 79, ".f16"),
        (0xf66, 79, ".f16"),
        (0xb66, 77, ".cl"),
        (0x367, 77, ".cl"),
        (0xf66, 77, ".cl"),
    ] {
        let (word, decoded) = with(opcode, bit);
        let instr = decoded.unwrap_or_else(|error| panic!("{opcode:#x} bit {bit}: {error}"));
        let text = instr.to_string();
        assert!(text.contains(flag), "{opcode:#x}: {text}");
        let plain = decode(Instruction::from_bits(word.bits() & !(1 << bit)));
        let plain_text = plain.expect("decodes").to_string();
        assert!(!plain_text.contains(flag), "{opcode:#x}: {plain_text}");
        assert_eq!(assemble(&instr), Ok(word), "{opcode:#x} bit {bit}");
    }
    for opcode in [0xb63, 0x364, 0xf63] {
        assert_eq!(
            with(opcode, 79).1,
            Err(DecodeError::UnknownBit { opcode, bit: 79 }),
            "{opcode:#x}"
        );
    }
    // The masks: bit 79 is known in exactly the six forms, bit 77 of TLD in all three.
    let known = |opcode: u16, bit: u32| lookup(opcode).expect("a form").known & (1 << bit) != 0;
    // Of the texture forms (the surface forms' bit 79 is their memory order's).
    let texture_forms = [
        TEX_OPCODES,
        TLD_OPCODES,
        TLD4_OPCODES,
        TMML_OPCODES,
        TXD_OPCODES,
        TXQ_OPCODES,
    ]
    .into_iter()
    .flat_map(|opcodes| [Some(opcodes.cbuf), Some(opcodes.bindless), opcodes.uniform])
    .flatten();
    let f16_forms: Vec<u16> = texture_forms.filter(|&opcode| known(opcode, 79)).collect();
    assert_eq!(f16_forms, [0xb60, 0x361, 0xf60, 0xb66, 0x367, 0xf66]);
    assert!(known(0xb66, 77) && known(0x367, 77) && known(0xf66, 77));
}

/// The rows' own bits, before the forms add any: PR #4588's rows mark bit 79 known in the
/// constant-buffer and `.UR` forms of `TEX` and `TLD` only, and bit 77 of `TLD` in its `.UR`
/// form only, so the added bits are exactly the others (`framework::added_bits_are_the_listed_cases`).
#[test]
fn rows_leave_out_the_added_texture_bits() {
    let row_has = |opcode: u16, bit: u32| {
        texture::REFERENCE_ROWS
            .iter()
            .find(|row| row.opcode == opcode)
            .expect("a row")
            .mask()
            & (1 << bit)
            != 0
    };
    for opcode in [0xb60, 0xf60, 0xb66, 0xf66] {
        assert!(row_has(opcode, 79), "{opcode:#x}");
    }
    for opcode in [0x361, 0x367, 0xb63, 0x364, 0xf63] {
        assert!(!row_has(opcode, 79), "{opcode:#x}");
    }
    assert!(row_has(0xf66, 77) && !row_has(0xb66, 77) && !row_has(0x367, 77));
}

/// The assembler writes `.F16` with every handle of `TEX` and `TLD`, and `.CL` with every handle
/// of `TLD` (`f16_and_clamp_in_every_form_that_names_them`): at exactly bits 79 and 77, each
/// word decoding back to the op.
#[test]
fn encoder_writes_f16_and_clamp_with_every_handle() {
    let tex = |tex, f16| {
        assemble(&Instr::new(Op::Tex(OpTex {
            dsts: [r(1), Gpr::RZ],
            fault: Pred::PT,
            tex,
            srcs: [r(2), r(3)],
            dim: TexDim::D2,
            lod_mode: TexLodMode::Auto,
            ndv: false,
            z_cmpr: false,
            offset: false,
            f16,
            eviction: MemEvictionPriority::Normal,
            nodep: false,
            channel_mask: ChannelMask::RGBA,
            scalar: false,
        })))
    };
    let cbuf = TexRef::CBuf(TexCBufRef {
        index: 1,
        offset: 8,
    });
    let handles = [cbuf, TexRef::Bindless, TexRef::Uniform(ur(2))];
    for handle in handles {
        let on = tex(handle, true).expect("encodable");
        let off = tex(handle, false).expect("encodable");
        assert_eq!(on.bits() ^ off.bits(), 1 << 79, "{handle:?}");
        let Ok(Instr {
            op: Op::Tex(decoded),
            ..
        }) = decode(on)
        else {
            panic!("{handle:?}: a TEX word");
        };
        assert!(decoded.f16 && decoded.tex == handle, "{handle:?}");
    }
    let tld = |tex, f16, clamp| {
        assemble(&Instr::new(Op::Tld(OpTld {
            dsts: [r(1), Gpr::RZ],
            fault: Pred::PT,
            tex,
            srcs: [r(2), r(3)],
            dim: TexDim::D2,
            is_ms: false,
            clamp,
            lod_mode: TexLodMode::Zero,
            offset: false,
            f16,
            eviction: MemEvictionPriority::Normal,
            nodep: false,
            channel_mask: ChannelMask::RGBA,
            scalar: false,
        })))
    };
    for handle in handles {
        let plain = tld(handle, false, false).expect("encodable");
        for (f16, clamp, bit) in [(true, false, 79), (false, true, 77)] {
            let word = tld(handle, f16, clamp).expect("encodable");
            assert_eq!(word.bits() ^ plain.bits(), 1 << bit, "{handle:?}");
            let Ok(Instr {
                op: Op::Tld(decoded),
                ..
            }) = decode(word)
            else {
                panic!("{handle:?}: a TLD word");
            };
            assert_eq!((decoded.f16, decoded.clamp), (f16, clamp), "{handle:?}");
            assert_eq!(decoded.tex, handle);
        }
    }
}

/// `TLD4.UR` rejects bit 89, which PR #4588's row marks known (module docs); it is the only bit
/// the group drops.
#[test]
fn tld4_ur_rejects_bit_89() {
    let dropping: Vec<(u16, u128)> = texture::REFERENCE_ROWS
        .iter()
        .filter(|row| row.dropped != 0)
        .map(|row| (row.opcode, row.dropped))
        .collect();
    assert_eq!(dropping, [(0xf63, 1 << 89)]);
    let plain = word(0xf63, &[(91, 1)]);
    assert!(decode(plain).is_ok());
    assert_eq!(
        decode(plain.with_flag(Field::bit(89), true)),
        Err(DecodeError::UnknownBit {
            opcode: 0xf63,
            bit: 89
        })
    );
    // The component's two bits below it decode.
    assert!(decode(plain.with_field(Field::new(87, 2), 3)).is_ok());
}

/// The surface forms with a constant-buffer handle, which PR #4588's table has rows for, are
/// unknown opcodes (module docs); so are the neighbours of the group's opcodes no reference
/// names.
#[test]
fn surface_rows_left_undecoded_are_unknown_opcodes() {
    for opcode in [
        0xb97, 0xb9b, 0x593, 0x360, 0x362, 0xb61, 0xf61, 0x395, 0x3a2, 0x999, 0x99f,
    ] {
        let word = Instruction::default().with_field(OPCODE, u64::from(opcode));
        assert_eq!(decode(word), Err(DecodeError::UnknownOpcode { opcode }));
    }
}

/// `SURED` and the compare-and-swap form are forms of `SUATOM`: one operation, three opcodes.
#[test]
fn sured_and_cas_are_forms_of_suatom() {
    let op = |word| decode(word).map(|instr| instr.op);
    let fields = |word| match op(word) {
        Ok(Op::SuAtom(su)) => Some((su.atom_op, su.reduce, su.dst)),
        _ => None,
    };
    let min = |opcode: u128| word(opcode, &[(16, 0xff), (87, 1)]);
    assert_eq!(fields(min(0x3a0)), Some((AtomOp::Min, true, Gpr::RZ)));
    assert_eq!(fields(min(0x394)), Some((AtomOp::Min, false, Gpr::RZ)));
    assert_eq!(op(min(0x3a0)).map(|op| op.mnemonic()), Ok("suatom"));
    // The compare-and-swap form has no operation field: bit 87 is unknown there.
    assert_eq!(
        op(min(0x396)),
        Err(DecodeError::UnknownBit {
            opcode: 0x396,
            bit: 87
        })
    );
    assert_eq!(
        fields(word(0x396, &[(16, 0xff)])),
        Some((AtomOp::CmpExch(AtomCmpSrc::Packed), false, Gpr::RZ))
    );
    // SURED's operation is three bits: bit 90 is outside its mask.
    assert_eq!(
        decode(word(0x3a0, &[(90, 1)])),
        Err(DecodeError::UnknownBit {
            opcode: 0x3a0,
            bit: 90
        })
    );
}

/// The assembler refuses what the forms cannot hold, with plain-word messages.
#[test]
fn unencodable_texture_instructions_are_errors() {
    let tmml = Op::Tmml(OpTmml {
        dsts: [r(1), Gpr::RZ],
        tex: TexRef::Uniform(ur(2)),
        srcs: [r(2), r(3)],
        dim: TexDim::D2,
        ndv: false,
        nodep: false,
        channel_mask: ChannelMask::RGBA,
    });
    assert_eq!(
        assemble(&Instr::new(tmml)).map_err(|error| error.to_string()),
        Err("tmml has no bits for a uniform-register texture handle".into())
    );
    let txd = Op::Txd(OpTxd {
        dsts: [r(1), Gpr::RZ],
        fault: Pred::PT,
        tex: TexRef::Uniform(ur(2)),
        srcs: [r(2), r(3)],
        dim: TexDim::D2,
        offset: false,
        eviction: MemEvictionPriority::Normal,
        nodep: false,
        channel_mask: ChannelMask::RGBA,
    });
    assert_eq!(
        assemble(&Instr::new(txd)),
        Err(EncodeError::NoBits {
            mnemonic: "txd",
            what: "a uniform-register texture handle"
        })
    );
    let txq = Op::Txq(OpTxq {
        dsts: [r(1), Gpr::RZ],
        tex: TexRef::Uniform(ur(2)),
        src: r(2),
        query: TexQuery::Dimension,
        nodep: false,
        channel_mask: ChannelMask::RGBA,
    });
    assert_eq!(
        assemble(&Instr::new(txq)),
        Err(EncodeError::NoBits {
            mnemonic: "txq",
            what: "a uniform-register texture handle"
        })
    );
    let suatom = |atom_op, reduce| {
        assemble(&Instr::new(Op::SuAtom(OpSuAtom {
            dst: r(1),
            fault: Pred::PT,
            dim: ImageDim::D2,
            atom_op,
            atom_type: AtomType::U32,
            order: MemOrder::Weak,
            eviction: MemEvictionPriority::Normal,
            handle: r(4),
            coord: r(2),
            data: r(3),
            ba: false,
            reduce,
        })))
    };
    assert_eq!(
        suatom(AtomOp::CmpExch(AtomCmpSrc::Separate { cmpr: r(5) }), false),
        Err(EncodeError::NoBits {
            mnemonic: "suatom",
            what: "a separate compare source"
        })
    );
    assert_eq!(
        suatom(AtomOp::CmpExch(AtomCmpSrc::Packed), true),
        Err(EncodeError::NoBits {
            mnemonic: "suatom",
            what: "a compare-and-swap without a result"
        })
    );
    assert_eq!(
        suatom(AtomOp::Exch, true),
        Err(EncodeError::TooWide {
            what: "atomic operation",
            value: 8,
            bits: SURED_OP.width
        })
    );
    assert!(suatom(AtomOp::Xor, true).is_ok());
    assert!(suatom(AtomOp::Exch, false).is_ok());
    assert!(suatom(AtomOp::CmpExch(AtomCmpSrc::Packed), false).is_ok());
    assert_eq!(
        EncodeError::Misaligned {
            what: "texture handle offset",
            value: 6,
            align: 4
        }
        .to_string(),
        "texture handle offset 0x6 is not a multiple of 4"
    );
}

/// Channel masks hold four bits: 15 is the last, 16 is refused.
#[test]
fn channel_masks_hold_four_bits() {
    let last = (1_u8 << CHANNEL_MASK.width) - 1;
    assert_eq!(ChannelMask::new(last), Some(ChannelMask::RGBA));
    assert_eq!(ChannelMask::new(last + 1), None);
    assert_eq!(ChannelMask::new(0).map(ChannelMask::bits), Some(0));
    assert_eq!(ChannelMask::RGBA.bits(), 0xf);
}

/// The group's opcodes, field positions and encodings against the references' values, written
/// out here: NAK's `set_opcode`, field ranges and encoding tables (`sm70_encode.rs` @ 52238762)
/// and PR #4588's `.UR` opcodes and bits (`ampere_isa_table.h`, `ops_texture.cpp` @ e5fb3b97).
#[test]
fn layout_constants_match_the_references() {
    let tex_opcodes = [
        (TEX_OPCODES, (0xb60, 0x361, Some(0xf60))),
        (TLD_OPCODES, (0xb66, 0x367, Some(0xf66))),
        (TLD4_OPCODES, (0xb63, 0x364, Some(0xf63))),
        (TMML_OPCODES, (0xb69, 0x36a, None)),
        (TXD_OPCODES, (0xb6c, 0x36d, None)),
        (TXQ_OPCODES, (0xb6f, 0x370, None)),
    ];
    for (opcodes, (cbuf, bindless, uniform)) in tex_opcodes {
        assert_eq!(
            (opcodes.cbuf, opcodes.bindless, opcodes.uniform),
            (cbuf, bindless, uniform)
        );
    }
    assert_eq!(
        (TEX_UR_OPCODE, TLD_UR_OPCODE, TLD4_UR_OPCODE),
        (0xf60, 0xf66, 0xf63)
    );
    let opcodes = [
        (SULD_FORMATTED_OPCODE, 0x998),
        (SULD_BINARY_OPCODE, 0x99a),
        (SUST_FORMATTED_OPCODE, 0x99c),
        (SUST_BINARY_OPCODE, 0x99e),
        (SUATOM_OPCODE, 0x394),
        (SUATOM_CAS_OPCODE, 0x396),
        (SURED_OPCODE, 0x3a0),
    ];
    for (constant, literal) in opcodes {
        assert_eq!(constant, literal);
    }
    let fields = [
        (REG_A, (24, 8)),
        (REG_B, (32, 8)),
        (TEX_CBUF_OFFSET, (40, 14)),
        (TEX_CBUF_INDEX, (54, 5)),
        (TEX_BINDLESS, (59, 1)),
        (TEX_UNIFORM_HANDLE, (40, 6)),
        (TEX_UNIFORM, (91, 1)),
        (TEX_SCALAR, (60, 1)),
        (TEX_DIM, (61, 3)),
        (TEX_DST2, (64, 8)),
        (CHANNEL_MASK, (72, 4)),
        (TEX_OFFSET, (76, 1)),
        (TEX_NDV, (77, 1)),
        (TEX_DC, (78, 1)),
        (TEX_F16, (79, 1)),
        (TLD_CLAMP, (77, 1)),
        (TLD_MS, (78, 1)),
        (TLD4_OFFSET, (76, 2)),
        (TLD4_COMP, (87, 2)),
        (EVICT, (84, 3)),
        (TEX_LOD, (87, 3)),
        (TEX_NODEP, (90, 1)),
        (TXQ_QUERY, (62, 2)),
        (FAULT, (81, 3)),
        (IMAGE_DIM, (61, 3)),
        (SU_HANDLE, (64, 8)),
        (MEM_TYPE, (73, 3)),
        (MEM_ORDER, (77, 4)),
        (SUATOM_BA, (72, 1)),
        (ATOM_TYPE, (73, 3)),
        (SUATOM_OP, (87, 4)),
        (SURED_OP, (87, 3)),
    ];
    for (field, (start, width)) in fields {
        assert_eq!((field.start, field.width), (start, width));
    }
    assert_eq!(TEX_CBUF_UNIT, 4);
}

/// The group's enumerations against NAK's encoding tables, written out here, in each
/// enumeration's declaration order (`set_tex_dim`, `set_tex_lod_mode`, `OpTld4`'s and `OpTxq`'s
/// encoders, `set_image_dim`, `set_mem_type`, `set_eviction_priority`, `set_atom_type`,
/// `set_atom_op` and `set_mem_order` in `sm70_encode.rs` @ 52238762).
#[test]
fn encodings_match_the_references() {
    assert_eq!(
        TexDim::ALL.iter().map(|dim| dim.bits()).collect::<Vec<_>>(),
        [0, 4, 1, 5, 2, 3, 7]
    );
    let bits = |all: &[u64]| all.to_vec();
    assert_eq!(
        TexLodMode::ALL
            .iter()
            .map(|mode| mode.bits())
            .collect::<Vec<_>>(),
        bits(&[0, 1, 2, 3, 4, 5])
    );
    assert_eq!(
        TexOffsetMode::ALL
            .iter()
            .map(|mode| mode.bits())
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(
        TexChannel::ALL
            .iter()
            .map(|comp| comp.bits())
            .collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert_eq!(
        TexQuery::ALL
            .iter()
            .map(|query| query.bits())
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(
        ImageDim::ALL
            .iter()
            .map(|dim| dim.bits())
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4, 5]
    );
    assert_eq!(
        MemType::ALL.iter().map(|ty| ty.bits()).collect::<Vec<_>>(),
        [0, 1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        MemEvictionPriority::ALL
            .iter()
            .map(|eviction| eviction.bits())
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4, 5]
    );
    // `set_atom_type(.., true)`: U32 0, I32 1, U64 2, F32 3, F16v2 4, I64 5, F64 6.
    let atom_types = [
        (AtomType::U32, 0),
        (AtomType::I32, 1),
        (AtomType::U64, 2),
        (AtomType::F32, 3),
        (AtomType::F16v2, 4),
        (AtomType::I64, 5),
        (AtomType::F64, 6),
    ];
    assert_eq!(atom_types.len(), AtomType::ALL.len());
    for (atom_type, literal) in atom_types {
        assert_eq!(atom_type.bits(), literal);
    }
    assert_eq!(
        AtomOp::ENCODED.map(AtomOp::bits),
        [0, 1, 2, 3, 4, 5, 6, 7, 8].map(Some)
    );
    assert_eq!(AtomOp::CmpExch(AtomCmpSrc::Packed).bits(), None);
    // `set_mem_order`, `self.sm >= 80`.
    let orders = [
        (MemOrder::Constant, 0x4),
        (MemOrder::Weak, 0x0),
        (MemOrder::Strong(MemScope::Cta), 0x5),
        (MemOrder::Strong(MemScope::Gpu), 0x7),
        (MemOrder::Strong(MemScope::System), 0xa),
    ];
    assert_eq!(MemOrder::ALL.len(), orders.len());
    for (order, literal) in orders {
        assert_eq!(order.bits(), literal);
        assert!(MemOrder::ALL.contains(&order));
    }
}

proptest! {
    #![proptest_config(config())]

    /// The group's operations alone, so each gets many more cases than in the crate-wide round
    /// trip: assembling then decoding gives the instruction back.
    #[test]
    fn texture_assemble_then_decode_round_trips(
        guard in crate::testing::any_pred_src(),
        op in Union::new(texture::any_ops()),
    ) {
        let instr = Instr { guard, op };
        let word = assemble(&instr).expect("the generator makes encodable instructions");
        prop_assert_eq!(decode(word), Ok(instr));
    }

    /// The group's words alone: a word decodes and re-assembles to itself (scheduling bits
    /// aside), or is refused for a value that names nothing, in one of the fields the module
    /// docs list. No form reads a bit only in some layouts, so no word is refused at step 5.
    #[test]
    fn texture_decode_then_assemble_round_trips(word in any_word_of(texture::FORMS)) {
        match decode(word) {
            Ok(instr) => {
                let again = assemble(&instr).expect("a decoded instruction is encodable");
                prop_assert_eq!(again.bits(), word.bits() & !SCHEDULING.mask());
            }
            Err(DecodeError::UnknownValue { bit, width, .. }) => {
                let field = Field::new(bit, width);
                let refusable = [
                    TEX_DIM, EVICT, TEX_LOD, TLD4_OFFSET, TXQ_QUERY, MEM_TYPE, MEM_ORDER,
                    SUATOM_OP,
                ];
                prop_assert!(refusable.contains(&field), "bit {}", bit);
            }
            Err(error) => prop_assert!(false, "{error}"),
        }
    }
}
