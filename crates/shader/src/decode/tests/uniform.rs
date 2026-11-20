//! Tests of slice 6d, the uniform datapath (`op::uniform`): NAK's expected text, one hand-built
//! word per form, the registers and predicates at the edges of their six- and three-bit fields,
//! the bits the group refuses or requires, the references' constants written out, the twins'
//! derivation from the vector forms, the reader's width check, the assembler's refusals and
//! generated round trips of the group alone.

use proptest::prelude::*;
use proptest::strategy::Union;

use super::super::form::Reader;
use super::super::op::uniform::{self, TWIN_MNEMONICS, TWINS, Twin};
use super::*;
use crate::asm::{EncodeError, assemble};
use crate::testing::form_and_row;

fn up(index: u8) -> UniformPred {
    UniformPred::new(index).expect("a uniform predicate")
}

fn ups(index: u8, negated: bool) -> UniformPredSrc {
    UniformPredSrc {
        reg: up(index),
        negated,
    }
}

/// Every type NAK's `test_ldc` checks for SM 8.6 with the text it expects (`.128` only from SM
/// 10.0), as `(type, text)`.
const LDC_TYPES: [(MemType, &str); 6] = [
    (MemType::U8, ".u8"),
    (MemType::I8, ".s8"),
    (MemType::U16, ".u16"),
    (MemType::I16, ".s16"),
    (MemType::B32, ""),
    (MemType::B64, ".64"),
];

/// NAK's `test_ldc`, its uniform cases for SM 8.6, ported (`nvdisasm_tests.rs` @ 52238762):
/// `uldc` into `ur4` from `c[0x5]` and `cx[ur2]` at `0x248`, every type but `.128`, without an
/// offset register (NAK adds one for a uniform destination only from SM 10.0).
#[test]
fn uldc_text_matches_nak_test_ldc() {
    let bufs = [
        (CBuf::Binding(5), "c[0x5]"),
        (CBuf::BindlessUniform(ur(2)), "cx[ur2]"),
    ];
    for (buf, buf_text) in bufs {
        for (mem_type, type_text) in LDC_TYPES {
            let op = Op::ULdc(OpULdc {
                dst: ur(4),
                src: ULdcSrc::CBuf(CBufRef { buf, offset: 0x248 }),
                mem_type,
            });
            assert_eq!(
                round_text(Instr::new(op)),
                format!("uldc{type_text} ur4, {buf_text}[0x248];")
            );
        }
    }
}

/// NAK's `test_uldc_global` for SM 8.6, ported: `OpLdcg` into `ur2` from `[ur4+0x100]` under
/// `up1`, every type but `.128`.
#[test]
fn ldcg_text_matches_nak_test_uldc_global() {
    for (mem_type, type_text) in LDC_TYPES {
        let op = Op::ULdc(OpULdc {
            dst: ur(2),
            src: ULdcSrc::Global {
                addr: ur(4),
                offset: 0x100,
                pred: ups(1, false),
            },
            mem_type,
        });
        assert_eq!(
            round_text(Instr::new(op)),
            format!("uldc{type_text} ur2, [ur4+0x100], up1;")
        );
    }
}

/// One hand-built word per form of the non-twin instructions, each field a distinct value, with
/// positions written out from the references (NAK's encoders; PR #4588's `Visitor::ULDC` for
/// `0xabb`), and each one-bit field set alone in some word.
#[test]
fn hand_built_words_decode() {
    let cases = [
        // R2UR: dst 16..22, src 24..32, PT at 81..84.
        (0x3c2, vec![(16, 5), (24, 0x9a), (81, 7)], "r2ur ur5, r154;"),
        // UMOV: dst, the immediate at 32..64; or the register at 32..38 and bit 91.
        (
            0x882,
            vec![(16, 6), (32, 0x8765_4321)],
            "umov ur6, 0x87654321;",
        ),
        (0xc82, vec![(16, 7), (32, 9), (91, 1)], "umov ur7, ur9;"),
        // ULDC: dst, offset 38..54, index 54..59, type 73..76; bindless: handle 24..30, bit 91.
        (
            0xab9,
            vec![(16, 3), (38, 0x1234), (54, 17), (73, 2)],
            "uldc.u16 ur3, c[0x11][0x1234];",
        ),
        (
            0xab9,
            vec![(16, 3), (24, 11), (38, 0x10), (73, 5), (91, 1)],
            "uldc.64 ur3, cx[ur11][0x10];",
        ),
        // The indexed ULDC: the register at 24..30, bit 91 fixed.
        (
            0xabb,
            vec![(16, 8), (24, 12), (38, 0x44), (54, 31), (73, 1), (91, 1)],
            "uldc.s8 ur8, c[0x1f][ur12+0x44];",
        ),
        // LDCG: addr 24..30, offset 38..70, type, the predicate 87..91, bit 91.
        (
            0x8b8,
            vec![
                (16, 2),
                (24, 4),
                (38, 0xffff_fff0),
                (73, 3),
                (87, 5),
                (91, 1),
            ],
            "uldc.s16 ur2, [ur4+-0x10], up5;",
        ),
        (
            0x8b8,
            vec![(16, 2), (24, 4), (73, 4), (87, 7), (90, 1), (91, 1)],
            "uldc ur2, [ur4], !upt;",
        ),
        // UPLOP3: lut1 16..24, lut0 64..67 and 72..77, c 68..72, b 77..81, dst0 81..84, dst1
        // 84..87, a 87..91.
        (
            0x89c,
            vec![
                (16, 0xa5),
                (64, 0b110),
                (68, 3),
                (72, 0b10011),
                (77, 2),
                (81, 4),
                (84, 6),
                (87, 1),
            ],
            "uplop3.lut up4, up6, up1, up2, up3, 0x9e, 0xa5;",
        ),
        (
            0x89c,
            vec![(68, 7), (71, 1), (77, 7), (81, 7), (84, 7), (87, 7)],
            "uplop3.lut upt, upt, upt, upt, !upt, 0x0, 0x0;",
        ),
        (
            0x89c,
            vec![(77, 7), (80, 1), (81, 7), (84, 7), (87, 7)],
            "uplop3.lut upt, upt, upt, !upt, up0, 0x0, 0x0;",
        ),
        (
            0x89c,
            vec![(81, 7), (84, 7), (87, 7), (90, 1)],
            "uplop3.lut upt, upt, !upt, up0, up0, 0x0, 0x0;",
        ),
    ];
    for (opcode, fields, expected) in cases {
        assert_eq!(text(word(opcode, &fields)), expected, "{opcode:#x}");
    }
}

/// A hand-built word: its opcode, its fields as `(start, value)` and the text it decodes to.
type Case = (u128, &'static [(u32, u128)], &'static str);

/// One hand-built word per twin form (`hand_built_twin_words_decode`), each field a distinct value, its registers six bits wide
/// and bit 91 set; the vector op's fields at the vector op's positions (`op::integer`'s tests).
const TWIN_CASES: &[Case] = &[
    // UIADD3: dst 16, a 24, b 32 (form 1) or the immediate, c 64; .X bit 74; the negations
    // 72, 63 and 75; the carries 87..91 and 77..81; the overflows 81..84 and 84..87.
    (
        0x290,
        &[
            (16, 1),
            (24, 2),
            (32, 3),
            (64, 4),
            (77, 7),
            (81, 5),
            (84, 6),
            (87, 7),
        ],
        "uiadd3 ur1, up5, up6, ur2, ur3, ur4;",
    ),
    (
        0x490,
        &[
            (16, 1),
            (24, 2),
            (32, 0x99),
            (64, 4),
            (72, 1),
            (74, 1),
            (77, 2),
            (81, 7),
            (84, 7),
            (87, 3),
        ],
        "uiadd3.x ur1, ~ur2, ur4, 0x99, up3, up2;",
    ),
    (
        0x890,
        &[
            (16, 62),
            (24, 63),
            (32, 0x123),
            (64, 0),
            (75, 1),
            (77, 7),
            (81, 7),
            (84, 7),
            (87, 7),
        ],
        "uiadd3 ur62, urz, 0x123, -ur0;",
    ),
    // ULEA: a 24, b 32 or the immediate, a_high 64; .HI 80, .X 74, .SX32 73, the shifted
    // value's modifier 72, the shift 75..80, the overflow 81..84, the carry 87..91.
    (
        0x291,
        &[
            (16, 1),
            (24, 2),
            (32, 3),
            (63, 1),
            (64, 4),
            (75, 5),
            (80, 1),
            (81, 7),
            (87, 7),
        ],
        "ulea.hi ur1, ur2, -ur3, ur4, 0x5;",
    ),
    (
        0x491,
        &[
            (16, 1),
            (24, 2),
            (32, 0x40),
            (64, 4),
            (72, 1),
            (73, 1),
            (74, 1),
            (75, 31),
            (81, 3),
            (87, 2),
        ],
        "ulea.x.sx32 ur1, up3, ~ur2, ur4, 0x40, 0x1f, up2;",
    ),
    (
        0x891,
        &[(16, 1), (24, 2), (32, 0x40), (64, 63), (81, 7), (87, 7)],
        "ulea ur1, ur2, 0x40, 0x0;",
    ),
    // ULOP3: the table 72..80, the predicate 81..84, !UPT at 87..91 (fixed).
    (
        0x292,
        &[
            (16, 1),
            (24, 2),
            (32, 3),
            (64, 4),
            (72, 0x96),
            (81, 7),
            (87, 15),
        ],
        "ulop3.lut ur1, ur2, ur3, ur4, 0x96, !upt;",
    ),
    (
        0x492,
        &[
            (16, 1),
            (24, 2),
            (32, 0xff),
            (64, 4),
            (72, 0xc0),
            (81, 2),
            (87, 15),
        ],
        "ulop3.lut up2, ur1, ur2, ur4, 0xff, 0xc0, !upt;",
    ),
    (
        0x892,
        &[
            (16, 1),
            (24, 2),
            (32, 0xff),
            (64, 4),
            (72, 0x3c),
            (81, 7),
            (87, 15),
        ],
        "ulop3.lut ur1, ur2, 0xff, ur4, 0x3c, !upt;",
    ),
    // UPRMT: a 24, the selector 32 or the immediate, b 64, the mode 72..75.
    (
        0x296,
        &[(16, 1), (24, 2), (32, 3), (64, 4), (72, 1)],
        "uprmt.f4e ur1, ur2, ur3, ur4;",
    ),
    (
        0x496,
        &[(16, 1), (24, 2), (32, 0x3210), (64, 4), (72, 6)],
        "uprmt.rc16 ur1, ur2, ur4, 0x3210;",
    ),
    (
        0x896,
        &[(16, 1), (24, 2), (32, 0x5410), (64, 4)],
        "uprmt ur1, ur2, 0x5410, ur4;",
    ),
    // USHF: low 24, shift 32 or the immediate, high 64; the type 73..75, .W 75, .R 76,
    // .HI 80.
    (
        0x299,
        &[(16, 1), (24, 2), (32, 3), (64, 4), (73, 3), (76, 1)],
        "ushf.r.u32 ur1, ur2, ur3, ur4;",
    ),
    (
        0x499,
        &[(16, 1), (24, 2), (32, 0x1f), (64, 4), (73, 2), (75, 1)],
        "ushf.l.w.s32 ur1, ur2, ur4, 0x1f;",
    ),
    (
        0x899,
        &[(16, 1), (24, 2), (32, 0x10), (64, 4), (80, 1)],
        "ushf.l.s64.hi ur1, ur2, 0x10, ur4;",
    ),
    // UIMAD: .X 74, signed 73, the carry 87..91, the negations, UPT at 81..84 (fixed).
    (
        0x2a4,
        &[
            (16, 1),
            (24, 2),
            (32, 3),
            (64, 4),
            (73, 1),
            (81, 7),
            (87, 7),
        ],
        "uimad ur1, ur2, ur3, ur4;",
    ),
    (
        0x4a4,
        &[
            (16, 1),
            (24, 2),
            (32, 0x80),
            (64, 4),
            (74, 1),
            (81, 7),
            (87, 4),
        ],
        "uimad.x.u32 ur1, ur2, ur4, 0x80, up4;",
    ),
    (
        0x8a4,
        &[
            (16, 1),
            (24, 2),
            (32, 0x80),
            (64, 4),
            (72, 1),
            (75, 1),
            (81, 7),
            (87, 7),
        ],
        "uimad.u32 ur1, -ur2, 0x80, -ur4;",
    ),
    // UIMAD.WIDE: signed 73, the carry 87..91 (not printed), UPT at 81..84.
    (
        0x2a5,
        &[
            (16, 2),
            (24, 3),
            (32, 4),
            (64, 6),
            (73, 1),
            (81, 7),
            (87, 7),
        ],
        "uimad.wide ur2, ur3, ur4, ur6;",
    ),
    (
        0x4a5,
        &[(16, 2), (24, 3), (32, 0x1000), (64, 6), (81, 7), (87, 1)],
        "uimad.wide.u32 ur2, ur3, ur6, 0x1000;",
    ),
    (
        0x8a5,
        &[(16, 2), (24, 3), (32, 0x1000), (64, 6), (81, 7)],
        "uimad.wide.u32 ur2, ur3, 0x1000, ur6;",
    ),
    // UISETP: the comparison 76..79, signed 73, the combination 74..76, .EX 72, the
    // destinations 81..84 and 84..87, the accumulator 87..91, the low comparison 68..72.
    (
        0x28c,
        &[
            (24, 2),
            (32, 3),
            (68, 7),
            (73, 1),
            (74, 1),
            (76, 1),
            (81, 0),
            (84, 7),
            (87, 7),
        ],
        "uisetp.lt.or up0, upt, ur2, ur3, upt;",
    ),
    (
        0x88c,
        &[
            (24, 2),
            (32, 0x7),
            (68, 5),
            (72, 1),
            (74, 2),
            (76, 6),
            (81, 1),
            (84, 2),
            (87, 3),
            (90, 1),
        ],
        "uisetp.ge.u32.xor.ex up1, up2, ur2, 0x7, !up3, up5;",
    ),
    // USEL: the condition 87..91.
    (
        0x287,
        &[(16, 1), (24, 2), (32, 3), (87, 6)],
        "usel ur1, ur2, ur3, up6;",
    ),
    (
        0x887,
        &[(16, 1), (24, 2), (32, 0x11), (87, 6), (90, 1)],
        "usel ur1, ur2, 0x11, !up6;",
    ),
    // USGXT: signed 73, .W 75.
    (
        0x29a,
        &[(16, 1), (24, 2), (32, 3), (73, 1), (75, 1)],
        "usgxt.w ur1, ur2, ur3;",
    ),
    (
        0x89a,
        &[(16, 1), (24, 2), (32, 0x8)],
        "usgxt.u32 ur1, ur2, 0x8;",
    ),
    // UBMSK: .W 75.
    (
        0x29b,
        &[(16, 1), (24, 2), (32, 3), (75, 1)],
        "ubmsk.w ur1, ur2, ur3;",
    ),
    (
        0x89b,
        &[(16, 1), (24, 2), (32, 0x8)],
        "ubmsk ur1, ur2, 0x8;",
    ),
    // UFLO: the source 32 or the immediate, its not 63 (form 1), signed 73, .SH 74, UPT
    // at 81..84.
    (
        0x2bd,
        &[(16, 1), (32, 3), (63, 1), (73, 1), (81, 7)],
        "uflo ur1, ~ur3;",
    ),
    (
        0x8bd,
        &[(16, 1), (32, 0x8000_0000), (74, 1), (81, 7)],
        "uflo.u32.sh ur1, 0x80000000;",
    ),
    // UBREV.
    (0x2be, &[(16, 1), (32, 3)], "ubrev ur1, ur3;"),
    (0x8be, &[(16, 1), (32, 0xf0)], "ubrev ur1, 0xf0;"),
    // UPOPC: the not 63 (form 1).
    (0x2bf, &[(16, 1), (32, 3), (63, 1)], "upopc ur1, ~ur3;"),
    (0x8bf, &[(16, 1), (32, 0xf0)], "upopc ur1, 0xf0;"),
    (0x2bf, &[(16, 63), (32, 63)], "upopc urz, urz;"),
];

/// Each of [`TWIN_CASES`] decodes to its text, with bit 91 set.
#[test]
fn hand_built_twin_words_decode() {
    let wrong: Vec<(u128, String, &str)> = TWIN_CASES
        .iter()
        .map(|&(opcode, fields, expected)| {
            let word = word(opcode, fields).with_field(Field::bit(91), 1);
            (opcode, text(word), expected)
        })
        .filter(|(_, actual, expected)| actual != expected)
        .collect();
    assert_eq!(wrong, []);
}

/// Every twin form, every non-twin form and every opcode constant against the references'
/// literals, written out here (NAK's `encode_ualu(..)` and `set_opcode(..)`; PR #4588's
/// `Visitor::ULDC` for `0xabb`).
#[test]
fn opcodes_are_the_references_literals() {
    let twins: Vec<u16> = TWINS.iter().map(|twin| twin.base).collect();
    assert_eq!(
        twins,
        [
            0x090, 0x091, 0x092, 0x096, 0x099, 0x0a4, 0x0a5, 0x08c, 0x087, 0x09a, 0x09b, 0x0bd,
            0x0be, 0x0bf
        ]
    );
    let vectors: Vec<u16> = TWINS.iter().map(|twin| twin.vector.base()).collect();
    assert_eq!(
        vectors,
        [
            0x010, 0x011, 0x012, 0x016, 0x019, 0x024, 0x025, 0x00c, 0x007, 0x01a, 0x01b, 0x100,
            0x101, 0x109
        ]
    );
    let opcodes: Vec<u16> = uniform::FORMS.iter().map(|form| form.opcode).collect();
    assert_eq!(
        opcodes,
        [
            0x290, 0x490, 0x890, 0x291, 0x491, 0x891, 0x292, 0x492, 0x892, 0x296, 0x496, 0x896,
            0x299, 0x499, 0x899, 0x2a4, 0x4a4, 0x8a4, 0x2a5, 0x4a5, 0x8a5, 0x28c, 0x88c, 0x287,
            0x887, 0x29a, 0x89a, 0x29b, 0x89b, 0x2bd, 0x8bd, 0x2be, 0x8be, 0x2bf, 0x8bf, 0x3c2,
            0x882, 0xc82, 0xab9, 0xabb, 0x8b8, 0x89c
        ]
    );
    assert_eq!(
        [
            uniform::R2UR_OPCODE,
            uniform::UMOV_IMM_OPCODE,
            uniform::UMOV_UR_OPCODE,
            uniform::ULDC_OPCODE,
            uniform::ULDC64_OPCODE,
            uniform::LDCG_OPCODE,
            uniform::UPLOP3_OPCODE,
        ],
        [0x3c2, 0x882, 0xc82, 0xab9, 0xabb, 0x8b8, 0x89c]
    );
    assert_eq!(
        [
            uniform::ULDC_INDEX,
            uniform::LDCG_ADDR,
            uniform::LDCG_OFFSET
        ],
        [Field::new(24, 6), Field::new(24, 6), Field::new(38, 32)]
    );
}

/// Each twin's mnemonic is `u` and its vector op's, and `Op::mnemonic` names it.
#[test]
fn twins_are_named_u_and_their_vector_ops() {
    assert_eq!(TWIN_MNEMONICS.len(), 16);
    for &(twin, vector) in TWIN_MNEMONICS {
        assert_eq!(twin, format!("u{vector}"));
    }
}

/// Each twin form is its vector form with the register fields narrowed to six bits and bit 91
/// fixed, and its row drops exactly the vector form's row's bits: the twins keep the vector
/// ops' divergences (module docs).
#[test]
fn twin_rows_drop_the_vector_ops_bits() {
    let mut checked = 0;
    for twin in TWINS {
        for &form in twin.forms() {
            let uniform_form = twin.form(form);
            let vector = twin.vector.form(form);
            let (_, row) = form_and_row(uniform_form.opcode);
            let (_, vector_row) = form_and_row(vector.opcode);
            assert_eq!(
                row.dropped, vector_row.dropped,
                "{:#x}",
                uniform_form.opcode
            );
            assert_eq!(uniform_form.fixed, vector.fixed | 1 << 91);
            // The narrowed bits: the top two of each register field the vector form has.
            let narrowed: u128 = twin
                .register_fields(form)
                .into_iter()
                .flatten()
                .map(|field| 0b11 << (field.start + 6))
                .sum();
            assert_eq!(uniform_form.known, (vector.known & !narrowed) | 1 << 91);
            checked += 1;
        }
    }
    assert_eq!(checked, 7 * 3 + 7 * 2);
}

/// The register fields of each twin's forms, written out: the destination (16) but `UISETP`'s,
/// source 0 (24) but the one-source instructions', the wide slot's register (32) in form 1 only,
/// and the register slot (64) where source 2, or source 1 in form 2, is.
#[test]
fn twin_register_fields_are_the_listed_ones() {
    let starts = |base: u16, form: AluForm| -> Vec<u32> {
        let twin = Twin::of_base(u64::from(base)).expect("a twin");
        twin.register_fields(form)
            .into_iter()
            .flatten()
            .map(|field| field.start)
            .collect()
    };
    assert_eq!(starts(0x090, AluForm::Reg), [16, 24, 32, 64]);
    assert_eq!(starts(0x090, AluForm::ImmSrc2), [16, 24, 64]);
    assert_eq!(starts(0x090, AluForm::ImmSrc1), [16, 24, 64]);
    assert_eq!(starts(0x08c, AluForm::Reg), [24, 32]);
    assert_eq!(starts(0x08c, AluForm::ImmSrc1), [24]);
    assert_eq!(starts(0x087, AluForm::Reg), [16, 24, 32]);
    assert_eq!(starts(0x0bd, AluForm::Reg), [16, 32]);
    assert_eq!(starts(0x0bf, AluForm::ImmSrc1), [16]);
    for field in Twin::of_base(0x090)
        .expect("a twin")
        .register_fields(AluForm::Reg)
    {
        assert_eq!(field.map(|field| field.width), Some(8));
    }
    assert_eq!(Twin::of_base(0x010), None);
    assert_eq!(Twin::of_vector_base(0x090), None);
    assert_eq!(
        Twin::of_vector_base(0x109),
        Twin::of_base(0x0bf),
        "POPC's twin"
    );
}

/// The uniform datapath's forms with a fixed bit 91, written out: every twin form, `UMOV`'s
/// register form, the indexed `ULDC` and `LDCG` (and not `R2UR`, `UMOV`'s immediate form, `ULDC`
/// `0xab9`, where bit 91 picks a bindless buffer, or `UPLOP3`).
#[test]
fn uniform_datapath_forms_require_bit_91() {
    let required = [
        0x290, 0x490, 0x890, 0x291, 0x491, 0x891, 0x292, 0x492, 0x892, 0x296, 0x496, 0x896, 0x299,
        0x499, 0x899, 0x2a4, 0x4a4, 0x8a4, 0x2a5, 0x4a5, 0x8a5, 0x28c, 0x88c, 0x287, 0x887, 0x29a,
        0x89a, 0x29b, 0x89b, 0x2bd, 0x8bd, 0x2be, 0x8be, 0x2bf, 0x8bf, 0xc82, 0xabb, 0x8b8,
    ];
    for form in uniform::FORMS {
        let bit_91 = form.fixed & (1 << 91) != 0;
        assert_eq!(
            bit_91,
            required.contains(&form.opcode),
            "{:#x}",
            form.opcode
        );
        if bit_91 {
            let canonical = canonical_word(form.opcode);
            assert!(decode(canonical).is_ok());
            assert_eq!(
                decode(Instruction::from_bits(canonical.bits() & !(1 << 91))),
                Err(DecodeError::MissingBit {
                    opcode: form.opcode,
                    bit: 91
                }),
                "{:#x}",
                form.opcode
            );
        }
    }
    // R2UR's predicate destination is fixed instead.
    let r2ur = word(0x3c2, &[(16, 1), (24, 2), (81, 3)]);
    assert_eq!(
        decode(r2ur),
        Err(DecodeError::MissingBit {
            opcode: 0x3c2,
            bit: 83
        })
    );
}

/// PR #4588's rows that this slice leaves undecoded (module docs, "Left as unknown opcodes")
/// are unknown opcodes: the twins' forms 3, 5, 6 and 7, form 2 of the twins without source 2,
/// the float, `IABS` and `IMNMX` twins of SM 12.0, `UF2FP`, and `UP2UR`, which has no row.
#[test]
fn uniform_rows_left_undecoded_are_unknown_opcodes() {
    let unknown = [
        // Forms 3, 5, 6, 7 of the three-source twins.
        0x690, 0xa90, 0xc90, 0xe90, 0x691, 0xa91, 0xc91, 0xe91, 0x692, 0xa92, 0xc92, 0xe92, 0x696,
        0xa96, 0xc96, 0xe96, 0x699, 0xa99, 0xc99, 0xe99, 0x6a4, 0xaa4, 0xca4, 0xea4, 0x6a5, 0xaa5,
        0xca5, 0xea5, // Forms 2, 3, 5, 6, 7 of the twins without source 2.
        0x48c, 0x68c, 0xa8c, 0xc8c, 0xe8c, 0x487, 0x687, 0xa87, 0xc87, 0xe87, 0x49a, 0x69a, 0xa9a,
        0xc9a, 0xe9a, 0x49b, 0x69b, 0xa9b, 0xc9b, 0xe9b, 0x4bd, 0x6bd, 0xabd, 0xcbd, 0xebd, 0x4be,
        0x6be, 0xabe, 0xcbe, 0xebe, 0x4bf, 0x6bf, 0xabf, 0xcbf, 0xebf,
        // The SM 12.0 twins, form 1 of each (PR #4588 has rows for every form).
        0x24d, 0x250, 0x252, 0x253, 0x254, 0x255, 0x256, 0x25a, 0x25b, 0x25c, 0x25d, 0x285,
        // UF2FP's forms, and UP2UR.
        0x2ba, 0x4ba, 0x8ba, 0x883,
    ];
    for opcode in unknown {
        let word = Instruction::default().with_field(OPCODE, u64::from(opcode));
        assert_eq!(
            decode(word),
            Err(DecodeError::UnknownOpcode { opcode }),
            "{opcode:#x}"
        );
    }
}

/// `UMOV`'s register form refuses each of bits 38..64, alone, naming it (module docs).
#[test]
fn umov_rejects_the_bits_above_its_register() {
    let base = word(0xc82, &[(16, 1), (32, 2), (91, 1)]);
    assert_eq!(text(base), "umov ur1, ur2;");
    for bit in 38..64 {
        assert_eq!(
            decode(Instruction::from_bits(base.bits() | 1 << bit)),
            Err(DecodeError::UnknownBit { opcode: 0xc82, bit })
        );
    }
    // The immediate form reads all 32.
    assert_eq!(
        text(word(0x882, &[(16, 1), (63, 1)])),
        "umov ur1, 0x80000000;"
    );
}

/// `ULDC` `0xab9` reads the bound layout's index only with bit 91 clear and the bindless
/// layout's handle only with it set, so a word with the other layout's bits is refused at step
/// 5, naming the lowest; the indexed form reads both its register and its index.
#[test]
fn uldc_layouts_reject_the_other_layouts_bits() {
    let bound = word(0xab9, &[(16, 1), (54, 3)]);
    assert_eq!(text(bound), "uldc.u8 ur1, c[0x3][0x0];");
    assert_eq!(
        decode(Instruction::from_bits(bound.bits() | 1 << 24 | 1 << 29)),
        Err(DecodeError::UnknownBit {
            opcode: 0xab9,
            bit: 24
        })
    );
    let bindless = word(0xab9, &[(16, 1), (24, 2), (91, 1)]);
    assert_eq!(text(bindless), "uldc.u8 ur1, cx[ur2][0x0];");
    assert_eq!(
        decode(Instruction::from_bits(bindless.bits() | 1 << 58 | 1 << 54)),
        Err(DecodeError::UnknownBit {
            opcode: 0xab9,
            bit: 54
        })
    );
    assert_eq!(
        text(word(0xabb, &[(16, 1), (24, 2), (54, 3), (91, 1)])),
        "uldc.u8 ur1, c[0x3][ur2+0x0];"
    );
}

/// The six-bit register fields and three-bit predicate fields at their edges, in every form
/// that has them: `UR0`, `UR62` and `URZ` decode and re-assemble (a twin's `URZ` as
/// [`Gpr::RZ`]), and each bit above the six (a value of 64 or 128 in the vector op's eight bits)
/// is refused, naming it. The uniform predicates `UP0`, `UP6` and `UPT` in `UPLOP3`'s every
/// source and destination.
#[test]
fn uniform_registers_at_the_edges_of_their_encoding() {
    // Each twin form's register fields, then the other forms' uniform register fields, written
    // out, each with a base word that reads the field (a bindless ULDC for its handle).
    let mut fields: Vec<(u16, u32)> = Vec::new();
    for twin in TWINS {
        for &form in twin.forms() {
            let opcode = form.opcode(twin.base);
            let starts = twin.register_fields(form).into_iter().flatten();
            fields.extend(starts.map(|field| (opcode, field.start)));
        }
    }
    let twin_fields = fields.len();
    fields.extend([
        (0x3c2, 16),
        (0x882, 16),
        (0xc82, 16),
        (0xc82, 32),
        (0xab9, 16),
        (0xab9, 24),
        (0xabb, 16),
        (0xabb, 24),
        (0x8b8, 16),
        (0x8b8, 24),
    ]);
    for (opcode, start) in fields {
        let canonical = canonical_word(opcode);
        let base = if (opcode, start) == (0xab9, 24) {
            Instruction::from_bits(canonical.bits() | 1 << 91)
        } else {
            canonical
        };
        for value in [0, 62, 63] {
            let word = base.with_field(Field::new(start, 6), value);
            let decoded = decode(word);
            assert!(decoded.is_ok(), "{opcode:#x} {start}: {decoded:?}");
            let instr = decoded.expect("decodes");
            assert_eq!(assemble(&instr).map(Instruction::bits), Ok(word.bits()));
        }
        for (bit, value) in [(start + 6, 64), (start + 7, 128)] {
            let word = base.with_field(Field::new(start, 8), value);
            assert_eq!(
                decode(word),
                Err(DecodeError::UnknownBit { opcode, bit }),
                "{opcode:#x} {start}"
            );
        }
    }
    // Three-source twins in form 1: four fields; in forms 2 and 4: three; UISETP two and one;
    // USEL, USGXT, UBMSK three and two; UFLO, UBREV, UPOPC two and one.
    assert_eq!(
        twin_fields,
        7 * (4 + 3 + 3) + (2 + 1) + 3 * (3 + 2) + 3 * (2 + 1)
    );
    let edges = [(0, "up0"), (6, "up6"), (7, "upt")];
    for (value, name) in edges {
        let fields = [
            (68, value),
            (77, value),
            (81, value),
            (84, value),
            (87, value),
        ];
        assert_eq!(
            text(word(0x89c, &fields)),
            format!("uplop3.lut {name}, {name}, {name}, {name}, {name}, 0x0, 0x0;")
        );
    }
    let twin_rz = word(
        0x290,
        &[
            (16, 63),
            (24, 63),
            (32, 63),
            (64, 63),
            (77, 7),
            (81, 7),
            (84, 7),
            (87, 7),
            (91, 1),
        ],
    );
    let Ok(Instr {
        op: Op::UIAdd3(add),
        ..
    }) = decode(twin_rz)
    else {
        unreachable!("a UIADD3");
    };
    assert_eq!(add.op().dst, Gpr::RZ);
    assert_eq!(add.op().srcs, [AluSrc::reg(Gpr::RZ); 3]);
    assert_eq!(text(twin_rz), "uiadd3 urz, urz, urz, urz;");
}

/// The text of the twins rewrites every register and predicate operand of the vector op's and
/// nothing else: not a modifier that starts with `r` or `p` (`.rc8`, `PRMT`'s mode), not a hex
/// immediate, not the mnemonic's letters.
#[test]
fn twin_text_marks_exactly_the_register_operands() {
    let starts = |text: &str| -> Vec<usize> {
        (0..text.len())
            .filter(|&index| uniform::starts_operand(text, index))
            .collect()
    };
    assert_eq!(
        starts("prmt.rc8 r1, rz, -r2, |r3|, ~r4, !pt, p0, 0x7, [r5]"),
        [9, 13, 18, 23, 29, 34, 38, 48]
    );
    assert_eq!(starts("x r p q pz rr r1x p12 r255"), [18, 22]);
    let rc8 = word(
        0x296,
        &[(16, 1), (24, 2), (32, 3), (64, 4), (72, 3), (91, 1)],
    );
    assert_eq!(text(rc8), "uprmt.rc8 ur1, ur2, ur3, ur4;");
}

/// [`Uniform::new`] holds exactly the vector ops the uniform encoding can: registers up to
/// `UR62` and `RZ`, a register source 0, at most one immediate, no constant buffer or uniform
/// register, and not `IMAD.HI`; each refusal differs from an accepted op in that one thing.
#[test]
fn uniform_holds_exactly_what_the_encoding_can() {
    let op = |dst: Gpr, srcs: [AluSrc; 3]| OpIAdd3 {
        dst,
        overflow: [Pred::PT; 2],
        srcs,
        carry: [PredSrc::TRUE; 2],
    };
    let reg = AluSrc::reg;
    let fine = op(r(62), [reg(r(1)), reg(Gpr::RZ), AluSrc::Imm(5)]);
    assert!(Uniform::new(fine).is_some());
    assert!(Uniform::new(op(Gpr::RZ, [reg(r(0)), reg(r(1)), reg(r(2))])).is_some());
    assert!(Uniform::new(op(r(63), [reg(r(1)), reg(r(2)), reg(r(3))])).is_none());
    assert!(Uniform::new(op(r(1), [reg(r(63)), reg(r(2)), reg(r(3))])).is_none());
    assert!(Uniform::new(op(r(1), [reg(r(2)), reg(r(254)), reg(r(3))])).is_none());
    assert!(Uniform::new(op(r(1), [AluSrc::Imm(1), reg(r(2)), reg(r(3))])).is_none());
    assert!(Uniform::new(op(r(1), [reg(r(2)), AluSrc::Imm(1), AluSrc::Imm(2)])).is_none());
    let uniform_src = AluSrc::UniformReg {
        reg: ur(1),
        abs: false,
        neg: false,
    };
    assert!(Uniform::new(op(r(1), [reg(r(2)), uniform_src, reg(r(3))])).is_none());
    let cbuf = AluSrc::CBuf {
        cbuf: CBufRef {
            buf: CBuf::Binding(1),
            offset: 4,
        },
        abs: false,
        neg: false,
    };
    assert!(Uniform::new(op(r(1), [reg(r(2)), reg(r(3)), cbuf])).is_none());
    let imad = |mode| OpIMad {
        dst: r(1),
        srcs: [reg(r(2)), reg(r(3)), reg(r(4))],
        signed: false,
        mode,
    };
    let lo = IMadMode::Lo {
        x: false,
        carry: PredSrc::TRUE,
    };
    assert!(Uniform::new(imad(lo)).is_some());
    assert!(Uniform::new(imad(IMadMode::Hi)).is_none());
}

/// The reader's width check: a register or predicate field wider than its file is not turned
/// into `URZ`, `PT` or `UPT` quietly but kept as the exact error, the first one when there are
/// two; a field of the file's width never faults.
#[test]
fn the_reader_refuses_a_too_wide_field_with_the_exact_error() {
    let word = Instruction::default()
        .with_field(OPCODE, 0x3c2)
        .with_field(Field::new(16, 8), 64)
        .with_field(Field::new(40, 4), 8)
        .with_field(Field::new(48, 4), 9);
    let mut reader = Reader::new(word);
    assert_eq!(reader.uniform_reg(Field::new(16, 6)), ur(0));
    assert_eq!(reader.fault(), None);
    assert_eq!(reader.uniform_reg(Field::new(16, 8)), UniformReg::URZ);
    let first = DecodeError::UnknownValue {
        opcode: 0x3c2,
        bit: 16,
        width: 8,
        value: 64,
    };
    assert_eq!(reader.fault(), Some(first));
    assert_eq!(
        reader.pred_src(Field::new(40, 4), Field::bit(100)).reg,
        Pred::PT
    );
    assert_eq!(reader.fault(), Some(first), "the first fault is kept");
    let mut reader = Reader::new(word);
    assert_eq!(
        reader.pred_src(Field::new(40, 4), Field::bit(100)).reg,
        Pred::PT
    );
    assert_eq!(
        reader.fault(),
        Some(DecodeError::UnknownValue {
            opcode: 0x3c2,
            bit: 40,
            width: 4,
            value: 8
        })
    );
    let mut reader = Reader::new(word);
    assert_eq!(
        reader
            .uniform_pred_src(Field::new(48, 4), Field::bit(100))
            .reg,
        UniformPred::UPT
    );
    assert_eq!(
        reader.fault().map(|error| error.to_string()),
        Some(
            "opcode 0x3c2 has the value 9 in its 4-bit field at bit 48, which its form does not \
             define"
                .into()
        )
    );
    let mut reader = Reader::new(word);
    assert_eq!(reader.uniform_pred(Field::new(48, 4)), UniformPred::UPT);
    assert!(reader.fault().is_some());
    // Values at the edge of each file decode without a fault.
    let edges = Instruction::default()
        .with_field(Field::new(16, 8), 63)
        .with_field(Field::new(40, 4), 7)
        .with_field(Field::new(48, 4), 7);
    let mut reader = Reader::new(edges);
    assert_eq!(reader.uniform_reg(Field::new(16, 8)), UniformReg::URZ);
    assert!(
        reader
            .pred_src(Field::new(40, 4), Field::bit(100))
            .reg
            .is_true()
    );
    assert!(reader.uniform_pred(Field::new(48, 4)).is_true());
    assert_eq!(reader.fault(), None);
}

/// The test assembler's refusals for the group: an indexed `ULDC`'s buffer past 31.
#[test]
fn unencodable_uniform_instructions_are_errors() {
    let uldc = |index| {
        assemble(&Instr::new(Op::ULdc(OpULdc {
            dst: ur(1),
            src: ULdcSrc::Indexed {
                index,
                reg: ur(2),
                offset: 0,
            },
            mem_type: MemType::B32,
        })))
    };
    assert!(uldc(31).is_ok());
    assert_eq!(
        uldc(32),
        Err(EncodeError::TooWide {
            what: "constant buffer index",
            value: 32,
            bits: 5
        })
    );
    let bound = |index| {
        assemble(&Instr::new(Op::ULdc(OpULdc {
            dst: ur(1),
            src: ULdcSrc::CBuf(CBufRef {
                buf: CBuf::Binding(index),
                offset: 0,
            }),
            mem_type: MemType::B32,
        })))
    };
    assert!(bound(31).is_ok());
    assert_eq!(
        bound(32),
        Err(EncodeError::TooWide {
            what: "constant buffer index",
            value: 32,
            bits: 5
        })
    );
    assert_eq!(
        assemble(&Instr::new(uniform::SHELLS[0])),
        Err(EncodeError::NoEncoding { mnemonic: "up2ur" })
    );
}

/// A uniform datapath instruction's guard prints as a predicate (`@p2`), though PR #4588 reads
/// it as `UP2` (module docs, "Guards": a guess pinned here).
#[test]
fn guards_of_uniform_instructions_print_as_predicates() {
    let guarded = word(0x882, &[(16, 1), (32, 5)]).with_field(Field::new(12, 4), 2);
    assert_eq!(text(guarded), "@p2 umov ur1, 0x5;");
}

/// Every operation of the group the generators make.
fn any_uniform_op() -> impl Strategy<Value = Op> {
    Union::new(uniform::any_ops())
}

proptest! {
    #![proptest_config(config())]

    /// The group's operations alone: assembling then decoding gives the instruction back, so
    /// every field is read from where the encoder put it.
    #[test]
    fn uniform_assemble_then_decode_round_trips(
        guard in crate::testing::any_pred_src(),
        op in any_uniform_op(),
    ) {
        let instr = Instr { guard, op };
        let word = assemble(&instr).expect("the generator makes encodable instructions");
        prop_assert_eq!(decode(word), Ok(instr));
    }

    /// The group's words alone: a word decodes and re-assembles to itself (scheduling bits
    /// aside), or is refused for a bit of `ULDC`'s other layout, or for the vector ops' refused
    /// values (`UISETP`'s combination 3, `UPRMT`'s mode 7, a memory type of 7).
    #[test]
    fn uniform_decode_then_assemble_round_trips(word in any_word_of(uniform::FORMS)) {
        match decode(word) {
            Ok(instr) => {
                let again = assemble(&instr).expect("a decoded instruction is encodable");
                prop_assert_eq!(again.bits(), word.bits() & !SCHEDULING.mask());
            }
            Err(DecodeError::UnknownBit { opcode, bit }) => {
                prop_assert_eq!(opcode, 0xab9);
                let read = bits_read(word).expect("the form's function accepts the word");
                prop_assert_eq!(read & (1 << bit), 0);
                prop_assert!((24..30).contains(&bit) || (54..59).contains(&bit), "{bit}");
            }
            Err(DecodeError::UnknownValue { opcode, bit, width, value }) => {
                prop_assert!(
                    (opcode & 0x1ff == 0x08c && (bit, width, value) == (74, 2, 3))
                        || (opcode & 0x1ff == 0x096 && (bit, width, value) == (72, 3, 7))
                        || ([0xab9, 0xabb, 0x8b8].contains(&opcode)
                            && (bit, width, value) == (73, 3, 7)),
                    "{opcode:#x}"
                );
            }
            Err(error) => prop_assert!(false, "{error}"),
        }
    }
}
