//! Tests of slice 6a, the integer ALU (`op::integer`): NAK's expected text, one hand-built word
//! per form and each field set alone, the values, bits and opcodes the group refuses, the fixed
//! fields, the references' constants written out, the assembler's refusals and generated round
//! trips of the group alone.

use proptest::prelude::*;
use proptest::strategy::Union;

use super::super::op::integer::{self, *};
use super::*;
use crate::asm::{EncodeError, assemble};

/// A word built by hand: `opcode`, the guard `PT` (7 at bit 12), and each `(start, value)` placed
/// at its bit, with positions written out here rather than taken from the decoder's constants.
fn word(opcode: u128, fields: &[(u32, u128)]) -> Instruction {
    let mut bits = opcode | (7 << 12);
    for &(start, value) in fields {
        bits |= value << start;
    }
    Instruction::from_bits(bits)
}

/// Decodes a hand-built word and prints it.
fn text(word: Instruction) -> String {
    match decode(word) {
        Ok(instr) => instr.to_string(),
        Err(error) => format!("error: {error}"),
    }
}

fn ur(index: u8) -> UniformReg {
    UniformReg::new(index).expect("a uniform register")
}

fn ps(index: u8, negated: bool) -> PredSrc {
    PredSrc {
        reg: p(index),
        negated,
    }
}

fn neg(reg: Gpr) -> AluSrc {
    AluSrc::Reg {
        reg,
        abs: false,
        neg: true,
    }
}

/// NAK's `test_lea`, ported (`nvdisasm_tests.rs` @ 52238762): `r0`, `p0`, `r1`, `r2` and, with
/// `.hi`, `r3`, every shift 0 to 31, with no modifier, the shifted value negated and `b`
/// negated. NAK's `LEA` leaves bits 87..91 clear (`P0`), which `nvdisasm` does not print.
#[test]
fn lea_text_matches_nak_nvdisasm_cases() {
    let mods = [(false, false, ""), (true, false, "-"), (false, true, "")];
    let mut checked = 0;
    for (intermediate, b_neg, intermediate_str) in mods {
        let b = if b_neg { neg(r(2)) } else { AluSrc::reg(r(2)) };
        let b_str = if b_neg { "-r2" } else { "r2" };
        for shift in 0..32 {
            for (dst_high, a_high, high_str) in [
                (false, AluSrc::reg(Gpr::RZ), ""),
                (true, AluSrc::reg(r(3)), "r3, "),
            ] {
                let instr = Instr::new(Op::Lea(OpLea {
                    dst: r(0),
                    overflow: p(0),
                    a: AluSrc::reg(r(1)),
                    b,
                    a_high,
                    shift,
                    dst_high,
                    sx32: false,
                    intermediate,
                    carry: ps(0, false),
                }));
                let hi = if dst_high { ".hi" } else { "" };
                let expected = format!(
                    "lea{hi} r0, p0, {intermediate_str}r1, {b_str}, {high_str}0x{shift:x};"
                );
                assert_eq!(round_text(instr), expected);
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 3 * 32 * 2);
}

/// NAK's `test_sgxt`, ported: `sgxt r0, r1, r2` signed and `sgxt.u32` unsigned.
#[test]
fn sgxt_text_matches_nak_nvdisasm_cases() {
    for (signed, expected) in [(true, "sgxt r0, r1, r2;"), (false, "sgxt.u32 r0, r1, r2;")] {
        let instr = Instr::new(Op::Sgxt(OpSgxt {
            dst: r(0),
            a: AluSrc::reg(r(1)),
            bits: AluSrc::reg(r(2)),
            signed,
            wrap: false,
        }));
        assert_eq!(round_text(instr), expected);
    }
}

/// NAK's `test_plop3`, ported: `p0`, `p1` from `p2`, `p3`, `p4`, each negated or not, with each
/// of the sixteen one-bit truth-table pairs.
#[test]
fn plop3_text_matches_nak_nvdisasm_cases() {
    let mut checked = 0;
    for mods in 0..8_u8 {
        let [a, b, c] = [mods & 1 != 0, mods & 2 != 0, mods & 4 != 0];
        for lut_bit in 0..16 {
            let lut = 1_u16 << lut_bit;
            let luts = [
                u8::try_from(lut & 0xff).expect("low byte"),
                u8::try_from(lut >> 8).expect("high byte"),
            ];
            let instr = Instr::new(Op::PLop3(OpPLop3 {
                dsts: [p(0), p(1)],
                srcs: [ps(2, a), ps(3, b)],
                c: PLop3Src::Pred(ps(4, c)),
                luts,
            }));
            let not = |negated| if negated { "!" } else { "" };
            let expected = format!(
                "plop3.lut p0, p1, {}p2, {}p3, {}p4, {:#x}, {:#x};",
                not(a),
                not(b),
                not(c),
                luts[0],
                luts[1]
            );
            assert_eq!(round_text(instr), expected);
            checked += 1;
        }
    }
    assert_eq!(checked, 8 * 16);
}

/// NAK's `test_iabs`, ported: `iabs r1, r2`.
#[test]
fn iabs_text_matches_nak_nvdisasm_case() {
    let instr = Instr::new(Op::IAbs(OpIAbs {
        dst: r(1),
        src: AluSrc::reg(r(2)),
    }));
    assert_eq!(round_text(instr), "iabs r1, r2;");
}

/// NAK's `test_imnmx`, ported, with its text below SM 12.0: `imnmx[.u32] r1, r2, r3, p4`.
#[test]
fn imnmx_text_matches_nak_nvdisasm_cases() {
    for (signed, suffix) in [(false, ".u32"), (true, "")] {
        let instr = Instr::new(Op::IMnMx(OpIMnMx {
            dst: r(1),
            srcs: [AluSrc::reg(r(2)), AluSrc::reg(r(3))],
            signed,
            min: ps(4, false),
        }));
        assert_eq!(round_text(instr), format!("imnmx{suffix} r1, r2, r3, p4;"));
    }
}

/// The seven operand forms written out: the wide slot at bit 32 holds `r3`, the immediate
/// `0x1234`, `c[0x2][0x10]` (offset at 38, index at 54) or `ur4` (with bit 91); the register
/// slot at 64 holds `r5`. Returns the fields and the text of sources 1 and 2.
fn form_fields(form: u128, slot: bool) -> (Vec<(u32, u128)>, &'static str, &'static str) {
    let (wide, wide_text): (Vec<(u32, u128)>, _) = match form {
        1 => (vec![(32, 3)], "r3"),
        2 | 4 => (vec![(32, 0x1234)], "0x1234"),
        3 | 5 => (vec![(38, 0x10), (54, 2)], "c[0x2][0x10]"),
        _ => (vec![(32, 4), (91, 1)], "ur4"),
    };
    let mut fields = wide;
    fields.push((9, form));
    if slot {
        fields.push((64, 5));
    }
    match form {
        2 | 3 | 7 => (fields, "r5", wide_text),
        _ => (fields, wide_text, "r5"),
    }
}

/// The operand forms of an instruction with source 2.
const ALL_FORMS: &[u128] = &[1, 2, 3, 4, 5, 6, 7];
/// The operand forms of an instruction without source 2.
const SRC1_FORMS: &[u128] = &[1, 4, 5, 6];

/// Fields placed by hand: `(start, value)`.
type Fields = &'static [(u32, u128)];

/// One instruction's hand-built words, one per form.
struct FormCase {
    /// The 9-bit opcode.
    base: u128,
    /// The forms (bits 9..12).
    forms: &'static [u128],
    /// `r1` at 16 and `r2` at 24.
    dst_and_src0: bool,
    /// `r5` in the register slot at 64.
    slot: bool,
    /// The instruction's own fields.
    own: Fields,
    /// The text, with `{b}` and `{c}` for sources 1 and 2.
    text: &'static str,
}

const fn case(
    base: u128,
    forms: &'static [u128],
    (dst_and_src0, slot): (bool, bool),
    own: Fields,
    text: &'static str,
) -> FormCase {
    FormCase {
        base,
        forms,
        dst_and_src0,
        slot,
        own,
        text,
    }
}

/// Three sources: a destination, source 0 and the register slot.
const THREE: (bool, bool) = (true, true);
/// Sources 0 and 1.
const TWO: (bool, bool) = (true, false);
/// Source 1 alone (the destination, where there is one, is in `own`).
const ONE: (bool, bool) = (false, false);

/// The cases of [`hand_built_words_of_each_alu_form_decode`].
const FORM_CASES: [FormCase; 21] = [
    case(
        0x002,
        SRC1_FORMS,
        ONE,
        &[(16, 1), (72, 0xf)],
        "mov r1, {b};",
    ),
    case(
        0x010,
        ALL_FORMS,
        THREE,
        &[(77, 0xf), (81, 7), (84, 7), (87, 0xf)],
        "iadd3 r1, r2, {b}, {c};",
    ),
    case(
        0x010,
        ALL_FORMS,
        THREE,
        &[(74, 1), (77, 2 | 8), (81, 7), (84, 7), (87, 1)],
        "iadd3.x r1, r2, {b}, {c}, p1, !p2;",
    ),
    case(
        0x011,
        ALL_FORMS,
        THREE,
        &[(75, 2), (81, 7)],
        "lea r1, r2, {b}, {c}, 0x2;",
    ),
    case(
        0x011,
        ALL_FORMS,
        THREE,
        &[(74, 1), (75, 2), (81, 7), (87, 3)],
        "lea.x r1, r2, {b}, {c}, 0x2, p3;",
    ),
    case(
        0x012,
        ALL_FORMS,
        THREE,
        &[(72, 0x96), (81, 7), (87, 0xf)],
        "lop3.lut r1, r2, {b}, {c}, 0x96, !pt;",
    ),
    case(0x013, SRC1_FORMS, ONE, &[(16, 1)], "iabs r1, {b};"),
    case(0x016, ALL_FORMS, THREE, &[], "prmt r1, r2, {b}, {c};"),
    case(
        0x017,
        SRC1_FORMS,
        TWO,
        &[(87, 3)],
        "imnmx.u32 r1, r2, {b}, p3;",
    ),
    case(
        0x019,
        ALL_FORMS,
        THREE,
        &[(73, 3)],
        "shf.l.u32 r1, r2, {b}, {c};",
    ),
    case(0x01a, SRC1_FORMS, TWO, &[], "sgxt.u32 r1, r2, {b};"),
    case(0x01b, SRC1_FORMS, TWO, &[], "bmsk r1, r2, {b};"),
    case(
        0x024,
        ALL_FORMS,
        THREE,
        &[(81, 7), (87, 0xf)],
        "imad.u32 r1, r2, {b}, {c};",
    ),
    case(
        0x025,
        ALL_FORMS,
        THREE,
        &[(81, 7)],
        "imad.wide.u32 r1, r2, {b}, {c};",
    ),
    case(
        0x026,
        ALL_FORMS,
        THREE,
        &[],
        "idp.4a.u8.u8 r1, r2, {b}, {c};",
    ),
    case(
        0x027,
        &[1, 4, 5],
        THREE,
        &[(81, 7), (87, 0xf)],
        "imad.hi.u32 r1, r2, {b}, {c};",
    ),
    case(0x007, SRC1_FORMS, TWO, &[(87, 4)], "sel r1, r2, {b}, p4;"),
    case(
        0x100,
        SRC1_FORMS,
        ONE,
        &[(16, 1), (81, 7)],
        "flo.u32 r1, {b};",
    ),
    case(0x101, SRC1_FORMS, ONE, &[(16, 1)], "brev r1, {b};"),
    case(0x109, SRC1_FORMS, ONE, &[(16, 1)], "popc r1, {b};"),
    // ISETP has no register destination: p1 and pt, the accumulator and lower comparison pt.
    case(
        0x00c,
        SRC1_FORMS,
        ONE,
        &[(24, 2), (68, 7), (76, 1), (81, 1), (84, 7), (87, 7)],
        "isetp.lt.u32.and p1, pt, r2, {b}, pt;",
    ),
];

/// One hand-built word per ALU form of every instruction: `r1` the destination at bit 16 and
/// `r2` source 0 at bit 24 where the instruction has them, the sources of [`form_fields`], and
/// each instruction's own fields as given.
#[test]
fn hand_built_words_of_each_alu_form_decode() {
    let mut checked = 0;
    for case in &FORM_CASES {
        for &form in case.forms {
            let (mut fields, b, c) = form_fields(form, case.slot);
            if case.dst_and_src0 {
                fields.extend([(16, 1), (24, 2)]);
            }
            fields.extend_from_slice(case.own);
            let instruction = word(case.base, &fields);
            let expected = case.text.replace("{b}", b).replace("{c}", c);
            assert_eq!(text(instruction), expected, "{:#x} form {form}", case.base);
            checked += 1;
        }
    }
    // MOV 4, IADD3 and IADD3.X 7 each, LEA and LEA.X 7 each, LOP3 7, IABS 4, PRMT 7, IMNMX 4,
    // SHF 7, SGXT 4, BMSK 4, IMAD 7, IMAD.WIDE 7, IDP 7, IMAD.HI 3, SEL 4, FLO 4, BREV 4, POPC
    // 4, ISETP 4.
    assert_eq!(
        checked,
        4 + 14 + 14 + 7 + 4 + 7 + 4 + 7 + 4 + 4 + 7 + 7 + 7 + 3 + 4 + 4 + 4 + 4 + 4
    );
}

/// Hand-built words of the three forms that are not ALU forms.
#[test]
fn hand_built_words_of_plop3_p2r_and_r2p_decode() {
    // PLOP3 p3, pt, !p1, p2, up5: lut1 0x3c at 16, lut0 0xa5 split as 0b101 at 64 and 0b10100
    // at 72, the uniform bit 67, c up5 at 68, b p2 at 77, dst0 p3 at 81, dst1 pt at 84, a p1
    // negated at 87 (with 90).
    let plop3 = word(
        0x81c,
        &[
            (16, 0x3c),
            (64, 0b101),
            (72, 0b10100),
            (67, 1),
            (68, 5),
            (77, 2),
            (81, 3),
            (84, 7),
            (87, 1 | 8),
        ],
    );
    assert_eq!(text(plop3), "plop3.lut p3, pt, !p1, p2, up5, 0xa5, 0x3c;");
    // Without the uniform bit, c is a predicate; negated with bit 71.
    let plop3 = word(
        0x81c,
        &[(68, 5 | 8), (77, 2 | 8), (81, 7), (84, 6), (87, 7)],
    );
    assert_eq!(text(plop3), "plop3.lut pt, p6, pt, !p2, !p5, 0x0, 0x0;");
    // P2R r4, PR, r9, 0x55: dst at 16, source at 24, mask at 32.
    assert_eq!(
        text(word(0x803, &[(16, 4), (24, 9), (32, 0x55)])),
        "p2r r4, pr, r9, 0x55;"
    );
    // R2P PR, r9.b2, 0x7f: source at 24, mask at 32, byte at 76.
    assert_eq!(
        text(word(0x804, &[(24, 9), (32, 0x7f), (76, 2)])),
        "r2p pr, r9.b2, 0x7f;"
    );
    assert_eq!(text(word(0x804, &[(24, 9), (32, 1)])), "r2p pr, r9, 0x1;");
    // The byte suffix at both ends of the field past 0: byte 1 and byte 3 (bits 76..78).
    assert_eq!(
        text(word(0x804, &[(24, 9), (32, 1), (76, 1)])),
        "r2p pr, r9.b1, 0x1;"
    );
    assert_eq!(
        text(word(0x804, &[(24, 9), (32, 1), (76, 3)])),
        "r2p pr, r9.b3, 0x1;"
    );
}

/// One field case: the opcode, the fields beside [`field_text`]'s base, and the text.
type FieldCase = (u128, Fields, &'static str);

/// The text of a form 1 word of `opcode`: `r1` at 16, `r2` at 24 (with source 0), `r3` at 32,
/// `r5` at 64 (with source 2), and `own`.
fn field_text(opcode: u128, own: &[(u32, u128)]) -> String {
    let mut fields = vec![(16, 1), (32, 3), (9, 1)];
    if ![0x100, 0x109].contains(&opcode) {
        fields.push((24, 2));
    }
    if ![0x100, 0x109, 0x017, 0x01a, 0x01b, 0x007].contains(&opcode) {
        fields.push((64, 5));
    }
    fields.extend_from_slice(own);
    text(word(opcode, &fields))
}

/// `IADD3` (each source's negation, the overflow predicates, `.X`) and `LEA` (the shifted
/// value's modifier, `.SX32`, `.X`, the shift, `.HI`, the overflow, `b`'s negation, the carry).
const ADD_FIELDS: [FieldCase; 15] = [
    (
        0x010,
        &[(77, 0xf), (81, 7), (84, 7), (87, 0xf), (72, 1)],
        "iadd3 r1, -r2, r3, r5;",
    ),
    (
        0x010,
        &[(77, 0xf), (81, 7), (84, 7), (87, 0xf), (63, 1)],
        "iadd3 r1, r2, -r3, r5;",
    ),
    (
        0x010,
        &[(77, 0xf), (81, 7), (84, 7), (87, 0xf), (75, 1)],
        "iadd3 r1, r2, r3, -r5;",
    ),
    (
        0x010,
        &[(77, 0xf), (81, 2), (84, 7), (87, 0xf)],
        "iadd3 r1, p2, pt, r2, r3, r5;",
    ),
    (
        0x010,
        &[(77, 0xf), (81, 7), (84, 6), (87, 0xf)],
        "iadd3 r1, pt, p6, r2, r3, r5;",
    ),
    (
        0x010,
        &[(77, 0xf), (81, 7), (84, 7), (87, 0xf), (74, 1)],
        "iadd3.x r1, r2, r3, r5, !pt, !pt;",
    ),
    (
        0x010,
        &[(72, 1), (74, 1), (77, 7), (81, 7), (84, 7), (87, 7)],
        "iadd3.x r1, ~r2, r3, r5, pt, pt;",
    ),
    (0x011, &[(72, 1), (81, 7)], "lea r1, -r2, r3, r5, 0x0;"),
    (0x011, &[(73, 1), (81, 7)], "lea.sx32 r1, r2, r3, r5, 0x0;"),
    (
        0x011,
        &[(74, 1), (81, 7), (87, 0xf)],
        "lea.x r1, r2, r3, r5, 0x0, !pt;",
    ),
    (
        0x011,
        &[(72, 1), (63, 1), (74, 1), (81, 7)],
        "lea.x r1, ~r2, ~r3, r5, 0x0, p0;",
    ),
    (0x011, &[(75, 0x1f), (81, 7)], "lea r1, r2, r3, r5, 0x1f;"),
    (0x011, &[(80, 1), (81, 7)], "lea.hi r1, r2, r3, r5, 0x0;"),
    (0x011, &[(81, 4)], "lea r1, p4, r2, r3, r5, 0x0;"),
    (0x011, &[(63, 1), (81, 7)], "lea r1, r2, -r3, r5, 0x0;"),
];

/// `LOP3`, `PRMT`, `IMNMX`, `SHF`, `SGXT`, `BMSK`, `SEL`, `FLO` and `POPC`.
const LOGIC_FIELDS: [FieldCase; 24] = [
    (
        0x012,
        &[(72, 0x80), (81, 7), (87, 0xf)],
        "lop3.lut r1, r2, r3, r5, 0x80, !pt;",
    ),
    (
        0x012,
        &[(72, 0x01), (81, 2), (87, 0xf)],
        "lop3.lut p2, r1, r2, r3, r5, 0x1, !pt;",
    ),
    (0x016, &[(72, 1)], "prmt.f4e r1, r2, r3, r5;"),
    (0x016, &[(72, 2)], "prmt.b4e r1, r2, r3, r5;"),
    (0x016, &[(72, 3)], "prmt.rc8 r1, r2, r3, r5;"),
    (0x016, &[(72, 4)], "prmt.ecl r1, r2, r3, r5;"),
    (0x016, &[(72, 5)], "prmt.ecr r1, r2, r3, r5;"),
    (0x016, &[(72, 6)], "prmt.rc16 r1, r2, r3, r5;"),
    (0x017, &[(73, 1), (87, 2)], "imnmx r1, r2, r3, p2;"),
    (0x017, &[(87, 2), (90, 1)], "imnmx.u32 r1, r2, r3, !p2;"),
    (0x019, &[], "shf.l.s64 r1, r2, r3, r5;"),
    (0x019, &[(73, 1)], "shf.l.u64 r1, r2, r3, r5;"),
    (0x019, &[(73, 2)], "shf.l.s32 r1, r2, r3, r5;"),
    (0x019, &[(75, 1)], "shf.l.w.s64 r1, r2, r3, r5;"),
    (0x019, &[(76, 1)], "shf.r.s64 r1, r2, r3, r5;"),
    (0x019, &[(80, 1)], "shf.l.s64.hi r1, r2, r3, r5;"),
    (0x01a, &[(73, 1)], "sgxt r1, r2, r3;"),
    (0x01a, &[(75, 1)], "sgxt.w.u32 r1, r2, r3;"),
    (0x01b, &[(75, 1)], "bmsk.w r1, r2, r3;"),
    (0x007, &[(87, 1), (90, 1)], "sel r1, r2, r3, !p1;"),
    (0x100, &[(63, 1), (81, 7)], "flo.u32 r1, ~r3;"),
    (0x100, &[(73, 1), (81, 7)], "flo r1, r3;"),
    (0x100, &[(74, 1), (81, 7)], "flo.u32.sh r1, r3;"),
    (0x109, &[(63, 1)], "popc r1, ~r3;"),
];

/// `IMAD`, `IMAD.WIDE`, `IMAD.HI` and `IDP`.
const MULTIPLY_FIELDS: [FieldCase; 15] = [
    (
        0x024,
        &[(72, 1), (81, 7), (87, 0xf)],
        "imad.u32 r1, -r2, r3, r5;",
    ),
    (
        0x024,
        &[(63, 1), (81, 7), (87, 0xf)],
        "imad.u32 r1, r2, -r3, r5;",
    ),
    (
        0x024,
        &[(75, 1), (81, 7), (87, 0xf)],
        "imad.u32 r1, r2, r3, -r5;",
    ),
    (
        0x024,
        &[(73, 1), (81, 7), (87, 0xf)],
        "imad r1, r2, r3, r5;",
    ),
    (
        0x024,
        &[(74, 1), (81, 7), (87, 6)],
        "imad.x.u32 r1, r2, r3, r5, p6;",
    ),
    (0x025, &[(73, 1), (81, 7)], "imad.wide r1, r2, r3, r5;"),
    (
        0x027,
        &[(73, 1), (81, 7), (87, 0xf)],
        "imad.hi r1, r2, r3, r5;",
    ),
    (0x026, &[(72, 1)], "idp.4a.u8.u8 r1, -r2, r3, r5;"),
    (0x026, &[(63, 1)], "idp.4a.u8.u8 r1, r2, -r3, r5;"),
    (0x026, &[(75, 1)], "idp.4a.u8.u8 r1, r2, r3, -r5;"),
    (0x026, &[(73, 1)], "idp.4a.s8.u8 r1, r2, r3, r5;"),
    (0x026, &[(74, 1)], "idp.4a.u8.s8 r1, r2, r3, r5;"),
    (0x026, &[(76, 1)], "idp.2a.lo.u16.u8 r1, r2, r3, r5;"),
    (
        0x026,
        &[(76, 1), (72, 1)],
        "idp.2a.lo.u16.u8 r1, -r2, r3, r5;",
    ),
    (
        0x026,
        &[(76, 1), (73, 1), (74, 1)],
        "idp.2a.lo.s16.s8 r1, r2, r3, r5;",
    ),
];

/// Each instruction's own fields and source modifiers, each set alone in a form 1 word, against
/// the text.
#[test]
fn hand_built_words_pin_each_field() {
    for cases in [&ADD_FIELDS[..], &LOGIC_FIELDS, &MULTIPLY_FIELDS] {
        for &(opcode, own, expected) in cases {
            assert_eq!(field_text(opcode, own), expected, "{opcode:#x} {own:?}");
        }
    }
}

/// `ISETP`'s fields, each set alone.
#[test]
fn hand_built_isetp_words_pin_each_field() {
    // ISETP: .EX (72) with the lower comparison (68..72), signed (73), each combining
    // operation (74..76) and each comparison (76..79); r2 at 24, r3 at 32.
    let isetp = |own: &[(u32, u128)]| {
        let mut fields = vec![(24, 2), (32, 3), (9, 1), (81, 0), (84, 7), (87, 7), (68, 7)];
        fields.extend_from_slice(own);
        text(word(0x00c, &fields))
    };
    assert_eq!(
        isetp(&[(72, 1)]),
        "isetp.f.u32.and.ex p0, pt, r2, r3, pt, pt;"
    );
    assert_eq!(isetp(&[(73, 1)]), "isetp.f.and p0, pt, r2, r3, pt;");
    assert_eq!(isetp(&[(74, 1)]), "isetp.f.u32.or p0, pt, r2, r3, pt;");
    assert_eq!(isetp(&[(74, 2)]), "isetp.f.u32.xor p0, pt, r2, r3, pt;");
    // The lower comparison !p1 (68..72) and the accumulator negated (90), without .EX.
    let low = word(
        0x20c,
        &[
            (24, 2),
            (32, 3),
            (81, 0),
            (84, 7),
            (87, 7),
            (90, 1),
            (68, 8 | 1),
        ],
    );
    assert_eq!(text(low), "isetp.f.u32.and p0, pt, r2, r3, !pt, !p1;");
    let comparisons = ["f", "lt", "eq", "le", "gt", "ne", "ge", "t"];
    for (value, name) in (0..).zip(comparisons) {
        assert_eq!(
            isetp(&[(76, value)]),
            format!("isetp.{name}.u32.and p0, pt, r2, r3, pt;")
        );
    }
}

/// `IADD3`'s carry inputs and `LEA`'s and `IMAD`'s without `.X` are kept but not printed: two
/// words that differ only there decode to different operations with the same text, and each
/// re-assembles to itself.
#[test]
fn carries_without_x_are_kept_but_not_printed() {
    let base = [(16, 1), (24, 2), (32, 3), (64, 5), (9, 1)];
    let pairs: [(u128, Fields, Fields); 4] = [
        (
            0x010,
            &[(77, 0xf), (81, 7), (84, 7), (87, 0xf)],
            &[(77, 2), (81, 7), (84, 7), (87, 0xf)],
        ),
        (
            0x010,
            &[(77, 0xf), (81, 7), (84, 7), (87, 0xf)],
            &[(77, 0xf), (81, 7), (84, 7), (87, 1)],
        ),
        (0x011, &[(81, 7)], &[(81, 7), (87, 0xf)]),
        (0x024, &[(81, 7)], &[(81, 7), (87, 0xf)]),
    ];
    for (opcode, one, other) in pairs {
        let words = [one, other].map(|own| word(opcode, &[&base[..], own].concat()));
        let decoded = words.map(|word| decode(word).expect("decodes"));
        assert_ne!(decoded[0], decoded[1], "{opcode:#x}");
        assert_eq!(decoded[0].to_string(), decoded[1].to_string());
        for (word, instr) in words.into_iter().zip(decoded) {
            assert_eq!(assemble(&instr), Ok(word));
        }
    }
}

/// `LEA` prints its high word with `.HI` (as `test_lea` does) or when it is not `RZ`, which NAK
/// writes without `.HI`.
#[test]
fn lea_prints_its_high_word_with_hi_or_when_not_rz() {
    let lea = |a_high, dst_high| {
        Instr::new(Op::Lea(OpLea {
            dst: r(0),
            overflow: Pred::PT,
            a: AluSrc::reg(r(1)),
            b: AluSrc::reg(r(2)),
            a_high,
            shift: 1,
            dst_high,
            sx32: false,
            intermediate: false,
            carry: ps(0, false),
        }))
    };
    assert_eq!(
        round_text(lea(AluSrc::reg(Gpr::RZ), false)),
        "lea r0, r1, r2, 0x1;"
    );
    assert_eq!(
        round_text(lea(AluSrc::reg(Gpr::RZ), true)),
        "lea.hi r0, r1, r2, rz, 0x1;"
    );
    assert_eq!(
        round_text(lea(AluSrc::reg(r(4)), false)),
        "lea r0, r1, r2, r4, 0x1;"
    );
}

#[test]
fn undefined_field_values_are_refused() {
    // ISETP's combining operation at 74..76: 2 (XOR) decodes, 3 does not.
    let isetp = |op| word(0x20c, &[(74, op), (81, 7), (84, 7), (87, 7), (68, 7)]);
    assert_eq!(text(isetp(2)), "isetp.f.u32.xor pt, pt, r0, r0, pt;");
    assert_eq!(
        decode(isetp(3)),
        Err(DecodeError::UnknownValue {
            opcode: 0x20c,
            bit: 74,
            width: 2,
            value: 3
        })
    );
    // PRMT's mode at 72..75: 6 (RC16) decodes, 7 does not, in every form.
    for form in [0x216, 0x416, 0x616, 0x816, 0xa16] {
        let prmt = |mode| word(form, &[(72, mode)]);
        assert!(decode(prmt(6)).is_ok(), "{form:#x}");
        assert_eq!(
            decode(prmt(7)),
            Err(DecodeError::UnknownValue {
                opcode: u16::try_from(form).expect("12 bits"),
                bit: 72,
                width: 3,
                value: 7
            })
        );
    }
    assert_eq!(
        decode(word(0x216, &[(72, 7)])).map_err(|error| error.to_string()),
        Err(
            "opcode 0x216 has the value 7 in its 3-bit field at bit 72, which its form does not \
             define"
                .into()
        )
    );
}

/// Every fixed field is required, naming its lowest clear bit, at each of its bits: `IMAD`'s,
/// `IMAD.WIDE`'s, `IMAD.HI`'s and `FLO`'s predicate destination (81..84), `LOP3`'s and
/// `IMAD.HI`'s predicate input (87..91).
#[test]
fn integer_fixed_fields_are_required() {
    let pred_dst = [81, 82, 83];
    let pred_input = [87, 88, 89, 90];
    let cases: [(u16, &[u32]); 7] = [
        (0x224, &pred_dst),
        (0x225, &pred_dst),
        (0x300, &pred_dst),
        (0x212, &pred_input),
        (0x227, &[81, 82, 83, 87, 88, 89, 90]),
        (0x827, &[81, 82, 83, 87, 88, 89, 90]),
        (0xa27, &[81, 82, 83, 87, 88, 89, 90]),
    ];
    for (opcode, fixed) in cases {
        let all = fixed.iter().fold(0_u128, |bits, &bit| bits | (1 << bit));
        assert_eq!(lookup(opcode).expect("a form").fixed, all, "{opcode:#x}");
        let full = Instruction::from_bits(all).with_field(OPCODE, u64::from(opcode));
        assert!(decode(full).is_ok(), "{opcode:#x}");
        for &bit in fixed {
            let missing = Instruction::from_bits(full.bits() & !(1 << bit));
            assert_eq!(
                decode(missing),
                Err(DecodeError::MissingBit { opcode, bit }),
                "{opcode:#x}"
            );
        }
    }
}

/// The order of checks with the group's faults: a stray bit (step 2) before a missing fixed
/// field (step 3) and before an undefined value (step 4).
#[test]
fn integer_checks_run_in_the_documented_order() {
    // Step 2 before step 3: a LOP3 without its predicate input and with bit 80 (.PAND) set.
    let lop3 = word(0x212, &[(80, 1)]);
    assert_eq!(
        decode(lop3),
        Err(DecodeError::UnknownBit {
            opcode: 0x212,
            bit: 80
        })
    );
    assert_eq!(
        decode(word(0x212, &[])),
        Err(DecodeError::MissingBit {
            opcode: 0x212,
            bit: 87
        })
    );
    // Step 2 before step 4: an ISETP with combining operation 3 and bit 62 set.
    assert_eq!(
        decode(word(0x20c, &[(74, 3), (62, 1)])),
        Err(DecodeError::UnknownBit {
            opcode: 0x20c,
            bit: 62
        })
    );
    // Step 3 cannot meet step 4 in one form of this group: the forms with fixed fields have no
    // field with undefined values.
    for form in integer::FORMS {
        let refuses_values = [0x00c, 0x016].contains(&base_of(form.opcode));
        assert!(!(refuses_values && form.fixed & !SLOT_UNIFORM.mask() != 0));
    }
}

/// Each bit a row marks known that the group's form rejects (the module docs' divergences) is
/// an unknown bit, at each such bit, on a word that otherwise decodes.
#[test]
fn integer_dropped_bits_are_refused() {
    let mut refused = 0;
    for row in integer::REFERENCE_ROWS {
        let form = lookup(row.opcode).expect("a form");
        let valid = Instruction::from_bits(form.fixed).with_field(OPCODE, u64::from(row.opcode));
        assert!(decode(valid).is_ok(), "{:#x}", row.opcode);
        for bit in 0..128 {
            if row.dropped & (1 << bit) == 0 {
                continue;
            }
            assert_eq!(
                decode(Instruction::from_bits(valid.bits() | (1 << bit))),
                Err(DecodeError::UnknownBit {
                    opcode: row.opcode,
                    bit
                }),
                "{:#x}",
                row.opcode
            );
            refused += 1;
        }
    }
    // The bits every row drops, summed over the rows (a row that loses one fails).
    assert_eq!(refused, 212);
}

/// `POPC`'s and `FLO`'s form 4 read bit 63 as the immediate's bit 31, where NAK and PR #4588
/// read it as the bitwise-not modifier (`op::integer`'s module docs, "Conflict" and the
/// divergence table): a word with bit 63 set decodes to the immediate `0x80000000`, unmodified,
/// and re-assembles to itself.
#[test]
fn popc_and_flo_form_4_read_bit_63_as_immediate_bit_31() {
    let popc = word(0x909, &[(16, 1), (32, 0x8000_0000)]);
    let decoded = decode(popc).expect("form 4 of POPC decodes");
    assert_eq!(
        decoded.op,
        Op::PopC(OpPopC {
            dst: r(1),
            src: AluSrc::Imm(0x8000_0000),
        })
    );
    assert_eq!(decoded.to_string(), "popc r1, 0x80000000;");
    assert_eq!(assemble(&decoded).map(Instruction::bits), Ok(popc.bits()));
    let flo = word(0x900, &[(16, 1), (32, 0x8000_0001), (81, 7)]);
    let decoded = decode(flo).expect("form 4 of FLO decodes");
    assert_eq!(
        decoded.op,
        Op::Flo(OpFlo {
            dst: r(1),
            src: AluSrc::Imm(0x8000_0001),
            signed: false,
            return_shift_amount: false,
        })
    );
    assert_eq!(decoded.to_string(), "flo.u32 r1, 0x80000001;");
    assert_eq!(assemble(&decoded).map(Instruction::bits), Ok(flo.bits()));
}

/// NAK's encoder asserts that `IADD3` and `IADD3.X` leave source 0 or source 1 unmodified and
/// that `LEA` and `LEA.X` leave the shifted value or `b` unmodified ("Hardware requires at
/// least one of these be unmodified"); this decoder and the assembler accept both modified
/// (`op::integer`'s module docs, the divergence table). Each such form 1 word decodes, prints
/// both modifiers and re-assembles to itself.
#[test]
fn both_modified_add_operands_are_accepted() {
    let cases: [FieldCase; 4] = [
        (
            0x010,
            &[(77, 0xf), (81, 7), (84, 7), (87, 0xf), (72, 1), (63, 1)],
            "iadd3 r1, -r2, -r3, r5;",
        ),
        (
            0x010,
            &[
                (77, 0xf),
                (81, 7),
                (84, 7),
                (87, 0xf),
                (72, 1),
                (63, 1),
                (74, 1),
            ],
            "iadd3.x r1, ~r2, ~r3, r5, !pt, !pt;",
        ),
        (
            0x011,
            &[(72, 1), (63, 1), (81, 7)],
            "lea r1, -r2, -r3, r5, 0x0;",
        ),
        (
            0x011,
            &[(72, 1), (63, 1), (74, 1), (81, 7), (87, 0xf)],
            "lea.x r1, ~r2, ~r3, r5, 0x0, !pt;",
        ),
    ];
    for (opcode, own, expected) in cases {
        assert_eq!(field_text(opcode, own), expected);
        let mut fields = vec![(16, 1), (24, 2), (32, 3), (64, 5), (9, 1)];
        fields.extend_from_slice(own);
        let built = word(opcode, &fields);
        let decoded = decode(built).expect("decodes");
        assert_eq!(
            assemble(&decoded).map(Instruction::bits),
            Ok(built.bits()),
            "{expected}"
        );
    }
}

/// The rows of PR #4588's table this group does not decode: forms 2, 3 and 7 of the
/// instructions without source 2, and `IMAD.HI`'s forms 2, 3, 6 and 7 (no rows).
#[test]
fn integer_rows_left_undecoded_are_unknown_opcodes() {
    let unknown = [
        0x407, 0x607, 0xe07, 0x40c, 0x60c, 0xe0c, 0x413, 0x613, 0xe13, 0x417, 0x617, 0xe17, 0x41a,
        0x61a, 0xe1a, 0x41b, 0x61b, 0xe1b, 0x500, 0x700, 0xf00, 0x501, 0x701, 0xf01, 0x509, 0x709,
        0xf09, 0x427, 0x627, 0xc27, 0xe27,
    ];
    for opcode in unknown {
        let word = Instruction::default().with_field(OPCODE, u64::from(opcode));
        assert_eq!(decode(word), Err(DecodeError::UnknownOpcode { opcode }));
    }
}

#[test]
fn integer_opcodes_match_the_references() {
    let opcodes = [
        (MOV_OPCODE, 0x002),
        (IADD3_OPCODE, 0x010),
        (LEA_OPCODE, 0x011),
        (LOP3_OPCODE, 0x012),
        (IABS_OPCODE, 0x013),
        (PRMT_OPCODE, 0x016),
        (IMNMX_OPCODE, 0x017),
        (SHF_OPCODE, 0x019),
        (SGXT_OPCODE, 0x01a),
        (BMSK_OPCODE, 0x01b),
        (IMAD_OPCODE, 0x024),
        (IMAD_WIDE_OPCODE, 0x025),
        (IDP_OPCODE, 0x026),
        (IMAD_HI_OPCODE, 0x027),
        (ISETP_OPCODE, 0x00c),
        (SEL_OPCODE, 0x007),
        (FLO_OPCODE, 0x100),
        (BREV_OPCODE, 0x101),
        (POPC_OPCODE, 0x109),
        (PLOP3_OPCODE, 0x81c),
        (P2R_OPCODE, 0x803),
        (R2P_OPCODE, 0x804),
    ];
    for (shipping, reference) in opcodes {
        assert_eq!(shipping, reference);
    }
    // Every form's 12-bit opcode, in table order, written out from PR #4588's rows.
    let all: Vec<u16> = integer::FORMS.iter().map(|form| form.opcode).collect();
    let expected = [
        0x202, 0x802, 0xa02, 0xc02, 0x210, 0x410, 0x610, 0x810, 0xa10, 0xc10, 0xe10, 0x211, 0x411,
        0x611, 0x811, 0xa11, 0xc11, 0xe11, 0x212, 0x412, 0x612, 0x812, 0xa12, 0xc12, 0xe12, 0x213,
        0x813, 0xa13, 0xc13, 0x216, 0x416, 0x616, 0x816, 0xa16, 0xc16, 0xe16, 0x217, 0x817, 0xa17,
        0xc17, 0x219, 0x419, 0x619, 0x819, 0xa19, 0xc19, 0xe19, 0x21a, 0x81a, 0xa1a, 0xc1a, 0x21b,
        0x81b, 0xa1b, 0xc1b, 0x224, 0x424, 0x624, 0x824, 0xa24, 0xc24, 0xe24, 0x225, 0x425, 0x625,
        0x825, 0xa25, 0xc25, 0xe25, 0x226, 0x426, 0x626, 0x826, 0xa26, 0xc26, 0xe26, 0x227, 0x827,
        0xa27, 0x20c, 0x80c, 0xa0c, 0xc0c, 0x207, 0x807, 0xa07, 0xc07, 0x300, 0x900, 0xb00, 0xd00,
        0x301, 0x901, 0xb01, 0xd01, 0x309, 0x909, 0xb09, 0xd09, 0x81c, 0x803, 0x804,
    ];
    assert_eq!(all, expected);
}

/// The group's field positions against the cited reference ranges, written out (NAK's
/// `set_field`, `set_bit` and helper ranges; PR #4588's `Take` calls).
#[test]
fn integer_fields_match_the_references() {
    let fields = [
        (MOV_QUAD_LANES, (72, 4)),
        (PRED_DST1, (84, 3)),
        (PRED_B, (77, 3)),
        (PRED_B_NOT, (80, 1)),
        (PRED_C, (68, 3)),
        (PRED_C_NOT, (71, 1)),
        (IADD3_X, (74, 1)),
        (LEA_INTERMEDIATE, (72, 1)),
        (LEA_SX32, (73, 1)),
        (LEA_X, (74, 1)),
        (LEA_SHIFT, (75, 5)),
        (LEA_HI, (80, 1)),
        (LOP3_LUT, (72, 8)),
        (PRMT_MODE, (72, 3)),
        (IMNMX_SIGNED, (73, 1)),
        (SHF_TYPE, (73, 2)),
        (SHF_WRAP, (75, 1)),
        (SHF_RIGHT, (76, 1)),
        (SHF_HI, (80, 1)),
        (SGXT_SIGNED, (73, 1)),
        (SGXT_WRAP, (75, 1)),
        (BMSK_WRAP, (75, 1)),
        (IMAD_SIGNED, (73, 1)),
        (IMAD_X, (74, 1)),
        (IDP_SIGNED_A, (73, 1)),
        (IDP_SIGNED_B, (74, 1)),
        (IDP_2A, (76, 1)),
        (ISETP_EX, (72, 1)),
        (ISETP_SIGNED, (73, 1)),
        (ISETP_SET_OP, (74, 2)),
        (ISETP_CMP, (76, 3)),
        (FLO_SIGNED, (73, 1)),
        (FLO_SH, (74, 1)),
        (PLOP3_LUT1, (16, 8)),
        (PLOP3_LUT0_LOW, (64, 3)),
        (PLOP3_LUT0_HIGH, (72, 5)),
        (PLOP3_UNIFORM, (67, 1)),
        (PR_MASK, (32, 7)),
        (R2P_BYTE, (76, 2)),
    ];
    for (shipping, (start, width)) in fields {
        assert_eq!((shipping.start, shipping.width), (start, width));
    }
    assert_eq!(NOT_PT_INPUT, 0xf << 87);
}

/// The enumerations' encodings against NAK's `set_int_cmp_op`, `set_pred_set_op`, `OpShf`'s
/// and `OpPrmt`'s encoders, written out.
#[test]
fn integer_enumerations_match_the_references() {
    let cmp = [
        (IntCmpOp::False, 0),
        (IntCmpOp::True, 7),
        (IntCmpOp::Eq, 2),
        (IntCmpOp::Ne, 5),
        (IntCmpOp::Lt, 1),
        (IntCmpOp::Le, 3),
        (IntCmpOp::Gt, 4),
        (IntCmpOp::Ge, 6),
    ];
    for (op, bits) in cmp {
        assert_eq!(op.bits(), bits);
        assert_eq!(IntCmpOp::from_bits(bits), op);
    }
    for (op, bits) in [(PredSetOp::And, 0), (PredSetOp::Or, 1), (PredSetOp::Xor, 2)] {
        assert_eq!(op.bits(), bits);
        assert_eq!(PredSetOp::from_bits(bits), Some(op));
    }
    assert_eq!(PredSetOp::from_bits(3), None);
    let shf = [
        (ShfType::I64, 0),
        (ShfType::U64, 1),
        (ShfType::I32, 2),
        (ShfType::U32, 3),
    ];
    for (ty, bits) in shf {
        assert_eq!(ty.bits(), bits);
        assert_eq!(ShfType::from_bits(bits), ty);
    }
    let prmt = [
        (PrmtMode::Index, 0),
        (PrmtMode::Forward4Extract, 1),
        (PrmtMode::Backward4Extract, 2),
        (PrmtMode::Replicate8, 3),
        (PrmtMode::EdgeClampLeft, 4),
        (PrmtMode::EdgeClampRight, 5),
        (PrmtMode::Replicate16, 6),
    ];
    for (mode, bits) in prmt {
        assert_eq!(mode.bits(), bits);
        assert_eq!(PrmtMode::from_bits(bits), Some(mode));
    }
    assert_eq!(PrmtMode::from_bits(7), None);
}

/// Which sources read a negation bit, and where, in each form (the module docs' table), against
/// positions written out: source 0 at 72, the wide slot at 63, the register slot at 75; none
/// for an immediate.
#[test]
fn negation_bits_are_where_the_slots_put_them() {
    use AluForm::{CBufSrc1, CBufSrc2, ImmSrc1, ImmSrc2, Reg, UniformSrc1, UniformSrc2};
    let at = |field: Option<Field>| field.map(|field| field.start);
    let neg = IntSrc::Neg;
    let wide = IntSrc::NegWide;
    let cases = [
        (Reg, [Some(72), Some(63), Some(75)], [None, Some(63), None]),
        (ImmSrc2, [Some(72), Some(75), None], [None, None, None]),
        (
            CBufSrc2,
            [Some(72), Some(75), Some(63)],
            [None, None, Some(63)],
        ),
        (ImmSrc1, [Some(72), None, Some(75)], [None, None, None]),
        (
            CBufSrc1,
            [Some(72), Some(63), Some(75)],
            [None, Some(63), None],
        ),
        (
            UniformSrc1,
            [Some(72), Some(63), Some(75)],
            [None, Some(63), None],
        ),
        (
            UniformSrc2,
            [Some(72), Some(75), Some(63)],
            [None, None, Some(63)],
        ),
    ];
    for (form, negs, wides) in cases {
        for index in 0..3 {
            assert_eq!(
                at(neg_field(form, index, neg)),
                negs[index],
                "{form:?} {index}"
            );
            assert_eq!(
                at(neg_field(form, index, wide)),
                wides[index],
                "{form:?} {index}"
            );
            assert_eq!(neg_field(form, index, IntSrc::Plain), None);
            assert_eq!(neg_field(form, index, IntSrc::Absent), None);
        }
    }
}

/// The assembler refuses what the decoder cannot produce, naming the source or the form, and
/// fields past their width.
#[test]
fn integer_unencodable_operations_are_errors() {
    let imad = |srcs, mode| {
        assemble(&Instr::new(Op::IMad(OpIMad {
            dst: r(0),
            srcs,
            signed: false,
            mode,
        })))
    };
    let reg = AluSrc::reg(r(1));
    let ureg = AluSrc::UniformReg {
        reg: ur(1),
        abs: false,
        neg: false,
    };
    let imm = AluSrc::Imm(1);
    // IMAD.HI has forms 1, 4 and 5 only.
    assert_eq!(
        imad([reg, ureg, reg], IMadMode::Hi),
        Err(EncodeError::NoForm {
            mnemonic: "imad",
            form: 6
        })
    );
    assert_eq!(
        imad([reg, reg, imm], IMadMode::Hi).map_err(|error| error.to_string()),
        Err("imad has no encoding in operand form 2".into())
    );
    assert!(imad([reg, imm, reg], IMadMode::Hi).is_ok());
    // IMAD.HI has no negation bits; IMAD has.
    assert_eq!(
        imad([reg, neg(r(2)), reg], IMadMode::Hi),
        Err(EncodeError::Modifier {
            mnemonic: "imad",
            index: 1
        })
    );
    let lo = IMadMode::Lo {
        x: false,
        carry: PredSrc::TRUE,
    };
    assert!(imad([reg, neg(r(2)), reg], lo).is_ok());
    // A modifier is refused before the form: both faults at once report the modifier.
    assert_eq!(
        imad([reg, ureg, neg(r(2))], IMadMode::Hi),
        Err(EncodeError::Modifier {
            mnemonic: "imad",
            index: 2
        })
    );
}

/// The assembler refuses an absolute value on any integer source, a negation where the source
/// has no bit for it, and `IDP.2A` outside form 1.
#[test]
fn integer_unencodable_sources_are_errors() {
    let reg = AluSrc::reg(r(1));
    let imm = AluSrc::Imm(1);
    // No integer source has an absolute value.
    let abs = AluSrc::Reg {
        reg: r(2),
        abs: true,
        neg: false,
    };
    let iadd3 = OpIAdd3 {
        dst: r(0),
        overflow: [Pred::PT; 2],
        srcs: [abs, reg, reg],
        carry: [PredSrc::TRUE; 2],
    };
    assert_eq!(
        assemble(&Instr::new(Op::IAdd3(iadd3))),
        Err(EncodeError::Modifier {
            mnemonic: "iadd3",
            index: 0
        })
    );
    // LEA's b has a negation bit in the wide slot only: in form 2 (an immediate a_high) it is
    // in the register slot, whose negation bit is the shift.
    let lea = |b, a_high| {
        assemble(&Instr::new(Op::Lea(OpLea {
            dst: r(0),
            overflow: Pred::PT,
            a: reg,
            b,
            a_high,
            shift: 0,
            dst_high: true,
            sx32: false,
            intermediate: false,
            carry: PredSrc::TRUE,
        })))
    };
    assert_eq!(
        lea(neg(r(2)), imm),
        Err(EncodeError::Modifier {
            mnemonic: "lea",
            index: 1
        })
    );
    assert!(lea(neg(r(2)), reg).is_ok());
    // LOP3's sources have no negation bit.
    let lop3 = OpLop3 {
        pred: Pred::PT,
        dst: r(0),
        srcs: [reg, reg, neg(r(2))],
        lut: 0,
    };
    assert_eq!(
        assemble(&Instr::new(Op::Lop3(lop3))),
        Err(EncodeError::Modifier {
            mnemonic: "lop3",
            index: 2
        })
    );
    // IDP.2A is in form 1 only.
    let idp = |srcs| {
        assemble(&Instr::new(Op::IDp4(OpIDp4 {
            dst: r(0),
            srcs,
            signed: [false; 2],
            two_a: true,
        })))
    };
    assert!(idp([reg, reg, reg]).is_ok());
    assert_eq!(
        idp([reg, imm, reg]),
        Err(EncodeError::NoForm {
            mnemonic: "idp",
            form: 4
        })
    );
}

/// Each width-checked field at its last value and one past it.
#[test]
fn integer_fields_past_their_width_are_refused() {
    let lea = |shift| {
        assemble(&Instr::new(Op::Lea(OpLea {
            dst: r(0),
            overflow: Pred::PT,
            a: AluSrc::reg(r(1)),
            b: AluSrc::reg(r(2)),
            a_high: AluSrc::reg(Gpr::RZ),
            shift,
            dst_high: false,
            sx32: false,
            intermediate: false,
            carry: PredSrc::TRUE,
        })))
    };
    assert!(lea(31).is_ok());
    assert_eq!(
        lea(32),
        Err(EncodeError::TooWide {
            what: "shift",
            value: 32,
            bits: 5
        })
    );
    let p2r = |mask| {
        assemble(&Instr::new(Op::P2R(OpP2R {
            dst: r(0),
            src: r(1),
            mask,
        })))
    };
    assert!(p2r(0x7f).is_ok());
    assert_eq!(
        p2r(0x80),
        Err(EncodeError::TooWide {
            what: "predicate mask",
            value: 0x80,
            bits: 7
        })
    );
    let r2p = |byte, mask| {
        assemble(&Instr::new(Op::R2P(OpR2P {
            src: r(1),
            byte,
            mask,
        })))
    };
    assert!(r2p(3, 0x7f).is_ok());
    assert_eq!(
        r2p(4, 0),
        Err(EncodeError::TooWide {
            what: "byte",
            value: 4,
            bits: 2
        })
    );
    assert_eq!(
        r2p(0, 0x80),
        Err(EncodeError::TooWide {
            what: "predicate mask",
            value: 0x80,
            bits: 7
        })
    );
}

/// The printer's text for operands and modifiers NAK's tests do not cover. **Unconfirmed**: these
/// pin the printer's guesses until a developer with `nvdisasm` runs the cross-check.
#[test]
fn unconfirmed_text_of_the_integer_group() {
    let cbuf = AluSrc::CBuf {
        cbuf: CBufRef {
            buf: CBuf::Binding(1),
            offset: 8,
        },
        abs: false,
        neg: true,
    };
    let cases = [
        (
            Op::IAdd3X(OpIAdd3X {
                add: OpIAdd3 {
                    dst: r(4),
                    overflow: [p(1), Pred::PT],
                    srcs: [neg(r(1)), AluSrc::reg(r(2)), cbuf],
                    carry: [ps(0, false), ps(2, true)],
                },
            }),
            "iadd3.x r4, p1, pt, ~r1, r2, ~c[0x1][0x8], p0, !p2;",
        ),
        (
            Op::IMad64(OpIMad64 {
                dst: r(6),
                srcs: [AluSrc::reg(r(1)), AluSrc::Imm(0x10), AluSrc::reg(r(8))],
                signed: true,
                carry: ps(0, false),
            }),
            "imad.wide r6, r1, 0x10, r8;",
        ),
        (
            Op::Shf(OpShf {
                dst: r(1),
                low: AluSrc::reg(Gpr::RZ),
                shift: AluSrc::Imm(0x1f),
                high: AluSrc::reg(r(3)),
                data_type: ShfType::U32,
                wrap: false,
                right: true,
                dst_high: true,
            }),
            "shf.r.u32.hi r1, rz, 0x1f, r3;",
        ),
    ];
    for (op, expected) in cases {
        assert_eq!(round_text(Instr::new(op)), expected);
    }
}

/// More unconfirmed text: `PRMT`, `ISETP`, `PLOP3` with a uniform predicate, `LOP3`.
#[test]
fn unconfirmed_text_of_the_integer_group_continued() {
    let cases = [
        (
            Op::Prmt(OpPrmt {
                dst: r(1),
                srcs: [AluSrc::reg(r(2)), AluSrc::reg(r(3))],
                sel: AluSrc::Imm(0x3210),
                mode: PrmtMode::Index,
            }),
            "prmt r1, r2, 0x3210, r3;",
        ),
        (
            Op::ISetP(OpISetP {
                dsts: [p(0), Pred::PT],
                srcs: [AluSrc::reg(r(1)), AluSrc::reg(r(2))],
                cmp_op: IntCmpOp::Ge,
                signed: false,
                set_op: PredSetOp::And,
                ex: true,
                accum: PredSrc::TRUE,
                low_cmp: ps(1, false),
            }),
            "isetp.ge.u32.and.ex p0, pt, r1, r2, pt, p1;",
        ),
        (
            Op::PLop3(OpPLop3 {
                dsts: [p(0), Pred::PT],
                srcs: [PredSrc::TRUE, ps(1, true)],
                c: PLop3Src::Uniform(UniformPredSrc {
                    reg: UniformPred::UPT,
                    negated: true,
                }),
                luts: [0x80, 0],
            }),
            "plop3.lut p0, pt, pt, !p1, !upt, 0x80, 0x0;",
        ),
        (
            Op::Lop3(OpLop3 {
                pred: Pred::PT,
                dst: r(0),
                srcs: [AluSrc::reg(r(1)), AluSrc::reg(Gpr::RZ), AluSrc::reg(r(2))],
                lut: 0xfc,
            }),
            "lop3.lut r0, r1, rz, r2, 0xfc, !pt;",
        ),
    ];
    for (op, expected) in cases {
        assert_eq!(round_text(Instr::new(op)), expected);
    }
}

fn any_integer_word() -> impl Strategy<Value = Instruction> {
    let forms: Vec<(u16, u128, u128)> = integer::FORMS
        .iter()
        .map(|form| (form.opcode, form.known, form.fixed))
        .collect();
    (0..forms.len(), any::<u128>()).prop_map(move |(index, noise)| {
        let (opcode, known, fixed) = forms[index];
        Instruction::from_bits((noise & known) | fixed).with_field(OPCODE, u64::from(opcode))
    })
}

proptest! {
    #![proptest_config(config())]

    /// The group's operations alone: assembling then decoding gives the instruction back, so
    /// every field is read from where the encoder put it.
    #[test]
    fn integer_assemble_then_decode_round_trips(
        guard in crate::testing::any_pred_src(),
        op in Union::new(integer::any_ops()),
    ) {
        let instr = Instr { guard, op };
        let word = assemble(&instr).expect("the generator makes encodable instructions");
        prop_assert_eq!(decode(word), Ok(instr));
    }

    /// The group's words alone: a word decodes and re-assembles to itself (scheduling bits
    /// aside), or is refused for a bit of the other constant-buffer layout, or for `ISETP`'s
    /// combining operation 3 or `PRMT`'s mode 7.
    #[test]
    fn integer_decode_then_assemble_round_trips(word in any_integer_word()) {
        match decode(word) {
            Ok(instr) => {
                let again = assemble(&instr).expect("a decoded instruction is encodable");
                prop_assert_eq!(again.bits(), word.bits() & !SCHEDULING.mask());
            }
            Err(DecodeError::UnknownBit { opcode, bit }) => {
                // Only the other constant-buffer layout's bits (forms 3 and 5): a bound buffer
                // does not read the bindless register's bits, nor a bindless one the index.
                let read = bits_read(word).expect("the form's function accepts the word");
                prop_assert_eq!(read & (1 << bit), 0);
                let layout = SLOT_UNIFORM_REG.mask() | CBUF_INDEX.mask();
                prop_assert!(layout & (1 << bit) != 0, "{opcode:#x} bit {bit}");
                prop_assert!([3, 5].contains(&(opcode >> 9)), "{opcode:#x}");
            }
            Err(DecodeError::UnknownValue { opcode, bit, width, value }) => {
                let base = base_of(opcode);
                prop_assert!(
                    (base == 0x00c && (bit, width, value) == (74, 2, 3))
                        || (base == 0x016 && (bit, width, value) == (72, 3, 7)),
                    "{opcode:#x}"
                );
            }
            Err(error) => prop_assert!(false, "{error}"),
        }
    }
}
