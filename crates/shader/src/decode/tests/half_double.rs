//! Tests of slice 6c, the half and double precision group (`op::half_double`): NAK's expected
//! text, one hand-built word per form and per modifier bit, every swizzle value in each swizzle
//! field, the values and bits the group refuses, the divergences from PR #4588's table, the
//! reference constants written out, and generated round trips of the group alone.

use proptest::prelude::*;
use proptest::strategy::Union;

use super::super::alu::{HALF_SRC0, WIDE_OPERAND_FORMS, half_fields};
use super::super::op::half_double::{self, *};
use super::*;
use crate::asm::{EncodeError, assemble};

/// A packed pair of halves, 1.0 high and 2.0 low, as an immediate.
const HALVES: u128 = 0x3c00_4000;
/// The high word of the double 1.0, as an immediate.
const DOUBLE_ONE: u128 = 0x3ff0_0000;

/// The 12-bit opcode of a word.
fn opcode_of(word: Instruction) -> u16 {
    u16::try_from(word.field(OPCODE)).expect("12 bits")
}

/// Decodes `opcode` with the fields of `base` and each case's extra fields, checking the text.
fn check_bits(opcode: u128, base: &[(u32, u128)], cases: &[(&[(u32, u128)], &str)]) {
    for (extra, expected) in cases {
        let fields: Vec<(u32, u128)> = base.iter().chain(extra.iter()).copied().collect();
        let word = word(opcode, &fields);
        assert_eq!(text(word), *expected, "{:#x}", word.bits());
    }
}

/// Fields of a hand-built word: `(start, value)` pairs.
type Fields<'a> = &'a [(u32, u128)];
/// How NAK's `Src` printer writes a register under one modifier.
type ModText = fn(&str) -> String;

/// The error for `value` in the field at `bit` of `width` bits of `opcode`.
fn unknown_value(opcode: u16, bit: u32, width: u32, value: u64) -> Result<Instr, DecodeError> {
    Err(DecodeError::UnknownValue {
        opcode,
        bit,
        width,
        value,
    })
}

// ---------------------------------------------------------------------------------------------
// Expected text.

/// NAK's `test_hfma2`, ported (`nvdisasm_tests.rs` @ 52238762): `hfma2 r0, r1, r2, r3` with
/// every combination of `SrcMod::None`, `FAbs`, `FNeg` and `FNegAbs` on the three sources, and
/// the text NAK checks against `nvdisasm` (NAK's `Src` printer: `|r1|`, `-r1`, `-|r1|`).
#[test]
fn hfma2_text_matches_nak_nvdisasm_cases() {
    // (abs, neg) of NAK's four modifiers, and how NAK's `Src` printer writes register `r`.
    let mods: [((bool, bool), ModText); 4] = [
        ((false, false), |r| r.to_string()),
        ((true, false), |r| format!("|{r}|")),
        ((false, true), |r| format!("-{r}")),
        ((true, true), |r| format!("-|{r}|")),
    ];
    let src = |index: u8, (abs, neg): (bool, bool)| HalfSrc {
        src: AluSrc::Reg {
            reg: r(index),
            abs,
            neg,
        },
        swizzle: HalfSwizzle::H1H0,
    };
    let mut checked = 0;
    for (a_mod, a_text) in mods {
        for (b_mod, b_text) in mods {
            for (c_mod, c_text) in mods {
                let instr = Instr::new(Op::HFma2(OpHFma2 {
                    dst: r(0),
                    srcs: [src(1, a_mod), src(2, b_mod), src(3, c_mod)],
                    saturate: false,
                    f32: false,
                    ftz: false,
                    dnz: false,
                    relu: None,
                    bf16: false,
                    neg_high_b: false,
                }));
                let expected = format!(
                    "hfma2 r0, {}, {}, {};",
                    a_text("r1"),
                    b_text("r2"),
                    c_text("r3")
                );
                assert_eq!(round_text(instr), expected);
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 64);
}

// ---------------------------------------------------------------------------------------------
// Hand-built words.

/// The destination r1 at 16 and source 0 r2 at 24.
const DST_A: &[(u32, u128)] = &[(16, 1), (24, 2)];
/// A bound constant buffer `c[0x1][0x8]`: offset 8 at 38, index 1 at 54.
const CBUF: &[(u32, u128)] = &[(38, 8), (54, 1)];
/// Source 0 r2, comparison 1 (`.lt`), destinations p1 at 81 and pt at 84, the accumulator PT.
const PREDICATE_BASE: &[(u32, u128)] = &[(24, 2), (76, 1), (81, 1), (84, 7), (87, 7)];

/// The fields of `base` and then `more`.
fn with(base: &[(u32, u128)], more: &[(u32, u128)]) -> Vec<(u32, u128)> {
    base.iter().chain(more).copied().collect()
}

/// Hand-built words of `HADD2`, `HFMA2` and `HMUL2`, one per form (and a bindless constant
/// buffer), and their text.
fn half_words() -> Vec<(Instruction, String)> {
    let mut cases: Vec<(Instruction, String)> = Vec::new();
    let mut push = |opcode: u128, fields: &[(u32, u128)], expected: &str| {
        cases.push((word(opcode, fields), expected.to_string()));
    };
    // HADD2: the second operand in the wide slot.
    push(0x230, &with(DST_A, &[(32, 3)]), "hadd2 r1, r2, r3;");
    push(
        0x430,
        &with(DST_A, &[(32, HALVES)]),
        "hadd2 r1, r2, 0x3c004000;",
    );
    push(0x630, &with(DST_A, CBUF), "hadd2 r1, r2, c[0x1][0x8];");
    push(
        0xe30,
        &with(DST_A, &[(32, 3), (91, 1)]),
        "hadd2 r1, r2, ur3;",
    );
    // A bindless constant buffer, which PR #4588's translator refuses.
    push(
        0x630,
        &with(DST_A, &[(32, 5), (38, 8), (91, 1)]),
        "hadd2 r1, r2, cx[ur5][0x8];",
    );
    // HFMA2 in its seven forms.
    let hfma2 = with(DST_A, &[(64, 4)]);
    push(0x231, &with(&hfma2, &[(32, 3)]), "hfma2 r1, r2, r3, r4;");
    push(
        0x431,
        &with(&hfma2, &[(32, HALVES)]),
        "hfma2 r1, r2, r4, 0x3c004000;",
    );
    push(0x631, &with(&hfma2, CBUF), "hfma2 r1, r2, r4, c[0x1][0x8];");
    push(
        0x831,
        &with(&hfma2, &[(32, HALVES)]),
        "hfma2 r1, r2, 0x3c004000, r4;",
    );
    push(0xa31, &with(&hfma2, CBUF), "hfma2 r1, r2, c[0x1][0x8], r4;");
    push(
        0xc31,
        &with(&hfma2, &[(32, 3), (91, 1)]),
        "hfma2 r1, r2, ur3, r4;",
    );
    push(
        0xe31,
        &with(&hfma2, &[(32, 3), (91, 1)]),
        "hfma2 r1, r2, r4, ur3;",
    );
    // HMUL2: the second operand in source 1.
    push(0x232, &with(DST_A, &[(32, 3)]), "hmul2 r1, r2, r3;");
    push(
        0x832,
        &with(DST_A, &[(32, HALVES)]),
        "hmul2 r1, r2, 0x3c004000;",
    );
    push(0xa32, &with(DST_A, CBUF), "hmul2 r1, r2, c[0x1][0x8];");
    push(
        0xc32,
        &with(DST_A, &[(32, 3), (91, 1)]),
        "hmul2 r1, r2, ur3;",
    );
    cases
}

/// Hand-built words of the packed-half comparisons and `HMNMX2`, one per form, and their text.
fn half_comparison_words() -> Vec<(Instruction, String)> {
    let mut cases: Vec<(Instruction, String)> = Vec::new();
    let mut push = |opcode: u128, fields: &[(u32, u128)], expected: &str| {
        cases.push((word(opcode, fields), expected.to_string()));
    };
    // HSET2, with .bf as NAK writes it, comparison 2 (.eq) and the accumulator PT.
    let set_base = with(DST_A, &[(71, 1), (76, 2), (87, 7)]);
    push(
        0x233,
        &with(&set_base, &[(32, 3)]),
        "hset2.bf.eq.and r1, r2, r3, pt;",
    );
    push(
        0x433,
        &with(&set_base, &[(32, HALVES)]),
        "hset2.bf.eq.and r1, r2, 0x3c004000, pt;",
    );
    push(
        0x633,
        &with(&set_base, CBUF),
        "hset2.bf.eq.and r1, r2, c[0x1][0x8], pt;",
    );
    push(
        0xe33,
        &with(&set_base, &[(32, 3), (91, 1)]),
        "hset2.bf.eq.and r1, r2, ur3, pt;",
    );
    // HSETP2: comparison 1 (.lt), destinations p1 and pt, the accumulator PT.
    push(
        0x234,
        &with(PREDICATE_BASE, &[(32, 3)]),
        "hsetp2.lt.and p1, pt, r2, r3, pt;",
    );
    push(
        0x434,
        &with(PREDICATE_BASE, &[(32, HALVES)]),
        "hsetp2.lt.and p1, pt, r2, 0x3c004000, pt;",
    );
    push(
        0x634,
        &with(PREDICATE_BASE, CBUF),
        "hsetp2.lt.and p1, pt, r2, c[0x1][0x8], pt;",
    );
    push(
        0xe34,
        &with(PREDICATE_BASE, &[(32, 3), (91, 1)]),
        "hsetp2.lt.and p1, pt, r2, ur3, pt;",
    );
    // HMNMX2 with the predicate p1.
    let hmnmx2 = with(DST_A, &[(87, 1)]);
    push(0x240, &with(&hmnmx2, &[(32, 3)]), "hmnmx2 r1, r2, r3, p1;");
    push(
        0x840,
        &with(&hmnmx2, &[(32, HALVES)]),
        "hmnmx2 r1, r2, 0x3c004000, p1;",
    );
    push(
        0xa40,
        &with(&hmnmx2, CBUF),
        "hmnmx2 r1, r2, c[0x1][0x8], p1;",
    );
    push(
        0xc40,
        &with(&hmnmx2, &[(32, 3), (91, 1)]),
        "hmnmx2 r1, r2, ur3, p1;",
    );
    cases
}

/// Hand-built words of the double instructions, one per form, and their text.
fn double_words() -> Vec<(Instruction, String)> {
    let mut cases: Vec<(Instruction, String)> = Vec::new();
    let mut push = |opcode: u128, fields: &[(u32, u128)], expected: &str| {
        cases.push((word(opcode, fields), expected.to_string()));
    };
    // DADD: the second operand in source 2, the register slot in form 1.
    push(0x229, &with(DST_A, &[(64, 4)]), "dadd r1, r2, r4;");
    push(
        0x429,
        &with(DST_A, &[(32, DOUBLE_ONE)]),
        "dadd r1, r2, 0x3ff00000;",
    );
    push(0x629, &with(DST_A, CBUF), "dadd r1, r2, c[0x1][0x8];");
    push(
        0xe29,
        &with(DST_A, &[(32, 3), (91, 1)]),
        "dadd r1, r2, ur3;",
    );
    // DMUL: the second operand in source 1.
    push(0x228, &with(DST_A, &[(32, 3)]), "dmul r1, r2, r3;");
    push(
        0x828,
        &with(DST_A, &[(32, DOUBLE_ONE)]),
        "dmul r1, r2, 0x3ff00000;",
    );
    push(0xa28, &with(DST_A, CBUF), "dmul r1, r2, c[0x1][0x8];");
    push(
        0xc28,
        &with(DST_A, &[(32, 3), (91, 1)]),
        "dmul r1, r2, ur3;",
    );
    // DFMA in its seven forms.
    let dfma = with(DST_A, &[(64, 4)]);
    push(0x22b, &with(&dfma, &[(32, 3)]), "dfma r1, r2, r3, r4;");
    push(
        0x42b,
        &with(&dfma, &[(32, DOUBLE_ONE)]),
        "dfma r1, r2, r4, 0x3ff00000;",
    );
    push(0x62b, &with(&dfma, CBUF), "dfma r1, r2, r4, c[0x1][0x8];");
    push(
        0x82b,
        &with(&dfma, &[(32, DOUBLE_ONE)]),
        "dfma r1, r2, 0x3ff00000, r4;",
    );
    push(0xa2b, &with(&dfma, CBUF), "dfma r1, r2, c[0x1][0x8], r4;");
    push(
        0xc2b,
        &with(&dfma, &[(32, 3), (91, 1)]),
        "dfma r1, r2, ur3, r4;",
    );
    push(
        0xe2b,
        &with(&dfma, &[(32, 3), (91, 1)]),
        "dfma r1, r2, r4, ur3;",
    );
    // DSETP: as HSETP2.
    push(
        0x22a,
        &with(PREDICATE_BASE, &[(32, 3)]),
        "dsetp.lt.and p1, pt, r2, r3, pt;",
    );
    push(
        0x42a,
        &with(PREDICATE_BASE, &[(32, DOUBLE_ONE)]),
        "dsetp.lt.and p1, pt, r2, 0x3ff00000, pt;",
    );
    push(
        0x62a,
        &with(PREDICATE_BASE, CBUF),
        "dsetp.lt.and p1, pt, r2, c[0x1][0x8], pt;",
    );
    push(
        0xe2a,
        &with(PREDICATE_BASE, &[(32, 3), (91, 1)]),
        "dsetp.lt.and p1, pt, r2, ur3, pt;",
    );
    cases
}

/// One hand-built word of every form of the group, and its text. Fields: destination at 16,
/// source 0 at 24, the wide slot at 32 (register, immediate, or a constant buffer's offset at 38
/// and index at 54, or a uniform register with bit 91), the register slot at 64, a comparison at
/// 76, predicate destinations at 81 and 84, the accumulator at 87.
#[test]
fn hand_built_words_of_each_form_decode() {
    let cases = [half_words(), half_comparison_words(), double_words()].concat();
    let mut opcodes: Vec<u16> = Vec::new();
    for (word, expected) in &cases {
        assert_eq!(text(*word), *expected, "{:#x}", word.bits());
        opcodes.push(opcode_of(*word));
    }
    for form in half_double::FORMS {
        assert!(opcodes.contains(&form.opcode), "{:#x}", form.opcode);
    }
}

/// Each modifier, flag and mode bit of `HADD2` and `HFMA2` set alone (or with the bits it
/// needs) in a word the assembler did not make, against the same word without it.
#[test]
fn hand_built_words_pin_each_arithmetic_modifier_bit() {
    check_bits(
        0x230,
        &[(16, 1), (24, 2), (32, 3)],
        &[
            (&[], "hadd2 r1, r2, r3;"),
            (&[(72, 1)], "hadd2 r1, -r2, r3;"),
            (&[(73, 1)], "hadd2 r1, |r2|, r3;"),
            (&[(62, 1)], "hadd2 r1, r2, |r3|;"),
            (&[(63, 1)], "hadd2 r1, r2, -r3;"),
            (&[(77, 1)], "hadd2.sat r1, r2, r3;"),
            (&[(78, 1)], "hadd2.f32 r1, r2, r3;"),
            (&[(80, 1)], "hadd2.ftz r1, r2, r3;"),
            // `.F32` with `.sat` or `.ftz`, which both of PR #4588's paths refuse.
            (&[(78, 1), (77, 1)], "hadd2.f32.sat r1, r2, r3;"),
            (&[(78, 1), (80, 1)], "hadd2.f32.ftz r1, r2, r3;"),
        ],
    );
    let hfma2 = [(16, 1), (24, 2), (32, 3), (64, 4)];
    check_bits(
        0x231,
        &hfma2,
        &[
            (&[], "hfma2 r1, r2, r3, r4;"),
            (&[(72, 1)], "hfma2 r1, -r2, r3, r4;"),
            (&[(73, 1)], "hfma2 r1, |r2|, r3, r4;"),
            (&[(62, 1)], "hfma2 r1, r2, |r3|, r4;"),
            (&[(63, 1)], "hfma2 r1, r2, -r3, r4;"),
            (&[(83, 1)], "hfma2 r1, r2, r3, |r4|;"),
            (&[(84, 1)], "hfma2 r1, r2, r3, -r4;"),
            (&[(76, 1)], "hfma2.fmz r1, r2, r3, r4;"),
            (&[(77, 1)], "hfma2.sat r1, r2, r3, r4;"),
            (&[(78, 1)], "hfma2.f32 r1, r2, r3, r4;"),
            (&[(79, 1), (87, 7)], "hfma2.relu r1, r2, r3, r4, pt;"),
            (&[(80, 1)], "hfma2.ftz r1, r2, r3, r4;"),
            (&[(85, 1)], "hfma2.bf16_v2 r1, r2, r3, r4;"),
            (&[(86, 1)], "hfma2 r1, r2, r3.h0_nh1, r4;"),
            // `.relu` with `.sat`, which both of PR #4588's paths refuse.
            (
                &[(79, 1), (87, 7), (77, 1)],
                "hfma2.sat.relu r1, r2, r3, r4, pt;",
            ),
        ],
    );
    // In form 2, `b` is in the register slot, with its modifiers at 83 and 84.
    check_bits(
        0x431,
        &[(16, 1), (24, 2), (32, HALVES), (64, 4)],
        &[
            (&[], "hfma2 r1, r2, r4, 0x3c004000;"),
            (&[(83, 1)], "hfma2 r1, r2, |r4|, 0x3c004000;"),
            (&[(84, 1)], "hfma2 r1, r2, -r4, 0x3c004000;"),
            (&[(81, 2)], "hfma2 r1, r2, r4.h0_h0, 0x3c004000;"),
        ],
    );
}

/// Each modifier, flag and predicate bit of `HMUL2`, `HSET2`, `HSETP2` and `HMNMX2` set alone.
#[test]
fn hand_built_words_pin_each_comparison_modifier_bit() {
    check_bits(
        0x232,
        &[(16, 1), (24, 2), (32, 3)],
        &[
            (&[], "hmul2 r1, r2, r3;"),
            (&[(72, 1)], "hmul2 r1, -r2, r3;"),
            (&[(73, 1)], "hmul2 r1, |r2|, r3;"),
            (&[(62, 1)], "hmul2 r1, r2, |r3|;"),
            (&[(63, 1)], "hmul2 r1, r2, -r3;"),
            (&[(76, 1)], "hmul2.fmz r1, r2, r3;"),
            (&[(77, 1)], "hmul2.sat r1, r2, r3;"),
            (&[(79, 1)], "hmul2.relu r1, r2, r3;"),
            (&[(80, 1)], "hmul2.ftz r1, r2, r3;"),
        ],
    );
    check_bits(
        0x233,
        &[(16, 1), (24, 2), (32, 3), (76, 2), (87, 7)],
        &[
            (&[], "hset2.eq.and r1, r2, r3, pt;"),
            (&[(71, 1)], "hset2.bf.eq.and r1, r2, r3, pt;"),
            (&[(69, 1)], "hset2.eq.or r1, r2, r3, pt;"),
            (&[(69, 2)], "hset2.eq.xor r1, r2, r3, pt;"),
            (&[(80, 1)], "hset2.eq.ftz.and r1, r2, r3, pt;"),
            (&[(90, 1)], "hset2.eq.and r1, r2, r3, !pt;"),
            (&[(72, 1)], "hset2.eq.and r1, -r2, r3, pt;"),
            (&[(73, 1)], "hset2.eq.and r1, |r2|, r3, pt;"),
            (&[(62, 1)], "hset2.eq.and r1, r2, |r3|, pt;"),
            (&[(63, 1)], "hset2.eq.and r1, r2, -r3, pt;"),
        ],
    );
    check_bits(
        0x234,
        &[(24, 2), (32, 3), (76, 1), (81, 1), (84, 7), (87, 7)],
        &[
            (&[], "hsetp2.lt.and p1, pt, r2, r3, pt;"),
            (&[(71, 1)], "hsetp2.lt.h_and.and p1, pt, r2, r3, pt;"),
            // `.h_and` before `.ftz`.
            (
                &[(71, 1), (80, 1)],
                "hsetp2.lt.h_and.ftz.and p1, pt, r2, r3, pt;",
            ),
            (&[(69, 1)], "hsetp2.lt.or p1, pt, r2, r3, pt;"),
            (&[(80, 1)], "hsetp2.lt.ftz.and p1, pt, r2, r3, pt;"),
            (&[(90, 1)], "hsetp2.lt.and p1, pt, r2, r3, !pt;"),
            (&[(72, 1)], "hsetp2.lt.and p1, pt, -r2, r3, pt;"),
            (&[(73, 1)], "hsetp2.lt.and p1, pt, |r2|, r3, pt;"),
            (&[(62, 1)], "hsetp2.lt.and p1, pt, r2, |r3|, pt;"),
            (&[(63, 1)], "hsetp2.lt.and p1, pt, r2, -r3, pt;"),
        ],
    );
    check_bits(
        0x240,
        &[(16, 1), (24, 2), (32, 3), (87, 1)],
        &[
            (&[], "hmnmx2 r1, r2, r3, p1;"),
            (&[(78, 1)], "hmnmx2.f32 r1, r2, r3, p1;"),
            (&[(80, 1)], "hmnmx2.ftz r1, r2, r3, p1;"),
            (&[(81, 1)], "hmnmx2.nan r1, r2, r3, p1;"),
            (&[(82, 1)], "hmnmx2.xorsign r1, r2, r3, p1;"),
            (&[(85, 1)], "hmnmx2.bf16_v2 r1, r2, r3, p1;"),
            (&[(90, 1)], "hmnmx2 r1, r2, r3, !p1;"),
            (&[(72, 1)], "hmnmx2 r1, -r2, r3, p1;"),
            (&[(73, 1)], "hmnmx2 r1, |r2|, r3, p1;"),
            (&[(62, 1)], "hmnmx2 r1, r2, |r3|, p1;"),
            (&[(63, 1)], "hmnmx2 r1, r2, -r3, p1;"),
        ],
    );
}

/// Each modifier, rounding and predicate bit of the double instructions set alone.
#[test]
fn hand_built_words_pin_each_double_modifier_bit() {
    // DADD's form 1 keeps `b` in the register slot, with its modifiers at 74 and 75.
    check_bits(
        0x229,
        &[(16, 1), (24, 2), (64, 4)],
        &[
            (&[], "dadd r1, r2, r4;"),
            (&[(72, 1)], "dadd r1, -r2, r4;"),
            (&[(73, 1)], "dadd r1, |r2|, r4;"),
            (&[(74, 1)], "dadd r1, r2, |r4|;"),
            (&[(75, 1)], "dadd r1, r2, -r4;"),
            (&[(78, 1)], "dadd.rm r1, r2, r4;"),
            (&[(78, 2)], "dadd.rp r1, r2, r4;"),
            (&[(78, 3)], "dadd.rz r1, r2, r4;"),
        ],
    );
    check_bits(
        0x228,
        &[(16, 1), (24, 2), (32, 3)],
        &[
            (&[], "dmul r1, r2, r3;"),
            (&[(72, 1)], "dmul r1, -r2, r3;"),
            (&[(73, 1)], "dmul r1, |r2|, r3;"),
            (&[(62, 1)], "dmul r1, r2, |r3|;"),
            (&[(63, 1)], "dmul r1, r2, -r3;"),
            (&[(78, 3)], "dmul.rz r1, r2, r3;"),
        ],
    );
    check_bits(
        0x22b,
        &[(16, 1), (24, 2), (32, 3), (64, 4)],
        &[
            (&[], "dfma r1, r2, r3, r4;"),
            (&[(72, 1)], "dfma r1, -r2, r3, r4;"),
            (&[(73, 1)], "dfma r1, |r2|, r3, r4;"),
            (&[(62, 1)], "dfma r1, r2, |r3|, r4;"),
            (&[(63, 1)], "dfma r1, r2, -r3, r4;"),
            (&[(74, 1)], "dfma r1, r2, r3, |r4|;"),
            (&[(75, 1)], "dfma r1, r2, r3, -r4;"),
            (&[(78, 1)], "dfma.rm r1, r2, r3, r4;"),
        ],
    );
    check_bits(
        0x22a,
        &[(24, 2), (32, 3), (76, 1), (81, 1), (84, 7), (87, 7)],
        &[
            (&[], "dsetp.lt.and p1, pt, r2, r3, pt;"),
            (&[(74, 1)], "dsetp.lt.or p1, pt, r2, r3, pt;"),
            (&[(74, 2)], "dsetp.lt.xor p1, pt, r2, r3, pt;"),
            (&[(76, 13)], "dsetp.neu.and p1, pt, r2, r3, pt;"),
            (&[(90, 1)], "dsetp.lt.and p1, pt, r2, r3, !pt;"),
            (&[(72, 1)], "dsetp.lt.and p1, pt, -r2, r3, pt;"),
            (&[(73, 1)], "dsetp.lt.and p1, pt, |r2|, r3, pt;"),
            (&[(62, 1)], "dsetp.lt.and p1, pt, r2, |r3|, pt;"),
            (&[(63, 1)], "dsetp.lt.and p1, pt, r2, -r3, pt;"),
        ],
    );
}

// ---------------------------------------------------------------------------------------------
// Swizzles.

/// Every swizzle value in each of the three swizzle fields, in `HFMA2`'s register form, which
/// has all three (source 0's at 74, `b`'s in the wide slot at 60, `c`'s in the register slot at
/// 81), with the bits on both sides of the field clear and set: each of the four values decodes
/// to its swizzle (1 to `.f32`, a guess; `HalfSwizzle`'s docs) and leaves the neighbours'
/// meaning alone. Bit 59, below the wide slot's field, is in no register form (the register is
/// bits 32 to 39), so only the bit above it, `b`'s absolute value at 62, is a neighbour there.
#[test]
fn every_swizzle_value_at_both_edges_of_its_field() {
    type Neighbours<'a> = &'a [(&'a [(u32, u128)], &'a str)];
    let base = [(16, 1), (24, 2), (32, 3), (64, 4)];
    // (field start, neighbour bits with the text they give; `{}` is where the swizzle's name
    // goes).
    let fields: [(u32, Neighbours<'_>); 3] = [
        (
            74,
            &[
                (&[], "hfma2 r1, r2{}, r3, r4;"),
                (&[(73, 1)], "hfma2 r1, |r2{}|, r3, r4;"),
                (&[(76, 1)], "hfma2.fmz r1, r2{}, r3, r4;"),
                (&[(73, 1), (76, 1)], "hfma2.fmz r1, |r2{}|, r3, r4;"),
            ],
        ),
        (
            60,
            &[
                (&[], "hfma2 r1, r2, r3{}, r4;"),
                (&[(62, 1)], "hfma2 r1, r2, |r3{}|, r4;"),
            ],
        ),
        (
            81,
            &[
                (&[], "hfma2 r1, r2, r3, r4{};"),
                (&[(80, 1)], "hfma2.ftz r1, r2, r3, r4{};"),
                (&[(83, 1)], "hfma2 r1, r2, r3, |r4{}|;"),
                (&[(80, 1), (83, 1)], "hfma2.ftz r1, r2, r3, |r4{}|;"),
            ],
        ),
    ];
    let names = [(0, ""), (1, ".f32"), (2, ".h0_h0"), (3, ".h1_h1")];
    let mut checked = 0;
    for (start, neighbours) in fields {
        for (extra, template) in neighbours {
            let fields: Vec<(u32, u128)> = base.iter().chain(extra.iter()).copied().collect();
            for (value, name) in names {
                let mut with_swizzle = fields.clone();
                with_swizzle.push((start, value));
                assert_eq!(
                    text(word(0x231, &with_swizzle)),
                    template.replace("{}", name),
                    "bit {start} value {value}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, (4 + 2 + 4) * 4);
    // The other wide-slot kinds read the same swizzle field: a constant buffer and a uniform
    // register, and in `HMUL2`'s form 5 a swizzle beside a bindless buffer.
    assert_eq!(
        text(word(
            0xa31,
            &[(16, 1), (24, 2), (38, 8), (54, 1), (60, 3), (64, 4)]
        )),
        "hfma2 r1, r2, c[0x1][0x8].h1_h1, r4;"
    );
    assert_eq!(
        text(word(
            0xc31,
            &[(16, 1), (24, 2), (32, 3), (60, 2), (91, 1), (64, 4)]
        )),
        "hfma2 r1, r2, ur3.h0_h0, r4;"
    );
    assert_eq!(
        text(word(
            0xa32,
            &[(16, 1), (24, 2), (32, 5), (38, 8), (60, 3), (91, 1)]
        )),
        "hmul2 r1, r2, cx[ur5][0x8].h1_h1;"
    );
    // An immediate's bits 60 and 61 are part of the value, not a swizzle.
    assert_eq!(
        text(word(0x832, &[(16, 1), (24, 2), (32, 0x1000_0000)])),
        "hmul2 r1, r2, 0x10000000;"
    );
}

/// Swizzle 1 decodes as `.f32` in each field of each packed-half instruction that has it, as
/// PR #4588's lowering passes it into Maxwell's swizzle for `HADD2`, `HMUL2`, `HFMA2` and
/// `HSET2` (graphics-server decision 8); in `HMNMX2` and in `HSETP2` (its lowering's `0x434`
/// case and the native translator refuse it there) it decodes too, a listed divergence that
/// leaves the refusal to the translator.
#[test]
fn swizzle_1_decodes_as_f32() {
    let cases: [(u16, Fields<'_>, &str); 12] = [
        (0x230, &[(74, 1)][..], "hadd2 r0, r0.f32, r0;"),
        (0x230, &[(60, 1)][..], "hadd2 r0, r0, r0.f32;"),
        (0x231, &[(81, 1)][..], "hfma2 r0, r0, r0, r0.f32;"),
        (0x232, &[(60, 1)][..], "hmul2 r0, r0, r0.f32;"),
        (0x233, &[(74, 1)][..], "hset2.f.and r0, r0.f32, r0, p0;"),
        (
            0x234,
            &[(60, 1)][..],
            "hsetp2.f.and p0, p0, r0, r0.f32, p0;",
        ),
        (
            0x434,
            &[(74, 1)][..],
            "hsetp2.f.and p0, p0, r0.f32, 0x0, p0;",
        ),
        (0x240, &[(74, 1)][..], "hmnmx2 r0, r0.f32, r0, p0;"),
        (0x240, &[(60, 1)][..], "hmnmx2 r0, r0, r0.f32, p0;"),
        (0x431, &[(81, 1)][..], "hfma2 r0, r0, r0.f32, 0x0;"),
        (
            0x630,
            &[(60, 1), (54, 1)][..],
            "hadd2 r0, r0, c[0x1][0x0].f32;",
        ),
        (
            0xe34,
            &[(60, 1), (91, 1)][..],
            "hsetp2.f.and p0, p0, r0, ur0.f32, p0;",
        ),
    ];
    for (opcode, fields, expected) in cases {
        assert_eq!(
            text(word(u128::from(opcode), fields)),
            expected,
            "{opcode:#x}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Values refused, and the order of the checks.

/// `HFMA2`'s bits 87 to 90 with `.RELU` are its predicate, index and negation, as PR #4588's
/// lowering reads them (`insn.Bits(87, 4) << 39` into Maxwell's `SEL`; 7, `PT`, the
/// always-true case); every one of the 16 values decodes, the negation bit 90 included. Without
/// `.RELU` both of PR #4588's paths require them clear (the lowering's `Bits(87, 4) != 0`, the
/// native translator's `Take(87, 4) == 0`): 0 decodes and every other value is refused with the
/// whole field.
#[test]
fn hfma2_relu_predicate_follows_relu() {
    let base = [(16, 1), (24, 2), (32, 3), (64, 4)];
    for value in 0..16_u64 {
        let index = u8::try_from(value & 7).expect("three bits");
        let pred = PredSrc {
            reg: Pred::new(index).expect("a predicate"),
            negated: value & 8 != 0,
        };
        let mut fields = base.to_vec();
        fields.extend([(79, 1), (87, u128::from(value))]);
        assert_eq!(
            text(word(0x231, &fields)),
            format!("hfma2.relu r1, r2, r3, r4, {pred};"),
            "value {value}"
        );
        let mut fields = base.to_vec();
        fields.push((87, u128::from(value)));
        let decoded = decode(word(0x231, &fields));
        if value == 0 {
            assert_eq!(text(word(0x231, &fields)), "hfma2 r1, r2, r3, r4;");
        } else {
            assert_eq!(decoded, unknown_value(0x231, 87, 4, value), "value {value}");
        }
    }
    // Bit 90 alone, the negation, without `.RELU`; with it, `!pt`.
    assert_eq!(
        decode(word(0x231, &[(90, 1)])),
        unknown_value(0x231, 87, 4, 8)
    );
    assert_eq!(
        text(word(0x231, &[(90, 1)])),
        "error: opcode 0x231 has the value 8 in its 4-bit field at bit 87, which its form does \
         not define"
    );
    assert_eq!(
        text(word(0x231, &[(79, 1), (87, 7), (90, 1)])),
        "hfma2.relu r0, r0, r0, r0, !pt;"
    );
}

/// `HSETP2`'s form 2 reads `b` from the wide slot (an immediate), as NAK encodes it and PR
/// #4588's lowering reads it (case `0x434`: `half_source(2)`), not from the register slot as
/// its native translator does (`Visitor::HSETP2`: `HalfSource(1)`), whose bits overlap the
/// combine operation (69 and 70) and `.H_AND` (71): with both set, they keep their meaning and
/// `b` stays the immediate.
#[test]
fn hsetp2_immediate_form_reads_b_from_the_wide_slot() {
    assert_eq!(
        text(word(
            0x434,
            &with(PREDICATE_BASE, &[(32, HALVES), (69, 1), (71, 1)])
        )),
        "hsetp2.lt.h_and.or p1, pt, r2, 0x3c004000, pt;"
    );
}

/// The combine operation 3 of `HSET2`, `HSETP2` (bits 69 and 70) and `DSETP` (74 and 75) is refused
/// in every form, with the exact field.
#[test]
fn combine_operation_3_is_refused() {
    for form in WIDE_OPERAND_FORMS {
        for (base, bit) in [(HSET2_OPCODE, 69), (HSETP2_OPCODE, 69), (DSETP_OPCODE, 74)] {
            let opcode = form.opcode(base);
            let word = canonical_word(opcode).with_field(Field::new(bit, 2), 3);
            assert_eq!(
                decode(word),
                unknown_value(opcode, bit, 2, 3),
                "{opcode:#x}"
            );
            assert!(
                decode(canonical_word(opcode).with_field(Field::new(bit, 2), 2)).is_ok(),
                "{opcode:#x}"
            );
        }
    }
}

/// The framework's steps 2 and 3 come before the group's step-4 checks. Each instruction has at
/// most one step-4 check (the combine operation, or `HFMA2`'s bits 87 to 90 without `.RELU`),
/// and a swizzle names something in every value, so a swizzle of 1 beside either check leaves
/// the check's error unchanged (in `HSET2` and `HSETP2`, whose combine operation is read before
/// the sources, and in `HFMA2`, whose bits are read after them).
#[test]
fn half_double_checks_run_in_the_documented_order() {
    // The combine operation, with source 0's swizzle 1 beside it.
    assert_eq!(
        decode(word(0x233, &[(69, 3), (74, 1)])),
        unknown_value(0x233, 69, 2, 3)
    );
    assert_eq!(
        decode(word(0x234, &[(69, 3), (74, 1)])),
        unknown_value(0x234, 69, 2, 3)
    );
    // HFMA2's bits 87 to 90 without `.RELU`, with `c`'s swizzle 1 beside them.
    assert_eq!(
        decode(word(0x231, &[(81, 1), (87, 1)])),
        unknown_value(0x231, 87, 4, 1)
    );
    // Step 2 (an unknown bit) before step 4.
    assert_eq!(
        decode(word(0x233, &[(69, 3), (64, 1)])),
        Err(DecodeError::UnknownBit {
            opcode: 0x233,
            bit: 64
        })
    );
    // Step 3 (bit 91 of a uniform form) before step 4.
    assert_eq!(
        decode(word(0xe34, &[(69, 3)])),
        Err(DecodeError::MissingBit {
            opcode: 0xe34,
            bit: 91
        })
    );
}

/// Every uniform-register form of the group requires bit 91.
#[test]
fn half_double_uniform_forms_require_bit_91() {
    let uniform: Vec<u16> = half_double::FORMS
        .iter()
        .map(|form| form.opcode)
        .filter(|opcode| matches!(opcode >> 9, 6 | 7))
        .collect();
    assert_eq!(
        uniform,
        [
            0xe30, 0xc31, 0xe31, 0xc32, 0xe33, 0xe34, 0xc40, 0xe29, 0xc28, 0xc2b, 0xe2b, 0xe2a
        ]
    );
    for opcode in uniform {
        let bare = Instruction::default().with_field(OPCODE, u64::from(opcode));
        assert_eq!(
            decode(bare),
            Err(DecodeError::MissingBit { opcode, bit: 91 }),
            "{opcode:#x}"
        );
        assert!(decode(canonical_word(opcode)).is_ok(), "{opcode:#x}");
    }
}

// ---------------------------------------------------------------------------------------------
// Divergences.

/// Each of the group's rows drops exactly the bits the module docs list, written out, and each
/// dropped bit set alone in an otherwise valid word is refused, naming it; only `HFMA2`'s rows
/// add a bit (90).
#[test]
fn half_double_rows_drop_exactly_the_listed_bits() {
    let range = |start: u32, end: u32| (start..end).fold(0_u128, |bits, bit| bits | 1 << bit);
    let expected = |opcode: u16| -> u128 {
        match opcode & 0x1ff {
            // HADD2: the register slot (64..72, 81..85) and bit 85.
            0x030 => range(64, 72) | range(81, 86),
            // HMUL2: bits 78 and 85.
            0x032 => 1 << 78 | 1 << 85,
            // HSET2: 64..69 and 81..85.
            0x033 => range(64, 69) | range(81, 85),
            // HSETP2: 64..69.
            0x034 => range(64, 69),
            // DSETP: 64..72.
            0x02a => range(64, 72),
            _ => 0,
        }
    };
    let mut dropping = 0;
    for (form, row) in half_double::FORMS.iter().zip(half_double::REFERENCE_ROWS) {
        let opcode = form.opcode;
        assert_eq!(row.dropped, expected(opcode), "{opcode:#x}");
        // HFMA2 adds bit 90, `.RELU`'s predicate negation; no other form adds a bit.
        let added = if opcode & 0x1ff == 0x031 { 1 << 90 } else { 0 };
        assert_eq!(row.added(), added, "{opcode:#x}");
        if row.dropped != 0 {
            dropping += 1;
        }
        let base = canonical_word(opcode);
        assert!(decode(base).is_ok(), "{opcode:#x}");
        for bit in 0..128 {
            if row.dropped & (1 << bit) != 0 {
                assert_eq!(
                    decode(Instruction::from_bits(base.bits() | (1 << bit))),
                    Err(DecodeError::UnknownBit { opcode, bit }),
                    "{opcode:#x} bit {bit}"
                );
            }
        }
    }
    // HADD2, HMUL2, HSET2, HSETP2 and DSETP, four forms each.
    assert_eq!(dropping, 5 * 4);
}

/// PR #4588's rows of these instructions that NAK never encodes are unknown opcodes.
#[test]
fn half_double_rows_left_undecoded_are_unknown_opcodes() {
    let unknown = [
        // HADD2, HSET2, HSETP2, DADD and DSETP forms 4, 5, 6.
        0x830, 0xa30, 0xc30, 0x833, 0xa33, 0xc33, 0x834, 0xa34, 0xc34, 0x829, 0xa29, 0xc29, 0x82a,
        0xa2a, 0xc2a, // HMUL2, HMNMX2 and DMUL forms 2, 3, 7.
        0x432, 0x632, 0xe32, 0x440, 0x640, 0xe40, 0x428, 0x628, 0xe28,
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

// ---------------------------------------------------------------------------------------------
// The test assembler's refusals.

/// An immediate has no swizzle bits, so the assembler refuses one with a swizzle; the first
/// source must be a register; and at most one source takes the wide slot.
#[test]
fn unencodable_half_sources_are_refused() {
    let reg = |index| HalfSrc::new(AluSrc::reg(r(index)));
    let hadd2 = |a, b| {
        assemble(&Instr::new(Op::HAdd2(OpHAdd2 {
            dst: r(1),
            srcs: [a, b],
            saturate: false,
            f32: false,
            ftz: false,
        })))
    };
    let swizzled_imm = HalfSrc {
        src: AluSrc::Imm(1),
        swizzle: HalfSwizzle::H0H0,
    };
    assert_eq!(
        hadd2(reg(2), swizzled_imm),
        Err(EncodeError::NoBits {
            mnemonic: "hadd2",
            what: "a swizzle on an immediate"
        })
    );
    assert_eq!(
        hadd2(reg(2), swizzled_imm).map_err(|error| error.to_string()),
        Err("hadd2 has no bits for a swizzle on an immediate".into())
    );
    assert!(hadd2(reg(2), HalfSrc::new(AluSrc::Imm(1))).is_ok());
    assert_eq!(
        hadd2(HalfSrc::new(AluSrc::Imm(1)), reg(2)),
        Err(EncodeError::FirstSourceNotRegister { mnemonic: "hadd2" })
    );
    let hfma2 = |b, c| {
        assemble(&Instr::new(Op::HFma2(OpHFma2 {
            dst: r(1),
            srcs: [reg(2), b, c],
            saturate: false,
            f32: false,
            ftz: false,
            dnz: false,
            relu: None,
            bf16: false,
            neg_high_b: false,
        })))
    };
    let imm = HalfSrc::new(AluSrc::Imm(1));
    assert_eq!(
        hfma2(imm, imm),
        Err(EncodeError::TwoWideSources { mnemonic: "hfma2" })
    );
    assert_eq!(
        hfma2(reg(3), swizzled_imm),
        Err(EncodeError::NoBits {
            mnemonic: "hfma2",
            what: "a swizzle on an immediate"
        })
    );
    let dfma = |b, c| {
        assemble(&Instr::new(Op::DFma(OpDFma {
            dst: r(1),
            srcs: [AluSrc::reg(r(2)), b, c],
            rnd: RoundMode::NearestEven,
        })))
    };
    assert_eq!(
        dfma(AluSrc::Imm(1), AluSrc::Imm(2)),
        Err(EncodeError::TwoWideSources { mnemonic: "dfma" })
    );
}

/// The assembler writes each packed-half source's modifiers and swizzle where its slot keeps
/// them: source 0's at 72, 73 and 74; the wide slot's at 62, 63 and 60; the register slot's at
/// 83, 84 and 81 (not the 32-bit positions 74 and 75); HFMA2's `.relu` predicate goes to bits
/// 87 to 90.
#[test]
fn assembled_words_place_each_half_field() {
    let src = |index, abs, neg, swizzle| HalfSrc {
        src: AluSrc::Reg {
            reg: r(index),
            abs,
            neg,
        },
        swizzle,
    };
    let hfma2 = OpHFma2 {
        dst: r(1),
        srcs: [
            src(2, true, false, HalfSwizzle::H0H0),
            src(3, false, true, HalfSwizzle::H1H1),
            src(4, true, true, HalfSwizzle::H0H0),
        ],
        saturate: false,
        f32: false,
        ftz: false,
        dnz: false,
        relu: Some(PredSrc {
            reg: p(2),
            negated: true,
        }),
        bf16: false,
        neg_high_b: false,
    };
    let assembled = assemble(&Instr::new(Op::HFma2(hfma2))).expect("encodable");
    let expected = word(
        0x231,
        &[
            (16, 1),
            (24, 2),
            (73, 1),
            (74, 2),
            (32, 3),
            (63, 1),
            (60, 3),
            (64, 4),
            (83, 1),
            (84, 1),
            (81, 2),
            (79, 1),
            (87, 2),
            (90, 1),
        ],
    );
    assert_eq!(assembled.bits(), expected.bits());
    // The shared field tables the decoder and the assembler both use.
    assert_eq!(
        (
            HALF_SRC0.abs.start,
            HALF_SRC0.neg.start,
            HALF_SRC0.swizzle.start
        ),
        (73, 72, 74)
    );
    let wide = half_fields(Slot::WideReg).expect("fields");
    assert_eq!(
        (wide.abs.start, wide.neg.start, wide.swizzle.start),
        (62, 63, 60)
    );
    let reg_slot = half_fields(Slot::Reg).expect("fields");
    assert_eq!(
        (
            reg_slot.abs.start,
            reg_slot.neg.start,
            reg_slot.swizzle.start
        ),
        (83, 84, 81)
    );
    assert_eq!(half_fields(Slot::CBuf), Some(wide));
    assert_eq!(half_fields(Slot::UniformReg), Some(wide));
    assert_eq!(half_fields(Slot::Imm), None);
}

// ---------------------------------------------------------------------------------------------
// The reference constants, written out.

/// The group's opcodes against NAK's `encode_alu` and `encode_fp16v2_alu` calls, and its forms'
/// 12-bit opcodes.
#[test]
fn half_double_opcodes_match_the_references() {
    let opcodes = [
        (HADD2_OPCODE, 0x030),
        (HFMA2_OPCODE, 0x031),
        (HMUL2_OPCODE, 0x032),
        (HSET2_OPCODE, 0x033),
        (HSETP2_OPCODE, 0x034),
        (HMNMX2_OPCODE, 0x040),
        (DMUL_OPCODE, 0x028),
        (DADD_OPCODE, 0x029),
        (DSETP_OPCODE, 0x02a),
        (DFMA_OPCODE, 0x02b),
    ];
    for (shipping, reference) in opcodes {
        assert_eq!(shipping, reference);
    }
    let forms: Vec<u16> = half_double::FORMS.iter().map(|form| form.opcode).collect();
    assert_eq!(
        forms,
        [
            0x230, 0x430, 0x630, 0xe30, 0x231, 0x431, 0x631, 0x831, 0xa31, 0xc31, 0xe31, 0x232,
            0x832, 0xa32, 0xc32, 0x233, 0x433, 0x633, 0xe33, 0x234, 0x434, 0x634, 0xe34, 0x240,
            0x840, 0xa40, 0xc40, 0x229, 0x429, 0x629, 0xe29, 0x228, 0x828, 0xa28, 0xc28, 0x22b,
            0x42b, 0x62b, 0x82b, 0xa2b, 0xc2b, 0xe2b, 0x22a, 0x42a, 0x62a, 0xe2a
        ]
    );
}

/// The group's and the shared packed-half fields' positions against the cited reference ranges,
/// written out (NAK's `set_bit`, `set_field`, `set_pred_set_op` and `set_alu_reg` ranges; PR
/// #4588's `Take` calls).
#[test]
fn half_double_fields_match_the_references() {
    let fields = [
        (HALF_F32, (78, 1)),
        (RELU, (79, 1)),
        (BF16_V2, (85, 1)),
        (HFMA2_NEG_HIGH_B, (86, 1)),
        (HFMA2_RELU_PRED, (87, 4)),
        (HALF_SET_OP, (69, 2)),
        (HSET2_BF, (71, 1)),
        (HSETP2_H_AND, (71, 1)),
        (HMNMX2_NAN, (81, 1)),
        (HMNMX2_XORSIGN, (82, 1)),
        (SRC0_SWIZZLE, (74, 2)),
        (SLOT_SWIZZLE, (60, 2)),
        (SRC2_HALF_SWIZZLE, (81, 2)),
        (SRC2_HALF_ABS, (83, 1)),
        (SRC2_HALF_NEG, (84, 1)),
    ];
    for (shipping, (start, width)) in fields {
        assert_eq!((shipping.start, shipping.width), (start, width));
    }
    // `.RELU`'s predicate is the accumulator's fields, read together.
    assert_eq!(HFMA2_RELU_PRED.mask(), COND.mask() | COND_NOT.mask());
}

/// The swizzle encodings against NAK's `set_swizzle` for an `F16v2` source (`None` 0, `Xx` 2,
/// `Yy` 3) and Eden's Maxwell `Swizzle` (`F32` 1); a value wider than the field names nothing,
/// and the field is two bits wide.
#[test]
fn half_swizzle_encodings_match_the_references() {
    let swizzles = [
        (HalfSwizzle::H1H0, 0),
        (HalfSwizzle::F32, 1),
        (HalfSwizzle::H0H0, 2),
        (HalfSwizzle::H1H1, 3),
    ];
    assert_eq!(HalfSwizzle::ALL.len(), swizzles.len());
    for (swizzle, bits) in swizzles {
        assert_eq!(swizzle.bits(), bits);
        assert_eq!(HalfSwizzle::from_bits(bits), Some(swizzle));
    }
    assert_eq!(HalfSwizzle::from_bits(4), None);
    assert_eq!(HalfSwizzle::BITS, 2);
    assert_eq!(HalfSrc::new(AluSrc::Imm(5)).swizzle, HalfSwizzle::H1H0);
}

// ---------------------------------------------------------------------------------------------
// Generated round trips of the group alone.

/// Whether `bit` of a word of `opcode` is one the documented layouts read only sometimes: the
/// other constant-buffer layout's bits in a constant-buffer form.
fn layout_bit(opcode: u16, bit: u32) -> bool {
    matches!(opcode >> 9, 3 | 5) && matches!(bit, 32..38 | 54..59)
}

proptest! {
    #![proptest_config(config())]

    /// The group's operations alone, so each gets many more cases than in the crate-wide round
    /// trip: assembling then decoding gives the instruction back.
    #[test]
    fn half_double_assemble_then_decode_round_trips(
        guard in crate::testing::any_pred_src(),
        op in Union::new(half_double::any_ops()),
    ) {
        let instr = Instr { guard, op };
        let word = assemble(&instr).expect("the generator makes encodable instructions");
        prop_assert_eq!(decode(word), Ok(instr));
    }

    /// The group's words alone: a word decodes and re-assembles to itself (scheduling bits
    /// aside), or is refused for a bit only another layout reads, for a combine operation of 3,
    /// or for `HFMA2` bits 87 to 90 set without `.RELU`.
    #[test]
    fn half_double_decode_then_assemble_round_trips(
        word in any_word_of(half_double::FORMS),
    ) {
        match decode(word) {
            Ok(instr) => {
                let again = assemble(&instr).expect("a decoded instruction is encodable");
                prop_assert_eq!(again.bits(), word.bits() & !SCHEDULING.mask());
            }
            Err(DecodeError::UnknownBit { opcode, bit }) => {
                let read = bits_read(word).expect("the form's function accepts the word");
                prop_assert_eq!(read & (1 << bit), 0);
                prop_assert!(layout_bit(opcode, bit), "{opcode:#x} bit {bit}");
            }
            Err(DecodeError::UnknownValue { opcode, bit, width, value }) => {
                let base = opcode & 0x1ff;
                let set_op = width == 2
                    && value == 3
                    && ((matches!(base, HSET2_OPCODE | HSETP2_OPCODE) && bit == 69)
                        || (base == DSETP_OPCODE && bit == 74));
                let relu = base == HFMA2_OPCODE && bit == 87 && width == 4 && value != 0;
                prop_assert!(set_op || relu, "{opcode:#x} bit {bit}");
            }
            Err(error) => prop_assert!(false, "{error}"),
        }
    }
}
