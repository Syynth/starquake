//! Tests of slice 6b, the 32-bit float and conversion group (`op::float`): NAK's expected text,
//! one hand-built word per form and per modifier bit, the values and bits the group refuses, the
//! divergences from PR #4588's table, the reference constants written out, and generated round
//! trips of the group alone.

use proptest::prelude::*;
use proptest::strategy::Union;

use super::super::op::float::{self, *};
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

/// Decodes a hand-built word and prints it, or the error.
fn text(word: Instruction) -> String {
    match decode(word) {
        Ok(instr) => instr.to_string(),
        Err(error) => format!("error: {error}"),
    }
}

/// The 12-bit opcode of a word.
fn opcode_of(word: Instruction) -> u16 {
    u16::try_from(word.field(OPCODE)).expect("12 bits")
}

fn ur(index: u8) -> UniformReg {
    UniformReg::new(index).expect("a uniform register")
}

/// The integer type of size encoding `size` (0 to 3) and signedness.
fn int(size: u8, signed: bool) -> IntType {
    IntType::new(size, signed).expect("a size of 0 to 3")
}

fn imm(value: f32) -> AluSrc {
    AluSrc::Imm(value.to_bits())
}

/// The bits of 2.0 and 1.0 as 32-bit floats, for hand-built immediates.
const TWO: u128 = 0x4000_0000;
const ONE: u128 = 0x3f80_0000;

// ---------------------------------------------------------------------------------------------
// NAK's expected text.

/// NAK's two-source pairs in `test_float_ops`: `r2, r3` and `r2, 2`, with the register NAK's
/// encoder leaves in `FADD`'s unread slot for each (R0 in form 1, `RZ` in form 2).
fn nak_two_src() -> [([AluSrc; 2], &'static str, Gpr); 2] {
    [
        (
            [AluSrc::reg(Gpr::new(2)), AluSrc::reg(Gpr::new(3))],
            "r2, r3",
            Gpr::new(0),
        ),
        ([AluSrc::reg(Gpr::new(2)), imm(2.0)], "r2, 2", Gpr::RZ),
    ]
}

/// `.ftz` off and on, as NAK's tests loop over it.
const NAK_FTZ: [(bool, &str); 2] = [(false, ""), (true, ".ftz")];

/// The rounding modes NAK's tests use, with their float and integer text.
const NAK_RND: [(RoundMode, &str, &str); 2] = [
    (RoundMode::NearestEven, "", ""),
    (RoundMode::Zero, ".rz", ".trunc"),
];

/// Assembles, decodes and prints each op, checking it against its text; returns the count.
fn check_all(cases: &[(Op, String)]) -> usize {
    for (op, expected) in cases {
        assert_eq!(round_text(Instr::new(*op)), *expected);
    }
    cases.len()
}

/// NAK's `test_float_ops` cases for SM 8.6, ported (`nvdisasm_tests.rs` @ 52238762): `FMNMX`,
/// `FSET` and `FSETP` with each of NAK's fourteen comparisons and three combine operations,
/// `.ftz` on and off, the source pairs `r2, r3` and `r2, 2`; `FSETP`'s second destination `PT`,
/// as NAK's encoder writes it.
#[test]
fn comparison_text_matches_nak_nvdisasm_cases() {
    let pred = |index| PredSrc {
        reg: p(index),
        negated: false,
    };
    let cmp_ops = [
        (FloatCmpOp::Eq, ".eq"),
        (FloatCmpOp::Ne, ".ne"),
        (FloatCmpOp::Lt, ".lt"),
        (FloatCmpOp::Le, ".le"),
        (FloatCmpOp::Gt, ".gt"),
        (FloatCmpOp::Ge, ".ge"),
        (FloatCmpOp::EqU, ".equ"),
        (FloatCmpOp::NeU, ".neu"),
        (FloatCmpOp::LtU, ".ltu"),
        (FloatCmpOp::LeU, ".leu"),
        (FloatCmpOp::GtU, ".gtu"),
        (FloatCmpOp::GeU, ".geu"),
        (FloatCmpOp::Num, ".num"),
        (FloatCmpOp::Nan, ".nan"),
    ];
    let set_ops = [
        (PredSetOp::And, ".and"),
        (PredSetOp::Or, ".or"),
        (PredSetOp::Xor, ".xor"),
    ];
    let mut cases = Vec::new();
    for (ftz, ftz_str) in NAK_FTZ {
        for (srcs, srcs_str, _) in nak_two_src() {
            cases.push((
                Op::FMnMx(OpFMnMx {
                    dst: Gpr::new(1),
                    srcs,
                    min: pred(4),
                    ftz,
                    nan: false,
                }),
                format!("fmnmx{ftz_str} r1, {srcs_str}, p4;"),
            ));
            for (cmp, cmp_str) in cmp_ops {
                cases.push((
                    Op::FSet(OpFSet {
                        dst: Gpr::new(1),
                        cmp,
                        srcs,
                        set_op: PredSetOp::And,
                        accum: PredSrc::TRUE,
                        ftz,
                    }),
                    format!("fset.bf{cmp_str}{ftz_str}.and r1, {srcs_str}, pt;"),
                ));
                for (set_op, set_str) in set_ops {
                    cases.push((
                        Op::FSetP(OpFSetP {
                            dst: p(2),
                            dst1: Pred::PT,
                            cmp,
                            set_op,
                            srcs,
                            accum: pred(1),
                            ftz,
                        }),
                        format!("fsetp{cmp_str}{ftz_str}{set_str} p2, pt, {srcs_str}, p1;"),
                    ));
                }
            }
        }
    }
    // FMNMX 4, FSET 56, FSETP 168.
    assert_eq!(check_all(&cases), 4 + 56 + 168);
}

/// NAK's `test_float_ops` conversion cases for SM 8.6, ported: `F2F` and `F2I` from `f64` and
/// `f16`, `FRND` in 64 and 32 bits, `I2F` to `f64` and `f16`, with `.rz` (`.trunc`) and the
/// default rounding and `.ftz` on and off; the 64-bit opcode exactly when a type is 64 bits, as
/// NAK's encoder picks it.
#[test]
fn conversion_text_matches_nak_nvdisasm_cases() {
    let r = Gpr::new;
    let mut cases = Vec::new();
    for (ftz, ftz_str) in NAK_FTZ {
        for (rnd, rnd_str, int_str) in NAK_RND {
            for (src_type, src_str) in [(FloatType::F64, ".f64"), (FloatType::F16, ".f16")] {
                let wide = src_type == FloatType::F64;
                cases.push((
                    Op::F2F(OpF2F {
                        dst: r(1),
                        src: AluSrc::reg(r(2)),
                        dst_type: FloatType::F32,
                        src_type,
                        rnd,
                        ftz,
                        high: false,
                        wide,
                    }),
                    format!("f2f{ftz_str}.f32{src_str}{rnd_str} r1, r2;"),
                ));
                cases.push((
                    Op::F2I(OpF2I {
                        dst: r(1),
                        src: AluSrc::reg(r(2)),
                        dst_type: IntType::U32,
                        src_type,
                        rnd,
                        ftz,
                        ntz: false,
                        high: false,
                        wide,
                    }),
                    format!("f2i{ftz_str}.u32{src_str}{int_str} r1, r2;"),
                ));
            }
            for (ty, dst, type_str, dst_str, wide) in [
                (FloatType::F64, 2, ".f64", "r2", true),
                (FloatType::F32, 1, "", "r1", false),
            ] {
                cases.push((
                    Op::FRnd(OpFRnd {
                        dst: r(dst),
                        src: AluSrc::reg(r(2)),
                        dst_type: ty,
                        src_type: ty,
                        rnd,
                        ftz,
                        high: false,
                        wide,
                    }),
                    format!("frnd{ftz_str}{type_str}{int_str} {dst_str}, r2;"),
                ));
            }
            for (dst_type, dst_str) in [(FloatType::F64, ".f64"), (FloatType::F16, ".f16")] {
                cases.push((
                    Op::I2F(OpI2F {
                        dst: r(1),
                        src: AluSrc::reg(r(2)),
                        dst_type,
                        src_type: IntType::U32,
                        rnd,
                        select: 0,
                        wide: dst_type == FloatType::F64,
                    }),
                    format!("i2f{dst_str}.u32{rnd_str} r1, r2;"),
                ));
            }
        }
    }
    // F2F, F2I, FRND and I2F 8 each.
    assert_eq!(check_all(&cases), 4 * 8);
}

/// NAK's `test_float_ops` `FADD` and `FMUL` cases for SM 8.6, ported (`FFMA`'s are
/// `ffma_text_matches_nak_nvdisasm_cases`): `.ftz`, `.rz` and `.sat` on and off with the source
/// pairs `r2, r3` and `r2, 2`; the unread slot as NAK's encoder fills it (`FADD`'s R0 in form 1
/// and `RZ` in form 2, `FMUL`'s `RZ`) and `FMUL`'s scale 4.
#[test]
fn fadd_and_fmul_text_matches_nak_nvdisasm_cases() {
    let mut cases = Vec::new();
    for (ftz, ftz_str) in NAK_FTZ {
        for (rnd, rnd_str, _) in NAK_RND {
            for (saturate, sat_str) in [(false, ""), (true, ".sat")] {
                for (srcs, srcs_str, unread) in nak_two_src() {
                    cases.push((
                        Op::FAdd(OpFAdd {
                            dst: Gpr::new(1),
                            srcs,
                            ignored: unread,
                            saturate,
                            rnd,
                            ftz,
                        }),
                        format!("fadd{ftz_str}{rnd_str}{sat_str} r1, {srcs_str};"),
                    ));
                    cases.push((
                        Op::FMul(OpFMul {
                            dst: Gpr::new(1),
                            srcs,
                            ignored: Gpr::RZ,
                            saturate,
                            rnd,
                            ftz,
                            dnz: false,
                            scale: FMulScale::None,
                        }),
                        format!("fmul{ftz_str}{rnd_str}{sat_str} r1, {srcs_str};"),
                    ));
                }
            }
        }
    }
    // FADD and FMUL 16 each.
    assert_eq!(check_all(&cases), 2 * 16);
}

/// NAK's `test_mufu` for SM 8.6, ported: every function in 32 and 16 bits (but `RCP64H` and
/// `RSQ64H`, which NAK does not test in 16 bits), `.ex2` and `.lg2` for the exponential and
/// logarithm.
#[test]
fn mufu_text_matches_nak_nvdisasm_cases() {
    let names = [
        (MuFuOp::Cos, ".cos"),
        (MuFuOp::Sin, ".sin"),
        (MuFuOp::Exp2, ".ex2"),
        (MuFuOp::Log2, ".lg2"),
        (MuFuOp::Rcp, ".rcp"),
        (MuFuOp::Rsq, ".rsq"),
        (MuFuOp::Rcp64H, ".rcp64h"),
        (MuFuOp::Rsq64H, ".rsq64h"),
        (MuFuOp::Sqrt, ".sqrt"),
        (MuFuOp::Tanh, ".tanh"),
    ];
    let mut checked = 0;
    for (op, op_str) in names {
        for (f16, type_str) in [(false, ""), (true, ".f16")] {
            if f16 && matches!(op, MuFuOp::Rcp64H | MuFuOp::Rsq64H) {
                continue;
            }
            let mufu = Op::MuFu(OpMuFu {
                dst: Gpr::new(2),
                op,
                src: AluSrc::reg(Gpr::new(3)),
                f16,
                high: false,
            });
            assert_eq!(
                round_text(Instr::new(mufu)),
                format!("mufu{op_str}{type_str} r2, r3;")
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 18);
}

/// NAK's `test_f2fp` for SM 8.6, ported (its vector cases; the uniform `UF2FP` is slice 6d's):
/// `.pack_ab` with and without `.rz`, NAK's zero register as the third source.
#[test]
fn f2fp_text_matches_nak_nvdisasm_cases() {
    for (rnd, rnd_str) in [(RoundMode::NearestEven, ""), (RoundMode::Zero, ".rz")] {
        let f2fp = Op::F2FP(OpF2FP {
            dst: Gpr::new(1),
            srcs: [AluSrc::reg(Gpr::new(2)), AluSrc::reg(Gpr::new(3))],
            c: Gpr::RZ,
            merge_c: false,
            h1: false,
            rnd,
        });
        assert_eq!(
            round_text(Instr::new(f2fp)),
            format!("f2fp.pack_ab{rnd_str} r1, r2, r3;")
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Hand-built words.

/// One hand-built word per form of the group, decoded and printed; every form of the group is
/// here.
#[test]
fn hand_built_words_of_each_form_decode() {
    let cases = [
        arithmetic_words(),
        comparison_words(),
        fsetp_and_fsel_words(),
        conversion_words(),
        f2i_words(),
        int_conversion_words(),
    ]
    .concat();
    let mut opcodes: Vec<u16> = Vec::new();
    for (word, expected) in &cases {
        assert_eq!(text(*word), *expected, "{:#x}", word.bits());
        opcodes.push(opcode_of(*word));
    }
    for form in float::FORMS {
        assert!(opcodes.contains(&form.opcode), "{:#x}", form.opcode);
    }
}

/// Hand-built words of `FFMA`, `FADD`, `FMUL`, `MUFU` and `FSWZADD`. Fields: destination at 16,
/// source 0 at 24, the wide slot at 32 (register, immediate, or a constant buffer's offset at 38
/// and index at 54, or a uniform register with bit 91), the register slot at 64.
fn arithmetic_words() -> Vec<(Instruction, &'static str)> {
    vec![
        // FFMA in its seven forms.
        (
            word(0x223, &[(16, 1), (24, 2), (32, 3), (64, 4)]),
            "ffma r1, r2, r3, r4;",
        ),
        (
            word(0x423, &[(16, 1), (24, 2), (32, TWO), (64, 4)]),
            "ffma r1, r2, r4, 2;",
        ),
        (
            word(0x623, &[(16, 1), (24, 2), (38, 8), (54, 1), (64, 4)]),
            "ffma r1, r2, r4, c[0x1][0x8];",
        ),
        (
            word(0x823, &[(16, 1), (24, 2), (32, TWO), (64, 4)]),
            "ffma r1, r2, 2, r4;",
        ),
        (
            word(0xa23, &[(16, 1), (24, 2), (38, 8), (54, 1), (64, 4)]),
            "ffma r1, r2, c[0x1][0x8], r4;",
        ),
        (
            word(0xc23, &[(16, 1), (24, 2), (32, 3), (91, 1), (64, 4)]),
            "ffma r1, r2, ur3, r4;",
        ),
        (
            word(0xe23, &[(16, 1), (24, 2), (32, 3), (91, 1), (64, 4)]),
            "ffma r1, r2, r4, ur3;",
        ),
        // FADD 0x221: b in the wide slot as source 1, the unread slot R0 (NAK's form 1).
        (
            word(0x221, &[(16, 1), (24, 2), (32, 3)]),
            "fadd r1, r2, r3;",
        ),
        // FADD 0x421, 0x621, 0xe21: b in the wide slot as source 2, the unread slot RZ.
        (
            word(0x421, &[(16, 1), (24, 2), (32, TWO), (64, 0xff)]),
            "fadd r1, r2, 2;",
        ),
        (
            word(0x621, &[(16, 1), (24, 2), (38, 0x10), (54, 3), (64, 0xff)]),
            "fadd r1, r2, c[0x3][0x10];",
        ),
        (
            word(0xe21, &[(16, 1), (24, 2), (32, 5), (91, 1), (64, 0xff)]),
            "fadd r1, r2, ur5;",
        ),
        // FMUL 0x220, 0x820, 0xa20 (bindless: ur6 at 32, bit 91), 0xc20; RZ at 64, scale 4.
        (
            word(0x220, &[(16, 1), (24, 2), (32, 3), (64, 0xff), (84, 4)]),
            "fmul r1, r2, r3;",
        ),
        (
            word(0x820, &[(16, 1), (24, 2), (32, ONE), (64, 0xff), (84, 4)]),
            "fmul r1, r2, 1;",
        ),
        (
            word(
                0xa20,
                &[
                    (16, 1),
                    (24, 2),
                    (32, 6),
                    (38, 0x20),
                    (91, 1),
                    (64, 0xff),
                    (84, 4),
                ],
            ),
            "fmul r1, r2, cx[ur6][0x20];",
        ),
        (
            word(
                0xc20,
                &[(16, 1), (24, 2), (32, 7), (91, 1), (64, 0xff), (84, 4)],
            ),
            "fmul r1, r2, ur7;",
        ),
        // MUFU 0x308: RCP (4 at 74) of r3; 0x908: RSQ (5) of 4.0; 0xb08: EX2 (2) of a constant;
        // 0xd08: SQRT (8) of ur9.
        (
            word(0x308, &[(16, 2), (32, 3), (74, 4)]),
            "mufu.rcp r2, r3;",
        ),
        (
            word(0x908, &[(16, 2), (32, 0x4080_0000), (74, 5)]),
            "mufu.rsq r2, 4;",
        ),
        (
            word(0xb08, &[(16, 2), (38, 0x40), (54, 2), (74, 2)]),
            "mufu.ex2 r2, c[0x2][0x40];",
        ),
        (
            word(0xd08, &[(16, 2), (32, 9), (91, 1), (74, 8)]),
            "mufu.sqrt r2, ur9;",
        ),
        // FSWZADD 0x822: the operation byte 0b00_01_10_11 at 32, lane 0 in its top two bits.
        (
            word(0x822, &[(16, 1), (24, 2), (32, 0x1b), (64, 3)]),
            "fswzadd r1, r2, r3, [add, sub, subr, mov2];",
        ),
    ]
}

/// Hand-built words of `FMNMX`, `FSET`, `FSETP` and `FSEL`: the predicate at 87 (bit 90 its
/// negation), `FSET`'s and `FSETP`'s comparison at 76 and combine operation at 74, `FSETP`'s
/// destinations at 81 and 84.
fn comparison_words() -> Vec<(Instruction, &'static str)> {
    vec![
        (
            word(0x209, &[(16, 1), (24, 2), (32, 3), (87, 4)]),
            "fmnmx r1, r2, r3, p4;",
        ),
        (
            word(0x809, &[(16, 1), (24, 2), (32, TWO), (87, 4), (90, 1)]),
            "fmnmx r1, r2, 2, !p4;",
        ),
        (
            word(0xa09, &[(16, 1), (24, 2), (38, 8), (54, 1), (87, 7)]),
            "fmnmx r1, r2, c[0x1][0x8], pt;",
        ),
        (
            word(0xc09, &[(16, 1), (24, 2), (32, 3), (91, 1), (87, 0)]),
            "fmnmx r1, r2, ur3, p0;",
        ),
        (
            word(0x20a, &[(16, 1), (24, 2), (32, 3), (76, 2), (87, 7)]),
            "fset.bf.eq.and r1, r2, r3, pt;",
        ),
        (
            word(
                0x80a,
                &[(16, 1), (24, 2), (32, TWO), (76, 1), (74, 1), (87, 3)],
            ),
            "fset.bf.lt.or r1, r2, 2, p3;",
        ),
        (
            word(
                0xa0a,
                &[
                    (16, 1),
                    (24, 2),
                    (38, 4),
                    (54, 5),
                    (76, 14),
                    (74, 2),
                    (87, 7),
                ],
            ),
            "fset.bf.geu.xor r1, r2, c[0x5][0x4], pt;",
        ),
        (
            word(
                0xc0a,
                &[(16, 1), (24, 2), (32, 4), (91, 1), (76, 8), (87, 7)],
            ),
            "fset.bf.nan.and r1, r2, ur4, pt;",
        ),
    ]
}

/// Hand-built words of `FSETP` and `FSEL` (fields as for `comparison_words`).
fn fsetp_and_fsel_words() -> Vec<(Instruction, &'static str)> {
    vec![
        (
            word(
                0x20b,
                &[
                    (81, 2),
                    (84, 7),
                    (24, 2),
                    (32, 3),
                    (76, 4),
                    (74, 1),
                    (87, 1),
                ],
            ),
            "fsetp.gt.or p2, pt, r2, r3, p1;",
        ),
        (
            word(
                0x80b,
                &[(81, 0), (84, 1), (24, 2), (32, ONE), (76, 6), (87, 7)],
            ),
            "fsetp.ge.and p0, p1, r2, 1, pt;",
        ),
        (
            word(
                0xa0b,
                &[
                    (81, 5),
                    (84, 7),
                    (24, 2),
                    (38, 8),
                    (54, 1),
                    (76, 15),
                    (87, 7),
                ],
            ),
            "fsetp.t.and p5, pt, r2, c[0x1][0x8], pt;",
        ),
        (
            word(
                0xc0b,
                &[
                    (81, 3),
                    (84, 7),
                    (24, 2),
                    (32, 1),
                    (91, 1),
                    (76, 0),
                    (74, 2),
                    (87, 2),
                    (90, 1),
                ],
            ),
            "fsetp.f.xor p3, pt, r2, ur1, !p2;",
        ),
        (
            word(0x208, &[(16, 1), (24, 2), (32, 3), (87, 0)]),
            "fsel r1, r2, r3, p0;",
        ),
        (
            word(0x808, &[(16, 1), (24, 2), (32, TWO), (87, 6), (90, 1)]),
            "fsel r1, r2, 2, !p6;",
        ),
        (
            word(0xa08, &[(16, 1), (24, 2), (38, 0x10), (54, 0), (87, 1)]),
            "fsel r1, r2, c[0x0][0x10], p1;",
        ),
    ]
}

/// Hand-built words of the conversions: sizes at 75 (destination) and 84 (source), log2 of the
/// byte count; `F2I`'s signed bit at 72, `I2F`'s and `I2FP`'s at 74.
fn conversion_words() -> Vec<(Instruction, &'static str)> {
    vec![
        // FRND f32 in forms 1, 4, 5, 6; then f64 in the 64-bit opcode's.
        (
            word(0x307, &[(16, 1), (32, 2), (75, 2), (84, 2)]),
            "frnd r1, r2;",
        ),
        (
            word(
                0x907,
                &[(16, 1), (32, 0x3fc0_0000), (75, 2), (84, 2), (78, 1)],
            ),
            "frnd.floor r1, 1.5;",
        ),
        (
            word(
                0xb07,
                &[(16, 1), (38, 4), (54, 1), (75, 2), (84, 2), (78, 2)],
            ),
            "frnd.ceil r1, c[0x1][0x4];",
        ),
        (
            word(0xd07, &[(16, 1), (32, 2), (91, 1), (75, 2), (84, 2)]),
            "frnd r1, ur2;",
        ),
        (
            word(0x313, &[(16, 1), (32, 2), (75, 3), (84, 3)]),
            "frnd.f64 r1, r2;",
        ),
        (
            word(0x913, &[(16, 1), (32, TWO), (75, 3), (84, 3), (78, 3)]),
            "frnd.f64.trunc r1, 2;",
        ),
        (
            word(0xb13, &[(16, 1), (38, 8), (54, 2), (75, 3), (84, 3)]),
            "frnd.f64 r1, c[0x2][0x8];",
        ),
        (
            word(0xd13, &[(16, 1), (32, 2), (91, 1), (75, 3), (84, 3)]),
            "frnd.f64 r1, ur2;",
        ),
        // F2F f32 to f16 in the narrow opcode, f64 to f32 in the 64-bit one.
        (
            word(0x304, &[(16, 1), (32, 2), (75, 1), (84, 2)]),
            "f2f.f16.f32 r1, r2;",
        ),
        (
            word(0x904, &[(16, 1), (32, ONE), (75, 1), (84, 2)]),
            "f2f.f16.f32 r1, 1;",
        ),
        (
            word(
                0xb04,
                &[(16, 1), (38, 4), (54, 1), (75, 1), (84, 2), (78, 3)],
            ),
            "f2f.f16.f32.rz r1, c[0x1][0x4];",
        ),
        (
            word(0xd04, &[(16, 1), (32, 2), (91, 1), (75, 2), (84, 1)]),
            "f2f.f32.f16 r1, ur2;",
        ),
        (
            word(0x310, &[(16, 1), (32, 2), (75, 2), (84, 3)]),
            "f2f.f32.f64 r1, r2;",
        ),
        (
            word(0x910, &[(16, 1), (32, ONE), (75, 3), (84, 2)]),
            "f2f.f64.f32 r1, 1;",
        ),
        (
            word(
                0xb10,
                &[(16, 1), (38, 4), (54, 1), (75, 2), (84, 3), (78, 1)],
            ),
            "f2f.f32.f64.rm r1, c[0x1][0x4];",
        ),
        (
            word(0xd10, &[(16, 1), (32, 2), (91, 1), (75, 2), (84, 3)]),
            "f2f.f32.f64 r1, ur2;",
        ),
        // F2I f32 to s32 (the default types), u16, s64 and from f64.
    ]
}

/// Hand-built words of `F2I` (fields as for `conversion_words`).
fn f2i_words() -> Vec<(Instruction, &'static str)> {
    vec![
        (
            word(0x305, &[(16, 1), (32, 2), (72, 1), (75, 2), (84, 2)]),
            "f2i r1, r2;",
        ),
        (
            word(0x905, &[(16, 1), (32, TWO), (75, 1), (84, 2), (78, 3)]),
            "f2i.u16.trunc r1, 2;",
        ),
        (
            word(
                0xb05,
                &[(16, 1), (38, 4), (54, 1), (72, 1), (75, 2), (84, 2)],
            ),
            "f2i r1, c[0x1][0x4];",
        ),
        (
            word(
                0xd05,
                &[(16, 1), (32, 2), (91, 1), (72, 1), (75, 0), (84, 1)],
            ),
            "f2i.s8.f16 r1, ur2;",
        ),
        (
            word(0x311, &[(16, 1), (32, 2), (72, 1), (75, 2), (84, 3)]),
            "f2i.f64 r1, r2;",
        ),
        (
            word(0x911, &[(16, 1), (32, TWO), (72, 1), (75, 3), (84, 2)]),
            "f2i.s64 r1, 2;",
        ),
        (
            word(0xb11, &[(16, 1), (38, 4), (54, 1), (75, 3), (84, 3)]),
            "f2i.u64.f64 r1, c[0x1][0x4];",
        ),
        (
            word(
                0xd11,
                &[
                    (16, 1),
                    (32, 2),
                    (91, 1),
                    (72, 1),
                    (75, 2),
                    (84, 3),
                    (78, 1),
                ],
            ),
            "f2i.f64.floor r1, ur2;",
        ),
        // I2F s32 to f32 (the default types), from u32, to f16, and into f64.
    ]
}

/// Hand-built words of `I2F`, `I2FP` and `F2FP` (fields as for `conversion_words`).
fn int_conversion_words() -> Vec<(Instruction, &'static str)> {
    vec![
        (
            word(0x306, &[(16, 1), (32, 2), (74, 1), (75, 2), (84, 2)]),
            "i2f r1, r2;",
        ),
        (
            word(0x906, &[(16, 1), (32, 0x10), (75, 2), (84, 2), (78, 2)]),
            "i2f.u32.rp r1, 0x10;",
        ),
        (
            word(
                0xb06,
                &[(16, 1), (38, 4), (54, 1), (74, 1), (75, 1), (84, 2)],
            ),
            "i2f.f16 r1, c[0x1][0x4];",
        ),
        (
            word(
                0xd06,
                &[(16, 1), (32, 2), (91, 1), (74, 1), (75, 2), (84, 2)],
            ),
            "i2f r1, ur2;",
        ),
        (
            word(0x312, &[(16, 1), (32, 2), (74, 1), (75, 3), (84, 2)]),
            "i2f.f64 r1, r2;",
        ),
        (
            word(0x912, &[(16, 1), (32, 0x10), (74, 1), (75, 2), (84, 3)]),
            "i2f.s64 r1, 0x10;",
        ),
        (
            word(0xb12, &[(16, 1), (38, 4), (54, 1), (75, 3), (84, 3)]),
            "i2f.f64.u64 r1, c[0x1][0x4];",
        ),
        (
            word(
                0xd12,
                &[(16, 1), (32, 2), (91, 1), (74, 1), (75, 3), (84, 2)],
            ),
            "i2f.f64 r1, ur2;",
        ),
        // I2FP s32 to f32 and u32 to f32.
        (
            word(0x245, &[(16, 1), (32, 2), (74, 1), (75, 2), (84, 2)]),
            "i2fp.f32.s32 r1, r2;",
        ),
        (
            word(0x845, &[(16, 1), (32, 0x10), (75, 2), (84, 2)]),
            "i2fp.f32.u32 r1, 0x10;",
        ),
        (
            word(
                0xa45,
                &[(16, 1), (38, 4), (54, 1), (75, 2), (84, 2), (78, 3)],
            ),
            "i2fp.f32.u32.rz r1, c[0x1][0x4];",
        ),
        (
            word(
                0xc45,
                &[(16, 1), (32, 2), (91, 1), (74, 1), (75, 2), (84, 2)],
            ),
            "i2fp.f32.s32 r1, ur2;",
        ),
        // F2FP with NAK's RZ as the third source.
        (
            word(0x23e, &[(16, 1), (24, 2), (32, 3), (64, 0xff)]),
            "f2fp.pack_ab r1, r2, r3;",
        ),
        (
            word(0x83e, &[(16, 1), (24, 2), (32, ONE), (64, 0xff)]),
            "f2fp.pack_ab r1, r2, 1;",
        ),
        (
            word(
                0xa3e,
                &[(16, 1), (24, 2), (38, 4), (54, 1), (64, 0xff), (79, 3)],
            ),
            "f2fp.pack_ab.rz r1, r2, c[0x1][0x4];",
        ),
        (
            word(0xc3e, &[(16, 1), (24, 2), (32, 3), (91, 1), (64, 0xff)]),
            "f2fp.pack_ab r1, r2, ur3;",
        ),
    ]
}

/// Decodes `opcode` with the fields of `base` and each case's extra fields, checking the text.
fn check_bits(opcode: u128, base: &[(u32, u128)], cases: &[(&[(u32, u128)], &str)]) {
    for (extra, expected) in cases {
        let fields: Vec<(u32, u128)> = base.iter().chain(extra.iter()).copied().collect();
        let word = word(opcode, &fields);
        assert_eq!(text(word), *expected, "{:#x}", word.bits());
    }
}

/// Each modifier, flag and mode bit of `FADD`, `FMUL`, `MUFU` and `FSWZADD` set alone in a word
/// the assembler did not make, against the same word without it.
#[test]
fn hand_built_words_pin_each_arithmetic_modifier_bit() {
    let fadd = [(16, 1), (24, 2), (32, 3)];
    check_bits(
        0x221,
        &fadd,
        &[
            (&[], "fadd r1, r2, r3;"),
            (&[(72, 1)], "fadd r1, -r2, r3;"),
            (&[(73, 1)], "fadd r1, |r2|, r3;"),
            (&[(62, 1)], "fadd r1, r2, |r3|;"),
            (&[(63, 1)], "fadd r1, r2, -r3;"),
            (&[(77, 1)], "fadd.sat r1, r2, r3;"),
            (&[(80, 1)], "fadd.ftz r1, r2, r3;"),
            (&[(64, 7)], "fadd r1, r2, r3;"),
        ],
    );
    let fmul = [(16, 1), (24, 2), (32, 3), (64, 0xff), (84, 4)];
    check_bits(
        0x220,
        &fmul,
        &[
            (&[], "fmul r1, r2, r3;"),
            (&[(76, 1)], "fmul.fmz r1, r2, r3;"),
            (&[(77, 1)], "fmul.sat r1, r2, r3;"),
            (&[(80, 1)], "fmul.ftz r1, r2, r3;"),
            (&[(72, 1)], "fmul r1, -r2, r3;"),
            (&[(73, 1)], "fmul r1, |r2|, r3;"),
            (&[(62, 1)], "fmul r1, r2, |r3|;"),
            (&[(63, 1)], "fmul r1, r2, -r3;"),
        ],
    );
    check_bits(
        0x308,
        &[(16, 1), (32, 2)],
        &[
            (&[], "mufu.cos r1, r2;"),
            (&[(73, 1)], "mufu.cos.f16 r1, r2;"),
            (&[(73, 1), (60, 1)], "mufu.cos.f16 r1, r2.h1;"),
            (&[(62, 1)], "mufu.cos r1, |r2|;"),
            (&[(63, 1)], "mufu.cos r1, -r2;"),
        ],
    );
    check_bits(
        0x822,
        &[(16, 1), (24, 2), (64, 3)],
        &[
            (&[], "fswzadd r1, r2, r3, [add, add, add, add];"),
            (&[(77, 1)], "fswzadd.ndv r1, r2, r3, [add, add, add, add];"),
            (&[(80, 1)], "fswzadd.ftz r1, r2, r3, [add, add, add, add];"),
            (&[(38, 1)], "fswzadd r1, r2, r3, [sub, add, add, add];"),
            (&[(32, 2)], "fswzadd r1, r2, r3, [add, add, add, subr];"),
        ],
    );
}

/// Each modifier, flag and predicate bit of `FMNMX`, `FSET`, `FSETP` and `FSEL` set alone.
#[test]
fn hand_built_words_pin_each_comparison_modifier_bit() {
    check_bits(
        0x209,
        &[(16, 1), (24, 2), (32, 3), (87, 7)],
        &[
            (&[], "fmnmx r1, r2, r3, pt;"),
            (&[(80, 1)], "fmnmx.ftz r1, r2, r3, pt;"),
            (&[(81, 1)], "fmnmx.nan r1, r2, r3, pt;"),
            (&[(90, 1)], "fmnmx r1, r2, r3, !pt;"),
            (&[(72, 1)], "fmnmx r1, -r2, r3, pt;"),
            (&[(62, 1)], "fmnmx r1, r2, |r3|, pt;"),
        ],
    );
    check_bits(
        0x20a,
        &[(16, 1), (24, 2), (32, 3), (76, 2), (87, 7)],
        &[
            (&[], "fset.bf.eq.and r1, r2, r3, pt;"),
            (&[(80, 1)], "fset.bf.eq.ftz.and r1, r2, r3, pt;"),
            (&[(74, 1)], "fset.bf.eq.or r1, r2, r3, pt;"),
            (&[(75, 1)], "fset.bf.eq.xor r1, r2, r3, pt;"),
            (&[(90, 1)], "fset.bf.eq.and r1, r2, r3, !pt;"),
            (&[(73, 1)], "fset.bf.eq.and r1, |r2|, r3, pt;"),
        ],
    );
    check_bits(
        0x20b,
        &[(81, 0), (84, 7), (24, 2), (32, 3), (76, 2), (87, 7)],
        &[
            (&[], "fsetp.eq.and p0, pt, r2, r3, pt;"),
            (&[(80, 1)], "fsetp.eq.ftz.and p0, pt, r2, r3, pt;"),
            (&[(74, 1)], "fsetp.eq.or p0, pt, r2, r3, pt;"),
            (&[(75, 1)], "fsetp.eq.xor p0, pt, r2, r3, pt;"),
            (&[(90, 1)], "fsetp.eq.and p0, pt, r2, r3, !pt;"),
            (&[(63, 1)], "fsetp.eq.and p0, pt, r2, -r3, pt;"),
        ],
    );
    check_bits(
        0x208,
        &[(16, 1), (24, 2), (32, 3), (87, 7)],
        &[
            (&[(72, 1)], "fsel r1, -r2, r3, pt;"),
            (&[(73, 1)], "fsel r1, |r2|, r3, pt;"),
            (&[(62, 1)], "fsel r1, r2, |r3|, pt;"),
            (&[(63, 1)], "fsel r1, r2, -r3, pt;"),
            (&[(90, 1)], "fsel r1, r2, r3, !pt;"),
        ],
    );
}

/// Each modifier and flag bit of the conversions set alone.
#[test]
fn hand_built_words_pin_each_conversion_modifier_bit() {
    let cvt = [(16, 1), (32, 2), (75, 2), (84, 2)];
    check_bits(
        0x307,
        &cvt,
        &[
            (&[(80, 1)], "frnd.ftz r1, r2;"),
            (&[(62, 1)], "frnd r1, |r2|;"),
        ],
    );
    check_bits(
        0x304,
        &cvt,
        &[
            (&[(80, 1)], "f2f.ftz.f32.f32 r1, r2;"),
            (&[(63, 1)], "f2f.f32.f32 r1, -r2;"),
        ],
    );
    check_bits(
        0x305,
        &cvt,
        &[
            (&[], "f2i.u32 r1, r2;"),
            (&[(72, 1)], "f2i r1, r2;"),
            (&[(77, 1)], "f2i.u32.ntz r1, r2;"),
            (&[(80, 1)], "f2i.ftz.u32 r1, r2;"),
            (&[(62, 1)], "f2i.u32 r1, |r2|;"),
        ],
    );
    check_bits(
        0x306,
        &cvt,
        &[(&[], "i2f.u32 r1, r2;"), (&[(74, 1)], "i2f r1, r2;")],
    );
    check_bits(0x245, &cvt, &[(&[(74, 1)], "i2fp.f32.s32 r1, r2;")]);
    check_bits(
        0x23e,
        &[(16, 1), (24, 2), (32, 3), (64, 4)],
        &[
            (&[], "f2fp.pack_ab r1, r2, r3;"),
            (&[(78, 1)], "f2fp.pack_ab.merge_c r1, r2, r3, r4;"),
            (
                &[(78, 1), (72, 1)],
                "f2fp.pack_ab.merge_c.h1 r1, r2, r3, r4;",
            ),
            (&[(80, 1)], "f2fp.pack_ab.rp r1, r2, r3;"),
        ],
    );
}

/// Every rounding mode of every instruction that has one, in hand-built words: bits 78..80
/// (79..81 for `F2FP`) set to 0 to 3, printed as a float rounding or, for `FRND` and `F2I`, an
/// integer rounding.
#[test]
fn hand_built_words_pin_every_rounding_mode() {
    let suffixes = [
        ("", ""),
        (".rm", ".floor"),
        (".rp", ".ceil"),
        (".rz", ".trunc"),
    ];
    for (mode, (fl, it)) in (0..).zip(suffixes) {
        let at = |opcode, fields: &[(u32, u128)], start| {
            let mut fields = fields.to_vec();
            fields.push((start, mode));
            text(word(opcode, &fields))
        };
        let ops = [(24, 2), (32, 3)];
        let cvt = [(16, 1), (32, 2), (75, 2), (84, 2)];
        let cases = [
            (
                at(0x221, &[(16, 1), (24, 2), (32, 3)], 78),
                format!("fadd{fl} r1, r2, r3;"),
            ),
            (
                at(0x220, &[(16, 1), ops[0], ops[1], (64, 0xff), (84, 4)], 78),
                format!("fmul{fl} r1, r2, r3;"),
            ),
            (
                at(0x223, &[(16, 1), ops[0], ops[1], (64, 4)], 78),
                format!("ffma{fl} r1, r2, r3, r4;"),
            ),
            (
                at(0x822, &[(16, 1), (24, 2), (64, 3)], 78),
                format!("fswzadd{fl} r1, r2, r3, [add, add, add, add];"),
            ),
            (at(0x307, &cvt, 78), format!("frnd{it} r1, r2;")),
            (
                at(0x304, &[(16, 1), (32, 2), (75, 1), (84, 2)], 78),
                format!("f2f.f16.f32{fl} r1, r2;"),
            ),
            (
                at(0x305, &[cvt[0], cvt[1], cvt[2], cvt[3], (72, 1)], 78),
                format!("f2i{it} r1, r2;"),
            ),
            (
                at(0x306, &[cvt[0], cvt[1], cvt[2], cvt[3], (74, 1)], 78),
                format!("i2f{fl} r1, r2;"),
            ),
            (
                at(0x245, &[cvt[0], cvt[1], cvt[2], cvt[3], (74, 1)], 78),
                format!("i2fp.f32.s32{fl} r1, r2;"),
            ),
            (
                at(0x23e, &[(16, 1), ops[0], ops[1], (64, 0xff)], 79),
                format!("f2fp.pack_ab{fl} r1, r2, r3;"),
            ),
        ];
        for (actual, expected) in cases {
            assert_eq!(actual, expected, "mode {mode}");
        }
    }
}

/// Every `MUFU` function in a hand-built word, in 32 and 16 bits, and the first and last
/// undefined values refused with the exact error.
#[test]
fn mufu_functions_and_refusals() {
    let names = [
        "cos", "sin", "ex2", "lg2", "rcp", "rsq", "rcp64h", "rsq64h", "sqrt", "tanh",
    ];
    for (function, name) in (0..).zip(names) {
        assert_eq!(
            text(word(0x308, &[(16, 1), (32, 2), (74, function)])),
            format!("mufu.{name} r1, r2;")
        );
        assert_eq!(
            text(word(0x308, &[(16, 1), (32, 2), (74, function), (73, 1)])),
            format!("mufu.{name}.f16 r1, r2;")
        );
    }
    for value in [10, 63] {
        assert_eq!(
            decode(word(0x308, &[(16, 1), (32, 2), (74, value)])),
            Err(DecodeError::UnknownValue {
                opcode: 0x308,
                bit: 74,
                width: 6,
                value: u64::try_from(value).expect("small"),
            })
        );
    }
    assert_eq!(
        text(word(0x908, &[(74, 10)])),
        "error: opcode 0x908 has the value 10 in its 6-bit field at bit 74, which its form does \
         not define"
    );
}

/// `FSET`'s and `FSETP`'s combine operation 3 is refused; 2 (`XOR`) is the last accepted. In
/// `FSET`'s form 6 the field is not in the row, so bit 74 is an unknown bit there.
#[test]
fn combine_operation_3_is_refused() {
    for opcode in [0x20a_u16, 0x80a, 0xa0a, 0x20b, 0x80b, 0xa0b, 0xc0b] {
        let fixed = u128::from(opcode >> 9 == 6);
        let base = |set_op| word(u128::from(opcode), &[(74, set_op), (91, fixed), (87, 7)]);
        assert!(decode(base(2)).is_ok(), "{opcode:#x}");
        assert_eq!(
            decode(base(3)),
            Err(DecodeError::UnknownValue {
                opcode,
                bit: 74,
                width: 2,
                value: 3
            }),
            "{opcode:#x}"
        );
    }
    for bit in [74, 75] {
        assert_eq!(
            decode(word(0xc0a, &[(91, 1), (bit, 1)])),
            Err(DecodeError::UnknownBit { opcode: 0xc0a, bit })
        );
    }
}

/// Bits a form reads in some layouts only are unknown bits in the others (`decode`'s step 5),
/// and accepted in the layout that reads them.
#[test]
fn layout_bits_are_read_only_where_documented() {
    let unknown = |word, opcode, bit| {
        assert_eq!(
            decode(word),
            Err(DecodeError::UnknownBit { opcode, bit }),
            "{opcode:#x} bit {bit}"
        );
    };
    // MUFU's high half: only with .f16 (bit 73).
    unknown(word(0x308, &[(60, 1)]), 0x308, 60);
    assert!(decode(word(0x308, &[(60, 1), (73, 1)])).is_ok());
    // The narrow F2I's high half: only for an f16 source (size 1 at 84).
    for size in [2, 3] {
        unknown(word(0x305, &[(84, size), (60, 1)]), 0x305, 60);
    }
    assert_eq!(
        text(word(0x305, &[(84, 1), (75, 2), (60, 1)])),
        "f2i.u32.f16 r0, r0.h1;"
    );
    // I2F's byte select: two bits for an 8-bit source, bit 60 for a 16-bit one, none wider.
    assert_eq!(
        text(word(0x306, &[(84, 0), (75, 2), (60, 3)])),
        "i2f.u8 r0, r0.b3;"
    );
    assert_eq!(
        text(word(0x306, &[(84, 1), (75, 2), (60, 1)])),
        "i2f.u16 r0, r0.h1;"
    );
    unknown(word(0x306, &[(84, 1), (75, 2), (61, 1)]), 0x306, 61);
    for size in [2, 3] {
        unknown(word(0x306, &[(84, size), (75, 2), (60, 1)]), 0x306, 60);
        unknown(word(0x312, &[(84, size), (75, 2), (61, 1)]), 0x312, 61);
    }
    assert_eq!(
        text(word(0xd06, &[(91, 1), (84, 0), (75, 2), (60, 2)])),
        "i2f.u8 r0, ur0.b2;"
    );
    // F2FP's .H1: only with .MERGE_C (bit 78).
    unknown(word(0x23e, &[(72, 1)]), 0x23e, 72);
    // FMNMX's .NAN and FSET's combine operation are not in form 6's rows: step 2.
    unknown(word(0xc09, &[(91, 1), (81, 1)]), 0xc09, 81);
    // The high half of F2F, FRND and the wide F2I: only for an f16 source (size 1 at 84), in
    // every form that has bit 60 (`nak_high_halves_decode_in_every_form_with_bit_60`).
    for opcode in [0x304_u16, 0xb04, 0x310, 0x307, 0xb07, 0x313] {
        for size in [2, 3] {
            let fields = [(75, 2), (84, size), (60, 1)];
            unknown(word(u128::from(opcode), &fields), opcode, 60);
        }
    }
    for opcode in [0x311_u16, 0xb11] {
        unknown(word(u128::from(opcode), &[(84, 2), (60, 1)]), opcode, 60);
    }
    // MUFU's form 5: only with .f16.
    unknown(word(0xb08, &[(60, 1)]), 0xb08, 60);
}

/// NAK writes a 16-bit source's high-half select (bit 60) in every form and opcode but the
/// immediate one (`encode_alu_base` passes `src_types()[0]`, and `encode_alu_reg`,
/// `encode_alu_cb` and `encode_alu_ureg` each call `set_swizzle(60..62, ..)` for an `F16`
/// source), and the forms add it where PR #4588's rows leave it out. Each word here is one NAK
/// can emit (`opt_copy_prop` turns an `F16` source's high half into `SrcSwizzle::Yy`): it
/// decodes with bit 60 set, to the same op as with bit 60 clear but for `high`, prints `.h1`
/// on the operand, and re-assembles to itself. These are the divergence-table rows "a 16-bit
/// source's high half" in `op::float`'s module docs; bit 61, the swizzle's other bit, which
/// only an `F16v2` source sets, stays unknown.
#[test]
fn nak_high_halves_decode_in_every_form_with_bit_60() {
    // (opcode, fields besides bit 60): the F16 source is size 1 at bit 84; .f16 is bit 73.
    let cases: [(u16, &[(u32, u128)]); 17] = [
        // F2F.F32.F16 of a high half (f2f32 of unpackHalf2x16's .y), forms 1, 5 and 6.
        (0x304, &[(84, 1), (75, 2)]),
        (0xb04, &[(84, 1), (75, 2)]),
        (0xd04, &[(84, 1), (75, 2), (91, 1)]),
        // The wide F2F.F64.F16.
        (0x310, &[(84, 1), (75, 3)]),
        // FRND with an F16 source, both opcodes.
        (0x307, &[(84, 1), (75, 2)]),
        (0x307, &[(84, 1), (75, 1)]),
        (0xb07, &[(84, 1), (75, 1)]),
        (0xd07, &[(84, 1), (75, 1), (91, 1)]),
        (0x313, &[(84, 1), (75, 3)]),
        // MUFU.F16 in forms 5 and 6.
        (0xb08, &[(73, 1)]),
        (0xd08, &[(73, 1), (91, 1)]),
        // The narrow F2I.S32.F16 in forms 5 and 6.
        (0xb05, &[(84, 1), (75, 2), (72, 1)]),
        (0xd05, &[(84, 1), (75, 2), (72, 1), (91, 1)]),
        // The wide F2I.S64.F16 in forms 1, 5 and 6.
        (0x311, &[(84, 1), (75, 3), (72, 1)]),
        (0xb11, &[(84, 1), (75, 3), (72, 1)]),
        (0xd11, &[(84, 1), (75, 3), (72, 1), (91, 1)]),
        // The wide F2I.S32.F16 (a 64-bit opcode NAK does not emit for these types).
        (0x311, &[(84, 1), (75, 2), (72, 1)]),
    ];
    /// The op's high-half select, and the op with it cleared.
    fn split(op: Op) -> (bool, Op) {
        match op {
            Op::F2F(op) => (op.high, Op::F2F(OpF2F { high: false, ..op })),
            Op::FRnd(op) => (op.high, Op::FRnd(OpFRnd { high: false, ..op })),
            Op::MuFu(op) => (op.high, Op::MuFu(OpMuFu { high: false, ..op })),
            Op::F2I(op) => (op.high, Op::F2I(OpF2I { high: false, ..op })),
            other => unreachable!("{other:?}"),
        }
    }
    for (opcode, fields) in cases {
        let clear = word(u128::from(opcode), fields);
        let clear_instr = decode(clear).expect("decodes without bit 60");
        let mut with = fields.to_vec();
        with.push((60, 1));
        let high = word(u128::from(opcode), &with);
        let high_instr = decode(high).expect("decodes with bit 60");
        assert_eq!(split(clear_instr.op), (false, clear_instr.op), "{opcode:#x}");
        assert_eq!(split(high_instr.op), (true, clear_instr.op), "{opcode:#x}");
        let text = high_instr.to_string();
        assert!(text.ends_with(".h1;"), "{opcode:#x}: {text}");
        assert_eq!(assemble(&high_instr), Ok(high), "{opcode:#x}");
        with.push((61, 1));
        assert_eq!(
            decode(word(u128::from(opcode), &with)),
            Err(DecodeError::UnknownBit { opcode, bit: 61 }),
            "{opcode:#x}"
        );
    }
    // The forms whose rows have it already: MUFU.F16 and the narrow F2I in form 1.
    assert!(decode(word(0x308, &[(73, 1), (60, 1)])).is_ok());
    assert!(decode(word(0x305, &[(84, 1), (75, 2), (60, 1)])).is_ok());
    // Hand-built text, one per instruction and handle kind.
    assert_eq!(
        text(word(0x304, &[(16, 1), (32, 2), (75, 2), (84, 1), (60, 1)])),
        "f2f.f32.f16 r1, r2.h1;"
    );
    assert_eq!(
        text(word(0xd13, &[(16, 1), (32, 5), (75, 3), (84, 1), (91, 1), (60, 1)])),
        "frnd.f64.f16 r1, ur5.h1;"
    );
    assert_eq!(
        text(word(0xb08, &[(16, 1), (38, 0x10), (54, 2), (73, 1), (60, 1)])),
        "mufu.cos.f16 r1, c[0x2][0x10].h1;"
    );
}

/// The order of checks with the group's faults present at once: a stray bit (step 2) before a
/// missing fixed bit (step 3) and before an undefined value (step 4); a missing fixed bit before
/// an undefined value; an undefined value before an unread bit (step 5).
#[test]
fn float_checks_run_in_the_documented_order() {
    // Step 2 before step 4: MUFU function 10 and bit 100.
    assert_eq!(
        decode(word(0x308, &[(74, 10), (100, 1)])),
        Err(DecodeError::UnknownBit {
            opcode: 0x308,
            bit: 100
        })
    );
    // Step 2 before step 3: the uniform MUFU without bit 91 and with bit 100.
    assert_eq!(
        decode(word(0xd08, &[(100, 1)])),
        Err(DecodeError::UnknownBit {
            opcode: 0xd08,
            bit: 100
        })
    );
    // Step 3 before step 4: the uniform MUFU without bit 91 and with function 10.
    assert_eq!(
        decode(word(0xd08, &[(74, 10)])),
        Err(DecodeError::MissingBit {
            opcode: 0xd08,
            bit: 91
        })
    );
    // Step 4 before step 5: MUFU function 10 with a high half and no .f16.
    assert_eq!(
        decode(word(0x308, &[(74, 10), (60, 1)])),
        Err(DecodeError::UnknownValue {
            opcode: 0x308,
            bit: 74,
            width: 6,
            value: 10
        })
    );
    // Step 2 before step 4 in FSET: combine operation 3 and bit 100.
    assert_eq!(
        decode(word(0x20a, &[(74, 3), (100, 1)])),
        Err(DecodeError::UnknownBit {
            opcode: 0x20a,
            bit: 100
        })
    );
}

/// Every uniform-register form of the group requires bit 91: refused without it, naming bit 91,
/// and accepted with it.
#[test]
fn float_uniform_forms_require_bit_91() {
    let uniform: Vec<u16> = float::FORMS
        .iter()
        .map(|form| form.opcode)
        .filter(|opcode| matches!(opcode >> 9, 6 | 7))
        .collect();
    assert_eq!(
        uniform,
        [
            0xc23, 0xe23, 0xe21, 0xc20, 0xc09, 0xc0a, 0xc0b, 0xd08, 0xd07, 0xd13, 0xd04, 0xd10,
            0xd05, 0xd11, 0xd06, 0xd12, 0xc45, 0xc3e
        ]
    );
    for opcode in uniform {
        let bare = Instruction::default().with_field(OPCODE, u64::from(opcode));
        assert_eq!(
            decode(bare),
            Err(DecodeError::MissingBit { opcode, bit: 91 }),
            "{opcode:#x}"
        );
        let canonical = canonical_word(opcode);
        assert!(canonical.flag(SLOT_UNIFORM), "{opcode:#x}");
        assert!(decode(canonical).is_ok(), "{opcode:#x}");
        assert_eq!(
            decode(canonical.with_flag(SLOT_UNIFORM, false)),
            Err(DecodeError::MissingBit { opcode, bit: 91 }),
            "{opcode:#x}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Divergences.

/// Each of the group's rows drops exactly the bits the module docs list, and each dropped bit
/// set alone in an otherwise valid word is refused, naming it.
#[test]
fn float_rows_drop_exactly_the_listed_bits() {
    let slot = (1 << 62) | (1 << 63);
    let src2_mods = (1 << 74) | (1 << 75);
    let expected = |opcode: u16| -> u128 {
        let form = opcode >> 9;
        match opcode & 0x1ff {
            0x021 | 0x020 => src2_mods,
            0x009 => 0xf << 65,
            0x106 | 0x112 if form != 4 => slot,
            0x045 => match form {
                1 => slot | (1 << 91),
                4 => 1 << 91,
                _ => slot,
            },
            0x03e => {
                let extra = (1 << 73) | src2_mods;
                if form == 4 { extra } else { extra | slot }
            }
            _ => 0,
        }
    };
    for (form, row) in float::FORMS.iter().zip(float::REFERENCE_ROWS) {
        let opcode = form.opcode;
        assert_eq!(row.dropped, expected(opcode), "{opcode:#x}");
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
}

/// PR #4588's rows of these instructions that NAK never encodes are unknown opcodes (module
/// docs), and so are the uniform datapath's rows, which are slice 6d's.
#[test]
fn float_rows_left_undecoded_are_unknown_opcodes() {
    let unknown = [
        // FADD forms 4, 5, 6.
        0x821, 0xa21, 0xc21, // FMUL, FMNMX, FSET, FSETP forms 2, 3, 7.
        0x420, 0x620, 0xe20, 0x409, 0x609, 0xe09, 0x40a, 0x60a, 0xe0a, 0x40b, 0x60b, 0xe0b,
        // MUFU, FRND (both opcodes), F2F, F2I, I2F and F2FP forms 2, 3, 7.
        0x508, 0x708, 0xf08, 0x507, 0x707, 0xf07, 0x513, 0x713, 0xf13, 0x504, 0x704, 0xf04, 0x510,
        0x710, 0xf10, 0x505, 0x705, 0xf05, 0x511, 0x711, 0xf11, 0x506, 0x706, 0xf06, 0x512, 0x712,
        0xf12, 0x43e, 0x63e, 0xe3e, // The uniform datapath's rows (slice 6d).
        0x250, 0x252, 0x253, 0x254, 0x256, 0x25a, 0x25b, 0x25c, 0x25d, 0x2ba,
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

/// `FMUL`'s scales 1 to 7 decode, each in a hand-built word, with PR #4588's exponent `code -
/// 4`; scale 0, which PR #4588 refuses (`Require(scale >= 1 && scale <= 7, "native FMUL
/// scale")`), is refused with the exact field and value, in every form.
#[test]
fn fmul_scales_decode_and_zero_is_refused() {
    let names = [".d8", ".d4", ".d2", "", ".m2", ".m4", ".m8"];
    for (code, name) in (1..).zip(names) {
        let word = word(0x220, &[(16, 1), (24, 2), (32, 3), (64, 0xff), (84, code)]);
        assert_eq!(text(word), format!("fmul{name} r1, r2, r3;"));
        let scale = FMulScale::from_bits(u64::try_from(code).expect("small")).expect("a scale");
        assert_eq!(
            i64::from(scale.exponent()),
            i64::try_from(code).expect("small") - 4
        );
    }
    let zero = word(0x220, &[(16, 1), (24, 2), (32, 3), (64, 0xff)]);
    assert_eq!(
        text(zero),
        "error: opcode 0x220 has the value 0 in its 3-bit field at bit 84, which its form does \
         not define"
    );
    for (opcode, fixed) in [(0x220, 0), (0x820, 0), (0xa20, 0), (0xc20, 1 << 91)] {
        assert_eq!(
            decode(word(opcode, &[(91, fixed >> 91)])),
            Err(DecodeError::UnknownValue {
                opcode: u16::try_from(opcode).expect("12 bits"),
                bit: 84,
                width: 3,
                value: 0
            }),
            "{opcode:#x}"
        );
    }
}

/// Every pair of float sizes 1 to 3 decodes in both `F2F` opcodes, each keeping its opcode, and
/// prints the same text in both (the opcode is not printed); `FRND` likewise prints each pair,
/// its equal pairs other than F32 with one type. Size 0 is refused in either field
/// (`float_sizes_of_zero_are_refused`).
#[test]
fn conversion_sizes_decode_every_type() {
    let names = [".f16", ".f32", ".f64"];
    for (dst, dst_name) in (1..).zip(names) {
        for (src, src_name) in (1..).zip(names) {
            let narrow = word(0x304, &[(16, 1), (32, 2), (75, dst), (84, src)]);
            let wide = word(0x310, &[(16, 1), (32, 2), (75, dst), (84, src)]);
            let expected = format!("f2f{dst_name}{src_name} r1, r2;");
            assert_eq!(text(narrow), expected);
            assert_eq!(text(wide), expected);
            for (word, wide) in [(narrow, false), (wide, true)] {
                let Ok(Instr {
                    op: Op::F2F(op), ..
                }) = decode(word)
                else {
                    unreachable!("an F2F word");
                };
                assert_eq!(op.wide, wide);
                assert_eq!(assemble(&Instr::new(Op::F2F(op))), Ok(word));
            }
        }
    }
    // FRND prints both types when they differ, one when they are equal and not 32-bit, and
    // none for F32 to F32 (NAK's `test_float_ops` pins `.f64` and the bare 32-bit form).
    assert_eq!(
        text(word(0x307, &[(16, 1), (32, 2), (75, 1), (84, 1)])),
        "frnd.f16 r1, r2;"
    );
    for (dst, dst_name) in (1..).zip(names) {
        for (src, src_name) in (1..).zip(names) {
            let expected = match (dst == src, dst_name) {
                (true, ".f32") => "frnd r1, r2;".to_owned(),
                (true, name) => format!("frnd{name} r1, r2;"),
                (false, _) => format!("frnd{dst_name}{src_name} r1, r2;"),
            };
            for opcode in [0x307, 0x313] {
                let word = word(opcode, &[(16, 1), (32, 2), (75, dst), (84, src)]);
                assert_eq!(text(word), expected, "{opcode:#x} {dst} {src}");
            }
        }
    }
    // F2I's integer sizes and sign.
    let ints = [".u8", ".u16", ".u32", ".u64"];
    for (size, name) in (0..).zip(ints) {
        assert_eq!(
            text(word(0x305, &[(75, size), (84, 2)])),
            format!("f2i{name} r0, r0;")
        );
        let signed = name.replace('u', "s");
        let expected = if signed == ".s32" {
            "f2i r0, r0;".to_owned()
        } else {
            format!("f2i{signed} r0, r0;")
        };
        assert_eq!(text(word(0x305, &[(75, size), (84, 2), (72, 1)])), expected);
    }
}

/// A float size of 0 names no type (NAK's `FloatType` has `F16`, `F32` and `F64`; PR #4588's
/// translator refuses it in every conversion: `Visitor::F2I` `Require(src_size == 2 ||
/// src_size == 1, ..)`, `I2F` `Require(dst_size == 2, ..)`, `FRND` `Require(Take(75, 2) == 2,
/// ..)` and `Require(Take(84, 2) == 2, ..)`, `F2F` `Require(Take(75, 2) == 1, ..)` and
/// `Require(Take(84, 2) == 2, ..)`): refused in each float size field of each conversion, both
/// opcodes, with the exact field; one below (none) and one above (size 1, `F16`) decode. With
/// both of `F2F`'s and `FRND`'s fields 0, the lower field (75) is reported. The integer size
/// fields (`F2I`'s destination, `I2F`'s source) decode 0, an 8-bit integer.
#[test]
fn float_sizes_of_zero_are_refused() {
    let refused = |opcode: u16, bit: u32| DecodeError::UnknownValue {
        opcode,
        bit,
        width: 2,
        value: 0,
    };
    // (opcode, its float size fields in ascending order).
    let cases: [(u16, &[u32]); 10] = [
        (0x304, &[75, 84]),
        (0x310, &[75, 84]),
        (0x307, &[75, 84]),
        (0x313, &[75, 84]),
        (0x305, &[84]),
        (0x311, &[84]),
        (0x306, &[75]),
        (0x312, &[75]),
        (0x245, &[75]),
        (0x845, &[75]),
    ];
    for (opcode, fields) in cases {
        let op = u128::from(opcode);
        // Both fields F32: decodes.
        let all_f32: Vec<(u32, u128)> = fields.iter().map(|&bit| (bit, 2)).collect();
        assert!(decode(word(op, &all_f32)).is_ok(), "{opcode:#x}");
        for &zero in fields {
            for (size, ok) in [(0, false), (1, true)] {
                let mut set: Vec<(u32, u128)> =
                    fields.iter().filter(|&&bit| bit != zero).map(|&bit| (bit, 2)).collect();
                set.push((zero, size));
                let result = decode(word(op, &set));
                if ok {
                    assert!(result.is_ok(), "{opcode:#x} {zero} {size}");
                } else {
                    assert_eq!(result, Err(refused(opcode, zero)), "{opcode:#x} {zero}");
                }
            }
        }
        // Every float size field 0 at once: the lowest is reported.
        assert_eq!(
            decode(word(op, &[])),
            Err(refused(opcode, fields[0])),
            "{opcode:#x}"
        );
    }
    assert_eq!(
        text(word(0x304, &[(75, 2)])),
        "error: opcode 0x304 has the value 0 in its 2-bit field at bit 84, which its form does \
         not define"
    );
    // The integer sizes: 0 is `.u8`.
    assert_eq!(text(word(0x305, &[(84, 2)])), "f2i.u8 r0, r0;");
    assert_eq!(text(word(0x306, &[(75, 2)])), "i2f.u8 r0, r0;");
}

/// Every form's bare word (its opcode and fixed bits alone) whose zero PR #4588 refuses is
/// refused with that exact field: `FMUL`'s scale (84, three bits) in its four forms, and the
/// lowest float size field of each conversion's eight (or `I2FP`'s four) forms, 75 where the
/// conversion has a float destination and 84 for `F2I`. These are the forms with a canonical
/// word in `op::float`'s rows (`framework::canonical_words_decode`), each of which decodes.
#[test]
fn bare_words_refuse_the_zeros_pr_4588_refuses() {
    let mut refused = 0;
    for (form, row) in float::FORMS.iter().zip(float::REFERENCE_ROWS) {
        let opcode = form.opcode;
        let bare = Instruction::from_bits(form.fixed).with_field(OPCODE, u64::from(opcode));
        let field = match opcode & 0x1ff {
            0x020 => Some((84, 3)),
            0x107 | 0x113 | 0x104 | 0x110 | 0x106 | 0x112 | 0x045 => Some((75, 2)),
            0x105 | 0x111 => Some((84, 2)),
            _ => None,
        };
        assert_eq!(row.canonical != 0, field.is_some(), "{opcode:#x}");
        let Some((bit, width)) = field else {
            assert!(decode(bare).is_ok(), "{opcode:#x}");
            continue;
        };
        assert_eq!(
            decode(bare),
            Err(DecodeError::UnknownValue {
                opcode,
                bit,
                width,
                value: 0
            }),
            "{opcode:#x}"
        );
        assert!(decode(canonical_word(opcode)).is_ok(), "{opcode:#x}");
        refused += 1;
    }
    assert_eq!(refused, 4 + 4 * 8 + 4);
}

/// The canonical words, written out from the references: `FMUL`'s scale 4 (NAK's
/// `set_field(84..87, 0x4)`), and `F32` (size 2, NAK's `(32 / 8).ilog2()`) in each float size
/// field.
#[test]
fn canonical_words_match_the_references() {
    let cases: [(u16, u128); 6] = [
        (0x220, 4 << 84),
        (0x307, (2 << 75) | (2 << 84)),
        (0x304, (2 << 75) | (2 << 84)),
        (0x305, 2 << 84),
        (0x306, 2 << 75),
        (0x245, 2 << 75),
    ];
    for (opcode, canonical) in cases {
        assert_eq!(
            canonical_word(opcode),
            Instruction::from_bits(canonical).with_field(OPCODE, u64::from(opcode)),
            "{opcode:#x}"
        );
    }
    assert_eq!(
        canonical_word(0xd04).bits(),
        (1 << 91) | (2 << 75) | (2 << 84) | 0xd04
    );
}

/// The register slot `FADD` and `FMUL` do not read is kept: two words that differ only there
/// decode to ops that differ only in `ignored`, print the same, and re-assemble to themselves.
#[test]
fn ignored_slot_is_kept_and_not_printed() {
    for (opcode, base) in [
        (0x221, vec![(16, 1), (24, 2), (32, 3)]),
        (0x421, vec![(16, 1), (24, 2), (32, TWO)]),
        (0x220, vec![(16, 1), (24, 2), (32, 3), (84, 4)]),
    ] {
        let mut previous = None;
        for slot in [0_u128, 0x80, 0xff] {
            let mut fields = base.clone();
            fields.push((64, slot));
            let word = word(opcode, &fields);
            let instr = decode(word).expect("decodes");
            let ignored = match instr.op {
                Op::FAdd(op) => op.ignored,
                Op::FMul(op) => op.ignored,
                _ => unreachable!("FADD or FMUL"),
            };
            assert_eq!(u128::from(ignored.index()), slot);
            assert_eq!(assemble(&instr), Ok(word));
            let printed = instr.to_string();
            if let Some(previous) = &previous {
                assert_eq!(&printed, previous);
            }
            previous = Some(printed);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The reference constants, written out.

/// The group's opcodes against the cited reference values: NAK's `sm70_encode.rs` @ 52238762 for
/// its encoders, PR #4588's `visitor.cpp` @ e5fb3b97 for `FSEL` and `I2FP`.
#[test]
fn float_opcodes_match_the_references() {
    let opcodes = [
        (FFMA_OPCODE, 0x023),
        (FADD_OPCODE, 0x021),
        (FMUL_OPCODE, 0x020),
        (FMNMX_OPCODE, 0x009),
        (FSET_OPCODE, 0x00a),
        (FSETP_OPCODE, 0x00b),
        (FSEL_OPCODE, 0x008),
        (MUFU_OPCODE, 0x108),
        (FSWZADD_OPCODE, 0x822),
        (FRND_OPCODE, 0x107),
        (FRND_WIDE_OPCODE, 0x113),
        (F2F_OPCODE, 0x104),
        (F2F_WIDE_OPCODE, 0x110),
        (F2I_OPCODE, 0x105),
        (F2I_WIDE_OPCODE, 0x111),
        (I2F_OPCODE, 0x106),
        (I2F_WIDE_OPCODE, 0x112),
        (I2FP_OPCODE, 0x045),
        (F2FP_OPCODE, 0x03e),
    ];
    for (shipping, reference) in opcodes {
        assert_eq!(shipping, reference);
    }
    let forms: Vec<u16> = FADD_FORMS
        .iter()
        .map(|form| form.opcode(FADD_OPCODE))
        .collect();
    assert_eq!(forms, [0x221, 0x421, 0x621, 0xe21]);
    let forms: Vec<u16> = FSEL_FORMS
        .iter()
        .map(|form| form.opcode(FSEL_OPCODE))
        .collect();
    assert_eq!(forms, [0x208, 0x808, 0xa08]);
}

/// The group's field positions against the cited reference ranges, written out (NAK's
/// `set_field`, `set_bit` and helper ranges; PR #4588's `Take` calls).
#[test]
fn float_fields_match_the_references() {
    let fields = [
        (FTZ, (80, 1)),
        (RND, (78, 2)),
        (SAT, (77, 1)),
        (DNZ, (76, 1)),
        (HIGH_HALF, (60, 1)),
        (CVT_DST_SIZE, (75, 2)),
        (CVT_SRC_SIZE, (84, 2)),
        (SET_OP, (74, 2)),
        (CMP_OP, (76, 4)),
        (FMUL_SCALE, (84, 3)),
        (FMNMX_NAN, (81, 1)),
        (FSETP_DST1, (84, 3)),
        (MUFU_F16, (73, 1)),
        (MUFU_OP, (74, 6)),
        (FSWZADD_SRC1, (64, 8)),
        (FSWZADD_NDV, (77, 1)),
        (F2I_SIGNED, (72, 1)),
        (F2I_NTZ, (77, 1)),
        (I2F_SIGNED, (74, 1)),
        (I2F_SELECT, (60, 2)),
        (F2FP_C, (64, 8)),
        (F2FP_H1, (72, 1)),
        (F2FP_MERGE_C, (78, 1)),
        (F2FP_RND, (79, 2)),
        // NAK: lane 0's operation in the top two bits of the byte at 32.
        (fswzadd_op(0), (38, 2)),
        (fswzadd_op(1), (36, 2)),
        (fswzadd_op(2), (34, 2)),
        (fswzadd_op(3), (32, 2)),
        // PR #4588: 32 >> (3 + size) parts, so 2, 1, 0 and 0 bits.
        (i2f_select(int(0, false)), (60, 2)),
        (i2f_select(int(1, true)), (60, 1)),
        (i2f_select(int(2, false)), (60, 0)),
        (i2f_select(int(3, true)), (60, 0)),
    ];
    for (shipping, (start, width)) in fields {
        assert_eq!((shipping.start, shipping.width), (start, width));
    }
    assert_eq!(FSWZADD_OP_BITS, 2);
    assert_eq!(OpFSwzAdd::LANES, 4);
}

/// The enumerations' encodings against NAK's `set_float_cmp_op`, `set_pred_set_op`, `OpMuFu`
/// and `OpFSwzAdd` encoders, Eden's `FPCompareOp` for 0 and 15, NAK's size encoding and PR
/// #4588's scale series, written out.
#[test]
fn float_enumerations_match_the_references() {
    let cmps = [
        (FloatCmpOp::False, 0x00),
        (FloatCmpOp::Lt, 0x01),
        (FloatCmpOp::Eq, 0x02),
        (FloatCmpOp::Le, 0x03),
        (FloatCmpOp::Gt, 0x04),
        (FloatCmpOp::Ne, 0x05),
        (FloatCmpOp::Ge, 0x06),
        (FloatCmpOp::Num, 0x07),
        (FloatCmpOp::Nan, 0x08),
        (FloatCmpOp::LtU, 0x09),
        (FloatCmpOp::EqU, 0x0a),
        (FloatCmpOp::LeU, 0x0b),
        (FloatCmpOp::GtU, 0x0c),
        (FloatCmpOp::NeU, 0x0d),
        (FloatCmpOp::GeU, 0x0e),
        (FloatCmpOp::True, 0x0f),
    ];
    for (op, bits) in cmps {
        assert_eq!(op.bits(), bits);
        assert_eq!(FloatCmpOp::from_bits(bits), op);
    }
    for (op, bits) in [(PredSetOp::And, 0), (PredSetOp::Or, 1), (PredSetOp::Xor, 2)] {
        assert_eq!(op.bits(), bits);
        assert_eq!(PredSetOp::from_bits(bits), Some(op));
    }
    assert_eq!(PredSetOp::from_bits(3), None);
    let mufus = [
        (MuFuOp::Cos, 0),
        (MuFuOp::Sin, 1),
        (MuFuOp::Exp2, 2),
        (MuFuOp::Log2, 3),
        (MuFuOp::Rcp, 4),
        (MuFuOp::Rsq, 5),
        (MuFuOp::Rcp64H, 6),
        (MuFuOp::Rsq64H, 7),
        (MuFuOp::Sqrt, 8),
        (MuFuOp::Tanh, 9),
    ];
    for (op, bits) in mufus {
        assert_eq!(op.bits(), bits);
        assert_eq!(MuFuOp::from_bits(bits), Some(op));
    }
    assert_eq!(MuFuOp::from_bits(10), None);
    let swz = [
        (FSwzAddOp::Add, 0),
        (FSwzAddOp::SubLeft, 1),
        (FSwzAddOp::SubRight, 2),
        (FSwzAddOp::MoveLeft, 3),
    ];
    for (op, bits) in swz {
        assert_eq!(op.bits(), bits);
        assert_eq!(FSwzAddOp::from_bits(bits), op);
    }
    // NAK's `FloatType`: F16, F32 and F64, sized `(bits / 8).ilog2()`; size 0 names none.
    let floats = [
        (FloatType::F16, 1, 16),
        (FloatType::F32, 2, 32),
        (FloatType::F64, 3, 64),
    ];
    assert_eq!(FloatType::ALL.len(), floats.len());
    for (ty, bits, width) in floats {
        assert_eq!((ty.bits(), ty.width()), (bits, width));
        assert_eq!(FloatType::from_bits(bits), Some(ty));
    }
    assert_eq!(FloatType::from_bits(0), None);
    assert_eq!(FloatType::from_bits(4), None);
    // PR #4588's `FMUL` scales 1 to 7 (`ScaleFactor`: `2^(code - 4)`); 0 names none.
    let scales = [
        (FMulScale::D8, 1, -3),
        (FMulScale::D4, 2, -2),
        (FMulScale::D2, 3, -1),
        (FMulScale::None, 4, 0),
        (FMulScale::M2, 5, 1),
        (FMulScale::M4, 6, 2),
        (FMulScale::M8, 7, 3),
    ];
    assert_eq!(FMulScale::ALL.len(), scales.len());
    for (scale, bits, exponent) in scales {
        assert_eq!((scale.bits(), scale.exponent()), (bits, exponent));
        assert_eq!(FMulScale::from_bits(bits), Some(scale));
    }
    assert_eq!(FMulScale::from_bits(0), None);
    assert_eq!(FMulScale::from_bits(8), None);
    for (size, width) in [(0, 8), (1, 16), (2, 32), (3, 64)] {
        for signed in [false, true] {
            let ty = IntType::new(size, signed).expect("a size of 0 to 3");
            assert_eq!(ty.size_bits(), u64::from(size));
            assert_eq!(ty.width(), width);
            assert_eq!(ty.is_signed(), signed);
        }
    }
    // The constructor refuses a size the two-bit field cannot hold, from one past the largest.
    assert_eq!(IntType::MAX_SIZE, 3);
    assert_eq!(
        u128::from(IntType::MAX_SIZE),
        CVT_SRC_SIZE.mask() >> CVT_SRC_SIZE.start
    );
    assert_eq!(
        u128::from(IntType::MAX_SIZE),
        CVT_DST_SIZE.mask() >> CVT_DST_SIZE.start
    );
    assert!(IntType::new(IntType::MAX_SIZE, false).is_some());
    for signed in [false, true] {
        assert_eq!(IntType::new(IntType::MAX_SIZE + 1, signed), None);
        assert_eq!(IntType::new(u8::MAX, signed), None);
    }
    assert_eq!(IntType::U32, int(2, false));
    assert_eq!(IntType::S32, int(2, true));
    assert_eq!(IntType::S32.to_string(), ".s32");
    assert_eq!(int(0, false).to_string(), ".u8");
    assert_eq!(FMulScale::None.bits(), 4);
    assert_eq!(FMulScale::D2.exponent(), -1);
    assert_eq!(FMulScale::M8.exponent(), 3);
}

// ---------------------------------------------------------------------------------------------
// The assembler's refusals and unconfirmed text.

/// What the forms have no bits for is refused with the exact error, and the neighbouring
/// encodable value is accepted.
#[test]
fn unencodable_comparisons_are_refused() {
    let r = Gpr::new;
    let ureg = AluSrc::UniformReg {
        reg: ur(1),
        abs: false,
        neg: false,
    };
    let fmnmx = |b, nan| {
        assemble(&Instr::new(Op::FMnMx(OpFMnMx {
            dst: r(1),
            srcs: [AluSrc::reg(r(2)), b],
            min: PredSrc::TRUE,
            ftz: false,
            nan,
        })))
    };
    assert_eq!(
        fmnmx(ureg, true),
        Err(EncodeError::NoBits {
            mnemonic: "fmnmx",
            what: ".nan with a uniform-register source"
        })
    );
    assert!(fmnmx(ureg, false).is_ok());
    assert!(fmnmx(AluSrc::reg(r(3)), true).is_ok());
    assert_eq!(
        fmnmx(ureg, true).map_err(|error| error.to_string()),
        Err("fmnmx has no bits for .nan with a uniform-register source".into())
    );
    let fset = |b, set_op| {
        assemble(&Instr::new(Op::FSet(OpFSet {
            dst: r(1),
            cmp: FloatCmpOp::Eq,
            srcs: [AluSrc::reg(r(2)), b],
            set_op,
            accum: PredSrc::TRUE,
            ftz: false,
        })))
    };
    assert_eq!(
        fset(ureg, PredSetOp::Or),
        Err(EncodeError::NoBits {
            mnemonic: "fset",
            what: "a combine operation with a uniform-register source"
        })
    );
    assert!(fset(ureg, PredSetOp::And).is_ok());
    let select = |b| {
        assemble(&Instr::new(Op::FSel(OpFSel {
            dst: r(1),
            srcs: [AluSrc::reg(r(2)), b],
            cond: PredSrc::TRUE,
        })))
    };
    assert_eq!(
        select(ureg),
        Err(EncodeError::NoBits {
            mnemonic: "fsel",
            what: "a uniform-register source"
        })
    );
    assert!(select(imm(1.0)).is_ok());
}

/// A high half outside a 16-bit register in the narrow form is refused exactly, and a clear one
/// leaves an immediate's bit 60 alone.
#[test]
fn unencodable_high_halves_are_refused() {
    let r = Gpr::new;
    let mufu = |src, f16, high| {
        assemble(&Instr::new(Op::MuFu(OpMuFu {
            dst: r(1),
            op: MuFuOp::Cos,
            src,
            f16,
            high,
        })))
    };
    let what = "the high half of a source other than a 16-bit register, constant buffer or \
                uniform register";
    let half_refusal = |mnemonic| Err(EncodeError::NoBits { mnemonic, what });
    let cbuf = AluSrc::CBuf {
        cbuf: CBufRef {
            buf: CBuf::Binding(1),
            offset: 8,
        },
        abs: false,
        neg: false,
    };
    let ureg = AluSrc::UniformReg {
        reg: ur(3),
        abs: false,
        neg: false,
    };
    assert_eq!(mufu(AluSrc::reg(r(2)), false, true), half_refusal("mufu"));
    assert_eq!(mufu(imm(1.0), true, true), half_refusal("mufu"));
    for src in [AluSrc::reg(r(2)), cbuf, ureg] {
        assert!(mufu(src, true, true).is_ok());
        assert_eq!(mufu(src, false, true), half_refusal("mufu"));
    }
    // A clear high half leaves an immediate's bit 60 alone.
    let imm_word = mufu(AluSrc::Imm(1 << 28), true, false).expect("encodable");
    assert!(imm_word.flag(Field::bit(60)));
    let f2i = |src, src_type, wide, high| {
        assemble(&Instr::new(Op::F2I(OpF2I {
            dst: Gpr::new(1),
            src,
            dst_type: IntType::S32,
            src_type,
            rnd: RoundMode::Zero,
            ftz: false,
            ntz: false,
            high,
            wide,
        })))
    };
    let f2f = |src, src_type, wide, high| {
        assemble(&Instr::new(Op::F2F(OpF2F {
            dst: Gpr::new(1),
            src,
            dst_type: FloatType::F32,
            src_type,
            rnd: RoundMode::Zero,
            ftz: false,
            high,
            wide,
        })))
    };
    let frnd = |src, src_type, wide, high| {
        assemble(&Instr::new(Op::FRnd(OpFRnd {
            dst: Gpr::new(1),
            src,
            dst_type: FloatType::F32,
            src_type,
            rnd: RoundMode::Zero,
            ftz: false,
            high,
            wide,
        })))
    };
    type Encoder = fn(AluSrc, FloatType, bool, bool) -> Result<Instruction, EncodeError>;
    let conversions: [(&str, Encoder); 3] = [("f2i", f2i), ("f2f", f2f), ("frnd", frnd)];
    for (mnemonic, encode) in conversions {
        for wide in [false, true] {
            for src in [AluSrc::reg(r(2)), cbuf, ureg] {
                assert!(encode(src, FloatType::F16, wide, true).is_ok(), "{mnemonic}");
                for src_type in [FloatType::F32, FloatType::F64] {
                    assert_eq!(
                        encode(src, src_type, wide, true),
                        half_refusal(mnemonic),
                        "{mnemonic}"
                    );
                }
            }
            assert_eq!(
                encode(imm(1.0), FloatType::F16, wide, true),
                half_refusal(mnemonic),
                "{mnemonic}"
            );
        }
    }
}

/// `I2F`'s byte select past its source's width or on an immediate, a modified integer source,
/// and `F2FP`'s `.h1` without `.merge_c` or an immediate first source are refused exactly.
#[test]
fn unencodable_conversions_are_refused() {
    let r = Gpr::new;
    let i2f = |src, src_type, select| {
        assemble(&Instr::new(Op::I2F(OpI2F {
            dst: r(1),
            src,
            dst_type: FloatType::F32,
            src_type,
            rnd: RoundMode::NearestEven,
            select,
            wide: false,
        })))
    };
    let byte_type = int(0, false);
    let half_type = int(1, false);
    assert!(i2f(AluSrc::reg(r(2)), byte_type, 3).is_ok());
    assert_eq!(
        i2f(AluSrc::reg(r(2)), byte_type, 4),
        Err(EncodeError::TooWide {
            what: "byte select",
            value: 4,
            bits: 2
        })
    );
    assert!(i2f(AluSrc::reg(r(2)), half_type, 1).is_ok());
    assert_eq!(
        i2f(AluSrc::reg(r(2)), half_type, 2),
        Err(EncodeError::TooWide {
            what: "byte select",
            value: 2,
            bits: 1
        })
    );
    assert_eq!(
        i2f(AluSrc::reg(r(2)), IntType::S32, 1),
        Err(EncodeError::TooWide {
            what: "byte select",
            value: 1,
            bits: 0
        })
    );
    assert_eq!(
        i2f(AluSrc::Imm(1), byte_type, 1),
        Err(EncodeError::NoBits {
            mnemonic: "i2f",
            what: "a byte select of an immediate"
        })
    );
    assert_eq!(
        i2f(
            AluSrc::Reg {
                reg: r(2),
                abs: false,
                neg: true
            },
            IntType::S32,
            0
        ),
        Err(EncodeError::Modifier {
            mnemonic: "i2f",
            index: 1
        })
    );
    let f2fp = |a, merge_c, h1| {
        assemble(&Instr::new(Op::F2FP(OpF2FP {
            dst: r(1),
            srcs: [a, AluSrc::reg(r(3))],
            c: r(4),
            merge_c,
            h1,
            rnd: RoundMode::NearestEven,
        })))
    };
    assert_eq!(
        f2fp(AluSrc::reg(r(2)), false, true),
        Err(EncodeError::NoBits {
            mnemonic: "f2fp",
            what: ".h1 without .merge_c"
        })
    );
    assert!(f2fp(AluSrc::reg(r(2)), true, true).is_ok());
    assert_eq!(
        f2fp(imm(1.0), false, false),
        Err(EncodeError::FirstSourceNotRegister { mnemonic: "f2fp" })
    );
}

/// The printer's guesses for the forms NAK's tests do not cover (module docs): **unconfirmed**
/// until the `nvdisasm` cross-check runs.
#[test]
fn unconfirmed_text_of_the_float_group() {
    let r = Gpr::new;
    let cases = [
        (
            Op::FMul(OpFMul {
                dst: r(0),
                srcs: [
                    AluSrc::Reg {
                        reg: r(1),
                        abs: true,
                        neg: false,
                    },
                    imm(0.5),
                ],
                ignored: Gpr::RZ,
                saturate: true,
                rnd: RoundMode::PosInf,
                ftz: true,
                dnz: true,
                scale: FMulScale::D2,
            }),
            "fmul.ftz.fmz.d2.rp.sat r0, |r1|, 0.5;",
        ),
        (
            Op::FMnMx(OpFMnMx {
                dst: r(0),
                srcs: [AluSrc::reg(r(1)), AluSrc::reg(r(2))],
                min: PredSrc {
                    reg: p(3),
                    negated: true,
                },
                ftz: true,
                nan: true,
            }),
            "fmnmx.ftz.nan r0, r1, r2, !p3;",
        ),
        (
            Op::FSwzAdd(OpFSwzAdd {
                dst: r(0),
                srcs: [r(1), r(2)],
                ops: [
                    FSwzAddOp::SubRight,
                    FSwzAddOp::MoveLeft,
                    FSwzAddOp::Add,
                    FSwzAddOp::SubLeft,
                ],
                rnd: RoundMode::Zero,
                ftz: true,
                ndv: true,
            }),
            "fswzadd.ftz.rz.ndv r0, r1, r2, [subr, mov2, add, sub];",
        ),
    ];
    for (op, expected) in cases {
        assert_eq!(round_text(Instr::new(op)), expected);
    }
}

/// More of the printer's guesses: the conversions, `F2FP`'s `.merge_c`, `FSEL` and `FSET`'s
/// always-true comparison.
#[test]
fn unconfirmed_text_of_the_float_conversions() {
    let r = Gpr::new;
    let cases = [
        (
            Op::FRnd(OpFRnd {
                dst: r(0),
                src: AluSrc::reg(r(1)),
                dst_type: FloatType::F16,
                src_type: FloatType::F32,
                rnd: RoundMode::NegInf,
                ftz: false,
                high: false,
                wide: false,
            }),
            "frnd.f16.f32.floor r0, r1;",
        ),
        (
            Op::FRnd(OpFRnd {
                dst: r(0),
                src: AluSrc::UniformReg {
                    reg: ur(4),
                    abs: false,
                    neg: true,
                },
                dst_type: FloatType::F16,
                src_type: FloatType::F16,
                rnd: RoundMode::Zero,
                ftz: false,
                high: true,
                wide: false,
            }),
            "frnd.f16.trunc r0, -ur4.h1;",
        ),
        (
            Op::F2F(OpF2F {
                dst: r(2),
                src: AluSrc::CBuf {
                    cbuf: CBufRef {
                        buf: CBuf::Binding(3),
                        offset: 0x10,
                    },
                    abs: true,
                    neg: false,
                },
                dst_type: FloatType::F32,
                src_type: FloatType::F16,
                rnd: RoundMode::NearestEven,
                ftz: true,
                high: true,
                wide: false,
            }),
            "f2f.ftz.f32.f16 r2, |c[0x3][0x10]|.h1;",
        ),
        (
            Op::F2I(OpF2I {
                dst: r(0),
                src: AluSrc::reg(r(1)),
                dst_type: int(1, true),
                src_type: FloatType::F16,
                rnd: RoundMode::PosInf,
                ftz: true,
                ntz: true,
                high: true,
                wide: false,
            }),
            "f2i.ftz.s16.f16.ceil.ntz r0, r1.h1;",
        ),
        (
            Op::I2F(OpI2F {
                dst: r(0),
                src: AluSrc::reg(r(1)),
                dst_type: FloatType::F16,
                src_type: int(0, true),
                rnd: RoundMode::NearestEven,
                select: 2,
                wide: false,
            }),
            "i2f.f16.s8 r0, r1.b2;",
        ),
        (
            Op::I2FP(OpI2FP {
                dst: r(0),
                src: AluSrc::Imm(0x20),
                dst_type: FloatType::F32,
                src_type: IntType::U32,
                rnd: RoundMode::NegInf,
            }),
            "i2fp.f32.u32.rm r0, 0x20;",
        ),
        (
            Op::F2FP(OpF2FP {
                dst: r(0),
                srcs: [AluSrc::reg(r(1)), AluSrc::reg(r(2))],
                c: r(3),
                merge_c: true,
                h1: true,
                rnd: RoundMode::Zero,
            }),
            "f2fp.pack_ab.merge_c.h1.rz r0, r1, r2, r3;",
        ),
        (
            Op::FSel(OpFSel {
                dst: r(0),
                srcs: [
                    AluSrc::reg(r(1)),
                    AluSrc::CBuf {
                        cbuf: CBufRef {
                            buf: CBuf::BindlessUniform(ur(4)),
                            offset: 0x10,
                        },
                        abs: true,
                        neg: true,
                    },
                ],
                cond: PredSrc::TRUE,
            }),
            "fsel r0, r1, -|cx[ur4][0x10]|, pt;",
        ),
        (
            Op::FSet(OpFSet {
                dst: r(0),
                cmp: FloatCmpOp::True,
                srcs: [AluSrc::reg(r(1)), AluSrc::reg(r(2))],
                set_op: PredSetOp::Xor,
                accum: PredSrc {
                    reg: p(1),
                    negated: false,
                },
                ftz: false,
            }),
            "fset.bf.t.xor r0, r1, r2, p1;",
        ),
    ];
    for (op, expected) in cases {
        assert_eq!(round_text(Instr::new(op)), expected);
    }
}

// ---------------------------------------------------------------------------------------------
// Generated round trips of the group alone.

/// A word of one of the group's forms (as `testing::any_form_word`, over this group alone).
fn any_float_word() -> impl Strategy<Value = Instruction> {
    let forms: Vec<(u16, u128, u128)> = float::FORMS
        .iter()
        .map(|form| (form.opcode, form.known, form.fixed))
        .collect();
    (0..forms.len(), any::<u128>()).prop_map(move |(index, noise)| {
        let (opcode, known, fixed) = forms[index];
        Instruction::from_bits((noise & known) | fixed).with_field(OPCODE, u64::from(opcode))
    })
}

/// Whether `bit` of a word of `opcode` is one the documented layouts read only sometimes: the
/// high half (60) of `MUFU`, `F2I`, `F2F` and `FRND` (both opcodes of each conversion) in forms
/// 1, 5 and 6, `I2F`'s select (60, 61), `F2FP`'s `.H1` (72), and the other constant-buffer
/// layout's bits in a constant-buffer form.
fn layout_bit(opcode: u16, bit: u32) -> bool {
    let base = opcode & 0x1ff;
    let form = opcode >> 9;
    let cbuf_layout = matches!(form, 3 | 5) && matches!(bit, 32..38 | 54..59);
    let half = bit == 60
        && matches!(form, 1 | 5 | 6)
        && matches!(
            base,
            0x108 | 0x105 | 0x111 | 0x104 | 0x110 | 0x107 | 0x113
        );
    let select = matches!(bit, 60 | 61) && matches!(base, 0x106 | 0x112);
    let h1 = bit == 72 && base == 0x03e;
    cbuf_layout || half || select || h1
}

proptest! {
    #![proptest_config(config())]

    /// The group's operations alone, so each gets many more cases than in the crate-wide round
    /// trip: assembling then decoding gives the instruction back.
    #[test]
    fn float_assemble_then_decode_round_trips(
        guard in crate::testing::any_pred_src(),
        op in Union::new(float::any_ops()),
    ) {
        let instr = Instr { guard, op };
        let word = assemble(&instr).expect("the generator makes encodable instructions");
        prop_assert_eq!(decode(word), Ok(instr));
    }

    /// The group's words alone: a word decodes and re-assembles to itself (scheduling bits
    /// aside), or is refused for a bit only another layout reads, or for a `MUFU` function,
    /// combine operation, `FMUL` scale or float size that names nothing.
    #[test]
    fn float_decode_then_assemble_round_trips(word in any_float_word()) {
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
            Err(DecodeError::UnknownValue { opcode, bit, value, .. }) => {
                let base = opcode & 0x1ff;
                let dst_size = bit == CVT_DST_SIZE.start;
                let src_size = bit == CVT_SRC_SIZE.start;
                let float_size = match base {
                    FRND_OPCODE | FRND_WIDE_OPCODE | F2F_OPCODE | F2F_WIDE_OPCODE => {
                        dst_size || src_size
                    }
                    F2I_OPCODE | F2I_WIDE_OPCODE => src_size,
                    I2F_OPCODE | I2F_WIDE_OPCODE | I2FP_OPCODE => dst_size,
                    _ => false,
                };
                let zero = (base == FMUL_OPCODE && bit == FMUL_SCALE.start) || float_size;
                let refusable = (base == MUFU_OPCODE && bit == MUFU_OP.start)
                    || (matches!(base, 0x00a | 0x00b) && bit == SET_OP.start)
                    || (zero && value == 0);
                prop_assert!(refusable, "{opcode:#x} bit {bit}");
            }
            Err(error) => prop_assert!(false, "{error}"),
        }
    }
}
