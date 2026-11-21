//! Each instruction and modifier run on the interpreter, expected values from the documented
//! behaviour (the twins' from their vector instructions', the integer group's module docs).

use super::*;
use crate::decode::{
    IMadMode, IntCmpOp, OpBRev, OpFlo, OpIAdd3X, OpIMad, OpIMad64, OpISetP, OpLea, OpLop3, OpPopC,
    OpPrmt, OpSel, OpSgxt, OpShf, PredSetOp, PrmtMode, ShfType,
};

/// A `LEA` of `(a << shift) + b`, no carries.
pub(super) fn lea(dst: u8, a: AluSrc, b: AluSrc, shift: u8) -> OpLea {
    OpLea {
        dst: r(dst),
        overflow: Pred::PT,
        a,
        b,
        a_high: RZ,
        shift,
        dst_high: false,
        sx32: false,
        intermediate: false,
        carry: NOT_PT,
    }
}

/// `UMOV` of an immediate and of a register, `URZ` reading zero and a write to it dropped.
#[test]
fn umov_moves_immediates_and_registers() {
    let mut machine = machine_with(&[]);
    machine.uniform_regs[5] = 0x5555;
    run(
        &[
            Instr::new(Op::UMov(OpUMov {
                dst: ur(3),
                src: UMovSrc::Imm(0x1234_5678),
            })),
            Instr::new(Op::UMov(OpUMov {
                dst: ur(4),
                src: UMovSrc::Reg(ur(3)),
            })),
            Instr::new(Op::UMov(OpUMov {
                dst: ur(5),
                src: UMovSrc::Reg(UniformReg::URZ),
            })),
            Instr::new(Op::UMov(OpUMov {
                dst: UniformReg::URZ,
                src: UMovSrc::Imm(7),
            })),
        ],
        &mut machine,
    );
    assert_eq!(machine.uniform_regs[3], 0x1234_5678);
    assert_eq!(machine.uniform_regs[4], 0x1234_5678);
    assert_eq!(machine.uniform_regs[5], 0);
    assert_eq!(
        machine.uniform_regs[usize::from(UniformReg::URZ.index())],
        0
    );
    assert_eq!(
        machine.regs[3], 0xbad0_0003,
        "the per-thread file is untouched"
    );
}

/// `UIADD3` sums its three uniform sources with each negation, in form 1 and with an immediate
/// source 1 (form 4).
#[test]
fn uiadd3_sums_with_negations() {
    for (a, b, c) in [
        (1, 2, 3),
        (0xffff_ffff, 1, 0x8000_0000),
        (0x1234_5678, 0x7fff_ffff, 0),
    ] {
        let mut machine = machine_with(&[0, a, b, c]);
        run_op(
            Op::UIAdd3(twin(iadd3(4, [reg(1), neg(2), reg(3)]))),
            &mut machine,
        );
        assert_eq!(
            machine.uniform_regs[4],
            a.wrapping_sub(b).wrapping_add(c),
            "{a} {b} {c}"
        );
        run_op(
            Op::UIAdd3(twin(iadd3(5, [neg(1), AluSrc::Imm(b), neg(3)]))),
            &mut machine,
        );
        assert_eq!(
            machine.uniform_regs[5],
            b.wrapping_sub(a).wrapping_sub(c),
            "{a} {b} {c}"
        );
    }
}

/// A uniform carry chain (rows 63, 83): `UIADD3` with a carry out into `UP0`, then `UIADD3.X`
/// reading it, gives the exact 64-bit sum at the 32-bit carry boundary.
#[test]
fn a_uniform_carry_chain_adds_64_bits() {
    let pairs = [
        (0xffff_ffff_u64, 1_u64),
        (0x0000_0001_ffff_ffff, 0x0000_0000_0000_0001),
        (0x1234_5678_9abc_def0, 0xfedc_ba98_7654_3210),
        (0, 0),
    ];
    for (x, y) in pairs {
        let low = |v: u64| u32::try_from(v & u64::from(u32::MAX)).expect("low");
        let high = |v: u64| u32::try_from(v >> 32).expect("high");
        let mut machine = machine_with(&[low(x), high(x), low(y), high(y)]);
        machine.preds[0] = true;
        let chain = [
            Instr::new(Op::UIAdd3(twin(OpIAdd3 {
                overflow: [p(0), Pred::PT],
                ..iadd3(4, [reg(0), reg(2), RZ])
            }))),
            Instr::new(Op::UIAdd3X(twin(OpIAdd3X {
                add: OpIAdd3 {
                    carry: [ps(0, false), NOT_PT],
                    ..iadd3(5, [reg(1), reg(3), RZ])
                },
            }))),
        ];
        run(&chain, &mut machine);
        let sum = x.wrapping_add(y);
        assert_eq!(machine.uniform_regs[4], low(sum), "{x:#x} {y:#x}");
        assert_eq!(machine.uniform_regs[5], high(sum), "{x:#x} {y:#x}");
        let carried = u64::from(low(x)) + u64::from(low(y)) > u64::from(u32::MAX);
        assert_eq!(machine.uniform_preds[0], carried, "{x:#x} {y:#x}");
        assert!(machine.preds[0], "P0 is untouched");
    }
}

/// `ULEA` shifts and adds; `ULEA.HI` takes the high word of `a_high:a` shifted, plus `b`.
#[test]
fn ulea_shifts_and_adds() {
    for shift in [0_u8, 1, 16, 31] {
        for (a, b) in [
            (1, 2),
            (0xffff_ffff, 0x8000_0000),
            (0x1234_5678, 0xffff_fff0),
        ] {
            let mut machine = machine_with(&[0, a, b, 0xdead_0002]);
            run_op(Op::ULea(twin(lea(4, reg(1), reg(2), shift))), &mut machine);
            assert_eq!(
                machine.uniform_regs[4],
                (a << shift).wrapping_add(b),
                "{shift} {a}"
            );
            run_op(
                Op::ULea(twin(OpLea {
                    a_high: reg(3),
                    dst_high: true,
                    ..lea(5, reg(1), reg(2), shift)
                })),
                &mut machine,
            );
            let wide = (u64::from(0xdead_0002_u32) << 32 | u64::from(a)) << shift;
            let high = u32::try_from(wide >> 32 & u64::from(u32::MAX)).expect("a word");
            assert_eq!(machine.uniform_regs[5], high.wrapping_add(b), "{shift} {a}");
        }
    }
}

/// `ULOP3` applies its truth table bit by bit (`0x96` is a three-way exclusive or, `0xe8` the
/// majority, `0xf0` `a` alone).
#[test]
fn ulop3_applies_its_table() {
    let (a, b, c) = (0xff00_ff00_u32, 0xf0f0_f0f0_u32, 0xcccc_cccc_u32);
    for (lut, expected) in [
        (0x96, a ^ b ^ c),
        (0xe8, (a & b) | (a & c) | (b & c)),
        (0xf0, a),
        (0x00, 0),
        (0xff, u32::MAX),
    ] {
        let mut machine = machine_with(&[0, a, b, c]);
        run_op(
            Op::ULop3(twin(OpLop3 {
                pred: Pred::PT,
                dst: r(4),
                srcs: [reg(1), reg(2), reg(3)],
                lut,
            })),
            &mut machine,
        );
        assert_eq!(machine.uniform_regs[4], expected, "{lut:#x}");
    }
}

/// `UPRMT` with an immediate selector: each nibble picks a byte of `b:a`, or with bit 3 that
/// byte's sign repeated; `0x9910` sign-extends the low half-word (the lowering's one selector).
/// The native translator has no case for it (the module docs, "Divergences").
#[test]
fn uprmt_permutes_uniform_bytes() {
    let prmt = |sel: u32| {
        Op::UPrmt(twin(OpPrmt {
            dst: r(4),
            srcs: [reg(1), reg(2)],
            sel: AluSrc::Imm(sel),
            mode: PrmtMode::Index,
        }))
    };
    for value in [0_u32, 1, 0x7fff, 0x8000, 0xffff, 0xdead_8000, 0xabcd_7fff] {
        let mut machine = machine_with(&[0, value, 0x7654_3210]);
        run_op(prmt(0x9910), &mut machine);
        let half = u16::try_from(value & 0xffff).expect("a half-word");
        let expected = i32::from(half.cast_signed()).cast_unsigned();
        assert_eq!(machine.uniform_regs[4], expected, "{value:#x}");
    }
    let mut machine = machine_with(&[0, 0x3322_1100, 0x7766_5544]);
    run_op(prmt(0x4062), &mut machine);
    assert_eq!(machine.uniform_regs[4], 0x4400_6622);
}

/// `USHF`, NAK's types: a left `U32` shift of the low word, and a right `I32` shift of the high
/// word that sign-extends (PR #4588's "uniform signed right shift sign extends the high word").
#[test]
fn ushf_shifts_uniform_words() {
    let shf = |dst: u8, low: AluSrc, high: AluSrc, right: bool, data_type: ShfType| OpShf {
        dst: r(dst),
        low,
        shift: reg(3),
        high,
        data_type,
        wrap: false,
        right,
        dst_high: right,
    };
    for n in [0_u32, 1, 4, 31] {
        for value in [0x8000_0001_u32, 0x7fff_ffff, 0xffff_ffff] {
            let mut machine = machine_with(&[0, value, value, n]);
            run_op(
                Op::UShf(twin(shf(4, reg(1), RZ, false, ShfType::U32))),
                &mut machine,
            );
            assert_eq!(machine.uniform_regs[4], value << n, "{n} {value:#x}");
            run_op(
                Op::UShf(twin(shf(5, RZ, reg(2), true, ShfType::I32))),
                &mut machine,
            );
            let expected = (value.cast_signed() >> n).cast_unsigned();
            assert_eq!(machine.uniform_regs[5], expected, "{n} {value:#x}");
        }
    }
}

/// `UIMAD`: the low word of `a * b + c`, with an immediate factor too, wrapping.
#[test]
fn uimad_multiplies_and_adds() {
    let imad = |dst: u8, b: AluSrc| OpIMad {
        dst: r(dst),
        srcs: [reg(1), b, reg(3)],
        signed: false,
        mode: IMadMode::Lo {
            x: false,
            carry: NOT_PT,
        },
    };
    for (a, b, c) in [
        (3, 5, 7),
        (0xffff_ffff, 0xffff_ffff, 1),
        (0x1234_5678, 0x10, 0),
    ] {
        let mut machine = machine_with(&[0, a, b, c]);
        run_op(Op::UIMad(twin(imad(4, reg(2)))), &mut machine);
        assert_eq!(machine.uniform_regs[4], a.wrapping_mul(b).wrapping_add(c));
        run_op(Op::UIMad(twin(imad(5, AluSrc::Imm(b)))), &mut machine);
        assert_eq!(machine.uniform_regs[5], a.wrapping_mul(b).wrapping_add(c));
    }
}

/// `UIMAD.WIDE`: the 64-bit `a * b` plus a uniform register pair, signed and unsigned, into a
/// uniform register pair; the last pair it may start at is `UR60`.
#[test]
fn uimad_wide_writes_a_uniform_pair() {
    let limit = integer::UNIFORM_WIDE_PAIR_LIMIT;
    let last = limit - 2;
    for signed in [false, true] {
        for (a, b) in [(0xffff_ffff_u32, 0xffff_ffff_u32), (0x8000_0000, 2), (7, 9)] {
            let mut machine = machine_with(&[0, a, b, 0, 0x0000_0010, 0x0000_0001]);
            run_op(
                Op::UIMad64(twin(OpIMad64 {
                    dst: r(last),
                    srcs: [reg(1), reg(2), reg(4)],
                    signed,
                    carry: NOT_PT,
                })),
                &mut machine,
            );
            let product = if signed {
                i64::from(a.cast_signed())
                    .wrapping_mul(i64::from(b.cast_signed()))
                    .cast_unsigned()
            } else {
                u64::from(a) * u64::from(b)
            };
            let sum = product.wrapping_add(0x1_0000_0010);
            let index = usize::from(last);
            let low = u32::try_from(sum & u64::from(u32::MAX)).expect("low");
            let high = u32::try_from(sum >> 32).expect("high");
            assert_eq!(machine.uniform_regs[index], low, "{signed} {a:#x}");
            assert_eq!(machine.uniform_regs[index + 1], high, "{signed} {a:#x}");
        }
    }
}

/// `UISETP`: the comparison combined with the accumulator into the first uniform predicate, its
/// inverse into the second, unsigned and signed.
#[test]
fn uisetp_compares_into_uniform_predicates() {
    for signed in [false, true] {
        for (a, b) in [(1_u32, 2_u32), (0xffff_ffff, 1), (5, 5)] {
            for accum in [false, true] {
                let mut machine = machine_with(&[0, a, b]);
                machine.uniform_preds[3] = accum;
                run_op(
                    Op::UISetP(twin(OpISetP {
                        dsts: [p(1), p(2)],
                        srcs: [reg(1), reg(2)],
                        cmp_op: IntCmpOp::Lt,
                        signed,
                        set_op: PredSetOp::And,
                        ex: false,
                        accum: ps(3, false),
                        low_cmp: PredSrc::TRUE,
                    })),
                    &mut machine,
                );
                let less = if signed {
                    a.cast_signed() < b.cast_signed()
                } else {
                    a < b
                };
                assert_eq!(machine.uniform_preds[1], less && accum, "{signed} {a} {b}");
                assert_eq!(machine.uniform_preds[2], !less && accum, "{signed} {a} {b}");
            }
        }
    }
}

/// `USEL`: `a` where the uniform predicate holds, else `b`.
#[test]
fn usel_selects_by_a_uniform_predicate() {
    for (cond, negated) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut machine = machine_with(&[0, 0x1111, 0x2222]);
        machine.uniform_preds[1] = cond;
        run_op(
            Op::USel(twin(OpSel {
                dst: r(4),
                srcs: [reg(1), reg(2)],
                cond: ps(1, negated),
            })),
            &mut machine,
        );
        let expected = if cond == negated { 0x2222 } else { 0x1111 };
        assert_eq!(machine.uniform_regs[4], expected, "{cond} {negated}");
    }
}

/// `USGXT`: the low `bits` bits sign- or zero-extended; 0 gives 0 and 32 or more the value.
#[test]
fn usgxt_extends_the_low_bits() {
    for signed in [false, true] {
        for bits in [0_u32, 1, 8, 31, 32, 40] {
            let value = 0x0000_00f0_u32;
            let mut machine = machine_with(&[0, value, bits]);
            run_op(
                Op::USgxt(twin(OpSgxt {
                    dst: r(4),
                    a: reg(1),
                    bits: reg(2),
                    signed,
                    wrap: false,
                })),
                &mut machine,
            );
            let expected = match bits {
                0 => 0,
                32.. => value,
                _ => {
                    let shift = 32 - bits;
                    if signed {
                        ((value << shift).cast_signed() >> shift).cast_unsigned()
                    } else {
                        (value << shift) >> shift
                    }
                }
            };
            assert_eq!(machine.uniform_regs[4], expected, "{signed} {bits}");
        }
    }
}

/// `UFLO`, `UBREV` and `UPOPC` on one uniform source.
#[test]
fn unary_twins() {
    for value in VALUES {
        let mut machine = machine_with(&[0, value]);
        run(
            &[
                Instr::new(Op::UFlo(twin(OpFlo {
                    dst: r(4),
                    src: reg(1),
                    signed: false,
                    return_shift_amount: false,
                }))),
                Instr::new(Op::UFlo(twin(OpFlo {
                    dst: r(5),
                    src: reg(1),
                    signed: false,
                    return_shift_amount: true,
                }))),
                Instr::new(Op::UBRev(twin(OpBRev {
                    dst: r(6),
                    src: reg(1),
                }))),
                Instr::new(Op::UPopC(twin(OpPopC {
                    dst: r(7),
                    src: reg(1),
                }))),
            ],
            &mut machine,
        );
        let msb = if value == 0 { u32::MAX } else { value.ilog2() };
        assert_eq!(machine.uniform_regs[4], msb, "{value:#x}");
        assert_eq!(
            machine.uniform_regs[5],
            31_u32.wrapping_sub(msb),
            "{value:#x}"
        );
        assert_eq!(machine.uniform_regs[6], value.reverse_bits(), "{value:#x}");
        assert_eq!(machine.uniform_regs[7], value.count_ones(), "{value:#x}");
    }
}

/// `UPLOP3` with every truth table: each destination is bit `a << 2 | b << 1 | c` of its table,
/// over every value of its three uniform predicates (row 88).
#[test]
fn uplop3_takes_any_truth_table() {
    let function = |lut: u8| {
        build(&[Instr::new(Op::UPLop3(OpUPLop3 {
            dsts: [up(1), up(2)],
            srcs: [ups(3, false), ups(4, false), ups(5, true)],
            luts: [lut, !lut],
        }))])
        .expect("translates")
    };
    for lut in 0..=u8::MAX {
        let function = function(lut);
        for row in 0..8_u8 {
            let (a, b, c) = (row & 4 != 0, row & 2 != 0, row & 1 != 0);
            let mut machine = Machine::new();
            machine.uniform_preds[3] = a;
            machine.uniform_preds[4] = b;
            machine.uniform_preds[5] = !c;
            // The per-thread predicates of the same numbers hold the opposite values.
            machine.preds = [a, b, !c, !a, !b, c, a, false];
            let preds = machine.preds;
            machine.run(&function).expect("runs");
            let expected = lut >> row & 1 != 0;
            assert_eq!(machine.uniform_preds[1], expected, "{lut:#x} {row}");
            assert_eq!(machine.uniform_preds[2], !expected, "{lut:#x} {row}");
            assert_eq!(machine.preds, preds, "{lut:#x} {row}");
        }
    }
}

/// `R2UR` copies the source register of the lowest active lane into the uniform register: on
/// the interpreter's one-lane warp (lane 5 here), its own value; the IR is the ballot, its
/// lowest bit, that bit's lane and a shuffle from it.
#[test]
fn r2ur_copies_the_lowest_active_lane() {
    let r2ur = Instr::new(Op::R2UR(OpR2UR {
        dst: ur(3),
        src: r(7),
    }));
    let mut machine = machine_with(&[]);
    machine.system.lane_id = 5;
    run(&[r2ur], &mut machine);
    assert_eq!(machine.uniform_regs[3], 0xbad0_0007);
    assert_eq!(
        text(&[r2ur]),
        "block0:\n    %0: u32 = SubgroupBallot u1 true\n    %1: u32 = INeg32 %0\n    \
         %2: u32 = BitwiseAnd32 %0, %1\n    %3: u32 = FindUMsb32 %2\n    \
         %4: u32 = GetRegister r7\n    %5: u32 = ShuffleIndex %4, %3, u32 0x1f, u32 0x0\n    \
         %6 = SetUniformRegister ur3, %5\n    return\n"
    );
}
