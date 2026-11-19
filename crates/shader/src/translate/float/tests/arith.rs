//! Arithmetic, comparisons, selects, `MUFU` and `FSWZADD`.

use super::*;

const THREE: u32 = 0x4040_0000;
const HALF: u32 = 0x3f00_0000;
const NEG_ONE: u32 = 0xbf80_0000;

/// `FADD` adds, saturates with `.SAT`, and applies each operand's absolute value before its
/// negation.
#[test]
fn fadd_adds_saturates_and_modifies() {
    let add = |op: OpFAdd, a: u32, b: u32| run_op(Op::FAdd(op), &[(1, a), (2, b)]).regs[0];
    let plain = fadd(r(0), reg(1), reg(2));
    assert_eq!(add(plain, ONE, TWO), THREE);
    // |-1| negated is -1; -1 + -1 = -2.
    let modified = fadd(r(0), reg_mod(1, true, true), reg_mod(2, false, true));
    assert_eq!(add(modified, NEG_ONE, ONE), 0xc000_0000);
    let saturated = OpFAdd {
        saturate: true,
        ..plain
    };
    assert_eq!(add(saturated, ONE, TWO), ONE);
    assert_eq!(add(saturated, NEG_ONE, NEG_ONE), 0);
    assert_eq!(add(saturated, HALF, 0), HALF);
    // A write to `RZ` is dropped.
    let to_rz = fadd(Gpr::RZ, reg(1), reg(2));
    let machine = run_op(Op::FAdd(to_rz), &[(1, ONE), (2, TWO)]);
    assert_eq!(machine.regs[usize::from(Gpr::RZ.index())], 0);
    // The second operand in each wide slot: an immediate, a constant buffer, a uniform register.
    for b in [imm(2.0), cbuf(0, 4), ureg(7)] {
        let mut machine = Machine::new();
        machine.regs[1] = ONE;
        machine
            .cbufs
            .insert(0, [0, 0, 0, 0, 0, 0, 0, 0x40].to_vec());
        machine.uniform_regs[7] = TWO;
        run_on(&[Instr::new(Op::FAdd(fadd(r(0), reg(1), b)))], &mut machine);
        assert_eq!(machine.regs[0], THREE, "{b:?}");
    }
}

/// PR #4588's "Native scaled multiplication preserves captured power-of-two scales": with
/// `.FTZ` and to-nearest, scale `s` multiplies by `2^(s - 4)`: `8 * 4` and `-2 * 3`.
#[test]
fn fmul_scales_like_the_pr() {
    let factors = [
        (FMulScale::D8, 0.125_f32),
        (FMulScale::D4, 0.25),
        (FMulScale::D2, 0.5),
        (FMulScale::None, 1.0),
        (FMulScale::M2, 2.0),
        (FMulScale::M4, 4.0),
        (FMulScale::M8, 8.0),
    ];
    for (scale, factor) in factors {
        let op = OpFMul {
            scale,
            ftz: true,
            ..fmul(r(0), reg(0), reg(1))
        };
        for (a, b, product) in [(8.0_f32, 4.0_f32, 32.0_f32), (-2.0, 3.0, -6.0)] {
            let machine = run_op(Op::FMul(op), &[(0, a.to_bits()), (1, b.to_bits())]);
            assert_eq!(machine.regs[0], (product * factor).to_bits(), "{scale:?}");
        }
    }
    // An unscaled multiply needs neither `.FTZ` nor to-nearest.
    let op = OpFMul {
        rnd: RoundMode::Zero,
        ..fmul(r(0), reg(0), reg(1))
    };
    let machine = run_op(Op::FMul(op), &[(0, THREE), (1, HALF)]);
    assert_eq!(machine.regs[0], 1.5_f32.to_bits());
}

/// The scale's guard (`scale == 4 || (ftz && rounding == 0)`) with each condition the only one
/// that varies.
#[test]
fn fmul_scale_needs_ftz_and_nearest() {
    let scaled = OpFMul {
        scale: FMulScale::D2,
        ftz: true,
        ..fmul(r(0), reg(1), reg(2))
    };
    translated(&[Instr::new(Op::FMul(scaled))]);
    for op in [
        OpFMul {
            ftz: false,
            ..scaled
        },
        OpFMul {
            rnd: RoundMode::Zero,
            ..scaled
        },
        OpFMul {
            rnd: RoundMode::NegInf,
            ..scaled
        },
        OpFMul {
            rnd: RoundMode::PosInf,
            ..scaled
        },
    ] {
        assert_refused(Op::FMul(op), FloatUnsupported::FmulScaleModifiers);
    }
    // Scale 1, the first the PR accepts; scale 0, the last it refuses, the decoder refuses
    // (slice 4b), so no decoded `FMUL` carries it.
    translated(&[Instr::new(Op::FMul(OpFMul {
        scale: FMulScale::D8,
        ..scaled
    }))]);
}

/// PR #4588's "Native fused multiply-add reads a uniform multiplier or addend": `UR4 = 3`,
/// then `FFMA` with `UR4` as the multiplier (`8 * 3 + 4`, `-2 * 3 + 5`) or the addend
/// (`8 * 4 + 3`, `-2 * 5 + 3`).
#[test]
fn ffma_reads_a_uniform_multiplier_or_addend() {
    for uniform_addend in [false, true] {
        let op = if uniform_addend {
            ffma(r(2), reg(0), reg(1), ureg(4))
        } else {
            ffma(r(2), reg(0), ureg(4), reg(1))
        };
        let cases = if uniform_addend {
            [(8.0_f32, 4.0_f32, 35.0_f32), (-2.0, 5.0, -7.0)]
        } else {
            [(8.0, 4.0, 28.0), (-2.0, 5.0, -1.0)]
        };
        for (a, b, expected) in cases {
            let mut machine = Machine::new();
            machine.regs[0] = a.to_bits();
            machine.regs[1] = b.to_bits();
            machine.uniform_regs[4] = THREE;
            run_on(&[Instr::new(Op::FFma(op))], &mut machine);
            assert_eq!(machine.regs[2], expected.to_bits());
        }
    }
}

/// PR #4588's "Native fused multiply add retains its uniform multiplier and aliased
/// accumulator": `FFMA R18, R5, UR8, R18`, with and without `.SAT`. The last case is fused:
/// `(1 + 2^-23) * (1 - 2^-23) - 1` is `-2^-46`, where a separate multiply would round the
/// product to 1 and give 0.
#[test]
fn ffma_fuses_and_saturates() {
    let cases: [([u32; 3], u32, u32); 5] = [
        // a, b, c; the result plain; saturated.
        ([NEG_ONE, THREE, TWO], 0xbf80_0000, 0),
        ([THREE, TWO, 0], 0x40c0_0000, ONE),
        ([0x3e80_0000, TWO, 0x3e80_0000], 0x3f40_0000, 0x3f40_0000),
        ([0xc000_0000, 0xc040_0000, 0xc0a0_0000], ONE, ONE),
        ([0x3f80_0001, 0x3f7f_fffe, NEG_ONE], 0xa880_0000, 0),
    ];
    for saturate in [false, true] {
        let op = OpFFma {
            saturate,
            ..ffma(r(18), reg(5), ureg(8), reg(18))
        };
        for ([a, b, c], plain, saturated) in cases {
            let mut machine = Machine::new();
            machine.regs[5] = a;
            machine.uniform_regs[8] = b;
            machine.regs[18] = c;
            run_on(&[Instr::new(Op::FFma(op))], &mut machine);
            let expected = if saturate { saturated } else { plain };
            assert_eq!(machine.regs[18], expected);
            assert_eq!(machine.regs[5], a);
            assert_eq!(machine.uniform_regs[8], b);
        }
    }
}

/// `FADD`, `FMUL` and `FFMA` carry their rounding field and `.FTZ` (owner's decision 10), and
/// the interpreter models the carried mode: `1 + 2^-24` rounds up only toward positive
/// infinity.
#[test]
fn arithmetic_carries_rounding_and_flushing() {
    let modes = [
        (RoundMode::NearestEven, FpRounding::Rn),
        (RoundMode::NegInf, FpRounding::Rm),
        (RoundMode::PosInf, FpRounding::Rp),
        (RoundMode::Zero, FpRounding::Rz),
    ];
    for (rnd, rounding) in modes {
        for (ftz, fmz_mode) in [(false, FmzMode::None), (true, FmzMode::Ftz)] {
            let wanted = vec![Flags::Fp(FpControl {
                no_contraction: true,
                rounding,
                fmz_mode,
            })];
            let ops = [
                (
                    Op::FAdd(OpFAdd {
                        rnd,
                        ftz,
                        ..fadd(r(0), reg(1), reg(2))
                    }),
                    Opcode::FPAdd32,
                ),
                (
                    Op::FMul(OpFMul {
                        rnd,
                        ftz,
                        ..fmul(r(0), reg(1), reg(2))
                    }),
                    Opcode::FPMul32,
                ),
                (
                    Op::FFma(OpFFma {
                        rnd,
                        ftz,
                        ..ffma(r(0), reg(1), reg(2), reg(3))
                    }),
                    Opcode::FPFma32,
                ),
            ];
            for (op, opcode) in ops {
                let function = translated(&[Instr::new(op)]);
                assert_eq!(flags_of(&function, opcode), wanted, "{op}");
            }
        }
        let op = OpFAdd {
            rnd,
            ..fadd(r(0), reg(1), reg(2))
        };
        let sum = run_op(Op::FAdd(op), &[(1, ONE), (2, 0x3380_0000)]).regs[0];
        let expected = if rnd == RoundMode::PosInf {
            0x3f80_0001
        } else {
            ONE
        };
        assert_eq!(sum, expected, "{rnd:?}");
    }
}

/// PR #4588's "Native floating min/max clamps actual shader input": `FMNMX R0, R0, c[3][0],
/// !PT` (the maximum) then `FMNMX R0, R0, c[3][4], PT` (the minimum) clamp to `[-1, 2]`.
#[test]
fn fmnmx_clamps() {
    let program = [
        Instr::new(Op::FMnMx(OpFMnMx {
            ftz: true,
            ..fmnmx(r(0), reg(0), cbuf(3, 0), NOT_PT)
        })),
        Instr::new(Op::FMnMx(OpFMnMx {
            ftz: true,
            ..fmnmx(r(0), reg(0), cbuf(3, 4), PredSrc::TRUE)
        })),
    ];
    for (value, clamped) in [
        (-20.0_f32, -1.0_f32),
        (-1.0, -1.0),
        (0.25, 0.25),
        (2.0, 2.0),
        (100.0, 2.0),
    ] {
        let mut machine = Machine::new();
        machine.regs[0] = value.to_bits();
        let mut bank = (-1.0_f32).to_le_bytes().to_vec();
        bank.extend(2.0_f32.to_le_bytes());
        machine.cbufs.insert(3, bank);
        run_on(&program, &mut machine);
        assert_eq!(machine.regs[0], clamped.to_bits());
    }
}

/// The predicate picks the minimum, a negated one the maximum, read from the register.
#[test]
fn fmnmx_predicate_picks_the_minimum() {
    for negated in [false, true] {
        for p2 in [false, true] {
            let op = fmnmx(r(0), reg(1), reg(2), pred(2, negated));
            let mut machine = Machine::new();
            machine.regs[1] = ONE;
            machine.regs[2] = TWO;
            machine.preds[2] = p2;
            run_on(&[Instr::new(Op::FMnMx(op))], &mut machine);
            let minimum = p2 != negated;
            assert_eq!(machine.regs[0], if minimum { ONE } else { TWO });
        }
    }
}

/// PR #4588's "Native NaN-propagating floating min/max preserves finite values and aliases":
/// `FMNMX.NAN` gives `0x7fffffff` when either operand is a NaN and the minimum or maximum
/// otherwise; without `.NAN` the operand that is not a NaN.
#[test]
fn fmnmx_nan_propagates() {
    let values = [
        0.0_f32,
        1.0,
        -1.0,
        123.5,
        -23.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for maximum in [false, true] {
        let min = if maximum { NOT_PT } else { PredSrc::TRUE };
        for dst in [0, 1, 2] {
            let op = OpFMnMx {
                nan: true,
                ..fmnmx(r(dst), reg(0), reg(1), min)
            };
            for a in values {
                for b in values {
                    let machine = run_op(Op::FMnMx(op), &[(0, a.to_bits()), (1, b.to_bits())]);
                    let expected = if a.is_nan() || b.is_nan() {
                        FMNMX_NAN
                    } else if maximum {
                        a.max(b).to_bits()
                    } else {
                        a.min(b).to_bits()
                    };
                    assert_eq!(machine.regs[usize::from(dst)], expected, "{a} {b}");
                }
            }
        }
    }
    let plain = fmnmx(r(0), reg(0), reg(1), PredSrc::TRUE);
    let machine = run_op(Op::FMnMx(plain), &[(0, QNAN), (1, TWO)]);
    assert_eq!(machine.regs[0], TWO);
}

/// `FMNMX`, `FSET` and `FSETP` carry `.FTZ` and no rounding on their comparisons.
#[test]
fn comparisons_carry_flushing() {
    for (ftz, fmz_mode) in [(false, FmzMode::None), (true, FmzMode::Ftz)] {
        let wanted = Flags::Fp(FpControl {
            no_contraction: false,
            rounding: FpRounding::DontCare,
            fmz_mode,
        });
        let minmax = translated(&[Instr::new(Op::FMnMx(OpFMnMx {
            ftz,
            ..fmnmx(r(0), reg(1), reg(2), PredSrc::TRUE)
        }))]);
        assert_eq!(flags_of(&minmax, Opcode::FPMax32), [wanted]);
        assert_eq!(flags_of(&minmax, Opcode::FPMin32), [wanted]);
        let set = translated(&[Instr::new(Op::FSet(OpFSet {
            ftz,
            ..fset(r(0), FloatCmpOp::Lt, reg(1), reg(2))
        }))]);
        assert_eq!(flags_of(&set, Opcode::FPOrdLessThan32), [wanted]);
        let setp = translated(&[Instr::new(Op::FSetP(OpFSetP {
            ftz,
            ..fsetp(Pred::PT, Pred::PT, FloatCmpOp::GeU, reg(1), reg(2))
        }))]);
        assert_eq!(flags_of(&setp, Opcode::FPUnordGreaterThanEqual32), [wanted]);
    }
}

/// Every comparison, against IEEE 754's definitions (the `u` forms are also true for an
/// unordered pair), on a less, an equal, a greater and an unordered pair.
#[test]
fn every_comparison() {
    use FloatCmpOp as C;
    // For (1, 2), (2, 2), (2, 1), (NaN, 2).
    let table = [
        (C::False, [false, false, false, false]),
        (C::Lt, [true, false, false, false]),
        (C::Eq, [false, true, false, false]),
        (C::Le, [true, true, false, false]),
        (C::Gt, [false, false, true, false]),
        (C::Ne, [true, false, true, false]),
        (C::Ge, [false, true, true, false]),
        (C::Num, [true, true, true, false]),
        (C::Nan, [false, false, false, true]),
        (C::LtU, [true, false, false, true]),
        (C::EqU, [false, true, false, true]),
        (C::LeU, [true, true, false, true]),
        (C::GtU, [false, false, true, true]),
        (C::NeU, [true, false, true, true]),
        (C::GeU, [false, true, true, true]),
        (C::True, [true, true, true, true]),
    ];
    // The unordered pair both ways round, so `.num` and `.nan` see a NaN in either operand.
    let pairs = [(ONE, TWO), (TWO, TWO), (TWO, ONE), (QNAN, TWO), (TWO, QNAN)];
    for (cmp, results) in table {
        let [less, equal, greater, unordered] = results;
        let holds = [less, equal, greater, unordered, unordered];
        for ((a, b), holds) in pairs.into_iter().zip(holds) {
            let set = run_op(Op::FSet(fset(r(0), cmp, reg(1), reg(2))), &[(1, a), (2, b)]);
            assert_eq!(
                set.regs[0],
                if holds { FSET_TRUE } else { 0 },
                "{cmp} {a:#x} {b:#x}"
            );
        }
    }
}

/// PR #4588's "Native boolean float comparison preserves register moves and absolute inputs":
/// `FSET.BF.LT.FTZ R1, |R0|, c[3][0]` with `c[3][0] = 2` writes `1.0` when `|x| < 2`.
#[test]
fn fset_writes_a_boolean_float() {
    let op = OpFSet {
        ftz: true,
        ..fset(r(1), FloatCmpOp::Lt, reg_mod(0, true, false), cbuf(3, 0))
    };
    for value in [-3.0_f32, -1.0, 0.0, 2.0, 4.0] {
        let mut machine = Machine::new();
        machine.regs[0] = value.to_bits();
        machine.cbufs.insert(3, 2.0_f32.to_le_bytes().to_vec());
        run_on(&[Instr::new(Op::FSet(op))], &mut machine);
        let expected = if value.abs() < 2.0 { FSET_TRUE } else { 0 };
        assert_eq!(machine.regs[1], expected, "{value}");
    }
}

/// The accumulator joins the comparison by each combine operation, for `FSET` and `FSETP`.
#[test]
fn combine_operations() {
    for set_op in [PredSetOp::And, PredSetOp::Or, PredSetOp::Xor] {
        for accum_value in [false, true] {
            for (a, holds) in [(ONE, true), (THREE, false)] {
                let expected = match set_op {
                    PredSetOp::And => holds && accum_value,
                    PredSetOp::Or => holds || accum_value,
                    PredSetOp::Xor => holds != accum_value,
                };
                let set = OpFSet {
                    set_op,
                    accum: pred(4, false),
                    ..fset(r(0), FloatCmpOp::Lt, reg(1), imm(2.0))
                };
                let mut machine = Machine::new();
                machine.regs[1] = a;
                machine.preds[4] = accum_value;
                run_on(&[Instr::new(Op::FSet(set))], &mut machine);
                assert_eq!(machine.regs[0], if expected { FSET_TRUE } else { 0 });
                // A negated accumulator holding the opposite value gives the same.
                let negated = OpFSet {
                    accum: pred(4, true),
                    ..set
                };
                machine.preds[4] = !accum_value;
                run_on(&[Instr::new(Op::FSet(negated))], &mut machine);
                assert_eq!(machine.regs[0], if expected { FSET_TRUE } else { 0 });
                // FSETP's second predicate combines the negated comparison.
                let setp = OpFSetP {
                    set_op,
                    accum: pred(4, false),
                    ..fsetp(
                        Pred::new(0).expect("p0"),
                        Pred::new(1).expect("p1"),
                        FloatCmpOp::Lt,
                        reg(1),
                        imm(2.0),
                    )
                };
                machine.preds[4] = accum_value;
                run_on(&[Instr::new(Op::FSetP(setp))], &mut machine);
                let second = match set_op {
                    PredSetOp::And => !holds && accum_value,
                    PredSetOp::Or => !holds || accum_value,
                    PredSetOp::Xor => holds == accum_value,
                };
                assert_eq!((machine.preds[0], machine.preds[1]), (expected, second));
            }
        }
    }
}

/// PR #4588's "Native float predicate comparisons retain uniform values and source modifiers":
/// `FSETP.LE.FTZ.AND P2, PT, RZ, URn, PT` sets `P2` to `0 <= operand` with every modifier and
/// `URZ` reading 0, and leaves the other predicates; a `PT` destination writes nothing (the
/// PR computes it and discards it, a listed divergence).
#[test]
fn fsetp_writes_both_predicates() {
    let p2 = Pred::new(2).expect("p2");
    for (abs, neg) in [(false, false), (true, false), (false, true), (true, true)] {
        for source in [0_u8, 5, 15, 16, 17, 18, 30, 32, 46, 48, 62, 63] {
            let b = AluSrc::UniformReg {
                reg: UniformReg::new(source).expect("a uniform register"),
                abs,
                neg,
            };
            let op = OpFSetP {
                ftz: true,
                ..fsetp(p2, Pred::PT, FloatCmpOp::Le, AluSrc::reg(Gpr::RZ), b)
            };
            let values = [
                -2.0_f32,
                -0.0,
                0.0,
                3.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
            ];
            for value in values {
                let mut machine = Machine::new();
                if source != UniformReg::URZ.index() {
                    machine.uniform_regs[usize::from(source)] = value.to_bits();
                }
                for (index, slot) in machine.preds.iter_mut().enumerate().take(7) {
                    *slot = index % 2 == 1;
                }
                let mut expected = machine.preds;
                let mut operand = if source == UniformReg::URZ.index() {
                    0.0
                } else {
                    value
                };
                if abs {
                    operand = operand.abs();
                }
                if neg {
                    operand = -operand;
                }
                expected[2] = 0.0 <= operand;
                run_on(&[Instr::new(Op::FSetP(op))], &mut machine);
                assert_eq!(machine.preds, expected, "{source} {value}");
            }
        }
    }
    let function = translated(&[Instr::new(Op::FSetP(fsetp(
        p2,
        Pred::PT,
        FloatCmpOp::Lt,
        reg(0),
        reg(1),
    )))]);
    assert_eq!(flags_of(&function, Opcode::LogicalNot), []);
    // The same destination twice: the second write, the negated comparison, wins.
    let op = fsetp(p2, p2, FloatCmpOp::Lt, reg(0), reg(1));
    let machine = run_op(Op::FSetP(op), &[(0, ONE), (1, TWO)]);
    assert!(!machine.preds[2]);
}

/// `FSEL` copies the bits of the operand its predicate picks, after the operand's modifiers.
#[test]
fn fsel_selects_bits() {
    for negated in [false, true] {
        for p1 in [false, true] {
            let op = OpFSel {
                dst: r(0),
                srcs: [reg_mod(2, true, false), reg_mod(3, false, true)],
                cond: pred(1, negated),
            };
            let mut machine = Machine::new();
            machine.regs[2] = 0xffc0_1234; // A NaN with a payload: abs clears only the sign.
            machine.regs[3] = ONE;
            machine.preds[1] = p1;
            run_on(&[Instr::new(Op::FSel(op))], &mut machine);
            let first = p1 != negated;
            assert_eq!(machine.regs[0], if first { 0x7fc0_1234 } else { NEG_ONE });
        }
    }
}

/// Each `MUFU` function the PR translates, at points where the result is exact.
#[test]
fn mufu_functions() {
    let cases = [
        (MuFuOp::Exp2, 3.0_f32, 8.0_f32),
        (MuFuOp::Log2, 8.0, 3.0),
        (MuFuOp::Rcp, 4.0, 0.25),
        (MuFuOp::Rsq, 4.0, 0.5),
        (MuFuOp::Sqrt, 9.0, 3.0),
        (MuFuOp::Sin, 0.0, 0.0),
        (MuFuOp::Cos, 0.0, 1.0),
    ];
    for (op, input, output) in cases {
        let machine = run_op(Op::MuFu(mufu(r(0), op, reg(1))), &[(1, input.to_bits())]);
        assert_eq!(machine.regs[0], output.to_bits(), "{op}");
    }
    // The operand's modifiers: `rcp(-|-2|)`.
    let op = mufu(r(0), MuFuOp::Rcp, reg_mod(1, true, true));
    let machine = run_op(Op::MuFu(op), &[(1, 0xc000_0000)]);
    assert_eq!(machine.regs[0], 0xbf00_0000);
}

/// `MUFU.SIN` and `MUFU.COS` take turns (row 95), not radians as Eden's Maxwell `MUFU` and PR
/// #4588's native one do: a quarter turn's sine is 1 and half a turn's cosine is -1, where
/// radians would give `sin 0.25` and `cos 0.5`.
#[test]
fn mufu_sine_and_cosine_take_turns() {
    let sine = run_op(
        Op::MuFu(mufu(r(0), MuFuOp::Sin, reg(1))),
        &[(1, 0.25_f32.to_bits())],
    );
    assert_eq!(sine.regs[0], ONE);
    let cosine = run_op(Op::MuFu(mufu(r(0), MuFuOp::Cos, reg(1))), &[(1, HALF)]);
    assert_eq!(cosine.regs[0], NEG_ONE);
    let function = translated(&[Instr::new(Op::MuFu(mufu(r(0), MuFuOp::Sin, reg(1))))]);
    assert_eq!(
        opcodes(&function),
        [
            Opcode::GetRegister,
            Opcode::BitCastF32U32,
            Opcode::FPMul32,
            Opcode::FPSin,
            Opcode::BitCastU32F32,
            Opcode::SetRegister,
        ]
    );
    // The other functions are not scaled.
    let function = translated(&[Instr::new(Op::MuFu(mufu(r(0), MuFuOp::Exp2, reg(1))))]);
    assert!(!opcodes(&function).contains(&Opcode::FPMul32));
}

/// The 16-bit `MUFU` reads the chosen half of a register, with its modifiers, and writes the
/// result into the low half with zero above.
#[test]
fn mufu_half_precision() {
    // R1: high half 2.0 (0x4000), low half 4.0 (0x4400).
    for (high, neg, expected) in [
        (true, false, 0x0000_3800),  // rcp 2 = 0.5
        (false, false, 0x0000_3400), // rcp 4 = 0.25
        (true, true, 0x0000_b800),   // rcp -2 = -0.5
    ] {
        let op = OpMuFu {
            f16: true,
            high,
            ..mufu(r(0), MuFuOp::Rcp, reg_mod(1, false, neg))
        };
        let machine = run_op(Op::MuFu(op), &[(1, 0x4000_4400)]);
        assert_eq!(machine.regs[0], expected, "high {high} neg {neg}");
    }
}

/// PR #4588's "Native uniform reciprocal uses the uniform source and preserves both register
/// banks": `MUFU.RCP R4, URn` with each modifier, `1 / x` correctly rounded (binary32 division
/// is exact to the last bit on every host).
#[test]
fn mufu_uniform_reciprocal() {
    for (abs, neg) in [(false, false), (true, false), (false, true), (true, true)] {
        let src = AluSrc::UniformReg {
            reg: UniformReg::new(4).expect("ur4"),
            abs,
            neg,
        };
        for bits in [0_u32, 0x8000_0000, ONE, NEG_ONE, TWO, INF, NEG_INF] {
            let mut machine = Machine::new();
            machine.uniform_regs[4] = bits;
            machine.regs[4] = 0x1234_5678;
            run_on(
                &[Instr::new(Op::MuFu(mufu(r(4), MuFuOp::Rcp, src)))],
                &mut machine,
            );
            let mut value = f32::from_bits(bits);
            if abs {
                value = value.abs();
            }
            if neg {
                value = -value;
            }
            assert_eq!(
                machine.regs[4],
                (1.0 / value).to_bits(),
                "{bits:#x} {abs} {neg}"
            );
            assert_eq!(machine.uniform_regs[4], bits);
        }
    }
}

/// PR #4588's "Native quad butterfly and swizzled add retain horizontal and vertical
/// differences", the `FSWZADD` half (the butterfly shuffle is slice 12h's): with each lane's
/// partner value in R16, `FSWZADD R12, R16, R5` gives the horizontal difference 3 (operation
/// byte `0x66`) or the vertical difference 11 (`0x5a`) in every lane.
#[test]
fn fswzadd_takes_quad_differences() {
    let value = |lane: u32| {
        let lane = f32::from(u16::try_from(lane).expect("a lane"));
        (lane / 4.0).floor() * 100.0 + (lane % 2.0) * 3.0 + ((lane / 2.0).floor() % 2.0) * 11.0
    };
    for (vertical, (ops, _)) in [
        (false, FSWZADD_OPERATIONS[1]),
        (true, FSWZADD_OPERATIONS[0]),
    ] {
        let op = fswzadd(r(12), r(16), r(5), ops);
        for lane in 0..32_u32 {
            let partner = lane ^ if vertical { 2 } else { 1 };
            let mut machine = Machine::new();
            machine.system.lane_id = lane;
            machine.regs[5] = value(lane).to_bits();
            machine.regs[16] = value(partner).to_bits();
            run_on(&[Instr::new(Op::FSwzAdd(op))], &mut machine);
            let expected = if vertical { 11.0_f32 } else { 3.0_f32 };
            assert_eq!(machine.regs[12], expected.to_bits(), "lane {lane}");
        }
    }
}

/// `FSWZADD` with `.NDV`, a rounding mode or `.FTZ` is refused (`Take(77, 4) == 0`), each the
/// only fault.
#[test]
fn fswzadd_refuses_its_modes() {
    let good = fswzadd(r(0), r(1), r(2), FSWZADD_OPERATIONS[0].0);
    translated(&[Instr::new(Op::FSwzAdd(good))]);
    let modes = [
        OpFSwzAdd { ndv: true, ..good },
        OpFSwzAdd { ftz: true, ..good },
        OpFSwzAdd {
            rnd: RoundMode::NegInf,
            ..good
        },
        OpFSwzAdd {
            rnd: RoundMode::PosInf,
            ..good
        },
        OpFSwzAdd {
            rnd: RoundMode::Zero,
            ..good
        },
    ];
    for op in modes {
        assert_refused(Op::FSwzAdd(op), FloatUnsupported::SwizzledAddMode);
    }
    let function = translated(&[Instr::new(Op::FSwzAdd(good))]);
    assert_eq!(
        flags_of(&function, Opcode::FSwizzleAdd),
        [Flags::Fp(FpControl {
            no_contraction: false,
            rounding: FpRounding::Rn,
            fmz_mode: FmzMode::None,
        })]
    );
}
