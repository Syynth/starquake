//! The packed-half instructions: values per lane, every swizzle and modifier, PR #4588's
//! synthetic cases, and the comparison tables.

use super::*;

/// A constant buffer holding `words`, little-endian.
fn cbuf_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// A packed-half constant-buffer source with a swizzle and modifiers.
fn hcbuf(bank: u8, offset: u16, swizzle: HalfSwizzle, abs: bool, neg: bool) -> HalfSrc {
    HalfSrc {
        src: AluSrc::CBuf {
            cbuf: CBufRef {
                buf: CBuf::Binding(bank),
                offset,
            },
            abs,
            neg,
        },
        swizzle,
    }
}

/// `HADD2` adds lane by lane, for every swizzle (both halves, the low half in both lanes, the
/// high half in both lanes), the absolute value and the negation alone and together (absolute
/// value first), and every source kind: a register, `RZ`, an immediate, and constant-buffer
/// words, aligned and not, with a swizzle and a negation.
#[test]
fn hadd2_adds_each_lane() {
    let mut machine = Machine::new();
    machine.regs[1] = pack(H_ONE, H_TWO);
    machine.regs[2] = pack(H_HALF, H_FOUR);
    machine.regs[3] = pack(H_NEG_ONE, H_NEG_TWO);
    machine.regs[4] = pack(H_FOUR, H_FOUR);
    machine
        .cbufs
        .insert(2, cbuf_bytes(&[pack(H_TWO, H_ONE), pack(H_THREE, H_FOUR)]));
    let cases = [
        (h(1), h(2), pack(H_ONE_HALF, H_SIX)),
        (
            h(1),
            hs(2, HalfSwizzle::H0H0, false, false),
            pack(H_ONE_HALF, H_TWO_HALF),
        ),
        (
            hs(1, HalfSwizzle::H1H1, false, false),
            h(2),
            pack(H_TWO_HALF, H_SIX),
        ),
        // |(-1, -2)| + -(0.5, 4) = (0.5, -2).
        (
            hs(3, HalfSwizzle::H1H0, true, false),
            hs(2, HalfSwizzle::H1H0, false, true),
            pack(H_HALF, H_NEG_TWO),
        ),
        // -|(-1, -2)| + (4, 4) = (3, 2); negating first would give (5, 6).
        (
            hs(3, HalfSwizzle::H1H0, true, true),
            h(4),
            pack(H_THREE, H_TWO),
        ),
        (h(255), h(2), pack(H_HALF, H_FOUR)),
        (h(1), himm(pack(H_ONE, H_TWO)), pack(H_TWO, H_FOUR)),
        (
            h(1),
            hcbuf(2, 0, HalfSwizzle::H1H0, false, false),
            pack(H_THREE, H_THREE),
        ),
        // The constant buffer's high half, negated, in both lanes: (1, 2) + (-1, -1).
        (
            h(1),
            hcbuf(2, 0, HalfSwizzle::H1H1, false, true),
            pack(0, H_ONE),
        ),
        // Bytes 2 to 5: the high half of word 0 and the low half of word 1, (1, 3).
        (
            h(1),
            hcbuf(2, 2, HalfSwizzle::H1H0, false, false),
            pack(H_TWO, H_FIVE),
        ),
    ];
    for (a, b, expected) in cases {
        let op = Op::HAdd2(hadd2(r(0), a, b));
        let mut run = machine.clone();
        run_on(&[Instr::new(op)], &mut run);
        assert_eq!(run.regs[0], expected, "{op}");
    }
}

/// `.SAT` clamps each lane to [0, 1]: 1.25 to 1, -0.5 to 0.
#[test]
fn hadd2_saturates_each_lane() {
    let op = Op::HAdd2(OpHAdd2 {
        saturate: true,
        ..hadd2(r(0), h(1), h(2))
    });
    let machine = run_op(
        op,
        &[(1, pack(H_HALF, H_NEG_ONE)), (2, pack(0x3a00, H_HALF))],
    );
    assert_eq!(machine.regs[0], pack(H_ONE, 0));
}

/// `.FTZ` is carried on the arithmetic (decision 10), and the interpreter honours it: the
/// smallest denormal plus zero is flushed with it and kept without. Contraction is off and the
/// rounding unspecified, as the PR's controls say, for `HADD2`, `HMUL2` and `HFMA2`.
#[test]
fn arithmetic_carries_flushing() {
    for ftz in [false, true] {
        let ops = [
            (
                Op::HAdd2(OpHAdd2 {
                    ftz,
                    ..hadd2(r(0), h(1), h(2))
                }),
                Opcode::FPAdd16,
            ),
            (
                Op::HMul2(OpHMul2 {
                    ftz,
                    ..hmul2(r(0), h(1), h(3))
                }),
                Opcode::FPMul16,
            ),
            (
                Op::HFma2(OpHFma2 {
                    ftz,
                    ..hfma2(r(0), h(1), h(3), h(2))
                }),
                Opcode::FPFma16,
            ),
        ];
        for (op, opcode) in ops {
            let function = translated(&[Instr::new(op)]);
            let fmz_mode = if ftz { FmzMode::Ftz } else { FmzMode::None };
            let expected = Flags::Fp(control(true, FpRounding::DontCare, fmz_mode));
            assert_eq!(flags_of(&function, opcode), [expected, expected], "{op}");
            let mut machine = Machine::new();
            machine.regs[1] = pack(H_DENORM, H_DENORM);
            machine.regs[2] = 0;
            machine.regs[3] = pack(H_ONE, H_ONE);
            machine.run(&function).expect("runs");
            let kept = pack(H_DENORM, H_DENORM);
            assert_eq!(machine.regs[0], if ftz { 0 } else { kept }, "{op}");
        }
    }
}

/// `HADD2.F32`: `b`'s selected half with its modifiers, widened to a 32-bit float to nearest
/// without flushing; `a` must be `-RZ`.
#[test]
fn hadd2_f32_widens_one_half() {
    let neg_rz = hs(255, HalfSwizzle::H1H0, false, true);
    let cases = [
        (HalfSwizzle::H1H1, false, false, 1.5_f32),
        (HalfSwizzle::H1H1, false, true, -1.5),
        (HalfSwizzle::H0H0, true, false, 0.5),
        (HalfSwizzle::H0H0, true, true, -0.5),
        (HalfSwizzle::H0H0, false, false, -0.5),
    ];
    for (swizzle, abs, neg, expected) in cases {
        let op = Op::HAdd2(OpHAdd2 {
            f32: true,
            ..hadd2(r(0), neg_rz, hs(2, swizzle, abs, neg))
        });
        let function = translated(&[Instr::new(op)]);
        assert_eq!(
            flags_of(&function, Opcode::ConvertF32F16),
            [Flags::Fp(control(false, FpRounding::Rn, FmzMode::None))]
        );
        let mut machine = Machine::new();
        machine.regs[2] = pack(0xb800, H_ONE_HALF);
        machine.run(&function).expect("runs");
        assert_eq!(machine.regs[0], expected.to_bits(), "{op}");
    }
}

/// `HADD2.F32` takes only `-RZ` as `a`: the native translator ignores `a`, which is right only
/// for `-RZ`, so any other `a` is refused (the lowering's check; a listed divergence).
#[test]
fn hadd2_f32_needs_negated_rz() {
    let b = hs(2, HalfSwizzle::H0H0, false, false);
    let f32_add = |a: HalfSrc| {
        Op::HAdd2(OpHAdd2 {
            f32: true,
            ..hadd2(r(0), a, b)
        })
    };
    translated(&[Instr::new(f32_add(hs(255, HalfSwizzle::H1H0, false, true)))]);
    for a in [
        hs(1, HalfSwizzle::H1H0, false, true),
        h(255),
        hs(255, HalfSwizzle::H1H0, true, true),
        hs(255, HalfSwizzle::H1H1, false, true),
    ] {
        assert_refused(f32_add(a), HalfDoubleUnsupported::HalfAddFloatOperand);
    }
}

/// `HMUL2` multiplies lane by lane, with `b` a register, an immediate (form 4) or a constant
/// buffer (form 5), and saturates with `.SAT`.
#[test]
fn hmul2_multiplies_each_lane() {
    let mut machine = Machine::new();
    machine.regs[1] = pack(H_ONE_HALF, H_NEG_ONE);
    machine.regs[2] = pack(H_TWO, H_FOUR);
    machine
        .cbufs
        .insert(5, cbuf_bytes(&[0, pack(H_FOUR, H_TWO)]));
    let cases = [
        (hmul2(r(0), h(1), h(2)), pack(H_THREE, H_NEG_FOUR)),
        (
            hmul2(r(0), h(1), himm(pack(H_TWO, H_TWO))),
            pack(H_THREE, H_NEG_TWO),
        ),
        (
            hmul2(r(0), h(1), hcbuf(5, 4, HalfSwizzle::H1H0, false, false)),
            pack(H_SIX, H_NEG_TWO),
        ),
        (
            OpHMul2 {
                saturate: true,
                ..hmul2(r(0), h(1), h(2))
            },
            pack(H_ONE, 0),
        ),
        (
            hmul2(
                r(0),
                hs(1, HalfSwizzle::H1H0, true, false),
                hs(2, HalfSwizzle::H1H1, false, true),
            ),
            pack(0xc600, H_NEG_FOUR),
        ),
    ];
    for (op, expected) in cases {
        let op = Op::HMul2(op);
        let mut run = machine.clone();
        run_on(&[Instr::new(op)], &mut run);
        assert_eq!(run.regs[0], expected, "{op}");
    }
}

/// `HFMA2` fuses lane by lane, in each of its five forms (`b` and `c` registers, `c` or `b`
/// an immediate, `c` or `b` a constant buffer), with modifiers on every source and `.H0_NH1`,
/// which negates `b`'s high lane after its modifiers.
#[test]
fn hfma2_fuses_each_lane() {
    let mut machine = Machine::new();
    machine.regs[1] = pack(H_ONE, H_TWO);
    machine.regs[2] = pack(H_TWO, H_TWO);
    machine.regs[3] = pack(H_ONE, H_NEG_ONE);
    machine.regs[4] = pack(H_NEG_ONE, H_NEG_TWO);
    machine
        .cbufs
        .insert(7, cbuf_bytes(&[pack(H_TWO, H_TWO), pack(H_ONE, H_NEG_ONE)]));
    let neg_high = |op: OpHFma2| OpHFma2 {
        neg_high_b: true,
        ..op
    };
    let cases = [
        // (1, 2) * (2, 2) + (1, -1) = (3, 3).
        (hfma2(r(0), h(1), h(2), h(3)), pack(H_THREE, H_THREE)),
        (
            hfma2(r(0), h(1), h(2), himm(pack(H_ONE, H_NEG_ONE))),
            pack(H_THREE, H_THREE),
        ),
        (
            hfma2(r(0), h(1), himm(pack(H_TWO, H_TWO)), h(3)),
            pack(H_THREE, H_THREE),
        ),
        (
            hfma2(
                r(0),
                h(1),
                h(2),
                hcbuf(7, 4, HalfSwizzle::H1H0, false, false),
            ),
            pack(H_THREE, H_THREE),
        ),
        (
            hfma2(
                r(0),
                h(1),
                hcbuf(7, 0, HalfSwizzle::H1H0, false, false),
                h(3),
            ),
            pack(H_THREE, H_THREE),
        ),
        // (1, 2) * (2, -2) + (1, -1) = (3, -5).
        (
            neg_high(hfma2(r(0), h(1), h(2), h(3))),
            pack(H_THREE, 0xc500),
        ),
        // -(2, 2) is (-2, -2), then the high lane negated: (1, 2) * (-2, 2) + (1, -1) = (-1, 3).
        (
            neg_high(hfma2(
                r(0),
                h(1),
                hs(2, HalfSwizzle::H1H0, false, true),
                h(3),
            )),
            pack(H_NEG_ONE, H_THREE),
        ),
        // |(-1, -2)| * (2, 2) - (1, -1) = (1, 5).
        (
            hfma2(
                r(0),
                hs(4, HalfSwizzle::H1H0, true, false),
                h(2),
                hs(3, HalfSwizzle::H1H0, false, true),
            ),
            pack(H_ONE, H_FIVE),
        ),
        // (2, 2) * |(-1, -1)| + |(1, 1)|, the swizzles picking halves: (3, 3).
        (
            hfma2(
                r(0),
                hs(2, HalfSwizzle::H1H1, false, false),
                hs(4, HalfSwizzle::H0H0, true, false),
                hs(3, HalfSwizzle::H0H0, true, false),
            ),
            pack(H_THREE, H_THREE),
        ),
        (
            OpHFma2 {
                saturate: true,
                ..hfma2(r(0), h(1), h(2), h(3))
            },
            pack(H_ONE, H_ONE),
        ),
    ];
    for (op, expected) in cases {
        let op = Op::HFma2(op);
        let mut run = machine.clone();
        run_on(&[Instr::new(op)], &mut run);
        assert_eq!(run.regs[0], expected, "{op}");
    }
}

/// `HFMA2.RELU` clamps each lane as the lowering does (`lower_to_maxwell.h` case `0x231`, ruling
/// 7): at or below zero to `+0.0`, `-0.0` and a negative denormal included, a NaN of either sign
/// to `RELU_NAN` narrowed (the IR's canonical half NaN, `0x7fff`, as the interpreter models a
/// conversion's NaN result), and everything else kept, positive infinity and a positive
/// denormal included. The native translator keeps `-0.0` (module docs, "Divergences").
#[test]
fn relu_clamps_as_the_lowering_does() {
    let neg_zero = hs(255, HalfSwizzle::H1H0, false, true);
    let op = Op::HFma2(OpHFma2 {
        relu: Some(PredSrc::TRUE),
        ..hfma2(r(0), h(1), himm(pack(H_ONE, H_ONE)), neg_zero)
    });
    let cases = [
        (H_THREE, H_THREE),
        (H_NEG_TWO, 0),
        (H_INF, H_INF),
        (H_NEG_INF, 0),
        (0x7e01, H_CANONICAL_NAN),
        (0xfe01, H_CANONICAL_NAN),
        (H_NZERO, 0),
        (H_DENORM, H_DENORM),
        (0x8001, 0),
    ];
    for (input, expected) in cases {
        let machine = run_op(op, &[(1, pack(input, input))]);
        assert_eq!(machine.regs[0], pack(expected, expected), "{input:#x}");
    }
}

/// PR #4588's `HFMA2.RELU` cases (`native_shader.cpp`, the `ReLU` case "preserves fusion
/// values above one and canonical NaNs"), from its words: `-R13.H1_H1 * R13.H1_H1 +
/// R16.H0_H0`, then the fused residual with `R14`. One row differs from the PR's expectation,
/// which is its lowering's: `-0.0` stays `-0.0`, not `+0.0` (module docs, owner question).
#[test]
fn pr_relu_cases() {
    let instr = pr_word([0x0d12_7231, 0x3000_000d, 0x0384_8d10, 0]);
    assert_eq!(
        instr,
        Instr::new(Op::HFma2(OpHFma2 {
            relu: Some(PredSrc::TRUE),
            ..hfma2(
                r(18),
                hs(13, HalfSwizzle::H1H1, false, true),
                hs(13, HalfSwizzle::H1H1, false, false),
                hs(16, HalfSwizzle::H0H0, false, false),
            )
        }))
    );
    let cases = [
        (0x3c00, 0x4400, 0x4200),
        (0x4000, 0x3c00, 0x0000),
        (0x3c00, 0x3c00, 0x0000),
        (0x0000, 0x0001, 0x0001),
        (0x7c00, 0x0000, 0x0000),
        (0x0000, 0x7c00, 0x7c00),
        (0x7e01, 0x3c00, H_CANONICAL_NAN),
        (0xfe01, 0x3c00, H_CANONICAL_NAN),
        // The PR's own row `{0x8000, 0x8000, 0x0000}`, the lowering's result (ruling 7); the
        // native translator keeps `-0.0`.
        (0x8000, 0x8000, 0x0000),
    ];
    for (a, c, expected) in cases {
        let machine = run_op(instr.op, &[(13, (a << 16) | 0x4400), (16, 0xc000_0000 | c)]);
        assert_eq!(machine.regs[18], pack(expected, expected), "{a:#x} {c:#x}");
    }
    let fused = pr_word([0x0d12_7231, 0x3000_000e, 0x0384_8d10, 0]);
    let machine = run_op(
        fused.op,
        &[(13, 0x3c01_3c01), (14, 0x3bfe_3bfe), (16, 0x3c00_3c00)],
    );
    // 1 - (1 + 2^-10)(1 - 2^-10) = 2^-20, a denormal half, kept only by fusing.
    assert_eq!(machine.regs[18], 0x0010_0010);
}

/// PR #4588's `HFMA2.RELU` with an immediate (`native_shader.cpp`, "accepts packed immediates
/// without losing input aliases"): `R2.H0_H0 * 1 - R5.H1_H1` into `R2`, with the PR's expected
/// values.
#[test]
fn pr_relu_immediate_cases() {
    let instr = pr_word([0x0202_7831, 0x3c00_3c00, 0x0396_8805, 0]);
    for (source, addend, expected) in [
        (0x4000, 0x3c00, 0x3c00),
        (0x3c00, 0x4400, 0x0000),
        (0x4400, 0x3c00, 0x4200),
        (0x7e01, 0x3c00, H_CANONICAL_NAN),
    ] {
        let machine = run_op(
            instr.op,
            &[(2, source | 0xc400_0000), (5, (addend << 16) | 0x4800)],
        );
        assert_eq!(machine.regs[2], pack(expected, expected), "{source:#x}");
    }
}

/// PR #4588's absolute-value case (`native_shader.cpp`, "absolute source clears each half sign
/// before broadcast"): `|R26| * R8.H0_H0 + R13.H0_H0`, with `a` as stored and broadcast.
#[test]
fn pr_absolute_source_cases() {
    for (word, expected) in [(0x0004_020d, 0x4700_4500), (0x0004_0a0d, 0x4500_4500)] {
        let instr = pr_word([0x1a08_7231, 0x2000_0008, word, 0]);
        let machine = run_op(
            instr.op,
            &[(26, 0xc400_c000), (8, 0x4000_3c00), (13, 0x4400_4200)],
        );
        assert_eq!(machine.regs[8], expected, "{instr}");
    }
}

/// PR #4588's conditional `ReLU` (`native_shader.cpp`, "Native conditional half `ReLU` keeps
/// unclamped results when disabled"): `R8 * 1 + R11` into `R4`, `R8` or `R11`, with bits 87 to
/// 90 as the predicate (three bits and its negation), translated as the lowering's case
/// `0x231` does (ruling 7): the clamped lanes where the predicate holds, the unclamped result
/// elsewhere. The PR's cases, with its predicates `{true, false, true, false, false, true,
/// false}` and the named one set to each value; `-1 + 1` and `inf` keep their value either way,
/// `-2 + 1` and `-inf` differ. The predicates are not written.
#[test]
fn relu_under_a_predicate_keeps_the_clamp_where_it_holds() {
    // (`R8`, clamped, unclamped): `R8` plus one in each lane.
    let values = [
        (0xbc00_3c00, 0x0000_4000, 0x0000_4000),
        (0x3c00_bc00, 0x4000_0000, 0x4000_0000),
        (0xc000_c000, 0x0000_0000, pack(H_NEG_ONE, H_NEG_ONE)),
        (0x7c00_fc00, 0x7c00_0000, pack(H_NEG_INF, H_INF)),
    ];
    for condition in [0_u32, 1, 6, 7, 8, 9, 14, 15] {
        for destination in [4_u32, 8, 11] {
            let instr = pr_word([
                (8 << 24) | (destination << 16) | 0x7831,
                0x3c00_3c00,
                0x800b | (condition << 23),
                0,
            ]);
            let index = u8::try_from(condition & 7).expect("three bits");
            let negated = condition & 8 != 0;
            assert_eq!(
                instr.op,
                Op::HFma2(OpHFma2 {
                    relu: Some(PredSrc {
                        reg: Pred::new(index).expect("a predicate"),
                        negated,
                    }),
                    ..hfma2(
                        r(u8::try_from(destination).expect("a register")),
                        h(8),
                        himm(0x3c00_3c00),
                        h(11),
                    )
                })
            );
            for enabled in [false, true] {
                for (value, clamped, unclamped) in values {
                    let mut machine = Machine::new();
                    machine.regs[8] = value;
                    machine.regs[11] = 0x3c00_3c00;
                    machine.preds[..7]
                        .copy_from_slice(&[true, false, true, false, false, true, false]);
                    if index != 7 {
                        machine.preds[usize::from(index)] = enabled;
                    }
                    let preds = machine.preds;
                    let clamp = (index == 7 || enabled) != negated;
                    run_on(&[instr], &mut machine);
                    let expected = if clamp { clamped } else { unclamped };
                    let at = usize::try_from(destination).expect("an index");
                    assert_eq!(machine.regs[at], expected, "{instr} {enabled} {value:#x}");
                    assert_eq!(machine.preds, preds, "{instr}");
                }
            }
        }
    }
}

/// `HFMA2.RELU` under `P0`, `!P0` and `!PT` builds the lowering's final select of the clamped
/// word over the unclamped one, reading the predicate with its negation; under `PT` there is no
/// select. A lane of `-2` (clamped to `+0.0`, unclamped `-2`) shows which word is kept.
#[test]
fn relu_selects_by_its_predicate() {
    let fma = |relu: PredSrc| {
        Op::HFma2(OpHFma2 {
            relu: Some(relu),
            ..hfma2(r(0), h(1), himm(pack(H_ONE, H_ONE)), h(255))
        })
    };
    for (relu, p0, clamped) in [
        (pred(0, false), true, true),
        (pred(0, false), false, false),
        (pred(0, true), true, false),
        (pred(0, true), false, true),
        (NOT_PT, false, false),
        (NOT_PT, true, false),
        (PredSrc::TRUE, false, true),
    ] {
        let mut machine = Machine::new();
        machine.regs[1] = pack(H_NEG_TWO, H_THREE);
        machine.preds[0] = p0;
        run_on(&[Instr::new(fma(relu))], &mut machine);
        let expected = if clamped {
            pack(0, H_THREE)
        } else {
            pack(H_NEG_TWO, H_THREE)
        };
        assert_eq!(machine.regs[0], expected, "{relu} {p0}");
        let selects = opcodes(&translated(&[Instr::new(fma(relu))]))
            .into_iter()
            .filter(|&opcode| opcode == Opcode::SelectU32)
            .count();
        assert_eq!(selects, usize::from(relu != PredSrc::TRUE), "{relu}");
    }
}

/// PR #4588's lane-order case (`native_shader.cpp`, "Native half packing and arithmetic
/// preserve lane order numerically"), its half instructions from their words in order, with
/// `R2` starting as the packed pair (1, 2) its `F2FP` would leave.
#[test]
fn pr_lane_order_case() {
    let words = [
        [0x0202_7232, 0x0000_0002, 0x0000_0000, 0], // HMUL2 R2, R2, R2
        [0xff03_7230, 0x2000_0002, 0x0000_4100, 0], // low half to F32 R3
        [0xff04_7230, 0x3000_0002, 0x0000_4100, 0], // high half to F32 R4
        [0x0206_7230, 0x2000_0002, 0x0000_0800, 0], // two low halves added
        [0x0207_7231, 0x2000_0002, 0x0004_0802, 0], // fused 1 * 1 + 1
        [0x0208_7240, 0x3000_0002, 0x0380_0800, 0], // min of low and high
        [0x0209_7840, 0x3c00_3c00, 0x0780_0c00, 0], // max of high and one
        [0xff0c_7230, 0xa000_0008, 0x0000_4100, 0], // negated low half to F32
        [0x020d_7230, 0x3000_0002, 0x0000_0f00, 0], // -|high| + high
        [0x020e_7431, 0x3c00_3c00, 0x0040_0802, 0], // (1, 1) * (1, -4) + (1, 1)
    ];
    let program: Vec<Instr> = words.into_iter().map(pr_word).collect();
    let mut machine = Machine::new();
    machine.regs[2] = pack(H_ONE, H_TWO);
    run_on(&program, &mut machine);
    let expected: [(u8, u32); 10] = [
        (2, pack(H_ONE, H_FOUR)),
        (3, 1.0_f32.to_bits()),
        (4, 4.0_f32.to_bits()),
        (6, pack(H_TWO, H_TWO)),
        (7, pack(H_TWO, H_TWO)),
        (8, pack(H_ONE, H_ONE)),
        (9, pack(H_FOUR, H_FOUR)),
        (12, (-1.0_f32).to_bits()),
        (13, 0),
        (14, pack(H_TWO, H_NEG_THREE)),
    ];
    for (index, bits) in expected {
        assert_eq!(machine.regs[usize::from(index)], bits, "r{index}");
    }
}

/// `HMNMX2` picks the minimum of each lane where the predicate (read with its negation) holds
/// and the maximum elsewhere, with `b` a register, an immediate (form 4) or a constant buffer
/// (form 5), and modifiers on the sources.
#[test]
fn hmnmx2_picks_per_lane() {
    let mut machine = Machine::new();
    machine.regs[1] = pack(H_ONE, H_FOUR);
    machine.regs[2] = pack(H_TWO, H_THREE);
    machine.preds[1] = true;
    machine.preds[2] = false;
    machine.cbufs.insert(3, cbuf_bytes(&[pack(H_TWO, H_THREE)]));
    let min = pack(H_ONE, H_THREE);
    let max = pack(H_TWO, H_FOUR);
    let cases = [
        (hmnmx2(r(0), h(1), h(2), pred(1, false)), min),
        (hmnmx2(r(0), h(1), h(2), pred(2, false)), max),
        (hmnmx2(r(0), h(1), h(2), pred(1, true)), max),
        (hmnmx2(r(0), h(1), h(2), pred(2, true)), min),
        (hmnmx2(r(0), h(1), h(2), PredSrc::TRUE), min),
        (hmnmx2(r(0), h(1), h(2), NOT_PT), max),
        (
            hmnmx2(r(0), h(1), himm(pack(H_TWO, H_THREE)), PredSrc::TRUE),
            min,
        ),
        (
            hmnmx2(
                r(0),
                h(1),
                hcbuf(3, 0, HalfSwizzle::H1H0, false, false),
                NOT_PT,
            ),
            max,
        ),
        // min(-(1, 4), (2, 3)) = (-1, -4); min((1, 4), |(2, 2)|) = (1, 2).
        (
            hmnmx2(
                r(0),
                hs(1, HalfSwizzle::H1H0, false, true),
                h(2),
                PredSrc::TRUE,
            ),
            pack(H_NEG_ONE, H_NEG_FOUR),
        ),
        (
            hmnmx2(
                r(0),
                h(1),
                hs(2, HalfSwizzle::H0H0, true, false),
                PredSrc::TRUE,
            ),
            pack(H_ONE, H_TWO),
        ),
    ];
    for (op, expected) in cases {
        let op = Op::HMnMx2(op);
        let mut run = machine.clone();
        run_on(&[Instr::new(op)], &mut run);
        assert_eq!(run.regs[0], expected, "{op}");
    }
}

/// Each comparison's result on six pairs: (1, 2), (NaN, 1), (2, 2), (3, 2), (1, NaN) and
/// (NaN, NaN), written out from the comparison's meaning (`.u` true when unordered, `.num` when
/// both are numbers, `.nan` when either is not). The NaN only in `b` catches a `.num` or `.nan`
/// that tests `a` twice.
pub(super) const COMPARISONS: [(FloatCmpOp, [bool; 6]); 16] = [
    (
        FloatCmpOp::False,
        [false, false, false, false, false, false],
    ),
    (FloatCmpOp::Lt, [true, false, false, false, false, false]),
    (FloatCmpOp::Eq, [false, false, true, false, false, false]),
    (FloatCmpOp::Le, [true, false, true, false, false, false]),
    (FloatCmpOp::Gt, [false, false, false, true, false, false]),
    (FloatCmpOp::Ne, [true, false, false, true, false, false]),
    (FloatCmpOp::Ge, [false, false, true, true, false, false]),
    (FloatCmpOp::Num, [true, false, true, true, false, false]),
    (FloatCmpOp::Nan, [false, true, false, false, true, true]),
    (FloatCmpOp::LtU, [true, true, false, false, true, true]),
    (FloatCmpOp::EqU, [false, true, true, false, true, true]),
    (FloatCmpOp::LeU, [true, true, true, false, true, true]),
    (FloatCmpOp::GtU, [false, true, false, true, true, true]),
    (FloatCmpOp::NeU, [true, true, false, true, true, true]),
    (FloatCmpOp::GeU, [false, true, true, true, true, true]),
    (FloatCmpOp::True, [true, true, true, true, true, true]),
];

/// [`COMPARISONS`]' six pairs as three runs of a packed-half comparison, `a`'s word and `b`'s,
/// each run's low lane one pair and its high lane the next.
const COMPARED_WORDS: [(u32, u32); 3] = [
    (pack(H_ONE, H_QNAN), pack(H_TWO, H_ONE)),
    (pack(H_TWO, H_THREE), pack(H_TWO, H_TWO)),
    (pack(H_ONE, H_QNAN), pack(H_QNAN, H_QNAN)),
];

/// `HSETP2` compares each lane, every comparison: the low lane's result to the first
/// predicate, the high lane's to the second. Each run of [`COMPARED_WORDS`] holds two of
/// [`COMPARISONS`]' pairs as its lanes.
#[test]
fn hsetp2_compares_each_lane() {
    let ops: Vec<FloatCmpOp> = COMPARISONS.iter().map(|&(cmp, _)| cmp).collect();
    assert_eq!(ops, FloatCmpOp::ALL);
    for (cmp, results) in COMPARISONS {
        let op = Op::HSetP2(hsetp2(p(1), p(2), cmp, h(3), h(4)));
        for (run, (a, b)) in COMPARED_WORDS.into_iter().enumerate() {
            let (low, high) = (results[2 * run], results[2 * run + 1]);
            let mut machine = Machine::new();
            machine.preds[1] = !low;
            machine.preds[2] = !high;
            machine.regs[3] = a;
            machine.regs[4] = b;
            run_on(&[Instr::new(op)], &mut machine);
            assert_eq!((machine.preds[1], machine.preds[2]), (low, high), "{op}");
        }
    }
}

/// `HSETP2` combines each lane with the accumulator by its operation; `.H_AND` writes both
/// lanes' conjunction and its negation; a `PT` destination is not written; `.FTZ` is carried
/// on the comparisons.
#[test]
fn hsetp2_combines_and_writes() {
    let base = || hsetp2(p(1), p(2), FloatCmpOp::Lt, h(3), h(4));
    // Lt of (1, NaN) against (2, 1): (true, false).
    let run = |op: OpHSetP2, accum: bool| {
        let mut machine = Machine::new();
        machine.regs[3] = pack(H_ONE, H_QNAN);
        machine.regs[4] = pack(H_TWO, H_ONE);
        machine.preds[4] = accum;
        machine.preds[1] = false;
        machine.preds[2] = true;
        run_on(&[Instr::new(Op::HSetP2(op))], &mut machine);
        (machine.preds[1], machine.preds[2])
    };
    let with = |set_op, accum| OpHSetP2 {
        set_op,
        accum,
        ..base()
    };
    assert_eq!(
        run(with(PredSetOp::And, pred(4, false)), true),
        (true, false)
    );
    assert_eq!(
        run(with(PredSetOp::And, pred(4, false)), false),
        (false, false)
    );
    assert_eq!(run(with(PredSetOp::Or, pred(4, false)), true), (true, true));
    assert_eq!(
        run(with(PredSetOp::Xor, pred(4, false)), true),
        (false, true)
    );
    assert_eq!(
        run(with(PredSetOp::Xor, pred(4, true)), true),
        (true, false)
    );
    let horizontal = OpHSetP2 {
        horizontal: true,
        ..base()
    };
    assert_eq!(run(horizontal, true), (false, true));
    let both = OpHSetP2 {
        cmp: FloatCmpOp::LtU,
        ..horizontal
    };
    assert_eq!(run(both, true), (true, false));
    let no_first = OpHSetP2 {
        dst: Pred::PT,
        cmp: FloatCmpOp::LtU,
        ..base()
    };
    assert_eq!(run(no_first, true), (false, true));
    let function = translated(&[Instr::new(Op::HSetP2(no_first))]);
    assert_eq!(
        opcodes(&function)
            .iter()
            .filter(|&&opcode| opcode == Opcode::SetPred)
            .count(),
        1
    );
    for ftz in [false, true] {
        let function = translated(&[Instr::new(Op::HSetP2(OpHSetP2 { ftz, ..base() }))]);
        let fmz_mode = if ftz { FmzMode::Ftz } else { FmzMode::None };
        let expected = Flags::Fp(control(false, FpRounding::DontCare, fmz_mode));
        assert_eq!(
            flags_of(&function, Opcode::FPOrdLessThan16),
            [expected, expected]
        );
    }
}

/// `HSET2` writes each lane's combined result as `1.0` (`.BF`) or an all-ones lane, and zero
/// where false, for every comparison on every pair of [`COMPARISONS`]; the accumulator and
/// `.FTZ` work as in `HSETP2`.
#[test]
fn hset2_writes_each_lane() {
    for (bf, truth) in [(true, HSET2_TRUE_HALF), (false, HSET2_TRUE_MASK)] {
        for (cmp, results) in COMPARISONS {
            let op = Op::HSet2(OpHSet2 {
                bf,
                ..hset2(r(0), cmp, h(3), h(4))
            });
            for (run, (a, b)) in COMPARED_WORDS.into_iter().enumerate() {
                let machine = run_op(op, &[(3, a), (4, b)]);
                let lane = |on: bool| if on { truth } else { 0 };
                let expected = pack(lane(results[2 * run]), lane(results[2 * run + 1]));
                assert_eq!(machine.regs[0], expected, "{op} {run}");
            }
        }
    }
    let op = Op::HSet2(OpHSet2 {
        set_op: PredSetOp::Or,
        accum: pred(5, false),
        bf: true,
        ..hset2(r(0), FloatCmpOp::False, h(3), h(4))
    });
    for accum in [false, true] {
        let mut machine = Machine::new();
        machine.preds[5] = accum;
        run_on(&[Instr::new(op)], &mut machine);
        let lane = if accum { H_ONE } else { 0 };
        assert_eq!(machine.regs[0], pack(lane, lane));
    }
    let function = translated(&[Instr::new(Op::HSet2(OpHSet2 {
        ftz: true,
        ..hset2(r(0), FloatCmpOp::Gt, h(3), h(4))
    }))]);
    let expected = Flags::Fp(control(false, FpRounding::DontCare, FmzMode::Ftz));
    assert_eq!(
        flags_of(&function, Opcode::FPOrdGreaterThan16),
        [expected, expected]
    );
}

/// A 32-bit float's bits, for a source read with swizzle 1.
fn f32_bits(value: f32) -> u32 {
    value.to_bits()
}

/// Swizzle 1 reads the whole word as one 32-bit float in both lanes, and `HADD2`, `HMUL2`,
/// `HFMA2` and `HSET2` are promoted as Eden's Maxwell translators promote them (the lowering's
/// reading, ruling 7): the other sources' half lanes widened, the arithmetic, saturation and
/// comparison in single precision, each lane narrowed back. The source's absolute value and
/// negation are the 32-bit ones, in any operand position and from a register or a constant
/// buffer. `1 + 2^-11`, a float no half holds, plus `2^-11` gives `1 + 2^-10` (`0x3c01`) only
/// when added before narrowing; narrowing the source first would give `1.0`.
#[test]
fn swizzle_1_promotes_to_single_precision() {
    let wide = |index: u8, abs: bool, neg: bool| hs(index, HalfSwizzle::F32, abs, neg);
    let mut machine = Machine::new();
    machine.regs[1] = f32_bits(1.5);
    machine.regs[2] = pack(H_ONE, H_TWO);
    machine.regs[3] = f32_bits(-2.0);
    machine.regs[4] = 0x3f80_1000; // 1 + 2^-11
    machine.regs[5] = pack(0x1000, 0x1000); // 2^-11 in both lanes
    machine.cbufs.insert(2, cbuf_bytes(&[f32_bits(3.0)]));
    let sat = |op: OpHFma2| OpHFma2 {
        saturate: true,
        ..op
    };
    let cases = [
        // 1.5 + (1, 2) = (2.5, 3.5).
        (
            Op::HAdd2(hadd2(r(0), wide(1, false, false), h(2))),
            pack(H_TWO_HALF, 0x4300),
        ),
        // (1, 2) - 1.5 = (-0.5, 0.5).
        (
            Op::HAdd2(hadd2(r(0), h(2), wide(1, false, true))),
            pack(0xb800, H_HALF),
        ),
        (
            Op::HAdd2(hadd2(r(0), wide(4, false, false), h(5))),
            pack(0x3c01, 0x3c01),
        ),
        // (1, 2) * 3.0 from the constant buffer = (3, 6).
        (
            Op::HMul2(hmul2(
                r(0),
                h(2),
                hcbuf(2, 0, HalfSwizzle::F32, false, false),
            )),
            pack(H_THREE, H_SIX),
        ),
        // |-2| * (1, 2) = (2, 4); -|-2| * (1, 2) = (-2, -4).
        (
            Op::HMul2(hmul2(r(0), wide(3, true, false), h(2))),
            pack(H_TWO, H_FOUR),
        ),
        (
            Op::HMul2(hmul2(r(0), wide(3, true, true), h(2))),
            pack(H_NEG_TWO, H_NEG_FOUR),
        ),
        // (1, 2) * (1, 2) + 1.5 = (2.5, 5.5).
        (
            Op::HFma2(hfma2(r(0), h(2), h(2), wide(1, false, false))),
            pack(H_TWO_HALF, 0x4580),
        ),
        // 1.5 * (1, 2) + 0 = (1.5, 3), saturated in single precision to (1, 1).
        (
            Op::HFma2(sat(hfma2(r(0), wide(1, false, false), h(2), h(255)))),
            pack(H_ONE, H_ONE),
        ),
        // -2 * (1, 2) + 0 = (-2, -4), clamped by `.RELU` after the narrowing to (+0, +0).
        (
            Op::HFma2(OpHFma2 {
                relu: Some(PredSrc::TRUE),
                ..hfma2(r(0), wide(3, false, false), h(2), h(255))
            }),
            0,
        ),
        // 1.5 > (1, 2) = (true, false).
        (
            Op::HSet2(OpHSet2 {
                bf: true,
                ..hset2(r(0), FloatCmpOp::Gt, wide(1, false, false), h(2))
            }),
            pack(H_ONE, 0),
        ),
        (
            Op::HSet2(OpHSet2 {
                bf: true,
                ..hset2(r(0), FloatCmpOp::Gt, h(2), wide(1, false, false))
            }),
            pack(0, H_ONE),
        ),
    ];
    for (op, expected) in cases {
        let mut run = machine.clone();
        run_on(&[Instr::new(op)], &mut run);
        assert_eq!(run.regs[0], expected, "{op}");
    }
    // The controls carry over to the 32-bit arithmetic, and the narrowing and widening take
    // Eden's default controls.
    for ftz in [false, true] {
        let op = Op::HAdd2(OpHAdd2 {
            ftz,
            ..hadd2(r(0), wide(1, false, false), h(2))
        });
        let function = translated(&[Instr::new(op)]);
        let expected = Flags::Fp(control(true, FpRounding::DontCare, flushing(ftz)));
        assert_eq!(flags_of(&function, Opcode::FPAdd32), [expected, expected]);
        assert_eq!(flags_of(&function, Opcode::FPAdd16), []);
        let plain = Flags::Fp(FpControl::default());
        assert_eq!(flags_of(&function, Opcode::ConvertF32F16), [plain, plain]);
        assert_eq!(flags_of(&function, Opcode::ConvertF16F32), [plain, plain]);
    }
    let function = translated(&[Instr::new(Op::HSet2(hset2(
        r(0),
        FloatCmpOp::Gt,
        wide(1, false, false),
        h(2),
    )))]);
    assert_eq!(flags_of(&function, Opcode::FPOrdGreaterThan32).len(), 2);
    assert_eq!(flags_of(&function, Opcode::FPOrdGreaterThan16), []);
}

/// Swizzle 1 is refused in `HMNMX2` and `HSETP2`, in either operand and from a constant buffer,
/// where neither reference translates it (`HalfLanes`; the lowering's cases `0x240` and
/// `0x434`).
#[test]
fn swizzle_1_is_refused_in_min_max_and_hsetp2() {
    use HalfDoubleUnsupported as U;
    let f = hs(1, HalfSwizzle::F32, false, false);
    let f_cbuf = hcbuf(1, 0, HalfSwizzle::F32, false, false);
    let cases = [
        (Op::HMnMx2(hmnmx2(r(0), f, h(2), PredSrc::TRUE)), 0),
        (Op::HMnMx2(hmnmx2(r(0), h(2), f, PredSrc::TRUE)), 1),
        (Op::HMnMx2(hmnmx2(r(0), h(2), f_cbuf, PredSrc::TRUE)), 1),
        (Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, f, h(2))), 0),
        (Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, h(2), f)), 1),
        (
            Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, h(2), f_cbuf)),
            1,
        ),
    ];
    for (op, index) in cases {
        assert_refused(op, U::HalfAsFloat(index));
    }
}

/// `.FMZ` without `.SAT` on packed halves (no source read as a float) is refused in `HMUL2`
/// and `HFMA2`: Eden's Maxwell translators, which the lowering forwards to, compare a half lane
/// with a single-precision zero there and throw (a listed divergence). With `.FTZ` as well it
/// is refused the same way.
#[test]
fn fmz_on_halves_without_saturation_is_refused() {
    for ftz in [false, true] {
        assert_refused(
            Op::HMul2(OpHMul2 {
                dnz: true,
                ftz,
                ..hmul2(r(0), h(1), h(2))
            }),
            HalfDoubleUnsupported::HalfMultiplyZero,
        );
        assert_refused(
            Op::HFma2(OpHFma2 {
                dnz: true,
                ftz,
                ..hfma2(r(0), h(1), h(2), h(3))
            }),
            HalfDoubleUnsupported::HalfFmaMultiplyZero,
        );
    }
}

/// `.FMZ` is translated where the lowering's path translates it (ruling 7). With `.SAT` only
/// the controls carry it (and win over `.FTZ`, the lowering's precision field), on halves or
/// promoted. Without `.SAT` on a promoted instruction Eden's select runs in single precision:
/// a lane whose `a` or `b` is zero becomes `+0.0` (`HMUL2`) or `c`'s lane (`HFMA2`; a `c` of
/// `-0.0` shows the select, since an FMZ product of `+0` plus `-0.0` would give `+0.0`).
#[test]
fn fmz_with_saturation_or_a_float_source() {
    let fmz = Flags::Fp(control(true, FpRounding::DontCare, FmzMode::Fmz));
    let wide = hs(1, HalfSwizzle::F32, false, false);
    for ftz in [false, true] {
        // Halves with `.SAT`: inf * 0 is +0 under FMZ, 2 * 0.5 is 1.
        let op = Op::HMul2(OpHMul2 {
            dnz: true,
            saturate: true,
            ftz,
            ..hmul2(r(0), h(1), h(2))
        });
        let function = translated(&[Instr::new(op)]);
        assert_eq!(flags_of(&function, Opcode::FPMul16), [fmz, fmz]);
        assert!(!opcodes(&function).contains(&Opcode::FPOrdEqual32));
        let machine = run_op(op, &[(1, pack(H_INF, H_TWO)), (2, pack(0, H_HALF))]);
        assert_eq!(machine.regs[0], pack(0, H_ONE), "{op}");
        let op = Op::HFma2(OpHFma2 {
            dnz: true,
            saturate: true,
            ftz,
            ..hfma2(r(0), h(1), h(2), h(3))
        });
        let function = translated(&[Instr::new(op)]);
        assert_eq!(flags_of(&function, Opcode::FPFma16), [fmz, fmz]);
        // Promoted with `.SAT`: the 32-bit arithmetic carries FMZ, and no select.
        let op = Op::HMul2(OpHMul2 {
            dnz: true,
            saturate: true,
            ftz,
            ..hmul2(r(0), wide, h(2))
        });
        let function = translated(&[Instr::new(op)]);
        assert_eq!(flags_of(&function, Opcode::FPMul32), [fmz, fmz]);
        assert!(!opcodes(&function).contains(&Opcode::FPOrdEqual32));
    }
    // Promoted without `.SAT`: the select, two comparisons with zero per lane.
    let op = Op::HMul2(OpHMul2 {
        dnz: true,
        ..hmul2(r(0), wide, h(2))
    });
    let function = translated(&[Instr::new(op)]);
    assert_eq!(flags_of(&function, Opcode::FPMul32), [fmz, fmz]);
    assert_eq!(
        flags_of(&function, Opcode::FPOrdEqual32),
        [Flags::Fp(FpControl::default()); 4]
    );
    for (a, expected) in [
        (f32_bits(f32::INFINITY), pack(0, H_INF)),
        (f32_bits(f32::NAN), pack(0, H_CANONICAL_NAN)),
    ] {
        let machine = run_op(op, &[(1, a), (2, pack(0, H_ONE))]);
        assert_eq!(machine.regs[0], expected, "{a:#x}");
    }
    let op = Op::HFma2(OpHFma2 {
        dnz: true,
        ..hfma2(r(0), wide, h(2), h(3))
    });
    let machine = run_op(
        op,
        &[
            (1, f32_bits(f32::INFINITY)),
            (2, pack(0, H_ONE)),
            (3, pack(H_NZERO, H_ONE)),
        ],
    );
    assert_eq!(machine.regs[0], pack(H_NZERO, H_INF));
    // `b` zero in the high lane only, `a` zero: both lanes take `c`.
    let machine = run_op(
        op,
        &[(1, 0), (2, pack(H_TWO, H_TWO)), (3, pack(H_THREE, H_NZERO))],
    );
    assert_eq!(machine.regs[0], pack(H_THREE, H_NZERO));
}

/// A promoted `HFMA2` takes the lowering's source checks (case `0x231`), the only reference that
/// translates it: `.H0_NH1` needs a `b` with swizzle 0, no absolute value and a register below
/// `R250` (`R250` and `RZ` refused); an absolute value is refused on `b` or `c`, and on an `a`
/// in `R248` or above. One step inside each edge is taken
/// ([`a_promoted_fma_one_step_inside_the_lowerings_edges`]); without a promoted source the same
/// inputs are translated (the owner-question rows, tested on their own).
#[test]
fn a_promoted_fma_takes_the_lowerings_source_checks() {
    use HalfDoubleUnsupported as U;
    let wide = |index: u8, abs: bool| hs(index, HalfSwizzle::F32, abs, false);
    let neg_high = |op: OpHFma2| {
        Op::HFma2(OpHFma2 {
            neg_high_b: true,
            ..op
        })
    };
    let refused = [
        (
            neg_high(hfma2(
                r(0),
                wide(1, false),
                hs(2, HalfSwizzle::H0H0, false, false),
                h(3),
            )),
            U::HalfFmaMixedSign,
        ),
        (
            neg_high(hfma2(
                r(0),
                wide(1, false),
                hs(2, HalfSwizzle::H1H1, false, false),
                h(3),
            )),
            U::HalfFmaMixedSign,
        ),
        (
            neg_high(hfma2(r(0), h(1), wide(2, false), h(3))),
            U::HalfFmaMixedSign,
        ),
        (
            neg_high(hfma2(
                r(0),
                wide(1, false),
                hs(2, HalfSwizzle::H1H0, true, false),
                h(3),
            )),
            U::HalfFmaMixedSign,
        ),
        (
            neg_high(hfma2(
                r(0),
                wide(1, false),
                h(FMA_MIXED_SIGN_FIRST_SCRATCH),
                h(3),
            )),
            U::HalfFmaMixedSign,
        ),
        (
            neg_high(hfma2(r(0), wide(1, false), h(255), h(3))),
            U::HalfFmaMixedSign,
        ),
        (
            Op::HFma2(hfma2(
                r(0),
                wide(1, false),
                hs(2, HalfSwizzle::H1H0, true, false),
                h(3),
            )),
            U::HalfFmaAbsolute,
        ),
        (
            Op::HFma2(hfma2(
                r(0),
                wide(1, false),
                h(2),
                hs(3, HalfSwizzle::H1H0, true, false),
            )),
            U::HalfFmaAbsolute,
        ),
        (
            Op::HFma2(hfma2(
                r(0),
                wide(FMA_ABSOLUTE_FIRST_SCRATCH, true),
                h(2),
                h(3),
            )),
            U::HalfFmaAbsolute,
        ),
        (
            Op::HFma2(hfma2(r(0), wide(255, true), h(2), h(3))),
            U::HalfFmaAbsolute,
        ),
    ];
    for (op, what) in refused {
        assert_refused(op, what);
    }
}

/// A promoted `HFMA2` one step inside each of the lowering's edges is translated: `.H0_NH1`
/// with `b` in `R249`, and `|a|` in `R247`.
#[test]
fn a_promoted_fma_one_step_inside_the_lowerings_edges() {
    let wide = |index: u8, abs: bool| hs(index, HalfSwizzle::F32, abs, false);
    let neg_high = |op: OpHFma2| {
        Op::HFma2(OpHFma2 {
            neg_high_b: true,
            ..op
        })
    };
    let below_b = FMA_MIXED_SIGN_FIRST_SCRATCH - 1;
    let below_a = FMA_ABSOLUTE_FIRST_SCRATCH - 1;
    // 1.5 * (1, -2) + (1, 1) = (2.5, -2); 1.5 * (1, 2) + (1, 1) = (2.5, 4).
    let mut machine = Machine::new();
    machine.regs[1] = f32_bits(1.5);
    machine.regs[usize::from(below_a)] = f32_bits(-1.5);
    machine.regs[usize::from(below_b)] = pack(H_ONE, H_TWO);
    machine.regs[3] = pack(H_ONE, H_ONE);
    for (op, expected) in [
        (
            neg_high(hfma2(r(0), wide(1, false), h(below_b), h(3))),
            pack(H_TWO_HALF, H_NEG_TWO),
        ),
        (
            Op::HFma2(hfma2(r(0), wide(below_a, true), h(below_b), h(3))),
            pack(H_TWO_HALF, H_FOUR),
        ),
    ] {
        let mut run = machine.clone();
        run_on(&[Instr::new(op)], &mut run);
        assert_eq!(run.regs[0], expected, "{op}");
    }
}

/// Owner-question row: `HFMA2.H0_NH1` on packed halves with a `b` the lowering refuses
/// ("native half mixed-sign source mode": a swizzle other than `H1_H0`, an absolute value, a
/// `b` in `R250` or above, or a constant buffer) is translated as the native translator does,
/// `b`'s high lane negated after its modifiers. `(1, 2) * (2, -2) + (1, -1) = (3, -5)` in each.
#[test]
fn fma_mixed_sign_sources_the_lowering_refuses() {
    let mut machine = Machine::new();
    machine.regs[1] = pack(H_ONE, H_TWO);
    machine.regs[2] = pack(H_TWO, H_FOUR);
    machine.regs[3] = pack(H_ONE, H_NEG_ONE);
    machine.regs[4] = pack(H_NEG_TWO, H_NEG_TWO);
    machine.regs[usize::from(FMA_MIXED_SIGN_FIRST_SCRATCH)] = pack(H_TWO, H_TWO);
    machine.cbufs.insert(7, cbuf_bytes(&[pack(H_TWO, H_TWO)]));
    let neg_high = |b: HalfSrc| {
        Op::HFma2(OpHFma2 {
            neg_high_b: true,
            ..hfma2(r(0), h(1), b, h(3))
        })
    };
    for b in [
        hs(2, HalfSwizzle::H0H0, false, false),
        hs(4, HalfSwizzle::H1H0, true, false),
        h(FMA_MIXED_SIGN_FIRST_SCRATCH),
        hcbuf(7, 0, HalfSwizzle::H1H0, false, false),
    ] {
        let op = neg_high(b);
        let mut run = machine.clone();
        run_on(&[Instr::new(op)], &mut run);
        assert_eq!(run.regs[0], pack(H_THREE, 0xc500), "{op}");
    }
}

/// Owner-question row: `HFMA2` on packed halves with an absolute value the lowering refuses
/// ("native half fused multiply-add absolute source": on `b`, on `c`, or on an `a` in `R248` or
/// above, `RZ` included) is translated as the native translator does.
#[test]
fn fma_absolute_sources_the_lowering_refuses() {
    let first = FMA_ABSOLUTE_FIRST_SCRATCH;
    let mut machine = Machine::new();
    machine.regs[1] = pack(H_ONE, H_TWO);
    machine.regs[2] = pack(H_TWO, H_TWO);
    machine.regs[3] = pack(H_ONE, H_NEG_ONE);
    machine.regs[4] = pack(H_NEG_TWO, H_NEG_TWO);
    machine.regs[usize::from(first)] = pack(H_NEG_ONE, H_NEG_TWO);
    let abs = |index: u8| hs(index, HalfSwizzle::H1H0, true, false);
    for (op, expected) in [
        // (1, 2) * |(-2, -2)| + (1, -1) = (3, 3).
        (hfma2(r(0), h(1), abs(4), h(3)), pack(H_THREE, H_THREE)),
        // (1, 2) * (2, 2) + |(1, -1)| = (3, 5).
        (hfma2(r(0), h(1), h(2), abs(3)), pack(H_THREE, H_FIVE)),
        // |(-1, -2)| * (2, 2) + (1, -1) = (3, 3).
        (hfma2(r(0), abs(first), h(2), h(3)), pack(H_THREE, H_THREE)),
        // |RZ| * (2, 2) + (1, -1) = (1, -1).
        (hfma2(r(0), abs(255), h(2), h(3)), pack(H_ONE, H_NEG_ONE)),
    ] {
        let op = Op::HFma2(op);
        let mut run = machine.clone();
        run_on(&[Instr::new(op)], &mut run);
        assert_eq!(run.regs[0], expected, "{op}");
    }
}

/// Owner-question row: `HSETP2` outside the lowering's one case (`0x434`: form 2, an immediate
/// `b`, `a`'s swizzle 0 or 2) is translated as the native translator does: a register `b` (form
/// 1), a constant-buffer `b`, and `a`'s swizzle `H1_H1`. `(3, 3) < (2, 4)` is `(false, true)`.
#[test]
fn hsetp2_forms_the_lowering_refuses() {
    let mut machine = Machine::new();
    machine.regs[3] = pack(H_ONE, H_THREE);
    machine.regs[4] = pack(H_TWO, H_FOUR);
    machine.cbufs.insert(5, cbuf_bytes(&[pack(H_TWO, H_FOUR)]));
    let a = hs(3, HalfSwizzle::H1H1, false, false);
    for b in [h(4), hcbuf(5, 0, HalfSwizzle::H1H0, false, false)] {
        let op = Op::HSetP2(hsetp2(p(1), p(2), FloatCmpOp::Lt, a, b));
        let mut run = machine.clone();
        run.preds[1] = true;
        run.preds[2] = false;
        run_on(&[Instr::new(op)], &mut run);
        assert_eq!((run.preds[1], run.preds[2]), (false, true), "{op}");
    }
}

/// Owner-question row: `HMNMX2` with a `b` the lowering refuses (a constant buffer, opcode
/// `0xa40`, which has no case; a register from `R250` to `R254`, case `0x240`'s "native half min/max
/// source mode") is translated as the native translator does. The minimum of `(1, 4)` and
/// `(2, 3)` is `(1, 3)`.
#[test]
fn hmnmx2_sources_the_lowering_refuses() {
    let mut machine = Machine::new();
    machine.regs[1] = pack(H_ONE, H_FOUR);
    machine.regs[250] = pack(H_TWO, H_THREE);
    machine.regs[254] = pack(H_TWO, H_THREE);
    machine.cbufs.insert(3, cbuf_bytes(&[pack(H_TWO, H_THREE)]));
    for b in [h(250), h(254), hcbuf(3, 0, HalfSwizzle::H1H0, false, false)] {
        let op = Op::HMnMx2(hmnmx2(r(0), h(1), b, PredSrc::TRUE));
        let mut run = machine.clone();
        run_on(&[Instr::new(op)], &mut run);
        assert_eq!(run.regs[0], pack(H_ONE, H_THREE), "{op}");
    }
}

/// The IR of one `HADD2`, as the PR's translator builds it: each source unpacked, its lanes
/// taken by the swizzle, its modifiers, the two additions and the pack.
#[test]
fn golden_hadd2() {
    let op = Op::HAdd2(hadd2(r(0), h(1), hs(2, HalfSwizzle::H1H1, false, true)));
    assert_eq!(
        translated(&[Instr::new(op)]).to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r1\n\
         \x20   %1: f16x2 = UnpackFloat2x16 %0\n\
         \x20   %2: f16 = CompositeExtractF16x2 %1, u32 0x0\n\
         \x20   %3: f16 = CompositeExtractF16x2 %1, u32 0x1\n\
         \x20   %4: u32 = GetRegister r2\n\
         \x20   %5: f16x2 = UnpackFloat2x16 %4\n\
         \x20   %6: f16 = CompositeExtractF16x2 %5, u32 0x1\n\
         \x20   %7: f16 = FPNeg16 %6\n\
         \x20   %8: f16 = FPNeg16 %6\n\
         \x20   %9: f16 = FPAdd16 %2, %7 {no_contraction, fmz_none}\n\
         \x20   %10: f16 = FPAdd16 %3, %8 {no_contraction, fmz_none}\n\
         \x20   %11: f16x2 = CompositeConstructF16x2 %9, %10\n\
         \x20   %12: u32 = PackFloat2x16 %11\n\
         \x20   %13 = SetRegister r0, %12\n\
         \x20   return\n"
    );
}

/// The IR of `HFMA2.RELU`'s clamp of one lane: widened, a value at or below zero (`-0.0`
/// included, the lowering's sign test) replaced by zero, a NaN by `RELU_NAN`, narrowed back. The NaN replacement
/// changes no value the interpreter gives (module docs), so this pins it.
#[test]
fn golden_relu_clamp() {
    let op = Op::HFma2(OpHFma2 {
        relu: Some(PredSrc::TRUE),
        ..hfma2(r(0), h(1), h(2), h(3))
    });
    let text = translated(&[Instr::new(op)]).to_string();
    let clamp = "\
         \x20   %14: f32 = ConvertF32F16 %12\n\
         \x20   %15: u1 = FPOrdLessThanEqual32 %14, f32 0x0\n\
         \x20   %16: f32 = SelectF32 %15, f32 0x0, %14\n\
         \x20   %17: u1 = FPIsNan32 %14\n\
         \x20   %18: f32 = SelectF32 %17, f32 0x7fc00000, %16\n\
         \x20   %19: f16 = ConvertF16F32 %18\n";
    assert!(text.contains(clamp), "{text}");
}
