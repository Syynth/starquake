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

/// `HFMA2.RELU` clamps each lane as the native translator does: below zero to `+0.0`, a NaN of
/// either sign to `RELU_NAN` narrowed (the IR's canonical half NaN, `0x7fff`, as the
/// interpreter models a conversion's NaN result), and everything else kept, `-0.0` and positive
/// infinity included. The lowering gives `+0.0` for `-0.0` (module docs, owner question).
#[test]
fn relu_mirrors_the_native_clamp() {
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
        (H_NZERO, H_NZERO),
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
        // The PR's lowering: 0x0000.
        (0x8000, 0x8000, H_NZERO),
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

/// PR #4588's conditional `ReLU` (`native_shader.cpp`, "keeps unclamped results when
/// disabled"): `R8 * 1 + R11` with bits 87 to 90 as the predicate. Only `PT` (7) is translated,
/// as the native translator requires; every other predicate is refused (module docs, owner
/// question), where the PR's lowering keeps the clamped result only where it holds.
#[test]
fn relu_with_a_predicate_is_refused() {
    for condition in [0_u32, 1, 6, 7, 8, 9, 14, 15] {
        let instr = pr_word([
            (8 << 24) | (4 << 16) | 0x7831,
            0x3c00_3c00,
            0x800b | (condition << 23),
            0,
        ]);
        if condition != 7 {
            let index = u8::try_from(condition & 7).expect("three bits");
            let relu = PredSrc {
                reg: Pred::new(index).expect("a predicate"),
                negated: condition & 8 != 0,
            };
            assert_refused(instr.op, HalfDoubleUnsupported::HalfFmaReluPredicate(relu));
            continue;
        }
        for (value, expected) in [
            (0xbc00_3c00, 0x0000_4000),
            (0x3c00_bc00, 0x4000_0000),
            (0xc000_c000, 0x0000_0000),
            (0x7c00_fc00, 0x7c00_0000),
        ] {
            let machine = run_op(instr.op, &[(8, value), (11, 0x3c00_3c00)]);
            assert_eq!(machine.regs[4], expected, "{value:#x}");
        }
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

/// Each comparison's result on four pairs: (1, 2), (NaN, 1), (2, 2) and (3, 2), written out
/// from the comparison's meaning (`.u` true when unordered, `.num` when both are numbers,
/// `.nan` when either is not).
pub(super) const COMPARISONS: [(FloatCmpOp, [bool; 4]); 16] = [
    (FloatCmpOp::False, [false, false, false, false]),
    (FloatCmpOp::Lt, [true, false, false, false]),
    (FloatCmpOp::Eq, [false, false, true, false]),
    (FloatCmpOp::Le, [true, false, true, false]),
    (FloatCmpOp::Gt, [false, false, false, true]),
    (FloatCmpOp::Ne, [true, false, false, true]),
    (FloatCmpOp::Ge, [false, false, true, true]),
    (FloatCmpOp::Num, [true, false, true, true]),
    (FloatCmpOp::Nan, [false, true, false, false]),
    (FloatCmpOp::LtU, [true, true, false, false]),
    (FloatCmpOp::EqU, [false, true, true, false]),
    (FloatCmpOp::LeU, [true, true, true, false]),
    (FloatCmpOp::GtU, [false, true, false, true]),
    (FloatCmpOp::NeU, [true, true, false, true]),
    (FloatCmpOp::GeU, [false, true, true, true]),
    (FloatCmpOp::True, [true, true, true, true]),
];

/// `HSETP2` compares each lane, every comparison: the low lane's result to the first
/// predicate, the high lane's to the second. The pairs (1, 2) and (NaN, 1) are the lanes of
/// one comparison, (2, 2) and (3, 2) of another.
#[test]
fn hsetp2_compares_each_lane() {
    let ops: Vec<FloatCmpOp> = COMPARISONS.iter().map(|&(cmp, _)| cmp).collect();
    assert_eq!(ops, FloatCmpOp::ALL);
    for (cmp, [first, second, third, fourth]) in COMPARISONS {
        let op = Op::HSetP2(hsetp2(p(1), p(2), cmp, h(3), h(4)));
        for (a, b, low, high) in [
            (pack(H_ONE, H_QNAN), pack(H_TWO, H_ONE), first, second),
            (pack(H_TWO, H_THREE), pack(H_TWO, H_TWO), third, fourth),
        ] {
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
/// where false; the accumulator and `.FTZ` work as in `HSETP2`.
#[test]
fn hset2_writes_each_lane() {
    for (bf, truth) in [(true, HSET2_TRUE_HALF), (false, HSET2_TRUE_MASK)] {
        for (cmp, [first, second, ..]) in COMPARISONS {
            let op = Op::HSet2(OpHSet2 {
                bf,
                ..hset2(r(0), cmp, h(3), h(4))
            });
            let machine = run_op(op, &[(3, pack(H_ONE, H_QNAN)), (4, pack(H_TWO, H_ONE))]);
            let lane = |on: bool| if on { truth } else { 0 };
            assert_eq!(machine.regs[0], pack(lane(first), lane(second)), "{op}");
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

/// Swizzle 1 (a 32-bit float) is refused in every packed-half source of every instruction, as
/// the native translator's `HalfLanes` refuses it (module docs: the lowering widens to single
/// precision in `HADD2`, `HMUL2`, `HFMA2` and `HSET2`).
#[test]
fn swizzle_1_is_refused_everywhere() {
    use HalfDoubleUnsupported as U;
    let f = hs(1, HalfSwizzle::F32, false, false);
    let cases = [
        (Op::HAdd2(hadd2(r(0), f, h(2))), 0),
        (Op::HAdd2(hadd2(r(0), h(2), f)), 1),
        (Op::HMul2(hmul2(r(0), f, h(2))), 0),
        (Op::HMul2(hmul2(r(0), h(2), f)), 1),
        (Op::HFma2(hfma2(r(0), f, h(2), h(3))), 0),
        (Op::HFma2(hfma2(r(0), h(2), f, h(3))), 1),
        (Op::HFma2(hfma2(r(0), h(2), h(3), f)), 2),
        (Op::HMnMx2(hmnmx2(r(0), f, h(2), PredSrc::TRUE)), 0),
        (Op::HMnMx2(hmnmx2(r(0), h(2), f, PredSrc::TRUE)), 1),
        (Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, f, h(2))), 0),
        (Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, h(2), f)), 1),
        (Op::HSet2(hset2(r(0), FloatCmpOp::Lt, f, h(2))), 0),
        (Op::HSet2(hset2(r(0), FloatCmpOp::Lt, h(2), f)), 1),
        (
            Op::HSet2(hset2(
                r(0),
                FloatCmpOp::Lt,
                h(2),
                hcbuf(1, 0, HalfSwizzle::F32, false, false),
            )),
            1,
        ),
    ];
    for (op, index) in cases {
        assert_refused(op, U::HalfAsFloat(index));
    }
}

/// `.FMZ` is refused in `HMUL2` and `HFMA2`, as the native translator refuses it (module docs,
/// owner question).
#[test]
fn fmz_is_refused() {
    assert_refused(
        Op::HMul2(OpHMul2 {
            dnz: true,
            ..hmul2(r(0), h(1), h(2))
        }),
        HalfDoubleUnsupported::HalfMultiplyZero,
    );
    assert_refused(
        Op::HFma2(OpHFma2 {
            dnz: true,
            ..hfma2(r(0), h(1), h(2), h(3))
        }),
        HalfDoubleUnsupported::HalfFmaMultiplyZero,
    );
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

/// The IR of `HFMA2.RELU`'s clamp of one lane, as the PR's translator builds it: widened, a
/// negative value replaced by zero, a NaN by `RELU_NAN`, narrowed back. The NaN replacement
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
         \x20   %15: u1 = FPOrdLessThan32 %14, f32 0x0\n\
         \x20   %16: f32 = SelectF32 %15, f32 0x0, %14\n\
         \x20   %17: u1 = FPIsNan32 %14\n\
         \x20   %18: f32 = SelectF32 %17, f32 0x7fc00000, %16\n\
         \x20   %19: f16 = ConvertF16F32 %18\n";
    assert!(text.contains(clamp), "{text}");
}
