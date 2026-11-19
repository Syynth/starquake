//! Conversions: `FRND`, `F2I`, `I2F`, `I2FP`, `F2F` and `F2FP`, at each rounding mode and at
//! the integer range edges.

use super::*;
use crate::interp::Val;

const MODES: [RoundMode; 4] = [
    RoundMode::NearestEven,
    RoundMode::NegInf,
    RoundMode::PosInf,
    RoundMode::Zero,
];

/// `FRND` rounds to an integral float by its field, with the operation the mode names, and
/// carries `.FTZ`.
#[test]
fn frnd_rounds_by_its_field() {
    // Results for nearest, down, up, toward zero.
    let cases: [(f32, [u32; 4]); 4] = [
        (2.5, [0x4000_0000, 0x4000_0000, 0x4040_0000, 0x4000_0000]),
        (-2.5, [0xc000_0000, 0xc040_0000, 0xc000_0000, 0xc000_0000]),
        (1.5, [0x4000_0000, 0x3f80_0000, 0x4000_0000, 0x3f80_0000]),
        (-0.5, [0x8000_0000, 0xbf80_0000, 0x8000_0000, 0x8000_0000]),
    ];
    let opcodes_by_mode = [
        Opcode::FPRoundEven32,
        Opcode::FPFloor32,
        Opcode::FPCeil32,
        Opcode::FPTrunc32,
    ];
    for (index, rnd) in MODES.into_iter().enumerate() {
        for (value, results) in cases {
            let machine = run_op(Op::FRnd(frnd(r(0), reg(1), rnd)), &[(1, value.to_bits())]);
            assert_eq!(machine.regs[0], results[index], "{value} {rnd:?}");
        }
        for (ftz, fmz_mode) in [(false, FmzMode::None), (true, FmzMode::Ftz)] {
            let op = OpFRnd {
                ftz,
                ..frnd(r(0), reg(1), rnd)
            };
            let function = translated(&[Instr::new(Op::FRnd(op))]);
            assert_eq!(
                flags_of(&function, opcodes_by_mode[index]),
                [Flags::Fp(FpControl {
                    no_contraction: false,
                    rounding: FpRounding::DontCare,
                    fmz_mode,
                })]
            );
        }
    }
}

/// Runs `F2I` on `value` (a 32-bit float's bits) and returns the destination.
fn f2i_of(op: OpF2I, value: u32) -> u32 {
    run_op(Op::F2I(op), &[(1, value)]).regs[0]
}

/// `F2I` toward zero at each destination type's range edges: NaN, the infinities, the bounds,
/// the first values past them. Narrow results are sign- or zero-extended into the register.
#[test]
fn f2i_range_edges() {
    let nan = QNAN;
    let cases: [(IntType, &[(f32, u32)]); 6] = [
        (
            int(0, false),
            &[
                (255.0, 255),
                (256.0, 255),
                (254.9, 254),
                (-0.9, 0),
                (-1.0, 0),
                (f32::INFINITY, 255),
                (f32::NEG_INFINITY, 0),
            ],
        ),
        (
            int(0, true),
            &[
                (127.0, 127),
                (128.0, 127),
                (-128.0, 0xffff_ff80),
                (-129.0, 0xffff_ff80),
                (-1.9, 0xffff_ffff),
                (f32::INFINITY, 127),
                (f32::NEG_INFINITY, 0xffff_ff80),
            ],
        ),
        (
            int(1, false),
            &[(65535.0, 65535), (65536.0, 65535), (-1.0, 0)],
        ),
        (
            int(1, true),
            &[
                (32767.0, 32767),
                (32768.0, 32767),
                (-32768.0, 0xffff_8000),
                (-32769.0, 0xffff_8000),
            ],
        ),
        (
            IntType::U32,
            &[
                (4_294_967_040.0, 0xffff_ff00),
                (4_294_967_296.0, 0xffff_ffff),
                (f32::INFINITY, 0xffff_ffff),
                (-1.0, 0),
            ],
        ),
        (
            IntType::S32,
            &[
                (2_147_483_520.0, 0x7fff_ff80),
                (2_147_483_648.0, 0x7fff_ffff),
                (-2_147_483_648.0, 0x8000_0000),
                (-2_147_483_904.0, 0x8000_0000),
                (f32::INFINITY, 0x7fff_ffff),
                (f32::NEG_INFINITY, 0x8000_0000),
            ],
        ),
    ];
    for (dst_type, values) in cases {
        let op = f2i(r(0), reg(1), dst_type, RoundMode::Zero);
        for &(value, expected) in values {
            assert_eq!(f2i_of(op, value.to_bits()), expected, "{dst_type} {value}");
        }
        // A NaN gives 0, signed (the PR's select) and unsigned (the clamp's low bound).
        assert_eq!(f2i_of(op, nan), 0, "{dst_type} NaN");
    }
}

/// `F2I` rounds by the field before clamping: a value one rounding from each bound.
#[test]
fn f2i_rounds_before_clamping() {
    // u8 254.5: nearest even 254, down 254, up 255, toward zero 254.
    // s8 -128.5: nearest -128, down -129 clamped to -128, up -128, toward zero -128.
    // s32 -0.5: nearest -0 = 0, down -1, up 0, toward zero 0.
    let cases: [(IntType, f32, [u32; 4]); 3] = [
        (int(0, false), 254.5, [254, 254, 255, 254]),
        (int(0, true), -128.5, [0xffff_ff80; 4]),
        (IntType::S32, -0.5, [0, 0xffff_ffff, 0, 0]),
    ];
    for (dst_type, value, results) in cases {
        for (rnd, expected) in MODES.into_iter().zip(results) {
            let op = f2i(r(0), reg(1), dst_type, rnd);
            assert_eq!(
                f2i_of(op, value.to_bits()),
                expected,
                "{dst_type} {value} {rnd:?}"
            );
        }
    }
}

/// PR #4588's "Native unsigned conversion saturates, clears NaNs, and extracts the low byte":
/// `F2I.U32` (toward zero, `.FTZ`, `.NTZ`), then `I2F.F32.U8` of its low byte; with the source
/// in a register or a constant buffer, and as `F2I.U32.FLOOR` and `F2I.U16` to nearest.
#[test]
fn f2i_then_low_byte() {
    let cases: [(u32, u32); 9] = [
        (QNAN, 0),
        (NEG_INF, 0),
        ((-3.0_f32).to_bits(), 0),
        (1.9_f32.to_bits(), 1),
        (300.75_f32.to_bits(), 44),
        (65535.0_f32.to_bits(), 255),
        (4_294_967_040.0_f32.to_bits(), 0),
        (4_294_967_296.0_f32.to_bits(), 255),
        (INF, 255),
    ];
    // Nearest, clamped to 16 bits, low byte.
    let short = [0, 0, 0, 2, 45, 255, 255, 255, 255];
    let forms = [
        (IntType::U32, RoundMode::Zero, true, true),
        (IntType::U32, RoundMode::NegInf, false, false),
        (int(1, false), RoundMode::NearestEven, false, false),
    ];
    for constant in [false, true] {
        for (dst_type, rnd, ftz, ntz) in forms {
            let src = if constant { cbuf(3, 0) } else { reg(0) };
            let convert = OpF2I {
                ftz,
                ntz,
                ..f2i(r(1), src, dst_type, rnd)
            };
            let byte = i2f(r(0), reg(1), int(0, false), RoundMode::NearestEven);
            let program = [Instr::new(Op::F2I(convert)), Instr::new(Op::I2F(byte))];
            for (index, (value, expected)) in cases.into_iter().enumerate() {
                let mut machine = Machine::new();
                machine.regs[0] = value;
                machine.cbufs.insert(3, value.to_le_bytes().to_vec());
                run_on(&program, &mut machine);
                let expected = if dst_type == IntType::U32 {
                    expected
                } else {
                    short[index]
                };
                let expected = f32::from(u16::try_from(expected).expect("a byte")).to_bits();
                assert_eq!(machine.regs[0], expected, "{value:#x} {rnd:?} {dst_type}");
            }
        }
    }
}

/// PR #4588's "Native NTZ nearest conversion rounds ties to even and clears NaNs" (`F2I.S32`
/// to nearest), "Native signed NTZ floor conversion rounds negative fractions down"
/// (`F2I.S32.FLOOR`) and "Native signed NTZ truncation preserves denormal control and aliases"
/// (`F2I.S32.TRUNC`, with and without `.FTZ`, into the source register or another).
#[test]
fn f2i_signed_cases_from_the_pr() {
    let nearest: [(f32, i32); 9] = [
        (0.5, 0),
        (1.5, 2),
        (2.5, 2),
        (-0.5, 0),
        (-1.5, -2),
        (-2.5, -2),
        (0.9, 1),
        (-0.9, -1),
        (f32::NAN, 0),
    ];
    let floor: [(f32, i32); 8] = [
        (1.9, 1),
        (-1.1, -2),
        (-0.1, -1),
        (0.9, 0),
        (0.0, 0),
        (-2.0, -2),
        (12345.5, 12345),
        (f32::NAN, 0),
    ];
    let trunc: [(f32, i32); 11] = [
        (0.0, 0),
        (-0.0, 0),
        (0.9, 0),
        (-0.9, 0),
        (1.9, 1),
        (-1.9, -1),
        (12345.75, 12345),
        (-12345.75, -12345),
        (f32::from_bits(1), 0),
        (f32::from_bits(0x8000_0001), 0),
        (f32::NAN, 0),
    ];
    let check = |op: OpF2I, src: u8, cases: &[(f32, i32)]| {
        for &(value, expected) in cases {
            let machine = run_op(Op::F2I(op), &[(src, value.to_bits())]);
            assert_eq!(
                machine.regs[usize::from(op.dst.index())],
                expected.cast_unsigned(),
                "{value} {:?}",
                op.rnd
            );
        }
    };
    let op = OpF2I {
        ntz: true,
        ..f2i(r(17), reg(2), IntType::S32, RoundMode::NearestEven)
    };
    check(op, 2, &nearest);
    let op = OpF2I {
        ntz: true,
        ..f2i(r(35), reg(36), IntType::S32, RoundMode::NegInf)
    };
    check(op, 36, &floor);
    for ftz in [false, true] {
        for dst in [0, 2] {
            let op = OpF2I {
                ntz: true,
                ftz,
                ..f2i(r(dst), reg(0), IntType::S32, RoundMode::Zero)
            };
            check(op, 0, &trunc);
            let function = translated(&[Instr::new(Op::F2I(op))]);
            let fmz_mode = if ftz { FmzMode::Ftz } else { FmzMode::None };
            assert_eq!(
                flags_of(&function, Opcode::FPTrunc32),
                [Flags::Fp(FpControl {
                    no_contraction: true,
                    rounding: FpRounding::DontCare,
                    fmz_mode,
                })]
            );
        }
    }
}

/// PR #4588's "Native unsigned half conversion truncates and saturates NTZ edge cases":
/// `F2I.U32.F16.TRUNC` of R13's low half, the high half holding other bits; the bounds clamp
/// in the half format, where `2^32 - 1` is infinity, so infinity saturates to `0xffffffff`.
/// Also the high half, and a signed destination, where a NaN gives 0.
#[test]
fn f2i_from_a_half() {
    let cases: [(u32, u32); 8] = [
        (0x3e00, 1),
        (0x4100, 2),
        (0xbe00, 0),
        (0x0001, 0),
        (0x7bff, 65504),
        (0x7c00, 0xffff_ffff),
        (0xfc00, 0),
        (0x7e00, 0),
    ];
    let low = OpF2I {
        src_type: FloatType::F16,
        ntz: true,
        ..f2i(r(14), reg(13), IntType::U32, RoundMode::Zero)
    };
    let high = OpF2I { high: true, ..low };
    for (half, expected) in cases {
        let machine = run_op(Op::F2I(low), &[(13, 0xdead_0000 | half)]);
        assert_eq!(machine.regs[14], expected, "{half:#x}");
        let machine = run_op(Op::F2I(high), &[(13, (half << 16) | 0xdead)]);
        assert_eq!(machine.regs[14], expected, "high {half:#x}");
    }
    let signed = OpF2I {
        dst_type: IntType::S32,
        ..low
    };
    // -2^31 is -infinity in the half format, so -infinity stays and converts to the minimum.
    for (half, expected) in [
        (0x7e00, 0),
        (0xfc00, 0x8000_0000),
        (0xbe00, 0xffff_ffff),
        (0xc100, 0xffff_fffe),
    ] {
        let machine = run_op(Op::F2I(signed), &[(13, half)]);
        assert_eq!(machine.regs[14], expected, "signed {half:#x}");
    }
    // A negated half: `-(1.5)` toward zero is -1.
    let negated = OpF2I {
        src: reg_mod(13, false, true),
        ..signed
    };
    let machine = run_op(Op::F2I(negated), &[(13, 0x3e00)]);
    assert_eq!(machine.regs[14], 0xffff_ffff);
}

/// `.FTZ` on a 32-bit `F2I` operand is emulated: a denormal is a zero before rounding, so
/// rounding down or up gives 0 where the denormal would give -1 or 1, with the control
/// carried (the function as translated) and ignored (as a backend that cannot honour it sees
/// it). A 16-bit operand is not flushed: the PR leaves its denormal mode unset.
#[test]
fn f2i_flushes_denormals_before_rounding() {
    let cases = [
        (RoundMode::NegInf, 0x8000_0001_u32, 0xffff_ffff_u32),
        (RoundMode::PosInf, 0x0000_0001, 1),
    ];
    for (rnd, denormal, kept) in cases {
        for ftz in [false, true] {
            let op = OpF2I {
                ftz,
                ..f2i(r(0), reg(1), IntType::S32, rnd)
            };
            let function = translated(&[Instr::new(Op::F2I(op))]);
            for function in [function.clone(), without_controls(function)] {
                let mut machine = Machine::new();
                machine.regs[1] = denormal;
                machine.run(&function).expect("runs");
                assert_eq!(machine.regs[0], if ftz { 0 } else { kept }, "{rnd:?} {ftz}");
            }
        }
    }
    // The smallest normal is kept.
    let op = OpF2I {
        ftz: true,
        ..f2i(r(0), reg(1), IntType::S32, RoundMode::PosInf)
    };
    assert_eq!(f2i_of(op, f32::MIN_POSITIVE.to_bits()), 1);
    // A half operand's denormal rounds up to 1 even with `.FTZ`.
    let half = OpF2I {
        ftz: true,
        src_type: FloatType::F16,
        ..f2i(r(0), reg(1), IntType::S32, RoundMode::PosInf)
    };
    assert_eq!(f2i_of(half, 0x0001), 1);
    let function = translated(&[Instr::new(Op::F2I(half))]);
    assert_eq!(
        flags_of(&function, Opcode::FPCeil16),
        [Flags::Fp(FpControl {
            no_contraction: true,
            rounding: FpRounding::DontCare,
            fmz_mode: FmzMode::DontCare,
        })]
    );
}

/// PR #4588's "Native byte float conversions select all four bytes": `I2F.F32.U8` of each byte
/// of `0xff807f00`, the source left as it was; and the signed and 16-bit selects.
#[test]
fn i2f_selects_bytes_and_halves() {
    let word = 0xff80_7f00;
    let cases: [(IntType, u8, f32); 10] = [
        (int(0, false), 0, 0.0),
        (int(0, false), 1, 127.0),
        (int(0, false), 2, 128.0),
        (int(0, false), 3, 255.0),
        (int(0, true), 2, -128.0),
        (int(0, true), 3, -1.0),
        (int(1, false), 0, 32512.0),
        (int(1, false), 1, 65408.0),
        (int(1, true), 1, -128.0),
        (IntType::S32, 0, -8_356_096.0),
    ];
    for (src_type, select, expected) in cases {
        let op = crate::decode::OpI2F {
            select,
            ..i2f(r(59), reg(33), src_type, RoundMode::NearestEven)
        };
        let machine = run_op(Op::I2F(op), &[(33, word)]);
        assert_eq!(machine.regs[59], expected.to_bits(), "{src_type} {select}");
        assert_eq!(machine.regs[33], word);
    }
}

/// The select's bound (`select < 32 >> (3 + size)`, zero for a full word) at its edge on both
/// sides; the decoder produces only the accepted side, so the refused one is hand-made.
#[test]
fn i2f_select_edges() {
    for (src_type, last) in [(int(0, false), 3_u8), (int(1, true), 1), (IntType::U32, 0)] {
        let accepted = crate::decode::OpI2F {
            select: last,
            ..i2f(r(0), reg(1), src_type, RoundMode::NearestEven)
        };
        translated(&[Instr::new(Op::I2F(accepted))]);
        let past = crate::decode::OpI2F {
            select: last + 1,
            ..accepted
        };
        assert_refused_raw(
            Op::I2F(past),
            FloatUnsupported::I2fSelect {
                select: last + 1,
                width: src_type.width(),
            },
        );
    }
}

/// PR #4588's "Native unsigned conversion toward zero is exact at float precision boundaries"
/// and "Native unsigned 32-bit conversion preserves nearest and toward-zero rounding":
/// `I2FP.F32.U32` to nearest and toward zero. The rounding is carried on the conversion, not
/// emulated (decision 10 names only float-to-integer and float-to-half conversions).
#[test]
fn i2f_carries_its_rounding() {
    // Source; toward zero (the PR's table); to nearest.
    let cases: [(u32, u32, u32); 10] = [
        (0, 0, 0),
        (1, 0x3f80_0000, 0x3f80_0000),
        (0x00ff_ffff, 0x4b7f_ffff, 0x4b7f_ffff),
        (0x0100_0000, 0x4b80_0000, 0x4b80_0000),
        (0x0100_0001, 0x4b80_0000, 0x4b80_0000),
        (0x0100_0003, 0x4b80_0001, 0x4b80_0002),
        (0x7fff_ffff, 0x4eff_ffff, 0x4f00_0000),
        (0x8000_0000, 0x4f00_0000, 0x4f00_0000),
        (0x8000_00ff, 0x4f00_0000, 0x4f00_0001),
        (0xffff_ffff, 0x4f7f_ffff, 0x4f80_0000),
    ];
    for (rnd, rounding) in [
        (RoundMode::Zero, FpRounding::Rz),
        (RoundMode::NearestEven, FpRounding::Rn),
    ] {
        let op = i2fp(r(21), reg(5), IntType::U32, rnd);
        for (source, toward_zero, nearest) in cases {
            let machine = run_op(Op::I2FP(op), &[(5, source)]);
            let expected = if rnd == RoundMode::Zero {
                toward_zero
            } else {
                nearest
            };
            assert_eq!(machine.regs[21], expected, "{source:#x} {rnd:?}");
        }
        let wanted = [Flags::Fp(FpControl {
            no_contraction: false,
            rounding,
            fmz_mode: FmzMode::DontCare,
        })];
        let function = translated(&[Instr::new(Op::I2FP(op))]);
        assert_eq!(flags_of(&function, Opcode::ConvertF32U32), wanted);
        let signed = i2f(r(0), reg(1), IntType::S32, rnd);
        let function = translated(&[Instr::new(Op::I2F(signed))]);
        assert_eq!(flags_of(&function, Opcode::ConvertF32S32), wanted);
    }
    // Upward and downward, carried and modelled: 2^24 + 1 sits between two floats.
    for (rnd, expected) in [
        (RoundMode::PosInf, 0x4b80_0001),
        (RoundMode::NegInf, 0x4b80_0000),
    ] {
        let op = i2fp(r(0), reg(1), IntType::U32, rnd);
        assert_eq!(run_op(Op::I2FP(op), &[(1, 0x0100_0001)]).regs[0], expected);
    }
    // `I2FP` from a signed 16-bit source reads the low half, sign-extended.
    let op = i2fp(
        r(0),
        AluSrc::Imm(0x0001_8000),
        int(1, true),
        RoundMode::NearestEven,
    );
    assert_eq!(run_op(Op::I2FP(op), &[]).regs[0], (-32768.0_f32).to_bits());
}

/// PR #4588's "Native scalar half conversion applies absolute value before rounding":
/// `F2F.F16.F32 R4, |R0|` and `F2F.F16.F32 R4, -R0`, to nearest even.
#[test]
fn f2f_applies_modifiers_before_rounding() {
    let cases: [(f32, u32); 7] = [
        (-1.0, 0x3c00),
        (-0.0, 0),
        (-2.5, 0x4100),
        (1.000_488_3, 0x3c00),
        (1.001_464_8, 0x3c02),
        (-5.960_464_5e-8, 1),
        (f32::NEG_INFINITY, 0x7c00),
    ];
    for negate in [false, true] {
        let src = reg_mod(0, !negate, negate);
        let op = f2f(r(4), src, RoundMode::NearestEven);
        for (value, expected) in cases {
            let machine = run_op(Op::F2F(op), &[(0, value.to_bits())]);
            let sign = if negate && !value.is_sign_negative() {
                F16_SIGN
            } else {
                0
            };
            assert_eq!(machine.regs[4], expected | sign, "{value} negate {negate}");
        }
    }
}

/// The half the interpreter's own model of `ConvertF16F32` with the mode and flushing as
/// flags gives for `input`: the oracle the emulation is compared with.
fn modelled_half(input: u32, rnd: RoundMode, ftz: bool) -> u32 {
    let mode = match rnd {
        RoundMode::NearestEven => "rn",
        RoundMode::NegInf => "rm",
        RoundMode::PosInf => "rp",
        RoundMode::Zero => "rz",
    };
    let flush = if ftz { "ftz" } else { "fmz_none" };
    let text =
        format!("block0:\n    %0 = ConvertF16F32 f32 {input:#x} {{{mode}, {flush}}}\n    return\n");
    let function = Function::parse(&text).expect("parses");
    let finished = Machine::new().run(&function).expect("runs");
    let bits = finished
        .value(0)
        .and_then(|value| match value {
            Val::F16(bits) => Some(bits),
            _ => None,
        })
        .expect("a half");
    u32::from(bits)
}

/// `F2F`'s result for `input`, with its controls carried and with them ignored.
fn f2f_results(rnd: RoundMode, ftz: bool, input: u32) -> [u32; 2] {
    let op = OpF2F {
        ftz,
        ..f2f(r(0), reg(1), rnd)
    };
    let function = translated(&[Instr::new(Op::F2F(op))]);
    [function.clone(), without_controls(function)].map(|function| {
        let mut machine = Machine::new();
        machine.regs[1] = input;
        machine.run(&function).expect("runs");
        machine.regs[0]
    })
}

/// `F2F`'s rounding and `.FTZ` are emulated (decision 10): for each mode, with and without
/// flushing, the result equals the interpreter's model of the conversion with the flags,
/// whether the translated function's own flags are honoured or ignored. Hand-picked edges,
/// worked out by hand: overflow, the largest finite half, ties, values below the smallest
/// denormal, denormal results, and a 32-bit denormal operand.
#[test]
fn f2f_emulation_matches_the_modelled_conversion() {
    use RoundMode::{NearestEven as N, NegInf as M, PosInf as P, Zero as Z};
    let cases: [(f32, bool, [u32; 4]); 12] = [
        // Input, .FTZ, then the half for nearest, down, up, toward zero.
        (65519.0, false, [0x7bff, 0x7bff, 0x7c00, 0x7bff]),
        (65520.0, false, [0x7c00, 0x7bff, 0x7c00, 0x7bff]),
        (-70000.0, false, [0xfc00, 0xfc00, 0xfbff, 0xfbff]),
        (1.000_488_3, false, [0x3c00, 0x3c00, 0x3c01, 0x3c00]),
        (-1.000_488_3, false, [0xbc00, 0xbc01, 0xbc00, 0xbc00]),
        (1.0e-8, false, [0x0000, 0x0000, 0x0001, 0x0000]),
        (-1.0e-8, false, [0x8000, 0x8001, 0x8000, 0x8000]),
        (1.0e-6, false, [0x0011, 0x0010, 0x0011, 0x0010]),
        (1.0e-6, true, [0x0000; 4]),
        (f32::from_bits(1), false, [0x0000, 0x0000, 0x0001, 0x0000]),
        (f32::from_bits(1), true, [0x0000; 4]),
        (f32::INFINITY, false, [0x7c00; 4]),
    ];
    for (input, ftz, results) in cases {
        for (rnd, expected) in [N, M, P, Z].into_iter().zip(results) {
            assert_eq!(
                modelled_half(input.to_bits(), rnd, ftz),
                expected,
                "model {input}"
            );
            for result in f2f_results(rnd, ftz, input.to_bits()) {
                assert_eq!(result, expected, "{input} {rnd:?} ftz {ftz}");
            }
        }
    }
    // A NaN converts to the canonical half NaN in every mode.
    for rnd in [N, M, P, Z] {
        assert_eq!(f2f_results(rnd, true, QNAN), [0x7fff; 2]);
    }
    // The emulation adds the step and the flush only where needed.
    let nearest = translated(&[Instr::new(Op::F2F(f2f(r(0), reg(1), N)))]);
    assert!(!opcodes(&nearest).contains(&Opcode::ConvertF32F16));
    let directed = translated(&[Instr::new(Op::F2F(f2f(r(0), reg(1), Z)))]);
    assert!(opcodes(&directed).contains(&Opcode::ConvertF32F16));
    // The conversion carries the PR's controls.
    let carried = translated(&[Instr::new(Op::F2F(OpF2F {
        ftz: true,
        ..f2f(r(0), reg(1), M)
    }))]);
    assert_eq!(
        flags_of(&carried, Opcode::ConvertF16F32)[0],
        Flags::Fp(FpControl {
            no_contraction: false,
            rounding: FpRounding::Rm,
            fmz_mode: FmzMode::Ftz,
        })
    );
}

/// Inputs that reach every region of a half: any bits, and binary32 values with exponents
/// around the half range.
fn half_range_input() -> impl Strategy<Value = u32> {
    prop_oneof![
        any::<u32>(),
        (any::<bool>(), 90_u32..150, 0_u32..(1 << 23)).prop_map(|(sign, exponent, mantissa)| {
            (u32::from(sign) << 31) | (exponent << 23) | mantissa
        }),
    ]
}

proptest! {
    #![proptest_config(config())]

    /// The emulated `F2F` equals the modelled conversion for generated inputs, in every mode,
    /// with and without flushing, its own controls honoured or ignored.
    #[test]
    fn f2f_emulation_is_exact(input in half_range_input(), mode in 0_usize..4, ftz in any::<bool>()) {
        let rnd = MODES[mode];
        let expected = modelled_half(input, rnd, ftz);
        for result in f2f_results(rnd, ftz, input) {
            prop_assert_eq!(result, expected);
        }
    }
}

/// `F2FP.PACK_AB`: the first source in the high half, the second in the low, each converted to
/// nearest; with `.MERGE_C` the low half of the third register goes in the high half.
#[test]
fn f2fp_packs_two_halves() {
    let pack = f2fp(r(0), reg(1), reg(2));
    let machine = run_op(Op::F2FP(pack), &[(1, ONE), (2, TWO)]);
    assert_eq!(machine.regs[0], 0x3c00_4000);
    // Ties to even and overflow to infinity.
    let machine = run_op(
        Op::F2FP(pack),
        &[(1, 1.000_488_3_f32.to_bits()), (2, 70000.0_f32.to_bits())],
    );
    assert_eq!(machine.regs[0], 0x3c00_7c00);
    let merged = OpF2FP {
        merge_c: true,
        c: r(3),
        ..pack
    };
    let machine = run_op(Op::F2FP(merged), &[(1, ONE), (2, TWO), (3, 0x1234_abcd)]);
    assert_eq!(machine.regs[0], 0xabcd_4000);
    // The second source from a constant buffer.
    let mut machine = Machine::new();
    machine.regs[1] = TWO;
    machine.cbufs.insert(4, ONE.to_le_bytes().to_vec());
    run_on(
        &[Instr::new(Op::F2FP(f2fp(r(0), reg(1), cbuf(4, 0))))],
        &mut machine,
    );
    assert_eq!(machine.regs[0], 0x4000_3c00);
}

/// `F2FP`'s selector and every rounding but to nearest are refused, each the only fault.
#[test]
fn f2fp_refuses_the_selector_and_rounding() {
    let merged = OpF2FP {
        merge_c: true,
        c: r(3),
        ..f2fp(r(0), reg(1), reg(2))
    };
    translated(&[Instr::new(Op::F2FP(merged))]);
    assert_refused(
        Op::F2FP(OpF2FP { h1: true, ..merged }),
        FloatUnsupported::F2fpSelector,
    );
    for rnd in [RoundMode::NegInf, RoundMode::PosInf, RoundMode::Zero] {
        assert_refused(
            Op::F2FP(OpF2FP { rnd, ..merged }),
            FloatUnsupported::F2fpRounding(rnd),
        );
    }
}

/// The size checks at their edges: each accepted size and its refused neighbours.
#[test]
fn conversion_sizes() {
    use FloatUnsupported as U;
    // F2I: destinations of 8, 16 and 32 bits, sources of 16 and 32.
    for size in 0..=2 {
        translated(&[Instr::new(Op::F2I(f2i(
            r(0),
            reg(1),
            int(size, true),
            RoundMode::Zero,
        )))]);
    }
    assert_refused(
        Op::F2I(f2i(r(0), reg(1), int(3, false), RoundMode::Zero)),
        U::F2iDestinationSize(int(3, false)),
    );
    for src_type in [FloatType::F8, FloatType::F64] {
        assert_refused(
            Op::F2I(OpF2I {
                src_type,
                ..f2i(r(0), reg(1), IntType::S32, RoundMode::Zero)
            }),
            U::F2iSourceSize(src_type),
        );
    }
    // I2F and I2FP: sources up to 32 bits, a 32-bit destination only.
    for size in 0..=2 {
        translated(&[Instr::new(Op::I2F(i2f(
            r(0),
            reg(1),
            int(size, false),
            RoundMode::Zero,
        )))]);
        translated(&[Instr::new(Op::I2FP(i2fp(
            r(0),
            reg(1),
            int(size, true),
            RoundMode::Zero,
        )))]);
    }
    assert_refused(
        Op::I2FP(i2fp(r(0), reg(1), int(3, true), RoundMode::Zero)),
        U::I2fSourceSize(int(3, true)),
    );
    for dst_type in [FloatType::F16, FloatType::F64] {
        assert_refused(
            Op::I2F(crate::decode::OpI2F {
                dst_type,
                ..i2f(r(0), reg(1), IntType::S32, RoundMode::Zero)
            }),
            U::I2fDestinationSize(dst_type),
        );
    }
    // FRND: 32 to 32 bits only; F2F: 32 to 16 bits only.
    for other in [FloatType::F16, FloatType::F64] {
        assert_refused(
            Op::FRnd(OpFRnd {
                src_type: other,
                ..frnd(r(0), reg(1), RoundMode::Zero)
            }),
            U::FrndSourceSize(other),
        );
        assert_refused(
            Op::FRnd(OpFRnd {
                dst_type: other,
                ..frnd(r(0), reg(1), RoundMode::Zero)
            }),
            U::FrndDestinationSize(other),
        );
        assert_refused(
            Op::F2F(OpF2F {
                src_type: if other == FloatType::F16 {
                    FloatType::F16
                } else {
                    FloatType::F64
                },
                ..f2f(r(0), reg(1), RoundMode::Zero)
            }),
            U::F2fSourceSize(other),
        );
    }
    for dst_type in [FloatType::F32, FloatType::F64] {
        assert_refused(
            Op::F2F(OpF2F {
                dst_type,
                ..f2f(r(0), reg(1), RoundMode::Zero)
            }),
            U::F2fDestinationSize(dst_type),
        );
    }
}
