//! PR #4588's cases for this group (`eden-pr4588/src/tests/video_core/native_shader.cpp` @
//! e5fb3b97), rebuilt with the test assembler from the instructions their words encode, with
//! the PR's expected values. Each test names the PR's `TEST_CASE`.
//!
//! The PR runs these through its lowering to Maxwell (`TranslateScalarNative`, which calls
//! `LowerToMaxwell`), not through the native visitor this module ports; where the lowering
//! accepts a form the native visitor refuses, the case is left out and named here:
//!
//! - "Native wide multiply reads both words of a constant-buffer addend" with a carry output
//!   (`output` 0): `IMAD.WIDE`'s bits 81..84 are a fixed `PT` field in the decoder, as the
//!   native visitor requires (`Require(Take(81, 3) == 7, "native wide multiply carry
//!   output")`), so that word does not decode.
//! - "Native constant integer absolute values preserve signed boundaries", the unaligned
//!   constant address the lowering refuses: the native visitor reads it from two words
//!   (`Visitor::Cbuf`), as this module does (`alu::unaligned_constant_buffer_reads_span_two_words`).

use super::*;
use crate::decode::{
    IMadMode, OpIAbs, OpIAdd3X, OpIDp4, OpIMad, OpIMad64, OpLea, OpLeaX, OpP2R, OpPrmt, OpR2P,
    PrmtMode,
};

/// "Native three-input addition produces both cumulative carry predicates": `IADD3` and
/// `IADD3.X` with carry outputs P0 and P1, every combination of negated sources, a register or
/// constant-buffer second source, five destinations and four carry-input codes.
#[test]
fn three_input_addition_produces_both_cumulative_carry_predicates() {
    let cases: [[u32; 3]; 8] = [
        [0, 0, 0],
        [0xffff_ffff, 0xffff_ffff, 0xffff_ffff],
        [0xffff_ffff, 1, 0],
        [0xffff_ffff, 0xffff_ffff, 1],
        [0, 1, 0xffff_ffff],
        [1, 0xffff_ffff, 0],
        [0x8000_0000, 0x7fff_ffff, 1],
        [0x1234_5678, 0x8765_4321, 0xabcd_ef01],
    ];
    // The PR's carry codes: the low three bits name the predicate (7 is PT), bit 3 negates.
    let code = |code: u8| ps(code & 7, code & 8 != 0);
    for extended in [false, true] {
        for constant_b in [false, true] {
            for negatives in 0..8_u8 {
                for destination in [0_u8, 1, 2, 4, 255] {
                    for carry_mode in 0..4 {
                        let (first, second) = match carry_mode {
                            0 => (0, 1),
                            1 => (8, 1),
                            2 => (7, 15),
                            _ => (15, 7),
                        };
                        let src = |index: u8| {
                            let negated = negatives & (1 << index) != 0;
                            if index == 1 && constant_b {
                                AluSrc::CBuf {
                                    cbuf: CBufRef {
                                        buf: CBuf::Binding(3),
                                        offset: 0x100,
                                    },
                                    abs: false,
                                    neg: negated,
                                }
                            } else if negated {
                                neg(index)
                            } else {
                                reg(index)
                            }
                        };
                        let add = OpIAdd3 {
                            overflow: [p(0), p(1)],
                            carry: if extended {
                                [code(first), code(second)]
                            } else {
                                [NOT_PT; 2]
                            },
                            ..iadd3(destination, [src(0), src(1), src(2)])
                        };
                        let op = if extended {
                            Op::IAdd3X(OpIAdd3X { add })
                        } else {
                            Op::IAdd3(add)
                        };
                        let function = build(&[Instr::new(op)]).expect("translates");
                        for values in cases {
                            for carry_values in 0..4_u8 {
                                let mut machine = machine_with(&values);
                                if constant_b {
                                    let mut bytes = vec![0; 0x104];
                                    bytes[0x100..].copy_from_slice(&values[1].to_le_bytes());
                                    machine.cbufs.insert(3, bytes);
                                }
                                let preds = [
                                    carry_values & 1 != 0,
                                    carry_values & 2 != 0,
                                    true,
                                    false,
                                    true,
                                    false,
                                    true,
                                ];
                                machine.preds[..7].copy_from_slice(&preds);
                                let mut expected = 0_u64;
                                for (i, value) in (0_u8..).zip(values) {
                                    expected += if negatives & (1 << i) != 0 {
                                        u64::from(value ^ 0xffff_ffff) + u64::from(!extended)
                                    } else {
                                        u64::from(value)
                                    };
                                }
                                let carry = |c: u8| {
                                    let value = if c & 7 == 7 {
                                        true
                                    } else {
                                        preds[usize::from(c & 7)]
                                    };
                                    value != (c & 8 != 0)
                                };
                                if extended {
                                    expected += u64::from(carry(first)) + u64::from(carry(second));
                                }
                                let mut expected_preds = preds;
                                expected_preds[0] = expected >= 1 << 32;
                                expected_preds[1] = expected >= 2 << 32;
                                machine.run(&function).expect("runs");
                                if destination != 255 {
                                    assert_eq!(
                                        machine.regs[usize::from(destination)],
                                        expected as u32
                                    );
                                }
                                assert_eq!(machine.preds[..7], expected_preds);
                                for (i, value) in values.iter().enumerate() {
                                    if i != usize::from(destination) {
                                        assert_eq!(machine.regs[i], *value);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// "Native subtraction carry chains retain the full unsigned 64-bit difference":
/// `IADD3 Rd, P0, PT, R0, -R2, RZ` then `IADD3.X Rd+1, PT, PT, R1, ~R3, RZ, P0, !PT`.
#[test]
fn subtraction_carry_chains_retain_the_full_unsigned_64_bit_difference() {
    let values = [
        0_u64,
        1,
        0xffff_ffff,
        0x1_0000_0000,
        0x7fff_ffff_ffff_ffff,
        0x8000_0000_0000_0000,
        u64::MAX,
        0x1234_5678_8765_4321,
    ];
    for destination in [0_u8, 2, 4] {
        let low = Op::IAdd3(OpIAdd3 {
            overflow: [p(0), Pred::PT],
            ..iadd3(destination, [reg(0), neg(2), AluSrc::reg(Gpr::RZ)])
        });
        let high = Op::IAdd3X(OpIAdd3X {
            add: OpIAdd3 {
                carry: [ps(0, false), NOT_PT],
                ..iadd3(destination + 1, [reg(1), neg(3), AluSrc::reg(Gpr::RZ)])
            },
        });
        let function = build(&[Instr::new(low), Instr::new(high)]).expect("translates");
        for a in values {
            for b in values {
                let mut machine =
                    machine_with(&[a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32]);
                let preds = [true, false, true, false, true, false, true];
                machine.preds[..7].copy_from_slice(&preds);
                let mut expected_preds = preds;
                expected_preds[0] = a as u32 >= b as u32;
                machine.run(&function).expect("runs");
                let d = usize::from(destination);
                let result = u64::from(machine.regs[d + 1]) << 32 | u64::from(machine.regs[d]);
                assert_eq!(result, a.wrapping_sub(b), "{a:#x} - {b:#x}");
                assert_eq!(machine.preds[..7], expected_preds);
            }
        }
    }
}

/// "Native register-to-predicate writes only masked predicate bits": `R2P PR, R5.Bn, 0x7`.
#[test]
fn register_to_predicate_writes_only_masked_predicate_bits() {
    for byte in 0..4_u8 {
        let function = build(&[Instr::new(Op::R2P(OpR2P {
            src: r(5),
            byte,
            mask: 7,
        }))])
        .expect("translates");
        for value in 0..128_u32 {
            let mut machine = Machine::new();
            let shift = 8 * u32::from(byte);
            machine.regs[5] = (value << shift) | (0xa5a5_a5a5 & !(0xff << shift));
            machine.preds[..7].copy_from_slice(&[true, false, true, false, true, false, true]);
            machine.run(&function).expect("runs");
            for bit in 0..7 {
                let expected = if bit < 3 {
                    value & (1 << bit) != 0
                } else {
                    bit % 2 == 0
                };
                assert_eq!(
                    machine.preds[bit as usize], expected,
                    "byte {byte} value {value}"
                );
            }
        }
    }
}

/// "Native predicate packing merges only selected predicate bits": `P2R R19, PR, R2, mask`.
#[test]
fn predicate_packing_merges_only_selected_predicate_bits() {
    for mask in [0_u8, 8, 16, 0x55, 0x7f] {
        let function = build(&[Instr::new(Op::P2R(OpP2R {
            dst: r(19),
            src: r(2),
            mask,
        }))])
        .expect("translates");
        for bits in 0..128_u32 {
            let mut machine = Machine::new();
            machine.regs[2] = 0xa5a5_a5a5;
            for (i, slot) in machine.preds.iter_mut().take(7).enumerate() {
                *slot = bits >> i & 1 != 0;
            }
            let saved = machine.preds;
            machine.run(&function).expect("runs");
            let mask = u32::from(mask);
            assert_eq!(machine.regs[19], (0xa5a5_a5a5 & !mask) | (bits & mask));
            assert_eq!(machine.preds, saved);
        }
    }
}

/// "Native integer absolute values preserve the minimum integer and source aliases" and
/// "Native constant integer absolute values preserve signed boundaries": `IABS Rd, R17` and
/// `IABS Rd, c[3][0x4]`.
#[test]
fn integer_absolute_values_preserve_the_minimum_integer() {
    for destination in [5_u8, 17] {
        for constant in [false, true] {
            let src = if constant { cb(3, 4) } else { reg(17) };
            let function = build(&[Instr::new(Op::IAbs(OpIAbs {
                dst: r(destination),
                src,
            }))])
            .expect("translates");
            for bits in [0_u32, 1, 0xffff_ffff, 0x8000_0000, 0x7fff_ffff, 0xfedc_ba98] {
                let mut machine = Machine::new();
                if constant {
                    with_cbuf(&mut machine, 3, &[0, bits]);
                } else {
                    machine.regs[17] = bits;
                }
                machine.regs[18] = 0x1234_5678;
                machine.run(&function).expect("runs");
                let expected = if bits & 0x8000_0000 != 0 {
                    0_u32.wrapping_sub(bits)
                } else {
                    bits
                };
                assert_eq!(machine.regs[usize::from(destination)], expected);
                assert_eq!(machine.regs[18], 0x1234_5678);
            }
        }
    }
}

/// "Native byte permutation retains selectors repeated bytes and input aliases":
/// `PRMT Rd, R0, selector, R1` (or `RZ`).
#[test]
fn byte_permutation_retains_selectors_and_aliases() {
    let selectors = [
        0x4431_u32, 0x7531, 0x0000, 0x7777, 0x3210, 0x7654, 0x1357, 0x6420,
    ];
    let inputs = [0_u32, 0xffff_ffff, 0x0123_4567, 0x89ab_cdef, 0x8000_0000];
    for selector in selectors {
        for destination in [0_u8, 1, 2] {
            for zero_second in [false, true] {
                let b = if zero_second {
                    AluSrc::reg(Gpr::RZ)
                } else {
                    reg(1)
                };
                let op = OpPrmt {
                    dst: r(destination),
                    srcs: [reg(0), b],
                    sel: AluSrc::Imm(selector),
                    mode: PrmtMode::Index,
                };
                let function = build(&[Instr::new(Op::Prmt(op))]).expect("translates");
                for a in inputs {
                    for b in inputs {
                        let mut machine = machine_with(&[a, b]);
                        let bytes = u64::from(if zero_second { 0 } else { b }) << 32 | u64::from(a);
                        let expected = (0..4).fold(0_u32, |acc, byte| {
                            let pick = (selector >> (byte * 4)) & 7;
                            acc | (((bytes >> (pick * 8)) & 0xff) as u32) << (byte * 8)
                        });
                        machine.run(&function).expect("runs");
                        assert_eq!(machine.regs[usize::from(destination)], expected);
                    }
                }
            }
        }
    }
}

/// "Native packed dot product selects unsigned halves bytes and accumulator":
/// `IDP.2A.LO.U16.U8 Rd, R0, R1, R2` (or `RZ`).
#[test]
fn packed_dot_product_selects_unsigned_halves_and_bytes() {
    for destination in [0_u8, 1, 2, 4] {
        for zero_accumulator in [false, true] {
            let c = if zero_accumulator {
                AluSrc::reg(Gpr::RZ)
            } else {
                reg(2)
            };
            let op = OpIDp4 {
                dst: r(destination),
                srcs: [reg(0), reg(1), c],
                signed: [false, false],
                two_a: true,
            };
            let function = build(&[Instr::new(Op::IDp4(op))]).expect("translates");
            for a in [0_u32, 0x0001_ffff, 0xffff_0001, 0x8000_8000, 0xffff_ffff] {
                for b in [0_u32, 0x0102_0304, 0xabcd_0102, 0xffff_ffff] {
                    for c in [0_u32, 1, 0xffff_ffff] {
                        let mut machine = machine_with(&[a, b, c]);
                        let expected = (a & 0xffff)
                            .wrapping_mul(b & 0xff)
                            .wrapping_add((a >> 16).wrapping_mul((b >> 8) & 0xff))
                            .wrapping_add(if zero_accumulator { 0 } else { c });
                        machine.run(&function).expect("runs");
                        assert_eq!(machine.regs[usize::from(destination)], expected);
                    }
                }
            }
        }
    }
}

/// "Native wide multiply reads both words of a constant-buffer addend", without a carry
/// output: `IMAD.WIDE[.U32] Rd, R0, R2, c[0][0x200]`.
#[test]
fn wide_multiply_reads_both_words_of_a_constant_buffer_addend() {
    for destination in [0_u8, 2, 4] {
        for signed in [false, true] {
            let op = OpIMad64 {
                dst: r(destination),
                srcs: [reg(0), reg(2), cb(0, 0x200)],
                signed,
                carry: NOT_PT,
            };
            let function = build(&[Instr::new(Op::IMad64(op))]).expect("translates");
            for a in [0_u32, 1, 0x8000_0000, 0xffff_ffff] {
                for b in [0_u32, 1, 0x8765_4321, 0xffff_ffff] {
                    for c in [0_u64, 0xffff_ffff, 0x1234_5678_8765_4321, u64::MAX] {
                        let mut machine = machine_with(&[a, 0, b]);
                        let mut words = vec![0; 0x80];
                        words.extend([c as u32, (c >> 32) as u32]);
                        with_cbuf(&mut machine, 0, &words);
                        let product = if signed {
                            (i64::from(a as i32) * i64::from(b as i32)) as u64
                        } else {
                            u64::from(a) * u64::from(b)
                        };
                        machine.run(&function).expect("runs");
                        let d = usize::from(destination);
                        let result =
                            u64::from(machine.regs[d + 1]) << 32 | u64::from(machine.regs[d]);
                        assert_eq!(result, product.wrapping_add(c));
                    }
                }
            }
        }
    }
}

/// "Native high LEA adds the base after unsigned scaling": `LEA.HI[.SX32] R0, R0, R1 (or UR4),
/// RZ, shift`.
#[test]
fn high_lea_adds_the_base_after_scaling() {
    for uniform in [false, true] {
        for signed_scale in [false, true] {
            for shift in [0_u8, 1, 16, 28, 31] {
                let op = OpLea {
                    dst: r(0),
                    overflow: Pred::PT,
                    a: reg(0),
                    b: if uniform { ureg(4) } else { reg(1) },
                    a_high: AluSrc::reg(Gpr::RZ),
                    shift,
                    dst_high: true,
                    sx32: signed_scale,
                    intermediate: false,
                    carry: NOT_PT,
                };
                let function = build(&[Instr::new(Op::Lea(op))]).expect("translates");
                for value in [0_u32, 1, 0x8000_0000, 0xffff_ffff] {
                    for base in [0_u32, 1, 0xffff_ffff] {
                        let mut machine = machine_with(&[value, base]);
                        machine.uniform_regs[4] = base;
                        machine.run(&function).expect("runs");
                        let extended = if signed_scale {
                            i64::from(value as i32) as u64
                        } else {
                            u64::from(value)
                        };
                        let expected = ((extended << shift) >> 32) as u32;
                        assert_eq!(machine.regs[0], expected.wrapping_add(base));
                    }
                }
            }
        }
    }
}

/// "Native LEA predicate carry reaches the high pointer word": `LEA R0, P0, R0, c[0][0x20],
/// 0x2` then `IADD3.X R1, RZ, c[0][0x24], RZ, P0, !PT` (the PR's store of the pointer is the
/// memory group's).
#[test]
fn lea_predicate_carry_reaches_the_high_pointer_word() {
    let lea = Op::Lea(OpLea {
        dst: r(0),
        overflow: p(0),
        a: reg(0),
        b: cb(0, 0x20),
        a_high: AluSrc::reg(Gpr::RZ),
        shift: 2,
        dst_high: false,
        sx32: false,
        intermediate: false,
        carry: NOT_PT,
    });
    let high = Op::IAdd3X(OpIAdd3X {
        add: OpIAdd3 {
            carry: [ps(0, false), NOT_PT],
            ..iadd3(1, [AluSrc::reg(Gpr::RZ), cb(0, 0x24), AluSrc::reg(Gpr::RZ)])
        },
    });
    let function = build(&[Instr::new(lea), Instr::new(high)]).expect("translates");
    for base in [0x100_u32, 0xffff_fff0] {
        let mut machine = machine_with(&[8]);
        let mut words = vec![0; 8];
        words.extend([base, 0x21]);
        with_cbuf(&mut machine, 0, &words);
        machine.run(&function).expect("runs");
        let pointer = u64::from(machine.regs[1]) << 32 | u64::from(machine.regs[0]);
        assert_eq!(pointer, (0x21_u64 << 32) + u64::from(base) + 32);
    }
}

/// "Native integer multiply preserves high words and modular multiply-add", its six modes:
/// `IMAD.HI.U32`, `IMAD` with an immediate, a constant-buffer and a register factor, `LEA` with
/// an immediate addend, and `IMAD` with a negated addend.
#[test]
fn integer_multiply_preserves_high_words_and_modular_multiply_add() {
    let imad = |srcs, signed, mode| {
        Op::IMad(OpIMad {
            dst: r(3),
            srcs,
            signed,
            mode,
        })
    };
    let lo = IMadMode::Lo {
        x: false,
        carry: NOT_PT,
    };
    let rz = AluSrc::reg(Gpr::RZ);
    let modes = [
        imad([reg(0), AluSrc::Imm(0xcccc_cccd), rz], false, IMadMode::Hi),
        imad([reg(0), AluSrc::Imm(0xffff_fff6), reg(2)], true, lo),
        imad([reg(0), cb(3, 0), reg(2)], true, lo),
        imad([reg(0), reg(1), reg(2)], true, lo),
        Op::Lea(OpLea {
            dst: r(3),
            overflow: Pred::PT,
            a: reg(0),
            b: AluSrc::Imm(0x70),
            a_high: rz,
            shift: 4,
            dst_high: false,
            sx32: false,
            intermediate: false,
            carry: NOT_PT,
        }),
        imad([reg(0), AluSrc::Imm(1), neg(2)], true, lo),
    ];
    for (mode, op) in modes.into_iter().enumerate() {
        let function = build(&[Instr::new(op)]).expect("translates");
        for a in [
            0_u32,
            1,
            9,
            10,
            65535,
            65536,
            0x8000_0000,
            0xffff_fffe,
            0xffff_ffff,
            0x1234_5678,
        ] {
            let mut machine = machine_with(&[a, 0x7654_3210, 0xfedc_ba98]);
            with_cbuf(&mut machine, 3, &[0x7654_3210]);
            machine.run(&function).expect("runs");
            let b: u32 = match mode {
                0 => 0xcccc_cccd,
                1 => 0xffff_fff6,
                _ => 0x7654_3210,
            };
            let product = u64::from(a) * u64::from(b);
            let expected = match mode {
                5 => a.wrapping_sub(0xfedc_ba98),
                4 => (a << 4).wrapping_add(0x70),
                0 => (product >> 32) as u32,
                _ => (product as u32).wrapping_add(0xfedc_ba98),
            };
            assert_eq!(machine.regs[3], expected, "mode {mode} a {a:#x}");
        }
    }
}

/// `LEA.X` in the PR's carry tests is the same as `LEA` with a carry input: the PR's
/// "Native LEA predicate carry" chain written with `LEA.HI.X` for the high word.
#[test]
fn lea_hi_x_carries_into_the_high_word() {
    let low = Op::Lea(OpLea {
        dst: r(0),
        overflow: p(0),
        a: reg(0),
        b: reg(2),
        a_high: AluSrc::reg(Gpr::RZ),
        shift: 2,
        dst_high: false,
        sx32: false,
        intermediate: false,
        carry: NOT_PT,
    });
    let high = Op::LeaX(OpLeaX {
        lea: OpLea {
            dst: r(1),
            overflow: Pred::PT,
            a: reg(0),
            b: reg(3),
            a_high: reg(1),
            shift: 2,
            dst_high: true,
            sx32: false,
            intermediate: false,
            carry: ps(0, false),
        },
    });
    // The high word reads the old R0, so it runs first in the chain the compiler emits; here
    // the low word is written to R4 to keep R0.
    let low = match low {
        Op::Lea(lea) => Op::Lea(OpLea { dst: r(4), ..lea }),
        other => other,
    };
    let function = build(&[Instr::new(low), Instr::new(high)]).expect("translates");
    for (index, base) in [(0x8000_0000_u64, 0xffff_fff0_u64), (0x1_0000_0001, 0x10)] {
        let mut machine = machine_with(&[
            index as u32,
            (index >> 32) as u32,
            base as u32,
            (base >> 32) as u32,
        ]);
        machine.run(&function).expect("runs");
        let pointer = u64::from(machine.regs[1]) << 32 | u64::from(machine.regs[4]);
        assert_eq!(
            pointer,
            (index << 2).wrapping_add(base),
            "{index:#x} {base:#x}"
        );
    }
}
