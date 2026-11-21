//! PR #4588's cases for this group (`eden-pr4588/src/tests/video_core/native_shader.cpp` @
//! e5fb3b97), rebuilt with the test assembler: the PR's words may come from games, so each
//! case is the same instruction built as an [`Instr`] and the same expectations, with the PR's
//! packed register numbers (`216 + n` for `URn`, bit `6 - n` of register 239 for `UPn`) read as
//! the uniform registers and predicates they stand for.

use super::alu::lea;
use super::*;
use crate::decode::{OpBRev, OpIAdd3X, OpLop3, OpPrmt, PrmtMode};

/// "Native uniform bit tests update one uniform predicate and preserve scalar predicates":
/// `ULOP3.LUT URZ, UPn, UR10, mask, URZ, 0xc0` sets `UPn` to whether `UR10 & mask` is not
/// zero, for every `n` below 7, leaving the other uniform predicates, the per-thread
/// predicates and `UR10` as they were.
#[test]
fn uniform_bit_tests() {
    for mask in [0_u32, 1, 0x20000, 0x40000, 0xffff_ffff] {
        for output in 0..7_u8 {
            let instr = Instr::new(Op::ULop3(twin(OpLop3 {
                pred: p(output),
                dst: Gpr::RZ,
                srcs: [reg(10), AluSrc::Imm(mask), RZ],
                lut: 0xc0,
            })));
            let function = build(&[instr]).expect("translates");
            for value in [0_u32, 1, 2, 0x20000, 0x40000, 0x8000_0000, 0xffff_ffff] {
                for bank in [0_u8, 0x55, 0x7f] {
                    let mut machine = machine_with(&[]);
                    machine.uniform_regs[10] = value;
                    for (n, slot) in machine.uniform_preds.iter_mut().take(7).enumerate() {
                        *slot = bank >> n & 1 != 0;
                    }
                    machine.preds = [true, false, true, false, true, false, true, false];
                    let original = machine.clone();
                    machine.run(&function).expect("runs");
                    let mut expected = original.uniform_preds;
                    expected[usize::from(output)] = value & mask != 0;
                    assert_eq!(
                        machine.uniform_preds, expected,
                        "{mask:#x} {output} {value:#x}"
                    );
                    assert_eq!(machine.preds, original.preds);
                    assert_eq!(machine.uniform_regs[10], value);
                }
            }
        }
    }
}

/// "Native uniform carry chains preserve 64-bit overflow and both predicate banks":
/// `UIADD3 UR4, UPn, UR4, UR8, URZ` and `UIADD3.X UR5, URZ, UR9, URZ, UPn` give `UR4` the low
/// word of `UR4 + UR8`, `UR5` `UR9` plus its carry, and `UPn` the carry, for every `n` below
/// 7, leaving the per-thread predicates and `UR8` as they were.
#[test]
fn uniform_carry_chains() {
    for output in 0..7_u8 {
        let chain = [
            Instr::new(Op::UIAdd3(twin(OpIAdd3 {
                overflow: [p(output), Pred::PT],
                ..iadd3(4, [reg(4), reg(8), RZ])
            }))),
            Instr::new(Op::UIAdd3X(twin(OpIAdd3X {
                add: OpIAdd3 {
                    carry: [ps(output, false), NOT_PT],
                    ..iadd3(5, [RZ, reg(9), RZ])
                },
            }))),
        ];
        let function = build(&chain).expect("translates");
        for a in [0_u32, 1, 0xffff_fff0, 0xffff_ffff] {
            for b in [0_u32, 1, 32, 0xffff_ffff] {
                for high in [0_u32, 0xffff_ffff] {
                    let mut machine = machine_with(&[]);
                    machine.uniform_regs[4] = a;
                    machine.uniform_regs[8] = b;
                    machine.uniform_regs[9] = high;
                    machine.preds = [true, false, true, false, true, false, true, false];
                    let preds = machine.preds;
                    machine.run(&function).expect("runs");
                    let sum = u64::from(a) + u64::from(b);
                    let carry = u32::try_from(sum >> 32).expect("a carry");
                    let low = u32::try_from(sum & u64::from(u32::MAX)).expect("low");
                    assert_eq!(machine.uniform_regs[4], low, "{output} {a} {b}");
                    assert_eq!(machine.uniform_regs[5], high.wrapping_add(carry));
                    assert_eq!(machine.uniform_preds[usize::from(output)], carry != 0);
                    assert_eq!(machine.preds, preds);
                    assert_eq!(machine.uniform_regs[8], b);
                }
            }
        }
    }
}

/// "Native uniform register effective addresses preserve scaled addition and aliasing":
/// `ULEA UR4, UR5, UR8, shift` gives `(UR5 << shift) + UR8`, leaving both sources.
#[test]
fn uniform_effective_addresses() {
    for shift in [0_u8, 1, 2, 16, 31] {
        let function =
            build(&[Instr::new(Op::ULea(twin(lea(4, reg(5), reg(8), shift))))]).expect("ok");
        for value in [0_u32, 1, 0x8000_0000, 0xffff_ffff] {
            let mut machine = machine_with(&[]);
            machine.uniform_regs[5] = value;
            machine.uniform_regs[8] = 0xffff_fff0;
            machine.run(&function).expect("runs");
            assert_eq!(
                machine.uniform_regs[4],
                (value << shift).wrapping_add(0xffff_fff0),
                "{shift} {value:#x}"
            );
            assert_eq!(machine.uniform_regs[5], value);
            assert_eq!(machine.uniform_regs[8], 0xffff_fff0);
        }
    }
}

/// "Native uniform permutation sign extends its low half and supports aliasing":
/// `UPRMT URd, UR8, 0x9910, URZ` gives the low half-word of `UR8` sign-extended, into `UR6`
/// and into `UR8` itself.
#[test]
fn uniform_permutation_sign_extends() {
    for destination in [6_u8, 8] {
        let function = build(&[Instr::new(Op::UPrmt(twin(OpPrmt {
            dst: r(destination),
            srcs: [reg(8), RZ],
            sel: AluSrc::Imm(0x9910),
            mode: PrmtMode::Index,
        })))])
        .expect("translates");
        for value in [0_u32, 1, 0x7fff, 0x8000, 0xffff, 0xdead_8000, 0xabcd_7fff] {
            let mut machine = machine_with(&[]);
            machine.uniform_regs[8] = value;
            machine.run(&function).expect("runs");
            let half = u16::try_from(value & 0xffff).expect("a half-word");
            let expected = i32::from(half.cast_signed()).cast_unsigned();
            assert_eq!(machine.uniform_regs[usize::from(destination)], expected);
            if destination != 8 {
                assert_eq!(machine.uniform_regs[8], value);
            }
        }
    }
}

/// "Native uniform bit reverse preserves every bit and permits source destination aliases":
/// `UBREV URd, UR4` reverses every bit, into each destination the PR checks (`UR4` itself
/// among them); from `URZ` it gives zero.
#[test]
fn uniform_bit_reverse() {
    let mut inputs = vec![0_u32, !0, 0x0123_4567, 0x89ab_cdef];
    inputs.extend((0..32).map(|bit| 1_u32 << bit));
    for dest in [4_u8, 17, 18, 30, 32, 46, 48, 62] {
        let brev = |src: AluSrc| {
            build(&[Instr::new(Op::UBRev(twin(OpBRev { dst: r(dest), src })))]).expect("translates")
        };
        let function = brev(reg(4));
        for value in &inputs {
            let mut machine = machine_with(&[]);
            machine.uniform_regs[4] = *value;
            machine.run(&function).expect("runs");
            assert_eq!(
                machine.uniform_regs[usize::from(dest)],
                value.reverse_bits(),
                "{dest} {value:#x}"
            );
        }
        let mut machine = machine_with(&[]);
        machine.uniform_regs[usize::from(dest)] = 1;
        machine.run(&brev(RZ)).expect("runs");
        assert_eq!(machine.uniform_regs[usize::from(dest)], 0);
    }
}
