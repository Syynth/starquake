//! The opcode classes against `classify.h`'s rows written out as literals, and the control
//! classes against what the decoder makes of each opcode.

use super::super::classify::{OpInfo, OpKind, classify, is_warp_wide};
use super::{b, bsync, exit, kill, offset, p, pure, r, to_start};
use crate::asm::assemble;
use crate::decode::{
    BraCond, BrxSrc, Instr, Op, OpBSSy, OpBra, OpBreak, OpBrx, OpCall, OpRet, PredSrc, decode,
};

const fn info(kind: OpKind, pure: bool, uniform: bool) -> OpInfo {
    OpInfo {
        kind,
        pure,
        uniform,
    }
}

#[test]
fn classify_matches_the_reference() {
    use OpKind::{Bra, Brx, Bssy, Bsync, Call, Exit, Kill, Nop, Plain, Ret};
    // `classify.h` @ e5fb3b97, `Classify`'s exact rows.
    let exact: &[(u16, OpInfo)] = &[
        (0x947, info(Bra, false, false)),
        (0x94d, info(Exit, false, false)),
        (0x95b, info(Kill, false, false)),
        (0x945, info(Bssy, false, false)),
        (0x941, info(Bsync, false, false)),
        (0x949, info(Brx, false, false)),
        (0x958, info(Brx, false, false)),
        (0x944, info(Call, false, false)),
        (0x950, info(Ret, false, false)),
        (0x942, info(Nop, false, false)),
        (0x948, info(Nop, false, false)),
        (0x918, info(Nop, false, false)),
        (0x91a, info(Nop, false, false)),
        (0x326, info(Plain, true, false)),
        (0x321, info(Plain, true, false)),
        (0xb82, info(Plain, true, false)),
        (0x805, info(Plain, true, false)),
        (0x301, info(Plain, true, false)),
        (0x822, info(Plain, true, false)),
        (0x3c2, info(Plain, true, false)),
        (0x919, info(Plain, true, false)),
        (0x803, info(Plain, true, false)),
        (0x804, info(Plain, true, false)),
        (0x81c, info(Plain, true, false)),
        (0x882, info(Plain, true, true)),
        (0xc82, info(Plain, true, true)),
        (0xab9, info(Plain, true, true)),
        (0xabb, info(Plain, true, true)),
        (0x9c3, info(Plain, true, true)),
        (0x89c, info(Plain, true, true)),
        (0x886, info(Plain, true, true)),
        // This crate's one change: the uniform `BRA` is a branch (module docs).
        (0x547, info(Bra, false, false)),
    ];
    for &(opcode, expected) in exact {
        assert_eq!(classify(opcode), expected, "{opcode:#x}");
    }
    // The uniform ALU bases, pure and uniform in forms 1 and 4 only.
    let uniform_bases = [
        0x090, 0x091, 0x092, 0x099, 0x0a4, 0x0a5, 0x0be, 0x09a, 0x0bd, 0x0bf, 0x08c, 0x087,
    ];
    for base in uniform_bases {
        for form in 0_u16..8 {
            let opcode = base | form << 9;
            let expected = match form {
                1 | 4 => info(Plain, true, true),
                _ => info(Plain, false, false),
            };
            if exact.iter().all(|&(row, _)| row != opcode) {
                assert_eq!(classify(opcode), expected, "{opcode:#x}");
            }
        }
    }
    // The pure bases, in every form but 0.
    let pure_bases = [
        0x020, 0x108, 0x002, 0x021, 0x023, 0x009, 0x007, 0x008, 0x00b, 0x00a, 0x040, 0x00c, 0x010,
        0x012, 0x019, 0x011, 0x017, 0x016, 0x030, 0x031, 0x032, 0x024, 0x027, 0x025, 0x013, 0x100,
        0x109, 0x01a, 0x034, 0x03e, 0x105, 0x106, 0x045, 0x107, 0x104,
    ];
    for base in pure_bases {
        for form in 0_u16..8 {
            let opcode = base | form << 9;
            if exact.iter().any(|&(row, _)| row == opcode) {
                continue;
            }
            let expected = if form == 0 {
                info(Plain, false, false)
            } else {
                info(Plain, true, false)
            };
            assert_eq!(classify(opcode), expected, "{opcode:#x}");
        }
    }
    // Everything else is a plain instruction with side effects: every opcode not covered
    // above.
    let covered = |opcode: u16| {
        exact.iter().any(|&(row, _)| row == opcode)
            || (opcode >> 9 != 0 && pure_bases.contains(&(opcode & 0x1ff)))
            || (matches!(opcode >> 9, 1 | 4) && uniform_bases.contains(&(opcode & 0x1ff)))
    };
    for opcode in 0_u16..0x1000 {
        if !covered(opcode) {
            assert_eq!(classify(opcode), info(Plain, false, false), "{opcode:#x}");
        }
    }
}

#[test]
fn warp_wide_opcodes_are_the_reference_four() {
    // `flow.cpp`: `op == 0xb1d || op == 0x83b || op == 0x23a || op == 0x23c`.
    for opcode in 0_u16..0x1000 {
        assert_eq!(
            is_warp_wide(opcode),
            [0xb1d, 0x83b, 0x23a, 0x23c].contains(&opcode),
            "{opcode:#x}"
        );
    }
}

#[test]
fn control_classes_match_the_decoded_operations() {
    // An instruction of each control operation, assembled: its opcode's class names the
    // operation the decoder reads back.
    let cases: Vec<(Instr, OpKind)> = vec![
        (
            Instr::new(Op::Bra(OpBra {
                cond: BraCond::Pred(PredSrc::TRUE),
                uniform: false,
                offset: offset(0, 1),
            })),
            OpKind::Bra,
        ),
        (exit(), OpKind::Exit),
        (kill(PredSrc::TRUE), OpKind::Kill),
        (
            Instr::new(Op::BSSy(OpBSSy {
                bar: b(1),
                offset: offset(0, 1),
                cond: PredSrc::TRUE,
            })),
            OpKind::Bssy,
        ),
        (bsync(1), OpKind::Bsync),
        (
            Instr::new(Op::Brx(OpBrx {
                src: BrxSrc::Reg(r(2)),
                offset: to_start(0),
                cond: PredSrc::TRUE,
            })),
            OpKind::Brx,
        ),
        (
            Instr::new(Op::Call(OpCall {
                offset: offset(0, 1),
                no_inc: true,
                cond: PredSrc::TRUE,
            })),
            OpKind::Call,
        ),
        (
            Instr::new(Op::Ret(OpRet {
                src: r(10),
                offset: to_start(0),
                no_dec: true,
                cond: PredSrc::TRUE,
            })),
            OpKind::Ret,
        ),
        (
            Instr {
                guard: p(0),
                op: Op::Break(OpBreak {
                    bar: b(0),
                    cond: PredSrc::TRUE,
                }),
            },
            OpKind::Nop,
        ),
        (pure(1), OpKind::Plain),
    ];
    for (instr, kind) in cases {
        let word = assemble(&instr).expect("encodable");
        let opcode = super::super::analysis::opcode_of(word);
        assert_eq!(classify(opcode).kind, kind, "{}", instr.op);
        assert_eq!(
            decode(word).expect("decodes").op.mnemonic(),
            instr.op.mnemonic()
        );
    }
}
