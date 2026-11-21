//! The predicate and register files each twin reads and writes (the module docs, "Predicate
//! files"): the lowering's readings where it and the native translator differ, and the uniform
//! file elsewhere. Each case gives the two files different values, so a read or write of the
//! wrong one changes the result.

use super::alu::lea;
use super::*;
use crate::decode::{
    IMadMode, IntCmpOp, OpIAdd3X, OpIMad, OpISetP, OpLea, OpLeaX, OpLop3, OpSel, PredSetOp,
};

/// `ULOP3`'s predicate output is a uniform predicate (the lowering's `uniform_predicate_bank`
/// for `0x892`), where the native translator writes the per-thread predicate (`SetPred`).
#[test]
fn ulop3_writes_a_uniform_predicate() {
    for value in [0_u32, 0x20000] {
        let mut machine = machine_with(&[0, value, 0x20000]);
        machine.preds[2] = value == 0;
        machine.uniform_preds[2] = value == 0;
        run_op(
            Op::ULop3(twin(OpLop3 {
                pred: p(2),
                dst: Gpr::RZ,
                srcs: [reg(1), reg(2), RZ],
                lut: 0xc0,
            })),
            &mut machine,
        );
        assert_eq!(machine.uniform_preds[2], value != 0, "{value:#x}");
        assert_eq!(machine.preds[2], value == 0, "P2 is untouched: {value:#x}");
    }
}

/// `ULEA.X`'s carry input and `ULEA`'s carry output are per-thread predicates (the lowering
/// rewrites `0x291`/`0x891` to `LEA` outside `uniform_predicate_bank`), where the native
/// translator reads and writes uniform predicates.
#[test]
fn ulea_carries_are_predicates() {
    for carry in [false, true] {
        let mut machine = machine_with(&[0, 0xffff_ffff, 1]);
        machine.preds[1] = carry;
        machine.uniform_preds[1] = !carry;
        run_op(
            Op::ULeaX(twin(OpLeaX {
                lea: OpLea {
                    carry: ps(1, false),
                    ..lea(4, reg(1), reg(2), 0)
                },
            })),
            &mut machine,
        );
        assert_eq!(machine.uniform_regs[4], u32::from(carry), "{carry}");
    }
    for (a, carried) in [(0xffff_ffff_u32, true), (1, false)] {
        let mut machine = machine_with(&[0, a, 1]);
        machine.uniform_preds[3] = !carried;
        machine.preds[3] = !carried;
        run_op(
            Op::ULea(twin(OpLea {
                overflow: p(3),
                ..lea(4, reg(1), reg(2), 0)
            })),
            &mut machine,
        );
        assert_eq!(machine.preds[3], carried, "{a:#x}");
        assert_eq!(
            machine.uniform_preds[3], !carried,
            "UP3 is untouched: {a:#x}"
        );
    }
}

/// `UIMAD.X`'s carry input is a per-thread predicate (the lowering rewrites `0x2a4`/`0x8a4` to
/// `IMAD`), where the native translator reads a uniform predicate.
#[test]
fn uimad_x_carry_is_a_predicate() {
    for carry in [false, true] {
        let mut machine = machine_with(&[0, 3, 5, 7]);
        machine.preds[2] = carry;
        machine.uniform_preds[2] = !carry;
        run_op(
            Op::UIMad(twin(OpIMad {
                dst: r(4),
                srcs: [reg(1), reg(2), reg(3)],
                signed: false,
                mode: IMadMode::Lo {
                    x: true,
                    carry: ps(2, false),
                },
            })),
            &mut machine,
        );
        assert_eq!(machine.uniform_regs[4], 22 + u32::from(carry), "{carry}");
    }
}

/// `ULEA.HI` writes its uniform destination, as the lowering does, where the native
/// translator's `.HI` path writes the per-thread register of the same number.
#[test]
fn ulea_hi_writes_a_uniform_register() {
    let mut machine = machine_with(&[0, 0x8000_0000, 0x10, 0x3]);
    run_op(
        Op::ULea(twin(OpLea {
            a_high: reg(3),
            dst_high: true,
            ..lea(4, reg(1), reg(2), 1)
        })),
        &mut machine,
    );
    // (0x3:0x8000_0000) << 1 has the high word 0x7, plus 0x10.
    assert_eq!(machine.uniform_regs[4], 0x17);
    assert_eq!(machine.regs[4], 0xbad0_0004, "R4 is untouched");
}

/// `UIADD3`'s and `UIADD3.X`'s carries, `UISETP`'s accumulator and results and `USEL`'s
/// condition are uniform predicates in both references.
#[test]
fn the_other_twins_use_uniform_predicates() {
    // UIADD3.X: the carry input UP1 (P1 the opposite).
    for carry in [false, true] {
        let mut machine = machine_with(&[0, 1, 2]);
        machine.uniform_preds[1] = carry;
        machine.preds[1] = !carry;
        run_op(
            Op::UIAdd3X(twin(OpIAdd3X {
                add: OpIAdd3 {
                    carry: [ps(1, false), NOT_PT],
                    ..iadd3(4, [reg(1), reg(2), RZ])
                },
            })),
            &mut machine,
        );
        assert_eq!(machine.uniform_regs[4], 3 + u32::from(carry), "{carry}");
    }
    // UISETP: the accumulator UP3, the results UP1 and UP2.
    for accum in [false, true] {
        let mut machine = machine_with(&[0, 1, 2]);
        machine.uniform_preds[3] = accum;
        machine.preds = [true; 8];
        let preds = machine.preds;
        run_op(
            Op::UISetP(twin(OpISetP {
                dsts: [p(1), p(2)],
                srcs: [reg(1), reg(2)],
                cmp_op: IntCmpOp::Lt,
                signed: false,
                set_op: PredSetOp::Or,
                ex: false,
                accum: ps(3, false),
                low_cmp: PredSrc::TRUE,
            })),
            &mut machine,
        );
        assert!(machine.uniform_preds[1], "1 < 2");
        assert_eq!(machine.uniform_preds[2], accum);
        assert_eq!(machine.preds, preds);
    }
    // USEL: the condition UP1.
    for cond in [false, true] {
        let mut machine = machine_with(&[0, 0x1111, 0x2222]);
        machine.uniform_preds[1] = cond;
        machine.preds[1] = !cond;
        run_op(
            Op::USel(twin(OpSel {
                dst: r(4),
                srcs: [reg(1), reg(2)],
                cond: ps(1, false),
            })),
            &mut machine,
        );
        assert_eq!(machine.uniform_regs[4], if cond { 0x1111 } else { 0x2222 });
    }
}

/// Every twin's register operands name uniform registers: each register a twin reads holds a
/// different value in the per-thread file, and no per-thread register is written.
#[test]
fn twins_read_and_write_the_uniform_registers() {
    let mut machine = machine_with(&[0, 10, 20, 30]);
    let regs = machine.regs;
    run_op(
        Op::UIAdd3(twin(iadd3(4, [reg(1), reg(2), reg(3)]))),
        &mut machine,
    );
    assert_eq!(machine.uniform_regs[4], 60);
    assert_eq!(machine.regs, regs);
    let function = build(&[Instr::new(Op::UIAdd3(twin(iadd3(
        4,
        [reg(1), reg(2), reg(3)],
    ))))])
    .expect("translates");
    assert_eq!(count(&function, Opcode::GetRegister), 0);
    assert_eq!(count(&function, Opcode::SetRegister), 0);
    assert_eq!(count(&function, Opcode::GetUniformRegister), 3);
    assert_eq!(count(&function, Opcode::SetUniformRegister), 1);
}
