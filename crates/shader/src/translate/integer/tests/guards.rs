//! Guards: a conditional block for a guard other than `PT`, nothing for `!PT`, the guard
//! ignored once tested, and refusals that leave the function as it was.

use super::*;
use crate::decode::{IntCmpOp, MovSrc, OpBMsk, OpISetP, PredSetOp};

fn mov(dst: u8, src: u8) -> Op {
    Op::Mov(OpMov {
        dst: r(dst),
        src: MovSrc::Reg(r(src)),
        quad_lanes: OpMov::ALL_LANES,
    })
}

fn guarded(op: Op, guard: PredSrc) -> Instr {
    Instr { guard, op }
}

/// `@P1 MOV R1, R2` tests P1 in the block, runs the move in a block of its own and continues
/// in a third block that holds the first block's terminator; it moves only when P1 holds.
#[test]
fn a_guard_makes_a_conditional_block() {
    let program = [guarded(mov(1, 2), ps(1, false))];
    let function = build(&program).expect("translates");
    assert_eq!(function.verify(), Ok(()));
    assert_eq!(
        function.to_string(),
        "block0:\n\
         \x20   %0: u1 = GetPred p1\n\
         \x20   branch %0, block1, block2\n\
         block1:\n\
         \x20   %1: u32 = GetRegister r2\n\
         \x20   %2 = SetRegister r1, %1\n\
         \x20   jump block2\n\
         block2:\n\
         \x20   return\n"
    );
    for (cond, expected) in [(false, 7), (true, 9)] {
        let mut machine = machine_with(&[0, 7, 9]);
        machine.preds[1] = cond;
        machine.run(&function).expect("runs");
        assert_eq!(machine.regs[1], expected);
    }
}

/// A negated guard runs the instruction where its predicate is false.
#[test]
fn a_negated_guard_runs_where_its_predicate_is_false() {
    for (cond, expected) in [(false, 9), (true, 7)] {
        let mut machine = machine_with(&[0, 7, 9]);
        machine.preds[3] = cond;
        run(&[guarded(mov(1, 2), ps(3, true))], &mut machine);
        assert_eq!(machine.regs[1], expected);
    }
}

/// Two guarded instructions in a row make two conditional blocks, each testing its own guard.
#[test]
fn consecutive_guards_chain_their_blocks() {
    for preds in 0..4_u8 {
        let mut machine = machine_with(&[0, 0, 5, 6]);
        machine.preds[0] = preds & 1 != 0;
        machine.preds[1] = preds & 2 != 0;
        run(
            &[
                guarded(mov(0, 2), ps(0, false)),
                guarded(mov(1, 3), ps(1, false)),
            ],
            &mut machine,
        );
        assert_eq!(machine.regs[0], if preds & 1 != 0 { 5 } else { 0 });
        assert_eq!(machine.regs[1], if preds & 2 != 0 { 6 } else { 0 });
    }
}

/// An instruction that writes its own guard predicate reads it first: the guard decides, and
/// the write lands only where it held.
#[test]
fn an_instruction_may_write_its_own_guard() {
    let isetp = Op::ISetP(OpISetP {
        dsts: [p(0), Pred::PT],
        srcs: [reg(1), reg(2)],
        cmp_op: IntCmpOp::Eq,
        signed: false,
        set_op: PredSetOp::And,
        ex: false,
        accum: PredSrc::TRUE,
        low_cmp: PredSrc::TRUE,
    });
    for guard in [false, true] {
        let mut machine = machine_with(&[0, 1, 2]);
        machine.preds[0] = guard;
        run(&[guarded(isetp, ps(0, false))], &mut machine);
        // 1 == 2 is false, so where the guard held P0 is cleared, else it stays clear.
        assert!(!machine.preds[0]);
    }
}

/// `@!PT` never runs: nothing is appended, not even for a form that would be refused.
#[test]
fn never_guarded_instructions_append_nothing() {
    let never = PredSrc {
        reg: Pred::PT,
        negated: true,
    };
    let bmsk = Op::BMsk(OpBMsk {
        dst: r(0),
        pos: reg(1),
        width: reg(2),
        wrap: false,
    });
    let partial = Op::Mov(OpMov {
        dst: r(1),
        src: MovSrc::Reg(r(2)),
        quad_lanes: 1,
    });
    for op in [mov(1, 2), bmsk, partial] {
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        let result = translate_integer(&mut builder, &guarded(op, never), GuardMode::Honour);
        assert_eq!(result, Ok(()));
        assert_eq!(builder.function(), &Function::new());
        assert_eq!(builder.block(), entry);
    }
    // The group check still comes first.
    let nop = Op::Nop(crate::decode::OpNop {});
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    assert_eq!(
        translate_integer(&mut builder, &guarded(nop, never), GuardMode::Honour),
        Err(IntegerError::OtherGroup("nop"))
    );
}

/// With [`GuardMode::Tested`] the guard is ignored: the instruction runs in the block, `@!PT`
/// and an unset predicate included.
#[test]
fn a_tested_guard_is_ignored() {
    for guard in [
        ps(1, false),
        PredSrc {
            reg: Pred::PT,
            negated: true,
        },
    ] {
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        translate_integer(&mut builder, &guarded(mov(1, 2), guard), GuardMode::Tested)
            .expect("translates");
        assert_eq!(builder.block(), entry);
        builder.terminate(Terminator::Return).expect("terminates");
        assert_eq!(function.block_count(), 1);
        let mut machine = machine_with(&[0, 7, 9]);
        machine.run(&function).expect("runs");
        assert_eq!(machine.regs[1], 9);
    }
}

/// The block after the conditional block takes the original block's terminator, here a jump
/// to a block that was already there, and the builder is left in it.
#[test]
fn the_split_keeps_the_blocks_terminator() {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let exit = builder.add_block().expect("adds");
    builder
        .terminate(Terminator::Jump(exit))
        .expect("terminates");
    builder.switch_to(exit).expect("switches");
    builder.terminate(Terminator::Return).expect("terminates");
    builder.switch_to(entry).expect("switches");
    translate_integer(
        &mut builder,
        &guarded(mov(1, 2), ps(2, false)),
        GuardMode::Honour,
    )
    .expect("translates");
    let join = builder.block();
    assert_ne!(join, entry);
    assert_eq!(
        function.block(join).map(|block| block.terminator),
        Some(Terminator::Jump(exit))
    );
    assert_eq!(function.verify(), Ok(()));
    let mut machine = machine_with(&[0, 7, 9]);
    machine.preds[2] = true;
    machine.run(&function).expect("runs");
    assert_eq!(machine.regs[1], 9);
}

/// A guarded instruction that is refused leaves the function and the builder as they were,
/// whichever check refuses it (the bindless check before the instruction's own, and checks
/// inside the instruction's translation), and the next instruction translates normally.
#[test]
fn refusals_leave_the_function_untouched() {
    let refused = [
        Op::Mov(OpMov {
            dst: r(1),
            src: MovSrc::CBuf(CBufRef {
                buf: CBuf::BindlessUniform(ur(1)),
                offset: 0,
            }),
            quad_lanes: OpMov::ALL_LANES,
        }),
        Op::IMad64(crate::decode::OpIMad64 {
            dst: r(4),
            srcs: [reg(1), reg(2), reg(7)],
            signed: false,
            carry: NOT_PT,
        }),
        Op::IAdd3(OpIAdd3 {
            carry: [ps(0, false), NOT_PT],
            ..iadd3(0, [reg(1), reg(2), reg(3)])
        }),
    ];
    for op in refused {
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        builder
            .op(
                crate::ir::Opcode::GetRegister,
                &[Operand::Imm(Imm::Reg(r(9)))],
            )
            .expect("appends");
        let before = builder.function().clone();
        let error = translate_integer(&mut builder, &guarded(op, ps(1, false)), GuardMode::Honour);
        assert!(
            matches!(error, Err(IntegerError::Unsupported(_))),
            "{error:?}"
        );
        assert_eq!(builder.function(), &before);
        assert_eq!(builder.block(), entry);
        translate_integer(
            &mut builder,
            &guarded(mov(1, 2), ps(1, false)),
            GuardMode::Honour,
        )
        .expect("translates");
        assert_ne!(builder.block(), entry);
        assert_eq!(builder.function().block_count(), 3);
    }
}
