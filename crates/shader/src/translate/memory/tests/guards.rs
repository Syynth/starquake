//! Guards: a conditional block for a guard other than `PT`, nothing for `!PT`, the guard
//! ignored once tested, and refusals that leave the function as it was.

use super::*;
use crate::ir::Type;

fn guarded(op: Op, guard: PredSrc) -> Instr {
    Instr { guard, op }
}

/// A shared store of `R1` to `R2`.
fn sts() -> Op {
    Op::St(st(
        1,
        MemType::B32,
        MemSpace::Shared(OffsetStride::X1),
        plain(2, false, 0),
    ))
}

/// `@P1 STS [R2], R1` tests P1, stores in a block of its own and continues in a third block
/// that holds the first block's terminator; memory changes only where P1 holds.
#[test]
fn a_guard_makes_a_conditional_block() {
    let function = build(&[guarded(sts(), ps(1, false))]).expect("translates");
    assert_eq!(
        function.to_string(),
        "block0:\n\
         \x20   %0: u1 = GetPred p1\n\
         \x20   branch %0, block1, block2\n\
         block1:\n\
         \x20   %1: u32 = GetRegister r2\n\
         \x20   %2: u32 = GetRegister r1\n\
         \x20   %3 = WriteSharedU32 %1, %2\n\
         \x20   jump block2\n\
         block2:\n\
         \x20   return\n"
    );
    for (cond, negated, expected) in [
        (false, false, 0),
        (true, false, 9),
        (false, true, 9),
        (true, true, 0),
    ] {
        let mut machine = machine_with(&[0, 9, 4]);
        machine.shared = vec![0; 8];
        machine.preds[1] = cond;
        run(&[guarded(sts(), ps(1, negated))], &mut machine);
        assert_eq!(shared_word(&machine, 4), expected, "{cond} {negated}");
    }
}

/// `LDC` is pure in PR #4588's `classify.h`, which keeps its guard as a select; here it is a
/// conditional block like the rest of the group (the module docs' divergence table), with the
/// same registers.
#[test]
fn a_guarded_ldc_is_a_conditional_block() {
    let op = Op::Ldc(OpLdc {
        dst: r(1),
        cbuf: cbuf(2, 4),
        offset: Gpr::RZ,
        mode: LdcMode::Indexed,
        mem_type: MemType::B32,
    });
    let function = build(&[guarded(op, ps(0, false))]).expect("translates");
    assert_eq!(
        function.to_string(),
        "block0:\n\
         \x20   %0: u1 = GetPred p0\n\
         \x20   branch %0, block1, block2\n\
         block1:\n\
         \x20   %1: u32 = GetCbufU32 u32 0x2, u32 0x4\n\
         \x20   %2 = SetRegister r1, %1\n\
         \x20   jump block2\n\
         block2:\n\
         \x20   return\n"
    );
    for (cond, expected) in [(false, 7), (true, 0x55)] {
        let mut machine = machine_with(&[0, 7]);
        with_cbuf(&mut machine, 2, &[0, 0x55]);
        machine.preds[0] = cond;
        machine.run(&function).expect("runs");
        assert_eq!(machine.regs[1], expected);
    }
}

/// An instruction under `!PT` never runs and appends nothing, without its checks (a refused
/// form under `!PT` translates to nothing too), as PR #4588's `Visitor::Run` returns.
#[test]
fn never_guarded_instructions_append_nothing() {
    let refused = Op::MemBar(OpMemBar {
        scope: MemBarScope::Sm,
        mmio: false,
        sc: false,
    });
    for op in [sts(), refused] {
        let function = build(&[guarded(op, ps(7, true))]).expect("translates");
        assert_eq!(function.to_string(), "block0:\n    return\n", "{op}");
    }
}

/// With [`GuardMode::Tested`] the guard is ignored: the instruction is appended in the
/// builder's block.
#[test]
fn a_tested_guard_is_ignored() {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    translate_memory(
        &mut builder,
        &guarded(sts(), ps(1, false)),
        GuardMode::Tested,
    )
    .expect("translates");
    builder.terminate(Terminator::Return).expect("terminates");
    assert_eq!(
        function.to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r2\n\
         \x20   %1: u32 = GetRegister r1\n\
         \x20   %2 = WriteSharedU32 %0, %1\n\
         \x20   return\n"
    );
}

/// A refused guarded instruction leaves the function as it was: no test of the guard, no new
/// blocks.
#[test]
fn refusals_leave_the_function_untouched() {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let refused = Op::MemBar(OpMemBar {
        scope: MemBarScope::Sm,
        mmio: false,
        sc: false,
    });
    let before = builder.function().clone();
    let result = translate_memory(
        &mut builder,
        &guarded(refused, ps(2, false)),
        GuardMode::Honour,
    );
    assert_eq!(
        result,
        Err(MemoryError::Unsupported(MemoryUnsupported::BarrierScope(
            MemBarScope::Sm
        )))
    );
    assert_eq!(builder.function(), &before);
}

/// A guarded instruction in a block whose successor has a phi naming it is refused before
/// anything is appended; an instruction's own refusal comes first; an unguarded one in the
/// same block translates.
#[test]
fn a_phi_on_a_successor_refuses_the_split() {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let next = builder.add_block().expect("adds");
    builder.terminate(Terminator::Jump(next)).expect("jumps");
    builder.switch_to(next).expect("switches");
    builder
        .phi(Type::U32, vec![(entry, imm32(1))])
        .expect("phi");
    builder.terminate(Terminator::Return).expect("returns");
    builder.switch_to(entry).expect("switches");
    let before = builder.function().clone();
    assert_eq!(
        translate_memory(
            &mut builder,
            &guarded(sts(), ps(1, false)),
            GuardMode::Honour
        ),
        Err(MemoryError::SuccessorPhi {
            block: entry,
            successor: next,
        })
    );
    assert_eq!(builder.function(), &before);
    let refused = Op::MemBar(OpMemBar {
        scope: MemBarScope::Gpu,
        mmio: true,
        sc: false,
    });
    assert_eq!(
        translate_memory(
            &mut builder,
            &guarded(refused, ps(1, false)),
            GuardMode::Honour
        ),
        Err(MemoryError::Unsupported(MemoryUnsupported::BarrierMmio))
    );
    assert!(translate_memory(&mut builder, &Instr::new(sts()), GuardMode::Honour).is_ok());
    let error = MemoryError::SuccessorPhi {
        block: entry,
        successor: next,
    };
    assert_eq!(
        error.to_string(),
        "a guarded instruction cannot split block0: block1 has a phi naming it"
    );
}

/// Two guarded instructions chain their blocks, each testing its own guard; the second keeps
/// the first's join as its block.
#[test]
fn consecutive_guards_chain_their_blocks() {
    let second = Op::St(st(
        3,
        MemType::B32,
        MemSpace::Shared(OffsetStride::X1),
        plain(2, false, 4),
    ));
    for preds in 0..4_u8 {
        let mut machine = machine_with(&[0, 5, 0, 6]);
        machine.shared = vec![0; 8];
        machine.preds[0] = preds & 1 != 0;
        machine.preds[1] = preds & 2 != 0;
        run(
            &[guarded(sts(), ps(0, false)), guarded(second, ps(1, false))],
            &mut machine,
        );
        assert_eq!(shared_word(&machine, 0), if preds & 1 != 0 { 5 } else { 0 });
        assert_eq!(shared_word(&machine, 4), if preds & 2 != 0 { 6 } else { 0 });
    }
}
