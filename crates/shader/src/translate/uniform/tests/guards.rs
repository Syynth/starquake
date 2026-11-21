//! Guards: the conditional block, which file the guard is read from, `R2UR`'s guard read as
//! data, `!PT`, a tested guard and a successor's phi.

use super::*;
use crate::decode::{OpPrmt, OpUP2UR, PrmtMode};
use crate::flow::{classify, opcode_of};
use crate::ir::Type;

/// A `UMOV` of an immediate.
fn umov(dst: u8, bits: u32) -> Op {
    Op::UMov(OpUMov {
        dst: ur(dst),
        src: UMovSrc::Imm(bits),
    })
}

/// The first line of the entry block: the guard's read.
fn first_read(function: &Function) -> String {
    function
        .to_string()
        .lines()
        .nth(1)
        .unwrap_or_default()
        .trim()
        .to_owned()
}

/// A guarded `UMOV` is a conditional block, its guard read as the uniform predicate of the
/// same number: it runs where `UP3` holds (negated: where it does not), whatever `P3` holds.
#[test]
fn a_guard_makes_a_conditional_block() {
    for negated in [false, true] {
        let function = build(&[under(ps(3, negated), umov(4, 0x77))]).expect("translates");
        assert_eq!(first_read(&function), "%0: u1 = GetUniformPred up3");
        assert_eq!(function.block_count(), 3);
        for holds in [false, true] {
            let mut machine = machine_with(&[]);
            machine.uniform_preds[3] = holds;
            machine.preds[3] = !holds;
            machine.run(&function).expect("runs");
            let runs = holds != negated;
            let expected = if runs { 0x77 } else { 0 };
            assert_eq!(machine.uniform_regs[4], expected, "{negated} {holds}");
        }
    }
}

/// The guard of each form is read from the file `Visitor::Run` reads it from: a uniform
/// predicate for `ULDC`, `UPLOP3` and the twins `classify.h` marks uniform, a predicate for
/// `UPRMT` and `LDCG`, which it does not.
#[test]
fn each_form_reads_its_guard_from_its_file() {
    let uldc = Op::ULdc(OpULdc {
        dst: ur(1),
        src: ULdcSrc::CBuf(CBufRef {
            buf: CBuf::Binding(0),
            offset: 0,
        }),
        mem_type: MemType::B32,
    });
    let ldcg = Op::ULdc(OpULdc {
        dst: ur(1),
        src: ULdcSrc::Global {
            addr: ur(4),
            offset: 0,
            pred: ups(1, true),
        },
        mem_type: MemType::B32,
    });
    let uplop3 = Op::UPLop3(OpUPLop3 {
        dsts: [up(1), UniformPred::UPT],
        srcs: [ups(2, false), ups(3, false), ups(4, false)],
        luts: [0x80, 0],
    });
    let uiadd3 = Op::UIAdd3(twin(iadd3(4, [reg(1), reg(2), RZ])));
    let uprmt = Op::UPrmt(twin(OpPrmt {
        dst: r(4),
        srcs: [reg(1), reg(2)],
        sel: AluSrc::Imm(0x3210),
        mode: PrmtMode::Index,
    }));
    for (op, read) in [
        (uldc, "GetUniformPred up2"),
        (uplop3, "GetUniformPred up2"),
        (uiadd3, "GetUniformPred up2"),
        (uprmt, "GetPred p2"),
        (ldcg, "GetPred p2"),
    ] {
        let function = build(&[under(ps(2, false), op)]).expect("translates");
        assert_eq!(first_read(&function), format!("%0: u1 = {read}"), "{op}");
        assert_eq!(function.verify(), Ok(()), "{op}");
    }
}

/// [`reads_uniform_guard`] agrees with flow analysis's classes (`classify.h`'s `uniform`) for
/// every decoded sample of the group: two independent statements of one fact.
#[test]
fn the_guard_file_is_classify_s() {
    let mut uniform = 0;
    let mut vector = 0;
    for instr in crate::translate::driver::tests::decoded_samples() {
        if !in_group(&instr.op) {
            continue;
        }
        let class = classify(opcode_of(assemble(&instr).expect("encodable")));
        assert_eq!(reads_uniform_guard(&instr.op), class.uniform, "{instr}");
        if class.uniform {
            uniform += 1;
        } else {
            vector += 1;
        }
    }
    assert!(uniform > 0 && vector > 0);
    assert!(!reads_uniform_guard(&Op::UP2UR(OpUP2UR {})));
}

/// A guarded `R2UR` reads its guard as data (a predicate, `classify.h`: "reads the guard as
/// data, so it is not split into a block"): no conditional block, the guard the ballot's
/// input, and the uniform register written in every lane. The translation driver hands it over
/// unsplit too.
#[test]
fn r2ur_reads_its_guard_as_data() {
    let r2ur = Op::R2UR(OpR2UR {
        dst: ur(3),
        src: r(7),
    });
    for negated in [false, true] {
        let function = build(&[under(ps(3, negated), r2ur)]).expect("translates");
        assert_eq!(function.block_count(), 1);
        let text = function.to_string();
        let read = if negated {
            "%0: u1 = GetPred p3\n    %1: u1 = LogicalNot %0\n    %2: u32 = SubgroupBallot %1\n"
        } else {
            "%0: u1 = GetPred p3\n    %1: u32 = SubgroupBallot %0\n"
        };
        assert!(text.starts_with(&format!("block0:\n    {read}")), "{text}");
        assert_eq!(count(&function, Opcode::SetUniformRegister), 1);
    }
    let code = [
        word(&under(ps(3, false), r2ur)),
        word(&Instr::new(Op::Exit(crate::decode::OpExit {
            pred: PredSrc::TRUE,
            keep_ref_count: false,
            no_at_exit: false,
        }))),
    ];
    let stage = crate::translate::Stage {
        stage: crate::translate::GuestStage::Compute,
        color_outputs: crate::translate::ColorOutputs::default(),
    };
    let blocks =
        crate::translate::translate_flow(stage, None, &code, &crate::flow::BranchTables::new(0))
            .expect("translates");
    assert_eq!(blocks.function.block_count(), 1);
    assert_eq!(count(&blocks.function, Opcode::SubgroupBallot), 1);
    assert_eq!(count(&blocks.function, Opcode::GetPred), 1);
}

/// `!PT` appends nothing, without any check (a refused `R2UR`, `UBMSK`, `UP2UR`, a bindless
/// `ULDC`), in either mode; the group check still comes first.
#[test]
fn never_guarded_instructions_append_nothing() {
    let never = NOT_PT;
    let refused = [
        Op::R2UR(OpR2UR {
            dst: UniformReg::URZ,
            src: r(1),
        }),
        Op::UBMsk(twin(crate::decode::OpBMsk {
            dst: r(1),
            pos: reg(2),
            width: reg(3),
            wrap: false,
        })),
        Op::UP2UR(OpUP2UR {}),
        Op::ULdc(OpULdc {
            dst: ur(1),
            src: ULdcSrc::CBuf(CBufRef {
                buf: CBuf::BindlessUniform(ur(2)),
                offset: 0,
            }),
            mem_type: MemType::B32,
        }),
        umov(1, 2),
    ];
    for mode in [GuardMode::Honour, GuardMode::Tested] {
        for op in refused {
            let mut function = Function::new();
            let entry = function.entry();
            let mut builder = Builder::new(&mut function, entry);
            assert_eq!(
                translate_uniform(&mut builder, &under(never, op), mode),
                Ok(()),
                "{op}"
            );
            assert_eq!(builder.function(), &Function::new(), "{op}");
        }
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        let nop = Op::Nop(crate::decode::OpNop {});
        assert_eq!(
            translate_uniform(&mut builder, &under(never, nop), mode),
            Err(UniformError::OtherGroup("nop"))
        );
    }
}

/// With [`GuardMode::Tested`] a guard other than `!PT` is ignored: no read, no block, for a
/// non-twin, a twin and `R2UR` alike.
#[test]
fn a_tested_guard_is_ignored() {
    let ops = [
        umov(4, 9),
        Op::UIAdd3(twin(iadd3(4, [reg(1), reg(2), RZ]))),
        Op::R2UR(OpR2UR {
            dst: ur(4),
            src: r(1),
        }),
    ];
    for op in ops {
        for guard in [ps(1, false), ps(1, true)] {
            let mut function = Function::new();
            let entry = function.entry();
            let mut builder = Builder::new(&mut function, entry);
            translate_uniform(&mut builder, &under(guard, op), GuardMode::Tested)
                .expect("translates");
            assert_eq!(builder.block(), entry);
            builder.terminate(Terminator::Return).expect("terminates");
            assert_eq!(function.block_count(), 1, "{op}");
            assert_eq!(count(&function, Opcode::GetPred), 0, "{op}");
            assert_eq!(count(&function, Opcode::GetUniformPred), 0, "{op}");
        }
    }
}

/// `entry` jumps to `exit`, whose phi names `entry`.
fn phi_rig() -> (Function, BlockId, BlockId) {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let exit = builder.add_block().expect("adds");
    builder
        .terminate(Terminator::Jump(exit))
        .expect("terminates");
    builder.switch_to(exit).expect("switches");
    builder
        .phi(Type::U32, vec![(entry, Operand::Imm(Imm::U32(1)))])
        .expect("appends");
    builder.terminate(Terminator::Return).expect("terminates");
    (function, entry, exit)
}

/// A conditional block, a twin's or another form's, or an `LDCG`'s own split, in a block whose
/// successor has a phi naming it is refused before anything is appended; unguarded, the same
/// instructions translate there.
#[test]
fn a_phi_on_a_successor_refuses_the_split() {
    let ldcg = Op::ULdc(OpULdc {
        dst: ur(1),
        src: ULdcSrc::Global {
            addr: ur(4),
            offset: 0,
            pred: ups(1, false),
        },
        mem_type: MemType::B32,
    });
    let cases = [
        (under(ps(1, false), umov(4, 9)), true),
        (
            under(
                ps(1, false),
                Op::UIAdd3(twin(iadd3(4, [reg(1), reg(2), RZ]))),
            ),
            true,
        ),
        (Instr::new(ldcg), true),
        (Instr::new(umov(4, 9)), false),
    ];
    for (instr, refused) in cases {
        let (mut function, entry, exit) = phi_rig();
        let mut builder = Builder::new(&mut function, entry);
        let before = builder.function().clone();
        let result = translate_uniform(&mut builder, &instr, GuardMode::Honour);
        if refused {
            assert_eq!(
                result,
                Err(UniformError::SuccessorPhi {
                    block: entry,
                    successor: exit,
                }),
                "{instr}"
            );
            assert_eq!(builder.function(), &before, "{instr}");
        } else {
            assert_eq!(result, Ok(()), "{instr}");
        }
    }
}

/// Consecutive guarded instructions chain their blocks, each running where its own guard
/// holds.
#[test]
fn consecutive_guards_chain_their_blocks() {
    let function = build(&[
        under(ps(1, false), umov(4, 1)),
        under(ps(2, false), umov(5, 2)),
    ])
    .expect("translates");
    assert_eq!(function.block_count(), 5);
    for (a, b) in [(false, false), (false, true), (true, false), (true, true)] {
        let mut machine = machine_with(&[]);
        machine.uniform_preds[1] = a;
        machine.uniform_preds[2] = b;
        machine.run(&function).expect("runs");
        assert_eq!(machine.uniform_regs[4], u32::from(a));
        assert_eq!(machine.uniform_regs[5], 2 * u32::from(b));
    }
}
