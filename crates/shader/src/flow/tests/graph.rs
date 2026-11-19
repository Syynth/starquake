//! Block graphs: synthetic programs against the graphs written out by hand. Each also builds
//! its IR blocks, which must verify.

use super::super::{BranchTables, FlowGraph, flow_graph};
use super::{
    Item, NEVER, Program, SIZE, block, branch, bsync, exit, exit_if, graph, indirect, jump,
    kill, not_p, offset, p, pure, r, side_effect, tested,
};
use crate::asm::assemble;
use crate::decode::{BrxSrc, Instr, Op, OpCall, PredSrc};

use super::super::BlockEnd::Exit;

/// The graph of `code`, whose IR blocks must verify.
fn checked(code: &[crate::decode::Instruction]) -> FlowGraph {
    let graph = graph(code).expect("accepted");
    let ir = graph.build_ir().expect("IR blocks");
    ir.function.verify().expect("verified");
    graph
}

#[test]
fn a_straight_line_program_is_one_block() {
    let code = Program::new().op(pure(1)).op(pure(2)).op(exit()).build();
    assert_eq!(checked(&code).blocks(), &[block(0..3, Exit)]);
}

#[test]
fn an_if_else_reconverges_at_its_barrier() {
    // 0 BSSY B0 -> 6; 1 @P0 BRA 4; 2 pure; 3 BSYNC B0; 4 pure; 5 BSYNC B0; 6 pure; 7 EXIT.
    let code = Program::new()
        .bssy(0, "join")
        .bra(p(0), "else")
        .op(pure(1))
        .op(bsync(0))
        .label("else")
        .op(pure(2))
        .op(bsync(0))
        .label("join")
        .op(pure(3))
        .op(exit())
        .build();
    assert_eq!(
        checked(&code).blocks(),
        &[
            block(0..1, branch(p(0), 2, 1)),
            block(2..3, jump(3)),
            block(4..5, jump(3)),
            block(6..8, Exit),
        ]
    );
}

#[test]
fn a_loop_with_a_break() {
    // 0 BSSY B0 -> 5; 1 head: pure; 2 @P1 BRA 4; 3 @!P0 BRA 1; 4 BSYNC B0; 5 EXIT.
    let code = Program::new()
        .bssy(0, "after")
        .label("head")
        .op(pure(1))
        .bra(p(1), "out")
        .bra(not_p(0), "head")
        .label("out")
        .op(bsync(0))
        .label("after")
        .op(exit())
        .build();
    assert_eq!(
        checked(&code).blocks(),
        &[
            block(0..1, jump(1)),
            block(1..2, branch(p(1), 3, 2)),
            block(3..3, branch(not_p(0), 1, 3)),
            block(4..4, jump(4)),
            block(5..6, Exit),
        ]
    );
}

#[test]
fn a_guarded_side_effect_gets_its_own_block() {
    let code = [side_effect(p(0)), assemble(&exit()).expect("encodable")];
    assert_eq!(
        checked(&code).blocks(),
        &[
            block(0..0, branch(p(0), 2, 1)),
            block(1..2, Exit),
            tested(0..1, jump(1)),
        ]
    );
}

#[test]
fn a_kill_gets_its_own_block() {
    let code = Program::new().op(kill(p(0))).op(exit()).build();
    assert_eq!(
        checked(&code).blocks(),
        &[
            block(0..0, branch(p(0), 2, 1)),
            block(1..2, Exit),
            tested(0..1, jump(1)),
        ]
    );
    let code = Program::new().op(kill(PredSrc::TRUE)).op(exit()).build();
    assert_eq!(
        checked(&code).blocks(),
        &[block(0..0, jump(2)), block(1..2, Exit), tested(0..1, jump(1))]
    );
}

#[test]
fn conditional_exits_test_each_predicate() {
    // The guard alone.
    let code = Program::new()
        .op(exit_if(p(0), PredSrc::TRUE))
        .op(exit())
        .build();
    assert_eq!(
        checked(&code).blocks(),
        &[
            block(0..0, branch(p(0), 2, 1)),
            block(1..2, Exit),
            tested(0..1, Exit),
        ]
    );
    // The exit's own predicate alone.
    let code = Program::new()
        .op(exit_if(PredSrc::TRUE, not_p(1)))
        .op(exit())
        .build();
    assert_eq!(
        checked(&code).blocks(),
        &[
            block(0..0, branch(not_p(1), 2, 1)),
            block(1..2, Exit),
            tested(0..1, Exit),
        ]
    );
    // Both: the guard, then the predicate in a block in front of the next instruction.
    let code = Program::new()
        .op(exit_if(p(0), p(1)))
        .op(pure(1))
        .op(exit())
        .build();
    assert_eq!(
        checked(&code).blocks(),
        &[
            block(0..0, branch(p(0), 3, 1)),
            block(1..3, Exit),
            tested(0..1, Exit),
            block(1..1, branch(p(1), 2, 1)),
        ]
    );
}

#[test]
fn a_branch_with_two_predicates_tests_both() {
    let code = Program::new()
        .item(Item::Bra {
            guard: p(0),
            cond: p(1),
            label: "target",
        })
        .op(exit())
        .label("target")
        .op(exit())
        .build();
    assert_eq!(
        checked(&code).blocks(),
        &[
            block(0..0, branch(p(0), 3, 1)),
            block(1..2, Exit),
            block(2..3, Exit),
            block(0..0, branch(p(1), 2, 1)),
        ]
    );
    // The second predicate alone, `!PT`: a test the IR makes on a constant.
    let code = Program::new()
        .item(Item::Bra {
            guard: PredSrc::TRUE,
            cond: NEVER,
            label: "target",
        })
        .op(exit())
        .label("target")
        .op(exit())
        .build();
    assert_eq!(
        checked(&code).blocks(),
        &[
            block(0..0, branch(NEVER, 2, 1)),
            block(1..2, Exit),
            block(2..3, Exit),
        ]
    );
}

#[test]
fn disabled_instructions_stay_in_their_block() {
    let disabled_exit = exit_if(NEVER, PredSrc::TRUE);
    let code = Program::new()
        .op(pure(1))
        .op(disabled_exit)
        .word(side_effect(NEVER))
        .op(exit())
        .build();
    assert_eq!(checked(&code).blocks(), &[block(0..4, Exit)]);
}

#[test]
fn a_jump_table_branches_to_each_target() {
    // 0 pure R2; 1 BRX R2; 2 EXIT; 3 EXIT; the table names 2 and 3, from a base of 0x80.
    let code = Program::new()
        .op(pure(2))
        .item(Item::Brx(BrxSrc::Reg(r(2))))
        .op(exit())
        .op(exit())
        .build();
    let base = 0x80;
    let mut tables = BranchTables::new(base);
    tables.insert(1, vec![3, 2]);
    let graph = flow_graph(&code, &tables).expect("accepted");
    graph.build_ir().expect("IR").function.verify().expect("verified");
    assert_eq!(
        graph.blocks(),
        &[
            block(0..2, indirect(BrxSrc::Reg(r(2)), &[(base + 2 * SIZE, 1), (base + 3 * SIZE, 2)])),
            block(2..3, Exit),
            block(3..4, Exit),
        ]
    );
}

#[test]
fn a_function_is_inlined_at_each_call() {
    // 0 MOV R10; 1 CALL f; 2 MOV R10; 3 CALL f; 4 EXIT; 5 f: pure; 6 RET R10.
    let code = Program::new()
        .call(10, "f")
        .call(10, "f")
        .op(exit())
        .label("f")
        .op(pure(1))
        .ret(10)
        .build();
    assert_eq!(
        checked(&code).blocks(),
        &[
            block(0..1, jump(3)),
            block(2..3, jump(4)),
            block(4..5, Exit),
            block(5..6, jump(1)),
            block(5..6, jump(2)),
        ]
    );
}

#[test]
fn a_guarded_call_and_a_conditional_return() {
    // 0 MOV R10; 1 @P0 CALL f; 2 EXIT; 3 f: @P1 RET R10; 4 pure; 5 RET R10.
    let call = assemble(&Instr {
        guard: p(0),
        op: Op::Call(OpCall {
            offset: offset(1, 3),
            no_inc: true,
            cond: PredSrc::TRUE,
        }),
    })
    .expect("encodable");
    let code = Program::new()
        .item(Item::Link { dst: 10 })
        .word(call)
        .op(exit())
        .item(Item::Ret {
            guard: p(1),
            reg: 10,
        })
        .op(pure(1))
        .ret(10)
        .build();
    assert_eq!(
        checked(&code).blocks(),
        &[
            // The program: the link, the test, the exit, the conditional call.
            block(0..1, jump(1)),
            block(1..1, branch(p(0), 3, 2)),
            block(2..3, Exit),
            tested(1..1, jump(4)),
            // The function, inlined: its conditional return, then the rest.
            block(3..3, branch(p(1), 2, 5)),
            block(4..5, jump(2)),
        ]
    );
}

#[test]
fn a_skip_call_is_a_branch_block() {
    // 0 @P0 CALL 2; 1 BRA 3; 2 pure; 3 EXIT.
    let call = assemble(&Instr {
        guard: p(0),
        op: Op::Call(OpCall {
            offset: offset(0, 2),
            no_inc: true,
            cond: PredSrc::TRUE,
        }),
    })
    .expect("encodable");
    let code = Program::new()
        .word(call)
        .bra(PredSrc::TRUE, "end")
        .op(pure(1))
        .label("end")
        .op(exit())
        .build();
    assert_eq!(
        checked(&code).blocks(),
        &[
            block(0..0, branch(p(0), 2, 1)),
            block(1..1, jump(3)),
            block(2..3, jump(3)),
            block(3..4, Exit),
        ]
    );
}
