//! Linear time near the instruction cap (row 121): the walk's work against its documented
//! bound, and the passes the reference does quadratically, run at the cap.

use super::super::{BarrierModel, BranchTables, MAX_PROGRAM, analyze, find_jump_tables, flow_graph};
use super::{Item, Program, bsync, exit, p, pure, r};
use crate::asm::assemble;
use crate::decode::{BrxSrc, Instr, Instruction, MovSrc, Op, OpMov};

/// The most applications of one instruction under each model (module docs, "Linear time").
const REGISTER_VISITS: u64 = 33;
const STACK_VISITS: u64 = 17;

/// `count` instructions: if/else diamonds reconverging at barrier 0, padded with pure
/// instructions, then an `EXIT`.
fn diamonds(count: u32) -> Vec<Instruction> {
    let mut program = Program::new();
    let mut n = 0;
    while program.here() + 7 < count {
        let other = super::name(format!("other{n}"));
        let join = super::name(format!("join{n}"));
        program
            .bssy(0, join)
            .bra(p(0), other)
            .op(pure(1))
            .op(bsync(0))
            .label(other)
            .op(pure(2))
            .op(bsync(0))
            .label(join);
        n += 1;
    }
    while program.here() + 1 < count {
        program.op(pure(3));
    }
    program.op(exit());
    program.build()
}

#[test]
fn the_walk_is_linear_near_the_cap() {
    let code = diamonds(MAX_PROGRAM);
    let count = u64::try_from(code.len()).expect("small");
    assert_eq!(count, u64::from(MAX_PROGRAM));
    let tables = BranchTables::new(0);
    let registers = analyze(&code, &tables, BarrierModel::Registers).expect("accepted");
    let stack = analyze(&code, &tables, BarrierModel::Stack).expect("accepted");
    assert!(registers.visits() <= REGISTER_VISITS * count);
    assert!(stack.visits() <= STACK_VISITS * count);
    // Every reachable instruction is applied at least once.
    assert!(registers.visits() >= count - 1);
    let graph = flow_graph(&code, &tables).expect("accepted");
    graph.build_ir().expect("IR").function.verify().expect("verified");
}

#[test]
fn barrier_states_that_keep_changing_stay_within_the_bound() {
    // Sixteen paths reach one long tail in turn, each with one more barrier set, so the
    // register model's state at the tail grows on every arrival and the tail is walked again
    // each time.
    let mut program = Program::new();
    for bar in (0_u8..16).rev() {
        program.label(super::name(format!("set{bar}"))).bssy(bar, "end");
        if bar > 0 {
            program.bra(p(0), super::name(format!("set{}", bar - 1)));
        }
        program.bra(crate::decode::PredSrc::TRUE, "tail");
    }
    program.label("tail");
    let tail_start = program.here();
    while program.here() + 1 < MAX_PROGRAM {
        program.op(pure(1));
    }
    program.label("end").op(exit());
    let code = program.build();
    let count = u64::try_from(code.len()).expect("small");
    let tail = count - u64::from(tail_start);
    let registers = analyze(&code, &BranchTables::new(0), BarrierModel::Registers)
        .expect("accepted");
    assert!(registers.visits() >= 16 * tail);
    assert!(registers.visits() <= REGISTER_VISITS * count);
}

#[test]
fn many_unmatched_branches_are_searched_once() {
    // The reference searches back from each BRX to the program's start when nothing matches:
    // quadratic. Here: one pass.
    let mut program = Program::new();
    while program.here() + 1 < MAX_PROGRAM {
        program.item(Item::Brx(BrxSrc::Reg(r(6))));
    }
    program.op(exit());
    assert_eq!(find_jump_tables(&program.build()), vec![]);
}

#[test]
fn many_returns_through_copies_are_checked_once() {
    // Functions that each return through a copy of the link register: the reference scans the
    // whole program for branches into each copy. The program calls each function once.
    let functions = 4_000;
    let mut program = Program::new();
    for n in 0..functions {
        program.call(10, super::name(format!("f{n}")));
    }
    program.op(exit());
    let copy = assemble(&Instr::new(Op::Mov(OpMov {
        dst: r(11),
        src: MovSrc::Reg(r(10)),
        quad_lanes: OpMov::ALL_LANES,
    })))
    .expect("encodable");
    for n in 0..functions {
        program.label(super::name(format!("f{n}"))).word(copy).ret(11);
    }
    let code = program.build();
    let analysis = analyze(&code, &BranchTables::new(0), BarrierModel::Registers)
        .expect("accepted");
    assert_eq!(analysis.function_entries().len(), functions + 1);
}
