//! The IR's blocks: one per flow block, in order, with the tests that end them in blocks of
//! their own after any instructions.

use super::super::{BranchTables, FlowError, IrBlocks, flow_graph};
use super::{Item, NEVER, Program, bsync, exit, graph, not_p, p, pure, r, size};
use crate::decode::{BrxSrc, Instruction, PredSrc, UniformReg};
use crate::ir::{Block, BlockId, Imm, Inst, Opcode, Operand, Terminator, Value};

/// The IR blocks of `code` with `tables`, verified.
fn ir_of(code: &[Instruction], tables: &BranchTables) -> IrBlocks {
    let ir = flow_graph(code, tables)
        .expect("accepted")
        .build_ir()
        .expect("IR");
    ir.function.verify().expect("verified");
    ir
}

/// Block `n` of `ir`.
fn nth(ir: &IrBlocks, n: usize) -> (BlockId, &Block) {
    ir.function.blocks().nth(n).expect("a block")
}

/// An operation: its opcode and arguments.
type Operation = (Opcode, Vec<Operand>);

/// The operation `value` names (never a phi: flow analysis makes none).
fn op(ir: &IrBlocks, value: Value) -> Operation {
    let op = match ir.function.inst(value).expect("an instruction") {
        Inst::Op(op) => Some((op.opcode, op.args.clone())),
        Inst::Phi(_) => None,
    };
    op.expect("an operation")
}

/// The if/else program of the graph tests.
fn if_else() -> Vec<Instruction> {
    Program::new()
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
        .build()
}

#[test]
fn each_flow_block_is_the_ir_block_of_its_index() {
    let ir = ir_of(&if_else(), &BranchTables::new(0));
    assert_eq!(ir.sources.len(), 4);
    for (index, source) in (0_u32..).zip(&ir.sources) {
        assert_eq!(source.block.index(), index);
        assert_eq!(source.flow.index(), index);
    }
    // Four blocks and the test block of block 0, whose code is not empty.
    assert_eq!(ir.function.block_count(), 5);
    // No block of the program's own holds an instruction: the translators fill them.
    for n in 0..4 {
        assert_eq!(nth(&ir, n).1.insts, Vec::<Value>::new());
    }
}

#[test]
fn a_test_after_instructions_gets_its_own_block() {
    let ir = ir_of(&if_else(), &BranchTables::new(0));
    let (test, _) = nth(&ir, 4);
    let (block1, _) = nth(&ir, 1);
    let (block2, _) = nth(&ir, 2);
    let (block3, _) = nth(&ir, 3);
    assert_eq!(nth(&ir, 0).1.terminator, Terminator::Jump(test));
    let test_block = nth(&ir, 4).1;
    assert_eq!(test_block.insts.len(), 1);
    let read = test_block.insts[0];
    assert_eq!(
        op(&ir, read),
        (Opcode::GetPred, vec![Operand::Imm(Imm::Pred(p(0).reg))])
    );
    assert_eq!(
        test_block.terminator,
        Terminator::Branch {
            cond: Operand::Value(read),
            then: block2,
            otherwise: block1,
        }
    );
    assert_eq!(nth(&ir, 1).1.terminator, Terminator::Jump(block3));
    assert_eq!(nth(&ir, 3).1.terminator, Terminator::Return);
}

#[test]
fn an_empty_block_holds_its_test_and_negation_is_a_logical_not() {
    // The loop of the graph tests: block 2 (`@!P0 BRA head`) holds no instructions.
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
    let ir = ir_of(&code, &BranchTables::new(0));
    // Five flow blocks and one test block, for block 1 (code 1..2).
    assert_eq!(ir.function.block_count(), 6);
    let block = nth(&ir, 2).1;
    assert_eq!(block.insts.len(), 2);
    let (read, not) = (block.insts[0], block.insts[1]);
    assert_eq!(
        op(&ir, read),
        (Opcode::GetPred, vec![Operand::Imm(Imm::Pred(p(0).reg))])
    );
    assert_eq!(
        op(&ir, not),
        (Opcode::LogicalNot, vec![Operand::Value(read)])
    );
    assert_eq!(
        block.terminator,
        Terminator::Branch {
            cond: Operand::Value(not),
            then: nth(&ir, 1).0,
            otherwise: nth(&ir, 3).0,
        }
    );
}

#[test]
fn a_test_of_not_pt_is_the_constant_false() {
    // `BRA` with its own predicate `!PT`: the IR cannot read `PT`, so the test is `false`.
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
    let ir = ir_of(&code, &BranchTables::new(0));
    let block = nth(&ir, 0).1;
    assert_eq!(block.insts, Vec::<Value>::new());
    assert_eq!(
        block.terminator,
        Terminator::Branch {
            cond: Operand::Imm(Imm::U1(false)),
            then: nth(&ir, 2).0,
            otherwise: nth(&ir, 1).0,
        }
    );
}

/// `pure R2; BRX src; EXIT; EXIT` with the table naming 2 and 3.
fn jump_table(src: BrxSrc, base: u32) -> (Vec<Instruction>, BranchTables) {
    let code = Program::new()
        .op(pure(2))
        .item(Item::Brx(src))
        .op(exit())
        .op(exit())
        .build();
    let mut tables = BranchTables::new(base);
    tables.insert(1, vec![2, 3]);
    (code, tables)
}

/// The register read and the (address, target block) of each comparison, in order, and the
/// block the chain ends in.
fn chain(ir: &IrBlocks, first: usize) -> (Option<Operation>, Vec<(u32, BlockId)>, Terminator) {
    let mut block = nth(ir, first).1;
    let mut read = None;
    let mut targets = Vec::new();
    loop {
        for &value in &block.insts {
            let (opcode, args) = op(ir, value);
            if opcode == Opcode::IEqual {
                let address = match args[1] {
                    Operand::Imm(Imm::U32(address)) => Some(address),
                    _ => None,
                };
                let ends = match block.terminator {
                    Terminator::Branch {
                        then, otherwise, ..
                    } => Some((then, otherwise)),
                    _ => None,
                };
                let (then, otherwise) = ends.expect("a branch");
                targets.push((address.expect("an address"), then));
                block = ir.function.block(otherwise).expect("a block");
                break;
            }
            read = Some((opcode, args));
        }
        if block.insts.is_empty() {
            return (read, targets, block.terminator);
        }
    }
}

#[test]
fn an_indirect_branch_compares_each_address() {
    let base = 0x80;
    let (code, tables) = jump_table(BrxSrc::Reg(r(2)), base);
    let ir = ir_of(&code, &tables);
    // Three flow blocks, the test block of block 0, and one block per comparison.
    assert_eq!(ir.function.block_count(), 6);
    let (test, _) = nth(&ir, 3);
    assert_eq!(nth(&ir, 0).1.terminator, Terminator::Jump(test));
    let (read, targets, end) = chain(&ir, 3);
    assert_eq!(
        read,
        Some((Opcode::GetRegister, vec![Operand::Imm(Imm::Reg(r(2)))]))
    );
    assert_eq!(
        targets,
        vec![
            (base + 2 * size(), nth(&ir, 1).0),
            (base + 3 * size(), nth(&ir, 2).0)
        ]
    );
    assert_eq!(end, Terminator::Unreachable);
}

#[test]
fn indirect_addresses_use_the_table_base() {
    // A table counting from instruction 0 (base 0): the addresses are 32 and 48, not the
    // header's size plus those (the reference's `NativeSourceBase`).
    let (code, tables) = jump_table(BrxSrc::Reg(r(2)), 0);
    let ir = ir_of(&code, &tables);
    let (_, targets, _) = chain(&ir, 3);
    assert_eq!(
        targets
            .iter()
            .map(|&(address, _)| address)
            .collect::<Vec<_>>(),
        vec![2 * size(), 3 * size()]
    );
}

#[test]
fn indirect_registers_rz_and_uniform() {
    let (code, tables) = jump_table(BrxSrc::Reg(crate::decode::Gpr::RZ), 0);
    let ir = ir_of(&code, &tables);
    let first = nth(&ir, 3).1;
    let (opcode, args) = op(&ir, first.insts[0]);
    assert_eq!(opcode, Opcode::IEqual);
    assert_eq!(args[0], Operand::Imm(Imm::U32(0)));
    let ur5 = UniformReg::new(5).expect("ur5");
    let (code, tables) = jump_table(BrxSrc::UniformReg(ur5), 0);
    let ir = ir_of(&code, &tables);
    let (read, _, _) = chain(&ir, 3);
    assert_eq!(
        read,
        Some((
            Opcode::GetUniformRegister,
            vec![Operand::Imm(Imm::UniformReg(ur5))]
        ))
    );
    let (code, tables) = jump_table(BrxSrc::UniformReg(UniformReg::URZ), 0);
    assert_eq!(
        flow_graph(&code, &tables),
        Err(FlowError::BrxThroughUrz { index: 1 })
    );
}

#[test]
fn exits_return_and_jumps_jump() {
    let code = Program::new().op(pure(1)).op(exit()).build();
    let ir = graph(&code).expect("accepted").build_ir().expect("IR");
    assert_eq!(ir.function.block_count(), 1);
    assert_eq!(nth(&ir, 0).1.terminator, Terminator::Return);
}
