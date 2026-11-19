//! Flow analysis's blocks through the structurizer (slice 11, `crate::structure`): the sources
//! carried through, the layout that puts each test block after its block, a `KILL`'s discard,
//! and a `BRXU` jump table, each run in the interpreter against the graph with the source
//! blocks filled as a translator would fill them, after structuring.

use super::super::{BlockIndex, BlockSource, BranchTables, IrBlocks, flow_graph};
use super::{Item, Program, exit, kill, p, pure, side_effect};
use crate::asm::assemble;
use crate::decode::{BrxSrc, Gpr, Instruction, UniformReg};
use crate::interp::Machine;
use crate::ir::{BlockId, Builder, Function, Imm, Node, Opcode, Operand};
use crate::structure::tests::{Ending, same_run};
use crate::structure::{MAX_STRUCTURE_WORK, StructuredBlocks};

/// The register the filled blocks fold their flow block numbers into.
const PATH: u8 = 20;

/// The IR blocks of `code` with `tables`, and their structured form.
fn both(code: &[Instruction], tables: &BranchTables) -> (IrBlocks, StructuredBlocks) {
    let ir = flow_graph(code, tables)
        .expect("accepted")
        .build_ir()
        .expect("IR");
    let structured = ir
        .clone()
        .structure(MAX_STRUCTURE_WORK)
        .expect("structures");
    (ir, structured)
}

/// Fills each source block of `function` that holds guest instructions as a translator would:
/// it folds the flow block's number into r20, so the register records the path taken.
fn fill(function: &mut Function, ir: &IrBlocks, code: &[Instruction], tables: &BranchTables) {
    let graph = flow_graph(code, tables).expect("accepted");
    for source in &ir.sources {
        let flow = graph.block(source.flow).expect("a flow block");
        if flow.code.is_empty() {
            continue;
        }
        let mut b = Builder::new(function, source.block);
        let reg = Operand::Imm(Imm::Reg(Gpr::new(PATH)));
        let path = b.op(Opcode::GetRegister, &[reg]).expect("op");
        let path = b
            .op(
                Opcode::IMul32,
                &[Operand::Value(path), Operand::Imm(Imm::U32(31))],
            )
            .expect("op");
        let number = Operand::Imm(Imm::U32(source.flow.index() + 1));
        let path = b
            .op(Opcode::IAdd32, &[Operand::Value(path), number])
            .expect("op");
        b.op(Opcode::SetRegister, &[reg, Operand::Value(path)])
            .expect("op");
    }
}

/// The blocks the tree places, in the order it runs through them (pre-order).
fn placed(nodes: &[Node], out: &mut Vec<BlockId>) {
    for node in nodes {
        match node {
            Node::Block(block) => out.push(*block),
            Node::If { body, .. } | Node::Loop { body, .. } => placed(body, out),
            Node::Break { .. } | Node::Return | Node::Unreachable => {}
        }
    }
}

/// The graph's original blocks in the order the structured tree places them.
fn layout(ir: &IrBlocks, structured: &StructuredBlocks) -> Vec<u32> {
    let mut order = Vec::new();
    placed(
        structured.blocks.function.structure().expect("structured"),
        &mut order,
    );
    let original = u32::try_from(ir.function.block_count()).expect("small");
    order
        .into_iter()
        .map(BlockId::index)
        .filter(|&index| index < original)
        .collect()
}

/// Fills both forms, runs them from `machine` and returns the ending and the state.
fn run_filled(code: &[Instruction], tables: &BranchTables, machine: &Machine) -> (Ending, Machine) {
    let (ir, structured) = both(code, tables);
    let mut graph = ir.function.clone();
    fill(&mut graph, &ir, code, tables);
    let mut tree = structured.blocks.function.clone();
    fill(&mut tree, &ir, code, tables);
    tree.verify().expect("the filled tree verifies");
    same_run(&graph, &tree, machine)
}

/// The if/else program: block 0 tests p0 after its `BSSY`, blocks 1 and 2 are the arms,
/// block 3 the join.
fn if_else() -> Vec<Instruction> {
    Program::new()
        .bssy(0, "join")
        .bra(p(0), "else")
        .op(pure(1))
        .op(super::bsync(0))
        .label("else")
        .op(pure(2))
        .op(super::bsync(0))
        .label("join")
        .op(pure(3))
        .op(exit())
        .build()
}

#[test]
fn the_sources_are_carried_through_unchanged() {
    let code = if_else();
    let tables = BranchTables::new(0);
    let (ir, structured) = both(&code, &tables);
    assert_eq!(structured.blocks.sources, ir.sources);
    // Every source block keeps what it held: nothing (blocks with code, which the
    // translators fill) or its own test.
    for source in &ir.sources {
        assert_eq!(
            structured
                .blocks
                .function
                .block(source.block)
                .expect("a block")
                .insts,
            ir.function.block(source.block).expect("a block").insts
        );
    }
}

#[test]
fn each_test_block_follows_its_block() {
    // Flow blocks 0 to 3; block 0 holds code, so its test is block 4, made after the sources.
    // Laid out in arena order it would come last; the tree places it after block 0.
    let code = if_else();
    let tables = BranchTables::new(0);
    let (ir, structured) = both(&code, &tables);
    assert_eq!(ir.function.block_count(), 5);
    assert_eq!(layout(&ir, &structured), vec![0, 4, 1, 2, 3]);
    let arena =
        crate::structure::structure(ir.function.clone(), MAX_STRUCTURE_WORK).expect("structures");
    assert_ne!(arena.function, structured.blocks.function);
}

#[test]
fn the_if_else_runs_as_the_graph_does() {
    let code = if_else();
    let tables = BranchTables::new(0);
    // Path: block 0 (1), then block 1 (2) or block 2 (3), then block 3 (4).
    for (p0, path) in [(false, (31 + 2) * 31 + 4), (true, (31 + 3) * 31 + 4)] {
        let mut machine = Machine::new();
        machine.preds[0] = p0;
        let (ending, after) = run_filled(&code, &tables, &machine);
        assert_eq!(ending, Ending::Returned);
        assert_eq!(after.regs[usize::from(PATH)], path);
    }
}

#[test]
fn a_guard_tested_block_keeps_its_source() {
    // Flow blocks: 0 tests p0, 1 the EXIT, 2 the guarded side effect with its guard tested.
    let code = [side_effect(p(0)), assemble(&exit()).expect("encodable")];
    let tables = BranchTables::new(0);
    let graph = flow_graph(&code, &tables).expect("accepted");
    let (ir, structured) = both(&code, &tables);
    let source = &structured.blocks.sources[2];
    assert!(graph.block(source.flow).expect("a flow block").guard_tested);
    assert_eq!(source, &ir.sources[2]);
    let mut order = Vec::new();
    placed(
        structured.blocks.function.structure().expect("structured"),
        &mut order,
    );
    assert_eq!(order.iter().filter(|&&b| b == source.block).count(), 1);
    // Block 0 holds only its test, so it records nothing. The guarded block (3) runs only
    // when p0 holds, then the EXIT block (2).
    for (p0, path) in [(true, 3 * 31 + 2), (false, 2)] {
        let mut machine = Machine::new();
        machine.preds[0] = p0;
        let (ending, after) = run_filled(&code, &tables, &machine);
        assert_eq!(ending, Ending::Returned);
        assert_eq!(after.regs[usize::from(PATH)], path);
    }
}

#[test]
fn a_kill_demotes_in_the_structured_tree() {
    // Flow blocks: 0 tests p0, 1 the EXIT, 2 the KILL, whose demote block (3) follows it.
    let code = Program::new().op(kill(p(0))).op(exit()).build();
    let tables = BranchTables::new(0);
    let (ir, structured) = both(&code, &tables);
    assert_eq!(layout(&ir, &structured), vec![0, 1, 2, 3]);
    for p0 in [false, true] {
        let mut machine = Machine::new();
        machine.preds[0] = p0;
        let (ending, after) = run_filled(&code, &tables, &machine);
        assert_eq!(ending, Ending::Returned);
        assert_eq!(after.demoted, p0);
    }
}

#[test]
fn an_indirect_branch_is_structured() {
    // `MOV R2; BRX UR5; MOV R3; EXIT; EXIT`, the table naming instructions 2 and 4. Flow
    // blocks: 0 (to the BRX), 1 (instructions 2 and 3), 2 (instruction 4).
    let ur5 = UniformReg::new(5).expect("ur5");
    let code = Program::new()
        .op(pure(2))
        .item(Item::Brx(BrxSrc::UniformReg(ur5)))
        .op(pure(3))
        .op(exit())
        .op(exit())
        .build();
    let base = 0x40;
    let mut tables = BranchTables::new(base);
    tables.insert(1, vec![2, 4]);
    let size = super::size();
    let (ir, structured) = both(&code, &tables);
    // The test block (3) and the comparison blocks (4, then 5, which is unreachable) follow
    // block 0.
    assert_eq!(layout(&ir, &structured), vec![0, 3, 4, 5, 1, 2]);
    for (address, ending, path) in [
        (base + 2 * size, Ending::Returned, 31 + 2),
        (base + 4 * size, Ending::Returned, 31 + 3),
        (base + 3 * size, Ending::Unreachable, 1),
    ] {
        let mut machine = Machine::new();
        machine.uniform_regs[usize::from(ur5.index())] = address;
        let (ended, after) = run_filled(&code, &tables, &machine);
        assert_eq!(ended, ending);
        assert_eq!(after.regs[usize::from(PATH)], path);
        assert_eq!(after.indirect_branch, address);
    }
}

#[test]
fn the_entry_comes_first_when_the_sources_leave_it_out() {
    // The if/else program with flow block 0's source taken out: block 0 is still the entry,
    // and the tree must start there, not at the first block a source names.
    let code = if_else();
    let tables = BranchTables::new(0);
    let mut ir = flow_graph(&code, &tables)
        .expect("accepted")
        .build_ir()
        .expect("IR");
    let entry = ir.sources.remove(0);
    assert_eq!(entry.block, ir.function.entry());
    let structured = ir
        .clone()
        .structure(MAX_STRUCTURE_WORK)
        .expect("structures");
    // Block 0 jumps to its test block (4), which no source names, so the chain places it next.
    assert_eq!(layout(&ir, &structured), vec![0, 4, 1, 2, 3]);
    let mut graph = ir.function.clone();
    fill(&mut graph, &ir, &code, &tables);
    let mut tree = structured.blocks.function.clone();
    fill(&mut tree, &ir, &code, &tables);
    // Block 0 is not filled now: the path holds block 1 (2) or block 2 (3), then block 3 (4).
    for (p0, path) in [(false, 2 * 31 + 4), (true, 3 * 31 + 4)] {
        let mut machine = Machine::new();
        machine.preds[0] = p0;
        let (ending, after) = same_run(&graph, &tree, &machine);
        assert_eq!(ending, Ending::Returned);
        assert_eq!(after.regs[usize::from(PATH)], path);
    }
}

/// A graph whose blocks 1 to 3 no source names and no chain from a source reaches: block 0
/// branches on p0 to block 1 (`then`, which starts no chain) or block 4 (`otherwise`, attached
/// to block 0); block 1 jumps over block 2, which nothing reaches, to block 3.
const LEFTOVERS: &str = "block0:
    %0 = GetPred p0
    branch %0, block1, block4
block1:
    %1 = SetRegister r1, u32 1
    jump block3
block2:
    %2 = SetRegister r2, u32 2
    return
block3:
    %3 = SetRegister r3, u32 3
    return
block4:
    %4 = SetRegister r4, u32 4
    return
";

#[test]
fn blocks_no_chain_reaches_follow_in_arena_order() {
    let function = Function::parse(LEFTOVERS).expect("parses");
    let blocks: Vec<BlockId> = function.blocks().map(|(id, _)| id).collect();
    let source = |block: BlockId| BlockSource {
        block,
        flow: BlockIndex(0),
    };
    // Only block 0 is a source. Block 1 is not, so it starts no chain of its own: block 3,
    // its jump's target, keeps its arena place after block 2.
    let only_entry = IrBlocks {
        function: function.clone(),
        sources: vec![source(blocks[0])],
    };
    // The same, with block 0 named twice and a block the function does not have.
    let mut stray = Function::parse(LEFTOVERS).expect("parses");
    let mut missing = blocks[0];
    for _ in 0..=blocks.len() {
        missing = stray.add_block().expect("a block");
    }
    let odd = IrBlocks {
        function: function.clone(),
        sources: vec![source(blocks[0]), source(missing), source(blocks[0])],
    };
    for ir in [only_entry, odd] {
        let structured = ir
            .clone()
            .structure(MAX_STRUCTURE_WORK)
            .expect("structures");
        assert_eq!(layout(&ir, &structured), vec![0, 4, 1, 2, 3]);
        for p0 in [false, true] {
            let mut machine = Machine::new();
            machine.preds[0] = p0;
            let (ending, after) = same_run(&function, &structured.blocks.function, &machine);
            assert_eq!(ending, Ending::Returned);
            let expected = if p0 { [1, 0, 3, 0] } else { [0, 0, 0, 4] };
            assert_eq!(after.regs[1..=4], expected);
        }
    }
}
