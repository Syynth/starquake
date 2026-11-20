//! Whole programs from flow analysis through the structurizer and then the SSA rewrite and the
//! passes after it, in Eden's order (slice 8b; the owner's ruling of 2026-11-19): loops, nested
//! `if`s, a lifted goto and a `BRX` jump table. Each source block is filled as a translator
//! would fill it, and the optimized structured function must run as the graph does, with no
//! goto or condition variable left.

use super::super::{BranchTables, IrBlocks, flow_graph};
use super::{Item, Program, bsync, exit, kill, p, pure, side_effect};
use crate::asm::assemble;
use crate::decode::{BrxSrc, Gpr, Instruction, UniformReg};
use crate::interp::{InterpError, Machine};
use crate::ir::{Attribute, Builder, Function, GENERIC_0_X, Imm, Opcode, Operand, Value};
use crate::opt::optimize;
use crate::structure::MAX_STRUCTURE_WORK;

/// The register the filled blocks fold their flow block numbers into.
const PATH: u8 = 20;
/// The register the filled blocks count in.
const COUNT: u8 = 21;
/// The uniform register a `BRX` reads.
const TARGET: u8 = 5;
/// The constant buffer words a program reads: the loop limit, the `if` bits, the `BRX`
/// address.
const LIMIT_WORD: u32 = 0;
const BITS_WORD: u32 = 1;
const TARGET_WORD: u32 = 2;

fn reg(index: u8) -> Operand {
    Operand::Imm(Imm::Reg(Gpr::new(index)))
}

fn word(value: u32) -> Operand {
    Operand::Imm(Imm::U32(value))
}

/// Fills each source block of `function` that holds guest instructions as a translator would,
/// reading and writing guest state only through registers, predicates and uniform registers:
/// it folds the flow block's number into r20 and stores it to generic 0's `x`, counts in r21,
/// sets p0 while the count is below the limit word, sets p1 to the bit of the bits word the
/// count selects, and loads the `BRX` address into ur5.
fn fill(function: &mut Function, ir: &IrBlocks, code: &[Instruction], tables: &BranchTables) {
    let graph = flow_graph(code, tables).expect("accepted");
    for source in &ir.sources {
        let flow = graph.block(source.flow).expect("a flow block");
        if flow.code.is_empty() {
            continue;
        }
        let mut b = Builder::new(function, source.block);
        let mut op = |opcode: Opcode, args: &[Operand]| -> Operand {
            Operand::Value(b.op(opcode, args).expect("well-formed"))
        };
        let cbuf = |op: &mut dyn FnMut(Opcode, &[Operand]) -> Operand, index: u32| {
            op(Opcode::GetCbufU32, &[word(0), word(index * 4)])
        };
        let path = op(Opcode::GetRegister, &[reg(PATH)]);
        let path = op(Opcode::IMul32, &[path, word(31)]);
        let path = op(Opcode::IAdd32, &[path, word(source.flow.index() + 1)]);
        op(Opcode::SetRegister, &[reg(PATH), path]);
        let bits = op(Opcode::BitCastF32U32, &[path]);
        let generic = Operand::Imm(Imm::Attribute(Attribute(GENERIC_0_X)));
        op(Opcode::SetAttribute, &[generic, bits, word(0)]);
        let count = op(Opcode::GetRegister, &[reg(COUNT)]);
        let count = op(Opcode::IAdd32, &[count, word(1)]);
        op(Opcode::SetRegister, &[reg(COUNT), count]);
        let limit = cbuf(&mut op, LIMIT_WORD);
        let more = op(Opcode::ULessThan, &[count, limit]);
        op(Opcode::SetPred, &[pred(0), more]);
        let selector = cbuf(&mut op, BITS_WORD);
        let shifted = op(Opcode::ShiftRightLogical32, &[selector, count]);
        let bit = op(Opcode::BitwiseAnd32, &[shifted, word(1)]);
        let set = op(Opcode::INotEqual, &[bit, word(0)]);
        op(Opcode::SetPred, &[pred(1), set]);
        let target = cbuf(&mut op, TARGET_WORD);
        let ur = Operand::Imm(Imm::UniformReg(
            crate::ir::UniformReg::new(TARGET).expect("ur5"),
        ));
        op(Opcode::SetUniformRegister, &[ur, target]);
    }
}

fn pred(index: u8) -> Operand {
    Operand::Imm(Imm::Pred(
        crate::decode::Pred::new(index).expect("a predicate"),
    ))
}

/// How a run ended.
#[derive(Debug, PartialEq, Eq)]
enum Ending {
    Returned,
    Unreachable,
}

/// Runs `function` from `machine` and says how it ended; any other error fails the test.
fn ending(function: &Function, machine: &mut Machine) -> Ending {
    match machine.run(function) {
        Ok(_) => Ending::Returned,
        Err(InterpError::ReachedUnreachable(_)) => Ending::Unreachable,
        Err(error) => panic!("{error}\n{function}"),
    }
}

/// The machine with constant buffer 0 holding `words`.
fn start(words: [u32; 3]) -> Machine {
    let mut machine = Machine::new();
    let bytes = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    machine.cbufs.insert(0, bytes);
    machine
}

/// Builds `code`'s blocks, fills the graph, structures the blocks, fills the structured
/// function, runs every pass on it, and checks the optimized function from each of `starts`
/// against the graph: the same ending, outputs and demotion. Returns the graph's endings and
/// final machines, and the optimized function.
fn check(
    code: &[Instruction],
    tables: &BranchTables,
    starts: &[[u32; 3]],
) -> (Vec<(Ending, Machine)>, Function) {
    let ir = flow_graph(code, tables)
        .expect("accepted")
        .build_ir()
        .expect("IR");
    let structured = ir
        .clone()
        .structure(MAX_STRUCTURE_WORK)
        .expect("structures")
        .blocks;
    let mut graph = ir.function.clone();
    fill(&mut graph, &ir, code, tables);
    let mut tree = structured.function.clone();
    fill(&mut tree, &ir, code, tables);
    tree.verify().expect("the filled tree verifies");
    let mut optimized = tree.clone();
    optimize(&mut optimized).expect("the passes run on the structured function");
    let text = optimized.to_string();
    assert!(!text.contains("GotoVariable"), "{text}");
    assert!(!text.contains("GetRegister"), "{text}");
    let mut results = Vec::new();
    for &words in starts {
        let mut by_graph = start(words);
        let mut by_tree = start(words);
        let mut by_optimized = start(words);
        let graph_ending = ending(&graph, &mut by_graph);
        assert_eq!(ending(&tree, &mut by_tree), graph_ending, "{words:?}");
        assert_eq!(
            ending(&optimized, &mut by_optimized),
            graph_ending,
            "{words:?}"
        );
        for machine in [&by_tree, &by_optimized] {
            assert_eq!(machine.outputs, by_graph.outputs, "{words:?}\n{text}");
            assert_eq!(machine.demoted, by_graph.demoted, "{words:?}");
        }
        results.push((graph_ending, by_graph));
    }
    (results, optimized)
}

/// The path r20 holds after the flow blocks `blocks` ran, in order.
fn path(blocks: &[u32]) -> u32 {
    blocks
        .iter()
        .fold(0_u32, |path, &block| path.wrapping_mul(31).wrapping_add(block + 1))
}

/// The output the filled blocks store: the path's bits.
fn output(machine: &Machine) -> u32 {
    machine
        .outputs
        .get(&(Attribute(GENERIC_0_X), 0))
        .copied()
        .expect("an output")
}

/// Whether `function` has a phi.
fn has_phi(function: &Function) -> bool {
    function.listed().any(|(_, inst)| inst.is_phi())
}

/// The id of a listed phi of `function`, for messages.
fn first_phi(function: &Function) -> Option<Value> {
    function
        .listed()
        .find(|(_, inst)| inst.is_phi())
        .map(|(value, _)| value)
}

#[test]
fn a_loop_runs_the_same_after_ssa() {
    // Flow blocks: 0 (MOV), 1 (MOV, BRA back to itself while p0), 2 (MOV, EXIT). The count
    // reaches the limit after block 0 and each pass through block 1.
    let code = Program::new()
        .op(pure(1))
        .label("top")
        .op(pure(2))
        .bra(p(0), "top")
        .op(pure(3))
        .op(exit())
        .build();
    let tables = BranchTables::new(0);
    let (runs, optimized) = check(&code, &tables, &[[0, 0, 0], [2, 0, 0], [5, 0, 0]]);
    let expected = [path(&[0, 1, 2]), path(&[0, 1, 2]), path(&[0, 1, 1, 1, 1, 2])];
    for ((ending, machine), expected) in runs.iter().zip(expected) {
        assert_eq!(*ending, Ending::Returned);
        assert_eq!(output(machine), expected);
    }
    assert!(has_phi(&optimized), "{optimized}");
    assert!(optimized.to_string().contains("loop {"), "{optimized}");
}

/// An `if` inside an `if`, both on p1, so the bits word picks the path.
fn nested_ifs() -> Vec<Instruction> {
    Program::new()
        .op(pure(1))
        .bssy(0, "join")
        .bra(p(1), "else")
        .op(pure(2))
        .bssy(1, "inner_join")
        .bra(p(1), "inner_else")
        .op(pure(3))
        .op(bsync(1))
        .label("inner_else")
        .op(pure(4))
        .op(bsync(1))
        .label("inner_join")
        .op(pure(5))
        .op(bsync(0))
        .label("else")
        .op(pure(6))
        .op(bsync(0))
        .label("join")
        .op(pure(7))
        .op(exit())
        .build()
}

#[test]
fn nested_ifs_run_the_same_after_ssa() {
    let code = nested_ifs();
    let tables = BranchTables::new(0);
    // The bits word's bit n is p1 after the n-th filled block: every pair of choices.
    let starts = [[0, 0b000, 0], [0, 0b010, 0], [0, 0b100, 0], [0, 0b110, 0]];
    let (runs, optimized) = check(&code, &tables, &starts);
    let outputs: Vec<u32> = runs.iter().map(|(_, machine)| output(machine)).collect();
    // Four starts, and the two choices give at least three distinct paths.
    let mut distinct = outputs.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(distinct.len() >= 3, "{outputs:?}");
    assert!(has_phi(&optimized), "{:?}\n{optimized}", first_phi(&optimized));
}

#[test]
fn a_lifted_goto_runs_the_same_after_ssa() {
    // A guarded side effect whose guard is tested, then an EXIT: structuring this program lifts
    // gotos into loops (counted by hand while building this slice), so the rewrite reads
    // predecessors through the loops those lifts make.
    let code = vec![
        assemble(&pure(1)).expect("encodable"),
        side_effect(p(1)),
        assemble(&exit()).expect("encodable"),
    ];
    let tables = BranchTables::new(0);
    let (runs, optimized) = check(&code, &tables, &[[0, 0, 0], [0, 0b10, 0]]);
    assert_ne!(output(&runs[0].1), output(&runs[1].1));
    assert!(optimized.to_string().contains("loop {"), "{optimized}");
}

#[test]
fn a_kill_runs_the_same_after_ssa() {
    let code = Program::new().op(pure(1)).op(kill(p(1))).op(exit()).build();
    let tables = BranchTables::new(0);
    let (runs, _) = check(&code, &tables, &[[0, 0, 0], [0, 0b10, 0]]);
    assert_eq!(
        runs.iter().map(|(_, m)| m.demoted).collect::<Vec<_>>(),
        [false, true]
    );
}

#[test]
fn an_indirect_branch_runs_the_same_after_ssa() {
    // `MOV R2; BRX UR5; MOV R3; EXIT; EXIT`, the table naming instructions 2 and 4.
    let ur5 = UniformReg::new(TARGET).expect("ur5");
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
    let (runs, optimized) = check(
        &code,
        &tables,
        &[
            [0, 0, base + 2 * size],
            [0, 0, base + 4 * size],
            [0, 0, base + 3 * size],
        ],
    );
    let endings: Vec<&Ending> = runs.iter().map(|(ending, _)| ending).collect();
    assert_eq!(
        endings,
        [&Ending::Returned, &Ending::Returned, &Ending::Unreachable]
    );
    assert_eq!(output(&runs[0].1), path(&[0, 1]));
    assert_eq!(output(&runs[1].1), path(&[0, 2]));
    assert!(!optimized.to_string().contains("IndirectBranchVariable"), "{optimized}");
}
