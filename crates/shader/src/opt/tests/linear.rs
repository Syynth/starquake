//! The work bound (spec, "Compilation, threads and stutter": every stage linear or close to it):
//! each pass's counted work on a large generated function grows in proportion to its size.

use std::fmt::Write;

use crate::interp::Machine;
use crate::ir::Function;
use crate::opt::{Pass, Work, optimize, run};
use crate::structure::{MAX_STRUCTURE_WORK, structure};

/// IR text being written, with the next instruction id.
struct Text {
    text: String,
    id: usize,
}

impl Text {
    /// Writes instruction `body` (what follows its id) and returns its id.
    fn line(&mut self, body: &str) -> usize {
        let id = self.id;
        let _ = writeln!(self.text, "    %{id}{body}");
        self.id += 1;
        id
    }

    /// Writes a line that is not an instruction.
    fn raw(&mut self, line: &str) {
        let _ = writeln!(self.text, "{line}");
    }
}

/// `count` diamonds in a row, each followed by a loop of one block: every block reads and
/// writes the same registers, so the SSA rewrite places phis at every join and loop header.
fn program(count: usize) -> Function {
    let mut t = Text {
        text: String::new(),
        id: 0,
    };
    t.raw("block0:");
    t.line(" = SetRegister r0, u32 1");
    t.line(" = SetRegister r1, u32 2");
    t.raw("    jump block1");
    let mut block = 1_usize;
    for _ in 0..count {
        let (head, left, right, join, body, exit) =
            (block, block + 1, block + 2, block + 3, block + 4, block + 5);
        t.raw(&format!("block{head}:"));
        let r0 = t.line(": u32 = GetRegister r0");
        let test = t.line(&format!(": u1 = INotEqual %{r0}, u32 0"));
        t.raw(&format!("    branch %{test}, block{left}, block{right}"));
        for side in [left, right] {
            t.raw(&format!("block{side}:"));
            let r1 = t.line(": u32 = GetRegister r1");
            let sum = t.line(&format!(": u32 = IAdd32 %{r1}, u32 {side}"));
            t.line(&format!(" = SetRegister r1, %{sum}"));
            t.raw(&format!("    jump block{join}"));
        }
        t.raw(&format!("block{join}:"));
        let r1 = t.line(": u32 = GetRegister r1");
        t.line(&format!(" = SetRegister r0, %{r1}"));
        t.raw(&format!("    jump block{body}"));
        t.raw(&format!("block{body}:"));
        let r0 = t.line(": u32 = GetRegister r0");
        let next = t.line(&format!(": u32 = ISub32 %{r0}, u32 1"));
        t.line(&format!(" = SetRegister r0, %{next}"));
        let again = t.line(&format!(": u1 = INotEqual %{next}, u32 0"));
        t.raw(&format!("    branch %{again}, block{body}, block{exit}"));
        block = exit;
    }
    t.raw(&format!("block{block}:"));
    let r0 = t.line(": u32 = GetRegister r0");
    let bits = t.line(&format!(": f32 = BitCastF32U32 %{r0}"));
    t.line(&format!(" = SetAttribute a0x80, %{bits}, u32 0"));
    t.raw("    return");
    Function::parse(&t.text).expect("parses")
}

/// Each pass's work on `function`, in order.
fn works(mut function: Function) -> Vec<Work> {
    Pass::ORDER
        .iter()
        .map(|&pass| run(&mut function, pass).expect("runs"))
        .collect()
}

/// How many diamonds the smaller program has (about 16,000 instructions; the larger has twice
/// as many).
const DIAMONDS: usize = 1_000;

/// The fixed part of a pass's work that does not scale with the program: the entry and exit
/// blocks' share.
const SLACK: u64 = 64;

#[test]
fn every_pass_does_work_in_proportion_to_the_program() {
    let small = works(program(DIAMONDS));
    let large = works(program(2 * DIAMONDS));
    for ((pass, small), large) in Pass::ORDER.iter().zip(&small).zip(&large) {
        assert!(small.steps() > 0, "{pass}");
        assert!(
            large.steps() <= 2 * small.steps() + SLACK,
            "{pass}: {} steps for twice the program, {} for once",
            large.steps(),
            small.steps()
        );
    }
}

#[test]
fn a_long_identity_chain_is_followed_once() {
    // %0, then a chain of `count` identities, each used by the next and by an add.
    let count = 20_000_usize;
    let mut t = Text {
        text: String::new(),
        id: 0,
    };
    t.raw("block0:");
    let mut previous = t.line(": u32 = GetCbufU32 u32 0, u32 0");
    for _ in 0..count {
        let link = t.line(&format!(": opaque = Identity %{previous}"));
        t.line(&format!(": u32 = IAdd32 %{link}, u32 1"));
        previous = link;
    }
    t.raw("    return");
    let mut function = Function::parse(&t.text).expect("parses");
    let work = run(&mut function, Pass::IdentityRemoval).expect("runs");
    // Two instructions per link with up to two arguments each: one step per instruction and per
    // argument resolved, and one more per link the first time its chain is walked.
    let size = u64::try_from(2 * count + 1).expect("small");
    assert!(work.steps() <= 4 * size, "{} steps", work.steps());
}

/// Goto variable 0, copied to goto variables 1 to `count` in turn in block0, then all of them
/// read at a join of two blocks: each read's phi is trivial, and its operand is a chain of
/// `Identity` operations as long as the variable's index.
fn copies(count: u32) -> Function {
    let mut t = Text {
        text: String::new(),
        id: 0,
    };
    t.raw("block0:");
    let cond = t.line(": u1 = GetPred p0");
    t.line(&format!(" = SetGotoVariable u32 0, %{cond}"));
    for index in 1..=count {
        let read = t.line(&format!(": u1 = GetGotoVariable u32 {}", index - 1));
        t.line(&format!(" = SetGotoVariable u32 {index}, %{read}"));
    }
    t.raw(&format!("    branch %{cond}, block1, block2"));
    t.raw("block1:");
    t.raw("    jump block3");
    t.raw("block2:");
    t.raw("    jump block3");
    t.raw("block3:");
    for index in 0..=count {
        let read = t.line(&format!(": u1 = GetGotoVariable u32 {index}"));
        t.line(&format!(" = SetPred p1, %{read}"));
    }
    t.raw("    return");
    Function::parse(&t.text).expect("parses")
}

#[test]
fn many_variables_fed_from_one_chain_cost_their_size() {
    // The trivial phis' operands are chains of 1 to `count` links; following each from its
    // start would cost about count^2 / 2 steps, following each once costs about count.
    let count = 2_000;
    let small = run(&mut copies(count), Pass::SsaRewrite).expect("runs");
    let large = run(&mut copies(2 * count), Pass::SsaRewrite).expect("runs");
    assert!(
        large.steps() <= 2 * small.steps() + SLACK,
        "{} steps for twice the variables, {} for once",
        large.steps(),
        small.steps()
    );
}

#[test]
fn every_pass_does_work_in_proportion_to_the_structured_program() {
    // The same programs structured first, in Eden's order: the rewrite reads predecessors
    // from the tree.
    let structured = |count| {
        structure(program(count), MAX_STRUCTURE_WORK)
            .expect("structures")
            .function
    };
    let small = works(structured(DIAMONDS));
    let large = works(structured(2 * DIAMONDS));
    for ((pass, small), large) in Pass::ORDER.iter().zip(&small).zip(&large) {
        assert!(small.steps() > 0, "{pass}");
        assert!(
            large.steps() <= 2 * small.steps() + SLACK,
            "{pass}: {} steps for twice the program, {} for once",
            large.steps(),
            small.steps()
        );
    }
}

/// A structured tree of `count` `if`s and `count` loops with no block between them, where the
/// rewrite adds merge blocks and loop headers: each `if` and each loop body adds one to r0, and
/// each loop leaves through a `break`.
fn bare_tree(count: usize) -> Function {
    let mut t = Text {
        text: String::new(),
        id: 0,
    };
    t.raw("block0:");
    t.line(" = SetRegister r0, u32 0");
    let word = t.line(": u32 = GetCbufU32 u32 0, u32 0");
    let cond = t.line(&format!(": u1 = INotEqual %{word}, u32 0"));
    let mut tree = String::from("structure:\n    block0\n");
    for index in 0..2 * count {
        let block = index + 1;
        t.raw(&format!("block{block}:"));
        let r0 = t.line(": u32 = GetRegister r0");
        let sum = t.line(&format!(": u32 = IAdd32 %{r0}, u32 1"));
        t.line(&format!(" = SetRegister r0, %{sum}"));
        if index % 2 == 0 {
            let _ = write!(tree, "    if %{cond} {{\n        block{block}\n    }}\n");
        } else {
            let _ = write!(
                tree,
                "    loop {{\n        if %{cond} {{\n            block{block}\n        }}\n        \
                 break u1 true\n    }} repeat u1 false\n"
            );
        }
    }
    let last = 2 * count + 1;
    t.raw(&format!("block{last}:"));
    let r0 = t.line(": u32 = GetRegister r0");
    let bits = t.line(&format!(": f32 = BitCastF32U32 %{r0}"));
    t.line(&format!(" = SetAttribute a0x80, %{bits}, u32 0"));
    let _ = write!(tree, "    block{last}\n    return\n");
    t.text.push_str(&tree);
    Function::parse(&t.text).expect("parses")
}

#[test]
fn a_tree_with_no_block_where_paths_meet_costs_its_size() {
    // Without the merge blocks the rewrite adds, the k-th `if`'s block would have k
    // predecessors (every earlier body could have run last), and the rewrite's work would grow
    // with the square of the count.
    let small = run(&mut bare_tree(DIAMONDS), Pass::SsaRewrite).expect("runs");
    let large = run(&mut bare_tree(2 * DIAMONDS), Pass::SsaRewrite).expect("runs");
    assert!(
        large.steps() <= 2 * small.steps() + SLACK,
        "{} steps for twice the tree, {} for once",
        large.steps(),
        small.steps()
    );
    // The tree runs the same after every pass: each of the 4 `if`s and 4 loops adds one.
    let mut machine = Machine::new();
    machine.cbufs.insert(0, 1_u32.to_le_bytes().to_vec());
    let before = super::ran(&bare_tree(4), &machine);
    let mut after = bare_tree(4);
    optimize(&mut after).expect("optimizes");
    let after = super::ran(&after, &machine);
    assert_eq!(after.outputs, before.outputs);
    assert_eq!(before.outputs.values().copied().collect::<Vec<_>>(), [8]);
}
