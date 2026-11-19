//! The work bound (spec, "Compilation, threads and stutter": every stage linear or close to it):
//! each pass's counted work on a large generated function grows in proportion to its size.

use std::fmt::Write;

use crate::ir::Function;
use crate::opt::{Pass, Work, run};

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
