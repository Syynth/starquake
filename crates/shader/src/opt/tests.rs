//! The passes' tests: golden IR (hand-written IR in, `insta` snapshots out), the departures
//! from Eden each module lists, interpreter equivalence on hand-written and generated
//! functions, and the work bound.

mod constant;
mod dce;
mod equivalence;
mod identity;
mod linear;
mod pipeline;
mod ssa;
mod structured;

use std::collections::{BTreeMap, BTreeSet};

use super::{Pass, PassError, run};
use crate::interp::Machine;
use crate::ir::{BlockId, Function, Inst, Node, Operand, Value};

/// Parses `text`, which must be well-formed.
fn parse(text: &str) -> Function {
    Function::parse(text).expect("parses")
}

/// The text of `text` after `pass`, which must succeed (and verify: [`run`] checks in tests).
fn after(text: &str, pass: Pass) -> String {
    let mut function = parse(text);
    run(&mut function, pass).expect("the pass runs");
    function.to_string()
}

/// The text of `text` after each pass of `passes` in turn.
fn after_each(text: &str, passes: &[Pass]) -> String {
    let mut function = parse(text);
    for &pass in passes {
        run(&mut function, pass).expect("the pass runs");
    }
    function.to_string()
}

/// The error `pass` gives on `text`.
fn error(text: &str, pass: Pass) -> PassError {
    let mut function = parse(text);
    run(&mut function, pass).expect_err("the pass fails")
}

/// The value id `index` (ids have no public constructor; a parsed function hands one out).
fn value(index: u32) -> Value {
    let text = format!("block0:\n    %{index} = Void\n    return\n");
    parse(&text)
        .blocks()
        .next()
        .and_then(|(_, block)| block.insts.first().copied())
        .expect("one instruction")
}

/// The machine a run of `function` on `machine` leaves.
fn ran(function: &Function, machine: &Machine) -> Machine {
    let mut machine = machine.clone();
    machine.run(function).expect("runs");
    machine
}

/// Checks the scope rule the SSA rewrite promises on a structured function (`ssa` module docs,
/// "Scope"): each value a phi takes from a block is in scope at the end of that block, which is
/// where the lowering stores it (`crate::lower`, "Structured control flow"). The structured
/// verifier only checks that such a value is listed somewhere, and the interpreter ignores
/// scope, so neither sees a value that has left scope.
///
/// The tree is walked as the verifier walks it: a value is in scope from where its block lists
/// it to the end of the `if` or loop body that holds that block. A phi that names a block the
/// tree does not place is an error too. A control-flow graph has nothing to check.
pub(crate) fn check_phi_scope(function: &Function) -> Result<(), String> {
    let Some(tree) = function.structure() else {
        return Ok(());
    };
    let mut wanted: BTreeMap<BlockId, Vec<(Value, Value)>> = BTreeMap::new();
    for (at, inst) in function.listed() {
        let Inst::Phi(phi) = inst else { continue };
        for &(from, operand) in &phi.incoming {
            if let Operand::Value(value) = operand {
                wanted.entry(from).or_default().push((at, value));
            }
        }
    }
    let mut walk = ScopeWalk {
        function,
        wanted: &wanted,
        visible: BTreeSet::new(),
        log: Vec::new(),
        placed: BTreeSet::new(),
    };
    walk.nodes(tree)?;
    match wanted.keys().find(|block| !walk.placed.contains(block)) {
        Some(block) => Err(format!(
            "a phi takes a value from {block}, which the tree does not place\n{function}"
        )),
        None => Ok(()),
    }
}

/// [`check_phi_scope`]'s walk of the tree, with the values in scope.
struct ScopeWalk<'f> {
    function: &'f Function,
    /// The values each block's phis take from it: the phi and the value.
    wanted: &'f BTreeMap<BlockId, Vec<(Value, Value)>>,
    visible: BTreeSet<Value>,
    /// The values brought into scope, in order, so a body's can leave scope when it ends.
    log: Vec<Value>,
    /// The blocks walked.
    placed: BTreeSet<BlockId>,
}

impl ScopeWalk<'_> {
    fn nodes(&mut self, nodes: &[Node]) -> Result<(), String> {
        for node in nodes {
            match node {
                Node::Block(block) => self.block(*block)?,
                Node::If { body, .. } | Node::Loop { body, .. } => {
                    let mark = self.log.len();
                    self.nodes(body)?;
                    for value in self.log.drain(mark..) {
                        self.visible.remove(&value);
                    }
                }
                Node::Break { .. } | Node::Return | Node::Unreachable => {}
            }
        }
        Ok(())
    }

    fn block(&mut self, block: BlockId) -> Result<(), String> {
        self.placed.insert(block);
        let insts = self
            .function
            .block(block)
            .map_or(&[][..], |block| block.insts.as_slice());
        for &value in insts {
            self.visible.insert(value);
            self.log.push(value);
        }
        for &(phi, value) in self.wanted.get(&block).map_or(&[][..], Vec::as_slice) {
            if !self.visible.contains(&value) {
                return Err(format!(
                    "{phi} takes {value} from {block}, where it is not in scope\n{}",
                    self.function
                ));
            }
        }
        Ok(())
    }
}
