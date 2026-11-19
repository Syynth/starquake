//! The structurizer: a control-flow graph turned into the structured tree naga needs
//! (`docs/specs/shader-pipeline.md`, "Control-flow recovery", "Structuring"; port slice 11).
//!
//! # What it does
//!
//! [`structure`] takes an IR [`Function`] in control-flow-graph form, before SSA, and returns
//! it structured: the same blocks, a few new ones holding the instructions structuring adds,
//! no terminators, and a tree of [`Node`](crate::ir::Node)s (`if`, `loop` with a repeat condition, conditional
//! `break`, `return`, `unreachable`) that runs the blocks as the graph did.
//! [`IrBlocks::structure`] does the same for the blocks flow analysis builds (slice 10,
//! [`FlowGraph::build_ir`](crate::flow::FlowGraph::build_ir)) and carries their
//! [`IrBlocks::sources`] through unchanged: every source block keeps its id, is placed in the
//! tree once and is still empty or holds only its test, so the translators fill the
//! structured function's blocks from their guest instructions afterwards, reading each
//! flow block's `code` and `guard_tested` through [`BlockSource::flow`]. This is the IR build
//! step the owner's decision of 2026-11-18 gives slice 11 ("the structured IR the translators
//! fill"); slice 10 built the blocks (`flow::ir_build`), so this slice did not take that part.
//!
//! The pass is a port, by behaviour, of Eden's goto elimination
//! (`eden/src/shader_recompiler/frontend/maxwell/structured_control_flow.cpp` @ cf654052,
//! GPL-3.0-or-later, Copyright 2026 Eden Emulator Project and 2021 yuzu Emulator Project):
//!
//! 1. **The statement list** (`GotoPass::BuildTree`): a label per block, in the layout order
//!    below; then for each block, a reset of its goto variable to false at the front of the
//!    list and again after its label, the block's code, and its end: a goto to the target for a
//!    jump, a conditional goto to `then` and an unconditional goto to `otherwise` for a branch
//!    (one unconditional goto when the condition is the immediate `true` or `false`), a
//!    `return`, or an `unreachable`.
//! 2. **Goto elimination** (`GotoPass::RemoveGoto`), each goto in turn from the last made to
//!    the first: outward movement out of `if`s and loops until the goto and its label are
//!    directly related, lifting and inward movement into the statement that holds the label,
//!    then elimination as an `if` (a forward goto) or a loop (a backward goto). Moving a goto
//!    sets the goto variable of its label to its condition and tests the variable in its place
//!    (Eden's `SetVariable` and `Variable` statements).
//! 3. **Emission** (`TranslatePass::Visit`): the tree becomes [`Node`](crate::ir::Node)s. Code statements become
//!    [`Node::Block`](crate::ir::Node::Block)s of the function's own blocks; the instructions structuring adds go in
//!    new blocks (below), as Eden adds them to the blocks it makes: `Prologue` first,
//!    `SetGotoVariable` for each variable set, `GetGotoVariable`, `LogicalNot` and
//!    `LogicalOr` for the conditions, `ConditionRef` for the condition of each `if`, `break`
//!    and loop repeat, and `Epilogue` before each `return`. Control that falls off the end of
//!    the program reaches a [`Node::Unreachable`](crate::ir::Node::Unreachable), as in Eden.
//!
//! # What it takes
//!
//! The input must verify ([`Function::verify`]) and be a control-flow graph, not already
//! structured. It must also be in the form the translators leave before SSA, and these two
//! rules are checked, in this order after verification, block by block in arena order and in
//! each block instruction by instruction, then its terminator:
//!
//! - **No phis** ([`StructureError::Phi`]). This is the rule for structured phis: a phi of a
//!   control-flow graph names its predecessors, while a phi of a structured function takes the
//!   value from the block the tree ran last (the interpreter's rule, `crate::interp`), and
//!   structuring changes which block that is (it adds the blocks below in front of many
//!   blocks). So phis are made only after structuring, by the SSA pass (slice 8), over the
//!   structured tree, as in Eden, whose SSA rewrite runs after its structurizer
//!   (`frontend/maxwell/translate_program.cpp`); none is carried through.
//! - **No value used outside the block that defines it** ([`StructureError::CrossBlock`]),
//!   terminator conditions included. Before SSA, values cross blocks only through registers,
//!   predicates and variables. The rule is what keeps every condition in scope in the
//!   structured tree (the verifier's scope rule, `ir`'s module docs): a goto's condition is
//!   used where the goto first moves, and a goto has only other gotos between it and its
//!   block's code until then, so the condition's block is the statement before it in the same
//!   list. Flow analysis's indirect branches read the indirect branch variable again in each
//!   comparison block to meet it (`flow::ir_build`), and the uniform register a `BRXU` reads
//!   goes into that variable in the first block.
//!
//! A `KILL`'s discard is the `DemoteToHelperInvocation` flow analysis puts in a block of its
//! own; that block is placed like any other and runs where the graph ran it, so the discard is
//! carried through (pinned by `tests::flow::a_kill_demotes_in_the_structured_tree`).
//!
//! The output is checked with [`Function::verify`] before it is returned; a failure there is a
//! fault in this module ([`Invariant::Output`]), not in the input.
//!
//! # Layout
//!
//! [`structure`] lays the blocks out in arena order. [`IrBlocks::structure`] lays out the
//! source blocks in arena order and places each block that no source names (a test block,
//! an indirect branch's comparison block or a kill's demote block, which flow analysis adds
//! after all the source blocks) right after the block whose jump, or whose branch's
//! `otherwise`, leads to it: that is where Eden's own statement list has the test, the
//! comparisons and the demote, inside the flow block's statements. A block no source names
//! that no block leads to that way is placed after the rest, in arena order.
//!
//! # Limits
//!
//! The goto pass and the emission count their work ([`Structured::work`]): one unit for each
//! statement or condition made, each step up from a statement to its parent, each statement
//! stepped over or moved to a new parent, each statement emitted and each condition
//! evaluated. A run that would go past its limit stops with [`StructureError::WorkLimit`], so
//! a run of exactly the limit finishes; every loop of the pass is bounded by the tree it walks
//! and charged to this budget. [`MAX_STRUCTURE_WORK`] is the limit [`structure`] and
//! [`IrBlocks::structure`] use. The tree may nest at most [`MAX_NESTING`] levels
//! ([`StructureError::TooDeep`]), the verifier's bound, which keeps the emission's recursion
//! bounded. Eden has neither limit.
//!
//! # Divergences from Eden
//!
//! | What | Here | Eden | Pinned by |
//! |---|---|---|---|
//! | The input | An IR control-flow graph before SSA, with no phis and no value crossing blocks | Its flow graph of guest blocks, translated while the tree is emitted | `tests::checks::*` |
//! | Conditions | IR values computed in the block that ends in the branch, used where the goto first moves | Guest conditions (`Identity`), read again from the predicates where they are used | `tests::checks::cross_block_values_are_refused`, the interpreter equivalence of `tests::generated` |
//! | Blocks it adds | One block for each run of instructions structuring adds; source blocks stay as they are | A block for each `if` merge, loop header, continue block and return, the guest code translated into them | `tests::golden::*` |
//! | `Epilogue` | In the block in front of the `return` | In a block of its own after the current one | `tests::golden::a_single_block_returns` |
//! | A jump table | Flow analysis's chain of blocks, one comparison each, laid out after the `BRX`'s block | A `SetIndirectBranchVariable` and every `IndirectBranchCond` goto after the block, then `Unreachable` | `tests::flow::an_indirect_branch_is_structured` |
//! | A `KILL` | A block of flow analysis's holding `DemoteToHelperInvocation` | A `Kill` statement emitted as `DemoteToHelperInvocation` | `tests::flow::a_kill_demotes_in_the_structured_tree` |
//! | Combining demotes (`DemoteCombinationPass`) | Not ported: a vendor workaround (`needs_demote_reorder`), which the spec leaves out | Run when the host asks | none |
//! | Work and nesting | [`StructureError::WorkLimit`], [`StructureError::TooDeep`] | No bound | `tests::limits::*` |
//! | A goto lifted over a `break` | [`StructureError::LiftOverBreak`] | `NotImplementedException` ("Capturing statement with break nodes"), the same check (`SanitizeNoBreaks`, direct children only) | `tests::limits::a_lift_over_a_break_is_refused` |
//!
//! `MoveOutwardIf` in Eden drops the statement after the goto from the new `if`'s body
//! (`if_body.pop_front()` after splicing from `std::next(goto_stmt)`). That is ported as it
//! is; see `tree::Tree::move_outward_if`.

mod emit;
#[cfg(test)]
mod tests;
mod tree;

use thiserror::Error;

use crate::flow::{BlockSource, IrBlocks};
use crate::ir::{
    BlockId, BuildError, Function, Inst, MAX_NESTING, Operand, Terminator, Value, VerifyError,
};

/// The work limit [`structure`] and [`IrBlocks::structure`] use (see the module docs, "Limits").
/// A **guess**: 2^28 units, about a second on a host core, far more than a program at the
/// capture's cap should need; to check against real programs (an owner question, like flow
/// analysis's `MAX_FLOW_BLOCKS`).
pub const MAX_STRUCTURE_WORK: u64 = 1 << 28;

/// Why a function cannot be structured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum StructureError {
    /// The function is structured already.
    #[error("the function is structured already")]
    AlreadyStructured,
    /// The function does not verify.
    #[error("the function is malformed: {0}")]
    Invalid(VerifyError),
    /// The function has a phi (see the module docs, "What it takes").
    #[error("{0} is a phi, and the structurizer takes a function before SSA")]
    Phi(Value),
    /// A value is used outside the block that defines it (see the module docs, "What it
    /// takes").
    #[error("{block} uses {value}, which another block defines")]
    CrossBlock {
        /// The block that uses it.
        block: BlockId,
        /// The value.
        value: Value,
    },
    /// Structuring reached its work limit.
    #[error("structuring stopped at its work limit of {limit}")]
    WorkLimit {
        /// The limit.
        limit: u64,
    },
    /// The structured tree would nest deeper than [`MAX_NESTING`] levels.
    #[error("the structured tree would nest deeper than {max} levels", max = MAX_NESTING)]
    TooDeep,
    /// A goto would be lifted into a loop over a `break` of the loop around it, which would
    /// make the `break` leave the new loop instead (Eden's `SanitizeNoBreaks`).
    #[error("a goto would be lifted over a break, which the structurizer does not support")]
    LiftOverBreak,
    /// The IR could not take another block or instruction.
    #[error("the IR cannot hold the structured function: {0}")]
    Ir(#[from] BuildError),
    /// The structurizer broke an invariant of its own: a fault in this module, not in the
    /// input. It is an error, not a panic, in case that ever happens.
    #[error("the structurizer broke an invariant of its own: {0}")]
    Internal(Invariant),
}

/// Which of the structurizer's own invariants broke ([`StructureError::Internal`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum Invariant {
    /// A statement's chain of parents does not end at the root.
    #[error("a statement's parents do not lead to the root")]
    Parents,
    /// A list operation was given statements that are not siblings in the right order.
    #[error("a range of statements is not a run of siblings")]
    Siblings,
    /// A goto was to move out of the root or into a statement that is not an `if` or a loop.
    #[error("a goto cannot move from where it is")]
    Movement,
    /// A statement that should be a goto or a label is not.
    #[error("a statement is not of the kind expected")]
    Kind,
    /// A block reached the end of the statement list with no terminator.
    #[error("{0} has no terminator")]
    Unterminated(BlockId),
    /// The structured function does not verify.
    #[error("the structured function does not verify: {0}")]
    Output(VerifyError),
}

/// A structured function and the work structuring it took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Structured {
    /// The function, structured.
    pub function: Function,
    /// The work units structuring took (see the module docs, "Limits").
    pub work: u64,
}

/// Structures `function` with a work limit of `work_limit` units, the blocks laid out in arena
/// order (see the module docs). [`MAX_STRUCTURE_WORK`] is the limit for production use.
pub fn structure(function: Function, work_limit: u64) -> Result<Structured, StructureError> {
    let order: Vec<BlockId> = function.blocks().map(|(id, _)| id).collect();
    run(function, &order, work_limit)
}

/// Flow analysis's blocks, structured, and the work it took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuredBlocks {
    /// The blocks: the function structured, the sources unchanged.
    pub blocks: IrBlocks,
    /// The work units structuring took (see the module docs, "Limits").
    pub work: u64,
}

impl IrBlocks {
    /// Structures the function with a work limit of `work_limit` units, keeping the sources
    /// (see the module docs, "Layout"). [`MAX_STRUCTURE_WORK`] is the limit for production use.
    pub fn structure(self, work_limit: u64) -> Result<StructuredBlocks, StructureError> {
        let Self { function, sources } = self;
        let order = source_layout(&function, &sources);
        let Structured { function, work } = run(function, &order, work_limit)?;
        Ok(StructuredBlocks {
            blocks: Self { function, sources },
            work,
        })
    }
}

/// The layout of [`IrBlocks::structure`] (see the module docs, "Layout").
fn source_layout(function: &Function, sources: &[BlockSource]) -> Vec<BlockId> {
    let count = function.block_count();
    let mut is_source = vec![false; count];
    for source in sources {
        if let Some(slot) = slot(source.block).and_then(|slot| is_source.get_mut(slot)) {
            *slot = true;
        }
    }
    let mut placed = vec![false; count];
    let mut order = Vec::with_capacity(count);
    let place = |block: BlockId, placed: &mut Vec<bool>, order: &mut Vec<BlockId>| {
        if let Some(seen) = slot(block).and_then(|slot| placed.get_mut(slot))
            && !*seen
        {
            *seen = true;
            order.push(block);
        }
    };
    for (block, _) in function.blocks() {
        let is_source_block = slot(block)
            .and_then(|slot| is_source.get(slot))
            .copied()
            .unwrap_or(false);
        if !is_source_block {
            continue;
        }
        place(block, &mut placed, &mut order);
        // Follow the chain of attached blocks; each step places a new block, so the chain
        // ends within the function's blocks.
        let mut current = block;
        while let Some(next) = fall_through(function, current) {
            let attached = slot(next).is_some_and(|slot| {
                !is_source.get(slot).copied().unwrap_or(true)
                    && !placed.get(slot).copied().unwrap_or(true)
            });
            if !attached {
                break;
            }
            place(next, &mut placed, &mut order);
            current = next;
        }
    }
    for (block, _) in function.blocks() {
        place(block, &mut placed, &mut order);
    }
    order
}

/// The block a jump, or a branch's `otherwise`, leads to.
fn fall_through(function: &Function, block: BlockId) -> Option<BlockId> {
    match function.block(block)?.terminator {
        Terminator::Jump(target) => Some(target),
        Terminator::Branch { otherwise, .. } => Some(otherwise),
        Terminator::None | Terminator::Return | Terminator::Unreachable => None,
    }
}

/// The arena slot of `block`.
fn slot(block: BlockId) -> Option<usize> {
    usize::try_from(block.index()).ok()
}

/// Checks the input, structures it in `order` and checks the output.
fn run(mut function: Function, order: &[BlockId], work_limit: u64) -> Result<Structured, StructureError> {
    check_input(&function)?;
    let mut tree = tree::Tree::build(&function, order, work_limit)?;
    tree.remove_gotos()?;
    let nodes = emit::emit(&mut tree, &mut function)?;
    let work = tree.work();
    let ids: Vec<BlockId> = function.blocks().map(|(id, _)| id).collect();
    for id in ids {
        if let Some(block) = function.block_mut(id) {
            block.terminator = Terminator::None;
        }
    }
    function.set_structure(Some(nodes));
    function
        .verify()
        .map_err(|error| StructureError::Internal(Invariant::Output(error)))?;
    Ok(Structured { function, work })
}

/// The checks of the module docs' "What it takes", in their order.
fn check_input(function: &Function) -> Result<(), StructureError> {
    if function.structure().is_some() {
        return Err(StructureError::AlreadyStructured);
    }
    function.verify().map_err(StructureError::Invalid)?;
    // The block that lists each value.
    let mut home: Vec<Option<BlockId>> = vec![None; function.insts().count()];
    for (block, listed) in function.blocks() {
        for &value in &listed.insts {
            if let Some(slot) = usize::try_from(value.index())
                .ok()
                .and_then(|slot| home.get_mut(slot))
            {
                *slot = Some(block);
            }
        }
    }
    let local = |block: BlockId, operand: Operand| match operand {
        Operand::Value(value)
            if usize::try_from(value.index())
                .ok()
                .and_then(|slot| home.get(slot))
                .copied()
                .flatten()
                != Some(block) =>
        {
            Err(StructureError::CrossBlock { block, value })
        }
        _ => Ok(()),
    };
    for (block, listed) in function.blocks() {
        for &value in &listed.insts {
            match function.inst(value) {
                Some(Inst::Phi(_)) => return Err(StructureError::Phi(value)),
                Some(Inst::Op(op)) => {
                    for &arg in &op.args {
                        local(block, arg)?;
                    }
                }
                None => {}
            }
        }
        if let Terminator::Branch { cond, .. } = listed.terminator {
            local(block, cond)?;
        }
    }
    Ok(())
}

/// The nesting check of the emission, shared so the limit has one definition: whether a body
/// at nesting level `inner` may be emitted.
const fn nests(inner: usize) -> bool {
    inner <= MAX_NESTING
}
