//! The structurizer: a control-flow graph turned into the structured tree naga needs
//! (`docs/specs/shader-pipeline.md`, "Control-flow recovery", "Structuring"; port slice 11).
//!
//! # What it does
//!
//! [`structure`] takes an IR [`Function`] in control-flow-graph form, before SSA, and returns
//! it structured: the same blocks, some new ones holding the instructions structuring adds, no
//! terminators, and a tree of [`Node`](crate::ir::Node)s (`if`, `loop` with a repeat
//! condition, conditional `break`, `return`, `unreachable`) that runs the blocks as the graph
//! did. [`IrBlocks::structure`] does the same for the blocks flow analysis builds (slice 10,
//! [`FlowGraph::build_ir`](crate::flow::FlowGraph::build_ir)) and carries their
//! [`IrBlocks::sources`] through unchanged: every source block keeps its id and its
//! instructions (none, or only its own test), and is placed in the tree once, so the
//! translators fill the structured function's blocks afterwards, reading each flow block's
//! `code` and `guard_tested` through [`BlockSource::flow`]. This is the IR build step the
//! owner's decision of 2026-11-18 gives slice 11 ("the structured IR the translators fill");
//! slice 10 built the blocks (`flow::ir_build`), so this slice did not take that part.
//!
//! The pass is a port, by behaviour, of Eden's goto elimination
//! (`eden/src/shader_recompiler/frontend/maxwell/structured_control_flow.cpp` @ cf654052,
//! GPL-3.0-or-later, Copyright 2026 Eden Emulator Project and 2021 yuzu Emulator Project;
//! PR #4588's copy, `eden-pr4588/src/shader_recompiler/frontend/maxwell/`, differs only in
//! formatting and in which translator it calls):
//!
//! 1. **The statement list** (`GotoPass::BuildTree`): a label per block, in the layout order
//!    below; then for each block, a reset of its label's goto variable to false at the front of
//!    the list and again after its label, the block's code, and its end: a goto to the target
//!    for a jump; for a branch, a conditional goto to `then` and an unconditional goto to
//!    `otherwise` (one unconditional goto when the condition is the immediate `true` or
//!    `false`); a `return`; or an `unreachable`. The goto variables are numbered by the labels'
//!    places in the layout, as Eden numbers them.
//! 2. **Goto elimination** (`GotoPass::RemoveGoto`), each goto in turn from the last made to
//!    the first: outward movement out of loops (and `if`s) until the goto and its label are
//!    directly related, lifting and inward movement into the statement that holds the label,
//!    then elimination as an `if` (a forward goto) or a loop (a backward goto). Moving a goto
//!    sets the goto variable of its label to its condition and tests the variable in its place
//!    (Eden's `SetVariable` and `Variable` statements).
//! 3. **Emission** (`TranslatePass::Visit`): the tree becomes [`Node`](crate::ir::Node)s. Code
//!    statements become [`Node::Block`](crate::ir::Node::Block)s of the function's own blocks;
//!    the instructions structuring adds go in new blocks, as Eden adds them to the blocks it
//!    makes: `Prologue` first, `SetGotoVariable` for each variable set, `GetGotoVariable`,
//!    `LogicalNot` and `LogicalOr` for the conditions, `ConditionRef` for the condition of each
//!    `if`, `break` and loop repeat, and `Epilogue` before each `return`. Control that falls off
//!    the end of the program reaches a [`Node::Unreachable`](crate::ir::Node::Unreachable), as
//!    in Eden.
//!
//! **Conditions.** Eden's goto conditions are guest conditions (`Identity`), read from the
//! predicates again wherever the tree finally tests them. A branch condition here is an IR
//! value of the block that ends in the branch, and goto elimination can test it far from that
//! block: an `if` made by eliminating one goto keeps the condition, and a later goto can then
//! move the block into another `if` in front of it, out of the value's scope (the verifier's
//! scope rule, `ir`'s module docs; a generated graph found this). So each computed branch
//! condition goes into a goto variable of its own, numbered after the labels' in the order the
//! branches are laid out: it is reset to false at the front of the list and set right after
//! its block, before the block's gotos, and the conditional goto tests that variable. A
//! variable set next to its block is never separated from it: every run of statements goto
//! elimination moves starts after a goto, at a label or at an `if` or loop, and ends at a
//! label, a goto, an `if` or loop, or the end of a list. The SSA pass (slice 8, reworked as 8b
//! to take the structured tree), which runs after this one, turns these variables into values,
//! as Eden's turns its goto variables (the owner's ruling of 2026-11-19, row 3, which also
//! accepted keeping conditions in variables, row 4).
//!
//! # What it takes
//!
//! The input must not be structured already ([`StructureError::AlreadyStructured`]) and must
//! verify ([`Function::verify`], [`StructureError::Invalid`]), checked in that order. It must
//! also be in the form the translators leave before SSA; these two rules are checked next,
//! block by block in arena order and in each block instruction by instruction, then its
//! terminator:
//!
//! - **No phis** ([`StructureError::Phi`]). This is the rule for structured phis: a phi of a
//!   control-flow graph names its predecessors, while a phi of a structured function takes the
//!   value from the block the tree ran last (the interpreter's rule, `crate::interp`), and
//!   structuring changes which block that is (it adds blocks in front of many blocks). So phis
//!   are made only after structuring, by the SSA pass (slice 8), over the structured tree, as
//!   in Eden, whose SSA rewrite runs after its structurizer (`translate_program.cpp`,
//!   `BuildASL` then `SsaRewritePass`); none is carried through.
//! - **No value used outside the block that defines it** ([`StructureError::CrossBlock`]),
//!   terminator conditions included. Before SSA, values cross blocks only through registers,
//!   predicates and variables, and the rule keeps every value the tree uses in scope: an
//!   instruction's arguments are in its own block, and a branch condition is read only by the
//!   variable set right after its block. Flow analysis's indirect branches meet it by reading
//!   the indirect branch variable again in each comparison block (`flow::ir_build`); the
//!   register, or the uniform register a `BRXU` reads (`GetUniformRegister`), goes into that
//!   variable once, in the first block.
//!
//! A `KILL`'s discard is the `DemoteToHelperInvocation` flow analysis puts in a block of its
//! own; that block is placed like any other and runs where the graph ran it, so the discard is
//! carried through (pinned by `flow::tests::structure::a_kill_demotes_in_the_structured_tree`).
//!
//! The output is checked with [`Function::verify`] before it is returned; a failure there is a
//! fault in this module ([`Invariant::Output`]), not in the input.
//!
//! # Layout
//!
//! [`structure`] lays the blocks out in arena order. [`IrBlocks::structure`] lays out the
//! function's entry first, whether or not a source names it, so the tree starts where the graph
//! does (`flow::tests::structure::the_entry_comes_first_when_the_sources_leave_it_out`); then
//! the source blocks in arena order. After each of these it places each block that no source
//! names (a test block, an indirect branch's comparison block or a kill's demote block, which
//! flow analysis makes after all the source blocks) right after the block whose jump, or whose
//! branch's `otherwise`, leads to it, following such chains: that is where Eden's statement
//! list has the test, the comparisons and the demote, among the flow block's own statements.
//! A conditional block, a branch whose `then` is such a block and jumps straight to the
//! branch's `otherwise`, leads the chain to its `then` first: the translation driver's
//! conditional blocks have that shape (slice 12i, `crate::translate::split_for_guard`), so a
//! guarded instruction lies between its test and the code after it, as its statement does in
//! Eden's list, and its goto is eliminated as an `if`, not a loop
//! (`translate::driver::tests::a_guarded_instruction_structures_as_an_if`). Any other branch's
//! `then` starts no chain (`flow::tests::structure::blocks_no_chain_reaches_follow_in_arena_order`).
//! Flow analysis's own branches lead only to source blocks or, for an indirect branch's
//! comparisons, to their `otherwise`, so their layout is unchanged.
//! Only the entry and the source blocks start chains. A block no source names that no chain
//! reaches is placed after the rest, in arena order
//! (`flow::tests::structure::blocks_no_chain_reaches_follow_in_arena_order`). The sources are
//! not checked: one naming a block the function does not have is ignored, and one naming a
//! block twice counts once; whatever they name, each block is placed exactly once, the entry
//! first. Goto elimination only wraps runs of statements, never reorders them, so the tree runs
//! through the blocks in layout order
//! (`flow::tests::structure::each_test_block_follows_its_block`).
//!
//! # Limits
//!
//! The goto pass and the emission count their work ([`Structured::work`]): one unit for each
//! statement or condition made, each step up from a statement to its parent, each statement
//! stepped over or moved to a new parent, each statement emitted and each step of evaluating a
//! condition. A run that would go past its limit stops with [`StructureError::WorkLimit`], so a
//! run of exactly the limit finishes (`tests::limits::the_work_limit_at_its_edge`). Every loop
//! of the pass that walks the tree is charged to this budget: in the goto pass `level`,
//! `directly_related`, `sibling_from_nephew`, `are_ordered`, `splice` and the
//! `SanitizeNoBreaks` walk of `lift`, and in the emission its statement list and its condition
//! walk; `tests::golden::the_work_of_each_golden_graph` pins the exact work of graphs that run
//! each of them, and fails when any one of these charges, or the ones for making a statement or
//! a condition and for removing a goto that is right before its label, is dropped.
//! [`MAX_STRUCTURE_WORK`] is the production limit.
//!
//! **Memory.** The budget also bounds what the pass allocates, beyond the input and the
//! layout (one entry per block of the input): each unit makes at most one statement (a
//! 104-byte arena entry on a 64-bit host: a 24-byte kind and five 16-byte links), one condition
//! (24 bytes), or, in the emission, at most one IR instruction (a 40-byte `Inst`, at most two
//! 16-byte argument operands and its 4-byte `Value` in its block's list), one new glue block
//! (a 48-byte `Block` and the 48-byte `Node` that places it) and one more 48-byte `Node`: at
//! most 220 bytes a unit before allocator overhead, and the vectors that hold them can have up
//! to twice that capacity as they grow (`tree::tests::the_sizes_the_memory_bound_counts`). The
//! condition walk's own stacks are bounded by the condition it walks and freed after it. So a
//! run of [`MAX_STRUCTURE_WORK`] units can make a hostile input allocate up to about 55 GiB
//! (2^28 × 220 bytes), or twice that in capacity: far more than a host has, so the limit as it
//! stands bounds time, not memory. The owner accepted the limit as built, still a guess
//! (decision log, 2026-11-19, "Owner rulings on the wave 44 and 45 questions", row 4); real
//! programs would show whether to lower it or to charge allocations against a byte budget.
//! The tree may nest at most [`MAX_NESTING`] levels ([`StructureError::TooDeep`],
//! `tests::limits::nesting_at_its_edge`), the verifier's bound, which also bounds the
//! emission's recursion. Eden has neither limit. A chain of 257 conditional branches to one
//! block already nests 256 levels deep, so whether real programs fit is to be checked: the
//! same ruling accepted `MAX_NESTING` = 256 as built, a guess too.
//!
//! # Divergences from Eden
//!
//! | What | Here | Eden | Pinned by |
//! |---|---|---|---|
//! | The input | An IR control-flow graph before SSA, with no phis and no value crossing blocks | Its flow graph of guest blocks, translated while the tree is emitted | `tests::checks::*` |
//! | Branch conditions | Kept in a goto variable of their own, set right after their block | Guest conditions (`Identity`), read again where they are tested | `tests::golden::*`, `tests::generated::generated_graphs_run_the_same_structured` |
//! | Blocks it adds | One block for each run of instructions structuring adds; the graph's blocks stay as they are | A block for each `if` merge, loop header, continue block and return, the guest code translated into them | `tests::golden::*` |
//! | `Epilogue` | In the block in front of the `return` | In a block of its own after the current one | `tests::golden::a_single_block_returns` |
//! | A jump table | Flow analysis's chain of blocks, one comparison each, laid out after the `BRX`'s block | A `SetIndirectBranchVariable` and every `IndirectBranchCond` goto after the block, then `Unreachable` | `flow::tests::structure::an_indirect_branch_is_structured`, `tests::golden::a_switch_like_jump_table` |
//! | A `KILL` | A block of flow analysis's holding `DemoteToHelperInvocation` | A `Kill` statement emitted as `DemoteToHelperInvocation` | `flow::tests::structure::a_kill_demotes_in_the_structured_tree` |
//! | `MoveOutwardIf` | Keeps every statement after the goto in the new `if` | Drops the first of them (`if_body.pop_front()` after splicing from `std::next(goto_stmt)`) | `tree::tests::moving_out_of_an_if_keeps_every_statement` |
//! | Combining demotes (`DemoteCombinationPass`) | Not ported: a vendor workaround (`needs_demote_reorder`), which the spec leaves out | Run when the host asks | none |
//! | Work and nesting | [`StructureError::WorkLimit`], [`StructureError::TooDeep`] | No bound | `tests::limits::*` |
//! | A goto lifted over a `break` | [`StructureError::LiftOverBreak`] | `NotImplementedException` ("Capturing statement with break nodes"), the same check (`SanitizeNoBreaks`, the new loop's direct children only) | `tree::tests::a_lift_over_a_break_is_refused` |
//!
//! `MoveOutwardIf` and the lift over a `break` were not reached by any of 20,000 generated
//! graphs (a count taken by hand while building this slice), so their tests drive the goto
//! pass's transformations on hand-built trees.

mod emit;
#[cfg(test)]
pub(crate) mod tests;
mod tree;

use thiserror::Error;

use crate::flow::{BlockSource, IrBlocks};
use crate::ir::{
    BlockId, BuildError, Function, Inst, MAX_NESTING, Operand, Terminator, Value, VerifyError,
};

/// The work limit [`structure`] and [`IrBlocks::structure`] use (see the module docs, "Limits").
/// A **guess**: 2^28 units, to check against real programs, like flow analysis's
/// `MAX_FLOW_BLOCKS`; the owner accepted it as built (decision log, 2026-11-19, "Owner rulings
/// on the wave 44 and 45 questions", row 4). A run of diamonds takes the same work for each
/// diamond (`tests::limits::work_grows_linearly_with_a_run_of_diamonds`; 145 units each when
/// this was written), so the limit allows about 1.8 million of them; graphs that nest deeply
/// cost more per block. It also bounds memory, at up to 220 bytes a unit, so about 55 GiB
/// (twice that in vector capacity) for a hostile input (the module docs, "Limits"): too much
/// for a host, so real programs are also what would show whether to lower it.
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
    // The entry first, wherever the sources put it, then the source blocks in arena order.
    let sources_in_order = function.blocks().map(|(block, _)| block).filter(|&block| {
        slot(block)
            .and_then(|slot| is_source.get(slot))
            .copied()
            .unwrap_or(false)
    });
    for block in std::iter::once(function.entry()).chain(sources_in_order) {
        if slot(block).is_none_or(|slot| placed.get(slot).copied().unwrap_or(true)) {
            continue;
        }
        place(block, &mut placed, &mut order);
        // Follow the chain of attached blocks; each step places a new block, so the chain
        // ends within the function's blocks.
        let mut current = block;
        let attached = |next: BlockId, placed: &[bool]| {
            slot(next).is_some_and(|slot| {
                !is_source.get(slot).copied().unwrap_or(true)
                    && !placed.get(slot).copied().unwrap_or(true)
            })
        };
        while let Some(next) = fall_through(function, current, |then| attached(then, &placed)) {
            if !attached(next, &placed) {
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

/// The block a jump, or a branch's `otherwise`, leads to; for a conditional block, a branch
/// whose `then` is `attachable` (a block no source names and not placed yet) and jumps straight
/// to the branch's `otherwise`, its `then` instead (see the module docs, "Layout").
fn fall_through(
    function: &Function,
    block: BlockId,
    attachable: impl Fn(BlockId) -> bool,
) -> Option<BlockId> {
    match function.block(block)?.terminator {
        Terminator::Jump(target) => Some(target),
        Terminator::Branch {
            then, otherwise, ..
        } if attachable(then)
            && function.block(then).map(|block| block.terminator)
                == Some(Terminator::Jump(otherwise)) =>
        {
            Some(then)
        }
        Terminator::Branch { otherwise, .. } => Some(otherwise),
        Terminator::None | Terminator::Return | Terminator::Unreachable => None,
    }
}

/// The arena slot of `block`.
fn slot(block: BlockId) -> Option<usize> {
    usize::try_from(block.index()).ok()
}

/// Checks the input, structures it in `order` and checks the output.
fn run(
    mut function: Function,
    order: &[BlockId],
    work_limit: u64,
) -> Result<Structured, StructureError> {
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
