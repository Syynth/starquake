//! The IR passes of slice 8 of the spec's port slices: the SSA rewrite, identity removal,
//! constant propagation and dead code elimination (spec, "Passes needed for correctness",
//! items 1 and 7).
//!
//! Ported by behaviour, not line by line (owner's decision of 2026-11-12, "The shader translator
//! uses a new arena-based Rust IR"), from Eden's `eden/src/shader_recompiler/ir_opt/` @ cf654052
//! (`ssa_rewrite_pass.cpp`, `identity_removal_pass.cpp`, `constant_propagation_pass.cpp`,
//! `dead_code_elimination_pass.cpp`; GPL-3.0-or-later over GPL-2.0-or-later yuzu code). Each
//! submodule lists where it departs from its Eden pass.
//!
//! # Order and verification
//!
//! [`Pass::ORDER`] is the order [`optimize`] runs them in. Eden's order is
//! `frontend/maxwell/translate_program.cpp` @ cf654052, lines 389–402: `SsaRewritePass`,
//! `ConstantPropagationPass`, then the position, global-memory, geometry and texture passes
//! (slices 13 to 15 here), the rescaling pass (not ported), and `DeadCodeEliminationPass`.
//! Eden at that commit never calls `IdentityRemovalPass` (it is defined, and declared in
//! `ir_opt/passes.h`, but nothing calls it); the spec lists it between the SSA rewrite and
//! constant propagation ("Passes needed for correctness", item 1), so it runs there. The
//! passes of slices 13 to 16 go between [`Pass::ConstantPropagation`] and
//! [`Pass::DeadCodeElimination`] when they are built.
//!
//! [`run`] runs one pass and, in tests and debug builds, then runs [`Function::verify`],
//! returning [`PassError::Verify`] if the output does not verify (spec, "The intermediate
//! representation": "checks types, SSA dominance and structure after every pass in debug
//! builds and in tests"). Eden runs its `VerificationPass` once, after dead code elimination,
//! and only when `renderer_debug` is set.
//!
//! # Block order
//!
//! Eden walks `post_order_blocks`, which `frontend/ir/post_order.cpp` (`PostOrder`) builds with
//! a depth-first walk from the first block, taking each block's successors in order; Eden
//! removes unreachable blocks first (`RemoveUnreachableBlocks`). Here a control-flow graph's
//! post order is that same walk along the terminators (in
//! [`Terminator::successors`](crate::ir::Terminator::successors) order), and blocks the entry
//! cannot reach, which this IR keeps, come after the reachable ones in reverse post order
//! (before them in post order), in arena order. A structured function's order is the order its
//! tree lists its blocks (its "reverse post order"; post order is that reversed), then any
//! block the tree does not place, in arena order.
//!
//! # Cost
//!
//! Each pass's work is close to linear in the function's size (spec, "Compilation, threads and
//! stutter"). Each returns the [`Work`] it did, counted in steps a test can bound
//! (`tests::linear`).

mod constant;
mod dce;
mod identity;
mod ssa;
#[cfg(test)]
mod tests;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::ir::{
    BlockId, BuildError, Function, Inst, Node, Opcode, Operand, Operation, VerifyError,
};

pub use dce::may_have_side_effects;

/// One of the passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pass {
    /// Guest registers, predicates, uniform registers and predicates, flags and the goto and
    /// indirect-branch variables become SSA values and phis (Eden's `SsaRewritePass`).
    SsaRewrite,
    /// `Identity` and `Void` operations are removed and their uses rewired (Eden's
    /// `IdentityRemovalPass`).
    IdentityRemoval,
    /// Operations are folded and simplified (Eden's `ConstantPropagationPass`).
    ConstantPropagation,
    /// Unused operations without side effects are removed (Eden's
    /// `DeadCodeEliminationPass`).
    DeadCodeElimination,
}

impl Pass {
    /// The order [`optimize`] runs the passes in (see the module docs).
    pub const ORDER: [Self; 4] = [
        Self::SsaRewrite,
        Self::IdentityRemoval,
        Self::ConstantPropagation,
        Self::DeadCodeElimination,
    ];

    /// The pass's name, for errors.
    pub const fn name(self) -> &'static str {
        match self {
            Self::SsaRewrite => "the SSA rewrite",
            Self::IdentityRemoval => "identity removal",
            Self::ConstantPropagation => "constant propagation",
            Self::DeadCodeElimination => "dead code elimination",
        }
    }
}

impl Display for Pass {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Why a pass failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PassError {
    /// The SSA rewrite was given a structured function (see the `ssa` module docs).
    #[error("the SSA rewrite needs a control-flow graph, not a structured function")]
    Structured,
    /// The entry block has predecessors and a phi, which the SSA rewrite cannot give a value on
    /// entry (see the `ssa` module docs).
    #[error("{0} is a phi in the entry block, which other blocks jump to")]
    EntryPhi(crate::ir::Value),
    /// A goto variable is named by a value, not an immediate (Eden reads the index with
    /// `Arg(0).U32()`, which only an immediate has).
    #[error("{0} names its goto variable with a value, not an immediate")]
    GotoIndex(crate::ir::Value),
    /// A chain of `Identity` operations loops, or ends at an id with no instruction.
    #[error("{0} is a chain of Identity operations that loops or ends without a value")]
    Identity(crate::ir::Value),
    /// A block lists an id with no instruction.
    #[error("{0} is listed in a block but does not exist")]
    Missing(crate::ir::Value),
    /// A pass could not add an instruction or a block.
    #[error(transparent)]
    Build(#[from] BuildError),
    /// The output of a pass does not verify.
    #[error("after {pass}: {error}")]
    Verify {
        /// The pass.
        pass: Pass,
        /// What the verifier found.
        error: VerifyError,
    },
}

/// The work a pass did, in steps: one per instruction, operand, block, terminator or tree
/// node visited, and one per step of the SSA rewrite's variable lookup.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Work {
    steps: u64,
}

impl Work {
    /// The steps counted.
    pub const fn steps(self) -> u64 {
        self.steps
    }

    /// Counts one step.
    pub(crate) const fn step(&mut self) {
        self.steps = self.steps.saturating_add(1);
    }

    /// Adds another pass's work.
    const fn absorb(&mut self, other: Self) {
        self.steps = self.steps.saturating_add(other.steps);
    }
}

/// Whether [`run`] verifies each pass's output: in tests and debug builds.
const VERIFY: bool = cfg!(any(test, debug_assertions));

/// Runs `pass` on `function`, then, in tests and debug builds, verifies the result.
///
/// A pass that fails leaves the function partly changed; the error says why.
pub fn run(function: &mut Function, pass: Pass) -> Result<Work, PassError> {
    let work = match pass {
        Pass::SsaRewrite => ssa::run(function)?,
        Pass::IdentityRemoval => identity::run(function)?,
        Pass::ConstantPropagation => constant::run(function)?,
        Pass::DeadCodeElimination => dce::run(function)?,
    };
    if VERIFY {
        function
            .verify()
            .map_err(|error| PassError::Verify { pass, error })?;
    }
    Ok(work)
}

/// Runs every pass of [`Pass::ORDER`] on `function`, in order, each through [`run`].
pub fn optimize(function: &mut Function) -> Result<Work, PassError> {
    let mut work = Work::default();
    for pass in Pass::ORDER {
        work.absorb(run(function, pass)?);
    }
    Ok(work)
}

/// `Identity` of `operand`: what Eden's `Inst::ReplaceUsesWith` turns an instruction into.
pub(crate) fn identity_of(operand: Operand) -> Inst {
    Inst::Op(Operation::new(Opcode::Identity, vec![operand]))
}

/// What a removed instruction's arena slot holds: a `Void` with no arguments (Eden's
/// `Inst::Invalidate`).
pub(crate) fn invalidated() -> Inst {
    Inst::Op(Operation::new(Opcode::Void, Vec::new()))
}

/// Each block's predecessors: the blocks whose terminator names it, each once, in arena order.
pub(crate) fn predecessors(function: &Function) -> Vec<Vec<BlockId>> {
    let mut preds = vec![Vec::new(); function.block_count()];
    for (from, block) in function.blocks() {
        for to in block.terminator.successors() {
            let Some(list) = slot(to).and_then(|index| preds.get_mut(index)) else {
                continue;
            };
            // A source's successors are visited together, so a repeat is the last entry.
            if list.last() != Some(&from) {
                list.push(from);
            }
        }
    }
    preds
}

/// How many instructions the function's arena holds, whether listed or not.
pub(crate) fn arena_len(function: &Function) -> usize {
    // `insts` zips the arena with an endless counter, so its lower size bound is exact.
    function.insts().size_hint().0
}

/// A block id's index into per-block vectors.
pub(crate) fn slot(block: BlockId) -> Option<usize> {
    usize::try_from(block.index()).ok()
}

/// The order the passes visit blocks in (see the module docs): a control-flow graph's reverse
/// post order from the entry, or a structured function's tree order, then the rest in arena
/// order.
pub(crate) fn block_order(function: &Function, work: &mut Work) -> Vec<BlockId> {
    let count = function.block_count();
    let mut placed = vec![false; count];
    let mut order = Vec::with_capacity(count);
    if let Some(tree) = function.structure() {
        tree_order(tree, &mut placed, &mut order, work);
    } else {
        let mut post = post_order(function, &mut placed, work);
        post.reverse();
        order = post;
    }
    for (id, _) in function.blocks() {
        if let Some(seen) = slot(id).and_then(|index| placed.get_mut(index))
            && !*seen
        {
            *seen = true;
            order.push(id);
        }
    }
    order
}

/// Whether the entry reaches each block, by block index (the blocks [`post_order`] visits).
pub(crate) fn reachable(function: &Function, work: &mut Work) -> Vec<bool> {
    let mut placed = vec![false; function.block_count()];
    post_order(function, &mut placed, work);
    placed
}

/// The post order of the blocks the entry reaches (Eden's `PostOrder`): a depth-first walk
/// that takes each block's first successor not yet visited, and emits a block once none is
/// left. Marks each visited block in `placed`.
fn post_order(function: &Function, placed: &mut [bool], work: &mut Work) -> Vec<BlockId> {
    let mut out = Vec::new();
    let entry = function.entry();
    let Some(first) = slot(entry).and_then(|index| placed.get_mut(index)) else {
        return out;
    };
    *first = true;
    let mut stack = vec![entry];
    while let Some(block) = stack.pop() {
        work.step();
        let next = function.block(block).and_then(|b| {
            b.terminator.successors().find(|&to| {
                slot(to)
                    .and_then(|index| placed.get_mut(index))
                    .is_some_and(|seen| !std::mem::replace(seen, true))
            })
        });
        match next {
            Some(to) => {
                stack.push(block);
                stack.push(to);
            }
            None => out.push(block),
        }
    }
    out
}

/// The blocks a structured tree lists, in order (each once), marking each in `placed`.
fn tree_order(tree: &[Node], placed: &mut [bool], order: &mut Vec<BlockId>, work: &mut Work) {
    let mut stack: Vec<std::slice::Iter<'_, Node>> = vec![tree.iter()];
    while let Some(top) = stack.last_mut() {
        let Some(node) = top.next() else {
            stack.pop();
            continue;
        };
        work.step();
        match node {
            Node::Block(block) => {
                if let Some(seen) = slot(*block).and_then(|index| placed.get_mut(index))
                    && !*seen
                {
                    *seen = true;
                    order.push(*block);
                }
            }
            Node::If { body, .. } | Node::Loop { body, .. } => stack.push(body.iter()),
            Node::Break { .. } | Node::Return | Node::Unreachable => {}
        }
    }
}

/// Calls `f` on every condition of a structured tree (`if`, `break` and loop repeat
/// conditions), in tree order.
pub(crate) fn tree_conditions(tree: &[Node], work: &mut Work, mut f: impl FnMut(Operand)) {
    let mut stack: Vec<std::slice::Iter<'_, Node>> = vec![tree.iter()];
    while let Some(top) = stack.last_mut() {
        let Some(node) = top.next() else {
            stack.pop();
            continue;
        };
        work.step();
        match node {
            Node::If { cond, body } => {
                f(*cond);
                stack.push(body.iter());
            }
            Node::Loop { body, repeat } => {
                f(*repeat);
                stack.push(body.iter());
            }
            Node::Break { cond } => f(*cond),
            Node::Block(_) | Node::Return | Node::Unreachable => {}
        }
    }
}

/// [`tree_conditions`], changing each condition.
pub(crate) fn tree_conditions_mut(
    tree: &mut [Node],
    work: &mut Work,
    mut f: impl FnMut(&mut Operand) -> Result<(), PassError>,
) -> Result<(), PassError> {
    let mut stack: Vec<std::slice::IterMut<'_, Node>> = vec![tree.iter_mut()];
    while let Some(top) = stack.last_mut() {
        let Some(node) = top.next() else {
            stack.pop();
            continue;
        };
        work.step();
        match node {
            Node::If { cond, body } => {
                f(cond)?;
                stack.push(body.iter_mut());
            }
            Node::Loop { body, repeat } => {
                f(repeat)?;
                stack.push(body.iter_mut());
            }
            Node::Break { cond } => f(cond)?,
            Node::Block(_) | Node::Return | Node::Unreachable => {}
        }
    }
    Ok(())
}
