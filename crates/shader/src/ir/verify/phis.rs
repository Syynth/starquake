//! Step 5 of the verifier (the parent module's docs): a structured function's phis take each
//! value from a block that can run last before the phi's block, and the value is in scope at
//! the end of that block.
//!
//! This is the structured-phi rule the interpreter runs (`crate::interp`: a phi takes the value
//! from the block run last) and the lowering compiles (`crate::lower`, "Structured control
//! flow": each incoming value is stored in the phi's local variable at the end of the block it
//! names, and the local is loaded at the start of the phi's own block). A phi that names a
//! block that cannot run last before its own would load whatever another path stored, and a
//! value out of scope at the end of the named block has no naga expression there to store.
//! Step 4 checks only that a structured phi's incoming values are listed somewhere. Eden's
//! verifier checks neither rule: `ValidateTypes` and `ValidateForwardDeclarations` skip phis
//! (`eden/src/shader_recompiler/ir_opt/verification_pass.cpp` @ cf654052), and Eden's phis
//! name the predecessors its structurizer records. The `ir` module docs list this among the
//! places this verifier is stricter than Eden.
//!
//! # The blocks that can run last
//!
//! One walk of the tree builds a graph of points: each block is a point, each `if` adds a join
//! point, and each loop a start point and an exit point. The point run last is carried along:
//!
//! - a block follows it, and is then the point run last;
//! - an `if`'s body starts from it, and the join after the `if` follows both it (the condition
//!   may be false) and the end of the body;
//! - a loop's start follows it and the end of the body (the repeat), the body starts from the
//!   start, and the exit follows the end of the body and the point run last at each `break` of
//!   that loop; a `break` passes the point on (its condition may be false);
//! - after a `return` or an `unreachable` nothing has run.
//!
//! A block's predecessors are the blocks reached walking back from it through join, start and
//! exit points only. Conditions are not looked at, so an immediate condition does not remove
//! an edge, as in the SSA rewrite (`opt::ssa`'s module docs, "Structured functions"). A
//! predecessor with no incoming pair is not checked (the interpreter refuses it when that
//! block does run last, `InterpError::PhiNoIncoming`): `tests::a_missing_predecessor_is_not_checked`.
//!
//! The SSA rewrite derives its structured predecessors by the same rule (`opt::ssa`'s `tree`
//! module, `structured`), keeping lists of blocks rather than points. On the trees it leaves
//! (normalized: every loop body starts with a block) the two agree, block for block
//! (`opt::ssa::tree::tests::the_verifier_takes_the_rewrites_predecessors`). They differ on a
//! loop body that does not start with a block: the rewrite's walk adds no back edge then (it
//! normalizes the tree first, so it never meets one), while here the end of the body reaches
//! what runs first in it, as the interpreter runs it
//! (`tests::a_loop_body_that_starts_with_an_if_takes_the_back_edge`, and the rewrite's side in
//! `opt::ssa::tree::tests::the_rewrites_walk_adds_no_back_edge_to_a_body_that_starts_with_an_if`).
//!
//! # Scope
//!
//! A value is in scope at the end of a block when the value's block comes no later in tree
//! order and the `if` or loop body that holds the value's block encloses (or is) the body that
//! holds the named block. That is step 4's rule (the parent module's docs: in scope from where
//! a block lists it to the end of the body that holds that block), asked by place and by body
//! interval instead of by walking, so each pair costs a constant number of comparisons.
//!
//! # Cost
//!
//! The walk takes one step per node and stores at most two predecessors per point, except a
//! loop's exit, which stores one per `break` of the loop plus the end of its body: memory
//! linear in the tree. Then each block that holds a phi is walked back once, marking each point
//! it passes once (marks are stamped with the block's number, so nothing is allocated per
//! block); the points it passes are the joins, starts and exits between it and the blocks
//! before it. A hand-built tree can make that quadratic: with a run of k `if`s and no block
//! between them, a phi in the k-th body walks back through k - 1 joins. The trees phis come
//! from in a translation are the SSA rewrite's, which puts a block after every `if` and loop
//! that is followed by anything but a block, a `return` or an `unreachable`
//! (`opt::ssa`'s module docs, "Blocks are added where paths meet").
//!
//! The walk recurses once per nesting level, which step 3 has bounded by [`MAX_NESTING`].

#[cfg(test)]
mod tests;

#[cfg(doc)]
use super::super::MAX_NESTING;
use super::super::{BlockId, Function, Inst, Node, Operand};
use super::{Site, VerifyError};

/// Step 5 for a structured function whose tree is `tree`, after steps 1 to 4 passed (so every
/// block is placed once, every phi comes first in its block and names blocks that exist).
/// `sites` is where step 2 found each instruction listed, as (block index, position).
pub(super) fn check(
    function: &Function,
    tree: &[Node],
    sites: &[Option<(usize, usize)>],
) -> Result<(), VerifyError> {
    let has_phi = function.blocks.iter().any(|block| {
        block
            .insts
            .first()
            .and_then(|&value| function.inst(value))
            .is_some_and(Inst::is_phi)
    });
    if !has_phi {
        return Ok(());
    }
    let blocks = function.blocks.len();
    let mut walk = Walk::new(blocks);
    walk.nodes(tree, ROOT);
    walk.close(ROOT);
    let mut marks = vec![0; walk.points.len()];
    let mut stack = Vec::new();
    // Stamps start at 1, so no point is marked before its block's walk back.
    for (stamp, &block) in (1..).zip(&walk.order) {
        let insts = function
            .block(block)
            .map_or(&[][..], |b| b.insts.as_slice());
        let phis = insts.iter().map_while(|&value| match function.inst(value) {
            Some(Inst::Phi(phi)) => Some((value, phi)),
            _ => None,
        });
        let mut marked = false;
        for (at, phi) in phis {
            if !marked {
                let own = block.slot().unwrap_or(usize::MAX);
                walk.mark_predecessors(own, stamp, &mut marks, &mut stack);
                marked = true;
            }
            for &(from, operand) in &phi.incoming {
                // Step 4 refused a block that does not exist.
                let index = from.slot().filter(|&index| index < blocks).ok_or(
                    VerifyError::NoSuchBlock {
                        at: Site::Inst(at),
                        block: from,
                    },
                )?;
                if marks.get(index) != Some(&stamp) {
                    return Err(VerifyError::PhiNotPredecessor { at, block: from });
                }
                if let Operand::Value(value) = operand {
                    let defined = value
                        .slot()
                        .and_then(|slot| sites.get(slot))
                        .copied()
                        .flatten();
                    if !defined.is_some_and(|(def, _)| walk.in_scope_at_end(def, index)) {
                        return Err(VerifyError::PhiOutOfScope {
                            at,
                            operand: value,
                            block: from,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

/// The root list's body number.
const ROOT: usize = 0;

/// The walk of the tree (see the module docs): its points and where each block is.
struct Walk {
    /// How many blocks the function has: points `0..blocks` are the blocks, by index; the rest
    /// are joins, loop starts and loop exits.
    blocks: usize,
    /// Each point's predecessors.
    points: Vec<Vec<usize>>,
    /// The point run last.
    last: Option<usize>,
    /// The exit points of the loops open, innermost last.
    exits: Vec<usize>,
    /// The blocks in tree order.
    order: Vec<BlockId>,
    /// Each block's place in tree order, by block index.
    place: Vec<usize>,
    /// Each block's body, by block index.
    body: Vec<usize>,
    /// Each body's interval: the walk's count when it opened and when it closed. The root is
    /// body [`ROOT`].
    bodies: Vec<(usize, usize)>,
    /// The count the intervals are taken from.
    clock: usize,
}

impl Walk {
    fn new(blocks: usize) -> Self {
        Self {
            blocks,
            points: vec![Vec::new(); blocks],
            last: None,
            exits: Vec::new(),
            order: Vec::with_capacity(blocks),
            place: vec![usize::MAX; blocks],
            body: vec![ROOT; blocks],
            bodies: vec![(0, 0)],
            clock: 1,
        }
    }

    /// A new join, start or exit point.
    fn point(&mut self) -> usize {
        self.points.push(Vec::new());
        self.points.len().saturating_sub(1)
    }

    /// Records that `point` follows `pred`, when something has run.
    fn follow(&mut self, point: usize, pred: Option<usize>) {
        if let (Some(pred), Some(list)) = (pred, self.points.get_mut(point)) {
            list.push(pred);
        }
    }

    /// Opens a body and returns its number.
    fn open(&mut self) -> usize {
        let body = self.bodies.len();
        self.bodies.push((self.clock, self.clock));
        self.clock = self.clock.saturating_add(1);
        body
    }

    /// Closes body `body`.
    fn close(&mut self, body: usize) {
        if let Some(interval) = self.bodies.get_mut(body) {
            interval.1 = self.clock;
        }
        self.clock = self.clock.saturating_add(1);
    }

    /// Walks `nodes`, the list of body `body`.
    fn nodes(&mut self, nodes: &[Node], body: usize) {
        for node in nodes {
            match node {
                Node::Block(block) => {
                    // Step 3 placed every block that exists, and only those.
                    let Some(index) = block.slot().filter(|&index| index < self.blocks) else {
                        continue;
                    };
                    self.follow(index, self.last);
                    if let Some(place) = self.place.get_mut(index) {
                        *place = self.order.len();
                    }
                    if let Some(slot) = self.body.get_mut(index) {
                        *slot = body;
                    }
                    self.order.push(*block);
                    self.last = Some(index);
                }
                Node::If { body: inner, .. } => {
                    let before = self.last;
                    let opened = self.open();
                    self.nodes(inner, opened);
                    self.close(opened);
                    let join = self.point();
                    self.follow(join, before);
                    self.follow(join, self.last);
                    self.last = Some(join);
                }
                Node::Loop { body: inner, .. } => {
                    let start = self.point();
                    let exit = self.point();
                    self.follow(start, self.last);
                    self.exits.push(exit);
                    self.last = Some(start);
                    let opened = self.open();
                    self.nodes(inner, opened);
                    self.close(opened);
                    self.exits.pop();
                    self.follow(start, self.last);
                    self.follow(exit, self.last);
                    self.last = Some(exit);
                }
                Node::Break { .. } => {
                    // Step 3 refused a `break` outside every loop.
                    if let Some(&exit) = self.exits.last() {
                        self.follow(exit, self.last);
                    }
                }
                Node::Return | Node::Unreachable => self.last = None,
            }
        }
    }

    /// Marks with `stamp` the blocks that can run last before block `own`: the blocks reached
    /// walking back from it through join, start and exit points. Each point is marked once.
    fn mark_predecessors(
        &self,
        own: usize,
        stamp: usize,
        marks: &mut [usize],
        stack: &mut Vec<usize>,
    ) {
        stack.clear();
        stack.push(own);
        while let Some(point) = stack.pop() {
            for &pred in self.points.get(point).map_or(&[][..], Vec::as_slice) {
                if let Some(mark) = marks.get_mut(pred)
                    && *mark != stamp
                {
                    *mark = stamp;
                    if pred >= self.blocks {
                        stack.push(pred);
                    }
                }
            }
        }
    }

    /// Whether a value listed in block `def` is in scope at the end of block `at` (both block
    /// indices; see the module docs, "Scope").
    fn in_scope_at_end(&self, def: usize, at: usize) -> bool {
        let interval = |block: usize| {
            self.body
                .get(block)
                .and_then(|&body| self.bodies.get(body))
                .copied()
        };
        let placed = matches!(
            (self.place.get(def), self.place.get(at)),
            (Some(def_place), Some(at_place)) if def_place <= at_place
        );
        let enclosed = matches!(
            (interval(def), interval(at)),
            (Some((open, close)), Some((at_open, at_close))) if open <= at_open && at_close <= close
        );
        placed && enclosed
    }
}
