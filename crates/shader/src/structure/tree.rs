//! The statement tree and the goto pass (Eden's `Statement`, `GotoPass`; parent module docs,
//! steps 1 and 2).
//!
//! Statements live in an arena and are linked as Eden's intrusive lists are: each knows its
//! parent, its siblings and, for the root, an `if` or a loop, its first and last child, so a
//! run of siblings moves to a new parent in time linear in its length (Eden's `splice`, then
//! `UpdateTreeUp`). Conditions (Eden's expression statements `Identity`, `Not`, `Or` and
//! `Variable`) live in a second arena and may be shared, as Eden shares them. A statement taken
//! out of the tree stays in the arena, unlinked. Every step is charged to the work budget
//! (parent module docs, "Limits").

use super::{Invariant, StructureError};
use crate::ir::{BlockId, Function, Imm, Operand, Terminator};

/// A statement's index in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Id(usize);

/// A condition's index in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExprId(usize);

/// A condition (Eden's expression statements).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Expr {
    /// A value or immediate of the graph (Eden's `Identity`, a guest condition).
    Operand(Operand),
    /// Its operand is false.
    Not(ExprId),
    /// Either operand is true.
    Or(ExprId, ExprId),
    /// The goto variable of a label (Eden's `Variable`).
    Variable(u32),
}

/// What a statement is (Eden's `StatementType`, without the expression kinds, `Kill`,
/// `Function` and `SetIndirectBranchVariable`, which this IR does not need).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// The function: the tree's root (Eden's `FunctionTag`).
    Root,
    /// A block of the graph.
    Code(BlockId),
    /// Go to a label when the condition holds.
    Goto {
        /// The condition.
        cond: ExprId,
        /// The label.
        label: Id,
    },
    /// A label, by its id (the block's place in the layout).
    Label(u32),
    /// Run the children when the condition holds.
    If {
        /// The condition.
        cond: ExprId,
    },
    /// Run the children, then again while the condition holds.
    Loop {
        /// The repeat condition.
        cond: ExprId,
    },
    /// Leave the innermost loop when the condition holds.
    Break {
        /// The condition.
        cond: ExprId,
    },
    /// Return from the program.
    Return,
    /// Control never gets here.
    Unreachable,
    /// Set a label's goto variable.
    SetVariable {
        /// The label's id.
        id: u32,
        /// The value.
        value: ExprId,
    },
}

/// A statement and its links.
#[derive(Clone, Debug)]
struct Stmt {
    kind: Kind,
    up: Option<Id>,
    prev: Option<Id>,
    next: Option<Id>,
    first: Option<Id>,
    last: Option<Id>,
}

/// The statement tree, its conditions and the work spent on them.
#[derive(Debug)]
pub(super) struct Tree {
    stmts: Vec<Stmt>,
    exprs: Vec<Expr>,
    /// The gotos `build` made, in the order it made them.
    gotos: Vec<Id>,
    work: u64,
    limit: u64,
}

/// The root's id: the first statement made.
pub(super) const ROOT: Id = Id(0);

impl Tree {
    /// The work spent so far.
    pub(super) const fn work(&self) -> u64 {
        self.work
    }

    /// Charges one unit of work, refusing one past the limit.
    pub(super) fn charge(&mut self) -> Result<(), StructureError> {
        if self.work >= self.limit {
            return Err(StructureError::WorkLimit { limit: self.limit });
        }
        self.work += 1;
        Ok(())
    }

    /// The statement list for `function`, its blocks laid out in `order` (step 1).
    pub(super) fn build(
        function: &Function,
        order: &[BlockId],
        limit: u64,
    ) -> Result<Self, StructureError> {
        let mut tree = Self {
            stmts: Vec::new(),
            exprs: Vec::new(),
            gotos: Vec::new(),
            work: 0,
            limit,
        };
        tree.make(Kind::Root)?;
        let false_cond = tree.expr(Expr::Operand(Operand::Imm(Imm::U1(false))))?;
        // The label of each block, by the block's arena slot.
        let mut labels: Vec<Option<Id>> = vec![None; function.block_count()];
        let mut ordered = Vec::with_capacity(order.len());
        for (id, &block) in (0_u32..).zip(order) {
            let label = tree.make(Kind::Label(id))?;
            tree.insert(ROOT, None, label)?;
            if let Some(slot) = slot(block).and_then(|slot| labels.get_mut(slot)) {
                *slot = Some(label);
            }
            ordered.push((id, block, label));
        }
        let label_of = |block: BlockId| {
            slot(block)
                .and_then(|slot| labels.get(slot))
                .copied()
                .flatten()
                .ok_or(StructureError::Internal(Invariant::Kind))
        };
        // The goto variables that hold computed branch conditions come after the labels'.
        let mut next_condition =
            u32::try_from(order.len()).map_err(|_| StructureError::Internal(Invariant::Kind))?;
        for (id, block, label) in ordered {
            let ip = tree.stmt(label)?.next;
            let first = tree.stmt(ROOT)?.first;
            let reset_front = tree.make(Kind::SetVariable {
                id,
                value: false_cond,
            })?;
            tree.insert(ROOT, first, reset_front)?;
            let reset = tree.make(Kind::SetVariable {
                id,
                value: false_cond,
            })?;
            tree.insert(ROOT, ip, reset)?;
            let code = tree.make(Kind::Code(block))?;
            tree.insert(ROOT, ip, code)?;
            let terminator = function
                .block(block)
                .map_or(Terminator::None, |b| b.terminator);
            let always = Expr::Operand(Operand::Imm(Imm::U1(true)));
            match terminator {
                Terminator::Jump(target) => {
                    tree.goto(ip, always, label_of(target)?)?;
                }
                Terminator::Branch {
                    cond,
                    then,
                    otherwise,
                } => match cond {
                    Operand::Imm(Imm::U1(true)) => tree.goto(ip, always, label_of(then)?)?,
                    Operand::Imm(Imm::U1(false)) => tree.goto(ip, always, label_of(otherwise)?)?,
                    _ => {
                        // The condition is a value of the block: it is kept in a goto variable
                        // of its own, reset at the front and set right after the block, which
                        // the goto reads wherever it moves (Eden reads the guest predicate
                        // again there).
                        let variable = next_condition;
                        next_condition = next_condition
                            .checked_add(1)
                            .ok_or(StructureError::Internal(Invariant::Kind))?;
                        let first = tree.stmt(ROOT)?.first;
                        let reset_front = tree.make(Kind::SetVariable {
                            id: variable,
                            value: false_cond,
                        })?;
                        tree.insert(ROOT, first, reset_front)?;
                        let value = tree.expr(Expr::Operand(cond))?;
                        let set = tree.make(Kind::SetVariable { id: variable, value })?;
                        tree.insert(ROOT, ip, set)?;
                        tree.goto(ip, Expr::Variable(variable), label_of(then)?)?;
                        tree.goto(ip, always, label_of(otherwise)?)?;
                    }
                },
                Terminator::Return => {
                    let ret = tree.make(Kind::Return)?;
                    tree.insert(ROOT, ip, ret)?;
                }
                Terminator::Unreachable => {
                    let unreachable = tree.make(Kind::Unreachable)?;
                    tree.insert(ROOT, ip, unreachable)?;
                }
                Terminator::None => {
                    return Err(StructureError::Internal(Invariant::Unterminated(block)));
                }
            }
        }
        Ok(tree)
    }

    /// Adds a goto to `label` when `cond` holds before `ip` in the root and records it.
    fn goto(&mut self, ip: Option<Id>, cond: Expr, label: Id) -> Result<(), StructureError> {
        let cond = self.expr(cond)?;
        let goto = self.make(Kind::Goto { cond, label })?;
        self.insert(ROOT, ip, goto)?;
        self.gotos.push(goto);
        Ok(())
    }

    /// Removes every goto, the last made first (step 2, Eden's `GotoPass` constructor).
    pub(super) fn remove_gotos(&mut self) -> Result<(), StructureError> {
        let gotos = std::mem::take(&mut self.gotos);
        for &goto in gotos.iter().rev() {
            self.remove_goto(goto)?;
        }
        Ok(())
    }

    /// Eden's `RemoveGoto`.
    fn remove_goto(&mut self, mut goto: Id) -> Result<(), StructureError> {
        let (_, label) = self.goto_parts(goto)?;
        // Make the goto and its label directly related.
        if self.indirectly_related(goto, label)? {
            while !self.directly_related(goto, label)? {
                goto = self.move_outward(goto)?;
            }
        }
        // Make them siblings.
        if self.directly_related(goto, label)? {
            let label_level = self.level(label)?;
            let mut goto_level = self.level(goto)?;
            if goto_level > label_level {
                while goto_level > label_level {
                    goto = self.move_outward(goto)?;
                    goto_level -= 1;
                }
            } else {
                if self.needs_lift(goto, label)? {
                    goto = self.lift(goto)?;
                }
                while goto_level < label_level {
                    goto = self.move_inward(goto)?;
                    goto_level += 1;
                }
            }
        }
        // They are siblings: eliminate the goto.
        if self.stmt(goto)?.next == Some(label) {
            self.charge()?;
            self.unlink(goto)?;
        } else if self.are_ordered(goto, label)? {
            self.eliminate_as_conditional(goto, label)?;
        } else {
            self.eliminate_as_loop(goto, label)?;
        }
        Ok(())
    }

    /// Eden's `EliminateAsConditional`: `if (!cond) { the statements up to the label }`.
    fn eliminate_as_conditional(&mut self, goto: Id, label: Id) -> Result<(), StructureError> {
        let (cond, _) = self.goto_parts(goto)?;
        let parent = self.parent(goto)?;
        let not = self.expr(Expr::Not(cond))?;
        let if_stmt = self.make(Kind::If { cond: not })?;
        let start = self.stmt(goto)?.next;
        self.splice(if_stmt, start, Some(label))?;
        self.insert(parent, Some(goto), if_stmt)?;
        self.unlink(goto)
    }

    /// Eden's `EliminateAsLoop`: `do { the statements from the label } while (cond)`.
    fn eliminate_as_loop(&mut self, goto: Id, label: Id) -> Result<(), StructureError> {
        let (cond, _) = self.goto_parts(goto)?;
        let parent = self.parent(goto)?;
        let loop_stmt = self.make(Kind::Loop { cond })?;
        self.splice(loop_stmt, Some(label), Some(goto))?;
        self.insert(parent, Some(goto), loop_stmt)?;
        self.unlink(goto)
    }

    /// Eden's `MoveOutward`.
    fn move_outward(&mut self, goto: Id) -> Result<Id, StructureError> {
        let parent = self.parent(goto)?;
        match self.stmt(parent)?.kind {
            Kind::If { .. } => self.move_outward_if(goto),
            Kind::Loop { .. } => self.move_outward_loop(goto),
            _ => Err(StructureError::Internal(Invariant::Movement)),
        }
    }

    /// Eden's `MoveInward`: set the variable, skip the statements before the one holding the
    /// label when it is set, and test it again first thing inside that statement.
    fn move_inward(&mut self, goto: Id) -> Result<Id, StructureError> {
        let parent = self.parent(goto)?;
        let (cond, label) = self.goto_parts(goto)?;
        let label_nested = self.sibling_from_nephew(goto, label)?;
        let label_id = self.label_id(label)?;

        let set_var = self.make(Kind::SetVariable {
            id: label_id,
            value: cond,
        })?;
        self.insert(parent, Some(goto), set_var)?;

        let variable = self.expr(Expr::Variable(label_id))?;
        let neg_var = self.expr(Expr::Not(variable))?;
        let start = self.stmt(goto)?.next;
        if start != Some(label_nested) {
            let if_stmt = self.make(Kind::If { cond: neg_var })?;
            self.splice(if_stmt, start, Some(label_nested))?;
            self.insert(parent, Some(goto), if_stmt)?;
        }
        self.unlink(goto)?;

        match self.stmt(label_nested)?.kind {
            Kind::If { cond: nested_cond } => {
                let or = self.expr(Expr::Or(variable, nested_cond))?;
                self.stmt_mut(label_nested)?.kind = Kind::If { cond: or };
            }
            Kind::Loop { .. } => {}
            _ => return Err(StructureError::Internal(Invariant::Movement)),
        }
        let new_goto = self.make(Kind::Goto {
            cond: variable,
            label,
        })?;
        let first = self.stmt(label_nested)?.first;
        self.insert(label_nested, first, new_goto)?;
        Ok(new_goto)
    }

    /// Eden's `Lift`: wrap the statements from the one holding the label up to the goto in a
    /// loop that starts with the goto, tested through the variable, and ends by setting it.
    fn lift(&mut self, goto: Id) -> Result<Id, StructureError> {
        let parent = self.parent(goto)?;
        let (cond, label) = self.goto_parts(goto)?;
        let label_id = self.label_id(label)?;
        let label_nested = self.sibling_from_nephew(goto, label)?;

        let variable = self.expr(Expr::Variable(label_id))?;
        let loop_stmt = self.make(Kind::Loop { cond: variable })?;
        self.splice(loop_stmt, Some(label_nested), Some(goto))?;
        // Eden's `SanitizeNoBreaks`: the loop's direct children only.
        let mut child = self.stmt(loop_stmt)?.first;
        while let Some(id) = child {
            self.charge()?;
            let stmt = self.stmt(id)?;
            if matches!(stmt.kind, Kind::Break { .. }) {
                return Err(StructureError::LiftOverBreak);
            }
            child = stmt.next;
        }
        self.insert(parent, Some(goto), loop_stmt)?;

        let new_goto = self.make(Kind::Goto {
            cond: variable,
            label,
        })?;
        let first = self.stmt(loop_stmt)?.first;
        self.insert(loop_stmt, first, new_goto)?;

        let set_var = self.make(Kind::SetVariable {
            id: label_id,
            value: cond,
        })?;
        self.insert(loop_stmt, None, set_var)?;

        self.unlink(goto)?;
        Ok(new_goto)
    }

    /// Eden's `MoveOutwardIf`: set the variable, skip the rest of the `if` when it is set, and
    /// test it again after the `if`.
    ///
    /// Eden then drops the first statement of the new `if`'s body (`if_body.pop_front()` after
    /// splicing from `std::next(goto_stmt)`), which would lose that statement. This keeps it
    /// (a listed divergence, pinned by `tree::tests::moving_out_of_an_if_keeps_every_statement`).
    /// No input reaches this function through [`Tree::remove_gotos`] as far as the tests
    /// show: gotos are removed from the last made to the first, so the gotos still waiting are
    /// the ones before the goto being removed, and the only gotos an `if` body receives are the
    /// ones inward movement puts there, whose labels are inside it. None of 20,000 generated
    /// graphs reached it (a count taken by hand while building this slice).
    fn move_outward_if(&mut self, goto: Id) -> Result<Id, StructureError> {
        let parent = self.parent(goto)?;
        let (cond, label) = self.goto_parts(goto)?;
        let label_id = self.label_id(label)?;
        let set_var = self.make(Kind::SetVariable {
            id: label_id,
            value: cond,
        })?;
        self.insert(parent, Some(goto), set_var)?;

        let variable = self.expr(Expr::Variable(label_id))?;
        let neg_cond = self.expr(Expr::Not(variable))?;
        let if_stmt = self.make(Kind::If { cond: neg_cond })?;
        let start = self.stmt(goto)?.next;
        self.splice(if_stmt, start, None)?;
        self.insert(parent, Some(goto), if_stmt)?;
        self.unlink(goto)?;

        let new_cond = self.expr(Expr::Variable(label_id))?;
        let new_goto = self.make(Kind::Goto {
            cond: new_cond,
            label,
        })?;
        let grandparent = self.parent(parent)?;
        let after = self.stmt(parent)?.next;
        self.insert(grandparent, after, new_goto)?;
        Ok(new_goto)
    }

    /// Eden's `MoveOutwardLoop`: set the variable and break when it is set, and test it again
    /// after the loop.
    fn move_outward_loop(&mut self, goto: Id) -> Result<Id, StructureError> {
        let parent = self.parent(goto)?;
        let (cond, label) = self.goto_parts(goto)?;
        let label_id = self.label_id(label)?;
        let set_var = self.make(Kind::SetVariable {
            id: label_id,
            value: cond,
        })?;
        let variable = self.expr(Expr::Variable(label_id))?;
        let break_stmt = self.make(Kind::Break { cond: variable })?;
        self.insert(parent, Some(goto), set_var)?;
        self.insert(parent, Some(goto), break_stmt)?;
        self.unlink(goto)?;

        let new_cond = self.expr(Expr::Variable(label_id))?;
        let new_goto = self.make(Kind::Goto {
            cond: new_cond,
            label,
        })?;
        let grandparent = self.parent(parent)?;
        let after = self.stmt(parent)?.next;
        self.insert(grandparent, after, new_goto)?;
        Ok(new_goto)
    }

    /// Eden's `Level`: how many statements are above `stmt`.
    fn level(&mut self, stmt: Id) -> Result<usize, StructureError> {
        let mut level = 0_usize;
        let mut node = self.stmt(stmt)?.up;
        while let Some(id) = node {
            self.charge()?;
            level = level
                .checked_add(1)
                .filter(|&level| level <= self.stmts.len())
                .ok_or(StructureError::Internal(Invariant::Parents))?;
            node = self.stmt(id)?.up;
        }
        Ok(level)
    }

    /// Eden's `IsDirectlyRelated`: the shallower of the two is a sibling of an ancestor of the
    /// deeper one (or of the deeper one itself).
    fn directly_related(&mut self, goto: Id, label: Id) -> Result<bool, StructureError> {
        let goto_level = self.level(goto)?;
        let label_level = self.level(label)?;
        let (min_level, mut max_level, min, mut max) = if label_level < goto_level {
            (label_level, goto_level, label, goto)
        } else {
            (goto_level, label_level, goto, label)
        };
        while max_level > min_level {
            self.charge()?;
            max_level -= 1;
            max = self.parent(max)?;
        }
        Ok(self.stmt(min)?.up == self.stmt(max)?.up)
    }

    /// Eden's `IsIndirectlyRelated`.
    fn indirectly_related(&mut self, goto: Id, label: Id) -> Result<bool, StructureError> {
        Ok(self.stmt(goto)?.up != self.stmt(label)?.up && !self.directly_related(goto, label)?)
    }

    /// Eden's `SiblingFromNephew`: the ancestor of `nephew` (or `nephew`) that is a sibling of
    /// `uncle`.
    fn sibling_from_nephew(&mut self, uncle: Id, nephew: Id) -> Result<Id, StructureError> {
        let parent = self.stmt(uncle)?.up;
        let mut it = nephew;
        let mut steps = 0_usize;
        while self.stmt(it)?.up != parent {
            self.charge()?;
            steps += 1;
            if steps > self.stmts.len() {
                return Err(StructureError::Internal(Invariant::Parents));
            }
            it = self.parent(it)?;
        }
        Ok(it)
    }

    /// Eden's `AreOrdered`: whether `left` comes before `right`, two siblings. Eden scans from
    /// `right` to the end of the list for `left`; this scans from both at once and stops at
    /// whichever settles it first, with the same answer for siblings (false when they are the
    /// same statement).
    fn are_ordered(&mut self, left: Id, right: Id) -> Result<bool, StructureError> {
        let mut from_right = Some(right);
        let mut from_left = Some(left);
        loop {
            self.charge()?;
            match from_right {
                None => return Ok(true),
                Some(id) if id == left => return Ok(false),
                Some(id) => from_right = self.stmt(id)?.next,
            }
            match from_left {
                None => return Ok(false),
                Some(id) if id == right => return Ok(true),
                Some(id) => from_left = self.stmt(id)?.next,
            }
        }
    }

    /// Eden's `NeedsLift`: whether the statement holding the label comes before the goto.
    fn needs_lift(&mut self, goto: Id, label: Id) -> Result<bool, StructureError> {
        let sibling = self.sibling_from_nephew(goto, label)?;
        self.are_ordered(sibling, goto)
    }

    /// A goto's condition and label.
    fn goto_parts(&self, goto: Id) -> Result<(ExprId, Id), StructureError> {
        match self.stmt(goto)?.kind {
            Kind::Goto { cond, label } => Ok((cond, label)),
            _ => Err(StructureError::Internal(Invariant::Kind)),
        }
    }

    /// A label's id.
    fn label_id(&self, label: Id) -> Result<u32, StructureError> {
        match self.stmt(label)?.kind {
            Kind::Label(id) => Ok(id),
            _ => Err(StructureError::Internal(Invariant::Kind)),
        }
    }

    /// The parent of `stmt`, which must have one.
    fn parent(&self, stmt: Id) -> Result<Id, StructureError> {
        self.stmt(stmt)?
            .up
            .ok_or(StructureError::Internal(Invariant::Parents))
    }

    /// The statement `id` names.
    fn stmt(&self, id: Id) -> Result<&Stmt, StructureError> {
        self.stmts
            .get(id.0)
            .ok_or(StructureError::Internal(Invariant::Parents))
    }

    fn stmt_mut(&mut self, id: Id) -> Result<&mut Stmt, StructureError> {
        self.stmts
            .get_mut(id.0)
            .ok_or(StructureError::Internal(Invariant::Parents))
    }

    /// What `id` is.
    pub(super) fn kind(&self, id: Id) -> Result<Kind, StructureError> {
        Ok(self.stmt(id)?.kind)
    }

    /// The first child of `id`.
    pub(super) fn first_child(&self, id: Id) -> Result<Option<Id>, StructureError> {
        Ok(self.stmt(id)?.first)
    }

    /// The sibling after `id`.
    pub(super) fn next_sibling(&self, id: Id) -> Result<Option<Id>, StructureError> {
        Ok(self.stmt(id)?.next)
    }

    /// The condition `id` names.
    pub(super) fn expr_of(&self, id: ExprId) -> Result<Expr, StructureError> {
        self.exprs
            .get(id.0)
            .copied()
            .ok_or(StructureError::Internal(Invariant::Kind))
    }

    /// Makes an unlinked statement.
    fn make(&mut self, kind: Kind) -> Result<Id, StructureError> {
        self.charge()?;
        let id = Id(self.stmts.len());
        self.stmts.push(Stmt {
            kind,
            up: None,
            prev: None,
            next: None,
            first: None,
            last: None,
        });
        Ok(id)
    }

    /// Makes a condition.
    fn expr(&mut self, expr: Expr) -> Result<ExprId, StructureError> {
        self.charge()?;
        let id = ExprId(self.exprs.len());
        self.exprs.push(expr);
        Ok(id)
    }

    /// Links the unlinked `stmt` into `parent`'s children before `before`, or at the end.
    fn insert(&mut self, parent: Id, before: Option<Id>, stmt: Id) -> Result<(), StructureError> {
        let prev = match before {
            Some(before) => {
                if self.stmt(before)?.up != Some(parent) {
                    return Err(StructureError::Internal(Invariant::Siblings));
                }
                self.stmt(before)?.prev
            }
            None => self.stmt(parent)?.last,
        };
        {
            let node = self.stmt_mut(stmt)?;
            node.up = Some(parent);
            node.prev = prev;
            node.next = before;
        }
        match prev {
            Some(prev) => self.stmt_mut(prev)?.next = Some(stmt),
            None => self.stmt_mut(parent)?.first = Some(stmt),
        }
        match before {
            Some(before) => self.stmt_mut(before)?.prev = Some(stmt),
            None => self.stmt_mut(parent)?.last = Some(stmt),
        }
        Ok(())
    }

    /// Takes `stmt` out of its parent's children; it stays in the arena, unlinked.
    fn unlink(&mut self, stmt: Id) -> Result<(), StructureError> {
        let (up, prev, next) = {
            let node = self.stmt(stmt)?;
            (node.up, node.prev, node.next)
        };
        let parent = up.ok_or(StructureError::Internal(Invariant::Parents))?;
        match prev {
            Some(prev) => self.stmt_mut(prev)?.next = next,
            None => self.stmt_mut(parent)?.first = next,
        }
        match next {
            Some(next) => self.stmt_mut(next)?.prev = prev,
            None => self.stmt_mut(parent)?.last = prev,
        }
        let node = self.stmt_mut(stmt)?;
        node.up = None;
        node.prev = None;
        node.next = None;
        Ok(())
    }

    /// Moves the siblings from `start` up to, not including, `end` (to the end of the list for
    /// `None`) to the end of `to`'s children (Eden's `splice` and `UpdateTreeUp`).
    fn splice(
        &mut self,
        to: Id,
        start: Option<Id>,
        end: Option<Id>,
    ) -> Result<(), StructureError> {
        let mut current = start;
        while current != end {
            let Some(id) = current else {
                // The end is not after the start in the list.
                return Err(StructureError::Internal(Invariant::Siblings));
            };
            self.charge()?;
            current = self.stmt(id)?.next;
            self.unlink(id)?;
            self.insert(to, None, id)?;
        }
        Ok(())
    }
}

/// The arena slot of `block`.
fn slot(block: BlockId) -> Option<usize> {
    usize::try_from(block.index()).ok()
}
