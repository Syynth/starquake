//! The emission: the statement tree as the IR's structured [`Node`]s (Eden's `TranslatePass`
//! and `VisitExpr`; parent module docs, step 3).
//!
//! Eden emits into blocks it makes as it goes, translating each code statement into the
//! current one. Here code statements are the function's own blocks and stay as they are; the
//! instructions structuring adds go into a new block of their own (a "glue" block), made when
//! the first of them is emitted and closed by the next node that is not one of them, so the
//! instructions run in Eden's order. A condition is computed in the glue block in front of
//! the node that tests it; a loop's repeat condition in a glue block at the end of its body
//! (Eden's continue block).

use super::tree::{Expr, ExprId, Id, Kind, ROOT, Tree};
use super::{Invariant, StructureError};
use crate::ir::{BlockId, Builder, Function, Imm, MAX_NESTING, Node, Opcode, Operand};

/// Emits the tree into `function` and returns the structured tree's nodes.
pub(super) fn emit(tree: &mut Tree, function: &mut Function) -> Result<Vec<Node>, StructureError> {
    let mut emitter = Emitter { tree, function };
    let mut list = List::new();
    // Eden puts `Prologue` at the start of the first block it makes, which holds the goto
    // variables' first resets.
    emitter.glue_op(&mut list, Opcode::Prologue, &[])?;
    emitter.list(ROOT, 0, &mut list)?;
    if list.open {
        list.nodes.push(Node::Unreachable);
    }
    Ok(list.nodes)
}

/// A statement list being emitted.
struct List {
    nodes: Vec<Node>,
    /// The open glue block, if any.
    glue: Option<BlockId>,
    /// Whether control can reach the end of what is emitted so far (Eden's `current_block` is
    /// set).
    open: bool,
}

impl List {
    /// An empty list, with no glue block open and nothing to fall off the end of yet.
    const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            glue: None,
            open: false,
        }
    }
}

/// The tree being emitted and the function it is emitted into.
struct Emitter<'a> {
    tree: &'a mut Tree,
    function: &'a mut Function,
}

impl Emitter<'_> {
    /// Emits the children of `parent`, a list at nesting level `depth`, into `list`.
    fn list(&mut self, parent: Id, depth: usize, list: &mut List) -> Result<(), StructureError> {
        let mut child = self.tree.first_child(parent)?;
        while let Some(id) = child {
            self.tree.charge()?;
            match self.tree.kind(id)? {
                Kind::Root | Kind::Goto { .. } => {
                    return Err(StructureError::Internal(Invariant::Kind));
                }
                Kind::Label(_) => {}
                Kind::Code(block) => {
                    list.glue = None;
                    list.nodes.push(Node::Block(block));
                    list.open = true;
                }
                Kind::SetVariable { id, value } => {
                    let value = self.condition(list, value)?;
                    self.glue_op(
                        list,
                        Opcode::SetGotoVariable,
                        &[Operand::Imm(Imm::U32(id)), value],
                    )?;
                }
                Kind::If { cond } => {
                    let cond = self.condition_ref(list, cond)?;
                    let body = self.body(id, depth)?.nodes;
                    list.glue = None;
                    list.nodes.push(Node::If { cond, body });
                    list.open = true;
                }
                Kind::Loop { cond } => {
                    // The repeat condition goes in a block of its own at the end of the body
                    // (Eden's continue block).
                    let mut body = self.body(id, depth)?;
                    let repeat = self.condition_ref(&mut body, cond)?;
                    list.glue = None;
                    list.nodes.push(Node::Loop {
                        body: body.nodes,
                        repeat,
                    });
                    list.open = true;
                }
                Kind::Break { cond } => {
                    let cond = self.condition_ref(list, cond)?;
                    list.glue = None;
                    list.nodes.push(Node::Break { cond });
                    list.open = true;
                }
                Kind::Return => {
                    self.glue_op(list, Opcode::Epilogue, &[])?;
                    list.glue = None;
                    list.nodes.push(Node::Return);
                    list.open = false;
                }
                Kind::Unreachable => {
                    list.glue = None;
                    list.nodes.push(Node::Unreachable);
                    list.open = false;
                }
            }
            child = self.tree.next_sibling(id)?;
        }
        Ok(())
    }

    /// The body of `parent`, an `if` or loop in a list at `depth`: its nesting level is checked
    /// against [`MAX_NESTING`] before it is emitted, so the recursion is bounded.
    fn body(&mut self, parent: Id, depth: usize) -> Result<List, StructureError> {
        let inner = depth
            .checked_add(1)
            .filter(|&inner| inner <= MAX_NESTING)
            .ok_or(StructureError::TooDeep)?;
        let mut body = List::new();
        self.list(parent, inner, &mut body)?;
        // Whatever follows the body's last statement (a loop's repeat condition) goes in a new
        // block.
        body.glue = None;
        Ok(body)
    }

    /// `ConditionRef` of the condition `cond`, in the list's glue block (Eden wraps the
    /// condition of each `if`, `break` and loop repeat in one).
    fn condition_ref(&mut self, list: &mut List, cond: ExprId) -> Result<Operand, StructureError> {
        let value = self.condition(list, cond)?;
        self.glue_op(list, Opcode::ConditionRef, &[value])
    }

    /// Computes the condition `cond` in the list's glue block (Eden's `VisitExpr`), without
    /// recursion: conditions can nest as deep as the gotos that moved into one `if`.
    fn condition(&mut self, list: &mut List, cond: ExprId) -> Result<Operand, StructureError> {
        /// A step of the walk: compute a condition, or finish a `Not` or an `Or` whose
        /// operands are computed.
        enum Step {
            Visit(ExprId),
            Not,
            Or,
        }
        let mut steps = vec![Step::Visit(cond)];
        let mut values: Vec<Operand> = Vec::new();
        while let Some(step) = steps.pop() {
            self.tree.charge()?;
            match step {
                Step::Visit(id) => match self.tree.expr_of(id)? {
                    Expr::Operand(operand) => values.push(operand),
                    Expr::Variable(label) => {
                        let value = self.glue_op(
                            list,
                            Opcode::GetGotoVariable,
                            &[Operand::Imm(Imm::U32(label))],
                        )?;
                        values.push(value);
                    }
                    Expr::Not(operand) => {
                        steps.push(Step::Not);
                        steps.push(Step::Visit(operand));
                    }
                    Expr::Or(a, b) => {
                        steps.push(Step::Or);
                        steps.push(Step::Visit(b));
                        steps.push(Step::Visit(a));
                    }
                },
                Step::Not => {
                    let operand = pop(&mut values)?;
                    let value = self.glue_op(list, Opcode::LogicalNot, &[operand])?;
                    values.push(value);
                }
                Step::Or => {
                    let b = pop(&mut values)?;
                    let a = pop(&mut values)?;
                    let value = self.glue_op(list, Opcode::LogicalOr, &[a, b])?;
                    values.push(value);
                }
            }
        }
        let value = pop(&mut values)?;
        if values.is_empty() {
            Ok(value)
        } else {
            Err(StructureError::Internal(Invariant::Kind))
        }
    }

    /// Appends `opcode` with `args` to the list's glue block, making one if none is open.
    fn glue_op(
        &mut self,
        list: &mut List,
        opcode: Opcode,
        args: &[Operand],
    ) -> Result<Operand, StructureError> {
        let block = if let Some(block) = list.glue {
            block
        } else {
            let block = self.function.add_block()?;
            list.nodes.push(Node::Block(block));
            list.glue = Some(block);
            list.open = true;
            block
        };
        let value = Builder::new(self.function, block).op(opcode, args)?;
        Ok(Operand::Value(value))
    }
}

/// The last computed value of a condition walk.
fn pop(values: &mut Vec<Operand>) -> Result<Operand, StructureError> {
    values
        .pop()
        .ok_or(StructureError::Internal(Invariant::Kind))
}
