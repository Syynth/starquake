//! Structured control flow: the IR's tree of [`Node`]s as naga statements (the parent module's
//! docs, "Structured control flow").
//!
//! The walk recurses once per nested `if` or loop. The verifier, which [`super::lower_with`]
//! runs first, refuses a tree nesting deeper than [`crate::ir::MAX_NESTING`] levels, so the
//! recursion is at most that deep.

use naga::{Block, Expression, Literal, Statement};

use super::builder::push;
use super::{LowerError, Lowering};
use crate::ir::{Function, Inst, Node, Operand};

impl Lowering {
    /// Lowers the statements `nodes` of `function` into `body`.
    pub(super) fn nodes(
        &mut self,
        function: &Function,
        nodes: &[Node],
        body: &mut Block,
    ) -> Result<(), LowerError> {
        for node in nodes {
            match node {
                Node::Block(id) => {
                    let block = function
                        .block(*id)
                        .ok_or_else(|| LowerError::Internal(format!("no {id}")))?;
                    for &at in &block.insts {
                        match function.inst(at) {
                            Some(Inst::Op(op)) => self.inst(body, at, op)?,
                            // Phis come first in a block (the verifier checks), so every load
                            // happens before any store: they take their values together.
                            Some(Inst::Phi(_)) => self.phi_load(body, at)?,
                            None => {
                                return Err(LowerError::Internal(format!("{at} is missing")));
                            }
                        }
                    }
                    self.phi_stores(body, *id)?;
                }
                Node::If { cond, body: inner } => {
                    let condition = self.condition(body, *cond)?;
                    let mut accept = Block::new();
                    let mark = self.scope_log.len();
                    self.nodes(function, inner, &mut accept)?;
                    self.end_scope(mark);
                    push(
                        body,
                        Statement::If {
                            condition,
                            accept,
                            reject: Block::new(),
                        },
                    );
                }
                Node::Loop {
                    body: inner,
                    repeat,
                } => {
                    let mut looped = Block::new();
                    let mark = self.scope_log.len();
                    self.nodes(function, inner, &mut looped)?;
                    // The repeat condition is read after the body, in its scope.
                    let condition = self.condition(&mut looped, *repeat)?;
                    self.end_scope(mark);
                    push(
                        &mut looped,
                        Statement::If {
                            condition,
                            accept: Block::new(),
                            reject: Block::from_vec(vec![Statement::Break]),
                        },
                    );
                    push(
                        body,
                        Statement::Loop {
                            body: looped,
                            continuing: Block::new(),
                            break_if: None,
                        },
                    );
                }
                Node::Break { cond } => {
                    let condition = self.condition(body, *cond)?;
                    push(
                        body,
                        Statement::If {
                            condition,
                            accept: Block::from_vec(vec![Statement::Break]),
                            reject: Block::new(),
                        },
                    );
                }
                // A return does the epilogue's emulation first (the `emulation` submodule).
                Node::Return => {
                    self.epilogue(body)?;
                    self.ret(body);
                }
                // `unreachable` returns too (a divergence in the parent module's docs), with no
                // epilogue.
                Node::Unreachable => self.ret(body),
            }
        }
        Ok(())
    }

    /// A tree condition (a `u1`): a value's expression, or a boolean literal.
    fn condition(
        &mut self,
        body: &mut Block,
        cond: Operand,
    ) -> Result<naga::Handle<Expression>, LowerError> {
        match cond {
            Operand::Value(value) => self.value(value),
            Operand::Imm(crate::ir::Imm::U1(value)) => {
                Ok(self.expr(body, Expression::Literal(Literal::Bool(value))))
            }
            Operand::Imm(other) => Err(LowerError::Internal(format!(
                "a tree condition is the immediate {other}"
            ))),
        }
    }
}
