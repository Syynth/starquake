//! Identity removal: every argument that names an `Identity` operation is rewired to the value
//! at the end of its chain, and `Identity` and `Void` operations leave their blocks.
//!
//! Ported by behaviour from Eden's `eden/src/shader_recompiler/ir_opt/identity_removal_pass.cpp`
//! @ cf654052 (GPL-3.0-or-later, over GPL-2.0-or-later yuzu code). As there, every block is
//! walked (in arena order here, Eden's `program.blocks`), each instruction's arguments (a phi's
//! incoming values included) are rewired, `Identity` and `Void` operations are taken out of
//! their block, and the removed instructions are invalidated (a `Void` with no arguments) only
//! after the walk, so a chain through one removed earlier still resolves.
//!
//! Where this pass departs from Eden's, each pinned by a test in `tests`:
//!
//! - Terminator conditions and the structured tree's conditions are rewired too. Eden keeps
//!   conditions as instruction arguments (`ConditionRef`), which its walk reaches; here they
//!   live in the terminator and the tree, and an `Identity` they name would otherwise be left
//!   dangling (`tests::identity::conditions_are_rewired`).
//! - Each chain is followed once and its end remembered, so a long chain costs its length once
//!   rather than once per use (Eden follows it from every use). A chain that loops, or ends at
//!   an `Identity` without an argument or at an id with no instruction, is
//!   [`PassError::Identity`] naming the first `Identity` of the chain met; Eden would loop
//!   forever or read past its arguments (`tests::identity::cycles_are_refused`).

use crate::ir::{Function, Inst, Opcode, Operand, Terminator, Value};

use super::{PassError, Work, arena_len, invalidated, tree_conditions_mut};

/// Follows `Identity` chains, remembering each chain's end.
pub(super) struct Resolver {
    /// The end of each `Identity`'s chain, once found.
    ends: Vec<Option<Operand>>,
    /// Scratch: the chain being followed.
    chain: Vec<Value>,
}

impl Resolver {
    /// A resolver for `function`'s arena.
    pub(super) fn new(function: &Function) -> Self {
        Self {
            ends: vec![None; arena_len(function)],
            chain: Vec::new(),
        }
    }

    /// `operand`, or the end of its `Identity` chain.
    pub(super) fn resolve(
        &mut self,
        function: &Function,
        operand: Operand,
        work: &mut Work,
    ) -> Result<Operand, PassError> {
        let Operand::Value(start) = operand else {
            return Ok(operand);
        };
        if !is_identity(function, start) {
            return Ok(operand);
        }
        // Walk to the end, or to a chain already resolved.
        let mut chain = std::mem::take(&mut self.chain);
        chain.clear();
        let mut current = operand;
        let mut steps = 0_usize;
        let end = loop {
            work.step();
            let Operand::Value(value) = current else {
                break current;
            };
            if !is_identity(function, value) {
                break current;
            }
            steps = steps.saturating_add(1);
            if steps > self.ends.len() {
                return Err(PassError::Identity(start));
            }
            // A remembered end may itself have become an `Identity` since (constant
            // propagation folds as it walks), so the walk goes on from it.
            if let Some(end) = self.end(value) {
                current = end;
                continue;
            }
            chain.push(value);
            current = match function.inst(value) {
                Some(Inst::Op(op)) => *op.args.first().ok_or(PassError::Identity(start))?,
                _ => return Err(PassError::Identity(start)),
            };
            if let Operand::Value(next) = current
                && function.inst(next).is_none()
            {
                return Err(PassError::Identity(start));
            }
        };
        for &value in &chain {
            if let Some(slot) = usize::try_from(value.index())
                .ok()
                .and_then(|index| self.ends.get_mut(index))
            {
                *slot = Some(end);
            }
        }
        self.chain = chain;
        Ok(end)
    }

    fn end(&self, value: Value) -> Option<Operand> {
        usize::try_from(value.index())
            .ok()
            .and_then(|index| self.ends.get(index))
            .copied()
            .flatten()
    }
}

/// Whether `value` names an `Identity` operation.
fn is_identity(function: &Function, value: Value) -> bool {
    matches!(function.inst(value), Some(Inst::Op(op)) if op.opcode == Opcode::Identity)
}

/// Argument `index` of instruction `value` (a phi's incoming value), if it has one.
fn argument(function: &Function, value: Value, index: usize) -> Result<Option<Operand>, PassError> {
    Ok(match function.inst(value) {
        Some(Inst::Op(op)) => op.args.get(index).copied(),
        Some(Inst::Phi(phi)) => phi.incoming.get(index).map(|&(_, arg)| arg),
        None => return Err(PassError::Missing(value)),
    })
}

/// Sets argument `index` of instruction `value` (a phi's incoming value).
fn set_argument(function: &mut Function, value: Value, index: usize, arg: Operand) {
    let slot = match function.inst_mut(value) {
        Some(Inst::Op(op)) => op.args.get_mut(index),
        Some(Inst::Phi(phi)) => phi.incoming.get_mut(index).map(|(_, slot)| slot),
        None => None,
    };
    if let Some(slot) = slot {
        *slot = arg;
    }
}

/// Runs the pass (see the module docs).
pub(super) fn run(function: &mut Function) -> Result<Work, PassError> {
    let mut work = Work::default();
    let mut resolver = Resolver::new(function);
    let mut removed = Vec::new();
    let ids: Vec<_> = function.blocks().map(|(id, _)| id).collect();
    for id in ids {
        work.step();
        let list = function
            .block_mut(id)
            .map_or_default(|block| std::mem::take(&mut block.insts));
        let mut kept = Vec::with_capacity(list.len());
        for value in list {
            work.step();
            let mut index = 0;
            while let Some(arg) = argument(function, value, index)? {
                let end = resolver.resolve(function, arg, &mut work)?;
                set_argument(function, value, index, end);
                index += 1;
            }
            let remove = matches!(
                function.inst(value),
                Some(Inst::Op(op)) if matches!(op.opcode, Opcode::Identity | Opcode::Void)
            );
            if remove {
                removed.push(value);
            } else {
                kept.push(value);
            }
        }
        let terminator = function.block(id).map(|block| block.terminator);
        let terminator = match terminator {
            Some(Terminator::Branch {
                cond,
                then,
                otherwise,
            }) => Some(Terminator::Branch {
                cond: resolver.resolve(function, cond, &mut work)?,
                then,
                otherwise,
            }),
            other => other,
        };
        if let Some(block) = function.block_mut(id) {
            block.insts = kept;
            if let Some(terminator) = terminator {
                block.terminator = terminator;
            }
        }
    }
    if let Some(tree) = function.structure() {
        let mut tree = tree.to_vec();
        let mut chase = Work::default();
        tree_conditions_mut(&mut tree, &mut work, |cond| {
            *cond = resolver.resolve(function, *cond, &mut chase)?;
            Ok(())
        })?;
        work.absorb(chase);
        function.set_structure(Some(tree));
    }
    for value in removed {
        work.step();
        if let Some(inst) = function.inst_mut(value) {
            *inst = invalidated();
        }
    }
    Ok(work)
}
