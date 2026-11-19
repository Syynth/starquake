//! The IR's basic blocks from the flow graph (the owner's decision of 2026-11-18: slice 10
//! builds the blocks, slice 11 the structured IR the translators fill).
//!
//! Every flow block becomes one IR block, in the same order, so the entry is block 0. The
//! translators fill it later from its instructions ([`IrBlocks::sources`]). A test reads the
//! state the block's instructions leave, so it cannot go in front of them: a block with
//! instructions that ends in a test jumps to a test block of its own, where the test runs; a
//! block with no instructions holds its test itself. The tests:
//!
//! - **A branch** reads its predicate (`GetPred`, then `LogicalNot` when negated) and ends in
//!   [`Terminator::Branch`]. A test of `PT` is the immediate `true` and of `!PT` the immediate
//!   `false` (the IR refuses to read `PT`: `ir`'s `ReservedName`).
//! - **An indirect branch** reads its register (`GetRegister`, or `GetUniformRegister` for
//!   `BRXU`; `RZ` reads as the immediate 0, as PR #4588's `Visitor::X` reads it), then compares
//!   it with each target's address in turn (`IEqual`), each comparison in its own block, as
//!   Eden's structurizer chains `IndirectBranchCond` gotos (`structured_control_flow.cpp` @
//!   cf654052); when none matches, control never gets there ([`Terminator::Unreachable`], Eden's
//!   `Unreachable` after the gotos).
//! - **An exit** is [`Terminator::Return`].

use super::{BlockEnd, BlockIndex, FlowError, FlowGraph, IndirectTarget};
use crate::decode::{BrxSrc, PredSrc};
use crate::ir::{BlockId, Builder, Function, Imm, Opcode, Operand, Terminator};

/// Which flow block an IR block holds the instructions of.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockSource {
    /// The IR block.
    pub block: BlockId,
    /// The flow block, whose `code` and `guard_tested` the translators read.
    pub flow: BlockIndex,
}

/// The IR's blocks for a flow graph: a control-flow graph whose blocks the translators fill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrBlocks {
    /// The function: every block terminated, the test blocks holding their tests.
    pub function: Function,
    /// For each flow block in order, the IR block that holds its instructions: block `n` for
    /// flow block `n`. Test blocks hold no instructions of the program and are not listed.
    pub sources: Vec<BlockSource>,
}

impl FlowGraph {
    /// Builds the IR's blocks (see the module docs).
    pub fn build_ir(&self) -> Result<IrBlocks, FlowError> {
        let mut function = Function::new();
        let mut sources = Vec::with_capacity(self.blocks.len());
        // The entry exists already; one more block per flow block after it.
        for (index, _) in (0_u32..).zip(&self.blocks) {
            let block = if index == 0 {
                function.entry()
            } else {
                function.add_block()?
            };
            sources.push(BlockSource {
                block,
                flow: BlockIndex(index),
            });
        }
        let ir_block = |target: BlockIndex| {
            usize::try_from(target.0)
                .ok()
                .and_then(|slot| sources.get(slot))
                .map(|source| source.block)
                .ok_or(FlowError::Unwalked {
                    index: target.0,
                    target: target.0,
                })
        };
        for (source, flow) in sources.iter().zip(&self.blocks) {
            let mut builder = Builder::new(&mut function, source.block);
            match &flow.end {
                BlockEnd::Jump(target) => {
                    builder.terminate(Terminator::Jump(ir_block(*target)?))?;
                }
                BlockEnd::Exit => builder.terminate(Terminator::Return)?,
                BlockEnd::Branch {
                    cond,
                    then,
                    otherwise,
                } => {
                    to_test_block(&mut builder, flow.code.is_empty())?;
                    let cond = read_pred(&mut builder, *cond)?;
                    builder.terminate(Terminator::Branch {
                        cond,
                        then: ir_block(*then)?,
                        otherwise: ir_block(*otherwise)?,
                    })?;
                }
                BlockEnd::Indirect { src, targets } => {
                    to_test_block(&mut builder, flow.code.is_empty())?;
                    let value = read_reg(&mut builder, *src)?;
                    for &IndirectTarget { address, block } in targets {
                        let matches = builder
                            .op(Opcode::IEqual, &[value, Operand::Imm(Imm::U32(address))])?;
                        let otherwise = builder.add_block()?;
                        builder.terminate(Terminator::Branch {
                            cond: Operand::Value(matches),
                            then: ir_block(block)?,
                            otherwise,
                        })?;
                        builder.switch_to(otherwise)?;
                    }
                    builder.terminate(Terminator::Unreachable)?;
                }
            }
        }
        Ok(IrBlocks { function, sources })
    }
}

/// Moves `builder` to a new test block after its block, unless the block holds no
/// instructions and can hold the test itself.
fn to_test_block(builder: &mut Builder<'_>, empty: bool) -> Result<(), FlowError> {
    if empty {
        return Ok(());
    }
    let test = builder.add_block()?;
    builder.terminate(Terminator::Jump(test))?;
    builder.switch_to(test)?;
    Ok(())
}

/// The value of predicate `cond`.
fn read_pred(builder: &mut Builder<'_>, cond: PredSrc) -> Result<Operand, FlowError> {
    if cond.reg.is_true() {
        return Ok(Operand::Imm(Imm::U1(!cond.negated)));
    }
    let value = builder.op(Opcode::GetPred, &[Operand::Imm(Imm::Pred(cond.reg))])?;
    if !cond.negated {
        return Ok(Operand::Value(value));
    }
    Ok(Operand::Value(
        builder.op(Opcode::LogicalNot, &[Operand::Value(value)])?,
    ))
}

/// The value of the register an indirect branch reads.
fn read_reg(builder: &mut Builder<'_>, src: BrxSrc) -> Result<Operand, FlowError> {
    let value = match src {
        BrxSrc::Reg(reg) if reg.is_zero() => return Ok(Operand::Imm(Imm::U32(0))),
        BrxSrc::Reg(reg) => builder.op(Opcode::GetRegister, &[Operand::Imm(Imm::Reg(reg))])?,
        BrxSrc::UniformReg(reg) => builder.op(
            Opcode::GetUniformRegister,
            &[Operand::Imm(Imm::UniformReg(reg))],
        )?,
    };
    Ok(Operand::Value(value))
}
