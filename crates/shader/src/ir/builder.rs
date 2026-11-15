//! The builder: appends checked instructions to a block (the role of Eden's `IREmitter`,
//! `eden/src/shader_recompiler/frontend/ir/ir_emitter.h` @ cf654052).
//!
//! Eden's emitter has one typed C++ method per operation; here one [`Builder::op`] takes any
//! [`Opcode`] and checks the instruction against the opcode table as it appends it: the
//! argument count, the flags' kind and ranges, that every value argument names an instruction
//! with a result of the type its slot needs, and that no reserved register or predicate is
//! accessed (the verifier's steps 4b to 4e, shared code). Whether each value is defined where it
//! is used depends on the finished control flow, so that is left to [`Function::verify`].
//! Nothing is appended when a check fails.

use thiserror::Error;

use super::verify::{check_operation, check_phi, check_terminator};
use super::{
    BlockId, Flags, Function, Inst, Opcode, Operand, Operation, Phi, Terminator, Type, Value,
    VerifyError,
};

/// Why an instruction, block or terminator was not added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BuildError {
    /// An arena has no room for another id.
    #[error("the function has more than {max} instructions or blocks", max = u32::MAX)]
    Full,
    /// The block does not exist.
    #[error("{0} does not exist")]
    NoBlock(BlockId),
    /// The instruction or terminator is malformed.
    #[error(transparent)]
    Invalid(#[from] VerifyError),
}

/// Appends instructions to one block of a function at a time.
#[derive(Debug)]
pub struct Builder<'f> {
    function: &'f mut Function,
    block: BlockId,
}

impl<'f> Builder<'f> {
    /// A builder that appends to `block` of `function`.
    pub fn new(function: &'f mut Function, block: BlockId) -> Self {
        Self { function, block }
    }

    /// The block instructions are appended to.
    pub const fn block(&self) -> BlockId {
        self.block
    }

    /// Appends to `block` from now on.
    pub fn switch_to(&mut self, block: BlockId) -> Result<(), BuildError> {
        if self.function.block(block).is_none() {
            return Err(BuildError::NoBlock(block));
        }
        self.block = block;
        Ok(())
    }

    /// The function being built.
    pub fn function(&self) -> &Function {
        self.function
    }

    /// Adds an empty block (it does not become the current one).
    pub fn add_block(&mut self) -> Result<BlockId, BuildError> {
        self.function.add_block()
    }

    /// The id the next instruction will get, if the arena has room.
    fn next_id(&self) -> Result<Value, BuildError> {
        u32::try_from(self.function.insts.len())
            .map(Value)
            .map_err(|_| BuildError::Full)
    }

    fn append(&mut self, inst: Inst) -> Result<Value, BuildError> {
        if self.function.block(self.block).is_none() {
            return Err(BuildError::NoBlock(self.block));
        }
        self.function.push(self.block, inst)
    }

    /// Appends `opcode` with `args` and the default flags of its kind.
    pub fn op(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Value, BuildError> {
        self.op_with(opcode, args, Flags::default_of(opcode.flags_kind()))
    }

    /// Appends `opcode` with `args` and `flags`.
    pub fn op_with(
        &mut self,
        opcode: Opcode,
        args: &[Operand],
        flags: Flags,
    ) -> Result<Value, BuildError> {
        let at = self.next_id()?;
        let op = Operation {
            opcode,
            args: args.to_vec(),
            flags,
        };
        check_operation(self.function, at, &op)?;
        self.append(Inst::Op(op))
    }

    /// Appends a phi of type `ty`. Phis go before the block's operations, which the verifier
    /// checks.
    pub fn phi(
        &mut self,
        ty: Type,
        incoming: Vec<(BlockId, Operand)>,
    ) -> Result<Value, BuildError> {
        let at = self.next_id()?;
        let phi = Phi { ty, incoming };
        check_phi(self.function, at, &phi)?;
        self.append(Inst::Phi(phi))
    }

    /// Ends the current block with `terminator`, replacing any it had.
    pub fn terminate(&mut self, terminator: Terminator) -> Result<(), BuildError> {
        check_terminator(self.function, self.block, terminator)?;
        let block = self
            .function
            .block_mut(self.block)
            .ok_or(BuildError::NoBlock(self.block))?;
        block.terminator = terminator;
        Ok(())
    }
}
