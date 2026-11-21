//! The constant-buffer half of the pass: every constant-buffer read at a register offset is
//! bounded by the size of the host binding it reads (the parent module's "Constant-buffer
//! reads").

use super::Inserter;
use crate::info::{INDEXED_CBUF_SIZE, cbuf_element_size};
use crate::ir::{
    BlockId, BuildError, Function, Imm, Inst, Opcode, Operand, Operation, Type, Value,
};
use crate::opt::Work;

/// A read the rewrite bounds: its block and id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Read {
    pub(super) block: BlockId,
    pub(super) at: Value,
}

/// How many instructions bounding one read of result type `ty` adds: the comparison, the
/// clamped offset and the raw read, then one select, or for a pair two extracts, two selects
/// and the construction in the read's slot.
pub(super) const fn added(ty: Type) -> usize {
    match ty {
        Type::U32x2 => 3 + 4,
        _ => 3 + 1,
    }
}

/// Every constant-buffer read whose offset is a value, in the order the blocks list them, each
/// with its result type.
pub(super) fn reads(function: &Function, work: &mut Work) -> Vec<(Read, Type)> {
    let mut out = Vec::new();
    for (block, listed) in function.blocks() {
        for &at in &listed.insts {
            work.step();
            let Some(Inst::Op(op)) = function.inst(at) else {
                continue;
            };
            if cbuf_element_size(op.opcode).is_some()
                && op
                    .args
                    .get(1)
                    .is_some_and(|offset| offset.value().is_some())
            {
                out.push((Read { block, at }, op.opcode.result()));
            }
        }
    }
    out
}

/// The zero a read of result type `ty` gives past the bound.
const fn zero(ty: Type) -> Operand {
    match ty {
        Type::F32 => Operand::Imm(Imm::F32(0)),
        _ => Operand::Imm(Imm::U32(0)),
    }
}

/// Bounds `read`: the raw read moves to a new id that reads at the offset where it is below
/// [`INDEXED_CBUF_SIZE`] and at 0 elsewhere, and the read's own id becomes the raw value where
/// the offset is in bounds and zero elsewhere, so every use sees the bounded value unchanged.
pub(super) fn bound(
    function: &mut Function,
    inserter: &mut Inserter,
    read: Read,
) -> Result<(), BuildError> {
    let Some(Inst::Op(op)) = function.inst(read.at).cloned() else {
        return Ok(());
    };
    let (Some(&bank), Some(&offset)) = (op.args.first(), op.args.get(1)) else {
        return Ok(());
    };
    let ty = op.opcode.result();
    let mut add = |function: &mut Function, opcode: Opcode, args: Vec<Operand>| {
        inserter
            .push_before(
                function,
                read.block,
                read.at,
                Inst::Op(Operation::new(opcode, args)),
            )
            .map(Operand::Value)
    };
    let bound = Operand::Imm(Imm::U32(INDEXED_CBUF_SIZE));
    let inside = add(function, Opcode::ULessThan, vec![offset, bound])?;
    let clamped = add(
        function,
        Opcode::SelectU32,
        vec![inside, offset, zero(Type::U32)],
    )?;
    let raw = add(function, op.opcode, vec![bank, clamped])?;
    let replacement = match ty {
        Type::U32x2 => {
            let mut halves = [zero(Type::U32); 2];
            for (index, half) in (0_u32..).zip(halves.iter_mut()) {
                let element = add(
                    function,
                    Opcode::CompositeExtractU32x2,
                    vec![raw, Operand::Imm(Imm::U32(index))],
                )?;
                *half = add(
                    function,
                    Opcode::SelectU32,
                    vec![inside, element, zero(Type::U32)],
                )?;
            }
            Operation::new(Opcode::CompositeConstructU32x2, halves.to_vec())
        }
        Type::F32 => Operation::new(Opcode::SelectF32, vec![inside, raw, zero(Type::F32)]),
        _ => Operation::new(Opcode::SelectU32, vec![inside, raw, zero(Type::U32)]),
    };
    if let Some(slot) = function.inst_mut(read.at) {
        *slot = Inst::Op(replacement);
    }
    Ok(())
}
