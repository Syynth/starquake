//! Dead code elimination: an instruction nothing uses and that has no side effect leaves its
//! block.
//!
//! Ported by behaviour from Eden's
//! `eden/src/shader_recompiler/ir_opt/dead_code_elimination_pass.cpp` @ cf654052
//! (GPL-2.0-or-later yuzu code), with Eden's side-effect list (`Inst::MayHaveSideEffects`,
//! `frontend/ir/microinstruction.cpp`) as [`may_have_side_effects`]. As there, blocks are
//! walked once in post order (the parent module's "Block order") and each block's
//! instructions from last to first, so an instruction whose only users come after it in that
//! walk is removed with them; an unused instruction is removed and its arguments lose a use.
//! As in Eden, one walk does not reach a fixed point: a dead value used by a phi in a block the
//! walk has already passed (a loop's header, whose back edge comes from later), and dead
//! cycles of phis, stay (`tests::dce::dead_loop_values_stay_as_in_eden`).
//!
//! Uses are counted over every instruction the blocks list, a phi's incoming values included,
//! plus the terminators' and the structured tree's conditions, which are never removed (Eden
//! keeps its conditions alive through `ConditionRef`, an operation with a side effect). The
//! set operations of guest state (`SetRegister` and the rest) are not side effects, as in
//! Eden: after the SSA rewrite nothing reads them.
//!
//! Where this pass departs from Eden's, each pinned by a test in `tests`:
//!
//! - Blocks the entry cannot reach, which Eden removes before its passes run, are walked
//!   first, in reverse arena order, so their dead instructions release their uses of
//!   reachable values before those are reached
//!   (`tests::dce::unreachable_users_are_removed_first`).
//! - The uniform register class's set operations (`SetUniformRegister`, `SetUniformPred`),
//!   which Eden does not have, are not side effects, like `SetRegister` and `SetPred`
//!   (`tests::dce::side_effects_match_eden`).

use crate::ir::{BlockId, Function, Inst, Opcode, Operand, Terminator};

use super::{PassError, Work, arena_len, block_order, invalidated, tree_conditions};

/// Whether an operation may have a side effect, so dead code elimination keeps it unused:
/// Eden's `Inst::MayHaveSideEffects` (`frontend/ir/microinstruction.cpp` @ cf654052), opcode
/// for opcode (the uniform register class's set operations, which Eden lacks, are not side
/// effects, as `SetRegister` is not).
pub const fn may_have_side_effects(opcode: Opcode) -> bool {
    use Opcode as O;
    matches!(
        opcode,
        O::ConditionRef
            | O::Reference
            | O::PhiMove
            | O::Prologue
            | O::Epilogue
            | O::Join
            | O::DemoteToHelperInvocation
            | O::Barrier
            | O::WorkgroupMemoryBarrier
            | O::DeviceMemoryBarrier
            | O::EmitVertex
            | O::EndPrimitive
            | O::SetAttribute
            | O::SetAttributeIndexed
            | O::SetPatch
            | O::SetFragColor
            | O::SetSampleMask
            | O::SetFragDepth
            | O::WriteGlobalU8
            | O::WriteGlobalS8
            | O::WriteGlobalU16
            | O::WriteGlobalS16
            | O::WriteGlobal32
            | O::WriteGlobal64
            | O::WriteGlobal128
            | O::WriteStorageU8
            | O::WriteStorageS8
            | O::WriteStorageU16
            | O::WriteStorageS16
            | O::WriteStorage32
            | O::WriteStorage64
            | O::WriteStorage128
            | O::WriteLocal
            | O::WriteSharedU8
            | O::WriteSharedU16
            | O::WriteSharedU32
            | O::WriteSharedU64
            | O::WriteSharedU128
            | O::SharedAtomicIAdd32
            | O::SharedAtomicSMin32
            | O::SharedAtomicUMin32
            | O::SharedAtomicSMax32
            | O::SharedAtomicUMax32
            | O::SharedAtomicInc32
            | O::SharedAtomicDec32
            | O::SharedAtomicAnd32
            | O::SharedAtomicOr32
            | O::SharedAtomicXor32
            | O::SharedAtomicExchange32
            | O::SharedAtomicExchange64
            | O::SharedAtomicExchange32x2
            | O::GlobalAtomicIAdd32
            | O::GlobalAtomicSMin32
            | O::GlobalAtomicUMin32
            | O::GlobalAtomicSMax32
            | O::GlobalAtomicUMax32
            | O::GlobalAtomicInc32
            | O::GlobalAtomicDec32
            | O::GlobalAtomicAnd32
            | O::GlobalAtomicOr32
            | O::GlobalAtomicXor32
            | O::GlobalAtomicExchange32
            | O::GlobalAtomicIAdd64
            | O::GlobalAtomicSMin64
            | O::GlobalAtomicUMin64
            | O::GlobalAtomicSMax64
            | O::GlobalAtomicUMax64
            | O::GlobalAtomicAnd64
            | O::GlobalAtomicOr64
            | O::GlobalAtomicXor64
            | O::GlobalAtomicExchange64
            | O::GlobalAtomicIAdd32x2
            | O::GlobalAtomicSMin32x2
            | O::GlobalAtomicUMin32x2
            | O::GlobalAtomicSMax32x2
            | O::GlobalAtomicUMax32x2
            | O::GlobalAtomicAnd32x2
            | O::GlobalAtomicOr32x2
            | O::GlobalAtomicXor32x2
            | O::GlobalAtomicExchange32x2
            | O::GlobalAtomicAddF32
            | O::GlobalAtomicAddF16x2
            | O::GlobalAtomicAddF32x2
            | O::GlobalAtomicMinF16x2
            | O::GlobalAtomicMinF32x2
            | O::GlobalAtomicMaxF16x2
            | O::GlobalAtomicMaxF32x2
            | O::StorageAtomicIAdd32
            | O::StorageAtomicSMin32
            | O::StorageAtomicUMin32
            | O::StorageAtomicSMax32
            | O::StorageAtomicUMax32
            | O::StorageAtomicInc32
            | O::StorageAtomicDec32
            | O::StorageAtomicAnd32
            | O::StorageAtomicOr32
            | O::StorageAtomicXor32
            | O::StorageAtomicExchange32
            | O::StorageAtomicIAdd64
            | O::StorageAtomicSMin64
            | O::StorageAtomicUMin64
            | O::StorageAtomicSMax64
            | O::StorageAtomicUMax64
            | O::StorageAtomicAnd64
            | O::StorageAtomicOr64
            | O::StorageAtomicXor64
            | O::StorageAtomicExchange64
            | O::StorageAtomicIAdd32x2
            | O::StorageAtomicSMin32x2
            | O::StorageAtomicUMin32x2
            | O::StorageAtomicSMax32x2
            | O::StorageAtomicUMax32x2
            | O::StorageAtomicAnd32x2
            | O::StorageAtomicOr32x2
            | O::StorageAtomicXor32x2
            | O::StorageAtomicExchange32x2
            | O::StorageAtomicAddF32
            | O::StorageAtomicAddF16x2
            | O::StorageAtomicAddF32x2
            | O::StorageAtomicMinF16x2
            | O::StorageAtomicMinF32x2
            | O::StorageAtomicMaxF16x2
            | O::StorageAtomicMaxF32x2
            | O::BindlessImageWrite
            | O::BoundImageWrite
            | O::ImageWrite
            | O::BindlessImageAtomicIAdd32
            | O::BindlessImageAtomicSMin32
            | O::BindlessImageAtomicUMin32
            | O::BindlessImageAtomicSMax32
            | O::BindlessImageAtomicUMax32
            | O::BindlessImageAtomicInc32
            | O::BindlessImageAtomicDec32
            | O::BindlessImageAtomicAnd32
            | O::BindlessImageAtomicOr32
            | O::BindlessImageAtomicXor32
            | O::BindlessImageAtomicExchange32
            | O::BoundImageAtomicIAdd32
            | O::BoundImageAtomicSMin32
            | O::BoundImageAtomicUMin32
            | O::BoundImageAtomicSMax32
            | O::BoundImageAtomicUMax32
            | O::BoundImageAtomicInc32
            | O::BoundImageAtomicDec32
            | O::BoundImageAtomicAnd32
            | O::BoundImageAtomicOr32
            | O::BoundImageAtomicXor32
            | O::BoundImageAtomicExchange32
            | O::ImageAtomicIAdd32
            | O::ImageAtomicSMin32
            | O::ImageAtomicUMin32
            | O::ImageAtomicSMax32
            | O::ImageAtomicUMax32
            | O::ImageAtomicInc32
            | O::ImageAtomicDec32
            | O::ImageAtomicAnd32
            | O::ImageAtomicOr32
            | O::ImageAtomicXor32
            | O::ImageAtomicExchange32
    )
}

/// One more use of `operand`, if it is a value.
fn count(uses: &mut [u32], operand: Operand) {
    if let Some(slot) = operand
        .value()
        .and_then(|value| usize::try_from(value.index()).ok())
        .and_then(|index| uses.get_mut(index))
    {
        *slot = slot.saturating_add(1);
    }
}

/// One use fewer of `operand`, if it is a value.
fn release(uses: &mut [u32], operand: Operand) {
    if let Some(slot) = operand
        .value()
        .and_then(|value| usize::try_from(value.index()).ok())
        .and_then(|index| uses.get_mut(index))
    {
        *slot = slot.saturating_sub(1);
    }
}

/// The value operands of `inst`, a phi's incoming values included.
fn operands(inst: &Inst) -> impl Iterator<Item = Operand> + '_ {
    let (args, incoming) = match inst {
        Inst::Op(op) => (op.args.as_slice(), &[][..]),
        Inst::Phi(phi) => (&[][..], phi.incoming.as_slice()),
    };
    args.iter()
        .copied()
        .chain(incoming.iter().map(|&(_, operand)| operand))
}

/// Runs the pass (see the module docs).
pub(super) fn run(function: &mut Function) -> Result<Work, PassError> {
    let mut work = Work::default();
    let mut uses = vec![0_u32; arena_len(function)];
    for (_, inst) in function.listed() {
        work.step();
        for operand in operands(inst) {
            count(&mut uses, operand);
        }
    }
    for (_, block) in function.blocks() {
        work.step();
        if let Terminator::Branch { cond, .. } = block.terminator {
            count(&mut uses, cond);
        }
    }
    if let Some(tree) = function.structure() {
        tree_conditions(tree, &mut work, |cond| count(&mut uses, cond));
    }
    // The visiting order is reverse post order then the unreachable blocks in arena order, so
    // its reverse is the unreachable blocks in reverse arena order, then post order.
    let mut order = block_order(function, &mut work);
    order.reverse();
    for block in order {
        sweep(function, block, &mut uses, &mut work)?;
    }
    Ok(work)
}

/// Removes `block`'s dead instructions, last to first.
fn sweep(
    function: &mut Function,
    block: BlockId,
    uses: &mut [u32],
    work: &mut Work,
) -> Result<(), PassError> {
    let list = function
        .block_mut(block)
        .map(|b| std::mem::take(&mut b.insts))
        .unwrap_or_default();
    let mut kept = Vec::with_capacity(list.len());
    for &value in list.iter().rev() {
        work.step();
        let inst = function.inst(value).ok_or(PassError::Missing(value))?;
        let used = usize::try_from(value.index())
            .ok()
            .and_then(|index| uses.get(index))
            .is_some_and(|&n| n > 0);
        let effect = matches!(inst, Inst::Op(op) if may_have_side_effects(op.opcode));
        if used || effect {
            kept.push(value);
            continue;
        }
        for operand in operands(inst) {
            release(uses, operand);
        }
        if let Some(slot) = function.inst_mut(value) {
            *slot = invalidated();
        }
    }
    kept.reverse();
    if let Some(b) = function.block_mut(block) {
        b.insts = kept;
    }
    Ok(())
}
