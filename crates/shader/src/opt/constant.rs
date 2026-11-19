//! Constant propagation: operations are folded and simplified, one walk over the blocks in
//! reverse post order (the parent module's "Block order"), each block's instructions in order.
//!
//! Ported by behaviour from Eden's
//! `eden/src/shader_recompiler/ir_opt/constant_propagation_pass.cpp` @ cf654052
//! (GPL-3.0-or-later, over GPL-2.0-or-later yuzu code). A fold turns the instruction into an
//! `Identity` of its replacement (Eden's `ReplaceUsesWith`), rewrites its arguments in place,
//! or puts new operations in front of it, which the walk does not visit (as Eden's list walk
//! does not). Arguments are looked through `Identity` chains as Eden's `Resolve` and
//! `InstRecursive` do. The rules, in Eden's dispatch order (`ConstantPropagation`):
//!
//! - `GetRegister RZ` is `0` and `GetPred PT` is `true` (`FoldGetRegister`, `FoldGetPred`);
//!   here also `GetUniformRegister URZ` is `0` and `GetUniformPred UPT` is `true`. The verifier
//!   refuses these reads (`ir`'s `ReservedName`), so only an unverified function has them.
//! - `IAdd32` and `IAdd64` (`FoldAdd`): unless the instruction has a pseudo-operation
//!   (`GetZeroFromOp` and its siblings name it), two immediates fold; an immediate on the
//!   left moves to the right, combining with an immediate on the right of a left operand of
//!   the same opcode (`FoldCommutative`); `x + 0` is `x`; and, for `IAdd32`, the two `XMAD`
//!   patterns (`FoldXmadMultiply`, then `FoldXmadMultiplyAdd`) become `IMul32` (and `IAdd32`).
//! - `ISub32` (`FoldISub32`): two immediates fold; two reads of the same constant buffer word
//!   subtract to `0`; `(x + c) - c` is `x` when both `c` read the same constant buffer word.
//! - `IMul32`, `ShiftRightArithmetic32`, the integer comparisons, `BitwiseAnd32`,
//!   `BitwiseOr32`, `BitwiseXor32`, `BitFieldUExtract`, `BitFieldSExtract`, `BitFieldInsert`
//!   and `FPNeg32` fold when every argument is an immediate and there is no pseudo-operation
//!   (`FoldWhenAllImmediates`).
//! - `BitCastF32U32` and `BitCastU32F32` (`FoldBitCast`): an immediate folds; a cast of the
//!   reverse cast is its argument; `BitCastF32U32` of `GetCbufU32` becomes `GetCbufF32`; and
//!   `BitCastU32F32` of `GetAttribute` of the primitive, instance, vertex, base vertex, base
//!   instance or draw id becomes `GetAttributeU32` (Eden's workaround for an Nvidia driver
//!   bug).
//! - `PackHalf2x16`, `UnpackHalf2x16`, `PackFloat2x16`, `UnpackFloat2x16` of their inverse are
//!   the inverse's argument (`FoldInverseFunc`).
//! - The selects with an immediate condition are the chosen argument (`FoldSelect`).
//! - `FPAdd32` (`FoldFPAdd32`): two immediates fold; `x + 0.0` and `x + -0.0` (either side)
//!   are `x`.
//! - `FPMul32` (`FoldFPMul32`): two immediates fold; then, unless the instruction forbids
//!   contraction and when neither argument is an immediate, `YDirection * DPdyFine(x)` (either
//!   order) is `DPdyFine(x)` (`FoldDerivativeYFromCorrection`), and `(a * A) * (1 / A)` is `a`
//!   when both `A` read the same attribute word (the interpolation fold).
//! - `LogicalAnd` and `LogicalOr`: `FoldCommutative`, then `x && true` is `x`, `x && false` is
//!   `false`, `x || true` is `true` and `x || false` is `x`. `LogicalNot` of an immediate
//!   folds and of a `LogicalNot` is its argument.
//! - The composite extracts with an immediate index read through a chain of inserts at other
//!   indices to the insert at that index, or to the construct (`FoldCompositeExtract`).
//! - `FSwizzleAdd` with swizzle `0x99` or `0xA5` of a quad butterfly (a `QuadSwap`, or a
//!   `ShuffleButterfly` with clamp 3 and segmentation mask 28) of its own second argument
//!   becomes `DPdxFine` or `DPdyFine` of that argument (`FoldFSwizzleAdd`).
//!
//! Folding two immediates goes through the interpreter's evaluation
//! (`interp::eval_immediates`), so a folded constant is what a run computes (the IR's
//! semantics, decision 15). Eden computes with host C++ arithmetic. Where that differs, each
//! pinned by a test in `tests`:
//!
//! - **Undefined folds are left alone.** Eden's `BitFieldUExtract` and `BitFieldSExtract` and
//!   `BitFieldInsert` folds throw a `LogicError` (translation fails) when the field leaves the
//!   word (`shift + count > 32`; for the signed extract also `count == 0`; for insert
//!   `count >= 32` or `offset >= 32`), and its `ShiftRightArithmetic32` fold shifts by a
//!   `s32`, which C++ leaves undefined from 32 up. Here those instructions are not folded
//!   (`tests::constant::undefined_folds_are_left_alone`).
//! - **`BitFieldInsert` masks the inserted value** to the field, as SPIR-V's
//!   `OpBitFieldInsert` and the interpreter do; Eden's fold ORs in `insert << offset`
//!   unmasked (`tests::constant::bit_field_insert_masks_the_inserted_value`).
//! - **Float folds honour the instruction's controls** (rounding, flush to zero); Eden folds
//!   with the host's round-to-nearest and no flushing (`tests::constant::float_folds_honour_controls`).
//! - **`(x + c) - c` only.** Eden's `FoldISub32` swaps its operands when the right one is the
//!   `IAdd32`, so it also folds `c - (x + c)` to `x`, which is `-x`; here only the left
//!   operand may be the `IAdd32` (`tests::constant::isub_folds_only_the_add_on_the_left`).
//!
//! Folds Eden makes that do not keep the interpreter's result, ported as Eden makes them
//! (the interpreter equivalence tests do not generate these inputs; each is pinned by a test
//! that shows the difference):
//!
//! - `x + 0.0` gives `x` where the interpreter gives `+0.0` for `x = -0.0`, and `x` unflushed
//!   where a flush-to-zero control flushes a denormal `x`
//!   (`tests::constant::float_add_of_zero_is_not_exact`).
//! - The interpolation fold `(a * A) * (1 / A) = a` is not exact in floating point
//!   (`tests::constant::interpolation_fold_is_not_exact`).
//! - `UnpackHalf2x16(PackHalf2x16(v)) = v` drops the rounding to half precision
//!   (`tests::constant::half_round_trip_fold_is_not_exact`).
//!
//! Not ported here, with where they belong:
//!
//! - Eden's `GetCbufU32` and `GetCbufF32` folds (`FoldConstBuffer`, `FoldDriverConstBuffer`)
//!   read the translation environment (whether the HLE macro state replaces a word with the
//!   base vertex, base instance or draw id; the proprietary driver's bank 1). Here the
//!   environment is the question-and-answer exchange (slice 9), which this pass does not
//!   take; they belong with the constant-buffer pass (slice 13).
//! - Eden's `FoldImageSampleImplicitLod` (an `.NDV` sample of a 3D texture whose coordinates
//!   are built from quad shuffles becomes an `ImageGradient`) is a texture workaround left for
//!   the texture pass (slice 13).

use crate::interp::{Val, eval_immediates};
use crate::ir::{
    Attribute, BlockId, Flags, Function, Imm, Inst, Opcode, Operand, Operation, Value,
};

use super::identity::Resolver;
use super::{PassError, Work, block_order, identity_of};

/// The attribute words of the `BitCastU32F32` workaround, Eden's numbers
/// (`eden/src/shader_recompiler/frontend/ir/attribute.h` @ cf654052): `PrimitiveId` 24,
/// `InstanceId` 190, `VertexId` 191, `BaseInstance` 256, `BaseVertex` 257, `DrawID` 258.
pub(super) const INTEGER_ATTRIBUTES: [Attribute; 6] = [
    Attribute(24),
    Attribute(190),
    Attribute(191),
    Attribute(256),
    Attribute(257),
    Attribute(258),
];

/// The swizzle of `FSwizzleAdd` that a horizontal derivative uses (Eden's `0x99`).
pub(super) const SWIZZLE_DX: u32 = 0x99;
/// The swizzle of `FSwizzleAdd` that a vertical derivative uses (Eden's `0xA5`).
pub(super) const SWIZZLE_DY: u32 = 0xA5;
/// A `ShuffleButterfly`'s clamp in a quad butterfly (Eden's `QuadButterflyMask`: 3).
pub(super) const QUAD_CLAMP: u32 = 3;
/// A `ShuffleButterfly`'s segmentation mask in a quad butterfly (Eden's
/// `QuadButterflyMask`: 28).
pub(super) const QUAD_SEGMENTATION: u32 = 28;
/// The bit width the bit-field folds are defined within (Eden's checks against 32).
const WORD_BITS: u32 = 32;

/// The pass's state.
struct Fold<'f> {
    function: &'f mut Function,
    resolver: Resolver,
    /// Whether a pseudo-operation (`GetZeroFromOp` and its siblings) names each instruction
    /// (Eden's `HasAssociatedPseudoOperation`), by arena index.
    pseudo: Vec<bool>,
    /// The block being walked.
    block: BlockId,
    /// Operations made in front of the current instruction.
    made: Vec<Value>,
    work: Work,
}

/// Runs the pass (see the module docs).
pub(super) fn run(function: &mut Function) -> Result<Work, PassError> {
    let mut work = Work::default();
    let order = block_order(function, &mut work);
    let mut pseudo = vec![false; super::arena_len(function)];
    for (_, inst) in function.listed() {
        work.step();
        if let Inst::Op(op) = inst
            && is_pseudo(op.opcode)
            && let Some(Operand::Value(parent)) = op.args.first()
            && let Some(mark) = usize::try_from(parent.index())
                .ok()
                .and_then(|index| pseudo.get_mut(index))
        {
            *mark = true;
        }
    }
    let resolver = Resolver::new(function);
    let entry = function.entry();
    let mut fold = Fold {
        function,
        resolver,
        pseudo,
        block: entry,
        made: Vec::new(),
        work,
    };
    for block in order {
        fold.block = block;
        let list = fold
            .function
            .block_mut(block)
            .map(|b| std::mem::take(&mut b.insts))
            .unwrap_or_default();
        let mut out = Vec::with_capacity(list.len());
        for value in list {
            fold.work.step();
            fold.made.clear();
            fold.visit(value)?;
            out.extend_from_slice(&fold.made);
            out.push(value);
        }
        if let Some(b) = fold.function.block_mut(block) {
            b.insts = out;
        }
    }
    Ok(fold.work)
}

/// Whether `opcode` is a pseudo-operation (Eden's `Inst::IsPseudoInstruction`).
const fn is_pseudo(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::GetZeroFromOp
            | Opcode::GetSignFromOp
            | Opcode::GetCarryFromOp
            | Opcode::GetOverflowFromOp
            | Opcode::GetSparseFromOp
            | Opcode::GetInBoundsFromOp
    )
}

/// The immediate of a value the interpreter computed; `None` for a type no immediate has.
const fn imm_of(value: Val) -> Option<Imm> {
    Some(match value {
        Val::U1(bit) => Imm::U1(bit),
        Val::U8(bits) => Imm::U8(bits),
        Val::U16(bits) => Imm::U16(bits),
        Val::U32(bits) => Imm::U32(bits),
        Val::U64(bits) => Imm::U64(bits),
        Val::F32(bits) => Imm::F32(bits),
        Val::F64(bits) => Imm::F64(bits),
        _ => return None,
    })
}

/// Whether `operand` is the immediate zero of a 32- or 64-bit integer (Eden's
/// `Arg<T>(rhs) == 0`).
const fn is_int_zero(operand: Operand) -> bool {
    matches!(operand, Operand::Imm(Imm::U32(0) | Imm::U64(0)))
}

/// Whether `operand` is an immediate `0.0` or `-0.0` (Eden's `std::abs(F32()) == 0.0f`): every
/// bit but the sign clear.
const fn is_float_zero(operand: Operand) -> bool {
    matches!(operand, Operand::Imm(Imm::F32(bits)) if bits & !F32_SIGN == 0)
}

/// The sign bit of a 32-bit float.
const F32_SIGN: u32 = 1 << 31;

/// An instruction's opcode, arguments and flags, copied out.
#[derive(Clone)]
struct Seen {
    opcode: Opcode,
    args: [Option<Operand>; 4],
    flags: Flags,
}

impl Seen {
    fn of(op: &Operation) -> Self {
        let mut args = [None; 4];
        for (slot, &arg) in args.iter_mut().zip(&op.args) {
            *slot = Some(arg);
        }
        Self {
            opcode: op.opcode,
            args,
            flags: op.flags,
        }
    }

    fn arg(&self, index: usize) -> Option<Operand> {
        self.args.get(index).copied().flatten()
    }
}

impl Fold<'_> {
    /// `operand` resolved through `Identity` chains (Eden's `Resolve`).
    fn resolve(&mut self, operand: Operand) -> Result<Operand, PassError> {
        self.resolver.resolve(self.function, operand, &mut self.work)
    }

    /// The operation at `value` with each argument that resolves to an immediate replaced by
    /// it (Eden's `IsImmediate` and immediate reads look through `Identity` chains); `None`
    /// for a phi or a missing id.
    fn look(&mut self, value: Value) -> Result<Option<Seen>, PassError> {
        let mut seen = match self.function.inst(value) {
            Some(Inst::Op(op)) => Seen::of(op),
            _ => return Ok(None),
        };
        for slot in &mut seen.args {
            if let Some(arg @ Operand::Value(_)) = *slot
                && let resolved @ Operand::Imm(_) = self.resolve(arg)?
            {
                *slot = Some(resolved);
            }
        }
        Ok(Some(seen))
    }

    /// The operation `operand` names after resolving (Eden's `InstRecursive`, which also
    /// returns a phi; a phi matches no opcode a fold looks for, so it is `None` here).
    fn producer(&mut self, operand: Operand) -> Result<Option<(Value, Seen)>, PassError> {
        let Operand::Value(value) = self.resolve(operand)? else {
            return Ok(None);
        };
        Ok(self.look(value)?.map(|seen| (value, seen)))
    }

    /// The operation `operand` names, if it has `opcode`.
    fn producer_of(
        &mut self,
        operand: Option<Operand>,
        opcode: Opcode,
    ) -> Result<Option<Seen>, PassError> {
        let Some(operand) = operand else {
            return Ok(None);
        };
        Ok(self
            .producer(operand)?
            .map(|(_, seen)| seen)
            .filter(|seen| seen.opcode == opcode))
    }

    fn has_pseudo(&self, value: Value) -> bool {
        usize::try_from(value.index())
            .ok()
            .and_then(|index| self.pseudo.get(index))
            .copied()
            .unwrap_or(false)
    }

    /// Makes `value` an `Identity` of `operand` (Eden's `ReplaceUsesWith`).
    fn replace(&mut self, value: Value, operand: Operand) {
        if let Some(inst) = self.function.inst_mut(value) {
            *inst = identity_of(operand);
        }
    }

    /// Sets argument `index` of `value`.
    fn set_arg(&mut self, value: Value, index: usize, operand: Operand) {
        if let Some(Inst::Op(op)) = self.function.inst_mut(value)
            && let Some(slot) = op.args.get_mut(index)
        {
            *slot = operand;
        }
    }

    /// Makes a new operation in front of the current instruction.
    fn make(&mut self, opcode: Opcode, args: Vec<Operand>) -> Result<Value, PassError> {
        let value = self
            .function
            .push(self.block, Inst::Op(Operation::new(opcode, args)))?;
        self.made.push(value);
        Ok(value)
    }

    /// The value of `opcode` with `flags` on immediates `args`, as the interpreter computes it.
    fn eval(at: Value, opcode: Opcode, flags: Flags, args: &[Operand]) -> Option<Operand> {
        let op = Operation {
            opcode,
            args: args.to_vec(),
            flags,
        };
        eval_immediates(at, &op).and_then(imm_of).map(Operand::Imm)
    }

    /// Visits instruction `value` (Eden's `ConstantPropagation`).
    fn visit(&mut self, value: Value) -> Result<(), PassError> {
        let seen = match self.look(value)? {
            Some(seen) => seen,
            None if self.function.inst(value).is_some() => return Ok(()),
            None => return Err(PassError::Missing(value)),
        };
        use Opcode as O;
        match seen.opcode {
            O::GetRegister => self.reserved(value, &seen),
            O::GetPred => self.reserved(value, &seen),
            O::GetUniformRegister => self.reserved(value, &seen),
            O::GetUniformPred => self.reserved(value, &seen),
            O::IAdd32 => self.add(value, &seen, true)?,
            O::ISub32 => self.isub(value, &seen)?,
            O::IMul32 | O::FPNeg32 => {
                self.all_immediates(value, &seen);
            }
            O::ShiftRightArithmetic32 => {
                // C++'s `a >> b` on an `s32` is defined for shifts 0 to 31.
                if matches!(seen.arg(1), Some(Operand::Imm(Imm::U32(shift))) if shift < WORD_BITS)
                {
                    self.all_immediates(value, &seen);
                }
            }
            O::BitCastF32U32 => self.bit_cast(value, &seen, O::BitCastU32F32)?,
            O::BitCastU32F32 => self.bit_cast(value, &seen, O::BitCastF32U32)?,
            O::IAdd64 => self.add(value, &seen, false)?,
            O::PackHalf2x16 => self.inverse(value, &seen, O::UnpackHalf2x16)?,
            O::UnpackHalf2x16 => self.inverse(value, &seen, O::PackHalf2x16)?,
            O::PackFloat2x16 => self.inverse(value, &seen, O::UnpackFloat2x16)?,
            O::UnpackFloat2x16 => self.inverse(value, &seen, O::PackFloat2x16)?,
            O::SelectU1
            | O::SelectU8
            | O::SelectU16
            | O::SelectU32
            | O::SelectU64
            | O::SelectF16
            | O::SelectF32
            | O::SelectF64 => self.select(value, &seen),
            O::FPAdd32 => self.fp_add(value, &seen),
            O::FPMul32 => self.fp_mul(value, &seen)?,
            O::LogicalAnd => self.logical(value, &seen, false)?,
            O::LogicalOr => self.logical(value, &seen, true)?,
            O::LogicalNot => self.logical_not(value, &seen)?,
            O::SLessThan
            | O::ULessThan
            | O::SLessThanEqual
            | O::ULessThanEqual
            | O::SGreaterThan
            | O::UGreaterThan
            | O::SGreaterThanEqual
            | O::UGreaterThanEqual
            | O::IEqual
            | O::INotEqual
            | O::BitwiseAnd32
            | O::BitwiseOr32
            | O::BitwiseXor32 => {
                self.all_immediates(value, &seen);
            }
            O::BitFieldUExtract | O::BitFieldSExtract | O::BitFieldInsert => {
                if bit_field_defined(&seen) {
                    self.all_immediates(value, &seen);
                }
            }
            O::CompositeExtractU32x2 => self.extract(value, &seen, O::CompositeConstructU32x2, O::CompositeInsertU32x2)?,
            O::CompositeExtractU32x3 => self.extract(value, &seen, O::CompositeConstructU32x3, O::CompositeInsertU32x3)?,
            O::CompositeExtractU32x4 => self.extract(value, &seen, O::CompositeConstructU32x4, O::CompositeInsertU32x4)?,
            O::CompositeExtractF32x2 => self.extract(value, &seen, O::CompositeConstructF32x2, O::CompositeInsertF32x2)?,
            O::CompositeExtractF32x3 => self.extract(value, &seen, O::CompositeConstructF32x3, O::CompositeInsertF32x3)?,
            O::CompositeExtractF32x4 => self.extract(value, &seen, O::CompositeConstructF32x4, O::CompositeInsertF32x4)?,
            O::CompositeExtractF16x2 => self.extract(value, &seen, O::CompositeConstructF16x2, O::CompositeInsertF16x2)?,
            O::CompositeExtractF16x3 => self.extract(value, &seen, O::CompositeConstructF16x3, O::CompositeInsertF16x3)?,
            O::CompositeExtractF16x4 => self.extract(value, &seen, O::CompositeConstructF16x4, O::CompositeInsertF16x4)?,
            O::FSwizzleAdd => self.swizzle_add(value, &seen)?,
            _ => {}
        }
        Ok(())
    }

    /// `FoldGetRegister`, `FoldGetPred` and their uniform counterparts.
    fn reserved(&mut self, value: Value, seen: &Seen) {
        let replacement = match seen.arg(0) {
            Some(Operand::Imm(Imm::Reg(reg))) if reg.is_zero() => Imm::U32(0),
            Some(Operand::Imm(Imm::UniformReg(reg))) if reg.is_zero() => Imm::U32(0),
            Some(Operand::Imm(Imm::Pred(pred))) if pred.is_true() => Imm::U1(true),
            Some(Operand::Imm(Imm::UniformPred(pred))) if pred.is_true() => Imm::U1(true),
            _ => return,
        };
        self.replace(value, Operand::Imm(replacement));
    }

    /// `FoldWhenAllImmediates`: whether it folded.
    fn all_immediates(&mut self, value: Value, seen: &Seen) -> bool {
        let count = seen.opcode.args().len();
        let mut args = Vec::with_capacity(count);
        for index in 0..count {
            match seen.arg(index) {
                Some(arg @ Operand::Imm(_)) => args.push(arg),
                _ => return false,
            }
        }
        if self.has_pseudo(value) {
            return false;
        }
        match Self::eval(value, seen.opcode, seen.flags, &args) {
            Some(result) => {
                self.replace(value, result);
                true
            }
            None => false,
        }
    }

    /// `FoldCommutative`: whether the caller goes on with its own rules.
    fn commutative(&mut self, value: Value, seen: &Seen) -> Result<bool, PassError> {
        let (Some(lhs), Some(rhs)) = (seen.arg(0), seen.arg(1)) else {
            return Ok(false);
        };
        let combine = |a: Operand, b: Operand| Self::eval(value, seen.opcode, seen.flags, &[a, b]);
        match (lhs, rhs) {
            (Operand::Imm(_), Operand::Imm(_)) => {
                if let Some(result) = combine(lhs, rhs) {
                    self.replace(value, result);
                }
                return Ok(false);
            }
            (Operand::Imm(_), Operand::Value(_)) => {
                let inner = self.producer_of(Some(rhs), seen.opcode)?;
                match inner.and_then(|inner| {
                    let first = inner.arg(0)?;
                    let imm = inner.arg(1).filter(|a| a.imm().is_some())?;
                    Some((first, combine(lhs, imm)?))
                }) {
                    Some((first, combined)) => {
                        self.set_arg(value, 0, first);
                        self.set_arg(value, 1, combined);
                    }
                    None => {
                        // Normalise: the immediate goes on the right.
                        self.set_arg(value, 0, rhs);
                        self.set_arg(value, 1, lhs);
                    }
                }
            }
            (Operand::Value(_), Operand::Imm(_)) => {
                let inner = self.producer_of(Some(lhs), seen.opcode)?;
                if let Some((first, combined)) = inner.and_then(|inner| {
                    let first = inner.arg(0)?;
                    let imm = inner.arg(1).filter(|a| a.imm().is_some())?;
                    Some((first, combine(rhs, imm)?))
                }) {
                    self.set_arg(value, 0, first);
                    self.set_arg(value, 1, combined);
                }
            }
            (Operand::Value(_), Operand::Value(_)) => {}
        }
        Ok(true)
    }

    /// The arguments of `value` now.
    fn args_now(&self, value: Value) -> Seen {
        match self.function.inst(value) {
            Some(Inst::Op(op)) => Seen::of(op),
            _ => Seen {
                opcode: Opcode::Void,
                args: [None; 4],
                flags: Flags::None,
            },
        }
    }

    /// `FoldAdd`: `word` is `IAdd32` (where the `XMAD` folds apply).
    fn add(&mut self, value: Value, seen: &Seen, word: bool) -> Result<(), PassError> {
        if self.has_pseudo(value) {
            return Ok(());
        }
        if !self.commutative(value, seen)? {
            return Ok(());
        }
        let now = self.args_now(value);
        if now.arg(1).is_some_and(is_int_zero)
            && let Some(lhs) = now.arg(0)
        {
            self.replace(value, lhs);
            return Ok(());
        }
        if word {
            if self.xmad_multiply(value, &now)? {
                return Ok(());
            }
            self.xmad_multiply_add(value, &now)?;
        }
        Ok(())
    }

    /// `FoldXmadMultiply`: `(bfe(a, 16, 16) * b) << 16 + bfe(a, 0, 16) * b` is `a * b`.
    fn xmad_multiply(&mut self, value: Value, now: &Seen) -> Result<bool, PassError> {
        let sixteen = Operand::Imm(Imm::U32(16));
        let zero = Operand::Imm(Imm::U32(0));
        let Some(lhs_shl) = self.producer_of(now.arg(0), Opcode::ShiftLeftLogical32)? else {
            return Ok(false);
        };
        let Some(rhs_mul) = self.producer_of(now.arg(1), Opcode::IMul32)? else {
            return Ok(false);
        };
        if lhs_shl.arg(1) != Some(sixteen) {
            return Ok(false);
        }
        let Some(lhs_mul) = self.producer_of(lhs_shl.arg(0), Opcode::IMul32)? else {
            return Ok(false);
        };
        let (Some(factor_b), Some(rhs_b)) = (lhs_mul.arg(1), rhs_mul.arg(1)) else {
            return Ok(false);
        };
        if self.resolve(factor_b)? != self.resolve(rhs_b)? {
            return Ok(false);
        }
        let Some(lhs_bfe) = self.producer_of(lhs_mul.arg(0), Opcode::BitFieldUExtract)? else {
            return Ok(false);
        };
        let Some(rhs_bfe) = self.producer_of(rhs_mul.arg(0), Opcode::BitFieldUExtract)? else {
            return Ok(false);
        };
        if lhs_bfe.arg(1) != Some(sixteen) || lhs_bfe.arg(2) != Some(sixteen) {
            return Ok(false);
        }
        if rhs_bfe.arg(1) != Some(zero) || rhs_bfe.arg(2) != Some(sixteen) {
            return Ok(false);
        }
        let (Some(factor_a), Some(rhs_a)) = (lhs_bfe.arg(0), rhs_bfe.arg(0)) else {
            return Ok(false);
        };
        if self.resolve(factor_a)? != self.resolve(rhs_a)? {
            return Ok(false);
        }
        let product = self.make(Opcode::IMul32, vec![factor_a, factor_b])?;
        self.replace(value, Operand::Value(product));
        Ok(true)
    }

    /// `FoldXmadMultiplyAdd`: the sixteen-operation pattern of an integer multiply-add becomes
    /// `IMul32 a, b` then `IAdd32` of it and `c` (see Eden's comment for the pattern).
    #[expect(
        clippy::too_many_lines,
        reason = "one check per link of Eden's sixteen-instruction pattern, in its order"
    )]
    fn xmad_multiply_add(&mut self, value: Value, now: &Seen) -> Result<bool, PassError> {
        use Opcode as O;
        let sixteen = Some(Operand::Imm(Imm::U32(16)));
        let zero = Some(Operand::Imm(Imm::U32(0)));
        let Some(op25) = self.producer_of(now.arg(0), O::ShiftLeftLogical32)? else {
            return Ok(false);
        };
        let Some(op27) = self.producer_of(now.arg(1), O::IAdd32)? else {
            return Ok(false);
        };
        if op25.arg(1) != sixteen {
            return Ok(false);
        }
        let Some(op24) = self.producer_of(op25.arg(0), O::IMul32)? else {
            return Ok(false);
        };
        let Some(op22) = self.producer_of(op24.arg(0), O::BitFieldUExtract)? else {
            return Ok(false);
        };
        let Some(op23) = self.producer_of(op24.arg(1), O::BitFieldUExtract)? else {
            return Ok(false);
        };
        if op22.arg(1) != sixteen || op22.arg(2) != sixteen {
            return Ok(false);
        }
        if op23.arg(1) != sixteen || op23.arg(2) != sixteen {
            return Ok(false);
        }
        let Some(arg11) = op23.arg(0) else {
            return Ok(false);
        };
        let Some((id11, op11)) = self.producer(arg11)? else {
            return Ok(false);
        };
        if op11.opcode != O::BitFieldInsert || op11.arg(2) != sixteen || op11.arg(3) != sixteen {
            return Ok(false);
        }
        let Some(op8) = self.producer_of(op11.arg(0), O::IMul32)? else {
            return Ok(false);
        };
        let Some(op10) = self.producer_of(op11.arg(1), O::BitFieldUExtract)? else {
            return Ok(false);
        };
        let Some(op6) = self.producer_of(op8.arg(0), O::BitFieldUExtract)? else {
            return Ok(false);
        };
        let Some(op7) = self.producer_of(op8.arg(1), O::BitFieldUExtract)? else {
            return Ok(false);
        };
        if op6.arg(1) != zero || op6.arg(2) != sixteen {
            return Ok(false);
        }
        if op7.arg(1) != sixteen || op7.arg(2) != sixteen {
            return Ok(false);
        }
        let Some(op26) = self.producer_of(op27.arg(0), O::ShiftLeftLogical32)? else {
            return Ok(false);
        };
        if op26.arg(1) != sixteen {
            return Ok(false);
        }
        match op26.arg(0) {
            Some(arg) if self.producer(arg)?.map(|(id, _)| id) == Some(id11) => {}
            _ => return Ok(false),
        }
        let Some(op18) = self.producer_of(op27.arg(1), O::IAdd32)? else {
            return Ok(false);
        };
        let Some(op17) = self.producer_of(op18.arg(0), O::IMul32)? else {
            return Ok(false);
        };
        let Some(op15) = self.producer_of(op17.arg(0), O::BitFieldUExtract)? else {
            return Ok(false);
        };
        let Some(op16) = self.producer_of(op17.arg(1), O::BitFieldUExtract)? else {
            return Ok(false);
        };
        if op15.arg(1) != zero || op16.arg(1) != zero || op10.arg(1) != zero {
            return Ok(false);
        }
        if op15.arg(2) != sixteen || op16.arg(2) != sixteen || op10.arg(2) != sixteen {
            return Ok(false);
        }
        let mut operands = [op7.arg(0), op16.arg(0), op10.arg(0), op22.arg(0), op6.arg(0), op15.arg(0)];
        for slot in &mut operands {
            *slot = match *slot {
                Some(operand) => Some(self.resolve(operand)?),
                None => return Ok(false),
            };
        }
        let [Some(a7), Some(a16), Some(a10), Some(b22), Some(b6), Some(b15)] = operands else {
            return Ok(false);
        };
        let Some(op_c) = op18.arg(1) else {
            return Ok(false);
        };
        if !self.all_equal(&[a7, a16, a10]) || !self.all_equal(&[b22, b6, b15]) {
            return Ok(false);
        }
        let product = self.make(O::IMul32, vec![a7, b6])?;
        let sum = self.make(O::IAdd32, vec![Operand::Value(product), op_c])?;
        self.replace(value, Operand::Value(sum));
        Ok(true)
    }

    /// The bank and offset `operand` reads, if it is a `GetCbufU32`.
    fn cbuf_word(&self, operand: Operand) -> Option<(Option<Operand>, Option<Operand>)> {
        match self.function.inst(operand.value()?) {
            Some(Inst::Op(op)) if op.opcode == Opcode::GetCbufU32 => {
                Some((op.args.first().copied(), op.args.get(1).copied()))
            }
            _ => None,
        }
    }

    /// Whether `operand` is a `GetCbufU32`.
    fn is_cbuf(&self, operand: Operand) -> bool {
        self.cbuf_word(operand).is_some()
    }

    /// Whether two operands read the same constant buffer word: both `GetCbufU32` with equal
    /// arguments (Eden's `equal_cbuf`).
    fn same_cbuf(&self, a: Operand, b: Operand) -> bool {
        matches!((self.cbuf_word(a), self.cbuf_word(b)), (Some(x), Some(y)) if x == y)
    }

    /// Eden's `AreEqual` on resolved operands: equal, or reads of the same constant buffer
    /// word.
    fn all_equal(&self, operands: &[Operand]) -> bool {
        operands
            .windows(2)
            .all(|pair| matches!(pair, [a, b] if a == b || self.same_cbuf(*a, *b)))
    }

    /// `FoldISub32`.
    fn isub(&mut self, value: Value, seen: &Seen) -> Result<(), PassError> {
        if self.all_immediates(value, seen) {
            return Ok(());
        }
        let (Some(lhs), Some(rhs)) = (seen.arg(0), seen.arg(1)) else {
            return Ok(());
        };
        if lhs.imm().is_some() || rhs.imm().is_some() {
            return Ok(());
        }
        let (op_a, op_b) = (self.resolve(lhs)?, self.resolve(rhs)?);
        if self.same_cbuf(op_a, op_b) {
            self.replace(value, Operand::Imm(Imm::U32(0)));
            return Ok(());
        }
        // `(x + c) - c`; Eden also swaps the operands here (see the module docs).
        let Some(add) = self.producer_of(Some(op_a), Opcode::IAdd32)? else {
            return Ok(());
        };
        if !self.is_cbuf(op_b) {
            return Ok(());
        }
        let (Some(mut add_a), Some(mut add_b)) = (add.arg(0), add.arg(1)) else {
            return Ok(());
        };
        if add_b.imm().is_some() {
            std::mem::swap(&mut add_a, &mut add_b);
        }
        if add_b.imm().is_some() {
            return Ok(());
        }
        let add_cbuf = self.resolve(add_b)?;
        if self.same_cbuf(add_cbuf, op_b) {
            self.replace(value, add_a);
        }
        Ok(())
    }

    /// `FoldSelect`.
    fn select(&mut self, value: Value, seen: &Seen) {
        let chosen = match seen.arg(0) {
            Some(Operand::Imm(Imm::U1(true))) => seen.arg(1),
            Some(Operand::Imm(Imm::U1(false))) => seen.arg(2),
            _ => None,
        };
        if let Some(chosen) = chosen {
            self.replace(value, chosen);
        }
    }

    /// `FoldFPAdd32`.
    fn fp_add(&mut self, value: Value, seen: &Seen) {
        if self.all_immediates(value, seen) {
            return;
        }
        let (Some(lhs), Some(rhs)) = (seen.arg(0), seen.arg(1)) else {
            return;
        };
        if is_float_zero(lhs) {
            self.replace(value, rhs);
        }
        if is_float_zero(rhs) {
            self.replace(value, lhs);
        }
    }

    /// `FoldFPMul32`, with `FoldDerivativeYFromCorrection`.
    fn fp_mul(&mut self, value: Value, seen: &Seen) -> Result<(), PassError> {
        if self.all_immediates(value, seen) {
            return Ok(());
        }
        if matches!(seen.flags, Flags::Fp(control) if control.no_contraction) {
            return Ok(());
        }
        let (Some(lhs), Some(rhs)) = (seen.arg(0), seen.arg(1)) else {
            return Ok(());
        };
        if lhs.imm().is_some() || rhs.imm().is_some() {
            return Ok(());
        }
        let lhs_op = self.producer(lhs)?.map(|(_, seen)| seen);
        let rhs_op = self.producer(rhs)?.map(|(_, seen)| seen);
        let opcode = |op: &Option<Seen>| op.as_ref().map(|seen| seen.opcode);
        // FoldDerivativeYFromCorrection.
        if opcode(&lhs_op) == Some(Opcode::YDirection) {
            if opcode(&rhs_op) == Some(Opcode::DPdyFine) {
                self.replace(value, rhs);
            }
            return Ok(());
        }
        if opcode(&rhs_op) == Some(Opcode::YDirection) {
            if opcode(&lhs_op) == Some(Opcode::DPdyFine) {
                self.replace(value, lhs);
            }
            return Ok(());
        }
        // The interpolation fold.
        let (Some(lhs_op), Some(rhs_op)) = (lhs_op, rhs_op) else {
            return Ok(());
        };
        if lhs_op.opcode != Opcode::FPMul32 || rhs_op.opcode != Opcode::FPRecip32 {
            return Ok(());
        }
        let (Some(recip_source), Some(mul_source)) = (rhs_op.arg(0), lhs_op.arg(1)) else {
            return Ok(());
        };
        let mul_source = self.resolve(mul_source)?;
        if recip_source.imm().is_some() || mul_source.imm().is_some() {
            return Ok(());
        }
        let attr_a = self.producer_of(Some(recip_source), Opcode::GetAttribute)?;
        let attr_b = self.producer_of(Some(mul_source), Opcode::GetAttribute)?;
        if let (Some(a), Some(b)) = (attr_a, attr_b)
            && a.arg(0) == b.arg(0)
            && let Some(replacement) = lhs_op.arg(0)
        {
            self.replace(value, replacement);
        }
        Ok(())
    }

    /// `FoldLogicalAnd` (`or` false) and `FoldLogicalOr` (`or` true).
    fn logical(&mut self, value: Value, seen: &Seen, or: bool) -> Result<(), PassError> {
        if !self.commutative(value, seen)? {
            return Ok(());
        }
        let now = self.args_now(value);
        if let (Some(lhs), Some(Operand::Imm(Imm::U1(rhs)))) = (now.arg(0), now.arg(1)) {
            // `x && true` and `x || false` are `x`; `x && false` and `x || true` are `rhs`.
            let replacement = if rhs == or { Operand::Imm(Imm::U1(rhs)) } else { lhs };
            self.replace(value, replacement);
        }
        Ok(())
    }

    /// `FoldLogicalNot`.
    fn logical_not(&mut self, value: Value, seen: &Seen) -> Result<(), PassError> {
        match seen.arg(0) {
            Some(Operand::Imm(Imm::U1(bit))) => self.replace(value, Operand::Imm(Imm::U1(!bit))),
            Some(arg) => {
                if let Some(inner) = self.producer_of(Some(arg), Opcode::LogicalNot)?
                    && let Some(original) = inner.arg(0)
                {
                    self.replace(value, original);
                }
            }
            None => {}
        }
        Ok(())
    }

    /// `FoldBitCast`.
    fn bit_cast(&mut self, value: Value, seen: &Seen, reverse: Opcode) -> Result<(), PassError> {
        let Some(arg) = seen.arg(0) else {
            return Ok(());
        };
        if arg.imm().is_some() {
            if let Some(result) = Self::eval(value, seen.opcode, seen.flags, &[arg]) {
                self.replace(value, result);
            }
            return Ok(());
        }
        let Some((_, inner)) = self.producer(arg)? else {
            return Ok(());
        };
        if inner.opcode == reverse {
            if let Some(original) = inner.arg(0) {
                self.replace(value, original);
            }
            return Ok(());
        }
        let rewrite = match (seen.opcode, inner.opcode) {
            // A typed constant buffer read.
            (Opcode::BitCastF32U32, Opcode::GetCbufU32) => Some(Opcode::GetCbufF32),
            // Eden's workaround for an Nvidia driver bug: an integer attribute read.
            (Opcode::BitCastU32F32, Opcode::GetAttribute)
                if matches!(inner.arg(0), Some(Operand::Imm(Imm::Attribute(a)))
                    if INTEGER_ATTRIBUTES.contains(&a)) =>
            {
                Some(Opcode::GetAttributeU32)
            }
            _ => None,
        };
        if let (Some(opcode), Some(a0), Some(a1)) = (rewrite, inner.arg(0), inner.arg(1))
            && let Some(Inst::Op(op)) = self.function.inst_mut(value)
        {
            *op = Operation::new(opcode, vec![a0, a1]);
        }
        Ok(())
    }

    /// `FoldInverseFunc`.
    fn inverse(&mut self, value: Value, seen: &Seen, reverse: Opcode) -> Result<(), PassError> {
        let Some(arg) = seen.arg(0) else {
            return Ok(());
        };
        if arg.imm().is_some() {
            return Ok(());
        }
        if let Some(inner) = self.producer_of(Some(arg), reverse)?
            && let Some(original) = inner.arg(0)
        {
            self.replace(value, original);
        }
        Ok(())
    }

    /// `FoldCompositeExtract`, with `FoldCompositeExtractImpl` as a loop.
    fn extract(
        &mut self,
        value: Value,
        seen: &Seen,
        construct: Opcode,
        insert: Opcode,
    ) -> Result<(), PassError> {
        let (Some(mut composite), Some(Operand::Imm(Imm::U32(index)))) = (seen.arg(0), seen.arg(1))
        else {
            return Ok(());
        };
        let Ok(slot) = usize::try_from(index) else {
            return Ok(());
        };
        if composite.imm().is_some() {
            return Ok(());
        }
        // Each step moves to an earlier instruction, so a chain is at most the arena long.
        for _ in 0..=super::arena_len(self.function) {
            self.work.step();
            let Some((_, inner)) = self.producer(composite)? else {
                return Ok(());
            };
            if inner.opcode == construct {
                if let Some(component) = inner.arg(slot).filter(|_| slot < construct.args().len()) {
                    self.replace(value, component);
                }
                return Ok(());
            }
            if inner.opcode != insert {
                return Ok(());
            }
            let Some(Operand::Imm(Imm::U32(at))) = inner.arg(2) else {
                return Ok(());
            };
            if at == index {
                if let Some(inserted) = inner.arg(1) {
                    self.replace(value, inserted);
                }
                return Ok(());
            }
            match inner.arg(0) {
                Some(next @ Operand::Value(_)) => composite = next,
                _ => return Ok(()),
            }
        }
        Ok(())
    }

    /// `FoldFSwizzleAdd`.
    fn swizzle_add(&mut self, value: Value, seen: &Seen) -> Result<(), PassError> {
        let Some(Operand::Imm(Imm::U32(swizzle))) = seen.arg(2) else {
            return Ok(());
        };
        let (Some(a), Some(b)) = (seen.arg(0), seen.arg(1)) else {
            return Ok(());
        };
        let a = self.resolve(a)?;
        let value_1 = self.through_cast(a, Opcode::BitCastF32U32)?;
        let b = self.resolve(b)?;
        let value_2 = self.through_cast(b, Opcode::BitCastF32U32)?;
        if value_1.imm().is_some() {
            return Ok(());
        }
        if swizzle != SWIZZLE_DX && swizzle != SWIZZLE_DY {
            return Ok(());
        }
        let Some((_, shuffle)) = self.producer(value_1)? else {
            return Ok(());
        };
        let lane_mask = quad_butterfly_mask(&shuffle);
        if lane_mask == 0 {
            return Ok(());
        }
        let Some(source) = shuffle.arg(0) else {
            return Ok(());
        };
        let source = self.resolve(source)?;
        let value_3 = self.through_cast(source, Opcode::BitCastU32F32)?;
        if value_2 != value_3 {
            match (value_2, value_3) {
                (Operand::Imm(Imm::F32(x)), Operand::Imm(Imm::U32(y))) if x == y => {}
                _ => return Ok(()),
            }
        }
        let derivative = match (swizzle, lane_mask) {
            (SWIZZLE_DX, 1) => Opcode::DPdxFine,
            (SWIZZLE_DY, 2) => Opcode::DPdyFine,
            _ => return Ok(()),
        };
        let Some(argument) = seen.arg(1) else {
            return Ok(());
        };
        let made = self.make(derivative, vec![argument])?;
        self.replace(value, Operand::Value(made));
        Ok(())
    }

    /// Eden's `GetThroughCast`: the argument of `cast`, resolved, if `operand` is one.
    fn through_cast(&mut self, operand: Operand, cast: Opcode) -> Result<Operand, PassError> {
        if operand.imm().is_some() {
            return Ok(operand);
        }
        match self.producer_of(Some(operand), cast)? {
            Some(inner) => match inner.arg(0) {
                Some(arg) => self.resolve(arg),
                None => Ok(operand),
            },
            None => Ok(operand),
        }
    }
}

/// Eden's `QuadButterflyMask`: the lane mask of a quad butterfly, or 0.
fn quad_butterfly_mask(shuffle: &Seen) -> u32 {
    match shuffle.opcode {
        Opcode::QuadSwap => match shuffle.arg(1) {
            Some(Operand::Imm(Imm::U32(direction))) => direction.wrapping_add(1),
            _ => 0,
        },
        Opcode::ShuffleButterfly => match (shuffle.arg(1), shuffle.arg(2), shuffle.arg(3)) {
            (
                Some(Operand::Imm(Imm::U32(index))),
                Some(Operand::Imm(Imm::U32(QUAD_CLAMP))),
                Some(Operand::Imm(Imm::U32(QUAD_SEGMENTATION))),
            ) => index,
            _ => 0,
        },
        _ => 0,
    }
}

/// Whether Eden's fold of a bit-field operation is defined (see the module docs): every
/// argument but the base an immediate, and the field inside the word.
fn bit_field_defined(seen: &Seen) -> bool {
    let imm = |index: usize| match seen.arg(index) {
        Some(Operand::Imm(Imm::U32(value))) => Some(u64::from(value)),
        _ => None,
    };
    let word = u64::from(WORD_BITS);
    match seen.opcode {
        Opcode::BitFieldUExtract => matches!(
            (imm(1), imm(2)),
            (Some(shift), Some(count)) if shift + count <= word && count < word
        ),
        Opcode::BitFieldSExtract => matches!(
            (imm(1), imm(2)),
            (Some(shift), Some(count)) if shift + count <= word && count != 0
        ),
        Opcode::BitFieldInsert => matches!(
            (imm(2), imm(3)),
            (Some(offset), Some(count)) if count < word && offset < word
        ),
        _ => false,
    }
}
