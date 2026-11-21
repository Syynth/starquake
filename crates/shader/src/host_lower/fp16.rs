//! The f16 pass's table: what each opcode that names a 16-bit float type becomes when the host
//! target lacks 16-bit floats.
//!
//! Ported by behaviour from Eden's `Replace`
//! (`eden/src/shader_recompiler/ir_opt/lower_fp16_to_fp32.cpp` @ cf654052, GPL-2.0-or-later
//! yuzu code; identical in `eden-pr4588` @ e5fb3b97): each 16-bit operation becomes its 32-bit
//! twin, `ConvertF16F32` and `ConvertF32F16` become `Identity`, the packed-half atomics become
//! their `F32x2` forms, and `PackFloat2x16` and `UnpackFloat2x16` become `PackHalf2x16` and
//! `UnpackHalf2x16`.
//!
//! Where this differs from Eden (each a listed divergence of the parent module):
//!
//! - `PackFloat2x16`, `UnpackFloat2x16`, `PackHalf2x16` and `UnpackHalf2x16` are written out
//!   with 32-bit integer operations (the parent's `bits` module) rather than left as
//!   `PackHalf2x16` and `UnpackHalf2x16`: naga 30.0.1 needs `SHADER_FLOAT16_IN_FLOAT32` for
//!   those, which a target without 16-bit floats may lack (`crate::lower::capabilities`).
//!   Pinned by `tests::equivalence::every_mapped_opcode_matches_the_interpreter` and
//!   `tests::validation::lowered_modules_validate_without_f16_or_f64`.
//! - `SelectF16` becomes `SelectF32`; Eden's list has no select, so its backend writes a 16-bit
//!   select of 32-bit values.
//! - `BitCastU16F16` and `BitCastF16U16` are refused (`HostLowerError::NoNarrowForm`): their
//!   other side is a 16-bit integer, which the lowering has no type for either. Eden leaves
//!   them. Pinned by `tests::refusals::a_16_bit_bit_cast_is_refused`.

use super::{Expansion, Rewrite};
use crate::ir::{Opcode, Type};

/// The type a 16-bit float type becomes: its 32-bit twin.
pub(super) const fn narrow_type(ty: Type) -> Option<Type> {
    Some(match ty {
        Type::F16 => Type::F32,
        Type::F16x2 => Type::F32x2,
        Type::F16x3 => Type::F32x3,
        Type::F16x4 => Type::F32x4,
        _ => return None,
    })
}

/// What `opcode` becomes (module docs).
pub(super) const fn rewrite(opcode: Opcode) -> Rewrite {
    use Opcode as O;
    let renamed = match opcode {
        // Eden's `Replace`, in its order.
        O::FPAbs16 => O::FPAbs32,
        O::FPAdd16 => O::FPAdd32,
        O::FPCeil16 => O::FPCeil32,
        O::FPFloor16 => O::FPFloor32,
        O::FPFma16 => O::FPFma32,
        O::FPMul16 => O::FPMul32,
        O::FPNeg16 => O::FPNeg32,
        O::FPRoundEven16 => O::FPRoundEven32,
        O::FPSaturate16 => O::FPSaturate32,
        O::FPClamp16 => O::FPClamp32,
        O::FPTrunc16 => O::FPTrunc32,
        O::CompositeConstructF16x2 => O::CompositeConstructF32x2,
        O::CompositeConstructF16x3 => O::CompositeConstructF32x3,
        O::CompositeConstructF16x4 => O::CompositeConstructF32x4,
        O::CompositeExtractF16x2 => O::CompositeExtractF32x2,
        O::CompositeExtractF16x3 => O::CompositeExtractF32x3,
        O::CompositeExtractF16x4 => O::CompositeExtractF32x4,
        O::CompositeInsertF16x2 => O::CompositeInsertF32x2,
        O::CompositeInsertF16x3 => O::CompositeInsertF32x3,
        O::CompositeInsertF16x4 => O::CompositeInsertF32x4,
        O::FPOrdEqual16 => O::FPOrdEqual32,
        O::FPUnordEqual16 => O::FPUnordEqual32,
        O::FPOrdNotEqual16 => O::FPOrdNotEqual32,
        O::FPUnordNotEqual16 => O::FPUnordNotEqual32,
        O::FPOrdLessThan16 => O::FPOrdLessThan32,
        O::FPUnordLessThan16 => O::FPUnordLessThan32,
        O::FPOrdGreaterThan16 => O::FPOrdGreaterThan32,
        O::FPUnordGreaterThan16 => O::FPUnordGreaterThan32,
        O::FPOrdLessThanEqual16 => O::FPOrdLessThanEqual32,
        O::FPUnordLessThanEqual16 => O::FPUnordLessThanEqual32,
        O::FPOrdGreaterThanEqual16 => O::FPOrdGreaterThanEqual32,
        O::FPUnordGreaterThanEqual16 => O::FPUnordGreaterThanEqual32,
        O::FPIsNan16 => O::FPIsNan32,
        O::ConvertS16F16 => O::ConvertS16F32,
        O::ConvertS32F16 => O::ConvertS32F32,
        O::ConvertS64F16 => O::ConvertS64F32,
        O::ConvertU16F16 => O::ConvertU16F32,
        O::ConvertU32F16 => O::ConvertU32F32,
        O::ConvertU64F16 => O::ConvertU64F32,
        O::ConvertF32F16 | O::ConvertF16F32 => O::Identity,
        O::ConvertF16S8 => O::ConvertF32S8,
        O::ConvertF16S16 => O::ConvertF32S16,
        O::ConvertF16S32 => O::ConvertF32S32,
        O::ConvertF16S64 => O::ConvertF32S64,
        O::ConvertF16U8 => O::ConvertF32U8,
        O::ConvertF16U16 => O::ConvertF32U16,
        O::ConvertF16U32 => O::ConvertF32U32,
        O::ConvertF16U64 => O::ConvertF32U64,
        O::GlobalAtomicAddF16x2 => O::GlobalAtomicAddF32x2,
        O::StorageAtomicAddF16x2 => O::StorageAtomicAddF32x2,
        O::GlobalAtomicMinF16x2 => O::GlobalAtomicMinF32x2,
        O::StorageAtomicMinF16x2 => O::StorageAtomicMinF32x2,
        O::GlobalAtomicMaxF16x2 => O::GlobalAtomicMaxF32x2,
        O::StorageAtomicMaxF16x2 => O::StorageAtomicMaxF32x2,
        // Not in Eden's list (module docs).
        O::SelectF16 => O::SelectF32,
        // Eden renames these two to `PackHalf2x16` and `UnpackHalf2x16`; here all four are
        // written out (module docs).
        O::PackFloat2x16 | O::PackHalf2x16 => return Rewrite::Expand(Expansion::PackHalves),
        O::UnpackFloat2x16 | O::UnpackHalf2x16 => {
            return Rewrite::Expand(Expansion::UnpackHalves);
        }
        _ => return Rewrite::Keep,
    };
    Rewrite::Rename(renamed)
}
