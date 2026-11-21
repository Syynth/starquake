//! The f64 pass's table: what each opcode that names a 64-bit float type becomes when the host
//! target lacks 64-bit floats.
//!
//! Ported by behaviour from Eden's `Replace` and `Lower`
//! (`eden/src/shader_recompiler/ir_opt/lower_fp64_to_fp32.cpp` @ cf654052, GPL-3.0-or-later
//! over GPL-2.0-or-later yuzu code; identical in `eden-pr4588` @ e5fb3b97): each 64-bit
//! operation becomes its 32-bit twin, `ConvertF32F64` and `ConvertF64F32` become `Identity`,
//! and `PackDouble2x32` and `UnpackDouble2x32` are written out with 32-bit integer operations
//! (the parent's `bits` module).
//!
//! Beyond Eden's list (each a listed divergence of the parent module, pinned by
//! `tests::table::every_narrowed_opcode_is_mapped_or_refused` and
//! `tests::table::mappings_match_the_reference_literals`): `FPMax64`, `FPMin64`, `FPRecip64`,
//! `FPRecipSqrt64` and `SelectF64` become their 32-bit twins, and `BitCastU64F64` and
//! `BitCastF64U64` are written out through the same conversions. Eden leaves these as 64-bit
//! operations, which its backend then writes with 64-bit float types the host lacks.

use super::{Expansion, Rewrite};
use crate::ir::{Opcode, Type};

/// The type a 64-bit float type becomes: its 32-bit twin.
pub(super) const fn narrow_type(ty: Type) -> Option<Type> {
    Some(match ty {
        Type::F64 => Type::F32,
        Type::F64x2 => Type::F32x2,
        Type::F64x3 => Type::F32x3,
        Type::F64x4 => Type::F32x4,
        _ => return None,
    })
}

/// What `opcode` becomes (module docs).
pub(super) const fn rewrite(opcode: Opcode) -> Rewrite {
    use Opcode as O;
    let renamed = match opcode {
        // Eden's `Replace`, in its order.
        O::FPAbs64 => O::FPAbs32,
        O::FPAdd64 => O::FPAdd32,
        O::FPCeil64 => O::FPCeil32,
        O::FPFloor64 => O::FPFloor32,
        O::FPFma64 => O::FPFma32,
        O::FPMul64 => O::FPMul32,
        O::FPNeg64 => O::FPNeg32,
        O::FPRoundEven64 => O::FPRoundEven32,
        O::FPSaturate64 => O::FPSaturate32,
        O::FPClamp64 => O::FPClamp32,
        O::FPTrunc64 => O::FPTrunc32,
        O::CompositeConstructF64x2 => O::CompositeConstructF32x2,
        O::CompositeConstructF64x3 => O::CompositeConstructF32x3,
        O::CompositeConstructF64x4 => O::CompositeConstructF32x4,
        O::CompositeExtractF64x2 => O::CompositeExtractF32x2,
        O::CompositeExtractF64x3 => O::CompositeExtractF32x3,
        O::CompositeExtractF64x4 => O::CompositeExtractF32x4,
        O::CompositeInsertF64x2 => O::CompositeInsertF32x2,
        O::CompositeInsertF64x3 => O::CompositeInsertF32x3,
        O::CompositeInsertF64x4 => O::CompositeInsertF32x4,
        O::FPOrdEqual64 => O::FPOrdEqual32,
        O::FPUnordEqual64 => O::FPUnordEqual32,
        O::FPOrdNotEqual64 => O::FPOrdNotEqual32,
        O::FPUnordNotEqual64 => O::FPUnordNotEqual32,
        O::FPOrdLessThan64 => O::FPOrdLessThan32,
        O::FPUnordLessThan64 => O::FPUnordLessThan32,
        O::FPOrdGreaterThan64 => O::FPOrdGreaterThan32,
        O::FPUnordGreaterThan64 => O::FPUnordGreaterThan32,
        O::FPOrdLessThanEqual64 => O::FPOrdLessThanEqual32,
        O::FPUnordLessThanEqual64 => O::FPUnordLessThanEqual32,
        O::FPOrdGreaterThanEqual64 => O::FPOrdGreaterThanEqual32,
        O::FPUnordGreaterThanEqual64 => O::FPUnordGreaterThanEqual32,
        O::FPIsNan64 => O::FPIsNan32,
        O::ConvertS16F64 => O::ConvertS16F32,
        O::ConvertS32F64 => O::ConvertS32F32,
        O::ConvertS64F64 => O::ConvertS64F32,
        O::ConvertU16F64 => O::ConvertU16F32,
        O::ConvertU32F64 => O::ConvertU32F32,
        O::ConvertU64F64 => O::ConvertU64F32,
        O::ConvertF32F64 | O::ConvertF64F32 => O::Identity,
        O::ConvertF64S8 => O::ConvertF32S8,
        O::ConvertF64S16 => O::ConvertF32S16,
        O::ConvertF64S32 => O::ConvertF32S32,
        O::ConvertF64S64 => O::ConvertF32S64,
        O::ConvertF64U8 => O::ConvertF32U8,
        O::ConvertF64U16 => O::ConvertF32U16,
        O::ConvertF64U32 => O::ConvertF32U32,
        O::ConvertF64U64 => O::ConvertF32U64,
        // Not in Eden's list (module docs).
        O::FPMax64 => O::FPMax32,
        O::FPMin64 => O::FPMin32,
        O::FPRecip64 => O::FPRecip32,
        O::FPRecipSqrt64 => O::FPRecipSqrt32,
        O::SelectF64 => O::SelectF32,
        // Eden's `Lower`.
        O::PackDouble2x32 => return Rewrite::Expand(Expansion::PackDouble),
        O::UnpackDouble2x32 => return Rewrite::Expand(Expansion::UnpackDouble),
        // Not in Eden's list (module docs).
        O::BitCastF64U64 => return Rewrite::Expand(Expansion::DoubleFromBits),
        O::BitCastU64F64 => return Rewrite::Expand(Expansion::BitsFromDouble),
        _ => return Rewrite::Keep,
    };
    Rewrite::Rename(renamed)
}
