//! Which opcodes are lowered (the module docs' list, enumerated over the whole table), and the
//! refusals of operand forms the lowering does not handle.

use super::ops::{bit, f, op, u};
use super::*;
use crate::decode::{Pred, UniformPred, UniformReg};
use crate::host::Feature;
use crate::ir::{Builder, Flags, FpControl, FpRounding, Patch, Type};

/// An operand of type `ty` for the enumeration below: an immediate where the type has one, else
/// a value made in front of the operation (a vector is constructed, an f16 converted).
fn operand_of(builder: &mut Builder<'_>, ty: Type) -> Operand {
    let zero = u(0);
    match ty {
        Type::U1 => bit(false),
        Type::U8 => Operand::Imm(Imm::U8(0)),
        Type::U16 => Operand::Imm(Imm::U16(0)),
        Type::U64 => Operand::Imm(Imm::U64(0)),
        Type::F32 => f(0.0),
        Type::F64 => Operand::Imm(Imm::F64(0)),
        Type::Reg => Operand::Imm(Imm::Reg(Gpr::new(1))),
        Type::Pred => Operand::Imm(Imm::Pred(Pred::new(1).expect("p1"))),
        Type::UniformReg => Operand::Imm(Imm::UniformReg(UniformReg::new(1).expect("ur1"))),
        Type::UniformPred => Operand::Imm(Imm::UniformPred(UniformPred::new(1).expect("up1"))),
        Type::Attribute => Operand::Imm(Imm::Attribute(Attribute(GENERIC_0_X))),
        Type::Patch => Operand::Imm(Imm::Patch(Patch::new(0).expect("patch 0"))),
        Type::F16 => op(builder, Opcode::ConvertF16F32, &[f(0.0)]).into(),
        Type::U32x2 => op(builder, Opcode::CompositeConstructU32x2, &[zero, zero]).into(),
        Type::U32x3 => op(builder, Opcode::CompositeConstructU32x3, &[zero; 3]).into(),
        Type::U32x4 => op(builder, Opcode::CompositeConstructU32x4, &[zero; 4]).into(),
        Type::F32x2 => op(builder, Opcode::CompositeConstructF32x2, &[f(0.0); 2]).into(),
        Type::F32x3 => op(builder, Opcode::CompositeConstructF32x3, &[f(0.0); 3]).into(),
        Type::F32x4 => op(builder, Opcode::CompositeConstructF32x4, &[f(0.0); 4]).into(),
        Type::F16x2 | Type::F16x3 | Type::F16x4 => {
            let half: Operand = op(builder, Opcode::ConvertF16F32, &[f(0.0)]).into();
            let construct = match ty {
                Type::F16x2 => Opcode::CompositeConstructF16x2,
                Type::F16x3 => Opcode::CompositeConstructF16x3,
                _ => Opcode::CompositeConstructF16x4,
            };
            let count = construct.args().len();
            op(builder, construct, &vec![half; count]).into()
        }
        Type::F64x2 | Type::F64x3 | Type::F64x4 => {
            let construct = match ty {
                Type::F64x2 => Opcode::CompositeConstructF64x2,
                Type::F64x3 => Opcode::CompositeConstructF64x3,
                _ => Opcode::CompositeConstructF64x4,
            };
            let count = construct.args().len();
            op(builder, construct, &vec![Operand::Imm(Imm::F64(0)); count]).into()
        }
        // `u32` and the any-type slots.
        Type::U32 | Type::Opaque | Type::Void => zero,
    }
}

/// The stage an opcode is tried in: the one that may use it.
fn stage_for(opcode: Opcode) -> GuestStage {
    match opcode {
        Opcode::DemoteToHelperInvocation | Opcode::SetFragColor | Opcode::LaneId => {
            GuestStage::Pixel
        }
        Opcode::GetAttribute | Opcode::SetAttribute => GuestStage::Vertex,
        _ => GuestStage::Compute,
    }
}

/// Every opcode the lowering handles, written out from the module docs' groups, in table
/// order.
const LOWERED: &[Opcode] = &[
    Opcode::Identity,
    Opcode::Void,
    Opcode::ConditionRef,
    Opcode::Prologue,
    Opcode::Epilogue,
    Opcode::DemoteToHelperInvocation,
    Opcode::Barrier,
    Opcode::WorkgroupMemoryBarrier,
    Opcode::DeviceMemoryBarrier,
    Opcode::GetRegister,
    Opcode::SetRegister,
    Opcode::GetPred,
    Opcode::SetPred,
    Opcode::GetUniformRegister,
    Opcode::SetUniformRegister,
    Opcode::GetUniformPred,
    Opcode::SetUniformPred,
    Opcode::GetGotoVariable,
    Opcode::SetGotoVariable,
    Opcode::GetIndirectBranchVariable,
    Opcode::SetIndirectBranchVariable,
    Opcode::GetAttribute,
    Opcode::SetAttribute,
    Opcode::SetFragColor,
    Opcode::GetZFlag,
    Opcode::GetSFlag,
    Opcode::GetCFlag,
    Opcode::GetOFlag,
    Opcode::SetZFlag,
    Opcode::SetSFlag,
    Opcode::SetCFlag,
    Opcode::SetOFlag,
    Opcode::WorkgroupId,
    Opcode::LocalInvocationId,
    Opcode::UndefU1,
    Opcode::UndefU32,
    Opcode::UndefU64,
    Opcode::LoadLocal,
    Opcode::WriteLocal,
    Opcode::LoadSharedU8,
    Opcode::LoadSharedS8,
    Opcode::LoadSharedU16,
    Opcode::LoadSharedS16,
    Opcode::LoadSharedU32,
    Opcode::LoadSharedU64,
    Opcode::LoadSharedU128,
    Opcode::WriteSharedU32,
    Opcode::WriteSharedU64,
    Opcode::WriteSharedU128,
    Opcode::CompositeConstructU32x2,
    Opcode::CompositeConstructU32x3,
    Opcode::CompositeConstructU32x4,
    Opcode::CompositeExtractU32x2,
    Opcode::CompositeExtractU32x3,
    Opcode::CompositeExtractU32x4,
    Opcode::CompositeInsertU32x2,
    Opcode::CompositeInsertU32x3,
    Opcode::CompositeInsertU32x4,
    Opcode::CompositeConstructF16x2,
    Opcode::CompositeConstructF16x3,
    Opcode::CompositeConstructF16x4,
    Opcode::CompositeExtractF16x2,
    Opcode::CompositeExtractF16x3,
    Opcode::CompositeExtractF16x4,
    Opcode::CompositeInsertF16x2,
    Opcode::CompositeInsertF16x3,
    Opcode::CompositeInsertF16x4,
    Opcode::CompositeConstructF32x2,
    Opcode::CompositeConstructF32x3,
    Opcode::CompositeConstructF32x4,
    Opcode::CompositeExtractF32x2,
    Opcode::CompositeExtractF32x3,
    Opcode::CompositeExtractF32x4,
    Opcode::CompositeInsertF32x2,
    Opcode::CompositeInsertF32x3,
    Opcode::CompositeInsertF32x4,
    Opcode::SelectU1,
    Opcode::SelectU32,
    Opcode::SelectU64,
    Opcode::SelectF16,
    Opcode::SelectF32,
    Opcode::BitCastU32F32,
    Opcode::BitCastF32U32,
    Opcode::PackUint2x32,
    Opcode::UnpackUint2x32,
    Opcode::PackFloat2x16,
    Opcode::UnpackFloat2x16,
    Opcode::PackHalf2x16,
    Opcode::UnpackHalf2x16,
    Opcode::FPAbs16,
    Opcode::FPAbs32,
    Opcode::FPAdd16,
    Opcode::FPAdd32,
    Opcode::FPFma16,
    Opcode::FPFma32,
    Opcode::FPMax32,
    Opcode::FPMin32,
    Opcode::FPMul16,
    Opcode::FPMul32,
    Opcode::FPNeg16,
    Opcode::FPNeg32,
    Opcode::FPRecip32,
    Opcode::FPRecipSqrt32,
    Opcode::FPSqrt,
    Opcode::FPSin,
    Opcode::FPExp2,
    Opcode::FPCos,
    Opcode::FPLog2,
    Opcode::FPSaturate16,
    Opcode::FPSaturate32,
    Opcode::FPClamp16,
    Opcode::FPClamp32,
    Opcode::FPRoundEven16,
    Opcode::FPRoundEven32,
    Opcode::FPFloor16,
    Opcode::FPFloor32,
    Opcode::FPCeil16,
    Opcode::FPCeil32,
    Opcode::FPTrunc16,
    Opcode::FPTrunc32,
    Opcode::FPOrdEqual16,
    Opcode::FPOrdEqual32,
    Opcode::FPUnordEqual16,
    Opcode::FPUnordEqual32,
    Opcode::FPOrdNotEqual16,
    Opcode::FPOrdNotEqual32,
    Opcode::FPUnordNotEqual16,
    Opcode::FPUnordNotEqual32,
    Opcode::FPOrdLessThan16,
    Opcode::FPOrdLessThan32,
    Opcode::FPUnordLessThan16,
    Opcode::FPUnordLessThan32,
    Opcode::FPOrdGreaterThan16,
    Opcode::FPOrdGreaterThan32,
    Opcode::FPUnordGreaterThan16,
    Opcode::FPUnordGreaterThan32,
    Opcode::FPOrdLessThanEqual16,
    Opcode::FPOrdLessThanEqual32,
    Opcode::FPUnordLessThanEqual16,
    Opcode::FPUnordLessThanEqual32,
    Opcode::FPOrdGreaterThanEqual16,
    Opcode::FPOrdGreaterThanEqual32,
    Opcode::FPUnordGreaterThanEqual16,
    Opcode::FPUnordGreaterThanEqual32,
    Opcode::FPIsNan16,
    Opcode::FPIsNan32,
    Opcode::IAdd32,
    Opcode::IAdd64,
    Opcode::ISub32,
    Opcode::ISub64,
    Opcode::IMul32,
    Opcode::SDiv32,
    Opcode::UDiv32,
    Opcode::INeg32,
    Opcode::INeg64,
    Opcode::IAbs32,
    Opcode::IAbs64,
    Opcode::ShiftLeftLogical32,
    Opcode::ShiftLeftLogical64,
    Opcode::ShiftRightLogical32,
    Opcode::ShiftRightLogical64,
    Opcode::ShiftRightArithmetic32,
    Opcode::ShiftRightArithmetic64,
    Opcode::BitwiseAnd32,
    Opcode::BitwiseOr32,
    Opcode::BitwiseXor32,
    Opcode::BitFieldInsert,
    Opcode::BitFieldSExtract,
    Opcode::BitFieldUExtract,
    Opcode::BitReverse32,
    Opcode::BitCount32,
    Opcode::BitwiseNot32,
    Opcode::FindSMsb32,
    Opcode::FindUMsb32,
    Opcode::SMin32,
    Opcode::UMin32,
    Opcode::SMax32,
    Opcode::UMax32,
    Opcode::SClamp32,
    Opcode::UClamp32,
    Opcode::SLessThan,
    Opcode::ULessThan,
    Opcode::IEqual,
    Opcode::SLessThanEqual,
    Opcode::ULessThanEqual,
    Opcode::SGreaterThan,
    Opcode::UGreaterThan,
    Opcode::INotEqual,
    Opcode::SGreaterThanEqual,
    Opcode::UGreaterThanEqual,
    Opcode::LogicalOr,
    Opcode::LogicalAnd,
    Opcode::LogicalXor,
    Opcode::LogicalNot,
    Opcode::ConvertS16F16,
    Opcode::ConvertS16F32,
    Opcode::ConvertS32F16,
    Opcode::ConvertS32F32,
    Opcode::ConvertU16F16,
    Opcode::ConvertU16F32,
    Opcode::ConvertU32F16,
    Opcode::ConvertU32F32,
    Opcode::ConvertU64U32,
    Opcode::ConvertU32U64,
    Opcode::ConvertF16F32,
    Opcode::ConvertF32F16,
    Opcode::ConvertF16S8,
    Opcode::ConvertF16S16,
    Opcode::ConvertF16S32,
    Opcode::ConvertF16U8,
    Opcode::ConvertF16U16,
    Opcode::ConvertF16U32,
    Opcode::ConvertF32S8,
    Opcode::ConvertF32S16,
    Opcode::ConvertF32S32,
    Opcode::ConvertF32U8,
    Opcode::ConvertF32U16,
    Opcode::ConvertF32U32,
    Opcode::LaneId,
];

/// Every opcode of the table, each alone in a function of the stage that may use it: the ones
/// [`LOWERED`] lists lower (and validate with wgpu's flags too); every other one is refused,
/// with [`LowerError::Operation`] naming it when its own instruction is the first fault.
#[test]
fn every_opcode_is_lowered_or_refused() {
    let inputs = LowerInputs {
        local_memory_bytes: 16,
        shared_memory_bytes: 16,
        ..LowerInputs::DEFAULT
    };
    let mut refused_alone = 0;
    for &opcode in Opcode::ALL {
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        let args: Vec<Operand> = opcode
            .args()
            .iter()
            .map(|&ty| operand_of(&mut builder, ty))
            .collect();
        let at = op(&mut builder, opcode, &args);
        builder
            .terminate(Terminator::Return)
            .expect("the block returns");
        let stage = stage_for(opcode);
        let result = lower_with(stage, &function, &inputs);
        if LOWERED.contains(&opcode) {
            assert!(result.is_ok(), "{opcode} is lowered: {result:?}");
            lowered(stage, &function, &inputs);
        } else {
            match result {
                Ok(_) => panic!("{opcode} is lowered but not listed"),
                // An input of an unlowered type (an f64 vector) fails first.
                Err(LowerError::Operation { at: fault, .. }) if fault != at => {}
                Err(error) => {
                    assert_eq!(error, LowerError::Operation { at, opcode }, "{opcode}");
                    refused_alone += 1;
                }
            }
        }
    }
    assert!(refused_alone > 0);
}

/// A conversion with a directed rounding mode is refused with the mode named; round to
/// nearest and no mode at all are lowered. Each kind of conversion that carries controls.
#[test]
fn directed_rounding_on_a_conversion_is_refused() {
    for opcode in [
        Opcode::ConvertF32S32,
        Opcode::ConvertF16F32,
        Opcode::ConvertF32F16,
        Opcode::ConvertF16U8,
    ] {
        for (rounding, refused) in [
            (FpRounding::DontCare, None),
            (FpRounding::Rn, None),
            (FpRounding::Rm, Some("towards negative infinity")),
            (FpRounding::Rp, Some("towards positive infinity")),
            (FpRounding::Rz, Some("towards zero")),
        ] {
            let mut function = Function::new();
            let entry = function.entry();
            let mut builder = Builder::new(&mut function, entry);
            let arg = operand_of(&mut builder, opcode.args()[0]);
            let flags = Flags::Fp(FpControl {
                rounding,
                ..FpControl::default()
            });
            let at = builder.op_with(opcode, &[arg], flags).expect("valid");
            builder
                .terminate(Terminator::Return)
                .expect("the block returns");
            let result = lower(GuestStage::Compute, &function);
            match refused {
                None => assert!(result.is_ok(), "{opcode} {rounding:?}"),
                Some(rounding) => assert_eq!(
                    result.err(),
                    Some(LowerError::Rounding {
                        at,
                        opcode,
                        rounding
                    })
                ),
            }
        }
    }
}

/// `PackHalf2x16` and `UnpackHalf2x16` need f16 (naga's `SHADER_FLOAT16_IN_FLOAT32` comes with
/// it; a divergence in the module docs): refused without it, and recorded with it.
#[test]
fn f16_packing_needs_f16() {
    for text in [
        "block0:\n    %0: f32x2 = UnpackHalf2x16 u32 0\n    return\n",
        "block0:\n    %0: f32x2 = CompositeConstructF32x2 f32 1.0, f32 2.0\n    \
         %1: u32 = PackHalf2x16 %0\n    return\n",
    ] {
        let function = ir(text);
        let without = LowerInputs {
            caps: HostCaps {
                f16: false,
                ..HostCaps::ALL
            },
            ..LowerInputs::DEFAULT
        };
        assert_eq!(
            lower_with(GuestStage::Compute, &function, &without).err(),
            Some(LowerError::Missing(Feature::Float16))
        );
        let shader = lowered(GuestStage::Compute, &function, &LowerInputs::DEFAULT);
        assert!(shader.requirements.f16);
    }
}

/// A component index of an extract or insert must be an immediate inside the vector: the last
/// component is lowered, one past it and a value are refused with the argument named.
#[test]
fn component_indices_are_immediates_inside_the_vector() {
    let extract = |index: &str| {
        ir(&format!(
            "block0:\n    %0: u32x3 = CompositeConstructU32x3 u32 1, u32 2, u32 3\n    \
             %1: u32 = CompositeExtractU32x3 %0, {index}\n    return\n"
        ))
    };
    assert!(lower(GuestStage::Compute, &extract("u32 2")).is_ok());
    let past = extract("u32 3");
    assert_eq!(
        lower(GuestStage::Compute, &past).err(),
        Some(LowerError::Operand {
            at: value_at(&past, 1),
            index: 1
        })
    );
    let value = ir("block0:
    %0: u32x3 = CompositeConstructU32x3 u32 1, u32 2, u32 3
    %1: u32 = CompositeExtractU32x3 %0, u32 0
    %2: u32 = CompositeExtractU32x3 %0, %1
    return
");
    assert_eq!(
        lower(GuestStage::Compute, &value).err(),
        Some(LowerError::Operand {
            at: value_at(&value, 2),
            index: 1
        })
    );
    let insert = |index: &str| {
        ir(&format!(
            "block0:\n    %0: f32x2 = CompositeConstructF32x2 f32 1.0, f32 2.0\n    \
             %1: f32x2 = CompositeInsertF32x2 %0, f32 3.0, {index}\n    return\n"
        ))
    };
    assert!(lower(GuestStage::Compute, &insert("u32 1")).is_ok());
    let past = insert("u32 2");
    assert_eq!(
        lower(GuestStage::Compute, &past).err(),
        Some(LowerError::Operand {
            at: value_at(&past, 1),
            index: 2
        })
    );
}

/// A NaN or infinite `f32` immediate becomes its bits reinterpreted (naga refuses such
/// literals); a finite one, the largest included, a literal.
#[test]
fn non_finite_immediates_are_bit_casts() {
    for (bits, literal) in [
        (f32::NAN.to_bits(), false),
        (f32::INFINITY.to_bits(), false),
        (f32::NEG_INFINITY.to_bits(), false),
        (f32::MAX.to_bits(), true),
        (1.0_f32.to_bits(), true),
    ] {
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        op(
            &mut builder,
            Opcode::FPAbs32,
            &[Operand::Imm(Imm::F32(bits))],
        );
        builder
            .terminate(Terminator::Return)
            .expect("the block returns");
        let shader = lowered(GuestStage::Compute, &function, &LowerInputs::DEFAULT);
        let expressions = &naga_module(&shader).entry_points[0].function.expressions;
        let first: Vec<&Expression> = expressions.iter().map(|(_, e)| e).take(2).collect();
        let is_literal = matches!(
            first.first(),
            Some(Expression::Literal(naga::Literal::F32(value))) if value.to_bits() == bits
        );
        let is_cast = matches!(
            first.as_slice(),
            [
                Expression::Literal(naga::Literal::U32(found)),
                Expression::As {
                    kind: ScalarKind::Float,
                    convert: None,
                    ..
                }
            ] if *found == bits
        );
        assert_eq!((is_literal, is_cast), (literal, !literal), "{bits:#x}");
    }
}
