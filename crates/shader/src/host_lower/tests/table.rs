//! The tables: every opcode naming a narrowed type is mapped or refused as the module docs
//! list, each renaming keeps the signature with the narrowed types replaced, and the mappings
//! are Eden's, written out here as literals from its source.

use super::super::{Expansion, HostPass, Rewrite};
use crate::ir::{Opcode, Type};

/// The opcodes each pass refuses (module docs, "Divergences from Eden").
fn refused(pass: HostPass) -> &'static [Opcode] {
    match pass {
        HostPass::Fp64ToFp32 => &[],
        HostPass::Fp16ToFp32 => &[Opcode::BitCastU16F16, Opcode::BitCastF16U16],
    }
}

/// `ty` with the pass's narrowed type replaced.
fn narrowed(pass: HostPass, ty: Type) -> Type {
    pass.narrow_type(ty).unwrap_or(ty)
}

/// Every opcode is checked against the pass's rule: one that names a narrowed type is renamed
/// to an opcode with the same signature, narrowed (or to `Identity` for the float-to-float
/// conversions, whose result and argument then have one type), written out, or in the refused
/// list; one that names none is kept, but for the f16 pass's two `Half2x16` packings, which
/// it writes out too.
#[test]
fn every_narrowed_opcode_is_mapped_or_refused() {
    for pass in HostPass::ORDER {
        let mut refusals = Vec::new();
        for &opcode in Opcode::ALL {
            let names = pass.names_narrowed_type(opcode);
            match pass.rewrite(opcode) {
                Rewrite::Keep if names => refusals.push(opcode),
                Rewrite::Keep => {}
                Rewrite::Rename(Opcode::Identity) => {
                    assert!(names, "{opcode}");
                    let [arg] = opcode.args() else {
                        panic!("{opcode} has one argument");
                    };
                    assert_eq!(narrowed(pass, opcode.result()), narrowed(pass, *arg), "{opcode}");
                    assert_ne!(opcode.result(), *arg, "{opcode} converts");
                }
                Rewrite::Rename(renamed) => {
                    assert!(names, "{opcode}");
                    assert!(!pass.names_narrowed_type(renamed), "{opcode} -> {renamed}");
                    assert_eq!(narrowed(pass, opcode.result()), renamed.result(), "{opcode}");
                    let args: Vec<Type> =
                        opcode.args().iter().map(|&ty| narrowed(pass, ty)).collect();
                    assert_eq!(args, renamed.args(), "{opcode} -> {renamed}");
                }
                Rewrite::Expand(_) => {
                    let half_packing = matches!(
                        opcode,
                        Opcode::PackHalf2x16 | Opcode::UnpackHalf2x16
                    );
                    assert!(
                        names || (pass == HostPass::Fp16ToFp32 && half_packing),
                        "{opcode}"
                    );
                }
            }
        }
        assert_eq!(refusals, refused(pass), "{pass}");
    }
}

/// Eden's `Replace` of `lower_fp64_to_fp32.cpp` @ cf654052, row for row, and the additions the
/// `fp64` module lists.
const F64_RENAMES: [(Opcode, Opcode); 53] = [
    (Opcode::FPAbs64, Opcode::FPAbs32),
    (Opcode::FPAdd64, Opcode::FPAdd32),
    (Opcode::FPCeil64, Opcode::FPCeil32),
    (Opcode::FPFloor64, Opcode::FPFloor32),
    (Opcode::FPFma64, Opcode::FPFma32),
    (Opcode::FPMul64, Opcode::FPMul32),
    (Opcode::FPNeg64, Opcode::FPNeg32),
    (Opcode::FPRoundEven64, Opcode::FPRoundEven32),
    (Opcode::FPSaturate64, Opcode::FPSaturate32),
    (Opcode::FPClamp64, Opcode::FPClamp32),
    (Opcode::FPTrunc64, Opcode::FPTrunc32),
    (Opcode::CompositeConstructF64x2, Opcode::CompositeConstructF32x2),
    (Opcode::CompositeConstructF64x3, Opcode::CompositeConstructF32x3),
    (Opcode::CompositeConstructF64x4, Opcode::CompositeConstructF32x4),
    (Opcode::CompositeExtractF64x2, Opcode::CompositeExtractF32x2),
    (Opcode::CompositeExtractF64x3, Opcode::CompositeExtractF32x3),
    (Opcode::CompositeExtractF64x4, Opcode::CompositeExtractF32x4),
    (Opcode::CompositeInsertF64x2, Opcode::CompositeInsertF32x2),
    (Opcode::CompositeInsertF64x3, Opcode::CompositeInsertF32x3),
    (Opcode::CompositeInsertF64x4, Opcode::CompositeInsertF32x4),
    (Opcode::FPOrdEqual64, Opcode::FPOrdEqual32),
    (Opcode::FPUnordEqual64, Opcode::FPUnordEqual32),
    (Opcode::FPOrdNotEqual64, Opcode::FPOrdNotEqual32),
    (Opcode::FPUnordNotEqual64, Opcode::FPUnordNotEqual32),
    (Opcode::FPOrdLessThan64, Opcode::FPOrdLessThan32),
    (Opcode::FPUnordLessThan64, Opcode::FPUnordLessThan32),
    (Opcode::FPOrdGreaterThan64, Opcode::FPOrdGreaterThan32),
    (Opcode::FPUnordGreaterThan64, Opcode::FPUnordGreaterThan32),
    (Opcode::FPOrdLessThanEqual64, Opcode::FPOrdLessThanEqual32),
    (Opcode::FPUnordLessThanEqual64, Opcode::FPUnordLessThanEqual32),
    (Opcode::FPOrdGreaterThanEqual64, Opcode::FPOrdGreaterThanEqual32),
    (Opcode::FPUnordGreaterThanEqual64, Opcode::FPUnordGreaterThanEqual32),
    (Opcode::FPIsNan64, Opcode::FPIsNan32),
    (Opcode::ConvertS16F64, Opcode::ConvertS16F32),
    (Opcode::ConvertS32F64, Opcode::ConvertS32F32),
    (Opcode::ConvertS64F64, Opcode::ConvertS64F32),
    (Opcode::ConvertU16F64, Opcode::ConvertU16F32),
    (Opcode::ConvertU32F64, Opcode::ConvertU32F32),
    (Opcode::ConvertU64F64, Opcode::ConvertU64F32),
    (Opcode::ConvertF32F64, Opcode::Identity),
    (Opcode::ConvertF64F32, Opcode::Identity),
    (Opcode::ConvertF64S8, Opcode::ConvertF32S8),
    (Opcode::ConvertF64S16, Opcode::ConvertF32S16),
    (Opcode::ConvertF64S32, Opcode::ConvertF32S32),
    (Opcode::ConvertF64S64, Opcode::ConvertF32S64),
    (Opcode::ConvertF64U8, Opcode::ConvertF32U8),
    (Opcode::ConvertF64U16, Opcode::ConvertF32U16),
    (Opcode::ConvertF64U32, Opcode::ConvertF32U32),
    (Opcode::ConvertF64U64, Opcode::ConvertF32U64),
    // Not in Eden's list.
    (Opcode::FPMax64, Opcode::FPMax32),
    (Opcode::FPMin64, Opcode::FPMin32),
    (Opcode::FPRecip64, Opcode::FPRecip32),
    (Opcode::FPRecipSqrt64, Opcode::FPRecipSqrt32),
];

/// The f64 pass's written-out operations: Eden's `Lower` (`PackDouble2x32`,
/// `UnpackDouble2x32`) and the additions.
const F64_EXPANSIONS: [(Opcode, Expansion); 4] = [
    (Opcode::PackDouble2x32, Expansion::PackDouble),
    (Opcode::UnpackDouble2x32, Expansion::UnpackDouble),
    (Opcode::BitCastF64U64, Expansion::DoubleFromBits),
    (Opcode::BitCastU64F64, Expansion::BitsFromDouble),
];

/// `SelectF64`, the fifth addition the `fp64` module lists ([`F64_RENAMES`] holds Eden's 49
/// rows and the other four).
const F64_SELECT: (Opcode, Opcode) = (Opcode::SelectF64, Opcode::SelectF32);

/// Eden's `Replace` of `lower_fp16_to_fp32.cpp` @ cf654052, row for row but for its two
/// packings (written out here: [`F16_EXPANSIONS`]).
const F16_RENAMES: [(Opcode, Opcode); 55] = [
    (Opcode::FPAbs16, Opcode::FPAbs32),
    (Opcode::FPAdd16, Opcode::FPAdd32),
    (Opcode::FPCeil16, Opcode::FPCeil32),
    (Opcode::FPFloor16, Opcode::FPFloor32),
    (Opcode::FPFma16, Opcode::FPFma32),
    (Opcode::FPMul16, Opcode::FPMul32),
    (Opcode::FPNeg16, Opcode::FPNeg32),
    (Opcode::FPRoundEven16, Opcode::FPRoundEven32),
    (Opcode::FPSaturate16, Opcode::FPSaturate32),
    (Opcode::FPClamp16, Opcode::FPClamp32),
    (Opcode::FPTrunc16, Opcode::FPTrunc32),
    (Opcode::CompositeConstructF16x2, Opcode::CompositeConstructF32x2),
    (Opcode::CompositeConstructF16x3, Opcode::CompositeConstructF32x3),
    (Opcode::CompositeConstructF16x4, Opcode::CompositeConstructF32x4),
    (Opcode::CompositeExtractF16x2, Opcode::CompositeExtractF32x2),
    (Opcode::CompositeExtractF16x3, Opcode::CompositeExtractF32x3),
    (Opcode::CompositeExtractF16x4, Opcode::CompositeExtractF32x4),
    (Opcode::CompositeInsertF16x2, Opcode::CompositeInsertF32x2),
    (Opcode::CompositeInsertF16x3, Opcode::CompositeInsertF32x3),
    (Opcode::CompositeInsertF16x4, Opcode::CompositeInsertF32x4),
    (Opcode::FPOrdEqual16, Opcode::FPOrdEqual32),
    (Opcode::FPUnordEqual16, Opcode::FPUnordEqual32),
    (Opcode::FPOrdNotEqual16, Opcode::FPOrdNotEqual32),
    (Opcode::FPUnordNotEqual16, Opcode::FPUnordNotEqual32),
    (Opcode::FPOrdLessThan16, Opcode::FPOrdLessThan32),
    (Opcode::FPUnordLessThan16, Opcode::FPUnordLessThan32),
    (Opcode::FPOrdGreaterThan16, Opcode::FPOrdGreaterThan32),
    (Opcode::FPUnordGreaterThan16, Opcode::FPUnordGreaterThan32),
    (Opcode::FPOrdLessThanEqual16, Opcode::FPOrdLessThanEqual32),
    (Opcode::FPUnordLessThanEqual16, Opcode::FPUnordLessThanEqual32),
    (Opcode::FPOrdGreaterThanEqual16, Opcode::FPOrdGreaterThanEqual32),
    (Opcode::FPUnordGreaterThanEqual16, Opcode::FPUnordGreaterThanEqual32),
    (Opcode::FPIsNan16, Opcode::FPIsNan32),
    (Opcode::ConvertS16F16, Opcode::ConvertS16F32),
    (Opcode::ConvertS32F16, Opcode::ConvertS32F32),
    (Opcode::ConvertS64F16, Opcode::ConvertS64F32),
    (Opcode::ConvertU16F16, Opcode::ConvertU16F32),
    (Opcode::ConvertU32F16, Opcode::ConvertU32F32),
    (Opcode::ConvertU64F16, Opcode::ConvertU64F32),
    (Opcode::ConvertF32F16, Opcode::Identity),
    (Opcode::ConvertF16F32, Opcode::Identity),
    (Opcode::ConvertF16S8, Opcode::ConvertF32S8),
    (Opcode::ConvertF16S16, Opcode::ConvertF32S16),
    (Opcode::ConvertF16S32, Opcode::ConvertF32S32),
    (Opcode::ConvertF16S64, Opcode::ConvertF32S64),
    (Opcode::ConvertF16U8, Opcode::ConvertF32U8),
    (Opcode::ConvertF16U16, Opcode::ConvertF32U16),
    (Opcode::ConvertF16U32, Opcode::ConvertF32U32),
    (Opcode::ConvertF16U64, Opcode::ConvertF32U64),
    (Opcode::GlobalAtomicAddF16x2, Opcode::GlobalAtomicAddF32x2),
    (Opcode::StorageAtomicAddF16x2, Opcode::StorageAtomicAddF32x2),
    (Opcode::GlobalAtomicMinF16x2, Opcode::GlobalAtomicMinF32x2),
    (Opcode::StorageAtomicMinF16x2, Opcode::StorageAtomicMinF32x2),
    (Opcode::GlobalAtomicMaxF16x2, Opcode::GlobalAtomicMaxF32x2),
    (Opcode::StorageAtomicMaxF16x2, Opcode::StorageAtomicMaxF32x2),
];

/// `SelectF16`, the addition the `fp16` module lists.
const F16_SELECT: (Opcode, Opcode) = (Opcode::SelectF16, Opcode::SelectF32);

/// The f16 pass's written-out operations: Eden's two renamed packings and the two
/// `Half2x16` ones (module docs).
const F16_EXPANSIONS: [(Opcode, Expansion); 4] = [
    (Opcode::PackFloat2x16, Expansion::PackHalves),
    (Opcode::PackHalf2x16, Expansion::PackHalves),
    (Opcode::UnpackFloat2x16, Expansion::UnpackHalves),
    (Opcode::UnpackHalf2x16, Expansion::UnpackHalves),
];

/// Each pass's table is exactly the literal rows above, from Eden's source: every listed row
/// maps as written, and no other opcode is renamed or written out.
#[test]
fn mappings_match_the_reference_literals() {
    let cases: [(HostPass, Vec<(Opcode, Opcode)>, &[(Opcode, Expansion)]); 2] = [
        (
            HostPass::Fp64ToFp32,
            F64_RENAMES.iter().copied().chain([F64_SELECT]).collect(),
            &F64_EXPANSIONS,
        ),
        (
            HostPass::Fp16ToFp32,
            F16_RENAMES.iter().copied().chain([F16_SELECT]).collect(),
            &F16_EXPANSIONS,
        ),
    ];
    for (pass, renames, expansions) in cases {
        for &(from, to) in &renames {
            assert_eq!(pass.rewrite(from), Rewrite::Rename(to), "{pass}: {from}");
        }
        for &(from, expansion) in expansions {
            assert_eq!(pass.rewrite(from), Rewrite::Expand(expansion), "{pass}: {from}");
        }
        let mapped = Opcode::ALL
            .iter()
            .filter(|&&opcode| pass.rewrite(opcode) != Rewrite::Keep)
            .count();
        assert_eq!(mapped, renames.len() + expansions.len(), "{pass}");
    }
}
