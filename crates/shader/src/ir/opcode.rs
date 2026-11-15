//! The operation set: every opcode with its result type, argument types and flags kind.
//!
//! The table is Eden's `eden/src/shader_recompiler/frontend/ir/opcodes.inc` @ cf654052
//! (GPL-3.0-or-later), row for row and in its order, with the comments that group it. Which
//! opcodes carry floating-point controls and which carry texture information (the last column)
//! comes from what Eden's emitter passes as flags (`frontend/ir/ir_emitter.cpp`: the calls with
//! `Flags{control}` and `Flags{info}`); in Eden the flags are an untyped 32-bit word, here they
//! are a typed [`Flags`](super::Flags) that the verifier checks against this column.
//!
//! Differences from Eden's table, pinned by the IR's `tests::opcode_table_differs_from_eden_as_listed`
//! (and the rows by `tests::opcode_rows_match_eden`):
//!
//! - `Phi` is not an opcode here but its own kind of instruction ([`Inst::Phi`](super::Inst)),
//!   because its arguments pair a block with a value; the text form still calls it `Phi`.
//! - `IsTextureScaled`, `IsImageScaled` and `ResolutionDownFactor` are left out: they serve only
//!   Eden's resolution-scaling pass, which the spec deliberately does not port ("Passes needed
//!   for correctness").
//! - `GetUniformRegister`, `SetUniformRegister`, `GetUniformPred` and `SetUniformPred` are
//!   added: the uniform register class (owner's decision 1, 2026-11-12), read and written as
//!   Eden reads and writes the per-thread registers before SSA. PR #4588 has no such opcodes
//!   because it packs uniform registers into the top of Maxwell's register file
//!   (`eden-pr4588/src/shader_recompiler/frontend/ampere/native.h`, `UniformLayout`).

use std::fmt::{self, Display, Formatter};

use super::Type;

/// Which [`Flags`](super::Flags) an opcode carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlagsKind {
    /// None.
    None,
    /// Floating-point controls ([`FpControl`](super::FpControl)).
    Fp,
    /// Texture information ([`TextureInfo`](super::TextureInfo)).
    Texture,
}

macro_rules! opcodes {
    ($(
        $(#[$meta:meta])*
        $name:ident => $result:ident ( $($arg:ident),* ) $flags:ident;
    )*) => {
        /// An operation (Eden's `IR::Opcode`).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Opcode {
            $(
                #[doc = concat!("Eden's `", stringify!($name), "`.")]
                $(#[$meta])*
                $name,
            )*
        }

        impl Opcode {
            /// Every opcode, in table order.
            pub const ALL: &'static [Self] = &[$(Self::$name),*];

            /// The opcode's name, Eden's, as the text form writes it.
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$name => stringify!($name),)*
                }
            }

            /// The type of the result; [`Type::Void`] for an operation with no result.
            pub const fn result(self) -> Type {
                match self {
                    $(Self::$name => Type::$result,)*
                }
            }

            /// The type each argument must have, in order.
            pub const fn args(self) -> &'static [Type] {
                match self {
                    $(Self::$name => &[$(Type::$arg),*],)*
                }
            }

            /// Which flags the operation carries.
            pub const fn flags_kind(self) -> FlagsKind {
                match self {
                    $(Self::$name => opcodes!(@flags $flags),)*
                }
            }
        }
    };
    (@flags none) => { FlagsKind::None };
    (@flags fp) => { FlagsKind::Fp };
    (@flags texture) => { FlagsKind::Texture };
}

impl Opcode {
    /// The opcode named `name` in the text form.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|op| op.name() == name)
    }
}

impl Display for Opcode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

opcodes! {
    Identity => Opaque(Opaque) none;
    Void => Void() none;
    ConditionRef => U1(U1) none;
    Reference => Void(Opaque) none;
    PhiMove => Void(Opaque, Opaque) none;

    // Special operations
    Prologue => Void() none;
    Epilogue => Void() none;
    Join => Void() none;
    DemoteToHelperInvocation => Void() none;
    EmitVertex => Void(U32) none;
    EndPrimitive => Void(U32) none;

    // Barriers
    Barrier => Void() none;
    WorkgroupMemoryBarrier => Void() none;
    DeviceMemoryBarrier => Void() none;

    // Context getters/setters
    GetRegister => U32(Reg) none;
    SetRegister => Void(Reg, U32) none;
    GetPred => U1(Pred) none;
    SetPred => Void(Pred, U1) none;
    // starquake: the uniform register class (owner's decision 1), not in Eden
    GetUniformRegister => U32(UniformReg) none;
    SetUniformRegister => Void(UniformReg, U32) none;
    GetUniformPred => U1(UniformPred) none;
    SetUniformPred => Void(UniformPred, U1) none;
    GetGotoVariable => U1(U32) none;
    SetGotoVariable => Void(U32, U1) none;
    GetIndirectBranchVariable => U32() none;
    SetIndirectBranchVariable => Void(U32) none;
    GetCbufU8 => U32(U32, U32) none;
    GetCbufS8 => U32(U32, U32) none;
    GetCbufU16 => U32(U32, U32) none;
    GetCbufS16 => U32(U32, U32) none;
    GetCbufU32 => U32(U32, U32) none;
    GetCbufF32 => F32(U32, U32) none;
    GetCbufU32x2 => U32x2(U32, U32) none;
    GetAttribute => F32(Attribute, U32) none;
    GetAttributeU32 => U32(Attribute, U32) none;
    SetAttribute => Void(Attribute, F32, U32) none;
    GetAttributeIndexed => F32(U32, U32) none;
    SetAttributeIndexed => Void(U32, F32, U32) none;
    GetPatch => F32(Patch) none;
    SetPatch => Void(Patch, F32) none;
    SetFragColor => Void(U32, U32, F32) none;
    SetSampleMask => Void(U32) none;
    SetFragDepth => Void(F32) none;
    GetZFlag => U1() none;
    GetSFlag => U1() none;
    GetCFlag => U1() none;
    GetOFlag => U1() none;
    SetZFlag => Void(U1) none;
    SetSFlag => Void(U1) none;
    SetCFlag => Void(U1) none;
    SetOFlag => Void(U1) none;
    WorkgroupId => U32x3() none;
    LocalInvocationId => U32x3() none;
    InvocationId => U32() none;
    InvocationInfo => U32() none;
    SampleId => U32() none;
    IsHelperInvocation => U1() none;
        #[expect(non_camel_case_types, reason = "Eden's opcode name, kept so the text form matches Eden's")]
    SR_WScaleFactorXY => U32() none;
        #[expect(non_camel_case_types, reason = "Eden's opcode name, kept so the text form matches Eden's")]
    SR_WScaleFactorZ => U32() none;
    YDirection => F32() none;
    RenderArea => F32x4() none;

    // Undefined
    UndefU1 => U1() none;
    UndefU8 => U8() none;
    UndefU16 => U16() none;
    UndefU32 => U32() none;
    UndefU64 => U64() none;

    // Memory operations
    LoadGlobalU8 => U32(Opaque) none;
    LoadGlobalS8 => U32(Opaque) none;
    LoadGlobalU16 => U32(Opaque) none;
    LoadGlobalS16 => U32(Opaque) none;
    LoadGlobal32 => U32(Opaque) none;
    LoadGlobal64 => U32x2(Opaque) none;
    LoadGlobal128 => U32x4(Opaque) none;
    WriteGlobalU8 => Void(Opaque, U32) none;
    WriteGlobalS8 => Void(Opaque, U32) none;
    WriteGlobalU16 => Void(Opaque, U32) none;
    WriteGlobalS16 => Void(Opaque, U32) none;
    WriteGlobal32 => Void(Opaque, U32) none;
    WriteGlobal64 => Void(Opaque, U32x2) none;
    WriteGlobal128 => Void(Opaque, U32x4) none;

    // Storage buffer operations
    LoadStorageU8 => U32(U32, U32) none;
    LoadStorageS8 => U32(U32, U32) none;
    LoadStorageU16 => U32(U32, U32) none;
    LoadStorageS16 => U32(U32, U32) none;
    LoadStorage32 => U32(U32, U32) none;
    LoadStorage64 => U32x2(U32, U32) none;
    LoadStorage128 => U32x4(U32, U32) none;
    WriteStorageU8 => Void(U32, U32, U32) none;
    WriteStorageS8 => Void(U32, U32, U32) none;
    WriteStorageU16 => Void(U32, U32, U32) none;
    WriteStorageS16 => Void(U32, U32, U32) none;
    WriteStorage32 => Void(U32, U32, U32) none;
    WriteStorage64 => Void(U32, U32, U32x2) none;
    WriteStorage128 => Void(U32, U32, U32x4) none;

    // Local memory operations
    LoadLocal => U32(U32) none;
    WriteLocal => Void(U32, U32) none;

    // Shared memory operations
    LoadSharedU8 => U32(U32) none;
    LoadSharedS8 => U32(U32) none;
    LoadSharedU16 => U32(U32) none;
    LoadSharedS16 => U32(U32) none;
    LoadSharedU32 => U32(U32) none;
    LoadSharedU64 => U32x2(U32) none;
    LoadSharedU128 => U32x4(U32) none;
    WriteSharedU8 => Void(U32, U32) none;
    WriteSharedU16 => Void(U32, U32) none;
    WriteSharedU32 => Void(U32, U32) none;
    WriteSharedU64 => Void(U32, U32x2) none;
    WriteSharedU128 => Void(U32, U32x4) none;

    // Vector utility
    CompositeConstructU32x2 => U32x2(U32, U32) none;
    CompositeConstructU32x3 => U32x3(U32, U32, U32) none;
    CompositeConstructU32x4 => U32x4(U32, U32, U32, U32) none;
    CompositeExtractU32x2 => U32(U32x2, U32) none;
    CompositeExtractU32x3 => U32(U32x3, U32) none;
    CompositeExtractU32x4 => U32(U32x4, U32) none;
    CompositeInsertU32x2 => U32x2(U32x2, U32, U32) none;
    CompositeInsertU32x3 => U32x3(U32x3, U32, U32) none;
    CompositeInsertU32x4 => U32x4(U32x4, U32, U32) none;
    CompositeConstructF16x2 => F16x2(F16, F16) none;
    CompositeConstructF16x3 => F16x3(F16, F16, F16) none;
    CompositeConstructF16x4 => F16x4(F16, F16, F16, F16) none;
    CompositeExtractF16x2 => F16(F16x2, U32) none;
    CompositeExtractF16x3 => F16(F16x3, U32) none;
    CompositeExtractF16x4 => F16(F16x4, U32) none;
    CompositeInsertF16x2 => F16x2(F16x2, F16, U32) none;
    CompositeInsertF16x3 => F16x3(F16x3, F16, U32) none;
    CompositeInsertF16x4 => F16x4(F16x4, F16, U32) none;
    CompositeConstructF32x2 => F32x2(F32, F32) none;
    CompositeConstructF32x3 => F32x3(F32, F32, F32) none;
    CompositeConstructF32x4 => F32x4(F32, F32, F32, F32) none;
    CompositeExtractF32x2 => F32(F32x2, U32) none;
    CompositeExtractF32x3 => F32(F32x3, U32) none;
    CompositeExtractF32x4 => F32(F32x4, U32) none;
    CompositeInsertF32x2 => F32x2(F32x2, F32, U32) none;
    CompositeInsertF32x3 => F32x3(F32x3, F32, U32) none;
    CompositeInsertF32x4 => F32x4(F32x4, F32, U32) none;
    CompositeConstructF64x2 => F64x2(F64, F64) none;
    CompositeConstructF64x3 => F64x3(F64, F64, F64) none;
    CompositeConstructF64x4 => F64x4(F64, F64, F64, F64) none;
    CompositeExtractF64x2 => F64(F64x2, U32) none;
    CompositeExtractF64x3 => F64(F64x3, U32) none;
    CompositeExtractF64x4 => F64(F64x4, U32) none;
    CompositeInsertF64x2 => F64x2(F64x2, F64, U32) none;
    CompositeInsertF64x3 => F64x3(F64x3, F64, U32) none;
    CompositeInsertF64x4 => F64x4(F64x4, F64, U32) none;

    // Select operations
    SelectU1 => U1(U1, U1, U1) none;
    SelectU8 => U8(U1, U8, U8) none;
    SelectU16 => U16(U1, U16, U16) none;
    SelectU32 => U32(U1, U32, U32) none;
    SelectU64 => U64(U1, U64, U64) none;
    SelectF16 => F16(U1, F16, F16) none;
    SelectF32 => F32(U1, F32, F32) none;
    SelectF64 => F64(U1, F64, F64) none;

    // Bitwise conversions
    BitCastU16F16 => U16(F16) none;
    BitCastU32F32 => U32(F32) none;
    BitCastU64F64 => U64(F64) none;
    BitCastF16U16 => F16(U16) none;
    BitCastF32U32 => F32(U32) none;
    BitCastF64U64 => F64(U64) none;
    PackUint2x32 => U64(U32x2) none;
    UnpackUint2x32 => U32x2(U64) none;
    PackFloat2x16 => U32(F16x2) none;
    UnpackFloat2x16 => F16x2(U32) none;
    PackHalf2x16 => U32(F32x2) none;
    UnpackHalf2x16 => F32x2(U32) none;
    PackDouble2x32 => F64(U32x2) none;
    UnpackDouble2x32 => U32x2(F64) none;

    // Pseudo-operation, handled specially at final emit
    GetZeroFromOp => U1(Opaque) none;
    GetSignFromOp => U1(Opaque) none;
    GetCarryFromOp => U1(Opaque) none;
    GetOverflowFromOp => U1(Opaque) none;
    GetSparseFromOp => U1(Opaque) none;
    GetInBoundsFromOp => U1(Opaque) none;

    // Floating-point operations
    FPAbs16 => F16(F16) none;
    FPAbs32 => F32(F32) none;
    FPAbs64 => F64(F64) none;
    FPAdd16 => F16(F16, F16) fp;
    FPAdd32 => F32(F32, F32) fp;
    FPAdd64 => F64(F64, F64) fp;
    FPFma16 => F16(F16, F16, F16) fp;
    FPFma32 => F32(F32, F32, F32) fp;
    FPFma64 => F64(F64, F64, F64) fp;
    FPMax32 => F32(F32, F32) fp;
    FPMax64 => F64(F64, F64) fp;
    FPMin32 => F32(F32, F32) fp;
    FPMin64 => F64(F64, F64) fp;
    FPMul16 => F16(F16, F16) fp;
    FPMul32 => F32(F32, F32) fp;
    FPMul64 => F64(F64, F64) fp;
    FPNeg16 => F16(F16) none;
    FPNeg32 => F32(F32) none;
    FPNeg64 => F64(F64) none;
    FPRecip32 => F32(F32) none;
    FPRecip64 => F64(F64) none;
    FPRecipSqrt32 => F32(F32) none;
    FPRecipSqrt64 => F64(F64) none;
    FPSqrt => F32(F32) none;
    FPSin => F32(F32) none;
    FPExp2 => F32(F32) none;
    FPCos => F32(F32) none;
    FPLog2 => F32(F32) none;
    FPSaturate16 => F16(F16) none;
    FPSaturate32 => F32(F32) none;
    FPSaturate64 => F64(F64) none;
    FPClamp16 => F16(F16, F16, F16) none;
    FPClamp32 => F32(F32, F32, F32) none;
    FPClamp64 => F64(F64, F64, F64) none;
    FPRoundEven16 => F16(F16) fp;
    FPRoundEven32 => F32(F32) fp;
    FPRoundEven64 => F64(F64) fp;
    FPFloor16 => F16(F16) fp;
    FPFloor32 => F32(F32) fp;
    FPFloor64 => F64(F64) fp;
    FPCeil16 => F16(F16) fp;
    FPCeil32 => F32(F32) fp;
    FPCeil64 => F64(F64) fp;
    FPTrunc16 => F16(F16) fp;
    FPTrunc32 => F32(F32) fp;
    FPTrunc64 => F64(F64) fp;

    FPOrdEqual16 => U1(F16, F16) fp;
    FPOrdEqual32 => U1(F32, F32) fp;
    FPOrdEqual64 => U1(F64, F64) fp;
    FPUnordEqual16 => U1(F16, F16) fp;
    FPUnordEqual32 => U1(F32, F32) fp;
    FPUnordEqual64 => U1(F64, F64) fp;
    FPOrdNotEqual16 => U1(F16, F16) fp;
    FPOrdNotEqual32 => U1(F32, F32) fp;
    FPOrdNotEqual64 => U1(F64, F64) fp;
    FPUnordNotEqual16 => U1(F16, F16) fp;
    FPUnordNotEqual32 => U1(F32, F32) fp;
    FPUnordNotEqual64 => U1(F64, F64) fp;
    FPOrdLessThan16 => U1(F16, F16) fp;
    FPOrdLessThan32 => U1(F32, F32) fp;
    FPOrdLessThan64 => U1(F64, F64) fp;
    FPUnordLessThan16 => U1(F16, F16) fp;
    FPUnordLessThan32 => U1(F32, F32) fp;
    FPUnordLessThan64 => U1(F64, F64) fp;
    FPOrdGreaterThan16 => U1(F16, F16) fp;
    FPOrdGreaterThan32 => U1(F32, F32) fp;
    FPOrdGreaterThan64 => U1(F64, F64) fp;
    FPUnordGreaterThan16 => U1(F16, F16) fp;
    FPUnordGreaterThan32 => U1(F32, F32) fp;
    FPUnordGreaterThan64 => U1(F64, F64) fp;
    FPOrdLessThanEqual16 => U1(F16, F16) fp;
    FPOrdLessThanEqual32 => U1(F32, F32) fp;
    FPOrdLessThanEqual64 => U1(F64, F64) fp;
    FPUnordLessThanEqual16 => U1(F16, F16) fp;
    FPUnordLessThanEqual32 => U1(F32, F32) fp;
    FPUnordLessThanEqual64 => U1(F64, F64) fp;
    FPOrdGreaterThanEqual16 => U1(F16, F16) fp;
    FPOrdGreaterThanEqual32 => U1(F32, F32) fp;
    FPOrdGreaterThanEqual64 => U1(F64, F64) fp;
    FPUnordGreaterThanEqual16 => U1(F16, F16) fp;
    FPUnordGreaterThanEqual32 => U1(F32, F32) fp;
    FPUnordGreaterThanEqual64 => U1(F64, F64) fp;
    FPIsNan16 => U1(F16) none;
    FPIsNan32 => U1(F32) none;
    FPIsNan64 => U1(F64) none;

    // Integer operations
    IAdd32 => U32(U32, U32) none;
    IAdd64 => U64(U64, U64) none;
    ISub32 => U32(U32, U32) none;
    ISub64 => U64(U64, U64) none;
    IMul32 => U32(U32, U32) none;
    SDiv32 => U32(U32, U32) none;
    UDiv32 => U32(U32, U32) none;
    INeg32 => U32(U32) none;
    INeg64 => U64(U64) none;
    IAbs32 => U32(U32) none;
    IAbs64 => U64(U64) none;
    ShiftLeftLogical32 => U32(U32, U32) none;
    ShiftLeftLogical64 => U64(U64, U32) none;
    ShiftRightLogical32 => U32(U32, U32) none;
    ShiftRightLogical64 => U64(U64, U32) none;
    ShiftRightArithmetic32 => U32(U32, U32) none;
    ShiftRightArithmetic64 => U64(U64, U32) none;
    BitwiseAnd32 => U32(U32, U32) none;
    BitwiseOr32 => U32(U32, U32) none;
    BitwiseXor32 => U32(U32, U32) none;
    BitFieldInsert => U32(U32, U32, U32, U32) none;
    BitFieldSExtract => U32(U32, U32, U32) none;
    BitFieldUExtract => U32(U32, U32, U32) none;
    BitReverse32 => U32(U32) none;
    BitCount32 => U32(U32) none;
    BitwiseNot32 => U32(U32) none;

    FindSMsb32 => U32(U32) none;
    FindUMsb32 => U32(U32) none;
    SMin32 => U32(U32, U32) none;
    UMin32 => U32(U32, U32) none;
    SMax32 => U32(U32, U32) none;
    UMax32 => U32(U32, U32) none;
    SClamp32 => U32(U32, U32, U32) none;
    UClamp32 => U32(U32, U32, U32) none;
    SLessThan => U1(U32, U32) none;
    ULessThan => U1(U32, U32) none;
    IEqual => U1(U32, U32) none;
    SLessThanEqual => U1(U32, U32) none;
    ULessThanEqual => U1(U32, U32) none;
    SGreaterThan => U1(U32, U32) none;
    UGreaterThan => U1(U32, U32) none;
    INotEqual => U1(U32, U32) none;
    SGreaterThanEqual => U1(U32, U32) none;
    UGreaterThanEqual => U1(U32, U32) none;

    // Atomic operations
    SharedAtomicIAdd32 => U32(U32, U32) none;
    SharedAtomicSMin32 => U32(U32, U32) none;
    SharedAtomicUMin32 => U32(U32, U32) none;
    SharedAtomicSMax32 => U32(U32, U32) none;
    SharedAtomicUMax32 => U32(U32, U32) none;
    SharedAtomicInc32 => U32(U32, U32) none;
    SharedAtomicDec32 => U32(U32, U32) none;
    SharedAtomicAnd32 => U32(U32, U32) none;
    SharedAtomicOr32 => U32(U32, U32) none;
    SharedAtomicXor32 => U32(U32, U32) none;
    SharedAtomicExchange32 => U32(U32, U32) none;
    SharedAtomicExchange64 => U64(U32, U64) none;
    SharedAtomicExchange32x2 => U32x2(U32, U32x2) none;

    GlobalAtomicIAdd32 => U32(U64, U32) none;
    GlobalAtomicSMin32 => U32(U64, U32) none;
    GlobalAtomicUMin32 => U32(U64, U32) none;
    GlobalAtomicSMax32 => U32(U64, U32) none;
    GlobalAtomicUMax32 => U32(U64, U32) none;
    GlobalAtomicInc32 => U32(U64, U32) none;
    GlobalAtomicDec32 => U32(U64, U32) none;
    GlobalAtomicAnd32 => U32(U64, U32) none;
    GlobalAtomicOr32 => U32(U64, U32) none;
    GlobalAtomicXor32 => U32(U64, U32) none;
    GlobalAtomicExchange32 => U32(U64, U32) none;
    GlobalAtomicIAdd64 => U64(U64, U64) none;
    GlobalAtomicSMin64 => U64(U64, U64) none;
    GlobalAtomicUMin64 => U64(U64, U64) none;
    GlobalAtomicSMax64 => U64(U64, U64) none;
    GlobalAtomicUMax64 => U64(U64, U64) none;
    GlobalAtomicAnd64 => U64(U64, U64) none;
    GlobalAtomicOr64 => U64(U64, U64) none;
    GlobalAtomicXor64 => U64(U64, U64) none;
    GlobalAtomicExchange64 => U64(U64, U64) none;
    GlobalAtomicIAdd32x2 => U32x2(U32x2, U32x2) none;
    GlobalAtomicSMin32x2 => U32x2(U32x2, U32x2) none;
    GlobalAtomicUMin32x2 => U32x2(U32x2, U32x2) none;
    GlobalAtomicSMax32x2 => U32x2(U32x2, U32x2) none;
    GlobalAtomicUMax32x2 => U32x2(U32x2, U32x2) none;
    GlobalAtomicAnd32x2 => U32x2(U32x2, U32x2) none;
    GlobalAtomicOr32x2 => U32x2(U32x2, U32x2) none;
    GlobalAtomicXor32x2 => U32x2(U32x2, U32x2) none;
    GlobalAtomicExchange32x2 => U32x2(U32x2, U32x2) none;
    GlobalAtomicAddF32 => F32(U64, F32) fp;
    GlobalAtomicAddF16x2 => U32(U64, F16x2) fp;
    GlobalAtomicAddF32x2 => U32(U64, F32x2) none;
    GlobalAtomicMinF16x2 => U32(U64, F16x2) fp;
    GlobalAtomicMinF32x2 => U32(U64, F32x2) none;
    GlobalAtomicMaxF16x2 => U32(U64, F16x2) fp;
    GlobalAtomicMaxF32x2 => U32(U64, F32x2) none;

    StorageAtomicIAdd32 => U32(U32, U32, U32) none;
    StorageAtomicSMin32 => U32(U32, U32, U32) none;
    StorageAtomicUMin32 => U32(U32, U32, U32) none;
    StorageAtomicSMax32 => U32(U32, U32, U32) none;
    StorageAtomicUMax32 => U32(U32, U32, U32) none;
    StorageAtomicInc32 => U32(U32, U32, U32) none;
    StorageAtomicDec32 => U32(U32, U32, U32) none;
    StorageAtomicAnd32 => U32(U32, U32, U32) none;
    StorageAtomicOr32 => U32(U32, U32, U32) none;
    StorageAtomicXor32 => U32(U32, U32, U32) none;
    StorageAtomicExchange32 => U32(U32, U32, U32) none;
    StorageAtomicIAdd64 => U64(U32, U32, U64) none;
    StorageAtomicSMin64 => U64(U32, U32, U64) none;
    StorageAtomicUMin64 => U64(U32, U32, U64) none;
    StorageAtomicSMax64 => U64(U32, U32, U64) none;
    StorageAtomicUMax64 => U64(U32, U32, U64) none;
    StorageAtomicAnd64 => U64(U32, U32, U64) none;
    StorageAtomicOr64 => U64(U32, U32, U64) none;
    StorageAtomicXor64 => U64(U32, U32, U64) none;
    StorageAtomicExchange64 => U64(U32, U32, U64) none;
    StorageAtomicIAdd32x2 => U32x2(U32, U32, U32x2) none;
    StorageAtomicSMin32x2 => U32x2(U32, U32, U32x2) none;
    StorageAtomicUMin32x2 => U32x2(U32, U32, U32x2) none;
    StorageAtomicSMax32x2 => U32x2(U32, U32, U32x2) none;
    StorageAtomicUMax32x2 => U32x2(U32, U32, U32x2) none;
    StorageAtomicAnd32x2 => U32x2(U32, U32, U32x2) none;
    StorageAtomicOr32x2 => U32x2(U32, U32, U32x2) none;
    StorageAtomicXor32x2 => U32x2(U32, U32, U32x2) none;
    StorageAtomicExchange32x2 => U32x2(U32, U32, U32x2) none;
    StorageAtomicAddF32 => F32(U32, U32, F32) none;
    StorageAtomicAddF16x2 => U32(U32, U32, F16x2) none;
    StorageAtomicAddF32x2 => U32(U32, U32, F32x2) none;
    StorageAtomicMinF16x2 => U32(U32, U32, F16x2) none;
    StorageAtomicMinF32x2 => U32(U32, U32, F32x2) none;
    StorageAtomicMaxF16x2 => U32(U32, U32, F16x2) none;
    StorageAtomicMaxF32x2 => U32(U32, U32, F32x2) none;

    // Logical operations
    LogicalOr => U1(U1, U1) none;
    LogicalAnd => U1(U1, U1) none;
    LogicalXor => U1(U1, U1) none;
    LogicalNot => U1(U1) none;

    // Conversion operations
    ConvertS16F16 => U32(F16) none;
    ConvertS16F32 => U32(F32) none;
    ConvertS16F64 => U32(F64) none;
    ConvertS32F16 => U32(F16) none;
    ConvertS32F32 => U32(F32) none;
    ConvertS32F64 => U32(F64) none;
    ConvertS64F16 => U64(F16) none;
    ConvertS64F32 => U64(F32) none;
    ConvertS64F64 => U64(F64) none;
    ConvertU16F16 => U32(F16) none;
    ConvertU16F32 => U32(F32) none;
    ConvertU16F64 => U32(F64) none;
    ConvertU32F16 => U32(F16) none;
    ConvertU32F32 => U32(F32) none;
    ConvertU32F64 => U32(F64) none;
    ConvertU64F16 => U64(F16) none;
    ConvertU64F32 => U64(F32) none;
    ConvertU64F64 => U64(F64) none;
    ConvertU64U32 => U64(U32) none;
    ConvertU32U64 => U32(U64) none;
    ConvertF16F32 => F16(F32) fp;
    ConvertF32F16 => F32(F16) fp;
    ConvertF32F64 => F32(F64) fp;
    ConvertF64F32 => F64(F32) fp;
    ConvertF16S8 => F16(U32) fp;
    ConvertF16S16 => F16(U32) fp;
    ConvertF16S32 => F16(U32) fp;
    ConvertF16S64 => F16(U64) fp;
    ConvertF16U8 => F16(U32) fp;
    ConvertF16U16 => F16(U32) fp;
    ConvertF16U32 => F16(U32) fp;
    ConvertF16U64 => F16(U64) fp;
    ConvertF32S8 => F32(U32) fp;
    ConvertF32S16 => F32(U32) fp;
    ConvertF32S32 => F32(U32) fp;
    ConvertF32S64 => F32(U64) fp;
    ConvertF32U8 => F32(U32) fp;
    ConvertF32U16 => F32(U32) fp;
    ConvertF32U32 => F32(U32) fp;
    ConvertF32U64 => F32(U64) fp;
    ConvertF64S8 => F64(U32) fp;
    ConvertF64S16 => F64(U32) fp;
    ConvertF64S32 => F64(U32) fp;
    ConvertF64S64 => F64(U64) fp;
    ConvertF64U8 => F64(U32) fp;
    ConvertF64U16 => F64(U32) fp;
    ConvertF64U32 => F64(U32) fp;
    ConvertF64U64 => F64(U64) fp;
    ConvertU16U32 => U16(U32) none;
    ConvertU32U16 => U32(U16) none;
    ConvertU8U32 => U8(U32) none;
    ConvertU32U8 => U32(U8) none;
    ConvertS32S8 => U32(U8) none;
    ConvertS32S16 => U32(U16) none;

    // Image operations
    BindlessImageSampleImplicitLod => F32x4(U32, Opaque, Opaque, Opaque) texture;
    BindlessImageSampleExplicitLod => F32x4(U32, Opaque, Opaque, Opaque) texture;
    BindlessImageSampleDrefImplicitLod => F32(U32, Opaque, F32, Opaque, Opaque) texture;
    BindlessImageSampleDrefExplicitLod => F32(U32, Opaque, F32, Opaque, Opaque) texture;
    BindlessImageGather => F32x4(U32, Opaque, Opaque, Opaque) texture;
    BindlessImageGatherDref => F32x4(U32, Opaque, Opaque, Opaque, F32) texture;
    BindlessImageFetch => F32x4(U32, Opaque, Opaque, U32, Opaque) texture;
    BindlessImageQueryDimensions => U32x4(U32, U32, U1) texture;
    BindlessImageQueryLod => F32x4(U32, Opaque) texture;
    BindlessImageGradient => F32x4(U32, Opaque, Opaque, Opaque, Opaque) texture;
    BindlessImageRead => U32x4(U32, Opaque) texture;
    BindlessImageWrite => Void(U32, Opaque, U32x4) texture;

    BoundImageSampleImplicitLod => F32x4(U32, Opaque, Opaque, Opaque) texture;
    BoundImageSampleExplicitLod => F32x4(U32, Opaque, Opaque, Opaque) texture;
    BoundImageSampleDrefImplicitLod => F32(U32, Opaque, F32, Opaque, Opaque) texture;
    BoundImageSampleDrefExplicitLod => F32(U32, Opaque, F32, Opaque, Opaque) texture;
    BoundImageGather => F32x4(U32, Opaque, Opaque, Opaque) texture;
    BoundImageGatherDref => F32x4(U32, Opaque, Opaque, Opaque, F32) texture;
    BoundImageFetch => F32x4(U32, Opaque, Opaque, U32, Opaque) texture;
    BoundImageQueryDimensions => U32x4(U32, U32, U1) texture;
    BoundImageQueryLod => F32x4(U32, Opaque) texture;
    BoundImageGradient => F32x4(U32, Opaque, Opaque, Opaque, Opaque) texture;
    BoundImageRead => U32x4(U32, Opaque) texture;
    BoundImageWrite => Void(U32, Opaque, U32x4) texture;

    ImageSampleImplicitLod => F32x4(Opaque, Opaque, Opaque, Opaque) texture;
    ImageSampleExplicitLod => F32x4(Opaque, Opaque, Opaque, Opaque) texture;
    ImageSampleDrefImplicitLod => F32(Opaque, Opaque, F32, Opaque, Opaque) texture;
    ImageSampleDrefExplicitLod => F32(Opaque, Opaque, F32, Opaque, Opaque) texture;
    ImageGather => F32x4(Opaque, Opaque, Opaque, Opaque) texture;
    ImageGatherDref => F32x4(Opaque, Opaque, Opaque, Opaque, F32) texture;
    ImageFetch => F32x4(Opaque, Opaque, Opaque, U32, Opaque) texture;
    ImageQueryDimensions => U32x4(Opaque, U32, U1) texture;
    ImageQueryLod => F32x4(Opaque, Opaque) texture;
    ImageGradient => F32x4(Opaque, Opaque, Opaque, Opaque, Opaque) texture;
    ImageRead => U32x4(Opaque, Opaque) texture;
    ImageWrite => Void(Opaque, Opaque, U32x4) texture;

    // Atomic Image operations

    BindlessImageAtomicIAdd32 => U32(U32, Opaque, U32) texture;
    BindlessImageAtomicSMin32 => U32(U32, Opaque, U32) texture;
    BindlessImageAtomicUMin32 => U32(U32, Opaque, U32) texture;
    BindlessImageAtomicSMax32 => U32(U32, Opaque, U32) texture;
    BindlessImageAtomicUMax32 => U32(U32, Opaque, U32) texture;
    BindlessImageAtomicInc32 => U32(U32, Opaque, U32) texture;
    BindlessImageAtomicDec32 => U32(U32, Opaque, U32) texture;
    BindlessImageAtomicAnd32 => U32(U32, Opaque, U32) texture;
    BindlessImageAtomicOr32 => U32(U32, Opaque, U32) texture;
    BindlessImageAtomicXor32 => U32(U32, Opaque, U32) texture;
    BindlessImageAtomicExchange32 => U32(U32, Opaque, U32) texture;

    BoundImageAtomicIAdd32 => U32(U32, Opaque, U32) texture;
    BoundImageAtomicSMin32 => U32(U32, Opaque, U32) texture;
    BoundImageAtomicUMin32 => U32(U32, Opaque, U32) texture;
    BoundImageAtomicSMax32 => U32(U32, Opaque, U32) texture;
    BoundImageAtomicUMax32 => U32(U32, Opaque, U32) texture;
    BoundImageAtomicInc32 => U32(U32, Opaque, U32) texture;
    BoundImageAtomicDec32 => U32(U32, Opaque, U32) texture;
    BoundImageAtomicAnd32 => U32(U32, Opaque, U32) texture;
    BoundImageAtomicOr32 => U32(U32, Opaque, U32) texture;
    BoundImageAtomicXor32 => U32(U32, Opaque, U32) texture;
    BoundImageAtomicExchange32 => U32(U32, Opaque, U32) texture;

    ImageAtomicIAdd32 => U32(Opaque, Opaque, U32) texture;
    ImageAtomicSMin32 => U32(Opaque, Opaque, U32) texture;
    ImageAtomicUMin32 => U32(Opaque, Opaque, U32) texture;
    ImageAtomicSMax32 => U32(Opaque, Opaque, U32) texture;
    ImageAtomicUMax32 => U32(Opaque, Opaque, U32) texture;
    ImageAtomicInc32 => U32(Opaque, Opaque, U32) texture;
    ImageAtomicDec32 => U32(Opaque, Opaque, U32) texture;
    ImageAtomicAnd32 => U32(Opaque, Opaque, U32) texture;
    ImageAtomicOr32 => U32(Opaque, Opaque, U32) texture;
    ImageAtomicXor32 => U32(Opaque, Opaque, U32) texture;
    ImageAtomicExchange32 => U32(Opaque, Opaque, U32) texture;

    // Warp operations
    LaneId => U32() none;
    VoteAll => U1(U1) none;
    VoteAny => U1(U1) none;
    VoteEqual => U1(U1) none;
    SubgroupBallot => U32(U1) none;
    SubgroupEqMask => U32() none;
    SubgroupLtMask => U32() none;
    SubgroupLeMask => U32() none;
    SubgroupGtMask => U32() none;
    SubgroupGeMask => U32() none;
    ShuffleIndex => U32(U32, U32, U32, U32) none;
    ShuffleUp => U32(U32, U32, U32, U32) none;
    ShuffleDown => U32(U32, U32, U32, U32) none;
    ShuffleButterfly => U32(U32, U32, U32, U32) none;
    QuadBroadcast => U32(U32, U32) none;
    QuadSwap => U32(U32, U32) none;
    FSwizzleAdd => F32(F32, F32, U32) fp;
    DPdxFine => F32(F32) none;
    DPdyFine => F32(F32) none;
    DPdxCoarse => F32(F32) none;
    DPdyCoarse => F32(F32) none;
}
