//! The operations: which are lowered, their expression trees at the edges the interpreter
//! defines, and their results on the host's GPU against the interpreter's.

use super::run::{gpu_registers, interp_registers, opaque};
use super::*;
use crate::decode::{Pred, UniformPred, UniformReg};
use crate::ir::{Builder, Flags, FpControl, FpRounding};

/// One case of the GPU comparison: a name and the instructions that compute a `u32`.
type Case = (&'static str, fn(&mut Builder<'_>) -> Value);

pub(super) fn u(value: u32) -> Operand {
    Operand::Imm(Imm::U32(value))
}

pub(super) fn f(value: f32) -> Operand {
    Operand::Imm(Imm::F32(value.to_bits()))
}

pub(super) fn bit(value: bool) -> Operand {
    Operand::Imm(Imm::U1(value))
}

pub(super) fn op(builder: &mut Builder<'_>, opcode: Opcode, args: &[Operand]) -> Value {
    builder
        .op(opcode, args)
        .expect("the case's instruction builds")
}

/// `opcode` of `args`, whose result is a `u32`.
fn int(builder: &mut Builder<'_>, opcode: Opcode, args: &[Operand]) -> Value {
    op(builder, opcode, args)
}

/// `opcode` of `args`, whose result is an `f32`, as its bits.
fn float(builder: &mut Builder<'_>, opcode: Opcode, args: &[Operand]) -> Value {
    let value = op(builder, opcode, args);
    op(builder, Opcode::BitCastU32F32, &[value.into()])
}

/// `opcode` of `args`, whose result is a `u1`, as 1 or 0.
fn boolean(builder: &mut Builder<'_>, opcode: Opcode, args: &[Operand]) -> Value {
    let value = op(builder, opcode, args);
    op(builder, Opcode::SelectU32, &[value.into(), u(1), u(0)])
}

/// A function that writes case `n`'s value to register `n`, and the registers.
fn cases_function(cases: &[Case]) -> (Function, Vec<u8>) {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let mut regs = Vec::new();
    for (index, (_, case)) in cases.iter().enumerate() {
        let reg = u8::try_from(index).expect("fewer cases than registers");
        let value = case(&mut builder);
        op(
            &mut builder,
            Opcode::SetRegister,
            &[Operand::Imm(Imm::Reg(Gpr::new(reg))), value.into()],
        );
        regs.push(reg);
    }
    builder
        .terminate(Terminator::Return)
        .expect("the block returns");
    (function, regs)
}

/// The compute inputs of a workgroup of one.
const ONE_INVOCATION: LowerInputs = LowerInputs {
    workgroup_size: [1, 1, 1],
    ..LowerInputs::DEFAULT
};

/// Lowers `cases` as a compute shader, checks both validations, and, where the host can run
/// it, compares each register the GPU leaves with the interpreter's; returns the names of the
/// cases that differ, with both values.
fn gpu_mismatches(test: &str, cases: &[Case]) -> Vec<String> {
    let (function, regs) = cases_function(cases);
    let function = opaque(&function);
    let shader = lowered(GuestStage::Compute, &function, &ONE_INVOCATION);
    let expected = interp_registers(&function, &regs);
    let Some(found) = gpu_registers(test, &shader, &regs) else {
        return Vec::new();
    };
    cases
        .iter()
        .zip(expected.iter().zip(&found))
        .filter(|(_, (expected, found))| expected != found)
        .map(|((name, _), (expected, found))| {
            format!("{name}: interpreter {expected:#x}, GPU {found:#x}")
        })
        .collect()
}

const MIN: u32 = 0x8000_0000;
const NAN: f32 = f32::NAN;

/// 32-bit integer cases, at the edges the interpreter defines.
const INTEGER_CASES: &[Case] = &[
    ("iadd wraps", |b| {
        int(b, Opcode::IAdd32, &[u(u32::MAX), u(2)])
    }),
    ("isub wraps", |b| int(b, Opcode::ISub32, &[u(1), u(2)])),
    ("imul wraps", |b| {
        int(b, Opcode::IMul32, &[u(0x1_0001), u(0x1_0001)])
    }),
    ("ineg", |b| int(b, Opcode::INeg32, &[u(5)])),
    ("iabs min", |b| int(b, Opcode::IAbs32, &[u(MIN)])),
    ("iabs negative", |b| {
        int(b, Opcode::IAbs32, &[u((-5_i32).cast_unsigned())])
    }),
    ("shl 31", |b| {
        int(b, Opcode::ShiftLeftLogical32, &[u(0x8000_0001), u(31)])
    }),
    ("shl 32", |b| {
        int(b, Opcode::ShiftLeftLogical32, &[u(0x8000_0001), u(32)])
    }),
    ("shl 33", |b| {
        int(b, Opcode::ShiftLeftLogical32, &[u(0x8000_0001), u(33)])
    }),
    ("shl max", |b| {
        int(b, Opcode::ShiftLeftLogical32, &[u(1), u(u32::MAX)])
    }),
    ("shr 31", |b| {
        int(b, Opcode::ShiftRightLogical32, &[u(0x8000_0001), u(31)])
    }),
    ("shr 32", |b| {
        int(b, Opcode::ShiftRightLogical32, &[u(0x8000_0001), u(32)])
    }),
    ("sar 31", |b| {
        int(b, Opcode::ShiftRightArithmetic32, &[u(MIN), u(31)])
    }),
    ("sar 32", |b| {
        int(b, Opcode::ShiftRightArithmetic32, &[u(MIN), u(32)])
    }),
    ("sar 100 positive", |b| {
        int(b, Opcode::ShiftRightArithmetic32, &[u(0x4000_0000), u(100)])
    }),
    ("udiv by zero", |b| int(b, Opcode::UDiv32, &[u(7), u(0)])),
    ("udiv", |b| int(b, Opcode::UDiv32, &[u(7), u(2)])),
    ("sdiv by zero", |b| int(b, Opcode::SDiv32, &[u(7), u(0)])),
    ("sdiv min by -1", |b| {
        int(b, Opcode::SDiv32, &[u(MIN), u(u32::MAX)])
    }),
    ("sdiv negative", |b| {
        int(b, Opcode::SDiv32, &[u((-7_i32).cast_unsigned()), u(2)])
    }),
    ("and", |b| {
        int(b, Opcode::BitwiseAnd32, &[u(0xff00), u(0x0ff0)])
    }),
    ("or", |b| {
        int(b, Opcode::BitwiseOr32, &[u(0xff00), u(0x0ff0)])
    }),
    ("xor", |b| {
        int(b, Opcode::BitwiseXor32, &[u(0xff00), u(0x0ff0)])
    }),
    ("not", |b| int(b, Opcode::BitwiseNot32, &[u(0xff00)])),
    ("uextract", |b| {
        int(b, Opcode::BitFieldUExtract, &[u(0xf0f0_f0f0), u(4), u(8)])
    }),
    ("uextract past bit 31", |b| {
        int(b, Opcode::BitFieldUExtract, &[u(0xf000_0000), u(30), u(4)])
    }),
    ("uextract offset 32", |b| {
        int(b, Opcode::BitFieldUExtract, &[u(u32::MAX), u(32), u(4)])
    }),
    ("uextract count 33", |b| {
        int(b, Opcode::BitFieldUExtract, &[u(0x1234_5678), u(0), u(33)])
    }),
    ("uextract count 0", |b| {
        int(b, Opcode::BitFieldUExtract, &[u(u32::MAX), u(4), u(0)])
    }),
    ("sextract top bit outside", |b| {
        int(b, Opcode::BitFieldSExtract, &[u(MIN), u(31), u(4)])
    }),
    ("sextract negative", |b| {
        int(b, Opcode::BitFieldSExtract, &[u(0xf0), u(4), u(4)])
    }),
    ("sextract count 0", |b| {
        int(b, Opcode::BitFieldSExtract, &[u(u32::MAX), u(0), u(0)])
    }),
    ("sextract count 32", |b| {
        int(b, Opcode::BitFieldSExtract, &[u(MIN), u(0), u(32)])
    }),
    ("sextract count 40", |b| {
        int(b, Opcode::BitFieldSExtract, &[u(0xf0), u(4), u(40)])
    }),
    ("insert", |b| {
        int(b, Opcode::BitFieldInsert, &[u(u32::MAX), u(0), u(4), u(8)])
    }),
    ("insert past bit 31", |b| {
        int(b, Opcode::BitFieldInsert, &[u(0), u(0xff), u(28), u(8)])
    }),
    ("insert offset 32", |b| {
        int(b, Opcode::BitFieldInsert, &[u(0), u(0xff), u(32), u(8)])
    }),
    ("insert count 40", |b| {
        int(b, Opcode::BitFieldInsert, &[u(0), u(u32::MAX), u(0), u(40)])
    }),
    ("reverse", |b| int(b, Opcode::BitReverse32, &[u(0x1)])),
    ("count", |b| int(b, Opcode::BitCount32, &[u(0xf0f0)])),
    ("umsb 0", |b| int(b, Opcode::FindUMsb32, &[u(0)])),
    ("umsb top", |b| int(b, Opcode::FindUMsb32, &[u(MIN)])),
    ("smsb 0", |b| int(b, Opcode::FindSMsb32, &[u(0)])),
    ("smsb -1", |b| int(b, Opcode::FindSMsb32, &[u(u32::MAX)])),
    ("smsb min", |b| int(b, Opcode::FindSMsb32, &[u(MIN)])),
    ("smsb 5", |b| int(b, Opcode::FindSMsb32, &[u(5)])),
    ("smin", |b| int(b, Opcode::SMin32, &[u(u32::MAX), u(1)])),
    ("smax", |b| int(b, Opcode::SMax32, &[u(u32::MAX), u(1)])),
    ("umin", |b| int(b, Opcode::UMin32, &[u(u32::MAX), u(1)])),
    ("umax", |b| int(b, Opcode::UMax32, &[u(u32::MAX), u(1)])),
    ("sclamp", |b| {
        let value = (-5_i32).cast_unsigned();
        int(
            b,
            Opcode::SClamp32,
            &[u(value), u((-3_i32).cast_unsigned()), u(10)],
        )
    }),
    ("uclamp low above high", |b| {
        int(b, Opcode::UClamp32, &[u(5), u(10), u(3)])
    }),
    ("slt", |b| {
        boolean(b, Opcode::SLessThan, &[u(u32::MAX), u(1)])
    }),
    ("ult", |b| {
        boolean(b, Opcode::ULessThan, &[u(u32::MAX), u(1)])
    }),
    ("sle equal", |b| {
        boolean(b, Opcode::SLessThanEqual, &[u(MIN), u(MIN)])
    }),
    ("sgt", |b| {
        boolean(b, Opcode::SGreaterThan, &[u(1), u(u32::MAX)])
    }),
    ("uge", |b| {
        boolean(b, Opcode::UGreaterThanEqual, &[u(1), u(u32::MAX)])
    }),
    ("ule", |b| {
        boolean(b, Opcode::ULessThanEqual, &[u(u32::MAX), u(1)])
    }),
    ("sge", |b| {
        boolean(b, Opcode::SGreaterThanEqual, &[u(u32::MAX), u(1)])
    }),
    ("ugt", |b| {
        boolean(b, Opcode::UGreaterThan, &[u(u32::MAX), u(1)])
    }),
    ("ieq", |b| boolean(b, Opcode::IEqual, &[u(3), u(3)])),
    ("ine", |b| boolean(b, Opcode::INotEqual, &[u(3), u(3)])),
    ("logical", |b| {
        let or = op(b, Opcode::LogicalOr, &[bit(false), bit(true)]);
        let xor = op(b, Opcode::LogicalXor, &[or.into(), bit(true)]);
        let not = op(b, Opcode::LogicalNot, &[xor.into()]);
        let and = op(b, Opcode::LogicalAnd, &[not.into(), or.into()]);
        let select = op(b, Opcode::SelectU1, &[and.into(), xor.into(), or.into()]);
        op(b, Opcode::SelectU32, &[select.into(), u(1), u(2)])
    }),
];

/// 32-bit float cases, NaNs included.
const FLOAT_CASES: &[Case] = &[
    ("fadd", |b| float(b, Opcode::FPAdd32, &[f(1.5), f(2.25)])),
    ("fmul", |b| float(b, Opcode::FPMul32, &[f(1.5), f(-2.0)])),
    ("ffma", |b| {
        float(b, Opcode::FPFma32, &[f(2.0), f(3.0), f(1.0)])
    }),
    ("fneg", |b| float(b, Opcode::FPNeg32, &[f(2.0)])),
    ("fabs", |b| float(b, Opcode::FPAbs32, &[f(-2.0)])),
    ("fmax", |b| float(b, Opcode::FPMax32, &[f(1.0), f(2.0)])),
    ("fmin", |b| float(b, Opcode::FPMin32, &[f(1.0), f(2.0)])),
    ("fmax nan first", |b| {
        float(b, Opcode::FPMax32, &[f(NAN), f(1.0)])
    }),
    ("fmin nan second", |b| {
        float(b, Opcode::FPMin32, &[f(1.0), f(NAN)])
    }),
    ("fclamp", |b| {
        float(b, Opcode::FPClamp32, &[f(5.0), f(0.0), f(1.0)])
    }),
    ("fclamp nan", |b| {
        float(b, Opcode::FPClamp32, &[f(NAN), f(0.25), f(1.0)])
    }),
    ("fsaturate", |b| float(b, Opcode::FPSaturate32, &[f(2.0)])),
    ("frecip", |b| float(b, Opcode::FPRecip32, &[f(2.0)])),
    ("frecip zero", |b| float(b, Opcode::FPRecip32, &[f(0.0)])),
    ("fsqrt", |b| float(b, Opcode::FPSqrt, &[f(4.0)])),
    ("frsqrt", |b| float(b, Opcode::FPRecipSqrt32, &[f(4.0)])),
    ("fexp2", |b| float(b, Opcode::FPExp2, &[f(3.0)])),
    ("flog2", |b| float(b, Opcode::FPLog2, &[f(8.0)])),
    ("fsin 0", |b| float(b, Opcode::FPSin, &[f(0.0)])),
    ("fcos 0", |b| float(b, Opcode::FPCos, &[f(0.0)])),
    ("round half even down", |b| {
        float(b, Opcode::FPRoundEven32, &[f(2.5)])
    }),
    ("round half even up", |b| {
        float(b, Opcode::FPRoundEven32, &[f(3.5)])
    }),
    ("floor", |b| float(b, Opcode::FPFloor32, &[f(-1.5)])),
    ("ceil", |b| float(b, Opcode::FPCeil32, &[f(-1.5)])),
    ("trunc", |b| float(b, Opcode::FPTrunc32, &[f(-1.5)])),
    ("ord eq", |b| {
        boolean(b, Opcode::FPOrdEqual32, &[f(1.0), f(1.0)])
    }),
    ("ord eq nan", |b| {
        boolean(b, Opcode::FPOrdEqual32, &[f(NAN), f(NAN)])
    }),
    ("unord eq nan", |b| {
        boolean(b, Opcode::FPUnordEqual32, &[f(NAN), f(1.0)])
    }),
    ("ord ne nan", |b| {
        boolean(b, Opcode::FPOrdNotEqual32, &[f(NAN), f(1.0)])
    }),
    ("ord ne", |b| {
        boolean(b, Opcode::FPOrdNotEqual32, &[f(2.0), f(1.0)])
    }),
    ("unord ne nan", |b| {
        boolean(b, Opcode::FPUnordNotEqual32, &[f(NAN), f(1.0)])
    }),
    ("unord ne equal", |b| {
        boolean(b, Opcode::FPUnordNotEqual32, &[f(1.0), f(1.0)])
    }),
    ("ord lt", |b| {
        boolean(b, Opcode::FPOrdLessThan32, &[f(1.0), f(2.0)])
    }),
    ("ord lt nan", |b| {
        boolean(b, Opcode::FPOrdLessThan32, &[f(NAN), f(2.0)])
    }),
    ("unord lt nan", |b| {
        boolean(b, Opcode::FPUnordLessThan32, &[f(NAN), f(2.0)])
    }),
    ("unord lt false", |b| {
        boolean(b, Opcode::FPUnordLessThan32, &[f(3.0), f(2.0)])
    }),
    ("unord gt nan", |b| {
        boolean(b, Opcode::FPUnordGreaterThan32, &[f(1.0), f(NAN)])
    }),
    ("ord le", |b| {
        boolean(b, Opcode::FPOrdLessThanEqual32, &[f(2.0), f(2.0)])
    }),
    ("unord le nan", |b| {
        boolean(b, Opcode::FPUnordLessThanEqual32, &[f(NAN), f(2.0)])
    }),
    ("ord ge", |b| {
        boolean(b, Opcode::FPOrdGreaterThanEqual32, &[f(1.0), f(2.0)])
    }),
    ("unord ge nan", |b| {
        boolean(b, Opcode::FPUnordGreaterThanEqual32, &[f(NAN), f(2.0)])
    }),
    ("ord gt", |b| {
        boolean(b, Opcode::FPOrdGreaterThan32, &[f(3.0), f(2.0)])
    }),
    ("is nan", |b| boolean(b, Opcode::FPIsNan32, &[f(NAN)])),
    ("is nan not", |b| boolean(b, Opcode::FPIsNan32, &[f(1.0)])),
    ("select f32", |b| {
        float(b, Opcode::SelectF32, &[bit(true), f(1.0), f(2.0)])
    }),
];

/// Conversion, composite and guest-state variable cases.
const CONVERSION_CASES: &[Case] = &[
    ("f2s nan", |b| int(b, Opcode::ConvertS32F32, &[f(NAN)])),
    ("f2s big", |b| int(b, Opcode::ConvertS32F32, &[f(3.0e9)])),
    ("f2s small", |b| int(b, Opcode::ConvertS32F32, &[f(-3.0e9)])),
    ("f2s truncates", |b| {
        int(b, Opcode::ConvertS32F32, &[f(-1.7)])
    }),
    ("f2u negative", |b| {
        int(b, Opcode::ConvertU32F32, &[f(-1.0)])
    }),
    ("f2u big", |b| int(b, Opcode::ConvertU32F32, &[f(5.0e9)])),
    ("f2u nan", |b| int(b, Opcode::ConvertU32F32, &[f(NAN)])),
    ("f2s16 big", |b| {
        int(b, Opcode::ConvertS16F32, &[f(40000.0)])
    }),
    ("f2s16 small", |b| {
        int(b, Opcode::ConvertS16F32, &[f(-40000.0)])
    }),
    ("f2u16 big", |b| {
        int(b, Opcode::ConvertU16F32, &[f(70000.0)])
    }),
    ("f2u16 nan", |b| int(b, Opcode::ConvertU16F32, &[f(NAN)])),
    ("s2f", |b| {
        float(b, Opcode::ConvertF32S32, &[u((-3_i32).cast_unsigned())])
    }),
    ("u2f", |b| {
        float(b, Opcode::ConvertF32U32, &[u(0x00ff_ffff)])
    }),
    ("s8 to f", |b| {
        float(b, Opcode::ConvertF32S8, &[u(0x1_0080)])
    }),
    ("u8 to f", |b| {
        float(b, Opcode::ConvertF32U8, &[u(0x1_0080)])
    }),
    ("s16 to f", |b| {
        float(b, Opcode::ConvertF32S16, &[u(0x1_ffff)])
    }),
    ("u16 to f", |b| {
        float(b, Opcode::ConvertF32U16, &[u(0x1_8000)])
    }),
    ("bitcast", |b| {
        let bits = op(b, Opcode::BitCastF32U32, &[u(0x4040_0000)]);
        float(b, Opcode::FPAdd32, &[bits.into(), f(1.0)])
    }),
    ("composite insert", |b| {
        let vector = op(b, Opcode::CompositeConstructU32x3, &[u(1), u(2), u(3)]);
        let inserted = op(
            b,
            Opcode::CompositeInsertU32x3,
            &[vector.into(), u(9), u(1)],
        );
        let first = op(b, Opcode::CompositeExtractU32x3, &[inserted.into(), u(0)]);
        let second = op(b, Opcode::CompositeExtractU32x3, &[inserted.into(), u(1)]);
        let third = op(b, Opcode::CompositeExtractU32x3, &[inserted.into(), u(2)]);
        let sum = op(b, Opcode::IAdd32, &[first.into(), second.into()]);
        let shifted = op(b, Opcode::ShiftLeftLogical32, &[third.into(), u(8)]);
        int(b, Opcode::BitwiseOr32, &[sum.into(), shifted.into()])
    }),
    ("composite f32x4", |b| {
        let vector = op(
            b,
            Opcode::CompositeConstructF32x4,
            &[f(1.0), f(2.0), f(3.0), f(4.0)],
        );
        let inserted = op(
            b,
            Opcode::CompositeInsertF32x4,
            &[vector.into(), f(8.0), u(3)],
        );
        float(b, Opcode::CompositeExtractF32x4, &[inserted.into(), u(3)])
    }),
    ("variables", |b| {
        let pred = Operand::Imm(Imm::Pred(Pred::new(2).expect("p2")));
        let upred = Operand::Imm(Imm::UniformPred(UniformPred::new(1).expect("up1")));
        let ureg = Operand::Imm(Imm::UniformReg(UniformReg::new(3).expect("ur3")));
        op(b, Opcode::SetPred, &[pred, bit(true)]);
        op(b, Opcode::SetUniformPred, &[upred, bit(true)]);
        op(b, Opcode::SetUniformRegister, &[ureg, u(40)]);
        op(b, Opcode::SetGotoVariable, &[u(7), bit(true)]);
        op(b, Opcode::SetIndirectBranchVariable, &[u(2)]);
        op(b, Opcode::SetCFlag, &[bit(true)]);
        let p = op(b, Opcode::GetPred, &[pred]);
        let up = op(b, Opcode::GetUniformPred, &[upred]);
        let g = op(b, Opcode::GetGotoVariable, &[u(7)]);
        let c = op(b, Opcode::GetCFlag, &[]);
        let z = op(b, Opcode::GetZFlag, &[]);
        let all = op(b, Opcode::LogicalAnd, &[p.into(), up.into()]);
        let all = op(b, Opcode::LogicalAnd, &[all.into(), g.into()]);
        let all = op(b, Opcode::LogicalAnd, &[all.into(), c.into()]);
        let not_z = op(b, Opcode::LogicalNot, &[z.into()]);
        let all = op(b, Opcode::LogicalAnd, &[all.into(), not_z.into()]);
        let r = op(b, Opcode::GetUniformRegister, &[ureg]);
        let i = op(b, Opcode::GetIndirectBranchVariable, &[]);
        let sum = op(b, Opcode::IAdd32, &[r.into(), i.into()]);
        int(b, Opcode::SelectU32, &[all.into(), sum.into(), u(0)])
    }),
    ("undef and identity", |b| {
        let undef = op(b, Opcode::UndefU32, &[]);
        let same = op(b, Opcode::Identity, &[undef.into()]);
        int(b, Opcode::IAdd32, &[same.into(), u(4)])
    }),
];

/// The half of `value`.
fn half(builder: &mut Builder<'_>, value: f32) -> Operand {
    op(builder, Opcode::ConvertF16F32, &[f(value)]).into()
}

/// The 16-bit comparison `opcode` of four operand pairs, one bit each: (1, 2) in bit 0, (2, 1)
/// in bit 1, (1, 1) in bit 2 and (NaN, 1) in bit 3, so each operator, ordered or unordered,
/// gives its own pattern.
fn f16_compare(builder: &mut Builder<'_>, opcode: Opcode) -> Value {
    let pairs = [(1.0, 2.0), (2.0, 1.0), (1.0, 1.0), (NAN, 1.0)];
    let mut bits: Operand = u(0);
    for (index, (x, y)) in pairs.into_iter().enumerate() {
        let x = half(builder, x);
        let y = half(builder, y);
        let result = op(builder, opcode, &[x, y]);
        let set = op(
            builder,
            Opcode::SelectU32,
            &[result.into(), u(1 << index), u(0)],
        );
        bits = op(builder, Opcode::BitwiseOr32, &[bits, set.into()]).into();
    }
    op(builder, Opcode::Identity, &[bits])
}

/// `opcode` (a conversion to f16) of `value`, widened to f32, as its bits.
fn to_half(builder: &mut Builder<'_>, opcode: Opcode, value: u32) -> Value {
    let x = op(builder, opcode, &[u(value)]);
    float(builder, Opcode::ConvertF32F16, &[x.into()])
}

/// `opcode` (a conversion from f16) of the half of `value`.
fn from_half(builder: &mut Builder<'_>, opcode: Opcode, value: f32) -> Value {
    let x = half(builder, value);
    int(builder, opcode, &[x])
}

/// 16-bit float cases (`SHADER_F16`), with the half packing that counts as needing it.
const F16_CASES: &[Case] = &[
    ("pack half", |b| {
        let pair = op(b, Opcode::CompositeConstructF32x2, &[f(1.0), f(-2.0)]);
        int(b, Opcode::PackHalf2x16, &[pair.into()])
    }),
    ("unpack half", |b| {
        let pair = op(b, Opcode::UnpackHalf2x16, &[u(0xc000_3c00)]);
        float(b, Opcode::CompositeExtractF32x2, &[pair.into(), u(1)])
    }),
    ("f16 add", |b| {
        let x = op(b, Opcode::ConvertF16F32, &[f(1.5)]);
        let y = op(b, Opcode::ConvertF16F32, &[f(2.25)]);
        let sum = op(b, Opcode::FPAdd16, &[x.into(), y.into()]);
        float(b, Opcode::ConvertF32F16, &[sum.into()])
    }),
    ("f16 fma neg abs", |b| {
        let x = op(b, Opcode::ConvertF16F32, &[f(-2.0)]);
        let abs = op(b, Opcode::FPAbs16, &[x.into()]);
        let neg = op(b, Opcode::FPNeg16, &[abs.into()]);
        let fma = op(b, Opcode::FPFma16, &[neg.into(), x.into(), abs.into()]);
        float(b, Opcode::ConvertF32F16, &[fma.into()])
    }),
    ("f16 rounding", |b| {
        let x = op(b, Opcode::ConvertF16F32, &[f(2.5)]);
        let even = op(b, Opcode::FPRoundEven16, &[x.into()]);
        let floor = op(b, Opcode::FPFloor16, &[x.into()]);
        let sum = op(b, Opcode::FPMul16, &[even.into(), floor.into()]);
        float(b, Opcode::ConvertF32F16, &[sum.into()])
    }),
    ("f16 clamp nan", |b| {
        let x = op(b, Opcode::ConvertF16F32, &[f(NAN)]);
        let low = op(b, Opcode::ConvertF16F32, &[f(0.5)]);
        let high = op(b, Opcode::ConvertF16F32, &[f(1.0)]);
        let clamped = op(b, Opcode::FPClamp16, &[x.into(), low.into(), high.into()]);
        float(b, Opcode::ConvertF32F16, &[clamped.into()])
    }),
    ("f16 to s32", |b| {
        let x = op(b, Opcode::ConvertF16F32, &[f(-3.75)]);
        int(b, Opcode::ConvertS32F16, &[x.into()])
    }),
    ("f16 to u16 big", |b| {
        let x = op(b, Opcode::ConvertF16F32, &[f(65504.0)]);
        int(b, Opcode::ConvertU16F16, &[x.into()])
    }),
    ("s32 to f16", |b| {
        let x = op(b, Opcode::ConvertF16S32, &[u((-3_i32).cast_unsigned())]);
        float(b, Opcode::ConvertF32F16, &[x.into()])
    }),
    ("f16 compare", |b| {
        let x = op(b, Opcode::ConvertF16F32, &[f(1.0)]);
        let nan = op(b, Opcode::ConvertF16F32, &[f(NAN)]);
        let lt = op(b, Opcode::FPUnordLessThan16, &[nan.into(), x.into()]);
        let is_nan = op(b, Opcode::FPIsNan16, &[nan.into()]);
        let both = op(b, Opcode::LogicalAnd, &[lt.into(), is_nan.into()]);
        int(b, Opcode::SelectU32, &[both.into(), u(1), u(0)])
    }),
    ("f16 ord eq", |b| f16_compare(b, Opcode::FPOrdEqual16)),
    ("f16 unord eq", |b| f16_compare(b, Opcode::FPUnordEqual16)),
    ("f16 ord ne", |b| f16_compare(b, Opcode::FPOrdNotEqual16)),
    ("f16 unord ne", |b| {
        f16_compare(b, Opcode::FPUnordNotEqual16)
    }),
    ("f16 ord lt", |b| f16_compare(b, Opcode::FPOrdLessThan16)),
    ("f16 unord lt", |b| {
        f16_compare(b, Opcode::FPUnordLessThan16)
    }),
    ("f16 ord gt", |b| f16_compare(b, Opcode::FPOrdGreaterThan16)),
    ("f16 unord gt", |b| {
        f16_compare(b, Opcode::FPUnordGreaterThan16)
    }),
    ("f16 ord le", |b| {
        f16_compare(b, Opcode::FPOrdLessThanEqual16)
    }),
    ("f16 unord le", |b| {
        f16_compare(b, Opcode::FPUnordLessThanEqual16)
    }),
    ("f16 ord ge", |b| {
        f16_compare(b, Opcode::FPOrdGreaterThanEqual16)
    }),
    ("f16 unord ge", |b| {
        f16_compare(b, Opcode::FPUnordGreaterThanEqual16)
    }),
    // The narrow sources have their sign bit set and a bit above the field.
    ("s8 to f16", |b| to_half(b, Opcode::ConvertF16S8, 0x1_0080)),
    ("u8 to f16", |b| to_half(b, Opcode::ConvertF16U8, 0x1_0080)),
    ("s16 to f16", |b| {
        to_half(b, Opcode::ConvertF16S16, 0x1_8000)
    }),
    ("u16 to f16", |b| {
        to_half(b, Opcode::ConvertF16U16, 0x1_8000)
    }),
    ("u32 to f16", |b| {
        to_half(b, Opcode::ConvertF16U32, 0x0000_0400)
    }),
    ("u32 to f16 top bit", |b| {
        to_half(b, Opcode::ConvertF16U32, 0x8000_0000)
    }),
    ("s32 to f16 small", |b| {
        to_half(b, Opcode::ConvertF16S32, (-1024_i32).cast_unsigned())
    }),
    // Negative, past the range, infinite and NaN halves to each integer type.
    ("f16 to s16 negative", |b| {
        from_half(b, Opcode::ConvertS16F16, -3.75)
    }),
    ("f16 to s16 big", |b| {
        from_half(b, Opcode::ConvertS16F16, 40000.0)
    }),
    ("f16 to s16 small", |b| {
        from_half(b, Opcode::ConvertS16F16, -40000.0)
    }),
    ("f16 to s16 nan", |b| {
        from_half(b, Opcode::ConvertS16F16, NAN)
    }),
    ("f16 to u16 negative", |b| {
        from_half(b, Opcode::ConvertU16F16, -3.75)
    }),
    ("f16 to u16 infinite", |b| {
        from_half(b, Opcode::ConvertU16F16, f32::INFINITY)
    }),
    ("f16 to s32 infinite", |b| {
        from_half(b, Opcode::ConvertS32F16, f32::INFINITY)
    }),
    ("f16 to s32 negative infinite", |b| {
        from_half(b, Opcode::ConvertS32F16, f32::NEG_INFINITY)
    }),
    ("f16 to s32 nan", |b| {
        from_half(b, Opcode::ConvertS32F16, NAN)
    }),
    ("f16 to u32 negative", |b| {
        from_half(b, Opcode::ConvertU32F16, -3.75)
    }),
    ("f16 to u32 big", |b| {
        from_half(b, Opcode::ConvertU32F16, 65504.0)
    }),
    ("f16 to u32 infinite", |b| {
        from_half(b, Opcode::ConvertU32F16, f32::INFINITY)
    }),
    ("f16 to u32 nan", |b| {
        from_half(b, Opcode::ConvertU32F16, NAN)
    }),
    ("f16 pack", |b| {
        let x = op(b, Opcode::ConvertF16F32, &[f(1.0)]);
        let y = op(b, Opcode::ConvertF16F32, &[f(-2.0)]);
        let pair = op(b, Opcode::CompositeConstructF16x2, &[x.into(), y.into()]);
        int(b, Opcode::PackFloat2x16, &[pair.into()])
    }),
    ("f16 unpack", |b| {
        let pair = op(b, Opcode::UnpackFloat2x16, &[u(0xc000_3c00)]);
        let y = op(b, Opcode::CompositeExtractF16x2, &[pair.into(), u(1)]);
        float(b, Opcode::ConvertF32F16, &[y.into()])
    }),
];

/// 64-bit integer cases (`SHADER_INT64`).
const I64_CASES: &[Case] = &[
    ("u64 add and unpack", |b| {
        let pair = op(b, Opcode::CompositeConstructU32x2, &[u(u32::MAX), u(1)]);
        let wide = op(b, Opcode::PackUint2x32, &[pair.into()]);
        let one = op(b, Opcode::ConvertU64U32, &[u(1)]);
        let sum = op(b, Opcode::IAdd64, &[wide.into(), one.into()]);
        let halves = op(b, Opcode::UnpackUint2x32, &[sum.into()]);
        let low = op(b, Opcode::CompositeExtractU32x2, &[halves.into(), u(0)]);
        let high = op(b, Opcode::CompositeExtractU32x2, &[halves.into(), u(1)]);
        let shifted = op(b, Opcode::ShiftLeftLogical32, &[high.into(), u(8)]);
        int(b, Opcode::BitwiseOr32, &[low.into(), shifted.into()])
    }),
    ("u64 shl 63", |b| {
        let wide = op(b, Opcode::ConvertU64U32, &[u(3)]);
        let shifted = op(b, Opcode::ShiftLeftLogical64, &[wide.into(), u(63)]);
        let high = op(b, Opcode::ShiftRightLogical64, &[shifted.into(), u(32)]);
        int(b, Opcode::ConvertU32U64, &[high.into()])
    }),
    ("u64 shl 64", |b| {
        let wide = op(b, Opcode::ConvertU64U32, &[u(3)]);
        let shifted = op(b, Opcode::ShiftLeftLogical64, &[wide.into(), u(64)]);
        int(b, Opcode::ConvertU32U64, &[shifted.into()])
    }),
    ("u64 sar 64", |b| {
        let one = op(b, Opcode::ConvertU64U32, &[u(1)]);
        let negative = op(b, Opcode::INeg64, &[one.into()]);
        let shifted = op(b, Opcode::ShiftRightArithmetic64, &[negative.into(), u(64)]);
        int(b, Opcode::ConvertU32U64, &[shifted.into()])
    }),
    ("u64 abs and sub", |b| {
        let one = op(b, Opcode::ConvertU64U32, &[u(5)]);
        let ten = op(b, Opcode::ConvertU64U32, &[u(10)]);
        let negative = op(b, Opcode::ISub64, &[one.into(), ten.into()]);
        let abs = op(b, Opcode::IAbs64, &[negative.into()]);
        let picked = op(b, Opcode::SelectU64, &[bit(true), abs.into(), one.into()]);
        int(b, Opcode::ConvertU32U64, &[picked.into()])
    }),
];

/// The lowered integer operations give the interpreter's results on the host's GPU, at every
/// edge the interpreter defines (shifts at and past the width, division by zero, bit fields
/// past bit 31, the bit searches of 0 and -1).
#[test]
fn integer_operations_match_the_interpreter_on_the_gpu() {
    let mismatches = gpu_mismatches("integer_operations", INTEGER_CASES);
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// The lowered 32-bit float operations give the interpreter's results on the host's GPU,
/// NaN operands included.
#[test]
fn float_operations_match_the_interpreter_on_the_gpu() {
    let mismatches = gpu_mismatches("float_operations", FLOAT_CASES);
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// Conversions, composites and the guest-state variables, on the host's GPU.
#[test]
fn conversions_and_variables_match_the_interpreter_on_the_gpu() {
    let mismatches = gpu_mismatches("conversions_and_variables", CONVERSION_CASES);
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// The 16-bit float operations need `f16` and match the interpreter on the host's GPU.
#[test]
fn f16_operations_match_the_interpreter_on_the_gpu() {
    let (function, _) = cases_function(F16_CASES);
    let shader = lowered(GuestStage::Compute, &function, &ONE_INVOCATION);
    assert_eq!(
        shader.requirements,
        Requirements {
            f16: true,
            ..Requirements::default()
        }
    );
    let mismatches = gpu_mismatches("f16_operations", F16_CASES);
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// The 64-bit integer operations need `i64` and match the interpreter on the host's GPU.
#[test]
fn i64_operations_match_the_interpreter_on_the_gpu() {
    let (function, _) = cases_function(I64_CASES);
    let shader = lowered(GuestStage::Compute, &function, &ONE_INVOCATION);
    assert_eq!(
        shader.requirements,
        Requirements {
            i64: true,
            ..Requirements::default()
        }
    );
    let mismatches = gpu_mismatches("i64_operations", I64_CASES);
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// `function` made opaque and lowered as a compute shader of one invocation, with the
/// interpreter's registers `regs` and, where the host can run it, the GPU's.
fn interp_and_gpu(test: &str, function: &Function, regs: &[u8]) -> (Vec<u32>, Option<Vec<u32>>) {
    let function = opaque(function);
    let shader = lowered(GuestStage::Compute, &function, &ONE_INVOCATION);
    let expected = interp_registers(&function, regs);
    (expected.clone(), gpu_registers(test, &shader, regs))
}

/// `F2F` from 32 to 16 bits with each rounding mode, with and without `.FTZ`, as the translator
/// makes it (`translate::float`, "Rounding modes and flush-to-zero"): converted to nearest,
/// then moved one step, with the mode and the flushing also carried on `ConvertF16F32`, which
/// this lowering ignores. On the host's GPU the results are the interpreter's, which honours
/// the carried controls: hand-worked halves for each mode at overflow, ties, below the
/// smallest denormal, a denormal result and a flushed one (the translator's own
/// `f2f_emulation_matches_the_modelled_conversion` cases).
#[test]
fn translated_f2f_rounding_matches_the_interpreter_on_the_gpu() {
    use crate::decode::{AluSrc, FloatType, Instr, Op, OpExit, OpF2F, PredSrc, RoundMode};
    use crate::translate::driver::translate_instrs;
    use crate::translate::{COLOR_TARGETS, ColorOutputs, Stage};
    // The input, `.FTZ`, then the half for nearest, down, up and toward zero.
    let cases: [(f32, bool, [u32; 4]); 10] = [
        (65519.0, false, [0x7bff, 0x7bff, 0x7c00, 0x7bff]),
        (65520.0, false, [0x7c00, 0x7bff, 0x7c00, 0x7bff]),
        (-70000.0, false, [0xfc00, 0xfc00, 0xfbff, 0xfbff]),
        (1.000_488_3, false, [0x3c00, 0x3c00, 0x3c01, 0x3c00]),
        (-1.000_488_3, false, [0xbc00, 0xbc01, 0xbc00, 0xbc00]),
        (1.0e-8, false, [0x0000, 0x0000, 0x0001, 0x0000]),
        (-1.0e-8, false, [0x8000, 0x8001, 0x8000, 0x8000]),
        (1.0e-6, false, [0x0011, 0x0010, 0x0011, 0x0010]),
        (1.0e-6, true, [0x0000; 4]),
        (f32::from_bits(1), true, [0x0000; 4]),
    ];
    // The translator takes vertex and pixel programs; a vertex program's exit adds nothing
    // the compute lowering refuses.
    let stage = Stage {
        stage: GuestStage::Vertex,
        color_outputs: ColorOutputs {
            targets: [0; COLOR_TARGETS],
        },
    };
    for (index, mode) in RoundMode::ALL.into_iter().enumerate() {
        let mut program = Vec::new();
        let mut regs = Vec::new();
        for (reg, (input, ftz, _)) in cases.iter().enumerate() {
            let reg = u8::try_from(reg).expect("few cases");
            program.push(Instr::new(Op::F2F(OpF2F {
                dst: Gpr::new(reg),
                src: AluSrc::Imm(input.to_bits()),
                dst_type: FloatType::F16,
                src_type: FloatType::F32,
                rnd: mode,
                ftz: *ftz,
                high: false,
                wide: false,
            })));
            regs.push(reg);
        }
        program.push(Instr::new(Op::Exit(OpExit {
            keep_ref_count: false,
            no_at_exit: false,
            pred: PredSrc::TRUE,
        })));
        let function = translate_instrs(stage, None, &program).expect("translates");
        let test = format!("translated_f2f_{mode:?}");
        let (expected, found) = interp_and_gpu(&test, &function, &regs);
        let worked: Vec<u32> = cases.iter().map(|(_, _, halves)| halves[index]).collect();
        assert_eq!(expected, worked, "the interpreter, {mode:?}");
        if let Some(found) = found {
            assert_eq!(found, worked, "the GPU, {mode:?}");
        }
    }
}

/// An integer-to-float conversion ignores its carried rounding mode (decision 10 (a) treats
/// `I2F` like arithmetic): toward zero, the interpreter rounds toward zero, while the host's
/// GPU rounds to nearest even, as it does with no mode. Measured on an Apple GPU; the
/// interpreter's results are the translator's `i2f_carries_its_rounding` literals.
#[test]
fn int_to_float_rounds_as_the_host_does_on_the_gpu() {
    // The opcode, the source; toward zero; to nearest.
    let cases: [(Opcode, u32, u32, u32); 3] = [
        (Opcode::ConvertF32S32, 0x0100_0003, 0x4b80_0001, 0x4b80_0002),
        (Opcode::ConvertF32S32, 0x7fff_ffff, 0x4eff_ffff, 0x4f00_0000),
        (Opcode::ConvertF32U32, 0xffff_ffff, 0x4f7f_ffff, 0x4f80_0000),
    ];
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    let mut regs = Vec::new();
    for (reg, (opcode, source, _, _)) in cases.iter().enumerate() {
        let reg = u8::try_from(reg).expect("few cases");
        let flags = Flags::Fp(FpControl {
            rounding: FpRounding::Rz,
            ..FpControl::default()
        });
        let value = builder
            .op_with(*opcode, &[u(*source)], flags)
            .expect("a conversion");
        let bits = op(&mut builder, Opcode::BitCastU32F32, &[value.into()]);
        op(
            &mut builder,
            Opcode::SetRegister,
            &[Operand::Imm(Imm::Reg(Gpr::new(reg))), bits.into()],
        );
        regs.push(reg);
    }
    builder
        .terminate(Terminator::Return)
        .expect("the block returns");
    let (expected, found) = interp_and_gpu("int_to_float_rounding", &function, &regs);
    let toward_zero: Vec<u32> = cases.iter().map(|case| case.2).collect();
    let nearest: Vec<u32> = cases.iter().map(|case| case.3).collect();
    assert_eq!(expected, toward_zero, "the interpreter");
    if let Some(found) = found {
        assert_eq!(found, nearest, "the GPU");
    }
}

/// NaN results of arithmetic keep the host's bits: the lowering does not canonicalise them
/// (the parent module's divergence table). Each case is a NaN on the host's GPU and in the
/// interpreter, but the bits may differ: the interpreter gives its canonical binary32 NaN,
/// itself a guess (`interp::float`), and an Apple GPU gave `0x7fc0_0000` for every case here.
#[test]
fn nan_results_are_the_hosts_on_the_gpu() {
    const NAN_CASES: &[Case] = &[
        ("fadd nan", |b| float(b, Opcode::FPAdd32, &[f(NAN), f(1.0)])),
        ("fmul inf by 0", |b| {
            float(b, Opcode::FPMul32, &[f(f32::INFINITY), f(0.0)])
        }),
        ("fsqrt -1", |b| float(b, Opcode::FPSqrt, &[f(-1.0)])),
        ("fmax nan nan", |b| {
            float(b, Opcode::FPMax32, &[f(NAN), f(NAN)])
        }),
        ("f16 add nan widened", |b| {
            let x = half(b, NAN);
            let y = half(b, 1.0);
            let sum = op(b, Opcode::FPAdd16, &[x, y]);
            float(b, Opcode::ConvertF32F16, &[sum.into()])
        }),
    ];
    let (function, regs) = cases_function(NAN_CASES);
    let (expected, found) = interp_and_gpu("nan_results", &function, &regs);
    let is_nan = |bits: &u32| f32::from_bits(*bits).is_nan();
    assert!(
        expected.iter().all(is_nan),
        "the interpreter: {expected:#x?}"
    );
    assert!(
        expected.windows(2).all(|pair| pair[0] == pair[1]),
        "the interpreter's NaN is canonical: {expected:#x?}"
    );
    if let Some(found) = found {
        assert!(found.iter().all(is_nan), "the GPU: {found:#x?}");
    }
}

/// The MSL naga writes for each group of cases above (snapshots, so a change to an
/// operation's lowering shows where no GPU runs; a change is explained in its PR).
#[test]
fn operations_snapshot() {
    for (name, cases) in [
        ("operations_integer", INTEGER_CASES),
        ("operations_float", FLOAT_CASES),
        ("operations_conversion", CONVERSION_CASES),
        ("operations_f16", F16_CASES),
        ("operations_i64", I64_CASES),
    ] {
        let (function, _) = cases_function(cases);
        let shader = lowered(GuestStage::Compute, &function, &ONE_INVOCATION);
        insta::assert_snapshot!(name, msl(&shader));
    }
}
