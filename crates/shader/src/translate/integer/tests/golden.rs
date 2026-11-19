//! Golden IR: the exact text of representative forms, so a change to what a form translates
//! to shows in review.

use super::*;
use crate::decode::{OpLea, OpLop3, OpShf, ShfType};

fn text(op: Op) -> String {
    let function = build(&[Instr::new(op)]).expect("translates");
    assert_eq!(function.verify(), Ok(()));
    function.to_string()
}

/// `IADD3 R0, P1, P2, R1, -R2, c[3][0x6]` with both carry outputs: a 64-bit sum of the terms,
/// the negated one as `~x + 1`, and an unaligned constant-buffer read from two words.
#[test]
fn iadd3_with_carry_outputs() {
    let op = Op::IAdd3(OpIAdd3 {
        overflow: [p(1), p(2)],
        ..iadd3(0, [reg(1), neg(2), cb(3, 6)])
    });
    assert_eq!(
        text(op),
        "block0:\n\
         \x20   %0: u32 = GetRegister r1\n\
         \x20   %1: u32 = GetRegister r2\n\
         \x20   %2: u32 = GetCbufU32 u32 0x3, u32 0x4\n\
         \x20   %3: u32 = GetCbufU32 u32 0x3, u32 0x8\n\
         \x20   %4: u32 = ShiftRightLogical32 %2, u32 0x10\n\
         \x20   %5: u32 = ShiftLeftLogical32 %3, u32 0x10\n\
         \x20   %6: u32 = BitwiseOr32 %4, %5\n\
         \x20   %7: u32x2 = CompositeConstructU32x2 %0, u32 0x0\n\
         \x20   %8: u64 = PackUint2x32 %7\n\
         \x20   %9: u32 = BitwiseNot32 %1\n\
         \x20   %10: u32x2 = CompositeConstructU32x2 %9, u32 0x0\n\
         \x20   %11: u64 = PackUint2x32 %10\n\
         \x20   %12: u64 = IAdd64 %11, u64 0x1\n\
         \x20   %13: u32x2 = CompositeConstructU32x2 %6, u32 0x0\n\
         \x20   %14: u64 = PackUint2x32 %13\n\
         \x20   %15: u64 = IAdd64 %8, %12\n\
         \x20   %16: u64 = IAdd64 %15, %14\n\
         \x20   %17: u32x2 = UnpackUint2x32 %16\n\
         \x20   %18: u32 = CompositeExtractU32x2 %17, u32 0x0\n\
         \x20   %19: u32 = CompositeExtractU32x2 %17, u32 0x1\n\
         \x20   %20 = SetRegister r0, %18\n\
         \x20   %21: u1 = INotEqual %19, u32 0x0\n\
         \x20   %22 = SetPred p1, %21\n\
         \x20   %23: u1 = UGreaterThanEqual %19, u32 0x2\n\
         \x20   %24 = SetPred p2, %23\n\
         \x20   return\n"
    );
}

/// `LEA.HI R0, R1, R2, R3, 0x4`: the high word of `R3:R1 << 4`, plus `R2`.
#[test]
fn lea_hi() {
    let op = Op::Lea(OpLea {
        dst: r(0),
        overflow: Pred::PT,
        a: reg(1),
        b: reg(2),
        a_high: reg(3),
        shift: 4,
        dst_high: true,
        sx32: false,
        intermediate: false,
        carry: NOT_PT,
    });
    assert_eq!(
        text(op),
        "block0:\n\
         \x20   %0: u32 = GetRegister r1\n\
         \x20   %1: u32 = GetRegister r2\n\
         \x20   %2: u32 = GetRegister r3\n\
         \x20   %3: u32 = ShiftLeftLogical32 %2, u32 0x4\n\
         \x20   %4: u32 = ShiftRightLogical32 %0, u32 0x1c\n\
         \x20   %5: u32 = BitwiseOr32 %3, %4\n\
         \x20   %6: u32 = IAdd32 %5, %1\n\
         \x20   %7 = SetRegister r0, %6\n\
         \x20   return\n"
    );
}

/// `LOP3.LUT P0, R0, R1, R2, RZ, 0x3c`: `a ^ b` as a sum of products (rows 2 to 5), and its
/// predicate output.
#[test]
fn lop3_sum_of_products() {
    let op = Op::Lop3(OpLop3 {
        pred: p(0),
        dst: r(0),
        srcs: [reg(1), reg(2), AluSrc::reg(Gpr::RZ)],
        lut: 0x3c,
    });
    assert_eq!(
        text(op),
        "block0:\n\
         \x20   %0: u32 = GetRegister r1\n\
         \x20   %1: u32 = GetRegister r2\n\
         \x20   %2: u32 = BitwiseNot32 %0\n\
         \x20   %3: u32 = BitwiseAnd32 %2, %1\n\
         \x20   %4: u32 = BitwiseNot32 u32 0x0\n\
         \x20   %5: u32 = BitwiseAnd32 %3, %4\n\
         \x20   %6: u32 = BitwiseAnd32 %2, %1\n\
         \x20   %7: u32 = BitwiseAnd32 %6, u32 0x0\n\
         \x20   %8: u32 = BitwiseOr32 %5, %7\n\
         \x20   %9: u32 = BitwiseNot32 %1\n\
         \x20   %10: u32 = BitwiseAnd32 %0, %9\n\
         \x20   %11: u32 = BitwiseAnd32 %10, %4\n\
         \x20   %12: u32 = BitwiseOr32 %8, %11\n\
         \x20   %13: u32 = BitwiseAnd32 %0, %9\n\
         \x20   %14: u32 = BitwiseAnd32 %13, u32 0x0\n\
         \x20   %15: u32 = BitwiseOr32 %12, %14\n\
         \x20   %16 = SetRegister r0, %15\n\
         \x20   %17: u1 = INotEqual %15, u32 0x0\n\
         \x20   %18 = SetPred p0, %17\n\
         \x20   return\n"
    );
}

/// `SHF.R.S64.HI R0, R1, R2, R3`: a 64-bit arithmetic shift by the amount clamped to 64 (NAK's
/// fold), high word.
#[test]
fn shf_right_signed_64_high() {
    let op = Op::Shf(OpShf {
        dst: r(0),
        low: reg(1),
        shift: reg(2),
        high: reg(3),
        data_type: ShfType::I64,
        wrap: false,
        right: true,
        dst_high: true,
    });
    assert_eq!(
        text(op),
        "block0:\n\
         \x20   %0: u32 = GetRegister r1\n\
         \x20   %1: u32 = GetRegister r3\n\
         \x20   %2: u32x2 = CompositeConstructU32x2 %0, %1\n\
         \x20   %3: u64 = PackUint2x32 %2\n\
         \x20   %4: u32 = GetRegister r2\n\
         \x20   %5: u32 = UMin32 %4, u32 0x40\n\
         \x20   %6: u64 = ShiftRightArithmetic64 %3, %5\n\
         \x20   %7: u32x2 = UnpackUint2x32 %6\n\
         \x20   %8: u32 = CompositeExtractU32x2 %7, u32 0x1\n\
         \x20   %9 = SetRegister r0, %8\n\
         \x20   return\n"
    );
}
