//! The integer family: arithmetic, shifts, bit fields, bit searches, minimum and maximum,
//! logic, the integer conversions and the undefined values, with the defined results the
//! parent module lists for SPIR-V's undefined cases.

use super::{eval, u1_of, u32_of, u64_of, value_of};
use crate::interp::Val;

fn u32_op(op: &str) -> u32 {
    u32_of(eval(op))
}

fn u64_op(op: &str) -> u64 {
    u64_of(eval(op))
}

#[test]
fn arithmetic_wraps() {
    assert_eq!(u32_op("IAdd32 u32 0xffffffff, u32 2"), 1);
    assert_eq!(u32_op("ISub32 u32 1, u32 2"), 0xffff_ffff);
    assert_eq!(u32_op("IMul32 u32 0x10001, u32 0x10000"), 0x1_0000);
    assert_eq!(u32_op("INeg32 u32 1"), 0xffff_ffff);
    assert_eq!(u32_op("INeg32 u32 0x80000000"), 0x8000_0000);
    assert_eq!(u32_op("IAbs32 u32 0xfffffffb"), 5);
    assert_eq!(u32_op("IAbs32 u32 5"), 5);
    assert_eq!(u32_op("IAbs32 u32 0x80000000"), 0x8000_0000);
    assert_eq!(u64_op("IAdd64 u64 0xffffffffffffffff, u64 2"), 1);
    assert_eq!(u64_op("IAdd64 u64 0xffffffff, u64 1"), 0x1_0000_0000);
    assert_eq!(u64_op("ISub64 u64 0, u64 1"), u64::MAX);
    assert_eq!(u64_op("INeg64 u64 1"), u64::MAX);
    assert_eq!(u64_op("IAbs64 u64 0xfffffffffffffffe"), 2);
    assert_eq!(
        u64_op("IAbs64 u64 0x8000000000000000"),
        0x8000_0000_0000_0000
    );
}

#[test]
fn division_truncates_and_defines_the_undefined_cases() {
    assert_eq!(u32_op("UDiv32 u32 7, u32 2"), 3);
    assert_eq!(u32_op("UDiv32 u32 0xfffffffe, u32 2"), 0x7fff_ffff);
    assert_eq!(u32_op("SDiv32 u32 0xfffffff9, u32 2"), 0xffff_fffd); // -7 / 2 = -3
    assert_eq!(u32_op("SDiv32 u32 7, u32 0xfffffffe"), 0xffff_fffd); // 7 / -2 = -3
    // By zero (guesses) and the one overflowing quotient.
    assert_eq!(u32_op("UDiv32 u32 5, u32 0"), 0xffff_ffff);
    assert_eq!(u32_op("SDiv32 u32 5, u32 0"), 0xffff_ffff);
    assert_eq!(u32_op("SDiv32 u32 0x80000000, u32 0xffffffff"), 0x8000_0000);
}

#[test]
fn shifts_at_and_past_the_width() {
    assert_eq!(u32_op("ShiftLeftLogical32 u32 1, u32 31"), 0x8000_0000);
    assert_eq!(u32_op("ShiftLeftLogical32 u32 1, u32 32"), 0);
    assert_eq!(u32_op("ShiftLeftLogical32 u32 1, u32 0xffffffff"), 0);
    assert_eq!(u32_op("ShiftRightLogical32 u32 0x80000000, u32 31"), 1);
    assert_eq!(u32_op("ShiftRightLogical32 u32 0x80000000, u32 32"), 0);
    assert_eq!(
        u32_op("ShiftRightArithmetic32 u32 0x80000000, u32 4"),
        0xf800_0000
    );
    assert_eq!(
        u32_op("ShiftRightArithmetic32 u32 0x80000000, u32 31"),
        0xffff_ffff
    );
    assert_eq!(
        u32_op("ShiftRightArithmetic32 u32 0x80000000, u32 32"),
        0xffff_ffff
    );
    assert_eq!(u32_op("ShiftRightArithmetic32 u32 0x40000000, u32 32"), 0);
    assert_eq!(u32_op("ShiftRightArithmetic32 u32 0x40000000, u32 30"), 1);
    assert_eq!(u64_op("ShiftLeftLogical64 u64 1, u32 63"), 1 << 63);
    assert_eq!(u64_op("ShiftLeftLogical64 u64 1, u32 64"), 0);
    assert_eq!(u64_op("ShiftLeftLogical64 u64 1, u32 32"), 1 << 32);
    assert_eq!(
        u64_op("ShiftRightLogical64 u64 0x8000000000000000, u32 63"),
        1
    );
    assert_eq!(
        u64_op("ShiftRightLogical64 u64 0x8000000000000000, u32 64"),
        0
    );
    assert_eq!(
        u64_op("ShiftRightArithmetic64 u64 0x8000000000000000, u32 64"),
        u64::MAX
    );
    assert_eq!(
        u64_op("ShiftRightArithmetic64 u64 0x8000000000000000, u32 60"),
        0xffff_ffff_ffff_fff8
    );
    assert_eq!(
        u64_op("ShiftRightArithmetic64 u64 0x4000000000000000, u32 64"),
        0
    );
}

#[test]
fn bitwise_operations() {
    assert_eq!(
        u32_op("BitwiseAnd32 u32 0xff00ff00, u32 0x0ff00ff0"),
        0x0f00_0f00
    );
    assert_eq!(
        u32_op("BitwiseOr32 u32 0xff00ff00, u32 0x0ff00ff0"),
        0xfff0_fff0
    );
    assert_eq!(
        u32_op("BitwiseXor32 u32 0xff00ff00, u32 0x0ff00ff0"),
        0xf0f0_f0f0
    );
    assert_eq!(u32_op("BitwiseNot32 u32 0xff00ff00"), 0x00ff_00ff);
    assert_eq!(u32_op("BitReverse32 u32 0x00000003"), 0xc000_0000);
    assert_eq!(u32_op("BitCount32 u32 0xf0f00001"), 9);
}

#[test]
fn bit_field_extract_inside_and_past_bit_31() {
    let base = "u32 0xabcd1234";
    assert_eq!(
        u32_op(&format!("BitFieldUExtract {base}, u32 8, u32 8")),
        0x12
    );
    assert_eq!(
        u32_op(&format!("BitFieldUExtract {base}, u32 0, u32 32")),
        0xabcd_1234
    );
    // Bits above 31 read as zero; offset and count are clamped to 32.
    assert_eq!(
        u32_op(&format!("BitFieldUExtract {base}, u32 28, u32 8")),
        0xa
    );
    assert_eq!(
        u32_op(&format!("BitFieldUExtract {base}, u32 0, u32 40")),
        0xabcd_1234
    );
    assert_eq!(
        u32_op(&format!("BitFieldUExtract {base}, u32 32, u32 8")),
        0
    );
    assert_eq!(
        u32_op(&format!("BitFieldUExtract {base}, u32 33, u32 8")),
        0
    );
    assert_eq!(u32_op(&format!("BitFieldUExtract {base}, u32 4, u32 0")), 0);
    // Signed: the field's top bit is copied upwards.
    assert_eq!(
        u32_op("BitFieldSExtract u32 0x8000, u32 12, u32 4"),
        0xffff_fff8
    );
    assert_eq!(u32_op("BitFieldSExtract u32 0x7000, u32 12, u32 4"), 7);
    assert_eq!(u32_op("BitFieldSExtract u32 0x8000, u32 12, u32 0"), 0);
    assert_eq!(
        u32_op("BitFieldSExtract u32 0x80000000, u32 0, u32 32"),
        0x8000_0000
    );
    assert_eq!(
        u32_op("BitFieldSExtract u32 0x80000000, u32 0, u32 33"),
        0x8000_0000
    );
    // The field past bit 31 has zeros above bit 31, so its top bit (bit 35) is clear.
    assert_eq!(u32_op("BitFieldSExtract u32 0x80000000, u32 28, u32 8"), 8);
    assert_eq!(
        u32_op("BitFieldSExtract u32 0x80000000, u32 28, u32 4"),
        0xffff_fff8
    );
}

#[test]
fn bit_field_insert_inside_and_past_bit_31() {
    assert_eq!(
        u32_op("BitFieldInsert u32 0xffffffff, u32 0x12, u32 8, u32 8"),
        0xffff_12ff
    );
    // Only the low `count` bits of the insert are used.
    assert_eq!(
        u32_op("BitFieldInsert u32 0x0, u32 0xfff, u32 4, u32 4"),
        0xf0
    );
    // Bits that would land past bit 31 are dropped.
    assert_eq!(
        u32_op("BitFieldInsert u32 0x0, u32 0xff, u32 28, u32 8"),
        0xf000_0000
    );
    assert_eq!(
        u32_op("BitFieldInsert u32 0x1234, u32 0xff, u32 32, u32 8"),
        0x1234
    );
    assert_eq!(
        u32_op("BitFieldInsert u32 0x1234, u32 0xff, u32 33, u32 8"),
        0x1234
    );
    assert_eq!(
        u32_op("BitFieldInsert u32 0x1234, u32 0xff, u32 4, u32 0"),
        0x1234
    );
    assert_eq!(
        u32_op("BitFieldInsert u32 0x1234, u32 0xabcdef01, u32 0, u32 32"),
        0xabcd_ef01
    );
    assert_eq!(
        u32_op("BitFieldInsert u32 0x1234, u32 0xabcdef01, u32 0, u32 40"),
        0xabcd_ef01
    );
}

#[test]
fn bit_searches_follow_glsl() {
    assert_eq!(u32_op("FindUMsb32 u32 0"), 0xffff_ffff);
    assert_eq!(u32_op("FindUMsb32 u32 1"), 0);
    assert_eq!(u32_op("FindUMsb32 u32 0x80000000"), 31);
    assert_eq!(u32_op("FindUMsb32 u32 0x00f00000"), 23);
    assert_eq!(u32_op("FindSMsb32 u32 0"), 0xffff_ffff);
    assert_eq!(u32_op("FindSMsb32 u32 0xffffffff"), 0xffff_ffff);
    assert_eq!(u32_op("FindSMsb32 u32 0xfffffffe"), 0);
    assert_eq!(u32_op("FindSMsb32 u32 5"), 2);
    assert_eq!(u32_op("FindSMsb32 u32 0x80000000"), 30);
    assert_eq!(u32_op("FindSMsb32 u32 0xfff0ffff"), 19);
}

#[test]
fn minimum_maximum_and_clamp_by_signedness() {
    // 0xffffffff is -1 signed and the largest unsigned value.
    assert_eq!(u32_op("SMin32 u32 0xffffffff, u32 1"), 0xffff_ffff);
    assert_eq!(u32_op("UMin32 u32 0xffffffff, u32 1"), 1);
    assert_eq!(u32_op("SMax32 u32 0xffffffff, u32 1"), 1);
    assert_eq!(u32_op("UMax32 u32 0xffffffff, u32 1"), 0xffff_ffff);
    assert_eq!(u32_op("SMin32 u32 1, u32 0xffffffff"), 0xffff_ffff);
    assert_eq!(u32_op("UMax32 u32 1, u32 0xffffffff"), 0xffff_ffff);
    assert_eq!(u32_op("SClamp32 u32 0xffffffff, u32 0, u32 3"), 0);
    assert_eq!(u32_op("UClamp32 u32 0xffffffff, u32 0, u32 3"), 3);
    assert_eq!(
        u32_op("SClamp32 u32 0xfffffffb, u32 0xfffffffe, u32 3"),
        0xffff_fffe
    );
    assert_eq!(u32_op("SClamp32 u32 2, u32 0xfffffffe, u32 3"), 2);
    assert_eq!(u32_op("UClamp32 u32 1, u32 2, u32 3"), 2);
    assert_eq!(u32_op("UClamp32 u32 2, u32 1, u32 3"), 2);
}

#[test]
fn comparisons_by_signedness() {
    // Rows: opcode, then the result for (0xffffffff, 1), (1, 0xffffffff) and (5, 5).
    let rows: [(&str, [bool; 3]); 10] = [
        ("SLessThan", [true, false, false]),
        ("ULessThan", [false, true, false]),
        ("IEqual", [false, false, true]),
        ("SLessThanEqual", [true, false, true]),
        ("ULessThanEqual", [false, true, true]),
        ("SGreaterThan", [false, true, false]),
        ("UGreaterThan", [true, false, false]),
        ("INotEqual", [true, true, false]),
        ("SGreaterThanEqual", [false, true, true]),
        ("UGreaterThanEqual", [true, false, true]),
    ];
    let pairs = [("0xffffffff", "1"), ("1", "0xffffffff"), ("5", "5")];
    for (opcode, expected) in rows {
        for ((a, b), want) in pairs.iter().zip(expected) {
            assert_eq!(
                u1_of(eval(&format!("{opcode} u32 {a}, u32 {b}"))),
                want,
                "{opcode} {a} {b}"
            );
        }
    }
}

#[test]
fn logic() {
    let table = [(false, false), (false, true), (true, false), (true, true)];
    for (a, b) in table {
        let operands = format!("u1 {a}, u1 {b}");
        assert_eq!(u1_of(eval(&format!("LogicalOr {operands}"))), a | b);
        assert_eq!(u1_of(eval(&format!("LogicalAnd {operands}"))), a & b);
        assert_eq!(u1_of(eval(&format!("LogicalXor {operands}"))), a ^ b);
    }
    assert!(u1_of(eval("LogicalNot u1 false")));
    assert!(!u1_of(eval("LogicalNot u1 true")));
}

#[test]
fn integer_conversions() {
    assert_eq!(u64_op("ConvertU64U32 u32 0xffffffff"), 0xffff_ffff);
    assert_eq!(u32_op("ConvertU32U64 u64 0x123456789"), 0x2345_6789);
    assert_eq!(eval("ConvertU16U32 u32 0x12345"), Val::U16(0x2345));
    assert_eq!(eval("ConvertU8U32 u32 0x1ff"), Val::U8(0xff));
    assert_eq!(u32_op("ConvertU32U16 u16 0x8000"), 0x8000);
    assert_eq!(u32_op("ConvertU32U8 u8 0x80"), 0x80);
    assert_eq!(u32_op("ConvertS32S16 u16 0x8000"), 0xffff_8000);
    assert_eq!(u32_op("ConvertS32S16 u16 0x7fff"), 0x7fff);
    assert_eq!(u32_op("ConvertS32S8 u8 0x80"), 0xffff_ff80);
    assert_eq!(u32_op("ConvertS32S8 u8 0x7f"), 0x7f);
}

#[test]
fn undefined_values_are_zero() {
    assert_eq!(eval("UndefU1"), Val::U1(false));
    assert_eq!(eval("UndefU8"), Val::U8(0));
    assert_eq!(eval("UndefU16"), Val::U16(0));
    assert_eq!(eval("UndefU32"), Val::U32(0));
    assert_eq!(eval("UndefU64"), Val::U64(0));
}

#[test]
fn selects_identity_and_condition_refs() {
    let body = "    %0 = SelectU32 u1 true, u32 1, u32 2
    %1 = SelectU32 u1 false, u32 1, u32 2
    %2 = Identity %1
    %3 = ConditionRef u1 true
    %4 = SelectU64 u1 false, u64 1, u64 0x200000000
    %5 = SelectU1 u1 true, u1 false, u1 true";
    assert_eq!(value_of(body, 0), Val::U32(1));
    assert_eq!(value_of(body, 1), Val::U32(2));
    assert_eq!(value_of(body, 2), Val::U32(2));
    assert_eq!(value_of(body, 3), Val::U1(true));
    assert_eq!(value_of(body, 4), Val::U64(0x2_0000_0000));
    assert_eq!(value_of(body, 5), Val::U1(false));
}
