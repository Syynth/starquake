//! The float family: arithmetic in every rounding mode, flush-to-zero and FMZ, NaN
//! canonicalisation and payload-keeping bit operations, minimum and maximum, clamping, rounding
//! to integers, the 16- and 64-bit forms and the binary32 functions.

use proptest::prelude::*;

use super::{eval, eval_error, f16_bits, f32_bits, f32_imm, f64_bits, value_of};
use crate::interp::float::{F16_NAN, F32_NAN, F64_NAN};
use crate::testing::config;

/// `1.0` as binary32 bits.
const ONE: u32 = 0x3f80_0000;
/// The binary32 value after `1.0`.
const ONE_UP: u32 = 0x3f80_0001;
/// The binary32 value before `1.0`.
const ONE_DOWN: u32 = 0x3f7f_ffff;
/// The largest finite binary32 value.
const MAX: u32 = 0x7f7f_ffff;
/// Positive infinity in binary32.
const INF: u32 = 0x7f80_0000;
/// The binary32 sign bit.
const SIGN: u32 = 0x8000_0000;

fn f32_op(op: &str) -> u32 {
    f32_bits(eval(op))
}

/// `FPAdd32 a, b` with the flags `flags` (empty for none).
fn add(a: u32, b: u32, flags: &str) -> u32 {
    let flags = if flags.is_empty() {
        String::new()
    } else {
        format!(" {{{flags}}}")
    };
    f32_op(&format!("FPAdd32 {}, {}{flags}", f32_imm(a), f32_imm(b)))
}

#[test]
fn addition_rounds_in_each_mode() {
    assert_eq!(f32_op("FPAdd32 f32 1.5, f32 2.25"), 0x4070_0000);
    // 1 + 2^-24 is a tie between 1 and the next value up.
    let half_ulp = 0x3380_0000;
    assert_eq!(add(ONE, half_ulp, ""), ONE);
    assert_eq!(add(ONE, half_ulp, "rn"), ONE);
    assert_eq!(add(ONE, half_ulp, "rp"), ONE_UP);
    assert_eq!(add(ONE, half_ulp, "rm"), ONE);
    assert_eq!(add(ONE, half_ulp, "rz"), ONE);
    assert_eq!(add(ONE | SIGN, half_ulp | SIGN, "rn"), ONE | SIGN);
    assert_eq!(add(ONE | SIGN, half_ulp | SIGN, "rm"), ONE_UP | SIGN);
    assert_eq!(add(ONE | SIGN, half_ulp | SIGN, "rp"), ONE | SIGN);
    assert_eq!(add(ONE | SIGN, half_ulp | SIGN, "rz"), ONE | SIGN);
    // 1.5 * 2^-23 + 1 ties between 1 + 2^-23 (odd) and 1 + 2^-22 (even): to even goes up.
    assert_eq!(add(ONE_UP, half_ulp, "rn"), 0x3f80_0002);
    // 1 + 2^-24 + 2^-30 is above the tie.
    let above_half_ulp = 0x3381_0000;
    assert_eq!(add(ONE, above_half_ulp, "rn"), ONE_UP);
    assert_eq!(add(ONE, above_half_ulp, "rz"), ONE);
    // 1 - 2^-30: just below a power of two, where the spacing halves.
    let tiny = 0x3080_0000 | SIGN;
    assert_eq!(add(ONE, tiny, "rn"), ONE);
    assert_eq!(add(ONE, tiny, "rp"), ONE);
    assert_eq!(add(ONE, tiny, "rz"), ONE_DOWN);
    assert_eq!(add(ONE, tiny, "rm"), ONE_DOWN);
    // 1 + 2^-80 is not exact even in binary64: the lost bits still decide a directed mode.
    let far = 0x1780_0000;
    assert_eq!(add(ONE, far, "rn"), ONE);
    assert_eq!(add(ONE, far, "rp"), ONE_UP);
    assert_eq!(add(ONE, far, "rz"), ONE);
    assert_eq!(add(ONE, far | SIGN, "rz"), ONE_DOWN);
    assert_eq!(add(ONE, far | SIGN, "rp"), ONE);
}

#[test]
fn overflow_follows_the_rounding_mode() {
    assert_eq!(add(MAX, MAX, "rn"), INF);
    assert_eq!(add(MAX, MAX, "rp"), INF);
    assert_eq!(add(MAX, MAX, "rz"), MAX);
    assert_eq!(add(MAX, MAX, "rm"), MAX);
    assert_eq!(add(MAX | SIGN, MAX | SIGN, "rn"), INF | SIGN);
    assert_eq!(add(MAX | SIGN, MAX | SIGN, "rm"), INF | SIGN);
    assert_eq!(add(MAX | SIGN, MAX | SIGN, "rz"), MAX | SIGN);
    assert_eq!(add(MAX | SIGN, MAX | SIGN, "rp"), MAX | SIGN);
}

#[test]
fn an_exact_zero_sum_is_negative_only_rounding_down() {
    assert_eq!(add(ONE, ONE | SIGN, "rn"), 0);
    assert_eq!(add(ONE, ONE | SIGN, "rz"), 0);
    assert_eq!(add(ONE, ONE | SIGN, "rp"), 0);
    assert_eq!(add(ONE, ONE | SIGN, "rm"), SIGN);
    assert_eq!(add(0, 0, "rm"), 0);
    assert_eq!(add(SIGN, SIGN, "rn"), SIGN);
    assert_eq!(add(0, SIGN, "rn"), 0);
    assert_eq!(add(0, SIGN, "rm"), SIGN);
}

#[test]
fn multiplication_rounds_once() {
    assert_eq!(f32_op("FPMul32 f32 1.5, f32 -2.0"), 0xc040_0000);
    // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24: a tie, to even gives 1 + 2^-11.
    let a = f32_imm(0x3f80_0800);
    assert_eq!(f32_op(&format!("FPMul32 {a}, {a}")), 0x3f80_1000);
    assert_eq!(f32_op(&format!("FPMul32 {a}, {a} {{rp}}")), 0x3f80_1001);
    assert_eq!(f32_op(&format!("FPMul32 {a}, {a} {{rz}}")), 0x3f80_1000);
    assert_eq!(f32_op("FPMul32 f32 -0.0, f32 5.0"), SIGN);
}

#[test]
fn fused_multiply_add_rounds_once() {
    // (1 + 2^-12)^2 - 1 = 2^-11 + 2^-24 exactly; a separate multiply would lose the 2^-24.
    let a = f32_imm(0x3f80_0800);
    let body = format!(
        "    %0 = FPFma32 {a}, {a}, f32 -1.0
    %1 = FPMul32 {a}, {a}
    %2 = FPAdd32 %1, f32 -1.0"
    );
    assert_eq!(f32_bits(value_of(&body, 0)), 0x3a00_0400);
    assert_eq!(f32_bits(value_of(&body, 2)), 0x3a00_0000);
    // A rounding mode applies to the one rounding: 1 * 1 + 2^-24 ties.
    let half_ulp = f32_imm(0x3380_0000);
    assert_eq!(
        f32_op(&format!("FPFma32 f32 1.0, f32 1.0, {half_ulp}")),
        ONE
    );
    assert_eq!(
        f32_op(&format!("FPFma32 f32 1.0, f32 1.0, {half_ulp} {{rp}}")),
        ONE_UP
    );
    // Infinities: inf * 2 + 1 = inf; inf * 1 - inf is a NaN.
    let inf = f32_imm(INF);
    assert_eq!(f32_op(&format!("FPFma32 {inf}, f32 2.0, f32 1.0")), INF);
    let minus_inf = f32_imm(INF | SIGN);
    assert_eq!(
        u64::from(f32_op(&format!("FPFma32 {inf}, f32 1.0, {minus_inf}"))),
        F32_NAN
    );
    assert_eq!(f32_op(&format!("FPFma32 f32 2.0, f32 3.0, {inf}")), INF);
    // (1 + 2^-12)^2 + 2^-80 = 1 + 2^-11 + 2^-24 + 2^-80: the binary64 sum is the tie
    // 1 + 2^-11 + 2^-24, and the lost 2^-80 breaks it upward (to even would go down).
    let far = f32_imm(0x1780_0000);
    let minus_far = f32_imm(0x9780_0000);
    assert_eq!(f32_op(&format!("FPFma32 {a}, {a}, {far}")), 0x3f80_1001);
    assert_eq!(
        f32_op(&format!("FPFma32 {a}, {a}, {minus_far}")),
        0x3f80_1000
    );
    // 1 * 1 + 2^-80: exactly representable in binary64 only as 1 plus a lost error.
    assert_eq!(
        f32_op(&format!("FPFma32 f32 1.0, f32 1.0, {far} {{rp}}")),
        ONE_UP
    );
    assert_eq!(
        f32_op(&format!("FPFma32 f32 1.0, f32 1.0, {far} {{rm}}")),
        ONE
    );
    assert_eq!(
        f32_op(&format!("FPFma32 f32 1.0, f32 1.0, {minus_far} {{rm}}")),
        ONE_DOWN
    );
    assert_eq!(
        f32_op(&format!("FPFma32 f32 1.0, f32 1.0, {minus_far} {{rp}}")),
        ONE
    );
}

#[test]
fn ftz_flushes_operands_and_results() {
    let smallest = f32_imm(1);
    assert_eq!(f32_op(&format!("FPAdd32 {smallest}, f32 0.0")), 1);
    assert_eq!(
        f32_op(&format!("FPAdd32 {smallest}, f32 0.0 {{fmz_none}}")),
        1
    );
    assert_eq!(f32_op(&format!("FPAdd32 {smallest}, f32 0.0 {{ftz}}")), 0);
    assert_eq!(f32_op(&format!("FPAdd32 {smallest}, f32 0.0 {{fmz}}")), 0);
    // A normal times a half: the result is denormal, and is flushed keeping its sign.
    let min_normal = f32_imm(0x0080_0000);
    assert_eq!(
        f32_op(&format!("FPMul32 {min_normal}, f32 0.5")),
        0x0040_0000
    );
    assert_eq!(f32_op(&format!("FPMul32 {min_normal}, f32 0.5 {{ftz}}")), 0);
    assert_eq!(
        f32_op(&format!("FPMul32 {min_normal}, f32 -0.5 {{ftz}}")),
        SIGN
    );
    // Other operations with controls flush too.
    assert_eq!(f32_op(&format!("FPMax32 {smallest}, f32 0.0")), 1);
    assert_eq!(f32_op(&format!("FPMax32 {smallest}, f32 -0.0 {{ftz}}")), 0);
    assert_eq!(f32_op(&format!("FPFloor32 {smallest} {{ftz}}")), 0);
    assert_eq!(
        f32_op(&format!("FPFma32 {smallest}, f32 1.0, f32 0.0 {{ftz}}")),
        0
    );
    assert_eq!(f32_op(&format!("FPFma32 {smallest}, f32 1.0, f32 0.0")), 1);
}

#[test]
fn fmz_makes_a_zero_factor_win() {
    let inf = f32_imm(INF);
    let nan = f32_imm(0x7fc0_0001);
    assert_eq!(
        u64::from(f32_op(&format!("FPMul32 {inf}, f32 0.0"))),
        F32_NAN
    );
    assert_eq!(f32_op(&format!("FPMul32 {inf}, f32 0.0 {{fmz}}")), 0);
    assert_eq!(f32_op(&format!("FPMul32 {nan}, f32 -0.0 {{fmz}}")), 0);
    assert_eq!(
        u64::from(f32_op(&format!("FPMul32 {inf}, f32 0.0 {{ftz}}"))),
        F32_NAN
    );
    assert_eq!(
        f32_op(&format!("FPFma32 {inf}, f32 0.0, f32 2.0 {{fmz}}")),
        0x4000_0000
    );
    assert_eq!(
        u64::from(f32_op(&format!("FPFma32 {inf}, f32 0.0, f32 2.0"))),
        F32_NAN
    );
}

#[test]
fn nans_are_canonical_except_through_bit_operations() {
    let quiet = f32_imm(0x7fc0_0001);
    let negative = f32_imm(0xffc0_0001);
    let inf = f32_imm(INF);
    let minus_inf = f32_imm(INF | SIGN);
    for op in [
        format!("FPAdd32 {quiet}, f32 1.0"),
        format!("FPAdd32 f32 1.0, {negative}"),
        format!("FPAdd32 {inf}, {minus_inf}"),
        format!("FPMul32 {negative}, f32 2.0"),
        format!("FPFma32 f32 1.0, f32 1.0, {quiet}"),
        format!("FPFloor32 {negative}"),
        format!("FPRecip32 {negative}"),
        "FPSqrt f32 -1.0".to_owned(),
    ] {
        assert_eq!(u64::from(f32_op(&op)), F32_NAN, "{op}");
    }
    assert_eq!(f32_op(&format!("FPNeg32 {quiet}")), 0xffc0_0001);
    assert_eq!(f32_op(&format!("FPAbs32 {negative}")), 0x7fc0_0001);
    assert_eq!(f32_op("FPNeg32 f32 1.0"), ONE | SIGN);
    assert_eq!(f32_op("FPAbs32 f32 -2.0"), 0x4000_0000);
    assert_eq!(
        f32_bits(value_of(
            &format!("    %0 = SelectF32 u1 true, {negative}, f32 1.0"),
            0
        )),
        0xffc0_0001
    );
}

#[test]
fn minimum_and_maximum_prefer_numbers_and_order_zeros() {
    let nan = f32_imm(0x7fc0_0001);
    assert_eq!(f32_op(&format!("FPMax32 {nan}, f32 2.0")), 0x4000_0000);
    assert_eq!(f32_op(&format!("FPMax32 f32 2.0, {nan}")), 0x4000_0000);
    assert_eq!(f32_op(&format!("FPMin32 {nan}, f32 2.0")), 0x4000_0000);
    assert_eq!(u64::from(f32_op(&format!("FPMax32 {nan}, {nan}"))), F32_NAN);
    assert_eq!(f32_op("FPMax32 f32 1.0, f32 2.0"), 0x4000_0000);
    assert_eq!(f32_op("FPMax32 f32 2.0, f32 1.0"), 0x4000_0000);
    assert_eq!(f32_op("FPMin32 f32 1.0, f32 2.0"), ONE);
    assert_eq!(f32_op("FPMin32 f32 2.0, f32 1.0"), ONE);
    assert_eq!(f32_op("FPMax32 f32 -0.0, f32 0.0"), 0);
    assert_eq!(f32_op("FPMax32 f32 0.0, f32 -0.0"), 0);
    assert_eq!(f32_op("FPMin32 f32 -0.0, f32 0.0"), SIGN);
    assert_eq!(f32_op("FPMin32 f32 0.0, f32 -0.0"), SIGN);
    assert_eq!(
        f64_bits(eval("FPMax64 f64 -3.0, f64 -4.0")),
        (-3.0_f64).to_bits()
    );
    assert_eq!(
        f64_bits(eval("FPMin64 f64 -3.0, f64 -4.0")),
        (-4.0_f64).to_bits()
    );
}

#[test]
fn clamp_and_saturate_take_a_nan_to_the_low_bound() {
    let nan = f32_imm(0x7fc0_0001);
    assert_eq!(f32_op(&format!("FPClamp32 {nan}, f32 1.0, f32 2.0")), ONE);
    assert_eq!(f32_op("FPClamp32 f32 5.0, f32 1.0, f32 2.0"), 0x4000_0000);
    assert_eq!(f32_op("FPClamp32 f32 0.0, f32 1.0, f32 2.0"), ONE);
    assert_eq!(f32_op("FPClamp32 f32 1.5, f32 1.0, f32 2.0"), 0x3fc0_0000);
    assert_eq!(f32_op(&format!("FPSaturate32 {nan}")), 0);
    assert_eq!(f32_op("FPSaturate32 f32 2.0"), ONE);
    assert_eq!(f32_op("FPSaturate32 f32 -1.0"), 0);
    assert_eq!(f32_op("FPSaturate32 f32 -0.0"), 0);
    assert_eq!(f32_op("FPSaturate32 f32 0.5"), 0x3f00_0000);
    assert_eq!(f64_bits(eval("FPSaturate64 f64 7.0")), 1.0_f64.to_bits());
    assert_eq!(
        f64_bits(eval("FPClamp64 f64 -7.0, f64 -1.0, f64 1.0")),
        (-1.0_f64).to_bits()
    );
}

#[test]
fn rounding_to_integers() {
    assert_eq!(f32_op("FPRoundEven32 f32 2.5"), 0x4000_0000);
    assert_eq!(f32_op("FPRoundEven32 f32 3.5"), 0x4080_0000);
    assert_eq!(f32_op("FPRoundEven32 f32 -2.5"), 0xc000_0000);
    assert_eq!(f32_op("FPRoundEven32 f32 -0.25"), SIGN);
    assert_eq!(f32_op("FPFloor32 f32 -1.5"), 0xc000_0000);
    assert_eq!(f32_op("FPFloor32 f32 1.5"), ONE);
    assert_eq!(f32_op("FPCeil32 f32 -1.5"), ONE | SIGN);
    assert_eq!(f32_op("FPCeil32 f32 -0.5"), SIGN);
    assert_eq!(f32_op("FPCeil32 f32 1.25"), 0x4000_0000);
    assert_eq!(f32_op("FPTrunc32 f32 -1.75"), ONE | SIGN);
    assert_eq!(f32_op("FPTrunc32 f32 1.75"), ONE);
    assert_eq!(f64_bits(eval("FPFloor64 f64 -0.5")), (-1.0_f64).to_bits());
    assert_eq!(f64_bits(eval("FPRoundEven64 f64 0.5")), 0);
}

/// The value `%2` of `op` applied to the binary16 values with bits `a` and `b`.
fn f16_binary(op: &str, a: u16, b: u16) -> u16 {
    let body = format!(
        "    %0 = BitCastF16U16 u16 {a:#x}
    %1 = BitCastF16U16 u16 {b:#x}
    %2 = {op} %0, %1"
    );
    f16_bits(value_of(&body, 2))
}

#[test]
fn half_precision_arithmetic() {
    assert_eq!(f16_binary("FPAdd16", 0x3c00, 0x3c00), 0x4000);
    assert_eq!(f16_binary("FPMul16", 0x3e00, 0xc000), 0xc200); // 1.5 * -2 = -3
    // The smallest denormal survives (PR #4588's half FMA test), unless flushed.
    assert_eq!(f16_binary("FPAdd16", 0x0001, 0x0000), 0x0001);
    let body = "    %0 = BitCastF16U16 u16 0x1
    %1 = BitCastF16U16 u16 0x0
    %2 = FPAdd16 %0, %1 {ftz}";
    assert_eq!(f16_bits(value_of(body, 2)), 0);
    // A NaN operand gives PR #4588's canonical half NaN.
    assert_eq!(u64::from(f16_binary("FPAdd16", 0x7e01, 0x3c00)), F16_NAN);
    // 65504 + 16 ties between the largest finite half and 65536: to even overflows.
    assert_eq!(f16_binary("FPAdd16", 0x7bff, 0x4c00), 0x7c00);
    // 2049 is not a half: 2048 + 1 ties to 2048 (even), up with rp.
    assert_eq!(f16_binary("FPAdd16", 0x6800, 0x3c00), 0x6800);
    let up = "    %0 = BitCastF16U16 u16 0x6800
    %1 = BitCastF16U16 u16 0x3c00
    %2 = FPAdd16 %0, %1 {rp}
    %3 = FPAdd16 %0, %1 {rz}
    %4 = BitCastF16U16 u16 0x7bff
    %5 = BitCastF16U16 u16 0x4c00
    %6 = FPAdd16 %4, %5 {rz}";
    assert_eq!(f16_bits(value_of(up, 2)), 0x6801);
    assert_eq!(f16_bits(value_of(up, 3)), 0x6800);
    assert_eq!(f16_bits(value_of(up, 6)), 0x7bff);
}

#[test]
fn half_precision_fma_neg_abs_and_rounding() {
    let body = "    %0 = BitCastF16U16 u16 0x3c01
    %1 = BitCastF16U16 u16 0x3bfe
    %2 = BitCastF16U16 u16 0xbc00
    %3 = FPFma16 %0, %1, %2
    %4 = FPMul16 %0, %1
    %5 = FPAdd16 %4, %2
    %6 = FPNeg16 %0
    %7 = FPAbs16 %2
    %8 = FPFloor16 %1
    %9 = FPCeil16 %1
    %10 = FPClamp16 %6, %2, %0
    %11 = FPSaturate16 %6
    %12 = FPRoundEven16 %0
    %13 = FPTrunc16 %6";
    // (1 + 2^-10) * (1 - 2^-10) - 1 = -2^-20: kept by the fused form (PR #4588's fused case),
    // lost by a separate multiply, which rounds the product to 1.
    assert_eq!(f16_bits(value_of(body, 3)), 0x8010);
    assert_eq!(f16_bits(value_of(body, 5)), 0);
    assert_eq!(f16_bits(value_of(body, 6)), 0xbc01);
    assert_eq!(f16_bits(value_of(body, 7)), 0x3c00);
    assert_eq!(f16_bits(value_of(body, 8)), 0);
    assert_eq!(f16_bits(value_of(body, 9)), 0x3c00);
    assert_eq!(f16_bits(value_of(body, 10)), 0xbc00);
    assert_eq!(f16_bits(value_of(body, 11)), 0);
    assert_eq!(f16_bits(value_of(body, 12)), 0x3c00);
    assert_eq!(f16_bits(value_of(body, 13)), 0xbc00);
}

#[test]
fn double_precision_rounds_to_nearest_and_refuses_directed_rounding() {
    assert_eq!(
        f64_bits(eval("FPAdd64 f64 1.0, f64 2.0")),
        3.0_f64.to_bits()
    );
    assert_eq!(
        f64_bits(eval("FPMul64 f64 1.5, f64 -2.0")),
        (-3.0_f64).to_bits()
    );
    // (1 + 2^-30)^2 - 1 = 2^-29 + 2^-60 exactly, which the fused form keeps.
    let a = format!("f64 {:#x}", (1.0_f64 + 2.0_f64.powi(-30)).to_bits());
    let fused = f64_bits(eval(&format!("FPFma64 {a}, {a}, f64 -1.0")));
    assert_eq!(fused, (2.0_f64.powi(-29) + 2.0_f64.powi(-60)).to_bits());
    assert_eq!(f64_bits(eval("FPNeg64 f64 2.0")), (-2.0_f64).to_bits());
    assert_eq!(f64_bits(eval("FPAbs64 f64 -2.0")), 2.0_f64.to_bits());
    let nan = format!("f64 {:#x}", 0x7ff8_0000_0000_0001_u64);
    assert_eq!(f64_bits(eval(&format!("FPAdd64 {nan}, f64 1.0"))), F64_NAN);
    assert_eq!(f64_bits(eval(&format!("FPMul64 {nan}, f64 1.0"))), F64_NAN);
    // The smallest denormal survives without controls and is flushed with ftz.
    assert_eq!(f64_bits(eval("FPAdd64 f64 0x1, f64 0.0")), 1);
    assert_eq!(f64_bits(eval("FPAdd64 f64 0x1, f64 0.0 {ftz}")), 0);
    let max = format!("f64 {:#x}", f64::MAX.to_bits());
    let toward_zero_overflow = format!("FPAdd64 {max}, {max} {{rz}}");
    let downward_overflow = format!("FPAdd64 {max}, {max} {{rm}}");
    for (op, opcode) in [
        ("FPAdd64 f64 1.0, f64 2.0 {rz}", "FPAdd64"),
        ("FPMul64 f64 1.0, f64 2.0 {rm}", "FPMul64"),
        ("FPFma64 f64 1.0, f64 2.0, f64 3.0 {rp}", "FPFma64"),
        // A finite sum that overflows: Rz and Rm would give the largest finite value, not the
        // infinity that rounding to nearest gives, so it is refused too.
        (toward_zero_overflow.as_str(), "FPAdd64"),
        (downward_overflow.as_str(), "FPAdd64"),
    ] {
        assert_eq!(
            eval_error(op),
            format!("%0: the interpreter does not support directed rounding on {opcode}")
        );
    }
    assert_eq!(
        f64_bits(eval("FPAdd64 f64 1.0, f64 2.0 {rn}")),
        3.0_f64.to_bits()
    );
}

#[test]
fn division_square_roots_and_transcendentals() {
    assert_eq!(f32_op("FPRecip32 f32 4.0"), 0x3e80_0000);
    assert_eq!(f32_op("FPRecip32 f32 3.0"), 0x3eaa_aaab);
    assert_eq!(f32_op("FPRecip32 f32 0.0"), INF);
    assert_eq!(f32_op("FPRecip32 f32 -0.0"), INF | SIGN);
    assert_eq!(f32_op("FPRecipSqrt32 f32 4.0"), 0x3f00_0000);
    assert_eq!(f32_op("FPSqrt f32 4.0"), 0x4000_0000);
    assert_eq!(f32_op("FPSqrt f32 2.0"), 0x3fb5_04f3);
    assert_eq!(f32_op("FPSin f32 0.0"), 0);
    assert_eq!(f32_op("FPCos f32 0.0"), ONE);
    assert_eq!(f32_op("FPExp2 f32 3.0"), 0x4100_0000);
    assert_eq!(f32_op("FPExp2 f32 -200.0"), 0);
    assert_eq!(f32_op("FPLog2 f32 8.0"), 0x4040_0000);
    assert_eq!(f32_op("FPLog2 f32 0.0"), INF | SIGN);
    assert_eq!(u64::from(f32_op("FPLog2 f32 -1.0")), F32_NAN);
    assert_eq!(f64_bits(eval("FPRecip64 f64 2.0")), 0.5_f64.to_bits());
    assert_eq!(f64_bits(eval("FPRecipSqrt64 f64 4.0")), 0.5_f64.to_bits());
    let nan = format!("f64 {:#x}", 0xfff8_0000_0000_0001_u64);
    assert_eq!(f64_bits(eval(&format!("FPRecip64 {nan}"))), F64_NAN);
}

#[test]
fn is_nan_in_every_width() {
    let body = "    %0 = BitCastF16U16 u16 0x7c01
    %1 = BitCastF16U16 u16 0x7c00
    %2 = FPIsNan16 %0
    %3 = FPIsNan16 %1
    %4 = FPIsNan32 f32 0x7f800001
    %5 = FPIsNan32 f32 0x7f800000
    %6 = FPIsNan64 f64 0x7ff0000000000001
    %7 = FPIsNan64 f64 0x7ff0000000000000";
    let expected = [
        (2, true),
        (3, false),
        (4, true),
        (5, false),
        (6, true),
        (7, false),
    ];
    for (id, want) in expected {
        assert_eq!(value_of(body, id), crate::interp::Val::U1(want), "%{id}");
    }
}

/// The interpreter's result of `op` on the binary32 operands `operands`, as a host float.
fn host(op: &str, operands: &[f32]) -> u32 {
    let args: Vec<String> = operands.iter().map(|x| f32_imm(x.to_bits())).collect();
    f32_op(&format!("{op} {}", args.join(", ")))
}

/// Host binary32 results with NaNs made canonical, for comparison.
fn canonical(x: f32) -> u32 {
    if x.is_nan() {
        u32::try_from(F32_NAN).expect("32 bits")
    } else {
        x.to_bits()
    }
}

proptest! {
    #![proptest_config(config())]

    /// Round to nearest agrees with the host's IEEE binary32 operations, which are exactly
    /// rounded the same way, for any bits (denormals, infinities and NaNs included).
    #[test]
    fn nearest_matches_host_binary32(a in any::<u32>(), b in any::<u32>(), c in any::<u32>()) {
        let (a, b, c) = (f32::from_bits(a), f32::from_bits(b), f32::from_bits(c));
        prop_assert_eq!(host("FPAdd32", &[a, b]), canonical(a + b));
        prop_assert_eq!(host("FPMul32", &[a, b]), canonical(a * b));
        prop_assert_eq!(host("FPFma32", &[a, b, c]), canonical(a.mul_add(b, c)));
    }

    /// Directed rounding of an exact binary64 product brackets it: down at or below, up at or
    /// above, the two adjacent or equal, toward zero the one nearer zero.
    #[test]
    fn directed_products_bracket_the_exact_value(a in any::<u32>(), b in any::<u32>()) {
        let (x, y) = (f32::from_bits(a), f32::from_bits(b));
        let exact = f64::from(x) * f64::from(y);
        prop_assume!(exact.is_finite() && exact != 0.0);
        let mode = |flag: &str| {
            f32_op(&format!("FPMul32 {}, {} {{{flag}}}", f32_imm(a), f32_imm(b)))
        };
        let down = f64::from(f32::from_bits(mode("rm")));
        let up = f64::from(f32::from_bits(mode("rp")));
        prop_assert!(down <= exact && exact <= up);
        let (down_bits, up_bits) = (mode("rm"), mode("rp"));
        prop_assert!(
            down_bits == up_bits || f32::from_bits(down_bits).next_up().to_bits() == up_bits
        );
        let toward_zero = if exact > 0.0 { down_bits } else { up_bits };
        prop_assert_eq!(mode("rz"), toward_zero);
    }
}

#[test]
fn canonical_nan_constants_are_the_documented_bits() {
    assert_eq!(F16_NAN, 0x7fff);
    assert_eq!(F32_NAN, 0x7fff_ffff);
    assert_eq!(F64_NAN, 0x7fff_ffff_ffff_ffff);
}

/// `FSwizzleAdd a, b, swizzle` per Eden's `EmitFSwizzleAdd`: the lane's two bits of the
/// swizzle (at `(lane_id & 3) * 2`) pick `-a - b`, `a - b`, `-a + b` or `-b`. With `a = 3`,
/// `b = 5`: -8, -2, 2, -5. The table's multipliers are Eden's `fswzadd_lut_a` and
/// `fswzadd_lut_b`, written out here as literals.
#[test]
fn swizzled_add_follows_the_lane() {
    use crate::interp::Machine;
    use crate::interp::exec::SWIZZLE_ADD_FACTORS;
    assert_eq!(
        SWIZZLE_ADD_FACTORS,
        [
            (0xbf80_0000, 0xbf80_0000),
            (0x3f80_0000, 0xbf80_0000),
            (0xbf80_0000, 0x3f80_0000),
            (0x0000_0000, 0xbf80_0000),
        ]
    );
    let by_index = [
        (-8.0_f32).to_bits(),
        (-2.0_f32).to_bits(),
        2.0_f32.to_bits(),
        (-5.0_f32).to_bits(),
    ];
    // The swizzle 0b11_10_01_00 gives lane n index n; 0xa5 (`~0x5a`) gives 1, 1, 2, 2.
    for (swizzle, indices) in [(0xe4_u32, [0_usize, 1, 2, 3]), (0xa5, [1, 1, 2, 2])] {
        for lane in 0..8_u32 {
            let text = format!(
                "block0:\n    %0 = FSwizzleAdd f32 3.0, f32 5.0, u32 {swizzle:#x} {{rn, fmz_none}}\n    return\n"
            );
            let mut machine = Machine::new();
            machine.system.lane_id = lane;
            let finished = super::run(&text, &mut machine).expect("runs");
            let slot = usize::try_from(lane % 4).expect("a lane");
            let expected = by_index[indices[slot]];
            assert_eq!(
                f32_bits(finished.value(0).expect("a value")),
                expected,
                "lane {lane}"
            );
        }
    }
    // The controls apply: `1 - (-2^-24)` rounds up only toward positive infinity.
    let tiny = f32_imm(0xb380_0000);
    let up = eval(&format!("FSwizzleAdd f32 1.0, {tiny}, u32 0x1 {{rp}}"));
    assert_eq!(f32_bits(up), ONE_UP);
    let near = eval(&format!("FSwizzleAdd f32 1.0, {tiny}, u32 0x1 {{rn}}"));
    assert_eq!(f32_bits(near), ONE);
}
