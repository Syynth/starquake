//! Conversions: float to integer (truncating, saturating, NaN to zero, 16-bit results
//! extended), integer to float in each rounding mode with 8- and 16-bit sources taken from the
//! low bits, float to float with rounding and flushing, bit casts and packing.

use super::{eval, f16_bits, f32_bits, f64_bits, u32_of, u64_of, value_of};
use crate::interp::Val;
use crate::interp::float::{F16_NAN, F32_NAN};

fn u32_op(op: &str) -> u32 {
    u32_of(eval(op))
}

fn u64_op(op: &str) -> u64 {
    u64_of(eval(op))
}

#[test]
fn float_to_integer_truncates_and_saturates() {
    assert_eq!(u32_op("ConvertS32F32 f32 1.9"), 1);
    assert_eq!(u32_op("ConvertS32F32 f32 -1.9"), 0xffff_ffff);
    assert_eq!(u32_op("ConvertS32F32 f32 3e9"), 0x7fff_ffff);
    assert_eq!(u32_op("ConvertS32F32 f32 -3e9"), 0x8000_0000);
    assert_eq!(u32_op("ConvertS32F32 f32 0x7f800000"), 0x7fff_ffff);
    assert_eq!(u32_op("ConvertS32F32 f32 0x7fc00001"), 0);
    assert_eq!(u32_op("ConvertU32F32 f32 -1.0"), 0);
    assert_eq!(u32_op("ConvertU32F32 f32 5e9"), 0xffff_ffff);
    assert_eq!(u32_op("ConvertU32F32 f32 0xffc00000"), 0);
    // A 16-bit result saturates to 16 bits and is sign- or zero-extended.
    assert_eq!(u32_op("ConvertS16F32 f32 -40000.0"), 0xffff_8000);
    assert_eq!(u32_op("ConvertS16F32 f32 40000.0"), 0x7fff);
    assert_eq!(u32_op("ConvertS16F32 f32 -1.0"), 0xffff_ffff);
    assert_eq!(u32_op("ConvertU16F32 f32 70000.0"), 0xffff);
    assert_eq!(u32_op("ConvertU16F64 f64 -3.0"), 0);
    assert_eq!(u32_op("ConvertS16F64 f64 -32768.9"), 0xffff_8000);
    assert_eq!(u32_op("ConvertS32F64 f64 -2147483648.9"), 0x8000_0000);
    assert_eq!(u32_op("ConvertU32F64 f64 4294967295.9"), 0xffff_ffff);
    assert_eq!(u64_op("ConvertS64F64 f64 1e19"), 0x7fff_ffff_ffff_ffff);
    assert_eq!(u64_op("ConvertS64F64 f64 -1e19"), 0x8000_0000_0000_0000);
    assert_eq!(u64_op("ConvertS64F32 f32 -2.5"), 0xffff_ffff_ffff_fffe);
    assert_eq!(u64_op("ConvertU64F32 f32 -5.0"), 0);
    assert_eq!(u64_op("ConvertU64F64 f64 1e20"), u64::MAX);
    let half = |bits: u16, op: &str| {
        u32_of(value_of(
            &format!("    %0 = BitCastF16U16 u16 {bits:#x}\n    %1 = {op} %0"),
            1,
        ))
    };
    assert_eq!(half(0xc500, "ConvertS32F16"), 0xffff_fffb); // -5.0
    assert_eq!(half(0xc500, "ConvertS16F16"), 0xffff_fffb);
    assert_eq!(half(0xc500, "ConvertU16F16"), 0);
    assert_eq!(half(0x7bff, "ConvertU32F16"), 65504);
    assert_eq!(half(0x7e00, "ConvertS32F16"), 0);
    let wide = value_of(
        "    %0 = BitCastF16U16 u16 0xfc00\n    %1 = ConvertS64F16 %0\n    %2 = ConvertU64F16 %0",
        1,
    );
    assert_eq!(wide, Val::U64(0x8000_0000_0000_0000));
}

#[test]
fn integer_to_float_rounds_in_each_mode() {
    // 2^24 + 1 lies between 2^24 and 2^24 + 2.
    assert_eq!(f32_bits(eval("ConvertF32S32 u32 0x1000001")), 0x4b80_0000);
    assert_eq!(
        f32_bits(eval("ConvertF32S32 u32 0x1000001 {rp}")),
        0x4b80_0001
    );
    assert_eq!(
        f32_bits(eval("ConvertF32S32 u32 0x1000001 {rm}")),
        0x4b80_0000
    );
    assert_eq!(
        f32_bits(eval("ConvertF32S32 u32 0x1000001 {rz}")),
        0x4b80_0000
    );
    assert_eq!(
        f32_bits(eval("ConvertF32S32 u32 0xfeffffff {rm}")),
        0xcb80_0001
    );
    assert_eq!(
        f32_bits(eval("ConvertF32S32 u32 0xfeffffff {rp}")),
        0xcb80_0000
    );
    assert_eq!(
        f32_bits(eval("ConvertF32S32 u32 0xfeffffff {rz}")),
        0xcb80_0000
    );
    // 2^24 + 3 ties between 2^24 + 2 (odd) and 2^24 + 4 (even).
    assert_eq!(f32_bits(eval("ConvertF32U32 u32 0x1000003")), 0x4b80_0002);
    assert_eq!(f32_bits(eval("ConvertF32U32 u32 0xffffffff")), 0x4f80_0000);
    assert_eq!(
        f32_bits(eval("ConvertF32U32 u32 0xffffffff {rz}")),
        0x4f7f_ffff
    );
    assert_eq!(f32_bits(eval("ConvertF32S32 u32 0x0")), 0);
    assert_eq!(
        f32_bits(eval("ConvertF32S64 u64 0xffffffffffffffff")),
        0xbf80_0000
    );
    assert_eq!(
        f32_bits(eval("ConvertF32U64 u64 0xffffffffffffffff")),
        0x5f80_0000
    );
    assert_eq!(
        f32_bits(eval("ConvertF32U64 u64 0xffffffffffffffff {rz}")),
        0x5f7f_ffff
    );
    assert_eq!(
        f64_bits(eval("ConvertF64U64 u64 0xffffffffffffffff")),
        0x43f0_0000_0000_0000
    );
    assert_eq!(
        f64_bits(eval("ConvertF64U64 u64 0xffffffffffffffff {rz}")),
        0x43ef_ffff_ffff_ffff
    );
    assert_eq!(
        f64_bits(eval("ConvertF64S64 u64 0x8000000000000001 {rp}")),
        0xc3df_ffff_ffff_ffff
    );
    assert_eq!(
        f64_bits(eval("ConvertF64S64 u64 0x8000000000000001")),
        0xc3e0_0000_0000_0000
    );
    assert_eq!(
        f64_bits(eval("ConvertF64S32 u32 0xffffffff")),
        (-1.0_f64).to_bits()
    );
    assert_eq!(
        f64_bits(eval("ConvertF64U32 u32 0xffffffff")),
        4_294_967_295.0_f64.to_bits()
    );
    // Half results: 2049 ties to 2048; 70000 overflows unless rounded toward zero.
    assert_eq!(f16_bits(eval("ConvertF16S32 u32 2049")), 0x6800);
    assert_eq!(f16_bits(eval("ConvertF16S32 u32 2049 {rp}")), 0x6801);
    assert_eq!(f16_bits(eval("ConvertF16U32 u32 70000")), 0x7c00);
    assert_eq!(f16_bits(eval("ConvertF16U32 u32 70000 {rz}")), 0x7bff);
    assert_eq!(f16_bits(eval("ConvertF16U64 u64 70000 {rm}")), 0x7bff);
    assert_eq!(
        f16_bits(eval("ConvertF16S64 u64 0xfffffffffffeee90")),
        0xfc00
    ); // -70000
}

#[test]
fn small_integer_sources_take_the_low_bits() {
    assert_eq!(f32_bits(eval("ConvertF32S8 u32 0x180")), 0xc300_0000); // -128
    assert_eq!(f32_bits(eval("ConvertF32U8 u32 0x180")), 0x4300_0000); // 128
    assert_eq!(f32_bits(eval("ConvertF32S16 u32 0x18000")), 0xc700_0000); // -32768
    assert_eq!(f32_bits(eval("ConvertF32U16 u32 0x18000")), 0x4700_0000); // 32768
    assert_eq!(f16_bits(eval("ConvertF16S8 u32 0xff")), 0xbc00); // -1
    assert_eq!(f16_bits(eval("ConvertF16U8 u32 0x101")), 0x3c00); // 1
    assert_eq!(f16_bits(eval("ConvertF16S16 u32 0xfffe")), 0xc000); // -2
    assert_eq!(f16_bits(eval("ConvertF16U16 u32 0x10002")), 0x4000); // 2
    assert_eq!(f64_bits(eval("ConvertF64S8 u32 0x7f")), 127.0_f64.to_bits());
    assert_eq!(f64_bits(eval("ConvertF64U8 u32 0xff")), 255.0_f64.to_bits());
    assert_eq!(
        f64_bits(eval("ConvertF64S16 u32 0x8000")),
        (-32768.0_f64).to_bits()
    );
    assert_eq!(
        f64_bits(eval("ConvertF64U16 u32 0x8000")),
        32768.0_f64.to_bits()
    );
}

#[test]
fn float_to_float_rounds_and_flushes() {
    let just_above_one = format!("f64 {:#x}", (1.0_f64 + 2.0_f64.powi(-30)).to_bits());
    assert_eq!(
        f32_bits(eval(&format!("ConvertF32F64 {just_above_one}"))),
        0x3f80_0000
    );
    assert_eq!(
        f32_bits(eval(&format!("ConvertF32F64 {just_above_one} {{rp}}"))),
        0x3f80_0001
    );
    assert_eq!(f32_bits(eval("ConvertF32F64 f64 1e300")), 0x7f80_0000);
    assert_eq!(f32_bits(eval("ConvertF32F64 f64 1e300 {rz}")), 0x7f7f_ffff);
    assert_eq!(f32_bits(eval("ConvertF32F64 f64 -1e300 {rp}")), 0xff7f_ffff);
    assert_eq!(
        u64::from(f32_bits(eval("ConvertF32F64 f64 0x7ff8000000000001"))),
        F32_NAN
    );
    assert_eq!(
        f64_bits(eval("ConvertF64F32 f32 0x1")),
        2.0_f64.powi(-149).to_bits()
    );
    assert_eq!(f64_bits(eval("ConvertF64F32 f32 0x1 {ftz}")), 0);
    assert_eq!(f16_bits(eval("ConvertF16F32 f32 1.0")), 0x3c00);
    assert_eq!(f16_bits(eval("ConvertF16F32 f32 65520.0")), 0x7c00);
    assert_eq!(f16_bits(eval("ConvertF16F32 f32 65520.0 {rz}")), 0x7bff);
    // Half of the smallest half denormal ties to zero; three quarters rounds up to it.
    assert_eq!(f16_bits(eval("ConvertF16F32 f32 0x33000000")), 0);
    assert_eq!(f16_bits(eval("ConvertF16F32 f32 0x33400000")), 1);
    assert_eq!(f16_bits(eval("ConvertF16F32 f32 0x33400000 {ftz}")), 0);
    assert_eq!(f16_bits(eval("ConvertF16F32 f32 0x1 {rp}")), 1);
    assert_eq!(f16_bits(eval("ConvertF16F32 f32 0x1 {rp, ftz}")), 0);
    assert_eq!(f16_bits(eval("ConvertF16F32 f32 0xbf800000")), 0xbc00);
    assert_eq!(
        u64::from(f16_bits(eval("ConvertF16F32 f32 0x7fc00001"))),
        F16_NAN
    );
    let widen = |bits: u16| {
        f32_bits(value_of(
            &format!("    %0 = BitCastF16U16 u16 {bits:#x}\n    %1 = ConvertF32F16 %0"),
            1,
        ))
    };
    assert_eq!(widen(0x0001), 0x3380_0000);
    assert_eq!(widen(0xfc00), 0xff80_0000);
    assert_eq!(widen(0x7bff), 0x477f_e000);
    assert_eq!(u64::from(widen(0x7e01)), F32_NAN);
}

#[test]
fn bit_casts_and_packing() {
    assert_eq!(eval("BitCastU32F32 f32 1.0"), Val::U32(0x3f80_0000));
    assert_eq!(eval("BitCastF32U32 u32 0x7fc00001"), Val::F32(0x7fc0_0001));
    assert_eq!(
        eval("BitCastF64U64 u64 0x7ff8000000000001"),
        Val::F64(0x7ff8_0000_0000_0001)
    );
    assert_eq!(
        eval("BitCastU64F64 f64 1.0"),
        Val::U64(0x3ff0_0000_0000_0000)
    );
    assert_eq!(eval("BitCastF16U16 u16 0x7e01"), Val::F16(0x7e01));
    let body = "    %0 = BitCastF16U16 u16 0x3c00
    %1 = BitCastU16F16 %0
    %2 = CompositeConstructU32x2 u32 0x1, u32 0x2
    %3 = PackUint2x32 %2
    %4 = UnpackUint2x32 u64 0x300000004
    %5 = PackDouble2x32 %2
    %6 = UnpackDouble2x32 f64 0x300000004
    %7 = UnpackFloat2x16 u32 0xc0003c00
    %8 = PackFloat2x16 %7
    %9 = CompositeConstructF32x2 f32 1.0, f32 -2.0
    %10 = PackHalf2x16 %9
    %11 = UnpackHalf2x16 u32 0xc0003c00
    %12 = CompositeConstructF32x2 f32 0x7fc00001, f32 0x33000000
    %13 = PackHalf2x16 %12";
    assert_eq!(value_of(body, 1), Val::U16(0x3c00));
    assert_eq!(value_of(body, 3), Val::U64(0x2_0000_0001));
    assert_eq!(value_of(body, 4), Val::U32x2([4, 3]));
    assert_eq!(value_of(body, 5), Val::F64(0x2_0000_0001));
    assert_eq!(value_of(body, 6), Val::U32x2([4, 3]));
    assert_eq!(value_of(body, 7), Val::F16x2([0x3c00, 0xc000]));
    assert_eq!(value_of(body, 8), Val::U32(0xc000_3c00));
    assert_eq!(value_of(body, 10), Val::U32(0xc000_3c00));
    assert_eq!(value_of(body, 11), Val::F32x2([0x3f80_0000, 0xc000_0000]));
    // A NaN packs as the canonical half NaN; half the smallest denormal ties to zero.
    assert_eq!(value_of(body, 13), Val::U32(0x7fff));
}
