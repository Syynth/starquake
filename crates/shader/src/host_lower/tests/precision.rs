//! Each precision loss the module docs list, pinned with a case where the result before and
//! after the pass differ as stated.

use super::super::bits::{F16_CANONICAL_NAN, F64_CANONICAL_NAN};
use super::super::{F16_FMA_TOLERANCE_ULPS, F64_RELATIVE_TOLERANCE, HostPass};
use super::{half_ulps, halves_of, pair, passed, ran, reg, set_pair};
use crate::interp::Machine;
use crate::ir::{
    Builder, Flags, FmzMode, FpControl, FpRounding, Function, Imm, Opcode, Operand, Terminator,
};

/// A one-block function builder with the boundary conversions the tests read and write
/// through.
struct B<'f>(Builder<'f>);

impl B<'_> {
    fn op(&mut self, opcode: Opcode, args: &[Operand]) -> Operand {
        Operand::Value(self.0.op(opcode, args).expect("builds"))
    }

    fn op_with(&mut self, opcode: Opcode, args: &[Operand], flags: Flags) -> Operand {
        Operand::Value(self.0.op_with(opcode, args, flags).expect("builds"))
    }

    fn get(&mut self, at: u8) -> Operand {
        self.op(Opcode::GetRegister, &[reg(at)])
    }

    fn set(&mut self, at: u8, value: Operand) {
        self.op(Opcode::SetRegister, &[reg(at), value]);
    }

    /// The 64-bit float in the register pair at `lo`.
    fn double(&mut self, lo: u8) -> Operand {
        let (low, high) = (self.get(lo), self.get(lo + 1));
        let words = self.op(Opcode::CompositeConstructU32x2, &[low, high]);
        self.op(Opcode::PackDouble2x32, &[words])
    }

    /// Writes the 64-bit float `value` to the register pair at `lo`.
    fn put_double(&mut self, lo: u8, value: Operand) {
        let words = self.op(Opcode::UnpackDouble2x32, &[value]);
        for i in 0..2 {
            let word = self.op(Opcode::CompositeExtractU32x2, &[words, imm(i)]);
            self.set(lo + u8::try_from(i).expect("small"), word);
        }
    }

    /// The low half of register `at`.
    fn half(&mut self, at: u8) -> Operand {
        let word = self.get(at);
        let halves = self.op(Opcode::UnpackFloat2x16, &[word]);
        self.op(Opcode::CompositeExtractF16x2, &[halves, imm(0)])
    }

    /// Writes the 16-bit float `value` to both halves of register `at`.
    fn put_half(&mut self, at: u8, value: Operand) {
        let pair = self.op(Opcode::CompositeConstructF16x2, &[value, value]);
        let word = self.op(Opcode::PackFloat2x16, &[pair]);
        self.set(at, word);
    }
}

fn imm(value: u32) -> Operand {
    Operand::Imm(Imm::U32(value))
}

fn build(body: impl FnOnce(&mut B<'_>)) -> Function {
    let mut function = Function::new();
    let entry = function.entry();
    let mut b = B(Builder::new(&mut function, entry));
    body(&mut b);
    b.0.terminate(Terminator::Return).expect("returns");
    function
}

/// The machines `function` leaves before and after `pass`, from `machine`.
fn before_after(function: &Function, pass: HostPass, machine: &Machine) -> (Machine, Machine) {
    (
        ran(function, machine),
        ran(&passed(function, pass), machine),
    )
}

/// A machine with the 64-bit floats `values` in register pairs 0, 2, 4, and so on.
fn doubles(values: &[f64]) -> Machine {
    let mut machine = Machine::default();
    for (i, value) in values.iter().enumerate() {
        set_pair(
            &mut machine,
            2 * u8::try_from(i).expect("small"),
            value.to_bits(),
        );
    }
    machine
}

/// A machine with the halves `values` in the low half of registers 0, 1, 2, and so on.
fn halves(values: &[u16]) -> Machine {
    let mut machine = Machine::default();
    for (i, &value) in values.iter().enumerate() {
        machine.regs[i] = u32::from(value);
    }
    machine
}

fn double_at(machine: &Machine, lo: u8) -> f64 {
    f64::from_bits(pair(machine, lo))
}

fn half_at(machine: &Machine, at: usize) -> u16 {
    halves_of(machine.regs[at]).0
}

/// `FPNeg64` of `1 + 2^-30`, exact on its operand, gives `-1` after the pass: the operand was
/// rounded to the nearest binary32.
#[test]
fn f64_operands_round_to_binary32() {
    let function = build(|b| {
        let x = b.double(0);
        let neg = b.op(Opcode::FPNeg64, &[x]);
        b.put_double(10, neg);
    });
    let x = 1.0 + 2.0_f64.powi(-30);
    let (before, after) = before_after(&function, HostPass::Fp64ToFp32, &doubles(&[x]));
    assert_eq!(double_at(&before, 10), -x);
    assert_eq!(double_at(&after, 10), -1.0);
}

/// Arithmetic on binary32 operands rounds at binary32, within [`F64_RELATIVE_TOLERANCE`] but
/// not exact: `1 + 2^-30` is `1`, and `1 / 3` and `1 / sqrt(3)` are the binary32 ones.
#[test]
fn f64_arithmetic_rounds_at_binary32() {
    let function = build(|b| {
        let (one, tiny, three) = (b.double(0), b.double(2), b.double(4));
        let sum = b.op(Opcode::FPAdd64, &[one, tiny]);
        b.put_double(10, sum);
        let recip = b.op(Opcode::FPRecip64, &[three]);
        b.put_double(12, recip);
        let rsqrt = b.op(Opcode::FPRecipSqrt64, &[three]);
        b.put_double(14, rsqrt);
    });
    let machine = doubles(&[1.0, 2.0_f64.powi(-30), 3.0]);
    let (before, after) = before_after(&function, HostPass::Fp64ToFp32, &machine);
    assert_eq!(double_at(&before, 10), 1.0 + 2.0_f64.powi(-30));
    assert_eq!(double_at(&after, 10), 1.0);
    assert_eq!(double_at(&before, 12), 1.0 / 3.0);
    assert_eq!(double_at(&after, 12), f64::from(1.0_f32 / 3.0));
    for at in [10, 12, 14] {
        let (b, a) = (double_at(&before, at), double_at(&after, at));
        assert_ne!(a, b, "r{at}");
        assert!((a - b).abs() <= F64_RELATIVE_TOLERANCE * b.abs(), "r{at}");
    }
}

/// A finite value past binary32's range becomes an infinity, read in (`FPAbs64` of `1e300`)
/// or computed (`2^100 * 2^100`).
#[test]
fn f64_values_past_binary32_become_infinities() {
    let function = build(|b| {
        let (big, power) = (b.double(0), b.double(2));
        let abs = b.op(Opcode::FPAbs64, &[big]);
        b.put_double(10, abs);
        let square = b.op(Opcode::FPMul64, &[power, power]);
        b.put_double(12, square);
    });
    let machine = doubles(&[-1e300, 2.0_f64.powi(100)]);
    let (before, after) = before_after(&function, HostPass::Fp64ToFp32, &machine);
    assert_eq!(double_at(&before, 10), 1e300);
    assert_eq!(double_at(&before, 12), 2.0_f64.powi(200));
    assert_eq!(double_at(&after, 10), f64::INFINITY);
    assert_eq!(double_at(&after, 12), f64::INFINITY);
}

/// A value below 2^-126 in magnitude becomes a zero of its sign: `FPNeg64` of `1e-40` gives
/// `-0`.
#[test]
fn f64_values_below_binary32_normals_become_zeros() {
    let function = build(|b| {
        let x = b.double(0);
        let neg = b.op(Opcode::FPNeg64, &[x]);
        b.put_double(10, neg);
    });
    let (before, after) = before_after(&function, HostPass::Fp64ToFp32, &doubles(&[1e-40]));
    assert_eq!(double_at(&before, 10), -1e-40);
    assert_eq!(pair(&after, 10), (-0.0_f64).to_bits());
}

/// A binary32 subnormal result widens to a zero: `2^-70 * 2^-70`, both operands binary32
/// normals, is `2^-140` before and `+0` after.
#[test]
fn binary32_subnormal_results_widen_to_zero() {
    let function = build(|b| {
        let x = b.double(0);
        let square = b.op(Opcode::FPMul64, &[x, x]);
        b.put_double(10, square);
    });
    let machine = doubles(&[2.0_f64.powi(-70)]);
    let (before, after) = before_after(&function, HostPass::Fp64ToFp32, &machine);
    assert_eq!(double_at(&before, 10), 2.0_f64.powi(-140));
    assert_eq!(pair(&after, 10), 0);
}

/// A NaN's payload is lost: `FPNeg64` keeps it before the f64 pass and gives the canonical NaN
/// after; `FPNeg16` likewise for the f16 pass.
#[test]
fn nan_payloads_are_lost() {
    let double = build(|b| {
        let x = b.double(0);
        let neg = b.op(Opcode::FPNeg64, &[x]);
        b.put_double(10, neg);
    });
    let mut machine = Machine::default();
    set_pair(&mut machine, 0, 0x7ff0_0000_0000_0001);
    let (before, after) = before_after(&double, HostPass::Fp64ToFp32, &machine);
    assert_eq!(pair(&before, 10), 0xfff0_0000_0000_0001);
    assert_eq!(pair(&after, 10), F64_CANONICAL_NAN);

    let half = build(|b| {
        let x = b.half(0);
        let neg = b.op(Opcode::FPNeg16, &[x]);
        b.put_half(10, neg);
    });
    let (before, after) = before_after(&half, HostPass::Fp16ToFp32, &halves(&[0x7e01]));
    assert_eq!(half_at(&before, 10), 0xfe01);
    assert_eq!(u32::from(half_at(&after, 10)), F16_CANONICAL_NAN);
}

/// Comparisons and conversions to integers see the rounded operand: `1 < 1 + 2^-30` is true
/// before and false after, and `2^24 + 1` converts to `16777217` before and `16777216` after.
#[test]
fn f64_comparisons_and_conversions_see_rounded_values() {
    let function = build(|b| {
        let (one, near, big) = (b.double(0), b.double(2), b.double(4));
        let less = b.op(Opcode::FPOrdLessThan64, &[one, near]);
        let word = b.op(Opcode::SelectU32, &[less, imm(1), imm(0)]);
        b.set(10, word);
        let int = b.op(Opcode::ConvertS32F64, &[big]);
        b.set(11, int);
    });
    let machine = doubles(&[1.0, 1.0 + 2.0_f64.powi(-30), 16_777_217.0]);
    let (before, after) = before_after(&function, HostPass::Fp64ToFp32, &machine);
    assert_eq!((before.regs[10], before.regs[11]), (1, 16_777_217));
    assert_eq!((after.regs[10], after.regs[11]), (0, 16_777_216));
}

/// An integer converted to a 64-bit float is rounded to binary32: `2^24 + 1` is exact before
/// and `2^24` after.
#[test]
fn integers_convert_to_binary32() {
    let function = build(|b| {
        let int = b.get(0);
        let wide = b.op(Opcode::ConvertF64U32, &[int]);
        b.put_double(10, wide);
    });
    let mut machine = Machine::default();
    machine.regs[0] = (1 << 24) + 1;
    let (before, after) = before_after(&function, HostPass::Fp64ToFp32, &machine);
    assert_eq!(double_at(&before, 10), 16_777_217.0);
    assert_eq!(double_at(&after, 10), 16_777_216.0);
}

/// `ConvertF32F64` with round-toward-zero keeps `1 + 2^-23` from `1 + 2^-23 + 3 * 2^-25`
/// before the pass; after it, the value was rounded to nearest at the boundary, to
/// `1 + 2^-22`, and the mode is gone with the conversion.
#[test]
fn f64_to_f32_conversion_modes_are_lost() {
    let toward_zero = Flags::Fp(FpControl {
        rounding: FpRounding::Rz,
        ..FpControl::default()
    });
    let function = build(|b| {
        let x = b.double(0);
        let single = b.op_with(Opcode::ConvertF32F64, &[x], toward_zero);
        let bits = b.op(Opcode::BitCastU32F32, &[single]);
        b.set(10, bits);
    });
    let x = 1.0 + 2.0_f64.powi(-23) + 3.0 * 2.0_f64.powi(-25);
    let (before, after) = before_after(&function, HostPass::Fp64ToFp32, &doubles(&[x]));
    assert_eq!(before.regs[10], (1.0_f32 + 2.0_f32.powi(-23)).to_bits());
    assert_eq!(after.regs[10], (1.0_f32 + 2.0_f32.powi(-22)).to_bits());
}

/// Intermediate 16-bit results keep binary32 precision and range until they are packed:
/// `(1 + 2^-10)^2 - (1 + 2^-9)` is `0` before and `2^-20` (a binary16 subnormal) after;
/// `256 * 256 * 0.5` overflows to infinity before and is `32768` after; `(1 + 2^-10)^2 ==
/// 1 + 2^-9` holds only before; and `1023 * (1 + 2^-10)` truncates to `1024` before and `1023`
/// after.
#[test]
fn f16_intermediates_keep_binary32_precision_and_range() {
    let function = build(|b| {
        let (near_one, sum, big, half) = (b.half(0), b.half(1), b.half(2), b.half(3));
        let (just_under, factor) = (b.half(4), b.half(5));
        let square = b.op(Opcode::FPMul16, &[near_one, near_one]);
        let neg_sum = b.op(Opcode::FPNeg16, &[sum]);
        let difference = b.op(Opcode::FPAdd16, &[square, neg_sum]);
        b.put_half(10, difference);
        let wide = b.op(Opcode::FPMul16, &[big, big]);
        let halved = b.op(Opcode::FPMul16, &[wide, half]);
        b.put_half(11, halved);
        let equal = b.op(Opcode::FPOrdEqual16, &[square, sum]);
        let word = b.op(Opcode::SelectU32, &[equal, imm(1), imm(0)]);
        b.set(12, word);
        let product = b.op(Opcode::FPMul16, &[just_under, factor]);
        let int = b.op(Opcode::ConvertS32F16, &[product]);
        b.set(13, int);
    });
    // 1 + 2^-10, 1 + 2^-9, 256, 0.5, 1023, 1 + 2^-10.
    let machine = halves(&[0x3c01, 0x3c02, 0x5c00, 0x3800, 0x63fe, 0x3c01]);
    let (before, after) = before_after(&function, HostPass::Fp16ToFp32, &machine);
    assert_eq!(half_at(&before, 10), 0x0000);
    assert_eq!(half_at(&after, 10), 0x0010);
    assert_eq!(half_at(&before, 11), 0x7c00);
    assert_eq!(half_at(&after, 11), 0x7800);
    assert_eq!((before.regs[12], after.regs[12]), (1, 0));
    assert_eq!((before.regs[13], after.regs[13]), (1024, 1023));
}

/// `ConvertF16F32` becomes `Identity`: `1 + 2^-12` narrowed to a half and widened back is `1`
/// before and keeps its value after.
#[test]
fn f16_narrowing_waits_for_the_pack() {
    let function = build(|b| {
        let bits = b.get(0);
        let x = b.op(Opcode::BitCastF32U32, &[bits]);
        let narrow = b.op(Opcode::ConvertF16F32, &[x]);
        let back = b.op(Opcode::ConvertF32F16, &[narrow]);
        let out = b.op(Opcode::BitCastU32F32, &[back]);
        b.set(10, out);
    });
    let mut machine = Machine::default();
    let x = 1.0_f32 + 2.0_f32.powi(-12);
    machine.regs[0] = x.to_bits();
    let (before, after) = before_after(&function, HostPass::Fp16ToFp32, &machine);
    assert_eq!(before.regs[10], 1.0_f32.to_bits());
    assert_eq!(after.regs[10], x.to_bits());
}

/// Flushing applies to binary32 denormals after the pass, so `FPAdd16.FTZ` of the smallest
/// binary16 subnormal and zero gives zero before and the subnormal after.
#[test]
fn half_flushing_is_lost() {
    let flush = Flags::Fp(FpControl {
        fmz_mode: FmzMode::Ftz,
        ..FpControl::default()
    });
    let function = build(|b| {
        let (tiny, zero) = (b.half(0), b.half(1));
        let sum = b.op_with(Opcode::FPAdd16, &[tiny, zero], flush);
        b.put_half(10, sum);
    });
    let (before, after) = before_after(&function, HostPass::Fp16ToFp32, &halves(&[0x0001, 0]));
    assert_eq!(half_at(&before, 10), 0x0000);
    assert_eq!(half_at(&after, 10), 0x0001);
}

/// A rounding mode on 16-bit arithmetic rounds at binary32, where the sum is exact, and the
/// pack rounds to nearest even: `1 + 3 * 2^-12` toward zero is `1` before and `1 + 2^-10`
/// after.
#[test]
fn half_rounding_modes_apply_at_binary32() {
    let toward_zero = Flags::Fp(FpControl {
        rounding: FpRounding::Rz,
        ..FpControl::default()
    });
    let function = build(|b| {
        let (one, small) = (b.half(0), b.half(1));
        let sum = b.op_with(Opcode::FPAdd16, &[one, small], toward_zero);
        b.put_half(10, sum);
    });
    // 1 and 3 * 2^-12 (1.5 * 2^-11).
    let (before, after) = before_after(&function, HostPass::Fp16ToFp32, &halves(&[0x3c00, 0x1200]));
    assert_eq!(half_at(&before, 10), 0x3c00);
    assert_eq!(half_at(&after, 10), 0x3c01);
}

/// `FPFma16` rounds twice: `1.75 * (293 / 512) - 2^-24` lies just below the binary16 tie
/// `1 + 3 * 2^-11`, so it rounds to `1 + 2^-10` before; after, binary32 rounds it onto the
/// tie, which goes to the even `1 + 2^-9`: one binary16 ulp, the stated tolerance.
#[test]
fn half_fma_rounds_twice() {
    let function = build(|b| {
        let (x, y, z) = (b.half(0), b.half(1), b.half(2));
        let fma = b.op(Opcode::FPFma16, &[x, y, z]);
        b.put_half(10, fma);
    });
    let machine = halves(&[0x3f00, 0x3894, 0x8001]);
    let (before, after) = before_after(&function, HostPass::Fp16ToFp32, &machine);
    assert_eq!(half_at(&before, 10), 0x3c01);
    assert_eq!(half_at(&after, 10), 0x3c02);
    assert_eq!(
        half_ulps(half_at(&before, 10), half_at(&after, 10)),
        F16_FMA_TOLERANCE_ULPS
    );
}
