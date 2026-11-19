//! Floating-point semantics: the three formats, rounding with the IR's rounding modes,
//! denormal flushing, NaN canonicalisation, and the arithmetic built on them.
//!
//! Every result is computed from the operands' exact values, so it does not depend on the host:
//!
//! - **16- and 32-bit addition, multiplication and fused multiply-add are exact, then rounded
//!   once** with the operation's rounding mode. A binary16 or binary32 product is exact in
//!   binary64 (at most 48 significant bits), and Knuth's two-sum gives a sum's exact error, so
//!   the exact result is a pair of binary64 values that [`round`] rounds to the target format in
//!   any of the four modes.
//! - **64-bit arithmetic** uses the host's binary64 operations, which IEEE 754 defines exactly
//!   in round-to-nearest-even; directed rounding on a 64-bit operation is not modelled (the
//!   caller reports it as unsupported).
//! - **Division and square root** (`FPRecip32`, `FPSqrt`) are computed in binary64 and rounded
//!   to binary32; binary64 has more than twice binary32's precision plus two bits, so the double
//!   rounding gives the correctly rounded result (Figueroa, "When is double rounding
//!   innocuous?", 1995).
//! - **Transcendentals** (`FPSin`, `FPCos`, `FPExp2`, `FPLog2`) and `FPRecipSqrt32` are the
//!   documented exception: they call the host's binary64 `sin`, `cos`, `exp2`, `ln`-based
//!   `log2` and `sqrt` and round the result to binary32. Hosts' binary64 `sin`, `cos`, `exp2`
//!   and `log2` can differ in the last binary64 bit, which changes the binary32 result only when
//!   it lies within that bit of a rounding boundary. Tests compare them at exactly
//!   representable points (`sin 0`, `exp2 3`, `log2 8`) or within a tolerance.
//!
//! NaN results of arithmetic and conversions are the canonical NaN of their format (positive,
//! all mantissa bits set): `0x7fff`, `0x7fff_ffff` and `0x7fff_ffff_ffff_ffff`. The binary16
//! one is PR #4588's (`eden-pr4588/src/tests/video_core/native_shader.cpp`, the test case
//! `Native half FMA ReLU preserves fusion values above one and canonical NaNs`: an input NaN of
//! `0x7e01` gives `0x7fff`); the binary32 and binary64 ones are a guess by analogy, to be confirmed by a
//! hardware trace. Sign and absolute value (`FPNeg*`, `FPAbs*`), selects, bit casts and moves
//! are bit operations and keep a NaN's payload (PR #4588's `ExecuteNativeWarp` computes
//! `FPNeg16` as `a ^ 0x8000` and `FPAbs16` as `a & 0x7fff`).
//!
//! Denormals are kept, except that an operation whose floating-point controls say
//! [`FmzMode::Ftz`] or [`FmzMode::Fmz`] flushes denormal operands and a denormal rounded result
//! to a zero of the same sign, and [`FmzMode::Fmz`] also makes a product with a zero factor
//! `+0` whatever the other factor is (Eden's `modifiers.h`: "FTZ: Flush denorms to zero, NAN is
//! propagated", "FMZ: Flush denorms to zero, x * 0 == 0"). [`FmzMode::None`] and
//! [`FmzMode::DontCare`] keep denormals. Flushing the result after rounding, and the `+0` of an
//! FMZ product, are guesses at the hardware's order and sign. [`FpRounding::DontCare`] rounds to
//! nearest even.

use std::cmp::Ordering;

use crate::ir::{FmzMode, FpControl, FpRounding};

use super::value::{low16, low32};

/// A floating-point format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Format {
    /// IEEE 754 binary16.
    F16,
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary64.
    F64,
}

/// The binary16 canonical NaN (see the module docs).
pub(super) const F16_NAN: u64 = 0x7fff;
/// The binary32 canonical NaN (a guess; see the module docs).
pub(super) const F32_NAN: u64 = 0x7fff_ffff;
/// The binary64 canonical NaN (a guess; see the module docs).
pub(super) const F64_NAN: u64 = 0x7fff_ffff_ffff_ffff;

impl Format {
    /// Significand bits, the hidden bit included.
    const fn precision(self) -> i32 {
        match self {
            Self::F16 => 11,
            Self::F32 => 24,
            Self::F64 => 53,
        }
    }

    /// The exponent of the smallest normal number.
    const fn min_exp(self) -> i32 {
        match self {
            Self::F16 => -14,
            Self::F32 => -126,
            Self::F64 => -1022,
        }
    }

    /// The exponent of the largest finite number.
    const fn max_exp(self) -> i32 {
        match self {
            Self::F16 => 15,
            Self::F32 => 127,
            Self::F64 => 1023,
        }
    }

    /// Stored mantissa bits.
    const fn mantissa_bits(self) -> u32 {
        match self {
            Self::F16 => 10,
            Self::F32 => 23,
            Self::F64 => 52,
        }
    }

    /// The sign bit.
    const fn sign_bit(self) -> u64 {
        1 << (self.mantissa_bits() + self.exponent_bits())
    }

    /// Stored exponent bits.
    const fn exponent_bits(self) -> u32 {
        match self {
            Self::F16 => 5,
            Self::F32 => 8,
            Self::F64 => 11,
        }
    }

    /// The canonical NaN.
    pub(super) const fn nan(self) -> u64 {
        match self {
            Self::F16 => F16_NAN,
            Self::F32 => F32_NAN,
            Self::F64 => F64_NAN,
        }
    }

    /// The exponent field, shifted down.
    const fn exponent_field(self, bits: u64) -> u64 {
        (bits >> self.mantissa_bits()) & ((1 << self.exponent_bits()) - 1)
    }

    /// The mantissa field.
    const fn mantissa_field(self, bits: u64) -> u64 {
        bits & ((1 << self.mantissa_bits()) - 1)
    }

    /// Whether `bits` is a NaN.
    pub(super) const fn is_nan(self, bits: u64) -> bool {
        self.exponent_field(bits) == (1 << self.exponent_bits()) - 1
            && self.mantissa_field(bits) != 0
    }

    /// Whether `bits` is a denormal (non-zero, with a zero exponent field).
    pub(super) const fn is_denormal(self, bits: u64) -> bool {
        self.exponent_field(bits) == 0 && self.mantissa_field(bits) != 0
    }

    /// `bits` with a denormal replaced by a zero of the same sign.
    pub(super) const fn flush(self, bits: u64) -> u64 {
        if self.is_denormal(bits) {
            bits & self.sign_bit()
        } else {
            bits
        }
    }

    /// The largest finite value.
    fn max_finite(self) -> f64 {
        let significand = pow2(self.precision()) - 1.0;
        significand * pow2(self.max_exp() - self.precision() + 1)
    }

    /// The exact value of `bits` (NaN for a NaN, whose payload callers handle first).
    pub(super) fn decode(self, bits: u64) -> f64 {
        match self {
            Self::F64 => f64::from_bits(bits),
            Self::F32 => f64::from(f32::from_bits(low32(bits))),
            Self::F16 => {
                let bits = low16(bits);
                let exponent = (bits >> 10) & 0x1f;
                let mantissa = bits & 0x3ff;
                let magnitude = match exponent {
                    0x1f if mantissa == 0 => f64::INFINITY,
                    0x1f => f64::NAN,
                    0 => f64::from(mantissa) * pow2(Self::F16.min_exp() - 10),
                    _ => f64::from(mantissa | 0x400) * pow2(i32::from(exponent) - 15 - 10),
                };
                if bits & 0x8000 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            }
        }
    }

    /// The bits of `value`, which must be representable in this format, or infinite; a NaN
    /// becomes the canonical NaN.
    pub(super) fn encode(self, value: f64) -> u64 {
        if value.is_nan() {
            return self.nan();
        }
        match self {
            Self::F64 => value.to_bits(),
            Self::F32 => {
                // Exact: `value` is representable, so nothing is rounded.
                let narrowed = round_to_f32_bits(value);
                u64::from(narrowed)
            }
            Self::F16 => {
                let sign = if value.is_sign_negative() { 0x8000 } else { 0 };
                let magnitude = value.abs();
                if magnitude.is_infinite() {
                    return sign | 0x7c00;
                }
                if magnitude == 0.0 {
                    return sign;
                }
                if magnitude < pow2(Self::F16.min_exp()) {
                    let mantissa = trunc_u64(magnitude / pow2(Self::F16.min_exp() - 10));
                    return sign | mantissa;
                }
                let exponent = exponent_of(magnitude);
                let mantissa = trunc_u64(magnitude / pow2(exponent - 10)) & 0x3ff;
                let field = u64::try_from(exponent + 15).unwrap_or(0);
                sign | (field << 10) | mantissa
            }
        }
    }
}

/// The binary32 bits of an f64 that is representable in binary32 or infinite.
fn round_to_f32_bits(value: f64) -> u32 {
    // Split into sign, exponent and mantissa from the binary64 bits; the value is exact in
    // binary32, so the low 29 mantissa bits are zero (or it is a binary32 denormal).
    let format = Format::F32;
    let sign = if value.is_sign_negative() {
        0x8000_0000
    } else {
        0
    };
    let magnitude = value.abs();
    if magnitude.is_infinite() {
        return sign | 0x7f80_0000;
    }
    if magnitude == 0.0 {
        return sign;
    }
    if magnitude < pow2(format.min_exp()) {
        let mantissa = low32(trunc_u64(magnitude / pow2(format.min_exp() - 23)));
        return sign | mantissa;
    }
    let exponent = exponent_of(magnitude);
    let mantissa = low32(trunc_u64(magnitude / pow2(exponent - 23))) & 0x7f_ffff;
    let field = u32::try_from(exponent + 127).unwrap_or(0);
    sign | (field << 23) | mantissa
}

/// `2^exponent`, for exponents of binary64's normal range (clamped to it).
pub(super) fn pow2(exponent: i32) -> f64 {
    let biased = exponent.saturating_add(1023).clamp(1, 2046);
    f64::from_bits(u64::from(biased.unsigned_abs()) << 52)
}

/// `floor(log2(magnitude))` of a positive normal binary64 value.
fn exponent_of(magnitude: f64) -> i32 {
    let field = (magnitude.to_bits() >> 52) & 0x7ff;
    i32::try_from(field).unwrap_or(0) - 1023
}

/// The non-negative integral value `value` as an integer, saturating; NaN gives 0. Rust's
/// float-to-integer `as` is defined to truncate toward zero, saturate and map NaN to zero
/// (the Rust reference, "Numeric cast").
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Rust's float-to-integer `as` saturates and maps NaN to zero, which is the semantics wanted"
)]
pub(super) fn trunc_u64(value: f64) -> u64 {
    value as u64
}

/// `value` truncated toward zero as a signed integer, saturating; NaN gives 0 (as
/// [`trunc_u64`]).
#[expect(
    clippy::cast_possible_truncation,
    reason = "Rust's float-to-integer `as` saturates and maps NaN to zero, which is the semantics wanted"
)]
pub(super) fn trunc_i64(value: f64) -> i64 {
    value as i64
}

/// `value` rounded to the nearest binary64 value, ties to even (Rust's integer-to-float `as`,
/// the Rust reference, "Numeric cast").
#[expect(
    clippy::cast_precision_loss,
    reason = "the rounding is wanted; the caller keeps the exact error"
)]
fn nearest_f64(value: i128) -> f64 {
    value as f64
}

/// `value`, an integral binary64 value of magnitude below 2^127, as an integer.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is integral and in range, so the cast is exact"
)]
fn integral_i128(value: f64) -> i128 {
    value as i128
}

/// The exact error of `a + b` given its rounded sum `sum` (Knuth's two-sum; exact when the
/// sum does not overflow).
fn two_sum_error(a: f64, b: f64, sum: f64) -> f64 {
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    (a - a_virtual) + (b - b_virtual)
}

/// Which way to round a magnitude.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    Nearest,
    Down,
    Up,
}

fn direction(rounding: FpRounding, negative: bool) -> Direction {
    match (rounding, negative) {
        (FpRounding::DontCare | FpRounding::Rn, _) => Direction::Nearest,
        (FpRounding::Rz, _) | (FpRounding::Rp, true) | (FpRounding::Rm, false) => Direction::Down,
        (FpRounding::Rp, false) | (FpRounding::Rm, true) => Direction::Up,
    }
}

/// Whether `rounding` rounds to nearest.
pub(super) const fn is_nearest(rounding: FpRounding) -> bool {
    matches!(rounding, FpRounding::DontCare | FpRounding::Rn)
}

/// The exact real `hi + lo` rounded to `format` with `rounding`, as a binary64 value that the
/// format represents exactly (or an infinity). `hi` is finite and `lo` is zero or the exact
/// error of the binary64 operation that gave `hi`, so `|lo|` is at most half a binary64 unit in
/// the last place of `hi`.
pub(super) fn round(format: Format, hi: f64, lo: f64, rounding: FpRounding) -> f64 {
    if hi == 0.0 {
        return hi;
    }
    let negative = hi < 0.0;
    let dir = direction(rounding, negative);
    if format == Format::F64 {
        // `hi` is the binary64 result rounded to nearest; `lo` says on which side the exact
        // value lies.
        let toward_zero_side = (lo < 0.0) != negative && lo != 0.0;
        let away_side = (lo > 0.0) != negative && lo != 0.0;
        let magnitude = hi.abs();
        let rounded = match dir {
            Direction::Down if toward_zero_side => magnitude.next_down(),
            Direction::Up if away_side => magnitude.next_up(),
            Direction::Nearest | Direction::Down | Direction::Up => magnitude,
        };
        return if negative { -rounded } else { rounded };
    }
    let magnitude = hi.abs();
    let low = if negative { -lo } else { lo };
    let exponent = exponent_of(magnitude);
    let quantum_exponent = exponent.max(format.min_exp()) - (format.precision() - 1);
    let quantum = pow2(quantum_exponent);
    let scaled = magnitude / quantum;
    let whole = scaled.trunc();
    let fraction = scaled - whole;
    let rounded = if fraction == 0.0 && low < 0.0 && dir == Direction::Down {
        // The exact value is just below a representable one: the next one down, whose spacing
        // halves below a power of two in the normal range.
        let at_binade_start = whole.total_cmp(&pow2(format.precision() - 1)).is_eq();
        if at_binade_start && exponent > format.min_exp() {
            magnitude - quantum / 2.0
        } else {
            magnitude - quantum
        }
    } else {
        let up = match dir {
            Direction::Down => false,
            Direction::Up => fraction > 0.0 || low > 0.0,
            Direction::Nearest => match fraction.total_cmp(&0.5) {
                Ordering::Greater => true,
                Ordering::Less => false,
                // A tie in `hi`: the exact value's side decides, else the even neighbour.
                Ordering::Equal => low > 0.0 || (low >= 0.0 && (whole / 2.0).fract() > 0.0),
            },
        };
        let count = if up { whole + 1.0 } else { whole };
        count * quantum
    };
    let limited = if rounded > format.max_finite() {
        match dir {
            Direction::Down => format.max_finite(),
            Direction::Nearest | Direction::Up => f64::INFINITY,
        }
    } else {
        rounded
    };
    if negative { -limited } else { limited }
}

/// How an operation's floating-point controls are applied (see the module docs).
#[derive(Clone, Copy, Debug)]
pub(super) struct Control {
    /// The rounding mode.
    pub(super) rounding: FpRounding,
    /// Denormal operands and results are flushed.
    pub(super) flush: bool,
    /// A product with a zero factor is `+0`.
    pub(super) mul_zero: bool,
}

impl Control {
    /// The controls an operation carries.
    pub(super) const fn of(control: FpControl) -> Self {
        Self {
            rounding: control.rounding,
            flush: matches!(control.fmz_mode, FmzMode::Ftz | FmzMode::Fmz),
            mul_zero: matches!(control.fmz_mode, FmzMode::Fmz),
        }
    }

    /// No controls: round to nearest even, keep denormals.
    pub(super) const NONE: Self = Self {
        rounding: FpRounding::DontCare,
        flush: false,
        mul_zero: false,
    };

    /// `bits` as an operand: flushed if the controls say so.
    pub(super) const fn operand(self, format: Format, bits: u64) -> u64 {
        if self.flush { format.flush(bits) } else { bits }
    }

    /// The bits of a rounded result: NaN canonical, flushed if the controls say so.
    fn result(self, format: Format, value: f64) -> u64 {
        let bits = format.encode(value);
        if self.flush { format.flush(bits) } else { bits }
    }
}

/// A 64-bit operation with directed rounding, which is not modelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DirectedF64;

/// `-0` when an exact zero sum of operands not both `+0` is rounded toward negative infinity
/// (IEEE 754-2019, 6.3), else `sum`.
fn signed_zero_sum(sum: f64, a: f64, b: f64, rounding: FpRounding) -> f64 {
    if sum == 0.0 && rounding == FpRounding::Rm && (a.is_sign_negative() || b.is_sign_negative()) {
        -0.0
    } else {
        sum
    }
}

/// The exactly rounded sum `a + b`.
fn sum(format: Format, a: f64, b: f64, control: Control) -> Result<u64, DirectedF64> {
    let rough = a + b;
    if !rough.is_finite() {
        return Ok(control.result(format, rough));
    }
    if format == Format::F64 {
        if !is_nearest(control.rounding) {
            return Err(DirectedF64);
        }
        return Ok(control.result(format, rough));
    }
    let error = two_sum_error(a, b, rough);
    let hi = signed_zero_sum(rough, a, b, control.rounding);
    Ok(control.result(format, round(format, hi, error, control.rounding)))
}

/// `a + b` (`FPAdd16`, `FPAdd32`, `FPAdd64`).
pub(super) fn add(format: Format, a: u64, b: u64, control: Control) -> Result<u64, DirectedF64> {
    let (a, b) = (control.operand(format, a), control.operand(format, b));
    if format.is_nan(a) || format.is_nan(b) {
        return Ok(format.nan());
    }
    sum(format, format.decode(a), format.decode(b), control)
}

/// The product of two operands for multiplication and fused multiply-add: `None` for a NaN
/// operand, `+0` for a zero factor under FMZ.
fn product(format: Format, a: u64, b: u64, control: Control) -> Option<f64> {
    let (x, y) = (format.decode(a), format.decode(b));
    if control.mul_zero && (x == 0.0 || y == 0.0) {
        return Some(0.0);
    }
    if format.is_nan(a) || format.is_nan(b) {
        return None;
    }
    Some(x * y)
}

/// `a * b` (`FPMul16`, `FPMul32`, `FPMul64`).
pub(super) fn mul(format: Format, a: u64, b: u64, control: Control) -> Result<u64, DirectedF64> {
    let (a, b) = (control.operand(format, a), control.operand(format, b));
    let Some(p) = product(format, a, b, control) else {
        return Ok(format.nan());
    };
    if format == Format::F64 && !is_nearest(control.rounding) {
        return Err(DirectedF64);
    }
    if !p.is_finite() || format == Format::F64 {
        return Ok(control.result(format, p));
    }
    // Exact: a binary16 or binary32 product fits binary64.
    Ok(control.result(format, round(format, p, 0.0, control.rounding)))
}

/// `a * b + c` with one rounding (`FPFma16`, `FPFma32`, `FPFma64`).
pub(super) fn fma(
    format: Format,
    factor: u64,
    multiplier: u64,
    addend: u64,
    control: Control,
) -> Result<u64, DirectedF64> {
    let factor = control.operand(format, factor);
    let multiplier = control.operand(format, multiplier);
    let addend = control.operand(format, addend);
    let Some(exact_product) = product(format, factor, multiplier, control) else {
        return Ok(format.nan());
    };
    if format.is_nan(addend) {
        return Ok(format.nan());
    }
    let addend = format.decode(addend);
    if format == Format::F64 {
        if !is_nearest(control.rounding) {
            return Err(DirectedF64);
        }
        let zero_product = control.mul_zero && exact_product == 0.0;
        let fused = if zero_product {
            exact_product + addend
        } else {
            format
                .decode(factor)
                .mul_add(format.decode(multiplier), addend)
        };
        return Ok(control.result(format, fused));
    }
    if !exact_product.is_finite() {
        return Ok(control.result(format, exact_product + addend));
    }
    sum(format, exact_product, addend, control)
}

/// The larger (`max`) or smaller of `a` and `b` (`FPMax32`, `FPMax64`, `FPMin32`, `FPMin64`):
/// a NaN operand gives the other one, two NaNs the canonical NaN, and `-0` is below `+0`
/// (IEEE 754-2019 `maximumNumber`, `minimumNumber`).
pub(super) fn min_max(format: Format, a: u64, b: u64, control: Control, max: bool) -> u64 {
    let (a, b) = (control.operand(format, a), control.operand(format, b));
    match (format.is_nan(a), format.is_nan(b)) {
        (true, true) => return format.nan(),
        (true, false) => return b,
        (false, true) => return a,
        (false, false) => {}
    }
    let (x, y) = (format.decode(a), format.decode(b));
    let a_wins = match x.partial_cmp(&y) {
        Some(Ordering::Greater) => max,
        Some(Ordering::Less) => !max,
        // Equal values differ at most in the sign of a zero.
        Some(Ordering::Equal) | None => x.is_sign_positive() == max,
    };
    if a_wins { a } else { b }
}

/// `value` clamped to `[low, high]` as `min(max(value, low), high)` (`FPClamp*`,
/// `FPSaturate*`).
pub(super) fn clamp(format: Format, value: u64, low: u64, high: u64) -> u64 {
    let raised = min_max(format, value, low, Control::NONE, true);
    min_max(format, raised, high, Control::NONE, false)
}

/// The rounding to an integral value an operation does.
#[derive(Clone, Copy, Debug)]
pub(super) enum Integral {
    /// To nearest, ties to even (`FPRoundEven*`).
    Even,
    /// Down (`FPFloor*`).
    Floor,
    /// Up (`FPCeil*`).
    Ceil,
    /// Toward zero (`FPTrunc*`).
    Trunc,
}

/// `value` rounded to an integral value; the operation's rounding field is not used, since the
/// operation names its rounding.
pub(super) fn integral(format: Format, value: u64, how: Integral, control: Control) -> u64 {
    let value = control.operand(format, value);
    if format.is_nan(value) {
        return format.nan();
    }
    let x = format.decode(value);
    let rounded = match how {
        Integral::Even => x.round_ties_even(),
        Integral::Floor => x.floor(),
        Integral::Ceil => x.ceil(),
        Integral::Trunc => x.trunc(),
    };
    control.result(format, rounded)
}

/// A comparison.
#[derive(Clone, Copy, Debug)]
pub(super) enum Compare {
    /// Equal.
    Equal,
    /// Not equal.
    NotEqual,
    /// Less than.
    Less,
    /// Greater than.
    Greater,
    /// Less than or equal.
    LessEqual,
    /// Greater than or equal.
    GreaterEqual,
}

/// The comparison of `a` and `b`: an ordered comparison is false and an unordered one true when
/// either operand is a NaN (SPIR-V's `OpFOrd*` and `OpFUnord*`, which Eden's emitter uses).
pub(super) fn compare(
    format: Format,
    a: u64,
    b: u64,
    how: Compare,
    ordered: bool,
    control: Control,
) -> bool {
    let (a, b) = (control.operand(format, a), control.operand(format, b));
    if format.is_nan(a) || format.is_nan(b) {
        return !ordered;
    }
    let Some(order) = format.decode(a).partial_cmp(&format.decode(b)) else {
        return !ordered;
    };
    match how {
        Compare::Equal => order.is_eq(),
        Compare::NotEqual => order.is_ne(),
        Compare::Less => order.is_lt(),
        Compare::Greater => order.is_gt(),
        Compare::LessEqual => order.is_le(),
        Compare::GreaterEqual => order.is_ge(),
    }
}

/// `value` converted from `from` to `to` with the operation's rounding (`ConvertF16F32` and the
/// other float-to-float conversions).
pub(super) fn convert(from: Format, to: Format, value: u64, control: Control) -> u64 {
    let value = control.operand(from, value);
    if from.is_nan(value) {
        return to.nan();
    }
    let x = from.decode(value);
    if !x.is_finite() || to == Format::F64 {
        // Widening is exact.
        return control.result(to, x);
    }
    control.result(to, round(to, x, 0.0, control.rounding))
}

/// The integer `value` converted to `to` with the operation's rounding (`ConvertF32S32` and
/// the other integer-to-float conversions).
pub(super) fn from_integer(to: Format, value: i128, control: Control) -> u64 {
    let hi = nearest_f64(value);
    // Exact: the integers converted have at most 64 bits, so the error has at most 11.
    let lo = nearest_f64(value - integral_i128(hi));
    control.result(to, round(to, hi, lo, control.rounding))
}

/// `value` truncated toward zero to an integer in `[min, max]`, saturating, NaN giving 0
/// (`ConvertS32F32` and the other float-to-integer conversions; see the parent module).
pub(super) fn to_integer(from: Format, value: u64, min: i128, max: i128) -> i128 {
    if from.is_nan(value) {
        return 0;
    }
    let x = from.decode(value).trunc();
    let wide = if x < 0.0 {
        i128::from(trunc_i64(x))
    } else {
        i128::from(trunc_u64(x))
    };
    wide.clamp(min, max)
}

/// A binary32 function computed in binary64 by `f` and rounded to nearest (`FPSin`, `FPCos`,
/// `FPExp2`, `FPLog2`, `FPRecip32`, `FPRecipSqrt32`, `FPSqrt`; see the module docs).
pub(super) fn unary_f32(value: u64, f: impl FnOnce(f64) -> f64) -> u64 {
    let format = Format::F32;
    if format.is_nan(value) {
        return format.nan();
    }
    let result = f(format.decode(value));
    if !result.is_finite() {
        return format.encode(result);
    }
    format.encode(round(format, result, 0.0, FpRounding::Rn))
}

/// A binary64 function computed by `f` (`FPRecip64`, `FPRecipSqrt64`).
pub(super) fn unary_f64(value: u64, f: impl FnOnce(f64) -> f64) -> u64 {
    let format = Format::F64;
    if format.is_nan(value) {
        return format.nan();
    }
    format.encode(f(format.decode(value)))
}
