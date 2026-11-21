//! The format conversions the host lowering passes write out with 32-bit integer operations:
//! binary64 to binary32 and back (the f64 pass's `PackDouble2x32` and `UnpackDouble2x32`), and
//! binary32 to binary16 and back (the f16 pass's `PackHalf2x16` and `UnpackHalf2x16`).
//!
//! Each conversion is an [`Emitter`] method that appends the operations to a block, and the
//! binary64-to-binary32 one also has a Rust model, [`narrow_f64`], which the pass uses for
//! 64-bit float immediates so that an immediate and a computed value narrow alike. The tests
//! run each emitted sequence in the interpreter against its model and against the interpreter's
//! own conversion (`tests::bits`). Only 32-bit integer operations are used, so a target without
//! 64-bit integers can run them (Eden's `PackedF64ToF32` and `F32ToPackedF64` also use 32-bit
//! operations only).
//!
//! # What each conversion gives
//!
//! - **binary64 to binary32** ([`Emitter::narrow`], [`narrow_f64`]): rounded to nearest, ties
//!   to even; a finite value that rounds past the largest binary32 is an infinity of its sign;
//!   a value whose binary64 exponent field is at or below [`F64_TO_F32_BIAS`] (magnitude below
//!   2^-126, the smallest normal binary32) is a zero of its sign (binary32 subnormals are not
//!   made); an infinity is an infinity; every NaN is [`F32_CANONICAL_NAN`], the interpreter's
//!   canonical binary32 NaN.
//! - **binary32 to binary64** ([`Emitter::widen`]): exact for zeros, normal numbers and
//!   infinities; a binary32 subnormal is a zero of its sign; every NaN is
//!   [`F64_CANONICAL_NAN`], the interpreter's canonical binary64 NaN.
//! - **binary32 to binary16** ([`Emitter::pack_half`]): what the interpreter's
//!   `PackHalf2x16` gives for each lane, bit for bit: rounded to nearest even, binary16
//!   subnormals made and rounded, overflow to infinity, every NaN [`F16_CANONICAL_NAN`].
//! - **binary16 to binary32** ([`Emitter::unpack_half`]): what the interpreter's
//!   `UnpackHalf2x16` gives, bit for bit: exact, subnormals included, every NaN
//!   [`F32_CANONICAL_NAN`].
//!
//! # Divergences from Eden
//!
//! Eden's `PackedF64ToF32` and `F32ToPackedF64`
//! (`eden/src/shader_recompiler/ir_opt/lower_fp64_to_fp32.cpp` @ cf654052, the same in
//! `eden-pr4588` @ e5fb3b97) differ from this in four ways, each pinned by a test in
//! `tests::bits`:
//!
//! - **The exponent's rebias.** Eden adds `F64ToF32Exp = +1023 - 127` to the binary64 exponent
//!   field when narrowing and `F32ToF64Exp = +127 - 1023` when widening, which is the right
//!   distance with the wrong sign (narrowing 1.0, exponent field 1023, gives 1919, which does
//!   not fit the 8-bit field). Here narrowing subtracts [`F64_TO_F32_BIAS`] and widening adds
//!   it (`narrowing_rebiases_the_exponent_the_right_way`).
//! - **Rounding.** Eden keeps the top 23 mantissa bits (truncation); here the dropped bits round
//!   to nearest even (`narrowing_rounds_to_nearest_even`).
//! - **Range.** Eden neither saturates an exponent past binary32's range to infinity nor
//!   flushes one below it; here both are handled (`narrowing_saturates_and_flushes`), and a
//!   binary32 subnormal widens to a zero rather than to Eden's garbage exponent
//!   (`widening_flushes_subnormals`).
//! - **NaNs.** Eden keeps the top mantissa bits, so a NaN whose payload is only in the dropped
//!   bits becomes an infinity; here every NaN becomes the canonical NaN
//!   (`nans_become_canonical`).
//!
//! The binary16 conversions have no Eden counterpart: Eden's f16 pass turns `PackFloat2x16`
//! into `PackHalf2x16`, which its SPIR-V backend writes as GLSL.std.450 `PackHalf2x16`, needing
//! no device feature. naga 30.0.1 needs `SHADER_FLOAT16_IN_FLOAT32` for `pack2x16float` and
//! `unpack2x16float` (`crate::lower::capabilities`), so a target without 16-bit floats gets
//! them written out here.

use crate::ir::{BuildError, Builder, Imm, Opcode, Operand};
use crate::opt::Work;

/// The binary64 exponent field's lowest bit in the high word.
pub(crate) const F64_EXPONENT_SHIFT: u32 = 20;
/// The binary64 exponent field's width.
pub(crate) const F64_EXPONENT_WIDTH: u32 = 11;
/// The high word's mantissa bits (the top 20 of binary64's 52).
pub(crate) const F64_HIGH_MANTISSA_MASK: u32 = 0x000f_ffff;
/// The low word's bits that binary32's mantissa keeps start here: 3 of them (52 - 23 = 29).
pub(crate) const F64_LOW_KEPT_SHIFT: u32 = 29;
/// The low word's round bit when narrowing, the highest bit dropped.
pub(crate) const F64_LOW_ROUND_BIT: u32 = 28;
/// The low word's bits below the round bit (the sticky bits).
pub(crate) const F64_LOW_STICKY_MASK: u32 = (1 << F64_LOW_ROUND_BIT) - 1;
/// The binary64 exponent field of infinities and NaNs.
pub(crate) const F64_EXPONENT_MAX: u32 = (1 << F64_EXPONENT_WIDTH) - 1;
/// The difference of the two exponent biases, `1023 - 127`.
pub(crate) const F64_TO_F32_BIAS: u32 = 1023 - 127;

/// The sign bit of a 32-bit word (binary32, and binary64's high word).
pub(crate) const SIGN_BIT: u32 = 0x8000_0000;
/// The binary32 exponent field's lowest bit.
pub(crate) const F32_EXPONENT_SHIFT: u32 = 23;
/// The binary32 exponent field's width.
pub(crate) const F32_EXPONENT_WIDTH: u32 = 8;
/// The binary32 exponent field of infinities and NaNs.
pub(crate) const F32_EXPONENT_MAX: u32 = (1 << F32_EXPONENT_WIDTH) - 1;
/// The binary32 mantissa bits.
pub(crate) const F32_MANTISSA_MASK: u32 = (1 << F32_EXPONENT_SHIFT) - 1;
/// A binary32 infinity's magnitude bits.
pub(crate) const F32_INFINITY: u32 = F32_EXPONENT_MAX << F32_EXPONENT_SHIFT;
/// Everything but the sign of a binary32.
pub(crate) const F32_MAGNITUDE_MASK: u32 = !SIGN_BIT;
/// The interpreter's canonical binary32 NaN (`interp::float`'s `F32_NAN`, itself a guess there).
pub(crate) const F32_CANONICAL_NAN: u32 = 0x7fff_ffff;
/// The interpreter's canonical binary64 NaN's high word (`interp::float`'s `F64_NAN`).
pub(crate) const F64_CANONICAL_NAN_HIGH: u32 = 0x7fff_ffff;
/// The interpreter's canonical binary64 NaN's low word.
pub(crate) const F64_CANONICAL_NAN_LOW: u32 = 0xffff_ffff;
/// The interpreter's canonical binary64 NaN.
pub(crate) const F64_CANONICAL_NAN: u64 =
    (F64_CANONICAL_NAN_HIGH as u64) << 32 | F64_CANONICAL_NAN_LOW as u64;
/// A binary64 infinity's high word magnitude bits.
pub(crate) const F64_INFINITY_HIGH: u32 = F64_EXPONENT_MAX << F64_EXPONENT_SHIFT;

/// The binary16 sign bit.
pub(crate) const F16_SIGN_BIT: u32 = 0x8000;
/// How far a binary32's sign is above a binary16's.
pub(crate) const F16_SIGN_SHIFT: u32 = 16;
/// The binary16 exponent field's lowest bit.
pub(crate) const F16_EXPONENT_SHIFT: u32 = 10;
/// The binary16 exponent field's width.
pub(crate) const F16_EXPONENT_WIDTH: u32 = 5;
/// The binary16 exponent field of infinities and NaNs.
pub(crate) const F16_EXPONENT_MAX: u32 = (1 << F16_EXPONENT_WIDTH) - 1;
/// The binary16 mantissa bits.
pub(crate) const F16_MANTISSA_MASK: u32 = (1 << F16_EXPONENT_SHIFT) - 1;
/// A binary16 infinity's bits.
pub(crate) const F16_INFINITY: u32 = F16_EXPONENT_MAX << F16_EXPONENT_SHIFT;
/// The interpreter's canonical binary16 NaN (`interp::float`'s `F16_NAN`, from PR #4588's test).
pub(crate) const F16_CANONICAL_NAN: u32 = 0x7fff;
/// The difference of the two exponent biases, `127 - 15`.
pub(crate) const F32_TO_F16_BIAS: u32 = 127 - 15;
/// The binary32 mantissa bits a binary16 drops: 23 - 10.
pub(crate) const F32_TO_F16_DROPPED: u32 = F32_EXPONENT_SHIFT - F16_EXPONENT_SHIFT;
/// The round bit of a binary32 mantissa narrowed to a normal binary16.
pub(crate) const F32_TO_F16_ROUND_BIT: u32 = F32_TO_F16_DROPPED - 1;
/// The sticky bits of a binary32 mantissa narrowed to a normal binary16.
pub(crate) const F32_TO_F16_STICKY_MASK: u32 = (1 << F32_TO_F16_ROUND_BIT) - 1;
/// The binary32 hidden bit.
pub(crate) const F32_HIDDEN_BIT: u32 = 1 << F32_EXPONENT_SHIFT;
/// The smallest binary32 exponent field that is a normal binary16: `1 + 112`.
pub(crate) const F16_NORMAL_MIN_F32_EXPONENT: u32 = 1 + F32_TO_F16_BIAS;
/// The smallest binary32 exponent field past every finite binary16: `31 + 112`.
pub(crate) const F16_OVERFLOW_F32_EXPONENT: u32 = F16_EXPONENT_MAX + F32_TO_F16_BIAS;
/// A binary16 subnormal is the binary32 significand (hidden bit set) shifted right by this
/// minus the binary32 exponent field: the significand is in units of 2^(e - 150) and a binary16
/// subnormal counts 2^-24, so the shift is `150 - 24 - e`.
pub(crate) const F16_SUBNORMAL_SHIFT_BASE: u32 = 126;
/// The largest such shift that matters: a 24-bit significand shifted by 25 is below half the
/// smallest binary16 subnormal, so every larger shift gives the same zero.
pub(crate) const F16_SUBNORMAL_SHIFT_MAX: u32 = 25;
/// A binary16 subnormal `m` (from 1 to 1023) with its highest set bit at `msb` is the binary32
/// with exponent field `msb + 103` (`m * 2^-24 = 1.f * 2^(msb - 24)`, and `127 - 24 = 103`).
pub(crate) const F16_SUBNORMAL_EXPONENT_BASE: u32 = 127 - 24;

/// How many operations [`Emitter::narrow`] appends.
pub(crate) const NARROW_OPS: u64 = 27;
/// How many operations [`Emitter::widen`] appends.
pub(crate) const WIDEN_OPS: u64 = 20;
/// How many operations [`Emitter::pack_half`] appends.
pub(crate) const PACK_HALF_OPS: u64 = 41;
/// How many operations [`Emitter::unpack_half`] appends.
pub(crate) const UNPACK_HALF_OPS: u64 = 25;

/// `bits` (binary64) narrowed to binary32 as [`Emitter::narrow`] does it (module docs): the
/// model the pass uses for 64-bit float immediates.
#[must_use]
pub(crate) const fn narrow_f64(bits: u64) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the low and high words of a 64-bit value"
    )]
    let (lo, hi) = (bits as u32, (bits >> 32) as u32);
    let exponent = (hi >> F64_EXPONENT_SHIFT) & F64_EXPONENT_MAX;
    let sign = hi & SIGN_BIT;
    let high_mantissa = hi & F64_HIGH_MANTISSA_MASK;
    if exponent == F64_EXPONENT_MAX && (high_mantissa | lo) != 0 {
        return F32_CANONICAL_NAN;
    }
    if exponent >= F64_TO_F32_BIAS + F32_EXPONENT_MAX {
        return sign | F32_INFINITY;
    }
    if exponent <= F64_TO_F32_BIAS {
        return sign;
    }
    let mantissa = (high_mantissa << (32 - F64_LOW_KEPT_SHIFT)) | (lo >> F64_LOW_KEPT_SHIFT);
    let body = ((exponent - F64_TO_F32_BIAS) << F32_EXPONENT_SHIFT) | mantissa;
    let round = (lo >> F64_LOW_ROUND_BIT) & 1;
    let sticky = (lo & F64_LOW_STICKY_MASK != 0) as u32;
    sign | (body + (round & (sticky | (body & 1))))
}

/// `value` as a 32-bit immediate operand.
const fn imm(value: u32) -> Operand {
    Operand::Imm(Imm::U32(value))
}

/// Appends operations to one block, counting each as a step of the pass's [`Work`].
pub(crate) struct Emitter<'f, 'w> {
    builder: Builder<'f>,
    work: &'w mut Work,
}

impl<'f, 'w> Emitter<'f, 'w> {
    /// An emitter appending through `builder`.
    pub(crate) const fn new(builder: Builder<'f>, work: &'w mut Work) -> Self {
        Self { builder, work }
    }

    /// Appends `opcode` with `args` (default flags) and returns its value.
    pub(crate) fn op(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Operand, BuildError> {
        self.work.step();
        self.builder.op(opcode, args).map(Operand::Value)
    }

    /// `a` and `b` (both `u32`) combined by `opcode`.
    fn bin(&mut self, opcode: Opcode, a: Operand, b: Operand) -> Result<Operand, BuildError> {
        self.op(opcode, &[a, b])
    }

    /// `cond ? a : b`, all three `u32` but `cond`.
    fn select(&mut self, cond: Operand, a: Operand, b: Operand) -> Result<Operand, BuildError> {
        self.op(Opcode::SelectU32, &[cond, a, b])
    }

    /// The binary64 `(lo, hi)` narrowed to binary32 bits ([`NARROW_OPS`] operations; module
    /// docs). Mirrors [`narrow_f64`] step for step, with selects for its early returns.
    pub(crate) fn narrow(&mut self, lo: Operand, hi: Operand) -> Result<Operand, BuildError> {
        use Opcode as O;
        let exponent = self.op(
            O::BitFieldUExtract,
            &[hi, imm(F64_EXPONENT_SHIFT), imm(F64_EXPONENT_WIDTH)],
        )?;
        let sign = self.bin(O::BitwiseAnd32, hi, imm(SIGN_BIT))?;
        let high_mantissa = self.bin(O::BitwiseAnd32, hi, imm(F64_HIGH_MANTISSA_MASK))?;
        let shifted = self.bin(
            O::ShiftLeftLogical32,
            high_mantissa,
            imm(32 - F64_LOW_KEPT_SHIFT),
        )?;
        let kept_low = self.bin(O::ShiftRightLogical32, lo, imm(F64_LOW_KEPT_SHIFT))?;
        let mantissa = self.bin(O::BitwiseOr32, shifted, kept_low)?;
        let rebiased = self.bin(O::ISub32, exponent, imm(F64_TO_F32_BIAS))?;
        let placed = self.bin(O::ShiftLeftLogical32, rebiased, imm(F32_EXPONENT_SHIFT))?;
        let body = self.bin(O::BitwiseOr32, placed, mantissa)?;
        let round = self.op(
            O::BitFieldUExtract,
            &[lo, imm(F64_LOW_ROUND_BIT), imm(1)],
        )?;
        let sticky_bits = self.bin(O::BitwiseAnd32, lo, imm(F64_LOW_STICKY_MASK))?;
        let sticky = self.bin(O::INotEqual, sticky_bits, imm(0))?;
        let odd = self.bin(O::BitwiseAnd32, body, imm(1))?;
        let sticky_word = self.select(sticky, imm(1), imm(0))?;
        let tie_breaker = self.bin(O::BitwiseOr32, sticky_word, odd)?;
        let up = self.bin(O::BitwiseAnd32, round, tie_breaker)?;
        let rounded = self.bin(O::IAdd32, body, up)?;
        let payload = self.bin(O::BitwiseOr32, high_mantissa, lo)?;
        let has_payload = self.bin(O::INotEqual, payload, imm(0))?;
        let top = self.bin(O::IEqual, exponent, imm(F64_EXPONENT_MAX))?;
        let nan = self.bin(O::LogicalAnd, top, has_payload)?;
        let over = self.bin(
            O::UGreaterThanEqual,
            exponent,
            imm(F64_TO_F32_BIAS + F32_EXPONENT_MAX),
        )?;
        let under = self.bin(O::ULessThanEqual, exponent, imm(F64_TO_F32_BIAS))?;
        let magnitude = self.select(under, imm(0), rounded)?;
        let magnitude = self.select(over, imm(F32_INFINITY), magnitude)?;
        let signed = self.bin(O::BitwiseOr32, sign, magnitude)?;
        self.select(nan, imm(F32_CANONICAL_NAN), signed)
    }

    /// The binary32 `bits` widened to binary64, as `(lo, hi)` ([`WIDEN_OPS`] operations; module
    /// docs).
    pub(crate) fn widen(&mut self, bits: Operand) -> Result<(Operand, Operand), BuildError> {
        use Opcode as O;
        let sign = self.bin(O::BitwiseAnd32, bits, imm(SIGN_BIT))?;
        let exponent = self.op(
            O::BitFieldUExtract,
            &[bits, imm(F32_EXPONENT_SHIFT), imm(F32_EXPONENT_WIDTH)],
        )?;
        let mantissa = self.bin(O::BitwiseAnd32, bits, imm(F32_MANTISSA_MASK))?;
        let rebiased = self.bin(O::IAdd32, exponent, imm(F64_TO_F32_BIAS))?;
        let placed = self.bin(O::ShiftLeftLogical32, rebiased, imm(F64_EXPONENT_SHIFT))?;
        let high_mantissa = self.bin(
            O::ShiftRightLogical32,
            mantissa,
            imm(32 - F64_LOW_KEPT_SHIFT),
        )?;
        let magnitude = self.bin(O::BitwiseOr32, placed, high_mantissa)?;
        let normal_high = self.bin(O::BitwiseOr32, sign, magnitude)?;
        let normal_low = self.bin(O::ShiftLeftLogical32, mantissa, imm(F64_LOW_KEPT_SHIFT))?;
        let top = self.bin(O::IEqual, exponent, imm(F32_EXPONENT_MAX))?;
        let zero = self.bin(O::IEqual, exponent, imm(0))?;
        let has_payload = self.bin(O::INotEqual, mantissa, imm(0))?;
        let nan = self.bin(O::LogicalAnd, top, has_payload)?;
        let infinity_high = self.bin(O::BitwiseOr32, sign, imm(F64_INFINITY_HIGH))?;
        let high = self.select(zero, sign, normal_high)?;
        let high = self.select(top, infinity_high, high)?;
        let high = self.select(nan, imm(F64_CANONICAL_NAN_HIGH), high)?;
        let no_low = self.bin(O::LogicalOr, top, zero)?;
        let low = self.select(no_low, imm(0), normal_low)?;
        let low = self.select(nan, imm(F64_CANONICAL_NAN_LOW), low)?;
        Ok((low, high))
    }

    /// The binary32 `bits` narrowed to binary16, in the low half of a word whose high half is
    /// zero ([`PACK_HALF_OPS`] operations; module docs).
    pub(crate) fn pack_half(&mut self, bits: Operand) -> Result<Operand, BuildError> {
        use Opcode as O;
        let high_half = self.bin(O::ShiftRightLogical32, bits, imm(F16_SIGN_SHIFT))?;
        let sign = self.bin(O::BitwiseAnd32, high_half, imm(F16_SIGN_BIT))?;
        let magnitude = self.bin(O::BitwiseAnd32, bits, imm(F32_MAGNITUDE_MASK))?;
        let exponent = self.bin(O::ShiftRightLogical32, magnitude, imm(F32_EXPONENT_SHIFT))?;
        let mantissa = self.bin(O::BitwiseAnd32, magnitude, imm(F32_MANTISSA_MASK))?;
        // A normal binary16.
        let rebiased = self.bin(O::ISub32, exponent, imm(F32_TO_F16_BIAS))?;
        let placed = self.bin(O::ShiftLeftLogical32, rebiased, imm(F16_EXPONENT_SHIFT))?;
        let kept = self.bin(O::ShiftRightLogical32, mantissa, imm(F32_TO_F16_DROPPED))?;
        let body = self.bin(O::BitwiseOr32, placed, kept)?;
        let round = self.op(
            O::BitFieldUExtract,
            &[mantissa, imm(F32_TO_F16_ROUND_BIT), imm(1)],
        )?;
        let sticky_bits = self.bin(O::BitwiseAnd32, mantissa, imm(F32_TO_F16_STICKY_MASK))?;
        let sticky = self.bin(O::INotEqual, sticky_bits, imm(0))?;
        let odd = self.bin(O::BitwiseAnd32, body, imm(1))?;
        let sticky_word = self.select(sticky, imm(1), imm(0))?;
        let tie_breaker = self.bin(O::BitwiseOr32, sticky_word, odd)?;
        let up = self.bin(O::BitwiseAnd32, round, tie_breaker)?;
        let normal = self.bin(O::IAdd32, body, up)?;
        // A subnormal binary16 (or zero).
        let significand = self.bin(O::BitwiseOr32, mantissa, imm(F32_HIDDEN_BIT))?;
        let distance = self.bin(O::ISub32, imm(F16_SUBNORMAL_SHIFT_BASE), exponent)?;
        let shift = self.bin(O::UMin32, distance, imm(F16_SUBNORMAL_SHIFT_MAX))?;
        let truncated = self.bin(O::ShiftRightLogical32, significand, shift)?;
        let unit = self.bin(O::ShiftLeftLogical32, imm(1), shift)?;
        let mask = self.bin(O::ISub32, unit, imm(1))?;
        let remainder = self.bin(O::BitwiseAnd32, significand, mask)?;
        let half_shift = self.bin(O::ISub32, shift, imm(1))?;
        let halfway = self.bin(O::ShiftLeftLogical32, imm(1), half_shift)?;
        let above = self.bin(O::UGreaterThan, remainder, halfway)?;
        let tie = self.bin(O::IEqual, remainder, halfway)?;
        let low_bit = self.bin(O::BitwiseAnd32, truncated, imm(1))?;
        let truncated_odd = self.bin(O::INotEqual, low_bit, imm(0))?;
        let tie_up = self.bin(O::LogicalAnd, tie, truncated_odd)?;
        let round_up = self.bin(O::LogicalOr, above, tie_up)?;
        let increment = self.select(round_up, imm(1), imm(0))?;
        let subnormal = self.bin(O::IAdd32, truncated, increment)?;
        // The choice.
        let nan = self.bin(O::UGreaterThan, magnitude, imm(F32_INFINITY))?;
        let big = self.bin(
            O::UGreaterThanEqual,
            exponent,
            imm(F16_OVERFLOW_F32_EXPONENT),
        )?;
        let small = self.bin(
            O::ULessThan,
            exponent,
            imm(F16_NORMAL_MIN_F32_EXPONENT),
        )?;
        let half = self.select(small, subnormal, normal)?;
        let half = self.select(big, imm(F16_INFINITY), half)?;
        let signed = self.bin(O::BitwiseOr32, sign, half)?;
        self.select(nan, imm(F16_CANONICAL_NAN), signed)
    }

    /// The binary16 in the low half of `word` (whose high half must be zero) widened to binary32
    /// bits ([`UNPACK_HALF_OPS`] operations; module docs).
    pub(crate) fn unpack_half(&mut self, word: Operand) -> Result<Operand, BuildError> {
        use Opcode as O;
        let sign_bit = self.bin(O::BitwiseAnd32, word, imm(F16_SIGN_BIT))?;
        let sign = self.bin(O::ShiftLeftLogical32, sign_bit, imm(F16_SIGN_SHIFT))?;
        let exponent = self.op(
            O::BitFieldUExtract,
            &[word, imm(F16_EXPONENT_SHIFT), imm(F16_EXPONENT_WIDTH)],
        )?;
        let mantissa = self.bin(O::BitwiseAnd32, word, imm(F16_MANTISSA_MASK))?;
        // A normal binary16.
        let rebiased = self.bin(O::IAdd32, exponent, imm(F32_TO_F16_BIAS))?;
        let placed = self.bin(O::ShiftLeftLogical32, rebiased, imm(F32_EXPONENT_SHIFT))?;
        let wide = self.bin(O::ShiftLeftLogical32, mantissa, imm(F32_TO_F16_DROPPED))?;
        let normal = self.bin(O::BitwiseOr32, placed, wide)?;
        // A subnormal binary16, normalised.
        let msb = self.op(O::FindUMsb32, &[mantissa])?;
        let sub_exponent = self.bin(O::IAdd32, msb, imm(F16_SUBNORMAL_EXPONENT_BASE))?;
        let sub_placed = self.bin(O::ShiftLeftLogical32, sub_exponent, imm(F32_EXPONENT_SHIFT))?;
        let to_top = self.bin(O::ISub32, imm(F32_EXPONENT_SHIFT), msb)?;
        let raised = self.bin(O::ShiftLeftLogical32, mantissa, to_top)?;
        let fraction = self.bin(O::BitwiseAnd32, raised, imm(F32_MANTISSA_MASK))?;
        let subnormal = self.bin(O::BitwiseOr32, sub_placed, fraction)?;
        // The choice.
        let zero_exponent = self.bin(O::IEqual, exponent, imm(0))?;
        let zero_mantissa = self.bin(O::IEqual, mantissa, imm(0))?;
        let top = self.bin(O::IEqual, exponent, imm(F16_EXPONENT_MAX))?;
        let low = self.select(zero_mantissa, imm(0), subnormal)?;
        let magnitude = self.select(zero_exponent, low, normal)?;
        let magnitude = self.select(top, imm(F32_INFINITY), magnitude)?;
        let signed = self.bin(O::BitwiseOr32, sign, magnitude)?;
        let has_payload = self.op(O::LogicalNot, &[zero_mantissa])?;
        let nan = self.bin(O::LogicalAnd, top, has_payload)?;
        self.select(nan, imm(F32_CANONICAL_NAN), signed)
    }
}
