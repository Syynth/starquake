//! Integer semantics where SPIR-V leaves a result undefined, given the defined results the
//! parent module lists (shifts past the width, bit fields past bit 31, division by zero), and
//! the GLSL.std.450 bit searches.

/// The bit width of a 32-bit operand, as a shift count.
const WIDTH32: u32 = u32::BITS;

/// `value << shift`, 0 for a shift of the width or more (PR #4588's `ExecuteNativeWarp`).
pub(super) const fn shl32(value: u32, shift: u32) -> u32 {
    match value.checked_shl(shift) {
        Some(shifted) => shifted,
        None => 0,
    }
}

/// `value >> shift` (logical), 0 for a shift of the width or more.
pub(super) const fn shr32(value: u32, shift: u32) -> u32 {
    match value.checked_shr(shift) {
        Some(shifted) => shifted,
        None => 0,
    }
}

/// `value >> shift` (arithmetic), the sign bit everywhere for a shift of the width or more.
pub(super) const fn sar32(value: u32, shift: u32) -> u32 {
    let shift = if shift >= WIDTH32 { WIDTH32 - 1 } else { shift };
    (value.cast_signed() >> shift).cast_unsigned()
}

/// `value << shift` on 64 bits, 0 for a shift of 64 or more.
pub(super) const fn shl64(value: u64, shift: u32) -> u64 {
    match value.checked_shl(shift) {
        Some(shifted) => shifted,
        None => 0,
    }
}

/// `value >> shift` (logical) on 64 bits, 0 for a shift of 64 or more.
pub(super) const fn shr64(value: u64, shift: u32) -> u64 {
    match value.checked_shr(shift) {
        Some(shifted) => shifted,
        None => 0,
    }
}

/// `value >> shift` (arithmetic) on 64 bits, the sign bit everywhere for a shift of 64 or
/// more.
pub(super) const fn sar64(value: u64, shift: u32) -> u64 {
    let shift = if shift >= u64::BITS {
        u64::BITS - 1
    } else {
        shift
    };
    (value.cast_signed() >> shift).cast_unsigned()
}

/// A mask of the low `count` bits, `count` at most 32, as 64 bits.
const fn low_mask(count: u32) -> u64 {
    (1_u64 << count) - 1
}

/// `count` bits of `base` from bit `offset`, zero-extended; bits above 31 read as zero, and
/// offset and count are clamped to 32 (`BitFieldUExtract`).
pub(super) const fn bitfield_uextract(base: u32, offset: u32, count: u32) -> u32 {
    let offset = if offset > WIDTH32 { WIDTH32 } else { offset };
    let count = if count > WIDTH32 { WIDTH32 } else { count };
    let field = ((base as u64) >> offset) & low_mask(count);
    super::value::low32(field)
}

/// [`bitfield_uextract`] sign-extended from the field's top bit; a zero count gives 0
/// (`BitFieldSExtract`).
pub(super) const fn bitfield_sextract(base: u32, offset: u32, count: u32) -> u32 {
    let count = if count > WIDTH32 { WIDTH32 } else { count };
    if count == 0 {
        return 0;
    }
    let field = bitfield_uextract(base, offset, count);
    let spare = WIDTH32 - count;
    ((field << spare).cast_signed() >> spare).cast_unsigned()
}

/// `base` with `count` bits from bit `offset` replaced by the low bits of `insert`; bits that
/// would land above bit 31 are dropped, and offset and count are clamped to 32
/// (`BitFieldInsert`).
pub(super) const fn bitfield_insert(base: u32, insert: u32, offset: u32, count: u32) -> u32 {
    let offset = if offset > WIDTH32 { WIDTH32 } else { offset };
    let count = if count > WIDTH32 { WIDTH32 } else { count };
    let mask = super::value::low32(low_mask(count) << offset);
    let moved = super::value::low32((insert as u64) << offset);
    (base & !mask) | (moved & mask)
}

/// The index of the most significant set bit, or `u32::MAX` (-1) for 0 (GLSL.std.450
/// `FindUMsb`).
pub(super) const fn find_umsb(value: u32) -> u32 {
    match value.checked_ilog2() {
        Some(bit) => bit,
        None => u32::MAX,
    }
}

/// The index of the most significant bit that differs from the sign bit, or `u32::MAX` (-1) for
/// 0 and -1 (GLSL.std.450 `FindSMsb`).
pub(super) const fn find_smsb(value: u32) -> u32 {
    if value.cast_signed() < 0 {
        find_umsb(!value)
    } else {
        find_umsb(value)
    }
}

/// `a / b` unsigned; `0xffff_ffff` for a zero divisor (a guess).
pub(super) const fn udiv32(a: u32, b: u32) -> u32 {
    match a.checked_div(b) {
        Some(quotient) => quotient,
        None => u32::MAX,
    }
}

/// `a / b` signed, truncating; -1 for a zero divisor (a guess), `i32::MIN` for `i32::MIN / -1`.
pub(super) const fn sdiv32(a: u32, b: u32) -> u32 {
    let (a, b) = (a.cast_signed(), b.cast_signed());
    if b == 0 {
        return u32::MAX;
    }
    a.wrapping_div(b).cast_unsigned()
}

/// Increment with wrap at `limit` (the shared, global and storage `AtomicInc32`): 0 when
/// `old >= limit`, else `old + 1`.
pub(super) const fn atomic_inc(old: u32, limit: u32) -> u32 {
    if old >= limit { 0 } else { old + 1 }
}

/// Decrement with wrap at `limit` (`AtomicDec32`): `limit` when `old == 0 || old > limit`, else
/// `old - 1`.
pub(super) const fn atomic_dec(old: u32, limit: u32) -> u32 {
    if old == 0 || old > limit {
        limit
    } else {
        old - 1
    }
}
