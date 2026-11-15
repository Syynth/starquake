//! Colour endpoints: unquantizing a block's colour values (C.2.13) and turning them into the
//! two colours of a partition by its endpoint mode (C.2.14 for LDR, C.2.15 for HDR).
//!
//! Ported from Eden's `astc.cpp` (`DecodeColorValues`, `ComputeEndpoints`,
//! `BitTransferSigned`, `BlueContract`, `DecodeHDREndpointMode7`, `DecodeHDREndpointMode11`);
//! see the parent module for the attribution.

use super::integer::{Digit, Encoding, Value, replicate};

/// The value of 1.0 in the 12-bit HDR representation (C.2.15: "0x780 is 1.0f").
const HDR_ONE: i32 = 0x780;
/// The largest 12-bit HDR value.
const HDR_MAX: i32 = 0xFFF;
/// Bits 10..7 of HDR mode 15's first alpha, which come from `v7` (C.2.15:
/// "v6 |= (v7 << (mode+1)) & 0x780"). The same number as [`HDR_ONE`], by coincidence.
const HDR_ALPHA_TOP_BITS: i32 = 0x780;

/// The two colours of a partition, as red, green, blue, alpha.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Endpoints {
    /// The colour at weight 0.
    pub(super) e0: [u32; 4],
    /// The colour at weight 64.
    pub(super) e1: [u32; 4],
    /// Whether red, green and blue are 12-bit HDR values rather than bytes.
    pub(super) hdr_rgb: bool,
    /// Whether alpha is a 12-bit HDR value rather than a byte.
    pub(super) hdr_alpha: bool,
}

/// How many colour values an endpoint mode takes (table C.2.17): two for each class step.
pub(super) const fn value_count(mode: u32) -> usize {
    match mode >> 2 {
        0 => 2,
        1 => 4,
        2 => 6,
        _ => 8,
    }
}

/// Unquantizes one colour value to 0..255 (C.2.13, table C.2.16), or `None` for an encoding
/// colours never have (a range below 0..5).
pub(super) fn unquantize_color(encoding: Encoding, value: Value) -> Option<i32> {
    let bits = value.bits;
    // The table's A: the lowest bit, nine times over.
    let fill = if bits & 1 == 0 { 0 } else { 0x1FF };
    // The bits above the lowest: "b", "cb", "dcb", "edcb" or "fedcb" in the table's B column.
    let x = bits >> 1;
    let (b, c) = match (encoding.digit, encoding.bits) {
        (Digit::None, 1..=8) => return i32::try_from(replicate(bits, encoding.bits, 8)).ok(),
        (Digit::Trit, 1) => (0, 204),
        (Digit::Trit, 2) => ((x << 8) | (x << 4) | (x << 2) | (x << 1), 93),
        (Digit::Trit, 3) => ((x << 7) | (x << 2) | x, 44),
        (Digit::Trit, 4) => ((x << 6) | x, 22),
        (Digit::Trit, 5) => ((x << 5) | (x >> 2), 11),
        (Digit::Trit, 6) => ((x << 4) | (x >> 4), 5),
        (Digit::Quint, 1) => (0, 113),
        (Digit::Quint, 2) => ((x << 8) | (x << 3) | (x << 2), 54),
        (Digit::Quint, 3) => ((x << 7) | (x << 1) | (x >> 1), 26),
        (Digit::Quint, 4) => ((x << 6) | (x >> 1), 13),
        (Digit::Quint, 5) => ((x << 5) | (x >> 3), 6),
        _ => return None,
    };
    let mixed = (value.digit * c + b) ^ fill;
    i32::try_from((fill & 0x80) | (mixed >> 2)).ok()
}

/// C.2.14's `bit_transfer_signed`: moves the top bit of `a` into `b` and leaves `a` a signed
/// value of -32..31. Returns the new `(a, b)`.
const fn bit_transfer_signed(a: i32, b: i32) -> (i32, i32) {
    let b = (b >> 1) | (a & 0x80);
    let a = (a >> 1) & 0x3F;
    (if a & 0x20 == 0 { a } else { a - 0x40 }, b)
}

/// C.2.14's `blue_contract`.
const fn blue_contract(r: i32, g: i32, b: i32, a: i32) -> [i32; 4] {
    [(r + b) >> 1, (g + b) >> 1, b, a]
}

/// An LDR pair, clamped to bytes (C.2.14's `clamp_unorm8`; a no-op for the modes that need
/// none).
fn ldr(e0: [i32; 4], e1: [i32; 4]) -> Endpoints {
    let clamp = |color: [i32; 4]| color.map(|c| c.clamp(0, 0xFF).unsigned_abs());
    Endpoints {
        e0: clamp(e0),
        e1: clamp(e1),
        hdr_rgb: false,
        hdr_alpha: false,
    }
}

/// An HDR pair. `alpha` is each endpoint's alpha: 12-bit HDR values when `hdr_alpha`, bytes
/// otherwise (mode 14).
fn hdr(e0: [i32; 3], e1: [i32; 3], alpha: [i32; 2], hdr_alpha: bool) -> Endpoints {
    let pack = |rgb: [i32; 3], a: i32| [rgb[0], rgb[1], rgb[2], a].map(i32::unsigned_abs);
    Endpoints {
        e0: pack(e0, alpha[0]),
        e1: pack(e1, alpha[1]),
        hdr_rgb: true,
        hdr_alpha,
    }
}

/// HDR mode 7, RGB base and scale (C.2.15).
fn hdr_base_scale(v0: i32, v1: i32, v2: i32, v3: i32) -> ([i32; 3], [i32; 3]) {
    let modeval = ((v0 & 0xC0) >> 6) | ((v1 & 0x80) >> 5) | ((v2 & 0x80) >> 4);
    let (majcomp, mode) = if modeval & 0xC != 0xC {
        (modeval >> 2, modeval & 3)
    } else if modeval != 0xF {
        (modeval & 3, 4)
    } else {
        (0, 5)
    };

    let mut red = v0 & 0x3F;
    let mut green = v1 & 0x1F;
    let mut blue = v2 & 0x1F;
    let mut scale = v3 & 0x1F;

    let x0 = (v1 >> 6) & 1;
    let x1 = (v1 >> 5) & 1;
    let x2 = (v2 >> 6) & 1;
    let x3 = (v2 >> 5) & 1;
    let x4 = (v3 >> 7) & 1;
    let x5 = (v3 >> 6) & 1;
    let x6 = (v3 >> 5) & 1;

    // Which of the modes 0 to 5 take each extra bit, as a mask indexed by the mode.
    let ohm = 1 << mode;
    let when = |modes: i32, bit: i32| if ohm & modes == 0 { 0 } else { bit };
    green |= when(0x30, x0 << 6);
    green |= when(0x3A, x1 << 5);
    blue |= when(0x30, x2 << 6);
    blue |= when(0x3A, x3 << 5);
    scale |= when(0x3D, x6 << 5);
    scale |= when(0x2D, x5 << 6);
    scale |= when(0x04, x4 << 7);
    red |= when(0x3B, x4 << 6);
    red |= when(0x04, x3 << 6);
    red |= when(0x10, x5 << 7);
    red |= when(0x0F, x2 << 7);
    red |= when(0x05, x1 << 8);
    red |= when(0x0A, x0 << 8);
    red |= when(0x05, x0 << 9);
    red |= when(0x02, x6 << 9);
    red |= when(0x01, x3 << 10);
    red |= when(0x02, x5 << 10);

    let shift = [1, 1, 2, 3, 4, 5][mode.unsigned_abs() as usize % 6];
    red <<= shift;
    green <<= shift;
    blue <<= shift;
    scale <<= shift;

    if mode != 5 {
        green = red - green;
        blue = red - blue;
    }
    match majcomp {
        1 => core::mem::swap(&mut red, &mut green),
        2 => core::mem::swap(&mut red, &mut blue),
        _ => {}
    }

    let clamp = |c: i32| c.clamp(0, HDR_MAX);
    (
        [
            clamp(red - scale),
            clamp(green - scale),
            clamp(blue - scale),
        ],
        [clamp(red), clamp(green), clamp(blue)],
    )
}

/// `value`'s low `bits` bits as a signed number; the bits above them are dropped.
const fn sign_extend(value: i32, bits: u32) -> i32 {
    let sign = 1 << (bits - 1);
    ((value & ((1 << bits) - 1)) ^ sign) - sign
}

/// HDR mode 11, RGB direct (C.2.15); modes 14 and 15 use it for their colours.
fn hdr_direct(v: [i32; 6]) -> ([i32; 3], [i32; 3]) {
    let [v0, v1, v2, v3, v4, v5] = v;
    let majcomp = ((v4 & 0x80) >> 7) | ((v5 & 0x80) >> 6);
    if majcomp == 3 {
        return (
            [v0 << 4, v2 << 4, (v4 & 0x7F) << 5],
            [v1 << 4, v3 << 4, (v5 & 0x7F) << 5],
        );
    }

    let mode = ((v1 & 0x80) >> 7) | ((v2 & 0x80) >> 6) | ((v3 & 0x80) >> 5);
    let mut va = v0 | ((v1 & 0x40) << 2);
    let mut vb0 = v2 & 0x3F;
    let mut vb1 = v3 & 0x3F;
    let mut vc = v1 & 0x3F;
    // d0 and d1 are 5, 6 or 7 bits wide by the mode (table C.2.23); the bits of v4 and v5
    // above them are the extra bits X2 to X5, which belong to other fields in those modes.
    let d_bits = [7, 6, 7, 6, 5, 6, 5, 6][mode.unsigned_abs() as usize % 8];
    let mut vd0 = sign_extend(v4, d_bits);
    let mut vd1 = sign_extend(v5, d_bits);

    let x0 = (v2 >> 6) & 1;
    let x1 = (v3 >> 6) & 1;
    let x2 = (v4 >> 6) & 1;
    let x3 = (v5 >> 6) & 1;
    let x4 = (v4 >> 5) & 1;
    let x5 = (v5 >> 5) & 1;

    // Which of the modes 0 to 7 take each extra bit, as a mask indexed by the mode.
    let ohm = 1 << mode;
    let when = |modes: i32, bit: i32| if ohm & modes == 0 { 0 } else { bit };
    va |= when(0xA4, x0 << 9);
    va |= when(0x08, x2 << 9);
    va |= when(0x50, x4 << 9);
    va |= when(0x50, x5 << 10);
    va |= when(0xA0, x1 << 10);
    va |= when(0xC0, x2 << 11);
    vc |= when(0x04, x1 << 6);
    vc |= when(0xE8, x3 << 6);
    vc |= when(0x20, x2 << 7);
    vb0 |= when(0x5B, x0 << 6);
    vb1 |= when(0x5B, x1 << 6);
    vb0 |= when(0x12, x2 << 7);
    vb1 |= when(0x12, x3 << 7);

    let shift = (mode >> 1) ^ 3;
    va <<= shift;
    vb0 <<= shift;
    vb1 <<= shift;
    vc <<= shift;
    vd0 <<= shift;
    vd1 <<= shift;

    let clamp = |c: i32| c.clamp(0, HDR_MAX);
    let mut e1 = [clamp(va), clamp(va - vb0), clamp(va - vb1)];
    let mut e0 = [
        clamp(va - vc),
        clamp(va - vb0 - vc - vd0),
        clamp(va - vb1 - vc - vd1),
    ];
    match majcomp {
        1 => {
            e0.swap(0, 1);
            e1.swap(0, 1);
        }
        2 => {
            e0.swap(0, 2);
            e1.swap(0, 2);
        }
        _ => {}
    }
    (e0, e1)
}

/// HDR mode 15's alphas, from `v6` and `v7` (C.2.15, table C.2.24).
fn hdr_alpha(v6: i32, v7: i32) -> [i32; 2] {
    let mode = ((v6 >> 7) & 1) | ((v7 >> 6) & 2);
    let mut a = v6 & 0x7F;
    let mut b = v7 & 0x7F;
    if mode == 3 {
        return [a << 5, b << 5];
    }
    a |= (b << (mode + 1)) & HDR_ALPHA_TOP_BITS;
    b &= 0x3F >> mode;
    b ^= 0x20 >> mode;
    b -= 0x20 >> mode;
    a <<= 4 - mode;
    b <<= 4 - mode;
    [a, (b + a).clamp(0, HDR_MAX)]
}

/// The endpoints of a partition with endpoint mode `mode` (0 to 15), from its unquantized
/// values `v`, each 0..255. `None` when `v` is not the [`value_count`] values the mode takes.
pub(super) fn decode(mode: u32, v: &[i32]) -> Option<Endpoints> {
    match mode {
        2 | 3 | 7 | 11 | 14 | 15 => decode_hdr(mode, v),
        _ => decode_ldr(mode, v),
    }
}

/// The LDR endpoint modes (C.2.14).
fn decode_ldr(mode: u32, v: &[i32]) -> Option<Endpoints> {
    let opaque = 0xFF;
    Some(match (mode, v) {
        (0, &[v0, v1]) => ldr([v0, v0, v0, opaque], [v1, v1, v1, opaque]),
        (1, &[v0, v1]) => {
            let l0 = (v0 >> 2) | (v1 & 0xC0);
            let l1 = l0 + (v1 & 0x3F);
            ldr([l0, l0, l0, opaque], [l1, l1, l1, opaque])
        }
        (4, &[v0, v1, v2, v3]) => ldr([v0, v0, v0, v2], [v1, v1, v1, v3]),
        (5, &[v0, v1, v2, v3]) => {
            let (v1, v0) = bit_transfer_signed(v1, v0);
            let (v3, v2) = bit_transfer_signed(v3, v2);
            ldr([v0, v0, v0, v2], [v0 + v1, v0 + v1, v0 + v1, v2 + v3])
        }
        (6, &[v0, v1, v2, v3]) => ldr(
            [(v0 * v3) >> 8, (v1 * v3) >> 8, (v2 * v3) >> 8, opaque],
            [v0, v1, v2, opaque],
        ),
        (8, &[v0, v1, v2, v3, v4, v5]) => {
            if v1 + v3 + v5 >= v0 + v2 + v4 {
                ldr([v0, v2, v4, opaque], [v1, v3, v5, opaque])
            } else {
                ldr(
                    blue_contract(v1, v3, v5, opaque),
                    blue_contract(v0, v2, v4, opaque),
                )
            }
        }
        (9, &[v0, v1, v2, v3, v4, v5]) => {
            let (v1, v0) = bit_transfer_signed(v1, v0);
            let (v3, v2) = bit_transfer_signed(v3, v2);
            let (v5, v4) = bit_transfer_signed(v5, v4);
            if v1 + v3 + v5 >= 0 {
                ldr([v0, v2, v4, opaque], [v0 + v1, v2 + v3, v4 + v5, opaque])
            } else {
                ldr(
                    blue_contract(v0 + v1, v2 + v3, v4 + v5, opaque),
                    blue_contract(v0, v2, v4, opaque),
                )
            }
        }
        (10, &[v0, v1, v2, v3, v4, v5]) => ldr(
            [(v0 * v3) >> 8, (v1 * v3) >> 8, (v2 * v3) >> 8, v4],
            [v0, v1, v2, v5],
        ),
        (12, &[v0, v1, v2, v3, v4, v5, v6, v7]) => {
            if v1 + v3 + v5 >= v0 + v2 + v4 {
                ldr([v0, v2, v4, v6], [v1, v3, v5, v7])
            } else {
                ldr(blue_contract(v1, v3, v5, v7), blue_contract(v0, v2, v4, v6))
            }
        }
        (13, &[v0, v1, v2, v3, v4, v5, v6, v7]) => {
            let (v1, v0) = bit_transfer_signed(v1, v0);
            let (v3, v2) = bit_transfer_signed(v3, v2);
            let (v5, v4) = bit_transfer_signed(v5, v4);
            let (v7, v6) = bit_transfer_signed(v7, v6);
            if v1 + v3 + v5 >= 0 {
                ldr([v0, v2, v4, v6], [v0 + v1, v2 + v3, v4 + v5, v6 + v7])
            } else {
                ldr(
                    blue_contract(v0 + v1, v2 + v3, v4 + v5, v6 + v7),
                    blue_contract(v0, v2, v4, v6),
                )
            }
        }
        _ => return None,
    })
}

/// The HDR endpoint modes (C.2.15).
fn decode_hdr(mode: u32, v: &[i32]) -> Option<Endpoints> {
    Some(match (mode, v) {
        (2, &[v0, v1]) => {
            let (y0, y1) = if v1 >= v0 {
                (v0 << 4, v1 << 4)
            } else {
                ((v1 << 4) + 8, (v0 << 4) - 8)
            };
            hdr([y0; 3], [y1; 3], [HDR_ONE; 2], true)
        }
        (3, &[v0, v1]) => {
            let (y0, d) = if v0 & 0x80 == 0 {
                (((v1 & 0xF0) << 4) | ((v0 & 0x7F) << 1), (v1 & 0x0F) << 1)
            } else {
                (((v1 & 0xE0) << 4) | ((v0 & 0x7F) << 2), (v1 & 0x1F) << 2)
            };
            let y1 = (y0 + d).min(HDR_MAX);
            hdr([y0; 3], [y1; 3], [HDR_ONE; 2], true)
        }
        (7, &[v0, v1, v2, v3]) => {
            let (e0, e1) = hdr_base_scale(v0, v1, v2, v3);
            hdr(e0, e1, [HDR_ONE; 2], true)
        }
        (11, &[v0, v1, v2, v3, v4, v5]) => {
            let (e0, e1) = hdr_direct([v0, v1, v2, v3, v4, v5]);
            hdr(e0, e1, [HDR_ONE; 2], true)
        }
        (14, &[v0, v1, v2, v3, v4, v5, v6, v7]) => {
            let (e0, e1) = hdr_direct([v0, v1, v2, v3, v4, v5]);
            hdr(e0, e1, [v6, v7], false)
        }
        (15, &[v0, v1, v2, v3, v4, v5, v6, v7]) => {
            let (e0, e1) = hdr_direct([v0, v1, v2, v3, v4, v5]);
            hdr(e0, e1, hdr_alpha(v6, v7), true)
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
