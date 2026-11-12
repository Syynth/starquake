//! The bytes a clear writes into one texel, for the formats whose answer is exact.
//!
//! **Bit layouts.** Each modelled format is laid out as the Vulkan format Eden maps it to
//! (`eden/src/video_core/renderer_vulkan/maxwell_to_vk.cpp` @ cf654052, the `SURFACE_FORMAT_ELEM`
//! table): a little-endian word with each channel at a bit offset. Vulkan's `_PACKnn` formats
//! name their channels from the most significant bit down (`A8B8G8R8_UNORM_PACK32`: red in bits
//! 0..8, alpha in bits 24..32); its other formats name them in memory order (`R16G16_UNORM`: red
//! in bytes 0..2). That naming rule is the Vulkan specification's ("Format Definition"), cited
//! from memory, not from a pinned copy; R5's differential test against wgpu confirms it.
//!
//! Formats left out, whose cleared texels are recorded as unknown:
//!
//! - `A1B5G5R5_UNORM`, `A5B5G5R1_UNORM` and `G4R4_UNORM`: Eden maps them to a Vulkan format with
//!   another channel order and fixes the order with a view swizzle, so the guest's bit layout is
//!   not the Vulkan format's;
//! - `R16G16B16X16_FLOAT`: what a clear writes into the X channel is the host's business;
//! - `B10G11R11_FLOAT` and `E5B9G9R9_FLOAT`: packed floats, whose rounding is not modelled;
//! - `X8_D24_UNORM` (the X bits) and every depth-stencil format (a copy or readback moves one
//!   aspect at a time on a real backend);
//! - every compressed format (the validation refuses clearing them).
//!
//! **Exact answers only.** A real device converts a clear value with its own rounding (Vulkan
//! allows either neighbour when a float is converted to a normalised integer, "Fixed-Point Data
//! Conversions"; again from memory). So a channel is known only when every conforming device
//! must write the same bits, and the whole texel is unknown otherwise:
//!
//! - unsigned normalised (sRGB channels included): the value clamped to 0..=1 (NaN: unknown)
//!   times `2^bits - 1` must be a whole number. A float is a power-of-two fraction and
//!   `2^bits - 1` is odd, so that holds only for 0 and 1: 0.5 on 8 bits is 127.5, which a device
//!   may write as 127 or 128;
//! - signed normalised: likewise only -1, 0 and 1 (times the odd `2^(bits-1) - 1`);
//! - 32-bit float: any value but NaN and subnormals (which a device may flush);
//! - 16-bit float: a value exactly representable as a normal binary16 number, zero or infinity;
//! - integers: a value representable in the channel's bits (Vulkan leaves others undefined);
//! - depth: within 0..=1, under the float or normalised rule of the format; stencil: 0..=255.
//!
//! **Buffer copies.** An upload, a readback and a copy between a buffer and a texture move bytes
//! exactly for every format but these, for which the oracle claims nothing
//! ([`buffer_copy_exact`]):
//!
//! - the two-aspect depth-stencil formats (`D24_UNORM_S8_UINT`, `S8_UINT_D24_UNORM`,
//!   `D32_FLOAT_S8_UINT`): Vulkan's buffer copies move one aspect per region, in a packing of
//!   their own, and wgpu cannot copy the depth aspect of `Depth24PlusStencil8` at all (guess:
//!   both from memory of their documentation, to be confirmed by slice R5);
//! - `X8_D24_UNORM`: the X byte of a texel copied to a buffer is undefined in Vulkan (guess, as
//!   above).
//!
//! A copy between two textures of one of these formats still moves bytes and their known flags,
//! but no byte of such a texture is ever known: uploads and buffer copies into it, clears of it
//! (left out above) and draws all leave it unknown.

use crate::command::ClearValue;
use crate::format::{FormatAspect, NumericType, PixelFormat};

/// A channel's position in a clear value: red, green, blue, alpha (depth and stencil use 0).
#[derive(Clone, Copy, Debug)]
enum Channel {
    R = 0,
    G = 1,
    B = 2,
    A = 3,
}

/// One channel's place in the texel's little-endian word: (channel, lowest bit, width).
type Field = (Channel, u32, u32);

use Channel::{A, B, G, R};

const RGBA8: &[Field] = &[(R, 0, 8), (G, 8, 8), (B, 16, 8), (A, 24, 8)];
const BGRA8: &[Field] = &[(B, 0, 8), (G, 8, 8), (R, 16, 8), (A, 24, 8)];
const RGBA16: &[Field] = &[(R, 0, 16), (G, 16, 16), (B, 32, 16), (A, 48, 16)];
const RGBA32: &[Field] = &[(R, 0, 32), (G, 32, 32), (B, 64, 32), (A, 96, 32)];
const RGB32: &[Field] = &[(R, 0, 32), (G, 32, 32), (B, 64, 32)];
const RG32: &[Field] = &[(R, 0, 32), (G, 32, 32)];
const RG16: &[Field] = &[(R, 0, 16), (G, 16, 16)];
const RG8: &[Field] = &[(R, 0, 8), (G, 8, 8)];
const R32: &[Field] = &[(R, 0, 32)];
const R16: &[Field] = &[(R, 0, 16)];
const R8: &[Field] = &[(R, 0, 8)];

/// The channel layout of a modelled format, or `None` for one this module leaves out (module
/// docs). Layouts follow the Vulkan format of Eden's `maxwell_to_vk.cpp` row for each format.
fn layout(format: PixelFormat) -> Option<&'static [Field]> {
    use PixelFormat as F;
    Some(match format {
        // VK_FORMAT_A8B8G8R8_*_PACK32
        F::A8B8G8R8Unorm
        | F::A8B8G8R8Snorm
        | F::A8B8G8R8Sint
        | F::A8B8G8R8Uint
        | F::A8B8G8R8Srgb => RGBA8,
        // VK_FORMAT_R5G6B5_UNORM_PACK16
        F::R5G6B5Unorm => &[(B, 0, 5), (G, 5, 6), (R, 11, 5)],
        // VK_FORMAT_B5G6R5_UNORM_PACK16
        F::B5G6R5Unorm => &[(R, 0, 5), (G, 5, 6), (B, 11, 5)],
        // VK_FORMAT_A1R5G5B5_UNORM_PACK16
        F::A1R5G5B5Unorm => &[(B, 0, 5), (G, 5, 5), (R, 10, 5), (A, 15, 1)],
        // VK_FORMAT_A2B10G10R10_{UNORM,UINT}_PACK32
        F::A2B10G10R10Unorm | F::A2B10G10R10Uint => {
            &[(R, 0, 10), (G, 10, 10), (B, 20, 10), (A, 30, 2)]
        }
        // VK_FORMAT_A2R10G10B10_UNORM_PACK32
        F::A2R10G10B10Unorm => &[(B, 0, 10), (G, 10, 10), (R, 20, 10), (A, 30, 2)],
        // VK_FORMAT_A4B4G4R4_UNORM_PACK16_EXT
        F::A4B4G4R4Unorm => &[(R, 0, 4), (G, 4, 4), (B, 8, 4), (A, 12, 4)],
        // VK_FORMAT_S8_UINT is one byte, like the R8 formats.
        F::R8Unorm | F::R8Snorm | F::R8Sint | F::R8Uint | F::S8Uint => R8,
        F::R8G8Unorm | F::R8G8Snorm | F::R8G8Sint | F::R8G8Uint => RG8,
        // VK_FORMAT_D16_UNORM is one 16-bit channel.
        F::R16Float | F::R16Unorm | F::R16Snorm | F::R16Uint | F::R16Sint | F::D16Unorm => R16,
        F::R16G16Unorm | F::R16G16Float | F::R16G16Uint | F::R16G16Sint | F::R16G16Snorm => RG16,
        F::R16G16B16A16Float
        | F::R16G16B16A16Unorm
        | F::R16G16B16A16Snorm
        | F::R16G16B16A16Sint
        | F::R16G16B16A16Uint => RGBA16,
        // VK_FORMAT_D32_SFLOAT is one 32-bit channel.
        F::R32Float | F::R32Uint | F::R32Sint | F::D32Float => R32,
        F::R32G32Float | F::R32G32Sint | F::R32G32Uint => RG32,
        F::R32G32B32Float => RGB32,
        F::R32G32B32A32Uint | F::R32G32B32A32Float | F::R32G32B32A32Sint => RGBA32,
        // VK_FORMAT_B8G8R8A8_{UNORM,SRGB}
        F::B8G8R8A8Unorm | F::B8G8R8A8Srgb => BGRA8,
        _ => return None,
    })
}

/// Whether a copy between a buffer and a texture of `format` (an upload, a readback, a buffer
/// copy) moves bytes exactly: not for a two-aspect depth-stencil format nor `X8_D24_UNORM`
/// (module docs).
pub(super) fn buffer_copy_exact(format: PixelFormat) -> bool {
    format.aspect() != FormatAspect::DepthStencil && format != PixelFormat::X8D24Unorm
}

/// The bytes one texel of `format` holds after a clear to `value`, or `None` when the answer is
/// not exact on every device or the format is not modelled (module docs). A value whose kind
/// does not suit the format gives `None` too; the validation refuses those first.
pub(super) fn encode_clear(format: PixelFormat, value: ClearValue) -> Option<Vec<u8>> {
    let fields = layout(format)?;
    let numeric = format.numeric();
    let mut word: u128 = 0;
    for &(channel, shift, bits) in fields {
        let i = channel as usize;
        let encoded = match (value, numeric) {
            (ClearValue::Float(c), NumericType::Unorm) => {
                let v = *c.get(i)?;
                // sRGB colour channels and linear ones are exact at the same two points.
                unorm(v, bits)?
            }
            (ClearValue::Float(c), NumericType::Snorm) => snorm(*c.get(i)?, bits)?,
            (ClearValue::Float(c), NumericType::Float) => float(*c.get(i)?, bits)?,
            (ClearValue::Uint(c), NumericType::Uint) => uint(*c.get(i)?, bits)?,
            (ClearValue::Sint(c), NumericType::Sint) => sint(*c.get(i)?, bits)?,
            (ClearValue::Depth(d), NumericType::Unorm) if format.has_depth() => {
                depth_unorm(d, bits)?
            }
            (ClearValue::Depth(d), NumericType::Float) if format.has_depth() => depth_float(d)?,
            (ClearValue::Stencil(s), NumericType::Uint) if format.has_stencil() => uint(s, bits)?,
            _ => return None,
        };
        word |= u128::from(encoded) << shift;
    }
    let len = usize::try_from(format.bytes_per_block()).ok()?;
    Some(word.to_le_bytes().get(..len)?.to_vec())
}

/// `v` as a `bits`-wide unsigned normalised integer, when exact: only for the clamped
/// endpoints 0 and 1 (module docs).
fn unorm(v: f32, bits: u32) -> Option<u64> {
    if v.is_nan() {
        None
    } else if v <= 0.0 {
        Some(0)
    } else if v >= 1.0 {
        Some(max_unsigned(bits))
    } else {
        None
    }
}

/// `v` as a `bits`-wide signed normalised integer, two's complement, when exact: only for the
/// clamped endpoints -1 and 1, and 0 (module docs).
fn snorm(v: f32, bits: u32) -> Option<u64> {
    let max = i64::try_from(max_unsigned(bits - 1)).ok()?;
    let n = if v.is_nan() {
        return None;
    } else if v <= -1.0 {
        -max
    } else if v >= 1.0 {
        max
    } else if v == 0.0 {
        0
    } else {
        return None;
    };
    Some(twos_complement(n, bits))
}

/// `2^bits - 1`.
const fn max_unsigned(bits: u32) -> u64 {
    (1_u64 << bits) - 1
}

/// A float channel, 32 or 16 bits wide.
fn float(v: f32, bits: u32) -> Option<u64> {
    match bits {
        32 if !v.is_nan() && !v.is_subnormal() => Some(v.to_bits().into()),
        16 => f16_exact(v).map(u64::from),
        _ => None,
    }
}

/// An unsigned integer channel, when the value fits.
fn uint(v: u32, bits: u32) -> Option<u64> {
    let v = u64::from(v);
    (v >> bits == 0).then_some(v)
}

/// A signed integer channel, two's complement, when the value fits.
fn sint(v: i32, bits: u32) -> Option<u64> {
    let v = i64::from(v);
    let half = 1_i64 << (bits - 1);
    (-half..half).contains(&v).then(|| twos_complement(v, bits))
}

fn depth_unorm(d: f32, bits: u32) -> Option<u64> {
    if (0.0..=1.0).contains(&d) {
        unorm(d, bits)
    } else {
        None
    }
}

fn depth_float(d: f32) -> Option<u64> {
    if (0.0..=1.0).contains(&d) {
        float(d, 32)
    } else {
        None
    }
}

/// The low `bits` bits of `n` in two's complement.
fn twos_complement(n: i64, bits: u32) -> u64 {
    u64::from_ne_bytes(n.to_ne_bytes()) & max_unsigned(bits)
}

/// Exponent bias of binary16 (IEEE 754).
const F16_BIAS: i32 = 15;
/// Exponent bias of binary32 (IEEE 754).
const F32_BIAS: i32 = 127;
/// Mantissa bits of binary32 that binary16 lacks (23 - 10).
const MANTISSA_DROP: u32 = 13;

/// `v` as binary16, when it is zero, infinite, or a normal binary16 number exactly. Subnormal
/// binary16 values are left out: a device may flush them.
pub(super) fn f16_exact(v: f32) -> Option<u16> {
    let bits = v.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).ok()?;
    if v.is_nan() {
        return None;
    }
    if v.is_infinite() {
        return Some(sign | 0x7c00);
    }
    if v == 0.0 {
        return Some(sign);
    }
    let exponent = i32::try_from((bits >> 23) & 0xff).ok()? - F32_BIAS;
    let mantissa = bits & 0x007f_ffff;
    if !(1 - F16_BIAS..=F16_BIAS).contains(&exponent) {
        return None;
    }
    if mantissa & ((1 << MANTISSA_DROP) - 1) != 0 {
        return None;
    }
    let e = u16::try_from(exponent + F16_BIAS).ok()?;
    let m = u16::try_from(mantissa >> MANTISSA_DROP).ok()?;
    Some(sign | (e << 10) | m)
}

#[cfg(test)]
mod tests;
