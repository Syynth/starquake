//! Decoding one block: its layout (C.2.9 to C.2.11), the checks for illegal encodings
//! (C.2.24), the amount of colour data (C.2.22), and the interpolation of each texel (C.2.19,
//! C.2.20), plus void-extent blocks (C.2.23).
//!
//! Ported from Eden's `astc.cpp` (`DecompressBlock`, `FillVoidExtentLDR`, `HalfToFloat`,
//! `HalfToClampedByte`); see the parent module for the attribution and for where this follows
//! the specification instead.

use super::endpoints::{self, Endpoints};
use super::integer::{BitReader, Encoding, Value, decode_sequence, mask};
use super::mode::{BlockMode, MODE_BITS, WeightGrid, block_mode};
use super::weights::{self, MAX_WEIGHTS};
use super::{ERROR_COLOR, Footprint};
use crate::format::ColorEncoding;

/// The fewest and the most bits a block's weights may take (C.2.24).
const WEIGHT_BITS: core::ops::RangeInclusive<u32> = 24..=96;
/// The most colour values a block may have (C.2.24).
const MAX_COLOR_VALUES: usize = 18;
/// The most partitions a block has.
const MAX_PARTITIONS: usize = 4;
/// Bits in a block.
const BLOCK_BITS: u32 = 128;
/// A footprint with fewer texels than this is a "small block" for partitioning (C.2.21).
const SMALL_BLOCK_TEXELS: u32 = 31;

/// The colour ranges a block can use, widest first (table C.2.16, and plain bits up to 8).
const COLOR_ENCODINGS: [Encoding; 17] = {
    let tops = [
        255, 191, 159, 127, 95, 79, 63, 47, 39, 31, 23, 19, 15, 11, 9, 7, 5,
    ];
    let mut encodings = [Encoding::for_max(0); 17];
    let mut index = 0;
    while index < tops.len() {
        encodings[index] = Encoding::for_max(tops[index]);
        index += 1;
    }
    encodings
};

/// A number that is at most 255 as a byte; anything larger saturates.
fn byte(value: u32) -> u8 {
    u8::try_from(value).unwrap_or(u8::MAX)
}

/// An FP16 bit pattern as a byte: the value clamped to 0..1, times 255, rounded half up.
/// A NaN is 0.
///
/// Worked in integers: a positive value below 1.0 is `m * 2^-shift` for an 11-bit `m`, and
/// `floor(255 * m / 2^shift + 1/2)` is exact where the same sum in `f32` (Eden's
/// `HalfToClampedByte`) would be, since 255 * m needs 19 bits and the sum 24.
/// `halves_as_bytes_match_float_arithmetic` compares all 65,536 patterns.
pub(super) fn half_to_byte(half: u32) -> u8 {
    let negative = half & 0x8000 != 0;
    let exponent = (half >> 10) & 0x1F;
    let mantissa = half & 0x3FF;
    if exponent == 0x1F {
        // Infinities clamp like any other value; a NaN, of either sign, is 0.
        return if mantissa == 0 && !negative {
            u8::MAX
        } else {
            0
        };
    }
    if negative {
        return 0;
    }
    if exponent >= 15 {
        return u8::MAX;
    }
    // The value is (1024 + mantissa) * 2^(exponent - 25). A subnormal (exponent 0) has no
    // implicit leading bit, but everything below 2^-9 rounds to 0 with or without it.
    let shift = 25 - exponent;
    byte(((mantissa | 0x400) * 255 + (1 << (shift - 1))) >> shift)
}

/// How one colour component of a texel is interpolated (C.2.19).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Component {
    /// Byte endpoints widened by replication; the result is `C / 65536` rounded to a byte.
    Linear,
    /// Byte endpoints widened with 0x80; the result is the top 8 bits.
    Srgb,
    /// 12-bit endpoints; the result is the FP16 value the specification builds, as a byte.
    Hdr,
}

/// One component of a texel: `e0` and `e1` mixed by `weight` (0 to 64) as `kind` says.
fn interpolate(kind: Component, e0: u32, e1: u32, weight: u32) -> u8 {
    let mix = |c0: u32, c1: u32| (c0 * (64 - weight) + c1 * weight + 32) / 64;
    match kind {
        Component::Linear => {
            let c = mix((e0 << 8) | e0, (e1 << 8) | e1);
            // 65535 is 1.0 by the specification; the rounding below gives 255 for it too.
            byte((255 * c + 0x8000) >> 16)
        }
        Component::Srgb => byte(mix((e0 << 8) | 0x80, (e1 << 8) | 0x80) >> 8),
        Component::Hdr => {
            let c = mix(e0 << 4, e1 << 4);
            let (e, m) = ((c & 0xF800) >> 11, c & 0x7FF);
            let mt = if m < 512 {
                3 * m
            } else if m >= 1536 {
                5 * m - 2048
            } else {
                4 * m - 512
            };
            let cf = (e << 10) + (mt >> 3);
            // "If the result is +Inf or NaN, it is converted to the bit pattern 0x7BFF."
            half_to_byte(if cf >= 0x7C00 { 0x7BFF } else { cf })
        }
    }
}

/// The colour of a void-extent block (C.2.23), or `None` when its reserved bits are not set.
fn void_extent(block: u128, hdr: bool, encoding: ColorEncoding) -> Option<[u8; 4]> {
    if (block >> 10) & 0b11 != 0b11 {
        return None;
    }
    // Red, green, blue and alpha are the top four 16-bit words.
    let mut reader = BitReader::at(block, 64);
    let components = [(); 4].map(|()| reader.take(16));
    Some(match (hdr, encoding) {
        (false, _) => components.map(|c| byte(c >> 8)),
        (true, ColorEncoding::Linear) => components.map(half_to_byte),
        // "If a void-extent block with HDR values is decoded in LDR mode, then the result
        // will be the error color."
        (true, ColorEncoding::Srgb) => ERROR_COLOR,
    })
}

/// What a block with a weight grid says about itself, with every limit of C.2.24 checked,
/// before any colour or weight is decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Layout {
    /// The weight grid.
    pub(super) grid: WeightGrid,
    /// The bits the weights take, 24 to 96, at the top of the block.
    pub(super) weight_bits: u32,
    /// The number of partitions, 1 to 4.
    pub(super) partitions: usize,
    /// The partition pattern's seed; 0 with one partition.
    pub(super) partition_index: u32,
    /// Each partition's endpoint mode, 0 to 15; the entries past `partitions` mean nothing.
    pub(super) modes: [u32; MAX_PARTITIONS],
    /// The colour component that uses the second weight plane, when there are two.
    pub(super) plane_selector: u32,
    /// The first bit of the colour data: 17 with one partition, 29 with more.
    pub(super) color_start: u32,
    /// The bits of colour data.
    pub(super) color_bits: u32,
    /// The colour values the endpoint modes take together, at most 18.
    pub(super) value_count: usize,
    /// How the colour values are stored: the widest range that fits the colour bits.
    pub(super) color_encoding: Encoding,
}

/// What a legal block is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Parsed {
    /// One colour, FP16 when `hdr`.
    VoidExtent {
        /// Whether the colour is FP16.
        hdr: bool,
    },
    /// Endpoints and weights.
    Weights(Layout),
}

/// Reads a block's configuration and applies C.2.24's checks to it (the reserved bits of a
/// void-extent block are checked with its colour, in `void_extent`). `None` means the block
/// is an illegal encoding.
pub(super) fn parse(block: u128, footprint: Footprint) -> Option<Parsed> {
    let mut reader = BitReader::new(block);
    let grid = match block_mode(reader.take(MODE_BITS))? {
        BlockMode::VoidExtent { hdr } => return Some(Parsed::VoidExtent { hdr }),
        BlockMode::Weights(grid) => grid,
    };
    if grid.width > footprint.width || grid.height > footprint.height {
        return None;
    }
    if usize::try_from(grid.weight_count()).ok()? > MAX_WEIGHTS {
        return None;
    }
    let weight_bits = grid.encoding().bit_length(grid.weight_count());
    if !WEIGHT_BITS.contains(&weight_bits) {
        return None;
    }

    let partition_bits = reader.take(2);
    let partitions = usize::try_from(partition_bits).ok()? + 1;
    if partitions == MAX_PARTITIONS && grid.dual_plane {
        return None;
    }
    // One partition: a 4-bit endpoint mode. More: a 10-bit partition index and the first 6
    // bits of the endpoint modes (figures C.2 and C.3).
    let (partition_index, cem) = if partitions == 1 {
        (0, reader.take(4) << 2)
    } else {
        (reader.take(10), reader.take(6))
    };
    // Table C.2.11: selector 0 is one mode for every partition; otherwise each partition has
    // a class bit here and two more mode bits just below the weights (figure C.4).
    let selector = cem & 0b11;
    let extra_cem_bits = if selector == 0 {
        0
    } else {
        3 * partition_bits - 1
    };
    let plane_bits = if grid.dual_plane { 2 } else { 0 };
    let color_start = reader.position();
    let color_bits =
        BLOCK_BITS.checked_sub(color_start + weight_bits + extra_cem_bits + plane_bits)?;
    // Above the colour data: the plane selector, then the extra mode bits, then the weights.
    let mut above = BitReader::at(block, color_start + color_bits);
    let plane_selector = above.take(plane_bits);
    let extra_cem = above.take(extra_cem_bits);

    let mut modes = [cem >> 2; MAX_PARTITIONS];
    if selector != 0 {
        let mut rest = ((extra_cem << 6) | cem) >> 2;
        let mut classes = [0; MAX_PARTITIONS];
        for class in classes.iter_mut().take(partitions) {
            *class = rest & 1;
            rest >>= 1;
        }
        for (mode, class) in modes.iter_mut().zip(classes).take(partitions) {
            *mode = ((selector - 1 + class) << 2) | (rest & 0b11);
            rest >>= 2;
        }
    }

    let value_count: usize = modes
        .iter()
        .take(partitions)
        .map(|mode| endpoints::value_count(*mode))
        .sum();
    if value_count > MAX_COLOR_VALUES {
        return None;
    }
    // C.2.22: the widest range whose encoding of the values fits the colour bits. When not
    // even the narrowest fits, the block is illegal: 0..5 takes a trit and a bit a value,
    // `ceil(8 * values / 5) + values` bits, which is C.2.24's `ceil(13 * values / 5)`.
    let value_count_bits = u32::try_from(value_count).ok()?;
    let color_encoding = COLOR_ENCODINGS
        .into_iter()
        .find(|encoding| encoding.bit_length(value_count_bits) <= color_bits)?;
    Some(Parsed::Weights(Layout {
        grid,
        weight_bits,
        partitions,
        partition_index,
        modes,
        plane_selector,
        color_start,
        color_bits,
        value_count,
        color_encoding,
    }))
}

/// Decodes `block` into `texels`, which holds exactly the footprint's texels row by row.
/// `None` means the block is an illegal encoding, and `texels` may be partly written.
pub(super) fn decode(
    block: u128,
    footprint: Footprint,
    encoding: ColorEncoding,
    texels: &mut [[u8; 4]],
) -> Option<()> {
    let layout = match parse(block, footprint)? {
        Parsed::VoidExtent { hdr } => {
            texels.fill(void_extent(block, hdr, encoding)?);
            return Some(());
        }
        Parsed::Weights(layout) => layout,
    };
    let Layout {
        grid,
        weight_bits,
        partitions,
        partition_index,
        plane_selector,
        color_start,
        color_bits,
        value_count,
        color_encoding,
        ..
    } = layout;
    let modes = layout.modes.get(..partitions)?;
    let weight_count = usize::try_from(grid.weight_count()).ok()?;
    let weight_encoding = grid.encoding();
    let partition_count = u32::try_from(partitions).ok()?;

    let mut raw = [Value::default(); MAX_COLOR_VALUES];
    let raw = raw.get_mut(..value_count)?;
    let color_data = (block >> color_start) & mask(color_bits);
    decode_sequence(&mut BitReader::new(color_data), color_encoding, raw);
    let mut values = [0; MAX_COLOR_VALUES];
    for (value, raw) in values.iter_mut().zip(raw.iter()) {
        *value = endpoints::unquantize_color(color_encoding, *raw)?;
    }
    let mut partition_endpoints = [Endpoints::default(); MAX_PARTITIONS];
    let mut rest = values.get(..value_count)?;
    for (slot, mode) in partition_endpoints.iter_mut().zip(modes) {
        let (mine, others) = rest.split_at_checked(endpoints::value_count(*mode))?;
        *slot = endpoints::decode(*mode, mine)?;
        rest = others;
    }

    // The weights are read from the top of the block downward (C.2.16).
    let weight_data = block.reverse_bits() & mask(weight_bits);
    let mut raw = [Value::default(); MAX_WEIGHTS];
    decode_sequence(
        &mut BitReader::new(weight_data),
        weight_encoding,
        raw.get_mut(..weight_count)?,
    );
    let mut stored = [0; MAX_WEIGHTS];
    for (weight, raw) in stored.iter_mut().zip(raw).take(weight_count) {
        *weight = weights::unquantize_weight(weight_encoding, raw)?;
    }
    let texel_weights = weights::infill(&grid, &stored, footprint);

    let small_block = footprint.width * footprint.height < SMALL_BLOCK_TEXELS;
    let positions = (0..footprint.height).flat_map(|y| (0..footprint.width).map(move |x| (x, y)));
    for (index, ((x, y), texel)) in positions.zip(texels.iter_mut()).enumerate() {
        let partition =
            weights::select_partition(partition_index, x, y, partition_count, small_block);
        let ends = partition_endpoints.get(partition)?;
        if ends.hdr_rgb && encoding == ColorEncoding::Srgb {
            // The specification's LDR mode: "Only those texels which belong to the HDR
            // partition will result in the error color."
            *texel = ERROR_COLOR;
            continue;
        }
        let rgb = match (ends.hdr_rgb, encoding) {
            (true, _) => Component::Hdr,
            (false, ColorEncoding::Linear) => Component::Linear,
            (false, ColorEncoding::Srgb) => Component::Srgb,
        };
        let alpha = if ends.hdr_alpha {
            Component::Hdr
        } else {
            Component::Linear
        };
        // C.2.20: with two planes, the selected component uses the second plane's weight.
        for (channel, kind) in (0..).zip([rgb, rgb, rgb, alpha]) {
            let plane = usize::from(grid.dual_plane && plane_selector == channel);
            let weight = *texel_weights.get(plane)?.get(index)?;
            let component = channel as usize;
            texel[component] = interpolate(kind, ends.e0[component], ends.e1[component], weight);
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::{COLOR_ENCODINGS, Component, half_to_byte, interpolate};
    use crate::astc::integer::Digit;

    /// Table C.2.16's ranges and the plain ones above 0..5, widest first: 0..255, 0..191,
    /// 0..159, .., 0..5.
    #[test]
    fn colour_ranges_are_widest_first() {
        let tops: Vec<u32> = COLOR_ENCODINGS
            .iter()
            .map(|encoding| {
                let base = match encoding.digit {
                    Digit::None => 1,
                    Digit::Trit => 3,
                    Digit::Quint => 5,
                };
                (base << encoding.bits) - 1
            })
            .collect();
        assert_eq!(
            tops,
            [
                255, 191, 159, 127, 95, 79, 63, 47, 39, 31, 23, 19, 15, 11, 9, 7, 5
            ]
        );
    }

    /// FP16 patterns by hand: sign, 5 exponent bits, 10 mantissa bits.
    #[test]
    fn halves_as_bytes() {
        let cases = [
            (0x0000, 0),   // 0.0
            (0x3C00, 255), // 1.0
            (0x3800, 128), // 0.5: 127.5 + 0.5
            (0x3400, 64),  // 0.25: 63.75 + 0.5
            (0x3BFF, 255), // just under 1.0: 254.87 + 0.5
            (0x3BF8, 254), // 2040 / 2048: 254.004 + 0.5
            (0x4000, 255), // 2.0
            (0x7BFF, 255), // the largest finite value
            (0x7C00, 255), // +infinity
            (0xFC00, 0),   // -infinity
            (0x7C01, 0),   // a NaN
            (0x7E00, 0),   // a NaN
            (0xFE00, 0),   // a NaN with the sign bit
            (0x8000, 0),   // -0.0
            (0xBC00, 0),   // -1.0
            (0x8001, 0),   // a negative subnormal
            (0x0001, 0),   // the smallest subnormal, 2^-24
            (0x03FF, 0),   // the largest subnormal
            (0x0400, 0),   // the smallest normal, 2^-14
            (0x1800, 0),   // 2^-9: 0.498 + 0.5
            (0x1BFF, 1),   // just under 2^-8: 0.9956 + 0.5
            (0x1C00, 1),   // 2^-8: 0.996 + 0.5
            (0x1805, 1),   // 2^-9 * (1 + 5/1024): 0.50047.. + 0.5, the first to round up
            (0x1804, 0),   // one below it: 0.49999.. + 0.5
        ];
        for (half, expected) in cases {
            assert_eq!(half_to_byte(half), expected, "{half:#06x}");
        }
    }

    /// Every FP16 pattern against the same conversion done in `f32`, as Eden does it: the
    /// integer arithmetic is exact.
    #[test]
    fn halves_as_bytes_match_float_arithmetic() {
        for half in 0..=0xFFFF_u32 {
            let exponent = (half >> 10) & 0x1F;
            let mantissa = f32::from(u16::try_from(half & 0x3FF).expect("10 bits"));
            let magnitude = match exponent {
                0 => mantissa * 2.0_f32.powi(-24),
                31 if mantissa == 0.0 => f32::INFINITY,
                31 => f32::NAN,
                _ => {
                    (1024.0 + mantissa)
                        * 2.0_f32.powi(i32::try_from(exponent).expect("5 bits") - 25)
                }
            };
            let value = if half & 0x8000 == 0 {
                magnitude
            } else {
                -magnitude
            };
            let expected = if value.is_nan() {
                0.0
            } else {
                (value.clamp(0.0, 1.0) * 255.0 + 0.5).floor()
            };
            assert_eq!(f32::from(half_to_byte(half)), expected, "{half:#06x}");
        }
    }

    /// C.2.19 by hand, for each kind of component.
    #[test]
    fn interpolation_by_hand() {
        // Linear: endpoints widen by replication (0x12 -> 0x1212); the ends are exact.
        for value in [0, 1, 0x12, 0x80, 0xFE, 0xFF] {
            assert_eq!(
                interpolate(Component::Linear, value, 0xAB, 0),
                u8::try_from(value).expect("a byte")
            );
            assert_eq!(
                interpolate(Component::Linear, 0xAB, value, 64),
                u8::try_from(value).expect("a byte")
            );
            assert_eq!(
                interpolate(Component::Srgb, value, 0xAB, 0),
                u8::try_from(value).expect("a byte")
            );
            assert_eq!(
                interpolate(Component::Srgb, 0xAB, value, 64),
                u8::try_from(value).expect("a byte")
            );
        }
        // 0 to 255 at weight 32: C = floor((65535 * 32 + 32) / 64) = 32768, exactly a half,
        // which rounds up to 128.
        assert_eq!(interpolate(Component::Linear, 0, 255, 32), 128);
        // 2 to 255 at weight 32: linear C = 33025 -> 128; sRGB C = 33024 = 0x8100 -> 129.
        assert_eq!(interpolate(Component::Linear, 2, 255, 32), 128);
        assert_eq!(interpolate(Component::Srgb, 2, 255, 32), 129);
        // 0 to 224 at weight 43: linear C = floor((57568 * 43 + 32) / 64) = 38678 -> 150;
        // sRGB C0 = 128, C1 = 57472: C = floor((128 * 21 + 57472 * 43 + 32) / 64) = 38656
        // = 0x9700 -> 151.
        assert_eq!(interpolate(Component::Linear, 0, 224, 43), 150);
        assert_eq!(interpolate(Component::Srgb, 0, 224, 43), 151);
        // HDR: 0x780 is 1.0 and 0x700 is 0.5 at either end.
        assert_eq!(interpolate(Component::Hdr, 0x780, 0, 0), 255);
        assert_eq!(interpolate(Component::Hdr, 0, 0x700, 64), 128);
        // The three slopes of the mantissa: C = v << 4 with M below 512, from 512, from 1536.
        // v = 0x710: C = 0x7100, E = 14, M = 256, Mt = 768, Cf = 0x3860 = 0.5 * (1 + 96/1024)
        // = 0.546875 -> 139.95.. -> 139.
        assert_eq!(interpolate(Component::Hdr, 0x710, 0, 0), 139);
        // v = 0x740: C = 0x7400, M = 1024, Mt = 3584, Cf = 0x39C0 = 0.5 * (1 + 448/1024)
        // = 0.71875 -> 183.78 -> 183.
        assert_eq!(interpolate(Component::Hdr, 0x740, 0, 0), 183);
        // v = 0x770: C = 0x7700, M = 1792, Mt = 6912, Cf = 0x3B60 = 0.5 * (1 + 864/1024)
        // = 0.921875 -> 235.57 -> 235.
        assert_eq!(interpolate(Component::Hdr, 0x770, 0, 0), 235);
        // The largest 12-bit value: C = 0xFFF0, E = 31, so Cf is past 0x7C00 and becomes
        // 0x7BFF, the largest finite value.
        assert_eq!(interpolate(Component::Hdr, 0xFFF, 0, 0), 255);
        // Small values: v = 0x400: C = 0x4000, E = 8, M = 0, Cf = 0x2000 = 2^-7 -> 2.49 -> 2.
        assert_eq!(interpolate(Component::Hdr, 0x400, 0, 0), 2);
        // Values whose byte moves if a constant of the mantissa's slopes is off by one.
        // v = 0x66C: C = 0x66C0, E = 12, M = 1728, Mt = 5 * 1728 - 2048 = 6592, Cf = 0x3338
        // = 0.125 * (1 + 824/1024) = 0.2256 -> 57.52 + 0.5 -> 58.
        assert_eq!(interpolate(Component::Hdr, 0x66C, 0, 0), 58);
        assert_eq!(interpolate(Component::Hdr, 0x4F4, 0, 0), 8);
        assert_eq!(interpolate(Component::Hdr, 0x6E8, 0, 0), 113);
        // 0x400 to 0x68A at weight 43: C = floor((0x4000 * 21 + 0x68A0 * 43 + 32) / 64) =
        // 0x5B4C, E = 11, M = 844, Mt = 4 * 844 - 512 = 2864, Cf = 0x2D66 = 0.0625 * (1 +
        // 358/1024) = 0.08435 -> 21.51 + 0.5 -> 22.
        assert_eq!(interpolate(Component::Hdr, 0x400, 0x68A, 43), 22);
        assert_eq!(interpolate(Component::Hdr, 0x407, 0x75A, 43), 47);
        assert_eq!(interpolate(Component::Hdr, 0x415, 0x608, 21), 6);
        assert_eq!(interpolate(Component::Hdr, 0x41C, 0x7E9, 43), 81);
    }
}
