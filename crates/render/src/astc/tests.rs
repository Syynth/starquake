//! Tests of the CPU ASTC decoder.
//!
//! - [`vectors`]: blocks laid out by hand from the ASTC specification's figures and tables,
//!   with expected texels worked out from its formulas (C.2.x of
//!   `KHR_texture_compression_astc_hdr`), not taken from this decoder.
//! - [`illegal`]: one block for each illegal encoding of C.2.24, with a legal neighbour exactly
//!   at each limit.
//! - [`oracle`]: the same blocks decoded by a second decoder (the `astc-decode` crate), and the
//!   property test that arbitrary bytes never panic.
//! - [`image`]: whole images: block order, cropping, layers and the size checks.
//!
//! The helpers here place fields at the bit positions the specification gives. Block bit 0 is
//! the lowest bit of the first byte (C.2.9), so a block is written as a `u128` and stored
//! little-endian.

use super::{BLOCK_BYTES, ERROR_COLOR, Footprint, MAX_TEXELS, decode_block};
use crate::format::{ColorEncoding, Compression, PixelFormat};

mod illegal;
mod image;
mod oracle;
mod tables;
mod vectors;

/// The format's constants, as the specification states them: a block is 128 bits (C.2.9), the
/// largest footprint is 12 by 12 (table C.2.2), and the error colour is opaque magenta,
/// "R,G,B,A = 0xFF, 0x00, 0xFF, 0xFF" (C.2.5).
#[test]
fn the_formats_constants() {
    assert_eq!(BLOCK_BYTES, 16);
    assert_eq!(MAX_TEXELS, 144);
    assert_eq!(ERROR_COLOR, [0xFF, 0x00, 0xFF, 0xFF]);
    let largest = astc_formats()
        .into_iter()
        .map(texel_count)
        .max()
        .expect("fourteen footprints");
    assert_eq!(largest, MAX_TEXELS);
}

/// What a texel slot holds before a decode: neither the error colour nor anything a test
/// expects, so an untouched slot shows.
const UNTOUCHED: [u8; 4] = [0xA5, 0x5A, 0xA5, 0x5A];

/// Every ASTC format, in `PixelFormat::ALL`'s order.
fn astc_formats() -> Vec<PixelFormat> {
    PixelFormat::ALL
        .into_iter()
        .filter(|format| format.compression() == Compression::Astc)
        .collect()
}

fn footprint(format: PixelFormat) -> Footprint {
    Footprint::of(format).expect("an ASTC format")
}

fn texel_count(format: PixelFormat) -> usize {
    let footprint = footprint(format);
    usize::try_from(footprint.width() * footprint.height()).expect("at most 144")
}

fn encoding(format: PixelFormat) -> ColorEncoding {
    if format.is_srgb() {
        ColorEncoding::Srgb
    } else {
        ColorEncoding::Linear
    }
}

/// Decodes `block` as one block of `format`: whether it is legal, and its texels row by row.
/// Also checks that the slots past the footprint were left alone.
fn decode_one(block: u128, format: PixelFormat) -> (bool, Vec<[u8; 4]>) {
    let bytes: [u8; BLOCK_BYTES] = block.to_le_bytes();
    let mut texels = [UNTOUCHED; MAX_TEXELS];
    let legal = decode_block(&bytes, footprint(format), encoding(format), &mut texels);
    let count = texel_count(format);
    assert!(
        texels[count..].iter().all(|texel| *texel == UNTOUCHED),
        "{format}: a slot past the footprint was written"
    );
    let _ = oracle::check(block, format, legal, &texels[..count]);
    (legal, texels[..count].to_vec())
}

/// The texels of a legal block.
fn legal(block: u128, format: PixelFormat) -> Vec<[u8; 4]> {
    let (legal, texels) = decode_one(block, format);
    assert!(legal, "{format}: {block:#034x} should be legal");
    texels
}

/// Asserts that `block` is illegal in `format` and decodes to the error colour throughout.
fn assert_illegal(block: u128, format: PixelFormat) {
    let (legal, texels) = decode_one(block, format);
    assert!(!legal, "{format}: {block:#034x} should be illegal");
    assert_eq!(texels, vec![ERROR_COLOR; texel_count(format)], "{format}");
}

/// An opaque grey.
const fn grey(level: u8) -> [u8; 4] {
    [level, level, level, 0xFF]
}

// Field positions, from the specification's figures.

/// Bits 12..11: the number of partitions, minus one (figure C.1).
const PARTITIONS_AT: u32 = 11;
/// Bits 16..13 of a single-partition block: the colour endpoint mode (figure C.2).
const CEM_AT: u32 = 13;
/// The first bit of a single-partition block's colour endpoint data (figure C.2).
const COLOR_AT: u32 = 17;
/// Bits 22..13 of a multi-partition block: the partition index (figure C.3).
const PARTITION_INDEX_AT: u32 = 13;
/// Bits 28..23 of a multi-partition block: the first six bits of the endpoint modes
/// (figure C.3).
const MULTI_CEM_AT: u32 = 23;
/// The first bit of a multi-partition block's colour endpoint data (figure C.3).
const MULTI_COLOR_AT: u32 = 29;

// Block modes, from table C.2.8. In its first row, bits 1..0 are R2 R1, bit 4 is R0, bits
// 6..5 are A and bits 8..7 are B, for a grid B+4 wide and A+2 high.

/// A weight grid 4 wide (B = 0) and 4 high (A = 2).
const GRID_4X4: u128 = 2 << 5;
/// Weight range 0..1 (table C.2.7: R = 010, H = 0): one bit a weight.
const WEIGHTS_0_TO_1: u128 = 0b01;
/// Weight range 0..2 (R = 011, H = 0): one trit a weight.
const WEIGHTS_0_TO_2: u128 = (1 << 4) | 0b01;
/// Weight range 0..3 (R = 100, H = 0): two bits a weight.
const WEIGHTS_0_TO_3: u128 = 0b10;
/// Weight range 0..4 (R = 101, H = 0): one quint a weight.
const WEIGHTS_0_TO_4: u128 = (1 << 4) | 0b10;
/// Weight range 0..5 (R = 110, H = 0): one trit and one bit a weight.
const WEIGHTS_0_TO_5: u128 = 0b11;
/// Weight range 0..31 (R = 111, H = 1): five bits a weight.
const WEIGHTS_0_TO_31: u128 = (1 << 9) | (1 << 4) | 0b11;
/// The D bit: two weight planes.
const DUAL_PLANE: u128 = 1 << 10;

/// The bit of the block that holds bit `n` of the weight stream: the stream starts at the
/// block's top bit and grows downward (C.2.16).
const fn stream_bit(n: u32) -> u128 {
    1 << (127 - n)
}

/// Weights of `bits` bits each, in raster order: bit `j` of weight `i` is stream bit
/// `i * bits + j`.
fn plain_weights(values: &[u32], bits: u32) -> u128 {
    let mut block = 0;
    for (index, value) in values.iter().enumerate() {
        assert!(
            *value < (1 << bits),
            "weight {value} does not fit {bits} bits"
        );
        let index = u32::try_from(index).expect("a few weights");
        for bit in 0..bits {
            if value & (1 << bit) != 0 {
                block |= stream_bit(index * bits + bit);
            }
        }
    }
    block
}

/// Colour values of 8 bits each (the range a block uses when it has the room), the first at
/// block bit `at`.
fn byte_values(values: &[u8], at: u32) -> u128 {
    let mut block = 0;
    for (index, value) in values.iter().enumerate() {
        let index = u32::try_from(index).expect("a few values");
        block |= u128::from(*value) << (at + 8 * index);
    }
    block
}

/// A single-partition block with a 4 by 4 grid of two-bit weights: `mode` is the colour
/// endpoint mode, `values` its 8-bit colour values and `weights` the sixteen weights, 0 to 3.
///
/// The weights take 32 bits, which leaves 128 - 17 - 32 = 79 bits of colour data: enough for
/// up to eight values of the widest range, 0..255, so each value is one byte (C.2.22).
fn two_bit_block(mode: u128, values: &[u8], weights: &[u32; 16]) -> u128 {
    assert!(values.len() <= 8);
    GRID_4X4
        | WEIGHTS_0_TO_3
        | (mode << CEM_AT)
        | byte_values(values, COLOR_AT)
        | plain_weights(weights, 2)
}

/// Sixteen weights alternating 0 and 3: after unquantization (C.2.17) 0 and 64, so the texels
/// alternate between the first endpoint exactly and the second exactly.
const ENDS: [u32; 16] = [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3];
