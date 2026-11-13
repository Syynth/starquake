//! The decoder compared with a second one, and fed arbitrary bytes.
//!
//! The second decoder is the `astc-decode` crate (0.3.1, Apache-2.0, by Weiyi Wang). It is a
//! separate Rust port, with its own corrections, of the `FasTC` decoder that Eden's `astc.cpp`
//! also descends from: it catches mistakes made in this port, but a mistake both inherited
//! would pass. The hand-computed blocks in `vectors` do not share that ancestry.
//!
//! [`check`] states exactly where the two may differ:
//!
//! - **HDR void-extent blocks** are always the error colour there; here they are decoded in a
//!   linear format (module docs). They are not compared.
//! - **Blocks with an HDR endpoint mode** are compared for legality only. `astc-decode` gives
//!   such a partition the error colour, and then does not step over the partition's colour
//!   values, so the partitions after it get the wrong endpoints (`compute_endpoints`' last
//!   arm, in 0.3.1). Which blocks have one is read from this decoder's own layout step, so a
//!   block wrongly thought to be HDR would go uncompared; `plenty_of_blocks_are_compared`
//!   bounds how many are, and `vectors` pins the endpoint modes by hand.
//! - **Rounding at exactly one half.** Both round `255 * C / 65536` to a byte; it rounds
//!   `C = 32768` (127.5) down to 127 and this decoder, like Eden, rounds it up to 128.
//! - **sRGB.** It has one interpolation for both encodings. In an sRGB format this decoder
//!   uses the specification's sRGB one for red, green and blue, which lands within 1.

use super::{ERROR_COLOR, astc_formats, decode_one, encoding, footprint, texel_count};
use crate::astc::block::{Parsed, parse};
use crate::astc::{BLOCK_BYTES, MAX_TEXELS, decode_block};
use crate::format::{ColorEncoding, PixelFormat};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};
use std::sync::OnceLock;

/// `block` decoded by `astc-decode`: whether it calls it legal, and the texels row by row.
fn second_opinion(block: u128, format: PixelFormat) -> (bool, Vec<[u8; 4]>) {
    let footprint = footprint(format);
    let width = footprint.width();
    let mut texels = vec![[0; 4]; texel_count(format)];
    let legal = astc_decode::astc_decode_block(
        &block.to_le_bytes(),
        astc_decode::Footprint::new(width, footprint.height()),
        |x, y, color| {
            let index = usize::try_from(y * width + x).expect("at most 144");
            texels[index] = color;
        },
    );
    (legal, texels)
}

/// Whether `block` is an HDR void-extent block: bits 8..0 are 111111100 and bit 9 is set
/// (C.2.23).
const fn is_hdr_void_extent(block: u128) -> bool {
    block & 0x3FF == 0x3FC
}

/// Equal, or the one value the two decoders round differently (module docs).
const fn same_byte(ours: u8, theirs: u8) -> bool {
    ours == theirs || (ours == 128 && theirs == 127)
}

/// The HDR endpoint modes (table C.2.10).
const HDR_MODES: [u32; 6] = [2, 3, 7, 11, 14, 15];

/// Whether a partition of `block` has an HDR endpoint mode, by this decoder's layout step.
fn has_hdr_partition(block: u128, format: PixelFormat) -> bool {
    match parse(block, footprint(format)) {
        Some(Parsed::Weights(layout)) => layout.modes[..layout.partitions]
            .iter()
            .any(|mode| HDR_MODES.contains(mode)),
        Some(Parsed::VoidExtent { .. }) | None => false,
    }
}

/// How far a comparison went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Compared {
    /// Not at all: an HDR void-extent block.
    Nothing,
    /// Whether the block is legal, which both decoders said it is not.
    Illegal,
    /// Whether the block is legal: it is, and has an HDR partition.
    LegalityOnly,
    /// Every texel.
    Texels,
}

/// Compares this decoder's result for `block` with the second decoder's, under the rules in
/// the module docs. Called by `decode_one`, so every block any test decodes goes through it.
pub(super) fn check(block: u128, format: PixelFormat, legal: bool, texels: &[[u8; 4]]) -> Compared {
    if is_hdr_void_extent(block) {
        return Compared::Nothing;
    }
    let (their_legal, theirs) = second_opinion(block, format);
    assert_eq!(
        legal, their_legal,
        "{format}: {block:#034x} is legal to one decoder only"
    );
    if !legal {
        assert!(
            texels.iter().all(|texel| *texel == ERROR_COLOR),
            "{format}: {block:#034x} is illegal but not all the error colour"
        );
        // The second decoder's error colour is the specification's too.
        assert_eq!(texels, theirs, "{format}: {block:#034x}");
        return Compared::Illegal;
    }
    if has_hdr_partition(block, format) {
        return Compared::LegalityOnly;
    }
    for (index, (ours, theirs)) in texels.iter().zip(&theirs).enumerate() {
        let agree = match encoding(format) {
            ColorEncoding::Linear => (0..4).all(|c| same_byte(ours[c], theirs[c])),
            ColorEncoding::Srgb => {
                (0..3).all(|c| ours[c].abs_diff(theirs[c]) <= 1) && same_byte(ours[3], theirs[3])
            }
        };
        assert!(
            agree,
            "{format}: {block:#034x} texel {index}: {ours:?} here, {theirs:?} there"
        );
    }
    Compared::Texels
}

/// Every block mode (all 2048 values of bits 10..0) with every partition count, on the
/// smallest, a middle and the largest footprint, with the rest of the block clear and with it
/// set to a fixed pattern.
#[test]
fn every_block_mode_agrees() {
    let pattern: u128 = 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C834;
    let mut legal_blocks = 0_u32;
    for format in [
        PixelFormat::Astc2d4x4Unorm,
        PixelFormat::Astc2d8x5Srgb,
        PixelFormat::Astc2d12x12Unorm,
    ] {
        for mode in 0..0x800_u128 {
            for partitions in 0..4_u128 {
                for rest in [0, pattern] {
                    let block = mode | (partitions << 11) | (rest & !0x1FFF);
                    let (legal, _) = decode_one(block, format);
                    legal_blocks += u32::from(legal);
                }
            }
        }
    }
    // The sweep is not all one kind: thousands of its blocks are legal and thousands are not.
    assert!(legal_blocks > 5_000, "{legal_blocks}");
    assert!(legal_blocks < 40_000, "{legal_blocks}");
}

/// The block modes the second decoder accepts on `format` with one partition of mode 0 and
/// nothing else set: used to steer generated blocks toward legal ones.
fn accepted_modes(format: PixelFormat) -> &'static [u128] {
    static MODES: OnceLock<Vec<(PixelFormat, Vec<u128>)>> = OnceLock::new();
    let all = MODES.get_or_init(|| {
        astc_formats()
            .into_iter()
            .map(|format| {
                let modes = (0..0x800_u128)
                    .filter(|mode| second_opinion(*mode, format).0)
                    .collect();
                (format, modes)
            })
            .collect()
    });
    all.iter()
        .find(|(listed, _)| *listed == format)
        .map(|(_, modes)| modes.as_slice())
        .expect("an ASTC format")
}

proptest! {
    #![proptest_config(Config {
        cases: 384,
        rng_seed: RngSeed::Fixed(0xA57C_0155),
        failure_persistence: None,
        ..Config::default()
    })]

    /// Any 16 bytes decode without panicking, on every footprint and in both encodings; an
    /// illegal block is the error colour throughout, and nothing is written past the
    /// footprint.
    #[test]
    fn any_block_decodes_without_panicking(bytes in any::<[u8; BLOCK_BYTES]>()) {
        for format in astc_formats() {
            let before = [[0x5A; 4]; MAX_TEXELS];
            let mut texels = before;
            let legal = decode_block(&bytes, footprint(format), encoding(format), &mut texels);
            let count = texel_count(format);
            prop_assert_eq!(&texels[count..], &before[count..]);
            if !legal {
                prop_assert!(texels[..count].iter().all(|texel| *texel == ERROR_COLOR));
            }
        }
    }

    /// Arbitrary blocks agree with the second decoder on every format.
    #[test]
    fn arbitrary_blocks_agree(block in any::<u128>()) {
        for format in astc_formats() {
            decode_one(block, format);
        }
    }

    /// Blocks steered toward legal ones: a block mode the second decoder accepts on the
    /// format, and arbitrary bits everywhere else.
    #[test]
    fn mostly_legal_blocks_agree(rest in any::<u128>(), pick in any::<prop::sample::Index>()) {
        for format in astc_formats() {
            let modes = accepted_modes(format);
            let block = (rest & !0x7FF) | modes[pick.index(modes.len())];
            decode_one(block, format);
        }
    }
}

/// The comparisons are not hollow: of blocks steered toward legal ones, most are legal and
/// most of those are compared texel by texel.
#[test]
fn plenty_of_blocks_are_compared() {
    let format = PixelFormat::Astc2d8x8Unorm;
    let modes = accepted_modes(format);
    assert!(modes.len() > 100, "{}", modes.len());
    let (mut legal, mut texels) = (0_u32, 0_u32);
    let mut rest: u128 = 0x0123_4567_89AB_CDEF_0F1E_2D3C_4B5A_6978;
    for mode in modes {
        // A fixed xorshift step, so the blocks differ and the test is the same every run.
        rest ^= rest << 13;
        rest ^= rest >> 7;
        rest ^= rest << 17;
        let block = (rest & !0x7FF) | mode;
        let (is_legal, decoded) = decode_one(block, format);
        legal += u32::from(is_legal);
        texels += u32::from(check(block, format, is_legal, &decoded) == Compared::Texels);
    }
    let total = u32::try_from(modes.len()).expect("at most 2048");
    assert!(legal * 2 > total, "{legal} legal of {total}");
    assert!(texels * 2 > legal, "{texels} compared of {legal} legal");
}
