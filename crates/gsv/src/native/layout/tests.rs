//! The native image sizer, against hand-computed sizes at its limits, PR's checks one by one, and
//! the layout crate.
//!
//! Preferences are laid out by hand from PR's word positions
//! (`eden-pr4588/src/core/hle/service/am/service/native_image_layout.h` @ e5fb3b97: words 0 and
//! 3 zero, the kind at 1, the extent at 2, the block height at 4, the pitch at 5, the format at
//! 6, 1 at 7, zeros from 8), never by a builder of the slice under test.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_layout::{
    Extent4D, Format, GOB_HEIGHT, GOB_WIDTH_B, GobType, Image, ImageDesc, ImageDim,
    MIN_IMAGE_ALIGN_B, SampleLayout, Tiling,
};

use super::*;
use crate::native::tests::laid_out;

/// A 1280 by 720 RGBA8 native image: 80 GOBs a row (1280 * 4 / 64), 128-row blocks.
fn hd() -> Preference {
    laid_out(1, 1280, 720, 7, 80, 1)
}

/// PR's constants, written out as the reference writes them.
#[test]
fn the_constants_are_pr_s() {
    assert_eq!(PREFERENCE_WORDS, 16);
    assert_eq!(
        (WORD_KIND, WORD_EXTENT, WORD_BLOCK_HEIGHT, WORD_PITCH),
        (1, 2, 4, 5)
    );
    assert_eq!((WORD_FORMAT, WORD_ONE, FIRST_TAIL_WORD), (6, 7, 8));
    assert_eq!(ZERO_WORDS, [0, 3]);
    assert_eq!((KIND_NATIVE, KIND_BLOCK), (1, 2));
    assert_eq!((FORMAT_RGBA8, FORMAT_RGB10A2), (1, 4));
    assert_eq!((BYTES_PER_PIXEL, GOB_ROW_BYTES), (4, 64));
    assert_eq!((MIN_BLOCK_HEIGHT_LOG2, MAX_BLOCK_HEIGHT_LOG2), (3, 8));
    assert_eq!(MAX_PITCH_GOBS, 4096);
    assert_eq!((EXTENT_WIDTH_MASK, EXTENT_HEIGHT_SHIFT), (0xffff, 16));
    assert_eq!(MAX_SIZE, 0x1_0000_0000_0000);
    assert_eq!(IMAGE_ALIGNMENT, 0x1000);
    // A pitch GOB is the layout crate's GOB: 64 bytes wide, 8 rows tall (so block log2 3 is one
    // GOB).
    assert_eq!(GOB_ROW_BYTES, u64::from(GOB_WIDTH_B));
    assert_eq!(1 << MIN_BLOCK_HEIGHT_LOG2, GOB_HEIGHT);
}

/// Sizes worked out by hand: one image's padded block storage times the count, alignment one
/// page, the format not scaling it.
#[test]
fn hand_computed_sizes() {
    // 80 GOBs * 64 bytes * 768 rows (720 up to 128) = 3,932,160 bytes.
    assert_eq!(
        requirements(&hd(), 1),
        Ok(Requirements {
            size: 3_932_160,
            alignment: 4096
        })
    );
    // RGB10A2 is the same size.
    assert_eq!(
        requirements(&laid_out(1, 1280, 720, 7, 80, 4), 1).map(|r| r.size),
        Ok(3_932_160)
    );
    // The smallest: one pixel, one GOB of pitch, an 8-row block: 64 * 8 = 512 bytes, not
    // rounded to the page.
    assert_eq!(
        requirements(&laid_out(2, 1, 1, 3, 1, 1), 1).map(|r| r.size),
        Ok(512)
    );
    // The largest image: 65,536 by 65,536 at 4096 GOBs, 256-row blocks: 2^34 bytes.
    assert_eq!(
        requirements(&laid_out(1, 65_536, 65_536, 8, 4096, 1), 1).map(|r| r.size),
        Ok(17_179_869_184)
    );
    // A height one row past a block pads to the next: 129 rows in 128-row blocks is 256 rows,
    // 2 GOBs * 64 * 256 = 32,768; 128 rows is 16,384.
    assert_eq!(
        requirements(&laid_out(1, 32, 129, 7, 2, 1), 1).map(|r| r.size),
        Ok(32_768)
    );
    assert_eq!(
        requirements(&laid_out(1, 32, 128, 7, 2, 1), 1).map(|r| r.size),
        Ok(16_384)
    );
    // A pitch wider than a row counts: 3 GOBs for a 32-pixel row, 8 rows: 1536.
    assert_eq!(
        requirements(&laid_out(1, 32, 8, 3, 3, 1), 1).map(|r| r.size),
        Ok(1536)
    );
}

/// The extent word's halves are the width and height less one, each read from its own half.
#[test]
fn the_extent_word_is_read_half_by_half() {
    assert_eq!(extent(0x0031_0063), (100, 50));
    assert_eq!(extent(0), (1, 1));
    assert_eq!(extent(u32::MAX), (65_536, 65_536));
    // Differs from the 1280-wide image only above the width's half: the height changes, the
    // width does not.
    assert_eq!(extent(0x0001_04FF), (1280, 2));
}

/// The count multiplies the size, for 435 and 1510 alike (NX2's 1510 does not read it).
#[test]
fn the_count_multiplies_the_size() {
    for kind in [KIND_NATIVE, KIND_BLOCK] {
        let mut preference = hd();
        preference[1] = kind;
        assert_eq!(
            requirements(&preference, 3).map(|r| r.size),
            Ok(3 * 3_932_160)
        );
    }
}

/// A count of 0 is its own error, reported only for a preference that passes every check
/// (NX2 sizes it as a count of 1).
#[test]
fn a_zero_count_is_its_own_error() {
    assert_eq!(requirements(&hd(), 0), Err(PreferenceError::ZeroCount));
    let mut bad = hd();
    bad[6] = 2;
    assert_eq!(requirements(&bad, 0), Err(PreferenceError::Format(2)));
}

/// The alignment is one page whatever the block (NX2: `max(64 * block rows, 0x1000)`, 16 KiB
/// for 256-row blocks).
#[test]
fn the_alignment_is_one_page_whatever_the_block() {
    for block in 3..=8 {
        let preference = laid_out(1, 64, 300, block, 4, 1);
        assert_eq!(
            requirements(&preference, 1).map(|r| r.alignment),
            Ok(4096),
            "{block}"
        );
    }
}

/// Each of PR's checks refuses a preference that is good but for it, with its own error; values
/// that differ from a good one only above the low byte are refused too.
#[test]
fn each_preference_check_refuses_on_its_own() {
    let cases: [(usize, u32, PreferenceError); 13] = [
        (0, 1, PreferenceError::WordNotZero { word: 0, value: 1 }),
        (1, 0, PreferenceError::Kind(0)),
        (1, 3, PreferenceError::Kind(3)),
        (1, 0x101, PreferenceError::Kind(0x101)),
        (
            3,
            0x100,
            PreferenceError::WordNotZero {
                word: 3,
                value: 0x100,
            },
        ),
        (6, 2, PreferenceError::Format(2)),
        (6, 0x104, PreferenceError::Format(0x104)),
        (7, 0, PreferenceError::WordNotOne(0)),
        (7, 0x101, PreferenceError::WordNotOne(0x101)),
        (4, 0x107, PreferenceError::BlockHeight(0x107)),
        (8, 1, PreferenceError::WordNotZero { word: 8, value: 1 }),
        (
            15,
            0x100,
            PreferenceError::WordNotZero {
                word: 15,
                value: 0x100,
            },
        ),
        (
            5,
            79,
            PreferenceError::Pitch {
                pitch: 79,
                width: 1280,
            },
        ),
    ];
    for (word, value, error) in cases {
        let mut preference = hd();
        preference[word] = value;
        assert_eq!(
            requirements(&preference, 1),
            Err(error),
            "word {word} = {value:#x}"
        );
    }
    // Every tail word is checked, not only the first and last.
    for word in 8..16 {
        let mut preference = hd();
        preference[word] = 1;
        assert_eq!(
            requirements(&preference, 1),
            Err(PreferenceError::WordNotZero { word, value: 1 })
        );
    }
}

/// With two faults, the one PR's condition lists first is reported, for every pair of the
/// checks (the count's and the size's last).
#[test]
fn checks_run_in_pr_s_order() {
    // One fault per check, in PR's order.
    let faults: [(usize, u32); 9] = [
        (0, 1),
        (1, 9),
        (3, 1),
        (6, 9),
        (7, 9),
        (4, 9),
        (8, 1),
        (15, 1),
        (5, 9999),
    ];
    for (first, &(word_a, value_a)) in faults.iter().enumerate() {
        let mut alone = hd();
        alone[word_a] = value_a;
        let expected = requirements(&alone, 0).unwrap_err();
        for &(word_b, value_b) in &faults[first + 1..] {
            let mut both = alone;
            both[word_b] = value_b;
            // A count of 0 as well: every preference fault comes before it.
            assert_eq!(
                requirements(&both, 0),
                Err(expected),
                "{word_a} then {word_b}"
            );
        }
    }
}

/// The block height's edges, each paired so the accepted side reaches the next check (a zero
/// count): 2 and 9 are refused, 3 and 8 accepted (NX2 accepts 0 to 2).
#[test]
fn block_height_edges() {
    for (block, outcome) in [
        (2, PreferenceError::BlockHeight(2)),
        (3, PreferenceError::ZeroCount),
        (8, PreferenceError::ZeroCount),
        (9, PreferenceError::BlockHeight(9)),
        (0, PreferenceError::BlockHeight(0)),
    ] {
        let mut preference = hd();
        preference[4] = block;
        assert_eq!(requirements(&preference, 0), Err(outcome), "{block}");
    }
}

/// The pitch's edges: a row of 17 pixels (68 bytes) needs 2 GOBs; 4096 is the widest, and a
/// 65,536-pixel row needs exactly that.
#[test]
fn pitch_edges() {
    let zero = PreferenceError::ZeroCount;
    let cases = [
        (
            17,
            1,
            PreferenceError::Pitch {
                pitch: 1,
                width: 17,
            },
        ),
        (17, 2, zero),
        (16, 1, zero),
        (17, 4096, zero),
        (
            17,
            4097,
            PreferenceError::Pitch {
                pitch: 4097,
                width: 17,
            },
        ),
        (
            65_536,
            4095,
            PreferenceError::Pitch {
                pitch: 4095,
                width: 65_536,
            },
        ),
        (65_536, 4096, zero),
    ];
    for (width, pitch, outcome) in cases {
        let preference = laid_out(1, width, 8, 3, pitch, 1);
        assert_eq!(
            requirements(&preference, 0),
            Err(outcome),
            "{width} {pitch}"
        );
    }
    assert_eq!(min_pitch_gobs(16), 1);
    assert_eq!(min_pitch_gobs(17), 2);
    assert_eq!(min_pitch_gobs(65_536), 4096);
}

/// The size limit: images times count up to exactly 2^48 are accepted, one more image is not
/// (PR: `image_size > (1ULL << 48) / count`).
#[test]
fn the_size_limit_is_exact() {
    // The largest image is 2^34 bytes: 2^14 of them are exactly 2^48.
    let largest = laid_out(1, 65_536, 65_536, 8, 4096, 1);
    let image = 17_179_869_184;
    assert_eq!(
        requirements(&largest, 16_384).map(|r| r.size),
        Ok(0x1_0000_0000_0000)
    );
    assert_eq!(
        requirements(&largest, 16_385),
        Err(PreferenceError::TooLarge {
            image,
            count: 16_385
        })
    );
    // An image that does not divide 2^48: 71,582,788 of 3,932,160 bytes fit, one more does not.
    assert_eq!(
        requirements(&hd(), 71_582_788).map(|r| r.size),
        Ok(281_474_975_662_080)
    );
    assert_eq!(
        requirements(&hd(), 71_582_789),
        Err(PreferenceError::TooLarge {
            image: 3_932_160,
            count: 71_582_789
        })
    );
    // The largest count: small images still fit.
    assert_eq!(
        requirements(&laid_out(1, 1, 1, 3, 1, 1), u32::MAX).map(|r| r.size),
        Ok(512 * u64::from(u32::MAX))
    );
}

/// The layout crate's description of the same image: 4-byte texels, `pitch * 16` texels wide,
/// the Turing colour GOB one GOB wide and `block - 3` GOBs tall, `count` layers.
fn layout_image(width_px: u32, height: u32, block: u32, count: u32) -> Image {
    let desc = ImageDesc {
        dim: ImageDim::D2,
        format: Format::uncompressed(4).unwrap(),
        extent_px: Extent4D::new(width_px, height, 1, count),
        levels: 1,
        sample_layout: SampleLayout::S1x1,
        tiling: Tiling::new(
            GobType::TuringColor2d,
            0,
            u8::try_from(block - 3).unwrap(),
            0,
        )
        .unwrap(),
        pitch_b: None,
    };
    Image::new(&desc).unwrap()
}

/// The formula against `starquake_layout` for the same description (module docs, "Against
/// `starquake-layout`"): PR's size is the layout's layer stride times the count where the
/// layout keeps the requested block, and larger where it clamps it; the layout rounds the whole
/// up to its alignment, the larger of a tile and a page, where PR's is always a page.
#[test]
fn the_size_agrees_with_the_layout_crate() {
    let mut agreed = 0;
    let mut clamped = 0;
    for block in 3..=8 {
        for height in [1, 7, 8, 9, 31, 64, 65, 129, 255, 256, 257, 720, 1080] {
            for (width, pitch) in [(1, 1), (16, 1), (17, 2), (100, 7), (1280, 80), (1920, 120)] {
                for count in [1, 2, 3] {
                    let preference = laid_out(1, width, height, block, pitch, 1);
                    let ours = requirements(&preference, count).unwrap();
                    // The same storage: a row of `pitch` GOBs holds `pitch * 16` texels.
                    let image = layout_image(pitch * 16, height, block, count);
                    let stride = image.array_stride_b() * u64::from(count);
                    let kept =
                        image.levels()[0].tiling.y_log2() == u8::try_from(block - 3).unwrap();
                    if kept {
                        assert_eq!(ours.size, stride, "{width}x{height} block {block} x{count}");
                        agreed += 1;
                    } else {
                        assert!(
                            ours.size > stride,
                            "{width}x{height} block {block} x{count}"
                        );
                        clamped += 1;
                    }
                    let tile = image.levels()[0].tiling.size_b();
                    assert_eq!(image.align_b(), tile.max(MIN_IMAGE_ALIGN_B));
                    assert_eq!(
                        image.size_b(),
                        stride.next_multiple_of(u64::from(image.align_b()))
                    );
                    assert_eq!(ours.alignment, 4096);
                }
            }
        }
    }
    // Both branches ran.
    assert!(agreed > 0 && clamped > 0, "{agreed} {clamped}");
}

/// For a row at its narrowest pitch, the layout crate's own width (the image's) gives the same
/// storage as `pitch * 16` texels: PR's narrowest pitch is the layout's padded row.
#[test]
fn the_narrowest_pitch_is_the_layout_s_padded_row() {
    for width in [1, 15, 16, 17, 100, 1280, 1921, 65_536] {
        let pitch = u32::try_from(min_pitch_gobs(u64::from(width))).unwrap();
        let own = layout_image(width, 256, 8, 1);
        let padded = layout_image(pitch * 16, 256, 8, 1);
        assert_eq!(own.array_stride_b(), padded.array_stride_b(), "{width}");
        assert_eq!(own.levels()[0].row_stride_b, pitch * 64, "{width}");
    }
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 256,
        failure_persistence: None,
        rng_seed: RngSeed::Fixed(0x4E41_5449_5645_4C59),
        rng_algorithm: RngAlgorithm::ChaCha,
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(config())]

    /// No preference and count panics, and every accepted one is within the limit and a
    /// whole number of 512-byte GOBs.
    #[test]
    fn no_preference_panics(preference in any::<[u32; 16]>(), count in any::<u32>()) {
        if let Ok(made) = requirements(&preference, count) {
            prop_assert!(made.size <= MAX_SIZE);
            prop_assert!(made.size > 0);
            prop_assert_eq!(made.size % 512, 0);
        }
    }

    /// Accepted preferences from their fields: the size is the formula over the fields read
    /// from their own words, so a swap of two reads changes it.
    #[test]
    fn the_size_is_the_formula_of_each_field(
        width in 1..=65_536_u32,
        height in 1..=65_536_u32,
        block in 3..=8_u32,
        extra in 0..=64_u32,
        count in 1..=8_u32,
        kind in 1..=2_u32,
        rgba in any::<bool>(),
    ) {
        let narrowest = width.div_ceil(16);
        let pitch = (narrowest + extra).min(4096);
        let preference = laid_out(kind, width, height, block, pitch, if rgba { 1 } else { 4 });
        let rows = u64::from(height).next_multiple_of(1 << block);
        let expected = u64::from(pitch) * 64 * rows * u64::from(count);
        prop_assert_eq!(requirements(&preference, count).map(|r| r.size), Ok(expected));
    }
}
