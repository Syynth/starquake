//! The size of a native image: what root commands 435 and 1510 answer and what 429 checks its
//! memory against (`docs/specs/graphics-server.md`, "Native memory and images"; port slice 8).
//!
//! Ported from Eden PR #4588,
//! `eden-pr4588/src/core/hle/service/am/service/native_image_layout.h` @ e5fb3b97
//! (`CalculateNativeImageRequirements`), GPL-3.0-or-later, Copyright 2026 Eden Emulator Project;
//! "PR" below. Eden-NX2's sizer, `Nx2CalculateImageLayout` in
//! `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb ("NX2"), is read for the conflicts
//! below and not ported.
//!
//! # The preference
//!
//! A native image is described by a 16-word preference (NX2: `InterModuleUserImagePreference`;
//! the request's words 2 to 17). [`requirements`] checks it as PR does, every check in PR's one
//! condition, and reports the first that fails, in this order:
//!
//! | Word | What | PR's check | Error |
//! |---|---|---|---|
//! | 0 | (unknown) | `preference[0] != 0` | [`PreferenceError::WordNotZero`] |
//! | [`WORD_KIND`] | the kind: [`KIND_NATIVE`] (435, 429) or [`KIND_BLOCK`] (1510) | `!= 1 && != 2` | [`PreferenceError::Kind`] |
//! | 3 | (unknown) | `preference[3] != 0` | [`PreferenceError::WordNotZero`] |
//! | [`WORD_FORMAT`] | the format: [`FORMAT_RGBA8`] or [`FORMAT_RGB10A2`] | `!= 1 && != 4` | [`PreferenceError::Format`] |
//! | [`WORD_ONE`] | (unknown; **guess**: a level or layer count) | `preference[7] != 1` | [`PreferenceError::WordNotOne`] |
//! | [`WORD_BLOCK_HEIGHT`] | log2 of the block height in rows of pixels | `< 3 \|\| > 8` | [`PreferenceError::BlockHeight`] |
//! | 8 to 15 | (unknown) | each `!= 0` | [`PreferenceError::WordNotZero`], the first such word |
//! | [`WORD_PITCH`] | the row pitch in 64-byte GOBs | `< (width * 4 + 63) / 64 \|\| > 4096` | [`PreferenceError::Pitch`] |
//!
//! [`WORD_EXTENT`] is `(width - 1) | (height - 1) << 16` and is not checked: every value is an
//! extent from 1 by 1 to 65,536 by 65,536. Then the count: 0 is [`PreferenceError::ZeroCount`]
//! (PR checks `count == 0` in the same first condition; reporting it after the preference's
//! checks is what lets 435 and 1510 tell a zero count with a good preference apart, as PR's
//! handler does by asking again with a count of 1). Last, an image whose size times the count
//! passes [`MAX_SIZE`] is [`PreferenceError::TooLarge`] (PR: `image_size > (1ULL << 48) /
//! count`).
//!
//! The spec reads word 4 as "the block height (log2 of GOB rows)"; both references shift 1 by
//! it to get rows of pixels (PR: "block height is in pixels", `1ULL << preference[4]`; NX2:
//! `block_rows = u64{1} << layout.block_height_log2`), which is what is done here. A GOB is 8
//! rows, so 3 to 8 is one to 32 GOBs.
//!
//! # The size
//!
//! `pitch_gobs * 64 * align(height, 1 << word 4) * count`, with an alignment of
//! [`IMAGE_ALIGNMENT`] (PR, "run against the real module": "the size is one image's padded
//! block storage times the request's count word, the format word does not scale it, and the
//! alignment is always one page"). The size itself is not rounded to the alignment.
//!
//! # Where NX2 differs (conflict)
//!
//! The spec marks the sizer a **conflict** and picks PR's (graphics-server, "Native memory and
//! images"), by decision 8's first leg: PR's sizer is a reading that cites the system module
//! (its comment names the module's handlers, `gsv1 0x4cf44` and `0x909a0`). Each difference is
//! pinned by the test named, in `layout/tests.rs`:
//!
//! | Case | NX2 | Here (PR) |
//! |---|---|---|
//! | Alignment | `max(64 * block rows, 0x1000)` | always [`IMAGE_ALIGNMENT`] (`the_alignment_is_one_page_whatever_the_block`) |
//! | A preference PR refuses | sized anyway: the pitch falls back to `align(width * 4, 64)` and the block to 128 rows; no word but 2, 4 and 5 is read | an error (`each_preference_check_refuses_on_its_own`) |
//! | Block height log2 0 to 2 | accepted | [`PreferenceError::BlockHeight`] (`block_height_edges`) |
//! | Count 0 | sized as a count of 1 (435: `std::max<u32>(w[18], 1)`) | [`PreferenceError::ZeroCount`] (`a_zero_count_is_its_own_error`) |
//! | 1510's count | not read: one image's size | multiplies the size, as 435's does (`the_count_multiplies_the_size`) |
//!
//! # Against `starquake-layout`
//!
//! The formula is kept here, not taken from the layout crate, because it reproduces the system
//! module's answer, a protocol fact (gpu-protocol, "Block-linear layout"; review guide, M8).
//! `layout/tests.rs` checks it against `starquake_layout::Image` for the same description (a 2D
//! image of 4-byte texels, `pitch_gobs * 16` texels wide, the Turing colour GOB, one GOB wide and
//! word 4 - 3 GOBs tall, `count` layers), in
//! `the_size_agrees_with_the_layout_crate`. They agree on one image's storage when the layout
//! crate keeps the requested block for the image: PR's size is the layout's layer stride times
//! the count. They differ in three listed ways, each asserted there:
//!
//! - **The block is clamped there, not here.** The layout crate shrinks the block to the
//!   smallest power of two of GOBs that covers the image (Mesa's `Tiling::clamp`, "done by the
//!   hardware at each LOD"); PR pads the height to the full block. For an image shorter than
//!   half its block, PR's size is the larger.
//! - **The size is rounded there, not here.** The layout crate rounds the whole image up to its
//!   alignment; PR does not round.
//! - **The alignment.** The layout crate's is the larger of a tile and 4096 bytes
//!   (`MIN_IMAGE_ALIGN_B`); PR's is always [`IMAGE_ALIGNMENT`].

use thiserror::Error;

#[cfg(test)]
mod tests;

/// How many words a preference holds (PR: `preference.size() != 16`).
pub const PREFERENCE_WORDS: usize = 16;

/// The preference word holding the kind (PR `preference[1]`; NX2: "kind 2 at word 3", counting
/// the request's words).
pub const WORD_KIND: usize = 1;
/// The preference word holding `(width - 1) | (height - 1) << 16` (PR and NX2 `preference[2]`;
/// spec: "word 2").
pub const WORD_EXTENT: usize = 2;
/// The preference word holding log2 of the block height in rows (PR and NX2 `preference[4]`).
pub const WORD_BLOCK_HEIGHT: usize = 4;
/// The preference word holding the row pitch in GOBs (PR and NX2 `preference[5]`).
pub const WORD_PITCH: usize = 5;
/// The preference word holding the format (PR `preference[6]`; NX2: "the pixel format at word
/// 8", counting the request's words).
pub const WORD_FORMAT: usize = 6;
/// The preference word PR requires to be 1 (`preference[7] != 1`). Its meaning is a **guess**
/// (a level or layer count); a trace of a title's preferences, or the SDK's
/// `MakeInterModuleBlockUserImagePreference`, would confirm it.
pub const WORD_ONE: usize = 7;
/// The first of the trailing words PR requires to be zero (`for (size_t i = 8; ...)`).
pub const FIRST_TAIL_WORD: usize = 8;
/// The words before [`FIRST_TAIL_WORD`] PR requires to be zero, in the order it checks them
/// (`preference[0] != 0`, then the kind, then `preference[3] != 0`).
pub const ZERO_WORDS: [usize; 2] = [0, 3];

/// The kind of a native GPU image: what 435 and 429 require (PR `CalculateImageRequirements`:
/// `packet[3] == (gpu_image ? 1U : 2U)`; `CreateNativeImage`: `packet[3] == 1`).
pub const KIND_NATIVE: u32 = 1;
/// The kind of an inter-module block image: what 1510 requires (PR, as above; NX2: "A
/// block-linear preference (`MakeInterModuleBlockUserImagePreference`) holds kind 2").
pub const KIND_BLOCK: u32 = 2;

/// RGBA8 (spec: "word 6 the format (1 RGBA8, 4 RGB10A2)"; PR `preference[6] != 1`).
pub const FORMAT_RGBA8: u32 = 1;
/// RGB10A2 (spec, as above; PR `preference[6] != 4`).
pub const FORMAT_RGB10A2: u32 = 4;

/// Bytes per pixel of both formats (PR: "restricted to the observed four-byte color formats";
/// `width * 4`).
pub const BYTES_PER_PIXEL: u64 = 4;
/// Bytes in one row of a GOB: the unit of the pitch (PR `pitch_gobs * 64`; "Row pitch is in
/// 64-byte GOBs").
pub const GOB_ROW_BYTES: u64 = 64;

/// The smallest block height, log2 of rows (PR `preference[4] < 3`).
pub const MIN_BLOCK_HEIGHT_LOG2: u32 = 3;
/// The largest block height, log2 of rows (PR `preference[4] > 8`).
pub const MAX_BLOCK_HEIGHT_LOG2: u32 = 8;
/// The widest pitch, in GOBs (PR `pitch_gobs > 4096`).
pub const MAX_PITCH_GOBS: u64 = 4096;

/// The width bits of the extent word (PR `preference[2] & 0xffffU`).
pub const EXTENT_WIDTH_MASK: u32 = 0xFFFF;
/// Where the height starts in the extent word (PR `preference[2] >> 16`).
pub const EXTENT_HEIGHT_SHIFT: u32 = 16;

/// The largest size, all images together (PR `image_size > (1ULL << 48) / count`).
pub const MAX_SIZE: u64 = 1 << 48;
/// The alignment every answer carries (PR `NativeImageRequirements{image_size * count, 4096}`).
pub const IMAGE_ALIGNMENT: u64 = 4096;

/// A native image's 16-word preference, as the request carries it.
pub type Preference = [u32; PREFERENCE_WORDS];

/// What a preference and a count need: PR's `NativeImageRequirements`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requirements {
    /// Bytes, all images together.
    pub size: u64,
    /// The alignment: always [`IMAGE_ALIGNMENT`].
    pub alignment: u64,
}

/// Why a preference and count have no size (module docs, "The preference"): each is one check
/// of PR's `CalculateNativeImageRequirements`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum PreferenceError {
    /// A word PR requires to be zero is not.
    #[error("preference word {word} is {value}, not 0")]
    WordNotZero {
        /// The word's index in the preference.
        word: usize,
        /// Its value.
        value: u32,
    },
    /// The kind is neither [`KIND_NATIVE`] nor [`KIND_BLOCK`].
    #[error("preference kind {0} is neither 1 nor 2")]
    Kind(u32),
    /// The format is neither [`FORMAT_RGBA8`] nor [`FORMAT_RGB10A2`].
    #[error("preference format {0} is neither 1 nor 4")]
    Format(u32),
    /// [`WORD_ONE`] is not 1.
    #[error("preference word 7 is {0}, not 1")]
    WordNotOne(u32),
    /// The block height is outside [`MIN_BLOCK_HEIGHT_LOG2`] to [`MAX_BLOCK_HEIGHT_LOG2`].
    #[error("block height log2 {0} is outside 3 to 8")]
    BlockHeight(u32),
    /// The pitch is narrower than a row of the image or wider than [`MAX_PITCH_GOBS`].
    #[error("a pitch of {pitch} GOBs does not fit a row of {width} pixels or passes 4096")]
    Pitch {
        /// The pitch, in GOBs.
        pitch: u32,
        /// The width, in pixels.
        width: u64,
    },
    /// The count is zero.
    #[error("the image count is 0")]
    ZeroCount,
    /// The images together pass [`MAX_SIZE`].
    #[error("{count} images of {image} bytes pass 2^48 bytes")]
    TooLarge {
        /// One image's size.
        image: u64,
        /// The count.
        count: u32,
    },
}

/// The extent word's width and height in pixels: `(word & 0xffff) + 1` and `(word >> 16) + 1`.
#[must_use]
pub fn extent(word: u32) -> (u64, u64) {
    (
        u64::from(word & EXTENT_WIDTH_MASK) + 1,
        u64::from(word >> EXTENT_HEIGHT_SHIFT) + 1,
    )
}

/// The narrowest pitch, in GOBs, that holds a row of `width` pixels (PR `(width * 4 + 63) /
/// 64`). `width` is at most 65,536, so this cannot overflow.
#[must_use]
pub const fn min_pitch_gobs(width: u64) -> u64 {
    (width * BYTES_PER_PIXEL).div_ceil(GOB_ROW_BYTES)
}

/// One image's bytes for a preference that passed the checks: `pitch * 64 * align(height,
/// block)`. At most 4096 GOBs times 64 bytes times 65,536 rows, 2^34, so this cannot overflow.
const fn image_size(pitch_gobs: u64, height: u64, block_height_log2: u32) -> u64 {
    let block = 1 << block_height_log2;
    pitch_gobs * GOB_ROW_BYTES * height.next_multiple_of(block)
}

/// What `count` images of `preference` need (PR `CalculateNativeImageRequirements`), or the
/// first check that fails, in the order of the module docs ("The preference").
pub fn requirements(preference: &Preference, count: u32) -> Result<Requirements, PreferenceError> {
    let zero = |index: usize| match preference[index] {
        0 => Ok(()),
        value => Err(PreferenceError::WordNotZero { word: index, value }),
    };
    zero(ZERO_WORDS[0])?;
    let kind = preference[WORD_KIND];
    if kind != KIND_NATIVE && kind != KIND_BLOCK {
        return Err(PreferenceError::Kind(kind));
    }
    zero(ZERO_WORDS[1])?;
    let format = preference[WORD_FORMAT];
    if format != FORMAT_RGBA8 && format != FORMAT_RGB10A2 {
        return Err(PreferenceError::Format(format));
    }
    if preference[WORD_ONE] != 1 {
        return Err(PreferenceError::WordNotOne(preference[WORD_ONE]));
    }
    let block = preference[WORD_BLOCK_HEIGHT];
    if !(MIN_BLOCK_HEIGHT_LOG2..=MAX_BLOCK_HEIGHT_LOG2).contains(&block) {
        return Err(PreferenceError::BlockHeight(block));
    }
    for index in FIRST_TAIL_WORD..PREFERENCE_WORDS {
        zero(index)?;
    }
    let (width, height) = extent(preference[WORD_EXTENT]);
    let pitch = preference[WORD_PITCH];
    let pitch_gobs = u64::from(pitch);
    if pitch_gobs < min_pitch_gobs(width) || pitch_gobs > MAX_PITCH_GOBS {
        return Err(PreferenceError::Pitch { pitch, width });
    }
    if count == 0 {
        return Err(PreferenceError::ZeroCount);
    }
    let image = image_size(pitch_gobs, height, block);
    if image > MAX_SIZE / u64::from(count) {
        return Err(PreferenceError::TooLarge { image, count });
    }
    Ok(Requirements {
        // At most MAX_SIZE: the check above.
        size: image * u64::from(count),
        alignment: IMAGE_ALIGNMENT,
    })
}
