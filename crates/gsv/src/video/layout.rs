//! The layout of a video image pool: how big a pool of decoder output images is, where each
//! image and plane lies in the pool's memory, and the rows' pitch (`docs/specs/graphics-server.md`,
//! "Video codec sessions": "991 sizes a pool (from a decode preference: an output layout code at
//! word 1, the size at word 2; codes 1 and 4 are NV12)"; port slice 15). Root 991 answers it,
//! 911 makes a pool with it, 961 and 428 describe its images by it.
//!
//! Ported from (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project):
//!
//! - **PR:** Eden PR #4588, `eden-pr4588/src/core/hle/service/am/service/native_video_layout.h`
//!   @ e5fb3b97 (`CalculateNativeVideoImageLayout`). Its kind-1 comment reads the layout from
//!   the system module: "gsv size calculation for kind 1 (run in emulation over a size grid)".
//! - **NX2:** Eden-NX2, `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb
//!   (`Nx2CalculateDecoderImageLayout`, `CalculateDecoderImagePoolBuffer`), whose comment names
//!   the codes: "1/4 NV12 (format 3, what the movie players use), 2/5 BGRA, 3/6 RGBA; the HDR
//!   codes (10-13) are written as RGBA". Row 96 of `eden-nx2/NX2_NOTES.md` (observed): an
//!   auto-stubbed 991 echoed the request and a title aborted; NX2's answer for 640x360 NV12 is
//!   0x64000.
//!
//! # The preference
//!
//! Sixteen words, as 429's ([`Preference`]): word 0 is 0, word 1 ([`WORD_CODE`]) the output
//! layout code, word 2 ([`WORD_EXTENT`]) `(width - 1) | (height - 1) << 16`, words 3 to 15 zero.
//! [`layout`] checks, in PR's order, and answers the first failure: word 0 not zero
//! ([`LayoutError::WordNotZero`]); a code not in [`CODES`] ([`LayoutError::Code`]); a count of 0
//! ([`LayoutError::ZeroCount`]); a word from 3 up not zero, lowest first
//! ([`LayoutError::WordNotZero`]); then the code's own checks below. PR applies these checks to
//! its two codes; they are applied to every code here (NX2 checks nothing: divergence below).
//!
//! # The layouts, by code
//!
//! | Code | Planes | Layout | Source |
//! |---|---|---|---|
//! | [`CODE_NV12`] (1) | NV12: luma, then interleaved chroma | pitch-linear: pitch `align(width, 256)`; chroma at `pitch * align(height, 4)`; image `max(4096, chroma offset + pitch * align(height / 2, 4))`; images back to back; width and height even and at least 2 | PR (the module's reading) |
//! | [`CODE_RGBA_BLOCK`] (5) | RGBA, 4 bytes a pixel | block-linear: row bytes `align(width, 16) * 4`; image `max(4096, row bytes * align(height, 128))`; images `align(image, 8192)` apart | PR (`PumpOutput` swizzles it) |
//! | [`CODE_NV12_NX2`] (4) | NV12 | pitch-linear: pitch `align(width, 64)`; chroma at `pitch * align(height, 128)`; image chroma offset + `pitch * align((height + 1) / 2, 128)`; images back to back | NX2 |
//! | [`CODE_BGRA_NX2`] (2) | BGRA, 4 bytes a pixel | pitch-linear: pitch `align(width * 4, 64)`; image `pitch * align(height, 128)`; images back to back | NX2 |
//! | [`CODES_RGBA_NX2`] (3, 6, 10 to 13) | RGBA, 4 bytes a pixel | as code 2 | NX2 |
//!
//! The pool's size is `stride * (count - 1) + image` (for back-to-back images, `image * count`).
//! Its alignment is [`PR_ALIGNMENT`] for PR's codes and [`NX2_ALIGNMENT`] for NX2's. One image
//! must fit in a `u32` ([`LayoutError::TooLarge`]): PR's bound (`image_bytes >
//! numeric_limits<u32>::max()` for code 1, the stride for code 5), applied to NX2's codes too,
//! because 961 and 428 describe an image with 32-bit offsets and sizes. The pool's size then fits
//! in a `u64` for any count: `(2^32 - 1) * (2^32 - 1) + 2^32 - 1` is under 2^64.
//!
//! # Where the references disagree (decision 8)
//!
//! Each case is a **conflict**, verification 1 of the spec's "What to verify before building",
//! pinned by the test named, in `video/layout/tests.rs`:
//!
//! | Case | PR | NX2 | Here |
//! |---|---|---|---|
//! | Code 1's layout | the module's sizes (pitch to 256 bytes, rows to 4) | pitch to 64 bytes, rows to 128 (640x360: 0x64000, row 96) | PR's, by the house rule's first clause (a reading of the system module for a layout): 640x360 is 0x65400 (`code_1_is_pr_s_layout_not_nx2_s`) |
//! | Code 5 | block-linear RGBA | pitch-linear BGRA | PR's, as code 1 (`code_5_is_pr_s_block_linear_rgba`) |
//! | Codes 2, 3, 4, 6, 10 to 13 | refused (`NotImplemented`): PR implements only 1 and 5 | NX2's layouts | NX2's: PR's refusal is not a reading of the module, and an answer contains a refusal (the house rule's second clause) (`nx2_s_codes_have_nx2_s_layouts`) |
//! | Any other code | refused | RGBA (NX2's `default:` case) | refused, `NotImplemented`: only the codes NX2's comment names are taken, so that an unknown code is seen in the log rather than sized as a guess (`an_unnamed_code_is_refused`) |
//! | Word 0, words 3 to 15, a count of 0 | refused | not looked at; a count of 0 is taken as 1 | PR's checks for every code (`the_request_checks_apply_to_every_code`) |
//! | Code 4 with an odd or one-pixel extent | not taken | taken | taken: PR's evenness check is part of its code-1 layout, not NX2's (`code_4_takes_an_odd_extent`) |
//! | An image over 4 GiB | refused for PR's codes | sized in `u64`, then cut to 32 bits by 961 | refused for every code (`an_image_over_four_gib_is_refused`) |
//! | The alignment | 4096 | 8192 (`64 * 128`) | each code's own reference's (`each_code_s_alignment_is_its_reference_s`) |

use thiserror::Error;

pub use crate::native::layout::{PREFERENCE_WORDS, Preference, WORD_EXTENT, extent};

#[cfg(test)]
mod tests;

/// The preference word holding the output layout code (PR `preference[1]`; NX2 "an output
/// layout code at word 1").
pub const WORD_CODE: usize = 1;
/// The first preference word PR checks is zero (`preference[0] != 0`).
pub const WORD_ZERO: usize = 0;
/// The first of the trailing preference words PR checks are zero (`preference.begin() + 3`).
pub const FIRST_TAIL_WORD: usize = 3;

/// NV12, PR's layout (PR `preference[1] == 1`, "Kind 1: two-plane 8-bit 4:2:0 (NV12)
/// pitch-linear surface").
pub const CODE_NV12: u32 = 1;
/// RGBA, block-linear, PR's layout (PR `preference[1] == 5`; its `PumpOutput` swizzles RGBA
/// into it with `SwizzleTexture`).
pub const CODE_RGBA_BLOCK: u32 = 5;
/// NV12, NX2's layout (NX2 `case 4:` with `case 1:`, `Nx2PictureFormat::Nv12`).
pub const CODE_NV12_NX2: u32 = 4;
/// BGRA, NX2's layout (NX2 `case 2:`, `Nx2PictureFormat::Bgra8`).
pub const CODE_BGRA_NX2: u32 = 2;
/// RGBA, NX2's layout: the codes NX2's comment names that fall to its `default:` case, "3/6
/// RGBA; the HDR codes (10-13) are written as RGBA".
pub const CODES_RGBA_NX2: [u32; 6] = [3, 6, 10, 11, 12, 13];
/// Every code [`layout`] takes.
pub const CODES: [u32; 10] = [
    CODE_NV12,
    CODE_BGRA_NX2,
    3,
    CODE_NV12_NX2,
    CODE_RGBA_BLOCK,
    6,
    10,
    11,
    12,
    13,
];

/// PR's pitch alignment for code 1 (`(width + 255) & ~u64{255}`).
pub const PR_NV12_PITCH_ALIGN: u64 = 256;
/// PR's row alignment of each plane for code 1 (`(height + 3) & ~u64{3}`).
pub const PR_NV12_ROW_ALIGN: u64 = 4;
/// PR's smallest image (`(std::max)(u64{4096}, ...)`).
pub const PR_MIN_IMAGE: u64 = 4096;
/// PR's width alignment for code 5, in pixels (`((width + 15) & ~u64{15}) * 4`).
pub const PR_RGBA_WIDTH_ALIGN: u64 = 16;
/// PR's row alignment for code 5 (`(height + 127) & ~u64{127}`).
pub const PR_RGBA_ROW_ALIGN: u64 = 128;
/// PR's image stride alignment for code 5 (`(image_bytes + 8191) & ~u64{8191}`).
pub const PR_RGBA_STRIDE_ALIGN: u64 = 8192;
/// PR's pool alignment for both codes (`NativeVideoImageLayout{..., 4096, ...}`).
pub const PR_ALIGNMENT: u64 = 4096;
/// NX2's pitch alignment (`Common::AlignUp(u64{layout.width} * (nv12 ? 1 : 4), 64)`).
pub const NX2_PITCH_ALIGN: u64 = 64;
/// NX2's row alignment of each plane (`constexpr u64 BlockRows = 128`).
pub const NX2_ROW_ALIGN: u64 = 128;
/// NX2's pool alignment (`CalculateDecoderImagePoolBuffer`: `const u64 alignment = 64 * 128`).
pub const NX2_ALIGNMENT: u64 = 64 * 128;
/// Bytes per pixel of the RGBA and BGRA layouts (PR `* 4`; NX2 `(nv12 ? 1 : 4)`).
pub const BYTES_PER_PIXEL: u64 = 4;

/// What the image's planes hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Planes {
    /// Luma, then interleaved chroma at half height.
    Nv12,
    /// One plane of RGBA.
    Rgba,
    /// One plane of BGRA.
    Bgra,
}

/// How the rows are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tiling {
    /// Rows one after another, `pitch` bytes apart.
    PitchLinear,
    /// The GPU's block-linear tiling (PR's code 5).
    BlockLinear,
}

/// Which reference's layout a code has (module docs, "The layouts, by code").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reading {
    /// PR's: codes 1 and 5.
    Pr,
    /// NX2's: codes 2, 3, 4, 6 and 10 to 13.
    Nx2,
}

/// A pool's layout: PR's `NativeVideoImageLayout`, with NX2's codes laid out the same way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoLayout {
    /// What the planes hold.
    pub planes: Planes,
    /// How the rows are laid out.
    pub tiling: Tiling,
    /// Whose layout it is.
    pub reading: Reading,
    /// The extent word, as the preference carries it (PR `packed_dimensions`).
    pub extent: u32,
    /// Bytes from one row to the next (PR `row_bytes`).
    pub pitch: u32,
    /// One image's bytes (PR `image_bytes`).
    pub image_bytes: u32,
    /// Bytes from one image to the next (PR `image_stride`).
    pub image_stride: u32,
    /// Where the chroma plane starts in an NV12 image; 0 otherwise (PR `chroma_offset`).
    pub chroma_offset: u32,
    /// The luma plane's bytes, `pitch * height`, in an NV12 image; 0 otherwise (PR
    /// `luma_bytes`).
    pub luma_bytes: u32,
    /// The chroma plane's bytes in an NV12 image (PR `pitch * (height / 2)`; NX2's codes:
    /// `pitch * ((height + 1) / 2)`, the rows NX2 writes); 0 otherwise.
    pub chroma_bytes: u32,
    /// The pool's bytes, all images together (PR `size`).
    pub size: u64,
    /// The pool's alignment (PR `alignment`).
    pub alignment: u64,
}

/// Why a preference and count have no layout (module docs, "The preference"). 991 answers each
/// with `NotImplemented` (PR: `layout ? ResultSuccess : Kernel::ResultNotImplemented`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum LayoutError {
    /// A word that must be zero is not.
    #[error("video preference word {word} is {value}, not 0")]
    WordNotZero {
        /// The word's index in the preference.
        word: usize,
        /// Its value.
        value: u32,
    },
    /// The code is not one [`CODES`] holds.
    #[error("video output layout code {0} is not one starquake sizes")]
    Code(u32),
    /// The count is zero.
    #[error("the video image count is 0")]
    ZeroCount,
    /// Code 1 with a width or height that is odd or 1.
    #[error("an NV12 image of {width}x{height} pixels needs an even width and height of at least 2")]
    Extent {
        /// The width.
        width: u64,
        /// The height.
        height: u64,
    },
    /// One image (its stride, for code 5) over `u32::MAX` bytes.
    #[error("a video image of {image} bytes does not fit in 32 bits (count {count})")]
    TooLarge {
        /// One image's bytes (its stride for code 5).
        image: u64,
        /// The count.
        count: u32,
    },
}

/// `value` rounded up to a multiple of `align`, a power of two. `value` is at most 2^18 and
/// `align` at most 256 here, so this cannot overflow.
const fn align(value: u64, align: u64) -> u64 {
    value.next_multiple_of(align)
}

/// `value` as a `u32`, or [`LayoutError::TooLarge`] for `count` images.
fn fits(value: u64, image: u64, count: u32) -> Result<u32, LayoutError> {
    u32::try_from(value).map_err(|_| LayoutError::TooLarge { image, count })
}

/// One image's sizes before the pool's: (planes, tiling, pitch, chroma offset, image bytes,
/// stride, chroma bytes), all in `u64`. At most 2^18 bytes a row times 2^17 rows, under 2^36,
/// so nothing here overflows.
struct Image {
    planes: Planes,
    tiling: Tiling,
    reading: Reading,
    pitch: u64,
    chroma_offset: u64,
    luma_bytes: u64,
    chroma_bytes: u64,
    bytes: u64,
    stride: u64,
}

/// PR's code 1 (`if (preference[1] == 1)`).
fn pr_nv12(width: u64, height: u64) -> Result<Image, LayoutError> {
    // PR: `if (width < 2 || height < 2 || ((width | height) & 1) != 0) return {};`
    if width < 2 || height < 2 || (width | height) & 1 != 0 {
        return Err(LayoutError::Extent { width, height });
    }
    let pitch = align(width, PR_NV12_PITCH_ALIGN);
    let chroma_offset = pitch * align(height, PR_NV12_ROW_ALIGN);
    let image = PR_MIN_IMAGE.max(chroma_offset + pitch * align(height / 2, PR_NV12_ROW_ALIGN));
    Ok(Image {
        planes: Planes::Nv12,
        tiling: Tiling::PitchLinear,
        reading: Reading::Pr,
        pitch,
        chroma_offset,
        luma_bytes: pitch * height,
        chroma_bytes: pitch * (height / 2),
        bytes: image,
        stride: image,
    })
}

/// PR's code 5 (the code after `if (preference[1] == 1)`).
fn pr_rgba(width: u64, height: u64) -> Image {
    let pitch = align(width, PR_RGBA_WIDTH_ALIGN) * BYTES_PER_PIXEL;
    let image = PR_MIN_IMAGE.max(pitch * align(height, PR_RGBA_ROW_ALIGN));
    Image {
        planes: Planes::Rgba,
        tiling: Tiling::BlockLinear,
        reading: Reading::Pr,
        pitch,
        chroma_offset: 0,
        luma_bytes: 0,
        chroma_bytes: 0,
        bytes: image,
        stride: align(image, PR_RGBA_STRIDE_ALIGN),
    }
}

/// NX2's layouts (`Nx2CalculateDecoderImageLayout`).
fn nx2(planes: Planes, width: u64, height: u64) -> Image {
    let nv12 = planes == Planes::Nv12;
    let bytes = if nv12 { 1 } else { BYTES_PER_PIXEL };
    let pitch = align(width * bytes, NX2_PITCH_ALIGN);
    let chroma_offset = pitch * align(height, NX2_ROW_ALIGN);
    let chroma_rows = height.div_ceil(2);
    let image = if nv12 {
        chroma_offset + pitch * align(chroma_rows, NX2_ROW_ALIGN)
    } else {
        chroma_offset
    };
    Image {
        planes,
        tiling: Tiling::PitchLinear,
        reading: Reading::Nx2,
        pitch,
        chroma_offset: if nv12 { chroma_offset } else { 0 },
        luma_bytes: if nv12 { pitch * height } else { 0 },
        chroma_bytes: if nv12 { pitch * chroma_rows } else { 0 },
        bytes: image,
        stride: image,
    }
}

/// The layout of `count` images of `preference` (module docs), or the first check that fails,
/// in the order of the module docs ("The preference").
pub fn layout(preference: &Preference, count: u32) -> Result<VideoLayout, LayoutError> {
    let zero = |index: usize| match preference[index] {
        0 => Ok(()),
        value => Err(LayoutError::WordNotZero { word: index, value }),
    };
    zero(WORD_ZERO)?;
    let code = preference[WORD_CODE];
    if !CODES.contains(&code) {
        return Err(LayoutError::Code(code));
    }
    if count == 0 {
        return Err(LayoutError::ZeroCount);
    }
    for index in FIRST_TAIL_WORD..PREFERENCE_WORDS {
        zero(index)?;
    }
    let packed = preference[WORD_EXTENT];
    let (width, height) = extent(packed);
    let image = match code {
        CODE_NV12 => pr_nv12(width, height)?,
        CODE_RGBA_BLOCK => pr_rgba(width, height),
        CODE_NV12_NX2 => nx2(Planes::Nv12, width, height),
        CODE_BGRA_NX2 => nx2(Planes::Bgra, width, height),
        _ => nx2(Planes::Rgba, width, height),
    };
    let (stride, bytes) = (image.stride, image.bytes);
    let image_stride = fits(stride, stride, count)?;
    // The stride and the image are at most `u32::MAX` (the check above; the image is at most
    // the stride) and `count - 1` below it, so this is at most `(2^32 - 1)^2 + 2^32 - 1`, under
    // 2^64: it cannot overflow. `count` is not 0 (checked above).
    let size = u64::from(image_stride) * u64::from(count - 1) + bytes;
    Ok(VideoLayout {
        planes: image.planes,
        tiling: image.tiling,
        reading: image.reading,
        extent: packed,
        // Each is at most the image's bytes, which is at most the stride, a `u32`.
        pitch: fits(image.pitch, stride, count)?,
        image_bytes: fits(bytes, stride, count)?,
        image_stride,
        chroma_offset: fits(image.chroma_offset, stride, count)?,
        luma_bytes: fits(image.luma_bytes, stride, count)?,
        chroma_bytes: fits(image.chroma_bytes, stride, count)?,
        size,
        alignment: match image.reading {
            Reading::Pr => PR_ALIGNMENT,
            Reading::Nx2 => NX2_ALIGNMENT,
        },
    })
}

impl VideoLayout {
    /// The width and height in pixels.
    #[must_use]
    pub fn extent(&self) -> (u64, u64) {
        extent(self.extent)
    }

    /// Where image `index` of the pool starts: `index * stride`. `None` past the pool.
    #[must_use]
    pub fn offset(&self, index: u32) -> Option<u64> {
        u64::from(index).checked_mul(u64::from(self.image_stride))
    }
}
