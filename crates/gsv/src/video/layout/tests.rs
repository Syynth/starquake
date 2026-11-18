//! The pool layout against values computed by hand from the cited formulas.

use super::*;

/// A preference with `code` and an extent of `width` by `height`, zeros elsewhere.
fn preference(code: u32, width: u32, height: u32) -> Preference {
    let mut words = [0; PREFERENCE_WORDS];
    words[WORD_CODE] = code;
    words[WORD_EXTENT] = (width - 1) | ((height - 1) << 16);
    words
}

/// The references' constants, written out as the cited sources write them, so that a wrong
/// shipping constant fails here and not only where the constant is imported.
#[test]
fn the_constants_are_the_references_values() {
    assert_eq!(WORD_ZERO, 0);
    assert_eq!(WORD_CODE, 1);
    assert_eq!(WORD_EXTENT, 2);
    assert_eq!(FIRST_TAIL_WORD, 3);
    assert_eq!(PREFERENCE_WORDS, 16);
    assert_eq!(CODE_NV12, 1);
    assert_eq!(CODE_RGBA_BLOCK, 5);
    assert_eq!(CODE_NV12_NX2, 4);
    assert_eq!(CODE_BGRA_NX2, 2);
    assert_eq!(CODES_RGBA_NX2, [3, 6, 10, 11, 12, 13]);
    assert_eq!(PR_NV12_PITCH_ALIGN, 256);
    assert_eq!(PR_NV12_ROW_ALIGN, 4);
    assert_eq!(PR_MIN_IMAGE, 4096);
    assert_eq!(PR_RGBA_WIDTH_ALIGN, 16);
    assert_eq!(PR_RGBA_ROW_ALIGN, 128);
    assert_eq!(PR_RGBA_STRIDE_ALIGN, 8192);
    assert_eq!(PR_ALIGNMENT, 4096);
    assert_eq!(NX2_PITCH_ALIGN, 64);
    assert_eq!(NX2_ROW_ALIGN, 128);
    assert_eq!(NX2_ALIGNMENT, 0x2000);
    // Every code taken, and only those: PR's two and the ones NX2's comment names.
    let mut codes = CODES.to_vec();
    codes.sort_unstable();
    assert_eq!(codes, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]);
}

#[test]
fn code_1_is_pr_s_layout_not_nx2_s() {
    // 640x360: pitch align(640, 256) = 768; chroma at 768 * align(360, 4) = 276480; image
    // 276480 + 768 * align(180, 4) = 414720 = 0x65400. NX2's answer (row 96) is 0x64000.
    let layout = layout(&preference(CODE_NV12, 640, 360), 1).unwrap();
    assert_eq!(
        layout,
        VideoLayout {
            planes: Planes::Nv12,
            tiling: Tiling::PitchLinear,
            reading: Reading::Pr,
            extent: 0x27F | (0x167 << 16),
            pitch: 768,
            image_bytes: 0x65400,
            image_stride: 0x65400,
            chroma_offset: 276_480,
            luma_bytes: 276_480,
            chroma_bytes: 138_240,
            size: 0x65400,
            alignment: 4096,
        }
    );
    // Three images back to back.
    let three = super::layout(&preference(CODE_NV12, 640, 360), 3).unwrap();
    assert_eq!(three.size, 3 * 0x65400);
    assert_eq!(three.offset(2), Some(2 * 0x65400));
}

#[test]
fn code_1_rounds_rows_to_4_and_raises_small_images_to_4096() {
    // 2x2: pitch 256, chroma at 256 * 4 = 1024, image max(4096, 1024 + 256 * 4) = 4096.
    let small = layout(&preference(CODE_NV12, 2, 2), 1).unwrap();
    assert_eq!(
        (small.pitch, small.chroma_offset, small.image_bytes),
        (256, 1024, 4096)
    );
    // 258x6: pitch 512 (one past 256), chroma at 512 * 8 = 4096, image 4096 + 512 * 4 = 6144.
    let rows = layout(&preference(CODE_NV12, 258, 6), 1).unwrap();
    assert_eq!(
        (rows.pitch, rows.chroma_offset, rows.image_bytes),
        (512, 4096, 6144)
    );
    assert_eq!((rows.luma_bytes, rows.chroma_bytes), (512 * 6, 512 * 3));
}

#[test]
fn code_1_needs_an_even_extent_of_at_least_2() {
    for (width, height) in [(3, 2), (2, 3), (1, 2), (2, 1)] {
        assert_eq!(
            layout(&preference(CODE_NV12, width, height), 1),
            Err(LayoutError::Extent {
                width: u64::from(width),
                height: u64::from(height),
            }),
            "{width}x{height}"
        );
    }
    assert!(layout(&preference(CODE_NV12, 2, 2), 1).is_ok());
}

#[test]
fn code_4_takes_an_odd_extent() {
    // NX2's layout: 3x1 has a pitch of 64, chroma at 64 * 128, one chroma row in 128.
    let odd = layout(&preference(CODE_NV12_NX2, 3, 1), 1).unwrap();
    assert_eq!(
        (odd.pitch, odd.chroma_offset, odd.image_bytes),
        (64, 64 * 128, 64 * 256)
    );
    assert_eq!((odd.luma_bytes, odd.chroma_bytes), (64, 64));
}

#[test]
fn code_5_is_pr_s_block_linear_rgba() {
    // 100x50: row bytes align(100, 16) * 4 = 448, image 448 * 128 = 57344, a multiple of 8192.
    let layout = layout(&preference(CODE_RGBA_BLOCK, 100, 50), 3).unwrap();
    assert_eq!(
        (layout.planes, layout.tiling, layout.reading),
        (Planes::Rgba, Tiling::BlockLinear, Reading::Pr)
    );
    assert_eq!(
        (layout.pitch, layout.image_bytes, layout.image_stride),
        (448, 57_344, 57_344)
    );
    assert_eq!(layout.size, 2 * 57_344 + 57_344);
    assert_eq!(
        (layout.chroma_offset, layout.luma_bytes, layout.chroma_bytes),
        (0, 0, 0)
    );
    // A one-pixel image is raised to one block of rows: 64 * 128 = 8192, over PR's 4096 floor.
    let tiny = super::layout(&preference(CODE_RGBA_BLOCK, 1, 1), 1).unwrap();
    assert_eq!((tiny.pitch, tiny.image_bytes), (64, 8192));
}

#[test]
fn nx2_s_codes_have_nx2_s_layouts() {
    // Code 4, 640x360: NX2's row 96 answer, 0x64000: pitch 640, chroma at 640 * 384 = 245760,
    // chroma 640 * align(180, 128) = 163840.
    let nv12 = layout(&preference(CODE_NV12_NX2, 640, 360), 1).unwrap();
    assert_eq!(
        (nv12.planes, nv12.reading, nv12.image_bytes),
        (Planes::Nv12, Reading::Nx2, 0x64000)
    );
    assert_eq!((nv12.pitch, nv12.chroma_offset), (640, 245_760));
    // Code 2, 100x50: pitch align(400, 64) = 448, image 448 * 128 = 57344, back to back.
    let bgra = layout(&preference(CODE_BGRA_NX2, 100, 50), 2).unwrap();
    assert_eq!(
        (bgra.planes, bgra.tiling, bgra.reading),
        (Planes::Bgra, Tiling::PitchLinear, Reading::Nx2)
    );
    assert_eq!(
        (bgra.pitch, bgra.image_bytes, bgra.size),
        (448, 57_344, 114_688)
    );
    for code in CODES_RGBA_NX2 {
        let rgba = layout(&preference(code, 100, 50), 2).unwrap();
        assert_eq!(
            (rgba.planes, rgba.tiling, rgba.reading),
            (Planes::Rgba, Tiling::PitchLinear, Reading::Nx2),
            "code {code}"
        );
        assert_eq!(
            (rgba.pitch, rgba.image_bytes, rgba.size),
            (448, 57_344, 114_688)
        );
    }
}

#[test]
fn each_code_s_alignment_is_its_reference_s() {
    for code in CODES {
        let layout = layout(&preference(code, 2, 2), 1).unwrap();
        let expected = if code == CODE_NV12 || code == CODE_RGBA_BLOCK {
            PR_ALIGNMENT
        } else {
            NX2_ALIGNMENT
        };
        assert_eq!(layout.alignment, expected, "code {code}");
    }
}

#[test]
fn an_unnamed_code_is_refused() {
    for code in [0, 7, 8, 9, 14, u32::MAX] {
        assert_eq!(
            layout(&preference(code, 2, 2), 1),
            Err(LayoutError::Code(code))
        );
    }
}

#[test]
fn the_request_checks_apply_to_every_code() {
    for code in CODES {
        let mut word_zero = preference(code, 2, 2);
        word_zero[WORD_ZERO] = 9;
        assert_eq!(
            layout(&word_zero, 1),
            Err(LayoutError::WordNotZero { word: 0, value: 9 }),
            "code {code}"
        );
        assert_eq!(
            layout(&preference(code, 2, 2), 0),
            Err(LayoutError::ZeroCount)
        );
        for word in FIRST_TAIL_WORD..PREFERENCE_WORDS {
            let mut tail = preference(code, 2, 2);
            tail[word] = 0x100 + u32::try_from(word).unwrap();
            assert_eq!(
                layout(&tail, 1),
                Err(LayoutError::WordNotZero {
                    word,
                    value: 0x100 + u32::try_from(word).unwrap(),
                }),
                "code {code}, word {word}"
            );
        }
    }
}

#[test]
fn the_checks_come_in_pr_s_order() {
    // Word 0 before the code.
    let mut both = preference(7, 2, 2);
    both[WORD_ZERO] = 1;
    assert_eq!(
        layout(&both, 0),
        Err(LayoutError::WordNotZero { word: 0, value: 1 })
    );
    // The code before the count.
    assert_eq!(layout(&preference(7, 2, 2), 0), Err(LayoutError::Code(7)));
    // The count before the trailing words.
    let mut tail = preference(CODE_NV12, 2, 2);
    tail[15] = 1;
    assert_eq!(layout(&tail, 0), Err(LayoutError::ZeroCount));
    // The trailing words before the extent: the lowest first.
    let mut odd = preference(CODE_NV12, 3, 3);
    odd[4] = 4;
    odd[3] = 3;
    assert_eq!(
        layout(&odd, 1),
        Err(LayoutError::WordNotZero { word: 3, value: 3 })
    );
}

#[test]
fn an_image_over_four_gib_is_refused() {
    // Code 2 at 65536 pixels a row: pitch 2^18. 16256 rows (127 blocks of 128) is
    // 4_261_412_864 bytes, under 2^32; one more row rounds to 128 blocks, 2^32: too large.
    let fits = layout(&preference(CODE_BGRA_NX2, 65_536, 16_256), 1).unwrap();
    assert_eq!(fits.image_bytes, 4_261_412_864);
    assert_eq!(
        layout(&preference(CODE_BGRA_NX2, 65_536, 16_257), 1),
        Err(LayoutError::TooLarge {
            image: 1 << 32,
            count: 1,
        })
    );
    // Code 1 at its largest extent: pitch 65536, chroma at 2^32: too large.
    assert!(matches!(
        layout(&preference(CODE_NV12, 65_536, 65_536), 1),
        Err(LayoutError::TooLarge { .. })
    ));
    // The largest image at the largest count still has a size: no overflow.
    let most = layout(&preference(CODE_BGRA_NX2, 65_536, 16_256), u32::MAX).unwrap();
    assert_eq!(most.size, 4_261_412_864 * u64::from(u32::MAX));
}

#[test]
fn error_text_is_plain_words() {
    assert_eq!(
        LayoutError::Code(7).to_string(),
        "video output layout code 7 is not one starquake sizes"
    );
    assert_eq!(
        LayoutError::ZeroCount.to_string(),
        "the video image count is 0"
    );
    assert_eq!(
        LayoutError::Extent {
            width: 3,
            height: 2
        }
        .to_string(),
        "an NV12 image of 3x2 pixels needs an even width and height of at least 2"
    );
}
