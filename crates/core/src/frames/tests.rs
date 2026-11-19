//! The frame keeper: which frames it keeps within its budget, what it counts, and the files it
//! makes.

use starquake_render::command::PresentTransform;
use starquake_render::frame::RGBA8_BYTES;

use super::*;
use crate::host_log::HostLog;

/// A `width` by `height` frame numbered `number` whose every byte is `fill`.
fn frame(number: u64, width: u32, height: u32, fill: u8) -> Frame {
    let len = usize::try_from(width * height).unwrap() * RGBA8_BYTES;
    Frame {
        frame: number,
        width,
        height,
        transform: PresentTransform::Identity,
        pixels: FramePixels::Rgba8(vec![fill; len]),
    }
}

/// The bytes of a 2 by 2 frame.
const SMALL: usize = 2 * 2 * RGBA8_BYTES;

/// The frame numbers of what `keeper` kept, and its two counts; it is left empty.
fn taken(keeper: &FrameKeeper) -> (Vec<u64>, u64, u64) {
    let dumped = keeper.take();
    let numbers = dumped.files.iter().map(|file| file.frame).collect();
    (numbers, dumped.unknown, dumped.dropped)
}

/// The pixels of a PNG file, as RGBA8 rows.
fn decoded(png: &[u8]) -> (u32, u32, Vec<u8>) {
    let decoder = png::Decoder::new(std::io::Cursor::new(png));
    let mut reader = decoder.read_info().unwrap();
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut pixels).unwrap();
    assert_eq!(
        (info.color_type, info.bit_depth),
        (png::ColorType::Rgba, png::BitDepth::Eight)
    );
    pixels.truncate(info.buffer_size());
    (info.width, info.height, pixels)
}

#[test]
fn a_file_name_is_the_frame_number_padded_to_six_digits() {
    assert_eq!(FRAMES_FOLDER, "frames");
    assert_eq!(FRAME_NUMBER_DIGITS, 6);
    assert_eq!(frame_file_name(0), "000000.png");
    assert_eq!(frame_file_name(42), "000042.png");
    // The last number the padding holds, and the first past it, written in full.
    assert_eq!(frame_file_name(999_999), "999999.png");
    assert_eq!(frame_file_name(1_000_000), "1000000.png");
    assert_eq!(frame_file_name(u64::MAX), "18446744073709551615.png");
}

#[test]
fn a_new_keeper_keeps_the_runs_budget_and_nothing_yet() {
    assert_eq!(FRAME_BYTES, 64 << 20);
    let keeper = FrameKeeper::default();
    assert_eq!(keeper.budget, FRAME_BYTES);
    assert_eq!(taken(&keeper), (Vec::new(), 0, 0));
}

#[test]
fn the_newest_frames_that_fit_are_kept_in_order() {
    // Room for exactly two: the third pushes out the first.
    let keeper = FrameKeeper::new(2 * SMALL);
    for number in 0..3 {
        keeper.frame(frame(number, 2, 2, 0x11));
    }
    assert_eq!(taken(&keeper), (vec![1, 2], 0, 1));
    // One byte less: only the newest stays.
    let keeper = FrameKeeper::new(2 * SMALL - 1);
    for number in 0..3 {
        keeper.frame(frame(number, 2, 2, 0x11));
    }
    assert_eq!(taken(&keeper), (vec![2], 0, 2));
}

#[test]
fn a_frame_of_exactly_the_budget_is_kept_and_one_byte_more_is_not() {
    let keeper = FrameKeeper::new(SMALL);
    keeper.frame(frame(7, 2, 2, 1));
    assert_eq!(taken(&keeper), (vec![7], 0, 0));
    let keeper = FrameKeeper::new(SMALL - 1);
    keeper.frame(frame(7, 2, 2, 1));
    assert_eq!(taken(&keeper), (Vec::new(), 0, 1));
    // A frame too large alone pushes out nothing already kept.
    let keeper = FrameKeeper::new(SMALL);
    keeper.frame(frame(1, 1, 1, 1));
    keeper.frame(frame(2, 4, 4, 1));
    assert_eq!(taken(&keeper), (vec![1], 0, 1));
}

#[test]
fn frames_of_different_sizes_are_counted_by_their_own_bytes() {
    // 4, 16 and 4 bytes; a budget of 20 holds the first two, then the third pushes out only the
    // first (16 + 4 = 20 fits). Counting every frame as the first's size would keep all three;
    // counting each as the largest would keep one.
    let one = RGBA8_BYTES;
    let keeper = FrameKeeper::new(5 * one);
    keeper.frame(frame(0, 1, 1, 1));
    keeper.frame(frame(1, 2, 2, 2));
    keeper.frame(frame(2, 1, 1, 3));
    assert_eq!(taken(&keeper), (vec![1, 2], 0, 1));
    // Across two takes the budget starts again.
    keeper.frame(frame(3, 2, 2, 4));
    keeper.frame(frame(4, 1, 1, 5));
    assert_eq!(taken(&keeper), (vec![3, 4], 0, 0));
}

#[test]
fn an_evicted_frame_frees_its_own_bytes_not_the_new_frames() {
    // 4, 16 and 16 bytes in a budget of 20: the third pushes out the first (4 bytes) and still
    // does not fit (16 + 16 = 32), so it pushes out the second too. Freeing the new frame's 16
    // bytes for the first instead would leave the count at 4 and keep 1 and 2, 32 bytes.
    let one = RGBA8_BYTES;
    let keeper = FrameKeeper::new(5 * one);
    keeper.frame(frame(0, 1, 1, 1));
    keeper.frame(frame(1, 2, 2, 2));
    keeper.frame(frame(2, 2, 2, 3));
    assert_eq!(taken(&keeper), (vec![2], 0, 2));
}

#[test]
fn an_unknown_frame_is_counted_and_takes_no_room() {
    let keeper = FrameKeeper::new(SMALL);
    keeper.frame(frame(1, 2, 2, 1));
    keeper.frame(Frame {
        pixels: FramePixels::Unknown,
        ..frame(2, 2, 2, 0)
    });
    assert_eq!(taken(&keeper), (vec![1], 1, 0));
    // What was taken is gone.
    assert_eq!(taken(&keeper), (Vec::new(), 0, 0));
}

#[test]
fn a_kept_frame_is_written_as_a_png_of_its_pixels() {
    let keeper = FrameKeeper::default();
    let pixels: Vec<u8> = (0..u8::try_from(3 * 2 * RGBA8_BYTES).unwrap()).collect();
    keeper.frame(Frame {
        pixels: FramePixels::Rgba8(pixels.clone()),
        ..frame(9, 3, 2, 0)
    });
    let dumped = keeper.take();
    assert_eq!(dumped.files.len(), 1);
    assert_eq!(dumped.files[0].frame, 9);
    assert_eq!(decoded(&dumped.files[0].png), (3, 2, pixels));
}

#[test]
fn a_frame_the_encoder_refuses_is_logged_and_counted_as_dropped() {
    let keeper = FrameKeeper::default();
    // Pixels for a 2 by 2 frame, said to be 3 by 2: the encoder refuses the size.
    keeper.frame(Frame {
        width: 3,
        ..frame(5, 2, 2, 1)
    });
    keeper.frame(frame(6, 2, 2, 1));
    let log = HostLog::new();
    let dumped = tracing::subscriber::with_default(log.clone(), || keeper.take());
    assert_eq!(
        dumped.files.iter().map(|f| f.frame).collect::<Vec<_>>(),
        [6]
    );
    assert_eq!((dumped.unknown, dumped.dropped), (0, 1));
    let (lines, _) = log.take();
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(
        lines[0].contains("a dumped frame could not be encoded") && lines[0].contains("frame=5"),
        "{lines:?}"
    );
}

#[test]
fn a_frame_number_is_read_back_only_from_a_name_this_module_makes() {
    for number in [0, 42, 999_999, 1_000_000, u64::MAX] {
        assert_eq!(frame_number(&frame_file_name(number)), Some(number));
    }
    for name in [
        "42.png",
        "0000042.png",
        "+00042.png",
        "000042.PNG",
        "000042.png.bak",
        "000042",
        ".png",
        "png",
        "00004x.png",
        "-00042.png",
        // One past `u64::MAX`.
        "18446744073709551616.png",
        "",
    ] {
        assert_eq!(frame_number(name), None, "{name}");
    }
}
