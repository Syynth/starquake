//! The work-buffer size against Eden-NX2's (module docs).
//!
//! The expected sizes were taken from Eden-NX2's `System::GetWorkBufferSize`
//! (`eden-nx2/src/audio_core/renderer/system.cpp` @ 50fa86bb, with
//! `SplitterContext::CalcWorkBufferSize` from `splitter_context.cpp`) compiled over Eden-NX2's
//! own headers and `behavior_info.cpp` (Apple clang, libc++, arm64), fed each parameter below.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use super::*;

/// A parameter with the fields the size reads; the others are fixed (none of them changes the
/// size: `the_fields_the_size_does_not_read_do_not_change_it`).
#[expect(
    clippy::too_many_arguments,
    reason = "one argument per field of a golden row, in the oracle's order"
)]
const fn parameter(
    sample_count: u32,
    mixes: u32,
    sub_mixes: u32,
    voices: u32,
    sinks: u32,
    effects: u32,
    perf_frames: u32,
    splitter_infos: u32,
    splitter_destinations: i32,
    revision: u32,
) -> RendererParameter {
    RendererParameter {
        sample_rate: 48_000,
        sample_count,
        mixes,
        sub_mixes,
        voices,
        sinks,
        effects,
        perf_frames,
        voice_drop_enabled: 1,
        unknown_21: 0,
        rendering_device: 0,
        execution_mode: 0,
        splitter_infos,
        splitter_destinations,
        external_context_size: 0,
        revision,
    }
}

/// A golden row: a parameter and Eden-NX2's size for it.
#[expect(
    clippy::too_many_arguments,
    reason = "one argument per field of a golden row, in the oracle's order"
)]
const fn golden(
    sample_count: u32,
    mixes: u32,
    sub_mixes: u32,
    voices: u32,
    sinks: u32,
    effects: u32,
    perf_frames: u32,
    splitter_infos: u32,
    splitter_destinations: i32,
    revision: u32,
    size: u64,
) -> (RendererParameter, u64) {
    (
        parameter(
            sample_count,
            mixes,
            sub_mixes,
            voices,
            sinks,
            effects,
            perf_frames,
            splitter_infos,
            splitter_destinations,
            revision,
        ),
        size,
    )
}

/// Eden-NX2's sizes. In order: one parameter at every revision 0 to 15; the same at tagged
/// words (`REV?`, `REv?`, `REV0`, `REV4`, `REv5`, `REV9`, and `0x3900_0000`, which carries no
/// tag and reads as revision 8); fields with bytes above the low byte at revisions on both sides
/// of each feature; each field alone at revision 1 and 15; and 48 parameters from a fixed linear
/// congruential sequence (seed `0x5EED_0498`, Numerical Recipes' constants).
#[rustfmt::skip]
const GOLDEN: &[(RendererParameter, u64)] = &[
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x0, 249856),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x1, 249856),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x2, 258048),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x3, 258048),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x4, 258048),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x5, 262144),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x6, 262144),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x7, 262144),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x8, 262144),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x9, 266240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0xa, 266240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0xb, 266240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0xc, 266240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0xd, 266240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0xe, 266240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0xf, 266240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x3f574552, 266240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x3f774552, 266240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x30574552, 249856),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x34574552, 258048),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x35774552, 262144),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x39574552, 266240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x39000000, 262144),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x1, 211259392),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x2, 212459520),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x4, 212459520),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x5, 257490944),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x7, 257490944),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x8, 257495040),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x9, 257630208),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0xf, 257630208),
    golden(0x1234, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 212992),
    golden(0x1234, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf, 147456),
    golden(0x0, 0x1335, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 122880),
    golden(0x0, 0x1335, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf, 53248),
    golden(0x0, 0x0, 0x1436, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 50683904),
    golden(0x0, 0x0, 0x1436, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf, 161583104),
    golden(0x0, 0x0, 0x0, 0x1537, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 8183808),
    golden(0x0, 0x0, 0x0, 0x1537, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf, 12939264),
    golden(0x0, 0x0, 0x0, 0x0, 0x1638, 0x0, 0x0, 0x0, 0x0, 0x1, 38371328),
    golden(0x0, 0x0, 0x0, 0x0, 0x1638, 0x0, 0x0, 0x0, 0x0, 0xf, 40488960),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x1739, 0x0, 0x0, 0x0, 0x1, 9756672),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x1739, 0x0, 0x0, 0x0, 0xf, 21057536),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x183a, 0x0, 0x0, 0x1, 10276864),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x183a, 0x0, 0x0, 0xf, 15368192),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x193b, 0x0, 0x1, 102400),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x193b, 0x0, 0xf, 241664),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1a3c, 0x1, 102400),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1a3c, 0xf, 9572352),
    golden(0x4bf, 0x11, 0x13, 0x215, 0x0, 0x8, 0x4, 0x37, 0x1, 0x2, 1572864),
    golden(0xb5f, 0x1e, 0x37, 0x3f3, 0x0, 0x3e, 0x5, 0x10, 0xa3, 0xc, 7180288),
    golden(0x7ff, 0xc, 0x25, 0x7b, 0x1, 0x3b, 0x1, 0x3f, 0x48, 0x1, 1347584),
    golden(0x9bb, 0xd, 0x2f, 0x43, 0x0, 0x17, 0x2, 0x3, 0x9b, 0x3, 1540096),
    golden(0x275, 0x8, 0x8, 0x3f5, 0x5, 0x23, 0x3, 0x2f, 0x8d, 0x9, 7700480),
    golden(0x3f1, 0x37, 0x29, 0x3ef, 0x4, 0x0, 0x5, 0x8, 0x72, 0xe, 6594560),
    golden(0xdf6, 0x9, 0x2f, 0xe5, 0x7, 0x4, 0xd, 0x2a, 0x24, 0xf, 2904064),
    golden(0x572, 0x0, 0x39, 0x17d, 0x1, 0x1b, 0x3, 0x23, 0x1e, 0x0, 1347584),
    golden(0xc95, 0x13, 0xe, 0xf1, 0x3, 0x23, 0x3, 0x14, 0xa1, 0x6, 2244608),
    golden(0x3f3, 0xd, 0x34, 0x2b0, 0x2, 0xe, 0xf, 0x16, 0xd2, 0xd, 5091328),
    golden(0xca6, 0x2b, 0x19, 0x2fd, 0x5, 0x3, 0xa, 0x12, 0xd9, 0x2, 3616768),
    golden(0x369, 0x0, 0x2d, 0x8d, 0x3, 0x3c, 0xd, 0x27, 0x3, 0x9, 1728512),
    golden(0xcbe, 0x14, 0x5, 0x32b, 0x2, 0x28, 0x5, 0x7, 0xdf, 0x8, 3633152),
    golden(0xa07, 0x3f, 0x38, 0x356, 0x0, 0x9, 0x2, 0x18, 0x63, 0x8, 8937472),
    golden(0xfaf, 0x12, 0x4, 0x2e1, 0x5, 0x16, 0x8, 0x17, 0x7, 0xa, 4304896),
    golden(0x442, 0x2e, 0x2a, 0x291, 0x6, 0x1a, 0x0, 0x30, 0xe8, 0xf, 7860224),
    golden(0x8f, 0x29, 0xd, 0xc2, 0x2, 0x12, 0x7, 0x26, 0xe6, 0xe, 2420736),
    golden(0x8cc, 0x2f, 0x15, 0x101, 0x0, 0x12, 0xd, 0x30, 0xc5, 0x1, 2236416),
    golden(0x6b1, 0x1e, 0x11, 0xb2, 0x6, 0x1f, 0x7, 0x18, 0x4e, 0x7, 2265088),
    golden(0xc9b, 0x28, 0xf, 0xab, 0x2, 0x15, 0xc, 0x1e, 0x6e, 0x3, 1896448),
    golden(0x9aa, 0x34, 0x5, 0x3d6, 0x4, 0x1, 0xb, 0x12, 0x55, 0xa, 5607424),
    golden(0x6e2, 0x3c, 0x2c, 0x2d1, 0x0, 0x7, 0x2, 0x3d, 0x98, 0x7, 9711616),
    golden(0x54d, 0x2f, 0x1e, 0x90, 0x7, 0x3c, 0x7, 0x37, 0x4f, 0x9, 3837952),
    golden(0x516, 0xf, 0x3e, 0x3f9, 0x7, 0xc, 0x2, 0x11, 0x36, 0x3, 3481600),
    golden(0xdae, 0x13, 0xf, 0x286, 0x6, 0x15, 0xf, 0x2b, 0xce, 0xa, 5832704),
    golden(0xee9, 0x6, 0x1b, 0x3e6, 0x1, 0x9, 0x0, 0x1e, 0x7c, 0xa, 6111232),
    golden(0x321, 0x25, 0xd, 0x29c, 0x2, 0xf, 0x8, 0x13, 0xa8, 0x0, 2048000),
    golden(0xa51, 0x3, 0x16, 0x19d, 0x4, 0x21, 0x0, 0x29, 0xde, 0x0, 1183744),
    golden(0x63a, 0x28, 0xa, 0x2f4, 0x1, 0x2e, 0x4, 0x12, 0xf0, 0x0, 2195456),
    golden(0xf81, 0x2c, 0x3, 0x360, 0x4, 0x38, 0x6, 0x35, 0x11, 0xa, 8028160),
    golden(0xbce, 0x20, 0x3c, 0x2f4, 0x6, 0x36, 0xa, 0xc, 0xfa, 0xc, 6643712),
    golden(0xced, 0x25, 0x36, 0x2b5, 0x3, 0x33, 0x7, 0x6, 0x8, 0xa, 5607424),
    golden(0xf0, 0x11, 0x16, 0x33f, 0x5, 0x2c, 0xb, 0x25, 0x5a, 0xa, 6254592),
    golden(0xb4a, 0x10, 0x4, 0x60, 0x7, 0x35, 0x6, 0x20, 0xf7, 0x6, 1638400),
    golden(0xef4, 0x0, 0xc, 0x1b9, 0x2, 0x13, 0xc, 0x2, 0xe5, 0xd, 1966080),
    golden(0x18b, 0x13, 0x0, 0x63, 0x3, 0x22, 0x7, 0x8, 0x52, 0x2, 471040),
    golden(0xf6e, 0x30, 0x13, 0x187, 0x2, 0x2f, 0x2, 0x5, 0xae, 0xa, 3629056),
    golden(0x8e4, 0x11, 0x3e, 0x204, 0x4, 0x1e, 0xf, 0x1, 0xce, 0x0, 3022848),
    golden(0x33, 0x25, 0x1e, 0x30c, 0x0, 0x5, 0x0, 0x14, 0x9, 0x3, 2646016),
    golden(0xc9, 0x2e, 0x13, 0x3c5, 0x4, 0xd, 0x0, 0xf, 0x5d, 0x6, 5160960),
    golden(0x756, 0x23, 0x21, 0x3ec, 0x0, 0x17, 0xb, 0x13, 0x8b, 0xe, 6615040),
    golden(0x2ef, 0xf, 0x11, 0x36d, 0x1, 0x14, 0x0, 0x36, 0x36, 0x7, 7299072),
    golden(0x72b, 0x2f, 0x10, 0x30b, 0x7, 0x2b, 0x5, 0x24, 0x6, 0xe, 6512640),
    golden(0x749, 0x17, 0xe, 0x1fd, 0x6, 0x16, 0x2, 0x3b, 0xca, 0x7, 5369856),
    golden(0x247, 0xc, 0x20, 0x8d, 0x7, 0x4, 0x4, 0x2d, 0x90, 0x8, 2093056),
    golden(0x80b, 0x2a, 0x1d, 0x3bb, 0x7, 0x38, 0xb, 0x21, 0xce, 0xb, 8339456),
    golden(0xf7d, 0x3e, 0x4, 0x1d7, 0x6, 0x27, 0xb, 0x11, 0x7d, 0xe, 4173824),
    golden(0x5a9, 0x2c, 0x13, 0x2a, 0x1, 0x1c, 0xe, 0x2c, 0x37, 0x5, 1884160),
];

#[test]
fn the_size_is_edens_on_every_golden_parameter() {
    for (index, (params, size)) in GOLDEN.iter().enumerate() {
        assert_eq!(
            work_buffer_size(params),
            Ok(*size),
            "row {index}: {params:?}"
        );
    }
}

#[test]
fn the_fields_the_size_does_not_read_do_not_change_it() {
    // Eden's `System::GetWorkBufferSize` reads neither the sample rate, the voice-drop flag,
    // `unk_21`, the rendering device, the execution mode nor the external context size.
    let (params, size) = GOLDEN[15];
    let other = RendererParameter {
        sample_rate: 32_000,
        voice_drop_enabled: 0,
        unknown_21: 0x21,
        rendering_device: 1,
        execution_mode: 1,
        external_context_size: 0x1_0000,
        ..params
    };
    assert_eq!(work_buffer_size(&other), Ok(size));
}

/// `parameter`'s fixed fields and zeroes elsewhere, at `revision`, with `change` applied.
fn only(revision: u32, change: impl FnOnce(&mut RendererParameter)) -> RendererParameter {
    let mut params = parameter(0, 0, 0, 0, 0, 0, 0, 0, 0, revision);
    change(&mut params);
    params
}

#[test]
fn every_32_bit_term_is_refused_one_past_its_edge() {
    let max = u32::MAX;
    // (term, the parameter at its edge with Eden's size, the parameter one past it). Each pair
    // differs only at the edge, and in the one past it no other term wraps: a term computed with
    // Eden's wrapping arithmetic would answer a size there instead of refusing.
    let cases = [
        (
            "sub_mixes * MAX_EFFECTS",
            only(1, |p| p.sub_mixes = max / MAX_EFFECTS),
            164_014_157_824,
            only(1, |p| p.sub_mixes = max / MAX_EFFECTS + 1),
        ),
        (
            "(sinks + sub_mixes) * TARGET_SAMPLE_COUNT",
            only(1, |p| p.sinks = max / TARGET_SAMPLE_COUNT),
            120_402_354_176,
            only(1, |p| p.sinks = max / TARGET_SAMPLE_COUNT + 1),
        ),
        (
            "mixes + MAX_CHANNELS",
            only(1, |p| p.mixes = max - MAX_CHANNELS),
            17_179_971_584,
            only(1, |p| p.mixes = max - MAX_CHANNELS + 1),
        ),
        (
            // `sub_mixes + 1` = 32767 and 32768: `count * (count * 4)` is 2^32 - 262140, then
            // 2^32.
            "node states: count * (count * 4)",
            only(Feature::Splitter.since(), |p| p.sub_mixes = 32_766),
            4_750_393_344,
            only(Feature::Splitter.since(), |p| p.sub_mixes = 32_767),
        ),
        (
            "voices * MAX_WAVE_BUFFERS",
            only(1, |p| p.voices = max / MAX_WAVE_BUFFERS),
            1_597_727_936_512,
            only(1, |p| p.voices = max / MAX_WAVE_BUFFERS + 1),
        ),
        (
            "effects + voices * MAX_WAVE_BUFFERS",
            only(1, |p| {
                p.voices = 1;
                p.effects = max - MAX_WAVE_BUFFERS;
            }),
            6_975_026_987_008,
            only(1, |p| {
                p.voices = 1;
                p.effects = max - MAX_WAVE_BUFFERS + 1;
            }),
        ),
        (
            "performance entries: voices + effects + sub_mixes + sinks + 1",
            only(1, |p| {
                p.perf_frames = 1;
                p.effects = max - 1;
            }),
            7_112_465_944_576,
            only(1, |p| {
                p.perf_frames = 1;
                p.effects = max;
            }),
        ),
        (
            "perf_frames + 1",
            only(1, |p| p.perf_frames = max - 1),
            7_043_746_467_840,
            only(1, |p| p.perf_frames = max),
        ),
        (
            "command nodes: effects + voices + sinks + sub_mixes + 1",
            only(Feature::VariadicCommandBufferSize.since(), |p| {
                p.effects = max - 1;
            }),
            14_087_492_759_552,
            only(Feature::VariadicCommandBufferSize.since(), |p| {
                p.effects = max
            }),
        ),
    ];
    for (term, edge, size, past) in cases {
        assert_eq!(work_buffer_size(&edge), Ok(size), "{term} at its edge");
        assert_eq!(
            work_buffer_size(&past),
            Err(WorkBufferError::Overflow),
            "{term} one past its edge"
        );
    }
}

#[test]
fn a_sum_of_sinks_and_sub_mixes_past_32_bits_is_refused() {
    // `sinks + sub_mixes` wraps to 0 here in Eden, and every later term would then pass; the
    // product with `TARGET_SAMPLE_COUNT` refuses any sum this large on its own, so the sum's own
    // edge is the product's (above).
    let past = only(1, |p| {
        p.sinks = u32::MAX;
        p.sub_mixes = 1;
    });
    assert_eq!(work_buffer_size(&past), Err(WorkBufferError::Overflow));
}

#[test]
fn a_parameter_whose_64_bit_size_would_wrap_is_refused() {
    // Not at each term's exact edge: one parameter per 64-bit product that wraps in Eden.
    // `(samples) * (mixes + 6)`: (4 * u32::MAX) * u32::MAX passes 2^64.
    let mix_buffers = only(1, |p| {
        p.sample_count = u32::MAX;
        p.mixes = u32::MAX - MAX_CHANNELS;
    });
    assert_eq!(
        work_buffer_size(&mix_buffers),
        Err(WorkBufferError::Overflow)
    );
    // The performance frames: about 2^36 bytes a frame times 2^32 frames.
    let performance = only(1, |p| {
        p.effects = u32::MAX - 1;
        p.perf_frames = u32::MAX - 1;
    });
    assert_eq!(
        work_buffer_size(&performance),
        Err(WorkBufferError::Overflow)
    );
}

#[test]
fn a_revision_past_15_is_refused_before_any_term() {
    for (word, number) in [
        (CURRENT_REVISION + 1, 16),
        (0xFF, 0xFF),
        (u32::from_le_bytes(*b"REV@"), 16),
        (REVISION_TAG_FROM, 0xCF),
    ] {
        let expected = Err(WorkBufferError::InvalidRevision { word, number });
        assert_eq!(work_buffer_size(&only(word, |_| {})), expected, "{word:#x}");
        // With a term that would also wrap, and a negative destination count: the revision is
        // reported.
        let both = only(word, |p| {
            p.sub_mixes = u32::MAX;
            p.splitter_destinations = -1;
        });
        assert_eq!(work_buffer_size(&both), expected, "{word:#x}");
    }
    // The newest revision is accepted, as number and as tag.
    assert!(work_buffer_size(&only(CURRENT_REVISION, |_| {})).is_ok());
    assert!(work_buffer_size(&only(u32::from_le_bytes(*b"REV?"), |_| {})).is_ok());
}

#[test]
fn a_native_revision_16_is_refused_as_eden_nx2_does() {
    // PR #4588 accepts `REv@` (`NativeCurrentRevision` = 16); Eden-NX2 reads it as 16 and
    // refuses it (module docs, "The revision word").
    let word = u32::from_le_bytes(*b"REv@");
    assert_eq!(
        work_buffer_size(&only(word, |_| {})),
        Err(WorkBufferError::InvalidRevision { word, number: 16 })
    );
}

#[test]
fn a_word_without_the_rev_tag_is_read_as_eden_nx2_reads_it() {
    // PR #4588 refuses a word whose low three bytes are not `REV` or `REv`; Eden-NX2 reads
    // `0x3900_0000` as (0x3900_0000 - 0x3056_4552) >> 24 = 8 (module docs).
    assert_eq!(revision_number(0x3900_0000), 8);
    let (rev_8, size) = GOLDEN[8];
    let untagged = RendererParameter {
        revision: 0x3900_0000,
        ..rev_8
    };
    assert_eq!(work_buffer_size(&untagged), Ok(size));
}

#[test]
fn revision_number_reads_numbers_and_tags_as_eden_nx2_does() {
    // Hand-worked from `GetRevisionNum`: 'R' 0x52, 'E' 0x45, 'V' 0x56, 'v' 0x76, '0' 0x30.
    assert_eq!(REVISION_MAGIC, 0x3056_4552);
    let cases = [
        (0, 0),
        (15, 15),
        // Below the tag's start every word is its own number.
        (0xFF, 0xFF),
        // The tag's start: 0x100 - 0x3056_4552 wraps to 0xCFA9_BBAE.
        (0x100, 0xCF),
        (0x3056_4552, 0),
        (0x3F56_4552, 15),
        (0x3F76_4552, 15),
        (0x4056_4552, 16),
        (0x4076_4552, 16),
        // The top byte below '0': wraps.
        (0x2F56_4552, 0xFF),
    ];
    for (word, number) in cases {
        assert_eq!(revision_number(word), number, "{word:#x}");
    }
    assert_eq!(
        (REVISION_TAG_FROM, REVISION_TAG_SHIFT, CURRENT_REVISION),
        (0x100, 24, 15)
    );
}

#[test]
fn a_negative_splitter_destination_count_is_refused_only_where_a_term_reads_it() {
    // Below the splitter no term reads it: Eden's size, the same as with no destinations.
    let unread = only(Feature::Splitter.since() - 1, |p| {
        p.splitter_destinations = -1
    });
    assert_eq!(work_buffer_size(&unread), Ok(102_400));
    let none = only(Feature::Splitter.since() - 1, |_| {});
    assert_eq!(work_buffer_size(&none), Ok(102_400));
    // The splitter reads it, and from revision 5 the command buffer too.
    for revision in [
        Feature::Splitter.since(),
        Feature::VariadicCommandBufferSize.since(),
        CURRENT_REVISION,
    ] {
        for count in [-1, i32::MIN] {
            let read = only(revision, |p| p.splitter_destinations = count);
            assert_eq!(
                work_buffer_size(&read),
                Err(WorkBufferError::NegativeSplitterDestinations(count)),
                "revision {revision}, {count}"
            );
        }
        // Zero, the edge, is a size.
        assert!(work_buffer_size(&only(revision, |_| {})).is_ok());
    }
}

#[test]
fn the_command_buffer_alone_refuses_a_negative_destination_count() {
    // `CalculateCommandBufferSize` reads the count too, after the splitter: with the splitter's
    // read in place a wrong command-buffer read could not be seen, so this asks the command
    // buffer directly.
    let params = only(CURRENT_REVISION, |p| p.splitter_destinations = -2);
    assert_eq!(
        command_buffer_size(&params, CURRENT_REVISION),
        Err(WorkBufferError::NegativeSplitterDestinations(-2))
    );
    assert_eq!(splitter_size(&params, Feature::Splitter.since() - 1), Ok(0));
}

#[test]
fn faults_are_reported_in_edens_order() {
    let splitter = Feature::Splitter.since();
    // `sub_mixes * MAX_EFFECTS` comes before the splitter: the overflow.
    let overflow_first = only(splitter, |p| {
        p.sub_mixes = u32::MAX / MAX_EFFECTS + 1;
        p.splitter_destinations = -1;
    });
    assert_eq!(
        work_buffer_size(&overflow_first),
        Err(WorkBufferError::Overflow)
    );
    // The memory pools come after it: the negative count.
    let negative_first = only(splitter, |p| {
        p.voices = u32::MAX / MAX_WAVE_BUFFERS + 1;
        p.splitter_destinations = -1;
    });
    assert_eq!(
        work_buffer_size(&negative_first),
        Err(WorkBufferError::NegativeSplitterDestinations(-1))
    );
}

#[test]
fn the_features_start_at_edens_revisions() {
    // `CheckFeatureSupported`'s table: `{SupportTags::Splitter, 2}`, `{SplitterBugFix, 5}`,
    // `{AudioRendererVariadicCommandBufferSize, 5}`, `{PerformanceMetricsDataFormatVersion2,
    // 5}`, `{WaveBufferVer2, 8}`, `{EffectInfoVer2, 9}`.
    let table = [
        (Feature::Splitter, 2),
        (Feature::SplitterBugFix, 5),
        (Feature::VariadicCommandBufferSize, 5),
        (Feature::PerformanceMetricsVersion2, 5),
        (Feature::WaveBufferVersion2, 8),
        (Feature::EffectInfoVersion2, 9),
    ];
    for (feature, since) in table {
        assert_eq!(feature.since(), since, "{feature:?}");
        assert!(!feature.in_revision(since - 1), "{feature:?}");
        assert!(feature.in_revision(since), "{feature:?}");
        assert!(feature.in_revision(since + 1), "{feature:?}");
    }
}

#[test]
fn the_counts_are_edens() {
    // `common.h`, `performance_manager.h`, `audio_renderer_parameter.h`.
    assert_eq!(
        (
            MAX_EFFECTS,
            TARGET_SAMPLE_COUNT,
            MAX_CHANNELS,
            MAX_MIX_BUFFERS,
            MAX_WAVE_BUFFERS,
            MAX_BIQUAD_FILTERS,
            MAX_DETAIL_ENTRIES,
            PARAMETER_SIZE,
        ),
        (256, 240, 6, 24, 4, 2, 100, 0x34)
    );
}

#[test]
fn the_sizes_are_edens_measured_host_sizes() {
    // `sizeof` of each class, printed by a program compiled over Eden-NX2's headers (module
    // docs, "The sizes are Eden's host sizes").
    use sizes::*;
    let measured = [
        (WORD, 4),
        (U64, 8),
        (POINTER, 8),
        (MIX_INFO, 2368),
        (VOICE_INFO, 560),
        (VOICE_CHANNEL_RESOURCE, 200),
        (VOICE_STATE, 296),
        (BIT_ARRAY, 32),
        (SPLITTER_DESTINATION_DATA, 264),
        (SPLITTER_INFO, 32),
        (MEMORY_POOL_INFO, 32),
        (EFFECT_RESULT_STATE, 128),
        (UPSAMPLER_INFO, 616),
        (EFFECT_INFO_BASE, 1592),
        (SINK_INFO_BASE, 352),
        (PERFORMANCE_FRAME_HEADER_V1, 24),
        (PERFORMANCE_DETAIL_V1, 16),
        (PERFORMANCE_ENTRY_V1, 16),
        (PERFORMANCE_FRAME_HEADER_V2, 48),
        (PERFORMANCE_DETAIL_V2, 24),
        (PERFORMANCE_ENTRY_V2, 24),
        (ADPCM_V1_COMMAND, 288),
        (PCM_INT16_V1_COMMAND, 280),
        (PCM_FLOAT_V1_COMMAND, 280),
        (ADPCM_V2_COMMAND, 296),
        (PCM_INT16_V2_COMMAND, 280),
        (PCM_FLOAT_V2_COMMAND, 280),
        (BIQUAD_FILTER_COMMAND, 80),
        (VOLUME_RAMP_COMMAND, 40),
        (MIX_RAMP_GROUPED_COMMAND, 328),
        (DEPOP_PREPARE_COMMAND, 96),
        (DEPOP_FOR_MIX_BUFFERS_COMMAND, 48),
        (MIX_COMMAND, 40),
        (VOLUME_COMMAND, 40),
        (MIX_RAMP_COMMAND, 48),
        (DEVICE_SINK_COMMAND, 320),
        (CIRCULAR_BUFFER_SINK_COMMAND, 56),
        (PERFORMANCE_COMMAND, 64),
    ];
    for (index, (shipped, eden)) in measured.into_iter().enumerate() {
        assert_eq!(shipped, eden, "size {index}");
    }
}

#[test]
fn the_error_messages_are_plain_words() {
    assert_eq!(
        WorkBufferError::InvalidRevision {
            word: 0x4056_4552,
            number: 16
        }
        .to_string(),
        "revision word 0x40564552 reads as revision 16, past the newest, 15"
    );
    assert_eq!(
        WorkBufferError::Overflow.to_string(),
        "the work buffer's size does not fit Eden's arithmetic"
    );
    assert_eq!(
        WorkBufferError::NegativeSplitterDestinations(-1).to_string(),
        "the splitter destination count -1 is negative"
    );
}

/// The parameter's bytes as Eden lays them out, field by field in its order: what a title
/// sends.
fn encode(p: &RendererParameter) -> Vec<u8> {
    let mut bytes = Vec::new();
    for word in [
        p.sample_rate,
        p.sample_count,
        p.mixes,
        p.sub_mixes,
        p.voices,
        p.sinks,
        p.effects,
        p.perf_frames,
    ] {
        bytes.extend(word.to_le_bytes());
    }
    bytes.extend([
        p.voice_drop_enabled,
        p.unknown_21,
        p.rendering_device,
        p.execution_mode,
    ]);
    bytes.extend(p.splitter_infos.to_le_bytes());
    bytes.extend(p.splitter_destinations.to_le_bytes());
    bytes.extend(p.external_context_size.to_le_bytes());
    bytes.extend(p.revision.to_le_bytes());
    bytes
}

#[test]
fn a_hand_built_parameter_is_read_at_edens_offsets() {
    // Each field written at the offset `AudioRendererParameterInternal`'s comments give, with a
    // distinct value whose bytes above the low byte differ.
    let mut bytes = [0u8; PARAMETER_SIZE];
    let words: [(usize, u32); 12] = [
        (0x00, 0x0001_BB80),
        (0x04, 0x0102_00F0),
        (0x08, 0x0203_0003),
        (0x0C, 0x0304_0005),
        (0x10, 0x0405_0007),
        (0x14, 0x0506_0002),
        (0x18, 0x0607_000B),
        (0x1C, 0x0708_000D),
        (0x24, 0x0809_0011),
        (0x28, 0xFFFF_FFED),
        (0x2C, 0x0A0B_0017),
        (0x30, 0x3F56_4552),
    ];
    for (at, word) in words {
        bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes[0x20..0x24].copy_from_slice(&[0x01, 0x21, 0x02, 0x01]);
    let read = RendererParameter::read(&mut Reader::new(&bytes)).unwrap();
    assert_eq!(
        read,
        RendererParameter {
            sample_rate: 0x0001_BB80,
            sample_count: 0x0102_00F0,
            mixes: 0x0203_0003,
            sub_mixes: 0x0304_0005,
            voices: 0x0405_0007,
            sinks: 0x0506_0002,
            effects: 0x0607_000B,
            perf_frames: 0x0708_000D,
            voice_drop_enabled: 0x01,
            unknown_21: 0x21,
            rendering_device: 0x02,
            execution_mode: 0x01,
            splitter_infos: 0x0809_0011,
            splitter_destinations: -19,
            external_context_size: 0x0A0B_0017,
            revision: 0x3F56_4552,
        }
    );
}

#[test]
fn a_parameter_one_byte_short_is_refused() {
    let bytes = encode(&GOLDEN[0].0);
    assert_eq!(bytes.len(), PARAMETER_SIZE);
    let short = &bytes[..PARAMETER_SIZE - 1];
    assert_eq!(
        RendererParameter::read(&mut Reader::new(short)),
        Err(Fault::PayloadTruncated {
            offset: PARAMETER_SIZE - 4,
            needed: 4,
            len: PARAMETER_SIZE - 1,
        })
    );
    // Short of the last byte-wide field, the first read that passes the end is that byte's.
    let to_bytes = &bytes[..0x23];
    assert_eq!(
        RendererParameter::read(&mut Reader::new(to_bytes)),
        Err(Fault::PayloadTruncated {
            offset: 0x23,
            needed: 1,
            len: 0x23,
        })
    );
}

proptest! {
    #![proptest_config(Config {
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x4155_443A_5552_4E44),
        ..Config::default()
    })]

    /// Built, then read: the same parameter, whatever the fields hold. Random fields are
    /// distinct only with high likelihood; `a_hand_built_parameter_is_read_at_edens_offsets`
    /// gives each a distinct value for certain.
    #[test]
    fn a_built_parameter_reads_back(
        words in prop::array::uniform12(any::<u32>()),
        small in prop::array::uniform4(any::<u8>()),
    ) {
        let params = RendererParameter {
            sample_rate: words[0],
            sample_count: words[1],
            mixes: words[2],
            sub_mixes: words[3],
            voices: words[4],
            sinks: words[5],
            effects: words[6],
            perf_frames: words[7],
            voice_drop_enabled: small[0],
            unknown_21: small[1],
            rendering_device: small[2],
            execution_mode: small[3],
            splitter_infos: words[8],
            splitter_destinations: i32::from_le_bytes(words[9].to_le_bytes()),
            external_context_size: words[10],
            revision: words[11],
        };
        let bytes = encode(&params);
        let mut reader = Reader::new(&bytes);
        prop_assert_eq!(RendererParameter::read(&mut reader), Ok(params));
        prop_assert_eq!(reader.remaining(), 0);
    }
}
