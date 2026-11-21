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
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x0, 249_856),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x1, 249_856),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x2, 258_048),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x3, 258_048),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x4, 258_048),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x5, 262_144),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x6, 262_144),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x7, 262_144),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x8, 262_144),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x9, 266_240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0xa, 266_240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0xb, 266_240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0xc, 266_240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0xd, 266_240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0xe, 266_240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0xf, 266_240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x3f56_4552, 266_240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x3f76_4552, 266_240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x3056_4552, 249_856),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x3456_4552, 258_048),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x3576_4552, 262_144),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x3956_4552, 266_240),
    golden(0xf0, 0x3, 0x5, 0x7, 0x2, 0xb, 0xd, 0x11, 0x13, 0x3900_0000, 262_144),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x1, 211_259_392),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x2, 212_459_520),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x4, 212_459_520),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x5, 257_490_944),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x7, 257_490_944),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x8, 257_495_040),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0x9, 257_630_208),
    golden(0x3c0, 0x102, 0x203, 0x304, 0x105, 0x206, 0x107, 0x208, 0x109, 0xf, 257_630_208),
    golden(0x1234, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 212_992),
    golden(0x1234, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf, 147_456),
    golden(0x0, 0x1335, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 122_880),
    golden(0x0, 0x1335, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf, 53_248),
    golden(0x0, 0x0, 0x1436, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 50_683_904),
    golden(0x0, 0x0, 0x1436, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf, 161_583_104),
    golden(0x0, 0x0, 0x0, 0x1537, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 8_183_808),
    golden(0x0, 0x0, 0x0, 0x1537, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf, 12_939_264),
    golden(0x0, 0x0, 0x0, 0x0, 0x1638, 0x0, 0x0, 0x0, 0x0, 0x1, 38_371_328),
    golden(0x0, 0x0, 0x0, 0x0, 0x1638, 0x0, 0x0, 0x0, 0x0, 0xf, 40_488_960),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x1739, 0x0, 0x0, 0x0, 0x1, 9_756_672),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x1739, 0x0, 0x0, 0x0, 0xf, 21_057_536),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x183a, 0x0, 0x0, 0x1, 10_276_864),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x183a, 0x0, 0x0, 0xf, 15_368_192),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x193b, 0x0, 0x1, 102_400),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x193b, 0x0, 0xf, 241_664),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1a3c, 0x1, 102_400),
    golden(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1a3c, 0xf, 9_572_352),
    golden(0x4bf, 0x11, 0x13, 0x215, 0x0, 0x8, 0x4, 0x37, 0x1, 0x2, 1_572_864),
    golden(0xb5f, 0x1e, 0x37, 0x3f3, 0x0, 0x3e, 0x5, 0x10, 0xa3, 0xc, 7_180_288),
    golden(0x7ff, 0xc, 0x25, 0x7b, 0x1, 0x3b, 0x1, 0x3f, 0x48, 0x1, 1_347_584),
    golden(0x9bb, 0xd, 0x2f, 0x43, 0x0, 0x17, 0x2, 0x3, 0x9b, 0x3, 1_540_096),
    golden(0x275, 0x8, 0x8, 0x3f5, 0x5, 0x23, 0x3, 0x2f, 0x8d, 0x9, 7_700_480),
    golden(0x3f1, 0x37, 0x29, 0x3ef, 0x4, 0x0, 0x5, 0x8, 0x72, 0xe, 6_594_560),
    golden(0xdf6, 0x9, 0x2f, 0xe5, 0x7, 0x4, 0xd, 0x2a, 0x24, 0xf, 2_904_064),
    golden(0x572, 0x0, 0x39, 0x17d, 0x1, 0x1b, 0x3, 0x23, 0x1e, 0x0, 1_347_584),
    golden(0xc95, 0x13, 0xe, 0xf1, 0x3, 0x23, 0x3, 0x14, 0xa1, 0x6, 2_244_608),
    golden(0x3f3, 0xd, 0x34, 0x2b0, 0x2, 0xe, 0xf, 0x16, 0xd2, 0xd, 5_091_328),
    golden(0xca6, 0x2b, 0x19, 0x2fd, 0x5, 0x3, 0xa, 0x12, 0xd9, 0x2, 3_616_768),
    golden(0x369, 0x0, 0x2d, 0x8d, 0x3, 0x3c, 0xd, 0x27, 0x3, 0x9, 1_728_512),
    golden(0xcbe, 0x14, 0x5, 0x32b, 0x2, 0x28, 0x5, 0x7, 0xdf, 0x8, 3_633_152),
    golden(0xa07, 0x3f, 0x38, 0x356, 0x0, 0x9, 0x2, 0x18, 0x63, 0x8, 8_937_472),
    golden(0xfaf, 0x12, 0x4, 0x2e1, 0x5, 0x16, 0x8, 0x17, 0x7, 0xa, 4_304_896),
    golden(0x442, 0x2e, 0x2a, 0x291, 0x6, 0x1a, 0x0, 0x30, 0xe8, 0xf, 7_860_224),
    golden(0x8f, 0x29, 0xd, 0xc2, 0x2, 0x12, 0x7, 0x26, 0xe6, 0xe, 2_420_736),
    golden(0x8cc, 0x2f, 0x15, 0x101, 0x0, 0x12, 0xd, 0x30, 0xc5, 0x1, 2_236_416),
    golden(0x6b1, 0x1e, 0x11, 0xb2, 0x6, 0x1f, 0x7, 0x18, 0x4e, 0x7, 2_265_088),
    golden(0xc9b, 0x28, 0xf, 0xab, 0x2, 0x15, 0xc, 0x1e, 0x6e, 0x3, 1_896_448),
    golden(0x9aa, 0x34, 0x5, 0x3d6, 0x4, 0x1, 0xb, 0x12, 0x55, 0xa, 5_607_424),
    golden(0x6e2, 0x3c, 0x2c, 0x2d1, 0x0, 0x7, 0x2, 0x3d, 0x98, 0x7, 9_711_616),
    golden(0x54d, 0x2f, 0x1e, 0x90, 0x7, 0x3c, 0x7, 0x37, 0x4f, 0x9, 3_837_952),
    golden(0x516, 0xf, 0x3e, 0x3f9, 0x7, 0xc, 0x2, 0x11, 0x36, 0x3, 3_481_600),
    golden(0xdae, 0x13, 0xf, 0x286, 0x6, 0x15, 0xf, 0x2b, 0xce, 0xa, 5_832_704),
    golden(0xee9, 0x6, 0x1b, 0x3e6, 0x1, 0x9, 0x0, 0x1e, 0x7c, 0xa, 6_111_232),
    golden(0x321, 0x25, 0xd, 0x29c, 0x2, 0xf, 0x8, 0x13, 0xa8, 0x0, 2_048_000),
    golden(0xa51, 0x3, 0x16, 0x19d, 0x4, 0x21, 0x0, 0x29, 0xde, 0x0, 1_183_744),
    golden(0x63a, 0x28, 0xa, 0x2f4, 0x1, 0x2e, 0x4, 0x12, 0xf0, 0x0, 2_195_456),
    golden(0xf81, 0x2c, 0x3, 0x360, 0x4, 0x38, 0x6, 0x35, 0x11, 0xa, 8_028_160),
    golden(0xbce, 0x20, 0x3c, 0x2f4, 0x6, 0x36, 0xa, 0xc, 0xfa, 0xc, 6_643_712),
    golden(0xced, 0x25, 0x36, 0x2b5, 0x3, 0x33, 0x7, 0x6, 0x8, 0xa, 5_607_424),
    golden(0xf0, 0x11, 0x16, 0x33f, 0x5, 0x2c, 0xb, 0x25, 0x5a, 0xa, 6_254_592),
    golden(0xb4a, 0x10, 0x4, 0x60, 0x7, 0x35, 0x6, 0x20, 0xf7, 0x6, 1_638_400),
    golden(0xef4, 0x0, 0xc, 0x1b9, 0x2, 0x13, 0xc, 0x2, 0xe5, 0xd, 1_966_080),
    golden(0x18b, 0x13, 0x0, 0x63, 0x3, 0x22, 0x7, 0x8, 0x52, 0x2, 471_040),
    golden(0xf6e, 0x30, 0x13, 0x187, 0x2, 0x2f, 0x2, 0x5, 0xae, 0xa, 3_629_056),
    golden(0x8e4, 0x11, 0x3e, 0x204, 0x4, 0x1e, 0xf, 0x1, 0xce, 0x0, 3_022_848),
    golden(0x33, 0x25, 0x1e, 0x30c, 0x0, 0x5, 0x0, 0x14, 0x9, 0x3, 2_646_016),
    golden(0xc9, 0x2e, 0x13, 0x3c5, 0x4, 0xd, 0x0, 0xf, 0x5d, 0x6, 5_160_960),
    golden(0x756, 0x23, 0x21, 0x3ec, 0x0, 0x17, 0xb, 0x13, 0x8b, 0xe, 6_615_040),
    golden(0x2ef, 0xf, 0x11, 0x36d, 0x1, 0x14, 0x0, 0x36, 0x36, 0x7, 7_299_072),
    golden(0x72b, 0x2f, 0x10, 0x30b, 0x7, 0x2b, 0x5, 0x24, 0x6, 0xe, 6_512_640),
    golden(0x749, 0x17, 0xe, 0x1fd, 0x6, 0x16, 0x2, 0x3b, 0xca, 0x7, 5_369_856),
    golden(0x247, 0xc, 0x20, 0x8d, 0x7, 0x4, 0x4, 0x2d, 0x90, 0x8, 2_093_056),
    golden(0x80b, 0x2a, 0x1d, 0x3bb, 0x7, 0x38, 0xb, 0x21, 0xce, 0xb, 8_339_456),
    golden(0xf7d, 0x3e, 0x4, 0x1d7, 0x6, 0x27, 0xb, 0x11, 0x7d, 0xe, 4_173_824),
    golden(0x5a9, 0x2c, 0x13, 0x2a, 0x1, 0x1c, 0xe, 0x2c, 0x37, 0x5, 1_884_160),
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
                p.effects = max;
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
    // Not at each term's exact edge: one parameter for each of the three 64-bit products that
    // can wrap in Eden (module docs, "Where this differs from Eden").
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
    // The command buffer's `voices * (splitter_infos * sizeof(DepopPrepareCommand) +
    // voice_size)`: (2^30 - 1) * (96 * u32::MAX + 816) is about 2^68.6, while `voices * 4` still
    // fits 32 bits. Eden-NX2's function answers the wrapped size, 2_164_663_549_952.
    let command_buffer = only(Feature::VariadicCommandBufferSize.since(), |p| {
        p.voices = 0x3FFF_FFFF;
        p.splitter_infos = u32::MAX;
    });
    assert_eq!(
        work_buffer_size(&command_buffer),
        Err(WorkBufferError::Overflow)
    );
}

#[test]
fn a_running_sum_past_64_bits_is_refused() {
    // No product wraps here: the mix buffers' `(sample_count * 4) * (mixes + MAX_CHANNELS)` is
    // (2^33 - 4) * 2^31 = 2^64 - 2^33, but the mix-id term before it, `AlignUp(mixes * 4,
    // 0x40)`, is 2^33, so the running sum passes 2^64. Eden answers the wrapped size, 102_400.
    let params = only(1, |p| {
        p.mixes = (1 << 31) - MAX_CHANNELS;
        p.sample_count = (1 << 31) - 1;
    });
    assert_eq!(work_buffer_size(&params), Err(WorkBufferError::Overflow));
}

#[test]
fn the_graph_terms_are_edens() {
    // `NodeStates::GetWorkBufferSize(count)`: `(AlignUp(count, 0x40) / 8) * 2 + count * 32 +
    // count * (count * 4)`. 3: 16 + 96 + 36. 65: 32 + 2080 + 16900.
    assert_eq!(node_states_size(3), Ok(148));
    assert_eq!(node_states_size(65), Ok(19_012));
    // `EdgeMatrix::GetWorkBufferSize(count)`: `AlignUp(count * count, 0x40) / 8`. 3: 64 / 8.
    // 65: 4288 / 8.
    assert_eq!(edge_matrix_size(3), Ok(8));
    assert_eq!(edge_matrix_size(65), Ok(536));
}

#[test]
fn the_size_before_its_page_alignment_is_edens() {
    // The last `AlignUp(size, 0x1000)` hides inner alignments, so this checks the sum before
    // it. Revision 2, two sub-mixes (`count` 3) and one voice, hand-worked from
    // `System::GetWorkBufferSize`: 0 + 2048 + 7104 + 1056 + 32 + 16 + 11520, the graph
    // `AlignUp(148 + 8, 0x10)` = 160, 0 for the splitter, 128 for the memory pools and 0x50:
    // 22144, already a multiple of 0x40. Then 1232 (upsamplers), `AlignUp(296, 0x40)` = 320 and
    // 0x18000 + 126: 122126. The first part's other terms sum to 32 modulo 0x40, so the graph
    // aligned to 0x40 instead (192) would carry through its `AlignUp(size, 0x40)` as 64 more.
    let revision = Feature::Splitter.since();
    let params = only(revision, |p| {
        p.sub_mixes = 2;
        p.voices = 1;
    });
    assert_eq!(size_before_page_alignment(&params, revision), Ok(122_126));
    assert_eq!(work_buffer_size(&params), Ok(122_880));
}

/// A revision word from its four characters, low byte first (`Common::MakeMagic`).
const fn tag(chars: [u8; 4]) -> u32 {
    u32::from_le_bytes(chars)
}

#[test]
fn a_revision_past_the_bound_is_refused_before_any_term() {
    for (word, number) in [
        (CURRENT_REVISION + 1, 16),
        (0xFF, 0xFF),
        // Not tagged native: Eden-NX2's bound, 15.
        (tag(*b"REV@"), 16),
        (REVISION_TAG_FROM, 0xCF),
        // Tagged native: PR #4588's bound, 16.
        (tag(*b"REvA"), 17),
        (tag(*b"REv\xFF"), 0xCF),
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
    // A native word whose top byte is below '0' names no revision, and is refused first too.
    let word = tag(*b"REv/");
    let both = only(word, |p| {
        p.sub_mixes = u32::MAX;
        p.splitter_destinations = -1;
    });
    assert_eq!(
        work_buffer_size(&both),
        Err(WorkBufferError::NoRevision { word })
    );
    // The newest revisions are accepted: 15 as number and as either tag, 16 as native tag.
    for word in [
        CURRENT_REVISION,
        tag(*b"REV?"),
        tag(*b"REv?"),
        tag(*b"REv@"),
    ] {
        assert!(work_buffer_size(&only(word, |_| {})).is_ok(), "{word:#x}");
    }
}

#[test]
fn the_revision_bound_depends_on_the_native_tag() {
    // Each pair differs only at the edge, and the parameter wraps `sub_mixes * MAX_EFFECTS`: an
    // accepted word reaches that term and is `Overflow`, a refused one is refused by its
    // revision.
    let wraps = |word| {
        work_buffer_size(&only(word, |p| {
            p.sub_mixes = u32::MAX / MAX_EFFECTS + 1;
        }))
    };
    let overflow = Err(WorkBufferError::Overflow);
    // Native: 16 (`REv@`) is accepted, 17 (`REvA`) refused.
    assert_eq!(wraps(tag(*b"REv@")), overflow);
    assert_eq!(
        wraps(tag(*b"REvA")),
        Err(WorkBufferError::InvalidRevision {
            word: tag(*b"REvA"),
            number: 17
        })
    );
    // Not native: 15 (`REV?`) is accepted, 16 (`REV@`) refused under Eden-NX2's bound.
    assert_eq!(wraps(tag(*b"REV?")), overflow);
    assert_eq!(
        wraps(tag(*b"REV@")),
        Err(WorkBufferError::InvalidRevision {
            word: tag(*b"REV@"),
            number: 16
        })
    );
    // Native, at the bottom: '0' is revision 0, accepted; '/' is below '0' and names none.
    assert_eq!(wraps(tag(*b"REv0")), overflow);
    assert_eq!(
        wraps(tag(*b"REv/")),
        Err(WorkBufferError::NoRevision {
            word: tag(*b"REv/")
        })
    );
    // A plain number: 15 accepted, 16 refused.
    assert_eq!(wraps(CURRENT_REVISION), overflow);
    assert_eq!(
        wraps(CURRENT_REVISION + 1),
        Err(WorkBufferError::InvalidRevision {
            word: CURRENT_REVISION + 1,
            number: 16
        })
    );
}

#[test]
fn a_native_revision_16_is_sized_as_revision_15() {
    // No feature in `CheckFeatureSupported`'s table needs more than 15, so `REv@` turns on
    // exactly revision 15's: every golden parameter accepted as revision 15 has the same size
    // at `REv@`.
    let native_16 = tag(*b"REv@");
    assert_eq!(revision_number(native_16), Some(NATIVE_CURRENT_REVISION));
    let mut compared = 0;
    for (params, size) in GOLDEN {
        if revision_number(params.revision) != Some(CURRENT_REVISION) {
            continue;
        }
        let native = RendererParameter {
            revision: native_16,
            ..*params
        };
        assert_eq!(work_buffer_size(&native), Ok(*size), "{params:?}");
        compared += 1;
    }
    // Counted by hand in `GOLDEN`: row 15, the `REV?` and `REv?` rows, the row of fields with
    // high bytes at 15, the nine fields alone at 15, and two rows of the sequence.
    assert_eq!(compared, 15);
    for feature in [
        Feature::Splitter,
        Feature::SplitterBugFix,
        Feature::VariadicCommandBufferSize,
        Feature::PerformanceMetricsVersion2,
        Feature::WaveBufferVersion2,
        Feature::EffectInfoVersion2,
    ] {
        assert_eq!(
            feature.in_revision(NATIVE_CURRENT_REVISION),
            feature.in_revision(CURRENT_REVISION),
            "{feature:?}"
        );
    }
}

#[test]
fn pr_4588s_own_revision_cases_are_answered_as_it_answers_them() {
    // `eden-pr4588/src/tests/core/audio_revision.cpp` @ e5fb3b97: `GetRevisionNum` and
    // `CheckValidRevision` of each word, written out as its literals.
    assert_eq!(revision_number(0x3f56_4552), Some(15));
    assert_eq!(revision_number(0x4076_4552), Some(16));
    for word in [0x3f56_4552, 0x4076_4552] {
        assert!(work_buffer_size(&only(word, |_| {})).is_ok(), "{word:#x}");
    }
    // `0x41764552` (`REvA`) and `0x40564552` (`REV@`) are past their bounds, and `0x40764553`
    // carries no tag, so Eden-NX2's reading applies and reads it as 16, also past the bound.
    for (word, number) in [(0x4176_4552, 17), (0x4056_4552, 16), (0x4076_4553, 16)] {
        assert_eq!(
            work_buffer_size(&only(word, |_| {})),
            Err(WorkBufferError::InvalidRevision { word, number }),
            "{word:#x}"
        );
    }
    // `0x2f764552` (`REv/`) names no revision.
    assert_eq!(
        work_buffer_size(&only(0x2f76_4552, |_| {})),
        Err(WorkBufferError::NoRevision { word: 0x2f76_4552 })
    );
}

#[test]
fn a_word_without_the_rev_tag_is_read_as_eden_nx2_reads_it() {
    // PR #4588 refuses a word whose low three bytes are not `REV` or `REv`; Eden-NX2 reads
    // `0x3900_0000` as (0x3900_0000 - 0x3056_4552) >> 24 = 8 (module docs).
    assert_eq!(revision_number(0x3900_0000), Some(8));
    let (rev_8, size) = GOLDEN[8];
    let untagged = RendererParameter {
        revision: 0x3900_0000,
        ..rev_8
    };
    assert_eq!(work_buffer_size(&untagged), Ok(size));
}

#[test]
fn revision_number_reads_each_word_as_its_tag_says() {
    // Hand-worked from both `GetRevisionNum`s: 'R' 0x52, 'E' 0x45, 'V' 0x56, 'v' 0x76, '0' 0x30.
    assert_eq!(REVISION_MAGIC, 0x3056_4552);
    let cases = [
        (0, Some(0)),
        (15, Some(15)),
        // Below the tag's start every word is its own number.
        (0xFF, Some(0xFF)),
        // Eden-NX2's reading. The tag's start: 0x100 - 0x3056_4552 wraps to 0xCFA9_BBAE.
        (0x100, Some(0xCF)),
        (0x3056_4552, Some(0)),
        (0x3F56_4552, Some(15)),
        (0x4056_4552, Some(16)),
        // The top byte below '0': wraps.
        (0x2F56_4552, Some(0xFF)),
        // PR #4588's reading of a native word: the top byte less '0', none below '0'.
        (0x3076_4552, Some(0)),
        (0x3F76_4552, Some(15)),
        (0x4076_4552, Some(16)),
        (0x4176_4552, Some(17)),
        (0xFF76_4552, Some(0xCF)),
        (0x2F76_4552, None),
        (0x0076_4552, None),
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
fn the_native_revision_constants_are_pr_4588s() {
    // `feature_support.h` @ e5fb3b97: `NativeCurrentRevision = 16`, `IsNativeRevision`'s
    // `(revision & 0xffffff) == Common::MakeMagic('R', 'E', 'v', '\0')`, and `GetRevisionNum`'s
    // `encoded - '0'`.
    assert_eq!(
        (
            NATIVE_CURRENT_REVISION,
            REVISION_TAG_MASK,
            NATIVE_REVISION_TAG,
            REVISION_DIGIT_ZERO
        ),
        (16, 0x00FF_FFFF, 0x0076_4552, 0x30)
    );
    assert_eq!(newest_revision(tag(*b"REv@")), 16);
    assert_eq!(newest_revision(tag(*b"REV@")), 15);
    assert_eq!(newest_revision(16), 15);
    // Only the low three bytes are compared: the top byte never matters, and each of the three
    // matters on its own.
    for top in [0x00, 0x30, 0x40, 0xFF] {
        assert!(is_native_revision(0x0076_4552 | (top << 24)), "{top:#x}");
    }
    for word in [
        0x4076_4553,
        0x4076_4452,
        0x4077_4552,
        0x4056_4552,
        0x4076_4500,
    ] {
        assert!(!is_native_revision(word), "{word:#x}");
    }
}

/// Eden-NX2's `GetRevisionNum` and `CheckValidRevision`
/// (`eden-nx2/src/audio_core/common/feature_support.h` @ 50fa86bb), transcribed with their
/// literals: the number and whether it is accepted.
const fn eden_nx2_reading(word: u32) -> (u32, bool) {
    let number = if word >= 0x100 {
        word.wrapping_sub(0x3056_4552) >> 24
    } else {
        word
    };
    (number, number <= 15)
}

/// PR #4588's `GetRevisionNum` and `CheckValidRevision`
/// (`eden-pr4588/src/audio_core/common/feature_support.h` @ e5fb3b97), transcribed with their
/// literals.
const fn pr_4588_reading(word: u32) -> (u32, bool) {
    let native = word & 0xFF_FFFF == 0x76_4552;
    let number = if word < 0x100 {
        word
    } else if !native && word & 0xFF_FFFF != 0x56_4552 {
        u32::MAX
    } else if word >> 24 >= 0x30 {
        (word >> 24) - 0x30
    } else {
        u32::MAX
    };
    (number, number <= if native { 16 } else { 15 })
}

/// The words the union test walks, each once: every word below 0x1000, and every top byte over
/// low three bytes of `REV`, `REv`, each of those with one of its 24 bits flipped, zero and all
/// ones. (`REV` with bit 21 flipped is `REv`, hence the dedup.)
fn union_words() -> Vec<u32> {
    let mut lows = vec![0x56_4552_u32, 0x76_4552, 0, 0xFF_FFFF];
    for bit in 0..24 {
        lows.push(0x56_4552 ^ (1 << bit));
        lows.push(0x76_4552 ^ (1 << bit));
    }
    let mut words: Vec<u32> = (0..0x1000).collect();
    for top in 0..=0xFF_u32 {
        for low in &lows {
            words.push((top << 24) | low);
        }
    }
    words.sort_unstable();
    words.dedup();
    words
}

#[test]
fn the_accepted_words_are_the_union_of_both_readings() {
    // Over `union_words`, a word is accepted exactly when either reference accepts it, at the
    // number the accepting reference reads; where both accept they read the same number.
    let (mut both, mut only_nx2, mut only_pr) = (0, 0, 0);
    for word in union_words() {
        let (nx2_number, nx2) = eden_nx2_reading(word);
        let (pr_number, pr) = pr_4588_reading(word);
        let result = work_buffer_size(&only(word, |_| {}));
        assert_eq!(result.is_ok(), nx2 || pr, "{word:#x}: {result:?}");
        match (nx2, pr) {
            (true, true) => {
                assert_eq!(nx2_number, pr_number, "{word:#x}");
                both += 1;
            }
            (true, false) => only_nx2 += 1,
            (false, true) => only_pr += 1,
            (false, false) => continue,
        }
        let number = if pr { pr_number } else { nx2_number };
        assert_eq!(revision_number(word), Some(number), "{word:#x}");
    }
    // Each class is reached: words both accept; words only Eden-NX2 accepts (untagged ones of
    // 0x100 or more); and `REv@`, the one word only PR #4588 accepts.
    assert_eq!(only_pr, 1, "{both} {only_nx2} {only_pr}");
    assert!(both > 0 && only_nx2 > 0, "{both} {only_nx2} {only_pr}");
}

#[test]
fn a_negative_splitter_destination_count_is_refused_only_where_a_term_reads_it() {
    // Below the splitter no term reads it: Eden's size, the same as with no destinations.
    let unread = only(Feature::Splitter.since() - 1, |p| {
        p.splitter_destinations = -1;
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
    // A native word names PR #4588's bound instead.
    assert_eq!(
        WorkBufferError::InvalidRevision {
            word: 0x4176_4552,
            number: 17
        }
        .to_string(),
        "revision word 0x41764552 reads as revision 17, past the newest, 16"
    );
    assert_eq!(
        WorkBufferError::NoRevision { word: 0x2f76_4552 }.to_string(),
        "revision word 0x2f764552 is tagged native but names no revision"
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
    bytes[0x20..0x24].copy_from_slice(&[0x01, 0x21, 0x02, 0x03]);
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
            execution_mode: 0x03,
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
