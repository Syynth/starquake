//! PR #4588's cases for this group (`eden-pr4588/src/tests/video_core/native_shader.cpp` @
//! e5fb3b97), rebuilt from their words: each word is decoded and translated here, and run
//! through the interpreter with the registers, calls and results the PR's case checks through
//! its lowering. The words are synthetic (the PR's own encodings of the forms it tests).
//!
//! Checks a PR case makes that belong to its lowering (`lower_to_maxwell.h`), not to the
//! translator (`ops_texture.cpp`), are left out, each noted where it is: an odd destination
//! pair and a constant-buffer offset past what Maxwell's `TEXS` holds, a gather offset, a
//! `TLD` source at an odd register, a cube `TMML`, and `TMML` of both channels at once.

use super::*;

/// The word of a case's four 32-bit words, the first lowest.
fn word(words: [u32; 4]) -> Instruction {
    let [a, b, c, d] = words.map(u128::from);
    Instruction::from_bits(a | b << 32 | c << 64 | d << 96)
}

/// The instruction a case's words decode to.
fn instr(words: [u32; 4]) -> Instr {
    decode(word(words)).expect("decodes")
}

/// Translates a case's words, checks the descriptor buffer it leaves, and returns the function.
fn translate_case(words: [u32; 4], buffer: Option<u8>) -> Function {
    let (function, after) =
        translate_raw(instr(words), DescriptorBuffer::new()).expect("translates");
    assert_eq!(function.verify(), Ok(()));
    assert_eq!(after.index(), buffer);
    function
}

const X: u32 = f(0.375);
const Y: u32 = f(-1.5);
const Z: u32 = f(2.25);

/// "Native explicit-LOD 2D samples retain packed signed offsets and aliased outputs": `@!P0
/// TEX.LL.AOFFI` with the coordinates at R8, the level at R36 and the offsets at R37, every
/// mask, results at R8 or R36 (over the sources) and R40.
#[test]
fn explicit_lod_samples_with_offsets() {
    let level = f(1.25);
    // x = 0xe (-2), y = 0x3.
    let packed = 0x0000_003e;
    for destination in [8_u32, 36] {
        for bits in 1..16_u32 {
            let count = bits.count_ones();
            let second = if count > 2 { 40 } else { 255 };
            let words = [
                0x0800_8b60 | (destination << 16),
                0x3080_0824,
                0x019e_1000 | (bits << 8) | second,
                0,
            ];
            let function = translate_case(words, Some(2));
            let mut machine = machine_with(&[(8, X), (9, Y), (36, level), (37, packed)]);
            run_script(
                &function,
                &mut machine,
                vec![call(
                    Opcode::ImageSampleExplicitLod,
                    info(TextureType::Color2D),
                    vec![
                        Val::U32(32),
                        Val::F32x2([X, Y]),
                        Val::F32(level),
                        Val::U32x2([0xffff_fffe, 3]),
                    ],
                    Some(Val::F32x4(TEXEL)),
                )],
            );
            for (reg, value) in placed(bits, destination, second) {
                assert_eq!(
                    machine.regs[reg], value,
                    "mask {bits:#x}, destination {destination}"
                );
            }
        }
    }
}

/// "Native explicit-LOD offset samples reject unimplemented layouts": each change the
/// translator refuses, with the error it gives. The PR's mode 2 (an odd destination pair) and
/// mode 9 (an offset past the lowering's `TEXS` field) are its lowering's checks.
#[test]
fn offset_layouts_the_translator_refuses() {
    let valid = [0x0024_7b60, 0x3080_0824, 0x019e_13ff, 0];
    translate_case(valid, Some(2));
    let cases: [(usize, Option<u32>, u32, TextureUnsupported); 8] = [
        (0, None, 1 << 24, TextureUnsupported::Offset),
        (1, None, 1, TextureUnsupported::Offset),
        (2, None, 0x4000, TextureUnsupported::Offset),
        (2, None, 0x8000, TextureUnsupported::Offset),
        (2, None, 0x2000, TextureUnsupported::Derivative),
        (2, Some(0x019e_17ff), 0, TextureUnsupported::ResultLayout),
        (2, Some(0x019e_10ff), 0, TextureUnsupported::EmptyMask),
        (
            1,
            Some((0x3080_0824 & !0x0fc0_0000) | (18 << 22)),
            0,
            TextureUnsupported::DescriptorBuffer(18),
        ),
    ];
    for (index, replace, flip, what) in cases {
        let mut words = valid;
        words[index] = replace.unwrap_or(words[index]) | flip;
        assert_refused_raw(instr(words), DescriptorBuffer::new(), what);
    }
}

/// "Native volume samples pack exactly the selected channels without clobbering aliases":
/// `@!P0 TEX.LZ` of a 3D texture, coordinates at R32, R33 and R31, every mask, results from
/// R30 or R31 (over a source) and R40.
#[test]
fn volume_samples_with_every_mask() {
    for bits in [1_u32, 2, 4, 8, 3, 9, 10, 12, 7, 11, 13, 14, 15] {
        let count = bits.count_ones();
        let destination = if count == 1 { 31 } else { 30 };
        let second = if count > 2 { 40 } else { 255 };
        let words = [
            0x2000_8b60 | (destination << 16),
            0x5000_101f,
            0x049e_0000 | (bits << 8) | second,
            0,
        ];
        let function = translate_case(words, Some(0));
        let mut machine = machine_with(&[(32, X), (33, Y), (31, Z)]);
        run_script(
            &function,
            &mut machine,
            vec![call(
                Opcode::ImageSampleExplicitLod,
                info(TextureType::Color3D),
                vec![Val::U32(64), Val::F32x3([X, Y, Z]), Val::F32(0), NONE],
                Some(Val::F32x4(TEXEL)),
            )],
        );
        for (reg, value) in placed(bits, destination, second) {
            assert_eq!(machine.regs[reg], value, "mask {bits:#x}");
        }
    }
}

/// "Native half texture samples pack RG and BA into separate registers": a 2D array sample
/// with and without `.F16`, the array index at R6 and the coordinates at R7 and R4.
#[test]
fn half_samples_pack_pairs() {
    for half in [false, true] {
        let words = [
            0x0604_7b60,
            0xb080_0804,
            if half { 0x001e_8f06 } else { 0x001e_0f06 },
            0x101f_4400,
        ];
        let function = translate_case(words, Some(2));
        let mut machine = machine_with(&[(6, 0x0001_0002), (7, X), (4, Y)]);
        run_script(
            &function,
            &mut machine,
            vec![call(
                Opcode::ImageSampleImplicitLod,
                TextureInfo {
                    relaxed_precision: half,
                    ..info(TextureType::ColorArray2D)
                },
                vec![Val::U32(32), Val::F32x3([X, Y, f(2.0)]), NONE, NONE],
                Some(Val::F32x4(TEXEL)),
            )],
        );
        if half {
            assert_eq!(
                [machine.regs[4], machine.regs[6]],
                [0x3400_3000, 0x3a00_3800]
            );
            assert_eq!([machine.regs[5], machine.regs[7]], [0, X]);
        } else {
            assert_eq!(machine.regs[4..8], TEXEL);
        }
    }
}

/// "Native texture gathers preserve coordinates component and split result pairs": `TLD4` in
/// buffers 0, 5 and 17, every component, coordinates at R4 and R3, results to R4 to R7. (The
/// PR's refusal of a gather offset is its lowering's.)
#[test]
fn gathers_of_every_component() {
    for binding in [0_u32, 5, 17] {
        for component in 0..4_u32 {
            let words = [
                0x0404_7b63,
                0x3000_0703 | (binding << 22),
                0x041e_0f06 | (component << 23),
                0,
            ];
            let function = translate_case(words, Some(u8::try_from(binding).expect("small")));
            let mut machine = machine_with(&[(4, X), (3, Y)]);
            run_script(
                &function,
                &mut machine,
                vec![call(
                    Opcode::ImageGather,
                    TextureInfo {
                        gather_component: u8::try_from(component).expect("small"),
                        ..info(TextureType::Color2D)
                    },
                    vec![Val::U32(28), Val::F32x2([X, Y]), NONE, NONE],
                    Some(Val::F32x4(TEXEL)),
                )],
            );
            assert_eq!(machine.regs[4..8], TEXEL);
        }
    }
}

/// "Native volume texel fetch preserves XYZ LOD descriptor and split RGBA outputs": `TLD.LZ`
/// of a 3D texture, coordinates at R0, R1 and R2, results to R8 to R11. (The PR's refusal of a
/// source at R1 is its lowering's.)
#[test]
fn volume_texel_fetch() {
    let function = translate_case([0x0008_7b66, 0x5000_0a02, 0x049e_0f0a, 0], Some(0));
    let mut machine = machine_with(&[(0, 5), (1, 6), (2, 7)]);
    run_script(
        &function,
        &mut machine,
        vec![call(
            Opcode::ImageFetch,
            info(TextureType::Color3D),
            vec![Val::U32(40), Val::U32x3([5, 6, 7]), NONE, Val::U32(0), NONE],
            Some(Val::F32x4(TEXEL)),
        )],
    );
    assert_eq!(machine.regs[8..12], TEXEL);
}

/// "Native array gradient fetch retains array index XY derivatives descriptor and RGBA": `TXD`
/// of a 2D array, coordinates at R4 and R5, the array index at R6, derivatives at R12 to R15,
/// results to R4 to R7.
#[test]
fn array_gradient_sample() {
    let function = translate_case([0x0404_7b6c, 0xa080_080c, 0x001e_0f06, 0], Some(2));
    let derivatives = [f(0.5), f(-0.5), f(0.25), f(-0.25)];
    let mut machine = machine_with(&[
        (4, X),
        (5, Y),
        (6, 0x0009_0001),
        (12, derivatives[0]),
        (13, derivatives[1]),
        (14, derivatives[2]),
        (15, derivatives[3]),
    ]);
    run_script(
        &function,
        &mut machine,
        vec![call(
            Opcode::ImageGradient,
            TextureInfo {
                num_derivatives: 2,
                ..info(TextureType::ColorArray2D)
            },
            vec![
                Val::U32(32),
                Val::F32x3([X, Y, f(1.0)]),
                Val::F32x4(derivatives),
                NONE,
                NONE,
            ],
            Some(Val::F32x4(TEXEL)),
        )],
    );
    assert_eq!(machine.regs[4..8], TEXEL);
}

/// The words of the PR's 2D `TMML` cases, with destination `destination` and mask `bits`.
const fn tmml_words(destination: u32, bits: u32) -> [u32; 4] {
    [
        0x0400_7b69 | (destination << 16),
        0x2080_08ff,
        0x0000_00ff | (bits << 8),
        0x100f_4200,
    ]
}

/// "Native 2D mipmap query preserves selected channel coordinates and descriptor" and
/// "Native mipmap query encodes signed relative and unsigned absolute fractional LOD": one
/// channel to the destination, the relative level signed and the absolute one unsigned, as
/// 8.8 fixed point.
#[test]
fn mipmap_queries() {
    let levels: [(f32, f32, u32, u32); 5] = [
        (2.75, -1.5, 0x02c0, 0xfe80),
        (0.0, -0.003_906_25, 0x0000, 0xffff),
        (0.003_906_25, 0.5, 0x0001, 0x0080),
        (4095.0 / 256.0, 8.25, 0x0fff, 0x0840),
        (3.25, 2.75, 0x0340, 0x02c0),
    ];
    for destination in [0_u32, 4, 20] {
        for bits in [1_u32, 2] {
            let function = translate_case(tmml_words(destination, bits), Some(2));
            for (absolute, relative, absolute_fixed, relative_fixed) in levels {
                let mut machine = machine_with(&[(4, X), (5, Y)]);
                run_script(
                    &function,
                    &mut machine,
                    vec![call(
                        Opcode::ImageQueryLod,
                        info(TextureType::Color2D),
                        vec![Val::U32(32), Val::F32x2([X, Y])],
                        Some(Val::F32x4([f(absolute), f(relative), 0, 0])),
                    )],
                );
                let expected = if bits == 1 {
                    relative_fixed
                } else {
                    absolute_fixed
                };
                assert_eq!(
                    machine.regs[usize::try_from(destination).expect("small")],
                    expected,
                    "{absolute}, {relative}, mask {bits}"
                );
            }
        }
    }
}

/// "Native mipmap queries reject unsupported modes and descriptor bank conflicts": the
/// changes the translator refuses, a conflict between two queries' buffers, and a guarded
/// query. Of the PR's flipped bits, 59, 60, 76 and 91 are bits the forms do not have (the
/// decoder refuses the word), and bit 62 makes a cube query and a mask of 3 queries both
/// channels, both refused by the PR's lowering, not its translator.
#[test]
fn mipmap_query_refusals() {
    let query = tmml_words(0, 2);
    for (index, bit) in [(1_usize, 27_u32), (1, 28), (2, 12), (2, 27)] {
        let mut words = query;
        words[index] ^= 1 << bit;
        assert!(decode(word(words)).is_err(), "word {index}, bit {bit}");
    }
    for index in [1_usize, 2] {
        let mut words = query;
        words[index] ^= 1;
        assert_refused_raw(
            instr(words),
            DescriptorBuffer::new(),
            TextureUnsupported::MipQueryExtraRegister,
        );
    }
    for bits in [0_u8, 4, 8, 15] {
        let mut words = query;
        words[2] = 0xff | (u32::from(bits) << 8);
        assert_refused_raw(
            instr(words),
            DescriptorBuffer::new(),
            TextureUnsupported::MipQueryChannels(mask(bits)),
        );
    }
    let mut other_bank = query;
    other_bank[1] ^= 1 << 22;
    let (_, buffer) = translate_raw(instr(query), DescriptorBuffer::new()).expect("translates");
    assert_refused_raw(
        instr(other_bank),
        buffer,
        TextureUnsupported::DescriptorBufferConflict { bound: 2, index: 3 },
    );
    let mut predicated = query;
    predicated[0] = (predicated[0] & !0xf000) | 0x9000;
    let guarded = instr(predicated);
    assert_eq!(
        guarded.guard,
        PredSrc {
            reg: p(1),
            negated: true
        }
    );
    translate_raw(guarded, DescriptorBuffer::new()).expect("translates");
}

/// "Native mip count queries preserve aliased input and scalar output": `TXQ` of the level in
/// its own destination, the level count only (not skipped), to the destination.
#[test]
fn mip_count_queries() {
    for destination in [0_u32, 3, 249] {
        let words = [
            0x0000_7b6f | (destination << 16) | (destination << 24),
            0x0080_0800,
            0x0000_08ff,
            0x100f_4200,
        ];
        let function = translate_case(words, Some(2));
        let index = usize::try_from(destination).expect("small");
        let mut machine = machine_with(&[(u8::try_from(destination).expect("small"), 7)]);
        run_script(
            &function,
            &mut machine,
            vec![call(
                Opcode::ImageQueryDimensions,
                TextureInfo::default(),
                vec![Val::U32(32), Val::U32(7), Val::U1(false)],
                Some(Val::U32x4([64, 32, 1, 11])),
            )],
        );
        assert_eq!(machine.regs[index], 11);
        let mut invalid = words;
        invalid[2] = 0x0000_0800;
        assert_refused_raw(
            instr(invalid),
            DescriptorBuffer::new(),
            TextureUnsupported::QueryExtraRegister,
        );
    }
}
