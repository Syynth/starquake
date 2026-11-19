//! `TEX`: every dimension in both operand layouts, every level-of-detail mode, depth
//! comparison, offsets, half-precision results, every channel mask, the three kinds of handle
//! and the fault predicate.

use super::*;

/// The handle of [`tex`]: the byte offset of its constant-buffer slot.
const HANDLE: Val = Val::U32(0x40);

/// An array index register: layer 3 in the low half, other bits above it that must be ignored.
const LAYER_BITS: u32 = 0x0007_0003;
/// [`LAYER_BITS`] as the array index the sample reads.
const LAYER: u32 = f(3.0);

/// Distinct coordinate values.
const X: u32 = f(0.375);
const Y: u32 = f(-1.5);
const Z: u32 = f(2.25);

/// Runs `op` with `regs` against one call of `opcode` with `info` and `args` that replies
/// [`TEXEL`], and checks the four channels reach R0 to R3.
fn sample_rgba(op: OpTex, regs: &[(u8, u32)], opcode: Opcode, info: TextureInfo, args: Vec<Val>) {
    let machine = run_op(
        Op::Tex(op),
        regs,
        vec![call(opcode, info, args, Some(Val::F32x4(TEXEL)))],
    );
    assert_eq!(machine.regs[..4], TEXEL, "{op:?}");
}

/// In the scalar layout two operands take one register each at `A` and `B`; more take `A`,
/// `A + 1`, then `B` onwards. The array index comes first, as a float of its low 16 bits.
#[test]
fn every_dimension_in_the_scalar_layout() {
    let cases: [(TexDim, TextureType, &[(u8, u32)], Val); 7] = [
        (TexDim::D1, TextureType::Color1D, &[(10, X)], Val::F32(X)),
        (
            TexDim::Array1D,
            TextureType::ColorArray1D,
            &[(10, LAYER_BITS), (20, X)],
            Val::F32x2([X, LAYER]),
        ),
        (
            TexDim::D2,
            TextureType::Color2D,
            &[(10, X), (20, Y)],
            Val::F32x2([X, Y]),
        ),
        (
            TexDim::Array2D,
            TextureType::ColorArray2D,
            &[(10, LAYER_BITS), (11, X), (20, Y)],
            Val::F32x3([X, Y, LAYER]),
        ),
        (
            TexDim::D3,
            TextureType::Color3D,
            &[(10, X), (11, Y), (20, Z)],
            Val::F32x3([X, Y, Z]),
        ),
        (
            TexDim::Cube,
            TextureType::ColorCube,
            &[(10, X), (11, Y), (20, Z)],
            Val::F32x3([X, Y, Z]),
        ),
        (
            TexDim::ArrayCube,
            TextureType::ColorArrayCube,
            &[(10, LAYER_BITS), (11, X), (20, Y), (21, Z)],
            Val::F32x4([X, Y, Z, LAYER]),
        ),
    ];
    for (dim, ty, regs, coords) in cases {
        let op = OpTex { dim, ..tex() };
        sample_rgba(
            op,
            regs,
            Opcode::ImageSampleImplicitLod,
            info(ty),
            vec![HANDLE, coords, NONE, NONE],
        );
    }
}

/// In the vector layout every coordinate, the array index first, is consecutive from `A`, and
/// the one operand after them is `B`.
#[test]
fn every_dimension_in_the_vector_layout() {
    let cases: [(TexDim, TextureType, &[(u8, u32)], Val); 7] = [
        (TexDim::D1, TextureType::Color1D, &[(10, X)], Val::F32(X)),
        (
            TexDim::Array1D,
            TextureType::ColorArray1D,
            &[(10, LAYER_BITS), (11, X)],
            Val::F32x2([X, LAYER]),
        ),
        (
            TexDim::D2,
            TextureType::Color2D,
            &[(10, X), (11, Y)],
            Val::F32x2([X, Y]),
        ),
        (
            TexDim::Array2D,
            TextureType::ColorArray2D,
            &[(10, LAYER_BITS), (11, X), (12, Y)],
            Val::F32x3([X, Y, LAYER]),
        ),
        (
            TexDim::D3,
            TextureType::Color3D,
            &[(10, X), (11, Y), (12, Z)],
            Val::F32x3([X, Y, Z]),
        ),
        (
            TexDim::Cube,
            TextureType::ColorCube,
            &[(10, X), (11, Y), (12, Z)],
            Val::F32x3([X, Y, Z]),
        ),
        (
            TexDim::ArrayCube,
            TextureType::ColorArrayCube,
            &[(10, LAYER_BITS), (11, X), (12, Y), (13, Z)],
            Val::F32x4([X, Y, Z, LAYER]),
        ),
    ];
    for (dim, ty, regs, coords) in cases {
        let op = OpTex {
            dim,
            scalar: false,
            ..tex()
        };
        sample_rgba(
            op,
            regs,
            Opcode::ImageSampleImplicitLod,
            info(ty),
            vec![HANDLE, coords, NONE, NONE],
        );
    }
    // The vector layout's one extra operand, a level, is at `B`.
    let op = OpTex {
        lod_mode: TexLodMode::Lod,
        scalar: false,
        ..tex()
    };
    sample_rgba(
        op,
        &[(10, X), (11, Y), (20, Z)],
        Opcode::ImageSampleExplicitLod,
        info(TextureType::Color2D),
        vec![HANDLE, Val::F32x2([X, Y]), Val::F32(Z), NONE],
    );
}

/// `.LZ` samples level 0.0 explicitly, `.LL` the level after the coordinates, `.LB` biases the
/// implicit level by it.
#[test]
fn level_of_detail_modes() {
    let level = f(1.75);
    let coords = Val::F32x2([X, Y]);
    let ty = TextureType::Color2D;
    let zero = OpTex {
        lod_mode: TexLodMode::Zero,
        ..tex()
    };
    sample_rgba(
        zero,
        &[(10, X), (20, Y)],
        Opcode::ImageSampleExplicitLod,
        info(ty),
        vec![HANDLE, coords.clone(), Val::F32(f(0.0)), NONE],
    );
    let lod = OpTex {
        lod_mode: TexLodMode::Lod,
        ..tex()
    };
    sample_rgba(
        lod,
        &[(10, X), (11, Y), (20, level)],
        Opcode::ImageSampleExplicitLod,
        info(ty),
        vec![HANDLE, coords.clone(), Val::F32(level), NONE],
    );
    let bias = OpTex {
        lod_mode: TexLodMode::Bias,
        ..tex()
    };
    sample_rgba(
        bias,
        &[(10, X), (11, Y), (20, level)],
        Opcode::ImageSampleImplicitLod,
        TextureInfo {
            has_bias: true,
            ..info(ty)
        },
        vec![HANDLE, coords, Val::F32(level), NONE],
    );
}

/// `.LC` and `.LB.LC` carry the clamp in the bias slot (alone, or after the bias), read from
/// the high half of the array-index operand as 8.8 fixed point: a texture with no array index
/// gains that operand at the front of its list (module docs, "Level-of-detail clamp").
#[test]
fn lod_clamp_is_carried() {
    // 0x0280 / 256 = 2.5; the low half (an array index, or zero) is not the clamp.
    let clamp_word = 0x0280_0005;
    let clamp = f(2.5);
    let bias = f(-0.5);
    let ty = TextureType::Color2D;
    let with_clamp = TextureInfo {
        has_lod_clamp: true,
        ..info(ty)
    };
    let op = OpTex {
        lod_mode: TexLodMode::Clamp,
        ..tex()
    };
    sample_rgba(
        op,
        &[(10, clamp_word), (11, X), (20, Y)],
        Opcode::ImageSampleImplicitLod,
        with_clamp,
        vec![HANDLE, Val::F32x2([X, Y]), Val::F32(clamp), NONE],
    );
    let op = OpTex {
        lod_mode: TexLodMode::BiasClamp,
        ..tex()
    };
    sample_rgba(
        op,
        &[(10, clamp_word), (11, X), (20, Y), (21, bias)],
        Opcode::ImageSampleImplicitLod,
        TextureInfo {
            has_bias: true,
            ..with_clamp
        },
        vec![HANDLE, Val::F32x2([X, Y]), Val::F32x2([bias, clamp]), NONE],
    );
    // An array texture's index register holds the clamp in its high half.
    let op = OpTex {
        lod_mode: TexLodMode::Clamp,
        dim: TexDim::Array2D,
        ..tex()
    };
    sample_rgba(
        op,
        &[(10, 0x0280_0003), (11, X), (20, Y)],
        Opcode::ImageSampleImplicitLod,
        TextureInfo {
            has_lod_clamp: true,
            ..info(TextureType::ColorArray2D)
        },
        vec![HANDLE, Val::F32x3([X, Y, LAYER]), Val::F32(clamp), NONE],
    );
    // With a depth reference the clamp is the `Dref` form's bias slot.
    let op = OpTex {
        lod_mode: TexLodMode::Clamp,
        z_cmpr: true,
        ..tex()
    };
    let machine = run_op(
        Op::Tex(op),
        &[(10, clamp_word), (11, X), (20, Y), (21, Z)],
        vec![call(
            Opcode::ImageSampleDrefImplicitLod,
            TextureInfo {
                is_depth: true,
                ..with_clamp
            },
            vec![HANDLE, Val::F32x2([X, Y]), Val::F32(Z), Val::F32(clamp), NONE],
            Some(Val::F32(f(0.5))),
        )],
    );
    assert_eq!(machine.regs[..4], [f(0.5), f(0.5), f(0.5), f(1.0)]);
}

/// `.DC` reads the depth reference after the coordinates and the level; its one-float result
/// goes to red, green and blue, and alpha is 1.0.
#[test]
fn depth_comparison() {
    let dref = f(0.625);
    let result = f(0.25);
    let depth = TextureInfo {
        is_depth: true,
        ..info(TextureType::Color2D)
    };
    let implicit = OpTex {
        z_cmpr: true,
        ..tex()
    };
    let machine = run_op(
        Op::Tex(implicit),
        &[(10, X), (11, Y), (20, dref)],
        vec![call(
            Opcode::ImageSampleDrefImplicitLod,
            depth,
            vec![HANDLE, Val::F32x2([X, Y]), Val::F32(dref), NONE, NONE],
            Some(Val::F32(result)),
        )],
    );
    assert_eq!(machine.regs[..4], [result, result, result, f(1.0)]);
    let explicit = OpTex {
        z_cmpr: true,
        lod_mode: TexLodMode::Lod,
        channel_mask: mask(0xa),
        dsts: [r(0), Gpr::RZ],
        ..tex()
    };
    let level = f(3.0);
    let machine = run_op(
        Op::Tex(explicit),
        &[(10, X), (11, Y), (20, level), (21, dref), (1, 7), (2, 7)],
        vec![call(
            Opcode::ImageSampleDrefExplicitLod,
            depth,
            vec![HANDLE, Val::F32x2([X, Y]), Val::F32(dref), Val::F32(level), NONE],
            Some(Val::F32(result)),
        )],
    );
    // Green then alpha, to R0 and R1.
    assert_eq!(machine.regs[..3], [result, f(1.0), 7]);
}

/// `.AOFFI` reads two signed 4-bit offsets at bits 0 and 4 of `B` (or `B + 1` after an explicit
/// level), the coordinates from the even pair at `A`; bits above the fields are ignored.
#[test]
fn offsets() {
    // x = 0xd (-3), y = 5, and noise above bit 8.
    let packed = 0x1234_565d;
    let offset = Val::U32x2([0xffff_fffd, 5]);
    let op = OpTex {
        offset: true,
        ..tex()
    };
    sample_rgba(
        op,
        &[(10, X), (11, Y), (20, packed)],
        Opcode::ImageSampleImplicitLod,
        info(TextureType::Color2D),
        vec![HANDLE, Val::F32x2([X, Y]), NONE, offset.clone()],
    );
    let level = f(2.0);
    let op = OpTex {
        offset: true,
        lod_mode: TexLodMode::Lod,
        ..tex()
    };
    sample_rgba(
        op,
        &[(10, X), (11, Y), (20, level), (21, packed)],
        Opcode::ImageSampleExplicitLod,
        info(TextureType::Color2D),
        vec![HANDLE, Val::F32x2([X, Y]), Val::F32(level), offset],
    );
}

/// Each channel mask writes exactly its channels, in order, to the first destination and the
/// register after it, then to the second destination and the one after it; nothing else.
#[test]
fn every_mask_writes_its_channels_in_order() {
    const SENTINEL: u32 = 0xdead_0000;
    for bits in 1..16_u8 {
        let outputs = bits.count_ones();
        let second = if outputs > 2 { r(8) } else { Gpr::RZ };
        let op = OpTex {
            channel_mask: mask(bits),
            dsts: [r(4), second],
            ..tex()
        };
        let mut regs: Vec<(u8, u32)> = (0..16).map(|index| (index, SENTINEL)).collect();
        regs.extend([(10, X), (20, Y)]);
        let machine = run_op(
            Op::Tex(op),
            &regs,
            vec![call(
                Opcode::ImageSampleImplicitLod,
                info(TextureType::Color2D),
                vec![HANDLE, Val::F32x2([X, Y]), NONE, NONE],
                Some(Val::F32x4(TEXEL)),
            )],
        );
        let mut expected = [SENTINEL; 16];
        expected[10] = X;
        let mut stored = 0;
        for channel in 0..4 {
            if bits & (1 << channel) != 0 {
                let target = if stored < 2 { 4 + stored } else { 8 + stored - 2 };
                expected[target] = TEXEL[channel];
                stored += 1;
            }
        }
        assert_eq!(machine.regs[..16], expected, "mask {bits:#x}");
    }
}

/// `.F16` packs the channels in pairs, the first of each pair in the low half, and pads a
/// missing last channel with 0.0.
#[test]
fn half_precision_results_pack_pairs() {
    // 0.125, 0.25, 0.5 and 0.75 as halves.
    let halves = [0x3000, 0x3400, 0x3800, 0x3a00];
    let cases = [
        (0xf, [r(0), r(2)], [(0, halves[1] << 16 | halves[0]), (2, halves[3] << 16 | halves[2])]),
        (0x7, [r(0), r(2)], [(0, halves[1] << 16 | halves[0]), (2, halves[2])]),
        (0x1, [r(0), Gpr::RZ], [(0, halves[0]), (2, 9)]),
        (0x6, [r(0), Gpr::RZ], [(0, halves[2] << 16 | halves[1]), (2, 9)]),
    ];
    for (bits, dsts, expected) in cases {
        let op = OpTex {
            f16: true,
            channel_mask: mask(bits),
            dsts,
            ..tex()
        };
        let machine = run_op(
            Op::Tex(op),
            &[(10, X), (20, Y), (1, 9), (2, 9), (3, 9)],
            vec![call(
                Opcode::ImageSampleImplicitLod,
                TextureInfo {
                    relaxed_precision: true,
                    ..info(TextureType::Color2D)
                },
                vec![HANDLE, Val::F32x2([X, Y]), NONE, NONE],
                Some(Val::F32x4(TEXEL)),
            )],
        );
        for (reg, bits_expected) in expected {
            assert_eq!(machine.regs[reg], bits_expected, "mask {bits:#x}, r{reg}");
        }
        // The register after a packed one is not written.
        assert_eq!(machine.regs[1], 9, "mask {bits:#x}");
    }
}

/// A `.UR` handle is the uniform register's value, and leaves the descriptor buffer unset.
#[test]
fn a_uniform_register_handle() {
    let op = OpTex {
        tex: TexRef::Uniform(ur(62)),
        ..tex()
    };
    let function = translated(Op::Tex(op));
    let mut machine = machine_with(&[(10, X), (20, Y)]);
    machine.uniform_regs[62] = 0x0004_0123;
    run_script(
        &function,
        &mut machine,
        vec![call(
            Opcode::ImageSampleImplicitLod,
            info(TextureType::Color2D),
            vec![Val::U32(0x0004_0123), Val::F32x2([X, Y]), NONE, NONE],
            Some(Val::F32x4(TEXEL)),
        )],
    );
    let (_, buffer) =
        translate_raw(Instr::new(Op::Tex(op)), DescriptorBuffer::new()).expect("translates");
    assert_eq!(buffer.index(), None);
}

/// A constant-buffer handle sets the descriptor buffer to its buffer.
#[test]
fn a_constant_buffer_handle_sets_the_descriptor_buffer() {
    let op = OpTex {
        tex: cb(17, 0x1ffc),
        ..tex()
    };
    let (_, buffer) =
        translate_raw(Instr::new(Op::Tex(op)), DescriptorBuffer::new()).expect("translates");
    assert_eq!(buffer.index(), Some(17));
    let (_, again) = translate_raw(Instr::new(Op::Tex(op)), buffer).expect("translates");
    assert_eq!(again, buffer);
}

/// `TEX.B` reads its coordinates from `A`, the one operand after them at `B`, and its handle
/// in the register after that (or at `B` with no such operand).
#[test]
fn a_register_handle_follows_the_operands() {
    let handle = 0x00ab_cdef;
    let bindless = OpTex {
        tex: TexRef::Bindless,
        ..tex()
    };
    sample_rgba(
        bindless,
        &[(10, X), (11, Y), (20, handle)],
        Opcode::ImageSampleImplicitLod,
        info(TextureType::Color2D),
        vec![Val::U32(handle), Val::F32x2([X, Y]), NONE, NONE],
    );
    let level = f(4.0);
    let op = OpTex {
        lod_mode: TexLodMode::Lod,
        ..bindless
    };
    sample_rgba(
        op,
        &[(10, X), (11, Y), (20, level), (21, handle)],
        Opcode::ImageSampleExplicitLod,
        info(TextureType::Color2D),
        vec![Val::U32(handle), Val::F32x2([X, Y]), Val::F32(level), NONE],
    );
    let op = OpTex {
        dim: TexDim::Cube,
        z_cmpr: true,
        ..bindless
    };
    let machine = run_op(
        Op::Tex(op),
        &[(10, X), (11, Y), (12, Z), (20, f(0.5)), (21, handle)],
        vec![call(
            Opcode::ImageSampleDrefImplicitLod,
            TextureInfo {
                is_depth: true,
                ..info(TextureType::ColorCube)
            },
            vec![
                Val::U32(handle),
                Val::F32x3([X, Y, Z]),
                Val::F32(f(0.5)),
                NONE,
                NONE,
            ],
            Some(Val::F32(f(0.75))),
        )],
    );
    assert_eq!(machine.regs[..4], [f(0.75), f(0.75), f(0.75), f(1.0)]);
}

/// A fault predicate other than `PT` is written false: every texel is reported resident (the
/// module's divergence table). `PT` writes nothing.
#[test]
fn sparse_results_report_resident() {
    let op = OpTex {
        fault: p(2),
        ..tex()
    };
    let function = translated(Op::Tex(op));
    let mut machine = machine_with(&[(10, X), (20, Y)]);
    machine.preds[2] = true;
    machine.preds[3] = true;
    run_script(
        &function,
        &mut machine,
        vec![call(
            Opcode::ImageSampleImplicitLod,
            info(TextureType::Color2D),
            vec![HANDLE, Val::F32x2([X, Y]), NONE, NONE],
            Some(Val::F32x4(TEXEL)),
        )],
    );
    assert!(!machine.preds[2]);
    assert!(machine.preds[3]);
    let plain = translated(Op::Tex(tex()));
    assert!(!plain.to_string().contains("SetPred"));
}
