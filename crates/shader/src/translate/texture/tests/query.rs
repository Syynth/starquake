//! `TXD`, `TMML` and `TXQ`.

use super::*;

const HANDLE: Val = Val::U32(0x40);
const X: u32 = f(0.375);
const Y: u32 = f(-1.5);
const Z: u32 = f(2.25);
const W: u32 = f(-8.0);

/// `TXD` reads its coordinates from `A` (an array index after them) and four derivatives from
/// `B`, and samples with two derivative pairs.
#[test]
fn gradients() {
    let derivatives = [f(0.0625), f(-0.125), f(4.0), f(16.0)];
    let regs = [
        (10, X),
        (11, Y),
        (12, 0x0003_0004),
        (20, derivatives[0]),
        (21, derivatives[1]),
        (22, derivatives[2]),
        (23, derivatives[3]),
    ];
    let cases = [
        (TexDim::D2, TextureType::Color2D, Val::F32x2([X, Y])),
        (
            TexDim::Array2D,
            TextureType::ColorArray2D,
            Val::F32x3([X, Y, f(4.0)]),
        ),
    ];
    for (dim, ty, coords) in cases {
        let op = OpTxd {
            dim,
            fault: p(1),
            ..txd()
        };
        let function = translated(Op::Txd(op));
        let mut machine = machine_with(&regs);
        machine.preds[1] = true;
        run_script(
            &function,
            &mut machine,
            vec![call(
                Opcode::ImageGradient,
                TextureInfo {
                    num_derivatives: 2,
                    ..info(ty)
                },
                vec![HANDLE, coords, Val::F32x4(derivatives), NONE, NONE],
                Some(Val::F32x4(TEXEL)),
            )],
        );
        assert_eq!(machine.regs[..4], TEXEL, "{dim}");
        assert!(!machine.preds[1], "{dim}");
    }
}

/// `TMML` reads every coordinate of its dimension from `A`, skipping an array index, and
/// writes the selected levels as 8.8 fixed point: channel 0 the relative level (the host's
/// second), signed; channel 1 the absolute level, unsigned.
#[test]
fn level_queries_of_every_dimension() {
    let cases: [(TexDim, TextureType, Val); 7] = [
        (TexDim::D1, TextureType::Color1D, Val::F32(X)),
        (TexDim::Array1D, TextureType::ColorArray1D, Val::F32(Y)),
        (TexDim::D2, TextureType::Color2D, Val::F32x2([X, Y])),
        (
            TexDim::Array2D,
            TextureType::ColorArray2D,
            Val::F32x2([Y, Z]),
        ),
        (TexDim::D3, TextureType::Color3D, Val::F32x3([X, Y, Z])),
        (TexDim::Cube, TextureType::ColorCube, Val::F32x3([X, Y, Z])),
        (
            TexDim::ArrayCube,
            TextureType::ColorArrayCube,
            Val::F32x3([Y, Z, W]),
        ),
    ];
    for (dim, ty, coords) in cases {
        let op = OpTmml {
            dim,
            channel_mask: mask(0x3),
            ..tmml()
        };
        let machine = run_op(
            Op::Tmml(op),
            &[(10, X), (11, Y), (12, Z), (13, W)],
            vec![call(
                Opcode::ImageQueryLod,
                info(ty),
                vec![HANDLE, coords],
                // Absolute 2.75, relative -1.5.
                Some(Val::F32x4([f(2.75), f(-1.5), 0, 0])),
            )],
        );
        assert_eq!(machine.regs[4..6], [0xfe80, 0x02c0], "{dim}");
    }
}

/// One selected channel goes to the destination itself.
#[test]
fn level_query_channels_are_consecutive() {
    for (bits, expected) in [(0x1_u8, 0xfe80), (0x2, 0x02c0)] {
        let op = OpTmml {
            channel_mask: mask(bits),
            ..tmml()
        };
        let machine = run_op(
            Op::Tmml(op),
            &[(10, X), (11, Y), (5, 9)],
            vec![call(
                Opcode::ImageQueryLod,
                info(TextureType::Color2D),
                vec![HANDLE, Val::F32x2([X, Y])],
                Some(Val::F32x4([f(2.75), f(-1.5), 0, 0])),
            )],
        );
        assert_eq!(machine.regs[4..6], [expected, 9], "mask {bits:#x}");
    }
}

/// `TXQ` queries the level in `A`; the selected words go to consecutive registers, and the
/// level count is skipped unless the mask selects it.
#[test]
fn dimension_queries() {
    let words = [640, 480, 1, 10];
    let cases: [(u8, bool, &[u32]); 4] = [
        (0xf, false, &[640, 480, 1, 10]),
        (0x3, true, &[640, 480]),
        (0x8, false, &[10]),
        (0x5, true, &[640, 1]),
    ];
    for (bits, skip, expected) in cases {
        let op = OpTxq {
            channel_mask: mask(bits),
            ..txq()
        };
        let machine = run_op(
            Op::Txq(op),
            &[(10, 3)],
            vec![call(
                Opcode::ImageQueryDimensions,
                TextureInfo::default(),
                vec![HANDLE, Val::U32(3), Val::U1(skip)],
                Some(Val::U32x4(words)),
            )],
        );
        assert_eq!(&machine.regs[4..4 + expected.len()], expected, "mask {bits:#x}");
    }
}
