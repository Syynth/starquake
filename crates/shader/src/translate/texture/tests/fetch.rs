//! `TLD`: every dimension it takes, both level modes, half-precision results, a uniform handle
//! and the fault predicate.

use super::*;

const HANDLE: Val = Val::U32(0x40);
/// Distinct integer coordinates.
const X: u32 = 0x0000_0107;
const Y: u32 = 0x0001_0009;
const Z: u32 = 0x8000_0003;
/// An array index register: layer 6 below noise.
const LAYER_BITS: u32 = 0xffff_0006;

fn fetch(op: OpTld, regs: &[(u8, u32)], ty: TextureType, coords: Val, lod: Val) -> Machine {
    run_op(
        Op::Tld(op),
        regs,
        vec![call(
            Opcode::ImageFetch,
            info(ty),
            vec![HANDLE, coords, NONE, lod, NONE],
            Some(Val::F32x4(TEXEL)),
        )],
    )
}

/// Integer coordinates in the scalar layout; an array index first, its low 16 bits, placed
/// after the coordinates.
#[test]
fn every_dimension_it_takes() {
    let cases: [(TexDim, TextureType, &[(u8, u32)], Val); 5] = [
        (TexDim::D1, TextureType::Color1D, &[(10, X)], Val::U32(X)),
        (
            TexDim::Array1D,
            TextureType::ColorArray1D,
            &[(10, LAYER_BITS), (20, X)],
            Val::U32x2([X, 6]),
        ),
        (
            TexDim::D2,
            TextureType::Color2D,
            &[(10, X), (20, Y)],
            Val::U32x2([X, Y]),
        ),
        (
            TexDim::Array2D,
            TextureType::ColorArray2D,
            &[(10, LAYER_BITS), (11, X), (20, Y)],
            Val::U32x3([X, Y, 6]),
        ),
        (
            TexDim::D3,
            TextureType::Color3D,
            &[(10, X), (11, Y), (20, Z)],
            Val::U32x3([X, Y, Z]),
        ),
    ];
    for (dim, ty, regs, coords) in cases {
        let machine = fetch(OpTld { dim, ..tld() }, regs, ty, coords, Val::U32(0));
        assert_eq!(machine.regs[..4], TEXEL, "{dim}");
    }
}

/// `.LL` reads an integer level after the coordinates; `.LZ` fetches level 0.
#[test]
fn explicit_levels() {
    let op = OpTld {
        lod_mode: TexLodMode::Lod,
        ..tld()
    };
    let machine = fetch(
        op,
        &[(10, X), (11, Y), (20, 5)],
        TextureType::Color2D,
        Val::U32x2([X, Y]),
        Val::U32(5),
    );
    assert_eq!(machine.regs[..4], TEXEL);
}

/// `.F16` packs pairs of channels, as `TEX` does.
#[test]
fn half_precision_results() {
    let op = OpTld {
        f16: true,
        ..tld()
    };
    let machine = fetch(
        op,
        &[(10, X), (20, Y), (1, 9)],
        TextureType::Color2D,
        Val::U32x2([X, Y]),
        Val::U32(0),
    );
    assert_eq!(machine.regs[..3], [0x3400_3000, 9, 0x3a00_3800]);
}

/// A `.UR` handle is the uniform register's value; the fault predicate reports resident.
#[test]
fn a_uniform_handle_and_a_fault_predicate() {
    let op = OpTld {
        tex: TexRef::Uniform(ur(7)),
        fault: p(0),
        ..tld()
    };
    let function = translated(Op::Tld(op));
    let mut machine = machine_with(&[(10, X), (20, Y)]);
    machine.uniform_regs[7] = 0x55;
    machine.preds[0] = true;
    run_script(
        &function,
        &mut machine,
        vec![call(
            Opcode::ImageFetch,
            info(TextureType::Color2D),
            vec![Val::U32(0x55), Val::U32x2([X, Y]), NONE, Val::U32(0), NONE],
            Some(Val::F32x4(TEXEL)),
        )],
    );
    assert_eq!(machine.regs[..4], TEXEL);
    assert!(!machine.preds[0]);
}
