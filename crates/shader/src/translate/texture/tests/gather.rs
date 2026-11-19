//! `TLD4`: every dimension it takes, every gathered component, depth comparison, one offset
//! and per-texel offsets, and the fault predicate.

use super::*;

const HANDLE: Val = Val::U32(0x40);
const X: u32 = f(0.375);
const Y: u32 = f(-1.5);
const Z: u32 = f(2.25);
/// An array index register: layer 2 below noise.
const LAYER_BITS: u32 = 0x8000_0002;
const LAYER: u32 = f(2.0);

/// 2D, a 2D array and a cube, with every component.
#[test]
fn every_dimension_and_component() {
    let cases: [(TexDim, TextureType, &[(u8, u32)], Val); 3] = [
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
            TexDim::Cube,
            TextureType::ColorCube,
            &[(10, X), (11, Y), (20, Z)],
            Val::F32x3([X, Y, Z]),
        ),
    ];
    for (dim, ty, regs, coords) in cases {
        for (component, comp) in (0..).zip(TexChannel::ALL) {
            let op = OpTld4 {
                dim,
                comp: *comp,
                ..tld4()
            };
            let machine = run_op(
                Op::Tld4(op),
                regs,
                vec![call(
                    Opcode::ImageGather,
                    TextureInfo {
                        gather_component: component,
                        ..info(ty)
                    },
                    vec![HANDLE, coords.clone(), NONE, NONE],
                    Some(Val::F32x4(TEXEL)),
                )],
            );
            assert_eq!(machine.regs[..4], TEXEL, "{dim}{comp}");
        }
    }
}

/// `.DC` reads the depth reference last and gathers four comparisons.
#[test]
fn depth_comparison() {
    let dref = f(0.5);
    let op = OpTld4 {
        z_cmpr: true,
        ..tld4()
    };
    let machine = run_op(
        Op::Tld4(op),
        &[(10, X), (11, Y), (20, dref)],
        vec![call(
            Opcode::ImageGatherDref,
            TextureInfo {
                is_depth: true,
                gather_component: 1,
                ..info(TextureType::Color2D)
            },
            vec![HANDLE, Val::F32x2([X, Y]), NONE, NONE, Val::F32(dref)],
            Some(Val::F32x4(TEXEL)),
        )],
    );
    assert_eq!(machine.regs[..4], TEXEL);
}

/// `.AOFFI` reads one register after the coordinates: signed 6-bit offsets at bits 0 and 8,
/// the bits between and above ignored.
#[test]
fn one_offset() {
    // x = 0x3e (-2) with bits 6 and 7 set; y = 0x05 with bits 14 and 15 set; noise above.
    let packed = 0xffff_c5fe;
    let op = OpTld4 {
        offset_mode: TexOffsetMode::AddOffI,
        ..tld4()
    };
    let machine = run_op(
        Op::Tld4(op),
        &[(10, X), (11, Y), (20, packed)],
        vec![call(
            Opcode::ImageGather,
            TextureInfo {
                gather_component: 1,
                ..info(TextureType::Color2D)
            },
            vec![
                HANDLE,
                Val::F32x2([X, Y]),
                Val::U32x2([0xffff_fffe, 5]),
                NONE,
            ],
            Some(Val::F32x4(TEXEL)),
        )],
    );
    assert_eq!(machine.regs[..4], TEXEL);
}

/// `.PTP`: four gathers, the `k`th with texel `k`'s offset (fields `2k` and `2k + 1` of the
/// two registers), each giving its component 3 as result component `k` (module docs,
/// "Per-texel gather offsets").
#[test]
fn per_texel_offsets_take_four_gathers() {
    // Offsets (1, -1), (2, -2), (3, -3), (-32, 31), 6 bits each, one byte apart.
    let first = 0x3e02_3f01;
    let second = 0x1f20_3d03;
    let offsets = [
        [1, 0xffff_ffff],
        [2, 0xffff_fffe],
        [3, 0xffff_fffd],
        [0xffff_ffe0, 31],
    ];
    let replies: [[u32; 4]; 4] = [
        [1, 2, 3, f(0.125)],
        [4, 5, 6, f(0.25)],
        [7, 8, 9, f(0.5)],
        [10, 11, 12, f(0.75)],
    ];
    let gathers = |opcode, info, coords: Val, dref: Option<Val>| -> Vec<ScriptedImage> {
        offsets
            .iter()
            .zip(replies)
            .map(|(offset, reply)| {
                let mut args = vec![HANDLE, coords.clone(), Val::U32x2(*offset), NONE];
                args.extend(dref.clone());
                call(opcode, info, args, Some(Val::F32x4(reply)))
            })
            .collect()
    };
    let op = OpTld4 {
        offset_mode: TexOffsetMode::PerPx,
        ..tld4()
    };
    let machine = run_op(
        Op::Tld4(op),
        &[(10, X), (11, Y), (20, first), (21, second)],
        gathers(
            Opcode::ImageGather,
            TextureInfo {
                gather_component: 1,
                ..info(TextureType::Color2D)
            },
            Val::F32x2([X, Y]),
            None,
        ),
    );
    assert_eq!(machine.regs[..4], TEXEL);
    // With an array index and a depth reference: the offsets after the coordinates, the
    // reference last, in every gather.
    let dref = f(0.875);
    let op = OpTld4 {
        offset_mode: TexOffsetMode::PerPx,
        dim: TexDim::Array2D,
        z_cmpr: true,
        ..tld4()
    };
    let machine = run_op(
        Op::Tld4(op),
        &[
            (10, LAYER_BITS),
            (11, X),
            (20, Y),
            (21, first),
            (22, second),
            (23, dref),
        ],
        gathers(
            Opcode::ImageGatherDref,
            TextureInfo {
                is_depth: true,
                gather_component: 1,
                ..info(TextureType::ColorArray2D)
            },
            Val::F32x3([X, Y, LAYER]),
            Some(Val::F32(dref)),
        ),
    );
    assert_eq!(machine.regs[..4], TEXEL);
}

/// The fault predicate reports resident; channel masks place results as `TEX` does.
#[test]
fn a_fault_predicate_and_a_partial_mask() {
    let op = OpTld4 {
        fault: p(5),
        channel_mask: mask(0x9),
        dsts: [r(6), Gpr::RZ],
        ..tld4()
    };
    let function = translated(Op::Tld4(op));
    let mut machine = machine_with(&[(10, X), (20, Y)]);
    machine.preds[5] = true;
    run_script(
        &function,
        &mut machine,
        vec![call(
            Opcode::ImageGather,
            TextureInfo {
                gather_component: 1,
                ..info(TextureType::Color2D)
            },
            vec![HANDLE, Val::F32x2([X, Y]), NONE, NONE],
            Some(Val::F32x4(TEXEL)),
        )],
    );
    assert_eq!(machine.regs[6..8], [TEXEL[0], TEXEL[3]]);
    assert!(!machine.preds[5]);
}
