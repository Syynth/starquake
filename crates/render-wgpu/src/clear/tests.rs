//! How each clear is done, without a GPU: [`clear_how`] over every mask class, area, sample count
//! and value kind, the parameter block's bytes and the pass's load operations.

use super::*;
use starquake_render::id::ViewId;

const RGBA: PixelFormat = PixelFormat::A8B8G8R8Unorm;
const RGBA_HOST: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DS: PixelFormat = PixelFormat::D32FloatS8Uint;
const DS_HOST: wgpu::TextureFormat = wgpu::TextureFormat::Depth32FloatStencil8;
const DEPTH: PixelFormat = PixelFormat::D32Float;
const DEPTH_HOST: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const STENCIL: PixelFormat = PixelFormat::S8Uint;
const STENCIL_HOST: wgpu::TextureFormat = wgpu::TextureFormat::Stencil8;

fn desc(value: ClearValue, color_mask: ColorWrites, stencil_mask: u8) -> ClearDesc {
    ClearDesc {
        target: ViewId::new(1, 0),
        value,
        rect: None,
        color_mask,
        stencil_mask,
    }
}

fn drawn(how: ClearHow) -> DrawnClear {
    match how {
        ClearHow::Draw(d) => d,
        other => panic!("not drawn: {other:?}"),
    }
}

const FLOAT: ClearValue = ClearValue::Float([1.0, 0.0, 1.0, 0.0]);

#[test]
fn a_colour_clear_is_drawn_for_a_partial_mask_or_a_multisampled_rectangle() {
    let all = ColorWrites::ALL;
    let red = ColorWrites::RED;
    // (mask, samples, whole) -> how.
    let pass = ClearHow::Pass;
    let scratch = ClearHow::Scratch;
    for (mask, samples, whole, expect) in [
        (all, 1, true, Some(pass)),
        (all, 1, false, Some(scratch)),
        (all, 4, true, Some(pass)),
        (all, 4, false, None),
        (red, 1, true, None),
        (red, 1, false, None),
        (red, 4, true, None),
        (red, 4, false, None),
        (ColorWrites::EMPTY, 1, true, Some(ClearHow::Nothing)),
        (ColorWrites::EMPTY, 4, false, Some(ClearHow::Nothing)),
    ] {
        let how = clear_how(RGBA, RGBA_HOST, samples, whole, &desc(FLOAT, mask, 0xff));
        match expect {
            Some(expect) => assert_eq!(how, expect, "{mask:?} {samples} {whole}"),
            None => {
                let d = drawn(how);
                assert_eq!(
                    d.key,
                    PipelineKey {
                        format: RGBA_HOST,
                        samples,
                        target: DrawTarget::Color {
                            kind: ColorKind::Float,
                            mask: host_color_writes(mask),
                        },
                    },
                    "{mask:?} {samples} {whole}"
                );
            }
        }
    }
}

#[test]
fn the_stencil_mask_does_not_change_a_colour_clear() {
    for stencil_mask in [0, 0x0f, 0xff] {
        assert_eq!(
            clear_how(
                RGBA,
                RGBA_HOST,
                1,
                true,
                &desc(FLOAT, ColorWrites::ALL, stencil_mask)
            ),
            ClearHow::Pass
        );
    }
}

#[test]
fn every_colour_mask_maps_to_wgpus_bits() {
    let host = [
        wgpu::ColorWrites::RED,
        wgpu::ColorWrites::GREEN,
        wgpu::ColorWrites::BLUE,
        wgpu::ColorWrites::ALPHA,
    ];
    for bits in 0..16_u32 {
        let mask = ColorWrites::from_bits(bits).expect("four bits");
        let expected = (0..4)
            .filter(|i| bits & (1 << i) != 0)
            .fold(wgpu::ColorWrites::empty(), |all, i| all | host[i]);
        assert_eq!(host_color_writes(mask), expected, "{bits:#b}");
    }
}

#[test]
fn each_colour_kind_takes_its_entry_point() {
    for (format, kind, entry) in [
        (RGBA, ColorKind::Float, shader::FRAGMENT_COLOR_FLOAT),
        (
            PixelFormat::R16G16B16A16Float,
            ColorKind::Float,
            shader::FRAGMENT_COLOR_FLOAT,
        ),
        (
            PixelFormat::A8B8G8R8Sint,
            ColorKind::Sint,
            shader::FRAGMENT_COLOR_SINT,
        ),
        (
            PixelFormat::R32Uint,
            ColorKind::Uint,
            shader::FRAGMENT_COLOR_UINT,
        ),
    ] {
        assert_eq!(ColorKind::of(format), kind, "{format}");
        assert_eq!(kind.entry_point(), entry);
    }
    // The literal names of `clear.wgsl` (the render crate's tests pin them against the source).
    assert_eq!(ColorKind::Float.entry_point(), "fs_color_float");
    assert_eq!(ColorKind::Sint.entry_point(), "fs_color_sint");
    assert_eq!(ColorKind::Uint.entry_point(), "fs_color_uint");
}

#[test]
fn a_depth_stencil_clear_is_drawn_for_a_partial_stencil_mask_or_a_rectangle() {
    let both = ClearValue::DepthStencil {
        depth: 0.5,
        stencil: 0x44,
    };
    let pass = Some(ClearHow::Pass);
    // (value, stencil mask, whole) -> how, or the drawn target.
    for (value, mask, whole, expect, target) in [
        // Whole, every bit: a pass.
        (both, 0xff, true, pass, None),
        // Whole, an empty mask: a pass that keeps the stencil.
        (both, 0x00, true, pass, None),
        // Whole, a partial mask: drawn, depth too.
        (
            both,
            0x0f,
            true,
            None,
            Some(DrawTarget::DepthStencil {
                depth: true,
                stencil: Some(0x0f),
            }),
        ),
        (
            both,
            0xfe,
            true,
            None,
            Some(DrawTarget::DepthStencil {
                depth: true,
                stencil: Some(0xfe),
            }),
        ),
        // A rectangle: drawn, through every bit.
        (
            both,
            0xff,
            false,
            None,
            Some(DrawTarget::DepthStencil {
                depth: true,
                stencil: Some(0xff),
            }),
        ),
        // A rectangle and an empty mask: drawn, depth alone.
        (
            both,
            0x00,
            false,
            None,
            Some(DrawTarget::DepthStencil {
                depth: true,
                stencil: None,
            }),
        ),
        // Depth alone: the stencil mask is not read.
        (ClearValue::Depth(0.5), 0x0f, true, pass, None),
        (
            ClearValue::Depth(0.5),
            0x0f,
            false,
            None,
            Some(DrawTarget::DepthStencil {
                depth: true,
                stencil: None,
            }),
        ),
        // Stencil alone through an empty mask: nothing, whatever the area.
        (
            ClearValue::Stencil(0x44),
            0x00,
            true,
            Some(ClearHow::Nothing),
            None,
        ),
        (
            ClearValue::Stencil(0x44),
            0x00,
            false,
            Some(ClearHow::Nothing),
            None,
        ),
        (
            ClearValue::Stencil(0x44),
            0x0f,
            true,
            None,
            Some(DrawTarget::DepthStencil {
                depth: false,
                stencil: Some(0x0f),
            }),
        ),
    ] {
        let how = clear_how(DS, DS_HOST, 1, whole, &desc(value, ColorWrites::ALL, mask));
        let case = format!("{value:?} {mask:#x} {whole}");
        match (expect, target) {
            (Some(expect), None) => assert_eq!(how, expect, "{case}"),
            (None, Some(target)) => {
                let d = drawn(how);
                assert_eq!(d.key.target, target, "{case}");
                assert_eq!(d.key.format, DS_HOST);
            }
            _ => unreachable!("a case names one outcome"),
        }
    }
}

#[test]
fn single_aspect_formats_write_only_their_aspect() {
    // A depth-only format: a partial area is drawn with depth and no stencil, whatever the mask.
    let d = drawn(clear_how(
        DEPTH,
        DEPTH_HOST,
        1,
        false,
        &desc(ClearValue::Depth(1.0), ColorWrites::EMPTY, 0x0f),
    ));
    assert_eq!(
        d.key.target,
        DrawTarget::DepthStencil {
            depth: true,
            stencil: None
        }
    );
    // A stencil-only format: a partial mask is drawn with stencil and no depth.
    let d = drawn(clear_how(
        STENCIL,
        STENCIL_HOST,
        4,
        true,
        &desc(ClearValue::Stencil(3), ColorWrites::EMPTY, 0x81),
    ));
    assert_eq!(
        d.key,
        PipelineKey {
            format: STENCIL_HOST,
            samples: 4,
            target: DrawTarget::DepthStencil {
                depth: false,
                stencil: Some(0x81)
            }
        }
    );
    assert_eq!(d.stencil_reference, 3);
}

#[test]
fn the_parameter_block_holds_the_values_patterns() {
    let block = |value| {
        drawn(clear_how(
            PixelFormat::R32G32B32A32Sint,
            wgpu::TextureFormat::Rgba32Sint,
            1,
            true,
            &desc(value, ColorWrites::RED, 0xff),
        ))
        .params
    };
    let mut expected = Vec::new();
    for v in [-1_i32, 2, -3, 4] {
        expected.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(block(ClearValue::Sint([-1, 2, -3, 4])).to_vec(), expected);
    // Depth first; stencil goes in the reference, not the block.
    let d = drawn(clear_how(
        DS,
        DS_HOST,
        1,
        false,
        &desc(
            ClearValue::DepthStencil {
                depth: 0.25,
                stencil: 0x9a,
            },
            ColorWrites::ALL,
            0xff,
        ),
    ));
    let mut expected = 0.25_f32.to_bits().to_le_bytes().to_vec();
    expected.extend([0; 12]);
    assert_eq!(d.params.to_vec(), expected);
    assert_eq!(d.stencil_reference, 0x9a);
    assert_eq!(value_patterns(ClearValue::Stencil(7)), [0; 4]);
    assert_eq!(
        value_patterns(ClearValue::Float([1.0, -2.0, 0.5, 0.0])),
        [0x3f80_0000, 0xc000_0000, 0x3f00_0000, 0]
    );
    assert_eq!(
        value_patterns(ClearValue::Uint([1, 2, 3, u32::MAX])),
        [1, 2, 3, u32::MAX]
    );
    assert_eq!(PARAMS_LEN, 16);
}

#[test]
fn a_passes_stencil_is_kept_when_the_mask_is_empty() {
    let both = ClearValue::DepthStencil {
        depth: 0.5,
        stencil: 7,
    };
    for (mask, stencil) in [(0x00, None), (0x01, Some(7)), (0xff, Some(7))] {
        assert_eq!(
            pass_ops(&desc(both, ColorWrites::ALL, mask)),
            ClearOps::DepthStencil {
                depth: Some(0.5),
                stencil
            },
            "{mask:#x}"
        );
    }
    // A colour clear's ops are its colour, whatever the stencil mask.
    assert_eq!(
        pass_ops(&desc(FLOAT, ColorWrites::ALL, 0)),
        crate::plan::clear_ops(FLOAT)
    );
}

#[test]
fn the_kept_pipelines_are_dropped_only_for_a_new_key_at_the_cap() {
    let cap = MAX_CLEAR_PIPELINES;
    // A key not kept: dropped exactly at the cap and past it, kept one below.
    assert!(!must_evict(cap - 1, cap, false));
    assert!(must_evict(cap, cap, false));
    assert!(must_evict(cap + 1, cap, false));
    // A key already kept is reused, at the cap too.
    assert!(!must_evict(cap, cap, true));
    assert!(!must_evict(cap - 1, cap, true));
    // The cap is the documented guess.
    assert_eq!(MAX_CLEAR_PIPELINES, 256);
}
