//! Masked and partial clears on a real GPU (`crate::clear`): the clears wgpu cannot do as
//! clears are drawn, and every readback the recording backend knows must come back identical.
//! Multisampled and two-aspect depth-stencil views cannot be read back under wgpu, so their
//! drawn clears are checked for running with no host error only.

use super::*;
use starquake_render::command::ALL_STENCIL_BITS;
use starquake_render::pipeline::ColorWrites;

/// A clear through masks.
fn masked(
    target: ViewId,
    value: ClearValue,
    rect: Option<Rect>,
    color_mask: ColorWrites,
    stencil_mask: u8,
) -> Command {
    Command::Clear(ClearDesc {
        target,
        value,
        rect,
        color_mask,
        stencil_mask,
    })
}

const PART: Rect = Rect {
    x: 1,
    y: 1,
    width: 5,
    height: 2,
};

/// The packed bytes of an 8 by 4 level of `format`, one layer.
fn level_bytes(format: PixelFormat) -> usize {
    let size = validate::packed_size(
        format,
        Extent3d {
            width: 8,
            height: 4,
            depth_or_layers: 1,
        },
    )
    .expect("small");
    usize::try_from(size).expect("small")
}

#[test]
fn masked_colour_clears_write_what_the_recording_backend_knows() {
    bounded(|| {
        let Some(gpu) = backend("masked_colour_clears") else {
            return;
        };
        let mut p = Pair::new(gpu);
        let cases = [
            (
                PixelFormat::A8B8G8R8Unorm,
                ClearValue::Float([1.0, 0.0, 1.0, 0.0]),
            ),
            (
                PixelFormat::A8B8G8R8Uint,
                ClearValue::Uint([255, 0, 7, 128]),
            ),
            (
                PixelFormat::R16G16B16A16Sint,
                ClearValue::Sint([-1, 2, -300, 4000]),
            ),
            (
                PixelFormat::R32G32B32A32Float,
                ClearValue::Float([0.1, -2.5, 3.0e9, 1.0]),
            ),
            (PixelFormat::R32Uint, ClearValue::Uint([u32::MAX, 0, 0, 0])),
        ];
        let masks = [
            ColorWrites::RED,
            ColorWrites::GREEN.union(ColorWrites::ALPHA),
            ColorWrites::RED.union(ColorWrites::BLUE),
            ColorWrites::ALL,
        ];
        let mut lists = 0;
        for (format, value) in cases {
            for mask in masks {
                let tex = p.texture(&texture_desc(
                    TextureDimension::D2,
                    format,
                    (8, 4, 1),
                    1,
                    TARGET,
                ));
                let view = p.view(tex, &whole_view(format, ViewDimension::D2, (0, 1)));
                let whole = region(tex, 0, (0, 0, 0), (8, 4, 1));
                let mut list = vec![
                    upload(pattern(level_bytes(format), 31), whole),
                    masked(view, value, Some(PART), mask, ALL_STENCIL_BITS),
                ];
                list.push(p.readback(whole));
                list.push(masked(view, value, None, mask, ALL_STENCIL_BITS));
                list.push(p.readback(whole));
                assert_eq!(p.run(list), Ok(()), "{format} {mask:?}");
                lists += 1;
            }
        }
        assert_eq!(p.matched, 2 * lists);
    });
}

#[test]
fn two_drawn_clears_in_one_list_each_write_their_value() {
    bounded(|| {
        let Some(gpu) = backend("two_drawn_clears") else {
            return;
        };
        let mut p = Pair::new(gpu);
        let format = PixelFormat::R32Uint;
        let tex = p.texture(&texture_desc(
            TextureDimension::D2,
            format,
            (8, 4, 1),
            1,
            TARGET,
        ));
        let view = p.view(tex, &whole_view(format, ViewDimension::D2, (0, 1)));
        let whole = region(tex, 0, (0, 0, 0), (8, 4, 1));
        let left = Rect {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        };
        let right = Rect { x: 4, ..left };
        // An empty mask would write nothing, so the red mask is the one that is drawn and still
        // writes the whole of a one-channel format.
        let mut list = vec![
            upload(pattern(level_bytes(format), 37), whole),
            masked(
                view,
                ClearValue::Uint([11, 0, 0, 0]),
                Some(left),
                ColorWrites::RED,
                ALL_STENCIL_BITS,
            ),
            masked(
                view,
                ClearValue::Uint([22, 0, 0, 0]),
                Some(right),
                ColorWrites::RED,
                ALL_STENCIL_BITS,
            ),
        ];
        list.push(p.readback(whole));
        p.run(list).expect("accepted");
        assert_eq!(p.matched, 1);
        // The recording backend's answer, by hand: 11 on the left half, 22 on the right.
        let mut texels = Vec::new();
        for _ in 0..4 {
            for x in 0..8 {
                let v: u32 = if x < 4 { 11 } else { 22 };
                texels.extend(v.to_le_bytes());
            }
        }
        let id = p.readback(whole);
        let Command::Readback { id: read, .. } = id else {
            unreachable!("a readback")
        };
        let fence = p
            .gpu
            .submit(CommandList { commands: vec![id] })
            .expect("accepted");
        finish(&p.gpu, fence);
        assert_eq!(p.gpu.take_readback(read), Some(Readback::Bytes(texels)));
    });
}

#[test]
fn partial_depth_and_masked_stencil_clears_write_what_the_recording_backend_knows() {
    bounded(|| {
        let Some(gpu) = backend("partial_depth_stencil_clears") else {
            return;
        };
        let mut p = Pair::new(gpu);
        // Depth: a whole clear makes every texel known (wgpu takes no upload into D32_FLOAT),
        // then a partial clear is drawn.
        for (format, first, second) in [
            (PixelFormat::D32Float, 1.0, 0.25),
            (PixelFormat::D16Unorm, 1.0, 0.0),
        ] {
            let tex = p.texture(&texture_desc(
                TextureDimension::D2,
                format,
                (8, 4, 1),
                1,
                TARGET,
            ));
            let view = p.view(tex, &whole_view(format, ViewDimension::D2, (0, 1)));
            let whole = region(tex, 0, (0, 0, 0), (8, 4, 1));
            let mut list = vec![
                masked(view, ClearValue::Depth(first), None, ColorWrites::ALL, 0),
                masked(
                    view,
                    ClearValue::Depth(second),
                    Some(PART),
                    ColorWrites::EMPTY,
                    0,
                ),
            ];
            list.push(p.readback(whole));
            assert_eq!(p.run(list), Ok(()), "{format}");
        }
        // Stencil: through a partial mask, over the whole level and over a rectangle.
        let format = PixelFormat::S8Uint;
        let tex = p.texture(&texture_desc(
            TextureDimension::D2,
            format,
            (8, 4, 1),
            1,
            TARGET,
        ));
        let view = p.view(tex, &whole_view(format, ViewDimension::D2, (0, 1)));
        let whole = region(tex, 0, (0, 0, 0), (8, 4, 1));
        let mut list = vec![
            upload(pattern(level_bytes(format), 41), whole),
            masked(
                view,
                ClearValue::Stencil(0x5a),
                None,
                ColorWrites::ALL,
                0x0f,
            ),
        ];
        list.push(p.readback(whole));
        list.push(masked(
            view,
            ClearValue::Stencil(0xa5),
            Some(PART),
            ColorWrites::ALL,
            0xf0,
        ));
        list.push(p.readback(whole));
        // A rectangle through every bit, and an empty mask that writes nothing.
        list.push(masked(
            view,
            ClearValue::Stencil(0x33),
            Some(PART),
            ColorWrites::ALL,
            ALL_STENCIL_BITS,
        ));
        list.push(masked(
            view,
            ClearValue::Stencil(0xff),
            None,
            ColorWrites::ALL,
            0,
        ));
        list.push(p.readback(whole));
        p.run(list).expect("accepted");
        assert_eq!(p.matched, 2 + 3);
    });
}

#[test]
fn drawn_clears_of_layers_and_slices_write_each() {
    bounded(|| {
        let Some(gpu) = backend("drawn_layered_clears") else {
            return;
        };
        let mut p = Pair::new(gpu);
        let format = PixelFormat::R32G32Uint;
        let array = p.texture(&texture_desc(
            TextureDimension::D2,
            format,
            (4, 4, 5),
            1,
            TARGET,
        ));
        let layers = p.view(array, &whole_view(format, ViewDimension::D2Array, (1, 2)));
        let deep = p.texture(&texture_desc(
            TextureDimension::D3,
            format,
            (4, 4, 3),
            1,
            TARGET,
        ));
        let slices = p.view(deep, &whole_view(format, ViewDimension::D3, (0, 1)));
        let value = ClearValue::Uint([0xdead_beef, 0x0bad_f00d, 0, 0]);
        let mut list = vec![
            upload(
                pattern(4 * 4 * 5 * 8, 43),
                region(array, 0, (0, 0, 0), (4, 4, 5)),
            ),
            upload(
                pattern(4 * 4 * 3 * 8, 47),
                region(deep, 0, (0, 0, 0), (4, 4, 3)),
            ),
            masked(
                layers,
                value,
                Some(Rect {
                    x: 1,
                    y: 1,
                    width: 2,
                    height: 3,
                }),
                ColorWrites::GREEN,
                ALL_STENCIL_BITS,
            ),
            masked(slices, value, None, ColorWrites::RED, ALL_STENCIL_BITS),
        ];
        list.push(p.readback(region(array, 0, (0, 0, 0), (4, 4, 5))));
        list.push(p.readback(region(deep, 0, (0, 0, 0), (4, 4, 3))));
        p.run(list).expect("accepted");
        assert_eq!(p.matched, 2);
    });
}

#[test]
fn drawn_clears_wgpu_cannot_read_back_run_with_no_host_error() {
    bounded(|| {
        let Some(gpu) = backend("unreadable_drawn_clears") else {
            return;
        };
        let samples = 4;
        if !gpu.caps().sample_counts.supports(samples) {
            eprintln!("skipping the multisampled part: no 4-sample support");
        }
        let mut p = Pair::new(gpu);
        let mut list = Vec::new();
        if p.gpu.caps().sample_counts.supports(samples) {
            for format in [PixelFormat::A8B8G8R8Unorm, PixelFormat::D32Float] {
                let tex = p.texture(&TextureDesc {
                    sample_count: samples,
                    ..texture_desc(TextureDimension::D2, format, (8, 4, 1), 1, TARGET)
                });
                let view = p.view(tex, &whole_view(format, ViewDimension::D2, (0, 1)));
                let value = if format == PixelFormat::D32Float {
                    ClearValue::Depth(0.5)
                } else {
                    ClearValue::Float([0.0, 1.0, 0.0, 1.0])
                };
                list.push(masked(view, value, Some(PART), ColorWrites::ALL, 0));
                list.push(masked(view, value, None, ColorWrites::BLUE, 0));
            }
        }
        // A two-aspect format: depth and stencil drawn together through a partial mask.
        let format = PixelFormat::D32FloatS8Uint;
        let tex = p.texture(&texture_desc(
            TextureDimension::D2,
            format,
            (8, 4, 1),
            1,
            TARGET,
        ));
        let view = p.view(tex, &whole_view(format, ViewDimension::D2, (0, 1)));
        let both = ClearValue::DepthStencil {
            depth: 0.75,
            stencil: 0x3c,
        };
        list.push(masked(view, both, None, ColorWrites::ALL, 0x0f));
        list.push(masked(view, both, Some(PART), ColorWrites::ALL, 0));
        list.push(masked(
            view,
            ClearValue::Stencil(1),
            Some(PART),
            ColorWrites::ALL,
            0x80,
        ));
        p.run(list).expect("accepted");
    });
}
