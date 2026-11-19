//! Each wgpu rule: its refusal, the case one step inside it, and the documented order.

use super::*;
use crate::fixtures::{formats, sampler, test_caps, texture, view};
use starquake_render::command::Rect;
use starquake_render::resource::{Origin3d, TextureDimension, ViewDimension};
use starquake_render::{Caps, PixelFormat, RenderError, TextureUsage};

const SAMPLED: TextureUsage = TextureUsage::SAMPLED;

fn check(desc: &TextureDesc) -> Result<HostTexture, RenderError> {
    check_texture(&test_caps(), &formats(), wgpu::Features::all(), desc)
}

fn refused(message: &'static str) -> RenderError {
    RenderError::Unsupported(message)
}

fn one_d(layers: u32, mips: u32, format: PixelFormat, usage: TextureUsage) -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D1,
        size: Extent3d {
            width: 16,
            height: 1,
            depth_or_layers: layers,
        },
        mip_levels: mips,
        ..texture(format, 16, 1, usage)
    }
}

#[test]
fn one_d_textures_follow_wgpu_in_order() {
    let ok = one_d(1, 1, PixelFormat::R8Unorm, SAMPLED);
    assert_eq!(
        check(&ok).map(|h| (h.dimension, h.size)),
        Ok((
            wgpu::TextureDimension::D1,
            Extent3d {
                width: 16,
                height: 1,
                depth_or_layers: 1
            }
        ))
    );
    // Each fault alone, then two at once: the first in the documented order wins.
    assert_eq!(check(&one_d(2, 1, PixelFormat::R8Unorm, SAMPLED)), Err(refused(ONE_D_LAYERS)));
    assert_eq!(check(&one_d(1, 2, PixelFormat::R8Unorm, SAMPLED)), Err(refused(ONE_D_MIPS)));
    assert_eq!(check(&one_d(2, 2, PixelFormat::R8Unorm, SAMPLED)), Err(refused(ONE_D_LAYERS)));
    for format in [PixelFormat::Bc1RgbaUnorm, PixelFormat::D16Unorm, PixelFormat::S8Uint] {
        assert_eq!(check(&one_d(1, 1, format, SAMPLED)), Err(refused(ONE_D_FORMAT)), "{format}");
        assert_eq!(check(&one_d(1, 2, format, SAMPLED)), Err(refused(ONE_D_MIPS)), "{format}");
    }
    let target = TextureUsage::RENDER_TARGET;
    assert_eq!(check(&one_d(1, 1, PixelFormat::R8Unorm, target)), Err(refused(ONE_D_TARGET)));
    assert_eq!(
        check(&one_d(1, 1, PixelFormat::D16Unorm, target)),
        Err(refused(ONE_D_FORMAT))
    );
}

#[test]
fn depth_and_stencil_textures_are_flat() {
    for format in [PixelFormat::D32Float, PixelFormat::S8Uint, PixelFormat::D32FloatS8Uint] {
        let flat = texture(format, 8, 8, SAMPLED);
        assert!(check(&flat).is_ok(), "{format}");
        let cube = TextureDesc {
            dimension: TextureDimension::Cube,
            size: Extent3d {
                width: 8,
                height: 8,
                depth_or_layers: 6,
            },
            ..flat
        };
        assert!(check(&cube).is_ok(), "{format} cube");
        let deep = TextureDesc {
            dimension: TextureDimension::D3,
            ..flat
        };
        assert_eq!(check(&deep), Err(refused(DEPTH_DIMENSION)), "{format}");
    }
}

#[test]
fn compressed_3d_textures_need_their_feature() {
    let deep = |format| TextureDesc {
        dimension: TextureDimension::D3,
        ..texture(format, 8, 8, SAMPLED)
    };
    let caps = test_caps();
    let at = |features, format| check_texture(&caps, &formats(), features, &deep(format));
    let bc = wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D;
    let astc = wgpu::Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D;
    assert!(at(bc, PixelFormat::Bc1RgbaUnorm).is_ok());
    assert_eq!(at(astc, PixelFormat::Bc1RgbaUnorm), Err(refused(COMPRESSED_3D)));
    assert!(at(astc, PixelFormat::Astc2d8x8Unorm).is_ok());
    assert_eq!(at(bc, PixelFormat::Astc2d8x8Unorm), Err(refused(COMPRESSED_3D)));
    assert_eq!(
        at(wgpu::Features::all(), PixelFormat::Etc2RgbUnorm),
        Err(refused(COMPRESSED_3D))
    );
    assert!(at(wgpu::Features::empty(), PixelFormat::R8Unorm).is_ok());
}

#[test]
fn multisampled_textures_follow_wgpu_in_order() {
    let target = TextureUsage::RENDER_TARGET;
    let ms = |layers, usage, samples| TextureDesc {
        size: Extent3d {
            width: 8,
            height: 8,
            depth_or_layers: layers,
        },
        sample_count: samples,
        ..texture(PixelFormat::A8B8G8R8Unorm, 8, 8, usage)
    };
    assert!(check(&ms(1, target, 4)).is_ok());
    assert_eq!(check(&ms(2, target, 4)), Err(refused(MULTISAMPLED_LAYERS)));
    assert_eq!(check(&ms(1, SAMPLED, 4)), Err(refused(MULTISAMPLED_TARGET)));
    let storage = target.union(TextureUsage::STORAGE);
    assert_eq!(check(&ms(1, storage, 4)), Err(refused(MULTISAMPLED_STORAGE)));
    // The fixture's formats allow up to four samples: eight is the format's refusal.
    assert_eq!(check(&ms(1, target, 8)), Err(RenderError::SampleCount(8)));
    // Two faults: the layer count is reported first.
    assert_eq!(check(&ms(2, SAMPLED, 8)), Err(refused(MULTISAMPLED_LAYERS)));
    assert_eq!(check(&ms(1, SAMPLED, 8)), Err(refused(MULTISAMPLED_TARGET)));
    // One sample is never checked against the format.
    assert!(check(&ms(1, SAMPLED, 1)).is_ok());
}

#[test]
fn compressed_sizes_round_up_to_whole_blocks_within_the_limit() {
    let caps = test_caps();
    let limit = caps.limits.max_texture_dimension_2d;
    let at = |format: PixelFormat, w, h| {
        check_texture(&caps, &formats(), wgpu::Features::all(), &texture(format, w, h, SAMPLED))
            .map(|host| (host.size.width, host.size.height))
    };
    assert_eq!(at(PixelFormat::Bc1RgbaUnorm, 5, 9), Ok((8, 12)));
    assert_eq!(at(PixelFormat::Astc2d10x8Unorm, 1, 1), Ok((10, 8)));
    assert_eq!(at(PixelFormat::R8Unorm, 5, 9), Ok((5, 9)), "uncompressed: unchanged");
    // 12x12 blocks: the last multiple of 12 within the limit fits, one texel more does not.
    let last = limit - limit % 12;
    assert_eq!(at(PixelFormat::Astc2d12x12Unorm, last, 12), Ok((last, 12)));
    assert_eq!(at(PixelFormat::Astc2d12x12Unorm, last + 1, 12), Err(refused(PADDED_SIZE)));
    assert_eq!(at(PixelFormat::Astc2d12x12Unorm, 12, last + 1), Err(refused(PADDED_SIZE)));
    // A size that would overflow when rounded up.
    let huge = Caps {
        limits: starquake_render::Limits {
            max_texture_dimension_2d: u32::MAX,
            ..caps.limits
        },
        ..caps.clone()
    };
    let desc = texture(PixelFormat::Astc2d12x12Unorm, u32::MAX, 12, SAMPLED);
    assert_eq!(
        check_texture(&huge, &formats(), wgpu::Features::all(), &desc),
        Err(refused(PADDED_SIZE))
    );
}

#[test]
fn views_follow_wgpu() {
    let tex = texture(PixelFormat::A8B8G8R8Unorm, 8, 8, SAMPLED);
    let v = |format, dimension| view(format, dimension);
    assert_eq!(check_view(&tex, &v(PixelFormat::A8B8G8R8Unorm, ViewDimension::D2), false), Ok(()));
    assert_eq!(check_view(&tex, &v(PixelFormat::A8B8G8R8Srgb, ViewDimension::D2), true), Ok(()));
    assert_eq!(
        check_view(&tex, &v(PixelFormat::A8B8G8R8Srgb, ViewDimension::D2), false),
        Err(refused(VIEW_FORMAT)),
        "a device that cannot reinterpret"
    );
    assert_eq!(
        check_view(&tex, &v(PixelFormat::A8B8G8R8Uint, ViewDimension::D2), true),
        Err(refused(VIEW_FORMAT)),
        "more than sRGB differs"
    );
    let packed = texture(PixelFormat::R5G6B5Unorm, 8, 8, SAMPLED);
    assert_eq!(
        check_view(&packed, &v(PixelFormat::R16Unorm, ViewDimension::D2), true),
        Err(refused(VIEW_FORMAT)),
        "no wgpu form"
    );
    let line = one_d(1, 1, PixelFormat::R8Unorm, SAMPLED);
    assert_eq!(
        check_view(&line, &v(PixelFormat::R8Unorm, ViewDimension::D1Array), true),
        Err(refused(ONE_D_ARRAY_VIEW))
    );
    assert_eq!(check_view(&line, &v(PixelFormat::R8Unorm, ViewDimension::D1), true), Ok(()));
    let ms = TextureDesc {
        sample_count: 4,
        ..tex
    };
    assert_eq!(check_view(&ms, &v(PixelFormat::A8B8G8R8Unorm, ViewDimension::D2), true), Ok(()));
    assert_eq!(
        check_view(&ms, &v(PixelFormat::A8B8G8R8Unorm, ViewDimension::D2Array), true),
        Err(refused(MULTISAMPLED_VIEW))
    );
}

#[test]
fn buffer_ranges_are_aligned_at_both_ends() {
    assert_eq!(check_buffer_range(0, 4), Ok(()));
    assert_eq!(check_buffer_range(8, 12), Ok(()));
    for (offset, size) in [(1, 4), (3, 4), (0, 5), (0, 3), (4, 6), (2, 2)] {
        assert_eq!(
            check_buffer_range(offset, size),
            Err(refused(BUFFER_ALIGNMENT)),
            "{offset} {size}"
        );
    }
}

#[test]
fn buffer_sides_of_texture_copies_are_colour_on_block_boundaries() {
    let cases = [
        (PixelFormat::A8B8G8R8Unorm, 4, Ok(())),
        (PixelFormat::A8B8G8R8Unorm, 2, Err(refused(BLOCK_OFFSET))),
        (PixelFormat::A8B8G8R8Unorm, 5, Err(refused(BLOCK_OFFSET))),
        (PixelFormat::R8Unorm, 1, Ok(())),
        (PixelFormat::Bc1RgbaUnorm, 8, Ok(())),
        (PixelFormat::Bc1RgbaUnorm, 4, Err(refused(BLOCK_OFFSET))),
        (PixelFormat::R32G32B32A32Float, 16, Ok(())),
        (PixelFormat::R32G32B32A32Float, 8, Err(refused(BLOCK_OFFSET))),
        (PixelFormat::D16Unorm, 0, Err(refused(DEPTH_BUFFER_COPY))),
        (PixelFormat::S8Uint, 0, Err(refused(DEPTH_BUFFER_COPY))),
        // Both faults: depth is reported first.
        (PixelFormat::D32Float, 1, Err(refused(DEPTH_BUFFER_COPY))),
    ];
    for (format, offset, expected) in cases {
        assert_eq!(check_buffer_texture(format, offset), expected, "{format} {offset}");
    }
}

fn region(texture: u32, x: u32, width: u32, height: u32) -> TextureRegion {
    TextureRegion {
        texture: starquake_render::TextureId::new(texture, 0),
        mip: 0,
        origin: Origin3d { x, y: 0, z: 0 },
        extent: Extent3d {
            width,
            height,
            depth_or_layers: 1,
        },
    }
}

#[test]
fn texture_copies_follow_wgpu() {
    let rgba = texture(PixelFormat::A8B8G8R8Unorm, 8, 8, SAMPLED);
    let srgb = texture(PixelFormat::A8B8G8R8Srgb, 8, 8, SAMPLED);
    let uint = texture(PixelFormat::A8B8G8R8Uint, 8, 8, SAMPLED);
    // Regions of texture 0 and texture 1: copies between two textures.
    let part = region(0, 0, 4, 4);
    let part_b = region(1, 4, 4, 4);
    let whole = region(0, 0, 8, 8);
    let whole_b = region(1, 0, 8, 8);
    assert_eq!(check_texture_copy(&rgba, &part, &srgb, &part_b), Ok(()));
    assert_eq!(check_texture_copy(&rgba, &part, &uint, &part_b), Err(refused(COPY_FORMATS)));
    let depth = texture(PixelFormat::D32Float, 8, 8, SAMPLED);
    assert_eq!(check_texture_copy(&depth, &whole, &depth, &whole_b), Ok(()));
    assert_eq!(check_texture_copy(&depth, &part, &depth, &whole_b), Err(refused(WHOLE_COPY)));
    assert_eq!(check_texture_copy(&depth, &whole, &depth, &part_b), Err(refused(WHOLE_COPY)));
    let shifted = region(1, 1, 7, 8);
    assert_eq!(check_texture_copy(&depth, &whole, &depth, &shifted), Err(refused(WHOLE_COPY)));
    let ms = TextureDesc {
        sample_count: 4,
        ..rgba
    };
    assert_eq!(check_texture_copy(&ms, &whole, &ms, &whole_b), Ok(()));
    assert_eq!(check_texture_copy(&ms, &part, &ms, &part_b), Err(refused(WHOLE_COPY)));
    // Formats are checked before extents.
    assert_eq!(check_texture_copy(&depth, &part, &rgba, &part_b), Err(refused(COPY_FORMATS)));
    // A mip level the description lacks is never whole.
    let deep = TextureRegion { mip: 3, ..whole };
    assert_eq!(check_texture_copy(&depth, &deep, &depth, &whole_b), Err(refused(WHOLE_COPY)));
}

#[test]
fn copies_within_one_texture_need_another_level_or_layers() {
    let array = TextureDesc {
        size: Extent3d {
            width: 8,
            height: 8,
            depth_or_layers: 4,
        },
        mip_levels: 2,
        ..texture(PixelFormat::A8B8G8R8Unorm, 8, 8, SAMPLED)
    };
    let at = |mip, x, z, layers| TextureRegion {
        mip,
        origin: Origin3d { x, y: 0, z },
        extent: Extent3d {
            width: 2,
            height: 2,
            depth_or_layers: layers,
        },
        ..region(0, 0, 1, 1)
    };
    // Same level, regions apart in x but layers shared: wgpu refuses it anyway.
    assert_eq!(
        check_texture_copy(&array, &at(0, 0, 0, 2), &array, &at(0, 4, 1, 2)),
        Err(refused(SAME_LEVEL))
    );
    assert_eq!(
        check_texture_copy(&array, &at(0, 0, 2, 2), &array, &at(0, 4, 0, 3)),
        Err(refused(SAME_LEVEL)),
        "overlapping by one layer, the other way round"
    );
    // Layer ranges that touch at the end bound, a different mip level, another texture.
    assert_eq!(check_texture_copy(&array, &at(0, 0, 0, 2), &array, &at(0, 0, 2, 2)), Ok(()));
    assert_eq!(check_texture_copy(&array, &at(0, 0, 2, 2), &array, &at(0, 0, 0, 2)), Ok(()));
    assert_eq!(check_texture_copy(&array, &at(0, 0, 0, 2), &array, &at(1, 0, 0, 2)), Ok(()));
    let other = TextureRegion {
        texture: starquake_render::TextureId::new(1, 0),
        ..at(0, 4, 0, 2)
    };
    assert_eq!(check_texture_copy(&array, &at(0, 0, 0, 2), &array, &other), Ok(()));
    let same_index = TextureRegion {
        texture: starquake_render::TextureId::new(0, 1),
        ..at(0, 4, 0, 2)
    };
    assert_eq!(
        check_texture_copy(&array, &at(0, 0, 0, 2), &array, &same_index),
        Ok(()),
        "an id differing only in its generation is another texture"
    );
    // Formats first, then the level, then the whole-level rule.
    let uint = TextureDesc {
        format: PixelFormat::A8B8G8R8Uint,
        ..array
    };
    assert_eq!(
        check_texture_copy(&array, &at(0, 0, 0, 2), &uint, &at(0, 4, 0, 2)),
        Err(refused(COPY_FORMATS))
    );
    let depth = TextureDesc {
        format: PixelFormat::D32Float,
        ..array
    };
    assert_eq!(
        check_texture_copy(&depth, &at(0, 0, 0, 2), &depth, &at(0, 4, 0, 2)),
        Err(refused(SAME_LEVEL))
    );
}

#[test]
fn buffer_copies_are_between_two_buffers() {
    let a = BufferId::new(0, 0);
    assert_eq!(check_buffer_copy(a, a), Err(refused(SAME_BUFFER)));
    assert_eq!(check_buffer_copy(a, BufferId::new(1, 0)), Ok(()));
    assert_eq!(check_buffer_copy(a, BufferId::new(0, 1)), Ok(()), "another generation");
}

#[test]
fn partial_clears_are_colour_and_single_sampled() {
    let rect = |x, y, w, h| Rect {
        x,
        y,
        width: w,
        height: h,
    };
    let rgba = texture(PixelFormat::A8B8G8R8Unorm, 8, 8, SAMPLED);
    let depth = texture(PixelFormat::D32Float, 8, 8, SAMPLED);
    let ms = TextureDesc {
        sample_count: 4,
        ..rgba
    };
    let part = rect(1, 0, 7, 8);
    let full = rect(0, 0, 8, 8);
    assert_eq!(check_clear_area(&rgba, rgba.format, Some(&part), 8, 8), Ok(()));
    assert_eq!(check_clear_area(&depth, depth.format, Some(&full), 8, 8), Ok(()));
    assert_eq!(check_clear_area(&depth, depth.format, None, 8, 8), Ok(()));
    assert_eq!(
        check_clear_area(&depth, depth.format, Some(&part), 8, 8),
        Err(refused(WHOLE_CLEAR))
    );
    for edge in [rect(0, 1, 8, 7), rect(0, 0, 7, 8), rect(0, 0, 8, 7)] {
        assert_eq!(
            check_clear_area(&ms, ms.format, Some(&edge), 8, 8),
            Err(refused(WHOLE_CLEAR)),
            "{edge:?}"
        );
    }
    assert_eq!(check_clear_area(&ms, ms.format, Some(&full), 8, 8), Ok(()));
}

#[test]
fn border_colours_wgpu_has() {
    use wgpu::SamplerBorderColor as W;
    let cases = [
        (BorderColor::TransparentBlack, Some(W::TransparentBlack)),
        (BorderColor::OpaqueBlack, Some(W::OpaqueBlack)),
        (BorderColor::OpaqueWhite, Some(W::OpaqueWhite)),
        (BorderColor::Custom([0.0, 0.0, 0.0, 0.0]), Some(W::TransparentBlack)),
        (BorderColor::Custom([0.0, 0.0, 0.0, 1.0]), Some(W::OpaqueBlack)),
        (BorderColor::Custom([1.0, 1.0, 1.0, 1.0]), Some(W::OpaqueWhite)),
        (BorderColor::Custom([1.0, 1.0, 1.0, 0.0]), None),
        (BorderColor::Custom([0.5, 0.0, 0.0, 1.0]), None),
    ];
    for (color, expected) in cases {
        assert_eq!(border_color(color), expected, "{color:?}");
    }
    let custom = SamplerDesc {
        border_color: BorderColor::Custom([0.5, 0.5, 0.5, 1.0]),
        ..sampler()
    };
    assert_eq!(check_sampler(&custom), Ok(()), "no address mode reads the border");
    let mut clamped = custom;
    clamped.address[2] = AddressMode::ClampToBorder;
    assert_eq!(check_sampler(&clamped), Err(refused(BORDER_COLOR)));
    clamped.border_color = BorderColor::OpaqueWhite;
    assert_eq!(check_sampler(&clamped), Ok(()));
}

#[test]
fn host_buffers_round_up_to_the_copy_alignment() {
    assert_eq!(host_buffer_size(1), Some(4));
    assert_eq!(host_buffer_size(4), Some(4));
    assert_eq!(host_buffer_size(5), Some(8));
    assert_eq!(host_buffer_size(u64::MAX - 3), Some(u64::MAX - 3));
    assert_eq!(host_buffer_size(u64::MAX - 2), None);
}
