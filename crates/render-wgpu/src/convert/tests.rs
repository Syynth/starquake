//! The descriptions handed to wgpu, field by field.

use super::*;
use crate::fixtures::{sampler, texture, view};
use crate::rules::HostTexture;
use starquake_render::PixelFormat;
use starquake_render::resource::{BorderColor, Extent3d};

#[test]
fn buffer_usages_map_one_to_one() {
    use wgpu::BufferUsages as W;
    let pairs = [
        (BufferUsage::VERTEX, W::VERTEX),
        (BufferUsage::INDEX, W::INDEX),
        (BufferUsage::UNIFORM, W::UNIFORM),
        (BufferUsage::STORAGE, W::STORAGE),
        (BufferUsage::INDIRECT, W::INDIRECT),
        (BufferUsage::COPY_SRC, W::COPY_SRC),
        (BufferUsage::COPY_DST, W::COPY_DST),
    ];
    for mask in 0_u32..(1 << pairs.len()) {
        let mut ours = BufferUsage::EMPTY;
        let mut theirs = W::empty();
        for (i, (o, t)) in pairs.iter().enumerate() {
            if mask & (1 << i) != 0 {
                ours = ours.union(*o);
                theirs |= *t;
            }
        }
        assert_eq!(buffer_usages(ours), theirs, "mask {mask:#b}");
    }
}

#[test]
fn texture_usages_add_the_allowed_copies() {
    use wgpu::TextureUsages as W;
    let pairs = [
        (TextureUsage::SAMPLED, W::TEXTURE_BINDING),
        (TextureUsage::STORAGE, W::STORAGE_BINDING),
        (TextureUsage::RENDER_TARGET, W::RENDER_ATTACHMENT),
        (TextureUsage::COPY_SRC, W::COPY_SRC),
        (TextureUsage::COPY_DST, W::COPY_DST),
    ];
    for mask in 0_u32..(1 << pairs.len()) {
        let mut ours = TextureUsage::EMPTY;
        let mut theirs = W::empty();
        for (i, (o, t)) in pairs.iter().enumerate() {
            if mask & (1 << i) != 0 {
                ours = ours.union(*o);
                theirs |= *t;
            }
        }
        assert_eq!(texture_usages(ours, W::empty()), theirs, "mask {mask:#b}");
        assert_eq!(
            texture_usages(ours, W::all()),
            theirs | W::COPY_SRC | W::COPY_DST,
            "mask {mask:#b}"
        );
        // Only the copy usages are added, and only those allowed.
        assert_eq!(
            texture_usages(ours, W::COPY_SRC | W::STORAGE_BINDING),
            theirs | W::COPY_SRC,
            "mask {mask:#b}"
        );
    }
}

#[test]
fn texture_specs_list_the_srgb_counterpart() {
    let host = HostTexture {
        dimension: wgpu::TextureDimension::D2,
        size: Extent3d {
            width: 8,
            height: 4,
            depth_or_layers: 1,
        },
    };
    let desc = TextureDesc {
        mip_levels: 3,
        sample_count: 1,
        ..texture(PixelFormat::A8B8G8R8Unorm, 8, 4, TextureUsage::SAMPLED)
    };
    let spec = TextureSpec::new(&desc, host, wgpu::TextureUsages::empty(), true);
    assert_eq!(
        spec,
        Some(TextureSpec {
            format: wgpu::TextureFormat::Rgba8Unorm,
            host,
            mip_levels: 3,
            sample_count: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: vec![wgpu::TextureFormat::Rgba8UnormSrgb],
        })
    );
    let srgb = TextureDesc {
        format: PixelFormat::B8G8R8A8Srgb,
        ..desc
    };
    let spec = TextureSpec::new(&srgb, host, wgpu::TextureUsages::empty(), true);
    assert_eq!(
        spec.map(|s| s.view_formats),
        Some(vec![wgpu::TextureFormat::Bgra8Unorm])
    );
    let spec = TextureSpec::new(&srgb, host, wgpu::TextureUsages::empty(), false);
    assert_eq!(spec.map(|s| s.view_formats), Some(vec![]), "cannot reinterpret");
    let plain = TextureDesc {
        format: PixelFormat::R8Unorm,
        ..desc
    };
    let spec = TextureSpec::new(&plain, host, wgpu::TextureUsages::empty(), true);
    assert_eq!(spec.map(|s| s.view_formats), Some(vec![]), "no counterpart");
    let packed = TextureDesc {
        format: PixelFormat::R5G6B5Unorm,
        ..desc
    };
    assert_eq!(TextureSpec::new(&packed, host, wgpu::TextureUsages::empty(), true), None);
}

#[test]
fn view_descriptors_carry_every_field() {
    let v = ViewDesc {
        base_mip: 2,
        mip_count: 3,
        base_layer: 4,
        layer_count: 5,
        ..view(PixelFormat::A8B8G8R8Srgb, ViewDimension::D2Array)
    };
    let d = view_descriptor(&v).expect("a 2D array view");
    assert_eq!(d.format, Some(wgpu::TextureFormat::Rgba8UnormSrgb));
    assert_eq!(d.dimension, Some(wgpu::TextureViewDimension::D2Array));
    assert_eq!(
        (d.base_mip_level, d.mip_level_count, d.base_array_layer, d.array_layer_count),
        (2, Some(3), 4, Some(5))
    );
    assert_eq!(d.aspect, wgpu::TextureAspect::All);
    assert_eq!(d.usage, None);
    for (ours, theirs) in [
        (ViewDimension::D1, Some(wgpu::TextureViewDimension::D1)),
        (ViewDimension::D1Array, None),
        (ViewDimension::D2, Some(wgpu::TextureViewDimension::D2)),
        (ViewDimension::Cube, Some(wgpu::TextureViewDimension::Cube)),
        (ViewDimension::CubeArray, Some(wgpu::TextureViewDimension::CubeArray)),
        (ViewDimension::D3, Some(wgpu::TextureViewDimension::D3)),
    ] {
        let v = view(PixelFormat::R8Unorm, ours);
        assert_eq!(view_descriptor(&v).and_then(|d| d.dimension), theirs, "{ours:?}");
    }
    assert!(view_descriptor(&view(PixelFormat::R5G6B5Unorm, ViewDimension::D2)).is_none());
}

#[test]
fn sampler_descriptors_follow_the_module_rules() {
    let mut s = SamplerDesc {
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Nearest,
        mip_filter: FilterMode::Linear,
        address: [
            AddressMode::Repeat,
            AddressMode::MirrorRepeat,
            AddressMode::ClampToBorder,
        ],
        border_color: BorderColor::OpaqueWhite,
        lod_bias: 1.5,
        min_lod: -2.0,
        max_lod: 7.0,
        max_anisotropy: 8,
        compare: Some(CompareFunction::GreaterEqual),
        ..sampler()
    };
    let d = sampler_descriptor(&s);
    assert_eq!(
        (d.address_mode_u, d.address_mode_v, d.address_mode_w),
        (
            wgpu::AddressMode::Repeat,
            wgpu::AddressMode::MirrorRepeat,
            wgpu::AddressMode::ClampToBorder
        )
    );
    assert_eq!(
        (d.mag_filter, d.min_filter, d.mipmap_filter),
        (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Nearest,
            wgpu::MipmapFilterMode::Linear
        )
    );
    assert_eq!((d.lod_min_clamp, d.lod_max_clamp), (0.0, 7.0), "raised to zero");
    assert_eq!(d.anisotropy_clamp, 1, "a nearest filter drops anisotropy");
    assert_eq!(d.compare, Some(wgpu::CompareFunction::GreaterEqual));
    assert_eq!(d.border_color, Some(wgpu::SamplerBorderColor::OpaqueWhite));
    s.min_filter = FilterMode::Linear;
    assert_eq!(sampler_descriptor(&s).anisotropy_clamp, 8, "all linear keeps it");
    s.mip_filter = FilterMode::Nearest;
    assert_eq!(sampler_descriptor(&s).anisotropy_clamp, 1, "the mip filter counts too");
    s.address = [AddressMode::ClampToEdge; 3];
    assert_eq!(sampler_descriptor(&s).border_color, None, "no border read");
    s.max_lod = -1.0;
    assert_eq!(sampler_descriptor(&s).lod_max_clamp, 0.0);
    for (ours, theirs) in [
        (CompareFunction::Never, wgpu::CompareFunction::Never),
        (CompareFunction::Less, wgpu::CompareFunction::Less),
        (CompareFunction::Equal, wgpu::CompareFunction::Equal),
        (CompareFunction::LessEqual, wgpu::CompareFunction::LessEqual),
        (CompareFunction::Greater, wgpu::CompareFunction::Greater),
        (CompareFunction::NotEqual, wgpu::CompareFunction::NotEqual),
        (CompareFunction::GreaterEqual, wgpu::CompareFunction::GreaterEqual),
        (CompareFunction::Always, wgpu::CompareFunction::Always),
    ] {
        s.compare = Some(ours);
        assert_eq!(sampler_descriptor(&s).compare, Some(theirs));
    }
    s.compare = None;
    assert_eq!(sampler_descriptor(&s).compare, None);
}
