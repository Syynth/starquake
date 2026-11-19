//! Shared test values: a made-up device that can do everything with every format, and
//! descriptions built from it.

use crate::caps::{ByteLimits, FormatSupport, caps};
use starquake_render::resource::{
    AddressMode, BorderColor, Extent3d, FilterMode, ReductionMode, Swizzle, TextureDimension,
    ViewDimension,
};
use starquake_render::{Caps, PixelFormat, SamplerDesc, TextureDesc, TextureUsage, ViewDesc};

/// A device whose every format allows every usage and up to four samples.
pub(crate) fn formats() -> FormatSupport {
    FormatSupport::new(wgpu::Features::all(), |_| wgpu::TextureFormatFeatures {
        allowed_usages: wgpu::TextureUsages::all(),
        flags: wgpu::TextureFormatFeatureFlags::all()
            .difference(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X8)
            .difference(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X16),
    })
}

/// `Caps` for that device with wgpu's default limits and room for everything.
pub(crate) fn test_caps() -> Caps {
    caps(
        wgpu::Features::all(),
        &wgpu::Limits::default(),
        wgpu::DownlevelFlags::all(),
        &formats(),
        ByteLimits {
            memory: u64::MAX,
            staging: u64::MAX,
        },
    )
}

/// A one-level, one-sample 2D texture of `format`, `width` by `height`.
pub(crate) fn texture(format: PixelFormat, width: u32, height: u32, usage: TextureUsage) -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        format,
        size: Extent3d {
            width,
            height,
            depth_or_layers: 1,
        },
        mip_levels: 1,
        sample_count: 1,
        usage,
    }
}

/// A view of the whole of a one-level, one-layer texture, in `format`.
pub(crate) fn view(format: PixelFormat, dimension: ViewDimension) -> ViewDesc {
    ViewDesc {
        format,
        dimension,
        base_mip: 0,
        mip_count: 1,
        base_layer: 0,
        layer_count: 1,
        swizzle: Swizzle::IDENTITY,
    }
}

/// A plain sampler.
pub(crate) fn sampler() -> SamplerDesc {
    SamplerDesc {
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mip_filter: FilterMode::Linear,
        address: [AddressMode::Repeat; 3],
        border_color: BorderColor::TransparentBlack,
        lod_bias: 0.0,
        min_lod: 0.0,
        max_lod: 8.0,
        max_anisotropy: 1,
        compare: None,
        reduction: ReductionMode::Average,
    }
}
