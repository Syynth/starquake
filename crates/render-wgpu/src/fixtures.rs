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
pub(crate) fn texture(
    format: PixelFormat,
    width: u32,
    height: u32,
    usage: TextureUsage,
) -> TextureDesc {
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

/// A blend state with blending off that writes every channel.
pub(crate) fn no_blend() -> starquake_render::pipeline::BlendState {
    use starquake_render::pipeline::{
        BlendComponent, BlendFactor, BlendOperation, BlendState, ColorWrites,
    };
    let replace = BlendComponent {
        src: BlendFactor::One,
        dst: BlendFactor::Zero,
        op: BlendOperation::Add,
    };
    BlendState {
        enabled: false,
        color: replace,
        alpha: replace,
        write_mask: ColorWrites::RED
            .union(ColorWrites::GREEN)
            .union(ColorWrites::BLUE)
            .union(ColorWrites::ALPHA),
    }
}

/// A graphics pipeline of `vertex` and `fragment` drawing triangles into one `A8B8G8R8Unorm`
/// target, with no vertex buffers, no culling, no depth or stencil and no blending.
pub(crate) fn graphics_desc(
    vertex: starquake_render::ShaderId,
    fragment: Option<starquake_render::ShaderId>,
) -> starquake_render::GraphicsPipelineDesc {
    use starquake_render::caps::GUEST_COLOR_TARGETS;
    use starquake_render::pipeline::{
        CullMode, DepthBias, DepthStencilState, FrontFace, PolygonMode, RasterState, Stages,
        TargetFormats, Topology, VertexLayout,
    };
    let mut colors = [None; GUEST_COLOR_TARGETS];
    colors[0] = Some(PixelFormat::A8B8G8R8Unorm);
    starquake_render::GraphicsPipelineDesc {
        stages: Stages {
            vertex,
            tess_control: None,
            tess_eval: None,
            geometry: None,
            fragment,
        },
        vertex_input: VertexLayout::default(),
        topology: Topology::Triangles,
        primitive_restart: false,
        raster: RasterState {
            cull: CullMode::None,
            front_face: FrontFace::Ccw,
            polygon_mode: PolygonMode::Fill,
            depth_clamp: false,
            depth_bias: DepthBias::default(),
            line_width: 1.0,
        },
        depth_stencil: DepthStencilState {
            depth_test: None,
            depth_write: false,
            stencil: None,
        },
        blend: [no_blend(); GUEST_COLOR_TARGETS],
        logic_op: None,
        targets: TargetFormats {
            colors,
            depth_stencil: None,
            sample_count: 1,
        },
        emulation: starquake_render::EmulationFlags::EMPTY,
    }
}

/// Every `ShaderDesc` this crate's tests make is made here, so that a new field (render capture
/// v2's `origin`) touches one place (render-backend, "Who builds it": "R6: builds its tests'
/// `ShaderDesc`s through one helper").
pub(crate) fn shader_desc(
    stage: starquake_render::ShaderStage,
    entry: &str,
    source: starquake_render::ShaderSource,
    interface: starquake_render::ShaderInterface,
) -> starquake_render::ShaderDesc {
    starquake_render::ShaderDesc {
        stage,
        entry: entry.into(),
        source,
        interface,
    }
}

/// A hand-written WGSL module as naga IR, validated with every check and every capability, as
/// `ShaderSource::Naga` carries it.
pub(crate) fn wgsl(source: &str) -> starquake_render::ShaderSource {
    let module = naga::front::wgsl::parse_str(source).expect("the test module parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("the test module validates");
    starquake_render::ShaderSource::Naga {
        module: Box::new(module),
        info: Box::new(info),
    }
}
