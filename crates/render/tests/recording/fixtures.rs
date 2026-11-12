//! Builders shared by the recording backend's tests.

use starquake_render::caps::{DynamicState, StageSupport};
use starquake_render::command::{
    BufferSlice, ClearDesc, ClearValue, CopyRegion, DrawDesc, DrawParams, DynamicValues, Rect,
    TextureRegion, Viewport,
};
use starquake_render::pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorWrites, CullMode, DepthBias,
    DepthStencilState, EmulationFlags, FrontFace, PolygonMode, RasterState, Stages, TargetFormats,
    Topology, VertexLayout,
};
use starquake_render::resource::{Extent3d, Origin3d, Swizzle, TextureDimension, ViewDimension};
use starquake_render::shader::ShaderInterface;
use starquake_render::{
    Backend, BufferDesc, BufferId, BufferUsage, Caps, Command, CommandList, ComputePipelineDesc,
    Features, FenceValue, FormatFeatures, FormatTable, GraphicsPipelineDesc, Limits, PipelineId,
    PixelFormat, Readback, ReadbackId, RecordingBackend, SampleCounts, ShaderDesc, ShaderId,
    ShaderSource, ShaderStage, StagingSlice, TextureDesc, TextureId, TextureUsage, ViewDesc,
    ViewId,
};

/// The value of a call the test expects to succeed. (Test helpers are not `#[test]` functions,
/// so the lint that bans `expect` outside tests applies to them.)
pub fn ok<T, E: std::fmt::Debug>(result: Result<T, E>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => unreachable!("a call the test expects to succeed failed: {error:?}"),
    }
}

/// The budget the tests' backends report.
pub const MEMORY: u64 = 1 << 30;

/// A device that allows everything the tests need.
pub fn caps() -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: 1 << 20,
            max_staging_size: 1 << 16,
            max_texture_dimension_1d: 4096,
            max_texture_dimension_2d: 4096,
            max_texture_dimension_3d: 256,
            max_texture_array_layers: 64,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 16,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 16,
            max_compute_workgroups_per_dimension: 65_535,
            max_uniform_buffers_per_stage: 12,
            max_storage_buffers_per_stage: 8,
            max_sampled_textures_per_stage: 16,
            max_storage_textures_per_stage: 4,
            max_samplers_per_stage: 16,
        },
        features: Features::ALL,
        stages: StageSupport::ALL,
        dynamic_state: DynamicState::ALL,
        sample_counts: SampleCounts::X1 | SampleCounts::X4,
        view_swizzle: false,
        formats: FormatTable::uniform(FormatFeatures::ALL),
    }
}

pub fn backend() -> RecordingBackend {
    RecordingBackend::new(caps(), MEMORY)
}

pub fn buffer(b: &RecordingBackend, size: u64, usage: BufferUsage) -> BufferId {
    ok(b.create_buffer(&BufferDesc { size, usage }))
}

/// Every texture usage.
pub fn all_usage() -> TextureUsage {
    TextureUsage::ALL
}

pub fn texture_desc(
    format: PixelFormat,
    dimension: TextureDimension,
    (width, height, depth_or_layers): (u32, u32, u32),
    mip_levels: u32,
) -> TextureDesc {
    TextureDesc {
        dimension,
        format,
        size: Extent3d {
            width,
            height,
            depth_or_layers,
        },
        mip_levels,
        sample_count: 1,
        usage: all_usage(),
    }
}

/// A 2D texture with `layers` layers and `mips` mips, every usage.
pub fn texture(
    b: &RecordingBackend,
    format: PixelFormat,
    size: (u32, u32),
    layers: u32,
    mips: u32,
) -> TextureId {
    ok(b.create_texture(&texture_desc(
        format,
        TextureDimension::D2,
        (size.0, size.1, layers),
        mips,
    )))
}

/// A 2D array view of one mip and a layer range.
pub fn view(
    b: &RecordingBackend,
    t: TextureId,
    format: PixelFormat,
    mip: u32,
    layers: (u32, u32),
) -> ViewId {
    ok(b.create_view(
        t,
        &ViewDesc {
            format,
            dimension: ViewDimension::D2Array,
            base_mip: mip,
            mip_count: 1,
            base_layer: layers.0,
            layer_count: layers.1,
            swizzle: Swizzle::IDENTITY,
        },
    ))
}

pub fn staged(bytes: &[u8]) -> StagingSlice {
    StagingSlice::from_bytes(bytes.to_vec())
}

pub fn slice(buffer: BufferId, offset: u64, size: u64) -> CopyRegion {
    CopyRegion::Buffer(BufferSlice {
        buffer,
        offset,
        size,
    })
}

pub fn region(
    texture: TextureId,
    mip: u32,
    (x, y, z): (u32, u32, u32),
    (width, height, depth_or_layers): (u32, u32, u32),
) -> CopyRegion {
    CopyRegion::Texture(TextureRegion {
        texture,
        mip,
        origin: Origin3d { x, y, z },
        extent: Extent3d {
            width,
            height,
            depth_or_layers,
        },
    })
}

pub fn rect(x: u32, y: u32, width: u32, height: u32) -> Rect {
    Rect {
        x,
        y,
        width,
        height,
    }
}

pub fn upload(bytes: &[u8], to: CopyRegion) -> Command {
    Command::Upload {
        from: staged(bytes),
        to,
    }
}

pub fn clear(target: ViewId, value: ClearValue, r: Option<Rect>) -> Command {
    Command::Clear(ClearDesc {
        target,
        value,
        rect: r,
    })
}

pub fn submit(b: &RecordingBackend, commands: Vec<Command>) -> FenceValue {
    ok(b.submit(CommandList { commands }))
}

pub fn rb(n: u32) -> ReadbackId {
    ReadbackId::new(n, 0)
}

/// Reads `from` back in a list of its own.
pub fn read(b: &RecordingBackend, from: CopyRegion) -> Readback {
    let id = rb(u32::MAX);
    submit(b, vec![Command::Readback { from, id }]);
    ok(b.take_readback(id).ok_or("the readback is not waiting"))
}

/// Bytes counting up from `start`.
pub fn counting(start: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| start.wrapping_add(ok(u8::try_from(i % 256))))
        .collect()
}

pub fn shader(b: &RecordingBackend, stage: ShaderStage) -> ShaderId {
    ok(b.create_shader(ShaderDesc {
        stage,
        entry: "main".to_owned(),
        source: ShaderSource::SpirV(vec![0x0723_0203]),
        interface: ShaderInterface::default(),
    }))
}

pub fn blend_off() -> BlendState {
    let c = BlendComponent {
        src: BlendFactor::One,
        dst: BlendFactor::Zero,
        op: BlendOperation::Add,
    };
    BlendState {
        enabled: false,
        color: c,
        alpha: c,
        write_mask: ColorWrites::ALL,
    }
}

pub fn graphics_desc(vertex: ShaderId, fragment: Option<ShaderId>) -> GraphicsPipelineDesc {
    let mut colors = [None; 8];
    colors[0] = Some(PixelFormat::A8B8G8R8Unorm);
    GraphicsPipelineDesc {
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
        blend: [blend_off(); 8],
        logic_op: None,
        targets: TargetFormats {
            colors,
            depth_stencil: None,
            sample_count: 1,
        },
        emulation: EmulationFlags::EMPTY,
    }
}

/// A graphics pipeline with a vertex and a fragment shader.
pub fn graphics(b: &RecordingBackend) -> PipelineId {
    let vs = shader(b, ShaderStage::Vertex);
    let fs = shader(b, ShaderStage::Fragment);
    ok(b.create_graphics_pipeline(graphics_desc(vs, Some(fs))))
}

pub fn compute(b: &RecordingBackend) -> PipelineId {
    let cs = shader(b, ShaderStage::Compute);
    ok(b.create_compute_pipeline(ComputePipelineDesc { shader: cs }))
}

pub fn dynamic() -> DynamicValues {
    DynamicValues {
        viewports: vec![Viewport {
            x: 0.0,
            y: 0.0,
            width: 4.0,
            height: 4.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }],
        scissors: Vec::new(),
        blend_constant: [0.0; 4],
        stencil_reference: 0,
        stencil_read_mask: 0xff,
        stencil_write_mask: 0xff,
        depth_bias: DepthBias::default(),
        line_width: 1.0,
    }
}

pub fn draw(pipeline: PipelineId) -> DrawDesc {
    DrawDesc {
        pipeline,
        bindings: Vec::new(),
        vertex_buffers: Vec::new(),
        index_buffer: None,
        dynamic: dynamic(),
        params: DrawParams::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
    }
}
