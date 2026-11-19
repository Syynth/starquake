//! Tests of the pipeline-key cache, through the recording backend (slice G23).

mod list;

use super::key::{FixedState, GraphicsKey, PipelineKey};
use super::{
    CANONICAL_DEPTH_BIAS, CANONICAL_LINE_WIDTH, CANONICAL_STENCIL_MASK, FailCause, Failure,
    KEYED_DYNAMIC_STATE, KeyList, Lookup, MAX_PIPELINES, PipelineCache, PipelineCacheError,
    Precompiled, ShaderStamp, StageStamps,
};
use starquake_render::caps::{
    Caps, DynamicState, Features, FormatFeatures, FormatTable, Limits, SampleCounts, StageSupport,
};
use starquake_render::command::{CommandList, StagingSlice};
use starquake_render::pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorWrites, ComputePipelineDesc,
    CullMode, DepthBias, DepthStencilState, EmulationFlags, FrontFace, GraphicsPipelineDesc,
    LogicOp, PolygonMode, RasterState, Stages, StencilFace, StencilOperation, StencilState,
    TargetFormats, Topology, VertexAttribute, VertexBufferLayout, VertexComponents, VertexKind,
    VertexLayout, VertexStepMode,
};
use starquake_render::resource::{BufferDesc, CompareFunction, SamplerDesc, TextureDesc, ViewDesc};
use starquake_render::{
    Backend, BufferId, FenceValue, MemoryBudget, PipelineId, PipelineStatus, PixelFormat,
    PresentFeedback, PresentTarget, Readback, ReadbackId, RecordingBackend, RenderError,
    ResourceId, SamplerId, ShaderDesc, ShaderId, ShaderInterface, ShaderSource, ShaderStage,
    TextureId, ViewId, WaitOutcome,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// A device with every feature, stage and format, one and four samples, and `dynamic` set per
/// draw.
fn caps(dynamic: DynamicState) -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: 1 << 16,
            max_staging_size: 1 << 12,
            max_texture_dimension_1d: 256,
            max_texture_dimension_2d: 256,
            max_texture_dimension_3d: 64,
            max_texture_array_layers: 8,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 1,
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
        dynamic_state: dynamic,
        sample_counts: SampleCounts::X1 | SampleCounts::X4,
        view_swizzle: true,
        formats: FormatTable::uniform(FormatFeatures::ALL),
    }
}

/// The recording backend, with every pipeline description it is handed and every status
/// question it is asked counted. Everything is passed on unchanged, so it keeps the recording
/// backend's contract.
struct Spy {
    inner: RecordingBackend,
    graphics: Mutex<Vec<GraphicsPipelineDesc>>,
    compute: Mutex<Vec<ComputePipelineDesc>>,
    statuses: AtomicUsize,
}

impl Spy {
    fn new(caps: Caps) -> Arc<Self> {
        Arc::new(Self {
            inner: RecordingBackend::new(caps, 0),
            graphics: Mutex::new(Vec::new()),
            compute: Mutex::new(Vec::new()),
            statuses: AtomicUsize::new(0),
        })
    }

    fn graphics(&self) -> Vec<GraphicsPipelineDesc> {
        self.graphics.lock().unwrap().clone()
    }

    fn compute(&self) -> Vec<ComputePipelineDesc> {
        self.compute.lock().unwrap().clone()
    }

    fn statuses(&self) -> usize {
        self.statuses.load(Ordering::SeqCst)
    }

    fn shader(&self, stage: ShaderStage) -> ShaderId {
        self.inner
            .create_shader(ShaderDesc {
                stage,
                entry: "main".to_owned(),
                source: ShaderSource::Msl(String::new()),
                interface: ShaderInterface::default(),
                origin: None,
            })
            .unwrap()
    }
}

impl Backend for Spy {
    fn caps(&self) -> &Caps {
        self.inner.caps()
    }
    fn budget(&self) -> MemoryBudget {
        self.inner.budget()
    }
    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferId, RenderError> {
        self.inner.create_buffer(desc)
    }
    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureId, RenderError> {
        self.inner.create_texture(desc)
    }
    fn create_view(&self, texture: TextureId, desc: &ViewDesc) -> Result<ViewId, RenderError> {
        self.inner.create_view(texture, desc)
    }
    fn create_sampler(&self, desc: &SamplerDesc) -> Result<SamplerId, RenderError> {
        self.inner.create_sampler(desc)
    }
    fn create_shader(&self, desc: ShaderDesc) -> Result<ShaderId, RenderError> {
        self.inner.create_shader(desc)
    }
    fn destroy(&self, resource: ResourceId, after: FenceValue) {
        self.inner.destroy(resource, after);
    }
    fn create_graphics_pipeline(
        &self,
        desc: GraphicsPipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        self.graphics.lock().unwrap().push(desc.clone());
        self.inner.create_graphics_pipeline(desc)
    }
    fn create_compute_pipeline(
        &self,
        desc: ComputePipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        self.compute.lock().unwrap().push(desc);
        self.inner.create_compute_pipeline(desc)
    }
    fn pipeline_status(&self, id: PipelineId) -> PipelineStatus {
        self.statuses.fetch_add(1, Ordering::SeqCst);
        self.inner.pipeline_status(id)
    }
    fn staging(&self, size: u64) -> Result<StagingSlice, RenderError> {
        self.inner.staging(size)
    }
    fn submit(&self, list: CommandList) -> Result<FenceValue, RenderError> {
        self.inner.submit(list)
    }
    fn completed(&self) -> FenceValue {
        self.inner.completed()
    }
    fn wait(&self, value: FenceValue, timeout: Duration) -> Result<WaitOutcome, RenderError> {
        self.inner.wait(value, timeout)
    }
    fn take_readback(&self, id: ReadbackId) -> Option<Readback> {
        self.inner.take_readback(id)
    }
    fn set_present_target(&self, target: PresentTarget) -> Result<(), RenderError> {
        self.inner.set_present_target(target)
    }
    fn take_present_feedback(&self) -> Vec<PresentFeedback> {
        self.inner.take_present_feedback()
    }
}

/// The id of a lookup that made a new pipeline, which is compiling.
fn compiling(lookup: Lookup) -> PipelineId {
    match lookup {
        Lookup::Compiling(id) => id,
        other => unreachable!("a new pipeline is compiling, not {other:?}"),
    }
}

/// A change to a description.
type Change = fn(&mut GraphicsPipelineDesc);

/// A change to a description that may name another shader.
type ShaderChange = fn(&mut GraphicsPipelineDesc, ShaderId);

/// A change to a description and its stamps.
type StampCase = fn(&mut GraphicsPipelineDesc, &mut StageStamps);

/// A backend with a vertex, a fragment and a compute shader, and a cache over it.
struct Rig {
    spy: Arc<Spy>,
    cache: PipelineCache,
    vs: ShaderId,
    fs: ShaderId,
    cs: ShaderId,
}

impl Rig {
    fn new(dynamic: DynamicState) -> Self {
        Self::with_caps(caps(dynamic))
    }

    fn with_caps(caps: Caps) -> Self {
        Self::build(caps, MAX_PIPELINES)
    }

    fn build(caps: Caps, limit: usize) -> Self {
        let spy = Spy::new(caps);
        // A shader nothing uses first, so no id the tests use is the first one handed out.
        spy.shader(ShaderStage::Compute);
        let vs = spy.shader(ShaderStage::Vertex);
        let fs = spy.shader(ShaderStage::Fragment);
        let cs = spy.shader(ShaderStage::Compute);
        let backend: Arc<dyn Backend> = Arc::<Spy>::clone(&spy);
        let cache = PipelineCache::with_limit(backend, limit);
        Self {
            spy,
            cache,
            vs,
            fs,
            cs,
        }
    }

    fn desc(&self) -> GraphicsPipelineDesc {
        desc(self.vs, self.fs)
    }

    fn graphics(&mut self, desc: &GraphicsPipelineDesc) -> Lookup {
        self.cache.lookup_graphics(desc, &stamps()).unwrap()
    }

    fn compute(&mut self, stamp: ShaderStamp) -> Lookup {
        self.cache
            .lookup_compute(ComputePipelineDesc { shader: self.cs }, stamp)
            .unwrap()
    }
}

const VS_STAMP: ShaderStamp = ShaderStamp::new(0x5151);
const FS_STAMP: ShaderStamp = ShaderStamp::new(0xF5F5);
const CS_STAMP: ShaderStamp = ShaderStamp::new(0xC5C5);

/// The stamps of [`desc`]'s two shaders.
fn stamps() -> StageStamps {
    StageStamps {
        vertex: VS_STAMP,
        tess_control: None,
        tess_eval: None,
        geometry: None,
        fragment: Some(FS_STAMP),
    }
}

fn blend() -> BlendState {
    let replace = BlendComponent {
        src: BlendFactor::One,
        dst: BlendFactor::Zero,
        op: BlendOperation::Add,
    };
    BlendState {
        enabled: false,
        color: replace,
        alpha: replace,
        write_mask: ColorWrites::ALL,
    }
}

fn face() -> StencilFace {
    StencilFace {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    }
}

/// A valid description of a pipeline with vertex shader `vs` and fragment shader `fs`, with a
/// stencil test, so every dynamic field is there.
fn desc(vs: ShaderId, fs: ShaderId) -> GraphicsPipelineDesc {
    let mut colors = [None; starquake_render::caps::GUEST_COLOR_TARGETS];
    colors[0] = Some(PixelFormat::A8B8G8R8Unorm);
    GraphicsPipelineDesc {
        stages: Stages {
            vertex: vs,
            tess_control: None,
            tess_eval: None,
            geometry: None,
            fragment: Some(fs),
        },
        vertex_input: VertexLayout {
            buffers: vec![VertexBufferLayout {
                stride: 16,
                step: VertexStepMode::Vertex,
            }],
            attributes: vec![VertexAttribute {
                location: 0,
                buffer: 0,
                offset: 0,
                components: VertexComponents::R32G32B32A32,
                kind: VertexKind::Float,
            }],
        },
        topology: Topology::Triangles,
        primitive_restart: false,
        raster: RasterState {
            cull: CullMode::None,
            front_face: FrontFace::Ccw,
            polygon_mode: PolygonMode::Fill,
            depth_clamp: false,
            depth_bias: DepthBias {
                constant: 0.25,
                slope_scale: 0.5,
                clamp: 0.75,
            },
            line_width: 2.0,
        },
        depth_stencil: DepthStencilState {
            depth_test: Some(CompareFunction::Less),
            depth_write: true,
            stencil: Some(StencilState {
                front: face(),
                back: face(),
                read_mask: 0xFF,
                write_mask: 0x7F,
            }),
        },
        blend: [blend(); starquake_render::caps::GUEST_COLOR_TARGETS],
        logic_op: None,
        targets: TargetFormats {
            colors,
            depth_stencil: Some(PixelFormat::D32Float),
            sample_count: 1,
        },
        emulation: EmulationFlags::EMPTY,
    }
}

/// A change to each dynamic field, with the state that makes it dynamic.
fn dynamic_changes() -> Vec<(&'static str, DynamicState, Change)> {
    fn stencil(d: &mut GraphicsPipelineDesc) -> &mut StencilState {
        d.depth_stencil.stencil.as_mut().unwrap()
    }
    vec![
        ("bias constant", DynamicState::DEPTH_BIAS, |d| {
            d.raster.depth_bias.constant = 3.0;
        }),
        ("bias slope", DynamicState::DEPTH_BIAS, |d| {
            d.raster.depth_bias.slope_scale = 3.0;
        }),
        ("bias clamp", DynamicState::DEPTH_BIAS, |d| {
            d.raster.depth_bias.clamp = 3.0;
        }),
        ("line width", DynamicState::LINE_WIDTH, |d| {
            d.raster.line_width = 3.0;
        }),
        ("read mask", DynamicState::STENCIL_MASKS, |d| {
            stencil(d).read_mask = 0x0F;
        }),
        ("write mask", DynamicState::STENCIL_MASKS, |d| {
            stencil(d).write_mask = 0xF0;
        }),
    ]
}

#[test]
fn the_same_state_twice_is_one_pipeline() {
    let mut rig = Rig::new(DynamicState::EMPTY);
    let desc = rig.desc();
    let id = compiling(rig.graphics(&desc));
    assert_eq!(rig.graphics(&desc), Lookup::Ready(id));
    assert_eq!(rig.graphics(&desc.clone()), Lookup::Ready(id));
    assert_eq!(rig.spy.graphics(), vec![desc]);
    assert_eq!(rig.cache.len(), 1);

    let cid = compiling(rig.compute(CS_STAMP));
    assert_eq!(rig.compute(CS_STAMP), Lookup::Ready(cid));
    assert_ne!(cid, id);
    assert_eq!(
        rig.spy.compute(),
        vec![ComputePipelineDesc { shader: rig.cs }]
    );
    assert_eq!(rig.cache.len(), 2);
}

#[test]
fn a_dynamic_state_change_makes_no_new_pipeline_only_when_the_device_has_it() {
    for (what, state, change) in dynamic_changes() {
        // The state is the only difference between the two devices.
        for (dynamic, pipelines) in [
            (DynamicState::ALL, 1),
            (DynamicState::ALL.difference(state), 2),
            (state, 1),
            (DynamicState::EMPTY, 2),
        ] {
            let mut rig = Rig::new(dynamic);
            let first = rig.desc();
            let mut second = first.clone();
            change(&mut second);
            rig.graphics(&first);
            rig.graphics(&second);
            assert_eq!(
                rig.spy.graphics().len(),
                pipelines,
                "{what} under {dynamic:?}"
            );
        }
    }
}

#[test]
fn pipelines_get_canonical_values_in_their_dynamic_fields() {
    let mut rig = Rig::new(KEYED_DYNAMIC_STATE);
    let given = rig.desc();
    rig.graphics(&given);
    let mut expected = given.clone();
    expected.raster.depth_bias = CANONICAL_DEPTH_BIAS;
    expected.raster.line_width = CANONICAL_LINE_WIDTH;
    let stencil = expected.depth_stencil.stencil.as_mut().unwrap();
    stencil.read_mask = CANONICAL_STENCIL_MASK;
    stencil.write_mask = CANONICAL_STENCIL_MASK;
    assert_ne!(
        expected, given,
        "the rig's values differ from the canonical ones"
    );
    assert_eq!(rig.spy.graphics(), vec![expected]);

    // Each state alone replaces only its own field.
    for (what, state, _) in dynamic_changes() {
        let mut rig = Rig::new(state);
        rig.graphics(&given);
        let made = rig.spy.graphics().pop().unwrap();
        let s = made.depth_stencil.stencil.unwrap();
        let bias = state == DynamicState::DEPTH_BIAS;
        let width = state == DynamicState::LINE_WIDTH;
        let masks = state == DynamicState::STENCIL_MASKS;
        let want_bias = if bias {
            CANONICAL_DEPTH_BIAS
        } else {
            given.raster.depth_bias
        };
        assert_eq!(made.raster.depth_bias, want_bias, "{what}");
        let want_width = if width {
            CANONICAL_LINE_WIDTH
        } else {
            given.raster.line_width
        };
        assert_eq!(
            made.raster.line_width.to_bits(),
            want_width.to_bits(),
            "{what}"
        );
        let gs = given.depth_stencil.stencil.unwrap();
        let want = if masks {
            (CANONICAL_STENCIL_MASK, CANONICAL_STENCIL_MASK)
        } else {
            (gs.read_mask, gs.write_mask)
        };
        assert_eq!((s.read_mask, s.write_mask), want, "{what}");
    }

    // With nothing dynamic, the description goes to the backend as it was given.
    let mut rig = Rig::new(DynamicState::EMPTY);
    rig.graphics(&given);
    assert_eq!(rig.spy.graphics(), vec![given]);
}

#[test]
fn flags_with_no_field_in_the_description_do_not_change_the_key() {
    let unkeyed = DynamicState::ALL.difference(KEYED_DYNAMIC_STATE);
    assert_eq!(
        unkeyed,
        DynamicState::VIEWPORT
            | DynamicState::SCISSOR
            | DynamicState::BLEND_CONSTANT
            | DynamicState::STENCIL_REFERENCE
    );
    let list = |dynamic| {
        let mut rig = Rig::new(dynamic);
        let desc = rig.desc();
        rig.graphics(&desc);
        rig.cache.key_list().to_bytes().unwrap()
    };
    assert_eq!(list(DynamicState::ALL), list(KEYED_DYNAMIC_STATE));
    assert_eq!(list(unkeyed), list(DynamicState::EMPTY));
    assert_ne!(list(DynamicState::ALL), list(DynamicState::EMPTY));
}

/// A change to each field of a description that is never dynamic.
fn fixed_changes() -> Vec<(&'static str, ShaderChange)> {
    let changes: Vec<(&'static str, ShaderChange)> = vec![
        ("fragment shader", |d, other| {
            d.stages.fragment = Some(other);
        }),
        ("no fragment shader", |d, _| d.stages.fragment = None),
        ("vertex shader", |d, other| d.stages.vertex = other),
        ("stride", |d, _| d.vertex_input.buffers[0].stride = 32),
        ("step", |d, _| {
            d.vertex_input.buffers[0].step = VertexStepMode::Instance;
        }),
        ("second buffer", |d, _| {
            d.vertex_input.buffers.push(d.vertex_input.buffers[0]);
        }),
        ("location", |d, _| d.vertex_input.attributes[0].location = 1),
        ("attribute offset", |d, _| {
            d.vertex_input.attributes[0].offset = 4;
        }),
        ("components", |d, _| {
            d.vertex_input.attributes[0].components = VertexComponents::R8G8B8A8;
        }),
        ("kind", |d, _| {
            d.vertex_input.attributes[0].kind = VertexKind::Unorm;
        }),
        ("topology", |d, _| d.topology = Topology::Lines),
        ("restart", |d, _| d.primitive_restart = true),
        ("cull", |d, _| d.raster.cull = CullMode::Back),
        ("front face", |d, _| d.raster.front_face = FrontFace::Cw),
        ("polygon mode", |d, _| {
            d.raster.polygon_mode = PolygonMode::Line;
        }),
        ("depth clamp", |d, _| d.raster.depth_clamp = true),
        ("depth test", |d, _| {
            d.depth_stencil.depth_test = Some(CompareFunction::Greater);
        }),
        ("no depth test", |d, _| d.depth_stencil.depth_test = None),
        ("depth write", |d, _| d.depth_stencil.depth_write = false),
        ("no stencil", |d, _| d.depth_stencil.stencil = None),
        ("stencil front", |d, _| {
            d.depth_stencil.stencil.as_mut().unwrap().front.pass = StencilOperation::Zero;
        }),
        ("stencil back", |d, _| {
            d.depth_stencil.stencil.as_mut().unwrap().back.compare = CompareFunction::Never;
        }),
        ("blend on", |d, _| d.blend[0].enabled = true),
        ("blend of an unused target", |d, _| {
            d.blend[7].color.op = BlendOperation::Max;
        }),
        ("blend alpha", |d, _| {
            d.blend[0].alpha.dst = BlendFactor::DstAlpha;
        }),
        ("write mask", |d, _| {
            d.blend[0].write_mask = ColorWrites::RED;
        }),
        ("logic op", |d, _| d.logic_op = Some(LogicOp::Xor)),
        ("colour format", |d, _| {
            d.targets.colors[0] = Some(PixelFormat::R8Unorm);
        }),
        ("second target", |d, _| {
            d.targets.colors[1] = Some(PixelFormat::R8Unorm);
        }),
        ("depth format", |d, _| {
            d.targets.depth_stencil = Some(PixelFormat::D24UnormS8Uint);
        }),
        ("samples", |d, _| d.targets.sample_count = 4),
        ("alpha test", |d, _| {
            d.emulation = EmulationFlags::ALPHA_TEST;
        }),
        ("depth range", |d, _| {
            d.emulation = EmulationFlags::DEPTH_RANGE_REWRITE;
        }),
        ("restart rewrite", |d, _| {
            d.emulation = EmulationFlags::PRIMITIVE_RESTART_REWRITE;
        }),
        ("attribute fetch", |d, _| {
            d.emulation = EmulationFlags::ATTRIBUTE_FETCH;
        }),
        ("typed outputs", |d, _| {
            d.emulation = EmulationFlags::TYPED_OUTPUTS;
        }),
    ];
    changes
}

#[test]
fn every_field_that_is_not_dynamic_is_in_the_key() {
    // Every emulation flag is in the table, each alone.
    let flags: Vec<_> = fixed_changes()
        .into_iter()
        .filter_map(|(_, change)| {
            let mut d = desc(ShaderId::new(0, 0), ShaderId::new(0, 0));
            change(&mut d, ShaderId::new(0, 0));
            (d.emulation != EmulationFlags::EMPTY).then_some(d.emulation)
        })
        .collect();
    assert_eq!(
        flags,
        EmulationFlags::NAMED
            .iter()
            .map(|&(_, f)| f)
            .collect::<Vec<_>>()
    );

    let mut rig = Rig::new(DynamicState::ALL);
    let other = rig.spy.shader(ShaderStage::Fragment);
    let other_vs = rig.spy.shader(ShaderStage::Vertex);
    let base = rig.desc();
    assert!(matches!(rig.graphics(&base), Lookup::Compiling(_)));
    for (n, (what, change)) in fixed_changes().into_iter().enumerate() {
        let mut changed = base.clone();
        let other = if what == "vertex shader" {
            other_vs
        } else {
            other
        };
        change(&mut changed, other);
        assert_ne!(changed, base, "{what}");
        let mut stamps = stamps();
        stamps.fragment = changed.stages.fragment.map(|_| FS_STAMP);
        let lookup = rig.cache.lookup_graphics(&changed, &stamps).unwrap();
        assert!(matches!(lookup, Lookup::Compiling(_)), "{what}: {lookup:?}");
        assert_eq!(rig.spy.graphics().len(), n + 2, "{what}");
    }
}

#[test]
fn a_failed_pipeline_is_not_rebuilt() {
    let mut rig = Rig::new(DynamicState::EMPTY);
    let desc = rig.desc();
    let id = compiling(rig.graphics(&desc));
    assert_eq!(rig.spy.statuses(), 0, "creating does not ask the status");
    rig.spy
        .inner
        .script_pipeline(id, vec![PipelineStatus::Compiling, PipelineStatus::Failed])
        .unwrap();
    assert_eq!(rig.graphics(&desc), Lookup::Compiling(id));
    let failed = |newly| {
        Lookup::Failed(Failure {
            newly,
            cause: FailCause::Compile,
        })
    };
    assert_eq!(rig.graphics(&desc), failed(true));
    assert_eq!(rig.graphics(&desc), failed(false));
    assert_eq!(rig.graphics(&desc), failed(false));
    assert_eq!(
        rig.spy.statuses(),
        2,
        "a failed pipeline is not asked about again"
    );
    assert_eq!(rig.spy.graphics().len(), 1);

    // The same for compute.
    let id = compiling(rig.compute(CS_STAMP));
    rig.spy
        .inner
        .script_pipeline(id, vec![PipelineStatus::Failed])
        .unwrap();
    assert_eq!(rig.compute(CS_STAMP), failed(true));
    assert_eq!(rig.compute(CS_STAMP), failed(false));
    assert_eq!(rig.spy.statuses(), 3);
    assert_eq!(rig.spy.compute().len(), 1);
}

#[test]
fn a_ready_pipeline_is_not_asked_about_again() {
    let mut rig = Rig::new(DynamicState::EMPTY);
    let desc = rig.desc();
    let id = compiling(rig.graphics(&desc));
    rig.spy
        .inner
        .script_pipeline(
            id,
            vec![
                PipelineStatus::Compiling,
                PipelineStatus::Compiling,
                PipelineStatus::Ready,
                PipelineStatus::Failed,
            ],
        )
        .unwrap();
    assert_eq!(rig.graphics(&desc), Lookup::Compiling(id));
    assert_eq!(rig.graphics(&desc), Lookup::Compiling(id));
    assert_eq!(rig.spy.statuses(), 2);
    assert_eq!(rig.graphics(&desc), Lookup::Ready(id));
    // The script's next answer would be a failure; it is never asked for.
    assert_eq!(rig.graphics(&desc), Lookup::Ready(id));
    assert_eq!(rig.spy.statuses(), 3);
}

#[test]
fn a_refused_pipeline_is_not_rebuilt_or_recorded() {
    let mut no_geometry = caps(DynamicState::EMPTY);
    no_geometry.stages = StageSupport::TESSELLATION;
    let mut rig = Rig::with_caps(no_geometry);
    let gs = rig.spy.shader(ShaderStage::Fragment);
    let mut desc = rig.desc();
    desc.stages.geometry = Some(gs);
    let mut stamps = stamps();
    stamps.geometry = Some(ShaderStamp::new(0x6565));
    let cause = FailCause::Refused(RenderError::StageUnsupported(ShaderStage::Geometry));
    let failed = |newly| {
        Lookup::Failed(Failure {
            newly,
            cause: cause.clone(),
        })
    };
    assert_eq!(rig.cache.lookup_graphics(&desc, &stamps), Ok(failed(true)));
    assert_eq!(rig.cache.lookup_graphics(&desc, &stamps), Ok(failed(false)));
    assert_eq!(rig.spy.graphics().len(), 1);
    assert_eq!(rig.spy.statuses(), 0);
    assert_eq!(rig.cache.len(), 1);
    assert!(rig.cache.key_list().is_empty());

    // A compute pipeline over a vertex shader is refused too.
    let refused = rig
        .cache
        .lookup_compute(ComputePipelineDesc { shader: rig.vs }, CS_STAMP)
        .unwrap();
    assert_eq!(
        refused,
        Lookup::Failed(Failure {
            newly: true,
            cause: FailCause::Refused(RenderError::Invalid(
                "a compute pipeline's shader is not a compute shader"
            )),
        })
    );
    assert!(rig.cache.key_list().is_empty());
}

#[test]
fn a_pipeline_that_fails_to_compile_stays_in_the_key_list() {
    let mut rig = Rig::new(DynamicState::EMPTY);
    let desc = rig.desc();
    let id = compiling(rig.graphics(&desc));
    rig.spy
        .inner
        .script_pipeline(id, vec![PipelineStatus::Failed])
        .unwrap();
    assert!(matches!(rig.graphics(&desc), Lookup::Failed(_)));
    assert_eq!(rig.cache.key_list(), expected_list(&rig, &[0]));
}

#[test]
fn a_stage_with_a_shader_and_no_stamp_or_the_reverse_is_refused() {
    let tess = |d: &mut GraphicsPipelineDesc, s: &mut StageStamps, id| {
        d.stages.tess_control = Some(id);
        d.stages.tess_eval = Some(id);
        s.tess_control = Some(ShaderStamp::new(1));
        s.tess_eval = Some(ShaderStamp::new(2));
    };
    let cases: [(ShaderStage, StampCase, StampCase); 4] = [
        (
            ShaderStage::TessControl,
            |d, s| {
                d.stages.tess_control = Some(d.stages.vertex);
                s.tess_control = None;
            },
            |_, s| s.tess_control = Some(ShaderStamp::new(1)),
        ),
        (
            ShaderStage::TessEval,
            |d, s| {
                d.stages.tess_eval = Some(d.stages.vertex);
                s.tess_eval = None;
            },
            |_, s| s.tess_eval = Some(ShaderStamp::new(1)),
        ),
        (
            ShaderStage::Geometry,
            |d, _| d.stages.geometry = Some(d.stages.vertex),
            |_, s| s.geometry = Some(ShaderStamp::new(1)),
        ),
        (
            ShaderStage::Fragment,
            |d, _| d.stages.fragment = None,
            |_, s| s.fragment = None,
        ),
    ];
    for (stage, shader_only, stamp_only) in cases {
        for (how, case) in [("shader only", shader_only), ("stamp only", stamp_only)] {
            let mut rig = Rig::new(DynamicState::EMPTY);
            let mut d = rig.desc();
            let mut s = stamps();
            case(&mut d, &mut s);
            assert_eq!(
                rig.cache.lookup_graphics(&d, &s),
                Err(PipelineCacheError::StampMismatch { stage }),
                "{how}"
            );
            assert!(rig.cache.is_empty());
            assert_eq!(rig.spy.graphics().len(), 0);
            assert!(rig.cache.key_list().is_empty());
        }
    }
    // With both tessellation stages matched, the stamps pass.
    let mut rig = Rig::new(DynamicState::EMPTY);
    let mut d = rig.desc();
    let mut s = stamps();
    tess(&mut d, &mut s, rig.vs);
    d.topology = Topology::Patches;
    assert!(rig.cache.lookup_graphics(&d, &s).is_ok());

    // Two stages wrong at once: the first in stage order is named.
    let mut rig = Rig::new(DynamicState::EMPTY);
    let mut d = rig.desc();
    let mut s = stamps();
    s.fragment = None;
    s.tess_eval = Some(ShaderStamp::new(3));
    s.geometry = Some(ShaderStamp::new(4));
    assert_eq!(
        rig.cache.lookup_graphics(&d, &s),
        Err(PipelineCacheError::StampMismatch {
            stage: ShaderStage::TessEval
        })
    );
    d.stages.fragment = None;
    assert_eq!(
        rig.cache.lookup_graphics(&d, &s).map_err(|e| e.to_string()),
        Err(
            "the tessellation evaluation stage has a shader without a stamp or a stamp without a \
             shader"
                .to_owned()
        )
    );
}

#[test]
fn a_full_cache_refuses_new_keys_and_still_answers_known_ones() {
    let mut rig = Rig::build(caps(DynamicState::EMPTY), 2);
    let a = rig.desc();
    let mut b = a.clone();
    b.topology = Topology::Points;
    let mut c = a.clone();
    c.topology = Topology::Lines;
    let id = compiling(rig.graphics(&a));
    // One below the limit, a new key is still made.
    assert!(matches!(rig.graphics(&b), Lookup::Compiling(_)));
    assert_eq!(rig.cache.len(), 2);
    let full = Err(PipelineCacheError::Full { limit: 2 });
    assert_eq!(rig.cache.lookup_graphics(&c, &stamps()), full);
    assert_eq!(
        rig.cache
            .lookup_compute(ComputePipelineDesc { shader: rig.cs }, CS_STAMP),
        full
    );
    assert_eq!(rig.spy.graphics().len(), 2);
    assert_eq!(rig.spy.compute().len(), 0);
    assert_eq!(rig.cache.key_list().len(), 2);
    assert_eq!(rig.graphics(&a), Lookup::Ready(id));
    assert_eq!(
        PipelineCacheError::Full { limit: 2 }.to_string(),
        "the pipeline cache is full (2 pipelines)"
    );

    // A refused pipeline takes an entry too.
    let mut rig = Rig::build(caps(DynamicState::EMPTY), 1);
    let refused = rig
        .cache
        .lookup_compute(ComputePipelineDesc { shader: rig.vs }, CS_STAMP)
        .unwrap();
    assert!(matches!(refused, Lookup::Failed(_)));
    assert_eq!(
        rig.cache.lookup_graphics(&a, &stamps()),
        Err(PipelineCacheError::Full { limit: 1 })
    );
}

#[test]
fn a_new_cache_holds_up_to_the_shipping_limit() {
    let spy = Spy::new(caps(DynamicState::EMPTY));
    let backend: Arc<dyn Backend> = spy;
    let cache = PipelineCache::new(backend);
    assert_eq!(cache.limit, MAX_PIPELINES);
    assert!(cache.is_empty());
    assert_eq!(cache.unrecorded(), 0);
    assert!(cache.key_list().is_empty());
}

/// The key list holding the keys of the rig's pipelines `which`, in that order: 0 is
/// [`desc`] with [`stamps`], 1 its compute pipeline with [`CS_STAMP`].
fn expected_list(rig: &Rig, which: &[usize]) -> KeyList {
    let dynamic = rig.cache.dynamic;
    KeyList {
        keys: which
            .iter()
            .map(|&n| match n {
                0 => PipelineKey::Graphics(Box::new(GraphicsKey {
                    stamps: stamps(),
                    fixed: FixedState::of(&rig.desc(), dynamic),
                })),
                _ => PipelineKey::Compute(CS_STAMP),
            })
            .collect(),
    }
}

#[test]
fn the_key_list_holds_each_accepted_key_once_in_first_seen_order() {
    let mut rig = Rig::new(DynamicState::EMPTY);
    let desc = rig.desc();
    rig.compute(CS_STAMP);
    rig.graphics(&desc);
    rig.compute(CS_STAMP);
    rig.graphics(&desc);
    assert_eq!(rig.cache.key_list(), expected_list(&rig, &[1, 0]));

    // Stamps that differ only above their low byte are two keys; the second shader is another
    // pipeline, recorded after the first.
    let cs2 = rig.spy.shader(ShaderStage::Compute);
    let high = ShaderStamp::new(CS_STAMP.get() | (1 << 120));
    rig.cache
        .lookup_compute(ComputePipelineDesc { shader: cs2 }, high)
        .unwrap();
    let mut want = expected_list(&rig, &[1, 0]);
    want.keys.push(PipelineKey::Compute(high));
    assert_eq!(rig.cache.key_list(), want);
}

#[test]
fn keys_past_the_record_limit_are_counted_not_recorded() {
    let mut rig = Rig::build(caps(DynamicState::EMPTY), 2);
    let list = KeyList {
        keys: (1..=3)
            .map(|n| PipelineKey::Compute(ShaderStamp::new(n)))
            .collect(),
    };
    // No shader for any stamp: nothing is made, but every key is recorded while there is room.
    let done = rig.cache.precompile(&list, &mut |_| None);
    assert_eq!(
        done,
        Precompiled {
            missing_shader: 3,
            ..Precompiled::default()
        }
    );
    assert_eq!(rig.cache.unrecorded(), 1);
    assert_eq!(
        rig.cache.key_list(),
        KeyList {
            keys: list.keys[..2].to_vec()
        }
    );
    // The two keys already recorded are not counted again; the attempt the record refuses is.
    rig.cache.precompile(&list, &mut |_| None);
    assert_eq!(rig.cache.unrecorded(), 2);
    assert!(rig.cache.is_empty());
}

#[test]
fn precompile_reports_every_outcome() {
    let mut rig = Rig::build(caps(DynamicState::EMPTY), 4);
    let base = rig.desc();
    let fixed = |d: &GraphicsPipelineDesc, dynamic| FixedState::of(d, dynamic);
    let graphics = |stamps: StageStamps, fixed: FixedState| {
        PipelineKey::Graphics(Box::new(GraphicsKey { stamps, fixed }))
    };
    let mut lines = base.clone();
    lines.topology = Topology::Lines;
    let mut points = base.clone();
    points.topology = Topology::Points;
    let mut strip = base.clone();
    strip.topology = Topology::LineStrip;
    let mut bad = base.clone();
    // Two samples: a valid key (some device has them), which this device refuses.
    bad.targets.sample_count = 2;
    let unknown = ShaderStamp::new(0xDEAD);
    let mut unknown_stamps = stamps();
    unknown_stamps.fragment = Some(unknown);

    // Known before the precompile: the base pipeline.
    rig.graphics(&base);
    let list = KeyList {
        keys: vec![
            graphics(stamps(), fixed(&base, DynamicState::EMPTY)), // known
            graphics(stamps(), fixed(&lines, DynamicState::EMPTY)), // queued
            graphics(stamps(), fixed(&points, KEYED_DYNAMIC_STATE)), // other dynamic state
            graphics(unknown_stamps, fixed(&points, DynamicState::EMPTY)), // missing shader
            PipelineKey::Compute(unknown),                         // missing shader
            graphics(stamps(), fixed(&bad, DynamicState::EMPTY)),  // refused
            PipelineKey::Compute(CS_STAMP),                        // queued
            graphics(stamps(), fixed(&strip, DynamicState::EMPTY)), // full
        ],
    };
    let (vs, fs, cs) = (rig.vs, rig.fs, rig.cs);
    let done = rig.cache.precompile(&list, &mut |stamp| match stamp {
        VS_STAMP => Some(vs),
        FS_STAMP => Some(fs),
        CS_STAMP => Some(cs),
        _ => None,
    });
    assert_eq!(
        done,
        Precompiled {
            queued: 2,
            known: 1,
            refused: 1,
            other_dynamic_state: 1,
            missing_shader: 2,
            full: 1,
        }
    );
    // The base pipeline, then the two queued and the refused one, each with the shaders the
    // stamps map to.
    assert_eq!(rig.spy.graphics(), vec![base, lines, bad]);
    assert_eq!(rig.spy.compute(), vec![ComputePipelineDesc { shader: cs }]);
    assert_eq!(rig.cache.len(), 4);
    // Every key is recorded, in the list's order, even those not built; the record holds four.
    assert_eq!(rig.cache.unrecorded(), 4);
    assert_eq!(
        rig.cache.key_list(),
        KeyList {
            keys: list.keys[..4].to_vec()
        }
    );
}

const TC_STAMP: ShaderStamp = ShaderStamp::new(0x7C7C);
const TE_STAMP: ShaderStamp = ShaderStamp::new(0x7E7E);

#[test]
fn precompile_maps_each_tessellation_stamp_to_its_own_shader() {
    let mut rig = Rig::new(DynamicState::EMPTY);
    let tcs = rig.spy.shader(ShaderStage::TessControl);
    let tes = rig.spy.shader(ShaderStage::TessEval);
    assert_ne!(tcs, tes);
    let mut desc = rig.desc();
    desc.stages.tess_control = Some(tcs);
    desc.stages.tess_eval = Some(tes);
    desc.topology = Topology::Patches;
    let mut stamps = stamps();
    stamps.tess_control = Some(TC_STAMP);
    stamps.tess_eval = Some(TE_STAMP);
    let list = KeyList {
        keys: vec![PipelineKey::Graphics(Box::new(GraphicsKey {
            stamps,
            fixed: FixedState::of(&desc, DynamicState::EMPTY),
        }))],
    };
    let (vs, fs) = (rig.vs, rig.fs);
    let map = move |missing: Option<ShaderStamp>| {
        move |stamp: ShaderStamp| {
            if Some(stamp) == missing {
                return None;
            }
            match stamp {
                VS_STAMP => Some(vs),
                FS_STAMP => Some(fs),
                TC_STAMP => Some(tcs),
                TE_STAMP => Some(tes),
                _ => None,
            }
        }
    };

    // A stamp of either tessellation stage with no shader skips the key.
    for missing in [TC_STAMP, TE_STAMP] {
        let done = rig.cache.precompile(&list, &mut map(Some(missing)));
        assert_eq!(
            done,
            Precompiled {
                missing_shader: 1,
                ..Precompiled::default()
            },
            "{missing:?}"
        );
        assert!(rig.spy.graphics().is_empty(), "{missing:?}");
        assert!(rig.cache.is_empty(), "{missing:?}");
    }

    // With every shader known, the backend gets each stage's own shader.
    let done = rig.cache.precompile(&list, &mut map(None));
    assert_eq!(
        done,
        Precompiled {
            queued: 1,
            ..Precompiled::default()
        }
    );
    assert_eq!(rig.spy.graphics(), vec![desc.clone()]);
    assert_eq!(
        rig.spy.graphics()[0].stages,
        Stages {
            vertex: vs,
            tess_control: Some(tcs),
            tess_eval: Some(tes),
            geometry: None,
            fragment: Some(fs),
        }
    );
    // The draw finds the precompiled pipeline.
    assert!(matches!(
        rig.cache.lookup_graphics(&desc, &stamps),
        Ok(Lookup::Ready(_))
    ));
    assert_eq!(rig.spy.graphics().len(), 1);
}

#[test]
fn a_stamp_mismatch_is_reported_before_a_full_cache() {
    let mut rig = Rig::build(caps(DynamicState::EMPTY), 1);
    let a = rig.desc();
    compiling(rig.graphics(&a));
    let mut b = a.clone();
    b.topology = Topology::Points;
    // Each fault alone.
    assert_eq!(
        rig.cache.lookup_graphics(&b, &stamps()),
        Err(PipelineCacheError::Full { limit: 1 })
    );
    let mut mismatched = stamps();
    mismatched.geometry = Some(ShaderStamp::new(0x6565));
    assert_eq!(
        rig.cache.lookup_graphics(&a, &mismatched),
        Err(PipelineCacheError::StampMismatch {
            stage: ShaderStage::Geometry
        })
    );
    // Both at once: the mismatch is named.
    assert_eq!(
        rig.cache.lookup_graphics(&b, &mismatched),
        Err(PipelineCacheError::StampMismatch {
            stage: ShaderStage::Geometry
        })
    );
    assert_eq!(rig.cache.len(), 1);
    assert_eq!(rig.spy.graphics().len(), 1);
}

#[test]
fn the_first_lookup_reports_a_failure_precompile_found() {
    let mut rig = Rig::new(DynamicState::EMPTY);
    let mut bad = rig.desc();
    // Two samples: a valid key, which this device refuses.
    bad.targets.sample_count = 2;
    let list = KeyList {
        keys: vec![
            PipelineKey::Graphics(Box::new(GraphicsKey {
                stamps: stamps(),
                fixed: FixedState::of(&bad, DynamicState::EMPTY),
            })),
            PipelineKey::Compute(CS_STAMP),
        ],
    };
    let (vs, fs) = (rig.vs, rig.fs);
    // The compute stamp maps to a vertex shader, which the backend refuses.
    let done = rig.cache.precompile(&list, &mut |stamp| match stamp {
        VS_STAMP | CS_STAMP => Some(vs),
        FS_STAMP => Some(fs),
        _ => None,
    });
    assert_eq!(
        done,
        Precompiled {
            refused: 2,
            ..Precompiled::default()
        }
    );
    let refused = |newly, error| {
        Ok(Lookup::Failed(Failure {
            newly,
            cause: FailCause::Refused(error),
        }))
    };
    let samples = RenderError::SampleCount(2);
    assert_eq!(
        rig.cache.lookup_graphics(&bad, &stamps()),
        refused(true, samples.clone())
    );
    assert_eq!(
        rig.cache.lookup_graphics(&bad, &stamps()),
        refused(false, samples)
    );
    let not_compute = RenderError::Invalid("a compute pipeline's shader is not a compute shader");
    let compute = ComputePipelineDesc { shader: vs };
    assert_eq!(
        rig.cache.lookup_compute(compute, CS_STAMP),
        refused(true, not_compute.clone())
    );
    assert_eq!(
        rig.cache.lookup_compute(compute, CS_STAMP),
        refused(false, not_compute)
    );
    assert_eq!(rig.spy.graphics().len(), 1);
    assert_eq!(rig.spy.compute().len(), 1);
}

#[test]
fn the_first_lookup_reports_a_compile_failure_precompile_saw() {
    let failed = |newly| {
        Ok(Lookup::Failed(Failure {
            newly,
            cause: FailCause::Compile,
        }))
    };
    // `reported_first`: whether a lookup saw the failure before the precompile did.
    for reported_first in [false, true] {
        let mut rig = Rig::new(DynamicState::EMPTY);
        let desc = rig.desc();
        let id = compiling(rig.graphics(&desc));
        rig.spy
            .inner
            .script_pipeline(id, vec![PipelineStatus::Failed])
            .unwrap();
        if reported_first {
            assert_eq!(rig.cache.lookup_graphics(&desc, &stamps()), failed(true));
        }
        let list = rig.cache.key_list();
        let (vs, fs) = (rig.vs, rig.fs);
        let done = rig.cache.precompile(&list, &mut |stamp| match stamp {
            VS_STAMP => Some(vs),
            FS_STAMP => Some(fs),
            _ => None,
        });
        assert_eq!(
            done,
            Precompiled {
                known: 1,
                ..Precompiled::default()
            }
        );
        assert_eq!(
            rig.cache.lookup_graphics(&desc, &stamps()),
            failed(!reported_first),
            "reported first: {reported_first}"
        );
        assert_eq!(rig.cache.lookup_graphics(&desc, &stamps()), failed(false));
        assert_eq!(rig.spy.statuses(), 1);
    }
}

#[test]
fn the_next_boot_precompiles_the_key_list_and_its_draws_find_the_pipelines() {
    // Boot one.
    let mut one = Rig::new(DynamicState::ALL);
    let desc = one.desc();
    one.graphics(&desc);
    one.compute(CS_STAMP);
    let bytes = one.cache.key_list().to_bytes().unwrap();

    // Boot two: another backend, whose shaders have other ids.
    let mut two = Rig::new(DynamicState::ALL);
    two.spy.shader(ShaderStage::Vertex);
    let vs = two.spy.shader(ShaderStage::Vertex);
    let fs = two.spy.shader(ShaderStage::Fragment);
    let cs = two.spy.shader(ShaderStage::Compute);
    assert_ne!((vs, fs, cs), (one.vs, one.fs, one.cs));
    let list = KeyList::from_bytes(&bytes).unwrap();
    let done = two.cache.precompile(&list, &mut |stamp| match stamp {
        VS_STAMP => Some(vs),
        FS_STAMP => Some(fs),
        CS_STAMP => Some(cs),
        _ => None,
    });
    assert_eq!(
        done,
        Precompiled {
            queued: 2,
            ..Precompiled::default()
        }
    );
    let made = two.spy.graphics();
    let mut want = one.spy.graphics();
    want[0].stages.vertex = vs;
    want[0].stages.fragment = Some(fs);
    assert_eq!(made, want);

    // The first draw of boot two finds them.
    let mut draw = desc;
    draw.stages.vertex = vs;
    draw.stages.fragment = Some(fs);
    assert!(matches!(
        two.cache.lookup_graphics(&draw, &stamps()),
        Ok(Lookup::Ready(_))
    ));
    assert!(matches!(
        two.cache
            .lookup_compute(ComputePipelineDesc { shader: cs }, CS_STAMP),
        Ok(Lookup::Ready(_))
    ));
    assert_eq!(two.spy.graphics().len(), 1);
    assert_eq!(two.spy.compute().len(), 1);
    assert_eq!(two.cache.key_list().to_bytes().unwrap(), bytes);
}

#[test]
fn shipping_constants_match_their_sources() {
    // `DynamicState`'s bits in `starquake-render` (`caps.rs`): STENCIL_MASKS = 4,
    // DEPTH_BIAS = 5, LINE_WIDTH = 6.
    assert_eq!(KEYED_DYNAMIC_STATE.bits(), (1 << 4) | (1 << 5) | (1 << 6));
    assert_eq!(CANONICAL_LINE_WIDTH.to_bits(), 1.0f32.to_bits());
    assert_eq!(CANONICAL_STENCIL_MASK, 0);
    assert_eq!(
        CANONICAL_DEPTH_BIAS,
        DepthBias {
            constant: 0.0,
            slope_scale: 0.0,
            clamp: 0.0
        }
    );
    assert_eq!(MAX_PIPELINES, 65_536);
}
