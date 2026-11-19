//! Render capture v2 end to end (slice R11b's gate): a capture taken through
//! `CapturingBackend` around the recording backend replays on a fresh recording backend, with a
//! stub resolver that maps shader origins to hand-built naga modules, to the same results, with
//! no resources made by hand.

use super::fixtures::{
    MEMORY, caps, clear, counting, draw, graphics_desc, ok, rb, rect, region, slice, texture_desc,
    upload,
};
use starquake_render::capture::{
    CaptureBuffer, CaptureReader, CaptureSink, CapturingBackend, Finding, FindingKind,
    ReplayFrames, ReplayOptions,
    ReplayReport, ResolveError, ShaderResolver, WaitSetting, replay,
};
use starquake_render::command::{ClearValue, PresentDesc, PresentTransform, RenderTargets};
use starquake_render::resource::{
    AddressMode, BorderColor, FilterMode, ReductionMode, SamplerDesc, Swizzle, TextureDimension,
    ViewDimension,
};
use starquake_render::shader::ShaderInterface;
use starquake_render::{
    Backend, BufferDesc, BufferUsage, Command, CommandList, ComputePipelineDesc, FenceValue, Frame,
    FrameSink, PixelFormat, PresentTarget, RecordingBackend, ResourceId, ShaderDesc,
    ShaderOrigin, ShaderSource, ShaderStage, ViewDesc,
};
use std::sync::{Arc, Weak};
use std::time::Duration;

const RGBA: PixelFormat = PixelFormat::A8B8G8R8Unorm;

/// The frame tee the wiring installs: dumped frames go to the capture.
#[derive(Debug)]
struct Tee(Weak<CapturingBackend>);

impl FrameSink for Tee {
    fn frame(&self, frame: Frame) {
        if let Some(capture) = self.0.upgrade() {
            capture.record_frame(&frame);
        }
    }
}

/// An empty, validated naga module.
fn naga() -> ShaderSource {
    let module = naga::Module::default();
    let info = ok(naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module));
    ShaderSource::Naga {
        module: Box::new(module),
        info: Box::new(info),
    }
}

fn shader(stage: ShaderStage, origin: Option<&[u8]>) -> ShaderDesc {
    ShaderDesc {
        stage,
        entry: "main".to_owned(),
        source: naga(),
        interface: ShaderInterface::default(),
        origin: origin.map(|bytes| ShaderOrigin {
            record: Arc::from(bytes.to_vec()),
            stage_index: 0,
        }),
    }
}

/// Maps the scene's three origins to hand-built naga modules of their stage.
struct Stub;

impl ShaderResolver for Stub {
    fn resolve(&self, origin: &ShaderOrigin) -> Result<ShaderDesc, ResolveError> {
        let stage = match &*origin.record {
            b"vs" => ShaderStage::Vertex,
            b"fs" => ShaderStage::Fragment,
            b"cs" => ShaderStage::Compute,
            _ => return Err(ResolveError::Shader("an unknown origin".to_owned())),
        };
        Ok(shader(stage, None))
    }
}

/// Makes and uses every kind of resource through `b`: uploads, copies, a clear, a draw, a
/// dispatch, readbacks known and unknown, two dumped presents, destroys. With `decoys_first`,
/// a shader with no origin and a buffer of [`DECOY_BYTES`] are made first, which a replay with
/// a smaller memory limit refuses, so that the ids the fresh backend hands out differ.
fn scene(b: &dyn Backend, decoys_first: bool) {
    if decoys_first {
        ok(b.create_shader(shader(ShaderStage::Compute, None)));
        ok(b.create_buffer(&BufferDesc {
            size: DECOY_BYTES,
            usage: BufferUsage::ALL,
        }));
    }
    let buf = ok(b.create_buffer(&BufferDesc {
        size: 256,
        usage: BufferUsage::ALL,
    }));
    let tex = ok(b.create_texture(&texture_desc(RGBA, TextureDimension::D2, (4, 4, 1), 1)));
    let view = ok(b.create_view(
        tex,
        &ViewDesc {
            format: RGBA,
            dimension: ViewDimension::D2,
            base_mip: 0,
            mip_count: 1,
            base_layer: 0,
            layer_count: 1,
            swizzle: Swizzle::IDENTITY,
        },
    ));
    let sampler = ok(b.create_sampler(&SamplerDesc {
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Nearest,
        mip_filter: FilterMode::Nearest,
        address: [AddressMode::ClampToEdge; 3],
        border_color: BorderColor::OpaqueBlack,
        lod_bias: 0.0,
        min_lod: 0.0,
        max_lod: 1.0,
        max_anisotropy: 1,
        compare: None,
        reduction: ReductionMode::Average,
    }));
    let vs = ok(b.create_shader(shader(ShaderStage::Vertex, Some(b"vs"))));
    let fs = ok(b.create_shader(shader(ShaderStage::Fragment, Some(b"fs"))));
    let cs = ok(b.create_shader(shader(ShaderStage::Compute, Some(b"cs"))));
    let graphics = ok(b.create_graphics_pipeline(graphics_desc(vs, Some(fs))));
    let compute = ok(b.create_compute_pipeline(ComputePipelineDesc { shader: cs }));
    b.pipeline_status(graphics);
    b.pipeline_status(compute);
    ok(b.set_present_target(PresentTarget::Offscreen));
    let whole = region(tex, 0, (0, 0, 0), (4, 4, 1));
    let present = |frame| {
        Command::Present(PresentDesc {
            view,
            src_rect: rect(0, 0, 4, 4),
            transform: PresentTransform::Rotate90,
            frame,
            dump: true,
        })
    };
    let first = ok(b.submit(CommandList {
        commands: vec![
            upload(&counting(1, 64), slice(buf, 0, 64)),
            upload(&counting(100, 64), whole),
            Command::Copy(starquake_render::command::CopyDesc {
                src: slice(buf, 0, 32),
                dst: slice(buf, 128, 32),
            }),
            present(1),
            Command::Readback {
                from: slice(buf, 0, 64),
                id: rb(1),
            },
            Command::Readback {
                from: whole,
                id: rb(2),
            },
        ],
    }));
    b.take_readback(rb(1));
    b.take_readback(rb(2));
    let mut colors = [None; 8];
    colors[0] = Some(view);
    let second = ok(b.submit(CommandList {
        commands: vec![
            clear(
                view,
                ClearValue::Float([1.0, 0.0, 1.0, 0.0]),
                Some(rect(0, 0, 2, 1)),
            ),
            Command::SetRenderTargets(RenderTargets {
                colors,
                depth_stencil: None,
            }),
            Command::Draw(draw(graphics)),
            Command::Readback {
                from: whole,
                id: rb(3),
            },
            Command::Dispatch(starquake_render::command::DispatchDesc {
                pipeline: compute,
                bindings: Vec::new(),
                params: starquake_render::command::DispatchParams::Direct { x: 1, y: 1, z: 1 },
            }),
        ],
    }));
    b.take_readback(rb(3));
    b.destroy(ResourceId::Sampler(sampler), first);
    b.destroy(ResourceId::Buffer(buf), second);
    ok(b.submit(CommandList {
        commands: vec![
            upload(&counting(7, 64), whole),
            present(2),
            Command::Readback {
                from: whole,
                id: rb(4),
            },
        ],
    }));
    b.take_readback(rb(4));
}

/// The scene captured through the wrapper around a recording backend: the capture's bytes and
/// the backend.
fn capture(missing_first: bool) -> (Vec<u8>, Arc<RecordingBackend>) {
    let original = Arc::new(RecordingBackend::new(caps(), MEMORY));
    let buffer = Arc::new(CaptureBuffer::new());
    let inner: Arc<dyn Backend> = Arc::<RecordingBackend>::clone(&original);
    let sink: Arc<dyn CaptureSink> = Arc::<CaptureBuffer>::clone(&buffer);
    let wrapper = Arc::new(ok(CapturingBackend::new(
        inner,
        sink,
        "test".to_owned(),
        "recording".to_owned(),
        WaitSetting::Wait,
        u64::MAX,
    )));
    let tee: Arc<dyn FrameSink> = Arc::new(Tee(Arc::downgrade(&wrapper)));
    original.set_frame_sink(Some(tee));
    scene(&*wrapper, missing_first);
    ok(wrapper.finish());
    assert_eq!(wrapper.stopped(), None);
    (buffer.bytes(), original)
}

/// The decoy buffer's size: the device's largest.
const DECOY_BYTES: u64 = 1 << 20;

/// Replays `bytes` on a fresh recording backend whose budget is `budget`, with the replay's
/// memory limit at `limit` (the budget when `None`).
fn replay_fresh(bytes: &[u8], budget: u64, limit: Option<u64>) -> (ReplayReport, RecordingBackend) {
    let fresh = RecordingBackend::new(caps(), budget);
    let frames = Arc::new(ReplayFrames::new());
    let sink: Arc<dyn FrameSink> = Arc::<ReplayFrames>::clone(&frames);
    fresh.set_frame_sink(Some(sink));
    let len = ok(u64::try_from(bytes.len()));
    let mut reader = ok(CaptureReader::new(bytes, len));
    let options = ReplayOptions {
        backend_name: "recording".to_owned(),
        memory_limit: limit,
        fence_timeout: Duration::from_secs(5),
        pipeline_timeout: Duration::from_secs(5),
    };
    let report = ok(replay(&mut reader, &fresh, &Stub, &frames, &options));
    (report, fresh)
}

/// Results the scene compares: three readbacks with known bytes (the one after the draw is
/// unknown), two frames, two pipelines.
const COMPARED: u64 = 3 + 2 + 2;

/// Records in the scene without decoys: Start, 4 creates, 3 shaders with their 3 sources, 2
/// pipelines, 2 outcomes, the target, 3 lists with 2 frames, 4 readback results, 2 destroys, End.
const RECORDS: u64 = 1 + 4 + 6 + 2 + 2 + 1 + 3 + 2 + 4 + 2 + 1;

#[test]
fn a_capture_replays_on_a_fresh_backend_to_the_same_results() {
    let (bytes, original) = capture(false);
    let (report, fresh) = replay_fresh(&bytes, MEMORY, None);
    assert_eq!(
        report,
        ReplayReport {
            records: RECORDS,
            lists: 3,
            compared: COMPARED,
            ..ReplayReport::default()
        }
    );
    // The same calls in the same order on a backend just as fresh: the same log, ids and all.
    assert_eq!(fresh.snapshot(), original.snapshot());
    assert_eq!(fresh.lists(), original.lists());
    assert_eq!(fresh.completed(), FenceValue::new(3));
}

#[test]
fn ids_the_fresh_backend_hands_out_differently_are_mapped() {
    let (bytes, original) = capture(true);
    // The decoy buffer passes the replay's memory limit, and the shader has no origin: both are
    // refused, and every later buffer and shader has another id on the fresh backend.
    let (report, fresh) = replay_fresh(&bytes, MEMORY, Some(DECOY_BYTES - 1));
    assert_eq!(
        report,
        ReplayReport {
            records: RECORDS + 2,
            lists: 3,
            refused: 2,
            first_refused: Some(Finding {
                record: 1,
                kind: FindingKind::MissingShader
            }),
            compared: COMPARED,
            ..ReplayReport::default()
        }
    );
    let originals = original.lists();
    let replayed = fresh.lists();
    assert_eq!(replayed.len(), originals.len());
    assert_ne!(replayed, originals, "the lists name the fresh backend's ids");
}

#[test]
fn a_result_replay_cannot_reproduce_is_found() {
    let (bytes, _) = capture(false);
    // A fresh backend whose budget is one byte short of the buffer: the buffer is refused for
    // the memory limit, the list and the destroy naming it are refused, and the results the
    // list made are missing on replay.


    let (report, fresh) = replay_fresh(&bytes, 255, None);
    assert_eq!(
        report.first_refused,
        Some(Finding {
            record: 1,
            kind: FindingKind::MemoryLimit {
                needed: 256,
                limit: 255
            }
        })
    );
    // The buffer's create, the first list and the buffer's destroy.
    assert_eq!(report.refused, 3);
    assert_eq!(report.lists, 2);
    assert_eq!(fresh.lists().len(), 2);
    // The first frame and the first two readbacks are missing; the later frame and readback
    // compare equal; so do the pipelines.
    assert_eq!(report.differences, 3);
    assert_eq!(
        report.first_difference.map(|f| f.kind),
        Some(FindingKind::FrameMissing(1))
    );
    assert_eq!(report.compared, 1 + 1 + 2);
}
