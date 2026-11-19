//! The capturing backend: every state-changing call recorded in call order with what it
//! returned, results forwarded unchanged, frames after the call that delivered them, one total
//! order across threads, and an early stop that still closes the file.

use super::super::{
    CaptureBuffer, CaptureError, CaptureSink, CaptureWriter, CapturedTarget, CapturingBackend,
    Created, FrameResult, PipelineOutcome, PipelineResult, ReadbackResult, Record, RecordKind,
    ShaderCode, WaitSetting,
};
use super::{
    Knobs, Spy, caps, naga, read, sampler_desc, shader, start, texture_desc, view_desc,
};
use crate::backend::{
    Backend, MemoryBudget, PipelineStatus, PresentFeedback, PresentTarget, Readback, WaitOutcome,
};
use crate::caps::Caps;
use crate::command::{
    BufferSlice, Command, CommandList, CopyRegion, Marker, PresentDesc, PresentTransform, Rect,
    StagingSlice,
};
use crate::error::RenderError;
use crate::format::PixelFormat;
use crate::frame::{Frame, FramePixels, FrameSink};
use crate::id::{
    BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId,
    ViewId,
};
use crate::pipeline::{ComputePipelineDesc, GraphicsPipelineDesc};
use crate::recording::RecordingBackend;
use crate::resource::{
    AddressMode, BorderColor, BufferDesc, BufferUsage, Extent3d, FilterMode, ReductionMode,
    SamplerDesc, Swizzle, TextureDesc, TextureDimension, TextureUsage, ViewDesc, ViewDimension,
};
use crate::shader::{ShaderDesc, ShaderInterface, ShaderOrigin, ShaderSource, ShaderStage};
use std::sync::{Arc, Weak};
use std::time::Duration;

/// The tests' memory limit.
const MEMORY: u64 = 1 << 20;

/// A recording backend wrapped in a capturing one, writing to a buffer.
fn wrapped(limit: u64) -> (Arc<RecordingBackend>, Arc<CapturingBackend>, Arc<CaptureBuffer>) {
    let inner = Arc::new(RecordingBackend::new(caps(), MEMORY));
    let buffer = Arc::new(CaptureBuffer::new());
    let dyn_inner: Arc<dyn Backend> = Arc::<RecordingBackend>::clone(&inner);
    let sink: Arc<dyn CaptureSink> = Arc::<CaptureBuffer>::clone(&buffer);
    let capture = CapturingBackend::new(
        dyn_inner,
        sink,
        "build 7".to_owned(),
        "recording".to_owned(),
        WaitSetting::Skip,
        limit,
    )
    .expect("the start fits");
    (inner, Arc::new(capture), buffer)
}

/// The tee the wiring installs: frames go to the capture.
#[derive(Debug)]
struct Tee(Weak<CapturingBackend>);

impl FrameSink for Tee {
    fn frame(&self, frame: Frame) {
        if let Some(capture) = self.0.upgrade() {
            capture.record_frame(&frame);
        }
    }
}

fn tee(inner: &RecordingBackend, capture: &Arc<CapturingBackend>) {
    let sink: Arc<dyn FrameSink> = Arc::new(Tee(Arc::downgrade(capture)));
    inner.set_frame_sink(Some(sink));
}

fn kinds(records: &[Record]) -> Vec<RecordKind> {
    records.iter().map(Record::kind).collect()
}

/// Sixteen bytes of RGBA8.
const PIXELS: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

#[test]
fn every_state_changing_call_is_recorded_in_order_with_its_result() {
    let (inner, capture, buffer) = wrapped(u64::MAX);
    tee(&inner, &capture);
    let b: &dyn Backend = &*capture;

    // Calls that change nothing are forwarded unrecorded.
    assert_eq!(b.caps(), inner.caps());
    assert_eq!(b.budget(), inner.budget());
    assert_eq!(b.completed(), FenceValue::ZERO);
    assert_eq!(b.staging(4).map(|s| s.len()), Ok(4));
    assert_eq!(b.wait(FenceValue::ZERO, Duration::ZERO), Ok(WaitOutcome::Reached));
    assert!(b.take_present_feedback().is_empty());

    let buf = b
        .create_buffer(&BufferDesc {
            size: 64,
            usage: BufferUsage::ALL,
        })
        .expect("buffer");
    let too_big = TextureDesc {
        size: Extent3d {
            width: 1 << 20,
            ..texture_desc().size
        },
        ..texture_desc()
    };
    let refused = b.create_texture(&too_big).expect_err("too wide");
    let tex = b.create_texture(&texture_desc()).expect("texture");
    let view = b.create_view(tex, &view_desc()).expect("view");
    let sampler = b.create_sampler(&sampler_desc()).expect("sampler");
    let origin = ShaderOrigin {
        record: Arc::from(vec![9, 9]),
        stage_index: 3,
    };
    let from_origin = b
        .create_shader(shader(ShaderStage::Compute, naga(), Some(origin)))
        .expect("shader");
    let spirv = b
        .create_shader(shader(ShaderStage::Vertex, ShaderSource::SpirV(vec![7]), None))
        .expect("shader");
    let no_origin = b
        .create_shader(shader(ShaderStage::Compute, naga(), None))
        .expect("shader");
    let pipeline = b
        .create_compute_pipeline(ComputePipelineDesc {
            shader: from_origin,
        })
        .expect("pipeline");
    let bad_pipeline = super::golden::golden_pipeline();
    let refused_pipeline = b
        .create_graphics_pipeline(bad_pipeline.clone())
        .expect_err("not this device's");
    let fence = b
        .submit(CommandList {
            commands: vec![
                Command::Upload {
                    from: StagingSlice::from_bytes(PIXELS.to_vec()),
                    to: CopyRegion::Texture(crate::command::TextureRegion {
                        texture: tex,
                        mip: 0,
                        origin: crate::resource::Origin3d::default(),
                        extent: texture_desc().size,
                    }),
                },
                Command::Readback {
                    from: CopyRegion::Buffer(BufferSlice {
                        buffer: buf,
                        offset: 0,
                        size: 4,
                    }),
                    id: ReadbackId::new(5, 0),
                },
                Command::Present(PresentDesc {
                    view,
                    src_rect: Rect {
                        x: 0,
                        y: 0,
                        width: 2,
                        height: 2,
                    },
                    transform: PresentTransform::Identity,
                    frame: 7,
                    dump: true,
                }),
            ],
        })
        .expect("list");
    let readback = b.take_readback(ReadbackId::new(5, 0));
    let nothing = b.take_readback(ReadbackId::new(5, 0));
    assert_eq!(b.pipeline_status(pipeline), PipelineStatus::Ready);
    assert_eq!(b.pipeline_status(pipeline), PipelineStatus::Ready);
    assert_eq!(b.set_present_target(PresentTarget::Offscreen), Ok(()));
    b.destroy(ResourceId::Sampler(sampler), fence);
    capture.finish().expect("end");

    let records = read(&buffer.bytes()).expect("a well-formed capture");
    assert_eq!(
        kinds(&records),
        [
            RecordKind::Start,
            RecordKind::CreateBuffer,
            RecordKind::CreateTexture,
            RecordKind::CreateTexture,
            RecordKind::CreateView,
            RecordKind::CreateSampler,
            RecordKind::ShaderSource,
            RecordKind::CreateShader,
            RecordKind::CreateShader,
            RecordKind::CreateShader,
            RecordKind::CreateComputePipeline,
            RecordKind::CreateGraphicsPipeline,
            RecordKind::Submit,
            RecordKind::FrameResult,
            RecordKind::ReadbackResult,
            RecordKind::ReadbackResult,
            RecordKind::PipelineOutcome,
            RecordKind::PresentTarget,
            RecordKind::Destroy,
            RecordKind::End,
        ]
    );
    let Record::Start(s) = &records[0] else {
        unreachable!("start")
    };
    assert_eq!(
        (s.build.as_str(), s.backend.as_str(), &s.caps, s.wait, s.memory_limit),
        ("build 7", "recording", &caps(), WaitSetting::Skip, MEMORY)
    );
    let made = |r: &Record| match r {
        Record::CreateBuffer(c) => c.result.made().map(ResourceId::Buffer),
        Record::CreateTexture(c) => c.result.made().map(ResourceId::Texture),
        Record::CreateView(c) => c.result.made().map(ResourceId::View),
        Record::CreateSampler(c) => c.result.made().map(ResourceId::Sampler),
        Record::CreateShader(c) => c.result.made().map(ResourceId::Shader),
        Record::CreateComputePipeline(c) => c.result.made().map(ResourceId::Pipeline),
        _ => None,
    };
    assert_eq!(made(&records[1]), Some(ResourceId::Buffer(buf)));
    let Record::CreateTexture(t) = &records[2] else {
        unreachable!("texture")
    };
    assert_eq!(
        (t.desc, &t.result),
        (too_big, &Created::Refused(refused.to_string()))
    );
    assert_eq!(made(&records[3]), Some(ResourceId::Texture(tex)));
    let Record::CreateView(v) = &records[4] else {
        unreachable!("view")
    };
    assert_eq!((v.texture, v.desc), (tex, view_desc()));
    assert_eq!(made(&records[4]), Some(ResourceId::View(view)));
    assert_eq!(made(&records[5]), Some(ResourceId::Sampler(sampler)));
    let codes: Vec<(ShaderCode, Option<ResourceId>)> = records[7..10]
        .iter()
        .map(|r| match r {
            Record::CreateShader(c) => (c.code.clone(), made(r)),
            _ => unreachable!("a shader"),
        })
        .collect();
    assert_eq!(
        codes,
        [
            (
                ShaderCode::Origin {
                    source: 0,
                    stage_index: 3
                },
                Some(ResourceId::Shader(from_origin))
            ),
            (ShaderCode::SpirV(vec![7]), Some(ResourceId::Shader(spirv))),
            (ShaderCode::Missing, Some(ResourceId::Shader(no_origin))),
        ]
    );
    assert_eq!(made(&records[10]), Some(ResourceId::Pipeline(pipeline)));
    let Record::CreateGraphicsPipeline(g) = &records[11] else {
        unreachable!("pipeline")
    };
    assert_eq!(
        (&g.desc, &g.result),
        (&bad_pipeline, &Created::Refused(refused_pipeline.to_string()))
    );
    let Record::Submit(sub) = &records[12] else {
        unreachable!("submit")
    };
    assert_eq!(sub.result, Created::Made(fence));
    assert_eq!(sub.list.commands.len(), 3);
    assert_eq!(
        records[13],
        Record::FrameResult(FrameResult::of(&Frame {
            frame: 7,
            width: 2,
            height: 2,
            transform: PresentTransform::Identity,
            pixels: FramePixels::Rgba8(PIXELS.to_vec()),
        }))
    );
    assert_eq!(
        records[14],
        Record::ReadbackResult(ReadbackResult {
            id: ReadbackId::new(5, 0),
            value: readback,
        })
    );
    assert_eq!(
        records[15],
        Record::ReadbackResult(ReadbackResult {
            id: ReadbackId::new(5, 0),
            value: nothing,
        })
    );
    assert_eq!(
        records[16],
        Record::PipelineOutcome(PipelineOutcome {
            pipeline,
            outcome: PipelineResult::Ready
        })
    );
    let Record::PresentTarget(p) = &records[17] else {
        unreachable!("target")
    };
    assert_eq!((p.target, &p.result), (CapturedTarget::Offscreen, &Created::Made(())));
    let Record::Destroy(d) = records[18] else {
        unreachable!("destroy")
    };
    assert_eq!((d.id, d.after), (ResourceId::Sampler(sampler), fence));
    assert_eq!(capture.stopped(), None);
}

#[test]
fn the_hash_is_sha256_of_the_pixels() {
    // The published SHA-256 of "abc".
    let frame = Frame {
        frame: 1,
        width: 1,
        height: 1,
        transform: PresentTransform::Identity,
        pixels: FramePixels::Rgba8(b"abc".to_vec()),
    };
    let expected = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22,
        0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00,
        0x15, 0xad,
    ];
    assert_eq!(FrameResult::of(&frame).sha256, Some(expected));
    let unknown = Frame {
        pixels: FramePixels::Unknown,
        ..frame
    };
    assert_eq!(
        FrameResult::of(&unknown),
        FrameResult {
            frame: 1,
            width: 1,
            height: 1,
            sha256: None
        }
    );
}

#[test]
fn a_pipeline_outcome_is_recorded_once_when_it_settles() {
    let (inner, capture, buffer) = wrapped(u64::MAX);
    let b: &dyn Backend = &*capture;
    let shader_id = b
        .create_shader(shader(ShaderStage::Compute, ShaderSource::Msl(String::new()), None))
        .expect("shader");
    let p = b
        .create_compute_pipeline(ComputePipelineDesc { shader: shader_id })
        .expect("pipeline");
    inner
        .script_pipeline(
            p,
            vec![
                PipelineStatus::Compiling,
                PipelineStatus::Failed,
                PipelineStatus::Ready,
            ],
        )
        .expect("script");
    assert_eq!(b.pipeline_status(p), PipelineStatus::Compiling);
    assert_eq!(b.pipeline_status(p), PipelineStatus::Failed);
    assert_eq!(b.pipeline_status(p), PipelineStatus::Ready);
    // An id the capture never created: not recorded.
    let stranger = PipelineId::new(99, 0);
    assert_eq!(b.pipeline_status(stranger), PipelineStatus::Failed);
    // Destroyed and made again, a pipeline's outcome is recorded again.
    b.destroy(ResourceId::Pipeline(p), FenceValue::ZERO);
    let again = b
        .create_compute_pipeline(ComputePipelineDesc { shader: shader_id })
        .expect("pipeline");
    assert_eq!(b.pipeline_status(again), PipelineStatus::Ready);
    capture.finish().expect("end");
    let outcomes: Vec<Record> = read(&buffer.bytes())
        .expect("reads")
        .into_iter()
        .filter(|r| r.kind() == RecordKind::PipelineOutcome)
        .collect();
    assert_eq!(
        outcomes,
        [
            Record::PipelineOutcome(PipelineOutcome {
                pipeline: p,
                outcome: PipelineResult::Failed
            }),
            Record::PipelineOutcome(PipelineOutcome {
                pipeline: again,
                outcome: PipelineResult::Ready
            }),
        ]
    );
}

#[test]
fn a_frame_recorded_while_the_lock_is_free_is_written_at_once() {
    let (_, capture, buffer) = wrapped(u64::MAX);
    let before = buffer.bytes().len();
    capture.record_frame(&Frame {
        frame: 3,
        width: 1,
        height: 1,
        transform: PresentTransform::Identity,
        pixels: FramePixels::Unknown,
    });
    assert!(buffer.bytes().len() > before, "written before record_frame returned");
    // Finishing writes nothing more than End.
    capture.finish().expect("end");
    assert_eq!(
        kinds(&read(&buffer.bytes()).expect("reads")),
        [RecordKind::Start, RecordKind::FrameResult, RecordKind::End]
    );
}

#[test]
fn a_record_past_the_size_limit_stops_the_capture_and_calls_go_on() {
    // Room for the start, the closing records and one small record, not two.
    let probe = wrapped(u64::MAX);
    let opening = u64::try_from(probe.2.bytes().len()).expect("small");
    let marker = CommandList {
        commands: vec![Command::Marker(Marker { submit: 1, draw: 2 })],
    };
    // A submit of one marker with a one-byte fence: tag, length, count, three command bytes,
    // the result's tag and fence.
    let one_submit = 1 + 1 + 1 + 3 + 2;
    let close = (1 + 1 + 10) + 2;
    let (_, capture, buffer) = wrapped(opening + one_submit + close);
    let b: &dyn Backend = &*capture;
    assert_eq!(b.submit(marker.clone()), Ok(FenceValue::new(1)));
    assert_eq!(capture.stopped(), None);
    // The second does not fit: the capture stops, the call goes through.
    assert_eq!(b.submit(marker.clone()), Ok(FenceValue::new(2)));
    let needed = opening + 2 * one_submit + close;
    assert_eq!(
        capture.stopped(),
        Some(CaptureError::OverLimit {
            record: 2,
            needed,
            limit: needed - one_submit
        })
    );
    let closed = buffer.bytes();
    assert_eq!(b.submit(marker), Ok(FenceValue::new(3)));
    assert!(b.create_sampler(&sampler_desc()).is_ok());
    assert_eq!(buffer.bytes(), closed, "nothing recorded after the stop");
    assert_eq!(capture.finish(), Err(CaptureError::Finished));
    let records = read(&closed).expect("closed well formed");
    assert_eq!(
        kinds(&records),
        [
            RecordKind::Start,
            RecordKind::Submit,
            RecordKind::Limit,
            RecordKind::End
        ]
    );
    assert_eq!(
        records[2],
        Record::Limit(super::super::Limit {
            fence: FenceValue::new(1)
        })
    );
}

#[test]
fn a_shader_source_past_its_bound_stops_the_capture_and_the_shader_is_made() {
    let (_, capture, buffer) = wrapped(u64::MAX);
    let b: &dyn Backend = &*capture;
    let big = usize::try_from(super::super::MAX_SHADER_SOURCE_BYTES).expect("fits") + 1;
    let origin = ShaderOrigin {
        record: Arc::from(vec![0; big]),
        stage_index: 0,
    };
    assert!(
        b.create_shader(shader(ShaderStage::Compute, naga(), Some(origin)))
            .is_ok()
    );
    assert!(matches!(
        capture.stopped(),
        Some(CaptureError::TooLarge {
            what: "shader source",
            ..
        })
    ));
    assert_eq!(
        kinds(&read(&buffer.bytes()).expect("closed")),
        [RecordKind::Start, RecordKind::Limit, RecordKind::End]
    );
}

#[test]
fn a_list_past_its_bounds_stops_the_capture_and_is_submitted() {
    let (inner, capture, buffer) = wrapped(u64::MAX);
    let b: &dyn Backend = &*capture;
    let draws = usize::try_from(crate::command::MAX_COMMAND_BINDINGS).expect("fits") + 1;
    let list = CommandList {
        commands: vec![Command::Dispatch(crate::command::DispatchDesc {
            pipeline: PipelineId::new(1, 0),
            bindings: vec![
                crate::command::Binding {
                    group: 0,
                    binding: 0,
                    resource: crate::command::BindingResource::Sampler(SamplerId::new(1, 0)),
                };
                draws
            ],
            params: crate::command::DispatchParams::Direct { x: 1, y: 1, z: 1 },
        })],
    };
    // The recording backend refuses it (no such pipeline); the refusal is returned unchanged.
    let expected = inner.submit(list.clone());
    assert_eq!(b.submit(list).map_err(|e| e.to_string()), expected.map_err(|e| e.to_string()));
    assert!(matches!(
        capture.stopped(),
        Some(CaptureError::TooLarge {
            what: "dispatch bindings",
            ..
        })
    ));
    assert_eq!(
        kinds(&read(&buffer.bytes()).expect("closed")),
        [RecordKind::Start, RecordKind::Limit, RecordKind::End]
    );
}


#[test]
fn a_backend_handing_out_a_live_id_stops_the_capture() {
    let inner: Arc<dyn Backend> = Arc::new(Spy::new(Knobs {
        repeat_buffer: true,
        ..Knobs::default()
    }));
    let buffer = Arc::new(CaptureBuffer::new());
    let sink: Arc<dyn CaptureSink> = Arc::<CaptureBuffer>::clone(&buffer);
    let capture = CapturingBackend::new(
        inner,
        sink,
        String::new(),
        String::new(),
        WaitSetting::Wait,
        u64::MAX,
    )
    .expect("start");
    let desc = BufferDesc {
        size: 4,
        usage: BufferUsage::VERTEX,
    };
    assert_eq!(capture.create_buffer(&desc), Ok(BufferId::new(1, 1)));
    assert_eq!(capture.create_buffer(&desc), Ok(BufferId::new(1, 1)));
    assert_eq!(
        capture.stopped(),
        Some(CaptureError::IdLive {
            record: 2,
            id: ResourceId::Buffer(BufferId::new(1, 1))
        })
    );
    assert_eq!(
        kinds(&read(&buffer.bytes()).expect("closed")),
        [
            RecordKind::Start,
            RecordKind::CreateBuffer,
            RecordKind::Limit,
            RecordKind::End
        ]
    );
}

#[test]
fn the_constructor_refuses_a_limit_too_small_and_an_ended_writer() {
    let inner: Arc<dyn Backend> = Arc::new(RecordingBackend::new(caps(), MEMORY));
    let sink: Arc<dyn CaptureSink> = Arc::new(CaptureBuffer::new());
    assert!(matches!(
        CapturingBackend::new(
            Arc::clone(&inner),
            Arc::clone(&sink),
            String::new(),
            String::new(),
            WaitSetting::Wait,
            10,
        ),
        Err(CaptureError::LimitTooSmall { limit: 10, .. })
    ));
    let mut writer = CaptureWriter::new(sink, start(), u64::MAX).expect("start");
    writer.finish().expect("end");
    assert!(matches!(
        CapturingBackend::with_writer(inner, writer),
        Err(CaptureError::Finished)
    ));
}

#[test]
fn two_threads_submitting_get_one_order() {
    let (inner, capture, buffer) = wrapped(u64::MAX);
    let lists = 200;
    let threads: Vec<_> = (0..2u64)
        .map(|t| {
            let capture = Arc::clone(&capture);
            std::thread::spawn(move || {
                for i in 0..lists {
                    let list = CommandList {
                        commands: vec![Command::Marker(Marker { submit: t, draw: i })],
                    };
                    capture.submit(list).expect("a marker list");
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().expect("the thread ran");
    }
    capture.finish().expect("end");
    // The reader refuses fences out of order, so a record written in another order than the
    // calls reached the backend fails here.
    let records = read(&buffer.bytes()).expect("one order");
    let submits = records
        .iter()
        .filter(|r| r.kind() == RecordKind::Submit)
        .count();
    assert_eq!(u64::try_from(submits).expect("small"), 2 * lists);
    assert_eq!(inner.completed(), FenceValue::new(2 * lists));
}
