//! The capturing backend: every state-changing call recorded in call order with what it
//! returned, results forwarded unchanged, frames after the call that delivered them, one total
//! order across threads, and an early stop that still closes the file.

use super::super::{
    BoundedField, CaptureBuffer, CaptureError, CaptureSink, CaptureWriter, CapturedTarget,
    CapturingBackend, Created, FrameResult, PipelineOutcome, PipelineResult, ReadbackResult,
    Record, RecordKind, ShaderCode, WaitSetting,
};
use super::{Knob, Spy, caps, naga, read, sampler_desc, shader, start, texture_desc, view_desc};
use crate::backend::{Backend, PipelineStatus, PresentTarget, WaitOutcome};
use crate::command::{
    BufferSlice, Command, CommandList, CopyRegion, Marker, PresentDesc, PresentTransform, Rect,
    StagingSlice,
};
use crate::frame::{Frame, FramePixels, FrameSink};
use crate::id::{BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId};
use crate::pipeline::ComputePipelineDesc;
use crate::recording::RecordingBackend;
use crate::resource::{BufferDesc, BufferUsage, Extent3d, TextureDesc};
use crate::shader::{ShaderOrigin, ShaderSource, ShaderStage};
use std::sync::{Arc, Weak};
use std::time::Duration;

/// The tests' memory limit.
const MEMORY: u64 = 1 << 20;

/// A recording backend wrapped in a capturing one, writing to a buffer.
fn wrapped(
    limit: u64,
) -> (
    Arc<RecordingBackend>,
    Arc<CapturingBackend>,
    Arc<CaptureBuffer>,
) {
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
fn calls_that_change_nothing_are_forwarded_unrecorded() {
    let (inner, capture, buffer) = wrapped(u64::MAX);
    let backend: &dyn Backend = &*capture;
    assert_eq!(backend.caps(), inner.caps());
    assert_eq!(backend.budget(), inner.budget());
    assert_eq!(backend.completed(), FenceValue::ZERO);
    assert_eq!(backend.staging(4).map(|s| s.len()), Ok(4));
    assert_eq!(
        backend.wait(FenceValue::ZERO, Duration::ZERO),
        Ok(WaitOutcome::Reached)
    );
    assert_eq!(backend.take_present_feedback().len(), 0);
    capture.finish().expect("end");
    let records = read(&buffer.bytes()).expect("a well-formed capture");
    assert_eq!(kinds(&records), [RecordKind::Start, RecordKind::End]);
    let Record::Start(start) = &records[0] else {
        unreachable!("start")
    };
    assert_eq!(
        (
            start.build.as_str(),
            start.backend.as_str(),
            &start.caps,
            start.wait,
            start.memory_limit
        ),
        ("build 7", "recording", &caps(), WaitSetting::Skip, MEMORY)
    );
}

/// The id a `Create*` record says was made.
fn made(record: &Record) -> Option<ResourceId> {
    match record {
        Record::CreateBuffer(c) => c.result.made().map(ResourceId::Buffer),
        Record::CreateTexture(c) => c.result.made().map(ResourceId::Texture),
        Record::CreateView(c) => c.result.made().map(ResourceId::View),
        Record::CreateSampler(c) => c.result.made().map(ResourceId::Sampler),
        Record::CreateShader(c) => c.result.made().map(ResourceId::Shader),
        Record::CreateComputePipeline(c) => c.result.made().map(ResourceId::Pipeline),
        _ => None,
    }
}

#[test]
fn creates_are_recorded_in_order_with_what_they_returned() {
    let (_, capture, buffer) = wrapped(u64::MAX);
    let backend: &dyn Backend = &*capture;
    let too_big = TextureDesc {
        size: Extent3d {
            width: 1 << 20,
            ..texture_desc().size
        },
        ..texture_desc()
    };
    let refused = backend.create_texture(&too_big).expect_err("too wide");
    let tex = backend.create_texture(&texture_desc()).expect("texture");
    let view = backend.create_view(tex, &view_desc()).expect("view");
    let sampler = backend.create_sampler(&sampler_desc()).expect("sampler");
    let origin = ShaderOrigin {
        record: Arc::from(vec![9, 9]),
        stage_index: 3,
    };
    let shaders = [
        shader(ShaderStage::Compute, naga(), Some(origin)),
        shader(ShaderStage::Vertex, ShaderSource::SpirV(vec![7]), None),
        shader(ShaderStage::Compute, naga(), None),
    ]
    .map(|desc| backend.create_shader(desc).expect("shader"));
    let bad_pipeline = super::golden::golden_pipeline();
    let refused_pipeline = backend
        .create_graphics_pipeline(bad_pipeline.clone())
        .expect_err("not this device's");
    capture.finish().expect("end");

    let records = read(&buffer.bytes()).expect("a well-formed capture");
    assert_eq!(
        kinds(&records),
        [
            RecordKind::Start,
            RecordKind::CreateTexture,
            RecordKind::CreateTexture,
            RecordKind::CreateView,
            RecordKind::CreateSampler,
            RecordKind::ShaderSource,
            RecordKind::CreateShader,
            RecordKind::CreateShader,
            RecordKind::CreateShader,
            RecordKind::CreateGraphicsPipeline,
            RecordKind::End,
        ]
    );
    let Record::CreateTexture(first) = &records[1] else {
        unreachable!("texture")
    };
    assert_eq!(
        (first.desc, &first.result),
        (too_big, &Created::Refused(refused.to_string()))
    );
    assert_eq!(made(&records[2]), Some(ResourceId::Texture(tex)));
    let Record::CreateView(made_view) = &records[3] else {
        unreachable!("view")
    };
    assert_eq!((made_view.texture, made_view.desc), (tex, view_desc()));
    assert_eq!(made(&records[3]), Some(ResourceId::View(view)));
    assert_eq!(made(&records[4]), Some(ResourceId::Sampler(sampler)));
    let codes: Vec<(ShaderCode, Option<ResourceId>)> = records[6..9]
        .iter()
        .map(|r| match r {
            Record::CreateShader(c) => (c.code.clone(), made(r)),
            _ => unreachable!("a shader"),
        })
        .collect();
    let [from_origin, spirv, no_origin] = shaders.map(ResourceId::Shader);
    assert_eq!(
        codes,
        [
            (
                ShaderCode::Origin {
                    source: 0,
                    stage_index: 3
                },
                Some(from_origin)
            ),
            (ShaderCode::SpirV(vec![7]), Some(spirv)),
            (ShaderCode::Missing, Some(no_origin)),
        ]
    );
    let Record::CreateGraphicsPipeline(pipeline) = &records[9] else {
        unreachable!("pipeline")
    };
    assert_eq!(
        (&pipeline.desc, &pipeline.result),
        (
            &bad_pipeline,
            &Created::Refused(refused_pipeline.to_string())
        )
    );
}

/// A list that uploads [`PIXELS`] to `tex`, reads `buf` back as `rb`, and presents `view` as
/// dumped frame 7.
fn upload_read_present(
    tex: crate::id::TextureId,
    buf: BufferId,
    view: crate::id::ViewId,
    rb: ReadbackId,
) -> CommandList {
    CommandList {
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
                id: rb,
            },
            Command::Present(PresentDesc {
                view,
                src_rect: Rect {
                    width: 2,
                    height: 2,
                    ..Rect::default()
                },
                transform: PresentTransform::Identity,
                frame: 7,
                dump: true,
            }),
        ],
    }
}

#[test]
fn lists_and_results_are_recorded_in_order_with_frames_after_their_list() {
    let (inner, capture, buffer) = wrapped(u64::MAX);
    tee(&inner, &capture);
    let backend: &dyn Backend = &*capture;
    let buf = backend
        .create_buffer(&BufferDesc {
            size: 64,
            usage: BufferUsage::ALL,
        })
        .expect("buffer");
    let tex = backend.create_texture(&texture_desc()).expect("texture");
    let view = backend.create_view(tex, &view_desc()).expect("view");
    let sampler = backend.create_sampler(&sampler_desc()).expect("sampler");
    let cs = backend
        .create_shader(shader(
            ShaderStage::Compute,
            ShaderSource::Msl(String::new()),
            None,
        ))
        .expect("shader");
    let pipeline = backend
        .create_compute_pipeline(ComputePipelineDesc { shader: cs })
        .expect("pipeline");
    let rb = ReadbackId::new(5, 0);
    let fence = backend
        .submit(upload_read_present(tex, buf, view, rb))
        .expect("list");
    let readback = backend.take_readback(rb);
    let nothing = backend.take_readback(rb);
    assert_eq!(backend.pipeline_status(pipeline), PipelineStatus::Ready);
    assert_eq!(backend.pipeline_status(pipeline), PipelineStatus::Ready);
    assert_eq!(backend.set_present_target(PresentTarget::Offscreen), Ok(()));
    backend.destroy(ResourceId::Sampler(sampler), fence);
    capture.finish().expect("end");

    let records = read(&buffer.bytes()).expect("a well-formed capture");
    let Record::Submit(submitted) = &records[7] else {
        unreachable!("submit: {records:?}")
    };
    assert_eq!(submitted.result, Created::Made(fence));
    assert_eq!(submitted.list.commands.len(), 3);
    let frame = Frame {
        frame: 7,
        width: 2,
        height: 2,
        transform: PresentTransform::Identity,
        pixels: FramePixels::Rgba8(PIXELS.to_vec()),
    };
    assert_eq!(
        records[8..],
        [
            Record::FrameResult(FrameResult::of(&frame)),
            Record::ReadbackResult(ReadbackResult {
                id: rb,
                value: readback,
            }),
            Record::ReadbackResult(ReadbackResult {
                id: rb,
                value: nothing,
            }),
            Record::PipelineOutcome(PipelineOutcome {
                pipeline,
                outcome: PipelineResult::Ready
            }),
            Record::PresentTarget(super::super::TargetRecord {
                target: CapturedTarget::Offscreen,
                result: Created::Made(()),
            }),
            Record::Destroy(super::super::Destroy {
                id: ResourceId::Sampler(sampler),
                after: fence
            }),
            Record::End,
        ]
    );
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
        .create_shader(shader(
            ShaderStage::Compute,
            ShaderSource::Msl(String::new()),
            None,
        ))
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
    // Other calls in between do not make it settle again.
    b.submit(CommandList::new()).expect("an empty list");
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
fn a_pipeline_id_handed_out_again_after_its_destroy_is_recorded_again() {
    // The spy hands out the same pipeline id every time, as a backend may once the id is free.
    let buffer = Arc::new(CaptureBuffer::new());
    let sink: Arc<dyn CaptureSink> = Arc::<CaptureBuffer>::clone(&buffer);
    let capture = CapturingBackend::new(
        Arc::new(Spy::new(Knob::RepeatIds)),
        sink,
        String::new(),
        String::new(),
        WaitSetting::Wait,
        u64::MAX,
    )
    .expect("start");
    let cs = capture
        .create_shader(shader(
            ShaderStage::Compute,
            ShaderSource::Msl(String::new()),
            None,
        ))
        .expect("shader");
    let desc = ComputePipelineDesc { shader: cs };
    let first = capture.create_compute_pipeline(desc).expect("pipeline");
    // The recording backend under the spy never made this id, so it reports it failed.
    assert_eq!(capture.pipeline_status(first), PipelineStatus::Failed);
    capture.destroy(ResourceId::Pipeline(first), FenceValue::ZERO);
    let second = capture.create_compute_pipeline(desc).expect("pipeline");
    assert_eq!(first, second);
    assert_eq!(capture.pipeline_status(second), PipelineStatus::Failed);
    capture.finish().expect("end");
    let outcomes = read(&buffer.bytes())
        .expect("reads")
        .into_iter()
        .filter(|r| r.kind() == RecordKind::PipelineOutcome)
        .count();
    assert_eq!(outcomes, 2);
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
    assert!(
        buffer.bytes().len() > before,
        "written before record_frame returned"
    );
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
            what: BoundedField::ShaderSource,
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
    assert_eq!(
        b.submit(list).map_err(|e| e.to_string()),
        expected.map_err(|e| e.to_string())
    );
    assert!(matches!(
        capture.stopped(),
        Some(CaptureError::TooLarge {
            what: BoundedField::DispatchBindings,
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
    let inner: Arc<dyn Backend> = Arc::new(Spy::new(Knob::RepeatIds));
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

#[test]
fn a_window_target_is_recorded_without_its_handles() {
    use crate::window::raw_window_handle::{
        DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
    };
    use crate::window::{WindowHandles, WindowSize};
    /// A window whose handles are never available: the recording backend does not ask.
    struct NoWindow;
    impl HasWindowHandle for NoWindow {
        fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
            Err(HandleError::Unavailable)
        }
    }
    impl HasDisplayHandle for NoWindow {
        fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
            Err(HandleError::Unavailable)
        }
    }
    let (inner, capture, buffer) = wrapped(u64::MAX);
    let b: &dyn Backend = &*capture;
    let window = WindowHandles::new(Arc::new(NoWindow), WindowSize::new(4, 3));
    assert_eq!(b.set_present_target(PresentTarget::Window(window)), Ok(()));
    assert_eq!(b.set_present_target(PresentTarget::Offscreen), Ok(()));
    capture.finish().expect("end");
    // The window reached the wrapped backend; the capture holds only which kind it was.
    assert_eq!(
        inner.snapshot(),
        "present target window\npresent target offscreen\n"
    );
    let records = read(&buffer.bytes()).expect("a well-formed capture");
    assert_eq!(
        records[1..],
        [
            Record::PresentTarget(super::super::TargetRecord {
                target: CapturedTarget::Window,
                result: Created::Made(()),
            }),
            Record::PresentTarget(super::super::TargetRecord {
                target: CapturedTarget::Offscreen,
                result: Created::Made(()),
            }),
            Record::End,
        ]
    );
}
