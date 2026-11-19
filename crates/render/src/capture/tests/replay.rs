//! Replay: ids and fences mapped, lists naming ids not held refused and counted, destroys at
//! their mapped fence, the memory limit, shaders through a resolver, and every comparison,
//! on captures made with `CaptureBuilder` and replayed on a fresh recording backend.

use super::super::replay::MAX_REPLAY_FRAMES;
use super::super::{
    CaptureBuilder, CaptureError, CaptureReader, Create, Created, Finding, FindingKind, Record,
    ReplayError, ReplayFrames, ReplayOptions, ReplayReport, ResolveError, ShaderResolver, Start,
    Submit, WaitSetting, map_ids, replay,
};
use super::super::{CapturedTarget, Limit, PipelineResult, TargetRecord};
use super::{Knob, Spy, caps, naga, shader, texture_desc, view_desc};
use crate::backend::{Backend, PipelineStatus, Readback};
use crate::command::tests::every_command;
use crate::command::{
    BufferSlice, Command, CommandList, CopyRegion, PresentDesc, PresentTransform, Rect,
    StagingSlice, TextureRegion,
};
use crate::frame::{Frame, FramePixels, FrameSink};
use crate::id::{BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, ShaderId};
use crate::pipeline::ComputePipelineDesc;
use crate::resource::{BufferDesc, BufferUsage, Origin3d};
use crate::shader::{ShaderDesc, ShaderOrigin, ShaderSource, ShaderStage};
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

fn options() -> ReplayOptions {
    ReplayOptions {
        backend_name: "builder".to_owned(),
        memory_limit: None,
        fence_timeout: Duration::from_secs(5),
        pipeline_timeout: Duration::from_millis(20),
    }
}

/// Translates origins by their first byte: `X` cannot be translated, `F` stops the replay,
/// `E` gives another entry name, anything else a compute shader named `main`.
struct Stub;

impl ShaderResolver for Stub {
    fn resolve(&self, origin: &ShaderOrigin) -> Result<ShaderDesc, ResolveError> {
        match origin.record.first() {
            Some(b'X') => Err(ResolveError::Shader("cannot translate".to_owned())),
            Some(b'F') => Err(ResolveError::Fatal("translator 2, capture 1".to_owned())),
            Some(b'E') => Ok(ShaderDesc {
                entry: "other".to_owned(),
                ..shader(ShaderStage::Compute, naga(), None)
            }),
            _ => Ok(shader(ShaderStage::Compute, naga(), None)),
        }
    }
}

fn run_with(
    bytes: &[u8],
    backend: &dyn Backend,
    frames: &ReplayFrames,
    options: &ReplayOptions,
) -> Result<ReplayReport, ReplayError> {
    let len = u64::try_from(bytes.len()).expect("small");
    let mut reader = CaptureReader::new(bytes, len)?;
    replay(&mut reader, backend, &Stub, frames, options)
}

fn run(bytes: &[u8], backend: &dyn Backend) -> Result<ReplayReport, ReplayError> {
    run_with(bytes, backend, &ReplayFrames::new(), &options())
}

fn builder() -> CaptureBuilder {
    CaptureBuilder::new(caps()).expect("start")
}

fn buffer_desc(size: u64) -> BufferDesc {
    BufferDesc {
        size,
        usage: BufferUsage::ALL,
    }
}

fn whole(buffer: BufferId, size: u64) -> CopyRegion {
    CopyRegion::Buffer(BufferSlice {
        buffer,
        offset: 0,
        size,
    })
}

fn list(commands: Vec<Command>) -> CommandList {
    CommandList { commands }
}

fn upload(buffer: BufferId, bytes: &[u8]) -> Command {
    Command::Upload {
        from: StagingSlice::from_bytes(bytes.to_vec()),
        to: whole(buffer, u64::try_from(bytes.len()).expect("small")),
    }
}

fn readback(buffer: BufferId, size: u64, id: ReadbackId) -> Command {
    Command::Readback {
        from: whole(buffer, size),
        id,
    }
}

fn finding(record: u64, kind: FindingKind) -> Finding {
    Finding { record, kind }
}

#[test]
fn ids_and_fences_are_mapped_and_results_compared() {
    let mut b = builder();
    let buf = b.create_buffer(buffer_desc(4)).expect("buffer");
    let rb = ReadbackId::new(9, 2);
    b.submit(&list(vec![
        upload(buf, &[1, 2, 3, 4]),
        readback(buf, 4, rb),
    ]))
    .expect("list");
    b.readback_result(rb, Some(Readback::Bytes(vec![1, 2, 3, 4])))
        .expect("result");
    let bytes = b.finish().expect("end");
    let spy = Spy::new(Knob::Plain);
    let report = run(&bytes, &spy).expect("replays");
    assert_eq!(
        report,
        ReplayReport {
            records: 5,
            lists: 1,
            compared: 1,
            ..ReplayReport::default()
        }
    );
    // The builder handed out slot 1; the fresh backend its own first id, which the list names.
    let fresh = spy.inner.lists();
    let Some(Command::Upload { to, .. }) = fresh[0].1.commands.first() else {
        unreachable!("an upload")
    };
    assert_ne!(buf, BufferId::new(0, 0));
    assert_eq!(*to, whole(BufferId::new(0, 0), 4));
}

#[test]
fn map_ids_visits_and_rewrites_every_id_field() {
    const SHIFT: u32 = 1 << 30;
    let original = every_command();
    // Every id field of `every_command`, counted by hand: 16 in the transfers, 35 in the
    // drawing commands, 8 in the outputs. Readback ids are not mapped.
    let seen = RefCell::new(Vec::new());
    let mut same = original.clone();
    let identity = |id: ResourceId| {
        seen.borrow_mut().push(id);
        Some(id)
    };
    assert_eq!(map_ids(&mut same, &identity), Ok(()));
    assert_eq!(same, original);
    let seen = seen.into_inner();
    assert_eq!(seen.len(), 59);
    let distinct: HashSet<ResourceId> = seen.iter().copied().collect();
    assert_eq!(distinct.len(), 59, "every field holds a distinct id");
    // A map that does not hold one id: that id comes back, whichever field holds it.
    for &missing in &seen {
        let mut copy = original.clone();
        let refuse = |id: ResourceId| (id != missing).then_some(id);
        assert_eq!(map_ids(&mut copy, &refuse), Err(missing));
    }
    // A map to another kind is a miss too.
    let mut copy = original.clone();
    let wrong_kind = |id: ResourceId| {
        let (index, generation) = id.parts();
        Some(match id {
            ResourceId::Buffer(_) => ResourceId::Shader(ShaderId::new(index, generation)),
            other => other,
        })
    };
    assert!(matches!(
        map_ids(&mut copy, &wrong_kind),
        Err(ResourceId::Buffer(_))
    ));
    // Every field is rewritten: shifted once, every id seen the second time is shifted.
    let shift = |id: ResourceId| {
        let (index, generation) = id.parts();
        let index = index + SHIFT;
        Some(match id {
            ResourceId::Buffer(_) => ResourceId::Buffer(BufferId::new(index, generation)),
            ResourceId::Texture(_) => {
                ResourceId::Texture(crate::id::TextureId::new(index, generation))
            }
            ResourceId::View(_) => ResourceId::View(crate::id::ViewId::new(index, generation)),
            ResourceId::Sampler(_) => {
                ResourceId::Sampler(crate::id::SamplerId::new(index, generation))
            }
            ResourceId::Shader(_) => ResourceId::Shader(ShaderId::new(index, generation)),
            ResourceId::Pipeline(_) => ResourceId::Pipeline(PipelineId::new(index, generation)),
        })
    };
    let mut shifted = original.clone();
    assert_eq!(map_ids(&mut shifted, &shift), Ok(()));
    let only_shifted = |id: ResourceId| (id.parts().0 >= SHIFT).then_some(id);
    assert_eq!(map_ids(&mut shifted.clone(), &only_shifted), Ok(()));
    // Readback ids are left as they are.
    let readbacks = |l: &CommandList| -> Vec<ReadbackId> {
        l.commands
            .iter()
            .filter_map(|c| match c {
                Command::Readback { id, .. } => Some(*id),
                _ => None,
            })
            .collect()
    };
    assert_eq!(readbacks(&shifted), readbacks(&original));
}

#[test]
fn a_record_naming_an_id_not_held_is_refused_and_replay_goes_on() {
    let mut b = builder();
    let stranger = BufferId::new(77, 3);
    b.submit(&list(vec![upload(stranger, &[1])])).expect("list");
    let buf = b.create_buffer(buffer_desc(4)).expect("buffer");
    b.submit(&list(vec![upload(buf, &[2])])).expect("list");
    // A view of a texture never made, and a destroy of one.
    b.record(&Record::CreateView(super::super::CreateView {
        texture: crate::id::TextureId::new(5, 5),
        desc: view_desc(),
        result: Created::Made(crate::id::ViewId::new(6, 6)),
    }))
    .expect("view");
    b.destroy(
        ResourceId::Sampler(crate::id::SamplerId::new(8, 8)),
        FenceValue::ZERO,
    )
    .expect("destroy");
    let bytes = b.finish().expect("end");
    let spy = Spy::new(Knob::Plain);
    let report = run(&bytes, &spy).expect("replays");
    assert_eq!(report.lists, 1);
    assert_eq!(report.refused, 3);
    assert_eq!(
        report.first_refused,
        Some(finding(
            1,
            FindingKind::Unmapped(ResourceId::Buffer(stranger))
        ))
    );
    assert_eq!(report.differences, 0);
    assert_eq!(spy.inner.lists().len(), 1);
    assert_eq!(
        FindingKind::Unmapped(ResourceId::Buffer(stranger)).to_string(),
        "names buffer 77 (generation 3), which the replay does not hold"
    );
}

#[test]
fn a_destroyed_id_is_forgotten_once_its_fence_passes() {
    let mut b = builder();
    let buf = b.create_buffer(buffer_desc(4)).expect("buffer");
    let fence = b.submit(&list(vec![upload(buf, &[1])])).expect("list");
    b.destroy(ResourceId::Buffer(buf), fence).expect("destroy");
    b.submit(&list(vec![upload(buf, &[2])])).expect("list");
    let bytes = b.finish().expect("end");
    let spy = Spy::new(Knob::Plain);
    let report = run(&bytes, &spy).expect("replays");
    assert_eq!(
        spy.destroys(),
        [(ResourceId::Buffer(BufferId::new(0, 0)), fence)]
    );
    assert_eq!(
        report.first_refused,
        Some(finding(4, FindingKind::Unmapped(ResourceId::Buffer(buf))))
    );
    assert_eq!(report.lists, 1);
}

#[test]
fn a_destroy_past_every_captured_list_waits_for_one_to_reach_it() {
    let mut b = builder();
    let buf = b.create_buffer(buffer_desc(4)).expect("buffer");
    b.destroy(ResourceId::Buffer(buf), FenceValue::new(2))
        .expect("destroy ahead");
    b.submit(&list(vec![upload(buf, &[1])])).expect("fence 1");
    b.submit(&list(vec![upload(buf, &[2])])).expect("fence 2");
    b.submit(&list(vec![upload(buf, &[3])])).expect("fence 3");
    let bytes = b.finish().expect("end");
    let spy = Spy::new(Knob::Plain);
    let report = run(&bytes, &spy).expect("replays");
    // Both lists up to the fence keep the id; the destroy goes after the second, at its fence.
    assert_eq!(report.lists, 2);
    assert_eq!(
        spy.destroys(),
        [(ResourceId::Buffer(BufferId::new(0, 0)), FenceValue::new(2))]
    );
    assert_eq!(
        report.first_refused,
        Some(finding(5, FindingKind::Unmapped(ResourceId::Buffer(buf))))
    );
    // Held to the end, a destroy goes at the last fence.
    let mut b = builder();
    let buf = b.create_buffer(buffer_desc(4)).expect("buffer");
    b.submit(&list(vec![upload(buf, &[1])])).expect("fence 1");
    b.destroy(ResourceId::Buffer(buf), FenceValue::new(9))
        .expect("destroy ahead");
    let bytes = b.finish().expect("end");
    let spy = Spy::new(Knob::Plain);
    run(&bytes, &spy).expect("replays");
    assert_eq!(
        spy.destroys(),
        [(ResourceId::Buffer(BufferId::new(0, 0)), FenceValue::new(1))]
    );
}

#[test]
fn a_destroys_fence_maps_to_the_latest_list_at_or_below_it() {
    let mut b = builder();
    let ids: Vec<BufferId> = (0..5)
        .map(|_| b.create_buffer(buffer_desc(4)).expect("buffer"))
        .collect();
    let submit = |b: &mut CaptureBuilder, fence: u64, commands| {
        b.record(&Record::Submit(Submit {
            list: list(commands),
            result: Created::Made(FenceValue::new(fence)),
        }))
        .expect("list");
    };
    // Captured fences 10, 20 and 30; the one at 20 names an id replay does not hold, so the
    // fresh backend has fences 1 and 2 only, for 10 and 30.
    submit(&mut b, 10, vec![upload(ids[0], &[1])]);
    submit(&mut b, 20, vec![upload(BufferId::new(99, 0), &[1])]);
    submit(&mut b, 30, vec![upload(ids[0], &[1])]);
    for (id, after) in ids.iter().zip([9, 10, 15, 25, 30]) {
        b.destroy(ResourceId::Buffer(*id), FenceValue::new(after))
            .expect("destroy");
    }
    let bytes = b.finish().expect("end");
    let spy = Spy::new(Knob::Plain);
    run(&bytes, &spy).expect("replays");
    let mapped: Vec<u64> = spy.destroys().iter().map(|(_, f)| f.get()).collect();
    // Below the first: zero. At 10, 15, and 25 (20's list was not replayed): 1. At 30: 2.
    assert_eq!(mapped, [0, 1, 1, 1, 2]);
}

#[test]
fn the_memory_limit_refuses_a_create_past_it_and_frees_on_destroy() {
    let mut b = builder();
    let first = b.create_buffer(buffer_desc(64)).expect("buffer");
    b.create_buffer(buffer_desc(48)).expect("buffer");
    b.create_buffer(buffer_desc(1)).expect("buffer");
    b.destroy(ResourceId::Buffer(first), FenceValue::ZERO)
        .expect("destroy");
    b.create_buffer(buffer_desc(64)).expect("buffer");
    // A 2 by 2 RGBA8 texture: 16 bytes.
    b.create_texture(texture_desc()).expect("texture");
    let bytes = b.finish().expect("end");
    // 64 + 48 fit in 112 exactly; one more byte does not; after the destroy, 64 fits again,
    // and the texture does not.
    let limited = ReplayOptions {
        memory_limit: Some(112),
        ..options()
    };
    let report = run_with(
        &bytes,
        &Spy::new(Knob::Plain),
        &ReplayFrames::new(),
        &limited,
    )
    .expect("replays");
    assert_eq!(report.refused, 2);
    assert_eq!(
        report.first_refused,
        Some(finding(
            3,
            FindingKind::MemoryLimit {
                needed: 113,
                limit: 112
            }
        ))
    );
    assert_eq!(
        FindingKind::MemoryLimit {
            needed: 113,
            limit: 112
        }
        .to_string(),
        "needs 113 bytes in all, past the replay's memory limit of 112"
    );
    // With room for the one-byte buffer and the texture too, nothing is refused.
    let roomy = ReplayOptions {
        memory_limit: Some(113 + 16),
        ..options()
    };
    let report =
        run_with(&bytes, &Spy::new(Knob::Plain), &ReplayFrames::new(), &roomy).expect("replays");
    assert_eq!(report.refused, 0);
    // With no limit given, the backend's own: the spy's is 1 MiB.
    let mut b = builder();
    b.create_buffer(buffer_desc(1 << 16)).expect("buffer");
    let bytes = b.finish().expect("end");
    let report = run(&bytes, &Spy::new(Knob::Plain)).expect("replays");
    assert_eq!(report.refused, 0);
}

#[test]
fn a_call_refused_on_one_side_only_is_a_difference() {
    let mut b = builder();
    // Refused in the capture, made on replay: destroyed at once, since nothing names it.
    b.record(&Record::CreateBuffer(Create {
        desc: buffer_desc(4),
        result: Created::Refused("no room".to_owned()),
    }))
    .expect("refused");
    // Made in the capture, refused on replay: a buffer past the device's size (uncharged).
    let too_big = caps().limits.max_buffer_size + 1;
    b.create_buffer(buffer_desc(too_big)).expect("made");
    // Refused on both sides: no difference.
    b.record(&Record::CreateBuffer(Create {
        desc: buffer_desc(too_big),
        result: Created::Refused("too big".to_owned()),
    }))
    .expect("refused");
    // A list refused in the capture only.
    b.record(&Record::Submit(Submit {
        list: CommandList::new(),
        result: Created::Refused("lost".to_owned()),
    }))
    .expect("refused");
    let bytes = b.finish().expect("end");
    let spy = Spy::new(Knob::Plain);
    let report = run(&bytes, &spy).expect("replays");
    assert_eq!(report.differences, 3);
    assert_eq!(
        report.first_difference,
        Some(finding(1, FindingKind::RefusedInCapture))
    );
    assert_eq!(
        spy.destroys(),
        [(ResourceId::Buffer(BufferId::new(0, 0)), FenceValue::ZERO)]
    );
    assert_eq!(report.lists, 1);
    assert_eq!(
        FindingKind::RefusedOnReplay("buffer size 65537 is outside 1..=65536".to_owned())
            .to_string(),
        "refused on replay only: buffer size 65537 is outside 1..=65536"
    );
    // The second record's difference, alone.
    let mut b = builder();
    b.create_buffer(buffer_desc(too_big)).expect("made");
    let bytes = b.finish().expect("end");
    let report = run(&bytes, &Spy::new(Knob::Plain)).expect("replays");
    let expected = Spy::new(Knob::Plain)
        .inner
        .create_buffer(&buffer_desc(too_big))
        .expect_err("too big")
        .to_string();
    assert_eq!(
        report.first_difference,
        Some(finding(1, FindingKind::RefusedOnReplay(expected)))
    );
}

#[test]
fn shaders_are_translated_again_and_their_failures_counted() {
    let origin = |bytes: &[u8]| {
        Some(ShaderOrigin {
            record: Arc::from(bytes.to_vec()),
            stage_index: 0,
        })
    };
    let mut b = builder();
    let ok = b
        .create_shader(&shader(ShaderStage::Compute, naga(), origin(b"ok")))
        .expect("shader");
    b.create_shader(&shader(ShaderStage::Compute, naga(), origin(b"E")))
        .expect("shader");
    let untranslatable = b
        .create_shader(&shader(ShaderStage::Compute, naga(), origin(b"X")))
        .expect("shader");
    let missing = b
        .create_shader(&shader(ShaderStage::Compute, naga(), None))
        .expect("shader");
    b.create_shader(&shader(
        ShaderStage::Vertex,
        ShaderSource::SpirV(vec![7, 8]),
        None,
    ))
    .expect("shader");
    let good = b
        .create_compute_pipeline(ComputePipelineDesc { shader: ok })
        .expect("pipeline");
    b.create_compute_pipeline(ComputePipelineDesc {
        shader: untranslatable,
    })
    .expect("pipeline");
    b.create_compute_pipeline(ComputePipelineDesc { shader: missing })
        .expect("pipeline");
    b.pipeline_outcome(good, PipelineResult::Ready)
        .expect("outcome");
    let bytes = b.finish().expect("end");
    let spy = Spy::new(Knob::Plain);
    let report = run(&bytes, &spy).expect("replays");
    // Record 3 is the second source; the shaders follow each source.
    assert_eq!(report.differences, 1);
    assert_eq!(
        report.first_difference,
        Some(finding(4, FindingKind::ShaderChanged))
    );
    assert_eq!(report.refused, 4);
    assert_eq!(
        report.first_refused,
        Some(finding(
            6,
            FindingKind::Unresolved("cannot translate".to_owned())
        ))
    );
    assert_eq!(report.compared, 1);
    let snapshot = spy.inner.snapshot();
    assert!(snapshot.contains("SPIR-V 2 words"), "{snapshot}");
    assert_eq!(snapshot.matches("naga module").count(), 2, "{snapshot}");
    assert_eq!(
        FindingKind::MissingShader.to_string(),
        "the shader was captured with no origin, so it cannot be translated again"
    );
    // A resolver that cannot go on stops the replay.
    let mut b = builder();
    b.create_shader(&shader(ShaderStage::Compute, naga(), origin(b"F")))
        .expect("shader");
    let bytes = b.finish().expect("end");
    let error = run(&bytes, &Spy::new(Knob::Plain)).expect_err("fatal");
    assert_eq!(
        error,
        ReplayError::Shader {
            record: 2,
            message: "translator 2, capture 1".to_owned()
        }
    );
    assert_eq!(error.to_string(), "record 2: translator 2, capture 1");
}

#[test]
fn readbacks_are_compared_byte_for_byte_where_both_are_known() {
    let mut b = builder();
    let buf = b.create_buffer(buffer_desc(4)).expect("buffer");
    let ids = [1, 2, 3, 4, 5].map(|i| ReadbackId::new(i, 0));
    let mut commands = vec![upload(buf, &[1, 2, 3, 4])];
    commands.extend(ids[..4].iter().map(|&id| readback(buf, 4, id)));
    b.submit(&list(commands)).expect("list");
    // Nothing yet: not taken, so the same id still compares after it.
    b.readback_result(ids[0], None).expect("nothing yet");
    b.readback_result(ids[0], Some(Readback::Bytes(vec![1, 2, 3, 4])))
        .expect("same");
    b.readback_result(ids[1], Some(Readback::Bytes(vec![1, 2, 3, 5])))
        .expect("different");
    b.readback_result(ids[2], Some(Readback::Unknown))
        .expect("unknown");
    // Never read back on either side.
    b.readback_result(ids[4], Some(Readback::Bytes(vec![0])))
        .expect("missing");
    let bytes = b.finish().expect("end");
    let report = run(&bytes, &Spy::new(Knob::Plain)).expect("replays");
    assert_eq!(report.compared, 2);
    assert_eq!(report.differences, 2);
    assert_eq!(
        report.first_difference,
        Some(finding(5, FindingKind::Readback(ids[1])))
    );
    assert_eq!(
        FindingKind::ReadbackMissing(ids[4]).to_string(),
        "readback 5 (generation 0) was not there on replay"
    );
    assert_eq!(
        FindingKind::Readback(ids[1]).to_string(),
        "readback 2 (generation 0) differs"
    );
}

#[test]
fn pipeline_outcomes_are_compared_once_settled() {
    let mut b = builder();
    let s = b
        .create_shader(&shader(
            ShaderStage::Compute,
            ShaderSource::Msl(String::new()),
            None,
        ))
        .expect("shader");
    let p = b
        .create_compute_pipeline(ComputePipelineDesc { shader: s })
        .expect("pipeline");
    b.pipeline_outcome(p, PipelineResult::Ready).expect("ready");
    b.pipeline_outcome(p, PipelineResult::Failed)
        .expect("failed");
    b.pipeline_outcome(PipelineId::new(40, 0), PipelineResult::Ready)
        .expect("unknown");
    let bytes = b.finish().expect("end");
    let report = run(&bytes, &Spy::new(Knob::Plain)).expect("replays");
    assert_eq!(report.compared, 2);
    assert_eq!(report.differences, 1);
    let failed = FindingKind::Pipeline {
        captured: PipelineResult::Failed,
        replayed: PipelineStatus::Ready,
    };
    assert_eq!(report.first_difference, Some(finding(4, failed.clone())));
    assert_eq!(
        failed.to_string(),
        "the pipeline was failed in the capture and ready on replay"
    );
    assert_eq!(report.refused, 1);
    // A pipeline still compiling when the timeout passes.
    let report = run(&bytes, &Spy::new(Knob::Compiling)).expect("replays");
    assert_eq!(
        report.first_difference,
        Some(finding(
            3,
            FindingKind::Pipeline {
                captured: PipelineResult::Ready,
                replayed: PipelineStatus::Compiling
            }
        ))
    );
}

/// A capture that presents an uploaded 2 by 2 image as frame 7, then the frame results given.
fn presenting(results: &[(u64, u32, Option<[u8; 32]>)]) -> Vec<u8> {
    let mut b = builder();
    let tex = b.create_texture(texture_desc()).expect("texture");
    let view = b.create_view(tex, view_desc()).expect("view");
    b.submit(&list(vec![
        Command::Upload {
            from: StagingSlice::from_bytes(PIXELS.to_vec()),
            to: CopyRegion::Texture(TextureRegion {
                texture: tex,
                mip: 0,
                origin: Origin3d::default(),
                extent: texture_desc().size,
            }),
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
    ]))
    .expect("list");
    for &(frame, width, sha256) in results {
        b.record(&Record::FrameResult(super::super::FrameResult {
            frame,
            width,
            height: 2,
            sha256,
        }))
        .expect("frame");
    }
    b.finish().expect("end")
}

const PIXELS: [u8; 16] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6];

fn replay_frames(bytes: &[u8]) -> ReplayReport {
    let spy = Spy::new(Knob::Plain);
    let frames = Arc::new(ReplayFrames::new());
    let sink: Arc<dyn FrameSink> = Arc::<ReplayFrames>::clone(&frames);
    spy.inner.set_frame_sink(Some(sink));
    run_with(bytes, &spy, &frames, &options()).expect("replays")
}

#[test]
fn frames_are_compared_by_size_and_hash() {
    let hash = super::super::FrameResult::of(&Frame {
        frame: 7,
        width: 2,
        height: 2,
        transform: PresentTransform::Identity,
        pixels: FramePixels::Rgba8(PIXELS.to_vec()),
    })
    .sha256;
    let same = replay_frames(&presenting(&[(7, 2, hash)]));
    assert_eq!((same.compared, same.differences), (1, 0));
    let other = replay_frames(&presenting(&[(7, 2, Some([0; 32]))]));
    assert_eq!(
        other.first_difference,
        Some(finding(4, FindingKind::Frame(7)))
    );
    let wider = replay_frames(&presenting(&[(7, 3, hash)]));
    assert_eq!(
        wider.first_difference,
        Some(finding(4, FindingKind::Frame(7)))
    );
    assert_eq!(wider.compared, 0);
    // Unknown pixels in the capture: sizes compared, hashes not.
    let unknown = replay_frames(&presenting(&[(7, 2, None)]));
    assert_eq!((unknown.compared, unknown.differences), (0, 0));
    // A frame never presented on replay; and frame 7 taken once only.
    let missing = replay_frames(&presenting(&[(8, 2, hash), (7, 2, hash), (7, 2, hash)]));
    assert_eq!(
        missing.first_difference,
        Some(finding(4, FindingKind::FrameMissing(8)))
    );
    assert_eq!(missing.differences, 2);
    assert_eq!(
        FindingKind::FrameMissing(8).to_string(),
        "frame 8 was not presented on replay"
    );
}

#[test]
fn the_frame_sink_keeps_the_last_frames_up_to_its_cap() {
    let frames = ReplayFrames::new();
    let frame = |n| Frame {
        frame: n,
        width: 1,
        height: 1,
        transform: PresentTransform::Identity,
        pixels: FramePixels::Unknown,
    };
    let cap = u64::try_from(MAX_REPLAY_FRAMES).expect("small");
    for n in 0..cap {
        frames.frame(frame(n));
    }
    assert_eq!(frames.len(), MAX_REPLAY_FRAMES);
    assert!(frames.take(0).is_some(), "at the cap, the first is kept");
    frames.frame(frame(0));
    for n in cap..=cap {
        frames.frame(frame(n));
    }
    assert_eq!(frames.len(), MAX_REPLAY_FRAMES);
    // Past the cap the oldest (frame 1) went; taking a later one drops those before it.
    assert!(frames.take(1).is_none());
    assert!(frames.take(5).is_some());
    assert_eq!(frames.len(), MAX_REPLAY_FRAMES - 4);
    assert!(!frames.is_empty());
}

#[test]
fn the_report_says_where_the_capture_came_from_and_where_it_stopped() {
    let start = |backend: &str, wait, caps| Start {
        build: "b".to_owned(),
        backend: backend.to_owned(),
        caps,
        wait,
        memory_limit: 1,
    };
    let finish = |s: Start, limit: bool| {
        let mut b = CaptureBuilder::with_start(s).expect("start");
        if limit {
            b.record(&Record::Limit(Limit {
                fence: FenceValue::new(3),
            }))
            .expect("limit");
        }
        b.finish().expect("end")
    };
    let same = run(
        &finish(start("builder", WaitSetting::Wait, caps()), false),
        &Spy::new(Knob::Plain),
    )
    .expect("replays");
    assert!(!same.other_backend && !same.captured_skipping);
    assert_eq!((same.limit, same.records), (None, 2));
    let renamed = run(
        &finish(start("wgpu", WaitSetting::Skip, caps()), true),
        &Spy::new(Knob::Plain),
    )
    .expect("replays");
    assert!(renamed.other_backend && renamed.captured_skipping);
    assert_eq!(renamed.limit, Some(FenceValue::new(3)));
    let mut other_caps = caps();
    other_caps.view_swizzle = !other_caps.view_swizzle;
    let other_device = run(
        &finish(start("builder", WaitSetting::Wait, other_caps), false),
        &Spy::new(Knob::Plain),
    )
    .expect("replays");
    assert!(other_device.other_backend);
}

#[test]
fn a_window_target_is_replayed_offscreen() {
    let mut b = builder();
    b.record(&Record::PresentTarget(TargetRecord {
        target: CapturedTarget::Window,
        result: Created::Made(()),
    }))
    .expect("target");
    b.set_present_target(crate::backend::PresentTarget::Offscreen)
        .expect("target");
    let bytes = b.finish().expect("end");
    let spy = Spy::new(Knob::Plain);
    let report = run(&bytes, &spy).expect("replays");
    assert_eq!((report.differences, report.refused), (0, 0));
    assert_eq!(
        spy.inner
            .snapshot()
            .matches("present target offscreen")
            .count(),
        2
    );
}

#[test]
fn a_lost_device_a_fence_never_reached_and_a_malformed_capture_stop_the_replay() {
    let mut b = builder();
    b.create_buffer(buffer_desc(4)).expect("buffer");
    let bytes = b.finish().expect("end");
    let lost = run(&bytes, &Spy::new(Knob::Lost));
    assert_eq!(lost, Err(ReplayError::DeviceLost { record: 1 }));
    assert_eq!(
        ReplayError::DeviceLost { record: 1 }.to_string(),
        "record 1: the host device was lost"
    );

    let mut b = builder();
    b.submit(&CommandList::new()).expect("list");
    b.readback_result(ReadbackId::new(1, 0), Some(Readback::Unknown))
        .expect("result");
    let bytes = b.finish().expect("end");
    let timed_out = run(&bytes, &Spy::new(Knob::TimeOut));
    assert_eq!(
        timed_out,
        Err(ReplayError::Timeout {
            record: 2,
            fence: 1
        })
    );
    assert_eq!(
        ReplayError::Timeout {
            record: 2,
            fence: 1
        }
        .to_string(),
        "record 2: the backend did not reach fence value 1 in time"
    );

    // Cut short after its first list: that list was replayed before the error.
    let cut = &bytes[..bytes.len() - 1];
    let spy = Spy::new(Knob::Plain);
    assert_eq!(
        run(cut, &spy),
        Err(ReplayError::Capture(CaptureError::Truncated))
    );
    assert_eq!(spy.inner.lists().len(), 1);
}
