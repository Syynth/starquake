//! Captures: the lists of a run on the recording backend, written to a file's bytes, read
//! back and replayed on a second recording backend, give the results of the original run.

use super::fixtures::{
    backend, buffer, clear, counting, draw, graphics, ok, rb, rect, region, slice, submit, texture,
    upload, view,
};
use starquake_render::command::{
    ClearValue, CopyDesc, Marker, PresentDesc, PresentTransform, RenderTargets,
};
use starquake_render::{
    Backend, BufferId, BufferUsage, Capture, CaptureError, CapturedList, Command, CommandList,
    FenceValue, FrameDump, PipelineId, PixelFormat, PresentFeedback, Readback, RecordingBackend,
    RenderError, TextureId, ViewId,
};

const RGBA: PixelFormat = PixelFormat::A8B8G8R8Unorm;

/// The resources of the run: the same calls in the same order give the same ids on any fresh
/// recording backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Scene {
    data: BufferId,
    source: TextureId,
    target: TextureId,
    source_view: ViewId,
    target_view: ViewId,
    pipeline: PipelineId,
}

fn data_buffer(b: &RecordingBackend) -> BufferId {
    buffer(b, 16, BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST))
}

fn scene(b: &RecordingBackend) -> Scene {
    let data = data_buffer(b);
    let source = texture(b, RGBA, (2, 2), 1, 1);
    let target = texture(b, RGBA, (2, 2), 1, 1);
    Scene {
        data,
        source,
        target,
        source_view: view(b, source, RGBA, 0, (0, 1)),
        target_view: view(b, target, RGBA, 0, (0, 1)),
        pipeline: graphics(b),
    }
}

fn whole(texture: TextureId) -> starquake_render::command::CopyRegion {
    region(texture, 0, (0, 0, 0), (2, 2, 1))
}

fn present(view: ViewId, frame: u64) -> Command {
    Command::Present(PresentDesc {
        view,
        src_rect: rect(0, 0, 2, 2),
        transform: PresentTransform::Rotate90,
        frame,
        dump: true,
    })
}

/// The lists of the run: uploads, a copy, a clear, a draw, presents and readbacks, across
/// four lists of which one is empty.
fn lists(s: &Scene) -> Vec<CommandList> {
    let mut targets = RenderTargets::default();
    targets.colors[0] = Some(s.target_view);
    let list = |commands| CommandList { commands };
    vec![
        list(vec![
            Command::Marker(Marker { submit: 1, draw: 0 }),
            upload(&counting(1, 8), slice(s.data, 4, 8)),
            upload(&counting(100, 16), whole(s.source)),
            Command::Readback {
                from: slice(s.data, 4, 8),
                id: rb(1),
            },
        ]),
        list(vec![
            Command::Copy(CopyDesc {
                src: whole(s.source),
                dst: whole(s.target),
            }),
            clear(
                s.target_view,
                ClearValue::Float([0.0, 0.0, 0.0, 1.0]),
                Some(rect(1, 0, 1, 1)),
            ),
            Command::Readback {
                from: whole(s.target),
                id: rb(2),
            },
            present(s.target_view, 1),
        ]),
        list(Vec::new()),
        list(vec![
            Command::SetRenderTargets(targets),
            Command::Draw(draw(s.pipeline)),
            Command::Readback {
                from: whole(s.target),
                id: rb(3),
            },
            Command::Readback {
                from: whole(s.source),
                id: rb(4),
            },
            present(s.target_view, 2),
            present(s.source_view, 3),
        ]),
    ]
}

/// Everything a run leaves behind that a test can ask a recording backend for.
#[derive(Debug, PartialEq)]
struct Results {
    completed: FenceValue,
    /// Readbacks 1 to 5; nothing asks for the fifth.
    readbacks: Vec<Option<Readback>>,
    frames: Vec<FrameDump>,
    feedback: Vec<PresentFeedback>,
    snapshot: String,
}

fn results(b: &RecordingBackend) -> Results {
    Results {
        completed: b.completed(),
        readbacks: (1..=5).map(|n| b.take_readback(rb(n))).collect(),
        frames: b.take_frame_dumps(),
        feedback: b.take_present_feedback(),
        snapshot: b.snapshot(),
    }
}

fn fences(values: &[u64]) -> Vec<FenceValue> {
    values.iter().copied().map(FenceValue::new).collect()
}

/// The issue's last test: the original run, its capture as a file's bytes, and the replay of
/// what was read back from those bytes give the same readbacks (and frames, feedback, fence
/// values and snapshot).
#[test]
fn a_replayed_capture_gives_the_results_of_the_original_run() {
    // The original run.
    let original = backend();
    let s = scene(&original);
    let signalled: Vec<FenceValue> = lists(&s)
        .into_iter()
        .map(|list| ok(original.submit(list)))
        .collect();
    assert_eq!(signalled, fences(&[1, 2, 3, 4]));

    // Its capture, through a file's bytes.
    let capture = original.capture();
    let expected: Vec<CapturedList> = signalled
        .iter()
        .zip(lists(&s))
        .map(|(fence, list)| CapturedList {
            fence: *fence,
            list,
        })
        .collect();
    assert_eq!(capture.lists, expected);
    let file = ok(capture.to_bytes());
    let read = ok(Capture::from_bytes(&file));
    assert_eq!(read, capture);
    // The staging bytes are in the file as they were uploaded.
    assert!(
        file.windows(16).any(|w| w == counting(100, 16)),
        "the image is in the file"
    );

    // The replay, on a second backend with the same resources.
    let second = backend();
    assert_eq!(scene(&second), s);
    assert_eq!(read.replay(&second), Ok(signalled));

    let first = results(&original);
    // What the original run gave, so that "the same" is not two empty answers. The target is
    // the image with its second texel cleared, until the draw makes it unknown.
    let mut cleared = counting(100, 16);
    cleared[4..8].copy_from_slice(&[0, 0, 0, 255]);
    assert_eq!(
        first.readbacks,
        vec![
            Some(Readback::Bytes(counting(1, 8))),
            Some(Readback::Bytes(cleared.clone())),
            Some(Readback::Unknown),
            Some(Readback::Bytes(counting(100, 16))),
            None,
        ]
    );
    let pixels: Vec<(u64, Readback)> = first
        .frames
        .iter()
        .map(|f| (f.frame, f.pixels.clone()))
        .collect();
    assert_eq!(
        pixels,
        vec![
            (1, Readback::Bytes(cleared)),
            (2, Readback::Unknown),
            (3, Readback::Bytes(counting(100, 16))),
        ]
    );
    assert_eq!(first.feedback.len(), 3);
    assert_eq!(first.completed, FenceValue::new(4));

    assert_eq!(results(&second), first);
}

/// A replay needs the resources the lists name. On a backend without them the list that
/// names a missing one is refused, the error says which list, and the lists before it stay
/// submitted.
#[test]
fn a_replay_stops_at_the_list_a_backend_refuses() {
    let original = backend();
    let s = scene(&original);
    for list in lists(&s) {
        submit(&original, list.commands);
    }
    let capture = original.capture();

    // Nothing exists: the first list is refused, and nothing was submitted.
    let empty = backend();
    let refused = capture.replay(&empty);
    assert_eq!(
        refused,
        Err(CaptureError::Replay {
            list: 0,
            source: RenderError::Invalid("no live buffer has this id"),
        })
    );
    assert_eq!(empty.completed(), FenceValue::ZERO);
    assert_eq!(empty.capture(), Capture::default());

    // Only the buffer and the first texture exist: the first list runs, the second is refused.
    let partial = backend();
    assert_eq!(data_buffer(&partial), s.data);
    assert_eq!(texture(&partial, RGBA, (2, 2), 1, 1), s.source);
    let refused = capture.replay(&partial).expect_err("the target is missing");
    assert_eq!(
        refused,
        CaptureError::Replay {
            list: 1,
            source: RenderError::Invalid("no live texture has this id"),
        }
    );
    assert_eq!(
        refused.to_string(),
        "list 1 of the capture was refused: no live texture has this id"
    );
    let source = std::error::Error::source(&refused).expect("the backend's error");
    assert_eq!(source.to_string(), "no live texture has this id");
    assert_eq!(partial.completed(), FenceValue::new(1));
    assert_eq!(
        partial.take_readback(rb(1)),
        Some(Readback::Bytes(counting(1, 8)))
    );
    assert_eq!(partial.take_readback(rb(2)), None);
    assert_eq!(partial.capture().lists, capture.lists[..1]);
}

/// The fence values a replay returns are the ones the backend signals, not the captured ones:
/// a capture can start late in its run and be replayed on a backend that has done other work.
#[test]
fn a_replay_returns_the_fence_values_of_its_backend() {
    let marker = |n| CommandList {
        commands: vec![Command::Marker(Marker { submit: n, draw: n })],
    };
    let capture = Capture::new([
        (FenceValue::new(40), marker(1)),
        (FenceValue::new(41), CommandList::new()),
        (FenceValue::new(90), marker(2)),
    ]);
    let b = backend();
    assert_eq!(capture.replay(&b), Ok(fences(&[1, 2, 3])));
    // Again, on the same backend: the lists are submitted again.
    assert_eq!(capture.replay(&b), Ok(fences(&[4, 5, 6])));
    let submitted: Vec<CommandList> = b.lists().into_iter().map(|(_, list)| list).collect();
    assert_eq!(
        submitted,
        vec![
            marker(1),
            CommandList::new(),
            marker(2),
            marker(1),
            CommandList::new(),
            marker(2),
        ]
    );
    // The capture itself is unchanged.
    assert_eq!(capture.lists[2].fence, FenceValue::new(90));

    // An empty capture submits nothing.
    let untouched = backend();
    assert_eq!(Capture::default().replay(&untouched), Ok(Vec::new()));
    assert_eq!(untouched.completed(), FenceValue::ZERO);
}

/// The recording backend's capture is the lists it accepted, with their fence values: empty
/// lists are in it, refused lists are not.
#[test]
fn the_recording_backends_capture() {
    let b = backend();
    assert_eq!(b.capture(), Capture::default());
    let marker = CommandList {
        commands: vec![Command::Marker(Marker { submit: 7, draw: 9 })],
    };
    submit(&b, marker.commands.clone());
    submit(&b, Vec::new());
    let refused = b.submit(CommandList {
        commands: vec![Command::Readback {
            from: slice(BufferId::new(9, 9), 0, 4),
            id: rb(1),
        }],
    });
    assert_eq!(
        refused,
        Err(RenderError::Invalid("no live buffer has this id"))
    );
    submit(&b, marker.commands.clone());
    assert_eq!(
        b.capture(),
        Capture::new([
            (FenceValue::new(1), marker.clone()),
            (FenceValue::new(2), CommandList::new()),
            (FenceValue::new(3), marker),
        ])
    );
    assert_eq!(b.capture().lists.len(), b.lists().len());
}
