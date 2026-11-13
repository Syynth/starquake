//! Frame dumps: presents with `dump` set reach the installed `FrameSink` as RGBA8 frames, in
//! order, and a frame of an uploaded image is exactly that image as a PNG.

use super::fixtures::{
    MEMORY, backend, caps, counting, draw, graphics, ok, read, rect, region, submit, texture,
    upload, view,
};
use starquake_render::command::{PresentDesc, PresentTransform, Rect, RenderTargets};
use starquake_render::frame::RGBA8_BYTES;
use starquake_render::resource::{Swizzle, SwizzleSource, ViewDimension};
use starquake_render::{
    Backend, Caps, Command, CommandList, FenceValue, Frame, FrameDump, FramePixels, FrameSink,
    PixelFormat, PresentFeedback, PresentOutcome, Readback, ReadbackId, RecordingBackend,
    RenderError, ViewDesc, ViewId,
};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::Duration;

const RGBA: PixelFormat = PixelFormat::A8B8G8R8Unorm;

/// A sink that keeps what it is given.
#[derive(Debug, Default)]
struct Collect(Mutex<Vec<Frame>>);

impl Collect {
    fn take(&self) -> Vec<Frame> {
        std::mem::take(&mut self.0.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// The frame numbers taken.
    fn numbers(&self) -> Vec<u64> {
        self.take().iter().map(|f| f.frame).collect()
    }
}

impl FrameSink for Collect {
    fn frame(&self, frame: Frame) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(frame);
    }
}

/// Installs `sink` on `b`; the test keeps its own handle.
fn install<S: FrameSink + 'static>(b: &RecordingBackend, sink: &Arc<S>) {
    let sink: Arc<dyn FrameSink> = Arc::<S>::clone(sink);
    b.set_frame_sink(Some(sink));
}

/// Installs a collecting sink on `b`.
fn collecting(b: &RecordingBackend) -> Arc<Collect> {
    let sink = Arc::new(Collect::default());
    install(b, &sink);
    sink
}

fn present_with(view: ViewId, src_rect: Rect, frame: u64, dump: bool) -> Command {
    Command::Present(PresentDesc {
        view,
        src_rect,
        transform: PresentTransform::Identity,
        frame,
        dump,
    })
}

/// A dumped present of `src_rect`.
fn present(view: ViewId, src_rect: Rect, frame: u64) -> Command {
    present_with(view, src_rect, frame, true)
}

fn frame(frame: u64, (width, height): (u32, u32), pixels: FramePixels) -> Frame {
    Frame {
        frame,
        width,
        height,
        transform: PresentTransform::Identity,
        pixels,
    }
}

fn rgba(bytes: &[u8]) -> FramePixels {
    FramePixels::Rgba8(bytes.to_vec())
}

/// A PNG decoded with the `png` crate's decoder: (width, height, RGBA8 bytes).
fn decode(file: &[u8]) -> (u32, u32, Vec<u8>) {
    let decoder = png::Decoder::new(std::io::Cursor::new(file));
    let mut reader = ok(decoder.read_info());
    let mut pixels = vec![0; ok(reader.output_buffer_size().ok_or("the size does not fit"))];
    let info = ok(reader.next_frame(&mut pixels));
    assert_eq!(info.color_type, png::ColorType::Rgba);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    pixels.truncate(info.buffer_size());
    (info.width, info.height, pixels)
}

/// A one-layer, one-mip texture of `format` and a view of all of it, holding `image`.
fn uploaded(
    b: &RecordingBackend,
    format: PixelFormat,
    (width, height): (u32, u32),
    image: &[u8],
) -> ViewId {
    let t = texture(b, format, (width, height), 1, 1);
    submit(
        b,
        vec![upload(image, region(t, 0, (0, 0, 0), (width, height, 1)))],
    );
    view(b, t, format, 0, (0, 1))
}

/// The issue's first test: an uploaded image, presented, is that image as a PNG.
#[test]
fn a_present_of_an_uploaded_image_is_that_image_as_a_png() {
    let b = backend();
    let sink = collecting(&b);
    let image = counting(10, 4 * 3 * RGBA8_BYTES);
    let v = uploaded(&b, RGBA, (4, 3), &image);
    submit(
        &b,
        vec![Command::Present(PresentDesc {
            view: v,
            src_rect: rect(0, 0, 4, 3),
            transform: PresentTransform::FlipVertical,
            frame: 5,
            dump: true,
        })],
    );
    let frames = sink.take();
    // The transform comes with the frame and is not applied to it.
    assert_eq!(
        frames,
        vec![Frame {
            frame: 5,
            width: 4,
            height: 3,
            transform: PresentTransform::FlipVertical,
            pixels: rgba(&image),
        }]
    );
    let file = frames[0].png().expect("a valid frame").expect("known");
    assert_eq!(decode(&file), (4, 3, image));
    // The sink took the frame: it does not also wait to be taken.
    assert_eq!(b.take_frame_dumps(), Vec::new());
}

/// The frame is the presented rectangle of the view's mip level and layer, not the texture.
#[test]
fn a_frame_is_the_presented_rectangle_of_the_view() {
    let b = backend();
    let sink = collecting(&b);
    let t = texture(&b, RGBA, (4, 4), 2, 2);
    // Every level and layer holds different bytes; the view shows mip 1 of layer 1 (2 by 2).
    let shown = counting(100, 16);
    submit(
        &b,
        vec![
            upload(&counting(0, 64), region(t, 0, (0, 0, 0), (4, 4, 1))),
            upload(&counting(64, 64), region(t, 0, (0, 0, 1), (4, 4, 1))),
            upload(&counting(200, 16), region(t, 1, (0, 0, 0), (2, 2, 1))),
            upload(&shown, region(t, 1, (0, 0, 1), (2, 2, 1))),
        ],
    );
    let v = view(&b, t, RGBA, 1, (1, 1));
    submit(
        &b,
        vec![
            present(v, rect(0, 0, 2, 2), 1),
            present(v, rect(1, 0, 1, 2), 2),
            present(v, rect(0, 1, 2, 1), 3),
        ],
    );
    let frames = sink.take();
    assert_eq!(
        frames,
        vec![
            frame(1, (2, 2), rgba(&shown)),
            // The right column: the second texel of each row.
            frame(2, (1, 2), rgba(&[104, 105, 106, 107, 112, 113, 114, 115])),
            // The bottom row.
            frame(3, (2, 1), rgba(&shown[8..])),
        ]
    );
    let file = frames[1].png().expect("a valid frame").expect("known");
    assert_eq!(
        decode(&file),
        (1, 2, vec![104, 105, 106, 107, 112, 113, 114, 115])
    );
}

/// The four formats with an exact RGBA8 form, by the view's format.
#[test]
fn frames_of_the_presentable_8_bit_formats() {
    let texels = [1, 2, 3, 4, 250, 251, 252, 253];
    let swapped = [3, 2, 1, 4, 252, 251, 250, 253];
    for (format, expected) in [
        (PixelFormat::A8B8G8R8Unorm, texels),
        (PixelFormat::A8B8G8R8Srgb, texels),
        (PixelFormat::B8G8R8A8Unorm, swapped),
        (PixelFormat::B8G8R8A8Srgb, swapped),
    ] {
        let b = backend();
        let sink = collecting(&b);
        let v = uploaded(&b, format, (2, 1), &texels);
        submit(&b, vec![present(v, rect(0, 0, 2, 1), 1)]);
        let frames = sink.take();
        assert_eq!(frames, vec![frame(1, (2, 1), rgba(&expected))], "{format}");
        let file = frames[0].png().expect("a valid frame").expect("known");
        assert_eq!(decode(&file), (2, 1, expected.to_vec()), "{format}");
    }
}

/// Every other presentable format has no exact RGBA8 form: its frames are unknown although
/// every byte of the texture is known.
#[test]
fn frames_of_other_formats_are_unknown() {
    let mut checked = 0;
    for format in PixelFormat::ALL {
        let exact = matches!(
            format,
            PixelFormat::A8B8G8R8Unorm
                | PixelFormat::A8B8G8R8Srgb
                | PixelFormat::B8G8R8A8Unorm
                | PixelFormat::B8G8R8A8Srgb
        );
        // What a present accepts and the recording backend holds texels of.
        let presentable = format.aspect() == starquake_render::format::FormatAspect::Color
            && !format.is_integer()
            && !format.is_compressed();
        if exact || !presentable {
            continue;
        }
        let b = backend();
        let sink = collecting(&b);
        let texel = counting(1, format.bytes_per_block() as usize);
        let t = texture(&b, format, (1, 1), 1, 1);
        let all = region(t, 0, (0, 0, 0), (1, 1, 1));
        submit(&b, vec![upload(&texel, all)]);
        let v = view(&b, t, format, 0, (0, 1));
        submit(&b, vec![present(v, rect(0, 0, 1, 1), 1)]);
        assert_eq!(read(&b, all), Readback::Bytes(texel), "{format}");
        let frames = sink.take();
        assert_eq!(
            frames,
            vec![frame(1, (1, 1), FramePixels::Unknown)],
            "{format}"
        );
        assert_eq!(frames[0].png(), Ok(None), "{format}");
        checked += 1;
    }
    assert!(checked > 20, "{checked}");
}

/// A frame is known only when every byte of the presented rectangle is: the rectangle's edge
/// is tested one texel either side of a texel nothing wrote.
#[test]
fn a_frame_with_any_unknown_texel_is_unknown() {
    let b = backend();
    let sink = collecting(&b);
    let t = texture(&b, RGBA, (3, 1), 1, 1);
    let v = view(&b, t, RGBA, 0, (0, 1));
    submit(
        &b,
        vec![
            // Before anything was uploaded.
            present(v, rect(0, 0, 3, 1), 1),
            // The first two texels are uploaded; the third never is.
            upload(&counting(1, 8), region(t, 0, (0, 0, 0), (2, 1, 1))),
            present(v, rect(0, 0, 2, 1), 2),
            present(v, rect(0, 0, 3, 1), 3),
            present(v, rect(1, 0, 2, 1), 4),
            present(v, rect(1, 0, 1, 1), 5),
            present(v, rect(2, 0, 1, 1), 6),
        ],
    );
    let frames = sink.take();
    assert_eq!(
        frames,
        vec![
            frame(1, (3, 1), FramePixels::Unknown),
            frame(2, (2, 1), rgba(&counting(1, 8))),
            frame(3, (3, 1), FramePixels::Unknown),
            frame(4, (2, 1), FramePixels::Unknown),
            frame(5, (1, 1), rgba(&counting(5, 4))),
            frame(6, (1, 1), FramePixels::Unknown),
        ]
    );
    // An unknown frame has no PNG; it is not an error.
    assert_eq!(frames[0].png(), Ok(None));
}

/// A draw makes its targets unknown, so a frame the guest drew has no pixels here.
#[test]
fn a_drawn_frame_is_unknown() {
    let b = backend();
    let sink = collecting(&b);
    let pipeline = graphics(&b);
    let image = counting(0, 4 * 4 * RGBA8_BYTES);
    let v = uploaded(&b, RGBA, (4, 4), &image);
    let mut targets = RenderTargets::default();
    targets.colors[0] = Some(v);
    submit(
        &b,
        vec![
            present(v, rect(0, 0, 4, 4), 1),
            Command::SetRenderTargets(targets),
            Command::Draw(draw(pipeline)),
            present(v, rect(0, 0, 4, 4), 2),
        ],
    );
    assert_eq!(
        sink.take(),
        vec![
            frame(1, (4, 4), rgba(&image)),
            frame(2, (4, 4), FramePixels::Unknown),
        ]
    );
}

/// The present blit samples through the view's swizzle and decodes compressed blocks; the
/// recording backend models neither, so those frames are unknown.
#[test]
fn swizzled_and_compressed_frames_are_unknown() {
    let b = RecordingBackend::new(
        Caps {
            view_swizzle: true,
            ..caps()
        },
        MEMORY,
    );
    let sink = collecting(&b);
    let t = texture(&b, RGBA, (1, 1), 1, 1);
    let with = |swizzle| {
        ok(b.create_view(
            t,
            &ViewDesc {
                format: RGBA,
                dimension: ViewDimension::D2Array,
                base_mip: 0,
                mip_count: 1,
                base_layer: 0,
                layer_count: 1,
                swizzle,
            },
        ))
    };
    let swap = Swizzle {
        r: SwizzleSource::B,
        g: SwizzleSource::G,
        b: SwizzleSource::R,
        a: SwizzleSource::A,
    };
    let bc1 = PixelFormat::Bc1RgbaUnorm;
    let compressed = uploaded(&b, bc1, (4, 4), &[1; 8]);
    submit(
        &b,
        vec![
            upload(&[1, 2, 3, 4], region(t, 0, (0, 0, 0), (1, 1, 1))),
            present(with(Swizzle::IDENTITY), rect(0, 0, 1, 1), 1),
            present(with(swap), rect(0, 0, 1, 1), 2),
            present(compressed, rect(0, 0, 4, 4), 3),
        ],
    );
    assert_eq!(
        sink.take(),
        vec![
            frame(1, (1, 1), rgba(&[1, 2, 3, 4])),
            frame(2, (1, 1), FramePixels::Unknown),
            frame(3, (4, 4), FramePixels::Unknown),
        ]
    );
}

/// Only presents with `dump` set are read back; every present still gives feedback.
#[test]
fn only_dumped_presents_reach_the_sink() {
    let b = backend();
    let sink = collecting(&b);
    let v = uploaded(&b, RGBA, (1, 1), &[9, 8, 7, 6]);
    let one = rect(0, 0, 1, 1);
    submit(
        &b,
        vec![
            present_with(v, one, 1, false),
            present_with(v, one, 2, true),
            present_with(v, one, 3, false),
        ],
    );
    assert_eq!(sink.take(), vec![frame(2, (1, 1), rgba(&[9, 8, 7, 6]))]);
    let shown = |frame| PresentFeedback {
        frame,
        outcome: PresentOutcome::Shown,
        host_time: Duration::ZERO,
    };
    assert_eq!(
        b.take_present_feedback(),
        vec![shown(1), shown(2), shown(3)]
    );
    assert_eq!(b.take_frame_dumps(), Vec::new());
}

/// Frames arrive in the order they were presented, within a list and across lists, and a
/// list with no dumped present delivers nothing.
#[test]
fn frames_arrive_in_present_order() {
    let b = backend();
    let sink = collecting(&b);
    let v = uploaded(&b, RGBA, (1, 1), &[1, 1, 1, 1]);
    let one = rect(0, 0, 1, 1);
    assert_eq!(
        sink.numbers(),
        Vec::<u64>::new(),
        "an upload is not a frame"
    );
    submit(&b, vec![present(v, one, 4), present(v, one, 3)]);
    assert_eq!(sink.numbers(), vec![4, 3], "delivered when submit returns");
    submit(&b, vec![present(v, one, 9)]);
    submit(&b, Vec::new());
    submit(&b, vec![present(v, one, 2), present(v, one, 2)]);
    assert_eq!(sink.numbers(), vec![9, 2, 2]);
}

/// With no sink the dumps wait in the backend, as before; a frame goes to the sink that was
/// installed when its list was submitted.
#[test]
fn frames_go_to_the_sink_installed_at_submit() {
    let b = backend();
    let v = uploaded(&b, RGBA, (1, 1), &[5, 6, 7, 8]);
    let one = rect(0, 0, 1, 1);
    let dump = |frame| FrameDump {
        frame,
        view: v,
        format: RGBA,
        width: 1,
        height: 1,
        transform: PresentTransform::Identity,
        pixels: Readback::Bytes(vec![5, 6, 7, 8]),
    };

    // No sink: the dump waits to be taken.
    submit(&b, vec![present(v, one, 1)]);
    // A sink installed later is not given it.
    let first = collecting(&b);
    assert_eq!(first.take(), Vec::new());
    submit(&b, vec![present(v, one, 2)]);
    assert_eq!(first.numbers(), vec![2]);
    assert_eq!(b.take_frame_dumps(), vec![dump(1)]);

    // A second sink replaces the first.
    let second = collecting(&b);
    submit(&b, vec![present(v, one, 3)]);
    assert_eq!(first.take(), Vec::new());
    assert_eq!(second.take(), vec![frame(3, (1, 1), rgba(&[5, 6, 7, 8]))]);

    // With the sink removed, dumps wait again.
    b.set_frame_sink(None);
    submit(&b, vec![present(v, one, 4)]);
    assert_eq!(second.take(), Vec::new());
    assert_eq!(b.take_frame_dumps(), vec![dump(4)]);

    // A dump that was taken becomes the frame a sink would have been given.
    assert_eq!(Frame::from(dump(4)), frame(4, (1, 1), rgba(&[5, 6, 7, 8])));
    let unknown = FrameDump {
        pixels: Readback::Unknown,
        transform: PresentTransform::Rotate270,
        ..dump(6)
    };
    assert_eq!(
        Frame::from(unknown),
        Frame {
            frame: 6,
            width: 1,
            height: 1,
            transform: PresentTransform::Rotate270,
            pixels: FramePixels::Unknown,
        }
    );
}

/// A list that is refused changes nothing, so it delivers nothing, even when a dumped present
/// comes before the command that is refused.
#[test]
fn a_refused_list_delivers_no_frame() {
    let b = backend();
    let sink = collecting(&b);
    let v = uploaded(&b, RGBA, (1, 1), &[1, 2, 3, 4]);
    let one = rect(0, 0, 1, 1);
    let refused = b.submit(CommandList {
        commands: vec![present(v, one, 1), present(v, rect(0, 0, 0, 1), 2)],
    });
    assert_eq!(
        refused,
        Err(RenderError::Invalid("a rectangle must not be empty"))
    );
    assert_eq!(sink.take(), Vec::new());
    assert_eq!(b.take_frame_dumps(), Vec::new());
    submit(&b, vec![present(v, one, 3)]);
    assert_eq!(sink.numbers(), vec![3]);
}

/// A sink that asks the backend questions from inside `frame`.
#[derive(Debug)]
struct Asking {
    backend: Weak<RecordingBackend>,
    readback: ReadbackId,
    /// For each frame: its number, the fence value completed, and the list's readback.
    seen: Mutex<Vec<(u64, FenceValue, Option<Readback>)>>,
}

impl FrameSink for Asking {
    fn frame(&self, frame: Frame) {
        let Some(b) = self.backend.upgrade() else {
            return;
        };
        let answer = (frame.frame, b.completed(), b.take_readback(self.readback));
        self.seen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(answer);
    }
}

/// The sink is called after the list has completed and with no lock held: it can call the
/// backend, sees the list's fence value as completed and can take the list's readbacks.
#[test]
fn the_sink_is_called_after_the_list_completed() {
    let b = Arc::new(backend());
    let t = texture(&b, RGBA, (1, 1), 1, 1);
    let v = view(&b, t, RGBA, 0, (0, 1));
    let all = region(t, 0, (0, 0, 0), (1, 1, 1));
    let readback = ReadbackId::new(3, 0);
    let sink = Arc::new(Asking {
        backend: Arc::downgrade(&b),
        readback,
        seen: Mutex::default(),
    });
    install(&b, &sink);
    submit(&b, Vec::new());
    let fence = submit(
        &b,
        vec![
            upload(&[1, 2, 3, 4], all),
            present(v, rect(0, 0, 1, 1), 7),
            // After the present in the list, and still done when the sink runs.
            Command::Readback {
                from: all,
                id: readback,
            },
        ],
    );
    assert_eq!(fence, FenceValue::new(2));
    let seen = std::mem::take(&mut *sink.seen.lock().expect("not poisoned"));
    assert_eq!(
        seen,
        vec![(7, fence, Some(Readback::Bytes(vec![1, 2, 3, 4])))]
    );
}

/// A sink that submits a list of its own when it is given frame 1.
#[derive(Debug)]
struct Resubmitting {
    backend: Weak<RecordingBackend>,
    view: ViewId,
    seen: Mutex<Vec<(u64, Option<FenceValue>)>>,
}

impl FrameSink for Resubmitting {
    fn frame(&self, frame: Frame) {
        let inner = (frame.frame == 1)
            .then(|| self.backend.upgrade())
            .flatten()
            .map(|b| submit(&b, vec![present(self.view, rect(0, 0, 1, 1), 10)]));
        self.seen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((frame.frame, inner));
    }
}

/// A sink may submit from inside `frame`. The frames of that list come after the frames
/// already presented (the rest of the outer list), not in the middle of them.
#[test]
fn a_sink_may_submit_and_order_is_kept() {
    let b = Arc::new(backend());
    let v = uploaded(&b, RGBA, (1, 1), &[1, 2, 3, 4]);
    let sink = Arc::new(Resubmitting {
        backend: Arc::downgrade(&b),
        view: v,
        seen: Mutex::default(),
    });
    install(&b, &sink);
    let one = rect(0, 0, 1, 1);
    let outer = submit(&b, vec![present(v, one, 1), present(v, one, 2)]);
    assert_eq!(outer, FenceValue::new(2));
    let seen = std::mem::take(&mut *sink.seen.lock().expect("not poisoned"));
    assert_eq!(
        seen,
        vec![(1, Some(FenceValue::new(3))), (2, None), (10, None)]
    );
    assert_eq!(b.completed(), FenceValue::new(3));
    // Nothing is left behind: the next list delivers only its own frame.
    submit(&b, vec![present(v, one, 4)]);
    let seen = std::mem::take(&mut *sink.seen.lock().expect("not poisoned"));
    assert_eq!(seen, vec![(4, None)]);
}

/// A sink that installs another sink, or removes itself, when it is given frame 1.
#[derive(Debug)]
struct Replacing {
    backend: Weak<RecordingBackend>,
    next: Option<Arc<Collect>>,
    seen: Collect,
}

impl FrameSink for Replacing {
    fn frame(&self, frame: Frame) {
        if frame.frame == 1
            && let Some(b) = self.backend.upgrade()
        {
            match &self.next {
                Some(next) => install(&b, next),
                None => b.set_frame_sink(None),
            }
        }
        self.seen.frame(frame);
    }
}

/// A frame goes to the sink that was installed when its list was submitted, also when the
/// sink is replaced or removed before the frame is delivered.
#[test]
fn a_waiting_frame_keeps_the_sink_of_its_list() {
    let one = rect(0, 0, 1, 1);

    // Replaced: frame 2 is waiting behind frame 1 when the sink changes.
    let b = Arc::new(backend());
    let v = uploaded(&b, RGBA, (1, 1), &[1, 2, 3, 4]);
    let next = Arc::new(Collect::default());
    let sink = Arc::new(Replacing {
        backend: Arc::downgrade(&b),
        next: Some(Arc::clone(&next)),
        seen: Collect::default(),
    });
    install(&b, &sink);
    submit(&b, vec![present(v, one, 1), present(v, one, 2)]);
    assert_eq!(sink.seen.numbers(), vec![1, 2]);
    assert_eq!(next.numbers(), Vec::<u64>::new());
    submit(&b, vec![present(v, one, 3)]);
    assert_eq!(sink.seen.numbers(), Vec::<u64>::new());
    assert_eq!(next.numbers(), vec![3]);

    // Removed: frame 2 still reaches the sink, and later frames wait to be taken.
    let b = Arc::new(backend());
    let v = uploaded(&b, RGBA, (1, 1), &[1, 2, 3, 4]);
    let sink = Arc::new(Replacing {
        backend: Arc::downgrade(&b),
        next: None,
        seen: Collect::default(),
    });
    install(&b, &sink);
    submit(&b, vec![present(v, one, 1), present(v, one, 2)]);
    assert_eq!(sink.seen.numbers(), vec![1, 2]);
    assert_eq!(b.take_frame_dumps(), Vec::new());
    submit(&b, vec![present(v, one, 3)]);
    assert_eq!(sink.seen.numbers(), Vec::<u64>::new());
    let waiting: Vec<u64> = b.take_frame_dumps().iter().map(|d| d.frame).collect();
    assert_eq!(waiting, vec![3]);
}

/// A sink that stays inside `frame` for frame 1 until it is released.
#[derive(Debug)]
struct Held {
    entered: Mutex<Sender<()>>,
    release: Mutex<Receiver<()>>,
    seen: Mutex<Vec<u64>>,
}

impl Held {
    fn seen(&self) -> Vec<u64> {
        self.seen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl FrameSink for Held {
    fn frame(&self, frame: Frame) {
        if frame.frame == 1 {
            let entered = self.entered.lock().unwrap_or_else(PoisonError::into_inner);
            let release = self.release.lock().unwrap_or_else(PoisonError::into_inner);
            // A closed channel means the test has already failed; do not wait then.
            if entered.send(()).is_ok() {
                let _ = release.recv_timeout(WAIT);
            }
        }
        self.seen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(frame.frame);
    }
}

/// Longer than any step of the test takes; reaching it is a failure, not a pass.
const WAIT: Duration = Duration::from_secs(30);

/// Two threads submit. While the sink is busy with the first thread's frame, the second
/// thread's submit completes and returns without waiting for the sink, and its frame is
/// delivered after the first one, not past it.
#[test]
fn a_busy_sink_does_not_block_or_reorder_another_submit() {
    let b = Arc::new(backend());
    let v = uploaded(&b, RGBA, (1, 1), &[1, 2, 3, 4]);
    let one = rect(0, 0, 1, 1);
    let (entered_tx, entered_rx) = channel();
    let (release_tx, release_rx) = channel();
    let sink = Arc::new(Held {
        entered: Mutex::new(entered_tx),
        release: Mutex::new(release_rx),
        seen: Mutex::default(),
    });
    install(&b, &sink);
    let worker = {
        let b = Arc::clone(&b);
        std::thread::spawn(move || submit(&b, vec![present(v, one, 1)]))
    };
    entered_rx
        .recv_timeout(WAIT)
        .expect("the sink is given frame 1");
    // The sink is inside `frame` on the worker thread. This submit completes all the same.
    let second = submit(&b, vec![present(v, one, 2)]);
    assert_eq!(b.completed(), second);
    assert_eq!(sink.seen(), Vec::<u64>::new(), "frame 2 waits its turn");
    release_tx.send(()).expect("the sink is waiting");
    let first = worker.join().expect("the worker finished");
    assert!(first < second);
    assert_eq!(sink.seen(), vec![1, 2]);
}

/// A sink that fails on frame 1.
#[derive(Debug, Default)]
struct Failing(Collect);

impl FrameSink for Failing {
    fn frame(&self, frame: Frame) {
        if frame.frame == 1 {
            // A panic that does not print: the test expects it.
            std::panic::resume_unwind(Box::new("the sink failed"));
        }
        self.0.frame(frame);
    }
}

/// A sink that panics takes its caller's `submit` with it, but the list had completed, and
/// later submits still deliver: first the frames that were waiting behind the failed one.
#[test]
fn a_failing_sink_does_not_stop_later_frames() {
    let b = backend();
    let sink = Arc::new(Failing::default());
    install(&b, &sink);
    let v = uploaded(&b, RGBA, (1, 1), &[1, 2, 3, 4]);
    let one = rect(0, 0, 1, 1);
    let list = CommandList {
        commands: vec![present(v, one, 1), present(v, one, 2)],
    };
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| b.submit(list)));
    assert!(failed.is_err());
    assert_eq!(b.completed(), FenceValue::new(2), "the list completed");
    assert_eq!(sink.0.numbers(), Vec::<u64>::new());
    submit(&b, vec![present(v, one, 3)]);
    assert_eq!(sink.0.numbers(), vec![2, 3]);
}
