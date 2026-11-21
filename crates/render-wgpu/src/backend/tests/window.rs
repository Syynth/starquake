//! Window presentation on a real GPU (slice R8, `crate::window`). The tests have no window, so
//! a texture of the backend's own stands in for the window's (`Target::Texture`): the present
//! path draws into it exactly as into a window's texture (the same checks, the same copy, the
//! same blit, at the same place in the list), and acquires, discards and presents it through
//! the same steps, where acquiring, discarding and presenting only count (`TestWindow`). Only
//! wgpu's surface calls are left out; they, resizing and the main-thread layer are checked by
//! hand in a real window.

use super::*;
use crate::present::fit;
use crate::window::{NO_SURFACE, NOT_MAIN_THREAD, TestWindow};
use starquake_render::window::raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};
use starquake_render::{PresentOutcome, WindowHandles, WindowSize};
use std::sync::atomic::Ordering as Atomic;
use std::sync::mpsc;

/// The format of every image here: four bytes a texel, red first.
const RGBA: PixelFormat = PixelFormat::A8B8G8R8Unorm;

/// A guest image and its view, and the stand-in window texture.
struct Rig {
    gpu: WgpuBackend,
    image: TextureId,
    view: ViewId,
    format: PixelFormat,
    size: (u32, u32),
    target: TextureId,
    target_size: (u32, u32),
    /// What the stand-in counted.
    window: Arc<TestWindow>,
    next_readback: u32,
}

impl Rig {
    fn new(gpu: WgpuBackend, size: (u32, u32), target_size: (u32, u32)) -> Self {
        Self::with_formats(gpu, (RGBA, size), (RGBA, target_size))
    }

    /// An image and a stand-in target of the given formats and sizes.
    fn with_formats(
        gpu: WgpuBackend,
        (format, size): (PixelFormat, (u32, u32)),
        (target_format, target_size): (PixelFormat, (u32, u32)),
    ) -> Self {
        let image = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                format,
                (size.0, size.1, 1),
                1,
                TARGET,
            ))
            .expect("image");
        let view = gpu
            .create_view(image, &whole_view(format, ViewDimension::D2, (0, 1)))
            .expect("view");
        let target = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                target_format,
                (target_size.0, target_size.1, 1),
                1,
                TARGET,
            ))
            .expect("target");
        let window = gpu.set_test_target(target);
        Self {
            gpu,
            image,
            view,
            format,
            size,
            target,
            target_size,
            window,
            next_readback: 1,
        }
    }

    fn bytes(&self) -> usize {
        usize::try_from(self.size.0 * self.size.1 * self.format.bytes_per_block()).expect("small")
    }

    /// What the stand-in counted: acquired, presented, discarded.
    fn counts(&self) -> (u64, u64, u64) {
        counts(&self.window)
    }

    fn fill(&self, bytes: Vec<u8>) -> Command {
        upload(
            bytes,
            region(self.image, 0, (0, 0, 0), (self.size.0, self.size.1, 1)),
        )
    }

    fn present(&self, frame: u64, transform: PresentTransform, dump: bool) -> Command {
        Command::Present(PresentDesc {
            view: self.view,
            src_rect: Rect {
                x: 0,
                y: 0,
                width: self.size.0,
                height: self.size.1,
            },
            transform,
            frame,
            dump,
        })
    }

    fn run(&self, commands: Vec<Command>) {
        let fence = self.gpu.submit(CommandList { commands }).expect("accepted");
        finish(&self.gpu, fence);
    }

    /// The whole of `texture`, `size` texels, read back.
    fn read(&mut self, texture: TextureId, size: (u32, u32)) -> Vec<u8> {
        let id = ReadbackId::new(self.next_readback, 0);
        self.next_readback += 1;
        self.run(vec![Command::Readback {
            from: region(texture, 0, (0, 0, 0), (size.0, size.1, 1)),
            id,
        }]);
        match self.gpu.take_readback(id) {
            Some(Readback::Bytes(bytes)) => bytes,
            other => unreachable!("the readback came back as {other:?}"),
        }
    }

    fn shown(&mut self) -> Vec<u8> {
        self.read(self.target, self.target_size)
    }

    fn image_now(&mut self) -> Vec<u8> {
        self.read(self.image, self.size)
    }

    fn outcomes(&self) -> Vec<(u64, PresentOutcome)> {
        self.gpu
            .take_present_feedback()
            .iter()
            .map(|f| (f.frame, f.outcome))
            .collect()
    }
}

fn rig(test: &str, size: (u32, u32), target_size: (u32, u32)) -> Option<Rig> {
    backend(test).map(|gpu| Rig::new(gpu, size, target_size))
}

/// What `window` counted: acquired, presented, discarded.
fn counts(window: &TestWindow) -> (u64, u64, u64) {
    (
        window.acquired.load(Atomic::Relaxed),
        window.presented.load(Atomic::Relaxed),
        window.discarded.load(Atomic::Relaxed),
    )
}

/// Verification 9 ("Submit"): present an image, write into it, and the window still shows the
/// first frame. One case per kind of write, after the present in the same list and in the
/// next list: an upload (wgpu's queue write, which lands before the next submission's command
/// buffers), a clear and a copy.
#[test]
fn the_window_shows_the_presented_frame_after_the_image_is_written() {
    #[derive(Clone, Copy, Debug)]
    enum Write {
        Upload,
        Clear,
        Copy,
    }
    bounded(|| {
        for write in [Write::Upload, Write::Clear, Write::Copy] {
            for same_list in [true, false] {
                let Some(mut r) = rig("present_then_write", (4, 4), (4, 4)) else {
                    return;
                };
                let first = pattern(r.bytes(), 11);
                let second = pattern(r.bytes(), 97);
                let other = r
                    .gpu
                    .create_texture(&texture_desc(
                        TextureDimension::D2,
                        RGBA,
                        (4, 4, 1),
                        1,
                        TARGET,
                    ))
                    .expect("copy source");
                r.run(vec![upload(
                    second.clone(),
                    region(other, 0, (0, 0, 0), (4, 4, 1)),
                )]);
                let (writes, after) = match write {
                    Write::Upload => (vec![r.fill(second.clone())], second.clone()),
                    Write::Clear => (
                        vec![Command::Clear(ClearDesc::unmasked(
                            r.view,
                            ClearValue::Float([1.0, 0.0, 1.0, 1.0]),
                            None,
                        ))],
                        [255, 0, 255, 255].repeat(16),
                    ),
                    Write::Copy => (
                        vec![Command::Copy(CopyDesc {
                            src: region(other, 0, (0, 0, 0), (4, 4, 1)),
                            dst: region(r.image, 0, (0, 0, 0), (4, 4, 1)),
                        })],
                        second.clone(),
                    ),
                };
                let mut list = vec![
                    r.fill(first.clone()),
                    r.present(1, PresentTransform::Identity, false),
                ];
                if same_list {
                    list.extend(writes);
                    r.run(list);
                } else {
                    r.run(list);
                    r.run(writes);
                }
                let case = format!("{write:?}, same list: {same_list}");
                assert_eq!(r.image_now(), after, "the image was written ({case})");
                assert_eq!(r.shown(), first, "the window shows frame 1 ({case})");
                assert_eq!(r.outcomes(), vec![(1, PresentOutcome::Shown)], "{case}");
                assert_eq!(r.gpu.host_errors(), 0, "{case}");
            }
        }
    });
}

/// The ordering test with a draw as the write (slice R7): present an image, draw into it, and
/// the window still shows the presented frame, with the draw in the same list as the present
/// and in the next one.
#[test]
fn the_window_shows_the_presented_frame_after_a_draw_writes_the_image() {
    use super::draws::{ThreadPool, dynamic};
    use crate::fixtures::{graphics_desc, shader_desc, wgsl};
    use starquake_render::command::{DrawDesc, DrawParams, RenderTargets};
    use starquake_render::{ShaderInterface, ShaderStage};
    bounded(|| {
        for same_list in [true, false] {
            let Some(gpu) = super::open_with("present_then_draw", || {
                WgpuBackend::with_compile_pool(
                    WgpuConfig::DEFAULT,
                    CompletionWake::none(),
                    Arc::new(ThreadPool),
                    WhileCompiling::Wait,
                )
            }) else {
                return;
            };
            let mut r = Rig::new(gpu, (4, 4), (4, 4));
            let shader = |stage, source| {
                r.gpu
                    .create_shader(shader_desc(
                        stage,
                        "main",
                        wgsl(source),
                        ShaderInterface::default(),
                    ))
                    .expect("shader")
            };
            // One triangle that covers the whole target, in magenta.
            let vs = shader(
                ShaderStage::Vertex,
                "@vertex fn main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                     return vec4(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0);
                 }",
            );
            let fs = shader(
                ShaderStage::Fragment,
                "@fragment fn main() -> @location(0) vec4<f32> { return vec4(1.0, 0.0, 1.0, 1.0); }",
            );
            let pipeline = r
                .gpu
                .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
                .expect("accepted");
            let mut targets = RenderTargets::default();
            targets.colors[0] = Some(r.view);
            let mut viewport = dynamic();
            viewport.viewports[0].width = 4.0;
            viewport.viewports[0].height = 4.0;
            let writes = vec![
                Command::SetRenderTargets(targets),
                Command::Draw(DrawDesc {
                    pipeline,
                    bindings: Vec::new(),
                    vertex_buffers: Vec::new(),
                    index_buffer: None,
                    dynamic: viewport,
                    params: DrawParams::Direct {
                        vertex_count: 3,
                        instance_count: 1,
                        first_vertex: 0,
                        first_instance: 0,
                    },
                }),
            ];
            let first = pattern(r.bytes(), 11);
            let mut list = vec![
                r.fill(first.clone()),
                r.present(1, PresentTransform::Identity, false),
            ];
            if same_list {
                list.extend(writes);
                r.run(list);
            } else {
                r.run(list);
                r.run(writes);
            }
            let case = format!("same list: {same_list}");
            assert_eq!(
                r.image_now(),
                [255, 0, 255, 255].repeat(16),
                "the draw wrote the image ({case})"
            );
            assert_eq!(r.shown(), first, "the window shows frame 1 ({case})");
            assert_eq!(r.outcomes(), vec![(1, PresentOutcome::Shown)], "{case}");
            assert_eq!(r.gpu.host_errors(), 0, "{case}");
        }
    });
}

/// Where target pixel (x, y) of a `w` by `h` image turned by `transform` comes from, from the
/// geometry of each turn (independent of `present::corners`).
fn source_of(transform: PresentTransform, (w, h): (u32, u32), x: u32, y: u32) -> (u32, u32) {
    match transform {
        PresentTransform::Identity => (x, y),
        PresentTransform::FlipHorizontal => (w - 1 - x, y),
        PresentTransform::FlipVertical => (x, h - 1 - y),
        PresentTransform::Rotate90 => (y, h - 1 - x),
        PresentTransform::Rotate180 => (w - 1 - x, h - 1 - y),
        PresentTransform::Rotate270 => (w - 1 - y, x),
    }
}

/// The blit's image is the right way up and each transform turns it as `PresentTransform`
/// says, pixel for pixel at its own size (R4 left which way up to this slice). The image is 3
/// by 2, so a turn that swapped width and height, or rows and columns, shows.
#[test]
fn each_transform_shows_the_image_turned_as_named() {
    bounded(|| {
        let size = (3, 2);
        for transform in [
            PresentTransform::Identity,
            PresentTransform::FlipHorizontal,
            PresentTransform::FlipVertical,
            PresentTransform::Rotate90,
            PresentTransform::Rotate180,
            PresentTransform::Rotate270,
        ] {
            let rect = Rect {
                x: 0,
                y: 0,
                width: size.0,
                height: size.1,
            };
            let shown = crate::present::shown_size(rect, transform);
            let Some(mut r) = rig("transforms", size, shown) else {
                return;
            };
            let image = pattern(r.bytes(), 29);
            r.run(vec![r.fill(image.clone()), r.present(1, transform, false)]);
            let mut expected = Vec::new();
            for y in 0..shown.1 {
                for x in 0..shown.0 {
                    let (sx, sy) = source_of(transform, size, x, y);
                    let at = usize::try_from((sy * size.0 + sx) * 4).expect("small");
                    expected.extend_from_slice(&image[at..at + 4]);
                }
            }
            assert_eq!(r.shown(), expected, "{transform:?}");
            assert_eq!(r.gpu.host_errors(), 0);
        }
    });
}

/// An image of another aspect ratio is scaled to fit and centred, with black bars; a flat
/// colour stays exactly that colour however it is filtered.
#[test]
fn an_image_is_scaled_to_fit_with_black_bars() {
    bounded(|| {
        let colour = [10, 200, 30, 255];
        for (size, target) in [((2, 1), (4, 4)), ((1, 2), (6, 2)), ((4, 4), (8, 8))] {
            let Some(mut r) = rig("letterbox", size, target) else {
                return;
            };
            let image = colour.repeat(r.bytes() / 4);
            r.run(vec![
                r.fill(image),
                r.present(1, PresentTransform::Identity, false),
            ]);
            let screen = fit(size, target).expect("both sides non-zero");
            let mut expected = Vec::new();
            for y in 0..target.1 {
                for x in 0..target.0 {
                    let inside = (screen.x..screen.x + screen.width).contains(&x)
                        && (screen.y..screen.y + screen.height).contains(&y);
                    expected.extend_from_slice(if inside { &colour } else { &[0, 0, 0, 255] });
                }
            }
            assert_eq!(r.shown(), expected, "{size:?} in {target:?}");
            assert_eq!(r.gpu.host_errors(), 0);
        }
    });
}

/// Only the last present of a list is shown; the others are counted as dropped.
#[test]
fn only_the_last_present_of_a_list_is_shown() {
    bounded(|| {
        let Some(mut r) = rig("last_of_list", (2, 2), (2, 2)) else {
            return;
        };
        let (a, b, c) = (pattern(16, 1), pattern(16, 2), pattern(16, 3));
        r.run(vec![
            r.fill(a),
            r.present(1, PresentTransform::Identity, false),
            r.fill(b.clone()),
            r.present(2, PresentTransform::Identity, false),
            r.fill(c),
        ]);
        assert_eq!(r.shown(), b);
        assert_eq!(
            r.outcomes(),
            vec![(1, PresentOutcome::Dropped), (2, PresentOutcome::Shown)]
        );
        assert_eq!(r.gpu.presents(), 2, "both are counted");
    });
}

/// Holds the render thread inside the frame sink until released, so lists queue up behind it.
#[derive(Debug)]
struct Gate {
    entered: Mutex<Option<mpsc::Sender<()>>>,
    release: Mutex<Option<mpsc::Receiver<()>>>,
}

impl FrameSink for Gate {
    fn frame(&self, _frame: Frame) {
        if let Some(entered) = self.entered.lock().expect("unpoisoned").take() {
            entered.send(()).expect("the test waits");
            if let Some(release) = self.release.lock().expect("unpoisoned").take() {
                release.recv_timeout(LIST_WAIT).expect("released in time");
            }
        }
    }
}

/// A present whose list waited in the queue behind a newer present is dropped (the newest
/// frame wins), and the count of queued presents comes back down when the render thread takes
/// a list: a later lone present is shown.
#[test]
fn a_present_queued_behind_a_newer_one_is_dropped() {
    bounded(|| {
        let Some(mut r) = rig("newest_wins", (2, 2), (2, 2)) else {
            return;
        };
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        r.gpu.set_frame_sink(Some(Arc::new(Gate {
            entered: Mutex::new(Some(entered_tx)),
            release: Mutex::new(Some(release_rx)),
        })));
        let [one, two, three, four] = [1, 2, 3, 4].map(|seed| pattern(16, seed));
        // List 1's dumped frame stops the render thread in the sink, after list 1 completed.
        let first = r
            .gpu
            .submit(CommandList {
                commands: vec![r.fill(one), r.present(1, PresentTransform::Identity, true)],
            })
            .expect("accepted");
        entered
            .recv_timeout(LIST_WAIT)
            .expect("the sink was entered");
        assert_eq!(r.gpu.completed(), first);
        // Two lists queue behind it, each with a present.
        let lists = [(two, 2), (three.clone(), 3)].map(|(bytes, frame)| {
            r.gpu
                .submit(CommandList {
                    commands: vec![
                        r.fill(bytes),
                        r.present(frame, PresentTransform::Identity, false),
                    ],
                })
                .expect("accepted")
        });
        release.send(()).expect("the sink waits");
        finish(&r.gpu, lists[1]);
        assert_eq!(r.shown(), three, "the newest frame");
        assert_eq!(
            r.outcomes(),
            vec![
                (1, PresentOutcome::Shown),
                (2, PresentOutcome::Dropped),
                (3, PresentOutcome::Shown),
            ]
        );
        // Nothing is queued now: a lone present is shown.
        r.run(vec![
            r.fill(four.clone()),
            r.present(4, PresentTransform::Identity, false),
        ]);
        assert_eq!(r.outcomes(), vec![(4, PresentOutcome::Shown)]);
        assert_eq!(r.shown(), four);
        assert_eq!(r.gpu.host_errors(), 0);
    });
}

/// Back to the offscreen target, presents are counted as shown and nothing is drawn.
#[test]
fn the_offscreen_target_draws_nothing() {
    bounded(|| {
        let Some(mut r) = rig("offscreen_again", (2, 2), (2, 2)) else {
            return;
        };
        let (a, b) = (pattern(16, 5), pattern(16, 6));
        r.run(vec![
            r.fill(a.clone()),
            r.present(1, PresentTransform::Identity, false),
        ]);
        assert_eq!(r.gpu.set_present_target(PresentTarget::Offscreen), Ok(()));
        r.run(vec![
            r.fill(b),
            r.present(2, PresentTransform::Identity, false),
            r.present(3, PresentTransform::Identity, false),
        ]);
        assert_eq!(r.shown(), a, "the stand-in still holds frame 1");
        assert_eq!(
            r.outcomes(),
            vec![
                (1, PresentOutcome::Shown),
                (2, PresentOutcome::Shown),
                (3, PresentOutcome::Shown),
            ]
        );
    });
}

/// A window whose handles are never available.
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

/// A window target that cannot be made is refused with its exact error and changes nothing:
/// the target in force keeps getting frames. `bounded` runs the test on a thread of its own,
/// never the main thread, so on macOS the thread check refuses it first; elsewhere wgpu cannot
/// make a surface from handles that are unavailable.
#[test]
fn a_window_target_that_cannot_be_made_changes_nothing() {
    bounded(|| {
        let Some(mut r) = rig("refused_window", (2, 2), (2, 2)) else {
            return;
        };
        let window = WindowHandles::new(Arc::new(NoWindow), WindowSize::new(2, 2));
        let expected = if cfg!(target_os = "macos") {
            NOT_MAIN_THREAD
        } else {
            NO_SURFACE
        };
        assert_eq!(
            r.gpu
                .set_present_target(PresentTarget::Window(window))
                .map_err(|e| e.to_string()),
            Err(expected.to_owned())
        );
        let a = pattern(16, 7);
        r.run(vec![
            r.fill(a.clone()),
            r.present(1, PresentTransform::Identity, false),
        ]);
        assert_eq!(r.shown(), a, "the stand-in still gets frames");
        assert_eq!(r.outcomes(), vec![(1, PresentOutcome::Shown)]);
    });
}

/// A present of part of an array texture's layer: the window shows that rectangle of that
/// layer, so the blit's copy takes the layer and the corner the present names.
#[test]
fn a_rectangle_of_a_layer_is_shown() {
    bounded(|| {
        let Some(gpu) = backend("layer_rect") else {
            return;
        };
        let array = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                RGBA,
                (4, 4, 3),
                1,
                TARGET,
            ))
            .expect("array");
        let layer = gpu
            .create_view(array, &whole_view(RGBA, ViewDimension::D2, (2, 1)))
            .expect("view of layer 2");
        let target = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                RGBA,
                (2, 3, 1),
                1,
                TARGET,
            ))
            .expect("target");
        gpu.set_test_target(target);
        let bytes = pattern(4 * 4 * 3 * 4, 41);
        let present = Command::Present(PresentDesc {
            view: layer,
            src_rect: Rect {
                x: 1,
                y: 1,
                width: 2,
                height: 3,
            },
            transform: PresentTransform::Identity,
            frame: 1,
            dump: false,
        });
        let id = ReadbackId::new(1, 0);
        let fence = gpu
            .submit(CommandList {
                commands: vec![
                    upload(bytes.clone(), region(array, 0, (0, 0, 0), (4, 4, 3))),
                    present,
                    Command::Readback {
                        from: region(target, 0, (0, 0, 0), (2, 3, 1)),
                        id,
                    },
                ],
            })
            .expect("accepted");
        finish(&gpu, fence);
        let mut expected = Vec::new();
        for y in 1..4 {
            for x in 1..3 {
                let at = ((2 * 4 + y) * 4 + x) * 4;
                expected.extend_from_slice(&bytes[at..at + 4]);
            }
        }
        assert_eq!(gpu.take_readback(id), Some(Readback::Bytes(expected)));
        assert_eq!(gpu.host_errors(), 0);
    });
}

/// A compressed image is shown when its rectangle starts on a block, and dropped, with no
/// error from wgpu, when it starts inside one (`present::starts_on_block`). The dropped frame
/// acquires no window texture (`crate::window`, "A frame that cannot be shown": a texture
/// acquired and not drawn would be presented cleared), and each shown one acquires and presents
/// one.
#[test]
fn a_compressed_present_inside_a_block_is_dropped() {
    bounded(|| {
        let Some(gpu) = backend("compressed_present") else {
            return;
        };
        let bc = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                PixelFormat::Bc1RgbaUnorm,
                (8, 8, 1),
                1,
                COPIES,
            ))
            .expect("BC1 texture");
        let view = gpu
            .create_view(
                bc,
                &whole_view(PixelFormat::Bc1RgbaUnorm, ViewDimension::D2, (0, 1)),
            )
            .expect("view");
        let target = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                RGBA,
                (4, 4, 1),
                1,
                TARGET,
            ))
            .expect("target");
        let window = gpu.set_test_target(target);
        let present = |frame, x| {
            Command::Present(PresentDesc {
                view,
                src_rect: Rect {
                    x,
                    y: 4,
                    width: 3,
                    height: 4,
                },
                transform: PresentTransform::Identity,
                frame,
                dump: false,
            })
        };
        for (frame, x, outcome, after) in [
            (1, 4, PresentOutcome::Shown, (1, 1, 0)),
            (2, 1, PresentOutcome::Dropped, (1, 1, 0)),
            (3, 0, PresentOutcome::Shown, (2, 2, 0)),
        ] {
            let fence = gpu
                .submit(CommandList {
                    commands: vec![
                        upload(pattern(32, 3), region(bc, 0, (0, 0, 0), (8, 8, 1))),
                        present(frame, x),
                    ],
                })
                .expect("accepted");
            finish(&gpu, fence);
            let got: Vec<_> = gpu
                .take_present_feedback()
                .iter()
                .map(|f| (f.frame, f.outcome))
                .collect();
            assert_eq!(got, vec![(frame, outcome)], "x = {x}");
            assert_eq!(
                counts(&window),
                after,
                "acquired, presented, discarded; x = {x}"
            );
        }
        assert_eq!(gpu.host_errors(), 0);
    });
}

/// A blit that fails after the window's texture was acquired discards the texture instead of
/// presenting it (a presented undrawn texture would show black), reports the frame dropped, and
/// leaves the next frame unaffected.
#[test]
fn a_blit_that_fails_after_the_acquire_discards_the_texture() {
    bounded(|| {
        let Some(mut r) = rig("fault_after_acquire", (2, 2), (2, 2)) else {
            return;
        };
        let [a, b, c] = [1, 2, 3].map(|seed| pattern(16, seed));
        r.run(vec![
            r.fill(a.clone()),
            r.present(1, PresentTransform::Identity, false),
        ]);
        assert_eq!(r.counts(), (1, 1, 0));
        r.window.fail_next_blit.store(true, Atomic::Relaxed);
        r.run(vec![
            r.fill(b),
            r.present(2, PresentTransform::Identity, false),
        ]);
        assert_eq!(
            r.outcomes(),
            vec![(1, PresentOutcome::Shown), (2, PresentOutcome::Dropped)]
        );
        assert_eq!(
            r.counts(),
            (2, 1, 1),
            "acquired, then discarded, not presented"
        );
        assert_eq!(r.shown(), a, "the stand-in still holds frame 1");
        r.run(vec![
            r.fill(c.clone()),
            r.present(3, PresentTransform::Identity, false),
        ]);
        assert_eq!(r.outcomes(), vec![(3, PresentOutcome::Shown)]);
        assert_eq!(r.counts(), (3, 2, 1));
        assert_eq!(r.shown(), c);
        assert_eq!(r.gpu.host_errors(), 0);
    });
}

/// An 8-bit image reaches the window as the guest wrote it whatever the sRGB-ness of the
/// presented view and of the window's format: the blit writes through the target's view whose
/// sRGB-ness matches the presented view's (`present::write_format`), so each byte is decoded
/// and encoded by the same curve. Each case is one present at its own size of a pattern that
/// holds mid-range values, where a linear-to-sRGB mismatch moves a byte (200 becomes 147 or
/// 228).
#[test]
fn an_srgb_or_linear_present_reaches_the_window_as_written() {
    bounded(|| {
        let srgb = PixelFormat::A8B8G8R8Srgb;
        for (image, target) in [(srgb, RGBA), (srgb, srgb), (RGBA, srgb), (RGBA, RGBA)] {
            let Some(gpu) = backend("srgb_present") else {
                return;
            };
            let mut r = Rig::with_formats(gpu, (image, (4, 4)), (target, (4, 4)));
            let mut bytes = pattern(r.bytes(), 17);
            bytes[..4].copy_from_slice(&[200, 100, 50, 255]);
            r.run(vec![
                r.fill(bytes.clone()),
                r.present(1, PresentTransform::Identity, false),
            ]);
            let case = format!("{image:?} into {target:?}");
            assert_eq!(r.shown(), bytes, "{case}");
            assert_eq!(r.outcomes(), vec![(1, PresentOutcome::Shown)], "{case}");
            assert_eq!(r.gpu.host_errors(), 0, "{case}");
        }
    });
}

/// A float format the device cannot filter (`R32G32B32A32Float`, without wgpu's
/// `FLOAT32_FILTERABLE`, which the backend never asks for) is scaled with the nearest filter
/// through the unfilterable layout and sampler, with no error from wgpu: each of the 2 by 1
/// image's texels covers a 2 by 2 square of the 4 by 2 window.
#[test]
fn an_unfilterable_format_is_scaled_with_the_nearest_filter() {
    bounded(|| {
        let Some(gpu) = backend("unfilterable_present") else {
            return;
        };
        let float = PixelFormat::R32G32B32A32Float;
        let mut r = Rig::with_formats(gpu, (float, (2, 1)), (RGBA, (4, 2)));
        let texels: [[f32; 4]; 2] = [[1.0, 0.2, 0.0, 1.0], [0.0, 0.6, 1.0, 1.0]];
        let bytes: Vec<u8> = texels
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        r.run(vec![
            r.fill(bytes),
            r.present(1, PresentTransform::Identity, false),
        ]);
        // 0.2 and 0.6 of 255 are 51 and 153.
        let (left, right) = ([255, 51, 0, 255], [0, 153, 255, 255]);
        let row = [left, left, right, right].concat();
        assert_eq!(r.shown(), row.repeat(2));
        assert_eq!(r.outcomes(), vec![(1, PresentOutcome::Shown)]);
        assert_eq!(r.gpu.host_errors(), 0);
    });
}

/// A scaled image of a filterable format is filtered linearly: a 2 by 1 black and white image
/// shown at 8 by 4 ramps from black to white across each row, with values strictly between.
/// The nearest filter would give only black and white.
#[test]
fn a_scaled_image_is_filtered_linearly() {
    bounded(|| {
        let Some(mut r) = rig("linear_present", (2, 1), (8, 4)) else {
            return;
        };
        let (black, white) = ([0, 0, 0, 255], [255, 255, 255, 255]);
        r.run(vec![
            r.fill([black, white].concat()),
            r.present(1, PresentTransform::Identity, false),
        ]);
        let shown = r.shown();
        let rows: Vec<&[u8]> = shown.chunks(8 * 4).collect();
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|row| *row == rows[0]), "every row alike");
        let reds: Vec<u8> = rows[0].chunks(4).map(|p| p[0]).collect();
        assert_eq!(reds.first(), Some(&0), "clamped to the black texel");
        assert_eq!(reds.last(), Some(&255), "clamped to the white texel");
        assert!(reds.windows(2).all(|w| w[0] <= w[1]), "a ramp: {reds:?}");
        assert!(
            reds.iter().any(|&v| v != 0 && v != 255),
            "a value strictly between: {reds:?}"
        );
        assert_eq!(r.gpu.host_errors(), 0);
    });
}

/// A present of a view whose base level is 1 shows level 1's pixels, not level 0's.
#[test]
fn a_view_of_a_later_level_shows_that_level() {
    bounded(|| {
        let Some(gpu) = backend("level_present") else {
            return;
        };
        let image = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                RGBA,
                (4, 4, 1),
                2,
                TARGET,
            ))
            .expect("image with two levels");
        let level_one = gpu
            .create_view(
                image,
                &ViewDesc {
                    base_mip: 1,
                    ..whole_view(RGBA, ViewDimension::D2, (0, 1))
                },
            )
            .expect("view of level 1");
        let target = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                RGBA,
                (2, 2, 1),
                1,
                TARGET,
            ))
            .expect("target");
        gpu.set_test_target(target);
        let (zero, one) = (pattern(4 * 4 * 4, 23), pattern(2 * 2 * 4, 61));
        let id = ReadbackId::new(1, 0);
        let fence = gpu
            .submit(CommandList {
                commands: vec![
                    upload(zero, region(image, 0, (0, 0, 0), (4, 4, 1))),
                    upload(one.clone(), region(image, 1, (0, 0, 0), (2, 2, 1))),
                    Command::Present(PresentDesc {
                        view: level_one,
                        src_rect: Rect {
                            x: 0,
                            y: 0,
                            width: 2,
                            height: 2,
                        },
                        transform: PresentTransform::Identity,
                        frame: 1,
                        dump: false,
                    }),
                    Command::Readback {
                        from: region(target, 0, (0, 0, 0), (2, 2, 1)),
                        id,
                    },
                ],
            })
            .expect("accepted");
        finish(&gpu, fence);
        assert_eq!(gpu.take_readback(id), Some(Readback::Bytes(one)));
        assert_eq!(gpu.host_errors(), 0);
    });
}

/// Same-size presents of three formats back to back (RGBA, then BGRA, then sRGB RGBA), each in
/// its own list: the backend's sampled copy is made again for each format, so each is shown
/// exactly, with no error from wgpu (a copy between formats would be refused).
#[test]
fn same_size_presents_of_other_formats_are_each_shown() {
    bounded(|| {
        let Some(gpu) = backend("format_switch") else {
            return;
        };
        let target = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                RGBA,
                (2, 2, 1),
                1,
                TARGET,
            ))
            .expect("target");
        gpu.set_test_target(target);
        let formats = [
            PixelFormat::A8B8G8R8Unorm,
            PixelFormat::B8G8R8A8Unorm,
            PixelFormat::A8B8G8R8Srgb,
        ];
        for ((frame, format), seed) in (1..).zip(formats).zip([5, 9, 13]) {
            let image = gpu
                .create_texture(&texture_desc(
                    TextureDimension::D2,
                    format,
                    (2, 2, 1),
                    1,
                    COPIES,
                ))
                .expect("image");
            let view = gpu
                .create_view(image, &whole_view(format, ViewDimension::D2, (0, 1)))
                .expect("view");
            let bytes = pattern(16, seed);
            let id = ReadbackId::new(frame, 0);
            let fence = gpu
                .submit(CommandList {
                    commands: vec![
                        upload(bytes.clone(), region(image, 0, (0, 0, 0), (2, 2, 1))),
                        Command::Present(PresentDesc {
                            view,
                            src_rect: Rect {
                                x: 0,
                                y: 0,
                                width: 2,
                                height: 2,
                            },
                            transform: PresentTransform::Identity,
                            frame: u64::from(frame),
                            dump: false,
                        }),
                        Command::Readback {
                            from: region(target, 0, (0, 0, 0), (2, 2, 1)),
                            id,
                        },
                    ],
                })
                .expect("accepted");
            finish(&gpu, fence);
            // The window is red first: a BGRA texel's bytes come out with blue and red swapped.
            let expected: Vec<u8> = if format == PixelFormat::B8G8R8A8Unorm {
                bytes
                    .chunks(4)
                    .flat_map(|p| [p[2], p[1], p[0], p[3]])
                    .collect()
            } else {
                bytes
            };
            assert_eq!(
                gpu.take_readback(id),
                Some(Readback::Bytes(expected)),
                "{format:?}"
            );
            let outcome: Vec<_> = gpu
                .take_present_feedback()
                .iter()
                .map(|f| f.outcome)
                .collect();
            assert_eq!(outcome, vec![PresentOutcome::Shown], "{format:?}");
            assert_eq!(gpu.host_errors(), 0, "{format:?}");
        }
    });
}

/// A present of part of a compressed block shows that part, at its own size: the top-left 2
/// by 2 of a BC1 block, shown in a 2 by 2 window, equals the top-left 2 by 2 of the whole block
/// shown at 1:1 in a 4 by 4 one. The blit copies the whole block and samples only the
/// presented part of it (`present::Crop`).
#[test]
fn part_of_a_compressed_block_is_cropped_from_the_copied_block() {
    bounded(|| {
        let Some(gpu) = backend("compressed_crop") else {
            return;
        };
        let bc = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                PixelFormat::Bc1RgbaUnorm,
                (4, 4, 1),
                1,
                COPIES,
            ))
            .expect("BC1 texture");
        let view = gpu
            .create_view(
                bc,
                &whole_view(PixelFormat::Bc1RgbaUnorm, ViewDimension::D2, (0, 1)),
            )
            .expect("view");
        // Two colours far apart, and indices that differ texel to texel.
        let block = vec![0x00, 0xf8, 0x1f, 0x00, 0xe4, 0x1b, 0x4e, 0xb1];
        gpu.submit(CommandList {
            commands: vec![upload(block, region(bc, 0, (0, 0, 0), (4, 4, 1)))],
        })
        .expect("accepted");
        let mut shown = Vec::new();
        for (frame, side) in [(1, 4), (2, 2)] {
            let target = gpu
                .create_texture(&texture_desc(
                    TextureDimension::D2,
                    RGBA,
                    (side, side, 1),
                    1,
                    TARGET,
                ))
                .expect("target");
            gpu.set_test_target(target);
            let id = ReadbackId::new(frame, 0);
            let fence = gpu
                .submit(CommandList {
                    commands: vec![
                        Command::Present(PresentDesc {
                            view,
                            src_rect: Rect {
                                x: 0,
                                y: 0,
                                width: side,
                                height: side,
                            },
                            transform: PresentTransform::Identity,
                            frame: u64::from(frame),
                            dump: false,
                        }),
                        Command::Readback {
                            from: region(target, 0, (0, 0, 0), (side, side, 1)),
                            id,
                        },
                    ],
                })
                .expect("accepted");
            finish(&gpu, fence);
            match gpu.take_readback(id) {
                Some(Readback::Bytes(bytes)) => shown.push(bytes),
                other => unreachable!("the readback came back as {other:?}"),
            }
        }
        let corner: Vec<u8> = shown[0]
            .chunks(4 * 4)
            .take(2)
            .flat_map(|row| row[..2 * 4].to_vec())
            .collect();
        assert_ne!(
            corner,
            shown[0][..corner.len()],
            "the corner is not the block's first row"
        );
        assert_eq!(shown[1], corner);
        assert_eq!(gpu.host_errors(), 0);
    });
}

/// The block check reads x against the block's width and y against its height: with ASTC's 8 by
/// 5 blocks, a rectangle at (8, 5) starts on a block and is shown, one at (5, 8) does not and is
/// dropped (square blocks could not tell the axes apart).
#[test]
fn the_block_check_reads_each_axis_against_its_own_side() {
    bounded(|| {
        let Some(gpu) = backend("astc_present") else {
            return;
        };
        let astc = PixelFormat::Astc2d8x5Unorm;
        let image = match gpu.create_texture(&texture_desc(
            TextureDimension::D2,
            astc,
            (16, 10, 1),
            1,
            COPIES,
        )) {
            Ok(image) => image,
            Err(RenderError::Unsupported(why)) => {
                eprintln!("skipping astc_present: this device has no ASTC ({why})");
                return;
            }
            Err(other) => unreachable!("ASTC texture: {other}"),
        };
        let view = gpu
            .create_view(image, &whole_view(astc, ViewDimension::D2, (0, 1)))
            .expect("view");
        let target = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                RGBA,
                (8, 5, 1),
                1,
                TARGET,
            ))
            .expect("target");
        let window = gpu.set_test_target(target);
        let blocks = usize::try_from(2 * 2 * astc.bytes_per_block()).expect("small");
        gpu.submit(CommandList {
            commands: vec![upload(
                pattern(blocks, 3),
                region(image, 0, (0, 0, 0), (16, 10, 1)),
            )],
        })
        .expect("accepted");
        for (frame, rect, outcome, after) in [
            (1, (8, 5, 8, 5), PresentOutcome::Shown, (1, 1, 0)),
            (2, (5, 8, 8, 2), PresentOutcome::Dropped, (1, 1, 0)),
        ] {
            let fence = gpu
                .submit(CommandList {
                    commands: vec![Command::Present(PresentDesc {
                        view,
                        src_rect: Rect {
                            x: rect.0,
                            y: rect.1,
                            width: rect.2,
                            height: rect.3,
                        },
                        transform: PresentTransform::Identity,
                        frame,
                        dump: false,
                    })],
                })
                .expect("accepted");
            finish(&gpu, fence);
            let got: Vec<_> = gpu
                .take_present_feedback()
                .iter()
                .map(|f| (f.frame, f.outcome))
                .collect();
            assert_eq!(got, vec![(frame, outcome)], "{rect:?}");
            assert_eq!(counts(&window), after, "{rect:?}");
        }
        assert_eq!(gpu.host_errors(), 0);
    });
}

/// Once the device is lost, a present target is refused like every other call that would queue
/// work; a window that cannot be made is refused for that reason first.
#[test]
fn a_lost_device_refuses_a_present_target() {
    bounded(|| {
        let counter = Arc::new(WakeCounter::new());
        let Some(gpu) = open(
            "lost_target",
            WgpuConfig::DEFAULT,
            counter.completion_wake(),
        ) else {
            return;
        };
        assert_eq!(gpu.set_present_target(PresentTarget::Offscreen), Ok(()));
        let seen = counter.count();
        gpu.lose_device_for_test();
        assert!(counter.wait_beyond(seen, LIST_WAIT), "the loss is seen");
        assert_eq!(
            gpu.set_present_target(PresentTarget::Offscreen),
            Err(RenderError::DeviceLost)
        );
        let window = WindowHandles::new(Arc::new(NoWindow), WindowSize::new(2, 2));
        let first = if cfg!(target_os = "macos") {
            NOT_MAIN_THREAD
        } else {
            NO_SURFACE
        };
        assert_eq!(
            gpu.set_present_target(PresentTarget::Window(window)),
            Err(RenderError::Unsupported(first))
        );
    });
}
