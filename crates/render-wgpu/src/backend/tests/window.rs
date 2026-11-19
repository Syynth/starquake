//! Window presentation on a real GPU (slice R8, `crate::window`). The tests have no window, so
//! a texture of the backend's own stands in for the window's (`Target::Texture`): the present
//! path draws into it exactly as into a window's texture (the same copy, the same blit, at the
//! same place in the list), and only acquiring and presenting the window's texture are left
//! out. That path, resizing and the main-thread layer are checked by hand in a real window.

use super::*;
use crate::present::fit;
use crate::window::{NO_SURFACE, NOT_MAIN_THREAD};
use starquake_render::window::raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};
use starquake_render::{PresentOutcome, WindowHandles, WindowSize};
use std::sync::mpsc;

/// The format of every image here: four bytes a texel, red first.
const RGBA: PixelFormat = PixelFormat::A8B8G8R8Unorm;

/// A guest image and its view, and the stand-in window texture.
struct Rig {
    gpu: WgpuBackend,
    image: TextureId,
    view: ViewId,
    size: (u32, u32),
    target: TextureId,
    target_size: (u32, u32),
    next_readback: u32,
}

impl Rig {
    fn new(gpu: WgpuBackend, size: (u32, u32), target_size: (u32, u32)) -> Self {
        let image = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                RGBA,
                (size.0, size.1, 1),
                1,
                TARGET,
            ))
            .expect("image");
        let view = gpu
            .create_view(image, &whole_view(RGBA, ViewDimension::D2, (0, 1)))
            .expect("view");
        let target = gpu
            .create_texture(&texture_desc(
                TextureDimension::D2,
                RGBA,
                (target_size.0, target_size.1, 1),
                1,
                TARGET,
            ))
            .expect("target");
        gpu.set_test_target(target);
        Self {
            gpu,
            image,
            view,
            size,
            target,
            target_size,
            next_readback: 1,
        }
    }

    fn bytes(&self) -> usize {
        usize::try_from(self.size.0 * self.size.1 * 4).expect("small")
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
                        vec![Command::Clear(ClearDesc {
                            target: r.view,
                            value: ClearValue::Float([1.0, 0.0, 1.0, 1.0]),
                            rect: None,
                        })],
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
/// error from wgpu, when it starts inside one (`present::starts_on_block`).
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
        gpu.set_test_target(target);
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
        for (frame, x, outcome) in [
            (1, 4, PresentOutcome::Shown),
            (2, 1, PresentOutcome::Dropped),
            (3, 0, PresentOutcome::Shown),
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
