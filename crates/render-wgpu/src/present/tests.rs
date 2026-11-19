use super::*;
use starquake_render::TextureId;
use starquake_render::resource::{Extent3d, Origin3d};
use wgpu::{CompositeAlphaMode as Alpha, PresentMode as Mode, TextureFormat as F};

/// Every subset of `items`, as lists in the order given and reversed.
fn subsets<T: Copy>(items: &[T]) -> Vec<Vec<T>> {
    let n = u32::try_from(items.len()).expect("few");
    let mut out = Vec::new();
    for mask in 0..(1u32 << n) {
        let set: Vec<T> = (0..n)
            .zip(items)
            .filter(|(bit, _)| mask & (1 << bit) != 0)
            .map(|(_, &item)| item)
            .collect();
        let mut reversed = set.clone();
        reversed.reverse();
        out.push(set);
        out.push(reversed);
    }
    out
}

#[test]
fn the_present_mode_for_every_set_of_modes() {
    let all = [
        Mode::AutoVsync,
        Mode::AutoNoVsync,
        Mode::Fifo,
        Mode::FifoRelaxed,
        Mode::Immediate,
        Mode::Mailbox,
    ];
    let sets = subsets(&all);
    assert_eq!(sets.len(), 128);
    for set in sets {
        // The spec's order, read as a rule: mailbox wins, then immediate, then FIFO.
        let expected = if set.contains(&Mode::Mailbox) {
            Some(Mode::Mailbox)
        } else if set.contains(&Mode::Immediate) {
            Some(Mode::Immediate)
        } else if set.contains(&Mode::Fifo) {
            Some(Mode::Fifo)
        } else {
            None
        };
        assert_eq!(present_mode(&set), expected, "{set:?}");
    }
    // Hand-written cases.
    assert_eq!(
        present_mode(&[Mode::Fifo, Mode::Mailbox]),
        Some(Mode::Mailbox)
    );
    assert_eq!(
        present_mode(&[Mode::Immediate, Mode::Fifo]),
        Some(Mode::Immediate)
    );
    assert_eq!(
        present_mode(&[Mode::Mailbox, Mode::Immediate]),
        Some(Mode::Mailbox)
    );
    assert_eq!(present_mode(&[Mode::Fifo]), Some(Mode::Fifo));
    assert_eq!(present_mode(&[Mode::FifoRelaxed, Mode::AutoNoVsync]), None);
    assert_eq!(present_mode(&[]), None);
}

#[test]
fn the_surface_format_is_bgra8_unorm_else_the_first() {
    assert_eq!(
        surface_format(&[F::Rgba8Unorm, F::Bgra8UnormSrgb, F::Bgra8Unorm]),
        Some(F::Bgra8Unorm)
    );
    assert_eq!(
        surface_format(&[F::Bgra8Unorm, F::Rgba8Unorm]),
        Some(F::Bgra8Unorm)
    );
    // No B8G8R8A8_UNORM: the first listed, whatever it is.
    assert_eq!(
        surface_format(&[F::Rgba16Float, F::Bgra8UnormSrgb, F::Rgba8Unorm]),
        Some(F::Rgba16Float)
    );
    assert_eq!(
        surface_format(&[F::Bgra8UnormSrgb, F::Rgba16Float]),
        Some(F::Bgra8UnormSrgb)
    );
    assert_eq!(surface_format(&[]), None);
}

#[test]
fn the_alpha_mode_for_every_set_of_modes() {
    let all = [
        Alpha::Auto,
        Alpha::Opaque,
        Alpha::PreMultiplied,
        Alpha::PostMultiplied,
        Alpha::Inherit,
    ];
    let sets = subsets(&all);
    assert_eq!(sets.len(), 64);
    for set in sets {
        let expected = if set.contains(&Alpha::Opaque) {
            Some(Alpha::Opaque)
        } else if set.contains(&Alpha::Inherit) {
            Some(Alpha::Inherit)
        } else {
            None
        };
        assert_eq!(alpha_mode(&set), expected, "{set:?}");
    }
    assert_eq!(
        alpha_mode(&[Alpha::Inherit, Alpha::Opaque]),
        Some(Alpha::Opaque)
    );
    assert_eq!(
        alpha_mode(&[Alpha::PreMultiplied, Alpha::Inherit]),
        Some(Alpha::Inherit)
    );
    assert_eq!(alpha_mode(&[Alpha::PreMultiplied, Alpha::Auto]), None);
}

fn capabilities(
    formats: &[F],
    modes: &[Mode],
    alphas: &[Alpha],
    usages: wgpu::TextureUsages,
) -> wgpu::SurfaceCapabilities {
    wgpu::SurfaceCapabilities {
        formats: formats.to_vec(),
        format_capabilities: Vec::new(),
        present_modes: modes.to_vec(),
        alpha_modes: alphas.to_vec(),
        usages,
    }
}

#[test]
fn a_surface_is_refused_when_any_choice_has_nothing_to_choose() {
    let target = wgpu::TextureUsages::RENDER_ATTACHMENT;
    let formats = [F::Rgba16Float, F::Bgra8Unorm];
    let modes = [Mode::Fifo, Mode::Immediate];
    let alphas = [Alpha::Inherit, Alpha::PreMultiplied];
    assert_eq!(
        surface_choice(&capabilities(&formats, &modes, &alphas, target)),
        Ok(SurfaceChoice {
            format: F::Bgra8Unorm,
            present_mode: Mode::Immediate,
            alpha_mode: Alpha::Inherit,
        })
    );
    let refused = Err(RenderError::Unsupported(SURFACE_UNSUPPORTED));
    // Each list empty or unusable on its own, and the usage missing on its own.
    for caps in [
        capabilities(&[], &modes, &alphas, target),
        capabilities(&formats, &[Mode::FifoRelaxed], &alphas, target),
        capabilities(&formats, &modes, &[Alpha::PostMultiplied], target),
        capabilities(&formats, &modes, &alphas, wgpu::TextureUsages::COPY_SRC),
        wgpu::SurfaceCapabilities::default(),
    ] {
        assert_eq!(surface_choice(&caps), refused, "{caps:?}");
    }
    // Written out: the message the frontend sees.
    assert_eq!(
        RenderError::Unsupported(SURFACE_UNSUPPORTED).to_string(),
        "the GPU adapter cannot present to this window"
    );
}

#[test]
fn the_frame_latency_is_wgpus_default() {
    // Written out: wgpu's `Surface::get_default_config` asks for 2.
    assert_eq!(FRAME_LATENCY, 2);
}

#[test]
fn view_and_write_formats_follow_the_presented_views_srgb() {
    assert_eq!(view_formats(F::Bgra8Unorm), vec![F::Bgra8UnormSrgb]);
    assert_eq!(view_formats(F::Bgra8UnormSrgb), vec![F::Bgra8Unorm]);
    assert_eq!(view_formats(F::Rgba8Unorm), vec![F::Rgba8UnormSrgb]);
    assert_eq!(view_formats(F::Rgba16Float), Vec::new());
    assert_eq!(write_format(F::Bgra8Unorm, true), F::Bgra8UnormSrgb);
    assert_eq!(write_format(F::Bgra8Unorm, false), F::Bgra8Unorm);
    assert_eq!(write_format(F::Bgra8UnormSrgb, false), F::Bgra8Unorm);
    assert_eq!(write_format(F::Bgra8UnormSrgb, true), F::Bgra8UnormSrgb);
    assert_eq!(write_format(F::Rgba16Float, true), F::Rgba16Float);
    assert_eq!(write_format(F::Rgba16Float, false), F::Rgba16Float);
}

#[test]
fn the_surface_size_at_its_edges() {
    let max = 16384;
    assert_eq!(surface_size((0, 0), max), None);
    assert_eq!(surface_size((0, 1), max), None);
    assert_eq!(surface_size((1, 0), max), None);
    assert_eq!(surface_size((1, 1), max), Some((1, 1)));
    assert_eq!(surface_size((max - 1, max), max), Some((max - 1, max)));
    assert_eq!(surface_size((max + 1, 7), max), Some((max, 7)));
    assert_eq!(surface_size((7, max + 1), max), Some((7, max)));
    assert_eq!(surface_size((u32::MAX, u32::MAX), max), Some((max, max)));
    assert_eq!(surface_size((640, 480), max), Some((640, 480)));
}

#[test]
fn shown_size_turns_on_quarter_turns() {
    let rect = Rect {
        x: 1,
        y: 2,
        width: 30,
        height: 40,
    };
    for (transform, expected) in [
        (PresentTransform::Identity, (30, 40)),
        (PresentTransform::FlipHorizontal, (30, 40)),
        (PresentTransform::FlipVertical, (30, 40)),
        (PresentTransform::Rotate90, (40, 30)),
        (PresentTransform::Rotate180, (30, 40)),
        (PresentTransform::Rotate270, (40, 30)),
    ] {
        assert_eq!(shown_size(rect, transform), expected, "{transform:?}");
    }
}

fn screen(x: u32, y: u32, width: u32, height: u32) -> ScreenRect {
    ScreenRect {
        x,
        y,
        width,
        height,
    }
}

#[test]
fn fit_letterboxes_and_pillarboxes() {
    // The same aspect ratio: the whole target.
    assert_eq!(
        fit((1280, 720), (1920, 1080)),
        Some(screen(0, 0, 1920, 1080))
    );
    assert_eq!(fit((4, 2), (4, 2)), Some(screen(0, 0, 4, 2)));
    // A narrower target: full width, bars above and below.
    assert_eq!(
        fit((1280, 720), (1024, 768)),
        Some(screen(0, 96, 1024, 576))
    );
    // A wider target: full height, bars left and right.
    assert_eq!(
        fit((1280, 720), (2000, 720)),
        Some(screen(360, 0, 1280, 720))
    );
    // Rounding half up: 3 * 1 / 2 = 1.5 rounds to 2, and an odd margin's extra pixel goes
    // after the image.
    assert_eq!(fit((2, 1), (3, 3)), Some(screen(0, 0, 3, 2)));
    assert_eq!(fit((1, 2), (3, 3)), Some(screen(0, 0, 2, 3)));
    // Below one half rounds down: 5 * 2 / 3 = 3.33.
    assert_eq!(fit((3, 2), (5, 5)), Some(screen(0, 1, 5, 3)));
    assert_eq!(fit((1, 1), (4, 3)), Some(screen(0, 0, 3, 3)));
    // At least one pixel, and no overflow at the extremes.
    assert_eq!(
        fit((u32::MAX, 1), (1, u32::MAX)),
        Some(screen(0, (u32::MAX - 1) / 2, 1, 1))
    );
    assert_eq!(
        fit((u32::MAX, u32::MAX), (u32::MAX, u32::MAX)),
        Some(screen(0, 0, u32::MAX, u32::MAX))
    );
    assert_eq!(
        fit((1, u32::MAX), (u32::MAX, 1)),
        Some(screen((u32::MAX - 1) / 2, 0, 1, 1))
    );
    // A zero side anywhere: nothing to show.
    for (shown, target) in [
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
        ((1, 1), (0, 1)),
        ((1, 1), (1, 0)),
    ] {
        assert_eq!(fit(shown, target), None, "{shown:?} in {target:?}");
    }
}

#[test]
fn the_filter_is_nearest_at_one_to_one_or_unfilterable() {
    let shown = (30, 20);
    for (rect, filterable, expected) in [
        (screen(5, 0, 30, 20), true, wgpu::FilterMode::Nearest),
        (screen(5, 0, 30, 20), false, wgpu::FilterMode::Nearest),
        (screen(0, 0, 31, 20), true, wgpu::FilterMode::Linear),
        (screen(0, 0, 30, 21), true, wgpu::FilterMode::Linear),
        (screen(0, 0, 60, 40), true, wgpu::FilterMode::Linear),
        (screen(0, 0, 60, 40), false, wgpu::FilterMode::Nearest),
        // The width and height exchanged is not one to one.
        (screen(0, 0, 20, 30), true, wgpu::FilterMode::Linear),
    ] {
        assert_eq!(filter(shown, rect, filterable), expected, "{rect:?}");
    }
}

fn source(width: u32, height: u32) -> TexelBox {
    TexelBox {
        texture: TextureId::new(0, 0),
        mip: 0,
        origin: Origin3d { x: 4, y: 8, z: 0 },
        extent: Extent3d {
            width,
            height,
            depth_or_layers: 1,
        },
    }
}

#[test]
fn the_crop_is_the_rectangle_within_its_rounded_box() {
    let rect = Rect {
        x: 4,
        y: 8,
        width: 3,
        height: 2,
    };
    // A rectangle of 3 by 2 in a box of 4 by 4 (whole 4x4 blocks).
    assert_eq!(
        Crop::new(rect, source(4, 4)),
        Crop {
            right: 0.75,
            bottom: 0.5
        }
    );
    assert_eq!(
        Crop::new(rect, source(3, 2)),
        Crop {
            right: 1.0,
            bottom: 1.0
        }
    );
    assert_eq!(
        Crop::new(rect, source(0, 0)),
        Crop {
            right: 0.0,
            bottom: 0.0
        }
    );
}

/// Which corner of the source each corner of the screen shows, from the geometry: the source
/// point (u, v) shown at screen point (x, y), both 0 or 1, for an image turned clockwise or
/// mirrored. Independent of `corners`' table.
fn model(transform: PresentTransform, x: f32, y: f32) -> [f32; 2] {
    match transform {
        PresentTransform::Identity => [x, y],
        PresentTransform::FlipHorizontal => [1.0 - x, y],
        PresentTransform::FlipVertical => [x, 1.0 - y],
        // Turned clockwise: the screen's top edge shows the source's left edge, bottom to top.
        PresentTransform::Rotate90 => [y, 1.0 - x],
        PresentTransform::Rotate180 => [1.0 - x, 1.0 - y],
        PresentTransform::Rotate270 => [1.0 - y, x],
    }
}

const TRANSFORMS: [PresentTransform; 6] = [
    PresentTransform::Identity,
    PresentTransform::FlipHorizontal,
    PresentTransform::FlipVertical,
    PresentTransform::Rotate90,
    PresentTransform::Rotate180,
    PresentTransform::Rotate270,
];

#[test]
fn each_transform_puts_each_source_corner_in_its_place() {
    let crop = Crop {
        right: 1.0,
        bottom: 1.0,
    };
    let screen_corners = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    for transform in TRANSFORMS {
        let got = corners(crop, transform);
        for (corner, [x, y]) in got.iter().zip(screen_corners) {
            assert_eq!(
                *corner,
                model(transform, x, y),
                "{transform:?} at ({x}, {y})"
            );
        }
    }
    // Two quarter turns are a half turn; a quarter turn each way is none.
    let compose = |a, b, x, y| {
        let [u, v] = model(b, x, y);
        model(a, u, v)
    };
    for [x, y] in screen_corners {
        assert_eq!(
            compose(PresentTransform::Rotate90, PresentTransform::Rotate90, x, y),
            model(PresentTransform::Rotate180, x, y)
        );
        assert_eq!(
            compose(
                PresentTransform::Rotate90,
                PresentTransform::Rotate270,
                x,
                y
            ),
            [x, y]
        );
    }
    // The crop scales the far edges only.
    let crop = Crop {
        right: 0.5,
        bottom: 0.25,
    };
    assert_eq!(
        corners(crop, PresentTransform::Rotate90),
        [[0.0, 0.25], [0.0, 0.0], [0.5, 0.25], [0.5, 0.0]]
    );
}

/// The `f32` at byte `at` of `block`.
fn word(block: &[u8; PARAMS_LEN], at: usize) -> f32 {
    f32::from_le_bytes(block[at..at + 4].try_into().expect("four bytes"))
}

#[test]
fn the_parameter_block_byte_for_byte() {
    // Written out: `present.wgsl`'s block is 128 bytes, the matrix at 0 (column after column),
    // the vertices from 64, 16 bytes apart, each a position then a texture coordinate.
    assert_eq!(PARAMS_LEN, 128);
    let block = blit_params(
        (4, 2),
        screen(1, 0, 2, 2),
        Crop {
            right: 0.5,
            bottom: 0.75,
        },
        PresentTransform::FlipHorizontal,
    );
    let matrix: Vec<f32> = (0..16).map(|i| word(&block, i * 4)).collect();
    assert_eq!(
        matrix,
        vec![
            0.5, 0.0, 0.0, 0.0, //
            0.0, -1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            -1.0, 1.0, 0.0, 1.0,
        ]
    );
    let vertices: Vec<[f32; 4]> = (0..4)
        .map(|i| {
            let at = 64 + i * 16;
            [
                word(&block, at),
                word(&block, at + 4),
                word(&block, at + 8),
                word(&block, at + 12),
            ]
        })
        .collect();
    assert_eq!(
        vertices,
        vec![
            [1.0, 0.0, 0.5, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 0.5, 0.75],
            [3.0, 2.0, 0.0, 0.75],
        ]
    );
}

#[test]
fn the_matrix_puts_pixel_zero_at_the_top_left_of_clip_space() {
    let block = blit_params(
        (512, 256),
        screen(0, 0, 512, 256),
        Crop {
            right: 1.0,
            bottom: 1.0,
        },
        PresentTransform::Identity,
    );
    let m: Vec<f32> = (0..16).map(|i| word(&block, i * 4)).collect();
    let clip = |x: f32, y: f32| {
        // Column-major: clip = column 0 * x + column 1 * y + column 3.
        [m[0] * x + m[4] * y + m[12], m[1] * x + m[5] * y + m[13]]
    };
    assert_eq!(clip(0.0, 0.0), [-1.0, 1.0], "top-left");
    assert_eq!(clip(512.0, 256.0), [1.0, -1.0], "bottom-right");
    assert_eq!(clip(256.0, 128.0), [0.0, 0.0], "centre");
}

#[test]
fn the_frame_queue_shows_only_the_newest() {
    for (last, newer, expected) in [
        (true, false, PresentOutcome::Shown),
        (true, true, PresentOutcome::Dropped),
        (false, false, PresentOutcome::Dropped),
        (false, true, PresentOutcome::Dropped),
    ] {
        assert_eq!(frame_outcome(last, newer), expected, "{last} {newer}");
    }
}

#[test]
fn the_action_for_every_acquire_answer() {
    use AcquireAction::{Drop, Forget, Retry, Show};
    for (acquired, retried, expected) in [
        (Acquired::Success, false, Show { reconfigure: false }),
        (Acquired::Success, true, Show { reconfigure: false }),
        (Acquired::Suboptimal, false, Show { reconfigure: true }),
        (Acquired::Suboptimal, true, Show { reconfigure: true }),
        (Acquired::Timeout, false, Drop),
        (Acquired::Timeout, true, Drop),
        (Acquired::Occluded, false, Drop),
        (Acquired::Occluded, true, Drop),
        (Acquired::Outdated, false, Retry),
        (Acquired::Outdated, true, Drop),
        (Acquired::Lost, false, Forget),
        (Acquired::Lost, true, Forget),
        (Acquired::Validation, false, Drop),
        (Acquired::Validation, true, Drop),
    ] {
        assert_eq!(
            acquire_action(acquired, retried),
            expected,
            "{acquired:?} {retried}"
        );
    }
}

#[test]
fn a_copy_starts_on_a_block_at_each_edge() {
    // Uncompressed: every origin.
    assert!(starts_on_block((0, 0), (1, 1)));
    assert!(starts_on_block((3, 7), (1, 1)));
    // 4x4 blocks: each axis on its own, one below, at and one above a block edge.
    for (origin, expected) in [
        ((0, 0), true),
        ((4, 8), true),
        ((3, 0), false),
        ((5, 0), false),
        ((0, 3), false),
        ((0, 5), false),
        ((4, 3), false),
        ((3, 4), false),
    ] {
        assert_eq!(starts_on_block(origin, (4, 4)), expected, "{origin:?}");
    }
    // Blocks that are not square (ASTC 8x5): the axes are not swapped.
    assert!(starts_on_block((8, 5), (8, 5)));
    assert!(!starts_on_block((5, 8), (8, 5)));
    // No zero-sized block is a block.
    assert!(!starts_on_block((0, 0), (0, 4)));
    assert!(!starts_on_block((0, 0), (4, 0)));
}

#[test]
fn a_surface_off_the_main_thread_only_where_it_may_be() {
    assert!(may_make_surface(false, false));
    assert!(may_make_surface(false, true));
    assert!(!may_make_surface(true, false));
    assert!(may_make_surface(true, true));
}
