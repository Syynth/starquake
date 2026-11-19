//! `emulation()` against the spec's table ("Fixed-function state the host lacks"), with
//! synthetic devices.
//!
//! The expected flags are written out here as rows, one small table per flag, from the spec's
//! table and the module docs; `every_combination` then checks that each flag follows its own
//! rows whatever the rest of the device and the guest state are.

use super::{DepthRange, EmulationState, ViewportFlip, emulation, host_viewport};
use crate::caps::{
    Caps, DynamicState, Features, FormatFeatures, FormatTable, GUEST_COLOR_TARGETS,
    GUEST_VERTEX_ATTRIBUTES, Limits, SampleCounts, StageSupport,
};
use crate::command::Viewport;
use crate::format::PixelFormat;
use crate::pipeline::EmulationFlags;
use crate::resource::CompareFunction;

/// A device with `features` and room for `max_vertex_attributes` attributes. Nothing else in
/// it is read by `emulation`; `nothing_else_in_caps_matters` checks that.
fn device(features: Features, max_vertex_attributes: u32) -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: 1 << 16,
            max_staging_size: 1 << 12,
            max_texture_dimension_1d: 256,
            max_texture_dimension_2d: 256,
            max_texture_dimension_3d: 64,
            max_texture_array_layers: 8,
            max_sampler_anisotropy: 16,
            max_vertex_attributes,
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
        features,
        stages: StageSupport::EMPTY,
        dynamic_state: DynamicState::EMPTY,
        sample_counts: SampleCounts::X1,
        view_swizzle: false,
        formats: FormatTable::uniform(FormatFeatures::EMPTY),
    }
}

/// Guest state that needs no emulation on any device.
const PLAIN: EmulationState = EmulationState {
    alpha_test: None,
    depth_range: DepthRange::ZeroToOne,
    primitive_restart: true,
    vertex_attributes: 0,
    color_targets: [None; GUEST_COLOR_TARGETS],
    viewport: ViewportFlip::NONE,
};

const fn targets(first: PixelFormat) -> [Option<PixelFormat>; GUEST_COLOR_TARGETS] {
    let mut targets = [None; GUEST_COLOR_TARGETS];
    targets[0] = Some(first);
    targets
}

/// Alpha test: no host has one, so every comparison but "always" becomes a discard.
const ALPHA_ROWS: [(Option<CompareFunction>, bool); 9] = [
    (None, false),
    (Some(CompareFunction::Always), false),
    (Some(CompareFunction::Never), true),
    (Some(CompareFunction::Less), true),
    (Some(CompareFunction::Equal), true),
    (Some(CompareFunction::LessEqual), true),
    (Some(CompareFunction::Greater), true),
    (Some(CompareFunction::NotEqual), true),
    (Some(CompareFunction::GreaterEqual), true),
];

/// Depth range: (guest range, device has the -1..1 switch, rewrite).
const DEPTH_ROWS: [(DepthRange, bool, bool); 4] = [
    (DepthRange::ZeroToOne, false, false),
    (DepthRange::ZeroToOne, true, false),
    (DepthRange::MinusOneToOne, false, true),
    (DepthRange::MinusOneToOne, true, false),
];

/// Primitive restart: (guest has restart on, device can turn it off, rewrite).
const RESTART_ROWS: [(bool, bool, bool); 4] = [
    (true, false, false),
    (true, true, false),
    (false, false, true),
    (false, true, false),
];

/// Vertex attributes: (device limit, attributes the guest uses, fetch). Each limit has a row
/// exactly at it and one past it.
const ATTRIBUTE_ROWS: [(u32, u32, bool); 13] = [
    (16, 0, false),
    (16, 15, false),
    (16, 16, false),
    (16, 17, true),
    (16, GUEST_VERTEX_ATTRIBUTES, true),
    (GUEST_VERTEX_ATTRIBUTES, 17, false),
    (GUEST_VERTEX_ATTRIBUTES, GUEST_VERTEX_ATTRIBUTES, false),
    (64, GUEST_VERTEX_ATTRIBUTES, false),
    (1, 1, false),
    (1, 2, true),
    (0, 0, false),
    (0, 1, true),
    (u32::MAX, u32::MAX, false),
];

/// Typed outputs: (colour targets, device takes untyped outputs, typed).
const OUTPUT_ROWS: [([Option<PixelFormat>; GUEST_COLOR_TARGETS], bool, bool); 12] = [
    ([None; GUEST_COLOR_TARGETS], false, false),
    ([None; GUEST_COLOR_TARGETS], true, false),
    (targets(PixelFormat::A8B8G8R8Unorm), false, false),
    (targets(PixelFormat::A8B8G8R8Unorm), true, false),
    (targets(PixelFormat::R16G16B16A16Float), false, false),
    (targets(PixelFormat::A8B8G8R8Snorm), false, false),
    (targets(PixelFormat::A8B8G8R8Srgb), false, false),
    (targets(PixelFormat::R8Uint), false, true),
    (targets(PixelFormat::R8Uint), true, false),
    (targets(PixelFormat::R32G32B32A32Sint), false, true),
    (targets(PixelFormat::R32G32B32A32Sint), true, false),
    (
        [
            Some(PixelFormat::A8B8G8R8Unorm),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(PixelFormat::R16Sint),
        ],
        false,
        true,
    ),
];

/// A flipped viewport: (flip, device sets it natively, flipped-viewport flag).
const VIEWPORT_ROWS: [(ViewportFlip, bool, bool); 8] = [
    (ViewportFlip::NONE, false, false),
    (ViewportFlip::NONE, true, false),
    (flip(true, false), false, true),
    (flip(true, false), true, false),
    (flip(false, true), false, true),
    (flip(false, true), true, false),
    (flip(true, true), false, true),
    (flip(true, true), true, false),
];

const fn flip(flip_y: bool, reverse_depth: bool) -> ViewportFlip {
    ViewportFlip {
        flip_y,
        reverse_depth,
    }
}

fn feature_if(on: bool, feature: Features) -> Features {
    if on { feature } else { Features::EMPTY }
}

fn flag_if(on: bool, flag: EmulationFlags) -> EmulationFlags {
    if on { flag } else { EmulationFlags::EMPTY }
}

/// The four device features `emulation` reads.
const READ: Features = Features::DEPTH_RANGE_MINUS_ONE_TO_ONE
    .union(Features::PRIMITIVE_RESTART_OFF)
    .union(Features::UNTYPED_FRAGMENT_OUTPUTS)
    .union(Features::FLIPPED_VIEWPORT);

#[test]
fn plain_state_needs_nothing_on_any_device() {
    for features in [Features::EMPTY, READ, Features::ALL] {
        for limit in [0, 16, GUEST_VERTEX_ATTRIBUTES] {
            assert_eq!(
                emulation(&device(features, limit), &PLAIN),
                EmulationFlags::EMPTY
            );
        }
    }
}

/// Each flag alone, from its rows, with the rest of the state plain.
#[test]
fn each_flag_follows_its_rows() {
    let bare = device(Features::EMPTY, GUEST_VERTEX_ATTRIBUTES);
    for (alpha_test, discard) in ALPHA_ROWS {
        let state = EmulationState {
            alpha_test,
            ..PLAIN
        };
        assert_eq!(
            emulation(&bare, &state),
            flag_if(discard, EmulationFlags::ALPHA_TEST),
            "{alpha_test:?}"
        );
    }
    for (depth_range, native, rewrite) in DEPTH_ROWS {
        let caps = device(
            feature_if(native, Features::DEPTH_RANGE_MINUS_ONE_TO_ONE),
            GUEST_VERTEX_ATTRIBUTES,
        );
        let state = EmulationState {
            depth_range,
            ..PLAIN
        };
        assert_eq!(
            emulation(&caps, &state),
            flag_if(rewrite, EmulationFlags::DEPTH_RANGE_REWRITE),
            "{depth_range:?} native {native}"
        );
    }
    for (primitive_restart, can_turn_off, rewrite) in RESTART_ROWS {
        let caps = device(
            feature_if(can_turn_off, Features::PRIMITIVE_RESTART_OFF),
            GUEST_VERTEX_ATTRIBUTES,
        );
        let state = EmulationState {
            primitive_restart,
            ..PLAIN
        };
        assert_eq!(
            emulation(&caps, &state),
            flag_if(rewrite, EmulationFlags::PRIMITIVE_RESTART_REWRITE),
            "restart {primitive_restart} can turn off {can_turn_off}"
        );
    }
    for (limit, vertex_attributes, fetch) in ATTRIBUTE_ROWS {
        let state = EmulationState {
            vertex_attributes,
            ..PLAIN
        };
        assert_eq!(
            emulation(&device(Features::EMPTY, limit), &state),
            flag_if(fetch, EmulationFlags::ATTRIBUTE_FETCH),
            "{vertex_attributes} attributes, limit {limit}"
        );
    }
    for (color_targets, untyped, typed) in OUTPUT_ROWS {
        let caps = device(
            feature_if(untyped, Features::UNTYPED_FRAGMENT_OUTPUTS),
            GUEST_VERTEX_ATTRIBUTES,
        );
        let state = EmulationState {
            color_targets,
            ..PLAIN
        };
        assert_eq!(
            emulation(&caps, &state),
            flag_if(typed, EmulationFlags::TYPED_OUTPUTS),
            "{color_targets:?} untyped {untyped}"
        );
    }
    for (viewport, native, flipped) in VIEWPORT_ROWS {
        let caps = device(
            feature_if(native, Features::FLIPPED_VIEWPORT),
            GUEST_VERTEX_ATTRIBUTES,
        );
        let state = EmulationState { viewport, ..PLAIN };
        assert_eq!(
            emulation(&caps, &state),
            flag_if(flipped, EmulationFlags::FLIPPED_VIEWPORT),
            "{viewport:?} native {native}"
        );
    }
}

/// Every row of every table with every row of the others: each flag depends only on its own
/// inputs. 9 x 4 x 4 x 13 x 12 x 8 = 179,712 device and state pairs.
#[test]
fn every_combination() {
    let mut checked = 0_u32;
    for (alpha_test, discard) in ALPHA_ROWS {
        for (depth_range, native_depth, depth_rewrite) in DEPTH_ROWS {
            for (primitive_restart, can_turn_off, restart_rewrite) in RESTART_ROWS {
                for (limit, vertex_attributes, fetch) in ATTRIBUTE_ROWS {
                    for (color_targets, untyped, typed) in OUTPUT_ROWS {
                        for (viewport, native_flip, flipped) in VIEWPORT_ROWS {
                            let features =
                                feature_if(native_depth, Features::DEPTH_RANGE_MINUS_ONE_TO_ONE)
                                    | feature_if(can_turn_off, Features::PRIMITIVE_RESTART_OFF)
                                    | feature_if(untyped, Features::UNTYPED_FRAGMENT_OUTPUTS)
                                    | feature_if(native_flip, Features::FLIPPED_VIEWPORT);
                            let state = EmulationState {
                                alpha_test,
                                depth_range,
                                primitive_restart,
                                vertex_attributes,
                                color_targets,
                                viewport,
                            };
                            let expected = flag_if(discard, EmulationFlags::ALPHA_TEST)
                                | flag_if(depth_rewrite, EmulationFlags::DEPTH_RANGE_REWRITE)
                                | flag_if(
                                    restart_rewrite,
                                    EmulationFlags::PRIMITIVE_RESTART_REWRITE,
                                )
                                | flag_if(fetch, EmulationFlags::ATTRIBUTE_FETCH)
                                | flag_if(typed, EmulationFlags::TYPED_OUTPUTS)
                                | flag_if(flipped, EmulationFlags::FLIPPED_VIEWPORT);
                            assert_eq!(
                                emulation(&device(features, limit), &state),
                                expected,
                                "{features:?} limit {limit} {state:?}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 9 * 4 * 4 * 13 * 12 * 8);
}

/// The rows above cover every flag: together they can ask for all of them, and for none.
#[test]
fn the_rows_reach_every_flag() {
    let all = flag_if(ALPHA_ROWS.iter().any(|r| r.1), EmulationFlags::ALPHA_TEST)
        | flag_if(
            DEPTH_ROWS.iter().any(|r| r.2),
            EmulationFlags::DEPTH_RANGE_REWRITE,
        )
        | flag_if(
            RESTART_ROWS.iter().any(|r| r.2),
            EmulationFlags::PRIMITIVE_RESTART_REWRITE,
        )
        | flag_if(
            ATTRIBUTE_ROWS.iter().any(|r| r.2),
            EmulationFlags::ATTRIBUTE_FETCH,
        )
        | flag_if(
            OUTPUT_ROWS.iter().any(|r| r.2),
            EmulationFlags::TYPED_OUTPUTS,
        )
        | flag_if(
            VIEWPORT_ROWS.iter().any(|r| r.2),
            EmulationFlags::FLIPPED_VIEWPORT,
        );
    assert_eq!(all, EmulationFlags::ALL);
}

/// Guest state that uses every feature a host may lack.
const DEMANDING: EmulationState = EmulationState {
    alpha_test: Some(CompareFunction::Greater),
    depth_range: DepthRange::MinusOneToOne,
    primitive_restart: false,
    vertex_attributes: 17,
    color_targets: targets(PixelFormat::R32Uint),
    viewport: flip(true, true),
};

/// A device shaped like wgpu's, from the spec's decision 1 table: no depth-range switch,
/// restart that cannot be turned off, typed outputs (a guess the spec leaves to slice R7), and
/// no flipped viewport (wgpu-core's `set_viewport`).
/// Its attribute limit here is wgpu's default of 16 (`wgpu-types` 30.0.2, `limits.rs`); the
/// value on a real Mac is the spec's verification 1.
#[test]
fn a_wgpu_shaped_device_emulates_everything() {
    let wgpu = device(Features::ALL.difference(READ), 16);
    assert_eq!(emulation(&wgpu, &DEMANDING), EmulationFlags::ALL);
    assert_eq!(emulation(&wgpu, &PLAIN), EmulationFlags::EMPTY);
    // Sixteen attributes fit: everything but the fetch.
    let sixteen = EmulationState {
        vertex_attributes: 16,
        ..DEMANDING
    };
    assert_eq!(
        emulation(&wgpu, &sixteen),
        EmulationFlags::ALL.difference(EmulationFlags::ATTRIBUTE_FETCH)
    );
}

/// A device with every switch (a direct Vulkan backend on a desktop driver) still has no alpha
/// test: that is the one thing emulated on it.
#[test]
fn a_device_with_every_switch_emulates_only_the_alpha_test() {
    let vulkan = device(Features::ALL, GUEST_VERTEX_ATTRIBUTES);
    assert_eq!(emulation(&vulkan, &DEMANDING), EmulationFlags::ALPHA_TEST);
}

/// An integer target in any of the guest's eight slots asks for typed outputs.
#[test]
fn an_integer_target_in_any_slot_is_seen() {
    let bare = device(Features::EMPTY, GUEST_VERTEX_ATTRIBUTES);
    for slot in 0..GUEST_COLOR_TARGETS {
        let mut color_targets = [Some(PixelFormat::A8B8G8R8Unorm); GUEST_COLOR_TARGETS];
        color_targets[slot] = Some(PixelFormat::A2B10G10R10Uint);
        let state = EmulationState {
            color_targets,
            ..PLAIN
        };
        assert_eq!(
            emulation(&bare, &state),
            EmulationFlags::TYPED_OUTPUTS,
            "slot {slot}"
        );
    }
}

/// Only the three features and the attribute limit are read: every other part of `Caps` set to
/// its other extreme changes nothing.
#[test]
fn nothing_else_in_caps_matters() {
    for features in [Features::EMPTY, READ] {
        let lean = device(features, 16);
        let mut rich = device(features | Features::ALL.difference(READ), 16);
        rich.stages = StageSupport::ALL;
        rich.dynamic_state = DynamicState::ALL;
        rich.sample_counts = SampleCounts::ALL;
        rich.view_swizzle = true;
        rich.formats = FormatTable::uniform(FormatFeatures::ALL);
        rich.limits.max_vertex_buffers = 32;
        rich.limits.max_color_attachments = 1;
        for state in [PLAIN, DEMANDING] {
            assert_eq!(emulation(&lean, &state), emulation(&rich, &state));
        }
    }
}

/// A viewport at `(x, y)` of `width` by `height`, with depths `near` and `far`.
fn vp(x: f32, y: f32, width: f32, height: f32, near: f32, far: f32) -> Viewport {
    Viewport {
        x,
        y,
        width,
        height,
        min_depth: near,
        max_depth: far,
    }
}

/// Each height sign and depth order, with the edges that are not flips: a zero or `-0.0`
/// height, equal depths, and NaN in either.
#[test]
fn each_viewport_sign_and_depth_order_is_its_flip() {
    let nan = f32::NAN;
    let rows = [
        (vp(0.0, 0.0, 8.0, 8.0, 0.0, 1.0), flip(false, false)),
        (vp(0.0, 8.0, 8.0, -8.0, 0.0, 1.0), flip(true, false)),
        (vp(0.0, 0.0, 8.0, 8.0, 1.0, 0.0), flip(false, true)),
        (vp(0.0, 8.0, 8.0, -8.0, 0.75, 0.25), flip(true, true)),
        // The edges on each side: a height just below zero flips, zero and -0.0 do not.
        (
            vp(0.0, 0.0, 8.0, -f32::MIN_POSITIVE, 0.0, 1.0),
            flip(true, false),
        ),
        (vp(0.0, 0.0, 8.0, 0.0, 0.0, 1.0), flip(false, false)),
        (vp(0.0, 0.0, 8.0, -0.0, 0.0, 1.0), flip(false, false)),
        (vp(0.0, 0.0, 8.0, nan, 0.0, 1.0), flip(false, false)),
        // Depths one step apart in each order, equal, and NaN.
        (vp(0.0, 0.0, 8.0, 8.0, 0.5, 0.5), flip(false, false)),
        (
            vp(0.0, 0.0, 8.0, 8.0, 0.5, 0.5_f32.next_up()),
            flip(false, false),
        ),
        (
            vp(0.0, 0.0, 8.0, 8.0, 0.5_f32.next_up(), 0.5),
            flip(false, true),
        ),
        (vp(0.0, 0.0, 8.0, 8.0, nan, 0.5), flip(false, false)),
        (vp(0.0, 0.0, 8.0, 8.0, 0.5, nan), flip(false, false)),
        // A negative width is not a flip of either kind.
        (vp(8.0, 0.0, -8.0, 8.0, 0.0, 1.0), flip(false, false)),
    ];
    for (viewport, expected) in rows {
        assert_eq!(ViewportFlip::of(&viewport), expected, "{viewport:?}");
        assert_eq!(
            expected.any(),
            expected != ViewportFlip::NONE,
            "{viewport:?}"
        );
    }
    assert_eq!(ViewportFlip::default(), ViewportFlip::NONE);
    assert!(!ViewportFlip::NONE.any());
}

/// The host viewport of each case: positive, over the same rows, with the depths in order; a
/// viewport with no flip, a negative width included, is left as it is.
#[test]
fn the_host_viewport_is_positive_and_in_order() {
    let rows = [
        (
            vp(1.0, 2.0, 8.0, 6.0, 0.25, 0.75),
            vp(1.0, 2.0, 8.0, 6.0, 0.25, 0.75),
        ),
        (
            vp(1.0, 8.0, 8.0, -6.0, 0.25, 0.75),
            vp(1.0, 2.0, 8.0, 6.0, 0.25, 0.75),
        ),
        (
            vp(1.0, 2.0, 8.0, 6.0, 0.75, 0.25),
            vp(1.0, 2.0, 8.0, 6.0, 0.25, 0.75),
        ),
        (
            vp(1.0, 8.0, 8.0, -6.0, 1.0, 0.0),
            vp(1.0, 2.0, 8.0, 6.0, 0.0, 1.0),
        ),
        (
            vp(9.0, 2.0, -8.0, 6.0, 0.25, 0.75),
            vp(9.0, 2.0, -8.0, 6.0, 0.25, 0.75),
        ),
        // A flip whose top lands above the target keeps its arithmetic.
        (
            vp(0.0, 4.0, 8.0, -6.0, 0.5, 0.5),
            vp(0.0, -2.0, 8.0, 6.0, 0.5, 0.5),
        ),
    ];
    for (guest, host) in rows {
        assert_eq!(host_viewport(&guest), host, "{guest:?}");
        assert_eq!(ViewportFlip::of(&host), ViewportFlip::NONE, "{guest:?}");
    }
    // `y + height` past `f32`'s range is infinite, for the backend's check to refuse.
    let far = host_viewport(&vp(0.0, -f32::MAX, 8.0, -f32::MAX, 0.0, 1.0));
    assert_eq!(far.y, f32::NEG_INFINITY);
    assert_eq!(far.height, f32::MAX);
}

/// Window y of clip-space `y_ndc` through `viewport`, for a host whose y axis has sign `s`.
fn window_y(viewport: &Viewport, y_ndc: f32, s: f32) -> f32 {
    viewport.y + (1.0 + s * y_ndc) / 2.0 * viewport.height
}

/// Depth of clip-space `z` at `w` through `viewport`.
fn window_depth(viewport: &Viewport, z: f32, w: f32) -> f32 {
    viewport.min_depth + z / w * (viewport.max_depth - viewport.min_depth)
}

/// The module docs' claim, worked through: for each flip, the host viewport and the shader's
/// change (y negated, z made `w - z`) put every clip-space point on the guest's window y and
/// depth, for either sign of the host's y axis, and clip z where the guest clips it. Every value
/// is a short binary fraction, so the comparisons are exact.
#[test]
fn a_flipped_viewport_lands_on_the_guests_pixels_and_depths() {
    let guests = [
        vp(0.0, 0.0, 8.0, 8.0, 0.0, 1.0),
        vp(0.0, 8.0, 8.0, -8.0, 0.0, 1.0),
        vp(2.0, 6.0, 4.0, -4.0, 0.75, 0.25),
        vp(0.0, 0.0, 8.0, 8.0, 1.0, 0.0),
    ];
    let w = 2.0;
    for guest in guests {
        let flip = ViewportFlip::of(&guest);
        let host = host_viewport(&guest);
        for y_ndc in [-1.0, -0.5, 0.0, 0.25, 1.0] {
            let shaded = if flip.flip_y { -y_ndc } else { y_ndc };
            for s in [1.0, -1.0] {
                assert_eq!(
                    window_y(&host, shaded, s),
                    window_y(&guest, y_ndc, s),
                    "{guest:?} y {y_ndc} sign {s}"
                );
            }
        }
        for z in [-0.5, 0.0, 0.5, 1.5, 2.0, 2.5] {
            let shaded = if flip.reverse_depth { w - z } else { z };
            let clipped = |z: f32| !(0.0..=w).contains(&z);
            assert_eq!(clipped(shaded), clipped(z), "{guest:?} z {z}");
            if !clipped(z) {
                assert_eq!(
                    window_depth(&host, shaded, w),
                    window_depth(&guest, z, w),
                    "{guest:?} z {z}"
                );
            }
        }
    }
}
