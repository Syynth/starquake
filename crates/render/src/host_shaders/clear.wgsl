// Clears drawn as one triangle that covers the whole viewport, for the clears a render pass's
// load action cannot express (a clear within a rectangle: the draw's scissor bounds it).
//
// Ported from `eden/src/video_core/host_shaders/vulkan_color_clear.vert`,
// `vulkan_color_clear.frag` and `vulkan_depthstencil_clear.frag` @ cf654052 (GPL-2.0-or-later,
// Copyright 2023 yuzu Emulator Project). The vertex positions are those of
// `full_screen_triangle.vert` (Copyright 2020 yuzu Emulator Project), which Eden's clear vertex
// shader repeats without the texture coordinate. Differences from Eden:
//
// - Eden's push-constant blocks are one uniform buffer: wgpu offers no push constants without
//   an optional feature.
// - The value is four 32-bit patterns, and each fragment entry point reinterprets them as its
//   own type. Eden's colour clear writes floats only; a WGSL fragment output's type must match
//   its target's (float, signed integer or unsigned integer), so there is one entry point per
//   type, and reinterpreting the bits rather than converting the numbers keeps an integer
//   value exact.
// - As in Eden, nothing here is specific to stencil: a stencil clear is the pipeline's stencil
//   operation writing the reference value. On a target with depth it is drawn with `fs_depth`
//   and depth writes off. A stencil-only target has no fragment stage it can use (wgpu refuses
//   one that writes the fragment's depth when the target has no depth), and needs none: the
//   pipeline is made with the vertex stage alone.
//
// The interface (entry points, the binding, the block's layout) is named in
// `crates/render/src/host_shaders.rs`, and the tests beside it compare this file with those
// names.

struct ClearParams {
    value: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: ClearParams;

// Vertices 0, 1 and 2 are at (-1, -1), (3, -1) and (-1, 3): the triangle contains the whole of
// clip space's -1..1 square.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vertex_index & 1u) << 2u);
    let y = f32((vertex_index & 2u) << 1u);
    return vec4<f32>(x - 1.0, y - 1.0, 0.0, 1.0);
}

@fragment
fn fs_color_float() -> @location(0) vec4<f32> {
    return bitcast<vec4<f32>>(params.value);
}

@fragment
fn fs_color_sint() -> @location(0) vec4<i32> {
    return bitcast<vec4<i32>>(params.value);
}

@fragment
fn fs_color_uint() -> @location(0) vec4<u32> {
    return params.value;
}

@fragment
fn fs_depth() -> @builtin(frag_depth) f32 {
    return bitcast<f32>(params.value.x);
}
