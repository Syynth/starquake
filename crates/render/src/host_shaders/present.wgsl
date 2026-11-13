// The present blit: draws a presented view into the surface texture as a triangle strip of four
// vertices. The backend supplies each vertex's position and texture coordinate, which is how
// the image is scaled to the window, cropped to its source rectangle, and flipped or rotated.
//
// Ported from `eden/src/video_core/host_shaders/vulkan_present.vert` and `vulkan_present.frag`
// @ cf654052 (GPL-2.0-or-later, Copyright 2019 yuzu Emulator Project). Differences from Eden:
//
// - Eden's push-constant block is a uniform buffer: wgpu offers no push constants without an
//   optional feature. The block's members and their order are Eden's.
// - Eden's `GetVertex` picks a vertex with a chain of comparisons, because Vulkan allows a
//   push-constant array only dynamically uniform indices. A uniform buffer's array has no such
//   rule, so the vertex index is used directly.
// - Eden's combined `sampler2D` is a texture and a sampler: WGSL has no combined samplers.
// - The fragment stage samples mip level 0 explicitly, where Eden's `texture()` picks a level
//   from the coordinate's derivatives: a present reads the view's base mip level whatever the
//   scale (`crates/render/src/validate/view_ops.rs`), and Eden's presented images have one
//   level, so its result is the same.
//
// The interface (entry points, bindings, the block's layout) is named in
// `crates/render/src/host_shaders.rs`, and the tests beside it compare this file with those
// names.

struct ScreenRectVertex {
    position: vec2<f32>,
    tex_coord: vec2<f32>,
}

struct PresentParams {
    modelview_matrix: mat4x4<f32>,
    vertices: array<ScreenRectVertex, 4>,
}

@group(0) @binding(0) var<uniform> params: PresentParams;
@group(0) @binding(1) var color_texture: texture_2d<f32>;
@group(0) @binding(2) var color_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) frag_tex_coord: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex = params.vertices[vertex_index];
    var out: VertexOutput;
    out.position = params.modelview_matrix * vec4<f32>(vertex.position, 0.0, 1.0);
    out.frag_tex_coord = vertex.tex_coord;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(color_texture, color_sampler, in.frag_tex_coord, 0.0);
}
