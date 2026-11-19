// The index rewrite: turns a draw's vertices or indices into the 32-bit indices of a topology
// wgpu has (render-backend, "Fixed-function state the host lacks"): quads and quad strips into
// triangle lists, triangle fans and polygons into triangle lists, line loops into closed line
// strips, and 8-bit or 16-bit indices into 32-bit ones. One invocation writes one output index
// at a time, in a loop over the whole output, so any count fits one dispatch.
//
// Ported from `eden/src/video_core/host_shaders/vulkan_quad_indexed.comp` @ cf654052
// (GPL-2.0-or-later, Copyright 2020 yuzu Emulator Project): the quad and quad-strip orders, the
// packed read of 8-bit and 16-bit indices from 32-bit words, and the input and output buffers.
// Differences from Eden:
//
// - Eden's push-constant block is a uniform buffer: wgpu offers no push constants without an
//   optional feature. Eden's block has the base vertex, the index size and the strip flag; this
//   one has the output count, the mode, the source, the first index's byte offset, the input
//   count and the restart flag (`crates/render-wgpu/src/draw/rewrite.rs` names the layout).
// - Eden adds the base vertex to each index; here the draw keeps its own base vertex, so the
//   rewritten indices are the input's.
// - Eden writes six indices per invocation and stops at the output's length; here each
//   invocation loops over output indices, `WORKGROUP_SIZE` times the workgroup count apart, so
//   the dispatch's size does not limit the count.
// - Eden rewrites quads and quad strips from an index buffer only (it draws non-indexed quads
//   from a prepared buffer, `renderer_vulkan/vk_buffer_cache.cpp`, `QuadArrayIndexBuffer`, with
//   the same orders); here a non-indexed draw is the sequential source, the first vertex plus
//   the position. Fans, polygons, line loops and the widening of small indices (Eden's
//   `vulkan_uint8.comp` widens 8-bit indices to 16 bits) are new.

// How output index k picks its input position (`rewrite::Mode`).
const MODE_LIST: u32 = 0u;
const MODE_QUADS: u32 = 1u;
const MODE_QUAD_STRIP: u32 = 2u;
const MODE_FAN: u32 = 3u;
const MODE_LOOP: u32 = 4u;

// Where the value at an input position comes from (`rewrite::Source`).
const SOURCE_SEQUENTIAL: u32 = 0u;
const SOURCE_U8: u32 = 1u;
const SOURCE_U16: u32 = 2u;
const SOURCE_U32: u32 = 3u;

// The 32-bit restart value an input equal to its own format's restart value becomes.
const RESTART: u32 = 0xffffffffu;

// Invocations per workgroup.
const WORKGROUP_SIZE: u32 = 64u;

struct Params {
    // Output indices to write.
    count: u32,
    // One of the MODE_ values.
    mode: u32,
    // One of the SOURCE_ values.
    source: u32,
    // SOURCE_SEQUENTIAL: the first vertex. Otherwise: the byte offset of the first index in
    // `input`.
    first: u32,
    // Input positions: the draw's vertex or index count.
    input_count: u32,
    // Not 0: an input equal to its format's restart value is written as RESTART.
    restart: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

// The input position output index `k` reads.
fn position(k: u32) -> u32 {
    // Eden's `quads_swizzle` and `quad_strip_swizzle`.
    var quads = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
    var quad_strip = array<u32, 6>(0u, 3u, 1u, 0u, 2u, 3u);
    switch params.mode {
        case MODE_QUADS: {
            return (k / 6u) * 4u + quads[k % 6u];
        }
        case MODE_QUAD_STRIP: {
            return (k / 6u) * 2u + quad_strip[k % 6u];
        }
        case MODE_FAN: {
            // Triangle t is (t + 1, t + 2, 0): Vulkan's fan order, which keeps its first
            // (provoking) vertex.
            let corner = k % 3u;
            if corner == 2u {
                return 0u;
            }
            return k / 3u + corner + 1u;
        }
        case MODE_LOOP: {
            // The strip ends where it began.
            if k >= params.input_count {
                return 0u;
            }
            return k;
        }
        default: {
            return k;
        }
    }
}

// The index at input position `p`.
fn value(p: u32) -> u32 {
    if params.source == SOURCE_SEQUENTIAL {
        return params.first + p;
    }
    // 0, 1 or 2 for 8-bit, 16-bit or 32-bit indices.
    let shift = params.source - SOURCE_U8;
    let bits = 8u << shift;
    let byte = params.first + (p << shift);
    let raw = extractBits(input[byte / 4u], (byte % 4u) * 8u, bits);
    if params.restart != 0u && raw == (RESTART >> (32u - bits)) {
        return RESTART;
    }
    return raw;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = groups.x * WORKGROUP_SIZE;
    for (var k = id.x; k < params.count; k += stride) {
        output[k] = value(position(k));
    }
}
