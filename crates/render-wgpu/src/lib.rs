//! The wgpu render backend (docs/specs/render-backend.md, slice R5): the `Backend` trait on a
//! wgpu device, and the only crate that names a wgpu type (decision 10).
//!
//! [`WgpuBackend::new`] opens the host's default adapter (Metal on macOS, Vulkan on Linux),
//! builds `Caps` from what it reports ([`caps`]), with `view_swizzle` off, and starts the render
//! thread. `core` builds it with [`WgpuBackend::with_compile_pool`]: a [`WgpuConfig`], the GPU
//! crate's `CompletionWake`, the shared compile pool and the wait-while-compiling setting; and
//! hands the GPU crate an `Arc<dyn Backend>`; nothing downcasts it, so a capture wrapper can sit
//! in front (render-backend, "Render capture v2").
//!
//! What this slice does: buffers, textures, views and samplers, each checked against `Caps` and
//! against what wgpu refuses beyond the shared checks ([`rules`]); uploads through
//! `Queue::write_buffer` and `Queue::write_texture` from staging memory on the CPU side; copies,
//! clears and readbacks (mapped after the list completes); `destroy` deferred by fence value;
//! fences from wgpu's completion reports; the render thread with its bounded queue; the
//! offscreen present target, which counts presents and reads dumped frames back; and the
//! completion wake. What it refuses until a later slice, with a named error: blits
//! ([`plan::BLITS_NOT_YET`]; no slice builds the host shaders' blit pipelines into this backend
//! yet).
//!
//! Slice R6 adds shaders and pipelines, in [`pipeline`]: shaders kept as naga modules and handed
//! to wgpu (`ShaderSource::Naga`) by the first pipeline build that needs them; bind group and
//! pipeline layouts from each stage's `ShaderInterface`, with binding arrays given their size;
//! graphics and compute pipelines built on the compile pool, their status `Compiling` until the
//! build settles; failed pipelines cached as failed; and [`WgpuBackend::draw_pipeline`], which
//! applies the wait-while-compiling setting to a draw (wait, or skip and count) for the render
//! thread's encoder (slice R7). A pipeline whose shaders need subgroups of 32 is refused on a
//! device that cannot promise them; an Apple GPU under Metal counts as promising them, although
//! wgpu reports a range of 4 to 64 there ([`SubgroupSupport`], decided by the owner, 2026-11-19).
//! A pipeline whose vertex shader uses subgroups is also refused on a device without subgroups in
//! vertex shaders, which includes every Apple GPU under wgpu 30 ([`pipeline::VERTEX_SUBGROUPS`]).
//! No driver cache blob is made: wgpu's is Vulkan only and needs an `unsafe` call ([`pipeline`],
//! "Not here").
//!
//! Slice R7 adds draws and dispatches, in [`draw`]: render passes opened lazily at the first draw
//! after new targets and closed by any other command; bind groups from each draw's bindings,
//! checked against its pipeline's layout; vertex and index buffers; the viewport, scissor, blend
//! constant and stencil reference per draw; direct and indirect draws and dispatches; and the
//! index rewrite, a WGSL compute pass ([`draw::rewrite`], after Eden's
//! `vulkan_quad_indexed.comp`, validated with naga), for quads, quad strips, fans, polygons, line
//! loops, 8-bit indices and 16-bit strip indices. A strip pipeline takes 32-bit indices (R6 left
//! its index format unset, which wgpu refuses for every indexed strip draw;
//! [`pipeline::state::STRIP_INDEX_FORMAT`]). wgpu checks a fragment output's type against an
//! integer target, so typed outputs stay emulated (`draw`, "Typed outputs").
//!
//! Slice R8 adds window presentation (`window`, with its choices as pure functions in
//! [`present`]): a wgpu surface made from the frontend's window, on the calling thread (on
//! macOS the main thread, or [`NOT_MAIN_THREAD`]); a present mode that does not wait for the
//! display while a newer frame is ready; the newest frame winning, the others counted as dropped
//! in `PresentFeedback`; and the present blit, drawn into the window where the present stands in
//! its list, so a later write to the image cannot change the frame shown (the ordering test,
//! `backend::tests::window`). Surface creation needs no `unsafe` code: the window is handed over
//! as an owned, shared object, which wgpu's safe constructor takes (verification 11;
//! `starquake_render::window`).
//!
//! Where this backend refuses what the recording backend accepts, each a
//! `RenderError::Unsupported` whose message names the rule (except
//! [`pipeline::NO_ENTRY_POINT`], a `RenderError::Invalid`): the transfer and resource rules are
//! listed in [`rules`] with their wgpu sources, and `rules::tests` pins each; the shader and
//! pipeline rules are in [`pipeline`] and its `layout` and `state` modules:
//!
//! - an upload or buffer-to-buffer copy whose buffer range is not 4-byte aligned, and a buffer
//!   copy within one buffer;
//! - a texture copy between formats that differ by more than sRGB; within one mip level of one
//!   texture, a 1D, 2D or cube copy whose layer ranges overlap and a 3D copy whose regions start
//!   at the same slice; and a copy covering part of the width or height of a depth, stencil or
//!   multisampled level;
//! - an upload, buffer copy or readback of two-aspect depth-stencil texels; an upload or buffer
//!   copy into `D32_FLOAT`; any of them for depth or stencil texels on a device without
//!   `DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES`, covering part of the level's width or
//!   height, or with a buffer side off a 4-byte boundary; and a copy between a buffer and a
//!   depth or stencil texture of several rows that are not a multiple of 256 bytes;
//! - a buffer side of a colour texture copy that does not start on a texel block;
//! - a texture readback whose rows, padded to 256 bytes, would not fit the largest buffer (a
//!   dumped frame in that case is not refused: its pixels are reported unknown);
//! - a partial clear of a depth, stencil or multisampled view;
//! - the texture shapes, view formats and border colours wgpu cannot make;
//! - from slice R6, shaders and pipelines ([`pipeline`] gives each rule's reason and the order of
//!   the checks): a shader that is not a naga module ([`pipeline::NAGA_ONLY`]), whose module has no
//!   entry point of its name for its stage ([`pipeline::NO_ENTRY_POINT`]), or whose blocks nest
//!   deeper than [`pipeline::MAX_SHADER_NESTING`] ([`pipeline::NESTING`]); a pipeline asked of a
//!   backend without a compile pool ([`pipeline::NO_COMPILE_POOL`]), or with a shader that needs
//!   subgroups of 32 on a device that cannot promise them ([`pipeline::SUBGROUP_SIZE`]), or with a
//!   vertex shader that uses subgroups on a device without them in vertex shaders
//!   ([`pipeline::VERTEX_SUBGROUPS`]); layouts
//!   where two stages bind one group and binding as different resources
//!   ([`pipeline::layout::STAGE_CONFLICT`]), with a binding array whose size is not a constant
//!   ([`pipeline::layout::UNSIZED_BINDING_ARRAY`]), with a 1D array texture
//!   ([`pipeline::layout::ONE_D_ARRAY`]) or with a storage texture whose format has no wgpu form
//!   ([`pipeline::layout::STORAGE_FORMAT`]); and fixed-function state with a vertex attribute wgpu
//!   has no format for ([`pipeline::state::VERTEX_FORMAT`]), a topology with adjacency
//!   ([`pipeline::state::ADJACENCY`]), culling of both faces ([`pipeline::state::CULL_BOTH`]), a
//!   target format with no wgpu form ([`pipeline::state::TARGET_FORMAT`]), or colour targets
//!   without a fragment shader ([`pipeline::state::TARGETS_WITHOUT_FRAGMENT`]);
//! - from slice R7, draws and dispatches that break a rule wgpu applies as it encodes them, in
//!   the order and with the names [`draw`] lists ("The checks"): targets wgpu cannot render into
//!   together or that the pipeline was not made for, bindings that do not fit the pipeline's
//!   layout (a binding array among them), counts past their buffers, viewports wgpu cannot set,
//!   a resource written and otherwise used by one draw, an indirect draw that needs its indices
//!   rewritten, and more indirect draws in one command than [`draw::MAX_INDIRECT_DRAWS`].
//!
//! The differential tests in `backend::tests` run the same lists on this backend and the
//! recording backend (hand-built lists, and 40 lists drawn from a fixed seed, of uploads,
//! copies, clears, readbacks and presents with frame dumps) and require identical bytes for
//! every readback and frame the recording backend knows. The draw tests
//! (`backend::tests::draws`) draw scenes whose pixels can be computed, axis-aligned rectangles
//! on pixel edges in flat colours, and compare them exactly (decision 9).
//!
//! The pipeline tests (`pipeline::tests`) build hand-written WGSL modules, parsed into naga IR,
//! into pipelines, among them a binding array; scripted compile delays (a pool that holds its
//! jobs until the test releases them) exercise both settings and the skip count.
//!
//! The GPU tests run on a real adapter. They skip themselves, with a notice, where the host has
//! none, and in CI (the `CI` environment variable): they stay local-only until verification 3
//! shows wgpu runs cleanly on lavapipe in Linux CI (decision 9; `CLAUDE.md`, "Tests").

mod adapter;
mod backend;
pub mod caps;
mod config;
mod convert;
pub mod draw;
#[cfg(test)]
mod fixtures;
pub mod format;
pub mod pipeline;
pub mod plan;
pub mod present;
pub mod rules;
mod thread;
mod window;

pub use adapter::AdapterReport;
pub use backend::{NewError, OPEN_TIMEOUT, WgpuBackend};
pub use config::{WgpuConfig, queue_full};
pub use pipeline::{
    APPLE_GPU_NAME_PREFIX, DrawAction, DrawPipeline, PipelineCounts, ReadyPipeline, SkipReason,
    SubgroupSupport, apple_gpu_family, on_draw,
};
pub use present::SURFACE_UNSUPPORTED;
pub use window::{NO_SURFACE, NOT_MAIN_THREAD};
