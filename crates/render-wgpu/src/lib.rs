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
//! completion wake. What it refuses until later slices, each with a named error: draws and
//! dispatches (R7, [`plan::DRAWS_NOT_YET`]) and blits ([`plan::BLITS_NOT_YET`]).
//!
//! Slice R6 adds shaders and pipelines, in [`pipeline`]: shaders kept as naga modules and handed
//! to wgpu (`ShaderSource::Naga`) by the first pipeline build that needs them; bind group and
//! pipeline layouts from each stage's `ShaderInterface`, with binding arrays given their size;
//! graphics and compute pipelines built on the compile pool, their status `Compiling` until the
//! build settles; failed pipelines cached as failed; and [`WgpuBackend::draw_pipeline`], which
//! applies the wait-while-compiling setting to a draw (wait, or skip and count) for the render
//! thread's encoder (slice R7). A pipeline whose shaders need subgroups of 32 is refused on a
//! device that cannot promise them, which is every Mac under wgpu (an owner question). No driver
//! cache blob is made: wgpu's is Vulkan only and needs an `unsafe` call ([`pipeline`], "Not
//! here").
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
//! `RenderError::Unsupported` whose message names the rule ([`rules`] lists them with their
//! wgpu sources, and `rules::tests` pins each):
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
//! - the texture shapes, view formats and border colours wgpu cannot make.
//!
//! The differential tests in `backend::tests` run the same lists on this backend and the
//! recording backend (hand-built lists, and 40 lists drawn from a fixed seed, of uploads,
//! copies, clears, readbacks and presents with frame dumps) and require identical bytes for
//! every readback and frame the recording backend knows.
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
    DrawAction, DrawPipeline, PipelineCounts, ReadyPipeline, SkipReason, on_draw,
    subgroups_honoured,
};
pub use present::SURFACE_UNSUPPORTED;
pub use window::{NO_SURFACE, NOT_MAIN_THREAD};
