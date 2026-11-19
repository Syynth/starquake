//! The wgpu render backend (docs/specs/render-backend.md, slice R5): the `Backend` trait on a
//! wgpu device, and the only crate that names a wgpu type (decision 10).
//!
//! [`WgpuBackend::new`] opens the host's default adapter (Metal on macOS, Vulkan on Linux),
//! builds `Caps` from what it reports ([`caps`]), with `view_swizzle` off, and starts the render
//! thread. `core` builds it with a [`WgpuConfig`] and the GPU crate's `CompletionWake`, and
//! hands the GPU crate an `Arc<dyn Backend>`; nothing downcasts it, so a capture wrapper can sit
//! in front (render-backend, "Render capture v2").
//!
//! What this slice does: buffers, textures, views and samplers, each checked against `Caps` and
//! against what wgpu refuses beyond the shared checks ([`rules`]); uploads through
//! `Queue::write_buffer` and `Queue::write_texture` from staging memory on the CPU side; copies,
//! clears and readbacks (mapped after the list completes); `destroy` deferred by fence value;
//! fences from wgpu's completion reports; the render thread with its bounded queue; the
//! offscreen present target, which counts presents and reads dumped frames back; and the
//! completion wake. What it refuses until later slices, each with a named error: shaders and
//! pipelines (R6, [`SHADERS_NOT_YET`], [`PIPELINES_NOT_YET`]), draws and dispatches (R7,
//! [`plan::DRAWS_NOT_YET`]), blits ([`plan::BLITS_NOT_YET`]), and window targets (R8).
//!
//! Where this backend refuses what the recording backend accepts, each a
//! `RenderError::Unsupported` whose message names the rule ([`rules`] lists them with their
//! wgpu sources, and `rules::tests` pins each):
//!
//! - an upload or buffer-to-buffer copy whose buffer range is not 4-byte aligned, and a buffer
//!   copy within one buffer;
//! - a texture copy within one mip level of one texture whose layer ranges overlap, between
//!   formats that differ by more than sRGB, or covering part of a depth, stencil or
//!   multisampled level;
//! - an upload, buffer copy or readback of depth or stencil texels, and a buffer side of a
//!   texture copy that does not start on a texel block;
//! - a partial clear of a depth, stencil or multisampled view;
//! - the texture shapes, view formats and border colours wgpu cannot make.
//!
//! The differential tests in `backend::tests` run the same lists on this backend and the
//! recording backend (hand-built lists, and 40 lists drawn from a fixed seed, of uploads,
//! copies, clears, readbacks and presents with frame dumps) and require identical bytes for
//! every readback and frame the recording backend knows.
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
pub mod plan;
pub mod rules;
mod thread;

pub use adapter::AdapterReport;
pub use backend::{NewError, OPEN_TIMEOUT, PIPELINES_NOT_YET, SHADERS_NOT_YET, WgpuBackend};
pub use config::{WgpuConfig, queue_full};
