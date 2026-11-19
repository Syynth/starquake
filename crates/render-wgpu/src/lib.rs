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
//! Where this backend refuses what the recording backend accepts (each is a
//! `RenderError::Unsupported` naming the rule, listed in [`rules`] and pinned in its tests):
//!
//! - buffer ranges an upload or a buffer-to-buffer copy writes or reads that are not 4-byte
//!   aligned;
//! - buffer copies, uploads and readbacks of depth or stencil texels;
//! - the texture shapes, view formats, partial clears, partial depth or multisampled copies and
//!   border colours wgpu cannot make.
//!
//! Every list both backends accept gives the same readback bytes wherever the recording backend
//! knows them (the tests in `backend::tests` run the same lists on both).
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
