//! Host rendering backends behind one trait.
//!
//! This crate is the seam between the emulated GPU (`starquake-gpu`) and the host GPU
//! (docs/specs/render-backend.md). It owns the [`Backend`] trait, every type in it, the
//! description checks against [`Caps`], and the host-neutral [`PixelFormat`] table. Backends live
//! in their own crates (`starquake-render-wgpu` first); no wgpu type appears here, and this
//! crate depends on no workspace crate. It never sees a guest address: the GPU crate turns guest
//! state into the plain-data [`CommandList`]s and descriptions defined here.
//!
//! What is here (slice R0):
//!
//! - generational handles and [`FenceValue`], in [`id`];
//! - [`PixelFormat`] with block sizes and properties, in [`format`];
//! - [`Caps`], device limits, features and per-format support, in [`caps`];
//! - the descriptions `Backend::create_*` take, in [`resource`], [`shader`] and [`pipeline`],
//!   with [`ShaderSource::Naga`] carrying a naga module;
//! - [`Command`] and [`CommandList`], in [`command`];
//! - the [`Backend`] trait, in [`backend`];
//! - [`RenderError`], in [`error`];
//! - description checks against [`Caps`], in [`validate`].
//!
//! Slice R1 adds the [`RecordingBackend`], in [`recording`]: the trait with no GPU, recording
//! every list and modelling on the CPU the commands with exact answers.
//!
//! Slice R9 (its first part) adds frame dumps, in [`frame`]: a present with `dump` set hands a
//! [`Frame`] to the [`FrameSink`] installed on the backend, and a frame encodes itself as a
//! PNG. This crate still does no file I/O.
//!
//! Later slices add format fallbacks and `emulation()` (R2), the CPU ASTC decoder (R3) and the
//! host shaders (R4).

pub mod backend;
pub mod caps;
pub mod command;
pub mod error;
mod flags;
pub mod format;
pub mod frame;
pub mod id;
pub mod pipeline;
pub mod recording;
pub mod resource;
pub mod shader;
pub mod validate;

pub use backend::{
    Backend, MemoryBudget, PipelineStatus, PresentFeedback, PresentOutcome, PresentTarget,
    Readback, WaitOutcome,
};
pub use caps::{Caps, Features, FormatFeatures, FormatTable, Limits, SampleCounts};
pub use command::{Command, CommandList, StagingSlice};
pub use error::{RenderError, UnknownFlags};
pub use format::PixelFormat;
pub use frame::{Frame, FramePixels, FrameSink};
pub use id::{
    BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId,
    ViewId,
};
pub use pipeline::{ComputePipelineDesc, GraphicsPipelineDesc};
pub use recording::{FrameDump, RecordingBackend};
pub use resource::{BufferDesc, BufferUsage, SamplerDesc, TextureDesc, TextureUsage, ViewDesc};
pub use shader::{ResourceBinding, ShaderDesc, ShaderInterface, ShaderSource, ShaderStage};
