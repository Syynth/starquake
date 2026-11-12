//! The `Backend` trait: the one seam between the emulated GPU and the host GPU
//! (docs/specs/render-backend.md, "The trait").
//!
//! One implementation per host API (wgpu first, in `starquake-render-wgpu`), plus the recording
//! backend (slice R1). `core` builds the backend with the shared compile pool and the GPU crate's
//! wait-while-compiling setting, and hands the GPU crate an `Arc<dyn Backend>`; neither is a trait
//! method.
//!
//! Open, shared with the GPU front/back protocol: how the back thread wakes when a submitted list
//! completes (render-backend verification 10). No wake hook is defined here until it is settled.

use crate::caps::Caps;
use crate::command::{CommandList, StagingSlice};
use crate::error::RenderError;
use crate::id::{
    BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId,
    ViewId,
};
use crate::pipeline::{ComputePipelineDesc, GraphicsPipelineDesc};
use crate::resource::{BufferDesc, SamplerDesc, TextureDesc, ViewDesc};
use crate::shader::ShaderDesc;
use std::time::Duration;

/// Host memory the device allows and what the backend uses, for the GPU crate's eviction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Bytes the backend may use.
    pub allowed: u64,
    /// Bytes it uses now.
    pub used: u64,
}

/// Where a pipeline's compilation stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipelineStatus {
    /// Queued or running on the compile pool.
    Compiling,
    /// Usable.
    Ready,
    /// Failed; its draws are skipped and counted once per pipeline.
    Failed,
}

/// How a [`Backend::wait`] ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WaitOutcome {
    /// The host GPU reached the value.
    Reached,
    /// The timeout passed first.
    TimedOut,
}

/// The bytes a `Readback` command copied out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Readback {
    /// Tightly packed bytes of the region read.
    Bytes(Vec<u8>),
    /// The recording backend cannot know the contents (a draw wrote them); it returns this
    /// rather than wrong pixels.
    Unknown,
}

/// Where presents go.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PresentTarget {
    /// No window: presents are counted and dumped on request. Headless runs and tests.
    Offscreen,
    // `Window(WindowHandles)` (raw-window-handle display and window handles) arrives with the
    // window presentation slice, R8; `Exported(..)` for an embedding frontend later.
}

/// Whether a presented frame reached the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PresentOutcome {
    /// Shown.
    Shown,
    /// Replaced by a newer frame before it was shown.
    Dropped,
}

/// What happened to one presented frame; for statistics only, never guest-visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PresentFeedback {
    /// The frame number from the present command.
    pub frame: u64,
    /// Shown or dropped.
    pub outcome: PresentOutcome,
    /// Host time of the outcome, from the backend's own clock origin.
    pub host_time: Duration,
}

/// The seam to the host GPU. One implementation per host API, plus the recording backend.
/// `Send + Sync`: the GPU back thread and the compile workers share it.
pub trait Backend: Send + Sync {
    /// What the device can do; fixed for the backend's lifetime.
    fn caps(&self) -> &Caps;

    /// Host memory allowed and in use, for eviction.
    fn budget(&self) -> MemoryBudget;

    /// Allocates a buffer handle at once, on the calling thread; the host object is created on
    /// the render thread before the first list that uses it.
    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferId, RenderError>;

    /// Allocates a texture handle, as [`Backend::create_buffer`] does.
    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureId, RenderError>;

    /// Allocates a handle for a view of `texture`.
    fn create_view(&self, texture: TextureId, desc: &ViewDesc) -> Result<ViewId, RenderError>;

    /// Allocates a sampler handle.
    fn create_sampler(&self, desc: &SamplerDesc) -> Result<SamplerId, RenderError>;

    /// Allocates a shader handle; the shader is shared by every pipeline that uses it.
    fn create_shader(&self, desc: ShaderDesc) -> Result<ShaderId, RenderError>;

    /// Destroyed once the host GPU has passed `after`; until then the id stays reserved.
    fn destroy(&self, resource: ResourceId, after: FenceValue);

    /// Queues compilation on the compile workers and returns at once.
    fn create_graphics_pipeline(
        &self,
        desc: GraphicsPipelineDesc,
    ) -> Result<PipelineId, RenderError>;

    /// Queues compilation on the compile workers and returns at once.
    fn create_compute_pipeline(&self, desc: ComputePipelineDesc)
    -> Result<PipelineId, RenderError>;

    /// Compiling, ready or failed.
    fn pipeline_status(&self, id: PipelineId) -> PipelineStatus;

    /// Staging memory the caller fills and then names in an `Upload` command.
    fn staging(&self, size: u64) -> Result<StagingSlice, RenderError>;

    /// Hands a finished list to the render thread and returns the fence value it signals when
    /// the host GPU has finished it. Blocks only while the queue is full.
    fn submit(&self, list: CommandList) -> Result<FenceValue, RenderError>;

    /// The highest fence value the host GPU has finished.
    fn completed(&self) -> FenceValue;

    /// Waits until the host GPU reaches `value` or `timeout` passes.
    fn wait(&self, value: FenceValue, timeout: Duration) -> Result<WaitOutcome, RenderError>;

    /// The bytes a `Readback` command copied out, once its list's fence has passed.
    fn take_readback(&self, id: ReadbackId) -> Option<Readback>;

    /// Sets where presents go.
    fn set_present_target(&self, target: PresentTarget) -> Result<(), RenderError>;

    /// Feedback for the frames presented since the last call.
    fn take_present_feedback(&self) -> Vec<PresentFeedback>;
}
