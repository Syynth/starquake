//! `CapturingBackend`: the [`Backend`] wrapper that records every state-changing call
//! (docs/specs/render-backend.md, "Render capture v2", "Where a capture is recorded").
//!
//! Each call that changes backend state is forwarded unchanged and recorded, in the order the
//! calls reach the wrapper: the wrapper holds its one lock across the wrapped call and the
//! record, so the records have one total order even when two threads call, and the fence values
//! of the recorded lists rise as the wrapped backend handed them out. Results (`take_readback`,
//! the first settled `pipeline_status` of each live pipeline) are recorded the same way.
//! `caps`, `budget`, `staging`, `completed`, `wait` and `take_present_feedback` change no
//! backend state and are forwarded unrecorded.
//!
//! **Frames.** [`CapturingBackend::record_frame`] takes a dumped frame (the wiring installs a
//! frame sink on the wrapped backend that calls it, and then the app's own sink). It does not
//! take the lock: a backend may deliver frames inside `submit`, while the lock is held. The
//! frame's record waits in a queue and is written right after the record of the call that holds
//! the lock (so a frame delivered inside a `submit` follows that `Submit`), or, when no call
//! holds it, before `record_frame` returns, or at the latest by the next recorded call or
//! [`CapturingBackend::finish`].
//!
//! **Lock rules.** The lock is held while the wrapped backend runs a recorded call and while
//! the [`CaptureSink`] runs. A frame sink the wrapped backend calls inside `submit` must
//! therefore not call the wrapper (other than `record_frame`), and neither may the capture sink.
//!
//! **When writing fails.** A record any of the writer's checks refuses (a bound passed, the size
//! limit reached, or a wrapped backend that hands out an id still live) ends the capture: the
//! wrapper writes `Limit` and `End`, keeps the first such error ([`CapturingBackend::stopped`]),
//! and from then on forwards every call unrecorded. The file always closes well formed; the
//! wrapped call's result is returned whatever happened to its record.

use super::record::{
    CapturedTarget, Create, CreateShader, CreateView, Created, Destroy, FrameResult,
    PipelineOutcome, PipelineResult, ReadbackResult, Record, Start, TargetRecord,
};
use super::write::{CaptureSink, CaptureWriter};
use super::CaptureError;
use crate::backend::{
    Backend, MemoryBudget, PipelineStatus, PresentFeedback, PresentTarget, Readback, WaitOutcome,
};
use crate::caps::Caps;
use crate::command::{CommandList, StagingSlice};
use crate::error::RenderError;
use crate::frame::Frame;
use crate::id::{
    BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId,
    ViewId,
};
use crate::pipeline::{ComputePipelineDesc, GraphicsPipelineDesc};
use crate::resource::{BufferDesc, SamplerDesc, TextureDesc, ViewDesc};
use crate::shader::ShaderDesc;
use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

/// A backend that records every state-changing call to a [`CaptureSink`] and forwards it to the
/// backend it wraps (module docs).
pub struct CapturingBackend {
    inner: Arc<dyn Backend>,
    state: Mutex<State>,
    /// Frame records waiting for the lock (module docs, "Frames").
    frames: Mutex<VecDeque<FrameResult>>,
}

impl std::fmt::Debug for CapturingBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturingBackend")
            .field("state", &self.state)
            .field("frames", &self.frames)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct State {
    writer: CaptureWriter,
    /// The first error that ended the capture.
    stopped: Option<CaptureError>,
    /// Pipelines whose settled status has been recorded.
    settled: HashSet<PipelineId>,
}

impl CapturingBackend {
    /// Wraps `inner`, writing the capture's header and `Start` to `sink` at once. The `Start`
    /// record takes `build` and `backend` (the build id and the wrapped backend's name),
    /// `inner`'s [`Caps`], `wait`, and `inner`'s configured memory limit
    /// ([`MemoryBudget::allowed`]). The capture is kept within `size_limit` bytes.
    ///
    /// Errors: as [`CaptureWriter::new`] (a text past [`super::MAX_TEXT_BYTES`], or a size limit
    /// too small to hold the start).
    pub fn new(
        inner: Arc<dyn Backend>,
        sink: Arc<dyn CaptureSink>,
        build: String,
        backend: String,
        wait: super::WaitSetting,
        size_limit: u64,
    ) -> Result<Self, CaptureError> {
        let start = Start {
            build,
            backend,
            caps: inner.caps().clone(),
            wait,
            memory_limit: inner.budget().allowed,
        };
        Self::with_writer(inner, CaptureWriter::new(sink, start, size_limit)?)
    }

    /// Wraps `inner` around a writer that has written its `Start` (for a writer built with
    /// lowered [`super::CaptureLimits`]).
    pub fn with_writer(inner: Arc<dyn Backend>, writer: CaptureWriter) -> Result<Self, CaptureError> {
        if writer.is_finished() {
            return Err(CaptureError::Finished);
        }
        Ok(Self {
            inner,
            state: Mutex::new(State {
                writer,
                stopped: None,
                settled: HashSet::new(),
            }),
            frames: Mutex::new(VecDeque::new()),
        })
    }

    /// The error that ended the capture early, if one did (module docs, "When writing fails").
    pub fn stopped(&self) -> Option<CaptureError> {
        self.state().stopped.clone()
    }

    /// Records a dumped frame (module docs, "Frames"). Takes no lock that a recorded call
    /// holds.
    pub fn record_frame(&self, frame: &Frame) {
        self.frame_queue().push_back(FrameResult::of(frame));
        if let Ok(state) = self.state.try_lock() {
            self.flush_frames(state);
        }
    }

    /// Ends the capture with `End`, after any frame records still waiting. Later calls are
    /// forwarded unrecorded. Errors: [`CaptureError::Finished`] when the capture had already
    /// ended (by an earlier `finish`, or early, when [`CapturingBackend::stopped`] says why).
    pub fn finish(&self) -> Result<(), CaptureError> {
        let mut state = self.state();
        self.write_frames(&mut state);
        state.writer.finish()
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn frame_queue(&self) -> MutexGuard<'_, VecDeque<FrameResult>> {
        self.frames.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Writes the waiting frame records under `state`, releases it, and writes any that came in
    /// meanwhile, while the lock is free to take (module docs, "Frames").
    fn flush_frames<'a>(&'a self, mut state: MutexGuard<'a, State>) {
        loop {
            self.write_frames(&mut state);
            drop(state);
            if self.frame_queue().is_empty() {
                return;
            }
            match self.state.try_lock() {
                Ok(next) => state = next,
                // Another call holds it, and writes the queue before it lets go.
                Err(_) => return,
            }
        }
    }

    /// Writes every waiting frame record.
    fn write_frames(&self, state: &mut State) {
        loop {
            let Some(frame) = self.frame_queue().pop_front() else {
                return;
            };
            record(state, &Record::FrameResult(frame));
        }
    }

    /// Runs a recorded call: `call` forwards it under the lock and returns its result and the
    /// record to write.
    fn recorded<T>(&self, call: impl FnOnce(&mut State) -> (T, Option<Record>)) -> T {
        let mut state = self.state();
        let (value, rec) = call(&mut state);
        if let Some(rec) = rec {
            record(&mut state, &rec);
        }
        self.flush_frames(state);
        value
    }
}

/// Writes `rec` unless the capture has ended; a refused record ends it (module docs, "When
/// writing fails").
fn record(state: &mut State, rec: &Record) {
    if state.writer.is_finished() {
        return;
    }
    if let Err(error) = state.writer.write(rec) {
        stop(state, error);
    }
}

/// Ends the capture early with `Limit` and `End`, keeping the first error.
fn stop(state: &mut State, error: CaptureError) {
    if state.stopped.is_none() {
        state.stopped = Some(error);
    }
    // Room for the closing records is kept by every record, so this fails only when the
    // capture has already ended, and the first error is kept either way.
    if !state.writer.is_finished()
        && let Err(close) = state.writer.close_at_limit()
    {
        state.stopped.get_or_insert(close);
    }
}

impl Backend for CapturingBackend {
    fn caps(&self) -> &Caps {
        self.inner.caps()
    }

    fn budget(&self) -> MemoryBudget {
        self.inner.budget()
    }

    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferId, RenderError> {
        self.recorded(|_| {
            let result = self.inner.create_buffer(desc);
            let rec = Record::CreateBuffer(Create {
                desc: *desc,
                result: Created::of(&result),
            });
            (result, Some(rec))
        })
    }

    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureId, RenderError> {
        self.recorded(|_| {
            let result = self.inner.create_texture(desc);
            let rec = Record::CreateTexture(Create {
                desc: *desc,
                result: Created::of(&result),
            });
            (result, Some(rec))
        })
    }

    fn create_view(&self, texture: TextureId, desc: &ViewDesc) -> Result<ViewId, RenderError> {
        self.recorded(|_| {
            let result = self.inner.create_view(texture, desc);
            let rec = Record::CreateView(CreateView {
                texture,
                desc: *desc,
                result: Created::of(&result),
            });
            (result, Some(rec))
        })
    }

    fn create_sampler(&self, desc: &SamplerDesc) -> Result<SamplerId, RenderError> {
        self.recorded(|_| {
            let result = self.inner.create_sampler(desc);
            let rec = Record::CreateSampler(Create {
                desc: *desc,
                result: Created::of(&result),
            });
            (result, Some(rec))
        })
    }

    /// Writes a `ShaderSource` record first when the shader's origin is new, then forwards the
    /// call and writes `CreateShader`: the code as an origin, as SPIR-V or MSL as it is, or as
    /// missing for a naga module with no origin (decision 15).
    fn create_shader(&self, desc: ShaderDesc) -> Result<ShaderId, RenderError> {
        self.recorded(|state| {
            let code = if state.writer.is_finished() {
                None
            } else {
                match state.writer.shader_code(&desc) {
                    Ok(code) => Some(code),
                    Err(error) => {
                        stop(state, error);
                        None
                    }
                }
            };
            let (stage, entry, interface) = (desc.stage, desc.entry.clone(), desc.interface.clone());
            let result = self.inner.create_shader(desc);
            let rec = code.map(|code| {
                Record::CreateShader(CreateShader {
                    stage,
                    entry,
                    interface,
                    code,
                    result: Created::of(&result),
                })
            });
            (result, rec)
        })
    }

    fn destroy(&self, resource: ResourceId, after: FenceValue) {
        self.recorded(|state| {
            self.inner.destroy(resource, after);
            if let ResourceId::Pipeline(id) = resource {
                state.settled.remove(&id);
            }
            ((), Some(Record::Destroy(Destroy { id: resource, after })))
        });
    }

    fn create_graphics_pipeline(
        &self,
        desc: GraphicsPipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        self.recorded(|_| {
            let kept = desc.clone();
            let result = self.inner.create_graphics_pipeline(desc);
            let rec = Record::CreateGraphicsPipeline(Create {
                desc: kept,
                result: Created::of(&result),
            });
            (result, Some(rec))
        })
    }

    fn create_compute_pipeline(
        &self,
        desc: ComputePipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        self.recorded(|_| {
            let result = self.inner.create_compute_pipeline(desc);
            let rec = Record::CreateComputePipeline(Create {
                desc,
                result: Created::of(&result),
            });
            (result, Some(rec))
        })
    }

    /// Records the first `Ready` or `Failed` reported for each pipeline live in the capture.
    fn pipeline_status(&self, id: PipelineId) -> PipelineStatus {
        self.recorded(|state| {
            let status = self.inner.pipeline_status(id);
            let outcome = match status {
                PipelineStatus::Compiling => None,
                PipelineStatus::Ready => Some(PipelineResult::Ready),
                PipelineStatus::Failed => Some(PipelineResult::Failed),
            };
            let rec = outcome
                .filter(|_| {
                    state.writer.is_live(ResourceId::Pipeline(id)) && !state.settled.contains(&id)
                })
                .map(|outcome| {
                    state.settled.insert(id);
                    Record::PipelineOutcome(PipelineOutcome {
                        pipeline: id,
                        outcome,
                    })
                });
            (status, rec)
        })
    }

    fn staging(&self, size: u64) -> Result<StagingSlice, RenderError> {
        self.inner.staging(size)
    }

    /// Encodes the list before forwarding it (the call consumes it), then writes `Submit` with
    /// the fence value or refusal.
    fn submit(&self, list: CommandList) -> Result<FenceValue, RenderError> {
        self.recorded(|state| {
            let encoded = if state.writer.is_finished() {
                None
            } else {
                match state.writer.encode_list(&list) {
                    Ok(encoded) => Some(encoded),
                    Err(error) => {
                        stop(state, error);
                        None
                    }
                }
            };
            let result = self.inner.submit(list);
            if let Some(encoded) = encoded
                && let Err(error) = state.writer.write_submit(encoded, &Created::of(&result))
            {
                stop(state, error);
            }
            (result, None)
        })
    }

    fn completed(&self) -> FenceValue {
        self.inner.completed()
    }

    fn wait(&self, value: FenceValue, timeout: Duration) -> Result<WaitOutcome, RenderError> {
        self.inner.wait(value, timeout)
    }

    fn take_readback(&self, id: ReadbackId) -> Option<Readback> {
        self.recorded(|_| {
            let value = self.inner.take_readback(id);
            let rec = Record::ReadbackResult(ReadbackResult {
                id,
                value: value.clone(),
            });
            (value, Some(rec))
        })
    }

    fn set_present_target(&self, target: PresentTarget) -> Result<(), RenderError> {
        self.recorded(|_| {
            let result = self.inner.set_present_target(target);
            let rec = Record::PresentTarget(TargetRecord {
                target: CapturedTarget::of(&target),
                result: Created::of(&result),
            });
            (result, Some(rec))
        })
    }

    fn take_present_feedback(&self) -> Vec<PresentFeedback> {
        self.inner.take_present_feedback()
    }
}
