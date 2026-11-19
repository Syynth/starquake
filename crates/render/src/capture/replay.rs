//! Replay: a capture fed to a fresh backend, with the capture's ids and fence values mapped to
//! the backend's, and the capture's results compared with what the backend gives
//! (docs/specs/render-backend.md, "Render capture v2", "Identity" and "Determinism, and what
//! replay compares"; decision 16).
//!
//! # Identity
//!
//! One table maps each id the capture created to the id the fresh backend returned, filled by
//! each `Create*` record both sides accepted. A `Destroy` forwards the destroy with its fence
//! mapped, and the table keeps the id until the fresh backend's [`Backend::completed`] has passed
//! that fence (checked before each record), as the id stays usable until then in a live run.
//! Before a list is submitted, [`map_ids`] rewrites every id in it through the table; a list, a
//! `Create*` or a `Destroy` that names an id the table does not hold (never created, refused,
//! or already destroyed) is not passed on: replay refuses it with [`FindingKind::Unmapped`],
//! counts it, and goes on. [`ReadbackId`]s are the caller's choice, not the backend's, and are
//! used as they are.
//!
//! Fence values map the same way: each `Submit` both sides accepted pairs the captured fence
//! with the fresh one, and a `Destroy`'s fence maps to the fresh fence of the latest captured
//! list at or below it that both sides accepted (zero when there is none). A `Destroy` whose
//! fence is past every list captured so far waits until a captured list reaches it (or the
//! capture ends), so that it is not mapped to a fence the fresh backend passes too early.
//!
//! # What is compared
//!
//! - Each call's refusal: a call refused on one side only is a difference.
//! - Each `ReadbackResult` holding a value: after the fresh backend reaches the last fence,
//!   [`Backend::take_readback`] is called and bytes are compared with bytes. A
//!   [`Readback::Unknown`] on either side is not compared; nothing waiting on replay is a
//!   difference. A captured "nothing yet" is not replayed (taking it would use the readback up).
//! - Each `PipelineOutcome`: [`Backend::pipeline_status`] is polled until it settles or
//!   [`ReplayOptions::pipeline_timeout`] passes.
//! - Each `FrameResult`: after the last fence, the frame of that number the fresh backend
//!   handed the [`ReplayFrames`] sink, by size and hash; frames whose pixels either side did not
//!   know are not compared.
//! - Each retranslated shader's stage, entry and interface against the record.
//!
//! A difference is a finding, not a failure: the replay goes on, and the report counts them
//! and keeps the first. It ends early only for [`ReplayError`]: a malformed capture, a lost
//! device, a resolver that cannot go on, or a fence the backend does not reach in time.
//!
//! # Memory
//!
//! A `CreateBuffer` or `CreateTexture` that the fresh backend's [`Caps`] accept is charged its
//! described size ([`crate::validate::texture_size`] for a texture, with checked arithmetic)
//! against the replay's memory limit ([`ReplayOptions::memory_limit`], or the backend's
//! [`crate::MemoryBudget::allowed`]); a create past it is refused with
//! [`FindingKind::MemoryLimit`] before the backend sees it. The charge is released when the id
//! leaves the table. A description the device refuses is passed on uncharged, for the backend
//! to refuse.

use super::read::CaptureReader;
use super::record::{
    Created, FrameResult, PipelineResult, Record, ShaderCode, Submit,
};
use super::CaptureError;
use crate::backend::{Backend, PipelineStatus, PresentTarget, Readback, WaitOutcome};
use crate::command::{
    BindingResource, BufferSlice, Command, CommandList, CopyRegion, DispatchParams, DrawParams,
};
use crate::error::RenderError;
use crate::frame::{Frame, FrameSink};
use crate::id::{
    BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId,
    ViewId,
};
use crate::shader::{ShaderDesc, ShaderOrigin, ShaderSource};
use crate::validate;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Read;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Turns a shader origin back into a shader (decision 15). The real one is the GPU crate's: it
/// decodes the shader crate's record, translates it for the host target the record names, and
/// converts the result with the same function the back thread uses.
pub trait ShaderResolver {
    /// The shader `origin` names, with its stage, entry, interface and code.
    fn resolve(&self, origin: &ShaderOrigin) -> Result<ShaderDesc, ResolveError>;
}

/// Why a [`ShaderResolver`] could not give a shader.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ResolveError {
    /// This shader only: replay refuses it, fails its pipelines, and goes on.
    #[error("{0}")]
    Shader(String),
    /// Replay cannot go on (another translator version without leave to retranslate,
    /// decision 19).
    #[error("{0}")]
    Fatal(String),
}

/// How a replay runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayOptions {
    /// The fresh backend's name, compared with the capture's ([`ReplayReport::other_backend`]).
    pub backend_name: String,
    /// The replay's memory limit; `None` takes the backend's [`crate::MemoryBudget::allowed`].
    pub memory_limit: Option<u64>,
    /// How long to wait for a fence before comparing readbacks and frames.
    pub fence_timeout: Duration,
    /// How long to poll a pipeline before reporting it unsettled.
    pub pipeline_timeout: Duration,
}

/// The most frames a [`ReplayFrames`] keeps waiting to be compared; past it the oldest goes.
pub const MAX_REPLAY_FRAMES: usize = 1024;

/// The frame sink a replay's caller installs on the fresh backend, through which replay sees the
/// frames it presents. Keeps the last [`MAX_REPLAY_FRAMES`] frames' records.
#[derive(Debug, Default)]
pub struct ReplayFrames {
    frames: Mutex<VecDeque<FrameResult>>,
}

impl ReplayFrames {
    /// An empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes the frame numbered `frame`, dropping every frame that arrived before it.
    fn take(&self, frame: u64) -> Option<FrameResult> {
        let mut frames = self.frames.lock().unwrap_or_else(PoisonError::into_inner);
        let at = frames.iter().position(|f| f.frame == frame)?;
        frames.drain(..at);
        frames.pop_front()
    }

    /// How many frames are waiting.
    pub fn len(&self) -> usize {
        self.frames
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Whether no frame is waiting.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl FrameSink for ReplayFrames {
    fn frame(&self, frame: Frame) {
        let mut frames = self.frames.lock().unwrap_or_else(PoisonError::into_inner);
        if frames.len() >= MAX_REPLAY_FRAMES {
            frames.pop_front();
        }
        frames.push_back(FrameResult::of(&frame));
    }
}

/// One finding: a record replay refused or a result that differed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    /// The record's position in the capture.
    pub record: u64,
    /// What was found.
    pub kind: FindingKind,
}

/// What a finding is.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum FindingKind {
    /// The record names an id the replay's tables do not hold.
    #[error("names {0}, which the replay does not hold")]
    Unmapped(ResourceId),
    /// The shader was captured with no origin.
    #[error("the shader was captured with no origin, so it cannot be translated again")]
    MissingShader,
    /// The resolver refused the shader.
    #[error("the shader could not be translated again: {0}")]
    Unresolved(String),
    /// The create would pass the replay's memory limit.
    #[error("needs {needed} bytes in all, past the replay's memory limit of {limit}")]
    MemoryLimit {
        /// The live bytes with this resource.
        needed: u64,
        /// The limit.
        limit: u64,
    },
    /// The fresh backend refused a call the captured one accepted.
    #[error("refused on replay only: {0}")]
    RefusedOnReplay(String),
    /// The fresh backend accepted a call the captured one refused.
    #[error("refused in the capture only")]
    RefusedInCapture,
    /// The retranslated shader's stage, entry or interface is not the record's.
    #[error("the translated shader's stage, entry or interface differs from the capture's")]
    ShaderChanged,
    /// A readback's bytes differ.
    #[error("readback {} (generation {}) differs", .0.index(), .0.generation())]
    Readback(ReadbackId),
    /// A readback the capture holds was not waiting on replay.
    #[error("readback {} (generation {}) was not there on replay", .0.index(), .0.generation())]
    ReadbackMissing(ReadbackId),
    /// A pipeline settled differently.
    #[error("the pipeline was {} in the capture and {} on replay", result_words(*.captured), status_words(*.replayed))]
    Pipeline {
        /// The captured outcome.
        captured: PipelineResult,
        /// The replayed status when polling stopped.
        replayed: PipelineStatus,
    },
    /// A frame's size or pixels differ.
    #[error("frame {0} differs")]
    Frame(u64),
    /// A frame the capture holds was not presented on replay.
    #[error("frame {0} was not presented on replay")]
    FrameMissing(u64),
}

/// A pipeline outcome in words.
const fn result_words(result: PipelineResult) -> &'static str {
    match result {
        PipelineResult::Ready => "ready",
        PipelineResult::Failed => "failed",
    }
}

/// A pipeline status in words.
const fn status_words(status: PipelineStatus) -> &'static str {
    match status {
        PipelineStatus::Compiling => "still compiling",
        PipelineStatus::Ready => "ready",
        PipelineStatus::Failed => "failed",
    }
}

/// What a replay found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayReport {
    /// Records read, `End` included.
    pub records: u64,
    /// Lists the fresh backend accepted.
    pub lists: u64,
    /// Records replay refused itself (unmapped ids, shaders it could not make, the memory
    /// limit).
    pub refused: u64,
    /// The first of those.
    pub first_refused: Option<Finding>,
    /// Results and refusals that differed.
    pub differences: u64,
    /// The first of those.
    pub first_difference: Option<Finding>,
    /// Results compared: readbacks and frames compared byte for byte or by hash, and pipeline
    /// outcomes.
    pub compared: u64,
    /// The fence value the capture's `Limit` record names, when it stopped early.
    pub limit: Option<FenceValue>,
    /// The capture was made with draws skipped while their pipelines compiled.
    pub captured_skipping: bool,
    /// The capture came from another backend or device: its name or [`Caps`] differ, so drawn
    /// pixels may legitimately differ.
    pub other_backend: bool,
}

/// Why a replay stopped before the capture's end.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ReplayError {
    /// The capture is malformed.
    #[error(transparent)]
    Capture(#[from] CaptureError),
    /// The fresh backend lost its device.
    #[error("record {record}: the host device was lost")]
    DeviceLost {
        /// The record being replayed.
        record: u64,
    },
    /// The resolver cannot go on.
    #[error("record {record}: {message}")]
    Shader {
        /// The record being replayed.
        record: u64,
        /// The resolver's message.
        message: String,
    },
    /// The fresh backend did not reach a fence in time.
    #[error("record {record}: the backend did not reach fence value {fence} in time")]
    Timeout {
        /// The record being replayed.
        record: u64,
        /// The fence value waited for.
        fence: u64,
    },
    /// Waiting for a fence failed.
    #[error("record {record}: waiting for fence value {fence} failed: {message}")]
    Wait {
        /// The record being replayed.
        record: u64,
        /// The fence value waited for.
        fence: u64,
        /// The backend's error.
        message: String,
    },
}

/// Rewrites every id in `list` through `map` (module docs, "Identity"), command by command and
/// field by field in declaration order. Returns the first id `map` does not hold (or maps to
/// another kind); the list is then partly rewritten and must not be submitted.
pub fn map_ids(
    list: &mut CommandList,
    map: &dyn Fn(ResourceId) -> Option<ResourceId>,
) -> Result<(), ResourceId> {
    let m = Mapper { map };
    list.commands.iter_mut().try_for_each(|c| m.command(c))
}

/// Maps ids kind by kind.
struct Mapper<'a> {
    map: &'a dyn Fn(ResourceId) -> Option<ResourceId>,
}

/// Defines a kind-preserving mapping method.
macro_rules! kind_map {
    ($($method:ident: $ty:ty => $variant:ident),* $(,)?) => {$(
        fn $method(&self, id: &mut $ty) -> Result<(), ResourceId> {
            match (self.map)(ResourceId::$variant(*id)) {
                Some(ResourceId::$variant(mapped)) => {
                    *id = mapped;
                    Ok(())
                }
                _ => Err(ResourceId::$variant(*id)),
            }
        }
    )*};
}

impl Mapper<'_> {
    kind_map! {
        buffer: BufferId => Buffer,
        texture: TextureId => Texture,
        view: ViewId => View,
        sampler: SamplerId => Sampler,
        pipeline: PipelineId => Pipeline,
    }

    fn slice(&self, slice: &mut BufferSlice) -> Result<(), ResourceId> {
        self.buffer(&mut slice.buffer)
    }

    fn region(&self, region: &mut CopyRegion) -> Result<(), ResourceId> {
        match region {
            CopyRegion::Buffer(slice) => self.slice(slice),
            CopyRegion::Texture(texture) => self.texture(&mut texture.texture),
        }
    }

    fn bindings(&self, bindings: &mut [crate::command::Binding]) -> Result<(), ResourceId> {
        bindings
            .iter_mut()
            .try_for_each(|binding| match &mut binding.resource {
                BindingResource::Buffer(slice) => self.slice(slice),
                BindingResource::View(view) => self.view(view),
                BindingResource::Sampler(sampler) => self.sampler(sampler),
            })
    }

    fn command(&self, command: &mut Command) -> Result<(), ResourceId> {
        match command {
            Command::Upload { to, .. } => self.region(to),
            Command::Copy(copy) => {
                self.region(&mut copy.src)?;
                self.region(&mut copy.dst)
            }
            Command::Blit(blit) => {
                self.view(&mut blit.src)?;
                self.view(&mut blit.dst)
            }
            Command::Clear(clear) => self.view(&mut clear.target),
            Command::SetRenderTargets(targets) => {
                targets
                    .colors
                    .iter_mut()
                    .flatten()
                    .try_for_each(|view| self.view(view))?;
                targets
                    .depth_stencil
                    .as_mut()
                    .map_or(Ok(()), |view| self.view(view))
            }
            Command::Draw(draw) => {
                self.pipeline(&mut draw.pipeline)?;
                self.bindings(&mut draw.bindings)?;
                draw.vertex_buffers
                    .iter_mut()
                    .try_for_each(|slice| self.slice(slice))?;
                if let Some(index) = &mut draw.index_buffer {
                    self.slice(&mut index.slice)?;
                }
                match &mut draw.params {
                    DrawParams::Indirect { buffer, .. } => self.buffer(buffer),
                    DrawParams::Direct { .. } | DrawParams::Indexed { .. } => Ok(()),
                }
            }
            Command::Dispatch(dispatch) => {
                self.pipeline(&mut dispatch.pipeline)?;
                self.bindings(&mut dispatch.bindings)?;
                match &mut dispatch.params {
                    DispatchParams::Indirect { buffer, .. } => self.buffer(buffer),
                    DispatchParams::Direct { .. } => Ok(()),
                }
            }
            Command::Readback { from, .. } => self.region(from),
            Command::Present(present) => self.view(&mut present.view),
            Command::Marker(_) => Ok(()),
        }
    }
}

/// Replays the capture `reader` reads on `backend` (module docs). `frames` must be the frame
/// sink installed on `backend`, so that replay sees the frames it presents. The backend should
/// be fresh and built to wait while pipelines compile (render-backend, "Pipelines and compile
/// timing"); building it is the caller's.
pub fn replay<R: Read>(
    reader: &mut CaptureReader<R>,
    backend: &dyn Backend,
    resolver: &dyn ShaderResolver,
    frames: &ReplayFrames,
    options: &ReplayOptions,
) -> Result<ReplayReport, ReplayError> {
    let mut replay = Replay {
        backend,
        resolver,
        frames,
        options,
        limit: options
            .memory_limit
            .unwrap_or_else(|| backend.budget().allowed),
        used: 0,
        ids: HashMap::new(),
        charges: HashMap::new(),
        pending: BTreeMap::new(),
        deferred: BTreeMap::new(),
        seq: 0,
        fences: Vec::new(),
        last_captured: FenceValue::ZERO,
        last_replayed: FenceValue::ZERO,
        sources: Vec::new(),
        report: ReplayReport::default(),
    };
    while let Some(record) = reader.next_record()? {
        let position = reader.position().saturating_sub(1);
        replay.report.records = reader.position();
        replay.prune();
        if replay.record(position, record)? {
            break;
        }
    }
    Ok(replay.report)
}

/// A replay in progress.
struct Replay<'a> {
    backend: &'a dyn Backend,
    resolver: &'a dyn ShaderResolver,
    frames: &'a ReplayFrames,
    options: &'a ReplayOptions,
    limit: u64,
    /// Bytes charged for live buffers and textures.
    used: u64,
    /// The capture's ids to the fresh backend's.
    ids: HashMap<ResourceId, ResourceId>,
    /// What each fresh buffer and texture was charged.
    charges: HashMap<ResourceId, u64>,
    /// Destroys forwarded, by the fresh fence after which the table forgets them.
    pending: BTreeMap<(FenceValue, u64), (ResourceId, ResourceId)>,
    /// Destroys waiting for a captured list to reach their fence, by that captured fence.
    deferred: BTreeMap<(FenceValue, u64), (ResourceId, ResourceId)>,
    /// Orders entries of `pending` and `deferred` with equal fences.
    seq: u64,
    /// Captured fence to fresh fence, for lists both sides accepted, rising.
    fences: Vec<(FenceValue, FenceValue)>,
    last_captured: FenceValue,
    last_replayed: FenceValue,
    sources: Vec<Arc<[u8]>>,
    report: ReplayReport,
}

/// How a call's results compare.
enum Compared<T> {
    /// Both accepted; the fresh value.
    Both(T),
    /// Both refused, or one side only (counted).
    Not,
}

impl Replay<'_> {
    /// Replays one record; `true` at `End`.
    fn record(&mut self, at: u64, record: Record) -> Result<bool, ReplayError> {
        match record {
            Record::End => {
                self.resolve_deferred(FenceValue::new(u64::MAX));
                self.wait_last(at)?;
                return Ok(true);
            }
            Record::Start(start) => {
                self.report.captured_skipping = start.wait == super::WaitSetting::Skip;
                self.report.other_backend = start.backend != self.options.backend_name
                    || start.caps != *self.backend.caps();
            }
            Record::CreateBuffer(c) => {
                if let Some(charge) = self.charge_buffer(at, &c.desc) {
                    let result = self.backend.create_buffer(&c.desc);
                    self.created(at, &c.result, result, ResourceId::Buffer, charge)?;
                }
            }
            Record::CreateTexture(c) => {
                if let Some(charge) = self.charge_texture(at, &c.desc) {
                    let result = self.backend.create_texture(&c.desc);
                    self.created(at, &c.result, result, ResourceId::Texture, charge)?;
                }
            }
            Record::CreateView(c) => {
                let mut texture = c.texture;
                if self.map_one(at, &mut texture, ResourceId::Texture) {
                    let result = self.backend.create_view(texture, &c.desc);
                    self.created(at, &c.result, result, ResourceId::View, 0)?;
                }
            }
            Record::CreateSampler(c) => {
                let result = self.backend.create_sampler(&c.desc);
                self.created(at, &c.result, result, ResourceId::Sampler, 0)?;
            }
            Record::ShaderSource(s) => self.sources.push(s.bytes.into()),
            Record::CreateShader(c) => {
                if let Some(desc) = self.shader(at, &c)? {
                    let result = self.backend.create_shader(desc);
                    self.created(at, &c.result, result, ResourceId::Shader, 0)?;
                }
            }
            Record::CreateGraphicsPipeline(c) => {
                let mut desc = c.desc;
                let s = &mut desc.stages;
                let mapped = self.map_one(at, &mut s.vertex, ResourceId::Shader)
                    && [
                        &mut s.tess_control,
                        &mut s.tess_eval,
                        &mut s.geometry,
                        &mut s.fragment,
                    ]
                    .into_iter()
                    .flatten()
                    .all(|id| self.map_one(at, id, ResourceId::Shader));
                if mapped {
                    let result = self.backend.create_graphics_pipeline(desc);
                    self.created(at, &c.result, result, ResourceId::Pipeline, 0)?;
                }
            }
            Record::CreateComputePipeline(c) => {
                let mut desc = c.desc;
                if self.map_one(at, &mut desc.shader, ResourceId::Shader) {
                    let result = self.backend.create_compute_pipeline(desc);
                    self.created(at, &c.result, result, ResourceId::Pipeline, 0)?;
                }
            }
            Record::Destroy(d) => self.destroy(at, d.id, d.after),
            Record::Submit(s) => self.submit(at, s)?,
            Record::PresentTarget(t) => {
                // Replay presents offscreen, whatever the capture's target was.
                let result = self.backend.set_present_target(PresentTarget::Offscreen);
                self.compare(at, &t.result, result)?;
            }
            Record::ReadbackResult(r) => {
                if let Some(captured) = r.value {
                    self.wait_last(at)?;
                    self.readback(at, r.id, &captured);
                }
            }
            Record::PipelineOutcome(p) => {
                let mut id = p.pipeline;
                if self.map_one(at, &mut id, ResourceId::Pipeline) {
                    self.pipeline(at, id, p.outcome);
                }
            }
            Record::FrameResult(f) => {
                self.wait_last(at)?;
                self.frame(at, &f);
            }
            Record::Limit(l) => self.report.limit = Some(l.fence),
        }
        Ok(false)
    }

    /// Counts a record replay refused.
    fn refuse(&mut self, record: u64, kind: FindingKind) {
        self.report.refused = self.report.refused.saturating_add(1);
        self.report
            .first_refused
            .get_or_insert(Finding { record, kind });
    }

    /// Counts a difference.
    fn differ(&mut self, record: u64, kind: FindingKind) {
        self.report.differences = self.report.differences.saturating_add(1);
        self.report
            .first_difference
            .get_or_insert(Finding { record, kind });
    }

    /// Compares a call's results; a lost device stops the replay.
    fn compare<C, T>(
        &mut self,
        at: u64,
        captured: &Created<C>,
        result: Result<T, RenderError>,
    ) -> Result<Compared<T>, ReplayError> {
        Ok(match (captured, result) {
            (_, Err(RenderError::DeviceLost)) => return Err(ReplayError::DeviceLost { record: at }),
            (Created::Made(_), Ok(value)) => Compared::Both(value),
            (Created::Refused(_), Err(_)) => Compared::Not,
            (Created::Made(_), Err(error)) => {
                self.differ(at, FindingKind::RefusedOnReplay(error.to_string()));
                Compared::Not
            }
            (Created::Refused(_), Ok(_)) => {
                self.differ(at, FindingKind::RefusedInCapture);
                Compared::Not
            }
        })
    }

    /// Enters a create's result in the table; a resource only the fresh backend made is
    /// destroyed at once, since no captured id names it.
    fn created<T: Copy>(
        &mut self,
        at: u64,
        captured: &Created<T>,
        result: Result<T, RenderError>,
        kind: fn(T) -> ResourceId,
        charge: u64,
    ) -> Result<(), ReplayError> {
        let made_here = result.as_ref().ok().copied();
        match (self.compare(at, captured, result)?, captured.made()) {
            (Compared::Both(fresh), Some(old)) => {
                let fresh = kind(fresh);
                self.ids.insert(kind(old), fresh);
                if charge > 0 {
                    self.used = self.used.saturating_add(charge);
                    self.charges.insert(fresh, charge);
                }
            }
            _ => {
                if let Some(fresh) = made_here.filter(|_| captured.made().is_none()) {
                    self.backend.destroy(kind(fresh), FenceValue::ZERO);
                }
            }
        }
        Ok(())
    }

    /// Maps one id in place; an id the table does not hold is refused and counted.
    fn map_one<T: Copy + SameKind>(
        &mut self,
        at: u64,
        id: &mut T,
        kind: fn(T) -> ResourceId,
    ) -> bool {
        let captured = kind(*id);
        let Some(fresh) = self.ids.get(&captured).copied().and_then(T::from_resource) else {
            self.refuse(at, FindingKind::Unmapped(captured));
            return false;
        };
        *id = fresh;
        true
    }

    /// The charge for a buffer, or `None` when the create is refused for the memory limit.
    fn charge_buffer(&mut self, at: u64, desc: &crate::resource::BufferDesc) -> Option<u64> {
        let size = validate::validate_buffer(self.backend.caps(), desc)
            .ok()
            .map(|()| desc.size);
        self.charge(at, size)
    }

    /// The charge for a texture, or `None` when the create is refused for the memory limit.
    fn charge_texture(&mut self, at: u64, desc: &crate::resource::TextureDesc) -> Option<u64> {
        let size = validate::validate_texture(self.backend.caps(), desc)
            .ok()
            .and_then(|()| validate::texture_size(desc).ok());
        self.charge(at, size)
    }

    /// Checks `size` (when the device accepts the description) against the memory limit.
    fn charge(&mut self, at: u64, size: Option<u64>) -> Option<u64> {
        let Some(size) = size else {
            // The device refuses the description: pass it on uncharged, for the backend to
            // refuse.
            return Some(0);
        };
        match self.used.checked_add(size) {
            Some(needed) if needed <= self.limit => Some(size),
            needed => {
                self.refuse(
                    at,
                    FindingKind::MemoryLimit {
                        needed: needed.unwrap_or(u64::MAX),
                        limit: self.limit,
                    },
                );
                None
            }
        }
    }

    /// The shader a `CreateShader` record describes, or `None` when replay refuses it.
    fn shader(
        &mut self,
        at: u64,
        c: &super::record::CreateShader,
    ) -> Result<Option<ShaderDesc>, ReplayError> {
        let source = match &c.code {
            ShaderCode::SpirV(words) => ShaderSource::SpirV(words.clone()),
            ShaderCode::Msl(text) => ShaderSource::Msl(text.clone()),
            ShaderCode::Missing => {
                self.refuse(at, FindingKind::MissingShader);
                return Ok(None);
            }
            ShaderCode::Origin {
                source,
                stage_index,
            } => {
                let index = usize::try_from(*source).unwrap_or(usize::MAX);
                // The reader refuses a `CreateShader` naming a source not yet given.
                let Some(record) = self.sources.get(index).map(Arc::clone) else {
                    self.refuse(at, FindingKind::Unresolved(format!("no shader source {source}")));
                    return Ok(None);
                };
                let origin = ShaderOrigin {
                    record,
                    stage_index: *stage_index,
                };
                return match self.resolver.resolve(&origin) {
                    Ok(mut desc) => {
                        if desc.stage != c.stage
                            || desc.entry != c.entry
                            || desc.interface != c.interface
                        {
                            self.differ(at, FindingKind::ShaderChanged);
                        }
                        desc.origin = Some(origin);
                        Ok(Some(desc))
                    }
                    Err(ResolveError::Shader(message)) => {
                        self.refuse(at, FindingKind::Unresolved(message));
                        Ok(None)
                    }
                    Err(ResolveError::Fatal(message)) => {
                        Err(ReplayError::Shader { record: at, message })
                    }
                };
            }
        };
        Ok(Some(ShaderDesc {
            stage: c.stage,
            entry: c.entry.clone(),
            source,
            interface: c.interface.clone(),
            origin: None,
        }))
    }

    /// Forwards a destroy with its fence mapped, or holds it until a captured list reaches its
    /// fence (module docs, "Identity").
    fn destroy(&mut self, at: u64, id: ResourceId, after: FenceValue) {
        let Some(fresh) = self.ids.get(&id).copied() else {
            self.refuse(at, FindingKind::Unmapped(id));
            return;
        };
        self.seq = self.seq.saturating_add(1);
        if after > self.last_captured {
            self.deferred.insert((after, self.seq), (id, fresh));
        } else {
            self.forward_destroy(id, fresh, after);
        }
    }

    /// Destroys `fresh` after the fresh fence `after` maps to, and remembers to forget it.
    fn forward_destroy(&mut self, id: ResourceId, fresh: ResourceId, after: FenceValue) {
        let mapped = self.map_fence(after);
        self.backend.destroy(fresh, mapped);
        self.seq = self.seq.saturating_add(1);
        self.pending.insert((mapped, self.seq), (id, fresh));
    }

    /// The fresh fence of the latest captured list at or below `after` that both sides
    /// accepted; zero when there is none.
    fn map_fence(&self, after: FenceValue) -> FenceValue {
        let at_or_below = self.fences.partition_point(|&(captured, _)| captured <= after);
        at_or_below
            .checked_sub(1)
            .and_then(|i| self.fences.get(i))
            .map_or(FenceValue::ZERO, |&(_, fresh)| fresh)
    }

    /// Forwards the held destroys whose captured fence a list has now reached (all of them at
    /// the end, with `upto` at the largest value).
    fn resolve_deferred(&mut self, upto: FenceValue) {
        while let Some(entry) = self.deferred.first_entry() {
            let (after, _) = *entry.key();
            if after > upto {
                return;
            }
            let (id, fresh) = entry.remove();
            self.forward_destroy(id, fresh, after);
        }
    }

    /// Forgets the destroyed ids whose fresh fence has passed, releasing their charges.
    fn prune(&mut self) {
        let done = self.backend.completed();
        while let Some(entry) = self.pending.first_entry() {
            if entry.key().0 > done {
                return;
            }
            let (id, fresh) = entry.remove();
            if self.ids.get(&id) == Some(&fresh) {
                self.ids.remove(&id);
            }
            if let Some(charge) = self.charges.remove(&fresh) {
                self.used = self.used.saturating_sub(charge);
            }
        }
    }

    fn submit(&mut self, at: u64, s: Submit) -> Result<(), ReplayError> {
        let Submit { mut list, result } = s;
        let ids = &self.ids;
        let mapped = map_ids(&mut list, &|id| ids.get(&id).copied());
        if let Some(fence) = result.made() {
            self.last_captured = fence;
        }
        if let Err(id) = mapped {
            self.refuse(at, FindingKind::Unmapped(id));
        } else {
            let outcome = self.backend.submit(list);
            if let Ok(fresh) = outcome {
                self.report.lists = self.report.lists.saturating_add(1);
                self.last_replayed = fresh;
            }
            if let (Compared::Both(fresh), Some(captured)) =
                (self.compare(at, &result, outcome)?, result.made())
            {
                self.fences.push((captured, fresh));
            }
        }
        self.resolve_deferred(self.last_captured);
        Ok(())
    }

    /// Waits for the fresh backend to reach the last fence it handed out.
    fn wait_last(&self, at: u64) -> Result<(), ReplayError> {
        let fence = self.last_replayed;
        if fence == FenceValue::ZERO {
            return Ok(());
        }
        match self.backend.wait(fence, self.options.fence_timeout) {
            Ok(WaitOutcome::Reached) => Ok(()),
            Ok(WaitOutcome::TimedOut) => Err(ReplayError::Timeout {
                record: at,
                fence: fence.get(),
            }),
            Err(RenderError::DeviceLost) => Err(ReplayError::DeviceLost { record: at }),
            Err(error) => Err(ReplayError::Wait {
                record: at,
                fence: fence.get(),
                message: error.to_string(),
            }),
        }
    }

    fn readback(&mut self, at: u64, id: ReadbackId, captured: &Readback) {
        match (captured, self.backend.take_readback(id)) {
            (_, None) => self.differ(at, FindingKind::ReadbackMissing(id)),
            (Readback::Bytes(old), Some(Readback::Bytes(new))) => {
                self.report.compared = self.report.compared.saturating_add(1);
                if *old != new {
                    self.differ(at, FindingKind::Readback(id));
                }
            }
            (Readback::Unknown, Some(_)) | (_, Some(Readback::Unknown)) => {}
        }
    }

    fn pipeline(&mut self, at: u64, id: PipelineId, captured: PipelineResult) {
        let started = Instant::now();
        let status = loop {
            let status = self.backend.pipeline_status(id);
            if status != PipelineStatus::Compiling
                || started.elapsed() >= self.options.pipeline_timeout
            {
                break status;
            }
            std::thread::sleep(PIPELINE_POLL);
        };
        self.report.compared = self.report.compared.saturating_add(1);
        let same = matches!(
            (captured, status),
            (PipelineResult::Ready, PipelineStatus::Ready)
                | (PipelineResult::Failed, PipelineStatus::Failed)
        );
        if !same {
            self.differ(
                at,
                FindingKind::Pipeline {
                    captured,
                    replayed: status,
                },
            );
        }
    }

    fn frame(&mut self, at: u64, captured: &FrameResult) {
        let Some(fresh) = self.frames.take(captured.frame) else {
            self.differ(at, FindingKind::FrameMissing(captured.frame));
            return;
        };
        if (fresh.width, fresh.height) != (captured.width, captured.height) {
            self.differ(at, FindingKind::Frame(captured.frame));
            return;
        }
        if let (Some(old), Some(new)) = (captured.sha256, fresh.sha256) {
            self.report.compared = self.report.compared.saturating_add(1);
            if old != new {
                self.differ(at, FindingKind::Frame(captured.frame));
            }
        }
    }
}

/// How long replay sleeps between polls of a compiling pipeline.
const PIPELINE_POLL: Duration = Duration::from_millis(1);

/// An id type that can be taken back out of a [`ResourceId`] of its kind.
trait SameKind: Sized {
    /// The id inside `id`, when `id` is of this kind.
    fn from_resource(id: ResourceId) -> Option<Self>;
}

/// Implements [`SameKind`].
macro_rules! same_kind {
    ($($ty:ty => $variant:ident),* $(,)?) => {$(
        impl SameKind for $ty {
            fn from_resource(id: ResourceId) -> Option<Self> {
                match id {
                    ResourceId::$variant(id) => Some(id),
                    _ => None,
                }
            }
        }
    )*};
}

same_kind! {
    BufferId => Buffer,
    TextureId => Texture,
    ViewId => View,
    SamplerId => Sampler,
    ShaderId => Shader,
    PipelineId => Pipeline,
}
