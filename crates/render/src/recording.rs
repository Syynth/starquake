//! The recording backend: the [`Backend`] trait with no GPU (docs/specs/render-backend.md, "The
//! recording backend", slice R1).
//!
//! It keeps every submitted list and every resource description, and renders them as a text
//! [`RecordingBackend::snapshot`] for tests on the GPU side. It models on the CPU the commands
//! that have exact answers, so a test can check pixels with no GPU and the wgpu backend can be
//! checked against it: wherever this backend has an answer, a real backend must give the same
//! bytes. Its shape follows Eden's null renderer (`eden/src/video_core/renderer_null/` @
//! cf654052, shape only: a renderer that accepts all work and draws nothing); everything else
//! is new.
//!
//! **What is modelled.** Each buffer and each texture mip level has a CPU copy of its bytes, and
//! every byte is either known or unknown ([`Readback::Unknown`] is what a readback of any unknown
//! byte returns, never guessed pixels):
//!
//! - nothing is known of a new resource (wgpu zero-fills new resources, but a direct backend need
//!   not, so the oracle claims nothing);
//! - an upload makes its bytes known; a copy moves bytes and their known flags as they are;
//!   except that an upload, a readback or a copy between a buffer and a texture of a two-aspect
//!   depth-stencil format or `X8_D24_UNORM` makes its destination unknown (a readback reports
//!   [`Readback::Unknown`]), because a real backend does not move those bytes as they are
//!   ([`texel`]'s "Buffer copies");
//! - a clear makes its area known when the value's encoding is exact on every device
//!   ([`texel`]'s rules), and unknown otherwise;
//! - a blit copies texels only when it is unscaled, between views of one integer colour format
//!   (`uint` or `sint`), both with the identity swizzle; any other blit (scaled, converting,
//!   swizzled, or of a normalised, sRGB, float, depth, stencil or compressed format, whose texels
//!   a real blit converts through float) makes its destination rectangle unknown;
//! - a draw makes every mip and layer of its colour and depth/stencil target views unknown, and
//!   a draw or dispatch makes every bound storage resource unknown: each buffer range bound from
//!   a buffer with the storage usage, and every mip and layer of each bound view whose texture
//!   has the storage usage (whether the shader writes it is not known here);
//! - a readback copies the bytes it reads, or reports them unknown; a present with `dump` set
//!   keeps the presented rectangle the same way ([`FrameDump`]), untransformed, and reports it
//!   unknown when the view has a non-identity swizzle (the present blit samples through it) or a
//!   compressed format;
//! - multisampled textures hold one copy of each texel: known bytes mean every sample holds them.
//!
//! **Lists.** `submit` checks the whole list before it changes anything (`plan::check`): every id
//! names a live resource, every description passes [`crate::validate`] (which refuses a copy
//! whose sides overlap), and a blit does not read and write overlapping texels of one texture
//! (refused rather than modelled: Vulkan leaves overlapping copies and blits undefined). A refused list leaves every resource, readback and the fence
//! unchanged and is not kept. Applying a checked list looks the same ids up again in tables
//! nothing has changed since, over regions the check bounded, so it meets no error the check did
//! not; its error path exists only so that no lookup can panic. A list that passes is applied in order and completes at once: its
//! fence value is [`Backend::completed`] when `submit` returns. Render targets set by
//! `SetRenderTargets` last until the end of the list (render-backend, "Commands": the list's end
//! closes the host pass), so each list starts with none.
//!
//! **Other behaviour.** Pipelines are ready at once unless a test scripts their status
//! ([`RecordingBackend::script_pipeline`]). `destroy` frees a resource at once when its fence
//! value has passed and otherwise after the submit that reaches it; until then the id stays
//! usable. `wait` returns at once when the value has been reached, and otherwise waits on the
//! host clock for another thread's submit, up to the timeout. Present feedback reports every
//! present as shown at host time zero: there is no display and no clock.

mod plan;
mod slots;
mod snapshot;
mod store;
mod texel;

use crate::backend::{
    Backend, MemoryBudget, PipelineStatus, PresentFeedback, PresentOutcome, PresentTarget,
    Readback, WaitOutcome,
};
use crate::caps::Caps;
use crate::capture::Capture;
use crate::command::{CommandList, PresentTransform, StagingSlice};
use crate::error::RenderError;
use crate::format::PixelFormat;
use crate::id::{
    BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId,
    ViewId,
};
use crate::pipeline::{ComputePipelineDesc, GraphicsPipelineDesc};
use crate::resource::{BufferDesc, SamplerDesc, TextureDesc, ViewDesc};
use crate::shader::{ShaderDesc, ShaderStage};
use crate::validate;
use plan::{Op, Place};
use slots::Slots;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;
use store::{LevelShape, Packed, Rows, Store};

/// The pixels of a present whose `dump` flag was set, read from the presented view's base mip
/// level at its base layer, before the transform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameDump {
    /// The frame number from the present.
    pub frame: u64,
    /// The view presented.
    pub view: ViewId,
    /// The view's format; the bytes are tightly packed texels of it.
    pub format: PixelFormat,
    /// Width of the presented rectangle, in texels.
    pub width: u32,
    /// Height of the presented rectangle, in texels.
    pub height: u32,
    /// The transform the present asked for, not applied to the bytes.
    pub transform: PresentTransform,
    /// The rectangle's bytes, or [`Readback::Unknown`] when any of them is unknown, the format
    /// is compressed, or the view has a non-identity swizzle.
    pub pixels: Readback,
}

/// The backend with no GPU: records everything and models what has exact answers (module docs).
#[derive(Debug)]
pub struct RecordingBackend {
    caps: Caps,
    memory_allowed: u64,
    state: Mutex<State>,
    completed_moved: Condvar,
}

#[derive(Debug)]
struct BufferRec {
    desc: BufferDesc,
    store: Store,
}

#[derive(Debug)]
struct TextureRec {
    desc: TextureDesc,
    /// One store per mip level.
    levels: Vec<Store>,
    /// What the host object would take: every level, every sample.
    bytes: u64,
}

#[derive(Debug)]
struct ViewRec {
    texture: TextureId,
    desc: ViewDesc,
}

/// Which kind of pipeline a slot holds; its description is kept in the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineKind {
    Graphics,
    Compute,
}

#[derive(Debug)]
struct PipelineRec {
    kind: PipelineKind,
    /// Statuses still to report, the last of which stays.
    script: VecDeque<PipelineStatus>,
}

/// One recorded call, for the snapshot.
#[derive(Debug)]
enum Event {
    Created(ResourceId, String),
    Destroy {
        id: ResourceId,
        after: FenceValue,
        outcome: DestroyOutcome,
    },
    Freed(ResourceId),
    Submitted(FenceValue, CommandList),
    Target(PresentTarget),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DestroyOutcome {
    Now,
    Pending,
    Refused,
}

#[derive(Debug, Default)]
struct State {
    buffers: Slots<BufferRec>,
    textures: Slots<TextureRec>,
    views: Slots<ViewRec>,
    samplers: Slots<SamplerDesc>,
    shaders: Slots<ShaderDesc>,
    pipelines: Slots<PipelineRec>,
    completed: FenceValue,
    /// Destroys waiting for their fence value, in call order.
    pending: Vec<(ResourceId, FenceValue)>,
    readbacks: BTreeMap<ReadbackId, Readback>,
    feedback: Vec<PresentFeedback>,
    dumps: Vec<FrameDump>,
    refused_destroys: u64,
    log: Vec<Event>,
}

/// The error for an id that names no live resource of its kind.
const fn gone(text: &'static str) -> RenderError {
    RenderError::Invalid(text)
}

const NO_BUFFER: &str = "no live buffer has this id";
const NO_TEXTURE: &str = "no live texture has this id";
const NO_VIEW: &str = "no live view has this id";
const NO_SAMPLER: &str = "no live sampler has this id";
const NO_SHADER: &str = "no live shader has this id";
const NO_PIPELINE: &str = "no live pipeline has this id";

/// The block layout of `desc`'s mip level `mip`.
fn level_shape(desc: &TextureDesc, mip: u32) -> Result<LevelShape, RenderError> {
    let extent = desc.mip_extent(mip).ok_or(RenderError::NotBelow {
        what: "mip level",
        index: mip.into(),
        count: desc.mip_levels.into(),
    })?;
    let f = desc.format;
    Ok(LevelShape {
        wide: extent.width.div_ceil(f.block_width()),
        high: extent.height.div_ceil(f.block_height()),
        deep: extent.depth_or_layers,
        block_bytes: f.bytes_per_block(),
    })
}

impl State {
    fn texture(&self, id: TextureId) -> Result<&TextureRec, RenderError> {
        self.textures
            .get(id.index(), id.generation())
            .ok_or(gone(NO_TEXTURE))
    }

    fn buffer(&self, id: BufferId) -> Result<&BufferRec, RenderError> {
        self.buffers
            .get(id.index(), id.generation())
            .ok_or(gone(NO_BUFFER))
    }

    fn view(&self, id: ViewId) -> Result<(&ViewRec, &TextureRec), RenderError> {
        let view = self
            .views
            .get(id.index(), id.generation())
            .ok_or(gone(NO_VIEW))?;
        Ok((view, self.texture(view.texture)?))
    }

    fn shader_stage(&self, id: ShaderId) -> Result<ShaderStage, RenderError> {
        self.shaders
            .get(id.index(), id.generation())
            .map(|s| s.stage)
            .ok_or(gone(NO_SHADER))
    }

    fn is_live(&self, id: ResourceId) -> bool {
        match id {
            ResourceId::Buffer(i) => self.buffers.get(i.index(), i.generation()).is_some(),
            ResourceId::Texture(i) => self.textures.get(i.index(), i.generation()).is_some(),
            ResourceId::View(i) => self.views.get(i.index(), i.generation()).is_some(),
            ResourceId::Sampler(i) => self.samplers.get(i.index(), i.generation()).is_some(),
            ResourceId::Shader(i) => self.shaders.get(i.index(), i.generation()).is_some(),
            ResourceId::Pipeline(i) => self.pipelines.get(i.index(), i.generation()).is_some(),
        }
    }

    fn free(&mut self, id: ResourceId) {
        let freed = match id {
            ResourceId::Buffer(i) => self.buffers.remove(i.index(), i.generation()).is_some(),
            ResourceId::Texture(i) => self.textures.remove(i.index(), i.generation()).is_some(),
            ResourceId::View(i) => self.views.remove(i.index(), i.generation()).is_some(),
            ResourceId::Sampler(i) => self.samplers.remove(i.index(), i.generation()).is_some(),
            ResourceId::Shader(i) => self.shaders.remove(i.index(), i.generation()).is_some(),
            ResourceId::Pipeline(i) => self.pipelines.remove(i.index(), i.generation()).is_some(),
        };
        if freed {
            self.log.push(Event::Freed(id));
        }
    }

    /// Frees every pending destroy whose fence value has passed, in call order.
    fn free_passed(&mut self) {
        let completed = self.completed;
        let (due, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending)
            .into_iter()
            .partition(|(_, after)| *after <= completed);
        self.pending = waiting;
        for (id, _) in due {
            self.free(id);
        }
    }

    /// The store and rows a place names.
    fn rows(&self, place: &Place) -> Result<(&Store, Rows), RenderError> {
        match *place {
            Place::Buffer { id, offset, size } => {
                let b = self.buffer(id)?;
                Ok((&b.store, Rows::range(offset, size, b.store.len())?))
            }
            Place::Texture { id, mip, area } => {
                let t = self.texture(id)?;
                let store = level(&t.levels, mip)?;
                Ok((store, Rows::blocks(level_shape(&t.desc, mip)?, area)?))
            }
        }
    }

    /// The store a place names, to change, and its rows.
    fn rows_mut(&mut self, place: &Place) -> Result<(&mut Store, Rows), RenderError> {
        match *place {
            Place::Buffer { id, offset, size } => {
                let b = self
                    .buffers
                    .get_mut(id.index(), id.generation())
                    .ok_or(gone(NO_BUFFER))?;
                let rows = Rows::range(offset, size, b.store.len())?;
                Ok((&mut b.store, rows))
            }
            Place::Texture { id, mip, area } => {
                let t = self
                    .textures
                    .get_mut(id.index(), id.generation())
                    .ok_or(gone(NO_TEXTURE))?;
                let rows = Rows::blocks(level_shape(&t.desc, mip)?, area)?;
                let store = usize::try_from(mip)
                    .ok()
                    .and_then(|m| t.levels.get_mut(m))
                    .ok_or(RenderError::NotBelow {
                        what: "mip level",
                        index: mip.into(),
                        count: t.desc.mip_levels.into(),
                    })?;
                Ok((store, rows))
            }
        }
    }

    fn gather(&self, place: &Place) -> Result<Packed, RenderError> {
        let (store, rows) = self.rows(place)?;
        store.gather(&rows)
    }

    /// Applies one checked op.
    fn apply(&mut self, op: Op<'_>) -> Result<(), RenderError> {
        match op {
            Op::Nothing => Ok(()),
            Op::Upload { bytes, to } => {
                let (store, rows) = self.rows_mut(&to)?;
                store.scatter(&rows, &Packed::known(bytes.to_vec()))
            }
            Op::Copy { from, to } => {
                let packed = self.gather(&from)?;
                let (store, rows) = self.rows_mut(&to)?;
                store.scatter(&rows, &packed)
            }
            Op::Fill { to, texel } => {
                let (store, rows) = self.rows_mut(&to)?;
                store.fill(&rows, &texel)
            }
            Op::Forget(places) => {
                for place in &places {
                    let (store, rows) = self.rows_mut(place)?;
                    store.forget(&rows)?;
                }
                Ok(())
            }
            Op::Readback { from, id } => {
                let read = match &from {
                    Some(place) => readback(self.gather(place)?),
                    None => Readback::Unknown,
                };
                self.readbacks.insert(id, read);
                Ok(())
            }
            Op::Present { desc, format, from } => {
                if desc.dump {
                    let pixels = match &from {
                        Some(place) => readback(self.gather(place)?),
                        None => Readback::Unknown,
                    };
                    self.dumps.push(FrameDump {
                        frame: desc.frame,
                        view: desc.view,
                        format,
                        width: desc.src_rect.width,
                        height: desc.src_rect.height,
                        transform: desc.transform,
                        pixels,
                    });
                }
                self.feedback.push(PresentFeedback {
                    frame: desc.frame,
                    outcome: PresentOutcome::Shown,
                    host_time: Duration::ZERO,
                });
                Ok(())
            }
        }
    }
}

/// The store of mip level `mip`.
fn level(levels: &[Store], mip: u32) -> Result<&Store, RenderError> {
    usize::try_from(mip)
        .ok()
        .and_then(|m| levels.get(m))
        .ok_or(RenderError::NotBelow {
            what: "mip level",
            index: mip.into(),
            count: u64::try_from(levels.len()).unwrap_or(u64::MAX),
        })
}

fn readback(packed: Packed) -> Readback {
    packed
        .into_known()
        .map_or(Readback::Unknown, Readback::Bytes)
}

/// Bytes a texture's host object takes: every level, every sample.
fn texture_bytes(desc: &TextureDesc) -> Result<(u64, Vec<Store>), RenderError> {
    let mut total: u64 = 0;
    let mut levels = Vec::new();
    for mip in 0..desc.mip_levels {
        let len = level_shape(desc, mip)?.len()?;
        let bytes = u64::try_from(len)
            .ok()
            .and_then(|b| b.checked_mul(desc.sample_count.into()))
            .ok_or(RenderError::Overflow {
                what: "texture size",
            })?;
        total = total.saturating_add(bytes);
        levels.push(Store::new(len));
    }
    Ok((total, levels))
}

impl RecordingBackend {
    /// A recording backend for a device with `caps`, reporting `memory_allowed` bytes as its
    /// budget.
    pub fn new(caps: Caps, memory_allowed: u64) -> Self {
        Self {
            caps,
            memory_allowed,
            state: Mutex::new(State::default()),
            completed_moved: Condvar::new(),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Every call recorded so far, as text: resources created and destroyed, present targets
    /// set, and each submitted list with its fence value and commands. Deterministic for a
    /// given call order; meant for `insta` snapshots.
    pub fn snapshot(&self) -> String {
        snapshot::render(&self.state().log)
    }

    /// Every list submitted so far with the fence value it signalled, in order.
    pub fn lists(&self) -> Vec<(FenceValue, CommandList)> {
        self.state()
            .log
            .iter()
            .filter_map(|e| match e {
                Event::Submitted(fence, list) => Some((*fence, list.clone())),
                _ => None,
            })
            .collect()
    }

    /// Every list submitted so far, as a [`Capture`] ready to be written out: what
    /// [`RecordingBackend::lists`] returns. Refused lists are not in it.
    pub fn capture(&self) -> Capture {
        Capture::new(self.lists())
    }

    /// Makes [`Backend::pipeline_status`] for `id` report `steps` in turn, one per call, then
    /// keep reporting the last; for tests of the wait and skip policies.
    pub fn script_pipeline(
        &self,
        id: PipelineId,
        steps: Vec<PipelineStatus>,
    ) -> Result<(), RenderError> {
        if steps.is_empty() {
            return Err(RenderError::Invalid(
                "a pipeline script needs at least one status",
            ));
        }
        let mut state = self.state();
        let rec = state
            .pipelines
            .get_mut(id.index(), id.generation())
            .ok_or(gone(NO_PIPELINE))?;
        rec.script = steps.into();
        Ok(())
    }

    /// The frames presented with `dump` set since the last call, in order.
    pub fn take_frame_dumps(&self) -> Vec<FrameDump> {
        std::mem::take(&mut self.state().dumps)
    }

    /// How many `destroy` calls named no live resource or one already waiting to be destroyed.
    pub fn refused_destroys(&self) -> u64 {
        self.state().refused_destroys
    }
}

impl Backend for RecordingBackend {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    /// `allowed` is the configured limit; `used` is what every live buffer and texture would
    /// take on a device (texture bytes times sample count), whether or not anything wrote them.
    fn budget(&self) -> MemoryBudget {
        let state = self.state();
        let buffers = state.buffers.iter().map(|(_, _, b)| b.desc.size);
        let textures = state.textures.iter().map(|(_, _, t)| t.bytes);
        MemoryBudget {
            allowed: self.memory_allowed,
            used: buffers.chain(textures).fold(0, u64::saturating_add),
        }
    }

    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferId, RenderError> {
        validate::validate_buffer(&self.caps, desc)?;
        let len = usize::try_from(desc.size).map_err(|_| RenderError::Overflow {
            what: "buffer size",
        })?;
        let mut state = self.state();
        let (index, generation) = state.buffers.insert(BufferRec {
            desc: *desc,
            store: Store::new(len),
        })?;
        let id = BufferId::new(index, generation);
        state.log.push(Event::Created(
            ResourceId::Buffer(id),
            snapshot::buffer(desc),
        ));
        Ok(id)
    }

    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureId, RenderError> {
        validate::validate_texture(&self.caps, desc)?;
        let (bytes, levels) = texture_bytes(desc)?;
        let mut state = self.state();
        let (index, generation) = state.textures.insert(TextureRec {
            desc: *desc,
            levels,
            bytes,
        })?;
        let id = TextureId::new(index, generation);
        state.log.push(Event::Created(
            ResourceId::Texture(id),
            snapshot::texture(desc),
        ));
        Ok(id)
    }

    fn create_view(&self, texture: TextureId, desc: &ViewDesc) -> Result<ViewId, RenderError> {
        let mut state = self.state();
        let tex = state.texture(texture)?;
        validate::validate_view(&self.caps, &tex.desc, desc)?;
        let (index, generation) = state.views.insert(ViewRec {
            texture,
            desc: *desc,
        })?;
        let id = ViewId::new(index, generation);
        state.log.push(Event::Created(
            ResourceId::View(id),
            snapshot::view(texture, desc),
        ));
        Ok(id)
    }

    fn create_sampler(&self, desc: &SamplerDesc) -> Result<SamplerId, RenderError> {
        validate::validate_sampler(&self.caps, desc)?;
        let mut state = self.state();
        let (index, generation) = state.samplers.insert(*desc)?;
        let id = SamplerId::new(index, generation);
        state.log.push(Event::Created(
            ResourceId::Sampler(id),
            snapshot::sampler(desc),
        ));
        Ok(id)
    }

    fn create_shader(&self, desc: ShaderDesc) -> Result<ShaderId, RenderError> {
        validate::validate_shader(&self.caps, &desc)?;
        let line = snapshot::shader(&desc);
        let mut state = self.state();
        let (index, generation) = state.shaders.insert(desc)?;
        let id = ShaderId::new(index, generation);
        state.log.push(Event::Created(ResourceId::Shader(id), line));
        Ok(id)
    }

    fn destroy(&self, resource: ResourceId, after: FenceValue) {
        let mut state = self.state();
        let outcome =
            if !state.is_live(resource) || state.pending.iter().any(|(id, _)| *id == resource) {
                state.refused_destroys = state.refused_destroys.saturating_add(1);
                DestroyOutcome::Refused
            } else if after <= state.completed {
                DestroyOutcome::Now
            } else {
                state.pending.push((resource, after));
                DestroyOutcome::Pending
            };
        state.log.push(Event::Destroy {
            id: resource,
            after,
            outcome,
        });
        if outcome == DestroyOutcome::Now {
            state.free(resource);
        }
    }

    fn create_graphics_pipeline(
        &self,
        desc: GraphicsPipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        validate::validate_graphics_pipeline(&self.caps, &desc)?;
        let mut state = self.state();
        let s = &desc.stages;
        for (id, stage, wrong) in [
            (
                Some(s.vertex),
                ShaderStage::Vertex,
                "a pipeline's vertex shader is not a vertex shader",
            ),
            (
                s.tess_control,
                ShaderStage::TessControl,
                "a pipeline's tessellation control shader is not one",
            ),
            (
                s.tess_eval,
                ShaderStage::TessEval,
                "a pipeline's tessellation evaluation shader is not one",
            ),
            (
                s.geometry,
                ShaderStage::Geometry,
                "a pipeline's geometry shader is not a geometry shader",
            ),
            (
                s.fragment,
                ShaderStage::Fragment,
                "a pipeline's fragment shader is not a fragment shader",
            ),
        ] {
            if let Some(id) = id
                && state.shader_stage(id)? != stage
            {
                return Err(RenderError::Invalid(wrong));
            }
        }
        let line = snapshot::graphics_pipeline(&desc);
        let (index, generation) = state.pipelines.insert(PipelineRec {
            kind: PipelineKind::Graphics,
            script: VecDeque::new(),
        })?;
        let id = PipelineId::new(index, generation);
        state
            .log
            .push(Event::Created(ResourceId::Pipeline(id), line));
        Ok(id)
    }

    fn create_compute_pipeline(
        &self,
        desc: ComputePipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        let mut state = self.state();
        if state.shader_stage(desc.shader)? != ShaderStage::Compute {
            return Err(RenderError::Invalid(
                "a compute pipeline's shader is not a compute shader",
            ));
        }
        let (index, generation) = state.pipelines.insert(PipelineRec {
            kind: PipelineKind::Compute,
            script: VecDeque::new(),
        })?;
        let id = PipelineId::new(index, generation);
        state.log.push(Event::Created(
            ResourceId::Pipeline(id),
            snapshot::compute_pipeline(desc),
        ));
        Ok(id)
    }

    /// Ready, or the next scripted status; [`PipelineStatus::Failed`] for an id that names no
    /// live pipeline.
    fn pipeline_status(&self, id: PipelineId) -> PipelineStatus {
        let mut state = self.state();
        let Some(rec) = state.pipelines.get_mut(id.index(), id.generation()) else {
            return PipelineStatus::Failed;
        };
        if rec.script.len() > 1 {
            rec.script.pop_front().unwrap_or(PipelineStatus::Ready)
        } else {
            rec.script.front().copied().unwrap_or(PipelineStatus::Ready)
        }
    }

    fn staging(&self, size: u64) -> Result<StagingSlice, RenderError> {
        validate::validate_staging_size(&self.caps, size)?;
        let len = usize::try_from(size).map_err(|_| RenderError::Overflow {
            what: "staging size",
        })?;
        Ok(StagingSlice::from_bytes(vec![0; len]))
    }

    fn submit(&self, list: CommandList) -> Result<FenceValue, RenderError> {
        let mut state = self.state();
        let fence = state.completed.next().ok_or(RenderError::Overflow {
            what: "fence value",
        })?;
        let ops = plan::check(&state, &self.caps, &list)?;
        for op in ops {
            state.apply(op)?;
        }
        state.completed = fence;
        state.log.push(Event::Submitted(fence, list));
        state.free_passed();
        drop(state);
        self.completed_moved.notify_all();
        Ok(fence)
    }

    fn completed(&self) -> FenceValue {
        self.state().completed
    }

    fn wait(&self, value: FenceValue, timeout: Duration) -> Result<WaitOutcome, RenderError> {
        let state = self.state();
        let (state, _) = self
            .completed_moved
            .wait_timeout_while(state, timeout, |s| s.completed < value)
            .unwrap_or_else(PoisonError::into_inner);
        Ok(if state.completed >= value {
            WaitOutcome::Reached
        } else {
            WaitOutcome::TimedOut
        })
    }

    fn take_readback(&self, id: ReadbackId) -> Option<Readback> {
        self.state().readbacks.remove(&id)
    }

    fn set_present_target(&self, target: PresentTarget) -> Result<(), RenderError> {
        self.state().log.push(Event::Target(target));
        Ok(())
    }

    fn take_present_feedback(&self) -> Vec<PresentFeedback> {
        std::mem::take(&mut self.state().feedback)
    }
}

#[cfg(test)]
mod tests;
