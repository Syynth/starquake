//! `WgpuBackend`: the [`Backend`] trait on a wgpu device (docs/specs/render-backend.md, slice R5).
//!
//! # Threads
//!
//! Calls come from any thread (the GPU back thread; the compile workers run pipeline builds,
//! which take no lock of this module, `crate::pipeline`). Each takes the one state lock,
//! briefly: a `create_*` checks its description, allocates the id in the
//! submit-side tables ([`crate::plan::Tables`]) and queues the creation; `submit` checks the
//! whole list ([`crate::plan`]), takes the next fence value and queues it. The render thread
//! (`starquake-render`, [`crate::thread`]) takes the queue in order, creates host objects,
//! encodes, submits to wgpu's one queue and settles completed lists (decision 3). It holds no
//! lock while it calls wgpu, so wgpu's callbacks, which take the lock, run while it polls.
//!
//! # The bounded queue
//!
//! `submit` waits while [`WgpuConfig::queue_depth`] lists are submitted and unfinished
//! ([`crate::config::queue_full`]): that is the back-pressure that slows the guest when the host
//! GPU falls behind (render-backend, "Command lists are data"). It waits for nothing else.
//!
//! # Fences and the completion wake
//!
//! A list's fence value is its place in submit order. The render thread asks wgpu to report when
//! the submission holding the list has finished (`Queue::on_submitted_work_done`, the owner's
//! decision of 2026-11-12). Each readback buffer is mapped (`Buffer::map_async`) right after the
//! list's last submit, and wgpu completes the mapping only once that submission has finished,
//! so the bytes are read after the list completes ("Buffers and uploads"). The thread copies
//! out the list's readbacks, and then, under the lock, stores
//! the readbacks, the present feedback and the new [`Backend::completed`] value together, and
//! frees the destroys that value lets go. Only after the lock is released does it call the
//! [`CompletionWake`], once for all the lists that finished together, and then hand dumped
//! frames to their sink (render-backend, "The completion wake"; decision 14). So a back thread
//! the wake rouses reads the new value or a later one, and a readback is there to take once its
//! list's fence has passed.
//!
//! # Destroys
//!
//! `destroy(id, after)` frees at once when `after` has passed, and otherwise when the render
//! thread stores a completed value at or past it. Freeing makes the id stale for every later
//! call (its slot moves to the next generation) and queues the host object's release behind
//! every list already queued, so a list checked while the id was live is encoded with the object
//! still there. A destroy of an id that is not live, or is already waiting, is refused and
//! counted ([`WgpuBackend::refused_destroys`]). The waiting destroys are indexed by id and by
//! fence value ([`Pending`]): a `destroy` looks its id up instead of scanning them, and a
//! settle frees the ones due, in fence order and in call order among equal values, without
//! visiting the others.
//!
//! # Lost device
//!
//! When wgpu reports the device lost (or the render thread stops by a panic), the `create_*`
//! calls for buffers, textures, views, samplers, shaders and pipelines (each after the checks of
//! its description that need no lock), `submit`, and a `wait` for a value not yet
//! reached return [`RenderError::DeviceLost`]; a `submit` waiting for room and every `wait`
//! stop waiting; and the wake is called so the back thread looks. The run ends there
//! (render-backend, "Errors and untrusted input": exit code 2, which `core` maps). Lists still
//! queued or in flight never complete.
//!
//! # Dropping
//!
//! Dropping the backend sets the shutdown flag and joins the render thread, which waits up to
//! [`thread::SHUTDOWN_WAIT`] for lists in flight. When the drop runs on the render thread itself
//! (a completion wake or a frame sink held the last handle), it does not join: the thread sees
//! the flag once the wake or sink returns and ends by itself.
//!
//! # No hidden inputs
//!
//! The backend's results depend on the trait calls, the adapter and the [`WgpuConfig`] only: it
//! reads no environment variable (wgpu's `from_env` instance options are not used) and no file
//! (render-backend, "Render capture v2", "What R5 and R6 must do now"). Present feedback carries
//! host time, which is never guest-visible.
//!
//! # What the trait cannot report
//!
//! Every refusal the backend decides is the `Err` of the call that asked. Errors wgpu itself
//! reports later, on the render thread, would mean a check here missed one of wgpu's rules: the
//! backend logs each and counts it ([`WgpuBackend::host_errors`]); the tests assert none occur.
//! A readback whose buffer wgpu fails to map is stored as [`Readback::Unknown`] and counted the
//! same way.

use crate::adapter::AdapterReport;
use crate::caps::{ByteLimits, FormatSupport, caps, device_features};
use crate::config::{WgpuConfig, queue_full};
use crate::convert::{TextureSpec, buffer_usages, sampler_descriptor, view_descriptor};
use crate::pipeline::{
    self as pipelines, Compile, ComputeBuild, Counters, DrawPipeline, GraphicsBuild,
    NO_COMPILE_POOL, PipelineCell, PipelineCounts, Registry, ShaderRec,
};
use crate::plan::{self, Tables, TextureRec, ViewRec};
use crate::rules;
use crate::thread::{self, Create, ListWork, Message};
use starquake_render::validate;
use starquake_render::{
    Backend, BufferDesc, BufferId, Caps, CommandList, CompilePool, CompletionWake,
    ComputePipelineDesc, FenceValue, FrameDump, FrameSink, GraphicsPipelineDesc, MemoryBudget,
    PipelineId, PipelineStatus, PresentFeedback, PresentTarget, Readback, ReadbackId, RenderError,
    ResourceId, SamplerDesc, SamplerId, ShaderDesc, ShaderId, StagingSlice, TextureDesc, TextureId,
    ViewDesc, ViewId, WaitOutcome, WhileCompiling,
};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::future::Future;
use std::pin::pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use thiserror::Error;

/// How long [`WgpuBackend::new`] waits for wgpu to find an adapter or open a device. Native
/// wgpu answers at once; the bound only keeps a broken driver from hanging the caller.
pub const OPEN_TIMEOUT: Duration = Duration::from_secs(30);

/// Why the backend could not be made.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum NewError {
    /// wgpu found no adapter on this host.
    #[error("no GPU adapter on this host: {0}")]
    NoAdapter(String),
    /// The adapter would not open a device with its own features and limits.
    #[error("the GPU adapter would not open a device: {0}")]
    Device(String),
    /// wgpu did not answer within [`OPEN_TIMEOUT`].
    #[error("the GPU adapter did not answer in time")]
    Timeout,
    /// The render thread could not be started.
    #[error("the render thread could not be started: {0}")]
    Thread(String),
}

/// Runs a wgpu future to its end on this thread, up to [`OPEN_TIMEOUT`].
pub(crate) fn block_on<F: Future>(future: F) -> Result<F::Output, NewError> {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    let start = Instant::now();
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return Ok(value);
        }
        if start.elapsed() >= OPEN_TIMEOUT {
            return Err(NewError::Timeout);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Destroys waiting for their fence value, indexed by id (`contains` is a lookup) and by fence
/// value (`take_due` visits only the entries it returns, and the first one it leaves).
#[derive(Debug)]
pub(crate) struct Pending<K> {
    /// The ids waiting.
    waiting: BTreeSet<K>,
    /// The ids waiting for each fence value, in call order.
    by_fence: BTreeMap<FenceValue, Vec<K>>,
}

impl<K> Default for Pending<K> {
    fn default() -> Self {
        Self {
            waiting: BTreeSet::new(),
            by_fence: BTreeMap::new(),
        }
    }
}

impl<K: Ord + Copy> Pending<K> {
    /// Whether `id` is waiting.
    pub(crate) fn contains(&self, id: &K) -> bool {
        self.waiting.contains(id)
    }

    /// Makes `id` wait for `after`; returns `false`, changing nothing, when it is waiting already.
    pub(crate) fn push(&mut self, id: K, after: FenceValue) -> bool {
        if !self.waiting.insert(id) {
            return false;
        }
        self.by_fence.entry(after).or_default().push(id);
        true
    }

    /// Takes the ids whose fence value is at or below `completed`, in fence order and in call
    /// order among equal values.
    pub(crate) fn take_due(&mut self, completed: FenceValue) -> Vec<K> {
        let mut due = Vec::new();
        while let Some(entry) = self.by_fence.first_entry() {
            if *entry.key() > completed {
                break;
            }
            for id in entry.remove() {
                self.waiting.remove(&id);
                due.push(id);
            }
        }
        due
    }

    /// How many ids are waiting.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.waiting.len()
    }
}

/// A [`ResourceId`] ordered by kind, index and generation, for [`Pending`]'s index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PendingId(pub(crate) ResourceId);

impl PendingId {
    /// The fields the order compares: one value per id, so equal keys are equal ids.
    fn key(self) -> (u8, u32, u32) {
        match self.0 {
            ResourceId::Buffer(i) => (0, i.index(), i.generation()),
            ResourceId::Texture(i) => (1, i.index(), i.generation()),
            ResourceId::View(i) => (2, i.index(), i.generation()),
            ResourceId::Sampler(i) => (3, i.index(), i.generation()),
            ResourceId::Shader(i) => (4, i.index(), i.generation()),
            ResourceId::Pipeline(i) => (5, i.index(), i.generation()),
        }
    }
}

impl Ord for PendingId {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.key().cmp(&other.key())
    }
}

impl PartialOrd for PendingId {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

/// Everything the submitting side keeps, under one lock.
#[derive(Debug, Default)]
pub(crate) struct State {
    /// The live resources.
    pub(crate) tables: Tables,
    /// The live shaders and pipelines (`crate::pipeline`).
    pub(crate) pipelines: Registry,
    /// Destroys waiting for their fence value.
    pub(crate) pending: Pending<PendingId>,
    /// The last fence value handed out.
    pub(crate) submitted: FenceValue,
    /// The highest fence value finished.
    pub(crate) completed: FenceValue,
    /// Work for the render thread, in order.
    pub(crate) queue: VecDeque<Message>,
    /// Readbacks waiting to be taken.
    pub(crate) readbacks: BTreeMap<ReadbackId, Readback>,
    /// Readback ids waiting to be taken or in flight.
    pub(crate) reserved: BTreeSet<ReadbackId>,
    /// Feedback not yet taken.
    pub(crate) feedback: Vec<PresentFeedback>,
    /// Dumped frames, for lists submitted with no sink installed.
    pub(crate) dumps: Vec<FrameDump>,
    /// Where dumped frames of lists submitted from now on go.
    pub(crate) sink: Option<Arc<dyn FrameSink>>,
    /// Presents of finished lists.
    pub(crate) presents: u64,
    /// Destroys refused.
    pub(crate) refused_destroys: u64,
    /// The device was lost (or the render thread stopped).
    pub(crate) lost: bool,
    /// The backend is being dropped.
    pub(crate) shutdown: bool,
}

impl State {
    /// Frees `id` now: its slot moves on and its host object's release is queued.
    fn free(&mut self, id: ResourceId) {
        let freed = match id {
            ResourceId::Buffer(i) => self
                .tables
                .buffers
                .remove(i.index(), i.generation())
                .is_some(),
            ResourceId::Texture(i) => self
                .tables
                .textures
                .remove(i.index(), i.generation())
                .is_some(),
            ResourceId::View(i) => self
                .tables
                .views
                .remove(i.index(), i.generation())
                .is_some(),
            ResourceId::Sampler(i) => self
                .tables
                .samplers
                .remove(i.index(), i.generation())
                .is_some(),
            // Shaders and pipelines have no host object on the render thread: a build that
            // still needs one holds it (`crate::pipeline`).
            ResourceId::Shader(i) => {
                self.pipelines.shaders.remove(i.index(), i.generation());
                false
            }
            ResourceId::Pipeline(i) => {
                self.pipelines.pipelines.remove(i.index(), i.generation());
                false
            }
        };
        if freed {
            self.queue.push_back(Message::Free(id));
        }
    }

    fn is_live(&self, id: ResourceId) -> bool {
        let t = &self.tables;
        match id {
            ResourceId::Buffer(i) => t.buffers.get(i.index(), i.generation()).is_some(),
            ResourceId::Texture(i) => t.textures.get(i.index(), i.generation()).is_some(),
            ResourceId::View(i) => t.views.get(i.index(), i.generation()).is_some(),
            ResourceId::Sampler(i) => t.samplers.get(i.index(), i.generation()).is_some(),
            ResourceId::Shader(i) => self
                .pipelines
                .shaders
                .get(i.index(), i.generation())
                .is_some(),
            ResourceId::Pipeline(i) => self
                .pipelines
                .pipelines
                .get(i.index(), i.generation())
                .is_some(),
        }
    }

    /// Frees every waiting destroy whose fence value has passed, in fence order and in call
    /// order among equal values ([`Pending::take_due`]).
    pub(crate) fn free_passed(&mut self) {
        for PendingId(id) in self.pending.take_due(self.completed) {
            self.free(id);
        }
    }
}

/// What the calling threads and the render thread share.
#[derive(Debug)]
pub(crate) struct Shared {
    state: Mutex<State>,
    /// The render thread waits here for work.
    pub(crate) work: Condvar,
    /// `submit` and `wait` wait here for a completed value to move (or the device to be lost).
    pub(crate) moved: Condvar,
    /// Called after each advance of the completed value.
    pub(crate) wake: CompletionWake,
    /// Errors wgpu reported that the checks did not catch, and failed readback maps.
    pub(crate) host_errors: AtomicU64,
    /// Scratch textures the render thread has made for rectangle clears (`crate::thread`).
    pub(crate) scratch_textures: AtomicU64,
}

impl Shared {
    pub(crate) fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Marks the device lost, releases every waiter, and wakes the back thread so it looks.
    pub(crate) fn lose(&self) {
        self.state().lost = true;
        self.work.notify_all();
        self.moved.notify_all();
        self.wake.wake();
    }

    /// Counts and logs an error wgpu reported on the render thread.
    pub(crate) fn host_error(&self, what: &dyn std::fmt::Display) {
        self.host_errors.fetch_add(1, Ordering::Relaxed);
        tracing::error!("wgpu reported an error the backend's checks did not catch: {what}");
    }
}

/// The wgpu backend (module docs).
#[derive(Debug)]
pub struct WgpuBackend {
    caps: Caps,
    formats: FormatSupport,
    /// The device features, for the rules that depend on them.
    features: wgpu::Features,
    /// Whether views may reinterpret their texture's format (`DownlevelFlags::VIEW_FORMATS`).
    reinterpret: bool,
    /// Whether depth and stencil texels may be copied to and from buffers
    /// (`DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES`).
    depth_copies: bool,
    config: WgpuConfig,
    report: AdapterReport,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    /// The device, for the pipeline builds on the compile pool (`crate::pipeline`).
    device: wgpu::Device,
    /// The compile pool and the wait-while-compiling setting, if the backend was given them.
    compile: Option<Compile>,
    /// Whether the device runs every subgroup 32 wide (`pipeline::subgroups_honoured`).
    subgroups: bool,
    /// Draws skipped for their pipelines.
    counters: Counters,
}

impl WgpuBackend {
    /// Opens the host's default high-performance adapter, builds `Caps` from it, and starts the
    /// render thread. `wake` is called after each advance of [`Backend::completed`]. A backend
    /// made this way has no compile pool: it creates shaders but refuses every pipeline
    /// ([`NO_COMPILE_POOL`]); `core` uses [`WgpuBackend::with_compile_pool`].
    pub fn new(config: WgpuConfig, wake: CompletionWake) -> Result<Self, NewError> {
        Self::open(config, wake, None)
    }

    /// [`WgpuBackend::new`], with the shared compile pool that builds its pipelines and the GPU
    /// crate's wait-while-compiling setting, which the render thread applies to a draw whose
    /// pipeline is still compiling (`crate::pipeline`; render-backend, "The trait": `core`
    /// constructs each backend with the shared compile pool, the wait setting and the completion
    /// wake).
    pub fn with_compile_pool(
        config: WgpuConfig,
        wake: CompletionWake,
        pool: Arc<dyn CompilePool>,
        setting: WhileCompiling,
    ) -> Result<Self, NewError> {
        Self::open(config, wake, Some(Compile { pool, setting }))
    }

    fn open(
        config: WgpuConfig,
        wake: CompletionWake,
        compile: Option<Compile>,
    ) -> Result<Self, NewError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..wgpu::RequestAdapterOptions::default()
        }))?
        .map_err(|e| NewError::NoAdapter(e.to_string()))?;
        let limits = adapter.limits();
        let downlevel = adapter.get_downlevel_capabilities().flags;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("starquake"),
            required_features: device_features(adapter.features()),
            required_limits: limits.clone(),
            ..wgpu::DeviceDescriptor::default()
        }))?
        .map_err(|e| NewError::Device(e.to_string()))?;
        let features = device.features();
        let specific = features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
        let describe = |format: wgpu::TextureFormat| {
            if specific {
                adapter.get_texture_format_features(format)
            } else {
                format.guaranteed_format_features(features)
            }
        };
        let formats = FormatSupport::new(features, describe);
        let caps = caps(
            features,
            &device.limits(),
            downlevel,
            &formats,
            ByteLimits {
                memory: config.memory_limit,
                staging: config.max_staging_size,
            },
        );
        let info = adapter.get_info();
        let report = AdapterReport::new(&info, adapter.features(), &limits);
        let subgroups =
            pipelines::subgroups_honoured(features, info.subgroup_min_size, info.subgroup_max_size);
        let host_device = device.clone();
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            work: Condvar::new(),
            moved: Condvar::new(),
            wake,
            host_errors: AtomicU64::new(0),
            scratch_textures: AtomicU64::new(0),
        });
        let thread = thread::start(device, queue, Arc::clone(&shared))
            .map_err(|e| NewError::Thread(e.to_string()))?;
        Ok(Self {
            caps,
            formats,
            features,
            reinterpret: downlevel.contains(wgpu::DownlevelFlags::VIEW_FORMATS),
            depth_copies: downlevel.contains(wgpu::DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES),
            config,
            report,
            shared,
            thread: Some(thread),
            device: host_device,
            compile,
            subgroups,
            counters: Counters::default(),
        })
    }

    /// What the adapter reported (verification 1).
    pub fn adapter_report(&self) -> &AdapterReport {
        &self.report
    }

    /// How many presents have finished on the offscreen target.
    pub fn presents(&self) -> u64 {
        self.shared.state().presents
    }

    /// How many errors wgpu reported that the backend's checks did not catch, plus readbacks
    /// whose buffer wgpu failed to map (module docs). Zero unless the backend has a bug.
    pub fn host_errors(&self) -> u64 {
        self.shared.host_errors.load(Ordering::Relaxed)
    }

    /// What a draw gets for pipeline `id`, under the wait-while-compiling setting the backend
    /// was built with (`crate::pipeline`, "Draws: wait or skip"). Under
    /// [`WhileCompiling::Wait`] it blocks while the pipeline compiles, until its build settles;
    /// a skipped draw is counted in [`WgpuBackend::pipeline_counts`]. The render thread calls it
    /// for each draw (slice R7). `Err` for an id that names no live pipeline.
    pub fn draw_pipeline(&self, id: PipelineId) -> Result<DrawPipeline, RenderError> {
        let cell = self.pipeline_cell(id)?;
        let setting = self
            .compile
            .as_ref()
            .map_or(WhileCompiling::default(), |c| c.setting);
        Ok(pipelines::acquire(&cell, setting, &self.counters))
    }

    /// The draws skipped for their pipelines so far.
    pub fn pipeline_counts(&self) -> PipelineCounts {
        self.counters.get()
    }

    /// Whether the device runs every subgroup 32 wide, so pipelines whose shaders use subgroups
    /// are accepted (`crate::pipeline`, "Requirements").
    pub fn honours_subgroups(&self) -> bool {
        self.subgroups
    }

    fn pipeline_cell(&self, id: PipelineId) -> Result<Arc<PipelineCell>, RenderError> {
        self.shared
            .state()
            .pipelines
            .pipelines
            .get(id.index(), id.generation())
            .cloned()
            .ok_or(RenderError::Invalid(validate::NO_PIPELINE))
    }

    /// How many `destroy` calls named no live resource or one already waiting to be destroyed.
    pub fn refused_destroys(&self) -> u64 {
        self.shared.state().refused_destroys
    }

    /// Installs the sink that receives the frames presented with `dump` set, or removes it
    /// with `None`. It applies to lists submitted from now on; frames of lists submitted with no
    /// sink wait for [`WgpuBackend::take_frame_dumps`]. The sink is called on the render thread,
    /// after the list's completed value is stored and the wake called, with no lock held, in
    /// present order.
    pub fn set_frame_sink(&self, sink: Option<Arc<dyn FrameSink>>) {
        // Released before the old sink drops: its `Drop` may call the backend.
        let replaced = std::mem::replace(&mut self.shared.state().sink, sink);
        drop(replaced);
    }

    /// The frames presented with `dump` set, by lists submitted with no sink, since the last
    /// call, in order.
    pub fn take_frame_dumps(&self) -> Vec<FrameDump> {
        std::mem::take(&mut self.shared.state().dumps)
    }

    /// Locks the state, refusing every call once the device is lost.
    fn live_state(&self) -> Result<MutexGuard<'_, State>, RenderError> {
        let state = self.shared.state();
        if state.lost {
            Err(RenderError::DeviceLost)
        } else {
            Ok(state)
        }
    }

    /// Queues `message` and tells the render thread.
    fn queue(&self, mut state: MutexGuard<'_, State>, message: Message) {
        state.queue.push_back(message);
        drop(state);
        self.shared.work.notify_one();
    }

    /// How many scratch textures the render thread has made for rectangle clears.
    #[cfg(test)]
    pub(crate) fn scratch_textures(&self) -> u64 {
        self.shared.scratch_textures.load(Ordering::Relaxed)
    }

    /// The device the pipeline builds use.
    #[cfg(test)]
    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Makes the render thread lose the device, as a driver failure would.
    #[cfg(test)]
    pub(crate) fn lose_device_for_test(&self) {
        let state = self.shared.state();
        self.queue(state, Message::LoseDevice);
    }
}

/// Whether a backend dropped on `current` should join its render thread `render`: not when they
/// are the same thread, which happens when a completion wake or a frame sink, called on the
/// render thread, holds the last handle to the backend. That thread then sees the shutdown flag
/// when the call returns and ends by itself.
fn should_join(render: std::thread::ThreadId, current: std::thread::ThreadId) -> bool {
    render != current
}

impl Drop for WgpuBackend {
    fn drop(&mut self) {
        self.shared.state().shutdown = true;
        self.shared.work.notify_all();
        let Some(thread) = self.thread.take() else {
            return;
        };
        if !should_join(thread.thread().id(), std::thread::current().id()) {
            return;
        }
        if thread.join().is_err() {
            tracing::error!("the render thread ended by a panic");
        }
    }
}

/// Bytes a texture's host object takes: every level, every sample (the recording backend's
/// rule, so both budgets agree).
fn texture_bytes(desc: &TextureDesc) -> Result<u64, RenderError> {
    let mut total: u64 = 0;
    for mip in 0..desc.mip_levels {
        let extent = desc.mip_extent(mip).ok_or(RenderError::NotBelow {
            what: "mip level",
            index: mip.into(),
            count: desc.mip_levels.into(),
        })?;
        let level = validate::packed_size(desc.format, extent)?
            .checked_mul(desc.sample_count.into())
            .ok_or(RenderError::Overflow {
                what: "texture size",
            })?;
        total = total.saturating_add(level);
    }
    Ok(total)
}

impl Backend for WgpuBackend {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    /// `allowed` is the configured limit (verification 8 is open); `used` is what every live
    /// buffer and texture takes by its description, as the recording backend counts it.
    fn budget(&self) -> MemoryBudget {
        let state = self.shared.state();
        let buffers = state.tables.buffers.iter().map(|(_, _, b)| b.size);
        let textures = state.tables.textures.iter().map(|(_, _, t)| t.bytes);
        MemoryBudget {
            allowed: self.config.memory_limit,
            used: buffers.chain(textures).fold(0, u64::saturating_add),
        }
    }

    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferId, RenderError> {
        validate::validate_buffer(&self.caps, desc)?;
        let size = rules::host_buffer_size(desc.size).ok_or(RenderError::Overflow {
            what: "buffer size",
        })?;
        let mut state = self.live_state()?;
        let (index, generation) = state.tables.buffers.insert(*desc)?;
        let id = BufferId::new(index, generation);
        let usage = buffer_usages(desc.usage);
        self.queue(state, Message::Create(Create::Buffer { id, size, usage }));
        Ok(id)
    }

    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureId, RenderError> {
        validate::validate_texture(&self.caps, desc)?;
        let host = rules::check_texture(&self.caps, &self.formats, self.features, desc)?;
        let bytes = texture_bytes(desc)?;
        let allowed = self.formats.usages(desc.format);
        let spec = TextureSpec::new(desc, host, allowed, self.reinterpret)
            .ok_or(RenderError::Unsupported(rules::VIEW_FORMAT))?;
        let mut state = self.live_state()?;
        let (index, generation) = state.tables.textures.insert(TextureRec {
            desc: *desc,
            host,
            bytes,
        })?;
        let id = TextureId::new(index, generation);
        self.queue(state, Message::Create(Create::Texture { id, spec }));
        Ok(id)
    }

    fn create_view(&self, texture: TextureId, desc: &ViewDesc) -> Result<ViewId, RenderError> {
        let mut state = self.live_state()?;
        let tex = state
            .tables
            .textures
            .get(texture.index(), texture.generation())
            .ok_or(RenderError::Invalid(plan::NO_TEXTURE))?;
        validate::validate_view(&self.caps, &tex.desc, desc)?;
        rules::check_view(&tex.desc, desc, self.reinterpret)?;
        let view = view_descriptor(desc).ok_or(RenderError::Unsupported(rules::VIEW_FORMAT))?;
        let (index, generation) = state.tables.views.insert(ViewRec {
            texture,
            desc: *desc,
        })?;
        let id = ViewId::new(index, generation);
        self.queue(state, Message::Create(Create::View { id, texture, view }));
        Ok(id)
    }

    fn create_sampler(&self, desc: &SamplerDesc) -> Result<SamplerId, RenderError> {
        validate::validate_sampler(&self.caps, desc)?;
        rules::check_sampler(desc)?;
        let sampler = sampler_descriptor(desc);
        let mut state = self.live_state()?;
        let (index, generation) = state.tables.samplers.insert(*desc)?;
        let id = SamplerId::new(index, generation);
        self.queue(state, Message::Create(Create::Sampler { id, sampler }));
        Ok(id)
    }

    /// Checks the shader and keeps its module (`crate::pipeline`, "Shaders"); wgpu first sees
    /// it when a pipeline that uses it is built.
    fn create_shader(&self, desc: ShaderDesc) -> Result<ShaderId, RenderError> {
        validate::validate_shader(&self.caps, &desc)?;
        let rec = ShaderRec::new(desc)?;
        let mut state = self.live_state()?;
        let (index, generation) = state.pipelines.shaders.insert(Arc::new(rec))?;
        Ok(ShaderId::new(index, generation))
    }

    fn destroy(&self, resource: ResourceId, after: FenceValue) {
        let mut state = self.shared.state();
        let waiting = state.pending.contains(&PendingId(resource));
        if !state.is_live(resource) || waiting {
            state.refused_destroys = state.refused_destroys.saturating_add(1);
        } else if after <= state.completed {
            state.free(resource);
            drop(state);
            self.shared.work.notify_one();
        } else {
            state.pending.push(PendingId(resource), after);
        }
    }

    /// Checks the description in the order `crate::pipeline` lists, then queues the build on
    /// the compile pool and returns at once.
    fn create_graphics_pipeline(
        &self,
        desc: GraphicsPipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        let compile = self
            .compile
            .clone()
            .ok_or(RenderError::Unsupported(NO_COMPILE_POOL))?;
        validate::validate_graphics_pipeline(&self.caps, &desc)?;
        let mut state = self.live_state()?;
        let shader = |state: &State, id: ShaderId| {
            state
                .pipelines
                .shaders
                .get(id.index(), id.generation())
                .cloned()
                .ok_or(RenderError::Invalid(validate::NO_SHADER))
        };
        validate::validate_pipeline_stages(&desc.stages, |id| shader(&state, id).map(|s| s.stage))?;
        let vertex = shader(&state, desc.stages.vertex)?;
        let fragment = desc
            .stages
            .fragment
            .map(|id| shader(&state, id))
            .transpose()?;
        let shaders = || std::iter::once(&*vertex).chain(fragment.as_deref());
        pipelines::check_requirements(shaders(), self.subgroups)?;
        let layout = pipelines::plan_layout(shaders())?;
        let plan = pipelines::state::graphics(&desc)?;
        let cell = Arc::new(PipelineCell::new());
        let (index, generation) = state.pipelines.pipelines.insert(Arc::clone(&cell))?;
        drop(state);
        let build = GraphicsBuild {
            device: self.device.clone(),
            vertex,
            fragment,
            layout,
            plan,
        };
        compile
            .pool
            .spawn(pipelines::job(cell, move || build.run()));
        Ok(PipelineId::new(index, generation))
    }

    /// As [`Backend::create_graphics_pipeline`], for a compute shader.
    fn create_compute_pipeline(
        &self,
        desc: ComputePipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        let compile = self
            .compile
            .clone()
            .ok_or(RenderError::Unsupported(NO_COMPILE_POOL))?;
        let mut state = self.live_state()?;
        let shader = state
            .pipelines
            .shaders
            .get(desc.shader.index(), desc.shader.generation())
            .cloned();
        validate::validate_compute_stage(&desc, |_| {
            shader
                .as_ref()
                .map(|s| s.stage)
                .ok_or(RenderError::Invalid(validate::NO_SHADER))
        })?;
        let shader = shader.ok_or(RenderError::Invalid(validate::NO_SHADER))?;
        pipelines::check_requirements([&*shader], self.subgroups)?;
        let layout = pipelines::plan_layout([&*shader])?;
        let cell = Arc::new(PipelineCell::new());
        let (index, generation) = state.pipelines.pipelines.insert(Arc::clone(&cell))?;
        drop(state);
        let build = ComputeBuild {
            device: self.device.clone(),
            shader,
            layout,
        };
        compile
            .pool
            .spawn(pipelines::job(cell, move || build.run()));
        Ok(PipelineId::new(index, generation))
    }

    /// The pipeline's status; [`PipelineStatus::Failed`] for an id that names no live pipeline
    /// (the trait has no error here).
    fn pipeline_status(&self, id: PipelineId) -> PipelineStatus {
        self.pipeline_cell(id)
            .map_or(PipelineStatus::Failed, |cell| cell.status())
    }

    fn staging(&self, size: u64) -> Result<StagingSlice, RenderError> {
        validate::validate_staging_size(&self.caps, size)?;
        let len = usize::try_from(size).map_err(|_| RenderError::Overflow {
            what: "staging size",
        })?;
        Ok(StagingSlice::from_bytes(vec![0; len]))
    }

    /// Waits while the queue is full (module docs), then checks the whole list; a refused list
    /// changes nothing.
    fn submit(&self, list: CommandList) -> Result<FenceValue, RenderError> {
        let depth = self.config.queue_depth;
        let state = self.live_state()?;
        let mut state = self
            .shared
            .moved
            .wait_while(state, |s| {
                !s.lost && queue_full(s.submitted.get(), s.completed.get(), depth)
            })
            .unwrap_or_else(PoisonError::into_inner);
        if state.lost {
            return Err(RenderError::DeviceLost);
        }
        let fence = state.submitted.next().ok_or(RenderError::Overflow {
            what: "fence value",
        })?;
        let checked = plan::check(
            &state.tables,
            &self.caps,
            self.depth_copies,
            &state.reserved,
            list,
        )?;
        state.reserved.extend(checked.readbacks.iter().copied());
        state.submitted = fence;
        let sink = state.sink.clone();
        self.queue(
            state,
            Message::List(ListWork {
                fence,
                steps: checked.steps,
                sink,
            }),
        );
        Ok(fence)
    }

    fn completed(&self) -> FenceValue {
        self.shared.state().completed
    }

    fn wait(&self, value: FenceValue, timeout: Duration) -> Result<WaitOutcome, RenderError> {
        let state = self.shared.state();
        let (state, _) = self
            .shared
            .moved
            .wait_timeout_while(state, timeout, |s| !s.lost && s.completed < value)
            .unwrap_or_else(PoisonError::into_inner);
        if state.completed >= value {
            Ok(WaitOutcome::Reached)
        } else if state.lost {
            Err(RenderError::DeviceLost)
        } else {
            Ok(WaitOutcome::TimedOut)
        }
    }

    fn take_readback(&self, id: ReadbackId) -> Option<Readback> {
        let mut state = self.shared.state();
        let readback = state.readbacks.remove(&id)?;
        state.reserved.remove(&id);
        Some(readback)
    }

    /// Accepts the offscreen target, the only one before slice R8.
    fn set_present_target(&self, target: PresentTarget) -> Result<(), RenderError> {
        match target {
            PresentTarget::Offscreen => Ok(()),
            // `PresentTarget` is non-exhaustive: windows arrive with slice R8.
            _ => Err(RenderError::Unsupported(
                "the wgpu backend presents offscreen only until slice R8",
            )),
        }
    }

    fn take_present_feedback(&self) -> Vec<PresentFeedback> {
        std::mem::take(&mut self.shared.state().feedback)
    }
}

#[cfg(test)]
pub(crate) mod tests;
