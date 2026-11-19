//! The capture builder (spec, "Fixtures and CI"): runs the real front and back over hand-written
//! method streams ([`crate::fifo::PushBuffer`], G1's encoder after Mesa's `nv_push.h`) on one
//! thread, recording every input each thread consumes, and gives the [`Capture`].
//!
//! # How it runs
//!
//! The builder owns a fresh GPU: a [`Front`] and an [`Executor`] over the recording backend,
//! with production memory ([`VmMemory`] over a [`GpuVm`] and a [`FlatMemory`]) wrapped in
//! [`RecordingMemory`] per thread. Address spaces and channels are made first; the first
//! submission, update, barrier, present, destroy or run starts the capture, which saves each
//! address table and the syncpoints as the starting state. A CPU write
//! ([`CaptureBuilder::cpu_write`]) goes to guest memory at once, unrecorded: the threads record
//! what they read of it.
//!
//! [`CaptureBuilder::run`] turns the front until it is idle, and after each idle turn hands the
//! back every entry the front pushed, closes the open list and runs completion processing: the
//! same three steps the back thread takes from `Fences::wait` (completed work, entries, a close
//! for the empty ring), here in a fixed order so a capture is the same on every run. Reports
//! the back makes reach the front at its next turn. A channel waiting on the acquire poll gets
//! [`MAX_IDLE_POLLS`] more polls, the front's clock moved on by the poll's sleep each time.
//!
//! Before each turn the builder records, in the front log, the reports the front will take, the
//! syncpoints the back changed and the waiters it latched since the front's last turn; before
//! each step of the back, the waiter table if it changed since the back's last step. Those are
//! the inputs one thread takes from the other through shared state rather than through memory.
//!
//! The back's write watch is never armed (no kernel takes its requests), so every range the
//! buffer cache uses is streamed and no drain finds anything: no capture made here has a
//! `Drain` record.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use starquake_memory::{FlatMemory, PageRange, WriteWatch};
use starquake_render::Backend;
use starquake_render::caps::Caps;
use starquake_render::recording::RecordingBackend;
use starquake_shader::lookup::{EngineState, ProgramMemory, ProgramRequest, Ticket, VariantKey};
use thiserror::Error;

use super::drive::{CollectError, Collector, Discard, MAX_COLLECTED_WORDS, Retired, entry_hash};
use super::format::{
    BackRecord, Capture, CaptureError, CapturedChannel, ContentHash, FrontRecord, Header,
    MAX_CAPTURE_BYTES, Output, StateRecord,
};
use super::record::{
    RecordError, Recorder, RecordingFenceClock, RecordingFrontClock, RecordingMemory,
    RecordingShaders, Thread,
};
use crate::back::buffer::InvalidationSink;
use crate::back::draw::WaitWhileCompiling;
use crate::back::draw::shaders::{SetLookup, ShaderLookup};
use crate::back::exec::{ExecConfig, ExecError, Executor, ShaderCache};
use crate::back::fence::{
    CloseReason, FenceClock, FenceEnv, Fences, FrontWake, Mode, Report, Waiter, WaiterTable,
};
use crate::front::{
    CreateError, Front, FrontClock, FrontDeps, FrontError, FrontShared, OrderedTable, Turn,
};
use crate::interface::{
    AsId, ChannelConfig, ChannelId, PresentRequest, SYNCPOINT_COUNT, Submission, Submitted,
    SyncpointId, Target, Ticket as GpuTicket, VmUpdate,
};
use crate::memory::{GpuFault, GpuMemory, VmMemory};
use crate::ring::{RING_WORDS, ring};
use crate::syncpoint::Syncpoints;
use crate::vm::{GpuVm, VmError};

/// The timestamp the back's clock gives every release that writes one: a fixed value, so a
/// capture is the same on every run.
pub const BUILDER_TIMESTAMP: u64 = 0x5147_4300_0000_0001;

/// The most front turns one [`CaptureBuilder::run`] takes.
pub const MAX_RUN_TURNS: usize = 1 << 16;

/// The polls a channel waiting on one gets in [`CaptureBuilder::run`] once nothing else moves.
pub const MAX_IDLE_POLLS: u32 = 8;

/// How long the builder waits for the ring's thread to pop what the front published: a bound
/// that only a stuck thread reaches.
pub const COLLECT_PATIENCE: Duration = Duration::from_secs(10);

/// What the builder is made with.
#[derive(Clone, Debug)]
pub struct BuilderConfig {
    /// The header's build id.
    pub build_id: String,
    /// The recording backend's capabilities.
    pub caps: Caps,
    /// The recording backend's memory budget.
    pub memory_allowed: u64,
    /// The CPU address of the guest memory's first byte.
    pub guest_base: u64,
    /// The guest memory's size in bytes.
    pub guest_size: usize,
    /// The executor's wait-while-compiling setting.
    pub setting: WaitWhileCompiling,
    /// The fence path's mode.
    pub mode: Mode,
}

impl BuilderConfig {
    /// A configuration for synthetic fixtures: `guest_size` bytes of guest memory at
    /// `guest_base`, a device that supports everything with limits large enough for that
    /// memory (every buffer and staging size the guest's size), a budget of 1 GiB, the default
    /// wait setting and fenced releases.
    pub fn synthetic(build_id: &str, guest_base: u64, guest_size: usize) -> Self {
        use starquake_render::caps::{
            DynamicState, Features, FormatFeatures, FormatTable, Limits, SampleCounts,
            StageSupport,
        };
        let size = u64::try_from(guest_size).unwrap_or(u64::MAX);
        Self {
            build_id: build_id.to_owned(),
            caps: Caps {
                limits: Limits {
                    max_buffer_size: size,
                    max_staging_size: size,
                    max_texture_dimension_1d: 4096,
                    max_texture_dimension_2d: 4096,
                    max_texture_dimension_3d: 256,
                    max_texture_array_layers: 64,
                    max_sampler_anisotropy: 16,
                    max_vertex_attributes: 16,
                    max_vertex_buffers: 8,
                    max_vertex_buffer_stride: 2048,
                    max_color_attachments: 8,
                    max_viewports: 1,
                    max_bind_groups: 4,
                    max_bindings_per_bind_group: 16,
                    max_compute_workgroups_per_dimension: 64,
                    max_uniform_buffers_per_stage: 12,
                    max_storage_buffers_per_stage: 8,
                    max_sampled_textures_per_stage: 16,
                    max_storage_textures_per_stage: 4,
                    max_samplers_per_stage: 16,
                },
                features: Features::ALL,
                stages: StageSupport::ALL,
                dynamic_state: DynamicState::ALL,
                sample_counts: SampleCounts::X1,
                view_swizzle: true,
                formats: FormatTable::uniform(FormatFeatures::ALL),
            },
            memory_allowed: 1 << 30,
            guest_base,
            guest_size,
            setting: WaitWhileCompiling::Wait,
            mode: Mode::Fenced,
        }
    }
}

/// Why the builder failed.
#[derive(Debug, Error)]
pub enum BuildError {
    /// An address space or channel was added after the capture started.
    #[error("address spaces and channels are made before the capture starts")]
    Started,
    /// An address space was added twice.
    #[error("address space {} was already made", .0.0)]
    DuplicateSpace(AsId),
    /// A mapping was refused.
    #[error("the mapping was refused: {0}")]
    Map(#[from] VmError),
    /// A channel was refused.
    #[error("the channel was refused: {0}")]
    Channel(#[from] CreateError),
    /// A submission did not fit its channel's queue.
    #[error("channel {}'s queue is full", .0.0)]
    Full(ChannelId),
    /// A CPU write faulted.
    #[error("the CPU write faulted: {0}")]
    Cpu(#[from] GpuFault),
    /// The front failed.
    #[error("the front failed: {0}")]
    Front(#[from] FrontError),
    /// The back refused a setup call.
    #[error("the back failed: {0}")]
    Back(#[from] ExecError),
    /// The ring could not be made or its thread could not start.
    #[error("the ring failed: {0}")]
    Ring(String),
    /// The ring lost entries.
    #[error(transparent)]
    Collect(#[from] CollectError),
    /// The ring's thread did not pop what the front published within [`COLLECT_PATIENCE`].
    #[error("the ring's thread did not take what the front published")]
    Stalled,
    /// A run took more than [`MAX_RUN_TURNS`] turns.
    #[error("the GPU did not go idle within {MAX_RUN_TURNS} front turns")]
    TooManyTurns,
    /// A state snapshot could not be taken.
    #[error("a starting-state snapshot failed: {0}")]
    Snapshot(String),
    /// The recording failed.
    #[error(transparent)]
    Record(#[from] RecordError),
    /// The capture does not encode.
    #[error(transparent)]
    Capture(#[from] CaptureError),
}

/// A shader cache whose every lookup is pending and never finishes: a draw or dispatch is
/// skipped. The builder's default.
#[derive(Clone, Copy, Debug, Default)]
pub struct PendingShaders;

impl ShaderLookup for PendingShaders {
    fn lookup(
        &mut self,
        _requests: &[ProgramRequest],
        _variant: &VariantKey,
        _memory: &dyn ProgramMemory,
        _state: &dyn EngineState,
    ) -> SetLookup {
        SetLookup::Pending(Ticket::new(0))
    }

    fn wait(&mut self, ticket: Ticket) -> SetLookup {
        SetLookup::Pending(ticket)
    }
}

impl InvalidationSink for PendingShaders {
    fn invalidate(&mut self, _range: PageRange) {}
}

/// The front's clock in the builder: still, except when the builder moves it for a poll.
#[derive(Default)]
struct ManualClock(AtomicU64);

impl ManualClock {
    fn advance(&self, by: Duration) {
        let nanos = super::format::nanos(by).max(1);
        let _previous = self
            .0
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |now| {
                Some(now.saturating_add(nanos))
            });
    }
}

impl FrontClock for ManualClock {
    fn now(&self) -> Duration {
        Duration::from_nanos(self.0.load(Ordering::SeqCst))
    }
}

/// The back's clock in the builder: never moves, so no list closes for the time it recorded.
struct StillClock;

impl FenceClock for StillClock {
    fn elapsed(&self) -> Duration {
        Duration::ZERO
    }

    fn gpu_timestamp(&self) -> u64 {
        BUILDER_TIMESTAMP
    }
}

/// The front's wake-up as the back sees it: keeps each report for the front log, then wakes the
/// front.
struct RecordingWake {
    shared: Arc<FrontShared>,
    pending: Mutex<Vec<FrontRecord>>,
}

impl FrontWake for RecordingWake {
    fn wake(&self, report: &Report) {
        self.pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(FrontRecord::Report {
                markers: report.markers.clone(),
                released: report.released > 0 || report.latched || !report.failed.is_empty(),
                incremented: report.incremented > 0,
            });
        self.shared.wake(report);
    }
}

impl RecordingWake {
    fn take(&self) -> Vec<FrontRecord> {
        std::mem::take(&mut *self.pending.lock().unwrap_or_else(PoisonError::into_inner))
    }

    fn is_empty(&self) -> bool {
        self.pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty()
    }
}

/// The syncpoint values and the waiter table, as a thread last saw them.
#[derive(Clone, PartialEq, Eq)]
struct View {
    values: Vec<u32>,
    waiters: Vec<(Waiter, bool)>,
}

/// One address space: its table, and the memory the CPU's writes go through.
struct Space {
    id: AsId,
    vm: Arc<GpuVm>,
    cpu: VmMemory<FlatMemory>,
}

/// Builds a capture by running the real GPU on one thread (module docs).
pub struct CaptureBuilder {
    config: BuilderConfig,
    guest: Arc<FlatMemory>,
    syncpoints: Arc<Syncpoints>,
    recorder: Arc<Recorder>,
    shared: Arc<FrontShared>,
    front: Option<Front>,
    collector: Option<Collector>,
    executor: Option<Executor>,
    backend: Arc<RecordingBackend>,
    waiters: Arc<WaiterTable>,
    wake: Arc<RecordingWake>,
    clock: Arc<ManualClock>,
    retired: Arc<Retired>,
    spaces: Vec<Space>,
    channels: Vec<CapturedChannel>,
    state: Option<Vec<StateRecord>>,
    ring_hashes: Vec<ContentHash>,
    applied: u64,
    front_view: View,
    back_view: Vec<(Waiter, bool)>,
}

impl CaptureBuilder {
    /// A fresh GPU over `config.guest_size` bytes of guest memory at `config.guest_base`, asking
    /// `shaders` for shaders (recorded; [`PendingShaders`] when nothing draws).
    pub fn new(config: BuilderConfig, shaders: Box<dyn ShaderCache>) -> Result<Self, BuildError> {
        let guest = Arc::new(FlatMemory::new(config.guest_base, config.guest_size));
        let syncpoints = Arc::new(Syncpoints::new(Arc::new(Discard)));
        let recorder = Arc::new(Recorder::new(MAX_CAPTURE_BYTES));
        let shared = Arc::new(FrontShared::new());
        let (producer, consumer) =
            ring(RING_WORDS).map_err(|error| BuildError::Ring(error.to_string()))?;
        let collector = Collector::start(consumer, MAX_COLLECTED_WORDS)
            .map_err(|error| BuildError::Ring(error.to_string()))?;
        let waiters = Arc::new(WaiterTable::new());
        let ordered = Arc::new(OrderedTable::new());
        let clock = Arc::new(ManualClock::default());
        let retired = Arc::new(Retired::default());
        let front = Front::new(FrontDeps {
            shared: Arc::clone(&shared),
            ring: producer,
            syncpoints: Arc::clone(&syncpoints),
            completions: Arc::new(Discard),
            waiters: Arc::clone(&waiters),
            ordered: Arc::clone(&ordered),
            clock: Arc::new(RecordingFrontClock::new(
                Arc::<ManualClock>::clone(&clock),
                Arc::clone(&recorder),
            )),
        })
        .with_retired(Arc::<Retired>::clone(&retired));
        let backend = Arc::new(RecordingBackend::new(
            config.caps.clone(),
            config.memory_allowed,
        ));
        let wake = Arc::new(RecordingWake {
            shared: Arc::clone(&shared),
            pending: Mutex::new(Vec::new()),
        });
        let fences = Fences::new(
            Arc::<RecordingBackend>::clone(&backend),
            FenceEnv {
                syncpoints: Arc::clone(&syncpoints),
                waiters: Arc::clone(&waiters),
                completions: Arc::new(Discard),
                clock: Arc::new(RecordingFenceClock::new(
                    Arc::new(StillClock),
                    Arc::clone(&recorder),
                )),
                front: Arc::<RecordingWake>::clone(&wake),
            },
            config.mode,
        );
        let executor = Executor::new(
            Arc::<RecordingBackend>::clone(&backend),
            fences,
            Box::new(RecordingShaders::new(shaders, Arc::clone(&recorder))),
            ordered,
            ExecConfig {
                setting: config.setting,
                ..ExecConfig::default()
            },
        );
        let front_view = View {
            values: vec![0; SYNCPOINT_COUNT as usize],
            waiters: Vec::new(),
        };
        Ok(Self {
            config,
            guest,
            syncpoints,
            recorder,
            shared,
            front: Some(front),
            collector: Some(collector),
            executor: Some(executor),
            backend,
            waiters,
            wake,
            clock,
            retired,
            spaces: Vec::new(),
            channels: Vec::new(),
            state: None,
            ring_hashes: Vec::new(),
            applied: 0,
            front_view,
            back_view: Vec::new(),
        })
    }

    fn front(&mut self) -> Result<&mut Front, BuildError> {
        self.front
            .as_mut()
            .ok_or_else(|| BuildError::Ring("the front has stopped".to_owned()))
    }

    fn executor(&mut self) -> Result<&mut Executor, BuildError> {
        self.executor
            .as_mut()
            .ok_or_else(|| BuildError::Ring("the back has stopped".to_owned()))
    }

    /// Adds address space `space`, mapping `gpu` onto guest memory at CPU address `cpu_va`. The
    /// first space made is the one presents read through, as in the wiring.
    pub fn address_space(
        &mut self,
        space: AsId,
        gpu: PageRange,
        cpu_va: u64,
    ) -> Result<(), BuildError> {
        if self.state.is_some() {
            return Err(BuildError::Started);
        }
        if self.spaces.iter().any(|known| known.id == space) {
            return Err(BuildError::DuplicateSpace(space));
        }
        let vm = Arc::new(GpuVm::new());
        vm.map(
            gpu,
            Target::Guest { cpu_va },
            crate::interface::Kind(0),
        )?;
        let front_memory = VmMemory::new(
            Arc::clone(&vm),
            Arc::clone(&self.guest),
            Arc::clone(&self.syncpoints),
        );
        let back_memory = front_memory.for_thread();
        let cpu = front_memory.for_thread();
        let recorder = Arc::clone(&self.recorder);
        self.shared.create_address_space(
            space,
            Arc::clone(&vm),
            Arc::new(RecordingMemory::new(
                Arc::new(front_memory),
                Arc::clone(&recorder),
                Thread::Front,
                space,
            )),
        );
        let back: Arc<dyn GpuMemory> = Arc::new(RecordingMemory::new(
            Arc::new(back_memory),
            recorder,
            Thread::Back,
            space,
        ));
        let first = self.spaces.is_empty();
        let executor = self.executor()?;
        executor.add_space(space, back, Arc::new(WriteWatch::new()))?;
        if first {
            executor.set_present_space(space)?;
        }
        self.spaces.push(Space { id: space, vm, cpu });
        Ok(())
    }

    /// Adds a channel in `space`, incrementing `syncpoint`.
    pub fn channel(
        &mut self,
        space: AsId,
        syncpoint: SyncpointId,
    ) -> Result<ChannelId, BuildError> {
        if self.state.is_some() {
            return Err(BuildError::Started);
        }
        let config = ChannelConfig { space, syncpoint };
        let id = self.shared.create_channel(config)?;
        self.executor()?.add_channel(id, space)?;
        self.channels.push(CapturedChannel { id, config });
        Ok(id)
    }

    /// Writes `data` at GPU address `va` of `space` as the CPU would: at once, unrecorded.
    pub fn cpu_write(&self, space: AsId, va: u64, data: &[u8]) -> Result<(), BuildError> {
        let memory = self
            .spaces
            .iter()
            .find(|known| known.id == space)
            .map(|known| &known.cpu)
            .ok_or(GpuFault::Unmapped { va })?;
        memory.write(va, data)?;
        Ok(())
    }

    /// Reads `buf.len()` bytes at GPU address `va` of `space` as the CPU would, unrecorded.
    pub fn cpu_read(&self, space: AsId, va: u64, buf: &mut [u8]) -> Result<(), BuildError> {
        let memory = self
            .spaces
            .iter()
            .find(|known| known.id == space)
            .map(|known| &known.cpu)
            .ok_or(GpuFault::Unmapped { va })?;
        memory.read(va, buf)?;
        Ok(())
    }

    /// Starts the capture: saves the starting state (module docs).
    fn start(&mut self) -> Result<(), BuildError> {
        if self.state.is_some() {
            return Ok(());
        }
        let mut state = Vec::new();
        for space in &self.spaces {
            let bytes = space
                .vm
                .save()
                .map_err(|error| BuildError::Snapshot(error.to_string()))?;
            state.push(StateRecord::AddressTable {
                space: space.id,
                blob: self.recorder.blob(&bytes),
            });
        }
        let bytes = self
            .syncpoints
            .save()
            .map_err(|error| BuildError::Snapshot(error.to_string()))?;
        state.push(StateRecord::Syncpoints {
            blob: self.recorder.blob(&bytes),
        });
        self.front_view = self.view();
        self.back_view = self.waiters.snapshot();
        self.state = Some(state);
        Ok(())
    }

    /// Queues `submission` on `channel`, as gsv would.
    pub fn submit(&mut self, channel: ChannelId, submission: Submission) -> Result<(), BuildError> {
        self.start()?;
        if self.shared.submit(channel, submission.clone()) == Submitted::Full {
            return Err(BuildError::Full(channel));
        }
        self.recorder.front(FrontRecord::Submit {
            channel,
            submission,
        });
        Ok(())
    }

    /// Queues an ordered update of `space`'s table.
    pub fn update_vm(
        &mut self,
        space: AsId,
        update: VmUpdate,
        ticket: Option<GpuTicket>,
    ) -> Result<(), BuildError> {
        self.start()?;
        self.shared.update_vm(space, update, ticket);
        self.recorder.front(FrontRecord::Update {
            space,
            update,
            ticket,
        });
        Ok(())
    }

    /// Queues a barrier.
    pub fn barrier(&mut self, ticket: GpuTicket) -> Result<(), BuildError> {
        self.start()?;
        self.shared.barrier(ticket);
        self.recorder.front(FrontRecord::Barrier { ticket });
        Ok(())
    }

    /// Queues a present.
    pub fn present(&mut self, request: PresentRequest) -> Result<(), BuildError> {
        self.start()?;
        self.shared.present(request);
        self.recorder.front(FrontRecord::Present { request });
        Ok(())
    }

    /// Destroys `channel`.
    pub fn destroy_channel(&mut self, channel: ChannelId) -> Result<(), BuildError> {
        self.start()?;
        self.shared.destroy_channel(channel);
        self.recorder.front(FrontRecord::Destroy { channel });
        Ok(())
    }

    /// The syncpoints the GPU increments.
    pub fn syncpoints(&self) -> &Arc<Syncpoints> {
        &self.syncpoints
    }

    fn view(&self) -> View {
        View {
            values: (0..SYNCPOINT_COUNT)
                .filter_map(|id| u8::try_from(id).ok().and_then(SyncpointId::new))
                .map(|id| self.syncpoints.read(id))
                .collect(),
            waiters: self.waiters.snapshot(),
        }
    }

    /// Records what the front takes from the back's side before a turn (module docs).
    fn before_turn(&mut self) {
        for report in self.wake.take() {
            self.recorder.front(report);
        }
        let now = self.view();
        for ((index, &value), &seen) in now.values.iter().enumerate().zip(&self.front_view.values)
        {
            if value != seen
                && let Some(id) = u8::try_from(index).ok().and_then(SyncpointId::new)
            {
                self.recorder.front(FrontRecord::Syncpoint { id, value });
            }
        }
        for &(waiter, latched) in &now.waiters {
            if latched && self.front_view.waiters.contains(&(waiter, false)) {
                self.recorder.front(FrontRecord::Latch {
                    channel: waiter.channel,
                });
            }
        }
    }

    /// Records the waiter table before a step of the back if it changed since the back's last.
    fn before_back_step(&mut self) {
        let now = self.waiters.snapshot();
        if now != self.back_view {
            self.recorder.back(BackRecord::Waiters { waiters: now });
        }
    }

    /// One round of the back (module docs); whether it did anything.
    fn back_round(&mut self) -> Result<bool, BuildError> {
        let entries = self
            .collector
            .as_ref()
            .map_or_else(Vec::new, Collector::drain);
        let mut progress = false;
        for entry in entries {
            self.ring_hashes
                .push(entry_hash(&entry).unwrap_or(ContentHash::new(0)));
            for channel in self.retired.due(self.applied) {
                let _removed = self.executor()?.remove_channel(channel);
            }
            self.before_back_step();
            self.recorder.back(BackRecord::Apply);
            // A refused entry is logged and counted; a replay refuses it the same way.
            let _applied = self.executor()?.apply(&entry);
            self.applied = self.applied.saturating_add(1);
            self.back_view = self.waiters.snapshot();
            progress = true;
        }
        if self.executor()?.fences().open_commands() > 0 {
            self.before_back_step();
            self.recorder.back(BackRecord::Close);
            let _closed = self.executor()?.close(CloseReason::RingEmpty);
            self.back_view = self.waiters.snapshot();
            progress = true;
        }
        if self.executor()?.fences().lists_in_flight() > 0 {
            self.before_back_step();
            let completed = self.backend.completed().get();
            self.recorder.back(BackRecord::Complete { completed });
            self.executor()?.complete();
            self.back_view = self.waiters.snapshot();
            progress = true;
        }
        Ok(progress)
    }

    /// Runs the GPU until it is idle (module docs, "How it runs").
    pub fn run(&mut self) -> Result<(), BuildError> {
        self.start()?;
        let mut idle_polls = 0;
        for _ in 0..MAX_RUN_TURNS {
            self.before_turn();
            self.recorder.front(FrontRecord::Turn);
            let turn = self.front()?.turn()?;
            self.front_view = self.view();
            let park = match turn {
                Turn::Worked => continue,
                Turn::Stopped => return Ok(()),
                Turn::Idle { park } => park,
            };
            let sent = self.front()?.sent();
            let taken = self
                .collector
                .as_ref()
                .is_some_and(|collector| collector.wait_taken(sent, COLLECT_PATIENCE));
            if !taken {
                return Err(BuildError::Stalled);
            }
            if self.collector.as_ref().is_some_and(Collector::overflowed) {
                return Err(BuildError::Collect(CollectError::Overflow));
            }
            if self.back_round()? || !self.wake.is_empty() {
                idle_polls = 0;
                continue;
            }
            if self.front()?.waits_on_poll() && idle_polls < MAX_IDLE_POLLS {
                self.clock.advance(park);
                idle_polls += 1;
                continue;
            }
            return Ok(());
        }
        Err(BuildError::TooManyTurns)
    }

    /// Stops the GPU and gives the capture: the header, the starting state, both logs, the
    /// blobs, a ring hash for every entry the front pushed, and the render command stream.
    pub fn finish(mut self) -> Result<Capture, BuildError> {
        self.start()?;
        self.shared.stop();
        if let Some(mut front) = self.front.take() {
            // A stopping front publishes what it pushed and reads nothing.
            front.turn()?;
        }
        let (rest, ended) = match self.collector.take() {
            Some(collector) => collector.finish(),
            None => (Vec::new(), Ok(())),
        };
        ended?;
        for entry in &rest {
            self.ring_hashes
                .push(entry_hash(entry).unwrap_or(ContentHash::new(0)));
        }
        let render = self.backend.snapshot();
        let caps = postcard::to_allocvec(&self.config.caps)
            .map_err(|error| BuildError::Snapshot(error.to_string()))?;
        let caps = self.recorder.blob(&caps);
        let render = self.recorder.blob(render.as_bytes());
        drop(self.executor.take());
        let recorded = self.recorder.take()?;
        let mut outputs: Vec<Output> = self
            .ring_hashes
            .iter()
            .map(|&hash| Output::Ring { hash })
            .collect();
        outputs.push(Output::Render { text: render });
        let capture = Capture {
            header: Header {
                build_id: self.config.build_id.clone(),
                caps,
                memory_allowed: self.config.memory_allowed,
                setting: self.config.setting,
                mode: self.config.mode,
                spaces: self.spaces.iter().map(|space| space.id).collect(),
                channels: self.channels.clone(),
            },
            state: self.state.take().unwrap_or_default(),
            front: recorded.front,
            back: recorded.back,
            blobs: recorded.blobs,
            outputs,
        };
        // Every capture the builder gives reads back.
        capture.to_bytes()?;
        Ok(capture)
    }
}
