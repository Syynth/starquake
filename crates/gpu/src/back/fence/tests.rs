//! Tests of the fence path, against a render backend that completes lists only when a test says
//! so ([`LateBackend`]), a fake GPU memory over flat guest memory, and recording doubles for the
//! completion queue and the front's wake-up. Every wait is bounded.

mod production;
mod waiters;

use std::collections::VecDeque;
use std::fmt::{self, Write as _};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::Duration;

use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE, PageRange};
use starquake_render::caps::{DynamicState, StageSupport};
use starquake_render::command::Marker;
use starquake_render::{
    Backend, BufferDesc, BufferId, Caps, Command, CommandList, CompletionWake, ComputePipelineDesc,
    Features, FenceValue, FormatFeatures, FormatTable, GraphicsPipelineDesc, Limits, MemoryBudget,
    PipelineId, PipelineStatus, PresentFeedback, PresentTarget, Readback, ReadbackId,
    RecordingBackend, RenderError, ResourceId, SampleCounts, SamplerDesc, SamplerId, ShaderDesc,
    ShaderId, StagingSlice, TextureDesc, TextureId, ViewDesc, ViewId, WaitOutcome,
    WakeContractError, WakeCounter, check_completion_wake,
};

use super::*;
use crate::interface::{AsId, ChannelId, Kind, Target};
use crate::memory::TestGpuMemory;
use crate::ring::{ClassId, ring};
use crate::sema::{
    AcquireOp, RELEASE_TIMESTAMP_OFFSET, Reduction, ReductionFormat, ReductionOp, ReleaseForm,
};

// ---------------------------------------------------------------------------------------------
// The rig
// ---------------------------------------------------------------------------------------------

/// The CPU side of the test memory.
const CPU_BASE: u64 = 0x8000_0000;
/// Pages of guest memory behind the GPU mapping.
const PAGES: u64 = 4;
/// Where the memory is mapped on the GPU: above 32 bits, so a truncated address misses it.
const GPU_BASE: u64 = 0x1_2345_0000_0000;
/// One past the mapped GPU range.
const GPU_END: u64 = GPU_BASE + PAGES * PAGE_SIZE;
/// A semaphore.
const SEM: u64 = GPU_BASE + 0x100;
/// A second semaphore, on another page.
const SEM2: u64 = GPU_BASE + PAGE_SIZE + 0x40;
/// The address space the rig's releases are issued in and its waiters belong to: not 0, so a
/// hard-coded space fails.
const SPACE: AsId = AsId(0x5a);
/// A GPU address nothing maps.
const UNMAPPED: u64 = 0x7_0000_0000;
/// The byte the guest memory is filled with, so bytes a release must not touch are visible.
const FILL: u8 = 0xee;
/// What the fake clock's timestamps read.
const TIMESTAMP: u64 = 0x1122_3344_5566_7788;
/// Long enough for any of these tests' threads to make progress; only reached when broken.
const PATIENCE: Duration = Duration::from_secs(10);
/// A wait that only a wake or an entry can end before [`PATIENCE`].
const FOREVER: Duration = Duration::from_secs(1000);

type Memory = TestGpuMemory<FlatMemory>;

fn memory() -> Arc<Memory> {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mem = TestGpuMemory::new(FlatMemory::new(CPU_BASE, size));
    mem.map(
        PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(),
        CPU_BASE,
    )
    .unwrap();
    mem.guest().write(CPU_BASE, &vec![FILL; size]).unwrap();
    Arc::new(mem)
}

fn read(mem: &Memory, va: u64, len: usize) -> Vec<u8> {
    let mut bytes = vec![0; len];
    mem.read(va, &mut bytes).unwrap();
    bytes
}

fn word(mem: &Memory, va: u64) -> u32 {
    let mut bytes = [0; 4];
    mem.read(va, &mut bytes).unwrap();
    u32::from_le_bytes(bytes)
}

/// Runs `body` on its own thread and returns its result, failing the test after [`PATIENCE`].
fn within<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    let (done, result) = mpsc::channel();
    thread::spawn(move || {
        let _ = done.send(body());
    });
    result
        .recv_timeout(PATIENCE)
        .expect("the fence test did not finish in time")
}

/// A settable clock whose timestamps read [`TIMESTAMP`].
#[derive(Default)]
struct Clock(Mutex<Duration>);

impl Clock {
    fn set(&self, now: Duration) {
        *self.0.lock().unwrap() = now;
    }
}

impl FenceClock for Clock {
    fn elapsed(&self) -> Duration {
        *self.0.lock().unwrap()
    }
    fn gpu_timestamp(&self) -> u64 {
        TIMESTAMP
    }
}

/// gsv's completion queue, recorded.
#[derive(Default)]
struct Pushed(Mutex<Vec<GpuCompletion>>);

impl Pushed {
    fn all(&self) -> Vec<GpuCompletion> {
        self.0.lock().unwrap().clone()
    }
}

impl Completions for Pushed {
    fn push(&self, completion: GpuCompletion) {
        self.0.lock().unwrap().push(completion);
    }
}

/// The front's wake-up, recorded.
#[derive(Default)]
struct Front(Mutex<Vec<Report>>);

impl Front {
    fn all(&self) -> Vec<Report> {
        self.0.lock().unwrap().clone()
    }
}

impl FrontWake for Front {
    fn wake(&self, report: &Report) {
        self.0.lock().unwrap().push(report.clone());
    }
}

fn caps() -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: 1 << 16,
            max_staging_size: 1 << 12,
            max_texture_dimension_1d: 256,
            max_texture_dimension_2d: 256,
            max_texture_dimension_3d: 64,
            max_texture_array_layers: 8,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 1,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 16,
            max_compute_workgroups_per_dimension: 65_535,
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
    }
}

/// What [`LateBackend`] refuses a list with.
const REFUSED: RenderError = RenderError::Invalid("the late test backend refused this list");

#[derive(Default)]
struct LateState {
    completed: FenceValue,
    /// Lists submitted so far.
    count: u64,
    /// Fence values to hand out instead of the count, first first.
    script: VecDeque<u64>,
    refuse_next: bool,
    submitted: Vec<FenceValue>,
}

/// A recording backend whose lists complete only when the test calls [`LateBackend::finish`],
/// which stores the new completed value and then calls the completion wake with no lock held,
/// as render-backend's "The completion wake" requires of every backend. It passes the same
/// contract check as the recording backend (`the_late_backend_meets_the_wake_contract`).
///
/// Where it differs from the recording backend, on purpose: completion is the test's call; fence
/// values can be scripted out of order ([`LateBackend::script`]); [`LateBackend::refuse_next`]
/// refuses one list (after which the next is accepted); and [`LateBackend::finish_silently`]
/// completes without the wake, to play a backend that breaks the contract.
struct LateBackend {
    inner: RecordingBackend,
    state: Mutex<LateState>,
    moved: Condvar,
    wake: CompletionWake,
}

impl LateBackend {
    fn new(wake: CompletionWake) -> Self {
        Self {
            inner: RecordingBackend::new(caps(), 0),
            state: Mutex::new(LateState::default()),
            moved: Condvar::new(),
            wake,
        }
    }

    fn state(&self) -> MutexGuard<'_, LateState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn script(&self, fences: &[u64]) {
        self.state().script.extend(fences);
    }

    fn refuse_next(&self) {
        self.state().refuse_next = true;
    }

    fn submitted(&self) -> Vec<FenceValue> {
        self.state().submitted.clone()
    }

    fn set_completed(&self, fence: FenceValue) {
        self.state().completed = fence;
        self.moved.notify_all();
    }

    /// Completes everything up to `fence`, then wakes.
    fn finish(&self, fence: FenceValue) {
        self.set_completed(fence);
        self.wake.wake();
    }

    /// Completes everything up to `fence` and forgets the wake.
    fn finish_silently(&self, fence: FenceValue) {
        self.set_completed(fence);
    }
}

impl Backend for LateBackend {
    fn caps(&self) -> &Caps {
        self.inner.caps()
    }
    fn budget(&self) -> MemoryBudget {
        self.inner.budget()
    }
    fn create_buffer(&self, desc: &BufferDesc) -> Result<BufferId, RenderError> {
        self.inner.create_buffer(desc)
    }
    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureId, RenderError> {
        self.inner.create_texture(desc)
    }
    fn create_view(&self, texture: TextureId, desc: &ViewDesc) -> Result<ViewId, RenderError> {
        self.inner.create_view(texture, desc)
    }
    fn create_sampler(&self, desc: &SamplerDesc) -> Result<SamplerId, RenderError> {
        self.inner.create_sampler(desc)
    }
    fn create_shader(&self, desc: ShaderDesc) -> Result<ShaderId, RenderError> {
        self.inner.create_shader(desc)
    }
    fn destroy(&self, resource: ResourceId, after: FenceValue) {
        self.inner.destroy(resource, after);
    }
    fn create_graphics_pipeline(
        &self,
        desc: GraphicsPipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        self.inner.create_graphics_pipeline(desc)
    }
    fn create_compute_pipeline(
        &self,
        desc: ComputePipelineDesc,
    ) -> Result<PipelineId, RenderError> {
        self.inner.create_compute_pipeline(desc)
    }
    fn pipeline_status(&self, id: PipelineId) -> PipelineStatus {
        self.inner.pipeline_status(id)
    }
    fn staging(&self, size: u64) -> Result<StagingSlice, RenderError> {
        self.inner.staging(size)
    }
    fn submit(&self, list: CommandList) -> Result<FenceValue, RenderError> {
        let mut state = self.state();
        if std::mem::take(&mut state.refuse_next) {
            return Err(REFUSED);
        }
        self.inner.submit(list)?;
        state.count += 1;
        let fence = FenceValue::new(state.script.pop_front().unwrap_or(state.count));
        state.submitted.push(fence);
        Ok(fence)
    }
    fn completed(&self) -> FenceValue {
        self.state().completed
    }
    fn wait(&self, value: FenceValue, timeout: Duration) -> Result<WaitOutcome, RenderError> {
        let state = self.state();
        let (state, _) = self
            .moved
            .wait_timeout_while(state, timeout, |s| s.completed < value)
            .unwrap_or_else(PoisonError::into_inner);
        Ok(if state.completed >= value {
            WaitOutcome::Reached
        } else {
            WaitOutcome::TimedOut
        })
    }
    fn take_readback(&self, id: ReadbackId) -> Option<Readback> {
        self.inner.take_readback(id)
    }
    fn set_present_target(&self, target: PresentTarget) -> Result<(), RenderError> {
        self.inner.set_present_target(target)
    }
    fn take_present_feedback(&self) -> Vec<PresentFeedback> {
        self.inner.take_present_feedback()
    }
}

/// The fence path and everything around it.
struct Rig {
    fences: Fences,
    backend: Arc<LateBackend>,
    mem: Arc<Memory>,
    syncpoints: Arc<Syncpoints>,
    waiters: Arc<WaiterTable>,
    pushed: Arc<Pushed>,
    front: Arc<Front>,
    clock: Arc<Clock>,
}

fn rig_on(backend: Arc<LateBackend>, mode: Mode) -> Rig {
    let pushed = Arc::new(Pushed::default());
    let syncpoints = Arc::new(Syncpoints::new(Arc::<Pushed>::clone(&pushed)));
    let waiters = Arc::new(WaiterTable::new());
    let front = Arc::new(Front::default());
    let clock = Arc::new(Clock::default());
    let env = FenceEnv {
        syncpoints: Arc::clone(&syncpoints),
        waiters: Arc::clone(&waiters),
        completions: Arc::<Pushed>::clone(&pushed),
        clock: Arc::<Clock>::clone(&clock),
        front: Arc::<Front>::clone(&front),
    };
    let fences = Fences::new(Arc::<LateBackend>::clone(&backend), env, mode);
    Rig {
        fences,
        backend,
        mem: memory(),
        syncpoints,
        waiters,
        pushed,
        front,
        clock,
    }
}

fn rig() -> Rig {
    rig_on(
        Arc::new(LateBackend::new(CompletionWake::none())),
        Mode::Fenced,
    )
}

/// A command for the open list; the fence path never looks inside one.
fn command(submit: u64) -> Command {
    Command::Marker(Marker { submit, draw: 0 })
}

fn four(address: u64, payload: u64) -> Release {
    Release {
        address,
        payload,
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: false,
        },
        op: ReleaseOp::Write,
    }
}

fn reduce(address: u64, payload: u64, op: ReductionOp) -> Release {
    Release {
        op: ReleaseOp::Reduce(Reduction {
            op,
            format: ReductionFormat::Unsigned,
        }),
        ..four(address, payload)
    }
}

impl Rig {
    fn action(&self, release: Release) -> FencedAction {
        FencedAction::Release {
            memory: Arc::<Memory>::clone(&self.mem),
            space: SPACE,
            release,
        }
    }

    /// Records one command, attaches a 4-byte release, and closes the list.
    fn list_with_release(&mut self, address: u64, payload: u64) -> FenceValue {
        self.fences.record(command(1)).unwrap();
        let action = self.action(four(address, payload));
        assert_eq!(self.fences.attach(action), Ok(Attached::Open));
        self.fences.close(CloseReason::Present).unwrap().unwrap()
    }
}

fn syncpoint(id: u8) -> SyncpointId {
    SyncpointId::new(id).unwrap()
}

fn released(count: u64) -> Report {
    Report {
        released: count,
        ..Report::default()
    }
}

// ---------------------------------------------------------------------------------------------
// Log capture (the same subscriber as the macro host's tests)
// ---------------------------------------------------------------------------------------------

/// A subscriber that keeps every event's level and fields.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<(tracing::Level, String)>>>);

struct Fields<'a>(&'a mut String);

impl tracing::field::Visit for Fields<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        write!(self.0, "{}={value:?};", field.name()).unwrap();
    }
}

impl tracing::Subscriber for Capture {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut fields = String::new();
        event.record(&mut Fields(&mut fields));
        self.0
            .lock()
            .unwrap()
            .push((*event.metadata().level(), fields));
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

fn captured<T>(f: impl FnOnce() -> T) -> (T, Vec<(tracing::Level, String)>) {
    let capture = Capture::default();
    let result = tracing::subscriber::with_default(capture.clone(), f);
    let events = capture.0.lock().unwrap().clone();
    (result, events)
}

/// The events at `level`, as captured.
fn at(events: &[(tracing::Level, String)], level: tracing::Level) -> Vec<String> {
    events
        .iter()
        .filter(|(l, _)| *l == level)
        .map(|(_, fields)| fields.clone())
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Constants and defaults
// ---------------------------------------------------------------------------------------------

#[test]
fn the_close_interval_is_the_specs_guess() {
    // gpu-protocol, "Releases and completion": "or 1 ms of recording has passed (guess)".
    assert_eq!(CLOSE_AFTER, Duration::from_millis(1));
}

#[test]
fn a_new_fence_path_has_nothing_recorded_in_flight_or_counted() {
    let r = rig();
    assert_eq!(r.fences.open_commands(), 0);
    assert_eq!(r.fences.lists_in_flight(), 0);
    assert_eq!(
        r.fences.counters(),
        FenceCounters {
            fenced: 0,
            skipped_empty: 0,
            closed: ClosedCounts {
                ring_empty: 0,
                present: 0,
                wait_for_idle: 0,
                waiter: 0,
                elapsed: 0,
                set_reference: 0,
            },
            lost_wakes: 0,
            failed_releases: 0,
        }
    );
    assert!(Report::default().is_empty());
}

#[test]
fn errors_read_as_plain_words() {
    let failure = ReleaseFailure {
        address: UNMAPPED,
        fault: GpuFault::Unmapped { va: UNMAPPED },
    };
    assert_eq!(
        failure.to_string(),
        "the release at GPU address 0x700000000 was not performed: GPU address 0x700000000 is \
         not mapped"
    );
    assert_eq!(
        FenceError::Submit(REFUSED).to_string(),
        "the render backend refused a command list: the late test backend refused this list"
    );
    assert_eq!(
        FenceError::Ring(PopError::Disconnected).to_string(),
        "the ring is empty and its producer is gone"
    );
}

#[test]
fn a_report_with_any_one_thing_in_it_is_not_empty() {
    let failure = ReleaseFailure {
        address: UNMAPPED,
        fault: GpuFault::Unmapped { va: UNMAPPED },
    };
    let cases = [
        released(1),
        Report {
            incremented: 1,
            ..Report::default()
        },
        Report {
            latched: true,
            ..Report::default()
        },
        Report {
            markers: vec![MarkerToken(0)],
            ..Report::default()
        },
        Report {
            failed: vec![failure],
            ..Report::default()
        },
    ];
    for report in cases {
        assert!(!report.is_empty(), "{report:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Nothing before completion; first in, first out
// ---------------------------------------------------------------------------------------------

#[test]
fn nothing_is_visible_before_its_list_completes() {
    let mut r = rig();
    let watched = syncpoint(5);
    r.syncpoints.watch(watched);
    let vm = Arc::new(GpuVm::new());
    let vm_base = 0x40_0000_0000;
    vm.map(
        PageRange::new(vm_base, 2 * PAGE_SIZE).unwrap(),
        Target::Guest { cpu_va: CPU_BASE },
        Kind(0),
    )
    .unwrap();
    let mapped = vm.mappings();

    r.fences.record(command(1)).unwrap();
    let release = r.action(four(SEM, 0xaabb_ccdd));
    assert_eq!(r.fences.attach(release), Ok(Attached::Open));
    assert_eq!(
        r.fences.attach(FencedAction::Increment(watched)),
        Ok(Attached::Open)
    );
    let marker = FencedAction::Marker {
        token: MarkerToken(7),
        ordered: Some(OrderedUpdate {
            change: Some(VmChange {
                vm: Arc::clone(&vm),
                update: VmUpdate::Unmap {
                    range: PageRange::new(vm_base, PAGE_SIZE).unwrap(),
                },
            }),
            ticket: Some(Ticket(9)),
        }),
    };
    assert_eq!(r.fences.attach(marker), Ok(Attached::Open));
    let fence = r.fences.close(CloseReason::Present).unwrap().unwrap();

    // In flight: nothing of the three is visible yet.
    assert_eq!(r.fences.complete(), Report::default());
    assert_eq!(read(&r.mem, SEM, 4), vec![FILL; 4]);
    assert_eq!(r.syncpoints.read(watched), 0);
    assert_eq!(r.pushed.all(), vec![]);
    assert_eq!(vm.mappings(), mapped);
    assert_eq!(r.front.all(), vec![]);
    assert_eq!(r.fences.lists_in_flight(), 1);

    r.backend.finish(fence);
    let report = r.fences.complete();
    let expected = Report {
        released: 1,
        incremented: 1,
        latched: false,
        markers: vec![MarkerToken(7)],
        failed: vec![],
    };
    assert_eq!(report, expected);
    assert_eq!(word(&r.mem, SEM), 0xaabb_ccdd);
    assert_eq!(r.syncpoints.read(watched), 1);
    assert_eq!(
        r.pushed.all(),
        vec![
            GpuCompletion::SyncpointChanged {
                id: watched,
                value: 1
            },
            GpuCompletion::GpuOrdered { ticket: Ticket(9) },
        ]
    );
    assert_eq!(vm.mappings().len(), 1);
    assert_eq!(
        vm.mappings()[0].range,
        PageRange::new(vm_base + PAGE_SIZE, PAGE_SIZE).unwrap()
    );
    // One wake for the batch, after all of it.
    assert_eq!(r.front.all(), vec![expected]);
    assert_eq!(r.fences.lists_in_flight(), 0);
}

#[test]
fn actions_stay_in_order_when_lists_complete_out_of_order() {
    // Three lists whose fence values the backend hands out as 3, 1, 2. Their releases to one
    // semaphore give 0xA only in submission order: write 2, add 3, xor 0xF.
    let mut r = rig();
    r.backend.script(&[3, 1, 2]);
    let ops = [
        four(SEM, 2),
        reduce(SEM, 3, ReductionOp::Add),
        reduce(SEM, 0xf, ReductionOp::Xor),
    ];
    let mut fences = Vec::new();
    for op in ops {
        r.fences.record(command(1)).unwrap();
        let action = r.action(op);
        assert_eq!(r.fences.attach(action), Ok(Attached::Open));
        fences.push(r.fences.close(CloseReason::Present).unwrap().unwrap());
    }
    assert_eq!(
        fences,
        vec![FenceValue::new(3), FenceValue::new(1), FenceValue::new(2)]
    );
    // The second and third lists have completed, but wait behind the first.
    r.backend.finish(FenceValue::new(2));
    assert_eq!(r.fences.complete(), Report::default());
    assert_eq!(read(&r.mem, SEM, 4), vec![FILL; 4]);
    r.backend.finish(FenceValue::new(3));
    assert_eq!(r.fences.complete(), released(3));
    assert_eq!(word(&r.mem, SEM), 0xa);
}

#[test]
fn lists_completed_together_are_one_batch() {
    let mut r = rig();
    r.fences.record(command(1)).unwrap();
    let first = r.action(four(SEM, 1));
    r.fences.attach(first).unwrap();
    r.fences.close(CloseReason::Present).unwrap();
    r.fences.record(command(2)).unwrap();
    let second = r.action(four(SEM2, 2));
    r.fences.attach(second).unwrap();
    let last = r.fences.close(CloseReason::Present).unwrap().unwrap();
    // Nothing due: no wake.
    assert_eq!(r.fences.complete(), Report::default());
    assert_eq!(r.front.all(), vec![]);
    r.backend.finish(last);
    assert_eq!(r.fences.complete(), released(2));
    assert_eq!(r.front.all(), vec![released(2)]);
    assert_eq!((word(&r.mem, SEM), word(&r.mem, SEM2)), (1, 2));
}

// ---------------------------------------------------------------------------------------------
// The empty-fence rule
// ---------------------------------------------------------------------------------------------

/// Eden-NX2's `SkipEmptyFence` makes one fence in this many (`skipped_fences >= 64`).
const EDEN_NX2_SKIPS_BEFORE_A_FENCE: u64 = 64;

#[test]
fn every_action_with_nothing_behind_it_is_performed_at_once() {
    // Divergence from Eden-NX2: one more than its skip run, and every one is performed at once.
    let mut r = rig();
    for payload in 0..=EDEN_NX2_SKIPS_BEFORE_A_FENCE {
        let action = r.action(four(SEM, payload));
        assert_eq!(
            r.fences.attach(action),
            Ok(Attached::Performed(released(1))),
            "{payload}"
        );
        assert_eq!(u64::from(word(&r.mem, SEM)), payload);
    }
    let count = EDEN_NX2_SKIPS_BEFORE_A_FENCE + 1;
    assert_eq!(r.backend.submitted(), vec![]);
    assert_eq!(r.fences.counters().skipped_empty, count);
    assert_eq!(r.fences.counters().fenced, count);
    assert_eq!(r.front.all().len(), usize::try_from(count).unwrap());
}

#[test]
fn an_action_with_no_new_work_joins_the_newest_list_in_flight() {
    let mut r = rig();
    r.fences.record(command(1)).unwrap();
    let older = r.fences.close(CloseReason::Present).unwrap().unwrap();
    r.fences.record(command(2)).unwrap();
    let newer = r.fences.close(CloseReason::Present).unwrap().unwrap();
    let action = r.action(four(SEM, 5));
    assert_eq!(r.fences.attach(action), Ok(Attached::InFlight(newer)));
    assert_eq!(r.fences.counters().skipped_empty, 0);
    // The older list completing does not perform it.
    r.backend.finish(older);
    assert_eq!(r.fences.complete(), Report::default());
    assert_eq!(read(&r.mem, SEM, 4), vec![FILL; 4]);
    r.backend.finish(newer);
    assert_eq!(r.fences.complete(), released(1));
    assert_eq!(word(&r.mem, SEM), 5);
}

#[test]
fn an_action_after_recorded_work_joins_the_open_list() {
    let mut r = rig();
    r.fences.record(command(1)).unwrap();
    let action = r.action(four(SEM, 6));
    assert_eq!(r.fences.attach(action), Ok(Attached::Open));
    assert_eq!(r.backend.submitted(), vec![]);
    assert_eq!(r.fences.open_commands(), 1);
    let fence = r.fences.close(CloseReason::WaitForIdle).unwrap().unwrap();
    assert_eq!(r.backend.submitted(), vec![fence]);
    assert_eq!(r.fences.open_commands(), 0);
    r.backend.finish(fence);
    assert_eq!(r.fences.complete(), released(1));
    assert_eq!(word(&r.mem, SEM), 6);
}

#[test]
fn the_debugging_switch_performs_every_action_when_attached() {
    let mut r = rig_on(
        Arc::new(LateBackend::new(CompletionWake::none())),
        Mode::AtRecord,
    );
    r.fences.record(command(1)).unwrap();
    let action = r.action(four(SEM, 9));
    assert_eq!(
        r.fences.attach(action),
        Ok(Attached::Performed(released(1)))
    );
    assert_eq!(word(&r.mem, SEM), 9);
    // The list still closes as usual, with nothing waiting on it.
    let fence = r.fences.close(CloseReason::Present).unwrap().unwrap();
    r.backend.finish(fence);
    assert_eq!(r.fences.complete(), Report::default());
    assert_eq!(r.fences.counters().fenced, 1);
    assert_eq!(r.fences.counters().skipped_empty, 0);
}

// ---------------------------------------------------------------------------------------------
// Closing the open list
// ---------------------------------------------------------------------------------------------

#[test]
fn each_close_reason_is_counted_on_its_own() {
    let reasons = [
        CloseReason::RingEmpty,
        CloseReason::Present,
        CloseReason::WaitForIdle,
        CloseReason::Waiter,
        CloseReason::Elapsed,
        CloseReason::SetReference,
    ];
    for reason in reasons {
        let mut r = rig();
        r.fences.record(command(1)).unwrap();
        assert_eq!(r.fences.close(reason), Ok(Some(FenceValue::new(1))));
        let mut expected = ClosedCounts::default();
        match reason {
            CloseReason::RingEmpty => expected.ring_empty = 1,
            CloseReason::Present => expected.present = 1,
            CloseReason::WaitForIdle => expected.wait_for_idle = 1,
            CloseReason::Waiter => expected.waiter = 1,
            CloseReason::Elapsed => expected.elapsed = 1,
            CloseReason::SetReference => expected.set_reference = 1,
        }
        assert_eq!(r.fences.counters().closed, expected, "{reason:?}");
    }
}

#[test]
fn closing_a_list_with_no_commands_submits_nothing() {
    let mut r = rig();
    assert_eq!(r.fences.close(CloseReason::Present), Ok(None));
    assert_eq!(r.backend.submitted(), vec![]);
    assert_eq!(r.fences.counters().closed, ClosedCounts::default());
}

#[test]
fn a_release_a_channel_waits_on_closes_the_open_list() {
    let mut r = rig();
    r.waiters.register(Waiter {
        space: SPACE,
        channel: ChannelId(3),
        address: SEM,
        op: AcquireOp::Equal,
        payload: 4,
    });
    r.fences.record(command(1)).unwrap();
    let action = r.action(four(SEM, 4));
    let fence = FenceValue::new(1);
    assert_eq!(r.fences.attach(action), Ok(Attached::InFlight(fence)));
    assert_eq!(r.backend.submitted(), vec![fence]);
    assert_eq!(r.fences.counters().closed.waiter, 1);
    // Performed, and latched, only once the list completes.
    assert!(!r.waiters.is_latched(ChannelId(3)));
    r.backend.finish(fence);
    let report = r.fences.complete();
    assert!(report.latched);
    assert!(r.waiters.is_latched(ChannelId(3)));
}

#[test]
fn actions_nobody_waits_on_leave_the_list_open() {
    // A waiter elsewhere, a waiter already latched, and an increment (not a release) at the
    // waiter's syncpoint-free address: none closes the list.
    let mut r = rig();
    r.waiters.register(Waiter {
        space: SPACE,
        channel: ChannelId(1),
        address: SEM + 4,
        op: AcquireOp::Equal,
        payload: 1,
    });
    r.waiters.register(Waiter {
        space: SPACE,
        channel: ChannelId(2),
        address: SEM2,
        op: AcquireOp::Equal,
        payload: 1,
    });
    assert!(r.waiters.on_write(SPACE, SEM2, 1));
    r.fences.record(command(1)).unwrap();
    for action in [
        r.action(four(SEM, 1)),
        r.action(four(SEM2, 1)),
        FencedAction::Increment(syncpoint(7)),
    ] {
        assert_eq!(r.fences.attach(action), Ok(Attached::Open));
    }
    assert_eq!(r.backend.submitted(), vec![]);
    assert_eq!(r.fences.counters().closed.waiter, 0);
}

/// One nanosecond before `at`.
fn just_before(at: Duration) -> Duration {
    at.checked_sub(Duration::from_nanos(1)).unwrap()
}

#[test]
fn a_command_recorded_at_the_close_interval_closes_the_list() {
    let mut r = rig();
    let start = Duration::from_millis(40);
    r.clock.set(start);
    r.fences.record(command(1)).unwrap();
    r.clock.set(just_before(start + CLOSE_AFTER));
    r.fences.record(command(2)).unwrap();
    assert_eq!(r.fences.open_commands(), 2);
    assert_eq!(r.backend.submitted(), vec![]);
    r.clock.set(start + CLOSE_AFTER);
    r.fences.record(command(3)).unwrap();
    assert_eq!(r.fences.open_commands(), 0);
    assert_eq!(r.backend.submitted(), vec![FenceValue::new(1)]);
    assert_eq!(r.fences.counters().closed.elapsed, 1);
    // The next list measures from its own first command, not the first list's.
    let next = start + CLOSE_AFTER * 5;
    r.clock.set(next);
    r.fences.record(command(4)).unwrap();
    r.clock.set(just_before(next + CLOSE_AFTER));
    r.fences.record(command(5)).unwrap();
    assert_eq!(r.fences.open_commands(), 2);
    r.clock.set(next + CLOSE_AFTER);
    r.fences.record(command(6)).unwrap();
    assert_eq!(r.fences.open_commands(), 0);
    assert_eq!(r.fences.counters().closed.elapsed, 2);
}

#[test]
fn an_empty_ring_closes_the_open_list_before_the_back_thread_sleeps() {
    let (_producer, mut consumer) = ring(64).unwrap();
    let mut r = rig();
    r.fences.record(command(1)).unwrap();
    assert_eq!(
        r.fences.wait(&mut consumer, Duration::ZERO),
        Ok(Woke::TimedOut)
    );
    assert_eq!(r.backend.submitted(), vec![FenceValue::new(1)]);
    assert_eq!(r.fences.counters().closed.ring_empty, 1);
}

#[test]
fn a_ring_entry_comes_before_closing_the_list() {
    let (mut producer, mut consumer) = ring(64).unwrap();
    let entry = RingEntry::Write {
        channel: ChannelId(1),
        class: ClassId(0xc797),
        method: 0x45,
        value: 3,
    };
    producer.try_push(&entry).unwrap();
    producer.publish();
    let mut r = rig();
    r.fences.record(command(1)).unwrap();
    assert_eq!(
        r.fences.wait(&mut consumer, FOREVER),
        Ok(Woke::Entry(entry))
    );
    assert_eq!(r.fences.open_commands(), 1);
    assert_eq!(r.backend.submitted(), vec![]);
}

#[test]
fn completed_work_is_performed_while_the_ring_stays_busy() {
    // The back thread is the bottleneck: the ring always has an entry for it. A list in flight
    // completes; its release must not wait for the ring to drain. The first wait performs it and
    // leaves the entry on the ring, and the entries then come in their order, none lost.
    const ENTRIES: u32 = 4;
    let entry = |value| RingEntry::Write {
        channel: ChannelId(1),
        class: ClassId(0xc797),
        method: 0x45,
        value,
    };
    let (mut producer, mut consumer) = ring(64).unwrap();
    for value in 0..ENTRIES {
        producer.try_push(&entry(value)).unwrap();
    }
    producer.publish();
    let mut r = rig();
    let fence = r.list_with_release(SEM, 0x7d);
    // A second list, not complete, so the check looks only at the oldest list.
    let later = r.list_with_release(SEM2, 0x7e);
    r.backend.finish(fence);
    assert_eq!(
        r.fences.wait(&mut consumer, FOREVER),
        Ok(Woke::Completed(released(1)))
    );
    assert_eq!(word(&r.mem, SEM), 0x7d);
    assert_eq!(read(&r.mem, SEM2, 4), vec![FILL; 4]);
    assert_eq!(r.fences.lists_in_flight(), 1);
    // Nothing more has completed: the entries come next, in order.
    for value in 0..ENTRIES {
        assert_eq!(
            r.fences.wait(&mut consumer, FOREVER),
            Ok(Woke::Entry(entry(value)))
        );
    }
    // The later list completes while entries keep arriving: again performed first.
    producer.try_push(&entry(ENTRIES)).unwrap();
    producer.publish();
    r.backend.finish(later);
    assert_eq!(
        r.fences.wait(&mut consumer, FOREVER),
        Ok(Woke::Completed(released(1)))
    );
    assert_eq!(word(&r.mem, SEM2), 0x7e);
    assert_eq!(
        r.fences.wait(&mut consumer, FOREVER),
        Ok(Woke::Entry(entry(ENTRIES)))
    );
    assert_eq!(r.fences.lists_in_flight(), 0);
}

// ---------------------------------------------------------------------------------------------
// A refused list
// ---------------------------------------------------------------------------------------------

#[test]
fn a_refused_list_with_nothing_in_flight_performs_its_actions() {
    let mut r = rig();
    r.fences.record(command(1)).unwrap();
    let action = r.action(four(SEM, 0x31));
    r.fences.attach(action).unwrap();
    r.backend.refuse_next();
    let (result, events) = captured(|| r.fences.close(CloseReason::Present));
    assert_eq!(result, Err(FenceError::Submit(REFUSED)));
    assert_eq!(word(&r.mem, SEM), 0x31);
    assert_eq!(r.front.all(), vec![released(1)]);
    assert_eq!(
        (r.fences.open_commands(), r.fences.lists_in_flight()),
        (0, 0)
    );
    assert_eq!(r.fences.counters().closed, ClosedCounts::default());
    let errors = at(&events, tracing::Level::ERROR);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("refused a command list"), "{errors:?}");
    // The next list is submitted as usual.
    r.fences.record(command(2)).unwrap();
    assert_eq!(
        r.fences.close(CloseReason::Present),
        Ok(Some(FenceValue::new(1)))
    );
}

#[test]
fn a_refused_list_hands_its_actions_on() {
    let mut r = rig();
    r.fences.record(command(1)).unwrap();
    let older = r.fences.close(CloseReason::Present).unwrap().unwrap();
    r.fences.record(command(2)).unwrap();
    let action = r.action(four(SEM, 0x32));
    r.fences.attach(action).unwrap();
    r.backend.refuse_next();
    assert_eq!(
        r.fences.close(CloseReason::Present),
        Err(FenceError::Submit(REFUSED))
    );
    // Not performed yet: it waits on the list still in flight.
    assert_eq!(read(&r.mem, SEM, 4), vec![FILL; 4]);
    assert_eq!(r.fences.lists_in_flight(), 1);
    r.backend.finish(older);
    assert_eq!(r.fences.complete(), released(1));
    assert_eq!(word(&r.mem, SEM), 0x32);
}

#[test]
fn a_refused_list_hands_its_actions_to_the_newest_list_not_the_oldest() {
    // Two lists in flight; the second releases SEM = 1. A refused third list releases SEM = 2:
    // its release joins the newest list, so it is written after the second list's release, and
    // completing only the oldest list performs nothing (ordering guarantee 4).
    let mut r = rig();
    r.fences.record(command(1)).unwrap();
    let oldest = r.fences.close(CloseReason::Present).unwrap().unwrap();
    let newest = r.list_with_release(SEM, 1);
    r.fences.record(command(3)).unwrap();
    let action = r.action(four(SEM, 2));
    assert_eq!(r.fences.attach(action), Ok(Attached::Open));
    r.backend.refuse_next();
    assert_eq!(
        r.fences.close(CloseReason::Present),
        Err(FenceError::Submit(REFUSED))
    );
    assert_eq!(r.fences.lists_in_flight(), 2);
    r.backend.finish(oldest);
    assert_eq!(r.fences.complete(), Report::default());
    assert_eq!(read(&r.mem, SEM, 4), vec![FILL; 4]);
    r.backend.finish(newest);
    assert_eq!(r.fences.complete(), released(2));
    assert_eq!(word(&r.mem, SEM), 2);
    assert_eq!(r.front.all(), vec![released(2)]);
}

#[test]
fn a_refused_close_for_a_waiter_performs_the_release() {
    let mut r = rig();
    r.waiters.register(Waiter {
        space: SPACE,
        channel: ChannelId(4),
        address: SEM,
        op: AcquireOp::Equal,
        payload: 8,
    });
    r.fences.record(command(1)).unwrap();
    r.backend.refuse_next();
    let action = r.action(four(SEM, 8));
    assert_eq!(r.fences.attach(action), Err(FenceError::Submit(REFUSED)));
    assert_eq!(word(&r.mem, SEM), 8);
    assert!(r.waiters.is_latched(ChannelId(4)));
}

#[test]
fn a_refused_close_for_an_empty_ring_is_returned_by_wait() {
    let (_producer, mut consumer) = ring(64).unwrap();
    let mut r = rig();
    r.fences.record(command(1)).unwrap();
    r.backend.refuse_next();
    assert_eq!(
        r.fences.wait(&mut consumer, FOREVER),
        Err(FenceError::Submit(REFUSED))
    );
}

// ---------------------------------------------------------------------------------------------
// Performing releases, increments and markers
// ---------------------------------------------------------------------------------------------

#[test]
fn each_release_form_writes_exactly_its_bytes() {
    let ts = TIMESTAMP.to_le_bytes();
    let offset = usize::try_from(RELEASE_TIMESTAMP_OFFSET).unwrap();
    // (release, what the semaphore holds first, the bytes expected from the semaphore on)
    let mut cases: Vec<(Release, Option<Vec<u8>>, Vec<u8>)> = Vec::new();
    // 4 bytes: only the payload's low word, and the word after it untouched (row 52).
    cases.push((
        four(SEM, 0xdead_beef_1234_5678),
        None,
        [&[0x78, 0x56, 0x34, 0x12][..], &[FILL; 12]].concat(),
    ));
    let eight = Release {
        form: ReleaseForm {
            size: PayloadSize::Eight,
            timestamp: false,
        },
        ..four(SEM, 0x0102_0304_0506_0708)
    };
    cases.push((
        eight,
        None,
        [&0x0102_0304_0506_0708_u64.to_le_bytes()[..], &[FILL; 8]].concat(),
    ));
    // 16 bytes: the value widened to 64 bits, then the timestamp at its offset.
    let four_ts = Release {
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: true,
        },
        ..four(SEM, 0xffff_ffff_8765_4321)
    };
    let mut bytes = 0x8765_4321_u64.to_le_bytes().to_vec();
    bytes.resize(offset, 0);
    cases.push((four_ts, None, [&bytes[..], &ts[..], &[FILL; 4]].concat()));
    let eight_ts = Release {
        form: ReleaseForm {
            size: PayloadSize::Eight,
            timestamp: true,
        },
        ..four(SEM, 0x8877_6655_4433_2211)
    };
    cases.push((
        eight_ts,
        None,
        [
            &0x8877_6655_4433_2211_u64.to_le_bytes()[..],
            &ts[..],
            &[FILL; 4],
        ]
        .concat(),
    ));
    // Reductions read at the payload's width: a carry out of the low byte, and out of the low
    // word for 8 bytes.
    cases.push((
        reduce(SEM, 0x0101, ReductionOp::Add),
        Some(0x0000_01ff_u32.to_le_bytes().to_vec()),
        [&0x0000_0300_u32.to_le_bytes()[..], &[FILL; 12]].concat(),
    ));
    let add8 = Release {
        form: ReleaseForm {
            size: PayloadSize::Eight,
            timestamp: false,
        },
        ..reduce(SEM, 1, ReductionOp::Add)
    };
    cases.push((
        add8,
        Some(0x1_ffff_ffff_u64.to_le_bytes().to_vec()),
        [&0x2_0000_0000_u64.to_le_bytes()[..], &[FILL; 8]].concat(),
    ));
    // A reduction with a timestamp (a guess): the reduced value in the 16-byte structure.
    let add_ts = Release {
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: true,
        },
        ..reduce(SEM, 2, ReductionOp::Add)
    };
    cases.push((
        add_ts,
        Some(5_u32.to_le_bytes().to_vec()),
        [&7_u64.to_le_bytes()[..], &ts[..], &[FILL; 4]].concat(),
    ));
    for (release, before, expected) in cases {
        let mut r = rig();
        if let Some(before) = before {
            r.mem.write(SEM, &before).unwrap();
        }
        let action = r.action(release);
        assert_eq!(
            r.fences.attach(action),
            Ok(Attached::Performed(released(1))),
            "{release:?}"
        );
        assert_eq!(read(&r.mem, SEM, expected.len()), expected, "{release:?}");
        // The bytes before the semaphore are untouched too.
        assert_eq!(read(&r.mem, SEM - 4, 4), vec![FILL; 4], "{release:?}");
    }
}

#[test]
fn a_release_latches_the_waiters_its_value_meets() {
    let mut r = rig();
    // Met by the payload.
    r.waiters.register(Waiter {
        space: SPACE,
        channel: ChannelId(3),
        address: SEM,
        op: AcquireOp::Equal,
        payload: 0x5293,
    });
    // Met only by the reduced value (8 + 2), not the payload.
    r.waiters.register(Waiter {
        space: SPACE,
        channel: ChannelId(4),
        address: SEM2,
        op: AcquireOp::GreaterOrEqual,
        payload: 10,
    });
    // Met by the low word of an 8-byte payload whose high word differs.
    r.waiters.register(Waiter {
        space: SPACE,
        channel: ChannelId(5),
        address: SEM + 0x20,
        op: AcquireOp::Equal,
        payload: 1,
    });
    let action = r.action(four(SEM, 0x5292));
    assert_eq!(
        r.fences.attach(action),
        Ok(Attached::Performed(released(1)))
    );
    assert!(!r.waiters.is_latched(ChannelId(3)));
    let action = r.action(four(SEM, 0x5293));
    let latched = Report {
        latched: true,
        ..released(1)
    };
    assert_eq!(
        r.fences.attach(action),
        Ok(Attached::Performed(latched.clone()))
    );
    assert!(r.waiters.is_latched(ChannelId(3)));

    r.mem.write(SEM2, &8_u32.to_le_bytes()).unwrap();
    let action = r.action(reduce(SEM2, 2, ReductionOp::Add));
    assert_eq!(
        r.fences.attach(action),
        Ok(Attached::Performed(latched.clone()))
    );
    assert!(r.waiters.is_latched(ChannelId(4)));

    let action = r.action(Release {
        form: ReleaseForm {
            size: PayloadSize::Eight,
            timestamp: false,
        },
        ..four(SEM + 0x20, 0x2_0000_0001)
    });
    assert_eq!(r.fences.attach(action), Ok(Attached::Performed(latched)));
    assert!(r.waiters.is_latched(ChannelId(5)));
}

#[test]
fn a_release_that_faults_is_reported_and_the_batch_goes_on() {
    let mut r = rig();
    let past_end = UNMAPPED + PAGE_SIZE;
    r.fences.record(command(1)).unwrap();
    for release in [four(UNMAPPED, 1), four(SEM, 2), four(past_end, 3)] {
        let action = r.action(release);
        r.fences.attach(action).unwrap();
    }
    let fence = r.fences.close(CloseReason::Present).unwrap().unwrap();
    r.backend.finish(fence);
    let (report, events) = captured(|| r.fences.complete());
    assert_eq!(
        report,
        Report {
            released: 1,
            failed: vec![
                ReleaseFailure {
                    address: UNMAPPED,
                    fault: GpuFault::Unmapped { va: UNMAPPED },
                },
                ReleaseFailure {
                    address: past_end,
                    fault: GpuFault::Unmapped { va: past_end },
                },
            ],
            ..Report::default()
        }
    );
    assert_eq!(word(&r.mem, SEM), 2);
    assert_eq!(r.fences.counters().failed_releases, 2);
    let warnings = at(&events, tracing::Level::WARN);
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(warnings[0].contains("0x700000000"), "{warnings:?}");
}

#[test]
fn a_release_whose_end_is_unmapped_writes_nothing() {
    // A 16-byte release whose payload fits before the end of the mapping and whose timestamp
    // does not: one write, so nothing is written.
    let mut r = rig();
    let address = GPU_END - 8;
    let release = Release {
        form: ReleaseForm {
            size: PayloadSize::Eight,
            timestamp: true,
        },
        ..four(address, 0x1234)
    };
    let action = r.action(release);
    let failed = Report {
        failed: vec![ReleaseFailure {
            address,
            fault: GpuFault::Unmapped { va: GPU_END },
        }],
        ..Report::default()
    };
    assert_eq!(r.fences.attach(action), Ok(Attached::Performed(failed)));
    assert_eq!(read(&r.mem, address, 8), vec![FILL; 8]);
    // At the top of the address space the end does not fit in 64 bits.
    let top = u64::MAX - 7;
    let action = r.action(Release {
        address: top,
        ..release
    });
    let out_of_range = Report {
        failed: vec![ReleaseFailure {
            address: top,
            fault: GpuFault::OutOfRange { va: top, len: 16 },
        }],
        ..Report::default()
    };
    assert_eq!(
        r.fences.attach(action),
        Ok(Attached::Performed(out_of_range))
    );
}

#[test]
fn a_reduction_that_cannot_read_writes_nothing() {
    let mut r = rig();
    let action = r.action(reduce(UNMAPPED, 1, ReductionOp::Add));
    let failed = Report {
        failed: vec![ReleaseFailure {
            address: UNMAPPED,
            fault: GpuFault::Unmapped { va: UNMAPPED },
        }],
        ..Report::default()
    };
    assert_eq!(r.fences.attach(action), Ok(Attached::Performed(failed)));
}

#[test]
fn a_watched_increment_is_pushed_and_an_unwatched_one_is_not() {
    let mut r = rig();
    let watched = syncpoint(5);
    let unwatched = syncpoint(6);
    r.syncpoints.watch(watched);
    r.fences.record(command(1)).unwrap();
    for id in [watched, unwatched] {
        r.fences.attach(FencedAction::Increment(id)).unwrap();
    }
    let fence = r.fences.close(CloseReason::Present).unwrap().unwrap();
    assert_eq!(
        (r.syncpoints.read(watched), r.syncpoints.read(unwatched)),
        (0, 0)
    );
    r.backend.finish(fence);
    assert_eq!(
        r.fences.complete(),
        Report {
            incremented: 2,
            ..Report::default()
        }
    );
    assert_eq!(
        (r.syncpoints.read(watched), r.syncpoints.read(unwatched)),
        (1, 1)
    );
    assert_eq!(
        r.pushed.all(),
        vec![GpuCompletion::SyncpointChanged {
            id: watched,
            value: 1
        }]
    );
}

/// A completion queue that records how many mappings a table has at each push.
struct MappingsAtPush {
    vm: Arc<GpuVm>,
    seen: Mutex<Vec<(GpuCompletion, usize)>>,
}

impl Completions for MappingsAtPush {
    fn push(&self, completion: GpuCompletion) {
        let mappings = self.vm.mappings().len();
        self.seen.lock().unwrap().push((completion, mappings));
    }
}

#[test]
fn an_ordered_update_is_applied_before_its_report() {
    let vm = Arc::new(GpuVm::new());
    vm.map(
        PageRange::new(GPU_BASE, PAGE_SIZE).unwrap(),
        Target::Guest { cpu_va: CPU_BASE },
        Kind(0),
    )
    .unwrap();
    let queue = Arc::new(MappingsAtPush {
        vm: Arc::clone(&vm),
        seen: Mutex::new(Vec::new()),
    });
    let mut r = rig();
    let completions: Arc<dyn Completions> = Arc::<MappingsAtPush>::clone(&queue);
    r.fences.env.completions = completions;
    let marker = FencedAction::Marker {
        token: MarkerToken(1),
        ordered: Some(OrderedUpdate {
            change: Some(VmChange {
                vm: Arc::clone(&vm),
                update: VmUpdate::DestroyMemory {
                    range: PageRange::new(GPU_BASE, PAGE_SIZE).unwrap(),
                },
            }),
            ticket: Some(Ticket(4)),
        }),
    };
    r.fences.attach(marker).unwrap();
    assert_eq!(
        queue.seen.lock().unwrap().clone(),
        vec![(GpuCompletion::GpuOrdered { ticket: Ticket(4) }, 0)]
    );
}

#[test]
fn markers_report_their_token_and_only_what_they_carry() {
    let mut r = rig();
    let vm = Arc::new(GpuVm::new());
    vm.map(
        PageRange::new(GPU_BASE, PAGE_SIZE).unwrap(),
        Target::Guest { cpu_va: CPU_BASE },
        Kind(0),
    )
    .unwrap();
    let generation = crate::memory::Translate::generation(&*vm);
    let unmap = VmUpdate::Unmap {
        range: PageRange::new(GPU_BASE, PAGE_SIZE).unwrap(),
    };
    let cases = [
        // A bare marker: its token, nothing pushed, nothing applied.
        (MarkerToken(1), None, vec![], 1),
        // A barrier: a report, nothing applied.
        (
            MarkerToken(2),
            Some(OrderedUpdate {
                change: None,
                ticket: Some(Ticket(20)),
            }),
            vec![GpuCompletion::GpuOrdered { ticket: Ticket(20) }],
            1,
        ),
        // An update nobody asked to hear about: applied, nothing pushed.
        (
            MarkerToken(3),
            Some(OrderedUpdate {
                change: Some(VmChange {
                    vm: Arc::clone(&vm),
                    update: unmap,
                }),
                ticket: None,
            }),
            vec![GpuCompletion::GpuOrdered { ticket: Ticket(20) }],
            0,
        ),
    ];
    for (token, ordered, pushed, mappings) in cases {
        let marker = FencedAction::Marker { token, ordered };
        let report = Report {
            markers: vec![token],
            ..Report::default()
        };
        assert_eq!(r.fences.attach(marker), Ok(Attached::Performed(report)));
        assert_eq!(r.pushed.all(), pushed, "{token:?}");
        assert_eq!(vm.mappings().len(), mappings, "{token:?}");
    }
    assert_ne!(crate::memory::Translate::generation(&*vm), generation);
}

// ---------------------------------------------------------------------------------------------
// The back thread's sleep and the completion wake
// ---------------------------------------------------------------------------------------------

#[test]
fn a_completion_in_the_window_before_the_sleep_is_not_lost() {
    // The list completes, and the backend wakes, after the back thread's last look at
    // `completed()` and before it sleeps. Under test the ring's safety net is far longer than
    // PATIENCE, so only the wake can end the sleep.
    let (_producer, mut consumer) = ring(64).unwrap();
    let backend = Arc::new(LateBackend::new(consumer.waker().completion_wake()));
    let mut r = rig_on(Arc::clone(&backend), Mode::Fenced);
    let fence = r.list_with_release(SEM, 0x77);
    let mut pending = Some(fence);
    consumer.set_after_last_look(Box::new(move || {
        if let Some(fence) = pending.take() {
            backend.finish(fence);
        }
    }));
    let (woke, r) = within(move || {
        let woke = r.fences.wait(&mut consumer, FOREVER);
        (woke, r)
    });
    assert_eq!(woke, Ok(Woke::Completed(released(1))));
    assert_eq!(word(&r.mem, SEM), 0x77);
    assert_eq!(r.fences.counters().lost_wakes, 0);
}

#[test]
fn a_completion_while_the_back_thread_sleeps_wakes_it() {
    let (_producer, mut consumer) = ring(64).unwrap();
    let backend = Arc::new(LateBackend::new(consumer.waker().completion_wake()));
    let mut r = rig_on(Arc::clone(&backend), Mode::Fenced);
    let fence = r.list_with_release(SEM, 0x78);
    let mut pending = Some(fence);
    // From another thread, once the back thread is on its way to sleep.
    consumer.set_after_last_look(Box::new(move || {
        if let Some(fence) = pending.take() {
            let backend = Arc::clone(&backend);
            thread::spawn(move || backend.finish(fence));
        }
    }));
    let (woke, r) = within(move || {
        let woke = r.fences.wait(&mut consumer, FOREVER);
        (woke, r)
    });
    assert_eq!(woke, Ok(Woke::Completed(released(1))));
    assert_eq!(word(&r.mem, SEM), 0x78);
}

#[test]
fn a_backend_that_forgets_the_wake_is_caught_by_the_recheck_and_counted() {
    let (_producer, mut consumer) = ring(64).unwrap();
    consumer.set_longest_sleep(Duration::from_millis(5));
    let backend = Arc::new(LateBackend::new(consumer.waker().completion_wake()));
    let mut r = rig_on(Arc::clone(&backend), Mode::Fenced);
    // A first list completes and is performed by a direct completion check (its wake is not
    // needed, and would be left for the wait below), so the value last seen is not the
    // starting one.
    let first = r.list_with_release(SEM2, 0x78);
    backend.finish_silently(first);
    assert_eq!(r.fences.complete(), released(1));
    let fence = r.list_with_release(SEM, 0x79);
    let mut pending = Some(fence);
    consumer.set_after_last_look(Box::new(move || {
        if let Some(fence) = pending.take() {
            backend.finish_silently(fence);
        }
    }));
    let ((woke, events), r) = within(move || {
        let woke = captured(|| r.fences.wait(&mut consumer, FOREVER));
        (woke, r)
    });
    assert_eq!(woke, Ok(Woke::Completed(released(1))));
    assert_eq!(word(&r.mem, SEM), 0x79);
    assert_eq!(r.fences.counters().lost_wakes, 1);
    let warnings = at(&events, tracing::Level::WARN);
    assert_eq!(warnings.len(), 1, "{events:?}");
    assert!(
        warnings[0].contains("moved without a completion wake"),
        "{warnings:?}"
    );
    assert!(warnings[0].contains("seen=1;completed=2;"), "{warnings:?}");
}

#[test]
fn a_recheck_that_finds_nothing_moved_counts_no_lost_wake() {
    let (_producer, mut consumer) = ring(64).unwrap();
    consumer.set_longest_sleep(Duration::from_millis(5));
    let mut r = rig();
    // A list completed and performed earlier: `completed()` is not at its starting value.
    let first = r.list_with_release(SEM2, 0x78);
    r.backend.finish(first);
    assert_eq!(r.fences.complete(), released(1));
    let ((woke, events), r) = within(move || {
        let woke = captured(|| r.fences.wait(&mut consumer, FOREVER));
        (woke, r)
    });
    assert_eq!(woke, Ok(Woke::Completed(Report::default())));
    assert_eq!(r.fences.counters().lost_wakes, 0);
    assert_eq!(at(&events, tracing::Level::WARN), Vec::<String>::new());
}

#[test]
fn completed_work_found_before_the_sleep_is_performed_without_sleeping() {
    let (_producer, mut consumer) = ring(64).unwrap();
    // The rig's backend has no wake, so only the look at `completed()` can end this wait.
    let mut r = rig();
    let fence = r.list_with_release(SEM, 0x7a);
    r.backend.finish(fence);
    let woke = within(move || r.fences.wait(&mut consumer, FOREVER));
    assert_eq!(woke, Ok(Woke::Completed(released(1))));
}

#[test]
fn a_wake_for_work_already_performed_costs_one_look() {
    let (_producer, mut consumer) = ring(64).unwrap();
    let backend = Arc::new(LateBackend::new(consumer.waker().completion_wake()));
    let mut r = rig_on(Arc::clone(&backend), Mode::Fenced);
    let fence = r.list_with_release(SEM, 0x7c);
    backend.finish(fence);
    // Performed by a direct completion check; the wake stays on the ring.
    assert_eq!(r.fences.complete(), released(1));
    let woke = within(move || r.fences.wait(&mut consumer, FOREVER));
    assert_eq!(woke, Ok(Woke::Completed(Report::default())));
}

#[test]
fn a_gone_front_is_an_error_from_wait() {
    let (producer, mut consumer) = ring(64).unwrap();
    drop(producer);
    let mut r = rig();
    assert_eq!(
        r.fences.wait(&mut consumer, FOREVER),
        Err(FenceError::Ring(PopError::Disconnected))
    );
}

#[test]
fn the_recording_backends_wake_reaches_the_back_threads_ring() {
    // The production arrangement: the recording backend built with the ring's completion wake.
    // First the back thread's own list goes through it and its release is performed. Then,
    // with nothing in flight (so the back thread's look at `completed()` sees no work), another
    // thread's list completes inside `submit` just before the back thread sleeps: only the
    // recording backend's wake can end that sleep before the short timeout.
    for wakes in [true, false] {
        let (_producer, mut consumer) = ring(64).unwrap();
        let wake = if wakes {
            consumer.waker().completion_wake()
        } else {
            CompletionWake::none()
        };
        let backend = Arc::new(RecordingBackend::with_completion_wake(caps(), 0, wake));
        let late = rig();
        let mut fences = Fences::new(
            Arc::<RecordingBackend>::clone(&backend),
            late.fences.env.clone(),
            Mode::Fenced,
        );
        fences.record(command(1)).unwrap();
        let action = late.action(four(SEM, 0x7b));
        assert_eq!(fences.attach(action), Ok(Attached::Open));
        let first = within(move || {
            let first = fences.wait(&mut consumer, FOREVER);
            (first, fences, consumer)
        });
        let (first, mut fences, mut consumer) = first;
        assert_eq!(first, Ok(Woke::Completed(released(1))), "wakes: {wakes}");
        assert_eq!(word(&late.mem, SEM), 0x7b);
        assert_eq!(backend.completed(), FenceValue::new(1));

        let mut other = Some(Arc::clone(&backend));
        consumer.set_after_last_look(Box::new(move || {
            if let Some(backend) = other.take() {
                backend.submit(CommandList::new()).unwrap();
            }
        }));
        let second = within(move || fences.wait(&mut consumer, Duration::from_millis(50)));
        let expected = if wakes {
            Woke::Completed(Report::default())
        } else {
            Woke::TimedOut
        };
        assert_eq!(second, Ok(expected), "wakes: {wakes}");
        assert_eq!(backend.completed(), FenceValue::new(2));
    }
}

#[test]
fn the_late_backend_meets_the_wake_contract() {
    // Parity with the recording backend: the same contract check, with a thread playing the host
    // GPU that finishes each list as it arrives.
    let wakes = Arc::new(WakeCounter::new());
    let backend = Arc::new(LateBackend::new(wakes.completion_wake()));
    let stop = Arc::new(AtomicBool::new(false));
    let host = {
        let backend = Arc::clone(&backend);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                let newest = backend.submitted().last().copied();
                if let Some(fence) = newest
                    && backend.completed() < fence
                {
                    backend.finish(fence);
                }
                thread::yield_now();
            }
        })
    };
    let result = check_completion_wake(&*backend, &wakes, 3, PATIENCE);
    stop.store(true, Ordering::SeqCst);
    host.join().unwrap();
    assert_eq!(result, Ok(()));

    // And it fails the check the way the recording backend would when it does not complete or
    // does not wake.
    let wakes = Arc::new(WakeCounter::new());
    let idle = LateBackend::new(wakes.completion_wake());
    assert_eq!(
        check_completion_wake(&idle, &wakes, 1, Duration::from_millis(5)),
        Err(WakeContractError::NotCompleted { list: 0, fence: 1 })
    );
    let wakes = Arc::new(WakeCounter::new());
    let silent = LateBackend::new(wakes.completion_wake());
    silent.finish_silently(FenceValue::new(1));
    assert_eq!(
        check_completion_wake(&silent, &wakes, 1, Duration::from_millis(5)),
        Err(WakeContractError::NoWake { list: 0, fence: 1 })
    );
}

// ---------------------------------------------------------------------------------------------
// Waiters keyed by address space, and every word of a wide release (slice G14)
// ---------------------------------------------------------------------------------------------

fn waiter_on(channel: u32, space: AsId, address: u64, payload: u32) -> Waiter {
    Waiter {
        channel: ChannelId(channel),
        space,
        address,
        op: AcquireOp::Equal,
        payload,
    }
}

fn eight(address: u64, payload: u64) -> Release {
    Release {
        form: ReleaseForm {
            size: PayloadSize::Eight,
            timestamp: false,
        },
        ..four(address, payload)
    }
}

/// What attaching one release with nothing recorded or in flight answers: performed at once,
/// with whether it latched a waiter.
fn performed(latched: bool) -> Attached {
    Attached::Performed(Report {
        released: 1,
        latched,
        ..Report::default()
    })
}

#[test]
fn a_wide_release_latches_a_waiter_on_its_high_word() {
    // An 8-byte release at SEM writes the words at SEM and SEM + 4: a waiter on each word is
    // latched by that word's value, and one on the word after (SEM + 8, the next neighbour) is
    // not, though its payload is the release's low word.
    let mut r = rig();
    r.waiters.register(waiter_on(1, SPACE, SEM, 0x5566_7788));
    r.waiters
        .register(waiter_on(2, SPACE, SEM + 4, 0x1122_3344));
    r.waiters
        .register(waiter_on(3, SPACE, SEM + 8, 0x5566_7788));
    assert_eq!(
        r.fences.attach(r.action(eight(SEM, 0x1122_3344_5566_7788))),
        Ok(performed(true))
    );
    assert!(r.waiters.is_latched(ChannelId(1)));
    assert!(r.waiters.is_latched(ChannelId(2)));
    assert!(!r.waiters.is_latched(ChannelId(3)));

    // A 4-byte release writes only its own word: a waiter on SEM + 4 is not latched by it,
    // whatever the payload's high half.
    let mut r = rig();
    r.waiters
        .register(waiter_on(4, SPACE, SEM + 4, 0x1122_3344));
    assert_eq!(
        r.fences.attach(r.action(four(SEM, 0x1122_3344_5566_7788))),
        Ok(performed(false))
    );
    assert!(!r.waiters.is_latched(ChannelId(4)));

    // The 16-byte structure covers four words: the timestamp's halves at SEM + 8 and SEM + 12.
    let mut r = rig();
    let [t0, t1, t2, t3, t4, t5, t6, t7] = TIMESTAMP.to_le_bytes();
    r.waiters.register(waiter_on(
        5,
        SPACE,
        SEM + 12,
        u32::from_le_bytes([t4, t5, t6, t7]),
    ));
    r.waiters.register(waiter_on(
        6,
        SPACE,
        SEM + 8,
        u32::from_le_bytes([t0, t1, t2, t3]),
    ));
    let release = Release {
        form: ReleaseForm {
            size: PayloadSize::Eight,
            timestamp: true,
        },
        ..four(SEM, 1)
    };
    assert_eq!(r.fences.attach(r.action(release)), Ok(performed(true)));
    assert!(r.waiters.is_latched(ChannelId(5)));
    assert!(r.waiters.is_latched(ChannelId(6)));
}

#[test]
fn a_waiter_on_a_wide_releases_high_word_closes_the_open_list() {
    // A waiter on SEM + 4: an 8-byte release at SEM closes the open list for it, a 4-byte one
    // does not.
    let mut r = rig();
    r.waiters.register(waiter_on(1, SPACE, SEM + 4, 9));
    r.fences.record(command(1)).unwrap();
    assert_eq!(
        r.fences.attach(r.action(four(SEM, 9 << 32))),
        Ok(Attached::Open)
    );
    assert_eq!(
        r.fences.attach(r.action(eight(SEM, 9 << 32))),
        Ok(Attached::InFlight(FenceValue::new(1)))
    );
    assert_eq!(r.fences.counters().closed.waiter, 1);
}

#[test]
fn a_release_latches_and_closes_only_for_its_own_address_space() {
    // A waiter of another address space on the release's address: the release neither closes
    // the open list for it nor latches it.
    let mut r = rig();
    let other = AsId(SPACE.0 + 1);
    r.waiters.register(waiter_on(1, other, SEM, 4));
    r.fences.record(command(1)).unwrap();
    assert_eq!(r.fences.attach(r.action(four(SEM, 4))), Ok(Attached::Open));
    let fence = r.fences.close(CloseReason::Present).unwrap().unwrap();
    r.backend.finish(fence);
    let report = r.fences.complete();
    assert!(!report.latched);
    assert!(!r.waiters.is_latched(ChannelId(1)));
    // The same release issued in the waiter's space closes the list for it.
    let action = FencedAction::Release {
        memory: Arc::<Memory>::clone(&r.mem),
        space: other,
        release: four(SEM, 4),
    };
    r.fences.record(command(2)).unwrap();
    assert_eq!(
        r.fences.attach(action),
        Ok(Attached::InFlight(FenceValue::new(2)))
    );
}
