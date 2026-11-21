//! The run's GPU (gpu-protocol, "Graphics server sessions (#11)": "`core` creates the `Gpu` ...
//! and hands it to gsv"; issue #386, which takes over the run wiring slice G21 left): which
//! front a run's graphics server has, the GPU's parts, its liveness during the run, and what it
//! leaves for the run's report at the end.
//!
//! No reference is ported: the wiring is the specs'.
//!
//! # Which front
//!
//! A run's graphics server has the GPU ([`Graphics::with_gpu`]) when the run is live
//! ([`EnvSource::Live`]) and its guest memory can be shared with the GPU's threads
//! ([`super::Session::gpu_memory`]: the real run's address space; a test's scripted run gives
//! none). Otherwise it has the null front ([`Graphics::new`]), as slice G21 decided: every
//! replay, and a run with no GPU back end ([`builds_gpu`], tested for every combination).
//!
//! # The GPU's parts
//!
//! - **The render backend** is the one the run's [`RenderSetup`] chose
//!   (`crate::renderer`, "The choice"; slice R12): the recording backend, or the wgpu backend
//!   ("The wgpu backend", below). The recording backend ([`RecordingBackend`], render-backend
//!   "The recording backend": "the `null` backend for runs that need no pixels") is built with
//!   the ring's completion wake, with [`recording_caps`] and [`RECORDING_MEMORY`] as its budget, and
//!   without its call log ([`RecordingBackend::without_call_log`]): it keeps no submitted list
//!   (so none of an upload's staging bytes), no record of a create, destroy or present target,
//!   and no present feedback, none of which the run reads. What it holds for the run is then the
//!   CPU copy of every live resource's bytes (counted against [`RECORDING_MEMORY`] by the
//!   budget the GPU's caches evict to), the destroys waiting for their fence, and readbacks not
//!   yet taken; it does not grow with the number of presents
//!   (`the_run_s_backend_holds_no_more_after_fifty_presents_than_after_one`, in
//!   `hle/tests/graphics/present.rs`). Keeping no log departs from render-backend's "It keeps
//!   every submitted list and every resource description" (an owner question, listed in the
//!   render crate's `recording` module docs).
//! - **Frame dumps**: the GPU is asked to dump every present
//!   ([`FrameDumps::Every`] in `GpuConfig::exec`, the setting through which `core` asks for
//!   dumps), and the backend hands them to the run's [`RunFrames`]: its frame keeper
//!   ([`FrameKeeper`], `crate::frames`) and a count of the frames handed over, which [`finish`]
//!   compares with the GPU's presents ("At the end").
//! - **The shader cache** behind G18's seam is [`NoShaders`] until shader-pipeline slice 19
//!   builds the real one: every lookup is pending and never finishes, so every draw and
//!   dispatch is skipped and counted (the draw path's `ShaderPending` skip).
//! - **The guest memory** is the run's ([`super::Session::gpu_memory`]).
//! - **The write watch** is one the run makes for the GPU. The kernel half of `WriteWatch`
//!   (taking and arming requests, recording faults) is a kernel slice not yet built, so no
//!   request is ever armed: the GPU's caches treat every range as untracked and read it again at
//!   each use, which is safe (`starquake_memory::WriteWatch`'s module docs). Each request the
//!   caches make waits in the watch for the rest of the run: the watch grows by one request per
//!   cache entry the GPU makes, which the kernel half bounds once it takes them.
//! - **The GPU's clocks** ([`GuestTimeClock`], issue #435). The timestamp a 16-byte semaphore
//!   release writes is guest time from the kernel's clock (gpu-protocol, "Fenced completion";
//!   the owner's ruling of 2026-11-19, row 29): the back thread's clock reads a copy of the
//!   run's clock, the one whose other copy the environment wraps for the kernel
//!   (`crate::clock::Timebase`; a copy shares the zero, so the timestamp counts from guest time's
//!   zero), and converts the reading with the kernel's
//!   [`starquake_kernel::time::gpu_ticks_at`], into the guest GPU timer's ticks (614.4 MHz, a
//!   guess for Switch 2 listed there). Nothing in `core` converts: the kernel owns the
//!   conversion. How long a list has been open, which only decides when the back closes it, is
//!   the host's monotonic time, as is the front thread's clock (the acquire poll's back-off).
//!   The reading is not recorded in the environment's recording: the GPU's threads read it in an
//!   order the kernel's readings do not fix, and no replay builds the GPU ("Which front"). A
//!   GPU capture records each one (`starquake_gpu::capture`, the `timestamp` record), and its
//!   replay writes the recorded value. With a clock that counts its own readings (`--clock
//!   fixed`), the GPU's copy counts its own, starting from the state of the loop's copy when
//!   `Machine::start` builds the GPU (`RunClock::for_gpu`): the `n`th timestamp is that copy's
//!   `n`th reading, not the time the kernel last read (pinned by
//!   `the_gpu_s_copy_of_a_fixed_clock_starts_where_the_loop_s_copy_is`).
//! - The GPU's other settings are `GpuConfig`'s defaults ([`gpu_config`]).
//!
//! # The wgpu backend
//!
//! A run whose setup chose the wgpu backend opens it through [`WgpuOpener`] (the
//! `wgpu_backend` module: the shared compile pool, the GPU crate's wait setting and the ring's
//! completion wake at construction, the run's frames as its frame sink, and the window as its
//! present target, made on the frontend's main thread). Its other parts are the recording
//! backend's: the same frame dumps, shader cache, guest memory, write watch, clocks and
//! settings. When it cannot be opened, the choice says what happens
//! ([`crate::renderer::IfUnavailable`]): the run renders with the recording backend instead,
//! dumping into the same run's frames
//! (`a_run_that_falls_back_to_the_recording_backend_dumps_into_its_own_frames`, in
//! `hle/tests/graphics/present.rs`), and logs [`FELL_BACK`] at warning level, with the reason, or the run ends as an emulator error
//! with [`WGPU_UNAVAILABLE`] and the reason (both pinned by
//! `an_unavailable_wgpu_backend_falls_back_or_ends_the_run_as_chosen`, in `gpu/tests.rs`). A
//! window that cannot be presented into ([`Unopened::Window`]) always ends the run, with the
//! window's own words (`wgpu_backend::WINDOW_FAILED` and the reason; pinned on a real GPU by
//! `a_window_the_backend_refuses_ends_the_run_with_the_window_s_words`, in
//! `hle/tests/graphics/present.rs`). The local GPU test of the whole path is
//! `the_wgpu_backend_dumps_the_same_frame_as_the_recording_backend`, there too.
//!
//! **The shader cache shares no pool yet.** The compile pool a wgpu run makes implements the
//! shader crate's executor trait as well as the render crate's (`crate::compile_pool`), so the
//! shader cache can take the same pool; but the run's shader cache is still [`NoShaders`], since
//! no translator exists to build the real one with (shader-pipeline slice 19's `ShaderCache`
//! takes a `Translator`, which slice 17 builds). A run with the recording backend makes no pool:
//! the recording backend builds no pipelines, and nothing else would use it.
//!
//! **Departure: the clock is not in `GpuConfig`.** gpu-protocol says the timestamp is guest time
//! "which `core` passes through the `Gpu`'s config". `GpuConfig` is a plain settings value
//! (`Copy`, compared whole by `a_run_asks_the_gpu_to_dump_every_present`), so the clock goes
//! where the GPU already takes its clocks: [`GpuParts::fence_clock`], into `GpuDeps::fence_clock`,
//! beside the config. **Owner question**: whether the config should carry the clock instead.
//!
//! # During the run
//!
//! The timer thread asks [`Gpu::is_running`] at every turn: a GPU thread that ended (it failed
//! or panicked) ends the run as an emulator error with the words [`GPU_STOPPED`] and the
//! thread's error ([`stopped_message`]), so the run does not wait on a GPU that will never
//! complete anything (`machine/timer.rs`; `a_dead_gpu_thread_ends_the_run`, in `machine/tests/graphics/frames.rs`).
//!
//! # At the end
//!
//! Once every core and the timer thread have stopped, [`finish`] waits for the GPU to be idle
//! ([`Gpu::is_idle`]), so that the GPU records the presents handed to it before the end; a GPU
//! still busy at [`GPU_DRAIN_WAIT`] is logged at warning level and stopped all the same, and
//! what it had not done is not in the dumps. An idle GPU is not yet enough for the frame
//! keeper: the wgpu backend's render thread stores a list's completion and calls the completion
//! wake before it hands that list's frame to the run's sink (render-wgpu's `backend` module
//! docs, "Fences and the completion wake"), so the GPU can be idle while the last frame is still
//! on its way. The recording backend hands its frames over inside `submit`, before the GPU's back
//! thread goes on. So, once the GPU is idle, `finish` also waits until the run's frames
//! ([`RunFrames`]) have been handed a frame for every present the GPU recorded
//! (`GpuCounters::presents`; every present is dumped, [`FrameDumps::Every`]), within the same
//! [`GPU_DRAIN_WAIT`] counted from the start of the drain. A frame still missing then (a
//! readback the backend could not finish, for one) is logged at warning level as
//! [`FRAMES_MISSING`], with the frames received and the presents, and is not in the dumps; the
//! run ends all the same. Both are pinned with a backend double that hands its frame over after
//! the GPU is idle: `finishing_waits_for_a_frame_handed_over_after_the_gpu_is_idle` and
//! `a_frame_that_never_comes_is_logged_and_left_out`, in `hle/tests/graphics/present.rs`. A GPU
//! still busy at the end is not waited for again: its warning says the dumps may be short. It
//! then stops the GPU
//! ([`Gpu::stop`]) and reads its counters ([`Gpu::counters`], `summary.json`'s `gpu`), its
//! flight recorder ([`Gpu::flight_recorder`], `gpu.txt`) and the run's frames. A GPU thread that
//! failed makes a run that otherwise reached its stop condition an emulator error
//! ([`GPU_FAILED`]); a crash, a timeout or an emulator error keeps its own reason, which matters
//! more, and the failure is logged (`after_stop`, tested for every ending with and without a
//! failure; its use here by
//! `a_gpu_thread_that_failed_turns_a_clean_ending_into_an_emulator_error`, in
//! `hle/tests/graphics/present.rs`, whose GPU thread panics on a read of guest memory). A GPU
//! whose threads already ended is not waited for: it can never be idle again (the same test
//! asserts that `finish` returns in under half of [`GPU_DRAIN_WAIT`]). The frames the keeper
//! holds reach the report through `finish` (`a_presented_image_is_dumped_as_a_png_of_its_pixels`,
//! which presents a readable frame and writes the report `finish` returns), and the run's
//! aftermath keeps the graphics report for every ending
//! (`every_ending_keeps_the_gpu_report_and_the_kept_frames` and
//! `a_crashed_or_timed_out_run_keeps_its_graphics_report`, in `machine/tests/graphics/frames.rs`).
//!
//! **No test pins the wait for the GPU to be idle** (the wait for the frames, above, is pinned).
//! Removing it leaves every test here passing: whether the GPU's front has read a request by the
//! time it is stopped (which drops what it has not read, `starquake_gpu::wiring`, "Shutdown")
//! depends on host timing, and in the tests' runs the last present is handed long enough before
//! the end that the front has always read it. The warning for a GPU still busy is not pinned
//! either: no test can make the GPU stay busy for [`GPU_DRAIN_WAIT`] without a backend that never
//! completes, which the run does not build.

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use starquake_gpu::back::buffer::InvalidationSink;
use starquake_gpu::back::draw::shaders::{SetLookup, ShaderLookup};
use starquake_gpu::back::exec::{ExecConfig, FrameDumps};
use starquake_gpu::back::fence::FenceClock;
use starquake_gpu::wiring::{Gpu, GpuConfig, GpuRing, IDLE_POLL_WAIT, WiringError};
use starquake_gsv::completion::Wake;
use starquake_kernel::time::gpu_ticks_at;
use starquake_memory::{GuestMemory, PageRange, WriteWatch};
use starquake_render::caps::{DynamicState, StageSupport};
use starquake_render::{
    Backend, Caps, Features, FormatFeatures, FormatTable, Frame, FrameSink, Limits,
    RecordingBackend, SampleCounts,
};
use starquake_shader::lookup::{EngineState, ProgramMemory, ProgramRequest, Ticket, VariantKey};
use starquake_trace::HostClock;

use super::Ending;
use crate::frames::{DumpedFrames, FrameKeeper};
use crate::hle::Graphics;
use crate::hle::graphics::{FrontMode, GpuParts};
use crate::renderer::{Choice, IfUnavailable, PRESENT_TARGET_WAIT, RenderSetup};
use crate::report::{EnvSource, GraphicsReport};

pub(crate) mod wgpu_backend;

pub(crate) use wgpu_backend::{Unopened, WgpuOpener};

/// What a run logs at warning level, with the reason, when the wgpu backend it chose by default
/// cannot be opened and it renders with the recording backend instead (module docs, "The wgpu
/// backend").
pub(crate) const FELL_BACK: &str =
    "the wgpu backend could not be opened; the run renders with the recording backend";

/// The words a run ends with when the wgpu backend it asked for cannot be opened (module docs,
/// "The wgpu backend"), before the reason.
pub(crate) const WGPU_UNAVAILABLE: &str = "the wgpu backend could not be opened";

/// The memory budget the run's recording backend reports: starquake's own number. The recording
/// backend keeps a copy of every resource in host memory; the GPU's caches evict to stay within
/// the budget it reports.
pub(crate) const RECORDING_MEMORY: u64 = 1 << 30;

/// The longest [`finish`] waits for the GPU to be idle at the end of a run: starquake's own
/// number, a hundred times [`IDLE_POLL_WAIT`], the longest one look at its idleness waits.
pub(crate) const GPU_DRAIN_WAIT: Duration = IDLE_POLL_WAIT.saturating_mul(100);

/// How long [`finish`] sleeps between two looks at the GPU's idleness.
const DRAIN_LOOK: Duration = Duration::from_millis(1);

/// What [`finish`] logs at warning level, with the counts, when the render backend has not
/// handed over a frame for every present the GPU recorded within [`GPU_DRAIN_WAIT`] (module
/// docs, "At the end").
pub(crate) const FRAMES_MISSING: &str = "the render backend had not handed over a frame for \
     every present when the run ended; the missing frames are not in the frame dumps";

/// How a run ends when a GPU thread ended during it (module docs, "During the run"), before
/// the thread's error.
pub(crate) const GPU_STOPPED: &str = "the GPU's threads ended during the run";

/// How a run that reached its stop condition ends when a GPU thread failed (module docs, "At
/// the end"), before the thread's error.
pub(crate) const GPU_FAILED: &str = "a GPU thread failed";

/// Whether a run builds the GPU (module docs, "Which front"): only a live run whose guest memory
/// the GPU's threads can share.
#[must_use]
pub(crate) fn builds_gpu(environment: &EnvSource, shared_memory: bool) -> bool {
    matches!(environment, EnvSource::Live) && shared_memory
}

/// The capabilities the run's recording backend reports. Its numeric limits are wgpu's
/// defaults (wgpu-types 30.0.1 `Limits::defaults`, `src/limits.rs`), with the anisotropy and
/// viewports render-wgpu reports for any wgpu device (`WGPU_MAX_ANISOTROPY`, 16;
/// `WGPU_VIEWPORTS`, 1), so a run refuses what a default wgpu device would; the largest buffer
/// and staging slice are wgpu's default buffer size, 256 MiB. Since it runs nothing, it claims
/// every optional feature, stage and dynamic state, every format use, view swizzles and the
/// sample counts wgpu guarantees (1 and 4): starquake's own choice, so that nothing a guest asks
/// is refused for want of a feature no host would be asked to run here.
#[must_use]
pub(crate) fn recording_caps() -> Caps {
    const WGPU_DEFAULT_BUFFER_SIZE: u64 = 256 << 20;
    Caps {
        limits: Limits {
            max_buffer_size: WGPU_DEFAULT_BUFFER_SIZE,
            max_staging_size: WGPU_DEFAULT_BUFFER_SIZE,
            max_texture_dimension_1d: 8192,
            max_texture_dimension_2d: 8192,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 256,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 1,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 1000,
            max_compute_workgroups_per_dimension: 65535,
            max_uniform_buffers_per_stage: 12,
            max_storage_buffers_per_stage: 8,
            max_sampled_textures_per_stage: 16,
            max_storage_textures_per_stage: 4,
            max_samplers_per_stage: 16,
        },
        features: Features::ALL,
        stages: StageSupport::ALL,
        dynamic_state: DynamicState::ALL,
        sample_counts: SampleCounts::X1.union(SampleCounts::X4),
        view_swizzle: true,
        formats: FormatTable::uniform(FormatFeatures::ALL),
    }
}

/// The run's frame dumps as its render backend hands them over (module docs, "The GPU's parts"
/// and "At the end"): the frame keeper, and how many frames it has been handed, which [`finish`]
/// compares with the GPU's presents before it reads the keeper. Both backends hand their frames
/// here: the recording backend directly, the wgpu backend through its run sink
/// (`wgpu_backend`'s `RunSink`).
#[derive(Debug, Default)]
pub(crate) struct RunFrames {
    keeper: FrameKeeper,
    /// Frames handed over, each counted once the keeper has it.
    received: AtomicU64,
}

impl RunFrames {
    /// The frames handed over so far.
    pub(crate) fn received(&self) -> u64 {
        self.received.load(Ordering::Acquire)
    }

    /// What the keeper holds ([`FrameKeeper::take`]); the keeper is left empty.
    #[must_use]
    pub(crate) fn take(&self) -> DumpedFrames {
        self.keeper.take()
    }
}

impl FrameSink for RunFrames {
    /// Keeps `frame`, then counts it: a count that [`RunFrames::received`] shows is a frame
    /// already in the keeper.
    fn frame(&self, frame: Frame) {
        self.keeper.frame(frame);
        self.received.fetch_add(1, Ordering::Release);
    }
}

/// The shader cache behind G18's seam until shader-pipeline slice 19 (module docs, "The GPU's
/// parts"): nothing is ever translated.
#[derive(Debug, Default)]
pub(crate) struct NoShaders;

/// The ticket every lookup of [`NoShaders`] answers with.
const NO_TICKET: Ticket = Ticket::new(0);

impl ShaderLookup for NoShaders {
    /// Pending, for ever.
    fn lookup(
        &mut self,
        _requests: &[ProgramRequest],
        _variant: &VariantKey,
        _memory: &dyn ProgramMemory,
        _state: &dyn EngineState,
    ) -> SetLookup {
        SetLookup::Pending(NO_TICKET)
    }

    /// Still pending: the draw path then skips the draw.
    fn wait(&mut self, ticket: Ticket) -> SetLookup {
        SetLookup::Pending(ticket)
    }
}

impl InvalidationSink for NoShaders {
    /// Nothing is cached, so nothing is dropped.
    fn invalidate(&mut self, _range: PageRange) {}
}

/// The GPU back thread's clock in a run (module docs, "The GPU's clocks"): its timestamp is guest
/// time, the kernel's conversion of a reading of the run's clock; its elapsed time is the host's.
pub(crate) struct GuestTimeClock<C> {
    /// A copy of the run's clock: the same zero as the environment's copy.
    guest: Mutex<C>,
    /// The host's monotonic clock, for how long a list has been open.
    host: starquake_gpu::front::HostClock,
}

impl<C> GuestTimeClock<C> {
    /// The back thread's clock over `guest`, a copy of the run's clock.
    pub(crate) fn new(guest: C) -> Self {
        Self {
            guest: Mutex::new(guest),
            host: starquake_gpu::front::HostClock::new(),
        }
    }
}

impl<C: HostClock> FenceClock for GuestTimeClock<C> {
    /// The host's monotonic time since this clock was made.
    fn elapsed(&self) -> Duration {
        FenceClock::elapsed(&self.host)
    }

    /// The guest GPU timer now: [`gpu_ticks_at`] of a reading of the run's clock.
    fn gpu_timestamp(&self) -> u64 {
        let now = self
            .guest
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .now();
        gpu_ticks_at(now).0
    }
}

/// The GPU's settings in a run: `GpuConfig`'s defaults, with every present dumped (module docs,
/// "The GPU's parts").
#[must_use]
pub(crate) fn gpu_config() -> GpuConfig {
    GpuConfig {
        exec: ExecConfig {
            dumps: FrameDumps::Every,
            ..ExecConfig::default()
        },
        ..GpuConfig::default()
    }
}

/// A run's graphics server (module docs, "Which front"), rung through `wake`, handing its front
/// no more than `frame_limit` frames, and the frames its GPU dumps into when it has one.
/// A GPU's release timestamps read `clock`, a copy of the run's clock, as guest time (module
/// docs, "The GPU's clocks"). The error is the words the run ends with when the GPU's threads
/// cannot be started. Always the recording backend: a run builds its graphics server with
/// [`graphics_rendering`], which takes the backend the run chose.
#[cfg(test)]
pub(crate) fn graphics<C: HostClock + 'static>(
    environment: &EnvSource,
    memory: Option<&Arc<dyn GuestMemory>>,
    frame_limit: Option<NonZeroU64>,
    wake: Arc<dyn Wake>,
    clock: C,
) -> Result<(Graphics, Option<Arc<RunFrames>>), String> {
    let (graphics, gpu) = graphics_and_backend(environment, memory, frame_limit, wake, clock)?;
    Ok((graphics, gpu.map(|(keeper, _)| keeper)))
}

/// The frames a run's GPU dumps into, and the render backend it was built over.
pub(crate) type KeeperAndBackend = (Arc<RunFrames>, Arc<RecordingBackend>);

/// [`graphics`], with the render backend the GPU was built over when it has one, so that a test
/// can look at what the backend holds (module docs, "The GPU's parts").
pub(crate) fn graphics_and_backend<C: HostClock + 'static>(
    environment: &EnvSource,
    memory: Option<&Arc<dyn GuestMemory>>,
    frame_limit: Option<NonZeroU64>,
    wake: Arc<dyn Wake>,
    clock: C,
) -> Result<(Graphics, Option<KeeperAndBackend>), String> {
    let shared = if builds_gpu(environment, memory.is_some()) {
        memory
    } else {
        None
    };
    let Some(memory) = shared else {
        let graphics = Graphics::new(FrontMode::Live, Some(wake)).with_frame_limit(frame_limit);
        return Ok((graphics, None));
    };
    let ring = standard_ring()?;
    let keeper = Arc::new(RunFrames::default());
    let backend = recording_backend(&ring, &keeper);
    let dyn_backend: Arc<dyn Backend> = Arc::<RecordingBackend>::clone(&backend);
    let graphics = start_gpu(ring, dyn_backend, memory, (wake, frame_limit, clock))?;
    Ok((graphics, Some((keeper, backend))))
}

/// The ring a run's GPU is built over.
fn standard_ring() -> Result<GpuRing, String> {
    GpuRing::standard().map_err(|error| format!("the GPU's ring: {error}"))
}

/// The run's recording backend (module docs, "The GPU's parts"), built with `ring`'s completion
/// wake and dumping into `keeper`.
fn recording_backend(ring: &GpuRing, keeper: &Arc<RunFrames>) -> Arc<RecordingBackend> {
    let backend = Arc::new(RecordingBackend::without_call_log(
        recording_caps(),
        RECORDING_MEMORY,
        ring.completion_wake(),
    ));
    backend.set_frame_sink(Some(Arc::<RunFrames>::clone(keeper)));
    backend
}

/// Starts the run's GPU over `ring` and `backend` (module docs, "The GPU's parts"), rung
/// through `wake`, handing its front no more than `frame_limit` frames, its release timestamps
/// read from `clock`.
fn start_gpu<C: HostClock + 'static>(
    ring: GpuRing,
    backend: Arc<dyn Backend>,
    memory: &Arc<dyn GuestMemory>,
    (wake, frame_limit, clock): (Arc<dyn Wake>, Option<NonZeroU64>, C),
) -> Result<Graphics, String> {
    let parts = GpuParts {
        ring,
        backend,
        shaders: Box::new(NoShaders),
        guest: Arc::clone(memory),
        watch: Arc::new(WriteWatch::new()),
        config: gpu_config(),
        fence_clock: Arc::new(GuestTimeClock::new(clock)),
    };
    Ok(Graphics::with_gpu(Some(wake), parts)
        .map_err(|error| format!("the GPU could not be started: {error}"))?
        .with_frame_limit(frame_limit))
}

/// A run's GPU over the recording backend (module docs, "The GPU's parts"), whose frames go to
/// the sink `sink` makes of the run's frames instead of to the frames themselves: for the tests
/// of a backend that hands its frames over late ("At the end"). The GPU, rung through `wake`,
/// its release timestamps read from `clock`, and the run's frames.
#[cfg(test)]
pub(crate) fn graphics_with_sink<C: HostClock + 'static>(
    memory: &Arc<dyn GuestMemory>,
    (wake, clock): (Arc<dyn Wake>, C),
    sink: impl FnOnce(Arc<RunFrames>) -> Arc<dyn FrameSink>,
) -> Result<(Graphics, Arc<RunFrames>), String> {
    let ring = standard_ring()?;
    let frames = Arc::new(RunFrames::default());
    let backend = recording_backend(&ring, &frames);
    backend.set_frame_sink(Some(sink(Arc::clone(&frames))));
    let graphics = start_gpu(ring, backend, memory, (wake, None, clock))?;
    Ok((graphics, frames))
}

/// The frames a run's GPU dumps into, and the render backend it was built over, as the GPU
/// takes it.
pub(crate) type KeeperAndDynBackend = (Arc<RunFrames>, Arc<dyn Backend>);

/// [`graphics`], with the render backend `render` chose (module docs, "The GPU's parts" and "The
/// wgpu backend"): what `Machine::start` builds.
pub(crate) fn graphics_rendering<C: HostClock + 'static>(
    environment: &EnvSource,
    memory: Option<&Arc<dyn GuestMemory>>,
    (frame_limit, wake, clock): (Option<NonZeroU64>, Arc<dyn Wake>, C),
    render: &RenderSetup,
) -> Result<(Graphics, Option<Arc<RunFrames>>), String> {
    let (graphics, built) = graphics_opening(
        environment,
        memory,
        (frame_limit, wake, clock),
        render,
        &WgpuOpener::host(),
    )?;
    Ok((graphics, built.map(|(keeper, _)| keeper)))
}

/// [`graphics_rendering`], opening the wgpu backend with `opener` (the host's in a run; one that
/// fails in the tests), with the backend the GPU was built over when it has one.
pub(crate) fn graphics_opening<C: HostClock + 'static>(
    environment: &EnvSource,
    memory: Option<&Arc<dyn GuestMemory>>,
    (frame_limit, wake, clock): (Option<NonZeroU64>, Arc<dyn Wake>, C),
    render: &RenderSetup,
    opener: &WgpuOpener,
) -> Result<(Graphics, Option<KeeperAndDynBackend>), String> {
    let if_unavailable = match render.choice() {
        Choice::Recording => {
            let (graphics, built) =
                graphics_and_backend(environment, memory, frame_limit, wake, clock)?;
            let built = built.map(|(keeper, backend)| {
                let backend: Arc<dyn Backend> = backend;
                (keeper, backend)
            });
            return Ok((graphics, built));
        }
        Choice::Wgpu(if_unavailable) => if_unavailable,
    };
    let shared = if builds_gpu(environment, memory.is_some()) {
        memory
    } else {
        None
    };
    let Some(memory) = shared else {
        let graphics = Graphics::new(FrontMode::Live, Some(wake)).with_frame_limit(frame_limit);
        return Ok((graphics, None));
    };
    let ring = standard_ring()?;
    let keeper = Arc::new(RunFrames::default());
    let attempt = opener.open(
        ring.completion_wake(),
        gpu_config().exec.setting,
        &keeper,
        render.window(),
        PRESENT_TARGET_WAIT,
    );
    let backend = match (attempt, if_unavailable) {
        (Ok(backend), _) => backend,
        (Err(Unopened::Backend(reason)), IfUnavailable::UseRecording) => {
            tracing::warn!(%reason, "{FELL_BACK}");
            recording_backend(&ring, &keeper)
        }
        (Err(Unopened::Backend(reason)), IfUnavailable::EndRun) => {
            return Err(format!("{WGPU_UNAVAILABLE}: {reason}"));
        }
        (Err(Unopened::Window(message)), _) => return Err(message),
    };
    let graphics = start_gpu(
        ring,
        Arc::clone(&backend),
        memory,
        (wake, frame_limit, clock),
    )?;
    Ok((graphics, Some((keeper, backend))))
}

/// The words a run ends with when a GPU thread ended during it, given what stopping the GPU
/// returned (module docs, "During the run").
#[must_use]
pub(crate) fn stopped_message(stopped: &Result<(), WiringError>) -> String {
    match stopped {
        Ok(()) => GPU_STOPPED.to_owned(),
        Err(error) => format!("{GPU_STOPPED}: {error}"),
    }
}

/// Ends the run if its GPU's threads have ended (module docs, "During the run").
pub(super) fn check_running(graphics: &Graphics) -> Result<(), Ending> {
    match graphics.gpu() {
        Some(gpu) if !gpu.is_running() => Err(Ending::Failed(stopped_message(&gpu.stop()))),
        _ => Ok(()),
    }
}

/// How a run that ended as `ending` ends once its GPU was stopped with `stopped` (module docs,
/// "At the end"): a GPU thread's failure replaces only an ending that reached the stop
/// condition.
fn after_stop(ending: Ending, stopped: &Result<(), WiringError>) -> Ending {
    match (ending, stopped) {
        (Ending::Exited | Ending::FramesReached, Err(error)) => {
            Ending::Failed(format!("{GPU_FAILED}: {error}"))
        }
        (ending, _) => ending,
    }
}

/// Waits until `wait` has passed since `started` for `gpu` to be idle; whether it was.
fn drain(gpu: &Gpu, started: Instant, wait: Duration) -> bool {
    loop {
        if gpu.is_idle() {
            return true;
        }
        if started.elapsed() >= wait {
            return false;
        }
        std::thread::sleep(DRAIN_LOOK);
    }
}

/// Waits until `wait` has passed since `started` for `frames` to have been handed a frame for
/// every present `gpu` recorded (module docs, "At the end"); whether they were. Each look reads
/// the GPU's present count again.
fn frames_arrive(gpu: &Gpu, frames: &RunFrames, started: Instant, wait: Duration) -> bool {
    loop {
        if frames.received() >= gpu.counters().presents {
            return true;
        }
        if started.elapsed() >= wait {
            return false;
        }
        std::thread::sleep(DRAIN_LOOK);
    }
}

/// What the run's graphics path leaves for its report, once every core and the timer thread
/// have stopped (module docs, "At the end"), with `ending` changed when a GPU thread failed in a
/// run that reached its stop condition.
pub(crate) fn finish(
    graphics: &Graphics,
    keeper: Option<&RunFrames>,
    ending: &mut Ending,
) -> GraphicsReport {
    let mut report = GraphicsReport {
        frames: graphics.frames(),
        ..GraphicsReport::default()
    };
    if let Some(gpu) = graphics.gpu() {
        // A GPU whose threads have ended is never idle again: there is nothing to wait for.
        if gpu.is_running() {
            let started = Instant::now();
            if !drain(gpu, started, GPU_DRAIN_WAIT) {
                tracing::warn!(
                    "the GPU was still busy when the run ended; what it had not done is not in \
                     the frame dumps"
                );
            } else if let Some(frames) = keeper
                && !frames_arrive(gpu, frames, started, GPU_DRAIN_WAIT)
            {
                tracing::warn!(
                    received = frames.received(),
                    presents = gpu.counters().presents,
                    "{FRAMES_MISSING}"
                );
            }
        }
        let stopped = gpu.stop();
        if let Err(error) = &stopped {
            tracing::error!(%error, "a GPU thread failed");
        }
        let before = std::mem::replace(ending, Ending::TimedOut);
        *ending = after_stop(before, &stopped);
        report.gpu = Some(gpu.counters());
        report.gpu_recorder = gpu.flight_recorder();
    }
    if let Some(keeper) = keeper {
        report.dumps = keeper.take();
    }
    report
}

#[cfg(test)]
mod tests;
