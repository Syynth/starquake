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
//! - **The render backend** is the recording backend ([`RecordingBackend`], render-backend "The
//!   recording backend": "the `null` backend for runs that need no pixels"), built with the
//!   ring's completion wake, with [`recording_caps`] and [`RECORDING_MEMORY`] as its budget.
//!   Choosing the wgpu backend is render-backend slice R12's.
//! - **Frame dumps**: the GPU is asked to dump every present
//!   ([`FrameDumps::Every`] in `GpuConfig::exec`, the setting through which `core` asks for
//!   dumps), and the backend hands them to the run's [`FrameKeeper`] (`crate::frames`).
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
//! - The GPU's clocks are the host's (`Graphics::with_gpu`), and its other settings
//!   `GpuConfig`'s defaults.
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
//! ([`Gpu::is_idle`]), at most [`GPU_DRAIN_WAIT`], so that the presents handed to it before the
//! end reach the frame keeper; a GPU still busy then is logged at warning level and stopped all
//! the same, and what it had not done is not in the dumps. It then stops the GPU
//! ([`Gpu::stop`]) and reads its counters ([`Gpu::counters`], `summary.json`'s `gpu`), its
//! flight recorder ([`Gpu::flight_recorder`], `gpu.txt`) and the frame keeper. A GPU thread that
//! failed makes a run that otherwise reached its stop condition an emulator error
//! ([`GPU_FAILED`]); a crash, a timeout or an emulator error keeps its own reason, which matters
//! more, and the failure is logged (`after_stop`, tested for every ending with and without a
//! failure). A GPU whose threads already ended is not waited for: it can never be idle again.

use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::{Duration, Instant};

use starquake_gpu::back::buffer::InvalidationSink;
use starquake_gpu::back::draw::shaders::{SetLookup, ShaderLookup};
use starquake_gpu::back::exec::{ExecConfig, FrameDumps};
use starquake_gpu::wiring::{Gpu, GpuConfig, GpuRing, WiringError};
use starquake_gsv::completion::Wake;
use starquake_memory::{GuestMemory, PageRange, WriteWatch};
use starquake_render::caps::{DynamicState, StageSupport};
use starquake_render::{
    Caps, Features, FormatFeatures, FormatTable, Limits, RecordingBackend, SampleCounts,
};
use starquake_shader::lookup::{EngineState, ProgramMemory, ProgramRequest, Ticket, VariantKey};

use super::Ending;
use crate::frames::FrameKeeper;
use crate::hle::graphics::{FrontMode, GpuParts};
use crate::hle::Graphics;
use crate::report::{EnvSource, GraphicsReport};

/// The memory budget the run's recording backend reports: starquake's own number. The recording
/// backend keeps a copy of every resource in host memory; the GPU's caches evict to stay within
/// the budget it reports.
pub(crate) const RECORDING_MEMORY: u64 = 1 << 30;

/// The longest [`finish`] waits for the GPU to be idle at the end of a run: starquake's own
/// number, a hundred times `starquake_gpu::wiring::IDLE_POLL_WAIT`, the longest one look at its
/// idleness waits.
pub(crate) const GPU_DRAIN_WAIT: Duration = Duration::from_secs(1);

/// How long [`finish`] sleeps between two looks at the GPU's idleness.
const DRAIN_LOOK: Duration = Duration::from_millis(1);

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
/// no more than `frame_limit` frames, and the frame keeper its GPU dumps into when it has one.
/// The error is the words the run ends with when the GPU's threads cannot be started.
pub(crate) fn graphics(
    environment: &EnvSource,
    memory: Option<&Arc<dyn GuestMemory>>,
    frame_limit: Option<NonZeroU64>,
    wake: Arc<dyn Wake>,
) -> Result<(Graphics, Option<Arc<FrameKeeper>>), String> {
    let shared = if builds_gpu(environment, memory.is_some()) {
        memory
    } else {
        None
    };
    let Some(memory) = shared else {
        let graphics = Graphics::new(FrontMode::Live, Some(wake)).with_frame_limit(frame_limit);
        return Ok((graphics, None));
    };
    let ring = GpuRing::standard().map_err(|error| format!("the GPU's ring: {error}"))?;
    let backend = Arc::new(RecordingBackend::with_completion_wake(
        recording_caps(),
        RECORDING_MEMORY,
        ring.completion_wake(),
    ));
    let keeper = Arc::new(FrameKeeper::default());
    backend.set_frame_sink(Some(Arc::<FrameKeeper>::clone(&keeper)));
    let parts = GpuParts {
        ring,
        backend,
        shaders: Box::new(NoShaders),
        guest: Arc::clone(memory),
        watch: Arc::new(WriteWatch::new()),
        config: gpu_config(),
    };
    let graphics = Graphics::with_gpu(Some(wake), parts)
        .map_err(|error| format!("the GPU could not be started: {error}"))?
        .with_frame_limit(frame_limit);
    Ok((graphics, Some(keeper)))
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

/// Waits at most `wait` for `gpu` to be idle; whether it was.
fn drain(gpu: &Gpu, wait: Duration) -> bool {
    let started = Instant::now();
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

/// What the run's graphics path leaves for its report, once every core and the timer thread
/// have stopped (module docs, "At the end"), with `ending` changed when a GPU thread failed in a
/// run that reached its stop condition.
pub(super) fn finish(
    graphics: &Graphics,
    keeper: Option<&FrameKeeper>,
    ending: &mut Ending,
) -> GraphicsReport {
    let mut report = GraphicsReport {
        frames: graphics.frames(),
        ..GraphicsReport::default()
    };
    if let Some(gpu) = graphics.gpu() {
        // A GPU whose threads have ended is never idle again: there is nothing to wait for.
        if gpu.is_running() && !drain(gpu, GPU_DRAIN_WAIT) {
            tracing::warn!(
                "the GPU was still busy when the run ended; what it had not done is not in the \
                 frame dumps"
            );
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
