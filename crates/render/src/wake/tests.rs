//! The completion wake and its contract check: the recording backend passes, and each way a
//! backend can fail is reported as its own error.

use super::{CompletionWake, WakeContractError, WakeCounter, check_completion_wake};
use crate::backend::{
    Backend, MemoryBudget, PipelineStatus, PresentFeedback, PresentTarget, Readback, WaitOutcome,
};
use crate::caps::{
    Caps, DynamicState, Features, FormatFeatures, FormatTable, Limits, SampleCounts, StageSupport,
};
use crate::command::{CommandList, StagingSlice};
use crate::error::RenderError;
use crate::id::{
    BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId,
    ViewId,
};
use crate::pipeline::{ComputePipelineDesc, GraphicsPipelineDesc};
use crate::recording::RecordingBackend;
use crate::resource::{BufferDesc, SamplerDesc, TextureDesc, ViewDesc};
use crate::shader::ShaderDesc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Long enough that a backend that does wake is never cut short.
const PATIENCE: Duration = Duration::from_secs(10);
/// How long the failing cases wait for what never comes.
const SHORT: Duration = Duration::from_millis(20);

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

#[test]
fn the_recording_backend_meets_the_contract() {
    let wakes = Arc::new(WakeCounter::new());
    let backend = RecordingBackend::with_completion_wake(caps(), 0, wakes.completion_wake());
    assert_eq!(check_completion_wake(&backend, &wakes, 3, PATIENCE), Ok(()));
    // One wake per list: the recording backend completes each at its submit.
    assert_eq!(wakes.count(), 3);
    assert_eq!(backend.completed(), FenceValue::new(3));
}

#[test]
fn a_backend_that_never_wakes_fails_at_its_first_list() {
    let wakes = Arc::new(WakeCounter::new());
    // Built without the counter's wake: it completes lists and never calls a wake.
    let backend = RecordingBackend::new(caps(), 0);
    let error = check_completion_wake(&backend, &wakes, 3, SHORT);
    assert_eq!(error, Err(WakeContractError::NoWake { list: 0, fence: 1 }));
    assert_eq!(
        error.map_err(|e| e.to_string()),
        Err("list 0 (fence value 1) completed and no wake followed".to_owned())
    );
}

/// What [`Broken`] does wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    /// Refuses every list.
    RefuseSubmit,
    /// `wait` fails.
    FailWait,
    /// Reports a fence value one past what completes, so `wait` times out.
    NeverComplete,
    /// Wakes for the first list only.
    WakeFirstOnly,
}

/// A recording backend with one fault, for the contract check's failure cases.
struct Broken {
    inner: RecordingBackend,
    wake: CompletionWake,
    fault: Fault,
    submits: AtomicU64,
}

impl Broken {
    fn new(fault: Fault, wakes: &Arc<WakeCounter>) -> Self {
        Self {
            inner: RecordingBackend::new(caps(), 0),
            wake: wakes.completion_wake(),
            fault,
            submits: AtomicU64::new(0),
        }
    }
}

const REFUSED: RenderError = RenderError::Invalid("this test backend refuses every list");
const WAIT_FAILED: RenderError = RenderError::Invalid("this test backend cannot wait");

impl Backend for Broken {
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
        if self.fault == Fault::RefuseSubmit {
            return Err(REFUSED);
        }
        let fence = self.inner.submit(list)?;
        let first = self.submits.fetch_add(1, Ordering::SeqCst) == 0;
        match self.fault {
            Fault::NeverComplete => Ok(fence.next().unwrap_or(fence)),
            Fault::WakeFirstOnly if first => {
                self.wake.wake();
                Ok(fence)
            }
            _ => Ok(fence),
        }
    }
    fn completed(&self) -> FenceValue {
        self.inner.completed()
    }
    fn wait(&self, value: FenceValue, timeout: Duration) -> Result<WaitOutcome, RenderError> {
        if self.fault == Fault::FailWait {
            return Err(WAIT_FAILED);
        }
        self.inner.wait(value, timeout)
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

#[test]
fn each_failure_of_the_check_is_its_own_error() {
    let cases = [
        (
            Fault::RefuseSubmit,
            WakeContractError::Submit {
                list: 0,
                error: REFUSED,
            },
            "the backend refused list 0: this test backend refuses every list",
        ),
        (
            Fault::FailWait,
            WakeContractError::Wait {
                list: 0,
                error: WAIT_FAILED,
            },
            "waiting for list 0 failed: this test backend cannot wait",
        ),
        (
            Fault::NeverComplete,
            WakeContractError::NotCompleted { list: 0, fence: 2 },
            "list 0 (fence value 2) did not complete in time",
        ),
        (
            // The first list passes, so the failure names the second: the list index is the
            // list's own, not a constant.
            Fault::WakeFirstOnly,
            WakeContractError::NoWake { list: 1, fence: 2 },
            "list 1 (fence value 2) completed and no wake followed",
        ),
    ];
    for (fault, error, text) in cases {
        let wakes = Arc::new(WakeCounter::new());
        let backend = Broken::new(fault, &wakes);
        let result = check_completion_wake(&backend, &wakes, 3, SHORT);
        assert_eq!(result, Err(error), "{fault:?}");
        assert_eq!(
            result.map_err(|e| e.to_string()),
            Err(text.to_owned()),
            "{fault:?}"
        );
    }
}

#[test]
fn zero_lists_check_nothing() {
    let wakes = Arc::new(WakeCounter::new());
    let backend = Broken::new(Fault::RefuseSubmit, &wakes);
    assert_eq!(check_completion_wake(&backend, &wakes, 0, SHORT), Ok(()));
}

#[test]
fn the_counter_waits_for_a_count_strictly_beyond_the_one_seen() {
    let wakes = Arc::new(WakeCounter::new());
    assert_eq!(wakes.count(), 0);
    assert!(!wakes.wait_beyond(0, Duration::ZERO));
    let wake = wakes.completion_wake();
    wake.wake();
    assert_eq!(wakes.count(), 1);
    assert!(wakes.wait_beyond(0, Duration::ZERO));
    // Exactly at the count seen is not beyond it.
    assert!(!wakes.wait_beyond(1, Duration::ZERO));
    // A clone rings the same counter.
    wake.clone().wake();
    assert_eq!(wakes.count(), 2);
}

#[test]
fn the_counter_waits_while_the_count_is_the_one_seen() {
    // One wake already counted; a thread waits for a count beyond it, and a second wake comes
    // later. The waiting thread must still be waiting then, not have given up because the
    // count equalled the one it saw. (Whether the thread reached its wait before the second
    // wake depends on scheduling; the outcome for a correct counter does not.)
    let wakes = Arc::new(WakeCounter::new());
    let wake = wakes.completion_wake();
    wake.wake();
    let ready = Arc::new(std::sync::Barrier::new(2));
    let waiter = {
        let wakes = Arc::clone(&wakes);
        let ready = Arc::clone(&ready);
        std::thread::spawn(move || {
            ready.wait();
            wakes.wait_beyond(1, PATIENCE)
        })
    };
    ready.wait();
    std::thread::sleep(Duration::from_millis(50));
    wake.wake();
    assert!(waiter.join().unwrap());
}

#[test]
fn the_counter_wakes_a_thread_that_waits_for_it() {
    let wakes = Arc::new(WakeCounter::new());
    let waiter = {
        let wakes = Arc::clone(&wakes);
        std::thread::spawn(move || wakes.wait_beyond(0, PATIENCE))
    };
    wakes.completion_wake().wake();
    assert!(waiter.join().unwrap());
}

#[test]
fn a_wake_calls_its_function_and_none_calls_nothing() {
    let calls = Arc::new(AtomicU64::new(0));
    let wake = {
        let calls = Arc::clone(&calls);
        CompletionWake::new(move || {
            calls.fetch_add(1, Ordering::SeqCst);
        })
    };
    wake.wake();
    wake.clone().wake();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    CompletionWake::none().wake();
    assert_eq!(format!("{wake:?}"), "CompletionWake");
}
