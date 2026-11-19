//! The graphics server with the GPU as its front ([`Graphics::with_gpu`]): the production
//! constructor, its hand-off to gsv's servers, idleness and the counters' JSON.

use std::time::{Duration, Instant};

use starquake_gpu::back::buffer::InvalidationSink;
use starquake_gpu::back::draw::shaders::{SetLookup, ShaderLookup};
use starquake_gpu::wiring::{GpuConfig, GpuCounters, GpuRing};
use starquake_memory::{FlatMemory, PageRange, WriteWatch};
use starquake_render::caps::{
    Caps, DynamicState, Features, FormatFeatures, FormatTable, Limits, SampleCounts, StageSupport,
};
use starquake_render::recording::RecordingBackend;
use starquake_shader::lookup::{EngineState, ProgramMemory, ProgramRequest, VariantKey};

use super::*;
use crate::hle::graphics::{GpuParts, Idleness};

/// How long any wait for the GPU's threads may take before the test fails.
const PATIENCE: Duration = Duration::from_secs(20);

fn caps() -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: 1 << 20,
            max_staging_size: 1 << 20,
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
    }
}

/// A shader cache with no shader (no test here draws).
struct NoShaders;

impl ShaderLookup for NoShaders {
    fn lookup(
        &mut self,
        _requests: &[ProgramRequest],
        _variant: &VariantKey,
        _memory: &dyn ProgramMemory,
        _state: &dyn EngineState,
    ) -> SetLookup {
        SetLookup::Pending(starquake_shader::lookup::Ticket::new(0))
    }

    fn wait(&mut self, ticket: starquake_shader::lookup::Ticket) -> SetLookup {
        SetLookup::Pending(ticket)
    }
}

impl InvalidationSink for NoShaders {
    fn invalidate(&mut self, _range: PageRange) {}
}

/// A graphics server whose front is the GPU, over the recording backend.
fn with_gpu() -> Graphics {
    let ring = GpuRing::new(1 << 16).unwrap();
    let backend = Arc::new(RecordingBackend::with_completion_wake(
        caps(),
        1 << 30,
        ring.completion_wake(),
    ));
    Graphics::with_gpu(
        None,
        GpuParts {
            ring,
            backend,
            shaders: Box::new(NoShaders),
            guest: Arc::new(FlatMemory::new(0x8000_0000, 0x1000)),
            watch: Arc::new(WriteWatch::new()),
            config: GpuConfig::default(),
        },
    )
    .unwrap()
}

/// Waits until `done` holds; fails the test after [`PATIENCE`].
fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + PATIENCE;
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting until {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The session of `a_run_s_turns_complete_what_waits_on_the_gpu`, with the GPU's threads
/// performing the submission's increments and the ordered destroy: each turn waits until the GPU
/// has finished and pushed what the turn drains.
#[test]
fn a_run_with_the_gpu_completes_what_waits_on_it() {
    let graphics = with_gpu();
    let gpu = Arc::clone(graphics.gpu().unwrap());
    let rig = rig_with(&graphics);
    let mut turns = Vec::new();
    let done = session(&rig, |n| {
        wait_until("the GPU has pushed its completions", || {
            gpu.is_idle() && !graphics.shared().queue().is_empty()
        });
        let before = rig.all_events().len();
        rig.turn(&graphics);
        turns.push(notes(&rig.all_events()[before..]));
        assert_eq!(turns.len(), n + 1);
    });
    assert_eq!(graphics.shared().syncpoints().read(done.syncpoint), 2);
    assert!(matches!(turns[1].as_slice(), [Note::GpuOrderedDone { .. }]));
    // The GPU did the work: its front took the submission and pushed both increments (one
    // entry with a count of 2), and its back performed them.
    let counters = gpu.counters();
    assert_eq!(counters.submissions, 1);
    assert_eq!(counters.increments, 2);
    // gsv sends the destroy with no ticket and then a barrier with the job's: two ordered.
    assert_eq!(counters.ordered, 2);
    let Some(Note::GpuOrderedDone { ticket }) = turns[1].first() else {
        unreachable!("matched above");
    };
    let channel_made = format!(
        "1 channel 0 made in address space 1, incrementing syncpoint {}\n",
        done.syncpoint.get()
    );
    let expected = [
        "0 address space 1 made\n",
        &channel_made,
        "2 submission on channel 0: 0 entries, 2 increments, wait for idle: queued\n",
        // Both increments are one `Syncpoint` entry, performed as one action.
        "3 performed 0 releases, 2 increments, 0 markers, no waiter latched\n",
        "4 address space 2 made\n",
        &format!(
            "5 ordered update of address space 2: sparse region destroy of {REGION_SIZE:#x} bytes \
             at {REGION:#x} (region 1), no ticket\n"
        ),
        &format!("6 barrier, ticket {}\n", ticket.0),
        "7 performed 0 releases, 0 increments, 1 markers, no waiter latched\n",
        &format!("8 pushed: ticket {} done\n", ticket.0),
        "9 performed 0 releases, 0 increments, 1 markers, no waiter latched\n",
    ]
    .concat();
    assert_eq!(gpu.flight_recorder(), expected);
    gpu.stop().unwrap();
}

#[test]
fn a_graphics_server_with_a_busy_gpu_is_not_idle() {
    let graphics = with_gpu();
    let gpu = graphics.gpu().unwrap();
    // While its threads run, an idle GPU still takes a turn at each safety-net wake
    // (`IDLE_PARK`) and reads busy during it, so "idle" is waited for, never asserted once.
    wait_until("a new GPU is idle", || gpu.is_idle());
    wait_until("the graphics server over an idle GPU is idle", || {
        graphics.is_idle()
    });
    // A stopped GPU answers "busy" for good (`Gpu::is_idle`), so the server is busy though gsv's
    // queue is empty: the GPU is asked, not only the queue.
    gpu.stop().unwrap();
    assert!(graphics.shared().queue().is_empty());
    assert!(!gpu.is_idle());
    assert!(!graphics.is_idle());
    // The busy answer is the GPU's alone; with something queued too, the server is busy for
    // more than the GPU.
    assert_eq!(graphics.idleness(), Idleness::GpuBusy);
    graphics
        .shared()
        .queue()
        .push(starquake_gsv::completion::Completion::Gpu(
            starquake_gpu::interface::GpuCompletion::GpuOrdered {
                ticket: starquake_gpu::interface::Ticket(0x93),
            },
        ));
    assert_eq!(graphics.idleness(), Idleness::Busy);
    assert!(!graphics.is_idle());
}

/// A running GPU that is idle leaves the server idle once its turn ends; with a completion
/// queued the server reads busy, not `GpuBusy`, whatever the GPU answers. An idle GPU does read
/// busy while a safety-net turn runs, so the server can read `GpuBusy` meanwhile: idle is
/// waited for, not asserted once.
#[test]
fn a_graphics_server_over_an_idle_gpu_becomes_idle_and_a_queued_completion_reads_busy() {
    let graphics = with_gpu();
    wait_until("the server over a new GPU is idle", || {
        graphics.idleness() == Idleness::Idle
    });
    graphics
        .shared()
        .queue()
        .push(starquake_gsv::completion::Completion::Gpu(
            starquake_gpu::interface::GpuCompletion::GpuOrdered {
                ticket: starquake_gpu::interface::Ticket(0x94),
            },
        ));
    // Whatever the GPU answers, the queue makes the server busy.
    for _ in 0..20 {
        assert_eq!(graphics.idleness(), Idleness::Busy);
    }
    graphics.gpu().unwrap().stop().unwrap();
}

#[test]
fn each_front_is_reached_through_its_own_accessor() {
    let gpu = with_gpu();
    assert!(gpu.gpu().is_some());
    assert!(gpu.null_front().is_none());
    let null = Graphics::new(FrontMode::Live, None);
    assert!(null.gpu().is_none());
    assert!(null.null_front().is_some());
    // The null front never reports a channel error.
    assert_eq!(
        null.front()
            .channel_error(starquake_gpu::interface::ChannelId(1)),
        None
    );
}

#[test]
fn the_gpus_counters_json_is_pinned() {
    // `summary.json`'s GPU counters (gpu-protocol, "Observability"): every field a distinct
    // non-zero value, so a swapped or renamed field changes the text.
    let counters = GpuCounters {
        words: 1,
        submissions: 2,
        submissions_full: 3,
        submissions_dropped: 4,
        increments: 5,
        acquires_parked: 6,
        fence_waits: 7,
        sync_waits: 8,
        macro_catch_ups: 9,
        polls: 10,
        ordered: 11,
        macro_failures: 12,
        channels_stopped: 13,
        presents_pushed: 14,
        entries_applied: 15,
        entries_refused: 16,
        unhandled_words: 17,
        captures: 18,
        presents: 19,
        dispatches: 20,
        dispatches_skipped: 21,
        write_backs: 22,
        write_back_failures: 23,
        render_catch_ups: 24,
        render_catch_up_failures: 25,
        fenced_actions: 26,
        fenced_actions_skipped_empty: 27,
        lists_submitted: 28,
        lost_wakes: 29,
        failed_releases: 30,
        back_wait_failures: 31,
        refused_controls: 32,
    };
    assert_eq!(
        serde_json::to_string(&counters).unwrap(),
        "{\"words\":1,\"submissions\":2,\"submissions_full\":3,\"submissions_dropped\":4,\
         \"increments\":5,\"acquires_parked\":6,\"fence_waits\":7,\"sync_waits\":8,\
         \"macro_catch_ups\":9,\"polls\":10,\"ordered\":11,\"macro_failures\":12,\
         \"channels_stopped\":13,\"presents_pushed\":14,\"entries_applied\":15,\
         \"entries_refused\":16,\"unhandled_words\":17,\"captures\":18,\"presents\":19,\
         \"dispatches\":20,\"dispatches_skipped\":21,\"write_backs\":22,\
         \"write_back_failures\":23,\"render_catch_ups\":24,\"render_catch_up_failures\":25,\
         \"fenced_actions\":26,\"fenced_actions_skipped_empty\":27,\"lists_submitted\":28,\
         \"lost_wakes\":29,\"failed_releases\":30,\"back_wait_failures\":31,\
         \"refused_controls\":32}"
    );
}
