//! The run's GPU: which front a run gets, the settings and parts it is built with, and the words
//! a dead GPU ends a run with.

use starquake_gpu::wiring::BackError;
use starquake_memory::FlatMemory;

use super::*;

/// A wake that does nothing.
struct NoWake;

impl Wake for NoWake {
    fn completion_queued(&self) {}
}

#[test]
fn only_a_live_run_with_shared_memory_builds_the_gpu() {
    let replay = EnvSource::Replay(Vec::new());
    for (environment, shared, builds) in [
        (&EnvSource::Live, true, true),
        (&EnvSource::Live, false, false),
        (&replay, true, false),
        (&replay, false, false),
    ] {
        assert_eq!(
            builds_gpu(environment, shared),
            builds,
            "{environment:?}, {shared}"
        );
    }
}

#[test]
fn each_combination_gets_its_front() {
    let memory: Arc<dyn GuestMemory> = Arc::new(FlatMemory::new(0x8000_0000, 0x1000));
    let replay = EnvSource::Replay(Vec::new());
    for (environment, memory, gpu) in [
        (&EnvSource::Live, Some(&memory), true),
        (&EnvSource::Live, None, false),
        (&replay, Some(&memory), false),
        (&replay, None, false),
    ] {
        let (graphics, keeper) = super::graphics(environment, memory, None, Arc::new(NoWake))
            .unwrap_or_else(|error| unreachable!("{error}"));
        assert_eq!(graphics.gpu().is_some(), gpu, "{environment:?}");
        assert_eq!(graphics.null_front().is_some(), !gpu, "{environment:?}");
        assert_eq!(keeper.is_some(), gpu, "{environment:?}");
        if let Some(gpu) = graphics.gpu() {
            assert!(gpu.is_running());
            gpu.stop().unwrap();
        }
    }
}

#[test]
fn the_frame_limit_reaches_either_front() {
    let memory: Arc<dyn GuestMemory> = Arc::new(FlatMemory::new(0x8000_0000, 0x1000));
    let limit = NonZeroU64::new(1);
    for memory in [Some(&memory), None] {
        let (graphics, _) = super::graphics(&EnvSource::Live, memory, limit, Arc::new(NoWake))
            .unwrap_or_else(|error| unreachable!("{error}"));
        // Nothing handed yet: the limit of one is not reached; with no limit it never is.
        assert!(!graphics.frame_limit_reached());
        let (unlimited, _) = super::graphics(&EnvSource::Live, memory, None, Arc::new(NoWake))
            .unwrap_or_else(|error| unreachable!("{error}"));
        assert!(!unlimited.frame_limit_reached());
        for built in [&graphics, &unlimited] {
            if let Some(gpu) = built.gpu() {
                gpu.stop().unwrap();
            }
        }
    }
}

#[test]
fn a_run_asks_the_gpu_to_dump_every_present() {
    let config = gpu_config();
    assert_eq!(config.exec.dumps, FrameDumps::Every);
    // Everything else is the GPU's default.
    let defaults = GpuConfig::default();
    assert_eq!(
        GpuConfig {
            exec: ExecConfig {
                dumps: defaults.exec.dumps,
                ..config.exec
            },
            ..config
        },
        defaults
    );
}

#[test]
fn the_recording_backend_reports_wgpus_default_limits() {
    // wgpu-types 30.0.1, `Limits::defaults` (`src/limits.rs`), written out.
    let caps = recording_caps();
    let limits = caps.limits;
    assert_eq!(limits.max_buffer_size, 256 << 20);
    assert_eq!(limits.max_staging_size, 256 << 20);
    assert_eq!(
        (
            limits.max_texture_dimension_1d,
            limits.max_texture_dimension_2d,
            limits.max_texture_dimension_3d,
            limits.max_texture_array_layers,
        ),
        (8192, 8192, 2048, 256)
    );
    assert_eq!(
        (
            limits.max_vertex_attributes,
            limits.max_vertex_buffers,
            limits.max_vertex_buffer_stride,
            limits.max_color_attachments,
        ),
        (16, 8, 2048, 8)
    );
    assert_eq!(
        (
            limits.max_bind_groups,
            limits.max_bindings_per_bind_group,
            limits.max_compute_workgroups_per_dimension,
        ),
        (4, 1000, 65535)
    );
    assert_eq!(
        (
            limits.max_uniform_buffers_per_stage,
            limits.max_storage_buffers_per_stage,
            limits.max_sampled_textures_per_stage,
            limits.max_storage_textures_per_stage,
            limits.max_samplers_per_stage,
        ),
        (12, 8, 16, 4, 16)
    );
    // render-wgpu's `WGPU_MAX_ANISOTROPY` and `WGPU_VIEWPORTS`.
    assert_eq!(
        (limits.max_sampler_anisotropy, limits.max_viewports),
        (16, 1)
    );
    assert_eq!(caps.features, Features::ALL);
    assert_eq!(caps.stages, StageSupport::ALL);
    assert_eq!(caps.dynamic_state, DynamicState::ALL);
    assert!(caps.sample_counts.supports(1) && caps.sample_counts.supports(4));
    assert!(!caps.sample_counts.supports(2) && !caps.sample_counts.supports(8));
    assert!(caps.view_swizzle);
    assert_eq!(caps.formats, FormatTable::uniform(FormatFeatures::ALL));
    assert_eq!(RECORDING_MEMORY, 1 << 30);
}

#[test]
fn a_wait_for_no_shaders_is_still_pending() {
    let mut shaders = NoShaders;
    let ticket = Ticket::new(9);
    assert!(matches!(shaders.wait(ticket), SetLookup::Pending(t) if t == ticket));
}

#[test]
fn a_dead_gpus_words_are_plain_and_carry_its_error() {
    assert_eq!(stopped_message(&Ok(())), GPU_STOPPED);
    assert_eq!(GPU_STOPPED, "the GPU's threads ended during the run");
    let failed = Err(WiringError::Back(BackError::Panicked));
    assert_eq!(
        stopped_message(&failed),
        format!("{GPU_STOPPED}: {}", BackError::Panicked)
    );
    assert_eq!(GPU_FAILED, "a GPU thread failed");
}

#[test]
fn a_stopped_gpu_ends_the_run_and_a_running_one_does_not() {
    let memory: Arc<dyn GuestMemory> = Arc::new(FlatMemory::new(0x8000_0000, 0x1000));
    let (graphics, _) = super::graphics(&EnvSource::Live, Some(&memory), None, Arc::new(NoWake))
        .unwrap_or_else(|error| unreachable!("{error}"));
    assert!(check_running(&graphics).is_ok());
    graphics.gpu().unwrap().stop().unwrap();
    match check_running(&graphics) {
        Err(Ending::Failed(message)) => assert_eq!(message, GPU_STOPPED),
        _ => unreachable!("a stopped GPU must end the run"),
    }
    // The null front has no threads to end.
    let null = Graphics::new(FrontMode::Live, None);
    assert!(check_running(&null).is_ok());
}

#[test]
fn finishing_reads_the_gpu_and_keeps_a_clean_ending() {
    let memory: Arc<dyn GuestMemory> = Arc::new(FlatMemory::new(0x8000_0000, 0x1000));
    let (graphics, keeper) =
        super::graphics(&EnvSource::Live, Some(&memory), None, Arc::new(NoWake))
            .unwrap_or_else(|error| unreachable!("{error}"));
    graphics
        .front()
        .barrier(starquake_gpu::interface::Ticket(7));
    for (clean, name) in [
        (Ending::Exited, "exited"),
        (Ending::FramesReached, "frames reached"),
    ] {
        let mut ending = clean;
        let report = finish(&graphics, keeper.as_deref(), &mut ending);
        assert_eq!(said(&ending), name);
        let counters = report.gpu.unwrap();
        assert_eq!(counters.ordered, 1);
        assert_eq!(
            report.gpu_recorder,
            graphics.gpu().unwrap().flight_recorder()
        );
        assert!(report.gpu_recorder.contains("barrier, ticket 7"));
        assert_eq!(report.frames, 0);
        assert!(!graphics.gpu().unwrap().is_running());
    }
    // The null front leaves no GPU report.
    let null = Graphics::new(FrontMode::Live, None);
    let mut ending = Ending::Exited;
    let report = finish(&null, None, &mut ending);
    assert_eq!(report, GraphicsReport::default());
    assert_eq!(said(&ending), "exited");
}

/// An ending in words, so that two can be compared.
fn said(ending: &Ending) -> String {
    match ending {
        Ending::Exited => "exited".to_owned(),
        Ending::FramesReached => "frames reached".to_owned(),
        Ending::TimedOut => "timed out".to_owned(),
        Ending::Failed(message) => format!("failed: {message}"),
        Ending::Crashed(exception, ..) => format!("crashed: {exception:?}"),
    }
}

#[test]
fn a_failed_gpu_thread_replaces_only_an_ending_that_reached_the_stop_condition() {
    let crashed = || {
        Ending::Crashed(
            crate::crash::Exception::UndefinedInstruction,
            None,
            Box::default(),
        )
    };
    let endings: [(fn() -> Ending, bool); 5] = [
        (|| Ending::Exited, true),
        (|| Ending::FramesReached, true),
        (|| Ending::TimedOut, false),
        (|| Ending::Failed("the guest's own".to_owned()), false),
        (crashed, false),
    ];
    let failure = Err(WiringError::Back(BackError::Panicked));
    let expected_failure = format!("failed: {GPU_FAILED}: {}", BackError::Panicked);
    for (ending, replaced) in endings {
        let before = said(&ending());
        // A clean stop changes no ending.
        assert_eq!(said(&after_stop(ending(), &Ok(()))), before);
        let after = said(&after_stop(ending(), &failure));
        if replaced {
            assert_eq!(after, expected_failure, "{before}");
        } else {
            assert_eq!(after, before);
        }
    }
}
