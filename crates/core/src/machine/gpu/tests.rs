//! The run's GPU: which front a run gets, the settings and parts it is built with, and the words
//! a dead GPU ends a run with.

use starquake_gpu::wiring::BackError;
use starquake_memory::FlatMemory;
use starquake_trace::SystemClock;

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
        let (graphics, keeper) = super::graphics(
            environment,
            memory,
            None,
            Arc::new(NoWake),
            SystemClock::new(),
        )
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
        let (graphics, _) = super::graphics(
            &EnvSource::Live,
            memory,
            limit,
            Arc::new(NoWake),
            SystemClock::new(),
        )
        .unwrap_or_else(|error| unreachable!("{error}"));
        // Nothing handed yet: the limit of one is not reached; with no limit it never is.
        assert!(!graphics.frame_limit_reached());
        let (unlimited, _) = super::graphics(
            &EnvSource::Live,
            memory,
            None,
            Arc::new(NoWake),
            SystemClock::new(),
        )
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
    let (graphics, _) = super::graphics(
        &EnvSource::Live,
        Some(&memory),
        None,
        Arc::new(NoWake),
        SystemClock::new(),
    )
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
    let (graphics, keeper) = super::graphics(
        &EnvSource::Live,
        Some(&memory),
        None,
        Arc::new(NoWake),
        SystemClock::new(),
    )
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

/// A run's clock held at one reading, counting how often it is read (by every copy).
#[derive(Clone)]
struct HeldClock {
    now: starquake_trace::HostNanos,
    reads: Arc<std::sync::atomic::AtomicUsize>,
}

impl HeldClock {
    fn at(nanos: u64) -> Self {
        Self {
            now: starquake_trace::HostNanos(nanos),
            reads: Arc::default(),
        }
    }

    fn reads(&self) -> usize {
        self.reads.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl HostClock for HeldClock {
    fn now(&mut self) -> starquake_trace::HostNanos {
        self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.now
    }
}

/// The run's clock in the GPU tests below: 1,000,000,007 ns, which is 614,400,004 ticks of the
/// guest GPU's timer (614.4 MHz; `starquake_kernel::time`'s
/// `the_gpu_timer_shares_guest_time_s_zero_and_rounds_down_at_a_tick` works the same reading
/// by hand), 31,250,000 Switch 2 counter ticks and 19,200,000 Switch 1 ones: every unit gives a
/// different number.
const RUN_NOW: u64 = 1_000_000_007;
const RUN_NOW_GPU_TICKS: u64 = 614_400_004;

#[test]
fn the_back_s_timestamp_is_the_kernel_s_gpu_timer_reading_of_the_run_s_clock() {
    let run = HeldClock::at(RUN_NOW);
    let clock = GuestTimeClock::new(run.clone());
    assert_eq!(clock.gpu_timestamp(), RUN_NOW_GPU_TICKS);
    assert_eq!(
        clock.gpu_timestamp(),
        gpu_ticks_at(starquake_trace::HostNanos(RUN_NOW)).0
    );
    for other_unit in [
        RUN_NOW,
        starquake_kernel::time::CounterRate::SWITCH_2
            .ticks_at(starquake_trace::HostNanos(RUN_NOW))
            .0,
        starquake_kernel::time::CounterRate::SWITCH_1
            .ticks_at(starquake_trace::HostNanos(RUN_NOW))
            .0,
    ] {
        assert_ne!(clock.gpu_timestamp(), other_unit);
    }
    // Each timestamp is one reading of the run's clock; the time a list has been open is the
    // host's and reads it not at all.
    let reads = run.reads();
    let _ = clock.elapsed();
    assert_eq!(run.reads(), reads);
    let _ = clock.gpu_timestamp();
    assert_eq!(run.reads(), reads + 1);
}

/// The production wiring end to end (issue #435, "Tests"): a run's GPU, built by [`graphics`]
/// over a run's clock held still, performs a 16-byte semaphore release whose timestamp is the
/// guest GPU timer's reading of that clock.
#[test]
fn a_run_s_gpu_writes_guest_time_after_a_release_s_payload() {
    use starquake_gpu::classes::clc76f;
    use starquake_gpu::fifo::{GpEntry, MethodMode, PushBuffer, Sync};
    use starquake_gpu::interface::{
        AsId, ChannelConfig, GpfifoEntry, Kind, Submission, Submitted, SyncpointId, Target,
    };
    use starquake_gpu::sema::ReleaseForm;
    use starquake_gpu::sema::{self, HostSemaphore, PayloadSize, RELEASE_TIMESTAMP_OFFSET};
    use starquake_gpu::vm::GpuVm;
    use starquake_memory::PAGE_SIZE;

    const CPU_BASE: u64 = 0x8000_0000;
    const PAGES: u64 = 4;
    // Not equal to the CPU address, so a missed translation writes elsewhere.
    const GPU_BASE: u64 = 0x12_3400_0000;
    const SEM_OFFSET: u64 = PAGE_SIZE + 0x10;
    const PAYLOAD: u32 = 0x5a5a_0001;
    const SPACE: AsId = AsId(0x21);
    const PATIENCE: Duration = Duration::from_secs(20);

    let flat = Arc::new(FlatMemory::new(
        CPU_BASE,
        usize::try_from(PAGES * PAGE_SIZE).unwrap(),
    ));
    let memory: Arc<dyn GuestMemory> = Arc::<FlatMemory>::clone(&flat);
    let run = HeldClock::at(RUN_NOW);
    let (graphics, _) = super::graphics(
        &EnvSource::Live,
        Some(&memory),
        None,
        Arc::new(NoWake),
        run.clone(),
    )
    .unwrap_or_else(|error| unreachable!("{error}"));
    let vm = Arc::new(GpuVm::new());
    vm.map(
        PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(),
        Target::Guest { cpu_va: CPU_BASE },
        Kind(0),
    )
    .unwrap();
    let front = graphics.front();
    front.create_address_space(SPACE, vm);
    let channel = front
        .create_channel(ChannelConfig {
            space: SPACE,
            syncpoint: SyncpointId::new(5).unwrap(),
        })
        .unwrap();

    let form = ReleaseForm {
        size: PayloadSize::Four,
        timestamp: true,
    };
    let operation = sema::encode_semaphored(&HostSemaphore::Release { form, wfi: false }).unwrap();
    let semaphore = GPU_BASE + SEM_OFFSET;
    let mut pb = PushBuffer::new();
    pb.methods(
        MethodMode::Incrementing,
        0,
        u16::try_from(clc76f::SEMAPHOREA.method()).unwrap(),
        &[
            u32::try_from(semaphore >> 32).unwrap(),
            u32::try_from(semaphore & 0xffff_ffff).unwrap(),
            PAYLOAD,
            operation,
        ],
    )
    .unwrap();
    let words = pb.into_words();
    let stream: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    flat.write(CPU_BASE, &stream).unwrap();
    let entry = GpEntry {
        address: GPU_BASE,
        length: u32::try_from(words.len()).unwrap(),
        sync: Sync::Proceed,
        ..GpEntry::default()
    };
    let submitted = front.submit(
        channel,
        Submission {
            entries: vec![GpfifoEntry(entry.to_raw().unwrap())],
            wait: None,
            increments: 0,
            wfi: false,
        },
    );
    assert_eq!(submitted, Submitted::Queued);

    let written = usize::try_from(form.bytes()).unwrap();
    let mut bytes = vec![0; written];
    let deadline = Instant::now() + PATIENCE;
    loop {
        flat.read(CPU_BASE + SEM_OFFSET, &mut bytes).unwrap();
        if bytes[..4] == PAYLOAD.to_le_bytes() {
            break;
        }
        assert!(Instant::now() < deadline, "the release was never written");
        std::thread::sleep(Duration::from_millis(1));
    }
    let at = usize::try_from(RELEASE_TIMESTAMP_OFFSET).unwrap();
    assert_eq!(bytes[..at], u64::from(PAYLOAD).to_le_bytes());
    assert_eq!(bytes[at..], RUN_NOW_GPU_TICKS.to_le_bytes());
    assert!(run.reads() >= 1);
    graphics.gpu().unwrap().stop().unwrap();
}

/// The hand-off `Machine::start` makes ([`crate::machine::RunClock::for_gpu`]; module docs,
/// "The GPU's clocks"): under `--clock fixed` the GPU's copy starts from the run clock's state.
/// The loop has read the clock `READ` times, so the GPU's first timestamp is the guest GPU
/// timer's reading of the clock's next reading, `START + READ * STEP`; its second is one step on.
#[test]
fn the_gpu_s_copy_of_a_fixed_clock_starts_where_the_loop_s_copy_is() {
    use starquake_trace::{FixedClock, HostNanos};

    use crate::clock::{NRO_CONSOLE, Timebase};
    use crate::machine::RunClock;

    // Not a whole number of GPU ticks, so a reading off by a step or by the zero shows.
    const START: u64 = 1_000_000_007;
    const STEP: u64 = 1_000_003;
    const READ: u64 = 3;

    let run = RunClock::new(Timebase {
        console: NRO_CONSOLE,
        clock: FixedClock::new(START, STEP),
    });
    for _ in 0..READ {
        let _ = run.now();
    }
    let clock = GuestTimeClock::new(run.for_gpu());
    let next = START + READ * STEP;
    assert_eq!(clock.gpu_timestamp(), gpu_ticks_at(HostNanos(next)).0);
    assert_eq!(
        clock.gpu_timestamp(),
        gpu_ticks_at(HostNanos(next + STEP)).0
    );
    // The copy is the GPU's own: reading it does not move the loop's copy.
    assert_eq!(run.host_now(), HostNanos(next));
}

/// The live memory a GPU is built over in the tests of the backend choice.
fn choice_memory() -> Arc<dyn GuestMemory> {
    Arc::new(FlatMemory::new(0x8000_0000, 0x1000))
}

/// [`graphics_opening`] for a live run over `memory`, with `render` and `opener`: whether it
/// built a GPU, and the backend it built it over. The GPU is stopped before returning.
fn opened(
    memory: Option<&Arc<dyn GuestMemory>>,
    render: &RenderSetup,
    opener: &WgpuOpener,
) -> Result<(bool, Option<Arc<dyn Backend>>), String> {
    let (graphics, built) = graphics_opening(
        &EnvSource::Live,
        memory,
        (None, Arc::new(NoWake), SystemClock::new()),
        render,
        opener,
    )?;
    let has_gpu = graphics.gpu().is_some();
    assert_eq!(built.is_some(), has_gpu);
    if let Some(gpu) = graphics.gpu() {
        gpu.stop().unwrap();
    }
    Ok((has_gpu, built.map(|(_, backend)| backend)))
}

/// An opener that fails as a host with no GPU adapter does, and counts its calls.
#[cfg(feature = "wgpu")]
fn no_adapter(calls: &Arc<std::sync::atomic::AtomicUsize>) -> WgpuOpener {
    let calls = Arc::clone(calls);
    WgpuOpener::with(Box::new(move |_, _, _, _| {
        calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Err(starquake_render_wgpu::NewError::NoAdapter(
            "none on the test's host".to_owned(),
        ))
    }))
}

/// The recording choice builds the recording backend and never asks for the wgpu backend.
#[cfg(feature = "wgpu")]
#[test]
fn the_recording_choice_never_opens_the_wgpu_backend() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let memory = choice_memory();
    let (gpu, backend) = opened(
        Some(&memory),
        &RenderSetup::recording(),
        &no_adapter(&calls),
    )
    .unwrap();
    assert!(gpu);
    assert_eq!(*backend.unwrap().caps(), recording_caps());
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

/// A wgpu backend that cannot be opened: the default choice falls back to the recording backend
/// and logs [`FELL_BACK`] with the reason at warning level; a choice that asked for it ends the
/// run with [`WGPU_UNAVAILABLE`] and the reason (module docs, "The wgpu backend").
#[cfg(feature = "wgpu")]
#[test]
fn an_unavailable_wgpu_backend_falls_back_or_ends_the_run_as_chosen() {
    use crate::host_log::HostLog;
    use crate::renderer::{Choice, IfUnavailable};

    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let memory = choice_memory();
    let opener = no_adapter(&calls);

    let fallback = RenderSetup::headless(Choice::Wgpu(IfUnavailable::UseRecording));
    let log = HostLog::new();
    let built = tracing::subscriber::with_default(log.clone(), || {
        opened(Some(&memory), &fallback, &opener)
    });
    let (gpu, backend) = built.unwrap();
    assert!(gpu, "the run still has its GPU");
    assert_eq!(
        *backend.unwrap().caps(),
        recording_caps(),
        "over the recording backend"
    );
    let (lines, dropped) = log.take();
    assert_eq!(dropped, 0);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].contains("WARN"), "{lines:?}");
    assert!(lines[0].contains(FELL_BACK), "{lines:?}");
    assert!(
        lines[0].contains("no GPU adapter on this host: none on the test's host"),
        "{lines:?}"
    );

    let asked = RenderSetup::headless(Choice::Wgpu(IfUnavailable::EndRun));
    let log = HostLog::new();
    let ended =
        tracing::subscriber::with_default(log.clone(), || opened(Some(&memory), &asked, &opener));
    assert_eq!(
        ended.err(),
        Some(
            "the wgpu backend could not be opened: no GPU adapter on this host: none on the \
             test's host"
                .to_owned()
        )
    );
    assert_eq!(
        log.take().0,
        Vec::<String>::new(),
        "nothing logged: the run ends"
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 2);
}

/// A run with no GPU (no shared memory, or a replay) has the null front whatever it chose, and
/// opens no backend.
#[cfg(feature = "wgpu")]
#[test]
fn a_run_with_no_gpu_opens_no_wgpu_backend() {
    use crate::renderer::{Choice, IfUnavailable};

    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let opener = no_adapter(&calls);
    let memory = choice_memory();
    for if_unavailable in [IfUnavailable::UseRecording, IfUnavailable::EndRun] {
        let render = RenderSetup::headless(Choice::Wgpu(if_unavailable));
        assert!(matches!(opened(None, &render, &opener), Ok((false, None))));
        let (graphics, built) = graphics_opening(
            &EnvSource::Replay(Vec::new()),
            Some(&memory),
            (None, Arc::new(NoWake), SystemClock::new()),
            &render,
            &opener,
        )
        .unwrap();
        assert!(graphics.gpu().is_none() && built.is_none());
    }
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

/// On a build without the wgpu backend, a setup that chose it anyway (`RenderSetup::headless`;
/// `choose_backend` never does) is told the build leaves it out.
#[cfg(not(feature = "wgpu"))]
#[test]
fn a_build_without_the_wgpu_backend_cannot_open_it() {
    use crate::renderer::{Choice, IfUnavailable};

    let memory = choice_memory();
    let asked = RenderSetup::headless(Choice::Wgpu(IfUnavailable::EndRun));
    assert_eq!(
        opened(Some(&memory), &asked, &WgpuOpener::host()).err(),
        Some(
            "the wgpu backend could not be opened: this build of starquake leaves the wgpu \
             backend out"
                .to_owned()
        )
    );
    let fallback = RenderSetup::headless(Choice::Wgpu(IfUnavailable::UseRecording));
    let (gpu, backend) = opened(Some(&memory), &fallback, &WgpuOpener::host()).unwrap();
    assert!(gpu);
    assert_eq!(*backend.unwrap().caps(), recording_caps());
}

/// The wiring end to end, through the production path (`System::run_rendering`, the hypervisor
/// backend and `Machine::start`): a run whose setup has a window its main thread will not make
/// ends with the window's words before the guest runs, and the same program with the recording
/// setup runs to its end. A Mac test with a GPU: it skips itself only when `CI` is set, or, with
/// a notice, where the host has no GPU adapter.
#[cfg(all(feature = "wgpu", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn a_run_renders_with_the_setup_it_is_given() {
    use crate::machine::gpu::wgpu_backend::WINDOW_FAILED;
    use crate::renderer::MainThreadGone;
    use crate::renderer::tests::{Gone, no_window};
    use crate::report::{ExitReason, RunOptions};

    if std::env::var_os("CI").is_some() {
        eprintln!("skipping a_run_renders_with_the_setup_it_is_given: it needs a VM and a GPU");
        return;
    }
    let system = crate::System::new(&starquake_testkit::programs::hello().unwrap()).unwrap();
    let options = RunOptions {
        timeout: Some(Duration::from_secs(30)),
        ..RunOptions::default()
    };
    let windowed = RenderSetup::windowed(no_window(64, 48, Arc::new(Gone)));
    let report = system.run_rendering(&options, windowed);
    let ExitReason::EmulatorError(message) = report.exit else {
        unreachable!("the run went on: {:?}", report.exit);
    };
    if message.starts_with(&format!("{WGPU_UNAVAILABLE}: no GPU adapter on this host")) {
        eprintln!("skipping a_run_renders_with_the_setup_it_is_given: {message}");
        return;
    }
    assert_eq!(message, format!("{WINDOW_FAILED}: {MainThreadGone}"));
    assert_eq!(report.guest_log.len(), 0, "the guest never ran");

    let recorded = system.run_rendering(&options, RenderSetup::recording());
    assert_eq!(recorded.exit, ExitReason::ProcessExited);
    assert_ne!(recorded.guest_log.len(), 0);
}

/// No program memory and no engine state: for lookups of no programs.
struct Nothing;

impl ProgramMemory for Nothing {
    fn read(
        &self,
        address: u64,
        _out: &mut [u8],
    ) -> Result<(), starquake_shader::lookup::Unmapped> {
        Err(starquake_shader::lookup::Unmapped { address })
    }
}

impl EngineState for Nothing {
    fn cbuf_word(
        &self,
        _stage: starquake_shader::lookup::GuestStage,
        _bank: u8,
        _offset: u32,
    ) -> Option<u32> {
        None
    }

    fn texture_kind(
        &self,
        _stage: starquake_shader::lookup::GuestStage,
        _handle: u32,
    ) -> Option<starquake_shader::exchange::TextureKind> {
        None
    }
}

/// With a pool, the run's shader cache is the shader crate's cache: a lookup of no programs is
/// a ready set of none. Without one it is [`NoShaders`], whose every lookup is pending.
#[test]
fn the_run_s_shader_cache_is_the_real_one_on_a_pool() {
    let pool = Arc::new(CompileWorkers::new(std::num::NonZeroUsize::MIN).unwrap());
    let mut real = run_shaders(Some(pool));
    assert!(matches!(
        real.lookup(&[], &VariantKey::default(), &Nothing, &Nothing),
        SetLookup::Ready(set) if set.stages.is_empty()
    ));
    let mut none = run_shaders(None);
    assert!(matches!(
        none.lookup(&[], &VariantKey::default(), &Nothing, &Nothing),
        SetLookup::Pending(ticket) if ticket == NO_TICKET
    ));
    assert_eq!(SHADER_TARGET.form, CodeForm::NagaModule);
    assert_eq!(SHADER_TARGET.caps, Requirements::default());
}

/// A compile pool that could not be started is the wgpu backend's failure
/// ([`wgpu_opening`]): the backend is not opened, a run that chose to fall back renders with
/// the recording backend and has no shader cache ([`NoShaders`]), and a run that chose to end
/// ends with the pool's words. A pool that started reaches the opener, whose backend failure
/// then follows the same choice.
#[test]
fn a_compile_pool_that_cannot_start_is_the_backend_s_failure() {
    use crate::renderer::IfUnavailable;

    let pool = started_pool(Err(std::io::Error::other("no threads left")));
    assert_eq!(
        pool.as_ref().err().map(String::as_str),
        Some("the compile pool could not be started: no threads left")
    );
    // Were the backend opened without a pool, the outcome would be `Opened`, which the
    // assertions below refuse.
    let never = |_: Arc<CompileWorkers>| -> Result<(), Unopened> { Ok(()) };
    assert_eq!(
        wgpu_opening(&pool, never, IfUnavailable::UseRecording),
        Ok(Opening::FallBack(
            "the compile pool could not be started: no threads left".to_owned()
        ))
    );
    let mut shaders = run_shaders(pool.clone().ok());
    assert!(matches!(
        shaders.lookup(&[], &VariantKey::default(), &Nothing, &Nothing),
        SetLookup::Pending(ticket) if ticket == NO_TICKET
    ));
    assert_eq!(
        wgpu_opening(&pool, never, IfUnavailable::EndRun),
        Err(
            "the wgpu backend could not be opened: the compile pool could not be started: no \
             threads left"
                .to_owned()
        )
    );

    let started = started_pool(CompileWorkers::new(std::num::NonZeroUsize::MIN));
    assert!(started.is_ok());
    let opens = |_: Arc<CompileWorkers>| -> Result<u8, Unopened> { Ok(7) };
    let refuses = |_: Arc<CompileWorkers>| -> Result<u8, Unopened> {
        Err(Unopened::Backend("no GPU adapter".to_owned()))
    };
    for if_unavailable in [IfUnavailable::UseRecording, IfUnavailable::EndRun] {
        assert_eq!(
            wgpu_opening(&started, opens, if_unavailable),
            Ok(Opening::Opened(7))
        );
    }
    assert_eq!(
        wgpu_opening(&started, refuses, IfUnavailable::UseRecording),
        Ok(Opening::FallBack("no GPU adapter".to_owned()))
    );
    assert_eq!(
        wgpu_opening(&started, refuses, IfUnavailable::EndRun),
        Err("the wgpu backend could not be opened: no GPU adapter".to_owned())
    );
}

/// A run that chose the wgpu backend hands the compile pool it gave the backend to its shader
/// cache, which keeps it when the backend cannot be opened and the run falls back to the
/// recording backend; the pool ends with the run (module docs, "The shader cache shares the
/// pool").
#[cfg(feature = "wgpu")]
#[test]
fn the_shader_cache_holds_the_pool_the_backend_was_given() {
    use std::sync::Weak;

    use starquake_render::CompilePool;

    use crate::renderer::{Choice, IfUnavailable};

    let given: Arc<Mutex<Option<Weak<dyn CompilePool>>>> = Arc::new(Mutex::new(None));
    let record = Arc::clone(&given);
    let opener = WgpuOpener::with(Box::new(move |_, _, pool, _| {
        *record.lock().unwrap() = Some(Arc::downgrade(&pool));
        Err(starquake_render_wgpu::NewError::NoAdapter(
            "none on the test's host".to_owned(),
        ))
    }));
    let memory = choice_memory();
    let render = RenderSetup::headless(Choice::Wgpu(IfUnavailable::UseRecording));
    let (graphics, built) = graphics_opening(
        &EnvSource::Live,
        Some(&memory),
        (None, Arc::new(NoWake), SystemClock::new()),
        &render,
        &opener,
    )
    .unwrap();
    let pool = given
        .lock()
        .unwrap()
        .clone()
        .expect("the opener was given a pool");
    assert!(
        pool.upgrade().is_some(),
        "the shader cache keeps the backend's pool after the fallback"
    );
    graphics.gpu().expect("the run has its GPU").stop().unwrap();
    drop(graphics);
    drop(built);
    assert!(pool.upgrade().is_none(), "the pool ends with the run");
}
