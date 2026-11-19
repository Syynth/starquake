//! Frames and the GPU in a real run (`machine/timer.rs`, "The GPU"; `crate::hle::graphics`,
//! "Frames"; `machine/gpu.rs`): a scripted guest presents a native image through gsv's vi2
//! session a given number of times and then sleeps, and a run with `until_frames` stops as soon
//! as the timer thread's turn has handed the front the last frame it allows, with either front.
//! A dead GPU thread ends the run; a run with the GPU reports its counters and flight recorder,
//! and keeps its graphics report and kept frames whatever its ending, a crash and a timeout
//! included.
//!
//! Requests are laid out by hand from PR's checks, as `crate::hle`'s present tests lay them out
//! (`eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97):
//! `CreateNativeMemory` (`0x7f, 3`: mode, size), `BindNativeMemory` (`0x82, 9`: id, address,
//! size, flags, offset), `CreateNativeImage` (`0x1bd, 18`: the preference at 2–17, the count at
//! 18, the memory at 19), `CreateNativeWindowAliasImagePool` (`0x14b, 2`: image, count),
//! `OpenDisplaySession` (`0x131, 2`: the flags), and the vi2 session's `OpenDisplay` (`0x75, 3`),
//! `RegisterClientState` (`0xd9, 4`), `BindClientDisplay` (`0x10b, 4`),
//! `GetClientAvailability` (`0xe4, 2`) and `PresentClientBuffer` (`0xf5, 24`).

use std::num::NonZeroU64;

use starquake_gsv::native::layout::{FORMAT_RGBA8, KIND_NATIVE, PREFERENCE_WORDS};
use starquake_gsv::root::DISPLAY_SESSION_FLAGS;
use starquake_gsv::vi2::command as vi2_command;
use starquake_gsv::vi2::{
    DEFAULT_DISPLAY, NO_FENCE, OPEN_DISPLAY_FLAGS, PRESENT_BLOCK_HEIGHT, PRESENT_LAYOUT,
    PRESENT_WORD_FENCE, PRESENT_WORD_LAYOUT,
};

use super::*;
// The image: 16 by 16 RGBA8 in two pages, as `crate::hle`'s present tests make it.
use crate::hle::tests::graphics::present::{EXTENT, IMAGE_SIZE};
use crate::image::PAGE;
use crate::machine::gpu::GPU_STOPPED;

/// The frame limit the runs here stop at.
const LIMIT: u32 = 3;

/// The client's token and its two notification bits (the test's own values).
const TOKEN: u64 = 0x0000_0055_0000_0066;
const RETIRE: u32 = 1;
const AVAILABLE: u32 = 2;

/// How long the presenting guest sleeps after its presents, before it exits: far longer than
/// the ticks that latch them (one each 16.7 ms).
const SETTLE: Duration = Duration::from_secs(2);

/// A program whose `.bss` holds the image, page-aligned: its code is never run.
fn image_program() -> Vec<u8> {
    let pages = IMAGE_SIZE / PAGE + 1;
    NroBuilder::new(&[a64::nop()])
        .bss_size(u32::try_from(pages * PAGE).unwrap())
        .build()
        .unwrap()
}

/// How the presenting guest ends, once it has slept [`SETTLE`] after its presents.
#[derive(Clone, Copy)]
enum End {
    /// It calls `ExitProcess`.
    Exit,
    /// It runs an undefined instruction at this address: a crash.
    Crash(u64),
}

/// The scripted presenting guest: gets the root, makes and binds a native image at `image_at`
/// in a window pool, opens a vi2 session and the default display, registers, binds and asks the
/// availability of its client, presents the image `count` times, one slot each from slot 0, then
/// sleeps [`SETTLE`] and ends as `end` says.
#[expect(
    clippy::too_many_lines,
    reason = "one step per request of the conversation, in order"
)]
fn presenting_guest(script: &Script, image_at: u64, count: u32, end: End) -> ScriptedVcpu {
    let send = || svc(SvcId::SendSyncRequest);
    let [size_low, size_high] = halves(IMAGE_SIZE);
    let [at_low, at_high] = halves(image_at);
    let [token_low, token_high] = halves(TOKEN);
    let [name_low, name_high] = halves(DEFAULT_DISPLAY);
    let mut guest = asking_for_the_root(script)
        .then(
            step(script, move |script, vars| {
                vars.gsv = script.cmif(SessionKind::Domain, 0).2[0];
                script.write_words(&[
                    tipc_type(root_command::CREATE_NATIVE_MEMORY),
                    3,
                    0,
                    size_low,
                    size_high,
                ]);
                vars.gsv
            }),
            send(),
        )
        .then(
            step(script, move |script, vars| {
                vars.native = script.tipc().0[0];
                script.write_words(&[
                    tipc_type(root_command::BIND_NATIVE_MEMORY),
                    9,
                    vars.native,
                    at_low,
                    at_high,
                    size_low,
                    size_high,
                    0,
                    0,
                    0,
                    0,
                ]);
                vars.gsv
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                script.tipc();
                let mut preference = [0; PREFERENCE_WORDS];
                preference[1] = KIND_NATIVE;
                preference[2] = EXTENT;
                preference[4] = PRESENT_BLOCK_HEIGHT;
                preference[5] = 1;
                preference[6] = FORMAT_RGBA8;
                preference[7] = 1;
                let mut words = vec![tipc_type(root_command::CREATE_NATIVE_IMAGE), 18];
                words.extend(preference);
                words.extend([1, vars.native]);
                script.write_words(&words);
                vars.gsv
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                vars.image = script.tipc().0[0];
                script.write_words(&[
                    tipc_type(root_command::CREATE_NATIVE_WINDOW_ALIAS_IMAGE_POOL),
                    2,
                    vars.image,
                    1,
                ]);
                vars.gsv
            }),
            send(),
        )
        .then(
            step(script, |script, vars| {
                script.tipc();
                let [low, high] = halves(DISPLAY_SESSION_FLAGS[0]);
                script.write_words(&[tipc_type(root_command::OPEN_DISPLAY_SESSION), 2, low, high]);
                vars.gsv
            }),
            send(),
        )
        .then(
            step(script, move |script, vars| {
                vars.vi2 = script.tipc().1[0];
                script.write_words(&[
                    tipc_type(vi2_command::OPEN_DISPLAY),
                    3,
                    name_low,
                    name_high,
                    OPEN_DISPLAY_FLAGS,
                ]);
                vars.vi2
            }),
            send(),
        )
        .then(
            step(script, move |script, vars| {
                script.tipc();
                script.write_words(&[
                    tipc_type(vi2_command::REGISTER_CLIENT_STATE),
                    4,
                    token_low,
                    token_high,
                    RETIRE,
                    AVAILABLE,
                ]);
                vars.vi2
            }),
            send(),
        )
        .then(
            step(script, move |script, vars| {
                script.tipc();
                script.write_words(&[
                    tipc_type(vi2_command::BIND_CLIENT_DISPLAY),
                    4,
                    token_low,
                    token_high,
                    name_low,
                    name_high,
                ]);
                vars.vi2
            }),
            send(),
        )
        .then(
            step(script, move |script, vars| {
                script.tipc();
                script.write_words(&[
                    tipc_type(vi2_command::GET_CLIENT_AVAILABILITY),
                    2,
                    token_low,
                    token_high,
                ]);
                vars.vi2
            }),
            send(),
        );
    for slot in 0..count {
        guest = guest.then(
            step(script, move |script, vars| {
                script.tipc();
                // 229: the token, the slot, the image, offset 0, the whole extent, the layout,
                // then zeros up to the fence, which is none.
                let mut words = vec![tipc_type(vi2_command::PRESENT_CLIENT_BUFFER), 24];
                words.extend([
                    token_low,
                    token_high,
                    slot,
                    vars.image,
                    0,
                    EXTENT,
                    PRESENT_LAYOUT,
                ]);
                words.extend(vec![0; PRESENT_WORD_FENCE - PRESENT_WORD_LAYOUT - 1]);
                words.extend([NO_FENCE, 0, 0, 0]);
                script.write_words(&words);
                vars.vi2
            }),
            send(),
        );
    }
    let done = script.clone();
    let slept = guest.then(
        step(script, |script, _| {
            script.tipc();
            u32::try_from(SETTLE.as_nanos()).unwrap_or(u32::MAX)
        }),
        svc(SvcId::SleepThread),
    );
    // `SleepThread` returns no result.
    let wake = move |_: &mut GuestRegs| done.vars.borrow_mut().waited = Some(0);
    match end {
        End::Exit => slept.then(wake, svc(SvcId::ExitProcess)),
        End::Crash(pc) => slept.then(
            move |regs| {
                wake(regs);
                regs.pc = pc;
            },
            Exit::Undefined {
                pc,
                syndrome: Syndrome::undefined(),
            },
        ),
    }
}

/// What a run of the presenting guest presenting `count` times came to, with `until_frames` at
/// [`LIMIT`]; with a GPU over `gpu_memory` when given.
fn presenting_run(count: u32, gpu_memory: Option<Arc<dyn GuestMemory>>) -> (RunReport, bool) {
    presenting_run_until(count, gpu_memory, RUN_LIMIT, false)
}

/// [`presenting_run`] with the run's timeout at `timeout`, and a guest that crashes instead of
/// exiting when `crash` is set.
fn presenting_run_until(
    count: u32,
    gpu_memory: Option<Arc<dyn GuestMemory>>,
    timeout: Duration,
    crash: bool,
) -> (RunReport, bool) {
    let guest = super::super::Guest::new(&image_program());
    let image_at = guest.start_of("main.bss").next_multiple_of(PAGE);
    let page = usize::try_from(PAGE).unwrap();
    let tls = Arc::new(FlatMemory::new(guest.tls, page));
    let space = WithTls {
        image: &guest.memory,
        tls: Arc::clone(&tls),
    };
    let vars = Rc::new(RefCell::new(Vars::default()));
    let script = Script {
        tls,
        base: guest.tls,
        vars: Rc::clone(&vars),
    };
    let end = if crash {
        End::Crash(guest.placed.entry)
    } else {
        End::Exit
    };
    let mut main = presenting_guest(&script, image_at, count, end);
    let options = RunOptions {
        timeout: Some(timeout),
        until_frames: NonZeroU64::new(u64::from(LIMIT)),
        ..RunOptions::default()
    };
    let mut program = guest.program(&space);
    program.gpu_memory = gpu_memory;
    let report = run(
        vcpus(&mut main),
        program,
        &options,
        Instant::now(),
        timebase(),
    );
    drop((script, main));
    let slept = vars.borrow().waited.is_some();
    (report, slept)
}

/// A stand-in for the guest memory a real run shares with the GPU: the scripted runs' flat
/// memory is not shared, and the GPU here reads nothing the guest wrote.
fn stand_in_memory() -> Arc<dyn GuestMemory> {
    Arc::new(FlatMemory::new(0x8000_0000, usize::try_from(PAGE).unwrap()))
}

/// `--until frames=N` (issue #386, "Tests"): the guest presents `N - 1`, `N` and `N + 1`
/// frames and then sleeps. `N - 1` frames run to the guest's exit with `N - 1` presented; `N`
/// and `N + 1` stop the run at exactly `N`, while the guest sleeps, as `frames_reached` with
/// exit code 0. The same with the null front and with the GPU, whose front was handed exactly
/// the frames counted.
#[test]
fn a_run_stops_at_exactly_its_frame_limit() {
    for gpu in [false, true] {
        for (count, frames, exit) in [
            (LIMIT - 1, LIMIT - 1, ExitReason::ProcessExited),
            (LIMIT, LIMIT, ExitReason::FramesReached),
            (LIMIT + 1, LIMIT, ExitReason::FramesReached),
        ] {
            let memory = gpu.then(stand_in_memory);
            let (report, slept) = presenting_run(count, memory);
            let what = format!("{count} presents, gpu {gpu}");
            assert_eq!(report.exit, exit, "{what}: {}", report.trace);
            assert_eq!(report.exit_code(), exit_code::STOP_CONDITION, "{what}");
            assert_eq!(report.graphics.frames, u64::from(frames), "{what}");
            assert_eq!(report.summary().frames, u64::from(frames), "{what}");
            // Only the run that was not stopped got to the guest's exit.
            assert_eq!(slept, exit == ExitReason::ProcessExited, "{what}");
            match report.graphics.gpu {
                Some(counters) => {
                    assert!(gpu, "{what}");
                    assert_eq!(counters.presents_pushed, u64::from(frames), "{what}");
                }
                None => assert!(!gpu, "{what}"),
            }
        }
    }
}

/// A run with the GPU that crashes, or times out, after its presents keeps its graphics report
/// (`Machine::finish`'s crashed and timed-out arms): the frames counted, and the GPU's counters,
/// which saw every frame the front was handed.
#[test]
fn a_crashed_or_timed_out_run_keeps_its_graphics_report() {
    let count = LIMIT - 1;
    for (timeout, crash) in [(RUN_LIMIT, true), (SETTLE / 2, false)] {
        let (report, slept) = presenting_run_until(count, Some(stand_in_memory()), timeout, crash);
        let what = if crash { "crashed" } else { "timed out" };
        if crash {
            assert!(report.crash().is_some(), "{what}: {}", report.trace);
            assert!(slept, "{what}");
        } else {
            assert_eq!(report.exit, ExitReason::Timeout, "{what}: {}", report.trace);
            assert!(!slept, "{what}");
        }
        assert_eq!(report.graphics.frames, u64::from(count), "{what}");
        assert_eq!(report.summary().frames, u64::from(count), "{what}");
        let counters = report.graphics.gpu.unwrap();
        assert_eq!(counters.presents_pushed, u64::from(count), "{what}");
        assert_eq!(report.summary().gpu, Some(counters), "{what}");
    }
}

/// Runs `prepare` on a machine with the GPU over [`stand_in_memory`], started on the hello
/// program with a real run's services and no thread of its own running, then finishes it: how it
/// ended and its aftermath.
fn finished_gpu_machine(
    prepare: impl FnOnce(&Machine<'_, ScriptedVcpu, FixedClock>),
) -> (Ending, Aftermath) {
    let guest = super::super::Guest::new(&programs::hello().unwrap());
    let space: &dyn GuestSpace<ScriptedVcpu> = &guest.memory;
    let options = RunOptions::default();
    let session = Session {
        memory: space,
        options: &options,
        clock: RunClock::new(Timebase {
            console: NRO_CONSOLE,
            clock: FixedClock::new(CLOCK_STEP, CLOCK_STEP),
        }),
        recorder: Arc::new(Recorder::new()),
        gpu_memory: Some(stand_in_memory()),
    };
    let env = session.environment();
    let (dispatcher, _) = boot(guest.address_space(), &guest.placed, space, &*env).unwrap();
    let Ok(machine) = Machine::start(
        dispatcher,
        space,
        &session,
        &env,
        Limits::DEFAULT,
        &real_services,
    ) else {
        unreachable!("the services could not be installed");
    };
    prepare(&machine);
    let (ending, _, _, aftermath) = machine.finish();
    (ending, aftermath)
}

/// Whatever the run's ending, `Machine::finish` hands the report the GPU's counters and the
/// frames its keeper holds (`machine/gpu.rs`, "At the end"): a frame fed to the run's keeper is
/// in the aftermath's dumps for every ending.
#[test]
fn every_ending_keeps_the_gpu_report_and_the_kept_frames() {
    use starquake_render::command::PresentTransform;
    use starquake_render::frame::RGBA8_BYTES;
    use starquake_render::{Frame, FramePixels, FrameSink};

    type Make = fn() -> Ending;
    let endings: [(Make, &str); 5] = [
        (|| Ending::Exited, "exited"),
        (|| Ending::FramesReached, "frames reached"),
        (
            || {
                Ending::Crashed(
                    crate::crash::Exception::UndefinedInstruction,
                    None,
                    Box::default(),
                )
            },
            "crashed",
        ),
        (|| Ending::TimedOut, "timed out"),
        (|| Ending::Failed("the test's".to_owned()), "failed"),
    ];
    for (ending, what) in endings {
        let (ended, aftermath) = finished_gpu_machine(|machine| {
            let keeper = machine.dumps.as_ref().unwrap();
            keeper.frame(Frame {
                frame: 7,
                width: 1,
                height: 1,
                transform: PresentTransform::Identity,
                pixels: FramePixels::Rgba8(vec![0x42; RGBA8_BYTES]),
            });
            machine
                .graphics
                .front()
                .barrier(starquake_gpu::interface::Ticket(9));
            machine.stop(ending());
        });
        // The GPU stopped cleanly, so the ending is the one the run stopped with.
        assert_eq!(
            std::mem::discriminant(&ended),
            std::mem::discriminant(&ending()),
            "{what}"
        );
        let graphics = aftermath.graphics;
        assert_eq!(
            graphics.gpu.map(|counters| counters.ordered),
            Some(1),
            "{what}"
        );
        assert!(
            graphics.gpu_recorder.contains("barrier, ticket 9"),
            "{what}"
        );
        let numbers: Vec<u64> = graphics.dumps.files.iter().map(|file| file.frame).collect();
        assert_eq!(numbers, [7], "{what}");
    }
}

/// A run whose GPU thread has ended stops at the timer thread's next turn, as an emulator error
/// with [`GPU_STOPPED`] (`machine/gpu.rs`, "During the run"). The timer thread runs on its own
/// thread; if it never stopped the run, the test stops it after a bound and fails.
#[test]
fn a_dead_gpu_thread_ends_the_run() {
    let guest = super::super::Guest::new(&programs::hello().unwrap());
    let space: &dyn GuestSpace<ScriptedVcpu> = &guest.memory;
    let options = RunOptions::default();
    let session = Session {
        memory: space,
        options: &options,
        clock: RunClock::new(Timebase {
            console: NRO_CONSOLE,
            clock: FixedClock::new(CLOCK_STEP, CLOCK_STEP),
        }),
        recorder: Arc::new(Recorder::new()),
        gpu_memory: Some(stand_in_memory()),
    };
    let env = session.environment();
    let (dispatcher, _) = boot(guest.address_space(), &guest.placed, space, &*env).unwrap();
    let Ok(machine) = Machine::start(
        dispatcher,
        space,
        &session,
        &env,
        Limits::DEFAULT,
        &real_services,
    ) else {
        unreachable!("the services could not be installed");
    };
    let gpu = Arc::clone(machine.graphics.gpu().unwrap());
    assert!(gpu.is_running());
    // A running GPU does not end the run: one turn of the timer thread's check.
    assert!(crate::machine::gpu::check_running(&machine.graphics).is_ok());
    gpu.stop().unwrap();
    std::thread::scope(|scope| {
        let machine = &machine;
        scope.spawn(move || timer::run(machine));
        let deadline = Instant::now() + Duration::from_secs(20);
        while !machine.stopped.load(std::sync::atomic::Ordering::SeqCst) {
            if Instant::now() >= deadline {
                machine.stop(Ending::Failed("the test stopped the run".to_owned()));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    });
    let (ending, ..) = machine.finish();
    match ending {
        Ending::Failed(message) => assert_eq!(message, GPU_STOPPED),
        _ => unreachable!("the run did not fail"),
    }
}

/// The waiting guest (`graphics.rs`) on a run with the GPU, built by the run itself
/// (`machine/gpu.rs`): the GPU completes the ordered update the guest waits on, and the run's
/// report carries the GPU's counters and flight recorder (`gpu.txt`), which name the address
/// space the guest made, the ordered update and the barrier.
#[test]
fn a_run_with_the_gpu_reports_its_counters_and_flight_recorder() {
    let guest = super::super::Guest::new(&programs::hello().unwrap());
    let page = usize::try_from(PAGE).unwrap();
    let tls = Arc::new(FlatMemory::new(guest.tls, page));
    let space = WithTls {
        image: &guest.memory,
        tls: Arc::clone(&tls),
    };
    let vars = Rc::new(RefCell::new(Vars::default()));
    let script = Script {
        tls,
        base: guest.tls,
        vars: Rc::clone(&vars),
    };
    let mut main = waiting_guest(&script);
    let options = RunOptions {
        timeout: Some(RUN_LIMIT),
        ..RunOptions::default()
    };
    let mut program = guest.program(&space);
    program.gpu_memory = Some(stand_in_memory());
    let report = run(
        vcpus(&mut main),
        program,
        &options,
        Instant::now(),
        timebase(),
    );
    drop((script, main));
    assert_eq!(report.exit, ExitReason::ProcessExited, "{}", report.trace);
    assert_eq!(vars.borrow().waited, Some(0));
    let counters = report.graphics.gpu.unwrap();
    // gsv sends the destroy with no ticket and then a barrier with the job's: two ordered.
    assert_eq!(counters.ordered, 2);
    let recorder = &report.graphics.gpu_recorder;
    assert!(
        recorder.starts_with("0 address space 1 made\n"),
        "{recorder}"
    );
    assert!(recorder.contains("sparse region destroy"), "{recorder}");
    assert!(recorder.contains("barrier, ticket"), "{recorder}");
    assert_eq!(report.summary().gpu, Some(counters));
    // Nothing was presented, so nothing was dumped.
    assert_eq!(report.graphics.frames, 0);
    assert_eq!(
        report.graphics.dumps,
        crate::frames::DumpedFrames::default()
    );
}
