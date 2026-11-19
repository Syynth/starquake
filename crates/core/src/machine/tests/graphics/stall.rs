//! A stalled run with the GPU (`machine/cores.rs`, "Telling a stalled run"; gpu-protocol,
//! "Idleness"): every guest thread waits on nothing that can end, so the last core to park is
//! told the run is stalled and the run ends as [`WAITING`], also when a question lands in a GPU
//! turn. The cores run on threads of their own over the run's production wiring
//! ([`Machine::start`], which gives the GPU the cores as its quiet wake), with the GPU's real
//! threads on the host clock. A question that lands in a GPU turn is made by
//! `Cores::read_gpu_busy`: questions that come to "stalled" read busy only because of the GPU
//! until a given number of the GPU front thread's own quiet signals have reached an armed core,
//! as if the GPU stayed in a turn until each of them. Without the retry no core is ever armed,
//! so such a run never ends on its own: the three tests that use `Cores::read_gpu_busy` failed
//! that way with the retry removed, at the test's bound. `a_stalled_run_with_the_gpu_ends_waiting`
//! forces nothing and passes without the retry too, unless a question happens to land in one of
//! the GPU's own turns. `a_stalled_run_whose_gpu_answers_busy_arms_the_last_core_for_the_quiet_signal`
//! forces nothing either: it stops the GPU, so the GPU's real busy answer goes through the core
//! loop's production question. There is no timer thread: nothing here has a deadline.

use std::sync::atomic::Ordering;

use super::frames::stand_in_memory;
use super::*;
use crate::machine::core_loop;
use crate::machine::cores::GPU_QUIET_RETRIES;

/// How long a run here may take before the test stops it and fails.
const PATIENCE: Duration = Duration::from_secs(20);

/// What the test stops a run with when it has not ended on its own.
const STOPPED_BY_THE_TEST: &str = "the test stopped the run";

type TestMachine<'m> = Machine<'m, ScriptedVcpu, FixedClock>;

/// Stops the run when dropped, so a test whose check panics on the watching thread still ends
/// the cores instead of hanging the gate.
struct StopOnDrop<'a, 'm>(&'a TestMachine<'m>);

impl Drop for StopOnDrop<'_, '_> {
    fn drop(&mut self) {
        self.0.stop(Ending::Failed(STOPPED_BY_THE_TEST.to_owned()));
    }
}

/// Runs the hello program with the GPU, its main thread sleeping for ever on core 0, after
/// `prepare` and with `during` run on a thread of its own while the cores run, until the run
/// ends, or [`PATIENCE`] after `during` returned, when the test stops it with
/// [`STOPPED_BY_THE_TEST`]. Returns how it ended.
fn stalled_gpu_run(
    prepare: impl FnOnce(&TestMachine<'_>),
    during: impl FnOnce(&TestMachine<'_>) + Send,
) -> Ending {
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
    assert!(machine.graphics.gpu().is_some(), "the run has the GPU");
    prepare(&machine);
    // A sleep too long for any deadline: no timer, and no other thread.
    let mut main = ScriptedVcpu::new().then(
        |regs| regs.x[0] = i64::MAX.cast_unsigned(),
        svc(SvcId::SleepThread),
    );
    std::thread::scope(|scope| {
        let machine = &machine;
        for index in 1..machine.cores.count() {
            let core = CoreId::new(index).unwrap();
            scope.spawn(move || core_loop::run(machine, core, &mut ScriptedVcpu::default()));
        }
        scope.spawn(move || {
            let stop = StopOnDrop(machine);
            during(machine);
            let deadline = Instant::now() + PATIENCE;
            while !machine.stopped.load(Ordering::SeqCst) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            // A run that ended on its own keeps its ending: the first one wins.
            drop(stop);
        });
        // Core 0 runs on the test's thread, which made its vCPU, as a run's does.
        core_loop::run(machine, CoreId::new(CORE).unwrap(), &mut main);
    });
    let (ending, ..) = machine.finish();
    ending
}

fn failed_with(ending: &Ending) -> &str {
    match ending {
        Ending::Failed(message) => message,
        _ => unreachable!("the run did not end as an emulator error"),
    }
}

/// A stalled run with the GPU ends as `WAITING`, whatever the GPU's own turns overlap.
#[test]
fn a_stalled_run_with_the_gpu_ends_waiting() {
    let ending = stalled_gpu_run(|_| {}, |_| {});
    assert_eq!(failed_with(&ending), WAITING);
}

/// The issue's case: the first question that finds the run stalled lands in a GPU turn and
/// reads busy only because of the GPU, until the turn ends. The GPU's front thread tells the
/// quiet signal at the end of its next idle turn, the parked core asks again, and the run still
/// ends as `WAITING`, not at the test's bound.
#[test]
fn a_stalled_run_whose_first_question_lands_in_a_gpu_turn_still_ends_waiting() {
    let ending = stalled_gpu_run(|machine| machine.cores.read_gpu_busy(1), |_| {});
    assert_eq!(failed_with(&ending), WAITING);
}

/// The retry bound at its limit, through the run: the GPU reads busy until its
/// `GPU_QUIET_RETRIES`th quiet signal, so the first question and the first
/// `GPU_QUIET_RETRIES - 1` retries read busy, and the last retry is still asked.
#[test]
fn a_stalled_run_whose_questions_land_in_gpu_turns_up_to_the_bound_ends_waiting() {
    let ending = stalled_gpu_run(
        |machine| machine.cores.read_gpu_busy(GPU_QUIET_RETRIES),
        |_| {},
    );
    assert_eq!(failed_with(&ending), WAITING);
}

/// The production path from the GPU's answer to the stall question, with nothing forced: a
/// stopped GPU answers busy for good (`Gpu::is_idle`), so the graphics server's idleness is
/// `GpuBusy` ([`crate::hle::Graphics::idleness`]), the core loop's question comes to
/// `OnlyTheGpu` once the kernel says the run is stalled, and the last core to park stays armed
/// for a quiet signal a stopped GPU never tells. A question that only knew "idle or busy" would
/// read the stopped GPU as busy, come to "not stalled" and arm no core.
#[test]
fn a_stalled_run_whose_gpu_answers_busy_arms_the_last_core_for_the_quiet_signal() {
    let ending = stalled_gpu_run(
        |machine| machine.graphics.gpu().unwrap().stop().unwrap(),
        |machine| {
            let cores = usize::from(machine.cores.count());
            let deadline = Instant::now() + PATIENCE;
            while machine.cores.parked() != cores {
                assert!(Instant::now() < deadline, "the cores never all parked");
                std::thread::sleep(Duration::from_millis(1));
            }
            // Every core parked: the last one's question has been asked, and a stopped GPU
            // tells no quiet signal that could disarm it.
            assert!(
                machine.cores.armed().is_some(),
                "no core was armed for the GPU's quiet signal"
            );
            assert!(
                !machine.stopped.load(Ordering::SeqCst),
                "the run ended though the GPU answered busy"
            );
            machine.stop(Ending::Failed(STOPPED_BY_THE_TEST.to_owned()));
        },
    );
    assert_eq!(failed_with(&ending), STOPPED_BY_THE_TEST);
}

/// One past the bound: the GPU reads busy until one more quiet signal than a park waits for.
/// Once the park has taken its `GPU_QUIET_RETRIES` signals the core is no longer armed and
/// waits as before, with no timeout: no further signal reaches it, and the run does not end on
/// its own while several more safety-net turns of the GPU pass (`IDLE_PARK` each), so the test
/// stops it.
#[test]
fn past_the_bound_a_stalled_run_waits_for_a_wake() {
    let ending = stalled_gpu_run(
        |machine| machine.cores.read_gpu_busy(GPU_QUIET_RETRIES + 1),
        |machine| {
            let deadline = Instant::now() + PATIENCE;
            while machine.cores.busy_turns_left() != 1 {
                assert!(Instant::now() < deadline, "the retries were never asked");
                std::thread::sleep(Duration::from_millis(1));
            }
            std::thread::sleep(starquake_gpu::front::IDLE_PARK.saturating_mul(3));
            assert_eq!(machine.cores.busy_turns_left(), 1);
            assert!(
                !machine.stopped.load(Ordering::SeqCst),
                "the run ended though every question was read busy"
            );
            machine.stop(Ending::Failed(STOPPED_BY_THE_TEST.to_owned()));
        },
    );
    assert_eq!(failed_with(&ending), STOPPED_BY_THE_TEST);
}
