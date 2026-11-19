//! A stalled run with the GPU (`machine/cores.rs`, "Telling a stalled run"; gpu-protocol,
//! "Idleness"): every guest thread waits on nothing that can end, so the last core to park is
//! told the run is stalled and the run ends as [`WAITING`], also when a question lands in a GPU
//! turn. The cores run on threads of their own over the run's production wiring
//! ([`Machine::start`], which gives the GPU the cores as its quiet wake), with the GPU's real
//! threads on the host clock; a question that lands in a GPU turn is made by
//! `Cores::read_gpu_busy`, which reads the next stalled answers as busy only because of the
//! GPU, and the GPU's own front thread then tells the quiet signal that has them asked again.
//! There is no timer thread: nothing here has a deadline.

use std::sync::atomic::Ordering;

use super::frames::stand_in_memory;
use super::*;
use crate::machine::cores::GPU_QUIET_RETRIES;
use crate::machine::core_loop;

/// How long a run here may take before the test stops it and fails.
const PATIENCE: Duration = Duration::from_secs(20);

/// What the test stops a run with when it has not ended on its own.
const STOPPED_BY_THE_TEST: &str = "the test stopped the run";

/// Runs the hello program with the GPU, its main thread sleeping for ever on core 0, after
/// `prepare` and with `during` run on a thread of its own while the cores run, until the run
/// ends, or [`PATIENCE`] after `during` returned, when the test stops it with
/// [`STOPPED_BY_THE_TEST`]. Returns how it ended.
fn stalled_gpu_run(
    prepare: impl FnOnce(&Machine<'_, ScriptedVcpu, FixedClock>),
    during: impl FnOnce(&Machine<'_, ScriptedVcpu, FixedClock>) + Send,
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
            during(machine);
            let deadline = Instant::now() + PATIENCE;
            while !machine.stopped.load(Ordering::SeqCst) {
                if Instant::now() >= deadline {
                    machine.stop(Ending::Failed(STOPPED_BY_THE_TEST.to_owned()));
                }
                std::thread::sleep(Duration::from_millis(1));
            }
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

/// The issue's case: the first question that finds the run stalled lands in a GPU turn and is
/// read busy only because of the GPU. The GPU's front thread tells the quiet signal at its next
/// idle turn, the parked core asks again, and the run still ends as `WAITING`, not at the
/// test's bound.
#[test]
fn a_stalled_run_whose_first_question_lands_in_a_gpu_turn_still_ends_waiting() {
    let ending = stalled_gpu_run(|machine| machine.cores.read_gpu_busy(1), |_| {});
    assert_eq!(failed_with(&ending), WAITING);
}

/// The retry bound at its limit, through the run: the first question and the first
/// `GPU_QUIET_RETRIES - 1` retries are read busy, and the last retry is still asked.
#[test]
fn a_stalled_run_whose_questions_land_in_gpu_turns_up_to_the_bound_ends_waiting() {
    let ending = stalled_gpu_run(
        |machine| machine.cores.read_gpu_busy(GPU_QUIET_RETRIES),
        |_| {},
    );
    assert_eq!(failed_with(&ending), WAITING);
}

/// One past the bound: every question the park may ask is read busy, and the core then waits
/// as before, with no timeout: the run does not end on its own while several more safety-net
/// turns of the GPU pass (`IDLE_PARK` each), and the test stops it.
#[test]
fn past_the_bound_a_stalled_run_waits_for_a_wake() {
    let ending = stalled_gpu_run(
        |machine| machine.cores.read_gpu_busy(GPU_QUIET_RETRIES + 1),
        |machine| {
            let deadline = Instant::now() + PATIENCE;
            while machine.cores.gpu_busy_reads() != 0 {
                assert!(Instant::now() < deadline, "the questions were never asked");
                std::thread::sleep(Duration::from_millis(1));
            }
            std::thread::sleep(starquake_gpu::front::IDLE_PARK.saturating_mul(3));
            assert!(
                !machine.stopped.load(Ordering::SeqCst),
                "the run ended though every question was read busy"
            );
            machine.stop(Ending::Failed(STOPPED_BY_THE_TEST.to_owned()));
        },
    );
    assert_eq!(failed_with(&ending), STOPPED_BY_THE_TEST);
}
