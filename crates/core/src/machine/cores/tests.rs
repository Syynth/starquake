//! The cores' parks, wakes and interrupts, and the stall question's retries at the GPU's quiet
//! signal (module docs, "Telling a stalled run"), on threads of the test's own: the quiet signal
//! is told by the test, as the GPU's front thread tells it in a run.

use std::sync::atomic::AtomicU32;
use std::time::{Duration, Instant};

use starquake_cpu::Error;

use super::*;

/// Counts its interrupts.
#[derive(Clone, Default)]
struct Counting(Arc<Mutex<u32>>);

impl Interrupter for Counting {
    fn interrupt(&self) -> Result<(), Error> {
        *lock(&self.0) += 1;
        Ok(())
    }
}

fn core(index: u8) -> CoreId {
    CoreId::new(index).unwrap()
}

/// How long a test waits for another thread before it fails.
const PATIENCE: Duration = Duration::from_secs(10);

/// Runs `body`, stopping `cores` if it has not finished within [`PATIENCE`], so that a park
/// that is never woken fails its test (it then ends [`Parked::Stopped`]) instead of hanging
/// the gate.
fn bounded<R>(cores: &Cores<Counting>, body: impl FnOnce() -> R) -> R {
    let (done, finished) = std::sync::mpsc::channel::<()>();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            if finished.recv_timeout(PATIENCE).is_err() {
                cores.stop();
            }
        });
        let result = body();
        assert!(
            done.send(()).is_ok(),
            "the watchdog fired: the test ran out of time"
        );
        result
    })
}

/// Waits until `done` holds; fails the test after [`PATIENCE`].
fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + PATIENCE;
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting until {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn idle(cores: &Cores<Counting>) -> usize {
    *lock(&cores.idle)
}

fn armed(cores: &Cores<Counting>) -> Option<usize> {
    *lock(&cores.parks.armed)
}

fn quiet_flag(cores: &Cores<Counting>, index: usize) -> bool {
    lock(&cores.parks.slots[index].state).quiet
}

/// A question that answers `answers` in turn, counting how often it was asked; past the end
/// it answers the last one again.
struct Answers {
    answers: Vec<Stall>,
    asked: AtomicU32,
}

impl Answers {
    fn new(answers: Vec<Stall>) -> Self {
        Self {
            answers,
            asked: AtomicU32::new(0),
        }
    }

    fn ask(&self) -> Stall {
        let asked = self.asked.fetch_add(1, Ordering::SeqCst);
        let at = usize::try_from(asked).unwrap();
        *self
            .answers
            .get(at)
            .unwrap_or_else(|| self.answers.last().unwrap())
    }

    fn asked(&self) -> u32 {
        self.asked.load(Ordering::SeqCst)
    }
}

#[test]
fn a_wake_before_the_park_is_not_lost() {
    let cores: Cores<Counting> = Cores::new(2);
    cores.wake(core(1));
    assert_eq!(
        bounded(&cores, || cores.park(core(1), || Stall::NotStalled)),
        Parked::Woken
    );
}

#[test]
fn a_park_waits_for_its_wake_from_another_thread() {
    let cores: Cores<Counting> = Cores::new(2);
    let parked = bounded(&cores, || {
        std::thread::scope(|scope| {
            let parked = scope.spawn(|| cores.park(core(0), || Stall::NotStalled));
            // The parked thread is woken whatever this sleep finds.
            std::thread::sleep(Duration::from_millis(20));
            cores.wake(core(0));
            parked.join().unwrap()
        })
    });
    assert_eq!(parked, Parked::Woken);
}

#[test]
fn interrupts_reach_only_an_attached_vcpu_and_stopping_reaches_every_core() {
    let cores: Cores<Counting> = Cores::new(3);
    let count = Counting::default();
    // Before its vCPU exists an interrupt reaches nothing.
    cores.interrupt(core(1));
    cores.attach(core(1), count.clone());
    cores.interrupt(core(1));
    assert_eq!(*lock(&count.0), 1);
    cores.stop();
    assert_eq!(*lock(&count.0), 2);
    // A stopped run parks no more.
    assert_eq!(cores.park(core(2), || Stall::NotStalled), Parked::Stopped);
}

#[test]
fn only_the_last_core_to_park_asks_whether_the_run_is_stalled() {
    let cores: Cores<Counting> = Cores::new(2);
    bounded(&cores, || {
        std::thread::scope(|scope| {
            let first = scope.spawn(|| cores.park(core(0), || unreachable!()));
            // The second core is the last: it asks, and is told the run is stalled.
            wait_until("the first core parks", || idle(&cores) == 1);
            let mut asked = false;
            let parked = cores.park(core(1), || {
                asked = true;
                Stall::Stalled
            });
            assert_eq!(parked, Parked::Stalled);
            assert!(asked);
            // Not stalled: the last core parks like any other, and is not armed.
            cores.wake(core(1));
            assert_eq!(cores.park(core(1), || Stall::NotStalled), Parked::Woken);
            assert_eq!(armed(&cores), None);
            cores.stop();
            assert_eq!(first.join().unwrap(), Parked::Stopped);
        });
    });
}

/// The core loop's question, every graphics answer with each kernel answer: a busy GPU alone
/// with a stalled kernel is `OnlyTheGpu`; the kernel is not asked when something besides the
/// GPU is busy.
#[test]
fn the_question_tells_a_busy_gpu_alone_apart() {
    assert_eq!(stall(Idleness::Idle, || true), Stall::Stalled);
    assert_eq!(stall(Idleness::Idle, || false), Stall::NotStalled);
    assert_eq!(stall(Idleness::GpuBusy, || true), Stall::OnlyTheGpu);
    assert_eq!(stall(Idleness::GpuBusy, || false), Stall::NotStalled);
    assert_eq!(
        stall(Idleness::Busy, || unreachable!("the kernel is not asked")),
        Stall::NotStalled
    );
}

/// Parks core 0 on a thread of `scope` (it never asks), and waits until it has.
fn park_the_other_core<'s>(
    scope: &'s std::thread::Scope<'s, '_>,
    cores: &'s Cores<Counting>,
) -> std::thread::ScopedJoinHandle<'s, Parked> {
    let other = scope.spawn(|| cores.park(core(0), || unreachable!()));
    wait_until("the other core parks", || idle(cores) == 1);
    other
}

/// The issue's case (gpu-protocol, "Idleness"): the last core's first question lands in a GPU
/// turn, so the answer is busy only because of the GPU; the core asks again at the GPU's next
/// quiet signal, and the run is then told stalled. Not before: with no signal the core waits.
#[test]
fn a_question_that_lands_in_a_gpu_turn_is_asked_again_at_the_next_quiet_signal() {
    let cores: Cores<Counting> = Cores::new(2);
    let answers = Answers::new(vec![Stall::OnlyTheGpu, Stall::Stalled]);
    let parked = bounded(&cores, || {
        std::thread::scope(|scope| {
            let other = park_the_other_core(scope, &cores);
            let last = scope.spawn(|| cores.park(core(1), || answers.ask()));
            wait_until("the last core asks and is armed", || {
                answers.asked() == 1 && armed(&cores) == Some(1)
            });
            // Asked once, and armed: only the quiet signal makes it ask again.
            assert_eq!(answers.asked(), 1);
            cores.parks.front_quiet();
            let parked = last.join().unwrap();
            assert_eq!(answers.asked(), 2);
            cores.stop();
            assert_eq!(other.join().unwrap(), Parked::Stopped);
            parked
        })
    });
    assert_eq!(parked, Parked::Stalled);
    assert_eq!(idle(&cores), 0);
}

/// Tells the quiet signal each time core 1 is armed, until `last` has returned.
fn signal_while_armed(cores: &Cores<Counting>, last: &std::thread::ScopedJoinHandle<'_, Parked>) {
    wait_until("the last core's park ends", || {
        if armed(cores) == Some(1) {
            cores.parks.front_quiet();
        }
        last.is_finished()
    });
}

/// The retry bound at its limit: the first question and the first `GPU_QUIET_RETRIES - 1`
/// retries are busy only because of the GPU, and the last retry, the `GPU_QUIET_RETRIES`th, is
/// still asked and ends the park stalled.
#[test]
fn the_last_retry_within_the_bound_is_asked() {
    let cores: Cores<Counting> = Cores::new(2);
    let busy = usize::try_from(GPU_QUIET_RETRIES).unwrap();
    let mut script = vec![Stall::OnlyTheGpu; busy];
    script.push(Stall::Stalled);
    let answers = Answers::new(script);
    let parked = bounded(&cores, || {
        std::thread::scope(|scope| {
            let other = park_the_other_core(scope, &cores);
            let last = scope.spawn(|| cores.park(core(1), || answers.ask()));
            signal_while_armed(&cores, &last);
            let parked = last.join().unwrap();
            cores.stop();
            assert_eq!(other.join().unwrap(), Parked::Stopped);
            parked
        })
    });
    assert_eq!(parked, Parked::Stalled);
    assert_eq!(answers.asked(), GPU_QUIET_RETRIES + 1);
}

/// One past the bound: every question is busy only because of the GPU. After the
/// `GPU_QUIET_RETRIES`th retry the core is no longer armed, so a further signal asks nothing,
/// and the core waits as for "not stalled" until it is woken.
#[test]
fn past_the_retry_bound_the_core_waits_as_before() {
    let cores: Cores<Counting> = Cores::new(2);
    let answers = Answers::new(vec![Stall::OnlyTheGpu]);
    let parked = bounded(&cores, || {
        std::thread::scope(|scope| {
            let other = park_the_other_core(scope, &cores);
            let last = scope.spawn(|| cores.park(core(1), || answers.ask()));
            let asks = GPU_QUIET_RETRIES + 1;
            wait_until("every question allowed is asked", || {
                if armed(&cores) == Some(1) {
                    cores.parks.front_quiet();
                }
                answers.asked() == asks
            });
            // The last question was asked unarmed: nothing is left to wait for.
            assert_eq!(armed(&cores), None);
            cores.parks.front_quiet();
            assert!(!quiet_flag(&cores, 1));
            assert!(!last.is_finished());
            cores.wake(core(1));
            let parked = last.join().unwrap();
            assert_eq!(answers.asked(), asks);
            cores.stop();
            assert_eq!(other.join().unwrap(), Parked::Stopped);
            parked
        })
    });
    assert_eq!(parked, Parked::Woken);
}

/// The core is armed before it asks: a quiet signal told while the question is being asked
/// (the GPU's turn ended meanwhile) still makes it ask again.
#[test]
fn a_quiet_signal_during_the_question_is_not_lost() {
    let cores: Cores<Counting> = Cores::new(2);
    let answers = Answers::new(vec![Stall::OnlyTheGpu, Stall::Stalled]);
    let parked = bounded(&cores, || {
        std::thread::scope(|scope| {
            let other = park_the_other_core(scope, &cores);
            let parked = cores.park(core(1), || {
                if answers.asked() == 0 {
                    cores.parks.front_quiet();
                }
                answers.ask()
            });
            cores.stop();
            assert_eq!(other.join().unwrap(), Parked::Stopped);
            parked
        })
    });
    assert_eq!(parked, Parked::Stalled);
    assert_eq!(answers.asked(), 2);
}

/// A quiet signal that comes once another core has been woken asks nothing: that core will
/// park again and ask itself. The signal is taken (the core is no longer armed), and the
/// unchanged case, every core still parked, is
/// `a_question_that_lands_in_a_gpu_turn_is_asked_again_at_the_next_quiet_signal`.
#[test]
fn a_quiet_signal_while_another_core_runs_asks_nothing() {
    let cores: Cores<Counting> = Cores::new(2);
    let answers = Answers::new(vec![Stall::OnlyTheGpu, Stall::Stalled]);
    bounded(&cores, || {
        std::thread::scope(|scope| {
            let other = park_the_other_core(scope, &cores);
            let last = scope.spawn(|| cores.park(core(1), || answers.ask()));
            wait_until("the last core is armed", || armed(&cores) == Some(1));
            cores.wake(core(0));
            assert_eq!(other.join().unwrap(), Parked::Woken);
            assert_eq!(idle(&cores), 1);
            cores.parks.front_quiet();
            wait_until("the last core takes the signal", || {
                !quiet_flag(&cores, 1)
            });
            assert_eq!(answers.asked(), 1);
            assert_eq!(armed(&cores), None);
            assert!(!last.is_finished());
            // The other core parks again as the last, asks, and is told stalled.
            let asked_again = cores.park(core(0), || Stall::Stalled);
            assert_eq!(asked_again, Parked::Stalled);
            cores.stop();
            assert_eq!(last.join().unwrap(), Parked::Stopped);
        });
    });
    assert_eq!(answers.asked(), 1);
}

/// A wake while the core is armed ends its park as usual and disarms it: a later signal is
/// told to no one, and the next park starts unarmed.
#[test]
fn a_wake_while_armed_ends_the_park_and_disarms() {
    let cores: Cores<Counting> = Cores::new(2);
    let answers = Answers::new(vec![Stall::OnlyTheGpu]);
    bounded(&cores, || {
        std::thread::scope(|scope| {
            let other = park_the_other_core(scope, &cores);
            let last = scope.spawn(|| cores.park(core(1), || answers.ask()));
            wait_until("the last core is armed", || armed(&cores) == Some(1));
            cores.wake(core(1));
            assert_eq!(last.join().unwrap(), Parked::Woken);
            assert_eq!(armed(&cores), None);
            cores.parks.front_quiet();
            assert!(!quiet_flag(&cores, 1));
            assert_eq!(answers.asked(), 1);
            cores.stop();
            assert_eq!(other.join().unwrap(), Parked::Stopped);
        });
    });
}

/// A later question by another core moves the arming to it: the signal reaches the core that
/// asked last.
#[test]
fn the_arming_moves_to_the_core_that_asked_last() {
    let cores: Cores<Counting> = Cores::new(2);
    let first = Answers::new(vec![Stall::OnlyTheGpu]);
    let second = Answers::new(vec![Stall::OnlyTheGpu, Stall::Stalled]);
    bounded(&cores, || {
        std::thread::scope(|scope| {
            let other = park_the_other_core(scope, &cores);
            let last = scope.spawn(|| cores.park(core(1), || first.ask()));
            wait_until("core 1 is armed", || armed(&cores) == Some(1));
            // Core 0 is woken and parks again: now it is the last, and it asks.
            cores.wake(core(0));
            assert_eq!(other.join().unwrap(), Parked::Woken);
            let again = scope.spawn(|| cores.park(core(0), || second.ask()));
            wait_until("core 0 is armed", || armed(&cores) == Some(0));
            cores.parks.front_quiet();
            assert_eq!(again.join().unwrap(), Parked::Stalled);
            assert_eq!((first.asked(), second.asked()), (1, 2));
            cores.stop();
            assert_eq!(last.join().unwrap(), Parked::Stopped);
        });
    });
}
