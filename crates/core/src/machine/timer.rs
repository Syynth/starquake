//! The timer thread (execution model, "Timers"): the one host thread that ends the kernel's
//! timeouts (sleeps and wait timeouts, through the scheduler like any other waker), preempts
//! the cores every [`PREEMPTION_INTERVAL_NS`], delivers the deferred replies a run's services
//! say are ready, and ends the run at its timeout.
//!
//! When to look is decided on the loop's copy of the run's host clock (`RunClock`), which is not
//! recorded; whether a timeout has expired is decided by the kernel against one reading of the
//! environment's clock ([`Dispatcher::fire_timers`]), and only when the copy says the earliest
//! deadline has come, so a run with no timeouts reads the environment's clock no more than its
//! system calls do. A timer entry carries the wait it was made for, so one that comes late (its
//! wait ended some other way) is ignored by the kernel. The run's own timeout
//! ([`RunOptions::timeout`](crate::report::RunOptions)) is wall-clock time from when the guest
//! starts, as it always was.
//!
//! The thread sleeps until the earliest of the next kernel deadline, the next preemption and
//! the run's timeout, or until it is rung ([`Bell`]): by a core whose call began a wait (which
//! may have a deadline earlier than the one it sleeps for), by a [`Ready`](super::Ready), and by
//! the run stopping. Each turn also drains the dispatcher's report of woken threads
//! ([`Dispatcher::take_woken`]), which the scheduler has already made runnable.
//!
//! Being rung only shortens a sleep: a deadline a sleeping timer thread has not seen is found
//! at its next turn anyway, at the latest one [`PREEMPTION_INTERVAL_NS`] later. (So no test
//! can tell a missing ring from a late timeout without timing the host.)
//!
//! A delivery ([`Dispatcher::deliver_replies`]) runs two kinds of handler call here, on a thread
//! with no vCPU: the server's `take_reply`, for the replies that are ready, and its `handle`,
//! for the requests that were queued behind the deferred ones and may start once those are
//! answered (and for requests parked while the delivery had the handler out). Both are safe
//! for the address space because none of the calls `IpcContext` offers today changes a stage-1
//! mapping: they read and write guest memory and handle tables, and make objects (a shared
//! memory's backing included), which the guest maps with its own system calls, after which its
//! core settles the address space as after any call. A context call that maps would need a
//! settle here.
//!
//! A delivery that finds the server's handler out of its slot (a core is running a request
//! through it) is not dropped: the kernel keeps it owed, and the core delivers it when it puts
//! the handler back, before its own call returns.
//!
//! # A process a handler ends during a delivery
//!
//! A handler may ask to end its request's sender's process (`IpcContext::end_process`, for
//! `ISelfController::Exit`) while this thread delivers: as it gives a deferred reply, or in a
//! request the delivery starts once the deferred one freed the server. The kernel writes and
//! records every reply first, then ends the process through the teardown `ExitProcess` uses,
//! and returns it ([`Dispatcher::deliver_replies`], `starquake_kernel::dispatch::Ended`). The
//! turn then records the process's end for the sender whose request asked, on its core, as a
//! core records it for a thread whose call ended in `Outcome::ProcessExited`, and the run
//! ends as it ends for `ExitProcess` ([`Ending::Exited`]). The run has one process
//! (PROVISIONAL, as in the kernel), so any process a delivery ends is the run's. Tests:
//! `a_handler_ending_the_process_during_the_timers_delivery_ends_the_run_after_the_reply`
//! (stepped), `a_delivery_records_the_process_end_on_the_core_of_the_sender_that_asked`
//! (stepped, with the sender on core 1, not this thread's [`CORE`]), and
//! `the_timer_thread_ends_the_run_when_a_delivery_ends_the_process` (the real timer thread).
//!
//! # The graphics server's turn
//!
//! Each turn, after the deliveries, runs the graphics server's turn
//! (`crate::hle::Graphics::turn`): it drains the graphics server's completion queue, answers
//! its shared syncpoint page and ticks its vsync clock, through a [`KernelHost`] over the run's
//! dispatcher, on core [`CORE`]
//! and no guest thread, with the time read from the environment's clock (the spec: "The turn's
//! time is the timer thread's, from the `Env` clock"). The cores are then rescheduled, since the
//! drain's signals may have woken threads. While the graphics server has no work
//! (`crate::hle::Graphics::wants_turn`: nothing queued, no request held by the page, no page
//! signal waiting, and the vsync clock stopped or its deadline not yet come by the loop's copy
//! of the clock) the turn is skipped, the environment's clock not read: it is read only when
//! there is work, as for the kernel's timeouts and the deliveries, so an idle server, or a vsync
//! clock between ticks, adds no reading that falls among the cores' by host timing
//! (`an_idle_graphics_turn_reads_nothing_and_a_busy_one_reads_the_clock`,
//! `a_waiting_vsync_clock_reads_nothing_and_a_due_one_ticks`). The queue rings the timer thread's [`Bell`] when
//! something is queued onto it ([`Wake`]), and the thread sleeps no later than the graphics
//! server's deadline (`crate::hle::Graphics::deadline`). An environment that fails during the
//! turn ends the run (`crate::hle::graphics`, "The timer thread's turn"), with the words
//! [`GRAPHICS_ENV_FAILED`] and the failure's.
//!
//! A turn that hands the GPU front the last frame the run allows
//! ([`RunOptions::until_frames`](crate::report::RunOptions), `crate::hle::Graphics::frame_limit_reached`)
//! ends the run as `Ending::FramesReached`, before the reschedule
//! (`a_run_stops_at_exactly_its_frame_limit`, in `machine/tests/graphics/frames.rs`).
//!
//! A listener starts the vsync clock with its deadline one interval (16,666,666 ns) out, more
//! than [`PREEMPTION_INTERVAL_NS`], so a sleep that began before the start ends before the
//! deadline and the next one is bounded by it; no ring is needed for a start
//! (`a_waiting_vsync_clock_reads_nothing_and_a_due_one_ticks` checks the two intervals). A tick made by
//! this thread reaches a waiting guest
//! (`the_timer_thread_ticks_the_vsync_clock_and_wakes_a_listener_s_waiter`).
//!
//! # The GPU
//!
//! Each turn, after the graphics server's, asks whether the run's GPU threads still run
//! (`machine/gpu.rs`, "During the run"), and ends the run as an emulator error if one has ended
//! (`a_dead_gpu_thread_ends_the_run`, in `machine/tests/graphics/frames.rs`). A run with the null front
//! has no GPU threads and asks nothing.
//!
//! # Sleeping to a deadline
//!
//! A host may end a timed wait well after its deadline on purpose (macOS coalesces timers, by a
//! leeway that grows with the wait's length: `leeway`'s module docs have the measurements), and
//! then every deadline the thread sleeps for comes late by it: preemption, the kernel's
//! timeouts, the syncpoint page's bound and the vsync ticks. So [`sleep_for`] does not ask for
//! the whole sleep at once: it waits in shares of the time left, each short enough to end before
//! the deadline even as late as the host has been ending the thread's waits ([`Leeway`]),
//! measuring each wait that times out from its own start (a ring or a spurious wake-up is not
//! measured), until the time is up or the bell rings. Every wait that ends at most
//! `leeway::ON_TIME` late lowers the estimate, so after one late outlier a host that ends waits
//! on time brings it back to zero, and there a sleep is one wait. A sleep can end up to
//! `ON_TIME` late, or as late as its last wait of at most `leeway::LAST_WAIT` made it. The
//! rule is the same on every host; on one where every wait ends at most `ON_TIME` late it
//! changes nothing, since the estimate never leaves zero.
//!
//! Tests: against model hosts, `against_a_host_that_ends_waits_late_a_sleep_learns_to_end_on_time`,
//! `against_an_on_time_host_a_sleep_is_one_wait`,
//! `after_one_late_outlier_an_on_time_host_brings_a_sleep_back_to_one_wait`,
//! `from_the_largest_estimate_the_wake_rate_falls_back`, `a_spurious_or_rung_wake_is_not_measured`,
//! `each_wait_is_measured_from_its_own_start` and `a_limit_past_the_end_of_time_is_one_wait`; on
//! the real bell, `a_sleep_lasts_its_limit_and_a_ring_ends_it` and
//! `from_the_largest_estimate_the_real_bell_falls_back_to_a_few_waits`; and the real runs whose
//! deadlines are timed, `a_held_page_request_is_answered_at_the_bound_by_the_timer_thread` and
//! `the_timer_thread_ticks_the_vsync_clock_and_wakes_a_listener_s_waiter`.

use std::sync::{Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use starquake_cpu::Vcpu;
use starquake_gsv::completion::Wake;
use starquake_kernel::dispatch::{Caller, Dispatcher};
use starquake_kernel::sched::PREEMPTION_INTERVAL_NS;
use starquake_trace::{Env, EventKind, HostClock, HostNanos, LifecyclePhase, ThreadId};

use super::{CORE, Ending, LogSink, Machine, PROCESS_ID, lock};
use crate::hle::graphics::KernelHost;
use leeway::Leeway;

mod leeway;

/// What the timer thread sleeps on: rung when it should look again.
#[derive(Default)]
pub(super) struct Bell {
    rung: Mutex<bool>,
    ring: Condvar,
}

impl Bell {
    /// Wakes the timer thread, or keeps its next sleep from starting.
    pub(super) fn ring(&self) {
        *lock(&self.rung) = true;
        self.ring.notify_all();
    }

    /// Whether the bell has been rung since this was last asked, or since its timer thread last
    /// woke; asking clears it.
    #[cfg(test)]
    pub(super) fn take_ring(&self) -> bool {
        std::mem::take(&mut *lock(&self.rung))
    }
}

/// How one of a sleep's waits ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Woke {
    /// The bell was rung (before the wait or during it); the ring is used up.
    Rung,
    /// The time asked for passed.
    TimedOut,
    /// Neither: a spurious wake-up.
    Early,
}

/// What [`sleep_for`] waits on: the [`Bell`], or a test's model of a host.
trait Waiter {
    /// The host's monotonic clock now.
    fn now(&self) -> Instant;

    /// Waits for at most `asked`, or until rung.
    fn wait(&self, asked: Duration) -> Woke;
}

impl Waiter for Bell {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn wait(&self, asked: Duration) -> Woke {
        let mut rung = lock(&self.rung);
        if !*rung {
            let (guard, result) = self
                .ring
                .wait_timeout(rung, asked)
                .unwrap_or_else(PoisonError::into_inner);
            rung = guard;
            if !*rung {
                return if result.timed_out() {
                    Woke::TimedOut
                } else {
                    Woke::Early
                };
            }
        }
        *rung = false;
        Woke::Rung
    }
}

/// Sleeps on `waiter` for `limit`, or until rung (module docs, "Sleeping to a deadline"): in
/// waits sized by `leeway`, each wait that times out measured from its own start into `leeway`.
/// A ring or a spurious wake-up is not measured. A limit past the end of `Instant` is waited for
/// in one go, ended by a timeout, a ring or a spurious wake-up alike.
fn sleep_for(waiter: &impl Waiter, limit: Duration, leeway: &mut Leeway) {
    let end = waiter.now().checked_add(limit);
    loop {
        let now = waiter.now();
        let left = end.map_or(limit, |end| end.saturating_duration_since(now));
        if left.is_zero() {
            return;
        }
        let asked = leeway.slice(left);
        match waiter.wait(asked) {
            Woke::Rung => return,
            Woke::TimedOut => {
                let slept = waiter.now().saturating_duration_since(now);
                *leeway = leeway.measured(asked, slept);
            }
            Woke::Early => {}
        }
        if end.is_none() {
            return;
        }
    }
}

impl Wake for Bell {
    fn completion_queued(&self) {
        self.ring();
    }
}

/// How a run ends when the environment fails during the graphics server's turn (module docs),
/// before the failure's own words.
pub(super) const GRAPHICS_ENV_FAILED: &str =
    "the environment failed during the graphics server's turn";

/// The event stream's name for no guest thread: deliveries the timer thread makes are recorded
/// for the senders they answer, never for this.
const NO_THREAD: ThreadId = ThreadId(0);

/// How long from `now` until `then`, or nothing if it has come.
fn until(now: HostNanos, then: HostNanos) -> Duration {
    Duration::from_nanos(then.0.saturating_sub(now.0))
}

/// Runs the timer thread until the run stops.
pub(super) fn run<V: Vcpu, C: HostClock + Clone>(machine: &Machine<'_, V, C>) {
    if let Err(ending) = turns(machine) {
        machine.stop(ending);
    }
}

fn turns<V: Vcpu, C: HostClock + Clone>(machine: &Machine<'_, V, C>) -> Result<(), Ending> {
    let clock = &machine.session.clock;
    let end = machine
        .session
        .options
        .timeout
        .and_then(|timeout| Instant::now().checked_add(timeout));
    let mut next_preemption = HostNanos(clock.host_now().0.saturating_add(PREEMPTION_INTERVAL_NS));
    let scheduler_failed = |error| Ending::Failed(format!("the scheduler failed: {error}"));
    let mut leeway = Leeway::default();
    while !machine.is_stopped() {
        if end.is_some_and(|end| Instant::now() >= end) {
            return Err(Ending::TimedOut);
        }
        let now = clock.host_now();
        if next_deadline(&machine.dispatcher).is_some_and(|deadline| deadline <= now) {
            machine
                .dispatcher
                .fire_timers(&**machine.env, &machine.session.recorder)
                .map_err(|error| Ending::Failed(format!("a timeout could not end: {error}")))?;
            machine.env.keep();
            machine
                .dispatcher
                .reschedule(&mut machine.signals())
                .map_err(scheduler_failed)?;
        }
        if now >= next_preemption {
            machine
                .dispatcher
                .preempt(&mut machine.signals())
                .map_err(scheduler_failed)?;
            next_preemption = HostNanos(now.0.saturating_add(PREEMPTION_INTERVAL_NS));
        }
        deliver(machine)?;
        graphics(machine)?;
        super::gpu::check_running(&machine.graphics)?;
        let woken = machine.dispatcher.take_woken();
        if !woken.is_empty() {
            tracing::trace!(count = woken.len(), "guest threads woken");
        }

        let now = clock.host_now();
        let mut sleep = until(now, next_preemption);
        if let Some(deadline) = next_deadline(&machine.dispatcher) {
            sleep = sleep.min(until(now, deadline));
        }
        if let Some(deadline) = machine.graphics.deadline() {
            sleep = sleep.min(until(now, deadline));
        }
        if let Some(end) = end {
            sleep = sleep.min(end.saturating_duration_since(Instant::now()));
        }
        sleep_for(&*machine.bell, sleep, &mut leeway);
    }
    Ok(())
}

/// Delivers the deferred replies of every server a [`Ready`](super::Ready) named since the last
/// turn ([`Dispatcher::deliver_replies`]), then tells the cores whose thread changed. A
/// delivery that ended the process ends the run instead ([`Ending::Exited`]; module docs, "A
/// process a handler ends during a delivery").
pub(super) fn deliver<V: Vcpu, C: HostClock + Clone>(
    machine: &Machine<'_, V, C>,
) -> Result<(), Ending> {
    let servers = machine.ready.take();
    if servers.is_empty() {
        return Ok(());
    }
    let mut debug = LogSink {
        session: machine.session,
        log: &machine.log,
        core: CORE,
        thread: NO_THREAD,
    };
    let mut ended = None;
    for server in servers {
        // The caller's memory, clock and recorder are what a delivery uses; it names no core
        // or thread of its own (`Dispatcher::deliver_replies`).
        let mut caller = Caller {
            core: CORE,
            thread: NO_THREAD,
            env: &**machine.env,
            memory: machine.session.memory,
            debug: &mut debug,
            recorder: &machine.session.recorder,
        };
        let delivered = machine
            .dispatcher
            .deliver_replies(server, &mut caller)
            .map_err(|error| Ending::Failed(format!("a reply could not be delivered: {error}")))?;
        if let Some(end) = delivered.first() {
            ended = Some(*end);
            break;
        }
    }
    // The run keeps the environment's recording; the workers' recordings are dropped (`machine`
    // module docs, "The environment"). Every read `fsp-srv`'s workers finish rings this turn, so
    // what they read is dropped here.
    machine.env.keep();
    for reads in &machine.reads {
        drop(reads.take_recording());
    }
    if let Some(end) = ended {
        machine.session.record_on(
            end.core,
            end.thread,
            EventKind::Lifecycle {
                phase: LifecyclePhase::ProcessExit,
                subject: PROCESS_ID,
            },
        );
        return Err(Ending::Exited);
    }
    machine
        .dispatcher
        .reschedule(&mut machine.signals())
        .map_err(|error| Ending::Failed(format!("the scheduler failed: {error}")))?;
    Ok(())
}

/// The graphics server's turn (module docs, "The graphics server's turn"), then a reschedule;
/// nothing at all, the environment's clock not read, while the graphics server has no work by
/// the loop's copy of the clock.
pub(super) fn graphics<V: Vcpu, C: HostClock + Clone>(
    machine: &Machine<'_, V, C>,
) -> Result<(), Ending> {
    if !machine
        .graphics
        .wants_turn(machine.session.clock.host_now())
    {
        return Ok(());
    }
    let failed = |error| Ending::Failed(format!("{GRAPHICS_ENV_FAILED}: {error}"));
    let env = &**machine.env;
    let now = env.now().map_err(failed)?;
    let mut host = KernelHost::new(
        &machine.dispatcher,
        env,
        &machine.session.recorder,
        machine.session.memory,
        (CORE, NO_THREAD),
    );
    machine.graphics.turn(&mut host, now);
    let turned = host.finish();
    machine.env.keep();
    turned.map_err(failed)?;
    if machine.graphics.frame_limit_reached() {
        return Err(Ending::FramesReached);
    }
    machine
        .dispatcher
        .reschedule(&mut machine.signals())
        .map_err(|error| Ending::Failed(format!("the scheduler failed: {error}")))?;
    Ok(())
}

/// The kernel's next deadline: the earliest entry in its timer queue.
fn next_deadline(dispatcher: &Dispatcher) -> Option<HostNanos> {
    dispatcher.with::<(&starquake_kernel::tables::Scheduler,), _>(|(scheduler,)| {
        scheduler.sync().next_deadline()
    })
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::thread;

    use starquake_kernel::sched::PREEMPTION_INTERVAL_NS;

    use super::leeway::{LAST_WAIT, MOST, ON_TIME, PER_MILLE};
    use super::*;

    /// Longer than any test here may take: a sleep that a ring ends must end well before it.
    const LONG: Duration = Duration::from_secs(10);

    /// The longest sleep the timer thread makes.
    const LONGEST: Duration = Duration::from_nanos(PREEMPTION_INTERVAL_NS);

    /// More waits than any sleep here may take: a model host fails the test past this many.
    const MOST_WAITS: usize = 10_000;

    /// How a model host ends a wait: given its number in the host's life and the time asked, how
    /// long it took and how it ended.
    type Ending<'a> = Box<dyn FnMut(usize, Duration) -> (Duration, Woke) + 'a>;

    /// A model host for [`sleep_for`]: a clock that moves only by the waits, each ended as
    /// `ending` says; it keeps what each wait asked for.
    struct Host<'a> {
        start: Instant,
        elapsed: Cell<Duration>,
        ending: RefCell<Ending<'a>>,
        asked: RefCell<Vec<Duration>>,
    }

    impl<'a> Host<'a> {
        fn new(ending: impl FnMut(usize, Duration) -> (Duration, Woke) + 'a) -> Self {
            Self {
                start: Instant::now(),
                elapsed: Cell::new(Duration::ZERO),
                ending: RefCell::new(Box::new(ending)),
                asked: RefCell::new(Vec::new()),
            }
        }

        /// A host that ends every wait by its timeout, `per_mille` parts per thousand of its
        /// length and `extra` late.
        fn late(per_mille: u64, extra: Duration) -> Self {
            Self::new(move |_, asked| (asked + scaled(asked, per_mille) + extra, Woke::TimedOut))
        }

        /// How many waits it has made.
        fn waits(&self) -> usize {
            self.asked.borrow().len()
        }

        /// One sleep of `limit`: how late it ended and how many waits it took.
        fn sleep(&self, limit: Duration, leeway: &mut Leeway) -> (Duration, usize) {
            let (start, waits) = (self.now(), self.waits());
            sleep_for(self, limit, leeway);
            let over = (self.now() - start).saturating_sub(limit);
            (over, self.waits() - waits)
        }
    }

    impl Waiter for Host<'_> {
        fn now(&self) -> Instant {
            self.start + self.elapsed.get()
        }

        fn wait(&self, asked: Duration) -> Woke {
            let number = self.waits();
            assert!(number < MOST_WAITS, "the sleep did not end");
            self.asked.borrow_mut().push(asked);
            let (took, woke) = (self.ending.borrow_mut())(number, asked);
            self.elapsed.set(self.elapsed.get() + took);
            woke
        }
    }

    /// `length` scaled by `per_mille` parts per thousand, rounded down.
    fn scaled(length: Duration, per_mille: u64) -> Duration {
        let nanos = length.as_nanos() * u128::from(per_mille) / PER_MILLE;
        Duration::from_nanos(u64::try_from(nanos).unwrap())
    }

    /// Against a host that ends every wait late by a share of its length (half, as from an
    /// agent's shell; about seven times, as under `taskpolicy -c utility`; and the largest
    /// estimate), the first sleep, with nothing measured yet, is one wait that late; every later
    /// one ends at most [`ON_TIME`] late, or as late as its last wait of at most [`LAST_WAIT`]
    /// made it, in at most 8 waits.
    #[test]
    fn against_a_host_that_ends_waits_late_a_sleep_learns_to_end_on_time() {
        for late in [500, 7_000, MOST] {
            let host = Host::late(late, Duration::ZERO);
            let mut leeway = Leeway::default();
            let ended: Vec<_> = (0..8).map(|_| host.sleep(LONGEST, &mut leeway)).collect();
            assert_eq!(ended[0], (scaled(LONGEST, late), 1));
            let allowed = ON_TIME.max(LAST_WAIT + scaled(LAST_WAIT, late));
            for &(over, waits) in &ended[1..] {
                assert!(over <= allowed && waits <= 8, "{late}: {ended:?}");
            }
        }
    }

    /// A host that ends every wait at most [`ON_TIME`] late keeps each sleep one wait; a host a
    /// nanosecond later than that is measured, and a sleep becomes more than one wait.
    #[test]
    fn against_an_on_time_host_a_sleep_is_one_wait() {
        let host = Host::late(0, ON_TIME);
        let mut leeway = Leeway::default();
        for _ in 0..8 {
            assert_eq!(host.sleep(LONGEST, &mut leeway), (ON_TIME, 1));
        }
        assert_eq!(leeway, Leeway::default());

        let host = Host::late(0, ON_TIME + Duration::from_nanos(1));
        host.sleep(LONGEST, &mut leeway);
        assert!(leeway.per_mille() > 0);
        assert!(host.sleep(LONGEST, &mut leeway).1 > 1);
    }

    /// One wait 20 times late (the thread backgrounded or paused), then a host whose waits end
    /// 20 µs late: within 15 sleeps of [`PREEMPTION_INTERVAL_NS`] each sleep is one wait again
    /// and the estimate is back at zero. Before every wait could lower the estimate, it stuck at
    /// 20,000 and every sleep took 74 waits.
    #[test]
    fn after_one_late_outlier_an_on_time_host_brings_a_sleep_back_to_one_wait() {
        let host = Host::new(|number, asked| {
            let late = if number == 0 {
                asked * 20
            } else {
                Duration::from_micros(20)
            };
            (asked + late, Woke::TimedOut)
        });
        let mut leeway = Leeway::default();
        let waits: Vec<_> = (0..20)
            .map(|_| host.sleep(LONGEST, &mut leeway).1)
            .collect();
        assert_eq!(waits[0], 1);
        assert!(waits[15..].iter().all(|&count| count == 1), "{waits:?}");
        assert_eq!(leeway, Leeway::default());
    }

    /// From the largest estimate, on a host whose waits end on time, the waits per sleep fall
    /// from the most the estimate allows back to one within 30 sleeps.
    #[test]
    fn from_the_largest_estimate_the_wake_rate_falls_back() {
        let host = Host::late(0, Duration::ZERO);
        let mut leeway = Leeway::new(MOST);
        let waits: Vec<_> = (0..40)
            .map(|_| host.sleep(LONGEST, &mut leeway).1)
            .collect();
        assert!(waits[0] > 8, "{waits:?}");
        assert!(waits[30..].iter().all(|&count| count == 1), "{waits:?}");
    }

    /// The real bell, from the largest estimate, on this host: the last 10 of 50 sleeps of
    /// [`PREEMPTION_INTERVAL_NS`] take at most 8 waits each on average, where an estimate stuck
    /// at the largest takes about 300.
    #[test]
    fn from_the_largest_estimate_the_real_bell_falls_back_to_a_few_waits() {
        struct Counting<'a>(&'a Bell, Cell<usize>);
        impl Waiter for Counting<'_> {
            fn now(&self) -> Instant {
                self.0.now()
            }
            fn wait(&self, asked: Duration) -> Woke {
                self.1.set(self.1.get() + 1);
                self.0.wait(asked)
            }
        }
        let bell = Bell::default();
        let counting = Counting(&bell, Cell::new(0));
        let mut leeway = Leeway::new(MOST);
        let mut waits = Vec::new();
        for _ in 0..50 {
            let before = counting.1.get();
            sleep_for(&counting, LONGEST, &mut leeway);
            waits.push(counting.1.get() - before);
        }
        let last: usize = waits[40..].iter().sum();
        assert!(last <= 80, "{waits:?}");
    }

    /// Neither a spurious wake-up nor a ring is measured: a sleep whose first wait ends early
    /// (a measure would lower the estimate) and whose second is rung after 30 times its length
    /// (a measure would raise it) leaves the estimate as it was, and ends at the ring.
    #[test]
    fn a_spurious_or_rung_wake_is_not_measured() {
        let host = Host::new(|number, asked| match number {
            0 => (asked / 2, Woke::Early),
            _ => (asked * 30, Woke::Rung),
        });
        let mut leeway = Leeway::new(800);
        sleep_for(&host, LONGEST, &mut leeway);
        assert_eq!(host.waits(), 2);
        assert_eq!(leeway, Leeway::new(800));
    }

    /// Each wait is measured from its own start, not the sleep's: on a host that ends waits on
    /// time, a sleep of several waits only lowers the estimate, once per wait.
    #[test]
    fn each_wait_is_measured_from_its_own_start() {
        let host = Host::late(0, Duration::ZERO);
        let mut leeway = Leeway::new(4_000);
        sleep_for(&host, LONGEST, &mut leeway);
        let waits = host.waits();
        assert!(waits > 1, "{waits}");
        let mut expected = Leeway::new(4_000);
        for &asked in host.asked.borrow().iter() {
            expected = expected.measured(asked, asked);
        }
        assert_eq!(leeway, expected);
        assert!(leeway.per_mille() < 4_000);
    }

    /// A limit past the end of `Instant` is one wait for the whole of it (as long as a `Duration`
    /// of `u64` nanoseconds goes), whether it times out or wakes early.
    #[test]
    fn a_limit_past_the_end_of_time_is_one_wait() {
        for woke in [Woke::TimedOut, Woke::Early] {
            let host = Host::new(move |_, _| (Duration::from_millis(1), woke));
            sleep_for(&host, Duration::MAX, &mut Leeway::default());
            let whole = Leeway::default().slice(Duration::MAX);
            assert_eq!(*host.asked.borrow(), [whole], "{woke:?}");
        }
    }

    /// A sleep nobody rings lasts at least its limit, whether it is one wait (no lateness
    /// measured) or several (an estimate that slices it); a ring before the sleep keeps it from
    /// starting, and a ring from another thread ends it, each long before its limit; and either
    /// ring is used up by the sleep it ends.
    #[test]
    fn a_sleep_lasts_its_limit_and_a_ring_ends_it() {
        let bell = Bell::default();
        let limit = Duration::from_millis(30);
        for estimate in [Leeway::default(), Leeway::new(4_000), Leeway::new(MOST)] {
            let mut leeway = estimate;
            let start = Instant::now();
            sleep_for(&bell, limit, &mut leeway);
            assert!(
                start.elapsed() >= limit,
                "{estimate:?}: {:?}",
                start.elapsed()
            );
        }

        bell.ring();
        let start = Instant::now();
        sleep_for(&bell, LONG, &mut Leeway::new(4_000));
        assert!(start.elapsed() < LONG / 2, "{:?}", start.elapsed());
        assert!(!bell.take_ring());

        thread::scope(|scope| {
            scope.spawn(|| {
                thread::sleep(limit);
                bell.ring();
            });
            let start = Instant::now();
            sleep_for(&bell, LONG, &mut Leeway::new(4_000));
            assert!(start.elapsed() < LONG / 2, "{:?}", start.elapsed());
        });
        assert!(!bell.take_ring());
    }
}
