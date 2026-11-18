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
//! A listener starts the vsync clock with its deadline one interval (16,666,666 ns) out, more
//! than [`PREEMPTION_INTERVAL_NS`], so a sleep that began before the start ends before the
//! deadline and the next one is bounded by it; no ring is needed for a start
//! (`a_waiting_vsync_clock_reads_nothing_and_a_due_one_ticks` checks the two intervals). A tick made by
//! this thread reaches a waiting guest
//! (`the_timer_thread_ticks_the_vsync_clock_and_wakes_a_listener_s_waiter`).

use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use starquake_cpu::Vcpu;
use starquake_gsv::completion::Wake;
use starquake_kernel::dispatch::{Caller, Dispatcher};
use starquake_kernel::sched::PREEMPTION_INTERVAL_NS;
use starquake_trace::{Env, HostClock, HostNanos, ThreadId};

use super::{CORE, Ending, LogSink, Machine, lock};
use crate::hle::graphics::KernelHost;

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

    /// Sleeps for at most `limit`, or until rung.
    fn sleep(&self, limit: Duration) {
        let mut rung = lock(&self.rung);
        if !*rung {
            rung = self
                .ring
                .wait_timeout(rung, limit)
                .map_or_else(|poisoned| poisoned.into_inner().0, |(rung, _)| rung);
        }
        *rung = false;
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
            // PROVISIONAL: nothing keeps the environment's recording (`machine` module docs).
            drop(machine.env.take_recording());
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
        machine.bell.sleep(sleep);
    }
    Ok(())
}

/// Delivers the deferred replies of every server a [`Ready`](super::Ready) named since the last
/// turn ([`Dispatcher::deliver_replies`]), then tells the cores whose thread changed.
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
        machine
            .dispatcher
            .deliver_replies(server, &mut caller)
            .map_err(|error| Ending::Failed(format!("a reply could not be delivered: {error}")))?;
    }
    // PROVISIONAL: nothing keeps the environments' recordings (`machine` module docs). Every
    // read `fsp-srv`'s workers finish rings this turn, so what they read is dropped here.
    drop(machine.env.take_recording());
    for reads in &machine.reads {
        drop(reads.take_recording());
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
    // PROVISIONAL: nothing keeps the environment's recording (`machine` module docs).
    drop(machine.env.take_recording());
    turned.map_err(failed)?;
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
