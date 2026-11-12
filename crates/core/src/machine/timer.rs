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
//! Not here yet: vsync, and draining the graphics server's completion queue (both come with
//! the graphics server).

use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use starquake_cpu::Vcpu;
use starquake_kernel::dispatch::{Caller, Dispatcher};
use starquake_kernel::sched::PREEMPTION_INTERVAL_NS;
use starquake_trace::{HostClock, HostNanos, ThreadId};

use super::{CORE, Ending, LogSink, Machine, lock};

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
        if next_deadline(&machine.dispatcher).is_some_and(|deadline| deadline <= now)
        {
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
        let woken = machine.dispatcher.take_woken();
        if !woken.is_empty() {
            tracing::trace!(count = woken.len(), "guest threads woken");
        }

        let now = clock.host_now();
        let mut sleep = until(now, next_preemption);
        if let Some(deadline) = next_deadline(&machine.dispatcher) {
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
pub(super) fn deliver<V: Vcpu, C: HostClock + Clone>(machine: &Machine<'_, V, C>) -> Result<(), Ending> {
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
    drop(machine.env.take_recording());
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
