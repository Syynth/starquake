//! The acquire poll and its back-off (gpu-protocol spec, "Semaphore acquires", step 3; owner's
//! decision 3 (a) of 2026-11-12): while a channel waits on something the CPU may change without
//! telling the GPU (a semaphore in guest memory, a syncpoint gsv's video worker increments), the
//! front looks again on a timer that spins briefly and then sleeps in growing steps up to about
//! 100 µs.
//!
//! [`Backoff`] is the timer's schedule, a pure state machine; [`FrontClock`] is the clock the
//! front measures it on, injected so tests use a fake ([`HostClock`] in production).
//!
//! # The schedule
//!
//! [`Backoff::next_sleep`] returns how long the front may sleep before it polls again: zero for the
//! first [`POLL_SPINS`] polls (it spins), then [`POLL_FIRST_SLEEP`], doubling at each poll up to
//! [`POLL_LONGEST_SLEEP`], where it stays. [`Backoff::reset`] starts the schedule again; the front
//! resets it when a channel newly waits on an acquire or a fence, when a recheck lets a waiting
//! channel go on (activity that suggests the others' semaphores may move soon too), and when no
//! channel is left waiting (`crate::front`, "One turn", names the tests). The numbers other
//! than the 100 µs ceiling are **guesses**, to be measured (spec verification item 2).
//!
//! # Why the poll is far shorter than the ring's 100 ms safety net
//!
//! The ring's 100 ms recheck (`crate::ring`) and the front's own [`IDLE_PARK`] bound a sleep
//! against a lost wake-up: every event that should end them (a publish, a submission, a back
//! report) also wakes the sleeper, and the bound only catches a wake that went missing. The poll
//! is different: a CPU write to a semaphore wakes nobody (decision 3 chose polling over trapping
//! the write), so the poll is the only way the front learns of it, and its interval is the
//! latency a CPU-released channel pays. It is therefore bounded by decision 3's 100 µs, not by
//! the safety net.

use std::time::{Duration, Instant};

/// Polls made with no sleep before the first sleep (a **guess**: "spin briefly", decision 3).
pub const POLL_SPINS: u32 = 64;

/// The first sleep after the spins (a **guess**; the steps double from it).
pub const POLL_FIRST_SLEEP: Duration = Duration::from_micros(1);

/// The longest sleep between two polls: decision 3's "growing steps up to about 100 µs".
pub const POLL_LONGEST_SLEEP: Duration = Duration::from_micros(100);

/// How long the front sleeps when nothing waits on a poll: until a submission or a back report
/// wakes it, with this bound only as a safety net against a lost wake-up (the ring's own bound,
/// `crate::ring`, is the same 100 ms).
pub const IDLE_PARK: Duration = Duration::from_millis(100);

/// The front thread's clock, injected so tests and replays control it.
pub trait FrontClock: Send + Sync {
    /// Time since an arbitrary fixed origin, never going backwards.
    fn now(&self) -> Duration;
}

/// The host's monotonic clock, from when it was made.
#[derive(Clone, Copy, Debug)]
pub struct HostClock {
    origin: Instant,
}

impl HostClock {
    /// A clock that reads zero now.
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for HostClock {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontClock for HostClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
}

/// The poll's schedule (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Backoff {
    /// Polls made so far with no sleep.
    spins: u32,
    /// The sleep the next step after the spins returns.
    sleep: Duration,
}

impl Backoff {
    /// A schedule at its start: [`POLL_SPINS`] spins to come.
    pub const fn new() -> Self {
        Self {
            spins: 0,
            sleep: POLL_FIRST_SLEEP,
        }
    }

    /// Starts the schedule again.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// How long to sleep before the next poll, advancing the schedule.
    pub fn next_sleep(&mut self) -> Duration {
        if self.spins < POLL_SPINS {
            self.spins += 1;
            return Duration::ZERO;
        }
        let sleep = self.sleep;
        self.sleep = self
            .sleep
            .checked_mul(2)
            .map_or(POLL_LONGEST_SLEEP, |doubled| {
                doubled.min(POLL_LONGEST_SLEEP)
            });
        sleep
    }
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new()
    }
}
