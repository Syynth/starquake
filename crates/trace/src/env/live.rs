//! The live [`Env`]: reads the host and records what it returned.

use std::sync::Mutex;
use std::time::Instant;

use crate::env::{Env, EnvError, EnvRecord, HostIoReply, HostIoRequest};
use crate::event::HostNanos;
use crate::recorder::lock;

/// Where a [`LiveEnv`] gets the time from.
pub trait HostClock: Send {
    /// The current time: host nanoseconds since this clock's zero (see [`Env::now`]). Successive
    /// readings from one clock should not decrease.
    fn now(&mut self) -> HostNanos;
}

/// The real monotonic host clock: nanoseconds since the clock was created, saturating at
/// `u64::MAX`. Its zero is its own creation, which is not the zero of any counter a CPU backend
/// shows the guest; a run whose guest reads a counter directly needs a clock built on that
/// counter instead (`starquake-core` has one for the hypervisor backend). A copy reads the
/// same time: it shares the zero.
#[derive(Clone, Copy, Debug)]
pub struct SystemClock {
    start: Instant,
}

impl SystemClock {
    /// A clock whose zero is now.
    #[must_use]
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl HostClock for SystemClock {
    fn now(&mut self) -> HostNanos {
        HostNanos(u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX))
    }
}

/// A deterministic clock for `--clock fixed` runs (spec, "Headless runs"): the first reading is
/// `start` nanoseconds, and every later reading is `step` nanoseconds more than the one before
/// (saturating).
#[derive(Clone, Copy, Debug)]
pub struct FixedClock {
    next: u64,
    step: u64,
}

impl FixedClock {
    /// A clock that reads `start`, `start + step`, `start + 2 * step` nanoseconds, and so on.
    #[must_use]
    pub fn new(start: u64, step: u64) -> Self {
        Self { next: start, step }
    }
}

impl HostClock for FixedClock {
    fn now(&mut self) -> HostNanos {
        let reading = self.next;
        self.next = reading.saturating_add(self.step);
        HostNanos(reading)
    }
}

/// Answers host I/O requests for a [`LiveEnv`] (the extension point described in the
/// [`env`](crate::env) module docs).
pub trait HostIoHandler: Send {
    /// Handle one request.
    fn handle(&mut self, request: &HostIoRequest) -> HostIoReply;
}

/// `SplitMix64` (Steele, Lea and Flood, "Fast splittable pseudorandom number generators", 2014;
/// public-domain reference implementation by Vigna): a small seedable generator. It is not
/// cryptographic, which is fine because the guest only needs numbers it cannot predict from
/// outside a fixed-seed run, and replays need the seed to be the only input.
#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

struct State {
    clock: Box<dyn HostClock>,
    rng: SplitMix64,
    handler: Option<Box<dyn HostIoHandler>>,
    recording: Vec<EnvRecord>,
}

/// The live environment. Every answer it gives is appended to a recording that
/// [`LiveEnv::recording`] returns and [`ReplayEnv`](crate::ReplayEnv) can feed back.
///
/// The recording grows without bound; a long run that needs a cap should drain it with
/// [`LiveEnv::take_recording`] periodically.
pub struct LiveEnv {
    state: Mutex<State>,
}

impl LiveEnv {
    /// A live environment reading `clock`, drawing random numbers from a generator seeded with
    /// `seed` (spec: `--seed N`), and with no host I/O handler.
    #[must_use]
    pub fn new(clock: impl HostClock + 'static, seed: u64) -> Self {
        Self {
            state: Mutex::new(State {
                clock: Box::new(clock),
                rng: SplitMix64 { state: seed },
                handler: None,
                recording: Vec::new(),
            }),
        }
    }

    /// Install the handler that answers host I/O requests.
    #[must_use]
    pub fn with_host_io(self, handler: impl HostIoHandler + 'static) -> Self {
        lock(&self.state).handler = Some(Box::new(handler));
        self
    }

    /// A copy of everything answered so far, in order.
    #[must_use]
    pub fn recording(&self) -> Vec<EnvRecord> {
        lock(&self.state).recording.clone()
    }

    /// Take everything answered so far, leaving the recording empty.
    #[must_use]
    pub fn take_recording(&self) -> Vec<EnvRecord> {
        std::mem::take(&mut lock(&self.state).recording)
    }
}

impl Env for LiveEnv {
    fn now(&self) -> Result<HostNanos, EnvError> {
        let mut state = lock(&self.state);
        let reading = state.clock.now();
        state.recording.push(EnvRecord::Clock(reading));
        Ok(reading)
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        let mut state = lock(&self.state);
        let value = state.rng.next();
        state.recording.push(EnvRecord::Random(value));
        Ok(value)
    }

    fn host_io(&self, request: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        let mut state = lock(&self.state);
        let reply = match state.handler.as_mut() {
            Some(handler) => handler.handle(request),
            None => HostIoReply::Unavailable,
        };
        state.recording.push(EnvRecord::HostIo {
            request: request.clone(),
            reply: reply.clone(),
        });
        Ok(reply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_matches_the_published_reference_stream() {
        // First outputs of the reference implementation (https://prng.di.unimi.it/splitmix64.c)
        // seeded with 0, computed independently of this code.
        let mut rng = SplitMix64 { state: 0 };
        assert_eq!(rng.next(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn fixed_clock_steps_and_saturates() {
        let mut clock = FixedClock::new(u64::MAX - 1, 5);
        assert_eq!(clock.now(), HostNanos(u64::MAX - 1));
        assert_eq!(clock.now(), HostNanos(u64::MAX));
        assert_eq!(clock.now(), HostNanos(u64::MAX));
        let mut clock = FixedClock::new(10, 3);
        let readings: Vec<u64> = (0..3).map(|_| clock.now().0).collect();
        assert_eq!(readings, [10, 13, 16]);
    }

    #[test]
    fn system_clock_does_not_go_backwards() {
        let mut clock = SystemClock::default();
        let a = clock.now();
        let b = clock.now();
        assert!(b >= a);
    }

    #[test]
    fn a_copy_of_the_system_clock_shares_its_zero() {
        let mut clock = SystemClock::default();
        let mut copy = clock;
        let a = clock.now();
        let b = copy.now();
        let c = clock.now();
        assert!(a <= b && b <= c, "{a:?} {b:?} {c:?}");
    }

    #[test]
    fn the_same_seed_gives_the_same_numbers_and_different_seeds_differ() {
        let draw = |seed| {
            let env = LiveEnv::new(FixedClock::new(0, 1), seed);
            (0..4)
                .map(|_| env.random_u64().unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
        assert_ne!(draw(7), draw(8));
    }

    #[test]
    fn answers_are_recorded_in_call_order() {
        let env = LiveEnv::new(FixedClock::new(100, 10), 1);
        let t0 = env.now().unwrap();
        let r0 = env.random_u64().unwrap();
        let t1 = env.now().unwrap();
        assert_eq!(
            env.recording(),
            [
                EnvRecord::Clock(t0),
                EnvRecord::Random(r0),
                EnvRecord::Clock(t1)
            ]
        );
        assert_eq!(env.take_recording().len(), 3);
        assert_eq!(env.recording().len(), 0);
    }

    struct Echo;
    impl HostIoHandler for Echo {
        fn handle(&mut self, request: &HostIoRequest) -> HostIoReply {
            if request.channel == 0 {
                HostIoReply::Failed(7)
            } else {
                HostIoReply::Data(request.payload.iter().rev().copied().collect())
            }
        }
    }

    #[test]
    fn host_io_goes_to_the_handler_and_is_recorded_with_its_request() {
        let env = LiveEnv::new(FixedClock::new(0, 1), 0).with_host_io(Echo);
        let ok = HostIoRequest {
            channel: 1,
            payload: vec![1, 2, 3],
        };
        let bad = HostIoRequest {
            channel: 0,
            payload: vec![],
        };
        assert_eq!(env.host_io(&ok).unwrap(), HostIoReply::Data(vec![3, 2, 1]));
        assert_eq!(env.host_io(&bad).unwrap(), HostIoReply::Failed(7));
        assert_eq!(
            env.recording(),
            [
                EnvRecord::HostIo {
                    request: ok,
                    reply: HostIoReply::Data(vec![3, 2, 1])
                },
                EnvRecord::HostIo {
                    request: bad,
                    reply: HostIoReply::Failed(7)
                },
            ]
        );
    }

    #[test]
    fn without_a_handler_host_io_is_unavailable_and_still_recorded() {
        let env = LiveEnv::new(FixedClock::new(0, 1), 0);
        let request = HostIoRequest {
            channel: 3,
            payload: vec![9],
        };
        assert_eq!(env.host_io(&request).unwrap(), HostIoReply::Unavailable);
        assert_eq!(
            env.recording(),
            [EnvRecord::HostIo {
                request,
                reply: HostIoReply::Unavailable
            }]
        );
    }
}
