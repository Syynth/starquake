//! How late the host ends the timer thread's timed waits, and how long to wait next so that a
//! sleep still ends at its deadline (`machine/timer.rs`, "Sleeping to a deadline").
//!
//! A host may end a timed wait later than asked, on purpose: macOS coalesces timers, letting a
//! wait end anywhere up to a *leeway* after its deadline so that several wake-ups share one, and
//! the leeway grows with the wait's length and with how little the system thinks the process's
//! latency matters. A plain condition-variable wait, measured on an Apple Silicon Mac (median
//! lateness of 30 waits each):
//!
//! | wait  | from an agent's shell | under `taskpolicy -c utility` | under `taskpolicy -b` |
//! |-------|-----------------------|-------------------------------|-----------------------|
//! | 1 ms  | 0.52 ms               | 10.5 ms                       | 48 ms                 |
//! | 5 ms  | 2.5 ms                | 35 ms                         | 156 ms                |
//! | 10 ms | 5.0 ms                | 73 ms                         | 188 ms                |
//!
//! Raising the waiting thread's own quality-of-service class (`pthread_set_qos_class_self_np` with
//! `QOS_CLASS_USER_INTERACTIVE`) changed none of these. The hosted macOS runners of CI end waits
//! late in the same way: requests held to the syncpoint page's 1 ms bound were answered after up
//! to 5.7 ms there, and vsync ticks came up to 69 ms late, by the flight recorder.
//!
//! The leeway is a share of the wait's length, so a wait for a share of the time left ends before
//! the deadline even when it ends late, and each wait leaves less time for the next. The timer
//! thread measures how late its waits end ([`Leeway::measured`]) and asks each time for the share
//! of what is left that, made that late and a quarter more, still ends by the deadline
//! ([`Leeway::slice`]); the last [`LAST_WAIT`] is waited in one go. On a host that ends waits on
//! time the estimate stays at zero and a sleep is one wait, as before.

use std::time::Duration;

/// Parts per thousand: the unit of [`Leeway`].
const PER_MILLE: u128 = 1_000;

/// The largest estimate: a wait ending 64 times its length late. A host later than that still
/// sees short waits (`taskpolicy -b`, above, ends a 1 ms wait up to 64 ms late), and a sleep then
/// ends late; the bound keeps every wait at least 1/81 of the time left.
pub(super) const MOST: u64 = 64_000;

/// At or below this much time left, a sleep waits for the rest in one go: a wait this short ends
/// late by little even on a host that coalesces (0.5 ms for 1 ms, above), and waking more often
/// to save that little would cost more than it saves.
pub(super) const LAST_WAIT: Duration = Duration::from_micros(100);

/// A wait shorter than this is not measured: its lateness is mostly the cost of waking at all.
pub(super) const MEASURED_FROM: Duration = Duration::from_micros(500);

/// How quickly the estimate forgets: a measured wait that ended less late than estimated lowers
/// the estimate by one part in this many, rounded up, so that it reaches zero.
pub(super) const DECAY: u64 = 8;

/// How late the host has been ending the timer thread's timed waits: parts per thousand of a
/// wait's length (module docs). Zero for a host that ends them on time; a short-lived spike (the
/// thread was not scheduled at once) decays away by [`DECAY`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Leeway(u64);

impl Leeway {
    /// An estimate of `per_mille` parts per thousand, at most [`MOST`].
    #[cfg(test)]
    pub(super) fn new(per_mille: u64) -> Self {
        Self(per_mille.min(MOST))
    }

    /// The estimate in parts per thousand.
    #[cfg(test)]
    pub(super) const fn per_mille(self) -> u64 {
        self.0
    }

    /// How long to wait with `left` until the deadline: all of it when that is at most
    /// [`LAST_WAIT`]; otherwise the share of it that, ended late by the estimate and a quarter
    /// more, still ends by the deadline, `left * 1000 / (1000 + estimate * 5 / 4)`, rounded down.
    pub(super) fn slice(self, left: Duration) -> Duration {
        if left <= LAST_WAIT {
            return left;
        }
        let late = u128::from(self.0) + u128::from(self.0) / 4;
        let nanos = left.as_nanos().saturating_mul(PER_MILLE) / (PER_MILLE + late);
        Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
    }

    /// The estimate after a wait for `asked` that timed out after `slept`: unchanged for a wait
    /// shorter than [`MEASURED_FROM`]; otherwise the larger of what this wait shows (its
    /// lateness in parts per thousand of `asked`, rounded down, at most [`MOST`]) and the
    /// estimate less one part in [`DECAY`], rounded up.
    pub(super) fn measured(self, asked: Duration, slept: Duration) -> Self {
        if asked < MEASURED_FROM {
            return self;
        }
        let late = slept.saturating_sub(asked).as_nanos();
        let seen = late.saturating_mul(PER_MILLE) / asked.as_nanos();
        let seen = u64::try_from(seen).unwrap_or(MOST).min(MOST);
        Self(seen.max(self.0 - self.0.div_ceil(DECAY)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Estimates from none to the largest, in parts per thousand.
    const ESTIMATES: [u64; 6] = [0, 1, 125, 500, 4_000, MOST];

    /// Times left from just over [`LAST_WAIT`] to longer than any sleep of the timer thread.
    const LEFT: [Duration; 5] = [
        LAST_WAIT.saturating_add(Duration::from_nanos(1)),
        Duration::from_micros(733),
        Duration::from_millis(1),
        Duration::from_nanos(16_666_666),
        Duration::from_secs(3),
    ];

    /// `estimate * 5 / 4`, the lateness [`Leeway::slice`] allows for.
    fn allowed(estimate: u64) -> u128 {
        u128::from(estimate) + u128::from(estimate) / 4
    }

    #[test]
    fn at_most_the_last_wait_left_is_waited_in_one_go() {
        for estimate in ESTIMATES {
            let leeway = Leeway::new(estimate);
            assert_eq!(leeway.slice(LAST_WAIT), LAST_WAIT);
            assert_eq!(
                leeway.slice(Duration::from_nanos(1)),
                Duration::from_nanos(1)
            );
            assert_eq!(leeway.slice(Duration::ZERO), Duration::ZERO);
        }
        // One nanosecond more is sliced.
        let over = LAST_WAIT + Duration::from_nanos(1);
        assert!(Leeway::new(1).slice(over) < over);
        assert_eq!(Leeway::default().slice(over), over);
    }

    #[test]
    fn with_no_lateness_seen_a_sleep_is_one_wait() {
        for left in LEFT {
            assert_eq!(Leeway::default().slice(left), left);
        }
    }

    #[test]
    fn a_slice_is_the_longest_wait_that_ends_by_the_deadline_when_ended_late_as_allowed() {
        for estimate in ESTIMATES {
            let late = allowed(estimate);
            for left in LEFT {
                let slice = Leeway::new(estimate).slice(left).as_nanos();
                let left = left.as_nanos();
                assert!(
                    slice * (PER_MILLE + late) <= left * PER_MILLE,
                    "{estimate} {left}"
                );
                assert!(
                    (slice + 1) * (PER_MILLE + late) > left * PER_MILLE,
                    "{estimate} {left}"
                );
            }
        }
        // A 4x estimate allows 5x, so a sixth of the time left.
        assert_eq!(
            Leeway::new(4_000).slice(Duration::from_millis(12)),
            Duration::from_millis(2)
        );
    }

    #[test]
    fn a_wait_shorter_than_the_measured_minimum_changes_nothing() {
        let short = MEASURED_FROM.checked_sub(Duration::from_nanos(1)).unwrap();
        for estimate in ESTIMATES {
            let leeway = Leeway::new(estimate);
            assert_eq!(leeway.measured(short, short * 100), leeway);
            assert_eq!(leeway.measured(short, short), leeway);
        }
        // At the minimum, the wait is measured: 100 times late is capped at the most.
        let measured = Leeway::default().measured(MEASURED_FROM, MEASURED_FROM * 101);
        assert_eq!(measured.per_mille(), MOST);
    }

    #[test]
    fn a_later_wait_raises_the_estimate_to_what_it_shows() {
        let asked = Duration::from_millis(2);
        // 1.3 ms late on 2 ms: 650 per mille.
        let slept = asked + Duration::from_micros(1_300);
        assert_eq!(Leeway::default().measured(asked, slept).per_mille(), 650);
        assert_eq!(Leeway::new(649).measured(asked, slept).per_mille(), 650);
        // Rounded down: 1 ns late on 2 ms is nothing.
        let barely = asked + Duration::from_nanos(1);
        assert_eq!(Leeway::default().measured(asked, barely).per_mille(), 0);
        // Exactly the most, and past it.
        assert_eq!(
            Leeway::default().measured(asked, asked * 65).per_mille(),
            MOST
        );
        assert_eq!(
            Leeway::default().measured(asked, asked * 66).per_mille(),
            MOST
        );
        assert_eq!(
            Leeway::default()
                .measured(
                    asked,
                    (asked * 65).checked_sub(Duration::from_micros(2)).unwrap()
                )
                .per_mille(),
            MOST - 1
        );
    }

    #[test]
    fn a_wait_less_late_than_estimated_lowers_the_estimate_by_an_eighth_rounded_up() {
        let asked = Duration::from_millis(1);
        // On time: 800 falls by 100.
        assert_eq!(Leeway::new(800).measured(asked, asked).per_mille(), 700);
        // A wait that ended early (a spurious wake counted as a timeout) is on time.
        assert_eq!(Leeway::new(800).measured(asked, asked / 2).per_mille(), 700);
        // 650 late on 800: the decayed 700 is the larger. 750 late: 750 is.
        let late = |micros| asked + Duration::from_micros(micros);
        assert_eq!(Leeway::new(800).measured(asked, late(650)).per_mille(), 700);
        assert_eq!(Leeway::new(800).measured(asked, late(750)).per_mille(), 750);
        // Rounded up, so an estimate reaches zero.
        assert_eq!(Leeway::new(7).measured(asked, asked).per_mille(), 6);
        assert_eq!(Leeway::new(1).measured(asked, asked).per_mille(), 0);
        assert_eq!(Leeway::new(9).measured(asked, asked).per_mille(), 7);
    }

    /// The timer thread's sleep (`Bell::sleep`) against a modelled host that ends every wait
    /// `late` parts per thousand of its length late: the first sleep, with nothing measured yet,
    /// ends that late; once the estimate has been learnt, every sleep ends at its deadline to
    /// within the last wait's own lateness, and in a few waits. Returns, per sleep, how late it
    /// ended and how many waits it took.
    fn sleeps(late: u64, length: Duration, count: usize) -> Vec<(Duration, usize)> {
        let mut leeway = Leeway::default();
        let mut ended = Vec::new();
        for _ in 0..count {
            let mut slept = Duration::ZERO;
            let mut waits = 0;
            while slept < length {
                let asked = leeway.slice(length.checked_sub(slept).unwrap());
                let took = asked + asked * u32::try_from(late).unwrap() / 1_000;
                leeway = leeway.measured(asked, took);
                slept += took;
                waits += 1;
            }
            ended.push((slept.checked_sub(length).unwrap(), waits));
        }
        ended
    }

    #[test]
    fn against_a_host_that_ends_waits_late_a_sleep_learns_to_end_on_time() {
        let length = Duration::from_millis(10);
        // Late as measured from an agent's shell (half), under a utility clamp (about 7 times)
        // and as late as the estimate goes.
        for late in [500, 7_000, MOST] {
            let ended = sleeps(late, length, 8);
            let first = ended[0];
            assert_eq!(first, (length * u32::try_from(late).unwrap() / 1_000, 1));
            let last_wait = LAST_WAIT + LAST_WAIT * u32::try_from(late).unwrap() / 1_000;
            for &(over, waits) in &ended[1..] {
                assert!(over <= last_wait, "{late}: {ended:?}");
                assert!(waits <= 8, "{late}: {ended:?}");
            }
        }
        // A host that ends waits on time: one wait each, on time.
        assert!(
            sleeps(0, length, 8)
                .iter()
                .all(|&e| e == (Duration::ZERO, 1))
        );
    }
}
