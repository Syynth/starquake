//! A run's time: the console its process was built for, which decides its counter rate, and
//! the host clock its guest time is counted from.
//!
//! The kernel owns guest time (`docs/specs/kernel.md`, "Guest time"): it converts the
//! environment's clock, host nanoseconds, with the process's counter rate, and takes the
//! clock's zero as guest tick zero. Which rate a process gets is the kernel's choice too, made
//! from the console in its description (`starquake_kernel::process::Console::counter_rate`).
//! This module is the other half of that contract. It says which console the program was built
//! for, hands that console's rate to the loader's counter patcher, and builds the environment's
//! clock from the counter the patched instructions read, so that `GetSystemTick` and a patched
//! counter read agree on the rate and on where the counter started. `tests/skeleton.rs` reads a
//! tick each way and checks that they do.
//!
//! PROVISIONAL (walking skeleton, #74): every program is described as built for Switch 1
//! ([`PROCESS_CONSOLE`]). An NRO carries nothing that says which console it was built for; the
//! homebrew this skeleton runs is built for the first one (a guess, and the rate the skeleton
//! ran at before the kernel kept guest time). Wiring the loader's NPDM into process creation
//! (#33) takes the console from the program's metadata.

use std::num::NonZeroU64;

use starquake_kernel::process::Console;
use starquake_kernel::time::{CounterRate, NANOS_PER_SECOND};
use starquake_trace::HostNanos;

/// The console every program is described as built for, for now. See the module docs.
pub(crate) const PROCESS_CONSOLE: Console = Console::Switch1;

/// What a run counts time with: the console its process was built for, and the host clock.
///
/// The clock is cloned: one copy goes into the environment, where the kernel reads it and every
/// reading is recorded, and the loop keeps the other to stamp the events it records itself
/// (PROVISIONAL: lifecycle and fault events, until the kernel records them). Both copies must
/// read the same time, so the clock is one that reads the host, not one that counts its own
/// readings.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Timebase<C> {
    /// The console the process was built for: the kernel gives it that console's counter rate,
    /// which is the patcher's guest rate too ([`Timebase::rate`]).
    pub(crate) console: Console,
    /// The host clock whose zero is guest tick zero.
    pub(crate) clock: C,
}

impl<C> Timebase<C> {
    /// The rate of the process's counter, as the kernel decides it from the console.
    pub(crate) const fn rate(&self) -> CounterRate {
        self.console.counter_rate()
    }
}

/// A count of a `hz` hertz host counter, as host nanoseconds since the counter's zero: rounded
/// down, and saturating at `u64::MAX` (a counter slower than one tick per nanosecond reaches
/// that only after centuries).
pub(crate) fn counter_nanos(ticks: u64, hz: NonZeroU64) -> HostNanos {
    let nanos = u128::from(ticks) * u128::from(NANOS_PER_SECOND) / u128::from(hz.get());
    HostNanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}

/// The environment's clock for a run under the hypervisor backend: the counter guest code
/// reads there (`starquake_cpu::host_counter`), as nanoseconds since that counter's zero.
///
/// A patched counter read is `counter * guest_hz / host_hz`; the kernel's `GetSystemTick` is
/// `nanoseconds * guest_hz / 10^9` with nanoseconds from here, `counter * 10^9 / host_hz`. Same
/// counter, same zero, so the two differ only by rounding: each is the exact
/// `counter * guest_hz / host_hz` rounded down, or one tick less.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CounterClock {
    hz: NonZeroU64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl CounterClock {
    /// A clock over the backend's counter, which runs at `hz` hertz
    /// (`HvfBackend::counter_frequency`). `None` for a rate of zero.
    pub(crate) fn new(hz: u64) -> Option<Self> {
        NonZeroU64::new(hz).map(|hz| Self { hz })
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl starquake_trace::HostClock for CounterClock {
    fn now(&mut self) -> HostNanos {
        counter_nanos(starquake_cpu::host_counter::read(), self.hz)
    }
}

#[cfg(test)]
mod tests {
    use starquake_loader::patch::CounterRescale;

    use super::*;

    /// Apple Silicon's counter rate (`tools/hvprobe/README.md`: 24 MHz).
    const HOST_HZ: NonZeroU64 = NonZeroU64::new(24_000_000).unwrap();

    #[test]
    fn a_second_of_counter_ticks_is_a_second_of_nanoseconds() {
        assert_eq!(counter_nanos(0, HOST_HZ), HostNanos(0));
        assert_eq!(
            counter_nanos(HOST_HZ.get(), HOST_HZ),
            HostNanos(NANOS_PER_SECOND)
        );
        assert_eq!(
            counter_nanos(7 * HOST_HZ.get(), HOST_HZ),
            HostNanos(7 * NANOS_PER_SECOND)
        );
    }

    #[test]
    fn nanoseconds_round_down_and_saturate_at_the_top() {
        // One tick of a 24 MHz counter is 41.67 ns; three are exactly 125.
        assert_eq!(counter_nanos(1, HOST_HZ), HostNanos(41));
        assert_eq!(counter_nanos(3, HOST_HZ), HostNanos(125));
        // The largest count of a counter slower than a tick per nanosecond does not fit...
        assert_eq!(counter_nanos(u64::MAX, HOST_HZ), HostNanos(u64::MAX));
        // ...and at exactly one tick per nanosecond every count is its own nanoseconds.
        let per_nanosecond = NonZeroU64::new(NANOS_PER_SECOND).unwrap();
        assert_eq!(counter_nanos(u64::MAX, per_nanosecond), HostNanos(u64::MAX));
        assert_eq!(
            counter_nanos(u64::MAX - 1, per_nanosecond),
            HostNanos(u64::MAX - 1)
        );
    }

    #[test]
    fn the_kernels_tick_and_a_patched_read_of_one_count_differ_only_by_rounding() {
        // The two ways a guest reads its counter, computed for the same counter value: the
        // kernel's (through nanoseconds) and the patcher's trampoline (`FixedRatio::scale`).
        // Each is the exact value rounded down, or one tick less.
        for rate in [CounterRate::SWITCH_1, CounterRate::SWITCH_2] {
            let patched = CounterRescale {
                host_hz: HOST_HZ.get(),
                guest_hz: rate.hz(),
            }
            .ratio()
            .unwrap();
            // Small counts, a count with every remainder in play, and one a year into a run.
            let year = 365 * 24 * 60 * 60 * HOST_HZ.get();
            for count in [
                0,
                1,
                2,
                3,
                4,
                5,
                1_000_003,
                HOST_HZ.get() - 1,
                year + 12_345,
            ] {
                let exact = u128::from(count) * u128::from(rate.hz()) / u128::from(HOST_HZ.get());
                let exact = u64::try_from(exact).unwrap();
                let kernel = rate.ticks_at(counter_nanos(count, HOST_HZ)).0;
                let trampoline = patched.scale(count);
                for (name, value) in [("kernel", kernel), ("trampoline", trampoline)] {
                    assert!(
                        value <= exact && exact - value <= 1,
                        "{name}: {value} for {count} at {rate:?}, exact {exact}"
                    );
                }
            }
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn the_counter_clock_reads_the_backends_counter_and_refuses_a_rate_of_zero() {
        use starquake_trace::HostClock as _;

        assert!(CounterClock::new(0).is_none());
        let mut clock = CounterClock::new(HOST_HZ.get()).unwrap();
        let before = counter_nanos(starquake_cpu::host_counter::read(), HOST_HZ);
        let first = clock.now();
        let second = clock.now();
        let after = counter_nanos(starquake_cpu::host_counter::read(), HOST_HZ);
        assert!(before <= first && first <= second && second <= after);
        // A copy is the same clock: it reads the host, not its own count of readings.
        let mut copy = clock;
        assert!(copy.now() >= second);
    }
}
