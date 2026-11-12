//! Guest time: the kernel's one conversion from the environment's clock to guest counter ticks.
//!
//! Kernel spec, "Guest time": the kernel owns guest ticks. The environment's clock returns host
//! nanoseconds ([`HostNanos`]), and the kernel converts a reading with the counter rate of the
//! process it is for ([`CounterRate::ticks_at`]). Nothing else converts: other crates pass
//! [`HostNanos`] in and get [`GuestTicks`] out.
//!
//! **One rate.** A process's counter runs at [`CounterRate::SWITCH_1`] or
//! [`CounterRate::SWITCH_2`]. `starquake-core` hands the same rate to the loader's counter
//! patcher as its guest rate, so `GetSystemTick` and a counter read through a patched
//! instruction tick alike.
//!
//! **One zero point.** Guest tick zero is the zero of the environment's clock: reading
//! `HostNanos(0)` is tick 0, whatever the rate. So the two ways a guest reads its counter agree
//! on where it starts exactly when the environment's clock counts from the zero of the counter
//! the patched instructions read. Making that so is the job of whoever builds the environment
//! (`starquake-core`, which builds the clock from the CPU backend's own counter); a test there
//! reads a tick each way and checks they agree. The kernel takes no offset of its own, which is
//! also what keeps a replay exact: the recorded clock readings are the only input.
//!
//! Sources: Eden's `Hardware::CNTFREQ` (`eden/src/core/hardware_properties.h` @ cf654052,
//! "`CNTPCT_EL0` Frequency = 19.2 MHz") and PR #4588's `GetGuestCounterFrequency`
//! (`eden-pr4588/src/core/guest_clock.h` @ e5fb3b97: 31,250,000 Hz for a process that uses the
//! Switch 2 ABI, `CNTFREQ` otherwise; scoping doc, "CPU and ISA": "`CNTFRQ` is 31.25 MHz
//! (Switch 1: 19.2 MHz)").
//!
//! Divergence from PR #4588, pinned by `ticks_follow_the_process_rate_not_a_rescaled_base`: it
//! counts at 19.2 MHz and rescales that count for a Switch 2 process (`GetGuestCounterTicks`),
//! which rounds twice. Here a reading converts once, straight to the process's rate.

use core::num::NonZeroU64;

use starquake_trace::{GuestTicks, HostNanos};

/// Nanoseconds in one second: the unit of [`HostNanos`].
pub const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// The rate of a process's counter (`CNTPCT_EL0`, and what its `CNTFRQ_EL0` reads), in hertz.
/// Fixed when the process is created; never zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CounterRate(NonZeroU64);

impl CounterRate {
    /// A Switch 1 process's rate, 19.2 MHz: Eden's `Hardware::CNTFREQ`
    /// (`eden/src/core/hardware_properties.h`).
    pub const SWITCH_1: Self = Self::known(19_200_000);

    /// A Switch 2 process's rate, 31.25 MHz: PR #4588's `GetGuestCounterFrequency(true)`
    /// (`eden-pr4588/src/core/guest_clock.h`).
    pub const SWITCH_2: Self = Self::known(31_250_000);

    /// One of the constants above. A `const` can't unwrap, so the match needs a second arm; it
    /// is never taken, because both rates are non-zero.
    const fn known(hz: u64) -> Self {
        match NonZeroU64::new(hz) {
            Some(hz) => Self(hz),
            None => Self(NonZeroU64::MIN),
        }
    }

    /// A rate of `hz` hertz, or `None` for zero. The two consoles' rates are the constants
    /// above; this exists for tests and for a rate read from configuration.
    #[must_use]
    pub const fn new(hz: u64) -> Option<Self> {
        match NonZeroU64::new(hz) {
            Some(hz) => Some(Self(hz)),
            None => None,
        }
    }

    /// The rate in hertz.
    #[must_use]
    pub const fn hz(self) -> u64 {
        self.0.get()
    }

    /// The guest counter at a reading of the environment's clock: `now * hz / 10^9`, rounded
    /// down. See the module docs for the zero point.
    ///
    /// The product is formed in `u128`, so it cannot overflow. The quotient fits `u64` whenever
    /// the rate is at most [`NANOS_PER_SECOND`] hertz (both consoles' rates are); for a faster
    /// rate it saturates at `u64::MAX` instead of wrapping.
    #[must_use]
    pub fn ticks_at(self, now: HostNanos) -> GuestTicks {
        let ticks = u128::from(now.0) * u128::from(self.hz()) / u128::from(NANOS_PER_SECOND);
        GuestTicks(u64::try_from(ticks).unwrap_or(u64::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATES: [CounterRate; 2] = [CounterRate::SWITCH_1, CounterRate::SWITCH_2];

    #[test]
    fn the_console_rates_are_the_references() {
        // hardware_properties.h: `CNTFREQ = 19'200'000`; guest_clock.h: `31'250'000U`.
        assert_eq!(CounterRate::SWITCH_1.hz(), 19_200_000);
        assert_eq!(CounterRate::SWITCH_2.hz(), 31_250_000);
    }

    #[test]
    fn a_rate_of_zero_cannot_be_built() {
        assert_eq!(CounterRate::new(0), None);
        assert_eq!(CounterRate::new(1).map(CounterRate::hz), Some(1));
        assert_eq!(
            CounterRate::new(CounterRate::SWITCH_2.hz()),
            Some(CounterRate::SWITCH_2)
        );
    }

    #[test]
    fn the_clocks_zero_is_tick_zero_at_every_rate() {
        for rate in RATES {
            assert_eq!(rate.ticks_at(HostNanos(0)), GuestTicks(0));
        }
    }

    #[test]
    fn whole_seconds_are_exactly_the_rate() {
        for rate in RATES {
            assert_eq!(
                rate.ticks_at(HostNanos(NANOS_PER_SECOND)),
                GuestTicks(rate.hz())
            );
            assert_eq!(
                rate.ticks_at(HostNanos(7 * NANOS_PER_SECOND)),
                GuestTicks(7 * rate.hz())
            );
        }
    }

    #[test]
    fn the_conversion_rounds_down_at_the_tick_boundary() {
        for rate in RATES {
            // The first reading that is a whole tick: one nanosecond earlier is still tick 0.
            let first_whole_tick = NANOS_PER_SECOND.div_ceil(rate.hz());
            assert_eq!(
                rate.ticks_at(HostNanos(first_whole_tick - 1)),
                GuestTicks(0)
            );
            assert_eq!(rate.ticks_at(HostNanos(first_whole_tick)), GuestTicks(1));
        }
    }

    #[test]
    fn the_largest_reading_converts_without_overflow() {
        for rate in RATES {
            // Checked against a split into whole seconds and the remainder, which stays in u64.
            let seconds = u64::MAX / NANOS_PER_SECOND;
            let rest = u64::MAX % NANOS_PER_SECOND;
            let expected = seconds * rate.hz() + rest * rate.hz() / NANOS_PER_SECOND;
            assert_eq!(rate.ticks_at(HostNanos(u64::MAX)), GuestTicks(expected));
        }
    }

    #[test]
    fn a_rate_above_a_tick_per_nanosecond_saturates_instead_of_wrapping() {
        // Exactly one tick per nanosecond is the last rate whose every reading fits.
        let fits = CounterRate::new(NANOS_PER_SECOND).unwrap();
        assert_eq!(fits.ticks_at(HostNanos(u64::MAX)), GuestTicks(u64::MAX));
        assert_eq!(
            fits.ticks_at(HostNanos(u64::MAX - 1)),
            GuestTicks(u64::MAX - 1)
        );
        // Two ticks per nanosecond: a reading of n is exactly 2n ticks, so the last reading that
        // fits is u64::MAX / 2 and the next one saturates.
        let double = CounterRate::new(2 * NANOS_PER_SECOND).unwrap();
        let last_fitting = u64::MAX / 2;
        assert_eq!(
            double.ticks_at(HostNanos(last_fitting)),
            GuestTicks(2 * last_fitting)
        );
        assert!(2 * last_fitting < u64::MAX);
        assert_eq!(
            double.ticks_at(HostNanos(last_fitting + 1)),
            GuestTicks(u64::MAX)
        );
        assert_eq!(double.ticks_at(HostNanos(u64::MAX)), GuestTicks(u64::MAX));
    }

    #[test]
    fn ticks_follow_the_process_rate_not_a_rescaled_base() {
        // PR #4588 counts at 19.2 MHz and rescales for a Switch 2 process:
        // `(t / CNTFREQ) * f + (t % CNTFREQ) * f / CNTFREQ` over the already rounded count `t`.
        // At 100 ns that is 1 tick at 19.2 MHz, rescaled to 1; the direct conversion is 3.
        let base = CounterRate::SWITCH_1;
        let native = CounterRate::SWITCH_2;
        let reading = HostNanos(100);
        let GuestTicks(counted) = base.ticks_at(reading);
        let rescaled =
            (counted / base.hz()) * native.hz() + (counted % base.hz()) * native.hz() / base.hz();
        assert_eq!((counted, rescaled), (1, 1));
        assert_eq!(native.ticks_at(reading), GuestTicks(3));
    }
}
