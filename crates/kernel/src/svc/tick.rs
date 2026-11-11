//! `GetSystemTick` (0x1e).
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_tick.cpp` @ cf654052 (GPL-2.0-or-later): the
//! call returns the guest counter (`CoreTiming().GetClockTicks()`), as `int64_t`, in `x0`. Here
//! the reading comes from the injectable [`Env`] (observability spec, rule 1), so a replay sees
//! the same ticks.
//!
//! **Unit.** [`Env::now`] reports whatever unit its clock uses: `SystemClock` reports host
//! nanoseconds (`starquake-trace`, `env.rs`). The guest counter (`CNTPCT_EL0`) runs at
//! [`CNTFREQ`] = 19.2 MHz (`eden/src/core/hardware_properties.h`, `CNTFREQ`), so this module
//! converts: it takes the [`Env`] reading as nanoseconds and returns
//! `nanoseconds * CNTFREQ / NANOS_PER_SECOND`, rounded down, in 128-bit arithmetic. A clock
//! behind the [`Env`] that does not count nanoseconds (the trace crate's `FixedClock` counts
//! whatever its caller chose) therefore gets its readings scaled as if they were. The recording
//! keeps the raw reading, so a replay converts the same way and returns the same ticks.
//!
//! Guess: the counter's zero. Eden's counter starts at emulator boot; here it starts where the
//! [`Env`]'s clock does, which for `SystemClock` is the host's own epoch. Nothing in the kernel
//! spec or the scoping doc fixes it; reading how Eden's `CoreTiming` anchors `GetClockTicks` would
//! confirm or correct it.
//!
//! Divergence from the kernel spec's sketch (`fn get_system_tick(env: &dyn Env) -> u64`): the
//! result is fallible, because [`Env::now`] is. A replay that has run out of clock readings, or
//! whose next record isn't one, returns its [`EnvError`] here instead of a made-up tick.

use starquake_trace::{Env, EnvError, GuestTicks};

/// The guest counter's frequency in hertz: Eden's `Hardware::CNTFREQ`
/// (`eden/src/core/hardware_properties.h`, "`CNTPCT_EL0` Frequency = 19.2 MHz").
pub const CNTFREQ: u64 = 19_200_000;

/// Nanoseconds in one second, the unit [`Env::now`] is taken to report.
pub const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// Converts a nanosecond reading to guest counter ticks, rounding down.
///
/// The product is formed in `u128`, so it cannot overflow for any `u64` input. The quotient is at
/// most `u64::MAX * CNTFREQ / NANOS_PER_SECOND`, which is below `u64::MAX` while
/// `CNTFREQ < NANOS_PER_SECOND`; the saturation below is therefore never reached.
#[must_use]
pub fn nanos_to_ticks(nanos: u64) -> u64 {
    let ticks = u128::from(nanos) * u128::from(CNTFREQ) / u128::from(NANOS_PER_SECOND);
    u64::try_from(ticks).unwrap_or(u64::MAX)
}

/// Returns the guest's counter, read from `env` (nanoseconds) and converted to ticks.
pub fn get_system_tick(env: &dyn Env) -> Result<u64, EnvError> {
    env.now().map(|GuestTicks(nanos)| nanos_to_ticks(nanos))
}

#[cfg(test)]
mod tests {
    use starquake_trace::{EnvCall, EnvRecord, FixedClock, LiveEnv, ReplayEnv};

    use super::*;

    /// A reading that is not a whole number of seconds, so the rounding is exercised.
    const READING_NS: u64 = 0x1234_5678_9abc_def0;

    #[test]
    fn one_second_of_nanoseconds_is_exactly_cntfreq_ticks() {
        assert_eq!(nanos_to_ticks(NANOS_PER_SECOND), CNTFREQ);
        assert_eq!(nanos_to_ticks(0), 0);
        assert_eq!(nanos_to_ticks(7 * NANOS_PER_SECOND), 7 * CNTFREQ);
    }

    #[test]
    fn the_conversion_rounds_down_at_the_tick_boundary() {
        // One tick lasts NANOS_PER_SECOND / CNTFREQ = 52.08.. ns: 52 ns is still zero ticks and
        // 53 ns is the first whole one (19.2 MHz: 53 * 19_200_000 / 1e9 = 1.0176).
        let first_whole_tick = NANOS_PER_SECOND.div_ceil(CNTFREQ);
        assert_eq!(nanos_to_ticks(first_whole_tick - 1), 0);
        assert_eq!(nanos_to_ticks(first_whole_tick), 1);
    }

    #[test]
    fn the_largest_reading_converts_without_overflow() {
        // Checked against a split into whole seconds and the remainder, which stays in u64.
        let seconds = u64::MAX / NANOS_PER_SECOND;
        let rest = u64::MAX % NANOS_PER_SECOND;
        let expected = seconds * CNTFREQ + rest * CNTFREQ / NANOS_PER_SECOND;
        assert_eq!(nanos_to_ticks(u64::MAX), expected);
        assert!(nanos_to_ticks(u64::MAX) < u64::MAX);
    }

    #[test]
    fn the_guest_sees_nanoseconds_scaled_to_the_counter_rate_not_raw() {
        let env = LiveEnv::new(FixedClock::new(READING_NS, NANOS_PER_SECOND), 0);
        let first = get_system_tick(&env);
        let second = get_system_tick(&env);
        assert_eq!(first, Ok(nanos_to_ticks(READING_NS)));
        assert_ne!(first, Ok(READING_NS));
        // A clock one second later reads CNTFREQ ticks later, give or take the rounding.
        let (first, second) = (first.unwrap(), second.unwrap());
        assert!((CNTFREQ..=CNTFREQ + 1).contains(&(second - first)));
        // The recording keeps the clock's own reading, so a replay converts the same way.
        assert_eq!(
            env.recording(),
            [
                EnvRecord::Clock(GuestTicks(READING_NS)),
                EnvRecord::Clock(GuestTicks(READING_NS + NANOS_PER_SECOND))
            ]
        );
    }

    #[test]
    fn a_replay_feeds_back_the_recorded_readings_converted() {
        let env = ReplayEnv::new(vec![EnvRecord::Clock(GuestTicks(u64::MAX))]);
        assert_eq!(get_system_tick(&env), Ok(nanos_to_ticks(u64::MAX)));
        assert_eq!(
            get_system_tick(&env),
            Err(EnvError::Exhausted {
                index: 1,
                requested: EnvCall::Clock
            })
        );
    }
}
