//! `GetSystemTick` (0x1e).
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_tick.cpp` @ cf654052 (GPL-2.0-or-later): the
//! call returns the guest counter (`CoreTiming().GetClockTicks()`), as `int64_t`, in `x0`. Here
//! the reading comes from the injectable [`Env`] (observability spec, rule 1), so a replay sees
//! the same ticks.
//!
//! Guess: the kernel's [`Env`] reads a clock that already counts in guest-counter ticks, so the
//! reading is returned as it is. `starquake-trace` leaves the unit to the clock behind the `Env`
//! and says converting is the consumer's job; where that conversion lives is not decided yet.
//!
//! Divergence from the kernel spec's sketch (`fn get_system_tick(env: &dyn Env) -> u64`): the
//! result is fallible, because [`Env::now`] is. A replay that has run out of clock readings, or
//! whose next record isn't one, returns its [`EnvError`] here instead of a made-up tick.

use starquake_trace::{Env, EnvError, GuestTicks};

/// Returns the guest's counter, read from `env`.
pub fn get_system_tick(env: &dyn Env) -> Result<u64, EnvError> {
    env.now().map(|GuestTicks(ticks)| ticks)
}

#[cfg(test)]
mod tests {
    use starquake_trace::{EnvCall, EnvRecord, FixedClock, LiveEnv, ReplayEnv};

    use super::*;

    #[test]
    fn returns_the_environments_clock_reading() {
        let env = LiveEnv::new(FixedClock::new(0x1234_5678_9abc_def0, 1), 0);
        assert_eq!(get_system_tick(&env), Ok(0x1234_5678_9abc_def0));
        assert_eq!(get_system_tick(&env), Ok(0x1234_5678_9abc_def1));
        assert_eq!(
            env.recording(),
            [
                EnvRecord::Clock(GuestTicks(0x1234_5678_9abc_def0)),
                EnvRecord::Clock(GuestTicks(0x1234_5678_9abc_def1))
            ]
        );
    }

    #[test]
    fn a_replay_feeds_back_the_recorded_ticks() {
        let env = ReplayEnv::new(vec![EnvRecord::Clock(GuestTicks(u64::MAX))]);
        assert_eq!(get_system_tick(&env), Ok(u64::MAX));
        assert_eq!(
            get_system_tick(&env),
            Err(EnvError::Exhausted {
                index: 1,
                requested: EnvCall::Clock
            })
        );
    }
}
