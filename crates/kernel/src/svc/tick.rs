//! `GetSystemTick` (0x1e).
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_tick.cpp` @ cf654052 (GPL-2.0-or-later): the
//! call returns the counter value, as `int64_t`, in `x0`. Here the counter comes from the [`Env`],
//! so a replay sees the same ticks.

use super::Env;

/// Returns the guest's counter. The dispatcher writes it to `x0` as Eden's `int64_t` return.
#[must_use]
pub fn get_system_tick(env: &dyn Env) -> u64 {
    env.system_tick()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(u64);

    impl Env for Fixed {
        fn system_tick(&self) -> u64 {
            self.0
        }
    }

    #[test]
    fn returns_the_environments_counter() {
        assert_eq!(
            get_system_tick(&Fixed(0x1234_5678_9abc_def0)),
            0x1234_5678_9abc_def0
        );
        assert_eq!(get_system_tick(&Fixed(u64::MAX)), u64::MAX);
    }
}
