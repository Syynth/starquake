//! `GetSystemTick` (0x1e).
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_tick.cpp` @ cf654052 (GPL-2.0-or-later): the
//! call returns the guest counter (`CoreTiming().GetClockTicks()`), as `int64_t`, in `x0`. PR
//! #4588 returns it at the calling process's counter rate
//! (`eden-pr4588/src/core/hle/kernel/svc/svc_tick.cpp` @ e5fb3b97, `GetGuestCounterTicks`).
//!
//! Here the counter is the kernel's guest time ([`crate::time`]): the dispatcher reads the
//! environment's clock once per dispatch (observability spec, rule 1, so a replay sees the same
//! ticks), converts it with the process's counter rate, and passes the result in. The zero
//! point, and why a counter read through a patched instruction agrees with it, are in
//! [`crate::time`]'s docs.
//!
//! Divergence from the kernel spec's sketch (`fn get_system_tick(env: &dyn Env) -> u64`): the
//! call takes the dispatch's guest time, not the environment. The clock can fail (a replay that
//! has run out of readings, or whose next record isn't one), and that failure belongs to the
//! dispatcher, which stops the run; a system call never sees it.

use starquake_trace::GuestTicks;

/// Returns the guest's counter: the guest time of the dispatch the call was made in.
#[must_use]
pub fn get_system_tick(now: GuestTicks) -> u64 {
    now.0
}

#[cfg(test)]
mod tests {
    use starquake_trace::HostNanos;

    use super::*;
    use crate::time::{CounterRate, NANOS_PER_SECOND};

    #[test]
    fn the_tick_is_the_guest_time_it_is_given_in_full() {
        // A full 64-bit value (`int64_t`), not a 32-bit word.
        for ticks in [0, 1, u64::from(u32::MAX) + 1, u64::MAX] {
            assert_eq!(get_system_tick(GuestTicks(ticks)), ticks);
        }
    }

    #[test]
    fn a_second_of_host_time_is_one_second_of_the_process_counter() {
        for rate in [CounterRate::SWITCH_1, CounterRate::SWITCH_2] {
            let now = rate.ticks_at(HostNanos(NANOS_PER_SECOND));
            assert_eq!(get_system_tick(now), rate.hz());
        }
    }
}
