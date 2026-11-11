//! The injectable environment: the only way kernel code reads anything nondeterministic.
//!
//! The observability spec ("Replaying the kernel and services", rule 1) routes the clock, random
//! numbers and host I/O through one `Env` trait so a run can be replayed. This is the part the
//! kernel needs so far, the clock. The full trait belongs with the event stream in
//! `starquake-trace`; when it lands there, this one is replaced by it.

/// What the kernel reads from outside the emulated machine.
pub trait Env {
    /// The guest's counter (`CNTPCT_EL0`), in ticks of the guest's counter frequency. Eden's
    /// `GetSystemTick` returns `CoreTiming().GetClockTicks()` (`eden/src/core/hle/kernel/svc/svc_tick.cpp`
    /// @ cf654052).
    fn system_tick(&self) -> u64;
}
