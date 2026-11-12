//! The counter guest code reads under the hypervisor backend, read from the host.
//!
//! An EL0 `mrs xN, cntvct_el0` runs natively in the VM, and Hypervisor.framework can offset the
//! guest's view of the counter but not scale it (`tools/hvprobe/README.md`, "`CNTFRQ_EL0` /
//! counter rate"). The backend sets no offset of its own, and what the guest then reads is the
//! host's `mach_absolute_time`: the same ticks, at the rate
//! [`HvfBackend::counter_frequency`](crate::hvf::HvfBackend::counter_frequency) reports, from the
//! same zero. It is **not** the host's own `CNTVCT_EL0`: in the one measurement made, a host
//! `mrs xN, cntvct_el0` read about 20 days of ticks more than the guest's (a guess at why: the
//! time the machine had spent asleep, which `mach_absolute_time` leaves out). All of this is
//! measured, not documented by Apple. `tests/host_counter.rs` pins it by bracketing a guest
//! read between two host reads, which fails if the two differ by more than the time one entry
//! to the guest takes.
//!
//! That is what lets guest time have one zero point: `starquake-core` builds the environment's
//! clock from [`read`], so the kernel's `GetSystemTick` and a counter read the loader's patcher
//! rewrote (which reads `CNTVCT_EL0` in the guest and rescales it) count from the same instant.

// libSystem is linked into every Rust binary on macOS, so no `#[link]` attribute is needed.
unsafe extern "C" {
    /// `mach_absolute_time` (`<mach/mach_time.h>`): the host's monotonic tick count.
    fn mach_absolute_time() -> u64;
}

/// The counter as a guest's `CNTVCT_EL0` reads it now: the host's `mach_absolute_time`, in ticks
/// of [`HvfBackend::counter_frequency`](crate::hvf::HvfBackend::counter_frequency).
#[must_use]
pub fn read() -> u64 {
    // SAFETY: `mach_absolute_time` takes no arguments, touches no caller memory and cannot fail.
    unsafe { mach_absolute_time() }
}
