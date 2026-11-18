//! A run's time: the console its process was built for, which decides its counter rate, and
//! the host clock its guest time is counted from.
//!
//! The kernel owns guest time (`docs/specs/kernel.md`, "Guest time"): it converts the
//! environment's clock, host nanoseconds, with the process's counter rate, and takes the
//! clock's zero as guest tick zero. Which rate a process gets is the kernel's choice too, made
//! from the console in its description (`starquake_kernel::process::Console::counter_rate`).
//! This module is the other half of that contract. The run takes the console from the
//! program's description (`crate::image::Program::description`), hands that console's rate to
//! the loader's counter patcher, and builds the environment's clock from the counter the
//! patched instructions read, so that `GetSystemTick`, a patched counter read and a patched
//! read of the counter's frequency agree on the rate and on where the counter started.
//! `tests/skeleton.rs` reads a tick each way and checks that they do; the core loop refuses a
//! run whose clock and process name different consoles (`crate::machine`).
//!
//! A counter read the patcher missed is the backend's to answer. Under the hypervisor backend
//! an unpatched `mrs xN, cntpct_el0` traps to its shim, which rescales it with the scale each
//! vCPU is given ([`shim_scale`]: the patcher's own ratio, handed over unchanged), so that it
//! reads what a patched read would. The run makes the patcher's rates and that scale together,
//! as one [`CounterScaling`] ([`Timebase::counter_scaling`]). Unpatched `cntvct_el0` and `cntfrq_el0` reads still see the
//! host's rate: a divergence the backend lists (`starquake_cpu::hvf`, "Counter reads").
//!
//! An ExeFS's console is its metadata's (`crate::image`, "The process's description"). An NRO
//! carries nothing that says which console it was built for, so it is described as built for
//! [`NRO_CONSOLE`]: the homebrew this emulator runs is built for the first Switch (a guess, and
//! the rate the skeleton ran at before the kernel kept guest time).

use std::num::NonZeroU64;

use starquake_kernel::process::Console;
use starquake_kernel::time::{CounterRate, NANOS_PER_SECOND};
use starquake_trace::HostNanos;

/// The console an NRO is described as built for. See the module docs.
pub(crate) const NRO_CONSOLE: Console = Console::Switch1;

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

    /// What the loader's patcher rescales the program's counter reads with: from the backend's
    /// counter, which runs at `host_hz` hertz, to the process's [`Timebase::rate`]. Only a CPU
    /// backend's run patches counter reads.
    #[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
    pub(crate) const fn counter_rescale(
        &self,
        host_hz: u64,
    ) -> starquake_loader::patch::CounterRescale {
        starquake_loader::patch::CounterRescale {
            host_hz,
            guest_hz: self.rate().hz(),
        }
    }

    /// Both halves of a run's counter rescaling under the hypervisor backend, made from the one
    /// pair of rates [`Timebase::counter_rescale`] gives for `host_hz`: what the patcher
    /// rescales with, and the scale each vCPU rescales the reads the patcher missed with
    /// ([`shim_scale`]). Refused, as the patcher refuses it, for a rate of zero.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn counter_scaling(
        &self,
        host_hz: u64,
    ) -> Result<CounterScaling, starquake_loader::patch::PatchError> {
        let patcher = self.counter_rescale(host_hz);
        Ok(CounterScaling {
            patcher,
            shim: shim_scale(patcher)?,
        })
    }
}

/// A run's counter rescaling under the hypervisor backend: the rates the loader's patcher
/// rescales patched reads with, and the scale the backend rescales the reads the patcher missed
/// with. Only [`Timebase::counter_scaling`] makes one, from one pair of rates, so the two halves
/// cannot disagree; the run hands this one value to the patcher and to every vCPU.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CounterScaling {
    /// The patcher's rates.
    patcher: starquake_loader::patch::CounterRescale,
    /// [`shim_scale`] of `patcher`.
    shim: starquake_cpu::hvf::CounterScale,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl CounterScaling {
    /// The rates the loader's patcher rescales patched reads with.
    pub(crate) const fn patcher(&self) -> starquake_loader::patch::CounterRescale {
        self.patcher
    }

    /// The scale a vCPU rescales the reads the patcher missed with: the patcher's own ratio.
    pub(crate) const fn shim(&self) -> starquake_cpu::hvf::CounterScale {
        self.shim
    }
}

/// The scale the hypervisor backend rescales a counter read the patcher missed with: the
/// patcher's ratio for `rescale` (`CounterRescale::ratio`), handed over unchanged, so that an
/// unpatched read and a patched one taken at the same count read the same value
/// (`starquake_cpu::hvf`, "Counter reads"). Refused, as the patcher refuses it, for a rate of
/// zero.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn shim_scale(
    rescale: starquake_loader::patch::CounterRescale,
) -> Result<starquake_cpu::hvf::CounterScale, starquake_loader::patch::PatchError> {
    let ratio = rescale.ratio()?;
    Ok(starquake_cpu::hvf::CounterScale::new(
        ratio.int(),
        ratio.frac(),
    ))
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

    #[test]
    fn the_patchers_rates_are_the_backends_and_the_processs() {
        // Each console's rate is the kernel's (`Console::counter_rate`), and the host's is the
        // one given: a distinct value for each, so swapped fields fail.
        for console in [Console::Switch1, Console::Switch2] {
            let time = Timebase { console, clock: () };
            assert_eq!(
                time.counter_rescale(HOST_HZ.get()),
                CounterRescale {
                    host_hz: HOST_HZ.get(),
                    guest_hz: console.counter_rate().hz(),
                }
            );
        }
        assert_ne!(
            Console::Switch1.counter_rate(),
            Console::Switch2.counter_rate()
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn the_shims_scale_is_the_patchers_ratio_and_refuses_a_rate_of_zero() {
        for guest_hz in [CounterRate::SWITCH_1.hz(), CounterRate::SWITCH_2.hz()] {
            let rescale = CounterRescale {
                host_hz: HOST_HZ.get(),
                guest_hz,
            };
            let ratio = rescale.ratio().unwrap();
            let scale = shim_scale(rescale).unwrap();
            assert_eq!((scale.int(), scale.frac()), (ratio.int(), ratio.frac()));
        }
        let zero = CounterRescale {
            host_hz: 0,
            guest_hz: CounterRate::SWITCH_2.hz(),
        };
        assert_eq!(shim_scale(zero), Err(zero.ratio().unwrap_err()));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn a_runs_patcher_rates_and_shim_scale_are_made_as_one_pair() {
        // A host rate distinct from each console's, so a pair whose scale was made from other
        // rates (the guest's rate given as the host's, ratio 1, or the rates swapped) fails.
        for console in [Console::Switch1, Console::Switch2] {
            assert_ne!(HOST_HZ.get(), console.counter_rate().hz());
            let time = Timebase { console, clock: () };
            let counter = time.counter_scaling(HOST_HZ.get()).unwrap();
            let rescale = CounterRescale {
                host_hz: HOST_HZ.get(),
                guest_hz: console.counter_rate().hz(),
            };
            assert_eq!(counter.patcher(), rescale, "{console:?}");
            let ratio = rescale.ratio().unwrap();
            assert_eq!(
                (counter.shim().int(), counter.shim().frac()),
                (ratio.int(), ratio.frac()),
                "{console:?}"
            );
            assert_ne!(
                counter.shim(),
                starquake_cpu::hvf::CounterScale::IDENTITY,
                "{console:?}"
            );
        }
        // Refused for a rate of zero, as the patcher refuses it.
        let time = Timebase {
            console: Console::Switch2,
            clock: (),
        };
        assert_eq!(
            time.counter_scaling(0),
            Err(time.counter_rescale(0).ratio().unwrap_err())
        );
    }

    /// The vCPUs of a run under the hypervisor backend: what `crate::system` makes them with.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    mod on_hardware {
        use std::sync::PoisonError;

        use starquake_cpu::hvf::{HvfBackend, PhysicalMemory, RamConfig};
        use starquake_cpu::{Exit, GuestRegs, Vcpu, host_counter};
        use starquake_loader::patch::{PatchConfig, SiteKind, patch_text};
        use starquake_memory::{GuestMemory, PAGE_SIZE, Perms};
        use starquake_testkit::a64::{self, SysReg};

        use super::*;
        use crate::space::AddressSpace;
        use crate::system::host::{ONE_VM, process_vcpu};

        /// Where the test's code goes, and its trampolines a page above.
        const TEXT: u64 = 0x80_0000;
        const TRAMPOLINES: u64 = TEXT + PAGE_SIZE;
        /// A page of stack, for the trampolines to save their scratch registers on, two pages
        /// above the trampolines.
        const STACK: u64 = TRAMPOLINES + 2 * PAGE_SIZE;

        /// Patched reads into x19 and x21, each followed by an unpatched one (x20, x22) written
        /// after the patcher ran, as code written at run time would be, then `svc #0`. Returns
        /// the text and the trampolines.
        fn back_to_back(rescale: CounterRescale) -> (Vec<u8>, Vec<u8>) {
            let read = |rt| a64::mrs(rt, SysReg::CntPctEl0).unwrap();
            let words = [
                read(19),
                a64::nop(),
                read(21),
                a64::nop(),
                a64::svc(0).unwrap(),
            ];
            let mut text = a64::to_bytes(&words);
            let config = PatchConfig {
                counter: Some(rescale),
            };
            let patched = patch_text(&mut text, TEXT, TRAMPOLINES, &config).unwrap();
            let sites: Vec<_> = patched
                .sites
                .iter()
                .map(|site| (site.kind, site.register))
                .collect();
            assert_eq!(sites, [(SiteKind::CntPct, 19), (SiteKind::CntPct, 21)]);
            let insn = usize::try_from(a64::INSN_SIZE).unwrap();
            for (index, rt) in [(1, 20), (3, 22)] {
                text[index * insn..(index + 1) * insn].copy_from_slice(&read(rt).to_le_bytes());
            }
            (text, patched.trampolines)
        }

        #[test]
        fn a_runs_vcpu_answers_an_unpatched_read_as_the_patched_reads_around_it() {
            if std::env::var_os("CI").is_some() {
                return;
            }
            let _one_vm = ONE_VM.lock().unwrap_or_else(PoisonError::into_inner);
            let host_hz = HvfBackend::host_counter_frequency();
            let backend = HvfBackend::new(RamConfig::default()).unwrap();
            let space = AddressSpace::<PhysicalMemory>::new(
                backend.physical(),
                backend.ram(),
                backend.table_pool(),
            )
            .unwrap();
            space.map(TEXT, 2 * PAGE_SIZE, Perms::ReadExecute).unwrap();
            space.map(STACK, PAGE_SIZE, Perms::ReadWrite).unwrap();
            // Written once, so the page has backing before the guest stores to it.
            let page = vec![0; usize::try_from(PAGE_SIZE).unwrap()];
            space.write(STACK, &page).unwrap();
            // One console whose rate is below the host's and one above it: a vCPU that read the
            // host's counter unscaled would fall out of order either way.
            for console in [Console::Switch1, Console::Switch2] {
                let time = Timebase {
                    console,
                    clock: CounterClock::new(host_hz).unwrap(),
                };
                let counter = time.counter_scaling(host_hz).unwrap();
                let rescale = counter.patcher();
                let (text, trampolines) = back_to_back(rescale);
                space.write(TEXT, &text).unwrap();
                space.write(TRAMPOLINES, &trampolines).unwrap();
                // Made as the run makes every core's vCPU.
                let mut vcpu = process_vcpu(&backend, space.root(), counter).unwrap();
                vcpu.set_regs(&GuestRegs {
                    pc: TEXT,
                    sp: STACK + PAGE_SIZE,
                    ..GuestRegs::default()
                })
                .unwrap();
                let before = host_counter::read();
                assert_eq!(vcpu.run(), Ok(Exit::Svc(0)), "{console:?}");
                let after = host_counter::read();
                let x = vcpu.regs().unwrap().x;
                let reads = [x[19], x[20], x[21], x[22]];
                // One clock: each read at or after the one before, all four between the host's
                // readings around the run, scaled as the patcher scales.
                let ratio = rescale.ratio().unwrap();
                assert!(reads.is_sorted(), "{console:?}: {reads:?}");
                assert!(
                    ratio.scale(before) <= reads[0] && reads[3] <= ratio.scale(after),
                    "{console:?}: {reads:?} not in {}..={}",
                    ratio.scale(before),
                    ratio.scale(after)
                );
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
