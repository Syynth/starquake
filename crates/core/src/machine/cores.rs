//! The cores as the scheduler reaches them (`starquake_kernel::sched::CoreSignals`): an
//! interrupt stops a core's vCPU, a wake ends a core's park, and stopping the run does both to
//! every core.
//!
//! A wake is a flag a core consumes when it parks, so a wake that comes before the core has
//! begun to wait is not lost. A core's interrupter is attached once its vCPU exists; an
//! interrupt before that reaches nothing, which is harmless, because the core's first step is
//! to ask the scheduler what to run.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};

use starquake_cpu::Interrupter;
use starquake_kernel::sched::{CoreId, CoreSignals};

use super::lock;

/// One core's slot.
struct Slot<I> {
    /// The vCPU's interrupter, once the vCPU exists.
    interrupter: Mutex<Option<I>>,
    /// Whether a wake is waiting for the core.
    woken: Mutex<bool>,
    wake: Condvar,
}

/// How a park ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Parked {
    /// The scheduler woke the core: it should ask what to run.
    Woken,
    /// The run is stopping.
    Stopped,
    /// Every core had parked and nothing could ever wake one: the caller's check said so.
    Stalled,
}

/// Every core of a run.
pub(super) struct Cores<I> {
    slots: Vec<Slot<I>>,
    stopped: AtomicBool,
    /// How many cores are parked.
    idle: Mutex<usize>,
}

impl<I: Interrupter> Cores<I> {
    /// `count` cores, none with a vCPU yet.
    pub(super) fn new(count: u8) -> Self {
        Self {
            slots: (0..count)
                .map(|_| Slot {
                    interrupter: Mutex::new(None),
                    woken: Mutex::new(false),
                    wake: Condvar::new(),
                })
                .collect(),
            stopped: AtomicBool::new(false),
            idle: Mutex::new(0),
        }
    }

    fn slot(&self, core: CoreId) -> Option<&Slot<I>> {
        self.slots.get(usize::from(core.index()))
    }

    /// Gives `core` its vCPU's interrupter.
    pub(super) fn attach(&self, core: CoreId, interrupter: I) {
        if let Some(slot) = self.slot(core) {
            *lock(&slot.interrupter) = Some(interrupter);
        }
    }

    /// Stops `core`'s vCPU, if it has one; its next run returns at once if it is not running.
    pub(super) fn interrupt(&self, core: CoreId) {
        let Some(slot) = self.slot(core) else {
            return;
        };
        if let Some(interrupter) = lock(&slot.interrupter).as_ref()
            && let Err(error) = interrupter.interrupt()
        {
            tracing::error!(core = core.index(), %error, "a core could not be interrupted");
        }
    }

    /// Ends `core`'s park, or the next one if it is not parked.
    pub(super) fn wake(&self, core: CoreId) {
        let Some(slot) = self.slot(core) else {
            return;
        };
        *lock(&slot.woken) = true;
        slot.wake.notify_all();
    }

    /// Stops every core: each is interrupted and woken, and parks no more.
    pub(super) fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        for index in 0..self.slots.len() {
            let Some(core) = u8::try_from(index).ok().and_then(CoreId::new) else {
                continue;
            };
            self.interrupt(core);
            self.wake(core);
        }
    }

    /// Parks `core` until it is woken or the run stops. If it is the last core to park,
    /// `stalled` is asked first whether anything could ever wake one, and the park ends at once
    /// if nothing could. `stalled` takes the kernel lock; nothing takes this type's counts
    /// under the kernel lock, so the order is always counts, then kernel.
    pub(super) fn park(&self, core: CoreId, stalled: impl FnOnce() -> bool) -> Parked {
        let Some(slot) = self.slot(core) else {
            return Parked::Stopped;
        };
        {
            let mut idle = lock(&self.idle);
            *idle = idle.saturating_add(1);
            if *idle == self.slots.len() && stalled() {
                *idle = idle.saturating_sub(1);
                return Parked::Stalled;
            }
        }
        let mut woken = lock(&slot.woken);
        while !*woken && !self.stopped.load(Ordering::SeqCst) {
            woken = slot
                .wake
                .wait(woken)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        *woken = false;
        drop(woken);
        let mut idle = lock(&self.idle);
        *idle = idle.saturating_sub(1);
        if self.stopped.load(Ordering::SeqCst) {
            Parked::Stopped
        } else {
            Parked::Woken
        }
    }
}

/// The scheduler's way to the cores.
pub(super) struct Signals<'a, I>(pub(super) &'a Cores<I>);

impl<I: Interrupter> CoreSignals for Signals<'_, I> {
    fn interrupt(&mut self, core: CoreId) {
        self.0.interrupt(core);
    }

    fn wake(&mut self, core: CoreId) {
        self.0.wake(core);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use starquake_cpu::Error;

    use super::*;

    /// Counts its interrupts.
    #[derive(Clone, Default)]
    struct Counting(Arc<Mutex<u32>>);

    impl Interrupter for Counting {
        fn interrupt(&self) -> Result<(), Error> {
            *lock(&self.0) += 1;
            Ok(())
        }
    }

    fn core(index: u8) -> CoreId {
        CoreId::new(index).unwrap()
    }

    #[test]
    fn a_wake_before_the_park_is_not_lost() {
        let cores: Cores<Counting> = Cores::new(2);
        cores.wake(core(1));
        assert_eq!(cores.park(core(1), || false), Parked::Woken);
    }

    #[test]
    fn a_park_waits_for_its_wake_from_another_thread() {
        let cores: Cores<Counting> = Cores::new(2);
        std::thread::scope(|scope| {
            let parked = scope.spawn(|| cores.park(core(0), || false));
            // Bounded: the parked thread is woken whatever this sleep finds.
            std::thread::sleep(Duration::from_millis(20));
            cores.wake(core(0));
            assert_eq!(parked.join().unwrap(), Parked::Woken);
        });
    }

    #[test]
    fn interrupts_reach_only_an_attached_vcpu_and_stopping_reaches_every_core() {
        let cores: Cores<Counting> = Cores::new(3);
        let count = Counting::default();
        // Before its vCPU exists an interrupt reaches nothing.
        cores.interrupt(core(1));
        cores.attach(core(1), count.clone());
        cores.interrupt(core(1));
        assert_eq!(*lock(&count.0), 1);
        cores.stop();
        assert_eq!(*lock(&count.0), 2);
        // A stopped run parks no more.
        assert_eq!(cores.park(core(2), || false), Parked::Stopped);
    }

    #[test]
    fn only_the_last_core_to_park_asks_whether_the_run_is_stalled() {
        let cores: Cores<Counting> = Cores::new(2);
        std::thread::scope(|scope| {
            let first = scope.spawn(|| cores.park(core(0), || unreachable!()));
            // The second core is the last: it asks, and is told the run is stalled.
            let mut asked = false;
            // Bounded: the first core parks within this wait or the assertion below fails.
            for _ in 0..200 {
                if *lock(&cores.idle) == 1 {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            assert_eq!(*lock(&cores.idle), 1);
            let parked = cores.park(core(1), || {
                asked = true;
                true
            });
            assert_eq!(parked, Parked::Stalled);
            assert!(asked);
            // Not stalled: the last core parks like any other.
            cores.wake(core(1));
            assert_eq!(cores.park(core(1), || false), Parked::Woken);
            cores.stop();
            assert_eq!(first.join().unwrap(), Parked::Stopped);
        });
    }
}
