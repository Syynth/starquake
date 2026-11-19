//! The cores as the scheduler reaches them (`starquake_kernel::sched::CoreSignals`): an
//! interrupt stops a core's vCPU, a wake ends a core's park, and stopping the run does both to
//! every core.
//!
//! A wake is a flag a core consumes when it parks, so a wake that comes before the core has
//! begun to wait is not lost. A core's interrupter is attached once its vCPU exists; an
//! interrupt before that reaches nothing, which is harmless, because the core's first step is
//! to ask the scheduler what to run.
//!
//! # Telling a stalled run
//!
//! The last core to park asks whether the run is stalled ([`Cores::park`]); the core loop's
//! question ([`stall`]) asks the graphics server and then the kernel. The answer is one of
//! [`Stall`]'s three:
//!
//! - **Stalled**: the park ends at once ([`Parked::Stalled`]) and the run stops as `WAITING`.
//! - **Not stalled**: the core waits for a wake, with no timeout.
//! - **Only the GPU**: the kernel says nothing could end a wait and the rest of the graphics
//!   server is idle, but the GPU answered busy. An idle GPU reads busy during each turn its
//!   front takes, the `IDLE_PARK` safety-net turns included, so a question that lands in one
//!   would never be asked again (PR #422's review). The core then also waits for the GPU's next
//!   quiet signal (`starquake_gpu::wiring::Gpu::set_quiet_wake`, which [`Cores::listen_to`]
//!   wires) and asks again when it comes, up to [`GPU_QUIET_RETRIES`] times in one park; past
//!   that it waits as for "not stalled" (gpu-protocol, "Idleness"; observability, "Stalls and
//!   host timing"; decision log, 2026-11-19, ruling 37).
//!
//! The core arms itself for the quiet signal before it asks, so a signal that comes while the
//! question is being asked is not lost
//! (`a_quiet_signal_during_the_question_is_not_lost`). One signal is taken by one arming: the
//! core asks once per signal. It asks again only while every core is still parked: a core that
//! another core's wake has unparked meanwhile will park again and be the one that asks
//! (`a_quiet_signal_while_another_core_runs_asks_nothing`). At most one core is armed, the last
//! one to have asked: a later question by another core moves the arming to it.
//!
//! No clock is read for any of this: the wait for the quiet signal has no timeout, and the
//! retry bound counts questions.
//!
//! Lock order: the idle count, then the kernel (the question) or the arming, then a core's park
//! state. The GPU's front thread takes the arming, then a park state; the question waits on the
//! GPU with only the idle count held.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use starquake_cpu::Interrupter;
use starquake_gpu::front::{QuietWake, QuietWakeTaken};
use starquake_kernel::sched::{CoreId, CoreSignals};

use super::lock;
use crate::hle::Graphics;
use crate::hle::graphics::Idleness;

/// How many times one park of the last core asks again whether the run is stalled, each time
/// after the GPU's next quiet signal, when the answers were busy only because of the GPU (module
/// docs, "Telling a stalled run"). Starquake's own number, a **guess** (gpu-protocol asks only
/// for "a named retry count"): an idle GPU reads busy for a few microseconds in each 100 ms
/// safety-net turn, so a stalled run whose questions land in eight turns in a row is very
/// unlikely, while a GPU that stays busy is asked no more than nine times per park.
pub(super) const GPU_QUIET_RETRIES: u32 = 8;

/// What the last core to park was told (module docs, "Telling a stalled run").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stall {
    /// Nothing could ever wake a core.
    Stalled,
    /// Something can still wake a core.
    NotStalled,
    /// Stalled but for the GPU's busy answer: asked again at the GPU's next quiet signal.
    OnlyTheGpu,
}

/// The core loop's question, from the graphics server's `graphics` answer and the kernel's
/// (`kernel_stalled`, asked only when the graphics server is idle or busy only because of the
/// GPU, as before the GPU's answer could be told apart).
pub(super) fn stall(graphics: Idleness, kernel_stalled: impl FnOnce() -> bool) -> Stall {
    let stalled = match graphics {
        Idleness::Busy => return Stall::NotStalled,
        Idleness::Idle => Stall::Stalled,
        Idleness::GpuBusy => Stall::OnlyTheGpu,
    };
    if kernel_stalled() {
        stalled
    } else {
        Stall::NotStalled
    }
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

/// What a parked core waits for.
#[derive(Default)]
struct ParkState {
    /// A wake is waiting for the core.
    woken: bool,
    /// The GPU reported quiet while the core was armed for it.
    quiet: bool,
}

/// One core's park.
#[derive(Default)]
struct ParkSlot {
    state: Mutex<ParkState>,
    wake: Condvar,
}

/// What a park waits on, shared with the GPU's front thread, which tells it the quiet signal.
struct Parks {
    slots: Vec<ParkSlot>,
    stopped: AtomicBool,
    /// The core armed for the GPU's next quiet signal, by index, if one is.
    armed: Mutex<Option<usize>>,
}

impl Parks {
    /// Arms core `index` for the next quiet signal, forgetting any signal it was told before.
    fn arm(&self, index: usize) {
        let mut armed = lock(&self.armed);
        *armed = Some(index);
        if let Some(slot) = self.slots.get(index) {
            lock(&slot.state).quiet = false;
        }
    }

    /// Disarms core `index`, if it is the one armed.
    fn disarm(&self, index: usize) {
        let mut armed = lock(&self.armed);
        if *armed == Some(index) {
            *armed = None;
        }
    }
}

impl QuietWake for Parks {
    /// Tells the armed core, if any, and disarms it.
    fn front_quiet(&self) {
        let mut armed = lock(&self.armed);
        let Some(index) = armed.take() else {
            return;
        };
        if let Some(slot) = self.slots.get(index) {
            lock(&slot.state).quiet = true;
            slot.wake.notify_all();
        }
    }
}

/// Every core of a run.
pub(super) struct Cores<I> {
    /// Each core's vCPU interrupter, once the vCPU exists.
    interrupters: Vec<Mutex<Option<I>>>,
    parks: Arc<Parks>,
    /// How many cores are parked.
    idle: Mutex<usize>,
    /// How many of the next questions that come to "stalled" are read as busy only because of
    /// the GPU: a test's way to make a question land in a GPU turn.
    #[cfg(test)]
    gpu_busy_reads: std::sync::atomic::AtomicU32,
}

impl<I: Interrupter> Cores<I> {
    /// `count` cores, none with a vCPU yet.
    pub(super) fn new(count: u8) -> Self {
        Self {
            interrupters: (0..count).map(|_| Mutex::new(None)).collect(),
            parks: Arc::new(Parks {
                slots: (0..count).map(|_| ParkSlot::default()).collect(),
                stopped: AtomicBool::new(false),
                armed: Mutex::new(None),
            }),
            idle: Mutex::new(0),
            #[cfg(test)]
            gpu_busy_reads: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// How many cores there are.
    pub(super) fn count(&self) -> u8 {
        u8::try_from(self.interrupters.len()).unwrap_or(u8::MAX)
    }

    /// Gives `graphics`'s GPU, if it has one, the cores as its quiet wake (module docs,
    /// "Telling a stalled run").
    ///
    /// # Errors
    ///
    /// [`QuietWakeTaken`] if the GPU already has a quiet wake; a run's GPU is made fresh for
    /// the run, so it never has.
    pub(super) fn listen_to(&self, graphics: &Graphics) -> Result<(), QuietWakeTaken> {
        match graphics.gpu() {
            Some(gpu) => gpu.set_quiet_wake(Arc::<Parks>::clone(&self.parks)),
            None => Ok(()),
        }
    }

    /// Gives `core` its vCPU's interrupter.
    pub(super) fn attach(&self, core: CoreId, interrupter: I) {
        if let Some(slot) = self.interrupters.get(usize::from(core.index())) {
            *lock(slot) = Some(interrupter);
        }
    }

    /// Stops `core`'s vCPU, if it has one; its next run returns at once if it is not running.
    pub(super) fn interrupt(&self, core: CoreId) {
        let Some(slot) = self.interrupters.get(usize::from(core.index())) else {
            return;
        };
        if let Some(interrupter) = lock(slot).as_ref()
            && let Err(error) = interrupter.interrupt()
        {
            tracing::error!(core = core.index(), %error, "a core could not be interrupted");
        }
    }

    /// Ends `core`'s park, or the next one if it is not parked.
    pub(super) fn wake(&self, core: CoreId) {
        let Some(slot) = self.parks.slots.get(usize::from(core.index())) else {
            return;
        };
        lock(&slot.state).woken = true;
        slot.wake.notify_all();
    }

    /// Takes `core`'s waiting wake without parking: whether there was one. What a test that
    /// steps every core on one thread does instead of parking.
    #[cfg(test)]
    pub(super) fn take_wake(&self, core: CoreId) -> bool {
        self.parks
            .slots
            .get(usize::from(core.index()))
            .is_some_and(|slot| std::mem::take(&mut lock(&slot.state).woken))
    }

    /// Makes the next `count` questions that come to "stalled" read as busy only because of the
    /// GPU, as if each had landed in a GPU turn.
    #[cfg(test)]
    pub(super) fn read_gpu_busy(&self, count: u32) {
        self.gpu_busy_reads.store(count, Ordering::SeqCst);
    }

    /// Stops every core: each is interrupted and woken, and parks no more.
    pub(super) fn stop(&self) {
        self.parks.stopped.store(true, Ordering::SeqCst);
        for index in 0..self.interrupters.len() {
            let Some(core) = u8::try_from(index).ok().and_then(CoreId::new) else {
                continue;
            };
            self.interrupt(core);
            self.wake(core);
        }
    }

    fn is_stopped(&self) -> bool {
        self.parks.stopped.load(Ordering::SeqCst)
    }

    /// Asks `ask` as the last core to park, core `index`, with the idle count held: `None` if
    /// the run is stalled, else whether the core waits for the GPU's next quiet signal. When
    /// `retry`, the core is armed for that signal before it asks, and disarmed unless the
    /// answer is [`Stall::OnlyTheGpu`].
    fn question(&self, index: usize, retry: bool, ask: &mut impl FnMut() -> Stall) -> Option<bool> {
        if retry {
            self.parks.arm(index);
        }
        let answer = ask();
        #[cfg(test)]
        let answer = match answer {
            Stall::Stalled
                if self
                    .gpu_busy_reads
                    .try_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                        left.checked_sub(1)
                    })
                    .is_ok() =>
            {
                Stall::OnlyTheGpu
            }
            other => other,
        };
        let for_quiet = retry && answer == Stall::OnlyTheGpu;
        if retry && !for_quiet {
            self.parks.disarm(index);
        }
        (answer != Stall::Stalled).then_some(for_quiet)
    }

    /// Parks `core` until it is woken or the run stops. If it is the last core to park, `ask` is
    /// asked first whether the run is stalled, and the park ends at once if it is; an answer
    /// busy only because of the GPU is asked again at the GPU's next quiet signals, at most
    /// [`GPU_QUIET_RETRIES`] times (module docs, "Telling a stalled run"). `ask` takes the
    /// kernel lock; nothing takes this type's counts under the kernel lock, so the order is
    /// always counts, then kernel.
    pub(super) fn park(&self, core: CoreId, mut ask: impl FnMut() -> Stall) -> Parked {
        let index = usize::from(core.index());
        let Some(slot) = self.parks.slots.get(index) else {
            return Parked::Stopped;
        };
        let cores = self.parks.slots.len();
        let mut retries = GPU_QUIET_RETRIES;
        let mut for_quiet = false;
        {
            let mut idle = lock(&self.idle);
            *idle = idle.saturating_add(1);
            if *idle == cores {
                match self.question(index, retries > 0, &mut ask) {
                    None => {
                        *idle = idle.saturating_sub(1);
                        return Parked::Stalled;
                    }
                    Some(armed) => for_quiet = armed,
                }
            }
        }
        let mut state = lock(&slot.state);
        while !state.woken && !self.is_stopped() {
            if for_quiet && state.quiet {
                state.quiet = false;
                drop(state);
                for_quiet = false;
                retries = retries.saturating_sub(1);
                let mut idle = lock(&self.idle);
                if *idle == cores {
                    match self.question(index, retries > 0, &mut ask) {
                        None => {
                            *idle = idle.saturating_sub(1);
                            return Parked::Stalled;
                        }
                        Some(armed) => for_quiet = armed,
                    }
                }
                drop(idle);
                state = lock(&slot.state);
                continue;
            }
            state = slot
                .wake
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state.woken = false;
        drop(state);
        // A signal told after this is told to no one; one told before it is forgotten here.
        self.parks.disarm(index);
        lock(&slot.state).quiet = false;
        let mut idle = lock(&self.idle);
        *idle = idle.saturating_sub(1);
        if self.is_stopped() {
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
mod tests;
