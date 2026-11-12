//! What a core loop asks of the dispatcher between system calls (kernel slice 10; execution
//! model, "Core threads"): which thread to run next ([`Dispatcher::switch`]), time slicing
//! ([`Dispatcher::preempt`]), telling the cores about changes made outside any core
//! ([`Dispatcher::reschedule`]), and whether every thread waits on something nothing can end
//! ([`Dispatcher::is_stalled`]).
//!
//! Each takes the kernel lock for itself, like a system call. The scheduler's decisions are
//! [`Scheduler`]'s (`sched/scheduler.rs`); what is here is the bookkeeping around them that
//! Eden's `KScheduler::ScheduleImpl` and `KAbstractSchedulerLock::Unlock` do on a console core:
//! recompute each core's thread and signal the others, then switch this core's thread, saving
//! the old one's registers before the lock goes so that no other core can load a stale copy.
//!
//! Threads are stamped with the latest guest time read under the lock (`State::latest`): the
//! scheduler only compares stamps (`Scheduler::update`), and that time never runs backwards, so
//! no clock is read here and a replay sees no extra readings.

use starquake_cpu::GuestRegs;

use super::Dispatcher;
use crate::id::ThreadId;
use crate::sched::{AffinityMask, CoreId, CoreSignals, SchedError, ThreadState};
use crate::tables::Scheduler;

/// A thread a core is to run, with what loading it needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loaded {
    /// The thread, as the scheduler names it.
    pub thread: ThreadId,
    /// The thread's id as the guest knows it, which is how the event stream names it.
    pub guest_id: starquake_trace::ThreadId,
    /// Its saved registers: what the core's vCPU is given.
    pub context: GuestRegs,
}

/// What a core's vCPU holds of the thread the core runs, as [`Dispatcher::switch`] needs to
/// know it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Held<'a> {
    /// Nothing newer than the thread's saved context: the core runs no thread, or it has just
    /// loaded it.
    Same,
    /// These registers, newer than the saved context (the thread ran since it was loaded):
    /// they become the saved context if the core switches away from it.
    Newer(&'a GuestRegs),
    /// Something older than the saved context: the thread's last call began a wait, which
    /// saved its registers, and a wake may have written its result there. If the core keeps
    /// the thread, it loads them again.
    Older,
}

/// What a core does after [`Dispatcher::switch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Switch {
    /// Go on with the thread it runs: its registers are the vCPU's.
    Stay,
    /// Run another thread: give the vCPU its registers.
    Load(Box<Loaded>),
    /// Nothing to run here: park until the scheduler wakes the core.
    Park,
}

impl Dispatcher {
    /// Works out every core's thread again, signalling the other cores whose thread changed
    /// through `signals`, then, if `core` must reschedule, switches it
    /// ([`Scheduler::schedule`]). `held` says what the core's vCPU holds of the thread it ran:
    /// [`Held::Newer`] registers become its saved context before it is let go, unless it waits
    /// (a call that began a wait saved its registers already, and a wake may have written its
    /// result into them); with [`Held::Older`], a core that keeps its thread loads its saved
    /// context again, because the thread waited and was woken before its core rescheduled.
    ///
    /// A core that runs nothing the scheduler knows and has nothing chosen parks.
    pub fn switch(
        &self,
        core: CoreId,
        held: Held<'_>,
        signals: &mut dyn CoreSignals,
    ) -> Result<Switch, SchedError> {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        let latest = locked.state.latest;
        let (scheduler,) = locked.tables();
        scheduler.update(latest, Some(core), signals)?;
        let previous = scheduler.current(core)?;
        let next = if scheduler.needs_scheduling(core)? {
            if let (Some(previous), Held::Newer(regs)) = (previous, held)
                && let Ok(thread) = scheduler.thread(previous)
                && thread.state() != ThreadState::Waiting
            {
                *scheduler.context_mut(previous)? = regs.clone();
            }
            scheduler.schedule(core, signals)?
        } else {
            previous
        };
        Ok(match next {
            None => Switch::Park,
            Some(next) if Some(next) == previous && held != Held::Older => Switch::Stay,
            Some(next) => {
                let thread = scheduler.thread(next)?;
                Switch::Load(Box::new(Loaded {
                    thread: next,
                    guest_id: starquake_trace::ThreadId(thread.thread_id()),
                    context: thread.context().clone(),
                }))
            }
        })
    }

    /// Time slicing: rotates each core's queue at its preemption priority
    /// ([`Scheduler::preempt_threads`]), as the timer thread does every
    /// [`PREEMPTION_INTERVAL_NS`](crate::sched::PREEMPTION_INTERVAL_NS), then signals the cores
    /// whose thread changed. Returns those cores.
    pub fn preempt(&self, signals: &mut dyn CoreSignals) -> Result<AffinityMask, SchedError> {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        let latest = locked.state.latest;
        let (scheduler,) = locked.tables();
        scheduler.preempt_threads()?;
        scheduler.update(latest, None, signals)
    }

    /// Works out every core's thread again and signals each core whose thread changed: what
    /// a change made outside any core needs (the timer thread's [`Dispatcher::fire_timers`], a
    /// host thread's [`Dispatcher::signal_event`] or [`Dispatcher::deliver_replies`]). Returns
    /// those cores.
    pub fn reschedule(&self, signals: &mut dyn CoreSignals) -> Result<AffinityMask, SchedError> {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        let latest = locked.state.latest;
        let (scheduler,) = locked.tables();
        scheduler.update(latest, None, signals)
    }

    /// Whether no thread could ever run again by what the kernel knows: no thread is runnable,
    /// no wait has a timer, and no server holds a deferred request. Every wait left can then
    /// end only by another thread's call (there is none to make one) or by a host thread
    /// signalling an event through [`Dispatcher::signal_event`], which no service of a run does
    /// yet. A core loop whose cores have all parked asks this to tell a deadlocked guest from
    /// one that is waiting for time to pass.
    #[must_use]
    pub fn is_stalled(&self) -> bool {
        let mut locked = self.lock::<(&Scheduler,)>();
        let (scheduler,) = locked.tables();
        let ipc = scheduler.ipc();
        let deferred = ipc.servers().any(|server| {
            ipc.queue(server)
                .is_ok_and(|queue| queue.deferred().next().is_some())
        });
        !scheduler.has_runnable_thread() && scheduler.sync().next_deadline().is_none() && !deferred
    }
}

#[cfg(test)]
mod tests;
