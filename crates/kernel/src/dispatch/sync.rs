//! The dispatcher's synchronisation arms (`SignalEvent`, `ClearEvent`, `ResetSignal`,
//! `WaitSynchronization`, `CancelSynchronization`, `CreateEvent`), and the two ways a wait ends
//! outside a system call: [`Dispatcher::signal_event`] for a host thread (a service's vsync or
//! input thread) and [`Dispatcher::fire_timers`] for the timer thread.
//!
//! The functions behind the arms are [`crate::svc::event`] and [`crate::svc::sync`]; the
//! objects and waits are [`crate::sync`].
//!
//! # The wait path
//!
//! `WaitSynchronization` blocks the way `SendSyncRequest` does, through the same path
//! (`dispatch::ipc`): the dispatch ends in [`Outcome::Waiting`], writes no register and records
//! no event, and when the wait ends the thread is reported by [`Dispatcher::take_woken`] and its
//! event recorded then. What is new is where the result goes: the waiting thread is a scheduler
//! thread, so its registers are saved as its context when it begins to wait, the wake writes
//! the result and index there (`x0`, `x1`) and makes it runnable, and the core that next
//! schedules it loads that context. A core loop must therefore not save a thread's registers
//! after a `WaitSynchronization` that ended in [`Outcome::Waiting`]: the saved context is
//! already current, and a wake may have written its result into it.
//!
//! A wait can end under any dispatch (a signal, a cancel), under [`Dispatcher::signal_event`]
//! or [`Dispatcher::fire_timers`], or inside a server's handler, which signals through
//! [`IpcContext::signal_event`](super::IpcContext::signal_event) without the recorder. Each
//! ended wait is queued in the scheduler ([`Scheduler::take_wakes`]) and recorded, with the
//! clock reading of the dispatch that records it, by [`drain_wakes`]: every dispatch made
//! through the dispatcher's common step drains the queue when it takes the lock (so a wake a
//! handler caused is recorded before anything the woken thread does next) and again before it
//! releases it; `SendSyncRequest` and [`Dispatcher::deliver_replies`] drain it when they end.
//! The event is the waiter's own `WaitSynchronization` event, stamped with its thread and core.

use starquake_trace::{EnvError, EventKind, GuestTicks, HostNanos, Recorder};

use super::{
    Answer, Call, Dispatcher, Outcome, State, Woken, answering, calling_core, recorded_args,
    returning,
};
use crate::handle::{Handle, HandleTable};
use crate::id::ObjectId;
use crate::memory::AddressSpace;
use crate::result::{ResultCode, SvcError};
use crate::svc::sync::{WaitArgs, Waiter};
use crate::svc::{Convention, SvcId, event, sync};
use crate::sync::{WaitBegun, WaitOrigin};
use crate::tables::Scheduler;

#[cfg(test)]
mod tests;

/// Why [`Dispatcher::signal_event`] or [`Dispatcher::fire_timers`] failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WakeError {
    /// The environment's clock failed (a replay that diverged): nothing was done.
    #[error(transparent)]
    Env(#[from] EnvError),
    /// The operation failed: [`SvcError::Kernel`] for an object that is not an event,
    /// [`SvcError::Internal`] for the kernel's own bookkeeping.
    #[error(transparent)]
    Call(#[from] SvcError),
}

/// Records the event of every wait that ended since the last drain, at `time`, and reports its
/// thread woken: the recording point for a `WaitSynchronization` that waited.
pub(super) fn drain_wakes(state: &mut State, recorder: &Recorder, time: GuestTicks) {
    for wake in state.tables.scheduler.take_wakes() {
        let origin = wake.origin;
        recorder.record(
            time,
            origin.core,
            origin.thread,
            EventKind::Svc {
                id: u32::from(SvcId::WaitSynchronization.number()),
                args: origin.args,
                // The one place a woken waiter's result becomes the event stream's word.
                result: starquake_trace::ResultCode(wake.result.raw()),
            },
        );
        state.woken.push(Woken {
            thread: origin.thread,
            core: origin.core,
            result: wake.result,
        });
    }
}

impl Dispatcher {
    /// The synchronisation arms of the dispatcher's one `match`, each naming the tables its
    /// function declares.
    pub(super) fn synchronization(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        match id {
            SvcId::SignalEvent => self.call::<(&HandleTable, &mut Scheduler)>(
                call,
                |(handles, scheduler), call, _| {
                    returning(id, convention, call.regs, |[handle]| {
                        event::signal_event(handles, scheduler, Handle::from_register(handle))?;
                        Ok([])
                    })
                },
            ),
            SvcId::ClearEvent => self.call::<(&HandleTable, &mut Scheduler)>(
                call,
                |(handles, scheduler), call, _| {
                    returning(id, convention, call.regs, |[handle]| {
                        event::clear_event(handles, scheduler, Handle::from_register(handle))?;
                        Ok([])
                    })
                },
            ),
            SvcId::ResetSignal => self.call::<(&HandleTable, &mut Scheduler)>(
                call,
                |(handles, scheduler), call, _| {
                    returning(id, convention, call.regs, |[handle]| {
                        event::reset_signal(handles, scheduler, Handle::from_register(handle))?;
                        Ok([])
                    })
                },
            ),
            SvcId::CancelSynchronization => self.call::<(&HandleTable, &mut Scheduler)>(
                call,
                |(handles, scheduler), call, _| {
                    let core = call.caller.core;
                    returning(id, convention, call.regs, |[handle]| {
                        let handle = Handle::from_register(handle);
                        sync::cancel_synchronization(
                            handles,
                            scheduler,
                            calling_core(core)?,
                            handle,
                        )?;
                        Ok([])
                    })
                },
            ),
            SvcId::CreateEvent => self.call::<(&mut HandleTable, &mut Scheduler)>(
                call,
                |(handles, scheduler), call, _| {
                    returning(id, convention, call.regs, |[]| {
                        let (writable, readable) = event::create_event(handles, scheduler)?;
                        Ok([u64::from(writable.raw()), u64::from(readable.raw())])
                    })
                },
            ),
            SvcId::WaitSynchronization => self
                .call_at::<(&HandleTable, &AddressSpace, &mut Scheduler)>(
                    call,
                    |(handles, space, scheduler), call, now| {
                        let memory = call.caller.memory;
                        let origin = WaitOrigin {
                            thread: call.caller.thread,
                            core: call.caller.core,
                            args: recorded_args(call.id, &call.passed),
                        };
                        let core = call.caller.core;
                        let regs = call.regs.clone();
                        answering(id, convention, call.regs, |[address, count, timeout]| {
                            let waiter = Waiter {
                                core: calling_core(core)?,
                                now: now.host,
                                regs: &regs,
                                origin,
                            };
                            let args = WaitArgs {
                                handles: address,
                                num_handles: i32_argument(count)?,
                                timeout_ns: timeout.cast_signed(),
                            };
                            let begun = sync::wait_synchronization(
                                handles, space, memory, scheduler, waiter, args,
                            );
                            Ok(match begun? {
                                WaitBegun::Signalled(index) => Answer::Now {
                                    result: ResultCode::SUCCESS,
                                    outputs: [u64::try_from(index)
                                        .map_err(|error| SvcError::internal(&error))?],
                                },
                                // Eden's `Wait` returns before it assigns the index: the wrapper's
                                // zero.
                                WaitBegun::Ended(error) => Answer::Now {
                                    result: error.code(),
                                    outputs: [0],
                                },
                                WaitBegun::Waiting(_) => Answer::Later,
                            })
                        })
                    },
                ),
            _ => self.call::<()>(call, |(), call, _| super::unimplemented(id, call.regs)),
        }
    }

    /// Signals the event `event` names, either end, from a host thread outside any system call:
    /// how a service that runs on its own (vsync, input) wakes guest threads. Thread-safe like
    /// every dispatcher operation: it takes the kernel lock. The environment's clock is read
    /// once, under the lock, to stamp the events of the waits it ends, which go to `recorder`;
    /// their threads are reported by [`Dispatcher::take_woken`].
    ///
    /// [`KernelError::InvalidHandle`](crate::result::KernelError::InvalidHandle) for an id that
    /// names no live event, as [`Scheduler::signal_object`] says.
    pub fn signal_event(
        &self,
        event: ObjectId,
        env: &dyn starquake_trace::Env,
        recorder: &Recorder,
    ) -> Result<(), WakeError> {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        let now = locked.clock(env)?;
        let (scheduler,) = locked.tables();
        let signalled = scheduler.signal_object(event);
        drain_wakes(&mut locked.state, recorder, now.guest);
        Ok(signalled?)
    }

    /// Expires every timeout due at the environment's clock reading, read once under the lock,
    /// and returns the next deadline still in the timer queue: what the timer thread waits for
    /// next. The events of the waits it ends go to `recorder`, stamped with that reading, and
    /// their threads are reported by [`Dispatcher::take_woken`].
    pub fn fire_timers(
        &self,
        env: &dyn starquake_trace::Env,
        recorder: &Recorder,
    ) -> Result<Option<HostNanos>, WakeError> {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        let now = locked.clock(env)?;
        let (scheduler,) = locked.tables();
        let fired = scheduler.fire_timers(now.host);
        let next = scheduler.sync().next_deadline();
        drain_wakes(&mut locked.state, recorder, now.guest);
        fired?;
        Ok(next)
    }
}

/// A decoded `int32_t` argument as the `i32` it is: its four bytes, read as signed. The
/// convention truncated it to four bytes, so a value that does not fit is a dispatcher bug.
fn i32_argument(argument: u64) -> Result<i32, SvcError> {
    u32::try_from(argument)
        .map(u32::cast_signed)
        .map_err(|error| SvcError::internal(&error))
}
