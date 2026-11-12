//! The dispatcher's arms for lock arbitration, condition variables and the address arbiter
//! ([`ARMS`]). The functions behind them are [`crate::svc::arbitration`].
//!
//! The three calls that can wait block through slice 4's wait path (`dispatch/sync.rs`, "The
//! wait path"): a call that waits ends in [`Outcome::Waiting`] and records nothing; the wake
//! writes its result into the thread's saved `x0` and queues it, and the event is recorded, for
//! the call the thread was in, when the wake is drained. A `WaitProcessWideKeyAtomic` that a
//! signal hands on to a lock owner is still that call, and its event is recorded as such when
//! it gets the lock.

use super::{
    Answer, Call, Dispatcher, Now, Outcome, answering, calling_core, recorded_args, returning,
    signed, void, word,
};
use crate::handle::{Handle, HandleTable};
use crate::memory::AddressSpace;
use crate::result::ResultCode;
use crate::svc::sync::Waiter;
use crate::svc::{Convention, SvcId, arbitration};
use crate::sync::{Arbitrated, WaitOrigin};
use crate::tables::Scheduler;

#[cfg(test)]
mod tests;

/// The calls these arms dispatch.
pub(super) const ARMS: [SvcId; 6] = [
    SvcId::ArbitrateLock,
    SvcId::ArbitrateUnlock,
    SvcId::WaitProcessWideKeyAtomic,
    SvcId::SignalProcessWideKey,
    SvcId::WaitForAddress,
    SvcId::SignalToAddress,
];

/// Who a call that may wait is recorded for when its wait ends.
pub(super) fn origin(call: &Call<'_, '_>) -> WaitOrigin {
    WaitOrigin {
        thread: call.caller.thread,
        core: call.caller.core,
        args: recorded_args(call.id, &call.passed),
    }
}

/// What a call that may wait answers: success at once, or later, when its wait ends.
const fn answer(begun: Arbitrated) -> Answer<0> {
    match begun {
        Arbitrated::Done => Answer::Now {
            result: ResultCode::SUCCESS,
            outputs: [],
        },
        Arbitrated::Waiting(_) => Answer::Later,
    }
}

/// A call that may wait, decoded: its inputs, run with the caller as a [`Waiter`].
fn waiting<const I: usize>(
    id: SvcId,
    convention: &Convention,
    call: &mut Call<'_, '_>,
    now: Now,
    body: impl FnOnce(Waiter<'_>, [u64; I]) -> crate::result::CallResult<Arbitrated>,
) -> Outcome {
    let (core, origin) = (call.caller.core, origin(call));
    let regs = call.regs.clone();
    answering(id, convention, call.regs, |inputs| {
        let waiter = Waiter {
            core: calling_core(core)?,
            now: now.host,
            regs: &regs,
            origin,
        };
        Ok(answer(body(waiter, inputs)?))
    })
}

impl Dispatcher {
    /// The arbitration arms of the dispatcher's one `match` (the calls in [`ARMS`]), each
    /// naming the tables its function declares.
    pub(super) fn arbitration(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        match id {
            SvcId::ArbitrateLock | SvcId::ArbitrateUnlock => self.lock_arms(call, id, convention),
            SvcId::WaitProcessWideKeyAtomic | SvcId::SignalProcessWideKey => {
                self.condition_variable_arms(call, id, convention)
            }
            _ => self.arbiter_arms(call, id, convention),
        }
    }

    /// `ArbitrateLock` and `ArbitrateUnlock`.
    fn lock_arms(&self, call: &mut Call<'_, '_>, id: SvcId, convention: &Convention) -> Outcome {
        if id == SvcId::ArbitrateLock {
            return self.call_at::<(&HandleTable, &AddressSpace, &mut Scheduler)>(
                call,
                |(handles, space, scheduler), call, now| {
                    let memory = call.caller.memory;
                    waiting(
                        id,
                        convention,
                        call,
                        now,
                        |waiter, [handle, address, tag]| {
                            arbitration::arbitrate_lock(
                                (handles, space, scheduler),
                                memory,
                                waiter,
                                Handle::from_register(handle),
                                address,
                                word(tag)?,
                            )
                        },
                    )
                },
            );
        }
        self.call::<(&AddressSpace, &mut Scheduler)>(call, |(space, scheduler), call, _| {
            let (memory, core) = (call.caller.memory, call.caller.core);
            returning(id, convention, call.regs, |[address]| {
                let core = calling_core(core)?;
                arbitration::arbitrate_unlock(space, memory, scheduler, core, address)?;
                Ok([])
            })
        })
    }

    /// `WaitProcessWideKeyAtomic` and `SignalProcessWideKey`.
    fn condition_variable_arms(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        if id == SvcId::WaitProcessWideKeyAtomic {
            return self.call_at::<(&AddressSpace, &mut Scheduler)>(
                call,
                |(space, scheduler), call, now| {
                    let memory = call.caller.memory;
                    waiting(
                        id,
                        convention,
                        call,
                        now,
                        |waiter, [address, cv_key, tag, timeout]| {
                            let args = (address, cv_key, word(tag)?, timeout.cast_signed());
                            arbitration::wait_process_wide_key_atomic(
                                space, memory, scheduler, waiter, args,
                            )
                        },
                    )
                },
            );
        }
        self.call::<(&HandleTable, &AddressSpace, &mut Scheduler)>(
            call,
            |(handles, space, scheduler), call, _| {
                let (memory, core) = (call.caller.memory, call.caller.core);
                void(id, convention, call.regs, |[cv_key, count]| {
                    arbitration::signal_process_wide_key(
                        (handles, space, scheduler),
                        memory,
                        calling_core(core)?,
                        cv_key,
                        signed(count)?,
                    )
                })
            },
        )
    }

    /// `WaitForAddress` and `SignalToAddress`.
    fn arbiter_arms(&self, call: &mut Call<'_, '_>, id: SvcId, convention: &Convention) -> Outcome {
        if id == SvcId::WaitForAddress {
            return self.call_at::<(&AddressSpace, &mut Scheduler)>(
                call,
                |(space, scheduler), call, now| {
                    let memory = call.caller.memory;
                    waiting(
                        id,
                        convention,
                        call,
                        now,
                        |waiter, [address, kind, value, timeout]| {
                            let args =
                                (address, word(kind)?, signed(value)?, timeout.cast_signed());
                            arbitration::wait_for_address(space, memory, scheduler, waiter, args)
                        },
                    )
                },
            );
        }
        self.call::<(&AddressSpace, &mut Scheduler)>(call, |(space, scheduler), call, _| {
            let (memory, core) = (call.caller.memory, call.caller.core);
            returning(
                id,
                convention,
                call.regs,
                |[address, kind, value, count]| {
                    let args = (address, word(kind)?, signed(value)?, signed(count)?);
                    let core = calling_core(core)?;
                    arbitration::signal_to_address(space, memory, scheduler, core, args)?;
                    Ok([])
                },
            )
        })
    }
}
