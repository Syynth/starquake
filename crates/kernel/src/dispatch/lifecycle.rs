//! The dispatch arms of the process and thread lifecycle calls (kernel slice 8): `ExitProcess`,
//! `ExitThread`, `CreateThread`, `StartThread`, `SleepThread`, `SetThreadPriority`,
//! `GetThreadCoreMask`, `SetThreadCoreMask`, `GetThreadId` and `GetProcessId`. Kept out of
//! `Dispatcher::run` for its length; each arm decodes its call's arguments and writes its
//! outputs, as every arm does. A `SleepThread` with a positive time waits through the wait path
//! of `dispatch/sync.rs` (kernel slice 10): its event is recorded when its timer ends it.

use super::arbitration::origin;
use super::{Call, Dispatcher, Now, Outcome, calling_core, fault, returning, signed, void};
use crate::handle::{Handle, HandleTable};
use crate::sched::ThreadExit;
use crate::svc::sync::Waiter;
use crate::svc::{Convention, SvcId, process, thread};
use crate::sync::Arbitrated;
use crate::tables::{AddressSpace, Processes, Scheduler};

impl Dispatcher {
    /// The arm of a lifecycle call; any other call is not one ([`super::unimplemented`]).
    pub(super) fn lifecycle(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        match id {
            SvcId::ExitProcess | SvcId::ExitThread => self.exit(call, id),
            _ => self.thread_call(call, id, convention),
        }
    }

    /// The arms of `ExitProcess` and `ExitThread`, which write no register and may close the
    /// last reference to a session.
    fn exit(&self, call: &mut Call<'_, '_>, id: SvcId) -> Outcome {
        match id {
            SvcId::ExitProcess => self.call_then_settle::<(&mut HandleTable, &mut Scheduler)>(
                call,
                |(handles, scheduler), call, processes| {
                    tracing::info!("the process called ExitProcess");
                    let outcome = match process::exit_process(handles, scheduler, processes) {
                        Ok(()) => Outcome::ProcessExited,
                        Err(error) => fault(id, &error.to_string(), call.regs),
                    };
                    (outcome, scheduler.ipc().has_any_notices())
                },
            ),
            SvcId::ExitThread => self.call_then_settle::<(&mut HandleTable, &mut Scheduler)>(
                call,
                |(handles, scheduler), call, _| {
                    let exited = calling_core(call.caller.core)
                        .and_then(|core| thread::exit_thread(handles, scheduler, core));
                    let outcome = match exited {
                        Ok(ThreadExit::Others) => Outcome::ThreadExited,
                        Ok(ThreadExit::LastOf(_)) => Outcome::ProcessExited,
                        Err(error) => fault(id, &error.to_string(), call.regs),
                    };
                    (outcome, scheduler.ipc().has_any_notices())
                },
            ),
            _ => self.call::<()>(call, |(), call, _| super::unimplemented(id, call.regs)),
        }
    }

    /// The arms of the thread calls that return a result.
    fn thread_call(&self, call: &mut Call<'_, '_>, id: SvcId, convention: &Convention) -> Outcome {
        match id {
            SvcId::CreateThread => self
                .call_in_process::<(&mut HandleTable, &mut AddressSpace, &mut Scheduler)>(
                    call,
                    |tables, call, _, processes| {
                        create_thread(tables, processes, id, convention, call)
                    },
                ),
            SvcId::StartThread => self.call::<(&HandleTable, &mut Scheduler)>(
                call,
                |(handles, scheduler), call, _| {
                    let core = call.caller.core;
                    returning(id, convention, call.regs, |[handle]| {
                        let handle = Handle::from_register(handle);
                        thread::start_thread(handles, scheduler, calling_core(core)?, handle)?;
                        Ok([])
                    })
                },
            ),
            SvcId::SleepThread => self
                .call_at::<(&mut Scheduler,)>(call, |(scheduler,), call, now| {
                    sleep_thread(scheduler, id, convention, call, now)
                }),
            SvcId::SetThreadPriority => self.call_in_process::<(&HandleTable, &mut Scheduler)>(
                call,
                |(handles, scheduler), call, _, processes| {
                    let core = call.caller.core;
                    returning(id, convention, call.regs, |[handle, priority]| {
                        let handle = Handle::from_register(handle);
                        let core = calling_core(core)?;
                        let priority = signed(priority)?;
                        thread::set_thread_priority(
                            handles, scheduler, processes, core, handle, priority,
                        )?;
                        Ok([])
                    })
                },
            ),
            SvcId::GetThreadCoreMask => {
                self.call::<(&HandleTable, &Scheduler)>(call, |(handles, scheduler), call, _| {
                    let core = call.caller.core;
                    returning(id, convention, call.regs, |[handle]| {
                        let handle = Handle::from_register(handle);
                        let (ideal, mask) = thread::get_thread_core_mask(
                            scheduler,
                            handles,
                            calling_core(core)?,
                            handle,
                        )?;
                        Ok([u64::from(ideal.cast_unsigned()), mask])
                    })
                })
            }
            SvcId::SetThreadCoreMask => self.call_in_process::<(&HandleTable, &mut Scheduler)>(
                call,
                |(handles, scheduler), call, _, processes| {
                    let core = call.caller.core;
                    returning(id, convention, call.regs, |[handle, core_id, mask]| {
                        let handle = Handle::from_register(handle);
                        let core = calling_core(core)?;
                        let core_id = signed(core_id)?;
                        thread::set_thread_core_mask(
                            handles,
                            scheduler,
                            processes,
                            core,
                            handle,
                            (core_id, mask),
                        )?;
                        Ok([])
                    })
                },
            ),
            SvcId::GetThreadId => {
                self.call::<(&HandleTable, &Scheduler)>(call, |(handles, scheduler), call, _| {
                    let core = call.caller.core;
                    returning(id, convention, call.regs, |[handle]| {
                        let handle = Handle::from_register(handle);
                        let core = calling_core(core)?;
                        Ok([thread::get_thread_id(scheduler, handles, core, handle)?])
                    })
                })
            }
            SvcId::GetProcessId => self.call_in_process::<(&HandleTable, &Scheduler)>(
                call,
                |(handles, scheduler), call, _, processes| {
                    returning(id, convention, call.regs, |[handle]| {
                        let handle = Handle::from_register(handle);
                        Ok([process::get_process_id(
                            handles, scheduler, processes, handle,
                        )?])
                    })
                },
            ),
            _ => self.call::<()>(call, |(), call, _| super::unimplemented(id, call.regs)),
        }
    }
}

/// The body of `SleepThread`'s arm: a yield or nothing completes at once; a sleep that began
/// writes nothing and records nothing now, because its timer ends it (`dispatch/sync.rs`, "The
/// wait path").
fn sleep_thread(
    scheduler: &mut Scheduler,
    id: SvcId,
    convention: &Convention,
    call: &mut Call<'_, '_>,
    now: Now,
) -> Outcome {
    let (core, origin) = (call.caller.core, origin(call));
    let regs = call.regs.clone();
    let mut sleeps = false;
    let outcome = void(id, convention, call.regs, |[ns]| {
        let waiter = Waiter {
            core: calling_core(core)?,
            now: now.host,
            regs: &regs,
            origin,
        };
        let slept = thread::sleep_thread(scheduler, waiter, ns.cast_signed())?;
        sleeps = matches!(slept, Arbitrated::Waiting(_));
        Ok(())
    });
    if sleeps && outcome == Outcome::Completed {
        Outcome::Waiting
    } else {
        outcome
    }
}

fn create_thread(
    tables: (&mut HandleTable, &mut AddressSpace, &mut Scheduler),
    processes: &Processes,
    id: SvcId,
    convention: &Convention,
    call: &mut Call<'_, '_>,
) -> Outcome {
    let (core, memory) = (call.caller.core, call.caller.memory);
    returning(
        id,
        convention,
        call.regs,
        |[entry, argument, stack_top, priority, core_id]| {
            let args = thread::CreateThread {
                entry,
                argument,
                stack_top,
                priority: signed(priority)?,
                core_id: signed(core_id)?,
            };
            let handle =
                thread::create_thread(tables, processes, memory, calling_core(core)?, args)?;
            Ok([u64::from(handle.raw())])
        },
    )
}

#[cfg(test)]
mod tests;
