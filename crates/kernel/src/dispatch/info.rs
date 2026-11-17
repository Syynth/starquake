//! The dispatch arm of `GetInfo` (0x29): the call ([`info::get_info`]) with the handle table,
//! the address space and the scheduler, after the process's entropy is drawn the first time a
//! `RandomEntropy` call asks for it ([`draw_entropy`]). The scheduler is declared exclusive for
//! that draw, which keeps the entropy in the process's row; the call itself reads it shared.

use super::{Call, Dispatcher, Outcome, fault, returning, word};
use crate::handle::{Handle, HandleTable};
use crate::id::ProcessId;
use crate::process::ENTROPY_COUNT;
use crate::svc::info::{self, InfoType};
use crate::svc::{Convention, SvcId};
use crate::tables::{AddressSpace, Processes, Scheduler};

impl Dispatcher {
    /// `GetInfo`'s arm, for the calling process ([`crate::tables::Processes::own`]).
    pub(super) fn get_info(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        self.call_in_process::<(&HandleTable, &AddressSpace, &mut Scheduler)>(
            call,
            |(handles, space, scheduler), call, _, processes| {
                get_info((handles, space, scheduler), processes, id, convention, call)
            },
        )
    }
}

/// `GetInfo`'s arm for the calling process ([`Processes::own`]). A `RandomEntropy` call first
/// draws the process's entropy if it has none ([`draw_entropy`]); then the call runs with the
/// scheduler shared.
fn get_info(
    (handles, space, scheduler): (&HandleTable, &AddressSpace, &mut Scheduler),
    processes: &Processes,
    id: SvcId,
    convention: &Convention,
    call: &mut Call<'_, '_>,
) -> Outcome {
    if let Err(outcome) = draw_entropy(scheduler, processes.own(), id, convention, call) {
        return outcome;
    }
    let scheduler = &*scheduler;
    returning(id, convention, call.regs, |[info_type, handle, sub_id]| {
        let handle = Handle::from_register(handle);
        let info_type = word(info_type)?;
        Ok([info::get_info(
            handles, space, scheduler, processes, info_type, handle, sub_id,
        )?])
    })
}

/// Draws `process`'s entropy from the caller's environment when the call asks for
/// `RandomEntropy` and the process has none yet: [`ENTROPY_COUNT`] random numbers, in order,
/// under the kernel lock, so a replay feeds them back in the kernel's order. Whatever the call's
/// handle and sub id, the first such call draws them (`svc::info` lists the divergence). When
/// the environment fails, the call does not run and writes nothing ([`Outcome::EnvFailed`]),
/// as when its clock fails.
fn draw_entropy(
    scheduler: &mut Scheduler,
    process: ProcessId,
    id: SvcId,
    convention: &Convention,
    call: &mut Call<'_, '_>,
) -> Result<(), Outcome> {
    // A convention mismatch is reported by the call itself.
    let Ok([info_type, _, _]) = convention.read::<3>(call.regs) else {
        return Ok(());
    };
    let asks_entropy = info_type == u64::from(InfoType::RandomEntropy.number());
    // A process with no row is reported by the call itself, after the guest's checks.
    let undrawn = scheduler
        .process(process)
        .is_ok_and(|record| record.entropy().is_none());
    if !asks_entropy || !undrawn {
        return Ok(());
    }
    let mut entropy = [0; ENTROPY_COUNT];
    for word in &mut entropy {
        *word = call.caller.env.random_u64().map_err(|error| {
            tracing::error!(number = call.number, %error, "environment failed");
            Outcome::EnvFailed(error)
        })?;
    }
    scheduler
        .set_entropy(process, entropy)
        .map_err(|error| fault(id, &error.to_string(), call.regs))
}
