//! The dispatch arm of `GetInfo` (0x29): the call ([`info::get_info`]) with the handle table,
//! the address space, the scheduler and the calling core. The handle table and the scheduler are
//! declared exclusive for `MesosphereCurrentProcess`, which adds a handle (and so a reference to
//! the process's object). The arm reads nothing from the environment besides the dispatch's
//! clock: a process's entropy is drawn when it is created (`crate::process`, "What a process
//! draws from the environment").

use super::{Call, Dispatcher, Outcome, calling_core, returning, word};
use crate::handle::{Handle, HandleTable};
use crate::svc::info::{self, InfoTables};
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
        self.call_in_process::<(&mut HandleTable, &AddressSpace, &mut Scheduler)>(
            call,
            |(handles, space, scheduler), call, _, processes| {
                get_info((handles, space, scheduler), processes, id, convention, call)
            },
        )
    }
}

/// `GetInfo`'s arm for the calling process ([`Processes::own`]), on the calling core.
fn get_info(
    (handles, space, scheduler): (&mut HandleTable, &AddressSpace, &mut Scheduler),
    processes: &Processes,
    id: SvcId,
    convention: &Convention,
    call: &mut Call<'_, '_>,
) -> Outcome {
    let core = call.caller.core;
    returning(id, convention, call.regs, |[info_type, handle, sub_id]| {
        let handle = Handle::from_register(handle);
        let info_type = word(info_type)?;
        let tables = InfoTables {
            handles,
            space,
            scheduler,
            processes,
        };
        Ok([info::get_info(
            tables,
            calling_core(core)?,
            info_type,
            handle,
            sub_id,
        )?])
    })
}
