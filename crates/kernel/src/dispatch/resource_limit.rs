//! The dispatch arm of `GetResourceLimitLimitValue` (0x30) and `GetResourceLimitCurrentValue`
//! (0x31): the calls ([`crate::svc::resource_limit`]) with the handle table, the address space
//! and the scheduler, all shared, and the process facts.

use super::{Call, Dispatcher, Outcome, returning, word};
use crate::handle::{Handle, HandleTable};
use crate::svc::resource_limit::{self, LimitTables};
use crate::svc::{Convention, SvcId};
use crate::tables::{AddressSpace, Scheduler};

impl Dispatcher {
    /// The arm of both calls: decodes the handle (`x1`) and the resource (`w2`), and writes the
    /// value to `x1`.
    pub(super) fn resource_limit(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        self.call_in_process::<(&HandleTable, &AddressSpace, &Scheduler)>(
            call,
            |(handles, space, scheduler), call, _, processes| {
                returning(id, convention, call.regs, |[handle, which]| {
                    let tables = LimitTables {
                        handles,
                        space,
                        scheduler,
                        processes,
                    };
                    let handle = Handle::from_register(handle);
                    let which = word(which)?;
                    let value = if id == SvcId::GetResourceLimitLimitValue {
                        resource_limit::get_resource_limit_limit_value(tables, handle, which)?
                    } else {
                        resource_limit::get_resource_limit_current_value(tables, handle, which)?
                    };
                    Ok([value])
                })
            },
        )
    }
}
