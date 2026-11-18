//! The dispatch arm of `GetResourceLimitLimitValue` (0x30) and `GetResourceLimitCurrentValue`
//! (0x31): the calls ([`crate::svc::resource_limit`]) with the handle table, the address space
//! and the scheduler, all shared, and the process facts. The dispatcher's one `match` reaches it
//! through [`FACT_ARMS`], with `GetInfo`.

use super::{Call, Dispatcher, Outcome, returning, word};
use crate::handle::{Handle, HandleTable};
use crate::svc::resource_limit::{self, LimitTables};
use crate::svc::{Convention, SvcId};
use crate::tables::{AddressSpace, Scheduler};

/// The calls that read a process's facts: `GetInfo` and the two resource-limit calls. The
/// dispatcher's one `match` sends them to [`Dispatcher::facts`].
pub(super) const FACT_ARMS: [SvcId; 3] = [
    SvcId::GetInfo,
    SvcId::GetResourceLimitLimitValue,
    SvcId::GetResourceLimitCurrentValue,
];

impl Dispatcher {
    /// The arm of the calls in [`FACT_ARMS`]: `GetInfo`'s own arm (`dispatch/info.rs`), or
    /// [`Dispatcher::resource_limit`].
    pub(super) fn facts(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        if id == SvcId::GetInfo {
            self.get_info(call, id, convention)
        } else {
            self.resource_limit(call, id, convention)
        }
    }

    /// The arm of both resource-limit calls: decodes the handle (`x1`) and the resource (`w2`),
    /// and writes the value to `x1`.
    fn resource_limit(
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
