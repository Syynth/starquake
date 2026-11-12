//! The dispatch arms of the shared and transfer memory calls: `MapSharedMemory` (0x13),
//! `UnmapSharedMemory` (0x14) and `CreateTransferMemory` (0x15). Kept out of `Dispatcher::run`
//! for its length; each arm decodes its call's arguments and writes its outputs, as every arm
//! does.
//!
//! The registers are Eden's generated wrappers' (`eden/src/core/hle/kernel/svc.cpp` @ cf654052,
//! `SvcWrap_MapSharedMemory64`, `SvcWrap_UnmapSharedMemory64`,
//! `SvcWrap_CreateTransferMemory64`), as the convention table lists them
//! (`svc/reference/svc_wrappers64.txt`).
//!
//! A transfer memory's pages are unlocked when its last reference goes, by whichever call
//! drops it: the dispatcher does it as it lets the kernel lock go (`Locked`'s `Drop` in
//! `dispatch.rs`, [`transfer_memory::unlock_released`]).

use super::{Call, Dispatcher, Outcome, returning, word};
use crate::handle::{Handle, HandleTable};
use crate::svc::{Convention, SvcId, shared_memory, transfer_memory};
use crate::tables::{AddressSpace, Scheduler};

impl Dispatcher {
    /// The arm of a shared or transfer memory call; any other call is not one
    /// ([`super::unimplemented`]).
    pub(super) fn memory_object(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        match id {
            SvcId::MapSharedMemory => self
                .call::<(&HandleTable, &mut AddressSpace, &mut Scheduler)>(
                    call,
                    |(handles, space, scheduler), call, _| {
                        returning(
                            id,
                            convention,
                            call.regs,
                            |[handle, address, size, permission]| {
                                let handle = Handle::from_register(handle);
                                let permission = word(permission)?;
                                shared_memory::map_shared_memory(
                                    handles, space, scheduler, handle, address, size, permission,
                                )?;
                                Ok([])
                            },
                        )
                    },
                ),
            SvcId::UnmapSharedMemory => self
                .call::<(&HandleTable, &mut AddressSpace, &mut Scheduler)>(
                    call,
                    |(handles, space, scheduler), call, _| {
                        returning(id, convention, call.regs, |[handle, address, size]| {
                            let handle = Handle::from_register(handle);
                            shared_memory::unmap_shared_memory(
                                handles, space, scheduler, handle, address, size,
                            )?;
                            Ok([])
                        })
                    },
                ),
            SvcId::CreateTransferMemory => self
                .call_in_process::<(&mut HandleTable, &mut AddressSpace, &mut Scheduler)>(
                    call,
                    |(handles, space, scheduler), call, _, processes| {
                        returning(id, convention, call.regs, |[address, size, permission]| {
                            let handle = transfer_memory::create_transfer_memory(
                                handles,
                                space,
                                scheduler,
                                processes.own(),
                                address,
                                size,
                                word(permission)?,
                            )?;
                            Ok([u64::from(handle.raw())])
                        })
                    },
                ),
            _ => self.call::<()>(call, |(), call, _| super::unimplemented(id, call.regs)),
        }
    }
}

#[cfg(test)]
mod tests;
