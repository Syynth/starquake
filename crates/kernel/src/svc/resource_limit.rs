//! Resource-limit system calls: `GetResourceLimitLimitValue` (0x30) and
//! `GetResourceLimitCurrentValue` (0x31), answered from the limit a handle names
//! ([`crate::resource_limit`]).
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_resource_limit.cpp` @ cf654052
//! (GPL-3.0-or-later). Both calls check, in Eden's order:
//!
//! 1. The resource is valid (`IsValidResourceType(which)`: below `LimitableResource::Count`;
//!    [`LimitableResource::from_raw`]), else `ResultInvalidEnumValue`.
//! 2. The handle names a resource limit (`GetObject<KResourceLimit>`: neither pseudo-handle
//!    does, since `KResourceLimit` is neither a process nor a thread), else
//!    `ResultInvalidHandle`.
//!
//! Then the value is the limit's `GetLimitValue(which)` or `GetCurrentValue(which)`
//! ([`Values`]). Tests (in `svc/resource_limit/tests.rs`):
//! `an_invalid_resource_is_refused_before_the_handle`,
//! `only_a_handle_to_a_resource_limit_is_answered`,
//! `each_resource_is_answered_from_the_limit`.
//!
//! The calls that are not built (`CreateResourceLimit` 0x7d, `SetResourceLimitLimitValue` 0x7e,
//! `GetResourceLimitPeakValue` 0x37) are listed in [`crate::resource_limit`]'s divergences:
//! nothing on the path to a title makes them.
//!
//! A limit names its process ([`ResourceLimit::owner`]); its physical memory values are that
//! process's address space's, which is the calling process's, the only address space beside the
//! tables (`crate::process`, "One process per set of tables"). A limit of another process asked
//! for its physical memory is an internal failure, after the guest's checks; its thread values
//! are read from the scheduler, whichever process it is.

use crate::handle::{Handle, HandleTable, Resolved};
use crate::memory::AddressSpace;
use crate::object::Object;
use crate::resource_limit::{LimitableResource, ResourceLimit, Usage, Values};
use crate::result::{CallResult, KernelError, SvcError};
use crate::tables::{Processes, Scheduler};

/// The tables the resource-limit calls read: the calling process's handle table and address
/// space, the scheduler (the object table and the processes' thread rows) and the process
/// facts.
#[derive(Debug, Clone, Copy)]
pub struct LimitTables<'t> {
    /// The calling process's handle table.
    pub handles: &'t HandleTable,
    /// The calling process's address space.
    pub space: &'t AddressSpace,
    /// The scheduler.
    pub scheduler: &'t Scheduler,
    /// The process facts.
    pub processes: &'t Processes,
}

/// `GetResourceLimitLimitValue`: the limit of resource `which` in the resource limit `handle`
/// names, after Eden's two checks (see the module docs).
pub fn get_resource_limit_limit_value(
    tables: LimitTables<'_>,
    handle: Handle,
    which: u32,
) -> CallResult<u64> {
    Ok(values_for(tables, handle, which)?.limit)
}

/// `GetResourceLimitCurrentValue`: the current value of resource `which` in the resource limit
/// `handle` names, after Eden's two checks (see the module docs).
pub fn get_resource_limit_current_value(
    tables: LimitTables<'_>,
    handle: Handle,
    which: u32,
) -> CallResult<u64> {
    Ok(values_for(tables, handle, which)?.current)
}

/// Eden's two checks, then the values of `which` in the limit `handle` names.
fn values_for(tables: LimitTables<'_>, handle: Handle, which: u32) -> CallResult<Values> {
    // `R_UNLESS(IsValidResourceType(which), ResultInvalidEnumValue)`.
    let resource = LimitableResource::from_raw(which).ok_or(KernelError::InvalidEnumValue)?;
    // `R_UNLESS(resource_limit.IsNotNull(), ResultInvalidHandle)`.
    let limit = limit_from_handle(tables.handles, tables.scheduler, handle)?;
    values(tables, limit, resource)
}

/// The resource limit `handle` names (`GetObject<KResourceLimit>`):
/// [`KernelError::InvalidHandle`] for either pseudo-handle and for a handle to any other kind of
/// object.
fn limit_from_handle(
    handles: &HandleTable,
    scheduler: &Scheduler,
    handle: Handle,
) -> CallResult<ResourceLimit> {
    match handles.resolve(handle)? {
        Resolved::CurrentProcess | Resolved::CurrentThread => {
            Err(KernelError::InvalidHandle.into())
        }
        Resolved::Object(object) => match scheduler.objects().get(object)? {
            Object::ResourceLimit(limit) => Ok(limit),
            Object::Thread(_)
            | Object::Process(_)
            | Object::Session
            | Object::Event(_)
            | Object::ReadableEvent(_)
            | Object::SharedMemory(_)
            | Object::TransferMemory(_) => Err(KernelError::InvalidHandle.into()),
        },
    }
}

/// The values of `resource` in `limit`, read from its owner's tables (see the module docs).
pub(crate) fn values(
    tables: LimitTables<'_>,
    limit: ResourceLimit,
    resource: LimitableResource,
) -> CallResult<Values> {
    let owner = limit.owner();
    let threads = tables.scheduler.process(owner)?.thread_count();
    if resource == LimitableResource::PhysicalMemory && owner != tables.processes.own() {
        return Err(SvcError::Internal(format!(
            "the physical memory of process {owner:?}'s resource limit was asked for with another process's address space"
        )));
    }
    Ok(Values::of(
        resource,
        Usage {
            memory: tables.space.limit(),
            threads,
        },
    ))
}

#[cfg(test)]
mod tests;
