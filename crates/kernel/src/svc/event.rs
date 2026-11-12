//! The event system calls: `SignalEvent` (0x11), `ClearEvent` (0x12), `ResetSignal` (0x17) and
//! `CreateEvent` (0x45).
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later):
//! - `eden/src/core/hle/kernel/svc/svc_event.cpp`: `SignalEvent` (the handle must name an
//!   event's writable end, `GetObject<KEvent>`, else `ResultInvalidHandle`), `ClearEvent` (the
//!   writable end, else the readable end, else `ResultInvalidHandle`) and `CreateEvent`.
//! - `eden/src/core/hle/kernel/svc/svc_synchronization.cpp`: `ResetSignal` (a readable event's
//!   `Reset`, else a process's, else `ResultInvalidHandle`).
//! - `eden/src/core/hle/kernel/k_handle_table.h`: `GetObject<T>` gives a pseudo-handle a
//!   meaning only for `KThread` and `KProcess`, so the current-process pseudo-handle reaches
//!   `ResetSignal`'s process step, and every other lookup here refuses both pseudo-handles.
//!
//! `CreateEvent` in Eden's order: reserve an event from the resource limit (not here, see the
//! divergences in [`crate::sync`]); create the event (`ResultOutOfResource` when the object table
//! is full); add a handle to the writable end, then one to the readable end, removing the first
//! when the second can't be added (`ResultOutOfHandles`); close the creator's references, so the
//! handles hold the only ones.
//!
//! `SignalEvent`'s "fail-safe for system applets" in Eden (any handle succeeds for a program id
//! of the form `0x01000000000010xx`) is not ported: it is Eden's workaround for system applets
//! it runs, not Horizon's behaviour, and starquake runs no system applets.
//!
//! `ResetSignal` on a process: Eden's `KProcess::Reset` is `ResultInvalidState` unless the
//! process is signalled and not terminated. No process is ever signalled before kernel slice 8
//! (see [`crate::sync`]), so it is always [`KernelError::InvalidState`] here; PROVISIONAL.
//! Test (in `dispatch/sync/tests.rs`): `each_event_call_takes_the_ends_eden_takes_and_refuses_the_rest`.

use super::handle::close_handle;
use crate::handle::{Handle, HandleTable, Resolved};
use crate::id::ObjectId;
use crate::object::Object;
use crate::result::{CallResult, KernelError};
use crate::tables::Scheduler;

/// What `handle` names in the caller's table, without pseudo-handles (`GetObject<T>` for a type
/// that is neither a thread nor a process): the object's id and the object, or
/// [`KernelError::InvalidHandle`].
fn object_of(
    handles: &HandleTable,
    scheduler: &Scheduler,
    handle: Handle,
) -> CallResult<(ObjectId, Object)> {
    let id = handles
        .get(handle)
        .map_err(|_| KernelError::InvalidHandle)?;
    Ok((id, scheduler.objects().get(id)?))
}

/// Creates an event and returns handles to its writable and readable ends, in that order (the
/// order Eden's wrapper writes them, `x1` then `x2`).
pub fn create_event(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
) -> CallResult<(Handle, Handle)> {
    let (writable, readable) = scheduler.create_event()?;
    let added = add_both(handles, scheduler, writable, readable);
    // The creator's references, whatever happened (Eden's `SCOPE_EXIT`): the readable end's,
    // then the event's.
    let released = scheduler.release_objects(&[readable, writable]);
    let added = added?;
    released?;
    Ok(added)
}

/// Adds handles to both ends; all or nothing.
fn add_both(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
    writable: ObjectId,
    readable: ObjectId,
) -> CallResult<(Handle, Handle)> {
    let write_handle = handles.add(scheduler.objects_mut(), writable)?;
    match handles.add(scheduler.objects_mut(), readable) {
        Ok(read_handle) => Ok((write_handle, read_handle)),
        Err(error) => {
            // Eden's `ON_RESULT_FAILURE`: the first handle goes again, closed as `CloseHandle`
            // closes one (the creator's reference is still held, so it is not the last).
            close_handle(handles, scheduler, write_handle)?;
            Err(error.into())
        }
    }
}

/// Signals the event whose writable end `handle` names.
pub fn signal_event(
    handles: &HandleTable,
    scheduler: &mut Scheduler,
    handle: Handle,
) -> CallResult<()> {
    match object_of(handles, scheduler, handle)? {
        (_, Object::Event(event)) => scheduler.signal_event(event),
        _ => Err(KernelError::InvalidHandle.into()),
    }
}

/// Clears the event either end of which `handle` names.
pub fn clear_event(
    handles: &HandleTable,
    scheduler: &mut Scheduler,
    handle: Handle,
) -> CallResult<()> {
    match object_of(handles, scheduler, handle)? {
        (_, Object::Event(event)) => scheduler.clear_event(event),
        (readable, Object::ReadableEvent(_)) => scheduler.clear_readable(readable),
        _ => Err(KernelError::InvalidHandle.into()),
    }
}

/// Resets the readable event, or the process, `handle` names.
pub fn reset_signal(
    handles: &HandleTable,
    scheduler: &mut Scheduler,
    handle: Handle,
) -> CallResult<()> {
    let process_reset = Err(KernelError::InvalidState.into());
    match handles
        .resolve(handle)
        .map_err(|_| KernelError::InvalidHandle)?
    {
        Resolved::CurrentProcess => process_reset,
        Resolved::CurrentThread => Err(KernelError::InvalidHandle.into()),
        Resolved::Object(id) => match scheduler.objects().get(id)? {
            Object::ReadableEvent(_) => scheduler.reset_readable(id),
            Object::Process(_) => process_reset,
            Object::Thread(_)
            | Object::Session
            | Object::Event(_)
            | Object::SharedMemory(_)
            | Object::TransferMemory(_) => Err(KernelError::InvalidHandle.into()),
        },
    }
}
