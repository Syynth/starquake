//! Process system calls: `ExitProcess` (0x07) and `GetProcessId` (0x24), and ending a process,
//! which the last thread to exit does too.
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), paths relative to the reference folder:
//!
//! - `eden/src/core/hle/kernel/svc/svc_process.cpp`, `ExitProcess` and `GetProcessId`.
//!   `GetProcessId` looks the handle up with `GetObject<KAutoObject>`
//!   (`eden/src/core/hle/kernel/k_handle_table.h`), for which only the `CurrentProcess`
//!   pseudo-handle is special (`std::derived_from<KProcess, KAutoObject>`): `CurrentThread`
//!   names nothing there, so it is `ResultInvalidHandle`. A process object gives its own id, a
//!   thread its owner's; anything else, or a thread with no owner, is `ResultInvalidHandle`.
//! - `eden/src/core/hle/kernel/k_process.cpp`, `KProcess::Exit`, `Terminate`,
//!   `StartTermination` and `FinishTermination`: every thread of the process ends, its handle
//!   table is finalized (each handle closed, lowest index first, `KHandleTable::Finalize`), and
//!   it becomes `Terminated`.
//!
//! # Divergences
//!
//! - **`ExitProcess` changes kernel state here.** Eden's svc calls `system.Exit()`, which stops
//!   the emulator without ending the process's threads or closing its handles; Horizon's
//!   `KProcess::Exit` does both, and this follows Horizon. Test:
//!   `exiting_the_process_ends_every_thread_and_closes_every_handle`.
//! - **A process that has not run may exit.** Eden asserts the process is `Running`. Here a
//!   `Created` process ends the same way; one that has ended already is an internal failure.
//!   Test: `a_process_ends_once`.

use crate::handle::{Handle, HandleTable, Resolved};
use crate::id::ProcessId;
use crate::object::Object;
use crate::process::ProcessState;
use crate::result::{CallResult, KernelError, SvcError};
use crate::svc::handle::remove_handle;
use crate::tables::{Processes, Scheduler};

/// Ends `process`, whose handle table `handles` is: every thread of it that has not ended
/// ends, every handle is closed (tearing down what the last reference to anything named), and
/// it becomes [`ProcessState::Terminated`]. A process that has ended already is an internal
/// failure.
pub fn terminate_process(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
    process: ProcessId,
) -> CallResult<()> {
    if scheduler.process(process)?.state() == ProcessState::Terminated {
        return Err(SvcError::Internal(format!(
            "process {process:?} has ended already"
        )));
    }
    scheduler.terminate_threads(process)?;
    for handle in handles.handles() {
        remove_handle(handles, scheduler, handle)?;
    }
    scheduler.mark_terminated(process)?;
    Ok(())
}

/// `ExitProcess`: ends the process whose tables these are ([`Processes::own`]). See
/// [`terminate_process`].
pub fn exit_process(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
    processes: &Processes,
) -> CallResult<()> {
    terminate_process(handles, scheduler, processes.own())
}

/// `GetProcessId`: the id of the process `handle` names, or of the process that owns the
/// thread it names. See the module docs for which handles are [`KernelError::InvalidHandle`].
pub fn get_process_id(
    handles: &HandleTable,
    scheduler: &Scheduler,
    processes: &Processes,
    handle: Handle,
) -> CallResult<u64> {
    let process = match handles.resolve(handle)? {
        Resolved::CurrentProcess => return Ok(processes.process_id()),
        Resolved::CurrentThread => return Err(KernelError::InvalidHandle.into()),
        Resolved::Object(object) => match scheduler.objects().get(object)? {
            Object::Process(process) => process,
            Object::Thread(thread) => scheduler
                .thread(thread)?
                .process()
                .ok_or(KernelError::InvalidHandle)?,
            Object::Session
            | Object::Event(_)
            | Object::ReadableEvent(_)
            | Object::SharedMemory(_)
            | Object::TransferMemory(_) => {
                return Err(KernelError::InvalidHandle.into());
            }
        },
    };
    processes
        .get(process)
        .map(crate::process::ProcessInfo::process_id)
        .map_err(|error| SvcError::internal(&error))
}

#[cfg(test)]
mod tests;
