//! Processes: creating one from a description, starting its main thread, and the parts of a
//! process that its threads draw on (thread-local pages, address-space ids). Kernel slice 8
//! (kernel spec, "Port plan").
//!
//! A process is spread over the tables as the kernel spec splits Horizon's `KProcess`:
//!
//! | What | Where |
//! |---|---|
//! | Facts fixed at creation: ids, the core and priority masks, the ideal core, the console it was built for (and so its counter rate) | [`crate::tables::Processes`], one [`ProcessInfo`] per process |
//! | Its state, threads, running-thread count, thread-local pages and scheduled count | the scheduler table, one [`ProcessRecord`] per process ([`crate::sched::Scheduler::process`]) |
//! | Its handles | its [`HandleTable`] |
//! | Its memory | its [`AddressSpace`] |
//!
//! [`create`] builds the tables of a new process; [`run`] starts its main thread
//! (`KProcess::Run`, `eden/src/core/hle/kernel/k_process.cpp` @ cf654052, GPL-3.0-or-later),
//! without the stack, which the caller has already mapped
//! ([`AddressSpace::create_main_thread_stack`]).
//!
//! # Divergences from `KProcess::Run`
//!
//! - **The thread's register state** is Eden's kernel entry convention for a program with no
//!   homebrew argument block (`x0` zero, `x1` the main thread's handle). Eden gives an NRO that
//!   the loader passed an argument block `x0` = the block, `x1` = `-1`; no loader here passes
//!   one. Test: `the_main_thread_starts_at_the_entry_with_its_own_handle`.
//! - **No resource limit**: Eden reserves a thread from the process's limit
//!   (`ThreadCountMax`) and can fail with `ResultLimitReached`; the resource-limits slice
//!   adds it.

pub mod asid;
pub mod description;
pub mod record;
pub mod tls;

use starquake_memory::GuestMemory;

pub use asid::{AsidError, AsidPool};
pub use description::{
    AddressSpaceWidth, Console, KERNEL_PRIORITY_MASK, LEGACY_VIRTUAL_CORE_COUNT,
    ProcessDescription, ProcessError, ProcessInfo, Range, VIRTUAL_CORE_COUNT,
    VIRTUAL_TO_PHYSICAL_CORE, physical_core, physical_mask,
};
pub use record::{ProcessRecord, ProcessState};
pub use tls::{
    CURRENT_THREAD_HANDLE_OFFSET, REGIONS_PER_PAGE, THREAD_LOCAL_REGION_SIZE, ThreadLocalPages,
};

use crate::dispatch::Tables;
use crate::handle::{Handle, HandleError, HandleTable};
use crate::id::{ProcessId, ThreadId};
use crate::memory::AddressSpace;
use crate::result::{CallResult, KernelError, SvcError};
use crate::sched::{Scheduler, UserThread};
use crate::tables::Processes;

/// `Svc::IdealCoreDontCare` (`eden/src/core/hle/kernel/svc_types.h`): no preferred core.
pub const IDEAL_CORE_DONT_CARE: i32 = -1;
/// `Svc::IdealCoreUseProcessValue`: the process's ideal core.
pub const IDEAL_CORE_USE_PROCESS_VALUE: i32 = -2;
/// `Svc::IdealCoreNoUpdate`: keep the thread's ideal core.
pub const IDEAL_CORE_NO_UPDATE: i32 = -3;

/// Why the tables of a new process could not be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CreateError {
    /// The description was refused.
    #[error(transparent)]
    Process(#[from] ProcessError),
    /// Its handle table could not be made.
    #[error("the process's handle table: {0}")]
    Handles(#[from] HandleError),
    /// The scheduler refused the process.
    #[error("the scheduler refused the process: {0}")]
    Scheduler(#[from] crate::sched::SchedError),
}

/// The tables of a new process (`KProcess::Initialize` and `LoadFromMetadata`, the parts that
/// are not memory): its facts in a new [`Processes`] table, a handle table of the size the
/// description asks for, and a scheduler with the process's row in it, in
/// [`ProcessState::Created`]. `address_space` is its memory, created around its code.
pub fn create(
    description: &ProcessDescription,
    address_space: AddressSpace,
) -> Result<Tables, CreateError> {
    let processes = Processes::new(description)?;
    let handles = HandleTable::new(description.handle_table_size)?;
    let mut scheduler = Scheduler::default();
    scheduler.add_process(processes.own())?;
    Ok(Tables {
        handles,
        address_space,
        scheduler,
        processes,
    })
}

/// A started main thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MainThread {
    /// Its row in the scheduler.
    pub thread: ThreadId,
    /// The handle to it in the process's table, which it also finds in `x1`.
    pub handle: Handle,
}

/// Starts the process the tables belong to (`KProcess::Run`, after its stack): creates its main
/// thread at the description's priority on its ideal core, entering at `entry` with
/// `stack_top` as its stack, adds a handle to it, writes that handle into the thread's region
/// ([`CURRENT_THREAD_HANDLE_OFFSET`]) through `memory`, marks the process running and starts
/// the thread. The thread's context is what a core loads to run it
/// ([`Scheduler::thread`]`.context()`); which core runs it is the scheduler's to say
/// ([`Scheduler::update`], then [`Scheduler::schedule`]).
///
/// A process that is not [`ProcessState::Created`] is [`KernelError::InvalidState`], Eden's
/// result, before anything is created. A failure after the thread exists destroys it again, so
/// nothing of it is left.
pub fn run(
    tables: &mut Tables,
    memory: &dyn GuestMemory,
    entry: u64,
    stack_top: u64,
) -> CallResult<MainThread> {
    let Tables {
        handles,
        address_space: space,
        scheduler,
        processes,
    } = tables;
    let process = processes.own();
    let info = processes
        .get(process)
        .map_err(|error| SvcError::internal(&error))?;
    if scheduler.process(process)?.state() != ProcessState::Created {
        return Err(KernelError::InvalidState.into());
    }
    let spec = UserThread {
        entry,
        argument: 0,
        stack_top,
        priority: info.main_thread_priority(),
        virtual_core: info.ideal_core(),
    };
    let (thread, handle) =
        create_thread_with_handle((handles, space, scheduler), process, &spec, memory)?;
    scheduler.context_mut(thread)?.x[1] = u64::from(handle.raw());
    scheduler.mark_running(process)?;
    scheduler.start(thread)?;
    Ok(MainThread { thread, handle })
}

/// Creates a thread of `process` ([`Scheduler::create_user_thread`]) and a handle to it, writes
/// the handle into the thread's region, and closes the creator's reference, so that the handle
/// holds the thread's only one: the steps `CreateThread` and `KProcess::Run` share. When the
/// handle cannot be added, closing the creator's reference destroys the thread again and the
/// handle table's error is returned.
pub(crate) fn create_thread_with_handle(
    (handles, space, scheduler): (&mut HandleTable, &mut AddressSpace, &mut Scheduler),
    process: ProcessId,
    spec: &UserThread,
    memory: &dyn GuestMemory,
) -> CallResult<(ThreadId, Handle)> {
    let (thread, object) = scheduler.create_user_thread(space, process, spec)?;
    let handle = match handles.add(scheduler.objects_mut(), object) {
        Ok(handle) => handle,
        Err(error) => {
            scheduler.close_object(object)?;
            return Err(error.into());
        }
    };
    let tls = scheduler.thread(thread)?.tls_address();
    write_own_handle(memory, tls, handle);
    scheduler.close_object(object)?;
    Ok((thread, handle))
}

/// Writes a new thread's handle into its region (`Write32(tls + 0x110, handle)`, as
/// `CreateThread` and `KProcess::Run` do). Eden's write cannot fail; here guest memory that
/// refuses it is a host failure, logged, and the thread carries on without it, as a console's
/// would with a region the kernel had zeroed.
pub(crate) fn write_own_handle(memory: &dyn GuestMemory, tls: u64, handle: Handle) {
    let Some(address) = tls.checked_add(CURRENT_THREAD_HANDLE_OFFSET) else {
        tracing::error!(tls, "a thread-local region at the top of memory");
        return;
    };
    if let Err(fault) = memory.write(address, &handle.raw().to_le_bytes()) {
        tracing::error!(
            address,
            ?fault,
            "a thread's own handle could not be written"
        );
    }
}

#[cfg(test)]
mod tests;
