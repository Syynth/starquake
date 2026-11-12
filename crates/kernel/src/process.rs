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
//!
//! # Divergences from `KProcess::Initialize`
//!
//! - **No process-local region.** Eden's `KProcess::Initialize` reserves a thread-local region
//!   for the process itself and zeroes it before any thread exists
//!   (`CreateThreadLocalRegion(&m_plr_address)` then `ZeroBlock(m_plr_address,
//!   Svc::ThreadLocalRegionSize)`, `eden/src/core/hle/kernel/k_process.cpp` @ cf654052, lines
//!   188-193; both of Eden's callers pass `is_real = true`). Here nothing reserves it, so the
//!   main thread's region is the first region of the first thread-local page where Eden's is
//!   the second (page + `0x200`), and each process's count of regions in use is one lower than
//!   Eden's. Nothing reads the process-local region yet; porting it is one reservation in
//!   [`run`] before the main thread is created. PROVISIONAL. Test:
//!   `the_main_thread_takes_the_first_region_because_no_process_local_region_is_reserved`.
//!
//! # One process per set of tables (PROVISIONAL)
//!
//! [`create`] builds a fresh [`Scheduler`] and a fresh [`Processes`] table for each process, and
//! the system calls take the calling thread's process to be [`Processes::own`] (any other is an
//! internal failure). A second `create` would also start its thread ids at
//! [`crate::sched::FIRST_THREAD_ID`] and its process id at
//! [`crate::tables::FIRST_USER_PROCESS_ID`] again. This is narrower than the kernel spec
//! (`docs/specs/kernel.md`, rule 2, "Nothing assumes a single process", and its tables: one
//! scheduler holding every thread). The replacement adds a process to the existing tables: a
//! new row in the one [`Processes`] table and the one [`Scheduler`], with a handle table and an
//! address space of its own, keyed by process. `docs/architecture.md` lists it with the other
//! provisional parts.

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
/// result, before anything is created. A failure while the thread and its handle are made
/// leaves nothing of them ([`create_thread_with_handle`]). A failure after that (setting `x1`,
/// marking the process running, starting the thread) is undone as Eden's `ON_RESULT_FAILURE_2`
/// blocks in `KProcess::Run` undo it ([`undo_run`]): the thread ends, its handle is closed,
/// which destroys it, and the process is [`ProcessState::Created`] again. Each of those later
/// steps fails only when the kernel's own bookkeeping is broken.
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
    let started = (|| -> CallResult<()> {
        scheduler.context_mut(thread)?.x[1] = u64::from(handle.raw());
        scheduler.mark_running(process)?;
        scheduler.start(thread)?;
        Ok(())
    })();
    if let Err(error) = started {
        if let Err(undo) = undo_run(handles, scheduler, process, MainThread { thread, handle }) {
            tracing::error!(%undo, "a main thread that failed to start could not be undone");
        }
        return Err(error);
    }
    Ok(MainThread { thread, handle })
}

/// Undoes what [`run`] did after the main thread and its handle were made, when a later step
/// fails (Eden's `ON_RESULT_FAILURE_2` blocks in `KProcess::Run`: `FinalizeHandleTable`, and
/// `ChangeState(state)` back): the thread ends ([`Scheduler::terminate_thread`], which also
/// drops the reference it holds to itself if it was started), its handle is closed, which
/// destroys it and frees its region, and a process marked running is
/// [`ProcessState::Created`] again. Every step runs even if an earlier one failed; the first
/// failure is returned.
pub(crate) fn undo_run(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
    process: ProcessId,
    main: MainThread,
) -> CallResult<()> {
    let ended = scheduler.terminate_thread(main.thread).map(|_| ());
    let closed = crate::svc::handle::remove_handle(handles, scheduler, main.handle);
    let state = match scheduler.process(process) {
        Ok(record) if record.state() == ProcessState::Running => {
            scheduler.unmark_running(process).map_err(SvcError::from)
        }
        Ok(_) => Ok(()),
        Err(error) => Err(error.into()),
    };
    ended?;
    closed?;
    state
}

/// Creates a thread of `process` ([`Scheduler::create_user_thread`]) and a handle to it, zeroes
/// the thread's region and writes the handle into it ([`initialise_region`]), and closes the
/// creator's reference, so that the handle holds the thread's only one: the steps
/// `CreateThread` and `KProcess::Run` share. When the handle cannot be added, closing the
/// creator's reference destroys the thread again and the handle table's error is returned.
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
    if let Err(error) = initialise_region(memory, tls, handle) {
        tracing::error!(%error, "a new thread's region was not written");
    }
    scheduler.close_object(object)?;
    Ok((thread, handle))
}

/// Bytes of one thread-local region, as a buffer length.
#[expect(
    clippy::cast_possible_truncation,
    reason = "THREAD_LOCAL_REGION_SIZE is 0x200, which fits in a usize of any width"
)]
const REGION_BYTES: usize = THREAD_LOCAL_REGION_SIZE as usize;

/// Why a new thread's region could not be written. Eden's writes cannot fail; here either is a
/// host failure, logged, and the thread carries on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum RegionError {
    /// The region would run past the top of the address range.
    #[error("a thread-local region at {tls:#x} runs past the top of memory")]
    TopOfMemory {
        /// The region's address.
        tls: u64,
    },
    /// Guest memory refused the write.
    #[error("guest memory refused the thread-local region at {tls:#x}: {fault}")]
    Refused {
        /// The region's address.
        tls: u64,
        /// What the memory said.
        fault: starquake_memory::Fault,
    },
}

/// Zeroes a new thread's whole region and writes its own handle into it, in one write: Eden's
/// `KThread::Initialize` zeroes the region it reserves (`ZeroBlock(m_tls_address,
/// Svc::ThreadLocalRegionSize)`, `eden/src/core/hle/kernel/k_thread.cpp` @ cf654052, lines
/// 228-229), then `CreateThread` and `KProcess::Run` write the handle
/// (`Write32(tls + 0x110, handle)`, [`CURRENT_THREAD_HANDLE_OFFSET`]). A region reused from an
/// ended thread so keeps nothing of it.
pub(crate) fn initialise_region(
    memory: &dyn GuestMemory,
    tls: u64,
    handle: Handle,
) -> Result<(), RegionError> {
    if tls.checked_add(THREAD_LOCAL_REGION_SIZE).is_none() {
        return Err(RegionError::TopOfMemory { tls });
    }
    let mut region = [0_u8; REGION_BYTES];
    let bytes = handle.raw().to_le_bytes();
    let start = usize::try_from(CURRENT_THREAD_HANDLE_OFFSET).unwrap_or(REGION_BYTES);
    if let Some(slot) = start
        .checked_add(bytes.len())
        .and_then(|end| region.get_mut(start..end))
    {
        slot.copy_from_slice(&bytes);
    }
    memory
        .write(tls, &region)
        .map_err(|fault| RegionError::Refused { tls, fault })
}

#[cfg(test)]
mod tests;
