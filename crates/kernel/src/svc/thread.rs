//! Thread system calls: `CreateThread` (0x08), `StartThread` (0x09), `ExitThread` (0x0a),
//! `SleepThread` (0x0b), `GetThreadPriority` (0x0c), `SetThreadPriority` (0x0d),
//! `GetThreadCoreMask` (0x0e), `SetThreadCoreMask` (0x0f) and `GetThreadId` (0x25).
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_thread.cpp` @ cf654052 (GPL-3.0-or-later),
//! with Eden PR #4588's `IsValidVirtualCoreId` (`eden-pr4588/src/core/hle/kernel/svc/svc_thread.cpp`
//! @ e5fb3b97: a process built for Switch 2 may name any of the 64 virtual cores, any other
//! process cores 0-3). The handle lookup is `KHandleTable::GetObject<KThread>`
//! (`eden/src/core/hle/kernel/k_handle_table.h`): the `CurrentThread` pseudo-handle is the
//! calling thread, and anything else must be an entry whose object is a thread. For a thread
//! lookup the `CurrentProcess` pseudo-handle is not special, so it fails like any handle that
//! names no thread.
//!
//! The calling process is the owner of the thread the calling core runs (`GetCurrentProcess`).
//! Its facts come from the [`Processes`] table; the calls that create threads also need its
//! handle table and address space, so it must be the process those tables belong to
//! ([`Processes::own`]); any other is an internal failure.
//!
//! The order of checks is Eden's, argument by argument, and each call's docs list it; a call
//! given two faults at once reports the first.
//!
//! # Divergences
//!
//! Each is pinned by the named test in `svc/thread/lifecycle_tests.rs`.
//!
//! - **`SleepThread` with a positive time does not sleep.** Waiting needs a timer, which
//!   kernel slice 10 brings; until then the call returns at once and changes nothing. Test:
//!   `a_positive_sleep_returns_at_once_and_changes_nothing`.
//! - **`SetThreadCoreMask` does not wait.** When the thread runs on a core its new mask leaves
//!   out, Eden releases the scheduler lock and retries until that core lets it go (or, for a
//!   pinned thread, waits on it). Here the change is made and the call returns; the core finds
//!   it must reschedule ([`Scheduler::needs_scheduling`]). WAIT SEAM (kernel slices 4 and 10):
//!   the retry belongs with the core loop and the wait path. Test:
//!   `changing_the_mask_of_a_running_thread_returns_without_waiting`.
//! - **No resource limit**: `CreateThread` does not reserve from the process's thread limit
//!   (`ThreadCountMax`) and so never fails with `ResultLimitReached`; the resource-limits slice
//!   adds it. **No termination requests**: threads end at once
//!   ([`Scheduler::terminate_thread`]), so `StartThread` never sees one pending and never
//!   returns `ResultTerminationRequested`.

use starquake_memory::GuestMemory;

use crate::handle::{Handle, HandleTable, Resolved};
use crate::id::{ProcessId, ThreadId};
use crate::memory::AddressSpace;
use crate::object::Object;
use crate::process::{
    IDEAL_CORE_DONT_CARE, IDEAL_CORE_NO_UPDATE, IDEAL_CORE_USE_PROCESS_VALUE, ProcessInfo,
    create_thread_with_handle,
};
use crate::result::{CallResult, KernelError, SvcError};
use crate::sched::{
    CoreId, HIGHEST_THREAD_PRIORITY, LOWEST_THREAD_PRIORITY, Priority, SchedError, ThreadExit,
    UserThread,
};
use crate::svc::process::terminate_process;
use crate::tables::{Processes, Scheduler};

/// `Svc::YieldType::WithoutCoreMigration` (`eden/src/core/hle/kernel/svc_types.h`).
pub const YIELD_WITHOUT_CORE_MIGRATION: i64 = 0;
/// `Svc::YieldType::WithCoreMigration`.
pub const YIELD_WITH_CORE_MIGRATION: i64 = -1;
/// `Svc::YieldType::ToAnyThread`.
pub const YIELD_TO_ANY_THREAD: i64 = -2;

/// The thread `handle` names for a call made on `core`: Eden's `GetObject<KThread>`.
///
/// [`KernelError::InvalidHandle`] unless the handle is [`Handle::CURRENT_THREAD`] or names a
/// thread object. The thread system calls of later slices resolve their handles here too.
///
/// The calling thread is the one the scheduler says `core` is running. A core running nothing
/// the scheduler knows of is an internal failure ([`SchedError::NoCurrentThread`]), not a guest
/// error: Eden asserts the current thread is not null.
pub fn thread_from_handle(
    scheduler: &Scheduler,
    handles: &HandleTable,
    core: CoreId,
    handle: Handle,
) -> CallResult<ThreadId> {
    match handles.resolve(handle)? {
        Resolved::CurrentThread => Ok(scheduler
            .current(core)?
            .ok_or(SchedError::NoCurrentThread { core: core.index() })?),
        Resolved::CurrentProcess => Err(KernelError::InvalidHandle.into()),
        Resolved::Object(object) => match scheduler.objects().get(object)? {
            Object::Thread(thread) => Ok(thread),
            Object::Process(_) | Object::Session | Object::Event(_) | Object::ReadableEvent(_) => {
                Err(KernelError::InvalidHandle.into())
            }
        },
    }
}

/// Returns the priority of the thread `handle` names (`KThread::GetPriority`, the effective
/// priority), for a call made on `core`.
pub fn get_thread_priority(
    scheduler: &Scheduler,
    handles: &HandleTable,
    core: CoreId,
    handle: Handle,
) -> CallResult<i32> {
    let thread = thread_from_handle(scheduler, handles, core, handle)?;
    Ok(i32::from(scheduler.thread(thread)?.priority().get()))
}

/// The process that owns the thread `core` runs, and its facts: `GetCurrentProcess`. See the
/// module docs.
pub fn calling_process<'p>(
    scheduler: &Scheduler,
    processes: &'p Processes,
    core: CoreId,
) -> CallResult<(ProcessId, &'p ProcessInfo)> {
    let thread = scheduler
        .current(core)?
        .ok_or(SchedError::NoCurrentThread { core: core.index() })?;
    let process = scheduler
        .thread(thread)?
        .process()
        .ok_or_else(|| SvcError::Internal(format!("thread {thread:?} belongs to no process")))?;
    if process != processes.own() {
        return Err(SvcError::Internal(format!(
            "thread {thread:?} of process {process:?} called with another process's tables"
        )));
    }
    let info = processes
        .get(process)
        .map_err(|error| SvcError::internal(&error))?;
    Ok((process, info))
}

/// A priority a call passed, checked as `SetThreadPriority` and `CreateThread` check it: inside
/// [`HIGHEST_THREAD_PRIORITY`]..=[`LOWEST_THREAD_PRIORITY`], then allowed by the process
/// (`CheckThreadPriority`); [`KernelError::InvalidPriority`] for either.
fn checked_priority(info: &ProcessInfo, priority: i32) -> CallResult<Priority> {
    let in_range = (i32::from(HIGHEST_THREAD_PRIORITY)..=i32::from(LOWEST_THREAD_PRIORITY))
        .contains(&priority);
    if !in_range || !info.check_thread_priority(priority) {
        return Err(KernelError::InvalidPriority.into());
    }
    u8::try_from(priority)
        .ok()
        .and_then(Priority::new)
        .ok_or_else(|| KernelError::InvalidPriority.into())
}

/// What `CreateThread` passes besides the tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreateThread {
    /// Where the thread starts (`func`).
    pub entry: u64,
    /// Its `x0` (`arg`).
    pub argument: u64,
    /// Its stack pointer (Eden's `stack_bottom`, used as the top).
    pub stack_top: u64,
    /// Its priority.
    pub priority: i32,
    /// Its virtual core, or [`IDEAL_CORE_USE_PROCESS_VALUE`].
    pub core_id: i32,
}

/// `CreateThread`: a new thread of the calling process, not started, and a handle to it, which
/// is also written into the thread's region. Checks, in Eden's order:
///
/// 1. [`IDEAL_CORE_USE_PROCESS_VALUE`] becomes the process's ideal core.
/// 2. The core must be a valid virtual core for the process's console, and in the process's
///    core mask: [`KernelError::InvalidCoreId`].
/// 3. The priority must be 0-63 and allowed by the process: [`KernelError::InvalidPriority`].
///
/// Then a full object table is [`KernelError::OutOfResource`], a full handle table
/// [`KernelError::OutOfHandles`], and a thread-local page that cannot be mapped the address
/// space's error; none leaves a thread behind.
pub fn create_thread(
    tables: (&mut HandleTable, &mut AddressSpace, &mut Scheduler),
    processes: &Processes,
    memory: &dyn GuestMemory,
    core: CoreId,
    args: CreateThread,
) -> CallResult<Handle> {
    let (process, info) = calling_process(tables.2, processes, core)?;
    let core_id = if args.core_id == IDEAL_CORE_USE_PROCESS_VALUE {
        i32::from(info.ideal_core())
    } else {
        args.core_id
    };
    if !info.console().is_valid_virtual_core(core_id) {
        return Err(KernelError::InvalidCoreId.into());
    }
    let virtual_core = u8::try_from(core_id).map_err(|_| KernelError::InvalidCoreId)?;
    let bit = 1_u64.checked_shl(u32::from(virtual_core)).unwrap_or(0);
    if bit & info.core_mask() == 0 {
        return Err(KernelError::InvalidCoreId.into());
    }
    let priority = checked_priority(info, args.priority)?;
    let spec = UserThread {
        entry: args.entry,
        argument: args.argument,
        stack_top: args.stack_top,
        priority,
        virtual_core,
    };
    let (_, handle) = create_thread_with_handle(tables, process, &spec, memory)?;
    Ok(handle)
}

/// `StartThread`: starts the thread `handle` names ([`Scheduler::start`]).
/// [`KernelError::InvalidHandle`] for a handle that names no thread, then
/// [`KernelError::InvalidState`] for a thread that is not newly created.
pub fn start_thread(
    handles: &HandleTable,
    scheduler: &mut Scheduler,
    core: CoreId,
    handle: Handle,
) -> CallResult<()> {
    let thread = thread_from_handle(scheduler, handles, core, handle)?;
    scheduler.start(thread)?;
    Ok(())
}

/// `ExitThread`: ends the thread `core` runs. When it was its process's last running thread
/// the process ends too ([`terminate_process`]), and [`ThreadExit::LastOf`] says so.
pub fn exit_thread(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
    core: CoreId,
) -> CallResult<ThreadExit> {
    let exit = scheduler.exit_thread(core)?;
    if let ThreadExit::LastOf(process) = exit {
        terminate_process(handles, scheduler, process)?;
    }
    Ok(exit)
}

/// `SleepThread`: `ns` of [`YIELD_WITHOUT_CORE_MIGRATION`], [`YIELD_WITH_CORE_MIGRATION`] and
/// [`YIELD_TO_ANY_THREAD`] are the scheduler's three yields for the thread `core` runs; any
/// other value below zero does nothing, as in Eden; a positive time does not sleep (a listed
/// divergence).
pub fn sleep_thread(scheduler: &mut Scheduler, core: CoreId, ns: i64) -> CallResult<()> {
    match ns {
        YIELD_WITHOUT_CORE_MIGRATION => scheduler.yield_without_core_migration(core)?,
        YIELD_WITH_CORE_MIGRATION => scheduler.yield_with_core_migration(core)?,
        YIELD_TO_ANY_THREAD => scheduler.yield_to_any_thread(core)?,
        1.. => {
            // WAIT SEAM (kernel slice 10, timers): a sleep for `ns` nanoseconds.
            tracing::debug!(ns, "SleepThread does not sleep yet");
        }
        _ => {}
    }
    Ok(())
}

/// `SetThreadPriority`. Checks, in Eden's order: the priority
/// ([`KernelError::InvalidPriority`], as [`create_thread`] checks it), then the handle
/// ([`KernelError::InvalidHandle`]). Then the thread's base priority changes
/// ([`Scheduler::set_base_priority`]).
pub fn set_thread_priority(
    handles: &HandleTable,
    scheduler: &mut Scheduler,
    processes: &Processes,
    core: CoreId,
    handle: Handle,
    priority: i32,
) -> CallResult<()> {
    let (_, info) = calling_process(scheduler, processes, core)?;
    let priority = checked_priority(info, priority)?;
    let thread = thread_from_handle(scheduler, handles, core, handle)?;
    scheduler.set_base_priority(core, thread, priority)?;
    Ok(())
}

/// `GetThreadCoreMask`: the ideal core and affinity mask of the thread `handle` names, as the
/// guest named them (virtual cores).
pub fn get_thread_core_mask(
    scheduler: &Scheduler,
    handles: &HandleTable,
    core: CoreId,
    handle: Handle,
) -> CallResult<(i32, u64)> {
    let thread = thread_from_handle(scheduler, handles, core, handle)?;
    Ok(scheduler.thread(thread)?.virtual_core_mask())
}

/// `SetThreadCoreMask` with `core_id` and `mask`. Checks, in Eden's order:
///
/// 1. With `core_id` [`IDEAL_CORE_USE_PROCESS_VALUE`]: the process's ideal core and a mask of
///    that core alone, with no further checks. Otherwise:
/// 2. The mask must be within the process's core mask ([`KernelError::InvalidCoreId`]) and
///    not empty ([`KernelError::InvalidCombination`]).
/// 3. A valid virtual core must be in the mask ([`KernelError::InvalidCombination`]); any other
///    `core_id` must be [`IDEAL_CORE_NO_UPDATE`] or [`IDEAL_CORE_DONT_CARE`]
///    ([`KernelError::InvalidCoreId`]).
/// 4. The handle ([`KernelError::InvalidHandle`]).
/// 5. With [`IDEAL_CORE_NO_UPDATE`], the thread's ideal core must be in the mask
///    ([`KernelError::InvalidCombination`], [`Scheduler::set_virtual_core_mask`]).
pub fn set_thread_core_mask(
    handles: &HandleTable,
    scheduler: &mut Scheduler,
    processes: &Processes,
    core: CoreId,
    handle: Handle,
    (core_id, mask): (i32, u64),
) -> CallResult<()> {
    let (_, info) = calling_process(scheduler, processes, core)?;
    let (core_id, mask) = if core_id == IDEAL_CORE_USE_PROCESS_VALUE {
        let ideal = info.ideal_core();
        (
            i32::from(ideal),
            1_u64.checked_shl(u32::from(ideal)).unwrap_or(0),
        )
    } else {
        let process_mask = info.core_mask();
        if mask | process_mask != process_mask {
            return Err(KernelError::InvalidCoreId.into());
        }
        if mask == 0 {
            return Err(KernelError::InvalidCombination.into());
        }
        if info.console().is_valid_virtual_core(core_id) {
            let bit = u32::try_from(core_id)
                .ok()
                .and_then(|shift| 1_u64.checked_shl(shift))
                .unwrap_or(0);
            if bit & mask == 0 {
                return Err(KernelError::InvalidCombination.into());
            }
        } else if core_id != IDEAL_CORE_NO_UPDATE && core_id != IDEAL_CORE_DONT_CARE {
            return Err(KernelError::InvalidCoreId.into());
        }
        (core_id, mask)
    };
    let thread = thread_from_handle(scheduler, handles, core, handle)?;
    scheduler.set_virtual_core_mask(thread, core_id, mask)?;
    // WAIT SEAM (kernel slices 4 and 10): Eden waits here until the thread no longer runs on a
    // core the new mask leaves out (see the module docs).
    Ok(())
}

/// `GetThreadId`: the id the guest knows the thread `handle` names by.
pub fn get_thread_id(
    scheduler: &Scheduler,
    handles: &HandleTable,
    core: CoreId,
    handle: Handle,
) -> CallResult<u64> {
    let thread = thread_from_handle(scheduler, handles, core, handle)?;
    Ok(scheduler.thread(thread)?.thread_id())
}

#[cfg(test)]
mod lifecycle_tests;

#[cfg(test)]
mod tests {
    use starquake_trace::GuestTicks;

    use super::*;
    use crate::handle::ObjectRefs;
    use crate::id::{ProcessId, SlabId};
    use crate::result::SvcError;
    use crate::sched::{CoreSignals, LOWEST_THREAD_PRIORITY, Priority};
    use crate::slab::SlabError;

    struct NoSignals;

    impl CoreSignals for NoSignals {
        fn interrupt(&mut self, _: CoreId) {}

        fn wake(&mut self, _: CoreId) {}
    }

    fn core(index: u8) -> CoreId {
        CoreId::new(index).unwrap()
    }

    /// A thread of priority `priority` bound to core `on`, with a handle to it.
    fn thread_with_handle(
        scheduler: &mut Scheduler,
        handles: &mut HandleTable,
        priority: u8,
        on: CoreId,
    ) -> (ThreadId, Handle) {
        let thread = scheduler
            .create_thread(Priority::new(priority).unwrap(), on)
            .unwrap();
        let object = scheduler
            .objects_mut()
            .create(Object::Thread(thread))
            .unwrap();
        let handle = handles.add(scheduler.objects_mut(), object).unwrap();
        scheduler.objects_mut().close(object).unwrap();
        (thread, handle)
    }

    /// Makes `thread` the thread its core is running.
    fn run(scheduler: &mut Scheduler, thread: ThreadId, on: CoreId) {
        scheduler.start(thread).unwrap();
        scheduler
            .update(GuestTicks(0), None, &mut NoSignals)
            .unwrap();
        assert_eq!(scheduler.schedule(on, &mut NoSignals), Ok(Some(thread)));
    }

    #[test]
    fn a_handle_to_a_thread_reads_that_threads_priority() {
        let mut scheduler = Scheduler::default();
        let mut handles = HandleTable::new(8).unwrap();
        let (_, urgent) = thread_with_handle(&mut scheduler, &mut handles, 28, core(0));
        let (_, lowest) = thread_with_handle(
            &mut scheduler,
            &mut handles,
            LOWEST_THREAD_PRIORITY,
            core(1),
        );
        // The caller's core does not matter for a real handle.
        for caller in [core(0), core(1), core(5)] {
            assert_eq!(
                get_thread_priority(&scheduler, &handles, caller, urgent),
                Ok(28)
            );
            assert_eq!(
                get_thread_priority(&scheduler, &handles, caller, lowest),
                Ok(i32::from(LOWEST_THREAD_PRIORITY))
            );
        }
    }

    #[test]
    fn the_current_thread_pseudo_handle_is_the_thread_the_calling_core_runs() {
        let mut scheduler = Scheduler::default();
        let mut handles = HandleTable::new(8).unwrap();
        let (first, _) = thread_with_handle(&mut scheduler, &mut handles, 30, core(0));
        let (second, _) = thread_with_handle(&mut scheduler, &mut handles, 44, core(3));
        run(&mut scheduler, first, core(0));
        run(&mut scheduler, second, core(3));
        let current = Handle::CURRENT_THREAD;
        assert_eq!(
            get_thread_priority(&scheduler, &handles, core(0), current),
            Ok(30)
        );
        assert_eq!(
            get_thread_priority(&scheduler, &handles, core(3), current),
            Ok(44)
        );
        assert_eq!(
            thread_from_handle(&scheduler, &handles, core(3), current),
            Ok(second)
        );
    }

    #[test]
    fn a_handle_that_names_no_thread_is_invalid() {
        let mut scheduler = Scheduler::default();
        let mut handles = HandleTable::new(8).unwrap();
        let (_, thread) = thread_with_handle(&mut scheduler, &mut handles, 30, core(0));
        let process = scheduler
            .objects_mut()
            .create(Object::Process(ProcessId::from_parts(0, 1)))
            .unwrap();
        let process_handle = handles.add(scheduler.objects_mut(), process).unwrap();
        let invalid = Err(SvcError::Kernel(KernelError::InvalidHandle));
        for not_a_thread in [
            // `GetObject<KThread>` gives the process pseudo-handle no special meaning.
            Handle::CURRENT_PROCESS,
            Handle::INVALID,
            process_handle,
            Handle::from_raw(thread.raw() | 1 << 31),
        ] {
            assert_eq!(
                get_thread_priority(&scheduler, &handles, core(0), not_a_thread),
                invalid,
                "{not_a_thread:?}"
            );
        }
    }

    #[test]
    fn a_core_running_nothing_is_an_internal_failure_not_a_guest_error() {
        let scheduler = Scheduler::default();
        let handles = HandleTable::new(8).unwrap();
        let expected = SchedError::NoCurrentThread { core: 2 };
        assert_eq!(
            get_thread_priority(&scheduler, &handles, core(2), Handle::CURRENT_THREAD),
            Err(SvcError::Internal(expected.to_string()))
        );
        // A core the scheduler does not have, likewise.
        let missing = core(scheduler.core_count());
        let expected = SchedError::NoSuchCore {
            core: missing.index(),
            count: scheduler.core_count(),
        };
        assert_eq!(
            get_thread_priority(&scheduler, &handles, missing, Handle::CURRENT_THREAD),
            Err(SvcError::Internal(expected.to_string()))
        );
    }

    #[test]
    fn an_object_that_names_a_dead_thread_row_is_an_internal_failure() {
        let mut scheduler = Scheduler::default();
        let mut handles = HandleTable::new(8).unwrap();
        // No such row in the thread table: only broken bookkeeping stores this.
        let nowhere = ThreadId::from_parts(9, 9);
        let object = scheduler
            .objects_mut()
            .create(Object::Thread(nowhere))
            .unwrap();
        let handle = handles.add(scheduler.objects_mut(), object).unwrap();
        let expected = SchedError::Slab(SlabError::Stale {
            index: nowhere.index(),
            generation: nowhere.generation(),
        });
        assert_eq!(
            get_thread_priority(&scheduler, &handles, core(0), handle),
            Err(SvcError::Internal(expected.to_string()))
        );
    }
}
