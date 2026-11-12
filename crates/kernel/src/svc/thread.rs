//! Thread system calls that only read the scheduler. So far: `GetThreadPriority` (0x0c).
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_thread.cpp` @ cf654052 (GPL-3.0-or-later),
//! `GetThreadPriority`: get the thread from its handle (`ResultInvalidHandle` if the handle does
//! not name a thread), and return its priority. The handle lookup is
//! `KHandleTable::GetObject<KThread>` (`eden/src/core/hle/kernel/k_handle_table.h`): the
//! `CurrentThread` pseudo-handle is the calling thread, and anything else must be an entry whose
//! object is a thread. For a thread lookup the `CurrentProcess` pseudo-handle is not special, so
//! it fails like any handle that names no thread.

use crate::handle::{Handle, HandleTable, Resolved};
use crate::id::ThreadId;
use crate::object::Object;
use crate::result::{CallResult, KernelError};
use crate::sched::{CoreId, SchedError};
use crate::tables::Scheduler;

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
            Object::Process(_) => Err(KernelError::InvalidHandle.into()),
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
