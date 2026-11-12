//! `CloseHandle` (0x16).
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_synchronization.cpp` @ cf654052
//! (GPL-3.0-or-later), `CloseHandle`: remove the handle from the calling process's table, and
//! return `ResultInvalidHandle` if the table refuses. `KHandleTable::Remove`
//! (`eden/src/core/hle/kernel/k_handle_table.cpp`) refuses a pseudo-handle, a handle with
//! reserved bits set, and one that names no entry, and closes the entry's reference to its
//! object; [`HandleTable::remove`] is that port.
//!
//! The call declares two tables: the handle table, which loses an entry, and the scheduler,
//! whose object table holds the reference the handle had. When that was the object's last
//! reference its row is freed. Tearing down what the row named (a thread, say) belongs to the
//! slice that builds that kind's lifecycle; see [`crate::object`].

use crate::handle::{Handle, HandleTable};
use crate::result::CallResult;
use crate::tables::Scheduler;

/// Closes `handle` in the caller's handle table, dropping its reference to the object it names.
///
/// [`KernelError::InvalidHandle`](crate::result::KernelError::InvalidHandle) for a handle that
/// names nothing in the table, including both pseudo-handles and a handle already closed.
pub fn close_handle(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
    handle: Handle,
) -> CallResult<()> {
    handles.remove(scheduler.objects_mut(), handle)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::ObjectRefs;
    use crate::id::{SlabId, ThreadId};
    use crate::object::{Object, ObjectError};
    use crate::result::{KernelError, SvcError};

    /// A scheduler holding one object with only a handle's reference, as a creating system call
    /// leaves it, and that handle.
    fn one_handle() -> (HandleTable, Scheduler, Handle) {
        let mut handles = HandleTable::new(4).unwrap();
        let mut scheduler = Scheduler::default();
        let object = scheduler
            .objects_mut()
            .create(Object::Thread(ThreadId::from_parts(0, 1)))
            .unwrap();
        let handle = handles.add(scheduler.objects_mut(), object).unwrap();
        scheduler.objects_mut().close(object).unwrap();
        (handles, scheduler, handle)
    }

    #[test]
    fn closing_the_only_handle_frees_the_entry_and_the_object() {
        let (mut handles, mut scheduler, handle) = one_handle();
        let object = handles.get(handle).unwrap();
        assert_eq!(close_handle(&mut handles, &mut scheduler, handle), Ok(()));
        assert!(handles.is_empty());
        assert_eq!(
            scheduler.objects().get(object),
            Err(ObjectError::Stale { id: object })
        );
    }

    #[test]
    fn closing_one_of_two_handles_keeps_the_object() {
        let (mut handles, mut scheduler, first) = one_handle();
        let object = handles.get(first).unwrap();
        let second = handles.add(scheduler.objects_mut(), object).unwrap();
        assert_eq!(close_handle(&mut handles, &mut scheduler, first), Ok(()));
        assert_eq!(scheduler.objects().refs(object), 1);
        assert_eq!(handles.get(second), Ok(object));
        assert_eq!(close_handle(&mut handles, &mut scheduler, second), Ok(()));
        assert!(scheduler.objects().is_empty());
    }

    #[test]
    fn what_the_table_refuses_is_an_invalid_handle_and_changes_nothing() {
        let (mut handles, mut scheduler, handle) = one_handle();
        let invalid = Err(SvcError::Kernel(KernelError::InvalidHandle));
        // `KHandleTable::Remove`: a pseudo-handle, reserved bits, and no such entry.
        let with_reserved_bit = Handle::from_raw(handle.raw() | 1 << 30);
        let other_linear_id = Handle::from_parts(
            u16::try_from(handle.index()).unwrap(),
            crate::handle::MAX_LINEAR_ID,
        )
        .unwrap();
        for refused in [
            Handle::CURRENT_THREAD,
            Handle::CURRENT_PROCESS,
            Handle::INVALID,
            with_reserved_bit,
            other_linear_id,
        ] {
            assert_ne!(refused, handle);
            assert_eq!(
                close_handle(&mut handles, &mut scheduler, refused),
                invalid,
                "{refused:?}"
            );
            assert_eq!((handles.len(), scheduler.objects().len()), (1, 1));
        }
        // Closing twice: the second close finds nothing.
        assert_eq!(close_handle(&mut handles, &mut scheduler, handle), Ok(()));
        assert_eq!(close_handle(&mut handles, &mut scheduler, handle), invalid);
    }

    #[test]
    fn a_handle_whose_object_has_no_references_is_an_internal_failure_and_stays() {
        // Only broken bookkeeping gets here: the entry's object was closed behind its back.
        let (mut handles, mut scheduler, handle) = one_handle();
        let object = handles.get(handle).unwrap();
        scheduler.objects_mut().close(object).unwrap();
        let expected =
            crate::handle::HandleError::Refs(crate::handle::RefError::NotReferenced { object });
        assert_eq!(
            close_handle(&mut handles, &mut scheduler, handle),
            Err(SvcError::Internal(expected.to_string()))
        );
        assert_eq!(handles.get(handle), Ok(object));
    }
}
