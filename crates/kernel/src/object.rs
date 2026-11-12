//! The object table: what handles name, and how many references each object has.
//!
//! Kernel spec, "A closed set of object kinds": an [`Object`] is an enum, not a trait object,
//! so matches are exhaustive and there are no downcasts. The kinds built so far are the two
//! that are only an id into another table ([`Object::Thread`], [`Object::Process`]) and the
//! client end of a session ([`Object::Session`], whose row in the IPC table, [`crate::ipc`], is
//! keyed by the object's own id), and an event's two ends ([`Object::Event`],
//! [`Object::ReadableEvent`], whose state is the row itself; see [`crate::sync`]), and shared
//! and transfer memory ([`Object::SharedMemory`], [`Object::TransferMemory`], whose state is the
//! row itself too; see [`crate::memory::shared`]). Resource limits get their variant from the
//! slice that builds them, and the compiler then points at every match that must handle it.
//!
//! Rows live in the scheduler table ([`Scheduler::objects`](crate::sched::Scheduler::objects)):
//! the spec puts events and sessions under the scheduler's lock, and a thread's object row
//! names a row of the same table.
//!
//! # Reference counts
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_auto_object.h` and `k_auto_object.cpp` @
//! cf654052 (GPL-3.0-or-later):
//!
//! - `KAutoObject::Create` sets the count to 1, the creator's reference: [`Objects::create`].
//! - `Open` adds one, and fails once the count has reached zero: [`ObjectRefs::open`] on a row
//!   that is gone is [`RefError::NotReferenced`].
//! - `Close` takes one away and destroys the object when it was the last: [`ObjectRefs::close`]
//!   removes the row and reports [`Closed::Last`].
//!
//! A system call that creates an object therefore does what Horizon's does: create it (one
//! reference), add a handle (two), and close the creator's reference (one, held by the
//! handle), so closing the handle destroys the object.
//!
//! Divergences from Eden, each pinned by the named test below:
//! - `Open` on a full count: Eden asserts `cur_ref_count < cur_ref_count + 1`. Here it is
//!   [`RefError::Overflow`] and the count is unchanged. Test:
//!   `the_count_stops_at_the_maximum_instead_of_wrapping`.
//! - `Close` on a count of zero returns silently in Eden. A row with no references does not
//!   exist here, so that is [`RefError::NotReferenced`]. Test:
//!   `a_closed_object_is_gone_and_its_id_is_stale`.
//! - Destroying what the row named (a thread's row when its last reference closes) is not done
//!   here: closing the last reference frees the object's row and returns what it named, and
//!   [`crate::sched::Scheduler::tear_down`] destroys that (a session, a thread). Test:
//!   `closing_the_last_reference_frees_the_row_and_reports_what_it_named`.
//!
//! # A transfer memory's last reference
//!
//! Destroying a transfer memory whose owner's pages are locked unlocks them
//! (`KTransferMemory::Finalize`), which needs the owner's address space, a table that most of
//! the paths that drop a reference do not hold (closing a handle, settling an IPC request).
//! So the one place every last reference goes through, [`Objects::release`], keeps the
//! transfer memory as a follow-up ([`Objects::take_unlocks`]), and the dispatcher, the only
//! code that holds every table, unlocks the pages before it lets the kernel lock go
//! ([`crate::svc::transfer_memory::unlock_released`]). No other call can run in between, so no
//! guest sees the pages locked by an object that is gone. Test:
//! `releasing_a_locked_transfer_memory_leaves_its_unlock_to_do`.

use std::num::NonZeroU32;

use thiserror::Error;

use crate::handle::{Closed, ObjectRefs, RefError};
use crate::id::{ObjectId, ProcessId, ThreadId};
use crate::memory::{SharedMemory, TransferMemory};
use crate::result::SvcError;
use crate::slab::{Slab, SlabError};
use crate::sync::{Event, ReadableEvent};

/// A kernel object: what a handle names. See the module docs for the kinds still to come.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Object {
    /// A thread, by its row in the scheduler's thread table.
    Thread(ThreadId),
    /// A process, by its row in the process table.
    Process(ProcessId),
    /// The client end of a session (Horizon's `KClientSession`): what a handle from
    /// `ConnectToNamedPort` or a reply's moved handle names. The session's row in the IPC table
    /// ([`crate::ipc::Ipc`]) has the object's id as its key, so the variant carries nothing.
    Session,
    /// The writable end of an event (Horizon's `KEvent`): what `SignalEvent` takes.
    Event(Event),
    /// The readable end of an event (Horizon's `KReadableEvent`): what threads wait on.
    ReadableEvent(ReadableEvent),
    /// A shared memory (Horizon's `KSharedMemory`): what `MapSharedMemory` maps.
    SharedMemory(SharedMemory),
    /// A transfer memory (Horizon's `KTransferMemory`): what `CreateTransferMemory` makes.
    TransferMemory(TransferMemory),
}

/// Why an object table operation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ObjectError {
    /// The id names no live object: every reference to it was closed, or it never existed.
    #[error("{id:?} names no live object")]
    Stale {
        /// The id asked for.
        id: ObjectId,
    },
    /// The table is full.
    #[error("the object table is full ({max_slots} objects)")]
    Full {
        /// The table's slot limit.
        max_slots: u32,
    },
    /// Opening or closing a reference failed.
    #[error("reference count: {0}")]
    Refs(#[from] RefError),
    /// The table's bookkeeping is inconsistent. Unreachable through the public API.
    #[error("object table bookkeeping is inconsistent")]
    Corrupt,
}

impl From<ObjectError> for SvcError {
    /// The one place an object table failure becomes what a system call returns. A full table
    /// is Horizon's `ResultOutOfResource` (what `KSlabAllocated::Create` returning null becomes
    /// in the calls that allocate an object, for example `CreateThread` in
    /// `eden/src/core/hle/kernel/svc/svc_thread.cpp`). Every other failure is internal: a system
    /// call reaches an object through a handle, and the handle's reference keeps the object
    /// alive, so a stale id or a broken count is the kernel's own bookkeeping.
    fn from(error: ObjectError) -> Self {
        match error {
            ObjectError::Full { .. } => Self::Kernel(crate::result::KernelError::OutOfResource),
            ObjectError::Stale { .. } | ObjectError::Refs(_) | ObjectError::Corrupt => {
                Self::internal(&error)
            }
        }
    }
}

/// One object and the references held to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row {
    object: Object,
    refs: NonZeroU32,
}

/// The object table. See the module docs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Objects {
    rows: Slab<ObjectId, Row>,
    /// Transfer memories whose last reference went while their owner's pages were locked, in
    /// the order they went: the pages are still to be unlocked (see the module docs).
    unlocks: Vec<TransferMemory>,
}

impl Default for Objects {
    fn default() -> Self {
        Self::new()
    }
}

impl Objects {
    /// An empty table that can hold as many objects as an id can address.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rows: Slab::new(),
            unlocks: Vec::new(),
        }
    }

    /// An empty table that holds at most `max_slots` objects.
    #[must_use]
    pub fn with_max_slots(max_slots: u32) -> Self {
        Self {
            rows: Slab::with_max_slots(max_slots),
            unlocks: Vec::new(),
        }
    }

    fn stale(id: ObjectId) -> ObjectError {
        ObjectError::Stale { id }
    }

    /// Creates an object with one reference, its creator's (`KAutoObject::Create`). The creator
    /// closes that reference when it is done, usually right after adding a handle.
    pub fn create(&mut self, object: Object) -> Result<ObjectId, ObjectError> {
        let row = Row {
            object,
            refs: NonZeroU32::MIN,
        };
        self.rows.insert(row).map_err(|error| match error {
            SlabError::Full { max_slots } => ObjectError::Full { max_slots },
            SlabError::Stale { .. } | SlabError::Corrupt => ObjectError::Corrupt,
        })
    }

    /// The object `id` names.
    pub fn get(&self, id: ObjectId) -> Result<Object, ObjectError> {
        self.rows
            .get(id)
            .map(|row| row.object)
            .map_err(|_| Self::stale(id))
    }

    /// Replaces what `id` names, keeping its references, and returns what it named before: how
    /// an object whose state is its row changes (an event being signalled).
    pub fn replace(&mut self, id: ObjectId, object: Object) -> Result<Object, ObjectError> {
        let row = self.rows.get_mut(id).map_err(|_| Self::stale(id))?;
        Ok(std::mem::replace(&mut row.object, object))
    }

    /// References currently held to `id` (zero if it names no live object).
    #[must_use]
    pub fn refs(&self, id: ObjectId) -> u32 {
        self.rows.get(id).map_or(0, |row| row.refs.get())
    }

    /// Number of live objects.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether there are no live objects.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Drops one reference to `id`. If it was the last, the object's row is freed and the
    /// object it held is returned, so the caller can tear down what it named. A transfer
    /// memory whose owner's pages are locked is kept for [`Objects::take_unlocks`] as well.
    pub fn release(&mut self, id: ObjectId) -> Result<Option<Object>, ObjectError> {
        let row = self.rows.get_mut(id).map_err(|_| Self::stale(id))?;
        if let Some(left) = NonZeroU32::new(row.refs.get() - 1) {
            row.refs = left;
            return Ok(None);
        }
        let row = self.rows.remove(id).map_err(|_| ObjectError::Corrupt)?;
        if let Object::TransferMemory(transfer) = row.object
            && transfer.is_locked()
        {
            self.unlocks.push(transfer);
        }
        Ok(Some(row.object))
    }

    /// Whether a transfer memory is gone whose owner's pages are still locked.
    #[must_use]
    pub fn has_unlocks(&self) -> bool {
        !self.unlocks.is_empty()
    }

    /// The transfer memories that are gone while their owner's pages were locked, oldest
    /// first, and forgets them: the caller unlocks the pages
    /// ([`crate::svc::transfer_memory::unlock_released`]).
    pub fn take_unlocks(&mut self) -> Vec<TransferMemory> {
        std::mem::take(&mut self.unlocks)
    }
}

impl ObjectRefs for Objects {
    fn open(&mut self, object: ObjectId) -> Result<(), RefError> {
        let row = self
            .rows
            .get_mut(object)
            .map_err(|_| RefError::NotReferenced { object })?;
        row.refs = row
            .refs
            .checked_add(1)
            .ok_or(RefError::Overflow { object })?;
        Ok(())
    }

    fn close(&mut self, object: ObjectId) -> Result<Closed, RefError> {
        let gone = RefError::NotReferenced { object };
        match self.release(object).map_err(|_| gone)? {
            Some(_) => Ok(Closed::Last),
            None => NonZeroU32::new(self.refs(object))
                .map(Closed::Remaining)
                .ok_or(gone),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::{Handle, HandleError, HandleTable};
    use crate::id::SlabId;
    use crate::result::KernelError;

    fn thread(n: u32) -> Object {
        Object::Thread(ThreadId::from_parts(n, 1))
    }

    #[test]
    fn a_new_object_has_its_creators_reference() {
        let mut objects = Objects::new();
        assert!(objects.is_empty());
        let a = objects.create(thread(7)).unwrap();
        let b = objects
            .create(Object::Process(ProcessId::from_parts(3, 9)))
            .unwrap();
        assert_eq!(objects.get(a), Ok(thread(7)));
        assert_eq!(
            objects.get(b),
            Ok(Object::Process(ProcessId::from_parts(3, 9)))
        );
        assert_eq!((objects.refs(a), objects.refs(b), objects.len()), (1, 1, 2));
    }

    #[test]
    fn open_and_close_count_up_and_down() {
        let mut objects = Objects::new();
        let id = objects.create(thread(0)).unwrap();
        objects.open(id).unwrap();
        objects.open(id).unwrap();
        assert_eq!(objects.refs(id), 3);
        assert_eq!(
            objects.close(id),
            Ok(Closed::Remaining(NonZeroU32::new(2).unwrap()))
        );
        assert_eq!(objects.close(id), Ok(Closed::Remaining(NonZeroU32::MIN)));
        assert_eq!(objects.get(id), Ok(thread(0)));
        assert_eq!(objects.close(id), Ok(Closed::Last));
    }

    #[test]
    fn closing_the_last_reference_frees_the_row_and_reports_what_it_named() {
        let mut objects = Objects::new();
        let id = objects.create(thread(5)).unwrap();
        objects.open(id).unwrap();
        assert_eq!(objects.release(id), Ok(None));
        assert_eq!(objects.release(id), Ok(Some(thread(5))));
        assert!(objects.is_empty());
        // Nothing else was torn down: the object table holds no thread rows to destroy.
        assert_eq!(objects.release(id), Err(ObjectError::Stale { id }));
    }

    #[test]
    fn a_closed_object_is_gone_and_its_id_is_stale() {
        let mut objects = Objects::new();
        let id = objects.create(thread(1)).unwrap();
        assert_eq!(objects.close(id), Ok(Closed::Last));
        let gone = RefError::NotReferenced { object: id };
        assert_eq!(objects.get(id), Err(ObjectError::Stale { id }));
        assert_eq!(objects.refs(id), 0);
        assert_eq!(objects.open(id), Err(gone));
        assert_eq!(objects.close(id), Err(gone));
        // The slot is reused under a new generation; the old id still names nothing.
        let next = objects.create(thread(2)).unwrap();
        assert_eq!(next.index(), id.index());
        assert_ne!(next, id);
        assert_eq!(objects.get(id), Err(ObjectError::Stale { id }));
        assert_eq!(objects.get(next), Ok(thread(2)));
    }

    #[test]
    fn the_count_stops_at_the_maximum_instead_of_wrapping() {
        let mut objects = Objects::new();
        let id = objects.create(thread(0)).unwrap();
        objects.rows.get_mut(id).unwrap().refs = NonZeroU32::MAX;
        assert_eq!(objects.open(id), Err(RefError::Overflow { object: id }));
        assert_eq!(objects.refs(id), u32::MAX);
        assert_eq!(
            objects.close(id),
            Ok(Closed::Remaining(NonZeroU32::new(u32::MAX - 1).unwrap()))
        );
    }

    #[test]
    fn replacing_an_object_keeps_its_references_and_a_stale_id_is_refused() {
        let mut objects = Objects::new();
        let id = objects.create(thread(1)).unwrap();
        objects.open(id).unwrap();
        assert_eq!(objects.replace(id, thread(2)), Ok(thread(1)));
        assert_eq!((objects.get(id), objects.refs(id)), (Ok(thread(2)), 2));
        objects.release(id).unwrap();
        objects.release(id).unwrap();
        assert_eq!(
            objects.replace(id, thread(3)),
            Err(ObjectError::Stale { id })
        );
        assert!(objects.is_empty());
    }

    #[test]
    fn a_full_table_refuses_at_its_limit() {
        let mut objects = Objects::with_max_slots(2);
        objects.create(thread(0)).unwrap();
        let second = objects.create(thread(1)).unwrap();
        assert_eq!(
            objects.create(thread(2)),
            Err(ObjectError::Full { max_slots: 2 })
        );
        // Freeing one makes room for exactly one.
        assert_eq!(objects.close(second), Ok(Closed::Last));
        objects.create(thread(3)).unwrap();
        assert_eq!(
            objects.create(thread(4)),
            Err(ObjectError::Full { max_slots: 2 })
        );
    }

    #[test]
    fn a_handle_keeps_its_object_alive_until_it_is_closed() {
        // What a creating system call does, as Horizon's do: create, add a handle, close the
        // creator's reference.
        let mut objects = Objects::new();
        let mut handles = HandleTable::new(4).unwrap();
        let id = objects.create(thread(9)).unwrap();
        let handle = handles.add(&mut objects, id).unwrap();
        assert_eq!(objects.refs(id), 2);
        assert_eq!(objects.close(id), Ok(Closed::Remaining(NonZeroU32::MIN)));

        assert_eq!(handles.get(handle), Ok(id));
        assert_eq!(objects.get(id), Ok(thread(9)));
        let removed = handles.remove(&mut objects, handle).unwrap();
        assert_eq!((removed.object, removed.closed), (id, Closed::Last));
        assert_eq!(objects.get(id), Err(ObjectError::Stale { id }));
        assert!(objects.is_empty() && handles.is_empty());
    }

    #[test]
    fn a_handle_cannot_be_added_for_an_object_that_is_gone() {
        let mut objects = Objects::new();
        let mut handles = HandleTable::new(4).unwrap();
        let id = objects.create(thread(0)).unwrap();
        assert_eq!(objects.close(id), Ok(Closed::Last));
        assert_eq!(
            handles.add(&mut objects, id),
            Err(HandleError::Refs(RefError::NotReferenced { object: id }))
        );
        assert!(handles.is_empty());
        assert_eq!(handles.get(Handle::INVALID).ok(), None);
    }

    #[test]
    fn releasing_a_locked_transfer_memory_leaves_its_unlock_to_do() {
        use crate::memory::SvcPermission;
        let owner = ProcessId::from_parts(0, 1);
        let lent = TransferMemory::new(owner, 0x20_0000, 0x3000, SvcPermission::None);
        let locked = lent.with_pages_locked();
        let mut objects = Objects::new();
        // Pages never locked (a creation that failed before the lock): nothing to undo.
        let never = objects.create(Object::TransferMemory(lent)).unwrap();
        assert_eq!(
            objects.release(never),
            Ok(Some(Object::TransferMemory(lent)))
        );
        assert!(!objects.has_unlocks());
        // Locked, and a reference left: nothing yet.
        let id = objects.create(Object::TransferMemory(locked)).unwrap();
        objects.open(id).unwrap();
        assert_eq!(objects.release(id), Ok(None));
        assert!(!objects.has_unlocks());
        // The last reference, by a handle's close: the unlock waits for whoever holds the
        // owner's address space.
        assert_eq!(objects.close(id), Ok(Closed::Last));
        assert!(objects.has_unlocks());
        // Other kinds leave nothing to do.
        let event = objects.create(thread(1)).unwrap();
        objects.release(event).unwrap();
        assert_eq!(objects.take_unlocks(), vec![locked]);
        assert!(!objects.has_unlocks());
        assert_eq!(objects.take_unlocks(), Vec::new());
    }

    #[test]
    fn a_full_table_is_out_of_resource_and_every_other_failure_is_internal() {
        let id = ObjectId::from_parts(4, 2);
        assert_eq!(
            SvcError::from(ObjectError::Full { max_slots: 2 }),
            SvcError::Kernel(KernelError::OutOfResource)
        );
        for error in [
            ObjectError::Stale { id },
            ObjectError::Refs(RefError::NotReferenced { object: id }),
            ObjectError::Corrupt,
        ] {
            assert_eq!(SvcError::from(error), SvcError::Internal(error.to_string()));
        }
        assert_eq!(
            ObjectError::Full { max_slots: 2 }.to_string(),
            "the object table is full (2 objects)"
        );
    }
}
