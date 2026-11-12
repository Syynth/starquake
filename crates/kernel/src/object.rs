//! The object table: what handles name, and how many references each object has.
//!
//! Kernel spec, "A closed set of object kinds": an [`Object`] is an enum, not a trait object,
//! so matches are exhaustive and there are no downcasts. The kinds built so far are the two
//! that are only an id into another table ([`Object::Thread`], [`Object::Process`]) and the
//! client end of a session ([`Object::Session`], whose row in the IPC table, [`crate::ipc`], is
//! keyed by the object's own id); events, shared and transfer memory and resource limits get
//! their variants from the slices that build them, and the compiler then points at every match
//! that must handle them.
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

use std::num::NonZeroU32;

use thiserror::Error;

use crate::handle::{Closed, ObjectRefs, RefError};
use crate::id::{ObjectId, ProcessId, ThreadId};
use crate::result::SvcError;
use crate::slab::{Slab, SlabError};

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
        Self { rows: Slab::new() }
    }

    /// An empty table that holds at most `max_slots` objects.
    #[must_use]
    pub fn with_max_slots(max_slots: u32) -> Self {
        Self {
            rows: Slab::with_max_slots(max_slots),
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
    /// object it held is returned, so the caller can tear down what it named.
    pub fn release(&mut self, id: ObjectId) -> Result<Option<Object>, ObjectError> {
        let row = self.rows.get_mut(id).map_err(|_| Self::stale(id))?;
        if let Some(left) = NonZeroU32::new(row.refs.get() - 1) {
            row.refs = left;
            return Ok(None);
        }
        let row = self.rows.remove(id).map_err(|_| ObjectError::Corrupt)?;
        Ok(Some(row.object))
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
