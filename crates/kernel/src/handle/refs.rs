//! Reference counting hooks for the handle table.
//!
//! Horizon objects are reference counted: a handle holds one reference (`KHandleTable::Add` calls
//! `obj->Open`, `KHandleTable::Remove` calls `obj->Close`; `eden/src/core/hle/kernel/k_handle_table.cpp`),
//! and so do in-flight waits. The object is destroyed when the last reference is closed
//! (`eden/src/core/hle/kernel/k_auto_object.h`). Here the count lives outside the handle table,
//! behind [`ObjectRefs`], so that whoever owns the object table decides what "destroy" means; the
//! handle table only reports, from [`HandleTable::remove`](super::HandleTable::remove), whether
//! its close was the last.
//!
//! Divergence from Horizon, listed per `CLAUDE.md`: Horizon objects start with a count of one for
//! their creator, which the creating call closes after inserting a handle. [`RefCounts`] has no
//! creation step: the first `open` creates the entry at one. Callers that want Horizon's
//! creator reference call `open` themselves, and `close` it when done.

use std::collections::BTreeMap;
use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::id::ObjectId;

/// What closing one reference left behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Closed {
    /// Other references remain; this many.
    Remaining(NonZeroU32),
    /// That was the last reference: the object is now unreferenced and its owner should free it.
    Last,
}

/// Why a reference could not be opened or closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RefError {
    /// The count would pass `u32::MAX`.
    #[error("reference count of {object:?} overflowed")]
    Overflow {
        /// The object.
        object: ObjectId,
    },
    /// `close` on an object that has no references.
    #[error("{object:?} has no references to close")]
    NotReferenced {
        /// The object.
        object: ObjectId,
    },
}

/// Where the handle table opens and closes references.
///
/// [`RefCounts`] is the standalone implementation. The object table can implement this instead
/// once objects carry their own count.
pub trait ObjectRefs {
    /// Takes one more reference to `object`.
    fn open(&mut self, object: ObjectId) -> Result<(), RefError>;

    /// Drops one reference to `object` and reports whether it was the last.
    fn close(&mut self, object: ObjectId) -> Result<Closed, RefError>;
}

/// A reference count per object, held in a map ordered by id so iteration and serialisation are
/// deterministic. An object with no references has no entry.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefCounts {
    counts: BTreeMap<ObjectId, NonZeroU32>,
}

impl RefCounts {
    /// An empty set of counts.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// References currently held to `object` (zero if none).
    #[must_use]
    pub fn count(&self, object: ObjectId) -> u32 {
        self.counts.get(&object).map_or(0, |n| n.get())
    }

    /// Number of objects with at least one reference.
    #[must_use]
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Whether no object has a reference.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

impl ObjectRefs for RefCounts {
    fn open(&mut self, object: ObjectId) -> Result<(), RefError> {
        match self.counts.get_mut(&object) {
            Some(count) => {
                *count = count.checked_add(1).ok_or(RefError::Overflow { object })?;
            }
            None => {
                self.counts.insert(object, NonZeroU32::MIN);
            }
        }
        Ok(())
    }

    fn close(&mut self, object: ObjectId) -> Result<Closed, RefError> {
        let Some(count) = self.counts.get_mut(&object) else {
            return Err(RefError::NotReferenced { object });
        };
        if let Some(left) = NonZeroU32::new(count.get() - 1) {
            *count = left;
            Ok(Closed::Remaining(left))
        } else {
            self.counts.remove(&object);
            Ok(Closed::Last)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(n: u32) -> ObjectId {
        use crate::id::SlabId;
        ObjectId::from_parts(n, 1)
    }

    #[test]
    fn open_and_close_count_up_and_down() {
        let mut refs = RefCounts::new();
        assert_eq!(refs.count(obj(0)), 0);
        refs.open(obj(0)).unwrap();
        refs.open(obj(0)).unwrap();
        refs.open(obj(1)).unwrap();
        assert_eq!(
            (refs.count(obj(0)), refs.count(obj(1)), refs.len()),
            (2, 1, 2)
        );
        assert_eq!(refs.close(obj(0)), Ok(Closed::Remaining(NonZeroU32::MIN)));
        assert_eq!(refs.close(obj(0)), Ok(Closed::Last));
        assert_eq!(refs.count(obj(0)), 0);
        assert_eq!(refs.len(), 1);
    }

    #[test]
    fn closing_an_unreferenced_object_is_an_error() {
        let mut refs = RefCounts::new();
        let err = RefError::NotReferenced { object: obj(3) };
        assert_eq!(refs.close(obj(3)), Err(err));
        // Closing past the last reference is also an error, not a wrap.
        refs.open(obj(3)).unwrap();
        assert_eq!(refs.close(obj(3)), Ok(Closed::Last));
        assert_eq!(refs.close(obj(3)), Err(err));
    }

    #[test]
    fn the_count_stops_at_u32_max_instead_of_wrapping() {
        let mut refs = RefCounts::new();
        refs.counts.insert(obj(0), NonZeroU32::MAX);
        assert_eq!(
            refs.open(obj(0)),
            Err(RefError::Overflow { object: obj(0) })
        );
        assert_eq!(refs.count(obj(0)), u32::MAX);
        assert_eq!(
            refs.close(obj(0)),
            Ok(Closed::Remaining(NonZeroU32::new(u32::MAX - 1).unwrap()))
        );
    }
}
