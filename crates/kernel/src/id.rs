//! Typed ids for kernel objects (threads, processes, objects), used in place of pointers.
//!
//! Every id is a slot index plus a generation, minted by a [`Slab`](crate::slab::Slab). Two ids
//! of different kinds are different types, so a thread id can't be passed where a process id is
//! expected, and an id whose row was freed is rejected by the slab instead of reaching a
//! different row. The design is the kernel spec's ("Tables behind a dispatcher"); Horizon itself
//! uses raw pointers here, so there is no reference source for the id layout. It is starquake's
//! own (a guess in the sense of `CLAUDE.md`): nothing guest-visible depends on it, because guests
//! only ever see handles, see [`crate::handle`].

use core::fmt::Debug;

use serde::{Deserialize, Serialize};

/// An id that names a row of a [`Slab`](crate::slab::Slab): a slot index and a generation.
///
/// Implemented by the id types below. Ids are plain data; nothing is checked until a slab sees
/// one, and an id from another slab, or a hand-built one, is simply stale.
pub trait SlabId: Copy + Eq + Ord + Debug {
    /// Builds an id from its parts.
    fn from_parts(index: u32, generation: u32) -> Self;

    /// The slot index.
    fn index(self) -> u32;

    /// The generation of the slot this id was minted for.
    fn generation(self) -> u32;
}

/// Declares an id newtype and its [`SlabId`] impl.
macro_rules! typed_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name {
            index: u32,
            generation: u32,
        }

        impl SlabId for $name {
            fn from_parts(index: u32, generation: u32) -> Self {
                Self { index, generation }
            }

            fn index(self) -> u32 {
                self.index
            }

            fn generation(self) -> u32 {
                self.generation
            }
        }
    };
}

typed_id! {
    /// Names a guest thread in the scheduler's thread table.
    ThreadId
}

typed_id! {
    /// Names a process in the process table.
    ProcessId
}

typed_id! {
    /// Names a kernel object (event, session, memory block and so on) in the object table.
    /// Handles refer to objects by this id.
    ObjectId
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_round_trip() {
        let id = ThreadId::from_parts(7, 9);
        assert_eq!((id.index(), id.generation()), (7, 9));
        let id = ProcessId::from_parts(u32::MAX, 0);
        assert_eq!((id.index(), id.generation()), (u32::MAX, 0));
        let id = ObjectId::from_parts(0, u32::MAX);
        assert_eq!((id.index(), id.generation()), (0, u32::MAX));
    }

    #[test]
    fn ids_order_by_index_then_generation() {
        assert!(ObjectId::from_parts(1, 9) < ObjectId::from_parts(2, 0));
        assert!(ObjectId::from_parts(2, 0) < ObjectId::from_parts(2, 1));
    }

    #[test]
    fn serialise_round_trip_keeps_both_fields() {
        // Distinct values, so a swap of the two serialised fields shows up.
        let id = ObjectId::from_parts(3, 0x0102_0304);
        let bytes = postcard::to_allocvec(&id).unwrap();
        let back: ObjectId = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, id);
    }
}
