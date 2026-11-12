//! Generation-checked slabs: the storage behind every kernel table.
//!
//! A [`Slab<Id, T>`] stores rows of `T` and hands out typed ids ([`SlabId`]). An id carries the
//! generation of the slot it was minted for; freeing a row bumps the slot's generation, so an id
//! to a freed row, or to a row freed and then reused by another value, is an [`SlabError::Stale`],
//! never somebody else's row. This is the kernel spec's replacement for Horizon's intrusive
//! pointers (`docs/specs/kernel.md`, "Tables behind a dispatcher"); Horizon has no equivalent to
//! port from, so the layout below is starquake's own.
//!
//! Design choices (all starquake's own, none observable by a guest):
//!
//! - **Slot reuse is LIFO and deterministic**: the most recently freed slot is reused first, and a
//!   fresh slot is appended only when none is free. The same sequence of inserts and removes
//!   always yields the same ids, which replay needs. (The handle table makes the same choice,
//!   following Horizon's free list; see [`crate::handle`].)
//! - **Generations start at 1**, so an all-zero id never names a row.
//! - **A slot whose generation would overflow is retired** and never reused, rather than wrapping
//!   to a generation that an old id could still hold.
//! - **Capacity is configuration**: [`Slab::with_max_slots`] bounds the number of slots (the
//!   default is `u32::MAX`, the most an id can address). A full slab is [`SlabError::Full`].
//! - **Two kinds of lookup.** An id that came from outside the table (a handle resolved for a
//!   guest, an id a caller passes in) may be stale, and [`Slab::get`] says so with a
//!   [`SlabError::Stale`] the caller turns into a result. An id the kernel stored itself (a queue
//!   link, a core's record of the thread it runs) is live by construction; if it is not, the
//!   kernel's bookkeeping is broken. [`Slab::stored`] and [`Slab::stored_mut`] are the lookups
//!   for those, and fail with the one fixed [`StoredIdStale`]. The scheduler is the one table
//!   that uses them today: its `SchedError` converts [`StoredIdStale`] into `SchedError::Corrupt`,
//!   so such a lookup there is a bare `?`. `HandleError` and `ObjectError` have no such
//!   conversion; another table adds its own when it adopts the lookup.
//! - **Serialisation validates**: the whole structure derives `serde`, and deserialising checks
//!   every invariant (free list in range, vacant and duplicate-free; generations non-zero; counts
//!   consistent), so a corrupt snapshot is an error instead of a later wrong row.

use core::marker::PhantomData;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::id::SlabId;

/// The generation a fresh slot starts with. Zero is never issued.
const FIRST_GENERATION: u32 = 1;

/// Why a slab operation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SlabError {
    /// The id names no live row: it was freed (and possibly reused by another value), it never
    /// existed, or it came from a different slab.
    #[error("stale or unknown id (index {index}, generation {generation})")]
    Stale {
        /// Slot index in the id.
        index: u32,
        /// Generation in the id.
        generation: u32,
    },
    /// Every slot the slab may hold is in use.
    #[error("slab is full ({max_slots} slots)")]
    Full {
        /// The configured slot limit.
        max_slots: u32,
    },
    /// The slab's own bookkeeping is inconsistent. Unreachable through the public API; it exists
    /// so that a slab built from damaged data fails instead of panicking.
    #[error("slab bookkeeping is inconsistent")]
    Corrupt,
}

/// An id the kernel itself stored names no live row: its bookkeeping is inconsistent. Returned
/// by [`Slab::stored`] and [`Slab::stored_mut`]. Never something a guest causes; Horizon follows
/// a raw pointer here and has no such case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("an id the kernel stored names no live row")]
pub struct StoredIdStale;

/// One storage slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Slot<T> {
    /// Free, and on the free stack. The generation is the one the next occupant is issued with.
    Vacant { generation: u32 },
    /// Holds a row.
    Occupied { generation: u32, value: T },
    /// Free but never reused, because its generation was exhausted.
    Retired,
}

/// Generation-checked storage with typed ids. See the module docs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SlabData<T>", bound(deserialize = "T: Deserialize<'de>"))]
pub struct Slab<Id, T> {
    slots: Vec<Slot<T>>,
    /// Indices of vacant slots, most recently freed last.
    free: Vec<u32>,
    max_slots: u32,
    /// Number of occupied slots.
    live: u32,
    #[serde(skip)]
    _id: PhantomData<fn() -> Id>,
}

/// The serialised form of a [`Slab`], and what deserialisation validates.
#[derive(Deserialize)]
struct SlabData<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    max_slots: u32,
    live: u32,
}

/// Why serialised slab data was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SlabLayoutError {
    /// More slots than the configured limit.
    #[error("{slots} slots exceed the limit of {max_slots}")]
    TooManySlots {
        /// Slots present.
        slots: usize,
        /// The configured limit.
        max_slots: u32,
    },
    /// A slot has generation zero, which is never issued.
    #[error("slot {index} has generation zero")]
    ZeroGeneration {
        /// The slot.
        index: usize,
    },
    /// The free stack is damaged.
    #[error("free stack is damaged: {0}")]
    FreeStack(#[from] FreeStackError),
    /// The number of vacant slots, retired slots and free-stack entries disagree, or the live
    /// count is wrong.
    #[error("counts disagree with the slots")]
    Counts,
}

/// Why a free stack failed validation. Shared by [`Slab`] and the handle table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum FreeStackError {
    /// An entry names a slot that does not exist.
    #[error("entry {index} is out of range")]
    OutOfRange {
        /// The offending entry.
        index: u32,
    },
    /// An entry names a slot that is occupied.
    #[error("entry {index} names an occupied slot")]
    NotVacant {
        /// The offending entry.
        index: u32,
    },
    /// The same slot appears twice.
    #[error("entry {index} appears twice")]
    Duplicate {
        /// The offending entry.
        index: u32,
    },
}

/// Checks that every entry of a free stack names a distinct, existing, vacant slot.
///
/// The visited set is sized by the slots actually present, so a hostile stack can't make this
/// run longer than the stack itself.
pub(crate) fn check_free_stack<I: Copy + Into<u32>>(
    free: &[I],
    slot_count: usize,
    is_vacant: impl Fn(usize) -> bool,
) -> Result<(), FreeStackError> {
    let mut seen = vec![false; slot_count];
    for &entry in free {
        let index: u32 = entry.into();
        let at = index as usize;
        let Some(seen_slot) = seen.get_mut(at) else {
            return Err(FreeStackError::OutOfRange { index });
        };
        if !is_vacant(at) {
            return Err(FreeStackError::NotVacant { index });
        }
        if *seen_slot {
            return Err(FreeStackError::Duplicate { index });
        }
        *seen_slot = true;
    }
    Ok(())
}

impl<Id, T> TryFrom<SlabData<T>> for Slab<Id, T> {
    type Error = SlabLayoutError;

    fn try_from(data: SlabData<T>) -> Result<Self, Self::Error> {
        let SlabData {
            slots,
            free,
            max_slots,
            live,
        } = data;
        if slots.len() > max_slots as usize {
            return Err(SlabLayoutError::TooManySlots {
                slots: slots.len(),
                max_slots,
            });
        }
        let mut occupied = 0_usize;
        let mut vacant = 0_usize;
        for (index, slot) in slots.iter().enumerate() {
            match slot {
                Slot::Vacant { generation } | Slot::Occupied { generation, .. }
                    if *generation == 0 =>
                {
                    return Err(SlabLayoutError::ZeroGeneration { index });
                }
                Slot::Vacant { .. } => vacant += 1,
                Slot::Occupied { .. } => occupied += 1,
                Slot::Retired => {}
            }
        }
        check_free_stack(&free, slots.len(), |at| {
            matches!(slots.get(at), Some(Slot::Vacant { .. }))
        })?;
        // Every vacant slot is on the stack exactly once (the helper rules out repeats and
        // non-vacant entries, so equal counts mean no vacant slot is missing).
        if vacant != free.len() || usize::try_from(live) != Ok(occupied) {
            return Err(SlabLayoutError::Counts);
        }
        Ok(Self {
            slots,
            free,
            max_slots,
            live,
            _id: PhantomData,
        })
    }
}

impl<Id, T> Default for Slab<Id, T> {
    fn default() -> Self {
        Self::with_max_slots(u32::MAX)
    }
}

impl<Id, T> Slab<Id, T> {
    /// An empty slab that can hold as many rows as an id can address.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty slab that grows to at most `max_slots` slots.
    #[must_use]
    pub fn with_max_slots(max_slots: u32) -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            max_slots,
            live: 0,
            _id: PhantomData,
        }
    }

    /// Number of live rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live as usize
    }

    /// Whether there are no live rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// The configured slot limit.
    #[must_use]
    pub fn max_slots(&self) -> u32 {
        self.max_slots
    }
}

impl<Id: SlabId, T> Slab<Id, T> {
    fn stale(id: Id) -> SlabError {
        SlabError::Stale {
            index: id.index(),
            generation: id.generation(),
        }
    }

    /// Stores `value` and returns its id.
    ///
    /// Reuses the most recently freed slot if there is one, otherwise appends a slot.
    pub fn insert(&mut self, value: T) -> Result<Id, SlabError> {
        let index = if let Some(&index) = self.free.last() {
            index
        } else {
            let full = SlabError::Full {
                max_slots: self.max_slots,
            };
            let next = u32::try_from(self.slots.len()).map_err(|_| full)?;
            if next >= self.max_slots {
                return Err(full);
            }
            self.slots.push(Slot::Vacant {
                generation: FIRST_GENERATION,
            });
            self.free.push(next);
            next
        };
        let Some(slot) = self.slots.get_mut(index as usize) else {
            return Err(SlabError::Corrupt);
        };
        let Slot::Vacant { generation } = *slot else {
            return Err(SlabError::Corrupt);
        };
        *slot = Slot::Occupied { generation, value };
        self.free.pop();
        // `live` never exceeds the slot count, which fits in a u32.
        self.live = self.live.saturating_add(1);
        Ok(Id::from_parts(index, generation))
    }

    /// The row `id` names.
    pub fn get(&self, id: Id) -> Result<&T, SlabError> {
        match self.slots.get(id.index() as usize) {
            Some(Slot::Occupied { generation, value }) if *generation == id.generation() => {
                Ok(value)
            }
            _ => Err(Self::stale(id)),
        }
    }

    /// The row `id` names, mutably.
    pub fn get_mut(&mut self, id: Id) -> Result<&mut T, SlabError> {
        match self.slots.get_mut(id.index() as usize) {
            Some(Slot::Occupied { generation, value }) if *generation == id.generation() => {
                Ok(value)
            }
            _ => Err(Self::stale(id)),
        }
    }

    /// The row named by `id`, an id the kernel stored itself (see the module docs). A stale one
    /// is [`StoredIdStale`].
    pub fn stored(&self, id: Id) -> Result<&T, StoredIdStale> {
        self.get(id).map_err(|_| StoredIdStale)
    }

    /// The row named by `id`, an id the kernel stored itself, mutably.
    pub fn stored_mut(&mut self, id: Id) -> Result<&mut T, StoredIdStale> {
        self.get_mut(id).map_err(|_| StoredIdStale)
    }

    /// Whether `id` names a live row.
    #[must_use]
    pub fn contains(&self, id: Id) -> bool {
        self.get(id).is_ok()
    }

    /// Frees the row `id` names and returns it. Every id minted for that row is stale afterwards.
    pub fn remove(&mut self, id: Id) -> Result<T, SlabError> {
        let Some(slot) = self.slots.get_mut(id.index() as usize) else {
            return Err(Self::stale(id));
        };
        match core::mem::replace(slot, Slot::Retired) {
            Slot::Occupied { generation, value } if generation == id.generation() => {
                match generation.checked_add(1) {
                    Some(next) => {
                        *slot = Slot::Vacant { generation: next };
                        self.free.push(id.index());
                    }
                    None => *slot = Slot::Retired,
                }
                self.live = self.live.saturating_sub(1);
                Ok(value)
            }
            other => {
                *slot = other;
                Err(Self::stale(id))
            }
        }
    }

    /// Live rows in slot order (ascending index), with their ids.
    pub fn iter(&self) -> impl Iterator<Item = (Id, &T)> {
        self.slots
            .iter()
            .zip(0_u32..)
            .filter_map(|(slot, index)| match slot {
                Slot::Occupied { generation, value } => {
                    Some((Id::from_parts(index, *generation), value))
                }
                Slot::Vacant { .. } | Slot::Retired => None,
            })
    }
}

#[cfg(test)]
mod tests;
