//! A generational slot table: the recording backend's store for each kind of resource.
//!
//! An id is a slot index and the generation the slot had when the id was handed out
//! (docs/specs/render-backend.md, "Handles, lifetimes and fences"). Freeing a slot moves its
//! generation on, so an id kept past its resource's destruction no longer matches and looks up
//! nothing. Freed slots are reused last-freed first, which keeps ids deterministic for a given
//! call order. A slot whose generation has reached `u32::MAX` is retired when freed instead of
//! reused, so a generation never wraps and an old id can never match a new resource.

use crate::error::RenderError;

/// One slot: the generation of its current (or last) occupant, and the occupant.
#[derive(Debug)]
struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// Values of one kind, addressed by `(index, generation)`.
#[derive(Debug)]
pub(super) struct Slots<T> {
    slots: Vec<Slot<T>>,
    /// Free slot indices, reused from the end.
    free: Vec<u32>,
}

impl<T> Default for Slots<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }
}

impl<T> Slots<T> {
    /// Stores `value` and returns its `(index, generation)`.
    pub(super) fn insert(&mut self, value: T) -> Result<(u32, u32), RenderError> {
        if let Some(index) = self.free.pop()
            && let Some(slot) = usize::try_from(index)
                .ok()
                .and_then(|i| self.slots.get_mut(i))
        {
            slot.value = Some(value);
            return Ok((index, slot.generation));
        }
        let index = u32::try_from(self.slots.len()).map_err(|_| RenderError::Overflow {
            what: "resource slot count",
        })?;
        self.slots.push(Slot {
            generation: 0,
            value: Some(value),
        });
        Ok((index, 0))
    }

    fn slot(&self, index: u32, generation: u32) -> Option<&Slot<T>> {
        let slot = self.slots.get(usize::try_from(index).ok()?)?;
        (slot.generation == generation && slot.value.is_some()).then_some(slot)
    }

    /// The live value at `(index, generation)`.
    pub(super) fn get(&self, index: u32, generation: u32) -> Option<&T> {
        self.slot(index, generation)?.value.as_ref()
    }

    /// The live value at `(index, generation)`, to change.
    pub(super) fn get_mut(&mut self, index: u32, generation: u32) -> Option<&mut T> {
        let slot = self.slots.get_mut(usize::try_from(index).ok()?)?;
        if slot.generation == generation {
            slot.value.as_mut()
        } else {
            None
        }
    }

    /// Takes the live value at `(index, generation)` out and frees its slot.
    pub(super) fn remove(&mut self, index: u32, generation: u32) -> Option<T> {
        let slot = self.slots.get_mut(usize::try_from(index).ok()?)?;
        if slot.generation != generation {
            return None;
        }
        let value = slot.value.take()?;
        if let Some(next) = slot.generation.checked_add(1) {
            slot.generation = next;
            self.free.push(index);
        }
        Some(value)
    }

    /// Live values with their `(index, generation)`, in index order.
    pub(super) fn iter(&self) -> impl Iterator<Item = (u32, u32, &T)> {
        self.slots.iter().zip(0_u32..).filter_map(|(slot, index)| {
            slot.value
                .as_ref()
                .map(|value| (index, slot.generation, value))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Slot, Slots};

    #[test]
    fn freed_slots_are_reused_with_the_next_generation() {
        let mut s = Slots::default();
        assert_eq!(s.insert('a'), Ok((0, 0)));
        assert_eq!(s.insert('b'), Ok((1, 0)));
        assert_eq!(s.remove(0, 0), Some('a'));
        // The stale id looks up nothing, before and after the slot is reused.
        assert_eq!(s.get(0, 0), None);
        assert_eq!(s.insert('c'), Ok((0, 1)));
        assert_eq!(s.get(0, 0), None);
        assert_eq!(s.get(0, 1), Some(&'c'));
        assert_eq!(s.get_mut(0, 0), None);
        assert_eq!(s.remove(0, 0), None, "a stale id frees nothing");
        assert_eq!(s.get(1, 0), Some(&'b'), "the neighbour is unchanged");
        assert_eq!(
            s.iter().collect::<Vec<_>>(),
            vec![(0, 1, &'c'), (1, 0, &'b')]
        );
    }

    #[test]
    fn last_freed_is_reused_first() {
        let mut s = Slots::default();
        for v in 0..3 {
            s.insert(v).expect("room");
        }
        s.remove(0, 0);
        s.remove(2, 0);
        assert_eq!(s.insert(7), Ok((2, 1)));
        assert_eq!(s.insert(8), Ok((0, 1)));
        assert_eq!(s.insert(9), Ok((3, 0)));
    }

    #[test]
    fn unknown_index_and_double_remove() {
        let mut s = Slots::default();
        assert_eq!(s.get(0, 0), None::<&u8>);
        assert_eq!(s.remove(5, 0), None);
        s.insert(1_u8).expect("room");
        assert_eq!(s.remove(0, 0), Some(1));
        assert_eq!(s.remove(0, 0), None);
        assert_eq!(s.remove(0, 1), None, "the freed slot holds nothing");
    }

    #[test]
    fn a_slot_at_the_last_generation_is_retired() {
        let mut s = Slots {
            slots: vec![Slot {
                generation: u32::MAX,
                value: Some('x'),
            }],
            free: Vec::new(),
        };
        assert_eq!(s.remove(0, u32::MAX), Some('x'));
        assert_eq!(s.get(0, u32::MAX), None);
        // The slot is not reused: the next value gets a new slot.
        assert_eq!(s.insert('y'), Ok((1, 0)));

        // One generation earlier it is still reused.
        let mut s = Slots {
            slots: vec![Slot {
                generation: u32::MAX - 1,
                value: Some('x'),
            }],
            free: Vec::new(),
        };
        s.remove(0, u32::MAX - 1);
        assert_eq!(s.insert('y'), Ok((0, u32::MAX)));
    }
}
