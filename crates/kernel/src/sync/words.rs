//! The 32-bit words of user memory that the address arbiter, condition variables and lock
//! arbitration read and write, with the guest's access checked.
//!
//! The operations are facts from Atmosphère @ the commit `docs/references.md` pins (GPL-2.0-only,
//! fact only: nothing here is ported from it), paths relative to the reference folder:
//! - `atmosphere/libraries/libmesosphere/source/kern_k_address_arbiter.cpp`: `ReadFromUser` is a
//!   copy from user memory, which fails for memory the user may not read; `DecrementIfLessThan`
//!   and `UpdateIfEqual` first check `cpu::CanAccessAtomic`, then exchange atomically.
//! - `atmosphere/libraries/libmesosphere/include/mesosphere/arch/arm64/kern_cpu.hpp`:
//!   `CanAccessAtomic` translates the address for an EL0 write (`at s1e0w`).
//! - `atmosphere/libraries/libmesosphere/source/kern_k_condition_variable.cpp`: `WriteToUser`
//!   is a copy to user memory; `SignalImpl` checks `CanAccessAtomic` before `UpdateLockAtomic`.
//!
//! Eden (`eden/src/core/hle/kernel/k_address_arbiter.cpp` and `k_condition_variable.cpp` @
//! cf654052) checks nothing: its `ReadFromUser` and `WriteToUser` always succeed and its atomic
//! operations carry `TODO(bunnei): We should call CanAccessAtomic(..) here`. Starquake follows
//! Atmosphère, so a word the guest may not access is
//! [`KernelError::InvalidCurrentMemory`](crate::result::KernelError::InvalidCurrentMemory)
//! where each call says, not a silent zero. That divergence from Eden is pinned by the tests the
//! module docs of [`crate::sync`] name.
//!
//! # What is a guess
//!
//! - **Atomic access is checked as "the guest may read and write all four bytes"**
//!   ([`GuestWords`]). `CanAccessAtomic` asks the MMU whether an EL0 write would translate, and
//!   Horizon has no user permission that writes without reading. A hardware test of
//!   `SignalToAddress(SignalAndIncrementIfEqual)` on a read-only page would confirm it.
//! - **The read-modify-write operations are not atomic against guest code running on another
//!   core.** The kernel lock orders them against every other system call, but
//!   [`GuestMemory`] has only plain reads and writes, so a guest's own exclusive store on another
//!   core can interleave. Atomic guest-memory operations belong to the memory crate; this module
//!   is the one place that would call them.

use starquake_memory::GuestMemory;

use crate::memory::AddressSpace;
use crate::svc::sync::read_user_memory;

/// The size of one word: the arbiter's values are `s32` and the lock tags `u32`.
pub const WORD_SIZE: u64 = size_of::<u32>() as u64;

/// User memory as 32-bit little-endian words, with the guest's access checked.
pub trait UserWords {
    /// The word at `address`, or `None` if the guest may not read it (`ReadFromUser`).
    fn read(&self, address: u64) -> Option<u32>;
    /// Writes `value` at `address` and says whether it could: false if the guest may not write
    /// it (`WriteToUser`), and then nothing was written.
    fn write(&self, address: u64, value: u32) -> bool;
    /// Whether an atomic operation may touch the word at `address` (`CanAccessAtomic`).
    fn can_access_atomic(&self, address: u64) -> bool;
}

/// A process's user memory: its address space says what the guest may access, and
/// [`GuestMemory`] holds the bytes.
#[derive(Clone, Copy)]
pub struct GuestWords<'a> {
    /// The process's address space.
    pub space: &'a AddressSpace,
    /// Its memory.
    pub memory: &'a dyn GuestMemory,
}

impl std::fmt::Debug for GuestWords<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestWords").finish_non_exhaustive()
    }
}

impl UserWords for GuestWords<'_> {
    fn read(&self, address: u64) -> Option<u32> {
        let mut bytes = [0; size_of::<u32>()];
        read_user_memory(self.space, self.memory, address, &mut bytes)
            .then(|| u32::from_le_bytes(bytes))
    }

    fn write(&self, address: u64, value: u32) -> bool {
        self.space.user_can_write(address, WORD_SIZE)
            && self.memory.write(address, &value.to_le_bytes()).is_ok()
    }

    fn can_access_atomic(&self, address: u64) -> bool {
        self.space.user_can_read(address, WORD_SIZE)
            && self.space.user_can_write(address, WORD_SIZE)
    }
}

/// `DecrementIfLessThan`: the word at `address` as an `s32`, decremented in place (wrapping, as
/// the reference's `current_value - 1` on the two's-complement register does) when it is below
/// `value`. `None` when the word cannot be accessed atomically.
pub fn decrement_if_less_than(words: &dyn UserWords, address: u64, value: i32) -> Option<i32> {
    if !words.can_access_atomic(address) {
        return None;
    }
    let current = words.read(address)?.cast_signed();
    if current < value && !words.write(address, current.wrapping_sub(1).cast_unsigned()) {
        return None;
    }
    Some(current)
}

/// `UpdateIfEqual`: the word at `address` as an `s32`, replaced by `new_value` when it equals
/// `value`. `None` when the word cannot be accessed atomically.
pub fn update_if_equal(
    words: &dyn UserWords,
    address: u64,
    value: i32,
    new_value: i32,
) -> Option<i32> {
    if !words.can_access_atomic(address) {
        return None;
    }
    let current = words.read(address)?.cast_signed();
    if current == value && !words.write(address, new_value.cast_unsigned()) {
        return None;
    }
    Some(current)
}

/// `UpdateLockAtomic`: the word at `address`, replaced by `if_zero` when it is zero and by
/// itself with `or_mask` set otherwise. `None` when the word cannot be accessed atomically.
pub fn update_lock(words: &dyn UserWords, address: u64, if_zero: u32, or_mask: u32) -> Option<u32> {
    if !words.can_access_atomic(address) {
        return None;
    }
    let current = words.read(address)?;
    let value = if current == 0 {
        if_zero
    } else {
        current | or_mask
    };
    words.write(address, value).then_some(current)
}

#[cfg(test)]
pub(crate) mod testing {
    //! Words in a map, with addresses the guest may only read and addresses it may not touch.

    use std::cell::RefCell;
    use std::collections::{BTreeMap, BTreeSet};

    use super::UserWords;

    /// A test's user memory: every word reads as what was last written (zero to start), except
    /// the `read_only` ones, which can't be written or used atomically, and the `inaccessible`
    /// ones, which can't be touched at all.
    #[derive(Debug, Default)]
    pub(crate) struct MapWords {
        pub(crate) words: RefCell<BTreeMap<u64, u32>>,
        pub(crate) read_only: BTreeSet<u64>,
        pub(crate) inaccessible: BTreeSet<u64>,
    }

    impl MapWords {
        pub(crate) fn get(&self, address: u64) -> u32 {
            self.words.borrow().get(&address).copied().unwrap_or(0)
        }

        pub(crate) fn set(&self, address: u64, value: u32) {
            self.words.borrow_mut().insert(address, value);
        }
    }

    impl UserWords for MapWords {
        fn read(&self, address: u64) -> Option<u32> {
            (!self.inaccessible.contains(&address)).then(|| self.get(address))
        }

        fn write(&self, address: u64, value: u32) -> bool {
            let allowed =
                !self.inaccessible.contains(&address) && !self.read_only.contains(&address);
            if allowed {
                self.set(address, value);
            }
            allowed
        }

        fn can_access_atomic(&self, address: u64) -> bool {
            !self.inaccessible.contains(&address) && !self.read_only.contains(&address)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::MapWords;
    use super::*;

    const AT: u64 = 0x1000;

    #[test]
    fn decrement_if_less_than_decrements_only_below_the_value() {
        let words = MapWords::default();
        words.set(AT, 5);
        assert_eq!(decrement_if_less_than(&words, AT, 5), Some(5));
        assert_eq!(words.get(AT), 5, "equal is not less: unchanged");
        assert_eq!(decrement_if_less_than(&words, AT, 6), Some(5));
        assert_eq!(words.get(AT), 4);
        words.set(AT, i32::MIN.cast_unsigned());
        assert_eq!(decrement_if_less_than(&words, AT, 0), Some(i32::MIN));
        assert_eq!(words.get(AT), i32::MAX.cast_unsigned(), "wraps");
    }

    #[test]
    fn update_if_equal_writes_only_on_a_match() {
        let words = MapWords::default();
        words.set(AT, 7);
        assert_eq!(update_if_equal(&words, AT, 6, 9), Some(7));
        assert_eq!(words.get(AT), 7);
        assert_eq!(update_if_equal(&words, AT, 7, 9), Some(7));
        assert_eq!(words.get(AT), 9);
    }

    #[test]
    fn update_lock_takes_a_free_lock_and_marks_a_held_one() {
        let words = MapWords::default();
        assert_eq!(update_lock(&words, AT, 0x22, 0x4000_0000), Some(0));
        assert_eq!(words.get(AT), 0x22);
        assert_eq!(update_lock(&words, AT, 0x33, 0x4000_0000), Some(0x22));
        assert_eq!(words.get(AT), 0x4000_0022);
    }

    #[test]
    fn atomic_operations_need_atomic_access_and_leave_the_word_alone_without_it() {
        let mut words = MapWords::default();
        words.set(AT, 1);
        words.read_only.insert(AT);
        assert_eq!(decrement_if_less_than(&words, AT, 9), None);
        assert_eq!(update_if_equal(&words, AT, 1, 2), None);
        assert_eq!(update_lock(&words, AT, 3, 4), None);
        assert_eq!(words.get(AT), 1);
        assert_eq!(words.read(AT), Some(1), "a read-only word still reads");
        words.inaccessible.insert(AT);
        assert_eq!(words.read(AT), None);
    }
}
