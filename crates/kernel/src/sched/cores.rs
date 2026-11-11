//! Cores as the scheduler sees them: core ids, affinity masks, and the trait through which the
//! scheduler asks the outside world to interrupt or wake a core.
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_affinity_mask.h` @ cf654052 (GPL-3.0-or-later):
//! an affinity mask is a `u64` with one bit per physical core. Eden asserts that a mask has no
//! bit above `NUM_CPU_CORES`; here the core count is the scheduler's configuration (kernel spec,
//! rule 5) and an out-of-range mask is an error the scheduler returns.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The most cores a scheduler can have: one bit of a `u64` affinity mask each. Eden stores
/// affinity as a `u64` (`k_affinity_mask.h`, `m_mask`) and asserts every core id is below
/// `Common::BitSize<u64>()` (`k_thread.cpp`, `KThread::Initialize`).
pub const MAX_CORES: u8 = 64;

/// The number of application cores on a Switch 2: Eden PR #4588's
/// `eden-pr4588/src/core/hardware_properties.h` (`NUM_CPU_CORES = 6`). The default scheduler
/// has this many cores.
pub const APPLICATION_CORE_COUNT: u8 = 6;

/// A core index that is not below [`MAX_CORES`]. Also what deserialising a [`CoreId`] from such a
/// byte reports, so a saved state cannot hold a core whose [`CoreId::bit`] would overflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("core index {0} is not below the maximum of {MAX_CORES} cores")]
pub struct CoreIdOutOfRange(pub u8);

/// A physical core index, below [`MAX_CORES`]. Whether a scheduler has that core is checked by
/// the scheduler. Serialised as the bare index (one byte); deserialising goes through
/// [`CoreId::new`]'s bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u8")]
pub struct CoreId(u8);

impl TryFrom<u8> for CoreId {
    type Error = CoreIdOutOfRange;

    fn try_from(index: u8) -> Result<Self, Self::Error> {
        Self::new(index).ok_or(CoreIdOutOfRange(index))
    }
}

impl CoreId {
    /// The core with this index, or `None` if it is not below [`MAX_CORES`].
    #[must_use]
    pub const fn new(index: u8) -> Option<Self> {
        if index < MAX_CORES {
            Some(Self(index))
        } else {
            None
        }
    }

    /// The core's index.
    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// The core's bit in an affinity mask.
    #[must_use]
    pub const fn bit(self) -> u64 {
        1_u64 << self.0
    }
}

/// A set of cores, as Eden's `KAffinityMask`: bit `n` set means core `n` is in the set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AffinityMask(u64);

impl AffinityMask {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// A mask from its raw bits.
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// The set holding only `core`.
    #[must_use]
    pub const fn single(core: CoreId) -> Self {
        Self(core.bit())
    }

    /// The raw bits.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Whether `core` is in the set (`KAffinityMask::GetAffinity`).
    #[must_use]
    pub const fn contains(self, core: CoreId) -> bool {
        self.0 & core.bit() != 0
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The set without `core`.
    #[must_use]
    pub const fn without(self, core: CoreId) -> Self {
        Self(self.0 & !core.bit())
    }

    /// The highest core in the set, or `None` if it is empty. Eden computes this as
    /// `BitSize<u64>() - 1 - countl_zero(mask)` (`k_thread.cpp`, `KThread::SetCoreMask`).
    #[must_use]
    pub fn highest(self) -> Option<CoreId> {
        if self.0 == 0 {
            return None;
        }
        // A non-zero mask has at most 63 leading zeros, so this is 0..=63, below MAX_CORES.
        let index = (u64::BITS - 1).checked_sub(self.0.leading_zeros())?;
        CoreId::new(u8::try_from(index).ok()?)
    }

    /// The cores in the set, lowest first: Eden's `GetNextCore` loop (`countr_zero`, then clear
    /// the bit) in `k_priority_queue.h`.
    pub fn iter(self) -> impl Iterator<Item = CoreId> {
        let mut rest = self.0;
        core::iter::from_fn(move || {
            if rest == 0 {
                return None;
            }
            // trailing_zeros of a non-zero u64 is 0..=63, so the conversion and CoreId::new
            // always succeed.
            let core = CoreId::new(u8::try_from(rest.trailing_zeros()).ok()?)?;
            rest &= !core.bit();
            Some(core)
        })
    }
}

/// How the scheduler reaches the cores (kernel spec, "Scheduler": `interrupt(core)`,
/// `wake(core)`). The core loop implements it; tests use a fake that records the calls.
///
/// Eden sends every core the same inter-processor interrupt (`KScheduler::RescheduleCores`,
/// `PhysicalCore::Interrupt`). Starquake's cores are host threads that either run guest code or
/// park (execution-model spec, "Core threads"), so the scheduler picks the signal that reaches
/// each: [`CoreSignals::interrupt`] for a core that has a current guest thread,
/// [`CoreSignals::wake`] for one that has none. That split is starquake's own.
pub trait CoreSignals {
    /// Make a core that is running guest code exit to the core loop and reschedule.
    fn interrupt(&mut self, core: CoreId);

    /// Make a parked core (one with no current thread) reschedule.
    fn wake(&mut self, core: CoreId);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_id_is_bounded_by_max_cores() {
        assert_eq!(
            CoreId::new(MAX_CORES - 1).map(CoreId::index),
            Some(MAX_CORES - 1)
        );
        assert_eq!(CoreId::new(MAX_CORES), None);
    }

    #[test]
    fn mask_iterates_lowest_first_and_finds_highest() {
        let top = CoreId::new(MAX_CORES - 1).unwrap();
        let mask = AffinityMask::from_bits(0b1010_0110 | top.bit());
        let cores: Vec<u8> = mask.iter().map(CoreId::index).collect();
        assert_eq!(cores, vec![1, 2, 5, 7, MAX_CORES - 1]);
        assert_eq!(mask.highest(), Some(top));
        assert_eq!(AffinityMask::EMPTY.highest(), None);
        assert_eq!(AffinityMask::EMPTY.iter().count(), 0);
    }

    #[test]
    fn mask_set_operations() {
        let two = CoreId::new(2).unwrap();
        let mask = AffinityMask::single(two);
        assert!(mask.contains(two));
        assert!(!mask.contains(CoreId::new(3).unwrap()));
        assert!(mask.without(two).is_empty());
        assert_eq!(mask.bits(), two.bit());
    }

    #[test]
    fn core_id_serialises_as_its_index() {
        // Golden bytes: postcard writes a u8 as itself.
        let five = CoreId::new(5).unwrap();
        assert_eq!(postcard::to_allocvec(&five).unwrap(), vec![five.index()]);
        let top = CoreId::new(MAX_CORES - 1).unwrap();
        assert_eq!(postcard::to_allocvec(&top).unwrap(), vec![MAX_CORES - 1]);
        assert_eq!(postcard::from_bytes::<CoreId>(&[five.index()]), Ok(five));
    }

    #[test]
    fn core_id_deserialising_enforces_the_bound() {
        // Every possible byte: the in-range ones decode, the rest are errors, never a CoreId
        // whose `bit()` would overflow its shift.
        for byte in 0..=u8::MAX {
            let decoded = postcard::from_bytes::<CoreId>(&[byte]);
            if byte < MAX_CORES {
                assert_eq!(decoded.map(CoreId::index), Ok(byte));
            } else {
                assert_eq!(decoded, Err(postcard::Error::SerdeDeCustom), "byte {byte}");
            }
        }
    }

    #[test]
    fn core_id_out_of_range_error_names_the_index() {
        assert_eq!(
            CoreId::try_from(MAX_CORES),
            Err(CoreIdOutOfRange(MAX_CORES))
        );
        assert_eq!(
            CoreIdOutOfRange(MAX_CORES).to_string(),
            format!("core index {MAX_CORES} is not below the maximum of {MAX_CORES} cores")
        );
    }

    #[test]
    fn affinity_mask_serialises_as_a_varint_u64() {
        // Golden bytes. Cores 1 and 8: 0b1_0000_0010 = 0x102, LEB128 low group first with the
        // continuation bit set: 0x82, 0x02.
        let low =
            AffinityMask::single(CoreId::new(1).unwrap()).bits() | CoreId::new(8).unwrap().bit();
        let low = AffinityMask::from_bits(low);
        assert_eq!(postcard::to_allocvec(&low).unwrap(), vec![0x82, 0x02]);
        assert_eq!(postcard::from_bytes::<AffinityMask>(&[0x82, 0x02]), Ok(low));
        // Cores 1 and 63: the low group 0x02 with the continuation bit, eight empty groups
        // (0x80), and bit 63 alone in the tenth.
        let high = AffinityMask::from_bits(
            CoreId::new(1).unwrap().bit() | CoreId::new(MAX_CORES - 1).unwrap().bit(),
        );
        let mut want = vec![0x82];
        want.extend([0x80; 8]);
        want.push(0x01);
        assert_eq!(postcard::to_allocvec(&high).unwrap(), want);
        assert_eq!(postcard::from_bytes::<AffinityMask>(&want), Ok(high));
    }
}
