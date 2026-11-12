//! Address-space ids: the tag the host's TLB keeps each process's translations under.
//!
//! The ids are a host resource, not a Horizon one: Horizon's own allocator
//! (`KPageTable`'s ASID manager) is hardware-facing and not ported. What the pool keeps is the
//! two rules the hypervisor backend states (`crates/cpu/src/hvf.rs`, "Ids are as wide as the
//! host makes them"):
//!
//! - **Ids fit the width the backend reports** (`HvfVcpu::asid_bits`: 8 on the Mac this was
//!   measured on, `tools/hvprobe/README.md`). The pool is built with that width and never
//!   hands out a wider id.
//! - **An id is not reused before its flush has run.** Translations cached under an id outlive a
//!   change of tables, so an id given back ([`AsidPool::release`]) waits until the caller says
//!   the TLB flush for it has run ([`AsidPool::flushed`]) before it can be handed out again.
//!
//! Id 0 is never handed out, so a vCPU that was never given an address space shares no
//! translations with a process. That is starquake's own choice (a guess in the sense of
//! `CLAUDE.md`: nothing guest-visible depends on it); the first id is 1, the id the walking
//! skeleton gave its one process.

use std::collections::BTreeSet;

use thiserror::Error;

/// The widest id the pool knows: the 16 bits `TCR_EL1.AS` allows (Arm DDI 0487, `ASIDBits`).
pub const MAX_ASID_BITS: u8 = 16;

/// The first id the pool hands out (see the module docs).
pub const FIRST_ASID: u16 = 1;

/// Why the pool refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AsidError {
    /// A width of zero or more than [`MAX_ASID_BITS`].
    #[error("an address-space id is 1 to {MAX_ASID_BITS} bits wide, not {bits}")]
    Width {
        /// The width asked for.
        bits: u8,
    },
    /// Every id is in use or waiting for its flush.
    #[error("every address-space id of {bits} bits is in use or waiting for a flush")]
    Exhausted {
        /// The pool's width.
        bits: u8,
    },
    /// An id given back that the pool did not hand out, or one already given back.
    #[error("address-space id {asid} is not in use")]
    NotInUse {
        /// The id.
        asid: u16,
    },
    /// A flush reported for an id that was not waiting for one.
    #[error("address-space id {asid} is not waiting for a flush")]
    NotPending {
        /// The id.
        asid: u16,
    },
}

/// The address-space ids of one VM. See the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsidPool {
    bits: u8,
    in_use: BTreeSet<u16>,
    awaiting_flush: BTreeSet<u16>,
}

impl AsidPool {
    /// A pool of the ids `bits` wide, as the backend reports them.
    pub fn new(bits: u8) -> Result<Self, AsidError> {
        if bits == 0 || bits > MAX_ASID_BITS {
            return Err(AsidError::Width { bits });
        }
        Ok(Self {
            bits,
            in_use: BTreeSet::new(),
            awaiting_flush: BTreeSet::new(),
        })
    }

    /// The width of the pool's ids.
    #[must_use]
    pub const fn bits(&self) -> u8 {
        self.bits
    }

    /// The largest id the pool may hand out.
    #[must_use]
    pub fn max(&self) -> u16 {
        // `bits` is 1..=16, so the shift fits a u32 and the result a u16.
        u16::try_from((1_u32 << self.bits) - 1).unwrap_or(u16::MAX)
    }

    /// Hands out the lowest id that is neither in use nor waiting for its flush.
    pub fn allocate(&mut self) -> Result<u16, AsidError> {
        let asid = (FIRST_ASID..=self.max())
            .find(|asid| !self.in_use.contains(asid) && !self.awaiting_flush.contains(asid))
            .ok_or(AsidError::Exhausted { bits: self.bits })?;
        self.in_use.insert(asid);
        Ok(asid)
    }

    /// Gives an id back: its address space is gone. It is not handed out again until
    /// [`AsidPool::flushed`] says its flush has run.
    pub fn release(&mut self, asid: u16) -> Result<(), AsidError> {
        if !self.in_use.remove(&asid) {
            return Err(AsidError::NotInUse { asid });
        }
        self.awaiting_flush.insert(asid);
        Ok(())
    }

    /// The ids given back whose flush has not run yet, lowest first.
    pub fn awaiting_flush(&self) -> impl Iterator<Item = u16> + '_ {
        self.awaiting_flush.iter().copied()
    }

    /// The TLB flush for `asid` has run: the id may be handed out again.
    pub fn flushed(&mut self, asid: u16) -> Result<(), AsidError> {
        if !self.awaiting_flush.remove(&asid) {
            return Err(AsidError::NotPending { asid });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The width the hypervisor backend reports on the Mac it was measured on.
    const HOST_BITS: u8 = 8;

    #[test]
    fn the_width_is_one_to_sixteen_bits() {
        assert_eq!(AsidPool::new(0), Err(AsidError::Width { bits: 0 }));
        assert_eq!(
            AsidPool::new(MAX_ASID_BITS + 1),
            Err(AsidError::Width {
                bits: MAX_ASID_BITS + 1
            })
        );
        assert_eq!(AsidPool::new(1).map(|pool| pool.max()), Ok(1));
        assert_eq!(
            AsidPool::new(MAX_ASID_BITS).map(|pool| pool.max()),
            Ok(u16::MAX)
        );
    }

    #[test]
    fn ids_stay_within_the_width_and_start_at_one() {
        let mut pool = AsidPool::new(HOST_BITS).unwrap();
        let max = pool.max();
        assert_eq!(u32::from(max), (1 << HOST_BITS) - 1);
        let ids: Vec<u16> = (FIRST_ASID..=max).map(|_| pool.allocate().unwrap()).collect();
        assert_eq!(ids.first(), Some(&FIRST_ASID));
        assert_eq!(ids.last(), Some(&max));
        assert_eq!(
            pool.allocate(),
            Err(AsidError::Exhausted { bits: HOST_BITS })
        );
    }

    #[test]
    fn a_released_id_is_not_reused_until_its_flush_has_run() {
        let mut pool = AsidPool::new(1).unwrap();
        // One id only: 1.
        assert_eq!(pool.allocate(), Ok(FIRST_ASID));
        pool.release(FIRST_ASID).unwrap();
        assert_eq!(pool.awaiting_flush().collect::<Vec<_>>(), vec![FIRST_ASID]);
        assert_eq!(pool.allocate(), Err(AsidError::Exhausted { bits: 1 }));
        pool.flushed(FIRST_ASID).unwrap();
        assert_eq!(pool.awaiting_flush().count(), 0);
        assert_eq!(pool.allocate(), Ok(FIRST_ASID));
    }

    #[test]
    fn a_wider_pool_skips_an_id_waiting_for_its_flush() {
        let mut pool = AsidPool::new(HOST_BITS).unwrap();
        let first = pool.allocate().unwrap();
        let second = pool.allocate().unwrap();
        pool.release(first).unwrap();
        // The lowest free id is `first`, but its flush has not run.
        let third = pool.allocate().unwrap();
        assert!(third != first && third > second);
        pool.flushed(first).unwrap();
        assert_eq!(pool.allocate(), Ok(first));
    }

    #[test]
    fn giving_back_or_flushing_an_id_out_of_turn_is_refused() {
        let mut pool = AsidPool::new(HOST_BITS).unwrap();
        assert_eq!(pool.release(5), Err(AsidError::NotInUse { asid: 5 }));
        let id = pool.allocate().unwrap();
        assert_eq!(pool.flushed(id), Err(AsidError::NotPending { asid: id }));
        pool.release(id).unwrap();
        assert_eq!(pool.release(id), Err(AsidError::NotInUse { asid: id }));
        pool.flushed(id).unwrap();
        assert_eq!(pool.flushed(id), Err(AsidError::NotPending { asid: id }));
    }

    #[test]
    fn errors_say_what_was_refused() {
        assert_eq!(
            AsidError::Exhausted { bits: 8 }.to_string(),
            "every address-space id of 8 bits is in use or waiting for a flush"
        );
        assert_eq!(
            AsidError::NotInUse { asid: 3 }.to_string(),
            "address-space id 3 is not in use"
        );
    }
}
