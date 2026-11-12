//! How much memory a process may use: the console's memory arrangement, the size of the pool an
//! application draws from, and the count of what a process has reserved from it.
//!
//! Ported from Eden (GPL-3.0-or-later), paths relative to the reference folder:
//!
//! - `eden/src/core/hle/kernel/board/nintendo/nx/k_system_control.cpp` @ cf654052,
//!   `KSystemControl::Init::GetIntendedMemorySize` and `GetApplicationPoolSize`: the two
//!   arrangements starquake runs with, 4 GiB (a Switch 1) and 12 GiB (a Switch 2; kernel spec,
//!   "Address space": "Pool accounting ... for the 12 GB layout").
//! - `eden/src/core/hle/kernel/k_resource_limit.cpp` @ cf654052, `KResourceLimit::Reserve`,
//!   `Release` and `GetFreeValue`, for `LimitableResource::PhysicalMemoryMax`, and the rule of
//!   `KScopedResourceReservation` (`k_scoped_resource_reservation.h`) that a reservation of
//!   nothing succeeds without asking. `KProcess::LoadFromMetadata`
//!   (`eden/src/core/hle/kernel/k_process.cpp`) gives a process a limit of its pool's size.
//!
//! Horizon's page heaps are not ported (kernel spec, "Not planned"): a pool here is a size and
//! a count, and the count is what `GetInfo` reports. Only the application pool is modelled; the
//! applet and the two system pools arrive with the first process that draws from them.
//!
//! Divergences from Eden:
//!
//! - The 12 GiB arrangement's memory size is Eden's and PR #4588's 12 GiB. Eden-NX2 adds 8 GiB
//!   for its graphics server (`KSystemControl::Nx2GraphicsServerMemorySize` in
//!   `eden-nx2/.../k_system_control.h` @ 50fa86bb); how the graphics server's memory is
//!   accounted is the graphics server's slice to settle (scoping doc, rows 75-80).
//! - `KResourceLimit::Reserve` can wait up to ten seconds for another thread to give memory
//!   back, when releases are pending (`m_current_hints`). Nothing here has a pending release,
//!   so a reservation that does not fit fails at once, which is what Eden does in that state.

use crate::result::{KernelError, SvcResult};

/// One mebibyte.
const MIB: u64 = 1 << 20;
/// One gibibyte.
const GIB: u64 = 1 << 30;

/// How the console's memory is divided (`Smc::MemoryArrangement`): the two arrangements
/// starquake runs with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryArrangement {
    /// 4 GiB: a Switch 1.
    FourGb,
    /// 12 GiB: a Switch 2.
    TwelveGb,
}

impl MemoryArrangement {
    /// The console's memory in bytes (`GetIntendedMemorySize`).
    #[must_use]
    pub const fn physical_memory_size(self) -> u64 {
        match self {
            Self::FourGb => 4 * GIB,
            Self::TwelveGb => 12 * GIB,
        }
    }

    /// The size of the application pool in bytes (`GetApplicationPoolSize`).
    #[must_use]
    pub const fn application_pool_size(self) -> u64 {
        match self {
            Self::FourGb => 3285 * MIB,
            Self::TwelveGb => 9809 * MIB,
        }
    }
}

/// How much physical memory a process may reserve, and how much it has
/// (`KResourceLimit`, `LimitableResource::PhysicalMemoryMax`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryLimit {
    limit: u64,
    reserved: u64,
}

impl MemoryLimit {
    /// A limit of `limit` bytes with nothing reserved.
    #[must_use]
    pub const fn new(limit: u64) -> Self {
        Self { limit, reserved: 0 }
    }

    /// The most that can be reserved.
    #[must_use]
    pub const fn limit(&self) -> u64 {
        self.limit
    }

    /// What is reserved now (`GetCurrentValue`).
    #[must_use]
    pub const fn reserved(&self) -> u64 {
        self.reserved
    }

    /// What can still be reserved (`GetFreeValue`).
    #[must_use]
    pub const fn free(&self) -> u64 {
        self.limit - self.reserved
    }

    /// Reserves `bytes` (`KScopedResourceReservation`, committed). Reserving nothing succeeds.
    /// Anything else is [`KernelError::LimitReached`] when the limit is already reached, when
    /// the total would overflow, or when it would exceed the limit; nothing is then reserved.
    pub fn reserve(&mut self, bytes: u64) -> SvcResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        if self.reserved >= self.limit {
            return Err(KernelError::LimitReached);
        }
        match self.reserved.checked_add(bytes) {
            Some(total) if total <= self.limit => {
                self.reserved = total;
                Ok(())
            }
            _ => Err(KernelError::LimitReached),
        }
    }

    /// Gives `bytes` back (`Release`). `None`, with nothing changed, if that is more than is
    /// reserved: Eden asserts it is not.
    #[must_use]
    pub fn release(&mut self, bytes: u64) -> Option<()> {
        self.reserved = self.reserved.checked_sub(bytes)?;
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_arrangements_have_the_references_sizes() {
        // `GetIntendedMemorySize` and `GetApplicationPoolSize`, written out: 4 GiB with a
        // 3285 MiB application pool, 12 GiB with a 9809 MiB one. The first is what Eden-NX2
        // reported to the homebrew fixture as its total memory (`GetInfo` 6: 0xcd500000).
        let four = MemoryArrangement::FourGb;
        assert_eq!(four.physical_memory_size(), 0x1_0000_0000);
        assert_eq!(four.application_pool_size(), 0xcd50_0000);
        let twelve = MemoryArrangement::TwelveGb;
        assert_eq!(twelve.physical_memory_size(), 0x3_0000_0000);
        assert_eq!(twelve.application_pool_size(), 0x2_6510_0000);
    }

    #[test]
    fn a_reservation_fits_exactly_up_to_the_limit() {
        let mut limit = MemoryLimit::new(0x3000);
        assert_eq!(
            (limit.limit(), limit.reserved(), limit.free()),
            (0x3000, 0, 0x3000)
        );
        assert_eq!(limit.reserve(0x1000), Ok(()));
        // One byte more than what is left does not fit, and reserves nothing.
        assert_eq!(limit.reserve(0x2001), Err(KernelError::LimitReached));
        assert_eq!((limit.reserved(), limit.free()), (0x1000, 0x2000));
        // Exactly what is left does.
        assert_eq!(limit.reserve(0x2000), Ok(()));
        assert_eq!((limit.reserved(), limit.free()), (0x3000, 0));
        // At the limit nothing more fits, but nothing can still be reserved.
        assert_eq!(limit.reserve(1), Err(KernelError::LimitReached));
        assert_eq!(limit.reserve(0), Ok(()));
    }

    #[test]
    fn a_reservation_whose_total_overflows_is_refused() {
        let mut limit = MemoryLimit::new(u64::MAX);
        assert_eq!(limit.reserve(u64::MAX - 1), Ok(()));
        assert_eq!(limit.reserve(2), Err(KernelError::LimitReached));
        assert_eq!(limit.reserved(), u64::MAX - 1);
        assert_eq!(limit.reserve(1), Ok(()));
        assert_eq!(limit.free(), 0);
    }

    #[test]
    fn a_release_gives_back_no_more_than_was_reserved() {
        let mut limit = MemoryLimit::new(0x3000);
        limit.reserve(0x2000).unwrap();
        assert_eq!(limit.release(0x2001), None);
        assert_eq!(limit.reserved(), 0x2000);
        assert_eq!(limit.release(0x2000), Some(()));
        assert_eq!((limit.reserved(), limit.free()), (0, 0x3000));
        assert_eq!(limit.release(0), Some(()));
        assert_eq!(limit.release(1), None);
    }
}
