//! Thread-local pages: the pages a process's threads' thread-local regions are carved from.
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), paths relative to the reference folder:
//! `eden/src/core/hle/kernel/k_thread_local_page.{h,cpp}` (a page holds
//! `RegionsPerPage = PageSize / Svc::ThreadLocalRegionSize` regions; `Reserve` takes the lowest
//! free one; `Initialize` maps the page `KMemoryState::ThreadLocal`, `UserReadWrite`) and
//! `eden/src/core/hle/kernel/k_process.cpp`, `CreateThreadLocalRegion` and
//! `DeleteThreadLocalRegion` (a region comes from a partially used page if there is one, else
//! from a new page; releasing a region of a page this process does not have is
//! `ResultInvalidAddress`).
//!
//! Eden keeps the partially used pages in a tree ordered by address and takes the first, so the
//! lowest partially used page gives the next region. A map ordered by address does the same.
//!
//! # Divergence
//!
//! A page whose regions are all free again stays mapped and is kept for the next region. Eden
//! unmaps it and frees the page (`KThreadLocalPage::Finalize`). The address space has no unmap
//! for thread-local memory yet; when it does, freeing the page is one call here. Test:
//! `a_page_whose_regions_are_all_free_stays_and_is_reused`.

use std::collections::BTreeMap;

use starquake_memory::PAGE_SIZE;

use crate::memory::{AddressSpace, MemoryPermission, MemoryState};
use crate::result::{CallResult, KernelError};

/// Bytes of one thread's thread-local region (`Svc::ThreadLocalRegionSize = 0x200`,
/// `eden/src/core/hle/kernel/svc_types.h`).
pub const THREAD_LOCAL_REGION_SIZE: u64 = 0x200;

/// Regions in one page (`KThreadLocalPage::RegionsPerPage`).
pub const REGIONS_PER_PAGE: usize = (PAGE_SIZE / THREAD_LOCAL_REGION_SIZE) as usize;

/// Where in its thread-local region the kernel writes a thread's own handle: the
/// `current_thread_handle` field (`static_assert(offsetof(ThreadLocalRegion,
/// current_thread_handle) == 0x110)`, `eden/src/core/hle/kernel/k_thread.cpp`).
pub const CURRENT_THREAD_HANDLE_OFFSET: u64 = 0x110;

/// One process's thread-local pages: for each page, by address, which regions are in use.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThreadLocalPages {
    pages: BTreeMap<u64, [bool; REGIONS_PER_PAGE]>,
}

impl ThreadLocalPages {
    /// No pages yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The pages, by address, each with which of its regions are in use.
    pub fn pages(&self) -> impl Iterator<Item = (u64, &[bool; REGIONS_PER_PAGE])> {
        self.pages.iter().map(|(address, used)| (*address, used))
    }

    /// How many regions are in use.
    #[must_use]
    pub fn used(&self) -> usize {
        self.pages
            .values()
            .map(|used| used.iter().filter(|in_use| **in_use).count())
            .sum()
    }

    /// Takes a free region and returns its address (`CreateThreadLocalRegion`): the lowest free
    /// region of the lowest page that has one, or the first region of a page newly mapped in
    /// `space`. A page that cannot be mapped fails the call with the address space's error, and
    /// nothing changes.
    pub fn reserve(&mut self, space: &mut AddressSpace) -> CallResult<u64> {
        let free = self.pages.iter_mut().find_map(|(page, used)| {
            let index = used.iter().position(|in_use| !in_use)?;
            Some((*page, used, index))
        });
        if let Some((page, used, index)) = free {
            if let Some(slot) = used.get_mut(index) {
                *slot = true;
            }
            return Ok(region_address(page, index));
        }
        let page = space.map_pages_in_region(
            1,
            MemoryState::ThreadLocal,
            MemoryPermission::USER_READ_WRITE,
        )?;
        let mut used = [false; REGIONS_PER_PAGE];
        used[0] = true;
        self.pages.insert(page, used);
        Ok(page)
    }

    /// Gives back the region at `address` (`DeleteThreadLocalRegion`).
    ///
    /// [`KernelError::InvalidAddress`] when no page of this process holds `address`, Eden's
    /// result. An address that is not the start of a region, or a region that is already free,
    /// is also [`KernelError::InvalidAddress`]: Eden asserts both (`GetRegionIndex`), and only
    /// the kernel's own bookkeeping passes this an address.
    pub fn release(&mut self, address: u64) -> CallResult<()> {
        let invalid = KernelError::InvalidAddress;
        let page = address - address % PAGE_SIZE;
        let used = self.pages.get_mut(&page).ok_or(invalid)?;
        let offset = address - page;
        if !offset.is_multiple_of(THREAD_LOCAL_REGION_SIZE) {
            return Err(invalid.into());
        }
        let index = usize::try_from(offset / THREAD_LOCAL_REGION_SIZE).map_err(|_| invalid)?;
        let slot = used
            .get_mut(index)
            .filter(|in_use| **in_use)
            .ok_or(invalid)?;
        *slot = false;
        Ok(())
    }
}

/// The address of region `index` of the page at `page`.
fn region_address(page: u64, index: usize) -> u64 {
    // `index` is below REGIONS_PER_PAGE, so the offset is inside the page.
    page + (index as u64) * THREAD_LOCAL_REGION_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::fixture;
    use crate::memory::testing::MapperCall;
    use crate::memory::{GUARD_PAGES, MemoryState};
    use crate::result::SvcError;
    use starquake_memory::Perms;

    #[test]
    fn a_page_holds_eight_regions_in_order_then_a_new_page_is_mapped() {
        let (mut space, log, _) = fixture::loaded();
        let mut tls = ThreadLocalPages::new();
        let first = tls.reserve(&mut space).unwrap();
        // The page goes where the kernel places thread-local memory: the kernel-map region,
        // past its guard pages.
        let kernel_map = space.regions().kernel_map;
        assert_eq!(first, kernel_map.start + GUARD_PAGES * PAGE_SIZE);
        assert_eq!(
            log.take(),
            vec![MapperCall::Map {
                address: first,
                size: PAGE_SIZE,
                perms: Perms::ReadWrite
            }]
        );
        let info = space.query_info(first).unwrap();
        assert_eq!(info.state(), MemoryState::ThreadLocal);
        for index in 1..REGIONS_PER_PAGE {
            let expected = first + (index as u64) * THREAD_LOCAL_REGION_SIZE;
            assert_eq!(tls.reserve(&mut space), Ok(expected));
        }
        assert_eq!(
            log.take(),
            Vec::new(),
            "no page is mapped while one has room"
        );
        // The ninth region needs a second page.
        let second = tls.reserve(&mut space).unwrap();
        assert_eq!(second % PAGE_SIZE, 0);
        assert_ne!(second, first);
        assert_eq!(log.take().len(), 1);
        assert_eq!(tls.used(), REGIONS_PER_PAGE + 1);
    }

    #[test]
    fn a_released_region_is_the_next_one_reserved() {
        let (mut space, _, _) = fixture::loaded();
        let mut tls = ThreadLocalPages::new();
        let regions: Vec<u64> = (0..3).map(|_| tls.reserve(&mut space).unwrap()).collect();
        tls.release(regions[1]).unwrap();
        assert_eq!(tls.used(), 2);
        assert_eq!(tls.reserve(&mut space), Ok(regions[1]));
        // The lowest free region wins: free the last and the first, the first comes back first.
        tls.release(regions[2]).unwrap();
        tls.release(regions[0]).unwrap();
        assert_eq!(tls.reserve(&mut space), Ok(regions[0]));
        assert_eq!(tls.reserve(&mut space), Ok(regions[2]));
    }

    #[test]
    fn a_page_whose_regions_are_all_free_stays_and_is_reused() {
        let (mut space, log, _) = fixture::loaded();
        let mut tls = ThreadLocalPages::new();
        let first = tls.reserve(&mut space).unwrap();
        drop(log.take());
        tls.release(first).unwrap();
        assert_eq!(tls.used(), 0);
        // Still mapped (Eden would unmap it), and still this process's.
        assert_eq!(log.take(), Vec::new());
        assert_eq!(
            space.query_info(first).unwrap().state(),
            MemoryState::ThreadLocal
        );
        assert_eq!(tls.pages().count(), 1);
        assert_eq!(tls.reserve(&mut space), Ok(first));
        assert_eq!(log.take(), Vec::new());
    }

    #[test]
    fn releasing_what_is_not_a_reserved_region_is_an_invalid_address() {
        let (mut space, _, _) = fixture::loaded();
        let mut tls = ThreadLocalPages::new();
        let first = tls.reserve(&mut space).unwrap();
        let invalid = Err(SvcError::Kernel(KernelError::InvalidAddress));
        // No page there.
        assert_eq!(tls.release(first + PAGE_SIZE), invalid);
        // Inside a region, not at its start.
        assert_eq!(tls.release(first + 8), invalid);
        // A region of the page that is free.
        assert_eq!(tls.release(first + THREAD_LOCAL_REGION_SIZE), invalid);
        // The last region of the page, free, and the byte before the page.
        let last = first + PAGE_SIZE - THREAD_LOCAL_REGION_SIZE;
        assert_eq!(tls.release(last), invalid);
        assert_eq!(tls.release(first - 1), invalid);
        // Releasing twice.
        tls.release(first).unwrap();
        assert_eq!(tls.release(first), invalid);
    }

    #[test]
    fn a_page_the_address_space_cannot_place_fails_and_changes_nothing() {
        let (mut space, log, _) = fixture::loaded();
        let mut tls = ThreadLocalPages::new();
        log.refuse(true);
        assert!(matches!(
            tls.reserve(&mut space),
            Err(SvcError::Internal(_))
        ));
        assert_eq!(tls.pages().count(), 0);
        // The next try maps the page normally.
        log.refuse(false);
        assert!(tls.reserve(&mut space).is_ok());
    }
}
