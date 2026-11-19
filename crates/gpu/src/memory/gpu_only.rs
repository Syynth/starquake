//! [`GpuOnlyStore`]: the host pages behind GPU-only memory, the `Target::GpuOnly` pages of sparse
//! regions (gpu-protocol spec, "GPU-only memory"; review guide, M6; slice G22).
//!
//! A sparse region (gsv 214) is a GPU range with nothing of the title's behind it, yet render
//! targets and other GPU data live there (rows 57, 80, 81 and 85 of `eden-nx2/NX2_NOTES.md`).
//! The owner decided (decision log, 2026-11-12, "Ten coupled graphics decisions", item 2;
//! graphics-server decision 4) that it is backed lazily on the host and not charged to the
//! title's pools, revisited if a trace shows the console charges it to remote memory. This store
//! is that backing:
//!
//! - **Lazy.** A page of [`PAGE_SIZE`] bytes is allocated the first time the GPU writes any byte
//!   of it, keyed by its region and its page-aligned offset into the region.
//! - **Unwritten pages read as zero.** A **guess** (graphics-server, "What to verify before
//!   building", item 6). PR #4588 reads every reserved (sparse) page as zero too, because it
//!   backs nothing there: `ReadBlockImpl`'s `set_to_zero` for reserved pages, in
//!   `eden-pr4588/src/video_core/memory_manager.cpp` @ e5fb3b97.
//! - **Freed only by an applied destroy.** [`crate::vm::GpuVm::apply`] of a
//!   [`VmUpdate::DestroySparseRegion`](crate::interface::VmUpdate::DestroySparseRegion) frees
//!   every page of that region, under the table's write lock, so no write can translate to the
//!   region and land after the free (see [`crate::vm::GpuVm`]'s "The GPU-only store"). An unmap,
//!   a memory destroy or a kind change frees nothing: the region and its offsets outlive any one
//!   mapping of them.
//! - **Charged to nothing of the title's.** The pages are host allocations. Nothing here reaches
//!   guest memory or the kernel's pools; gsv keeps the region extents and their accounting.
//!
//! # Bounds
//!
//! A page is allocated only by a write that translated to it, so the store holds at most one page
//! per (region, page offset) that a `GpuOnly` mapping has led to and whose region has not been
//! destroyed since. gsv makes every `GpuOnly` mapping from a sparse region it validated and
//! accounts for (graphics-server slice 7), so the store is bounded by the live sparse regions'
//! extents. It sets no cap of its own: one would be a figure no reference gives.
//!
//! # Divergences from the references
//!
//! - **The pages are backed.** PR #4588 backs no sparse page: its `WriteBlockImpl` drops writes
//!   to reserved pages (`just_advance`) and its `ReadBlockImpl` reads them as zero
//!   (`eden-pr4588/src/video_core/memory_manager.cpp` @ e5fb3b97). Eden-NX2 backed each region
//!   with pages of the application pool, and the title ran out of memory (row 80). Here a write
//!   lands and reads back (`tests::a_write_reads_back_and_its_neighbours_read_zero`), as the
//!   owner's decision has it.
//!
//! # What is snapshotted
//!
//! Nothing: the store's contents are not in gsv's snapshot, and a GPU snapshot that reads them
//! back is deferred (gpu-protocol, "GPU-only memory"). [`crate::vm::GpuVm::restore`] starts with
//! an empty store.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use starquake_memory::PAGE_SIZE;
use thiserror::Error;

use super::page_pieces;
use crate::interface::RegionId;

#[cfg(test)]
mod tests;

/// [`PAGE_SIZE`] as a length.
#[expect(
    clippy::cast_possible_truncation,
    reason = "PAGE_SIZE is 4 KiB, which fits the usize of every target this builds for"
)]
const PAGE_BYTES: usize = PAGE_SIZE as usize;

/// One host page.
type Page = Box<[u8; PAGE_BYTES]>;

/// The pages, by region and page-aligned offset into it.
type Pages = BTreeMap<(RegionId, u64), Page>;

/// A GPU-only access whose end, offset plus length, is past the end of the 64-bit offsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error(
    "a GPU-only access of {len:#x} bytes at offset {offset:#x} of sparse region {region} runs \
     past the last offset",
    region = region.0
)]
pub struct GpuOnlyRangeError {
    /// The region.
    pub region: RegionId,
    /// The access's first offset.
    pub offset: u64,
    /// Its length in bytes.
    pub len: u64,
}

/// The host pages behind GPU-only memory (module docs).
///
/// One per address table: [`crate::vm::GpuVm`] owns one, so a region's pages are those of the
/// address space its mappings are in. Every method takes `&self`; the pages sit behind one
/// mutex, held for one access.
#[derive(Debug, Default)]
pub struct GpuOnlyStore {
    pages: Mutex<Pages>,
}

impl GpuOnlyStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, Pages> {
        self.pages.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Copies the `buf.len()` bytes at `offset` into `region` into `buf`; bytes of a page never
    /// written are zero.
    ///
    /// Allocates nothing. The only failure is an end past the 64-bit offsets, which leaves `buf`
    /// as it was.
    pub fn read(
        &self,
        region: RegionId,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<(), GpuOnlyRangeError> {
        let pieces = pieces(region, offset, buf.len())?;
        let pages = self.lock();
        for (at, range) in pieces {
            let (page, within) = split(at);
            let out = &mut buf[range];
            match pages.get(&(region, page)) {
                Some(bytes) => out.copy_from_slice(&bytes[within..within + out.len()]),
                None => out.fill(0),
            }
        }
        Ok(())
    }

    /// Copies `data` to `offset` into `region`, allocating each page it touches that is not
    /// allocated yet.
    ///
    /// Crate-private: the production memory calls it under the address table's read lock
    /// ([`crate::vm::GpuVm`]'s "The GPU-only store"), so an applied destroy cannot free a page
    /// between a write's translation and its store. The only failure is an end past the 64-bit
    /// offsets, which writes nothing.
    pub(crate) fn write(
        &self,
        region: RegionId,
        offset: u64,
        data: &[u8],
    ) -> Result<(), GpuOnlyRangeError> {
        let pieces = pieces(region, offset, data.len())?;
        let mut pages = self.lock();
        for (at, range) in pieces {
            let (page, within) = split(at);
            let input = &data[range];
            let bytes = pages
                .entry((region, page))
                .or_insert_with(|| Box::new([0; PAGE_BYTES]));
            bytes[within..within + input.len()].copy_from_slice(input);
        }
        Ok(())
    }

    /// Frees every page of `region` and returns how many there were.
    ///
    /// Crate-private: only an applied [`VmUpdate::DestroySparseRegion`] frees
    /// ([`crate::vm::GpuVm::apply`], under the table's write lock).
    ///
    /// [`VmUpdate::DestroySparseRegion`]: crate::interface::VmUpdate::DestroySparseRegion
    pub(crate) fn free_region(&self, region: RegionId) -> usize {
        let mut pages = self.lock();
        // The region's keys are contiguous in the map's order: split off everything from the
        // region's first key, then put back everything from the next region's first key.
        let mut from_region = pages.split_off(&(region, 0));
        let mut after = match region.0.checked_add(1) {
            Some(next) => from_region.split_off(&(RegionId(next), 0)),
            None => Pages::new(),
        };
        pages.append(&mut after);
        from_region.len()
    }

    /// How many host pages the store holds, over every region.
    pub fn resident_pages(&self) -> usize {
        self.lock().len()
    }

    /// How many host pages the store holds for `region`.
    pub fn resident_pages_of(&self, region: RegionId) -> usize {
        self.lock().range((region, 0)..=(region, u64::MAX)).count()
    }
}

/// The page-sized pieces of the `len` bytes at `offset` into `region`.
fn pieces(
    region: RegionId,
    offset: u64,
    len: usize,
) -> Result<impl Iterator<Item = (u64, std::ops::Range<usize>)>, GpuOnlyRangeError> {
    page_pieces(offset, len).map_err(|_| GpuOnlyRangeError {
        region,
        offset,
        len: u64::try_from(len).unwrap_or(u64::MAX),
    })
}

/// The page-aligned offset holding `at`, and `at`'s position within that page.
fn split(at: u64) -> (u64, usize) {
    let within = at % PAGE_SIZE;
    // `within` is below `PAGE_SIZE`, which is `PAGE_BYTES`.
    (at - within, usize::try_from(within).unwrap_or(0))
}
