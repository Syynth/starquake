//! How the GPU reads and writes memory (gpu-protocol spec, "Reading and writing guest memory",
//! slice G7, and "GPU-only memory", slice G22): the [`GpuMemory`] trait, the [`Translate`] seam
//! an address table offers, the per-thread [`TranslationCache`] keyed by the table's version, the
//! production memory [`VmMemory`] and the [`GpuOnlyStore`] behind sparse regions.
//!
//! - [`GpuMemory`] is the GPU's access trait. Its production implementation is [`VmMemory`]
//!   (slice G22): it translates through an address space's [`crate::vm::GpuVm`] and resolves
//!   each `Target` (guest memory, the GPU-only store, the syncpoint shim). The capture's
//!   recording and replay implementations are [`crate::capture::RecordingMemory`] and
//!   [`crate::capture::ReplayMemory`] (slice G20); with the `test-support` feature this module exports
//!   [`TestGpuMemory`], which maps page ranges of a GPU address space onto a
//!   [`starquake_memory::GuestMemory`].
//! - [`GpuOnlyStore`] holds the host pages behind `GpuOnly` targets: allocated on the first
//!   write, zero where never written, freed by an applied sparse-region destroy, and charged to
//!   nothing of the title's.
//! - [`Translate`] is what an address table offers a cache: a version ("generation") that every
//!   map, ordered update or kind change moves, and the translation of one page. The production
//!   table, [`crate::vm::GpuVm`] (graphics-server slice 6), implements it.
//! - [`TranslationCache`] is one GPU thread's last few page translations, emptied whenever the
//!   table's version has moved since they were made. Each thread owns its own: one cache shared by
//!   Eden-NX2's threads was cleared by a map on a service thread between the GPU thread's check of
//!   an entry and its use, and the GPU thread copied from address 0 (row 125 of
//!   `eden-nx2/NX2_NOTES.md`; `eden-nx2/src/core/device_memory_manager.h`, `TranslationCache`).
//!
//! Translation works on pages of [`PAGE_SIZE`] (4 KiB): Eden-NX2's device memory manager uses
//! `YUZU_PAGEBITS = 12` (`eden-nx2/src/core/memory.h`), and gsv maps GPU memory in 4 KiB units
//! (`NormalMapGpuMemory4K`, graphics-server "GPU memory: the gpu session and its worker").
//!
//! The spec's sketch of the trait also has `view(va, len) -> RawView`, a raw view for hot paths.
//! The memory crate's raw view type (execution model, "Host access to guest memory") does not
//! exist yet, so the method is not here; it joins the trait with that type.
//!
//! Where this differs from Eden-NX2 (`eden-nx2/src/core/device_memory_manager.inc`):
//!
//! - An access that reaches an unmapped page is an error, [`GpuFault::Unmapped`], naming the first
//!   unmapped address. Eden-NX2's `ReadBlock` logs and fills the unmapped part with zeros, and its
//!   `WriteBlock` logs and drops it (`WalkBlock`'s `on_unmapped`). Pinned by
//!   `tests::a_read_into_a_gap_faults_at_the_first_unmapped_byte`.
//! - A cache is bound to its table when it is made (it holds an `Arc` of it), so no access can
//!   reach a cache filled from another table. Eden-NX2 keeps one `thread_local` cache per thread
//!   and checks its `owner` pointer at every access instead.
//!
//! The rest mirrors Eden-NX2's cache: four entries ([`TRANSLATION_CACHE_SLOTS`]) replaced in
//! turn, a page that does not translate is not cached, and the version is read before the
//! entries are looked at (`Translations()`, which compares `translation_generation` with the
//! cache's and empties it on a difference).

mod cache;
mod gpu_only;
#[cfg(any(test, feature = "test-support"))]
mod test_memory;
mod vm_memory;

#[cfg(test)]
mod tests;

use starquake_memory::{Fault, PAGE_SIZE};

pub use cache::{TRANSLATION_CACHE_SLOTS, TranslationCache};
pub use gpu_only::{GpuOnlyRangeError, GpuOnlyStore};
#[cfg(any(test, feature = "test-support"))]
pub use test_memory::TestGpuMemory;
pub use vm_memory::{SHIM_STRIDE, VmMemory};

/// Why a GPU access failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GpuFault {
    /// No mapping covers this GPU address.
    #[error("GPU address {va:#x} is not mapped")]
    Unmapped {
        /// The first GPU address of the access that is not mapped.
        va: u64,
    },
    /// The access's end is past the 64-bit address space.
    #[error("a GPU access of {len:#x} bytes at {va:#x} runs past the end of the address space")]
    OutOfRange {
        /// The start of the access.
        va: u64,
        /// Its length in bytes.
        len: u64,
    },
    /// The GPU address is mapped, but the guest memory behind it refused the access.
    #[error("GPU address {va:#x}: {fault}")]
    Guest {
        /// The GPU address of the piece of the access that faulted.
        va: u64,
        /// The guest memory's fault, at the CPU address the GPU address translated to.
        fault: Fault,
    },
    /// A write reaches the syncpoint shim but is not a release to one syncpoint: it does not lie
    /// wholly within the shim page, or does not start on a syncpoint's word ([`VmMemory`]'s
    /// "The syncpoint shim").
    #[error(
        "GPU address {va:#x} is in the syncpoint shim, which takes only a write that starts on one \
         syncpoint's word and stays in the shim page"
    )]
    ShimWrite {
        /// The GPU address of the first byte of the write that lies in the shim.
        va: u64,
    },
}

/// Byte-level access to one GPU address space, by copy.
///
/// As the execution model requires, nothing hands out a Rust slice into guest memory.
pub trait GpuMemory: Send + Sync {
    /// Copies `buf.len()` bytes starting at GPU address `va` into `buf`. On an error, `buf` may
    /// hold part of the data.
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), GpuFault>;
    /// Copies `data` to GPU address `va`.
    fn write(&self, va: u64, data: &[u8]) -> Result<(), GpuFault>;
    /// The address table's version: moved by every map, ordered update or kind change, so a
    /// thread's caches know to drop what they derived from the old table.
    fn generation(&self) -> u64;
    /// Whether every page of the `len` bytes at `va` leads to GPU-only memory (a sparse region's
    /// `GpuOnly` pages), so a render target there stays a host-only texture (gpu-protocol,
    /// "GPU-only memory"). False for an empty range, a range whose end is past the 64-bit space,
    /// and a range any page of which is unmapped or leads elsewhere.
    ///
    /// The default answers false: a memory with no GPU-only pages ([`TestGpuMemory`]) has
    /// nothing to report. [`VmMemory`] answers from its table.
    fn is_gpu_only(&self, va: u64, len: u64) -> bool {
        let _ = (va, len);
        false
    }
}

/// What an address table offers a [`TranslationCache`].
pub trait Translate: Send + Sync {
    /// Where one page leads (for the production table, a `Target`). Only `Copy` values are
    /// cached.
    type Target: Copy;

    /// The table's version. An implementation moves it *after* each change to its translations
    /// is visible (a release store or stronger), so a reader that sees the old version may have
    /// seen either table, and one that sees the new version sees the new table.
    fn generation(&self) -> u64;

    /// Where the page starting at `page` (a multiple of [`PAGE_SIZE`]) leads, or `None` if it is
    /// not mapped.
    fn translate_page(&self, page: u64) -> Option<Self::Target>;
}

/// The start of the page holding `va`.
pub(crate) const fn page_of(va: u64) -> u64 {
    va - va % PAGE_SIZE
}

/// Splits the `len` bytes at GPU address `va` into the pieces that lie in one page each: for
/// each, its GPU address and its range within the caller's buffer, in address order.
///
/// A zero-length access has no pieces, whatever `va` is (Eden-NX2's `WalkBlock` runs no
/// iteration for size 0). Otherwise the access's end, `va + len`, must be representable, or the
/// error is [`GpuFault::OutOfRange`].
///
/// [`TestGpuMemory`], [`VmMemory`] and [`GpuOnlyStore`] (over a region's offsets) all walk their
/// accesses with it.
pub(crate) fn page_pieces(
    va: u64,
    len: usize,
) -> Result<impl Iterator<Item = (u64, std::ops::Range<usize>)>, GpuFault> {
    let len64 = u64::try_from(len).map_err(|_| GpuFault::OutOfRange { va, len: u64::MAX })?;
    if va.checked_add(len64).is_none() {
        return Err(GpuFault::OutOfRange { va, len: len64 });
    }
    let mut next_va = va;
    let mut done = 0usize;
    Ok(std::iter::from_fn(move || {
        if done >= len {
            return None;
        }
        let left_in_page = PAGE_SIZE - next_va % PAGE_SIZE;
        let step = usize::try_from(left_in_page).map_or(len - done, |left| left.min(len - done));
        let piece = (next_va, done..done + step);
        done += step;
        // `va + len` is representable, so every piece's end is too.
        next_va = next_va.wrapping_add(step as u64);
        Some(piece)
    }))
}
