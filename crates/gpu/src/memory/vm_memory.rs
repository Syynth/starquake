//! [`VmMemory`]: the production [`GpuMemory`], over one address space's [`GpuVm`] (gpu-protocol
//! spec, "Reading and writing guest memory" and "GPU-only memory"; slice G22).
//!
//! The owner decided that the production implementation lives in this crate, over the address
//! table (decision log, 2026-11-12, "Syncpoints and the GPU address table live in the GPU
//! crate"). Every access is split into pieces of one page ([`super::page_pieces`]), and each
//! piece is translated through the thread's own [`TranslationCache`] and resolved by its
//! [`Target`]:
//!
//! - `Target::Guest { cpu_va }` reads and writes the [`GuestMemory`] at that CPU address (GPU
//!   address equals CPU address for memory registered with the session, row 30 of
//!   `eden-nx2/NX2_NOTES.md`; a 211 map can lead elsewhere, and the target says where);
//! - `Target::GpuOnly { region, offset }` reads and writes the table's
//!   [`GpuOnlyStore`](super::GpuOnlyStore);
//! - `Target::SyncpointShim` reads through [`Syncpoints::read`], and a release written to it is a
//!   [`Syncpoints::increment`] ("The syncpoint shim", below).
//!
//! # One per GPU thread
//!
//! Each GPU thread keeps its own translation cache, keyed by the table's version (row 125: one
//! cache shared by Eden-NX2's threads was cleared by a map on a service thread between a check
//! and a use, and the GPU thread copied from address 0). A `VmMemory` holds one cache, behind a
//! mutex that its own thread never contends: a thread makes its own with [`VmMemory::new`] or
//! [`VmMemory::for_thread`], which shares the table, the guest memory and the syncpoints and
//! starts an empty cache. Sharing one `VmMemory` between threads is safe (the mutex is held for
//! the whole access, so no entry is cleared between its check and its use), but serialises
//! their accesses.
//!
//! [`GpuMemory::generation`] is the table's version, [`Translate::generation`] of the
//! [`GpuVm`]. The proposal that this slice take it from there (`docs/proposals.md`, review notes
//! on #197) is unvetted; it is the only version the trait's doc ("the address table's version")
//! can mean here, and it is raised as an owner question in the pull request.
//!
//! # Reads and writes
//!
//! A read translates and copies piece by piece, in address order; the first piece that is not
//! mapped is [`GpuFault::Unmapped`] naming that piece's first address, and one the guest memory
//! refuses is [`GpuFault::Guest`]. On an error, `buf` holds the pieces before the failing one.
//!
//! A write first translates every piece and checks each shim piece (below), in address order,
//! returning the first failure with nothing written. Then it writes piece by piece. A `GpuOnly`
//! piece is written through [`GpuVm::write_gpu_only`], which translates the page afresh under the
//! table's read lock, so that an applied sparse-region destroy cannot free the page between the
//! write's translation and its store (the table's module docs, "The GPU-only store"); if that
//! fresh translation leads elsewhere, the piece is written where it now leads. A piece that turns
//! unmapped, or a guest memory that refuses a piece, after the first check fails the write with
//! the pieces before it kept, as [`super::TestGpuMemory`] keeps them.
//!
//! # The syncpoint shim
//!
//! The shim page ([`crate::vm::SHIM_PAGES`] page, nvhost-as-gpu ioctl 0xB, row 46) holds one
//! 32-bit word per syncpoint, [`SHIM_STRIDE`] bytes apart, for
//! [`SYNCPOINT_COUNT`](crate::interface::SYNCPOINT_COUNT) syncpoints:
//! PR #4588's `GetNativeSyncpointShim` reports `stride = sizeof(u32)` and `count = 192`
//! (`eden-pr4588/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.cpp` @ e5fb3b97).
//!
//! - **Reads** see the page as bytes: the word at `id * SHIM_STRIDE` is
//!   [`Syncpoints::read`]`(id)`, little-endian, read once per word an access touches, and the
//!   bytes past the last syncpoint's word read as zero. PR's `MemoryManager::Read<u32>`
//!   (`eden-pr4588/src/video_core/memory_manager.cpp` @ e5fb3b97) answers the syncpoint's value
//!   for a 4-byte read with `addr - shim < 192 * sizeof(u32)` and `IsAligned(addr - shim,
//!   sizeof(u32))`; this reading answers the same for every such read.
//! - **A write is one increment.** The spec turns a release to the shim into an increment of
//!   that syncpoint, performed like any fenced release (gpu-protocol, "Semaphore acquires"; a
//!   **guess** about how increments reach syncpoints on this GPU, to confirm with a trace of
//!   NVN2's fence path). A release is one write of 4, 8 or 16 bytes (`back::fence`), so a write
//!   that lies wholly within the shim page and starts on syncpoint `id`'s word increments `id`
//!   by one, once, whatever its length and bytes; a reduction, which reads the word and writes
//!   the result, likewise increments by one. Any other write that reaches the shim, one that
//!   starts past the last syncpoint's word or off a word boundary, or that runs into or out of
//!   the shim page, is [`GpuFault::ShimWrite`] and increments nothing. An empty write has no
//!   pieces and increments nothing.
//!
//! # Divergences from the references
//!
//! Each is pinned by the test of `vm_memory/tests.rs` it names:
//!
//! - **The shim answers reads of any width and alignment.** PR answers only an aligned 4-byte
//!   `Read<u32>` from the counters; any other read of the shim falls through to the page, a
//!   reserved one with nothing behind it, so `Read<T>` asserts and returns zero and
//!   `ReadBlockImpl` reads zero (`set_to_zero`). Here every read sees the counters' bytes
//!   (`the_shim_reads_counters_at_any_width_and_alignment`). By graphics-server decision 8's
//!   "where one answer contains the other, the larger": this answer contains PR's for every read
//!   PR answers from the counters.
//! - **A write to the shim increments.** PR's `Write<T>` and `WriteBlockImpl` drop a write to the
//!   reserved shim page (`ASSERT(false)`, `just_advance`); the spec's guess makes a release there
//!   an increment (`a_release_to_the_shim_increments_its_syncpoint_once`), and a write that is
//!   not a release is refused rather than dropped (`writes_the_shim_cannot_take_are_refused`).
//! - **An unmapped page is an error**, as in [`super::TestGpuMemory`] (the trait's module docs
//!   list the departure from Eden-NX2's zero fill).
//!
//! # Not here
//!
//! - `GpuMemory::view` (a raw view for hot paths) waits for the memory crate's `RawView`, which
//!   is only proposed (`docs/proposals.md`); copies are enough here.
//! - The semaphore mirror page (row 35, graphics-server verification item 8): releases land at
//!   the address given.
//! - The fence path (`back::fence`) latches waiters with the words a release would have
//!   written, so a release to the shim latches with its payload, not the counter's new value;
//!   an open item listed in that module's docs.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use starquake_memory::{GuestMemory, PAGE_SIZE};

use super::{GpuFault, GpuMemory, Translate, TranslationCache, page_pieces};
use crate::interface::{SyncpointId, Target};
use crate::syncpoint::Syncpoints;
use crate::vm::{GpuOnlyWrite, GpuVm, Translation};

#[cfg(test)]
mod tests;

/// Bytes between two syncpoints' words in the shim: PR #4588's `GetNativeSyncpointShim` sets
/// `stride = sizeof(u32)` (`eden-pr4588/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.cpp`
/// @ e5fb3b97).
pub const SHIM_STRIDE: u64 = WORD_BYTES as u64;

/// The bytes of one shim word: a `u32`, as PR's stride.
const WORD_BYTES: usize = size_of::<u32>();

/// The production [`GpuMemory`] of one address space, for one GPU thread (module docs).
pub struct VmMemory<G: ?Sized> {
    /// The address space's table, with its GPU-only store.
    vm: Arc<GpuVm>,
    /// The title's memory, behind `Guest` targets.
    guest: Arc<G>,
    /// The counters behind the shim.
    syncpoints: Arc<Syncpoints>,
    /// This thread's translations.
    cache: Mutex<TranslationCache<GpuVm>>,
}

impl<G: GuestMemory + ?Sized> VmMemory<G> {
    /// The memory of the address space `vm`, whose `Guest` targets lead into `guest` and whose
    /// shim reads `syncpoints`, with an empty translation cache for the calling thread.
    pub fn new(vm: Arc<GpuVm>, guest: Arc<G>, syncpoints: Arc<Syncpoints>) -> Self {
        let cache = Mutex::new(TranslationCache::new(Arc::clone(&vm)));
        Self {
            vm,
            guest,
            syncpoints,
            cache,
        }
    }

    /// The same address space for another GPU thread: the same table, guest memory and
    /// syncpoints, and an empty translation cache of its own (module docs, "One per GPU thread").
    #[must_use]
    pub fn for_thread(&self) -> Self {
        Self::new(
            Arc::clone(&self.vm),
            Arc::clone(&self.guest),
            Arc::clone(&self.syncpoints),
        )
    }

    /// The address space's table.
    pub fn vm(&self) -> &Arc<GpuVm> {
        &self.vm
    }

    fn cache(&self) -> MutexGuard<'_, TranslationCache<GpuVm>> {
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Copies one piece, within the page holding `va`, into `out`, from where `translation`
    /// leads.
    fn read_piece(
        &self,
        va: u64,
        translation: Translation,
        out: &mut [u8],
    ) -> Result<(), GpuFault> {
        let within = va % PAGE_SIZE;
        match translation.target {
            Target::Guest { cpu_va } => {
                let cpu = at(cpu_va, within, va, out.len())?;
                self.guest
                    .read(cpu, out)
                    .map_err(|fault| GpuFault::Guest { va, fault })
            }
            Target::GpuOnly { region, offset } => {
                let at = at(offset, within, va, out.len())?;
                self.vm
                    .gpu_only()
                    .read(region, at, out)
                    .map_err(|_| out_of_range(va, out.len()))
            }
            Target::SyncpointShim => {
                read_shim(&self.syncpoints, within, out);
                Ok(())
            }
        }
    }

    /// Writes one piece, `data` within the page holding `va`, of a write of `write_len` bytes,
    /// where `translation` leads. A `GpuOnly` piece goes through [`GpuVm::write_gpu_only`]
    /// (module docs, "Reads and writes").
    fn write_piece(
        &self,
        va: u64,
        translation: Translation,
        data: &[u8],
        write_len: usize,
    ) -> Result<(), GpuFault> {
        match translation.target {
            Target::Guest { cpu_va } => {
                let cpu = at(cpu_va, va % PAGE_SIZE, va, data.len())?;
                self.guest
                    .write(cpu, data)
                    .map_err(|fault| GpuFault::Guest { va, fault })
            }
            Target::GpuOnly { .. } => {
                match self
                    .vm
                    .write_gpu_only(va, data)
                    .map_err(|_| out_of_range(va, data.len()))?
                {
                    GpuOnlyWrite::Written => Ok(()),
                    GpuOnlyWrite::Elsewhere(Some(
                        now @ Translation {
                            target: Target::Guest { .. } | Target::SyncpointShim,
                            ..
                        },
                    )) => self.write_piece(va, now, data, write_len),
                    // The fresh translation is never `GpuOnly` here: `write_gpu_only` writes
                    // that.
                    GpuOnlyWrite::Elsewhere(_) => Err(GpuFault::Unmapped { va }),
                }
            }
            Target::SyncpointShim => {
                // The shim check ran before the first write; a piece that only now leads to the
                // shim (a map landed since) is checked here.
                let id = shim_release(va, data.len(), write_len)?;
                self.syncpoints.increment(id);
                Ok(())
            }
        }
    }
}

impl<G: GuestMemory + ?Sized> GpuMemory for VmMemory<G> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), GpuFault> {
        let pieces = page_pieces(va, buf.len())?;
        let mut cache = self.cache();
        for (piece_va, range) in pieces {
            let translation = cache
                .translate(piece_va)
                .ok_or(GpuFault::Unmapped { va: piece_va })?;
            self.read_piece(piece_va, translation, &mut buf[range])?;
        }
        Ok(())
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), GpuFault> {
        let mut cache = self.cache();
        // Every piece translates, and every shim piece is a release, before any byte is written.
        for (piece_va, range) in page_pieces(va, data.len())? {
            let translation = cache
                .translate(piece_va)
                .ok_or(GpuFault::Unmapped { va: piece_va })?;
            if translation.target == Target::SyncpointShim {
                shim_release(piece_va, range.len(), data.len())?;
            }
        }
        for (piece_va, range) in page_pieces(va, data.len())? {
            let translation = cache
                .translate(piece_va)
                .ok_or(GpuFault::Unmapped { va: piece_va })?;
            self.write_piece(piece_va, translation, &data[range], data.len())?;
        }
        Ok(())
    }

    fn generation(&self) -> u64 {
        Translate::generation(&*self.vm)
    }

    fn is_gpu_only(&self, va: u64, len: u64) -> bool {
        let Ok(len) = usize::try_from(len) else {
            return false;
        };
        let Ok(mut pieces) = page_pieces(va, len) else {
            return false;
        };
        let mut cache = self.cache();
        let mut any = false;
        let all = pieces.all(|(piece_va, _)| {
            any = true;
            matches!(
                cache.translate(piece_va).map(|t| t.target),
                Some(Target::GpuOnly { .. })
            )
        });
        any && all
    }
}

/// `base + within`, the address a piece starts at behind its page's target, or
/// [`GpuFault::OutOfRange`] for the piece at `va` if that is not representable (which
/// [`GpuVm::map`]'s target check rules out).
fn at(base: u64, within: u64, va: u64, len: usize) -> Result<u64, GpuFault> {
    base.checked_add(within)
        .ok_or_else(|| out_of_range(va, len))
}

/// [`GpuFault::OutOfRange`] for the `len` bytes at `va`.
fn out_of_range(va: u64, len: usize) -> GpuFault {
    GpuFault::OutOfRange {
        va,
        len: u64::try_from(len).unwrap_or(u64::MAX),
    }
}

/// The syncpoint whose shim word holds byte `within` of the shim page, if any: none from word
/// [`SYNCPOINT_COUNT`](crate::interface::SYNCPOINT_COUNT) on, which [`SyncpointId::new`]
/// refuses.
fn shim_syncpoint(within: u64) -> Option<SyncpointId> {
    u8::try_from(within / SHIM_STRIDE)
        .ok()
        .and_then(SyncpointId::new)
}

/// Copies the shim page's bytes from `within` into `out` (module docs, "The syncpoint shim"):
/// each counter is read once per word the access touches.
fn read_shim(syncpoints: &Syncpoints, within: u64, out: &mut [u8]) {
    let mut done = 0usize;
    while done < out.len() {
        // `within + done` stays in the page, which `out` was cut to.
        let at = within + done as u64;
        // Below `SHIM_STRIDE`.
        let in_word = usize::try_from(at % SHIM_STRIDE).unwrap_or(0);
        let step = (WORD_BYTES - in_word).min(out.len() - done);
        let word = shim_syncpoint(at).map_or(0, |id| syncpoints.read(id));
        out[done..done + step].copy_from_slice(&word.to_le_bytes()[in_word..in_word + step]);
        done += step;
    }
}

/// The syncpoint a write to the shim increments (module docs, "The syncpoint shim"): the piece
/// of `piece_len` bytes at `va`, in the shim, must be the whole write of `write_len` bytes and
/// start on a syncpoint's word; otherwise [`GpuFault::ShimWrite`] at `va`.
fn shim_release(va: u64, piece_len: usize, write_len: usize) -> Result<SyncpointId, GpuFault> {
    let within = va % PAGE_SIZE;
    let refused = GpuFault::ShimWrite { va };
    if piece_len != write_len || !within.is_multiple_of(SHIM_STRIDE) {
        return Err(refused);
    }
    shim_syncpoint(within).ok_or(refused)
}
