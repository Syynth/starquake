//! The host memory behind the VM's guest-physical address space, and [`PhysicalMemory`], the
//! view of it that [`starquake_memory::stage1::Stage1Tables`] and the loader write through.
//!
//! Two rules live here, one per kind of memory:
//!
//! - **RAM** may be executed by the guest, and Hypervisor.framework does not keep the guest's
//!   instruction fetches coherent with host stores, so every write is followed by
//!   `sys_icache_invalidate` on the written range (`tools/hvprobe/README.md`, "Instruction cache
//!   and host writes").
//! - **The table pool** is walked by the MMU of every running vCPU while the host changes it. A
//!   write (or read) that is a whole number of eight-byte words at an eight-byte-aligned address
//!   is done one word at a time with an atomic store (or load), so a walker sees each
//!   descriptor either before or after a change, never half of one. `Stage1Tables` writes a
//!   descriptor as exactly one such word and zeroes a table as 512 of them
//!   (`starquake_memory::stage1`, module docs). Anything else is copied byte-wise; nothing in
//!   the workspace writes the pool that way.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use starquake_memory::{Fault, GuestMemory};

use super::{Vm, ffi};
use crate::Error;

/// Stage-2 (IPA → host) mappings must be aligned to the 16 KiB host page
/// (`tools/hvprobe/README.md`, "4 KiB stage-1 pages on a 16 KiB host").
pub(super) const HOST_PAGE: u64 = 0x4000;

/// Bytes in the words the table pool is read and written in: one translation table descriptor.
const WORD: usize = size_of::<u64>();

/// Whether an access of `len` bytes at byte offset `start` is a whole number of aligned words.
pub(super) const fn is_whole_words(start: usize, len: usize) -> bool {
    start.is_multiple_of(WORD) && len.is_multiple_of(WORD)
}

/// How a region's bytes are accessed (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    /// Memory the guest may execute: writes are followed by instruction-cache maintenance.
    Ram,
    /// Translation tables: aligned whole-word accesses are atomic per word.
    Tables,
    /// The backend's own pages, written once before the VM exists.
    System,
}

/// Host memory with the alignment HVF needs, zeroed, freed on drop.
pub(super) struct HostRegion {
    ptr: NonNull<u8>,
    layout: Layout,
    kind: Kind,
}

// SAFETY: HostRegion owns its allocation, and every access goes through raw pointers with bounds
// checks; no references into it are ever handed out, so sharing it between threads is sound as
// far as Rust is concerned. (Guest vCPUs read and write it concurrently, as real RAM would be.)
unsafe impl Send for HostRegion {}
// SAFETY: as above.
unsafe impl Sync for HostRegion {}

impl HostRegion {
    pub(super) fn new(len: u64, kind: Kind) -> Result<Self, Error> {
        let too_big = |_| Error::Host(format!("cannot allocate {len:#x} bytes of guest memory"));
        let size = usize::try_from(len).map_err(too_big)?;
        let align = usize::try_from(HOST_PAGE).map_err(too_big)?;
        let layout = Layout::from_size_align(size, align)
            .map_err(|_| Error::Host(format!("bad guest memory layout: {len:#x} bytes")))?;
        if size == 0 {
            return Err(Error::Host("guest memory region is empty".into()));
        }
        // SAFETY: the layout has a non-zero size (checked above).
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| Error::Host(format!("out of memory allocating {len:#x} bytes")))?;
        Ok(Self { ptr, layout, kind })
    }

    pub(super) fn len(&self) -> usize {
        self.layout.size()
    }

    /// The host address of byte `offset`, for mapping part of the region into the VM; `None`
    /// unless `offset..offset + len` is inside the region.
    pub(super) fn host_address(&self, offset: u64, len: usize) -> Option<*mut u8> {
        let start = self.range(offset, len)?;
        // SAFETY: `start` is inside the allocation (checked by `range`).
        Some(unsafe { self.ptr.as_ptr().add(start) })
    }

    /// The byte range `offset..offset + len`, if it is inside the region.
    fn range(&self, offset: u64, len: usize) -> Option<usize> {
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(len)?;
        (end <= self.len()).then_some(start)
    }

    pub(super) fn read(&self, offset: u64, buf: &mut [u8]) -> Option<()> {
        let start = self.range(offset, buf.len())?;
        if self.kind == Kind::Tables && is_whole_words(start, buf.len()) {
            let (words, _) = buf.as_chunks_mut::<WORD>();
            for (index, word) in words.iter_mut().enumerate() {
                *word = self
                    .word(start + index * WORD)
                    .load(Ordering::Acquire)
                    .to_ne_bytes();
            }
            return Some(());
        }
        // SAFETY: `start..start + buf.len()` is inside the allocation (checked by `range`), and
        // `buf` is a separate Rust allocation, so the two don't overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.as_ptr().add(start),
                buf.as_mut_ptr(),
                buf.len(),
            );
        }
        Some(())
    }

    pub(super) fn write(&self, offset: u64, data: &[u8]) -> Option<()> {
        let start = self.range(offset, data.len())?;
        if self.kind == Kind::Tables && is_whole_words(start, data.len()) {
            let (words, _) = data.as_chunks::<WORD>();
            for (index, word) in words.iter().enumerate() {
                self.word(start + index * WORD)
                    .store(u64::from_ne_bytes(*word), Ordering::Release);
            }
            return Some(());
        }
        // SAFETY: as in `read`, with the copy going the other way. The region is only ever
        // accessed through raw pointers, so writing through `&self` aliases no Rust reference.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr().add(start), data.len());
        }
        if self.kind != Kind::Tables {
            // The guest may execute what was just written, and Hypervisor.framework does not
            // keep its instruction fetches coherent with host stores.
            self.sync_icache(start, data.len());
        }
        Some(())
    }

    /// The word at byte offset `start`, which must be word aligned with `start + WORD` inside
    /// the region (callers check both with `range` and `is_whole_words`).
    fn word(&self, start: usize) -> &AtomicU64 {
        debug_assert!(start.is_multiple_of(WORD) && start + WORD <= self.len());
        // SAFETY: the word is inside the allocation, which outlives `&self`. The allocation is
        // aligned to the 16 KiB host page and `start` is a multiple of eight, so the pointer is
        // aligned for `AtomicU64`. Every Rust access to a table region's whole words goes
        // through this function, so they are all atomic; the guest's own reads of the same
        // memory are the hardware's, as with any memory shared with another machine.
        unsafe { AtomicU64::from_ptr(self.ptr.as_ptr().add(start).cast()) }
    }

    /// Makes guest instruction fetches see everything written to `start..start + len` so far.
    /// `start..start + len` must be inside the region (callers check it with `range`).
    fn sync_icache(&self, start: usize, len: usize) {
        debug_assert!(start.checked_add(len).is_some_and(|end| end <= self.len()));
        // SAFETY: the range is inside the allocation, which stays live for the call, and the
        // function only does cache maintenance by address: it neither reads nor writes through
        // the pointer in a way Rust can observe.
        unsafe { ffi::sys_icache_invalidate(self.ptr.as_ptr().add(start).cast(), len) };
    }
}

impl Drop for HostRegion {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc_zeroed` with this same layout and is freed only here.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// A host region placed at a guest-physical address.
pub(super) struct Extent {
    pub(super) base: u64,
    pub(super) region: HostRegion,
}

impl Extent {
    /// One past the last guest-physical address of the extent.
    pub(super) fn end(&self) -> u64 {
        // The backend only places extents whose end it has already computed without overflow.
        self.base.saturating_add(self.region.len() as u64)
    }

    /// Whether `pa` is inside the extent, or exactly at its end (where only an access of no
    /// bytes fits).
    fn holds(&self, pa: u64) -> bool {
        (self.base..=self.end()).contains(&pa)
    }
}

/// The VM's guest-physical memory: guest RAM and the translation table pool, by guest-physical
/// address. Readable and writable from any thread while vCPUs run.
///
/// This is the memory [`starquake_memory::stage1::Stage1Tables`] is built over (its pool inside
/// [`HvfBackend::table_pool`](super::HvfBackend::table_pool)), and the memory the pages it maps
/// are backed by ([`HvfBackend::ram`](super::HvfBackend::ram)). Guest code never sees these
/// addresses; it sees what the tables of its address space map.
///
/// An access must lie inside one of the two ranges. One that does not is a [`Fault`] at its
/// first address outside them, and reads or writes nothing.
///
/// Cloning is cheap: every clone is the same memory, and keeps the VM alive.
#[derive(Clone)]
pub struct PhysicalMemory {
    pub(super) vm: Arc<Vm>,
}

impl PhysicalMemory {
    /// The extent an access of `len` bytes at `pa` falls in and its offset there, or the fault.
    fn locate(&self, pa: u64, len: usize) -> Result<(&Extent, u64), Fault> {
        let extent = [&self.vm.ram, &self.vm.table_pool]
            .into_iter()
            .find(|extent| extent.holds(pa))
            .ok_or(Fault { address: pa })?;
        let fits = u64::try_from(len)
            .ok()
            .and_then(|len| pa.checked_add(len))
            .is_some_and(|end| end <= extent.end());
        if fits {
            Ok((extent, pa - extent.base))
        } else {
            Err(Fault {
                address: extent.end(),
            })
        }
    }
}

impl GuestMemory for PhysicalMemory {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        let (extent, offset) = self.locate(va, buf.len())?;
        extent.region.read(offset, buf).ok_or(Fault { address: va })
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        let (extent, offset) = self.locate(va, data.len())?;
        extent
            .region
            .write(offset, data)
            .ok_or(Fault { address: va })
    }
}

impl std::fmt::Debug for PhysicalMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicalMemory")
            .field("ram", &(self.vm.ram.base..self.vm.ram.end()))
            .field(
                "table_pool",
                &(self.vm.table_pool.base..self.vm.table_pool.end()),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_word_accesses_are_recognised_exactly() {
        // A descriptor: one word at a word-aligned offset, for every slot of a table.
        for slot in 0..512 {
            assert!(is_whole_words(slot * WORD, WORD), "slot {slot}");
        }
        // Zeroing a table: 512 words.
        assert!(is_whole_words(0x3000, 0x1000));
        assert!(is_whole_words(0, 0));
        // One byte off in either the address or the length is not.
        for (start, len) in [(1, WORD), (WORD - 1, WORD), (0, WORD - 1), (0, WORD + 1)] {
            assert!(!is_whole_words(start, len), "{start} {len}");
        }
    }

    #[test]
    fn every_kind_of_region_reads_back_what_was_written() {
        for kind in [Kind::Ram, Kind::Tables, Kind::System] {
            let region = HostRegion::new(HOST_PAGE, kind).unwrap();
            // Word-shaped, and not: a write that starts and ends inside words.
            let words: Vec<u8> = (1..=32).collect();
            region.write(0x100, &words).unwrap();
            region.write(0x203, &[0xaa, 0xbb, 0xcc]).unwrap();
            let mut back = [0; 32];
            region.read(0x100, &mut back).unwrap();
            assert_eq!(back.as_slice(), words, "{kind:?}");
            // The bytes around the unaligned write are untouched, read through both paths.
            let mut around = [0xff; 16];
            region.read(0x200, &mut around).unwrap();
            assert_eq!(
                around,
                [0, 0, 0, 0xaa, 0xbb, 0xcc, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "{kind:?}"
            );
            let mut unaligned = [0xff; 5];
            region.read(0x101, &mut unaligned).unwrap();
            assert_eq!(unaligned, [2, 3, 4, 5, 6], "{kind:?}");
        }
    }

    #[test]
    fn accesses_are_checked_over_their_whole_extent() {
        let region = HostRegion::new(HOST_PAGE, Kind::Tables).unwrap();
        let len = region.len();
        let last_word = u64::try_from(len - WORD).unwrap();
        assert_eq!(region.write(last_word, &[1; WORD]), Some(()));
        // One byte past the end, an offset past the end, and an offset that overflows.
        assert_eq!(region.write(last_word, &[1; WORD + 1]), None);
        assert_eq!(region.write(last_word + 8, &[1; WORD]), None);
        assert_eq!(region.read(u64::MAX, &mut [0; WORD]), None);
        // An empty access at the very end is inside; one past it is not.
        assert_eq!(region.read(HOST_PAGE, &mut []), Some(()));
        assert_eq!(region.read(HOST_PAGE + 1, &mut []), None);
        assert!(region.host_address(0, len).is_some());
        assert!(region.host_address(1, len).is_none());
        assert!(region.host_address(u64::MAX, 1).is_none());
    }

    #[test]
    fn an_empty_region_is_refused() {
        assert!(matches!(
            HostRegion::new(0, Kind::Ram),
            Err(Error::Host(message)) if message == "guest memory region is empty"
        ));
    }
}
