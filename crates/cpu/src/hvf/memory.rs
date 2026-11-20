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
//!
//! [`path`] decides between the two for every access, as a pure function of the kind of memory
//! and whether the access is whole words. Its test
//! (`the_path_of_an_access_is_fixed_for_every_kind_and_shape`) is what pins the atomic path: on
//! this host a byte-wise copy of one aligned word is most likely a single store as well, so no
//! test on a real vCPU can tell the two apart.
//!
//! # Lazy backing
//!
//! Every region is an anonymous private host mapping (`mmap` with `MAP_ANON`), which reads as
//! zeros and which the host commits a page at a time, the first time the page is touched, by
//! the guest through stage 2 or by the host. Making a region and mapping it into the VM
//! (`hv_vm_map`) commit nothing (`tests/hvf.rs`,
//! `a_vm_with_ram_of_a_whole_application_pool_costs_only_the_pages_touched`). So guest RAM as
//! large as a console's application pool, and a table pool as large as every table of the
//! address space, cost the host what is touched, not what is reserved
//! (`docs/specs/execution-model.md`, "Sparse and lazy memory").
//! [`PhysicalMemory::ram_resident_bytes`] and [`PhysicalMemory::table_pool_resident_bytes`]
//! count what each costs now, in whole host pages (`mincore`).

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

/// How an access to a region's bytes is carried out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Path {
    /// One atomic load or store per eight-byte word.
    AtomicWords,
    /// A byte-wise copy. A write is followed by instruction-cache maintenance on the written
    /// range when `sync_icache` is set; a read never needs any.
    Bytes {
        /// Whether a write must be followed by instruction-cache maintenance.
        sync_icache: bool,
    },
}

/// The path an access to memory of `kind` takes, given whether it is a whole number of aligned
/// words ([`is_whole_words`]). See the module docs for the two rules this encodes.
pub(super) const fn path(kind: Kind, whole_words: bool) -> Path {
    match (kind, whole_words) {
        (Kind::Tables, true) => Path::AtomicWords,
        // Tables are never executed, so they need no instruction-cache maintenance.
        (Kind::Tables, false) => Path::Bytes { sync_icache: false },
        (Kind::Ram | Kind::System, _) => Path::Bytes { sync_icache: true },
    }
}

/// Host memory with the alignment HVF needs, zeroed, backed lazily (see the module docs),
/// given back on drop.
pub(super) struct HostRegion {
    ptr: NonNull<u8>,
    len: usize,
    kind: Kind,
}

// SAFETY: HostRegion owns its allocation, and every access goes through raw pointers with bounds
// checks; no references into it are ever handed out, so sharing it between threads is sound as
// far as Rust is concerned. (Guest vCPUs read and write it concurrently, as real RAM would be.)
unsafe impl Send for HostRegion {}
// SAFETY: as above.
unsafe impl Sync for HostRegion {}

impl HostRegion {
    /// Reserves `len` bytes of zeroed memory, which must be a whole number of host pages, that
    /// the host commits a page at a time on first touch.
    pub(super) fn new(len: u64, kind: Kind) -> Result<Self, Error> {
        let too_big = |_| Error::Host(format!("cannot allocate {len:#x} bytes of guest memory"));
        let size = usize::try_from(len).map_err(too_big)?;
        if size == 0 {
            return Err(Error::Host("guest memory region is empty".into()));
        }
        if !len.is_multiple_of(HOST_PAGE) {
            return Err(Error::Host(format!(
                "guest memory of {len:#x} bytes is not a whole number of {HOST_PAGE:#x}-byte host \
                 pages"
            )));
        }
        // SAFETY: an anonymous private mapping at an address the host picks: it aliases no
        // memory Rust knows of. The result is checked before use.
        let raw = unsafe {
            ffi::mmap(
                std::ptr::null_mut(),
                size,
                ffi::PROT_READ | ffi::PROT_WRITE,
                ffi::MAP_PRIVATE | ffi::MAP_ANON,
                -1,
                0,
            )
        };
        if raw == ffi::MAP_FAILED {
            return Err(Error::Host(format!(
                "out of memory reserving {len:#x} bytes of guest memory"
            )));
        }
        let ptr = NonNull::new(raw.cast::<u8>())
            .ok_or_else(|| Error::Host(format!("out of memory reserving {len:#x} bytes")))?;
        let region = Self {
            ptr,
            len: size,
            kind,
        };
        // Stage-2 mappings need host-page alignment; the host's own pages on Apple Silicon are
        // 16 KiB, so `mmap` gives it. Checked rather than assumed, after `region` owns the
        // mapping, so it is given back on this path too.
        let aligned = u64::try_from(region.ptr.as_ptr().addr())
            .is_ok_and(|address| address.is_multiple_of(HOST_PAGE));
        if !aligned {
            return Err(Error::Host(format!(
                "the host placed guest memory at an address not aligned to {HOST_PAGE:#x}"
            )));
        }
        Ok(region)
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// How many bytes of the region the host holds in memory now, in whole host pages: what
    /// it costs the host. A page never touched is never counted. A page touched and since
    /// compressed or paged out by the host is not counted either, so this is at most what was
    /// touched, and equal to it until the host reclaims memory.
    pub(super) fn resident_bytes(&self) -> Result<u64, Error> {
        // SAFETY: `getpagesize` reads a constant of the host.
        let page = unsafe { ffi::getpagesize() };
        let page = usize::try_from(page)
            .ok()
            .filter(|&page| page != 0)
            .ok_or_else(|| Error::Host(format!("the host reports a page size of {page}")))?;
        let mut pages = vec![0; self.len.div_ceil(page)];
        // SAFETY: `ptr..ptr + len` is the live mapping this region owns, and `pages` has one
        // byte for each host page of it, as `mincore` writes.
        let status = unsafe {
            ffi::mincore(
                self.ptr.as_ptr().cast_const().cast(),
                self.len,
                pages.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(Error::Host(format!(
                "mincore failed on {:#x} bytes of guest memory",
                self.len
            )));
        }
        let resident = pages
            .iter()
            .filter(|&&state| state & ffi::MINCORE_INCORE != 0)
            .count();
        let bytes = resident.saturating_mul(page);
        Ok(u64::try_from(bytes).unwrap_or(u64::MAX))
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
        match path(self.kind, is_whole_words(start, buf.len())) {
            Path::AtomicWords => {
                let (words, _) = buf.as_chunks_mut::<WORD>();
                for (index, word) in words.iter_mut().enumerate() {
                    *word = self
                        .word(start + index * WORD)
                        .load(Ordering::Acquire)
                        .to_ne_bytes();
                }
            }
            Path::Bytes { .. } => {
                // SAFETY: `start..start + buf.len()` is inside the allocation (checked by
                // `range`), and `buf` is a separate Rust allocation, so the two don't overlap.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.ptr.as_ptr().add(start),
                        buf.as_mut_ptr(),
                        buf.len(),
                    );
                }
            }
        }
        Some(())
    }

    pub(super) fn write(&self, offset: u64, data: &[u8]) -> Option<()> {
        let start = self.range(offset, data.len())?;
        match path(self.kind, is_whole_words(start, data.len())) {
            Path::AtomicWords => {
                let (words, _) = data.as_chunks::<WORD>();
                for (index, word) in words.iter().enumerate() {
                    self.word(start + index * WORD)
                        .store(u64::from_ne_bytes(*word), Ordering::Release);
                }
            }
            Path::Bytes { sync_icache } => {
                // SAFETY: as in `read`, with the copy going the other way. The region is only
                // ever accessed through raw pointers, so writing through `&self` aliases no Rust
                // reference.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        self.ptr.as_ptr().add(start),
                        data.len(),
                    );
                }
                if sync_icache {
                    // The guest may execute what was just written, and Hypervisor.framework
                    // does not keep its instruction fetches coherent with host stores.
                    self.sync_icache(start, data.len());
                }
            }
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
        // SAFETY: `ptr..ptr + len` came from `mmap` and is given back only here. A failure
        // cannot be reported from a drop; it would leave the range reserved, never reused.
        unsafe { ffi::munmap(self.ptr.as_ptr().cast(), self.len) };
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
    /// How many bytes of guest RAM the host holds in memory now: the host pages the guest or
    /// the host has touched, less any the host has since compressed or paged out (see the
    /// module docs, "Lazy backing"). An [`Error::Host`] if the host cannot say.
    pub fn ram_resident_bytes(&self) -> Result<u64, Error> {
        self.vm.ram.region.resident_bytes()
    }

    /// As [`Self::ram_resident_bytes`], for the table pool.
    pub fn table_pool_resident_bytes(&self) -> Result<u64, Error> {
        self.vm.table_pool.region.resident_bytes()
    }

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
    fn the_path_of_an_access_is_fixed_for_every_kind_and_shape() {
        // Every kind of memory, whole words and not: only whole words of a table are atomic,
        // and only a table is left without instruction-cache maintenance.
        let cases = [
            (Kind::Ram, true, Path::Bytes { sync_icache: true }),
            (Kind::Ram, false, Path::Bytes { sync_icache: true }),
            (Kind::Tables, true, Path::AtomicWords),
            (Kind::Tables, false, Path::Bytes { sync_icache: false }),
            (Kind::System, true, Path::Bytes { sync_icache: true }),
            (Kind::System, false, Path::Bytes { sync_icache: true }),
        ];
        for (kind, whole_words, expected) in cases {
            assert_eq!(path(kind, whole_words), expected, "{kind:?} {whole_words}");
        }
        // What `Stage1Tables` does to the pool takes the atomic path: one descriptor, and a
        // whole table zeroed.
        for (start, len) in [(0x3ff8, WORD), (0x3000, 0x1000)] {
            assert_eq!(
                path(Kind::Tables, is_whole_words(start, len)),
                Path::AtomicWords,
                "{start:#x} {len:#x}"
            );
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

    #[test]
    fn a_region_that_is_not_whole_host_pages_is_refused() {
        for len in [HOST_PAGE - 1, HOST_PAGE + 1, 2 * HOST_PAGE - 4096] {
            assert!(
                matches!(
                    HostRegion::new(len, Kind::Ram),
                    Err(Error::Host(ref message)) if *message == format!(
                        "guest memory of {len:#x} bytes is not a whole number of \
                         {HOST_PAGE:#x}-byte host pages"
                    )
                ),
                "{len:#x}"
            );
        }
        let two = HostRegion::new(2 * HOST_PAGE, Kind::Ram).unwrap();
        assert_eq!(u64::try_from(two.len()).unwrap(), 2 * HOST_PAGE);
    }

    #[test]
    fn dropping_a_region_gives_its_address_space_back() {
        // 64 regions of 4 TiB, one after another: 256 TiB in all, more address space than a
        // process has, so this passes only if each drop gives its reservation back. Writing
        // the last byte of each makes sure the whole range was reserved; it commits one page.
        // (`mincore` on the dropped range cannot tell a leak from a release on macOS.)
        let len: u64 = 4 << 40;
        for n in 0..64 {
            let region = HostRegion::new(len, Kind::Ram)
                .map_err(|error| format!("region {n}: {error}"))
                .unwrap();
            region.write(len - 1, &[0x5a]).unwrap();
            let mut back = [0];
            region.read(len - 1, &mut back).unwrap();
            assert_eq!(back, [0x5a], "region {n}");
        }
    }

    /// The host's page size: the unit [`HostRegion::resident_bytes`] counts in.
    fn host_page() -> u64 {
        // SAFETY: `getpagesize` reads a constant of the host.
        u64::try_from(unsafe { ffi::getpagesize() }).unwrap()
    }

    #[test]
    fn a_region_costs_the_host_only_the_pages_written() {
        // 64 MiB: far more than a page, and nothing of it is committed when it is made.
        let len = 64 << 20;
        let host_page = host_page();
        for kind in [Kind::Ram, Kind::Tables, Kind::System] {
            let region = HostRegion::new(len, kind).unwrap();
            assert_eq!(region.resident_bytes(), Ok(0), "{kind:?}");
            // One byte in one page: that page.
            let middle = len / 2 + 5;
            region.write(middle, &[0x5a]).unwrap();
            assert_eq!(region.resident_bytes(), Ok(host_page), "{kind:?}");
            // Again in the same page: still one.
            region.write(middle + 8, &[0xa5; 8]).unwrap();
            assert_eq!(region.resident_bytes(), Ok(host_page), "{kind:?}");
            // The first and the last byte of the region: two pages more.
            region.write(0, &[1]).unwrap();
            region.write(len - 1, &[2]).unwrap();
            assert_eq!(region.resident_bytes(), Ok(3 * host_page), "{kind:?}");
            // What was written reads back, and the bytes around it read as zero.
            let mut back = [0xff; 10];
            region.read(middle - 1, &mut back).unwrap();
            assert_eq!(back, [0, 0x5a, 0, 0, 0, 0, 0, 0, 0, 0xa5], "{kind:?}");
        }
    }
}
