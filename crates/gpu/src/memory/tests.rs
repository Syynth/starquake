use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use starquake_memory::{Fault, FlatMemory, GuestMemory, MapError, PAGE_SIZE, PageRange};

use super::*;

/// The CPU-side guest memory every test maps onto: 64 pages from this address.
const CPU_BASE: u64 = 0x8000_0000;
const CPU_PAGES: u64 = 64;

fn pages(first: u64, count: u64) -> PageRange {
    PageRange::new(first * PAGE_SIZE, count * PAGE_SIZE).unwrap()
}

fn memory() -> TestGpuMemory<FlatMemory> {
    let size = usize::try_from(CPU_PAGES * PAGE_SIZE).unwrap();
    TestGpuMemory::new(FlatMemory::new(CPU_BASE, size))
}

/// The CPU address of CPU page `n`.
fn cpu_page(n: u64) -> u64 {
    CPU_BASE + n * PAGE_SIZE
}

/// Fills the guest memory so every byte is distinct from its neighbours: the byte at CPU offset
/// `o` is `o * 7 + o / 256` (wrapping), which differs across page boundaries too.
fn fill(mem: &TestGpuMemory<FlatMemory>) {
    let bytes: Vec<u8> = (0..CPU_PAGES * PAGE_SIZE)
        .map(|o| (o.wrapping_mul(7).wrapping_add(o / 256) & 0xff) as u8)
        .collect();
    mem.guest().write(CPU_BASE, &bytes).unwrap();
}

/// The bytes at CPU address `cpu`, read straight from guest memory.
fn cpu_bytes(mem: &TestGpuMemory<FlatMemory>, cpu: u64, len: usize) -> Vec<u8> {
    let mut out = vec![0; len];
    mem.guest().read(cpu, &mut out).unwrap();
    out
}

fn gpu_bytes(mem: &TestGpuMemory<FlatMemory>, va: u64, len: usize) -> Result<Vec<u8>, GpuFault> {
    let mut out = vec![0; len];
    mem.read(va, &mut out).map(|()| out)
}

// --- page_pieces ---

#[test]
fn pieces_split_at_every_page_boundary() {
    let pieces: Vec<_> = page_pieces(PAGE_SIZE - 2, usize::try_from(PAGE_SIZE).unwrap() + 4)
        .unwrap()
        .collect();
    let page = usize::try_from(PAGE_SIZE).unwrap();
    assert_eq!(
        pieces,
        [
            (PAGE_SIZE - 2, 0..2),
            (PAGE_SIZE, 2..2 + page),
            (2 * PAGE_SIZE, 2 + page..4 + page),
        ]
    );
    // An access that ends exactly at a page boundary has no empty last piece.
    let pieces: Vec<_> = page_pieces(PAGE_SIZE - 2, 2).unwrap().collect();
    assert_eq!(pieces, [(PAGE_SIZE - 2, 0..2)]);
    assert_eq!(page_pieces(5, 0).unwrap().count(), 0);
}

#[test]
fn pieces_refuse_an_end_past_the_address_space() {
    // Ends exactly at u64::MAX: representable.
    let pieces: Vec<_> = page_pieces(u64::MAX - 8, 8).unwrap().collect();
    assert_eq!(pieces, [(u64::MAX - 8, 0..8)]);
    // One byte more: the end is 2^64.
    assert!(matches!(
        page_pieces(u64::MAX - 7, 8),
        Err(GpuFault::OutOfRange {
            va: 0xffff_ffff_ffff_fff8,
            len: 8
        })
    ));
    // A zero-length access is fine anywhere.
    assert_eq!(page_pieces(u64::MAX, 0).unwrap().count(), 0);
}

// --- TestGpuMemory ---

#[test]
fn reads_and_writes_cross_pages_and_mappings() {
    let mem = memory();
    fill(&mem);
    // Two adjacent GPU mappings whose CPU backing is not adjacent, in reverse order.
    mem.map(pages(0x100, 2), cpu_page(10)).unwrap(); // GPU 0x100..0x102 -> CPU 10..12
    mem.map(pages(0x102, 1), cpu_page(3)).unwrap(); // GPU 0x102 -> CPU 3

    // Across the page boundary inside the first mapping.
    let va = 0x101 * PAGE_SIZE - 3;
    let got = gpu_bytes(&mem, va, 6).unwrap();
    assert_eq!(got, cpu_bytes(&mem, cpu_page(11) - 3, 6));

    // Across the mapping boundary: the last 3 bytes of one, the first 3 of the other.
    let va = 0x102 * PAGE_SIZE - 3;
    let got = gpu_bytes(&mem, va, 6).unwrap();
    let mut expected = cpu_bytes(&mem, cpu_page(12) - 3, 3);
    expected.extend(cpu_bytes(&mem, cpu_page(3), 3));
    assert_eq!(got, expected);

    // A write across the mapping boundary lands in both backings.
    mem.write(va, &[1, 2, 3, 4, 5, 6]).unwrap();
    assert_eq!(cpu_bytes(&mem, cpu_page(12) - 3, 3), [1, 2, 3]);
    assert_eq!(cpu_bytes(&mem, cpu_page(3), 3), [4, 5, 6]);
    // The neighbours on both sides are unchanged.
    let fresh = memory();
    fill(&fresh);
    assert_eq!(
        cpu_bytes(&mem, cpu_page(12) - 4, 1),
        cpu_bytes(&fresh, cpu_page(12) - 4, 1)
    );
    assert_eq!(
        cpu_bytes(&mem, cpu_page(3) + 3, 1),
        cpu_bytes(&fresh, cpu_page(3) + 3, 1)
    );
}

#[test]
fn a_read_into_a_gap_faults_at_the_first_unmapped_byte() {
    let mem = memory();
    fill(&mem);
    mem.map(pages(0x100, 2), cpu_page(0)).unwrap(); // GPU 0x100..0x102
    let end = 0x102 * PAGE_SIZE;
    // Ends exactly at the mapping's end: fine.
    assert_eq!(
        gpu_bytes(&mem, end - 4, 4).unwrap(),
        cpu_bytes(&mem, cpu_page(2) - 4, 4)
    );
    // One byte more: the first unmapped byte is the end.
    assert_eq!(
        gpu_bytes(&mem, end - 4, 5),
        Err(GpuFault::Unmapped { va: end })
    );
    // Starting exactly at the end.
    assert_eq!(gpu_bytes(&mem, end, 1), Err(GpuFault::Unmapped { va: end }));
    // Starting one byte below the start.
    let start = 0x100 * PAGE_SIZE;
    assert_eq!(
        gpu_bytes(&mem, start - 1, 2),
        Err(GpuFault::Unmapped { va: start - 1 })
    );
    assert_eq!(
        gpu_bytes(&mem, start, 1).unwrap(),
        cpu_bytes(&mem, cpu_page(0), 1)
    );
}

#[test]
fn a_write_that_reaches_a_gap_changes_nothing() {
    let mem = memory();
    mem.map(pages(0x100, 1), cpu_page(0)).unwrap();
    let end = 0x101 * PAGE_SIZE;
    assert_eq!(
        mem.write(end - 2, &[9, 9, 9]),
        Err(GpuFault::Unmapped { va: end })
    );
    assert_eq!(cpu_bytes(&mem, cpu_page(1) - 2, 2), [0, 0]);
    // The same write, one byte shorter, fits exactly.
    mem.write(end - 2, &[9, 9]).unwrap();
    assert_eq!(cpu_bytes(&mem, cpu_page(1) - 2, 2), [9, 9]);
}

#[test]
fn a_guest_fault_names_both_addresses() {
    let mem = memory();
    // The GPU range is mapped, but its second page leads past the end of the guest memory.
    mem.map(pages(0x100, 2), cpu_page(CPU_PAGES - 1)).unwrap();
    let second = 0x101 * PAGE_SIZE;
    let fault = GpuFault::Guest {
        va: second,
        fault: Fault {
            address: cpu_page(CPU_PAGES),
        },
    };
    assert_eq!(gpu_bytes(&mem, second - 1, 2), Err(fault));
    // A write keeps the piece written before the guest fault.
    assert_eq!(mem.write(second - 1, &[7, 7]), Err(fault));
    assert_eq!(cpu_bytes(&mem, cpu_page(CPU_PAGES) - 1, 1), [7]);
}

#[test]
fn an_access_past_the_address_space_is_out_of_range() {
    let mem = memory();
    assert_eq!(
        gpu_bytes(&mem, u64::MAX - 7, 8),
        Err(GpuFault::OutOfRange {
            va: u64::MAX - 7,
            len: 8
        })
    );
    assert_eq!(
        mem.write(u64::MAX - 7, &[0; 8]),
        Err(GpuFault::OutOfRange {
            va: u64::MAX - 7,
            len: 8
        })
    );
    // Ending exactly at u64::MAX gets as far as translation.
    assert_eq!(
        gpu_bytes(&mem, u64::MAX - 8, 8),
        Err(GpuFault::Unmapped { va: u64::MAX - 8 })
    );
    // Zero bytes anywhere succeed without translating.
    assert_eq!(gpu_bytes(&mem, u64::MAX, 0), Ok(Vec::new()));
    assert_eq!(mem.write(0x1234, &[]), Ok(()));
}

#[test]
fn map_and_unmap_check_overlap_at_the_edges_and_move_the_version() {
    let mem = memory();
    assert_eq!(GpuMemory::generation(&mem), 0);
    mem.map(pages(10, 2), cpu_page(0)).unwrap(); // 10..12
    assert_eq!(GpuMemory::generation(&mem), 1);
    // Neighbours that touch either end map; the version moves for each.
    mem.map(pages(12, 1), cpu_page(4)).unwrap();
    mem.map(pages(9, 1), cpu_page(5)).unwrap();
    assert_eq!(Translate::generation(&mem), 3);
    // Overlapping the first page from below, the last page, or inside: refused, version kept.
    for (range, va) in [
        (pages(8, 3), 9 * PAGE_SIZE),
        (pages(11, 1), 11 * PAGE_SIZE),
        (pages(11, 4), 11 * PAGE_SIZE),
    ] {
        assert_eq!(
            mem.map(range, cpu_page(6)),
            Err(MapError::AlreadyMapped { va }),
            "{range:?}"
        );
    }
    assert_eq!(GpuMemory::generation(&mem), 3);
    // CPU side checks.
    assert_eq!(
        mem.map(pages(20, 1), CPU_BASE + 1),
        Err(MapError::MisalignedAddress {
            address: CPU_BASE + 1
        })
    );
    let top = u64::MAX - (PAGE_SIZE - 1);
    assert_eq!(
        mem.map(pages(20, 1), top),
        Err(MapError::VirtualRange {
            va: top,
            size: PAGE_SIZE
        })
    );
    // Unmap takes exactly a mapping.
    assert_eq!(
        mem.unmap(pages(10, 1)),
        Err(MapError::NotMapped { va: 10 * PAGE_SIZE })
    );
    assert_eq!(
        mem.unmap(pages(11, 1)),
        Err(MapError::NotMapped { va: 11 * PAGE_SIZE })
    );
    assert_eq!(GpuMemory::generation(&mem), 3);
    mem.unmap(pages(10, 2)).unwrap();
    assert_eq!(GpuMemory::generation(&mem), 4);
    // The neighbours stay mapped.
    assert_eq!(mem.translate_page(9 * PAGE_SIZE), Some(cpu_page(5)));
    assert_eq!(mem.translate_page(12 * PAGE_SIZE), Some(cpu_page(4)));
    assert_eq!(mem.translate_page(10 * PAGE_SIZE), None);
}

#[test]
fn faults_are_plain_words() {
    assert_eq!(
        GpuFault::Unmapped { va: 0x1000 }.to_string(),
        "GPU address 0x1000 is not mapped"
    );
    assert_eq!(
        GpuFault::OutOfRange {
            va: 0x10,
            len: 0x20
        }
        .to_string(),
        "a GPU access of 0x20 bytes at 0x10 runs past the end of the address space"
    );
    assert_eq!(
        GpuFault::Guest {
            va: 0x2000,
            fault: Fault { address: 0x3000 }
        }
        .to_string(),
        "GPU address 0x2000: guest memory fault at 0x3000"
    );
}

// --- TranslationCache ---

/// A table that counts its translations, whose answers and version a test sets directly.
#[derive(Default)]
struct Scripted {
    generation: AtomicU64,
    /// Every page leads to `page + offset`; `u64::MAX` means unmapped.
    offset: AtomicU64,
    lookups: AtomicUsize,
    /// When set, a lookup moves the version as it runs (a remap racing the lookup).
    bump_during_lookup: std::sync::atomic::AtomicBool,
}

impl Translate for Scripted {
    type Target = u64;

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn translate_page(&self, page: u64) -> Option<u64> {
        self.lookups.fetch_add(1, Ordering::Relaxed);
        let offset = self.offset.load(Ordering::Relaxed);
        if self.bump_during_lookup.swap(false, Ordering::Relaxed) {
            self.generation.fetch_add(1, Ordering::Release);
        }
        (offset != u64::MAX).then_some(page + offset)
    }
}

impl Scripted {
    fn lookups(&self) -> usize {
        self.lookups.load(Ordering::Relaxed)
    }
}

#[test]
fn a_cached_page_is_answered_without_the_table() {
    let table = Arc::new(Scripted::default());
    table.offset.store(0x10_0000, Ordering::Relaxed);
    let mut cache = TranslationCache::new(Arc::clone(&table));
    assert!(Arc::ptr_eq(cache.source(), &table));
    assert_eq!(cache.translate(PAGE_SIZE + 5), Some(PAGE_SIZE + 0x10_0000));
    assert_eq!(table.lookups(), 1);
    // Any address in the same page hits, at both ends of the page.
    assert_eq!(cache.translate(PAGE_SIZE), Some(PAGE_SIZE + 0x10_0000));
    assert_eq!(
        cache.translate(2 * PAGE_SIZE - 1),
        Some(PAGE_SIZE + 0x10_0000)
    );
    assert_eq!(table.lookups(), 1);
    // The next page is another entry.
    assert_eq!(
        cache.translate(2 * PAGE_SIZE),
        Some(2 * PAGE_SIZE + 0x10_0000)
    );
    assert_eq!(table.lookups(), 2);
}

#[test]
fn a_moved_version_empties_the_cache_and_an_unmoved_one_keeps_it() {
    let table = Arc::new(Scripted::default());
    table.offset.store(0x1000, Ordering::Relaxed);
    let mut cache = TranslationCache::new(Arc::clone(&table));
    assert_eq!(cache.translate(0), Some(0x1000));

    // The table's answer changes but its version does not: the cache keeps the old answer.
    table.offset.store(0x5000, Ordering::Relaxed);
    assert_eq!(cache.translate(0), Some(0x1000));
    assert_eq!(table.lookups(), 1);

    // The version moves: the next translation asks again.
    table.generation.fetch_add(1, Ordering::Release);
    assert_eq!(cache.translate(0), Some(0x5000));
    assert_eq!(table.lookups(), 2);
    assert_eq!(cache.translate(0), Some(0x5000));
    assert_eq!(table.lookups(), 2);
}

#[test]
fn the_version_is_read_before_the_lookup() {
    let table = Arc::new(Scripted::default());
    table.offset.store(0x1000, Ordering::Relaxed);
    let mut cache = TranslationCache::new(Arc::clone(&table));
    // A remap lands while the table answers: the answer is from before it, so the cache must not
    // keep it under the new version.
    table.bump_during_lookup.store(true, Ordering::Relaxed);
    assert_eq!(cache.translate(0), Some(0x1000));
    table.offset.store(0x7000, Ordering::Relaxed);
    assert_eq!(cache.translate(0), Some(0x7000));
    assert_eq!(table.lookups(), 2);
}

#[test]
fn an_unmapped_page_is_not_cached() {
    let table = Arc::new(Scripted::default());
    table.offset.store(u64::MAX, Ordering::Relaxed);
    let mut cache = TranslationCache::new(Arc::clone(&table));
    assert_eq!(cache.translate(0), None);
    assert_eq!(cache.translate(0), None);
    assert_eq!(table.lookups(), 2);
    // Mapped later without a version move (as a map that the table published but whose version
    // this thread has not seen yet): seen at once, because the miss was not kept.
    table.offset.store(0x2000, Ordering::Relaxed);
    assert_eq!(cache.translate(0), Some(0x2000));
}

#[test]
fn slots_are_replaced_in_turn() {
    let table = Arc::new(Scripted::default());
    let mut cache = TranslationCache::new(Arc::clone(&table));
    let slots = u64::try_from(TRANSLATION_CACHE_SLOTS).unwrap();
    for page in 0..slots {
        cache.translate(page * PAGE_SIZE);
    }
    assert_eq!(table.lookups(), TRANSLATION_CACHE_SLOTS);
    for page in 0..slots {
        cache.translate(page * PAGE_SIZE);
    }
    assert_eq!(table.lookups(), TRANSLATION_CACHE_SLOTS);
    // One more page takes the first slot: page 0 is gone, pages 1.. are kept.
    cache.translate(slots * PAGE_SIZE);
    assert_eq!(table.lookups(), TRANSLATION_CACHE_SLOTS + 1);
    for page in 1..=slots {
        cache.translate(page * PAGE_SIZE);
    }
    assert_eq!(table.lookups(), TRANSLATION_CACHE_SLOTS + 1);
    cache.translate(0);
    assert_eq!(table.lookups(), TRANSLATION_CACHE_SLOTS + 2);
    // Page 0 went into the second slot, replacing page 1.
    cache.translate(PAGE_SIZE);
    assert_eq!(table.lookups(), TRANSLATION_CACHE_SLOTS + 3);
}

#[test]
fn a_cache_over_test_memory_follows_remaps() {
    let mem = Arc::new(memory());
    mem.map(pages(0x100, 1), cpu_page(1)).unwrap();
    let mut cache = TranslationCache::new(Arc::clone(&mem));
    assert_eq!(cache.translate(0x100 * PAGE_SIZE + 8), Some(cpu_page(1)));
    mem.unmap(pages(0x100, 1)).unwrap();
    assert_eq!(cache.translate(0x100 * PAGE_SIZE), None);
    mem.map(pages(0x100, 1), cpu_page(7)).unwrap();
    assert_eq!(cache.translate(0x100 * PAGE_SIZE), Some(cpu_page(7)));
}

#[test]
fn per_thread_caches_see_every_remap_once_the_version_moves() {
    const ROUNDS: u64 = 500;
    let mem = Arc::new(memory());
    let page = pages(0x100, 1);
    mem.map(page, cpu_page(0)).unwrap();
    let readers: Vec<_> = (0..2)
        .map(|_| {
            let mem = Arc::clone(&mem);
            std::thread::spawn(move || {
                let mut cache = TranslationCache::new(Arc::clone(&mem));
                let mut seen = 0u64;
                // Bounded by ROUNDS reads; never waits.
                for _ in 0..ROUNDS {
                    if let Some(cpu) = cache.translate(page.start()) {
                        // Only targets the remapper ever used.
                        assert!(cpu < cpu_page(ROUNDS.min(CPU_PAGES)), "{cpu:#x}");
                        seen += 1;
                    }
                }
                (cache, seen)
            })
        })
        .collect();
    let last = ROUNDS.min(CPU_PAGES) - 1;
    for round in 1..=last {
        mem.unmap(page).unwrap();
        mem.map(page, cpu_page(round)).unwrap();
    }
    for reader in readers {
        let (mut cache, _) = reader.join().unwrap();
        // After the remapper finished, each thread's cache answers with the final mapping.
        assert_eq!(cache.translate(page.start()), Some(cpu_page(last)));
    }
}
