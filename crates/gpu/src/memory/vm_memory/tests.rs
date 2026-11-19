//! Tests of the production memory over a real `GpuVm`, flat guest memory behind a call counter,
//! and real `Syncpoints` whose completions are recorded.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use starquake_memory::{Fault, FlatMemory, GuestMemory, PAGE_SIZE, PageRange};

use super::*;
use crate::interface::{Completions, GpuCompletion, Kind, RegionId, SYNCPOINT_COUNT, VmUpdate};
use crate::vm::SHIM_PAGES;

// ---------------------------------------------------------------------------------------------
// The rig
// ---------------------------------------------------------------------------------------------

/// The guest memory's CPU base and size in pages.
const CPU_BASE: u64 = 0x8000_0000;
const CPU_PAGES: u64 = 16;
/// The GPU addresses below, above 32 bits so a truncated address misses them.
const GPU_BASE: u64 = 0x1_2345_0000_0000;
/// Two pages leading to CPU page 3, not to the GPU address's own CPU address.
const GUEST: u64 = GPU_BASE;
const GUEST_PAGES: u64 = 2;
const GUEST_CPU_PAGE: u64 = 3;
/// Three pages of sparse region [`REGION`], from offset [`REGION_OFFSET`], starting exactly where
/// [`GUEST`] ends.
const ONLY: u64 = GUEST + GUEST_PAGES * PAGE_SIZE;
const ONLY_PAGES: u64 = 3;
const REGION: RegionId = RegionId(0x51);
const REGION_OFFSET: u64 = 5 * PAGE_SIZE;
/// One page leading to CPU page 10, starting exactly where [`ONLY`] ends.
const AFTER_ONLY: u64 = ONLY + ONLY_PAGES * PAGE_SIZE;
const AFTER_ONLY_CPU_PAGE: u64 = 10;
/// One page leading to CPU page 12, then the shim, then one page leading to CPU page 13.
const BEFORE_SHIM: u64 = GPU_BASE + 0x10 * PAGE_SIZE;
const SHIM: u64 = BEFORE_SHIM + PAGE_SIZE;
const AFTER_SHIM: u64 = SHIM + SHIM_PAGES * PAGE_SIZE;
/// Nothing maps this page or the one after it.
const HOLE: u64 = GPU_BASE + 0x20 * PAGE_SIZE;
/// The last syncpoint.
const LAST_ID: u8 = {
    let last = SYNCPOINT_COUNT - 1;
    assert!(last <= u8::MAX as u32);
    last.to_le_bytes()[0]
};
/// The first shim byte past the last syncpoint's word.
const PAST_LAST_WORD: u64 = (LAST_ID as u64 + 1) * SHIM_STRIDE;

fn cpu_page(n: u64) -> u64 {
    CPU_BASE + n * PAGE_SIZE
}

fn range(start: u64, pages: u64) -> PageRange {
    PageRange::new(start, pages * PAGE_SIZE).unwrap()
}

fn id(value: u8) -> SyncpointId {
    SyncpointId::new(value).unwrap()
}

/// Flat guest memory that counts the calls made to it.
struct Counting {
    flat: FlatMemory,
    calls: AtomicUsize,
}

impl GuestMemory for Counting {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.flat.read(va, buf)
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.flat.write(va, data)
    }
}

/// A [`Completions`] that keeps what was pushed, in order.
#[derive(Default)]
struct Recorder(Mutex<Vec<GpuCompletion>>);

impl Recorder {
    fn taken(&self) -> Vec<GpuCompletion> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl Completions for Recorder {
    fn push(&self, completion: GpuCompletion) {
        self.0.lock().unwrap().push(completion);
    }
}

struct Rig {
    vm: Arc<GpuVm>,
    guest: Arc<Counting>,
    syncpoints: Arc<Syncpoints>,
    pushed: Arc<Recorder>,
    mem: VmMemory<Counting>,
}

impl Rig {
    fn new() -> Self {
        let size = usize::try_from(CPU_PAGES * PAGE_SIZE).unwrap();
        let flat = FlatMemory::new(CPU_BASE, size);
        // Every byte distinct from its neighbours, across page boundaries too.
        let bytes: Vec<u8> = (0..CPU_PAGES * PAGE_SIZE)
            .map(|o| (o.wrapping_mul(7).wrapping_add(o / 253) & 0xff) as u8)
            .collect();
        flat.write(CPU_BASE, &bytes).unwrap();
        let guest = Arc::new(Counting {
            flat,
            calls: AtomicUsize::new(0),
        });
        let vm = Arc::new(GpuVm::new());
        let guest_target = |page| Target::Guest {
            cpu_va: cpu_page(page),
        };
        vm.map(
            range(GUEST, GUEST_PAGES),
            guest_target(GUEST_CPU_PAGE),
            Kind(1),
        )
        .unwrap();
        vm.map(
            range(ONLY, ONLY_PAGES),
            Target::GpuOnly {
                region: REGION,
                offset: REGION_OFFSET,
            },
            Kind(2),
        )
        .unwrap();
        vm.map(
            range(AFTER_ONLY, 1),
            guest_target(AFTER_ONLY_CPU_PAGE),
            Kind(3),
        )
        .unwrap();
        vm.map(range(BEFORE_SHIM, 1), guest_target(12), Kind(4))
            .unwrap();
        vm.map(range(SHIM, SHIM_PAGES), Target::SyncpointShim, Kind(5))
            .unwrap();
        vm.map(range(AFTER_SHIM, 1), guest_target(13), Kind(6))
            .unwrap();
        // Row 30: memory registered with the session, at GPU address = CPU address.
        vm.map(range(CPU_BASE, 2), guest_target(0), Kind(7))
            .unwrap();
        let pushed = Arc::new(Recorder::default());
        let syncpoints = Arc::new(Syncpoints::new(Arc::<Recorder>::clone(&pushed)));
        let mem = VmMemory::new(Arc::clone(&vm), Arc::clone(&guest), Arc::clone(&syncpoints));
        Self {
            vm,
            guest,
            syncpoints,
            pushed,
            mem,
        }
    }

    fn read(&self, va: u64, len: usize) -> Result<Vec<u8>, GpuFault> {
        let mut out = vec![0xa5; len];
        self.mem.read(va, &mut out).map(|()| out)
    }

    fn cpu(&self, va: u64, len: usize) -> Vec<u8> {
        let mut out = vec![0; len];
        self.guest.flat.read(va, &mut out).unwrap();
        out
    }

    /// Every byte of guest memory, read without counting.
    fn all_guest(&self) -> Vec<u8> {
        self.cpu(CPU_BASE, usize::try_from(CPU_PAGES * PAGE_SIZE).unwrap())
    }

    fn calls(&self) -> usize {
        self.guest.calls.load(Ordering::SeqCst)
    }

    fn store(&self, region: RegionId, offset: u64, len: usize) -> Vec<u8> {
        let mut out = vec![0xa5; len];
        self.vm.gpu_only().read(region, offset, &mut out).unwrap();
        out
    }

    fn counters(&self) -> Vec<u32> {
        (0..=LAST_ID).map(|i| self.syncpoints.read(id(i))).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Constants against the reference
// ---------------------------------------------------------------------------------------------

#[test]
fn the_shim_layout_is_prs() {
    // `GetNativeSyncpointShim` in `eden-pr4588/src/core/hle/service/nvdrv/devices/
    // nvhost_as_gpu.cpp` @ e5fb3b97: `stride = sizeof(u32); count = 192;`, and the read check
    // in `memory_manager.cpp`, `addr - shim < 192 * sizeof(u32)`.
    assert_eq!(SHIM_STRIDE, 4);
    assert_eq!(SYNCPOINT_COUNT, 192);
    assert_eq!(PAST_LAST_WORD, 0x300);
}

// ---------------------------------------------------------------------------------------------
// Each target kind
// ---------------------------------------------------------------------------------------------

#[test]
fn each_target_kind_resolves() {
    let r = Rig::new();
    // Guest, a page into the mapping: the CPU address a page further in, not the GPU address.
    let va = GUEST + PAGE_SIZE + 0x10;
    let cpu = cpu_page(GUEST_CPU_PAGE + 1) + 0x10;
    assert_eq!(r.read(va, 8).unwrap(), r.cpu(cpu, 8));
    r.mem.write(va, &[1, 2, 3]).unwrap();
    assert_eq!(r.cpu(cpu, 3), [1, 2, 3]);
    // Guest at GPU address = CPU address (row 30).
    assert_eq!(
        r.read(CPU_BASE + 0x20, 4).unwrap(),
        r.cpu(CPU_BASE + 0x20, 4)
    );

    // GPU-only: zero before any write, then the store at the region's offset.
    let only = ONLY + PAGE_SIZE + 0x30;
    assert_eq!(r.read(only, 4).unwrap(), [0; 4]);
    r.mem.write(only, &[9, 8, 7, 6]).unwrap();
    assert_eq!(r.read(only, 4).unwrap(), [9, 8, 7, 6]);
    assert_eq!(
        r.store(REGION, REGION_OFFSET + PAGE_SIZE + 0x30, 4),
        [9, 8, 7, 6]
    );

    // The shim: the syncpoint's value.
    r.syncpoints.increment_by(id(3), 0x8765_4321);
    assert_eq!(
        r.read(SHIM + 3 * SHIM_STRIDE, 4).unwrap(),
        0x8765_4321u32.to_le_bytes()
    );
}

#[test]
fn an_access_crosses_from_guest_into_gpu_only_at_the_exact_boundary() {
    let r = Rig::new();
    let guest_end = cpu_page(GUEST_CPU_PAGE + GUEST_PAGES);
    let before = r.cpu(guest_end - 5, 1);
    r.mem.write(ONLY - 4, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    // The guest's last four bytes, and nothing before them.
    assert_eq!(r.cpu(guest_end - 4, 4), [1, 2, 3, 4]);
    assert_eq!(r.cpu(guest_end - 5, 1), before);
    // The store's first four, and nothing after them.
    assert_eq!(r.store(REGION, REGION_OFFSET, 5), [5, 6, 7, 8, 0]);
    // The CPU page after the guest mapping's last is not touched by the GPU-only half.
    assert_ne!(r.cpu(guest_end, 4), [5, 6, 7, 8]);
    assert_eq!(r.read(ONLY - 4, 8).unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);

    // And out of GPU-only memory into the guest page after it.
    let only_end = AFTER_ONLY;
    r.mem
        .write(only_end - 2, &[0xa1, 0xa2, 0xa3, 0xa4])
        .unwrap();
    assert_eq!(
        r.store(REGION, REGION_OFFSET + ONLY_PAGES * PAGE_SIZE - 2, 2),
        [0xa1, 0xa2]
    );
    assert_eq!(r.cpu(cpu_page(AFTER_ONLY_CPU_PAGE), 2), [0xa3, 0xa4]);
    assert_eq!(r.read(only_end - 2, 4).unwrap(), [0xa1, 0xa2, 0xa3, 0xa4]);
}

#[test]
fn the_first_write_allocates_and_unwritten_pages_read_zero() {
    let r = Rig::new();
    let len = usize::try_from(ONLY_PAGES * PAGE_SIZE).unwrap();
    assert_eq!(r.read(ONLY, len).unwrap(), vec![0; len]);
    assert_eq!(r.vm.gpu_only().resident_pages(), 0);

    r.mem.write(ONLY + PAGE_SIZE + 7, &[0x5a]).unwrap();
    assert_eq!(r.vm.gpu_only().resident_pages(), 1);
    assert_eq!(r.vm.gpu_only().resident_pages_of(REGION), 1);
    let mut expected = vec![0; len];
    expected[usize::try_from(PAGE_SIZE).unwrap() + 7] = 0x5a;
    assert_eq!(r.read(ONLY, len).unwrap(), expected);
    // A second write to the same page allocates nothing more.
    r.mem.write(ONLY + PAGE_SIZE, &[1]).unwrap();
    assert_eq!(r.vm.gpu_only().resident_pages(), 1);
}

#[test]
fn nothing_is_charged_to_guest_memory() {
    let r = Rig::new();
    let guest = r.all_guest();
    let calls = r.calls();
    let len = usize::try_from(ONLY_PAGES * PAGE_SIZE).unwrap();
    r.mem.write(ONLY, &vec![0xee; len]).unwrap();
    assert_eq!(r.read(ONLY, len).unwrap(), vec![0xee; len]);
    r.mem.write(SHIM, &[1, 0, 0, 0]).unwrap();
    r.read(SHIM, 16).unwrap();
    // GPU-only and shim accesses make no call to guest memory and change none of its bytes.
    assert_eq!(r.calls(), calls);
    assert_eq!(r.all_guest(), guest);
    assert_eq!(
        r.vm.gpu_only().resident_pages(),
        usize::try_from(ONLY_PAGES).unwrap()
    );
}

// ---------------------------------------------------------------------------------------------
// Destroys
// ---------------------------------------------------------------------------------------------

#[test]
fn a_destroy_frees_only_when_applied_and_its_neighbour_survives() {
    let r = Rig::new();
    // A neighbouring region mapped from exactly the end of `REGION`'s range, over `AFTER_ONLY`.
    let neighbour = RegionId(REGION.0 + 1);
    r.vm.map(
        range(AFTER_ONLY, 1),
        Target::GpuOnly {
            region: neighbour,
            offset: 0,
        },
        Kind(2),
    )
    .unwrap();
    let len = usize::try_from(ONLY_PAGES * PAGE_SIZE).unwrap();
    r.mem.write(ONLY, &vec![0x11; len]).unwrap();
    r.mem.write(AFTER_ONLY, &[0x22; 4]).unwrap();
    assert_eq!(r.vm.gpu_only().resident_pages(), 4);
    let whole = range(ONLY, ONLY_PAGES);

    // Updates that are not this region's destroy free nothing.
    for update in [
        VmUpdate::ModifyKind {
            range: whole,
            kind: Kind(9),
        },
        VmUpdate::Unmap {
            range: range(ONLY, 1),
        },
        VmUpdate::DestroyMemory {
            range: range(ONLY + PAGE_SIZE, 1),
        },
        VmUpdate::DestroySparseRegion {
            range: range(HOLE, 1),
            region: RegionId(REGION.0 - 1),
        },
    ] {
        r.vm.apply(&update);
        assert_eq!(r.vm.gpu_only().resident_pages_of(REGION), 3, "{update:?}");
    }
    // The page still mapped reads its bytes.
    assert_eq!(r.read(ONLY + 2 * PAGE_SIZE, 2).unwrap(), [0x11; 2]);

    r.vm.apply(&VmUpdate::DestroySparseRegion {
        range: whole,
        region: REGION,
    });
    assert_eq!(r.vm.gpu_only().resident_pages_of(REGION), 0);
    assert_eq!(r.store(REGION, REGION_OFFSET + 2 * PAGE_SIZE, 2), [0; 2]);
    assert_eq!(
        r.read(ONLY + 2 * PAGE_SIZE, 1),
        Err(GpuFault::Unmapped {
            va: ONLY + 2 * PAGE_SIZE
        })
    );
    // The neighbour starting at the destroyed range's end keeps its page and its bytes.
    assert_eq!(r.vm.gpu_only().resident_pages_of(neighbour), 1);
    assert_eq!(r.read(AFTER_ONLY, 5).unwrap(), [0x22, 0x22, 0x22, 0x22, 0]);
}

#[test]
fn a_restored_table_starts_with_an_empty_store() {
    let r = Rig::new();
    r.mem.write(ONLY, &[1, 2]).unwrap();
    let restored = Arc::new(GpuVm::restore(&r.vm.save().unwrap()).unwrap());
    assert_eq!(restored.gpu_only().resident_pages(), 0);
    let mem = VmMemory::new(
        Arc::clone(&restored),
        Arc::clone(&r.guest),
        Arc::clone(&r.syncpoints),
    );
    let mut out = [0xa5; 2];
    mem.read(ONLY, &mut out).unwrap();
    assert_eq!(out, [0; 2]);
    // The restored table is used: a write lands in its own store.
    mem.write(ONLY, &[3]).unwrap();
    assert_eq!(restored.gpu_only().resident_pages(), 1);
    assert_eq!(r.store(REGION, REGION_OFFSET, 2), [1, 2]);
}

// ---------------------------------------------------------------------------------------------
// The syncpoint shim
// ---------------------------------------------------------------------------------------------

#[test]
fn the_shim_reads_the_syncpoints_value() {
    let r = Rig::new();
    // Distinct values that differ above the low byte, and one at or above 2^31.
    for (i, value) in [
        (0u8, 0x8000_0001u32),
        (1, 0x0102_0304),
        (LAST_ID, 0xfe00_00ff),
    ] {
        r.syncpoints.increment_by(id(i), value);
    }
    let word = |offset: u64| -> u32 {
        u32::from_le_bytes(r.read(SHIM + offset, 4).unwrap().try_into().unwrap())
    };
    assert_eq!(word(0), 0x8000_0001);
    assert_eq!(word(SHIM_STRIDE), 0x0102_0304);
    assert_eq!(word(2 * SHIM_STRIDE), 0);
    // The last syncpoint's word, and the first word past it.
    assert_eq!(word(PAST_LAST_WORD - SHIM_STRIDE), 0xfe00_00ff);
    assert_eq!(word(PAST_LAST_WORD), 0);
    assert_eq!(word(PAGE_SIZE - SHIM_STRIDE), 0);
    // The value is read at the access, not cached.
    r.syncpoints.increment(id(1));
    assert_eq!(word(SHIM_STRIDE), 0x0102_0305);
}

#[test]
fn the_shim_reads_counters_at_any_width_and_alignment() {
    let r = Rig::new();
    r.syncpoints.increment_by(id(0), 0x4433_2211);
    r.syncpoints.increment_by(id(1), 0x8877_6655);
    r.syncpoints.increment_by(id(LAST_ID), 0xddcc_bbaa);
    // Eight bytes: two counters.
    assert_eq!(
        r.read(SHIM, 8).unwrap(),
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
    );
    // Two bytes off a word boundary, and three spanning two words.
    assert_eq!(r.read(SHIM + 2, 2).unwrap(), [0x33, 0x44]);
    assert_eq!(r.read(SHIM + 3, 3).unwrap(), [0x44, 0x55, 0x66]);
    // Across the last syncpoint's word into the zero bytes after it.
    assert_eq!(
        r.read(SHIM + PAST_LAST_WORD - 2, 4).unwrap(),
        [0xcc, 0xdd, 0, 0]
    );
    // The whole page, then the guest page after it.
    let page = r
        .read(SHIM, usize::try_from(PAGE_SIZE + 2).unwrap())
        .unwrap();
    for i in 0..=LAST_ID {
        let at = usize::from(i) * WORD_BYTES;
        let value = u32::from_le_bytes(page[at..at + WORD_BYTES].try_into().unwrap());
        assert_eq!(value, r.syncpoints.read(id(i)), "syncpoint {i}");
    }
    let past = usize::try_from(PAST_LAST_WORD).unwrap();
    let page_bytes = usize::try_from(PAGE_SIZE).unwrap();
    assert!(page[past..page_bytes].iter().all(|&b| b == 0));
    assert_eq!(page[page_bytes..], r.cpu(cpu_page(13), 2)[..]);
}

#[test]
fn a_release_to_the_shim_increments_its_syncpoint_once() {
    let r = Rig::new();
    let five = id(5);
    r.syncpoints.watch(five);
    let at = SHIM + 5 * SHIM_STRIDE;
    // A 4-byte, an 8-byte and a 16-byte release, each with a payload the counter does not take.
    r.mem.write(at, &0xdead_beefu32.to_le_bytes()).unwrap();
    assert_eq!(r.syncpoints.read(five), 1);
    r.mem.write(at, &u64::MAX.to_le_bytes()).unwrap();
    assert_eq!(r.syncpoints.read(five), 2);
    r.mem.write(at, &[0x77; 16]).unwrap();
    assert_eq!(r.syncpoints.read(five), 3);
    // Its neighbours, whose words a wide release covers, are unchanged.
    let mut expected = vec![0; usize::from(LAST_ID) + 1];
    expected[5] = 3;
    assert_eq!(r.counters(), expected);
    // Watched: each increment pushed its new value.
    assert_eq!(
        r.pushed.taken(),
        (1..=3)
            .map(|value| GpuCompletion::SyncpointChanged { id: five, value })
            .collect::<Vec<_>>()
    );
    // The last syncpoint's word takes a release, a 16-byte one included (it stays in the page).
    r.mem
        .write(SHIM + PAST_LAST_WORD - SHIM_STRIDE, &[0; 16])
        .unwrap();
    assert_eq!(r.syncpoints.read(id(LAST_ID)), 1);
    // Not watched: nothing pushed.
    assert_eq!(r.pushed.taken(), vec![]);
    // An empty write increments nothing.
    r.mem.write(at, &[]).unwrap();
    assert_eq!(r.syncpoints.read(five), 3);
}

#[test]
fn writes_the_shim_cannot_take_are_refused() {
    let r = Rig::new();
    let guest = r.all_guest();
    let page = usize::try_from(PAGE_SIZE).unwrap();
    let cases: [(u64, usize, u64); 5] = [
        // Off a word boundary: one byte past syncpoint 1's word.
        (SHIM + SHIM_STRIDE + 1, 4, SHIM + SHIM_STRIDE + 1),
        // The first byte past the last syncpoint's word.
        (SHIM + PAST_LAST_WORD, 4, SHIM + PAST_LAST_WORD),
        // The shim's last word: past the last syncpoint's.
        (
            SHIM + PAGE_SIZE - SHIM_STRIDE,
            4,
            SHIM + PAGE_SIZE - SHIM_STRIDE,
        ),
        // From the guest page before into syncpoint 0's word.
        (SHIM - 4, 8, SHIM),
        // From syncpoint 0's word out of the page into the guest page after.
        (SHIM, page + 4, SHIM),
    ];
    for (va, len, refused_at) in cases {
        assert_eq!(
            r.mem.write(va, &vec![0x3c; len]),
            Err(GpuFault::ShimWrite { va: refused_at }),
            "{va:#x}+{len}"
        );
    }
    // Nothing written to the guest pages either side, and nothing incremented.
    assert_eq!(r.all_guest(), guest);
    assert_eq!(r.counters(), vec![0; usize::from(LAST_ID) + 1]);
    // The same start one byte shorter, ending exactly at the page's end, stays in the shim and
    // starts on syncpoint 0's word: one release.
    r.mem.write(SHIM, &vec![0x3c; page]).unwrap();
    let mut expected = vec![0; usize::from(LAST_ID) + 1];
    expected[0] = 1;
    assert_eq!(r.counters(), expected);
    // And the word before the first refused one is the last syncpoint's.
    r.mem
        .write(SHIM + PAST_LAST_WORD - SHIM_STRIDE, &[0; 4])
        .unwrap();
    assert_eq!(r.syncpoints.read(id(LAST_ID)), 1);
}

#[test]
fn the_first_fault_in_address_order_is_reported() {
    let r = Rig::new();
    let guest = r.all_guest();
    // A refused shim piece, then an unmapped page: the shim is reported. `AFTER_SHIM` is mapped,
    // so the hole is made below by unmapping it.
    r.vm.apply(&VmUpdate::Unmap {
        range: range(AFTER_SHIM, 1),
    });
    let page = usize::try_from(PAGE_SIZE).unwrap();
    assert_eq!(
        r.mem.write(SHIM + PAGE_SIZE - 2, &[1; 4]),
        Err(GpuFault::ShimWrite {
            va: SHIM + PAGE_SIZE - 2
        })
    );
    // An unmapped page, then a shim piece: the unmapped page is reported.
    r.vm.apply(&VmUpdate::Unmap {
        range: range(BEFORE_SHIM, 1),
    });
    assert_eq!(
        r.mem.write(SHIM - 4, &[1; 8]),
        Err(GpuFault::Unmapped { va: SHIM - 4 })
    );
    assert_eq!(
        r.mem.write(BEFORE_SHIM, &vec![1; page + 4]),
        Err(GpuFault::Unmapped { va: BEFORE_SHIM })
    );
    assert_eq!(r.all_guest(), guest);
    assert_eq!(r.counters(), vec![0; usize::from(LAST_ID) + 1]);
}

#[test]
fn the_shim_fault_reads_as_plain_words() {
    assert_eq!(
        GpuFault::ShimWrite { va: 0xff_ffff_f000 }.to_string(),
        "GPU address 0xfffffff000 is in the syncpoint shim, which takes only a write that starts \
         on one syncpoint's word and stays in the shim page"
    );
}

// ---------------------------------------------------------------------------------------------
// Faults
// ---------------------------------------------------------------------------------------------

#[test]
fn an_unmapped_page_faults_at_its_first_address() {
    let r = Rig::new();
    let guest = r.all_guest();
    // Read from the guest page before the hole into the hole: the first part is copied.
    r.vm.map(
        range(HOLE - PAGE_SIZE, 1),
        Target::Guest {
            cpu_va: cpu_page(14),
        },
        Kind(0),
    )
    .unwrap();
    let mut out = [0xa5; 8];
    assert_eq!(
        r.mem.read(HOLE - 4, &mut out),
        Err(GpuFault::Unmapped { va: HOLE })
    );
    assert_eq!(out[..4], r.cpu(cpu_page(15) - 4, 4)[..]);
    assert_eq!(out[4..], [0xa5; 4]);
    // A write into the hole writes nothing, not even the mapped part.
    assert_eq!(
        r.mem.write(HOLE - 4, &[1; 8]),
        Err(GpuFault::Unmapped { va: HOLE })
    );
    assert_eq!(r.all_guest(), guest);
    // A write wholly in GPU-only memory then into the hole allocates nothing.
    r.vm.map(
        range(HOLE - PAGE_SIZE, 1),
        Target::GpuOnly {
            region: REGION,
            offset: 0,
        },
        Kind(0),
    )
    .unwrap();
    assert_eq!(
        r.mem.write(HOLE - 4, &[1; 8]),
        Err(GpuFault::Unmapped { va: HOLE })
    );
    assert_eq!(r.vm.gpu_only().resident_pages(), 0);
}

#[test]
fn a_guest_fault_names_the_gpu_address_and_the_cpu_one() {
    let r = Rig::new();
    // A page leading past the end of the flat guest memory.
    let outside = cpu_page(CPU_PAGES);
    r.vm.map(range(HOLE, 1), Target::Guest { cpu_va: outside }, Kind(0))
        .unwrap();
    assert_eq!(
        r.read(HOLE + 8, 4),
        Err(GpuFault::Guest {
            va: HOLE + 8,
            fault: Fault {
                address: outside + 8
            }
        })
    );
    assert_eq!(
        r.mem.write(HOLE + 8, &[1]),
        Err(GpuFault::Guest {
            va: HOLE + 8,
            fault: Fault {
                address: outside + 8
            }
        })
    );
}

#[test]
fn an_access_past_the_end_of_the_address_space_is_out_of_range() {
    let r = Rig::new();
    let va = u64::MAX - 2;
    assert_eq!(r.read(va, 4), Err(GpuFault::OutOfRange { va, len: 4 }));
    assert_eq!(
        r.mem.write(va, &[0; 4]),
        Err(GpuFault::OutOfRange { va, len: 4 })
    );
    // Ending exactly at `u64::MAX` is representable: the page is merely unmapped.
    assert_eq!(r.read(va, 2), Err(GpuFault::Unmapped { va }));
}

// ---------------------------------------------------------------------------------------------
// Translation caches and versions
// ---------------------------------------------------------------------------------------------

#[test]
fn a_version_bump_empties_every_threads_cache() {
    let r = Rig::new();
    let other = r.mem.for_thread();
    let va = GUEST + 0x40;
    // Both threads' caches hold the page's translation.
    let old = r.cpu(cpu_page(GUEST_CPU_PAGE) + 0x40, 4);
    assert_eq!(r.read(va, 4).unwrap(), old);
    let mut out = [0; 4];
    other.read(va, &mut out).unwrap();
    assert_eq!(out[..], old[..]);
    let before = r.mem.generation();
    assert_eq!(before, other.generation());

    // A map moves the page elsewhere: both threads read the new place.
    r.vm.map(
        range(GUEST, 1),
        Target::Guest {
            cpu_va: cpu_page(8),
        },
        Kind(1),
    )
    .unwrap();
    let new = r.cpu(cpu_page(8) + 0x40, 4);
    assert_ne!(new, old);
    assert_eq!(r.read(va, 4).unwrap(), new);
    other.read(va, &mut out).unwrap();
    assert_eq!(out[..], new[..]);
    assert_eq!(r.mem.generation(), before + 1);
    assert_eq!(other.generation(), Translate::generation(&*r.vm));

    // An ordered unmap: both threads fault.
    r.vm.apply(&VmUpdate::Unmap {
        range: range(GUEST, 1),
    });
    assert_eq!(r.read(va, 4), Err(GpuFault::Unmapped { va: GUEST + 0x40 }));
    assert_eq!(other.read(va, &mut out), Err(GpuFault::Unmapped { va }));
    assert_eq!(r.mem.generation(), before + 2);
}

#[test]
fn another_threads_memory_shares_the_table_the_guest_and_the_syncpoints() {
    let r = Rig::new();
    let other = r.mem.for_thread();
    assert!(Arc::ptr_eq(other.vm(), r.mem.vm()));
    other.write(GUEST, &[4, 5]).unwrap();
    assert_eq!(r.read(GUEST, 2).unwrap(), [4, 5]);
    other.write(ONLY, &[6]).unwrap();
    assert_eq!(r.read(ONLY, 1).unwrap(), [6]);
    other.write(SHIM + 2 * SHIM_STRIDE, &[0; 4]).unwrap();
    assert_eq!(r.syncpoints.read(id(2)), 1);
}

// ---------------------------------------------------------------------------------------------
// GPU-only ranges
// ---------------------------------------------------------------------------------------------

#[test]
fn is_gpu_only_answers_for_whole_ranges() {
    let r = Rig::new();
    let size = ONLY_PAGES * PAGE_SIZE;
    assert!(r.mem.is_gpu_only(ONLY, size));
    assert!(r.mem.is_gpu_only(ONLY + 1, size - 1));
    assert!(r.mem.is_gpu_only(ONLY + size - 1, 1));
    // One byte into a neighbour on either side.
    assert!(!r.mem.is_gpu_only(ONLY, size + 1));
    assert!(!r.mem.is_gpu_only(ONLY - 1, size));
    assert!(!r.mem.is_gpu_only(ONLY - 1, 1));
    assert!(!r.mem.is_gpu_only(AFTER_ONLY, 1));
    // Empty, unmapped, the shim, and an end past the address space.
    assert!(!r.mem.is_gpu_only(ONLY, 0));
    assert!(!r.mem.is_gpu_only(HOLE, PAGE_SIZE));
    assert!(!r.mem.is_gpu_only(SHIM, 4));
    assert!(!r.mem.is_gpu_only(u64::MAX - 1, 2));
    // GPU-only pages of two regions next to each other: still all GPU-only.
    r.vm.map(
        range(AFTER_ONLY, 1),
        Target::GpuOnly {
            region: RegionId(REGION.0 + 1),
            offset: 0,
        },
        Kind(0),
    )
    .unwrap();
    assert!(r.mem.is_gpu_only(ONLY, size + PAGE_SIZE));
}

// ---------------------------------------------------------------------------------------------
// A GPU-only write whose page changed since its translation
// ---------------------------------------------------------------------------------------------

#[test]
fn a_gpu_only_write_goes_where_its_page_leads_now() {
    let r = Rig::new();
    let stale = Translation {
        target: Target::GpuOnly {
            region: REGION,
            offset: 0,
        },
        kind: Kind(0),
    };
    let len = 4;
    // The page still GPU-only: written to the store at the page's *current* offset.
    r.mem
        .write_piece(ONLY + 8, stale, &[1, 2, 3, 4], len)
        .unwrap();
    assert_eq!(r.store(REGION, REGION_OFFSET + 8, 4), [1, 2, 3, 4]);
    assert_eq!(r.store(REGION, 8, 4), [0; 4]);
    // Now a guest page: written to guest memory, nothing allocated.
    let pages = r.vm.gpu_only().resident_pages();
    r.mem.write_piece(GUEST + 8, stale, &[5, 6], 2).unwrap();
    assert_eq!(r.cpu(cpu_page(GUEST_CPU_PAGE) + 8, 2), [5, 6]);
    // Now the shim: an increment, checked as a release.
    r.mem
        .write_piece(SHIM + SHIM_STRIDE, stale, &[0; 4], len)
        .unwrap();
    assert_eq!(r.syncpoints.read(id(1)), 1);
    assert_eq!(
        r.mem.write_piece(SHIM + 2, stale, &[0; 2], 2),
        Err(GpuFault::ShimWrite { va: SHIM + 2 })
    );
    assert_eq!(
        r.mem.write_piece(SHIM + SHIM_STRIDE, stale, &[0; 4], 8),
        Err(GpuFault::ShimWrite {
            va: SHIM + SHIM_STRIDE
        })
    );
    // Now unmapped: a fault, nothing allocated.
    assert_eq!(
        r.mem.write_piece(HOLE, stale, &[7], 1),
        Err(GpuFault::Unmapped { va: HOLE })
    );
    assert_eq!(r.vm.gpu_only().resident_pages(), pages);
}

#[test]
fn write_gpu_only_reports_where_a_page_leads_now() {
    let r = Rig::new();
    assert_eq!(
        r.vm.write_gpu_only(ONLY + 3, &[1]),
        Ok(GpuOnlyWrite::Written)
    );
    assert_eq!(r.store(REGION, REGION_OFFSET + 3, 1), [1]);
    assert_eq!(
        r.vm.write_gpu_only(GUEST + PAGE_SIZE, &[1]),
        Ok(GpuOnlyWrite::Elsewhere(Some(Translation {
            target: Target::Guest {
                cpu_va: cpu_page(GUEST_CPU_PAGE + 1)
            },
            kind: Kind(1),
        })))
    );
    assert_eq!(
        r.vm.write_gpu_only(SHIM, &[1]),
        Ok(GpuOnlyWrite::Elsewhere(Some(Translation {
            target: Target::SyncpointShim,
            kind: Kind(5),
        })))
    );
    assert_eq!(
        r.vm.write_gpu_only(HOLE, &[1]),
        Ok(GpuOnlyWrite::Elsewhere(None))
    );
    assert_eq!(r.vm.gpu_only().resident_pages(), 1);
}

/// How long the threaded test may take; reached only when something hangs.
const PATIENCE: Duration = Duration::from_secs(30);

#[test]
fn a_write_racing_a_destroy_never_leaves_a_page_behind() {
    // Writers hammer a sparse region's page while the main thread destroys the region. Every
    // write that lands, lands before the free (which then frees it) or faults after the
    // destroy; none allocates a page of the destroyed region after the free. The race is
    // timing-dependent: this checks the outcome on every run and cannot show every
    // interleaving (module docs of `crate::vm`, "The GPU-only store").
    let (done, result) = mpsc::channel();
    thread::spawn(move || {
        for round in 0..200u32 {
            let r = Rig::new();
            let stop = Arc::new(AtomicBool::new(false));
            let writers: Vec<_> = (0..2)
                .map(|_| {
                    let mem = r.mem.for_thread();
                    let stop = Arc::clone(&stop);
                    thread::spawn(move || {
                        let mut writes = 0u32;
                        while !stop.load(Ordering::SeqCst) {
                            if mem.write(ONLY, &[1]).is_ok() {
                                writes += 1;
                            } else {
                                break;
                            }
                        }
                        writes
                    })
                })
                .collect();
            // Let the writers start before the destroy, some rounds more than others.
            for _ in 0..round % 7 {
                thread::yield_now();
            }
            r.vm.apply(&VmUpdate::DestroySparseRegion {
                range: range(ONLY, ONLY_PAGES),
                region: REGION,
            });
            stop.store(true, Ordering::SeqCst);
            for writer in writers {
                writer.join().unwrap();
            }
            assert_eq!(
                r.vm.gpu_only().resident_pages_of(REGION),
                0,
                "round {round}"
            );
        }
        let _ = done.send(());
    });
    result
        .recv_timeout(PATIENCE)
        .expect("the race test did not finish in time");
}
