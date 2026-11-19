//! Tests of the buffer cache, with the test GPU memory, the real `WriteWatch` driven by its fake
//! kernel, and the recording backend.

use std::sync::Arc;

use starquake_memory::{
    FakeWatchKernel, FlatMemory, GuestMemory, PAGE_SIZE, PageRange, WriteWatch,
};
use starquake_render::caps::{DynamicState, StageSupport};
use starquake_render::command::{BufferSlice, CopyDesc, CopyRegion};
use starquake_render::error::RenderError;
use starquake_render::{
    Backend, BufferId, Caps, Command, CommandList, Features, FenceValue, FormatFeatures,
    FormatTable, Limits, Readback, ReadbackId, RecordingBackend, SampleCounts,
};

use super::*;
use crate::interface::{Kind, RegionId, VmUpdate};
use crate::memory::{GpuFault, TestGpuMemory};

/// Where the test memory starts, at the same GPU and CPU address (row 30).
const BASE: u64 = 0x8000_0000;
/// Pages of test memory, each mapped on its own so a test can unmap one.
const PAGES: u64 = 128;
/// The device's largest buffer, in pages.
const MAX_BUFFER_PAGES: u64 = 32;
/// The device's largest staging slice, in pages: smaller than most entries, so uploads split.
const MAX_STAGING_PAGES: u64 = 2;
/// The budget most rigs report: large enough that nothing is collected or refused.
const LARGE_BUDGET: u64 = 1 << 40;

type Memory = TestGpuMemory<FlatMemory>;

fn page(n: u64) -> u64 {
    BASE + n * PAGE_SIZE
}

fn pages(first: u64, count: u64) -> PageRange {
    PageRange::new(page(first), count * PAGE_SIZE).unwrap()
}

fn caps(max_staging: u64) -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: MAX_BUFFER_PAGES * PAGE_SIZE,
            max_staging_size: max_staging,
            max_texture_dimension_1d: 4096,
            max_texture_dimension_2d: 4096,
            max_texture_dimension_3d: 256,
            max_texture_array_layers: 64,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 16,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 16,
            max_compute_workgroups_per_dimension: 65_535,
            max_uniform_buffers_per_stage: 12,
            max_storage_buffers_per_stage: 8,
            max_sampled_textures_per_stage: 16,
            max_storage_textures_per_stage: 4,
            max_samplers_per_stage: 16,
        },
        features: Features::ALL,
        stages: StageSupport::ALL,
        dynamic_state: DynamicState::ALL,
        sample_counts: SampleCounts::X1,
        view_swizzle: false,
        formats: FormatTable::uniform(FormatFeatures::ALL),
    }
}

/// Every byte of the test memory distinct from its neighbours, across pages too.
fn memory() -> Memory {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mem = TestGpuMemory::new(FlatMemory::new(BASE, size));
    for n in 0..PAGES {
        mem.map(pages(n, 1), page(n)).unwrap();
    }
    let bytes: Vec<u8> = (0..PAGES * PAGE_SIZE)
        .map(|o| (o.wrapping_mul(7).wrapping_add(o / 251) & 0xff) as u8)
        .collect();
    mem.guest().write(BASE, &bytes).unwrap();
    mem
}

/// Collects what the dirty drain hands on.
#[derive(Default)]
struct Sink(Vec<PageRange>);

impl InvalidationSink for Sink {
    fn invalidate(&mut self, range: PageRange) {
        self.0.push(range);
    }
}

struct Rig {
    backend: Arc<RecordingBackend>,
    watch: Arc<WriteWatch>,
    kernel: FakeWatchKernel,
    mem: Memory,
    cache: BufferCache,
    list: CommandList,
    readbacks: u32,
}

impl Rig {
    fn new() -> Self {
        Self::with(LARGE_BUDGET, MAX_STAGING_PAGES * PAGE_SIZE)
    }

    fn with_budget(allowed: u64) -> Self {
        Self::with(allowed, MAX_STAGING_PAGES * PAGE_SIZE)
    }

    fn with(allowed: u64, max_staging: u64) -> Self {
        let backend = Arc::new(RecordingBackend::new(caps(max_staging), allowed));
        let watch = Arc::new(WriteWatch::new());
        let cache = BufferCache::new(Arc::<RecordingBackend>::clone(&backend), Arc::clone(&watch));
        Self {
            kernel: FakeWatchKernel::new(Arc::clone(&watch)),
            backend,
            watch,
            mem: memory(),
            cache,
            list: CommandList::new(),
            readbacks: 0,
        }
    }

    fn try_bind(&mut self, va: u64, len: u64) -> Result<BufferSlice, BufferError> {
        self.cache.bind(&self.mem, va, len, &mut self.list)
    }

    fn bind(&mut self, va: u64, len: u64) -> BufferSlice {
        self.try_bind(va, len).unwrap()
    }

    /// Binds and returns the uploads that bind added, as (buffer, offset, size).
    fn bind_uploads(&mut self, va: u64, len: u64) -> (BufferSlice, Vec<(BufferId, u64, u64)>) {
        let before = self.list.commands.len();
        let slice = self.bind(va, len);
        let uploads = uploads(&self.list.commands[before..]);
        (slice, uploads)
    }

    /// Submits the open list and reports it to the cache.
    fn submit(&mut self) -> FenceValue {
        let list = core::mem::take(&mut self.list);
        let fence = self.backend.submit(list).unwrap();
        self.cache.submitted(fence);
        fence
    }

    /// The host buffer's bytes for `slice`, read back through a list of its own (after
    /// submitting the open list, so its uploads land first).
    fn host_bytes(&mut self, slice: BufferSlice) -> Vec<u8> {
        self.submit();
        self.readbacks += 1;
        let id = ReadbackId::new(self.readbacks, 0);
        let mut list = CommandList::new();
        list.push(Command::Readback {
            from: CopyRegion::Buffer(slice),
            id,
        });
        self.backend.submit(list).unwrap();
        match self.backend.take_readback(id) {
            Some(Readback::Bytes(bytes)) => bytes,
            other => unreachable!("readback {other:?}"),
        }
    }

    fn guest_bytes(&self, va: u64, len: u64) -> Vec<u8> {
        let mut out = vec![0; usize::try_from(len).unwrap()];
        self.mem.guest().read(va, &mut out).unwrap();
        out
    }

    /// A guest CPU store: the bytes change, and each watched page it touches faults once.
    fn cpu_write(&self, va: u64, data: &[u8]) -> u64 {
        self.mem.guest().write(va, data).unwrap();
        self.kernel
            .cpu_write(va, u64::try_from(data.len()).unwrap())
            .unwrap()
    }

    fn sync(&mut self) -> Vec<PageRange> {
        let mut sink = Sink::default();
        self.cache.sync_cpu_writes(&mut sink);
        sink.0
    }

    /// A kernel entry: arms every waiting watch request.
    fn enter(&mut self) -> usize {
        self.kernel.enter().unwrap()
    }

    fn entry(&self, va: u64) -> BufferEntry {
        self.cache.entry(va).unwrap()
    }

    /// The recording backend's log lines.
    fn log(&self) -> Vec<String> {
        self.backend.snapshot().lines().map(str::to_owned).collect()
    }
}

fn uploads(commands: &[Command]) -> Vec<(BufferId, u64, u64)> {
    commands
        .iter()
        .filter_map(|c| match c {
            Command::Upload {
                from,
                to: CopyRegion::Buffer(s),
            } => {
                assert_eq!(u64::try_from(from.len()).unwrap(), s.size);
                Some((s.buffer, s.offset, s.size))
            }
            _ => None,
        })
        .collect()
}

/// The uploads covering `count` pages from `offset` in staging-sized pieces.
fn chunked(buffer: BufferId, offset: u64, count: u64) -> Vec<(BufferId, u64, u64)> {
    let chunk = MAX_STAGING_PAGES * PAGE_SIZE;
    let mut out = Vec::new();
    let mut at = 0;
    while at < count * PAGE_SIZE {
        let size = chunk.min(count * PAGE_SIZE - at);
        out.push((buffer, offset + at, size));
        at += size;
    }
    out
}

/// The index of the first log line equal to `line`.
fn line_index(log: &[String], line: &str) -> usize {
    log.iter()
        .position(|l| l == line)
        .unwrap_or_else(|| unreachable!("no line {line:?} in\n{}", log.join("\n")))
}

fn destroy_line(buffer: BufferId, after: u64) -> String {
    let id = if buffer.generation() == 0 {
        format!("b{}", buffer.index())
    } else {
        format!("b{}@{}", buffer.index(), buffer.generation())
    };
    format!("destroy {id} after fence {after}: now")
}

// --- Entries ---

#[test]
fn a_first_bind_makes_one_buffer_and_uploads_every_page_it_touches() {
    let mut rig = Rig::new();
    // 16 bytes into page 1, three pages long: the bytes touch pages 1 to 4.
    let (slice, ups) = rig.bind_uploads(page(1) + 16, 3 * PAGE_SIZE);
    assert_eq!(slice.offset, 16);
    assert_eq!(slice.size, 3 * PAGE_SIZE);
    assert_eq!(ups, chunked(slice.buffer, 0, 4));
    let entry = rig.entry(page(1));
    assert_eq!(entry.range, pages(1, 4));
    assert_eq!(entry.buffer, slice.buffer);
    assert_eq!(rig.cache.bytes(), 4 * PAGE_SIZE);
    assert_eq!(rig.cache.len(), 1);
    assert!(!rig.cache.is_empty());
    assert!(rig.log().iter().any(|l| l.ends_with(&format!(
        "buffer {} bytes, usage VERTEX|INDEX|UNIFORM|STORAGE|INDIRECT|COPY_SRC|COPY_DST",
        4 * PAGE_SIZE
    ))));
    assert_eq!(
        rig.host_bytes(slice),
        rig.guest_bytes(page(1) + 16, 3 * PAGE_SIZE)
    );
}

#[test]
fn a_bind_inside_an_entry_reuses_it_and_a_touching_bind_does_not() {
    let mut rig = Rig::new();
    let a = rig.bind(page(2), 2 * PAGE_SIZE);
    let inner = rig.bind(page(3) + 8, 4);
    assert_eq!(
        (inner.buffer, inner.offset, inner.size),
        (a.buffer, PAGE_SIZE + 8, 4)
    );
    // Starts exactly at the entry's end: a separate entry.
    let next = rig.bind(page(4), PAGE_SIZE);
    assert_ne!(next.buffer, a.buffer);
    // Ends exactly at the entry's start: a separate entry.
    let before = rig.bind(page(1), PAGE_SIZE);
    assert_ne!(before.buffer, a.buffer);
    let ranges: Vec<_> = rig.cache.entries().map(|e| e.range).collect();
    assert_eq!(ranges, [pages(1, 1), pages(2, 2), pages(4, 1)]);
}

#[test]
fn every_new_entry_asks_for_a_watch_over_its_pages() {
    let mut rig = Rig::new();
    rig.bind(page(5), 3 * PAGE_SIZE);
    let mut requests = Vec::new();
    rig.watch.take_requests(&mut requests);
    assert_eq!(
        requests.iter().map(|r| r.range()).collect::<Vec<_>>(),
        [pages(5, 3)]
    );
    rig.watch.arm(&requests).unwrap();
}

#[test]
fn a_bind_overlapping_entries_joins_them_by_copy_and_uploads_only_the_gap() {
    let mut rig = Rig::new();
    let a = rig.bind(page(0), 2 * PAGE_SIZE);
    let b = rig.bind(page(3), 2 * PAGE_SIZE);
    rig.enter();
    rig.bind(page(0), 1);
    rig.bind(page(3), 1);
    rig.submit();
    // A guest store to page 1 (in A) before the join: stale when A joins.
    assert_eq!(rig.cpu_write(page(1) + 9, &[0xA5; 3]), 1);
    rig.sync();
    rig.enter();
    let before = rig.list.commands.len();
    // Pages 1 to 3: overlaps A and B, and page 2 is in neither.
    let joined = rig.bind(page(1), 3 * PAGE_SIZE);
    let commands = rig.list.commands[before..].to_vec();
    assert_ne!(joined.buffer, a.buffer);
    assert_ne!(joined.buffer, b.buffer);
    assert_eq!(joined.offset, PAGE_SIZE);
    let copy = |src: BufferId, dst_offset: u64| {
        Command::Copy(CopyDesc {
            src: CopyRegion::Buffer(BufferSlice {
                buffer: src,
                offset: 0,
                size: 2 * PAGE_SIZE,
            }),
            dst: CopyRegion::Buffer(BufferSlice {
                buffer: joined.buffer,
                offset: dst_offset,
                size: 2 * PAGE_SIZE,
            }),
        })
    };
    assert_eq!(commands[0], copy(a.buffer, 0));
    assert_eq!(commands[1], copy(b.buffer, 3 * PAGE_SIZE));
    // Page 1 (A's stale page, re-read once now its new watch is armed) and page 2 (the gap).
    assert_eq!(uploads(&commands), chunked(joined.buffer, PAGE_SIZE, 2));
    assert_eq!(commands.len(), 3);
    let entry = rig.entry(page(0));
    assert_eq!(entry.range, pages(0, 5));
    assert_eq!(rig.cache.len(), 1);
    assert_eq!(rig.cache.bytes(), 5 * PAGE_SIZE);
    // The gap's new watch is waiting; A's re-watch was armed and read.
    assert_eq!(entry.unarmed_bytes, PAGE_SIZE);
    assert_eq!(entry.stale_bytes, 0);
    let mut requests = Vec::new();
    rig.watch.take_requests(&mut requests);
    assert_eq!(
        requests.iter().map(|r| r.range()).collect::<Vec<_>>(),
        [pages(2, 1)]
    );
    rig.watch.arm(&requests).unwrap();
    // The open list copies from the old buffers, so they last until it is submitted.
    assert!(!rig.log().iter().any(|l| l.starts_with("destroy")));
    let fence = rig.submit();
    let log = rig.log();
    let submit = line_index(&log, &format!("submit fence {}:", fence.get()));
    assert!(submit < line_index(&log, &destroy_line(a.buffer, fence.get())));
    assert!(submit < line_index(&log, &destroy_line(b.buffer, fence.get())));
    let whole = BufferSlice {
        buffer: joined.buffer,
        offset: 0,
        size: 5 * PAGE_SIZE,
    };
    assert_eq!(
        rig.host_bytes(whole),
        rig.guest_bytes(page(0), 5 * PAGE_SIZE)
    );
}

#[test]
fn a_join_with_a_streamed_entry_is_streamed() {
    let mut rig = Rig::new();
    make_streamed(&mut rig, page(0));
    rig.bind(page(4), PAGE_SIZE);
    rig.enter();
    let (joined, ups) = rig.bind_uploads(page(0), 5 * PAGE_SIZE);
    let entry = rig.entry(page(0));
    assert!(entry.streamed);
    assert_eq!(entry.unarmed_bytes, 0);
    // Streamed: all of it, at this use and the next.
    assert_eq!(ups, chunked(joined.buffer, 0, 5));
    assert!(!rig.watch.has_requests());
    let (_, again) = rig.bind_uploads(page(0), 1);
    assert_eq!(again, chunked(joined.buffer, 0, 5));
}

#[test]
fn a_join_carries_the_fault_count_of_an_entry_written_since_its_use() {
    let mut rig = Rig::new();
    rig.bind(page(0), PAGE_SIZE);
    rig.enter();
    rig.bind(page(0), 1);
    for fault in 1..STREAMED_AFTER_FAULTS {
        rig.cpu_write(page(0), &[1]);
        rig.sync();
        rig.enter();
        // The last write is not followed by a use of this entry before the join.
        if fault + 1 < STREAMED_AFTER_FAULTS {
            rig.bind(page(0), 1);
        }
    }
    rig.bind(page(1), PAGE_SIZE);
    rig.enter();
    rig.bind(page(0), 2 * PAGE_SIZE);
    let entry = rig.entry(page(0));
    assert_eq!(entry.range, pages(0, 2));
    assert_eq!(
        (entry.faults, entry.streamed),
        (STREAMED_AFTER_FAULTS - 1, false)
    );
    rig.cpu_write(page(1), &[2]);
    rig.sync();
    assert!(rig.entry(page(0)).streamed);
}

// --- Learning of CPU writes ---

#[test]
fn an_unarmed_range_is_re_read_at_each_use() {
    let mut rig = Rig::new();
    let (slice, first) = rig.bind_uploads(page(1), 2 * PAGE_SIZE);
    assert_eq!(first, chunked(slice.buffer, 0, 2));
    for round in 0..3u8 {
        // Not protected yet, so the store faults nothing; only re-reading sees it.
        assert_eq!(
            rig.cpu_write(page(2) + u64::from(round), &[round + 1; 2]),
            0
        );
        let (_, ups) = rig.bind_uploads(page(1), 2 * PAGE_SIZE);
        assert_eq!(ups, chunked(slice.buffer, 0, 2));
        assert_eq!(
            rig.host_bytes(slice),
            rig.guest_bytes(page(1), 2 * PAGE_SIZE)
        );
    }
    assert_eq!(rig.entry(page(1)).unarmed_bytes, 2 * PAGE_SIZE);
}

#[test]
fn the_first_use_after_arming_reads_the_pages_again() {
    let mut rig = Rig::new();
    let slice = rig.bind(page(1), PAGE_SIZE);
    rig.submit();
    // A store after that read but before the protection: the watch never sees it.
    assert_eq!(rig.cpu_write(page(1) + 100, &[0xEE; 8]), 0);
    assert_eq!(rig.enter(), 1);
    let (_, ups) = rig.bind_uploads(page(1), PAGE_SIZE);
    assert_eq!(ups, chunked(slice.buffer, 0, 1));
    assert_eq!(rig.host_bytes(slice), rig.guest_bytes(page(1), PAGE_SIZE));
    assert_eq!(rig.entry(page(1)).unarmed_bytes, 0);
    // From now on it is trusted.
    let (_, ups) = rig.bind_uploads(page(1), PAGE_SIZE);
    assert_eq!(ups, []);
}

#[test]
fn a_cpu_write_re_uploads_once() {
    let mut rig = Rig::new();
    let slice = rig.bind(page(0), 4 * PAGE_SIZE);
    rig.enter();
    rig.bind(page(0), 1);
    assert_eq!(rig.bind_uploads(page(0), 1).1, []);
    assert_eq!(rig.cpu_write(page(2) + 5, &[0x11, 0x22, 0x33, 0x44]), 1);
    assert_eq!(rig.sync(), [pages(2, 1)]);
    let entry = rig.entry(page(0));
    assert_eq!(
        (entry.stale_bytes, entry.unarmed_bytes, entry.faults),
        (PAGE_SIZE, PAGE_SIZE, 1)
    );
    assert_eq!(rig.enter(), 1);
    let (_, ups) = rig.bind_uploads(page(0), 1);
    assert_eq!(ups, [(slice.buffer, 2 * PAGE_SIZE, PAGE_SIZE)]);
    assert_eq!(rig.bind_uploads(page(0), 1).1, []);
    let whole = BufferSlice {
        size: 4 * PAGE_SIZE,
        ..slice
    };
    assert_eq!(
        rig.host_bytes(whole),
        rig.guest_bytes(page(0), 4 * PAGE_SIZE)
    );
}

#[test]
fn a_rewatched_page_is_re_read_until_its_new_watch_is_armed() {
    let mut rig = Rig::new();
    let slice = rig.bind(page(0), 2 * PAGE_SIZE);
    rig.enter();
    rig.bind(page(0), 1);
    assert_eq!(rig.cpu_write(page(1), &[7]), 1);
    rig.sync();
    let page_one = [(slice.buffer, PAGE_SIZE, PAGE_SIZE)];
    // The new watch is requested but not armed: page 1 is untracked again.
    assert_eq!(rig.bind_uploads(page(0), 1).1, page_one);
    assert_eq!(rig.bind_uploads(page(0), 1).1, page_one);
    rig.enter();
    assert_eq!(rig.bind_uploads(page(0), 1).1, page_one);
    assert_eq!(rig.bind_uploads(page(0), 1).1, []);
}

#[test]
fn the_drain_hands_every_dirty_range_to_the_sink_even_with_no_entry() {
    let mut rig = Rig::new();
    // Watched by someone else on the same watch (the shader cache's tracked ranges).
    rig.watch.request(pages(40, 2));
    rig.enter();
    assert_eq!(rig.cpu_write(page(41), &[1]), 1);
    assert_eq!(rig.sync(), [pages(41, 1)]);
    assert!(rig.cache.is_empty());
    // A drain with nothing dirty hands nothing on.
    assert_eq!(rig.sync(), []);
}

#[test]
fn a_drain_counts_one_fault_per_entry_however_many_pages() {
    let mut rig = Rig::new();
    rig.bind(page(0), 4 * PAGE_SIZE);
    rig.enter();
    rig.bind(page(0), 1);
    rig.cpu_write(page(0), &[1]);
    rig.cpu_write(page(2), &[2]);
    assert_eq!(rig.sync(), [pages(0, 1), pages(2, 1)]);
    let entry = rig.entry(page(0));
    assert_eq!((entry.faults, entry.stale_bytes), (1, 2 * PAGE_SIZE));
}

/// Binds an entry of four pages at `va`, arms it, and writes it until it is streamed.
fn make_streamed(rig: &mut Rig, va: u64) {
    rig.bind(va, 4 * PAGE_SIZE);
    rig.enter();
    rig.bind(va, 1);
    for _ in 0..STREAMED_AFTER_FAULTS {
        rig.cpu_write(va, &[3]);
        rig.sync();
        rig.enter();
        rig.bind(va, 1);
    }
    assert!(rig.entry(va).streamed);
}

#[test]
fn a_range_turns_streamed_at_the_fault_threshold_not_before() {
    let mut rig = Rig::new();
    let slice = rig.bind(page(0), 4 * PAGE_SIZE);
    rig.enter();
    rig.bind(page(0), 1);
    for fault in 1..STREAMED_AFTER_FAULTS {
        assert_eq!(rig.cpu_write(page(1) + u64::from(fault), &[0x5A]), 1);
        rig.sync();
        let entry = rig.entry(page(0));
        assert_eq!((entry.faults, entry.streamed), (fault, false));
        // Still tracked: the page is watched again.
        assert!(rig.watch.has_requests());
        rig.enter();
        assert_eq!(
            rig.bind_uploads(page(0), 1).1,
            [(slice.buffer, PAGE_SIZE, PAGE_SIZE)]
        );
    }
    assert_eq!(rig.cpu_write(page(1), &[0xA5]), 1);
    rig.sync();
    let entry = rig.entry(page(0));
    assert_eq!(
        (entry.faults, entry.streamed, entry.unarmed_bytes),
        (STREAMED_AFTER_FAULTS, true, 0)
    );
    assert!(!rig.watch.has_requests());
    // Re-read whole at every use, written or not.
    for _ in 0..2 {
        assert_eq!(rig.bind_uploads(page(0), 1).1, chunked(slice.buffer, 0, 4));
    }
    let whole = BufferSlice {
        size: 4 * PAGE_SIZE,
        ..slice
    };
    assert_eq!(
        rig.host_bytes(whole),
        rig.guest_bytes(page(0), 4 * PAGE_SIZE)
    );
}

#[test]
fn a_clean_use_resets_the_fault_count() {
    let mut rig = Rig::new();
    rig.bind(page(0), PAGE_SIZE);
    rig.enter();
    rig.bind(page(0), 1);
    for _ in 1..STREAMED_AFTER_FAULTS {
        rig.cpu_write(page(0), &[1]);
        rig.sync();
        rig.enter();
        rig.bind(page(0), 1);
    }
    assert_eq!(rig.entry(page(0)).faults, STREAMED_AFTER_FAULTS - 1);
    // A use with no write since the last one.
    rig.bind(page(0), 1);
    assert_eq!(rig.entry(page(0)).faults, 0);
    rig.cpu_write(page(0), &[2]);
    rig.sync();
    let entry = rig.entry(page(0));
    assert_eq!((entry.faults, entry.streamed), (1, false));
}

// --- GPU-thread writes and unmaps ---

#[test]
fn a_gpu_thread_write_turns_its_pages_stale_without_a_fault_or_a_watch() {
    let mut rig = Rig::new();
    let slice = rig.bind(page(2), 3 * PAGE_SIZE);
    rig.enter();
    rig.bind(page(2), 1);
    // Ranges that only touch the entry, and an empty one, change nothing.
    rig.cache.invalidate(page(0)..page(2));
    rig.cache.invalidate(page(5)..page(6));
    rig.cache.invalidate(page(3)..page(3));
    assert_eq!(rig.entry(page(2)).stale_bytes, 0);
    // Bytes 10..20 of page 3 and the first byte of page 4.
    rig.mem.guest().write(page(3) + 10, &[9; 10]).unwrap();
    rig.cache.invalidate(page(3) + 10..page(4) + 1);
    let entry = rig.entry(page(2));
    assert_eq!((entry.stale_bytes, entry.faults), (2 * PAGE_SIZE, 0));
    assert_eq!(
        rig.bind_uploads(page(2), 1).1,
        chunked(slice.buffer, PAGE_SIZE, 2)
    );
    assert!(!rig.watch.has_requests());
    assert_eq!(rig.bind_uploads(page(2), 1).1, []);
    let whole = BufferSlice {
        size: 3 * PAGE_SIZE,
        ..slice
    };
    assert_eq!(
        rig.host_bytes(whole),
        rig.guest_bytes(page(2), 3 * PAGE_SIZE)
    );
}

#[test]
fn an_ordered_unmap_drops_the_range_and_keeps_the_neighbours_at_its_ends() {
    let mut rig = Rig::new();
    let a = rig.bind(page(0), 2 * PAGE_SIZE);
    let b = rig.bind(page(2), 2 * PAGE_SIZE);
    let c = rig.bind(page(4), 2 * PAGE_SIZE);
    rig.enter();
    for n in [0, 2, 4] {
        rig.bind(page(n), 1);
    }
    let fence = rig.submit();
    rig.cache
        .apply_update(&VmUpdate::Unmap { range: pages(2, 2) });
    assert!(rig.cache.entry(page(2)).is_none());
    assert_eq!(rig.cache.bytes(), 4 * PAGE_SIZE);
    line_index(&rig.log(), &destroy_line(b.buffer, fence.get()));
    // The neighbours ending at the range's start and starting at its end are unchanged.
    for (n, slice) in [(0, a), (4, c)] {
        let (again, ups) = rig.bind_uploads(page(n), 1);
        assert_eq!((again.buffer, ups), (slice.buffer, vec![]));
    }
    // The range comes back as a new buffer, read whole.
    let (fresh, ups) = rig.bind_uploads(page(2), 2 * PAGE_SIZE);
    assert_ne!(fresh.buffer, b.buffer);
    assert_eq!(ups, chunked(fresh.buffer, 0, 2));
}

#[test]
fn every_kind_of_ordered_update_drops_every_entry_it_overlaps() {
    let updates = [
        VmUpdate::Unmap { range: pages(3, 1) },
        VmUpdate::ModifyKind {
            range: pages(3, 1),
            kind: Kind(0xFE),
        },
        VmUpdate::DestroyMemory { range: pages(3, 1) },
        VmUpdate::DestroySparseRegion {
            range: pages(3, 1),
            region: RegionId(7),
        },
    ];
    for update in updates {
        let mut rig = Rig::new();
        // Page 3 is the last page of the first entry; the second starts right after it.
        rig.bind(page(0), 4 * PAGE_SIZE);
        let kept = rig.bind(page(4), PAGE_SIZE);
        rig.cache.apply_update(&update);
        let left: Vec<_> = rig.cache.entries().map(|e| (e.range, e.buffer)).collect();
        assert_eq!(left, [(pages(4, 1), kept.buffer)], "{update:?}");
    }
}

// --- Destruction ---

#[test]
fn a_buffer_dropped_while_its_list_is_open_is_destroyed_after_that_list() {
    let mut rig = Rig::new();
    let first = rig.bind(page(0), PAGE_SIZE);
    rig.submit();
    let used = rig.bind(page(8), PAGE_SIZE);
    rig.cache
        .apply_update(&VmUpdate::Unmap { range: pages(8, 1) });
    // Nothing is destroyed while the open list may still name it.
    assert!(!rig.log().iter().any(|l| l.starts_with("destroy")));
    let fence = rig.submit();
    let log = rig.log();
    let submit = line_index(&log, &format!("submit fence {}:", fence.get()));
    let destroy = line_index(&log, &destroy_line(used.buffer, fence.get()));
    assert!(submit < destroy);
    // An entry no open list uses goes after the last list that did.
    rig.bind(page(20), PAGE_SIZE);
    rig.submit();
    rig.cache
        .apply_update(&VmUpdate::Unmap { range: pages(0, 1) });
    line_index(&rig.log(), &destroy_line(first.buffer, 1));
    assert_eq!(rig.backend.refused_destroys(), 0);
}

#[test]
fn dropping_the_cache_destroys_every_buffer_after_the_last_fence() {
    let mut rig = Rig::new();
    let a = rig.bind(page(0), PAGE_SIZE);
    let b = rig.bind(page(4), PAGE_SIZE);
    let fence = rig.submit();
    let backend = Arc::clone(&rig.backend);
    drop(rig);
    let log: Vec<String> = backend.snapshot().lines().map(str::to_owned).collect();
    line_index(&log, &destroy_line(a.buffer, fence.get()));
    line_index(&log, &destroy_line(b.buffer, fence.get()));
    assert_eq!(backend.budget().used, 0);
}

// --- Errors: nothing changes ---

#[test]
fn a_bind_of_no_bytes_is_refused() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.try_bind(page(1), 0),
        Err(BufferError::Empty { va: page(1) })
    );
}

#[test]
fn the_last_page_of_the_address_space_is_out_of_range_and_the_one_before_reaches_memory() {
    let mut rig = Rig::new();
    // The first address of the last page, whose end (2^64) is not representable.
    let last = 0u64.wrapping_sub(PAGE_SIZE);
    assert_eq!(
        rig.try_bind(last, 1),
        Err(BufferError::OutOfRange { va: last, len: 1 })
    );
    // Two bytes from the byte before it reach the last page; one byte does not, and goes on to
    // read the (unmapped) page before it.
    assert_eq!(
        rig.try_bind(last - 1, 2),
        Err(BufferError::OutOfRange {
            va: last - 1,
            len: 2
        })
    );
    assert_eq!(
        rig.try_bind(last - 1, 1),
        Err(BufferError::Fault(GpuFault::Unmapped {
            va: last - PAGE_SIZE
        }))
    );
    assert_eq!(
        rig.try_bind(1, u64::MAX),
        Err(BufferError::OutOfRange {
            va: 1,
            len: u64::MAX
        })
    );
    assert!(rig.cache.is_empty());
    assert_eq!(rig.backend.budget().used, 0);
}

#[test]
fn the_largest_buffer_is_accepted_and_one_page_more_is_too_large() {
    let max = MAX_BUFFER_PAGES * PAGE_SIZE;
    let mut rig = Rig::new();
    assert_eq!(
        rig.try_bind(page(0) + 1, max),
        Err(BufferError::TooLarge {
            size: max + PAGE_SIZE,
            max
        })
    );
    let slice = rig.bind(page(0), max);
    assert_eq!(rig.entry(page(0)).range, pages(0, MAX_BUFFER_PAGES));
    assert_eq!(slice.size, max);
    // A join that would pass the largest buffer is refused too.
    assert_eq!(
        rig.try_bind(page(MAX_BUFFER_PAGES - 1), 2 * PAGE_SIZE),
        Err(BufferError::TooLarge {
            size: max + PAGE_SIZE,
            max
        })
    );
    assert_eq!(rig.cache.len(), 1);
}

#[test]
fn a_bind_past_the_budget_is_refused_at_the_exact_edge() {
    let allowed = 8 * PAGE_SIZE;
    let mut rig = Rig::with_budget(allowed);
    rig.bind(page(0), 4 * PAGE_SIZE);
    rig.bind(page(10), 4 * PAGE_SIZE);
    assert_eq!(rig.cache.bytes(), allowed);
    let list_before = rig.list.clone();
    assert_eq!(
        rig.try_bind(page(20), 1),
        Err(BufferError::OverBudget {
            bytes: allowed + PAGE_SIZE,
            allowed
        })
    );
    assert_eq!(rig.list, list_before);
    assert_eq!(rig.cache.len(), 2);
    // A join counts the bytes it frees: pages 0 to 4 replace pages 0 to 3, one page more.
    assert_eq!(
        rig.try_bind(page(3), 2 * PAGE_SIZE),
        Err(BufferError::OverBudget {
            bytes: allowed + PAGE_SIZE,
            allowed
        })
    );
    // Dropping one entry makes room for a join to exactly the budget.
    rig.cache.apply_update(&VmUpdate::Unmap {
        range: pages(10, 4),
    });
    rig.bind(page(3), 2 * PAGE_SIZE);
    rig.bind(page(10), 3 * PAGE_SIZE);
    assert_eq!(rig.cache.bytes(), allowed);
}

#[test]
fn checks_run_in_their_documented_order() {
    // Too large and over budget at once: too large.
    let mut rig = Rig::with_budget(PAGE_SIZE);
    let max = MAX_BUFFER_PAGES * PAGE_SIZE;
    assert_eq!(
        rig.try_bind(page(0), max + 1),
        Err(BufferError::TooLarge {
            size: max + PAGE_SIZE,
            max
        })
    );
    // Over budget and unmapped at once: over budget.
    let unmapped = page(PAGES);
    assert_eq!(
        rig.try_bind(unmapped, 2 * PAGE_SIZE),
        Err(BufferError::OverBudget {
            bytes: 2 * PAGE_SIZE,
            allowed: PAGE_SIZE
        })
    );
    // A staging refusal and an unmapped page at once: the backend's refusal.
    let mut rig = Rig::with(LARGE_BUDGET, 0);
    assert_eq!(
        rig.try_bind(unmapped, 1),
        Err(BufferError::Render(RenderError::OutOfRange {
            what: "staging size",
            value: 1,
            min: 1,
            max: 0
        }))
    );
    assert!(rig.cache.is_empty());
    assert_eq!(rig.backend.budget().used, 0);
}

#[test]
fn a_fault_on_a_new_entry_leaves_nothing_behind() {
    let mut rig = Rig::new();
    let a = rig.bind(page(0), PAGE_SIZE);
    let list_before = rig.list.clone();
    rig.mem.unmap(pages(2, 1)).unwrap();
    // Joins the entry at page 0 and reaches the unmapped page 2.
    let err = rig.try_bind(page(0), 3 * PAGE_SIZE);
    assert_eq!(
        err,
        Err(BufferError::Fault(GpuFault::Unmapped { va: page(2) }))
    );
    let entries: Vec<_> = rig.cache.entries().map(|e| (e.range, e.buffer)).collect();
    assert_eq!(entries, [(pages(0, 1), a.buffer)]);
    assert_eq!(rig.list, list_before);
    assert_eq!(rig.backend.budget().used, PAGE_SIZE);
    let mut requests = Vec::new();
    rig.watch.take_requests(&mut requests);
    assert_eq!(
        requests.iter().map(|r| r.range()).collect::<Vec<_>>(),
        [pages(0, 1)]
    );
    rig.watch.arm(&requests).unwrap();
    // The next attempt, with the page mapped again, works.
    rig.mem.map(pages(2, 1), page(2)).unwrap();
    let joined = rig.bind(page(0), 3 * PAGE_SIZE);
    assert_eq!(rig.entry(page(0)).range, pages(0, 3));
    assert_ne!(joined.buffer, a.buffer);
}

#[test]
fn a_fault_on_an_entry_in_use_changes_nothing_and_the_next_use_works() {
    let mut rig = Rig::new();
    let slice = rig.bind(page(0), 2 * PAGE_SIZE);
    let before = rig.entry(page(0));
    let list_before = rig.list.clone();
    rig.mem.unmap(pages(1, 1)).unwrap();
    assert_eq!(
        rig.try_bind(page(0), 1),
        Err(BufferError::Fault(GpuFault::Unmapped { va: page(1) }))
    );
    assert_eq!(rig.entry(page(0)), before);
    assert_eq!(rig.list, list_before);
    rig.mem.map(pages(1, 1), page(1)).unwrap();
    assert_eq!(rig.bind_uploads(page(0), 1).1, chunked(slice.buffer, 0, 2));
}

#[test]
fn error_messages_are_plain_words() {
    let cases = [
        (
            BufferError::Empty { va: 0x1000 },
            "a buffer binding at 0x1000 has no bytes",
        ),
        (
            BufferError::OutOfRange {
                va: 0x10,
                len: 0x20,
            },
            "a buffer binding of 0x20 bytes at 0x10 runs past the end of the address space",
        ),
        (
            BufferError::TooLarge {
                size: 0x3000,
                max: 0x2000,
            },
            "a cached buffer of 0x3000 bytes is larger than the device's largest, 0x2000",
        ),
        (
            BufferError::OverBudget {
                bytes: 0,
                allowed: 0,
            },
            "the buffer cache would hold 0x0 bytes, more than the 0x0 allowed",
        ),
        (
            BufferError::Fault(GpuFault::Unmapped { va: 0x4000 }),
            "reading guest memory for a buffer upload: GPU address 0x4000 is not mapped",
        ),
        (
            BufferError::Render(RenderError::Invalid("no")),
            "the render backend refused: no",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
}

// --- Eviction ---

#[test]
fn eviction_limits_at_every_edge() {
    let cases = [
        (0, 0, 0),
        (1, 0, 1),
        (3, 1, 3),
        (
            2 * ROUTINE_LIMIT_CAP - 1,
            ROUTINE_LIMIT_CAP - 1,
            2 * ROUTINE_LIMIT_CAP - 1,
        ),
        (
            2 * ROUTINE_LIMIT_CAP,
            ROUTINE_LIMIT_CAP,
            2 * ROUTINE_LIMIT_CAP,
        ),
        (
            AGGRESSIVE_LIMIT_CAP - 1,
            AGGRESSIVE_LIMIT_CAP / 2 - 1,
            AGGRESSIVE_LIMIT_CAP - 1,
        ),
        (
            AGGRESSIVE_LIMIT_CAP,
            AGGRESSIVE_LIMIT_CAP / 2,
            AGGRESSIVE_LIMIT_CAP,
        ),
        (
            AGGRESSIVE_LIMIT_CAP + 2,
            ROUTINE_LIMIT_CAP,
            AGGRESSIVE_LIMIT_CAP,
        ),
        (u64::MAX, ROUTINE_LIMIT_CAP, AGGRESSIVE_LIMIT_CAP),
    ];
    for (allowed, routine, aggressive) in cases {
        assert_eq!(
            eviction_limits(allowed),
            EvictionLimits {
                routine,
                aggressive
            },
            "{allowed}"
        );
    }
}

#[test]
fn eviction_caps_are_eden_nx2s_values() {
    // `buffer_bytes >= 1_GiB` and `>= 2_GiB` (eden-nx2 `buffer_cache.h`, `TickFrame`); 120 or
    // 60 ticks and 32 or 64 iterations (eden `buffer_cache.h`, `RunGarbageCollector`).
    assert_eq!(ROUTINE_LIMIT_CAP, 1_073_741_824);
    assert_eq!(AGGRESSIVE_LIMIT_CAP, 2_147_483_648);
    assert_eq!(
        (EVICT_AFTER_FRAMES, AGGRESSIVE_EVICT_AFTER_FRAMES),
        (120, 60)
    );
    assert_eq!((EVICT_PER_FRAME, AGGRESSIVE_EVICT_PER_FRAME), (32, 64));
}

#[test]
fn collection_at_every_edge() {
    let limits = EvictionLimits {
        routine: 100,
        aggressive: 200,
    };
    let routine = Some(Collection {
        after_frames: EVICT_AFTER_FRAMES,
        at_most: EVICT_PER_FRAME,
    });
    let aggressive = Some(Collection {
        after_frames: AGGRESSIVE_EVICT_AFTER_FRAMES,
        at_most: AGGRESSIVE_EVICT_PER_FRAME,
    });
    for (bytes, expected) in [
        (0, None),
        (99, None),
        (100, routine),
        (101, routine),
        (199, routine),
        (200, aggressive),
        (201, aggressive),
    ] {
        assert_eq!(collection(bytes, limits), expected, "{bytes}");
    }
}

/// Ends frames until `frames` have ended in all, asserting none evicts anything, then ends one
/// more and returns what it evicted.
fn frames_then_one(rig: &mut Rig, frames: u64) -> usize {
    for _ in 0..frames {
        assert_eq!(rig.cache.end_frame(), 0);
    }
    rig.cache.end_frame()
}

#[test]
fn eviction_at_the_budget_edge_on_both_sides() {
    let allowed = 16 * PAGE_SIZE;
    let limits = eviction_limits(allowed);
    let routine_pages = limits.routine / PAGE_SIZE;
    let aggressive_pages = limits.aggressive / PAGE_SIZE;

    // At the routine limit: evicted once it is EVICT_AFTER_FRAMES frames old, not before.
    let mut rig = Rig::with_budget(allowed);
    rig.bind(page(0), routine_pages * PAGE_SIZE);
    rig.submit();
    assert_eq!(frames_then_one(&mut rig, EVICT_AFTER_FRAMES), 1);
    assert!(rig.cache.is_empty());

    // One page below it: never.
    let mut rig = Rig::with_budget(allowed);
    rig.bind(page(0), (routine_pages - 1) * PAGE_SIZE);
    rig.submit();
    assert_eq!(frames_then_one(&mut rig, 4 * EVICT_AFTER_FRAMES), 0);

    // At the aggressive limit: after AGGRESSIVE_EVICT_AFTER_FRAMES.
    let mut rig = Rig::with_budget(allowed);
    rig.bind(page(0), aggressive_pages * PAGE_SIZE);
    rig.submit();
    assert_eq!(frames_then_one(&mut rig, AGGRESSIVE_EVICT_AFTER_FRAMES), 1);

    // One page below it: routine, after EVICT_AFTER_FRAMES.
    let mut rig = Rig::with_budget(allowed);
    rig.bind(page(0), (aggressive_pages - 1) * PAGE_SIZE);
    rig.submit();
    assert_eq!(frames_then_one(&mut rig, EVICT_AFTER_FRAMES), 1);
}

#[test]
fn a_use_restarts_an_entrys_age() {
    let allowed = 2 * PAGE_SIZE;
    let mut rig = Rig::with_budget(allowed);
    rig.bind(page(0), PAGE_SIZE);
    rig.submit();
    for _ in 0..EVICT_AFTER_FRAMES {
        rig.cache.end_frame();
    }
    rig.bind(page(0), 1);
    rig.submit();
    assert_eq!(frames_then_one(&mut rig, EVICT_AFTER_FRAMES), 1);
}

#[test]
fn collection_takes_the_least_recently_used_first_up_to_its_count() {
    // Routine: one entry more than a frame's count, bytes at the routine limit.
    let count = u64::try_from(EVICT_PER_FRAME).unwrap() + 1;
    let mut rig = Rig::with_budget(2 * count * PAGE_SIZE);
    for n in 0..count {
        rig.bind(page(n), PAGE_SIZE);
    }
    // Page 0 is used again last: least recently used is page 1, not the lowest address.
    rig.bind(page(0), 1);
    rig.submit();
    assert_eq!(
        frames_then_one(&mut rig, EVICT_AFTER_FRAMES),
        EVICT_PER_FRAME
    );
    let left: Vec<_> = rig.cache.entries().map(|e| e.range).collect();
    assert_eq!(left, [pages(0, 1)]);
    // What is left is below the limit.
    assert_eq!(rig.cache.end_frame(), 0);

    // Aggressive: one more than its count, bytes at the aggressive limit.
    let count = u64::try_from(AGGRESSIVE_EVICT_PER_FRAME).unwrap() + 1;
    let mut rig = Rig::with_budget(count * PAGE_SIZE);
    for n in 0..count {
        rig.bind(page(n), PAGE_SIZE);
    }
    rig.submit();
    assert_eq!(
        frames_then_one(&mut rig, AGGRESSIVE_EVICT_AFTER_FRAMES),
        AGGRESSIVE_EVICT_PER_FRAME
    );
    let left: Vec<_> = rig.cache.entries().map(|e| e.range).collect();
    assert_eq!(left, [pages(count - 1, 1)]);
}

#[test]
fn an_entry_the_open_list_uses_is_not_evicted() {
    let mut rig = Rig::with_budget(2 * PAGE_SIZE);
    let slice = rig.bind(page(0), PAGE_SIZE);
    for _ in 0..4 * EVICT_AFTER_FRAMES {
        assert_eq!(rig.cache.end_frame(), 0);
    }
    let fence = rig.submit();
    assert_eq!(rig.cache.end_frame(), 1);
    line_index(&rig.log(), &destroy_line(slice.buffer, fence.get()));
}

#[test]
fn eviction_destroys_after_the_last_list_that_used_the_entry() {
    let mut rig = Rig::with_budget(2 * PAGE_SIZE);
    let slice = rig.bind(page(0), PAGE_SIZE);
    let used = rig.submit();
    // Later lists that do not use it.
    rig.list
        .push(Command::Marker(starquake_render::command::Marker {
            submit: 1,
            draw: 0,
        }));
    rig.submit();
    assert_eq!(frames_then_one(&mut rig, EVICT_AFTER_FRAMES), 1);
    line_index(&rig.log(), &destroy_line(slice.buffer, used.get()));
}

#[test]
fn a_new_cache_holds_nothing() {
    let mut rig = Rig::new();
    assert_eq!(rig.cache.bytes(), 0);
    assert_eq!(rig.cache.len(), 0);
    assert!(rig.cache.is_empty());
    assert_eq!(rig.cache.entries().count(), 0);
    assert!(rig.cache.entry(page(0)).is_none());
    assert_eq!(rig.cache.end_frame(), 0);
    assert_eq!(rig.sync(), []);
    assert_eq!(rig.backend.snapshot(), "");
}

#[test]
fn turning_streamed_forgets_the_watches_it_waited_for() {
    let mut rig = Rig::new();
    rig.bind(page(0), 4 * PAGE_SIZE);
    rig.enter();
    rig.bind(page(0), 1);
    for _ in 2..STREAMED_AFTER_FAULTS {
        rig.cpu_write(page(0), &[1]);
        rig.sync();
        rig.enter();
        rig.bind(page(0), 1);
    }
    // Page 1 is written and watched again, and that watch is not armed yet...
    assert_eq!(rig.cpu_write(page(1), &[1]), 1);
    rig.sync();
    let entry = rig.entry(page(0));
    assert_eq!(
        (entry.faults, entry.streamed, entry.unarmed_bytes),
        (STREAMED_AFTER_FAULTS - 1, false, PAGE_SIZE)
    );
    // ...when a write to page 2 reaches the threshold.
    assert_eq!(rig.cpu_write(page(2), &[2]), 1);
    rig.sync();
    let entry = rig.entry(page(0));
    assert_eq!((entry.streamed, entry.unarmed_bytes), (true, 0));
}

#[test]
fn dropping_the_cache_destroys_a_buffer_its_open_list_had_dropped() {
    let mut rig = Rig::new();
    rig.bind(page(0), PAGE_SIZE);
    let fence = rig.submit();
    let doomed = rig.bind(page(8), PAGE_SIZE);
    rig.cache
        .apply_update(&VmUpdate::Unmap { range: pages(8, 1) });
    let backend = Arc::clone(&rig.backend);
    drop(rig);
    let log: Vec<String> = backend.snapshot().lines().map(str::to_owned).collect();
    line_index(&log, &destroy_line(doomed.buffer, fence.get()));
    assert_eq!(backend.budget().used, 0);
}
