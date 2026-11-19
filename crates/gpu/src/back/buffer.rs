//! The back thread's buffer cache (gpu-protocol spec, "Caches", "Learning of CPU writes", "GPU
//! writes and the CPU" and "Ordered address-table updates and barriers" step 4; slice G16;
//! Eden-NX2 notes rows 120, 131, 139 and 141).
//!
//! [`BufferCache`] maps guest GPU ranges to host buffers made through the render backend
//! ([`Backend::create_buffer`]) and filled with [`Command::Upload`]s in the command list the
//! back thread is building. The cache lives here and the host objects in the backend, which
//! only ever sees [`BufferId`]s (decision log 2026-11-12, "Draw commands cross to the backend as
//! plain data; caches are split by what they are keyed on").
//!
//! One cache serves one GPU address space, and is handed that space's process's
//! [`WriteWatch`]. The executor (slice G19) owns it and calls it on the back thread.
//!
//! # Entries
//!
//! An entry is a page-aligned GPU range ([`PageRange`], pages of
//! [`starquake_memory::PAGE_SIZE`]) and its host buffer, of the same size, made with every buffer
//! usage ([`BUFFER_USAGE`]). Entries never overlap. [`BufferCache::bind`] returns the
//! [`BufferSlice`] of the entry holding the requested bytes:
//!
//! - when one entry holds every page the bytes touch, that entry;
//! - otherwise a new entry covering the requested pages and every entry they overlap (entries
//!   that only touch are left alone). The new buffer is filled by a [`Command::Copy`] of each
//!   old buffer into its place, then uploads of the pages no old entry held; the old entries'
//!   page state (stale, waiting for their watch, streamed) carries over, and their buffers are
//!   destroyed ("Destruction" below). This mirrors Eden's `CreateBuffer`, `ResolveOverlaps` and
//!   `JoinOverlap` (`eden/src/video_core/buffer_cache/buffer_cache.h`), without its stream leap
//!   (an entry grown by 128 caching pages once its joins pass `STREAM_LEAP_THRESHOLD`) and with
//!   4 KiB pages instead of its 64 KiB `CACHING_PAGEBITS = 16`, so an entry is exactly the pages
//!   the write watch speaks in.
//!
//! Each use uploads the pages of the entry whose host copy may be older than guest memory, and
//! nothing else. Before any upload is read, [`BufferCache::sync_cpu_writes`] should have run
//! (the next section).
//!
//! # Learning of CPU writes
//!
//! A new entry's pages are watched: the cache asks [`WriteWatch::request`] to write-protect them,
//! and until [`WriteWatch::armed`] says the request took effect, those pages are *untracked*: a
//! guest store to them is not reported, so every use reads them again. Once armed, a guest store
//! to a page reaches the watch's dirty log, which [`BufferCache::sync_cpu_writes`] drains
//! ([`WriteWatch::take_dirty`]) before each draw, dispatch and copy. For each dirty range it:
//!
//! 1. hands the range to the [`InvalidationSink`], whatever the cache holds (the executor
//!    connects it to the shader cache's `invalidate`, which is not built yet; the spec has the
//!    drain call it, gpu-protocol "Learning of CPU writes");
//! 2. marks the range's pages stale in every entry that holds them, so the next use uploads them
//!    once;
//! 3. for each such entry that is not streamed, counts a fault against it, and either asks for
//!    the pages to be watched again (a page faults once per tracking period, and tracking it
//!    again takes a new request) or, at [`STREAMED_AFTER_FAULTS`], marks the entry *streamed*
//!    ("Streamed ranges"), forgetting the watches it was waiting for.
//!
//! **The first use after arming reads the pages once more.** A page read while its watch was
//! not yet armed may have been written between that read and the arming, and that write is not
//! in the dirty log; so the use that first sees the watch armed reads the page again (after
//! asking [`WriteWatch::armed`], so the read comes after the protection), and only later uses
//! trust it. This is the proposal "G16/G17 caches re-read a range on first use after arming"
//! (`docs/proposals.md`, review notes on #197), which the owner has not vetted: it is this
//! module's reading of the spec's "treats the range as untracked" (bytes read while untracked are
//! untracked bytes), and an **owner question** raised in the pull request. Pinned by
//! `tests::the_first_use_after_arming_reads_the_pages_again`.
//!
//! The watch speaks in guest CPU addresses and the cache in GPU addresses; the cache uses one as
//! the other. That holds for memory registered with the graphics session, which Switch 2 uses at
//! GPU address = CPU address (row 30 of `eden-nx2/NX2_NOTES.md`). For a GPU range whose pages
//! lead elsewhere (a `Target::Guest` at another CPU address) the watch protects the wrong CPU
//! pages: guest stores to the real ones are missed, so the cache keeps stale bytes, and stores
//! to the others cost spurious re-uploads. GPU-only memory has no CPU pages behind it, so only
//! the second applies. This is a **known gap** until the production memory (slice G22) can
//! translate a range to its CPU pages.
//!
//! # Streamed ranges
//!
//! A range the guest rewrites constantly (streamed uniform data, row 141) costs one fault per
//! page per tracking period, which buys nothing. An entry's faults are counted per drain that
//! finds it written (one per drain however many of its pages were written), and the count goes
//! back to zero at a use that finds no write since the previous use. When the count reaches
//! [`STREAMED_AFTER_FAULTS`], the entry is streamed: it asks for no more watches and uploads all
//! of itself at every use. It stays streamed until it is dropped, and an entry joined from it is
//! streamed too (`tests::a_join_with_a_streamed_entry_is_streamed`). Watches already armed on its pages are not cancelled
//! ([`WriteWatch`] has no cancel); their faults still reach the sink and are otherwise ignored.
//! Pinned by `tests::a_range_turns_streamed_at_the_fault_threshold_not_before` and
//! `tests::a_clean_use_resets_the_fault_count`.
//!
//! # GPU-thread writes and unmaps
//!
//! [`BufferCache::invalidate`] takes a range the back thread itself wrote into guest memory (a
//! DMA copy, an inline upload or a 2D blit on the CPU path, `back::copy`'s `CopySink::invalidate`):
//! its pages turn stale in every entry holding them. Their watches stay as they were: the write
//! went through the host's access, not a guest store, so it faulted nothing.
//!
//! [`BufferCache::apply_update`] takes an ordered address-table update once it is applied (step
//! 4 of "Ordered address-table updates and barriers"; the fence path, slice G15, applies them):
//! every entry overlapping the update's range is dropped whole, and an entry starting exactly at
//! the range's end, or ending exactly at its start, is kept. All four kinds of [`VmUpdate`] drop
//! the range, `ModifyKind` included: a kind change does not change the bytes a buffer reads, so
//! dropping costs a re-upload only, and keeps the cache from deciding what a kind means. Eden's
//! cache is keyed by device address, so a GPU unmap there only forgets the GPU range's
//! translation (`UnmapGPUMemory`, `buffer_cache.h`); this one is keyed by GPU address, as the
//! spec has it. Calling the shader cache for the update's range is step 4's own action, not the
//! buffer cache's.
//!
//! # Destruction
//!
//! A dropped, joined or evicted entry's buffer is destroyed with [`Backend::destroy`] after the
//! fence of the last list that used it, so it outlives every list that names it. The cache learns
//! a list's fence only when the executor reports its submission ([`BufferCache::submitted`]): an
//! entry used since the last report belongs to the open list, and its buffer's destroy waits for
//! that report. The buffers a join replaces are always in that case: the open list copies from
//! them. Pinned by `tests::a_buffer_dropped_while_its_list_is_open_is_destroyed_after_that_list`
//! and `tests::a_bind_overlapping_entries_joins_them_by_copy_and_uploads_only_the_gap`.
//! Dropping the cache destroys every buffer it holds after the last fence it was told of, so it
//! must be dropped after its open list is submitted or abandoned.
//!
//! # Eviction
//!
//! [`BufferCache::end_frame`] (once per presented frame) runs the collector when the cache's own
//! bytes reach the routine limit [`eviction_limits`] derives from [`Backend::budget`]: entries
//! not used for [`EVICT_AFTER_FRAMES`] frames are destroyed, least recently used first, at most
//! [`EVICT_PER_FRAME`] a frame; at the aggressive limit, after [`AGGRESSIVE_EVICT_AFTER_FRAMES`]
//! frames and up to [`AGGRESSIVE_EVICT_PER_FRAME`]. The ages and counts are Eden's
//! `RunGarbageCollector` (`eden/src/video_core/buffer_cache/buffer_cache.h`: 120 or 60 ticks, 32
//! or 64 iterations; it walks its LRU list from the least recently touched and destroys those at
//! least that many frames old, before `++frame_tick`). Collection is sized by the cache's own
//! bytes, not the device's use, as Eden-NX2 learned (row 139; `buffer_cache.h` in `eden-nx2`
//! collects at `buffer_bytes >= 1_GiB`, aggressively at `2_GiB`).
//!
//! # Divergences and guesses
//!
//! - **Limits from the budget (guess).** [`eviction_limits`] takes Eden-NX2's 1 GiB and 2 GiB and
//!   caps them at half the budget's `allowed` and the whole of it, so a small device still
//!   collects. The halves are not from a reference. Pinned by
//!   `tests::eviction_limits_at_every_edge`.
//! - **A hard cap (stricter than both references).** A bind that would take the cache's bytes past
//!   the budget's `allowed` is refused ([`BufferError::OverBudget`]) before anything changes;
//!   Eden has no cap, so a hostile guest binding fresh ranges faster than the 120-frame collector
//!   frees them could grow it without bound. Pinned by
//!   `tests::a_bind_past_the_budget_is_refused_at_the_exact_edge`.
//! - **GPU-written buffers.** Nothing writes a buffer on the host GPU yet: the copy engines write
//!   guest memory on the CPU, and the first host-GPU writer is a storage buffer in a draw or
//!   dispatch (slices G18 and G19). So this slice has no GPU-written state, no on-demand
//!   write-back (decision 4: copied back only when something reads them as plain memory, buffers
//!   too while wgpu cannot alias) and no exclusion of GPU-written buffers from routine eviction
//!   (row 139); they arrive with their first writer. Until then, every entry is evictable.
//! - **Streaming threshold (guess).** [`STREAMED_AFTER_FAULTS`] has no reference: Eden's stream
//!   heuristics count joins (`STREAM_LEAP_THRESHOLD = 16`) and uniform-buffer hit ratios, not
//!   write faults. A trace of row 141's uniform streams would size it.

mod spans;

#[cfg(test)]
mod tests;

use core::ops::Range;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use starquake_memory::{PageRange, WatchTicket, WriteWatch};
use starquake_render::command::{BufferSlice, CopyDesc, CopyRegion};
use starquake_render::{
    Backend, BufferDesc, BufferId, BufferUsage, Command, CommandList, FenceValue, RenderError,
    ResourceId,
};
use thiserror::Error;

use crate::interface::VmUpdate;
use crate::memory::{GpuFault, GpuMemory};

use spans::Spans;

/// The usage every cached buffer is made with: all of them, as Eden makes every cached buffer
/// usable as transfer source and destination, uniform, storage, index, vertex and indirect
/// buffer (`MakeBufferCreateInfo`, `eden/src/video_core/renderer_vulkan/vk_buffer_cache.cpp`),
/// since one guest range may be bound in any role.
pub const BUFFER_USAGE: BufferUsage = BufferUsage::ALL;

/// Drains finding an entry written, without a clean use between them, after which the entry is
/// streamed (module docs, "Streamed ranges"). A **guess**; see the module docs.
pub const STREAMED_AFTER_FAULTS: u32 = 4;

/// Frames an entry must go unused before routine collection destroys it (Eden's
/// `RunGarbageCollector`, `ticks_to_destroy` when not aggressive).
pub const EVICT_AFTER_FRAMES: u64 = 120;

/// Frames an entry must go unused before aggressive collection destroys it (Eden's
/// `ticks_to_destroy` when aggressive).
pub const AGGRESSIVE_EVICT_AFTER_FRAMES: u64 = 60;

/// Most entries routine collection destroys in one frame (Eden's `num_iterations`).
pub const EVICT_PER_FRAME: usize = 32;

/// Most entries aggressive collection destroys in one frame (Eden's `num_iterations` when
/// aggressive).
pub const AGGRESSIVE_EVICT_PER_FRAME: usize = 64;

/// The cache's bytes at which routine collection starts, on a device whose budget is at least
/// twice this (Eden-NX2's `buffer_bytes >= 1_GiB`).
pub const ROUTINE_LIMIT_CAP: u64 = 1 << 30;

/// The cache's bytes at which collection turns aggressive, on a device whose budget is at least
/// this (Eden-NX2's `buffer_bytes >= 2_GiB`).
pub const AGGRESSIVE_LIMIT_CAP: u64 = 2 << 30;

/// The cache sizes at which its collector runs, from the device's budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvictionLimits {
    /// At or above this many bytes, routine collection runs.
    pub routine: u64,
    /// At or above this many bytes, collection is aggressive.
    pub aggressive: u64,
}

/// The collector's limits for a device that allows `allowed` bytes: [`ROUTINE_LIMIT_CAP`] or half
/// the budget, whichever is less, and [`AGGRESSIVE_LIMIT_CAP`] or the whole budget, whichever is
/// less (module docs, "Divergences and guesses": the halves are a guess).
pub const fn eviction_limits(allowed: u64) -> EvictionLimits {
    let half = allowed / 2;
    EvictionLimits {
        routine: if half < ROUTINE_LIMIT_CAP {
            half
        } else {
            ROUTINE_LIMIT_CAP
        },
        aggressive: if allowed < AGGRESSIVE_LIMIT_CAP {
            allowed
        } else {
            AGGRESSIVE_LIMIT_CAP
        },
    }
}

/// How a collection runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Collection {
    /// Frames an entry must have gone unused to be destroyed.
    pub after_frames: u64,
    /// Most entries destroyed.
    pub at_most: usize,
}

/// Whether a cache of `bytes` collects under `limits`, and how: `None` below the routine limit,
/// routine from it, aggressive from the aggressive limit (both inclusive, as Eden-NX2's `>=`).
pub const fn collection(bytes: u64, limits: EvictionLimits) -> Option<Collection> {
    if bytes >= limits.aggressive {
        Some(Collection {
            after_frames: AGGRESSIVE_EVICT_AFTER_FRAMES,
            at_most: AGGRESSIVE_EVICT_PER_FRAME,
        })
    } else if bytes >= limits.routine {
        Some(Collection {
            after_frames: EVICT_AFTER_FRAMES,
            at_most: EVICT_PER_FRAME,
        })
    } else {
        None
    }
}

/// Why [`BufferCache::bind`] refused.
///
/// `Empty`, `OutOfRange`, `TooLarge` and `OverBudget` are checked in that order, before anything
/// else (the last two only when the bind makes a new entry). Then a new entry's buffer is made
/// (`Render`), and the uploads are prepared piece by piece in address order, each asking for its
/// staging slice (`Render`) before reading its bytes (`Fault`), so the first piece that fails
/// decides between the two. Pinned by `tests::checks_run_in_their_documented_order`.
///
/// Nothing changes on any of them: no entry, page state, watch request, live buffer or command is
/// left behind (a buffer made for the bind is destroyed at once, being in no list).
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BufferError {
    /// A binding of no bytes.
    #[error("a buffer binding at {va:#x} has no bytes")]
    Empty {
        /// Where it starts.
        va: u64,
    },
    /// The binding's last page ends past the 64-bit address space.
    #[error("a buffer binding of {len:#x} bytes at {va:#x} runs past the end of the address space")]
    OutOfRange {
        /// Where it starts.
        va: u64,
        /// Its length.
        len: u64,
    },
    /// The entry that would hold the binding (its pages, joined with every entry they overlap)
    /// is larger than the device's largest buffer.
    #[error("a cached buffer of {size:#x} bytes is larger than the device's largest, {max:#x}")]
    TooLarge {
        /// The entry's size.
        size: u64,
        /// `Limits::max_buffer_size`.
        max: u64,
    },
    /// The new entry would take the cache's bytes past what the device allows.
    #[error("the buffer cache would hold {bytes:#x} bytes, more than the {allowed:#x} allowed")]
    OverBudget {
        /// The cache's bytes with the new entry.
        bytes: u64,
        /// The budget's `allowed`.
        allowed: u64,
    },
    /// The backend refused a buffer or a staging slice.
    #[error("the render backend refused: {0}")]
    Render(RenderError),
    /// Reading the guest bytes to upload faulted.
    #[error("reading guest memory for a buffer upload: {0}")]
    Fault(GpuFault),
}

/// Where the dirty drain sends each range of CPU writes ([`BufferCache::sync_cpu_writes`]). The
/// executor connects it to the shader cache's `invalidate` (shader-pipeline slice 19).
pub trait InvalidationSink {
    /// The guest CPU wrote somewhere in `range` since the last drain.
    fn invalidate(&mut self, range: PageRange);
}

/// What the cache holds for one entry, for tests, logs and counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferEntry {
    /// The GPU range.
    pub range: PageRange,
    /// Its host buffer.
    pub buffer: BufferId,
    /// Whether it re-reads all of itself at every use instead of being watched.
    pub streamed: bool,
    /// Bytes whose watch has not been seen armed: re-read at every use, and once more at the
    /// use that first sees it armed.
    pub unarmed_bytes: u64,
    /// Bytes the next use uploads because guest memory changed (or they were never read).
    pub stale_bytes: u64,
    /// Drains that found it written since its last clean use.
    pub faults: u32,
}

/// One cached range.
#[derive(Debug)]
struct Entry {
    range: PageRange,
    buffer: BufferId,
    streamed: bool,
    /// Pages whose watch request has not been seen armed.
    unarmed: Spans,
    /// The newest watch request covering `unarmed`; once it is armed, every earlier one is too
    /// ([`WriteWatch::armed`]).
    pending: Option<WatchTicket>,
    /// Pages whose host copy may be older than guest memory.
    stale: Spans,
    faults: u32,
    /// Whether a drain found it written since its last use.
    written_since_use: bool,
    /// The fence of the last submitted list that used it.
    last_fence: FenceValue,
    /// Whether the open (not yet submitted) list uses it.
    in_open: bool,
    /// The cache's use counter at its last use: lower is less recently used.
    stamp: u64,
    /// The frame of its last use.
    frame: u64,
}

impl Entry {
    /// The pages this use uploads: all of a streamed entry, else the stale and unarmed pages.
    fn upload_spans(&self) -> Spans {
        let mut spans = Spans::default();
        if self.streamed {
            spans.insert(self.range.start()..self.range.end());
        } else {
            spans.insert_all(&self.stale);
            spans.insert_all(&self.unarmed);
        }
        spans
    }

    fn info(&self) -> BufferEntry {
        BufferEntry {
            range: self.range,
            buffer: self.buffer,
            streamed: self.streamed,
            unarmed_bytes: self.unarmed.len(),
            stale_bytes: self.stale.len(),
            faults: self.faults,
        }
    }
}

/// The buffer cache of one GPU address space (see the module docs).
pub struct BufferCache {
    backend: Arc<dyn Backend>,
    watch: Arc<WriteWatch>,
    /// Entries by start address; they never overlap.
    entries: BTreeMap<u64, Entry>,
    /// The sum of the entries' sizes.
    bytes: u64,
    /// Starts of entries the open list uses (an entry may have been replaced since).
    open: Vec<u64>,
    /// Buffers of entries the open list used, dropped before it was submitted.
    doomed: Vec<BufferId>,
    /// The fence of the last submitted list the cache was told of.
    last_fence: FenceValue,
    /// The use counter.
    stamp: u64,
    /// Frames ended so far.
    frame: u64,
    /// Scratch for the dirty drain.
    dirty: Vec<PageRange>,
}

impl core::fmt::Debug for BufferCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BufferCache")
            .field("entries", &self.entries)
            .field("bytes", &self.bytes)
            .field("frame", &self.frame)
            .finish_non_exhaustive()
    }
}

/// A use's uploads, prepared before anything changes.
struct Staged {
    commands: Vec<Command>,
}

/// A checked join, before anything changes ([`BufferCache::plan_join`]).
struct JoinPlan {
    /// Starts of the entries joined, in address order.
    olds: Vec<u64>,
    /// The joined entry's range.
    range: PageRange,
    /// The cache's bytes once the join is done.
    bytes: u64,
    /// The old entries' stale pages and the pages none held.
    stale: Spans,
    /// The old entries' unarmed pages.
    unarmed: Spans,
    /// The pages no old entry held.
    uncovered: Spans,
    /// The newest of the old entries' pending tickets.
    pending: Option<WatchTicket>,
    /// Whether any old entry was streamed.
    streamed: bool,
    /// The most faults of any old entry.
    faults: u32,
    /// Whether any old entry was written since its last use.
    written_since_use: bool,
}

impl BufferCache {
    /// An empty cache making buffers through `backend` and watching through `watch`.
    pub fn new(backend: Arc<dyn Backend>, watch: Arc<WriteWatch>) -> Self {
        Self {
            backend,
            watch,
            entries: BTreeMap::new(),
            bytes: 0,
            open: Vec::new(),
            doomed: Vec::new(),
            last_fence: FenceValue::ZERO,
            stamp: 0,
            frame: 0,
            dirty: Vec::new(),
        }
    }

    /// The sum of the cached entries' sizes, in bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// How many entries the cache holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no entry.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry holding GPU address `va`, if any.
    pub fn entry(&self, va: u64) -> Option<BufferEntry> {
        self.entries
            .range(..=va)
            .next_back()
            .filter(|(_, e)| e.range.contains(va))
            .map(|(_, e)| e.info())
    }

    /// Every entry, in address order.
    pub fn entries(&self) -> impl Iterator<Item = BufferEntry> + '_ {
        self.entries.values().map(Entry::info)
    }

    /// The host buffer range holding the `len` bytes at GPU address `va`, with the uploads it
    /// needs appended to `list` (module docs, "Entries"). `list` is the open list, submitted
    /// later with [`BufferCache::submitted`] reported.
    ///
    /// On an error nothing changes: `list`, the entries, the watch and the backend's live
    /// buffers are as they were ([`BufferError`] gives the order of checks).
    pub fn bind<M: GpuMemory + ?Sized>(
        &mut self,
        mem: &M,
        va: u64,
        len: u64,
        list: &mut CommandList,
    ) -> Result<BufferSlice, BufferError> {
        if len == 0 {
            return Err(BufferError::Empty { va });
        }
        let cover =
            PageRange::covering(va, len).map_err(|_| BufferError::OutOfRange { va, len })?;
        let (start, buffer) = match self.containing(cover) {
            Some(start) => self.use_entry(mem, start, cover, list)?,
            None => self.join(mem, cover, list)?,
        };
        Ok(BufferSlice {
            buffer,
            offset: va - start,
            size: len,
        })
    }

    /// The start of the entry holding all of `cover`, if one does.
    fn containing(&self, cover: PageRange) -> Option<u64> {
        self.entries
            .range(..=cover.start())
            .next_back()
            .filter(|(_, e)| e.range.end() >= cover.end())
            .map(|(&s, _)| s)
    }

    /// Uses the entry at `start`, which holds `cover`: uploads what it needs, then records the
    /// use. Returns the entry's start and buffer.
    fn use_entry<M: GpuMemory + ?Sized>(
        &mut self,
        mem: &M,
        start: u64,
        cover: PageRange,
        list: &mut CommandList,
    ) -> Result<(u64, BufferId), BufferError> {
        let Some(entry) = self.entries.get(&start) else {
            return self.join(mem, cover, list);
        };
        let buffer = entry.buffer;
        // Asked before the guest bytes are read, so a read after `true` comes after the
        // protection took effect.
        let armed_now = entry.pending.is_some_and(|t| self.watch.armed(t));
        let mut staged = Staged {
            commands: Vec::new(),
        };
        self.stage(mem, buffer, start, &entry.upload_spans(), &mut staged)?;

        // Nothing below can fail.
        let stamp = self.next_stamp();
        let frame = self.frame;
        if let Some(entry) = self.entries.get_mut(&start) {
            entry.stale.clear();
            if armed_now {
                entry.unarmed.clear();
                entry.pending = None;
            }
            Self::record_use(entry, stamp, frame, &mut self.open);
        }
        list.commands.append(&mut staged.commands);
        Ok((start, buffer))
    }

    /// Makes the entry covering `cover` and every entry it overlaps, and uses it.
    fn join<M: GpuMemory + ?Sized>(
        &mut self,
        mem: &M,
        cover: PageRange,
        list: &mut CommandList,
    ) -> Result<(u64, BufferId), BufferError> {
        let plan = self.plan_join(cover)?;
        let start = plan.range.start();
        let armed_now = plan.pending.is_some_and(|t| self.watch.armed(t));
        let buffer = self
            .backend
            .create_buffer(&BufferDesc {
                size: plan.range.size(),
                usage: BUFFER_USAGE,
            })
            .map_err(BufferError::Render)?;
        let mut staged = Staged {
            commands: self.join_copies(&plan.olds, buffer, start),
        };
        let mut uploads = Spans::default();
        if plan.streamed {
            uploads.insert(start..plan.range.end());
        } else {
            uploads.insert_all(&plan.stale);
            uploads.insert_all(&plan.unarmed);
        }
        if let Err(error) = self.stage(mem, buffer, start, &uploads, &mut staged) {
            // The buffer is in no list: it can go at once.
            self.backend
                .destroy(ResourceId::Buffer(buffer), FenceValue::ZERO);
            return Err(error);
        }

        // Nothing below can fail.
        let JoinPlan {
            olds,
            range,
            bytes,
            mut unarmed,
            uncovered,
            mut pending,
            streamed,
            faults,
            written_since_use,
            ..
        } = plan;
        if armed_now || streamed {
            unarmed.clear();
            pending = None;
        }
        if !streamed {
            for gap in uncovered.iter() {
                if let Ok(pages) = PageRange::new(gap.start, gap.end - gap.start) {
                    pending = Some(self.watch.request(pages));
                    unarmed.insert(gap);
                }
            }
        }
        // The open list now reads the old buffers (the copies above), so each is destroyed
        // after that list, whether or not it used them before.
        for s in &olds {
            if let Some(e) = self.entries.get_mut(s) {
                e.in_open = true;
            }
            self.retire(*s);
        }
        let stamp = self.next_stamp();
        let mut entry = Entry {
            range,
            buffer,
            streamed,
            unarmed,
            pending,
            stale: Spans::default(),
            faults,
            written_since_use,
            last_fence: self.last_fence,
            in_open: false,
            stamp,
            frame: self.frame,
        };
        Self::record_use(&mut entry, stamp, self.frame, &mut self.open);
        self.entries.insert(start, entry);
        self.bytes = bytes;
        list.commands.append(&mut staged.commands);
        Ok((start, buffer))
    }

    /// Checks a join of `cover` with the entries it overlaps and works out the joined entry's
    /// page state: the old entries' carried over, and the pages none held stale (never read)
    /// and waiting for a new watch. Changes nothing.
    fn plan_join(&self, cover: PageRange) -> Result<JoinPlan, BufferError> {
        // Entries never overlap, so those between the first and last lie inside `cover`.
        let olds = self.overlapping(cover.start()..cover.end());
        let old = |s: &u64| self.entries.get(s);
        let start = olds
            .first()
            .and_then(old)
            .map_or(cover.start(), |e| e.range.start().min(cover.start()));
        let end = olds
            .last()
            .and_then(old)
            .map_or(cover.end(), |e| e.range.end().max(cover.end()));
        let size = end - start;
        let max = self.backend.caps().limits.max_buffer_size;
        if size > max {
            return Err(BufferError::TooLarge { size, max });
        }
        let old_bytes = olds
            .iter()
            .filter_map(old)
            .map(|e| e.range.size())
            .fold(0, u64::saturating_add);
        let allowed = self.backend.budget().allowed;
        let bytes = self.bytes.saturating_sub(old_bytes).saturating_add(size);
        if bytes > allowed {
            return Err(BufferError::OverBudget { bytes, allowed });
        }
        let range = PageRange::new(start, size).map_err(|_| BufferError::OutOfRange {
            va: cover.start(),
            len: cover.size(),
        })?;
        let mut plan = JoinPlan {
            olds: Vec::new(),
            range,
            bytes,
            stale: Spans::default(),
            unarmed: Spans::default(),
            uncovered: Spans::default(),
            pending: None,
            streamed: false,
            faults: 0,
            written_since_use: false,
        };
        plan.uncovered.insert(start..end);
        for e in olds.iter().filter_map(old) {
            plan.stale.insert_all(&e.stale);
            plan.unarmed.insert_all(&e.unarmed);
            plan.uncovered.remove(e.range.start()..e.range.end());
            plan.pending = plan.pending.max(e.pending);
            plan.streamed |= e.streamed;
            plan.faults = plan.faults.max(e.faults);
            plan.written_since_use |= e.written_since_use;
        }
        plan.stale.insert_all(&plan.uncovered);
        plan.olds = olds;
        Ok(plan)
    }

    /// A copy of each old entry's buffer into its place in the joined `buffer` at `start`.
    fn join_copies(&self, olds: &[u64], buffer: BufferId, start: u64) -> Vec<Command> {
        olds.iter()
            .filter_map(|s| self.entries.get(s))
            .map(|e| {
                Command::Copy(CopyDesc {
                    src: CopyRegion::Buffer(BufferSlice {
                        buffer: e.buffer,
                        offset: 0,
                        size: e.range.size(),
                    }),
                    dst: CopyRegion::Buffer(BufferSlice {
                        buffer,
                        offset: e.range.start() - start,
                        size: e.range.size(),
                    }),
                })
            })
            .collect()
    }

    /// Reads the pages of `spans` (absolute GPU addresses inside the entry at `entry_start`)
    /// into staging slices of at most `Limits::max_staging_size` bytes each, as uploads into
    /// `buffer`, in address order.
    fn stage<M: GpuMemory + ?Sized>(
        &self,
        mem: &M,
        buffer: BufferId,
        entry_start: u64,
        spans: &Spans,
        staged: &mut Staged,
    ) -> Result<(), BufferError> {
        let chunk = self.backend.caps().limits.max_staging_size.max(1);
        for span in spans.iter() {
            let mut at = span.start;
            while at < span.end {
                let size = (span.end - at).min(chunk);
                let mut slice = self.backend.staging(size).map_err(BufferError::Render)?;
                mem.read(at, slice.bytes_mut())
                    .map_err(BufferError::Fault)?;
                staged.commands.push(Command::Upload {
                    from: slice,
                    to: CopyRegion::Buffer(BufferSlice {
                        buffer,
                        offset: at - entry_start,
                        size,
                    }),
                });
                // `at + size <= span.end`, which is representable.
                at += size;
            }
        }
        Ok(())
    }

    fn next_stamp(&mut self) -> u64 {
        // One per bind: 2^64 binds cannot happen, and saturating keeps the order of every
        // earlier stamp.
        self.stamp = self.stamp.saturating_add(1);
        self.stamp
    }

    /// Records a use of `entry` in the open list.
    fn record_use(entry: &mut Entry, stamp: u64, frame: u64, open: &mut Vec<u64>) {
        if !entry.written_since_use {
            entry.faults = 0;
        }
        entry.written_since_use = false;
        entry.stamp = stamp;
        entry.frame = frame;
        if !entry.in_open {
            entry.in_open = true;
            open.push(entry.range.start());
        }
    }

    /// Removes the entry at `start` and destroys its buffer once no list can use it.
    fn retire(&mut self, start: u64) {
        let Some(entry) = self.entries.remove(&start) else {
            return;
        };
        self.bytes = self.bytes.saturating_sub(entry.range.size());
        if entry.in_open {
            self.doomed.push(entry.buffer);
        } else {
            self.backend
                .destroy(ResourceId::Buffer(entry.buffer), entry.last_fence);
        }
    }

    /// Drains the write watch's dirty log (module docs, "Learning of CPU writes"): each dirty
    /// range goes to `sink`, turns stale in the entries holding it, and counts a fault against
    /// each of them. Call before each draw, dispatch and copy; with nothing dirty it is one
    /// atomic load.
    pub fn sync_cpu_writes<S: InvalidationSink + ?Sized>(&mut self, sink: &mut S) {
        let mut dirty = core::mem::take(&mut self.dirty);
        self.watch.take_dirty(&mut dirty);
        let mut counted = BTreeSet::new();
        for range in &dirty {
            sink.invalidate(*range);
            let starts = self.overlapping(range.start()..range.end());
            for start in starts {
                let Some(entry) = self.entries.get_mut(&start) else {
                    continue;
                };
                let part =
                    range.start().max(entry.range.start())..range.end().min(entry.range.end());
                entry.stale.insert(part.clone());
                if entry.streamed {
                    continue;
                }
                if counted.insert(start) {
                    entry.faults = entry.faults.saturating_add(1);
                    entry.written_since_use = true;
                }
                if entry.faults >= STREAMED_AFTER_FAULTS {
                    entry.streamed = true;
                    entry.unarmed.clear();
                    entry.pending = None;
                } else if let Ok(pages) = PageRange::new(part.start, part.end - part.start) {
                    entry.pending = Some(self.watch.request(pages));
                    entry.unarmed.insert(part);
                }
            }
        }
        dirty.clear();
        self.dirty = dirty;
    }

    /// The back thread itself wrote `range` of guest memory (module docs, "GPU-thread writes
    /// and unmaps"): every page of it an entry holds turns stale. An empty range changes nothing.
    pub fn invalidate(&mut self, range: Range<u64>) {
        if range.is_empty() {
            return;
        }
        for start in self.overlapping(range.clone()) {
            let Some(entry) = self.entries.get_mut(&start) else {
                continue;
            };
            let lo = range.start.max(entry.range.start());
            let hi = range.end.min(entry.range.end());
            // Whole pages, within the entry: its ends are page aligned and `hi` is at most its
            // end, so rounding `hi` up stays at or below it.
            let page = starquake_memory::PAGE_SIZE;
            let lo = lo - lo % page;
            let hi = hi
                .div_ceil(page)
                .saturating_mul(page)
                .min(entry.range.end());
            entry.stale.insert(lo..hi);
        }
    }

    /// An ordered address-table update has been applied (module docs, "GPU-thread writes and
    /// unmaps"): every entry overlapping its range is dropped, whatever its kind.
    pub fn apply_update(&mut self, update: &VmUpdate) {
        let range = update.range();
        for start in self.overlapping(range.start()..range.end()) {
            self.retire(start);
        }
    }

    /// Starts of the entries sharing an address with `range`, in address order.
    fn overlapping(&self, range: Range<u64>) -> Vec<u64> {
        let mut starts: Vec<u64> = self
            .entries
            .range(..range.end)
            .rev()
            .take_while(|(_, e)| e.range.end() > range.start)
            .map(|(&s, _)| s)
            .collect();
        starts.reverse();
        starts
    }

    /// The open list was submitted and signals `fence`: every entry it used now lasts until
    /// `fence`, and the buffers of those dropped meanwhile are destroyed after it. Call after
    /// every submit of a list in which a slice from [`BufferCache::bind`] is used.
    pub fn submitted(&mut self, fence: FenceValue) {
        self.last_fence = self.last_fence.max(fence);
        for start in core::mem::take(&mut self.open) {
            if let Some(entry) = self.entries.get_mut(&start)
                && entry.in_open
            {
                entry.in_open = false;
                entry.last_fence = fence;
            }
        }
        for buffer in core::mem::take(&mut self.doomed) {
            self.backend.destroy(ResourceId::Buffer(buffer), fence);
        }
    }

    /// Ends a frame (module docs, "Eviction"): runs the collector if the cache's bytes reach the
    /// limits from the backend's budget, then counts the frame. Returns how many entries it
    /// evicted.
    pub fn end_frame(&mut self) -> usize {
        let limits = eviction_limits(self.backend.budget().allowed);
        let mut evicted = 0;
        if let Some(run) = collection(self.bytes, limits) {
            let mut old: Vec<(u64, u64)> = self
                .entries
                .iter()
                .filter(|(_, e)| {
                    !e.in_open
                        && self
                            .frame
                            .checked_sub(e.frame)
                            .is_some_and(|age| age >= run.after_frames)
                })
                .map(|(&s, e)| (e.stamp, s))
                .collect();
            old.sort_unstable();
            for (_, start) in old.into_iter().take(run.at_most) {
                self.retire(start);
                evicted += 1;
            }
        }
        self.frame = self.frame.saturating_add(1);
        evicted
    }
}

impl Drop for BufferCache {
    fn drop(&mut self) {
        // `last_fence` is at least every entry's own last fence.
        let after = self.last_fence;
        for entry in core::mem::take(&mut self.entries).into_values() {
            self.backend
                .destroy(ResourceId::Buffer(entry.buffer), after);
        }
        for buffer in core::mem::take(&mut self.doomed) {
            self.backend.destroy(ResourceId::Buffer(buffer), after);
        }
    }
}
