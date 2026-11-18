//! Syncpoints: the 192 counters gsv and the GPU share, their maxima, allocation with reserved
//! ids, the watched bitmap, the 32-bit and 64-bit wait comparisons, and the extension of a
//! 32-bit counter to the 64-bit values of the shared syncpoint page (graphics-server spec,
//! "Syncpoints and the syncpt session", slice 2; decision 3 puts the type in this crate).
//!
//! Ported from:
//!
//! - `eden/src/core/hle/service/nvdrv/core/syncpoint_manager.cpp` and `.h` @ cf654052
//!   (GPL-3.0-or-later, Copyright 2022 yuzu Emulator Project and Skyline Team and
//!   Contributors): the reserved ids, first-free allocation from id 1, freeing, the maxima
//!   (`IncrementSyncpointMaxExt`, `GetSyncpointFence`);
//! - `eden-pr4588/src/core/hle/service/am/service/graphics_syncpoint_cache.h` @ e5fb3b97
//!   (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project): `ExtendCounter`, `Unallocated`
//!   and `CounterValue::Observe`;
//! - `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor_protocol.h` @ e5fb3b97
//!   (same licence and copyright): `IsSyncpointWaitSatisfied`;
//! - `eden-pr4588/src/core/hle/service/nvdrv/core/syncpoint_manager.cpp` @ e5fb3b97:
//!   `ReadAllocatedSyncpoint`, which [`Syncpoints::read_allocated`] mirrors.
//!
//! # Who calls what
//!
//! gsv allocates and frees ids, raises maxima to compute fences ([`Syncpoints::increment_max`]),
//! and marks the ids someone waits on ([`Syncpoints::watch`]). The GPU's back thread (after host
//! completion) and gsv's video worker call [`Syncpoints::increment`], which pushes
//! [`GpuCompletion::SyncpointChanged`] to gsv's [`Completions`] only for a watched id;
//! [`Syncpoints::increment_by`] does several at once with one push (the null GPU front of a run
//! with no GPU back end, `starquake-core`'s graphics wiring, and its replay). Anyone
//! reads ([`Syncpoints::read`], [`Syncpoints::reached`]).
//!
//! Watching and incrementing race. Both sides use sequentially consistent operations in the
//! same pattern (gsv: set the watched bit, then read the counter; the incrementer: add to the
//! counter, then read the watched bit), so for every increment either gsv's read sees its value
//! or the incrementer sees the bit and pushes. The threaded test
//! `an_increment_after_a_watch_is_either_seen_or_pushed` checks the outcome (no value above the
//! one gsv read goes unpushed) and fails if `increment` skips the push. It does not reliably
//! catch reordering `increment` to read the watched bit before the add, which loses pushes: the
//! loss needs a watch to land between the two steps, and in one review the test caught that
//! mutation once in six runs. Nor can it show that a weaker memory ordering is wrong, since a
//! weakly ordered race is not reliably reproduced on the host. Both the order of the two steps
//! and the orderings are for review to keep.
//!
//! # What is recorded and what is snapshotted
//!
//! Nothing here writes to the trace. The counter values gsv acts on reach it through the
//! completion queue and are recorded there, as `SyncpointObserved` notes, when gsv's timer
//! thread drains it (graphics-server, "What is recorded"; slice 3). Allocation picks the lowest
//! free id, so the same calls in the same order give the same ids, and replaying gsv's requests
//! reproduces it. Allocation, exhaustion and frees are logged with `tracing` at debug level.
//!
//! [`Syncpoints::save`] writes the allocation, the counter values, the maxima and the watched
//! bitmap (graphics-server, "What is snapshotted"), and [`Syncpoints::restore`] builds a new
//! `Syncpoints` from them, checking them first. [`CounterExtension`] is plain serialisable data
//! for the snapshot of whoever writes the shared page (gsv, slice 5). See "Snapshot format".
//!
//! # Divergences from Eden
//!
//! Each but the fourth, which changes no result, is pinned by the test of `syncpoint/tests.rs`
//! it names:
//!
//! - **[`Syncpoints::reached`] uses the 32-bit wrapping comparison for every id.** Eden's
//!   `HasSyncpointExpired` uses it only for client-managed ids (26 and 27); for the others it
//!   compares within the window up to the maximum, `(counter_max - threshold) >= (counter_min -
//!   threshold)`, so a threshold beyond the maximum counts as expired. The spec's interface
//!   names the wrapping comparison (`reached`, "32-bit wrapping comparison"), so a fence beyond
//!   the maximum is not reached here (`a_fence_beyond_the_maximum_is_not_reached`). The
//!   client-managed flag Eden keeps only for that choice is not kept.
//! - **Errors instead of assertions.** Eden asserts in these four cases and, past the
//!   assertion, carries on: `FindFreeSyncpoint` returns 0 with every id taken, so
//!   `AllocateSyncpoint` reserves and returns id 0; `FreeSyncpoint` of an id that is not
//!   allocated clears the flag anyway; `IncrementSyncpointMaxExt` returns 0; and
//!   `GetSyncpointFence` returns an empty fence. Here allocation with every id taken is
//!   [`SyncpointError::Exhausted`] (`allocation_exhausts_at_exactly_the_allocatable_count`),
//!   and the other three on an id that is not allocated are [`SyncpointError::NotAllocated`],
//!   changing nothing (`operations_on_an_unallocated_id_are_errors`).
//! - **[`Syncpoints::read`] reads any id.** Eden's `ReadSyncpointMinValue` asserts on an id
//!   that is not allocated; the GPU reads through the syncpoint shim and increments without
//!   asking gsv, so here reads and increments work on any id
//!   (`reads_and_increments_work_on_any_id`). Likewise [`Syncpoints::reached`] compares the
//!   counter of an id that is not allocated, where Eden's `HasSyncpointExpired` asserts and
//!   returns false (`if (!syncpoint.reserved)`), so it can be true here
//!   (`reached_compares_the_counter_wrapping`, on the unallocated id 40).
//!   [`Syncpoints::read_allocated`] is PR's `ReadAllocatedSyncpoint`, `None` for an id that is
//!   not allocated.
//! - **One counter, not a cache of host1x.** Eden's `counter_min` is a copy of the host1x
//!   counter synced by `UpdateMin`; here the counter is the value itself, incremented in place.
//! - **Allocation takes no lock.** Eden scans under a mutex; here each id is claimed with one
//!   atomic operation in the same lowest-first order, so concurrent allocations never hand out
//!   one id twice (`concurrent_allocations_hand_out_each_id_once`).
//!
//! Kept as Eden does it: an id's counter and maximum are not reset when it is freed or
//! allocated again (`a_reallocated_id_keeps_its_values`), and a reserved id can be freed like
//! any other and is then allocated in its turn (`a_reserved_id_can_be_freed_and_allocated`).
//!
//! # Guesses
//!
//! - The count (192, [`SYNCPOINT_COUNT`]) and the reserved ids ([`RESERVED_SYNCPOINTS`]) are
//!   Eden's Switch 1 values; that Switch 2 keeps them is a guess (graphics-server, "What to
//!   verify before building", item 10).
//!
//! # Conflicts
//!
//! - **conflict:** [`CompareMode::Wide64`] (mode 1). PR's `IsSyncpointWaitSatisfied` compares
//!   the 32-bit counter, zero-extended to 64 bits, with the 64-bit threshold; NX2 passes only
//!   the low 32 bits of the threshold whatever the mode
//!   (`eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb, `RequestSlotEvent`). The spec takes
//!   PR's reading under its decision 8 ("mode 1 compares as 64-bit"), and so does this code. It
//!   does not compare the extended value of [`CounterExtension`]. Pinned by
//!   `the_wide_comparison_at_the_edge_on_both_sides`, through
//!   `!wait_satisfied(u32::MAX, WRAP, Wide64)`, which NX2's reading would call reached. A trace
//!   of a mode-1 wait with a threshold above `u32::MAX` would settle it.
//!
//! # Snapshot format
//!
//! `postcard`, after a header of [`SNAPSHOT_MAGIC`] and [`SNAPSHOT_VERSION`] that is decoded
//! and checked before the body. The body is the allocated bitmap and the watched bitmap
//! ([`BITMAP_WORDS`] words each, id `i` at bit `i % 64` of word `i / 64`), then the counter
//! values and the maxima, [`SYNCPOINT_COUNT`] each, by id. The golden-bytes test
//! `the_snapshot_layout_is_pinned` pins it; changing it means bumping [`SNAPSHOT_VERSION`].
//!
//! A snapshot is a file and is not trusted. [`Syncpoints::restore`] checks, in this order, and
//! returns the first failure: the magic, the version, that the body decodes, that no bytes
//! follow it, that it holds [`SYNCPOINT_COUNT`] counter values, then as many maxima, and that
//! no id below [`FIRST_ALLOCATABLE`] is marked allocated (nothing allocates or reserves one).
//! [`Syncpoints::save`] runs the same checks on what it is about to write. Not checked, because
//! no relation holds between them: a counter against its maximum (the GPU can increment past
//! it, through the shim), a watched bit against the allocation (gsv can free an id while a
//! wait on it is still armed), and whether the reserved ids are allocated (they can be freed).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::interface::{Completions, Fence, GpuCompletion, SYNCPOINT_COUNT, SyncpointId};

#[cfg(test)]
mod tests;

/// [`SYNCPOINT_COUNT`] as an index bound.
const COUNT: usize = SYNCPOINT_COUNT as usize;

/// Bits in one bitmap word.
const WORD_BITS: usize = u64::BITS as usize;

/// Words in the allocated and the watched bitmaps.
pub const BITMAP_WORDS: usize = COUNT / WORD_BITS;

// Every bitmap word is full, so no bit names an id at or above the count.
const _: () = assert!(
    COUNT.is_multiple_of(WORD_BITS),
    "the bitmaps assume whole words of ids"
);

/// The ids reserved when [`Syncpoints`] is made, which [`Syncpoints::allocate`] never returns
/// while they stay reserved. Eden's `SyncpointManager` constructor
/// (`eden/src/core/hle/service/nvdrv/core/syncpoint_manager.cpp` @ cf654052) reserves 26 and 27
/// (`VBlank0SyncpointId`, `VBlank1SyncpointId`), then each non-zero entry of
/// `channel_syncpoints` (`syncpoint_manager.h`): 0xC (`VIC`), 0x36 (`NvDec`) and 0x37
/// (`NvJpg`). Whether Switch 2 reserves the same ids is a guess (graphics-server, verification
/// item 10).
pub const RESERVED_SYNCPOINTS: [u8; 5] = [26, 27, 0xC, 0x36, 0x37];

/// The lowest id [`Syncpoints::allocate`] returns: Eden's `FindFreeSyncpoint` scans from 1, so
/// id 0 is never handed out.
pub const FIRST_ALLOCATABLE: u8 = 1;

/// The value of a counter that is not allocated, in 64-bit form: all ones (PR
/// `GraphicsSyncpointCache::Unallocated`).
pub const UNALLOCATED: u64 = u64::MAX;

/// The low half of a 64-bit counter value.
const LOW_HALF: u64 = u32::MAX as u64;

/// One wrap of a 32-bit counter, in 64-bit form (PR `ExtendCounter`'s `1ULL << 32`).
const WRAP: u64 = 1 << u32::BITS;

/// The sign bit of a 32-bit difference (PR `IsSyncpointWaitSatisfied`'s `0x80000000U`).
const SIGN_BIT: u32 = 1 << (u32::BITS - 1);

/// First bytes of every syncpoint snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SQSP";

/// Layout version written into every syncpoint snapshot and required when reading one.
pub const SNAPSHOT_VERSION: u32 = 1;

/// How a wait compares a counter with its threshold (the mode word of syncpt request 201:
/// graphics-server, "Syncpoints and the syncpt session").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareMode {
    /// Mode 0: reached once `current - threshold`, both 32 bits and wrapping, is not negative.
    Wrapping32,
    /// Mode 1: reached once the counter, zero-extended, is at least the 64-bit threshold. A
    /// conflict between the references, settled by PR's reading (module docs, "Conflicts").
    Wide64,
}

/// Whether a counter at `current` satisfies a wait for `threshold` (PR
/// `IsSyncpointWaitSatisfied`): [`CompareMode::Wrapping32`] takes the threshold's low 32 bits
/// and checks the sign bit of the wrapping difference, [`CompareMode::Wide64`] compares
/// `u64::from(current) >= threshold`.
pub fn wait_satisfied(current: u32, threshold: u64, mode: CompareMode) -> bool {
    match mode {
        CompareMode::Wrapping32 => current.wrapping_sub(low_half(threshold)) & SIGN_BIT == 0,
        CompareMode::Wide64 => u64::from(current) >= threshold,
    }
}

/// The 64-bit value of a counter that was `previous` when last observed and is `current` now
/// (PR `ExtendCounter`): `current` itself after [`UNALLOCATED`]; otherwise `previous`'s high
/// half, plus one wrap when `current` is below `previous`'s low half, with `current` as the low
/// half. The high half wraps to 0 past `u64::MAX`, as the C++ unsigned addition does.
pub fn extend_counter(previous: u64, current: u32) -> u64 {
    if previous == UNALLOCATED {
        return u64::from(current);
    }
    let mut high = previous & !LOW_HALF;
    if current < low_half(previous) {
        high = high.wrapping_add(WRAP);
    }
    high | u64::from(current)
}

/// The low 32 bits of `value`.
fn low_half(value: u64) -> u32 {
    let [a, b, c, d, ..] = value.to_le_bytes();
    u32::from_le_bytes([a, b, c, d])
}

/// Why a syncpoint operation failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum SyncpointError {
    /// Every id from [`FIRST_ALLOCATABLE`] up is allocated or reserved.
    #[error("every syncpoint is allocated")]
    Exhausted,
    /// The operation needs an allocated id.
    #[error("syncpoint {} is not allocated", .0.get())]
    NotAllocated(SyncpointId),
}

/// Why a syncpoint snapshot could not be written or read.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SnapshotError {
    /// Encoding failed.
    #[error("could not encode the syncpoint snapshot: {0}")]
    Encode(postcard::Error),
    /// The header or the body is not a well-formed encoding.
    #[error("could not decode the syncpoint snapshot: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`SNAPSHOT_MAGIC`].
    #[error("not a syncpoint snapshot (bad magic)")]
    BadMagic,
    /// The snapshot was written with another layout version.
    #[error(
        "syncpoint snapshot layout version {found} is not supported (this build reads {SNAPSHOT_VERSION})"
    )]
    UnsupportedVersion {
        /// The version found.
        found: u32,
    },
    /// Bytes follow the body.
    #[error("{count} unexpected bytes after the syncpoint snapshot")]
    TrailingBytes {
        /// How many.
        count: usize,
    },
    /// The body holds the wrong number of counter values.
    #[error("the syncpoint snapshot holds {found} counter values, not {SYNCPOINT_COUNT}")]
    CounterCount {
        /// How many it holds.
        found: usize,
    },
    /// The body holds the wrong number of maxima.
    #[error("the syncpoint snapshot holds {found} maxima, not {SYNCPOINT_COUNT}")]
    MaximumCount {
        /// How many it holds.
        found: usize,
    },
    /// An id that is never allocated is marked allocated.
    #[error("the syncpoint snapshot marks syncpoint {id} allocated, which is never allocated")]
    NeverAllocated {
        /// The id.
        id: usize,
    },
}

/// The syncpoint counters, their maxima, which ids are allocated and which are watched.
///
/// Shared between gsv and the GPU behind an `Arc`; every method takes `&self` and none blocks.
pub struct Syncpoints {
    /// The value each counter has reached (Eden's `counter_min`).
    counters: [AtomicU32; COUNT],
    /// The value each counter will reach once the work issued so far is done (Eden's
    /// `counter_max`).
    maxima: [AtomicU32; COUNT],
    /// Allocated or reserved ids (Eden's `reserved`), one bit each.
    allocated: [AtomicU64; BITMAP_WORDS],
    /// Ids someone waits on, one bit each.
    watched: [AtomicU64; BITMAP_WORDS],
    /// Where watched changes go.
    completions: Arc<dyn Completions>,
}

/// The word and the bit of `id` in a bitmap.
fn bit(id: SyncpointId) -> (usize, u64) {
    let index = usize::from(id.get());
    (index / WORD_BITS, 1 << (index % WORD_BITS))
}

/// Every syncpoint id, in order.
fn all_ids() -> impl Iterator<Item = SyncpointId> {
    (0..=u8::MAX).map_while(SyncpointId::new)
}

impl Syncpoints {
    /// All counters and maxima at 0, nothing watched, and [`RESERVED_SYNCPOINTS`] reserved.
    /// Watched changes are pushed to `completions`.
    pub fn new(completions: Arc<dyn Completions>) -> Self {
        let syncpoints = Self::blank(completions);
        for id in RESERVED_SYNCPOINTS.into_iter().filter_map(SyncpointId::new) {
            let (word, mask) = bit(id);
            syncpoints.allocated[word].fetch_or(mask, Ordering::AcqRel);
        }
        syncpoints
    }

    /// Everything zero and nothing reserved.
    fn blank(completions: Arc<dyn Completions>) -> Self {
        Self {
            counters: std::array::from_fn(|_| AtomicU32::new(0)),
            maxima: std::array::from_fn(|_| AtomicU32::new(0)),
            allocated: std::array::from_fn(|_| AtomicU64::new(0)),
            watched: std::array::from_fn(|_| AtomicU64::new(0)),
            completions,
        }
    }

    /// Allocates the lowest free id from [`FIRST_ALLOCATABLE`] up (Eden `AllocateSyncpoint`).
    pub fn allocate(&self) -> Result<SyncpointId, SyncpointError> {
        for id in all_ids().skip(usize::from(FIRST_ALLOCATABLE)) {
            let (word, mask) = bit(id);
            let slot = &self.allocated[word];
            // A load first, so taken ids cost no write.
            if slot.load(Ordering::Acquire) & mask == 0
                && slot.fetch_or(mask, Ordering::AcqRel) & mask == 0
            {
                tracing::debug!(id = id.get(), "syncpoint allocated");
                return Ok(id);
            }
        }
        tracing::debug!("syncpoints exhausted");
        Err(SyncpointError::Exhausted)
    }

    /// Frees `id` (Eden `FreeSyncpoint`). Its counter and maximum keep their values.
    pub fn free(&self, id: SyncpointId) -> Result<(), SyncpointError> {
        let (word, mask) = bit(id);
        if self.allocated[word].fetch_and(!mask, Ordering::AcqRel) & mask == 0 {
            return Err(SyncpointError::NotAllocated(id));
        }
        tracing::debug!(id = id.get(), "syncpoint freed");
        Ok(())
    }

    /// Whether `id` is allocated or reserved (Eden `IsSyncpointAllocated`).
    pub fn is_allocated(&self, id: SyncpointId) -> bool {
        let (word, mask) = bit(id);
        self.allocated[word].load(Ordering::Acquire) & mask != 0
    }

    /// The value counter `id` has reached, allocated or not.
    pub fn read(&self, id: SyncpointId) -> u32 {
        self.counters[usize::from(id.get())].load(Ordering::SeqCst)
    }

    /// The value counter `id` has reached, or `None` if it is not allocated (PR
    /// `ReadAllocatedSyncpoint`).
    pub fn read_allocated(&self, id: SyncpointId) -> Option<u32> {
        self.is_allocated(id).then(|| self.read(id))
    }

    /// Adds one to counter `id`, wrapping, and returns the new value. If `id` is watched,
    /// pushes [`GpuCompletion::SyncpointChanged`] with that value; otherwise the cost beyond
    /// the add is one load.
    pub fn increment(&self, id: SyncpointId) -> u32 {
        self.increment_by(id, 1)
    }

    /// Adds `amount` to counter `id` at once, wrapping, and returns the new value: `amount`
    /// increments with one push, the one the last of them would make. If `id` is watched,
    /// pushes [`GpuCompletion::SyncpointChanged`] with the new value; an `amount` of 0 changes
    /// nothing and pushes nothing. For an incrementer that finishes several increments at one
    /// moment (a GPU with no host work behind them), so that the work does not grow with the
    /// count; the same order of the add and the watched check as [`Syncpoints::increment`]
    /// (module docs, "Who calls what").
    pub fn increment_by(&self, id: SyncpointId, amount: u32) -> u32 {
        let counter = &self.counters[usize::from(id.get())];
        if amount == 0 {
            return counter.load(Ordering::SeqCst);
        }
        let value = counter.fetch_add(amount, Ordering::SeqCst).wrapping_add(amount);
        if self.is_watched(id) {
            self.completions
                .push(GpuCompletion::SyncpointChanged { id, value });
        }
        value
    }

    /// Whether counter `id` has reached `fence.value`, by the 32-bit wrapping comparison
    /// ([`CompareMode::Wrapping32`]).
    pub fn reached(&self, fence: Fence) -> bool {
        wait_satisfied(
            self.read(fence.id),
            u64::from(fence.value),
            CompareMode::Wrapping32,
        )
    }

    /// Raises the maximum of `id` by `amount`, wrapping, and returns the new maximum (Eden
    /// `IncrementSyncpointMaxExt`).
    pub fn increment_max(&self, id: SyncpointId, amount: u32) -> Result<u32, SyncpointError> {
        if !self.is_allocated(id) {
            return Err(SyncpointError::NotAllocated(id));
        }
        Ok(self.maxima[usize::from(id.get())]
            .fetch_add(amount, Ordering::AcqRel)
            .wrapping_add(amount))
    }

    /// The fence on `id` at its maximum: done once the work issued so far is (Eden
    /// `GetSyncpointFence`).
    pub fn fence(&self, id: SyncpointId) -> Result<Fence, SyncpointError> {
        if !self.is_allocated(id) {
            return Err(SyncpointError::NotAllocated(id));
        }
        Ok(Fence {
            id,
            value: self.maxima[usize::from(id.get())].load(Ordering::Acquire),
        })
    }

    /// Marks `id` watched, so increments of it are pushed. Read the counter after this call, not
    /// before, to catch an increment that came first (module docs, "Who calls what").
    pub fn watch(&self, id: SyncpointId) {
        let (word, mask) = bit(id);
        self.watched[word].fetch_or(mask, Ordering::SeqCst);
    }

    /// Marks `id` not watched.
    pub fn unwatch(&self, id: SyncpointId) {
        let (word, mask) = bit(id);
        self.watched[word].fetch_and(!mask, Ordering::SeqCst);
    }

    /// Whether `id` is watched.
    pub fn is_watched(&self, id: SyncpointId) -> bool {
        let (word, mask) = bit(id);
        self.watched[word].load(Ordering::SeqCst) & mask != 0
    }

    /// Writes the allocation, counters, maxima and watched bitmap (module docs, "Snapshot
    /// format"). Take it while nothing increments or allocates, or it mixes moments.
    pub fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let load = |words: &[AtomicU64; BITMAP_WORDS]| {
            std::array::from_fn(|word| words[word].load(Ordering::Acquire))
        };
        let state = State {
            allocated: load(&self.allocated),
            watched: load(&self.watched),
            counters: self
                .counters
                .iter()
                .map(|value| value.load(Ordering::Acquire))
                .collect(),
            maxima: self
                .maxima
                .iter()
                .map(|value| value.load(Ordering::Acquire))
                .collect(),
        };
        state.check()?;
        let header = Header {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
        };
        let mut bytes = postcard::to_allocvec(&header).map_err(SnapshotError::Encode)?;
        bytes.extend(postcard::to_allocvec(&state).map_err(SnapshotError::Encode)?);
        Ok(bytes)
    }

    /// A `Syncpoints` holding what [`save`](Self::save) wrote, pushing watched changes to
    /// `completions`. The snapshot is checked first (module docs, "Snapshot format").
    pub fn restore(bytes: &[u8], completions: Arc<dyn Completions>) -> Result<Self, SnapshotError> {
        let (header, rest) =
            postcard::take_from_bytes::<Header>(bytes).map_err(SnapshotError::Decode)?;
        if header.magic != SNAPSHOT_MAGIC {
            return Err(SnapshotError::BadMagic);
        }
        if header.version != SNAPSHOT_VERSION {
            return Err(SnapshotError::UnsupportedVersion {
                found: header.version,
            });
        }
        let (state, rest) =
            postcard::take_from_bytes::<State>(rest).map_err(SnapshotError::Decode)?;
        if !rest.is_empty() {
            return Err(SnapshotError::TrailingBytes { count: rest.len() });
        }
        state.check()?;
        let syncpoints = Self::blank(completions);
        let store_words = |target: &[AtomicU64; BITMAP_WORDS], words: [u64; BITMAP_WORDS]| {
            for (slot, word) in target.iter().zip(words) {
                slot.store(word, Ordering::Release);
            }
        };
        store_words(&syncpoints.allocated, state.allocated);
        store_words(&syncpoints.watched, state.watched);
        for (slot, value) in syncpoints.counters.iter().zip(state.counters) {
            slot.store(value, Ordering::Release);
        }
        for (slot, value) in syncpoints.maxima.iter().zip(state.maxima) {
            slot.store(value, Ordering::Release);
        }
        Ok(syncpoints)
    }
}

/// The snapshot header.
#[derive(Serialize, Deserialize)]
struct Header {
    /// [`SNAPSHOT_MAGIC`].
    magic: [u8; 4],
    /// [`SNAPSHOT_VERSION`].
    version: u32,
}

/// The snapshot body (module docs, "Snapshot format").
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct State {
    /// Allocated or reserved ids.
    allocated: [u64; BITMAP_WORDS],
    /// Watched ids.
    watched: [u64; BITMAP_WORDS],
    /// Counter values, by id.
    counters: Vec<u32>,
    /// Maxima, by id.
    maxima: Vec<u32>,
}

impl State {
    /// The checks [`Syncpoints::restore`] and [`Syncpoints::save`] share, in the documented
    /// order.
    fn check(&self) -> Result<(), SnapshotError> {
        if self.counters.len() != COUNT {
            return Err(SnapshotError::CounterCount {
                found: self.counters.len(),
            });
        }
        if self.maxima.len() != COUNT {
            return Err(SnapshotError::MaximumCount {
                found: self.maxima.len(),
            });
        }
        for id in all_ids().take(usize::from(FIRST_ALLOCATABLE)) {
            let (word, mask) = bit(id);
            if self.allocated[word] & mask != 0 {
                return Err(SnapshotError::NeverAllocated {
                    id: usize::from(id.get()),
                });
            }
        }
        Ok(())
    }
}

/// The 64-bit value of each counter as last observed, for the shared syncpoint page (PR
/// `GraphicsSyncpointCache::CounterValue`, one per counter). Every entry starts at
/// [`UNALLOCATED`].
///
/// Serialises as the [`SYNCPOINT_COUNT`] values by id; deserialising checks the count.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<u64>", into = "Vec<u64>")]
pub struct CounterExtension {
    /// The last value observed, by id.
    previous: Vec<u64>,
}

/// A [`CounterExtension`] was given the wrong number of values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("a counter extension holds {found} values, not {SYNCPOINT_COUNT}")]
pub struct ExtensionLengthError {
    /// How many it was given.
    pub found: usize,
}

impl Default for CounterExtension {
    fn default() -> Self {
        Self {
            previous: vec![UNALLOCATED; COUNT],
        }
    }
}

impl CounterExtension {
    /// The 64-bit value of counter `id`, now `current` (`None` if it is not allocated, as
    /// [`Syncpoints::read_allocated`] gives it), and remembers it (PR `CounterValue::Observe`).
    /// `None` returns [`UNALLOCATED`] and leaves the remembered value as it was.
    pub fn observe(&mut self, id: SyncpointId, current: Option<u32>) -> u64 {
        let Some(current) = current else {
            return UNALLOCATED;
        };
        let previous = &mut self.previous[usize::from(id.get())];
        *previous = extend_counter(*previous, current);
        *previous
    }

    /// The 64-bit value of counter `id` as last observed ([`CounterExtension::observe`]), or
    /// [`UNALLOCATED`] if no allocated value has been observed: what the shared syncpoint page
    /// shows for it (gsv's `shared::page`). Reading changes nothing.
    #[must_use]
    pub fn value(&self, id: SyncpointId) -> u64 {
        self.previous[usize::from(id.get())]
    }
}

impl TryFrom<Vec<u64>> for CounterExtension {
    type Error = ExtensionLengthError;

    fn try_from(previous: Vec<u64>) -> Result<Self, Self::Error> {
        if previous.len() != COUNT {
            return Err(ExtensionLengthError {
                found: previous.len(),
            });
        }
        Ok(Self { previous })
    }
}

impl From<CounterExtension> for Vec<u64> {
    fn from(extension: CounterExtension) -> Self {
        extension.previous
    }
}
