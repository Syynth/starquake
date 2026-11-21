//! The GPU's address table: [`GpuVm`], one per GPU address space, from page ranges of GPU
//! addresses to a [`Target`] and a [`Kind`], with a version that every change moves
//! (graphics-server spec, "GPU memory: the gpu session and its worker" and "How mappings reach
//! the GPU", slice 6; decision 3 puts the type in this crate).
//!
//! # Who calls what
//!
//! gsv keeps the bookkeeping (registrations, address spaces, sparse regions) and validates every
//! request against it. It writes new mappings into the table itself, at once ([`GpuVm::map`]):
//! no work queued before a map can name its range, because the title waits for the job's event
//! before using it. Unmaps, kind changes and destroys go to the GPU as ordered updates
//! ([`VmUpdate`]), which the GPU applies ([`GpuVm::apply`]) only once no earlier work can still
//! use the old mapping. GPU threads translate through the table, each with its own
//! [`TranslationCache`](crate::memory::TranslationCache), which [`GpuVm`] serves through
//! [`Translate`]: the cache drops its entries when the version moves.
//!
//! Behaviour ported from Eden-NX2's `Nx2GpuMemoryRegistry`
//! (`eden-nx2/src/core/hle/service/nvdrv/devices/nx2_stub_device.h` @ 50fa86bb;
//! GPL-3.0-or-later, Copyright 2026 Eden Emulator Project), the reference the spec names for this
//! slice: `MapAt` and `Unmap` both start with `TrimMappings`, which drops the range from every
//! mapping it overlaps and keeps the parts outside it, the part after the range leading as far
//! into the old target as it starts into the old mapping (`m.cpu_address + (end -
//! m.gpu_address)`).
//!
//! # Rules
//!
//! - Ranges are whole pages of [`PAGE_SIZE`] (4 KiB, as gsv's `...4K` commands map them), given
//!   as a [`PageRange`], so a range is never empty and its end is representable.
//! - **A map replaces whatever it overlaps** (NX2 `MapAt`: `TrimMappings`, then the new mapping):
//!   the overlapped pages of older mappings lead to the new target, and their pages outside the
//!   range keep theirs. This is how a 211 inside a sparse region takes over `GpuOnly` pages
//!   (graphics-server, "The model"). A conflict with PR ("Conflicts between the references").
//!   Pinned by `tests::a_map_replaces_what_it_overlaps`.
//! - **A removal takes out every mapped page of its range and leaves the rest** (NX2 `Unmap`:
//!   `TrimMappings`): it may split a mapping or cover holes (a conflict with PR, below).
//!   [`VmUpdate::Unmap`], [`VmUpdate::DestroyMemory`] and [`VmUpdate::DestroySparseRegion`] all
//!   do this to the table.
//!   Pinned by `tests::an_unmap_splits_a_mapping_and_skips_holes`.
//! - **A kind change sets the kind of every mapped page of its range** and leaves holes as holes,
//!   splitting mappings at the range's ends (kinds and the range are conflicts, below). Pinned by
//!   `tests::a_kind_change_spans_mappings_and_skips_holes`.
//! - **A target must fit the range** ([`GpuVm::map`] lists the checks): a `Guest` CPU address or a
//!   `GpuOnly` offset is page aligned and its end is representable, and a `SyncpointShim` mapping
//!   is exactly [`SHIM_PAGES`] pages.
//! - **The version** ([`Translate::generation`]) moves by one with every successful
//!   [`GpuVm::map`] and every [`GpuVm::apply`], including one whose range holds no mapping: the
//!   spec's "moved by every map, ordered update or kind change". A refused map does not move it.
//!   It moves while the table is still locked for the change, after the change, with a release
//!   store, so a thread that reads the new version and then translates sees the new table. The
//!   threaded test `tests::a_reader_that_sees_a_version_sees_its_maps` checks this; it catches a
//!   version moved before the change only when a reader lands between the two (in review it
//!   caught that mutation in three runs of three), and it cannot show that a weaker memory
//!   ordering is wrong, so the order and the orderings are for review to keep.
//!
//! # Conflicts between the references
//!
//! Where PR and NX2 disagree, the table picks one reading by the house rule of graphics-server
//! decision 8 (Nvidia's headers and readings that cite the system module for layouts and
//! meanings; where one answer contains the other, the larger; otherwise the one a title ran on).
//! Each case below is marked **conflict** here and next to the code that makes the choice, names
//! the leg of the rule that picked it and the open question, and is pinned by the test of
//! `vm/tests.rs` it names. The spec's "What to verify before building", item 1, does not list
//! these three yet; until it does, the open question stated with each is the check that settles
//! it (a trace of the replies a title reads after such a request).
//!
//! - **conflict: a map over a live mapping.** PR's `nvhost_as_gpu::MapNative4K`
//!   (`eden-pr4588/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.cpp` @ e5fb3b97) refuses a
//!   range that overlaps any allocation, sparse reservations included (the `allocation_map`
//!   check, `BadParameter`). NX2's `MapAt` replaces what the range overlaps. Picked: NX2, by the
//!   leg "otherwise the one a title ran on" (neither answer contains the other, and NX2 is the
//!   reading a title ran on). Open question: does the system module refuse a 211 over a live
//!   mapping, and with what result? `a_map_replaces_what_it_overlaps`.
//! - **conflict: kinds.** PR's `nvhost_as_gpu::ModifyKind` remaps the range with the new kind and
//!   keeps a kind per mapping; NX2's registry has no kinds and its 212 changes nothing
//!   (`eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb, `SubModifyKind`: "Kinds do not matter
//!   to the emulated GPU"). Picked: PR, kinds are kept, by the leg "where one answer contains the
//!   other, the larger" (a table with kinds holds everything a table without them does). Open
//!   question: does any GPU work a title runs depend on a page's kind?
//!   `a_kind_change_spans_mappings_and_skips_holes`.
//! - **conflict: the range checks of 212, 213 and 215.** PR answers `ResultNotFound` for a range
//!   that is not inside one of its recorded maps (212; and `ModifyKind`'s `delta >= mapping.size
//!   || size > mapping.size - delta` answers `BadParameter`) or not exactly one (213 and 215:
//!   `mapping.address == begin && end - begin + 1 == mapping.size`). NX2's `Unmap` trims any
//!   range. Picked for the table: NX2's, by the leg "where one answer contains the other, the
//!   larger" (it applies every range PR applies, with the same result, and others too) and the
//!   one a title ran on: a kind change sets the mapped pages of any range, and a removal trims
//!   any range. An ordered update is applied after the guest has had its reply, so it cannot fail
//!   here; whether gsv refuses such a range at request time is deferred to slice 7. Open
//!   question: which ranges does the system module refuse for 212, 213 and 215, and with what
//!   result? `a_kind_change_spans_mappings_and_skips_holes`,
//!   `an_unmap_splits_a_mapping_and_skips_holes`.
//!
//! # Divergences from the references
//!
//! Each is pinned by the test of `vm/tests.rs` it names:
//!
//! - **Targets are checked.** NX2's `MapAt` takes any CPU address and size, and its
//!   `TrimMappings` adds them as unsigned 64-bit values, which wrap. Here a target that is not
//!   page aligned, or whose end is not representable, is an error.
//!   `a_target_must_be_aligned_and_end_in_range`.
//! - **The shim is one page.** PR's `GetNativeSyncpointShim` reserves one page
//!   (`allocation.pages = 1`) for 192 syncpoints of 4 bytes each; a `SyncpointShim` mapping of any
//!   other size is [`VmError::ShimSize`]. NX2's registry has no shim. `the_shim_is_exactly_one_page`.
//! - **One table per address space.** NX2's registry is one per emulator (`Get()` returns a
//!   static) and every nvhost-as-gpu address space applies each mapping; here gsv makes a
//!   [`GpuVm`] per address space (`GpuFront::create_address_space(space, vm: Arc<GpuVm>)` in the
//!   spec's interface block). Not a behaviour a test can pin inside one table.
//!
//! # The GPU-only store
//!
//! Each table owns the host pages behind its `GpuOnly` targets, a
//! [`GpuOnlyStore`](crate::memory::GpuOnlyStore) (gpu-protocol, "GPU-only memory"; slice G22),
//! reached through [`GpuVm::gpu_only`]. [`GpuVm::apply`] of a
//! [`VmUpdate::DestroySparseRegion`] frees every page of its region, after removing the range from
//! the table and before moving the version, under the table's write lock: the spec's "a
//! sparse-region destroy is an ordered update: the store frees the range only when it is
//! applied". The production memory ([`crate::memory::VmMemory`]) writes a `GpuOnly` page only
//! through [`GpuVm::write_gpu_only`], which translates the page afresh and writes it under the
//! table's read lock, so no write that translated to a destroyed region can land after the free
//! and allocate a page again. Reads take no table lock: a read racing a destroy sees the old
//! bytes or zeros, and allocates nothing. The lock order is the table, then the store.
//!
//! The store frees by region, not by range: a region's pages are keyed by their offset into it,
//! and a destroy ends the whole region. gsv must not hand a destroyed region's id to a new region
//! before the destroy is reported done (`GpuCompletion::GpuOrdered`), or the destroy, applied
//! later, frees the new region's pages too.
//!
//! # What is recorded and what is snapshotted
//!
//! Nothing here writes to the trace. Every change comes from a gsv request (recorded as IPC) or
//! from an ordered update gsv learns is done through `GpuOrderedDone` notes (graphics-server,
//! "What is recorded"); the table is a function of those inputs in their order. Maps and
//! applied updates are logged with `tracing` at debug level.
//!
//! [`GpuVm::save`] writes the version and the mappings (graphics-server, "What is snapshotted":
//! "address spaces, mappings, sparse regions"), and [`GpuVm::restore`] builds a table from them,
//! checking them first. See "Snapshot format". The GPU-only store's contents are in neither (a
//! GPU snapshot that reads them back is deferred, gpu-protocol "GPU-only memory"): a restored
//! table starts with an empty store, so its `GpuOnly` pages read as zero.
//!
//! # The semaphore mirror page
//!
//! A table also lists the registrations gsv says are two pages long
//! ([`GpuVm::add_semaphore_registration`]) and answers where a fenced release in the second page
//! of one is copied ([`GpuVm::semaphore_mirror`]); the fence path makes the copy. Tentative, by
//! the owner's ruling of 2026-11-20; the rule, its sources and its divergences from Eden-NX2 are
//! in the `mirror` submodule's docs. The list is not a translation: it does not move the version
//! and is not in the snapshot.
//!
//! # Bounds
//!
//! Each map adds one mapping and splits at most two; each update splits at most two. The table
//! therefore grows by at most three mappings per request gsv accepts. The semaphore list holds
//! one entry per start of a listed registration, so at most as many as gsv holds registrations;
//! nothing else allocates. A snapshot's mapping list is bounded by the snapshot's own length.
//!
//! # Snapshot format
//!
//! `postcard`, after a header of [`SNAPSHOT_MAGIC`] and [`SNAPSHOT_VERSION`] that is decoded and
//! checked before the body. The body is the version, then the mappings in address order, each
//! its GPU start, its size, its target (variant index 0 `Guest` with the CPU address, 1
//! `GpuOnly` with the region and the offset, 2 `SyncpointShim`) and its kind byte. The golden-bytes
//! test `the_snapshot_layout_is_pinned` pins it; changing it means bumping [`SNAPSHOT_VERSION`].
//!
//! A snapshot is a file and is not trusted. [`GpuVm::restore`] checks, in this order, and returns
//! the first failure: the magic, the version, that the body decodes, that no bytes follow it,
//! then each mapping in turn: that its GPU range is a [`PageRange`]
//! ([`SnapshotError::Range`]), that its target fits it (the checks of [`GpuVm::map`],
//! [`SnapshotError::Target`]), and that it starts at or after the end of the one before
//! ([`SnapshotError::Overlap`]). [`GpuVm::save`] runs the same checks on what it is about to
//! write. No test makes that check fail: [`GpuVm::map`] and [`GpuVm::apply`] keep the table
//! within the same rules, so the save-side check is a guard against a future change to them.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use serde::{Deserialize, Serialize};
use starquake_memory::{MapError, PAGE_SIZE, PageRange};
use thiserror::Error;

use crate::interface::{Kind, RegionId, Target, VmUpdate};
use crate::memory::{GpuOnlyRangeError, GpuOnlyStore, Translate};

mod mirror;
#[cfg(test)]
mod tests;

use mirror::SemaphorePages;
pub use mirror::{SEMAPHORE_MIRROR_OFFSET, SEMAPHORE_REGISTRATION_BYTES, SemaphoreRegistration};

/// The size of the syncpoint shim in pages: PR's `nvhost_as_gpu::GetNativeSyncpointShim`
/// (`eden-pr4588/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.cpp` @ e5fb3b97) reserves
/// `allocation.pages = 1` of `YUZU_PAGESIZE` at the top of the address space.
pub const SHIM_PAGES: u64 = 1;

/// First bytes of every GPU VM snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SQVM";

/// Layout version written into every GPU VM snapshot and required when reading one.
pub const SNAPSHOT_VERSION: u32 = 1;

/// Where one page leads: its own target (the mapping's target, as far further in as the page is
/// into the mapping) and the mapping's kind. What [`GpuVm`] gives a
/// [`TranslationCache`](crate::memory::TranslationCache).
///
/// For a page `d` bytes into a mapping, a `Guest { cpu_va }` target becomes `Guest { cpu_va + d }`
/// and a `GpuOnly { region, offset }` one `GpuOnly { region, offset + d }`; `SyncpointShim` stays
/// as it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Translation {
    /// Where the page leads.
    pub target: Target,
    /// The page's kind.
    pub kind: Kind,
}

/// One mapping of a [`GpuVm`], as [`GpuVm::mappings`] lists it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mapping {
    /// The GPU range.
    pub range: PageRange,
    /// Where its first page leads.
    pub target: Target,
    /// Its kind.
    pub kind: Kind,
}

/// Why a map was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum VmError {
    /// A `Guest` target's CPU address or a `GpuOnly` target's offset is not a multiple of
    /// [`PAGE_SIZE`].
    #[error("the target address or offset {value:#x} is not page aligned")]
    MisalignedTarget {
        /// The CPU address or the offset.
        value: u64,
    },
    /// A `Guest` target's CPU address or a `GpuOnly` target's offset, plus the range's size, is
    /// past the end of the 64-bit space.
    #[error("a target of {size:#x} bytes at {value:#x} runs past the end of the address space")]
    TargetOutOfRange {
        /// The CPU address or the offset.
        value: u64,
        /// The range's size.
        size: u64,
    },
    /// A `SyncpointShim` mapping is not exactly [`SHIM_PAGES`] pages.
    #[error(
        "the syncpoint shim is {expected:#x} bytes, not {size:#x} bytes",
        expected = SHIM_PAGES * PAGE_SIZE
    )]
    ShimSize {
        /// The range's size.
        size: u64,
    },
}

/// Why a GPU VM snapshot could not be written or read.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SnapshotError {
    /// Encoding failed.
    #[error("could not encode the GPU VM snapshot: {0}")]
    Encode(postcard::Error),
    /// The header or the body is not a well-formed encoding.
    #[error("could not decode the GPU VM snapshot: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`SNAPSHOT_MAGIC`].
    #[error("not a GPU VM snapshot (bad magic)")]
    BadMagic,
    /// The snapshot was written with another layout version.
    #[error(
        "GPU VM snapshot layout version {found} is not supported (this build reads {SNAPSHOT_VERSION})"
    )]
    UnsupportedVersion {
        /// The version found.
        found: u32,
    },
    /// Bytes follow the body.
    #[error("{count} unexpected bytes after the GPU VM snapshot")]
    TrailingBytes {
        /// How many.
        count: usize,
    },
    /// A mapping's GPU range is not a page range.
    #[error("mapping {index} of the GPU VM snapshot: {error}")]
    Range {
        /// The mapping's position in the list.
        index: usize,
        /// What is wrong with the range.
        error: MapError,
    },
    /// A mapping's target does not fit its range.
    #[error("mapping {index} of the GPU VM snapshot: {error}")]
    Target {
        /// The mapping's position in the list.
        index: usize,
        /// What is wrong with the target.
        error: VmError,
    },
    /// A mapping starts before the end of the one before it: the list overlaps or is out of order.
    #[error("mapping {index} of the GPU VM snapshot starts before the mapping before it ends")]
    Overlap {
        /// The mapping's position in the list.
        index: usize,
    },
}

/// A mapping in the table, keyed by its GPU start.
#[derive(Clone, Copy, Debug)]
struct Entry {
    /// One past its last GPU address.
    end: u64,
    /// Where its first page leads.
    target: Target,
    /// Its kind.
    kind: Kind,
}

/// The table, by GPU start; mappings never overlap.
type Table = BTreeMap<u64, Entry>;

/// `target` moved `delta` bytes further in.
///
/// Only called with `delta` below the size of a mapping whose target passed [`check_target`] (or
/// a part of one), so the sum is representable; it wraps rather than panic if that ever failed.
fn advance(target: Target, delta: u64) -> Target {
    match target {
        Target::Guest { cpu_va } => Target::Guest {
            cpu_va: cpu_va.wrapping_add(delta),
        },
        Target::GpuOnly { region, offset } => Target::GpuOnly {
            region,
            offset: offset.wrapping_add(delta),
        },
        Target::SyncpointShim => Target::SyncpointShim,
    }
}

/// Whether `target` fits a range of `size` bytes, checked in this order: a `Guest` CPU address or
/// a `GpuOnly` offset is page aligned ([`VmError::MisalignedTarget`]), and it plus `size` is
/// representable ([`VmError::TargetOutOfRange`]); a `SyncpointShim` range is [`SHIM_PAGES`] pages
/// ([`VmError::ShimSize`]).
fn check_target(target: Target, size: u64) -> Result<(), VmError> {
    let value = match target {
        Target::Guest { cpu_va } => cpu_va,
        Target::GpuOnly { offset, .. } => offset,
        Target::SyncpointShim => {
            // `SHIM_PAGES * PAGE_SIZE` is a small constant product.
            return if size == SHIM_PAGES * PAGE_SIZE {
                Ok(())
            } else {
                Err(VmError::ShimSize { size })
            };
        }
    };
    if !value.is_multiple_of(PAGE_SIZE) {
        return Err(VmError::MisalignedTarget { value });
    }
    if value.checked_add(size).is_none() {
        return Err(VmError::TargetOutOfRange { value, size });
    }
    Ok(())
}

/// Splits the mapping that holds `at` strictly inside it (not at its start) into the part below
/// `at` and the part from `at`, which leads as far into the old target as it starts into the old
/// mapping (NX2 `TrimMappings`).
fn split_at(table: &mut Table, at: u64) {
    let Some((&start, entry)) = table.range_mut(..at).next_back() else {
        return;
    };
    if entry.end <= at {
        return;
    }
    let upper = Entry {
        end: entry.end,
        target: advance(entry.target, at - start),
        kind: entry.kind,
    };
    entry.end = at;
    table.insert(at, upper);
}

/// The starts of the mappings that begin inside `range`, after [`split_at`] at both its ends.
fn starts_in(table: &Table, range: PageRange) -> Vec<u64> {
    table
        .range(range.start()..range.end())
        .map(|(&start, _)| start)
        .collect()
}

/// Removes every mapped page of `range` (NX2 `TrimMappings`) and returns how many there were.
fn trim(table: &mut Table, range: PageRange) -> u64 {
    split_at(table, range.start());
    split_at(table, range.end());
    let mut bytes = 0u64;
    for start in starts_in(table, range) {
        if let Some(entry) = table.remove(&start) {
            bytes = bytes.saturating_add(entry.end - start);
        }
    }
    bytes / PAGE_SIZE
}

/// Gives every mapped page of `range` `kind`, and returns how many there were.
fn set_kind(table: &mut Table, range: PageRange, kind: Kind) -> u64 {
    split_at(table, range.start());
    split_at(table, range.end());
    let mut bytes = 0u64;
    for (&start, entry) in table.range_mut(range.start()..range.end()) {
        entry.kind = kind;
        bytes = bytes.saturating_add(entry.end - start);
    }
    bytes / PAGE_SIZE
}

/// A GPU address space's translation table: GPU page ranges to a [`Target`] and a [`Kind`], with
/// a version (module docs, "Rules"), and the host pages behind its `GpuOnly` targets (module docs,
/// "The GPU-only store").
///
/// Shared between gsv and the GPU threads behind an `Arc`. Every method takes `&self`; changes
/// take a write lock and translations a read lock, held only for the table operation.
#[derive(Debug, Default)]
pub struct GpuVm {
    table: RwLock<Table>,
    generation: AtomicU64,
    gpu_only: GpuOnlyStore,
    /// The two-page registrations gsv lists (module docs, "The semaphore mirror page").
    semaphores: SemaphorePages,
}

/// What [`GpuVm::write_gpu_only`] found when it translated the page afresh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuOnlyWrite {
    /// The page leads to GPU-only memory, and the bytes are in the store.
    Written,
    /// The page no longer leads to GPU-only memory (a change landed since the caller's
    /// translation): nothing was written, and this is where it leads now, if anywhere.
    Elsewhere(Option<Translation>),
}

impl GpuVm {
    /// An empty table at version 0.
    pub fn new() -> Self {
        Self::default()
    }

    fn read(&self) -> RwLockReadGuard<'_, Table> {
        self.table.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, Table> {
        self.table.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// Moves the version. Called with the write lock held, after the change.
    fn bump(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Maps `range` to `target` with `kind`, replacing whatever it overlaps (module docs,
    /// "Rules"), and moves the version.
    ///
    /// The target is checked first, and a refused map changes nothing: a `Guest` CPU address or
    /// a `GpuOnly` offset must be page aligned ([`VmError::MisalignedTarget`]) and end within
    /// the 64-bit space ([`VmError::TargetOutOfRange`]); a `SyncpointShim` range must be
    /// [`SHIM_PAGES`] pages ([`VmError::ShimSize`]).
    pub fn map(&self, range: PageRange, target: Target, kind: Kind) -> Result<(), VmError> {
        check_target(target, range.size())?;
        let mut table = self.write();
        // conflict (decision 8, "the one a title ran on"): NX2's `MapAt` replaces what the range
        // overlaps; PR's `MapNative4K` refuses it (module docs, "Conflicts between the references").
        let replaced = trim(&mut table, range);
        table.insert(
            range.start(),
            Entry {
                end: range.end(),
                target,
                kind,
            },
        );
        self.bump();
        drop(table);
        tracing::debug!(
            start = range.start(),
            size = range.size(),
            ?target,
            kind = kind.0,
            replaced,
            "GPU VM map"
        );
        Ok(())
    }

    /// Applies an ordered update (module docs, "Rules"), moves the version, and returns how many
    /// mapped pages of its range it removed or gave the new kind.
    ///
    /// Never fails: the GPU applies it after gsv has replied to the guest, so anything gsv
    /// refuses it refuses at request time.
    pub fn apply(&self, update: &VmUpdate) -> u64 {
        let range = update.range();
        let mut table = self.write();
        // conflict (decision 8, "the larger"): any range, as NX2 takes it, not only one inside or
        // equal to a mapping, as PR checks; and kinds are kept, as PR keeps them (module docs,
        // "Conflicts between the references").
        let pages = match *update {
            VmUpdate::ModifyKind { kind, .. } => set_kind(&mut table, range, kind),
            VmUpdate::Unmap { .. }
            | VmUpdate::DestroyMemory { .. }
            | VmUpdate::DestroySparseRegion { .. } => trim(&mut table, range),
        };
        // The store frees under the table's write lock, before the version moves (module docs,
        // "The GPU-only store").
        let freed = match *update {
            VmUpdate::DestroySparseRegion { region, .. } => self.gpu_only.free_region(region),
            VmUpdate::Unmap { .. }
            | VmUpdate::ModifyKind { .. }
            | VmUpdate::DestroyMemory { .. } => 0,
        };
        self.bump();
        drop(table);
        tracing::debug!(?update, pages, freed, "GPU VM update applied");
        pages
    }

    /// The host pages behind this table's `GpuOnly` targets (module docs, "The GPU-only store").
    pub fn gpu_only(&self) -> &GpuOnlyStore {
        &self.gpu_only
    }

    /// Writes `data`, which lies within the page holding `va`, to the GPU-only memory that page
    /// leads to, translating the page afresh under the table's read lock and writing the store
    /// under it (module docs, "The GPU-only store").
    ///
    /// If the page no longer leads to GPU-only memory, nothing is written and the answer says
    /// where it leads now. The only error is an offset whose end is past the 64-bit offsets,
    /// which [`GpuVm::map`]'s target check rules out for a `data` within one page.
    pub(crate) fn write_gpu_only(
        &self,
        va: u64,
        data: &[u8],
    ) -> Result<GpuOnlyWrite, GpuOnlyRangeError> {
        let table = self.read();
        let within = va % PAGE_SIZE;
        let translation = translate_in(&table, va - within);
        match translation.map(|t| t.target) {
            Some(Target::GpuOnly { region, offset }) => {
                // The access from the page's own offset: its bytes before `data`, then `data`.
                let at = offset.checked_add(within).ok_or(GpuOnlyRangeError {
                    region,
                    offset,
                    len: within.saturating_add(u64::try_from(data.len()).unwrap_or(u64::MAX)),
                })?;
                self.gpu_only.write(region, at, data)?;
                drop(table);
                Ok(GpuOnlyWrite::Written)
            }
            Some(Target::Guest { .. } | Target::SyncpointShim) | None => {
                Ok(GpuOnlyWrite::Elsewhere(translation))
            }
        }
    }

    /// Lists `registration` as one whose second page is mirrored into its first (module docs,
    /// "The semaphore mirror page"): gsv calls it for each address space when it accepts a
    /// two-page 101, and for each two-page registration it holds when it gives a new or restored
    /// table. Each call counts once. The version does not move.
    pub fn add_semaphore_registration(&self, registration: SemaphoreRegistration) {
        self.semaphores.add(registration);
    }

    /// Lists `registration` once less: gsv calls it for each address space when a 102 destroys a
    /// two-page registration. Once no registration with its start is listed, releases there are
    /// no longer mirrored. A registration never added changes nothing. The version does not
    /// move.
    pub fn remove_semaphore_registration(&self, registration: SemaphoreRegistration) {
        self.semaphores.remove(registration);
    }

    /// Forgets every listed registration: gsv calls it when it destroys the address space. The
    /// version does not move.
    pub fn clear_semaphore_registrations(&self) {
        self.semaphores.clear();
    }

    /// The address a fenced release at `va` is also written at: `va` minus
    /// [`SEMAPHORE_MIRROR_OFFSET`] when `va` lies in the second page of a listed registration,
    /// otherwise `None` (module docs, "The semaphore mirror page"). Only `va` is looked at, not
    /// the release's length, as Eden-NX2's `Nx2SemaphoreMirror` does.
    pub fn semaphore_mirror(&self, va: u64) -> Option<u64> {
        self.semaphores.mirror_of(va)
    }

    /// Every mapping, in address order. Neighbouring mappings are not merged: a split leaves two.
    pub fn mappings(&self) -> Vec<Mapping> {
        mappings_of(&self.read())
    }

    /// Writes the version and the mappings (module docs, "Snapshot format").
    pub fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let table = self.read();
        let state = State {
            generation: self.generation.load(Ordering::Acquire),
            mappings: table
                .iter()
                .map(|(&start, entry)| SavedMapping {
                    start,
                    size: entry.end - start,
                    target: entry.target.into(),
                    kind: entry.kind.0,
                })
                .collect(),
        };
        drop(table);
        state.check()?;
        let header = Header {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
        };
        let mut bytes = postcard::to_allocvec(&header).map_err(SnapshotError::Encode)?;
        bytes.extend(postcard::to_allocvec(&state).map_err(SnapshotError::Encode)?);
        Ok(bytes)
    }

    /// A table holding what [`save`](Self::save) wrote. The snapshot is checked first (module
    /// docs, "Snapshot format").
    pub fn restore(bytes: &[u8]) -> Result<Self, SnapshotError> {
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
        let table = state.check()?;
        Ok(Self {
            table: RwLock::new(table),
            generation: AtomicU64::new(state.generation),
            gpu_only: GpuOnlyStore::new(),
            semaphores: SemaphorePages::default(),
        })
    }
}

/// The mappings of `table`, in address order.
fn mappings_of(table: &Table) -> Vec<Mapping> {
    table
        .iter()
        // Every entry's range was a `PageRange` when it was made, and splits keep it one, so
        // nothing is skipped.
        .filter_map(|(&start, entry)| {
            PageRange::new(start, entry.end - start)
                .ok()
                .map(|range| Mapping {
                    range,
                    target: entry.target,
                    kind: entry.kind,
                })
        })
        .collect()
}

impl Translate for GpuVm {
    type Target = Translation;

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn translate_page(&self, page: u64) -> Option<Translation> {
        translate_in(&self.read(), page)
    }
}

/// Where the page starting at `page` leads under `table`, if it is mapped.
fn translate_in(table: &Table, page: u64) -> Option<Translation> {
    let (&start, entry) = table.range(..=page).next_back()?;
    (page < entry.end).then(|| Translation {
        target: advance(entry.target, page - start),
        kind: entry.kind,
    })
}

/// The snapshot header.
#[derive(Serialize, Deserialize)]
struct Header {
    /// [`SNAPSHOT_MAGIC`].
    magic: [u8; 4],
    /// [`SNAPSHOT_VERSION`].
    version: u32,
}

/// A [`Target`] as the snapshot writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum SavedTarget {
    /// [`Target::Guest`].
    Guest {
        /// The CPU address.
        cpu_va: u64,
    },
    /// [`Target::GpuOnly`].
    GpuOnly {
        /// The region's id.
        region: u32,
        /// The offset.
        offset: u64,
    },
    /// [`Target::SyncpointShim`].
    SyncpointShim,
}

impl From<Target> for SavedTarget {
    fn from(target: Target) -> Self {
        match target {
            Target::Guest { cpu_va } => Self::Guest { cpu_va },
            Target::GpuOnly { region, offset } => Self::GpuOnly {
                region: region.0,
                offset,
            },
            Target::SyncpointShim => Self::SyncpointShim,
        }
    }
}

impl From<SavedTarget> for Target {
    fn from(target: SavedTarget) -> Self {
        match target {
            SavedTarget::Guest { cpu_va } => Self::Guest { cpu_va },
            SavedTarget::GpuOnly { region, offset } => Self::GpuOnly {
                region: RegionId(region),
                offset,
            },
            SavedTarget::SyncpointShim => Self::SyncpointShim,
        }
    }
}

/// One mapping as the snapshot writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SavedMapping {
    /// Its GPU start.
    start: u64,
    /// Its size in bytes.
    size: u64,
    /// Where its first page leads.
    target: SavedTarget,
    /// Its kind.
    kind: u8,
}

/// The snapshot body (module docs, "Snapshot format").
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct State {
    /// The version.
    generation: u64,
    /// The mappings, in address order.
    mappings: Vec<SavedMapping>,
}

impl State {
    /// The checks [`GpuVm::restore`] and [`GpuVm::save`] share, in the documented order; the
    /// table they describe.
    fn check(&self) -> Result<Table, SnapshotError> {
        let mut table = Table::new();
        let mut previous_end = 0u64;
        for (index, saved) in self.mappings.iter().enumerate() {
            let range = PageRange::new(saved.start, saved.size)
                .map_err(|error| SnapshotError::Range { index, error })?;
            let target = Target::from(saved.target);
            check_target(target, range.size())
                .map_err(|error| SnapshotError::Target { index, error })?;
            if range.start() < previous_end {
                return Err(SnapshotError::Overlap { index });
            }
            previous_end = range.end();
            table.insert(
                range.start(),
                Entry {
                    end: range.end(),
                    target,
                    kind: Kind(saved.kind),
                },
            );
        }
        Ok(table)
    }
}
