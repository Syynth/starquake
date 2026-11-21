//! The semaphore mirror page (gpu-protocol spec, "What this spec assumes of gsv"; graphics-server
//! spec, "The semaphore mirror page" and "What to verify", item 8; row 35 of
//! `docs/switch2-eden-diff-scoping.md`): the two-page registrations gsv lists in each address
//! table, and where a fenced release that lands in the second page of one is copied.
//!
//! **Tentative.** The owner's ruling of 2026-11-20 (decision log, "The semaphore mirror page:
//! copy releases, as Eden-NX2 does", status tentative) answers graphics-server verification
//! item 8 "no" for now: a fenced release that lands in the second page of a two-page
//! registration is also written at the same offset in its first page. A hardware trace may show
//! a different cause (a registration or a mapping starquake gets wrong); the copy is revisited
//! then.
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`. Behaviour ported from Eden-NX2 (Copyright 2026 Eden Emulator Project,
//! GPL-3.0-or-later), `@ 50fa86bb`:
//!
//! - `eden-nx2/src/core/hle/service/nx2_gsv.h`, `SubRegisterMemory`: a 101 registration of
//!   `start` to `last` is a semaphore region when `last - start + 1 == 0x2000`
//!   ([`SEMAPHORE_REGISTRATION_BYTES`]), and is added then (`Nx2AddSemaphoreRegion(start)`);
//!   `SubDestroyMemory` removes it at the 102 request when `size == 0x2000`
//!   (`Nx2RemoveSemaphoreRegion(address)`).
//! - `eden-nx2/src/video_core/nx2.h`, `Nx2SemaphoreMirror`: a release at `address` is mirrored
//!   when some region's start has `address >= start + 0x1000 && address < start + 0x2000`, to
//!   `address - 0x1000` ([`SEMAPHORE_MIRROR_OFFSET`]). Only the release's first byte is looked
//!   at.
//! - `eden-nx2/src/video_core/engines/puller.cpp`, `ProcessSemaphoreTriggerMethod`: the host
//!   class's `SEMAPHORED` release (`WriteLong`) asks `Nx2SemaphoreMirror` and, when it answers,
//!   queries the payload a second time at the mirror, with the same properties (so the same
//!   width, 4 bytes for `RELEASE_SIZE` set, 16 otherwise).
//!
//! # What lives here
//!
//! gsv decides which registrations are two pages long ([`SemaphoreRegistration::new`], which
//! only a range of exactly [`SEMAPHORE_REGISTRATION_BYTES`] that does not start at address 0
//! passes) and tells each address
//! table when one is made or destroyed ([`crate::vm::GpuVm::add_semaphore_registration`],
//! [`crate::vm::GpuVm::remove_semaphore_registration`]). The table answers
//! [`crate::vm::GpuVm::semaphore_mirror`], which the production memory
//! ([`crate::memory::VmMemory`]) passes on as [`crate::memory::GpuMemory::semaphore_mirror`];
//! the copy itself is the fence path's (`back::fence`, "The semaphore mirror page").
//!
//! The list is not a translation: changing it does not move the table's version, and it is not
//! in the table's snapshot ([`crate::vm::GpuVm::save`]). gsv gives a restored table its
//! registrations again when it rebuilds (its `gpu::saved` restore), as it gives a new table
//! every registration it holds (gpu 33).
//!
//! A lookup is one search of an ordered map (no scan of the list): a release at `va` is
//! mirrored when a registration starts in `va - 0x2000 + 1 ..= va - 0x1000`, which is the
//! condition of `Nx2SemaphoreMirror` above solved for the start. Every such registration
//! answers the same `va - 0x1000`, so which one is found does not matter.
//!
//! # Divergences from Eden-NX2
//!
//! Each is pinned by the test of `vm/mirror/tests.rs` named:
//!
//! - **Registrations are counted.** NX2's `Nx2AddSemaphoreRegion` keeps one slot per start
//!   (an add of a start already there does nothing) and `Nx2RemoveSemaphoreRegion` clears every
//!   slot holding the start, so destroying one of two registrations with the same start stops
//!   the mirror for both. Here each start is counted, and the mirror stays while any
//!   registration with that start is held
//!   (`two_registrations_at_one_start_mirror_until_both_are_removed`).
//! - **No cap.** NX2 keeps at most 32 regions (`Nx2MaxSemaphoreRegions`) and drops an add past
//!   them; here a table lists every two-page registration gsv holds, which gsv bounds
//!   (`MAX_PIECES` of its `gpu::memory`) (`more_than_32_registrations_all_mirror`).
//! - **One list per address table.** NX2's list is one per emulator; here each table holds the
//!   registrations, which gsv maps into every address space of the process (row 30), so a
//!   process's tables answer the same. A table gsv destroys is emptied
//!   ([`crate::vm::GpuVm::clear_semaphore_registrations`]): gsv's address space destroy. This
//!   one is pinned by gsv's tests, not by a test here: `gpu::tests::mirror::
//!   a_two_page_registration_is_listed_in_every_space` and
//!   `gpu::tests::mirror::a_destroyed_space_lists_nothing` (`crates/gsv`).
//!
//! # As in Eden-NX2
//!
//! - **A registration at address 0 is not listed.** In NX2, 0 is an empty slot:
//!   `Nx2AddSemaphoreRegion(0)` finds an empty slot already "holding" 0 and stores nothing,
//!   `Nx2SemaphoreMirror` skips a slot whose start is 0, and its answer 0 means "no mirror". So
//!   [`SemaphoreRegistration::new`] refuses a range that starts at 0, and gsv lists nothing for
//!   such a 101 (`a_registration_at_address_0_is_not_a_semaphore_registration`).
//! - **Removing a start never added changes nothing**, as in NX2 (its `compare_exchange`
//!   finds no slot) (`removing_a_start_never_added_changes_nothing`).

use std::collections::BTreeMap;
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use starquake_memory::{PAGE_SIZE, PageRange};

#[cfg(test)]
mod tests;

/// The size of a registration whose second page is mirrored into its first: NX2's
/// `SubRegisterMemory` adds a region when `last - start + 1 == 0x2000`
/// (`eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb). Two pages.
pub const SEMAPHORE_REGISTRATION_BYTES: u64 = 2 * PAGE_SIZE;

/// How far below a release its mirror lies, and where a registration's second page starts:
/// NX2's `Nx2SemaphoreMirror` tests `address >= start + 0x1000` and answers `address - 0x1000`
/// (`eden-nx2/src/video_core/nx2.h` @ 50fa86bb). One page.
pub const SEMAPHORE_MIRROR_OFFSET: u64 = PAGE_SIZE;

/// A registration exactly [`SEMAPHORE_REGISTRATION_BYTES`] long, whose second page is mirrored
/// into its first (module docs). Only [`SemaphoreRegistration::new`] makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SemaphoreRegistration {
    start: u64,
}

impl SemaphoreRegistration {
    /// The registration of `range` when it is exactly [`SEMAPHORE_REGISTRATION_BYTES`] long (NX2
    /// `SubRegisterMemory`: `last - start + 1 == 0x2000`) and does not start at address 0 (NX2's
    /// empty slot, which `Nx2AddSemaphoreRegion` never stores and `Nx2SemaphoreMirror` skips;
    /// module docs, "As in Eden-NX2"), or `None`.
    #[must_use]
    pub fn new(range: PageRange) -> Option<Self> {
        (range.size() == SEMAPHORE_REGISTRATION_BYTES && range.start() != 0).then_some(Self {
            start: range.start(),
        })
    }

    /// Its first address.
    #[must_use]
    pub const fn start(self) -> u64 {
        self.start
    }
}

/// The two-page registrations one address table lists, by start, each with how many
/// registrations share that start (module docs).
#[derive(Debug, Default)]
pub(crate) struct SemaphorePages {
    starts: RwLock<BTreeMap<u64, u64>>,
}

impl SemaphorePages {
    fn read(&self) -> RwLockReadGuard<'_, BTreeMap<u64, u64>> {
        self.starts.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, BTreeMap<u64, u64>> {
        self.starts.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// Lists `registration` once more. The count cannot reach `u64::MAX`: gsv holds far fewer
    /// registrations.
    pub(crate) fn add(&self, registration: SemaphoreRegistration) {
        let mut starts = self.write();
        let count = starts.entry(registration.start).or_insert(0);
        *count = count.saturating_add(1);
    }

    /// Lists `registration` once less; a start no longer held by any registration stops
    /// mirroring. A start never added changes nothing.
    pub(crate) fn remove(&self, registration: SemaphoreRegistration) {
        let mut starts = self.write();
        if let Some(count) = starts.get_mut(&registration.start) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                starts.remove(&registration.start);
            }
        }
    }

    /// Forgets every registration.
    pub(crate) fn clear(&self) {
        self.write().clear();
    }

    /// The address a release at `va` is also written at, if `va` lies in the second page of a
    /// listed registration (module docs).
    pub(crate) fn mirror_of(&self, va: u64) -> Option<u64> {
        mirror_in(&self.read(), va)
    }
}

/// [`SemaphorePages::mirror_of`] over `starts`: `va - 0x1000` when a start `s` has
/// `s + 0x1000 <= va < s + 0x2000` (NX2 `Nx2SemaphoreMirror`).
fn mirror_in(starts: &BTreeMap<u64, u64>, va: u64) -> Option<u64> {
    let mirror = va.checked_sub(SEMAPHORE_MIRROR_OFFSET)?;
    // The greatest start at or below `va - 0x1000`: if it is not above `va - 0x2000`, no smaller
    // one is either.
    let (&start, _) = starts.range(..=mirror).next_back()?;
    let end = start.checked_add(SEMAPHORE_REGISTRATION_BYTES);
    // A start whose end would pass the 64-bit space is not a `PageRange`'s, but the comparison
    // still holds: every `va` lies below such an end.
    end.is_none_or(|end| va < end).then_some(mirror)
}
