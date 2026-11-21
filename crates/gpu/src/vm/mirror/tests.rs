//! Tests of the semaphore mirror list: the constants against Eden-NX2's literals, the
//! registration size at its edges, the lookup at every edge of the second page, and the
//! divergences the module docs list.

use std::collections::BTreeMap;

use starquake_memory::{PAGE_SIZE, PageRange};

use super::*;
use crate::interface::{Kind, Target};
use crate::memory::Translate;
use crate::vm::GpuVm;

/// A two-page registration's start: above 32 bits, so a truncated address misses it.
const S: u64 = 0x12_3456_7000;

fn range(start: u64, size: u64) -> PageRange {
    PageRange::new(start, size).unwrap()
}

fn two_pages(start: u64) -> SemaphoreRegistration {
    SemaphoreRegistration::new(range(start, SEMAPHORE_REGISTRATION_BYTES)).unwrap()
}

fn listing(starts: &[u64]) -> SemaphorePages {
    let pages = SemaphorePages::default();
    for &start in starts {
        pages.add(two_pages(start));
    }
    pages
}

#[test]
fn the_constants_are_nx2_s_literals() {
    // `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb, `SubRegisterMemory`:
    // `last - start + 1 == 0x2000`; `eden-nx2/src/video_core/nx2.h`, `Nx2SemaphoreMirror`:
    // `address >= start + 0x1000 && address < start + 0x2000`, `return address - 0x1000`.
    assert_eq!(SEMAPHORE_REGISTRATION_BYTES, 0x2000);
    assert_eq!(SEMAPHORE_MIRROR_OFFSET, 0x1000);
}

#[test]
fn only_a_registration_of_exactly_two_pages_is_a_semaphore_registration() {
    let bytes = SEMAPHORE_REGISTRATION_BYTES;
    assert_eq!(
        SemaphoreRegistration::new(range(S, bytes - PAGE_SIZE)),
        None
    );
    assert_eq!(
        SemaphoreRegistration::new(range(S, bytes)).map(SemaphoreRegistration::start),
        Some(S)
    );
    assert_eq!(
        SemaphoreRegistration::new(range(S, bytes + PAGE_SIZE)),
        None
    );
}

/// NX2's empty slot: a two-page registration at address 0 is not one, and the lowest start that
/// is, one page up, is (module docs, "As in Eden-NX2").
#[test]
fn a_registration_at_address_0_is_not_a_semaphore_registration() {
    let bytes = SEMAPHORE_REGISTRATION_BYTES;
    assert_eq!(SemaphoreRegistration::new(range(0, bytes)), None);
    assert_eq!(
        SemaphoreRegistration::new(range(PAGE_SIZE, bytes)).map(SemaphoreRegistration::start),
        Some(PAGE_SIZE)
    );
}

/// Each edge of the second page, on both sides, and a release in the first page.
#[test]
fn a_release_is_mirrored_from_exactly_the_second_page() {
    let pages = listing(&[S]);
    let second = S + SEMAPHORE_MIRROR_OFFSET;
    let end = S + SEMAPHORE_REGISTRATION_BYTES;
    for (va, expected) in [
        (S, None),
        (S + 0x10, None),
        (second - 1, None),
        (second, Some(S)),
        (second + 0x10, Some(S + 0x10)),
        (end - 1, Some(S + SEMAPHORE_MIRROR_OFFSET - 1)),
        (end, None),
        (S - 1, None),
    ] {
        assert_eq!(pages.mirror_of(va), expected, "{va:#x}");
    }
}

/// A second two-page registration that starts exactly where the first ends: its first page is
/// not mirrored (into the first registration's second page), and its second page is mirrored
/// into its own first page.
#[test]
fn a_neighbour_starting_at_the_end_bound_keeps_its_own_pages() {
    let next = S + SEMAPHORE_REGISTRATION_BYTES;
    let pages = listing(&[S, next]);
    assert_eq!(
        pages.mirror_of(next - 4),
        Some(next - 4 - SEMAPHORE_MIRROR_OFFSET)
    );
    assert_eq!(pages.mirror_of(next), None);
    assert_eq!(pages.mirror_of(next + 0x10), None);
    assert_eq!(
        pages.mirror_of(next + SEMAPHORE_MIRROR_OFFSET + 0x10),
        Some(next + 0x10)
    );
}

/// Registrations a page apart: the second page of the lower one is the first page of the upper
/// one, and is mirrored; the upper one's second page is mirrored too. The lookup takes the
/// greatest start at or below the mirror, so one that looked at the smallest would miss the
/// upper registration.
#[test]
fn overlapping_registrations_each_mirror_their_second_page() {
    let upper = S + PAGE_SIZE;
    let pages = listing(&[S, upper]);
    assert_eq!(pages.mirror_of(S + 0x1800), Some(S + 0x800));
    assert_eq!(pages.mirror_of(upper + 0x1800), Some(upper + 0x800));
    assert_eq!(pages.mirror_of(upper + SEMAPHORE_REGISTRATION_BYTES), None);
}

#[test]
fn two_registrations_at_one_start_mirror_until_both_are_removed() {
    let pages = listing(&[S, S]);
    let va = S + SEMAPHORE_MIRROR_OFFSET + 0x10;
    pages.remove(two_pages(S));
    assert_eq!(pages.mirror_of(va), Some(S + 0x10));
    pages.remove(two_pages(S));
    assert_eq!(pages.mirror_of(va), None);
    // Adding again after the last removal mirrors again.
    pages.add(two_pages(S));
    assert_eq!(pages.mirror_of(va), Some(S + 0x10));
}

#[test]
fn removing_a_start_never_added_changes_nothing() {
    let other = S + 4 * PAGE_SIZE;
    let pages = listing(&[S]);
    pages.remove(two_pages(other));
    assert_eq!(pages.mirror_of(S + 0x1010), Some(S + 0x10));
    assert_eq!(pages.mirror_of(other + 0x1010), None);
    // And it did not leave a negative count behind: one add makes `other` mirror.
    pages.add(two_pages(other));
    assert_eq!(pages.mirror_of(other + 0x1010), Some(other + 0x10));
}

#[test]
fn clearing_forgets_every_registration() {
    let pages = listing(&[S, S, S + 4 * PAGE_SIZE]);
    pages.clear();
    assert_eq!(pages.mirror_of(S + 0x1010), None);
    assert_eq!(pages.mirror_of(S + 4 * PAGE_SIZE + 0x1010), None);
}

/// NX2 keeps 32 (`Nx2MaxSemaphoreRegions`); here the 33rd mirrors too.
#[test]
fn more_than_32_registrations_all_mirror() {
    let starts: Vec<u64> = (0..33)
        .map(|n| S + n * SEMAPHORE_REGISTRATION_BYTES)
        .collect();
    let pages = listing(&starts);
    for start in starts {
        assert_eq!(
            pages.mirror_of(start + 0x1004),
            Some(start + 4),
            "{start:#x}"
        );
    }
}

/// The lowest registration [`SemaphoreRegistration::new`] allows (one page up: address 0 is
/// NX2's empty slot) and the highest a `PageRange` allows: no lookup wraps.
#[test]
fn lookups_at_the_ends_of_the_address_space_do_not_wrap() {
    let low = listing(&[PAGE_SIZE]);
    let second = PAGE_SIZE + SEMAPHORE_MIRROR_OFFSET;
    assert_eq!(low.mirror_of(SEMAPHORE_MIRROR_OFFSET - 1), None);
    assert_eq!(low.mirror_of(second - 1), None);
    assert_eq!(low.mirror_of(second), Some(PAGE_SIZE));
    assert_eq!(low.mirror_of(u64::MAX), None);

    // The highest start of a two-page `PageRange`: its end, the last page's start, is
    // representable.
    let top = u64::MAX - (SEMAPHORE_REGISTRATION_BYTES + PAGE_SIZE) + 1;
    let high = listing(&[top]);
    let end = top + SEMAPHORE_REGISTRATION_BYTES;
    assert_eq!(
        high.mirror_of(end - 1),
        Some(end - 1 - SEMAPHORE_MIRROR_OFFSET)
    );
    assert_eq!(high.mirror_of(end), None);
    assert_eq!(high.mirror_of(u64::MAX), None);
}

/// A start whose end would pass the 64-bit space (no `PageRange` has one) still answers by the
/// reference's arithmetic: every address lies below that end.
#[test]
fn a_start_whose_end_overflows_still_answers() {
    let start = u64::MAX - SEMAPHORE_REGISTRATION_BYTES + 1;
    let starts = BTreeMap::from([(start, 1)]);
    assert_eq!(
        mirror_in(&starts, u64::MAX),
        Some(u64::MAX - SEMAPHORE_MIRROR_OFFSET)
    );
    assert_eq!(
        mirror_in(&starts, start + SEMAPHORE_MIRROR_OFFSET - 1),
        None
    );
}

/// The table's list: what gsv calls, answered by the table; not a translation, so the version
/// does not move and the snapshot does not hold it.
#[test]
fn the_table_lists_registrations_outside_its_version_and_snapshot() {
    let vm = GpuVm::new();
    vm.map(
        range(S, SEMAPHORE_REGISTRATION_BYTES),
        Target::Guest { cpu_va: S },
        Kind(0),
    )
    .unwrap();
    let saved = vm.save().unwrap();
    let version = vm.generation();

    vm.add_semaphore_registration(two_pages(S));
    assert_eq!(vm.semaphore_mirror(S + 0x1010), Some(S + 0x10));
    assert_eq!(vm.generation(), version);
    assert_eq!(vm.save().unwrap(), saved);
    let restored = GpuVm::restore(&saved).unwrap();
    assert_eq!(restored.semaphore_mirror(S + 0x1010), None);

    vm.remove_semaphore_registration(two_pages(S));
    assert_eq!(vm.semaphore_mirror(S + 0x1010), None);
    vm.add_semaphore_registration(two_pages(S));
    vm.clear_semaphore_registrations();
    assert_eq!(vm.semaphore_mirror(S + 0x1010), None);
    assert_eq!(vm.generation(), version);
}

#[test]
fn a_new_table_lists_nothing() {
    assert_eq!(GpuVm::new().semaphore_mirror(S + 0x1010), None);
}
