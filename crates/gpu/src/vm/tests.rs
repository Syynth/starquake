use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};
use starquake_memory::{MapError, PAGE_SIZE, PageRange};

use super::*;
use crate::interface::{Kind, RegionId, Target, VmUpdate};
use crate::memory::{Translate, TranslationCache};

/// The GPU page every test's numbering starts from: page `n` is at `BASE + n * PAGE_SIZE`.
const BASE: u64 = 0x10_0000;

/// The GPU address of test page `n`.
fn page(n: u64) -> u64 {
    BASE + n * PAGE_SIZE
}

/// Test pages `first..first + count`.
fn pages(first: u64, count: u64) -> PageRange {
    PageRange::new(page(first), count * PAGE_SIZE).unwrap()
}

fn guest(cpu_va: u64) -> Target {
    Target::Guest { cpu_va }
}

fn gpu_only(region: u32, offset: u64) -> Target {
    Target::GpuOnly {
        region: RegionId(region),
        offset,
    }
}

fn at(target: Target, kind: u8) -> Translation {
    Translation {
        target,
        kind: Kind(kind),
    }
}

fn translate(vm: &GpuVm, n: u64) -> Option<Translation> {
    vm.translate_page(page(n))
}

// --- maps and translation ---

#[test]
fn an_empty_table_translates_nothing() {
    let vm = GpuVm::new();
    assert_eq!(vm.generation(), 0);
    assert_eq!(translate(&vm, 0), None);
    assert_eq!(vm.translate_page(0), None);
    assert_eq!(vm.mappings(), []);
}

#[test]
fn each_page_of_a_mapping_leads_as_far_into_its_target() {
    let vm = GpuVm::new();
    vm.map(pages(2, 3), guest(0x8000_0000), Kind(0x11)).unwrap();
    vm.map(pages(10, 2), gpu_only(4, 0x3000), Kind(0x22))
        .unwrap();
    vm.map(pages(20, SHIM_PAGES), Target::SyncpointShim, Kind(0x33))
        .unwrap();

    // Guest: CPU address plus the page's offset into the mapping.
    assert_eq!(translate(&vm, 1), None, "the page below the start");
    assert_eq!(translate(&vm, 2), Some(at(guest(0x8000_0000), 0x11)));
    assert_eq!(
        translate(&vm, 3),
        Some(at(guest(0x8000_0000 + PAGE_SIZE), 0x11))
    );
    assert_eq!(
        translate(&vm, 4),
        Some(at(guest(0x8000_0000 + 2 * PAGE_SIZE), 0x11))
    );
    assert_eq!(translate(&vm, 5), None, "the page at the end bound");
    // GpuOnly: offset plus the page's offset.
    assert_eq!(translate(&vm, 10), Some(at(gpu_only(4, 0x3000), 0x22)));
    assert_eq!(
        translate(&vm, 11),
        Some(at(gpu_only(4, 0x3000 + PAGE_SIZE), 0x22))
    );
    assert_eq!(translate(&vm, 12), None);
    // The shim stays the shim.
    assert_eq!(translate(&vm, 19), None);
    assert_eq!(translate(&vm, 20), Some(at(Target::SyncpointShim, 0x33)));
    assert_eq!(translate(&vm, 21), None);

    assert_eq!(vm.generation(), 3);
    assert_eq!(
        vm.mappings(),
        [
            Mapping {
                range: pages(2, 3),
                target: guest(0x8000_0000),
                kind: Kind(0x11)
            },
            Mapping {
                range: pages(10, 2),
                target: gpu_only(4, 0x3000),
                kind: Kind(0x22)
            },
            Mapping {
                range: pages(20, 1),
                target: Target::SyncpointShim,
                kind: Kind(0x33)
            },
        ]
    );
}

#[test]
fn a_map_ending_at_a_neighbours_start_leaves_it_alone() {
    let vm = GpuVm::new();
    vm.map(pages(5, 2), guest(0x9000_0000), Kind(1)).unwrap();
    // Ends exactly where the neighbour starts, and starts exactly where it ends.
    vm.map(pages(3, 2), guest(0x7000_0000), Kind(2)).unwrap();
    vm.map(pages(7, 1), gpu_only(1, 0), Kind(3)).unwrap();
    assert_eq!(translate(&vm, 5), Some(at(guest(0x9000_0000), 1)));
    assert_eq!(
        translate(&vm, 6),
        Some(at(guest(0x9000_0000 + PAGE_SIZE), 1))
    );
    assert_eq!(
        translate(&vm, 4),
        Some(at(guest(0x7000_0000 + PAGE_SIZE), 2))
    );
    assert_eq!(translate(&vm, 7), Some(at(gpu_only(1, 0), 3)));
    assert_eq!(vm.mappings().len(), 3);
}

#[test]
fn a_map_replaces_what_it_overlaps() {
    // NX2 `MapAt`: `TrimMappings` keeps the parts of older mappings outside the new range, the
    // upper part leading as far into its old target as it starts into the old mapping.
    let vm = GpuVm::new();
    vm.map(pages(0, 4), gpu_only(9, 0x10_0000), Kind(1))
        .unwrap();
    vm.map(pages(6, 4), guest(0x4000_0000), Kind(2)).unwrap();
    vm.map(pages(12, 1), guest(0x5000_0000), Kind(4)).unwrap();
    // From inside the first to inside the second, over the hole at 4 and 5.
    vm.map(pages(2, 6), guest(0x6000_0000), Kind(3)).unwrap();

    assert_eq!(
        translate(&vm, 1),
        Some(at(gpu_only(9, 0x10_0000 + PAGE_SIZE), 1))
    );
    for n in 2..8 {
        assert_eq!(
            translate(&vm, n),
            Some(at(guest(0x6000_0000 + (n - 2) * PAGE_SIZE), 3))
        );
    }
    assert_eq!(
        translate(&vm, 8),
        Some(at(guest(0x4000_0000 + 2 * PAGE_SIZE), 2))
    );
    assert_eq!(
        translate(&vm, 9),
        Some(at(guest(0x4000_0000 + 3 * PAGE_SIZE), 2))
    );
    assert_eq!(translate(&vm, 12), Some(at(guest(0x5000_0000), 4)));
    assert_eq!(
        vm.mappings(),
        [
            Mapping {
                range: pages(0, 2),
                target: gpu_only(9, 0x10_0000),
                kind: Kind(1)
            },
            Mapping {
                range: pages(2, 6),
                target: guest(0x6000_0000),
                kind: Kind(3)
            },
            Mapping {
                range: pages(8, 2),
                target: guest(0x4000_0000 + 2 * PAGE_SIZE),
                kind: Kind(2)
            },
            Mapping {
                range: pages(12, 1),
                target: guest(0x5000_0000),
                kind: Kind(4)
            },
        ]
    );

    // A map strictly inside one mapping leaves both of its ends.
    vm.map(pages(3, SHIM_PAGES), Target::SyncpointShim, Kind(5))
        .unwrap();
    assert_eq!(translate(&vm, 2), Some(at(guest(0x6000_0000), 3)));
    assert_eq!(translate(&vm, 3), Some(at(Target::SyncpointShim, 5)));
    assert_eq!(
        translate(&vm, 3 + SHIM_PAGES),
        Some(at(guest(0x6000_0000 + (1 + SHIM_PAGES) * PAGE_SIZE), 3))
    );
}

#[test]
fn the_shim_is_exactly_one_page() {
    let vm = GpuVm::new();
    let two = PageRange::new(page(0), (SHIM_PAGES + 1) * PAGE_SIZE).unwrap();
    assert_eq!(
        vm.map(two, Target::SyncpointShim, Kind(0)),
        Err(VmError::ShimSize {
            size: (SHIM_PAGES + 1) * PAGE_SIZE
        })
    );
    assert_eq!(
        vm.generation(),
        0,
        "a refused map does not move the version"
    );
    assert_eq!(vm.mappings(), [], "nor change the table");
    let one = PageRange::new(page(0), SHIM_PAGES * PAGE_SIZE).unwrap();
    assert_eq!(vm.map(one, Target::SyncpointShim, Kind(0)), Ok(()));
    assert_eq!(vm.generation(), 1);
}

#[test]
fn a_target_must_be_aligned_and_end_in_range() {
    let one = pages(0, 1);
    let two = pages(4, 2);
    // The last page of the 64-bit space: a one-page target there ends at 2^64.
    let last_page = u64::MAX - (PAGE_SIZE - 1);
    let makers: [fn(u64) -> Target; 2] = [guest, |offset| gpu_only(1, offset)];
    for make in makers {
        let vm = GpuVm::new();
        assert_eq!(
            vm.map(one, make(PAGE_SIZE + 1), Kind(0)),
            Err(VmError::MisalignedTarget {
                value: PAGE_SIZE + 1
            })
        );
        assert_eq!(
            vm.map(one, make(last_page), Kind(0)),
            Err(VmError::TargetOutOfRange {
                value: last_page,
                size: PAGE_SIZE
            })
        );
        // Two pages from one below the last end exactly at 2^64 too.
        assert_eq!(
            vm.map(two, make(last_page - PAGE_SIZE), Kind(0)),
            Err(VmError::TargetOutOfRange {
                value: last_page - PAGE_SIZE,
                size: 2 * PAGE_SIZE
            })
        );
        // Both faults at once: the alignment is checked first.
        assert_eq!(
            vm.map(one, make(u64::MAX), Kind(0)),
            Err(VmError::MisalignedTarget { value: u64::MAX })
        );
        assert_eq!(vm.generation(), 0);
        // One page further down ends exactly at the last representable page boundary.
        assert_eq!(vm.map(one, make(last_page - PAGE_SIZE), Kind(0)), Ok(()));
        assert_eq!(
            vm.map(two, make(last_page - 2 * PAGE_SIZE), Kind(0)),
            Ok(())
        );
        assert_eq!(
            translate(&vm, 5),
            Some(Translation {
                target: make(last_page - PAGE_SIZE),
                kind: Kind(0)
            })
        );
    }
}

#[test]
fn a_gpu_range_at_the_top_of_the_space_translates() {
    // The highest page a `PageRange` can hold ends at the last page boundary.
    let vm = GpuVm::new();
    let top = u64::MAX - (PAGE_SIZE - 1) - PAGE_SIZE;
    let range = PageRange::new(top, PAGE_SIZE).unwrap();
    vm.map(range, guest(0), Kind(7)).unwrap();
    assert_eq!(vm.translate_page(top), Some(at(guest(0), 7)));
    assert_eq!(vm.translate_page(top - PAGE_SIZE), None);
    assert_eq!(vm.translate_page(top + PAGE_SIZE), None);
}

// --- ordered updates ---

#[test]
fn an_unmap_splits_a_mapping_and_skips_holes() {
    let vm = GpuVm::new();
    vm.map(pages(0, 6), guest(0x1000_0000), Kind(1)).unwrap();
    vm.map(pages(8, 2), gpu_only(2, 0), Kind(2)).unwrap();
    vm.map(pages(10, 1), guest(0x2000_0000), Kind(3)).unwrap();
    // Pages 2 and 3 of the first: it splits in two.
    assert_eq!(vm.apply(&VmUpdate::Unmap { range: pages(2, 2) }), 2);
    assert_eq!(
        translate(&vm, 1),
        Some(at(guest(0x1000_0000 + PAGE_SIZE), 1))
    );
    assert_eq!(translate(&vm, 2), None);
    assert_eq!(translate(&vm, 3), None);
    assert_eq!(
        translate(&vm, 4),
        Some(at(guest(0x1000_0000 + 4 * PAGE_SIZE), 1))
    );
    // From the second half's last page over the hole at 6 and 7 into the GpuOnly mapping, ending
    // exactly where the next one starts: 1 + 2 pages mapped.
    assert_eq!(vm.apply(&VmUpdate::Unmap { range: pages(5, 5) }), 3);
    assert_eq!(
        translate(&vm, 4),
        Some(at(guest(0x1000_0000 + 4 * PAGE_SIZE), 1))
    );
    for n in 5..10 {
        assert_eq!(translate(&vm, n), None);
    }
    assert_eq!(translate(&vm, 10), Some(at(guest(0x2000_0000), 3)));
    assert_eq!(
        vm.mappings(),
        [
            Mapping {
                range: pages(0, 2),
                target: guest(0x1000_0000),
                kind: Kind(1)
            },
            Mapping {
                range: pages(4, 1),
                target: guest(0x1000_0000 + 4 * PAGE_SIZE),
                kind: Kind(1)
            },
            Mapping {
                range: pages(10, 1),
                target: guest(0x2000_0000),
                kind: Kind(3)
            },
        ]
    );
}

#[test]
fn every_removal_takes_out_the_same_pages() {
    let removals = [
        VmUpdate::Unmap { range: pages(1, 2) },
        VmUpdate::DestroyMemory { range: pages(1, 2) },
        VmUpdate::DestroySparseRegion {
            range: pages(1, 2),
            region: RegionId(6),
        },
    ];
    for removal in removals {
        let vm = GpuVm::new();
        vm.map(pages(0, 4), gpu_only(6, 0), Kind(1)).unwrap();
        assert_eq!(vm.apply(&removal), 2, "{removal:?}");
        assert_eq!(translate(&vm, 0), Some(at(gpu_only(6, 0), 1)));
        assert_eq!(translate(&vm, 1), None);
        assert_eq!(translate(&vm, 2), None);
        assert_eq!(translate(&vm, 3), Some(at(gpu_only(6, 3 * PAGE_SIZE), 1)));
        assert_eq!(vm.generation(), 2);
    }
}

#[test]
fn a_kind_change_spans_mappings_and_skips_holes() {
    let vm = GpuVm::new();
    vm.map(pages(0, 3), guest(0x3000_0000), Kind(1)).unwrap();
    vm.map(pages(5, 3), gpu_only(1, 0x8000), Kind(2)).unwrap();
    vm.map(pages(8, 1), guest(0x4000_0000), Kind(4)).unwrap();
    // From the first's last two pages, over the hole at 3 and 4, to the second's end, which is
    // exactly where the third starts.
    let change = VmUpdate::ModifyKind {
        range: pages(1, 7),
        kind: Kind(9),
    };
    assert_eq!(vm.apply(&change), 5);
    assert_eq!(translate(&vm, 0), Some(at(guest(0x3000_0000), 1)));
    assert_eq!(
        translate(&vm, 1),
        Some(at(guest(0x3000_0000 + PAGE_SIZE), 9))
    );
    assert_eq!(
        translate(&vm, 2),
        Some(at(guest(0x3000_0000 + 2 * PAGE_SIZE), 9))
    );
    assert_eq!(translate(&vm, 3), None, "a hole stays a hole");
    assert_eq!(translate(&vm, 4), None);
    assert_eq!(translate(&vm, 5), Some(at(gpu_only(1, 0x8000), 9)));
    assert_eq!(
        translate(&vm, 7),
        Some(at(gpu_only(1, 0x8000 + 2 * PAGE_SIZE), 9))
    );
    assert_eq!(translate(&vm, 8), Some(at(guest(0x4000_0000), 4)));
    assert_eq!(vm.mappings().len(), 4, "only the first split");

    // Strictly inside one mapping: both ends keep the old kind.
    let inner = VmUpdate::ModifyKind {
        range: pages(6, 1),
        kind: Kind(3),
    };
    assert_eq!(vm.apply(&inner), 1);
    assert_eq!(translate(&vm, 5), Some(at(gpu_only(1, 0x8000), 9)));
    assert_eq!(
        translate(&vm, 6),
        Some(at(gpu_only(1, 0x8000 + PAGE_SIZE), 3))
    );
    assert_eq!(
        translate(&vm, 7),
        Some(at(gpu_only(1, 0x8000 + 2 * PAGE_SIZE), 9))
    );
}

#[test]
fn an_update_of_a_hole_changes_nothing_but_the_version() {
    let vm = GpuVm::new();
    vm.map(pages(0, 1), guest(0x1000), Kind(1)).unwrap();
    vm.map(pages(2, 1), guest(0x2000), Kind(1)).unwrap();
    let before = vm.mappings();
    for update in [
        VmUpdate::Unmap { range: pages(1, 1) },
        VmUpdate::ModifyKind {
            range: pages(1, 1),
            kind: Kind(5),
        },
    ] {
        let version = vm.generation();
        assert_eq!(vm.apply(&update), 0);
        assert_eq!(vm.generation(), version + 1);
        assert_eq!(vm.mappings(), before);
    }
}

#[test]
fn an_update_names_its_range() {
    let range = pages(3, 2);
    assert_eq!(VmUpdate::Unmap { range }.range(), range);
    assert_eq!(
        VmUpdate::ModifyKind {
            range,
            kind: Kind(1)
        }
        .range(),
        range
    );
    assert_eq!(VmUpdate::DestroyMemory { range }.range(), range);
    assert_eq!(
        VmUpdate::DestroySparseRegion {
            range,
            region: RegionId(1)
        }
        .range(),
        range
    );
}

// --- the version and the translation cache ---

#[test]
fn a_cache_over_the_table_drops_what_an_update_changed() {
    let vm = Arc::new(GpuVm::new());
    vm.map(pages(0, 2), guest(0x7000_0000), Kind(1)).unwrap();
    let mut cache = TranslationCache::new(Arc::clone(&vm));
    assert_eq!(
        cache.translate(page(1) + 0x10),
        Some(at(guest(0x7000_1000), 1))
    );
    vm.apply(&VmUpdate::ModifyKind {
        range: pages(1, 1),
        kind: Kind(2),
    });
    assert_eq!(
        cache.translate(page(1) + 0x10),
        Some(at(guest(0x7000_1000), 2))
    );
    vm.apply(&VmUpdate::Unmap { range: pages(1, 1) });
    assert_eq!(cache.translate(page(1)), None);
    assert_eq!(cache.translate(page(0)), Some(at(guest(0x7000_0000), 1)));
}

#[test]
fn a_reader_that_sees_a_version_sees_its_maps() {
    // The writer maps page k as its (k+1)th change; a reader that reads version v and then
    // translates must find page v - 1 mapped (and every page below it, checked at the end). This
    // catches a version moved before the change only when a reader lands between the two, so it
    // is a check, not a proof.
    const COUNT: u64 = 20_000;
    const READERS: usize = 3;
    let vm = GpuVm::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for k in 0..COUNT {
                vm.map(pages(k, 1), guest(k * PAGE_SIZE), Kind(0)).unwrap();
            }
        });
        for _ in 0..READERS {
            scope.spawn(|| {
                loop {
                    let seen = vm.generation();
                    if let Some(newest) = seen.checked_sub(1) {
                        assert_eq!(
                            translate(&vm, newest),
                            Some(at(guest(newest * PAGE_SIZE), 0)),
                            "version {seen}"
                        );
                    }
                    if seen == COUNT {
                        break;
                    }
                    assert!(Instant::now() < deadline, "the writer did not finish");
                }
            });
        }
    });
    for k in 0..COUNT {
        assert_eq!(translate(&vm, k), Some(at(guest(k * PAGE_SIZE), 0)));
    }
}

// --- snapshots ---

/// A table with one mapping of each target kind and every field distinct, at version 4.
fn scene() -> GpuVm {
    let vm = GpuVm::new();
    vm.map(
        PageRange::new(0x1000, 0x2000).unwrap(),
        guest(0x5000),
        Kind(0x11),
    )
    .unwrap();
    vm.map(
        PageRange::new(0x4000, 0x3000).unwrap(),
        gpu_only(5, 0x6000),
        Kind(0x22),
    )
    .unwrap();
    vm.map(
        PageRange::new(0x9000, SHIM_PAGES * PAGE_SIZE).unwrap(),
        Target::SyncpointShim,
        Kind(0x33),
    )
    .unwrap();
    vm.apply(&VmUpdate::Unmap {
        range: PageRange::new(0x2_0000, PAGE_SIZE).unwrap(),
    });
    vm
}

/// [`scene`]'s snapshot, byte for byte (`postcard`: integers above a byte are varints, a `u8` is
/// one byte, a `Vec` starts with its length, an enum with its variant index).
#[rustfmt::skip]
fn golden() -> Vec<u8> {
    vec![
        // Header: magic, version.
        b'S', b'Q', b'V', b'M', 1,
        // Version 4; three mappings.
        4, 3,
        // 0x1000 + 0x2000, Guest (0) at 0x5000, kind 0x11.
        0x80, 0x20, 0x80, 0x40, 0, 0x80, 0xA0, 0x01, 0x11,
        // 0x4000 + 0x3000, GpuOnly (1) region 5 offset 0x6000, kind 0x22.
        0x80, 0x80, 0x01, 0x80, 0x60, 1, 5, 0x80, 0xC0, 0x01, 0x22,
        // 0x9000 + 0x1000, SyncpointShim (2), kind 0x33.
        0x80, 0xA0, 0x02, 0x80, 0x20, 2, 0x33,
    ]
}

#[test]
fn the_snapshot_layout_is_pinned() {
    let bytes = scene().save().unwrap();
    assert_eq!(bytes, golden());
    assert_eq!(bytes[..4], SNAPSHOT_MAGIC);
    assert_eq!(u32::from(bytes[4]), SNAPSHOT_VERSION);
}

#[test]
fn a_hand_built_snapshot_restores_and_works() {
    let vm = Arc::new(GpuVm::restore(&golden()).unwrap());
    assert_eq!(vm.generation(), 4);
    assert_eq!(vm.translate_page(0x2000), Some(at(guest(0x6000), 0x11)));
    assert_eq!(
        vm.translate_page(0x6000),
        Some(at(gpu_only(5, 0x8000), 0x22))
    );
    assert_eq!(
        vm.translate_page(0x9000),
        Some(at(Target::SyncpointShim, 0x33))
    );
    assert_eq!(vm.translate_page(0x3000), None);
    // The restored table takes changes and moves on from the restored version.
    let mut cache = TranslationCache::new(Arc::clone(&vm));
    assert_eq!(cache.translate(0x1000), Some(at(guest(0x5000), 0x11)));
    vm.map(
        PageRange::new(0x1000, PAGE_SIZE).unwrap(),
        guest(0xA000),
        Kind(0x44),
    )
    .unwrap();
    assert_eq!(vm.generation(), 5);
    assert_eq!(cache.translate(0x1000), Some(at(guest(0xA000), 0x44)));
    assert_eq!(cache.translate(0x2000), Some(at(guest(0x6000), 0x11)));
    assert_eq!(
        vm.apply(&VmUpdate::Unmap {
            range: PageRange::new(0x1000, 0x8000).unwrap()
        }),
        5
    );
    assert_eq!(vm.mappings().len(), 1);
}

/// A snapshot of `state` under the shipping header, written without the checks.
fn raw(state: &State) -> Vec<u8> {
    let mut bytes = postcard::to_allocvec(&Header {
        magic: SNAPSHOT_MAGIC,
        version: SNAPSHOT_VERSION,
    })
    .unwrap();
    bytes.extend(postcard::to_allocvec(state).unwrap());
    bytes
}

fn saved(start: u64, size: u64, target: SavedTarget) -> SavedMapping {
    SavedMapping {
        start,
        size,
        target,
        kind: 0,
    }
}

fn state(mappings: Vec<SavedMapping>) -> State {
    State {
        generation: 1,
        mappings,
    }
}

const GUEST_0: SavedTarget = SavedTarget::Guest { cpu_va: 0 };

/// [`golden`]'s body under a header of `magic` and `version`.
fn reheaded(magic: [u8; 4], version: u32) -> Vec<u8> {
    let shipping = postcard::to_allocvec(&Header {
        magic: SNAPSHOT_MAGIC,
        version: SNAPSHOT_VERSION,
    })
    .unwrap();
    let mut bytes = postcard::to_allocvec(&Header { magic, version }).unwrap();
    bytes.extend_from_slice(&golden()[shipping.len()..]);
    bytes
}

#[test]
fn a_bad_header_is_refused() {
    let good = golden();
    assert_eq!(reheaded(SNAPSHOT_MAGIC, SNAPSHOT_VERSION), good);
    let mut magic = good.clone();
    magic[0] = b'X';
    assert_eq!(GpuVm::restore(&magic).unwrap_err(), SnapshotError::BadMagic);
    // The version check at its edge on both sides: one below and one above the shipping version.
    for found in [SNAPSHOT_VERSION - 1, SNAPSHOT_VERSION + 1] {
        assert_eq!(
            GpuVm::restore(&reheaded(SNAPSHOT_MAGIC, found)).unwrap_err(),
            SnapshotError::UnsupportedVersion { found },
            "version {found}"
        );
    }
    // Both at once: the magic is checked first.
    let both = reheaded(*b"XQVM", SNAPSHOT_VERSION + 1);
    assert_eq!(GpuVm::restore(&both).unwrap_err(), SnapshotError::BadMagic);
    assert!(matches!(
        GpuVm::restore(&good[..3]),
        Err(SnapshotError::Decode(_))
    ));
}

#[test]
fn a_bad_body_is_refused() {
    let good = golden();
    assert!(matches!(
        GpuVm::restore(&good[..good.len() - 1]),
        Err(SnapshotError::Decode(_))
    ));
    let mut trailing = good.clone();
    trailing.extend([0, 0]);
    assert_eq!(
        GpuVm::restore(&trailing).unwrap_err(),
        SnapshotError::TrailingBytes { count: 2 }
    );
    // An unknown target variant does not decode.
    let mut variant = good;
    let shim_variant = variant.len() - 2;
    variant[shim_variant] = 3;
    assert!(matches!(
        GpuVm::restore(&variant),
        Err(SnapshotError::Decode(_))
    ));
}

#[test]
fn every_mapping_check_is_enforced() {
    let cases = [
        (
            saved(PAGE_SIZE + 1, PAGE_SIZE, GUEST_0),
            SnapshotError::Range {
                index: 1,
                error: MapError::MisalignedAddress {
                    address: PAGE_SIZE + 1,
                },
            },
        ),
        (
            saved(4 * PAGE_SIZE, 0, GUEST_0),
            SnapshotError::Range {
                index: 1,
                error: MapError::Empty,
            },
        ),
        (
            saved(u64::MAX - (PAGE_SIZE - 1), PAGE_SIZE, GUEST_0),
            SnapshotError::Range {
                index: 1,
                error: MapError::VirtualRange {
                    va: u64::MAX - (PAGE_SIZE - 1),
                    size: PAGE_SIZE,
                },
            },
        ),
        (
            saved(4 * PAGE_SIZE, PAGE_SIZE, SavedTarget::Guest { cpu_va: 8 }),
            SnapshotError::Target {
                index: 1,
                error: VmError::MisalignedTarget { value: 8 },
            },
        ),
        (
            saved(
                4 * PAGE_SIZE,
                PAGE_SIZE,
                SavedTarget::GpuOnly {
                    region: 1,
                    offset: u64::MAX - (PAGE_SIZE - 1),
                },
            ),
            SnapshotError::Target {
                index: 1,
                error: VmError::TargetOutOfRange {
                    value: u64::MAX - (PAGE_SIZE - 1),
                    size: PAGE_SIZE,
                },
            },
        ),
        (
            saved(
                4 * PAGE_SIZE,
                (SHIM_PAGES + 1) * PAGE_SIZE,
                SavedTarget::SyncpointShim,
            ),
            SnapshotError::Target {
                index: 1,
                error: VmError::ShimSize {
                    size: (SHIM_PAGES + 1) * PAGE_SIZE,
                },
            },
        ),
        // Starts one page before the first mapping (0..2 pages) ends.
        (
            saved(PAGE_SIZE, PAGE_SIZE, GUEST_0),
            SnapshotError::Overlap { index: 1 },
        ),
        // Out of order: wholly below the first.
        (
            saved(0, PAGE_SIZE, GUEST_0),
            SnapshotError::Overlap { index: 1 },
        ),
    ];
    let first = saved(0, 2 * PAGE_SIZE, GUEST_0);
    for (bad, error) in cases {
        let bytes = raw(&state(vec![first, bad]));
        assert_eq!(GpuVm::restore(&bytes).unwrap_err(), error, "{bad:?}");
    }
    // Exactly at the first one's end is accepted.
    let touching = raw(&state(vec![
        first,
        saved(2 * PAGE_SIZE, PAGE_SIZE, GUEST_0),
    ]));
    let vm = GpuVm::restore(&touching).unwrap();
    assert_eq!(vm.translate_page(PAGE_SIZE), Some(at(guest(PAGE_SIZE), 0)));
    assert_eq!(vm.translate_page(2 * PAGE_SIZE), Some(at(guest(0), 0)));
}

#[test]
fn mapping_faults_are_reported_in_the_documented_order() {
    // One mapping with a bad range, a bad target and an overlap: the range wins.
    let all = saved(PAGE_SIZE + 1, PAGE_SIZE, SavedTarget::Guest { cpu_va: 1 });
    let first = saved(0, 4 * PAGE_SIZE, GUEST_0);
    assert!(matches!(
        GpuVm::restore(&raw(&state(vec![first, all]))),
        Err(SnapshotError::Range { index: 1, .. })
    ));
    // A bad target and an overlap: the target wins.
    let target_and_overlap = saved(PAGE_SIZE, PAGE_SIZE, SavedTarget::Guest { cpu_va: 1 });
    assert!(matches!(
        GpuVm::restore(&raw(&state(vec![first, target_and_overlap]))),
        Err(SnapshotError::Target { index: 1, .. })
    ));
    // Faults in two mappings: the earlier one is reported.
    let shim = SHIM_PAGES * PAGE_SIZE;
    let too_big = (SHIM_PAGES + 1) * PAGE_SIZE;
    let early = saved(4 * PAGE_SIZE, shim, SavedTarget::SyncpointShim);
    let shim_too_big = saved(4 * PAGE_SIZE + shim, too_big, SavedTarget::SyncpointShim);
    let late = saved(4 * PAGE_SIZE + shim + too_big, 0, GUEST_0);
    assert_eq!(
        GpuVm::restore(&raw(&state(vec![first, early, shim_too_big, late]))).unwrap_err(),
        SnapshotError::Target {
            index: 2,
            error: VmError::ShimSize { size: too_big }
        }
    );
}

#[test]
fn errors_read_as_plain_words() {
    assert_eq!(
        VmError::MisalignedTarget { value: 0x1001 }.to_string(),
        "the target address or offset 0x1001 is not page aligned"
    );
    assert_eq!(
        VmError::TargetOutOfRange {
            value: 0xF000,
            size: 0x1000
        }
        .to_string(),
        "a target of 0x1000 bytes at 0xf000 runs past the end of the address space"
    );
    let shim = SHIM_PAGES * PAGE_SIZE;
    assert_eq!(
        VmError::ShimSize { size: 0x7000 }.to_string(),
        format!("the syncpoint shim is {shim:#x} bytes, not 0x7000 bytes")
    );
    assert_eq!(
        SnapshotError::Overlap { index: 3 }.to_string(),
        "mapping 3 of the GPU VM snapshot starts before the mapping before it ends"
    );
    assert_eq!(
        SnapshotError::Target {
            index: 2,
            error: VmError::ShimSize { size: 0x7000 }
        }
        .to_string(),
        format!(
            "mapping 2 of the GPU VM snapshot: the syncpoint shim is {shim:#x} bytes, not 0x7000 bytes"
        )
    );
    assert_eq!(
        SnapshotError::Range {
            index: 1,
            error: MapError::Empty
        }
        .to_string(),
        "mapping 1 of the GPU VM snapshot: empty range"
    );
    assert_eq!(
        SnapshotError::BadMagic.to_string(),
        "not a GPU VM snapshot (bad magic)"
    );
    assert_eq!(
        SnapshotError::UnsupportedVersion { found: 9 }.to_string(),
        format!(
            "GPU VM snapshot layout version 9 is not supported (this build reads {SNAPSHOT_VERSION})"
        )
    );
    assert_eq!(
        SnapshotError::TrailingBytes { count: 2 }.to_string(),
        "2 unexpected bytes after the GPU VM snapshot"
    );
}

// --- the model test (graphics-server slice 6: "Model test against a page-by-page map under
// random map, unmap and modify") ---

/// Pages the model covers, from test page 0.
const MODEL_PAGES: u64 = 16;

/// A generated target. Offsets are in pages so they stay aligned unless `misaligned`.
#[derive(Clone, Copy, Debug)]
enum GenTarget {
    Guest { cpu_page: u64, misaligned: bool },
    GpuOnly { region: u32, offset_page: u64 },
    Shim,
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Map {
        first: u64,
        count: u64,
        target: GenTarget,
        kind: u8,
    },
    /// `which`: 0 unmap, 1 kind change, 2 memory destroy, 3 sparse-region destroy.
    Update {
        which: u8,
        first: u64,
        count: u64,
        kind: u8,
        region: u32,
    },
    /// Save, restore, and carry on with the restored table.
    Snapshot,
}

fn range_strategy() -> impl Strategy<Value = (u64, u64)> {
    (0..MODEL_PAGES).prop_flat_map(|first| (Just(first), 1..=MODEL_PAGES - first))
}

fn op_strategy() -> impl Strategy<Value = Op> {
    let target = prop_oneof![
        (0..64u64, prop::bool::weighted(0.1)).prop_map(|(cpu_page, misaligned)| {
            GenTarget::Guest {
                cpu_page,
                misaligned,
            }
        }),
        (1..4u32, 0..64u64).prop_map(|(region, offset_page)| GenTarget::GpuOnly {
            region,
            offset_page
        }),
        Just(GenTarget::Shim),
    ];
    prop_oneof![
        4 => (range_strategy(), target, any::<u8>()).prop_map(|((first, count), target, kind)| {
            Op::Map {
                first,
                count,
                target,
                kind,
            }
        }),
        4 => (0..4u8, range_strategy(), any::<u8>(), 1..4u32).prop_map(
            |(which, (first, count), kind, region)| Op::Update {
                which,
                first,
                count,
                kind,
                region,
            }
        ),
        1 => Just(Op::Snapshot),
    ]
}

/// The model: one translation per page, by GPU address.
type Model = BTreeMap<u64, Translation>;

/// What a generated map is expected to do: the target of its first page and the answer, worked
/// out from the rules, not by the code under test.
fn expected_map(count: u64, target: GenTarget) -> (Target, Result<(), VmError>) {
    match target {
        GenTarget::Guest {
            cpu_page,
            misaligned,
        } => {
            let cpu_va = 0x4000_0000 + cpu_page * PAGE_SIZE + if misaligned { 0x10 } else { 0 };
            let answer = if misaligned {
                Err(VmError::MisalignedTarget { value: cpu_va })
            } else {
                Ok(())
            };
            (Target::Guest { cpu_va }, answer)
        }
        GenTarget::GpuOnly {
            region,
            offset_page,
        } => (gpu_only(region, offset_page * PAGE_SIZE), Ok(())),
        GenTarget::Shim => {
            let answer = if count == SHIM_PAGES {
                Ok(())
            } else {
                Err(VmError::ShimSize {
                    size: count * PAGE_SIZE,
                })
            };
            (Target::SyncpointShim, answer)
        }
    }
}

/// `target` for the page `i` pages into a mapping, written out by hand.
fn model_target(target: Target, i: u64) -> Target {
    match target {
        Target::Guest { cpu_va } => Target::Guest {
            cpu_va: cpu_va + i * PAGE_SIZE,
        },
        Target::GpuOnly { region, offset } => Target::GpuOnly {
            region,
            offset: offset + i * PAGE_SIZE,
        },
        Target::SyncpointShim => Target::SyncpointShim,
    }
}

/// Checks `vm` and `cache` against `model` on every model page and the page on each side.
fn check(vm: &GpuVm, cache: &mut TranslationCache<GpuVm>, model: &Model, version: u64) {
    assert_eq!(vm.generation(), version);
    for va in (page(0) - PAGE_SIZE..=page(MODEL_PAGES)).step_by(usize::try_from(PAGE_SIZE).unwrap())
    {
        let want = model.get(&va).copied();
        assert_eq!(vm.translate_page(va), want, "page {va:#x}");
        assert_eq!(cache.translate(va + 0x123), want, "cached page {va:#x}");
    }
    // The listed mappings are in order, do not overlap, and expand to the model.
    let mut expanded = Model::new();
    let mut previous_end = 0;
    for mapping in vm.mappings() {
        assert!(mapping.range.start() >= previous_end, "{mapping:?}");
        previous_end = mapping.range.end();
        for i in 0..mapping.range.size() / PAGE_SIZE {
            expanded.insert(
                mapping.range.start() + i * PAGE_SIZE,
                Translation {
                    target: model_target(mapping.target, i),
                    kind: mapping.kind,
                },
            );
        }
    }
    assert_eq!(&expanded, model);
}

proptest! {
    #![proptest_config(Config {
        cases: 256,
        rng_seed: RngSeed::Fixed(0x4750_5556),
        ..Config::default()
    })]

    #[test]
    fn the_table_matches_a_page_by_page_model(ops in prop::collection::vec(op_strategy(), 1..48)) {
        let mut vm = Arc::new(GpuVm::new());
        let mut cache = TranslationCache::new(Arc::clone(&vm));
        let mut model = Model::new();
        let mut version = 0u64;
        for op in ops {
            match op {
                Op::Map { first, count, target, kind } => {
                    let (target, answer) = expected_map(count, target);
                    prop_assert_eq!(vm.map(pages(first, count), target, Kind(kind)), answer);
                    if answer.is_ok() {
                        for i in 0..count {
                            model.insert(page(first + i), Translation {
                                target: model_target(target, i),
                                kind: Kind(kind),
                            });
                        }
                        version += 1;
                    }
                }
                Op::Update { which, first, count, kind, region } => {
                    let range = pages(first, count);
                    let update = match which {
                        0 => VmUpdate::Unmap { range },
                        1 => VmUpdate::ModifyKind { range, kind: Kind(kind) },
                        2 => VmUpdate::DestroyMemory { range },
                        _ => VmUpdate::DestroySparseRegion { range, region: RegionId(region) },
                    };
                    let mut affected = 0;
                    for i in 0..count {
                        let va = page(first + i);
                        if which == 1 {
                            if let Some(entry) = model.get_mut(&va) {
                                entry.kind = Kind(kind);
                                affected += 1;
                            }
                        } else if model.remove(&va).is_some() {
                            affected += 1;
                        }
                    }
                    prop_assert_eq!(vm.apply(&update), affected);
                    version += 1;
                }
                Op::Snapshot => {
                    let bytes = vm.save().unwrap();
                    vm = Arc::new(GpuVm::restore(&bytes).unwrap());
                    prop_assert_eq!(vm.save().unwrap(), bytes);
                    cache = TranslationCache::new(Arc::clone(&vm));
                }
            }
            check(&vm, &mut cache, &model, version);
        }
    }
}
