//! Tests of [`Stage1Tables`] through the [`Translation`] trait, read back with the independent
//! [`walk`]er and the raw table bytes.

use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use super::descriptor::{self, LEAF_LEVEL, START_LEVEL, shift};
use super::*;
use crate::{Access, FlatMemory};

/// Where the pool lives in every test (hvprobe's guest base; any page-aligned address works).
const POOL_BASE: u64 = 0x8000_0000;
/// Where backing pages start in the model test, clear of every pool.
const BACKING_BASE: u64 = 0x1_0000_0000;

const ALL_PERMS: [Perms; 6] = [
    Perms::None,
    Perms::Read,
    Perms::ReadWrite,
    Perms::Execute,
    Perms::ReadExecute,
    Perms::ReadWriteExecute,
];

fn pages(n: u64) -> u64 {
    n * PAGE_SIZE
}

/// Tables over a memory that is exactly a pool of `pool_pages` pages.
fn tables(pool_pages: u64) -> Stage1Tables<FlatMemory> {
    let size = pages(pool_pages);
    let memory = FlatMemory::new(POOL_BASE, usize::try_from(size).unwrap());
    Stage1Tables::new(memory, BackingRange::new(POOL_BASE, size).unwrap()).unwrap()
}

fn backing(pa: u64, n: u64) -> BackingRange {
    BackingRange::new(pa, pages(n)).unwrap()
}

/// The pool's raw bytes.
fn image(t: &Stage1Tables<FlatMemory>) -> Vec<u8> {
    let mut bytes = vec![0; usize::try_from(pages(t.pool_pages())).unwrap()];
    t.memory().read(POOL_BASE, &mut bytes).unwrap();
    bytes
}

fn put(t: &Stage1Tables<FlatMemory>, at: u64, value: u64) {
    t.memory().write(at, &value.to_le_bytes()).unwrap();
}

/// A virtual address from its level-1, level-2 and level-3 indices.
fn va_of(l1: u64, l2: u64, l3: u64) -> u64 {
    (l1 << shift(1)) | (l2 << shift(2)) | (l3 << shift(3))
}

#[test]
fn writes_architectural_descriptors_for_every_perms() {
    let t = tables(3);
    let va = va_of(5, 7, 9);
    // Expected page descriptors, from Arm DDI 0487: bits[1:0] = 0b11 (page), AP[2:1] at bits
    // 7:6, SH = 0b11 at 9:8, AF bit 10, nG bit 11, output address 47:12, PXN bit 53, UXN
    // bit 54. Low 12 bits: 0xf03 with AP 0b00, 0xf43 with AP 0b01, 0xfc3 with AP 0b11.
    // High bits: 0x0060... with UXN and PXN, 0x0020... with PXN only.
    let cases = [
        (Perms::None, 0x1111_1000, 0x0060_0000_1111_1f03),
        (Perms::Read, 0x2222_2000, 0x0060_0000_2222_2fc3),
        (Perms::ReadWrite, 0x3333_3000, 0x0060_0000_3333_3f43),
        (Perms::Execute, 0x4444_4000, 0x0020_0000_4444_4f03),
        (Perms::ReadExecute, 0x5555_5000, 0x0020_0000_5555_5fc3),
        // The last page below the 48-bit output address limit.
        (
            Perms::ReadWriteExecute,
            0xffff_ffff_f000,
            0x0020_ffff_ffff_ff43,
        ),
    ];
    for (n, (perms, pa, _)) in (0..).zip(cases) {
        t.map(va + pages(n), backing(pa, 1), perms).unwrap();
    }

    // Root is the first pool page, then the level-2 and level-3 tables in allocation order.
    let (l2, l3) = (POOL_BASE + pages(1), POOL_BASE + pages(2));
    let mut expected = vec![0u8; usize::try_from(pages(3)).unwrap()];
    let mut set = |at: u64, value: u64| {
        let offset = usize::try_from(at - POOL_BASE).unwrap();
        expected[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    };
    // Table descriptors: next-level address | 0b11.
    set(POOL_BASE + 5 * 8, 0x8000_1003);
    set(l2 + 7 * 8, 0x8000_2003);
    for (n, (_, _, page)) in (0..).zip(cases) {
        set(l3 + (9 + n) * 8, page);
    }
    assert_eq!(image(&t), expected);
    assert_eq!(t.root().pa(), POOL_BASE);
    assert_eq!(l3 + 9 * 8, 0x8000_2048);
}

#[test]
fn maps_resolve_and_carry_the_page_offset() {
    let t = tables(4);
    let va = va_of(1, 2, 3);
    t.map(va, backing(0x9_0000_0000, 3), Perms::ReadWrite)
        .unwrap();
    for n in 0..3 {
        let at = va + pages(n) + 0x7f8;
        assert_eq!(
            walk(t.memory(), t.root(), at),
            Ok(Resolved {
                pa: 0x9_0000_0000 + pages(n) + 0x7f8,
                perms: Perms::ReadWrite,
                level: LEAF_LEVEL
            })
        );
        assert_eq!(t.walk(at), walk(t.memory(), t.root(), at));
    }
    assert_eq!(
        t.walk(va + pages(3)),
        Err(WalkError::Unmapped {
            va: va + pages(3),
            level: LEAF_LEVEL
        })
    );
}

#[test]
fn unmapped_and_permission_denied_are_distinct() {
    let t = tables(3);
    let va = va_of(0, 0, 1);
    t.map(va, backing(0x10_0000, 1), Perms::Read).unwrap();
    assert_eq!(
        translate(t.memory(), t.root(), va, Access::Read),
        Ok(0x10_0000)
    );
    assert_eq!(
        translate(t.memory(), t.root(), va, Access::Write),
        Err(WalkError::PermissionDenied {
            va,
            access: Access::Write,
            perms: Perms::Read
        })
    );
    let next = va + PAGE_SIZE;
    assert_eq!(
        translate(t.memory(), t.root(), next, Access::Write),
        Err(WalkError::Unmapped {
            va: next,
            level: LEAF_LEVEL
        })
    );
}

#[test]
fn virtual_ranges_are_checked_over_their_whole_extent() {
    let t = tables(4);
    let pa = 0x10_0000;
    assert_eq!(
        t.map(PAGE_SIZE + 8, backing(pa, 1), Perms::Read),
        Err(MapError::MisalignedAddress {
            address: PAGE_SIZE + 8
        })
    );
    assert_eq!(
        t.unmap(0, PAGE_SIZE + 8),
        Err(MapError::MisalignedSize {
            size: PAGE_SIZE + 8
        })
    );
    assert_eq!(t.protect(0, 0, Perms::Read), Err(MapError::Empty));
    // The last page of the space maps; one more page doesn't.
    let last = VA_SPACE_END - PAGE_SIZE;
    t.map(last, backing(pa, 1), Perms::Read).unwrap();
    let over = VA_SPACE_END - pages(2);
    assert_eq!(
        t.map(over, backing(pa, 3), Perms::Read),
        Err(MapError::VirtualRange {
            va: over,
            size: pages(3)
        })
    );
    assert_eq!(
        t.map(VA_SPACE_END, backing(pa, 1), Perms::Read),
        Err(MapError::VirtualRange {
            va: VA_SPACE_END,
            size: PAGE_SIZE
        })
    );
    // va + size overflows u64.
    let top = u64::MAX - (PAGE_SIZE - 1);
    assert_eq!(
        t.unmap(top, pages(2)),
        Err(MapError::VirtualRange {
            va: top,
            size: pages(2)
        })
    );
    assert_eq!(
        t.protect(top, PAGE_SIZE, Perms::Read),
        Err(MapError::VirtualRange {
            va: top,
            size: PAGE_SIZE
        })
    );
    assert_eq!(t.unmap(last, PAGE_SIZE), Ok(()));
}

#[test]
fn backing_must_fit_the_output_address_and_avoid_the_pool() {
    let t = tables(4);
    let pool_end = POOL_BASE + pages(4);
    // Ending exactly at the 48-bit limit is fine; one page further is not.
    let top = OUTPUT_ADDRESS_END - pages(2);
    t.map(0, backing(top, 2), Perms::Read).unwrap();
    let over = backing(top, 3);
    assert_eq!(
        t.map(pages(2), over, Perms::Read),
        Err(MapError::PhysicalRange {
            pa: top,
            size: pages(3)
        })
    );
    // Touching the pool on either side is fine; sharing its first or last page is not.
    t.map(pages(2), backing(POOL_BASE - pages(2), 2), Perms::Read)
        .unwrap();
    t.map(pages(4), backing(pool_end, 1), Perms::Read).unwrap();
    for (pa, n) in [(POOL_BASE - PAGE_SIZE, 2), (pool_end - PAGE_SIZE, 2)] {
        assert_eq!(
            t.map(pages(5), backing(pa, n), Perms::Read),
            Err(MapError::OverlapsTables { pa, size: pages(n) })
        );
    }
    assert_eq!(
        t.walk(pages(5)),
        Err(WalkError::Unmapped {
            va: pages(5),
            level: LEAF_LEVEL
        })
    );
}

#[test]
fn failed_operations_name_the_lowest_offending_page_and_change_nothing() {
    let t = tables(4);
    t.map(pages(4), backing(0x10_0000, 2), Perms::ReadWrite)
        .unwrap();
    t.take_pending_flushes();
    let before = image(&t);
    assert_eq!(
        t.map(pages(2), backing(0x20_0000, 8), Perms::Read),
        Err(MapError::AlreadyMapped { va: pages(4) })
    );
    assert_eq!(
        t.unmap(pages(4), pages(3)),
        Err(MapError::NotMapped { va: pages(6) })
    );
    assert_eq!(
        t.protect(pages(3), pages(2), Perms::Read),
        Err(MapError::NotMapped { va: pages(3) })
    );
    // Nothing under a whole missing level-2 table: reported at the range start.
    let far = va_of(3, 0, 0);
    assert_eq!(
        t.unmap(far, PAGE_SIZE),
        Err(MapError::NotMapped { va: far })
    );
    assert_eq!(image(&t), before);
    assert_eq!(t.take_pending_flushes(), []);
}

#[test]
fn running_out_of_table_pages_writes_nothing() {
    // A fresh mapping needs a level-2 and a level-3 table besides the root.
    let short = tables(2);
    let before = image(&short);
    assert_eq!(
        short.map(0, backing(0x10_0000, 1), Perms::Read),
        Err(MapError::OutOfTablePages)
    );
    assert_eq!(image(&short), before);
    assert_eq!(short.table_pages_in_use(), 1);

    let exact = tables(3);
    exact.map(0, backing(0x10_0000, 1), Perms::Read).unwrap();
    assert_eq!(exact.table_pages_in_use(), exact.pool_pages());
    // Crossing into the next level-2 slot needs one more level-3 table: refused whole, so the
    // page that would have fitted in the existing table stays unmapped too.
    let edge = va_of(0, 1, 0) - PAGE_SIZE;
    assert_eq!(
        exact.map(edge, backing(0x20_0000, 2), Perms::Read),
        Err(MapError::OutOfTablePages)
    );
    assert_eq!(
        exact.walk(edge),
        Err(WalkError::Unmapped {
            va: edge,
            level: LEAF_LEVEL
        })
    );
}

#[test]
fn table_pages_are_reclaimed_and_reused() {
    // A range across a level-1 boundary needs the root, two level-2 tables and two level-3
    // tables; a page in another level-2 slot of level-1 slot 1 needs a third level-3 table.
    let t = tables(6);
    let start = va_of(1, 0, 0) - pages(2);
    t.map(start, backing(0x10_0000, 4), Perms::ReadWrite)
        .unwrap();
    let far = va_of(1, 5, 0);
    t.map(far, backing(0x20_0000, 1), Perms::Read).unwrap();
    assert_eq!(t.table_pages_in_use(), 6);
    assert_eq!(t.take_pending_flushes(), []);

    // Removing one page of a table that still holds another frees nothing.
    t.unmap(start, PAGE_SIZE).unwrap();
    assert_eq!(t.table_pages_in_use(), 6);
    // Emptying the level-3 table under level-1 slot 0 frees it and its level-2 table.
    t.unmap(start + PAGE_SIZE, PAGE_SIZE).unwrap();
    assert_eq!(t.table_pages_in_use(), 4);
    assert_eq!(
        t.take_pending_flushes(),
        [
            FlushRange {
                va: start,
                size: PAGE_SIZE,
                leaf_only: true
            },
            FlushRange {
                va: start + PAGE_SIZE,
                size: PAGE_SIZE,
                leaf_only: false
            },
        ]
    );
    assert_eq!(
        t.walk(start),
        Err(WalkError::Unmapped {
            va: start,
            level: START_LEVEL
        })
    );

    // The pool is full again only by reusing the freed pages.
    t.map(va_of(7, 7, 7), backing(0x30_0000, 1), Perms::Read)
        .unwrap();
    assert_eq!(t.table_pages_in_use(), t.pool_pages());

    // Unmapping everything leaves just an empty root, and every freed page all zeros, ready
    // for reuse without zeroing.
    t.unmap(va_of(1, 0, 0), pages(2)).unwrap();
    t.unmap(far, PAGE_SIZE).unwrap();
    t.unmap(va_of(7, 7, 7), PAGE_SIZE).unwrap();
    assert_eq!(t.table_pages_in_use(), 1);
    assert!(image(&t).iter().all(|&b| b == 0));
}

#[test]
fn protect_keeps_the_backing_and_records_a_leaf_flush() {
    let t = tables(3);
    t.map(pages(8), backing(0x7_0000, 2), Perms::ReadWrite)
        .unwrap();
    t.protect(pages(9), PAGE_SIZE, Perms::ReadExecute).unwrap();
    assert_eq!(
        t.walk(pages(8)).map(|r| (r.pa, r.perms)),
        Ok((0x7_0000, Perms::ReadWrite))
    );
    assert_eq!(
        t.walk(pages(9)).map(|r| (r.pa, r.perms)),
        Ok((0x7_0000 + PAGE_SIZE, Perms::ReadExecute))
    );
    assert_eq!(
        t.take_pending_flushes(),
        [FlushRange {
            va: pages(9),
            size: PAGE_SIZE,
            leaf_only: true
        }]
    );
    assert_eq!(t.take_pending_flushes(), []);
}

#[test]
fn descriptors_the_builder_never_writes_are_errors() {
    // A level-1 block where a table should be.
    let t = tables(3);
    let block = 0x0040_0000_4000_0441;
    put(&t, POOL_BASE, block);
    assert_eq!(
        t.map(0, backing(0x10_0000, 1), Perms::Read),
        Err(MapError::UnexpectedDescriptor {
            va: 0,
            level: START_LEVEL,
            descriptor: block
        })
    );
    // A table descriptor pointing outside the pool.
    let outside = descriptor::table(POOL_BASE + pages(3));
    put(&t, POOL_BASE, outside);
    assert_eq!(
        t.unmap(0, PAGE_SIZE),
        Err(MapError::UnexpectedDescriptor {
            va: 0,
            level: START_LEVEL,
            descriptor: outside
        })
    );
    // The reserved level-3 encoding.
    put(&t, POOL_BASE, 0);
    t.map(0, backing(0x10_0000, 1), Perms::Read).unwrap();
    let reserved = 0x0040_0000_0010_0441;
    put(&t, POOL_BASE + pages(2) + 8, reserved);
    assert_eq!(
        t.protect(0, pages(2), Perms::Read),
        Err(MapError::UnexpectedDescriptor {
            va: PAGE_SIZE,
            level: LEAF_LEVEL,
            descriptor: reserved
        })
    );
}

#[test]
fn new_checks_the_pool() {
    // The memory covers two of the pool's three pages: the first refused page is reported.
    let memory = FlatMemory::new(POOL_BASE, usize::try_from(pages(2)).unwrap());
    assert_eq!(
        Stage1Tables::new(memory, backing(POOL_BASE, 3)).map(|_| ()),
        Err(MapError::Memory(Fault {
            address: POOL_BASE + pages(2)
        }))
    );
    // A pool ending exactly at the output address limit is fine; one page further is not.
    let top = OUTPUT_ADDRESS_END - PAGE_SIZE;
    let memory = FlatMemory::new(top, usize::try_from(PAGE_SIZE).unwrap());
    let t = Stage1Tables::new(memory, backing(top, 1)).unwrap();
    assert_eq!(t.root(), TableRoot::new(top).unwrap());
    let memory = FlatMemory::new(top, usize::try_from(pages(2)).unwrap());
    assert_eq!(
        Stage1Tables::new(memory, backing(top, 2)).map(|_| ()),
        Err(MapError::PhysicalRange {
            pa: top,
            size: pages(2)
        })
    );
}

#[test]
fn table_root_must_be_an_aligned_output_address() {
    assert_eq!(
        TableRoot::new(POOL_BASE + 8),
        Err(MapError::MisalignedAddress {
            address: POOL_BASE + 8
        })
    );
    assert_eq!(
        TableRoot::new(OUTPUT_ADDRESS_END),
        Err(MapError::PhysicalRange {
            pa: OUTPUT_ADDRESS_END,
            size: PAGE_SIZE
        })
    );
    assert_eq!(
        TableRoot::new(OUTPUT_ADDRESS_END - PAGE_SIZE).map(TableRoot::pa),
        Ok(OUTPUT_ADDRESS_END - PAGE_SIZE)
    );
}

// ---- model test ----------------------------------------------------------------------------

/// Pages per test window.
const WINDOW_PAGES: u64 = 16;

/// Window starts: the bottom of the space, across a level-2 boundary, across a level-1
/// boundary, and the top of the space.
fn windows() -> [u64; 4] {
    [
        0,
        (1 << shift(2)) - pages(4),
        (1 << shift(1)) - pages(4),
        VA_SPACE_END - pages(WINDOW_PAGES),
    ]
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Map {
        va: u64,
        pages: u64,
        pa: u64,
        perms: Perms,
    },
    Unmap {
        va: u64,
        pages: u64,
    },
    Protect {
        va: u64,
        pages: u64,
        perms: Perms,
    },
}

fn op() -> impl Strategy<Value = Op> {
    let range =
        (0..windows().len(), 0..WINDOW_PAGES, 1..=WINDOW_PAGES).prop_map(|(window, first, len)| {
            let va = windows()[window] + pages(first);
            (va, len.min(WINDOW_PAGES - first))
        });
    let perms = proptest::sample::select(ALL_PERMS.to_vec());
    prop_oneof![
        2 => (range.clone(), 0..64u64, perms.clone()).prop_map(|((va, n), page, perms)| {
            Op::Map {
                va,
                pages: n,
                pa: BACKING_BASE + pages(page),
                perms,
            }
        }),
        1 => range.clone().prop_map(|(va, pages)| Op::Unmap { va, pages }),
        1 => (range, perms).prop_map(|((va, pages), perms)| Op::Protect { va, pages, perms }),
    ]
}

/// What the tables should hold: virtual page -> (physical page, perms).
#[derive(Debug, Clone, Default)]
struct Model {
    pages: BTreeMap<u64, (u64, Perms)>,
}

impl Model {
    /// The table pages these mappings need when empty tables are freed at once.
    fn tables(&self) -> u64 {
        let distinct = |level: u8| {
            self.pages
                .keys()
                .map(|va| va >> shift(level))
                .collect::<BTreeSet<_>>()
                .len() as u64
        };
        1 + distinct(1) + distinct(2)
    }

    /// The level at which a walk of the unmapped page `va` finds an invalid descriptor.
    fn missing_level(&self, va: u64) -> u8 {
        let shares = |level: u8| {
            self.pages
                .keys()
                .any(|v| v >> shift(level) == va >> shift(level))
        };
        if !shares(1) {
            1
        } else if !shares(2) {
            2
        } else {
            3
        }
    }

    fn range(va: u64, n: u64) -> impl Iterator<Item = u64> {
        (0..n).map(move |i| va + pages(i))
    }

    /// Applies `op` and returns the result and flushes the tables should report.
    fn apply(&mut self, op: Op, pool_pages: u64) -> (Result<(), MapError>, Vec<FlushRange>) {
        match op {
            Op::Map {
                va,
                pages: n,
                pa,
                perms,
            } => {
                if let Some(v) = Self::range(va, n).find(|v| self.pages.contains_key(v)) {
                    return (Err(MapError::AlreadyMapped { va: v }), vec![]);
                }
                let mut after = self.clone();
                for (i, v) in (0..).zip(Self::range(va, n)) {
                    after.pages.insert(v, (pa + pages(i), perms));
                }
                if after.tables() > pool_pages {
                    return (Err(MapError::OutOfTablePages), vec![]);
                }
                *self = after;
                (Ok(()), vec![])
            }
            Op::Unmap { va, pages: n } => {
                if let Some(v) = Self::range(va, n).find(|v| !self.pages.contains_key(v)) {
                    return (Err(MapError::NotMapped { va: v }), vec![]);
                }
                let before = self.tables();
                for v in Self::range(va, n) {
                    self.pages.remove(&v);
                }
                let flush = FlushRange {
                    va,
                    size: pages(n),
                    leaf_only: self.tables() == before,
                };
                (Ok(()), vec![flush])
            }
            Op::Protect {
                va,
                pages: n,
                perms,
            } => {
                if let Some(v) = Self::range(va, n).find(|v| !self.pages.contains_key(v)) {
                    return (Err(MapError::NotMapped { va: v }), vec![]);
                }
                for v in Self::range(va, n) {
                    if let Some(entry) = self.pages.get_mut(&v) {
                        entry.1 = perms;
                    }
                }
                let flush = FlushRange {
                    va,
                    size: pages(n),
                    leaf_only: true,
                };
                (Ok(()), vec![flush])
            }
        }
    }
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 256,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x5354_4147_4531_5442),
        failure_persistence: None,
        ..Config::default()
    }
}

fn check_against_model(t: &Stage1Tables<FlatMemory>, model: &Model) {
    assert_eq!(t.table_pages_in_use(), model.tables());
    for (w, &window) in windows().iter().enumerate() {
        for (i, page) in (0..).zip(Model::range(window, WINDOW_PAGES)) {
            // A different byte of each page, so the offset is carried through.
            let offset = (i * 0x238 + u64::try_from(w).unwrap() * 8) % PAGE_SIZE;
            let va = page + offset;
            let resolved = walk(t.memory(), t.root(), va);
            match model.pages.get(&page) {
                Some(&(pa, perms)) => {
                    assert_eq!(
                        resolved,
                        Ok(Resolved {
                            pa: pa + offset,
                            perms,
                            level: LEAF_LEVEL
                        }),
                        "{va:#x}"
                    );
                    for access in [Access::Read, Access::Write, Access::Execute] {
                        let expected = if perms.allows(access) {
                            Ok(pa + offset)
                        } else {
                            Err(WalkError::PermissionDenied { va, access, perms })
                        };
                        assert_eq!(
                            translate(t.memory(), t.root(), va, access),
                            expected,
                            "{va:#x} {access:?}"
                        );
                    }
                }
                None => assert_eq!(
                    resolved,
                    Err(WalkError::Unmapped {
                        va,
                        level: model.missing_level(page)
                    }),
                    "{va:#x}"
                ),
            }
        }
    }
}

proptest! {
    #![proptest_config(config())]

    /// Random map, unmap and protect sequences agree with a page-by-page model: results,
    /// recorded flushes, table pages in use, and the walker's answer for every window page.
    #[test]
    fn walker_agrees_with_a_page_model(
        pool_pages in 3..=12u64,
        ops in proptest::collection::vec(op(), 1..48),
    ) {
        let t = tables(pool_pages);
        let mut model = Model::default();
        for op in ops {
            let (expected, flushes) = model.apply(op, pool_pages);
            let actual = match op {
                Op::Map { va, pages: n, pa, perms } => t.map(va, backing(pa, n), perms),
                Op::Unmap { va, pages: n } => t.unmap(va, pages(n)),
                Op::Protect { va, pages: n, perms } => t.protect(va, pages(n), perms),
            };
            prop_assert_eq!(actual, expected, "{:?}", op);
            prop_assert_eq!(t.take_pending_flushes(), flushes, "{:?}", op);
            check_against_model(&t, &model);
        }
    }
}
