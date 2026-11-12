//! The block map against hand-worked scenarios and against a naive page-by-page model.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use super::*;

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
const PROPTEST_SEED: u64 = 0x424C_4F43_4B4D_4150;

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 512,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(PROPTEST_SEED),
        failure_persistence: None,
        ..Config::default()
    }
}

/// Where the test maps start: not zero, so an address taken for a page index shows.
const BASE: u64 = 0x10_0000;
/// How many pages the test maps cover.
const PAGES: u64 = 40;

fn page(index: u64) -> u64 {
    BASE + index * PAGE_SIZE
}

fn map() -> BlockMap {
    BlockMap::new(BASE, page(PAGES)).unwrap()
}

/// What the tests read of a block: where it is, what it is, and its merge barriers.
type Row = (
    u64,
    u64,
    MemoryState,
    MemoryPermission,
    MemoryAttribute,
    DisableMerge,
);

fn rows(map: &BlockMap) -> Vec<Row> {
    map.blocks()
        .map(|block| {
            (
                block.address(),
                block.pages(),
                block.state(),
                block.permission(),
                block.attribute(),
                block.disable_merge(),
            )
        })
        .collect()
}

fn free(first: u64, pages: u64) -> Row {
    (
        page(first),
        pages,
        MemoryState::Free,
        MemoryPermission::NONE,
        MemoryAttribute::NONE,
        DisableMerge::NONE,
    )
}

fn heap(first: u64, pages: u64, barrier: DisableMerge) -> Row {
    (
        page(first),
        pages,
        MemoryState::Normal,
        MemoryPermission::USER_READ_WRITE,
        MemoryAttribute::NONE,
        barrier,
    )
}

/// Maps `pages` pages at page `first` as heap, with a barrier at its head.
fn map_heap(map: &mut BlockMap, first: u64, pages: u64) {
    map.update(
        page(first),
        pages,
        MemoryState::Normal,
        MemoryPermission::USER_READ_WRITE,
        MemoryAttribute::NONE,
        DisableMerge::NORMAL,
        DisableMerge::NONE,
    )
    .unwrap();
}

#[test]
fn a_new_map_is_one_free_block() {
    let map = map();
    assert_eq!(rows(&map), [free(0, PAGES)]);
    assert_eq!((map.start(), map.end()), (BASE, page(PAGES)));
    assert!(map.check_state());
}

#[test]
fn a_map_needs_aligned_bounds_in_order() {
    for (start, end) in [
        (BASE, BASE),
        (page(1), BASE),
        (BASE + 1, page(1)),
        (BASE, page(1) + 1),
    ] {
        assert_eq!(
            BlockMap::new(start, end),
            Err(BlockMapError::BadBounds { start, end }),
            "{start:#x}..{end:#x}"
        );
    }
    assert_eq!(
        BlockMapError::BadBounds {
            start: 0x1000,
            end: 0x1000
        }
        .to_string(),
        "a block map cannot cover 0x1000..0x1000"
    );
}

#[test]
fn find_returns_the_block_that_holds_an_address_at_both_of_its_edges() {
    let mut map = map();
    map_heap(&mut map, 4, 2);
    let at = |address: u64| {
        map.find(address)
            .map(|block| (block.address(), block.pages()))
    };
    assert_eq!(at(BASE), Some((BASE, 4)));
    assert_eq!(at(page(4) - 1), Some((BASE, 4)));
    assert_eq!(at(page(4)), Some((page(4), 2)));
    assert_eq!(at(page(6) - 1), Some((page(4), 2)));
    assert_eq!(at(page(6)), Some((page(6), PAGES - 6)));
    assert_eq!(at(page(PAGES) - 1), Some((page(6), PAGES - 6)));
    // Outside the map on either side.
    assert_eq!(at(BASE - 1), None);
    assert_eq!(at(page(PAGES)), None);
    assert_eq!(at(u64::MAX), None);
    // Iterating from an address starts at the block that holds it; outside there is nothing.
    let from: Vec<_> = map.blocks_from(page(5)).map(MemoryBlock::address).collect();
    assert_eq!(from, [page(4), page(6)]);
    assert_eq!(map.blocks_from(BASE - 1).count(), 0);
    assert_eq!(map.blocks_from(page(PAGES)).count(), 0);
}

#[test]
fn an_update_in_the_middle_splits_twice_and_undoing_it_merges_back() {
    let mut map = map();
    map_heap(&mut map, 4, 2);
    assert_eq!(
        rows(&map),
        [
            free(0, 4),
            heap(4, 2, DisableMerge::NORMAL),
            free(6, PAGES - 6)
        ]
    );
    assert!(map.check_state());
    // Unmapping clears the barrier, and the three blocks become one again.
    map.update(
        page(4),
        2,
        MemoryState::Free,
        MemoryPermission::NONE,
        MemoryAttribute::NONE,
        DisableMerge::NONE,
        DisableMerge::NORMAL,
    )
    .unwrap();
    assert_eq!(rows(&map), [free(0, PAGES)]);
}

#[test]
fn growing_a_mapping_merges_with_it_and_a_second_mapping_does_not() {
    let mut map = map();
    map_heap(&mut map, 0, 2);
    // More heap right after, without a barrier of its own (as `SetHeapSize` grows a heap).
    map.update(
        page(2),
        3,
        MemoryState::Normal,
        MemoryPermission::USER_READ_WRITE,
        MemoryAttribute::NONE,
        DisableMerge::NONE,
        DisableMerge::NONE,
    )
    .unwrap();
    assert_eq!(
        rows(&map),
        [heap(0, 5, DisableMerge::NORMAL), free(5, PAGES - 5)]
    );
    // A separate mapping with the same properties right after: its barrier keeps it apart.
    map_heap(&mut map, 5, 1);
    assert_eq!(
        rows(&map),
        [
            heap(0, 5, DisableMerge::NORMAL),
            heap(5, 1, DisableMerge::NORMAL),
            free(6, PAGES - 6)
        ]
    );
    assert!(map.check_state());
}

#[test]
fn an_update_that_starts_inside_a_block_sets_no_barrier_on_what_it_leaves_below() {
    let mut map = map();
    map_heap(&mut map, 0, 6);
    // Reprotect the last two pages: the lower part keeps the head's barrier, the changed part
    // gets none (the update asks for none).
    map.update(
        page(4),
        2,
        MemoryState::Normal,
        MemoryPermission::USER_READ,
        MemoryAttribute::NONE,
        DisableMerge::NONE,
        DisableMerge::NONE,
    )
    .unwrap();
    let read_only = (
        page(4),
        2,
        MemoryState::Normal,
        MemoryPermission::USER_READ,
        MemoryAttribute::NONE,
        DisableMerge::NONE,
    );
    assert_eq!(
        rows(&map),
        [
            heap(0, 4, DisableMerge::NORMAL),
            read_only,
            free(6, PAGES - 6)
        ]
    );
    // Back to read-write: it merges into the block below, which keeps its barrier.
    map.update(
        page(4),
        2,
        MemoryState::Normal,
        MemoryPermission::USER_READ_WRITE,
        MemoryAttribute::NONE,
        DisableMerge::NONE,
        DisableMerge::NONE,
    )
    .unwrap();
    assert_eq!(
        rows(&map),
        [heap(0, 6, DisableMerge::NORMAL), free(6, PAGES - 6)]
    );
}

#[test]
fn an_update_over_blocks_that_already_match_changes_only_the_others() {
    let mut map = map();
    map_heap(&mut map, 2, 2);
    // Heap over pages 0..6: pages 2 and 3 already are, so only the two ends change. The first
    // block of the range changes, so the barrier is set at page 0; the block at page 2 keeps
    // its own, so the three stay apart there.
    map_heap(&mut map, 0, 6);
    assert_eq!(
        rows(&map),
        [
            heap(0, 2, DisableMerge::NORMAL),
            heap(2, 4, DisableMerge::NORMAL),
            free(6, PAGES - 6)
        ]
    );
    // The same update again changes nothing at all.
    let before = map.clone();
    map_heap(&mut map, 0, 6);
    assert_eq!(map, before);
}

#[test]
fn an_attribute_update_splits_and_merges_like_any_other() {
    let mut map = map();
    map_heap(&mut map, 0, 6);
    map.update_attribute(
        page(2),
        2,
        MemoryAttribute::PERMISSION_LOCKED,
        MemoryAttribute::PERMISSION_LOCKED,
    )
    .unwrap();
    let locked = (
        page(2),
        2,
        MemoryState::Normal,
        MemoryPermission::USER_READ_WRITE,
        MemoryAttribute::PERMISSION_LOCKED,
        DisableMerge::NONE,
    );
    assert_eq!(
        rows(&map),
        [
            heap(0, 2, DisableMerge::NORMAL),
            locked,
            heap(4, 2, DisableMerge::NONE),
            free(6, PAGES - 6)
        ]
    );
    // Clearing the bit over a wider range touches only the block that has it.
    map.update_attribute(
        BASE,
        PAGES,
        MemoryAttribute::PERMISSION_LOCKED,
        MemoryAttribute::NONE,
    )
    .unwrap();
    assert_eq!(
        rows(&map),
        [heap(0, 6, DisableMerge::NORMAL), free(6, PAGES - 6)]
    );
    assert!(map.check_state());
}

#[test]
fn a_range_outside_the_map_or_off_a_page_is_refused_and_changes_nothing() {
    let mut map = map();
    map_heap(&mut map, 1, 1);
    let before = map.clone();
    let refused = [
        // No pages.
        (BASE, 0),
        // Off a page.
        (BASE + 1, 1),
        // Starts below, ends above, starts at the end.
        (BASE - PAGE_SIZE, 2),
        (page(PAGES - 1), 2),
        (page(PAGES), 1),
        // A size or an end that overflows.
        (BASE, u64::MAX / PAGE_SIZE + 1),
        (BASE, u64::MAX / PAGE_SIZE),
    ];
    for (address, pages) in refused {
        let error = Err(BlockMapError::BadRange { address, pages });
        assert_eq!(
            map.update(
                address,
                pages,
                MemoryState::Stack,
                MemoryPermission::USER_READ_WRITE,
                MemoryAttribute::NONE,
                DisableMerge::NONE,
                DisableMerge::NONE,
            ),
            error,
            "{address:#x} {pages:#x}"
        );
        assert_eq!(
            map.update_attribute(
                address,
                pages,
                MemoryAttribute::UNCACHED,
                MemoryAttribute::UNCACHED
            ),
            error
        );
        assert_eq!(map, before);
    }
    // Exactly the whole map is inside.
    assert_eq!(
        map.check_update(
            BASE,
            PAGES,
            MemoryState::Stack,
            MemoryPermission::USER_READ_WRITE,
            MemoryAttribute::NONE
        ),
        Ok(())
    );
    assert_eq!(
        BlockMapError::BadRange {
            address: 0x2000,
            pages: 3
        }
        .to_string(),
        "0x3 pages at 0x2000 are not a run of pages inside the block map"
    );
}

#[test]
fn a_counted_attribute_cannot_be_set_by_an_update() {
    let mut map = map();
    let before = map.clone();
    for attribute in [MemoryAttribute::IPC_LOCKED, MemoryAttribute::DEVICE_SHARED] {
        assert_eq!(
            map.update(
                BASE,
                1,
                MemoryState::Normal,
                MemoryPermission::USER_READ_WRITE,
                attribute,
                DisableMerge::NONE,
                DisableMerge::NONE,
            ),
            Err(BlockMapError::CountedAttribute)
        );
        assert_eq!(
            map.update_attribute(BASE, 1, attribute, attribute),
            Err(BlockMapError::CountedAttribute)
        );
        assert_eq!(
            map.update_attribute(BASE, 1, attribute, MemoryAttribute::NONE),
            Err(BlockMapError::CountedAttribute)
        );
    }
    assert_eq!(map, before);
}

#[test]
fn an_update_over_a_block_locked_for_ipc_is_refused_and_changes_nothing() {
    let mut map = map();
    map_heap(&mut map, 0, 4);
    // No operation of this slice locks a block, so lock one by hand, as `LockForIpc` would.
    let mut locked = *map.find(page(2)).unwrap();
    let lower = locked.split(page(2)).unwrap();
    locked.update_attribute(MemoryAttribute::IPC_LOCKED, MemoryAttribute::IPC_LOCKED);
    map.blocks.insert(lower.address(), lower);
    map.blocks.insert(locked.address(), locked);
    let before = map.clone();

    let update = |map: &mut BlockMap, first: u64, pages: u64, permission| {
        map.update(
            page(first),
            pages,
            MemoryState::Normal,
            permission,
            MemoryAttribute::NONE,
            DisableMerge::NONE,
            DisableMerge::NONE,
        )
    };
    // The range reaches the locked block, in its last page only.
    assert_eq!(
        update(&mut map, 0, 3, MemoryPermission::USER_READ),
        Err(BlockMapError::Locked { address: page(2) })
    );
    assert_eq!(map, before);
    // An update that would leave the locked block as it is goes through, and so does one
    // that stops just below it.
    assert_eq!(
        update(&mut map, 2, 2, MemoryPermission::USER_READ_WRITE),
        Ok(())
    );
    assert_eq!(map, before);
    assert_eq!(update(&mut map, 0, 2, MemoryPermission::USER_READ), Ok(()));
}

#[test]
fn a_broken_map_fails_its_check() {
    let whole = map();
    assert!(whole.check_state());
    // A hole.
    let mut holed = whole.clone();
    map_heap(&mut holed, 1, 1);
    holed.blocks.remove(&page(1));
    assert!(!holed.check_state());
    // Two neighbours that could be merged.
    let mut unmerged = whole.clone();
    let mut upper = *unmerged.find(BASE).unwrap();
    let lower = upper.split(page(1)).unwrap();
    unmerged.blocks.insert(lower.address(), lower);
    unmerged.blocks.insert(upper.address(), upper);
    assert!(!unmerged.check_state());
    // A block that does not reach the end of the map.
    let mut short = whole.clone();
    short.end += PAGE_SIZE;
    assert!(!short.check_state());
    // A lock attribute with no count behind it.
    for attribute in [MemoryAttribute::IPC_LOCKED, MemoryAttribute::DEVICE_SHARED] {
        let mut uncounted = whole.clone();
        let block = uncounted.blocks.get_mut(&BASE).unwrap();
        block.update_attribute(attribute, attribute);
        assert!(!uncounted.check_state(), "{attribute:?}");
    }
}

// ----- FindFreeArea ---------------------------------------------------------------------------

#[test]
fn a_free_area_leaves_guard_pages_on_both_sides() {
    let map = map();
    // An empty region: the first place is `guard` pages in.
    assert_eq!(
        map.find_free_area(BASE, PAGES, 2, PAGE_SIZE, 0, 4),
        Some(page(4))
    );
    // It fits exactly when the region has room for the pages and both guards.
    assert_eq!(
        map.find_free_area(BASE, 2 + 2 * 4, 2, PAGE_SIZE, 0, 4),
        Some(page(4))
    );
    assert_eq!(
        map.find_free_area(BASE, 2 + 2 * 4 - 1, 2, PAGE_SIZE, 0, 4),
        None
    );
    // Without guards the first page will do.
    assert_eq!(map.find_free_area(BASE, 1, 1, PAGE_SIZE, 0, 0), Some(BASE));
    assert_eq!(map.find_free_area(BASE, PAGES, 0, PAGE_SIZE, 0, 0), None);
}

#[test]
fn a_free_area_skips_mapped_blocks_and_keeps_its_distance_from_them() {
    let mut map = map();
    // Pages 6..8 are taken. One page with one guard page fits below (pages 1..=3 are free
    // with a guard on each side of them).
    map_heap(&mut map, 6, 2);
    assert_eq!(
        map.find_free_area(BASE, PAGES, 1, PAGE_SIZE, 0, 1),
        Some(page(1))
    );
    // Five pages with a guard on each side need seven free pages: pages 0..6 are six, so the
    // area goes above the mapping, one guard page clear of it.
    assert_eq!(
        map.find_free_area(BASE, PAGES, 5, PAGE_SIZE, 0, 1),
        Some(page(9))
    );
    // Four pages with their guards fit the six free pages below exactly.
    assert_eq!(
        map.find_free_area(BASE, PAGES, 4, PAGE_SIZE, 0, 1),
        Some(page(1))
    );
    // A region that starts inside the mapping finds the space above it.
    assert_eq!(
        map.find_free_area(page(7), PAGES - 7, 1, PAGE_SIZE, 0, 0),
        Some(page(8))
    );
    // Nothing fits when the region ends inside the lower free block and the guards do not fit.
    assert_eq!(map.find_free_area(BASE, 2, 1, PAGE_SIZE, 0, 1), None);
}

#[test]
fn a_free_area_honours_alignment_and_offset() {
    let map = BlockMap::new(0, 64 * PAGE_SIZE).unwrap();
    let alignment = 8 * PAGE_SIZE;
    // Guard pages push the start to page 1; the next multiple of eight pages is page 8.
    assert_eq!(
        map.find_free_area(0, 64, 1, alignment, 0, 1),
        Some(alignment)
    );
    // With an offset of three pages the first candidate is page 3, which is past the guard.
    assert_eq!(
        map.find_free_area(0, 64, 1, alignment, 3 * PAGE_SIZE, 1),
        Some(3 * PAGE_SIZE)
    );
    // An alignment that is not a power of two is refused.
    assert_eq!(map.find_free_area(0, 64, 1, 3 * PAGE_SIZE, 0, 0), None);
}

#[test]
fn a_free_area_search_that_overflows_finds_nothing() {
    let top = u64::MAX - (PAGE_SIZE - 1);
    let map = BlockMap::new(top - 8 * PAGE_SIZE, top).unwrap();
    let start = map.start();
    // The region's end overflows.
    assert_eq!(
        map.find_free_area(start, u64::MAX, 1, PAGE_SIZE, 0, 0),
        None
    );
    assert_eq!(
        map.find_free_area(start, 2 * (u64::MAX / PAGE_SIZE), 1, PAGE_SIZE, 0, 0),
        None
    );
    // The request's size or its guards overflow.
    assert_eq!(
        map.find_free_area(start, 8, u64::MAX, PAGE_SIZE, 0, 0),
        None
    );
    assert_eq!(
        map.find_free_area(start, 8, 1, PAGE_SIZE, 0, u64::MAX),
        None
    );
    // The candidate plus the offset overflows.
    assert_eq!(
        map.find_free_area(start, 8, 1, PAGE_SIZE, u64::MAX, 0),
        None
    );
    // The region as it is still has room.
    assert_eq!(
        map.find_free_area(start, 8, 8, PAGE_SIZE, 0, 0),
        Some(start)
    );
}

// ----- The model -----------------------------------------------------------------------------

/// What the naive model knows about one page: its properties, and whether a block that must
/// not be merged with the one below starts here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Page {
    state: MemoryState,
    permission: MemoryPermission,
    attribute: MemoryAttribute,
    barrier: bool,
}

impl Page {
    const FREE: Self = Self {
        state: MemoryState::Free,
        permission: MemoryPermission::NONE,
        attribute: MemoryAttribute::NONE,
        barrier: false,
    };

    fn properties(self) -> (MemoryState, MemoryPermission, MemoryAttribute) {
        (self.state, self.permission, self.attribute)
    }
}

/// The address space as a list of pages. It knows nothing of blocks: a block is a maximal run
/// of pages with the same properties that no barrier interrupts.
struct Model(Vec<Page>);

impl Model {
    fn new() -> Self {
        Self(vec![Page::FREE; usize::try_from(PAGES).unwrap()])
    }

    /// Whether a block starts at page `index`.
    fn is_head(&self, index: usize) -> bool {
        index == 0
            || self.0[index].barrier
            || self.0[index].properties() != self.0[index - 1].properties()
    }

    /// `KMemoryBlockManager::Update`, page by page. A page that already has the properties is
    /// left alone. Of the pages that change, the one at the start of the range gets the
    /// barrier if `set`; a page that started a block before loses its barrier if `clear`.
    fn update(
        &mut self,
        first: usize,
        count: usize,
        properties: (MemoryState, MemoryPermission, MemoryAttribute),
        set: bool,
        clear: bool,
    ) {
        for index in first..first + count {
            if self.0[index].properties() == properties {
                continue;
            }
            let page = &mut self.0[index];
            (page.state, page.permission, page.attribute) = properties;
            if index == first && set {
                page.barrier = true;
            }
            if clear {
                page.barrier = false;
            }
        }
    }

    /// `KMemoryBlockManager::UpdateAttribute`, page by page.
    fn update_attribute(
        &mut self,
        first: usize,
        count: usize,
        mask: MemoryAttribute,
        attribute: MemoryAttribute,
    ) {
        for page in &mut self.0[first..first + count] {
            if page.attribute & mask != attribute {
                page.attribute = (page.attribute & !mask) | attribute;
            }
        }
    }

    /// The blocks the pages imply.
    fn rows(&self) -> Vec<Row> {
        let mut rows: Vec<Row> = Vec::new();
        for (index, page) in self.0.iter().enumerate() {
            if self.is_head(index) {
                let barrier = if page.barrier {
                    DisableMerge::NORMAL
                } else {
                    DisableMerge::NONE
                };
                rows.push((
                    BASE + PAGE_SIZE * u64::try_from(index).unwrap(),
                    1,
                    page.state,
                    page.permission,
                    page.attribute,
                    barrier,
                ));
            } else {
                rows.last_mut().unwrap().1 += 1;
            }
        }
        rows
    }
}

/// One step of a generated sequence. Indexes pick from the lists below.
#[derive(Debug, Clone, Copy)]
enum Op {
    Update {
        first: u64,
        count: u64,
        state: usize,
        permission: usize,
        attribute: usize,
        set: bool,
        clear: bool,
    },
    Attribute {
        first: u64,
        count: u64,
        mask: usize,
        attribute: usize,
    },
}

const STATES: [MemoryState; 4] = [
    MemoryState::Free,
    MemoryState::Normal,
    MemoryState::CodeData,
    MemoryState::Stack,
];
const PERMISSIONS: [MemoryPermission; 3] = [
    MemoryPermission::NONE,
    MemoryPermission::USER_READ,
    MemoryPermission::USER_READ_WRITE,
];
/// Attributes an update may set, and the masks and values an attribute update may use: every
/// subset of the two settable bits.
const ATTRIBUTES: [MemoryAttribute; 4] = [
    MemoryAttribute::NONE,
    MemoryAttribute::UNCACHED,
    MemoryAttribute::PERMISSION_LOCKED,
    MemoryAttribute::SET_MASK,
];

fn range() -> impl Strategy<Value = (u64, u64)> {
    (0..PAGES).prop_flat_map(|first| (Just(first), 1..=PAGES - first))
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (
            range(),
            0..STATES.len(),
            0..PERMISSIONS.len(),
            0..ATTRIBUTES.len(),
            any::<bool>(),
            any::<bool>()
        )
            .prop_map(
                |((first, count), state, permission, attribute, set, clear)| Op::Update {
                    first,
                    count,
                    state,
                    permission,
                    attribute,
                    set,
                    clear,
                }
            ),
        (range(), 0..ATTRIBUTES.len(), 0..ATTRIBUTES.len()).prop_map(
            |((first, count), mask, attribute)| Op::Attribute {
                first,
                count,
                mask,
                attribute,
            }
        ),
    ]
}

proptest! {
    #![proptest_config(config())]

    /// Random updates and attribute updates leave the map with exactly the blocks a
    /// page-by-page model implies: every page has the model's state, permission and
    /// attributes, blocks start exactly where the model's properties change or a barrier
    /// stands (so everything that can be merged is, and nothing else), and the map passes its
    /// own check after every step.
    #[test]
    fn agrees_with_a_page_by_page_model(ops in proptest::collection::vec(op(), 1..40)) {
        let mut map = map();
        let mut model = Model::new();
        for op in ops {
            match op {
                Op::Update { first, count, state, permission, attribute, set, clear } => {
                    let properties = (STATES[state], PERMISSIONS[permission], ATTRIBUTES[attribute]);
                    let barrier = |on: bool| if on { DisableMerge::NORMAL } else { DisableMerge::NONE };
                    map.update(
                        page(first),
                        count,
                        properties.0,
                        properties.1,
                        properties.2,
                        barrier(set),
                        barrier(clear),
                    )
                    .unwrap();
                    model.update(
                        usize::try_from(first).unwrap(),
                        usize::try_from(count).unwrap(),
                        properties,
                        set,
                        clear,
                    );
                }
                Op::Attribute { first, count, mask, attribute } => {
                    // The value must lie inside the mask, as the system call requires.
                    let (mask, attribute) = (ATTRIBUTES[mask], ATTRIBUTES[attribute] & ATTRIBUTES[mask]);
                    map.update_attribute(page(first), count, mask, attribute).unwrap();
                    model.update_attribute(
                        usize::try_from(first).unwrap(),
                        usize::try_from(count).unwrap(),
                        mask,
                        attribute,
                    );
                }
            }
            prop_assert_eq!(rows(&map), model.rows(), "after {:?}", op);
            prop_assert!(map.check_state(), "after {:?}", op);
        }
    }

    /// Whatever the map holds, a free area that is found is free, inside its region, aligned,
    /// and has its guard pages free on both sides; and when none is found, the lowest page
    /// with room (searched page by page) does not exist either.
    #[test]
    fn a_found_free_area_is_free_with_its_guards_and_the_lowest_there_is(
        ops in proptest::collection::vec(op(), 0..12),
        (region_first, region_pages) in range(),
        pages in 1..6_u64,
        guard in 0..3_u64,
        alignment_pages in prop_oneof![Just(1_u64), Just(2), Just(4)],
    ) {
        let mut map = map();
        for op in ops {
            if let Op::Update { first, count, state, permission, attribute, set, clear } = op {
                let barrier = |on: bool| if on { DisableMerge::NORMAL } else { DisableMerge::NONE };
                map.update(
                    page(first), count, STATES[state], PERMISSIONS[permission],
                    ATTRIBUTES[attribute], barrier(set), barrier(clear),
                ).unwrap();
            }
        }
        let alignment = alignment_pages * PAGE_SIZE;
        let found = map.find_free_area(page(region_first), region_pages, pages, alignment, 0, guard);

        // The same question asked page by page. A candidate needs its pages and both guards
        // inside one free block, the guards inside the region too, and the alignment.
        let is_free = |index: u64| {
            map.find(page(index)).is_some_and(|block| block.state() == MemoryState::Free)
        };
        let same_block = |a: u64, b: u64| {
            map.find(page(a)).map(MemoryBlock::address) == map.find(page(b)).map(MemoryBlock::address)
        };
        let region_end = region_first + region_pages;
        let expected = (region_first..region_end).find(|&candidate| {
            let (low, high) = (candidate.checked_sub(guard), candidate + pages + guard);
            page(candidate).is_multiple_of(alignment)
                && low.is_some_and(|low| low >= region_first
                    && high <= region_end
                    && (low..high).all(is_free)
                    && same_block(low, high - 1))
        });
        prop_assert_eq!(found, expected.map(page));
    }
}
