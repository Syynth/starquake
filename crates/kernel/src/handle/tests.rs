//! Tests for the handle table: encoding fixtures worked out by hand from the reference's bit
//! layout, the free-list and linear-id behaviour, pseudo-handles, the size limit and its result,
//! reference counting, serialisation (including damaged data) and a property test against a map.

use std::collections::BTreeMap;
use std::num::NonZeroU32;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use super::*;
use crate::id::SlabId;

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
const PROPTEST_SEED: u64 = 0x4841_4E44_4C45_5F31;

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 256,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(PROPTEST_SEED),
        failure_persistence: None,
        ..Config::default()
    }
}

fn obj(n: u32) -> ObjectId {
    ObjectId::from_parts(n, 1)
}

fn table(size: i32) -> HandleTable {
    HandleTable::new(size).unwrap()
}

fn invalid(handle: Handle) -> HandleError {
    HandleError::InvalidHandle { raw: handle.raw() }
}

// --- Encoding -------------------------------------------------------------------------------

/// `(index, linear id, raw)`, computed from `index | linear_id << 15`.
const ENCODING_FIXTURES: [(u16, u16, u32); 6] = [
    (0, 1, 0x0000_8000),
    (1, 2, 0x0001_0001),
    (5, 0x1234, 0x091A_0005),
    (1023, 1, 0x0000_83FF),
    (1023, 0x7FFF, 0x3FFF_83FF),
    (0x7FFF, 0x7FFF, 0x3FFF_FFFF),
];

#[test]
fn encoding_fixtures() {
    for (index, linear_id, raw) in ENCODING_FIXTURES {
        let handle = Handle::from_parts(index, linear_id).unwrap();
        assert_eq!(handle.raw(), raw, "encode ({index}, {linear_id})");
        let back = Handle::from_raw(raw);
        assert_eq!(back, handle);
        assert_eq!(back.index(), u32::from(index), "index of {raw:#x}");
        assert_eq!(
            back.linear_id(),
            u32::from(linear_id),
            "linear id of {raw:#x}"
        );
        assert_eq!(back.reserved(), 0, "reserved bits of {raw:#x}");
    }
}

#[test]
fn field_widths_follow_the_layout() {
    assert_eq!((INDEX_BITS, LINEAR_ID_BITS, RESERVED_BITS), (15, 15, 2));
    assert_eq!(INDEX_BITS + LINEAR_ID_BITS + RESERVED_BITS, u32::BITS);
    // The largest value of each field uses all of the field and none of its neighbours.
    let top_index = Handle::from_parts(u16::MAX >> 1, 0).unwrap();
    assert_eq!(top_index.raw().count_ones(), INDEX_BITS);
    assert_eq!(top_index.linear_id(), 0);
    let top_linear = Handle::from_parts(0, MAX_LINEAR_ID).unwrap();
    assert_eq!(top_linear.raw().count_ones(), LINEAR_ID_BITS);
    assert_eq!(top_linear.index(), 0);
    assert_eq!(top_linear.reserved(), 0);
}

#[test]
fn fields_that_do_not_fit_are_not_encoded() {
    assert_eq!(Handle::from_parts(0x8000, 1), None);
    assert_eq!(Handle::from_parts(0, 0x8000), None);
    assert_eq!(Handle::from_parts(u16::MAX, u16::MAX), None);
}

#[test]
fn decoding_reads_the_reserved_bits() {
    let handle = Handle::from_raw(0xC000_0000 | ENCODING_FIXTURES[1].2);
    assert_eq!(handle.reserved(), 3);
    assert_eq!((handle.index(), handle.linear_id()), (1, 2));
}

#[test]
fn pseudo_handle_values_and_fields() {
    assert_eq!(Handle::INVALID.raw(), 0);
    assert_eq!(Handle::CURRENT_THREAD.raw(), 0xFFFF_8000);
    assert_eq!(Handle::CURRENT_PROCESS.raw(), 0xFFFF_8001);
    assert!(Handle::CURRENT_THREAD.is_pseudo());
    assert!(Handle::CURRENT_PROCESS.is_pseudo());
    assert!(!Handle::INVALID.is_pseudo());
    assert!(!Handle::from_raw(0xFFFF_8002).is_pseudo());
    // Their reserved bits are set, which is what keeps them out of every table.
    assert_eq!(Handle::CURRENT_THREAD.reserved(), 3);
    assert_eq!(Handle::CURRENT_PROCESS.reserved(), 3);
}

// --- Table behaviour ------------------------------------------------------------------------

#[test]
fn the_first_handles_follow_horizons_free_list() {
    // Initialize links entry i to i - 1 and leaves the head on the last entry, so a table of
    // four hands out indices 3, 2, 1, 0 with linear ids 1, 2, 3, 4.
    let mut t = table(4);
    let mut refs = RefCounts::new();
    let raws: Vec<u32> = (0..4)
        .map(|n| t.add(&mut refs, obj(n)).unwrap().raw())
        .collect();
    assert_eq!(raws, [0x0000_8003, 0x0001_0002, 0x0001_8001, 0x0002_0000]);
}

#[test]
fn a_full_size_table_starts_at_its_last_index() {
    let mut t = table(i32::from(MAX_TABLE_SIZE));
    let mut refs = RefCounts::new();
    let first = t.add(&mut refs, obj(0)).unwrap();
    assert_eq!(
        first,
        Handle::from_parts(MAX_TABLE_SIZE - 1, MIN_LINEAR_ID).unwrap()
    );
    assert_eq!(first.raw(), 0x0000_83FF);
}

#[test]
fn a_removed_slot_is_the_next_one_reused_with_a_fresh_linear_id() {
    let mut t = table(4);
    let mut refs = RefCounts::new();
    let h: Vec<_> = (0..4).map(|n| t.add(&mut refs, obj(n)).unwrap()).collect();
    t.remove(&mut refs, h[1]).unwrap();
    t.remove(&mut refs, h[3]).unwrap();
    // Freed 1 then 3 (the handles at indices 2 and 0), so 3's slot is first.
    let a = t.add(&mut refs, obj(10)).unwrap();
    let b = t.add(&mut refs, obj(11)).unwrap();
    assert_eq!((a.index(), a.linear_id()), (h[3].index(), 5));
    assert_eq!((b.index(), b.linear_id()), (h[1].index(), 6));
}

#[test]
fn get_resolves_and_rejects_stale_handles() {
    let mut t = table(4);
    let mut refs = RefCounts::new();
    let old = t.add(&mut refs, obj(1)).unwrap();
    assert_eq!(t.get(old), Ok(obj(1)));
    t.remove(&mut refs, old).unwrap();
    assert_eq!(t.get(old), Err(invalid(old)));
    // The slot is reused by another object; the old handle must not reach it.
    let new = t.add(&mut refs, obj(2)).unwrap();
    assert_eq!(new.index(), old.index());
    assert_ne!(new, old);
    assert_eq!(t.get(old), Err(invalid(old)));
    assert_eq!(t.get(new), Ok(obj(2)));
    assert_eq!(t.remove(&mut refs, old), Err(invalid(old)));
    assert_eq!(t.get(new), Ok(obj(2)));
}

#[test]
fn handles_that_name_nothing_are_invalid() {
    let mut t = table(4);
    let mut refs = RefCounts::new();
    let h = t.add(&mut refs, obj(1)).unwrap();
    let bad = [
        Handle::INVALID,
        // Right slot, wrong linear id.
        Handle::from_parts(h.index().try_into().unwrap(), 2).unwrap(),
        // Linear id zero.
        Handle::from_parts(h.index().try_into().unwrap(), 0).unwrap(),
        // A free slot of the table.
        Handle::from_parts(0, 1).unwrap(),
        // Beyond the table's size but within the index field.
        Handle::from_parts(4, 1).unwrap(),
        Handle::from_parts(0x7FFF, 1).unwrap(),
        // A valid handle with a reserved bit set.
        Handle::from_raw(h.raw() | (1 << (INDEX_BITS + LINEAR_ID_BITS))),
        Handle::from_raw(h.raw() | (1 << (INDEX_BITS + LINEAR_ID_BITS + 1))),
        Handle::from_raw(u32::MAX),
    ];
    for handle in bad {
        assert_eq!(
            t.get(handle),
            Err(invalid(handle)),
            "get {:#x}",
            handle.raw()
        );
        assert_eq!(
            t.resolve(handle),
            Err(invalid(handle)),
            "resolve {:#x}",
            handle.raw()
        );
        assert_eq!(
            t.remove(&mut refs, handle),
            Err(invalid(handle)),
            "remove {:#x}",
            handle.raw()
        );
    }
    // None of that disturbed the real entry.
    assert_eq!(t.get(h), Ok(obj(1)));
    assert_eq!(refs.count(obj(1)), 1);
}

#[test]
fn pseudo_handles_resolve_but_never_get_or_remove() {
    let mut t = table(4);
    let mut refs = RefCounts::new();
    t.add(&mut refs, obj(1)).unwrap();
    assert_eq!(
        t.resolve(Handle::CURRENT_THREAD),
        Ok(Resolved::CurrentThread)
    );
    assert_eq!(
        t.resolve(Handle::CURRENT_PROCESS),
        Ok(Resolved::CurrentProcess)
    );
    for pseudo in [Handle::CURRENT_THREAD, Handle::CURRENT_PROCESS] {
        assert_eq!(t.get(pseudo), Err(invalid(pseudo)));
        assert_eq!(t.remove(&mut refs, pseudo), Err(invalid(pseudo)));
    }
    assert_eq!(t.len(), 1);
}

#[test]
fn resolve_returns_table_objects() {
    let mut t = table(4);
    let mut refs = RefCounts::new();
    let h = t.add(&mut refs, obj(7)).unwrap();
    assert_eq!(t.resolve(h), Ok(Resolved::Object(obj(7))));
}

#[test]
fn the_table_refuses_to_grow_past_its_size() {
    let mut t = table(2);
    let mut refs = RefCounts::new();
    let a = t.add(&mut refs, obj(1)).unwrap();
    let b = t.add(&mut refs, obj(2)).unwrap();
    let before = (t.clone(), refs.clone());
    assert_eq!(t.add(&mut refs, obj(3)), Err(HandleError::OutOfHandles));
    assert_eq!(
        (t.clone(), refs.clone()),
        before,
        "a refused add changes nothing"
    );
    assert_eq!((t.len(), t.capacity()), (2, 2));
    // Freeing makes room again.
    t.remove(&mut refs, a).unwrap();
    let c = t.add(&mut refs, obj(3)).unwrap();
    assert_eq!(t.get(b), Ok(obj(2)));
    assert_eq!(t.get(c), Ok(obj(3)));
    assert_eq!(t.add(&mut refs, obj(4)), Err(HandleError::OutOfHandles));
}

#[test]
fn a_full_default_size_table_refuses_the_next_handle() {
    let mut t = table(0);
    let mut refs = RefCounts::new();
    for n in 0..u32::from(MAX_TABLE_SIZE) {
        t.add(&mut refs, obj(n)).unwrap();
    }
    assert_eq!(t.add(&mut refs, obj(0)), Err(HandleError::OutOfHandles));
    assert_eq!(t.len(), usize::from(MAX_TABLE_SIZE));
}

#[test]
fn horizons_results_for_the_failures() {
    // Raw values as guests see them: module 1 (kernel) plus description << 9, i.e. the
    // 2001-0104, 2001-0105 and 2001-0114 of the error code tables.
    assert_eq!(
        HandleError::TableTooLarge { requested: 2000 }.result(),
        Some(0xD001)
    );
    assert_eq!(HandleError::OutOfHandles.result(), Some(0xD201));
    assert_eq!(HandleError::InvalidHandle { raw: 0 }.result(), Some(0xE401));
    assert_eq!(HandleError::Corrupt.result(), None);
    assert_eq!(
        HandleError::Refs(RefError::NotReferenced { object: obj(0) }).result(),
        None
    );
}

#[test]
fn table_size_comes_from_configuration() {
    assert_eq!(table(1).capacity(), 1);
    assert_eq!(table(7).capacity(), 7);
    let max = i32::from(MAX_TABLE_SIZE);
    assert_eq!(table(max).capacity(), usize::from(MAX_TABLE_SIZE));
    // Zero and negative sizes mean the maximum, as in Horizon.
    assert_eq!(table(0).capacity(), usize::from(MAX_TABLE_SIZE));
    assert_eq!(table(-1).capacity(), usize::from(MAX_TABLE_SIZE));
    assert_eq!(table(i32::MIN).capacity(), usize::from(MAX_TABLE_SIZE));
}

#[test]
fn a_table_above_the_maximum_is_refused_with_out_of_memory() {
    for requested in [i32::from(MAX_TABLE_SIZE) + 1, 0x7FFF, 0x1_0000, i32::MAX] {
        let err = HandleTable::new(requested).unwrap_err();
        assert_eq!(err, HandleError::TableTooLarge { requested });
        assert_eq!(err.result(), Some(RESULT_OUT_OF_MEMORY));
    }
}

#[test]
fn counts_and_high_water_mark() {
    let mut t = table(8);
    let mut refs = RefCounts::new();
    assert!(t.is_empty());
    assert_eq!((t.len(), t.max_len()), (0, 0));
    let h: Vec<_> = (0..3).map(|n| t.add(&mut refs, obj(n)).unwrap()).collect();
    assert_eq!((t.len(), t.max_len()), (3, 3));
    t.remove(&mut refs, h[0]).unwrap();
    t.remove(&mut refs, h[1]).unwrap();
    assert_eq!((t.len(), t.max_len()), (1, 3));
    t.add(&mut refs, obj(9)).unwrap();
    assert_eq!((t.len(), t.max_len()), (2, 3));
}

#[test]
fn the_linear_id_wraps_after_the_maximum_and_a_stale_handle_revives_then() {
    // One slot, so every add reuses index 0 and the linear id alone tells handles apart.
    let mut t = table(1);
    let mut refs = RefCounts::new();
    let first = t.add(&mut refs, obj(1)).unwrap();
    assert_eq!(first.linear_id(), u32::from(MIN_LINEAR_ID));
    t.remove(&mut refs, first).unwrap();
    let mut last = first;
    for expected in MIN_LINEAR_ID + 1..=MAX_LINEAR_ID {
        last = t.add(&mut refs, obj(2)).unwrap();
        assert_eq!(last.linear_id(), u32::from(expected));
        // Until the counter wraps, the first handle stays rejected.
        assert_eq!(t.get(first), Err(invalid(first)));
        t.remove(&mut refs, last).unwrap();
    }
    assert_eq!(last.linear_id(), u32::from(MAX_LINEAR_ID));
    // The next add wraps to MIN_LINEAR_ID and reissues the very first handle value. This is
    // Horizon's behaviour (a 15-bit counter); a stale handle is only rejected for MAX_LINEAR_ID
    // further adds.
    let wrapped = t.add(&mut refs, obj(3)).unwrap();
    assert_eq!(wrapped, first);
    assert_eq!(t.get(first), Ok(obj(3)));
}

// --- Reference counting ---------------------------------------------------------------------

#[test]
fn closing_the_last_handle_to_an_object_is_observable() {
    let mut t = table(8);
    let mut refs = RefCounts::new();
    let a = t.add(&mut refs, obj(1)).unwrap();
    let b = t.add(&mut refs, obj(1)).unwrap();
    let other = t.add(&mut refs, obj(2)).unwrap();
    assert_eq!(refs.count(obj(1)), 2);
    assert_eq!(refs.count(obj(2)), 1);

    let first = t.remove(&mut refs, a).unwrap();
    assert_eq!(
        first,
        Removed {
            object: obj(1),
            closed: Closed::Remaining(NonZeroU32::MIN)
        }
    );
    assert_eq!(refs.count(obj(1)), 1);

    let last = t.remove(&mut refs, b).unwrap();
    assert_eq!(
        last,
        Removed {
            object: obj(1),
            closed: Closed::Last
        }
    );
    assert_eq!(refs.count(obj(1)), 0);

    // The other object is untouched.
    assert_eq!(refs.count(obj(2)), 1);
    assert_eq!(t.remove(&mut refs, other).unwrap().closed, Closed::Last);
    assert!(refs.is_empty());
}

#[test]
fn other_holders_keep_the_object_alive() {
    // A reference held by something besides the handle table (an in-flight wait) means closing
    // the handle is not the last close.
    let mut t = table(4);
    let mut refs = RefCounts::new();
    let h = t.add(&mut refs, obj(1)).unwrap();
    refs.open(obj(1)).unwrap();
    assert_eq!(
        t.remove(&mut refs, h).unwrap().closed,
        Closed::Remaining(NonZeroU32::MIN)
    );
    assert_eq!(refs.close(obj(1)), Ok(Closed::Last));
}

/// A reference counter that fails on demand.
#[derive(Default)]
struct Failing {
    fail_open: bool,
    fail_close: bool,
    opens: u32,
    closes: u32,
}

impl ObjectRefs for Failing {
    fn open(&mut self, object: ObjectId) -> Result<(), RefError> {
        if self.fail_open {
            return Err(RefError::Overflow { object });
        }
        self.opens += 1;
        Ok(())
    }

    fn close(&mut self, object: ObjectId) -> Result<Closed, RefError> {
        if self.fail_close {
            return Err(RefError::NotReferenced { object });
        }
        self.closes += 1;
        Ok(Closed::Last)
    }
}

#[test]
fn a_failed_open_leaves_the_table_untouched() {
    let mut t = table(4);
    let mut refs = Failing {
        fail_open: true,
        ..Failing::default()
    };
    let before = t.clone();
    assert_eq!(
        t.add(&mut refs, obj(1)),
        Err(HandleError::Refs(RefError::Overflow { object: obj(1) }))
    );
    assert_eq!(t, before);
    // The failed add did not consume a linear id: the next handle is the first one.
    refs.fail_open = false;
    assert_eq!(
        t.add(&mut refs, obj(1)).unwrap().linear_id(),
        u32::from(MIN_LINEAR_ID)
    );
}

#[test]
fn a_failed_close_keeps_the_handle() {
    let mut t = table(4);
    let mut refs = Failing::default();
    let h = t.add(&mut refs, obj(1)).unwrap();
    refs.fail_close = true;
    assert_eq!(
        t.remove(&mut refs, h),
        Err(HandleError::Refs(RefError::NotReferenced {
            object: obj(1)
        }))
    );
    assert_eq!(t.get(h), Ok(obj(1)));
    assert_eq!(t.len(), 1);
    refs.fail_close = false;
    assert_eq!(t.remove(&mut refs, h).unwrap().closed, Closed::Last);
    assert_eq!((refs.opens, refs.closes), (1, 1));
}

#[test]
fn a_full_table_does_not_open_a_reference() {
    let mut t = table(1);
    let mut refs = Failing::default();
    t.add(&mut refs, obj(1)).unwrap();
    assert_eq!(t.add(&mut refs, obj(2)), Err(HandleError::OutOfHandles));
    assert_eq!(refs.opens, 1);
}

// --- Damaged internal state -----------------------------------------------------------------

#[test]
fn a_damaged_free_stack_is_reported_not_trusted() {
    let mut refs = RefCounts::new();
    // Free stack names an occupied slot.
    let mut t = table(2);
    let h = t.add(&mut refs, obj(1)).unwrap();
    t.free.push(h.index().try_into().unwrap());
    assert_eq!(t.add(&mut refs, obj(2)), Err(HandleError::Corrupt));
    // Free stack names a slot past the table.
    let mut t = table(2);
    t.free.push(9);
    assert_eq!(t.add(&mut refs, obj(2)), Err(HandleError::Corrupt));
    // A table whose entry vanished from under a live handle.
    let mut t = table(2);
    let h = t.add(&mut refs, obj(1)).unwrap();
    t.entries.truncate(h.index() as usize);
    assert_eq!(t.get(h), Err(invalid(h)));
}

// --- Serialisation --------------------------------------------------------------------------

/// The serialised shape of a table, built by hand so tests can damage it.
#[derive(Serialize)]
struct RawTable {
    entries: Vec<Option<Entry>>,
    free: Vec<u16>,
    next_linear_id: u16,
    max_count: u16,
}

/// A named way to damage a [`RawTable`].
type Damage = (&'static str, Box<dyn Fn(&mut RawTable)>);

fn raw_of(t: &HandleTable) -> RawTable {
    RawTable {
        entries: t.entries.clone(),
        free: t.free.clone(),
        next_linear_id: t.next_linear_id,
        max_count: t.max_count,
    }
}

fn deserialise(raw: &RawTable) -> Result<HandleTable, postcard::Error> {
    postcard::from_bytes(&postcard::to_allocvec(raw).unwrap())
}

fn used_table() -> (HandleTable, RefCounts, Vec<Handle>) {
    let mut t = table(6);
    let mut refs = RefCounts::new();
    let h: Vec<_> = (0..5)
        .map(|n| t.add(&mut refs, obj(n % 3)).unwrap())
        .collect();
    t.remove(&mut refs, h[1]).unwrap();
    t.remove(&mut refs, h[3]).unwrap();
    (t, refs, h)
}

#[test]
fn serialise_then_deserialise_gives_an_equal_table_that_behaves_the_same() {
    let (mut t, mut refs, handles) = used_table();
    let mut back: HandleTable = postcard::from_bytes(&postcard::to_allocvec(&t).unwrap()).unwrap();
    let mut refs_back: RefCounts =
        postcard::from_bytes(&postcard::to_allocvec(&refs).unwrap()).unwrap();
    assert_eq!(back, t);
    assert_eq!(refs_back, refs);
    for h in &handles {
        assert_eq!(back.get(*h), t.get(*h));
    }
    // Free-list order and the linear id counter survived: the next handles are the same.
    for n in 20..23 {
        assert_eq!(back.add(&mut refs_back, obj(n)), t.add(&mut refs, obj(n)));
    }
    assert_eq!(back, t);
}

#[test]
fn the_wire_form_has_exactly_the_documented_fields() {
    let (t, _, _) = used_table();
    assert_eq!(
        postcard::to_allocvec(&t).unwrap(),
        postcard::to_allocvec(&raw_of(&t)).unwrap()
    );
}

#[test]
fn deserialising_accepts_a_consistent_layout() {
    let (t, _, _) = used_table();
    assert_eq!(deserialise(&raw_of(&t)).unwrap(), t);
}

#[test]
fn deserialising_rejects_damaged_layouts() {
    let damage: Vec<Damage> = vec![
        (
            "no entries",
            Box::new(|r| {
                r.entries.clear();
                r.free.clear();
                r.max_count = 0;
            }),
        ),
        (
            "too many entries",
            Box::new(|r| {
                r.entries = vec![None; usize::from(MAX_TABLE_SIZE) + 1];
                r.free = (0..=MAX_TABLE_SIZE).collect();
            }),
        ),
        ("free entry out of range", Box::new(|r| r.free[0] = 99)),
        (
            "free entry names an occupied slot",
            Box::new(|r| {
                let occupied = r.entries.iter().position(Option::is_some).unwrap();
                r.free[0] = u16::try_from(occupied).unwrap();
            }),
        ),
        ("free entry repeated", Box::new(|r| r.free[1] = r.free[0])),
        (
            "free entry missing",
            Box::new(|r| {
                r.free.pop();
            }),
        ),
        (
            "linear id zero",
            Box::new(|r| {
                let slot = r.entries.iter_mut().flatten().next().unwrap();
                slot.linear_id = 0;
            }),
        ),
        (
            "linear id too large",
            Box::new(|r| {
                let slot = r.entries.iter_mut().flatten().next().unwrap();
                slot.linear_id = MAX_LINEAR_ID + 1;
            }),
        ),
        ("next linear id zero", Box::new(|r| r.next_linear_id = 0)),
        (
            "next linear id too large",
            Box::new(|r| r.next_linear_id = MAX_LINEAR_ID + 1),
        ),
        (
            "high-water mark below the entries in use",
            Box::new(|r| r.max_count = 2),
        ),
        (
            "high-water mark above the table size",
            Box::new(|r| r.max_count = 7),
        ),
    ];
    for (name, apply) in damage {
        let (t, _, _) = used_table();
        let mut raw = raw_of(&t);
        apply(&mut raw);
        assert!(deserialise(&raw).is_err(), "{name} was accepted");
    }
}

#[test]
fn each_layout_error_is_distinguishable() {
    let (t, _, _) = used_table();
    let try_data = |apply: &dyn Fn(&mut HandleTableData)| {
        let mut data = HandleTableData {
            entries: t.entries.clone(),
            free: t.free.clone(),
            next_linear_id: t.next_linear_id,
            max_count: t.max_count,
        };
        apply(&mut data);
        HandleTable::try_from(data).err()
    };
    assert_eq!(
        try_data(&|d| d.entries.clear()),
        Some(HandleLayoutError::Size { size: 0 })
    );
    assert_eq!(
        try_data(&|d| d.next_linear_id = 0),
        Some(HandleLayoutError::NextLinearId(0))
    );
    assert_eq!(
        try_data(&|d| d.max_count = 7),
        Some(HandleLayoutError::MaxCount(7))
    );
    assert_eq!(
        try_data(&|d| d.free[0] = 99),
        Some(HandleLayoutError::FreeStack(FreeStackError::OutOfRange {
            index: 99
        }))
    );
    assert_eq!(
        try_data(&|d| {
            d.free.pop();
        }),
        Some(HandleLayoutError::FreeCount)
    );
    let occupied = t.entries.iter().position(Option::is_some).unwrap();
    assert_eq!(
        try_data(&|d| d.entries[occupied].as_mut().unwrap().linear_id = 0),
        Some(HandleLayoutError::LinearId {
            index: occupied,
            linear_id: 0
        })
    );
}

#[test]
fn handles_serialise_as_their_raw_value() {
    let h = Handle::from_raw(0x091A_0005);
    assert_eq!(
        postcard::to_allocvec(&h).unwrap(),
        postcard::to_allocvec(&0x091A_0005_u32).unwrap()
    );
}

// --- Property test --------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Op {
    /// Add one of a few objects, so several handles share an object.
    Add(u32),
    /// Index into every handle ever issued (live or stale), modulo its length.
    Get(usize),
    Remove(usize),
}

fn ops() -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(
        prop_oneof![
            4 => (0..4_u32).prop_map(Op::Add),
            2 => any::<usize>().prop_map(Op::Get),
            3 => any::<usize>().prop_map(Op::Remove),
        ],
        0..200,
    )
}

proptest! {
    #![proptest_config(config())]

    /// Random add, get and remove sequences on a small table agree with a map model. After every
    /// step each handle ever issued is checked: live ones resolve to the model's object, freed
    /// ones (including those whose slot was reused) are rejected. Reference counts match the
    /// number of live handles per object, `Closed::Last` is reported exactly when the last handle
    /// closes, and a full table refuses exactly when the model holds `size` handles.
    ///
    /// Sequences are far shorter than `MAX_LINEAR_ID` adds, so no handle value repeats and
    /// "freed means rejected" holds throughout (the wrap is pinned by its own test).
    #[test]
    fn agrees_with_a_map_model(size in 1..8_i32, ops in ops()) {
        let capacity = usize::try_from(size).unwrap();
        let mut t = HandleTable::new(size).unwrap();
        let mut refs = RefCounts::new();
        let mut model: BTreeMap<Handle, ObjectId> = BTreeMap::new();
        let mut issued: Vec<Handle> = Vec::new();
        let mut high_water = 0;
        for op in ops {
            match op {
                Op::Add(n) => match t.add(&mut refs, obj(n)) {
                    Ok(handle) => {
                        prop_assert!(model.len() < capacity, "added to a full table");
                        prop_assert!(!issued.contains(&handle), "handle {handle:?} issued twice");
                        issued.push(handle);
                        model.insert(handle, obj(n));
                        high_water = high_water.max(model.len());
                    }
                    Err(err) => {
                        prop_assert_eq!(err, HandleError::OutOfHandles);
                        prop_assert_eq!(model.len(), capacity);
                    }
                },
                Op::Get(pick) if !issued.is_empty() => {
                    let handle = issued[pick % issued.len()];
                    prop_assert_eq!(t.get(handle).ok(), model.get(&handle).copied());
                }
                Op::Remove(pick) if !issued.is_empty() => {
                    let handle = issued[pick % issued.len()];
                    let held = model.remove(&handle);
                    let result = t.remove(&mut refs, handle);
                    if let Some(object) = held {
                        let holders = model.values().filter(|o| **o == object).count();
                        let left = u32::try_from(holders).ok().and_then(NonZeroU32::new);
                        let closed = left.map_or(Closed::Last, Closed::Remaining);
                        prop_assert_eq!(result, Ok(Removed { object, closed }));
                    } else {
                        prop_assert_eq!(result, Err(invalid(handle)));
                    }
                }
                Op::Get(_) | Op::Remove(_) => {}
            }
            prop_assert_eq!(t.len(), model.len());
            prop_assert_eq!(t.max_len(), high_water);
            for &handle in &issued {
                let expected = model.get(&handle).copied().ok_or(invalid(handle));
                prop_assert_eq!(t.get(handle), expected);
            }
            for n in 0..4 {
                let holders = model.values().filter(|o| **o == obj(n)).count();
                prop_assert_eq!(refs.count(obj(n)) as usize, holders);
            }
        }
        // Whatever state the sequence reached survives serialisation and behaves identically.
        let mut back: HandleTable = postcard::from_bytes(&postcard::to_allocvec(&t).unwrap()).unwrap();
        let mut refs_back = refs.clone();
        prop_assert_eq!(&back, &t);
        prop_assert_eq!(back.add(&mut refs_back, obj(0)), t.add(&mut refs, obj(0)));
    }
}
