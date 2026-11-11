//! Tests for the slab: fixtures for the id sequence, staleness, slot reuse order, limits,
//! generation exhaustion, serialisation (including damaged data) and a property test against a
//! map model.

use std::collections::BTreeMap;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use super::*;
use crate::id::{ObjectId, ThreadId};

type TestSlab = Slab<ObjectId, i64>;

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
const PROPTEST_SEED: u64 = 0x534C_4142_5F4D_4F44;

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 256,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(PROPTEST_SEED),
        failure_persistence: None,
        ..Config::default()
    }
}

/// The serialised shape of a slab, built by hand so tests can damage it.
#[derive(Serialize)]
struct RawSlab {
    slots: Vec<Slot<i64>>,
    free: Vec<u32>,
    max_slots: u32,
    live: u32,
}

fn deserialise(raw: &RawSlab) -> Result<TestSlab, postcard::Error> {
    postcard::from_bytes(&postcard::to_allocvec(raw).unwrap())
}

fn stale(id: ObjectId) -> SlabError {
    SlabError::Stale {
        index: id.index(),
        generation: id.generation(),
    }
}

#[test]
fn first_ids_are_sequential_with_generation_one() {
    let mut slab = TestSlab::new();
    let a = slab.insert(10).unwrap();
    let b = slab.insert(20).unwrap();
    assert_eq!((a.index(), a.generation()), (0, FIRST_GENERATION));
    assert_eq!((b.index(), b.generation()), (1, FIRST_GENERATION));
    assert_eq!(slab.len(), 2);
    assert_eq!(slab.get(a), Ok(&10));
    assert_eq!(slab.get(b), Ok(&20));
}

#[test]
fn a_freed_id_is_stale() {
    let mut slab = TestSlab::new();
    let a = slab.insert(1).unwrap();
    assert_eq!(slab.remove(a), Ok(1));
    assert_eq!(slab.get(a), Err(stale(a)));
    assert_eq!(slab.get_mut(a), Err(stale(a)));
    assert_eq!(slab.remove(a), Err(stale(a)));
    assert!(!slab.contains(a));
    assert!(slab.is_empty());
}

#[test]
fn a_reused_slot_gets_a_new_generation_and_the_old_id_stays_stale() {
    let mut slab = TestSlab::new();
    let old = slab.insert(1).unwrap();
    slab.remove(old).unwrap();
    let new = slab.insert(2).unwrap();
    assert_eq!(new.index(), old.index());
    assert_eq!(new.generation(), old.generation() + 1);
    // The old id must not reach the new occupant.
    assert_eq!(slab.get(old), Err(stale(old)));
    assert_eq!(slab.remove(old), Err(stale(old)));
    assert_eq!(slab.get(new), Ok(&2));
}

#[test]
fn a_failed_remove_leaves_the_row_in_place() {
    let mut slab = TestSlab::new();
    let live = slab.insert(5).unwrap();
    let wrong_generation = ObjectId::from_parts(live.index(), live.generation() + 1);
    assert_eq!(slab.remove(wrong_generation), Err(stale(wrong_generation)));
    assert_eq!(slab.get(live), Ok(&5));
    assert_eq!(slab.len(), 1);
}

#[test]
fn ids_that_were_never_issued_are_stale() {
    let mut slab = TestSlab::new();
    let a = slab.insert(1).unwrap();
    // Generation zero is never issued, even for a live slot.
    let zero = ObjectId::from_parts(a.index(), 0);
    assert_eq!(slab.get(zero), Err(stale(zero)));
    // An index beyond the slab.
    let far = ObjectId::from_parts(u32::MAX, a.generation());
    assert_eq!(slab.get(far), Err(stale(far)));
    assert_eq!(slab.remove(far), Err(stale(far)));
}

#[test]
fn slots_are_reused_most_recently_freed_first() {
    let mut slab = TestSlab::new();
    let ids: Vec<_> = (0..4).map(|v| slab.insert(v).unwrap()).collect();
    slab.remove(ids[1]).unwrap();
    slab.remove(ids[3]).unwrap();
    slab.remove(ids[0]).unwrap();
    // Freed order was 1, 3, 0, so reuse order is 0, 3, 1, then a fresh slot.
    let reused: Vec<u32> = (0..4).map(|v| slab.insert(v).unwrap().index()).collect();
    assert_eq!(reused, [0, 3, 1, 4]);
}

#[test]
fn the_same_operations_give_the_same_ids() {
    fn run() -> Vec<ObjectId> {
        let mut slab = TestSlab::new();
        let mut issued = vec![slab.insert(0).unwrap(), slab.insert(1).unwrap()];
        slab.remove(issued[0]).unwrap();
        issued.push(slab.insert(2).unwrap());
        slab.remove(issued[1]).unwrap();
        slab.remove(issued[2]).unwrap();
        issued.push(slab.insert(3).unwrap());
        issued.push(slab.insert(4).unwrap());
        issued
    }
    assert_eq!(run(), run());
}

#[test]
fn a_full_slab_refuses_to_grow_but_reuses_freed_slots() {
    let mut slab = TestSlab::with_max_slots(2);
    assert_eq!(slab.max_slots(), 2);
    let a = slab.insert(1).unwrap();
    slab.insert(2).unwrap();
    assert_eq!(slab.insert(3), Err(SlabError::Full { max_slots: 2 }));
    assert_eq!(slab.len(), 2);
    slab.remove(a).unwrap();
    let c = slab.insert(3).unwrap();
    assert_eq!(c.index(), a.index());
    assert_eq!(slab.insert(4), Err(SlabError::Full { max_slots: 2 }));
}

#[test]
fn a_zero_sized_slab_is_always_full() {
    let mut slab = TestSlab::with_max_slots(0);
    assert_eq!(slab.insert(1), Err(SlabError::Full { max_slots: 0 }));
    assert!(slab.is_empty());
}

#[test]
fn an_exhausted_generation_retires_the_slot() {
    let mut slab = TestSlab::new();
    let a = slab.insert(1).unwrap();
    // Age the slot to the last generation.
    slab.slots[a.index() as usize] = Slot::Occupied {
        generation: u32::MAX,
        value: 1,
    };
    let last = ObjectId::from_parts(a.index(), u32::MAX);
    assert_eq!(slab.remove(last), Ok(1));
    assert_eq!(slab.get(last), Err(stale(last)));
    // The retired slot is never handed out again; a new slot is appended instead.
    let b = slab.insert(2).unwrap();
    assert_eq!(b.index(), a.index() + 1);
    assert_eq!(slab.get(last), Err(stale(last)));
}

#[test]
fn the_last_generation_is_issued_before_a_slot_is_retired() {
    let mut slab = TestSlab::new();
    let a = slab.insert(1).unwrap();
    slab.slots[a.index() as usize] = Slot::Occupied {
        generation: u32::MAX - 1,
        value: 1,
    };
    slab.remove(ObjectId::from_parts(a.index(), u32::MAX - 1))
        .unwrap();
    let b = slab.insert(2).unwrap();
    assert_eq!((b.index(), b.generation()), (a.index(), u32::MAX));
}

#[test]
fn iteration_is_in_slot_order_with_usable_ids() {
    let mut slab = TestSlab::new();
    let a = slab.insert(10).unwrap();
    let b = slab.insert(20).unwrap();
    let c = slab.insert(30).unwrap();
    slab.remove(b).unwrap();
    let rows: Vec<_> = slab.iter().map(|(id, v)| (id, *v)).collect();
    assert_eq!(rows, [(a, 10), (c, 30)]);
}

#[test]
fn get_mut_edits_the_row() {
    let mut slab = TestSlab::new();
    let a = slab.insert(1).unwrap();
    *slab.get_mut(a).unwrap() = 99;
    assert_eq!(slab.get(a), Ok(&99));
}

#[test]
fn different_id_types_work_on_the_same_storage() {
    let mut threads: Slab<ThreadId, &str> = Slab::new();
    let t = threads.insert("main").unwrap();
    assert_eq!(threads.get(t), Ok(&"main"));
}

#[test]
fn a_free_stack_pointing_at_an_occupied_slot_is_reported_not_trusted() {
    let mut slab = TestSlab::new();
    let a = slab.insert(1).unwrap();
    slab.free.push(a.index());
    assert_eq!(slab.insert(2), Err(SlabError::Corrupt));
    // A free stack pointing past the slots is reported too.
    let mut slab = TestSlab::new();
    slab.free.push(9);
    assert_eq!(slab.insert(2), Err(SlabError::Corrupt));
}

#[test]
fn serialise_then_deserialise_gives_an_equal_slab_that_behaves_the_same() {
    let mut slab = TestSlab::new();
    let ids: Vec<_> = (0..6).map(|v| slab.insert(v * 7).unwrap()).collect();
    slab.remove(ids[2]).unwrap();
    slab.remove(ids[4]).unwrap();
    slab.slots[0] = Slot::Retired;
    slab.live -= 1;

    let bytes = postcard::to_allocvec(&slab).unwrap();
    let mut back: TestSlab = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(back, slab);
    // The free stack order survived: the next ids are the same.
    assert_eq!(back.insert(100), slab.insert(100));
    assert_eq!(back.insert(101), slab.insert(101));
    assert_eq!(back, slab);
}

#[test]
fn the_wire_form_has_exactly_the_documented_fields() {
    // `RawSlab` mirrors the serialised layout field for field; a reordered or extra field in the
    // real type would make this differ.
    let mut slab = TestSlab::with_max_slots(77);
    let a = slab.insert(5).unwrap();
    slab.insert(6).unwrap();
    slab.remove(a).unwrap();
    let raw = RawSlab {
        slots: slab.slots.clone(),
        free: slab.free.clone(),
        max_slots: slab.max_slots,
        live: slab.live,
    };
    assert_eq!(
        postcard::to_allocvec(&slab).unwrap(),
        postcard::to_allocvec(&raw).unwrap()
    );
}

/// A named way to damage a [`RawSlab`].
type Damage = (&'static str, Box<dyn Fn(&mut RawSlab)>);

fn valid_raw() -> RawSlab {
    RawSlab {
        slots: vec![
            Slot::Occupied {
                generation: 2,
                value: 1,
            },
            Slot::Vacant { generation: 5 },
            Slot::Retired,
        ],
        free: vec![1],
        max_slots: 10,
        live: 1,
    }
}

#[test]
fn deserialising_accepts_a_consistent_layout() {
    let slab = deserialise(&valid_raw()).unwrap();
    assert_eq!(slab.len(), 1);
    assert_eq!(slab.get(ObjectId::from_parts(0, 2)), Ok(&1));
}

#[test]
fn deserialising_rejects_damaged_layouts() {
    let damage: Vec<Damage> = vec![
        ("free entry out of range", Box::new(|r| r.free = vec![3])),
        (
            "free entry names an occupied slot",
            Box::new(|r| r.free = vec![0]),
        ),
        (
            "free entry names a retired slot",
            Box::new(|r| r.free = vec![2]),
        ),
        ("free entry repeated", Box::new(|r| r.free = vec![1, 1])),
        (
            "vacant slot missing from the stack",
            Box::new(|r| r.free = vec![]),
        ),
        ("live count too high", Box::new(|r| r.live = 2)),
        ("live count too low", Box::new(|r| r.live = 0)),
        ("more slots than the limit", Box::new(|r| r.max_slots = 2)),
        (
            "zero generation (occupied)",
            Box::new(|r| {
                r.slots[0] = Slot::Occupied {
                    generation: 0,
                    value: 1,
                }
            }),
        ),
        (
            "zero generation (vacant)",
            Box::new(|r| r.slots[1] = Slot::Vacant { generation: 0 }),
        ),
    ];
    for (name, apply) in damage {
        let mut raw = valid_raw();
        apply(&mut raw);
        assert!(deserialise(&raw).is_err(), "{name} was accepted");
    }
}

impl<T> SlabData<T> {
    fn new(slots: Vec<Slot<T>>, free: Vec<u32>, max_slots: u32, live: u32) -> Self {
        Self {
            slots,
            free,
            max_slots,
            live,
        }
    }
}

#[test]
fn each_layout_error_is_distinguishable() {
    let try_raw = |slots: Vec<Slot<i64>>, free, max, live| {
        TestSlab::try_from(SlabData::new(slots, free, max, live)).err()
    };
    assert_eq!(
        try_raw(vec![Slot::Retired, Slot::Retired], vec![], 1, 0),
        Some(SlabLayoutError::TooManySlots {
            slots: 2,
            max_slots: 1
        })
    );
    assert_eq!(
        try_raw(vec![Slot::Vacant { generation: 0 }], vec![0], 4, 0),
        Some(SlabLayoutError::ZeroGeneration { index: 0 })
    );
    assert_eq!(
        try_raw(
            vec![Slot::Occupied {
                generation: 1,
                value: 0
            }],
            vec![0],
            4,
            1
        ),
        Some(SlabLayoutError::FreeStack(FreeStackError::NotVacant {
            index: 0
        }))
    );
    assert_eq!(
        try_raw(vec![Slot::Vacant { generation: 1 }], vec![0, 0], 4, 0),
        Some(SlabLayoutError::FreeStack(FreeStackError::Duplicate {
            index: 0
        }))
    );
    assert_eq!(
        try_raw(vec![Slot::Vacant { generation: 1 }], vec![], 4, 0),
        Some(SlabLayoutError::Counts)
    );
}

#[derive(Debug, Clone)]
enum Op {
    Insert(i64),
    /// Index into every id ever issued (live or stale), modulo its length.
    Get(usize),
    Remove(usize),
}

fn ops() -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(
        prop_oneof![
            3 => any::<i64>().prop_map(Op::Insert),
            2 => any::<usize>().prop_map(Op::Get),
            3 => any::<usize>().prop_map(Op::Remove),
        ],
        0..200,
    )
}

proptest! {
    #![proptest_config(config())]

    /// Random insert, get and remove sequences agree with a map. Every id ever issued is checked
    /// after every step: it resolves to exactly what the model holds under it, which for a freed
    /// id (including one whose slot was reused) is nothing.
    #[test]
    fn agrees_with_a_map_model(ops in ops()) {
        let mut slab = TestSlab::new();
        let mut model: BTreeMap<ObjectId, i64> = BTreeMap::new();
        let mut issued: Vec<ObjectId> = Vec::new();
        for op in ops {
            match op {
                Op::Insert(value) => {
                    let id = slab.insert(value).unwrap();
                    prop_assert!(!issued.contains(&id), "id {id:?} was issued twice");
                    issued.push(id);
                    model.insert(id, value);
                }
                Op::Get(pick) if !issued.is_empty() => {
                    let id = issued[pick % issued.len()];
                    prop_assert_eq!(slab.get(id).ok(), model.get(&id));
                }
                Op::Remove(pick) if !issued.is_empty() => {
                    let id = issued[pick % issued.len()];
                    let expected = model.remove(&id).ok_or(stale(id));
                    prop_assert_eq!(slab.remove(id), expected);
                }
                Op::Get(_) | Op::Remove(_) => {}
            }
            prop_assert_eq!(slab.len(), model.len());
            for &id in &issued {
                let expected = model.get(&id).ok_or(stale(id));
                prop_assert_eq!(slab.get(id), expected);
            }
            let live: Vec<_> = slab.iter().map(|(id, v)| (id, *v)).collect();
            let mut expected: Vec<_> = model.iter().map(|(id, v)| (*id, *v)).collect();
            expected.sort_by_key(|(id, _)| id.index());
            prop_assert_eq!(live, expected);
        }
        // Whatever state the sequence reached survives serialisation and behaves identically.
        let bytes = postcard::to_allocvec(&slab).unwrap();
        let mut back: TestSlab = postcard::from_bytes(&bytes).unwrap();
        prop_assert_eq!(&back, &slab);
        prop_assert_eq!(back.insert(1), slab.insert(1));
    }
}
