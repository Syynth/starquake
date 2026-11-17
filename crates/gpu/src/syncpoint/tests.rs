//! Tests of the syncpoint counters, their comparisons, allocation, watching, extension and
//! snapshots.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use super::*;
use crate::interface::{ChannelId, Ticket};

/// A [`Completions`] that keeps what was pushed, in order.
#[derive(Default)]
struct Recorder(Mutex<Vec<GpuCompletion>>);

impl Recorder {
    fn taken(&self) -> Vec<GpuCompletion> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl Completions for Recorder {
    fn push(&self, completion: GpuCompletion) {
        self.0.lock().unwrap().push(completion);
    }
}

fn id(value: u8) -> SyncpointId {
    SyncpointId::new(value).unwrap()
}

fn index(id: SyncpointId) -> usize {
    usize::from(id.get())
}

fn fresh() -> (Syncpoints, Arc<Recorder>) {
    let recorder = Arc::new(Recorder::default());
    let completions: Arc<dyn Completions> = Arc::<Recorder>::clone(&recorder);
    (Syncpoints::new(completions), recorder)
}

/// The ids [`Syncpoints::allocate`] hands out from a fresh `Syncpoints`, in order.
fn allocatable() -> Vec<SyncpointId> {
    all_ids()
        .filter(|id| id.get() >= FIRST_ALLOCATABLE && !RESERVED_SYNCPOINTS.contains(&id.get()))
        .collect()
}

/// The last id.
fn last_id() -> SyncpointId {
    id(u8::try_from(SYNCPOINT_COUNT - 1).unwrap())
}

/// Sets counter `id` to `value` directly, to reach values near the wrap without 2^32 increments.
fn set_counter(syncpoints: &Syncpoints, id: SyncpointId, value: u32) {
    syncpoints.counters[index(id)].store(value, Ordering::SeqCst);
}

fn set_max(syncpoints: &Syncpoints, id: SyncpointId, value: u32) {
    syncpoints.maxima[index(id)].store(value, Ordering::SeqCst);
}

// ---- comparisons -------------------------------------------------------------------------

#[test]
fn the_wrapping_comparison_at_the_wrap_edge_on_both_sides() {
    let mode = CompareMode::Wrapping32;
    // At and one below the threshold.
    assert!(wait_satisfied(100, 100, mode));
    assert!(!wait_satisfied(99, 100, mode));
    assert!(wait_satisfied(101, 100, mode));
    // Across the counter's wrap: one past u32::MAX is 0, which has reached u32::MAX.
    assert!(wait_satisfied(0, u64::from(u32::MAX), mode));
    assert!(!wait_satisfied(u32::MAX, 0, mode));
    assert!(wait_satisfied(u32::MAX, u64::from(u32::MAX), mode));
    assert!(!wait_satisfied(u32::MAX - 1, u64::from(u32::MAX), mode));
    // The edge of the half range: a difference just below the sign bit counts as reached, the
    // sign bit itself as not yet.
    let threshold = 7_u32;
    let last_reached = threshold.wrapping_add(SIGN_BIT - 1);
    let first_not = threshold.wrapping_add(SIGN_BIT);
    assert!(wait_satisfied(last_reached, u64::from(threshold), mode));
    assert!(!wait_satisfied(first_not, u64::from(threshold), mode));
    // Only the threshold's low half counts.
    assert!(wait_satisfied(5, WRAP | 5, mode));
    assert!(!wait_satisfied(4, WRAP | 5, mode));
}

#[test]
fn the_wide_comparison_at_the_edge_on_both_sides() {
    let mode = CompareMode::Wide64;
    assert!(wait_satisfied(100, 100, mode));
    assert!(!wait_satisfied(99, 100, mode));
    assert!(wait_satisfied(u32::MAX, u64::from(u32::MAX), mode));
    // No wrap: 0 is below u32::MAX, unlike the wrapping comparison.
    assert!(!wait_satisfied(0, u64::from(u32::MAX), mode));
    assert!(wait_satisfied(0, 0, mode));
    // A threshold above every 32-bit value is never reached, unlike the wrapping comparison of
    // its low half.
    assert!(!wait_satisfied(u32::MAX, WRAP, mode));
    assert!(wait_satisfied(0, WRAP, CompareMode::Wrapping32));
    // Half a range ahead is still reached: the wide comparison has no window.
    assert!(wait_satisfied(SIGN_BIT, 0, mode));
    assert!(!wait_satisfied(SIGN_BIT, 0, CompareMode::Wrapping32));
}

#[test]
fn reached_compares_the_counter_wrapping() {
    let (syncpoints, _) = fresh();
    let sp = id(40);
    set_counter(&syncpoints, sp, u32::MAX - 1);
    let at_max = Fence {
        id: sp,
        value: u32::MAX,
    };
    assert!(!syncpoints.reached(at_max));
    assert_eq!(syncpoints.increment(sp), u32::MAX);
    assert!(syncpoints.reached(at_max));
    // Past the wrap the counter reads 0 and the fence at u32::MAX is still reached; a fence at
    // 1 is not yet.
    assert_eq!(syncpoints.increment(sp), 0);
    assert!(syncpoints.reached(at_max));
    assert!(syncpoints.reached(Fence { id: sp, value: 0 }));
    assert!(!syncpoints.reached(Fence { id: sp, value: 1 }));
    // Another counter is not consulted.
    assert!(!syncpoints.reached(Fence {
        id: id(41),
        value: 1
    }));
}

#[test]
fn a_fence_beyond_the_maximum_is_not_reached() {
    // Divergence from Eden's `HasSyncpointExpired`, which would call this fence expired
    // because its threshold lies beyond the maximum.
    let (syncpoints, _) = fresh();
    let sp = syncpoints.allocate().unwrap();
    assert_eq!(syncpoints.increment_max(sp, 2).unwrap(), 2);
    syncpoints.increment(sp);
    syncpoints.increment(sp);
    let beyond = Fence { id: sp, value: 3 };
    assert!(!syncpoints.reached(beyond));
    assert!(syncpoints.reached(syncpoints.fence(sp).unwrap()));
}

// ---- extension ---------------------------------------------------------------------------

#[test]
fn extension_across_wraps() {
    // From unallocated, the value itself.
    assert_eq!(extend_counter(UNALLOCATED, 5), 5);
    assert_eq!(extend_counter(UNALLOCATED, u32::MAX), u64::from(u32::MAX));
    // Forward without a wrap, and equal: the high half stays.
    assert_eq!(extend_counter(5, 7), 7);
    assert_eq!(extend_counter(WRAP | 3, 3), WRAP | 3);
    assert_eq!(extend_counter(WRAP | 3, 4), WRAP | 4);
    // One below the low half counts a wrap; the edge on both sides.
    assert_eq!(extend_counter(WRAP | 3, 2), (2 * WRAP) | 2);
    assert_eq!(extend_counter(u64::from(u32::MAX), 0), WRAP);
    assert_eq!(
        extend_counter(u64::from(u32::MAX), u32::MAX),
        u64::from(u32::MAX)
    );
    // A second wrap.
    assert_eq!(extend_counter((2 * WRAP) | 0x10, 1), (3 * WRAP) | 1);
    // The high half wraps past u64::MAX, as the C++ addition does.
    let top = !LOW_HALF | 5;
    assert_eq!(extend_counter(top, 4), 4);
}

#[test]
fn counter_extension_observes_per_id() {
    let mut extension = CounterExtension::default();
    let a = id(3);
    let b = last_id();
    // Unallocated reads as all ones, and leaves the remembered value as it was.
    assert_eq!(extension.observe(a, None), UNALLOCATED);
    assert_eq!(extension.observe(a, Some(10)), 10);
    assert_eq!(extension.observe(a, None), UNALLOCATED);
    // 10 was remembered, so 3 is a wrap.
    assert_eq!(extension.observe(a, Some(3)), WRAP | 3);
    assert_eq!(extension.observe(a, Some(4)), WRAP | 4);
    // Another id has its own history.
    assert_eq!(extension.observe(b, Some(3)), 3);
    assert_eq!(extension.observe(a, Some(4)), WRAP | 4);
}

#[test]
fn counter_extension_layout_is_pinned() {
    let mut extension = CounterExtension::default();
    extension.observe(id(0), Some(5));
    extension.observe(id(1), Some(7));
    extension.observe(id(1), Some(3));
    let bytes = postcard::to_allocvec(&extension).unwrap();

    // `postcard`: the count as a varint, then each value as a varint.
    let mut golden = vec![0xC0, 0x01, 0x05, 0x83, 0x80, 0x80, 0x80, 0x10];
    let unallocated = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
    for _ in 2..COUNT {
        golden.extend(unallocated);
    }
    assert_eq!(bytes, golden);
    let back: CounterExtension = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(back, extension);
}

#[test]
fn counter_extension_refuses_the_wrong_count() {
    let short = vec![UNALLOCATED; COUNT - 1];
    let long = vec![UNALLOCATED; COUNT + 1];
    for values in [short, long] {
        let found = values.len();
        let error = CounterExtension::try_from(values.clone()).unwrap_err();
        assert_eq!(error, ExtensionLengthError { found });
        assert_eq!(
            error.to_string(),
            format!("a counter extension holds {found} values, not {SYNCPOINT_COUNT}")
        );
        let bytes = postcard::to_allocvec(&values).unwrap();
        assert!(postcard::from_bytes::<CounterExtension>(&bytes).is_err());
    }
    let exact = vec![UNALLOCATED; COUNT];
    let bytes = postcard::to_allocvec(&exact).unwrap();
    assert_eq!(
        postcard::from_bytes::<CounterExtension>(&bytes).unwrap(),
        CounterExtension::default()
    );
}

// ---- allocation --------------------------------------------------------------------------

#[test]
fn the_reserved_ids_are_eden_s_and_distinct() {
    let mut reserved = RESERVED_SYNCPOINTS.to_vec();
    reserved.sort_unstable();
    reserved.dedup();
    assert_eq!(reserved.len(), RESERVED_SYNCPOINTS.len());
    for value in RESERVED_SYNCPOINTS {
        assert!(value >= FIRST_ALLOCATABLE);
        assert!(SyncpointId::new(value).is_some());
    }
}

#[test]
fn a_new_set_reserves_exactly_the_reserved_ids() {
    let (syncpoints, _) = fresh();
    for sp in all_ids() {
        assert_eq!(
            syncpoints.is_allocated(sp),
            RESERVED_SYNCPOINTS.contains(&sp.get()),
            "id {}",
            sp.get()
        );
        assert_eq!(syncpoints.read(sp), 0);
        assert!(!syncpoints.is_watched(sp));
    }
}

#[test]
fn allocation_returns_the_lowest_free_id() {
    let (syncpoints, _) = fresh();
    let expected = allocatable();
    assert_eq!(expected[0].get(), FIRST_ALLOCATABLE);
    let got: Vec<_> = expected
        .iter()
        .map(|_| syncpoints.allocate().unwrap())
        .collect();
    assert_eq!(got, expected);
    // A freed id is the next one handed out, even though higher ones were freed too.
    let low = expected[5];
    let high = expected[100];
    syncpoints.free(high).unwrap();
    syncpoints.free(low).unwrap();
    assert_eq!(syncpoints.allocate(), Ok(low));
    assert_eq!(syncpoints.allocate(), Ok(high));
}

#[test]
fn allocation_exhausts_at_exactly_the_allocatable_count() {
    let (syncpoints, _) = fresh();
    let count = usize::try_from(SYNCPOINT_COUNT).unwrap()
        - usize::from(FIRST_ALLOCATABLE)
        - RESERVED_SYNCPOINTS.len();
    assert_eq!(allocatable().len(), count);
    for _ in 0..count {
        let sp = syncpoints.allocate().unwrap();
        assert!(sp.get() >= FIRST_ALLOCATABLE);
        assert!(!RESERVED_SYNCPOINTS.contains(&sp.get()));
    }
    let error = syncpoints.allocate().unwrap_err();
    assert_eq!(error, SyncpointError::Exhausted);
    assert_eq!(error.to_string(), "every syncpoint is allocated");
    // Every id but those below the first allocatable one is now taken.
    for sp in all_ids() {
        assert_eq!(syncpoints.is_allocated(sp), sp.get() >= FIRST_ALLOCATABLE);
    }
    // One free makes exactly one more allocation possible.
    let last = last_id();
    syncpoints.free(last).unwrap();
    assert_eq!(syncpoints.allocate(), Ok(last));
    assert_eq!(syncpoints.allocate(), Err(SyncpointError::Exhausted));
}

#[test]
fn a_reserved_id_can_be_freed_and_allocated() {
    let (syncpoints, _) = fresh();
    let reserved = id(RESERVED_SYNCPOINTS[0]);
    syncpoints.free(reserved).unwrap();
    assert!(!syncpoints.is_allocated(reserved));
    let before: Vec<_> = allocatable()
        .into_iter()
        .filter(|sp| sp.get() < reserved.get())
        .collect();
    for sp in &before {
        assert_eq!(syncpoints.allocate().as_ref(), Ok(sp));
    }
    assert_eq!(syncpoints.allocate(), Ok(reserved));
}

#[test]
fn operations_on_an_unallocated_id_are_errors() {
    let (syncpoints, _) = fresh();
    let sp = id(70);
    let error = SyncpointError::NotAllocated(sp);
    assert_eq!(error.to_string(), "syncpoint 70 is not allocated");
    assert_eq!(syncpoints.free(sp), Err(error));
    assert_eq!(syncpoints.increment_max(sp, 1), Err(error));
    assert_eq!(syncpoints.fence(sp), Err(error));
    // Nothing changed.
    assert_eq!(syncpoints.maxima[index(sp)].load(Ordering::SeqCst), 0);
    assert!(!syncpoints.is_allocated(sp));
    // A double free is an error too, and leaves the neighbours allocated.
    let allocated: Vec<_> = (0..3).map(|_| syncpoints.allocate().unwrap()).collect();
    syncpoints.free(allocated[1]).unwrap();
    assert_eq!(
        syncpoints.free(allocated[1]),
        Err(SyncpointError::NotAllocated(allocated[1]))
    );
    assert!(syncpoints.is_allocated(allocated[0]));
    assert!(syncpoints.is_allocated(allocated[2]));
}

#[test]
fn freeing_leaves_the_neighbours_allocated() {
    let (syncpoints, _) = fresh();
    while syncpoints.allocate().is_ok() {}
    // Ids at word boundaries, so a wrong word or bit shows.
    for value in [63, 64, 127, 128] {
        let sp = id(value);
        syncpoints.free(sp).unwrap();
        for other in all_ids() {
            let expected = other.get() >= FIRST_ALLOCATABLE && other != sp;
            assert_eq!(
                syncpoints.is_allocated(other),
                expected,
                "id {}",
                other.get()
            );
        }
        assert_eq!(syncpoints.allocate(), Ok(sp));
    }
}

#[test]
fn a_reallocated_id_keeps_its_values() {
    let (syncpoints, _) = fresh();
    let sp = syncpoints.allocate().unwrap();
    syncpoints.increment(sp);
    syncpoints.increment_max(sp, 4).unwrap();
    syncpoints.free(sp).unwrap();
    assert_eq!(syncpoints.allocate(), Ok(sp));
    assert_eq!(syncpoints.read(sp), 1);
    assert_eq!(syncpoints.fence(sp), Ok(Fence { id: sp, value: 4 }));
}

#[test]
fn concurrent_allocations_hand_out_each_id_once() {
    let (syncpoints, _) = fresh();
    let mut handed_out: Vec<SyncpointId> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    while let Ok(sp) = syncpoints.allocate() {
                        mine.push(sp);
                    }
                    mine
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect()
    });
    handed_out.sort_unstable();
    assert_eq!(handed_out, allocatable());
}

// ---- counters, maxima and watching -------------------------------------------------------

#[test]
fn reads_and_increments_work_on_any_id() {
    let (syncpoints, _) = fresh();
    let unallocated = id(100);
    assert_eq!(syncpoints.increment(unallocated), 1);
    assert_eq!(syncpoints.increment(unallocated), 2);
    assert_eq!(syncpoints.read(unallocated), 2);
    assert_eq!(syncpoints.read_allocated(unallocated), None);
    let reserved = id(RESERVED_SYNCPOINTS[1]);
    syncpoints.increment(reserved);
    assert_eq!(syncpoints.read_allocated(reserved), Some(1));
    // Neighbours are untouched.
    assert_eq!(syncpoints.read(id(99)), 0);
    assert_eq!(syncpoints.read(id(101)), 0);
}

#[test]
fn an_increment_wraps() {
    let (syncpoints, _) = fresh();
    let sp = last_id();
    set_counter(&syncpoints, sp, u32::MAX);
    assert_eq!(syncpoints.increment(sp), 0);
    assert_eq!(syncpoints.read(sp), 0);
}

#[test]
fn increment_pushes_only_when_watched() {
    let (syncpoints, recorder) = fresh();
    let sp = id(64);
    // Unwatched: nothing pushed.
    assert_eq!(syncpoints.increment(sp), 1);
    assert_eq!(recorder.taken(), []);
    // Watched: the new value is pushed.
    syncpoints.watch(sp);
    assert!(syncpoints.is_watched(sp));
    assert_eq!(syncpoints.increment(sp), 2);
    assert_eq!(
        recorder.taken(),
        [GpuCompletion::SyncpointChanged { id: sp, value: 2 }]
    );
    // The neighbours on both sides of the word boundary are not watched.
    for neighbour in [id(63), id(65)] {
        assert!(!syncpoints.is_watched(neighbour));
        syncpoints.increment(neighbour);
    }
    assert_eq!(recorder.taken(), []);
    // Unwatched again: nothing pushed, and watching a neighbour does not change that.
    syncpoints.watch(id(63));
    syncpoints.unwatch(sp);
    assert!(!syncpoints.is_watched(sp));
    assert!(syncpoints.is_watched(id(63)));
    assert_eq!(syncpoints.increment(sp), 3);
    assert_eq!(recorder.taken(), []);
}

#[test]
fn watched_increments_push_in_order() {
    let (syncpoints, recorder) = fresh();
    let a = id(2);
    let b = id(190);
    syncpoints.watch(a);
    syncpoints.watch(b);
    syncpoints.increment(a);
    syncpoints.increment(b);
    syncpoints.increment(a);
    assert_eq!(
        recorder.taken(),
        [
            GpuCompletion::SyncpointChanged { id: a, value: 1 },
            GpuCompletion::SyncpointChanged { id: b, value: 1 },
            GpuCompletion::SyncpointChanged { id: a, value: 2 },
        ]
    );
}

#[test]
fn an_increment_after_a_watch_is_either_seen_or_pushed() {
    // gsv's side: watch, then read. The incrementer's: add, then check the bit. Every value
    // above the one gsv read must have been pushed, or a waiter could miss it.
    const INCREMENTS: u32 = 20_000;
    for round in 0..8 {
        let (syncpoints, recorder) = fresh();
        let sp = id(77);
        let seen = std::thread::scope(|scope| {
            scope.spawn(|| {
                for _ in 0..INCREMENTS {
                    syncpoints.increment(sp);
                }
            });
            // Let the incrementer get going for a varying while before watching.
            for _ in 0..round * 500 {
                std::hint::spin_loop();
            }
            syncpoints.watch(sp);
            syncpoints.read(sp)
        });
        let pushed: std::collections::BTreeSet<u32> = recorder
            .taken()
            .into_iter()
            .map(|completion| match completion {
                GpuCompletion::SyncpointChanged { id, value } if id == sp => Some(value),
                _ => None,
            })
            .collect::<Option<_>>()
            .expect("only changes of the watched id are pushed");
        let missing: Vec<u32> = (seen + 1..=INCREMENTS)
            .filter(|value| !pushed.contains(value))
            .collect();
        assert!(missing.is_empty(), "round {round}: values {missing:?} lost");
    }
}

#[test]
fn maxima_and_fences() {
    let (syncpoints, _) = fresh();
    let a = syncpoints.allocate().unwrap();
    let b = syncpoints.allocate().unwrap();
    assert_eq!(syncpoints.fence(a), Ok(Fence { id: a, value: 0 }));
    assert_eq!(syncpoints.increment_max(a, 2), Ok(2));
    assert_eq!(syncpoints.increment_max(a, 3), Ok(5));
    assert_eq!(syncpoints.fence(a), Ok(Fence { id: a, value: 5 }));
    // The other id's maximum is its own.
    assert_eq!(syncpoints.fence(b), Ok(Fence { id: b, value: 0 }));
    // Raising the maximum does not move the counter.
    assert_eq!(syncpoints.read(a), 0);
    // The maximum wraps.
    set_max(&syncpoints, b, u32::MAX - 1);
    assert_eq!(syncpoints.increment_max(b, 3), Ok(1));
    assert_eq!(syncpoints.fence(b), Ok(Fence { id: b, value: 1 }));
}

#[test]
fn every_completion_reaches_the_sink_in_push_order() {
    let recorder = Recorder::default();
    let sink: &dyn Completions = &recorder;
    let sent = [
        GpuCompletion::GpuQueueRoom {
            channel: ChannelId(9),
        },
        GpuCompletion::SyncpointChanged {
            id: id(5),
            value: 6,
        },
        GpuCompletion::GpuOrdered { ticket: Ticket(7) },
    ];
    for completion in sent {
        sink.push(completion);
    }
    assert_eq!(recorder.taken(), sent);
}

// ---- snapshots ---------------------------------------------------------------------------

/// A `Syncpoints` in which every snapshotted field has a distinct non-zero value: the reserved
/// ids plus 1 and 2 (allocated) and 70 and 191 (set directly) allocated; 2, 64 and 131 watched;
/// counter 1 at 3 and 191 at `0x1234_5678`; maximum 1 at 4 and 2 at `0x80`.
fn scene() -> (Syncpoints, Arc<Recorder>) {
    let (syncpoints, recorder) = fresh();
    assert_eq!(syncpoints.allocate(), Ok(id(1)));
    assert_eq!(syncpoints.allocate(), Ok(id(2)));
    for value in [70, 191] {
        let (word, mask) = bit(id(value));
        syncpoints.allocated[word].fetch_or(mask, Ordering::SeqCst);
    }
    for value in [2, 64, 131] {
        syncpoints.watch(id(value));
    }
    set_counter(&syncpoints, id(1), 3);
    set_counter(&syncpoints, last_id(), 0x1234_5678);
    set_max(&syncpoints, id(1), 4);
    set_max(&syncpoints, id(2), 0x80);
    (syncpoints, recorder)
}

/// [`scene`]'s snapshot, byte for byte (`postcard`: integers are varints, arrays have no
/// length, a `Vec` starts with its length).
#[rustfmt::skip]
fn golden() -> Vec<u8> {
    let mut bytes = vec![
        // Header: magic, version.
        b'S', b'Q', b'S', b'P', 1,
        // Allocated: word 0 holds 1, 2, 12, 26, 27, 54 and 55 (0x00C0_0000_0C00_1006); word 1
        // holds 70 (bit 6); word 2 holds 191 (bit 63).
        0x86, 0xA0, 0x80, 0xE0, 0x80, 0x80, 0x80, 0x60,
        0x40,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01,
        // Watched: 2 (bit 2 of word 0), 64 (bit 0 of word 1), 131 (bit 3 of word 2).
        0x04, 0x01, 0x08,
        // Counters: 192 of them; 1 is 3, 191 is 0x1234_5678.
        0xC0, 0x01, 0x00, 0x03,
    ];
    bytes.extend([0; COUNT - 3]);
    bytes.extend([0xF8, 0xAC, 0xD1, 0x91, 0x01]);
    // Maxima: 192; 1 is 4, 2 is 0x80.
    bytes.extend([0xC0, 0x01, 0x00, 0x04, 0x80, 0x01]);
    bytes.extend([0; COUNT - 3]);
    bytes
}

#[test]
fn the_snapshot_layout_is_pinned() {
    let (syncpoints, _) = scene();
    let bytes = syncpoints.save().unwrap();
    assert_eq!(bytes, golden());
    assert_eq!(bytes[..4], SNAPSHOT_MAGIC);
    assert_eq!(u32::from(bytes[4]), SNAPSHOT_VERSION);
}

#[test]
fn a_restored_set_behaves_like_the_saved_one() {
    let (saved, _) = scene();
    let bytes = saved.save().unwrap();
    let recorder = Arc::new(Recorder::default());
    let restored = {
        let sink: Arc<dyn Completions> = Arc::<Recorder>::clone(&recorder);
        Syncpoints::restore(&bytes, sink).unwrap()
    };
    assert_eq!(restored.save().unwrap(), bytes);

    // Allocation continues after the restored ids: 3 is the lowest free one, and 70 is taken.
    assert_eq!(restored.allocate(), Ok(id(3)));
    assert!(restored.is_allocated(id(70)));
    assert!(!restored.is_allocated(id(69)));
    // A watched id pushes to the new sink; an unwatched one does not.
    assert_eq!(restored.increment(id(2)), 1);
    assert_eq!(restored.increment(id(1)), 4);
    assert_eq!(
        recorder.taken(),
        [GpuCompletion::SyncpointChanged {
            id: id(2),
            value: 1
        }]
    );
    // Counters and maxima carried over.
    assert_eq!(restored.read(last_id()), 0x1234_5678);
    assert_eq!(restored.increment_max(id(2), 1), Ok(0x81));
    assert_eq!(
        restored.fence(id(1)),
        Ok(Fence {
            id: id(1),
            value: 4
        })
    );
    assert!(restored.reached(Fence {
        id: id(1),
        value: 4
    }));
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

fn good_state() -> State {
    State {
        allocated: [0; BITMAP_WORDS],
        watched: [0; BITMAP_WORDS],
        counters: vec![0; COUNT],
        maxima: vec![0; COUNT],
    }
}

fn restore_error(bytes: &[u8]) -> SnapshotError {
    let recorder: Arc<dyn Completions> = Arc::new(Recorder::default());
    Syncpoints::restore(bytes, recorder)
        .err()
        .expect("the snapshot was accepted")
}

#[test]
fn restore_refuses_a_bad_header() {
    let good = raw(&good_state());
    let mut magic = good.clone();
    magic[0] = b'X';
    assert_eq!(restore_error(&magic), SnapshotError::BadMagic);
    assert_eq!(
        SnapshotError::BadMagic.to_string(),
        "not a syncpoint snapshot (bad magic)"
    );

    let mut version = good.clone();
    version[4] = 2;
    let error = restore_error(&version);
    assert_eq!(error, SnapshotError::UnsupportedVersion { found: 2 });
    assert_eq!(
        error.to_string(),
        format!(
            "syncpoint snapshot layout version 2 is not supported (this build reads {SNAPSHOT_VERSION})"
        )
    );

    // Both at once: the magic is reported.
    let mut both = magic;
    both[4] = 2;
    assert_eq!(restore_error(&both), SnapshotError::BadMagic);
    // And a bad version is reported before a body that would not decode.
    version.truncate(6);
    assert_eq!(
        restore_error(&version),
        SnapshotError::UnsupportedVersion { found: 2 }
    );
}

#[test]
fn restore_refuses_a_truncated_or_padded_body() {
    let good = raw(&good_state());
    let truncated = &good[..good.len() - 1];
    let error = restore_error(truncated);
    assert!(matches!(error, SnapshotError::Decode(_)), "{error:?}");
    assert!(
        error
            .to_string()
            .starts_with("could not decode the syncpoint snapshot: "),
        "{error}"
    );
    assert!(matches!(
        restore_error(&good[..3]),
        SnapshotError::Decode(_)
    ));

    let mut padded = good;
    padded.extend([0, 0]);
    let error = restore_error(&padded);
    assert_eq!(error, SnapshotError::TrailingBytes { count: 2 });
    assert_eq!(
        error.to_string(),
        "2 unexpected bytes after the syncpoint snapshot"
    );
}

#[test]
fn restore_refuses_the_wrong_number_of_counters_or_maxima() {
    for found in [COUNT - 1, COUNT + 1] {
        let mut state = good_state();
        state.counters = vec![0; found];
        let error = restore_error(&raw(&state));
        assert_eq!(error, SnapshotError::CounterCount { found });
        assert_eq!(
            error.to_string(),
            format!("the syncpoint snapshot holds {found} counter values, not {SYNCPOINT_COUNT}")
        );

        let mut state = good_state();
        state.maxima = vec![0; found];
        let error = restore_error(&raw(&state));
        assert_eq!(error, SnapshotError::MaximumCount { found });
        assert_eq!(
            error.to_string(),
            format!("the syncpoint snapshot holds {found} maxima, not {SYNCPOINT_COUNT}")
        );
    }
}

#[test]
fn restore_refuses_an_id_that_is_never_allocated() {
    let mut state = good_state();
    let (word, mask) = bit(id(0));
    state.allocated[word] |= mask;
    let error = restore_error(&raw(&state));
    assert_eq!(error, SnapshotError::NeverAllocated { id: 0 });
    assert_eq!(
        error.to_string(),
        "the syncpoint snapshot marks syncpoint 0 allocated, which is never allocated"
    );
    // The first allocatable id is fine, and so is a watched id 0.
    let mut state = good_state();
    let (word, mask) = bit(id(FIRST_ALLOCATABLE));
    state.allocated[word] |= mask;
    state.watched[0] |= 1;
    assert!(Syncpoints::restore(&raw(&state), Arc::new(Recorder::default())).is_ok());
}

#[test]
fn restore_reports_faults_in_the_documented_order() {
    let (word, mask) = bit(id(0));
    // Counters, maxima and the allocation all wrong: the counters are reported.
    let mut state = good_state();
    state.counters.pop();
    state.maxima.pop();
    state.allocated[word] |= mask;
    assert_eq!(
        restore_error(&raw(&state)),
        SnapshotError::CounterCount { found: COUNT - 1 }
    );
    // Maxima and the allocation wrong: the maxima.
    state.counters.push(0);
    assert_eq!(
        restore_error(&raw(&state)),
        SnapshotError::MaximumCount { found: COUNT - 1 }
    );
    // Trailing bytes after a body that fails its checks: the trailing bytes.
    let mut padded = raw(&state);
    padded.push(0);
    assert_eq!(
        restore_error(&padded),
        SnapshotError::TrailingBytes { count: 1 }
    );
}

#[test]
fn save_runs_the_restore_checks() {
    let (syncpoints, _) = fresh();
    let (word, mask) = bit(id(0));
    syncpoints.allocated[word].fetch_or(mask, Ordering::SeqCst);
    assert_eq!(
        syncpoints.save(),
        Err(SnapshotError::NeverAllocated { id: 0 })
    );
}
