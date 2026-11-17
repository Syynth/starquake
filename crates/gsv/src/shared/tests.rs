//! Tests of gsv's shared state: the watched-id fast path, the drain and its lock order, and
//! snapshots.

use std::sync::TryLockError;
use std::sync::mpsc;
use std::time::Duration;

use starquake_gpu::interface::{ChannelId, Completions, Ticket};
use starquake_kernel::id::ObjectId;
use starquake_kernel::ipc::EventHook;
use starquake_kernel::slab::SlabId;

use super::*;

/// The number the tests' gsv records its notes under: not the default, so a hard-coded one
/// fails.
const GSV: ServiceId = ServiceId(0x0657);

/// How long a threaded test waits before it calls a step stuck.
const PATIENCE: Duration = Duration::from_secs(30);

fn id(value: u8) -> SyncpointId {
    SyncpointId::new(value).unwrap()
}

fn last_id() -> SyncpointId {
    id(u8::try_from(SYNCPOINT_COUNT - 1).unwrap())
}

fn changed(sp: u8, value: u32) -> Completion {
    Completion::Gpu(GpuCompletion::SyncpointChanged { id: id(sp), value })
}

fn ordered(ticket: u64) -> Completion {
    Completion::Gpu(GpuCompletion::GpuOrdered {
        ticket: Ticket(ticket),
    })
}

fn room(channel: u32) -> Completion {
    Completion::Gpu(GpuCompletion::GpuQueueRoom {
        channel: ChannelId(channel),
    })
}

/// A queue, syncpoints that push to it, and the shared state over both: built in the order the
/// module docs give.
fn fresh() -> Shared {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    let syncpoints = Arc::new(Syncpoints::new(completions));
    Shared::new(syncpoints, queue, GSV)
}

/// A [`TimerHost`] that keeps what was recorded.
#[derive(Default)]
struct Host(Vec<EventKind>);

impl TimerHost for Host {
    fn record(&mut self, kind: EventKind) {
        self.0.push(kind);
    }
}

fn note_event(note: Note) -> EventKind {
    note.event(GSV)
}

#[test]
fn an_unwatched_increment_pushes_nothing() {
    // Graphics-server slice 3's offline test: an unwatched increment pushes nothing.
    let shared = fresh();
    let sp = id(5);
    shared.syncpoints().increment(sp);
    assert_eq!(shared.queue().take_all(), []);
    // Watched: pushed, with the value after the increment.
    assert_eq!(shared.lock().watch(sp), Ok(1));
    shared.syncpoints().increment(sp);
    assert_eq!(shared.queue().take_all(), [changed(5, 2)]);
    // A neighbour stays unwatched.
    shared.syncpoints().increment(id(6));
    shared.syncpoints().increment(id(4));
    assert_eq!(shared.queue().take_all(), []);
    // Unwatched again: nothing.
    shared.lock().unwatch(sp).unwrap();
    shared.syncpoints().increment(sp);
    assert_eq!(shared.queue().take_all(), []);
}

#[test]
fn watching_returns_the_counter_read_after_marking() {
    let shared = fresh();
    let sp = id(9);
    for _ in 0..3 {
        shared.syncpoints().increment(sp);
    }
    assert_eq!(shared.lock().watch(sp), Ok(3));
    // A second waiter reads the counter too.
    shared.syncpoints().increment(sp);
    assert_eq!(shared.lock().watch(sp), Ok(4));
}

#[test]
fn the_bit_follows_the_waiter_count() {
    let shared = fresh();
    let sp = id(7);
    let watched = || shared.syncpoints().is_watched(sp);
    assert!(!watched());
    // 0 -> 1: marked.
    shared.lock().watch(sp).unwrap();
    assert!(watched());
    assert_eq!(shared.lock().watchers(sp), 1);
    // 1 -> 2: still marked.
    shared.lock().watch(sp).unwrap();
    assert!(watched());
    assert_eq!(shared.lock().watchers(sp), 2);
    // 2 -> 1: still marked, and increments are still pushed.
    shared.lock().unwatch(sp).unwrap();
    assert!(watched());
    assert_eq!(shared.lock().watchers(sp), 1);
    shared.syncpoints().increment(sp);
    assert_eq!(shared.queue().take_all(), [changed(7, 1)]);
    // 1 -> 0: cleared.
    shared.lock().unwatch(sp).unwrap();
    assert!(!watched());
    assert_eq!(shared.lock().watchers(sp), 0);
}

#[test]
fn watching_one_id_leaves_its_neighbours() {
    // Ids at either side of a bitmap word's edge, and the last id.
    let shared = fresh();
    for (watch, neighbours) in [(63, [62, 64]), (64, [63, 65])] {
        shared.lock().watch(id(watch)).unwrap();
        for neighbour in neighbours {
            assert!(!shared.syncpoints().is_watched(id(neighbour)));
            assert_eq!(shared.lock().watchers(id(neighbour)), 0);
        }
        shared.lock().unwatch(id(watch)).unwrap();
    }
    let last = last_id();
    shared.lock().watch(last).unwrap();
    assert!(shared.syncpoints().is_watched(last));
    assert!(!shared.syncpoints().is_watched(id(last.get() - 1)));
}

#[test]
fn unwatching_an_id_with_no_waiters_is_an_error() {
    let shared = fresh();
    // Another id has a waiter, so a hard-coded id would be caught.
    shared.lock().watch(id(3)).unwrap();
    let error = shared.lock().unwatch(id(11)).unwrap_err();
    assert_eq!(error, WatchError::NotWatched(id(11)));
    assert_eq!(error.to_string(), "syncpoint 11 has no waiters");
    assert_eq!(shared.lock().watchers(id(11)), 0);
    assert_eq!(shared.lock().watchers(id(3)), 1);
    assert!(shared.syncpoints().is_watched(id(3)));
    assert!(!shared.syncpoints().is_watched(id(11)));
}

#[test]
fn the_waiter_count_stops_at_its_maximum() {
    let shared = fresh();
    let sp = id(12);
    shared.lock().watch(sp).unwrap();
    shared.lock().watchers[usize::from(sp.get())] = u32::MAX - 1;
    // One below the maximum: the last waiter fits.
    assert_eq!(shared.lock().watch(sp), Ok(0));
    assert_eq!(shared.lock().watchers(sp), u32::MAX);
    // At the maximum: refused, nothing changed.
    let error = shared.lock().watch(sp).unwrap_err();
    assert_eq!(error, WatchError::TooManyWaiters(sp));
    assert_eq!(error.to_string(), "syncpoint 12 has too many waiters");
    assert_eq!(shared.lock().watchers(sp), u32::MAX);
    assert!(shared.syncpoints().is_watched(sp));
}

#[test]
fn the_drain_records_each_completion_in_push_order() {
    let shared = fresh();
    let sp = id(21);
    shared.lock().watch(sp).unwrap();
    let gpu: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(shared.queue());
    let hook: Arc<dyn EventHook> = Arc::<CompletionQueue>::clone(shared.queue());
    let event = ObjectId::from_parts(4, 2);

    gpu.push(GpuCompletion::GpuOrdered {
        ticket: Ticket(0x0102_0304_0506),
    });
    shared.syncpoints().increment(sp);
    hook.signalled(event);
    gpu.push(GpuCompletion::GpuQueueRoom {
        channel: ChannelId(9),
    });

    let mut host = Host::default();
    assert_eq!(
        shared.drain(&mut host),
        [
            ordered(0x0102_0304_0506),
            changed(21, 1),
            Completion::GuestSignalled { event },
            room(9),
        ]
    );
    assert_eq!(
        host.0,
        [
            note_event(Note::GpuOrderedDone {
                ticket: Ticket(0x0102_0304_0506)
            }),
            note_event(Note::SyncpointObserved { id: sp, value: 1 }),
            note_event(Note::QueueRoom {
                channel: ChannelId(9)
            }),
        ]
    );
    // Drained: the next turn finds nothing and records nothing.
    let mut host = Host::default();
    assert_eq!(shared.drain(&mut host), []);
    assert_eq!(host.0, []);
}

/// A [`TimerHost`] that checks, on every call, that gsv's shared lock is held.
struct LockChecker<'a> {
    shared: &'a Shared,
    calls: usize,
}

impl TimerHost for LockChecker<'_> {
    fn record(&mut self, _kind: EventKind) {
        assert!(
            matches!(self.shared.state.try_lock(), Err(TryLockError::WouldBlock)),
            "the kernel was called without gsv's shared lock"
        );
        self.calls += 1;
    }
}

#[test]
fn the_drain_calls_the_kernel_with_the_shared_lock_held() {
    let shared = fresh();
    for ticket in 1..=3 {
        shared.queue().push(ordered(ticket));
    }
    let mut checker = LockChecker {
        shared: &shared,
        calls: 0,
    };
    assert_eq!(shared.drain(&mut checker).len(), 3);
    assert_eq!(checker.calls, 3);
    // And released after.
    assert!(shared.state.try_lock().is_ok());
}

/// A [`TimerHost`] that pushes onto the queue while the drain is under way, as the kernel's hook
/// may on another core.
struct Pusher<'a> {
    queue: &'a CompletionQueue,
    pushed: u64,
}

impl TimerHost for Pusher<'_> {
    fn record(&mut self, _kind: EventKind) {
        self.pushed += 1;
        self.queue.push(ordered(100 + self.pushed));
    }
}

#[test]
fn a_push_during_the_drain_waits_for_the_next_one() {
    let shared = Arc::new(fresh());
    shared.queue().push(ordered(1));
    shared.queue().push(ordered(2));
    let (done, finished) = mpsc::channel();
    let drainer = Arc::clone(&shared);
    std::thread::spawn(move || {
        let mut pusher = Pusher {
            queue: drainer.queue(),
            pushed: 0,
        };
        let first = drainer.drain(&mut pusher);
        done.send(first).unwrap();
    });
    let first = finished
        .recv_timeout(PATIENCE)
        .expect("the drain did not finish: a push during it waited on a lock it holds");
    assert_eq!(first, [ordered(1), ordered(2)]);
    assert_eq!(
        shared.drain(&mut Host::default()),
        [ordered(101), ordered(102)]
    );
}

#[test]
fn a_push_does_not_wait_for_the_shared_lock() {
    // gsv's lock is held (by a request handler, say) while the GPU, an incrementer and the
    // kernel's hook push: none of them waits for it.
    let shared = Arc::new(fresh());
    let sp = id(30);
    shared.lock().watch(sp).unwrap();
    let held = shared.lock();
    let (done, finished) = mpsc::channel();
    let pusher = Arc::clone(&shared);
    std::thread::spawn(move || {
        let gpu: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(pusher.queue());
        let hook: Arc<dyn EventHook> = Arc::<CompletionQueue>::clone(pusher.queue());
        gpu.push(GpuCompletion::GpuOrdered { ticket: Ticket(1) });
        pusher.syncpoints().increment(sp);
        hook.signalled(ObjectId::from_parts(1, 1));
        done.send(()).unwrap();
    });
    finished
        .recv_timeout(PATIENCE)
        .expect("a push waited for gsv's shared lock");
    drop(held);
    assert_eq!(shared.drain(&mut Host::default()).len(), 3);
}

/// Saves `syncpoints` and restores them over a new queue, and restores `shared`'s state over
/// them: a run brought back from a snapshot.
fn round_trip(shared: &Shared) -> Result<Shared, SnapshotError> {
    let queue = Arc::new(CompletionQueue::restore(&shared.queue().save().unwrap()).unwrap());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    let syncpoints =
        Arc::new(Syncpoints::restore(&shared.syncpoints().save().unwrap(), completions).unwrap());
    Shared::restore(&shared.save()?, syncpoints, queue, GSV)
}

/// Encodes `count` waiter counts, one per id from 0, each `waiters(id)`, as postcard varints.
fn counts(count: usize, waiters: impl Fn(usize) -> u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    push_varint(&mut bytes, u64::try_from(count).unwrap());
    for index in 0..count {
        push_varint(&mut bytes, u64::from(waiters(index)));
    }
    bytes
}

/// postcard's varint: seven bits a byte, low first, the top bit set on every byte but the last.
fn push_varint(bytes: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        bytes.push(u8::try_from(value & 0x7F).unwrap() | 0x80);
        value >>= 7;
    }
    bytes.push(u8::try_from(value).unwrap());
}

fn snapshot(body: &[u8]) -> Vec<u8> {
    let mut bytes = SNAPSHOT_MAGIC.to_vec();
    bytes.push(u8::try_from(SNAPSHOT_VERSION).unwrap());
    bytes.extend_from_slice(body);
    bytes
}

#[test]
fn the_snapshot_layout_is_pinned() {
    // Distinct non-zero counts at both ends, across a bitmap word's edge, and one that needs a
    // two-byte varint.
    let shared = fresh();
    let wanted: [(u8, u32); 4] = [(0, 1), (63, 2), (64, 3), (191, 0x80)];
    for (sp, waiters) in wanted {
        for _ in 0..waiters {
            shared.lock().watch(id(sp)).unwrap();
        }
    }
    // The magic, version 1, then 192 counts (a two-byte varint) and each count.
    let mut expected = vec![b'S', b'Q', b'G', b'S', 0x01, 0xC0, 0x01];
    for index in 0..COUNT {
        match index {
            0 => expected.push(0x01),
            63 => expected.push(0x02),
            64 => expected.push(0x03),
            191 => expected.extend([0x80, 0x01]),
            _ => expected.push(0x00),
        }
    }
    assert_eq!(shared.save().unwrap(), expected);
}

#[test]
fn a_restored_state_is_used_as_saved() {
    let shared = fresh();
    let sp = id(40);
    shared.lock().watch(sp).unwrap();
    shared.lock().watch(sp).unwrap();
    shared.lock().watch(id(41)).unwrap();
    shared.syncpoints().increment(sp);
    let restored = round_trip(&shared).unwrap();
    // The queued change came back, and drains with its note.
    let mut host = Host::default();
    assert_eq!(restored.drain(&mut host), [changed(40, 1)]);
    assert_eq!(
        host.0,
        [note_event(Note::SyncpointObserved { id: sp, value: 1 })]
    );
    // Two waiters came back: one unwatch leaves it watched, the second clears it.
    restored.lock().unwatch(sp).unwrap();
    restored.syncpoints().increment(sp);
    assert_eq!(restored.queue().take_all(), [changed(40, 2)]);
    restored.lock().unwatch(sp).unwrap();
    restored.syncpoints().increment(sp);
    assert_eq!(restored.queue().take_all(), []);
    assert_eq!(restored.lock().unwatch(sp), Err(WatchError::NotWatched(sp)));
    assert_eq!(restored.lock().watchers(id(41)), 1);
}

#[test]
fn a_snapshot_with_the_wrong_magic_is_refused() {
    let shared = fresh();
    let mut bytes = shared.save().unwrap();
    bytes[0] ^= 0xFF;
    let error = Shared::restore(
        &bytes,
        Arc::clone(shared.syncpoints()),
        Arc::clone(shared.queue()),
        GSV,
    )
    .err()
    .unwrap();
    assert_eq!(error, SnapshotError::BadMagic);
    assert_eq!(
        error.to_string(),
        "not a gsv shared state snapshot (bad magic)"
    );
}

fn restore_over_fresh(bytes: &[u8]) -> Result<Shared, SnapshotError> {
    let shared = fresh();
    Shared::restore(
        bytes,
        Arc::clone(shared.syncpoints()),
        Arc::clone(shared.queue()),
        GSV,
    )
}

#[test]
fn a_snapshot_of_another_version_is_refused() {
    let mut bytes = fresh().save().unwrap();
    bytes[SNAPSHOT_MAGIC.len()] = u8::try_from(SNAPSHOT_VERSION + 1).unwrap();
    let error = restore_over_fresh(&bytes).err().unwrap();
    assert_eq!(
        error,
        SnapshotError::UnsupportedVersion {
            found: SNAPSHOT_VERSION + 1
        }
    );
    assert_eq!(
        error.to_string(),
        format!(
            "gsv shared state snapshot layout version {} is not supported (this build reads {SNAPSHOT_VERSION})",
            SNAPSHOT_VERSION + 1
        )
    );
}

#[test]
fn a_cut_snapshot_is_refused() {
    let bytes = fresh().save().unwrap();
    for len in 0..bytes.len() {
        let error = restore_over_fresh(&bytes[..len]).err().unwrap();
        assert!(
            matches!(error, SnapshotError::Decode(_)),
            "cut to {len}: {error:?}"
        );
    }
}

#[test]
fn bytes_after_the_body_are_refused() {
    let mut bytes = fresh().save().unwrap();
    bytes.push(0x00);
    let error = restore_over_fresh(&bytes).err().unwrap();
    assert_eq!(error, SnapshotError::TrailingBytes { count: 1 });
    assert_eq!(
        error.to_string(),
        "1 unexpected bytes after the gsv shared state snapshot"
    );
}

#[test]
fn a_snapshot_must_hold_one_count_per_syncpoint() {
    for count in [COUNT - 1, COUNT + 1] {
        let error = restore_over_fresh(&snapshot(&counts(count, |_| 0)))
            .err()
            .unwrap();
        assert_eq!(error, SnapshotError::WaiterCounts { found: count });
        assert_eq!(
            error.to_string(),
            format!(
                "the gsv shared state snapshot holds {count} waiter counts, not {SYNCPOINT_COUNT}"
            )
        );
    }
    assert!(restore_over_fresh(&snapshot(&counts(COUNT, |_| 0))).is_ok());
}

#[test]
fn a_count_must_agree_with_the_watched_bit() {
    // A waiter the syncpoints do not have watched (fresh syncpoints watch nothing).
    let error = restore_over_fresh(&snapshot(&counts(COUNT, |index| u32::from(index == 17))))
        .err()
        .unwrap();
    assert_eq!(
        error,
        SnapshotError::WatchMismatch {
            id: 17,
            waiters: 1,
            watched: false
        }
    );
    assert_eq!(
        error.to_string(),
        "the gsv shared state snapshot gives syncpoint 17 1 waiters, but the syncpoints have it not watched"
    );
    // A watched id with no waiter.
    let shared = fresh();
    shared.syncpoints().watch(id(18));
    let error = Shared::restore(
        &snapshot(&counts(COUNT, |_| 0)),
        Arc::clone(shared.syncpoints()),
        Arc::clone(shared.queue()),
        GSV,
    )
    .err()
    .unwrap();
    assert_eq!(
        error,
        SnapshotError::WatchMismatch {
            id: 18,
            waiters: 0,
            watched: true
        }
    );
    assert_eq!(
        error.to_string(),
        "the gsv shared state snapshot gives syncpoint 18 0 waiters, but the syncpoints have it watched"
    );
}

#[test]
fn saving_runs_the_restore_checks() {
    // The watched bit set behind gsv's back: saving refuses what restoring would.
    let shared = fresh();
    shared.syncpoints().watch(last_id());
    assert_eq!(
        shared.save(),
        Err(SnapshotError::WatchMismatch {
            id: last_id().get(),
            waiters: 0,
            watched: true
        })
    );
}

#[test]
fn checks_run_in_the_documented_order() {
    // Trailing bytes and a short body: the trailing bytes are found first, since the body
    // decodes. A short body and a mismatch: the length is found first.
    let mut bytes = snapshot(&counts(COUNT - 1, |_| 1));
    bytes.push(0x00);
    assert_eq!(
        restore_over_fresh(&bytes).err().unwrap(),
        SnapshotError::TrailingBytes { count: 1 }
    );
    let bytes = snapshot(&counts(COUNT - 1, |_| 1));
    assert_eq!(
        restore_over_fresh(&bytes).err().unwrap(),
        SnapshotError::WaiterCounts { found: COUNT - 1 }
    );
    // Two mismatches: the lower id is reported.
    let bytes = snapshot(&counts(COUNT, |index| {
        u32::from(index == 30 || index == 20)
    }));
    assert_eq!(
        restore_over_fresh(&bytes).err().unwrap(),
        SnapshotError::WatchMismatch {
            id: 20,
            waiters: 1,
            watched: false
        }
    );
}
