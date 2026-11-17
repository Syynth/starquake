//! Tests of gsv's shared state: the watched-id fast path, the drain and its lock order, and
//! snapshots.

use std::sync::TryLockError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use starquake_gpu::interface::{ChannelId, Completions, Ticket};
use starquake_gpu::syncpoint::CompareMode;
use starquake_kernel::id::ObjectId;
use starquake_kernel::ipc::EventHook;
use starquake_kernel::result::KernelError;
use starquake_kernel::slab::SlabId;

use super::*;
use crate::completion::testing::{locks, released, token};
use crate::syncpt::waiters::{MODE_WIDE_64, MODE_WRAPPING_32, WaitError};

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

/// The slot event `n`: not generation 0 or 1, so an id built with another generation differs.
fn event(n: u32) -> ObjectId {
    ObjectId::from_parts(n, 5)
}

/// A wait on syncpoint `sp` for `threshold` by `mode`.
fn wait(sp: u8, threshold: u64, mode: CompareMode) -> Wait {
    Wait::new(id(sp), threshold, mode).unwrap()
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

/// A [`TimerHost`] that keeps what was recorded, the tokens it was asked to unlock, each of
/// which it unlocks, and the events it was asked to signal, each of which it signals.
#[derive(Default)]
struct Host(Vec<EventKind>, Vec<DeviceShareToken>, Vec<ObjectId>);

impl TimerHost for Host {
    fn record(&mut self, kind: EventKind) {
        self.0.push(kind);
    }

    fn unlock_device_share(&mut self, token: DeviceShareToken) -> Result<(), DeviceShareError> {
        self.1.push(token);
        Ok(())
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        self.2.push(event);
        Ok(())
    }
}

/// One call a [`Journal`] host received.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Call {
    Record(EventKind),
    Unlock(DeviceShareToken),
    Signal(ObjectId),
}

/// A [`TimerHost`] that keeps every call in one list, in order, answers each unlock with the
/// next of its answers (success once they run out), and refuses to signal the events in
/// `refused`.
#[derive(Default)]
struct Journal {
    calls: Vec<Call>,
    answers: std::collections::VecDeque<Result<(), DeviceShareError>>,
    refused: Vec<ObjectId>,
}

impl TimerHost for Journal {
    fn record(&mut self, kind: EventKind) {
        self.calls.push(Call::Record(kind));
    }

    fn unlock_device_share(&mut self, token: DeviceShareToken) -> Result<(), DeviceShareError> {
        self.calls.push(Call::Unlock(token));
        self.answers.pop_front().unwrap_or(Ok(()))
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        self.calls.push(Call::Signal(event));
        if self.refused.contains(&event) {
            Err(KernelError::InvalidHandle)
        } else {
            Ok(())
        }
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

/// A [`Completions`] that keeps every syncpoint value pushed to it, in push order and
/// uncoalesced.
#[derive(Default)]
struct Pushed(Mutex<Vec<u32>>);

impl Completions for Pushed {
    fn push(&self, completion: GpuCompletion) {
        if let GpuCompletion::SyncpointChanged { value, .. } = completion {
            self.0.lock().unwrap().push(value);
        }
    }
}

#[test]
fn watching_misses_no_increment_that_races_it() {
    // An incrementer runs while a waiter watches. Every value past the one `watch` returns must
    // have been pushed (module docs, "The watched-id fast path"): reading the counter before
    // marking the id would lose the increments that land between the two steps. The pushes go
    // to a recorder rather than a queue, so none is folded away. Fixed trials; each
    // incrementer stops at a flag or a cap, and every wait is bounded.
    const TRIALS: usize = 200;
    const CAP: u32 = 1 << 20;
    let sp = id(33);
    for trial in 0..TRIALS {
        let pushed = Arc::new(Pushed::default());
        let completions: Arc<dyn Completions> = Arc::<Pushed>::clone(&pushed);
        let syncpoints = Arc::new(Syncpoints::new(completions));
        let shared = Shared::new(
            Arc::clone(&syncpoints),
            Arc::new(CompletionQueue::new()),
            GSV,
        );
        let stop = Arc::new(AtomicBool::new(false));
        let (done, finished) = mpsc::channel();
        {
            let syncpoints = Arc::clone(&syncpoints);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut count = 0;
                while !stop.load(Ordering::SeqCst) && count < CAP {
                    syncpoints.increment(sp);
                    count += 1;
                }
                done.send(()).unwrap();
            });
        }
        let deadline = Instant::now() + PATIENCE;
        while syncpoints.read(sp) == 0 {
            assert!(Instant::now() < deadline, "the incrementer did not start");
            std::hint::spin_loop();
        }
        let seen = shared.lock().watch(sp).unwrap();
        stop.store(true, Ordering::SeqCst);
        finished
            .recv_timeout(PATIENCE)
            .expect("the incrementer did not stop");
        let last = syncpoints.read(sp);
        let pushed = pushed.0.lock().unwrap();
        assert!(
            pushed
                .iter()
                .copied()
                .filter(|&value| value > seen)
                .eq(seen + 1..=last),
            "trial {trial}: watch returned {seen} and the counter ended at {last}, but the values \
             pushed past {seen} start at {:?}",
            pushed.iter().find(|&&value| value > seen)
        );
    }
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
    shared.lock().state.watchers[usize::from(sp.get())] = u32::MAX - 1;
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

impl LockChecker<'_> {
    fn check(&mut self) {
        assert!(
            matches!(self.shared.state.try_lock(), Err(TryLockError::WouldBlock)),
            "the kernel was called without gsv's shared lock"
        );
        self.calls += 1;
    }
}

impl TimerHost for LockChecker<'_> {
    fn record(&mut self, _kind: EventKind) {
        self.check();
    }

    fn unlock_device_share(&mut self, _token: DeviceShareToken) -> Result<(), DeviceShareError> {
        self.check();
        Ok(())
    }

    fn signal_event(&mut self, _event: ObjectId) -> SvcResult<()> {
        self.check();
        Ok(())
    }
}

#[test]
fn the_drain_calls_the_kernel_with_the_shared_lock_held() {
    let shared = fresh();
    for ticket in 1..=3 {
        shared.queue().push(ordered(ticket));
        shared.queue().push(released(ticket));
    }
    // A change that fires a wait: one record and one signal.
    shared
        .lock()
        .arm(event(1), wait(9, 1, CompareMode::Wrapping32))
        .unwrap();
    shared.syncpoints().increment(id(9));
    let mut checker = LockChecker {
        shared: &shared,
        calls: 0,
    };
    assert_eq!(shared.drain(&mut checker).len(), 7);
    assert_eq!(checker.calls, 8);
    // And released after.
    assert!(shared.state.try_lock().is_ok());
}

#[test]
fn the_drain_takes_the_queue_after_the_shared_lock() {
    // The lock order is gsv's lock, then the queue (module docs, "Lock order and the drain"):
    // what is pushed while the drain waits for gsv's lock is in its take. Taking the queue
    // first would leave `ordered(2)` for the next drain. The pass does not depend on the pause;
    // the pause gives the drain time to take the queue first if the order were swapped.
    let shared = Arc::new(fresh());
    shared.queue().push(ordered(1));
    let held = shared.lock();
    let (done, finished) = mpsc::channel();
    let drainer = Arc::clone(&shared);
    std::thread::spawn(move || {
        done.send(drainer.drain(&mut Host::default())).unwrap();
    });
    std::thread::sleep(Duration::from_millis(200));
    shared.queue().push(ordered(2));
    drop(held);
    let taken = finished
        .recv_timeout(PATIENCE)
        .expect("the drain did not finish after gsv's lock was released");
    assert_eq!(taken, [ordered(1), ordered(2)]);
    assert_eq!(shared.queue().take_all(), []);
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

    fn unlock_device_share(&mut self, _token: DeviceShareToken) -> Result<(), DeviceShareError> {
        self.pushed += 1;
        self.queue.push(ordered(100 + self.pushed));
        Ok(())
    }

    fn signal_event(&mut self, _event: ObjectId) -> SvcResult<()> {
        self.pushed += 1;
        self.queue.push(ordered(100 + self.pushed));
        Ok(())
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
    let queue =
        Arc::new(CompletionQueue::restore(&shared.queue().save().unwrap(), &locks()).unwrap());
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

/// A snapshot whose body is `counts` and `waits`, both already encoded.
fn snapshot_with(counts: &[u8], waits: &[u8]) -> Vec<u8> {
    let mut bytes = SNAPSHOT_MAGIC.to_vec();
    bytes.push(u8::try_from(SNAPSHOT_VERSION).unwrap());
    bytes.extend_from_slice(counts);
    bytes.extend_from_slice(waits);
    bytes
}

/// A snapshot whose body is `counts`, already encoded, and no armed waits.
fn snapshot(counts: &[u8]) -> Vec<u8> {
    snapshot_with(counts, &waits(&[]))
}

/// Encodes `saved` as the snapshot's list of waits.
fn waits(saved: &[SavedWait]) -> Vec<u8> {
    postcard::to_allocvec(saved).unwrap()
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
    // Two waits, one of each mode, armed in the reverse of their events' order, with distinct
    // ids, thresholds and events; each adds a watch to its id.
    let wrapping = wait(63, 0x1_0000_0002, CompareMode::Wrapping32);
    let wide = wait(64, u64::from(u32::MAX), CompareMode::Wide64);
    shared.lock().arm(event(0x90), wrapping).unwrap();
    shared.lock().arm(event(0x81), wide).unwrap();
    // The magic, version 2, then 192 counts (a two-byte varint) and each count.
    let mut expected = vec![b'S', b'Q', b'G', b'S', 0x02, 0xC0, 0x01];
    for index in 0..COUNT {
        match index {
            0 => expected.push(0x01),
            63 => expected.push(0x03),
            64 => expected.push(0x04),
            191 => expected.extend([0x80, 0x01]),
            _ => expected.push(0x00),
        }
    }
    // Then two waits, in ascending event order: the event (index and generation varints), the
    // id byte, the threshold varint, the mode byte.
    expected.push(0x02);
    expected.extend([0x81, 0x01, 0x05, 0x40, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x01]);
    expected.extend([0x90, 0x01, 0x05, 0x3F, 0x82, 0x80, 0x80, 0x80, 0x10, 0x00]);
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
    // One version either side of the one this build reads.
    for found in [SNAPSHOT_VERSION - 1, SNAPSHOT_VERSION + 1] {
        let mut bytes = fresh().save().unwrap();
        bytes[SNAPSHOT_MAGIC.len()] = u8::try_from(found).unwrap();
        let error = restore_over_fresh(&bytes).err().unwrap();
        assert_eq!(error, SnapshotError::UnsupportedVersion { found });
        assert_eq!(
            error.to_string(),
            format!(
                "gsv shared state snapshot layout version {found} is not supported (this build reads {SNAPSHOT_VERSION})"
            )
        );
    }
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

#[test]
fn the_drain_unlocks_each_release_through_the_kernel_in_queue_order() {
    // Three releases among recorded completions, two of one range (two locks): each is unlocked
    // with its token, at its place in the queue's order.
    let shared = fresh();
    shared.queue().push(ordered(1));
    shared.queue().push(released(4));
    shared.queue().push(room(2));
    shared.queue().push(released(4));
    shared.queue().push(released(5));
    shared.queue().push(ordered(3));
    let mut host = Journal::default();
    assert_eq!(
        shared.drain(&mut host),
        [
            ordered(1),
            released(4),
            room(2),
            released(4),
            released(5),
            ordered(3),
        ]
    );
    let record = |note: Note| Call::Record(note_event(note));
    assert_eq!(
        host.calls,
        [
            record(Note::GpuOrderedDone { ticket: Ticket(1) }),
            Call::Unlock(token(4)),
            record(Note::QueueRoom {
                channel: ChannelId(2)
            }),
            Call::Unlock(token(4)),
            Call::Unlock(token(5)),
            record(Note::GpuOrderedDone { ticket: Ticket(3) }),
        ]
    );
}

#[test]
fn a_refused_release_is_logged_and_the_drain_goes_on() {
    // The kernel refuses the first two releases (one as ended, one as not locked); the drain
    // still asks for the third and records what follows each.
    let shared = fresh();
    for n in 1..=3 {
        shared.queue().push(released(n));
        shared.queue().push(ordered(n));
    }
    let mut host = Journal {
        answers: [
            Err(DeviceShareError::ProcessEnded),
            Err(DeviceShareError::NotLocked {
                address: token(2).address(),
                size: token(2).size(),
            }),
        ]
        .into(),
        ..Journal::default()
    };
    assert_eq!(shared.drain(&mut host).len(), 6);
    let record = |ticket| {
        Call::Record(note_event(Note::GpuOrderedDone {
            ticket: Ticket(ticket),
        }))
    };
    assert_eq!(
        host.calls,
        [
            Call::Unlock(token(1)),
            record(1),
            Call::Unlock(token(2)),
            record(2),
            Call::Unlock(token(3)),
            record(3),
        ]
    );
}

#[test]
fn a_release_restored_from_a_snapshot_is_unlocked_with_the_kernel_s_token() {
    let shared = fresh();
    shared.queue().push(released(6));
    let restored = round_trip(&shared).unwrap();
    let mut host = Host::default();
    assert_eq!(restored.drain(&mut host), [released(6)]);
    assert_eq!(host.1, [token(6)]);
    assert_eq!(host.0, []);
}

// ----- The waiter table -------------------------------------------------------------------------

#[test]
fn an_armed_wait_holds_one_watch_of_its_syncpoint() {
    let shared = fresh();
    let (a, b) = (id(30), id(31));
    // Arming takes a watch, and the counter it returns is read after it.
    shared.syncpoints().increment(a);
    assert_eq!(
        shared
            .lock()
            .arm(event(1), wait(30, 5, CompareMode::Wrapping32)),
        Ok(1)
    );
    assert_eq!(shared.lock().watchers(a), 1);
    assert!(shared.syncpoints().is_watched(a));
    // Re-arming the same event on the same id replaces the wait and keeps one watch.
    let replacement = wait(30, 6, CompareMode::Wrapping32);
    shared.lock().arm(event(1), replacement).unwrap();
    assert_eq!(shared.lock().watchers(a), 1);
    assert_eq!(shared.lock().waiters().get(event(1)), Some(replacement));
    // Re-arming it on another id moves the watch.
    let moved = wait(31, 7, CompareMode::Wide64);
    shared.lock().arm(event(1), moved).unwrap();
    assert_eq!(shared.lock().watchers(a), 0);
    assert!(!shared.syncpoints().is_watched(a));
    assert_eq!(shared.lock().watchers(b), 1);
    assert_eq!(shared.lock().waiters().get(event(1)), Some(moved));
    // A second event adds a second watch.
    shared.lock().arm(event(2), moved).unwrap();
    assert_eq!(shared.lock().watchers(b), 2);
    assert_eq!(shared.lock().waiters().on(b), 2);
    assert_eq!(shared.lock().waiters().len(), 2);
    // Disarming gives one back and returns the wait.
    assert_eq!(shared.lock().disarm(event(1)), Some(moved));
    assert_eq!(shared.lock().watchers(b), 1);
    assert_eq!(shared.lock().waiters().get(event(1)), None);
    // Disarming an event with no wait changes nothing.
    assert_eq!(shared.lock().disarm(event(1)), None);
    assert_eq!(shared.lock().watchers(b), 1);
    // Firing gives the last one back.
    shared.queue().push(changed(31, 7));
    let mut host = Host::default();
    shared.drain(&mut host);
    assert_eq!(host.2, [event(2)]);
    assert_eq!(shared.lock().watchers(b), 0);
    assert!(shared.lock().waiters().is_empty());
}

#[test]
fn an_arm_that_cannot_watch_changes_nothing() {
    let shared = fresh();
    let sp = id(12);
    let earlier = wait(13, 2, CompareMode::Wrapping32);
    shared.lock().arm(event(4), earlier).unwrap();
    shared.lock().watch(sp).unwrap();
    shared.lock().state.watchers[usize::from(sp.get())] = u32::MAX;
    let error = shared
        .lock()
        .arm(event(4), wait(12, 1, CompareMode::Wrapping32))
        .unwrap_err();
    assert_eq!(error, WatchError::TooManyWaiters(sp));
    // The earlier wait and its watch stand.
    assert_eq!(shared.lock().waiters().get(event(4)), Some(earlier));
    assert_eq!(shared.lock().watchers(id(13)), 1);
    assert_eq!(shared.lock().watchers(sp), u32::MAX);
}

/// The value [`arm_mixed`]'s change takes syncpoint 20 to.
const REACHED: u32 = 0x40;

/// Waits on syncpoint 20, armed out of event order: three that a change to [`REACHED`] reaches
/// (of both modes), one it does not, and one on syncpoint 21 that the same value would reach.
fn arm_mixed(shared: &Shared) {
    let mut state = shared.lock();
    let at = u64::from(REACHED);
    state
        .arm(event(9), wait(20, at, CompareMode::Wrapping32))
        .unwrap();
    state
        .arm(event(3), wait(20, at - 1, CompareMode::Wide64))
        .unwrap();
    state
        .arm(event(12), wait(20, at + 1, CompareMode::Wrapping32))
        .unwrap();
    state
        .arm(event(6), wait(20, 1, CompareMode::Wrapping32))
        .unwrap();
    state
        .arm(event(2), wait(21, 1, CompareMode::Wrapping32))
        .unwrap();
}

#[test]
fn a_change_fires_its_waits_after_its_note_in_event_order() {
    let shared = fresh();
    arm_mixed(&shared);
    shared.queue().push(changed(20, REACHED));
    let mut host = Journal::default();
    shared.drain(&mut host);
    assert_eq!(
        host.calls,
        [
            Call::Record(note_event(Note::SyncpointObserved {
                id: id(20),
                value: REACHED
            })),
            Call::Signal(event(3)),
            Call::Signal(event(6)),
            Call::Signal(event(9)),
        ]
    );
    // The wait the value does not reach, and the wait on another id, stay, with their watches.
    let state = shared.lock();
    assert_eq!(state.waiters().len(), 2);
    assert!(state.waiters().get(event(12)).is_some());
    assert!(state.waiters().get(event(2)).is_some());
    assert_eq!(state.watchers(id(20)), 1);
    assert_eq!(state.watchers(id(21)), 1);
}

#[test]
fn a_wait_fires_exactly_at_its_threshold_not_one_before() {
    // Both modes, with the threshold across a wrap for the wrapping one: a counter at 2 has
    // reached a threshold of 2 + 2^32 by the 32-bit comparison.
    for (mode, threshold) in [
        (CompareMode::Wrapping32, 0x1_0000_0002),
        (CompareMode::Wide64, 2),
    ] {
        let shared = fresh();
        let sp = id(44);
        shared
            .lock()
            .arm(event(8), wait(44, threshold, mode))
            .unwrap();
        // One before: the change is recorded, and nothing fires.
        shared.syncpoints().increment(sp);
        let mut host = Host::default();
        shared.drain(&mut host);
        assert_eq!(host.0.len(), 1, "{mode:?}");
        assert_eq!(host.2, [], "{mode:?}");
        assert!(shared.lock().waiters().get(event(8)).is_some());
        // At the threshold: it fires.
        shared.syncpoints().increment(sp);
        let mut host = Host::default();
        shared.drain(&mut host);
        assert_eq!(host.2, [event(8)], "{mode:?}");
        assert!(shared.lock().waiters().is_empty());
        // And it fires once: a later change finds no wait.
        shared.queue().push(changed(44, 3));
        let mut host = Host::default();
        shared.drain(&mut host);
        assert_eq!(host.2, [], "{mode:?}");
    }
}

#[test]
fn a_refused_signal_is_logged_and_the_drain_goes_on() {
    let shared = fresh();
    arm_mixed(&shared);
    shared.queue().push(changed(20, REACHED));
    shared.queue().push(ordered(5));
    let mut host = Journal {
        refused: vec![event(6)],
        ..Journal::default()
    };
    assert_eq!(shared.drain(&mut host).len(), 2);
    // Every reached wait was signalled, the refused one included, and the next completion was
    // acted on.
    assert_eq!(
        host.calls[1..],
        [
            Call::Signal(event(3)),
            Call::Signal(event(6)),
            Call::Signal(event(9)),
            Call::Record(note_event(Note::GpuOrderedDone { ticket: Ticket(5) })),
        ]
    );
    // The refused wait is gone, with its watch.
    assert_eq!(shared.lock().waiters().get(event(6)), None);
    assert_eq!(shared.lock().watchers(id(20)), 1);
}

#[test]
fn armed_waits_are_used_as_saved() {
    let shared = fresh();
    arm_mixed(&shared);
    let restored = round_trip(&shared).unwrap();
    assert_eq!(*restored.lock(), *shared.lock());
    // The restored watches came back with the waits: an increment of a watched id is pushed,
    // and fires its wait.
    restored.syncpoints().increment(id(21));
    restored.queue().push(changed(20, REACHED));
    let mut host = Host::default();
    restored.drain(&mut host);
    assert_eq!(host.2, [event(2), event(3), event(6), event(9)]);
    assert_eq!(restored.lock().watchers(id(20)), 1);
    assert_eq!(restored.lock().watchers(id(21)), 0);
}

/// A wait as a snapshot holds it.
fn saved(event_index: u32, sp: u8, threshold: u64, mode: u8) -> SavedWait {
    SavedWait {
        event: event(event_index),
        id: sp,
        threshold,
        mode,
    }
}

/// Restores counts with one waiter on each of `watched` over syncpoints that watch them, and
/// `entries` as the waits.
fn restore_waits(watched: &[u8], entries: &[SavedWait]) -> Result<Shared, SnapshotError> {
    let shared = fresh();
    for &sp in watched {
        shared.syncpoints().watch(id(sp));
    }
    let counts = counts(COUNT, |index| {
        u32::from(watched.iter().any(|&sp| usize::from(sp) == index))
    });
    Shared::restore(
        &snapshot_with(&counts, &waits(entries)),
        Arc::clone(shared.syncpoints()),
        Arc::clone(shared.queue()),
        GSV,
    )
}

#[test]
fn saved_waits_must_be_in_ascending_event_order() {
    let first = saved(4, 10, 1, MODE_WRAPPING_32);
    let second = saved(5, 11, 1, MODE_WRAPPING_32);
    assert!(restore_waits(&[10, 11], &[first, second]).is_ok());
    // Swapped, and one event twice.
    for entries in [[second, first], [first, SavedWait { id: 11, ..first }]] {
        let error = restore_waits(&[10, 11], &entries).err().unwrap();
        assert_eq!(
            error,
            SnapshotError::Wait(SavedWaitError::OutOfOrder { index: 1 })
        );
        assert_eq!(
            error.to_string(),
            "the gsv shared state snapshot's waits: saved wait 1 is not in ascending event order"
        );
    }
    // Events that differ only in their generation are two events, ordered by it.
    let older = SavedWait {
        event: ObjectId::from_parts(4, 4),
        ..second
    };
    assert!(restore_waits(&[10, 11], &[older, first]).is_ok());
    assert!(restore_waits(&[10, 11], &[first, older]).is_err());
}

#[test]
fn a_saved_wait_must_name_a_syncpoint() {
    let last = u8::try_from(SYNCPOINT_COUNT - 1).unwrap();
    assert!(restore_waits(&[last], &[saved(1, last, 1, MODE_WRAPPING_32)]).is_ok());
    let beyond = last + 1;
    let error = restore_waits(&[], &[saved(1, beyond, 1, MODE_WRAPPING_32)])
        .err()
        .unwrap();
    assert_eq!(
        error,
        SnapshotError::Wait(SavedWaitError::NoSuchSyncpoint {
            index: 0,
            id: beyond
        })
    );
    assert_eq!(
        error.to_string(),
        "the gsv shared state snapshot's waits: saved wait 0 names syncpoint 192, and there are 192"
    );
}

#[test]
fn a_saved_wait_must_have_a_mode_201_accepts() {
    assert!(restore_waits(&[3], &[saved(1, 3, 1, MODE_WIDE_64)]).is_ok());
    let mode = MODE_WIDE_64 + 1;
    let error = restore_waits(&[3], &[saved(1, 3, 1, mode)]).err().unwrap();
    assert_eq!(
        error,
        SnapshotError::Wait(SavedWaitError::UnknownMode { index: 0, mode })
    );
    assert_eq!(
        error.to_string(),
        "the gsv shared state snapshot's waits: saved wait 0 has mode 2, which no wait is armed with"
    );
}

#[test]
fn a_saved_wide_wait_must_be_one_a_counter_can_reach() {
    let edge = u64::from(u32::MAX);
    assert!(restore_waits(&[3], &[saved(1, 3, edge, MODE_WIDE_64)]).is_ok());
    // The same threshold is fine for a wrapping wait.
    assert!(restore_waits(&[3], &[saved(1, 3, edge + 1, MODE_WRAPPING_32)]).is_ok());
    let error = restore_waits(&[3], &[saved(1, 3, edge + 1, MODE_WIDE_64)])
        .err()
        .unwrap();
    assert_eq!(
        error,
        SnapshotError::Wait(SavedWaitError::Refused {
            index: 0,
            error: WaitError::WideThreshold {
                threshold: edge + 1
            }
        })
    );
    assert_eq!(
        error.to_string(),
        "the gsv shared state snapshot's waits: saved wait 0 could not have been armed: a 64-bit wait for 0x100000000 can never be reached by a 32-bit counter"
    );
}

#[test]
fn a_syncpoint_s_waits_may_not_outnumber_its_waiters() {
    // Exactly as many waits as waiters: accepted.
    let one = saved(1, 7, 1, MODE_WRAPPING_32);
    let two = saved(2, 7, 1, MODE_WRAPPING_32);
    assert!(restore_waits(&[7], &[one]).is_ok());
    // One more wait than waiters.
    let error = restore_waits(&[7], &[one, two]).err().unwrap();
    assert_eq!(
        error,
        SnapshotError::WaitsOutnumberWatchers {
            id: 7,
            waits: 2,
            waiters: 1
        }
    );
    assert_eq!(
        error.to_string(),
        "the gsv shared state snapshot arms 2 waits on syncpoint 7, which has 1 waiters"
    );
    // A wait on an id with no waiter at all, next to a watched neighbour.
    let error = restore_waits(&[7], &[saved(1, 8, 1, MODE_WRAPPING_32)])
        .err()
        .unwrap();
    assert_eq!(
        error,
        SnapshotError::WaitsOutnumberWatchers {
            id: 8,
            waits: 1,
            waiters: 0
        }
    );
}

#[test]
fn wait_checks_run_in_the_documented_order() {
    let bad_mode = MODE_WIDE_64 + 1;
    // A bad entry and a short count list: the count list is found first.
    let bad = saved(1, 3, 1, bad_mode);
    let short = snapshot_with(&counts(COUNT - 1, |_| 0), &waits(&[bad]));
    assert_eq!(
        restore_over_fresh(&short).err().unwrap(),
        SnapshotError::WaiterCounts { found: COUNT - 1 }
    );
    // A bad entry and a watched-bit mismatch: the entry is found first.
    let mismatch = snapshot_with(
        &counts(COUNT, |index| u32::from(index == 17)),
        &waits(&[bad]),
    );
    assert_eq!(
        restore_over_fresh(&mismatch).err().unwrap(),
        SnapshotError::Wait(SavedWaitError::UnknownMode {
            index: 0,
            mode: bad_mode
        })
    );
    // Two bad entries: the first in list order, though its fault is a later check than the
    // second's.
    let entries = [bad, saved(2, 200, 1, MODE_WRAPPING_32)];
    assert_eq!(
        restore_waits(&[3], &entries).err().unwrap(),
        SnapshotError::Wait(SavedWaitError::UnknownMode {
            index: 0,
            mode: bad_mode
        })
    );
    // Within one entry: the order, then the id, then the mode, then the threshold.
    let all = saved(1, 200, u64::MAX, bad_mode);
    let ahead = saved(2, 3, 1, MODE_WRAPPING_32);
    assert_eq!(
        restore_waits(&[3], &[ahead, all]).err().unwrap(),
        SnapshotError::Wait(SavedWaitError::OutOfOrder { index: 1 })
    );
    assert_eq!(
        restore_waits(&[], &[all]).err().unwrap(),
        SnapshotError::Wait(SavedWaitError::NoSuchSyncpoint { index: 0, id: 200 })
    );
    assert_eq!(
        restore_waits(&[3], &[SavedWait { id: 3, ..all }])
            .err()
            .unwrap(),
        SnapshotError::Wait(SavedWaitError::UnknownMode {
            index: 0,
            mode: bad_mode
        })
    );
    // On one id, the watched bit before the waits: two waits on an id whose count is 1 but
    // which the syncpoints do not watch.
    let unwatched = snapshot_with(
        &counts(COUNT, |index| u32::from(index == 17)),
        &waits(&[
            saved(1, 17, 1, MODE_WRAPPING_32),
            saved(2, 17, 1, MODE_WRAPPING_32),
        ]),
    );
    assert_eq!(
        restore_over_fresh(&unwatched).err().unwrap(),
        SnapshotError::WatchMismatch {
            id: 17,
            waiters: 1,
            watched: false
        }
    );
}

#[test]
fn saving_refuses_waits_that_outnumber_their_watchers() {
    // A watch given back behind the wait's back: saving refuses what restoring would.
    let shared = fresh();
    shared
        .lock()
        .arm(event(1), wait(9, 1, CompareMode::Wrapping32))
        .unwrap();
    shared.lock().state.watchers[9] = 0;
    shared.syncpoints().unwatch(id(9));
    assert_eq!(
        shared.save(),
        Err(SnapshotError::WaitsOutnumberWatchers {
            id: 9,
            waits: 1,
            waiters: 0
        })
    );
}
