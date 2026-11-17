//! Tests of the completion queue: push order, coalescing, waking the timer thread, the GPU's and
//! the kernel's ways in, and snapshots.

use std::sync::atomic::{AtomicU32, Ordering};

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_gpu::interface::SYNCPOINT_COUNT;
use starquake_kernel::slab::SlabId;
use starquake_services::framework::Kernel;
use starquake_services::framework::fake::FakeKernel;

use super::*;

fn id(value: u8) -> SyncpointId {
    SyncpointId::new(value).unwrap()
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

fn event(index: u32) -> ObjectId {
    ObjectId::from_parts(index, index + 1)
}

fn signalled(index: u32) -> Completion {
    Completion::GuestSignalled {
        event: event(index),
    }
}

/// A [`Wake`] that counts its calls.
#[derive(Default)]
struct Counter(AtomicU32);

impl Counter {
    fn count(&self) -> u32 {
        self.0.load(Ordering::SeqCst)
    }
}

impl Wake for Counter {
    fn completion_queued(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn counted() -> (CompletionQueue, Arc<Counter>) {
    let counter = Arc::new(Counter::default());
    let wake: Arc<dyn Wake> = Arc::<Counter>::clone(&counter);
    (CompletionQueue::new().with_wake(wake), counter)
}

/// One completion of every kind, each naming a different thing: nothing coalesces.
fn script() -> Vec<Completion> {
    vec![
        changed(5, 10),
        ordered(7),
        signalled(3),
        room(2),
        changed(6, 1),
        ordered(8),
        room(1),
        signalled(4),
        changed(4, 9),
    ]
}

#[test]
fn scripted_completions_drain_in_push_order() {
    // Graphics-server slice 3's offline test: scripted completions drain in push order.
    let queue = CompletionQueue::new();
    for completion in script() {
        queue.push(completion);
    }
    assert_eq!(queue.take_all(), script());
    assert_eq!(queue.take_all(), []);
}

#[test]
fn the_gpu_and_the_kernel_hook_push_through_their_traits() {
    let queue = Arc::new(CompletionQueue::new());
    let gpu: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    let hook: Arc<dyn EventHook> = Arc::<CompletionQueue>::clone(&queue);
    gpu.push(GpuCompletion::GpuOrdered { ticket: Ticket(3) });
    hook.signalled(event(9));
    gpu.push(GpuCompletion::GpuQueueRoom {
        channel: ChannelId(4),
    });
    assert_eq!(queue.take_all(), [ordered(3), signalled(9), room(4)]);
}

#[test]
fn a_guest_signal_on_a_hooked_event_reaches_the_queue() {
    // The kernel's side, as the services framework's fake plays it: the guest signals the
    // writable end of an event made with the queue as its hook.
    let queue = Arc::new(CompletionQueue::new());
    let mut kernel = FakeKernel::new(0).unwrap();
    let hook: Arc<dyn EventHook> = Arc::<CompletionQueue>::clone(&queue);
    let (writable, readable) = kernel.create_hooked_event(hook).unwrap();
    kernel.guest_signal(writable).unwrap();
    assert_eq!(
        queue.take_all(),
        [Completion::GuestSignalled { event: writable }]
    );
    assert_ne!(writable, readable);
}

#[test]
fn a_queued_syncpoint_change_keeps_its_place_and_the_value_further_ahead() {
    let queue = CompletionQueue::new();
    queue.push(changed(5, 10));
    queue.push(ordered(1));
    queue.push(changed(5, 12));
    // Pushed late by a second incrementer: behind the value held, so the held one stays.
    queue.push(changed(5, 11));
    queue.push(changed(6, 3));
    assert_eq!(
        queue.take_all(),
        [changed(5, 12), ordered(1), changed(6, 3)]
    );
    // Once taken, the next change is queued again.
    queue.push(changed(5, 13));
    assert_eq!(queue.take_all(), [changed(5, 13)]);
}

#[test]
fn a_queued_syncpoint_change_compares_wrapping_at_the_edge() {
    // The furthest a value can be ahead is `i32::MAX` past the held one; one more and the
    // wrapping difference is negative, so it counts as behind.
    let furthest = i32::MAX.cast_unsigned();
    let cases = [
        (u32::MAX, 0, 0),        // across the wrap: 0 is one ahead of u32::MAX
        (0, u32::MAX, 0),        // u32::MAX is one behind 0
        (0, furthest, furthest), // exactly as far ahead as the comparison allows
        (0, furthest + 1, 0),    // one further: behind
        (7, 7, 7),               // the same value
    ];
    for (held, pushed, kept) in cases {
        let queue = CompletionQueue::new();
        queue.push(changed(9, held));
        queue.push(changed(9, pushed));
        assert_eq!(
            queue.take_all(),
            [changed(9, kept)],
            "held {held:#x}, pushed {pushed:#x}"
        );
    }
}

#[test]
fn room_is_kept_once_per_channel() {
    let queue = CompletionQueue::new();
    queue.push(room(1));
    queue.push(room(2));
    queue.push(room(1));
    assert_eq!(queue.take_all(), [room(1), room(2)]);
    queue.push(room(1));
    assert_eq!(queue.take_all(), [room(1)]);
}

#[test]
fn a_guest_signal_is_kept_once_per_event() {
    let queue = CompletionQueue::new();
    queue.push(signalled(1));
    queue.push(signalled(2));
    queue.push(signalled(1));
    assert_eq!(queue.take_all(), [signalled(1), signalled(2)]);
    queue.push(signalled(1));
    assert_eq!(queue.take_all(), [signalled(1)]);
}

#[test]
fn ordered_reports_are_each_kept() {
    let queue = CompletionQueue::new();
    queue.push(ordered(7));
    queue.push(ordered(7));
    queue.push(ordered(6));
    assert_eq!(queue.take_all(), [ordered(7), ordered(7), ordered(6)]);
}

#[test]
fn the_timer_is_woken_once_per_empty_to_queued() {
    let (queue, counter) = counted();
    assert_eq!(counter.count(), 0);
    queue.push(changed(5, 1));
    assert_eq!(counter.count(), 1);
    // Onto a queue that holds something: already announced.
    queue.push(ordered(1));
    queue.push(changed(5, 2));
    assert_eq!(counter.count(), 1);
    assert_eq!(queue.take_all().len(), 2);
    // Empty again: the next push wakes.
    queue.push(changed(5, 3));
    assert_eq!(counter.count(), 2);
    // A take of an empty queue changes nothing.
    queue.take_all();
    assert_eq!(queue.take_all(), []);
    assert_eq!(counter.count(), 2);
}

#[test]
fn a_queue_with_no_wake_still_queues() {
    let queue = CompletionQueue::new();
    queue.push(room(3));
    assert_eq!(queue.take_all(), [room(3)]);
}

#[test]
fn concurrent_pushers_keep_their_own_order() {
    // Four threads push distinct tickets; each thread's tickets come out in the order it pushed
    // them, and none is lost. Bounded by a timeout so a deadlock fails instead of hanging.
    const THREADS: u64 = 4;
    const EACH: u64 = 2_000;
    let queue = Arc::new(CompletionQueue::new());
    let (done, finished) = std::sync::mpsc::channel();
    for thread in 0..THREADS {
        let queue = Arc::clone(&queue);
        let done = done.clone();
        std::thread::spawn(move || {
            for n in 0..EACH {
                queue.push(ordered(thread * EACH + n));
            }
            done.send(()).unwrap();
        });
    }
    for _ in 0..THREADS {
        finished
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("a pusher did not finish");
    }
    let taken = queue.take_all();
    assert_eq!(taken.len(), usize::try_from(THREADS * EACH).unwrap());
    for thread in 0..THREADS {
        let mine: Vec<_> = taken
            .iter()
            .filter(|completion| match completion {
                Completion::Gpu(GpuCompletion::GpuOrdered { ticket }) => ticket.0 / EACH == thread,
                _ => false,
            })
            .copied()
            .collect();
        let expected: Vec<_> = (0..EACH).map(|n| ordered(thread * EACH + n)).collect();
        assert_eq!(mine, expected);
    }
}

/// One step of a generated script.
#[derive(Clone, Copy, Debug)]
enum Step {
    Push(Completion),
    Take,
}

fn completion() -> impl Strategy<Value = Completion> {
    // Few distinct things of each kind, so coalescing happens often.
    prop_oneof![
        (0u8..4, any::<u32>()).prop_map(|(sp, value)| changed(sp, value)),
        (0u64..4).prop_map(ordered),
        (0u32..3).prop_map(room),
        (0u32..3).prop_map(signalled),
    ]
}

fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        8 => completion().prop_map(Step::Push),
        1 => Just(Step::Take),
    ]
}

/// The queue as a list scanned from the front: what the module docs say it keeps.
fn model_push(model: &mut Vec<Completion>, completion: Completion) {
    let same = |held: &Completion| match (held, completion) {
        (
            Completion::Gpu(GpuCompletion::SyncpointChanged { id: a, .. }),
            Completion::Gpu(GpuCompletion::SyncpointChanged { id: b, .. }),
        ) => *a == b,
        (
            Completion::Gpu(GpuCompletion::GpuQueueRoom { channel: a }),
            Completion::Gpu(GpuCompletion::GpuQueueRoom { channel: b }),
        ) => *a == b,
        (Completion::GuestSignalled { event: a }, Completion::GuestSignalled { event: b }) => {
            *a == b
        }
        _ => false,
    };
    match model.iter_mut().find(|held| same(held)) {
        Some(Completion::Gpu(GpuCompletion::SyncpointChanged { value: held, .. })) => {
            if let Completion::Gpu(GpuCompletion::SyncpointChanged { value, .. }) = completion
                && value.wrapping_sub(*held).cast_signed() >= 0
            {
                *held = value;
            }
        }
        Some(_) => {}
        None => model.push(completion),
    }
}

proptest! {
    #![proptest_config(Config {
        cases: 256,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x5351_4351_5545_5545),
        failure_persistence: None,
        ..Config::default()
    })]

    #[test]
    fn the_queue_matches_a_scanned_list(steps in prop::collection::vec(step(), 0..64)) {
        let queue = CompletionQueue::new();
        let mut model = Vec::new();
        for step in steps {
            match step {
                Step::Push(completion) => {
                    queue.push(completion);
                    model_push(&mut model, completion);
                }
                Step::Take => prop_assert_eq!(queue.take_all(), std::mem::take(&mut model)),
            }
        }
        prop_assert_eq!(queue.take_all(), model);
    }

    #[test]
    fn a_saved_queue_restores_to_the_same_entries(
        pushes in prop::collection::vec(completion(), 0..32),
    ) {
        let queue = CompletionQueue::new();
        for completion in &pushes {
            queue.push(*completion);
        }
        let restored = CompletionQueue::restore(&queue.save().unwrap()).unwrap();
        prop_assert_eq!(restored.take_all(), queue.take_all());
    }
}

/// The header bytes of a snapshot: the magic, then the version as a varint.
fn header() -> Vec<u8> {
    let mut bytes = SNAPSHOT_MAGIC.to_vec();
    bytes.push(u8::try_from(SNAPSHOT_VERSION).unwrap());
    bytes
}

/// A snapshot of `body` bytes after a good header.
fn snapshot(body: &[u8]) -> Vec<u8> {
    let mut bytes = header();
    bytes.extend_from_slice(body);
    bytes
}

#[test]
fn the_snapshot_layout_is_pinned() {
    // Every kind once, each field a distinct non-zero value. postcard: the entry count, then
    // each entry's variant index and its fields, `u8` as a byte and wider integers as varints;
    // an `ObjectId` is its index and then its generation.
    let queue = CompletionQueue::new();
    queue.push(changed(0x05, 0x0102));
    queue.push(ordered(0x0304));
    queue.push(room(0x06));
    queue.push(Completion::GuestSignalled {
        event: ObjectId::from_parts(0x07, 0x08),
    });
    let bytes = queue.save().unwrap();
    assert_eq!(
        bytes,
        [
            b'S', b'Q', b'C', b'Q', // magic
            0x01, // version 1
            0x04, // four entries
            0x00, 0x05, 0x82, 0x02, // SyncpointChanged: id 5, value 0x102
            0x01, 0x84, 0x06, // GpuOrdered: ticket 0x304
            0x02, 0x06, // GpuQueueRoom: channel 6
            0x03, 0x07, 0x08, // GuestSignalled: event index 7, generation 8
        ]
    );
    assert_eq!(&bytes[..SNAPSHOT_MAGIC.len()], SNAPSHOT_MAGIC);
}

#[test]
fn a_restored_queue_keeps_coalescing_and_order() {
    let queue = CompletionQueue::new();
    queue.push(changed(5, 10));
    queue.push(room(2));
    queue.push(signalled(1));
    queue.push(ordered(4));
    let (restored, counter) = {
        let counter = Arc::new(Counter::default());
        let wake: Arc<dyn Wake> = Arc::<Counter>::clone(&counter);
        let restored = CompletionQueue::restore(&queue.save().unwrap()).unwrap();
        (restored.with_wake(wake), counter)
    };
    // Used after the restore: repeats fold into the restored entries, a new thing queues
    // behind them, and a push onto the restored (non-empty) queue does not wake.
    restored.push(changed(5, 11));
    restored.push(room(2));
    restored.push(signalled(1));
    restored.push(room(3));
    assert_eq!(counter.count(), 0);
    assert_eq!(
        restored.take_all(),
        [changed(5, 11), room(2), signalled(1), ordered(4), room(3)]
    );
    restored.push(ordered(9));
    assert_eq!(counter.count(), 1);
}

#[test]
fn an_empty_queue_saves_and_restores() {
    let bytes = CompletionQueue::new().save().unwrap();
    assert_eq!(bytes, snapshot(&[0x00]));
    assert_eq!(CompletionQueue::restore(&bytes).unwrap().take_all(), []);
}

#[test]
fn a_snapshot_with_the_wrong_magic_is_refused() {
    let mut bytes = CompletionQueue::new().save().unwrap();
    bytes[0] ^= 0xFF;
    let error = CompletionQueue::restore(&bytes).err().unwrap();
    assert_eq!(error, SnapshotError::BadMagic);
    assert_eq!(
        error.to_string(),
        "not a completion queue snapshot (bad magic)"
    );
}

#[test]
fn a_snapshot_of_another_version_is_refused() {
    let mut bytes = CompletionQueue::new().save().unwrap();
    bytes[SNAPSHOT_MAGIC.len()] = u8::try_from(SNAPSHOT_VERSION + 1).unwrap();
    let error = CompletionQueue::restore(&bytes).err().unwrap();
    assert_eq!(
        error,
        SnapshotError::UnsupportedVersion {
            found: SNAPSHOT_VERSION + 1
        }
    );
    assert_eq!(
        error.to_string(),
        format!(
            "completion queue snapshot layout version {} is not supported (this build reads {SNAPSHOT_VERSION})",
            SNAPSHOT_VERSION + 1
        )
    );
}

#[test]
fn a_cut_snapshot_is_refused() {
    let queue = CompletionQueue::new();
    queue.push(ordered(0x0304));
    let bytes = queue.save().unwrap();
    for len in 0..bytes.len() {
        let error = CompletionQueue::restore(&bytes[..len]).err().unwrap();
        assert!(
            matches!(error, SnapshotError::Decode(_)),
            "cut to {len}: {error:?}"
        );
    }
}

#[test]
fn bytes_after_the_body_are_refused() {
    let mut bytes = CompletionQueue::new().save().unwrap();
    bytes.extend([0xAA, 0xBB]);
    let error = CompletionQueue::restore(&bytes).err().unwrap();
    assert_eq!(error, SnapshotError::TrailingBytes { count: 2 });
    assert_eq!(
        error.to_string(),
        "2 unexpected bytes after the completion queue snapshot"
    );
}

#[test]
fn an_entry_naming_no_syncpoint_is_refused_at_the_count() {
    let first_out = u8::try_from(SYNCPOINT_COUNT).unwrap();
    let last = first_out - 1;
    // The last syncpoint is accepted.
    let good = snapshot(&[0x02, 0x01, 0x05, 0x00, last, 0x01]);
    assert_eq!(
        CompletionQueue::restore(&good).unwrap().take_all(),
        [ordered(5), changed(last, 1)]
    );
    // One past it is refused, at its index.
    let bad = snapshot(&[0x02, 0x01, 0x05, 0x00, first_out, 0x01]);
    let error = CompletionQueue::restore(&bad).err().unwrap();
    assert_eq!(
        error,
        SnapshotError::NoSuchSyncpoint {
            index: 1,
            id: first_out
        }
    );
    assert_eq!(
        error.to_string(),
        format!(
            "entry 1 of the completion queue snapshot names syncpoint {first_out}, which does not exist"
        )
    );
}

#[test]
fn a_repeated_coalesced_entry_is_refused() {
    // Each coalesced kind twice, at index 2 after an unrelated entry.
    let repeats: [&[u8]; 3] = [
        &[0x03, 0x00, 0x05, 0x01, 0x01, 0x09, 0x00, 0x05, 0x02], // syncpoint 5 twice
        &[0x03, 0x02, 0x04, 0x01, 0x09, 0x02, 0x04],             // channel 4 twice
        &[0x03, 0x03, 0x04, 0x05, 0x01, 0x09, 0x03, 0x04, 0x05], // event (4, 5) twice
    ];
    for body in repeats {
        let error = CompletionQueue::restore(&snapshot(body)).err().unwrap();
        assert_eq!(error, SnapshotError::Repeated { index: 2 }, "{body:x?}");
        assert_eq!(
            error.to_string(),
            "entry 2 of the completion queue snapshot repeats an earlier entry"
        );
    }
    // A repeated ticket is not coalesced, so it is accepted.
    let tickets = snapshot(&[0x02, 0x01, 0x09, 0x01, 0x09]);
    assert_eq!(
        CompletionQueue::restore(&tickets).unwrap().take_all(),
        [ordered(9), ordered(9)]
    );
}

#[test]
fn entries_are_checked_in_order() {
    // A repeat at index 1 and a missing syncpoint at index 2: the earlier is reported.
    let first_out = u8::try_from(SYNCPOINT_COUNT).unwrap();
    let body = [0x03, 0x02, 0x04, 0x02, 0x04, 0x00, first_out, 0x01];
    assert_eq!(
        CompletionQueue::restore(&snapshot(&body)).err().unwrap(),
        SnapshotError::Repeated { index: 1 }
    );
    // The other way round.
    let body = [0x03, 0x02, 0x04, 0x00, first_out, 0x01, 0x02, 0x04];
    assert_eq!(
        CompletionQueue::restore(&snapshot(&body)).err().unwrap(),
        SnapshotError::NoSuchSyncpoint {
            index: 1,
            id: first_out
        }
    );
}

#[test]
fn the_header_is_checked_before_the_body() {
    // A bad version with a bad body and trailing bytes: the version is reported.
    let first_out = u8::try_from(SYNCPOINT_COUNT).unwrap();
    let mut bytes = SNAPSHOT_MAGIC.to_vec();
    bytes.push(u8::try_from(SNAPSHOT_VERSION + 1).unwrap());
    bytes.extend([0x01, 0x00, first_out, 0x01, 0xEE]);
    assert_eq!(
        CompletionQueue::restore(&bytes).err().unwrap(),
        SnapshotError::UnsupportedVersion {
            found: SNAPSHOT_VERSION + 1
        }
    );
    // Trailing bytes are reported before a bad entry.
    let mut bytes = snapshot(&[0x01, 0x00, first_out, 0x01]);
    bytes.push(0xEE);
    assert_eq!(
        CompletionQueue::restore(&bytes).err().unwrap(),
        SnapshotError::TrailingBytes { count: 1 }
    );
}
