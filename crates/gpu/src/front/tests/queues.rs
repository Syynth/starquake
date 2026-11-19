//! The queue bound: `Full` past 65,536 entries, `GpuQueueRoom` once the refused submission fits.

use super::*;
use crate::interface::CHANNEL_QUEUE_ENTRIES;

/// `count` one-word entries over a no-op word (an old-format header with no data).
fn nops(r: &Rig, count: usize) -> Vec<GpfifoEntry> {
    let nop = r.entry(CMD_A, &[0]);
    vec![nop; count]
}

fn room(channel: ChannelId) -> GpuCompletion {
    GpuCompletion::GpuQueueRoom { channel }
}

#[test]
fn a_submission_that_fills_the_queue_exactly_is_queued() {
    let r = rig();
    let a = r.channel(5);
    assert_eq!(
        r.submit(a, nops(&r, CHANNEL_QUEUE_ENTRIES - 1)),
        Submitted::Queued
    );
    assert_eq!(r.submit(a, nops(&r, 1)), Submitted::Queued);
    assert_eq!(r.shared.queued(a), Some(CHANNEL_QUEUE_ENTRIES));
}

#[test]
fn a_submission_that_would_pass_the_bound_is_full_and_queues_nothing() {
    let mut r = rig();
    let a = r.channel(5);
    assert_eq!(
        r.submit(a, nops(&r, CHANNEL_QUEUE_ENTRIES)),
        Submitted::Queued
    );
    assert_eq!(r.submit(a, nops(&r, 1)), Submitted::Full);
    assert_eq!(r.shared.queued(a), Some(CHANNEL_QUEUE_ENTRIES));
    // An empty submission still fits a full queue.
    assert_eq!(r.submit(a, vec![]), Submitted::Queued);
    // Another channel's queue is its own.
    let b = r.channel(6);
    assert_eq!(r.submit(b, nops(&r, 1)), Submitted::Queued);
    // Everything queued runs; the refused entry never does.
    r.settle();
    let total = u64::try_from(CHANNEL_QUEUE_ENTRIES).unwrap() + 1;
    assert_eq!(r.front.counters().words, total);
    // A's full one and its empty one, and B's.
    assert_eq!(r.front.counters().submissions, 3);
    assert_eq!(r.shared.queued(a), Some(0));
}

#[test]
fn room_is_reported_once_the_refused_submission_fits() {
    // A full queue drains a quantum of one-word entries per turn. A refused submission of a
    // quantum fits after one turn; one of a quantum and one entry needs a second.
    let quantum = usize::try_from(QUANTUM).unwrap();
    for (refused, turns) in [(quantum, 1), (quantum + 1, 2)] {
        let mut r = rig();
        let a = r.channel(5);
        let b = r.channel(6);
        assert_eq!(
            r.submit(a, nops(&r, CHANNEL_QUEUE_ENTRIES)),
            Submitted::Queued
        );
        assert_eq!(r.submit(a, nops(&r, refused)), Submitted::Full);
        for turn in 1..turns {
            r.front.turn().unwrap();
            assert_eq!(r.pushed.all(), [], "{refused}: turn {turn}");
        }
        r.front.turn().unwrap();
        assert_eq!(
            r.shared.queued(a),
            Some(CHANNEL_QUEUE_ENTRIES - turns * quantum)
        );
        assert_eq!(r.pushed.all(), [room(a)], "{refused}");
        // Once only, for the channel refused, not for another.
        r.submit(b, nops(&r, 1));
        r.settle();
        assert_eq!(r.pushed.all(), [room(a)], "{refused}");
        // The retried submission is queued now.
        assert_eq!(r.submit(a, nops(&r, refused)), Submitted::Queued);
    }
}

#[test]
fn room_waits_for_the_largest_refused_submission() {
    let quantum = usize::try_from(QUANTUM).unwrap();
    let mut r = rig();
    let a = r.channel(5);
    assert_eq!(
        r.submit(a, nops(&r, CHANNEL_QUEUE_ENTRIES)),
        Submitted::Queued
    );
    assert_eq!(r.submit(a, nops(&r, quantum + 1)), Submitted::Full);
    assert_eq!(r.submit(a, nops(&r, 3)), Submitted::Full);
    r.front.turn().unwrap();
    assert_eq!(r.pushed.all(), []);
    r.front.turn().unwrap();
    assert_eq!(r.pushed.all(), [room(a)]);
}

#[test]
fn a_submission_larger_than_the_queue_is_never_given_room() {
    let mut r = rig();
    let a = r.channel(5);
    assert_eq!(
        r.submit(a, nops(&r, CHANNEL_QUEUE_ENTRIES + 1)),
        Submitted::Full
    );
    assert_eq!(r.shared.queued(a), Some(0));
    r.submit(a, nops(&r, 1));
    r.settle();
    assert_eq!(r.pushed.all(), []);
}
