//! The queue bound: `Full` past 65,536 entries (a submission with no entries counted as one),
//! `GpuQueueRoom` once the refused submission fits.

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
    // Another channel's queue is its own.
    let b = r.channel(6);
    assert_eq!(r.submit(b, nops(&r, 1)), Submitted::Queued);
    // Everything queued runs; the refused entry never does.
    r.settle();
    let total = u64::try_from(CHANNEL_QUEUE_ENTRIES).unwrap() + 1;
    assert_eq!(r.front.counters().words, total);
    // A's full one and B's.
    assert_eq!(r.front.counters().submissions, 2);
    assert_eq!(r.shared.queued(a), Some(0));
}

/// `count` submissions with no entries on `channel`; what each was answered.
fn empties(r: &Rig, channel: ChannelId, count: usize) -> Vec<Submitted> {
    (0..count).map(|_| r.submit(channel, vec![])).collect()
}

#[test]
fn zero_entry_submissions_fill_the_queue_one_each() {
    // Each is charged one entry (`charge`): exactly a queue's worth is taken, the next is full.
    let mut r = rig();
    let a = r.channel(5);
    let answers = empties(&r, a, CHANNEL_QUEUE_ENTRIES);
    assert!(answers.iter().all(|&answer| answer == Submitted::Queued));
    assert_eq!(r.shared.queued(a), Some(CHANNEL_QUEUE_ENTRIES));
    assert_eq!(r.submit(a, vec![]), Submitted::Full);
    assert_eq!(r.submit(a, nops(&r, 1)), Submitted::Full);
    assert_eq!(r.shared.queued(a), Some(CHANNEL_QUEUE_ENTRIES));
    // Taken off the queue, each releases its one: room for the refused one is reported once.
    r.settle();
    assert_eq!(r.shared.queued(a), Some(0));
    assert_eq!(r.pushed.all(), [room(a)]);
    let finished = u64::try_from(CHANNEL_QUEUE_ENTRIES).unwrap();
    assert_eq!(r.front.counters().submissions, finished);
    assert_eq!(r.submit(a, vec![]), Submitted::Queued);
}

#[test]
fn a_zero_entry_submission_on_a_full_queue_is_full() {
    let mut r = rig();
    let a = r.channel(5);
    // One short of full: an empty submission takes the last place.
    assert_eq!(
        r.submit(a, nops(&r, CHANNEL_QUEUE_ENTRIES - 1)),
        Submitted::Queued
    );
    assert_eq!(r.submit(a, vec![]), Submitted::Queued);
    assert_eq!(r.shared.queued(a), Some(CHANNEL_QUEUE_ENTRIES));
    // Full: another is refused and queues nothing.
    assert_eq!(r.submit(a, vec![]), Submitted::Full);
    assert_eq!(r.shared.queued(a), Some(CHANNEL_QUEUE_ENTRIES));
    // One turn starts a quantum of entries: the refused empty one fits, and room is reported.
    r.front.turn().unwrap();
    assert_eq!(r.pushed.all(), [room(a)]);
    r.settle();
    assert_eq!(r.shared.queued(a), Some(0));
    assert_eq!(r.pushed.all(), [room(a)]);
}

#[test]
fn a_stopped_channel_releases_what_its_empty_submissions_were_charged() {
    // Queued behind an entry that stops the channel, and submitted after it stopped.
    let unmapped = GPU_BASE + PAGES * PAGE_SIZE;
    let mut r = rig();
    let a = r.channel(5);
    r.submit(a, vec![raw_entry(unmapped, 1, Sync::Proceed)]);
    empties(&r, a, 3);
    assert_eq!(r.shared.queued(a), Some(4));
    r.settle();
    assert!(r.shared.channel_error(a).is_some());
    assert_eq!(r.shared.queued(a), Some(0));
    empties(&r, a, 2);
    assert_eq!(r.shared.queued(a), Some(2));
    r.settle();
    assert_eq!(r.shared.queued(a), Some(0));
}

#[test]
fn a_channel_stopped_inside_a_submission_releases_its_unstarted_entries() {
    // The first of two entries stops the channel: the second, never started, leaves the bound.
    let unmapped = GPU_BASE + PAGES * PAGE_SIZE;
    let mut r = rig();
    let a = r.channel(5);
    let nop = r.entry(CMD_A, &[0]);
    r.submit(a, vec![raw_entry(unmapped, 1, Sync::Proceed), nop]);
    r.settle();
    assert!(r.shared.channel_error(a).is_some());
    assert_eq!(r.shared.queued(a), Some(0));
}

#[test]
fn a_full_queue_stopped_at_its_first_entry_reports_room_once() {
    let unmapped = GPU_BASE + PAGES * PAGE_SIZE;
    let mut r = rig();
    let a = r.channel(5);
    let mut entries = nops(&r, CHANNEL_QUEUE_ENTRIES);
    entries[0] = raw_entry(unmapped, 1, Sync::Proceed);
    assert_eq!(r.submit(a, entries), Submitted::Queued);
    assert_eq!(r.submit(a, nops(&r, 1)), Submitted::Full);
    r.settle();
    assert!(r.shared.channel_error(a).is_some());
    assert_eq!(r.shared.queued(a), Some(0));
    assert_eq!(r.pushed.all(), [room(a)]);
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
