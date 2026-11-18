//! Tests of the vsync clock: its start, ticks, deadlines, listeners, replay and snapshot.

use std::collections::BTreeSet;
use std::sync::Arc;

use starquake_gpu::interface::Completions;
use starquake_gpu::syncpoint::Syncpoints;
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::result::{KernelError, SvcResult};
use starquake_kernel::slab::SlabId;
use starquake_trace::{EventKind, FixedClock, HostClock};

use super::*;
use crate::completion::CompletionQueue;
use crate::note::VSYNC_TICK;

/// The number the tests' gsv records its notes under: not the default, so a hard-coded one
/// fails.
const GSV: ServiceId = ServiceId(0x0657);

/// When the tests' first listener starts the clock: not zero, so a deadline counted from zero
/// fails.
const START: HostNanos = HostNanos(0x0123_4567);

/// The deadline [`START`] gives the first tick.
const FIRST: HostNanos = HostNanos(START.0 + INTERVAL_NS);

/// Event `n`: not generation 0 or 1, so an id built with another generation differs.
fn event(n: u32) -> ObjectId {
    ObjectId::from_parts(n, 5)
}

fn fresh() -> Shared {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    Shared::new(Arc::new(Syncpoints::new(completions)), queue, GSV)
}

/// A clock with listeners `events`, the first of which started it at [`START`].
fn listening(events: &[ObjectId]) -> Shared {
    let shared = fresh();
    for &listener in events {
        assert_eq!(shared.lock().listen_vsync(listener, START), Ok(true));
    }
    shared
}

/// One call a [`Journal`] host received.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Call {
    Record(EventKind),
    Signal(ObjectId),
}

/// A [`TimerHost`] that keeps every call in one list, in order, and refuses to signal the
/// events in `refused` as the kernel refuses an event it does not hold.
#[derive(Default)]
struct Journal {
    calls: Vec<Call>,
    refused: BTreeSet<ObjectId>,
}

impl TimerHost for Journal {
    fn record(&mut self, kind: EventKind) {
        self.calls.push(Call::Record(kind));
    }

    fn unlock_device_share(&mut self, _: DeviceShareToken) -> Result<(), DeviceShareError> {
        unreachable!("the vsync clock unlocks nothing")
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

/// The note of tick `tick`, as the journal sees it.
fn ticked(tick: u64) -> Call {
    Call::Record(Note::VsyncTick { tick }.event(GSV))
}

/// A live turn at `now`; what it did.
fn turn(shared: &Shared, now: u64) -> Vec<Call> {
    let mut host = Journal::default();
    shared.tick_vsync(&mut host, HostNanos(now));
    host.calls
}

/// The calls of tick `tick` signalling `events`.
fn tick_of(tick: u64, events: &[ObjectId]) -> Vec<Call> {
    let mut calls = vec![ticked(tick)];
    calls.extend(events.iter().copied().map(Call::Signal));
    calls
}

#[test]
fn the_interval_and_the_note_code_are_the_references_values() {
    // NX2 `VblankInterval{1'000'000'000LL / hz}`, hz 60, and `PaceToVblank`'s
    // `100'000'000'000LL / 60 / SpeedLimit()` at 100: both 16,666,666 ns. PR's
    // `std::chrono::microseconds{16667}` would be 16,667,000.
    assert_eq!(INTERVAL_NS, 16_666_666);
    assert_eq!(INTERVAL_NS, 100_000_000_000 / 60 / 100);
    assert_ne!(INTERVAL_NS, 16_667_000);
    // The note code the note table reserves for `VsyncTick`.
    assert_eq!(VSYNC_TICK, 4);
    assert_eq!(MAX_LISTENERS, 64);
    assert_eq!((SNAPSHOT_MAGIC, SNAPSHOT_VERSION), (*b"SQVS", 1));
}

#[test]
fn a_fresh_clock_is_stopped_and_has_ticked_nothing() {
    let shared = fresh();
    let state = shared.lock();
    let vsync = state.vsync();
    assert_eq!(vsync.tick(), 0);
    assert_eq!(vsync.deadline(), None);
    assert_eq!(vsync.replayed(), 0);
    assert_eq!(vsync.listeners().count(), 0);
    assert!(!vsync.is_due(HostNanos(u64::MAX)));
    drop(state);
    assert_eq!(shared.vsync_deadline(), None);
    // A turn of a stopped clock does nothing, whatever the time.
    assert_eq!(turn(&shared, u64::MAX), []);
}

#[test]
fn the_first_listener_starts_the_clock_one_interval_out() {
    let shared = fresh();
    assert_eq!(shared.lock().listen_vsync(event(1), START), Ok(true));
    assert_eq!(shared.vsync_deadline(), Some(FIRST));
    assert_eq!(
        shared.lock().vsync().listeners().collect::<Vec<_>>(),
        [event(1)]
    );
    // Starting ticks nothing.
    assert_eq!(shared.lock().vsync().tick(), 0);
}

#[test]
fn a_second_listener_joins_the_running_clock() {
    let shared = listening(&[event(1)]);
    let later = HostNanos(START.0 + INTERVAL_NS / 2);
    assert_eq!(shared.lock().listen_vsync(event(2), later), Ok(true));
    assert_eq!(shared.vsync_deadline(), Some(FIRST));
    // The same event again changes nothing, the deadline included.
    assert_eq!(shared.lock().listen_vsync(event(1), later), Ok(false));
    assert_eq!(shared.vsync_deadline(), Some(FIRST));
    assert_eq!(turn(&shared, FIRST.0), tick_of(1, &[event(1), event(2)]));
}

#[test]
fn a_tick_comes_at_exactly_the_deadline_and_not_before() {
    let shared = listening(&[event(1)]);
    assert!(!shared.lock().vsync().is_due(HostNanos(FIRST.0 - 1)));
    assert_eq!(turn(&shared, FIRST.0 - 1), []);
    assert_eq!(shared.vsync_deadline(), Some(FIRST));
    assert_eq!(shared.lock().vsync().tick(), 0);

    assert!(shared.lock().vsync().is_due(FIRST));
    assert_eq!(turn(&shared, FIRST.0), tick_of(1, &[event(1)]));
    assert_eq!(shared.lock().vsync().tick(), 1);
    assert_eq!(
        shared.vsync_deadline(),
        Some(HostNanos(FIRST.0 + INTERVAL_NS))
    );
}

#[test]
fn a_tick_records_its_number_then_signals_every_listener_in_event_order() {
    // Added out of order, with two that differ only in their generation.
    let low_generation = ObjectId::from_parts(4, 2);
    let shared = listening(&[event(7), event(4), low_generation, event(9)]);
    let order = [low_generation, event(4), event(7), event(9)];
    assert_eq!(turn(&shared, FIRST.0), tick_of(1, &order));
    let second = FIRST.0 + INTERVAL_NS;
    assert_eq!(turn(&shared, second), tick_of(2, &order));
}

#[test]
fn on_a_fixed_clock_ticks_fall_at_exact_multiples() {
    // Turns a millisecond apart, the first at the start: each tick comes at the first turn at or
    // after `START + k * INTERVAL_NS`, and the deadline after it is the next multiple, exactly.
    let step = 1_000_000;
    let mut clock = FixedClock::new(START.0, step);
    let shared = listening(&[event(1)]);
    let mut ticks = 0;
    let mut last = START.0;
    for _ in 0..200 {
        let now = clock.now().0;
        let calls = turn(&shared, now);
        let grid = START.0 + (ticks + 1) * INTERVAL_NS;
        if now >= grid {
            ticks += 1;
            assert_eq!(calls, tick_of(ticks, &[event(1)]), "at {now}");
            assert!(now - grid < step, "a turn at {now} for {grid}");
            let next = START.0 + (ticks + 1) * INTERVAL_NS;
            assert_eq!(shared.vsync_deadline(), Some(HostNanos(next)));
        } else {
            assert_eq!(calls, [], "at {now}");
        }
        last = now;
    }
    assert_eq!(ticks, (last - START.0) / INTERVAL_NS);
    assert_eq!(ticks, 11);

    // Turns exactly one interval apart, at the deadlines: a tick at every turn.
    let mut clock = FixedClock::new(FIRST.0, INTERVAL_NS);
    let shared = listening(&[event(1)]);
    for tick in 1..=5 {
        assert_eq!(turn(&shared, clock.now().0), tick_of(tick, &[event(1)]));
        let next = START.0 + (tick + 1) * INTERVAL_NS;
        assert_eq!(shared.vsync_deadline(), Some(HostNanos(next)));
    }
}

#[test]
fn a_late_tick_does_not_double_up() {
    // A turn five intervals late ticks once, and the grid restarts one interval after it.
    let shared = listening(&[event(1)]);
    let late = FIRST.0 + 5 * INTERVAL_NS;
    assert_eq!(turn(&shared, late), tick_of(1, &[event(1)]));
    assert_eq!(shared.vsync_deadline(), Some(HostNanos(late + INTERVAL_NS)));
    // Another turn at the same time, or just before the new deadline, ticks nothing.
    assert_eq!(turn(&shared, late), []);
    assert_eq!(turn(&shared, late + INTERVAL_NS - 1), []);
    assert_eq!(turn(&shared, late + INTERVAL_NS), tick_of(2, &[event(1)]));
}

#[test]
fn a_tick_late_by_less_than_an_interval_keeps_the_grid() {
    // NX2 `PaceToVblank`: not more than one interval late, so `m_next_vblank += interval`.
    let shared = listening(&[event(1)]);
    let late = FIRST.0 + INTERVAL_NS - 1;
    assert_eq!(turn(&shared, late), tick_of(1, &[event(1)]));
    assert_eq!(
        shared.vsync_deadline(),
        Some(HostNanos(FIRST.0 + INTERVAL_NS))
    );
}

#[test]
fn a_tick_late_by_exactly_one_interval_keeps_the_grid() {
    // NX2's `now > m_next_vblank + interval` is false at exactly one interval: the grid is kept,
    // so the next deadline is the turn's own time, and the next turn ticks again.
    let shared = listening(&[event(1)]);
    let late = FIRST.0 + INTERVAL_NS;
    assert_eq!(turn(&shared, late), tick_of(1, &[event(1)]));
    assert_eq!(shared.vsync_deadline(), Some(HostNanos(late)));
    assert_eq!(turn(&shared, late), tick_of(2, &[event(1)]));
    assert_eq!(shared.vsync_deadline(), Some(HostNanos(late + INTERVAL_NS)));
}

#[test]
fn a_tick_late_by_one_interval_and_a_nanosecond_restarts_the_grid() {
    let shared = listening(&[event(1)]);
    let late = FIRST.0 + INTERVAL_NS + 1;
    assert_eq!(turn(&shared, late), tick_of(1, &[event(1)]));
    assert_eq!(shared.vsync_deadline(), Some(HostNanos(late + INTERVAL_NS)));
    assert_eq!(turn(&shared, late), []);
}

#[test]
fn the_last_listener_leaving_stops_the_clock_and_the_next_restarts_it() {
    let shared = listening(&[event(1), event(2)]);
    assert_eq!(turn(&shared, FIRST.0), tick_of(1, &[event(1), event(2)]));
    let deadline = shared.vsync_deadline();
    // One of two leaves: the clock runs on, its deadline unchanged.
    assert!(shared.lock().unlisten_vsync(event(1)));
    assert_eq!(shared.vsync_deadline(), deadline);
    // An event that does not listen, or one that differs only in its generation, changes
    // nothing.
    assert!(!shared.lock().unlisten_vsync(event(1)));
    assert!(!shared.lock().unlisten_vsync(ObjectId::from_parts(2, 6)));
    assert_eq!(shared.vsync_deadline(), deadline);
    assert_eq!(
        shared.lock().vsync().listeners().collect::<Vec<_>>(),
        [event(2)]
    );
    // The last leaves: the clock stops, keeping its count.
    assert!(shared.lock().unlisten_vsync(event(2)));
    assert_eq!(shared.vsync_deadline(), None);
    assert_eq!(turn(&shared, u64::MAX), []);
    assert_eq!(shared.lock().vsync().tick(), 1);
    // The next listener starts it again, one interval after its own time, counting on.
    let again = HostNanos(FIRST.0 + 7 * INTERVAL_NS + 3);
    assert_eq!(shared.lock().listen_vsync(event(3), again), Ok(true));
    let restarted = again.0 + INTERVAL_NS;
    assert_eq!(shared.vsync_deadline(), Some(HostNanos(restarted)));
    assert_eq!(turn(&shared, restarted - 1), []);
    assert_eq!(turn(&shared, restarted), tick_of(2, &[event(3)]));
}

#[test]
fn a_refused_listener_is_dropped_and_the_rest_are_signalled() {
    let shared = listening(&[event(1), event(2), event(3)]);
    let mut host = Journal {
        refused: [event(2)].into(),
        ..Journal::default()
    };
    shared.tick_vsync(&mut host, FIRST);
    assert_eq!(host.calls, tick_of(1, &[event(1), event(2), event(3)]));
    assert_eq!(
        shared.lock().vsync().listeners().collect::<Vec<_>>(),
        [event(1), event(3)]
    );
    let second = FIRST.0 + INTERVAL_NS;
    assert_eq!(shared.vsync_deadline(), Some(HostNanos(second)));
    assert_eq!(turn(&shared, second), tick_of(2, &[event(1), event(3)]));

    // Every listener refused: the tick is still delivered, and the clock stops after it.
    let mut host = Journal {
        refused: [event(1), event(3)].into(),
        ..Journal::default()
    };
    let third = second + INTERVAL_NS;
    shared.tick_vsync(&mut host, HostNanos(third));
    assert_eq!(host.calls, tick_of(3, &[event(1), event(3)]));
    assert_eq!(shared.lock().vsync().listeners().count(), 0);
    assert_eq!(shared.vsync_deadline(), None);
    assert_eq!(turn(&shared, third + INTERVAL_NS), []);
}

#[test]
fn listeners_are_refused_past_the_cap() {
    let shared = fresh();
    let count = u32::try_from(MAX_LISTENERS).unwrap();
    for n in 0..count {
        assert_eq!(shared.lock().listen_vsync(event(n), START), Ok(true));
    }
    let before = shared.lock().vsync().clone();
    let error = shared.lock().listen_vsync(event(count), START).unwrap_err();
    assert_eq!(error, VsyncError::TooManyListeners);
    assert_eq!(
        error.to_string(),
        "the vsync clock already has the most listeners it takes (64)"
    );
    assert_eq!(*shared.lock().vsync(), before);
    // At the cap an event already listening is accepted first, and changes nothing.
    assert_eq!(shared.lock().listen_vsync(event(0), START), Ok(false));
    assert_eq!(*shared.lock().vsync(), before);
    // One leaves, and the refused one fits.
    assert!(shared.lock().unlisten_vsync(event(0)));
    assert_eq!(shared.lock().listen_vsync(event(count), START), Ok(true));
    assert_eq!(shared.lock().vsync().listeners().count(), MAX_LISTENERS);
}

#[test]
fn a_deadline_past_the_end_of_time_stops_the_clock() {
    // A start whose deadline is exactly the last nanosecond runs; one a nanosecond later cannot.
    let shared = fresh();
    let last = HostNanos(u64::MAX - INTERVAL_NS);
    assert_eq!(shared.lock().listen_vsync(event(1), last), Ok(true));
    assert_eq!(shared.vsync_deadline(), Some(HostNanos(u64::MAX)));
    // Its tick is delivered; the one after would be past the end, so the clock stops.
    assert_eq!(turn(&shared, u64::MAX), tick_of(1, &[event(1)]));
    assert_eq!(shared.vsync_deadline(), None);
    assert_eq!(turn(&shared, u64::MAX), []);

    let shared = fresh();
    let past = HostNanos(last.0 + 1);
    assert_eq!(shared.lock().listen_vsync(event(1), past), Ok(true));
    assert_eq!(shared.vsync_deadline(), None);
    assert_eq!(turn(&shared, u64::MAX), []);

    // A deadline one interval short of the end ticks and keeps one more.
    let shared = fresh();
    let early = HostNanos(last.0 - INTERVAL_NS);
    assert_eq!(shared.lock().listen_vsync(event(1), early), Ok(true));
    assert_eq!(turn(&shared, last.0), tick_of(1, &[event(1)]));
    assert_eq!(shared.vsync_deadline(), Some(HostNanos(u64::MAX)));
}

/// The bytes of a snapshot holding `tick`, `next`, `replayed` and `listeners`, laid out by hand
/// (module docs, "What is saved"): the header, then each field as `postcard` writes it.
fn snapshot(tick: u64, next: Option<u64>, replayed: u64, listeners: &[(u32, u32)]) -> Vec<u8> {
    let mut bytes = b"SQVS".to_vec();
    varint(&mut bytes, 1);
    varint(&mut bytes, tick);
    match next {
        None => bytes.push(0),
        Some(next) => {
            bytes.push(1);
            varint(&mut bytes, next);
        }
    }
    varint(&mut bytes, replayed);
    varint(&mut bytes, u64::try_from(listeners.len()).unwrap());
    for &(index, generation) in listeners {
        varint(&mut bytes, u64::from(index));
        varint(&mut bytes, u64::from(generation));
    }
    bytes
}

/// `postcard`'s variable-length integer: seven bits a byte, low first, the top bit set on every
/// byte but the last.
fn varint(bytes: &mut Vec<u8>, mut value: u64) {
    loop {
        let low = u8::try_from(value & 0x7F).unwrap();
        value >>= 7;
        if value == 0 {
            bytes.push(low);
            return;
        }
        bytes.push(low | 0x80);
    }
}

#[test]
fn a_tick_count_at_its_maximum_stops_the_clock() {
    let shared = fresh();
    let one_short = snapshot(u64::MAX - 1, Some(FIRST.0), 0, &[(1, 5)]);
    shared.restore_vsync(&one_short).unwrap();
    assert_eq!(turn(&shared, FIRST.0), tick_of(u64::MAX, &[event(1)]));
    let next = FIRST.0 + INTERVAL_NS;
    assert_eq!(shared.vsync_deadline(), Some(HostNanos(next)));
    // No number is left: nothing is recorded or signalled, and the clock stops.
    assert_eq!(turn(&shared, next), []);
    assert_eq!(shared.vsync_deadline(), None);
    assert_eq!(shared.lock().vsync().tick(), u64::MAX);
}

#[test]
fn a_replayed_tick_is_delivered_as_the_live_one_was() {
    let live = listening(&[event(1), event(2)]);
    let ran = turn(&live, FIRST.0);
    assert_eq!(ran, tick_of(1, &[event(1), event(2)]));

    let replay = listening(&[event(1), event(2)]);
    assert_eq!(replay.replay_vsync(1), Ok(()));
    assert_eq!(replay.lock().vsync().replayed(), 1);
    let mut host = Journal::default();
    replay.tick_replayed_vsync(&mut host);
    assert_eq!(host.calls, ran);
    assert_eq!(replay.lock().vsync().tick(), 1);
    assert_eq!(replay.lock().vsync().replayed(), 0);
    // A replay turn with nothing fed back does nothing.
    let mut host = Journal::default();
    replay.tick_replayed_vsync(&mut host);
    assert_eq!(host.calls, []);
}

#[test]
fn ticks_fed_back_together_are_delivered_in_order() {
    let shared = listening(&[event(1)]);
    for tick in 1..=3 {
        assert_eq!(shared.replay_vsync(tick), Ok(()));
    }
    let mut host = Journal::default();
    shared.tick_replayed_vsync(&mut host);
    let mut expected = tick_of(1, &[event(1)]);
    expected.extend(tick_of(2, &[event(1)]));
    expected.extend(tick_of(3, &[event(1)]));
    assert_eq!(host.calls, expected);
}

#[test]
fn a_tick_fed_back_out_of_order_is_refused_and_changes_nothing() {
    let shared = listening(&[event(1)]);
    // Before the first: 0 and 2 are refused, 1 is next.
    for found in [0, 2, u64::MAX] {
        let error = shared.replay_vsync(found).unwrap_err();
        assert_eq!(error, VsyncReplayError::OutOfOrder { expected: 1, found });
        assert_eq!(shared.lock().vsync().replayed(), 0);
    }
    assert_eq!(
        VsyncReplayError::OutOfOrder {
            expected: 1,
            found: 2
        }
        .to_string(),
        "vsync tick 2 was fed back where tick 1 comes next"
    );
    // A tick fed back and not delivered counts: the same one again is refused, the next taken.
    assert_eq!(shared.replay_vsync(1), Ok(()));
    let again = shared.replay_vsync(1).unwrap_err();
    assert_eq!(
        again,
        VsyncReplayError::OutOfOrder {
            expected: 2,
            found: 1
        }
    );
    assert_eq!(shared.replay_vsync(2), Ok(()));
    // A delivered tick counts too.
    let mut host = Journal::default();
    shared.tick_replayed_vsync(&mut host);
    let after = shared.replay_vsync(2).unwrap_err();
    assert_eq!(
        after,
        VsyncReplayError::OutOfOrder {
            expected: 3,
            found: 2
        }
    );
    assert_eq!(shared.replay_vsync(3), Ok(()));
}

#[test]
fn no_tick_is_fed_back_past_the_last_number() {
    // A count one short of the end takes one more tick, then none.
    let shared = fresh();
    shared
        .restore_vsync(&snapshot(u64::MAX - 1, None, 0, &[]))
        .unwrap();
    assert_eq!(shared.replay_vsync(u64::MAX), Ok(()));
    let error = shared.replay_vsync(u64::MAX).unwrap_err();
    assert_eq!(error, VsyncReplayError::CountExhausted);
    assert_eq!(
        error.to_string(),
        "no vsync tick comes after tick 18446744073709551615"
    );
    assert_eq!(shared.lock().vsync().replayed(), 1);
    // The tick fed back is delivered with the last number.
    shared.lock().listen_vsync(event(1), START).unwrap();
    let mut host = Journal::default();
    shared.tick_replayed_vsync(&mut host);
    assert_eq!(host.calls, tick_of(u64::MAX, &[event(1)]));
}

#[test]
fn the_live_and_replay_turns_each_ignore_the_other_s_input() {
    // A live turn does not deliver a tick fed back, and a replay turn does not tick at a
    // deadline.
    let shared = listening(&[event(1)]);
    assert_eq!(shared.replay_vsync(1), Ok(()));
    assert_eq!(turn(&shared, FIRST.0 - 1), []);
    assert_eq!(shared.lock().vsync().replayed(), 1);

    let shared = listening(&[event(1)]);
    let mut host = Journal::default();
    shared.tick_replayed_vsync(&mut host);
    assert_eq!(host.calls, []);
    assert_eq!(shared.vsync_deadline(), Some(FIRST));
    assert_eq!(shared.lock().vsync().tick(), 0);
}

// ----- Snapshots ---------------------------------------------------------------------------------

#[test]
fn the_vsync_snapshot_layout_is_pinned() {
    // Seven ticks delivered, three more fed back, two listeners started at 5.
    let shared = fresh();
    for tick in 1..=7 {
        shared.replay_vsync(tick).unwrap();
    }
    shared.tick_replayed_vsync(&mut Journal::default());
    for tick in 8..=10 {
        shared.replay_vsync(tick).unwrap();
    }
    let first = ObjectId::from_parts(1, 2);
    let second = ObjectId::from_parts(4, 6);
    shared.lock().listen_vsync(second, HostNanos(5)).unwrap();
    shared.lock().listen_vsync(first, HostNanos(5)).unwrap();
    // The magic and version 1; the tick count; the deadline (some: 5 + 16,666,666 as a varint);
    // the ticks fed back; the listeners (two: each index and generation, ascending).
    let mut expected = vec![b'S', b'Q', b'V', b'S', 0x01];
    expected.push(0x07);
    expected.extend([0x01, 0xAF, 0xA0, 0xF9, 0x07]);
    expected.push(0x03);
    expected.extend([0x02, 0x01, 0x02, 0x04, 0x06]);
    assert_eq!(shared.save_vsync().unwrap(), expected);
    assert_eq!(
        expected,
        snapshot(7, Some(5 + INTERVAL_NS), 3, &[(1, 2), (4, 6)])
    );
}

#[test]
fn a_stopped_clock_s_snapshot_layout_is_pinned() {
    let shared = listening(&[event(1)]);
    turn(&shared, FIRST.0);
    turn(&shared, FIRST.0 + INTERVAL_NS);
    shared.lock().unlisten_vsync(event(1));
    // Two ticks; no deadline; nothing fed back; no listener.
    let expected = vec![b'S', b'Q', b'V', b'S', 0x01, 0x02, 0x00, 0x00, 0x00];
    assert_eq!(shared.save_vsync().unwrap(), expected);
}

#[test]
fn a_restored_clock_is_used_as_saved() {
    // The state live operations leave: ticks delivered, a running clock with two listeners.
    let saved = listening(&[event(1), event(2)]);
    turn(&saved, FIRST.0);
    let bytes = saved.save_vsync().unwrap();

    let restored = fresh();
    restored.restore_vsync(&bytes).unwrap();
    assert_eq!(*restored.lock().vsync(), *saved.lock().vsync());
    let next = FIRST.0 + INTERVAL_NS;
    assert_eq!(turn(&restored, next - 1), []);
    assert_eq!(turn(&restored, next), tick_of(2, &[event(1), event(2)]));
    assert!(restored.lock().unlisten_vsync(event(1)));
    assert_eq!(restored.replay_vsync(3), Ok(()));
}

/// Restores `bytes` into a clock that already runs; the error, after checking the clock is
/// unchanged.
fn refused(bytes: &[u8]) -> VsyncSnapshotError {
    let shared = listening(&[event(9)]);
    turn(&shared, FIRST.0);
    let before = shared.lock().vsync().clone();
    let error = shared.restore_vsync(bytes).unwrap_err();
    assert_eq!(*shared.lock().vsync(), before);
    error
}

#[test]
fn a_snapshot_s_header_is_checked_before_its_body() {
    let good = snapshot(1, Some(FIRST.0), 0, &[(1, 5)]);
    // A bad magic over a bad version and a body with a duplicate listener: the magic first.
    let mut bad = snapshot(1, Some(FIRST.0), 0, &[(1, 5), (1, 5)]);
    bad[0] = b'X';
    bad[4] = 0x02;
    let error = refused(&bad);
    assert_eq!(error, VsyncSnapshotError::BadMagic);
    assert_eq!(error.to_string(), "not a vsync clock snapshot (bad magic)");
    // A bad version over a bad body: the version.
    let mut version = snapshot(1, Some(FIRST.0), 0, &[(1, 5), (1, 5)]);
    version[4] = 0x02;
    let error = refused(&version);
    assert_eq!(error, VsyncSnapshotError::UnsupportedVersion { found: 2 });
    assert_eq!(
        error.to_string(),
        "vsync clock snapshot layout version 2 is not supported (this build reads 1)"
    );
    // A body cut short is a decode error; a byte after it is refused.
    let short = &good[..good.len() - 1];
    assert!(matches!(refused(short), VsyncSnapshotError::Decode(_)));
    let mut long = good.clone();
    long.push(0);
    let error = refused(&long);
    assert_eq!(error, VsyncSnapshotError::TrailingBytes { count: 1 });
    assert_eq!(
        error.to_string(),
        "1 unexpected bytes after the vsync clock snapshot"
    );
    // The good one restores.
    fresh().restore_vsync(&good).unwrap();
}

#[test]
fn a_snapshot_holds_at_most_the_cap_of_listeners() {
    let count = u32::try_from(MAX_LISTENERS).unwrap();
    let at_cap: Vec<(u32, u32)> = (0..count).map(|n| (n, 5)).collect();
    let shared = fresh();
    shared
        .restore_vsync(&snapshot(0, Some(FIRST.0), 0, &at_cap))
        .unwrap();
    assert_eq!(shared.lock().vsync().listeners().count(), MAX_LISTENERS);
    let past: Vec<(u32, u32)> = (0..=count).map(|n| (n, 5)).collect();
    let error = refused(&snapshot(0, Some(FIRST.0), 0, &past));
    assert_eq!(
        error,
        VsyncSnapshotError::TooManyListeners {
            count: MAX_LISTENERS + 1
        }
    );
    assert_eq!(
        error.to_string(),
        "the vsync clock snapshot holds 65 listeners, more than 64"
    );
    // Past the cap and with a duplicate: the count first.
    let mut both = past;
    both[1] = both[0];
    let error = refused(&snapshot(0, Some(FIRST.0), 0, &both));
    assert_eq!(
        error,
        VsyncSnapshotError::TooManyListeners {
            count: MAX_LISTENERS + 1
        }
    );
}

#[test]
fn a_snapshot_s_listeners_are_strictly_ascending() {
    // A duplicate, a swapped pair, and a pair that differs only in its generation, swapped;
    // each at index 2, after an ascending pair.
    let cases = [
        [(1, 5), (2, 5), (2, 5)],
        [(1, 5), (3, 5), (2, 5)],
        [(1, 5), (2, 6), (2, 5)],
    ];
    for listeners in cases {
        let error = refused(&snapshot(0, Some(FIRST.0), 0, &listeners));
        assert_eq!(
            error,
            VsyncSnapshotError::ListenersNotAscending { index: 2 },
            "{listeners:?}"
        );
    }
    assert_eq!(
        VsyncSnapshotError::ListenersNotAscending { index: 2 }.to_string(),
        "the vsync clock snapshot's listener 2 is not above the one before it (listed twice or \
         out of order)"
    );
    // The same generations ascending are taken.
    fresh()
        .restore_vsync(&snapshot(0, Some(FIRST.0), 0, &[(1, 5), (2, 5), (2, 6)]))
        .unwrap();
    // A misordered list and an overflowing count: the order first.
    let error = refused(&snapshot(u64::MAX, Some(FIRST.0), 1, &[(2, 5), (1, 5)]));
    assert_eq!(
        error,
        VsyncSnapshotError::ListenersNotAscending { index: 1 }
    );
}

#[test]
fn a_snapshot_s_deadline_needs_a_listener() {
    let error = refused(&snapshot(3, Some(FIRST.0), 0, &[]));
    assert_eq!(error, VsyncSnapshotError::DeadlineWithoutListeners);
    assert_eq!(
        error.to_string(),
        "the vsync clock snapshot has a deadline but no listeners"
    );
    // Neither, or both, are taken; and listeners with no deadline (the end of the counts).
    for (next, listeners) in [
        (None, &[][..]),
        (Some(FIRST.0), &[(1, 5)][..]),
        (None, &[(1, 5)][..]),
    ] {
        fresh()
            .restore_vsync(&snapshot(3, next, 0, listeners))
            .unwrap();
    }
    // A deadline with no listener and an overflowing count: the deadline first.
    let error = refused(&snapshot(u64::MAX, Some(FIRST.0), 1, &[]));
    assert_eq!(error, VsyncSnapshotError::DeadlineWithoutListeners);
}

#[test]
fn a_snapshot_s_ticks_fit_a_u64() {
    // A sum of exactly `u64::MAX` is taken, one more refused, either way round.
    for (tick, replayed) in [(u64::MAX - 1, 1), (1, u64::MAX - 1), (u64::MAX, 0)] {
        fresh()
            .restore_vsync(&snapshot(tick, None, replayed, &[]))
            .unwrap();
    }
    for (tick, replayed) in [(u64::MAX, 1), (1, u64::MAX), (u64::MAX, u64::MAX)] {
        let error = refused(&snapshot(tick, None, replayed, &[]));
        assert_eq!(error, VsyncSnapshotError::TicksOverflow { tick, replayed });
    }
    assert_eq!(
        VsyncSnapshotError::TicksOverflow {
            tick: 2,
            replayed: 3
        }
        .to_string(),
        "the vsync clock snapshot's 2 ticks and 3 ticks fed back add up past the largest tick \
         number"
    );
}
