//! The vsync clock in the graphics server's turn, deadline and idleness, in both front modes, and
//! the replay of its notes (module docs, "The timer thread's turn" and "Replay"), over a
//! [`Graphics`] of its own (no kernel). Through a real dispatcher: `hle/tests/graphics/vsync.rs`.

use starquake_gsv::shared::vsync::{INTERVAL_NS, VsyncReplayError};
use starquake_kernel::id::SlabId;

use super::*;

/// When the tests' listener starts the clock: not zero, so a deadline counted from zero fails.
const START: HostNanos = HostNanos(0x0123_4567);

/// The first tick's deadline.
const FIRST: HostNanos = HostNanos(START.0 + INTERVAL_NS);

/// The tests' listener: not generation 0 or 1.
fn listener() -> ObjectId {
    ObjectId::from_parts(3, 7)
}

/// A [`TimerHost`] that keeps what was recorded and the events it was asked to signal, each of
/// which it signals.
#[derive(Default)]
struct Journal(Vec<EventKind>, Vec<ObjectId>);

impl TimerHost for Journal {
    fn record(&mut self, kind: EventKind) {
        self.0.push(kind);
    }

    fn unlock_device_share(&mut self, _: DeviceShareToken) -> Result<(), DeviceShareError> {
        Ok(())
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        self.1.push(event);
        Ok(())
    }
}

/// The turn at `now`: what it recorded and signalled.
fn turn_at(graphics: &Graphics, now: HostNanos) -> Journal {
    let mut host = Journal::default();
    graphics.turn(&mut host, now);
    host
}

fn ticked(tick: u64) -> EventKind {
    Note::VsyncTick { tick }.event(GSV_ROOT)
}

/// In a run the turn ticks the clock at its deadline and not before, recording for the root
/// and signalling the listener; the deadline is the turn's next one.
#[test]
fn a_run_s_turn_ticks_the_vsync_clock_at_its_deadline() {
    let graphics = graphics(FrontMode::Live);
    assert_eq!(graphics.deadline(), None);
    let listen = graphics.shared().lock().listen_vsync(listener(), START);
    assert_eq!(listen, Ok(true));
    assert_eq!(graphics.deadline(), Some(FIRST));

    let early = turn_at(&graphics, HostNanos(FIRST.0 - 1));
    assert_eq!((early.0, early.1), (vec![], vec![]));
    let due = turn_at(&graphics, FIRST);
    assert_eq!((due.0, due.1), (vec![ticked(1)], vec![listener()]));
    assert_eq!(graphics.deadline(), Some(HostNanos(FIRST.0 + INTERVAL_NS)));
}

/// The timer thread wants a turn when any one input is there, each tried alone: something
/// queued, the clock's deadline come (not a nanosecond before), or a tick fed back for replay;
/// and none with nothing. A running clock keeps the server from being idle; a stopped one does
/// not.
#[test]
fn the_turn_is_wanted_when_any_one_input_is_there() {
    let graphics = graphics(FrontMode::Live);
    assert!(graphics.is_idle());
    assert!(!graphics.wants_turn(HostNanos(u64::MAX)));

    // Something queued, the clock stopped.
    graphics.front().barrier(Ticket(0x91));
    assert!(!graphics.is_idle());
    assert!(graphics.wants_turn(HostNanos(0)));
    turn_at(&graphics, HostNanos(0));
    assert!(graphics.is_idle());

    // The clock running, nothing queued: not idle, and a turn wanted from its deadline on.
    graphics
        .shared()
        .lock()
        .listen_vsync(listener(), START)
        .unwrap();
    assert!(!graphics.is_idle());
    assert!(!graphics.wants_turn(HostNanos(FIRST.0 - 1)));
    assert!(graphics.wants_turn(FIRST));
    // Stopped again: idle, and no turn wanted at any time.
    assert!(graphics.shared().lock().unlisten_vsync(listener()));
    assert!(graphics.is_idle());
    assert!(!graphics.wants_turn(HostNanos(u64::MAX)));

    // A tick fed back, in a replay, nothing else.
    let replay = super::graphics(FrontMode::Replay);
    assert!(replay.is_idle());
    replay.replay(Note::VsyncTick { tick: 1 }).unwrap();
    assert!(replay.shared().queue().is_empty());
    assert!(!replay.is_idle());
    assert!(replay.wants_turn(HostNanos(0)));
    let delivered = turn_at(&replay, HostNanos(0));
    assert_eq!(delivered.0, [ticked(1)]);
    assert!(replay.is_idle());
    assert!(!replay.wants_turn(HostNanos(u64::MAX)));
}

/// In a replay the clock does not tick or wake the timer thread on its own: a listener gives no
/// deadline and keeps nothing busy, and a turn long after its deadline ticks nothing.
#[test]
fn in_a_replay_the_clock_neither_ticks_nor_wakes_on_its_own() {
    let graphics = graphics(FrontMode::Replay);
    graphics
        .shared()
        .lock()
        .listen_vsync(listener(), START)
        .unwrap();
    assert_eq!(graphics.shared().vsync_deadline(), Some(FIRST));
    assert_eq!(graphics.deadline(), None);
    assert!(graphics.is_idle());
    assert!(!graphics.wants_turn(HostNanos(u64::MAX)));
    let late = turn_at(&graphics, HostNanos(u64::MAX));
    assert_eq!((late.0, late.1), (vec![], vec![]));
    assert_eq!(graphics.shared().lock().vsync().tick(), 0);
}

/// A recorded tick is fed back to the clock, not the queue, and only as the next number; the next
/// turn delivers it as the run did.
#[test]
fn a_vsync_tick_note_is_fed_back_to_the_clock_in_order() {
    let graphics = graphics(FrontMode::Replay);
    graphics
        .shared()
        .lock()
        .listen_vsync(listener(), START)
        .unwrap();
    assert_eq!(graphics.replay(Note::VsyncTick { tick: 1 }), Ok(()));
    assert!(graphics.shared().queue().is_empty());
    let error = graphics.replay(Note::VsyncTick { tick: 1 }).unwrap_err();
    assert_eq!(
        error,
        ReplayError::Vsync(VsyncReplayError::OutOfOrder {
            expected: 2,
            found: 1
        })
    );
    assert_eq!(
        error.to_string(),
        "a recorded vsync tick could not be fed back: vsync tick 1 was fed back where tick 2 \
         comes next"
    );
    assert_eq!(graphics.shared().lock().vsync().replayed(), 1);
    let delivered = turn_at(&graphics, HostNanos(0));
    assert_eq!(
        (delivered.0, delivered.1),
        (vec![ticked(1)], vec![listener()])
    );
}

/// A tick fed back is work only in a replay. In a run only a restored snapshot can hold one,
/// and the live turn never delivers it, so it neither keeps the run busy nor wants a turn
/// (`starquake_gsv::shared::vsync`, "Not checked here"); the same snapshot restored into a
/// replay is work, and its turn delivers the tick.
#[test]
fn a_tick_fed_back_in_a_live_run_is_not_work() {
    let source = graphics(FrontMode::Replay);
    source.replay(Note::VsyncTick { tick: 1 }).unwrap();
    let bytes = source.shared().save_vsync().unwrap();

    let live = graphics(FrontMode::Live);
    live.shared().restore_vsync(&bytes).unwrap();
    assert_eq!(live.shared().lock().vsync().replayed(), 1);
    assert!(live.is_idle());
    assert!(!live.wants_turn(HostNanos(u64::MAX)));
    let late = turn_at(&live, HostNanos(u64::MAX));
    assert_eq!((late.0, late.1), (vec![], vec![]));
    assert_eq!(live.shared().lock().vsync().replayed(), 1);

    let replay = graphics(FrontMode::Replay);
    replay.shared().restore_vsync(&bytes).unwrap();
    assert!(!replay.is_idle());
    assert!(replay.wants_turn(HostNanos(0)));
    let delivered = turn_at(&replay, HostNanos(0));
    assert_eq!(delivered.0, [ticked(1)]);
    assert!(replay.is_idle());
}
