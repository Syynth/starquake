//! The vsync clock as a run installs the graphics server ([`install_with_graphics`]), behind the
//! real dispatcher: a listener's event is signalled by the timer thread's turn through the
//! kernel host, an event the kernel no longer holds is dropped, a restored listener is signalled
//! whoever made its event, the turn's deadline is the
//! earlier of the shared page's and the clock's, and a recorded tick replays to the same signal
//! and state.

use starquake_gsv::shared::vsync::INTERVAL_NS;

use super::*;

/// When the tests' listener starts the clock: not zero, so a deadline counted from zero fails.
const START: HostNanos = HostNanos(0x0123_4567);

/// The first tick's deadline.
const FIRST: HostNanos = HostNanos(START.0 + INTERVAL_NS);

/// An event the guest makes with `CreateEvent`: its writable and readable ends' handles.
fn guest_event(rig: &Rig) -> (Handle, Handle) {
    let regs = rig.call(SvcId::CreateEvent, 0, 0);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    let handle = |raw: u64| Handle::from_raw(u32::try_from(raw).unwrap());
    (handle(regs.x[1]), handle(regs.x[2]))
}

/// A listener whose event the guest holds is signalled by the turn at the deadline, not by one
/// a nanosecond before, and the tick's note is recorded on the timer thread's core and thread.
/// Once the guest has closed the event, the kernel refuses the next tick's signal: the listener
/// is dropped, the clock stops, and the graphics server is idle again.
#[test]
fn a_vsync_listener_is_ticked_through_the_run_s_kernel() {
    let graphics = Graphics::new(FrontMode::Live, None);
    let rig = rig_with(&graphics);
    let (writable, readable) = guest_event(&rig);
    let event = rig.object(writable);
    assert!(graphics.is_idle());
    assert_eq!(
        graphics.shared().lock().listen_vsync(event, START),
        Ok(true)
    );
    assert!(!graphics.is_idle());
    assert_eq!(graphics.deadline(), Some(FIRST));

    let before = rig.all_events().len();
    rig.turn_at(&graphics, HostNanos(FIRST.0 - 1));
    assert_eq!(poll(&rig, readable), KernelError::TimedOut.code());
    assert_eq!(notes(&rig.all_events()[before..]), []);

    rig.turn_at(&graphics, FIRST);
    assert_eq!(poll(&rig, readable), ResultCode::SUCCESS);
    let events = rig.all_events();
    assert_eq!(notes(&events[before..]), [Note::VsyncTick { tick: 1 }]);
    let tick = Note::VsyncTick { tick: 1 }.event(GSV_ROOT);
    assert!(events[before..].contains(&(TIMER.0, TIMER.1, tick)));
    let second = HostNanos(FIRST.0 + INTERVAL_NS);
    assert_eq!(graphics.deadline(), Some(second));

    // The guest closes both ends: the event is gone, and the next tick's signal is refused.
    for handle in [writable, readable] {
        let regs = rig.call(SvcId::CloseHandle, u64::from(handle.raw()), 0);
        assert_eq!(result(&regs), ResultCode::SUCCESS);
    }
    rig.turn_at(&graphics, second);
    assert_eq!(
        notes(&rig.all_events()[before..]),
        [Note::VsyncTick { tick: 1 }, Note::VsyncTick { tick: 2 }]
    );
    assert_eq!(graphics.shared().lock().vsync().listeners().count(), 0);
    assert_eq!(graphics.deadline(), None);
    assert!(graphics.is_idle());
}

/// A restored listener is signalled on the production host whoever made its event: a guest's
/// own `CreateEvent` event, named by its writable end or its readable end, is signalled on every
/// tick (`starquake_gsv::shared::vsync`, "Not checked here"); only an id the kernel does not hold
/// is refused and dropped.
#[test]
fn a_restored_listener_is_signalled_whoever_made_its_event() {
    let graphics = Graphics::new(FrontMode::Live, None);
    let rig = rig_with(&graphics);
    let (by_writable, writable_readable) = guest_event(&rig);
    let (_, by_readable) = guest_event(&rig);
    let gone = ObjectId::from_parts(0x0FFF, 9);
    // The snapshot is written by another graphics server, so nothing here listened first.
    let source = Graphics::new(FrontMode::Live, None);
    for listener in [rig.object(by_writable), rig.object(by_readable), gone] {
        assert_eq!(
            source.shared().lock().listen_vsync(listener, START),
            Ok(true)
        );
    }
    let bytes = source.shared().save_vsync().unwrap();
    graphics.shared().restore_vsync(&bytes).unwrap();

    for n in 1..=2 {
        assert_eq!(poll(&rig, writable_readable), KernelError::TimedOut.code());
        assert_eq!(poll(&rig, by_readable), KernelError::TimedOut.code());
        rig.turn_at(&graphics, HostNanos(START.0 + n * INTERVAL_NS));
        assert_eq!(poll(&rig, writable_readable), ResultCode::SUCCESS);
        assert_eq!(poll(&rig, by_readable), ResultCode::SUCCESS);
        rig.reset(writable_readable);
        rig.reset(by_readable);
        // Only the id the kernel does not hold was dropped, at the first tick.
        let listeners: Vec<ObjectId> = graphics.shared().lock().vsync().listeners().collect();
        assert_eq!(
            listeners,
            [rig.object(by_writable), rig.object(by_readable)]
        );
    }
}

/// The turn's deadline is the shared page's while only it waits, the clock's when it comes
/// first, and the page's again when the clock's comes later; and a held page request wants a
/// turn whatever the clock says.
#[test]
fn the_turn_s_deadline_is_the_earlier_of_the_page_s_and_the_clock_s() {
    let graphics = Graphics::new(FrontMode::Live, None);
    let rig = rig_with(&graphics);
    let (request, answer, _, _) = bound(&rig);
    let asked = SyncpointId::new(ASKED).unwrap();
    let request_at = WRITABLE_AT + u64::from(page::GROUP_REQUEST_OFFSET);
    let slot_at = WRITABLE_AT + u64::from(page::SLOT_REQUEST_OFFSET);
    let ask = |summary: u64, slot: u64| {
        rig.memory
            .write(request_at, &summary.to_le_bytes())
            .unwrap();
        rig.memory.write(slot_at, &slot.to_le_bytes()).unwrap();
        let regs = rig.call(SvcId::SignalEvent, u64::from(request.raw()), 0);
        assert_eq!(result(&regs), ResultCode::SUCCESS);
    };
    // A first request, answered at once (its counter moved); then the same counter asked again,
    // which the turn at `held` holds to the bound.
    graphics.shared().syncpoints().increment_by(asked, 1);
    ask(1, 1 << ASKED);
    rig.turn(&graphics);
    assert_eq!(poll(&rig, answer), ResultCode::SUCCESS);
    rig.reset(answer);
    ask(0, 0);
    let held = HostNanos(100 * INTERVAL_NS);
    rig.turn_at(&graphics, held);
    let page_due = HostNanos(held.0 + page::ANSWER_BOUND);
    assert_eq!(graphics.deadline(), Some(page_due));
    assert!(graphics.wants_turn(HostNanos(0)));

    // The clock's deadline a nanosecond before the page's.
    let early = HostNanos(page_due.0 - 1 - INTERVAL_NS);
    let listener = ObjectId::from_parts(1, 9);
    graphics
        .shared()
        .lock()
        .listen_vsync(listener, early)
        .unwrap();
    assert_eq!(graphics.deadline(), Some(HostNanos(page_due.0 - 1)));
    // A nanosecond after it.
    assert!(graphics.shared().lock().unlisten_vsync(listener));
    let late = HostNanos(page_due.0 + 1 - INTERVAL_NS);
    graphics
        .shared()
        .lock()
        .listen_vsync(listener, late)
        .unwrap();
    assert_eq!(graphics.deadline(), Some(page_due));
}

/// A run's vsync ticks, recorded, replay to the same events and state: the listener's signals,
/// the guest's waits that they end, the notes, gsv's state, and the clock's count and listeners
/// (not its deadline, which is host timing). The replay
/// feeds each recorded tick back where the run's turn made it, and turns at a time that would
/// tick nothing on a live clock.
#[test]
fn a_recorded_vsync_tick_replays_to_the_same_signal_and_state() {
    // Two ticks, each polled for and cleared by the guest. `tick` is the turn of tick `n`.
    let session = |rig: &Rig, graphics: &Graphics, tick: &mut dyn FnMut(u64)| {
        let (writable, readable) = guest_event(rig);
        let listen = graphics
            .shared()
            .lock()
            .listen_vsync(rig.object(writable), START);
        assert_eq!(listen, Ok(true));
        for n in 1..=2 {
            assert_eq!(poll(rig, readable), KernelError::TimedOut.code());
            tick(n);
            assert_eq!(poll(rig, readable), ResultCode::SUCCESS);
            rig.reset(readable);
        }
    };
    let run = Graphics::new(FrontMode::Live, None);
    let recorded = rig_with(&run);
    let mut drained: Vec<Vec<Note>> = Vec::new();
    session(&recorded, &run, &mut |n| {
        let before = recorded.all_events().len();
        let deadline = HostNanos(START.0 + n * INTERVAL_NS);
        recorded.turn_at(&run, deadline);
        drained.push(notes(&recorded.all_events()[before..]));
    });
    assert_eq!(
        drained,
        [
            vec![Note::VsyncTick { tick: 1 }],
            vec![Note::VsyncTick { tick: 2 }]
        ]
    );

    let replay = Graphics::new(FrontMode::Replay, None);
    let replayed = rig_with(&replay);
    let mut fed = drained.iter();
    session(&replayed, &replay, &mut |_| {
        for &note in fed.next().unwrap() {
            replay.replay(note).unwrap();
        }
        replayed.turn_at(&replay, HostNanos(0));
    });
    assert_eq!(replayed.all_events(), recorded.all_events());
    assert_eq!(state_of(&replayed, &replay), state_of(&recorded, &run));
    // The clock's count and listeners replay; its deadline is host timing, which no note keeps
    // (`crate::hle::graphics`, "Replay").
    let clock = |graphics: &Graphics| {
        let state = graphics.shared().lock();
        let vsync = state.vsync();
        (vsync.tick(), vsync.listeners().collect::<Vec<_>>())
    };
    assert_eq!(clock(&replay), clock(&run));
    assert_eq!(clock(&replay).0, 2);
}
