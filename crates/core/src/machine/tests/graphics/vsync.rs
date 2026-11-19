//! The vsync clock in a real run's timer thread (`machine/timer.rs`, "The graphics server's
//! turn"): on a machine whose cores never step, a clock waiting for its deadline adds no reading
//! of the environment and a due one ticks; and in a real run, with the real timer thread and the
//! real `Env` clock, a listener's event is ticked at 60 Hz and wakes the guest waiting on it.
//!
//! There is no vi2 yet (slice 13) to add a listener for a guest, so the real run's guest asks a
//! small test server on its own port, `vsync:`, which does what vi2's handlers will: it makes an
//! event, adds it as a listener with the time it reads from the environment, and copies its
//! handle to the guest. Its request and reply are CMIF messages with no data, laid out by
//! `starquake-ipc`'s builders (the server is the test's own; no protocol fact is involved).

use std::collections::BTreeMap;

use starquake_gsv::note::VSYNC_TICK;
use starquake_gsv::shared::vsync::INTERVAL_NS;
use starquake_ipc::ReplyBuilder;
use starquake_kernel::id::{ObjectId, SlabId};
use starquake_kernel::ipc::{Handled, HandlerError, IpcContext, PortName, SessionHandler};
use starquake_trace::ServiceId;

use super::*;

/// How many ticks the real run's guest waits for.
const ROUNDS: usize = 24;

/// How long each of its waits may take at most: far more than a tick.
const TICK_WAIT: Duration = Duration::from_secs(2);

/// A listener the machine's kernel does not hold.
fn made_up() -> ObjectId {
    ObjectId::from_parts(0x3F00, 9)
}

/// On a machine with no thread running, a vsync clock whose deadline the loop's copy of the clock
/// has not reached makes the timer thread's graphics turn read nothing (the environment's
/// clock read just before and just after it are one step apart); once both clocks have reached
/// it, the turn reads the clock and ticks. The listener is an event the kernel does not hold, so
/// its signal is refused and the clock stops after the tick.
#[test]
fn a_waiting_vsync_clock_reads_nothing_and_a_due_one_ticks() {
    on_a_machine(|machine| {
        let reading = || machine.env.now().unwrap();
        let start = reading();
        let listen = machine
            .graphics
            .shared()
            .lock()
            .listen_vsync(made_up(), start);
        assert_eq!(listen, Ok(true));
        assert!(!machine.graphics.is_idle());
        let deadline = machine.graphics.deadline().unwrap();
        assert_eq!(deadline, HostNanos(start.0 + INTERVAL_NS));
        // The timer thread needs no ring for a start (`machine/timer.rs`): the first deadline is
        // further out than the longest sleep that can have begun before it.
        const { assert!(INTERVAL_NS > PREEMPTION_INTERVAL_NS) };

        let before = reading().0;
        assert!(timer::graphics(machine).is_ok());
        assert_eq!(reading().0 - before, CLOCK_STEP);
        assert_eq!(machine.graphics.shared().lock().vsync().tick(), 0);

        // Both copies of the clock brought to the deadline, a step at a time.
        while machine.session.clock.host_now() < deadline {}
        while reading() < deadline {}
        let before = reading().0;
        assert!(timer::graphics(machine).is_ok());
        assert!(reading().0 - before > CLOCK_STEP);
        assert_eq!(machine.graphics.shared().lock().vsync().tick(), 1);
        assert_eq!(machine.graphics.deadline(), None);
        assert!(machine.graphics.is_idle());
    });
}

/// The test server's port and number: not one a real run's services use.
const VSYNC_PORT: &str = "vsync:";
const VSYNC: ServiceId = ServiceId(0x7E57);

/// The environment's time the test server started the clock from.
type Started = Arc<Mutex<Option<HostNanos>>>;

/// The run's graphics server, for the guest to read the vsync clock from as it wakes.
type Watched = Arc<Mutex<Option<Arc<Graphics>>>>;

/// The test server (module docs).
struct Listening {
    graphics: Arc<Graphics>,
    env: Arc<dyn Env>,
    started: Started,
}

impl SessionHandler for Listening {
    fn accept(&mut self, _: ObjectId) -> Result<(), HandlerError> {
        Ok(())
    }

    fn closed(&mut self, _: ObjectId) {}

    fn handle(
        &mut self,
        cx: &mut IpcContext<'_>,
        _: ObjectId,
        message: &mut [u8],
    ) -> Result<Handled, HandlerError> {
        let failed = |what: String| HandlerError(what);
        let event = cx.create_event().map_err(|e| failed(e.to_string()))?;
        let now = self.env.now().map_err(|e| failed(e.to_string()))?;
        *lock(&self.started) = Some(now);
        self.graphics
            .shared()
            .lock()
            .listen_vsync(event, now)
            .map_err(|e| failed(e.to_string()))?;
        let handle = cx.insert_handle(event).map_err(|e| failed(e.to_string()))?;
        ReplyBuilder::cmif(0)
            .copy_handle(handle.raw())
            .build(message)
            .map_err(|e| failed(e.to_string()))?;
        Ok(Handled::Replied)
    }

    fn take_reply(
        &mut self,
        _: &mut IpcContext<'_>,
        _: &mut [u8],
    ) -> Option<(ObjectId, Result<(), HandlerError>)> {
        None
    }
}

/// A real run's services, and the test server beside them, over the run's graphics server.
fn install_listening(
    started: Started,
    watched: Watched,
) -> impl Fn(&Dispatcher, Given) -> Result<(), String> + Sync {
    move |dispatcher: &Dispatcher, given: Given| {
        *lock(&watched) = Some(Arc::clone(&given.graphics));
        let handler = Listening {
            graphics: Arc::clone(&given.graphics),
            env: Arc::clone(&given.env),
            started: Arc::clone(&started),
        };
        real_services(dispatcher, given)?;
        let port = PortName::new(VSYNC_PORT).map_err(|error| error.to_string())?;
        dispatcher
            .add_server(VSYNC_PORT, VSYNC, Some(port), Box::new(handler))
            .map_err(|error| error.to_string())?;
        Ok(())
    }
}

/// What the real run's guest saw.
#[derive(Default)]
struct Ticks {
    /// The event's handle the test server copied.
    event: u32,
    /// When the guest sent its request.
    asked: Option<Instant>,
    /// Each wait's result and when it ended.
    wakes: Vec<(u64, Instant)>,
    /// At each wake, the vsync clock's tick count and its next deadline, read together.
    clocks: Vec<(u64, Option<HostNanos>)>,
}

/// The real run's guest: connects to the test server, asks it for a listener's event, then
/// [`ROUNDS`] times waits on the event (at most [`TICK_WAIT`]) and clears it; then exits.
fn ticked_guest(
    tls: &Arc<FlatMemory>,
    base: u64,
    name: u64,
    seen: &Rc<RefCell<Ticks>>,
    watched: &Watched,
) -> ScriptedVcpu {
    let handles = base + HANDLES_AT;
    let request = Arc::clone(tls);
    let asking = Rc::clone(seen);
    let reply = Arc::clone(tls);
    let given = Rc::clone(seen);
    let wait = Arc::clone(tls);
    let mut guest = ScriptedVcpu::new()
        .then(move |regs| regs.x[1] = name, svc(SvcId::ConnectToNamedPort))
        .then(
            move |regs| {
                assert_eq!(regs.x[0], 0, "ConnectToNamedPort failed");
                let mut buffer = [0; BUFFER_SIZE];
                RequestBuilder::cmif(CommandType::Request, 0)
                    .layout(RequestLayout::Libnx)
                    .build(&mut buffer)
                    .unwrap();
                request.write(base, &buffer).unwrap();
                asking.borrow_mut().asked = Some(Instant::now());
                regs.x[0] = regs.x[1];
            },
            svc(SvcId::SendSyncRequest),
        )
        .then(
            move |regs| {
                assert_eq!(regs.x[0], 0, "the test server's request failed");
                let mut buffer = [0; BUFFER_SIZE];
                reply.read(base, &mut buffer).unwrap();
                let message = Message::parse_reply_sized(&buffer, SessionKind::Session, 0).unwrap();
                let event = message.copy_handles[0];
                given.borrow_mut().event = event;
                wait.write(handles, &event.to_le_bytes()).unwrap();
                regs.x[1] = handles;
                regs.x[2] = 1;
                regs.x[3] = u64::try_from(TICK_WAIT.as_nanos()).unwrap();
            },
            svc(SvcId::WaitSynchronization),
        );
    for round in 0..ROUNDS {
        let woke = Rc::clone(seen);
        let clock = Arc::clone(watched);
        guest = guest.then(
            move |regs| {
                let mut seen = woke.borrow_mut();
                seen.wakes.push((regs.x[0], Instant::now()));
                if let Some(graphics) = &*lock(&clock) {
                    let shared = graphics.shared().lock();
                    let vsync = shared.vsync();
                    seen.clocks.push((vsync.tick(), vsync.deadline()));
                }
                regs.x[0] = u64::from(seen.event);
            },
            svc(SvcId::ResetSignal),
        );
        if round + 1 < ROUNDS {
            guest = guest.then(
                move |regs| {
                    assert_eq!(regs.x[0], 0, "clearing the event failed");
                    regs.x[1] = handles;
                    regs.x[2] = 1;
                    regs.x[3] = u64::try_from(TICK_WAIT.as_nanos()).unwrap();
                },
                svc(SvcId::WaitSynchronization),
            );
        }
    }
    guest.then(
        |regs| assert_eq!(regs.x[0], 0, "clearing the event failed"),
        svc(SvcId::ExitProcess),
    )
}

/// In a real run the timer thread ticks the clock a listener started, on the real `Env` clock,
/// and each tick's signal ends the guest's wait on the listener's event: every wait ends with
/// success, never sooner after the request than its tick's deadline can come (the `j`-th wake
/// at least `j` intervals after the request, since each deadline is at least one interval after
/// the one before), and the run records a `VsyncTick` for every wake. Each recorded tick comes
/// at or after its deadline, within one counter tick of rounding, and, by the median, less than
/// half of [`PREEMPTION_INTERVAL_NS`] after it.
///
/// Each tick's deadline is read, not rebuilt: the first is one interval after the start, and as
/// the guest wakes from tick `n` it reads the clock's tick count and next deadline together
/// (through the run's graphics server, which the test server hands over), which is tick
/// `n + 1`'s. Rebuilding them by NX2's rule from the notes cannot be exact: a late turn restarts
/// the clock from its own time, which no note records, and following the note's time instead
/// once failed on CI (the next tick came 2.9 µs before the deadline so computed), while bounding
/// the turn's time by the note's left every tick after a restart looking on time. Where the
/// guest woke only after the next tick, so read no deadline for it, the bounds stand in: the
/// earliest the deadline can be, for the never-early check, and the latest, for the lateness.
/// Lateness is from the note's time plus one counter tick, so never less than the true one.
///
/// The median is what fails if the timer thread did not sleep to the clock's deadline
/// (`Graphics::deadline` without it): a tick then waits for the next preemption turn, up to
/// [`PREEMPTION_INTERVAL_NS`] later, so the median comes 4.3 to 5.9 ms late (19 of 20 runs of
/// [`ROUNDS`] ticks on an Apple Silicon Mac at or over the bound), against 0.03 to 0.06 ms with
/// it (30 runs). It times the timer thread's own tick, read from the recorded note, and not the
/// guest's wake, which also waits for a second host thread (the core's) and was seen over 5 ms
/// late, by the median, on a loaded host.
///
/// It is also what fails where the host ends the timer thread's timed waits late on purpose
/// (macOS timer coalescing; `machine/timer.rs`, "Sleeping to a deadline") and the thread asks
/// for each sleep in one wait, as it did before it measured that lateness: CI's hosted macOS
/// runners then saw medians of 14 to 44 ms and the latest tick 33 to 69 ms late (12 runs), and
/// the Mac under `taskpolicy -c utility` failed 20 runs of 20; with the waits sized by the
/// measured lateness it passed 30 of 30 there, the worst median 0.19 ms.
#[test]
fn the_timer_thread_ticks_the_vsync_clock_and_wakes_a_listener_s_waiter() {
    let guest = Guest::new(&programs::hello().unwrap());
    let name = guest.start_of("main.rodata");
    guest
        .memory
        .write(name, format!("{VSYNC_PORT}\0").as_bytes())
        .unwrap();
    let page = usize::try_from(crate::image::PAGE).unwrap();
    let tls = Arc::new(FlatMemory::new(guest.tls, page));
    let space = WithTls {
        image: &guest.memory,
        tls: Arc::clone(&tls),
    };
    let seen = Rc::new(RefCell::new(Ticks::default()));
    let started: Started = Arc::default();
    let watched: Watched = Arc::default();
    let mut main = ticked_guest(&tls, guest.tls, name, &seen, &watched);
    let install = install_listening(Arc::clone(&started), Arc::clone(&watched));
    let options = RunOptions {
        timeout: Some(RUN_LIMIT),
        command_line: Vec::new(),
        sd_card: None,
        romfs: None,
        rtc: RtcAtBoot::eden_default(),
        environment: crate::EnvSource::Live,
        until_frames: None,
    };
    let report = run_serving(
        vcpus(&mut main),
        guest.program(&space),
        &options,
        (Instant::now(), timebase()),
        Limits::DEFAULT,
        &install,
    );
    drop(main);
    assert_eq!(report.exit, ExitReason::ProcessExited, "{}", report.trace);
    let seen = seen.borrow();
    let asked = seen.asked.unwrap();
    assert_eq!(seen.wakes.len(), ROUNDS);
    let interval = Duration::from_nanos(INTERVAL_NS);
    for (round, &(result, woke)) in seen.wakes.iter().enumerate() {
        assert_eq!(result, 0, "wait {round} ended with {result:#x}");
        let ticks = u32::try_from(round + 1).unwrap();
        assert!(
            woke.duration_since(asked) >= interval * ticks,
            "wake {round} came {:?} after the request",
            woke.duration_since(asked)
        );
    }

    // The deadline of each tick, by number, where the guest read it: the first from the start,
    // and tick `n + 1`'s from the clock as tick `n` left it.
    let mut deadlines = BTreeMap::new();
    deadlines.insert(1, lock(&started).unwrap().0 + INTERVAL_NS);
    for &(tick, deadline) in &seen.clocks {
        if let Some(deadline) = deadline {
            deadlines.insert(tick + 1, deadline.0);
        }
    }
    let ticks = gsv_notes(&timed_events(&report), VSYNC_TICK);
    assert!(ticks.len() >= ROUNDS, "{ticks:?}");
    let rounding = counter_tick();
    let (mut earliest, mut latest) = (0, 0);
    let mut late = Vec::new();
    for (number, &(tick, at)) in (1..).zip(&ticks) {
        assert_eq!(tick, number, "{ticks:?}");
        if let Some(&deadline) = deadlines.get(&tick) {
            (earliest, latest) = (deadline, deadline);
        }
        assert!(
            at + rounding >= earliest,
            "tick {tick} at {at} for a deadline no earlier than {earliest}"
        );
        late.push(Duration::from_nanos((at + rounding).saturating_sub(latest)));
        // A deadline the guest did not read (it woke after the next tick): NX2 `PaceToVblank`,
        // as the clock follows it from the turn's own time, which is no earlier than the
        // deadline and no later than the note's time. A turn can have restarted the clock from
        // its time only if the note came more than an interval after the earliest deadline.
        latest = if at > earliest + INTERVAL_NS {
            latest.max(at)
        } else {
            latest
        } + INTERVAL_NS;
        earliest += INTERVAL_NS;
    }
    late.sort();
    let median = late[late.len() / 2];
    let preemption = Duration::from_nanos(PREEMPTION_INTERVAL_NS);
    assert!(median < preemption / 2, "lateness {late:?}");
}
