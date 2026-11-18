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
fn install_listening(started: Started) -> impl Fn(&Dispatcher, Given) -> Result<(), String> + Sync {
    move |dispatcher: &Dispatcher, given: Given| {
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
}

/// The real run's guest: connects to the test server, asks it for a listener's event, then
/// [`ROUNDS`] times waits on the event (at most [`TICK_WAIT`]) and clears it; then exits.
fn ticked_guest(
    tls: &Arc<FlatMemory>,
    base: u64,
    name: u64,
    seen: &Rc<RefCell<Ticks>>,
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
        guest = guest.then(
            move |regs| {
                let mut seen = woke.borrow_mut();
                seen.wakes.push((regs.x[0], Instant::now()));
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

/// The `VsyncTick` notes of a run's flight recorder, in order: each one's time on the
/// environment's clock, in nanoseconds (the event's guest time at the run's counter rate,
/// rounded down by less than one counter tick).
fn tick_times(report: &RunReport) -> Vec<u64> {
    let hz = u128::from(timebase().rate().hz());
    let dump = Dump::from_bytes(&report.events).unwrap();
    dump.merged()
        .into_iter()
        .filter(|event| {
            matches!(
                event.kind,
                EventKind::ServiceNote { service, code, .. }
                    if service == crate::hle::GSV_ROOT && code == VSYNC_TICK
            )
        })
        .map(|event| u64::try_from(u128::from(event.time.0) * 1_000_000_000 / hz).unwrap())
        .collect()
}

/// In a real run the timer thread ticks the clock a listener started, on the real `Env` clock,
/// and each tick's signal ends the guest's wait on the listener's event: every wait ends with
/// success, never sooner after the request than its tick's deadline can come (the `j`-th wake
/// at least `j` intervals after the request, since each deadline is at least one interval after
/// the one before), and the run records a `VsyncTick` for every wake. Each recorded tick comes
/// at or after its deadline (the deadlines followed from the start by NX2's rule, as the clock
/// does), within one counter tick of rounding, and, by the median, less than half of
/// [`PREEMPTION_INTERVAL_NS`] after it.
///
/// The median is what fails if the timer thread did not sleep to the clock's deadline
/// (`Graphics::deadline` without it): a tick then waits for the next preemption turn, up to
/// [`PREEMPTION_INTERVAL_NS`] later (and the host's sleeps overshoot), so the median comes 5.6
/// to 9.7 ms late, against 1 to 2.5 ms with it (seen over 10 and 30 runs of [`ROUNDS`] ticks on
/// a macOS host). It times the timer thread's own tick, read from the recorded note, and not
/// the guest's wake, which also waits for a second host thread (the core's) and was seen over
/// 5 ms late, by the median, on a loaded host.
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
    let mut main = ticked_guest(&tls, guest.tls, name, &seen);
    let install = install_listening(Arc::clone(&started));
    let options = RunOptions {
        timeout: Some(RUN_LIMIT),
        command_line: Vec::new(),
        sd_card: None,
        romfs: None,
        rtc: RtcAtBoot::eden_default(),
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

    let ticks = tick_times(&report);
    assert!(ticks.len() >= ROUNDS, "{ticks:?}");
    let rounding = 1_000_000_000_u64.div_ceil(timebase().rate().hz());
    let mut deadline = lock(&started).unwrap().0 + INTERVAL_NS;
    let mut late = Vec::new();
    for &tick in &ticks {
        assert!(
            tick + rounding >= deadline,
            "a tick at {tick} for {deadline}"
        );
        late.push(Duration::from_nanos(tick.saturating_sub(deadline)));
        // NX2 `PaceToVblank`, as the clock follows it.
        let from = if tick > deadline + INTERVAL_NS {
            tick
        } else {
            deadline
        };
        deadline = from + INTERVAL_NS;
    }
    late.sort();
    let median = late[late.len() / 2];
    let preemption = Duration::from_nanos(PREEMPTION_INTERVAL_NS);
    assert!(median < preemption / 2, "lateness {late:?}");
}
