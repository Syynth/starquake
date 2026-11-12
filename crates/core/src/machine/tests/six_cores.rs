//! Six cores driven by scripted vCPUs (kernel spec, "Port plan", slice 10's test column).
//!
//! Most tests here are deterministic: every core steps in turn on the test's own thread
//! ([`Stepper`]), with the kernel's clock a [`FixedClock`] and the timer thread's work done by
//! the test between rounds, so each script knows exactly which guest thread its core runs at
//! each exit. Each core's vCPU plays one script whichever thread it runs: a step's closure
//! asserts which thread it is (by the argument a thread starts with in `x0`, and by a mark it
//! keeps in `x19`, which a switch must save and restore). A core whose script runs out fails
//! the run, so a thread on a core it may not use, or on two cores, cannot pass unnoticed.
//!
//! The last tests run the real threads ([`run`]): stopping the run stops every core, and a run
//! whose every thread waits on nothing that could end ends as an emulator error.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::mpsc;

use starquake_cpu::GuestRegs;
use starquake_kernel::id::ObjectId;
use starquake_kernel::ipc::{Handled, HandlerError, IpcContext, PortName, SessionHandler};
use starquake_kernel::sched::PREEMPTION_PRIORITIES;
use starquake_trace::{ServiceId, VecSink};

use super::*;
use crate::machine::core_loop::{Core, Stepped};

/// Where the scripted threads "start": nothing fetches from it.
const ENTRY: u64 = RAM_BASE + 0x100;
/// Their stack: nothing touches it.
const STACK: u64 = RAM_BASE + 0x1_0000;
/// The main thread's priority, which the process's description gives it.
fn main_priority() -> i32 {
    i32::from(ProcessDescription::homebrew(NRO_CONSOLE).main_thread_priority)
}

/// The kernel's clock in these tests: every reading a microsecond after the last, so deadlines
/// fall at known readings.
const CLOCK_STEP: u64 = 1_000;

fn fixed_timebase() -> Timebase<FixedClock> {
    Timebase {
        console: NRO_CONSOLE,
        clock: FixedClock::new(CLOCK_STEP, CLOCK_STEP),
    }
}

/// A step that creates a thread at `entry` with `argument`, `priority` and virtual `core`
/// (`CreateThread`'s registers, Eden's `SvcWrap_CreateThread64`: x1-x5).
fn create(argument: u64, priority: i32, core: i32) -> impl FnOnce(&mut GuestRegs) + 'static {
    move |regs| {
        regs.x[1] = ENTRY;
        regs.x[2] = argument;
        regs.x[3] = STACK;
        regs.x[4] = u64::from(priority.cast_unsigned());
        regs.x[5] = u64::from(core.cast_unsigned());
    }
}

/// A step after `CreateThread`: it succeeded, and the handle it gave (x1) is passed on in x0.
fn start_created(regs: &mut GuestRegs) {
    assert_eq!(regs.x[0], 0, "CreateThread failed");
    regs.x[0] = regs.x[1];
}

/// A step after a call that must have succeeded.
fn succeeded(regs: &mut GuestRegs) {
    assert_eq!(regs.x[0], 0, "the call before failed");
}

/// A thread's first instruction: it is the thread started with `argument`, which it keeps in
/// `x19` from here on. Logs it in `order`.
fn begins(argument: u64, order: &Rc<RefCell<Vec<u64>>>) -> impl FnOnce(&mut GuestRegs) + 'static {
    let order = Rc::clone(order);
    move |regs| {
        assert_eq!(
            (regs.x[0], regs.pc),
            (argument, ENTRY),
            "thread {argument:#x}"
        );
        regs.x[19] = argument;
        order.borrow_mut().push(argument);
    }
}

/// A later instruction of the thread that keeps `mark` in `x19`.
fn still(mark: u64) -> impl FnOnce(&mut GuestRegs) + 'static {
    move |regs| assert_eq!(regs.x[19], mark, "the thread that runs is not {mark:#x}")
}

/// A later instruction of thread `mark` that checks `GetCurrentProcessorNumber` said `core`.
fn on_core(mark: u64, core: u8) -> impl FnOnce(&mut GuestRegs) + 'static {
    move |regs| {
        assert_eq!(regs.x[19], mark, "the thread that runs is not {mark:#x}");
        assert_eq!(
            regs.x[0],
            u64::from(core),
            "thread {mark:#x} is on another core"
        );
    }
}

/// Six cores of one machine, stepped in turn on the test's thread.
struct Stepper<'m, 'a> {
    machine: &'m Machine<'a, ScriptedVcpu, FixedClock>,
    cores: Vec<Core<'m, 'a, ScriptedVcpu, FixedClock>>,
    vcpus: Vec<ScriptedVcpu>,
    parked: Vec<bool>,
}

impl<'m, 'a> Stepper<'m, 'a> {
    /// The machine's cores, core `n` with `scripts[n]` (a core with no script gets an empty
    /// one).
    fn new(machine: &'m Machine<'a, ScriptedVcpu, FixedClock>, scripts: Vec<ScriptedVcpu>) -> Self {
        let count = machine.cores.count();
        let mut scripts = VecDeque::from(scripts);
        let mut cores = Vec::new();
        let mut vcpus = Vec::new();
        for index in 0..count {
            let core = CoreId::new(index).unwrap();
            let vcpu = scripts.pop_front().unwrap_or_default();
            machine.cores.attach(core, vcpu.interrupter());
            cores.push(Core::new(machine, core));
            vcpus.push(vcpu);
        }
        Self {
            machine,
            cores,
            vcpus,
            parked: vec![false; usize::from(count)],
        }
    }

    /// Rounds of one step per core that is not parked, core 0 first, with `between` run
    /// before each round, until the run stops; at most `rounds` of them. A parked core steps
    /// again once it was woken. Returns how many rounds ran.
    fn run(
        &mut self,
        rounds: usize,
        mut between: impl FnMut(usize, &Machine<'a, ScriptedVcpu, FixedClock>),
    ) -> usize {
        let machine = self.machine;
        for round in 0..rounds {
            between(round, machine);
            for (index, ((core, vcpu), parked)) in self
                .cores
                .iter_mut()
                .zip(&mut self.vcpus)
                .zip(&mut self.parked)
                .enumerate()
            {
                if machine.is_stopped() {
                    return round;
                }
                let id = CoreId::new(u8::try_from(index).unwrap()).unwrap();
                if *parked && !machine.cores.take_wake(id) {
                    continue;
                }
                match core.step(vcpu) {
                    Ok(Stepped::Running) => *parked = false,
                    Ok(Stepped::Parked) => *parked = true,
                    Err(ending) => machine.stop(ending),
                }
            }
            if machine.is_stopped() {
                return round;
            }
            if self.parked.iter().all(|parked| *parked) && machine.dispatcher.is_stalled() {
                machine.stop(Ending::Failed(WAITING.to_owned()));
                return round;
            }
        }
        rounds
    }
}

/// The timer thread's turn, done by a test between rounds: the kernel's timeouts that are due
/// at the next reading of its clock end, and the cores whose thread changed are told.
fn fire_timers(machine: &Machine<'_, ScriptedVcpu, FixedClock>) {
    machine
        .dispatcher
        .fire_timers(&**machine.env, &machine.session.recorder)
        .unwrap();
    machine
        .dispatcher
        .reschedule(&mut machine.signals())
        .unwrap();
}

/// What a stepped run left: how it ended, its events, and the scripts' leftovers.
struct Ran {
    ending: Ending,
    events: Vec<Event>,
    left: Vec<usize>,
}

/// Runs `guest` on six stepped cores with `scripts`, the services `install` puts up and
/// `space` as its memory, calling `between` before each round.
fn stepped(
    guest: &Guest,
    space: &dyn GuestSpace<ScriptedVcpu>,
    scripts: Vec<ScriptedVcpu>,
    install: &Install,
    between: impl FnMut(usize, &Machine<'_, ScriptedVcpu, FixedClock>),
) -> Ran {
    let options = RunOptions::default();
    let session = Session {
        memory: space,
        options: &options,
        clock: RunClock::new(fixed_timebase()),
        recorder: Recorder::new(),
    };
    let sink = Arc::new(VecSink::new());
    session.recorder.add_sink(Arc::clone(&sink));
    let env = session.environment();
    let (dispatcher, _) = boot(guest.address_space(), &guest.placed, space).unwrap();
    let Ok(machine) = Machine::start(dispatcher, space, &session, &env, Limits::DEFAULT, install)
    else {
        unreachable!("the services could not be installed");
    };
    let mut stepper = Stepper::new(&machine, scripts);
    let rounds = stepper.run(200, between);
    assert!(rounds < 200, "the run did not end");
    let left = stepper.vcpus.iter().map(ScriptedVcpu::remaining).collect();
    drop(stepper);
    let (ending, _, _) = machine.finish();
    Ran {
        ending,
        events: sink.events(),
        left,
    }
}

/// [`stepped`] in the guest's own flat memory, with a real run's services and no timer.
fn stepped_simply(guest: &Guest, scripts: Vec<ScriptedVcpu>) -> Ran {
    stepped(guest, &guest.memory, scripts, &real_services, |_, _| {})
}

fn assert_exited(outcome: &Ran) {
    match &outcome.ending {
        Ending::Exited => {}
        Ending::Failed(message) => assert_eq!(message, "", "the run failed"),
        Ending::Crashed(exception, ..) => assert_eq!(format!("{exception:?}"), "", "crashed"),
        Ending::TimedOut => assert_eq!("timed out", "", "the run timed out"),
    }
    assert!(
        outcome.left.iter().all(|left| *left == 0),
        "unplayed steps: {:?}",
        outcome.left
    );
}

#[test]
fn the_most_urgent_thread_of_a_core_runs_first_and_only_on_the_cores_it_may_use() {
    let guest = Guest::new(&programs::hello().unwrap());
    let order = Rc::new(RefCell::new(Vec::new()));
    // The main thread, on core 0, starts three threads that may use only core 1: the first at
    // priority 30, then 50, then 40. Then it ends.
    let main = ScriptedVcpu::new()
        .then(create(1, 30, 1), svc(SvcId::CreateThread))
        .then(start_created, svc(SvcId::StartThread))
        .then(
            |regs| {
                succeeded(regs);
                create(2, 50, 1)(regs);
            },
            svc(SvcId::CreateThread),
        )
        .then(start_created, svc(SvcId::StartThread))
        .then(
            |regs| {
                succeeded(regs);
                create(3, 40, 1)(regs);
            },
            svc(SvcId::CreateThread),
        )
        .then(start_created, svc(SvcId::StartThread))
        .then(succeeded, svc(SvcId::ExitThread));
    // Core 1: the first thread runs while the others are started, and asks which core it is
    // on; then the more urgent of the other two, then the last.
    let mut core1 =
        ScriptedVcpu::new().then(begins(1, &order), svc(SvcId::GetCurrentProcessorNumber));
    for _ in 0..8 {
        core1 = core1.then(on_core(1, 1), svc(SvcId::GetCurrentProcessorNumber));
    }
    let core1 = core1
        .then(on_core(1, 1), svc(SvcId::ExitThread))
        .then(begins(3, &order), svc(SvcId::ExitThread))
        .then(begins(2, &order), svc(SvcId::ExitThread));
    let outcome = stepped_simply(&guest, vec![main, core1]);
    assert_exited(&outcome);
    assert_eq!(*order.borrow(), [1, 3, 2]);
    // The last thread to end ended the process; every call of the three threads was made on
    // core 1, and every call of the main thread on core 0.
    for event in &outcome.events {
        if let EventKind::Svc { .. } = event.kind {
            let expected = u8::from(event.thread != MAIN_THREAD);
            assert_eq!(event.core, expected, "{event:?}");
        }
    }
}

#[test]
fn a_thread_woken_by_its_timer_preempts_a_less_urgent_one_on_its_core() {
    let guest = Guest::new(&programs::hello().unwrap());
    let order = Rc::new(RefCell::new(Vec::new()));
    let sleep_ns = CLOCK_STEP;
    // Core 0 runs both threads: the main thread starts one more urgent than itself on its own
    // core, which runs at once and sleeps; the main thread runs again until the sleep ends;
    // the woken thread then runs before it, and ends; then the main thread ends the process.
    let core0 = ScriptedVcpu::new()
        .then(create(0xA, 30, 0), svc(SvcId::CreateThread))
        .then(start_created, svc(SvcId::StartThread))
        .then(
            {
                let begin = begins(0xA, &order);
                move |regs: &mut GuestRegs| {
                    begin(regs);
                    regs.x[0] = sleep_ns;
                }
            },
            svc(SvcId::SleepThread),
        )
        .then(
            |regs: &mut GuestRegs| {
                succeeded(regs);
                regs.x[19] = 0x1;
            },
            svc(SvcId::GetCurrentProcessorNumber),
        )
        .then(on_core(0x1, 0), svc(SvcId::GetCurrentProcessorNumber))
        .then(still(0xA), svc(SvcId::ExitThread))
        .then(still(0x1), svc(SvcId::ExitProcess));
    let mut fired = 0;
    let outcome = stepped(
        &guest,
        &guest.memory,
        vec![core0],
        &real_services,
        |round, machine| {
            // The timer thread's turn, once the main thread has run twice since the sleep began.
            if round == 5 {
                fire_timers(machine);
                fired += 1;
            }
        },
    );
    assert_exited(&outcome);
    assert_eq!(fired, 1);
    // One sleep, recorded once, with success.
    let sleeps: Vec<_> = outcome
        .events
        .iter()
        .filter(|event| {
            matches!(event.kind, EventKind::Svc { id, .. } if id == u32::from(SvcId::SleepThread.number()))
        })
        .collect();
    assert_eq!(sleeps.len(), 1);
    assert!(matches!(
        sleeps[0].kind,
        EventKind::Svc {
            result: starquake_trace::ResultCode(0),
            ..
        }
    ));
}

#[test]
fn time_slicing_takes_turns_between_threads_of_one_level_and_keeps_their_registers() {
    let guest = Guest::new(&programs::hello().unwrap());
    let order = Rc::new(RefCell::new(Vec::new()));
    let level = i32::from(PREEMPTION_PRIORITIES[1].get());
    assert!(
        level > main_priority(),
        "the main thread must not be sliced away"
    );
    let main = ScriptedVcpu::new()
        .then(create(0xA, level, 1), svc(SvcId::CreateThread))
        .then(start_created, svc(SvcId::StartThread))
        .then(
            move |regs| {
                succeeded(regs);
                create(0xB, level, 1)(regs);
            },
            svc(SvcId::CreateThread),
        )
        .then(start_created, svc(SvcId::StartThread))
        .then(succeeded, svc(SvcId::ExitThread));
    // Core 1: A runs until the first rotation, B until the second, then A again, with the
    // registers it had, and each ends.
    let core1 = ScriptedVcpu::new()
        .then(begins(0xA, &order), svc(SvcId::GetCurrentProcessorNumber))
        .then(on_core(0xA, 1), svc(SvcId::GetCurrentProcessorNumber))
        .then(on_core(0xA, 1), svc(SvcId::GetCurrentProcessorNumber))
        .then(begins(0xB, &order), svc(SvcId::GetCurrentProcessorNumber))
        .then(on_core(0xB, 1), svc(SvcId::GetCurrentProcessorNumber))
        .then(on_core(0xA, 1), svc(SvcId::ExitThread))
        .then(on_core(0xB, 1), svc(SvcId::ExitThread));
    let outcome = stepped(
        &guest,
        &guest.memory,
        vec![main, core1],
        &real_services,
        |round, machine| {
            if round == 5 || round == 8 {
                machine.dispatcher.preempt(&mut machine.signals()).unwrap();
            }
        },
    );
    assert_exited(&outcome);
    assert_eq!(*order.borrow(), [0xA, 0xB]);
}

/// A handler that defers every request and answers it when the timer thread delivers: what a
/// service with a host worker does, without the worker.
struct Deferring {
    ready: Ready,
    server: Arc<Mutex<Option<ServerId>>>,
    waiting: VecDeque<ObjectId>,
    fill: u8,
}

impl SessionHandler for Deferring {
    fn accept(&mut self, _: ObjectId) -> Result<(), HandlerError> {
        Ok(())
    }

    fn closed(&mut self, _: ObjectId) {}

    fn handle(
        &mut self,
        _: &mut IpcContext<'_>,
        session: ObjectId,
        _: &mut [u8],
    ) -> Result<Handled, HandlerError> {
        self.waiting.push_back(session);
        if let Some(server) = *lock(&self.server) {
            self.ready.push(server);
        }
        Ok(Handled::Deferred)
    }

    fn take_reply(
        &mut self,
        _: &mut IpcContext<'_>,
        message: &mut [u8],
    ) -> Option<(ObjectId, Result<(), HandlerError>)> {
        let session = self.waiting.pop_front()?;
        message.fill(self.fill);
        Some((session, Ok(())))
    }
}

/// The deferring service's port and number.
const DEFER_PORT: &str = "defer:";
const DEFER: ServiceId = ServiceId(0x77);
/// What its replies are made of.
const REPLY_FILL: u8 = 0x5D;

fn install_deferring(dispatcher: &Dispatcher, _: Arc<dyn Env>, ready: Ready) -> Result<(), String> {
    let server = Arc::new(Mutex::new(None));
    let handler = Deferring {
        ready,
        server: Arc::clone(&server),
        waiting: VecDeque::new(),
        fill: REPLY_FILL,
    };
    let port = PortName::new(DEFER_PORT).map_err(|error| error.to_string())?;
    let id = dispatcher
        .add_server(DEFER_PORT, DEFER, Some(port), Box::new(handler))
        .map_err(|error| error.to_string())?;
    *lock(&server) = Some(id);
    Ok(())
}

/// The guest's memory for a run that sends requests: its image, and its main thread's
/// thread-local page, where the message goes.
struct WithTls<'a> {
    image: &'a FlatMemory,
    tls: FlatMemory,
}

impl GuestMemory for WithTls<'_> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.image.read(va, buf).or_else(|_| self.tls.read(va, buf))
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        self.image
            .write(va, data)
            .or_else(|_| self.tls.write(va, data))
    }
}

impl<V> GuestSpace<V> for WithTls<'_> {
    fn touch(&self, _: u64, _: Access) -> Touch {
        Touch::Refused
    }

    fn settle(&self, _: &mut V) -> Result<(), starquake_cpu::Error> {
        Ok(())
    }
}

#[test]
fn a_deferred_reply_is_delivered_by_the_timers_turn_and_wakes_its_sender() {
    let guest = Guest::new(&programs::hello().unwrap());
    let page = usize::try_from(crate::image::PAGE).unwrap();
    let space = WithTls {
        image: &guest.memory,
        tls: FlatMemory::new(guest.tls, page),
    };
    // The port's name goes where the kernel lets the guest read: the program's read-only data,
    // which a scripted guest does not use.
    let name = guest.start_of("main.rodata");
    guest
        .memory
        .write(name, format!("{DEFER_PORT}\0").as_bytes())
        .unwrap();
    let tls = guest.tls;
    let main = ScriptedVcpu::new()
        .then(move |regs| regs.x[1] = name, svc(SvcId::ConnectToNamedPort))
        .then(
            move |regs| {
                start_created(regs);
                assert_eq!(regs.tpidrro, tls);
            },
            svc(SvcId::SendSyncRequest),
        )
        .then(succeeded, svc(SvcId::ExitProcess));
    let mut delivered_at = None;
    let outcome = stepped(
        &guest,
        &space,
        vec![main],
        &install_deferring,
        |round, machine| {
            // Every round is a timer turn: nothing is ready until the request was deferred.
            let before = lock(&machine.ready.servers).len();
            assert!(
                crate::machine::timer::deliver(machine).is_ok(),
                "a delivery failed"
            );
            if before > 0 && delivered_at.is_none() {
                delivered_at = Some(round);
            }
        },
    );
    assert_exited(&outcome);
    assert!(delivered_at.is_some(), "the reply was never ready");
    let mut reply = vec![0; starquake_ipc::BUFFER_SIZE];
    space.tls.read(tls, &mut reply).unwrap();
    assert!(reply.iter().all(|byte| *byte == REPLY_FILL), "{reply:x?}");
    // The request's event is the sender's, with success, recorded when it was answered.
    let sends: Vec<_> = outcome
        .events
        .iter()
        .filter(|event| {
            matches!(event.kind, EventKind::Svc { id, .. } if id == u32::from(SvcId::SendSyncRequest.number()))
        })
        .collect();
    assert_eq!(sends.len(), 1);
    assert_eq!(sends[0].thread, MAIN_THREAD);
}

#[test]
fn a_run_whose_threads_all_wait_on_nothing_that_can_end_is_an_emulator_error() {
    let guest = Guest::new(&programs::hello().unwrap());
    // A sleep too long for any deadline: no timer, and no other thread.
    let main = ScriptedVcpu::new().then(
        |regs| regs.x[0] = i64::MAX.cast_unsigned(),
        svc(SvcId::SleepThread),
    );
    // Bounded: a run that missed the stall would end at this timeout instead, and fail below.
    let options = RunOptions {
        timeout: Some(Duration::from_secs(30)),
        command_line: Vec::new(),
        sd_card: None,
    };
    let report = guest.run(main, &options);
    assert_eq!(report.exit, ExitReason::EmulatorError(WAITING.to_owned()));
    // A stepped run says the same.
    let main = ScriptedVcpu::new().then(
        |regs| regs.x[0] = i64::MAX.cast_unsigned(),
        svc(SvcId::SleepThread),
    );
    let outcome = stepped_simply(&guest_again(), vec![main]);
    assert!(matches!(&outcome.ending, Ending::Failed(message) if message == WAITING));
}

/// A second guest of the hello program (a guest runs once).
fn guest_again() -> Guest {
    Guest::new(&programs::hello().unwrap())
}

#[test]
fn ending_the_process_stops_every_core_that_runs_guest_code() {
    let guest = Guest::new(&programs::hello().unwrap());
    // Two threads, on cores 1 and 2, each say they run and then never stop by themselves; the
    // main thread ends the process once both run. The run returns only when every core has
    // stopped, so returning is the test.
    let (arrived, arrivals) = mpsc::channel::<u8>();
    let arrived = Mutex::new(arrived);
    let interrupters = Interrupters::default();
    let made = Arc::clone(&interrupters);
    let make = move |core: CoreId| -> Result<ScriptedVcpu, starquake_cpu::Error> {
        let index = core.index();
        let vcpu = if index == 1 || index == 2 {
            let arrived = lock(&arrived).clone();
            ScriptedVcpu::new()
                .then(
                    move |_| arrived.send(index).unwrap(),
                    svc(SvcId::GetCurrentProcessorNumber),
                )
                .then_spin()
        } else {
            ScriptedVcpu::new()
        };
        lock(&made).push(vcpu.interrupter());
        Ok(vcpu)
    };
    let mut main = ScriptedVcpu::new()
        .then(create(1, 30, 1), svc(SvcId::CreateThread))
        .then(start_created, svc(SvcId::StartThread))
        .then(
            |regs| {
                succeeded(regs);
                create(2, 30, 2)(regs);
            },
            svc(SvcId::CreateThread),
        )
        .then(start_created, svc(SvcId::StartThread))
        .then(
            move |regs| {
                succeeded(regs);
                let mut seen = Vec::new();
                for _ in 0..2 {
                    seen.push(arrivals.recv_timeout(Duration::from_secs(10)).unwrap());
                }
                seen.sort_unstable();
                assert_eq!(seen, [1, 2]);
            },
            svc(SvcId::ExitProcess),
        );
    lock(&interrupters).push(main.interrupter());
    let started = Instant::now();
    let report = bounded(&interrupters, || {
        run(
            Vcpus {
                first: &mut main,
                make: &make,
            },
            guest.program(&guest.memory),
            &RunOptions::default(),
            started,
            timebase(),
        )
    });
    assert_eq!(report.exit, ExitReason::ProcessExited, "{}", report.trace);
    assert!(started.elapsed() < Duration::from_secs(20));
}

#[test]
fn the_runs_timeout_stops_every_core() {
    let guest = Guest::new(&programs::hello().unwrap());
    let interrupters = Interrupters::default();
    let made = Arc::clone(&interrupters);
    let make = move |core: CoreId| -> Result<ScriptedVcpu, starquake_cpu::Error> {
        let vcpu = if core.index() == 1 {
            ScriptedVcpu::new().then_spin()
        } else {
            ScriptedVcpu::new()
        };
        lock(&made).push(vcpu.interrupter());
        Ok(vcpu)
    };
    // The main thread starts a thread on core 1, then both spin until the timeout.
    let mut main = ScriptedVcpu::new()
        .then(create(1, 30, 1), svc(SvcId::CreateThread))
        .then(start_created, svc(SvcId::StartThread))
        .then_spin();
    lock(&interrupters).push(main.interrupter());
    let timeout = Duration::from_millis(50);
    let report = bounded(&interrupters, || {
        run(
            Vcpus {
                first: &mut main,
                make: &make,
            },
            guest.program(&guest.memory),
            &RunOptions {
                timeout: Some(timeout),
                command_line: Vec::new(),
                sd_card: None,
            },
            Instant::now(),
            timebase(),
        )
    });
    assert_eq!(report.exit, ExitReason::Timeout);
    assert!(report.duration >= timeout);
    assert!(report.duration < Duration::from_secs(20));
}
