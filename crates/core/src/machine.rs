//! The core loop of the walking skeleton: one vCPU runs one guest thread, and every exit is
//! handled here.
//!
//! ```text
//! loop
//!   match vcpu.run()
//!     Svc(n)            -> kernel dispatcher, with the vCPU's registers; ExitProcess ends the run;
//!                          then the address space is settled (its pending TLB flushes run)
//!     DataAbort         -> the address space backs the page and the guest resumes, or it crashed
//!     InstructionAbort  -> the same, for a fetch
//!     Undefined         -> the guest crashed
//!     Interrupted       -> the timeout passed, or resume
//! ```
//!
//! It is generic over [`Vcpu`], so everything here runs on any host against a scripted vCPU; the
//! hypervisor only enters in [`crate::system`].
//!
//! PROVISIONAL (walking skeleton, #74), and what replaces each part:
//!
//! - **One guest thread on one core.** The kernel creates the process and starts its main
//!   thread (`starquake_kernel::process::create` and `run`), and the scheduler is told that core
//!   [`CORE`] runs it, so calls about "the calling thread" find it. The loop runs only that
//!   thread: a call that leaves it waiting, or ends it while the process has other threads,
//!   ends the run. The event stream still names the thread [`MAIN_THREAD`] and the process
//!   [`PROCESS_ID`], both fixed. Kernel slice 10 (core loop and timers) replaces this loop with
//!   six core threads driven by the scheduler.
//! - **The process is described by Eden's defaults for a program with no NPDM**
//!   (`ProcessDescription::homebrew`: cores 0-3, priorities 16-63, the main thread at priority
//!   44 on core 0), built for the console the run's [`Timebase`] names. Wiring the loader's
//!   NPDM into process creation (#33) replaces it.
//! - **Whole register file per system call.** Each `svc` reads and writes all of [`GuestRegs`].
//!   The execution model's single-register accessors on `Vcpu` replace that.
//! - **Start state.** The vCPU is given the context the kernel's `process::run` gave the main
//!   thread (`KProcess::Run`, `eden/src/core/hle/kernel/k_process.cpp` @ cf654052): its
//!   program counter, stack pointer and `TPIDRRO_EL0` (its thread-local region) set, `x0` zero
//!   and `x1` a handle to itself, which libnx will not start without (`envGetMainThreadHandle`
//!   aborts on the invalid handle), and every other register zero. The vCPU's
//!   pointer-authentication keys are left as the backend creates them; the process's are not
//!   generated yet.
//! - **An abort is lazy backing or a crash.** The guest's address space ([`GuestSpace`]) is
//!   asked what a faulting access comes to: a page the kernel mapped that has no backing yet
//!   gets it and the guest resumes (execution model, "Sparse and lazy memory"); anything else
//!   ends the run as a crash. The decision uses the permissions the kernel's block map gave
//!   the translation, not the syndrome. Nothing is write-tracked yet (the GPU caches' slice).
//!   A page that is backed this way is not an event: a console has no fault there, and a trace
//!   must not depend on which pages the host happens to have backed.
//! - **Lifecycle, fault and guest-log events are recorded here**, not by the kernel. Lifecycle
//!   and fault events are recorded outside the kernel lock, which is harmless with one thread.
//!   A guest-log event is recorded by this loop's [`DebugSink`] while the dispatcher runs
//!   `OutputDebugString`, so under the kernel lock, just before the call's own event. All three
//!   move to the kernel's own recording points (process and thread lifecycle in slice 8, faults
//!   in slice 6).
//! - **Event times.** The kernel stamps a system call's event with guest time, and tells the
//!   [`DebugSink`] the same time, so a log line carries the time of the call that wrote it. The
//!   events this loop records itself (lifecycle and fault) are stamped from a second copy of
//!   the run's host clock ([`Timebase`]), converted by the kernel's conversion at the process's
//!   rate, so they are on the same scale; they go when those events move into the kernel.
//! - **What a run keeps in memory is capped by fixed numbers** ([`Limits`]): the golden trace
//!   keeps its first events and the guest log its first lines, and both say when they stopped.
//!   The observability spec's size cap per run, with its own flags, replaces the constants.
//! - **A thread that waits ends the run.** A `SendSyncRequest` whose request waits behind
//!   another or is deferred, or a `WaitSynchronization` that nothing has ended yet, leaves the
//!   calling thread waiting ([`Outcome::Waiting`]). With one thread and nothing to deliver a
//!   deferred reply or signal an object, the run ends there as an emulator error ([`WAITING`]).
//!   None of the services a run has yet ([`crate::hle`]) defers: `fsp-srv`'s read would, and is
//!   given no executor until then. Kernel slice 10 runs other threads while one waits, and
//!   delivers deferred replies.
//! - **The environment** is live, with a fixed random seed and nothing recorded to disk; the
//!   `--clock`, `--seed` and `--record` flags of the headless contract are not built. Its
//!   recording, which grows by a clock reading per system call, is dropped as it is made (test:
//!   `the_environments_recording_is_dropped_as_it_is_made`).

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use starquake_cpu::{Exit, GuestRegs, Interrupter, Syndrome, Vcpu};
use starquake_kernel::dispatch::{Caller, Dispatcher, Outcome, Record};
use starquake_kernel::process::{self, Console, ProcessDescription};
use starquake_kernel::sched::{CoreId, CoreSignals};
use starquake_kernel::svc::DebugSink;
use starquake_kernel::tables::AddressSpace;
use starquake_kernel::time::CounterRate;
use starquake_memory::{Access, GuestMemory};
use starquake_trace::{
    DumpError, Env, EventKind, FaultAction, FlightRecorder, GuestTicks, HostClock, LifecyclePhase,
    LiveEnv, Recorder, TextSink, ThreadId,
};

use crate::clock::Timebase;
use crate::crash::{CrashReport, Exception};
use crate::hle::{self, HostFolder};
use crate::image::{PAGE, Placement};
use crate::report::{ExitReason, Mapping, RunOptions, RunReport, empty_dump};
use crate::space::Touch;

/// The core the one vCPU stands for.
pub(crate) const CORE: u8 = 0;
/// The one guest thread, as the event stream names it.
pub(crate) const MAIN_THREAD: ThreadId = ThreadId(1);
/// The one process, as lifecycle events name it.
pub(crate) const PROCESS_ID: u64 = 1;
/// Events the flight recorder keeps for the core (observability spec: "a few thousand each").
const FLIGHT_CAPACITY: usize = 4096;
/// How many of the newest events a crash report quotes.
const LAST_EVENTS: usize = 32;
/// Seed of the environment's random numbers (the `--seed` flag is not built).
const RANDOM_SEED: u64 = 0;
/// How a run ends when its one guest thread waits (see the module docs).
pub(crate) const WAITING: &str = "the guest thread waits for an IPC reply or on an object, and this \
                                  loop has no other thread to run until the wait ends";
/// How a run ends when its one guest thread exits and its process has other threads (see the
/// module docs).
pub(crate) const THREAD_EXITED: &str = "the guest thread exited, and this loop has no other \
                                        thread to run";

/// The guest's address space as the loop uses it: its memory, as the kernel reads and writes
/// it, and the two things the loop itself asks of it. `V` is the vCPU the loop runs.
pub(crate) trait GuestSpace<V>: GuestMemory {
    /// What a guest access that faulted at `address` comes to: whether the page only lacked
    /// backing, which it then has (see [`Touch`]).
    fn touch(&self, address: u64, access: Access) -> Touch;

    /// Runs on `vcpu` whatever the address space owes its vCPUs since the last call: the TLB
    /// invalidations for what was unmapped or reprotected. The loop calls it before the guest
    /// first runs and after every system call, before the guest runs again, whatever the call
    /// returned.
    fn settle(&self, vcpu: &mut V) -> Result<(), starquake_cpu::Error>;
}

/// How much of a run is kept in memory for its report. A guest can make system calls and write
/// log lines for as long as it runs, and a run has no timeout unless it is given one, so without
/// these a looping guest would grow the host's memory without bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Limits {
    /// Events the golden trace keeps: the first this many; later ones are counted, and the
    /// trace ends with a line saying how many.
    pub(crate) trace_events: usize,
    /// Bytes of guest log text kept: whole lines, from the first, while their total stays at
    /// or below this. From the first line that does not fit, lines are counted and not kept.
    pub(crate) guest_log_bytes: usize,
}

impl Limits {
    /// The limits of a real run: 100,000 trace events (about 10 MiB of events) and 4 MiB of
    /// guest log text. Starquake's own numbers; the observability spec asks for a size cap per
    /// run without fixing one.
    pub(crate) const DEFAULT: Self = Self {
        trace_events: 100_000,
        guest_log_bytes: 4 << 20,
    };
}

/// Stamps the events the loop records itself with guest time (see the module docs): the loop's
/// copy of the run's host clock, and the process's counter rate.
struct RunClock<C> {
    clock: RefCell<C>,
    rate: CounterRate,
}

/// The loop's one core, as the scheduler signals it: there is no other core to interrupt or
/// wake, so there is nothing to do.
struct OneCore;

impl CoreSignals for OneCore {
    fn interrupt(&mut self, _: CoreId) {}

    fn wake(&mut self, _: CoreId) {}
}

impl<C: HostClock + Clone> RunClock<C> {
    fn new(time: Timebase<C>) -> Self {
        let rate = time.rate();
        Self {
            clock: RefCell::new(time.clock),
            rate,
        }
    }

    /// Guest time now: the kernel's conversion of a reading that is not recorded for replay.
    fn now(&self) -> GuestTicks {
        self.rate.ticks_at(self.clock.borrow_mut().now())
    }

    /// The other copy of the clock, for the environment.
    fn for_environment(&self) -> C {
        self.clock.borrow().clone()
    }
}

/// What the loop needs that outlives a single exit.
struct Session<'a, C> {
    memory: &'a dyn GuestMemory,
    options: &'a RunOptions,
    clock: RunClock<C>,
    recorder: Recorder,
}

impl<C: HostClock + Clone> Session<'_, C> {
    /// Records an event on the one core and thread, stamped with the time now.
    fn record(&self, kind: EventKind) {
        self.record_at(self.clock.now(), kind);
    }

    /// Records an event on the one core and thread, stamped with `time`.
    fn record_at(&self, time: GuestTicks, kind: EventKind) {
        self.recorder.record(time, CORE, MAIN_THREAD, kind);
    }

    /// The environment the kernel and the services read during this run: live, on the other
    /// copy of the run's clock, with the fixed random seed, and serving the SD card folder the
    /// run names ([`RunOptions::sd_card`]) to `fsp-srv`.
    fn environment(&self) -> Arc<LiveEnv>
    where
        C: 'static,
    {
        let env = LiveEnv::new(self.clock.for_environment(), RANDOM_SEED);
        Arc::new(match &self.options.sd_card {
            Some(folder) => env.with_host_io(HostFolder::new(folder)),
            None => env,
        })
    }
}

/// Collects what the guest writes with `OutputDebugString` and records each line as a guest-log
/// event.
struct GuestLog<'a, C> {
    /// The lines kept: a prefix of what the guest wrote.
    lines: Vec<Vec<u8>>,
    /// Bytes in `lines`.
    kept_bytes: usize,
    /// The most bytes `lines` may hold ([`Limits::guest_log_bytes`]).
    limit: usize,
    /// Lines that were not kept: the first that did not fit, and every line after it.
    dropped: u64,
    session: &'a Session<'a, C>,
}

impl<C: HostClock + Clone> DebugSink for GuestLog<'_, C> {
    fn output(&mut self, time: GuestTicks, message: &[u8]) {
        // The event is recorded either way: it is fixed-size, and the trace has its own limit.
        // `time` is the guest time of the `svc` being dispatched, which its own event carries.
        self.session.record_at(time, EventKind::guest_log(message));
        let fits = self
            .kept_bytes
            .checked_add(message.len())
            .filter(|total| *total <= self.limit);
        match fits {
            Some(total) if self.dropped == 0 => {
                self.kept_bytes = total;
                self.lines.push(message.to_vec());
            }
            _ => self.dropped = self.dropped.saturating_add(1),
        }
    }
}

/// Counts of the system calls the guest made.
#[derive(Default)]
struct Tally {
    calls: BTreeMap<String, u64>,
    stubbed: BTreeSet<String>,
}

impl Tally {
    fn count(&mut self, record: &Record) {
        let number = record.number;
        let name = record
            .id
            .map_or_else(|| format!("{number:#x}"), |id| id.name().to_owned());
        if record.outcome == Outcome::NotImplemented {
            self.stubbed.insert(match record.id {
                Some(_) => format!("{name} ({number:#x})"),
                None => name.clone(),
            });
        }
        *self.calls.entry(name).or_default() += 1;
    }
}

/// Interrupts the vCPU when the run's timeout passes.
struct Watchdog {
    fired: Arc<AtomicBool>,
    /// Dropping the sender wakes the thread early, so a finished run doesn't wait out its
    /// timeout.
    cancel: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Watchdog {
    /// Starts the timer, if there is a timeout.
    fn arm<I: Interrupter + 'static>(
        timeout: Option<Duration>,
        interrupter: I,
    ) -> std::io::Result<Self> {
        let fired = Arc::new(AtomicBool::new(false));
        let Some(timeout) = timeout else {
            return Ok(Self {
                fired,
                cancel: None,
                thread: None,
            });
        };
        let (cancel, cancelled) = mpsc::channel::<()>();
        let flag = Arc::clone(&fired);
        let thread = std::thread::Builder::new()
            .name("starquake-timeout".into())
            .spawn(move || {
                if cancelled.recv_timeout(timeout) == Err(RecvTimeoutError::Timeout) {
                    flag.store(true, Ordering::SeqCst);
                    if let Err(error) = interrupter.interrupt() {
                        tracing::error!(%error, "the timeout could not interrupt the vCPU");
                    }
                }
            })?;
        Ok(Self {
            fired,
            cancel: Some(cancel),
            thread: Some(thread),
        })
    }

    fn fired(&self) -> bool {
        self.fired.load(Ordering::SeqCst)
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        drop(self.cancel.take());
        if let Some(thread) = self.thread.take() {
            // The thread only sleeps and interrupts; if it panicked there is nothing to recover.
            let _ = thread.join();
        }
    }
}

/// The exception an [`Exit::Undefined`] is reported as; its syndrome says what it really was.
const UNDEFINED: Exception = Exception::UndefinedInstruction;

/// How the loop ended.
enum Ending {
    Exited,
    Crashed(Exception, Syndrome, Box<GuestRegs>),
    TimedOut,
    Failed(String),
}

/// A loaded program, as the loop takes it: the guest's address space as host code and the
/// loop see it (which must already hold the program), where the program is, and the kernel's
/// address space for it (`crate::image::load` makes the last two).
pub(crate) struct GuestProgram<'a, V> {
    /// The guest's address space.
    pub(crate) space: &'a dyn GuestSpace<V>,
    /// Where the program, its stack and its TLS block are.
    pub(crate) placed: &'a Placement,
    /// The process's memory, for the kernel's tables.
    pub(crate) address_space: AddressSpace,
}

/// Runs the loaded program `guest` on `vcpu` until it exits, crashes or times out, and reports
/// what happened. `started` is when the run began, for the report's duration. `time` is the
/// process's counter rate, which must be the rate the program's counter reads were patched
/// for, and the host clock guest time is counted from.
pub(crate) fn run<V: Vcpu, C: HostClock + Clone + 'static>(
    vcpu: &mut V,
    guest: GuestProgram<'_, V>,
    options: &RunOptions,
    started: Instant,
    time: Timebase<C>,
) -> RunReport {
    run_within(vcpu, guest, options, started, time, Limits::DEFAULT)
}

/// [`run`], keeping no more of the run in memory than `limits` allows.
pub(crate) fn run_within<V: Vcpu, C: HostClock + Clone + 'static>(
    vcpu: &mut V,
    guest: GuestProgram<'_, V>,
    options: &RunOptions,
    started: Instant,
    time: Timebase<C>,
    limits: Limits,
) -> RunReport {
    let GuestProgram {
        space,
        placed,
        address_space,
    } = guest;
    let console = time.console;
    let session = Session {
        memory: space,
        options,
        clock: RunClock::new(time),
        recorder: Recorder::new(),
    };
    let text = Arc::new(TextSink::with_limit(limits.trace_events));
    let flight = Arc::new(FlightRecorder::new(usize::from(CORE) + 1, FLIGHT_CAPACITY));
    session.recorder.add_sink(Arc::clone(&text));
    session.recorder.add_sink(Arc::clone(&flight));

    let mut log = GuestLog {
        lines: Vec::new(),
        kept_bytes: 0,
        limit: limits.guest_log_bytes,
        dropped: 0,
        session: &session,
    };
    let mut tally = Tally::default();
    let env = session.environment();
    let mut mappings = placed.mappings.clone();
    let ending = match boot(space, address_space, placed, console) {
        Ok((process, tls)) => {
            mappings.push(tls);
            mappings.sort_by_key(|mapping| mapping.start);
            drive(vcpu, process, &session, &env, &mut log, &mut tally)
        }
        Err(message) => Ending::Failed(message),
    };
    let exit = match ending {
        Ending::Exited => ExitReason::ProcessExited,
        Ending::TimedOut => ExitReason::Timeout,
        Ending::Failed(message) => {
            tracing::error!(%message, "the run failed inside the emulator");
            session.record(EventKind::Lifecycle {
                phase: LifecyclePhase::EmulatorError,
                subject: PROCESS_ID,
            });
            ExitReason::EmulatorError(message)
        }
        Ending::Crashed(exception, syndrome, regs) => ExitReason::GuestCrashed(Box::new(
            CrashReport::new(
                exception,
                &regs,
                &mappings,
                std::slice::from_ref(&placed.module),
                last_events(&flight),
                &options.command_line,
            )
            .with_syndrome(syndrome),
        )),
    };
    let (exit, events, events_error) = settle_dump(flight.dump(), exit);
    RunReport {
        exit,
        duration: started.elapsed(),
        guest_log: log.lines,
        guest_log_dropped: log.dropped,
        trace: text.render(),
        trace_events_dropped: text.dropped(),
        events,
        events_error,
        system_calls: tally.calls,
        stubbed: tally.stubbed,
        patched_counter_reads: placed.counter_reads.clone(),
        memory_map: mappings,
    }
}

/// Settles what `events.bin` holds when the flight recorder could or could not be written out,
/// and whether losing it changes how the run ended. Returns the exit reason, the dump's bytes,
/// and why the dump was lost if it was.
///
/// A lost dump is never silent. It is logged, the report says why (`events_error` in
/// `summary.json`), and `events.bin` holds a dump with no cores. A run that otherwise exited
/// cleanly becomes an emulator error; a crash, a timeout and an emulator error keep their own
/// reason, which matters more.
fn settle_dump(
    dump: Result<Vec<u8>, DumpError>,
    exit: ExitReason,
) -> (ExitReason, Vec<u8>, Option<String>) {
    match dump {
        Ok(events) => (exit, events, None),
        Err(error) => {
            let message = format!("the flight recorder could not be written out: {error}");
            tracing::error!(%error, "the flight recorder could not be written out");
            let exit = if exit == ExitReason::ProcessExited {
                ExitReason::EmulatorError(message.clone())
            } else {
                exit
            };
            (exit, empty_dump(), Some(message))
        }
    }
}

/// Creates the process with the kernel and starts its main thread on [`CORE`]
/// (`starquake_kernel::process::create` and `run`, see "Start state" in the module docs): the
/// process described by Eden's defaults for a program with no NPDM, built for `console`, whose
/// memory is `address_space`, entering at the program's entry with the stack `placed` has. The
/// scheduler is told the core runs the main thread. Returns the process, and the memory-map
/// entry of the main thread's thread-local page.
fn boot<'a, V>(
    space: &'a dyn GuestSpace<V>,
    address_space: AddressSpace,
    placed: &Placement,
    console: Console,
) -> Result<(Process<'a, V>, Mapping), String> {
    let failed = |error: &dyn std::fmt::Display| format!("the kernel could not be set up: {error}");
    let description = ProcessDescription::homebrew(console);
    let mut tables = process::create(&description, address_space).map_err(|e| failed(&e))?;
    let main =
        process::run(&mut tables, space, placed.entry, placed.stack_top).map_err(|e| failed(&e))?;
    let core = CoreId::new(CORE).ok_or("no core for the main thread")?;
    let scheduler = &mut tables.scheduler;
    scheduler
        .update(GuestTicks(0), None, &mut OneCore)
        .map_err(|e| failed(&e))?;
    let running = scheduler
        .schedule(core, &mut OneCore)
        .map_err(|e| failed(&e))?;
    if running != Some(main.thread) {
        return Err(failed(
            &"the scheduler did not give the core the main thread",
        ));
    }
    let regs = scheduler
        .thread(main.thread)
        .map_err(|e| failed(&e))?
        .context()
        .clone();
    let page = regs.tpidrro - regs.tpidrro % PAGE;
    let tls = Mapping {
        name: "tls".to_owned(),
        start: page,
        end: page + PAGE,
    };
    let process = Process {
        space,
        dispatcher: Dispatcher::new(tables),
        regs,
    };
    Ok((process, tls))
}

/// Readies `vcpu` to run the main thread from its first instruction: its registers as the
/// kernel left them (see "Start state" in the module docs), and the address space settled,
/// because loading reprotected the program's segments and the guest must not start with a
/// flush owed.
fn start<V: Vcpu>(
    vcpu: &mut V,
    space: &dyn GuestSpace<V>,
    regs: &GuestRegs,
) -> Result<(), starquake_cpu::Error> {
    vcpu.set_regs(regs)?;
    space.settle(vcpu)
}

/// The process the loop runs: the guest's address space, and the kernel's dispatcher over the
/// process's tables, with its main thread started on [`CORE`] from `regs`.
struct Process<'a, V> {
    space: &'a dyn GuestSpace<V>,
    dispatcher: Dispatcher,
    regs: GuestRegs,
}

/// The loop itself (see the module docs). `env` is the environment the kernel reads
/// ([`Session::environment`]); the caller owns it so that a test can see what the loop leaves
/// in its recording.
fn drive<V: Vcpu, C: HostClock + Clone + 'static>(
    vcpu: &mut V,
    process: Process<'_, V>,
    session: &Session<'_, C>,
    env: &Arc<LiveEnv>,
    log: &mut GuestLog<'_, C>,
    tally: &mut Tally,
) -> Ending {
    let Process {
        space,
        dispatcher,
        regs,
    } = process;
    let vcpu_failed = |error| Ending::Failed(format!("the vCPU failed: {error}"));
    let services_env: Arc<dyn Env> = Arc::<LiveEnv>::clone(env);
    if let Err(error) = hle::install(&dispatcher, services_env) {
        return Ending::Failed(format!("the services could not be set up: {error}"));
    }

    if let Err(error) = start(vcpu, space, &regs) {
        return vcpu_failed(error);
    }
    let watchdog = match Watchdog::arm(session.options.timeout, vcpu.interrupter()) {
        Ok(watchdog) => watchdog,
        Err(error) => return Ending::Failed(format!("the timeout could not be armed: {error}")),
    };
    session.record(EventKind::Lifecycle {
        phase: LifecyclePhase::ProcessStart,
        subject: PROCESS_ID,
    });

    loop {
        let exit = match vcpu.run() {
            Ok(exit) => exit,
            Err(error) => return vcpu_failed(error),
        };
        match exit {
            Exit::Svc(number) => {
                let mut regs = match vcpu.regs() {
                    Ok(regs) => regs,
                    Err(error) => return vcpu_failed(error),
                };
                let record = dispatcher.dispatch(
                    number,
                    &mut regs,
                    &mut Caller {
                        core: CORE,
                        thread: MAIN_THREAD,
                        env: &**env,
                        memory: session.memory,
                        debug: log,
                        recorder: &session.recorder,
                    },
                );
                // PROVISIONAL: nothing keeps the environment's recording (see the module docs),
                // and every dispatch adds a clock reading to it.
                drop(env.take_recording());
                tally.count(&record);
                // The call may have unmapped or reprotected memory. What the vCPUs have cached
                // about it must go before the guest runs again, also when the call failed
                // part-way.
                if let Err(error) = space.settle(vcpu) {
                    return vcpu_failed(error);
                }
                match record.outcome {
                    Outcome::Completed | Outcome::NotImplemented => {
                        if let Err(error) = vcpu.set_regs(&regs) {
                            return vcpu_failed(error);
                        }
                    }
                    Outcome::ProcessExited => {
                        session.record(EventKind::Lifecycle {
                            phase: LifecyclePhase::ProcessExit,
                            subject: PROCESS_ID,
                        });
                        return Ending::Exited;
                    }
                    Outcome::EnvFailed(error) => {
                        return Ending::Failed(format!(
                            "the environment failed during svc {number:#x}: {error}"
                        ));
                    }
                    Outcome::Waiting => return Ending::Failed(WAITING.to_owned()),
                    Outcome::ThreadExited => return Ending::Failed(THREAD_EXITED.to_owned()),
                }
            }
            Exit::DataAbort {
                address,
                write,
                syndrome,
            } => {
                let access = if write { Access::Write } else { Access::Read };
                if let Some(ending) = aborted(vcpu, space, session, address, access, syndrome) {
                    return ending;
                }
            }
            Exit::InstructionAbort { address, syndrome } => {
                let access = Access::Execute;
                if let Some(ending) = aborted(vcpu, space, session, address, access, syndrome) {
                    return ending;
                }
            }
            Exit::Undefined { syndrome, .. } => return crashed(vcpu, UNDEFINED, syndrome),
            Exit::Interrupted => {
                if watchdog.fired() {
                    return Ending::TimedOut;
                }
                // Nothing else interrupts the vCPU yet; resume.
            }
        }
    }
}

/// What an abort on an `access` to `address` comes to: `None` when the page only lacked backing
/// and now has it, so the guest retries the access; otherwise how the run ends. `syndrome` is
/// the abort's, for the crash report.
fn aborted<V: Vcpu, C: HostClock + Clone>(
    vcpu: &V,
    space: &dyn GuestSpace<V>,
    session: &Session<'_, C>,
    address: u64,
    access: Access,
    syndrome: Syndrome,
) -> Option<Ending> {
    match space.touch(address, access) {
        Touch::Backed => None,
        Touch::Refused => {
            // The event stream has one kind of fault; a fetch is recorded as a read, as it
            // was when the backend reported fetch faults as data aborts.
            let (exception, write) = match access {
                Access::Execute => (Exception::InstructionAbort { address }, false),
                Access::Write => (
                    Exception::DataAbort {
                        address,
                        write: true,
                    },
                    true,
                ),
                Access::Read => (
                    Exception::DataAbort {
                        address,
                        write: false,
                    },
                    false,
                ),
            };
            Some(fatal_fault(
                vcpu,
                session,
                address,
                write,
                (exception, syndrome),
            ))
        }
        Touch::Failed(error) => Some(unbacked(address, &error)),
    }
}

/// Ends the run because a page the guest may use could not be given backing: the emulator's
/// failure, not the guest's.
fn unbacked(address: u64, error: &crate::space::SpaceError) -> Ending {
    Ending::Failed(format!(
        "guest memory at {address:#x} could not be backed: {error}"
    ))
}

/// Ends the run as a crash with `exception` and its `syndrome`, with the guest's registers as
/// they are now.
fn crashed<V: Vcpu>(vcpu: &V, exception: Exception, syndrome: Syndrome) -> Ending {
    match vcpu.regs() {
        Ok(regs) => Ending::Crashed(exception, syndrome, Box::new(regs)),
        Err(error) => Ending::Failed(format!("the vCPU failed: {error}")),
    }
}

/// Records an abort at `address` as a fatal fault and ends the run as a crash with `exception`
/// and its syndrome.
fn fatal_fault<V: Vcpu, C: HostClock + Clone>(
    vcpu: &V,
    session: &Session<'_, C>,
    address: u64,
    write: bool,
    (exception, syndrome): (Exception, Syndrome),
) -> Ending {
    session.record(EventKind::Fault {
        address,
        write,
        action: FaultAction::Fatal,
    });
    crashed(vcpu, exception, syndrome)
}

/// The newest events the flight recorder holds, oldest first, one line each.
fn last_events(flight: &FlightRecorder) -> Vec<String> {
    let events = flight.snapshot().merged();
    let skip = events.len().saturating_sub(LAST_EVENTS);
    events
        .iter()
        .skip(skip)
        .map(|event| {
            format!(
                "#{} core {} thread {}: {}",
                event.seq, event.core, event.thread.0, event.kind
            )
        })
        .collect()
}

#[cfg(test)]
mod tests;
