//! The core loop of the walking skeleton: one vCPU runs one guest thread, and every exit is
//! handled here.
//!
//! ```text
//! loop
//!   match vcpu.run()
//!     Svc(n)        -> kernel dispatcher, with the vCPU's registers; ExitProcess ends the run
//!     DataAbort     -> the guest crashed
//!     Undefined     -> the guest crashed
//!     Interrupted   -> the timeout passed, or resume
//! ```
//!
//! It is generic over [`Vcpu`], so everything here runs on any host against a scripted vCPU; the
//! hypervisor only enters in [`crate::system`].
//!
//! PROVISIONAL (walking skeleton, #74), and what replaces each part:
//!
//! - **One guest thread on one core, no scheduler.** The thread is [`MAIN_THREAD`], the process
//!   [`PROCESS_ID`], the core [`CORE`], all fixed. Kernel slice 8 (process creation, thread
//!   lifecycle) creates the real process and its main thread; slice 10 (core loop and timers)
//!   replaces this loop with six core threads driven by the scheduler.
//! - **Whole register file per system call.** Each `svc` reads and writes all of [`GuestRegs`].
//!   The execution model's single-register accessors on `Vcpu` replace that.
//! - **Start state.** The thread starts with its program counter, stack pointer and
//!   `TPIDRRO_EL0` set and every other register zero, and the vCPU's pointer-authentication keys
//!   are left as the backend creates them. Kernel slice 8 sets what Horizon hands a main thread
//!   and generates the process's keys.
//! - **Every abort is fatal.** There is no address space to consult, so a data abort always ends
//!   the run. Kernel slice 6 (memory blocks) decides between lazy backing, write tracking and a
//!   crash.
//! - **Lifecycle, fault and guest-log events are recorded here**, not by the kernel. Lifecycle
//!   and fault events are recorded outside the kernel lock, which is harmless with one thread.
//!   A guest-log event is recorded by this loop's [`DebugSink`] while the dispatcher runs
//!   `OutputDebugString`, so under the kernel lock, just before the call's own event. All three
//!   move to the kernel's own recording points (process and thread lifecycle in slice 8, faults
//!   in slice 6).
//! - **Event times** are host nanoseconds since the run started, converted with the kernel's
//!   tick conversion ([`nanos_to_ticks`]). The clock is read once per `svc`: the call's event
//!   and a log line it writes carry the same time. Nothing reads the times back; the kernel's
//!   guest-time work (kernel spec, "Guest time") decides the real stamp.
//! - **What a run keeps in memory is capped by fixed numbers** ([`Limits`]): the golden trace
//!   keeps its first events and the guest log its first lines, and both say when they stopped.
//!   The observability spec's size cap per run, with its own flags, replaces the constants.
//! - **The environment** is live, with a fixed random seed and nothing recorded to disk; the
//!   `--clock`, `--seed` and `--record` flags of the headless contract are not built.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use starquake_cpu::{Exit, GuestRegs, Interrupter, Vcpu};
use starquake_kernel::dispatch::{Caller, Dispatcher, Outcome, Record, Tables};
use starquake_kernel::handle::{HandleError, HandleTable};
use starquake_kernel::svc::DebugSink;
use starquake_kernel::svc::tick::nanos_to_ticks;
use starquake_kernel::tables::{AddressSpace, Processes, Scheduler};
use starquake_memory::GuestMemory;
use starquake_trace::{
    DumpError, EventKind, FaultAction, FlightRecorder, GuestTicks, LifecyclePhase, LiveEnv,
    Recorder, SystemClock, TextSink, ThreadId,
};

use crate::crash::{CrashReport, Exception};
use crate::image::Placement;
use crate::report::{ExitReason, RunOptions, RunReport, empty_dump};

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

/// Stamps events with the time since the run started (see the module docs).
struct RunClock {
    started: Instant,
}

impl RunClock {
    fn now(&self) -> GuestTicks {
        let nanos = u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        GuestTicks(nanos_to_ticks(nanos))
    }
}

/// What the loop needs that outlives a single exit.
struct Session<'a> {
    memory: &'a dyn GuestMemory,
    placed: &'a Placement,
    options: &'a RunOptions,
    clock: RunClock,
    recorder: Recorder,
}

impl Session<'_> {
    /// Records an event on the one core and thread, stamped with the time now.
    fn record(&self, kind: EventKind) {
        self.record_at(self.clock.now(), kind);
    }

    /// Records an event on the one core and thread, stamped with `time`.
    fn record_at(&self, time: GuestTicks, kind: EventKind) {
        self.recorder.record(time, CORE, MAIN_THREAD, kind);
    }
}

/// Collects what the guest writes with `OutputDebugString` and records each line as a guest-log
/// event.
struct GuestLog<'a> {
    /// The lines kept: a prefix of what the guest wrote.
    lines: Vec<Vec<u8>>,
    /// Bytes in `lines`.
    kept_bytes: usize,
    /// The most bytes `lines` may hold ([`Limits::guest_log_bytes`]).
    limit: usize,
    /// Lines that were not kept: the first that did not fit, and every line after it.
    dropped: u64,
    /// The time of the `svc` being dispatched, which its log line is stamped with.
    time: GuestTicks,
    session: &'a Session<'a>,
}

impl DebugSink for GuestLog<'_> {
    fn output(&mut self, message: &[u8]) {
        // The event is recorded either way: it is fixed-size, and the trace has its own limit.
        self.session
            .record_at(self.time, EventKind::guest_log(message));
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

/// How the loop ended.
enum Ending {
    Exited,
    Crashed(Exception, Box<GuestRegs>),
    TimedOut,
    Failed(String),
}

/// The kernel's tables for one process. PROVISIONAL: the address space and the process table are
/// still placeholders in the kernel, and the handle table has Horizon's maximum size because an
/// NRO carries no NPDM to say otherwise.
fn tables() -> Result<Tables, HandleError> {
    Ok(Tables {
        handles: HandleTable::new(0)?,
        address_space: AddressSpace::default(),
        scheduler: Scheduler::default(),
        processes: Processes::default(),
    })
}

/// Runs the placed program on `vcpu` until it exits, crashes or times out, and reports what
/// happened. `memory` must already hold the program ([`Placement::write_into`]); `started` is
/// when the run began, for the report's duration and the events' times.
pub(crate) fn run<V: Vcpu>(
    vcpu: &mut V,
    memory: &dyn GuestMemory,
    placed: &Placement,
    options: &RunOptions,
    started: Instant,
) -> RunReport {
    run_within(vcpu, memory, placed, options, started, Limits::DEFAULT)
}

/// [`run`], keeping no more of the run in memory than `limits` allows.
pub(crate) fn run_within<V: Vcpu>(
    vcpu: &mut V,
    memory: &dyn GuestMemory,
    placed: &Placement,
    options: &RunOptions,
    started: Instant,
    limits: Limits,
) -> RunReport {
    let session = Session {
        memory,
        placed,
        options,
        clock: RunClock { started },
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
        time: session.clock.now(),
        session: &session,
    };
    let mut tally = Tally::default();
    let exit = match drive(vcpu, &session, &mut log, &mut tally) {
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
        Ending::Crashed(exception, regs) => ExitReason::GuestCrashed(Box::new(CrashReport::new(
            exception,
            &regs,
            &placed.mappings,
            std::slice::from_ref(&placed.module),
            last_events(&flight),
            &options.command_line,
        ))),
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
        memory_map: placed.mappings.clone(),
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

/// The loop itself (see the module docs).
fn drive<V: Vcpu>(
    vcpu: &mut V,
    session: &Session<'_>,
    log: &mut GuestLog<'_>,
    tally: &mut Tally,
) -> Ending {
    let dispatcher = match tables() {
        Ok(tables) => Dispatcher::new(tables),
        Err(error) => return Ending::Failed(format!("the kernel could not be set up: {error}")),
    };
    let env = LiveEnv::new(SystemClock::new(), RANDOM_SEED);
    let vcpu_failed = |error| Ending::Failed(format!("the vCPU failed: {error}"));

    // PROVISIONAL: everything else starts at zero (see "Start state" in the module docs).
    let start = GuestRegs {
        pc: session.placed.entry,
        sp: session.placed.stack_top,
        tpidrro: session.placed.tls,
        ..GuestRegs::default()
    };
    if let Err(error) = vcpu.set_regs(&start) {
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
                // One reading for the call's event and for a log line the call writes.
                let time = session.clock.now();
                log.time = time;
                let record = dispatcher.dispatch(
                    number,
                    &mut regs,
                    &mut Caller {
                        core: CORE,
                        thread: MAIN_THREAD,
                        time,
                        env: &env,
                        memory: session.memory,
                        debug: log,
                        recorder: &session.recorder,
                    },
                );
                tally.count(&record);
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
                }
            }
            Exit::DataAbort { address, write } => {
                session.record(EventKind::Fault {
                    address,
                    write,
                    action: FaultAction::Fatal,
                });
                return match vcpu.regs() {
                    Ok(regs) => {
                        Ending::Crashed(Exception::DataAbort { address, write }, Box::new(regs))
                    }
                    Err(error) => vcpu_failed(error),
                };
            }
            Exit::Undefined { .. } => {
                return match vcpu.regs() {
                    Ok(regs) => Ending::Crashed(Exception::UndefinedInstruction, Box::new(regs)),
                    Err(error) => vcpu_failed(error),
                };
            }
            Exit::Interrupted => {
                if watchdog.fired() {
                    return Ending::TimedOut;
                }
                // Nothing else interrupts the vCPU yet; resume.
            }
        }
    }
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
