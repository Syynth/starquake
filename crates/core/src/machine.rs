//! The core loop (kernel slice 10; execution model, "Guest threads on vCPUs"): one host thread
//! per core of the kernel's scheduler (six for every process today), each owning one vCPU, and
//! one timer thread. A guest thread is the saved context the scheduler names; a core loads it
//! into its vCPU, and a context switch is a register swap.
//!
//! ```text
//! core thread (machine/core_loop.rs):
//!   loop
//!     if it must: dispatcher.switch(core)   // saves the old thread's registers, picks the next
//!       Load(thread)  -> vcpu.set_regs(context); settle the address space
//!       Park          -> park until the scheduler wakes the core (or every core is stalled)
//!     match vcpu.run()
//!       Svc(n)            -> kernel dispatcher; settle the address space; then reschedule
//!                            (ExitProcess ends the run)
//!       DataAbort         -> the address space backs the page and the guest resumes, or it crashed
//!       InstructionAbort  -> the same, for a fetch
//!       Undefined         -> the guest crashed
//!       Interrupted       -> the run is stopping, or reschedule
//!
//! timer thread (machine/timer.rs):
//!   loop
//!     fire the kernel's timeouts that are due, then reschedule
//!     every 10 ms: preempt (rotate each core's queue at its preemption priority)
//!     deliver the deferred replies the run's services say are ready
//!     end the run at its timeout
//!     sleep until the next of those, or until a core or a service rings the timer
//! ```
//!
//! The scheduler talks to the cores through
//! [`CoreSignals`](starquake_kernel::sched::CoreSignals) (`machine/cores.rs`): it interrupts
//! a core running guest code (the vCPU's interrupter) and wakes a parked one (a condition
//! variable). Stopping the run interrupts and wakes every core. A run whose cores have all
//! parked while nothing could ever end a wait ([`Dispatcher::is_stalled`]) ends as an emulator
//! error ([`WAITING`]).
//!
//! It is generic over [`Vcpu`], so everything here runs on any host against scripted vCPUs; the
//! hypervisor only enters in [`crate::system`]. Core 0 runs on the calling thread with the vCPU
//! the caller made; every other core makes its own on its own thread ([`Vcpus::make`]), as
//! Hypervisor.framework requires a vCPU to be run by the thread that created it.
//!
//! PROVISIONAL (walking skeleton, #74, and what is left of it after slice 10), and what replaces
//! each part:
//!
//! - **The process is created from the program's description** ([`Placement::description`]:
//!   its metadata's, or Eden's defaults for an NRO, `crate::image`). The run's [`Timebase`]
//!   must name the description's console, whose rate the program's counter reads were patched
//!   for; a run whose two consoles differ ends as an emulator error before the guest runs
//!   ([`CONSOLE_MISMATCH`]).
//! - **Whole register file per system call.** Each `svc` reads and writes all of
//!   [`GuestRegs`]. The execution model's single-register accessors on `Vcpu` replace that.
//! - **Start state.** The main thread starts with the context the kernel's `process::run` gave
//!   it (`KProcess::Run`, `eden/src/core/hle/kernel/k_process.cpp` @ cf654052): its program
//!   counter, stack pointer and `TPIDRRO_EL0` (its thread-local region) set, `x0` zero and `x1`
//!   a handle to itself, which libnx will not start without (`envGetMainThreadHandle` aborts on
//!   the invalid handle), and every other register zero. The process's pointer-authentication
//!   keys are not generated yet: the hypervisor gives every vCPU the same fixed keys
//!   (`crate::system`), so a thread can move between cores.
//! - **An abort is lazy backing or a crash.** The guest's address space ([`GuestSpace`]) is
//!   asked what a faulting access comes to: a page the kernel mapped that has no backing yet
//!   gets it and the guest resumes (execution model, "Sparse and lazy memory"); anything else
//!   ends the run as a crash. The decision uses the permissions the kernel's block map gave
//!   the translation, not the syndrome. Nothing is write-tracked yet (the GPU caches' slice).
//!   A page that is backed this way is not an event: a console has no fault there, and a trace
//!   must not depend on which pages the host happens to have backed.
//! - **Lifecycle, fault and guest-log events are recorded here**, not by the kernel, on the core
//!   and thread they happened on (the process's start on core [`CORE`] and thread
//!   [`MAIN_THREAD`]). Lifecycle and fault events are recorded outside the kernel lock, so
//!   another core's event can come between a call and the event recorded after it. A guest-log
//!   event is recorded by the calling core's [`DebugSink`] while the dispatcher runs
//!   `OutputDebugString`, so under the kernel lock, just before the call's own event. All three
//!   move to the kernel's own recording points (process and thread lifecycle in slice 8,
//!   faults in slice 6).
//! - **Event times.** The kernel stamps a system call's event with guest time, and tells the
//!   [`DebugSink`] the same time, so a log line carries the time of the call that wrote it. The
//!   events this loop records itself (lifecycle and fault) are stamped from a second copy of
//!   the run's host clock ([`Timebase`]), converted by the kernel's conversion at the process's
//!   rate, so they are on the same scale; they go when those events move into the kernel. The
//!   timer thread reads the same copy to decide when to look at the kernel's timeouts; whether
//!   one has expired is decided by the kernel against the environment's clock
//!   (`machine/timer.rs`).
//! - **What a run keeps in memory is capped by fixed numbers** ([`Limits`]): the golden trace
//!   keeps its first events and the guest log its first lines, and both say when they stopped.
//!   The observability spec's size cap per run, with its own flags, replaces the constants.
//! - **Deferred replies.** A run's services are installed by a function the run is given
//!   ([`Install`]; a real run's is [`crate::hle::install`]), which may keep a [`Ready`] to say
//!   when a server's deferred replies are ready; the timer thread then delivers them
//!   ([`Dispatcher::deliver_replies`]). None of the services a real run has defers yet:
//!   `fsp-srv`'s read would, and is given no executor until then.
//! - **The environment** is live, with a fixed random seed and nothing recorded to disk; the
//!   `--clock`, `--seed` and `--record` flags of the headless contract are not built. Its
//!   recording, which grows by a clock reading per system call, is dropped as it is made (test:
//!   `the_environments_recording_is_dropped_as_it_is_made`).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use starquake_cpu::{GuestRegs, Syndrome, Vcpu};
use starquake_kernel::dispatch::{Dispatcher, Outcome, Record};
use starquake_kernel::id::ServerId;
use starquake_kernel::process;
use starquake_kernel::sched::{APPLICATION_CORE_COUNT, CoreId};
use starquake_kernel::svc::DebugSink;
use starquake_kernel::tables::AddressSpace;
use starquake_kernel::time::CounterRate;
use starquake_memory::{Access, GuestMemory};
use starquake_trace::{
    DumpError, Env, EventKind, FlightRecorder, GuestTicks, HostClock, HostNanos, LifecyclePhase,
    LiveEnv, Recorder, TextSink, ThreadId,
};

use crate::clock::Timebase;
use crate::crash::{CrashReport, Exception};
use crate::hle::HostFolder;
use crate::image::{PAGE, Placement};
use crate::report::{ExitReason, Mapping, RunOptions, RunReport, empty_dump};
use crate::space::Touch;

mod core_loop;
mod cores;
mod timer;

use cores::Cores;

/// The core the vCPU the caller made stands for, and the core the process's start is recorded
/// on.
pub(crate) const CORE: u8 = 0;
/// The main thread, as the event stream names it: the first thread id the kernel hands out
/// (`starquake_kernel::sched::FIRST_THREAD_ID`).
pub(crate) const MAIN_THREAD: ThreadId = ThreadId(starquake_kernel::sched::FIRST_THREAD_ID);
/// The one process, as lifecycle events name it.
pub(crate) const PROCESS_ID: u64 = 1;
/// Events the flight recorder keeps per core (observability spec: "a few thousand each").
const FLIGHT_CAPACITY: usize = 4096;
/// How many of the newest events a crash report quotes.
const LAST_EVENTS: usize = 32;
/// Seed of the environment's random numbers (the `--seed` flag is not built).
const RANDOM_SEED: u64 = 0;
/// How a run ends when its clock counts at a different console's rate than its process's
/// description names (see the module docs).
pub(crate) const CONSOLE_MISMATCH: &str = "the run's clock and the process's description name \
                                           different consoles, so the counter rates would differ";
/// How a run ends when every core has parked and nothing the run has could end a wait
/// ([`Dispatcher::is_stalled`]): every guest thread waits on something only another guest
/// thread could end.
pub(crate) const WAITING: &str = "every guest thread waits, and nothing left in the run can end \
                                  any of the waits";

/// The guest's address space as the loop uses it: its memory, as the kernel reads and writes
/// it, and the two things the loop itself asks of it. `V` is the vCPU the loop runs.
pub(crate) trait GuestSpace<V>: GuestMemory {
    /// What a guest access that faulted at `address` comes to: whether the page only lacked
    /// backing, which it then has (see [`Touch`]).
    fn touch(&self, address: u64, access: Access) -> Touch;

    /// Runs on `vcpu` whatever the address space owes its vCPUs since the last call: the TLB
    /// invalidations for what was unmapped or reprotected. A core calls it whenever it loads a
    /// thread, before the thread first runs there, and after every system call, before the
    /// guest runs again, whatever the call returned.
    fn settle(&self, vcpu: &mut V) -> Result<(), starquake_cpu::Error>;
}

/// The vCPUs of a run: core 0's, made by the caller on the calling thread, which runs core 0,
/// and how every other core makes its own, on its own thread.
pub(crate) struct Vcpus<'a, V> {
    /// Core 0's vCPU.
    pub(crate) first: &'a mut V,
    /// Makes the vCPU of the core it is given, on the calling thread; it must be ready to run
    /// in the run's address space.
    pub(crate) make: &'a (dyn Fn(CoreId) -> Result<V, starquake_cpu::Error> + Sync),
}

/// Says that a server's deferred replies are ready: the timer thread then delivers them
/// ([`Dispatcher::deliver_replies`]). Cloned freely; every clone rings the same timer.
#[derive(Clone)]
pub(crate) struct Ready {
    servers: Arc<Mutex<Vec<ServerId>>>,
    bell: Arc<timer::Bell>,
}

impl Ready {
    /// The replies of `server` that its handler has ready are to be delivered.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "no service of a real run defers yet (`fsp-srv` has no executor); the core \
                      loop's tests install services that do"
        )
    )]
    pub(crate) fn push(&self, server: ServerId) {
        lock(&self.servers).push(server);
        self.bell.ring();
    }

    /// The servers whose replies are ready, oldest first, each once.
    fn take(&self) -> Vec<ServerId> {
        let mut servers = std::mem::take(&mut *lock(&self.servers));
        let mut seen = BTreeSet::new();
        servers.retain(|server| seen.insert(*server));
        servers
    }
}

/// Puts a run's services in front of its dispatcher, with the environment they read the host
/// through and a [`Ready`] to say when deferred replies are ready. A failure ends the run
/// before the guest runs, with the message given.
pub(crate) type Install = dyn Fn(&Dispatcher, Arc<dyn Env>, Ready) -> Result<(), String> + Sync;

/// The services of a real run ([`crate::hle::install`]). None of them defers yet, so the
/// [`Ready`] is not kept.
fn real_services(dispatcher: &Dispatcher, env: Arc<dyn Env>, _: Ready) -> Result<(), String> {
    crate::hle::install(dispatcher, env).map_err(|error| error.to_string())
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

/// Locks a mutex whatever a panicking holder left: everything these mutexes guard stays valid
/// at each step.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Stamps the events the loop records itself with guest time (see the module docs): the loop's
/// copy of the run's host clock, and the process's counter rate.
struct RunClock<C> {
    clock: Mutex<C>,
    rate: CounterRate,
}

impl<C: HostClock + Clone> RunClock<C> {
    fn new(time: Timebase<C>) -> Self {
        let rate = time.rate();
        Self {
            clock: Mutex::new(time.clock),
            rate,
        }
    }

    /// The host clock now, as this copy reads it: not recorded for replay.
    fn host_now(&self) -> HostNanos {
        lock(&self.clock).now()
    }

    /// Guest time now: the kernel's conversion of a reading that is not recorded for replay.
    fn now(&self) -> GuestTicks {
        self.rate.ticks_at(self.host_now())
    }

    /// The other copy of the clock, for the environment.
    fn for_environment(&self) -> C {
        lock(&self.clock).clone()
    }
}

/// What the loop needs that outlives a single exit, shared by every core.
struct Session<'a, C> {
    memory: &'a dyn GuestMemory,
    options: &'a RunOptions,
    clock: RunClock<C>,
    recorder: Recorder,
}

impl<C: HostClock + Clone> Session<'_, C> {
    /// Records an event of the process as a whole, on core [`CORE`] and thread
    /// [`MAIN_THREAD`], stamped with the time now.
    fn record(&self, kind: EventKind) {
        self.record_on(CORE, MAIN_THREAD, kind);
    }

    /// Records an event on `core` and `thread`, stamped with the time now.
    fn record_on(&self, core: u8, thread: ThreadId, kind: EventKind) {
        self.recorder.record(self.clock.now(), core, thread, kind);
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

/// What the guest wrote with `OutputDebugString`, as far as [`Limits::guest_log_bytes`] keeps.
struct GuestLog {
    /// The lines kept: a prefix of what the guest wrote.
    lines: Vec<Vec<u8>>,
    /// Bytes in `lines`.
    kept_bytes: usize,
    /// The most bytes `lines` may hold ([`Limits::guest_log_bytes`]).
    limit: usize,
    /// Lines that were not kept: the first that did not fit, and every line after it.
    dropped: u64,
}

impl GuestLog {
    fn new(limit: usize) -> Self {
        Self {
            lines: Vec::new(),
            kept_bytes: 0,
            limit,
            dropped: 0,
        }
    }

    fn keep(&mut self, message: &[u8]) {
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

/// One core's way into the guest log: records each line as a guest-log event of the core and
/// thread that wrote it, and keeps it.
struct LogSink<'a, C> {
    session: &'a Session<'a, C>,
    log: &'a Mutex<GuestLog>,
    core: u8,
    thread: ThreadId,
}

impl<C: HostClock + Clone> DebugSink for LogSink<'_, C> {
    fn output(&mut self, time: GuestTicks, message: &[u8]) {
        // The event is recorded either way: it is fixed-size, and the trace has its own limit.
        // `time` is the guest time of the `svc` being dispatched, which its own event carries.
        self.session
            .recorder
            .record(time, self.core, self.thread, EventKind::guest_log(message));
        lock(self.log).keep(message);
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

/// The exception an [`Exit::Undefined`](starquake_cpu::Exit::Undefined) is reported as; its
/// syndrome says what it really was.
const UNDEFINED: Exception = Exception::UndefinedInstruction;

/// How the loop ended.
enum Ending {
    Exited,
    Crashed(Exception, Syndrome, Box<GuestRegs>),
    TimedOut,
    Failed(String),
}

/// What every core and the timer thread share during a run.
struct Machine<'a, V: Vcpu, C> {
    session: &'a Session<'a, C>,
    space: &'a dyn GuestSpace<V>,
    dispatcher: Dispatcher,
    env: &'a Arc<LiveEnv>,
    log: Mutex<GuestLog>,
    tally: Mutex<Tally>,
    cores: Cores<V::Interrupter>,
    /// How the run ended: the first ending wins.
    ending: Mutex<Option<Ending>>,
    stopped: AtomicBool,
    /// Rung when the timer thread should look again: a wait began, replies are ready, or the
    /// run is stopping.
    bell: Arc<timer::Bell>,
    ready: Ready,
}

impl<V: Vcpu, C: HostClock + Clone> Machine<'_, V, C> {
    /// Ends the run as `ending`, unless it has already ended: every core and the timer thread
    /// stop.
    fn stop(&self, ending: Ending) {
        {
            let mut slot = lock(&self.ending);
            if slot.is_none() {
                *slot = Some(ending);
            }
        }
        self.stopped.store(true, Ordering::SeqCst);
        self.cores.stop();
        self.bell.ring();
    }

    fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// The scheduler's signals to the cores.
    fn signals(&self) -> cores::Signals<'_, V::Interrupter> {
        cores::Signals(&self.cores)
    }
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

/// Runs the loaded program `guest` on `vcpus` until it exits, crashes or times out, and
/// reports what happened. `started` is when the run began, for the report's duration. `time` is
/// the process's counter rate, which must be the rate the program's counter reads were patched
/// for, and the host clock guest time is counted from.
pub(crate) fn run<V: Vcpu, C: HostClock + Clone + 'static>(
    vcpus: Vcpus<'_, V>,
    guest: GuestProgram<'_, V>,
    options: &RunOptions,
    started: Instant,
    time: Timebase<C>,
) -> RunReport {
    run_within(vcpus, guest, options, started, time, Limits::DEFAULT)
}

/// [`run`], keeping no more of the run in memory than `limits` allows.
pub(crate) fn run_within<V: Vcpu, C: HostClock + Clone + 'static>(
    vcpus: Vcpus<'_, V>,
    guest: GuestProgram<'_, V>,
    options: &RunOptions,
    started: Instant,
    time: Timebase<C>,
    limits: Limits,
) -> RunReport {
    run_serving(
        vcpus,
        guest,
        options,
        (started, time),
        limits,
        &real_services,
    )
}

/// [`run_within`], with the services `install` puts in front of the kernel instead of a real
/// run's.
pub(crate) fn run_serving<V: Vcpu, C: HostClock + Clone + 'static>(
    vcpus: Vcpus<'_, V>,
    guest: GuestProgram<'_, V>,
    options: &RunOptions,
    (started, time): (Instant, Timebase<C>),
    limits: Limits,
    install: &Install,
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
    let flight = Arc::new(FlightRecorder::new(
        usize::from(APPLICATION_CORE_COUNT),
        FLIGHT_CAPACITY,
    ));
    session.recorder.add_sink(Arc::clone(&text));
    session.recorder.add_sink(Arc::clone(&flight));

    let env = session.environment();
    let mut mappings = placed.mappings.clone();
    let booted = if console == placed.description.console {
        boot(address_space, placed, space)
    } else {
        Err(CONSOLE_MISMATCH.to_owned())
    };
    let (ending, log, tally) = match booted {
        Ok((dispatcher, tls)) => {
            mappings.push(tls);
            mappings.sort_by_key(|mapping| mapping.start);
            drive(vcpus, dispatcher, space, &session, &env, limits, install)
        }
        Err(message) => (
            Ending::Failed(message),
            GuestLog::new(limits.guest_log_bytes),
            Tally::default(),
        ),
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
                &placed.modules,
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

/// Creates the process with the kernel and starts its main thread
/// (`starquake_kernel::process::create` and `run`, see "Start state" in the module docs): the
/// process `placed` describes, whose memory is `address_space`, entering at the program's
/// entry with the stack `placed` has; `memory` is where its thread-local page is cleared. The
/// main thread is runnable and runs nowhere yet: the first core the scheduler gives it to loads
/// it. Returns the dispatcher over the process's tables, and the memory-map entry of the main
/// thread's thread-local page.
fn boot(
    address_space: AddressSpace,
    placed: &Placement,
    memory: &dyn GuestMemory,
) -> Result<(Dispatcher, Mapping), String> {
    let failed = |error: &dyn std::fmt::Display| format!("the kernel could not be set up: {error}");
    let mut tables = process::create(&placed.description, address_space).map_err(|e| failed(&e))?;
    let main = process::run(&mut tables, memory, placed.entry, placed.stack_top)
        .map_err(|e| failed(&e))?;
    let tpidrro = tables
        .scheduler
        .thread(main.thread)
        .map_err(|e| failed(&e))?
        .context()
        .tpidrro;
    let page = tpidrro - tpidrro % PAGE;
    let tls = Mapping {
        name: "tls".to_owned(),
        start: page,
        end: page + PAGE,
    };
    Ok((Dispatcher::new(tables), tls))
}

/// The loop itself (see the module docs): installs the services, then runs every core and the
/// timer thread until the run ends. `env` is the environment the kernel reads
/// ([`Session::environment`]); the caller owns it so that a test can see what the loop leaves
/// in its recording. Returns how the run ended, the guest log and the call counts.
fn drive<V: Vcpu, C: HostClock + Clone + 'static>(
    vcpus: Vcpus<'_, V>,
    dispatcher: Dispatcher,
    space: &dyn GuestSpace<V>,
    session: &Session<'_, C>,
    env: &Arc<LiveEnv>,
    limits: Limits,
    install: &Install,
) -> (Ending, GuestLog, Tally) {
    let machine = match Machine::start(dispatcher, space, session, env, limits, install) {
        Ok(machine) => machine,
        Err(failed) => return *failed,
    };
    let count = machine.cores.count();
    let Vcpus { first, make } = vcpus;
    std::thread::scope(|scope| {
        let machine = &machine;
        let timer = std::thread::Builder::new()
            .name("starquake-timer".into())
            .spawn_scoped(scope, move || timer::run(machine));
        if let Err(error) = timer {
            machine.stop(Ending::Failed(format!(
                "the timer thread could not start: {error}"
            )));
        }
        for index in 1..count {
            let Some(core) = CoreId::new(index) else {
                continue;
            };
            let spawned = std::thread::Builder::new()
                .name(format!("starquake-core-{index}"))
                .spawn_scoped(scope, move || match make(core) {
                    Ok(mut vcpu) => core_loop::run(machine, core, &mut vcpu),
                    Err(error) => machine.stop(Ending::Failed(format!(
                        "the vCPU of core {index} could not be created: {error}"
                    ))),
                });
            if let Err(error) = spawned {
                machine.stop(Ending::Failed(format!(
                    "core {index} could not start: {error}"
                )));
            }
        }
        match CoreId::new(CORE) {
            Some(core) => core_loop::run(machine, core, first),
            None => machine.stop(Ending::Failed("there is no core 0".to_owned())),
        }
    });
    machine.finish()
}

impl<'a, V: Vcpu, C: HostClock + Clone> Machine<'a, V, C> {
    /// The machine over `dispatcher`'s tables with one core per core of its scheduler, with
    /// the run's services installed ([`Install`]) and the process's start recorded. A failure
    /// to install the services is how the run ends, with the guest log and call counts it had.
    fn start(
        dispatcher: Dispatcher,
        space: &'a dyn GuestSpace<V>,
        session: &'a Session<'a, C>,
        env: &'a Arc<LiveEnv>,
        limits: Limits,
        install: &Install,
    ) -> Result<Self, Box<(Ending, GuestLog, Tally)>> {
        let count = dispatcher.with::<(&starquake_kernel::tables::Scheduler,), _>(|(scheduler,)| {
            scheduler.core_count()
        });
        let bell = Arc::new(timer::Bell::default());
        let machine = Self {
            session,
            space,
            dispatcher,
            env,
            log: Mutex::new(GuestLog::new(limits.guest_log_bytes)),
            tally: Mutex::new(Tally::default()),
            cores: Cores::new(count),
            ending: Mutex::new(None),
            stopped: AtomicBool::new(false),
            ready: Ready {
                servers: Arc::new(Mutex::new(Vec::new())),
                bell: Arc::clone(&bell),
            },
            bell,
        };
        let services_env: Arc<dyn Env> = Arc::<LiveEnv>::clone(env);
        if let Err(error) = install(&machine.dispatcher, services_env, machine.ready.clone()) {
            let ending = Ending::Failed(format!("the services could not be set up: {error}"));
            return Err(Box::new((
                ending,
                machine.log.into_inner_or_poisoned(),
                Tally::default(),
            )));
        }
        session.record(EventKind::Lifecycle {
            phase: LifecyclePhase::ProcessStart,
            subject: PROCESS_ID,
        });
        Ok(machine)
    }

    /// How the run ended, the guest log and the call counts, once every core and the timer
    /// thread have stopped.
    fn finish(self) -> (Ending, GuestLog, Tally) {
        // PROVISIONAL: nothing keeps the environment's recording (see the module docs).
        drop(self.env.take_recording());
        let ending = lock(&self.ending)
            .take()
            .unwrap_or_else(|| Ending::Failed("the run stopped without saying why".to_owned()));
        (
            ending,
            self.log.into_inner_or_poisoned(),
            self.tally.into_inner_or_poisoned(),
        )
    }
}

/// A mutex's value, whatever a panicking holder left.
trait IntoInner<T> {
    fn into_inner_or_poisoned(self) -> T;
}

impl<T> IntoInner<T> for Mutex<T> {
    fn into_inner_or_poisoned(self) -> T {
        self.into_inner().unwrap_or_else(PoisonError::into_inner)
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
