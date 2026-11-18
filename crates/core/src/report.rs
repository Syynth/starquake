//! What a run was asked to do and what it left behind: [`RunOptions`], [`RunReport`], the exit
//! codes of the headless contract, and the `summary.json` view.
//!
//! Source: `docs/specs/observability.md`, "Headless runs" (exit codes, the artifacts table). The
//! spec names what `summary.json` holds but not its field names, so the layout here is this
//! crate's own, versioned by [`SUMMARY_VERSION`] and pinned by a test.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Duration;

use serde::Serialize;
use starquake_kernel::sched::ThreadState;
use starquake_trace::Dump;

use crate::crash::{CrashReport, Location, Module, address_text};
use crate::services_report::{ServicesReport, StubbedCommand};

/// The process exit codes of `starquake run` (`docs/specs/observability.md`, "Headless runs").
pub mod exit_code {
    /// The run reached its stop condition. For the walking skeleton that is the guest calling
    /// `ExitProcess`.
    pub const STOP_CONDITION: u8 = 0;
    /// The guest crashed.
    pub const GUEST_CRASHED: u8 = 1;
    /// The emulator failed: the program could not be loaded, no CPU backend exists on this host,
    /// or something went wrong inside the emulator while the guest ran.
    pub const EMULATOR_ERROR: u8 = 2;
    /// The run was stopped by its timeout.
    pub const TIMEOUT: u8 = 3;
    /// An unimplemented feature was hit under the strict policy. Nothing returns this yet: the
    /// strict policy is not built.
    pub const UNIMPLEMENTED: u8 = 4;
}

/// Version of the `summary.json` layout. Bump it when a field is renamed, removed or changes
/// meaning.
///
/// 2: `service_commands` is filled in, and `service_commands_uncounted`, `stubbed_commands`,
/// `unknown_services`, `host_log_lines` and `host_log_lines_dropped` are added.
pub const SUMMARY_VERSION: u32 = 2;

/// Bytes of guest log text a run keeps (`guest_log.txt`), and of the emulator's own warnings
/// (`log.txt`, [`crate::host_log`]): whole lines from the first while their total fits.
/// starquake's own number; the observability spec asks for a size cap per run without fixing
/// one ("These numbers are provisional").
pub const GUEST_LOG_BYTES: usize = 4 << 20;

/// What a run is asked to do.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunOptions {
    /// Stop the run after this long (`--until timeout=<n>s`). `None` lets the guest run until
    /// it exits or crashes.
    pub timeout: Option<Duration>,
    /// The command line that started the run, for the crash report. Empty when there was none.
    pub command_line: Vec<String>,
    /// The host folder the guest sees as its SD card, read-only (`fsp-srv`'s
    /// `OpenSdCardFileSystem`). `None`: the run has no SD card.
    pub sd_card: Option<PathBuf>,
    /// The title's RomFS, which the guest opens as its data storage (`fsp-srv`'s
    /// `OpenDataStorageByCurrentProcess`), from the narrow title API (`crate::title::open`).
    /// `None`: the run has no data storage.
    pub romfs: Option<crate::title::RomFs>,
}

/// How a run ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitReason {
    /// The guest called `ExitProcess`.
    ProcessExited,
    /// The guest took an exception nothing handles.
    GuestCrashed(Box<CrashReport>),
    /// The run's timeout passed.
    Timeout,
    /// The emulator failed; the text says how.
    EmulatorError(String),
}

impl ExitReason {
    /// The process exit code for this reason (see [`exit_code`]).
    #[must_use]
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::ProcessExited => exit_code::STOP_CONDITION,
            Self::GuestCrashed(_) => exit_code::GUEST_CRASHED,
            Self::EmulatorError(_) => exit_code::EMULATOR_ERROR,
            Self::Timeout => exit_code::TIMEOUT,
        }
    }

    /// The reason's name in `summary.json`.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::ProcessExited => "process_exited",
            Self::GuestCrashed(_) => "guest_crashed",
            Self::Timeout => "timeout",
            Self::EmulatorError(_) => "emulator_error",
        }
    }
}

/// One range of guest memory the run set up.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Mapping {
    /// What the range holds, for example `main.text` or `stack`.
    pub name: String,
    /// Guest address of the first byte.
    #[serde(serialize_with = "crate::crash::hex")]
    pub start: u64,
    /// Guest address one past the last byte.
    #[serde(serialize_with = "crate::crash::hex")]
    pub end: u64,
}

impl Mapping {
    /// Whether `address` is inside the range.
    #[must_use]
    pub fn contains(&self, address: u64) -> bool {
        (self.start..self.end).contains(&address)
    }
}

/// Where one guest thread was when the run stopped at its timeout, so a hang shows where it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ThreadPc {
    /// The thread's id, as the guest and the event stream know it.
    pub thread: u64,
    /// What the kernel's scheduler had it doing.
    pub state: ThreadPcState,
    /// The core whose vCPU held the thread's registers when the run stopped; `null` when they
    /// were in the thread's saved context instead (no core ran it, or its last system call
    /// began a wait, which saved them), or the vCPU's registers could not be read.
    pub core: Option<u8>,
    /// Its program counter: the vCPU's when [`ThreadPc::core`] names one, its saved context's
    /// otherwise.
    #[serde(serialize_with = "crate::crash::hex")]
    pub pc: u64,
    /// That address as module plus offset, when it is inside a module.
    pub pc_location: Option<Location>,
}

/// What the kernel's scheduler had a thread doing when the run stopped ([`ThreadPc::state`]),
/// written in `summary.json` as its [`ThreadPcState::name`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadPcState {
    /// Created, not started.
    Initialized,
    /// Waiting for something (a synchronisation object, a sleep, a reply).
    Waiting,
    /// Ready to run, or running.
    Runnable,
    /// Ended.
    Terminated,
}

impl ThreadPcState {
    /// The state in words, as `summary.json` and the timeout's text write it: `initialized`,
    /// `waiting`, `runnable` or `terminated`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Initialized => "initialized",
            Self::Waiting => "waiting",
            Self::Runnable => "runnable",
            Self::Terminated => "terminated",
        }
    }
}

impl From<ThreadState> for ThreadPcState {
    fn from(state: ThreadState) -> Self {
        match state {
            ThreadState::Initialized => Self::Initialized,
            ThreadState::Waiting => Self::Waiting,
            ThreadState::Runnable => Self::Runnable,
            ThreadState::Terminated => Self::Terminated,
        }
    }
}

impl ThreadPc {
    /// Thread `thread` in `state`, held by `core`, at `pc`, which is placed in `modules`.
    #[must_use]
    pub fn new(
        thread: u64,
        state: ThreadState,
        core: Option<u8>,
        pc: u64,
        modules: &[Module],
    ) -> Self {
        Self {
            thread,
            state: ThreadPcState::from(state),
            core,
            pc,
            pc_location: Location::of(pc, modules),
        }
    }

    /// The thread in words: `thread 1, runnable on core 0, pc 0x800000b8 (main+0xb8)`.
    #[must_use]
    pub fn text(&self) -> String {
        let core = self
            .core
            .map_or_else(String::new, |core| format!(" on core {core}"));
        format!(
            "thread {}, {}{core}, pc {}",
            self.thread,
            self.state.name(),
            address_text(self.pc, self.pc_location.as_ref())
        )
    }
}

/// Everything a run left behind. The artifacts folder is written from this
/// ([`crate::artifacts::write`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunReport {
    /// How the run ended.
    pub exit: ExitReason,
    /// How long the run took, from loading the program to the guest stopping.
    pub duration: Duration,
    /// The strings the guest wrote with `OutputDebugString`, in order, as the guest's bytes:
    /// every one of them, unless there were more than a run keeps (4 MiB of text), in which case
    /// these are the first and [`RunReport::guest_log_dropped`] counts the rest.
    pub guest_log: Vec<Vec<u8>>,
    /// How many later guest log lines were not kept because the limit was reached.
    pub guest_log_dropped: u64,
    /// The golden-trace view of the event stream (`trace.txt`). It holds the run's first
    /// 100,000 events; when there were more, its last line says how many are missing.
    pub trace: String,
    /// How many later events are missing from [`RunReport::trace`] because the limit was
    /// reached. (The newest events are still in the flight recorder's dump.)
    pub trace_events_dropped: u64,
    /// The flight recorder's dump (`events.bin`), readable with `starquake_trace::Dump`.
    pub events: Vec<u8>,
    /// Why the flight recorder could not be written out, if it could not. [`RunReport::events`]
    /// is then a dump with no cores in it.
    pub events_error: Option<String>,
    /// How many times the guest made each system call, by the call's name (or its number in
    /// hex, for a number Horizon does not list).
    pub system_calls: BTreeMap<String, u64>,
    /// The system calls the guest made that are not implemented and were answered with "not
    /// implemented", as `Name (0xNN)`.
    pub stubbed: BTreeSet<String>,
    /// Guest addresses of the counter reads the loader patched, in address order.
    pub patched_counter_reads: Vec<u64>,
    /// The guest memory the run set up, in address order. Empty if the run failed before the
    /// program was placed.
    pub memory_map: Vec<Mapping>,
    /// Where each guest thread was when the run stopped, by thread id, for a run that ended at
    /// its timeout ([`ExitReason::Timeout`]); empty for every other ending.
    pub threads: Vec<ThreadPc>,
    /// What the services left behind: requests by command, the commands they auto-stubbed and
    /// the service names nothing provides. Empty if the run failed before its services were
    /// set up.
    pub services: ServicesReport,
    /// The emulator's own warnings and errors during the run (`log.txt`), as the command-line
    /// tool's [`HostLog`](crate::host_log::HostLog) kept them: up to [`GUEST_LOG_BYTES`] of
    /// text, the first lines. Empty unless whoever ran the run put them here.
    pub host_log: Vec<String>,
    /// How many later lines of the emulator's log were not kept because the limit was reached.
    pub host_log_dropped: u64,
}

impl RunReport {
    /// The report of a run that failed before any guest code ran: no trace, and a flight
    /// recorder dump with no cores in it.
    #[must_use]
    pub fn emulator_error(message: String, duration: Duration) -> Self {
        Self {
            exit: ExitReason::EmulatorError(message),
            duration,
            guest_log: Vec::new(),
            guest_log_dropped: 0,
            trace: String::new(),
            trace_events_dropped: 0,
            events: empty_dump(),
            events_error: None,
            system_calls: BTreeMap::new(),
            stubbed: BTreeSet::new(),
            patched_counter_reads: Vec::new(),
            memory_map: Vec::new(),
            threads: Vec::new(),
            services: ServicesReport::default(),
            host_log: Vec::new(),
            host_log_dropped: 0,
        }
    }

    /// The process exit code for this run (see [`exit_code`]).
    #[must_use]
    pub fn exit_code(&self) -> u8 {
        self.exit.exit_code()
    }

    /// The crash report, if the guest crashed.
    #[must_use]
    pub fn crash(&self) -> Option<&CrashReport> {
        match &self.exit {
            ExitReason::GuestCrashed(crash) => Some(crash),
            _ => None,
        }
    }

    /// The `summary.json` view of this run.
    #[must_use]
    pub fn summary(&self) -> Summary<'_> {
        Summary {
            version: SUMMARY_VERSION,
            exit_reason: self.exit.name(),
            exit_code: self.exit_code(),
            message: match &self.exit {
                ExitReason::EmulatorError(message) => Some(message),
                _ => None,
            },
            duration_ms: u64::try_from(self.duration.as_millis()).unwrap_or(u64::MAX),
            frames: 0,
            system_calls: &self.system_calls,
            service_commands: &self.services.commands,
            service_commands_uncounted: self.services.uncounted,
            stubbed: &self.stubbed,
            stubbed_commands: &self.services.stubbed,
            unknown_services: &self.services.unknown_services,
            guest_log_lines: u64::try_from(self.guest_log.len())
                .unwrap_or(u64::MAX)
                .saturating_add(self.guest_log_dropped),
            guest_log_lines_dropped: self.guest_log_dropped,
            host_log_lines: u64::try_from(self.host_log.len())
                .unwrap_or(u64::MAX)
                .saturating_add(self.host_log_dropped),
            host_log_lines_dropped: self.host_log_dropped,
            trace_events_dropped: self.trace_events_dropped,
            events_error: self.events_error.as_ref(),
            patched_counter_reads: self.patched_counter_reads.len(),
            threads: &self.threads,
        }
    }
}

/// A flight-recorder dump with no cores in it. It still has its header, so `events.bin` is
/// always readable. Encoding one cannot fail; if it ever did, the empty file that results is
/// refused by the reader.
pub(crate) fn empty_dump() -> Vec<u8> {
    Dump::new(Vec::new()).to_bytes().unwrap_or_default()
}

/// The contents of `summary.json` (`docs/specs/observability.md`, the artifacts table: "Exit
/// reason, duration, frames, counts per system call and service command, every stubbed command
/// hit").
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Summary<'a> {
    /// [`SUMMARY_VERSION`].
    pub version: u32,
    /// `process_exited`, `guest_crashed`, `timeout` or `emulator_error`.
    pub exit_reason: &'static str,
    /// The process exit code that goes with the reason.
    pub exit_code: u8,
    /// What went wrong, for `emulator_error`; `null` otherwise.
    pub message: Option<&'a String>,
    /// How long the run took, in milliseconds.
    pub duration_ms: u64,
    /// Frames presented. Always zero until there is a graphics path.
    pub frames: u64,
    /// Count per system call, by name.
    pub system_calls: &'a BTreeMap<String, u64>,
    /// Requests per service command, by `<service>:<interface>::<command>`
    /// ([`command_key`](crate::services_report::command_key)), in key order.
    pub service_commands: &'a BTreeMap<String, u64>,
    /// Requests missing from `service_commands` because a server had reached its bound of
    /// distinct commands. Zero means `service_commands` counts every request.
    pub service_commands_uncounted: u64,
    /// Every unimplemented system call the guest hit.
    pub stubbed: &'a BTreeSet<String>,
    /// Every command the services answered by auto-stub, with its service, its interface (for
    /// a session `sm:` handed out for a name nothing provides, that name), its protocol and its
    /// id, in that order.
    pub stubbed_commands: &'a [StubbedCommand],
    /// The service names the guest asked `sm:` for that nothing provides, in order.
    pub unknown_services: &'a [String],
    /// How many lines the guest wrote with `OutputDebugString`, kept or not.
    pub guest_log_lines: u64,
    /// How many of those lines are missing from `guest_log.txt` because the run reached its
    /// limit of guest log text. Zero means the file is complete.
    pub guest_log_lines_dropped: u64,
    /// How many warnings and errors the emulator logged during the run (`log.txt`), kept or
    /// not.
    pub host_log_lines: u64,
    /// How many of those lines are missing from `log.txt` because the run reached its limit.
    /// Zero means the file is complete.
    pub host_log_lines_dropped: u64,
    /// How many events are missing from the end of `trace.txt` because the run reached its
    /// limit of trace events. Zero means the trace is complete.
    pub trace_events_dropped: u64,
    /// Why `events.bin` holds no events, when the flight recorder could not be written out;
    /// `null` otherwise.
    pub events_error: Option<&'a String>,
    /// How many counter reads the loader patched.
    pub patched_counter_reads: usize,
    /// For a run that ended at its timeout, where each guest thread was ([`ThreadPc`]); empty
    /// otherwise.
    pub threads: &'a [ThreadPc],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crash::{CrashReport, Exception, Module};

    fn crash() -> Box<CrashReport> {
        Box::new(CrashReport::new(
            Exception::UndefinedInstruction,
            &starquake_cpu::GuestRegs::default(),
            &[],
            &[],
            Vec::new(),
            &[],
        ))
    }

    #[test]
    fn each_reason_has_the_exit_code_of_the_headless_contract() {
        // docs/specs/observability.md: "0 reached the stop condition, 1 the guest crashed,
        // 2 emulator error, 3 timeout, 4 an unimplemented feature was hit under the strict policy".
        let cases = [
            (ExitReason::ProcessExited, 0, "process_exited"),
            (ExitReason::GuestCrashed(crash()), 1, "guest_crashed"),
            (ExitReason::EmulatorError("x".into()), 2, "emulator_error"),
            (ExitReason::Timeout, 3, "timeout"),
        ];
        for (reason, code, name) in cases {
            assert_eq!(reason.exit_code(), code, "{name}");
            assert_eq!(reason.name(), name);
        }
        assert_eq!(
            [
                exit_code::STOP_CONDITION,
                exit_code::GUEST_CRASHED,
                exit_code::EMULATOR_ERROR,
                exit_code::TIMEOUT,
                exit_code::UNIMPLEMENTED
            ],
            [0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn the_summary_layout_is_pinned() {
        // Every field holds a distinct non-default value, so a swapped or dropped one shows.
        let report = RunReport {
            exit: ExitReason::EmulatorError("the vCPU failed".into()),
            duration: Duration::from_millis(1234),
            guest_log: vec![b"one".to_vec(), b"two".to_vec(), b"three".to_vec()],
            guest_log_dropped: 5,
            trace: "ignored".into(),
            trace_events_dropped: 9,
            events: vec![1, 2, 3],
            events_error: Some("the dump could not be encoded".into()),
            system_calls: BTreeMap::from([
                ("ExitProcess".to_owned(), 1),
                ("OutputDebugString".to_owned(), 7),
            ]),
            stubbed: BTreeSet::from(["Break (0x26)".to_owned(), "0x99".to_owned()]),
            patched_counter_reads: vec![0x8000_00b0, 0x8000_00c0],
            memory_map: Vec::new(),
            threads: Vec::new(),
            services: ServicesReport {
                commands: BTreeMap::from([
                    ("sm::Sm::GetService".to_owned(), 3),
                    ("sm::lm::5".to_owned(), 4),
                ]),
                uncounted: 6,
                stubbed: vec![StubbedCommand {
                    service: "sm:".to_owned(),
                    interface: "lm".to_owned(),
                    protocol: starquake_trace::IpcProtocol::Tipc,
                    command: 0x15,
                }],
                unknown_services: vec!["aoc:u".to_owned(), "lm".to_owned()],
                names: BTreeMap::new(),
            },
            host_log: vec!["WARN a".to_owned(), "ERROR b".to_owned()],
            host_log_dropped: 10,
        };
        let json = serde_json::to_string_pretty(&report.summary()).unwrap();
        assert_eq!(
            json,
            r#"{
  "version": 2,
  "exit_reason": "emulator_error",
  "exit_code": 2,
  "message": "the vCPU failed",
  "duration_ms": 1234,
  "frames": 0,
  "system_calls": {
    "ExitProcess": 1,
    "OutputDebugString": 7
  },
  "service_commands": {
    "sm::Sm::GetService": 3,
    "sm::lm::5": 4
  },
  "service_commands_uncounted": 6,
  "stubbed": [
    "0x99",
    "Break (0x26)"
  ],
  "stubbed_commands": [
    {
      "service": "sm:",
      "interface": "lm",
      "protocol": "tipc",
      "command": 21
    }
  ],
  "unknown_services": [
    "aoc:u",
    "lm"
  ],
  "guest_log_lines": 8,
  "guest_log_lines_dropped": 5,
  "host_log_lines": 12,
  "host_log_lines_dropped": 10,
  "trace_events_dropped": 9,
  "events_error": "the dump could not be encoded",
  "patched_counter_reads": 2,
  "threads": []
}"#
        );
        assert_eq!(report.summary().version, SUMMARY_VERSION);
    }

    fn modules() -> Vec<Module> {
        vec![
            Module {
                name: "rtld".into(),
                base: 0x8000_0000,
                size: 0x4000,
                build_id: vec![0x01],
            },
            Module {
                name: "main".into(),
                base: 0x8000_4000,
                size: 0x2000,
                build_id: vec![0x02],
            },
        ]
    }

    /// A timed-out run's threads: every field distinct and non-zero, one thread a core held and
    /// one whose saved context was used, one inside a module and one in none.
    fn timed_out() -> RunReport {
        RunReport {
            exit: ExitReason::Timeout,
            threads: vec![
                ThreadPc::new(
                    0x51,
                    ThreadState::Runnable,
                    Some(3),
                    0x8000_40b8,
                    &modules(),
                ),
                ThreadPc::new(0x52, ThreadState::Waiting, None, 0x7000_1234, &modules()),
            ],
            ..RunReport::emulator_error(String::new(), Duration::from_millis(77))
        }
    }

    #[test]
    fn a_timed_out_runs_threads_in_the_summary_are_pinned() {
        let report = timed_out();
        let json = serde_json::to_value(report.summary()).unwrap();
        assert_eq!(json["exit_reason"], "timeout");
        assert_eq!(
            json["threads"],
            serde_json::to_value(&report.threads).unwrap()
        );
        // In key order, as `summary.json` writes them.
        assert_eq!(
            serde_json::to_string_pretty(report.summary().threads).unwrap(),
            r#"[
  {
    "thread": 81,
    "state": "runnable",
    "core": 3,
    "pc": "0x800040b8",
    "pc_location": {
      "module": "main",
      "offset": "0xb8"
    }
  },
  {
    "thread": 82,
    "state": "waiting",
    "core": null,
    "pc": "0x70001234",
    "pc_location": null
  }
]"#
        );
    }

    #[test]
    fn a_thread_says_its_state_its_core_and_where_it_is_in_words() {
        let cases = [
            (
                ThreadState::Initialized,
                ThreadPcState::Initialized,
                "initialized",
            ),
            (ThreadState::Waiting, ThreadPcState::Waiting, "waiting"),
            (ThreadState::Runnable, ThreadPcState::Runnable, "runnable"),
            (
                ThreadState::Terminated,
                ThreadPcState::Terminated,
                "terminated",
            ),
        ];
        for (state, reported, name) in cases {
            let thread = ThreadPc::new(7, state, None, 0x8000_0000, &modules());
            assert_eq!(thread.state, reported);
            assert_eq!(reported.name(), name);
            // The JSON writes the name the text does.
            assert_eq!(serde_json::to_value(reported).unwrap(), name);
            assert_eq!(
                thread.text(),
                format!("thread 7, {name}, pc 0x80000000 (rtld+0x0)")
            );
        }
        let held = ThreadPc::new(9, ThreadState::Runnable, Some(5), 0x8000_5fff, &modules());
        assert_eq!(
            held.text(),
            "thread 9, runnable on core 5, pc 0x80005fff (main+0x1fff)"
        );
        // One past the last module's end is in no module.
        let outside = ThreadPc::new(9, ThreadState::Runnable, Some(0), 0x8000_6000, &modules());
        assert_eq!(outside.pc_location, None);
        assert_eq!(
            outside.text(),
            "thread 9, runnable on core 0, pc 0x80006000 (in no module)"
        );
    }

    #[test]
    fn a_run_that_never_started_reports_an_emulator_error_and_nothing_else() {
        let report = RunReport::emulator_error("no backend".into(), Duration::from_millis(3));
        assert_eq!(report.exit, ExitReason::EmulatorError("no backend".into()));
        assert_eq!(report.exit_code(), exit_code::EMULATOR_ERROR);
        assert_eq!(report.crash(), None);
        assert_eq!(report.summary().message, Some(&"no backend".to_owned()));
        assert_eq!(report.trace, "");
        assert_eq!(Dump::from_bytes(&report.events), Ok(Dump::new(Vec::new())));
    }

    #[test]
    fn a_mapping_holds_its_start_and_not_its_end() {
        let mapping = Mapping {
            name: "stack".into(),
            start: 0x1000,
            end: 0x2000,
        };
        assert!(mapping.contains(0x1000) && mapping.contains(0x1fff));
        assert!(!mapping.contains(0xfff) && !mapping.contains(0x2000));
    }
}
