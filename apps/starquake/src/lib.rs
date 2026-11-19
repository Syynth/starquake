//! The starquake command line: `starquake run <program>`, the headless contract of
//! `docs/specs/observability.md` as far as the walking skeleton goes, and `starquake inspect`
//! and `starquake verify`, which read a NUP title ([`title`]).
//!
//! The program `run` takes is an NRO, or a NUP title: a file whose first bytes are a NUP
//! container's is run as the title's newest package, whole (its newest update, or its base),
//! opened by `starquake_core::title::open`: that package's ExeFS is the program, and its RomFS
//! the data storage the guest opens, read from the file as the guest reads it. Any other file is
//! read as an NRO. A title that cannot be loaded is an emulator error, like an NRO that cannot.
//!
//! ```text
//! starquake run <program> [--until timeout=<n>s | frames=<n>] [--artifacts <folder>]
//!                          [--date <seconds>]
//! starquake inspect <title.nup>
//! starquake verify <title.nup>
//! starquake gpu-replay <capture> [--until present=<n>] [--dump ring|render|writes]
//! ```
//!
//! `gpu-replay` replays a GPU capture with no game and no GPU ([`gpu_replay`]); its exit codes
//! are 0 (replayed to its stop condition) and 2 (anything else).
//!
//! The process exit code says how the run ended (`starquake_core::exit_code`): 0 the program
//! exited or presented the frames `--until frames=<n>` asked for, 1 the guest crashed, 2 an
//! emulator error, 3 the timeout passed. A command line this
//! tool can't parse, a program file it can't read and an artifacts folder it can't write are
//! emulator errors too: the contract has no other code for them.
//!
//! The run's date is the host's, read once as the run boots and recorded, unless `--date` pins
//! one: whole seconds since 1970-01-01T00:00:00 UTC, which may be negative (decision log,
//! 2026-11-17, "The run's date comes from the host, recorded"; [`rtc_at_boot`]). Either way
//! `summary.json` says what it was and where it came from (its `date`), so a run that read the
//! host's date can be run again pinned to it. The flag is starquake's own: the observability
//! spec's headless contract names no flag for a date, and its `--clock fixed` fixes the monotonic
//! clock only.
//!
//! A pinned date is refused, as a usage error, outside [`FIRST_RTC_AT_BOOT`] to
//! [`LAST_RTC_AT_BOOT`] (-9223372036 to 9223372036): the seconds whose nanoseconds fit an `i64`,
//! which is what `time:u` sets its clocks up from. One exception remains inside the range: at
//! the last seconds, the room `time:u` has left for the uptime is under a second
//! (`starquake_services::time::LAST_RTC_AT_BOOT`), so a guest that first asks `time:u` for its
//! clocks' memory later than that is answered `OVERFLOW` and cannot set its clocks up.
//!
//! `--until` takes one condition ([`parse_until`]): `timeout=<n>s`, or `frames=<n>`, which stops
//! the run once `<n>` frames (1 or more) have been presented, with exit code 0. A frame is a
//! present the graphics server hands the GPU, one for each frame its vsync tick latches
//! (`starquake_core::hle::Graphics::frames`); `summary.json`'s `frames` counts the same, so a run
//! that stops at `frames=<n>` says `<n>` there. A present with a fence the GPU reaches is latched
//! only once the host GPU has completed that work, so the guest time at which such a run stops
//! depends on host timing; for presents with no fence it does not
//! (`starquake_core::hle::graphics`, "Frames").
//!
//! PROVISIONAL (walking skeleton, #74). Of the contract's flags only `--until` and `--artifacts`
//! exist; `--script`, `--clock`, `--seed`, `--record`, `--snapshot-every` and the `trace=`
//! condition of `--until` are refused by name, so a script that uses one finds out. The stop
//! condition of a run is the guest calling `ExitProcess`, or the frames `--until frames=<n>`
//! asks for.
//!
//! `inspect` and `verify` have exit codes of their own ([`title::exit`]): 0 done and intact, 1
//! the title is damaged or part of it could not be parsed, 2 the file could not be read or the
//! command line is wrong.
//!
//! Everything here writes to the two streams it is handed, so tests can run a command in
//! process; `main.rs` passes the real stdout and stderr.
//!
//! The emulator's own warnings and errors during a run go to the run's `log.txt`
//! (`starquake_core::artifacts::LOG_FILE`), bounded like the guest log, once
//! [`install_host_log`] has made a `starquake_core::host_log::HostLog` the process's log
//! subscriber; `main.rs` does that first.

use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use starquake_core::host_log::{HostLog, HostLogError};
use starquake_core::loader::read_at::{Positional, ReadAt};
use starquake_core::title::{self as title_api, RomFs};
use starquake_core::{
    ExitReason, FIRST_RTC_AT_BOOT, LAST_RTC_AT_BOOT, RtcAtBoot, RunOptions, RunReport, System,
    UnixSeconds, artifacts, exit_code,
};

pub mod gpu_replay;
pub mod title;

/// What `starquake` prints for `--help`, and after a command line it can't parse.
pub const USAGE: &str = "\
usage: starquake run <program> [--until timeout=<n>s | frames=<n>]
                     [--artifacts <folder>] [--date <seconds>]
       starquake inspect <title.nup>
       starquake verify <title.nup>
       starquake gpu-replay <capture> [--until present=<n>]
                            [--dump ring|render|writes]

run: runs a program, an NRO or a NUP title (the ExeFS of its newest update,
or of its base), and writes what it did to an artifacts folder:
summary.json, trace.txt, events.bin, guest_log.txt, gpu.txt and frames/, plus
crash.json and crash.txt when the guest crashed. Without --artifacts the folder
is made under the system's temporary folder and its path is printed. Without
--until the program runs until it exits or crashes, however long that takes;
--until frames=<n> stops it once it has presented <n> frames. The guest's
date is the host's, read once as the run starts, unless --date pins one:
whole seconds since 1970-01-01T00:00:00 UTC, from -9223372036 to 9223372036,
as in --date 1672531200. summary.json says which date the run used.
exit codes: 0 the program exited or presented its frames, 1 the guest crashed,
2 emulator error, 3 timeout

inspect: prints what a NUP title holds: its packages, the ExeFS of its newest
update (or its base), a summary of its main.npdm and its modules' headers.
verify: checks the SHA-256 of every file of a NUP title and names the first
that does not match.
exit codes: 0 done and intact, 1 the title is damaged or could not be parsed,
2 the file could not be read or the command line is wrong

gpu-replay: replays a GPU capture through the GPU's front and back with no
game and no GPU, and checks the ring and the render stream against it. --until
present=<n> stops after the n-th present; --dump prints the ring entries, the
render commands or the writes after the summary line.
exit codes: 0 replayed to its stop condition, 2 emulator error (an unreadable or
malformed capture, or a replay that diverged)
";

/// The flags of the headless contract that are not built yet.
const UNSUPPORTED_FLAGS: [&str; 5] = [
    "--script",
    "--clock",
    "--seed",
    "--record",
    "--snapshot-every",
];

/// What a `run` command asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunArgs {
    /// The program file: an NRO, or a NUP title.
    pub program: PathBuf,
    /// `--until timeout=<n>s`.
    pub timeout: Option<Duration>,
    /// `--until frames=<n>`.
    pub frames: Option<NonZeroU64>,
    /// `--artifacts <folder>`.
    pub artifacts: Option<PathBuf>,
    /// `--date <seconds>`: the date the run pins. `None`: the host's.
    pub date: Option<UnixSeconds>,
}

/// A parsed command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// `starquake run ...`.
    Run(RunArgs),
    /// `starquake inspect <file>`.
    Inspect(PathBuf),
    /// `starquake verify <file>`.
    Verify(PathBuf),
    /// `starquake gpu-replay ...`.
    GpuReplay(starquake_core::gpu_replay::GpuReplayArgs),
    /// `starquake --help`, `starquake help`, or no arguments at all.
    Help,
}

/// Why a command line could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UsageError {
    /// A `gpu-replay` command line that could not be parsed.
    #[error(transparent)]
    GpuReplay(#[from] gpu_replay::GpuReplayUsage),
    /// The first argument is not a command this tool has.
    #[error("unknown command `{0}`")]
    UnknownCommand(String),
    /// `run` was given no program.
    #[error("`run` needs a program file")]
    MissingProgram,
    /// `run` was given more than one program.
    #[error("`run` takes one program file; `{0}` is a second one")]
    ExtraProgram(String),
    /// `inspect` or `verify` was given no file.
    #[error("`{0}` needs a NUP file")]
    MissingFile(&'static str),
    /// `inspect` or `verify` was given more than one file.
    #[error("`{command}` takes one file; `{extra}` is a second one")]
    ExtraFile {
        /// The command.
        command: &'static str,
        /// The second file.
        extra: String,
    },
    /// A flag this tool has never heard of.
    #[error("unknown flag `{0}`")]
    UnknownFlag(String),
    /// A flag of the headless contract that is not built yet.
    #[error("`{0}` is not supported yet")]
    UnsupportedFlag(String),
    /// A flag that needs a value was the last argument.
    #[error("`{0}` needs a value")]
    MissingValue(&'static str),
    /// A flag was given twice.
    #[error("`{0}` was given twice")]
    Repeated(&'static str),
    /// `--artifacts` with an empty path.
    #[error("`--artifacts` needs a folder; it was given an empty path")]
    EmptyArtifacts,
    /// `--until` with a condition that is not built yet.
    #[error("`--until {0}` is not supported yet; only `timeout=<n>s` and `frames=<n>` are")]
    UnsupportedCondition(String),
    /// `--until frames=` with something other than a whole number of frames from 1.
    #[error(
        "`--until {0}` is not a frame count: write a whole number of frames from 1, as in \
         `frames=600`"
    )]
    BadFrames(String),
    /// `--until timeout=` with something other than a whole number of seconds.
    #[error("`--until {0}` is not a timeout: write a whole number of seconds, as in `timeout=60s`")]
    BadTimeout(String),
    /// `--date` with something other than a whole number of seconds from [`FIRST_RTC_AT_BOOT`]
    /// to [`LAST_RTC_AT_BOOT`].
    #[error(
        "`--date {0}` is not a date: write whole seconds since 1970-01-01T00:00:00 UTC, from \
         {first} to {last}, as in `--date 1672531200`",
        first = FIRST_RTC_AT_BOOT,
        last = LAST_RTC_AT_BOOT
    )]
    BadDate(String),
}

/// An argument as text, with anything that is not UTF-8 replaced (for flags and messages; a
/// program path is used as it was given).
fn text(arg: &OsString) -> String {
    arg.to_string_lossy().into_owned()
}

/// What `--until` stops a run at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Until {
    /// `timeout=<n>s`: after this long.
    Timeout(Duration),
    /// `frames=<n>`: once this many frames have been presented (module docs).
    Frames(NonZeroU64),
}

/// Parses `--until`'s condition: `timeout=<n>s`, whole seconds from 0, or `frames=<n>`, whole
/// frames from 1. Each is ASCII digits only (no sign, no space) and must fit a `u64`.
pub fn parse_until(condition: &str) -> Result<Until, UsageError> {
    if let Some(frames) = condition.strip_prefix("frames=") {
        return Some(frames)
            .filter(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|digits| digits.parse().ok())
            .and_then(NonZeroU64::new)
            .map(Until::Frames)
            .ok_or_else(|| UsageError::BadFrames(condition.to_owned()));
    }
    let Some(seconds) = condition.strip_prefix("timeout=") else {
        return Err(UsageError::UnsupportedCondition(condition.to_owned()));
    };
    seconds
        .strip_suffix('s')
        .filter(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|digits| digits.parse().ok())
        .map(|seconds| Until::Timeout(Duration::from_secs(seconds)))
        .ok_or_else(|| UsageError::BadTimeout(condition.to_owned()))
}

/// Parses `--date`'s value: an optional `-`, then one or more ASCII digits, as seconds since
/// the Unix epoch from [`FIRST_RTC_AT_BOOT`] to [`LAST_RTC_AT_BOOT`] (module docs).
fn parse_date(value: &str) -> Result<UnixSeconds, UsageError> {
    let digits = value.strip_prefix('-').unwrap_or(value);
    Some(digits)
        .filter(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|_| value.parse::<i64>().ok())
        .filter(|seconds| (FIRST_RTC_AT_BOOT..=LAST_RTC_AT_BOOT).contains(seconds))
        .map(UnixSeconds)
        .ok_or_else(|| UsageError::BadDate(value.to_owned()))
}

/// Where a run's date comes from: the date `--date` pinned, or the host's.
#[must_use]
pub fn rtc_at_boot(date: Option<UnixSeconds>) -> RtcAtBoot {
    date.map_or(RtcAtBoot::Host, RtcAtBoot::Pinned)
}

/// Parses the arguments after the program's own name.
pub fn parse(args: &[OsString]) -> Result<Command, UsageError> {
    let Some((command, rest)) = args.split_first() else {
        return Ok(Command::Help);
    };
    match text(command).as_str() {
        "help" | "--help" | "-h" => return Ok(Command::Help),
        "run" => {}
        "inspect" => return parse_file_command("inspect", rest, Command::Inspect),
        "verify" => return parse_file_command("verify", rest, Command::Verify),
        "gpu-replay" => {
            return Ok(gpu_replay::parse(rest)?.map_or(Command::Help, Command::GpuReplay));
        }
        other => return Err(UsageError::UnknownCommand(other.to_owned())),
    }

    let (mut program, mut until, mut artifacts, mut date) = (None, None, None, None);
    let mut rest = rest.iter();
    while let Some(arg) = rest.next() {
        let shown = text(arg);
        // A flag's value is the next argument, or follows an `=` in the same one.
        let (flag, inline) = match shown.split_once('=') {
            Some((flag, value)) if flag.starts_with("--") => (flag, Some(value)),
            _ => (shown.as_str(), None),
        };
        let mut value = |name: &'static str| match inline {
            Some(value) => Ok(OsString::from(value)),
            None => rest.next().cloned().ok_or(UsageError::MissingValue(name)),
        };
        match flag {
            "--help" | "-h" => return Ok(Command::Help),
            "--until" => {
                let condition = text(&value("--until")?);
                if until.replace(parse_until(&condition)?).is_some() {
                    return Err(UsageError::Repeated("--until"));
                }
            }
            "--artifacts" => {
                let folder = value("--artifacts")?;
                // An empty path would write the artifacts into the current directory.
                if folder.is_empty() {
                    return Err(UsageError::EmptyArtifacts);
                }
                if artifacts.replace(PathBuf::from(folder)).is_some() {
                    return Err(UsageError::Repeated("--artifacts"));
                }
            }
            "--date" => {
                let value = text(&value("--date")?);
                if date.replace(parse_date(&value)?).is_some() {
                    return Err(UsageError::Repeated("--date"));
                }
            }
            flag if UNSUPPORTED_FLAGS.contains(&flag) => {
                return Err(UsageError::UnsupportedFlag(flag.to_owned()));
            }
            flag if flag.starts_with('-') && flag.len() > 1 => {
                return Err(UsageError::UnknownFlag(flag.to_owned()));
            }
            _ => {
                if program.replace(PathBuf::from(arg)).is_some() {
                    return Err(UsageError::ExtraProgram(shown));
                }
            }
        }
    }
    Ok(Command::Run(RunArgs {
        program: program.ok_or(UsageError::MissingProgram)?,
        timeout: match until {
            Some(Until::Timeout(timeout)) => Some(timeout),
            _ => None,
        },
        frames: match until {
            Some(Until::Frames(frames)) => Some(frames),
            _ => None,
        },
        artifacts,
        date,
    }))
}

/// Parses the arguments of `inspect` or `verify` (`command`): one file, which `make` turns into
/// the command. An argument starting with `-` (but `-` alone) is a flag, and the only flags are
/// `--help` and `-h`.
fn parse_file_command(
    command: &'static str,
    rest: &[OsString],
    make: fn(PathBuf) -> Command,
) -> Result<Command, UsageError> {
    let mut file = None;
    for arg in rest {
        let shown = text(arg);
        match shown.as_str() {
            "--help" | "-h" => return Ok(Command::Help),
            flag if flag.starts_with('-') && flag.len() > 1 => {
                let flag = flag.split_once('=').map_or(flag, |(name, _)| name);
                return Err(UsageError::UnknownFlag(flag.to_owned()));
            }
            _ => {
                if file.replace(PathBuf::from(arg)).is_some() {
                    return Err(UsageError::ExtraFile {
                        command,
                        extra: shown,
                    });
                }
            }
        }
    }
    file.map(make).ok_or(UsageError::MissingFile(command))
}

/// A folder for a run that named none: under the system's temporary folder, outside any
/// repository, and distinct per run.
fn default_artifacts_folder() -> PathBuf {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    std::env::temp_dir()
        .join("starquake")
        .join(format!("run-{seconds}-{}", std::process::id()))
}

/// One line saying how the run ended.
fn describe(report: &RunReport) -> String {
    match &report.exit {
        ExitReason::ProcessExited => "the program exited".to_owned(),
        ExitReason::FramesReached => format!(
            "the program presented the {} frames it was to stop at",
            report.graphics.frames
        ),
        ExitReason::Timeout => "the timeout passed".to_owned(),
        ExitReason::EmulatorError(message) => format!("emulator error: {message}"),
        ExitReason::GuestCrashed(crash) => format!(
            "the guest crashed: {}, at {}",
            crash.exception_text(),
            crash.instruction_text()
        ),
    }
}

/// Loads the program file at `path` ([`load_program`]). The error is the line the run's summary
/// carries.
fn load(path: &Path) -> Result<Loaded, String> {
    let unreadable = |error: std::io::Error| format!("could not read {}: {error}", path.display());
    let file = File::open(path).map_err(unreadable)?;
    load_program(file).map_err(|failure| match failure {
        LoadFailure::Unreadable(error) => unreadable(error),
        LoadFailure::Refused(message) => format!("{}: {message}", path.display()),
    })
}

/// Why [`load_program`] could not load a program.
#[derive(Debug)]
enum LoadFailure {
    /// The source could not be read.
    Unreadable(std::io::Error),
    /// The source was read, but is not a program that loads: what the loader said.
    Refused(String),
}

/// A loaded program, and the title's RomFS when it is a NUP title that has one.
#[derive(Debug, PartialEq, Eq)]
struct Loaded {
    system: System,
    romfs: Option<RomFs>,
}

/// Loads the program in `source`: a NUP title when its first bytes are a NUP container's
/// (`starquake_core::title::is_nup`), opened through `starquake_core::title::open` so that only
/// its table of contents and its ExeFS's `main.npdm` and modules are read while loading (its
/// RomFS is read later, as the guest reads it), and an NRO, read whole, otherwise.
fn load_program<S: ReadAt + 'static>(source: S) -> Result<Loaded, LoadFailure> {
    let mut cursor = Positional::new(source);
    let mut prefix = Vec::with_capacity(title_api::PREFIX_LEN);
    let prefix_len = u64::try_from(title_api::PREFIX_LEN).unwrap_or(u64::MAX);
    (&mut cursor)
        .take(prefix_len)
        .read_to_end(&mut prefix)
        .map_err(LoadFailure::Unreadable)?;
    cursor.rewind().map_err(LoadFailure::Unreadable)?;
    if title_api::is_nup(&prefix) {
        let title = title_api::open(cursor.into_inner())
            .map_err(|error| LoadFailure::Refused(error.to_string()))?;
        return Ok(Loaded {
            system: title.system,
            romfs: title.romfs,
        });
    }
    let mut bytes = Vec::new();
    cursor
        .read_to_end(&mut bytes)
        .map_err(LoadFailure::Unreadable)?;
    let system = System::new(&bytes).map_err(|error| LoadFailure::Refused(error.to_string()))?;
    Ok(Loaded {
        system,
        romfs: None,
    })
}

/// What the run `args` asks for is asked to do, with the command line that started it and the
/// title's RomFS, if it has one.
fn run_options(args: &RunArgs, command_line: Vec<String>, romfs: Option<RomFs>) -> RunOptions {
    RunOptions {
        timeout: args.timeout,
        command_line,
        sd_card: None,
        romfs,
        rtc: rtc_at_boot(args.date),
        environment: starquake_core::EnvSource::Live,
        until_frames: args.frames,
    }
}

/// Runs the program `args` names and writes its artifacts. Returns the process exit code.
fn run(
    args: &RunArgs,
    command_line: Vec<String>,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> u8 {
    let started = Instant::now();
    let mut report = match load(&args.program) {
        Err(message) => RunReport::emulator_error(message, started.elapsed()),
        Ok(Loaded { system, romfs }) => system.run(&run_options(args, command_line, romfs)),
    };
    if let Some(log) = HOST_LOG.get() {
        (report.host_log, report.host_log_dropped) = log.take();
    }

    let folder = args
        .artifacts
        .clone()
        .unwrap_or_else(default_artifacts_folder);
    conclude(&report, &folder, stdout, stderr)
}

/// Finishes a run: prints what the guest wrote, writes `report`'s artifacts into `folder`, and
/// prints one line saying how the run ended. Returns the process exit code, which is the run's
/// own unless the artifacts could not be written: that is an emulator error whatever the run
/// did.
pub fn conclude(
    report: &RunReport,
    folder: &Path,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> u8 {
    // A stream that can't be written to (a closed pipe) must not change the exit code.
    for line in &report.guest_log {
        let _ = writeln!(stdout, "[guest] {}", line.escape_ascii());
    }
    if report.guest_log_dropped > 0 {
        let _ = writeln!(
            stdout,
            "starquake: the guest log was cut short: {} later lines were not kept",
            report.guest_log_dropped
        );
    }
    if report.trace_events_dropped > 0 {
        let _ = writeln!(
            stdout,
            "starquake: the trace was cut short: {} later events are not in it",
            report.trace_events_dropped
        );
    }
    // Where each thread was, when the run ended at its timeout (empty otherwise).
    for thread in &report.threads {
        let _ = writeln!(stdout, "starquake: at the timeout, {}", thread.text());
    }
    let ended = describe(report);
    match artifacts::write(folder, report) {
        Ok(()) => {
            let code = report.exit_code();
            let _ = writeln!(
                stdout,
                "starquake: {ended} (exit code {code}); artifacts in {}",
                folder.display()
            );
            code
        }
        Err(error) => {
            let code = exit_code::EMULATOR_ERROR;
            let _ = writeln!(stderr, "starquake: {error}");
            let _ = writeln!(
                stdout,
                "starquake: {ended}, but the artifacts could not be written (exit code {code})"
            );
            code
        }
    }
}

/// The emulator's log of its own warnings, once [`install_host_log`] has installed it.
static HOST_LOG: OnceLock<HostLog> = OnceLock::new();

/// Makes a [`HostLog`] the log subscriber of every thread of the process, so that a run's
/// artifacts hold the emulator's warnings and errors (`log.txt`): each `run` puts what the log
/// kept, since it was installed or since the last run took it, into its report. The binary does
/// this once, before [`execute`]; without it, `log.txt` is empty. It fails if the process
/// already has a log subscriber.
pub fn install_host_log() -> Result<(), HostLogError> {
    let log = HostLog::new();
    log.install()?;
    HOST_LOG
        .set(log)
        .map_err(|_| HostLogError::AlreadyInstalled)
}

/// Runs the command line `args` (the program's own name first, as `std::env::args_os` gives it)
/// and returns the process exit code.
pub fn execute(args: &[OsString], stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    match parse(args.get(1..).unwrap_or_default()) {
        Ok(Command::Help) => {
            let _ = write!(stdout, "{USAGE}");
            exit_code::STOP_CONDITION
        }
        Ok(Command::Run(run_args)) => {
            let command_line = args.iter().map(text).collect();
            run(&run_args, command_line, stdout, stderr)
        }
        Ok(Command::Inspect(file)) => title::inspect(&file, stdout, stderr),
        Ok(Command::Verify(file)) => title::verify(&file, stdout, stderr),
        Ok(Command::GpuReplay(replay)) => gpu_replay::run(&replay, stdout, stderr),
        Err(error) => {
            let _ = write!(stderr, "starquake: {error}\n\n{USAGE}");
            exit_code::EMULATOR_ERROR
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(words: &[&str]) -> Vec<OsString> {
        words.iter().map(OsString::from).collect()
    }

    fn run_args(program: &str, timeout: Option<u64>, artifacts: Option<&str>) -> Command {
        dated_run_args(program, timeout, artifacts, None)
    }

    fn dated_run_args(
        program: &str,
        timeout: Option<u64>,
        artifacts: Option<&str>,
        date: Option<i64>,
    ) -> Command {
        Command::Run(RunArgs {
            program: PathBuf::from(program),
            timeout: timeout.map(Duration::from_secs),
            frames: None,
            artifacts: artifacts.map(PathBuf::from),
            date: date.map(UnixSeconds),
        })
    }

    #[test]
    fn run_takes_a_program_and_its_two_flags_in_any_order() {
        let cases = [
            (&["run", "a.nro"][..], run_args("a.nro", None, None)),
            (
                &["run", "a.nro", "--until", "timeout=60s"],
                run_args("a.nro", Some(60), None),
            ),
            (
                &["run", "--artifacts", "out", "a.nro"],
                run_args("a.nro", None, Some("out")),
            ),
            (
                &[
                    "run",
                    "--until",
                    "timeout=0s",
                    "a.nro",
                    "--artifacts",
                    "out",
                ],
                run_args("a.nro", Some(0), Some("out")),
            ),
            (
                &["run", "--artifacts=out dir", "--until=timeout=7s", "a.nro"],
                run_args("a.nro", Some(7), Some("out dir")),
            ),
            // A program whose name holds an `=` is still a program.
            (&["run", "a=b.nro"], run_args("a=b.nro", None, None)),
            (&["run", "-"], run_args("-", None, None)),
        ];
        for (words, expected) in cases {
            assert_eq!(parse(&args(words)), Ok(expected), "{words:?}");
        }
    }

    #[test]
    fn run_takes_a_date_in_either_form_beside_the_other_flags() {
        let cases = [
            (
                &["run", "a.nro", "--date", "1709164800"][..],
                dated_run_args("a.nro", None, None, Some(1_709_164_800)),
            ),
            (
                &["run", "--date=-1", "a.nro"],
                dated_run_args("a.nro", None, None, Some(-1)),
            ),
            // A negative value after a space is the flag's value, not a flag.
            (
                &["run", "--date", "-86399", "a.nro", "--until", "timeout=5s"],
                dated_run_args("a.nro", Some(5), None, Some(-86_399)),
            ),
            (
                &["run", "--artifacts", "out", "--date", "0", "a.nro"],
                dated_run_args("a.nro", None, Some("out"), Some(0)),
            ),
        ];
        for (words, expected) in cases {
            assert_eq!(parse(&args(words)), Ok(expected), "{words:?}");
        }
    }

    #[test]
    fn a_date_is_whole_seconds_whose_nanoseconds_fit_an_i64() {
        let last = LAST_RTC_AT_BOOT.to_string();
        let first = FIRST_RTC_AT_BOOT.to_string();
        let past_last = (LAST_RTC_AT_BOOT + 1).to_string();
        let before_first = (FIRST_RTC_AT_BOOT - 1).to_string();
        let max = i64::MAX.to_string();
        let min = i64::MIN.to_string();
        for (good, expected) in [
            ("0", 0),
            ("-0", 0),
            ("007", 7),
            ("1672531200", 1_672_531_200),
            ("-1", -1),
            // Exactly at each end.
            (last.as_str(), LAST_RTC_AT_BOOT),
            (first.as_str(), FIRST_RTC_AT_BOOT),
        ] {
            assert_eq!(parse_date(good), Ok(UnixSeconds(expected)), "{good}");
        }
        for bad in [
            "",
            "-",
            "--1",
            "+1",
            " 1",
            "1 ",
            "1.5",
            "1e9",
            "2024-02-29",
            "0x10",
            // One past each end.
            past_last.as_str(),
            before_first.as_str(),
            // Whole seconds that fit an `i64` but whose nanoseconds do not.
            max.as_str(),
            min.as_str(),
            // And seconds past an `i64`.
            "9223372036854775808",
            "-9223372036854775809",
        ] {
            assert_eq!(
                parse_date(bad),
                Err(UsageError::BadDate(bad.to_owned())),
                "{bad}"
            );
        }
        assert_eq!(
            UsageError::BadDate("2024-02-29".into()).to_string(),
            "`--date 2024-02-29` is not a date: write whole seconds since \
             1970-01-01T00:00:00 UTC, from -9223372036 to 9223372036, as in \
             `--date 1672531200`"
        );
        // Through the command line: the run is refused before anything runs.
        assert_eq!(
            parse(&args(&["run", "a.nro", "--date", &past_last])),
            Err(UsageError::BadDate(past_last.clone()))
        );
    }

    #[test]
    fn a_run_pins_the_date_it_was_given_and_reads_the_hosts_otherwise() {
        assert_eq!(rtc_at_boot(None), RtcAtBoot::Host);
        for date in [i64::MIN, -1, 0, 1_709_164_800] {
            assert_eq!(
                rtc_at_boot(Some(UnixSeconds(date))),
                RtcAtBoot::Pinned(UnixSeconds(date))
            );
        }
        // And the run is asked for exactly that.
        let Ok(Command::Run(dated)) = parse(&args(&["run", "a.nro", "--date", "-5"])) else {
            unreachable!("not a run");
        };
        let options = run_options(&dated, vec!["x".into()], None);
        assert_eq!(options.rtc, RtcAtBoot::Pinned(UnixSeconds(-5)));
        assert_eq!(options.command_line, ["x"]);
        let Ok(Command::Run(undated)) = parse(&args(&["run", "a.nro"])) else {
            unreachable!("not a run");
        };
        assert_eq!(run_options(&undated, Vec::new(), None).rtc, RtcAtBoot::Host);
    }

    #[test]
    fn help_is_the_answer_to_no_arguments_and_to_asking() {
        for words in [
            &[][..],
            &["help"],
            &["--help"],
            &["-h"],
            &["run", "--help"],
            &["run", "a.nro", "-h"],
        ] {
            assert_eq!(parse(&args(words)), Ok(Command::Help), "{words:?}");
        }
    }

    #[test]
    fn each_malformed_command_line_has_its_own_error() {
        let cases = [
            (
                &["boot", "a.nro"][..],
                UsageError::UnknownCommand("boot".into()),
            ),
            (&["run"], UsageError::MissingProgram),
            (
                &["run", "--until", "timeout=1s"],
                UsageError::MissingProgram,
            ),
            (
                &["run", "a.nro", "b.nro"],
                UsageError::ExtraProgram("b.nro".into()),
            ),
            (
                &["run", "a.nro", "--fast"],
                UsageError::UnknownFlag("--fast".into()),
            ),
            (
                &["run", "a.nro", "-x"],
                UsageError::UnknownFlag("-x".into()),
            ),
            (
                &["run", "a.nro", "--until"],
                UsageError::MissingValue("--until"),
            ),
            (
                &["run", "a.nro", "--artifacts"],
                UsageError::MissingValue("--artifacts"),
            ),
            (
                &[
                    "run",
                    "a.nro",
                    "--until",
                    "timeout=1s",
                    "--until",
                    "timeout=2s",
                ],
                UsageError::Repeated("--until"),
            ),
            (
                &["run", "a.nro", "--artifacts", "x", "--artifacts=y"],
                UsageError::Repeated("--artifacts"),
            ),
            (
                &["run", "a.nro", "--date", "1", "--date=1"],
                UsageError::Repeated("--date"),
            ),
            (
                &["run", "a.nro", "--date"],
                UsageError::MissingValue("--date"),
            ),
            (
                &["run", "a.nro", "--date", "today"],
                UsageError::BadDate("today".into()),
            ),
            (
                &["run", "a.nro", "--date="],
                UsageError::BadDate(String::new()),
            ),
            (
                &["run", "a.nro", "--until", "trace=svc"],
                UsageError::UnsupportedCondition("trace=svc".into()),
            ),
            (
                &["run", "a.nro", "--until", "frames=0"],
                UsageError::BadFrames("frames=0".into()),
            ),
            // An empty folder would put the artifacts in the current directory.
            (
                &["run", "a.nro", "--artifacts", ""],
                UsageError::EmptyArtifacts,
            ),
            (
                &["run", "a.nro", "--artifacts="],
                UsageError::EmptyArtifacts,
            ),
            (
                &["run", "a.nro", "--until", "60s"],
                UsageError::UnsupportedCondition("60s".into()),
            ),
        ];
        for (words, expected) in cases {
            assert_eq!(parse(&args(words)), Err(expected), "{words:?}");
        }
    }

    #[test]
    fn a_timeout_is_a_whole_number_of_seconds_with_its_unit() {
        assert_eq!(
            parse_until("timeout=60s"),
            Ok(Until::Timeout(Duration::from_secs(60)))
        );
        assert_eq!(
            parse_until(&format!("timeout={}s", u64::MAX)),
            Ok(Until::Timeout(Duration::from_secs(u64::MAX)))
        );
        for bad in [
            "timeout=",
            "timeout=s",
            "timeout=60",
            "timeout=60ms",
            "timeout=1.5s",
            "timeout=-1s",
            "timeout=+1s",
            "timeout= 1s",
            // One more than fits.
            "timeout=18446744073709551616s",
        ] {
            assert_eq!(
                parse_until(bad),
                Err(UsageError::BadTimeout(bad.to_owned())),
                "{bad}"
            );
        }
    }

    #[test]
    fn a_frame_count_is_a_whole_number_of_frames_from_one() {
        let frames = |n: u64| Ok(Until::Frames(NonZeroU64::new(n).unwrap()));
        // The first count allowed, and the last.
        assert_eq!(parse_until("frames=1"), frames(1));
        assert_eq!(parse_until("frames=600"), frames(600));
        assert_eq!(parse_until("frames=007"), frames(7));
        assert_eq!(
            parse_until(&format!("frames={}", u64::MAX)),
            frames(u64::MAX)
        );
        for bad in [
            // One below the first.
            "frames=0",
            "frames=000",
            "frames=",
            "frames=-1",
            "frames=+1",
            "frames= 1",
            "frames=1 ",
            "frames=1.5",
            "frames=1f",
            "frames=0x10",
            // One more than fits.
            "frames=18446744073709551616",
        ] {
            assert_eq!(
                parse_until(bad),
                Err(UsageError::BadFrames(bad.to_owned())),
                "{bad}"
            );
        }
        assert_eq!(
            UsageError::BadFrames("frames=0".into()).to_string(),
            "`--until frames=0` is not a frame count: write a whole number of frames from 1, as \
             in `frames=600`"
        );
        assert_eq!(
            UsageError::UnsupportedCondition("trace=svc".into()).to_string(),
            "`--until trace=svc` is not supported yet; only `timeout=<n>s` and `frames=<n>` are"
        );
        // A prefix is not a condition.
        for other in ["frame=1", "Frames=1", "frames", "timeout", "60s"] {
            assert_eq!(
                parse_until(other),
                Err(UsageError::UnsupportedCondition(other.to_owned())),
                "{other}"
            );
        }
    }

    #[test]
    fn a_run_takes_a_frame_count_in_either_form_and_asks_the_run_for_it() {
        let frames = |n: u64, artifacts: Option<&str>| {
            Command::Run(RunArgs {
                program: PathBuf::from("a.nro"),
                timeout: None,
                frames: NonZeroU64::new(n),
                artifacts: artifacts.map(PathBuf::from),
                date: None,
            })
        };
        let cases = [
            (
                &["run", "a.nro", "--until", "frames=3"][..],
                frames(3, None),
            ),
            (&["run", "--until=frames=4", "a.nro"], frames(4, None)),
            (
                &["run", "--artifacts", "out", "--until", "frames=5", "a.nro"],
                frames(5, Some("out")),
            ),
        ];
        for (words, expected) in cases {
            assert_eq!(parse(&args(words)), Ok(expected), "{words:?}");
        }
        // The run is asked for exactly that, and no timeout; a timeout asks for no frame count.
        let Ok(Command::Run(counted)) = parse(&args(&["run", "a.nro", "--until", "frames=9"]))
        else {
            unreachable!("not a run");
        };
        let options = run_options(&counted, Vec::new(), None);
        assert_eq!(options.until_frames, NonZeroU64::new(9));
        assert_eq!(options.timeout, None);
        let Ok(Command::Run(timed)) = parse(&args(&["run", "a.nro", "--until", "timeout=9s"]))
        else {
            unreachable!("not a run");
        };
        let options = run_options(&timed, Vec::new(), None);
        assert_eq!(options.until_frames, None);
        assert_eq!(options.timeout, Some(Duration::from_secs(9)));
        let Ok(Command::Run(plain)) = parse(&args(&["run", "a.nro"])) else {
            unreachable!("not a run");
        };
        let options = run_options(&plain, Vec::new(), None);
        assert_eq!((options.until_frames, options.timeout), (None, None));
        // One condition only, whichever two are given, in either order and form.
        for words in [
            &[
                "run",
                "a.nro",
                "--until",
                "frames=1",
                "--until",
                "timeout=1s",
            ][..],
            &["run", "a.nro", "--until=timeout=1s", "--until=frames=2"],
            &["run", "a.nro", "--until=frames=1", "--until=frames=1"],
        ] {
            assert_eq!(
                parse(&args(words)),
                Err(UsageError::Repeated("--until")),
                "{words:?}"
            );
        }
    }

    #[test]
    fn a_run_that_presented_its_frames_says_so_with_exit_code_0() {
        let report = RunReport {
            exit: ExitReason::FramesReached,
            graphics: starquake_core::report::GraphicsReport {
                frames: 3,
                ..Default::default()
            },
            ..RunReport::emulator_error(String::new(), Duration::ZERO)
        };
        assert_eq!(
            describe(&report),
            "the program presented the 3 frames it was to stop at"
        );
        assert_eq!(report.exit_code(), exit_code::STOP_CONDITION);
    }

    #[test]
    fn the_flags_of_the_contract_that_are_not_built_are_refused_by_name() {
        for flag in UNSUPPORTED_FLAGS {
            assert_eq!(
                parse(&args(&["run", "a.nro", flag, "x"])),
                Err(UsageError::UnsupportedFlag(flag.to_owned())),
                "{flag}"
            );
        }
        assert_eq!(
            UsageError::UnsupportedFlag("--seed".into()).to_string(),
            "`--seed` is not supported yet"
        );
    }

    #[test]
    fn help_goes_to_stdout_with_code_0_and_a_usage_error_to_stderr_with_code_2() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        assert_eq!(execute(&args(&["starquake"]), &mut out, &mut err), 0);
        assert_eq!(String::from_utf8(out).unwrap(), USAGE);
        assert_eq!(err, b"");

        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = execute(&args(&["starquake", "run"]), &mut out, &mut err);
        assert_eq!(code, exit_code::EMULATOR_ERROR);
        assert_eq!(out, b"");
        assert_eq!(
            String::from_utf8(err).unwrap(),
            format!("starquake: `run` needs a program file\n\n{USAGE}")
        );
    }

    #[test]
    fn inspect_and_verify_take_exactly_one_file() {
        type Make = fn(PathBuf) -> Command;
        let commands: [(&str, Make); 2] =
            [("inspect", Command::Inspect), ("verify", Command::Verify)];
        for (command, make) in commands {
            let parsed = |words: &[&str]| {
                let mut line = vec![command];
                line.extend_from_slice(words);
                parse(&args(&line))
            };
            assert_eq!(parsed(&["t.nup"]), Ok(make(PathBuf::from("t.nup"))));
            // `-` alone and names holding `=` are files.
            assert_eq!(parsed(&["-"]), Ok(make(PathBuf::from("-"))));
            assert_eq!(parsed(&["a=b.nup"]), Ok(make(PathBuf::from("a=b.nup"))));
            assert_eq!(parsed(&["--help"]), Ok(Command::Help));
            assert_eq!(parsed(&["t.nup", "-h"]), Ok(Command::Help));
            assert_eq!(parsed(&[]), Err(UsageError::MissingFile(command)));
            assert_eq!(
                parsed(&["a.nup", "b.nup"]),
                Err(UsageError::ExtraFile {
                    command,
                    extra: "b.nup".into()
                })
            );
            assert_eq!(
                parsed(&["t.nup", "--deep"]),
                Err(UsageError::UnknownFlag("--deep".into()))
            );
            assert_eq!(
                parsed(&["--until=timeout=1s", "t.nup"]),
                Err(UsageError::UnknownFlag("--until".into()))
            );
            assert_eq!(parsed(&["-x"]), Err(UsageError::UnknownFlag("-x".into())));
        }
        assert_eq!(
            UsageError::MissingFile("verify").to_string(),
            "`verify` needs a NUP file"
        );
        assert_eq!(
            UsageError::ExtraFile {
                command: "inspect",
                extra: "b.nup".into()
            }
            .to_string(),
            "`inspect` takes one file; `b.nup` is a second one"
        );
    }

    #[test]
    fn a_usage_error_of_inspect_or_verify_exits_with_the_unreadable_code() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = execute(&args(&["starquake", "verify"]), &mut out, &mut err);
        assert_eq!(code, title::exit::UNREADABLE);
        assert_eq!(code, exit_code::EMULATOR_ERROR);
        assert_eq!(out, b"");
        assert_eq!(
            String::from_utf8(err).unwrap(),
            format!("starquake: `verify` needs a NUP file\n\n{USAGE}")
        );
    }

    /// A title in memory whose reads fail at and past `end`, as a disk that gives out after the
    /// title's ExeFS: a read that starts before `end` returns at most the bytes before it.
    struct FailingTail {
        inner: Vec<u8>,
        end: u64,
    }

    impl ReadAt for FailingTail {
        fn read_at(&self, at: u64, buf: &mut [u8]) -> std::io::Result<usize> {
            if at >= self.end {
                return Err(std::io::Error::other("read past the ExeFS"));
            }
            let room = usize::try_from(self.end - at).unwrap_or(usize::MAX);
            let len = buf.len().min(room);
            self.inner.read_at(at, &mut buf[..len])
        }

        fn size(&self) -> std::io::Result<u64> {
            self.inner.size()
        }
    }

    #[test]
    fn a_title_is_streamed_and_never_read_past_its_exefs() {
        use starquake_loader::exefs::{MAIN_MODULE, NPDM_FILE};
        use starquake_loader::nup::PackageKind;
        use starquake_loader::nup::build::{
            self, Content, EXEFS_DIR_NAME, PackageSpec, ROMFS_DIR_NAME, Tree,
        };

        const NSO: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nso");
        const NPDM: &[u8] = include_bytes!("../../../tests/homebrew/hello/main.npdm");
        let romfs_file = format!("{ROMFS_DIR_NAME}/data");
        let tree = [(NPDM_FILE, NPDM), (MAIN_MODULE, NSO)]
            .into_iter()
            .fold(Tree::new(), |tree, (name, bytes)| {
                tree.with_file(&format!("{EXEFS_DIR_NAME}/{name}"), Content::raw(bytes))
                    .unwrap()
            })
            .with_file(&romfs_file, Content::raw(vec![0x5a; 0x1000]))
            .unwrap();
        let base = PackageSpec::new(PackageKind::Base, 0x0100_0000_0000_5000, 0, tree);
        let built = build::build(&[base]).unwrap();

        // The end of the ExeFS's last extent, and the RomFS file placed after it.
        let extent = |index: usize| built.layout.extents[index];
        let exefs_end = built
            .layout
            .files
            .iter()
            .filter(|((_, path), _)| path.starts_with(&format!("{EXEFS_DIR_NAME}/")))
            .map(|(_, &index)| extent(index).0 + extent(index).1)
            .max()
            .unwrap();
        let (romfs_at, romfs_len) = extent(built.layout.files[&(0, romfs_file)]);
        assert!(romfs_at >= exefs_end && romfs_len > 0);
        assert_eq!(built.bytes.len(), romfs_at + romfs_len);

        let whole = System::from_nup(std::io::Cursor::new(built.bytes.clone())).unwrap();
        let source = FailingTail {
            inner: built.bytes,
            end: u64::try_from(exefs_end).unwrap(),
        };
        let loaded = load_program(source).unwrap();
        assert_eq!(loaded.system, whole);
        // The RomFS is there, laid out from the table of contents; reading its file's data is
        // what reaches past the ExeFS, and fails here.
        let romfs = loaded.romfs.unwrap();
        assert!(romfs.size() > u64::try_from(romfs_len).unwrap());
        let mut all = vec![0u8; usize::try_from(romfs.size()).unwrap()];
        let error = romfs.read_at(0, &mut all).unwrap_err();
        assert!(error.to_string().contains("read past the ExeFS"), "{error}");
    }
}
