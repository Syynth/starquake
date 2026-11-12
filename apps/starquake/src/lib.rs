//! The starquake command line: `starquake run <program>`, the headless contract of
//! `docs/specs/observability.md` as far as the walking skeleton goes, and `starquake inspect`
//! and `starquake verify`, which read a NUP title ([`title`]).
//!
//! ```text
//! starquake run <program> [--until timeout=<n>s] [--artifacts <folder>]
//! starquake inspect <title.nup>
//! starquake verify <title.nup>
//! ```
//!
//! The process exit code says how the run ended (`starquake_core::exit_code`): 0 the program
//! exited, 1 the guest crashed, 2 an emulator error, 3 the timeout passed. A command line this
//! tool can't parse, a program file it can't read and an artifacts folder it can't write are
//! emulator errors too: the contract has no other code for them.
//!
//! PROVISIONAL (walking skeleton, #74). Of the contract's flags only `--until timeout=<n>s` and
//! `--artifacts` exist; `--script`, `--clock`, `--seed`, `--record`, `--snapshot-every` and the
//! other `--until` conditions are refused by name, so a script that uses one finds out. The stop
//! condition of a run is the guest calling `ExitProcess`.
//!
//! `inspect` and `verify` have exit codes of their own ([`title::exit`]): 0 done and intact, 1
//! the title is damaged or part of it could not be parsed, 2 the file could not be read or the
//! command line is wrong.
//!
//! Everything here writes to the two streams it is handed, so tests can run a command in
//! process; `main.rs` passes the real stdout and stderr.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use starquake_core::{ExitReason, RunOptions, RunReport, System, artifacts, exit_code};

pub mod title;

/// What `starquake` prints for `--help`, and after a command line it can't parse.
pub const USAGE: &str = "\
usage: starquake run <program> [--until timeout=<n>s] [--artifacts <folder>]
       starquake inspect <title.nup>
       starquake verify <title.nup>

run: runs an NRO program and writes what it did to an artifacts folder:
summary.json, trace.txt, events.bin and guest_log.txt, plus crash.json and
crash.txt when the guest crashed. Without --artifacts the folder is made under
the system's temporary folder and its path is printed. Without --until the
program runs until it exits or crashes, however long that takes.
exit codes: 0 the program exited, 1 the guest crashed, 2 emulator error, 3 timeout

inspect: prints what a NUP title holds: its packages, the ExeFS of its newest
update (or its base), a summary of its main.npdm and its modules' headers.
verify: checks the SHA-256 of every file of a NUP title and names the first
that does not match.
exit codes: 0 done and intact, 1 the title is damaged or could not be parsed,
2 the file could not be read or the command line is wrong
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
    /// The program file.
    pub program: PathBuf,
    /// `--until timeout=<n>s`.
    pub timeout: Option<Duration>,
    /// `--artifacts <folder>`.
    pub artifacts: Option<PathBuf>,
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
    /// `starquake --help`, `starquake help`, or no arguments at all.
    Help,
}

/// Why a command line could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UsageError {
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
    #[error("`--until {0}` is not supported yet; only `timeout=<n>s` is")]
    UnsupportedCondition(String),
    /// `--until timeout=` with something other than a whole number of seconds.
    #[error("`--until {0}` is not a timeout: write a whole number of seconds, as in `timeout=60s`")]
    BadTimeout(String),
}

/// An argument as text, with anything that is not UTF-8 replaced (for flags and messages; a
/// program path is used as it was given).
fn text(arg: &OsString) -> String {
    arg.to_string_lossy().into_owned()
}

/// Parses `timeout=<n>s`.
fn parse_until(condition: &str) -> Result<Duration, UsageError> {
    let Some(seconds) = condition.strip_prefix("timeout=") else {
        return Err(UsageError::UnsupportedCondition(condition.to_owned()));
    };
    seconds
        .strip_suffix('s')
        .filter(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|digits| digits.parse().ok())
        .map(Duration::from_secs)
        .ok_or_else(|| UsageError::BadTimeout(condition.to_owned()))
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
        other => return Err(UsageError::UnknownCommand(other.to_owned())),
    }

    let (mut program, mut timeout, mut artifacts) = (None, None, None);
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
                if timeout.replace(parse_until(&condition)?).is_some() {
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
        timeout,
        artifacts,
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
        ExitReason::Timeout => "the timeout passed".to_owned(),
        ExitReason::EmulatorError(message) => format!("emulator error: {message}"),
        ExitReason::GuestCrashed(crash) => format!(
            "the guest crashed: {}, at {}",
            crash.exception_text(),
            crash.instruction_text()
        ),
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
    let options = RunOptions {
        timeout: args.timeout,
        command_line,
    };
    let failed = |message: String| RunReport::emulator_error(message, started.elapsed());
    let report = match std::fs::read(&args.program) {
        Err(error) => failed(format!(
            "could not read {}: {error}",
            args.program.display()
        )),
        Ok(bytes) => match System::new(&bytes) {
            Err(error) => failed(format!("{}: {error}", args.program.display())),
            Ok(system) => system.run(&options),
        },
    };

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
        Command::Run(RunArgs {
            program: PathBuf::from(program),
            timeout: timeout.map(Duration::from_secs),
            artifacts: artifacts.map(PathBuf::from),
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
                &["run", "a.nro", "--until", "frames=600"],
                UsageError::UnsupportedCondition("frames=600".into()),
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
        assert_eq!(parse_until("timeout=60s"), Ok(Duration::from_secs(60)));
        assert_eq!(
            parse_until(&format!("timeout={}s", u64::MAX)),
            Ok(Duration::from_secs(u64::MAX))
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
}
