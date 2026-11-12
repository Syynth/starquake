//! `starquake run` end to end: a program file goes in, an exit code and an artifacts folder
//! come out.
//!
//! The command runs in this process through `starquake::execute` (the test binary is signed with
//! the hypervisor entitlement by the workspace's cargo runner, like every test binary), and once
//! as a real child process, to check the exit status the shell sees.
//!
//! Tests that need a VM need a Mac with Apple Silicon and skip themselves only when the `CI`
//! environment variable is set; the rest run on every host.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use starquake_core::artifacts::{
    CRASH_JSON_FILE, CRASH_TEXT_FILE, EVENTS_FILE, GUEST_LOG_FILE, SUMMARY_FILE, TRACE_FILE,
};
use starquake_core::exit_code;
use starquake_testkit::programs;

/// Helpers, in a `cfg(test)` module so they may unwrap like the tests do.
#[cfg(test)]
mod support {
    use super::*;

    /// What a command printed and returned.
    pub(crate) struct Ran {
        pub(crate) code: u8,
        pub(crate) stdout: String,
        pub(crate) stderr: String,
    }

    /// An empty scratch folder for the test called `name`.
    pub(crate) fn scratch(name: &str) -> PathBuf {
        let folder = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("run")
            .join(name);
        if folder.exists() {
            std::fs::remove_dir_all(&folder).unwrap();
        }
        std::fs::create_dir_all(&folder).unwrap();
        folder
    }

    /// Writes a program file into `folder` and returns its path.
    pub(crate) fn program(folder: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = folder.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// Runs `starquake <args>` in this process.
    pub(crate) fn starquake(args: &[&OsString]) -> Ran {
        let mut line = vec![OsString::from("starquake")];
        line.extend(args.iter().map(|arg| (*arg).clone()));
        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        let code = starquake::execute(&line, &mut stdout, &mut stderr);
        Ran {
            code,
            stdout: String::from_utf8(stdout).unwrap(),
            stderr: String::from_utf8(stderr).unwrap(),
        }
    }

    /// `run <program> --artifacts <folder>/artifacts`, plus `extra`; returns what it printed and
    /// the artifacts folder.
    pub(crate) fn run(program: &Path, extra: &[&str]) -> (Ran, PathBuf) {
        let artifacts = program.parent().unwrap().join("artifacts");
        let mut args: Vec<OsString> = vec![
            "run".into(),
            program.into(),
            "--artifacts".into(),
            artifacts.clone().into(),
        ];
        args.extend(extra.iter().map(OsString::from));
        (starquake(&args.iter().collect::<Vec<_>>()), artifacts)
    }

    pub(crate) fn text(folder: &Path, name: &str) -> String {
        std::fs::read_to_string(folder.join(name)).unwrap()
    }

    pub(crate) fn json(folder: &Path, name: &str) -> serde_json::Value {
        serde_json::from_str(&text(folder, name)).unwrap()
    }

    /// The names of the files in `folder`, sorted.
    pub(crate) fn files(folder: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(folder)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    /// Whether this host can run guest code in a test: a Mac with Apple Silicon, off hosted CI.
    pub(crate) fn can_run_guests() -> bool {
        cfg!(all(target_os = "macos", target_arch = "aarch64")) && std::env::var_os("CI").is_none()
    }
}

use support::{can_run_guests, files, json, program, run, scratch, text};

/// The files every run leaves, crashed or not.
const ALWAYS: [&str; 4] = [EVENTS_FILE, GUEST_LOG_FILE, SUMMARY_FILE, TRACE_FILE];

#[test]
fn a_program_file_that_does_not_exist_is_an_emulator_error_with_a_summary() {
    let folder = scratch("missing");
    let missing = folder.join("nothing.nro");
    let (ran, artifacts) = run(&missing, &[]);
    assert_eq!(ran.code, exit_code::EMULATOR_ERROR);
    assert_eq!(ran.stderr, "");
    assert!(
        ran.stdout.starts_with(&format!(
            "starquake: emulator error: could not read {}: ",
            missing.display()
        )),
        "{}",
        ran.stdout
    );
    assert_eq!(files(&artifacts), ALWAYS);
    let summary = json(&artifacts, SUMMARY_FILE);
    assert_eq!(summary["exit_reason"], "emulator_error");
    assert_eq!(summary["exit_code"], 2);
    assert_eq!(text(&artifacts, TRACE_FILE), "");
}

#[test]
fn a_file_that_is_not_an_nro_is_an_emulator_error() {
    let folder = scratch("garbage");
    let garbage = program(&folder, "garbage.nro", &[0x5a; 0x200]);
    let (ran, artifacts) = run(&garbage, &[]);
    assert_eq!(ran.code, exit_code::EMULATOR_ERROR);
    assert_eq!(
        json(&artifacts, SUMMARY_FILE)["message"],
        format!(
            "{}: the program is not a loadable NRO: NRO magic is not NRO0",
            garbage.display()
        )
    );
}

#[test]
fn a_command_line_that_cannot_be_parsed_runs_nothing() {
    let folder = scratch("usage");
    let hello = program(&folder, "hello.nro", &programs::hello().unwrap());
    let (ran, artifacts) = run(&hello, &["--until", "frames=10"]);
    assert_eq!(ran.code, exit_code::EMULATOR_ERROR);
    assert_eq!(ran.stdout, "");
    assert!(
        ran.stderr.starts_with(
            "starquake: `--until frames=10` is not supported yet; only `timeout=<n>s` is\n"
        ),
        "{}",
        ran.stderr
    );
    assert!(!artifacts.exists());
}

#[test]
fn hello_exits_with_code_0_and_leaves_its_output_and_trace() {
    if !can_run_guests() {
        return;
    }
    let folder = scratch("hello");
    let hello = program(&folder, "hello.nro", &programs::hello().unwrap());
    let (ran, artifacts) = run(&hello, &["--until", "timeout=30s"]);
    assert_eq!(ran.code, exit_code::STOP_CONDITION, "{}", ran.stdout);
    assert_eq!(ran.stderr, "");
    // What the program must print, as constants of this test.
    assert_eq!(
        ran.stdout,
        format!(
            "[guest] hello from the guest\n[guest] starquake walking skeleton\n\
             starquake: the program exited (exit code 0); artifacts in {}\n",
            artifacts.display()
        )
    );
    assert_eq!(files(&artifacts), ALWAYS);
    assert_eq!(
        text(&artifacts, GUEST_LOG_FILE),
        "hello from the guest\nstarquake walking skeleton\n"
    );
    let summary = json(&artifacts, SUMMARY_FILE);
    assert_eq!(summary["exit_reason"], "process_exited");
    assert_eq!(summary["exit_code"], 0);
    assert_eq!(summary["system_calls"]["OutputDebugString"], 2);
    assert_eq!(summary["system_calls"]["ExitProcess"], 1);
    assert_eq!(summary["guest_log_lines"], 2);
    let trace = text(&artifacts, TRACE_FILE);
    assert!(
        trace.starts_with("thread 1\n  lifecycle ProcessStart 1\n"),
        "{trace}"
    );
    assert!(trace.ends_with("  lifecycle ProcessExit 1\n"), "{trace}");
    let dump = std::fs::read(artifacts.join(EVENTS_FILE)).unwrap();
    let events = starquake_trace::Dump::from_bytes(&dump).unwrap().merged();
    // Start, two lines each with its call, the exit call, and the exit.
    assert_eq!(events.len(), 7);
}

#[test]
fn a_crash_exits_with_code_1_and_leaves_a_crash_report() {
    if !can_run_guests() {
        return;
    }
    let folder = scratch("crash");
    let crash = programs::crash().unwrap();
    let file = program(&folder, "crash.nro", &crash.nro);
    let (ran, artifacts) = run(&file, &["--until=timeout=30s"]);
    assert_eq!(ran.code, exit_code::GUEST_CRASHED, "{}", ran.stdout);
    let mut expected = ALWAYS.to_vec();
    expected.extend([CRASH_JSON_FILE, CRASH_TEXT_FILE]);
    expected.sort_unstable();
    assert_eq!(files(&artifacts), expected);

    let report = json(&artifacts, CRASH_JSON_FILE);
    let fault = format!("{:#x}", crash.fault_address);
    assert_eq!(report["exception"]["kind"], "data_abort");
    assert_eq!(report["exception"]["address"], fault);
    assert_eq!(report["exception"]["write"], true);
    assert_eq!(
        report["pc_location"]["offset"],
        format!("{:#x}", crash.store_offset)
    );
    let pc = report["pc"].as_str().unwrap().to_owned();
    // The command line that started the run is in the report.
    assert_eq!(report["command_line"][1], "run");
    let crash_text = text(&artifacts, CRASH_TEXT_FILE);
    assert!(
        crash_text.contains(&format!("a write at address {fault} (not mapped)")),
        "{crash_text}"
    );
    assert!(
        crash_text.contains(&format!(
            "instruction: pc {pc} (main+{:#x})",
            crash.store_offset
        )),
        "{crash_text}"
    );
    // The one line the command prints names both as well.
    assert!(
        ran.stdout.contains(&format!("at address {fault}")),
        "{}",
        ran.stdout
    );
    assert!(ran.stdout.contains(&format!("pc {pc}")), "{}", ran.stdout);
    assert_eq!(json(&artifacts, SUMMARY_FILE)["exit_code"], 1);
}

#[test]
fn an_endless_loop_exits_with_code_3_at_its_timeout() {
    if !can_run_guests() {
        return;
    }
    let folder = scratch("loop");
    let file = program(&folder, "loop.nro", &programs::endless_loop().unwrap());
    let (ran, artifacts) = run(&file, &["--until", "timeout=1s"]);
    assert_eq!(ran.code, exit_code::TIMEOUT, "{}", ran.stdout);
    assert_eq!(files(&artifacts), ALWAYS);
    let summary = json(&artifacts, SUMMARY_FILE);
    assert_eq!(summary["exit_reason"], "timeout");
    assert_eq!(summary["exit_code"], 3);
    assert!(summary["duration_ms"].as_u64().unwrap() >= 1000);
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[test]
fn without_a_cpu_backend_a_run_exits_with_code_2_and_says_why() {
    let folder = scratch("no-backend");
    let hello = program(&folder, "hello.nro", &programs::hello().unwrap());
    let (ran, artifacts) = run(&hello, &[]);
    assert_eq!(ran.code, exit_code::EMULATOR_ERROR);
    assert_eq!(
        json(&artifacts, SUMMARY_FILE)["message"],
        starquake_core::Error::NoCpuBackend.to_string()
    );
    assert!(
        ran.stdout
            .contains("no CPU backend is available on this host")
    );
}

#[test]
fn the_process_exit_status_is_the_runs_exit_code() {
    use std::process::Command;

    let binary = env!("CARGO_BIN_EXE_starquake");
    let folder = scratch("process");
    let artifacts = folder.join("artifacts");
    // On a Mac the crash program must give 1. The binary needs the hypervisor entitlement, so it
    // is started through the workspace's cargo runner, as `cargo run` starts it. On hosts with no
    // CPU backend any program gives 2.
    let (mut command, expected) = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        if !can_run_guests() {
            return;
        }
        let runner = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/macos-runner.sh");
        let mut command = Command::new(runner);
        command.arg(binary);
        (command, exit_code::GUEST_CRASHED)
    } else {
        (Command::new(binary), exit_code::EMULATOR_ERROR)
    };
    let file = program(&folder, "crash.nro", &programs::crash().unwrap().nro);
    let output = command
        .arg("run")
        .arg(&file)
        .args(["--until", "timeout=30s", "--artifacts"])
        .arg(&artifacts)
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(i32::from(expected)),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        json(&artifacts, SUMMARY_FILE)["exit_code"],
        u64::from(expected)
    );
}
