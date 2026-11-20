//! `run --backend` and `run --window` (render-backend slice R12): how the command line takes
//! them, the backend each combination draws with, and a window that cannot be made.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use starquake::window::{NOT_MAIN_THREAD, NotMainThread, may_open};
use starquake::{
    Command, RunArgs, USAGE, UsageError, execute, headless_setup, parse, render_choice,
};
use starquake_core::artifacts::SUMMARY_FILE;
use starquake_core::exit_code;
use starquake_core::renderer::{BackendFlag, Choice, ChoiceError, IfUnavailable, WGPU_BUILT};
use starquake_testkit::programs;

fn args(words: &[&str]) -> Vec<OsString> {
    words.iter().map(OsString::from).collect()
}

/// The run `words` parse to.
fn parsed(words: &[&str]) -> RunArgs {
    match parse(&args(words)) {
        Ok(Command::Run(run)) => run,
        other => unreachable!("{words:?} is not a run: {other:?}"),
    }
}

#[test]
fn run_takes_a_backend_and_a_window_in_either_form_and_in_any_order() {
    let cases: [(&[&str], Option<BackendFlag>, bool); 6] = [
        (&["run", "a.nro"], None, false),
        (
            &["run", "a.nro", "--backend", "recording"],
            Some(BackendFlag::Recording),
            false,
        ),
        (
            &["run", "--backend=recording", "a.nro"],
            Some(BackendFlag::Recording),
            false,
        ),
        (
            &["run", "--backend", "wgpu", "a.nro"],
            Some(BackendFlag::Wgpu),
            false,
        ),
        (&["run", "--window", "a.nro"], None, true),
        (
            &["run", "a.nro", "--window", "--backend=wgpu"],
            Some(BackendFlag::Wgpu),
            true,
        ),
    ];
    for (words, backend, window) in cases {
        let needs_wgpu = window || backend == Some(BackendFlag::Wgpu);
        if needs_wgpu && !WGPU_BUILT {
            continue;
        }
        let run = parsed(words);
        assert_eq!((run.backend, run.window), (backend, window), "{words:?}");
        assert_eq!(run.program, PathBuf::from("a.nro"));
    }
}

#[test]
fn each_flag_reaches_the_choice_in_its_own_place() {
    // `--backend recording` alone, `--window` alone and neither give three different choices on a
    // build with the wgpu backend, so a run that swapped or dropped a flag on its way to
    // `choose_backend` would get another.
    let choice = |words: &[&str]| render_choice(&parsed(words));
    assert_eq!(
        choice(&["run", "a.nro", "--backend", "recording"]),
        Ok(Choice::Recording)
    );
    if WGPU_BUILT {
        assert_eq!(
            choice(&["run", "a.nro"]),
            Ok(Choice::Wgpu(IfUnavailable::UseRecording))
        );
        assert_eq!(
            choice(&["run", "a.nro", "--window"]),
            Ok(Choice::Wgpu(IfUnavailable::EndRun))
        );
        assert_eq!(
            choice(&["run", "a.nro", "--backend", "wgpu"]),
            Ok(Choice::Wgpu(IfUnavailable::EndRun))
        );
    } else {
        assert_eq!(choice(&["run", "a.nro"]), Ok(Choice::Recording));
    }
}

/// The setup a run without `--window` hands the run (`headless_setup`) has the backend its flags
/// choose, and no window: the default, `--backend recording` and `--backend wgpu` each give their
/// own on a build with the wgpu backend, so a run handed another setup (the recording backend
/// whatever the flags, say) fails here.
#[test]
fn a_headless_run_is_handed_the_backend_its_flags_choose() {
    let setup = |args: &RunArgs| {
        headless_setup(args).map(|setup| (setup.choice(), setup.window().is_none()))
    };
    let recording = parsed(&["run", "a.nro", "--backend", "recording"]);
    assert_eq!(setup(&recording), Ok((Choice::Recording, true)));
    let default = parsed(&["run", "a.nro"]);
    let mut wgpu = default.clone();
    wgpu.backend = Some(BackendFlag::Wgpu);
    if WGPU_BUILT {
        assert_eq!(
            setup(&default),
            Ok((Choice::Wgpu(IfUnavailable::UseRecording), true))
        );
        assert_eq!(
            setup(&wgpu),
            Ok((Choice::Wgpu(IfUnavailable::EndRun), true))
        );
    } else {
        assert_eq!(setup(&default), Ok((Choice::Recording, true)));
        assert_eq!(setup(&wgpu), Err(ChoiceError::WgpuNotBuilt));
    }
    // A combination the choice refuses is refused here too.
    let mut conflict = recording;
    conflict.window = true;
    assert_eq!(setup(&conflict), Err(ChoiceError::RecordingWindow));
}

#[test]
fn each_malformed_backend_or_window_flag_has_its_own_error() {
    let cases: [(&[&str], UsageError); 8] = [
        (
            &["run", "a.nro", "--backend"],
            UsageError::MissingValue("--backend"),
        ),
        (
            &["run", "a.nro", "--backend", "metal"],
            UsageError::BadBackend("metal".into()),
        ),
        (
            &["run", "a.nro", "--backend="],
            UsageError::BadBackend(String::new()),
        ),
        // The value is checked before the repeat, as `--date`'s is.
        (
            &[
                "run",
                "a.nro",
                "--backend",
                "recording",
                "--backend",
                "Wgpu",
            ],
            UsageError::BadBackend("Wgpu".into()),
        ),
        (
            &[
                "run",
                "a.nro",
                "--backend",
                "recording",
                "--backend",
                "recording",
            ],
            UsageError::Repeated("--backend"),
        ),
        (
            &["run", "a.nro", "--window=yes"],
            UsageError::TakesNoValue("--window"),
        ),
        (
            &["run", "a.nro", "--window", "--window"],
            UsageError::Repeated("--window"),
        ),
        (
            &["run", "a.nro", "--window", "--backend", "recording"],
            UsageError::Backend(ChoiceError::RecordingWindow),
        ),
    ];
    for (words, expected) in cases {
        assert_eq!(parse(&args(words)), Err(expected), "{words:?}");
    }
    // Both faults at once: a missing program is reported before the backend choice.
    assert_eq!(
        parse(&args(&["run", "--window", "--backend", "recording"])),
        Err(UsageError::MissingProgram)
    );
    // The words, exactly.
    assert_eq!(
        UsageError::BadBackend("metal".into()).to_string(),
        "`--backend metal` is not a backend: write `recording` or `wgpu`"
    );
    assert_eq!(
        UsageError::TakesNoValue("--window").to_string(),
        "`--window` takes no value"
    );
    assert_eq!(
        UsageError::Backend(ChoiceError::RecordingWindow).to_string(),
        ChoiceError::RecordingWindow.to_string()
    );
}

#[test]
fn a_build_without_the_wgpu_backend_refuses_both_flags_that_need_it() {
    if WGPU_BUILT {
        return;
    }
    assert_eq!(
        parse(&args(&["run", "a.nro", "--window"])),
        Err(UsageError::Backend(ChoiceError::WindowNotBuilt))
    );
    assert_eq!(
        parse(&args(&["run", "a.nro", "--backend", "wgpu"])),
        Err(UsageError::Backend(ChoiceError::WgpuNotBuilt))
    );
}

#[test]
fn the_usage_names_both_flags() {
    assert!(
        USAGE.contains("[--backend recording|wgpu] [--window]"),
        "{USAGE}"
    );
}

#[test]
fn only_the_main_thread_may_open_the_window() {
    assert_eq!(may_open(true), Ok(()));
    assert_eq!(may_open(false), Err(NotMainThread));
    assert_eq!(
        NotMainThread.to_string(),
        "`--window` needs the process's main thread, which owns the window and its events; this \
         run was started on another thread"
    );
    assert_eq!(NOT_MAIN_THREAD, NotMainThread.to_string());
}

/// Helpers, in a `cfg(test)` module so they may unwrap like the tests do.
#[cfg(test)]
mod support {
    use super::*;

    /// An empty scratch folder for the test called `name`.
    pub(crate) fn scratch(name: &str) -> PathBuf {
        let folder = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("window")
            .join(name);
        if folder.exists() {
            std::fs::remove_dir_all(&folder).unwrap();
        }
        std::fs::create_dir_all(&folder).unwrap();
        folder
    }

    /// The emulator error `summary.json` in `artifacts` gives.
    pub(crate) fn summary_error(artifacts: &Path) -> String {
        let summary: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(artifacts.join(SUMMARY_FILE)).unwrap())
                .unwrap();
        assert_eq!(summary["exit_reason"], "emulator_error", "{summary}");
        assert_eq!(summary["exit_code"], 2);
        summary["message"].as_str().unwrap().to_owned()
    }
}

use support::{scratch, summary_error};

/// How long a test here waits for a `run --window` it started on a thread of its own.
const REFUSAL_WAIT: std::time::Duration = std::time::Duration::from_secs(60);

/// `run --window` of the hello program in the scratch folder `name`, by `execute` on a thread
/// named `thread` (not the process's main thread in either test harness), is refused before
/// anything runs, as an emulator error with the exact words, exit code 2 and the artifacts
/// written.
fn assert_refused_off_the_main_thread(name: &str, thread: &str) {
    let folder = scratch(name);
    let hello = folder.join("hello.nro");
    std::fs::write(&hello, programs::hello().unwrap()).unwrap();
    let artifacts = folder.join("artifacts");
    let line: Vec<OsString> = vec![
        "starquake".into(),
        "run".into(),
        hello.into(),
        "--window".into(),
        "--artifacts".into(),
        artifacts.clone().into(),
    ];
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name(thread.to_owned())
        .spawn(move || {
            let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
            let code = execute(&line, &mut stdout, &mut stderr);
            done.send((code, stdout)).unwrap();
        })
        .unwrap();
    // A run that panicked drops the sender: `recv_timeout` then fails at once.
    let (code, stdout) = finished.recv_timeout(REFUSAL_WAIT).unwrap();
    let stdout = String::from_utf8(stdout).unwrap();
    assert_eq!(code, exit_code::EMULATOR_ERROR, "{stdout}");
    assert_eq!(
        stdout,
        format!(
            "starquake: emulator error: {NOT_MAIN_THREAD} (exit code 2); artifacts in {}\n",
            artifacts.display()
        )
    );
    assert_eq!(summary_error(&artifacts), NOT_MAIN_THREAD);
    std::fs::remove_dir_all(&folder).unwrap();
}

/// `run --window` from a thread that is not the process's main thread (a test's) is refused
/// before anything runs. It needs no VM, so it runs on every host.
#[test]
fn a_window_off_the_main_thread_is_refused_before_the_run() {
    if !WGPU_BUILT {
        return;
    }
    assert_refused_off_the_main_thread("off-main", "a test's");
}

/// A thread that is not the process's main thread but is named `main`, the name Rust gives the
/// main thread, is refused all the same (`starquake::window`'s module docs, "When no window can be
/// made"): on macOS by `AppKit`'s check, and on Linux by the thread id against the process id,
/// where a check of the name alone would pass it on to winit, whose `EventLoop::new` then
/// panics.
#[test]
fn a_thread_named_main_is_still_refused_the_window() {
    if !WGPU_BUILT {
        return;
    }
    assert_refused_off_the_main_thread("named-main", "main");
}

/// A Linux host with no X11 or Wayland display cannot make a window: `run --window`, on the
/// process's main thread (a child process), ends before the run with `NO_WINDOW` and winit's
/// reason, which is winit's own words and not pinned here.
#[cfg(target_os = "linux")]
#[test]
fn a_host_with_no_display_ends_the_run_before_it_starts() {
    use starquake::window::NO_WINDOW;

    if !WGPU_BUILT {
        return;
    }
    let folder = scratch("no-display");
    let hello = folder.join("hello.nro");
    std::fs::write(&hello, programs::hello().unwrap()).unwrap();
    let artifacts = folder.join("artifacts");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_starquake"))
        .arg("run")
        .arg(&hello)
        .arg("--window")
        .arg("--artifacts")
        .arg(&artifacts)
        .env_remove("DISPLAY")
        .env_remove("WAYLAND_DISPLAY")
        .env_remove("WAYLAND_SOCKET")
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(i32::from(exit_code::EMULATOR_ERROR))
    );
    let error = summary_error(&artifacts);
    assert!(error.starts_with(&format!("{NO_WINDOW}: ")), "{error}");
    std::fs::remove_dir_all(&folder).unwrap();
}
