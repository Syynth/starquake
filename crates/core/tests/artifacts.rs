//! The artifacts folder: what `artifacts::write` leaves on disk for each kind of run. Runs on
//! every host; the reports are built by hand.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use starquake_core::artifacts::{
    self, ArtifactError, CRASH_JSON_FILE, CRASH_TEXT_FILE, EVENTS_FILE, GUEST_LOG_FILE,
    SUMMARY_FILE, TRACE_FILE,
};
use starquake_core::crash::{CrashReport, Exception, Module};
use starquake_core::{ExitReason, Mapping, RunReport};
use starquake_cpu::GuestRegs;
use starquake_trace::{CoreDump, Dump, Event, EventKind, GuestTicks, LifecyclePhase, ThreadId};

/// Helpers, in a `cfg(test)` module so they may unwrap like the tests do.
#[cfg(test)]
mod support {
    use super::*;

    /// An empty folder path for the test called `name`, under the build's scratch folder.
    pub(crate) fn folder(name: &str) -> PathBuf {
        let folder = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("artifacts")
            .join(name);
        if folder.exists() {
            std::fs::remove_dir_all(&folder).unwrap();
        }
        folder
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

    pub(crate) fn dump() -> Dump {
        Dump::new(vec![CoreDump {
            core: 0,
            events: vec![Event {
                seq: 0,
                time: GuestTicks(7),
                core: 0,
                thread: ThreadId(1),
                kind: EventKind::Lifecycle {
                    phase: LifecyclePhase::ProcessStart,
                    subject: 1,
                },
            }],
        }])
    }

    /// A run that exited cleanly after printing three awkward lines.
    pub(crate) fn clean() -> RunReport {
        RunReport {
            exit: ExitReason::ProcessExited,
            duration: Duration::from_millis(42),
            guest_log: vec![
                b"plain".to_vec(),
                b"two\nlines and a \\".to_vec(),
                vec![0xff, b'x', 0x00],
            ],
            guest_log_dropped: 0,
            trace: "thread 1\n  lifecycle ProcessStart 1\n".into(),
            trace_events_dropped: 0,
            events: dump().to_bytes().unwrap(),
            events_error: None,
            system_calls: BTreeMap::from([("OutputDebugString".to_owned(), 3)]),
            stubbed: BTreeSet::new(),
            patched_counter_reads: vec![0x8000_00b0],
            memory_map: Vec::new(),
        }
    }

    /// The same run, ended by a write to an unmapped address.
    pub(crate) fn crashed() -> RunReport {
        let regs = GuestRegs {
            pc: 0x8000_00b8,
            ..GuestRegs::default()
        };
        let crash = CrashReport::new(
            Exception::DataAbort {
                address: 0xdea8,
                write: true,
            },
            &regs,
            &[Mapping {
                name: "main.text".into(),
                start: 0x8000_0000,
                end: 0x8000_1000,
            }],
            &[Module {
                name: "main".into(),
                base: 0x8000_0000,
                size: 0x1000,
                build_id: vec![0; 0x20],
            }],
            vec!["#0 core 0 thread 1: lifecycle ProcessStart 1".into()],
            &[],
        );
        RunReport {
            exit: ExitReason::GuestCrashed(Box::new(crash)),
            ..clean()
        }
    }

    pub(crate) fn read(folder: &Path, name: &str) -> Vec<u8> {
        std::fs::read(folder.join(name)).unwrap()
    }

    pub(crate) fn json(folder: &Path, name: &str) -> serde_json::Value {
        serde_json::from_slice(&read(folder, name)).unwrap()
    }
}

use support::{clean, crashed, dump, files, folder, json, read};

#[test]
fn a_clean_run_writes_the_summary_trace_events_and_guest_log_and_no_crash_report() {
    let folder = folder("clean");
    let report = clean();
    artifacts::write(&folder, &report).unwrap();
    assert_eq!(
        files(&folder),
        [EVENTS_FILE, GUEST_LOG_FILE, SUMMARY_FILE, TRACE_FILE]
    );

    let summary = json(&folder, SUMMARY_FILE);
    assert_eq!(summary, serde_json::to_value(report.summary()).unwrap());
    assert_eq!(summary["exit_reason"], "process_exited");
    assert_eq!(summary["exit_code"], 0);
    assert_eq!(summary["duration_ms"], 42);
    assert_eq!(summary["system_calls"]["OutputDebugString"], 3);
    assert!(read(&folder, SUMMARY_FILE).ends_with(b"}\n"));

    assert_eq!(read(&folder, TRACE_FILE), report.trace.as_bytes());
    assert_eq!(Dump::from_bytes(&read(&folder, EVENTS_FILE)), Ok(dump()));
    // One line of the file per guest string, whatever bytes the string holds.
    assert_eq!(
        read(&folder, GUEST_LOG_FILE),
        b"plain\ntwo\\nlines and a \\\\\n\\xffx\\x00\n"
    );
}

#[test]
fn a_run_that_reached_its_limits_says_so_in_the_guest_log_and_the_summary() {
    let folder = folder("truncated");
    let report = RunReport {
        guest_log: vec![b"kept".to_vec()],
        guest_log_dropped: 7,
        trace_events_dropped: 12,
        ..clean()
    };
    artifacts::write(&folder, &report).unwrap();
    // The kept lines, then one line that no kept line can be mistaken for only by its place:
    // it is the last, and the summary says whether it is there.
    assert_eq!(
        read(&folder, GUEST_LOG_FILE),
        b"kept\n[truncated: 7 later lines were not kept]\n"
    );
    let summary = json(&folder, SUMMARY_FILE);
    assert_eq!(summary["guest_log_lines"], 8);
    assert_eq!(summary["guest_log_lines_dropped"], 7);
    assert_eq!(summary["trace_events_dropped"], 12);
    assert_eq!(summary["events_error"], serde_json::Value::Null);
    // A complete log has no such line.
    let complete = folder.join("complete");
    artifacts::write(&complete, &clean()).unwrap();
    assert!(!read(&complete, GUEST_LOG_FILE).ends_with(b"kept]\n"));
    assert_eq!(json(&complete, SUMMARY_FILE)["guest_log_lines_dropped"], 0);
}

#[test]
fn a_crash_adds_the_crash_report_as_data_and_as_text() {
    let folder = folder("crashed");
    let report = crashed();
    artifacts::write(&folder, &report).unwrap();
    assert_eq!(
        files(&folder),
        [
            CRASH_JSON_FILE,
            CRASH_TEXT_FILE,
            EVENTS_FILE,
            GUEST_LOG_FILE,
            SUMMARY_FILE,
            TRACE_FILE
        ]
    );
    let crash = report.crash().unwrap();
    assert_eq!(
        json(&folder, CRASH_JSON_FILE),
        serde_json::to_value(crash).unwrap()
    );
    let data = json(&folder, CRASH_JSON_FILE);
    assert_eq!(data["exception"]["address"], "0xdea8");
    assert_eq!(data["pc"], "0x800000b8");
    assert_eq!(read(&folder, CRASH_TEXT_FILE), crash.to_text().as_bytes());
    let summary = json(&folder, SUMMARY_FILE);
    assert_eq!(summary["exit_reason"], "guest_crashed");
    assert_eq!(summary["exit_code"], 1);
}

#[test]
fn a_later_run_in_the_same_folder_leaves_no_stale_crash_report() {
    let folder = folder("reused");
    artifacts::write(&folder, &crashed()).unwrap();
    assert!(folder.join(CRASH_JSON_FILE).exists());
    artifacts::write(&folder, &clean()).unwrap();
    assert_eq!(
        files(&folder),
        [EVENTS_FILE, GUEST_LOG_FILE, SUMMARY_FILE, TRACE_FILE]
    );
    assert_eq!(json(&folder, SUMMARY_FILE)["exit_reason"], "process_exited");
}

#[test]
fn a_folder_that_cannot_be_created_is_an_error_naming_it() {
    // A file is in the way of the folder.
    let parent = folder("blocked");
    std::fs::create_dir_all(&parent).unwrap();
    let blocked = parent.join("a-file");
    std::fs::write(&blocked, b"").unwrap();
    match artifacts::write(&blocked, &clean()) {
        Err(ArtifactError::Io { path, .. }) => assert_eq!(path, blocked),
        other => assert!(other.is_err(), "expected an I/O error, got {other:?}"),
    }
    assert!(artifacts::write(&blocked, &clean()).is_err());
}
