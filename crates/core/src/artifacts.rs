//! Writing a run's artifacts folder.
//!
//! Source: `docs/specs/observability.md`, "Headless runs": the folder always holds
//! `summary.json`, `trace.txt` and `events.bin`, plus `crash.json` and `crash.txt` when the guest
//! crashed. Frames and snapshots are not written: nothing produces them yet.
//!
//! One file is added to the spec's table: [`GUEST_LOG_FILE`], the text the guest wrote with
//! `OutputDebugString`. The event stream keeps only each line's length and hash, so without it a
//! run's output could be checked but not read.

use std::io;
use std::path::{Path, PathBuf};

use crate::report::RunReport;

/// Exit reason, duration and counts (`RunReport::summary`), as JSON.
pub const SUMMARY_FILE: &str = "summary.json";
/// The golden-trace view of the event stream.
pub const TRACE_FILE: &str = "trace.txt";
/// The flight recorder's dump, readable with `starquake_trace::Dump::from_bytes`.
pub const EVENTS_FILE: &str = "events.bin";
/// The guest's `OutputDebugString` lines, one per line of the file, with every byte that is not
/// printable ASCII escaped (`\n`, `\\`, `\xNN`), so a line break inside a guest string cannot be
/// mistaken for the end of one.
pub const GUEST_LOG_FILE: &str = "guest_log.txt";
/// The crash report as data. Only present when the guest crashed.
pub const CRASH_JSON_FILE: &str = "crash.json";
/// The crash report as text. Only present when the guest crashed.
pub const CRASH_TEXT_FILE: &str = "crash.txt";

/// Why the artifacts folder could not be written.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    /// A file or the folder could not be written.
    #[error("could not write {path}: {source}", path = path.display())]
    Io {
        /// The path being written.
        path: PathBuf,
        /// What the host said.
        source: io::Error,
    },
    /// A report could not be encoded as JSON.
    #[error("could not encode {file}: {source}")]
    Json {
        /// The file being encoded.
        file: &'static str,
        /// What the encoder said.
        source: serde_json::Error,
    },
}

/// Writes `report`'s artifacts into `folder`, creating it if needed. Files from an earlier run in
/// the same folder are replaced, and a crash report left by an earlier run is removed when this
/// one did not crash.
pub fn write(folder: &Path, report: &RunReport) -> Result<(), ArtifactError> {
    let io_error = |path: &Path| {
        let path = path.to_path_buf();
        move |source| ArtifactError::Io { path, source }
    };
    std::fs::create_dir_all(folder).map_err(io_error(folder))?;
    let put = |name: &str, contents: &[u8]| {
        let path = folder.join(name);
        std::fs::write(&path, contents).map_err(io_error(&path))
    };
    let json = |file: &'static str, value: Result<String, serde_json::Error>| {
        value
            .map(|text| text + "\n")
            .map_err(|source| ArtifactError::Json { file, source })
    };

    let summary = json(
        SUMMARY_FILE,
        serde_json::to_string_pretty(&report.summary()),
    )?;
    put(SUMMARY_FILE, summary.as_bytes())?;
    put(TRACE_FILE, report.trace.as_bytes())?;
    put(EVENTS_FILE, &report.events)?;
    let mut guest_log = Vec::new();
    for line in &report.guest_log {
        guest_log.extend(line.escape_ascii());
        guest_log.push(b'\n');
    }
    put(GUEST_LOG_FILE, &guest_log)?;

    if let Some(crash) = report.crash() {
        let data = json(CRASH_JSON_FILE, serde_json::to_string_pretty(crash))?;
        put(CRASH_JSON_FILE, data.as_bytes())?;
        put(CRASH_TEXT_FILE, crash.to_text().as_bytes())?;
    } else {
        for name in [CRASH_JSON_FILE, CRASH_TEXT_FILE] {
            let path = folder.join(name);
            match std::fs::remove_file(&path) {
                Err(error) if error.kind() != io::ErrorKind::NotFound => {
                    return Err(io_error(&path)(error));
                }
                _ => {}
            }
        }
    }
    Ok(())
}
