//! Writing a run's artifacts folder.
//!
//! Source: `docs/specs/observability.md`, "Headless runs": the folder always holds
//! `summary.json`, `trace.txt`, `events.bin`, `gpu.txt` ([`GPU_FILE`]) and the `frames/` folder
//! ([`FRAMES_FOLDER`]), plus `crash.json` and `crash.txt` when the guest crashed. Snapshots are
//! not written: nothing produces them yet.
//!
//! `gpu.txt` is the GPU's flight recorder, empty for a run with no GPU (the null front).
//! `frames/` holds the run's frame dumps (`crate::frames`), one PNG per frame kept, named by
//! [`frame_file_name`]; it is made even when empty. A frame file left in it by an earlier run
//! (an entry that is not a folder, whose name [`frame_number`] reads) is removed first, so the
//! folder holds this run's frames only; any other entry in it, a folder with a frame file's name
//! included, is left alone. The frames are written last, after the crash report, so a frames
//! folder that cannot be written (one of this run's frame names taken by a folder, say) fails the
//! write without costing a crashed run its `crash.json` and `crash.txt`.
//!
//! Two files are added to the spec's table: [`GUEST_LOG_FILE`], the text the guest wrote with
//! `OutputDebugString` (the event stream keeps only each line's length and hash, so without it a
//! run's output could be checked but not read); and [`LOG_FILE`], the emulator's own warnings
//! and errors during the run (the owner's ruling of 2026-11-17 on issue #207).

use std::io;
use std::path::{Path, PathBuf};

use crate::frames::{frame_file_name, frame_number};
use crate::report::RunReport;

pub use crate::frames::FRAMES_FOLDER;

/// Exit reason, duration and counts (`RunReport::summary`), as JSON.
pub const SUMMARY_FILE: &str = "summary.json";
/// The golden-trace view of the event stream.
pub const TRACE_FILE: &str = "trace.txt";
/// The flight recorder's dump, readable with `starquake_trace::Dump::from_bytes`.
pub const EVENTS_FILE: &str = "events.bin";
/// The guest's `OutputDebugString` lines, one per line of the file, with every byte that is not
/// printable ASCII escaped (`\n`, `\\`, `\xNN`), so a line break inside a guest string cannot be
/// mistaken for the end of one.
///
/// A run keeps a limited amount of guest log text. When the guest wrote more, the file holds
/// the first lines and ends with `[truncated: <n> later lines were not kept]`. A guest could
/// print that text itself, so the authority is `guest_log_lines_dropped` in `summary.json`.
/// (`trace.txt` ends with a line of its own when it was cut short; see `trace_events_dropped`.)
pub const GUEST_LOG_FILE: &str = "guest_log.txt";
/// The emulator's own warnings and errors during the run ([`crate::host_log`]), one per line.
///
/// Bounded like the guest log: when the emulator logged more, the file holds the first lines
/// and ends with `[truncated: <n> later lines were not kept]`; the authority is
/// `host_log_lines_dropped` in `summary.json`. Always written, empty when nothing was logged.
pub const LOG_FILE: &str = "log.txt";
/// The GPU's own flight recorder (`docs/specs/gpu-protocol.md`, "Observability"). Always
/// written, empty for a run with no GPU.
pub const GPU_FILE: &str = "gpu.txt";
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
    if report.guest_log_dropped > 0 {
        guest_log.extend(truncated(report.guest_log_dropped).bytes());
    }
    put(GUEST_LOG_FILE, &guest_log)?;
    let mut host_log = String::new();
    for line in &report.host_log {
        host_log.push_str(line);
        host_log.push('\n');
    }
    if report.host_log_dropped > 0 {
        host_log.push_str(&truncated(report.host_log_dropped));
    }
    put(LOG_FILE, host_log.as_bytes())?;
    put(GPU_FILE, report.graphics.gpu_recorder.as_bytes())?;

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
    // Last: a frames folder that cannot be written does not cost a crashed run its crash report.
    write_frames(&folder.join(FRAMES_FOLDER), report)
}

/// Makes `frames` (the artifacts folder's [`FRAMES_FOLDER`]), removes the frame files an earlier
/// run left in it, and writes `report`'s (module docs).
fn write_frames(frames: &Path, report: &RunReport) -> Result<(), ArtifactError> {
    let io_error = |path: &Path| {
        let path = path.to_path_buf();
        move |source| ArtifactError::Io { path, source }
    };
    std::fs::create_dir_all(frames).map_err(io_error(frames))?;
    for entry in std::fs::read_dir(frames).map_err(io_error(frames))? {
        let entry = entry.map_err(io_error(frames))?;
        let named = entry.file_name().to_str().and_then(frame_number).is_some();
        // A folder with a frame file's name is not an earlier run's frame: it is left alone.
        let stale = named && !entry.file_type().map_err(io_error(&entry.path()))?.is_dir();
        if stale {
            let path = entry.path();
            std::fs::remove_file(&path).map_err(io_error(&path))?;
        }
    }
    for file in &report.graphics.dumps.files {
        let path = frames.join(frame_file_name(file.frame));
        std::fs::write(&path, &file.png).map_err(io_error(&path))?;
    }
    Ok(())
}

/// The last line of a log file that was cut short after its first lines: `dropped` later lines
/// were not kept.
fn truncated(dropped: u64) -> String {
    format!("[truncated: {dropped} later lines were not kept]\n")
}
