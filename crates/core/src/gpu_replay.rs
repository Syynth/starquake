//! `starquake gpu-replay`: replays a GPU capture (`SQGC`) through the real GPU front and back
//! with the recording backend, no game and no GPU (gpu-protocol spec, "Replay"; observability
//! spec, "Headless runs"). The frontend parses the command line and prints what this returns;
//! the replay itself is the GPU crate's (`starquake_gpu::capture`).
//!
//! The output is plain words: one line saying how the replay ended, then the dump asked for. The
//! exit code follows the headless contract: 0 when the replay reached its stop condition (the
//! capture's end, or `--until present=N`), and 2, an emulator error, for a capture that cannot be
//! read, is malformed, or diverges from its replay (a replay that disagrees stops as an emulator
//! error, decision log, 2026-11-17).
//!
//! A capture with fewer presents than `--until present=N` asks for replays to its end, and the
//! summary line says so: the end of the capture is the replay's other stop condition.

use std::fs::File;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use starquake_gpu::capture::{
    Capture, CaptureError, MAX_CAPTURE_BYTES, ReplayError, ReplayOptions, ReplayOutput, replay,
    writes_text,
};
use starquake_gpu::ring::write_text;
use thiserror::Error;

use crate::report::exit_code;

/// What `--dump` prints after the summary line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dump {
    /// The ring entries the back applied, as ring text.
    Ring,
    /// The render command stream: the recording backend's snapshot text.
    Render,
    /// The writes the replayed threads made.
    Writes,
}

impl Dump {
    /// The dump `--dump <name>` names, if any.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "ring" => Some(Self::Ring),
            "render" => Some(Self::Render),
            "writes" => Some(Self::Writes),
            _ => None,
        }
    }
}

/// What `starquake gpu-replay` asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuReplayArgs {
    /// The capture file.
    pub capture: PathBuf,
    /// `--until present=N`.
    pub until_present: Option<u64>,
    /// `--dump ring|render|writes`.
    pub dump: Option<Dump>,
}

/// Why a replay did not reach its stop condition.
#[derive(Debug, Error)]
pub enum GpuReplayError {
    /// The file could not be read.
    #[error("the capture {path} could not be read: {error}")]
    Read {
        /// The file.
        path: PathBuf,
        /// Why.
        error: std::io::Error,
    },
    /// The file is not a capture this build reads.
    #[error("the capture {path} is malformed: {error}")]
    Malformed {
        /// The file.
        path: PathBuf,
        /// Why.
        error: CaptureError,
    },
    /// The replay stopped.
    #[error("{0}")]
    Replay(#[from] ReplayError),
}

impl GpuReplayError {
    /// The process exit code: every failure is an emulator error (module docs).
    pub fn exit_code(&self) -> u8 {
        exit_code::EMULATOR_ERROR
    }
}

/// What reading a capture file found: its bytes, or its length when that is past `limit`.
enum Read {
    Bytes(Vec<u8>),
    TooLarge(usize),
}

/// Reads `path` unless its length is past `limit`, and then at most `limit + 1` bytes, so a
/// file that grows while it is read is refused by the reader without being read whole.
fn read_bounded(path: &Path, limit: usize) -> std::io::Result<Read> {
    let file = File::open(path)?;
    let len = usize::try_from(file.metadata()?.len()).unwrap_or(usize::MAX);
    if len > limit {
        return Ok(Read::TooLarge(len));
    }
    let mut bytes = Vec::new();
    let bound = u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1);
    file.take(bound).read_to_end(&mut bytes)?;
    Ok(Read::Bytes(bytes))
}

/// The summary line and the dump of a finished replay.
pub fn render_output(output: &ReplayOutput, args: &GpuReplayArgs) -> String {
    let mut text = if output.stopped_early {
        format!(
            "stopped after present {}: {} ring entries replayed\n",
            output.presents,
            output.ring.len()
        )
    } else {
        format!(
            "replayed the whole capture: {} ring entries, {} presents; the ring and the render \
             stream match the capture\n",
            output.ring.len(),
            output.presents
        )
    };
    if let Some(until) = args.until_present
        && !output.stopped_early
        && output.presents < until
    {
        text.push_str(&format!(
            "the capture ended before present {until}\n"
        ));
    }
    match args.dump {
        Some(Dump::Ring) => text.push_str(&write_text(&output.ring)),
        Some(Dump::Render) => text.push_str(&output.render),
        Some(Dump::Writes) => text.push_str(&writes_text(&output.writes)),
        None => {}
    }
    text
}

/// Replays the capture `args` names and gives what to print.
///
/// # Errors
///
/// [`GpuReplayError`]: the file could not be read, is malformed, or the replay stopped.
pub fn gpu_replay(args: &GpuReplayArgs) -> Result<String, GpuReplayError> {
    let malformed = |error| GpuReplayError::Malformed {
        path: args.capture.clone(),
        error,
    };
    let read =
        read_bounded(&args.capture, MAX_CAPTURE_BYTES).map_err(|error| GpuReplayError::Read {
            path: args.capture.clone(),
            error,
        })?;
    let bytes = match read {
        Read::Bytes(bytes) => bytes,
        Read::TooLarge(len) => return Err(malformed(CaptureError::TooLarge { len })),
    };
    let capture = Capture::parse(&bytes).map_err(malformed)?;
    let output = replay(
        &capture,
        ReplayOptions {
            until_present: args.until_present,
        },
    )?;
    Ok(render_output(&output, args))
}

#[cfg(test)]
mod tests;
