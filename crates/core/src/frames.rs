//! A run's frame dumps (`docs/specs/observability.md`, the artifacts table: "`frames/`: PNG frame
//! dumps on request or at the end"; `docs/specs/render-backend.md`, "Frame dumps and headless
//! runs"; slice R9's second part, issue #386).
//!
//! # How the frames get here
//!
//! A run with a GPU asks the GPU crate to dump every present: the executor's setting
//! `starquake_gpu::back::exec::FrameDumps::Every`, in the `Gpu`'s config (`GpuConfig::exec`).
//! The render backend then reads each presented view back at the guest's own resolution and
//! hands it to the [`FrameSink`] the run installed on it: a [`FrameKeeper`]. The render crate
//! does no file I/O; this module keeps the frames in memory, [`FrameKeeper::take`] encodes them
//! as PNG with the render crate's encoding ([`Frame::png`]) when the run ends, and
//! [`crate::artifacts::write`] writes them into the artifacts folder's [`FRAMES_FOLDER`].
//!
//! # What is kept
//!
//! "On request or at the end": the run keeps **the newest frames** whose pixels together fit in
//! [`FRAME_BYTES`] bytes of RGBA8, and drops the oldest to make room. So a run that stops at
//! `--until frames=N` keeps its last frames up to frame `N`, and any other run the frames it
//! ended on. A frame larger than [`FRAME_BYTES`] alone is never kept. Frames the backend cannot
//! read ([`FramePixels::Unknown`]: the recording backend's answer for a frame a draw rendered)
//! have no PNG and are only counted. What was not kept is counted
//! ([`DumpedFrames::dropped`], `summary.json`'s `frames_dropped`), as the observability spec asks
//! of everything a run cuts ("`summary.json` records what was cut").
//!
//! **Owner question.** The spec does not say which frames "at the end" means, nor how many a run
//! keeps. Keeping the newest within a byte budget is this module's choice; so is asking for a
//! dump of every present, which under the wgpu backend (slice R12) costs a readback per frame.
//!
//! # The file names
//!
//! Each frame is written as `frames/<n>.png` ([`frame_file_name`]), where `<n>` is the frame
//! number the GPU gave the present (`PresentDesc::frame`: the presents the GPU's back thread
//! recorded, counted from 0), in decimal, zero-padded to [`FRAME_NUMBER_DIGITS`] digits so that
//! the files sort in order up to a million frames; a larger number is written in full. That
//! number is the GPU's, not `summary.json`'s `frames`, which counts the presents the graphics
//! server handed the GPU (`crate::hle::Graphics::frames`): the two differ by the presents the GPU
//! refused (an image it could not read), which get no number.

use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard, PoisonError};

use starquake_render::{Frame, FramePixels, FrameSink};

/// The artifacts folder's folder of frame dumps.
pub const FRAMES_FOLDER: &str = "frames";

/// The most bytes of RGBA8 pixels a run keeps for its frame dumps: starquake's own number,
/// about eighteen 1280 by 720 frames (module docs, "What is kept"). The observability spec asks
/// for a size cap per run without fixing one.
pub const FRAME_BYTES: usize = 64 << 20;

/// The digits a frame file's number is padded to (module docs, "The file names").
pub const FRAME_NUMBER_DIGITS: usize = 6;

/// The name of frame `frame`'s file in [`FRAMES_FOLDER`] (module docs, "The file names"):
/// `000042.png` for frame 42.
#[must_use]
pub fn frame_file_name(frame: u64) -> String {
    format!("{frame:0width$}.png", width = FRAME_NUMBER_DIGITS)
}

/// The frame number a file of [`FRAMES_FOLDER`] is named for, when `name` is exactly what
/// [`frame_file_name`] makes of it; `None` for any other name.
#[must_use]
pub fn frame_number(name: &str) -> Option<u64> {
    let number = name.strip_suffix(".png")?.parse().ok()?;
    (frame_file_name(number) == name).then_some(number)
}

/// One dumped frame as a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameFile {
    /// The frame number the GPU gave the present (module docs, "The file names").
    pub frame: u64,
    /// The PNG file's bytes.
    pub png: Vec<u8>,
}

/// What a run kept of its frame dumps, once it has ended.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DumpedFrames {
    /// The frames kept, in the order they were presented.
    pub files: Vec<FrameFile>,
    /// Frames the render backend could not read ([`FramePixels::Unknown`]), which have no PNG.
    pub unknown: u64,
    /// Frames read back but not written: dropped for room (module docs, "What is kept"), or
    /// refused by the PNG encoder, which is logged.
    pub dropped: u64,
}

/// The frame sink a run installs on its render backend (module docs).
#[derive(Debug)]
pub struct FrameKeeper {
    budget: usize,
    kept: Mutex<Kept>,
}

/// The frames a [`FrameKeeper`] holds, and its counts.
#[derive(Debug, Default)]
struct Kept {
    /// Frames with known pixels, oldest first.
    frames: VecDeque<Frame>,
    /// Bytes of pixels in `frames`.
    bytes: usize,
    unknown: u64,
    dropped: u64,
}

impl Default for FrameKeeper {
    /// A keeper of [`FRAME_BYTES`].
    fn default() -> Self {
        Self::new(FRAME_BYTES)
    }
}

impl FrameKeeper {
    /// A keeper of the newest frames whose pixels fit in `budget` bytes.
    #[must_use]
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            kept: Mutex::new(Kept::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Kept> {
        // Every field is written whole at each step: a poisoned lock is usable.
        self.kept.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Everything kept so far, each frame encoded as a PNG, and the counts; the keeper is left
    /// empty. A frame the encoder refuses is logged and counted as dropped.
    #[must_use]
    pub fn take(&self) -> DumpedFrames {
        let kept = std::mem::take(&mut *self.lock());
        let mut dropped = kept.dropped;
        let mut files = Vec::with_capacity(kept.frames.len());
        for frame in kept.frames {
            match frame.png() {
                Ok(Some(png)) => files.push(FrameFile {
                    frame: frame.frame,
                    png,
                }),
                // Only frames with known pixels are kept, so `None` does not come here.
                Ok(None) => dropped = dropped.saturating_add(1),
                Err(error) => {
                    tracing::warn!(frame = frame.frame, %error, "a dumped frame could not be encoded");
                    dropped = dropped.saturating_add(1);
                }
            }
        }
        DumpedFrames {
            files,
            unknown: kept.unknown,
            dropped,
        }
    }
}

impl FrameSink for FrameKeeper {
    fn frame(&self, frame: Frame) {
        let size = match &frame.pixels {
            FramePixels::Rgba8(pixels) => pixels.len(),
            FramePixels::Unknown => {
                let mut kept = self.lock();
                kept.unknown = kept.unknown.saturating_add(1);
                return;
            }
        };
        let mut kept = self.lock();
        if size > self.budget {
            kept.dropped = kept.dropped.saturating_add(1);
            return;
        }
        // Room for the new frame: the oldest go first. `size <= budget`, so this ends at the
        // latest with every older frame gone.
        while kept.bytes.saturating_add(size) > self.budget {
            let Some(oldest) = kept.frames.pop_front() else {
                break;
            };
            kept.bytes = kept.bytes.saturating_sub(rgba_len(&oldest));
            kept.dropped = kept.dropped.saturating_add(1);
        }
        kept.bytes = kept.bytes.saturating_add(size);
        kept.frames.push_back(frame);
    }
}

/// The bytes of a kept frame's pixels.
fn rgba_len(frame: &Frame) -> usize {
    match &frame.pixels {
        FramePixels::Rgba8(pixels) => pixels.len(),
        FramePixels::Unknown => 0,
    }
}

#[cfg(test)]
mod tests;
