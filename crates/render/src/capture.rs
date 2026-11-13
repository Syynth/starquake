//! Host-level captures: submitted command lists, with their staging bytes, as a versioned file,
//! and their replay on a backend (docs/specs/render-backend.md, "Command lists are data" and
//! slice R11, first part).
//!
//! A [`CommandList`] is plain data, so the lists a backend was given can be written out and
//! submitted again later, with no emulator. This module owns the file's bytes ([`Capture`]) and
//! the replay ([`Capture::replay`]). It does no file I/O: the app reads and writes the file.
//!
//! # What a capture holds, and what it does not
//!
//! Each submitted list with the fence value it signalled, in order. An upload's staging bytes
//! are part of its command, so they are in the file. Nothing else is: not the resources the
//! lists name (the `create_*` and `destroy` calls), not the present target and not the results
//! (readbacks, frames). A replay therefore needs a backend on which the same resources already
//! exist under the same ids; [`Capture::replay`] says what happens when they do not.
//!
//! # Layout (version 1)
//!
//! Everything is `postcard`: unsigned integers are little-endian base-128 varints.
//!
//! | Bytes | What |
//! |---|---|
//! | 4 | [`MAGIC`], `SQRC` |
//! | varint | [`VERSION`] (`u32`) |
//! | per list: 1 | `0x01`: a list follows |
//! | per list: varint | the fence value the list signalled (`u64`) |
//! | per list: the rest | the [`CommandList`]: its command count, then each command |
//! | 1 | `0x00`: the end of the capture |
//!
//! The lists are a stream closed by an end mark, not a counted array, so that a writer can
//! append lists as they are submitted. A file without its end mark is truncated, and anything
//! after the end mark is an error.
//!
//! The header is read and checked before anything after it, magic first: a file of another
//! kind is [`CaptureError::BadMagic`], and a capture of another version is
//! [`CaptureError::UnsupportedVersion`] whatever follows, refused and not misread. The layout
//! is the spec's choice of `postcard` and otherwise this module's own; it follows the flight
//! recorder's dump (`starquake-trace`), which the spec does not lay out either.
//!
//! The bytes are pinned by `capture::tests::golden_bytes`, and those of a list by
//! `command::tests::golden_bytes`. Any change to them, or to the field order or types of
//! [`Command`](crate::Command) or anything inside one, changes the file and needs a new
//! [`VERSION`] in the same change.
//!
//! # What reading checks
//!
//! In this order: the magic; the version; that every list decodes and the end mark is there
//! ([`CaptureError::Truncated`] where the bytes run out, [`CaptureError::Decode`] for anything
//! else); that nothing follows the end mark; and that each list's fence value is above the one
//! before it, the first above zero, as a backend's are ([`CaptureError::FenceOrder`]). Writing
//! applies the same fence check, so a capture that was written can be read. The commands
//! themselves are not checked here: a list is checked against the device when it is submitted
//! ([`crate::validate`]).
//!
//! Reading keeps the whole capture in memory, which is a bounded multiple of the file's size
//! (a command takes more room decoded than its shortest encoding).

use crate::backend::Backend;
use crate::command::CommandList;
use crate::error::RenderError;
use crate::id::FenceValue;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// First bytes of every capture, so that a file of another kind is refused before its body is
/// decoded.
pub const MAGIC: [u8; 4] = *b"SQRC";

/// Layout version written into every capture and required when reading one.
pub const VERSION: u32 = 1;

/// Why a capture could not be written, read or replayed.
#[derive(Debug, Error, PartialEq)]
pub enum CaptureError {
    /// Encoding failed.
    #[error("could not encode the capture: {0}")]
    Encode(postcard::Error),
    /// The bytes end before the capture does.
    #[error("the capture ends early")]
    Truncated,
    /// The bytes are not a well-formed encoding (a field out of range).
    #[error("could not decode the capture: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`MAGIC`].
    #[error("not a render capture (bad magic)")]
    BadMagic,
    /// The capture was written with a different layout version.
    #[error("capture layout version {found} is not supported (this build reads {VERSION})")]
    UnsupportedVersion {
        /// The version found in the capture.
        found: u32,
    },
    /// Bytes remain after the capture's end mark.
    #[error("{count} trailing bytes after the capture")]
    TrailingBytes {
        /// How many bytes were left over.
        count: usize,
    },
    /// A list's fence value is not above the one before it (for the first list, above zero).
    #[error("list {list} of the capture has fence value {fence}, not above the one before it")]
    FenceOrder {
        /// The list's place in the capture, from zero.
        list: usize,
        /// Its fence value.
        fence: u64,
    },
    /// A backend refused a list during a replay.
    #[error("list {list} of the capture was refused: {source}")]
    Replay {
        /// The list's place in the capture, from zero.
        list: usize,
        /// Why the backend refused it.
        source: RenderError,
    },
}

/// One submitted list and the fence value it signalled.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapturedList {
    /// The fence value `submit` returned for the list in the captured run.
    pub fence: FenceValue,
    /// The list, with its staging bytes.
    pub list: CommandList,
}

/// The lists a backend was given, in the order they were submitted (module docs).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Capture {
    /// The lists, oldest first.
    pub lists: Vec<CapturedList>,
}

impl Capture {
    /// A capture of `lists`, each with the fence value it signalled, oldest first.
    pub fn new(lists: impl IntoIterator<Item = (FenceValue, CommandList)>) -> Self {
        Self {
            lists: lists
                .into_iter()
                .map(|(fence, list)| CapturedList { fence, list })
                .collect(),
        }
    }

    /// The capture as a file's bytes (module docs, "Layout").
    ///
    /// Errors: [`CaptureError::FenceOrder`] for fence values no backend gives.
    pub fn to_bytes(&self) -> Result<Vec<u8>, CaptureError> {
        check_fences(&self.lists)?;
        let mut bytes = postcard::to_allocvec(&(MAGIC, VERSION)).map_err(CaptureError::Encode)?;
        for list in &self.lists {
            bytes = postcard::to_extend(&Some(list), bytes).map_err(CaptureError::Encode)?;
        }
        postcard::to_extend(&END, bytes).map_err(CaptureError::Encode)
    }

    /// Reads a capture from a file's bytes. Malformed, truncated, foreign, other-version or
    /// over-long input is an `Err` (module docs, "What reading checks").
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CaptureError> {
        let (magic, rest) = take::<[u8; 4]>(bytes)?;
        if magic != MAGIC {
            return Err(CaptureError::BadMagic);
        }
        let (version, mut rest) = take::<u32>(rest)?;
        if version != VERSION {
            return Err(CaptureError::UnsupportedVersion { found: version });
        }
        let mut lists = Vec::new();
        // Each turn takes at least one byte or ends the loop, so the file's length bounds it.
        loop {
            let (next, after) = take::<Option<CapturedList>>(rest)?;
            rest = after;
            match next {
                Some(list) => lists.push(list),
                None => break,
            }
        }
        if !rest.is_empty() {
            return Err(CaptureError::TrailingBytes { count: rest.len() });
        }
        check_fences(&lists)?;
        Ok(Self { lists })
    }

    /// Submits the capture's lists to `backend`, in order, and returns the fence value each
    /// signalled there. On a backend that has submitted nothing yet these are the captured
    /// fence values when the capture starts at its run's first list.
    ///
    /// The backend must already hold the resources the lists name, under the same ids (module
    /// docs). A list the backend refuses ends the replay with [`CaptureError::Replay`], which
    /// says which list; the lists before it stay submitted. The capture's own fence values are
    /// not used. The call does not wait for the lists to complete.
    pub fn replay(&self, backend: &dyn Backend) -> Result<Vec<FenceValue>, CaptureError> {
        self.lists
            .iter()
            .enumerate()
            .map(|(list, captured)| {
                backend
                    .submit(captured.list.clone())
                    .map_err(|source| CaptureError::Replay { list, source })
            })
            .collect()
    }
}

/// The end mark: where a list would start, "no list". Written and read as the `Option` whose
/// `Some` is a list, so the two marks (`0x00`, `0x01`) are `postcard`'s own.
const END: Option<&CapturedList> = None;

/// Reads one value from the front of `bytes` and returns the bytes after it.
fn take<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<(T, &'a [u8]), CaptureError> {
    postcard::take_from_bytes(bytes).map_err(|error| match error {
        postcard::Error::DeserializeUnexpectedEnd => CaptureError::Truncated,
        other => CaptureError::Decode(other),
    })
}

/// Checks that each list's fence value is above the one before it, the first above zero: what
/// writing and reading both require (module docs).
fn check_fences(lists: &[CapturedList]) -> Result<(), CaptureError> {
    let mut before = FenceValue::ZERO;
    for (list, captured) in lists.iter().enumerate() {
        if captured.fence <= before {
            return Err(CaptureError::FenceOrder {
                list,
                fence: captured.fence.get(),
            });
        }
        before = captured.fence;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
