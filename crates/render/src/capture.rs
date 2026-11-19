//! Render capture v2: everything a backend was told and told back, as a versioned stream of
//! records that replays on a freshly built backend of any kind (docs/specs/render-backend.md,
//! "Render capture v2"; the owner's decisions 15 to 19 of 2026-11-18; slice R11b).
//!
//! - [`CapturingBackend`] wraps a real backend, records every call that changes backend state
//!   in the order the calls reach it, forwards it unchanged, and hands each encoded record to
//!   the [`CaptureSink`] the app installs. This crate does no file I/O.
//! - [`CaptureWriter`] encodes records and applies every check the reader applies, so that
//!   whatever was written can be read.
//! - [`CaptureReader`] reads a capture from a [`std::io::Read`] one record at a time, with a
//!   bound on every length.
//! - [`replay`] feeds a capture to a fresh backend, mapping the capture's ids and fence values
//!   to the backend's, and compares the results the capture holds.
//! - `CaptureBuilder` (behind the `test-support` feature) writes captures through the real
//!   writer with no backend, for other crates' tests.
//!
//! Version 1 (slice R11's first part) is replaced, not kept beside: a version-1 file is refused
//! with [`CaptureError::UnsupportedVersion`] (decision 18).
//!
//! # Layout (version 2)
//!
//! Everything is `postcard`: unsigned integers are little-endian base-128 varints.
//!
//! | Bytes | What |
//! |---|---|
//! | 4 | [`MAGIC`], `SQRC` |
//! | varint | [`VERSION`] (`u32`), 2 |
//! | per record: 1 | the record's tag ([`RecordKind::tag`]) |
//! | per record: varint | the body's length in bytes (`u32`) |
//! | per record: the rest | the body: the postcard encoding of that kind's struct |
//! | last record | `End` (tag 0, length 0) |
//!
//! The tags are fixed for this version and pinned by `capture::tests::golden::tags`. Every body
//! is its struct's fields in declaration order; a `Submit` body is the list (its command count,
//! then each command) followed by the result, and every `Create*` body ends with its result, so
//! that a writer can encode a call's input before the call consumes it. The bytes of every record
//! kind are pinned by `capture::tests::golden`; any change to them, or to the field order or
//! types of anything a record holds ([`Caps`](crate::Caps), the descriptions, [`Command`](crate::Command)), needs a new
//! [`VERSION`] in the same change.
//!
//! # What a capture holds
//!
//! In the order the backend saw them: `Start` (build id, backend name, [`Caps`](crate::Caps), the
//! wait-while-compiling setting and the memory limit), then one record per state-changing call
//! (`Create*` with the id returned or the refusal's message, `ShaderSource` once per distinct
//! shader origin, `Destroy` with its fence, `Submit` with its staging bytes and the fence or
//! refusal, `PresentTarget`), the results replay compares (`ReadbackResult`, `PipelineOutcome`,
//! `FrameResult`), an optional `Limit` when the capture stopped early, and `End`. The spec's
//! table ("What a capture holds") lists which trait call makes which record.
//!
//! # What reading checks, in order
//!
//! 1. The magic ([`CaptureError::BadMagic`]; a file too short to hold it is
//!    [`CaptureError::Truncated`] when it is the start of the magic).
//! 2. The version ([`CaptureError::UnsupportedVersion`]), before anything after it is read.
//! 3. Per record: the tag is known ([`CaptureError::UnknownRecord`]); the length is at most the
//!    kind's cap ([`CaptureError::RecordTooLarge`], below); the length is within the bytes left
//!    ([`CaptureError::Truncated`]); the body decodes within its length
//!    ([`CaptureError::Decode`], [`CaptureError::RecordTrailingBytes`], and for the fields with
//!    their own bounds [`CaptureError::TooLarge`], [`CaptureError::ListTooLong`],
//!    [`CaptureError::ListTooBig`]); and the record fits what came before (`Start` first and
//!    once, nothing but `End` after `Limit`, accepted `Submit` fences rising and the first above
//!    zero, each `ShaderSource` index the next one, each `CreateShader` naming an index already
//!    given, no `Create*` returning an id still live in the capture, at most
//!    [`MAX_LIVE_RESOURCES`] live per id kind).
//! 4. Nothing after `End` ([`CaptureError::TrailingBytes`]).
//!
//! The order checks are applied in this order: `Start` first, then `Start` once, then nothing
//! after `Limit`, then the kind's own check. Writing applies the same checks, then the size cap
//! ([`CaptureError::OverLimit`]).
//!
//! A failed read is sticky: every later [`CaptureReader::next_record`] returns the same error,
//! never a clean end.
//!
//! # Bounds
//!
//! A hostile capture must not make replay allocate without bound. Every length is checked
//! against its cap and against the bytes left before anything is allocated for it.
//!
//! | Bound | Value | What it bounds |
//! |---|---|---|
//! | [`MAX_RECORD_BYTES`] | 1 GiB | any record's body; the cap of `Submit` and `ReadbackResult`. A guess (spec) |
//! | [`MAX_DESCRIPTION_BYTES`] | 64 KiB | the body of every other kind, plus the code or source for `CreateShader` and `ShaderSource`. A guess, and a divergence (below) |
//! | [`MAX_LIST_COMMANDS`] | 2^20 | a list's command count, checked before any command is decoded. A guess (spec) |
//! | [`MAX_LIST_DECODED_BYTES`] | 2 GiB | a list's decoded size, charged after each command. A guess (below) |
//! | [`crate::command::MAX_COMMAND_BINDINGS`] and the guest's counts | | each list nested in one command, while it is decoded |
//! | [`MAX_SHADER_SOURCE_BYTES`] | 2 MiB | one shader origin's bytes. A guess (below) |
//! | [`MAX_SHADER_CODE_BYTES`] | 16 MiB | SPIR-V (four bytes a word) or MSL stored as it is. A guess (spec) |
//! | [`MAX_TEXT_BYTES`] | 4 KiB | build id, backend name, entry names, refusal messages (spec) |
//! | [`MAX_LIVE_RESOURCES`] | 2^20 per id kind | the reader's live set and the replay's tables. A guess (spec) |
//!
//! [`CaptureLimits`] carries the bounds a test may lower; [`CaptureLimits::SHIPPED`] is the
//! table's values. The text and shader bounds are fixed.
//!
//! **Divergences from the spec's bounds, for the owner.** (1) The spec caps every record at
//! `MAX_RECORD_BYTES`. Here only `Submit` and `ReadbackResult` may be that large; every other
//! kind is capped at [`MAX_DESCRIPTION_BYTES`] (with the code or source added for the two shader
//! kinds), so that a description record, whose lists decode several times larger than they
//! encode, cannot cost gigabytes. Pinned by `capture::tests::bounds::kind_caps`. (2)
//! [`MAX_LIST_DECODED_BYTES`] and the caps on nested lists are not in the spec: the spec's
//! command count bounds the commands but not the lists inside a draw, whose elements take up to
//! eight times their encoded size. (3) The spec says the render capture uses slice 20's own
//! maximum record size for [`MAX_SHADER_SOURCE_BYTES`]; this crate cannot depend on the shader
//! crate, so the value here is a guess (65,536 instructions of 16 bytes, 1 MiB, doubled for the
//! header and answers), which R11c should pin against slice 20's constant in a crate that sees
//! both.
//!
//! **Memory while reading.** The reader holds one record. A `Submit` takes at most its
//! [`MAX_LIST_DECODED_BYTES`] charge (each command is charged `size_of::<Command>()` plus its
//! nested elements and staging bytes), plus one command's overshoot, and a growing list may
//! hold up to twice its charge. A `CreateShader` with SPIR-V decodes up to four times its body
//! (each word is a varint of one to five bytes) before its size is checked. Every other record
//! is at most its body's length times a small factor.
//!
//! # Identity
//!
//! Replay maps ids and fence values (decision 16): see [`replay`].

mod builder;
mod order;
mod read;
mod record;
mod replay;
mod wrap;
mod write;

#[cfg(any(test, feature = "test-support"))]
pub use builder::CaptureBuilder;
pub use read::CaptureReader;
pub use record::{
    CapturedTarget, Create, CreateShader, CreateView, Created, Destroy, FrameResult, Limit,
    PipelineOutcome, PipelineResult, ReadbackResult, Record, RecordKind, ShaderCode,
    SourceRecord, Start, Submit, TargetRecord, WaitSetting,
};
pub use replay::{
    Finding, FindingKind, ReplayError, ReplayFrames, ReplayOptions, ReplayReport, ResolveError,
    ShaderResolver, map_ids, replay,
};
pub use wrap::CapturingBackend;
pub use write::{CaptureBuffer, CaptureSink, CaptureWriter, EncodedList};

use crate::id::ResourceId;
use thiserror::Error;

/// First bytes of every capture, so that a file of another kind is refused before its body is
/// read.
pub const MAGIC: [u8; 4] = *b"SQRC";

/// Layout version written into every capture and required when reading one.
pub const VERSION: u32 = 2;

/// The largest body of any record, and of a `Submit` or `ReadbackResult` (module docs, "Bounds").
pub const MAX_RECORD_BYTES: u64 = 1 << 30;

/// The largest body of every other record kind, before the code or source of the two shader
/// kinds is added (module docs, "Bounds"). A guess: a `Start` with its two texts at
/// [`MAX_TEXT_BYTES`] and its [`Caps`](crate::Caps) is under 9 KiB, and a description well under that.
pub const MAX_DESCRIPTION_BYTES: u64 = 64 << 10;

/// The most commands one list may hold.
pub const MAX_LIST_COMMANDS: u64 = 1 << 20;

/// The most memory one decoded list may be charged (module docs, "Memory while reading"). A
/// guess: the largest record's staging bytes, plus 2^20 commands of 1 KiB each.
pub const MAX_LIST_DECODED_BYTES: u64 = MAX_RECORD_BYTES + MAX_LIST_COMMANDS * 1024;

/// The largest shader origin (one slice 20 record). A guess (module docs, "Bounds").
pub const MAX_SHADER_SOURCE_BYTES: u64 = 2 << 20;

/// The largest SPIR-V or MSL code stored as it is.
pub const MAX_SHADER_CODE_BYTES: u64 = 16 << 20;

/// The longest text: build id, backend name, entry name, refusal message.
pub const MAX_TEXT_BYTES: u64 = 4 << 10;

/// The most ids of one kind live at once in a capture, and so in a replay's tables.
pub const MAX_LIVE_RESOURCES: u64 = 1 << 20;

/// The bounds a reader and a writer apply that a test may lower (module docs, "Bounds").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureLimits {
    /// [`MAX_RECORD_BYTES`].
    pub record_bytes: u64,
    /// [`MAX_DESCRIPTION_BYTES`].
    pub description_bytes: u64,
    /// [`MAX_LIST_COMMANDS`].
    pub list_commands: u64,
    /// [`MAX_LIST_DECODED_BYTES`].
    pub list_decoded_bytes: u64,
    /// [`MAX_LIVE_RESOURCES`].
    pub live_resources: u64,
}

impl CaptureLimits {
    /// The shipped bounds.
    pub const SHIPPED: Self = Self {
        record_bytes: MAX_RECORD_BYTES,
        description_bytes: MAX_DESCRIPTION_BYTES,
        list_commands: MAX_LIST_COMMANDS,
        list_decoded_bytes: MAX_LIST_DECODED_BYTES,
        live_resources: MAX_LIVE_RESOURCES,
    };

    /// The largest body a record of `kind` may have: the kind's own cap (module docs, "Bounds"),
    /// never more than [`CaptureLimits::record_bytes`].
    pub const fn max_body(&self, kind: RecordKind) -> u64 {
        let own = match kind {
            RecordKind::End => 0,
            RecordKind::Submit | RecordKind::ReadbackResult => self.record_bytes,
            RecordKind::ShaderSource => {
                MAX_SHADER_SOURCE_BYTES.saturating_add(self.description_bytes)
            }
            RecordKind::CreateShader => MAX_SHADER_CODE_BYTES.saturating_add(self.description_bytes),
            RecordKind::Start
            | RecordKind::CreateBuffer
            | RecordKind::CreateTexture
            | RecordKind::CreateView
            | RecordKind::CreateSampler
            | RecordKind::CreateGraphicsPipeline
            | RecordKind::CreateComputePipeline
            | RecordKind::Destroy
            | RecordKind::PresentTarget
            | RecordKind::PipelineOutcome
            | RecordKind::FrameResult
            | RecordKind::Limit => self.description_bytes,
        };
        if own < self.record_bytes {
            own
        } else {
            self.record_bytes
        }
    }
}

/// Why a capture could not be written or read. Every record is named by its position in the
/// capture, from zero (the `Start` record).
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CaptureError {
    /// Encoding failed. Postcard refuses to write only a sequence whose length is not known when
    /// it starts, which nothing in a record is, and a full buffer, which a `Vec` never is; the
    /// variant is here so that this module does not unwrap.
    #[error("could not encode record {record}: {source}")]
    Encode {
        /// The record's position.
        record: u64,
        /// Postcard's error.
        source: postcard::Error,
    },
    /// The bytes end before the capture does.
    #[error("the capture ends early")]
    Truncated,
    /// The bytes do not start with [`MAGIC`].
    #[error("not a render capture (bad magic)")]
    BadMagic,
    /// The capture was written with another layout version; version 1 is refused too
    /// (decision 18).
    #[error("capture layout version {found} is not supported (this build reads {VERSION})")]
    UnsupportedVersion {
        /// The version found in the capture.
        found: u32,
    },
    /// The version is not a well-formed varint.
    #[error("could not decode the capture's version: {0}")]
    HeaderDecode(postcard::Error),
    /// A record's tag names no record kind of this version.
    #[error("record {record} has unknown tag {tag}")]
    UnknownRecord {
        /// The tag.
        tag: u8,
        /// The record's position.
        record: u64,
    },
    /// A record's length is past its kind's cap.
    #[error("record {record} ({kind}) is {len} bytes long, more than the {max} allowed")]
    RecordTooLarge {
        /// The record's position.
        record: u64,
        /// Its kind.
        kind: RecordKind,
        /// Its length.
        len: u64,
        /// The kind's cap ([`CaptureLimits::max_body`]).
        max: u64,
    },
    /// A record's length or body is not a well-formed encoding (a field out of range, or the
    /// body ends inside a field).
    #[error("could not decode record {record}: {source}")]
    Decode {
        /// The record's position.
        record: u64,
        /// Postcard's error.
        source: postcard::Error,
    },
    /// A record's body decoded in fewer bytes than its length.
    #[error("record {record} has {count} bytes left over after its body")]
    RecordTrailingBytes {
        /// The record's position.
        record: u64,
        /// How many bytes were left.
        count: u64,
    },
    /// A field with its own bound is past it.
    #[error("record {record}: the {what} is {len} bytes long, more than the {max} allowed")]
    TooLarge {
        /// The record's position.
        record: u64,
        /// The field, in words ("entry name"); for a list nested in a command, its length is
        /// counted in elements, not bytes.
        what: &'static str,
        /// Its length.
        len: u64,
        /// Its bound.
        max: u64,
    },
    /// A list holds more commands than [`CaptureLimits::list_commands`].
    #[error("record {record}: the list holds {count} commands, more than the {max} allowed")]
    ListTooLong {
        /// The record's position.
        record: u64,
        /// The command count.
        count: u64,
        /// The bound.
        max: u64,
    },
    /// A list's decoded size passed [`CaptureLimits::list_decoded_bytes`].
    #[error("record {record}: command {command} takes the list past {max} bytes decoded")]
    ListTooBig {
        /// The record's position.
        record: u64,
        /// The command, from zero, whose charge passed the bound.
        command: u64,
        /// The bound.
        max: u64,
    },
    /// A record other than `Start` comes first.
    #[error("record {record} ({kind}) comes before the Start record")]
    MissingStart {
        /// The record's position.
        record: u64,
        /// Its kind.
        kind: RecordKind,
    },
    /// A second `Start`.
    #[error("record {record} is a second Start record")]
    DuplicateStart {
        /// The record's position.
        record: u64,
    },
    /// A record other than `End` follows `Limit`.
    #[error("record {record} ({kind}) follows the Limit record")]
    AfterLimit {
        /// The record's position.
        record: u64,
        /// Its kind.
        kind: RecordKind,
    },
    /// An accepted list's fence value is not above the one before it (for the first, above
    /// zero).
    #[error("record {record}: fence value {fence} is not above the one before it")]
    FenceOrder {
        /// The record's position.
        record: u64,
        /// Its fence value.
        fence: u64,
    },
    /// A `ShaderSource` index is not the next one.
    #[error("record {record}: shader source {index} should be source {expected}")]
    SourceIndex {
        /// The record's position.
        record: u64,
        /// The index given.
        index: u32,
        /// The next index.
        expected: u32,
    },
    /// A `CreateShader` names a source index not yet given.
    #[error("record {record}: shader source {index} has not been given")]
    UnknownSource {
        /// The record's position.
        record: u64,
        /// The index named.
        index: u32,
    },
    /// A `Create*` returned an id that is still live in the capture.
    #[error("record {record} creates {id}, which is still live")]
    IdLive {
        /// The record's position.
        record: u64,
        /// The id.
        id: ResourceId,
    },
    /// A `Create*` would make more ids of one kind live than [`CaptureLimits::live_resources`].
    #[error("record {record}: more than {max} {kind} ids would be live")]
    TooManyLive {
        /// The record's position.
        record: u64,
        /// The id kind, in words.
        kind: &'static str,
        /// The bound.
        max: u64,
    },
    /// Bytes remain after the `End` record.
    #[error("{count} bytes follow the capture's End record")]
    TrailingBytes {
        /// How many bytes were left, by the length the reader was given.
        count: u64,
    },
    /// Reading the input failed other than by ending early.
    #[error("could not read the capture: {message}")]
    Io {
        /// The I/O error's kind.
        kind: std::io::ErrorKind,
        /// Its message.
        message: String,
    },
    /// Writing a record would take the capture past its size limit, with room kept for the
    /// closing `Limit` and `End`.
    #[error("record {record} would take the capture to {needed} bytes, past its limit of {limit}")]
    OverLimit {
        /// The record's position.
        record: u64,
        /// The capture's size with the record and the closing records.
        needed: u64,
        /// The limit.
        limit: u64,
    },
    /// A writer's size limit cannot hold the header, the `Start` record and the closing records.
    #[error("a capture limit of {limit} bytes cannot hold its start, which needs {needed}")]
    LimitTooSmall {
        /// The limit.
        limit: u64,
        /// The bytes needed.
        needed: u64,
    },
    /// The writer has written `End`, or `Limit` and `End`.
    #[error("the capture has ended")]
    Finished,
}

#[cfg(test)]
mod tests;
