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
//!    [`CaptureError::ListTooBig`], [`CaptureError::TooManyDumps`]); and the record fits what
//!    came before (`Start` first and once, nothing but `End` after `Limit`, accepted `Submit`
//!    fences rising and the first above zero, each `ShaderSource` index the next one and the
//!    sources' charge within [`MAX_SHADER_SOURCES_BYTES`] ([`CaptureError::SourcesTooLarge`]),
//!    each `CreateShader` naming an index already given, no `Create*` returning an id still
//!    live in the capture, at most [`MAX_LIVE_RESOURCES`] held per id kind).
//!
//! An id is *held* from the `Create*` that returns it until its `Destroy`, or, when that
//! `Destroy` names a fence past every accepted list so far, until an accepted list reaches that
//! fence: replay keeps such an id, and the fresh backend's resource, until then (see
//! [`replay`], "Identity"). So the live count bounds the destroys replay holds back as well as
//! the ids in use (`capture::tests::read::held_ids_are_counted_until_a_list_reaches_their_fence`).
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
//! | [`MAX_LIVE_RESOURCES`] | 2^20 per id kind | the ids held (above): the reader's live set and the replay's tables of ids and held destroys; replay waits for its last fence before a create would pass it, so destroys forwarded but not yet reached count too. A guess (spec) |
//! | [`MAX_SHADER_SOURCES_BYTES`] | 1 GiB | every `ShaderSource` the capture gives, each charged its length plus [`SOURCE_ENTRY_BYTES`]: replay keeps them all. A guess, not in the spec (below) |
//! | [`MAX_LIST_DUMPS`] | 1024 | the presents with `dump` set in one list: the [`ReplayFrames`] sink's cap. Not in the spec (below) |
//!
//! [`CaptureLimits`] carries the bounds a test may lower; [`CaptureLimits::SHIPPED`] is the
//! table's values. The text and per-shader bounds are fixed.
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
//! both. Pinned by `capture::tests::bounds::the_bounds_are_the_documented_values`; (1) and (2)
//! by `capture::tests::bounds::each_kind_cap_at_its_edge`,
//! `a_lists_decoded_charge_at_its_edge` and `the_lists_nested_in_a_command_at_their_edge`. (7)
//! [`MAX_SHADER_SOURCES_BYTES`] is not in the spec, whose bounds table caps one source but not
//! how many a capture gives, while replay keeps every source for the shaders that name it; the
//! value is a guess (as much as the largest record), for the owner to rule on. Pinned by
//! `capture::tests::bounds::the_sources_charge_at_its_edge`. (8) [`MAX_LIST_DUMPS`] is not in
//! the spec either: a list with more dumped presents than the [`ReplayFrames`] sink keeps would
//! lose its first frames before replay compares them, and report them missing. Pinned by
//! `capture::tests::bounds::a_lists_dumped_presents_at_their_edge` and
//! `capture::tests::replay::a_list_of_dumped_frames_at_the_sinks_cap_replays_clean`. The bound
//! is per list: frames of an earlier list whose `FrameResult` records come after a later
//! `Submit` (a backend that delivers frames late) share the sink with that list's, so a run of
//! such lists may still pass the cap; replay then reports the evicted frames as
//! [`FindingKind::FrameMissing`].
//!
//! **Other departures from the spec's text, for the owner.** (4) The spec says a `Limit` record
//! "says the capture stopped at the size cap". The capturing backend also closes with `Limit`
//! when any other writer check refuses a record (a bound passed, or a wrapped backend handing out
//! a live id), so the file always ends well formed; it keeps the error for the app to report
//! ([`CapturingBackend::stopped`]; `capture::tests::wrap`). (5) "Nothing but `End` after `Limit`"
//! is a reading check the spec does not list (`capture::tests::read::only_end_follows_limit`).
//! (6) The spec says replay refuses a create past the memory limit "so a capture of many huge
//! textures fails cleanly"; here the create is refused and counted and the replay goes on, as for
//! any other id it does not hold (`capture::tests::replay`, the memory limit test).
//! (9) The spec says a `Destroy`'s `after` "maps to the replay fence of the latest captured
//! list at or below it (zero maps to zero)", and that "the order of calls and lists" must not
//! differ on any backend. Replay holds a `Destroy` whose fence is past every list captured so
//! far until a captured list reaches that fence, and forwards it then, after the `Submit`
//! records that followed it in the capture; mapping it at once would map it to a fence the fresh
//! backend has already passed. The GPU crate names only its last fence or zero, so only a
//! hand-made or hostile capture takes this path
//! (`capture::tests::replay::a_destroy_past_every_captured_list_waits_for_one_to_reach_it`).
//! (10) The spec calls the error for bytes left inside a record `TrailingBytes`. Here that is
//! [`CaptureError::RecordTrailingBytes`], and [`CaptureError::TrailingBytes`] is the error for
//! bytes after the `End` record (`capture::tests::read`).
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
    PipelineOutcome, PipelineResult, ReadbackResult, Record, RecordKind, ShaderCode, SourceRecord,
    Start, Submit, TargetRecord, WaitSetting,
};
pub use replay::{
    Finding, FindingKind, MAX_REPLAY_FRAMES, ReplayError, ReplayFrames, ReplayOptions,
    ReplayReport, ResolveError, ShaderResolver, map_ids, replay,
};
pub use wrap::CapturingBackend;
pub use write::{CaptureBuffer, CaptureSink, CaptureWriter, EncodedList};

use crate::id::{IdKind, ResourceId};
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

/// The most ids of one kind held at once in a capture (module docs, "What reading checks, in
/// order"), and so in a replay's tables.
pub const MAX_LIVE_RESOURCES: u64 = 1 << 20;

/// The most all of a capture's shader sources may be charged ([`SOURCE_ENTRY_BYTES`]): replay
/// keeps every source. A guess, as much as [`MAX_RECORD_BYTES`] (module docs, "Bounds").
pub const MAX_SHADER_SOURCES_BYTES: u64 = 1 << 30;

/// What a replay's table of sources costs per source beyond the source's bytes, on the 64-bit
/// hosts starquake runs on: the `Arc<[u8]>` in the table (a pointer and a length) and the
/// `Arc`'s two counts. Each source is charged at least this, so an empty one is not free.
pub const SOURCE_ENTRY_BYTES: u64 = 32;

/// The most presents with `dump` set that one list may hold: the [`ReplayFrames`] sink's cap,
/// so that replay keeps every frame one list dumps (module docs, "Bounds").
pub const MAX_LIST_DUMPS: u64 = MAX_REPLAY_FRAMES as u64;

/// What a source of `len` bytes is charged against [`CaptureLimits::source_bytes`].
pub(crate) fn source_charge(len: usize) -> u64 {
    write::len64(len).saturating_add(SOURCE_ENTRY_BYTES)
}

/// A field with its own bound, as [`CaptureError::TooLarge`] names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoundedField {
    /// The `Start` record's build id.
    BuildId,
    /// The `Start` record's backend name.
    BackendName,
    /// A shader's entry name.
    EntryName,
    /// A refusal's message.
    RefusalMessage,
    /// A shader origin's bytes.
    ShaderSource,
    /// SPIR-V code stored as it is.
    SpirVCode,
    /// MSL code stored as it is.
    MslCode,
    /// A draw's bindings.
    DrawBindings,
    /// A draw's vertex buffers.
    DrawVertexBuffers,
    /// A draw's viewports.
    DrawViewports,
    /// A draw's scissors.
    DrawScissors,
    /// A dispatch's bindings.
    DispatchBindings,
}

impl BoundedField {
    /// The field in words ("entry name").
    pub const fn name(self) -> &'static str {
        match self {
            Self::BuildId => "build id",
            Self::BackendName => "backend name",
            Self::EntryName => "entry name",
            Self::RefusalMessage => "refusal message",
            Self::ShaderSource => "shader source",
            Self::SpirVCode => "SPIR-V code",
            Self::MslCode => "MSL code",
            Self::DrawBindings => "draw bindings",
            Self::DrawVertexBuffers => "draw vertex buffers",
            Self::DrawViewports => "draw viewports",
            Self::DrawScissors => "draw scissors",
            Self::DispatchBindings => "dispatch bindings",
        }
    }

    /// Whether the field is a list nested in a command, whose length counts elements rather
    /// than bytes.
    pub const fn is_list(self) -> bool {
        match self {
            Self::BuildId
            | Self::BackendName
            | Self::EntryName
            | Self::RefusalMessage
            | Self::ShaderSource
            | Self::SpirVCode
            | Self::MslCode => false,
            Self::DrawBindings
            | Self::DrawVertexBuffers
            | Self::DrawViewports
            | Self::DrawScissors
            | Self::DispatchBindings => true,
        }
    }
}

/// In words, as errors print it: "entry name".
impl std::fmt::Display for BoundedField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// [`CaptureError::TooLarge`] after its record, in words: bytes for a text or code, elements
/// for a nested list.
fn too_large(what: BoundedField, len: u64, max: u64) -> String {
    if what.is_list() {
        format!("the {what} list holds {len} elements, more than the {max} allowed")
    } else {
        format!("the {what} is {len} bytes long, more than the {max} allowed")
    }
}

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
    /// [`MAX_SHADER_SOURCES_BYTES`].
    pub source_bytes: u64,
    /// [`MAX_LIST_DUMPS`].
    pub list_dumps: u64,
}

impl CaptureLimits {
    /// The shipped bounds.
    pub const SHIPPED: Self = Self {
        record_bytes: MAX_RECORD_BYTES,
        description_bytes: MAX_DESCRIPTION_BYTES,
        list_commands: MAX_LIST_COMMANDS,
        list_decoded_bytes: MAX_LIST_DECODED_BYTES,
        live_resources: MAX_LIVE_RESOURCES,
        source_bytes: MAX_SHADER_SOURCES_BYTES,
        list_dumps: MAX_LIST_DUMPS,
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
            RecordKind::CreateShader => {
                MAX_SHADER_CODE_BYTES.saturating_add(self.description_bytes)
            }
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
    #[error("record {record}: {}", too_large(*what, *len, *max))]
    TooLarge {
        /// The record's position.
        record: u64,
        /// The field; for a list nested in a command ([`BoundedField::is_list`]), its length is
        /// counted in elements, not bytes.
        what: BoundedField,
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
    /// A list holds more presents with `dump` set than [`CaptureLimits::list_dumps`].
    #[error(
        "record {record}: command {command} is dumped present {count}, more than the {max} allowed"
    )]
    TooManyDumps {
        /// The record's position.
        record: u64,
        /// The command, from zero, that passed the bound.
        command: u64,
        /// The dumped presents up to and including it.
        count: u64,
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
    /// A `ShaderSource` would take the sources' charge past [`CaptureLimits::source_bytes`].
    #[error(
        "record {record}: the shader sources would be charged {needed} bytes, more than the {max} allowed"
    )]
    SourcesTooLarge {
        /// The record's position.
        record: u64,
        /// The charge with this source ([`SOURCE_ENTRY_BYTES`] per source), saturating.
        needed: u64,
        /// The bound.
        max: u64,
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
    /// A `Create*` would make more ids of one kind held than [`CaptureLimits::live_resources`]
    /// (module docs, "What reading checks, in order").
    #[error("record {record}: more than {max} {kind} ids would be live")]
    TooManyLive {
        /// The record's position.
        record: u64,
        /// The id kind.
        kind: IdKind,
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
