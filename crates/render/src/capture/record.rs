//! The records of a version-2 capture: their kinds and tags, the struct each body holds, and how
//! a body is encoded and decoded (module docs of [`crate::capture`], "Layout").

use super::{
    CaptureError, CaptureLimits, MAX_SHADER_CODE_BYTES, MAX_SHADER_SOURCE_BYTES, MAX_TEXT_BYTES,
};
use crate::backend::{PresentTarget, Readback};
use crate::caps::Caps;
use crate::command::{Command, CommandList, check_nested, decoded_bytes};
use crate::frame::{Frame, FramePixels};
use crate::id::{
    BufferId, FenceValue, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId, TextureId,
    ViewId,
};
use crate::pipeline::{ComputePipelineDesc, GraphicsPipelineDesc};
use crate::resource::{BufferDesc, SamplerDesc, TextureDesc, ViewDesc};
use crate::shader::{ShaderInterface, ShaderStage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// The kind of a record, which its tag names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RecordKind {
    /// The last record.
    End,
    /// The first record: who wrote the capture, on what device.
    Start,
    /// `create_buffer`.
    CreateBuffer,
    /// `create_texture`.
    CreateTexture,
    /// `create_view`.
    CreateView,
    /// `create_sampler`.
    CreateSampler,
    /// One distinct shader origin, before the first `CreateShader` naming it.
    ShaderSource,
    /// `create_shader`.
    CreateShader,
    /// `create_graphics_pipeline`.
    CreateGraphicsPipeline,
    /// `create_compute_pipeline`.
    CreateComputePipeline,
    /// `destroy`.
    Destroy,
    /// `submit`.
    Submit,
    /// `set_present_target`.
    PresentTarget,
    /// What a `take_readback` returned.
    ReadbackResult,
    /// The first `Ready` or `Failed` a `pipeline_status` reported for a pipeline.
    PipelineOutcome,
    /// A dumped frame's hash.
    FrameResult,
    /// The capture stopped early.
    Limit,
}

impl RecordKind {
    /// Every kind, in tag order.
    pub const ALL: [Self; 17] = [
        Self::End,
        Self::Start,
        Self::CreateBuffer,
        Self::CreateTexture,
        Self::CreateView,
        Self::CreateSampler,
        Self::ShaderSource,
        Self::CreateShader,
        Self::CreateGraphicsPipeline,
        Self::CreateComputePipeline,
        Self::Destroy,
        Self::Submit,
        Self::PresentTarget,
        Self::ReadbackResult,
        Self::PipelineOutcome,
        Self::FrameResult,
        Self::Limit,
    ];

    /// The kind's tag, fixed for this version.
    pub const fn tag(self) -> u8 {
        match self {
            Self::End => 0,
            Self::Start => 1,
            Self::CreateBuffer => 2,
            Self::CreateTexture => 3,
            Self::CreateView => 4,
            Self::CreateSampler => 5,
            Self::ShaderSource => 6,
            Self::CreateShader => 7,
            Self::CreateGraphicsPipeline => 8,
            Self::CreateComputePipeline => 9,
            Self::Destroy => 10,
            Self::Submit => 11,
            Self::PresentTarget => 12,
            Self::ReadbackResult => 13,
            Self::PipelineOutcome => 14,
            Self::FrameResult => 15,
            Self::Limit => 16,
        }
    }

    /// The kind a tag names, if any.
    pub fn from_tag(tag: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.tag() == tag)
    }

    /// The kind's name, as errors print it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::End => "End",
            Self::Start => "Start",
            Self::CreateBuffer => "CreateBuffer",
            Self::CreateTexture => "CreateTexture",
            Self::CreateView => "CreateView",
            Self::CreateSampler => "CreateSampler",
            Self::ShaderSource => "ShaderSource",
            Self::CreateShader => "CreateShader",
            Self::CreateGraphicsPipeline => "CreateGraphicsPipeline",
            Self::CreateComputePipeline => "CreateComputePipeline",
            Self::Destroy => "Destroy",
            Self::Submit => "Submit",
            Self::PresentTarget => "PresentTarget",
            Self::ReadbackResult => "ReadbackResult",
            Self::PipelineOutcome => "PipelineOutcome",
            Self::FrameResult => "FrameResult",
            Self::Limit => "Limit",
        }
    }
}

impl fmt::Display for RecordKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The wait-while-compiling setting the captured run used, as the `Start` record holds it. The
/// GPU crate's own setting (`WaitWhileCompiling` in its back module) has the same two values;
/// the conversion between them is the wiring's (R11c), in the GPU crate, which sees both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WaitSetting {
    /// A draw waits for its pipeline.
    Wait,
    /// A draw whose pipeline is still compiling is skipped.
    Skip,
}

/// The `Start` record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Start {
    /// The writer's build id, at most [`MAX_TEXT_BYTES`].
    pub build: String,
    /// The captured backend's name, at most [`MAX_TEXT_BYTES`].
    pub backend: String,
    /// What the captured device could do.
    pub caps: Caps,
    /// The captured run's wait-while-compiling setting.
    pub wait: WaitSetting,
    /// The captured backend's configured memory limit, in bytes.
    pub memory_limit: u64,
}

/// What a call returned: the value, or the refusal's message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Created<T> {
    /// The call succeeded with this value.
    Made(T),
    /// The call was refused; the message is at most [`MAX_TEXT_BYTES`].
    Refused(String),
}

impl<T: Copy> Created<T> {
    /// The record of `result`, with the refusal's message cut to [`MAX_TEXT_BYTES`] at a
    /// character boundary (a capture must not stop because a message is long).
    pub fn of<E: fmt::Display>(result: &Result<T, E>) -> Self {
        match result {
            Ok(value) => Self::Made(*value),
            Err(error) => Self::Refused(bounded_text(error.to_string())),
        }
    }

    /// The value made, if any.
    pub const fn made(&self) -> Option<T> {
        match self {
            Self::Made(value) => Some(*value),
            Self::Refused(_) => None,
        }
    }
}

/// `text` cut to at most [`MAX_TEXT_BYTES`], at a character boundary.
pub(crate) fn bounded_text(mut text: String) -> String {
    let max = usize::try_from(MAX_TEXT_BYTES).unwrap_or(usize::MAX);
    if text.len() > max {
        let end = (0..=max)
            .rev()
            .find(|&i| text.is_char_boundary(i))
            .unwrap_or(0);
        text.truncate(end);
    }
    text
}

/// A `create_buffer`, `create_texture`, `create_sampler` or `create_*_pipeline` record: the
/// description, then the id returned or the refusal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Create<D, I> {
    /// The description the call was given. A pipeline's shader ids are the capture's.
    pub desc: D,
    /// What the call returned.
    pub result: Created<I>,
}

/// A `create_view` record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateView {
    /// The texture, by the capture's id.
    pub texture: TextureId,
    /// The description.
    pub desc: ViewDesc,
    /// What the call returned.
    pub result: Created<ViewId>,
}

/// A `ShaderSource` record: one distinct shader origin's bytes, under the next index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRecord {
    /// The index, one more than the last source's (the first is 0).
    pub index: u32,
    /// The shader crate's encoded record, at most [`MAX_SHADER_SOURCE_BYTES`].
    pub bytes: Vec<u8>,
}

/// How a `CreateShader` record holds the shader's code (decision 15).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShaderCode {
    /// Translated from a shader origin: the `ShaderSource` index and the stage's index in its
    /// shader set. Replay retranslates it through a [`super::ShaderResolver`].
    Origin {
        /// The `ShaderSource` index.
        source: u32,
        /// The stage's index in the source's shader set.
        stage_index: u32,
    },
    /// SPIR-V words the backend received, stored as they are.
    SpirV(Vec<u32>),
    /// MSL text the backend received, stored as it is.
    Msl(String),
    /// A naga module with no origin (a test's hand-built module): replay refuses the shader.
    Missing,
}

/// A `CreateShader` record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateShader {
    /// The stage.
    pub stage: ShaderStage,
    /// The entry point's name, at most [`MAX_TEXT_BYTES`].
    pub entry: String,
    /// The interface tables, as the call was given them.
    pub interface: ShaderInterface,
    /// The code.
    pub code: ShaderCode,
    /// What the call returned.
    pub result: Created<ShaderId>,
}

/// A `Destroy` record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Destroy {
    /// The resource, by the capture's id.
    pub id: ResourceId,
    /// The fence value after which it is destroyed, the capture's.
    pub after: FenceValue,
}

/// A `Submit` record: the list with its staging bytes, then the fence value returned or the
/// refusal.
#[derive(Clone, Debug, PartialEq)]
pub struct Submit {
    /// The list, naming the capture's ids.
    pub list: CommandList,
    /// What `submit` returned.
    pub result: Created<FenceValue>,
}

/// Where presents went, without a window's handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapturedTarget {
    /// [`PresentTarget::Offscreen`].
    Offscreen,
    /// A window (slice R8). Replay presents offscreen whatever the capture says.
    Window,
}

impl CapturedTarget {
    /// The captured form of `target`.
    pub const fn of(target: &PresentTarget) -> Self {
        match target {
            PresentTarget::Offscreen => Self::Offscreen,
        }
    }
}

/// A `PresentTarget` record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetRecord {
    /// The target.
    pub target: CapturedTarget,
    /// What `set_present_target` returned.
    pub result: Created<()>,
}

/// A `ReadbackResult` record: what one `take_readback` returned.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadbackResult {
    /// The readback id (chosen by the caller, so never mapped).
    pub id: ReadbackId,
    /// The bytes, [`Readback::Unknown`], or `None` when nothing was waiting.
    pub value: Option<Readback>,
}

/// A pipeline's settled status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineResult {
    /// Usable.
    Ready,
    /// Failed.
    Failed,
}

/// A `PipelineOutcome` record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineOutcome {
    /// The pipeline, by the capture's id.
    pub pipeline: PipelineId,
    /// The first settled status seen for it.
    pub outcome: PipelineResult,
}

/// A `FrameResult` record: a dumped frame, by its SHA-256.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameResult {
    /// The frame number from the present.
    pub frame: u64,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// SHA-256 of the RGBA8 pixels, or `None` when the backend did not know them.
    pub sha256: Option<[u8; 32]>,
}

impl FrameResult {
    /// The record of `frame`: the one hashing both the wrapper and the replay use.
    pub fn of(frame: &Frame) -> Self {
        Self {
            frame: frame.frame,
            width: frame.width,
            height: frame.height,
            sha256: match &frame.pixels {
                FramePixels::Rgba8(rgba) => Some(Sha256::digest(rgba).into()),
                FramePixels::Unknown => None,
            },
        }
    }
}

/// A `Limit` record: the capture stopped early.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limit {
    /// The fence value of the last list the capture holds (zero for none).
    pub fence: FenceValue,
}

/// One record.
#[derive(Clone, Debug, PartialEq)]
#[expect(
    clippy::large_enum_variant,
    reason = "records are read and written one at a time; boxing the large ones would only move \
              the same bytes to the heap"
)]
pub enum Record {
    /// The end.
    End,
    /// The start.
    Start(Start),
    /// `create_buffer`.
    CreateBuffer(Create<BufferDesc, BufferId>),
    /// `create_texture`.
    CreateTexture(Create<TextureDesc, TextureId>),
    /// `create_view`.
    CreateView(CreateView),
    /// `create_sampler`.
    CreateSampler(Create<SamplerDesc, SamplerId>),
    /// A shader origin.
    ShaderSource(SourceRecord),
    /// `create_shader`.
    CreateShader(CreateShader),
    /// `create_graphics_pipeline`.
    CreateGraphicsPipeline(Create<GraphicsPipelineDesc, PipelineId>),
    /// `create_compute_pipeline`.
    CreateComputePipeline(Create<ComputePipelineDesc, PipelineId>),
    /// `destroy`.
    Destroy(Destroy),
    /// `submit`.
    Submit(Submit),
    /// `set_present_target`.
    PresentTarget(TargetRecord),
    /// `take_readback`.
    ReadbackResult(ReadbackResult),
    /// A pipeline's settled status.
    PipelineOutcome(PipelineOutcome),
    /// A dumped frame.
    FrameResult(FrameResult),
    /// The capture stopped early.
    Limit(Limit),
}

impl Record {
    /// The record's kind.
    pub const fn kind(&self) -> RecordKind {
        match self {
            Self::End => RecordKind::End,
            Self::Start(_) => RecordKind::Start,
            Self::CreateBuffer(_) => RecordKind::CreateBuffer,
            Self::CreateTexture(_) => RecordKind::CreateTexture,
            Self::CreateView(_) => RecordKind::CreateView,
            Self::CreateSampler(_) => RecordKind::CreateSampler,
            Self::ShaderSource(_) => RecordKind::ShaderSource,
            Self::CreateShader(_) => RecordKind::CreateShader,
            Self::CreateGraphicsPipeline(_) => RecordKind::CreateGraphicsPipeline,
            Self::CreateComputePipeline(_) => RecordKind::CreateComputePipeline,
            Self::Destroy(_) => RecordKind::Destroy,
            Self::Submit(_) => RecordKind::Submit,
            Self::PresentTarget(_) => RecordKind::PresentTarget,
            Self::ReadbackResult(_) => RecordKind::ReadbackResult,
            Self::PipelineOutcome(_) => RecordKind::PipelineOutcome,
            Self::FrameResult(_) => RecordKind::FrameResult,
            Self::Limit(_) => RecordKind::Limit,
        }
    }

    /// What the order checks need to know of the record.
    pub(crate) fn facts(&self) -> Facts {
        let made = |id: Option<ResourceId>| Facts {
            created: id,
            ..Facts::default()
        };
        match self {
            Self::CreateBuffer(c) => made(c.result.made().map(ResourceId::Buffer)),
            Self::CreateTexture(c) => made(c.result.made().map(ResourceId::Texture)),
            Self::CreateView(c) => made(c.result.made().map(ResourceId::View)),
            Self::CreateSampler(c) => made(c.result.made().map(ResourceId::Sampler)),
            Self::CreateGraphicsPipeline(c) => made(c.result.made().map(ResourceId::Pipeline)),
            Self::CreateComputePipeline(c) => made(c.result.made().map(ResourceId::Pipeline)),
            Self::CreateShader(c) => Facts {
                created: c.result.made().map(ResourceId::Shader),
                uses_source: match c.code {
                    ShaderCode::Origin { source, .. } => Some(source),
                    ShaderCode::SpirV(_) | ShaderCode::Msl(_) | ShaderCode::Missing => None,
                },
                ..Facts::default()
            },
            Self::ShaderSource(s) => Facts {
                source: Some(s.index),
                ..Facts::default()
            },
            Self::Destroy(d) => Facts {
                destroyed: Some(d.id),
                ..Facts::default()
            },
            Self::Submit(s) => Facts {
                fence: s.result.made(),
                ..Facts::default()
            },
            Self::End
            | Self::Start(_)
            | Self::PresentTarget(_)
            | Self::ReadbackResult(_)
            | Self::PipelineOutcome(_)
            | Self::FrameResult(_)
            | Self::Limit(_) => Facts::default(),
        }
    }
}

/// What the order checks ([`super::order`]) need to know of one record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Facts {
    /// The id a `Create*` returned.
    pub(crate) created: Option<ResourceId>,
    /// The id a `Destroy` names.
    pub(crate) destroyed: Option<ResourceId>,
    /// The fence an accepted `Submit` signalled.
    pub(crate) fence: Option<FenceValue>,
    /// The index a `ShaderSource` gives.
    pub(crate) source: Option<u32>,
    /// The source index a `CreateShader` names.
    pub(crate) uses_source: Option<u32>,
}

/// Checks the fields with their own bounds, in field order (module docs, "Bounds"). A
/// `Submit`'s list is checked by [`check_list`].
pub(crate) fn check_fields(record: &Record, position: u64) -> Result<(), CaptureError> {
    let text = |what, text: &str| bound(position, what, text.len(), MAX_TEXT_BYTES);
    match record {
        Record::Start(s) => {
            text("build id", &s.build)?;
            text("backend name", &s.backend)
        }
        Record::CreateBuffer(c) => check_result(&c.result, position),
        Record::CreateTexture(c) => check_result(&c.result, position),
        Record::CreateView(c) => check_result(&c.result, position),
        Record::CreateSampler(c) => check_result(&c.result, position),
        Record::CreateGraphicsPipeline(c) => check_result(&c.result, position),
        Record::CreateComputePipeline(c) => check_result(&c.result, position),
        Record::ShaderSource(s) => bound(
            position,
            "shader source",
            s.bytes.len(),
            MAX_SHADER_SOURCE_BYTES,
        ),
        Record::CreateShader(c) => {
            text("entry name", &c.entry)?;
            match &c.code {
                ShaderCode::SpirV(words) => bound(
                    position,
                    "SPIR-V code",
                    words.len().saturating_mul(size_of::<u32>()),
                    MAX_SHADER_CODE_BYTES,
                )?,
                ShaderCode::Msl(source) => {
                    bound(position, "MSL code", source.len(), MAX_SHADER_CODE_BYTES)?;
                }
                ShaderCode::Origin { .. } | ShaderCode::Missing => {}
            }
            check_result(&c.result, position)
        }
        Record::Submit(s) => check_result(&s.result, position),
        Record::PresentTarget(t) => check_result(&t.result, position),
        Record::End
        | Record::Destroy(_)
        | Record::ReadbackResult(_)
        | Record::PipelineOutcome(_)
        | Record::FrameResult(_)
        | Record::Limit(_) => Ok(()),
    }
}

/// Checks a refusal's message against [`MAX_TEXT_BYTES`].
pub(crate) fn check_result<T>(result: &Created<T>, position: u64) -> Result<(), CaptureError> {
    match result {
        Created::Made(_) => Ok(()),
        Created::Refused(message) => bound(
            position,
            "refusal message",
            message.len(),
            MAX_TEXT_BYTES,
        ),
    }
}

/// [`CaptureError::TooLarge`] when `len` is past `max`.
fn bound(position: u64, what: &'static str, len: usize, max: u64) -> Result<(), CaptureError> {
    let len = u64::try_from(len).unwrap_or(u64::MAX);
    if len > max {
        return Err(CaptureError::TooLarge {
            record: position,
            what,
            len,
            max,
        });
    }
    Ok(())
}

/// Running checks on a list's commands, shared by the writer and the reader: the command
/// count first, then each command's nested lists and the list's decoded charge.
pub(crate) struct ListCheck {
    position: u64,
    limits: CaptureLimits,
    command: u64,
    charged: u64,
}

impl ListCheck {
    /// Checks the command count, before any command is looked at.
    pub(crate) fn new(
        position: u64,
        limits: CaptureLimits,
        count: u64,
    ) -> Result<Self, CaptureError> {
        if count > limits.list_commands {
            return Err(CaptureError::ListTooLong {
                record: position,
                count,
                max: limits.list_commands,
            });
        }
        Ok(Self {
            position,
            limits,
            command: 0,
            charged: 0,
        })
    }

    /// Checks one more command: its nested lists, then the list's charge with it.
    pub(crate) fn command(&mut self, command: &Command) -> Result<(), CaptureError> {
        check_nested(command).map_err(|over| CaptureError::TooLarge {
            record: self.position,
            what: over.what,
            len: over.len,
            max: over.max,
        })?;
        self.charged = self.charged.saturating_add(decoded_bytes(command));
        if self.charged > self.limits.list_decoded_bytes {
            return Err(CaptureError::ListTooBig {
                record: self.position,
                command: self.command,
                max: self.limits.list_decoded_bytes,
            });
        }
        self.command = self.command.saturating_add(1);
        Ok(())
    }
}

/// Checks a whole list (what the writer does before it encodes one).
pub(crate) fn check_list(
    list: &CommandList,
    position: u64,
    limits: CaptureLimits,
) -> Result<(), CaptureError> {
    let count = u64::try_from(list.commands.len()).unwrap_or(u64::MAX);
    let mut check = ListCheck::new(position, limits, count)?;
    list.commands.iter().try_for_each(|c| check.command(c))
}

/// Appends the body of `record` to `out`. A `Submit`'s list is encoded as [`CommandList`] is,
/// its command count then each command, and its result follows.
pub(crate) fn encode_body(record: &Record, out: &mut Vec<u8>) -> Result<(), postcard::Error> {
    let body = std::mem::take(out);
    *out = match record {
        Record::End => body,
        Record::Start(b) => postcard::to_extend(b, body)?,
        Record::CreateBuffer(b) => postcard::to_extend(b, body)?,
        Record::CreateTexture(b) => postcard::to_extend(b, body)?,
        Record::CreateView(b) => postcard::to_extend(b, body)?,
        Record::CreateSampler(b) => postcard::to_extend(b, body)?,
        Record::ShaderSource(b) => postcard::to_extend(b, body)?,
        Record::CreateShader(b) => postcard::to_extend(b, body)?,
        Record::CreateGraphicsPipeline(b) => postcard::to_extend(b, body)?,
        Record::CreateComputePipeline(b) => postcard::to_extend(b, body)?,
        Record::Destroy(b) => postcard::to_extend(b, body)?,
        Record::Submit(b) => postcard::to_extend(&b.result, postcard::to_extend(&b.list, body)?)?,
        Record::PresentTarget(b) => postcard::to_extend(b, body)?,
        Record::ReadbackResult(b) => postcard::to_extend(b, body)?,
        Record::PipelineOutcome(b) => postcard::to_extend(b, body)?,
        Record::FrameResult(b) => postcard::to_extend(b, body)?,
        Record::Limit(b) => postcard::to_extend(b, body)?,
    };
    Ok(())
}

/// Decodes a body of `kind`, which must take exactly its length; then checks the fields with
/// their own bounds ([`check_fields`]).
pub(crate) fn decode_body(
    kind: RecordKind,
    body: &[u8],
    position: u64,
    limits: CaptureLimits,
) -> Result<Record, CaptureError> {
    let mut cursor = Cursor { body, position };
    let record = match kind {
        RecordKind::End => Record::End,
        RecordKind::Start => Record::Start(cursor.take()?),
        RecordKind::CreateBuffer => Record::CreateBuffer(cursor.take()?),
        RecordKind::CreateTexture => Record::CreateTexture(cursor.take()?),
        RecordKind::CreateView => Record::CreateView(cursor.take()?),
        RecordKind::CreateSampler => Record::CreateSampler(cursor.take()?),
        RecordKind::ShaderSource => Record::ShaderSource(cursor.take()?),
        RecordKind::CreateShader => Record::CreateShader(cursor.take()?),
        RecordKind::CreateGraphicsPipeline => Record::CreateGraphicsPipeline(cursor.take()?),
        RecordKind::CreateComputePipeline => Record::CreateComputePipeline(cursor.take()?),
        RecordKind::Destroy => Record::Destroy(cursor.take()?),
        RecordKind::Submit => Record::Submit(cursor.submit(limits)?),
        RecordKind::PresentTarget => Record::PresentTarget(cursor.take()?),
        RecordKind::ReadbackResult => Record::ReadbackResult(cursor.take()?),
        RecordKind::PipelineOutcome => Record::PipelineOutcome(cursor.take()?),
        RecordKind::FrameResult => Record::FrameResult(cursor.take()?),
        RecordKind::Limit => Record::Limit(cursor.take()?),
    };
    if !cursor.body.is_empty() {
        return Err(CaptureError::RecordTrailingBytes {
            record: position,
            count: u64::try_from(cursor.body.len()).unwrap_or(u64::MAX),
        });
    }
    check_fields(&record, position)?;
    Ok(record)
}

/// The unread part of one record's body.
struct Cursor<'a> {
    body: &'a [u8],
    position: u64,
}

impl<'a> Cursor<'a> {
    /// Reads one value from the front of the body.
    fn take<T: Deserialize<'a>>(&mut self) -> Result<T, CaptureError> {
        let (value, rest) =
            postcard::take_from_bytes(self.body).map_err(|source| CaptureError::Decode {
                record: self.position,
                source,
            })?;
        self.body = rest;
        Ok(value)
    }

    /// Reads a `Submit` body: the command count, checked before any command is decoded; each
    /// command, checked as it is decoded; then the result.
    fn submit(&mut self, limits: CaptureLimits) -> Result<Submit, CaptureError> {
        // A sequence's length is a `usize` varint in postcard, the same bytes as a `u64` one on
        // the 64-bit hosts starquake runs on.
        let count: u64 = self.take()?;
        let mut check = ListCheck::new(self.position, limits, count)?;
        // Room for no more commands than the bytes left could hold or the charge allows,
        // whatever the count says.
        let affordable = usize::try_from(limits.list_decoded_bytes)
            .unwrap_or(usize::MAX)
            .checked_div(size_of::<Command>())
            .unwrap_or(0);
        let room = usize::try_from(count)
            .unwrap_or(usize::MAX)
            .min(self.body.len())
            .min(affordable);
        let mut commands = Vec::with_capacity(room);
        for _ in 0..count {
            let command: Command = self.take()?;
            check.command(&command)?;
            commands.push(command);
        }
        let result = self.take()?;
        Ok(Submit {
            list: CommandList { commands },
            result,
        })
    }
}
