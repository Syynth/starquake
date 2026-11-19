//! The writer: records to bytes, through the same checks the reader applies, handed to a
//! [`CaptureSink`] (module docs of [`crate::capture`]).

use super::order::Order;
use super::record::ShaderCode;
use super::record::{
    Created, Facts, Limit, Record, RecordKind, SourceRecord, Start, check_fields, check_list,
    check_result, encode_body,
};
use super::{CaptureError, CaptureLimits, MAGIC, VERSION};
use crate::command::CommandList;
use crate::id::{FenceValue, ResourceId};
use crate::shader::{ShaderDesc, ShaderSource};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, PoisonError};

/// Bytes of a record's tag.
const TAG_BYTES: usize = 1;

/// The most bytes a `u32` takes as a varint: seven bits a byte, so five.
pub(crate) const VARINT_U32_MAX: usize = 5;

/// Receives a capture's bytes, in order, as the writer produces them: the header and each
/// record, a record as one or two chunks (its tag and length, then its body). The file is the
/// chunks one after another.
///
/// The app installs one (the [`crate::FrameSink`] pattern). It is called with the writer's lock
/// held, on the thread whose backend call made the record, so it must not call the
/// [`super::CapturingBackend`] it is installed on. The spec's app sink hands the chunks to a
/// thread of its own through a bounded queue and blocks while the queue is full: a capture never
/// drops a record.
pub trait CaptureSink: Debug + Send + Sync {
    /// Takes the next chunk of the capture.
    fn write(&self, chunk: Vec<u8>);
}

/// A sink that keeps the capture in memory: for tests, and for captures a caller replays at
/// once.
#[derive(Debug, Default)]
pub struct CaptureBuffer {
    bytes: Mutex<Vec<u8>>,
}

impl CaptureBuffer {
    /// An empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// A copy of the bytes so far.
    pub fn bytes(&self) -> Vec<u8> {
        self.bytes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl CaptureSink for CaptureBuffer {
    fn write(&self, chunk: Vec<u8>) {
        self.bytes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend_from_slice(&chunk);
    }
}

/// A list's commands, encoded and checked, waiting for the result of the `submit` that consumes
/// the list ([`CaptureWriter::encode_list`], [`CaptureWriter::write_submit`]).
#[derive(Debug)]
pub struct EncodedList {
    body: Vec<u8>,
}

/// Writes a capture: the header and `Start` when it is made, then one record per call, each
/// checked as the reader would check it (module docs, "What reading checks, in order").
///
/// For each record, in this order: the record's fields with their own bounds (for a list, its
/// command count, nested lists and decoded charge), its length against its kind's cap, the
/// order checks, and last the size limit, which keeps room for a closing `Limit` and `End`.
/// A record any check refuses is not written and changes nothing, and the next one may still
/// be written; a caller that wants the capture to end there calls
/// [`CaptureWriter::close_at_limit`].
#[derive(Debug)]
pub struct CaptureWriter {
    sink: Arc<dyn CaptureSink>,
    limits: CaptureLimits,
    size_limit: u64,
    written: u64,
    order: Order,
    /// Each source given so far, by the SHA-256 of its bytes: a lookup by content, so equal
    /// bytes get the same index whatever holds them.
    sources: HashMap<[u8; 32], u32>,
    /// The room every record but `Limit` and `End` keeps for them ([`close_reserve`]), worked
    /// out once when the writer is made.
    close_reserve: u64,
    finished: bool,
}

impl CaptureWriter {
    /// A writer with the shipped bounds ([`CaptureLimits::SHIPPED`]); see
    /// [`CaptureWriter::with_limits`].
    pub fn new(
        sink: Arc<dyn CaptureSink>,
        start: Start,
        size_limit: u64,
    ) -> Result<Self, CaptureError> {
        Self::with_limits(sink, start, size_limit, CaptureLimits::SHIPPED)
    }

    /// A writer that has written the header and `start` to `sink`, and keeps the whole capture
    /// within `size_limit` bytes.
    ///
    /// Errors: `start`'s own checks ([`CaptureError::TooLarge`] for a long text); and
    /// [`CaptureError::LimitTooSmall`] when the header, `start` and the closing `Limit` and `End`
    /// do not fit in `size_limit`. Nothing is written on an error.
    pub fn with_limits(
        sink: Arc<dyn CaptureSink>,
        start: Start,
        size_limit: u64,
        limits: CaptureLimits,
    ) -> Result<Self, CaptureError> {
        let header = postcard::to_extend(&VERSION, MAGIC.to_vec())
            .map_err(|source| CaptureError::Encode { record: 0, source })?;
        let mut writer = Self {
            sink,
            limits,
            // Set below, once the start is known to fit (`LimitTooSmall`, not `OverLimit`).
            size_limit: u64::MAX,
            written: 0,
            order: Order::default(),
            sources: HashMap::new(),
            close_reserve: close_reserve(),
            finished: false,
        };
        let start = Record::Start(start);
        check_fields(&start, 0)?;
        let body = writer.encode(&start)?;
        let (prefix, body) = writer.checked(RecordKind::Start, body, Facts::default())?;
        let needed = len64(header.len())
            .saturating_add(len64(prefix.len()))
            .saturating_add(len64(body.len()))
            .saturating_add(writer.close_reserve);
        if needed > size_limit {
            return Err(CaptureError::LimitTooSmall {
                limit: size_limit,
                needed,
            });
        }
        writer.size_limit = size_limit;
        writer.written = len64(header.len());
        writer.sink.write(header);
        writer.emit(RecordKind::Start, Facts::default(), prefix, body);
        Ok(writer)
    }

    /// Writes one record (the type's docs give the checks). A `Submit` goes through
    /// [`CaptureWriter::encode_list`] and [`CaptureWriter::write_submit`]; an `End` ends the
    /// capture.
    ///
    /// Errors: as the checks say; [`CaptureError::Finished`] after `End`.
    pub fn write(&mut self, record: &Record) -> Result<(), CaptureError> {
        if self.finished {
            return Err(CaptureError::Finished);
        }
        if let Record::Submit(submit) = record {
            let list = self.encode_list(&submit.list)?;
            return self.write_submit(list, &submit.result);
        }
        check_fields(record, self.order.next())?;
        let body = self.encode(record)?;
        let facts = record.facts();
        let (prefix, body) = self.checked(record.kind(), body, facts)?;
        self.emit(record.kind(), facts, prefix, body);
        Ok(())
    }

    /// Checks and encodes a list's commands for the next record, before `submit` consumes the
    /// list: its command count, each command's nested lists and the list's decoded charge.
    pub fn encode_list(&self, list: &CommandList) -> Result<EncodedList, CaptureError> {
        let record = self.order.next();
        check_list(list, record, self.limits)?;
        let body = postcard::to_extend(list, Vec::new())
            .map_err(|source| CaptureError::Encode { record, source })?;
        Ok(EncodedList { body })
    }

    /// Writes a `Submit` record: `list` from [`CaptureWriter::encode_list`], then `result`.
    pub fn write_submit(
        &mut self,
        list: EncodedList,
        result: &Created<FenceValue>,
    ) -> Result<(), CaptureError> {
        if self.finished {
            return Err(CaptureError::Finished);
        }
        let record = self.order.next();
        check_result(result, record)?;
        let body = postcard::to_extend(result, list.body)
            .map_err(|source| CaptureError::Encode { record, source })?;
        let facts = Facts {
            fence: result.made(),
            ..Facts::default()
        };
        let (prefix, body) = self.checked(RecordKind::Submit, body, facts)?;
        self.emit(RecordKind::Submit, facts, prefix, body);
        Ok(())
    }

    /// The index of the shader source holding `bytes`: the one already given for equal bytes,
    /// or a new one, written as a `ShaderSource` record.
    pub fn shader_source(&mut self, bytes: &[u8]) -> Result<u32, CaptureError> {
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        if let Some(&index) = self.sources.get(&digest) {
            return Ok(index);
        }
        let index = u32::try_from(self.sources.len()).unwrap_or(u32::MAX);
        self.write(&Record::ShaderSource(SourceRecord {
            index,
            bytes: bytes.to_vec(),
        }))?;
        self.sources.insert(digest, index);
        Ok(index)
    }

    /// How a `CreateShader` record holds `desc`'s code (decision 15): its origin, written as a
    /// `ShaderSource` first when it is new; SPIR-V or MSL as it is; or missing, for a naga
    /// module with no origin. The one place the wrapper and the builder decide it.
    pub fn shader_code(&mut self, desc: &ShaderDesc) -> Result<ShaderCode, CaptureError> {
        Ok(match (&desc.origin, &desc.source) {
            (Some(origin), _) => ShaderCode::Origin {
                source: self.shader_source(&origin.record)?,
                stage_index: origin.stage_index,
            },
            (None, ShaderSource::SpirV(words)) => ShaderCode::SpirV(words.clone()),
            (None, ShaderSource::Msl(text)) => ShaderCode::Msl(text.clone()),
            (None, ShaderSource::Naga { .. }) => ShaderCode::Missing,
        })
    }

    /// Ends the capture early: a `Limit` naming the last list's fence value, then `End`. Room
    /// for both is kept by every other record, so this fails only after the capture has ended
    /// ([`CaptureError::Finished`]).
    pub fn close_at_limit(&mut self) -> Result<(), CaptureError> {
        let fence = self.order.last_fence();
        self.write(&Record::Limit(Limit { fence }))?;
        self.finish()
    }

    /// Ends the capture with `End`. Errors: [`CaptureError::Finished`] when it has ended.
    pub fn finish(&mut self) -> Result<(), CaptureError> {
        self.write(&Record::End)
    }

    /// Whether `End` has been written.
    pub const fn is_finished(&self) -> bool {
        self.finished
    }

    /// The bytes written so far, header included.
    pub const fn written(&self) -> u64 {
        self.written
    }

    /// Whether `id` was created in the capture and not destroyed.
    pub fn is_live(&self, id: ResourceId) -> bool {
        self.order.is_live(id)
    }

    /// The body of `record`.
    fn encode(&self, record: &Record) -> Result<Vec<u8>, CaptureError> {
        let mut body = Vec::new();
        encode_body(record, &mut body).map_err(|source| CaptureError::Encode {
            record: self.order.next(),
            source,
        })?;
        Ok(body)
    }

    /// Runs the checks after the fields' own: the length against the kind's cap, the order
    /// checks, then the size limit. Returns the record's tag-and-length prefix and its body.
    fn checked(
        &self,
        kind: RecordKind,
        body: Vec<u8>,
        facts: Facts,
    ) -> Result<(Vec<u8>, Vec<u8>), CaptureError> {
        let record = self.order.next();
        let len = len64(body.len());
        let max = self.limits.max_body(kind);
        let short_len =
            u32::try_from(len)
                .ok()
                .filter(|_| len <= max)
                .ok_or(CaptureError::RecordTooLarge {
                    record,
                    kind,
                    len,
                    max,
                })?;
        self.order.check(kind, facts, self.limits)?;
        let prefix =
            prefix(kind, short_len).map_err(|source| CaptureError::Encode { record, source })?;
        // `Limit` and `End` may use the room the others keep for them.
        let reserve = match kind {
            RecordKind::Limit | RecordKind::End => 0,
            _ => self.close_reserve,
        };
        let needed = self
            .written
            .saturating_add(len64(prefix.len()))
            .saturating_add(len)
            .saturating_add(reserve);
        if needed > self.size_limit {
            return Err(CaptureError::OverLimit {
                record,
                needed,
                limit: self.size_limit,
            });
        }
        Ok((prefix, body))
    }

    /// Hands a checked record to the sink and records it. Cannot fail.
    fn emit(&mut self, kind: RecordKind, facts: Facts, prefix: Vec<u8>, body: Vec<u8>) {
        self.order.apply(kind, facts);
        self.written = self
            .written
            .saturating_add(len64(prefix.len()))
            .saturating_add(len64(body.len()));
        self.sink.write(prefix);
        if !body.is_empty() {
            self.sink.write(body);
        }
        if kind == RecordKind::End {
            self.finished = true;
        }
    }
}

/// A record's tag and the varint of its body's length.
pub(crate) fn prefix(kind: RecordKind, len: u32) -> Result<Vec<u8>, postcard::Error> {
    let mut varint = [0; VARINT_U32_MAX];
    let used = postcard::to_slice(&len, &mut varint)?.len();
    let mut prefix = Vec::with_capacity(TAG_BYTES.saturating_add(used));
    prefix.push(kind.tag());
    prefix.extend_from_slice(varint.get(..used).unwrap_or_default());
    Ok(prefix)
}

/// The room every record but `Limit` and `End` keeps for them: the longest `Limit` (the
/// largest fence value) and `End`, each with its tag and length. It depends only on constants;
/// a writer works it out once ([`CaptureWriter::with_limits`]), not per record.
pub(crate) fn close_reserve() -> u64 {
    let framed = |record: &Record| {
        let mut body = Vec::new();
        let body_len = encode_body(record, &mut body).map_or(0, |()| body.len());
        let len_bytes = u32::try_from(body_len)
            .ok()
            .and_then(|len| prefix(record.kind(), len).ok())
            .map_or(TAG_BYTES.saturating_add(VARINT_U32_MAX), |p| p.len());
        len64(len_bytes.saturating_add(body_len))
    };
    let longest_limit = Record::Limit(Limit {
        fence: FenceValue::new(u64::MAX),
    });
    framed(&longest_limit).saturating_add(framed(&Record::End))
}

/// A length as a `u64`.
pub(crate) fn len64(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}
