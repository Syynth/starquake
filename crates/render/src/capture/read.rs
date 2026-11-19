//! The streaming reader: one record at a time from a [`Read`], each length checked before
//! anything is allocated for it (module docs of [`crate::capture`], "What reading checks, in
//! order" and "Bounds").

use super::order::Order;
use super::record::{Record, RecordKind, decode_body};
use super::write::VARINT_U32_MAX;
use super::{CaptureError, CaptureLimits, MAGIC, VERSION};
use std::io::{self, Read};

/// Reads a capture one record at a time.
///
/// The app supplies the input and the file's total length; the reader never reads past that
/// length, and holds one record at a time. A failed read is sticky: every later call returns the
/// same error.
#[derive(Debug)]
pub struct CaptureReader<R> {
    input: R,
    /// Bytes of the file not yet read, by the length the reader was given.
    left: u64,
    limits: CaptureLimits,
    order: Order,
    failed: Option<CaptureError>,
    ended: bool,
}

impl<R: Read> CaptureReader<R> {
    /// A reader with the shipped bounds ([`CaptureLimits::SHIPPED`]); see
    /// [`CaptureReader::with_limits`].
    pub fn new(input: R, len: u64) -> Result<Self, CaptureError> {
        Self::with_limits(input, len, CaptureLimits::SHIPPED)
    }

    /// Reads and checks the header of a capture of `len` bytes: the magic, then the version.
    /// No record is read yet.
    pub fn with_limits(
        mut input: R,
        len: u64,
        limits: CaptureLimits,
    ) -> Result<Self, CaptureError> {
        let mut magic = [0; MAGIC.len()];
        let have = usize::try_from(len).unwrap_or(usize::MAX).min(MAGIC.len());
        let start = magic.get_mut(..have).unwrap_or_default();
        input.read_exact(start).map_err(|error| io_error(&error))?;
        if have < MAGIC.len() {
            // Too short to hold the magic: a capture cut short if it starts as one does.
            return Err(if MAGIC.starts_with(start) {
                CaptureError::Truncated
            } else {
                CaptureError::BadMagic
            });
        }
        if magic != MAGIC {
            return Err(CaptureError::BadMagic);
        }
        let mut reader = Self {
            input,
            left: len.saturating_sub(super::write::len64(MAGIC.len())),
            limits,
            order: Order::default(),
            failed: None,
            ended: false,
        };
        let version = reader.varint().map_err(|error| match error {
            Malformed::Bytes(source) => CaptureError::HeaderDecode(source),
            Malformed::Read(error) => error,
        })?;
        if version != VERSION {
            return Err(CaptureError::UnsupportedVersion { found: version });
        }
        Ok(reader)
    }

    /// The next record: `Ok(Some(..))` for each record up to and including `End`, then
    /// `Ok(None)`. Errors are as the module docs list them, in that order, and sticky.
    pub fn next_record(&mut self) -> Result<Option<Record>, CaptureError> {
        if let Some(error) = &self.failed {
            return Err(error.clone());
        }
        if self.ended {
            return Ok(None);
        }
        match self.read_one() {
            Ok(record) => Ok(Some(record)),
            Err(error) => {
                self.failed = Some(error.clone());
                Err(error)
            }
        }
    }

    /// The bounds the reader applies.
    pub const fn limits(&self) -> CaptureLimits {
        self.limits
    }

    /// The position the next record has; the number of records read so far.
    pub const fn position(&self) -> u64 {
        self.order.next()
    }

    fn read_one(&mut self) -> Result<Record, CaptureError> {
        let record = self.order.next();
        let mut tag = [0];
        self.read(&mut tag)?;
        let [tag] = tag;
        let kind = RecordKind::from_tag(tag).ok_or(CaptureError::UnknownRecord { tag, record })?;
        let len = u64::from(self.varint().map_err(|error| match error {
            Malformed::Bytes(source) => CaptureError::Decode { record, source },
            Malformed::Read(error) => error,
        })?);
        let max = self.limits.max_body(kind);
        if len > max {
            return Err(CaptureError::RecordTooLarge {
                record,
                kind,
                len,
                max,
            });
        }
        if len > self.left {
            return Err(CaptureError::Truncated);
        }
        let mut body = vec![0; usize::try_from(len).map_err(|_| CaptureError::Truncated)?];
        self.read(&mut body)?;
        let decoded = decode_body(kind, &body, record, self.limits)?;
        drop(body);
        let facts = decoded.facts();
        self.order.check(kind, facts, self.limits)?;
        if kind == RecordKind::End && self.left > 0 {
            return Err(CaptureError::TrailingBytes { count: self.left });
        }
        self.order.apply(kind, facts);
        self.ended = kind == RecordKind::End;
        Ok(decoded)
    }

    /// Fills `buf` from the input, within the bytes left.
    fn read(&mut self, buf: &mut [u8]) -> Result<(), CaptureError> {
        let want = super::write::len64(buf.len());
        if want > self.left {
            return Err(CaptureError::Truncated);
        }
        self.input
            .read_exact(buf)
            .map_err(|error| io_error(&error))?;
        self.left -= want;
        Ok(())
    }

    /// A `u32` varint, read a byte at a time.
    fn varint(&mut self) -> Result<u32, Malformed> {
        let mut bytes = [0; VARINT_U32_MAX];
        let mut used = 0;
        for slot in &mut bytes {
            let mut byte = [0];
            self.read(&mut byte).map_err(Malformed::Read)?;
            *slot = byte[0];
            used += 1;
            if byte[0] & CONTINUES == 0 {
                break;
            }
        }
        // Postcard decides what is well formed (a fifth byte with bits past 32, or a fifth
        // byte that still continues, is an error), so a varint reads here as postcard reads it.
        postcard::from_bytes(bytes.get(..used).unwrap_or_default()).map_err(Malformed::Bytes)
    }
}

/// The bit of a varint byte that says another byte follows.
const CONTINUES: u8 = 0x80;

/// Why a varint could not be read.
enum Malformed {
    /// The bytes are not a well-formed varint.
    Bytes(postcard::Error),
    /// The input ended or failed.
    Read(CaptureError),
}

/// A read failure: the input ending early is [`CaptureError::Truncated`].
fn io_error(error: &io::Error) -> CaptureError {
    match error.kind() {
        io::ErrorKind::UnexpectedEof => CaptureError::Truncated,
        kind => CaptureError::Io {
            kind,
            message: error.to_string(),
        },
    }
}
