//! The log packets `ILogger::Log` carries: the header, the message a run of packets builds, and
//! the data chunks a finished message holds.
//!
//! Ported from `eden/src/core/hle/service/lm/lm.cpp` @ cf654052 (GPL-3.0-or-later): the
//! `LogPacketHeader` struct, the `LogPacketFlags`, `LogSeverity` and `LogDataChunkKey` enums,
//! `ILogger::Log` (how packets build a message), `ReadLeb128`, `ReadString`, `ReadAsU32`,
//! `ReadAsU64` and `ParseLog`. Eden-NX2 @ 50fa86bb and Eden PR #4588 @ e5fb3b97 have the same
//! file, apart from `std::format` spelled `fmt::format`: no Switch 2 change. Where the format is
//! checked against the console's own log server, the source is Atmosphère's
//! (`atmosphere/libraries/libstratosphere/source/lm/srv/lm_log_packet_parser.cpp`,
//! `ParsePacket` and `ParseDataChunk`; `lm/impl/lm_log_data_chunk.hpp`), GPL-2.0-only and so
//! read for facts only (`docs/references.md`).
//!
//! # The layout
//!
//! A packet is a [`HEADER_SIZE`]-byte header, then its payload:
//!
//! | Offset | Size | Field |
//! |---|---|---|
//! | 0x00 | 8 | process id |
//! | 0x08 | 8 | thread id |
//! | 0x10 | 1 | flags: [`FLAG_HEAD`], [`FLAG_TAIL`], [`FLAG_LITTLE_ENDIAN`] |
//! | 0x11 | 1 | padding (`INSERT_PADDING_BYTES(1)`), not read |
//! | 0x12 | 1 | severity ([`Severity`]) |
//! | 0x13 | 1 | verbosity |
//! | 0x14 | 4 | payload size |
//!
//! [`Header::parse`] reads the fields in that order; [`HEADER_SIZE`] is Eden's
//! `static_assert(sizeof(LogPacketHeader) == 0x18)`.
//!
//! A message is the payloads of a head packet and the packets after it with the same process
//! id, thread id, severity and verbosity, up to and including a tail packet ([`Pending`]). A
//! finished message is a list of data chunks, each a key and a size, both unsigned LEB128, then
//! that many bytes ([`LogRecord::parse`]). The keys are [`ChunkKey`]'s.
//!
//! # Where this differs from Eden
//!
//! Each difference is pinned by the test named, in `lm/packet/tests.rs`.
//!
//! - **The payload is the header's payload size.** Eden takes everything after the header and
//!   never reads the size. Here the payload is the next payload-size bytes, as the console's
//!   parser takes it (Atmosphère, `ParsePacket`, fact only): fewer is
//!   [`PacketError::Truncated`] at [`Field::Payload`], and bytes after it are ignored
//!   (`the_payload_is_the_size_the_header_gives`, `a_payload_shorter_than_its_size_is_refused`).
//! - **A chunk is skipped by its size whatever its key.** Eden's `switch` advances past the data
//!   of the keys it stores, but for `LogSessionBegin`, `LogSessionEnd` and keys it does not
//!   know it does not move, and reads the chunk's data as the next key. Here those chunks'
//!   data is skipped, as the console's parser skips every chunk by its size (Atmosphère,
//!   `ParseDataChunk`, fact only) (`session_and_unknown_chunks_are_skipped_by_their_size`).
//! - **A chunk that runs past the message is refused.** Eden's `ReadString` cuts a string down
//!   to the bytes left, and `ReadAsU32` and `ReadAsU64` read past the end. Here it is
//!   [`PacketError::Truncated`] at [`Field::ChunkData`] (`a_chunk_past_the_end_is_refused`).
//! - **A number chunk of the wrong size is refused.** Eden's `ReadAsU32` and `ReadAsU64`
//!   `ASSERT` the size, which logs and carries on reading four or eight bytes. Here it is
//!   [`PacketError::ChunkSize`] (`a_number_chunk_of_the_wrong_size_is_refused`).
//! - **A LEB128 number is at most [`MAX_LEB128_BYTES`] bytes, and must end inside the
//!   message.** Eden's `ReadLeb128` stops after eight bytes whether or not the eighth says more
//!   follow, and returns what it has when the message ends; for a chunk size it reads the
//!   byte past the end when a key is the message's last byte. Here a ninth byte is
//!   [`PacketError::NumberTooLong`], and a message that ends inside a number is
//!   [`PacketError::Truncated`] (`a_number_longer_than_eight_bytes_is_refused`,
//!   `a_message_that_ends_inside_a_number_is_refused`). Atmosphère's parser reads up to nine
//!   bytes, so the console may accept a nine-byte number that this port refuses. In the
//!   shortest encoding a number needs nine bytes only when it is at least 2^56, which no key or
//!   size reaches; but LEB128 also allows longer encodings, padded with continuation bytes whose
//!   value is 0, and one of those can be nine bytes for any value, even 0. No client is known to
//!   pad a key or size; a padded one of eight bytes or fewer is read as its value
//!   (`a_padded_number_is_read_up_to_eight_bytes`).
//! - **What a message may hold is bounded.** Eden keeps every message in progress and lets one
//!   grow without limit. Here at most [`MAX_IN_PROGRESS`] messages are in progress on one
//!   logger, and a message is at most [`MAX_MESSAGE_BYTES`]: [`Pending::push`] says which
//!   limit refused a packet (`messages_in_progress_are_capped`,
//!   `a_message_is_capped_at_its_limit`). Both limits are starquake's own guesses, not a
//!   console's: a trace of a title that logs more would show they are too small.
//!
//! What Eden refuses is refused the same way: a header shorter than [`HEADER_SIZE`] (Eden:
//! "Data size is too small for header!") and a packet that is not a head when no message is in
//! progress for its key (Eden: "Log entry does not exist!") ([`PacketError::NoHead`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The size of a packet's header (`static_assert(sizeof(LogPacketHeader) == 0x18)`).
pub const HEADER_SIZE: usize = 0x18;

/// `LogPacketFlags::Head`: the packet begins a message.
pub const FLAG_HEAD: u8 = 1 << 0;
/// `LogPacketFlags::Tail`: the packet ends a message.
pub const FLAG_TAIL: u8 = 1 << 1;
/// `LogPacketFlags::LittleEndian`. Eden never reads it, and neither does this port: every field
/// is read little-endian.
pub const FLAG_LITTLE_ENDIAN: u8 = 1 << 2;

/// The most bytes an unsigned LEB128 number may take: Eden's `ReadLeb128` loop bound
/// (`for (std::size_t i = 0; i < sizeof(u64); i++)`).
pub const MAX_LEB128_BYTES: usize = 8;
/// The bits of a LEB128 byte that hold value (`v & 0x7f` in `ReadLeb128`).
pub const LEB128_VALUE_MASK: u8 = 0x7f;
/// The bit of a LEB128 byte that says another byte follows (`(v & 0x80) == 0` ends the number
/// in `ReadLeb128`).
pub const LEB128_CONTINUE: u8 = 0x80;
/// How far each LEB128 byte's value is shifted past the one before (`shift += 7`).
pub const LEB128_SHIFT: u32 = 7;

/// The size of a [`ChunkKey::LineNumber`] chunk: `ReadAsU32`'s `ASSERT(length == sizeof(u32))`.
pub const U32_CHUNK_SIZE: usize = 4;
/// The size of a [`ChunkKey::PacketDropCount`] or [`ChunkKey::UserSystemClock`] chunk:
/// `ReadAsU64`'s `ASSERT(length == sizeof(u64))`.
pub const U64_CHUNK_SIZE: usize = 8;

/// The most messages one logger keeps in progress (module docs, "Where this differs from
/// Eden"). A guess of starquake's own: Eden has no limit.
pub const MAX_IN_PROGRESS: usize = 16;
/// The most bytes one message may hold, all its packets' payloads together (module docs,
/// "Where this differs from Eden"). A guess of starquake's own: Eden has no limit.
pub const MAX_MESSAGE_BYTES: usize = 0x4000;

/// A field of a packet or a message, for [`PacketError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    /// The header's process id.
    ProcessId,
    /// The header's thread id.
    ThreadId,
    /// The header's flags.
    Flags,
    /// The header's padding byte.
    Padding,
    /// The header's severity.
    Severity,
    /// The header's verbosity.
    Verbosity,
    /// The header's payload size.
    PayloadSize,
    /// The payload after the header.
    Payload,
    /// A chunk's key.
    ChunkKey,
    /// A chunk's size.
    ChunkSize,
    /// A chunk's data.
    ChunkData,
}

impl Field {
    /// The field in plain words.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ProcessId => "process id",
            Self::ThreadId => "thread id",
            Self::Flags => "flags",
            Self::Padding => "padding",
            Self::Severity => "severity",
            Self::Verbosity => "verbosity",
            Self::PayloadSize => "payload size",
            Self::Payload => "payload",
            Self::ChunkKey => "chunk key",
            Self::ChunkSize => "chunk size",
            Self::ChunkData => "chunk data",
        }
    }
}

/// Why a packet or a message was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PacketError {
    /// The bytes end inside a field.
    #[error(
        "the log packet ends inside its {}: {needed} bytes are needed at offset {offset}, and \
         there are {len}",
        .field.name()
    )]
    Truncated {
        /// The field.
        field: Field,
        /// Where the field starts, or for a LEB128 number the byte that was missing.
        offset: usize,
        /// The bytes the field needs from `offset`.
        needed: u64,
        /// The bytes there are, all told.
        len: usize,
    },
    /// A LEB128 number's last allowed byte says another follows.
    #[error(
        "the log packet's {} at offset {offset} is longer than {MAX_LEB128_BYTES} bytes",
        .field.name()
    )]
    NumberTooLong {
        /// The field.
        field: Field,
        /// Where the number starts.
        offset: usize,
    },
    /// A number chunk's size is not its number's.
    #[error("a {} chunk of {size} bytes, where it takes {expected}", .key.name())]
    ChunkSize {
        /// The chunk's key.
        key: ChunkKey,
        /// The size the chunk gives.
        size: u64,
        /// The size its number takes.
        expected: usize,
    },
    /// A packet that is not a head, with no message in progress for its key.
    #[error("a log packet continues a message that nothing began")]
    NoHead,
    /// A message would grow past [`MAX_MESSAGE_BYTES`].
    #[error("a log message of {len} bytes is more than the {MAX_MESSAGE_BYTES} kept")]
    MessageTooLarge {
        /// The size it would have.
        len: usize,
    },
    /// A head packet would begin a message when [`MAX_IN_PROGRESS`] are already in progress.
    #[error("a log message begins while {MAX_IN_PROGRESS} are already in progress")]
    TooManyInProgress,
}

/// `LogSeverity`: how serious a message is. Eden names the five values below and prints
/// "UNKNOWN" for any other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// `Trace = 0`.
    Trace,
    /// `Info = 1`.
    Info,
    /// `Warning = 2`.
    Warning,
    /// `Error = 3`.
    Error,
    /// `Fatal = 4`.
    Fatal,
    /// Any other byte.
    Other(u8),
}

impl Severity {
    /// The severity a header's byte stands for.
    #[must_use]
    pub const fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Trace,
            1 => Self::Info,
            2 => Self::Warning,
            3 => Self::Error,
            4 => Self::Fatal,
            other => Self::Other(other),
        }
    }

    /// The byte a header carries for it.
    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::Trace => 0,
            Self::Info => 1,
            Self::Warning => 2,
            Self::Error => 3,
            Self::Fatal => 4,
            Self::Other(other) => other,
        }
    }

    /// Eden's name for it (`NameOf`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Error => "ERROR",
            Self::Fatal => "FATAL",
            Self::Other(_) => "UNKNOWN",
        }
    }
}

/// A packet's header (module docs, "The layout"). The padding byte is not kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// The process id the client wrote.
    pub process_id: u64,
    /// The thread id the client wrote.
    pub thread_id: u64,
    /// [`FLAG_HEAD`], [`FLAG_TAIL`], [`FLAG_LITTLE_ENDIAN`]; other bits are kept and ignored.
    pub flags: u8,
    /// The severity byte ([`Severity::from_raw`]).
    pub severity: u8,
    /// The verbosity byte.
    pub verbosity: u8,
    /// How many payload bytes follow the header.
    pub payload_size: u32,
}

impl Header {
    /// Reads a packet: its header, then the payload the header's size gives. Bytes after the
    /// payload are ignored (module docs, "Where this differs from Eden").
    pub fn parse(packet: &[u8]) -> Result<(Self, &[u8]), PacketError> {
        let mut cursor = Cursor::new(packet);
        let process_id = u64::from_le_bytes(cursor.array(Field::ProcessId)?);
        let thread_id = u64::from_le_bytes(cursor.array(Field::ThreadId)?);
        let [flags] = cursor.array(Field::Flags)?;
        let [_padding] = cursor.array(Field::Padding)?;
        let [severity] = cursor.array(Field::Severity)?;
        let [verbosity] = cursor.array(Field::Verbosity)?;
        let payload_size = u32::from_le_bytes(cursor.array(Field::PayloadSize)?);
        let payload = cursor.take(u64::from(payload_size), Field::Payload)?;
        let header = Self {
            process_id,
            thread_id,
            flags,
            severity,
            verbosity,
            payload_size,
        };
        Ok((header, payload))
    }

    /// Whether the packet begins a message.
    #[must_use]
    pub const fn is_head(&self) -> bool {
        self.flags & FLAG_HEAD != 0
    }

    /// Whether the packet ends a message.
    #[must_use]
    pub const fn is_tail(&self) -> bool {
        self.flags & FLAG_TAIL != 0
    }

    /// The key the packet's message is kept under (Eden's `LogPacketHeaderEntry`).
    #[must_use]
    pub const fn key(&self) -> MessageKey {
        MessageKey {
            process_id: self.process_id,
            thread_id: self.thread_id,
            severity: self.severity,
            verbosity: self.verbosity,
        }
    }
}

/// What a message in progress is kept under: Eden's `LogPacketHeaderEntry`, the header without
/// its flags and payload size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MessageKey {
    /// The header's process id.
    pub process_id: u64,
    /// The header's thread id.
    pub thread_id: u64,
    /// The header's severity byte.
    pub severity: u8,
    /// The header's verbosity byte.
    pub verbosity: u8,
}

/// The messages in progress on one logger (Eden's `ILogger::entries`), within
/// [`MAX_IN_PROGRESS`] messages of at most [`MAX_MESSAGE_BYTES`] each. A snapshot is read back
/// through the same limits, so a saved logger that holds more is refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "PendingState")]
pub struct Pending {
    messages: BTreeMap<MessageKey, Vec<u8>>,
}

/// [`Pending`] as a snapshot holds it, before its limits are checked.
#[derive(Deserialize)]
struct PendingState {
    messages: BTreeMap<MessageKey, Vec<u8>>,
}

impl TryFrom<PendingState> for Pending {
    type Error = PacketError;

    fn try_from(state: PendingState) -> Result<Self, PacketError> {
        if state.messages.len() > MAX_IN_PROGRESS {
            return Err(PacketError::TooManyInProgress);
        }
        if let Some(message) = state
            .messages
            .values()
            .find(|message| message.len() > MAX_MESSAGE_BYTES)
        {
            return Err(PacketError::MessageTooLarge { len: message.len() });
        }
        Ok(Self {
            messages: state.messages,
        })
    }
}

impl Default for Pending {
    fn default() -> Self {
        Self::new()
    }
}

impl Pending {
    /// No message in progress.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            messages: BTreeMap::new(),
        }
    }

    /// How many messages are in progress.
    #[must_use]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether no message is in progress.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// The bytes of the message in progress under `key`, if there is one.
    #[must_use]
    pub fn message(&self, key: &MessageKey) -> Option<&[u8]> {
        self.messages.get(key).map(Vec::as_slice)
    }

    /// Adds a packet's payload to its message, as Eden's `ILogger::Log` does, and returns the
    /// message's bytes when the packet is a tail.
    ///
    /// - A head packet begins its key's message afresh, dropping one in progress (Eden's
    ///   `insert_or_assign`). A head that is also a tail is a whole message, returned at once.
    /// - Any other packet adds to its key's message; with none in progress it is
    ///   [`PacketError::NoHead`].
    ///
    /// The limits (module docs, "Where this differs from Eden"): a head that would begin a new
    /// key's message while [`MAX_IN_PROGRESS`] are in progress is [`PacketError::TooManyInProgress`]
    /// and changes nothing; a packet that would make its message more than
    /// [`MAX_MESSAGE_BYTES`] is [`PacketError::MessageTooLarge`], and its key's message is
    /// dropped, the bytes kept so far being only part of it.
    pub fn push(
        &mut self,
        header: &Header,
        payload: &[u8],
    ) -> Result<Option<Vec<u8>>, PacketError> {
        let key = header.key();
        if header.is_head() {
            if payload.len() > MAX_MESSAGE_BYTES {
                self.messages.remove(&key);
                return Err(PacketError::MessageTooLarge { len: payload.len() });
            }
            if header.is_tail() {
                self.messages.remove(&key);
                return Ok(Some(payload.to_vec()));
            }
            if !self.messages.contains_key(&key) && self.messages.len() >= MAX_IN_PROGRESS {
                return Err(PacketError::TooManyInProgress);
            }
            self.messages.insert(key, payload.to_vec());
            return Ok(None);
        }
        let message = self.messages.get_mut(&key).ok_or(PacketError::NoHead)?;
        let len = message.len().saturating_add(payload.len());
        if len > MAX_MESSAGE_BYTES {
            self.messages.remove(&key);
            return Err(PacketError::MessageTooLarge { len });
        }
        message.extend_from_slice(payload);
        if header.is_tail() {
            return Ok(self.messages.remove(&key));
        }
        Ok(None)
    }
}

/// `LogDataChunkKey`: what a chunk holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKey {
    /// `LogSessionBegin = 0`. Skipped.
    SessionBegin,
    /// `LogSessionEnd = 1`. Skipped.
    SessionEnd,
    /// `TextLog = 2`: the message's text.
    TextLog,
    /// `LineNumber = 3`: a `u32`.
    LineNumber,
    /// `FileName = 4`.
    FileName,
    /// `FunctionName = 5`.
    FunctionName,
    /// `ModuleName = 6`.
    ModuleName,
    /// `ThreadName = 7`.
    ThreadName,
    /// `LogPacketDropCount = 8`: a `u64`.
    PacketDropCount,
    /// `UserSystemClock = 9`: an `s64`.
    UserSystemClock,
    /// `ProcessName = 10`.
    ProcessName,
    /// A key Eden does not name. Skipped.
    Other(u64),
}

impl ChunkKey {
    /// The key a chunk's number stands for.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        match raw {
            0 => Self::SessionBegin,
            1 => Self::SessionEnd,
            2 => Self::TextLog,
            3 => Self::LineNumber,
            4 => Self::FileName,
            5 => Self::FunctionName,
            6 => Self::ModuleName,
            7 => Self::ThreadName,
            8 => Self::PacketDropCount,
            9 => Self::UserSystemClock,
            10 => Self::ProcessName,
            other => Self::Other(other),
        }
    }

    /// The key's number.
    #[must_use]
    pub const fn raw(self) -> u64 {
        match self {
            Self::SessionBegin => 0,
            Self::SessionEnd => 1,
            Self::TextLog => 2,
            Self::LineNumber => 3,
            Self::FileName => 4,
            Self::FunctionName => 5,
            Self::ModuleName => 6,
            Self::ThreadName => 7,
            Self::PacketDropCount => 8,
            Self::UserSystemClock => 9,
            Self::ProcessName => 10,
            Self::Other(other) => other,
        }
    }

    /// The key in plain words.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SessionBegin => "session begin",
            Self::SessionEnd => "session end",
            Self::TextLog => "text",
            Self::LineNumber => "line number",
            Self::FileName => "file name",
            Self::FunctionName => "function name",
            Self::ModuleName => "module name",
            Self::ThreadName => "thread name",
            Self::PacketDropCount => "packet drop count",
            Self::UserSystemClock => "user system clock",
            Self::ProcessName => "process name",
            Self::Other(_) => "unknown",
        }
    }
}

/// What a finished message holds: Eden's `ParseLog` locals. A string is the chunk's bytes, which
/// need not be UTF-8. A later chunk of a key replaces an earlier one, and a string chunk of no
/// bytes leaves its field `None` (Eden's `ReadString` returns `std::nullopt` for a length of
/// zero, which the assignment stores).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogRecord<'a> {
    /// [`ChunkKey::TextLog`].
    pub text: Option<&'a [u8]>,
    /// [`ChunkKey::LineNumber`].
    pub line_number: Option<u32>,
    /// [`ChunkKey::FileName`].
    pub file_name: Option<&'a [u8]>,
    /// [`ChunkKey::FunctionName`].
    pub function_name: Option<&'a [u8]>,
    /// [`ChunkKey::ModuleName`].
    pub module_name: Option<&'a [u8]>,
    /// [`ChunkKey::ThreadName`].
    pub thread_name: Option<&'a [u8]>,
    /// [`ChunkKey::PacketDropCount`].
    pub packet_drop_count: Option<u64>,
    /// [`ChunkKey::UserSystemClock`].
    pub user_system_clock: Option<i64>,
    /// [`ChunkKey::ProcessName`].
    pub process_name: Option<&'a [u8]>,
}

impl<'a> LogRecord<'a> {
    /// Reads a finished message's chunks (module docs, "The layout"). The loop ends with the
    /// message: each chunk takes at least its key's byte and its size's.
    pub fn parse(message: &'a [u8]) -> Result<Self, PacketError> {
        let mut record = Self::default();
        let mut cursor = Cursor::new(message);
        while cursor.remaining() > 0 {
            let key = ChunkKey::from_raw(cursor.leb128(Field::ChunkKey)?);
            let size = cursor.leb128(Field::ChunkSize)?;
            let data = cursor.take(size, Field::ChunkData)?;
            let string = || (!data.is_empty()).then_some(data);
            match key {
                ChunkKey::SessionBegin | ChunkKey::SessionEnd | ChunkKey::Other(_) => {}
                ChunkKey::TextLog => record.text = string(),
                ChunkKey::LineNumber => {
                    record.line_number = Some(u32::from_le_bytes(number(key, data)?));
                }
                ChunkKey::FileName => record.file_name = string(),
                ChunkKey::FunctionName => record.function_name = string(),
                ChunkKey::ModuleName => record.module_name = string(),
                ChunkKey::ThreadName => record.thread_name = string(),
                ChunkKey::PacketDropCount => {
                    record.packet_drop_count = Some(u64::from_le_bytes(number(key, data)?));
                }
                ChunkKey::UserSystemClock => {
                    record.user_system_clock = Some(i64::from_le_bytes(number(key, data)?));
                }
                ChunkKey::ProcessName => record.process_name = string(),
            }
        }
        Ok(record)
    }
}

/// A number chunk's bytes, which must be exactly the number's size.
fn number<const N: usize>(key: ChunkKey, data: &[u8]) -> Result<[u8; N], PacketError> {
    data.try_into().map_err(|_| PacketError::ChunkSize {
        key,
        size: u64::try_from(data.len()).unwrap_or(u64::MAX),
        expected: N,
    })
}

/// A bounds-checked reader over a packet or a message that names the field a read was for.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    const fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn truncated(&self, field: Field, needed: u64) -> PacketError {
        PacketError::Truncated {
            field,
            offset: self.pos,
            needed,
            len: self.data.len(),
        }
    }

    /// The next `len` bytes.
    fn take(&mut self, len: u64, field: Field) -> Result<&'a [u8], PacketError> {
        let rest = self.data.get(self.pos..).unwrap_or_default();
        let bytes = usize::try_from(len).ok().and_then(|len| rest.get(..len));
        let bytes = bytes.ok_or_else(|| self.truncated(field, len))?;
        self.pos = self.pos.saturating_add(bytes.len());
        Ok(bytes)
    }

    /// The next `N` bytes as an array.
    fn array<const N: usize>(&mut self, field: Field) -> Result<[u8; N], PacketError> {
        let rest = self.data.get(self.pos..).unwrap_or_default();
        let needed = u64::try_from(N).unwrap_or(u64::MAX);
        let array = *rest
            .first_chunk::<N>()
            .ok_or_else(|| self.truncated(field, needed))?;
        self.pos = self.pos.saturating_add(N);
        Ok(array)
    }

    /// An unsigned LEB128 number of at most [`MAX_LEB128_BYTES`] bytes.
    fn leb128(&mut self, field: Field) -> Result<u64, PacketError> {
        let start = self.pos;
        let mut value = 0;
        let mut shift = 0;
        for _ in 0..MAX_LEB128_BYTES {
            let [byte] = self.array(field)?;
            value |= u64::from(byte & LEB128_VALUE_MASK) << shift;
            if byte & LEB128_CONTINUE == 0 {
                return Ok(value);
            }
            shift += LEB128_SHIFT;
        }
        Err(PacketError::NumberTooLong {
            field,
            offset: start,
        })
    }
}
