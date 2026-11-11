//! The HIPC layer: the two header words, the special header, and the buffer descriptors.
//!
//! Layouts are ported from `eden/src/core/hle/ipc.h` (`CommandHeader`, `HandleDescriptorHeader`,
//! `BufferDescriptorX`, `BufferDescriptorABW`, `BufferDescriptorC`), which is GPL-3.0-or-later.
//! Every descriptor is a plain struct with `decode` (infallible, from raw words) and `encode`
//! (checked: a value that does not fit its bit-field is an error, never a silent truncation).

use crate::error::{IpcError, Result};

/// Size of a thread's IPC buffer in bytes (`COMMAND_BUFFER_LENGTH` is `0x100 / 4` words in
/// `eden/src/core/hle/ipc.h`).
pub const BUFFER_SIZE: usize = 0x100;
/// Largest count of X, A, B or W descriptors: each count is a 4-bit field of the first header
/// word (`eden/src/core/hle/ipc.h`, `CommandHeader`).
pub const MAX_DESCRIPTORS: u32 = 0xF;
/// Largest count of copy handles or of move handles: each is a 4-bit field of the special header
/// (`eden/src/core/hle/ipc.h`, `HandleDescriptorHeader`).
pub const MAX_HANDLES: u32 = 0xF;
/// Largest raw data size in words: the `data_size` field is 10 bits wide
/// (`eden/src/core/hle/ipc.h`, `CommandHeader`).
pub const MAX_RAW_WORDS: u32 = 0x3FF;
/// A TIPC command's type is `TIPC_COMMAND_OFFSET + command id`
/// (`eden/src/core/hle/ipc.h`, `CommandType::TIPC_CommandRegion`).
pub const TIPC_COMMAND_OFFSET: u16 = 16;
/// The largest TIPC command id whose command type still fits the 16-bit type field.
pub const MAX_TIPC_COMMAND_ID: u16 = u16::MAX - TIPC_COMMAND_OFFSET;

/// Highest address bit (exclusive) of an X or A/B/W descriptor: 32 low bits plus bits 32..36 and
/// 36..39.
const ADDRESS_BITS_ABW: u32 = 39;
/// Number of address bits in a C (receive) descriptor: 32 low bits plus 16 high bits.
const ADDRESS_BITS_C: u32 = 48;
/// Largest size of an A/B/W descriptor: 32 low bits plus 4 high bits.
const SIZE_BITS_ABW: u32 = 36;
/// The bits of an X descriptor counter that exist: 0..6 and 9..12 (bits 6..9 hold address bits).
const COUNTER_MASK: u32 = 0xE3F;

/// The low 32 bits of `value`.
pub(crate) fn low32(value: u64) -> u32 {
    let [a, b, c, d, ..] = value.to_le_bytes();
    u32::from_le_bytes([a, b, c, d])
}

/// Bits 32..64 of `value`.
pub(crate) fn high32(value: u64) -> u32 {
    low32(value >> 32)
}

/// The low 16 bits of `value`.
fn low16(value: u32) -> u16 {
    let [a, b, ..] = value.to_le_bytes();
    u16::from_le_bytes([a, b])
}

fn check(field: &'static str, value: u64, bits: u32) -> Result<()> {
    if value >> bits == 0 {
        Ok(())
    } else {
        Err(IpcError::FieldOutOfRange { field, value })
    }
}

/// The kind of message, from the low 16 bits of the first header word.
///
/// Values: `eden/src/core/hle/ipc.h`, `CommandType`. A CMIF *reply* carries
/// [`CommandType::Invalid`] (zero): Eden's `ResponseBuilder` leaves the type zero except for TIPC,
/// where it echoes the request's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommandType {
    /// Zero. Not a valid request type; it is what a CMIF reply carries.
    Invalid,
    /// Legacy request (1).
    LegacyRequest,
    /// Close the session (2). The rest of the header is not populated.
    Close,
    /// Legacy control (3).
    LegacyControl,
    /// Request (4).
    Request,
    /// Control (5).
    Control,
    /// Request with context (6).
    RequestWithContext,
    /// Control with context (7).
    ControlWithContext,
    /// TIPC close (15). Like [`CommandType::Close`], only the header is meaningful.
    TipcClose,
    /// A TIPC command: the type field is `16 + id`. The payload is the command id.
    Tipc(u16),
}

impl CommandType {
    /// Decodes the 16-bit type field. Values 8 to 14 are undefined and are an error.
    pub fn from_raw(raw: u16) -> Result<Self> {
        Ok(match raw {
            0 => Self::Invalid,
            1 => Self::LegacyRequest,
            2 => Self::Close,
            3 => Self::LegacyControl,
            4 => Self::Request,
            5 => Self::Control,
            6 => Self::RequestWithContext,
            7 => Self::ControlWithContext,
            15 => Self::TipcClose,
            16.. => Self::Tipc(raw - TIPC_COMMAND_OFFSET),
            8..=14 => return Err(IpcError::UnknownCommandType(raw)),
        })
    }

    /// Encodes the 16-bit type field. Fails for a TIPC command id above [`MAX_TIPC_COMMAND_ID`].
    pub fn to_raw(self) -> Result<u16> {
        Ok(match self {
            Self::Invalid => 0,
            Self::LegacyRequest => 1,
            Self::Close => 2,
            Self::LegacyControl => 3,
            Self::Request => 4,
            Self::Control => 5,
            Self::RequestWithContext => 6,
            Self::ControlWithContext => 7,
            Self::TipcClose => 15,
            Self::Tipc(id) => id
                .checked_add(TIPC_COMMAND_OFFSET)
                .ok_or(IpcError::TipcCommandOutOfRange(id))?,
        })
    }

    /// Whether this is a TIPC command (type 16 or above).
    pub fn is_tipc(self) -> bool {
        matches!(self, Self::Tipc(_))
    }

    /// Whether this closes the session: only the header is parsed
    /// (`eden/src/core/hle/ipc.h`, `CommandHeader::IsCloseCommand`).
    pub fn is_close(self) -> bool {
        matches!(self, Self::Close | Self::TipcClose)
    }
}

/// How the receive list (C descriptors) is described, from bits 10..14 of the second header word.
///
/// Eden: flag 0 disables it, 1 is an inline buffer (the reply is written straight after the raw
/// data and there are no descriptors), 2 is one descriptor, and any larger flag `n` is `n - 2`
/// descriptors (`eden/src/core/hle/service/hle_ipc.cpp`, `ParseCommandBuffer`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReceiveList {
    /// No receive list (flag 0).
    Disabled,
    /// Inline receive buffer (flag 1); there are no C descriptors.
    Inline,
    /// A single C descriptor (flag 2).
    One,
    /// `n` C descriptors (flag `n + 2`, so `n` is 1 to 13).
    Many(u32),
}

impl ReceiveList {
    /// Decodes the 4-bit flag.
    pub fn from_flag(flag: u32) -> Self {
        match flag {
            0 => Self::Disabled,
            1 => Self::Inline,
            2 => Self::One,
            n => Self::Many(n.saturating_sub(2)),
        }
    }

    /// Encodes the 4-bit flag. [`ReceiveList::Many`] must hold 1 to 13.
    pub fn to_flag(self) -> Result<u32> {
        match self {
            Self::Disabled => Ok(0),
            Self::Inline => Ok(1),
            Self::One => Ok(2),
            Self::Many(n) if (1..=13).contains(&n) => Ok(n + 2),
            Self::Many(n) => Err(IpcError::FieldOutOfRange {
                field: "receive list descriptor count",
                value: u64::from(n),
            }),
        }
    }

    /// How many C descriptors follow the raw data.
    pub fn descriptor_count(self) -> u32 {
        match self {
            Self::Disabled | Self::Inline => 0,
            Self::One => 1,
            Self::Many(n) => n,
        }
    }
}

/// The two HIPC header words.
///
/// Word 0: bits 0..16 type, then four 4-bit counts of X, A, B and W descriptors. Word 1: bits
/// 0..10 raw data size in words, 10..14 receive-list flag, bit 31 "a special header follows".
/// Other bits are ignored on decode and zero on encode
/// (`eden/src/core/hle/ipc.h`, `CommandHeader`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommandHeader {
    /// The message kind.
    pub command_type: CommandType,
    /// Number of X (static send) descriptors.
    pub num_x: u32,
    /// Number of A (send) descriptors.
    pub num_a: u32,
    /// Number of B (receive) descriptors.
    pub num_b: u32,
    /// Number of W (exchange) descriptors.
    pub num_w: u32,
    /// Size of the raw data section in words.
    pub raw_data_words: u32,
    /// How the receive list is described.
    pub receive_list: ReceiveList,
    /// Whether a [`SpecialHeader`] follows the header.
    pub has_special_header: bool,
}

impl CommandHeader {
    /// Decodes the two header words.
    pub fn decode(words: [u32; 2]) -> Result<Self> {
        let [w0, w1] = words;
        Ok(Self {
            command_type: CommandType::from_raw(low16(w0))?,
            num_x: (w0 >> 16) & 0xF,
            num_a: (w0 >> 20) & 0xF,
            num_b: (w0 >> 24) & 0xF,
            num_w: (w0 >> 28) & 0xF,
            raw_data_words: w1 & MAX_RAW_WORDS,
            receive_list: ReceiveList::from_flag((w1 >> 10) & 0xF),
            has_special_header: w1 >> 31 != 0,
        })
    }

    /// Encodes the two header words, checking every field against its bit-width.
    pub fn encode(&self) -> Result<[u32; 2]> {
        for (field, value) in [
            ("X descriptor count", self.num_x),
            ("A descriptor count", self.num_a),
            ("B descriptor count", self.num_b),
            ("W descriptor count", self.num_w),
        ] {
            check(field, u64::from(value), 4)?;
        }
        check("raw data words", u64::from(self.raw_data_words), 10)?;
        let w0 = u32::from(self.command_type.to_raw()?)
            | self.num_x << 16
            | self.num_a << 20
            | self.num_b << 24
            | self.num_w << 28;
        let w1 = self.raw_data_words
            | self.receive_list.to_flag()? << 10
            | u32::from(self.has_special_header) << 31;
        Ok([w0, w1])
    }
}

/// The special header that follows the command header when it says so: whether the caller's
/// process id is sent, and how many handles are copied and moved
/// (`eden/src/core/hle/ipc.h`, `HandleDescriptorHeader`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SpecialHeader {
    /// Bit 0: a two-word process id follows the special header.
    pub send_pid: bool,
    /// Bits 1..5: number of handles copied.
    pub num_copy_handles: u32,
    /// Bits 5..9: number of handles moved.
    pub num_move_handles: u32,
}

impl SpecialHeader {
    /// Decodes the special header word.
    pub fn decode(word: u32) -> Self {
        Self {
            send_pid: word & 1 != 0,
            num_copy_handles: (word >> 1) & 0xF,
            num_move_handles: (word >> 5) & 0xF,
        }
    }

    /// Encodes the special header word.
    pub fn encode(&self) -> Result<u32> {
        check("copy handle count", u64::from(self.num_copy_handles), 4)?;
        check("move handle count", u64::from(self.num_move_handles), 4)?;
        Ok(u32::from(self.send_pid) | self.num_copy_handles << 1 | self.num_move_handles << 5)
    }
}

/// An X (static send) descriptor: two words
/// (`eden/src/core/hle/ipc.h`, `BufferDescriptorX`).
///
/// Word 0: bits 0..6 counter bits 0..6, bits 6..9 address bits 36..39, bits 9..12 counter bits
/// 9..12, bits 12..16 address bits 32..36, bits 16..32 size. Word 1: address bits 0..32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticDescriptor {
    /// Index of the receive-list entry this buffer targets. Only bits 0..6 and 9..12 exist.
    pub counter: u32,
    /// Guest address (39 bits).
    pub address: u64,
    /// Size in bytes (16 bits).
    pub size: u16,
}

impl StaticDescriptor {
    /// Size of one descriptor in 32-bit words.
    pub const WORDS: usize = 2;

    /// Decodes two words.
    pub fn decode(words: [u32; 2]) -> Self {
        let [w0, w1] = words;
        Self {
            counter: (w0 & 0x3F) | (((w0 >> 9) & 0x7) << 9),
            address: u64::from(w1)
                | u64::from((w0 >> 12) & 0xF) << 32
                | u64::from((w0 >> 6) & 0x7) << 36,
            size: low16(w0 >> 16),
        }
    }

    /// Encodes two words. Fails if the counter has bits outside the existing ones or the address
    /// is wider than 39 bits.
    pub fn encode(&self) -> Result<[u32; 2]> {
        if self.counter & !COUNTER_MASK != 0 {
            return Err(IpcError::FieldOutOfRange {
                field: "static descriptor counter",
                value: u64::from(self.counter),
            });
        }
        check("static descriptor address", self.address, ADDRESS_BITS_ABW)?;
        let w0 = (self.counter & 0x3F)
            | (high32(self.address) >> 4 & 0x7) << 6
            | ((self.counter >> 9) & 0x7) << 9
            | (high32(self.address) & 0xF) << 12
            | u32::from(self.size) << 16;
        Ok([w0, low32(self.address)])
    }
}

/// An A, B or W descriptor (send, receive, exchange): three words
/// (`eden/src/core/hle/ipc.h`, `BufferDescriptorABW`).
///
/// Word 0: size bits 0..32. Word 1: address bits 0..32. Word 2: bits 0..2 flags, 2..5 address
/// bits 36..39, 24..28 size bits 32..36, 28..32 address bits 32..36.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferDescriptor {
    /// Guest address (39 bits).
    pub address: u64,
    /// Size in bytes (36 bits).
    pub size: u64,
    /// The 2-bit flags field, kept raw (Eden does not interpret it).
    pub flags: u32,
}

impl BufferDescriptor {
    /// Size of one descriptor in 32-bit words.
    pub const WORDS: usize = 3;

    /// Decodes three words.
    pub fn decode(words: [u32; 3]) -> Self {
        let [w0, w1, w2] = words;
        Self {
            address: u64::from(w1)
                | u64::from((w2 >> 28) & 0xF) << 32
                | u64::from((w2 >> 2) & 0x7) << 36,
            size: u64::from(w0) | u64::from((w2 >> 24) & 0xF) << 32,
            flags: w2 & 0x3,
        }
    }

    /// Encodes three words. Fails if the address is wider than 39 bits, the size wider than 36
    /// bits, or the flags wider than 2 bits.
    pub fn encode(&self) -> Result<[u32; 3]> {
        check("buffer descriptor address", self.address, ADDRESS_BITS_ABW)?;
        check("buffer descriptor size", self.size, SIZE_BITS_ABW)?;
        check("buffer descriptor flags", u64::from(self.flags), 2)?;
        let w2 = self.flags
            | (high32(self.address) >> 4 & 0x7) << 2
            | (high32(self.size) & 0xF) << 24
            | (high32(self.address) & 0xF) << 28;
        Ok([low32(self.size), low32(self.address), w2])
    }
}

/// A C (receive list) descriptor: two words, found after the raw data section
/// (`eden/src/core/hle/ipc.h`, `BufferDescriptorC`).
///
/// Word 0: address bits 0..32. Word 1: bits 0..16 address bits 32..48, bits 16..32 size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReceiveDescriptor {
    /// Guest address (48 bits).
    pub address: u64,
    /// Size in bytes (16 bits).
    pub size: u16,
}

impl ReceiveDescriptor {
    /// Size of one descriptor in 32-bit words.
    pub const WORDS: usize = 2;

    /// Decodes two words.
    pub fn decode(words: [u32; 2]) -> Self {
        let [w0, w1] = words;
        Self {
            address: u64::from(w0) | u64::from(w1 & 0xFFFF) << 32,
            size: low16(w1 >> 16),
        }
    }

    /// Encodes two words. Fails if the address is wider than 48 bits.
    pub fn encode(&self) -> Result<[u32; 2]> {
        check("receive descriptor address", self.address, ADDRESS_BITS_C)?;
        Ok([
            low32(self.address),
            (high32(self.address) & 0xFFFF) | u32::from(self.size) << 16,
        ])
    }
}
