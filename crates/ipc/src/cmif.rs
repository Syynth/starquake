//! The CMIF layer: `SFCI`/`SFCO` payload headers and domain message headers.
//!
//! Ported from `eden/src/core/hle/ipc.h` (`DataPayloadHeader`, `DomainMessageHeader`),
//! `eden/src/core/hle/service/hle_ipc.cpp` (`ParseCommandBuffer`) and
//! `eden/src/core/hle/service/ipc_helpers.h` (`ResponseBuilder`), GPL-3.0-or-later.
//!
//! Field names and layout were checked against libnx (`libnx/nx/include/switch/sf/cmif.h`,
//! `CmifInHeader`, `CmifOutHeader`, `CmifDomainInHeader`, `CmifDomainOutHeader`; ISC): the fourth
//! word of the `SFCI`/`SFCO` headers and of the domain request header is called `token` there,
//! and the third word of the domain request header and the last three of the domain reply header
//! are padding. Eden never reads these words.
//!
//! The domain request `size` field is the `SFCI` header plus the arguments. Eden never reads it;
//! libnx writes the exact argument length and places the object ids straight after it, and
//! [`DomainRequestHeader::size`] documents what each [`crate::RequestLayout`] writes.

use crate::error::{IpcError, Result};

/// `"SFCI"` as a little-endian word: opens a CMIF request payload
/// (`eden/src/core/hle/service/hle_ipc.cpp`, `Common::MakeMagic('S', 'F', 'C', 'I')`).
pub const SFCI_MAGIC: u32 = u32::from_le_bytes(*b"SFCI");
/// `"SFCO"` as a little-endian word: opens a CMIF reply payload.
pub const SFCO_MAGIC: u32 = u32::from_le_bytes(*b"SFCO");
/// Size in bytes of an `SFCI` or `SFCO` header: magic, version, command id or result, token.
pub const CMIF_HEADER_BYTES: usize = 16;
/// Size in bytes of a domain message header, request or reply
/// (`eden/src/core/hle/ipc.h`, `static_assert(sizeof(DomainMessageHeader) == 16)`).
pub const DOMAIN_HEADER_BYTES: usize = 16;
/// Bit position of the input object count in the first domain request header word.
const DOMAIN_COUNT_SHIFT: u32 = 8;
/// Bit position of the size in the first domain request header word.
const DOMAIN_SIZE_SHIFT: u32 = 16;
/// Mask of the 8-bit command at the bottom of the first domain request header word.
const DOMAIN_COMMAND_MASK: u32 = (1 << DOMAIN_COUNT_SHIFT) - 1;
/// Largest number of input object ids in a domain request: the count is bits 8..16 of the first
/// domain header word (`eden/src/core/hle/ipc.h`, `DomainMessageHeader`; `num_in_objects` is a
/// `u8` in `libnx/nx/include/switch/sf/cmif.h`, `CmifDomainInHeader`).
pub const MAX_DOMAIN_INPUT_OBJECTS: u32 = (1 << (DOMAIN_SIZE_SHIFT - DOMAIN_COUNT_SHIFT)) - 1;
/// Largest value of a domain request's `size` field: bits 16..32 of the first domain header word
/// (`eden/src/core/hle/ipc.h`, `DomainMessageHeader`; `data_size` is a `u16` in libnx's
/// `CmifDomainInHeader`).
pub const MAX_DOMAIN_SIZE: u32 = u32::MAX >> DOMAIN_SIZE_SHIFT;

/// What a domain request asks of an object
/// (`eden/src/core/hle/ipc.h`, `DomainMessageHeader::CommandType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DomainCommand {
    /// Send a message to the object (1). An `SFCI` header and arguments follow.
    SendMessage,
    /// Close the object's virtual handle (2). There is no `SFCI` header and no data.
    CloseVirtualHandle,
}

impl DomainCommand {
    /// Decodes the 8-bit command.
    pub fn from_raw(raw: u32) -> Result<Self> {
        match raw {
            1 => Ok(Self::SendMessage),
            2 => Ok(Self::CloseVirtualHandle),
            other => Err(IpcError::UnknownDomainCommand(
                other.to_le_bytes().first().copied().unwrap_or_default(),
            )),
        }
    }

    /// The 8-bit command value.
    pub fn to_raw(self) -> u32 {
        match self {
            Self::SendMessage => 1,
            Self::CloseVirtualHandle => 2,
        }
    }
}

/// The client-to-server domain header that precedes the `SFCI` header in a domain request.
///
/// Word 0: bits 0..8 command, 8..16 input object count, 16..32 size in bytes. Word 1: object id.
/// Word 2 is padding. Word 3 is the token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DomainRequestHeader {
    /// What the request asks.
    pub command: DomainCommand,
    /// How many object ids follow the arguments.
    pub input_object_count: u32,
    /// Bytes of `SFCI` header plus arguments that follow this header. libnx writes the exact
    /// argument length (`libnx/nx/include/switch/sf/cmif.h:131`), as [`crate::RequestBuilder`]
    /// does in [`crate::RequestLayout::Libnx`]; in [`crate::RequestLayout::Exact`] it rounds the
    /// arguments up to a whole word.
    pub size: u32,
    /// The target object.
    pub object_id: u32,
    /// The token word.
    pub token: u32,
}

impl DomainRequestHeader {
    /// Decodes four words.
    pub fn decode(words: [u32; 4]) -> Result<Self> {
        let [w0, object_id, _padding, token] = words;
        Ok(Self {
            command: DomainCommand::from_raw(w0 & DOMAIN_COMMAND_MASK)?,
            input_object_count: (w0 >> DOMAIN_COUNT_SHIFT) & MAX_DOMAIN_INPUT_OBJECTS,
            size: w0 >> DOMAIN_SIZE_SHIFT,
            object_id,
            token,
        })
    }

    /// Encodes four words, checking the count and size fit their fields.
    pub fn encode(&self) -> Result<[u32; 4]> {
        if self.input_object_count > MAX_DOMAIN_INPUT_OBJECTS {
            return Err(IpcError::FieldOutOfRange {
                field: "domain input object count",
                value: u64::from(self.input_object_count),
            });
        }
        if self.size > MAX_DOMAIN_SIZE {
            return Err(IpcError::FieldOutOfRange {
                field: "domain message size",
                value: u64::from(self.size),
            });
        }
        let w0 = self.command.to_raw()
            | self.input_object_count << DOMAIN_COUNT_SHIFT
            | self.size << DOMAIN_SIZE_SHIFT;
        Ok([w0, self.object_id, 0, self.token])
    }
}

/// The server-to-client domain header that precedes the `SFCO` header in a domain reply.
///
/// Word 0 is the number of object ids that trail the data; words 1 to 3 are padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DomainReplyHeader {
    /// How many object ids follow the data.
    pub num_objects: u32,
}

impl DomainReplyHeader {
    /// Decodes four words.
    pub fn decode(words: [u32; 4]) -> Self {
        Self {
            num_objects: words[0],
        }
    }

    /// Encodes four words.
    pub fn encode(&self) -> [u32; 4] {
        [self.num_objects, 0, 0, 0]
    }
}

/// An `SFCI` header: magic, version, command id, token.
///
/// Eden reads the command id as a `u64` of which it keeps the low half (`hle_ipc.cpp`); the high
/// half is the token here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CmifRequestHeader {
    /// The version word (Eden does not check it).
    pub version: u32,
    /// The command id.
    pub command_id: u32,
    /// The token word.
    pub token: u32,
}

impl CmifRequestHeader {
    /// Decodes four words, checking the magic.
    pub fn decode(words: [u32; 4]) -> Result<Self> {
        let [magic, version, command_id, token] = words;
        if magic != SFCI_MAGIC {
            return Err(IpcError::BadMagic {
                found: magic,
                expected: SFCI_MAGIC,
            });
        }
        Ok(Self {
            version,
            command_id,
            token,
        })
    }

    /// Encodes four words.
    pub fn encode(&self) -> [u32; 4] {
        [SFCI_MAGIC, self.version, self.command_id, self.token]
    }
}

/// An `SFCO` header: magic, version, result, token.
///
/// Eden writes the version as padding and the result as a 32-bit code followed by a zero word
/// (`ipc_helpers.h`, `PushImpl(Result)`); the zero word is the token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CmifReplyHeader {
    /// The version word.
    pub version: u32,
    /// The result code (zero is success).
    pub result: u32,
    /// The token word.
    pub token: u32,
}

impl CmifReplyHeader {
    /// Decodes four words, checking the magic.
    pub fn decode(words: [u32; 4]) -> Result<Self> {
        let [magic, version, result, token] = words;
        if magic != SFCO_MAGIC {
            return Err(IpcError::BadMagic {
                found: magic,
                expected: SFCO_MAGIC,
            });
        }
        Ok(Self {
            version,
            result,
            token,
        })
    }

    /// Encodes four words.
    pub fn encode(&self) -> [u32; 4] {
        [SFCO_MAGIC, self.version, self.result, self.token]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magics_spell_their_names_in_memory_order() {
        assert_eq!(SFCI_MAGIC.to_le_bytes(), *b"SFCI");
        assert_eq!(SFCO_MAGIC.to_le_bytes(), *b"SFCO");
    }

    fn domain_header(input_object_count: u32, size: u32) -> DomainRequestHeader {
        DomainRequestHeader {
            command: DomainCommand::SendMessage,
            input_object_count,
            size,
            object_id: 0x0BAD_0001,
            token: 0xC0DE_0002,
        }
    }

    /// Both limits are inclusive: a header at exactly `MAX_DOMAIN_INPUT_OBJECTS` objects and one at
    /// exactly `MAX_DOMAIN_SIZE` bytes encode and decode back unchanged, in separate headers so
    /// neither field masks the other.
    #[test]
    fn domain_request_header_accepts_exactly_the_limits() {
        for header in [
            domain_header(MAX_DOMAIN_INPUT_OBJECTS, 1),
            domain_header(1, MAX_DOMAIN_SIZE),
            domain_header(MAX_DOMAIN_INPUT_OBJECTS, MAX_DOMAIN_SIZE),
        ] {
            let words = header.encode();
            assert!(words.is_ok(), "{header:?}: {words:?}");
            assert_eq!(
                words.and_then(DomainRequestHeader::decode),
                Ok(header),
                "{header:?}"
            );
        }
    }

    /// One past either limit is a `FieldOutOfRange` naming the field and the value.
    #[test]
    fn domain_request_header_rejects_one_past_the_limits() {
        assert_eq!(
            domain_header(MAX_DOMAIN_INPUT_OBJECTS + 1, 1).encode(),
            Err(IpcError::FieldOutOfRange {
                field: "domain input object count",
                value: u64::from(MAX_DOMAIN_INPUT_OBJECTS) + 1,
            })
        );
        assert_eq!(
            domain_header(1, MAX_DOMAIN_SIZE + 1).encode(),
            Err(IpcError::FieldOutOfRange {
                field: "domain message size",
                value: u64::from(MAX_DOMAIN_SIZE) + 1,
            })
        );
    }

    #[test]
    fn headers_are_sixteen_bytes() {
        assert_eq!(
            CMIF_HEADER_BYTES,
            4 * CmifRequestHeader::decode([SFCI_MAGIC, 0, 0, 0])
                .map(|h| h.encode().len())
                .unwrap()
        );
        assert_eq!(
            DOMAIN_HEADER_BYTES,
            4 * DomainReplyHeader { num_objects: 0 }.encode().len()
        );
    }
}
