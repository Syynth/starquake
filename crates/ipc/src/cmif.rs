//! The CMIF layer: `SFCI`/`SFCO` payload headers and domain message headers.
//!
//! Ported from `eden/src/core/hle/ipc.h` (`DataPayloadHeader`, `DomainMessageHeader`),
//! `eden/src/core/hle/service/hle_ipc.cpp` (`ParseCommandBuffer`) and
//! `eden/src/core/hle/service/ipc_helpers.h` (`ResponseBuilder`), GPL-3.0-or-later.
//!
//! Guesses, because Eden treats these words as padding or never reads them: the name `token` for
//! the fourth word of the `SFCI`/`SFCO` headers and of the domain request header (the issue
//! brief names them so), and that the domain request `size` counts the `SFCI` header plus the
//! arguments (Eden never reads `size`; this is the only reading under which the trailing object
//! ids can be located).

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
    /// Bytes of `SFCI` header plus arguments that follow this header (see the module guesses).
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
            command: DomainCommand::from_raw(w0 & 0xFF)?,
            input_object_count: (w0 >> 8) & 0xFF,
            size: w0 >> 16,
            object_id,
            token,
        })
    }

    /// Encodes four words, checking the count and size fit their fields.
    pub fn encode(&self) -> Result<[u32; 4]> {
        if self.input_object_count > 0xFF {
            return Err(IpcError::FieldOutOfRange {
                field: "domain input object count",
                value: u64::from(self.input_object_count),
            });
        }
        if self.size > 0xFFFF {
            return Err(IpcError::FieldOutOfRange {
                field: "domain message size",
                value: u64::from(self.size),
            });
        }
        let w0 = self.command.to_raw() | self.input_object_count << 8 | self.size << 16;
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
