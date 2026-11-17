//! The kernel's snapshots: the header every kernel table's snapshot starts with, and the rules
//! that read it (`docs/specs/observability.md`, "Snapshots": "versioned; a snapshot from
//! another build is refused, not misread"; "Order of work", step 6, "kernel and memory first").
//!
//! Each table saves and restores itself; this module holds what they share, so that the header
//! is written and checked once. The tables with a snapshot so far:
//!
//! - A process's address space: [`crate::memory::AddressSpace::save`] and
//!   [`crate::memory::AddressSpace::restore`] (`memory::space::snapshot`), with the magic
//!   [`crate::memory::ADDRESS_SPACE_MAGIC`].
//!
//! # Format
//!
//! `postcard`: a header of four magic bytes, which name the table, and a layout version (a
//! `u32`, as a `postcard` varint), then the table's body. [`decode`] checks, in this order, and
//! returns the first failure: that the header decodes, the magic, the version, that the body
//! decodes, and that no bytes follow it. A table's own checks of what the body holds come
//! after these. Changing a body's layout means bumping its table's version; each table pins its
//! layout with a golden-bytes test.

use serde::Serialize;
use serde::de::DeserializeOwned;

/// Which snapshot a [`Format`] reads and writes: its name for messages, its magic and the one
/// layout version this build reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    /// What the snapshot holds, as a message names it ("address space").
    pub name: &'static str,
    /// The first four bytes of every snapshot of this kind.
    pub magic: [u8; 4],
    /// The layout version written into every snapshot of this kind, and the only one read.
    pub version: u32,
}

/// The header every kernel snapshot starts with.
#[derive(Serialize, serde::Deserialize)]
struct Header {
    /// [`Format::magic`].
    magic: [u8; 4],
    /// [`Format::version`].
    version: u32,
}

/// Why a kernel snapshot's header or body could not be written or read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FormatError {
    /// Encoding failed.
    #[error("could not encode the {name} snapshot: {error}")]
    Encode {
        /// The snapshot's [`Format::name`].
        name: &'static str,
        /// What `postcard` said.
        error: postcard::Error,
    },
    /// The header is too short or malformed.
    #[error("could not decode the {name} snapshot's header: {error}")]
    Header {
        /// The snapshot's [`Format::name`].
        name: &'static str,
        /// What `postcard` said.
        error: postcard::Error,
    },
    /// The bytes do not start with the format's magic.
    #[error("the {name} snapshot has the wrong magic: these bytes are not one")]
    BadMagic {
        /// The snapshot's [`Format::name`].
        name: &'static str,
    },
    /// The snapshot was written with another layout version.
    #[error(
        "{name} snapshot layout version {found} is not supported (this build reads {supported})"
    )]
    UnsupportedVersion {
        /// The snapshot's [`Format::name`].
        name: &'static str,
        /// The version found.
        found: u32,
        /// The version this build reads.
        supported: u32,
    },
    /// The body is not a well-formed encoding.
    #[error("could not decode the {name} snapshot: {error}")]
    Body {
        /// The snapshot's [`Format::name`].
        name: &'static str,
        /// What `postcard` said.
        error: postcard::Error,
    },
    /// Bytes follow the body.
    #[error("{count} unexpected bytes after the {name} snapshot")]
    TrailingBytes {
        /// The snapshot's [`Format::name`].
        name: &'static str,
        /// How many.
        count: usize,
    },
}

/// `body` written after `format`'s header.
pub fn encode<T: Serialize>(format: Format, body: &T) -> Result<Vec<u8>, FormatError> {
    let encode = |error| FormatError::Encode {
        name: format.name,
        error,
    };
    let mut bytes = postcard::to_allocvec(&Header {
        magic: format.magic,
        version: format.version,
    })
    .map_err(encode)?;
    bytes.extend(postcard::to_allocvec(body).map_err(encode)?);
    Ok(bytes)
}

/// The body of a snapshot written by [`encode`] with `format`, after the checks in the module
/// docs ("Format"), in their order.
pub fn decode<T: DeserializeOwned>(format: Format, bytes: &[u8]) -> Result<T, FormatError> {
    let name = format.name;
    let (header, rest) = postcard::take_from_bytes::<Header>(bytes)
        .map_err(|error| FormatError::Header { name, error })?;
    if header.magic != format.magic {
        return Err(FormatError::BadMagic { name });
    }
    if header.version != format.version {
        return Err(FormatError::UnsupportedVersion {
            name,
            found: header.version,
            supported: format.version,
        });
    }
    let (body, rest) =
        postcard::take_from_bytes::<T>(rest).map_err(|error| FormatError::Body { name, error })?;
    if !rest.is_empty() {
        return Err(FormatError::TrailingBytes {
            name,
            count: rest.len(),
        });
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A format for these tests.
    const TEST: Format = Format {
        name: "test",
        magic: *b"SQTT",
        version: 3,
    };

    /// A small body with two ordered fields.
    #[derive(Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
    struct Body {
        first: u8,
        second: u16,
    }

    const BODY: Body = Body {
        first: 0x11,
        second: 0x22,
    };

    #[test]
    fn the_header_and_body_are_pinned() {
        // The magic, the version as a varint, then the body's fields in order.
        let bytes = encode(TEST, &BODY).unwrap();
        assert_eq!(bytes, [b'S', b'Q', b'T', b'T', 3, 0x11, 0x22]);
        assert_eq!(decode::<Body>(TEST, &bytes), Ok(BODY));
    }

    #[test]
    fn a_hand_built_snapshot_with_a_long_version_varint_is_read() {
        // A version of 300 takes two varint bytes (0xac 0x02): the header is not fixed-size.
        let format = Format {
            version: 300,
            ..TEST
        };
        let bytes = [b'S', b'Q', b'T', b'T', 0xac, 0x02, 0x11, 0x22];
        assert_eq!(decode::<Body>(format, &bytes), Ok(BODY));
    }

    #[test]
    fn each_header_check_has_its_own_error_in_order() {
        let good = encode(TEST, &BODY).unwrap();
        // Too short for the magic.
        assert!(matches!(
            decode::<Body>(TEST, &good[..3]),
            Err(FormatError::Header { name: "test", .. })
        ));
        // The magic is checked before the version: both wrong reports the magic.
        let mut both = good.clone();
        both[0] = b'X';
        both[4] = u8::try_from(TEST.version + 1).unwrap();
        assert_eq!(
            decode::<Body>(TEST, &both),
            Err(FormatError::BadMagic { name: "test" })
        );
        // The version is checked before the body: a wrong version with a short body reports
        // the version.
        let mut version = good[..5].to_vec();
        version[4] = u8::try_from(TEST.version + 1).unwrap();
        assert_eq!(
            decode::<Body>(TEST, &version),
            Err(FormatError::UnsupportedVersion {
                name: "test",
                found: TEST.version + 1,
                supported: TEST.version
            })
        );
        // A short body.
        assert!(matches!(
            decode::<Body>(TEST, &good[..6]),
            Err(FormatError::Body { name: "test", .. })
        ));
        // A trailing byte.
        let mut trailing = good;
        trailing.push(0);
        assert_eq!(
            decode::<Body>(TEST, &trailing),
            Err(FormatError::TrailingBytes {
                name: "test",
                count: 1
            })
        );
    }

    #[test]
    fn the_errors_read_as_plain_words() {
        assert_eq!(
            FormatError::BadMagic { name: "test" }.to_string(),
            "the test snapshot has the wrong magic: these bytes are not one"
        );
        assert_eq!(
            FormatError::UnsupportedVersion {
                name: "test",
                found: 4,
                supported: 3
            }
            .to_string(),
            "test snapshot layout version 4 is not supported (this build reads 3)"
        );
        assert_eq!(
            FormatError::TrailingBytes {
                name: "test",
                count: 2
            }
            .to_string(),
            "2 unexpected bytes after the test snapshot"
        );
    }
}
