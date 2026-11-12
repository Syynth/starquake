//! How `fsp-srv` reaches host files: requests on the injectable environment's host I/O channel
//! ([`Env::host_io`]), and the replies it expects.
//!
//! The file systems a run has are host folders (for now one: the SD card), named in the run's
//! configuration. `crates/services` never touches the host file system itself (the gate's
//! host-access step forbids it): it sends a [`HostRequest`] on [`CHANNEL`], and whoever built the
//! environment answers it. `starquake-core` does that from the configured folder; a test answers
//! from memory. Live runs record each exchange and replays feed it back, as for every other use
//! of the environment (`docs/specs/observability.md`, rule 1).
//!
//! Everything here is starquake's own: no reference has a host I/O channel. The layout is
//! versioned ([`VERSION`]) because recorded runs keep these bytes.
//!
//! # Request layout
//!
//! Little-endian, one fixed header and the path after it:
//!
//! ```text
//! 0x00  u8  VERSION
//! 0x01  u8  operation: 1 probe, 2 size, 3 read
//! 0x02  u8  mount: 1 the SD card
//! 0x03  u8  reserved, zero
//! 0x04  u32 path length in bytes, at most ENTRY_NAME_LENGTH_MAX
//! 0x08  u64 offset (read; zero otherwise)
//! 0x10  u64 length (read; zero otherwise)
//! 0x18  the path: normalised (`fsp::path`), no terminator; empty for a probe
//! ```
//!
//! # Replies
//!
//! | Request | Success | Failure |
//! |---|---|---|
//! | [`HostRequest::Probe`] | `Data`, empty: the mount is there | `Failed` with a [`HostFailure`] code, or `Unavailable` (no handler): the mount is not there |
//! | [`HostRequest::Size`] | `Data`, the file's size as a little-endian `u64` | `Failed` with a [`HostFailure`] code |
//! | [`HostRequest::Read`] | `Data`, the bytes from the offset on, at most the length asked for; fewer at the end of the file, none past it | `Failed` with a [`HostFailure`] code |

use starquake_trace::{Env, HostIoReply, HostIoRequest};

use super::path::ENTRY_NAME_LENGTH_MAX;

/// The host I/O channel `fsp-srv` uses. starquake's own number: the first user of the channel
/// space ([`HostIoRequest::channel`]).
pub const CHANNEL: u32 = 1;

/// The layout version of a request. Bump it when the layout changes.
pub const VERSION: u8 = 1;

/// Bytes before the path.
pub const HEADER_SIZE: usize = 0x18;

/// Offsets of a request's header fields (see the [module docs](self)).
pub mod layout {
    /// The version.
    pub const VERSION: usize = 0x00;
    /// The operation.
    pub const OPERATION: usize = 0x01;
    /// The mount.
    pub const MOUNT: usize = 0x02;
    /// The reserved byte.
    pub const RESERVED: usize = 0x03;
    /// The path's length.
    pub const PATH_LEN: usize = 0x04;
    /// The offset.
    pub const OFFSET: usize = 0x08;
    /// The length.
    pub const LENGTH: usize = 0x10;
}

/// A file system the host provides.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Mount {
    /// The SD card (`OpenSdCardFileSystem`).
    #[default]
    SdCard,
}

impl Mount {
    /// The mount's number in a request.
    #[must_use]
    pub const fn number(self) -> u8 {
        match self {
            Self::SdCard => 1,
        }
    }

    /// The mount a number names.
    #[must_use]
    pub const fn from_number(number: u8) -> Option<Self> {
        match number {
            1 => Some(Self::SdCard),
            _ => None,
        }
    }
}

/// The operation numbers of a request.
mod operation {
    pub(super) const PROBE: u8 = 1;
    pub(super) const SIZE: u8 = 2;
    pub(super) const READ: u8 = 3;
}

/// What `fsp-srv` asks of the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostRequest {
    /// Whether the mount is there.
    Probe {
        /// The mount.
        mount: Mount,
    },
    /// The size of the file at `path`.
    Size {
        /// The mount.
        mount: Mount,
        /// A normalised path in the mount.
        path: Vec<u8>,
    },
    /// Up to `len` bytes of the file at `path`, from `offset` on.
    Read {
        /// The mount.
        mount: Mount,
        /// A normalised path in the mount.
        path: Vec<u8>,
        /// Where to start.
        offset: u64,
        /// The most bytes to return.
        len: u64,
    },
}

/// Why bytes are not a [`HostRequest`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RequestError {
    /// Fewer bytes than the header.
    #[error("the request is {len} bytes, shorter than its {HEADER_SIZE}-byte header")]
    Truncated {
        /// The bytes there are.
        len: usize,
    },
    /// The version is not [`VERSION`].
    #[error("the request is version {0}, not {VERSION}")]
    Version(u8),
    /// An operation number no request has.
    #[error("operation {0} is not a host file request")]
    Operation(u8),
    /// A mount number no mount has.
    #[error("mount {0} is not a host file system")]
    Mount(u8),
    /// The reserved byte is not zero.
    #[error("the reserved byte is {0:#x}, not zero")]
    Reserved(u8),
    /// The path length is not what follows the header, or is longer than a path may be.
    #[error(
        "the path length {stated} does not match the {follows} bytes after the header, or passes the {ENTRY_NAME_LENGTH_MAX}-byte limit"
    )]
    PathLength {
        /// The length the header states.
        stated: u32,
        /// The bytes after the header.
        follows: usize,
    },
    /// A field the operation does not use is not zero (or a probe has a path).
    #[error("a field operation {0} does not use is not zero")]
    UnusedField(u8),
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    let mut word = [0; 4];
    if let Some(field) = bytes.get(offset..offset.saturating_add(4)) {
        word.copy_from_slice(field);
    }
    u32::from_le_bytes(word)
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    let mut word = [0; 8];
    if let Some(field) = bytes.get(offset..offset.saturating_add(8)) {
        word.copy_from_slice(field);
    }
    u64::from_le_bytes(word)
}

impl HostRequest {
    /// The request's bytes (see the [module docs](self)).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let (op, mount, path, offset, len): (u8, Mount, &[u8], u64, u64) = match self {
            Self::Probe { mount } => (operation::PROBE, *mount, &[], 0, 0),
            Self::Size { mount, path } => (operation::SIZE, *mount, path, 0, 0),
            Self::Read {
                mount,
                path,
                offset,
                len,
            } => (operation::READ, *mount, path, *offset, *len),
        };
        // A path longer than a `u32` cannot be a normalised one; its length saturates and the
        // decoder refuses it.
        let path_len = u32::try_from(path.len()).unwrap_or(u32::MAX);
        let mut bytes = Vec::with_capacity(HEADER_SIZE.saturating_add(path.len()));
        bytes.extend_from_slice(&[VERSION, op, mount.number(), 0]);
        bytes.extend_from_slice(&path_len.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(path);
        bytes
    }

    /// The request in `bytes`. The header's version is checked first, then each
    /// field in the order of the layout; the first that is wrong is the error.
    pub fn decode(bytes: &[u8]) -> Result<Self, RequestError> {
        let (header, path) = bytes
            .split_at_checked(HEADER_SIZE)
            .ok_or(RequestError::Truncated { len: bytes.len() })?;
        let byte = |offset: usize| header.get(offset).copied().unwrap_or_default();
        let version = byte(layout::VERSION);
        if version != VERSION {
            return Err(RequestError::Version(version));
        }
        let op = byte(layout::OPERATION);
        if !matches!(op, operation::PROBE | operation::SIZE | operation::READ) {
            return Err(RequestError::Operation(op));
        }
        let mount_number = byte(layout::MOUNT);
        let mount = Mount::from_number(mount_number).ok_or(RequestError::Mount(mount_number))?;
        let reserved = byte(layout::RESERVED);
        if reserved != 0 {
            return Err(RequestError::Reserved(reserved));
        }
        let stated = u32_at(header, layout::PATH_LEN);
        let fits = usize::try_from(stated)
            .is_ok_and(|stated| stated == path.len() && stated <= ENTRY_NAME_LENGTH_MAX);
        if !fits {
            return Err(RequestError::PathLength {
                stated,
                follows: path.len(),
            });
        }
        let offset = u64_at(header, layout::OFFSET);
        let len = u64_at(header, layout::LENGTH);
        let path = path.to_vec();
        match op {
            operation::PROBE if path.is_empty() && offset == 0 && len == 0 => {
                Ok(Self::Probe { mount })
            }
            operation::SIZE if offset == 0 && len == 0 => Ok(Self::Size { mount, path }),
            operation::READ => Ok(Self::Read {
                mount,
                path,
                offset,
                len,
            }),
            _ => Err(RequestError::UnusedField(op)),
        }
    }

    /// The request as the environment takes it.
    #[must_use]
    pub fn to_host_io(&self) -> HostIoRequest {
        HostIoRequest {
            channel: CHANNEL,
            payload: self.encode(),
        }
    }
}

/// Why the host could not do what a [`HostRequest`] asked: the code of a
/// [`HostIoReply::Failed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostFailure {
    /// The mount is not there.
    NoDevice,
    /// Nothing is at the path.
    NotFound,
    /// What is at the path is not a file.
    NotAFile,
    /// The host failed to read what is there.
    Io,
    /// The request could not be decoded.
    BadRequest,
}

impl HostFailure {
    /// Every failure, in code order.
    pub const ALL: [Self; 5] = [
        Self::NoDevice,
        Self::NotFound,
        Self::NotAFile,
        Self::Io,
        Self::BadRequest,
    ];

    /// The failure's code in a [`HostIoReply::Failed`].
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::NoDevice => 1,
            Self::NotFound => 2,
            Self::NotAFile => 3,
            Self::Io => 4,
            Self::BadRequest => 5,
        }
    }

    /// The failure a code names.
    #[must_use]
    pub fn from_code(code: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|failure| failure.code() == code)
    }

    /// The reply that reports it.
    #[must_use]
    pub fn reply(self) -> HostIoReply {
        HostIoReply::Failed(self.code())
    }
}

/// What came back for a request, as `fsp-srv` reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Answer {
    /// The host's bytes.
    Data(Vec<u8>),
    /// The host failed, with a code this module knows.
    Failed(HostFailure),
    /// Nothing handles host I/O: the run has no host file systems.
    Unavailable,
    /// The host failed with a code this module does not know, or the environment itself failed
    /// (a replay that diverged from its recording).
    Broken,
}

/// Sends `request` on the environment and sorts the reply. An environment that fails, and a code
/// no [`HostFailure`] has, are logged.
pub(crate) fn ask(env: &dyn Env, request: &HostRequest) -> Answer {
    match env.host_io(&request.to_host_io()) {
        Ok(HostIoReply::Data(bytes)) => Answer::Data(bytes),
        Ok(HostIoReply::Failed(code)) => HostFailure::from_code(code).map_or_else(
            || {
                tracing::error!(
                    code,
                    "the host failed a file request with a code it has no name for"
                );
                Answer::Broken
            },
            Answer::Failed,
        ),
        Ok(HostIoReply::Unavailable) => Answer::Unavailable,
        Err(error) => {
            tracing::error!(%error, "the environment failed a host file request");
            Answer::Broken
        }
    }
}

#[cfg(test)]
mod tests;
