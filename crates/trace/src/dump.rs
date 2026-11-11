//! The flight-recorder dump: its bytes (`postcard`) and a reader for them.
//!
//! Source: `docs/specs/observability.md`, "Flight recorder" ("written out on a guest crash, an
//! emulator error, a timeout, or on request") and the artifacts table (`events.bin`: "Flight
//! recorder dump"). The spec fixes neither a byte layout nor a header, so the layout here is a
//! guess of this crate's own: a fixed [`DUMP_MAGIC`] and [`DUMP_VERSION`] followed by the cores,
//! each with its events oldest first, all encoded with `postcard`. A dump from another version is
//! refused, not misread (the same rule the spec sets for snapshots).

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::event::Event;

/// First bytes of every dump, so a file of another kind is refused early.
pub const DUMP_MAGIC: [u8; 4] = *b"SQFR";

/// Layout version written into every dump and required when reading one.
pub const DUMP_VERSION: u32 = 1;

/// Why a dump could not be written or read.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DumpError {
    /// Encoding failed.
    #[error("could not encode the dump: {0}")]
    Encode(postcard::Error),
    /// The bytes are not a well-formed encoding (truncated, or a field out of range).
    #[error("could not decode the dump: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`DUMP_MAGIC`].
    #[error("not a flight-recorder dump (bad magic)")]
    BadMagic,
    /// The dump was written with a different layout version.
    #[error("dump layout version {found} is not supported (this build reads {DUMP_VERSION})")]
    UnsupportedVersion {
        /// The version found in the dump.
        found: u32,
    },
    /// Bytes remain after the dump.
    #[error("{count} trailing bytes after the dump")]
    TrailingBytes {
        /// How many bytes were left over.
        count: usize,
    },
}

/// The events one core's ring held, oldest first.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreDump {
    /// The core the ring belongs to.
    pub core: u8,
    /// The retained events, oldest first (increasing `seq`).
    pub events: Vec<Event>,
}

/// A written-out flight recorder: one [`CoreDump`] per core.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dump {
    magic: [u8; 4],
    version: u32,
    /// Every core's retained events.
    pub cores: Vec<CoreDump>,
}

impl Dump {
    /// A dump of the given cores, stamped with the current magic and version.
    #[must_use]
    pub fn new(cores: Vec<CoreDump>) -> Self {
        Self {
            magic: DUMP_MAGIC,
            version: DUMP_VERSION,
            cores,
        }
    }

    /// Encode the dump.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DumpError> {
        postcard::to_allocvec(self).map_err(DumpError::Encode)
    }

    /// Decode a dump. Malformed, foreign, other-version or over-long input is an `Err`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DumpError> {
        let (dump, rest) = postcard::take_from_bytes::<Self>(bytes).map_err(DumpError::Decode)?;
        if dump.magic != DUMP_MAGIC {
            return Err(DumpError::BadMagic);
        }
        if dump.version != DUMP_VERSION {
            return Err(DumpError::UnsupportedVersion {
                found: dump.version,
            });
        }
        if !rest.is_empty() {
            return Err(DumpError::TrailingBytes { count: rest.len() });
        }
        Ok(dump)
    }

    /// Every retained event from every core in one list, sorted by `seq`: the order the
    /// recorder assigned.
    #[must_use]
    pub fn merged(&self) -> Vec<Event> {
        let mut all: Vec<Event> = self
            .cores
            .iter()
            .flat_map(|c| c.events.iter().copied())
            .collect();
        all.sort_by_key(|e| e.seq);
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EventKind, GuestTicks, LifecyclePhase, ThreadId};

    fn event(seq: u64, core: u8) -> Event {
        Event {
            seq,
            time: GuestTicks(seq * 10),
            core,
            thread: ThreadId(u64::from(core)),
            kind: EventKind::Lifecycle {
                phase: LifecyclePhase::ProcessStart,
                subject: seq,
            },
        }
    }

    fn sample() -> Dump {
        Dump::new(vec![
            CoreDump {
                core: 0,
                events: vec![event(0, 0), event(3, 0)],
            },
            CoreDump {
                core: 1,
                events: vec![event(1, 1), event(2, 1)],
            },
        ])
    }

    #[test]
    fn round_trips() {
        let bytes = sample().to_bytes().unwrap();
        assert_eq!(Dump::from_bytes(&bytes).unwrap(), sample());
    }

    #[test]
    fn merged_is_sorted_by_seq_across_cores() {
        let seqs: Vec<u64> = sample().merged().iter().map(|e| e.seq).collect();
        assert_eq!(seqs, [0, 1, 2, 3]);
    }

    #[test]
    fn wrong_magic_is_refused() {
        let mut dump = sample();
        dump.magic = [0; 4];
        let bytes = dump.to_bytes().unwrap();
        assert_eq!(Dump::from_bytes(&bytes), Err(DumpError::BadMagic));
    }

    #[test]
    fn other_version_is_refused() {
        let mut dump = sample();
        dump.version = DUMP_VERSION + 1;
        let bytes = dump.to_bytes().unwrap();
        assert_eq!(
            Dump::from_bytes(&bytes),
            Err(DumpError::UnsupportedVersion {
                found: DUMP_VERSION + 1
            })
        );
    }

    #[test]
    fn trailing_bytes_are_refused() {
        let mut bytes = sample().to_bytes().unwrap();
        bytes.extend_from_slice(&[1, 2, 3]);
        assert_eq!(
            Dump::from_bytes(&bytes),
            Err(DumpError::TrailingBytes { count: 3 })
        );
    }

    #[test]
    fn every_truncation_is_an_error_not_a_panic() {
        let bytes = sample().to_bytes().unwrap();
        for len in 0..bytes.len() {
            let prefix = bytes.get(..len).unwrap();
            assert!(
                matches!(Dump::from_bytes(prefix), Err(DumpError::Decode(_))),
                "prefix of {len} bytes decoded"
            );
        }
    }
}
