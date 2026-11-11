//! The flight-recorder dump: its bytes (`postcard`) and a reader for them.
//!
//! Source: `docs/specs/observability.md`, "Flight recorder" ("written out on a guest crash, an
//! emulator error, a timeout, or on request") and the artifacts table (`events.bin`: "Flight
//! recorder dump"). The spec fixes neither a byte layout nor a header, so the layout here is a
//! guess of this crate's own: a fixed [`DUMP_MAGIC`] and [`DUMP_VERSION`] followed by the cores,
//! each with its events oldest first, all encoded with `postcard`. A dump from another version is
//! refused, not misread (the same rule the spec sets for snapshots): the header is read and
//! checked before the cores are decoded, so a dump whose cores are laid out differently still
//! reports its version.
//!
//! The byte layout is pinned by a golden-bytes test (`the_byte_layout_is_pinned`). Any change to
//! it, or to the field order or types of [`Event`], [`EventKind`](crate::event::EventKind) or
//! [`CoreDump`], changes `events.bin` and requires bumping [`DUMP_VERSION`] in the same change.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::event::Event;

/// First bytes of every dump, so a file of another kind is refused before its body is decoded.
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
    ///
    /// The header (magic, then version) is decoded and checked before anything after it, so a
    /// file written with another layout is reported as [`DumpError::BadMagic`] or
    /// [`DumpError::UnsupportedVersion`] and not as a decode failure of a body this build cannot
    /// read. Input too short to hold the header is [`DumpError::Decode`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DumpError> {
        let (magic, after_magic) =
            postcard::take_from_bytes::<[u8; 4]>(bytes).map_err(DumpError::Decode)?;
        if magic != DUMP_MAGIC {
            return Err(DumpError::BadMagic);
        }
        let (version, body) =
            postcard::take_from_bytes::<u32>(after_magic).map_err(DumpError::Decode)?;
        if version != DUMP_VERSION {
            return Err(DumpError::UnsupportedVersion { found: version });
        }
        let (cores, rest) =
            postcard::take_from_bytes::<Vec<CoreDump>>(body).map_err(DumpError::Decode)?;
        if !rest.is_empty() {
            return Err(DumpError::TrailingBytes { count: rest.len() });
        }
        Ok(Self {
            magic,
            version,
            cores,
        })
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
    use crate::event::{
        EventKind, FaultAction, GuestTicks, IpcOutcome, LifecyclePhase, MapKind, ResultCode,
        ServiceId, SessionId, ThreadId, WakeReason,
    };

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

    /// A later layout: same magic, a different version, and a header field this build has never
    /// heard of. The version must be read before anything after it is decoded.
    #[derive(Serialize)]
    struct OtherLayout {
        magic: [u8; 4],
        version: u32,
        added_in_the_other_version: u64,
        cores: Vec<CoreDump>,
    }

    #[test]
    fn a_different_layout_is_reported_as_its_version_not_as_a_decode_failure() {
        let other = OtherLayout {
            magic: DUMP_MAGIC,
            version: DUMP_VERSION + 1,
            added_in_the_other_version: u64::MAX,
            cores: sample().cores,
        };
        let bytes = postcard::to_allocvec(&other).unwrap();
        assert_eq!(
            Dump::from_bytes(&bytes),
            Err(DumpError::UnsupportedVersion {
                found: DUMP_VERSION + 1
            })
        );
    }

    #[test]
    fn a_foreign_file_with_a_long_body_is_refused_on_its_magic() {
        // Not a dump at all: the first four bytes are wrong and what follows is not a valid body.
        let mut bytes = vec![0xff; 64];
        bytes.get_mut(..DUMP_MAGIC.len()).unwrap().fill(0);
        assert_eq!(Dump::from_bytes(&bytes), Err(DumpError::BadMagic));
    }

    /// One event of every kind, with every field a distinct non-zero value, so that swapping any
    /// two fields (or two variants) changes the bytes.
    fn every_kind() -> Dump {
        let kinds = [
            EventKind::Svc {
                id: 0x0102_0304,
                args: core::array::from_fn(|i| 0x1000 + u64::try_from(i).unwrap() * 0x111),
                result: ResultCode(0x0000_e401),
            },
            EventKind::Ipc {
                session: SessionId(0x21),
                service: ServiceId(0x22),
                command: 0x23,
                request_size: 0x24,
                reply_size: 0x25,
                result: ResultCode(0x26),
                outcome: IpcOutcome::Deferred,
            },
            EventKind::Wake {
                thread: ThreadId(0x31),
                reason: WakeReason::TimedOut,
            },
            EventKind::Schedule {
                core: 0x41,
                from: Some(ThreadId(0x42)),
                to: Some(ThreadId(0x43)),
            },
            EventKind::Map {
                va: 0x5000_0000,
                size: 0x5100,
                perms: 0x52,
                kind: MapKind::Protect,
            },
            EventKind::Fault {
                address: 0x6000_0000,
                write: true,
                action: FaultAction::Delivered,
            },
            EventKind::ServiceNote {
                service: ServiceId(0x71),
                code: 0x72,
                value: 0x73,
            },
            EventKind::GuestLog {
                len: 0x81,
                hash: 0x8283_8485,
            },
            EventKind::Lifecycle {
                phase: LifecyclePhase::ThreadExit,
                subject: 0x91,
            },
        ];
        let events = kinds
            .into_iter()
            .zip(1u64..)
            .map(|(kind, n)| Event {
                seq: 0x10 + n,
                time: GuestTicks(0x2000 + n * 0x100),
                core: 0x30 + u8::try_from(n).unwrap(),
                thread: ThreadId(0x400 + n),
                kind,
            })
            .collect();
        Dump::new(vec![CoreDump { core: 5, events }])
    }

    /// The bytes of [`every_kind`], after the header. **This pins the on-disk layout of
    /// `events.bin`.** If it fails, the layout changed: bump [`DUMP_VERSION`] in the same change
    /// and regenerate this constant, so that old dumps are refused instead of misread.
    const EVERY_KIND_BODY_HEX: &str = concat!(
        "01", // one core
        "05", // core 5
        "09", // nine events
        // seq, time, core, thread, then the kind's variant index and fields.
        "11",
        "8042",
        "31",
        "8108",
        "00",
        "84868808",
        "8020",
        "9122",
        "a224",
        "b326",
        "c428",
        "d52a",
        "e62c",
        "f72e",
        "81c803",
        "12",
        "8044",
        "32",
        "8208",
        "01",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "01",
        "13",
        "8046",
        "33",
        "8308",
        "02",
        "31",
        "01",
        "14",
        "8048",
        "34",
        "8408",
        "03",
        "41",
        "01",
        "42",
        "01",
        "43",
        "15",
        "804a",
        "35",
        "8508",
        "04",
        "8080808005",
        "80a201",
        "52",
        "02",
        "16",
        "804c",
        "36",
        "8608",
        "05",
        "8080808006",
        "01",
        "01",
        "17",
        "804e",
        "37",
        "8708",
        "06",
        "71",
        "72",
        "73",
        "18",
        "8050",
        "38",
        "8808",
        "07",
        "8101",
        "85898e9408",
        "19",
        "8052",
        "39",
        "8908",
        "08",
        "03",
        "9101",
    );

    fn hex(bytes: &[u8]) -> String {
        use core::fmt::Write as _;
        bytes.iter().fold(String::new(), |mut out, b| {
            write!(out, "{b:02x}").unwrap();
            out
        })
    }

    #[test]
    fn the_byte_layout_is_pinned() {
        let bytes = every_kind().to_bytes().unwrap();
        let header = postcard::to_allocvec(&(DUMP_MAGIC, DUMP_VERSION)).unwrap();
        assert_eq!(bytes.get(..header.len()), Some(header.as_slice()));
        let body = bytes.get(header.len()..).unwrap();
        assert_eq!(hex(body), EVERY_KIND_BODY_HEX);
        assert_eq!(Dump::from_bytes(&bytes).unwrap(), every_kind());
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
