//! Snapshots: saving a server's sessions, domains and requests in flight, and restoring them
//! (`docs/specs/observability.md`, "Snapshots").
//!
//! The rule the spec sets is that every session keeps its state in a plain serialisable struct.
//! The framework enforces it where a service is declared: [`commands!`](crate::commands)
//! implements [`Session::save`] with `serde`, so a session struct that holds something that
//! cannot be saved does not compile.
//!
//! [`Server::save`] writes, and [`Server::restore`] reads back:
//!
//! - the server's sessions and what each leads to;
//! - its table of interfaces and domains, with every interface's state and every domain's object
//!   ids, row for row, so ids handed out after a restore are the ones the saved run would have
//!   handed out;
//! - its request queue: which sessions have a request in flight, and how many senders wait;
//! - its deferred requests, by [`TokenId`]. The tokens themselves are host objects and are not
//!   saved; [`Server::reissue`] makes a new one for a saved id;
//! - its family state ([`Family`]), once.
//!
//! Not saved: the record of auto-stubbed commands ([`Server::stubbed`]), which is a log of the
//! run and not state a guest can observe; and a completion that was posted but not yet
//! delivered, whose request is saved as still deferred.
//!
//! # Format
//!
//! `postcard`. A header ([`SNAPSHOT_MAGIC`], [`SNAPSHOT_VERSION`]) is decoded and checked before
//! anything after it, so bytes of another kind or another layout are refused, not misread. The
//! layout is this crate's own design (the spec asks only that it be versioned) and is pinned by
//! the golden-bytes test `the_byte_layout_is_pinned` in `framework/tests/snapshot.rs`. Changing
//! it, or the fields of anything it holds, means bumping [`SNAPSHOT_VERSION`] in the same
//! change. A session's own state is opaque bytes inside it: changing a session struct changes
//! what its bytes mean without changing this layout, so a service whose state changes says so
//! in its own version field or the snapshot is refused by its decoder.
//!
//! # A snapshot is untrusted
//!
//! A snapshot is a file, so [`Server::restore`] checks everything in it against everything else
//! before it changes the server: every id names a row that exists and is of the right kind,
//! every reference count equals the number of references found, and the queue, the deferred
//! requests and the sessions agree. A snapshot that fails any check is an error and the server
//! is left as it was.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use core::any::{Any, TypeId};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use starquake_kernel::id::ObjectId;
use starquake_kernel::slab::Slab;

use super::commands::Session;
use super::defer::{Mailbox, Pending, TokenId};
use super::queue::RequestQueue;
use super::server::Server;
use super::targets::{Row, TargetId, Targets};

/// First bytes of every server snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SQSV";

/// Layout version written into every server snapshot and required when reading one.
pub const SNAPSHOT_VERSION: u32 = 1;

/// Why a snapshot could not be written or read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SnapshotError {
    /// Encoding failed.
    #[error("could not encode the snapshot: {0}")]
    Encode(postcard::Error),
    /// The bytes are not a well-formed encoding: truncated, a field out of range, or a table
    /// whose own invariants do not hold.
    #[error("could not decode the snapshot: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`SNAPSHOT_MAGIC`].
    #[error("not a server snapshot (bad magic)")]
    BadMagic,
    /// The snapshot was written with a different layout version.
    #[error(
        "snapshot layout version {found} is not supported (this build reads {SNAPSHOT_VERSION})"
    )]
    UnsupportedVersion {
        /// The version found.
        found: u32,
    },
    /// Bytes remain after the snapshot.
    #[error("{count} trailing bytes after the snapshot")]
    TrailingBytes {
        /// How many bytes were left over.
        count: usize,
    },
    /// The snapshot is of a different server.
    #[error("the snapshot is of server {found:?}, not {expected:?}")]
    WrongServer {
        /// The server being restored.
        expected: String,
        /// The server the snapshot names.
        found: String,
    },
    /// A session is of an interface the server's [`Interfaces`] do not list, so it could not be
    /// restored.
    #[error("interface {0:?} is not one the server can restore")]
    UnknownInterface(String),
    /// A session is of an interface whose name two different types were added under
    /// ([`Interfaces::with`]), so its saved state could be either's.
    #[error("two interface types are named {0:?}")]
    DuplicateInterface(&'static str),
    /// A session's saved state could not be decoded as its interface.
    #[error("the saved state of a {interface} could not be decoded: {error}")]
    State {
        /// The interface.
        interface: String,
        /// What the decoder said.
        error: postcard::Error,
    },
    /// The server has family state and the snapshot has none, or the reverse.
    #[error("the snapshot and the server disagree on whether there is family state")]
    FamilyMismatch,
    /// The parts of the snapshot disagree with each other.
    #[error("inconsistent snapshot: {0}")]
    Inconsistent(&'static str),
}

/// Encodes a session's or a family's state. [`commands!`](crate::commands) calls it; a service
/// has no need to.
pub fn encode<T: Serialize + ?Sized>(state: &T) -> Result<Vec<u8>, SnapshotError> {
    postcard::to_allocvec(state).map_err(SnapshotError::Encode)
}

/// Decodes a whole value: bytes left over are an error, since they mean the bytes were written
/// as some other type.
fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, postcard::Error> {
    let (value, rest) = postcard::take_from_bytes(bytes)?;
    if rest.is_empty() {
        Ok(value)
    } else {
        Err(postcard::Error::DeserializeBadEncoding)
    }
}

/// Turns saved bytes back into a session.
type Thaw = fn(&[u8]) -> Result<Box<dyn Session>, postcard::Error>;

fn thaw<T: Session + DeserializeOwned>(bytes: &[u8]) -> Result<Box<dyn Session>, postcard::Error> {
    Ok(Box::new(decode::<T>(bytes)?))
}

/// The interface types a server's sessions can be, by name: what a restore needs to turn saved
/// state back into sessions. Give it to the server with
/// [`Server::with_interfaces`](super::Server::with_interfaces).
///
/// ```
/// use starquake_services::commands;
/// use starquake_services::framework::{Interfaces, SnapshotError};
///
/// #[derive(serde::Serialize, serde::Deserialize)]
/// struct Volume {
///     level: u32,
/// }
///
/// commands! {
///     impl Volume {}
/// }
///
/// // Adding a type again changes nothing.
/// let interfaces = Interfaces::new().with::<Volume>().with::<Volume>();
/// assert_eq!(interfaces.check("Volume"), Ok(()));
/// assert_eq!(
///     interfaces.check("Mixer"),
///     Err(SnapshotError::UnknownInterface("Mixer".to_owned()))
/// );
/// ```
///
/// Two different types with one name cannot be told apart in a snapshot, where a session is
/// its interface's name and its state's bytes. Adding both is not refused at once (so that a
/// server is put together without a fallible step), but the name is then one no snapshot can be
/// written or read with: [`Interfaces::check`] reports
/// [`SnapshotError::DuplicateInterface`], and so do [`Server::save`] and [`Server::restore`]
/// when a session of that name is in the way.
#[derive(Clone, Default)]
pub struct Interfaces {
    by_name: BTreeMap<&'static str, (TypeId, Thaw)>,
    /// Names two different types were added under.
    ambiguous: BTreeSet<&'static str>,
}

impl Interfaces {
    /// No interfaces.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds the interface `T`, under [`Session::interface`].
    #[must_use]
    pub fn with<T: Session + DeserializeOwned>(mut self) -> Self {
        let (name, type_id) = (T::interface(), TypeId::of::<T>());
        let thaw: Thaw = thaw::<T>;
        let earlier = self.by_name.insert(name, (type_id, thaw));
        if earlier.is_some_and(|(earlier, _)| earlier != type_id) {
            self.ambiguous.insert(name);
        }
        self
    }

    /// Whether a session of the interface `name` can be saved and restored: exactly one type
    /// was added under the name.
    pub fn check(&self, name: &str) -> Result<(), SnapshotError> {
        match self.by_name.get_key_value(name) {
            Some((name, _)) if self.ambiguous.contains(name) => {
                Err(SnapshotError::DuplicateInterface(name))
            }
            Some(_) => Ok(()),
            None => Err(SnapshotError::UnknownInterface(name.to_owned())),
        }
    }

    /// The session a saved state of the interface `name` holds.
    pub(super) fn thaw(&self, name: &str, state: &[u8]) -> Result<Box<dyn Session>, SnapshotError> {
        self.check(name)?;
        let unknown = || SnapshotError::UnknownInterface(name.to_owned());
        let (_, thaw) = self.by_name.get(name).ok_or_else(unknown)?;
        thaw(state).map_err(|error| SnapshotError::State {
            interface: name.to_owned(),
            error,
        })
    }
}

/// A service family's own state: what all of a server's sessions share
/// (`docs/specs/services.md`, "Concurrency": "a server is one service family's state plus the
/// sessions opened on it"). A method reaches it with
/// [`Context::family`](super::Context::family).
///
/// Implemented for every type that is `serde` data, so a family's state is saved once in its
/// server's snapshot and loaded back in place; nothing implements it by hand.
pub trait Family: Any + Send {
    /// The state as bytes, for a snapshot.
    fn save_state(&self) -> Result<Vec<u8>, SnapshotError>;

    /// Replaces the state with what `bytes` hold. On an error the state is unchanged.
    fn load_state(&mut self, bytes: &[u8]) -> Result<(), SnapshotError>;
}

impl<T: Serialize + DeserializeOwned + Any + Send> Family for T {
    fn save_state(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(self)
    }

    fn load_state(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        *self = decode(bytes).map_err(SnapshotError::Decode)?;
        Ok(())
    }
}

/// What every snapshot starts with.
#[derive(Serialize, Deserialize)]
struct Header {
    magic: [u8; 4],
    version: u32,
}

/// A server's state as it is written. The field order is the layout.
#[derive(Serialize)]
struct BodyRef<'a> {
    name: &'a str,
    sessions: &'a BTreeMap<ObjectId, TargetId>,
    targets: &'a Slab<TargetId, Row>,
    queue: &'a RequestQueue,
    pending: &'a BTreeMap<TokenId, Pending>,
    next_token: u64,
    family: Option<Vec<u8>>,
}

/// A server's state as it is read: [`BodyRef`], owned.
#[derive(Deserialize)]
struct Body {
    name: String,
    sessions: BTreeMap<ObjectId, TargetId>,
    targets: Slab<TargetId, Row>,
    queue: RequestQueue,
    pending: BTreeMap<TokenId, Pending>,
    next_token: u64,
    family: Option<Vec<u8>>,
}

impl Server {
    /// The server's state as bytes. See the [module docs](self) for what is in them.
    ///
    /// Fails with [`SnapshotError::UnknownInterface`] if a session is of an interface the
    /// server's [`Interfaces`] do not list (or [`SnapshotError::DuplicateInterface`] if they
    /// list it twice), so that no snapshot is written that could not be restored.
    pub fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        self.targets.check_restorable(&self.interfaces)?;
        let family = self.family.as_deref().map(Family::save_state).transpose()?;
        let header = Header {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
        };
        let body = BodyRef {
            name: &self.name,
            sessions: &self.sessions,
            targets: self.targets.rows(),
            queue: &self.queue,
            pending: &self.pending,
            next_token: self.next_token,
            family,
        };
        let bytes = postcard::to_allocvec(&header).map_err(SnapshotError::Encode)?;
        postcard::to_extend(&body, bytes).map_err(SnapshotError::Encode)
    }

    /// Replaces the server's sessions, domains, queue, deferred requests and family state with
    /// those a [`Server::save`] of a server of the same name wrote. On an error the server is
    /// unchanged.
    ///
    /// Reply tokens taken before the restore no longer lead to this server: completing one does
    /// nothing. Deferred requests in the snapshot are resumed with [`Server::reissue`].
    pub fn restore(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        let (header, rest) =
            postcard::take_from_bytes::<Header>(bytes).map_err(SnapshotError::Decode)?;
        if header.magic != SNAPSHOT_MAGIC {
            return Err(SnapshotError::BadMagic);
        }
        if header.version != SNAPSHOT_VERSION {
            return Err(SnapshotError::UnsupportedVersion {
                found: header.version,
            });
        }
        let (body, rest) =
            postcard::take_from_bytes::<Body>(rest).map_err(SnapshotError::Decode)?;
        if !rest.is_empty() {
            return Err(SnapshotError::TrailingBytes { count: rest.len() });
        }
        if body.name != self.name {
            return Err(SnapshotError::WrongServer {
                expected: self.name.clone(),
                found: body.name,
            });
        }
        let targets = Targets::thaw(body.targets, &self.interfaces, body.sessions.values())?;
        check_requests(&body.sessions, &body.queue, &body.pending, body.next_token)?;
        // The family is loaded last: it is the one step that changes the server and can fail.
        match (self.family.as_deref_mut(), body.family) {
            (Some(family), Some(saved)) => family.load_state(&saved)?,
            (None, None) => {}
            _ => return Err(SnapshotError::FamilyMismatch),
        }
        self.sessions = body.sessions;
        self.targets = targets;
        self.queue = body.queue;
        self.pending = body.pending;
        self.next_token = body.next_token;
        // Tokens of the run that is being replaced still lead to the old mailbox.
        self.mailbox.close();
        self.mailbox = Arc::new(Mailbox::new(self.notify.clone()));
        Ok(())
    }
}

/// Checks that the queue and the deferred requests name sessions that exist and agree with each
/// other: a session's request is deferred in the queue exactly when one token is pending for it.
fn check_requests(
    sessions: &BTreeMap<ObjectId, TargetId>,
    queue: &RequestQueue,
    pending: &BTreeMap<TokenId, Pending>,
    next_token: u64,
) -> Result<(), SnapshotError> {
    if queue.sessions().any(|s| !sessions.contains_key(&s)) {
        return Err(SnapshotError::Inconsistent(
            "the queue names a session that is not open",
        ));
    }
    let mut deferred = BTreeSet::new();
    for (token, request) in pending {
        if token.raw() >= next_token {
            return Err(SnapshotError::Inconsistent(
                "a pending token was never handed out",
            ));
        }
        if !queue.is_deferred(request.session) {
            return Err(SnapshotError::Inconsistent(
                "a token is pending for a session with no deferred request",
            ));
        }
        if !deferred.insert(request.session) {
            return Err(SnapshotError::Inconsistent(
                "two tokens are pending for one session",
            ));
        }
    }
    if queue.deferred().any(|s| !deferred.contains(&s)) {
        return Err(SnapshotError::Inconsistent(
            "a deferred request has no pending token",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct Settings {
        volume: u8,
        name: String,
    }

    #[test]
    fn a_family_loads_what_it_saved_and_refuses_anything_else() {
        let saved = Settings {
            volume: 7,
            name: "console".to_owned(),
        };
        let bytes = Family::save_state(&saved).unwrap();
        // Golden bytes: a `u8`, then a length-prefixed string.
        assert_eq!(bytes, [7, 7, b'c', b'o', b'n', b's', b'o', b'l', b'e']);
        let mut family = Settings {
            volume: 0,
            name: String::new(),
        };
        family.load_state(&bytes).unwrap();
        assert_eq!(family, saved);

        // Truncated, and one byte too many: an error, and the state is as it was.
        let truncated = family.load_state(&bytes[..bytes.len() - 1]);
        assert_eq!(
            truncated,
            Err(SnapshotError::Decode(
                postcard::Error::DeserializeUnexpectedEnd
            ))
        );
        let mut long = bytes.clone();
        long.push(0);
        assert_eq!(
            family.load_state(&long),
            Err(SnapshotError::Decode(
                postcard::Error::DeserializeBadEncoding
            ))
        );
        assert_eq!(family, saved);
    }

    #[test]
    fn error_messages_are_stable() {
        assert_eq!(
            SnapshotError::UnsupportedVersion { found: 9 }.to_string(),
            format!(
                "snapshot layout version 9 is not supported (this build reads {SNAPSHOT_VERSION})"
            )
        );
        assert_eq!(
            SnapshotError::WrongServer {
                expected: "sm:".to_owned(),
                found: "set:sys".to_owned()
            }
            .to_string(),
            "the snapshot is of server \"set:sys\", not \"sm:\""
        );
        assert_eq!(
            SnapshotError::Inconsistent("a deferred request has no pending token").to_string(),
            "inconsistent snapshot: a deferred request has no pending token"
        );
    }
}
