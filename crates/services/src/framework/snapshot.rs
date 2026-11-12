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
//! run and not state a guest can observe; and a completion that was posted and that
//! [`Server::take_completed`] has not taken yet, whose request is saved as still deferred (a
//! server restored from the snapshot has nothing to deliver for it, and the work is resumed
//! with [`Server::reissue`]).
//!
//! Between [`Server::take_completed`] and the [`Server::deliver`] of what it returned, the
//! request is neither pending nor answered, and the completion is in the caller's hands. No
//! snapshot holds that state: [`Server::save`] fails with [`SnapshotError::Inconsistent`]
//! until the completion is delivered
//! (`a_server_is_not_saved_between_taking_a_completion_and_delivering_it` in
//! `framework/tests/snapshot.rs`).
//!
//! # Format
//!
//! `postcard`. A header ([`SNAPSHOT_MAGIC`], [`SNAPSHOT_VERSION`]) is decoded and checked before
//! anything after it, so bytes of another kind or another layout are refused, not misread. The
//! layout is this crate's own design (the spec asks only that it be versioned) and is pinned by
//! the golden-bytes test `the_byte_layout_is_pinned` in `framework/tests/snapshot.rs`. Changing
//! it, or the fields of anything it holds, means bumping [`SNAPSHOT_VERSION`] in the same
//! change.
//!
//! The version covers the framework's own layout and nothing more. A session's state and a
//! family's state are opaque bytes inside it, and `postcard` does not describe what it holds:
//! when a service's state struct changes shape, bytes an older build wrote can fail to decode
//! ([`SnapshotError::State`]) or, if they happen to fit, be read as the wrong values. So a
//! server snapshot is only good for the build that wrote it. The spec's rule that a snapshot
//! from another build is refused belongs to the snapshot of a whole run, which will hold this
//! one and can carry the build's identity; that format does not exist yet.
//!
//! # A snapshot is untrusted
//!
//! A snapshot is a file, so [`Server::restore`] believes nothing in it. Before it changes the
//! server it checks these, and a snapshot that fails one is an error that leaves the server
//! as it was:
//!
//! - The header, then that the body decodes with no bytes left over and names this server.
//!   Decoding the queue and the table's slab checks each one's own invariants
//!   (`QueueLayoutError` in `framework/queue.rs`, `SlabLayoutError` in the kernel crate's
//!   `slab.rs`).
//! - The table of interfaces and domains (`Targets::thaw` in `framework/targets.rs`): every
//!   session leads to a row that exists; every interface is of a type the server's
//!   [`Interfaces`] list once, and its state decodes as that type; every domain's table of
//!   object ids is one a domain can have built, and its objects are interfaces; every row's
//!   reference count is at least one and is the number of sessions and domain entries that
//!   name it.
//! - The requests: the queue has as many workers as the server; every session the queue names
//!   is open; every pending token is below the next token id, and is for a session whose
//!   request the queue marks deferred; every deferred session has exactly one pending token; a
//!   deferred request's reply has a domain header exactly when its session leads to a domain.
//! - The family state: present exactly when the server has some, and decodes as its type.
//!
//! Not checked, because the snapshot holds nothing to check them against: the values inside a
//! session's or a family's state (see "Format"), a deferred request's command id and argument
//! size (they only go into its trace event), and whether a deferred request on a session that
//! is not a domain was CMIF or TIPC.
//!
//! [`Server::save`] runs the same checks of the requests, the worker count aside, so a
//! server in a state that could not be restored is not saved.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use core::any::{Any, TypeId};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use starquake_kernel::id::ObjectId;
use starquake_kernel::slab::Slab;

use super::commands::Session;
use super::defer::{Framing, Mailbox, Pending, TokenId};
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
    /// list it twice), and with [`SnapshotError::Inconsistent`] if an interface is out of its
    /// row or the queue, the deferred requests and the sessions disagree, which they do
    /// between [`Server::take_completed`] and [`Server::deliver`]. These are the checks
    /// [`Server::restore`] makes of the same things, so that a snapshot it would refuse for
    /// one of them is not written.
    pub fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        self.targets.check_restorable(&self.interfaces)?;
        let requests = Requests {
            sessions: &self.sessions,
            targets: &self.targets,
            queue: &self.queue,
            pending: &self.pending,
            next_token: self.next_token,
        };
        requests.check()?;
        self.encode_state()
    }

    /// The server's state as bytes, unchecked: what [`Server::save`] writes once its checks
    /// have passed.
    pub(super) fn encode_state(&self) -> Result<Vec<u8>, SnapshotError> {
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
        if body.queue.workers() != self.queue.workers() {
            return Err(SnapshotError::Inconsistent(
                "the queue's count of workers is not the server's",
            ));
        }
        let requests = Requests {
            sessions: &body.sessions,
            targets: &targets,
            queue: &body.queue,
            pending: &body.pending,
            next_token: body.next_token,
        };
        requests.check()?;
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

/// A server's requests in flight and what they are checked against, as [`Server::save`] finds
/// them in the server and [`Server::restore`] in a snapshot.
struct Requests<'a> {
    sessions: &'a BTreeMap<ObjectId, TargetId>,
    targets: &'a Targets,
    queue: &'a RequestQueue,
    pending: &'a BTreeMap<TokenId, Pending>,
    next_token: u64,
}

impl Requests<'_> {
    /// Checks that the queue and the deferred requests name sessions that exist and agree with
    /// each other: a session's request is deferred in the queue exactly when one token is
    /// pending for it, and its reply is framed as the session is.
    fn check(&self) -> Result<(), SnapshotError> {
        let Self {
            sessions, queue, ..
        } = *self;
        if queue.sessions().any(|s| !sessions.contains_key(&s)) {
            return Err(SnapshotError::Inconsistent(
                "the queue names a session that is not open",
            ));
        }
        let mut deferred = BTreeSet::new();
        for (token, request) in self.pending {
            if token.raw() >= self.next_token {
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
            self.check_framing(request)?;
        }
        if queue.deferred().any(|s| !deferred.contains(&s)) {
            return Err(SnapshotError::Inconsistent(
                "a deferred request has no pending token",
            ));
        }
        Ok(())
    }

    /// Checks a deferred request's framing against its session: a reply has a domain header
    /// exactly when the session leads to a domain. This is what [`Server::handle`] keeps: a
    /// request to a domain's object has a domain header and so does its reply, TIPC is not
    /// served on a domain, and a session with a request in flight is not converted.
    fn check_framing(&self, request: &Pending) -> Result<(), SnapshotError> {
        // The queue marks the session's request deferred, and the queue's sessions are open.
        let target = self.sessions.get(&request.session);
        let target = target.ok_or(SnapshotError::Inconsistent(
            "the queue names a session that is not open",
        ))?;
        // `Targets::thaw` and a live server both hold a row for every session's target.
        let is_domain = self.targets.is_domain(*target);
        let is_domain = is_domain.map_err(|_| {
            SnapshotError::Inconsistent("a session leads to a target that does not exist")
        })?;
        let mismatch = match (request.framing, is_domain) {
            (Framing::Domain, true) | (Framing::Cmif | Framing::Tipc(_), false) => return Ok(()),
            (Framing::Domain, false) => {
                "a deferred request has a domain reply on a session that is not a domain"
            }
            (Framing::Cmif, true) => {
                "a deferred request on a domain session has a reply that is not a domain's"
            }
            (Framing::Tipc(_), true) => "a deferred TIPC request is on a domain session",
        };
        Err(SnapshotError::Inconsistent(mismatch))
    }
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
