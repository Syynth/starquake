//! The `SubmitGPFIFO` requests that wait for room in a full GPU queue, in gsv's shared state,
//! and their snapshot (`docs/specs/graphics-server.md`, decision 12: "a full queue makes the
//! submit wait as a deferred reply"; "Waiting without blocking a core": completed by "GPU front,
//! when it has room").
//!
//! No reference source is ported: neither reference bounds a channel's queue, so neither waits.
//! The design is the spec's.
//!
//! # The table
//!
//! [`Submits`] holds, per GPU channel, the submissions that wait, oldest first, each with the
//! reply token of its request and the front it goes to. It lives in [`crate::shared::GsvShared`]
//! because both nvsrv's handlers (which add to it) and the timer thread's drain (which empties
//! it) reach it, under gsv's shared lock.
//!
//! - [`Submits::push`] adds a submission behind the channel's others, then tries the channel's
//!   queue ([`Submits::room`]), so a room report that came between the front's `Full` and the
//!   push is not lost.
//! - [`Submits::room`] (the drain, on `GpuQueueRoom`) gives the channel's waiting submissions to
//!   the front, oldest first, completing the reply of each one queued with `Success`, and stops
//!   at the first `Full`: the front reports room again later.
//! - [`Submits::cancel`] (the channel's fd closed) answers each waiting reply `NotImplemented`,
//!   oldest first, without giving the front anything.
//!
//! How many wait is bounded by the requests in flight: a waiting submission keeps its request's
//! session busy (the service framework handles one request at a time per session), so a title
//! has at most one waiting per session it opened.
//!
//! A completion posts the reply to the server's mailbox (`ReplyToken::complete`), which tells the
//! kernel; it never blocks, and the kernel delivers the reply on a core thread. The drain calls
//! it with gsv's lock held: gsv's lock, then whatever the kernel's notify takes, the documented
//! order (`crate::shared`, "Lock order and the drain").
//!
//! # Snapshot format
//!
//! [`Submits::save`] writes the table: `postcard`, after a header of [`SNAPSHOT_MAGIC`] and
//! [`SNAPSHOT_VERSION`] that is decoded and checked before the body. The body is a list, in
//! channel order and oldest first within a channel, of each waiting submission: its channel, its
//! request's token id, its entries, its wait fence (syncpoint id and value) if any, its
//! increments and its wait-for-idle flag. The golden-bytes test `the_snapshot_layout_is_pinned`
//! (in `nvsrv/gpu/tests.rs`, with the other tests of this module, which need the server's reply
//! tokens) pins it.
//!
//! A snapshot is a file and is not trusted. [`Submits::restore`] checks, in this order, and
//! returns the first failure: the magic, the version, that the body decodes, that no bytes follow
//! it, then each entry in list order ([`SnapshotError`], in the order of its variants): its wait
//! fence's syncpoint, its entry count, that its token id is not one an earlier entry has, that it
//! comes after the entry before it (channel, then token id: a later request takes a later token),
//! and that the server holds its request as pending ([`Server::pending`]). Only once every entry
//! has passed are the requests given new tokens ([`Server::reissue`]) and the table built, so a
//! refused snapshot leaves every pending request as it was. The restored table gives nothing to the
//! front: [`Submits::retry_all`] does, once the front's channels exist again (the wiring slice,
//! 16, makes them).
//!
//! Not checked here: that each channel is one the restored nvsrv state made (the two snapshots
//! are composed by the wiring slice).
//!
//! [`Server::reissue`]: starquake_services::framework::Server::reissue
//! [`Server::pending`]: starquake_services::framework::Server::pending

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::{ChannelId, Fence, SyncpointId};
use starquake_services::framework::{Reply, ReplyToken, Server, TokenId};
use thiserror::Error;

use super::{CHANNEL_QUEUE_ENTRIES, ChannelFront, GpfifoEntry, Submission, Submitted};
use crate::nvsrv::NvResult;

/// First bytes of every waiting-submissions snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SQSB";

/// Layout version written into every waiting-submissions snapshot and required when reading one.
pub const SNAPSHOT_VERSION: u32 = 1;

/// One waiting submission.
struct Waiting {
    token: ReplyToken,
    submission: Submission,
    front: Arc<dyn ChannelFront>,
}

/// The submissions that wait for room, by channel, oldest first (module docs, "The table").
#[derive(Default)]
pub struct Submits {
    waiting: BTreeMap<ChannelId, VecDeque<Waiting>>,
}

impl fmt::Debug for Submits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.saved()).finish()
    }
}

/// Two tables are equal when they hold the same submissions for the same requests, in the same
/// order; the fronts are not compared.
impl PartialEq for Submits {
    fn eq(&self, other: &Self) -> bool {
        self.saved() == other.saved()
    }
}

impl Eq for Submits {}

/// The reply a waiting request gets: the IPC result success and the `NvResult` word
/// (`crate::nvsrv`, "The protocol").
fn reply(result: NvResult) -> Reply {
    let mut reply = Reply::default();
    reply.payload().write_u32(result.raw());
    reply
}

impl Submits {
    /// Nothing waits.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether submissions of `channel` wait.
    #[must_use]
    pub fn is_waiting(&self, channel: ChannelId) -> bool {
        self.waiting.contains_key(&channel)
    }

    /// How many submissions of `channel` wait.
    #[must_use]
    pub fn waiting(&self, channel: ChannelId) -> usize {
        self.waiting.get(&channel).map_or(0, VecDeque::len)
    }

    /// How many submissions wait, of every channel.
    #[must_use]
    pub fn len(&self) -> usize {
        self.waiting.values().map(VecDeque::len).sum()
    }

    /// Whether nothing waits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }

    /// The token ids of `channel`'s waiting requests, oldest first.
    pub fn tokens(&self, channel: ChannelId) -> impl Iterator<Item = TokenId> + '_ {
        self.waiting
            .get(&channel)
            .into_iter()
            .flatten()
            .map(|waiting| waiting.token.id())
    }

    /// Adds `submission`, whose request `token` answers, behind `channel`'s waiting ones, then
    /// tries the channel's queue (module docs, "The table").
    pub fn push(
        &mut self,
        channel: ChannelId,
        token: ReplyToken,
        submission: Submission,
        front: Arc<dyn ChannelFront>,
    ) {
        self.waiting.entry(channel).or_default().push_back(Waiting {
            token,
            submission,
            front,
        });
        self.room(channel);
    }

    /// `channel`'s queue has room: its waiting submissions go to the front, oldest first, until
    /// it answers `Full`; each one queued has its reply completed (module docs, "The table").
    pub fn room(&mut self, channel: ChannelId) {
        let Some(queue) = self.waiting.get_mut(&channel) else {
            return;
        };
        while let Some(head) = queue.front() {
            if head.front.submit(channel, head.submission.clone()) == Submitted::Full {
                break;
            }
            if let Some(queued) = queue.pop_front() {
                queued.token.complete(reply(NvResult::Success));
            }
        }
        if queue.is_empty() {
            self.waiting.remove(&channel);
        }
    }

    /// Tries every channel's queue, in channel order: after a restore, once the front's channels
    /// exist again (module docs, "Snapshot format").
    pub fn retry_all(&mut self) {
        let channels: Vec<ChannelId> = self.waiting.keys().copied().collect();
        for channel in channels {
            self.room(channel);
        }
    }

    /// `channel` is going: each waiting reply answers `NotImplemented`, oldest first, and nothing
    /// goes to the front (module docs, "The table").
    pub fn cancel(&mut self, channel: ChannelId) {
        for waiting in self.waiting.remove(&channel).into_iter().flatten() {
            waiting.token.complete(reply(NvResult::NotImplemented));
        }
    }

    /// The table as it is saved, in order.
    fn saved(&self) -> Vec<SavedSubmit> {
        self.waiting
            .iter()
            .flat_map(|(&channel, queue)| {
                queue.iter().map(move |waiting| SavedSubmit {
                    channel: channel.0,
                    token: waiting.token.id(),
                    entries: waiting.submission.entries.iter().map(|entry| entry.0).collect(),
                    wait: waiting
                        .submission
                        .wait
                        .map(|fence| (fence.id.get(), fence.value)),
                    increments: waiting.submission.increments,
                    wfi: waiting.submission.wfi,
                })
            })
            .collect()
    }

    /// Writes the table (module docs, "Snapshot format"). Take it while nothing changes it.
    pub fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let mut bytes = postcard::to_allocvec(&Header {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
        })
        .map_err(SnapshotError::Encode)?;
        bytes.extend(postcard::to_allocvec(&self.saved()).map_err(SnapshotError::Encode)?);
        Ok(bytes)
    }

    /// The table [`save`](Self::save) wrote, checked first (module docs, "Snapshot format"):
    /// each request must be pending on `server` (the restored nvsrv server), which gives it a new
    /// token, and every submission goes to `front`. Nothing is given to the front yet.
    pub fn restore(
        bytes: &[u8],
        server: &Server,
        front: &Arc<dyn ChannelFront>,
    ) -> Result<Self, SnapshotError> {
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
        let (saved, rest) =
            postcard::take_from_bytes::<Vec<SavedSubmit>>(rest).map_err(SnapshotError::Decode)?;
        if !rest.is_empty() {
            return Err(SnapshotError::TrailingBytes { count: rest.len() });
        }
        let mut seen = BTreeSet::new();
        let mut previous: Option<(u32, TokenId)> = None;
        let mut checked = Vec::with_capacity(saved.len());
        for (index, entry) in saved.into_iter().enumerate() {
            let wait = match entry.wait {
                None => None,
                Some((id, value)) => Some(Fence {
                    id: SyncpointId::new(id).ok_or(SnapshotError::WaitSyncpoint { index, id })?,
                    value,
                }),
            };
            if entry.entries.len() > CHANNEL_QUEUE_ENTRIES {
                let count = entry.entries.len();
                return Err(SnapshotError::TooManyEntries { index, count });
            }
            if !seen.insert(entry.token) {
                let token = entry.token.raw();
                return Err(SnapshotError::DuplicateToken { index, token });
            }
            let key = (entry.channel, entry.token);
            // An equal key has a token an earlier entry has, refused just above, so `>=` and
            // `>` refuse the same snapshots here.
            if previous.is_some_and(|previous| previous >= key) {
                return Err(SnapshotError::Misordered { index });
            }
            previous = Some(key);
            if !server.pending().any(|(pending, _)| pending == entry.token) {
                let token = entry.token.raw();
                return Err(SnapshotError::NotPending { index, token });
            }
            let submission = Submission {
                entries: entry.entries.into_iter().map(GpfifoEntry).collect(),
                wait,
                increments: entry.increments,
                wfi: entry.wfi,
            };
            checked.push((ChannelId(entry.channel), entry.token, submission));
        }
        let mut submits = Self::new();
        for (channel, id, submission) in checked {
            // Every id was checked pending above, and nothing ran on the server since.
            let Some(token) = server.reissue(id) else {
                continue;
            };
            submits.waiting.entry(channel).or_default().push_back(Waiting {
                token,
                submission,
                front: Arc::clone(front),
            });
        }
        Ok(submits)
    }
}

/// What every waiting-submissions snapshot starts with.
#[derive(Serialize, Deserialize)]
struct Header {
    /// [`SNAPSHOT_MAGIC`].
    magic: [u8; 4],
    /// [`SNAPSHOT_VERSION`].
    version: u32,
}

/// One waiting submission as it is saved. The field order is the layout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct SavedSubmit {
    /// The GPU channel.
    pub(super) channel: u32,
    /// The request's token id.
    pub(super) token: TokenId,
    /// The entries, raw.
    pub(super) entries: Vec<u64>,
    /// The wait fence: syncpoint id and value.
    pub(super) wait: Option<(u8, u32)>,
    /// The increments after the entries.
    pub(super) increments: u32,
    /// Whether a wait-for-idle comes before them.
    pub(super) wfi: bool,
}

/// Why a waiting-submissions snapshot could not be written or read.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SnapshotError {
    /// Encoding failed.
    #[error("could not encode the waiting submissions snapshot: {0}")]
    Encode(postcard::Error),
    /// The header or the body is not a well-formed encoding.
    #[error("could not decode the waiting submissions snapshot: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`SNAPSHOT_MAGIC`].
    #[error("not a waiting submissions snapshot (bad magic)")]
    BadMagic,
    /// The snapshot was written with another layout version.
    #[error(
        "waiting submissions snapshot layout version {found} is not supported (this build reads {SNAPSHOT_VERSION})"
    )]
    UnsupportedVersion {
        /// The version found.
        found: u32,
    },
    /// Bytes follow the body.
    #[error("{count} unexpected bytes after the waiting submissions snapshot")]
    TrailingBytes {
        /// How many.
        count: usize,
    },
    /// A wait fence on an id that is not a syncpoint.
    #[error("waiting submission {index} waits on syncpoint {id}, which does not exist")]
    WaitSyncpoint {
        /// The entry's place in the list.
        index: usize,
        /// The id.
        id: u8,
    },
    /// More entries than a channel's queue holds.
    #[error("waiting submission {index} has {count} entries, more than a queue holds")]
    TooManyEntries {
        /// The entry's place in the list.
        index: usize,
        /// How many.
        count: usize,
    },
    /// A token id an earlier entry has.
    #[error("waiting submission {index} answers request {token}, which an earlier one answers")]
    DuplicateToken {
        /// The entry's place in the list.
        index: usize,
        /// The token id.
        token: u64,
    },
    /// An entry that does not come after the one before it, by channel and then token id.
    #[error("waiting submission {index} is out of order")]
    Misordered {
        /// The entry's place in the list.
        index: usize,
    },
    /// A request the server does not hold as pending.
    #[error("waiting submission {index} answers request {token}, which is not pending")]
    NotPending {
        /// The entry's place in the list.
        index: usize,
        /// The token id.
        token: u64,
    },
}
