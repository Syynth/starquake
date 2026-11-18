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
//! reply token of its request, if it has one, and the front it goes to. It lives in [`crate::shared::GsvShared`]
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
//! **A submission without a token.** Once `SubmitGPFIFO` has raised the channel syncpoint's
//! maximum and given the fence out, its submission is GPU work the fence counts on, so it is
//! never dropped while its channel lives, even when nobody waits for its reply:
//!
//! - its out buffer could not be written, or no reply token could be taken: the request
//!   answers that fault, and the submission waits with no token (`run_ioctl` in
//!   [`crate::nvsrv`]; Eden writes nothing back but keeps the submission);
//! - its request's session was closed ([`Server::close`] drops the pending request): the entry
//!   keeps its token, whose completion then goes nowhere, and [`Submits::save`] saves it with
//!   no token, as restoring could not reissue it.
//!
//! When such a submission is queued, no reply is completed.
//!
//! **What bounds the table: nothing yet.** A waiting submission with a token keeps its
//! request's session busy (the service framework handles one request at a time per session),
//! but one without a token holds nothing of the title's. So while a channel's queue stays full,
//! a title can add one submission (with up to [`MAX_ENTRY_BYTES`](super::MAX_ENTRY_BYTES) of
//! entries) per request it sends and then abandons, by closing the session or by giving an
//! out buffer that cannot be written, without limit
//! (`gpu/tests.rs`: `abandoned_submissions_are_not_bounded_yet`). A bound needs a guest-visible
//! answer for a submission refused when the table is full, which neither reference has (neither
//! waits); it is an open question for the owner.
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
//! request's token id if it has one whose request `server` holds as pending (an option), its
//! entries, its wait fence (syncpoint id and value) if any, its
//! increments and its wait-for-idle flag. The golden-bytes test `the_snapshot_layout_is_pinned`
//! (in `nvsrv/gpu/tests.rs`, with the other tests of this module, which need the server's reply
//! tokens) pins it.
//!
//! A snapshot is a file and is not trusted. [`Submits::restore`] checks, in this order, and
//! returns the first failure: the magic, the version, that the body decodes, that no bytes follow
//! it, then each entry in list order ([`SnapshotError`], in the order of its variants): its wait
//! fence's syncpoint, its entry count, that its token id is not one an earlier entry has, that it
//! comes after the entry before it (channel, then token id: a later request takes a later token),
//! and that the server holds its request as pending ([`Server::pending`]); the last three only
//! for an entry with a token. "After" means: a channel not below the one before it, and within a
//! channel, a token id above the last token id of that channel (an entry without a token has
//! none to compare). Only once every entry
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

/// The most entries of submissions that may wait for room on one GPU channel (module docs, "The
/// cap"): the owner's ruling of 2026-11-17 in the decision log, "Cap a channel's waiting
/// submissions at 65,536 entries; past it, answer Busy", which takes the figure decision 12 uses
/// for a channel's queue. Each waiting submission counts its [`charge`] against it.
pub const MAX_WAITING_ENTRIES: usize = CHANNEL_QUEUE_ENTRIES;

/// What a waiting submission of `entries` GP entries counts against [`MAX_WAITING_ENTRIES`]: its
/// entries, and at least one, so that submissions with no entries cannot pile up without bound
/// either (module docs, "The cap": a departure from the ruling's wording, for the owner).
#[must_use]
pub const fn charge(entries: usize) -> usize {
    if entries == 0 { 1 } else { entries }
}

/// One waiting submission.
struct Waiting {
    token: Option<ReplyToken>,
    submission: Submission,
    front: Arc<dyn ChannelFront>,
}

/// One channel's waiting submissions, oldest first, and what they count against the cap: the
/// sum of each one's [`charge`], kept as they come and go, so that a check never walks the queue.
#[derive(Default)]
struct Queue {
    waiting: VecDeque<Waiting>,
    charged: usize,
}

impl Queue {
    /// Adds `waiting` at the back. The caller has checked that it fits under the cap.
    fn push_back(&mut self, waiting: Waiting) {
        let charged = charge(waiting.submission.entries.len());
        self.charged = self.charged.saturating_add(charged);
        self.waiting.push_back(waiting);
    }

    /// Takes the oldest submission out.
    fn pop_front(&mut self) -> Option<Waiting> {
        let waiting = self.waiting.pop_front()?;
        let charged = charge(waiting.submission.entries.len());
        self.charged = self.charged.saturating_sub(charged);
        Some(waiting)
    }
}

/// A submission [`Submits::push`] refused because its channel's waiting submissions would pass
/// the cap (module docs, "The cap").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error(
    "GPU channel {channel} has {waiting} entries waiting, and {entries} more would pass the cap of {MAX_WAITING_ENTRIES}"
)]
pub struct OverCap {
    /// The GPU channel.
    pub channel: u32,
    /// What its waiting submissions count against the cap.
    pub waiting: usize,
    /// The refused submission's entries.
    pub entries: usize,
}

/// The submissions that wait for room, by channel, oldest first (module docs, "The table").
#[derive(Default)]
pub struct Submits {
    waiting: BTreeMap<ChannelId, Queue>,
}

impl fmt::Debug for Submits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.saved(|_| true)).finish()
    }
}

/// Two tables are equal when they hold the same submissions for the same requests, in the same
/// order; the fronts are not compared.
impl PartialEq for Submits {
    fn eq(&self, other: &Self) -> bool {
        self.saved(|_| true) == other.saved(|_| true)
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
        self.waiting
            .get(&channel)
            .map_or(0, |queue| queue.waiting.len())
    }

    /// What `channel`'s waiting submissions count against [`MAX_WAITING_ENTRIES`]: the sum of
    /// each one's [`charge`].
    #[must_use]
    pub fn waiting_entries(&self, channel: ChannelId) -> usize {
        self.waiting.get(&channel).map_or(0, |queue| queue.charged)
    }

    /// Whether a submission of `entries` entries may wait behind `channel`'s waiting ones
    /// without passing [`MAX_WAITING_ENTRIES`] (module docs, "The cap"). It does not walk the
    /// queue.
    #[must_use]
    pub fn admits(&self, channel: ChannelId, entries: usize) -> bool {
        self.waiting_entries(channel)
            .checked_add(charge(entries))
            .is_some_and(|charged| charged <= MAX_WAITING_ENTRIES)
    }

    /// How many submissions wait, of every channel.
    #[must_use]
    pub fn len(&self) -> usize {
        self.waiting.values().map(|queue| queue.waiting.len()).sum()
    }

    /// Whether nothing waits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }

    /// The token ids of `channel`'s waiting requests, oldest first; a submission without a token
    /// has none.
    pub fn tokens(&self, channel: ChannelId) -> impl Iterator<Item = TokenId> + '_ {
        self.waiting
            .get(&channel)
            .into_iter()
            .flat_map(|queue| &queue.waiting)
            .filter_map(|waiting| waiting.token.as_ref().map(ReplyToken::id))
    }

    /// Adds `submission`, whose request `token` answers if given, behind `channel`'s waiting
    /// ones, then tries the channel's queue (module docs, "The table"). A submission that would
    /// take the channel past [`MAX_WAITING_ENTRIES`] is refused instead: `token`, if given,
    /// answers `Busy`, nothing is added, and the queue is not tried (module docs, "The cap").
    pub fn push(
        &mut self,
        channel: ChannelId,
        token: Option<ReplyToken>,
        submission: Submission,
        front: Arc<dyn ChannelFront>,
    ) -> Result<(), OverCap> {
        let entries = submission.entries.len();
        if !self.admits(channel, entries) {
            if let Some(token) = token {
                token.complete(reply(NvResult::Busy));
            }
            return Err(OverCap {
                channel: channel.0,
                waiting: self.waiting_entries(channel),
                entries,
            });
        }
        self.waiting.entry(channel).or_default().push_back(Waiting {
            token,
            submission,
            front,
        });
        self.room(channel);
        Ok(())
    }

    /// `channel`'s queue has room: its waiting submissions go to the front, oldest first, until
    /// it answers `Full`; each one queued has its reply completed (module docs, "The table").
    pub fn room(&mut self, channel: ChannelId) {
        let Some(queue) = self.waiting.get_mut(&channel) else {
            return;
        };
        while let Some(head) = queue.waiting.front() {
            if head.front.submit(channel, head.submission.clone()) == Submitted::Full {
                break;
            }
            if let Some(token) = queue.pop_front().and_then(|queued| queued.token) {
                token.complete(reply(NvResult::Success));
            }
        }
        if queue.waiting.is_empty() {
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
        let waiting = self.waiting.remove(&channel).into_iter();
        let waiting = waiting.flat_map(|queue| queue.waiting);
        for token in waiting.filter_map(|waiting| waiting.token) {
            token.complete(reply(NvResult::NotImplemented));
        }
    }

    /// The table as it is saved, in order, each token kept only if `pending` holds for its id.
    fn saved(&self, pending: impl Fn(TokenId) -> bool) -> Vec<SavedSubmit> {
        let pending = &pending;
        self.waiting
            .iter()
            .flat_map(|(&channel, queue)| {
                queue.waiting.iter().map(move |waiting| SavedSubmit {
                    channel: channel.0,
                    token: waiting
                        .token
                        .as_ref()
                        .map(ReplyToken::id)
                        .filter(|&id| pending(id)),
                    entries: waiting
                        .submission
                        .entries
                        .iter()
                        .map(|entry| entry.0)
                        .collect(),
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

    /// Writes the table (module docs, "Snapshot format"), with `server`, the nvsrv server saved
    /// with it: a token whose request `server` no longer holds as pending is saved as none. Take
    /// it while nothing changes either.
    pub fn save(&self, server: &Server) -> Result<Vec<u8>, SnapshotError> {
        let pending = |id: TokenId| server.pending().any(|(pending, _)| pending == id);
        let mut bytes = postcard::to_allocvec(&Header {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
        })
        .map_err(SnapshotError::Encode)?;
        bytes.extend(postcard::to_allocvec(&self.saved(pending)).map_err(SnapshotError::Encode)?);
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
        // What each channel's entries so far count against the cap.
        let mut charged_by: BTreeMap<u32, usize> = BTreeMap::new();
        // The channel of the entry before, and the last token id seen in that channel.
        let mut previous: Option<(u32, Option<TokenId>)> = None;
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
            let charged = charged_by.entry(entry.channel).or_default();
            *charged = charged
                .checked_add(charge(entry.entries.len()))
                .filter(|&charged| charged <= MAX_WAITING_ENTRIES)
                .ok_or(SnapshotError::OverCap {
                    index,
                    channel: entry.channel,
                })?;
            if let Some(token) = entry.token
                && !seen.insert(token)
            {
                let token = token.raw();
                return Err(SnapshotError::DuplicateToken { index, token });
            }
            let last = match previous {
                Some((channel, _)) if channel > entry.channel => {
                    return Err(SnapshotError::Misordered { index });
                }
                Some((channel, last)) if channel == entry.channel => last,
                _ => None,
            };
            // An equal id is a token an earlier entry has, refused just above, so `>=` and `>`
            // refuse the same snapshots here.
            if let (Some(last), Some(token)) = (last, entry.token)
                && last >= token
            {
                return Err(SnapshotError::Misordered { index });
            }
            previous = Some((entry.channel, entry.token.or(last)));
            if let Some(token) = entry.token
                && !server.pending().any(|(pending, _)| pending == token)
            {
                let token = token.raw();
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
            let token = match id {
                None => None,
                Some(id) => match server.reissue(id) {
                    Some(token) => Some(token),
                    None => continue,
                },
            };
            submits
                .waiting
                .entry(channel)
                .or_default()
                .push_back(Waiting {
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
    /// The request's token id, if it has a pending request.
    pub(super) token: Option<TokenId>,
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
    /// The first entry whose channel's waiting submissions, counted up to and with it, pass
    /// [`MAX_WAITING_ENTRIES`] (module docs, "The cap").
    #[error(
        "waiting submission {index} takes GPU channel {channel}'s waiting entries past the cap of {MAX_WAITING_ENTRIES}"
    )]
    OverCap {
        /// The entry's place in the list.
        index: usize,
        /// Its channel.
        channel: u32,
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
