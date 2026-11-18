//! vi2's presents in gsv's shared state: each client's queue of presented frames, the FIFO latch
//! the vsync tick runs, the release of the frame a latched one replaces, the deltas a session
//! takes back into its own state, and the fence wait fallback's held replies
//! (`docs/specs/graphics-server.md`, "Present (229) and pacing"; port slice 14). The commands
//! that fill it, 228 and 229, are [`crate::vi2`]'s.
//!
//! Ported from (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project), cited below by these
//! short forms, as in [`crate::protocol`]:
//!
//! - **PR:** Eden PR #4588, `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp`
//!   @ e5fb3b97: `IVi2Session::QueueNativeImage` (what a present queues and which slot it
//!   takes) and `IVi2Session::RefreshReleasedBuffers` (a released slot's bit set again, the
//!   client's "buffer available" notification raised and the notification event signalled,
//!   once per client that had a slot released).
//! - **NX2:** Eden-NX2, `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb:
//!   `INx2Vi2Session::PresentClientBuffer`, `WaitForFrame` and `PaceToVblank` (the fallback: the
//!   reply held until the previous frame's fence and the next vblank, every slot reported free).
//!
//! # Why the queues are shared
//!
//! A vi2 session keeps its clients ([`crate::vi2::Client`]) as PR keeps them, per session. The
//! latch runs on the vsync tick, on the timer thread, which cannot reach a session's state; so
//! what the tick reads and writes lives here, in gsv's shared state, one [`PresentClient`] per
//! registered client, keyed by the client's number: the lowest number below
//! [`MAX_PRESENT_CLIENTS`] that no client holds, which 201 gives the client
//! ([`Presents::register`]) and the session keeps beside its token. A token names a client only
//! within its session, as in PR: two sessions may register the same token, and each gets its
//! own entry (`a_token_is_a_client_per_session`, in `vi2/tests/present.rs`). The number is what
//! the latch's notes name the client by ([`crate::note`], codes 5 and 6). The table holds, for
//! each client:
//!
//! - the frames 229 queued, oldest first, and the one shown;
//! - the slots the ticks released and whether they raised the client's "buffer available"
//!   notification, which the session takes back into its own free-slot mask and notification
//!   bitmap ([`Presents::take_refresh`]) at the start of 11, 21, 212, 228 and 229, as PR runs
//!   `RefreshReleasedBuffers` at the start of 212 and 229 and from a 1 ms poll;
//! - the session's notification event, so that the tick can signal it;
//! - the fence wait fallback's held reply.
//!
//! 201 adds the client's entry and 202 removes it ([`Locked::unregister_present`]), so the table
//! holds exactly the clients the sessions registered, until a session closes without 202 (below,
//! "Bounds"); 202 frees the number for the next 201.
//!
//! # The latch
//!
//! Each vsync tick, after it has signalled the clock's listeners ([`crate::shared::vsync`],
//! "Tick"), runs the latch (`Locked::latch`) over the clients in ascending number order. For
//! each client:
//!
//! 1. If the oldest queued frame's fence is reached, it is taken out of the queue and becomes the
//!    shown frame, a `FrameLatched` note is recorded, and its [`PresentRequest`] is handed back to
//!    the caller, which gives it to the GPU front ([`PresentFront::present`]). At most one frame
//!    is latched per client per tick (spec: "At most one frame is latched per tick"), and only the
//!    oldest: a later frame whose fence is reached does not pass an older one that is not (the
//!    FIFO latch; `a_reached_frame_does_not_pass_an_older_unreached_one`). A frame that misses a
//!    tick waits for the next one; nothing is banked.
//! 2. The frame it replaces, if one was shown, is released on the same tick: a `BufferReleased`
//!    note is recorded, and if the frame holds its slot (a buffer queue present), the slot's bit
//!    is released, the "buffer available" notification is raised and the session's notification
//!    event, if 11 made it, is signalled (PR `RefreshReleasedBuffers`; spec step 2). A slot comes
//!    back exactly on the tick that latches the frame replacing it, and not before
//!    (`a_slot_returns_on_the_tick_that_latches_the_frame_replacing_it`).
//! 3. A held fallback reply whose previous frame has been latched is completed (below, "The fence
//!    wait fallback").
//!
//! **Reached means drained.** A frame's fence counts as reached once a drained
//! `SyncpointChanged` says so, as an armed syncpt wait fires on the drained value and not the
//! counter (`crate::shared`, "Waits fire on the drained value"): queuing a frame watches its
//! fence's syncpoint ([`Locked::watch`]) and keeps the watch until a drained value reaches it
//! ([`Presents::reach`], from [`crate::shared::Shared::drain`]), which records that value as a
//! `SyncpointObserved` note. When the counter read after the watch already reaches the fence
//! (32-bit wrapping, PR `static_cast<s32>(*counter - point.value) < 0`), the frame is still not
//! reached when it is queued: [`Locked::commit_frame`] hands back a
//! `SyncpointChanged { id, value }` of that reading, which the caller pushes onto the completion
//! queue, so the next drain records it and marks the frame reached, as syncpt 201 does for a
//! wait reached when armed (`crate::syncpt`, "Where this differs"). So every frame with a fence
//! is reached by a drained value that a `SyncpointObserved` note records, and a replay that
//! feeds the notes back (where the counter moves only on what is fed back) reaches and latches
//! it on the same ticks (`an_already_reached_fence_replays_to_the_same_latch`, in
//! `hle/tests/graphics/present.rs` of the core crate). A frame with no fence (the request's
//! fence word is 0, PR `fence_id == ~u32{0}`) is reached at once.
//!
//! **The clock runs while a present waits.** The vsync clock runs while it has a listener
//! ([`crate::shared::vsync`], "The clock") or while this table has work: a queued frame or a held
//! reply ([`Presents::has_work`]). Queuing a frame on a stopped clock starts it one interval after
//! the `Env` clock's reading, as a new listener does, and a tick that leaves no listener and no
//! work stops it. So a title that presents without a vsync request group still gets its
//! buffers back (PR frees them from a 1 ms poll of its own, NX2 from its own wall-clock wait).
//!
//! # The fence wait fallback
//!
//! Under [`crate::vi2::Pacing::FenceWait`] (the owner's decision 1, a per-run setting), 229
//! queues its frame without taking its slot and holds its reply ([`Waiting`]): the reply is
//! completed with every slot free ([`crate::vi2::ALL_SLOTS`], NX2 `AvailableSlots = 0x7fff`) on
//! the first tick at which the frame queued before it, if any, has been latched. A frame is
//! latched only once its fence is reached, so that is the first tick after the previous frame's
//! fence (NX2 waits for the previous frame's fence, `m_previous_fence_id`, then `PaceToVblank`).
//! With no previous frame queued, the next tick completes it. A session handles one request at a
//! time, so while a reply is held its client queues nothing more: at most two frames are queued
//! (`the_fallback_s_reply_completes_on_the_first_tick_after_the_previous_frame_s_fence`).
//!
//! # Bounds
//!
//! At most [`MAX_PRESENT_CLIENTS`] clients, at most [`SLOTS`] frames queued per client
//! ([`QueueError::Full`]: a buffer queue present holds a distinct slot, so only the fallback could
//! queue more, and it holds at most two), and at most one held reply per client. A session closed
//! without 202 leaves its clients here, as it leaves its events in the kernel (the framework tells
//! a session nothing when it closes; [`crate::vi2`], "Where this differs"): their frames still
//! latch, and their numbers stay taken. [`MAX_PRESENT_CLIENTS`] bounds what that can grow to;
//! past it 201 answers `OutOfResource`. Each tick's latch walks the table once.
//!
//! # What is recorded
//!
//! The latch records, for each client in number order, `FrameLatched` then `BufferReleased`
//! ([`crate::note`], codes 5 and 6: the client's number, the slot and the image), after the
//! tick's own `VsyncTick` and
//! before the release's signal. Both follow from the ticks and the drained syncpoint values a
//! replay feeds back, so a replay does not feed them back.
//!
//! # What is saved
//!
//! [`crate::shared::Shared::save_presents`] writes the table: `postcard`, after a header of
//! [`SNAPSHOT_MAGIC`] and [`SNAPSHOT_VERSION`], decoded and checked before the body. The body is
//! the clients in ascending number order, each its number, token, notification bit,
//! notification event, queued frames oldest first, shown frame, released slots, whether the
//! notification is raised,
//! and its held reply (the reply's token id, if the vi2 server holds the request as pending, and
//! how many frames were queued ahead of it). A frame is its slot, image, the present request's
//! fields (the image's address, width, height, pitch, block height and format, the shown area's
//! x, y, width and height, the layer and the display), its fence (syncpoint id and value),
//! whether it is reached and whether it holds its slot. The golden-bytes test
//! `the_snapshot_layout_is_pinned` pins it; changing it means bumping [`SNAPSHOT_VERSION`].
//!
//! A snapshot is a file and is not trusted. [`crate::shared::Shared::restore_presents`] checks,
//! in this order, and refuses the first failure ([`PresentSnapshotError`], in the order of its
//! variants): the magic, the version, that the body decodes, that no bytes follow it; at most
//! [`MAX_PRESENT_CLIENTS`] clients; then each client in order: a number below
//! [`MAX_PRESENT_CLIENTS`] and above the one before, a token that is not zero, a notification
//! bit below [`CLIENT_NOTIFICATIONS`], at most [`SLOTS`] queued frames, then each frame (queued
//! oldest first, then the shown one): a slot below [`SLOTS`], a fence on a syncpoint that
//! exists, reached if it has no fence, reached if it is the shown one, a slot no earlier frame
//! of the client holds (for frames that hold theirs); then released slots within [`ALL_SLOTS`],
//! none of them a slot a queued or shown frame holds (229 refreshes before it queues a slot
//! again, so a live client never has both), the notification raised exactly when a slot is
//! released (a release sets both and a refresh clears both), a held reply with fewer frames
//! ahead of it than are queued, a reply token no earlier client's reply has, and one the vi2
//! server holds as pending. Then,
//! syncpoint by syncpoint, that the unreached frames on it and the syncpt waits armed on it
//! together do not outnumber its waiters (each holds one watch; a frame's watch is what lets a
//! drained value reach it). [`crate::shared::Shared::save_presents`] runs the same relation checks
//! on what it writes. Only once all pass are the held replies given new tokens and the table
//! replaced.
//!
//! Not checked here, each with its consequence:
//!
//! - That each entry is a client a vi2 session registered under that number and token (the
//!   sessions' snapshots hold them), or that no two sessions' clients name one number. An entry
//!   no session holds is never refreshed, its frames latch and its number stays taken; a
//!   session client whose number has no entry here answers 229 with `InvalidState`; two session
//!   clients naming one entry share its queue and its releases.
//! - That a frame's fence is on a syncpoint that is allocated now. Live 229 refuses a fence on
//!   one that is not (PR `IsSyncpointAllocated(fence_id)`), but the syncpoint can be freed while
//!   the frame waits (closing the channel that holds it), and a freed counter keeps its value
//!   (`Syncpoints::free`), so a live table reaches this state. Such a frame waits for a drained
//!   value of the syncpoint, as it does live (`a_restore_accepts_a_frame_fenced_on_a_freed_syncpoint`).
//! - That each event is the notification event of the session holding the client. One the kernel
//!   does not hold is refused at the signal, logged, and the release goes on.
//! - That a frame's image still exists, or that its request matches the image: the frame keeps
//!   the description it was queued with (as it does live once 314 and 430 destroy the image; PR
//!   keeps the image's backing until its buffer retires, which is not ported).
//! - That the vsync clock runs when the table has work: restore this table before the clock
//!   ([`crate::shared::Shared::restore_vsync`] accepts a deadline with no listener only while
//!   this table has work). A table with work behind a stopped clock waits for the next listener
//!   or 229 to start it.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::{
    DisplayId, Fence, GpuCompletion, ImageDesc, LayerId, PresentRequest, Rect, SYNCPOINT_COUNT,
    SyncpointId,
};
use starquake_gpu::syncpoint::{CompareMode, wait_satisfied};
use starquake_kernel::id::ObjectId;
use starquake_services::framework::{Reply, ReplyToken, Server, TokenId};
use starquake_trace::{HostNanos, ServiceId};
use thiserror::Error;

use super::{Locked, TimerHost, WatchError};
use crate::note::Note;
use crate::syncpt::waiters::Waiters;
use crate::vi2::{ALL_SLOTS, CLIENT_NOTIFICATIONS, MAX_CLIENTS, SLOTS};

#[cfg(test)]
mod tests;

/// The most clients the table holds: room for eight vi2 sessions of PR's [`MAX_CLIENTS`] each,
/// as [`crate::shared::vsync::MAX_LISTENERS`] is room for eight sessions' request groups.
/// Starquake's own bound, a **guess** (module docs, "Bounds"): PR keeps clients per session and
/// does not count sessions. A client's number is below it, and fits the byte the latch's notes
/// give it ([`crate::note::CLIENT_SHIFT`]).
pub const MAX_PRESENT_CLIENTS: usize = 8 * MAX_CLIENTS;

/// The zero words a 229 or 228 reply ends with: PR `for (u32 i = 0; i < 4; ++i)
/// rb.Push<u32>(0)`; NX2 `TipcReplyWords(ctx, {..., 0, 0, 0, 0})`.
pub const REPLY_ZERO_WORDS: usize = 4;

/// First bytes of every present table snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SQPR";

/// Layout version written into every present table snapshot and required when reading one.
pub const SNAPSHOT_VERSION: u32 = 1;

/// Slot `slot`'s bit in a client's `u64` mask of free slots; 0 for a slot at or past 64.
#[must_use]
pub const fn slot_bit(slot: u8) -> u64 {
    match 1_u64.checked_shl(slot as u32) {
        Some(bit) => bit,
        None => 0,
    }
}

/// The reply a present answers with: the result, the `u64` mask of free slots, then
/// [`REPLY_ZERO_WORDS`] zero words (PR `PresentClientBuffer`, `CancelClientBuffer`; spec: "result,
/// u64 mask of the client's slots now free, four zero words").
pub fn write_slots(reply: &mut Reply, available: u64) {
    reply.payload().write_u64(available);
    for _ in 0..REPLY_ZERO_WORDS {
        reply.payload().write_u32(0);
    }
}

/// Where the GPU front takes a latched frame: the part of the spec's `GpuFront` the latch uses
/// (graphics-server, "What this spec offers and assumes of the GPU front/back": `fn present(&self,
/// request: PresentRequest)`, "no fence: gsv latched it after its fence was reached"). As
/// [`crate::gpu::GpuVmFront`] is for the address tables, `crates/gpu` does not define `GpuFront`
/// yet; when it does, it implements this, or this becomes it.
pub trait PresentFront: Send + Sync {
    /// Shows `request`, ordered after the work queued before it.
    fn present(&self, request: PresentRequest);
}

/// One presented frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    slot: u8,
    image: u32,
    request: PresentRequest,
    fence: Option<Fence>,
    reached: bool,
    holds_slot: bool,
}

impl Frame {
    /// The buffer slot 229 named.
    #[must_use]
    pub const fn slot(&self) -> u8 {
        self.slot
    }

    /// The native image 229 named.
    #[must_use]
    pub const fn image(&self) -> u32 {
        self.image
    }

    /// What the GPU front is handed when it is latched.
    #[must_use]
    pub const fn request(&self) -> &PresentRequest {
        &self.request
    }

    /// The fence it waits for, if 229 named one.
    #[must_use]
    pub const fn fence(&self) -> Option<Fence> {
        self.fence
    }

    /// Whether it is reached: it has no fence, or a drained value has reached its fence (module
    /// docs, "Reached means drained").
    #[must_use]
    pub const fn reached(&self) -> bool {
        self.reached
    }

    /// Whether it holds its slot: a buffer queue present does, a fence wait present does not.
    #[must_use]
    pub const fn holds_slot(&self) -> bool {
        self.holds_slot
    }
}

/// A frame 229 is about to queue ([`Locked::prepare_frame`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NewFrame {
    /// The buffer slot.
    pub slot: u8,
    /// The native image.
    pub image: u32,
    /// What the GPU front is handed.
    pub request: PresentRequest,
    /// The fence to wait for, if any.
    pub fence: Option<Fence>,
    /// Whether it holds its slot until it is released.
    pub holds_slot: bool,
}

/// A frame whose fence's syncpoint is watched if it needs to be, ready to be queued
/// ([`Locked::commit_frame`]) or given back ([`Locked::abandon_frame`]).
#[must_use = "a prepared frame holds a watch until it is committed or abandoned"]
#[derive(Debug, PartialEq, Eq)]
pub struct Prepared {
    client: u8,
    frame: Frame,
    /// The counter's reading after the watch, when it already reaches the fence: the change
    /// [`Locked::commit_frame`] hands back to push.
    observed: Option<u32>,
}

/// A fallback reply held until the frame queued before it is latched (module docs, "The fence
/// wait fallback").
pub struct Waiting {
    token: Option<ReplyToken>,
    ahead: usize,
}

impl Waiting {
    /// The held request's token id, if it has a token.
    #[must_use]
    pub fn token(&self) -> Option<TokenId> {
        self.token.as_ref().map(ReplyToken::id)
    }

    /// How many frames were queued ahead of the held one and are not latched yet.
    #[must_use]
    pub const fn ahead(&self) -> usize {
        self.ahead
    }
}

/// One registered client's presents (module docs, "Why the queues are shared").
pub struct PresentClient {
    token: u64,
    available_bit: u32,
    event: Option<ObjectId>,
    queued: VecDeque<Frame>,
    shown: Option<Frame>,
    released: u64,
    raised: bool,
    waiting: Option<Waiting>,
}

impl PresentClient {
    /// A client 201 registered: nothing queued or shown, nothing released.
    const fn new(token: u64, available_bit: u32, event: Option<ObjectId>) -> Self {
        Self {
            token,
            available_bit,
            event,
            queued: VecDeque::new(),
            shown: None,
            released: 0,
            raised: false,
            waiting: None,
        }
    }

    /// The client's token, which names it within its session.
    #[must_use]
    pub const fn token(&self) -> u64 {
        self.token
    }

    /// The client's "buffer available" notification bit.
    #[must_use]
    pub const fn available_bit(&self) -> u32 {
        self.available_bit
    }

    /// The session's notification event, once 11 made it.
    #[must_use]
    pub const fn event(&self) -> Option<ObjectId> {
        self.event
    }

    /// The queued frames, oldest first.
    pub fn queued(&self) -> impl ExactSizeIterator<Item = &Frame> {
        self.queued.iter()
    }

    /// The frame shown.
    #[must_use]
    pub const fn shown(&self) -> Option<&Frame> {
        self.shown.as_ref()
    }

    /// The slots the ticks released since the session last took them.
    #[must_use]
    pub const fn released(&self) -> u64 {
        self.released
    }

    /// Whether a tick raised the "buffer available" notification since the session last took it.
    #[must_use]
    pub const fn raised(&self) -> bool {
        self.raised
    }

    /// The held fallback reply.
    #[must_use]
    pub const fn waiting(&self) -> Option<&Waiting> {
        self.waiting.as_ref()
    }

    /// Whether the clock must run for it: a frame is queued or a reply held.
    fn has_work(&self) -> bool {
        !self.queued.is_empty() || self.waiting.is_some()
    }

    /// Whether a frame that holds `slot` is queued or shown (PR `state.queued[slot]`).
    fn holds(&self, slot: u8) -> bool {
        self.queued
            .iter()
            .chain(&self.shown)
            .any(|frame| frame.holds_slot && frame.slot == slot)
    }
}

/// What a session takes back into a client's own state ([`Presents::take_refresh`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Refresh {
    /// The slots released.
    pub released: u64,
    /// Whether the "buffer available" notification was raised.
    pub raised: bool,
}

/// Why 201 could not add a client to the table ([`Presents::register`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum RegisterError {
    /// The table already holds [`MAX_PRESENT_CLIENTS`] clients.
    #[error("the present table already holds the most clients it takes ({MAX_PRESENT_CLIENTS})")]
    Full,
}

/// Why a frame could not be queued ([`Locked::prepare_frame`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum QueueError {
    /// No client has the number in the table (only a restore without the table's snapshot
    /// leads here: module docs, "What is saved").
    #[error("the present table has no client {client}")]
    NoClient {
        /// The client's number.
        client: u8,
    },
    /// The client already has [`SLOTS`] frames queued.
    #[error("present table client {client} already has the most frames queued ({SLOTS})")]
    Full {
        /// The client's number.
        client: u8,
    },
    /// The fence's syncpoint could not be watched.
    #[error("the frame's fence could not be watched: {0}")]
    Watch(#[from] WatchError),
}

/// The presents of every registered client, by number (module docs).
#[derive(Default)]
pub struct Presents {
    clients: BTreeMap<u8, PresentClient>,
}

impl fmt::Debug for Presents {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.saved(|_| true)).finish()
    }
}

/// Two tables are equal when they save the same (held replies by token id).
impl PartialEq for Presents {
    fn eq(&self, other: &Self) -> bool {
        self.saved(|_| true) == other.saved(|_| true)
    }
}

impl Eq for Presents {}

impl Presents {
    /// No client.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Client number `client`.
    #[must_use]
    pub fn client(&self, client: u8) -> Option<&PresentClient> {
        self.clients.get(&client)
    }

    /// Every client with its number, in ascending number order.
    pub fn clients(&self) -> impl Iterator<Item = (u8, &PresentClient)> {
        self.clients
            .iter()
            .map(|(&number, client)| (number, client))
    }

    /// Whether the vsync clock must run for the table: a client has a frame queued or a reply
    /// held (module docs, "The clock runs while a present waits").
    #[must_use]
    pub fn has_work(&self) -> bool {
        self.clients.values().any(PresentClient::has_work)
    }

    /// Whether client `client` has a frame that holds `slot` queued or shown: a buffer queue
    /// present of the slot is refused then (PR `!state.queued[slot]`).
    #[must_use]
    pub fn holds(&self, client: u8, slot: u8) -> bool {
        self.clients
            .get(&client)
            .is_some_and(|client| client.holds(slot))
    }

    /// Adds a client with `token` (201), its "buffer available" bit and the session's
    /// notification event, if 11 made it, under the lowest number no client holds: that number.
    /// A token another client has is no refusal (module docs, "Why the queues are shared"). It
    /// refuses a full table ([`RegisterError::Full`]), and nothing changes then.
    pub fn register(
        &mut self,
        token: u64,
        available_bit: u32,
        event: Option<ObjectId>,
    ) -> Result<u8, RegisterError> {
        let number = (0..=u8::MAX)
            .take(MAX_PRESENT_CLIENTS)
            .find(|number| !self.clients.contains_key(number))
            .ok_or(RegisterError::Full)?;
        self.clients
            .insert(number, PresentClient::new(token, available_bit, event));
        Ok(number)
    }

    /// Tells client `client`'s entry the session's notification event (11 made it).
    pub fn set_event(&mut self, client: u8, event: ObjectId) {
        if let Some(client) = self.clients.get_mut(&client) {
            client.event = Some(event);
        }
    }

    /// Takes what the ticks released for client `client` since the last time (module docs,
    /// "Why the queues are shared"): nothing for a number the table does not have.
    pub fn take_refresh(&mut self, client: u8) -> Refresh {
        let Some(client) = self.clients.get_mut(&client) else {
            return Refresh::default();
        };
        let refresh = Refresh {
            released: client.released,
            raised: client.raised,
        };
        client.released = 0;
        client.raised = false;
        refresh
    }

    /// Marks reached every queued frame on syncpoint `id` whose fence `value` reaches (the
    /// drain, for a drained `SyncpointChanged`; module docs, "Reached means drained"); how many
    /// it marked, each of which gives back its watch.
    pub fn reach(&mut self, id: SyncpointId, value: u32) -> usize {
        let mut marked = 0;
        for frame in self.clients.values_mut().flat_map(|c| c.queued.iter_mut()) {
            if let Some(fence) = frame.fence
                && fence.id == id
                && !frame.reached
                && wait_satisfied(value, u64::from(fence.value), CompareMode::Wrapping32)
            {
                frame.reached = true;
                marked += 1;
            }
        }
        marked
    }

    /// How many queued frames on syncpoint `id` are not reached: each holds a watch of it.
    #[must_use]
    pub fn unreached_on(&self, id: SyncpointId) -> usize {
        self.clients
            .values()
            .flat_map(|client| client.queued.iter())
            .filter(|frame| !frame.reached && frame.fence.is_some_and(|fence| fence.id == id))
            .count()
    }

    /// The table as it is saved, each held reply's token kept only if `pending` holds for it.
    fn saved(&self, pending: impl Fn(TokenId) -> bool) -> Vec<SavedClient> {
        self.clients
            .iter()
            .map(|(&number, client)| SavedClient {
                number,
                token: client.token,
                available_bit: client.available_bit,
                event: client.event,
                queued: client.queued.iter().map(SavedFrame::from).collect(),
                shown: client.shown.as_ref().map(SavedFrame::from),
                released: client.released,
                raised: client.raised,
                waiting: client.waiting.as_ref().map(|waiting| SavedWaiting {
                    token: waiting.token().filter(|&id| pending(id)),
                    ahead: u32::try_from(waiting.ahead).unwrap_or(u32::MAX),
                }),
            })
            .collect()
    }
}

impl super::GsvShared {
    /// vi2's presents.
    #[must_use]
    pub const fn presents(&self) -> &Presents {
        &self.presents
    }
}

impl Locked<'_> {
    /// vi2's presents, to change where no watch changes (201, 11, the refreshes).
    pub fn presents_mut(&mut self) -> &mut Presents {
        &mut self.state.presents
    }

    /// Whether syncpoint `id` is allocated: a present's fence must name one (PR
    /// `IsSyncpointAllocated(fence_id)`).
    #[must_use]
    pub fn syncpoint_allocated(&self, id: SyncpointId) -> bool {
        self.syncpoints.is_allocated(id)
    }

    /// Removes client `client` (202), freeing its number: its queued frames go, each unreached
    /// one giving back its watch, its shown frame goes, and a held reply is answered `NotFound`
    /// (a session handles one request at a time, so its own 202 cannot come while its reply is
    /// held; `a_held_reply_is_refused_when_its_client_is_removed`). Whether the table had it.
    pub fn unregister_present(&mut self, client: u8) -> bool {
        let Some(client) = self.state.presents.clients.remove(&client) else {
            return false;
        };
        for frame in &client.queued {
            if let (Some(fence), false) = (frame.fence, frame.reached) {
                self.release(fence.id);
            }
        }
        if let Some(Waiting {
            token: Some(reply), ..
        }) = client.waiting
        {
            reply.fail(starquake_kernel::result::KernelError::NotFound.code());
        }
        true
    }

    /// The first half of queuing a frame for client `client` (229): it refuses, in this order,
    /// a number with no entry ([`QueueError::NoClient`]), a client with [`SLOTS`] frames queued
    /// ([`QueueError::Full`]), and a fence whose syncpoint cannot be watched
    /// ([`QueueError::Watch`]); then it compares the fence with the counter read after watching
    /// it, and keeps the watch either way: a frame with a fence is reached only by a drained
    /// value (module docs, "Reached means drained"). Nothing changes on an error. Commit or
    /// abandon what it returns before the lock is released.
    pub fn prepare_frame(&mut self, client: u8, frame: NewFrame) -> Result<Prepared, QueueError> {
        let entry = self
            .state
            .presents
            .clients
            .get(&client)
            .ok_or(QueueError::NoClient { client })?;
        if entry.queued.len() >= usize::from(SLOTS) {
            return Err(QueueError::Full { client });
        }
        let observed = match frame.fence {
            None => None,
            Some(fence) => {
                let value = self.watch(fence.id)?;
                wait_satisfied(value, u64::from(fence.value), CompareMode::Wrapping32)
                    .then_some(value)
            }
        };
        Ok(Prepared {
            client,
            frame: Frame {
                slot: frame.slot,
                image: frame.image,
                request: frame.request,
                fence: frame.fence,
                reached: frame.fence.is_none(),
                holds_slot: frame.holds_slot,
            },
            observed,
        })
    }

    /// Gives back what [`Locked::prepare_frame`] took, when a later step fails.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taking the prepared frame by value stops it being committed after it is abandoned"
    )]
    pub fn abandon_frame(&mut self, prepared: Prepared) {
        let Prepared { frame, .. } = prepared;
        if let (Some(fence), false) = (frame.fence, frame.reached) {
            self.release(fence.id);
        }
    }

    /// The second half of queuing a frame: puts `prepared` behind the client's queued frames,
    /// holds `reply` (the fallback's) behind the frames queued before it, and starts the vsync
    /// clock one [`crate::shared::vsync::INTERVAL_NS`] after `now` if it is stopped (module
    /// docs, "The clock runs while a present waits"). `now` is the `Env` clock's reading, which
    /// the caller takes when the clock is stopped. It cannot fail: [`Locked::prepare_frame`]
    /// checked the client under the same lock.
    ///
    /// When the counter already reached the frame's fence as it was prepared, it returns the
    /// `SyncpointChanged` of that reading, which the caller pushes onto the completion queue
    /// before the lock is released (module docs, "Reached means drained"); otherwise nothing.
    #[must_use = "a fence already reached is reached only once its change is pushed and drained"]
    pub fn commit_frame(
        &mut self,
        prepared: Prepared,
        now: Option<HostNanos>,
        reply: Option<ReplyToken>,
    ) -> Option<GpuCompletion> {
        let Some(client) = self.state.presents.clients.get_mut(&prepared.client) else {
            tracing::error!(
                client = prepared.client,
                "gsv:vi2: a prepared frame's client left the table under the lock"
            );
            self.abandon_frame(prepared);
            return None;
        };
        let change = prepared
            .frame
            .fence
            .zip(prepared.observed)
            .map(|(fence, value)| GpuCompletion::SyncpointChanged {
                id: fence.id,
                value,
            });
        if let Some(token) = reply {
            client.waiting = Some(Waiting {
                token: Some(token),
                ahead: client.queued.len(),
            });
        }
        client.queued.push_back(prepared.frame);
        let vsync = &mut self.state.vsync;
        if vsync.is_stopped() {
            if let Some(now) = now {
                vsync.start(now);
            } else {
                tracing::error!(
                    "gsv:vi2: a frame was queued on a stopped vsync clock with no time to start it"
                );
            }
        }
        change
    }

    /// The latch of one vsync tick (module docs, "The latch"), recording through `host` for
    /// `service`: what to hand the GPU front, in client order.
    pub(super) fn latch(
        &mut self,
        host: &mut dyn TimerHost,
        service: ServiceId,
    ) -> Vec<PresentRequest> {
        let mut presents = Vec::new();
        for (&number, client) in &mut self.state.presents.clients {
            let latched = client.queued.front().is_some_and(|frame| frame.reached);
            if let Some(frame) = client.queued.pop_front_if(|frame| frame.reached) {
                host.record(
                    Note::FrameLatched {
                        client: number,
                        slot: frame.slot,
                        image: frame.image,
                    }
                    .event(service),
                );
                presents.push(frame.request);
                if let Some(replaced) = client.shown.replace(frame) {
                    release_frame(host, service, number, client, replaced);
                }
            }
            if let Some(mut waiting) = client.waiting.take() {
                if latched {
                    waiting.ahead = waiting.ahead.saturating_sub(1);
                }
                if waiting.ahead == 0 {
                    complete(number, waiting);
                } else {
                    client.waiting = Some(waiting);
                }
            }
        }
        presents
    }
}

/// Releases `frame`, which a latched frame replaced on client `number`'s layer (module docs,
/// "The latch", step 2).
fn release_frame(
    host: &mut dyn TimerHost,
    service: ServiceId,
    number: u8,
    client: &mut PresentClient,
    frame: Frame,
) {
    host.record(
        Note::BufferReleased {
            client: number,
            slot: frame.slot,
            image: frame.image,
        }
        .event(service),
    );
    if !frame.holds_slot {
        return;
    }
    client.released |= slot_bit(frame.slot);
    client.raised = true;
    if let Some(event) = client.event
        && let Err(error) = host.signal_event(event)
    {
        tracing::error!(
            %error,
            client = number,
            token = client.token,
            ?event,
            "gsv:vi2: the kernel refused to signal a released buffer's notification event"
        );
    }
}

/// Completes client `number`'s held fallback reply: every slot free (module docs, "The fence
/// wait fallback").
fn complete(number: u8, waiting: Waiting) {
    let Some(reply_token) = waiting.token else {
        tracing::debug!(
            client = number,
            "gsv:vi2: a held present reply with no request left to answer"
        );
        return;
    };
    let mut reply = Reply::default();
    write_slots(&mut reply, ALL_SLOTS);
    reply_token.complete(reply);
}

/// Why a present table snapshot could not be written or read (module docs, "What is saved").
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum PresentSnapshotError {
    /// Encoding failed.
    #[error("could not encode the present table snapshot: {0}")]
    Encode(postcard::Error),
    /// The header or the body is not a well-formed encoding.
    #[error("could not decode the present table snapshot: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`SNAPSHOT_MAGIC`].
    #[error("not a present table snapshot (bad magic)")]
    BadMagic,
    /// The snapshot was written with another layout version.
    #[error(
        "present table snapshot layout version {found} is not supported (this build reads {SNAPSHOT_VERSION})"
    )]
    UnsupportedVersion {
        /// The version found.
        found: u32,
    },
    /// Bytes follow the body.
    #[error("{count} unexpected bytes after the present table snapshot")]
    TrailingBytes {
        /// How many.
        count: usize,
    },
    /// More clients than [`MAX_PRESENT_CLIENTS`].
    #[error("the present table snapshot holds {count} clients, more than {MAX_PRESENT_CLIENTS}")]
    TooManyClients {
        /// How many it holds.
        count: usize,
    },
    /// A client whose number is not below [`MAX_PRESENT_CLIENTS`]: 201 gives none.
    #[error(
        "the present table snapshot's client {index} has the number {number}, not below {MAX_PRESENT_CLIENTS}"
    )]
    ClientNumber {
        /// The client's index.
        index: usize,
        /// The number it has.
        number: u8,
    },
    /// A client whose number is not above the one before it: listed twice, or out of order.
    #[error(
        "the present table snapshot's client {index} is not above the one before it (listed twice or out of order)"
    )]
    NumbersNotAscending {
        /// The client's index.
        index: usize,
    },
    /// A client whose token is zero: 201 refuses it.
    #[error("the present table snapshot's client {index} has the token zero")]
    ZeroToken {
        /// The client's index.
        index: usize,
    },
    /// A notification bit 201 refuses.
    #[error(
        "the present table snapshot's client {index} has a notification bit not below {CLIENT_NOTIFICATIONS}"
    )]
    NotificationBit {
        /// The client's index.
        index: usize,
    },
    /// More queued frames than [`SLOTS`].
    #[error(
        "the present table snapshot's client {index} has {count} frames queued, more than {SLOTS}"
    )]
    TooManyFrames {
        /// The client's index.
        index: usize,
        /// How many it has queued.
        count: usize,
    },
    /// A frame whose slot is not below [`SLOTS`].
    #[error(
        "the present table snapshot's client {index} has a frame in slot {slot}, past the last"
    )]
    Slot {
        /// The client's index.
        index: usize,
        /// The slot.
        slot: u8,
    },
    /// A frame whose fence names no syncpoint.
    #[error(
        "the present table snapshot's client {index} has a frame fenced on syncpoint {id}, which does not exist (there are {SYNCPOINT_COUNT})"
    )]
    FenceSyncpoint {
        /// The client's index.
        index: usize,
        /// The id it names.
        id: u8,
    },
    /// A frame with no fence that is not reached: 229 reaches it at once.
    #[error("the present table snapshot's client {index} has an unfenced frame not reached")]
    UnfencedUnreached {
        /// The client's index.
        index: usize,
    },
    /// A shown frame that is not reached: only a reached frame is latched.
    #[error("the present table snapshot's client {index} shows a frame not reached")]
    ShownUnreached {
        /// The client's index.
        index: usize,
    },
    /// Two frames holding one slot: 229 refuses a slot a frame holds.
    #[error("the present table snapshot's client {index} has two frames holding slot {slot}")]
    SlotHeldTwice {
        /// The client's index.
        index: usize,
        /// The slot.
        slot: u8,
    },
    /// Released slots past the fifteen.
    #[error("the present table snapshot's client {index} has a released slot past the last")]
    Released {
        /// The client's index.
        index: usize,
    },
    /// A released slot that a queued or shown frame of the client holds: 229 refreshes before
    /// it queues a slot again, so a live client never has both.
    #[error(
        "the present table snapshot's client {index} has slot {slot} released while a frame holds it"
    )]
    ReleasedHeld {
        /// The client's index.
        index: usize,
        /// The lowest such slot.
        slot: u8,
    },
    /// The notification raised with no slot released: a release raises it with the slot, and a
    /// refresh clears both.
    #[error(
        "the present table snapshot's client {index} has its notification raised with no slot released"
    )]
    RaisedWithoutRelease {
        /// The client's index.
        index: usize,
    },
    /// A slot released with the notification not raised: a release raises it with the slot,
    /// and a refresh clears both.
    #[error(
        "the present table snapshot's client {index} has a slot released with its notification not raised"
    )]
    ReleasedNotRaised {
        /// The client's index.
        index: usize,
    },
    /// A held reply with at least as many frames ahead of it as are queued: its own frame is
    /// queued behind them until the tick that completes it.
    #[error(
        "the present table snapshot's client {index} holds a reply behind {ahead} frames, with only {queued} queued"
    )]
    Ahead {
        /// The client's index.
        index: usize,
        /// The frames ahead it holds.
        ahead: u32,
        /// The frames queued.
        queued: usize,
    },
    /// A held reply's token that an earlier client's has.
    #[error(
        "the present table snapshot's client {index} holds reply token {token}, which an earlier client holds"
    )]
    DuplicateToken {
        /// The client's index.
        index: usize,
        /// The token id.
        token: u64,
    },
    /// A held reply's request the vi2 server does not hold as pending.
    #[error(
        "the present table snapshot's client {index} holds reply token {token}, which the vi2 server does not hold as pending"
    )]
    NotPending {
        /// The client's index.
        index: usize,
        /// The token id.
        token: u64,
    },
    /// A syncpoint whose unreached frames and armed waits outnumber its waiters: each holds one
    /// watch.
    #[error(
        "the present table snapshot has {frames} unreached frames on syncpoint {id}, which with its {waits} armed waits outnumber its {waiters} waiters"
    )]
    FramesOutnumberWatchers {
        /// The syncpoint.
        id: u8,
        /// Its unreached frames.
        frames: usize,
        /// Its armed waits.
        waits: usize,
        /// Its waiter count.
        waiters: u32,
    },
}

/// The snapshot header.
#[derive(Serialize, Deserialize)]
struct Header {
    /// [`SNAPSHOT_MAGIC`].
    magic: [u8; 4],
    /// [`SNAPSHOT_VERSION`].
    version: u32,
}

/// A frame as it is saved (module docs, "What is saved"). The field order is the layout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SavedFrame {
    pub(crate) slot: u8,
    pub(crate) image: u32,
    pub(crate) address: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pitch: u32,
    pub(crate) block_height_log2: u32,
    pub(crate) format: u32,
    pub(crate) shown: [u32; 4],
    pub(crate) layer: u64,
    pub(crate) display: u64,
    pub(crate) fence: Option<(u8, u32)>,
    pub(crate) reached: bool,
    pub(crate) holds_slot: bool,
}

impl From<&Frame> for SavedFrame {
    fn from(frame: &Frame) -> Self {
        let request = &frame.request;
        Self {
            slot: frame.slot,
            image: frame.image,
            address: request.image.address,
            width: request.image.width,
            height: request.image.height,
            pitch: request.image.pitch,
            block_height_log2: request.image.block_height_log2,
            format: request.image.format,
            shown: [
                request.shown.x,
                request.shown.y,
                request.shown.width,
                request.shown.height,
            ],
            layer: request.layer.0,
            display: request.display.0,
            fence: frame.fence.map(|fence| (fence.id.get(), fence.value)),
            reached: frame.reached,
            holds_slot: frame.holds_slot,
        }
    }
}

impl SavedFrame {
    /// The frame it holds, once its fence's id is checked.
    fn frame(&self, index: usize) -> Result<Frame, PresentSnapshotError> {
        let fence = match self.fence {
            None => None,
            Some((id, value)) => Some(Fence {
                id: SyncpointId::new(id)
                    .ok_or(PresentSnapshotError::FenceSyncpoint { index, id })?,
                value,
            }),
        };
        let [x, y, width, height] = self.shown;
        Ok(Frame {
            slot: self.slot,
            image: self.image,
            request: PresentRequest {
                image: ImageDesc {
                    address: self.address,
                    width: self.width,
                    height: self.height,
                    pitch: self.pitch,
                    block_height_log2: self.block_height_log2,
                    format: self.format,
                },
                shown: Rect {
                    x,
                    y,
                    width,
                    height,
                },
                layer: LayerId(self.layer),
                display: DisplayId(self.display),
            },
            fence,
            reached: self.reached,
            holds_slot: self.holds_slot,
        })
    }
}

/// A held reply as it is saved.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SavedWaiting {
    pub(crate) token: Option<TokenId>,
    pub(crate) ahead: u32,
}

/// A client as it is saved. The field order is the layout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SavedClient {
    pub(crate) number: u8,
    pub(crate) token: u64,
    pub(crate) available_bit: u32,
    pub(crate) event: Option<ObjectId>,
    pub(crate) queued: Vec<SavedFrame>,
    pub(crate) shown: Option<SavedFrame>,
    pub(crate) released: u64,
    pub(crate) raised: bool,
    pub(crate) waiting: Option<SavedWaiting>,
}

/// A client that passed its checks, its held reply's token id still to be reissued.
struct Checked {
    number: u8,
    client: PresentClient,
    reply: Option<(TokenId, usize)>,
}

/// The checks of one client (module docs, "What is saved"), `previous` the number before it;
/// `pending` says whether the vi2 server holds a token id as pending, `seen` the reply tokens of
/// earlier clients.
fn check_client(
    index: usize,
    saved: &SavedClient,
    previous: Option<u8>,
    seen: &mut BTreeSet<TokenId>,
    pending: &dyn Fn(TokenId) -> bool,
) -> Result<Checked, PresentSnapshotError> {
    if usize::from(saved.number) >= MAX_PRESENT_CLIENTS {
        let number = saved.number;
        return Err(PresentSnapshotError::ClientNumber { index, number });
    }
    if previous.is_some_and(|previous| previous >= saved.number) {
        return Err(PresentSnapshotError::NumbersNotAscending { index });
    }
    if saved.token == 0 {
        return Err(PresentSnapshotError::ZeroToken { index });
    }
    if saved.available_bit >= CLIENT_NOTIFICATIONS {
        return Err(PresentSnapshotError::NotificationBit { index });
    }
    if saved.queued.len() > usize::from(SLOTS) {
        let count = saved.queued.len();
        return Err(PresentSnapshotError::TooManyFrames { index, count });
    }
    let mut held = 0_u64;
    let mut queued = VecDeque::with_capacity(saved.queued.len());
    let shown_index = saved.queued.len();
    for (at, saved_frame) in saved.queued.iter().chain(&saved.shown).enumerate() {
        if saved_frame.slot >= SLOTS {
            let slot = saved_frame.slot;
            return Err(PresentSnapshotError::Slot { index, slot });
        }
        let frame = saved_frame.frame(index)?;
        if frame.fence.is_none() && !frame.reached {
            return Err(PresentSnapshotError::UnfencedUnreached { index });
        }
        if at == shown_index && !frame.reached {
            return Err(PresentSnapshotError::ShownUnreached { index });
        }
        if frame.holds_slot {
            let bit = slot_bit(frame.slot);
            if held & bit != 0 {
                let slot = frame.slot;
                return Err(PresentSnapshotError::SlotHeldTwice { index, slot });
            }
            held |= bit;
        }
        if at < shown_index {
            queued.push_back(frame);
        }
    }
    let shown = match &saved.shown {
        Some(frame) => Some(frame.frame(index)?),
        None => None,
    };
    check_released(index, saved, held)?;
    let mut reply = None;
    let mut waiting = None;
    if let Some(saved_waiting) = &saved.waiting {
        let ahead = usize::try_from(saved_waiting.ahead).unwrap_or(usize::MAX);
        if ahead >= queued.len() {
            return Err(PresentSnapshotError::Ahead {
                index,
                ahead: saved_waiting.ahead,
                queued: queued.len(),
            });
        }
        if let Some(token) = saved_waiting.token {
            if !seen.insert(token) {
                let token = token.raw();
                return Err(PresentSnapshotError::DuplicateToken { index, token });
            }
            if !pending(token) {
                let token = token.raw();
                return Err(PresentSnapshotError::NotPending { index, token });
            }
            reply = Some((token, ahead));
        } else {
            waiting = Some(Waiting { token: None, ahead });
        }
    }
    Ok(Checked {
        number: saved.number,
        client: PresentClient {
            token: saved.token,
            available_bit: saved.available_bit,
            event: saved.event,
            queued,
            shown,
            released: saved.released,
            raised: saved.raised,
            waiting,
        },
        reply,
    })
}

/// The checks of client `index`'s released slots (module docs, "What is saved"), in order:
/// within [`ALL_SLOTS`], none of them in `held` (the slots its queued and shown frames hold), and
/// the notification raised exactly when a slot is released.
fn check_released(
    index: usize,
    saved: &SavedClient,
    held: u64,
) -> Result<(), PresentSnapshotError> {
    if saved.released & !ALL_SLOTS != 0 {
        return Err(PresentSnapshotError::Released { index });
    }
    let both = saved.released & held;
    if both != 0 {
        let slot = u8::try_from(both.trailing_zeros()).unwrap_or(u8::MAX);
        return Err(PresentSnapshotError::ReleasedHeld { index, slot });
    }
    if saved.raised && saved.released == 0 {
        return Err(PresentSnapshotError::RaisedWithoutRelease { index });
    }
    if !saved.raised && saved.released != 0 {
        return Err(PresentSnapshotError::ReleasedNotRaised { index });
    }
    Ok(())
}

/// The relation checks [`crate::shared::Shared::save_presents`] and
/// [`crate::shared::Shared::restore_presents`] share, in the documented order: the clients it
/// holds, by number, with the reply tokens still to reissue.
fn check(
    saved: &[SavedClient],
    pending: &dyn Fn(TokenId) -> bool,
    watchers: &dyn Fn(SyncpointId) -> u32,
    waiters: &Waiters,
) -> Result<Vec<Checked>, PresentSnapshotError> {
    if saved.len() > MAX_PRESENT_CLIENTS {
        return Err(PresentSnapshotError::TooManyClients { count: saved.len() });
    }
    let mut seen = BTreeSet::new();
    let mut previous = None;
    let mut checked = Vec::with_capacity(saved.len());
    for (index, client) in saved.iter().enumerate() {
        checked.push(check_client(index, client, previous, &mut seen, pending)?);
        previous = Some(client.number);
    }
    for id in (0..=u8::MAX).map_while(SyncpointId::new) {
        let frames = checked
            .iter()
            .flat_map(|entry| entry.client.queued.iter())
            .filter(|frame| !frame.reached && frame.fence.is_some_and(|fence| fence.id == id))
            .count();
        if frames == 0 {
            continue;
        }
        let waits = waiters.on(id);
        let waiters = watchers(id);
        let held = frames.saturating_add(waits);
        if u32::try_from(held).map_or(true, |held| held > waiters) {
            return Err(PresentSnapshotError::FramesOutnumberWatchers {
                id: id.get(),
                frames,
                waits,
                waiters,
            });
        }
    }
    Ok(checked)
}

impl super::Shared {
    /// Writes vi2's presents (module docs, "What is saved"), with `server`, the vi2 server saved
    /// with it: a held reply whose request `server` no longer holds as pending is saved with no
    /// token. Take it while nothing changes either.
    pub fn save_presents(&self, server: &Server) -> Result<Vec<u8>, PresentSnapshotError> {
        let pending = |id: TokenId| server.pending().any(|(pending, _)| pending == id);
        let state = self.lock();
        let saved = state.presents.saved(pending);
        let watchers = |id: SyncpointId| state.watchers(id);
        check(&saved, &|_| true, &watchers, &state.waiters)?;
        drop(state);
        let mut bytes = postcard::to_allocvec(&Header {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
        })
        .map_err(PresentSnapshotError::Encode)?;
        bytes.extend(postcard::to_allocvec(&saved).map_err(PresentSnapshotError::Encode)?);
        Ok(bytes)
    }

    /// Replaces vi2's presents with what [`Self::save_presents`] wrote, after the checks of the
    /// module docs ("What is saved"), against the live waiter counts and armed waits and the
    /// restored vi2 `server`, which gives each held reply a new token. On an error the table is
    /// as it was. Restore it before the vsync clock (module docs, "Not checked here").
    pub fn restore_presents(
        &self,
        bytes: &[u8],
        server: &Server,
    ) -> Result<(), PresentSnapshotError> {
        let (header, rest) =
            postcard::take_from_bytes::<Header>(bytes).map_err(PresentSnapshotError::Decode)?;
        if header.magic != SNAPSHOT_MAGIC {
            return Err(PresentSnapshotError::BadMagic);
        }
        if header.version != SNAPSHOT_VERSION {
            return Err(PresentSnapshotError::UnsupportedVersion {
                found: header.version,
            });
        }
        let (saved, rest) = postcard::take_from_bytes::<Vec<SavedClient>>(rest)
            .map_err(PresentSnapshotError::Decode)?;
        if !rest.is_empty() {
            return Err(PresentSnapshotError::TrailingBytes { count: rest.len() });
        }
        let pending = |id: TokenId| server.pending().any(|(pending, _)| pending == id);
        let mut state = self.lock();
        let checked = {
            let watchers = |id: SyncpointId| state.watchers(id);
            check(&saved, &pending, &watchers, &state.waiters)?
        };
        let mut table = Presents::new();
        for Checked {
            number,
            mut client,
            reply,
        } in checked
        {
            if let Some((id, ahead)) = reply {
                // Every id was checked pending above, and nothing ran on the server since.
                client.waiting = Some(Waiting {
                    token: server.reissue(id),
                    ahead,
                });
            }
            table.clients.insert(number, client);
        }
        state.state.presents = table;
        drop(state);
        Ok(())
    }
}
