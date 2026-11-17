//! The shared syncpoint page gsv 204 binds: its header, the request and acknowledge protocol,
//! and the three answer triggers (`docs/specs/graphics-server.md`, "The shared syncpoint page
//! (gsv 204)"; port slice 5).
//!
//! Ported from (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project), cited below by these
//! short forms, as in [`crate::protocol`]:
//!
//! - **PR:** Eden PR #4588,
//!   `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97
//!   (`IGraphicsSupervisor::BindSyncpointMemory`, `RefreshSyncpointCache`) and
//!   `graphics_syncpoint_cache.h` @ e5fb3b97 (the offsets, `Header`, `CounterCount`,
//!   `GroupCount`, and the `static_assert`s ported as the `const` checks below).
//! - **NX2:** Eden-NX2, `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb
//!   (`INx2GraphicsSystemSession::AttachSyncptMemory`, `UpdateSyncptValues`, `PollSyncpts`), and
//!   its blocker log `eden-nx2/NX2_NOTES.md` @ 50fa86bb, whose rows "row N" names.
//!
//! # The page
//!
//! The title lends two transfer memories with 204: [`WRITABLE_SIZE`] bytes it writes, and
//! [`READ_ONLY_SIZE`] bytes it keeps only read access to (observed, rows 19 and 28; PR checks both
//! sizes). gsv writes [`header`] at the start of the read-only one: eight `u32`s, the version
//! ([`HEADER_VERSION`]), the base id ([`BASE_ID`]), the count ([`HEADER_COUNT`]), then the offsets
//! of the five areas, in PR's `Header` order:
//!
//! | Area | Memory | Offset | What |
//! |---|---|---|---|
//! | Values | read-only | [`VALUES_OFFSET`] | one `u64` per syncpoint, by id: the counter extended to 64 bits ([`starquake_gpu::syncpoint::CounterExtension`]) |
//! | Request summary | writable | [`GROUP_REQUEST_OFFSET`] | one `u64`; bit `g` asks about group `g` |
//! | Acknowledge summary | read-only | [`GROUP_ACK_OFFSET`] | one `u64` gsv writes |
//! | Per-slot requests | writable | [`SLOT_REQUEST_OFFSET`] | [`GROUP_COUNT`] `u64`s; bit `b` of word `g` asks about syncpoint `64g + b` |
//! | Per-slot acknowledges | read-only | [`SLOT_ACK_OFFSET`] | [`GROUP_COUNT`] `u64`s gsv writes |
//!
//! The header is self-describing, and the references put the areas at different places (NX2
//! `SyncptValuesOffset` and neighbours; PR `GraphicsSyncpointCache`), so the offsets are ours to
//! choose: these are PR's, as the spec says. The `const` checks below are PR's
//! `static_assert`s, with the header and the summary words added: no two areas overlap, and each
//! fits its memory (`the_areas_fit_their_memories_and_do_not_overlap` checks the same facts).
//!
//! **conflict: the count field.** NX2 writes the client's requested count, capped at
//! `(0x1000 - 0x40) / 8` (504); PR writes `CounterCount - 1` (191). This follows PR
//! ([`HEADER_COUNT`]): with PR's offsets, which this page uses, a count of 192 would make the
//! client's last value overlap the acknowledge summary word (the values area holds exactly
//! [`SYNCPOINT_COUNT`] values and ends at [`GROUP_ACK_OFFSET`]). NX2's comment gives the client's
//! check, `((count + 0x40) >> 6) - 1 < 0x40`; read as "the client's group count is
//! `(count + 0x40) >> 6`", which is a **guess**, PR's value gives exactly [`GROUP_COUNT`] and
//! NX2's would give one more (`the_header_count_names_exactly_the_page_s_groups`). The spec's
//! "What to verify before building", item 1 (which reference wins where they disagree), covers
//! it; a trace of the client's reads of the page settles it.
//!
//! # The protocol
//!
//! To ask for fresh values the title flips request bits so they differ from the acknowledge
//! bits, and signals the request event. gsv answers (PR `RefreshSyncpointCache`, step for
//! step): it reads the request summary and the acknowledge summary, and for each group whose
//! bits differ (ascending) it reads that group's request and acknowledge words; for each slot
//! whose bits differ (ascending) it writes the syncpoint's value, then (after a release fence) the
//! group's acknowledge word, a copy of its request word; after the last group (and a release
//! fence) the acknowledge summary, a copy of the request summary; and then it signals the answer
//! event (`an_answer_writes_values_then_each_group_s_acknowledge_then_the_summary_then_signals`).
//! A group whose summary bit does not differ is not looked at, whatever its slot bits say, as in
//! PR (`slot_bits_of_a_group_whose_summary_bit_matches_are_not_answered`). The fences order the
//! host's stores for the title's cores; no host test can observe them, so they are for review.
//!
//! A value is the counter's present value, read when the answer is written
//! (`Syncpoints::read`), extended to 64 bits against the value the page showed before
//! (`CounterExtension`, PR `CounterValue::Observe`). It is never a value drained from the
//! completion queue, which can be stale (`crate::completion`, "Drained syncpoint values are not
//! monotone"): the page shows the furthest-ahead value (owner's note on issue 183), so it never
//! goes backwards and no stale value is taken for a wrap
//! (`the_page_shows_the_counter_not_a_stale_drained_value`).
//!
//! # When a request is answered
//!
//! The spec answers a request "when the guest signals the request event, when a requested
//! counter changes, and at the latest 1 ms after the request", and says the bound "keeps a
//! request for a counter nobody increments from hanging" (decision 9). So a signal alone does not
//! answer a request whose counters have not moved; this module reads the three triggers so (a
//! reading of the spec's words, not a reference's behaviour: both references answer every
//! request on their poll). Whether the bound matters at all is the spec's "What to verify before
//! building", item 7 (the request rate, and how long requests wait without it).
//!
//! - The guest's signal of the request event reaches gsv through the kernel's hook (the
//!   completion queue is the hook: [`Completion::GuestSignalled`](crate::completion::Completion)),
//!   and the drain marks the page signalled ([`super::Shared::drain`]).
//! - The timer thread's page turn ([`super::Shared::tick_page`], called with the time after each
//!   drain) reads the request when the page was signalled or a request is held. With nothing
//!   asked (no summary bit differs) it does nothing, and drops any hold. Otherwise it watches
//!   every syncpoint asked about (the hold), reads their counters, and answers at once if one of
//!   them shows a value newer than the page shows (**on signal**:
//!   `a_request_for_a_counter_that_moved_is_answered_on_signal`), if no slot is asked about (a
//!   summary bit flipped alone: no counter could change), or if the hold is due. If not, it keeps
//!   the hold.
//! - A held syncpoint's increment is pushed to the completion queue, because the hold watches it
//!   (`crate::shared`, "The watched-id fast path"); that wakes the timer thread, whose next page
//!   turn finds the newer value and answers (**on change**:
//!   `a_held_request_is_answered_when_its_counter_changes`).
//! - A hold is due [`ANSWER_BOUND`] after the page turn that first held it (so a request is
//!   answered at most [`ANSWER_BOUND`] after the first turn that reads it): a turn at exactly
//!   that time answers, one a nanosecond before does not (**at the bound**:
//!   `a_held_request_is_answered_at_exactly_the_bound_and_not_before`). [`super::Shared::page_deadline`]
//!   says when that is, for the timer thread's next wake-up.
//!
//! Watching before reading means no increment is missed: one before the read is in the value,
//! one after it is pushed. An answer, and a turn that finds nothing asked, give back every watch
//! of the hold (`answering_gives_back_the_hold_s_watches`). If a syncpoint cannot be watched (its
//! waiter count is at its maximum), the request is answered at once rather than held unwatched
//! (`a_request_that_cannot_be_watched_is_answered_at_once`).
//!
//! The page's memories are read and written through the [`TimerHost`]
//! ([`TimerHost::read_transfer_memory`], [`TimerHost::write_transfer_memory`]), which checks
//! each access against the transfer memory as it is now. A refused read drops the hold; a
//! refused write stops the answer where it is, leaving what was written, signals nothing, and
//! drops the hold; the next signal tries again
//! (`a_refused_access_drops_the_hold_and_the_next_signal_tries_again`). The kernel refuses once
//! the title has closed a transfer memory (`a_closed_transfer_memory_is_refused_by_the_kernel`):
//! PR keeps a reference to both (`Open(kernel)`), so it never meets this case. A host that does
//! not offer the two calls (their default) refuses every read, so a page bound in a run whose
//! timer host lacks them answers nothing (`a_host_without_transfer_memory_answers_nothing`).
//!
//! # Binding (204)
//!
//! [`bind`] is what the root's 204 does in a run with gsv's shared state ([`crate::root`]). In:
//! copy handles 0 and 1, the writable and the read-only transfer memory; `u32` version, `u32`
//! count, `u64` writable size, `u64` read-only size (PR's header check: `packet[1] ==
//! 0x80000006`, six data words, `packet[2] == 4`, two copied handles, then `1, 0x1ff, 0x1000, 0,
//! 0x2000, 0`; NX2 reads the first two as the requested version and the most syncpoints the
//! client wants). Out: copy handles to the request event's writable end, which the title
//! signals (row 28: "the first is attached writable"), and to the answer event. No data words.
//!
//! It checks, in this order, and answers the first failure: the two handles and the four data
//! words are there (the framework's `INVALID_CMIF_REQUEST`, the handles first); the page is not
//! bound already (`InvalidState`, PR `m_syncpoint_rw`); each handle names a transfer memory, and
//! not the same one (`InvalidHandle`, PR); their sizes are exactly [`WRITABLE_SIZE`] and
//! [`READ_ONLY_SIZE`] (`InvalidSize`, PR). The data words are read and, where they differ from
//! PR's, logged, not refused (they say nothing the transfer memories do not). Then it clears both
//! memories, writes the header and [`UNALLOCATED`] for every value (PR: zeros, `Header`,
//! `Unallocated`), makes the two events (the request event with the completion queue as its hook),
//! binds the page, and replies. An event made before a later failure is kept and handed out by
//! the next 204, so a failing 204 leaks nothing.
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `shared/page/tests.rs`:
//!
//! | Case | PR | NX2 | Here |
//! |---|---|---|---|
//! | When a request is answered | every 1 ms, all requests | every 500 µs, all requests | on signal or change if a counter moved, else at the 1 ms bound (above) |
//! | Header count | `CounterCount - 1` | the client's count, capped | **conflict**, PR (`the_header_is_pinned`) |
//! | Values at bind | `Unallocated` | the counters | `Unallocated`, as PR: what it shows is not a counter read, so there is nothing to record (`bind_clears_the_memories_and_writes_the_header_and_unallocated_values`) |
//! | The value of a syncpoint that is not allocated | `Unallocated` (`ReadAllocatedSyncpoint`) | the counter | the counter, as NX2 and as syncpt 103 reads it (`crate::syncpt`, "Where this differs"): nothing allocates ids before nvsrv's channels (`an_unallocated_syncpoint_s_counter_is_shown`) |
//! | Header words of the request other than PR's | `InvalidCombination` | read as they come | read, logged (`bind_reads_any_data_words`) |
//! | The syncpoint context (201) not initialised, or no NVIDIA driver yet | `InvalidState` | accepted | accepted: the root keeps no such state (201 is a logged stub) |
//! | A transfer memory whose owner is not the bound process | `InvalidHandle` | accepted | accepted: the kernel's `IpcContext` takes the sender as the owner (PROVISIONAL there) |
//! | A second 204 | `InvalidState` | rebinds to the new memories, the same events | `InvalidState`, as PR (`a_second_bind_is_refused_and_changes_nothing`) |
//! | Read-only memory larger than [`READ_ONLY_SIZE`] | `InvalidSize` | accepted | `InvalidSize`, as PR (`each_size_is_checked_at_its_edge`) |
//! | Handle kind of the two events | moved | copied | **conflict:** copied, as [`crate::protocol::HANDLE_KIND`] |
//! | The request event after an answer | cleared | left as it is | left as it is: the title holds only its writable end, so whether it is signalled is not visible to it |
//! | The transfer memories | referenced while bound | not referenced | not referenced: see "When a request is answered" |
//!
//! # What is recorded
//!
//! Each value an answer writes is recorded as a `SyncptPageAnswered` note ([`crate::note`]): the
//! syncpoint and the counter value read for it, in the order written, each after its write
//! (`each_written_value_is_recorded_after_its_write`). The 64-bit value written follows from the
//! counter and the page's saved state, and the acknowledge words are copies of what the title
//! wrote, so replay can rebuild them. Nothing else a page turn reads comes from outside the
//! guest: a turn that holds rather than answers found every counter equal to what the page
//! shows. The turn's time is the timer thread's, from the `Env` clock.
//!
//! # What is saved
//!
//! [`Shared::save_page`](super::Shared::save_page) writes the page's state: its events, its
//! transfer memories ("the shared page's location"), the values it shows, the hold and whether
//! a signal is waiting for the next turn. `postcard`, after a header of [`SNAPSHOT_MAGIC`] and
//! [`SNAPSHOT_VERSION`] that is decoded and checked before the body; the golden-bytes test
//! `the_page_snapshot_layout_is_pinned` pins it. The spec also lists "the last request bits
//! answered": they are the acknowledge words, which live in the read-only memory itself and are
//! saved with the title's memory, so this snapshot does not hold a second copy.
//!
//! A snapshot is a file and is not trusted. [`Shared::restore_page`](super::Shared::restore_page)
//! checks, in this order, and returns the first failure: the magic, the version, that the body
//! decodes, that no bytes follow it, then the page's own relations ([`PageStateError`], in the
//! order of its variants), then that each syncpoint's waiter count is its armed waits plus one
//! if the hold watches it ([`PageSnapshotError::Watchers`]: the page and the armed waits are the
//! only watchers), and last that the kernel takes the request event's hook back
//! ([`Rehook`]; owner's ruling of 2026-11-17: hooks are re-attached by their owner in its
//! restore step, checked against the restored events, before the guest runs again). Only then is
//! the page replaced. Saving runs the relation and watcher checks on what it is about to write.

use std::sync::Arc;
use std::sync::atomic::{Ordering, fence};

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::{SYNCPOINT_COUNT, SyncpointId};
use starquake_gpu::syncpoint::{CounterExtension, UNALLOCATED, extend_counter};
use starquake_kernel::dispatch::Dispatcher;
use starquake_kernel::id::ObjectId;
use starquake_kernel::ipc::EventHook;
use starquake_kernel::result::{KernelError, SvcResult};
use starquake_services::framework::{Context, Outcome, Reply, Request};
use starquake_trace::{HostNanos, ServiceId};
use thiserror::Error;

use super::{COUNT, Locked, Shared, TimerHost};
use crate::completion::CompletionQueue;
use crate::note::Note;

#[cfg(test)]
mod tests;

/// The header's version word (PR `Header[0]`, NX2 `// version`; the client requires 1: NX2's
/// comment on `AttachSyncptMemory`).
pub const HEADER_VERSION: u32 = 1;

/// The header's base id word: the id of the first value (PR `Header[1]` and NX2 both write 0;
/// NX2 marks it "guess").
pub const BASE_ID: u32 = 0;

/// The header's count word: [`SYNCPOINT_COUNT`] - 1 (PR `CounterCount - 1`). A **conflict**
/// with NX2 (module docs, "The page").
pub const HEADER_COUNT: u32 = SYNCPOINT_COUNT - 1;

/// Where the values start in the read-only memory (PR `ValuesOffset`).
pub const VALUES_OFFSET: u32 = 0x40;

/// Where the request summary word is in the writable memory (PR `GroupRequestOffset`).
pub const GROUP_REQUEST_OFFSET: u32 = 0;

/// Where the acknowledge summary word is in the read-only memory (PR `GroupAckOffset`).
pub const GROUP_ACK_OFFSET: u32 = 0x640;

/// Where the per-slot request words start in the writable memory (PR `SlotRequestOffset`).
pub const SLOT_REQUEST_OFFSET: u32 = 8;

/// Where the per-slot acknowledge words start in the read-only memory (PR `SlotAckOffset`).
pub const SLOT_ACK_OFFSET: u32 = 0x648;

/// Syncpoints per group: the bits of one request or acknowledge word.
pub const GROUP_BITS: usize = u64::BITS as usize;

/// The number of groups (PR `GroupCount = (CounterCount + 63) / 64`).
pub const GROUP_COUNT: usize = COUNT.div_ceil(GROUP_BITS);

/// The size of the memory the title writes (PR `rw->GetSize() != 0x1000`; rows 19, 28).
pub const WRITABLE_SIZE: u32 = 0x1000;

/// The size of the memory the title only reads (PR `ro->GetSize() != 0x2000`; rows 19, 28).
pub const READ_ONLY_SIZE: u32 = 0x2000;

/// The header's eight words, in bytes.
pub const HEADER_SIZE: usize = 8 * WORD;

/// The version word of PR's accepted 204 request (`packet[5] != 1`).
pub const REQUESTED_VERSION: u32 = 1;

/// The count word of PR's accepted 204 request (`packet[6] != 0x1ff`), which NX2 reads as the most
/// syncpoints the client wants.
pub const REQUESTED_COUNT: u32 = 0x1FF;

/// How long a held request waits at most, in host nanoseconds (spec decision 9: "at most 1 ms
/// stale"; PR polls every `std::chrono::milliseconds{1}`).
pub const ANSWER_BOUND: u64 = 1_000_000;

/// First bytes of every page snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SQPG";

/// Layout version written into every page snapshot and required when reading one.
pub const SNAPSHOT_VERSION: u32 = 1;

/// A `u32` in bytes.
const WORD: usize = 4;

/// A `u64` in bytes.
const DOUBLE: u32 = 8;

/// The summary bits that name a group.
const GROUP_MASK: u64 = (1 << GROUP_COUNT) - 1;

// PR's `static_assert`s, and the header and summary words besides: each area fits its memory and
// no two overlap.
const _: () = assert!(HEADER_SIZE <= VALUES_OFFSET as usize);
const _: () =
    assert!(VALUES_OFFSET as usize + COUNT * DOUBLE as usize <= GROUP_ACK_OFFSET as usize);
const _: () = assert!(GROUP_ACK_OFFSET + DOUBLE <= SLOT_ACK_OFFSET);
const _: () =
    assert!(SLOT_ACK_OFFSET as usize + GROUP_COUNT * DOUBLE as usize <= READ_ONLY_SIZE as usize);
const _: () = assert!(GROUP_REQUEST_OFFSET + DOUBLE <= SLOT_REQUEST_OFFSET);
const _: () =
    assert!(SLOT_REQUEST_OFFSET as usize + GROUP_COUNT * DOUBLE as usize <= WRITABLE_SIZE as usize);
// Every bit of every group names a syncpoint, so a set of ids ([`Ids`]) needs no range check.
const _: () = assert!(GROUP_COUNT * GROUP_BITS == COUNT);

/// The header gsv writes at the start of the read-only memory, in PR's `Header` order (module
/// docs, "The page").
#[must_use]
pub fn header() -> [u8; HEADER_SIZE] {
    let words = [
        HEADER_VERSION,
        BASE_ID,
        HEADER_COUNT,
        VALUES_OFFSET,
        GROUP_REQUEST_OFFSET,
        GROUP_ACK_OFFSET,
        SLOT_REQUEST_OFFSET,
        SLOT_ACK_OFFSET,
    ];
    let mut bytes = [0; HEADER_SIZE];
    for (chunk, word) in bytes.as_chunks_mut::<WORD>().0.iter_mut().zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Where syncpoint `id`'s value is in the read-only memory.
#[must_use]
pub fn value_offset(id: SyncpointId) -> u64 {
    u64::from(VALUES_OFFSET) + u64::from(id.get()) * u64::from(DOUBLE)
}

/// Where group `group`'s word is in an area of per-slot words that starts at `start`.
fn group_offset(start: u32, group: usize) -> u64 {
    // `group` is below `GROUP_COUNT` (3), so the cast is exact and the sum cannot overflow.
    u64::from(start) + group as u64 * u64::from(DOUBLE)
}

/// A set of syncpoints, one bit each, laid out as the page lays out its per-slot bits: syncpoint
/// `64g + b` is bit `b` of word `g`. Every bit names a syncpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ids([u64; GROUP_COUNT]);

impl Ids {
    /// The set of `ids`.
    #[must_use]
    pub fn of(ids: impl IntoIterator<Item = SyncpointId>) -> Self {
        let mut set = Self::default();
        for id in ids {
            let (group, bit) = place(id);
            set.0[group] |= 1 << bit;
        }
        set
    }

    /// Whether `id` is in the set.
    #[must_use]
    pub fn contains(&self, id: SyncpointId) -> bool {
        let (group, bit) = place(id);
        self.0[group] & (1 << bit) != 0
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }

    /// The syncpoints in the set, in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = SyncpointId> + '_ {
        (0..=u8::MAX)
            .map_while(SyncpointId::new)
            .filter(|id| self.contains(*id))
    }

    /// The syncpoints of group `group` in the set, in ascending order.
    fn in_group(&self, group: usize) -> impl Iterator<Item = SyncpointId> + '_ {
        self.iter().filter(move |id| place(*id).0 == group)
    }
}

/// The group and bit of syncpoint `id`.
fn place(id: SyncpointId) -> (usize, usize) {
    let index = usize::from(id.get());
    (index / GROUP_BITS, index % GROUP_BITS)
}

/// The two ends of the request event: the writable end the title signals, and its readable end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestEvent {
    /// The writable end, handed to the title, hooked.
    pub writable: ObjectId,
    /// The readable end.
    pub readable: ObjectId,
}

/// The page's two transfer memories.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    /// The memory the title writes ([`WRITABLE_SIZE`] bytes).
    pub writable: ObjectId,
    /// The memory the title only reads ([`READ_ONLY_SIZE`] bytes).
    pub read_only: ObjectId,
}

/// A request gsv holds until a counter it asks about moves or the bound passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hold {
    /// The syncpoints asked about, each watched.
    pub ids: Ids,
    /// The page turn that first held it.
    pub since: HostNanos,
}

impl Hold {
    /// When it is due: [`ANSWER_BOUND`] after it was first held, or the end of time if that is
    /// past the clock's range.
    #[must_use]
    pub fn due(&self) -> HostNanos {
        HostNanos(self.since.0.saturating_add(ANSWER_BOUND))
    }
}

/// The shared syncpoint page's state in gsv's shared state (module docs). Built unbound, bound
/// only by [`bind`], and restored only through the checks of
/// [`Shared::restore_page`](super::Shared::restore_page).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page {
    /// The request event, once made.
    request: Option<RequestEvent>,
    /// The answer event, once made.
    answer: Option<ObjectId>,
    /// The two memories, once bound.
    binding: Option<Binding>,
    /// The value the page shows for each syncpoint.
    shown: CounterExtension,
    /// The request held, if one is.
    held: Option<Hold>,
    /// Whether the title signalled the request event since the last page turn.
    signalled: bool,
}

impl Page {
    /// The request event, once 204 has made it.
    #[must_use]
    pub const fn request_event(&self) -> Option<RequestEvent> {
        self.request
    }

    /// The answer event, once 204 has made it.
    #[must_use]
    pub const fn answer_event(&self) -> Option<ObjectId> {
        self.answer
    }

    /// The two memories, once 204 has bound them.
    #[must_use]
    pub const fn binding(&self) -> Option<Binding> {
        self.binding
    }

    /// The request held, if one is.
    #[must_use]
    pub const fn held(&self) -> Option<Hold> {
        self.held
    }

    /// Whether a signal waits for the next page turn.
    #[must_use]
    pub const fn is_signalled(&self) -> bool {
        self.signalled
    }

    /// The value the page shows for syncpoint `id`.
    #[must_use]
    pub fn shown(&self, id: SyncpointId) -> u64 {
        self.shown.value(id)
    }

    /// Notes that the guest signalled `event`, if it is the bound page's request event: the next
    /// page turn reads the request ([`super::Shared::drain`]).
    pub(super) fn guest_signalled(&mut self, event: ObjectId) {
        let ours = self
            .request
            .is_some_and(|request| request.writable == event);
        if ours && self.binding.is_some() {
            self.signalled = true;
        }
    }

    /// Whether counter `id`, now `current`, would show a value other than the page shows.
    fn is_fresh(&self, id: SyncpointId, current: u32) -> bool {
        let shown = self.shown.value(id);
        extend_counter(shown, current) != shown
    }

    /// The page's own relations, in the order of [`PageStateError`]'s variants.
    fn check(&self) -> Result<(), PageStateError> {
        if let Some(request) = self.request {
            if request.writable == request.readable {
                return Err(PageStateError::RequestEnds);
            }
            if self
                .answer
                .is_some_and(|answer| answer == request.writable || answer == request.readable)
            {
                return Err(PageStateError::SharedEvent);
            }
        }
        if let Some(binding) = self.binding {
            if self.request.is_none() || self.answer.is_none() {
                return Err(PageStateError::BoundWithoutEvents);
            }
            if binding.writable == binding.read_only {
                return Err(PageStateError::SameMemory);
            }
        } else {
            let mut ids = (0..=u8::MAX).map_while(SyncpointId::new);
            if let Some(id) = ids.find(|id| self.shown(*id) != UNALLOCATED) {
                return Err(PageStateError::ShownWhileUnbound { id: id.get() });
            }
            if self.held.is_some() {
                return Err(PageStateError::HeldWhileUnbound);
            }
            if self.signalled {
                return Err(PageStateError::SignalledWhileUnbound);
            }
        }
        if self.held.is_some_and(|held| held.ids.is_empty()) {
            return Err(PageStateError::EmptyHold);
        }
        Ok(())
    }
}

/// A relation between the page's fields that no sequence of 204s and page turns leads to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum PageStateError {
    /// The request event's two ends are one object.
    #[error("the request event's two ends are the same object")]
    RequestEnds,
    /// The answer event is an end of the request event.
    #[error("the answer event is an end of the request event")]
    SharedEvent,
    /// The page is bound but an event is missing: 204 makes both before it binds.
    #[error("the page is bound without its two events")]
    BoundWithoutEvents,
    /// The two memories are one object: 204 refuses that.
    #[error("the page's two memories are the same object")]
    SameMemory,
    /// An unbound page shows a value: only an answer changes what it shows.
    #[error("an unbound page shows a value for syncpoint {id}")]
    ShownWhileUnbound {
        /// The syncpoint.
        id: u8,
    },
    /// An unbound page holds a request.
    #[error("an unbound page holds a request")]
    HeldWhileUnbound,
    /// An unbound page has a signal waiting.
    #[error("an unbound page has a signal waiting")]
    SignalledWhileUnbound,
    /// A hold asks about no syncpoint: such a request is answered at once.
    #[error("the page holds a request for no syncpoint")]
    EmptyHold,
}

/// Why the page's snapshot could not be written or read.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum PageSnapshotError {
    /// Encoding failed.
    #[error("could not encode the syncpoint page snapshot: {0}")]
    Encode(postcard::Error),
    /// The header or the body is not a well-formed encoding.
    #[error("could not decode the syncpoint page snapshot: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`SNAPSHOT_MAGIC`].
    #[error("not a syncpoint page snapshot (bad magic)")]
    BadMagic,
    /// The snapshot was written with another layout version.
    #[error(
        "syncpoint page snapshot layout version {found} is not supported (this build reads {SNAPSHOT_VERSION})"
    )]
    UnsupportedVersion {
        /// The version found.
        found: u32,
    },
    /// Bytes follow the body.
    #[error("{count} unexpected bytes after the syncpoint page snapshot")]
    TrailingBytes {
        /// How many.
        count: usize,
    },
    /// The page's fields do not fit together.
    #[error("the syncpoint page snapshot: {0}")]
    State(#[from] PageStateError),
    /// A syncpoint's waiter count is not its armed waits plus the hold's watch.
    #[error(
        "syncpoint {id} has {watchers} waiters, but its armed waits and the page's hold need {needed}"
    )]
    Watchers {
        /// The syncpoint.
        id: u8,
        /// Its waiter count.
        watchers: u32,
        /// Its armed waits, plus one if the hold watches it.
        needed: usize,
    },
    /// The kernel refused to re-attach the request event's hook.
    #[error("the kernel refused to re-attach the syncpoint page's request hook: {0}")]
    Rehook(KernelError),
}

/// How a restore re-attaches the request event's hook (module docs, "What is saved"): the kernel
/// checks the event against what it holds. The kernel's [`Dispatcher`] implements it.
pub trait Rehook {
    /// Attaches `hook` to `event`, the writable end of an event the server of `service` made, or
    /// says why the kernel refuses ([`Dispatcher::reattach_event_hook`]).
    fn reattach(
        &self,
        service: ServiceId,
        event: ObjectId,
        hook: Arc<dyn EventHook>,
    ) -> SvcResult<()>;
}

impl Rehook for Dispatcher {
    /// [`Dispatcher::reattach_event_hook`].
    fn reattach(
        &self,
        service: ServiceId,
        event: ObjectId,
        hook: Arc<dyn EventHook>,
    ) -> SvcResult<()> {
        self.reattach_event_hook(service, event, hook)
    }
}

/// The snapshot header.
#[derive(Serialize, Deserialize)]
struct Header {
    /// [`SNAPSHOT_MAGIC`].
    magic: [u8; 4],
    /// [`SNAPSHOT_VERSION`].
    version: u32,
}

/// What a page turn read: the request summary, and for each group whose summary bit differs
/// from the acknowledge summary's, its request word and the slots whose bits differ.
struct Asked {
    /// The request summary word.
    summary: u64,
    /// For each group whose summary bit differs, its request word.
    groups: [Option<u64>; GROUP_COUNT],
    /// The slots, in those groups, whose request bit differs from the acknowledge bit.
    ids: Ids,
}

impl Asked {
    /// Whether anything is asked: a summary bit differs.
    fn is_pending(&self) -> bool {
        self.groups.iter().any(Option::is_some)
    }
}

/// A `u64` read from `memory` at `offset` through `host`.
fn read_u64(host: &mut dyn TimerHost, memory: ObjectId, offset: u64) -> SvcResult<u64> {
    let mut bytes = [0; DOUBLE as usize];
    host.read_transfer_memory(memory, offset, &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// A `u64` written into `memory` at `offset` through `host`.
fn write_u64(host: &mut dyn TimerHost, memory: ObjectId, offset: u64, value: u64) -> SvcResult<()> {
    host.write_transfer_memory(memory, offset, &value.to_le_bytes())
}

/// What the title asks (module docs, "The protocol").
fn read_asked(host: &mut dyn TimerHost, binding: Binding) -> SvcResult<Asked> {
    let summary = read_u64(host, binding.writable, u64::from(GROUP_REQUEST_OFFSET))?;
    let acknowledged = read_u64(host, binding.read_only, u64::from(GROUP_ACK_OFFSET))?;
    let differ = (summary ^ acknowledged) & GROUP_MASK;
    let mut groups = [None; GROUP_COUNT];
    let mut ids = Ids::default();
    for (group, slot) in groups.iter_mut().enumerate() {
        if differ & (1 << group) == 0 {
            continue;
        }
        let requested = read_u64(
            host,
            binding.writable,
            group_offset(SLOT_REQUEST_OFFSET, group),
        )?;
        let acked = read_u64(
            host,
            binding.read_only,
            group_offset(SLOT_ACK_OFFSET, group),
        )?;
        *slot = Some(requested);
        ids.0[group] = requested ^ acked;
    }
    Ok(Asked {
        summary,
        groups,
        ids,
    })
}

impl Locked<'_> {
    /// The page's state.
    #[must_use]
    pub fn page(&self) -> &Page {
        &self.state.page
    }

    /// Makes the hold watch exactly `ids`, first held at `since` unless already held: watches
    /// the syncpoints newly asked about, then gives back the watches of those no longer asked
    /// about. On a watch that fails, every watch taken here is given back, the hold is as it
    /// was, and the error is returned.
    fn hold(&mut self, ids: Ids, since: HostNanos) -> Result<(), super::WatchError> {
        let before = self.state.page.held;
        let old = before.map_or_else(Ids::default, |held| held.ids);
        let added: Vec<SyncpointId> = ids.iter().filter(|id| !old.contains(*id)).collect();
        for (taken, id) in added.iter().enumerate() {
            if let Err(error) = self.watch(*id) {
                for given in added.iter().take(taken) {
                    self.give_back(*given);
                }
                return Err(error);
            }
        }
        for id in old.iter().filter(|id| !ids.contains(*id)) {
            self.give_back(id);
        }
        let since = before.map_or(since, |held| held.since);
        self.state.page.held = Some(Hold { ids, since });
        Ok(())
    }

    /// Drops the hold, giving back its watches.
    fn release_hold(&mut self) {
        if let Some(held) = self.state.page.held.take() {
            for id in held.ids.iter() {
                self.give_back(id);
            }
        }
    }

    /// Gives back the hold's watch of `id`. The hold always has one, so a refusal means gsv's
    /// own bookkeeping is wrong, and is logged.
    fn give_back(&mut self, id: SyncpointId) {
        if let Err(error) = self.unwatch(id) {
            tracing::error!(%error, "gsv: the syncpoint page's hold had no watch to give back");
        }
    }

    /// Answers `asked` (module docs, "The protocol"), recording each value written.
    fn answer(
        &mut self,
        host: &mut dyn TimerHost,
        binding: Binding,
        asked: &Asked,
        service: ServiceId,
    ) {
        if let Err(error) = self.write_answer(host, binding, asked, service) {
            tracing::warn!(%error, "gsv: the syncpoint page's memory refused the answer");
            return;
        }
        if let Some(answer) = self.state.page.answer
            && let Err(error) = host.signal_event(answer)
        {
            tracing::error!(%error, "gsv: the syncpoint page's answer event was not signalled");
        }
    }

    /// Writes the values, each group's acknowledge word and the acknowledge summary.
    fn write_answer(
        &mut self,
        host: &mut dyn TimerHost,
        binding: Binding,
        asked: &Asked,
        service: ServiceId,
    ) -> SvcResult<()> {
        for (group, requested) in asked.groups.iter().enumerate() {
            let Some(requested) = *requested else {
                continue;
            };
            for id in asked.ids.in_group(group) {
                let current = self.syncpoints.read(id);
                let value = extend_counter(self.state.page.shown.value(id), current);
                write_u64(host, binding.read_only, value_offset(id), value)?;
                self.state.page.shown.observe(id, Some(current));
                host.record(Note::SyncptPageAnswered { id, value: current }.event(service));
            }
            // PR: `std::atomic_thread_fence(std::memory_order_release)` before each group's
            // acknowledge word and before the summary.
            fence(Ordering::Release);
            write_u64(
                host,
                binding.read_only,
                group_offset(SLOT_ACK_OFFSET, group),
                requested,
            )?;
        }
        fence(Ordering::Release);
        write_u64(
            host,
            binding.read_only,
            u64::from(GROUP_ACK_OFFSET),
            asked.summary,
        )
    }

    /// The checks a page snapshot passes against this state, in the documented order (module
    /// docs, "What is saved"): the page's own relations, then each syncpoint's waiter count.
    fn check_page(&self, page: &Page) -> Result<(), PageSnapshotError> {
        page.check()?;
        let held = page.held.map_or_else(Ids::default, |held| held.ids);
        for id in (0..=u8::MAX).map_while(SyncpointId::new) {
            let needed = self
                .state
                .waiters
                .on(id)
                .saturating_add(usize::from(held.contains(id)));
            let watchers = self.watchers(id);
            if usize::try_from(watchers).ok() != Some(needed) {
                return Err(PageSnapshotError::Watchers {
                    id: id.get(),
                    watchers,
                    needed,
                });
            }
        }
        Ok(())
    }
}

impl Shared {
    /// The timer thread's page turn at `now`: call it after each [`Shared::drain`], with the
    /// time of that turn (module docs, "When a request is answered"). It takes gsv's shared lock
    /// and reaches the page's memories, the answer event and the event stream through `host`.
    pub fn tick_page(&self, host: &mut dyn TimerHost, now: HostNanos) {
        let mut state = self.lock();
        let Some(binding) = state.state.page.binding else {
            return;
        };
        let signalled = core::mem::take(&mut state.state.page.signalled);
        let held = state.state.page.held;
        if !signalled && held.is_none() {
            return;
        }
        let asked = match read_asked(host, binding) {
            Ok(asked) => asked,
            Err(error) => {
                tracing::warn!(%error, "gsv: the syncpoint page's memory refused a read");
                state.release_hold();
                return;
            }
        };
        if !asked.is_pending() {
            state.release_hold();
            return;
        }
        let due = held.is_some_and(|held| now >= held.due());
        let watched = !asked.ids.is_empty() && state.hold(asked.ids, now).is_ok();
        let fresh = asked
            .ids
            .iter()
            .any(|id| state.state.page.is_fresh(id, self.syncpoints.read(id)));
        if due || fresh || !watched {
            state.answer(host, binding, &asked, self.service);
            state.release_hold();
        }
    }

    /// When the held request is due, if one is held: the latest the timer thread may run its
    /// next page turn ([`Shared::tick_page`]).
    #[must_use]
    pub fn page_deadline(&self) -> Option<HostNanos> {
        self.lock().state.page.held.map(|held| held.due())
    }

    /// Writes the page's state (module docs, "What is saved"). Take it while nothing changes
    /// gsv's shared state.
    pub fn save_page(&self) -> Result<Vec<u8>, PageSnapshotError> {
        let state = self.lock();
        let page = state.state.page.clone();
        state.check_page(&page)?;
        drop(state);
        let mut bytes = postcard::to_allocvec(&Header {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
        })
        .map_err(PageSnapshotError::Encode)?;
        bytes.extend(postcard::to_allocvec(&page).map_err(PageSnapshotError::Encode)?);
        Ok(bytes)
    }

    /// Replaces the page with what [`Shared::save_page`] wrote, after the checks of the module
    /// docs ("What is saved"), the last of which re-attaches the request event's hook (the
    /// completion queue) through `kernel`, for the root's server, numbered `root`. Restore the
    /// rest of gsv's shared state first ([`Shared::restore`]), and run this before the guest
    /// runs again. On an error the page is as it was.
    pub fn restore_page(
        &self,
        bytes: &[u8],
        kernel: &dyn Rehook,
        root: ServiceId,
    ) -> Result<(), PageSnapshotError> {
        let (header, rest) =
            postcard::take_from_bytes::<Header>(bytes).map_err(PageSnapshotError::Decode)?;
        if header.magic != SNAPSHOT_MAGIC {
            return Err(PageSnapshotError::BadMagic);
        }
        if header.version != SNAPSHOT_VERSION {
            return Err(PageSnapshotError::UnsupportedVersion {
                found: header.version,
            });
        }
        let (page, rest) =
            postcard::take_from_bytes::<Page>(rest).map_err(PageSnapshotError::Decode)?;
        if !rest.is_empty() {
            return Err(PageSnapshotError::TrailingBytes { count: rest.len() });
        }
        let mut state = self.lock();
        state.check_page(&page)?;
        if let Some(request) = page.request {
            let queue: Arc<CompletionQueue> = Arc::clone(self.queue());
            let hook: Arc<dyn EventHook> = queue;
            kernel
                .reattach(root, request.writable, hook)
                .map_err(PageSnapshotError::Rehook)?;
        }
        state.state.page = page;
        Ok(())
    }
}

/// gsv 204 in a run with gsv's `shared` state (module docs, "Binding (204)").
pub(crate) fn bind(
    shared: &Shared,
    cx: &mut Context<'_>,
    req: &Request<'_>,
    reply: &mut Reply,
) -> Outcome {
    let writable = req.copy_handle(0)?;
    let read_only = req.copy_handle(1)?;
    let mut payload = req.payload();
    let version = payload.read_u32()?;
    let count = payload.read_u32()?;
    let writable_size = payload.read_u64()?;
    let read_only_size = payload.read_u64()?;
    let asked = (version, count, writable_size, read_only_size);
    let expected = (
        REQUESTED_VERSION,
        REQUESTED_COUNT,
        u64::from(WRITABLE_SIZE),
        u64::from(READ_ONLY_SIZE),
    );
    if asked != expected {
        tracing::warn!(
            version,
            count,
            writable_size,
            read_only_size,
            "gsv: 204 with data words PR refuses; read"
        );
    }
    let mut state = shared.lock();
    if state.state.page.binding.is_some() {
        tracing::debug!("gsv: 204 on a page already bound");
        return Err(KernelError::InvalidState.into());
    }
    let writable_memory = cx.transfer_memory(writable)?;
    let read_only_memory = cx.transfer_memory(read_only)?;
    if writable == read_only {
        tracing::debug!("gsv: 204 with one transfer memory for both");
        return Err(KernelError::InvalidHandle.into());
    }
    if writable_memory.size() != u64::from(WRITABLE_SIZE)
        || read_only_memory.size() != u64::from(READ_ONLY_SIZE)
    {
        tracing::debug!(
            writable = writable_memory.size(),
            read_only = read_only_memory.size(),
            "gsv: 204 with transfer memories of the wrong sizes"
        );
        return Err(KernelError::InvalidSize.into());
    }
    cx.write_transfer_memory(writable, 0, &[0; WRITABLE_SIZE as usize])?;
    cx.write_transfer_memory(read_only, 0, &[0; READ_ONLY_SIZE as usize])?;
    cx.write_transfer_memory(read_only, 0, &header())?;
    let unallocated: Vec<u8> = (0..COUNT).flat_map(|_| UNALLOCATED.to_le_bytes()).collect();
    cx.write_transfer_memory(read_only, u64::from(VALUES_OFFSET), &unallocated)?;
    // Each event is made once and kept, so a 204 that fails after making one hands that one
    // out next time (module docs, "Binding (204)").
    let request = if let Some(request) = state.state.page.request {
        request
    } else {
        let queue: Arc<CompletionQueue> = Arc::clone(shared.queue());
        let hook: Arc<dyn EventHook> = queue;
        let (writable, readable) = cx.create_hooked_event(hook)?;
        let request = RequestEvent { writable, readable };
        state.state.page.request = Some(request);
        request
    };
    let answer = if let Some(answer) = state.state.page.answer {
        answer
    } else {
        let answer = cx.create_event()?;
        state.state.page.answer = Some(answer);
        answer
    };
    state.state.page.binding = Some(Binding {
        writable,
        read_only,
    });
    drop(state);
    tracing::debug!(
        writable = writable_memory.address(),
        read_only = read_only_memory.address(),
        "gsv: bound the shared syncpoint page"
    );
    reply.copy_handle(request.writable);
    reply.copy_handle(answer);
    reply.ok()
}
