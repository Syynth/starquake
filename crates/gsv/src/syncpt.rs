//! The syncpt session gsv 202 returns: slot events, armed syncpoint waits and counter queries
//! (`docs/specs/graphics-server.md`, "Syncpoints and the syncpt session"; port slice 4).
//!
//! * [`waiters`]: the waiter table the armed waits live in, inside gsv's shared state.
//!
//! Ported from (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project), cited below by these
//! short forms, as in [`crate::protocol`]:
//!
//! - **NX2:** Eden-NX2, `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb
//!   (`INx2SyncptSession`), and its blocker log `eden-nx2/NX2_NOTES.md` @ 50fa86bb, whose rows
//!   "row N" names.
//! - **PR:** Eden PR #4588,
//!   `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97
//!   (`ISyncpointSupervisor`) and `graphics_supervisor_protocol.h` @ e5fb3b97
//!   (`DecodeSyncpointWait`, `IsSyncpointWaitSatisfied`).
//!
//! # The server
//!
//! [`syncpt_server`] is the syncpt family's server (decision 6: one server per family). Its
//! sessions are [`SyncptSession`]s; each keeps its own event count and slot events, as both
//! references keep them per session (PR `m_event_count`, `m_events`; NX2 `m_events`). The armed
//! waits are not the session's: they are in gsv's waiter table ([`crate::shared::GsvShared`]),
//! keyed by slot event, because the timer thread fires them ([`crate::shared::Shared::drain`]).
//! The family state ([`SyncptFamily`]) is the handle to gsv's shared state, nothing else.
//!
//! The root does not hand this server's sessions out yet: its 202 is still a logged stub
//! returning a placeholder ([`crate::root`], "Stubs"), as its 1402 is for nvsrv. Pointing 202 at
//! this server needs the server's number in the root, which the wiring (slice 16) gives.
//!
//! # Commands
//!
//! Every command is TIPC (PR `RegisterHandlersTipc`; NX2 `RegisterHandlersTipc`). Each reply is
//! a result word, then the data below. The request words follow the two TIPC header words, as
//! PR's header checks give them (`packet[0]` the type, `TIPC_COMMAND_OFFSET` plus the id;
//! `packet[1]` the data word count).
//!
//! | Id | Command | In | Out | Mark |
//! |---|---|---|---|---|
//! | 0 | [`HANDSHAKE`](command::HANDSHAKE) | `u64` token | the same `u64` | observed (row 43; NX2 `Handshake`; PR has no 0) |
//! | 1 | [`GET_MAXIMUM_EVENT_COUNT`](command::GET_MAXIMUM_EVENT_COUNT) | nothing | `u32` [`MAXIMUM_EVENT_COUNT`] | **conflict** (row 19; PR 63, NX2 32) |
//! | 21 | [`SET_EVENT_COUNT`](command::SET_EVENT_COUNT) | `u32` count | nothing | read (PR `SetEventCount`; NX2 has no 21) |
//! | 22 | [`GET_EVENT`](command::GET_EVENT) | `u32` slot | copy handle: the slot's event | observed (row 29; PR `GetEvent`, NX2 `GetSlotEvent`) |
//! | 103 | [`QUERY_COUNTER`](command::QUERY_COUNTER) | `u32` `SyncCounterId` | `u64` counter, `u32` 0 | read (PR and NX2 `QueryCounter`) |
//! | 201 | [`ARM_WAIT`](command::ARM_WAIT) | `u32` slot, `u32` `SyncCounterId`, `u64` threshold, `u32` mode | nothing | observed (rows 29, 147; PR `ArmWait`, `DecodeSyncpointWait`; NX2 `RequestSlotEvent`) |
//! | 202 | [`CANCEL_WAIT`](command::CANCEL_WAIT) | `u32` slot | nothing | observed (row 147; PR `CancelWait`, NX2 `CancelSlotEvent`) |
//!
//! A `SyncCounterId` is the bitwise NOT of the syncpoint id (NX2 `QueryCounter`: "the counter
//! is the bitwise NOT of the host1x syncpoint id"; PR `DecodeSyncpointWait`: `~packet[3]`;
//! [`syncpoint_of`]). 201's mode is the low byte of its word (PR `packet[6] & 0xff`,
//! [`mode_byte`]): 0 compares 32-bit wrapping, 1 as 64 bits ([`waiters::compare_mode`]).
//!
//! What each does, step by step:
//!
//! - **1** answers 63 ([`MAXIMUM_EVENT_COUNT`]): PR's `MaximumEventCount`, the size of its slot
//!   arrays; NX2 answers 32. The SDK requires a value below 0x400 and passes `min(value, 63)` to
//!   21 (NX2's comment on the class), so 63 is the larger answer, which contains NX2's (house
//!   rule, decision 8). Pinned by `get_maximum_event_count_answers_63`.
//! - **21** refuses a count above [`MAXIMUM_EVENT_COUNT`] with `InvalidCombination` (PR:
//!   `packet[2] <= MaximumEventCount`), changing nothing. Otherwise every slot from the count up
//!   to [`MAXIMUM_EVENT_COUNT`] loses its armed wait and its event, if it has one, is cleared
//!   (PR: `m_waits[slot].reset(); m_events[slot]->Clear(kernel)`), and the count is set. A
//!   session starts with a count of 0 (PR `m_event_count{}`).
//! - **22** refuses a slot at or above the count with `InvalidArgument` (PR); otherwise it
//!   makes the slot's event the first time any command needs it, and hands it out, copied.
//! - **103** answers the counter of the syncpoint, zero-extended to 64 bits (both: a `u32`
//!   counter pushed as a `u64`), and a `u32` 0 (PR: "Existing NVDRV counters use 32-bit modular
//!   comparisons"; NX2 pushes 0 there too). It reads the counter whether or not the id is
//!   allocated, and a `SyncCounterId` that names no syncpoint answers 0 (NX2).
//! - **201** checks, in this order, and answers the first failure: the mode
//!   (`InvalidCombination`, PR `DecodeSyncpointWait`), the slot (`InvalidArgument`, PR), and a
//!   64-bit wait whose threshold is above `u32::MAX` (`NotImplemented`, PR: no 32-bit counter
//!   reaches it). A `SyncCounterId` that names no syncpoint signals the slot's event at once
//!   (NX2: "out of range, signalling"), taking out any wait the slot had, before the threshold
//!   is looked at. Otherwise, under gsv's shared lock: the slot's event is cleared (row 147:
//!   "201 clears the slot's event first"), then the wait is armed in the waiter table,
//!   replacing the slot's earlier wait ([`crate::shared::Locked::arm`]). If the counter, read
//!   after its id is watched, already satisfies the wait, the value is pushed to the completion
//!   queue as a syncpoint change, so the timer thread fires the wait, signals the event and
//!   records the value, as it does for every value it acts on (spec, "What is recorded"). The
//!   handler never signals a reached wait itself and never blocks.
//! - **202** refuses a slot at or above the count with `InvalidArgument` (PR). Otherwise, under
//!   gsv's shared lock, it takes out the slot's wait, if any, and signals the slot's event,
//!   making it first if no command has (NX2: "a cancel signals it like a completion, also when
//!   nothing was pending"; row 147). It signals whether or not a wait was armed.
//!
//! A request that lacks a data word its command reads is the framework's `INVALID_CMIF_REQUEST`
//! (read before anything changes); one with more words is read. A TIPC command not in the table,
//! and a CMIF request, go to the server's unknown-command policy.
//!
//! # Why the lock is held across the clear
//!
//! 201 clears the event and arms the new wait with gsv's shared lock held, and 202 takes the
//! wait out and signals with it held, so the timer thread's drain (which fires waits under the
//! same lock) cannot fire the slot's old wait between the two steps. Without that, a change
//! drained between 201's clear and its arm would fire the replaced wait and leave the slot
//! signalled for a wait not yet reached: the bug of row 147 by another road. The kernel calls
//! are made with gsv's lock held, never the reverse (spec, "Who signals what": gsv's lock, then
//! the kernel's). Pinned by `arm_and_cancel_call_the_kernel_with_the_shared_lock_held`.
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `syncpt/tests.rs`:
//!
//! | Case | PR | NX2 | Here |
//! |---|---|---|---|
//! | 0 | not in the table: `ResultNotImplemented` | echoes the token | echoes the token (`the_handshake_echoes_the_token`) |
//! | 1's value | 63 | 32 | 63 (`get_maximum_event_count_answers_63`) |
//! | Handle kind of 22's event | moved | copied | **conflict:** copied, as [`crate::protocol::HANDLE_KIND`] (`get_event_hands_out_the_slot_s_event_copied`) |
//! | A slot at or above the count (22, 201, 202) | `ResultInvalidArgument` | accepted: any `u32` slot, an event made for each | `InvalidArgument` (`a_slot_at_the_count_is_refused_and_one_below_is_not`); NX2's reading would let a title make 2^32 events |
//! | 21 | as here | not in the table: auto-stubbed | as PR (`set_event_count_clears_the_slots_from_the_count_up`, `set_event_count_refuses_more_than_the_maximum`) |
//! | 103 or 201 on an id that is not allocated | `ResultNotFound` | the counter is read and waited on | as NX2 (`query_counter_reads_any_id`, `a_wait_on_an_unallocated_id_fires_on_its_counter`): the GPU reads and increments any id (`starquake_gpu::syncpoint`, "Divergences from Eden"), and nothing allocates ids before nvsrv's channels (slice 11) |
//! | 103 or 201 with a `SyncCounterId` that names no syncpoint | `ResultNotFound` | 103 answers 0; 201 signals at once | as NX2 (`a_counter_id_naming_no_syncpoint_reads_zero`, `a_wait_on_no_syncpoint_signals_at_once`); 201 also takes out the slot's earlier wait, which NX2 leaves armed |
//! | 201 with a mode byte above 1 | `ResultInvalidCombination` | accepted; the mode is only logged | as PR (`a_wait_with_an_unknown_mode_is_refused`) |
//! | 201 mode 1 with a threshold above `u32::MAX` | `ResultNotImplemented` | compares the low 32 bits | as PR (`a_wide_wait_above_any_counter_is_refused`) |
//! | 201 whose wait is reached when armed | signalled in the handler | signalled in the handler | signalled by the timer thread's next drain, which records the value (`a_wait_reached_when_armed_fires_on_the_next_drain`) |
//! | 202 on a slot with no event yet | nothing signalled (`if (m_events[...])`) | the event made and signalled | as NX2 (`cancel_makes_and_signals_an_event_no_command_made`) |
//! | Header words other than the reference's (`packet[0]`, `packet[1]`) | `ResultInvalidCombination` | read as they come | read as they come; only a missing data word is refused (`a_short_request_is_refused_and_a_longer_one_is_read`) |
//! | A closed session | its waits cancelled and its events closed (both destructors) | the same | its waits stay armed and its events stay made: the service framework tells a session nothing when it closes, so a closed session's waits fire into events no one waits on. Bounded by [`MAXIMUM_EVENT_COUNT`] per session |
//!
//! # What is recorded and what is saved
//!
//! Every request is one IPC event, recorded by the service framework with its payload, 103's
//! counter value included. The counter values gsv acts on (a wait fired on a change, or found
//! reached when armed) are recorded by the timer thread as `SyncpointObserved` notes
//! ([`crate::note`]) as it drains them. A session is plain data saved with the server: its event
//! count and slot events, checked as it is read back ([`SyncptSessionError`]). The armed waits are
//! saved with gsv's shared state ([`crate::shared`], "Snapshot format"). The family state saves
//! nothing ([`SyncptFamily`]).

pub mod waiters;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::{GpuCompletion, SyncpointId};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::KernelError;
use starquake_services::commands;
use starquake_services::framework::{
    Context, Family, Fault, Interfaces, Outcome, Reply, Request, Server, SnapshotError,
    UnknownPolicy,
};

use crate::completion::Completion;
use crate::shared::Shared;
use command::{
    ARM_WAIT, CANCEL_WAIT, GET_EVENT, GET_MAXIMUM_EVENT_COUNT, HANDSHAKE, QUERY_COUNTER,
    SET_EVENT_COUNT,
};
use waiters::{Wait, compare_mode};

#[cfg(test)]
mod tests;

/// The name of the syncpt server. Starquake's own: the session is reached only through gsv
/// 202, so it has no registered name in either reference (NX2 logs it as `"nx2:syncpt"`, PR as
/// `"ISyncpointSupervisor"`).
pub const SYNCPT_NAME: &str = "gsv:syncpt";

/// The syncpt session's TIPC command ids, each with its source.
pub mod command {
    /// NX2 `{0, &INx2SyncptSession::Handshake, "Handshake"}`; not in PR's table.
    pub const HANDSHAKE: u32 = 0;
    /// PR `{1, &ISyncpointSupervisor::GetMaximumEventCount, "GetMaximumEventCount"}`; NX2
    /// `"GetCount (guessed)"`.
    pub const GET_MAXIMUM_EVENT_COUNT: u32 = 1;
    /// PR `{21, &ISyncpointSupervisor::SetEventCount, "SetEventCount"}`; not in NX2's table.
    pub const SET_EVENT_COUNT: u32 = 21;
    /// PR `{22, &ISyncpointSupervisor::GetEvent, "GetEvent"}`; NX2 `"GetSlotEvent (guessed)"`.
    pub const GET_EVENT: u32 = 22;
    /// PR `{103, &ISyncpointSupervisor::QueryCounter, "QueryCounter"}`; NX2 `"QueryCounter"`.
    pub const QUERY_COUNTER: u32 = 103;
    /// PR `{201, &ISyncpointSupervisor::ArmWait, "ArmWait"}`; NX2 `"RequestSlotEvent
    /// (guessed)"`.
    pub const ARM_WAIT: u32 = 201;
    /// PR `{202, &ISyncpointSupervisor::CancelWait, "CancelWait"}`; NX2 `"CancelSlotEvent
    /// (guessed)"`.
    pub const CANCEL_WAIT: u32 = 202;
}

/// The number of event slots a session has, and what 1 answers: PR's `MaximumEventCount = 63`.
/// A **conflict**: NX2's `GetCount` answers 32 (module docs, "Commands").
pub const MAXIMUM_EVENT_COUNT: u32 = 63;

/// [`MAXIMUM_EVENT_COUNT`] as an index bound.
const SLOTS: usize = MAXIMUM_EVENT_COUNT as usize;

/// The syncpoint a `SyncCounterId` names: its bitwise NOT (NX2 `QueryCounter`, PR
/// `DecodeSyncpointWait`), or `None` if that is not below `SYNCPOINT_COUNT`.
#[must_use]
pub fn syncpoint_of(counter: u32) -> Option<SyncpointId> {
    u8::try_from(!counter).ok().and_then(SyncpointId::new)
}

/// The mode byte of 201's mode word: its low byte, the rest ignored (PR `DecodeSyncpointWait`
/// reads `packet[6] & 0xff`).
#[must_use]
pub fn mode_byte(word: u32) -> u8 {
    let [low, ..] = word.to_le_bytes();
    low
}

/// What every syncpt session shares: the handle to gsv's shared state, through which the
/// handlers read the syncpoints, arm and cancel waits, and push to the completion queue.
///
/// It is the run's wiring, not guest state, so a snapshot holds nothing of it: it saves no
/// bytes, and loading keeps the handle it has. That is why it implements [`Family`] itself
/// rather than through `serde`, which would have to make a handle from nothing.
pub struct SyncptFamily {
    shared: Arc<Shared>,
}

impl SyncptFamily {
    /// The family over gsv's `shared` state.
    #[must_use]
    pub fn new(shared: Arc<Shared>) -> Self {
        Self { shared }
    }

    /// gsv's shared state.
    #[must_use]
    pub fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }
}

impl Family for SyncptFamily {
    /// No bytes: the family holds only the run's wiring.
    fn save_state(&self) -> Result<Vec<u8>, SnapshotError> {
        Ok(Vec::new())
    }

    /// Accepts only what [`save_state`](Self::save_state) writes, and keeps the handle.
    fn load_state(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        if bytes.is_empty() {
            Ok(())
        } else {
            Err(SnapshotError::Decode(
                postcard::Error::DeserializeBadEncoding,
            ))
        }
    }
}

/// One syncpt session (module docs, "The server"): its event count and the event of each slot
/// that has one. Deserialising goes through `SavedSyncptSession`, which refuses a state no
/// sequence of commands leads to ([`SyncptSessionError`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SavedSyncptSession")]
pub struct SyncptSession {
    event_count: u32,
    events: Vec<Option<ObjectId>>,
}

/// The serialised form of [`SyncptSession`], field for field, and what deserialising validates.
#[derive(Deserialize)]
struct SavedSyncptSession {
    event_count: u32,
    events: Vec<Option<ObjectId>>,
}

/// Why a saved [`SyncptSession`] could not be read back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SyncptSessionError {
    /// The event count is above [`MAXIMUM_EVENT_COUNT`]: 21 refuses that.
    #[error(
        "a syncpt session's event count is {found}, above the maximum of {MAXIMUM_EVENT_COUNT}"
    )]
    EventCount {
        /// The count in the data.
        found: u32,
    },
    /// The data holds another number of slots than [`MAXIMUM_EVENT_COUNT`].
    #[error("a syncpt session holds {found} slots, not {MAXIMUM_EVENT_COUNT}")]
    Slots {
        /// How many it holds.
        found: usize,
    },
    /// Two slots hold one event: each slot's event is made for it.
    #[error("syncpt slots {first} and {second} hold the same event")]
    SharedEvent {
        /// The lower slot.
        first: usize,
        /// The higher slot.
        second: usize,
    },
}

impl TryFrom<SavedSyncptSession> for SyncptSession {
    type Error = SyncptSessionError;

    fn try_from(saved: SavedSyncptSession) -> Result<Self, Self::Error> {
        let SavedSyncptSession {
            event_count,
            events,
        } = saved;
        if event_count > MAXIMUM_EVENT_COUNT {
            return Err(SyncptSessionError::EventCount { found: event_count });
        }
        if events.len() != SLOTS {
            return Err(SyncptSessionError::Slots {
                found: events.len(),
            });
        }
        for (second, event) in events.iter().enumerate() {
            let Some(event) = event else { continue };
            if let Some(first) = events[..second].iter().position(|e| *e == Some(*event)) {
                return Err(SyncptSessionError::SharedEvent { first, second });
            }
        }
        Ok(Self {
            event_count,
            events,
        })
    }
}

impl Default for SyncptSession {
    fn default() -> Self {
        Self::new()
    }
}

impl SyncptSession {
    /// A session no command has reached: an event count of 0 (PR `m_event_count{}`), no events.
    #[must_use]
    pub fn new() -> Self {
        Self {
            event_count: 0,
            events: vec![None; SLOTS],
        }
    }

    /// The event count 21 set.
    #[must_use]
    pub const fn event_count(&self) -> u32 {
        self.event_count
    }

    /// The event of slot `slot`, if a command has made it.
    #[must_use]
    pub fn event(&self, slot: u32) -> Option<ObjectId> {
        let slot = usize::try_from(slot).ok()?;
        self.events.get(slot).copied().flatten()
    }

    /// The index of `slot`, if it is below the event count; `InvalidArgument` if not (PR
    /// `GetEvent`, `ArmWait`, `CancelWait`: `packet[2] < m_event_count`).
    fn slot(&self, slot: u32) -> Result<usize, Fault> {
        if slot < self.event_count {
            usize::try_from(slot).map_err(|_| KernelError::InvalidArgument.into())
        } else {
            tracing::debug!(
                slot,
                count = self.event_count,
                "gsv:syncpt: a slot at or above the event count"
            );
            Err(KernelError::InvalidArgument.into())
        }
    }

    /// The event of slot `index`, made now if no command has made it.
    fn slot_event(&mut self, cx: &mut Context<'_>, index: usize) -> Result<ObjectId, Fault> {
        let entry = self
            .events
            .get_mut(index)
            .ok_or(Fault::from(KernelError::InvalidArgument))?;
        if let Some(event) = *entry {
            return Ok(event);
        }
        let event = cx.create_event()?;
        *entry = Some(event);
        Ok(event)
    }
}

/// gsv's shared state, from the server's family.
fn shared(cx: &mut Context<'_>) -> Result<Arc<Shared>, Fault> {
    Ok(Arc::clone(cx.family::<SyncptFamily>()?.shared()))
}

commands! {
    impl SyncptSession {}
    tipc {
        HANDSHAKE => handshake,
        GET_MAXIMUM_EVENT_COUNT => get_maximum_event_count,
        SET_EVENT_COUNT => set_event_count,
        GET_EVENT => get_event,
        QUERY_COUNTER => query_counter,
        ARM_WAIT => arm_wait,
        CANCEL_WAIT => cancel_wait,
    }
}

impl SyncptSession {
    /// `Handshake` (0). in: `u64` token. out: the token (NX2: `TipcReplyWords(ctx, {w[2],
    /// w[3]})`; row 43: newer SDKs abort unless it is echoed).
    fn handshake(
        _: &mut Self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let token = req.payload().read_u64()?;
        tracing::debug!(token, "gsv:syncpt: handshake");
        reply.payload().write_u64(token);
        reply.ok()
    }

    /// `GetMaximumEventCount` (1). out: `u32` [`MAXIMUM_EVENT_COUNT`].
    fn get_maximum_event_count(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.payload().write_u32(MAXIMUM_EVENT_COUNT);
        reply.ok()
    }

    /// `SetEventCount` (21). in: `u32` count (module docs, "Commands").
    fn set_event_count(
        &mut self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let count = req.payload().read_u32()?;
        if count > MAXIMUM_EVENT_COUNT {
            tracing::debug!(count, "gsv:syncpt: an event count above the maximum");
            return Err(KernelError::InvalidCombination.into());
        }
        let first = usize::try_from(count).map_err(|_| KernelError::InvalidCombination)?;
        let shared = shared(cx)?;
        let mut state = shared.lock();
        let events: Vec<ObjectId> = self
            .events
            .get(first..)
            .unwrap_or_default()
            .iter()
            .flatten()
            .copied()
            .collect();
        self.event_count = count;
        // Every slot's wait is taken out and every event cleared, even after a refusal, so the
        // slots from the count up are left with no wait whatever the kernel says.
        let mut refusal = None;
        for event in events {
            state.disarm(event);
            if let Err(fault) = cx.clear_event(event) {
                refusal.get_or_insert(fault);
            }
        }
        drop(state);
        match refusal {
            Some(fault) => Err(fault),
            None => reply.ok(),
        }
    }

    /// `GetEvent` (22). in: `u32` slot. out: copy handle, the slot's event.
    fn get_event(&mut self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let index = self.slot(req.payload().read_u32()?)?;
        let event = self.slot_event(cx, index)?;
        reply.copy_handle(event);
        reply.ok()
    }

    /// `QueryCounter` (103). in: `u32` `SyncCounterId`. out: `u64` counter, `u32` 0.
    fn query_counter(
        _: &mut Self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let counter = req.payload().read_u32()?;
        let value = if let Some(id) = syncpoint_of(counter) {
            shared(cx)?.syncpoints().read(id)
        } else {
            tracing::debug!(counter, "gsv:syncpt: a counter id naming no syncpoint");
            0
        };
        tracing::trace!(counter, value, "gsv:syncpt: query counter");
        reply.payload().write_u64(u64::from(value)).write_u32(0);
        reply.ok()
    }

    /// `ArmWait` (201). in: `u32` slot, `u32` `SyncCounterId`, `u64` threshold, `u32` mode
    /// (module docs, "Commands", for the order of the checks).
    fn arm_wait(&mut self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let mut payload = req.payload();
        let slot = payload.read_u32()?;
        let counter = payload.read_u32()?;
        let threshold = payload.read_u64()?;
        let mode = mode_byte(payload.read_u32()?);
        tracing::debug!(slot, counter, threshold, mode, "gsv:syncpt: arm wait");
        let mode = compare_mode(mode).ok_or(KernelError::InvalidCombination)?;
        let index = self.slot(slot)?;
        let shared = shared(cx)?;
        let Some(id) = syncpoint_of(counter) else {
            tracing::warn!(
                counter,
                "gsv:syncpt: a wait on no syncpoint; signalled at once"
            );
            let event = self.slot_event(cx, index)?;
            let mut state = shared.lock();
            state.disarm(event);
            cx.signal_event(event)?;
            drop(state);
            return reply.ok();
        };
        let wait = Wait::new(id, threshold, mode).map_err(|error| {
            tracing::debug!(%error, "gsv:syncpt: a wait no counter can reach");
            KernelError::NotImplemented
        })?;
        let event = self.slot_event(cx, index)?;
        let mut state = shared.lock();
        cx.clear_event(event)?;
        let value = state.arm(event, wait).map_err(|error| {
            tracing::error!(%error, "gsv:syncpt: a wait could not watch its syncpoint");
            KernelError::OutOfResource
        })?;
        if wait.reached(value) {
            shared
                .queue()
                .push(Completion::Gpu(GpuCompletion::SyncpointChanged {
                    id,
                    value,
                }));
        }
        drop(state);
        reply.ok()
    }

    /// `CancelWait` (202). in: `u32` slot.
    fn cancel_wait(
        &mut self,
        cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let index = self.slot(req.payload().read_u32()?)?;
        let shared = shared(cx)?;
        let event = self.slot_event(cx, index)?;
        let mut state = shared.lock();
        state.disarm(event);
        cx.signal_event(event)?;
        drop(state);
        reply.ok()
    }
}

/// The interfaces a syncpt server's sessions can be.
#[must_use]
pub fn syncpt_interfaces() -> Interfaces {
    Interfaces::new().with::<SyncptSession>()
}

/// The syncpt server over gsv's `shared` state: every session a [`SyncptSession`], sharing a
/// [`SyncptFamily`].
#[must_use]
pub fn syncpt_server(policy: UnknownPolicy, shared: Arc<Shared>) -> Server {
    Server::new(SYNCPT_NAME, policy)
        .with_root(|| Box::new(SyncptSession::new()))
        .with_interfaces(syncpt_interfaces())
        .with_family(SyncptFamily::new(shared))
}
