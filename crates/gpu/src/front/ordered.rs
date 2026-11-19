//! Ordered address-table updates and barriers on the front (gpu-protocol spec, "Ordered
//! address-table updates and barriers", steps 1 and 2; row 68 of `eden-nx2/NX2_NOTES.md`; the
//! owner's decision of 2026-11-12, "Syncpoints and the GPU address table live in the GPU crate":
//! an update is reported done only when no earlier work, this thread included, can use the old
//! mapping).
//!
//! # The mechanism
//!
//! 1. An update (`update_vm`) or a barrier reaches the front in call order with the
//!    submissions. The front records, for every channel it covers, the sequence number of the
//!    last submission that channel took before it ([`Pending::positions`]). An update covers the
//!    channels of its address space, which the front takes as the process (**guess**: gsv makes
//!    one address space per GPU session, and the spec's "every channel of that process" names no
//!    process id; a channel's `ChannelConfig` names only its space). A barrier covers every
//!    channel.
//! 2. Once every covered channel has finished its recorded submission (read every entry and
//!    pushed its appended increments), or is destroyed or stopped, the front files the update in
//!    the [`OrderedTable`] under a fresh marker token and then pushes a `Marker` with that token
//!    into the ring.
//! 3. The back's executor (slice G19) takes the update from the table when it pops the marker
//!    ([`OrderedTable::take`]) and attaches it to the marker as a fenced action, which slice G15
//!    applies once everything before it is performed ([`crate::back::fence`]).
//!
//! Pending updates are looked at in call order, and each that is ready is emitted then, so one
//! whose channels are done is not held behind an earlier one whose channels are not. Within one
//! address space this keeps call order: a later update's positions are at or past an earlier
//! one's for every channel both cover, and a channel made between them is covered by the later
//! one only, so the later cannot be ready first. A barrier covers every channel, so the same
//! holds between a barrier and anything after it; an update queued after a barrier may be
//! emitted before the barrier when the barrier waits on another space's channel
//! (`an_update_after_a_barrier_is_not_held_by_another_spaces_channel`).
//!
//! # The two open cases (spec verification item 12)
//!
//! The spec's behaviour is taken for both, as the issue asks: (a) a channel parked on an acquire
//! holds every later update of its address space until it resumes
//! (`a_parked_channel_holds_its_spaces_update_until_it_resumes`); (b) work queued after an update
//! but run before it is applied sees the old mapping: nothing holds a channel behind a pending
//! update (`work_queued_after_an_update_runs_before_it_is_applied`).
//!
//! An update of an address space the front was never given covers no channel, so it is ready at
//! once; its marker carries no table change, only the ticket, since there is no table to change
//! (`an_update_of_an_unknown_space_is_reported_without_a_change`).

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::back::fence::OrderedUpdate;
use crate::interface::ChannelId;
use crate::ring::MarkerToken;

/// The ordered updates the front has pushed markers for and the back has not taken yet, by
/// token: shared by the front, which files them, and the back's executor, which takes each when
/// it pops its marker (module docs). Every method takes the lock only for its own operation.
#[derive(Default)]
pub struct OrderedTable {
    updates: Mutex<BTreeMap<MarkerToken, OrderedUpdate>>,
}

impl OrderedTable {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, BTreeMap<MarkerToken, OrderedUpdate>> {
        // Every method leaves the map whole, so a poisoned lock is usable.
        self.updates.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Files `update` under `token`, before the marker is pushed.
    pub(crate) fn insert(&self, token: MarkerToken, update: OrderedUpdate) {
        self.lock().insert(token, update);
    }

    /// Takes the update filed under `token`: what the executor attaches to the marker it popped.
    /// `None` for a marker that carries none (a catch-up or a `SYNC_WAIT`).
    pub fn take(&self, token: MarkerToken) -> Option<OrderedUpdate> {
        self.lock().remove(&token)
    }

    /// How many updates are filed and not yet taken.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether no update is filed.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An update or barrier waiting for the channels it covers.
pub(super) struct Pending {
    /// What the marker carries.
    pub(super) update: OrderedUpdate,
    /// Each covered channel with the sequence number of the last submission it took before the
    /// call.
    pub(super) positions: Vec<(ChannelId, u64)>,
}

impl Pending {
    /// Whether every covered channel is past its position: `done(channel, seq)` says whether
    /// the channel has finished submission `seq`, or is gone or stopped.
    pub(super) fn is_ready(&self, mut done: impl FnMut(ChannelId, u64) -> bool) -> bool {
        self.positions.iter().all(|&(channel, seq)| done(channel, seq))
    }
}
