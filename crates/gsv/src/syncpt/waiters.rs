//! The waiter table: every wait the syncpt sessions have armed (syncpt 201), by the slot event
//! it signals (`docs/specs/graphics-server.md`, "Syncpoints and the syncpt session": "Arming a
//! wait registers (slot, fence) in gsv's waiter table; it never blocks").
//!
//! The table lives in gsv's shared state ([`crate::shared::GsvShared`]), because the request
//! handlers arm and cancel waits and the timer thread fires them as it drains the completion
//! queue ([`crate::shared::Shared::drain`]). Each armed wait holds one of its syncpoint's
//! watches ([`crate::shared::Locked::arm`]), so the GPU pushes every change of a syncpoint
//! someone waits on.
//!
//! A wait is keyed by its slot's event. An event is one slot of one session, so a slot has at
//! most one wait, and arming a slot again replaces its wait, as both references do (PR
//! `ArmWait`: `m_waits[request->slot] = *request`; NX2 `RequestSlotEvent`: `Cancel(request)`
//! before registering the new one; reference paths in [`crate::syncpt`]).
//!
//! # Snapshot form
//!
//! The table is saved as a list of [`SavedWait`], one per wait in ascending event order, inside
//! gsv's shared-state snapshot ([`crate::shared`], "Snapshot format"). [`Waiters::restore`]
//! checks each entry, in list order, and returns the first failure: that its event comes after
//! the one before (so no event has two waits), that its syncpoint id is below
//! `SYNCPOINT_COUNT`, that its mode is one 201 accepts ([`compare_mode`]), and that a 64-bit
//! wait's threshold is one 201 accepts ([`Wait::new`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::{SYNCPOINT_COUNT, SyncpointId};
use starquake_gpu::syncpoint::{CompareMode, wait_satisfied};
use starquake_kernel::id::ObjectId;
use thiserror::Error;

/// The mode word of syncpt 201 that compares 32-bit wrapping (graphics-server, "Syncpoints and
/// the syncpt session": "mode 0 compares 32-bit wrapping"; PR `DecodeSyncpointWait`:
/// `bool(packet[6] & 0xff)` false).
pub const MODE_WRAPPING_32: u8 = 0;

/// The mode word of syncpt 201 that compares as 64 bits ("mode 1 compares as 64-bit"; PR
/// `DecodeSyncpointWait`: `bool(packet[6] & 0xff)` true). A **conflict** with NX2, which
/// compares only the low 32 bits whatever the mode; PR's reading is kept until a trace settles
/// it (decision log, 2026-11-17; graphics-server, "What to verify before building", item 1).
pub const MODE_WIDE_64: u8 = 1;

/// The comparison a 201 mode byte asks for, or `None` for a mode 201 refuses (PR
/// `DecodeSyncpointWait`: `(packet[6] & 0xff) > 1` fails the decode).
#[must_use]
pub const fn compare_mode(mode: u8) -> Option<CompareMode> {
    match mode {
        MODE_WRAPPING_32 => Some(CompareMode::Wrapping32),
        MODE_WIDE_64 => Some(CompareMode::Wide64),
        _ => None,
    }
}

/// The 201 mode byte of a comparison: the inverse of [`compare_mode`], for the snapshot.
#[must_use]
pub const fn mode_number(mode: CompareMode) -> u8 {
    match mode {
        CompareMode::Wrapping32 => MODE_WRAPPING_32,
        CompareMode::Wide64 => MODE_WIDE_64,
    }
}

/// One armed wait: done once syncpoint `id` satisfies `threshold` by `mode`
/// ([`wait_satisfied`]). Built only by [`Wait::new`], which refuses a wait 201 refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Wait {
    id: SyncpointId,
    threshold: u64,
    mode: CompareMode,
}

/// Why a wait cannot be armed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum WaitError {
    /// A 64-bit wait whose threshold no 32-bit counter can reach. PR's `ArmWait` refuses it
    /// with `ResultNotImplemented` (`request->wide && request->threshold >
    /// std::numeric_limits<u32>::max()`).
    #[error("a 64-bit wait for {threshold:#x} can never be reached by a 32-bit counter")]
    WideThreshold {
        /// The threshold.
        threshold: u64,
    },
}

impl Wait {
    /// A wait on `id` for `threshold` by `mode`. A [`CompareMode::Wide64`] wait whose threshold
    /// is above `u32::MAX` is [`WaitError::WideThreshold`]: nothing could ever satisfy it (PR
    /// `ArmWait`). A [`CompareMode::Wrapping32`] wait takes any threshold and compares its low
    /// 32 bits.
    pub fn new(id: SyncpointId, threshold: u64, mode: CompareMode) -> Result<Self, WaitError> {
        if mode == CompareMode::Wide64 && threshold > u64::from(u32::MAX) {
            return Err(WaitError::WideThreshold { threshold });
        }
        Ok(Self {
            id,
            threshold,
            mode,
        })
    }

    /// The syncpoint.
    #[must_use]
    pub const fn id(&self) -> SyncpointId {
        self.id
    }

    /// The threshold, as the request gave it.
    #[must_use]
    pub const fn threshold(&self) -> u64 {
        self.threshold
    }

    /// The comparison.
    #[must_use]
    pub const fn mode(&self) -> CompareMode {
        self.mode
    }

    /// Whether a counter at `value` satisfies the wait ([`wait_satisfied`]).
    #[must_use]
    pub fn reached(&self, value: u32) -> bool {
        wait_satisfied(value, self.threshold, self.mode)
    }
}

/// The waiter table (module docs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Waiters {
    by_event: BTreeMap<ObjectId, Wait>,
}

impl Waiters {
    /// The wait armed on `event`, if one is.
    #[must_use]
    pub fn get(&self, event: ObjectId) -> Option<Wait> {
        self.by_event.get(&event).copied()
    }

    /// How many waits are armed in all.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_event.len()
    }

    /// Whether no wait is armed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_event.is_empty()
    }

    /// How many waits are armed on syncpoint `id`.
    #[must_use]
    pub fn on(&self, id: SyncpointId) -> usize {
        self.by_event.values().filter(|wait| wait.id == id).count()
    }

    /// Arms `wait` on `event`, returning the wait it replaces.
    pub(crate) fn insert(&mut self, event: ObjectId, wait: Wait) -> Option<Wait> {
        self.by_event.insert(event, wait)
    }

    /// Takes the wait armed on `event` out of the table.
    pub(crate) fn remove(&mut self, event: ObjectId) -> Option<Wait> {
        self.by_event.remove(&event)
    }

    /// Takes out every wait on `id` that a counter at `value` satisfies, and returns their
    /// events in ascending order. Waits on other syncpoints, and waits on `id` that `value` does
    /// not satisfy, stay.
    pub(crate) fn take_reached(&mut self, id: SyncpointId, value: u32) -> Vec<ObjectId> {
        let reached: Vec<ObjectId> = self
            .by_event
            .iter()
            .filter(|(_, wait)| wait.id == id && wait.reached(value))
            .map(|(&event, _)| event)
            .collect();
        for event in &reached {
            self.by_event.remove(event);
        }
        reached
    }

    /// The table in its snapshot form (module docs, "Snapshot form").
    pub(crate) fn save(&self) -> Vec<SavedWait> {
        self.by_event
            .iter()
            .map(|(&event, wait)| SavedWait {
                event,
                id: wait.id.get(),
                threshold: wait.threshold,
                mode: mode_number(wait.mode),
            })
            .collect()
    }

    /// The table `saved` describes, checked in the order the module docs give.
    pub(crate) fn restore(saved: &[SavedWait]) -> Result<Self, SavedWaitError> {
        let mut by_event = BTreeMap::new();
        let mut previous: Option<ObjectId> = None;
        for (index, entry) in saved.iter().enumerate() {
            if previous.is_some_and(|previous| entry.event <= previous) {
                return Err(SavedWaitError::OutOfOrder { index });
            }
            previous = Some(entry.event);
            let id = SyncpointId::new(entry.id).ok_or(SavedWaitError::NoSuchSyncpoint {
                index,
                id: entry.id,
            })?;
            let mode = compare_mode(entry.mode).ok_or(SavedWaitError::UnknownMode {
                index,
                mode: entry.mode,
            })?;
            let wait = Wait::new(id, entry.threshold, mode)
                .map_err(|error| SavedWaitError::Refused { index, error })?;
            by_event.insert(entry.event, wait);
        }
        Ok(Self { by_event })
    }
}

/// One wait as a snapshot holds it (module docs, "Snapshot form"). The field order is the
/// layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SavedWait {
    /// The slot event the wait signals.
    pub(crate) event: ObjectId,
    /// The syncpoint id.
    pub(crate) id: u8,
    /// The threshold.
    pub(crate) threshold: u64,
    /// The 201 mode byte ([`MODE_WRAPPING_32`] or [`MODE_WIDE_64`]).
    pub(crate) mode: u8,
}

/// Why a saved waiter table could not be read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum SavedWaitError {
    /// The entry's event does not come after the one before it: two waits on one event, or a
    /// list not in ascending event order.
    #[error("saved wait {index} is not in ascending event order")]
    OutOfOrder {
        /// The entry's place in the list.
        index: usize,
    },
    /// The entry names no syncpoint.
    #[error("saved wait {index} names syncpoint {id}, and there are {SYNCPOINT_COUNT}")]
    NoSuchSyncpoint {
        /// The entry's place in the list.
        index: usize,
        /// The id it names.
        id: u8,
    },
    /// The entry's mode is not one syncpt 201 accepts.
    #[error("saved wait {index} has mode {mode}, which no wait is armed with")]
    UnknownMode {
        /// The entry's place in the list.
        index: usize,
        /// The mode.
        mode: u8,
    },
    /// The entry is a wait syncpt 201 refuses to arm.
    #[error("saved wait {index} could not have been armed: {error}")]
    Refused {
        /// The entry's place in the list.
        index: usize,
        /// Why 201 refuses it.
        error: WaitError,
    },
}
