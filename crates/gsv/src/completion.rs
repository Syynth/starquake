//! The completion queue: where host threads report to the graphics server
//! (`docs/specs/graphics-server.md`, "Who signals what: the completion queue"; port slice 3).
//!
//! No reference source is ported: neither reference has such a queue (NX2 and PR signal their
//! events from the host threads that learn of the change). The design is the spec's.
//!
//! # Who pushes what
//!
//! Host threads never touch gsv's tables or the kernel. They push a [`Completion`] onto the one
//! [`CompletionQueue`] gsv owns, which wakes the timer thread ([`Wake`]); the timer thread takes
//! everything queued, in the queue's order (push order but for folding, below), and acts on it
//! ([`crate::shared::Shared::drain`]). Draining on one thread gives one order, which is the
//! order recorded for replay (observability spec, rule 3).
//!
//! | Producer | Through | Pushes |
//! |---|---|---|
//! | The GPU (back thread after host completion; the front for room) | [`Completions::push`] (the trait `starquake-gpu` defines and gsv implements here) | [`Completion::Gpu`] |
//! | `Syncpoints::increment`, for a watched id only (the watched-id fast path, below) | the same | [`Completion::Gpu`] with [`GpuCompletion::SyncpointChanged`] |
//! | The kernel's hook on an event gsv created ([`EventHook`]), after the kernel lock is released | [`EventHook::signalled`] | [`Completion::GuestSignalled`] |
//!
//! A [`CompletionQueue`] is shared as an `Arc`: given to `Syncpoints::new` as its
//! `Arc<dyn Completions>`, to the GPU the same way, and to the kernel's `create_hooked_event` as
//! its `Arc<dyn EventHook>`.
//!
//! **The watched-id fast path.** `GpuCompletion::SyncpointChanged` is pushed only for ids someone
//! waits on: `Syncpoints` keeps an atomic bitmap of watched ids that `increment` checks first
//! (`starquake_gpu::syncpoint`, "Who calls what"), and gsv keeps the bitmap in step with its
//! waiters ([`crate::shared::Locked::watch`]). An unwatched increment costs the GPU one load and
//! pushes nothing (`an_unwatched_increment_pushes_nothing`, in `shared/tests.rs`).
//!
//! # What the spec's `Completion` holds that this one does not yet
//!
//! - **`VideoFrame { decoder, frame }`,** from the video worker, comes with that worker (decoding
//!   is deferred: graphics-server decision 10) and with its `DecoderId` and `FrameInfo` types.
//! - **A device-share release** (owner's ruling of 2026-11-17, graphics-server "The model"):
//!   taking a device-share lock will return a token naming the process and the exact range, the
//!   release is posted here with that token, and the drain unlocks through the kernel, which
//!   validates it. The kernel's lock returns no token yet, so the variant comes with gpu-session
//!   slice 7 and the kernel's token type. It is one more [`Completion`] variant, pushed by
//!   [`CompletionQueue::push`] and kept in push order like [`GpuCompletion::GpuOrdered`], and,
//!   like it, a syncpoint change is never folded across it (below, "Coalescing"): when it comes,
//!   it sets the same fold barrier `Pending::push` sets for an ordered entry. Nothing here
//!   assumes the set of variants is closed except the match in [`CompletionQueue::push`] and
//!   the snapshot's [`Saved`], which gain an arm each (and a version bump).
//!
//! # Coalescing: what bounds the queue
//!
//! A hostile title can make a host thread push without limit: one submission's increment value
//! (`SubmitGPFIFO` flag bit 8) is a `u32`, and each increment of a watched id is a push, and each
//! guest `SignalEvent` on a hooked event is one. So three kinds, each a report of state that a
//! later report of the same thing replaces, are kept once per thing while queued:
//!
//! - [`GpuCompletion::SyncpointChanged`]: a change of an id folds into the id's latest queued
//!   change if no [`GpuCompletion::GpuOrdered`] is queued after that entry; the entry keeps its
//!   place and takes the value further ahead by the 32-bit wrapping comparison
//!   (`starquake_gpu::syncpoint::wait_satisfied`, mode 0), so two incrementers that push out of
//!   order into one entry still leave the newer value
//!   (`a_queued_syncpoint_change_keeps_its_place_and_the_value_further_ahead`). If an ordered
//!   entry is queued after it, the change takes a new place at the back instead, and later
//!   changes of the id fold into that one: a change pushed after an ordered entry never drains
//!   before it, which the spec's "Order" assumption (graphics-server, "What gsv assumes":
//!   increments and ordered updates take effect in call order) needs
//!   (`a_syncpoint_change_is_not_folded_across_an_ordered_entry`). So an id has at most one
//!   entry more than the ordered entries queued
//!   (`a_syncpoint_takes_at_most_one_entry_more_than_the_ordered_entries`), and with none queued
//!   at most `SYNCPOINT_COUNT` entries in all.
//! - [`GpuCompletion::GpuQueueRoom`]: one entry per channel (`room_is_kept_once_per_channel`),
//!   folded across ordered entries: a room report carries no value, so the earlier entry says
//!   all the later one would.
//! - [`Completion::GuestSignalled`]: one entry per event (`a_guest_signal_is_kept_once_per_event`),
//!   folded across ordered entries likewise.
//!
//! [`GpuCompletion::GpuOrdered`] is never coalesced: each ticket is one ordered update or
//! barrier gsv asked for, so their number is bounded by gsv's own requests, and every one is
//! drained (`ordered_reports_are_each_kept`). Once an entry is drained, the next report of the
//! same thing is queued again.
//!
//! **Drained syncpoint values are not monotone.** Folding keeps the value further ahead only
//! within one entry. An incrementer that lost the race and pushes its older value after the
//! newer one was drained, or after an ordered entry was queued behind the newer one, gives that
//! older value a new entry, so the values of one id drain, and are recorded as
//! `SyncpointObserved`, out of order across drains and across an ordered entry
//! (`drained_syncpoint_values_are_not_monotone`). A consumer compares each with its thresholds
//! and must not take a drained value as the counter's latest; the counter itself
//! (`Syncpoints::read`) is.
//!
//! This is starquake's choice: the spec says completions drain in push order and does not say
//! whether repeats are kept. Exactly: entries drain in the order they took their places, and a
//! completion that took no place of its own is the one entry it was folded into. So completions
//! no two of which fold together drain exactly in push order
//! (`scripted_completions_drain_in_push_order`); a folded repeat drains at its entry's place,
//! which is earlier than its push, never across an ordered entry for a syncpoint change, and
//! possibly across one for a room report or a guest signal.
//!
//! # Never blocking
//!
//! [`Completions::push`] "never blocks" (graphics-server, "What gsv assumes"). The queue's lock
//! is held only to push or take, never across a call out of this module, and gsv's shared lock
//! is never taken by a push, so a pusher waits at most for one push or one take of the queue's
//! own entries (`a_push_does_not_wait_for_the_shared_lock`, in `shared/tests.rs`). The
//! [`Wake`] is told with no lock held (`the_wake_is_told_with_no_lock_held`), and only when the
//! queue goes from empty to not empty: the timer thread takes everything when it drains, so a
//! push onto a non-empty queue has already been announced
//! (`the_timer_is_woken_once_per_empty_to_queued`). That announcement can come before the timer
//! thread sleeps (a push onto the queue its drain just emptied), so the [`Wake`] must latch it;
//! see the trait.
//!
//! # Snapshot format
//!
//! [`CompletionQueue::save`] writes what is queued, in order, with the coalesced values;
//! [`CompletionQueue::restore`] builds a queue holding it. `postcard`, after a header of
//! [`SNAPSHOT_MAGIC`] and [`SNAPSHOT_VERSION`] that is decoded and checked before the body. The
//! body is a sequence of [`Saved`] entries. The golden-bytes test `the_snapshot_layout_is_pinned`
//! pins it; changing it means bumping [`SNAPSHOT_VERSION`].
//!
//! A snapshot is a file and is not trusted. Restoring checks, in this order, and returns the
//! first failure: the magic, the version, that the body decodes, that no bytes follow it, and
//! then each entry in order: that a syncpoint id is below `SYNCPOINT_COUNT`, and that the entry
//! is not one the queue would have folded into an earlier entry (a syncpoint change whose id
//! has an entry with no ordered entry after it, or a room report or guest signal naming what an
//! earlier entry names). Saving runs the same entry checks on what it is about to write.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::{ChannelId, Completions, GpuCompletion, SyncpointId, Ticket};
use starquake_gpu::syncpoint::{CompareMode, wait_satisfied};
use starquake_kernel::id::ObjectId;
use starquake_kernel::ipc::EventHook;
use thiserror::Error;

#[cfg(test)]
mod tests;

/// First bytes of every completion-queue snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SQCQ";

/// Layout version written into every completion-queue snapshot and required when reading one.
pub const SNAPSHOT_VERSION: u32 = 1;

/// What a host thread reports to the graphics server (graphics-server, "Who signals what: the
/// completion queue"). See the module docs for the variants the spec's enum holds that this one
/// does not yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Completion {
    /// A report from the GPU, or a watched syncpoint's change.
    Gpu(GpuCompletion),
    /// The guest signalled `event`, the writable end of an event gsv created with a hook.
    GuestSignalled {
        /// The writable end the guest signalled.
        event: ObjectId,
    },
}

/// Told when the queue goes from empty to holding something: the timer thread should drain it.
///
/// Called on the pushing thread with no lock held. It must not block and must not call into
/// gsv.
///
/// **An implementation must latch the notice.** The timer thread can be told before it sleeps:
/// a push onto the queue its drain has just emptied calls this while the timer thread is still
/// finishing that drain. A bare condition-variable notify with no flag behind it loses that
/// call, and pushes onto the then non-empty queue do not call again, so their entries would wait
/// for the timer thread's next deadline. Keep a permit or a flag that the timer thread checks
/// and clears before it sleeps, as `std::thread::Thread::unpark` does.
pub trait Wake: Send + Sync {
    /// Something was queued onto an empty queue.
    fn completion_queued(&self);
}

/// One place in the queue's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    /// A syncpoint's change, with the value further ahead of those folded into it.
    Syncpoint(SyncpointId, u32),
    /// An ordered update or barrier is done.
    Ordered(Ticket),
    /// A channel's queue has room.
    Room(ChannelId),
    /// The guest signalled a hooked event.
    Signalled(ObjectId),
}

/// What the queue holds.
#[derive(Default)]
struct Pending {
    /// Every queued entry, in push order. Only pushed to at the back and emptied whole, so an
    /// index into it stays valid while it is queued.
    order: VecDeque<Slot>,
    /// For each syncpoint with a queued change, the index in [`Pending::order`] of its latest
    /// change: the one a later change of the id folds into, unless an ordered entry follows it.
    syncpoints: BTreeMap<SyncpointId, usize>,
    /// The index in [`Pending::order`] of the latest ordered entry, if one is queued.
    last_ordered: Option<usize>,
    /// The channels with a queued room report.
    rooms: BTreeSet<ChannelId>,
    /// The events with a queued guest signal.
    signalled: BTreeSet<ObjectId>,
}

impl Pending {
    /// Queues `completion`, coalescing as the module docs say. True if it took a new place in
    /// the order, false if it was folded into an entry already queued.
    fn push(&mut self, completion: Completion) -> bool {
        let slot = match completion {
            Completion::Gpu(GpuCompletion::SyncpointChanged { id, value }) => {
                if let Some(held) = self.fold_target(id) {
                    if is_ahead(value, *held) {
                        *held = value;
                    }
                    return false;
                }
                self.syncpoints.insert(id, self.order.len());
                Slot::Syncpoint(id, value)
            }
            Completion::Gpu(GpuCompletion::GpuOrdered { ticket }) => {
                self.last_ordered = Some(self.order.len());
                Slot::Ordered(ticket)
            }
            Completion::Gpu(GpuCompletion::GpuQueueRoom { channel }) => {
                if !self.rooms.insert(channel) {
                    return false;
                }
                Slot::Room(channel)
            }
            Completion::GuestSignalled { event } => {
                if !self.signalled.insert(event) {
                    return false;
                }
                Slot::Signalled(event)
            }
        };
        self.order.push_back(slot);
        true
    }

    /// The value of syncpoint `id`'s latest queued change, if no ordered entry follows it: what
    /// a new change of `id` folds into (module docs, "Coalescing").
    fn fold_target(&mut self, id: SyncpointId) -> Option<&mut u32> {
        let at = *self.syncpoints.get(&id)?;
        if self.last_ordered.is_some_and(|ordered| ordered > at) {
            return None;
        }
        match self.order.get_mut(at) {
            Some(Slot::Syncpoint(held_id, value)) if *held_id == id => Some(value),
            _ => None,
        }
    }

    /// Every entry, in order, as the completion it stands for.
    fn completions(&self) -> impl Iterator<Item = Completion> + '_ {
        self.order.iter().map(|slot| match *slot {
            Slot::Syncpoint(id, value) => {
                Completion::Gpu(GpuCompletion::SyncpointChanged { id, value })
            }
            Slot::Ordered(ticket) => Completion::Gpu(GpuCompletion::GpuOrdered { ticket }),
            Slot::Room(channel) => Completion::Gpu(GpuCompletion::GpuQueueRoom { channel }),
            Slot::Signalled(event) => Completion::GuestSignalled { event },
        })
    }
}

/// Whether syncpoint value `new` is at or past `old` by the 32-bit wrapping comparison (mode 0 of
/// `starquake_gpu::syncpoint::wait_satisfied`).
fn is_ahead(new: u32, old: u32) -> bool {
    wait_satisfied(new, u64::from(old), CompareMode::Wrapping32)
}

/// The graphics server's completion queue: multi-producer, drained by the timer thread. See the
/// module docs.
#[derive(Default)]
pub struct CompletionQueue {
    pending: Mutex<Pending>,
    wake: Option<Arc<dyn Wake>>,
}

impl CompletionQueue {
    /// An empty queue that wakes no one.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The queue, telling `wake` when it goes from empty to holding something.
    #[must_use]
    pub fn with_wake(mut self, wake: Arc<dyn Wake>) -> Self {
        self.wake = Some(wake);
        self
    }

    /// What the queue holds. A poisoned lock means a panic elsewhere while an entry was pushed
    /// or taken; every step leaves [`Pending`] whole, so carry on.
    fn pending(&self) -> MutexGuard<'_, Pending> {
        self.pending.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Queues `completion`, coalescing as the module docs say, and wakes the timer thread if the
    /// queue was empty. Never waits for anything but the queue's own lock.
    pub fn push(&self, completion: Completion) {
        let woke = {
            let mut pending = self.pending();
            let was_empty = pending.order.is_empty();
            pending.push(completion) && was_empty
        };
        if woke && let Some(wake) = &self.wake {
            wake.completion_queued();
        }
    }

    /// Takes everything queued, in the queue's order (module docs, "Coalescing"), leaving the
    /// queue empty. The timer thread's half of the queue ([`crate::shared::Shared::drain`] calls
    /// it).
    pub fn take_all(&self) -> Vec<Completion> {
        let pending = std::mem::take(&mut *self.pending());
        pending.completions().collect()
    }

    /// Writes what is queued, in order (module docs, "Snapshot format"). Take it while nothing
    /// pushes, or it is a moment of the queue that may already have changed.
    pub fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let entries: Vec<Saved> = self.pending().completions().map(Saved::from).collect();
        rebuild(&entries)?;
        let mut bytes = postcard::to_allocvec(&Header {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
        })
        .map_err(SnapshotError::Encode)?;
        bytes.extend(postcard::to_allocvec(&entries).map_err(SnapshotError::Encode)?);
        Ok(bytes)
    }

    /// A queue holding what [`save`](Self::save) wrote, waking no one; the snapshot is checked
    /// first (module docs, "Snapshot format"). Add a [`Wake`] with [`with_wake`](Self::with_wake);
    /// a restored queue that holds something has not announced it, so the timer thread should
    /// drain once after a restore.
    pub fn restore(bytes: &[u8]) -> Result<Self, SnapshotError> {
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
        let (entries, rest) =
            postcard::take_from_bytes::<Vec<Saved>>(rest).map_err(SnapshotError::Decode)?;
        if !rest.is_empty() {
            return Err(SnapshotError::TrailingBytes { count: rest.len() });
        }
        Ok(Self {
            pending: Mutex::new(rebuild(&entries)?),
            wake: None,
        })
    }
}

/// Builds the queue `entries` describe, checking each entry in order (module docs, "Snapshot
/// format"). [`CompletionQueue::save`] and [`CompletionQueue::restore`] share it.
fn rebuild(entries: &[Saved]) -> Result<Pending, SnapshotError> {
    let mut pending = Pending::default();
    for (index, entry) in entries.iter().enumerate() {
        if !pending.push(entry.completion(index)?) {
            return Err(SnapshotError::Repeated { index });
        }
    }
    Ok(pending)
}

impl Completions for CompletionQueue {
    /// Queues the GPU's report as [`Completion::Gpu`] ([`CompletionQueue::push`]).
    fn push(&self, completion: GpuCompletion) {
        Self::push(self, Completion::Gpu(completion));
    }
}

impl EventHook for CompletionQueue {
    /// Queues [`Completion::GuestSignalled`] for `event` ([`CompletionQueue::push`]).
    fn signalled(&self, event: ObjectId) {
        self.push(Completion::GuestSignalled { event });
    }
}

/// Why a completion-queue snapshot could not be written or read.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SnapshotError {
    /// Encoding failed.
    #[error("could not encode the completion queue snapshot: {0}")]
    Encode(postcard::Error),
    /// The header or the body is not a well-formed encoding.
    #[error("could not decode the completion queue snapshot: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`SNAPSHOT_MAGIC`].
    #[error("not a completion queue snapshot (bad magic)")]
    BadMagic,
    /// The snapshot was written with another layout version.
    #[error(
        "completion queue snapshot layout version {found} is not supported (this build reads {SNAPSHOT_VERSION})"
    )]
    UnsupportedVersion {
        /// The version found.
        found: u32,
    },
    /// Bytes follow the body.
    #[error("{count} unexpected bytes after the completion queue snapshot")]
    TrailingBytes {
        /// How many.
        count: usize,
    },
    /// An entry names a syncpoint at or above the count.
    #[error(
        "entry {index} of the completion queue snapshot names syncpoint {id}, which does not exist"
    )]
    NoSuchSyncpoint {
        /// The entry's position.
        index: usize,
        /// The id it names.
        id: u8,
    },
    /// An entry of a coalesced kind names what an earlier entry already names.
    #[error("entry {index} of the completion queue snapshot repeats an earlier entry")]
    Repeated {
        /// The entry's position.
        index: usize,
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

/// One queued completion as a snapshot holds it (module docs, "Snapshot format"). Ids are plain
/// numbers here and checked on the way back ([`Saved::completion`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum Saved {
    /// [`GpuCompletion::SyncpointChanged`].
    SyncpointChanged {
        /// The syncpoint id.
        id: u8,
        /// Its value.
        value: u32,
    },
    /// [`GpuCompletion::GpuOrdered`].
    GpuOrdered {
        /// The ticket.
        ticket: u64,
    },
    /// [`GpuCompletion::GpuQueueRoom`].
    GpuQueueRoom {
        /// The channel.
        channel: u32,
    },
    /// [`Completion::GuestSignalled`].
    GuestSignalled {
        /// The event.
        event: ObjectId,
    },
}

impl From<Completion> for Saved {
    fn from(completion: Completion) -> Self {
        match completion {
            Completion::Gpu(GpuCompletion::SyncpointChanged { id, value }) => {
                Self::SyncpointChanged {
                    id: id.get(),
                    value,
                }
            }
            Completion::Gpu(GpuCompletion::GpuOrdered { ticket }) => {
                Self::GpuOrdered { ticket: ticket.0 }
            }
            Completion::Gpu(GpuCompletion::GpuQueueRoom { channel }) => {
                Self::GpuQueueRoom { channel: channel.0 }
            }
            Completion::GuestSignalled { event } => Self::GuestSignalled { event },
        }
    }
}

impl Saved {
    /// The completion this entry, at `index`, stands for, or why it names nothing.
    fn completion(self, index: usize) -> Result<Completion, SnapshotError> {
        Ok(match self {
            Self::SyncpointChanged { id, value } => {
                let id =
                    SyncpointId::new(id).ok_or(SnapshotError::NoSuchSyncpoint { index, id })?;
                Completion::Gpu(GpuCompletion::SyncpointChanged { id, value })
            }
            Self::GpuOrdered { ticket } => Completion::Gpu(GpuCompletion::GpuOrdered {
                ticket: Ticket(ticket),
            }),
            Self::GpuQueueRoom { channel } => Completion::Gpu(GpuCompletion::GpuQueueRoom {
                channel: ChannelId(channel),
            }),
            Self::GuestSignalled { event } => Completion::GuestSignalled { event },
        })
    }
}
