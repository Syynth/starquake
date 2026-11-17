//! The state the graphics server's request handlers and its timer thread share, the lock order
//! around it, and the timer thread's drain of the completion queue
//! (`docs/specs/graphics-server.md`, "Who signals what: the completion queue" and "Concurrency
//! inside gsv"; port slice 3).
//!
//! No reference source is ported: the design is the spec's.
//!
//! # The pieces
//!
//! [`Shared`] holds the [`GsvShared`] state behind gsv's one shared lock, the `Syncpoints` gsv
//! and the GPU share, and the [`CompletionQueue`]. Build them in that order backwards: the queue
//! first, then `Syncpoints::new` with the queue as its `Completions`, then [`Shared::new`].
//!
//! [`GsvShared`] holds, so far, the waiter count of each syncpoint ([`Locked::watch`],
//! [`Locked::unwatch`]); later slices add the waiter tables, the shared page, the GPU memory
//! bookkeeping, vi2's queues and the native object tables (spec, "Concurrency inside gsv").
//!
//! # The watched-id fast path
//!
//! `Syncpoints` pushes a change only for a watched id (`starquake_gpu::syncpoint`). gsv keeps a
//! count of waiters per id: [`Locked::watch`] sets the id's bit when its count goes from 0 to 1,
//! [`Locked::unwatch`] clears it when the count goes back to 0, and the bit is set exactly when
//! the count is not zero (a snapshot is checked for it). `watch` reads the counter after setting
//! the bit and returns it, so a waiter compares its threshold with that value and misses no
//! increment: one that came before is in the value, one that came after is pushed (the
//! threaded test of `starquake_gpu::syncpoint` checks the `Syncpoints` side;
//! `watching_misses_no_increment_that_races_it` checks the order of the two steps here with a
//! racing incrementer over fixed trials, so it catches a swap with high likelihood rather than
//! certainty, since only an increment that lands between the two steps is lost).
//!
//! # Lock order and the drain
//!
//! The order is gsv's shared lock, then the kernel's (spec, "Who signals what"). [`Shared::drain`]
//! is the timer thread's turn: it takes gsv's lock, then takes everything queued
//! (`the_drain_takes_the_queue_after_the_shared_lock`), and acts on each completion in the
//! queue's order through the [`TimerHost`], which is how it reaches the kernel; so
//! every kernel call it makes is made with gsv's lock held and never the other way round
//! (`the_drain_calls_the_kernel_with_the_shared_lock_held`). Nothing calls into gsv with the
//! kernel lock held: the kernel's event hook runs after it releases its lock and only pushes, and
//! a push takes only the queue's own lock, never gsv's
//! (`a_push_does_not_wait_for_the_shared_lock`). The queue's lock is not held while the drain
//! acts, so a completion pushed meanwhile (by the kernel's hook, say) waits for the next drain
//! (`a_push_during_the_drain_waits_for_the_next_one`).
//!
//! What the drain does with each completion, so far, is record it, except a device-share
//! release, which it unlocks:
//!
//! | Completion | What the drain does |
//! |---|---|
//! | `GpuCompletion::SyncpointChanged` | records `SyncpointObserved` ([`crate::note`]) |
//! | `GpuCompletion::GpuOrdered` | records `GpuOrderedDone` |
//! | `GpuCompletion::GpuQueueRoom` | records `QueueRoom` |
//! | [`Completion::GuestSignalled`] | nothing: the guest's `SignalEvent` is recorded as its system call, and what gsv does about it (the shared page's answer, slice 5) records its own note |
//! | [`Completion::DeviceShareReleased`] | unlocks through the kernel with the token ([`TimerHost::unlock_device_share`]), which checks it; records nothing (the spec's "What is recorded" has no note for it) |
//!
//! The kernel's refusal of a release is logged, and the drain goes on: a token whose process
//! has ended is refused as a matter of course (the process ended while the GPU still had work
//! that used its memory), and is logged at debug level; any other refusal means gsv's own
//! bookkeeping is wrong, and is logged as an error. Either way the lock stays as the kernel
//! keeps it, and the next completions are acted on
//! (`a_refused_release_is_logged_and_the_drain_goes_on`).
//!
//! Later slices act on them as well: the syncpt waiter table (slice 4), the shared page (5), the
//! ordered completion of unmaps and destroys (7), deferred submissions (11).
//!
//! # The timer thread, until the wiring
//!
//! The core loop's timer thread (kernel slice 10, `starquake-core`) does not call
//! [`Shared::drain`] yet; the wiring slice (16) makes it do so each turn and when the queue's
//! [`Wake`](crate::completion::Wake) rings it, with a [`TimerHost`] that records through the
//! kernel and unlocks through `Dispatcher::unlock_device_share`. Until then tests play the
//! timer thread with a fake host.
//!
//! # Snapshot format
//!
//! [`Shared::save`] writes [`GsvShared`] (graphics-server, "What is snapshotted": "the watched
//! bitmap and every armed wait"; the bitmap itself is in the `Syncpoints` snapshot, the counts
//! here). `postcard`, after a header of [`SNAPSHOT_MAGIC`] and [`SNAPSHOT_VERSION`] that is
//! decoded and checked before the body. The body is the waiter count of each syncpoint, by id.
//! The golden-bytes test `the_snapshot_layout_is_pinned` pins it; changing it means bumping
//! [`SNAPSHOT_VERSION`]. The queue has its own snapshot ([`CompletionQueue::save`]), because it
//! must exist before the `Syncpoints` that pushes to it.
//!
//! A snapshot is a file and is not trusted. [`Shared::restore`] checks, in this order, and
//! returns the first failure: the magic, the version, that the body decodes, that no bytes
//! follow it, that it holds `SYNCPOINT_COUNT` counts, and then, id by id, that the count is not
//! zero exactly when the restored `Syncpoints` has the id watched. [`Shared::save`] runs the same
//! checks on what it is about to write.

use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::{GpuCompletion, SYNCPOINT_COUNT, SyncpointId};
use starquake_gpu::syncpoint::Syncpoints;
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::memory::DeviceShareToken;
use starquake_trace::{EventKind, ServiceId};
use thiserror::Error;

use crate::completion::{Completion, CompletionQueue};
use crate::note::Note;

#[cfg(test)]
mod tests;

/// [`SYNCPOINT_COUNT`] as an index bound.
const COUNT: usize = SYNCPOINT_COUNT as usize;

/// First bytes of every gsv shared-state snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SQGS";

/// Layout version written into every gsv shared-state snapshot and required when reading one.
pub const SNAPSHOT_VERSION: u32 = 1;

/// How the timer thread's drain reaches the kernel and the event stream. The wiring implements
/// it over the kernel's dispatcher; tests implement it with a fake (module docs, "The timer
/// thread, until the wiring").
///
/// It is called only with gsv's shared lock held (module docs, "Lock order and the drain"), so it
/// must never call back into [`Shared`].
pub trait TimerHost {
    /// Records one event in the run's event stream, stamped and ordered by the kernel under its
    /// lock, as happening on no guest thread.
    fn record(&mut self, kind: EventKind);

    /// Releases the device-share lock `token` names, through the kernel, which checks the token
    /// (`Dispatcher::unlock_device_share`, whose refusals these are).
    fn unlock_device_share(&mut self, token: DeviceShareToken) -> Result<(), DeviceShareError>;
}

/// What gsv's request handlers and its timer thread share (module docs, "The pieces").
///
/// Built only here (fresh, or restored from a checked snapshot), and reached only through
/// [`Locked`], which gives shared access to it and changes the waiter counts only with their
/// watched bits; so no caller can replace it and leave a count out of step with its bit.
#[derive(Debug, PartialEq, Eq)]
pub struct GsvShared {
    /// The number of waiters on each syncpoint, by id.
    watchers: Vec<u32>,
}

impl GsvShared {
    /// No waiters on any syncpoint: the state over fresh syncpoints, which watch nothing.
    fn fresh() -> Self {
        Self {
            watchers: vec![0; COUNT],
        }
    }

    /// The number of waiters on syncpoint `id`.
    #[must_use]
    pub fn watchers(&self, id: SyncpointId) -> u32 {
        self.watchers
            .get(usize::from(id.get()))
            .copied()
            .unwrap_or(0)
    }
}

/// Why a syncpoint could not be watched or unwatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum WatchError {
    /// The syncpoint's waiter count is at its maximum.
    #[error("syncpoint {} has too many waiters", .0.get())]
    TooManyWaiters(SyncpointId),
    /// The syncpoint has no waiter to remove.
    #[error("syncpoint {} has no waiters", .0.get())]
    NotWatched(SyncpointId),
}

/// gsv's shared state, locked: [`GsvShared`] and the syncpoints it watches (module docs, "The
/// watched-id fast path"). It derefs to [`GsvShared`] for reading only; the waiter counts change
/// through [`Locked::watch`] and [`Locked::unwatch`], which keep each count in step with its
/// watched bit.
pub struct Locked<'a> {
    state: MutexGuard<'a, GsvShared>,
    syncpoints: &'a Syncpoints,
}

impl Locked<'_> {
    /// Adds a waiter on syncpoint `id`, marking it watched if it had none, and returns its
    /// counter, read after that (module docs). Nothing changes on an error.
    pub fn watch(&mut self, id: SyncpointId) -> Result<u32, WatchError> {
        let count = self
            .state
            .watchers
            .get_mut(usize::from(id.get()))
            .ok_or(WatchError::TooManyWaiters(id))?;
        *count = count.checked_add(1).ok_or(WatchError::TooManyWaiters(id))?;
        if *count == 1 {
            self.syncpoints.watch(id);
        }
        Ok(self.syncpoints.read(id))
    }

    /// Removes a waiter on syncpoint `id`, marking it not watched if that was the last. Nothing
    /// changes on an error.
    pub fn unwatch(&mut self, id: SyncpointId) -> Result<(), WatchError> {
        let count = self
            .state
            .watchers
            .get_mut(usize::from(id.get()))
            .ok_or(WatchError::NotWatched(id))?;
        *count = count.checked_sub(1).ok_or(WatchError::NotWatched(id))?;
        if *count == 0 {
            self.syncpoints.unwatch(id);
        }
        Ok(())
    }
}

impl Deref for Locked<'_> {
    type Target = GsvShared;

    fn deref(&self) -> &GsvShared {
        &self.state
    }
}

/// gsv's shared state, its syncpoints and its completion queue (module docs, "The pieces").
pub struct Shared {
    state: Mutex<GsvShared>,
    syncpoints: Arc<Syncpoints>,
    queue: Arc<CompletionQueue>,
    service: ServiceId,
}

impl Shared {
    /// Fresh state over `syncpoints` and `queue`, recording its notes for the service numbered
    /// `service`. `syncpoints` must push to `queue`.
    #[must_use]
    pub fn new(
        syncpoints: Arc<Syncpoints>,
        queue: Arc<CompletionQueue>,
        service: ServiceId,
    ) -> Self {
        Self::with_state(GsvShared::fresh(), syncpoints, queue, service)
    }

    fn with_state(
        state: GsvShared,
        syncpoints: Arc<Syncpoints>,
        queue: Arc<CompletionQueue>,
        service: ServiceId,
    ) -> Self {
        Self {
            state: Mutex::new(state),
            syncpoints,
            queue,
            service,
        }
    }

    /// Takes gsv's shared lock. Take it before the kernel's, never after (module docs, "Lock
    /// order and the drain"), and do not touch guest memory under it for longer than a copy.
    pub fn lock(&self) -> Locked<'_> {
        // A poisoned lock means a panic elsewhere mid-update; each update leaves `GsvShared`
        // whole (a count and its bit change together), so carry on.
        Locked {
            state: self.state.lock().unwrap_or_else(PoisonError::into_inner),
            syncpoints: &self.syncpoints,
        }
    }

    /// The syncpoints gsv and the GPU share.
    #[must_use]
    pub fn syncpoints(&self) -> &Arc<Syncpoints> {
        &self.syncpoints
    }

    /// The completion queue.
    #[must_use]
    pub fn queue(&self) -> &Arc<CompletionQueue> {
        &self.queue
    }

    /// The timer thread's turn: takes gsv's lock, then everything queued, and acts on each
    /// completion in the queue's order, recording its note through `host` (module docs, "Lock
    /// order and the drain"). Returns what it took, in order.
    pub fn drain(&self, host: &mut dyn TimerHost) -> Vec<Completion> {
        let state = self.lock();
        let taken = self.queue.take_all();
        for completion in &taken {
            match *completion {
                Completion::DeviceShareReleased { token } => release(host, token),
                Completion::GuestSignalled { .. } => {
                    tracing::trace!(?completion, "gsv drained a guest signal");
                }
                Completion::Gpu(gpu) => host.record(note(gpu).event(self.service)),
            }
        }
        drop(state);
        taken
    }

    /// Writes [`GsvShared`] (module docs, "Snapshot format"). Take it while nothing changes the
    /// state or the watched bitmap.
    pub fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let saved = State {
            watchers: self.lock().watchers.clone(),
        };
        saved.check(&self.syncpoints)?;
        let mut bytes = postcard::to_allocvec(&Header {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
        })
        .map_err(SnapshotError::Encode)?;
        bytes.extend(postcard::to_allocvec(&saved).map_err(SnapshotError::Encode)?);
        Ok(bytes)
    }

    /// A `Shared` holding what [`save`](Self::save) wrote, over the restored `syncpoints` and
    /// `queue`, recording for `service`. The snapshot is checked first, against `syncpoints`'
    /// watched bitmap (module docs, "Snapshot format").
    pub fn restore(
        bytes: &[u8],
        syncpoints: Arc<Syncpoints>,
        queue: Arc<CompletionQueue>,
        service: ServiceId,
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
            postcard::take_from_bytes::<State>(rest).map_err(SnapshotError::Decode)?;
        if !rest.is_empty() {
            return Err(SnapshotError::TrailingBytes { count: rest.len() });
        }
        saved.check(&syncpoints)?;
        let state = GsvShared {
            watchers: saved.watchers,
        };
        Ok(Self::with_state(state, syncpoints, queue, service))
    }
}

/// The note the drain records for a GPU's `completion` (module docs, "Lock order and the
/// drain").
fn note(completion: GpuCompletion) -> Note {
    match completion {
        GpuCompletion::SyncpointChanged { id, value } => Note::SyncpointObserved { id, value },
        GpuCompletion::GpuOrdered { ticket } => Note::GpuOrderedDone { ticket },
        GpuCompletion::GpuQueueRoom { channel } => Note::QueueRoom { channel },
    }
}

/// Unlocks the device-share lock `token` names through `host`, logging a refusal (module docs,
/// "Lock order and the drain").
fn release(host: &mut dyn TimerHost, token: DeviceShareToken) {
    match host.unlock_device_share(token) {
        Ok(()) => {}
        Err(DeviceShareError::ProcessEnded) => {
            tracing::debug!(
                ?token,
                "a device-share release came after its process ended"
            );
        }
        Err(error) => {
            tracing::error!(%error, ?token, "the kernel refused gsv's device-share release");
        }
    }
}

/// Why a gsv shared-state snapshot could not be written or read.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SnapshotError {
    /// Encoding failed.
    #[error("could not encode the gsv shared state snapshot: {0}")]
    Encode(postcard::Error),
    /// The header or the body is not a well-formed encoding.
    #[error("could not decode the gsv shared state snapshot: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`SNAPSHOT_MAGIC`].
    #[error("not a gsv shared state snapshot (bad magic)")]
    BadMagic,
    /// The snapshot was written with another layout version.
    #[error(
        "gsv shared state snapshot layout version {found} is not supported (this build reads {SNAPSHOT_VERSION})"
    )]
    UnsupportedVersion {
        /// The version found.
        found: u32,
    },
    /// Bytes follow the body.
    #[error("{count} unexpected bytes after the gsv shared state snapshot")]
    TrailingBytes {
        /// How many.
        count: usize,
    },
    /// The body holds the wrong number of waiter counts.
    #[error("the gsv shared state snapshot holds {found} waiter counts, not {SYNCPOINT_COUNT}")]
    WaiterCounts {
        /// How many it holds.
        found: usize,
    },
    /// A syncpoint's waiter count and its watched bit disagree.
    #[error(
        "the gsv shared state snapshot gives syncpoint {id} {waiters} waiters, but the syncpoints have it {}",
        if *.watched { "watched" } else { "not watched" }
    )]
    WatchMismatch {
        /// The syncpoint.
        id: u8,
        /// Its waiter count in the snapshot.
        waiters: u32,
        /// Whether the syncpoints have it watched.
        watched: bool,
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

/// The snapshot body (module docs, "Snapshot format").
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct State {
    /// The waiter count of each syncpoint, by id.
    watchers: Vec<u32>,
}

impl State {
    /// The checks [`Shared::save`] and [`Shared::restore`] share, in the documented order.
    fn check(&self, syncpoints: &Syncpoints) -> Result<(), SnapshotError> {
        if self.watchers.len() != COUNT {
            return Err(SnapshotError::WaiterCounts {
                found: self.watchers.len(),
            });
        }
        let ids = (0..=u8::MAX).map_while(SyncpointId::new);
        for (id, &waiters) in ids.zip(&self.watchers) {
            let watched = syncpoints.is_watched(id);
            if (waiters != 0) != watched {
                return Err(SnapshotError::WatchMismatch {
                    id: id.get(),
                    waiters,
                    watched,
                });
            }
        }
        Ok(())
    }
}
