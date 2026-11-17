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
//! [`Locked::unwatch`]), the syncpt sessions' waiter table ([`Waiters`]: [`Locked::arm`],
//! [`Locked::disarm`]; slice 4), the GPU memory bookkeeping ([`Locked::gpu`], slice 7) and the
//! shared syncpoint page ([`page`], slice 5); later slices add vi2's queues and the native object
//! tables (spec, "Concurrency inside gsv").
//!
//! Each armed wait holds one watch of its syncpoint: [`Locked::arm`] takes it, and gives back
//! the watch of the wait it replaces; [`Locked::disarm`] and the drain's firing give it back
//! (`an_armed_wait_holds_one_watch_of_its_syncpoint`). The shared page's hold
//! holds one more watch of each syncpoint it asks about ([`page`], "When a request is
//! answered"), so a syncpoint's waiter count is at least its number of armed waits. [`Locked::unwatch`] keeps it so: it refuses to take a watch an armed wait holds
//! ([`WatchError::HeldByWaits`]), which would leave the wait's id unwatched and the wait never
//! fired (`unwatch_cannot_take_an_armed_wait_s_watch`).
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
//! What the drain does with each completion, so far, is record it, then act on it as the table
//! says:
//!
//! | Completion | What the drain does |
//! |---|---|
//! | `GpuCompletion::SyncpointChanged` | records `SyncpointObserved` ([`crate::note`]), then fires every armed wait on the id that the drained value satisfies: takes it out of the table, unwatches its id, and signals its event through the kernel ([`TimerHost::signal_event`]), in ascending event order (`a_change_fires_its_waits_after_its_note_in_event_order`) |
//! | `GpuCompletion::GpuOrdered` | records `GpuOrderedDone`; then, if a gpu worker job waits on the ticket ([`crate::gpu::GpuMemory::ordered_done`]), publishes the ranges it removed again, releases its device-share locks as a `DeviceShareReleased` is released (below), and only then signals its event ([`TimerHost::signal_event`]) |
//! | `GpuCompletion::GpuQueueRoom` | records `QueueRoom` |
//! | [`Completion::GuestSignalled`] | records nothing (the guest's `SignalEvent` is recorded as its system call); if the event is the bound shared page's request event, marks the page signalled, and the next page turn ([`Shared::tick_page`]) reads the request and records what it answers ([`page`]) |
//! | [`Completion::DeviceShareReleased`] | unlocks through the kernel with the token ([`TimerHost::unlock_device_share`]), which checks it; records nothing (the spec's "What is recorded" has no note for it) |
//!
//! The kernel's refusal of a release is logged, and the drain goes on: a token whose process
//! has ended is refused as a matter of course (the process ended while the GPU still had work
//! that used its memory), and is logged at debug level; any other refusal means gsv's own
//! bookkeeping is wrong, and is logged as an error. Either way the lock stays as the kernel
//! keeps it, and the next completions are acted on
//! (`a_refused_release_is_logged_and_the_drain_goes_on`; the levels are pinned by
//! `tests/release_log_levels.rs`, a test binary of its own).
//!
//! A job's event that the kernel refuses to signal is logged as an error, and the drain goes on.
//!
//! **Waits fire on the drained value, not the counter.** A wait is compared with the value the
//! drain took from the queue, the one it records as `SyncpointObserved`, so that what gsv acts
//! on is what replay feeds back. Drained values are not monotone (the completion module's
//! "Coalescing"), which a wait tolerates: a wait fires on the first value that satisfies it and
//! is gone, and a stale value drained after a newer one satisfies no wait the newer one did
//! not, unless the two are 2^31 or more apart under the wrapping comparison. A refused signal
//! (the kernel no longer has the event) is logged as an error and the drain goes on; the wait
//! is gone either way (`a_refused_signal_is_logged_and_the_drain_goes_on`; the level is pinned
//! by `tests/signal_log_level.rs`, a test binary of its own).
//!
//! Later slices act on them as well: deferred submissions (11).
//!
//! **The page turn.** After each drain the timer thread runs [`Shared::tick_page`] with the
//! turn's time, which answers the shared page's requests ([`page`], "When a request is
//! answered"), and sleeps no later than [`Shared::page_deadline`].
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
//! and the waits here). `postcard`, after a header of [`SNAPSHOT_MAGIC`] and
//! [`SNAPSHOT_VERSION`] that is decoded and checked before the body. The body is the waiter
//! count of each syncpoint, by id, then the armed waits in ascending event order, each its
//! event, syncpoint id, threshold and mode byte (`crate::syncpt::waiters`, "Snapshot form").
//! The golden-bytes test `the_snapshot_layout_is_pinned` pins it; changing it means bumping
//! [`SNAPSHOT_VERSION`]. Version 2 added the waits. The queue has its own snapshot
//! ([`CompletionQueue::save`]), because it must exist before the `Syncpoints` that pushes to it.
//! The GPU memory bookkeeping has its own too ([`crate::gpu::saved`]), and so has the shared page
//! ([`Shared::save_page`], restored after this one by [`Shared::restore_page`]); composing them
//! into this one is the wiring slice's (16).
//!
//! A snapshot is a file and is not trusted. [`Shared::restore`] checks, in this order, and
//! returns the first failure: the magic, the version, that the body decodes, that no bytes
//! follow it, that it holds `SYNCPOINT_COUNT` counts, then each wait in list order
//! ([`SavedWaitError`]), and then, id by id, that the count is not zero exactly when the
//! restored `Syncpoints` has the id watched, and that the count is at least the number of waits
//! armed on the id ([`SnapshotError::WaitsOutnumberWatchers`]). [`Shared::save`] runs the same
//! checks on what it is about to write.
//!
//! Not checked here: that each wait's event is a slot event of an open syncpt session, which
//! the syncpt server's own snapshot holds. A whole-machine restore (slice 16) checks that
//! relation across the two.

use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::{GpuCompletion, SYNCPOINT_COUNT, SyncpointId};
use starquake_gpu::syncpoint::Syncpoints;
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::result::{KernelError, SvcResult};
use starquake_trace::{EventKind, ServiceId};
use thiserror::Error;

use crate::completion::{Completion, CompletionQueue};
use crate::gpu::GpuMemory;
use crate::note::Note;
use crate::syncpt::waiters::{SavedWait, SavedWaitError, Wait, Waiters};

pub mod page;
#[cfg(test)]
mod tests;

use page::Page;

/// [`SYNCPOINT_COUNT`] as an index bound.
const COUNT: usize = SYNCPOINT_COUNT as usize;

/// First bytes of every gsv shared-state snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SQGS";

/// Layout version written into every gsv shared-state snapshot and required when reading one.
pub const SNAPSHOT_VERSION: u32 = 2;

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

    /// Signals `event`, an event a gsv server made, through the kernel: how the drain fires an
    /// armed wait and completes a gpu worker job (module docs, "Lock order and the drain").
    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()>;

    /// Copies `out.len()` bytes at `offset` into the transfer memory `memory` names into `out`,
    /// through the kernel, which checks the transfer memory as it is now: how the page turn
    /// reads the shared page ([`page`]). The kernel's errors are its
    /// `IpcContext::read_transfer_memory`'s (`InvalidHandle` once the transfer memory is gone,
    /// `InvalidCurrentMemory` unless the bytes lie wholly inside it).
    ///
    /// The default answers `NotImplemented`: a host for a run with no shared page need not offer
    /// it.
    fn read_transfer_memory(
        &mut self,
        _memory: ObjectId,
        _offset: u64,
        _out: &mut [u8],
    ) -> SvcResult<()> {
        Err(KernelError::NotImplemented)
    }

    /// Copies `data` into the transfer memory `memory` names, at `offset`, through the kernel:
    /// how the page turn writes the shared page. The same rules and default as
    /// [`TimerHost::read_transfer_memory`].
    fn write_transfer_memory(
        &mut self,
        _memory: ObjectId,
        _offset: u64,
        _data: &[u8],
    ) -> SvcResult<()> {
        Err(KernelError::NotImplemented)
    }
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
    /// The GPU memory bookkeeping (slice 7).
    gpu: GpuMemory,
    /// The syncpt sessions' armed waits.
    waiters: Waiters,
    /// The shared syncpoint page (slice 5).
    page: Page,
}

impl GsvShared {
    /// No waiters on any syncpoint: the state over fresh syncpoints, which watch nothing.
    fn fresh() -> Self {
        Self {
            watchers: vec![0; COUNT],
            gpu: GpuMemory::new(),
            waiters: Waiters::default(),
            page: Page::default(),
        }
    }

    /// The syncpt sessions' armed waits.
    #[must_use]
    pub const fn waiters(&self) -> &Waiters {
        &self.waiters
    }

    /// The number of waiters on syncpoint `id`.
    #[must_use]
    pub fn watchers(&self, id: SyncpointId) -> u32 {
        self.watchers
            .get(usize::from(id.get()))
            .copied()
            .unwrap_or(0)
    }

    /// The GPU memory bookkeeping.
    #[must_use]
    pub fn gpu_memory(&self) -> &GpuMemory {
        &self.gpu
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
    /// Every waiter left on the syncpoint is the watch of an armed wait, which only disarming
    /// or firing the wait gives back (module docs, "The pieces").
    #[error("syncpoint {} has no waiters but its {waits} armed waits", .id.get())]
    HeldByWaits {
        /// The syncpoint.
        id: SyncpointId,
        /// The waits armed on it, each holding one of its waiters.
        waits: usize,
    },
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

    /// Removes a waiter on syncpoint `id`, marking it not watched if that was the last. It
    /// refuses, in this order, an id with no waiters ([`WatchError::NotWatched`]) and one whose
    /// waiters are all armed waits' watches ([`WatchError::HeldByWaits`]): a wait's watch is
    /// given back only by disarming or firing the wait (module docs, "The pieces"). Nothing
    /// changes on an error.
    pub fn unwatch(&mut self, id: SyncpointId) -> Result<(), WatchError> {
        let count = self.watchers(id);
        if count == 0 {
            return Err(WatchError::NotWatched(id));
        }
        let waits = self.state.waiters.on(id);
        if usize::try_from(count).is_ok_and(|count| count <= waits) {
            return Err(WatchError::HeldByWaits { id, waits });
        }
        self.drop_watch(id)
    }

    /// Removes a waiter on syncpoint `id` without the armed-wait check, marking it not watched
    /// if that was the last: for giving back the watch of a wait already out of the table, and
    /// for [`Locked::unwatch`] once it has checked. Nothing changes on an error.
    fn drop_watch(&mut self, id: SyncpointId) -> Result<(), WatchError> {
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

    /// The GPU memory bookkeeping, to change (the gpu server's handlers, slice 7).
    pub fn gpu(&mut self) -> &mut GpuMemory {
        &mut self.state.gpu
    }

    /// Arms `wait` on `event`, replacing the wait armed on it before, and returns the wait's
    /// counter, read after its id was watched ([`Locked::watch`]): the caller compares the
    /// threshold with it, and any later increment is pushed to the queue. The replaced wait's
    /// watch is given back (module docs, "The pieces"). Nothing changes on an error.
    pub fn arm(&mut self, event: ObjectId, wait: Wait) -> Result<u32, WatchError> {
        let value = self.watch(wait.id())?;
        if let Some(replaced) = self.state.waiters.insert(event, wait) {
            self.release(replaced.id());
        }
        Ok(value)
    }

    /// Takes the wait armed on `event` out of the table, unwatching its id, and returns it;
    /// `None` if no wait was armed on it.
    pub fn disarm(&mut self, event: ObjectId) -> Option<Wait> {
        let wait = self.state.waiters.remove(event)?;
        self.release(wait.id());
        Some(wait)
    }

    /// Fires the waits on `id` that `value` satisfies: takes them out, unwatches their id, and
    /// returns their events in ascending order (module docs, "Lock order and the drain").
    fn fire(&mut self, id: SyncpointId, value: u32) -> Vec<ObjectId> {
        let fired = self.state.waiters.take_reached(id, value);
        for _ in &fired {
            self.release(id);
        }
        fired
    }

    /// Gives back the watch a wait on `id` held, once the wait is out of the table. A wait
    /// always holds one, so a refusal means gsv's own bookkeeping is wrong: it is logged as an
    /// error.
    fn release(&mut self, id: SyncpointId) {
        if let Err(error) = self.drop_watch(id) {
            tracing::error!(%error, "gsv: an armed wait held no watch of its syncpoint");
        }
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
        let mut state = self.lock();
        let taken = self.queue.take_all();
        for completion in &taken {
            match *completion {
                Completion::DeviceShareReleased { token } => release(host, token),
                Completion::GuestSignalled { event } => {
                    tracing::trace!(?completion, "gsv drained a guest signal");
                    state.state.page.guest_signalled(event);
                }
                Completion::Gpu(gpu) => {
                    host.record(note(gpu).event(self.service));
                    if let GpuCompletion::GpuOrdered { ticket } = gpu
                        && let Some(finished) = state.gpu().ordered_done(ticket)
                    {
                        for token in finished.releases {
                            release(host, token);
                        }
                        if let Err(error) = host.signal_event(finished.event) {
                            tracing::error!(%error, ?ticket, "gsv: a gpu job's event was not signalled");
                        }
                    }
                    if let GpuCompletion::SyncpointChanged { id, value } = gpu {
                        for event in state.fire(id, value) {
                            signal(host, event);
                        }
                    }
                }
            }
        }
        drop(state);
        taken
    }

    /// Whether gsv's shared lock is held now, for tests of the lock order.
    #[cfg(test)]
    pub(crate) fn lock_is_held(&self) -> bool {
        matches!(
            self.state.try_lock(),
            Err(std::sync::TryLockError::WouldBlock)
        )
    }

    /// Sets syncpoint `id`'s waiter count to `count` behind the watched bit's back, for tests
    /// of a count at its maximum.
    #[cfg(test)]
    pub(crate) fn set_watchers(&self, id: SyncpointId, count: u32) {
        if let Some(watchers) = self.lock().state.watchers.get_mut(usize::from(id.get())) {
            *watchers = count;
        }
    }

    /// Writes [`GsvShared`] (module docs, "Snapshot format"). Take it while nothing changes the
    /// state or the watched bitmap.
    pub fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let saved = {
            let state = self.lock();
            State {
                watchers: state.watchers.clone(),
                waits: state.waiters.save(),
            }
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
        let waiters = saved.check(&syncpoints)?;
        let state = GsvShared {
            watchers: saved.watchers,
            gpu: GpuMemory::new(),
            waiters,
            page: Page::default(),
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

/// Signals `event` through `host` to fire its wait, logging a refusal (module docs, "Lock order
/// and the drain").
fn signal(host: &mut dyn TimerHost, event: ObjectId) {
    if let Err(error) = host.signal_event(event) {
        tracing::error!(%error, ?event, "the kernel refused to signal an armed wait's event");
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
    /// A saved wait could not be read back.
    #[error("the gsv shared state snapshot's waits: {0}")]
    Wait(#[from] SavedWaitError),
    /// A syncpoint has more armed waits than waiters: each wait holds one watch.
    #[error(
        "the gsv shared state snapshot arms {waits} waits on syncpoint {id}, which has {waiters} waiters"
    )]
    WaitsOutnumberWatchers {
        /// The syncpoint.
        id: u8,
        /// The waits armed on it.
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

/// The snapshot body (module docs, "Snapshot format").
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct State {
    /// The waiter count of each syncpoint, by id.
    watchers: Vec<u32>,
    /// The armed waits, in ascending event order.
    waits: Vec<SavedWait>,
}

impl State {
    /// The checks [`Shared::save`] and [`Shared::restore`] share, in the documented order.
    /// Returns the waiter table the snapshot holds.
    fn check(&self, syncpoints: &Syncpoints) -> Result<Waiters, SnapshotError> {
        if self.watchers.len() != COUNT {
            return Err(SnapshotError::WaiterCounts {
                found: self.watchers.len(),
            });
        }
        let waiters = Waiters::restore(&self.waits)?;
        let ids = (0..=u8::MAX).map_while(SyncpointId::new);
        for (id, &count) in ids.zip(&self.watchers) {
            let watched = syncpoints.is_watched(id);
            if (count != 0) != watched {
                return Err(SnapshotError::WatchMismatch {
                    id: id.get(),
                    waiters: count,
                    watched,
                });
            }
            let waits = waiters.on(id);
            if u32::try_from(waits).map_or(true, |waits| waits > count) {
                return Err(SnapshotError::WaitsOutnumberWatchers {
                    id: id.get(),
                    waits,
                    waiters: count,
                });
            }
        }
        Ok(waiters)
    }
}
