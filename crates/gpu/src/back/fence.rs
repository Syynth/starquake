//! Fenced actions and completion on the back thread (gpu-protocol spec, "Releases and
//! completion", "Ordering guarantees" 3–5, "Ordered address-table updates and barriers" step 4;
//! slice G15; rows 52, 70, 123 and 124 of `eden-nx2/NX2_NOTES.md`).
//!
//! The game sees GPU work as done only when the host GPU has finished it (owner's ruling of
//! 2026-11-12): a release, a syncpoint increment or a marker that comes after recorded work is
//! performed only once the render backend reports the list holding that work complete. This
//! module owns the open [`CommandList`], the fenced actions waiting on each submitted list, and
//! performing them; [`WaiterTable`] (in `waiters`) is the acquire waiter table those releases
//! latch.
//!
//! # Sources
//!
//! Paths are relative to the reference folder (`tools/refs-path.sh`), at the commits pinned in
//! `docs/references.md`. The shape is that of Eden-NX2's fence manager,
//! `eden-nx2/src/video_core/fence_manager.h` @ 50fa86bb (Copyright 2026 Eden Emulator Project,
//! GPL-3.0-or-later): `SignalFence` and `SkipEmptyFence` (an action with no work behind it runs
//! at once), and `ReleaseThreadFunc` (every fence the host has passed is released together, in
//! fence order). The release arithmetic is that of the QMD release callbacks in
//! `eden-nx2/src/video_core/engines/kepler_compute.cpp` @ 50fa86bb (`DispatchNativeCompute`'s
//! `SignalFence` lambdas: read, reduce, write, then `Nx2SemaphoreWaiters::OnWrite`), and the
//! 16-byte form that of `eden-nx2/src/video_core/query_cache/query_cache.h` @ 50fa86bb (the
//! payload widened to 64 bits at offset 0, the timestamp at offset 8; row 52: a 4-byte release
//! writes 4 bytes).
//!
//! # Attaching an action
//!
//! [`Fences::attach`] takes a [`FencedAction`] (a release or reduction, a syncpoint increment,
//! or a marker) and, in this order of preference (row 123; the owner's ruling: "signals with no
//! host work behind them are written at once"):
//!
//! 1. attaches it to the open list if the open list has recorded work ([`Attached::Open`]);
//! 2. otherwise to the newest list still in flight ([`Attached::InFlight`]);
//! 3. otherwise performs it at once ([`Attached::Performed`]), counted in
//!    [`FenceCounters::skipped_empty`].
//!
//! # Closing the open list
//!
//! [`Fences::close`] submits the open list ([`starquake_render::Backend::submit`]) and queues its
//! actions behind the returned [`FenceValue`]. The spec's triggers, each a [`CloseReason`]
//! counted in [`ClosedCounts`]:
//!
//! - the ring is empty ([`CloseReason::RingEmpty`]): [`Fences::wait`] closes before it sleeps;
//! - a present ([`CloseReason::Present`]), a wait for idle ([`CloseReason::WaitForIdle`]) or a
//!   `SET_REFERENCE` ([`CloseReason::SetReference`], decided by the owner on 2026-11-18): the
//!   executor (slice G19) calls [`Fences::close`];
//! - a fenced release one of whose 32-bit words has an unlatched registered waiter of the
//!   release's address space ([`CloseReason::Waiter`]): [`Fences::attach`] closes after
//!   attaching it, so the parked channel is not left waiting on recording. An 8-byte release
//!   covers two words and a 16-byte one four, and each is looked at
//!   (`a_waiter_on_a_wide_releases_high_word_closes_the_open_list`);
//! - [`CLOSE_AFTER`] (1 ms, a **guess**, so that a CPU spinning on a semaphore is not starved)
//!   of recording ([`CloseReason::Elapsed`]): [`Fences::record`] measures from the open list's
//!   first command, on [`FenceClock::elapsed`], and closes once a command is recorded at or past
//!   it. Nothing closes an open list that stops receiving commands before then, except the other
//!   triggers; the ring empties at the latest when recording stops.
//!
//! A list with no commands is never submitted: [`Fences::close`] returns `Ok(None)`, and since
//! actions join the open list only when it has commands, none is left behind.
//!
//! When the backend refuses a list, its commands are lost (the backend kept nothing) and its
//! actions follow the empty-fence rule as if it had had no work: they join the newest list in
//! flight, or are performed at once. The refusal is logged at error level and returned as
//! [`FenceError::Submit`]. Pinned by `a_refused_list_hands_its_actions_on` and
//! `a_refused_list_with_nothing_in_flight_performs_its_actions`.
//!
//! # Completion
//!
//! [`Fences::complete`] reads [`starquake_render::Backend::completed`] once and performs the
//! actions of every list in flight whose fence it has passed, first in, first out: lists in the
//! order they were submitted, and each list's actions in the order they were attached. A list
//! whose fence has passed waits behind an earlier list whose fence has not, so the order holds
//! even when a backend's fence values complete out of submission order (ordering guarantee 4;
//! pinned by `actions_stay_in_order_when_lists_complete_out_of_order`). All the actions of one
//! call are one batch: the front is woken once for it ([`FrontWake::wake`], with the batch's
//! [`Report`]), and not at all when the batch is empty (row 124; the owner's ruling: "several
//! are batched into one completion check").
//!
//! Performing each kind of action:
//!
//! - **A release** ([`ReleaseOp::Write`]) writes the payload at its width: 4 bytes for
//!   [`PayloadSize::Four`] (its low 32 bits; row 52), 8 for [`PayloadSize::Eight`]. With a
//!   timestamp it writes the 16-byte structure: the value widened to 64 bits, then
//!   [`FenceClock::gpu_timestamp`] at [`crate::sema::RELEASE_TIMESTAMP_OFFSET`], in one write.
//! - **A reduction** ([`ReleaseOp::Reduce`]) reads the semaphore at the payload's width, applies
//!   [`crate::sema::Reduction`] and writes the result back as a release of that value would.
//! - Either then latches, for each 32-bit word the release wrote (one for a 4-byte release, two
//!   for an 8-byte one, four for the 16-byte structure), every waiter of the release's address
//!   space on that word's address that the word meets ([`WaiterTable::on_write`]);
//!   [`Report::latched`] says whether one was. A waiter on the high word of an 8-byte release
//!   is latched by that word (`a_wide_release_latches_a_waiter_on_its_high_word`).
//! - A release whose memory access fails ([`GpuFault`]) writes nothing that the memory refused,
//!   latches nothing, and is logged at warning level and listed in [`Report::failed`]; the
//!   actions after it are still performed.
//! - **An increment** calls [`Syncpoints::increment`], which pushes
//!   [`GpuCompletion::SyncpointChanged`] when the syncpoint is watched.
//! - **A marker** applies its [`OrderedUpdate`], if it has one: [`GpuVm::apply`], then
//!   [`GpuCompletion::GpuOrdered`] with its ticket if it has one; its token goes in
//!   [`Report::markers`] for the front. Not here yet: dropping the range from the buffer and
//!   texture caches (G16, G17), the shader cache's `invalidate`, and freeing a destroyed sparse
//!   region's GPU-only storage (G22); each joins this step when its slice lands.
//!
//! **The debugging switch.** Under [`Mode::AtRecord`] every action is performed when it is
//! attached, whatever is recorded or in flight (the owner's ruling: "performing at record time is
//! only a debugging switch"). Lists are still closed and submitted as usual.
//!
//! # How the back thread sleeps
//!
//! [`Fences::wait`] is the back thread's wait for its next piece of work (gpu-protocol, "How the
//! back thread sleeps"; decided by the owner on 2026-11-18). It first looks at `completed()`
//! when a list is in flight: if the oldest list in flight has completed, it performs what has
//! completed and returns [`Woke::Completed`], leaving any ring entry for the next call, so a ring
//! that never empties (the back thread is the bottleneck) does not hold back releases,
//! increments and markers whose work the host has finished (pinned by
//! `completed_work_is_performed_while_the_ring_stays_busy`). This costs one `completed()` read
//! per entry while a list is in flight, and none when nothing is. When an entry ends the sleep
//! below while a completion is also pending, the entry is returned and the next call performs
//! the completion before taking another entry. Otherwise it returns the next ring entry if one
//! is there; otherwise it closes the open list (the ring is empty) and waits on the ring's
//! wake-up ([`crate::ring::Consumer::pop_or_wake`]), which the front's publish and the render
//! backend's completion wake both ring (`core` builds the backend with
//! [`crate::ring::ConsumerWake::completion_wake`]). The ring asks it, before it spins and again
//! as its last look before the sleep, whether `completed()` has passed the oldest list in
//! flight, and returns at once when it has; the ring's module docs say why a completion that
//! lands after that last look is not lost. Whatever ends the wait other than an entry or the
//! timeout, [`Fences::wait`] then performs what has completed and returns it. A sleep that reaches the ring's 100 ms
//! safety net and finds `completed()` moved since the last completion check is a lost wake: it is
//! logged at warning level, counted in [`FenceCounters::lost_wakes`], and the completed work is
//! performed. A wake that arrives just as the safety net runs out is returned as a wake, not
//! counted. A wake for a completion that an earlier [`Fences::complete`] already performed is
//! left on the ring, and makes the next wait return an empty [`Woke::Completed`] at once (a
//! spurious wake costs one look; pinned by `a_wake_for_work_already_performed_costs_one_look`).
//!
//! # Divergences from Eden-NX2
//!
//! Each is pinned by the test named, in `fence/tests.rs`:
//!
//! - **Every empty fence is skipped.** Eden-NX2's `SkipEmptyFence` still makes one fence in 64
//!   (`skipped_fences >= 64`) so the operations its fences carry (query slot cleanup) keep
//!   running; nothing rides on a fence here, so every action with no work behind it is performed
//!   at once (`every_action_with_nothing_behind_it_is_performed_at_once`).
//! - **One batch is everything that completed.** Eden-NX2's fencing thread releases at most 64
//!   fences per batch (`MaxBatch`); here one [`Fences::complete`] performs every list it finds
//!   passed, bounded by the lists in flight (`lists_completed_together_are_one_batch`).
//! - **Releases are written on the back thread**, in ring order, not on a separate fencing
//!   thread; the owner's decision of 2026-11-18. Eden-NX2's fencing thread releases a finished
//!   fence however busy the command thread is; here the back thread performs completed work
//!   only between ring entries, so [`Fences::wait`] puts completed work ahead of the next entry
//!   rather than draining the ring first. A completion is still performed no sooner than the
//!   entry the back thread is handling when it lands has been handled
//!   (`completed_work_is_performed_while_the_ring_stays_busy`).
//! - **A failed release is reported and skipped.** Eden-NX2's QMD callbacks refuse a release to
//!   an unmapped range when it is requested (`IsFullyMappedRange`) and its query path writes
//!   through a host pointer; here the memory is checked when the release is performed, and a
//!   fault is listed in the report (`a_release_that_faults_is_reported_and_the_batch_goes_on`).
//!
//! # Guesses and open items
//!
//! - [`CLOSE_AFTER`] (1 ms) is the spec's guess, to be measured.
//! - A reduction with a timestamp writes the 16-byte structure with the reduced value; no
//!   reference performs one. A trace of a `SEM_EXECUTE` reduction with `RELEASE_TIMESTAMP` set
//!   would settle it.
//! - The timestamp's source and unit are [`FenceClock`]'s; the spec's deferred queries take it
//!   "from guest time", and the wiring slice (G21) picks the clock. It is not recorded, so a
//!   replay that does not supply the same clock writes different timestamps.
//! - Releases to the syncpoint shim (gpu-protocol, "Semaphore acquires") go through the
//!   release's [`GpuMemory`] like any other; turning them into increments belongs to the
//!   production memory (G22), which is the one that translates addresses.
//! - Releases name no address space, so each [`FencedAction::Release`] carries the memory it
//!   writes and that memory's [`AsId`], chosen by the executor when it attaches the action
//!   (whether the ring's `Release` should name its channel is the executor's, slice G19); the
//!   waiter table's key is discussed in its own docs.
//! - A syncpoint increment closes nothing: a channel waiting on a submission fence is a front
//!   wait (slice G14, `crate::front`), not a waiter in the table, so the open list it waits on is
//!   closed by the other triggers (the ring empties once the front has nothing to send; at the
//!   latest [`CLOSE_AFTER`] of recording). Decided in slice G14; a measurement of fence-wait
//!   latency would show whether an increment trigger is worth adding.

mod waiters;

#[cfg(test)]
mod tests;

pub use waiters::{Waiter, WaiterTable};

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use starquake_render::{Backend, Command, CommandList, FenceValue, RenderError};
use thiserror::Error;

use crate::interface::{AsId, Completions, GpuCompletion, SyncpointId, Ticket, VmUpdate};
use crate::memory::{GpuFault, GpuMemory};
use crate::ring::{Consumer, MarkerToken, PopError, Popped, Release, ReleaseOp, RingEntry};
use crate::sema::{PayloadSize, RELEASE_TIMESTAMP_OFFSET, ReleaseForm};
use crate::syncpoint::Syncpoints;
use crate::vm::GpuVm;

/// How long the open list may record before a recorded command closes it (the spec's 1 ms, a
/// **guess**: long enough to batch, short enough that a CPU spinning on a semaphore is not
/// starved).
pub const CLOSE_AFTER: Duration = Duration::from_millis(1);

/// The back thread's clock, injected so tests and replays control it.
pub trait FenceClock: Send + Sync {
    /// Time since an arbitrary fixed origin, never going backwards; [`Fences::record`] measures
    /// [`CLOSE_AFTER`] on it.
    fn elapsed(&self) -> Duration;

    /// The timestamp a 16-byte release writes after its value (module docs, "Guesses").
    fn gpu_timestamp(&self) -> u64;
}

/// Where the back tells the front that it performed something: the front's wake-up (slice G14
/// rechecks parked channels and reports markers on it).
pub trait FrontWake: Send + Sync {
    /// Called once per non-empty batch of performed actions, after all of them (row 124).
    fn wake(&self, report: &Report);
}

/// An ordered update or barrier to apply when its marker is performed ("Ordered address-table
/// updates and barriers", step 4).
#[derive(Clone)]
pub struct OrderedUpdate {
    /// The address table the update applies to.
    pub vm: Arc<GpuVm>,
    /// The update; `None` for a barrier.
    pub update: Option<VmUpdate>,
    /// The ticket reported as [`GpuCompletion::GpuOrdered`] once applied; `None` when gsv asked
    /// for no report.
    pub ticket: Option<Ticket>,
}

/// An action performed only after the host GPU has finished the work recorded before it.
#[derive(Clone)]
pub enum FencedAction {
    /// A semaphore release or reduction, written through `memory`.
    Release {
        /// The address space the release writes, as the executor resolved it.
        memory: Arc<dyn GpuMemory>,
        /// The id of that address space: only its waiters are latched (slice G14).
        space: AsId,
        /// What to write.
        release: Release,
    },
    /// A syncpoint increment.
    Increment(SyncpointId),
    /// A marker: everything before it has been performed when it is.
    Marker {
        /// The token reported to the front.
        token: MarkerToken,
        /// The ordered update or barrier waiting on it, if any.
        ordered: Option<OrderedUpdate>,
    },
}

/// Why the open list was closed (module docs, "Closing the open list").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CloseReason {
    /// The ring is empty.
    RingEmpty,
    /// A present arrived.
    Present,
    /// A wait for idle arrived.
    WaitForIdle,
    /// A fenced release's address has a registered waiter.
    Waiter,
    /// The open list has recorded for [`CLOSE_AFTER`].
    Elapsed,
    /// A `SET_REFERENCE` arrived ([`RingEntry::SetReference`]): the back closes the open list
    /// with no fenced action attached, as Eden-NX2's back does (`puller.cpp`, `RefCnt` →
    /// `SignalReference`; decided by the owner on 2026-11-18). The executor (slice G19) calls
    /// [`Fences::close`] with it.
    SetReference,
}

/// When fenced actions are performed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// After the host GPU has finished the work before them (the normal mode).
    Fenced,
    /// When they are attached: the debugging switch.
    AtRecord,
}

/// A release whose memory access failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("the release at GPU address {address:#x} was not performed: {fault}")]
pub struct ReleaseFailure {
    /// The release's GPU address.
    pub address: u64,
    /// The memory's fault.
    pub fault: GpuFault,
}

/// What one batch performed, for the front.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Releases and reductions written.
    pub released: u64,
    /// Syncpoint increments made.
    pub incremented: u64,
    /// Whether a release latched a waiter.
    pub latched: bool,
    /// Markers performed, in order.
    pub markers: Vec<MarkerToken>,
    /// Releases that failed, in order.
    pub failed: Vec<ReleaseFailure>,
}

impl Report {
    /// Whether the batch performed or tried nothing.
    pub fn is_empty(&self) -> bool {
        self.released == 0
            && self.incremented == 0
            && !self.latched
            && self.markers.is_empty()
            && self.failed.is_empty()
    }
}

/// Where [`Fences::attach`] put an action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Attached {
    /// On the open list.
    Open,
    /// On the list in flight with this fence value (the newest, or the open list just closed
    /// for a waiter).
    InFlight(FenceValue),
    /// Performed at once, with what it did.
    Performed(Report),
}

/// Why a fence operation failed.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum FenceError {
    /// The render backend refused the open list (module docs, "Closing the open list").
    #[error("the render backend refused a command list: {0}")]
    Submit(RenderError),
    /// The ring failed.
    #[error(transparent)]
    Ring(#[from] PopError),
}

/// What [`Fences::wait`] ended with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Woke {
    /// The next ring entry.
    Entry(RingEntry<'static>),
    /// Completion processing ran; the report is empty for a wake that found nothing.
    Completed(Report),
    /// The timeout passed with no entry and no wake.
    TimedOut,
}

/// How many lists were closed for each [`CloseReason`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClosedCounts {
    /// [`CloseReason::RingEmpty`].
    pub ring_empty: u64,
    /// [`CloseReason::Present`].
    pub present: u64,
    /// [`CloseReason::WaitForIdle`].
    pub wait_for_idle: u64,
    /// [`CloseReason::Waiter`].
    pub waiter: u64,
    /// [`CloseReason::Elapsed`].
    pub elapsed: u64,
    /// [`CloseReason::SetReference`].
    pub set_reference: u64,
}

impl ClosedCounts {
    fn count(&mut self, reason: CloseReason) {
        let slot = match reason {
            CloseReason::RingEmpty => &mut self.ring_empty,
            CloseReason::Present => &mut self.present,
            CloseReason::WaitForIdle => &mut self.wait_for_idle,
            CloseReason::Waiter => &mut self.waiter,
            CloseReason::Elapsed => &mut self.elapsed,
            CloseReason::SetReference => &mut self.set_reference,
        };
        *slot = slot.saturating_add(1);
    }
}

/// The fence path's counters (gpu-protocol, "Observability": fenced actions and how many were
/// skipped as empty).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FenceCounters {
    /// Actions given to [`Fences::attach`].
    pub fenced: u64,
    /// Actions performed at once under [`Mode::Fenced`] because no list had work before them.
    pub skipped_empty: u64,
    /// Lists submitted, by why they were closed.
    pub closed: ClosedCounts,
    /// Safety-net rechecks that found `completed()` moved without a wake.
    pub lost_wakes: u64,
    /// Releases that failed.
    pub failed_releases: u64,
}

/// What performing an action touches besides guest memory, shared with the rest of the GPU.
#[derive(Clone)]
pub struct FenceEnv {
    /// The syncpoint counters increments go to.
    pub syncpoints: Arc<Syncpoints>,
    /// The acquire waiter table releases latch.
    pub waiters: Arc<WaiterTable>,
    /// gsv's completion queue, for [`GpuCompletion::GpuOrdered`].
    pub completions: Arc<dyn Completions>,
    /// The back thread's clock.
    pub clock: Arc<dyn FenceClock>,
    /// The front's wake-up.
    pub front: Arc<dyn FrontWake>,
}

/// A submitted list's fence value and the actions waiting on it.
struct InFlight {
    fence: FenceValue,
    actions: Vec<FencedAction>,
}

/// The back thread's open list and fenced actions (module docs). Owned by the back thread.
pub struct Fences {
    backend: Arc<dyn Backend>,
    env: FenceEnv,
    mode: Mode,
    open: CommandList,
    /// [`FenceClock::elapsed`] when the open list's first command was recorded.
    opened_at: Duration,
    open_actions: Vec<FencedAction>,
    /// Submitted lists, oldest first.
    in_flight: VecDeque<InFlight>,
    /// `completed()` as the last completion check read it.
    seen: FenceValue,
    counters: FenceCounters,
}

/// The low 32 bits of `value`: what a 32-bit acquire reads from a release of it.
fn low_word(value: u64) -> u32 {
    let [a, b, c, d, ..] = value.to_le_bytes();
    u32::from_le_bytes([a, b, c, d])
}

/// The bytes a performed release wrote, from its address.
struct Written {
    bytes: [u8; LARGEST_RELEASE_BYTES],
    len: usize,
}

impl Written {
    /// Each 32-bit word written, with its GPU address, in address order (a word whose address
    /// would pass `u64::MAX` is left out; the write itself would have failed).
    fn words(&self, address: u64) -> impl Iterator<Item = (u64, u32)> + '_ {
        let (words, _) = self.bytes.get(..self.len).unwrap_or(&[]).as_chunks::<4>();
        (0u64..).zip(words).filter_map(move |(index, bytes)| {
            let at = address.checked_add(index.checked_mul(SEMAPHORE_WORD_BYTES)?)?;
            Some((at, u32::from_le_bytes(*bytes)))
        })
    }
}

/// Bytes of the 32-bit semaphore word an acquire reads: the unit a release's words are latched
/// in.
const SEMAPHORE_WORD_BYTES: u64 = 4;

/// The GPU address of each 32-bit word `release` writes, with its index (module docs, "Closing
/// the open list"): one for a 4-byte release, two for an 8-byte one, four for the 16-byte
/// structure.
fn release_words(release: &Release) -> impl Iterator<Item = (u64, u64)> + '_ {
    let words = release.form.bytes() / SEMAPHORE_WORD_BYTES;
    (0..words).filter_map(move |index| {
        let at = release
            .address
            .checked_add(index.checked_mul(SEMAPHORE_WORD_BYTES)?)?;
        Some((at, index))
    })
}

/// Performs `release` through `memory` (module docs, "Completion") and returns the bytes it
/// wrote.
fn write_release(
    memory: &dyn GpuMemory,
    release: &Release,
    clock: &dyn FenceClock,
) -> Result<Written, GpuFault> {
    let address = release.address;
    let size = release.form.size;
    let payload = match size {
        PayloadSize::Four => u64::from(low_word(release.payload)),
        PayloadSize::Eight => release.payload,
    };
    let value = match release.op {
        ReleaseOp::Write => payload,
        ReleaseOp::Reduce(reduction) => match size {
            PayloadSize::Four => {
                let mut old = [0; 4];
                memory.read(address, &mut old)?;
                u64::from(reduction.apply_u32(u32::from_le_bytes(old), low_word(payload)))
            }
            PayloadSize::Eight => {
                let mut old = [0; 8];
                memory.read(address, &mut old)?;
                reduction.apply_u64(u64::from_le_bytes(old), payload)
            }
        },
    };
    // `value` already fits the payload's width, so its little-endian bytes start with the
    // payload and the form's length cuts it to 4, 8 or (widened to 64 bits) 16 bytes.
    let mut bytes = [0; LARGEST_RELEASE_BYTES];
    put(&mut bytes, 0, value);
    if release.form.timestamp {
        put(&mut bytes, TIMESTAMP_AT, clock.gpu_timestamp());
    }
    let len = usize::try_from(release.form.bytes())
        .unwrap_or(LARGEST_RELEASE_BYTES)
        .min(LARGEST_RELEASE_BYTES);
    memory.write(address, bytes.get(..len).unwrap_or(&bytes))?;
    Ok(Written { bytes, len })
}

/// The largest form a release writes: the 16-byte structure with a timestamp.
const LARGEST_RELEASE: ReleaseForm = ReleaseForm {
    size: PayloadSize::Eight,
    timestamp: true,
};

/// The bytes of [`LARGEST_RELEASE`]: the buffer [`write_release`] builds every form in, so a
/// release allocates nothing. Two 64-bit words, checked against [`ReleaseForm::bytes`] when the
/// crate is compiled.
const LARGEST_RELEASE_BYTES: usize = 2 * size_of::<u64>();
const _: () = assert!(LARGEST_RELEASE_BYTES as u64 == LARGEST_RELEASE.bytes());

/// [`RELEASE_TIMESTAMP_OFFSET`] as an index into a release's bytes, checked against it when the
/// crate is compiled.
const TIMESTAMP_AT: usize = size_of::<u64>();
const _: () = assert!(TIMESTAMP_AT as u64 == RELEASE_TIMESTAMP_OFFSET);

/// Copies `word`'s little-endian bytes into `bytes` from `at`, dropping any that fall past its
/// end.
fn put(bytes: &mut [u8], at: usize, word: u64) {
    for (slot, byte) in bytes.iter_mut().skip(at).zip(word.to_le_bytes()) {
        *slot = byte;
    }
}

impl Fences {
    /// The fence path for `backend`, with nothing recorded or in flight.
    pub fn new(backend: Arc<dyn Backend>, env: FenceEnv, mode: Mode) -> Self {
        let seen = backend.completed();
        Self {
            backend,
            env,
            mode,
            open: CommandList::new(),
            opened_at: Duration::ZERO,
            open_actions: Vec::new(),
            in_flight: VecDeque::new(),
            seen,
            counters: FenceCounters::default(),
        }
    }

    /// The counters so far.
    pub fn counters(&self) -> FenceCounters {
        self.counters
    }

    /// Commands recorded on the open list.
    pub fn open_commands(&self) -> usize {
        self.open.commands.len()
    }

    /// Lists submitted whose actions are not yet performed.
    pub fn lists_in_flight(&self) -> usize {
        self.in_flight.len()
    }

    /// Records `command` on the open list, and closes the list if it has recorded for
    /// [`CLOSE_AFTER`] (module docs, "Closing the open list").
    pub fn record(&mut self, command: Command) -> Result<(), FenceError> {
        let now = self.env.clock.elapsed();
        if self.open.commands.is_empty() {
            self.opened_at = now;
        }
        self.open.push(command);
        if now.saturating_sub(self.opened_at) >= CLOSE_AFTER {
            self.close(CloseReason::Elapsed)?;
        }
        Ok(())
    }

    /// Attaches `action` by the empty-fence rule, or performs it at once under
    /// [`Mode::AtRecord`] (module docs, "Attaching an action").
    ///
    /// # Errors
    ///
    /// [`FenceError::Submit`] when the action's waiter closed the open list and the backend
    /// refused it; the action then went where the module docs say.
    pub fn attach(&mut self, action: FencedAction) -> Result<Attached, FenceError> {
        self.counters.fenced = self.counters.fenced.saturating_add(1);
        if self.mode == Mode::AtRecord {
            return Ok(Attached::Performed(self.perform_now(vec![action])));
        }
        if !self.open.commands.is_empty() {
            let waited_on = match &action {
                FencedAction::Release { space, release, .. } => release_words(release)
                    .any(|(address, _)| self.env.waiters.has_waiter(*space, address)),
                FencedAction::Increment(_) | FencedAction::Marker { .. } => false,
            };
            self.open_actions.push(action);
            if waited_on {
                return Ok(self
                    .close(CloseReason::Waiter)?
                    .map_or(Attached::Open, Attached::InFlight));
            }
            return Ok(Attached::Open);
        }
        if let Some(newest) = self.in_flight.back_mut() {
            newest.actions.push(action);
            return Ok(Attached::InFlight(newest.fence));
        }
        self.counters.skipped_empty = self.counters.skipped_empty.saturating_add(1);
        Ok(Attached::Performed(self.perform_now(vec![action])))
    }

    /// Submits the open list, if it has commands, and queues its actions behind its fence value
    /// (module docs, "Closing the open list"). `Ok(None)` when there was nothing to submit.
    pub fn close(&mut self, reason: CloseReason) -> Result<Option<FenceValue>, FenceError> {
        if self.open.commands.is_empty() {
            return Ok(None);
        }
        let list = std::mem::take(&mut self.open);
        let actions = std::mem::take(&mut self.open_actions);
        match self.backend.submit(list) {
            Ok(fence) => {
                self.counters.closed.count(reason);
                tracing::trace!(?reason, fence = fence.get(), "command list submitted");
                self.in_flight.push_back(InFlight { fence, actions });
                Ok(Some(fence))
            }
            Err(error) => {
                tracing::error!(
                    %error,
                    ?reason,
                    "the render backend refused a command list; its fenced actions go on as if it \
                     had no work"
                );
                if let Some(newest) = self.in_flight.back_mut() {
                    newest.actions.extend(actions);
                } else if !actions.is_empty() {
                    self.perform_now(actions);
                }
                Err(FenceError::Submit(error))
            }
        }
    }

    /// Performs the actions of every list whose fence `completed()` has passed, first in, first
    /// out, as one batch, and wakes the front once if the batch did anything (module docs,
    /// "Completion").
    pub fn complete(&mut self) -> Report {
        let completed = self.backend.completed();
        self.seen = completed;
        let mut report = Report::default();
        while self
            .in_flight
            .front()
            .is_some_and(|list| list.fence <= completed)
        {
            if let Some(list) = self.in_flight.pop_front() {
                for action in list.actions {
                    self.perform(action, &mut report);
                }
            }
        }
        self.finish(&report);
        report
    }

    /// The back thread's wait for its next piece of work, at most `timeout` (module docs, "How
    /// the back thread sleeps").
    ///
    /// When `completed()` has passed the oldest list in flight, it performs that work and
    /// returns [`Woke::Completed`] without taking an entry, even when the ring has one, so the
    /// executor needs no completion check of its own between entries.
    ///
    /// # Errors
    ///
    /// [`FenceError::Ring`] from the ring; [`FenceError::Submit`] when closing the open list
    /// for an empty ring failed.
    pub fn wait(&mut self, ring: &mut Consumer, timeout: Duration) -> Result<Woke, FenceError> {
        if self.oldest_has_completed() {
            // Completed work first, so a ring that never empties cannot hold back the releases
            // of lists the host has finished; the entry stays on the ring for the next call.
            return Ok(Woke::Completed(self.complete()));
        }
        if let Some(entry) = ring.try_pop()? {
            return Ok(Woke::Entry(entry));
        }
        self.close(CloseReason::RingEmpty)?;
        let idle = || !self.oldest_has_completed();
        match ring.pop_or_wake(timeout, idle)? {
            Popped::Entry(entry) => Ok(Woke::Entry(entry)),
            Popped::Woken => Ok(Woke::Completed(self.complete())),
            Popped::Recheck => {
                let completed = self.backend.completed();
                if completed != self.seen {
                    self.counters.lost_wakes = self.counters.lost_wakes.saturating_add(1);
                    tracing::warn!(
                        seen = self.seen.get(),
                        completed = completed.get(),
                        "the render backend's completed value moved without a completion wake"
                    );
                }
                Ok(Woke::Completed(self.complete()))
            }
            Popped::TimedOut => Ok(Woke::TimedOut),
        }
    }

    /// Whether `completed()` has passed the oldest list in flight, so [`Fences::complete`] has
    /// work to perform. Reads `completed()` only when a list is in flight.
    fn oldest_has_completed(&self) -> bool {
        self.in_flight
            .front()
            .is_some_and(|list| list.fence <= self.backend.completed())
    }

    /// Performs `actions` in order as one batch, outside any list.
    fn perform_now(&mut self, actions: Vec<FencedAction>) -> Report {
        let mut report = Report::default();
        for action in actions {
            self.perform(action, &mut report);
        }
        self.finish(&report);
        report
    }

    /// Performs one action into `report` (module docs, "Completion").
    fn perform(&mut self, action: FencedAction, report: &mut Report) {
        match action {
            FencedAction::Release {
                memory,
                space,
                release,
            } => {
                match write_release(&*memory, &release, &*self.env.clock) {
                    Ok(written) => {
                        report.released = report.released.saturating_add(1);
                        for (address, word) in written.words(release.address) {
                            if self.env.waiters.on_write(space, address, word) {
                                report.latched = true;
                            }
                        }
                    }
                    Err(fault) => {
                        let failure = ReleaseFailure {
                            address: release.address,
                            fault,
                        };
                        tracing::warn!(%failure, "fenced release failed");
                        self.counters.failed_releases =
                            self.counters.failed_releases.saturating_add(1);
                        report.failed.push(failure);
                    }
                }
            }
            FencedAction::Increment(id) => {
                self.env.syncpoints.increment(id);
                report.incremented = report.incremented.saturating_add(1);
            }
            FencedAction::Marker { token, ordered } => {
                if let Some(ordered) = ordered {
                    if let Some(update) = ordered.update {
                        ordered.vm.apply(&update);
                    }
                    if let Some(ticket) = ordered.ticket {
                        self.env
                            .completions
                            .push(GpuCompletion::GpuOrdered { ticket });
                    }
                }
                report.markers.push(token);
            }
        }
    }

    /// Wakes the front once for a batch that did anything.
    fn finish(&self, report: &Report) {
        if !report.is_empty() {
            self.env.front.wake(report);
        }
    }
}
