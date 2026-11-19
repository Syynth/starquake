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
//! - a catch-up ([`CloseReason::CatchUp`]): the executor must see the host GPU finish the open
//!   list before it goes on, for a write-back's readbacks ([`Fences::catch_up`]) or a
//!   conditional-render report a release will write ([`Fences::settle`], below);
//! - a channel removed ([`CloseReason::ChannelRemoved`]): the executor closes a list its 3D
//!   engine may have drawn into before that engine's draw path destroys its host objects;
//! - a fenced release one of whose 32-bit words has an unlatched registered waiter of the
//!   release's address space ([`CloseReason::Waiter`]): [`Fences::attach`] closes after
//!   attaching it, so the parked channel is not left waiting on recording. An 8-byte release
//!   covers two words and a 16-byte one four, and each is looked at
//!   (`a_waiter_on_a_wide_releases_high_word_closes_the_open_list`);
//! - [`CLOSE_AFTER`] (1 ms, a **guess**, so that a CPU spinning on a semaphore is not starved)
//!   of recording ([`CloseReason::Elapsed`]): [`Fences::record`] measures from the open list's
//!   first command, on [`FenceClock::elapsed`], and closes once a command is recorded at or past
//!   it. [`Fences::record_list`] records a batch of commands (one draw's, say) and measures once
//!   after the whole batch, so a batch is never split between two lists
//!   (`seams::a_recorded_list_stays_in_one_list_and_closes_once_after_it`). Nothing closes an
//!   open list that stops receiving commands before then, except the other triggers; the ring
//!   empties at the latest when recording stops.
//!
//! [`Fences::newest_submitted`] gives the newest fence a list was submitted with, whichever
//! trigger closed it: the executor tells the caches of it before their next use.
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
//! - Either then reads back the bytes it wrote, through the same memory, and latches, for each
//!   32-bit word read back (one for a 4-byte release, two for an 8-byte one, four for the
//!   16-byte structure), every waiter of the release's address space on that word's address that
//!   the word meets ([`WaiterTable::on_write`]); [`Report::latched`] says whether one was. The
//!   read-back is what a release to the syncpoint shim latches with: the counter's new value,
//!   not the payload (`production::a_release_to_the_shim_latches_with_the_counter_it_reads_back`;
//!   the wiring note on issue #330). When the read-back faults, the waiters are latched with the
//!   bytes written, logged at warning level
//!   (`seams::a_release_whose_read_back_faults_latches_with_the_bytes_written`). A waiter on the
//!   high word of an 8-byte release is latched by that word
//!   (`a_wide_release_latches_a_waiter_on_its_high_word`).
//! - A release whose memory access fails ([`GpuFault`]) writes nothing that the memory refused,
//!   latches nothing, and is logged at warning level and listed in [`Report::failed`]; the
//!   actions after it are still performed.
//! - **An increment** of count `n` calls [`Syncpoints::increment_by`] once: one wrapping
//!   addition of `n` to the counter, as `n` of Eden's increments would leave it (the
//!   `std::atomic<u32>` `fetch_add(1)` of `SyncpointManager::Increment` in
//!   `eden/src/video_core/host1x/syncpoint_manager.cpp` @ cf654052 wraps past `u32::MAX`), and
//!   at most one [`GpuCompletion::SyncpointChanged`], with the final value, when the syncpoint
//!   is watched; a count of 0 changes and pushes nothing. The time it takes does not grow with
//!   the count (`increment::u32_max_increments_take_one_addition`). [`Report::incremented`]
//!   counts the increments. See "Waiters and a run of increments" below.
//! - **A marker** applies its [`OrderedUpdate`], if it has one: [`GpuVm::apply`], then
//!   [`GpuCompletion::GpuOrdered`] with its ticket if it has one; its token goes in
//!   [`Report::markers`] for the front. [`GpuVm::apply`] also frees a destroyed sparse region's
//!   GPU-only storage (slice G22, its module docs' "The GPU-only store"). The update and its
//!   address space are kept for [`Fences::take_applied`], from which the executor (slice G19)
//!   drops the range from that space's buffer and texture caches and calls the shader cache's
//!   `invalidate` (`seams::applied_updates_are_handed_over_in_order_with_their_space`).
//!
//! # The catch-up
//!
//! [`Fences::settle`] makes every fenced release of one address space that writes a byte of a
//! range, and is not yet performed, visible: it closes the open list if that list holds one
//! ([`CloseReason::CatchUp`]), waits for the newest list in flight that holds one
//! ([`starquake_render::Backend::wait`], bounded by its caller's timeout), and runs completion.
//! A wait that times out is [`FenceError::CatchUpTimedOut`], after completion has performed what
//! did complete. [`Fences::catch_up`] is the same wait for everything recorded so far: it closes
//! the open list and waits for the newest list submitted (the write-back before a copy). Pinned
//! in `fence/tests/seams.rs`.
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
//! # Waiters and a run of increments
//!
//! Eden-NX2 performs a submission's increments one at a time
//! (`BuildIncrementCountCommandList` in
//! `eden-nx2/src/core/hle/service/nvdrv/devices/nvhost_gpu.cpp` @ 50fa86bb pushes one
//! `SyncpointOperation` per increment; each reaches `FenceManager::SignalSyncPoint` in
//! `eden-nx2/src/video_core/fence_manager.h`, which calls `SyncpointManager::IncrementGuest`
//! and, after its fence, `IncrementHost`). `SyncpointManager::Increment`
//! (`eden-nx2/src/video_core/host1x/syncpoint_manager.cpp` @ 50fa86bb, the same as mainline's)
//! runs, after each `+ 1`, the registered actions in ascending `expected_value` order and stops
//! at the first whose `expected_value` is above the new value: a waiter between the old and the
//! final value is woken as the counter passes it. Here one action adds the whole count, so every
//! waiter looks at the final value only:
//!
//! - gsv's waiters see it through the one `SyncpointChanged` push, and the front's fence waits
//!   by reading the counter ([`Syncpoints::reached`]) after the report, both with the 32-bit
//!   wrapping comparison ([`crate::syncpoint::wait_satisfied`], mode 0).
//! - A waiter between, at threshold `t` with `old < t <= old + n` (as 32-bit values counted on
//!   from `old`), is met by the final value when `old + n - t`, which is below `n`, is below
//!   2^31 (the comparison's sign bit). So for every count up to 2^31 each waiter the run passes
//!   is met, as in Eden.
//! - **A divergence:** for a count above 2^31, a threshold more than 2^31 - 1 behind the final
//!   value is not met, where Eden would have woken it on the way. Pinned by
//!   `increment::a_run_meets_the_thresholds_it_passes_up_to_half_the_range`, at the edge on both
//!   sides. It needs a guest to ask for more than 2^31 increments in one submission. One entry
//!   per increment, as before the count, was not Eden's walk either: the front read the counter
//!   after each batch of increments the back performed, and gsv's completion queue folds the
//!   queued pushes of one id into the newest.
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
//! - **Every 32-bit word of a wide release latches.** Eden-NX2's query path latches only the
//!   32-bit word at the release's address: it calls `OnWrite(cpu_addr,
//!   static_cast<u32>(...value))` (or `OnWrite(cpu_addr, payload)` with the 32-bit payload) once
//!   per release (`QueryCacheBase::CounterReport` in
//!   `eden-nx2/src/video_core/query_cache/query_cache.h` @ 50fa86bb). Here every 32-bit word
//!   an 8- or 16-byte release writes is latched, the timestamp's halves included, and each is
//!   looked at for the waiter close trigger ([`WaiterTable::has_waiter`]), so a waiter on the
//!   high word of a 64-bit semaphore is not left parked
//!   (`a_wide_release_latches_a_waiter_on_its_high_word`,
//!   `a_waiter_on_a_wide_releases_high_word_closes_the_open_list`).
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
//!   "from guest time". The wiring (slice G21, `crate::wiring`) gives the host's monotonic clock,
//!   nanoseconds since the GPU was made: a **guess**, and not guest time, an open question for
//!   the owner. It is not recorded, so a replay that does not supply the same clock writes
//!   different timestamps.
//! - Releases to the syncpoint shim (gpu-protocol, "Semaphore acquires") go through the
//!   release's [`GpuMemory`] like any other; the production memory
//!   ([`crate::memory::VmMemory`], slice G22), which is the one that translates addresses, turns
//!   one into an increment of that syncpoint, and the read-back latches with the new counter
//!   ("Completion" above).
//! - Each [`FencedAction::Release`] carries the memory it writes and that memory's [`AsId`],
//!   which the executor resolves from the issuing channel the ring's `Release` names (decided by
//!   the owner on 2026-11-18); the waiter table's key is discussed in its own docs.
//! - A syncpoint increment closes nothing: a channel waiting on a submission fence is a front
//!   wait (slice G14, `crate::front`), not a waiter in the table, so the open list it waits on is
//!   closed by the other triggers (the ring empties once the front has nothing to send; at the
//!   latest [`CLOSE_AFTER`] of recording). Decided in slice G14; a measurement of fence-wait
//!   latency would show whether an increment trigger is worth adding.

mod waiters;

#[cfg(test)]
pub(crate) mod tests;

pub use waiters::{Waiter, WaiterTable};

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use starquake_render::{Backend, Command, CommandList, FenceValue, RenderError, WaitOutcome};
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
/// updates and barriers", step 4). The front loop (slice G14) makes them, and keeps them by
/// marker token for the executor in [`crate::front::OrderedTable`].
#[derive(Clone)]
pub struct OrderedUpdate {
    /// The change to an address table; `None` for a barrier, which names none.
    pub change: Option<VmChange>,
    /// The ticket reported as [`GpuCompletion::GpuOrdered`] once applied; `None` when gsv asked
    /// for no report.
    pub ticket: Option<Ticket>,
}

/// One update of one address table.
#[derive(Clone)]
pub struct VmChange {
    /// The address space whose table it is: the executor (slice G19) drops the update's range
    /// from that space's caches once it is applied ([`Fences::take_applied`]).
    pub space: AsId,
    /// The address table the update applies to.
    pub vm: Arc<GpuVm>,
    /// The update.
    pub update: VmUpdate,
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
    /// `count` increments of a syncpoint, added at once (module docs, "Completion").
    Increment {
        /// The syncpoint.
        id: SyncpointId,
        /// How many increments; 0 changes nothing.
        count: u32,
    },
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
    /// The executor must see the host GPU finish the open list before it goes on: a write-back
    /// whose readbacks the list holds, or a conditional-render catch-up whose release the list
    /// holds ([`Fences::settle`], [`Fences::catch_up`]).
    CatchUp,
    /// A channel with a 3D engine was removed: the executor closes the list its draw path may
    /// have recorded into, so the host objects the draw path destroys outlive it.
    ChannelRemoved,
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
    /// Syncpoint increments made, one per increment (an action of count `n` adds `n`).
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
    /// The render backend's wait for a catch-up failed ([`Fences::settle`],
    /// [`Fences::catch_up`]).
    #[error("the render backend's wait for a catch-up failed: {0}")]
    Wait(RenderError),
    /// The host GPU did not reach the fence a catch-up waited for in time ([`Fences::settle`],
    /// [`Fences::catch_up`]).
    #[error("the host GPU did not finish command list {fence} in time for a catch-up")]
    CatchUpTimedOut {
        /// The fence value waited for.
        fence: u64,
    },
}

/// What [`Fences::settle`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Settled {
    /// No release of the range was pending: nothing was closed, waited for or performed.
    Nothing,
    /// The list holding the newest such release completed, and completion processing performed
    /// this.
    Performed(Report),
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
    /// [`CloseReason::CatchUp`].
    pub catch_up: u64,
    /// [`CloseReason::ChannelRemoved`].
    pub channel_removed: u64,
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
            CloseReason::CatchUp => &mut self.catch_up,
            CloseReason::ChannelRemoved => &mut self.channel_removed,
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
    /// The newest fence any list was submitted with.
    newest: FenceValue,
    /// Updates applied since [`Fences::take_applied`] last looked, in the order applied.
    applied: Vec<(AsId, VmUpdate)>,
}

/// Whether `release` writes a byte of `range`.
fn overlaps(release: &Release, range: &Range<u64>) -> bool {
    let end = release.address.saturating_add(release.form.bytes());
    release.address < range.end && range.start < end
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

/// The words a release's waiters are latched with: what `memory` holds at `address` after the
/// write, read back (module docs, "Completion"), or, when that read faults, the bytes written,
/// logged at warning level.
fn read_back(memory: &dyn GpuMemory, address: u64, written: Written) -> Written {
    let mut bytes = [0; LARGEST_RELEASE_BYTES];
    let Some(out) = bytes.get_mut(..written.len) else {
        return written;
    };
    match memory.read(address, out) {
        Ok(()) => Written {
            bytes,
            len: written.len,
        },
        Err(fault) => {
            tracing::warn!(
                address,
                %fault,
                "a release could not be read back; its waiters are latched with the bytes written"
            );
            written
        }
    }
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
            newest: FenceValue::default(),
            applied: Vec::new(),
        }
    }

    /// The newest fence value any list was submitted with; [`FenceValue::default`] before the
    /// first. The executor (slice G19) tells the caches and draw paths of it
    /// ([`crate::back::buffer::BufferCache::submitted`]) before their next use, since lists are
    /// closed here for reasons of the fence path's own (a waiter, the elapsed time, an empty
    /// ring).
    pub fn newest_submitted(&self) -> FenceValue {
        self.newest
    }

    /// The ordered updates applied since the last call, with their address spaces, in the order
    /// they were applied, and forgets them: the executor drops their ranges from that space's
    /// caches and the shader cache (gpu-protocol, "Ordered address-table updates and barriers",
    /// step 4).
    pub fn take_applied(&mut self) -> Vec<(AsId, VmUpdate)> {
        std::mem::take(&mut self.applied)
    }

    /// Records every command of `commands` on the open list, in order, emptying it, and then
    /// closes the list if it has recorded for [`CLOSE_AFTER`]: the commands of one draw,
    /// dispatch or copy stay in one list (a draw's `SetRenderTargets` and its `Draw`, say).
    /// Nothing is recorded, and nothing closed, when `commands` is empty.
    ///
    /// # Errors
    ///
    /// [`FenceError::Submit`] as for [`Fences::record`].
    pub fn record_list(&mut self, commands: &mut CommandList) -> Result<(), FenceError> {
        if commands.commands.is_empty() {
            return Ok(());
        }
        self.record_with(|open| open.commands.append(&mut commands.commands))
    }

    /// Adds commands to the open list with `add`, then closes it if it has recorded for
    /// [`CLOSE_AFTER`]: the open time and the close rule of [`Fences::record`] and
    /// [`Fences::record_list`], in one place.
    fn record_with(&mut self, add: impl FnOnce(&mut CommandList)) -> Result<(), FenceError> {
        let now = self.env.clock.elapsed();
        if self.open.commands.is_empty() {
            self.opened_at = now;
        }
        add(&mut self.open);
        if now.saturating_sub(self.opened_at) >= CLOSE_AFTER {
            self.close(CloseReason::Elapsed)?;
        }
        Ok(())
    }

    /// Makes every fenced release of address space `space` that writes a byte of `range` and is
    /// not yet performed visible, waiting at most `timeout` for the host GPU (the conditional
    /// render catch-up of gpu-protocol, "3D: from state to a draw": "after a catch-up when that
    /// value is still pending"). When the open list holds such a release it is closed
    /// ([`CloseReason::CatchUp`]); then, if a list in flight holds one, the newest such list's
    /// fence is waited for ([`Backend::wait`]), and completion runs ([`Fences::complete`]).
    /// Returns [`Settled::Nothing`] when no such release was pending, and nothing is closed,
    /// waited for or performed.
    ///
    /// # Errors
    ///
    /// [`FenceError::Submit`] when closing failed; [`FenceError::Wait`] when the backend's wait
    /// failed; [`FenceError::CatchUpTimedOut`] when `timeout` passed first. Completion runs in
    /// every case once anything was waited for, so what did complete is performed.
    pub fn settle(
        &mut self,
        space: AsId,
        range: Range<u64>,
        timeout: Duration,
    ) -> Result<Settled, FenceError> {
        let writes = |action: &FencedAction| match action {
            FencedAction::Release {
                space: s, release, ..
            } => *s == space && overlaps(release, &range),
            FencedAction::Increment { .. } | FencedAction::Marker { .. } => false,
        };
        if self.open_actions.iter().any(writes) {
            self.close(CloseReason::CatchUp)?;
        }
        let Some(fence) = self
            .in_flight
            .iter()
            .rev()
            .find(|list| list.actions.iter().any(writes))
            .map(|list| list.fence)
        else {
            return Ok(Settled::Nothing);
        };
        self.wait_and_complete(fence, timeout)
            .map(Settled::Performed)
    }

    /// Closes the open list ([`CloseReason::CatchUp`]) and waits at most `timeout` for the
    /// newest list submitted, which then holds everything recorded so far (the list closed here,
    /// or one [`Fences::record_list`] closed for the elapsed time, when this had nothing to
    /// submit), then runs completion: the write-back before a copy, whose readbacks the executor
    /// has just recorded (its module docs, "Copies of images the texture cache holds").
    ///
    /// # Errors
    ///
    /// As [`Fences::settle`]: [`FenceError::Submit`], [`FenceError::Wait`] or
    /// [`FenceError::CatchUpTimedOut`]. Completion runs once the wait has returned, whatever it
    /// returned.
    pub fn catch_up(&mut self, timeout: Duration) -> Result<Report, FenceError> {
        self.close(CloseReason::CatchUp)?;
        self.wait_and_complete(self.newest, timeout)
    }

    /// Waits at most `timeout` for `fence`, then runs completion whatever the wait returned, so
    /// what did complete is performed: the wait of [`Fences::settle`] and
    /// [`Fences::catch_up`].
    fn wait_and_complete(
        &mut self,
        fence: FenceValue,
        timeout: Duration,
    ) -> Result<Report, FenceError> {
        let waited = self.backend.wait(fence, timeout);
        let report = self.complete();
        match waited {
            Ok(WaitOutcome::Reached) => Ok(report),
            Ok(WaitOutcome::TimedOut) => Err(FenceError::CatchUpTimedOut { fence: fence.get() }),
            Err(error) => Err(FenceError::Wait(error)),
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
        self.record_with(|open| open.push(command))
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
                FencedAction::Increment { .. } | FencedAction::Marker { .. } => false,
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
                self.newest = self.newest.max(fence);
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
            } => match write_release(&*memory, &release, &*self.env.clock) {
                Ok(written) => {
                    report.released = report.released.saturating_add(1);
                    let written = read_back(&*memory, release.address, written);
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
                    self.counters.failed_releases = self.counters.failed_releases.saturating_add(1);
                    report.failed.push(failure);
                }
            },
            FencedAction::Increment { id, count } => {
                self.env.syncpoints.increment_by(id, count);
                report.incremented = report.incremented.saturating_add(u64::from(count));
            }
            FencedAction::Marker { token, ordered } => {
                if let Some(ordered) = ordered {
                    if let Some(change) = ordered.change {
                        change.vm.apply(&change.update);
                        self.applied.push((change.space, change.update));
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
