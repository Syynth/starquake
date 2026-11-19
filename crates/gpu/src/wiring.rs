//! The GPU front gsv calls, its two threads and their wiring (gpu-protocol spec, slice G21:
//! "Threads and data flow", "Graphics server sessions (#11)", "Observability").
//!
//! No reference is ported here: the wiring is new code over the slices it joins (G14's front
//! loop, G19's executor, G15's fence path, G22's production memory).
//!
//! # What is built
//!
//! [`Gpu::new`] takes [`GpuDeps`] (the ring, made first so that the render backend can be built
//! with its completion wake, [`GpuRing::completion_wake`]; the `Syncpoints`; gsv's
//! [`Completions`]; the render backend; the shader cache behind G18's seam; the title's guest
//! memory and write watch; the two clocks) and a [`GpuConfig`], and starts:
//!
//! - **the front thread** (`gpu-front`): [`Front`] over the ring's producer, run by
//!   [`Front::run_observed`], which hands this module its counters after each turn;
//! - **the back thread** (`gpu-back`): the [`Executor`] over the ring's consumer, built on that
//!   thread (`back_thread`).
//!
//! [`Gpu`] implements [`GpuFront`], whose every method only queues and never blocks on the GPU
//! (spec, "Thread rules"):
//!
//! | Method | What it does |
//! |---|---|
//! | `create_address_space` | makes the space's [`VmMemory`] for the front and, with [`VmMemory::for_thread`], one for the back (each GPU thread keeps its own translation cache, row 125), posts the back's to the back thread, and queues the front's. A space made twice keeps the first, as the front does |
//! | `create_channel` | queues it on the front ([`FrontShared::create_channel`]) and posts it to the back |
//! | `bind_class` | `Ok` for a channel made and not destroyed, [`GpuError`] otherwise; it changes nothing, since the stream's bind-object method binds a class on the GPU |
//! | `submit` | [`FrontShared::submit`]: `Queued`, or `Full` with `GpuQueueRoom` to follow |
//! | `update_vm`, `barrier` | [`FrontShared::update_vm`], [`FrontShared::barrier`] |
//! | `present` | [`FrontShared::present`]: pushed once every submission queued before it is read |
//! | `channel_error` | [`FrontShared::channel_error`] |
//! | `destroy_channel` | [`FrontShared::destroy_channel`]; the back forgets the channel after its last entry (`back_thread`) |
//!
//! A refusal ([`GpuError`]) is logged at warning level with the front's reason
//! ([`crate::front::CreateError`]).
//!
//! # Completions
//!
//! Pushed to gsv's [`Completions`] by the front thread (`GpuQueueRoom`) and the back thread
//! (`GpuOrdered`, after the marker's update is applied; `SyncpointChanged`, by `Syncpoints` when
//! the back increments a watched counter). `GpuQueueRoom` and `GpuOrdered` are pushed through
//! the queue this module gives the front and the fence path, which records each push in the
//! flight recorder first (`recorder`). `SyncpointChanged` is not: `Syncpoints` is built by the
//! caller over gsv's own queue ([`GpuDeps::syncpoints`]) and pushes to it directly, so the
//! recorder never sees it.
//!
//! # Presents
//!
//! A present names no address space, so the back reads every present through one: the first
//! address space made. A **guess**: one GPU address space per title, which presents read (the
//! executor's module docs, "Presents"; issue #330's scope note). A title that presents from a
//! second space would show nothing from it (the present is refused and counted). The present
//! space stays the first made for the whole run, even once gsv has destroyed it (next section):
//! presents then read through a table gsv no longer changes. **Owner question**: whether a
//! present should follow a later space when the first is destroyed.
//!
//! # Address-space lifetime
//!
//! gsv destroys address spaces (closing `nvhost-as-gpu`) and never reuses their ids, but
//! [`GpuFront`] (the spec's interface) has no call that tells the GPU, so nothing here forgets a
//! space: [`Gpu`]'s set of spaces, the front's space table and its [`VmMemory`], and the back
//! executor's spaces with their translation caches grow by one space for each space a title
//! makes, for the rest of the run. A title that opens and closes `nvhost-as-gpu` in a loop grows
//! them without bound. **Owner question**: a destroy hook on `GpuVmFront` (a spec change), or a
//! cap on the spaces a title may make.
//!
//! # The channel on `Release`
//!
//! Each ring `Release` names the channel that issued it, and the executor writes it through that
//! channel's address space's memory, the back thread's own [`VmMemory`] (decided by the owner on
//! 2026-11-18, waves 37 and 38 errata item 10; the ring and executor already carry it, slice
//! G19). `a_release_is_written_through_the_back_threads_memory_of_its_channels_space` runs it
//! through the threads.
//!
//! # Observability
//!
//! [`Gpu::counters`] gives [`GpuCounters`] for `summary.json`: the exact set it flattens is in
//! its docs (the front's, the executor's and the fence path's counters, two of the back
//! thread's, and two of this module's). [`Gpu::flight_recorder`] gives `gpu.txt`'s text
//! (`recorder`).
//!
//! # Idleness
//!
//! [`Gpu::is_idle`] says whether nothing the GPU holds can still change anything a guest waits
//! for. It is built from looks ([`Gpu::look`]); one look finds the GPU idle when:
//!
//! - the front is quiet ([`FrontShared::quiet_turn`]: nothing queued for it, not stopping, and
//!   it has ended a turn idle since it last started one), in a first and a last read that name
//!   the same number of turns, so no turn started while it looked;
//! - the back has taken every entry the front pushed by the end of its last turn;
//! - no command list is in flight or open on the back;
//! - no back report reached the front while it looked.
//!
//! Every turn makes the front busy when it starts ([`FrontShared::take`]), whatever it took: a
//! turn the acquire poll starts takes nothing, and still resumes channels and pushes entries
//! (`a_turn_the_poll_starts_is_busy`). The front publishes its pushed count, its polls and
//! whether the poll is scheduled at the end of each turn, before an idle turn marks it quiet;
//! the back stores its lists in flight before its taken count; and the back's report of a
//! completion to the front bumps a counter before and after it reaches the front, which a look
//! reads first and last.
//!
//! **A channel waiting on the poll.** A channel parked on an acquire, or on a fence, waits on
//! something a CPU write (or an increment from outside the GPU) can end, and only the next poll
//! finds it. So when a look finds the GPU idle but the poll scheduled, [`Gpu::is_idle`] waits
//! for the front to finish a poll that started after the look and then to end a turn idle, at
//! most [`IDLE_POLL_WAIT`] for both, and looks again: idle only if that second look is idle too.
//! A poll counted past the first look's started after it, since no turn was in progress while
//! the first look read the front quiet. A poll that finds the wait ended
//! resumes the channel, and its turn ends busy, so the second look says "busy"
//! (`a_cpu_release_of_a_parked_acquire_is_busy_through_the_threads`; with no poll at all the
//! answer is "busy" too, `a_released_acquire_is_busy_until_a_poll_has_looked`); one that finds
//! nothing
//! leaves the GPU idle (`a_channel_parked_on_an_unreleased_acquire_is_idle`). A channel that
//! still waits after that poll waits on something only the guest can end.
//!
//! **Owner question**, the trade-off of that wait: `core` asks as the last vCPU parks, with its
//! cores' idle lock held, so the wait holds that lock for up to [`IDLE_POLL_WAIT`]; and a front
//! thread the host does not run within it makes the answer "busy", so `core` does not end a run
//! it would have called stalled on that question (it asks again at the next quiet signal, below,
//! a bounded number of times). The other choice is to count a scheduled poll as busy, which
//! never calls a run with a channel parked on an acquire stalled.
//!
//! **The quiet signal** ([`Gpu::set_quiet_wake`], [`QuietWake`]). The front thread tells the
//! one wake it was given each time it ends a turn idle: after it published the turn's counters
//! and marked itself quiet, with no lock of this crate held. That includes the turn each
//! [`crate::front::IDLE_PARK`] safety-net wake starts, which reads busy while it runs, so a
//! question that landed in it can be asked again once it has ended (gpu-protocol, "Idleness":
//! "the parked core asks again when the GPU next reports quiet"; `core`'s `machine/cores.rs`).
//! A busy answer that comes from the back alone (an entry not taken, a list in flight) is
//! signalled by the front's next idle turn: a report of what the back performed
//! ([`FrontWake`]) wakes the front, and with no report the next safety-net turn comes at most
//! `IDLE_PARK` later.
//! (`the_production_front_thread_tells_its_one_quiet_wake`;
//! `the_quiet_wake_is_told_after_the_front_is_quiet_with_no_lock_held` pins the order.)
//!
//! It can answer "busy" for a GPU that is idle (a look that overlaps a completion or a turn, or
//! a poll the host is slow to run); it does not answer "idle" while an entry is unapplied or a
//! list is in flight or open (`the_gpu_is_busy_while_a_list_is_in_flight`,
//! `an_entry_that_records_commands_is_in_flight_before_its_list_closes`). After [`Gpu::stop`]
//! it answers "busy" for good: a stopping front is never quiet
//! (`both_threads_end_when_the_gpu_stops`).
//!
//! # Shutdown
//!
//! [`Gpu::stop`] (and dropping the `Gpu`) stops the front thread at its next look: requests it
//! has not yet read are dropped, and it publishes what it pushed and drops the ring's producer;
//! that wakes the back thread, which takes and applies every
//! entry left on the ring and ends at [`crate::ring::PopError::Disconnected`]. The open list
//! and lists still in flight are not waited for: a fenced action attached to one is not
//! performed (one with no work before it is performed when it is taken). If the front thread
//! failed, its ring may be corrupt, so the back is told to stop at its next wake instead. Both
//! threads are joined (`both_threads_end_when_the_gpu_stops`, which checks a release submitted
//! just before the stop is written; `published_work_is_applied_after_the_producer_is_gone`).
//!
//! # Not here
//!
//! - The real shader cache: shader-pipeline slice 19 builds it, and an adapter from it to G18's
//!   seam ([`ShaderCache`]) is the caller's to give, in [`GpuDeps::shaders`].
//! - Choosing the render backend, and writing `gpu.txt` and the counters into a run's
//!   artifacts: `core` does both. It builds a `Gpu` over the recording backend for a live run
//!   (issue #386); render-backend slice R12 adds the choice of the wgpu backend.
//! - Frame dumps: `core` asks for them with the executor's setting in [`GpuConfig::exec`]
//!   ([`crate::back::exec::FrameDumps`]), and installs the frame sink on the backend it builds.

mod back_thread;
mod recorder;

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::Serialize;
use starquake_memory::{GuestMemory, WriteWatch};
use starquake_render::Backend;
use starquake_render::wake::CompletionWake;
use thiserror::Error;

use crate::back::exec::{ExecConfig, Executor, ShaderCache};
use crate::back::fence::{FenceClock, FenceEnv, Fences, FrontWake, Mode, Report};
use crate::channel::ChannelError;
use crate::front::Front;
use crate::front::{
    FrontClock, FrontCounters, FrontDeps, FrontError, FrontShared, FrontThread, HostClock,
    OrderedTable, POLL_LONGEST_SLEEP, spawn_observed,
};
use crate::front::{QuietWake, QuietWakeTaken};
use crate::interface::{
    AsId, ChannelConfig, ChannelFront, ChannelId, Completions, GpuCompletion, GpuError, GpuFront,
    GpuVmFront, PresentFront, PresentRequest, Submission, Submitted, Ticket, VmUpdate,
};
use crate::memory::{GpuMemory, VmMemory};
use crate::ring::{CapacityError, Consumer, Producer, RING_WORDS, ring};
use crate::syncpoint::Syncpoints;
use crate::vm::GpuVm;

use back_thread::{Back, BackInbox, BackShared, Control};

pub use back_thread::{BACK_WAIT, BackError, BackStatus};
pub use recorder::{FAILURES_KEPT, FailedReleases, FlightRecorder, GpuEvent};

/// The events the flight recorder keeps by default: the spec's "last few thousand".
pub const RECORDER_EVENTS: usize = 4096;

/// The ring between the two threads, made before the [`Gpu`] so that the render backend can be
/// built with its completion wake.
pub struct GpuRing {
    producer: Producer,
    consumer: Consumer,
}

impl GpuRing {
    /// A ring of `words` words.
    ///
    /// # Errors
    ///
    /// [`CapacityError`] when `words` is not a power of two.
    pub fn new(words: usize) -> Result<Self, CapacityError> {
        let (producer, consumer) = ring(words)?;
        Ok(Self { producer, consumer })
    }

    /// A ring of the spec's [`RING_WORDS`].
    ///
    /// # Errors
    ///
    /// None in practice: [`RING_WORDS`] is a power of two.
    pub fn standard() -> Result<Self, CapacityError> {
        Self::new(RING_WORDS)
    }

    /// The wake the render backend rings when a list completes (render-backend, "The completion
    /// wake"): it wakes the back thread's wait on this ring.
    pub fn completion_wake(&self) -> CompletionWake {
        self.consumer.waker().completion_wake()
    }
}

/// What the [`Gpu`] is made with (module docs, "What is built").
pub struct GpuDeps {
    /// The ring between the threads.
    pub ring: GpuRing,
    /// The syncpoint counters (gsv's too).
    pub syncpoints: Arc<Syncpoints>,
    /// gsv's completion queue.
    pub completions: Arc<dyn Completions>,
    /// The render backend, built with [`GpuRing::completion_wake`].
    pub backend: Arc<dyn Backend>,
    /// The shader cache behind G18's seam (module docs, "Not here").
    pub shaders: Box<dyn ShaderCache + Send>,
    /// The title's memory, behind every `Guest` target.
    pub guest: Arc<dyn GuestMemory>,
    /// The title's write watch, shared with the kernel.
    pub watch: Arc<WriteWatch>,
    /// The front thread's clock.
    pub front_clock: Arc<dyn FrontClock>,
    /// The back thread's clock.
    pub fence_clock: Arc<dyn FenceClock>,
}

/// The [`Gpu`]'s settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuConfig {
    /// The executor's settings, among them which presents are dumped
    /// ([`ExecConfig::dumps`]): the setting through which `core` asks for frame dumps.
    pub exec: ExecConfig,
    /// When fenced actions are performed ([`Mode::Fenced`] but for debugging).
    pub mode: Mode,
    /// How many events the flight recorder keeps.
    pub recorder_events: usize,
}

impl Default for GpuConfig {
    /// [`ExecConfig::default`], [`Mode::Fenced`] and [`RECORDER_EVENTS`].
    fn default() -> Self {
        Self {
            exec: ExecConfig::default(),
            mode: Mode::Fenced,
            recorder_events: RECORDER_EVENTS,
        }
    }
}

/// The host clock as the back thread's: [`FenceClock::elapsed`] is the time since the clock was
/// made, and the timestamp a 16-byte release writes is the same time in nanoseconds (a
/// **guess** at the timestamp's unit, as the fence path's module docs list).
impl FenceClock for HostClock {
    fn elapsed(&self) -> Duration {
        FrontClock::now(self)
    }

    fn gpu_timestamp(&self) -> u64 {
        u64::try_from(FrontClock::now(self).as_nanos()).unwrap_or(u64::MAX)
    }
}

/// Why the [`Gpu`] could not start, or ended badly.
#[derive(Debug, Error)]
pub enum WiringError {
    /// A thread could not be made.
    #[error("a GPU thread could not be started: {0}")]
    Spawn(#[from] io::Error),
    /// The front thread failed.
    #[error(transparent)]
    Front(#[from] FrontError),
    /// The back thread failed.
    #[error(transparent)]
    Back(#[from] BackError),
}

/// The GPU's counters (gpu-protocol, "Observability"), flattened for `summary.json`: every field
/// of [`FrontCounters`], [`ExecCounters`] and [`FenceCounters`] (the close reasons of
/// [`FenceCounters::closed`] summed into `lists_submitted`), [`BackStatus::wait_failures`] and
/// [`BackStatus::refused_controls`], and the wiring's own two counts of submissions answered
/// `Full` and dropped. [`GpuCounters::new`] names every field of those four structs, so a field
/// added to one does not compile until it is placed here or named as left out; the only ones left
/// out are [`BackStatus`]'s `channels`, `spaces`, `taken` and `in_flight`, which say what the
/// back holds now and are not counts of events. The spec lists, besides what is here: writes
/// dropped as same-value, macro calls, memo hits and the ring's peak occupancy, which no slice
/// counts yet.
///
/// [`ExecCounters`]: crate::back::exec::ExecCounters
/// [`FenceCounters`]: crate::back::fence::FenceCounters
/// [`FenceCounters::closed`]: crate::back::fence::FenceCounters::closed
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GpuCounters {
    /// Command words executed by the front.
    pub words: u64,
    /// Submissions finished by the front.
    pub submissions: u64,
    /// Submissions answered `Full`.
    pub submissions_full: u64,
    /// Submissions dropped because their channel did not exist.
    pub submissions_dropped: u64,
    /// Syncpoint increments the front pushed for submissions.
    pub increments: u64,
    /// Acquires that parked a channel.
    pub acquires_parked: u64,
    /// Submission fences that made a channel wait.
    pub fence_waits: u64,
    /// `SYNC_WAIT` entries that made a channel wait.
    pub sync_waits: u64,
    /// Macro calls that waited for a catch-up.
    pub macro_catch_ups: u64,
    /// Times the acquire poll fired.
    pub polls: u64,
    /// Ordered updates and barriers emitted.
    pub ordered: u64,
    /// Macro calls that failed and committed nothing.
    pub macro_failures: u64,
    /// Channels stopped by a structural error.
    pub channels_stopped: u64,
    /// Presents pushed by the front.
    pub presents_pushed: u64,
    /// Ring entries the back applied.
    pub entries_applied: u64,
    /// Ring entries (or parts) the back refused.
    pub entries_refused: u64,
    /// Words to classes with no engine.
    pub unhandled_words: u64,
    /// `CaptureControl` entries the back took.
    pub captures: u64,
    /// Presents recorded.
    pub presents: u64,
    /// Dispatches recorded.
    pub dispatches: u64,
    /// Dispatches not recorded.
    pub dispatches_skipped: u64,
    /// Write-backs before a copy.
    pub write_backs: u64,
    /// Write-backs that failed.
    pub write_back_failures: u64,
    /// Conditional-render catch-ups.
    pub render_catch_ups: u64,
    /// Conditional-render catch-ups that failed.
    pub render_catch_up_failures: u64,
    /// Fenced actions attached.
    pub fenced_actions: u64,
    /// Fenced actions performed at once, with no work before them.
    pub fenced_actions_skipped_empty: u64,
    /// Lists submitted to the render backend.
    pub lists_submitted: u64,
    /// Safety-net rechecks that found a lost completion wake.
    pub lost_wakes: u64,
    /// Releases that failed.
    pub failed_releases: u64,
    /// Waits of the back thread that failed other than by the ring.
    pub back_wait_failures: u64,
    /// Address-space and channel changes the back refused.
    pub refused_controls: u64,
}

impl GpuCounters {
    fn new(front: &FrontCounters, back: &BackStatus, full: u64, dropped: u64) -> Self {
        // Every field named, no `..`: a new counter is a compile error here (type docs).
        let FrontCounters {
            words,
            submissions,
            increments,
            parks,
            fence_waits,
            sync_waits,
            catch_ups: macro_catch_ups,
            polls,
            ordered,
            macro_failures,
            stopped_channels,
            presents: presents_pushed,
        } = *front;
        let BackStatus {
            exec,
            fences,
            channels: _,
            spaces: _,
            taken: _,
            in_flight: _,
            wait_failures,
            refused_controls,
        } = *back;
        let crate::back::exec::ExecCounters {
            applied,
            failed,
            unhandled_words,
            captures,
            presents,
            dispatches,
            dispatches_skipped,
            write_backs,
            write_back_failures,
            catch_ups: render_catch_ups,
            catch_up_failures: render_catch_up_failures,
        } = exec;
        let crate::back::fence::FenceCounters {
            fenced,
            skipped_empty,
            closed,
            lost_wakes,
            failed_releases,
        } = fences;
        let crate::back::fence::ClosedCounts {
            ring_empty,
            present,
            wait_for_idle,
            waiter,
            elapsed,
            set_reference,
            catch_up,
            channel_removed,
        } = closed;
        let lists = [
            ring_empty,
            present,
            wait_for_idle,
            waiter,
            elapsed,
            set_reference,
            catch_up,
            channel_removed,
        ]
        .iter()
        .fold(0u64, |sum, &count| sum.saturating_add(count));
        Self {
            words,
            submissions,
            submissions_full: full,
            submissions_dropped: dropped,
            increments,
            acquires_parked: parks,
            fence_waits,
            sync_waits,
            macro_catch_ups,
            polls,
            ordered,
            macro_failures,
            channels_stopped: stopped_channels,
            presents_pushed,
            entries_applied: applied,
            entries_refused: failed,
            unhandled_words,
            captures,
            presents,
            dispatches,
            dispatches_skipped,
            write_backs,
            write_back_failures,
            render_catch_ups,
            render_catch_up_failures,
            fenced_actions: fenced,
            fenced_actions_skipped_empty: skipped_empty,
            lists_submitted: lists,
            lost_wakes,
            failed_releases,
            back_wait_failures: wait_failures,
            refused_controls,
        }
    }
}

/// What the front thread publishes at the end of each turn, before an idle turn marks the front
/// quiet (module docs, "Idleness").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Published {
    /// The front's counters; `polls` counts the polls of finished turns only.
    counters: FrontCounters,
    /// [`Front::sent`].
    sent: u64,
    /// [`Front::waits_on_poll`].
    waiting: bool,
}

/// What the front thread publishes after each turn, and the idleness counter (module docs,
/// "Idleness").
#[derive(Default)]
struct Status {
    front: Mutex<Published>,
    /// Rung after each turn's publication, for [`Status::wait_for_poll`].
    turned: Condvar,
    /// Bumped before and after each back report reaches the front.
    activity: AtomicU64,
    /// Submissions answered `Full`.
    full: AtomicU64,
}

impl Status {
    fn lock(&self) -> MutexGuard<'_, Published> {
        // A plain value, written whole: a poisoned lock is usable.
        self.front.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn published(&self) -> Published {
        *self.lock()
    }

    /// Stores what the front thread publishes at the end of a turn, and wakes
    /// [`Status::wait_for_poll`].
    fn publish(&self, published: Published) {
        *self.lock() = published;
        self.turned.notify_all();
    }

    /// Waits at most `timeout` until the front has finished a turn that polled, past the
    /// `polls` it had published; whether one did.
    fn wait_for_poll(&self, polls: u64, timeout: Duration) -> bool {
        let guard = self.lock();
        let (guard, _) = self
            .turned
            .wait_timeout_while(guard, timeout, |published| {
                published.counters.polls <= polls
            })
            .unwrap_or_else(PoisonError::into_inner);
        guard.counters.polls > polls
    }

    fn bump(&self) {
        self.activity.fetch_add(1, Ordering::SeqCst);
    }
}

/// The back's report to the front, recorded, and counted for idleness (module docs).
struct ReportToFront {
    front: Arc<dyn FrontWake>,
    status: Arc<Status>,
    recorder: Arc<FlightRecorder>,
}

impl FrontWake for ReportToFront {
    fn wake(&self, report: &Report) {
        self.status.bump();
        self.recorder.record(GpuEvent::Performed {
            released: report.released,
            incremented: report.incremented,
            latched: report.latched,
            markers: report.markers.len(),
            failed: FailedReleases::new(&report.failed),
        });
        self.front.wake(report);
        self.status.bump();
    }
}

/// gsv's completion queue, with each push recorded first.
struct RecordedCompletions {
    inner: Arc<dyn Completions>,
    recorder: Arc<FlightRecorder>,
}

impl Completions for RecordedCompletions {
    fn push(&self, completion: GpuCompletion) {
        self.recorder.record(GpuEvent::Completion(completion));
        self.inner.push(completion);
    }
}

/// The running threads.
struct Threads {
    front: FrontThread,
    back: JoinHandle<Result<(), BackError>>,
}

/// The GPU front gsv calls, and its threads (module docs).
pub struct Gpu {
    shared: Arc<FrontShared>,
    inbox: Arc<BackInbox>,
    back: Arc<BackShared>,
    status: Arc<Status>,
    recorder: Arc<FlightRecorder>,
    syncpoints: Arc<Syncpoints>,
    guest: Arc<dyn GuestMemory>,
    watch: Arc<WriteWatch>,
    spaces: Mutex<BTreeSet<AsId>>,
    threads: Mutex<Option<Threads>>,
}

impl Gpu {
    /// Starts the front and back threads over `deps` (module docs, "What is built").
    ///
    /// # Errors
    ///
    /// [`WiringError::Spawn`] when a thread cannot be made; a front thread already started is
    /// stopped first.
    pub fn new(deps: GpuDeps, config: GpuConfig) -> Result<Self, WiringError> {
        let GpuDeps {
            ring,
            syncpoints,
            completions,
            backend,
            shaders,
            guest,
            watch,
            front_clock,
            fence_clock,
        } = deps;
        let GpuRing { producer, consumer } = ring;
        let recorder = Arc::new(FlightRecorder::new(config.recorder_events));
        let status = Arc::new(Status::default());
        let shared = Arc::new(FrontShared::new());
        let waiters = Arc::new(crate::back::fence::WaiterTable::new());
        let ordered = Arc::new(OrderedTable::new());
        let completions: Arc<dyn Completions> = Arc::new(RecordedCompletions {
            inner: completions,
            recorder: Arc::clone(&recorder),
        });
        let inbox = Arc::new(BackInbox::new(consumer.waker()));
        let back = Arc::new(BackShared::default());

        let front = Front::new(FrontDeps {
            shared: Arc::clone(&shared),
            ring: producer,
            syncpoints: Arc::clone(&syncpoints),
            completions: Arc::clone(&completions),
            waiters: Arc::clone(&waiters),
            ordered: Arc::clone(&ordered),
            clock: front_clock,
        })
        .with_retired(Arc::<BackInbox>::clone(&inbox));
        let observed = Arc::clone(&status);
        let front = spawn_observed(front, move |front| {
            observed.publish(Published {
                counters: front.counters(),
                sent: front.sent(),
                waiting: front.waits_on_poll(),
            });
        })?;

        let env = FenceEnv {
            syncpoints: Arc::clone(&syncpoints),
            waiters,
            completions,
            clock: fence_clock,
            front: Arc::new(ReportToFront {
                front: Arc::<FrontShared>::clone(&shared),
                status: Arc::clone(&status),
                recorder: Arc::clone(&recorder),
            }),
        };
        let back_inbox = Arc::clone(&inbox);
        let back_shared = Arc::clone(&back);
        let spawned = thread::Builder::new()
            .name("gpu-back".to_owned())
            .spawn(move || {
                let fences = Fences::new(Arc::clone(&backend), env, config.mode);
                let exec = Executor::new(backend, fences, shaders, ordered, config.exec);
                Back {
                    exec,
                    ring: consumer,
                    inbox: back_inbox,
                    shared: back_shared,
                    channels: BTreeSet::new(),
                    spaces: BTreeSet::new(),
                    taken: 0,
                    wait_failures: 0,
                    refused_controls: 0,
                }
                .run()
            });
        let back_handle = match spawned {
            Ok(handle) => handle,
            Err(error) => {
                if let Err(stopped) = front.stop() {
                    tracing::error!(%stopped, "the GPU front thread failed while it was stopped");
                }
                return Err(WiringError::Spawn(error));
            }
        };
        Ok(Self {
            shared,
            inbox,
            back,
            status,
            recorder,
            syncpoints,
            guest,
            watch,
            spaces: Mutex::new(BTreeSet::new()),
            threads: Mutex::new(Some(Threads {
                front,
                back: back_handle,
            })),
        })
    }

    fn threads(&self) -> MutexGuard<'_, Option<Threads>> {
        self.threads.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Stops both threads and waits for them (module docs, "Shutdown"). A second call does
    /// nothing.
    ///
    /// # Errors
    ///
    /// The front thread's [`FrontError`] if it failed, else the back thread's [`BackError`].
    pub fn stop(&self) -> Result<(), WiringError> {
        let Some(threads) = self.threads().take() else {
            return Ok(());
        };
        let front = threads.front.stop();
        // The front thread published what it pushed and dropped the ring's producer, which wakes
        // the back: it takes every entry left and ends at `PopError::Disconnected` (module docs,
        // "Shutdown"). Only a failed front is not waited for: its ring may be corrupt.
        if stops_the_back(&front) {
            self.inbox.stop();
        }
        let back = threads.back.join().unwrap_or(Err(BackError::Panicked));
        front?;
        back?;
        Ok(())
    }

    /// Whether both threads are still running.
    pub fn is_running(&self) -> bool {
        self.threads()
            .as_ref()
            .is_some_and(|threads| !threads.front.is_finished() && !threads.back.is_finished())
    }

    /// Every counter (module docs, "Observability").
    pub fn counters(&self) -> GpuCounters {
        GpuCounters::new(
            &self.status.published().counters,
            &self.back.status(),
            self.status.full.load(Ordering::Relaxed),
            self.shared.dropped_submissions(),
        )
    }

    /// What the back thread published last.
    pub fn back_status(&self) -> BackStatus {
        self.back.status()
    }

    /// `gpu.txt`: the flight recorder's text, with the channels a structural error stopped.
    pub fn flight_recorder(&self) -> String {
        self.recorder.text(&self.shared.channel_errors())
    }

    /// The flight recorder itself.
    pub fn recorder(&self) -> &FlightRecorder {
        &self.recorder
    }

    /// Whether nothing the GPU holds can still change anything a guest waits for (module docs,
    /// "Idleness"). While a channel waits on something only the acquire poll finds, it waits at
    /// most [`IDLE_POLL_WAIT`] for a poll that starts after it began and the idle turn after it,
    /// and answers "busy" when they have not come by then. "Busy" after [`Gpu::stop`].
    pub fn is_idle(&self) -> bool {
        let Some(first) = self.look() else {
            return false;
        };
        if !first.waiting {
            return true;
        }
        // A poll that started after the look, then the idle turn that follows it (the front
        // publishes before it marks itself quiet), within one budget.
        let started = Instant::now();
        if !self.status.wait_for_poll(first.polls, IDLE_POLL_WAIT) {
            return false;
        }
        let left = IDLE_POLL_WAIT.saturating_sub(started.elapsed());
        self.shared.wait_quiet(left) && self.look().is_some()
    }

    /// Tells `wake` each time the front thread ends a turn idle (module docs, "Idleness", "The
    /// quiet signal"): `core`'s parked cores ask again then. One GPU has at most one.
    ///
    /// # Errors
    ///
    /// [`QuietWakeTaken`] if a wake was set before; the first one is kept.
    pub fn set_quiet_wake(&self, wake: Arc<dyn QuietWake>) -> Result<(), QuietWakeTaken> {
        self.shared.set_quiet_wake(wake)
    }

    /// One look (module docs, "Idleness"): what the front published, if the GPU is idle but for
    /// what the poll may still find. It never waits.
    fn look(&self) -> Option<Seen> {
        let before = self.status.activity.load(Ordering::SeqCst);
        let quiet = self.shared.quiet_turn();
        let published = self.status.published();
        let taken = self.back.taken.load(Ordering::SeqCst);
        let in_flight = self.back.in_flight.load(Ordering::SeqCst);
        let quiet_after = self.shared.quiet_turn();
        let after = self.status.activity.load(Ordering::SeqCst);
        idle(Looked {
            quiet,
            sent: published.sent,
            taken,
            in_flight,
            quiet_after,
            before,
            after,
        })
        .then_some(Seen {
            polls: published.counters.polls,
            waiting: published.waiting,
        })
    }
}

/// Whether [`Gpu::stop`] tells the back thread to stop rather than letting it take what is left
/// on the ring (module docs, "Shutdown"): only when the front thread ended with `front`'s error.
fn stops_the_back(front: &Result<(), FrontError>) -> bool {
    front.is_err()
}

/// How long [`Gpu::is_idle`] waits for an acquire poll (module docs, "Idleness"): 100 times
/// [`POLL_LONGEST_SLEEP`], the longest the front sleeps between two polls, for the host to run
/// the front thread.
pub const IDLE_POLL_WAIT: Duration = POLL_LONGEST_SLEEP.saturating_mul(100);

/// What an idle look found of the acquire poll.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Seen {
    /// Polls the front had finished.
    polls: u64,
    /// Whether a channel waits on something only the poll finds.
    waiting: bool,
}

/// What [`Gpu::look`] read, in the order it read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Looked {
    /// [`FrontShared::quiet_turn`], read first.
    quiet: Option<u64>,
    /// Entries the front had pushed at the end of its last turn.
    sent: u64,
    /// Entries the back has taken.
    taken: u64,
    /// Lists in flight or open on the back.
    in_flight: u64,
    /// [`FrontShared::quiet_turn`], read again.
    quiet_after: Option<u64>,
    /// The activity counter, read first.
    before: u64,
    /// The activity counter, read last.
    after: u64,
}

/// Whether what [`Gpu::look`] read says the GPU is idle (module docs, "Idleness"): the front
/// quiet in both reads with no turn started between them, every entry it pushed taken, no list in
/// flight or open, and no back report reaching the front while it looked.
fn idle(looked: Looked) -> bool {
    looked.quiet.is_some()
        && looked.quiet == looked.quiet_after
        && looked.sent == looked.taken
        && looked.in_flight == 0
        && looked.before == looked.after
}

impl Drop for Gpu {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            tracing::error!(%error, "a GPU thread failed");
        }
    }
}

impl GpuVmFront for Gpu {
    fn create_address_space(&self, space: AsId, vm: Arc<GpuVm>) {
        let first = {
            let mut spaces = self.spaces.lock().unwrap_or_else(PoisonError::into_inner);
            let first = spaces.is_empty();
            if !spaces.insert(space) {
                tracing::warn!(
                    space = space.0,
                    "an address space was made twice; the first is kept"
                );
                return;
            }
            first
        };
        let front = VmMemory::new(vm, Arc::clone(&self.guest), Arc::clone(&self.syncpoints));
        let back: Arc<dyn GpuMemory> = Arc::new(front.for_thread());
        let table = Arc::clone(front.vm());
        self.inbox.post(Control::AddSpace {
            space,
            memory: back,
            watch: Arc::clone(&self.watch),
            present: first,
        });
        self.shared
            .create_address_space(space, table, Arc::new(front));
        self.recorder.record(GpuEvent::SpaceMade { space });
    }

    fn update_vm(&self, space: AsId, update: VmUpdate, done: Option<Ticket>) {
        self.recorder.record(GpuEvent::Update {
            space,
            update,
            ticket: done,
        });
        self.shared.update_vm(space, update, done);
    }

    fn barrier(&self, done: Ticket) {
        self.recorder.record(GpuEvent::Barrier { ticket: done });
        self.shared.barrier(done);
    }
}

impl ChannelFront for Gpu {
    fn create_channel(&self, config: ChannelConfig) -> Result<ChannelId, GpuError> {
        match self.shared.create_channel(config) {
            Ok(channel) => {
                self.inbox.post(Control::AddChannel {
                    channel,
                    space: config.space,
                });
                self.recorder.record(GpuEvent::ChannelMade {
                    channel,
                    space: config.space,
                    syncpoint: config.syncpoint,
                });
                Ok(channel)
            }
            Err(error) => {
                tracing::warn!(%error, "the GPU front refused a channel");
                self.recorder.record(GpuEvent::ChannelRefused {
                    space: config.space,
                });
                Err(GpuError)
            }
        }
    }

    fn bind_class(&self, channel: ChannelId, class: u32) -> Result<(), GpuError> {
        if self.shared.has_channel(channel) {
            Ok(())
        } else {
            tracing::warn!(
                channel = channel.0,
                class,
                "a class was bound on a channel the GPU front does not hold"
            );
            Err(GpuError)
        }
    }

    fn submit(&self, channel: ChannelId, submission: Submission) -> Submitted {
        let (entries, wait, increments, wfi) = (
            submission.entries.len(),
            submission.wait,
            submission.increments,
            submission.wfi,
        );
        let outcome = self.shared.submit(channel, submission);
        if outcome == Submitted::Full {
            self.status.full.fetch_add(1, Ordering::Relaxed);
        }
        self.recorder.record(GpuEvent::Submitted {
            channel,
            entries,
            wait,
            increments,
            wfi,
            outcome,
        });
        outcome
    }

    fn destroy_channel(&self, channel: ChannelId) {
        self.recorder.record(GpuEvent::ChannelDestroyed { channel });
        self.shared.destroy_channel(channel);
    }
}

impl PresentFront for Gpu {
    fn present(&self, request: PresentRequest) {
        self.recorder.record(GpuEvent::Present(request));
        self.shared.present(request);
    }
}

impl GpuFront for Gpu {
    fn channel_error(&self, channel: ChannelId) -> Option<ChannelError> {
        self.shared.channel_error(channel)
    }
}
