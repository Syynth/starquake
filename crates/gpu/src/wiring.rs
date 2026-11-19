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
//! the back increments a watched counter). Each push from the front or the back goes through
//! this module's recorder first (`recorder`).
//!
//! # Presents
//!
//! A present names no address space, so the back reads every present through one: the first
//! address space made. A **guess**: one GPU address space per title, which presents read (the
//! executor's module docs, "Presents"; issue #330's scope note). A title that presents from a
//! second space would show nothing from it (the present is refused and counted).
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
//! [`Gpu::counters`] gives [`GpuCounters`], every counter of the front, the executor and the
//! fence path, flattened for `summary.json`; [`Gpu::flight_recorder`] gives `gpu.txt`'s text
//! (`recorder`).
//!
//! # Idleness
//!
//! [`Gpu::is_idle`] says whether nothing the GPU holds can still change anything a guest waits
//! for: the front is quiet ([`FrontShared::is_quiet`]: nothing queued for it, and it has ended a
//! turn idle since it last took anything), the back has taken every entry the front pushed, and
//! no command list is in flight or open. A channel parked on an acquire the CPU must release, or
//! an ordered update behind it, does not make the GPU busy: only the guest can end that wait.
//! The front's pushed count is stored before it marks itself quiet; the back stores its lists in
//! flight before its taken count; and the back's report of a completion to the front bumps a
//! counter before and after it reaches the front, which [`Gpu::is_idle`] reads first and last, so
//! a completion that lands while it looks makes the answer "busy". It can answer "busy" for a GPU
//! that is idle (a look that overlaps a completion); it does not answer "idle" while an entry is
//! unapplied or a list is in flight (`the_gpu_is_busy_while_a_list_is_in_flight`).
//!
//! # Shutdown
//!
//! [`Gpu::stop`] (and dropping the `Gpu`) stops the front thread, which publishes what it pushed
//! and drops the ring's producer; the back thread then takes what is left and ends. Both threads
//! are joined (`both_threads_end_when_the_gpu_stops`).
//!
//! # Not here
//!
//! - The real shader cache: shader-pipeline slice 19 builds it, and an adapter from it to G18's
//!   seam ([`ShaderCache`]) is the caller's to give, in [`GpuDeps::shaders`].
//! - Choosing the render backend, and writing `gpu.txt` and the counters into a run's artifacts:
//!   the run builds a `Gpu` once a backend is chosen (render-backend slice R12).

mod back_thread;
mod recorder;

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde::Serialize;
use starquake_memory::{GuestMemory, WriteWatch};
use starquake_render::Backend;
use starquake_render::wake::CompletionWake;
use thiserror::Error;

use crate::back::exec::{ExecConfig, Executor, ShaderCache};
use crate::back::fence::{FenceClock, FenceEnv, Fences, FrontWake, Mode, Report};
use crate::channel::ChannelError;
use crate::front::{
    FrontClock, FrontCounters, FrontDeps, FrontError, FrontShared, FrontThread, HostClock,
    OrderedTable, spawn_observed,
};
use crate::front::Front;
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
pub use recorder::{FlightRecorder, GpuEvent};

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
    /// The executor's settings.
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

/// Every counter of the GPU (gpu-protocol, "Observability"), flattened for `summary.json`. The
/// spec lists, besides what is here: writes dropped as same-value, macro calls, memo hits and
/// the ring's peak occupancy, which no slice counts yet.
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
}

impl GpuCounters {
    fn new(front: &FrontCounters, back: &BackStatus, full: u64, dropped: u64) -> Self {
        let closed = back.fences.closed;
        let lists = [
            closed.ring_empty,
            closed.present,
            closed.wait_for_idle,
            closed.waiter,
            closed.elapsed,
            closed.set_reference,
            closed.catch_up,
            closed.channel_removed,
        ]
        .iter()
        .fold(0u64, |sum, &count| sum.saturating_add(count));
        Self {
            words: front.words,
            submissions: front.submissions,
            submissions_full: full,
            submissions_dropped: dropped,
            increments: front.increments,
            acquires_parked: front.parks,
            fence_waits: front.fence_waits,
            sync_waits: front.sync_waits,
            macro_catch_ups: front.catch_ups,
            polls: front.polls,
            ordered: front.ordered,
            macro_failures: front.macro_failures,
            channels_stopped: front.stopped_channels,
            presents_pushed: front.presents,
            entries_applied: back.exec.applied,
            entries_refused: back.exec.failed,
            unhandled_words: back.exec.unhandled_words,
            presents: back.exec.presents,
            dispatches: back.exec.dispatches,
            dispatches_skipped: back.exec.dispatches_skipped,
            write_backs: back.exec.write_backs,
            write_back_failures: back.exec.write_back_failures,
            render_catch_ups: back.exec.catch_ups,
            render_catch_up_failures: back.exec.catch_up_failures,
            fenced_actions: back.fences.fenced,
            fenced_actions_skipped_empty: back.fences.skipped_empty,
            lists_submitted: lists,
            lost_wakes: back.fences.lost_wakes,
            failed_releases: back.fences.failed_releases,
        }
    }
}

/// What the front thread publishes after each turn, and the idleness counter (module docs,
/// "Idleness").
#[derive(Default)]
struct Status {
    front: Mutex<FrontCounters>,
    /// [`Front::sent`], stored after each turn.
    sent: AtomicU64,
    /// Bumped before and after each back report reaches the front.
    activity: AtomicU64,
    /// Submissions answered `Full`.
    full: AtomicU64,
}

impl Status {
    fn front(&self) -> FrontCounters {
        *self.front.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn bump(&self) {
        self.activity.fetch_add(1, Ordering::SeqCst);
    }
}

/// The back's report to the front, recorded, and counted for idleness (module docs).
struct ReportToFront {
    front: Arc<FrontShared>,
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
            failed: report.failed.clone(),
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
            *observed.front.lock().unwrap_or_else(PoisonError::into_inner) = front.counters();
            observed.sent.store(front.sent(), Ordering::SeqCst);
        })?;

        let env = FenceEnv {
            syncpoints: Arc::clone(&syncpoints),
            waiters,
            completions,
            clock: fence_clock,
            front: Arc::new(ReportToFront {
                front: Arc::clone(&shared),
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
        // The producer is gone with the front thread; the stop is only a safety net for a back
        // thread that would otherwise wait for its next wake.
        self.inbox.stop();
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
            &self.status.front(),
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

    /// Whether nothing the GPU holds can still change anything (module docs, "Idleness").
    pub fn is_idle(&self) -> bool {
        let before = self.status.activity.load(Ordering::SeqCst);
        let quiet = self.shared.is_quiet();
        let sent = self.status.sent.load(Ordering::SeqCst);
        let taken = self.back.taken.load(Ordering::SeqCst);
        let in_flight = self.back.in_flight.load(Ordering::SeqCst);
        let after = self.status.activity.load(Ordering::SeqCst);
        quiet && sent == taken && in_flight == 0 && before == after
    }
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
                self.recorder
                    .record(GpuEvent::ChannelRefused { space: config.space });
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
        self.recorder
            .record(GpuEvent::ChannelDestroyed { channel });
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
