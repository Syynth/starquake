//! The graphics server in a run (`docs/specs/graphics-server.md`, port slice 16, "Wiring"):
//! the `Syncpoints`, the completion queue, gsv's shared state, the GPU front, gsv's servers in
//! front of the kernel, the timer thread's turn over the queue, and the replay of recorded notes.
//!
//! No reference source is ported: the wiring is the spec's.
//!
//! # What is built
//!
//! [`Graphics::new`] builds, in the order the spec's "The pieces" (`starquake_gsv::shared`) asks:
//! the [`CompletionQueue`] (rung through a [`Wake`] the run gives it), the `Syncpoints` that push
//! to it, gsv's [`Shared`] state over both, recording its notes for the root's number
//! ([`GSV_ROOT`]), and the GPU front ([`NullFront`]). [`install`](super::install) puts four of
//! gsv's per-family servers (decision 6) in front of the kernel, each known by its own number,
//! none on a port and none in `sm:`'s registry:
//!
//! | Server | Number | Built with |
//! |---|---|---|
//! | root | [`GSV_ROOT`] | `root_server_with_page`, its [`RootState`] naming the three below: 202, 204, 497 and 1402 reach them |
//! | syncpt | [`GSV_SYNCPT`] | `syncpt_server` over the shared state |
//! | gpu | [`GSV_GPU`] | `gpu_server` over the shared state and the front |
//! | nvsrv | [`GSV_NVSRV`] | `nvsrv_server` over the shared state and the front, with the front as its channel front too, and told of deferred replies as `fsp-srv` is ([`super::Reads::ready`]) |
//!
//! The root answers 202, 204, 497 and 1402 with its family's server (`starquake_gsv::root`,
//! "The family sessions"); every other root command is still the root's logged stub. The vsync
//! clock (slice 12, `starquake_gsv::shared::vsync`) needs no server: it is part of gsv's shared
//! state and runs in the timer thread's turn (below), and vi2 (slice 13) will start it by adding
//! listeners. Slices 13 to 15 (vi2, present and video: issues 191 to 193) are not built: when
//! they land, each adds its server here beside these and its number to the root, and 305, the
//! video commands and the native image commands stop being stubs.
//!
//! # The GPU front
//!
//! The GPU's own front (gpu-protocol, `crates/gpu`'s `GpuFront`, issue 12) is not built yet, so
//! the front is [`NullFront`], which runs nothing. It does what the spec asks of the front's
//! order and completion with no GPU behind it, so that a title waiting on the GPU is not stuck:
//!
//! - an ordered update with a ticket, and a barrier, are done at once: it pushes
//!   `GpuOrdered { ticket }` to the queue (spec: "Signals with no host work behind them happen at
//!   once");
//! - a channel is a number, counting up from [`FIRST_CHANNEL`], with the syncpoint it was made
//!   with; a class is bound to any channel it made;
//! - a submission's commands are dropped, and its increments are performed at once on the
//!   channel's syncpoint (`Syncpoints::increment_by`: one push if the syncpoint is watched,
//!   however many increments), and it answers `Queued`, never `Full`.
//!
//! Where it departs from what a GPU does, each pinned by the test named, in `graphics/tests.rs`:
//!
//! | Case | A GPU | Here |
//! |---|---|---|
//! | A submission's commands | run | dropped (`a_submission_s_increments_are_performed_at_once`) |
//! | A submission's wait fence (flag bit 0) | waited for before the increments | not waited for: the increments are performed at once whether it is reached or not (`a_submission_s_wait_fence_is_not_waited_for`) |
//! | A full queue | `Full`, then `GpuQueueRoom` | never full (`a_submission_s_increments_are_performed_at_once`) |
//! | A submission on a channel it did not make | the GPU's error | logged as an error, nothing performed, `Queued` (`a_submission_on_an_unknown_channel_does_nothing`) |
//!
//! **Tickets are gsv's.** gsv's GPU memory numbers every ordered update and barrier it hands the
//! front (`starquake_gsv::gpu::GpuMemory`, whose snapshot keeps the next ticket); the front
//! never makes one, and only gives back the ticket it was handed, as `GpuOrdered`
//! (`ordered_updates_and_barriers_are_done_at_once_in_a_run`).
//!
//! In [`FrontMode::Replay`] it does none of the completions: no ticket is pushed and no
//! increment performed, so that only the recorded notes complete anything (below, "Replay").
//! Channels are numbered the same way in both modes, so a replay hands out the numbers the run
//! did.
//!
//! The front keeps one entry per channel it made, removed when nvsrv destroys the channel;
//! nvsrv makes at most one per open nvhost-gpu fd (`starquake_gsv::nvsrv::MAX_OPEN_FILES`), so
//! the table is bounded by that.
//!
//! # The timer thread's turn
//!
//! Each turn of the run's timer thread (`crate::machine`'s `timer` module) calls
//! [`Graphics::turn`] with a [`KernelHost`] over the run's dispatcher, and the time: the drain
//! (`Shared::drain`), then the shared page's turn (`Shared::tick_page`), then the vsync clock's:
//! in a run ([`FrontMode::Live`]) the live turn at the turn's time (`Shared::tick_vsync`), in a
//! replay ([`FrontMode::Replay`]) the delivery of the ticks fed back
//! (`Shared::tick_replayed_vsync`; below, "Replay"). The thread sleeps no later than
//! [`Graphics::deadline`], which in a run includes the vsync clock's deadline, and the queue's
//! [`Wake`] rings it. The thread runs the turn only when [`Graphics::wants_turn`] says so: the
//! graphics server is not idle but for the clock, or the clock's deadline has come by the loop's
//! own copy of the clock, so a clock waiting for its deadline adds no reading of the
//! environment. [`KernelHost`] is the drain's [`TimerHost`] over the kernel:
//!
//! - a note is recorded by `Dispatcher::record_host_event`, under the kernel lock, on the core
//!   and thread the host is given (the timer thread's: core 0, no guest thread);
//! - an event is signalled by `Dispatcher::signal_event`, which records the waits it ends;
//! - a device-share lock is released by `Dispatcher::unlock_device_share`, which checks the
//!   token;
//! - the shared page's transfer memories are read and written through the run's guest memory
//!   by `Dispatcher::read_transfer_memory` and `Dispatcher::write_transfer_memory`, at the
//!   transfer memory as the kernel holds it now: the same calls `IpcContext`'s copies make, so
//!   a turn sees the same errors a request would: `InvalidHandle` for a transfer memory that is
//!   gone, `InvalidCurrentMemory` for bytes outside it.
//!
//! An environment that fails while the turn records or signals (a replay that diverged) is kept,
//! the first one only, and [`KernelHost::finish`] returns it: the run then stops, as for any
//! failure of the environment (decision log, 2026-11-17, "Env failures"). The drain carries on
//! to the end of what it took, since it cannot be stopped half-way: a signal the failure refused
//! is logged as the kernel's refusal, as `NotImplemented`.
//!
//! # Telling a stalled run
//!
//! A run ends as an emulator error when every core has parked and nothing could end a wait
//! (`crate::machine`'s `WAITING`). The kernel's own check (`Dispatcher::is_stalled`) does not
//! know that the timer thread's turn signals gsv's events, so the core loop asks
//! [`Graphics::is_idle`] first: a run is stalled only if the graphics server is idle too
//! (nothing queued, nothing held by the shared page, no page signal waiting, no vsync tick to
//! come: in a run the clock stopped, in a replay no tick fed back) and then the kernel says so. With [`NullFront`] nothing is in flight on a GPU, so that is everything the
//! turn could still signal; a GPU front with work in flight must add that (a follow-up for
//! issue 12's front).
//!
//! # Replay
//!
//! The spec's replay: "a fake `GpuFront` takes submissions, and the recorded notes are fed back
//! in their order". A run whose front is in [`FrontMode::Replay`] takes the same requests and
//! completes nothing on its own; [`Graphics::replay`] feeds one recorded note back as the input
//! it records, and the next turn drains it:
//!
//! | Note | Fed back as |
//! |---|---|
//! | `SyncpointObserved { id, value }` | the counter brought to `value` if it is behind it (by the 32-bit wrapping comparison: up to 2^31 - 1 ahead), then `SyncpointChanged { id, value }` pushed as it is, also for a value not ahead of the counter (a stale one, which the queue can drain after a newer one); the counter's own push, for a watched id, has the same value and the queue folds the two into one entry (`starquake_gsv::completion`, "Coalescing") |
//! | `GpuOrderedDone { ticket }` | `GpuOrdered { ticket }` pushed |
//! | `QueueRoom { channel }` | `GpuQueueRoom { channel }` pushed |
//! | `VsyncTick { tick }` | fed back to the vsync clock (`Shared::replay_vsync`), which takes only the next number ([`ReplayError::Vsync`]); the next turn delivers it, counted, recorded and signalled as the run's was. In a replay the clock does not tick on its own |
//! | `SyncptPageAnswered`, `SyncptPageAcknowledged` | refused ([`ReplayError::NotAnInput`]): they are what a page turn writes, not an input |
//!
//! Each note is fed back and drained on its own, in the recorded order: the drain acts on its
//! completions one at a time in the queue's order, so one drain of several notes and one drain
//! per note act alike, and feeding one at a time keeps the queue from folding two values of one
//! syncpoint into one, which a single drain of the run did not do.
//!
//! What this replays exactly, and what it does not (the test
//! `a_recorded_session_replays_to_the_same_state`, in `hle/tests/graphics.rs`, pins the first):
//!
//! - **Replayed:** a session of requests whose GPU inputs are the notes above: the IPC events,
//!   the notes in their order, the events the drain signals, and the state of gsv's shared
//!   state, its GPU memory, the syncpoints and the four servers, when the notes are fed back at
//!   the turns the run drained them.
//! - **Not replayed: the counters between a submission and its note.** In a run the front
//!   performs a submission's increments when it is handed it; in a replay the counter moves only
//!   when the note is fed back. A request in between that reads the counter (syncpt 103, the
//!   shared page) sees different values. The note records what gsv acted on, not when the
//!   counter moved: no note records that.
//! - **Not replayed: the shared page's answers.** A page turn's answer depends on the turn's
//!   time and on the counters as they are when it is written, neither of which a note fixes; its
//!   notes (codes 2 and 9) are what it wrote, and a replay that reproduces both inputs writes
//!   the same notes. Feeding them back is not built.
//! - **Replayed: the vsync clock's ticks.** Each `VsyncTick` note is fed back where the run's
//!   turn made it, and the replay turn delivers it: the same note, the same signals, the same
//!   count and listeners (`a_recorded_vsync_tick_replays_to_the_same_signal_and_state`, in
//!   `hle/tests/graphics/vsync.rs`).
//! - **Not replayed: the vsync clock's deadline.** It follows the times of the run's turns, which
//!   no note records; a replay's clock keeps the deadline its start gave it and never ticks on
//!   it.
//! - **Not replayed: where the run's turns fell.** The recorded stream says in which order the
//!   notes and the requests came, which is what a replay follows; a turn that drained nothing
//!   leaves no trace.
//!
//! # What is snapshotted
//!
//! Each piece saves itself (the syncpoints, the queue, the shared state, the page, the GPU
//! memory and each server); composing them into one snapshot of gsv that a restore can check as
//! a whole (the spec's "What is snapshotted", and the cross-checks the pieces' docs leave to
//! this slice) needs a kernel snapshot to check the restored ids against (issue 260), and is not
//! built here.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use starquake_gpu::interface::{
    ChannelId, Completions, GpuCompletion, SyncpointId, Ticket, VmUpdate,
};
use starquake_gpu::syncpoint::Syncpoints;
use starquake_gpu::vm::GpuVm;
use starquake_gsv::completion::{Completion, CompletionQueue, Wake};
use starquake_gsv::gpu::{self as gsv_gpu, AsId, GpuVmFront};
use starquake_gsv::note::Note;
use starquake_gsv::nvsrv::gpu::{ChannelConfig, ChannelFront, GpuError, Submission, Submitted};
use starquake_gsv::nvsrv::{self, NvsrvFamily};
use starquake_gsv::root::{self, RootState};
use starquake_gsv::shared::vsync::VsyncReplayError;
use starquake_gsv::shared::{Shared, TimerHost};
use starquake_gsv::syncpt;
use starquake_kernel::dispatch::{DeviceShareError, Dispatcher, WakeError};
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::result::{KernelError, SvcError, SvcResult};
use starquake_memory::GuestMemory;
use starquake_services::framework::{Notify, UnknownPolicy};
use starquake_trace::{Env, EnvError, EventKind, HostNanos, Recorder, ThreadId};

use super::{Deliveries, Delivering, GSV_GPU, GSV_NVSRV, GSV_ROOT, GSV_SYNCPT, ServicesError};

#[cfg(test)]
mod tests;

/// The first channel number [`NullFront`] hands out; the next are counted up from it.
/// Starquake's own: the number is the front's (gpu-protocol), and nothing reads it but gsv.
pub const FIRST_CHANNEL: u32 = 1;

/// What the GPU front completes on its own (module docs, "The GPU front").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontMode {
    /// A run: ordered updates and barriers are done at once, and a submission's increments are
    /// performed at once.
    Live,
    /// A replay: nothing is completed but by the recorded notes ([`Graphics::replay`]).
    Replay,
}

/// The channels [`NullFront`] made: each one's syncpoint, and the next number.
#[derive(Debug)]
struct Channels {
    next: u32,
    made: BTreeMap<ChannelId, SyncpointId>,
}

/// The GPU front of a run with no GPU back end (module docs, "The GPU front"): gsv's
/// [`GpuVmFront`] and [`ChannelFront`].
pub struct NullFront {
    mode: FrontMode,
    syncpoints: Arc<Syncpoints>,
    queue: Arc<CompletionQueue>,
    channels: Mutex<Channels>,
}

impl NullFront {
    /// A front over `syncpoints`, reporting to `queue`, in `mode`.
    #[must_use]
    pub fn new(mode: FrontMode, syncpoints: Arc<Syncpoints>, queue: Arc<CompletionQueue>) -> Self {
        Self {
            mode,
            syncpoints,
            queue,
            channels: Mutex::new(Channels {
                next: FIRST_CHANNEL,
                made: BTreeMap::new(),
            }),
        }
    }

    /// The channels it holds now, each with its syncpoint, in channel order.
    #[must_use]
    pub fn channels(&self) -> Vec<(ChannelId, SyncpointId)> {
        let channels = self.channels.lock().unwrap_or_else(PoisonError::into_inner);
        channels.made.iter().map(|(id, sp)| (*id, *sp)).collect()
    }

    /// Reports `ticket` done at once, in a run.
    fn done(&self, ticket: Ticket) {
        if self.mode == FrontMode::Live {
            self.queue
                .push(Completion::Gpu(GpuCompletion::GpuOrdered { ticket }));
        }
    }
}

impl GpuVmFront for NullFront {
    /// Nothing: no GPU reads the table.
    fn create_address_space(&self, _space: AsId, _vm: Arc<GpuVm>) {}

    fn update_vm(&self, _space: AsId, _update: VmUpdate, done: Option<Ticket>) {
        if let Some(ticket) = done {
            self.done(ticket);
        }
    }

    fn barrier(&self, done: Ticket) {
        self.done(done);
    }
}

impl ChannelFront for NullFront {
    /// The next number, with `config`'s syncpoint; [`GpuError`] once the numbers are used up.
    fn create_channel(&self, config: ChannelConfig) -> Result<ChannelId, GpuError> {
        let mut channels = self.channels.lock().unwrap_or_else(PoisonError::into_inner);
        let id = channels.next;
        channels.next = id.checked_add(1).ok_or(GpuError)?;
        channels.made.insert(ChannelId(id), config.syncpoint);
        Ok(ChannelId(id))
    }

    /// Any class, on a channel it made; [`GpuError`] on any other.
    fn bind_class(&self, channel: ChannelId, _class: u32) -> Result<(), GpuError> {
        let channels = self.channels.lock().unwrap_or_else(PoisonError::into_inner);
        if channels.made.contains_key(&channel) {
            Ok(())
        } else {
            Err(GpuError)
        }
    }

    /// Performs the increments at once in a run (module docs, "The GPU front"); always
    /// `Queued`.
    fn submit(&self, channel: ChannelId, submission: Submission) -> Submitted {
        let syncpoint = {
            let channels = self.channels.lock().unwrap_or_else(PoisonError::into_inner);
            channels.made.get(&channel).copied()
        };
        let Some(syncpoint) = syncpoint else {
            tracing::error!(
                ?channel,
                "the null GPU front was given a channel it did not make"
            );
            return Submitted::Queued;
        };
        if self.mode == FrontMode::Live {
            self.syncpoints
                .increment_by(syncpoint, submission.increments);
        }
        Submitted::Queued
    }

    fn destroy_channel(&self, channel: ChannelId) {
        let mut channels = self.channels.lock().unwrap_or_else(PoisonError::into_inner);
        channels.made.remove(&channel);
    }
}

/// Why a note could not be fed back ([`Graphics::replay`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReplayError {
    /// The note is one a page turn writes, not an input the run took from outside.
    #[error(
        "a note with code {code} is what the graphics server wrote, not something it was told, so it is not fed back"
    )]
    NotAnInput {
        /// The note's code.
        code: u32,
    },
    /// A recorded vsync tick the clock does not take: not the next one.
    #[error("a recorded vsync tick could not be fed back: {0}")]
    Vsync(#[from] VsyncReplayError),
}

/// The graphics server of a run: its shared state and its GPU front (module docs).
pub struct Graphics {
    shared: Arc<Shared>,
    front: Arc<NullFront>,
    /// Whether the run's own inputs complete things, or only recorded notes: the front's mode,
    /// which also picks the vsync clock's turn (module docs, "The timer thread's turn").
    mode: FrontMode,
}

impl Graphics {
    /// The queue, rung through `wake` if given, the syncpoints, gsv's shared state recording for
    /// [`GSV_ROOT`], and a [`NullFront`] in `mode` (module docs, "What is built").
    #[must_use]
    pub fn new(mode: FrontMode, wake: Option<Arc<dyn Wake>>) -> Self {
        let mut queue = CompletionQueue::new();
        if let Some(wake) = wake {
            queue = queue.with_wake(wake);
        }
        let queue = Arc::new(queue);
        let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
        let syncpoints = Arc::new(Syncpoints::new(completions));
        let front = Arc::new(NullFront::new(
            mode,
            Arc::clone(&syncpoints),
            Arc::clone(&queue),
        ));
        let shared = Arc::new(Shared::new(syncpoints, queue, GSV_ROOT));
        Self {
            shared,
            front,
            mode,
        }
    }

    /// gsv's shared state, with its syncpoints and queue.
    #[must_use]
    pub const fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }

    /// The GPU front.
    #[must_use]
    pub const fn front(&self) -> &Arc<NullFront> {
        &self.front
    }

    /// Puts gsv's servers in front of `dispatcher` (module docs, "What is built"), telling
    /// `ready` of nvsrv's deferred replies.
    pub(super) fn add_servers(
        &self,
        dispatcher: &Dispatcher,
        policy: UnknownPolicy,
        ready: Arc<dyn Deliveries>,
    ) -> Result<(), ServicesError> {
        let syncpt =
            syncpt::syncpt_server(policy, Arc::clone(&self.shared)).with_service_id(GSV_SYNCPT);
        dispatcher.add_server(syncpt::SYNCPT_NAME, GSV_SYNCPT, None, Box::new(syncpt))?;

        let vm: Arc<dyn GpuVmFront> = Arc::<NullFront>::clone(&self.front);
        let gpu = gsv_gpu::gpu_server(Arc::clone(&self.shared), Arc::clone(&vm), policy)
            .with_service_id(GSV_GPU);
        dispatcher.add_server(gsv_gpu::GPU_NAME, GSV_GPU, None, Box::new(gpu))?;

        let channels: Arc<dyn ChannelFront> = Arc::<NullFront>::clone(&self.front);
        let family = NvsrvFamily::new(Arc::clone(&self.shared), Arc::clone(&vm))
            .with_channel_front(channels);
        let delivering = Arc::new(Delivering {
            server: OnceLock::new(),
            ready,
        });
        let notify: Arc<dyn Notify> = Arc::<Delivering>::clone(&delivering);
        let driver = nvsrv::nvsrv_server(Arc::clone(&self.shared), vm, policy)
            .with_family(family)
            .with_service_id(GSV_NVSRV)
            .with_notify(notify);
        let driver = dispatcher.add_server(nvsrv::NVSRV_NAME, GSV_NVSRV, None, Box::new(driver))?;
        if delivering.server.set(driver).is_err() {
            tracing::error!("nvsrv's server was numbered twice");
        }

        let state = RootState::new()
            .with_syncpt_server(GSV_SYNCPT)
            .with_gpu_server(GSV_GPU)
            .with_nvsrv_server(GSV_NVSRV);
        let graphics_root = root::root_server_with_page(state, policy, Arc::clone(&self.shared))
            .with_service_id(GSV_ROOT);
        dispatcher.add_server(root::ROOT_NAME, GSV_ROOT, None, Box::new(graphics_root))?;
        Ok(())
    }

    /// The timer thread's turn (module docs, "The timer thread's turn"): drains the queue
    /// through `host`, then runs the shared page's turn at `now`, then the vsync clock's turn for
    /// the front's mode. Returns what the drain took, in order.
    pub fn turn(&self, host: &mut dyn TimerHost, now: HostNanos) -> Vec<Completion> {
        let taken = self.shared.drain(host);
        self.shared.tick_page(host, now);
        match self.mode {
            FrontMode::Live => self.shared.tick_vsync(host, now),
            FrontMode::Replay => self.shared.tick_replayed_vsync(host),
        }
        taken
    }

    /// When the timer thread must turn next at the latest: the earlier of the shared page's
    /// deadline, if it holds a request, and, in a run, the vsync clock's, while it runs.
    #[must_use]
    pub fn deadline(&self) -> Option<HostNanos> {
        let page = self.shared.page_deadline();
        let vsync = match self.mode {
            FrontMode::Live => self.shared.vsync_deadline(),
            FrontMode::Replay => None,
        };
        match (page, vsync) {
            (Some(page), Some(vsync)) => Some(page.min(vsync)),
            (page, vsync) => page.or(vsync),
        }
    }

    /// Whether the graphics server has nothing left that could signal an event without a guest's
    /// call first (module docs, "Telling a stalled run"): nothing queued, no request held by the
    /// shared page, no guest signal of the page waiting for its turn, and no vsync tick to come
    /// (in a run the clock stopped, in a replay no tick fed back). It takes gsv's shared lock,
    /// so it waits for a drain in progress, whose signals are then made.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        let ticking = match self.mode {
            FrontMode::Live => self.shared.vsync_deadline().is_some(),
            FrontMode::Replay => false,
        };
        !ticking && self.is_idle_but_the_clock()
    }

    /// Whether the timer thread should run [`Graphics::turn`] at `now`, the loop's own reading
    /// of the clock (module docs, "The timer thread's turn"): the graphics server is not idle
    /// but for the vsync clock, or, in a run, the clock's deadline has come by `now`.
    #[must_use]
    pub fn wants_turn(&self, now: HostNanos) -> bool {
        let due = match self.mode {
            FrontMode::Live => self.shared.lock().vsync().is_due(now),
            FrontMode::Replay => false,
        };
        due || !self.is_idle_but_the_clock()
    }

    /// [`Graphics::is_idle`] but for a running vsync clock: nothing queued, nothing held or
    /// signalled at the shared page, and no tick fed back for replay.
    fn is_idle_but_the_clock(&self) -> bool {
        let locked = self.shared.lock();
        let page = locked.page();
        let idle = page.held().is_none()
            && !page.is_signalled()
            && locked.vsync().replayed() == 0
            && self.shared.queue().is_empty();
        drop(locked);
        idle
    }

    /// Feeds the recorded `note` back as the input it records (module docs, "Replay"); the next
    /// turn drains it. A note a page turn writes is refused and changes nothing.
    pub fn replay(&self, note: Note) -> Result<(), ReplayError> {
        let queue = self.shared.queue();
        let completion = match note {
            Note::SyncpointObserved { id, value } => {
                self.advance(id, value);
                GpuCompletion::SyncpointChanged { id, value }
            }
            Note::GpuOrderedDone { ticket } => GpuCompletion::GpuOrdered { ticket },
            Note::QueueRoom { channel } => GpuCompletion::GpuQueueRoom { channel },
            Note::VsyncTick { tick } => {
                self.shared.replay_vsync(tick)?;
                return Ok(());
            }
            Note::SyncptPageAnswered { .. } | Note::SyncptPageAcknowledged { .. } => {
                return Err(ReplayError::NotAnInput { code: note.code() });
            }
        };
        queue.push(Completion::Gpu(completion));
        Ok(())
    }

    /// Brings counter `id` to `value` if it is behind it by the 32-bit wrapping comparison.
    fn advance(&self, id: SyncpointId, value: u32) {
        let syncpoints = self.shared.syncpoints();
        let behind = value.wrapping_sub(syncpoints.read(id));
        // `> 0`, not `>= 0`: the two differ only for a value equal to the counter, where adding
        // 0 changes nothing either way (an equivalent mutation).
        if behind.cast_signed() > 0 {
            syncpoints.increment_by(id, behind);
        }
    }
}

/// The drain's [`TimerHost`] over the run's kernel (module docs, "The timer thread's turn").
pub struct KernelHost<'a> {
    dispatcher: &'a Dispatcher,
    env: &'a dyn Env,
    recorder: &'a Recorder,
    memory: &'a dyn GuestMemory,
    core: u8,
    thread: ThreadId,
    failed: Option<EnvError>,
}

impl<'a> KernelHost<'a> {
    /// A host over `dispatcher`, reading `env`'s clock, recording to `recorder` on `core` and
    /// `thread`, and reaching the shared page through `memory`, the run's guest memory.
    #[must_use]
    pub fn new(
        dispatcher: &'a Dispatcher,
        env: &'a dyn Env,
        recorder: &'a Recorder,
        memory: &'a dyn GuestMemory,
        (core, thread): (u8, ThreadId),
    ) -> Self {
        Self {
            dispatcher,
            env,
            recorder,
            memory,
            core,
            thread,
            failed: None,
        }
    }

    /// The first failure of the environment the turn met, if one did (module docs).
    pub fn finish(self) -> Result<(), EnvError> {
        self.failed.map_or(Ok(()), Err)
    }

    /// Keeps `error` if it is the first.
    fn fail(&mut self, error: EnvError) {
        tracing::error!(%error, "the environment failed during the graphics server's turn");
        self.failed.get_or_insert(error);
    }
}

impl TimerHost for KernelHost<'_> {
    fn record(&mut self, kind: EventKind) {
        let recorded = self.dispatcher.record_host_event(
            self.core,
            self.thread,
            kind,
            self.env,
            self.recorder,
        );
        if let Err(error) = recorded {
            self.fail(error);
        }
    }

    fn unlock_device_share(&mut self, token: DeviceShareToken) -> Result<(), DeviceShareError> {
        self.dispatcher.unlock_device_share(token)
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        match self.dispatcher.signal_event(event, self.env, self.recorder) {
            Ok(()) => Ok(()),
            Err(WakeError::Call(SvcError::Kernel(error))) => Err(error),
            Err(WakeError::Call(SvcError::Internal(what))) => {
                tracing::error!(what, "kernel bookkeeping failed signalling a gsv event");
                Err(KernelError::InvalidState)
            }
            Err(WakeError::Env(error)) => {
                self.fail(error);
                Err(KernelError::NotImplemented)
            }
        }
    }

    fn read_transfer_memory(
        &mut self,
        memory: ObjectId,
        offset: u64,
        out: &mut [u8],
    ) -> SvcResult<()> {
        self.dispatcher
            .read_transfer_memory(self.memory, memory, offset, out)
    }

    fn write_transfer_memory(
        &mut self,
        memory: ObjectId,
        offset: u64,
        data: &[u8],
    ) -> SvcResult<()> {
        self.dispatcher
            .write_transfer_memory(self.memory, memory, offset, data)
    }
}
