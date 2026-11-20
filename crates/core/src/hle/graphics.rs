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
//! ([`GSV_ROOT`]), and the GPU front ([`NullFront`]). [`install`](super::install) puts five of
//! gsv's per-family servers (decision 6) in front of the kernel, each known by its own number,
//! none on a port and none in `sm:`'s registry:
//!
//! | Server | Number | Built with |
//! |---|---|---|
//! | root | [`GSV_ROOT`] | `root_server_with_page`, its [`RootState`] naming the four below: 202, 204, 497 and 1402 reach the first three, 902, 906 and 1002 the video server |
//! | syncpt | [`GSV_SYNCPT`] | `syncpt_server` over the shared state |
//! | gpu | [`GSV_GPU`] | `gpu_server` over the shared state and the front |
//! | nvsrv | [`GSV_NVSRV`] | `nvsrv_server` over the shared state and the front, with the front as its channel front too, and told of deferred replies as `fsp-srv` is ([`super::Reads::ready`]) |
//! | vi2 | [`GSV_VI2`] | `vi2_server` over the shared state, with the services' environment and halt, which [`Graphics::add_servers`] takes so that the root never names a vi2 server that is not added |
//! | video | [`GSV_VIDEO`] | `video_server` over the shared state: the video context 902, 906 and 1002 hand out, and its decoder services with the null decoder (`starquake_gsv::video`, slice 15) |
//!
//! The root answers 202, 204, 497 and 1402 with its family's server (`starquake_gsv::root`,
//! "The family sessions"), 305 with the vi2 server ("The display session (305)"), and 902, 906
//! and 1002 with the video server (`starquake_gsv::root`, "Video (slice 15)";
//! `the_root_hands_out_video_sessions_that_count_frames`). Built with the shared state, the root
//! also answers the native memory and image commands (111, 112, 114, 314, 315, 423, 429, 430,
//! 435, 1510) from the native tables in it (`starquake_gsv::native`, slice 8;
//! `the_root_answers_native_memory_and_images`), the video image pool commands (991, 911, 912,
//! 961, 426 to 428) from the same tables (`starquake_gsv::native::pools`, slice 15), and 602 with
//! a camera session (`starquake_gsv::video::camera`); every other root command is still the
//! root's logged stub. The vsync clock (slice 12, `starquake_gsv::shared::vsync`) needs no
//! server: it is part of gsv's shared state and runs in the timer thread's turn (below), and
//! vi2's request groups (slice 13) start it by listening
//! (`a_vi2_request_group_is_ticked_through_the_run_s_turn`, in `hle/tests/graphics/vsync.rs`).
//! Slice 14 (present, `starquake_gsv::shared::present`) needs no server of its own either: the
//! vi2 server answers 228 and 229, and the vsync tick in the timer thread's turn latches the
//! frames (`a_present_reaches_the_latch_on_a_vsync_tick`, in `hle/tests/graphics/present.rs`).
//! The vi2 server paces with the buffer queue; the fence wait fallback
//! (`starquake_gsv::vi2::Pacing::FenceWait`) is a per-run setting no command-line option
//! reaches yet.
//!
//! # The GPU front
//!
//! gsv's servers take the GPU front as one `Arc<dyn GpuFront>` (`starquake_gpu::interface`,
//! "`GpuFront` in parts"), each its part by upcasting. A graphics server has one of two:
//!
//! - **The GPU** ([`Graphics::with_gpu`]): `starquake_gpu::wiring::Gpu`, with its front and back
//!   threads, built here with the `Syncpoints`, gsv's completion queue as its `Completions`, and
//!   the render backend, shader cache, guest memory and write watch the caller gives in
//!   [`GpuParts`] (gpu-protocol, "Graphics server sessions (#11)": "`core` creates the `Gpu`
//!   ... and hands it to gsv"). Always a run ([`FrontMode::Live`]): the GPU completes its work
//!   on its own (`a_run_with_the_gpu_completes_what_waits_on_it`, in `hle/tests/graphics.rs`).
//!   A live run builds it over the recording backend when its guest memory can be shared with
//!   the GPU's threads (`crate::machine`'s `gpu` module, issue #386); render-backend slice R12
//!   adds the choice of the wgpu backend.
//! - **The null front** ([`NullFront`], [`Graphics::new`]): for a run with no GPU back end, and
//!   for every replay (`crate::machine`'s `gpu` module says when a run has neither).
//!
//! [`NullFront`] runs nothing. It does what the spec asks of the front's order and completion
//! with no GPU behind it, so that a title waiting on the GPU is not stuck:
//!
//! - an ordered update with a ticket, and a barrier, are done at once: it pushes
//!   `GpuOrdered { ticket }` to the queue (spec: "Signals with no host work behind them happen at
//!   once");
//! - a channel is a number, counting up from [`FIRST_CHANNEL`], with the syncpoint it was made
//!   with; a class is bound to any channel it made;
//! - a submission's commands are dropped, and its increments are performed at once on the
//!   channel's syncpoint (`Syncpoints::increment_by`: one push if the syncpoint is watched,
//!   however many increments), and it answers `Queued`, never `Full`;
//! - a latched frame's present is counted ([`NullFront::presented`]) and dropped: nothing is
//!   shown.
//!
//! Where it departs from what a GPU does, each pinned by the test named, in `graphics/tests.rs`:
//!
//! | Case | A GPU | Here |
//! |---|---|---|
//! | A submission's commands | run | dropped (`a_submission_s_increments_are_performed_at_once`) |
//! | A submission's wait fence (flag bit 0) | waited for before the increments | not waited for: the increments are performed at once whether it is reached or not (`a_submission_s_wait_fence_is_not_waited_for`) |
//! | A full queue | `Full`, then `GpuQueueRoom` | never full (`a_submission_s_increments_are_performed_at_once`) |
//! | A submission on a channel it did not make | the GPU's error | logged as an error, nothing performed, `Queued` (`a_submission_on_an_unknown_channel_does_nothing`) |
//! | A present | shown, after the work before it | counted and dropped (`a_present_reaches_the_latch_on_a_vsync_tick`) |
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
//! (`Shared::tick_replayed_vsync`; below, "Replay"). The frames the vsync tick latches are
//! handed to the front ([`PresentFront::present`]) once the clock's turn has released gsv's
//! lock, in the order the tick latched them. The thread sleeps no later than
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
//! # Frames
//!
//! A **frame** is a present [`Graphics::turn`] hands the front ([`PresentFront::present`]): one
//! for each frame the vsync tick latched, in either front and either mode. [`Graphics::frames`]
//! counts them; it is `summary.json`'s `frames`, and what `--until frames=N` counts
//! ([`Graphics::with_frame_limit`]). It is gsv's count, not the render backend's
//! (`PresentFeedback`): the GPU never has to finish presenting a frame for it to count.
//!
//! **When a frame counts depends on its fence.** gsv latches a present only once its fence has
//! been reached (render-backend, "Presenting a guest framebuffer", step 1). For a present with no
//! fence ([`NO_FENCE`](starquake_gsv::vi2::NO_FENCE)), or one whose fence was already reached
//! when the tick looked, the tick that counts it does not depend on the host: the count, and the
//! guest time at which `--until frames=N` stops a run, are the same with the GPU and with the
//! null front, and in a replay, which latches the same frames on the ticks fed back. For a present
//! whose fence the GPU itself reaches (a syncpoint its work increments), they are not: with the
//! GPU, the tick that counts the frame is the first after the host GPU completes that work,
//! which depends on host timing, while the null front completes work at once. Such a run can
//! count a frame at a later tick with the GPU than with the null front, or than in another live
//! run, and so stop at `--until frames=N` at a different guest time. The frame tests with the GPU
//! (`hle/tests/graphics/present.rs`, `machine/tests/graphics/frames.rs`) present with no fence.
//!
//! With a limit of `N`, a turn hands the front no more frames once `N` have been handed
//! ([`frames_to_hand`]): a turn that latches more than the room left hands the first ones and
//! drops the rest, which gsv has latched all the same (their notes are recorded and the buffers
//! they replace released). [`Graphics::frame_limit_reached`] then says so, and the run's timer
//! thread ends the run (`crate::machine`'s `timer` module). Pinned by
//! `a_frame_limit_hands_the_front_exactly_that_many_presents`, in `hle/tests/graphics/present.rs`.
//!
//! # Telling a stalled run
//!
//! A run ends as an emulator error when every core has parked and nothing could end a wait
//! (`crate::machine`'s `WAITING`). The kernel's own check (`Dispatcher::is_stalled`) does not
//! know that the timer thread's turn signals gsv's events, so the core loop asks
//! [`Graphics::is_idle`] first: a run is stalled only if the graphics server is idle too
//! (nothing queued, nothing held by the shared page, no page signal waiting, no vsync tick to
//! come: in a run the clock stopped, whatever a restored snapshot left fed back, which a live
//! turn never delivers; in a replay no tick fed back) and then the kernel says so. With
//! [`NullFront`] nothing is in flight on a GPU, so that is everything the turn could still
//! signal. With the GPU, the GPU must be idle too (`Gpu::is_idle`: nothing queued for its front,
//! every entry applied, no list in flight or open, and, while a channel is parked on an acquire,
//! a poll run after the question began that found it still parked, which it waits for at most
//! `starquake_gpu::wiring::IDLE_POLL_WAIT`), asked before gsv's queue, so
//! that a completion the GPU pushes while the question is asked is either seen as work in flight
//! or found queued. That order is by construction: no test can force the race, so none pins it
//! (`a_graphics_server_with_a_busy_gpu_is_not_idle` pins only that a busy GPU makes the server
//! busy).
//!
//! The core loop asks [`Graphics::idleness`], which is [`Graphics::is_idle`] with the busy
//! answer split in two: [`Idleness::GpuBusy`] when everything but the GPU is idle and the GPU
//! answered busy, [`Idleness::Busy`] when anything else is busy. The rest is asked whatever the
//! GPU answered, so that a run stalled but for the GPU's answer can be asked again at the GPU's
//! next quiet signal (`crate::machine`'s `cores` module, "Telling a stalled run"; gpu-protocol,
//! "Idleness"). The null front never answers `GpuBusy` (`the_null_front_is_never_gpu_busy`);
//! the split itself is pinned by `idleness_is_gpu_busy_only_when_the_gpu_alone_is_busy`, and,
//! with the GPU, by `a_graphics_server_with_a_busy_gpu_is_not_idle` (a stopped GPU alone reads
//! `GpuBusy`) and `a_graphics_server_over_an_idle_gpu_becomes_idle_and_a_queued_completion_reads_busy`
//! (an idle GPU's server becomes `Idle`, and a queued completion reads `Busy` whatever the GPU
//! answers).
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
//! | `FrameLatched`, `BufferReleased` | refused ([`ReplayError::NotAnInput`]): what the vsync tick's latch writes, from the ticks and the drained syncpoint values fed back above (`starquake_gsv::shared::present`, "What is recorded") |
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
//!   state, its GPU memory, the syncpoints and the five servers, when the notes are fed back at
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
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use starquake_gpu::back::exec::ShaderCache;
use starquake_gpu::back::fence::FenceClock;
use starquake_gpu::channel::ChannelError;
use starquake_gpu::front::HostClock;
use starquake_gpu::interface::{
    ChannelId, Completions, GpuCompletion, GpuFront, PresentRequest, SyncpointId, Ticket, VmUpdate,
};
use starquake_gpu::syncpoint::Syncpoints;
use starquake_gpu::vm::GpuVm;
use starquake_gpu::wiring::{Gpu, GpuConfig, GpuDeps, GpuRing, WiringError};
use starquake_gsv::completion::{Completion, CompletionQueue, Wake};
use starquake_gsv::gpu::{self as gsv_gpu, AsId, GpuVmFront};
use starquake_gsv::note::Note;
use starquake_gsv::nvsrv::gpu::{ChannelConfig, ChannelFront, GpuError, Submission, Submitted};
use starquake_gsv::nvsrv::{self, NvsrvFamily};
use starquake_gsv::root::{self, RootState};
use starquake_gsv::shared::present::PresentFront;
use starquake_gsv::shared::vsync::VsyncReplayError;
use starquake_gsv::shared::{Shared, TimerHost};
use starquake_gsv::syncpt;
use starquake_gsv::vi2;
use starquake_gsv::video;
use starquake_kernel::dispatch::{DeviceShareError, Dispatcher, WakeError};
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::result::{KernelError, SvcError, SvcResult};
use starquake_memory::{GuestMemory, WriteWatch};
use starquake_render::Backend;
use starquake_services::framework::{Halt, Notify, UnknownPolicy};
use starquake_trace::{Env, EnvError, EventKind, HostNanos, Recorder, ThreadId};

use super::{
    Deliveries, Delivering, GSV_GPU, GSV_NVSRV, GSV_ROOT, GSV_SYNCPT, GSV_VI2, GSV_VIDEO,
    ServicesError,
};

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
/// [`GpuVmFront`], [`ChannelFront`] and [`PresentFront`].
pub struct NullFront {
    mode: FrontMode,
    syncpoints: Arc<Syncpoints>,
    queue: Arc<CompletionQueue>,
    channels: Mutex<Channels>,
    presented: AtomicU64,
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
            presented: AtomicU64::new(0),
        }
    }

    /// How many presents it was handed.
    #[must_use]
    pub fn presented(&self) -> u64 {
        self.presented.load(Ordering::Relaxed)
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

impl PresentFront for NullFront {
    /// Counts the present and drops it: there is nothing to show it on.
    fn present(&self, request: PresentRequest) {
        tracing::trace!(?request, "the null GPU front was handed a present");
        self.presented.fetch_add(1, Ordering::Relaxed);
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

impl GpuFront for NullFront {
    /// Never: it reads no stream, so none can stop a channel.
    fn channel_error(&self, _channel: ChannelId) -> Option<ChannelError> {
        None
    }
}

/// What a run gives [`Graphics::with_gpu`] to build the GPU with (module docs, "The GPU front").
pub struct GpuParts {
    /// The ring between the GPU's threads, made first so `backend` is built with its completion
    /// wake ([`GpuRing::completion_wake`]).
    pub ring: GpuRing,
    /// The render backend.
    pub backend: Arc<dyn Backend>,
    /// The shader cache behind the GPU's draw path (its seam, `ShaderCache`).
    pub shaders: Box<dyn ShaderCache + Send>,
    /// The title's memory.
    pub guest: Arc<dyn GuestMemory>,
    /// The title's write watch, shared with the kernel.
    pub watch: Arc<WriteWatch>,
    /// The GPU's settings.
    pub config: GpuConfig,
    /// The GPU back thread's clock, whose timestamp a 16-byte semaphore release writes: in a
    /// run, guest time from the kernel's clock (`crate::machine`'s `gpu` module, "The GPU's
    /// clocks"). The front thread's clock is the host's.
    pub fence_clock: Arc<dyn FenceClock>,
}

/// What [`Graphics::idleness`] found (module docs, "Telling a stalled run").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Idleness {
    /// Nothing left could signal an event without a guest's call first: what
    /// [`Graphics::is_idle`] calls idle.
    Idle,
    /// Everything but the GPU is idle, and the GPU answered busy: a question that can be asked
    /// again when the GPU next reports quiet (gpu-protocol, "Idleness").
    GpuBusy,
    /// Something besides the GPU is busy; the GPU may be busy too.
    Busy,
}

/// The graphics server's [`Idleness`] from whether the GPU answered busy (`false` with no GPU)
/// and whether everything else is idle.
#[must_use]
pub(crate) const fn idleness(gpu_busy: bool, rest_idle: bool) -> Idleness {
    match (gpu_busy, rest_idle) {
        (false, true) => Idleness::Idle,
        (true, true) => Idleness::GpuBusy,
        (_, false) => Idleness::Busy,
    }
}

/// Why a note could not be fed back ([`Graphics::replay`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReplayError {
    /// The note is one a page turn or the vsync tick's latch writes, not an input the run took
    /// from outside.
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
    /// The front gsv's servers are given.
    front: Arc<dyn GpuFront>,
    /// The front, when it is the null front.
    null: Option<Arc<NullFront>>,
    /// The front, when it is the GPU.
    gpu: Option<Arc<Gpu>>,
    /// Whether the run's own inputs complete things, or only recorded notes: the front's mode,
    /// which also picks the vsync clock's turn (module docs, "The timer thread's turn").
    mode: FrontMode,
    /// Frames handed to the front (module docs, "Frames").
    frames: AtomicU64,
    /// The most frames the front is handed (module docs, "Frames").
    frame_limit: Option<NonZeroU64>,
}

impl Graphics {
    /// The queue, rung through `wake` if given, the syncpoints, gsv's shared state recording for
    /// [`GSV_ROOT`], and a [`NullFront`] in `mode` (module docs, "What is built").
    #[must_use]
    pub fn new(mode: FrontMode, wake: Option<Arc<dyn Wake>>) -> Self {
        let (queue, syncpoints) = Self::queue_and_syncpoints(wake);
        let null = Arc::new(NullFront::new(
            mode,
            Arc::clone(&syncpoints),
            Arc::clone(&queue),
        ));
        let shared = Arc::new(Shared::new(syncpoints, queue, GSV_ROOT));
        Self {
            shared,
            front: Arc::<NullFront>::clone(&null),
            null: Some(null),
            gpu: None,
            mode,
            frames: AtomicU64::new(0),
            frame_limit: None,
        }
    }

    /// The queue, rung through `wake` if given, the syncpoints, gsv's shared state recording for
    /// [`GSV_ROOT`], and the GPU over them and `parts`, in a run (module docs, "The GPU front").
    ///
    /// # Errors
    ///
    /// [`WiringError::Spawn`] when the GPU's threads cannot be started.
    pub fn with_gpu(wake: Option<Arc<dyn Wake>>, parts: GpuParts) -> Result<Self, WiringError> {
        let (queue, syncpoints) = Self::queue_and_syncpoints(wake);
        let GpuParts {
            ring,
            backend,
            shaders,
            guest,
            watch,
            config,
            fence_clock,
        } = parts;
        let gpu = Arc::new(Gpu::new(
            GpuDeps {
                ring,
                syncpoints: Arc::clone(&syncpoints),
                completions: Arc::<CompletionQueue>::clone(&queue),
                backend,
                shaders,
                guest,
                watch,
                front_clock: Arc::new(HostClock::new()),
                fence_clock,
            },
            config,
        )?);
        let shared = Arc::new(Shared::new(syncpoints, queue, GSV_ROOT));
        Ok(Self {
            shared,
            front: Arc::<Gpu>::clone(&gpu),
            null: None,
            gpu: Some(gpu),
            mode: FrontMode::Live,
            frames: AtomicU64::new(0),
            frame_limit: None,
        })
    }

    /// This graphics server, handing the front no more than `limit` frames (module docs,
    /// "Frames"); `None` hands it every frame.
    #[must_use]
    pub fn with_frame_limit(mut self, limit: Option<NonZeroU64>) -> Self {
        self.frame_limit = limit;
        self
    }

    /// Frames presented: the presents [`Graphics::turn`] handed the front (module docs,
    /// "Frames").
    #[must_use]
    pub fn frames(&self) -> u64 {
        self.frames.load(Ordering::SeqCst)
    }

    /// Whether the front has been handed as many frames as the limit allows: never without a
    /// limit (module docs, "Frames").
    #[must_use]
    pub fn frame_limit_reached(&self) -> bool {
        self.frame_limit
            .is_some_and(|limit| self.frames() >= limit.get())
    }

    /// The completion queue, rung through `wake` if given, and the syncpoints that push to it.
    fn queue_and_syncpoints(
        wake: Option<Arc<dyn Wake>>,
    ) -> (Arc<CompletionQueue>, Arc<Syncpoints>) {
        let mut queue = CompletionQueue::new();
        if let Some(wake) = wake {
            queue = queue.with_wake(wake);
        }
        let queue = Arc::new(queue);
        let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
        let syncpoints = Arc::new(Syncpoints::new(completions));
        (queue, syncpoints)
    }

    /// gsv's shared state, with its syncpoints and queue.
    #[must_use]
    pub const fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }

    /// The GPU front gsv's servers are given.
    #[must_use]
    pub const fn front(&self) -> &Arc<dyn GpuFront> {
        &self.front
    }

    /// The null front, when the front is the null front.
    #[must_use]
    pub const fn null_front(&self) -> Option<&Arc<NullFront>> {
        self.null.as_ref()
    }

    /// The GPU, when the front is the GPU: its counters (`summary.json`'s, gpu-protocol
    /// "Observability") and its flight recorder (`gpu.txt`).
    #[must_use]
    pub const fn gpu(&self) -> Option<&Arc<Gpu>> {
        self.gpu.as_ref()
    }

    /// Puts gsv's servers in front of `dispatcher` (module docs, "What is built"), telling
    /// `ready` of nvsrv's deferred replies. The vi2 server reads `env`, the services'
    /// environment, and tells `halt` when it fails: a request group that starts listening to the
    /// vsync clock reads its clock (`starquake_gsv::vi2`, "The server"). Every server the root
    /// names is added here, so the root's 305 never hands out a session on a number with no
    /// server.
    pub(super) fn add_servers(
        &self,
        dispatcher: &Dispatcher,
        policy: UnknownPolicy,
        ready: Arc<dyn Deliveries>,
        env: Arc<dyn Env>,
        halt: Arc<dyn Halt>,
    ) -> Result<(), ServicesError> {
        let syncpt =
            syncpt::syncpt_server(policy, Arc::clone(&self.shared)).with_service_id(GSV_SYNCPT);
        dispatcher.add_server(syncpt::SYNCPT_NAME, GSV_SYNCPT, None, Box::new(syncpt))?;

        let vm: Arc<dyn GpuVmFront> = Arc::<dyn GpuFront>::clone(&self.front);
        let gpu = gsv_gpu::gpu_server(Arc::clone(&self.shared), Arc::clone(&vm), policy)
            .with_service_id(GSV_GPU);
        dispatcher.add_server(gsv_gpu::GPU_NAME, GSV_GPU, None, Box::new(gpu))?;

        let channels: Arc<dyn ChannelFront> = Arc::<dyn GpuFront>::clone(&self.front);
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

        let display = vi2::vi2_server(policy, Arc::clone(&self.shared))
            .with_service_id(GSV_VI2)
            .with_env(env, halt);
        dispatcher.add_server(vi2::VI2_NAME, GSV_VI2, None, Box::new(display))?;

        let codecs =
            video::video_server(Arc::clone(&self.shared), policy).with_service_id(GSV_VIDEO);
        dispatcher.add_server(video::VIDEO_NAME, GSV_VIDEO, None, Box::new(codecs))?;

        let state = RootState::new()
            .with_syncpt_server(GSV_SYNCPT)
            .with_gpu_server(GSV_GPU)
            .with_nvsrv_server(GSV_NVSRV)
            .with_vi2_server(GSV_VI2)
            .with_video_server(GSV_VIDEO);
        let graphics_root = root::root_server_with_page(state, policy, Arc::clone(&self.shared))
            .with_service_id(GSV_ROOT);
        dispatcher.add_server(root::ROOT_NAME, GSV_ROOT, None, Box::new(graphics_root))?;
        Ok(())
    }

    /// The timer thread's turn (module docs, "The timer thread's turn"): drains the queue
    /// through `host`, then runs the shared page's turn at `now`, then the vsync clock's turn for
    /// the front's mode, and hands the frames it latched to the front. Returns what the drain
    /// took, in order.
    pub fn turn(&self, host: &mut dyn TimerHost, now: HostNanos) -> Vec<Completion> {
        let taken = self.shared.drain(host);
        self.shared.tick_page(host, now);
        let presents = match self.mode {
            FrontMode::Live => self.shared.tick_vsync(host, now),
            FrontMode::Replay => self.shared.tick_replayed_vsync(host),
        };
        let latched = presents.len();
        let handed = frames_to_hand(self.frames(), self.frame_limit, latched);
        if handed < latched {
            tracing::debug!(
                dropped = latched - handed,
                "frames latched past the run's frame limit are not presented"
            );
        }
        for request in presents.into_iter().take(handed) {
            self.front.present(request);
            self.frames.fetch_add(1, Ordering::SeqCst);
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
    /// so it waits for a drain in progress, whose signals are then made. It does so even when
    /// the GPU answered busy ([`Graphics::idleness`]): before the GPU's answer was told apart, a
    /// busy GPU returned at once, without the lock.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.idleness() == Idleness::Idle
    }

    /// [`Graphics::is_idle`], saying whether a busy answer came from the GPU alone (module docs,
    /// "Telling a stalled run"): the GPU is asked first, then the rest is asked whatever the
    /// GPU answered. So gsv's shared lock is taken, and a drain in progress waited for, even
    /// when the GPU answered busy.
    #[must_use]
    pub fn idleness(&self) -> Idleness {
        // The GPU first: a completion it pushes while this asks is then seen as work in flight
        // or found queued below (module docs, "Telling a stalled run").
        let gpu_busy = self.gpu.as_ref().is_some_and(|gpu| !gpu.is_idle());
        let ticking = match self.mode {
            FrontMode::Live => self.shared.vsync_deadline().is_some(),
            FrontMode::Replay => false,
        };
        idleness(gpu_busy, !ticking && self.is_idle_but_the_clock())
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
    /// signalled at the shared page, and, in a replay, no tick fed back. In a run a tick fed
    /// back (only a restored snapshot holds one) is not work: the live turn never delivers it
    /// (`a_tick_fed_back_in_a_live_run_is_not_work`).
    fn is_idle_but_the_clock(&self) -> bool {
        let locked = self.shared.lock();
        let page = locked.page();
        let fed_back = match self.mode {
            FrontMode::Live => false,
            FrontMode::Replay => locked.vsync().replayed() > 0,
        };
        let idle = page.held().is_none()
            && !page.is_signalled()
            && !fed_back
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
            Note::SyncptPageAnswered { .. }
            | Note::SyncptPageAcknowledged { .. }
            | Note::FrameLatched { .. }
            | Note::BufferReleased { .. } => {
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

/// How many of `latched` frames a turn hands the front when `handed` were handed before and the
/// run allows `limit` (module docs, "Frames"): every one without a limit, and otherwise no more
/// than the limit leaves room for.
#[must_use]
pub fn frames_to_hand(handed: u64, limit: Option<NonZeroU64>, latched: usize) -> usize {
    let Some(limit) = limit else {
        return latched;
    };
    let room = limit.get().saturating_sub(handed);
    // A room that does not fit a `usize` is more than any turn latches.
    usize::try_from(room).map_or(latched, |room| room.min(latched))
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
