//! The GPU's front thread: the loop that serves every channel (gpu-protocol spec, slice G14:
//! "Threads and data flow", "Channels and submissions", "Semaphore acquires", "Ordered
//! address-table updates and barriers", "The ring").
//!
//! The pieces:
//!
//! - [`FrontShared`] (in `shared`) is what gsv's threads and the back thread share with the
//!   front: the requests queued in call order, each channel's queue bound (65,536 GP entries,
//!   [`crate::interface::CHANNEL_QUEUE_ENTRIES`]; `Full` and `GpuQueueRoom`), the back's reports
//!   ([`crate::back::fence::FrontWake`]) and the front's wake-up.
//! - [`Front`] is the front thread's state: the address spaces, each with its process's macro
//!   code ([`ProcessMacros`]), the channels (in `lane`: pusher, macro state, register files,
//!   submission queue and what the channel waits on), the pending ordered updates, the ring and
//!   the acquire poll. [`Front::turn`] is one pass of the loop; [`Front::run`] loops it on the
//!   front thread, parking between passes, and [`spawn`] starts that thread.
//! - `ring_sink` adapts the ring's producer to the register files' [`crate::regs::EntrySink`]
//!   and answers the macro host's [`crate::macros::BackWork`].
//! - [`OrderedTable`] (in `ordered`) hands the back's executor the ordered update each marker
//!   carries.
//! - `poll` holds the acquire poll's back-off ([`Backoff`]) and the injected [`FrontClock`].
//!
//! Structure from `eden-nx2/src/video_core/gpu_split.cpp` @ 50fa86bb (GPL-3.0-or-later: one
//! front producing into one ring, publishing in batches and before it waits), and the waiter
//! table's use from `Nx2SemaphoreWaiters` in `eden-nx2/src/video_core/nx2.h` @ 50fa86bb (register,
//! then read again; resume a latched waiter). Eden-NX2 has no front loop of its own: its GPU
//! thread runs one channel's pushbuffer at a time and drains the back on every acquire (row
//! 142), which this design replaces. Paths are relative to the reference folder
//! (`tools/refs-path.sh`).
//!
//! # One turn
//!
//! [`Front::turn`] does, in this order:
//!
//! 1. Takes what the callers queued and the back reported since the last turn
//!    ([`FrontShared::take`]). Markers the back reported end `SYNC_WAIT`s and catch-ups (the
//!    waiting macro call runs again then).
//! 2. Applies the requests in call order: address spaces and channels are made, submissions go
//!    into their channel's queue, updates and barriers record their channels' positions, and
//!    destroyed channels are dropped (their waiter removed, their process's upload record of
//!    them forgotten, [`ProcessMacros::forget_channel`]).
//! 3. Rechecks every waiting channel if the back reported a release or an increment, a
//!    submission arrived, or the poll is due (spec, "Semaphore acquires", step 3). A latched
//!    acquire resumes at once, even if the word has moved on (row 70).
//! 4. Emits every ordered update that is ready (`ordered`).
//! 5. Runs every runnable channel once, round robin from the channel after the one that ran
//!    first in the previous turn, each for up to [`QUANTUM`] words.
//! 6. Emits ordered updates again (channels may have finished the submissions they waited for),
//!    and tells the queue bound how many entries each channel started, pushing
//!    [`GpuCompletion::GpuQueueRoom`] where a refused submission now fits.
//!
//! The turn returns [`Turn::Worked`] if anything changed, and the loop turns again at once.
//! Otherwise it publishes the ring and returns [`Turn::Idle`] with how long the front may sleep:
//! the poll's next step ([`Backoff::next_sleep`]) while a channel waits on an acquire or a fence, and
//! [`IDLE_PARK`] (a safety net only) when none does. A submission or a back report ends the sleep
//! at once.
//!
//! **The poll** is due when the clock has reached the time set when it was last scheduled; it is
//! scheduled (and the clock read) only while a channel waits on an acquire or a fence, so an idle
//! front reads no clock. The schedule restarts when a channel newly waits on an acquire or a
//! fence (`a_channel_that_newly_parks_restarts_the_polls_schedule`), when a recheck lets a waiting
//! channel go on (`a_channel_that_resumes_restarts_the_polls_schedule_for_the_others`), and when
//! no channel is left waiting. A back report of a release or an increment rechecks every waiting
//! channel at once, without waiting for the poll
//! (`a_release_the_back_writes_rechecks_a_parked_channel_at_once`,
//! `a_fence_wait_holds_the_submission_until_an_increment_reaches_it`). While other channels keep the front busy, the poll still fires on time, since
//! each turn looks at the clock while anything waits (`the_poll_fires_while_other_channels_keep_the_front_busy`).
//!
//! # Fairness
//!
//! The quantum is [`QUANTUM`] words (the spec's 4,096, a **guess**: it affects only fairness,
//! never correctness, since channels order themselves through semaphores). A channel that waits
//! (on an acquire, a fence, a `SYNC_WAIT` or a catch-up) emits nothing and is skipped, so it
//! never blocks another (`a_parked_channel_does_not_block_another`).
//!
//! # What stops a channel, and what stops the front
//!
//! A structural error in a channel's stream ([`ChannelError`]) stops that channel for good, as
//! hardware raises a channel error (decision 13 (c)): it is recorded for
//! [`FrontShared::channel_error`], its waiter is removed, its unstarted entries leave its queue
//! bound, the active submission's unstarted entries included
//! (`a_channel_stopped_inside_a_submission_releases_its_unstarted_entries`,
//! `a_full_queue_stopped_at_its_first_entry_reports_room_once`), and it no longer holds ordered
//! updates (none of its later work will run). A failed push to the ring stops the front: every
//! turn from then on returns [`FrontError::Ring`] with the first failure (`ring_sink`).
//!
//! **GP entries are checked when they start, not at submission.** [`FrontShared::submit`] looks
//! at nothing in a submission's entries; the 40-bit range check
//! ([`ChannelError::EntryPastAddressSpace`]) runs when the front starts the entry
//! (`crate::channel::Channel::start_entry`) and stops the channel
//! (`an_entry_past_the_address_space_stops_the_channel_when_it_starts`). The reasons: Eden-NX2's
//! `SubmitGPFIFOImpl` (`eden-nx2/src/core/hle/service/nvdrv/devices/nvhost_gpu.cpp` @ 50fa86bb)
//! checks no entry's address and pushes the entries as given, so refusing at submission would
//! answer an ioctl with an error no reference sends; and a bad entry is a structural error of the
//! channel's stream, which decision 13 (c) answers by stopping the channel, as hardware raises a
//! channel error when it fetches the entry. The alignment check
//! ([`ChannelError::UnalignedEntry`]) cannot fire through the front at all: a submission carries
//! raw entries, whose address is `GP_ENTRY0_GET` (bits 31:2) and `GP_ENTRY1_GET_HI`, so bits 1:0
//! of word 0 never reach the address (`crate::fifo::GpEntry::from_raw`;
//! `a_raw_entry_with_its_low_address_bits_set_is_not_unaligned`). Only a hand-built
//! `GpEntry` given straight to a `Channel` can be unaligned.
//!
//! # What the wiring adds
//!
//! The wiring (slice G21, `crate::wiring`) implements `GpuFront` over [`FrontShared`] and runs
//! this loop on the front thread with [`Front::run_observed`], which hands it the front after
//! every turn for its counters. Three hooks serve it:
//!
//! - **Presents** ([`FrontShared::present`]) wait like a barrier and are pushed as `Present`
//!   entries once every submission queued before them has been read (`ordered`, "Presents").
//! - **A destroyed channel is reported** to [`ChannelRetired`] ([`Front::with_retired`]) when
//!   the front drops it, with how many ring entries it had pushed by then ([`Front::sent`]): the
//!   back forgets the channel once it has taken that many, so every entry the channel made
//!   reaches the back before it is forgotten.
//! - **Idleness**: an idle turn marks [`FrontShared::is_quiet`] (in [`Front::run_observed`]).
//!
//! # Not here
//!
//! - Snapshotting the front's state.

mod lane;
mod ordered;
mod poll;
mod ring_sink;
mod shared;
mod thread;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;

use crate::back::fence::{OrderedUpdate, VmChange, WaiterTable};
use crate::channel::ChannelError;
use crate::interface::{AsId, ChannelId, Completions, GpuCompletion, charge};
use crate::macros::ProcessMacros;
use crate::memory::GpuMemory;
use crate::ring::{Producer, PushError, RingEntry};
use crate::syncpoint::Syncpoints;
use crate::vm::GpuVm;

pub use lane::MAX_OTHER_FILES;
pub use ordered::OrderedTable;
use ordered::Emit;
pub use poll::{
    Backoff, FrontClock, HostClock, IDLE_PARK, POLL_FIRST_SLEEP, POLL_LONGEST_SLEEP, POLL_SPINS,
};
pub use shared::{CreateError, FrontShared, fits};
pub use thread::{FrontThread, spawn, spawn_observed};

use lane::{Lane, LaneCx, Wait};
use ordered::Pending;
use ring_sink::RingSink;
use shared::{BackReports, Request};

/// Words one channel runs before the front moves on to the next (the spec's 4,096, a **guess**;
/// module docs, "Fairness").
pub const QUANTUM: u32 = 4096;

/// Why the front stopped.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum FrontError {
    /// A push to the ring failed (`ring_sink`); the front has stopped.
    #[error("the GPU front stopped: the ring refused an entry: {0}")]
    Ring(PushError),
    /// The front thread panicked.
    #[error("the GPU front thread panicked")]
    Panicked,
}

/// What a turn of the loop ended with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Turn {
    /// Something changed; turn again at once.
    Worked,
    /// Nothing to do: the front may sleep this long (zero: spin and look again) unless woken.
    Idle {
        /// How long.
        park: Duration,
    },
    /// [`FrontShared::stop`] was called.
    Stopped,
}

/// The front's counters (gpu-protocol, "Observability").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrontCounters {
    /// Command words executed.
    pub words: u64,
    /// Submissions finished (every entry read, every increment pushed).
    pub submissions: u64,
    /// Syncpoint increments pushed for submissions.
    pub increments: u64,
    /// Acquires that parked a channel (met neither at the first read nor at the second).
    pub parks: u64,
    /// Submission fences that made a channel wait.
    pub fence_waits: u64,
    /// `SYNC_WAIT` entries that made a channel wait.
    pub sync_waits: u64,
    /// Macro calls that waited for a catch-up.
    pub catch_ups: u64,
    /// Times the poll fired.
    pub polls: u64,
    /// Ordered updates and barriers emitted.
    pub ordered: u64,
    /// Macro calls that failed and committed nothing.
    pub macro_failures: u64,
    /// Channels stopped by a structural error.
    pub stopped_channels: u64,
    /// Presents pushed (`ordered`, "Presents").
    pub presents: u64,
}

/// Told when the front drops a destroyed channel (module docs, "What the wiring adds").
pub trait ChannelRetired: Send + Sync {
    /// The front dropped `channel` after pushing `after` ring entries ([`Front::sent`]); it
    /// pushes nothing of the channel from then on.
    fn retired(&self, channel: ChannelId, after: u64);
}

/// What the front is made with.
pub struct FrontDeps {
    /// What the callers and the back share with the front.
    pub shared: Arc<FrontShared>,
    /// The front's end of the ring.
    pub ring: Producer,
    /// The syncpoint counters fence waits read.
    pub syncpoints: Arc<Syncpoints>,
    /// gsv's completion queue, for [`GpuCompletion::GpuQueueRoom`].
    pub completions: Arc<dyn Completions>,
    /// The acquire waiter table, shared with the back.
    pub waiters: Arc<WaiterTable>,
    /// Where ordered updates wait for the back's executor.
    pub ordered: Arc<OrderedTable>,
    /// The front's clock.
    pub clock: Arc<dyn FrontClock>,
}

/// One address space: its table, the memory the front reads through, its process's macros and
/// its channels.
struct Space {
    vm: Arc<GpuVm>,
    memory: Arc<dyn GpuMemory>,
    macros: ProcessMacros,
}

/// The front thread's state (module docs).
pub struct Front {
    shared: Arc<FrontShared>,
    ring: RingSink,
    syncpoints: Arc<Syncpoints>,
    completions: Arc<dyn Completions>,
    waiters: Arc<WaiterTable>,
    ordered: Arc<OrderedTable>,
    clock: Arc<dyn FrontClock>,
    spaces: BTreeMap<AsId, Space>,
    lanes: BTreeMap<ChannelId, Lane>,
    pending: VecDeque<Pending>,
    /// The channel the next turn starts after.
    cursor: Option<ChannelId>,
    backoff: Backoff,
    /// When the poll is next due, while a channel waits on an acquire or a fence.
    poll_due: Option<Duration>,
    counters: FrontCounters,
    retired: Option<Arc<dyn ChannelRetired>>,
}

impl Front {
    /// The front, with no address space and no channel.
    pub fn new(deps: FrontDeps) -> Self {
        Self {
            shared: deps.shared,
            ring: RingSink::new(deps.ring),
            syncpoints: deps.syncpoints,
            completions: deps.completions,
            waiters: deps.waiters,
            ordered: deps.ordered,
            clock: deps.clock,
            spaces: BTreeMap::new(),
            lanes: BTreeMap::new(),
            pending: VecDeque::new(),
            cursor: None,
            backoff: Backoff::new(),
            poll_due: None,
            counters: FrontCounters::default(),
            retired: None,
        }
    }

    /// The front, telling `retired` of every channel it drops (module docs, "What the wiring
    /// adds").
    #[must_use]
    pub fn with_retired(mut self, retired: Arc<dyn ChannelRetired>) -> Self {
        self.retired = Some(retired);
        self
    }

    /// Ring entries of every kind pushed so far, markers included: the back takes the
    /// `n`-th entry (from 0) as the one pushed when this was `n`.
    pub fn sent(&self) -> u64 {
        self.ring.sent()
    }

    /// The counters so far.
    pub fn counters(&self) -> FrontCounters {
        self.counters
    }

    /// What the callers share with this front.
    pub fn shared(&self) -> &Arc<FrontShared> {
        &self.shared
    }

    /// One pass of the loop (module docs, "One turn").
    ///
    /// # Errors
    ///
    /// [`FrontError::Ring`] once a push to the ring has failed, on this turn and every later one.
    pub fn turn(&mut self) -> Result<Turn, FrontError> {
        if let Some(error) = self.ring.error() {
            return Err(FrontError::Ring(error));
        }
        let taken = self.shared.take();
        if taken.stopping {
            self.ring.publish();
            return Ok(Turn::Stopped);
        }
        let mut progress = false;
        let mut recheck = self.reports(&taken.reports, &mut progress);
        for request in taken.requests {
            recheck |= self.apply(request);
            progress = true;
        }
        if self.poll_is_due() {
            self.counters.polls = self.counters.polls.saturating_add(1);
            self.poll_due = None;
            recheck = true;
        }
        if recheck {
            progress |= self.recheck();
        }
        progress |= self.emit_ready();
        progress |= self.run_channels();
        progress |= self.emit_ready();
        self.release_entries();
        if let Some(error) = self.ring.error() {
            return Err(FrontError::Ring(error));
        }
        self.schedule_poll();
        if progress {
            return Ok(Turn::Worked);
        }
        self.ring.publish();
        let park = self
            .poll_due
            .map_or(IDLE_PARK, |due| due.saturating_sub(self.clock.now()));
        Ok(Turn::Idle { park })
    }

    /// Loops [`Front::turn`] until [`FrontShared::stop`], parking between idle turns: the front
    /// thread's body ([`spawn`]).
    ///
    /// # Errors
    ///
    /// The turn's [`FrontError`].
    pub fn run(self) -> Result<(), FrontError> {
        self.run_observed(|_| {})
    }

    /// [`Front::run`], calling `observe` with the front after every turn, and marking the
    /// shared state quiet ([`FrontShared::is_quiet`]) after each idle turn, after `observe`.
    ///
    /// # Errors
    ///
    /// The turn's [`FrontError`].
    pub fn run_observed(mut self, mut observe: impl FnMut(&Self)) -> Result<(), FrontError> {
        loop {
            let turn = self.turn();
            observe(&self);
            match turn? {
                Turn::Worked => {}
                Turn::Stopped => return Ok(()),
                Turn::Idle { park } => {
                    self.shared.mark_idle();
                    if park.is_zero() {
                        std::hint::spin_loop();
                    } else {
                        self.shared.park(park);
                    }
                }
            }
        }
    }

    /// Takes the back's reports (module docs, "One turn", step 1); whether waiting channels are
    /// rechecked.
    fn reports(&mut self, reports: &BackReports, progress: &mut bool) -> bool {
        for &token in &reports.markers {
            self.ring.reported(token);
        }
        if !reports.markers.is_empty() {
            let Self {
                ring,
                spaces,
                lanes,
                syncpoints,
                waiters,
                counters,
                ..
            } = self;
            for lane in lanes.values_mut() {
                let Some(space) = spaces.get_mut(&lane.config.space) else {
                    continue;
                };
                let mut cx = LaneCx {
                    ring,
                    process: &mut space.macros,
                    memory: &*space.memory,
                    syncpoints,
                    waiters,
                    counters,
                };
                *progress |= lane.reported(&mut cx);
            }
        }
        reports.released || reports.incremented
    }

    /// Applies one request (module docs, "One turn", step 2); whether waiting channels are
    /// rechecked (a submission arrived).
    fn apply(&mut self, request: Request) -> bool {
        match request {
            Request::CreateSpace { space, vm, memory } => {
                self.spaces.entry(space).or_insert_with(|| Space {
                    vm,
                    memory,
                    macros: ProcessMacros::new(),
                });
                false
            }
            Request::CreateChannel { id, config } => {
                self.lanes.insert(id, Lane::new(id, config));
                false
            }
            Request::Submit {
                channel,
                submission,
            } => {
                if let Some(lane) = self.lanes.get_mut(&channel) {
                    if lane.stopped {
                        lane.started = lane
                            .started
                            .saturating_add(charge(submission.entries.len()));
                    } else {
                        lane.take(submission);
                    }
                }
                true
            }
            Request::Update {
                space,
                update,
                ticket,
            } => {
                let change = self.spaces.get(&space).map(|known| VmChange {
                    space,
                    vm: Arc::clone(&known.vm),
                    update,
                });
                let positions = self
                    .lanes
                    .iter()
                    .filter(|(_, lane)| change.is_some() && lane.config.space == space)
                    .map(|(&id, lane)| (id, lane.taken))
                    .collect();
                self.pending.push_back(Pending {
                    what: Emit::Ordered(OrderedUpdate { change, ticket }),
                    positions,
                });
                false
            }
            Request::Barrier { ticket } => {
                let positions = self
                    .lanes
                    .iter()
                    .map(|(&id, lane)| (id, lane.taken))
                    .collect();
                self.pending.push_back(Pending {
                    what: Emit::Ordered(OrderedUpdate {
                        change: None,
                        ticket: Some(ticket),
                    }),
                    positions,
                });
                false
            }
            Request::Present { request } => {
                let positions = self
                    .lanes
                    .iter()
                    .map(|(&id, lane)| (id, lane.taken))
                    .collect();
                self.pending.push_back(Pending {
                    what: Emit::Present(request),
                    positions,
                });
                false
            }
            Request::Destroy { channel } => {
                if let Some(lane) = self.lanes.remove(&channel) {
                    self.waiters.remove(channel);
                    if let Some(space) = self.spaces.get_mut(&lane.config.space) {
                        space.macros.forget_channel(channel);
                    }
                }
                self.shared.forget(channel);
                if let Some(retired) = &self.retired {
                    retired.retired(channel, self.ring.sent());
                }
                false
            }
        }
    }

    /// Whether the poll is due: a channel waits on an acquire or a fence and the clock has
    /// reached the time set for it. Reads the clock only while one waits.
    fn poll_is_due(&self) -> bool {
        self.poll_due.is_some_and(|due| self.clock.now() >= due)
    }

    /// Whether a channel waits on something the poll rechecks: an acquire or a fence.
    fn any_polled_wait(&self) -> bool {
        self.lanes
            .values()
            .any(|lane| matches!(lane.wait, Some(Wait::Acquire(_) | Wait::Fence(_))))
    }

    /// Keeps the poll scheduled exactly while a channel waits on an acquire or a fence (module
    /// docs, "One turn"): when none is scheduled, the next one is due after the schedule's next
    /// step.
    fn schedule_poll(&mut self) {
        if !self.any_polled_wait() {
            self.poll_due = None;
            self.backoff.reset();
            return;
        }
        if self.poll_due.is_none() {
            let step = self.backoff.next_sleep();
            self.poll_due = Some(self.clock.now().saturating_add(step));
        }
    }

    /// Rechecks every waiting channel (module docs, "One turn", step 3); whether one can go on.
    fn recheck(&mut self) -> bool {
        let Self {
            ring,
            spaces,
            lanes,
            syncpoints,
            waiters,
            counters,
            shared,
            ..
        } = self;
        let mut resumed = false;
        for (&id, lane) in lanes.iter_mut() {
            let Some(space) = spaces.get_mut(&lane.config.space) else {
                continue;
            };
            let mut cx = LaneCx {
                ring,
                process: &mut space.macros,
                memory: &*space.memory,
                syncpoints,
                waiters,
                counters,
            };
            match lane.recheck(&mut cx) {
                Ok(true) => resumed = true,
                Ok(false) => {}
                Err(error) => {
                    stop_lane(lane, id, error, shared, waiters, counters);
                    resumed = true;
                }
            }
        }
        if resumed {
            self.backoff.reset();
        }
        resumed
    }

    /// Emits every pending ordered update that is ready, in call order (`ordered`); whether
    /// one was.
    fn emit_ready(&mut self) -> bool {
        let mut emitted = false;
        let mut index = 0;
        while let Some(pending) = self.pending.get(index) {
            let lanes = &self.lanes;
            let ready = pending
                .is_ready(|channel, seq| lanes.get(&channel).is_none_or(|lane| lane.is_past(seq)));
            if !ready {
                index += 1;
                continue;
            }
            if let Some(pending) = self.pending.remove(index) {
                match pending.what {
                    Emit::Ordered(update) => {
                        let token = self.ring.reserve_token();
                        self.ordered.insert(token, update);
                        self.ring.push_marker(token);
                        self.counters.ordered = self.counters.ordered.saturating_add(1);
                    }
                    Emit::Present(request) => {
                        crate::regs::EntrySink::send(&mut self.ring, RingEntry::Present(request));
                        self.counters.presents = self.counters.presents.saturating_add(1);
                    }
                }
                emitted = true;
            }
        }
        emitted
    }

    /// Runs every runnable channel once, round robin (module docs, "One turn", step 5); whether
    /// anything changed.
    fn run_channels(&mut self) -> bool {
        let order: Vec<ChannelId> = match self.cursor {
            Some(cursor) => self
                .lanes
                .range((
                    std::ops::Bound::Excluded(cursor),
                    std::ops::Bound::Unbounded,
                ))
                .chain(self.lanes.range(..=cursor))
                .map(|(&id, _)| id)
                .collect(),
            None => self.lanes.keys().copied().collect(),
        };
        let mut progress = false;
        let mut first = None;
        let Self {
            ring,
            spaces,
            lanes,
            syncpoints,
            waiters,
            counters,
            shared,
            backoff,
            poll_due,
            cursor,
            ..
        } = self;
        for id in order {
            let Some(lane) = lanes.get_mut(&id) else {
                continue;
            };
            if !lane.is_runnable() {
                continue;
            }
            let Some(space) = spaces.get_mut(&lane.config.space) else {
                continue;
            };
            first.get_or_insert(id);
            let was_waiting = lane.wait.is_some();
            let mut cx = LaneCx {
                ring,
                process: &mut space.macros,
                memory: &*space.memory,
                syncpoints,
                waiters,
                counters,
            };
            match lane.run(&mut cx, QUANTUM) {
                Ok(ran) => progress |= ran,
                Err(error) => {
                    stop_lane(lane, id, error, shared, waiters, counters);
                    progress = true;
                }
            }
            if !was_waiting && matches!(lane.wait, Some(Wait::Acquire(_) | Wait::Fence(_))) {
                // A channel newly waits: the poll's schedule starts again.
                backoff.reset();
                *poll_due = None;
            }
            if ring.error().is_some() {
                break;
            }
        }
        if first.is_some() {
            *cursor = first;
        }
        progress
    }

    /// Tells the queue bound how many entries each channel started or dropped, and pushes
    /// [`GpuCompletion::GpuQueueRoom`] where a refused submission now fits.
    fn release_entries(&mut self) {
        for (&id, lane) in &mut self.lanes {
            if lane.started == 0 {
                continue;
            }
            let started = std::mem::take(&mut lane.started);
            if self.shared.release_entries(id, started) {
                self.completions
                    .push(GpuCompletion::GpuQueueRoom { channel: id });
            }
        }
    }
}

/// Stops `lane` after `error` (module docs, "What stops a channel").
fn stop_lane(
    lane: &mut Lane,
    id: ChannelId,
    error: ChannelError,
    shared: &FrontShared,
    waiters: &WaiterTable,
    counters: &mut FrontCounters,
) {
    tracing::warn!(channel = id.0, %error, "a GPU channel stopped");
    lane.stop(waiters);
    shared.set_error(id, error);
    counters.stopped_channels = counters.stopped_channels.saturating_add(1);
}
