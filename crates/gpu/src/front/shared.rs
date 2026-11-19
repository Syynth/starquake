//! What the front thread shares with the threads that call it ([`FrontShared`]): the requests
//! gsv's threads queue in call order, each channel's queue bound, the back's reports, and the
//! front's wake-up (gpu-protocol spec, "Channels and submissions", "Back-pressure", "Threads and
//! data flow": "the submission queues").
//!
//! Every method a caller outside the front thread uses takes one mutex for a few operations on
//! plain data, never blocks on the GPU, and calls nothing outside this crate while holding it,
//! so a vCPU thread is never held by the GPU (spec, "Thread rules"). The back thread reaches it
//! through [`FrontWake`].
//!
//! # The queue bound
//!
//! A channel's queue holds at most [`CHANNEL_QUEUE_ENTRIES`] GP entries (graphics-server
//! decision 12): every entry of every submission [`FrontShared::submit`] accepted and the front
//! has not yet started (`crate::channel::Channel::start_entry`). A submission that would take the
//! count past the bound is answered [`Submitted::Full`] and nothing of it is queued; one that
//! takes it exactly to the bound is queued ([`fits`], `a_submission_that_fills_the_queue_exactly_is_queued`).
//! After a `Full`, the front pushes [`GpuCompletion::GpuQueueRoom`] for the channel once the
//! largest submission refused since the last room report would fit
//! (`room_is_reported_once_the_refused_submission_fits`). A submission of more entries than the
//! bound never fits, so no room report follows it; gsv refuses those itself before they reach
//! the front (`InvalidSize`).
//!
//! # Requests the front does not take
//!
//! - A submission for a channel that does not exist (never made, or destroyed) is dropped,
//!   counted in [`FrontShared::dropped_submissions`] and answered [`Submitted::Queued`]: gsv
//!   submits only on channels it made, and a destroyed channel's queued work is dropped too
//!   (`a_submission_for_an_unknown_channel_is_dropped`).
//! - A channel in an address space that was never made is refused
//!   ([`CreateError::UnknownSpace`]).
//! - Making an address space a second time keeps the first one's table and memory
//!   (`making_an_address_space_twice_keeps_the_first`).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use thiserror::Error;

use crate::back::fence::{FrontWake, Report};
use crate::channel::ChannelError;
use crate::interface::{
    AsId, CHANNEL_QUEUE_ENTRIES, ChannelConfig, ChannelId, Submission, Submitted, Ticket, VmUpdate,
};
use crate::memory::GpuMemory;
use crate::ring::MarkerToken;
use crate::vm::GpuVm;

/// Whether a submission of `entries` GP entries fits in a channel queue that holds `queued`
/// (module docs, "The queue bound"): the sum is at most [`CHANNEL_QUEUE_ENTRIES`].
pub fn fits(queued: usize, entries: usize) -> bool {
    queued
        .checked_add(entries)
        .is_some_and(|total| total <= CHANNEL_QUEUE_ENTRIES)
}

/// A channel [`FrontShared::create_channel`] cannot make.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum CreateError {
    /// The address space was never made ([`FrontShared::create_address_space`]).
    #[error("address space {} was never made", .0.0)]
    UnknownSpace(AsId),
    /// Every channel id has been handed out.
    #[error("every GPU channel id has been used")]
    OutOfIds,
}

/// A call queued for the front thread, in call order.
pub(super) enum Request {
    CreateSpace {
        space: AsId,
        vm: Arc<GpuVm>,
        memory: Arc<dyn GpuMemory>,
    },
    CreateChannel {
        id: ChannelId,
        config: ChannelConfig,
    },
    Submit {
        channel: ChannelId,
        submission: Submission,
    },
    Update {
        space: AsId,
        update: VmUpdate,
        ticket: Option<Ticket>,
    },
    Barrier {
        ticket: Ticket,
    },
    Destroy {
        channel: ChannelId,
    },
}

/// What the back reported since the front last looked.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct BackReports {
    /// Markers performed, in order.
    pub(super) markers: Vec<MarkerToken>,
    /// A release was written, failed or latched a waiter: parked acquires are rechecked.
    pub(super) released: bool,
    /// A syncpoint was incremented: fence waits are rechecked.
    pub(super) incremented: bool,
}

/// One channel's queue bound.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Queue {
    /// Entries accepted and not yet started.
    queued: usize,
    /// The largest submission refused since the last room report.
    wanted: Option<usize>,
}

/// Everything behind the mutex.
#[derive(Default)]
struct Inbox {
    requests: VecDeque<Request>,
    spaces: BTreeSet<AsId>,
    queues: BTreeMap<ChannelId, Queue>,
    errors: BTreeMap<ChannelId, ChannelError>,
    /// The id the next channel takes; past `u32::MAX` once every id is used.
    next_channel: u64,
    reports: BackReports,
    /// Something happened since the front last parked or took the inbox.
    signalled: bool,
    stopping: bool,
}

/// What the front takes in one look.
pub(super) struct Taken {
    pub(super) requests: VecDeque<Request>,
    pub(super) reports: BackReports,
    pub(super) stopping: bool,
}

/// What the front thread shares with its callers (module docs).
#[derive(Default)]
pub struct FrontShared {
    inbox: Mutex<Inbox>,
    wake: Condvar,
    dropped: AtomicU64,
}

impl FrontShared {
    /// Nothing queued, no address space, no channel.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, Inbox> {
        // Every method leaves the inbox whole before it could panic, so a poisoned lock is
        // usable.
        self.inbox.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Queues `request` and wakes the front.
    fn queue(&self, inbox: &mut Inbox, request: Request) {
        inbox.requests.push_back(request);
        inbox.signalled = true;
        self.wake.notify_one();
    }

    /// A new address space with its table, and the memory the front reads its command words
    /// through (module docs, "Requests the front does not take").
    pub fn create_address_space(&self, space: AsId, vm: Arc<GpuVm>, memory: Arc<dyn GpuMemory>) {
        let mut inbox = self.lock();
        if !inbox.spaces.insert(space) {
            tracing::warn!(space = space.0, "an address space was made twice; the first is kept");
            return;
        }
        self.queue(&mut inbox, Request::CreateSpace { space, vm, memory });
    }

    /// A new channel in `config`'s address space, incrementing `config`'s syncpoint.
    ///
    /// # Errors
    ///
    /// [`CreateError::UnknownSpace`] if the address space was never made;
    /// [`CreateError::OutOfIds`] once 2^32 channels have been made.
    pub fn create_channel(&self, config: ChannelConfig) -> Result<ChannelId, CreateError> {
        let mut inbox = self.lock();
        if !inbox.spaces.contains(&config.space) {
            return Err(CreateError::UnknownSpace(config.space));
        }
        let id = u32::try_from(inbox.next_channel)
            .map(ChannelId)
            .map_err(|_| CreateError::OutOfIds)?;
        inbox.next_channel += 1;
        inbox.queues.insert(id, Queue::default());
        self.queue(&mut inbox, Request::CreateChannel { id, config });
        Ok(id)
    }

    /// Queues `submission` on `channel`, or answers [`Submitted::Full`] and queues nothing
    /// (module docs, "The queue bound").
    pub fn submit(&self, channel: ChannelId, submission: Submission) -> Submitted {
        let mut inbox = self.lock();
        let entries = submission.entries.len();
        let Some(queue) = inbox.queues.get_mut(&channel) else {
            drop(inbox);
            self.dropped.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                channel = channel.0,
                "a submission for a channel that does not exist was dropped"
            );
            return Submitted::Queued;
        };
        if !fits(queue.queued, entries) {
            queue.wanted = Some(queue.wanted.map_or(entries, |wanted| wanted.max(entries)));
            return Submitted::Full;
        }
        queue.queued += entries;
        self.queue(
            &mut inbox,
            Request::Submit {
                channel,
                submission,
            },
        );
        Submitted::Queued
    }

    /// An ordered update of `space`'s table, reported with `done` (if given) once no work queued
    /// before it can use the old mapping (spec, "Ordered address-table updates and barriers").
    pub fn update_vm(&self, space: AsId, update: VmUpdate, done: Option<Ticket>) {
        let mut inbox = self.lock();
        self.queue(
            &mut inbox,
            Request::Update {
                space,
                update,
                ticket: done,
            },
        );
    }

    /// Reports `done` once all work and updates queued before it are finished.
    pub fn barrier(&self, done: Ticket) {
        let mut inbox = self.lock();
        self.queue(&mut inbox, Request::Barrier { ticket: done });
    }

    /// Destroys `channel`: later submissions to it are dropped, and the front drops what it
    /// still holds of it when it reaches this call.
    pub fn destroy_channel(&self, channel: ChannelId) {
        let mut inbox = self.lock();
        inbox.queues.remove(&channel);
        self.queue(&mut inbox, Request::Destroy { channel });
    }

    /// The structural error that stopped `channel`, if one did.
    pub fn channel_error(&self, channel: ChannelId) -> Option<ChannelError> {
        self.lock().errors.get(&channel).copied()
    }

    /// Submissions dropped because their channel did not exist.
    pub fn dropped_submissions(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Asks the front thread to stop at its next look.
    pub fn stop(&self) {
        let mut inbox = self.lock();
        inbox.stopping = true;
        inbox.signalled = true;
        self.wake.notify_one();
    }

    /// Takes everything queued and reported since the last look.
    pub(super) fn take(&self) -> Taken {
        let mut inbox = self.lock();
        inbox.signalled = false;
        Taken {
            requests: std::mem::take(&mut inbox.requests),
            reports: std::mem::take(&mut inbox.reports),
            stopping: inbox.stopping,
        }
    }

    /// Sleeps until something is queued or reported, or `timeout` passes; whether something
    /// woke it. A wake that came since the last [`FrontShared::take`] ends it at once, so none
    /// is lost between the front's last look and its sleep.
    pub(super) fn park(&self, timeout: Duration) -> bool {
        let mut inbox = self.lock();
        if !inbox.signalled {
            inbox = self
                .wake
                .wait_timeout(inbox, timeout)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        std::mem::take(&mut inbox.signalled)
    }

    /// The front started `started` entries of `channel`, or dropped them: they leave its queue.
    /// Returns whether the room a refused submission waits for is now there; the caller then
    /// pushes [`GpuCompletion::GpuQueueRoom`](crate::interface::GpuCompletion::GpuQueueRoom),
    /// with the lock released.
    #[must_use]
    pub(super) fn release_entries(&self, channel: ChannelId, started: usize) -> bool {
        let mut inbox = self.lock();
        let Some(queue) = inbox.queues.get_mut(&channel) else {
            return false;
        };
        queue.queued = queue.queued.saturating_sub(started);
        match queue.wanted {
            Some(wanted) if fits(queue.queued, wanted) => {
                queue.wanted = None;
                true
            }
            _ => false,
        }
    }

    /// Records the error that stopped `channel`.
    pub(super) fn set_error(&self, channel: ChannelId, error: ChannelError) {
        self.lock().errors.insert(channel, error);
    }

    /// Forgets a destroyed channel's error.
    pub(super) fn forget(&self, channel: ChannelId) {
        self.lock().errors.remove(&channel);
    }

    /// Entries `channel`'s queue holds, for tests.
    #[cfg(test)]
    pub(super) fn queued(&self, channel: ChannelId) -> Option<usize> {
        self.lock().queues.get(&channel).map(|queue| queue.queued)
    }

    /// Sets the id the next channel takes, for tests of the id bound.
    #[cfg(test)]
    pub(super) fn set_next_channel(&self, next: u64) {
        self.lock().next_channel = next;
    }
}

impl FrontWake for FrontShared {
    /// Notes what the back performed and wakes the front: markers for ordered updates,
    /// catch-ups and `SYNC_WAIT`s; releases and increments for parked channels.
    fn wake(&self, report: &Report) {
        let mut inbox = self.lock();
        inbox.reports.markers.extend_from_slice(&report.markers);
        if report.released > 0 || report.latched || !report.failed.is_empty() {
            inbox.reports.released = true;
        }
        if report.incremented > 0 {
            inbox.reports.incremented = true;
        }
        inbox.signalled = true;
        self.wake.notify_one();
    }
}
