//! The back thread (the wiring's module docs, "The back thread"): the executor's loop, and how it
//! learns of address spaces and channels at run time.
//!
//! The executor (slice G19) is owned by this thread and built on it. Address spaces and channels
//! reach it through [`BackInbox`], a queue of [`Control`]s applied between ring entries, in the
//! order they were posted:
//!
//! - **An address space** is posted by `GpuFront::create_address_space` and **a channel** by
//!   `GpuFront::create_channel`, at the call. Both are posted before the call returns, so before
//!   any submission on the channel is queued, and so before the front can push any entry of it:
//!   the post happens before the front's push and publish, the back's pop of the entry happens
//!   after the publish, and the back looks at its inbox after each pop and before it applies the
//!   entry (`a_channel_made_and_used_at_once_is_known_to_the_back`).
//! - **A channel is forgotten** when the front has dropped it, with the number of ring entries the
//!   front had pushed by then ([`crate::front::ChannelRetired`]): the back forgets it once it has
//!   taken that many, so every entry the channel made is applied first
//!   (`a_destroyed_channel_is_forgotten_after_its_last_entry`). A forget that is not yet due holds
//!   the controls posted after it, which is in time for them: a channel made after it is used
//!   only by entries pushed after the destroy.
//!
//! A posted control also wakes the back ([`crate::ring::ConsumerWake`]), so a forget that falls
//! due while the ring stays empty is applied without waiting for an entry.
//!
//! # How it ends
//!
//! The loop ends when the ring's producer is gone and every entry it published has been taken
//! (the front thread ended: [`crate::ring::PopError::Disconnected`]), or when [`BackInbox::stop`]
//! was called, at its next wake. A corrupt ring ends it with [`BackError::Ring`]. Any other
//! failure of the wait (the backend refused a list closed for an empty ring) is logged, counted
//! in [`BackStatus::wait_failures`], and the loop goes on.

use std::collections::{BTreeSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use starquake_memory::WriteWatch;
use thiserror::Error;

use crate::back::exec::{ExecCounters, Executor};
use crate::back::fence::{FenceCounters, FenceError, Woke};
use crate::front::ChannelRetired;
use crate::interface::{AsId, ChannelId};
use crate::memory::GpuMemory;
use crate::ring::{Consumer, ConsumerWake, PopError};

/// The longest one wait of the back thread lasts before it looks at its inbox again, though a
/// posted control or a stop wakes it at once: a safety net only.
pub const BACK_WAIT: Duration = Duration::from_millis(100);

/// What the back thread is told between ring entries (module docs).
pub(super) enum Control {
    /// Add an address space, with the back's own memory over its table; `present` makes it the
    /// space presents read (the wiring's module docs, "Presents").
    AddSpace {
        space: AsId,
        memory: Arc<dyn GpuMemory>,
        watch: Arc<WriteWatch>,
        present: bool,
    },
    /// Add a channel.
    AddChannel { channel: ChannelId, space: AsId },
    /// Forget a channel once `after` ring entries have been taken.
    Forget { channel: ChannelId, after: u64 },
}

/// Where the wiring posts [`Control`]s for the back thread (module docs).
pub(super) struct BackInbox {
    controls: Mutex<VecDeque<Control>>,
    stopping: AtomicBool,
    wake: ConsumerWake,
}

impl BackInbox {
    /// An empty inbox that wakes the back through `wake`.
    pub(super) fn new(wake: ConsumerWake) -> Self {
        Self {
            controls: Mutex::new(VecDeque::new()),
            stopping: AtomicBool::new(false),
            wake,
        }
    }

    fn lock(&self) -> MutexGuard<'_, VecDeque<Control>> {
        // Every method leaves the queue whole, so a poisoned lock is usable.
        self.controls.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Posts `control` and wakes the back.
    pub(super) fn post(&self, control: Control) {
        self.lock().push_back(control);
        self.wake.wake();
    }

    /// Asks the back thread to end at its next wake, and wakes it.
    pub(super) fn stop(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        self.wake.wake();
    }

    fn is_stopping(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }
}

impl ChannelRetired for BackInbox {
    fn retired(&self, channel: ChannelId, after: u64) {
        self.post(Control::Forget { channel, after });
    }
}

/// What the back thread publishes after each wait, for counters and idleness.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackStatus {
    /// The executor's counters.
    pub exec: ExecCounters,
    /// The fence path's counters.
    pub fences: FenceCounters,
    /// Channels the back knows now.
    pub channels: usize,
    /// Address spaces the back knows now.
    pub spaces: usize,
    /// Ring entries taken so far.
    pub taken: u64,
    /// Command lists submitted and not yet completed, plus one while the open list holds
    /// commands.
    pub in_flight: u64,
    /// Waits that failed other than by the ring (module docs, "How it ends").
    pub wait_failures: u64,
    /// Controls the executor refused (a duplicate or unknown space or channel), each logged.
    pub refused_controls: u64,
}

/// Why the back thread ended other than when asked.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum BackError {
    /// The ring is corrupt.
    #[error("the GPU back thread stopped: {0}")]
    Ring(PopError),
    /// The back thread panicked.
    #[error("the GPU back thread panicked")]
    Panicked,
}

/// What the back thread shares with the wiring.
#[derive(Default)]
pub(super) struct BackShared {
    status: Mutex<BackStatus>,
    /// [`BackStatus::taken`], readable without the lock.
    pub(super) taken: AtomicU64,
    /// [`BackStatus::in_flight`], readable without the lock; stored before `taken`.
    pub(super) in_flight: AtomicU64,
}

impl BackShared {
    pub(super) fn status(&self) -> BackStatus {
        *self.status.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The back thread's state (module docs).
pub(super) struct Back {
    pub(super) exec: Executor,
    pub(super) ring: Consumer,
    pub(super) inbox: Arc<BackInbox>,
    pub(super) shared: Arc<BackShared>,
    pub(super) channels: BTreeSet<ChannelId>,
    pub(super) spaces: BTreeSet<AsId>,
    pub(super) taken: u64,
    pub(super) wait_failures: u64,
    pub(super) refused_controls: u64,
}

impl Back {
    /// The loop (module docs).
    pub(super) fn run(mut self) -> Result<(), BackError> {
        while self.step(BACK_WAIT)? {}
        Ok(())
    }

    /// One pass of the loop, waiting at most `wait`: whether the loop goes on.
    ///
    /// # Errors
    ///
    /// [`BackError::Ring`] for a corrupt ring.
    pub(super) fn step(&mut self, wait: Duration) -> Result<bool, BackError> {
        self.apply_controls();
        if self.inbox.is_stopping() {
            self.publish();
            return Ok(false);
        }
        match self.exec.wait(&mut self.ring, wait) {
            Ok(Woke::Entry(entry)) => {
                // The entry is the `taken`-th: what was posted before it was pushed is due.
                self.apply_controls();
                // A refused entry is logged and counted by the executor.
                if self.exec.apply(&entry).is_err() {
                    tracing::trace!("the back refused an entry; counted");
                }
                self.taken = self.taken.saturating_add(1);
            }
            Ok(Woke::Completed(_) | Woke::TimedOut) => {}
            Err(FenceError::Ring(PopError::Disconnected)) => {
                self.apply_controls();
                self.publish();
                return Ok(false);
            }
            Err(FenceError::Ring(error)) => {
                tracing::error!(%error, "the GPU back thread's ring is corrupt; it stops");
                self.publish();
                return Err(BackError::Ring(error));
            }
            Err(error) => {
                tracing::warn!(%error, "the GPU back thread's wait failed");
                self.wait_failures = self.wait_failures.saturating_add(1);
            }
        }
        self.publish();
        Ok(true)
    }

    /// Applies every control that is due, in the order posted (module docs). Each is taken off
    /// the inbox under its lock and applied with the lock released, so a post never waits for
    /// the executor.
    pub(super) fn apply_controls(&mut self) {
        while let Some(control) = self.next_due() {
            self.apply_control(control);
        }
    }

    /// The oldest control posted, if it is due: anything but a forget, or a forget whose
    /// entries have all been taken.
    fn next_due(&self) -> Option<Control> {
        let mut controls = self.inbox.lock();
        if let Some(Control::Forget { after, .. }) = controls.front()
            && *after > self.taken
        {
            return None;
        }
        controls.pop_front()
    }

    fn apply_control(&mut self, control: Control) {
        let result = match control {
            Control::AddSpace {
                space,
                memory,
                watch,
                present,
            } => self.exec.add_space(space, memory, watch).and_then(|()| {
                self.spaces.insert(space);
                if present {
                    self.exec.set_present_space(space)
                } else {
                    Ok(())
                }
            }),
            Control::AddChannel { channel, space } => {
                self.exec.add_channel(channel, space).map(|()| {
                    self.channels.insert(channel);
                })
            }
            Control::Forget { channel, .. } => {
                self.channels.remove(&channel);
                self.exec.remove_channel(channel).map(|_| ())
            }
        };
        if let Err(error) = result {
            tracing::warn!(%error, "the GPU back thread refused a space or channel change");
            self.refused_controls = self.refused_controls.saturating_add(1);
        }
    }

    /// Publishes [`BackStatus`]: `in_flight` before `taken` (the wiring's module docs,
    /// "Idleness").
    fn publish(&self) {
        let fences = self.exec.fences();
        let open = u64::from(fences.open_commands() > 0);
        let lists = u64::try_from(fences.lists_in_flight()).unwrap_or(u64::MAX);
        let in_flight = lists.saturating_add(open);
        let status = BackStatus {
            exec: self.exec.counters(),
            fences: fences.counters(),
            channels: self.channels.len(),
            spaces: self.spaces.len(),
            taken: self.taken,
            in_flight,
            wait_failures: self.wait_failures,
            refused_controls: self.refused_controls,
        };
        *self
            .shared
            .status
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = status;
        self.shared.in_flight.store(in_flight, Ordering::SeqCst);
        self.shared.taken.store(self.taken, Ordering::SeqCst);
    }
}
