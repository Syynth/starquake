//! The GPU's own flight recorder (gpu-protocol spec, "Observability"; the observability spec's
//! artifacts table: "`gpu.txt`: the GPU's own flight recorder"): the last [`GpuConfig`]'s
//! `recorder_events` events the wiring sees, written as text by [`FlightRecorder::text`].
//!
//! # What is recorded
//!
//! Exactly these, each at the call or report that makes it:
//!
//! | Event | Recorded by |
//! |---|---|
//! | an address space made | `GpuFront::create_address_space` (the first making of a space; a second is ignored, as the front ignores it) |
//! | a channel made, or refused | `GpuFront::create_channel` |
//! | a submission, with its entries, fence wait, increments, wait for idle and whether it was queued or full | `GpuFront::submit` |
//! | an ordered update or a barrier queued | `GpuFront::update_vm`, `GpuFront::barrier` |
//! | a present queued | `GpuFront::present` |
//! | a channel destroyed | `GpuFront::destroy_channel` |
//! | a batch of fenced actions performed: releases, increments, whether a waiter was latched, markers, and each failed release | the back's report to the front (`back::fence`'s `FrontWake`) |
//! | a completion pushed by the front or the back: `GpuQueueRoom`, `GpuOrdered` | the GPU's completion queue (`SyncpointChanged` is pushed by `Syncpoints` itself, which the wiring does not wrap, and is not recorded) |
//!
//! [`FlightRecorder::text`] adds the channels a structural error stopped, from the front's own
//! table, when it is written.
//!
//! **Not recorded**, though the spec's list names them: acquires parked and resumed with the
//! value seen, and macro errors. The front loop keeps only counts of them (`parks`,
//! `macro_failures` in [`super::GpuCounters`]); recording each needs a hook inside the front's
//! channel lanes, a follow-up.
//!
//! The text is plain words, one event per line, numbered from 0 in the order recorded; when more
//! events were recorded than are kept, the first line says how many were dropped.
//!
//! [`GpuConfig`]: super::GpuConfig

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::back::fence::ReleaseFailure;
use crate::channel::ChannelError;
use crate::interface::{
    AsId, ChannelId, Fence, GpuCompletion, PresentRequest, Submitted, SyncpointId, Ticket, VmUpdate,
};
use crate::ring::RingEntry;

/// One event the flight recorder keeps (module docs, "What is recorded").
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuEvent {
    /// An address space was made.
    SpaceMade {
        /// The space.
        space: AsId,
    },
    /// A channel was made.
    ChannelMade {
        /// The channel.
        channel: ChannelId,
        /// Its address space.
        space: AsId,
        /// Its syncpoint.
        syncpoint: SyncpointId,
    },
    /// A channel was refused.
    ChannelRefused {
        /// The address space asked for.
        space: AsId,
    },
    /// A submission was queued, or refused as full.
    Submitted {
        /// The channel.
        channel: ChannelId,
        /// Its GP entries.
        entries: usize,
        /// The fence it waits on first.
        wait: Option<Fence>,
        /// The increments after it.
        increments: u32,
        /// Whether it waits for idle before the increments.
        wfi: bool,
        /// Queued or full.
        outcome: Submitted,
    },
    /// An ordered update was queued.
    Update {
        /// The address space.
        space: AsId,
        /// The update.
        update: VmUpdate,
        /// The ticket reported when it is applied.
        ticket: Option<Ticket>,
    },
    /// A barrier was queued.
    Barrier {
        /// The ticket reported when it is done.
        ticket: Ticket,
    },
    /// A present was queued.
    Present(PresentRequest),
    /// A channel was destroyed.
    ChannelDestroyed {
        /// The channel.
        channel: ChannelId,
    },
    /// The back performed a batch of fenced actions.
    Performed {
        /// Releases and reductions written.
        released: u64,
        /// Syncpoint increments made.
        incremented: u64,
        /// Whether a release latched a waiter.
        latched: bool,
        /// Markers performed.
        markers: usize,
        /// Releases that failed, in order.
        failed: Vec<ReleaseFailure>,
    },
    /// The front or the back pushed a completion to gsv.
    Completion(GpuCompletion),
}

/// A `VmUpdate` in words.
struct UpdateText<'a>(&'a VmUpdate);

impl fmt::Display for UpdateText<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let range = self.0.range();
        let name = match self.0 {
            VmUpdate::Unmap { .. } => "unmap",
            VmUpdate::ModifyKind { .. } => "kind change",
            VmUpdate::DestroyMemory { .. } => "memory destroy",
            VmUpdate::DestroySparseRegion { .. } => "sparse region destroy",
        };
        write!(
            f,
            "{name} of {:#x} bytes at {:#x}",
            range.size(),
            range.start()
        )?;
        match self.0 {
            VmUpdate::ModifyKind { kind, .. } => write!(f, " to kind {:#x}", kind.0),
            VmUpdate::DestroySparseRegion { region, .. } => write!(f, " (region {})", region.0),
            VmUpdate::Unmap { .. } | VmUpdate::DestroyMemory { .. } => Ok(()),
        }
    }
}

/// A ticket, or that there is none.
fn ticket_text(ticket: Option<Ticket>) -> String {
    ticket.map_or_else(|| "no ticket".to_owned(), |t| format!("ticket {}", t.0))
}

impl fmt::Display for GpuEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpaceMade { space } => write!(f, "address space {} made", space.0),
            Self::ChannelMade {
                channel,
                space,
                syncpoint,
            } => write!(
                f,
                "channel {} made in address space {}, incrementing syncpoint {}",
                channel.0,
                space.0,
                syncpoint.get()
            ),
            Self::ChannelRefused { space } => {
                write!(f, "a channel in address space {} was refused", space.0)
            }
            Self::Submitted {
                channel,
                entries,
                wait,
                increments,
                wfi,
                outcome,
            } => {
                let outcome = match outcome {
                    Submitted::Queued => "queued",
                    Submitted::Full => "refused, the queue is full",
                };
                write!(
                    f,
                    "submission on channel {}: {entries} entries, {increments} increments",
                    channel.0
                )?;
                if let Some(fence) = wait {
                    write!(
                        f,
                        ", waits for syncpoint {} to reach {}",
                        fence.id.get(),
                        fence.value
                    )?;
                }
                if *wfi {
                    f.write_str(", wait for idle")?;
                }
                write!(f, ": {outcome}")
            }
            Self::Update {
                space,
                update,
                ticket,
            } => write!(
                f,
                "ordered update of address space {}: {}, {}",
                space.0,
                UpdateText(update),
                ticket_text(*ticket)
            ),
            Self::Barrier { ticket } => write!(f, "barrier, ticket {}", ticket.0),
            Self::Present(request) => write!(f, "{}", RingEntry::Present(*request)),
            Self::ChannelDestroyed { channel } => write!(f, "channel {} destroyed", channel.0),
            Self::Performed {
                released,
                incremented,
                latched,
                markers,
                failed,
            } => {
                let latched = if *latched {
                    "a waiter latched"
                } else {
                    "no waiter latched"
                };
                write!(
                    f,
                    "performed {released} releases, {incremented} increments, {markers} markers, {latched}"
                )?;
                for failure in failed {
                    write!(f, "; {failure}")?;
                }
                Ok(())
            }
            Self::Completion(completion) => match completion {
                GpuCompletion::SyncpointChanged { id, value } => {
                    write!(f, "pushed: syncpoint {} changed to {value}", id.get())
                }
                GpuCompletion::GpuOrdered { ticket } => {
                    write!(f, "pushed: ticket {} done", ticket.0)
                }
                GpuCompletion::GpuQueueRoom { channel } => {
                    write!(f, "pushed: channel {} has queue room", channel.0)
                }
            },
        }
    }
}

/// What the recorder holds behind its lock.
#[derive(Debug, Default)]
struct Kept {
    events: VecDeque<(u64, GpuEvent)>,
    /// Events recorded so far, kept or not.
    recorded: u64,
}

/// The flight recorder: the last `capacity` events (module docs). Every method takes its lock
/// for its own operation only.
#[derive(Debug)]
pub struct FlightRecorder {
    capacity: usize,
    kept: Mutex<Kept>,
}

impl FlightRecorder {
    /// A recorder that keeps the last `capacity` events (none for 0).
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            kept: Mutex::new(Kept::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Kept> {
        // Every method leaves the deque whole, so a poisoned lock is usable.
        self.kept.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Records `event`, dropping the oldest kept one when `capacity` are kept.
    pub fn record(&self, event: GpuEvent) {
        let mut kept = self.lock();
        let number = kept.recorded;
        kept.recorded = kept.recorded.saturating_add(1);
        if self.capacity == 0 {
            return;
        }
        if kept.events.len() >= self.capacity {
            kept.events.pop_front();
        }
        kept.events.push_back((number, event));
    }

    /// The events kept, oldest first.
    pub fn events(&self) -> Vec<GpuEvent> {
        self.lock()
            .events
            .iter()
            .map(|(_, event)| event.clone())
            .collect()
    }

    /// Events recorded so far, kept or not.
    pub fn recorded(&self) -> u64 {
        self.lock().recorded
    }

    /// `gpu.txt`: the events kept, one numbered line each, after a line saying how many earlier
    /// ones were dropped (when any were), then the channels `errors` names, one line each.
    pub fn text(&self, errors: &[(ChannelId, ChannelError)]) -> String {
        use std::fmt::Write as _;
        let kept = self.lock();
        let mut text = String::new();
        let dropped = kept
            .recorded
            .saturating_sub(u64::try_from(kept.events.len()).unwrap_or(u64::MAX));
        if dropped > 0 {
            // Writing to a `String` cannot fail.
            let _ = writeln!(text, "{dropped} earlier events were dropped");
        }
        for (number, event) in &kept.events {
            let _ = writeln!(text, "{number} {event}");
        }
        drop(kept);
        for (channel, error) in errors {
            let _ = writeln!(text, "channel {} stopped: {error}", channel.0);
        }
        text
    }
}
