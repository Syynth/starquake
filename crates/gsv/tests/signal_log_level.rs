//! The level at which the drain logs the kernel's refusal to signal a fired wait's event
//! (`starquake_gsv::shared` module docs, "Lock order and the drain"): error, and nothing for a
//! signal the kernel takes.
//!
//! # Why this file holds exactly one test
//!
//! The logs are observed with a subscriber installed for the test's own thread
//! (`tracing::subscriber::with_default`), which is only reliable when no other thread in the
//! process logs from the same places (`release_log_levels.rs` says why). The crate's unit tests
//! drain refused signals with no subscriber of their own, so this check lives in its own test
//! binary, in a single test function. Do not add a second `#[test]` here; add to the one below.

use std::sync::{Arc, Mutex};

use starquake_gpu::interface::{Completions, GpuCompletion, SyncpointId};
use starquake_gpu::syncpoint::{CompareMode, Syncpoints};
use starquake_gsv::completion::{Completion, CompletionQueue};
use starquake_gsv::shared::{Shared, TimerHost};
use starquake_gsv::syncpt::waiters::Wait;
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::result::{KernelError, SvcResult};
use starquake_kernel::slab::SlabId;
use starquake_trace::{EventKind, ServiceId};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};

/// Records the level of every event logged while it is the thread's default subscriber.
#[derive(Clone, Default)]
struct Levels(Arc<Mutex<Vec<Level>>>);

impl Levels {
    fn seen(&self) -> Vec<Level> {
        self.0.lock().map_or_default(|seen| seen.clone())
    }
}

impl Subscriber for Levels {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        if let Ok(mut seen) = self.0.lock() {
            seen.push(*event.metadata().level());
        }
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

/// A [`TimerHost`] that refuses to signal `refused`, signals every other event, and records
/// nothing.
struct Refusing {
    refused: ObjectId,
    signalled: Vec<ObjectId>,
}

impl TimerHost for Refusing {
    fn record(&mut self, _kind: EventKind) {}

    fn unlock_device_share(&mut self, _token: DeviceShareToken) -> Result<(), DeviceShareError> {
        Ok(())
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        self.signalled.push(event);
        if event == self.refused {
            Err(KernelError::InvalidHandle)
        } else {
            Ok(())
        }
    }
}

#[test]
fn a_refused_signal_is_logged_at_error_and_a_taken_one_not_at_all() {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    let shared = Shared::new(
        Arc::new(Syncpoints::new(completions)),
        queue,
        ServiceId(0x0657),
    );
    // Three waits one change fires: the refused one between two taken ones.
    let id = SyncpointId::new(9).unwrap();
    let events = [1, 2, 3].map(|index| ObjectId::from_parts(index, 1));
    for event in events {
        let wait = Wait::new(id, 1, CompareMode::Wrapping32).unwrap();
        shared.lock().arm(event, wait).unwrap();
    }
    shared
        .queue()
        .push(Completion::Gpu(GpuCompletion::SyncpointChanged {
            id,
            value: 1,
        }));
    let mut host = Refusing {
        refused: events[1],
        signalled: Vec::new(),
    };
    let levels = Levels::default();
    let drained = tracing::subscriber::with_default(levels.clone(), || shared.drain(&mut host));
    assert_eq!(drained.len(), 1);
    assert_eq!(host.signalled, events);
    assert_eq!(levels.seen(), [Level::ERROR]);
}
