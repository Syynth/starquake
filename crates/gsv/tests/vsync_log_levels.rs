//! The levels at which the vsync clock logs (`starquake_gsv::shared::vsync` module docs, "A
//! refused signal" and "The end of the counts"): error for a listener the kernel refuses, for a
//! deadline past the end of time and for a tick count at its maximum, and nothing for a tick
//! whose listeners are all signalled.
//!
//! # Why this file holds exactly one test
//!
//! The logs are observed with a subscriber installed for the test's own thread
//! (`tracing::subscriber::with_default`), which is only reliable when no other thread in the
//! process logs from the same places (`release_log_levels.rs` says why). The crate's unit tests
//! tick the clock with no subscriber of their own, so this check lives in its own test binary,
//! in a single test function. Do not add a second `#[test]` here; add to the one below.

use std::sync::{Arc, Mutex};

use starquake_gpu::interface::Completions;
use starquake_gpu::syncpoint::Syncpoints;
use starquake_gsv::completion::CompletionQueue;
use starquake_gsv::shared::vsync::INTERVAL_NS;
use starquake_gsv::shared::{Shared, TimerHost};
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::result::{KernelError, SvcResult};
use starquake_kernel::slab::SlabId;
use starquake_trace::{EventKind, HostNanos, ServiceId};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};

/// Records the level of every event logged while it is the thread's default subscriber.
#[derive(Clone, Default)]
struct Levels(Arc<Mutex<Vec<Level>>>);

impl Levels {
    fn take(&self) -> Vec<Level> {
        self.0
            .lock()
            .map_or_default(|mut seen| std::mem::take(&mut *seen))
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

/// A [`TimerHost`] that refuses to signal `refused` and signals every other event.
struct Refusing {
    refused: Option<ObjectId>,
}

impl TimerHost for Refusing {
    fn record(&mut self, _kind: EventKind) {}

    fn unlock_device_share(&mut self, _token: DeviceShareToken) -> Result<(), DeviceShareError> {
        Ok(())
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        if Some(event) == self.refused {
            Err(KernelError::InvalidHandle)
        } else {
            Ok(())
        }
    }
}

fn fresh() -> Shared {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    Shared::new(
        Arc::new(Syncpoints::new(completions)),
        queue,
        ServiceId(0x0657),
    )
}

#[test]
fn the_vsync_clock_logs_its_refusals_and_its_ends_at_error_and_a_tick_not_at_all() {
    let levels = Levels::default();
    let taken = ObjectId::from_parts(1, 5);
    let refused = ObjectId::from_parts(2, 5);
    let first = INTERVAL_NS;

    // A tick that signals every listener logs nothing; one with a refused listener, one error.
    let shared = fresh();
    shared.lock().listen_vsync(taken, HostNanos(0)).unwrap();
    shared.lock().listen_vsync(refused, HostNanos(0)).unwrap();
    let mut host = Refusing { refused: None };
    tracing::subscriber::with_default(levels.clone(), || {
        shared.tick_vsync(&mut host, HostNanos(first));
    });
    assert_eq!(levels.take(), Vec::<Level>::new());
    let mut host = Refusing {
        refused: Some(refused),
    };
    tracing::subscriber::with_default(levels.clone(), || {
        shared.tick_vsync(&mut host, HostNanos(first + INTERVAL_NS));
    });
    assert_eq!(levels.take(), [Level::ERROR]);
    assert_eq!(shared.lock().vsync().listeners().count(), 1);

    // A deadline past the end of time: the tick, then one error.
    let shared = fresh();
    let last = HostNanos(u64::MAX - INTERVAL_NS);
    shared.lock().listen_vsync(taken, last).unwrap();
    let mut host = Refusing { refused: None };
    tracing::subscriber::with_default(levels.clone(), || {
        shared.tick_vsync(&mut host, HostNanos(u64::MAX));
    });
    assert_eq!(levels.take(), [Level::ERROR]);
    assert_eq!(shared.vsync_deadline(), None);

    // A tick count at its maximum: one error, and the clock stops. The snapshot, by hand: the
    // magic and version 1, `u64::MAX` ticks, a deadline at 1, nothing fed back, one listener.
    let shared = fresh();
    let mut bytes = vec![b'S', b'Q', b'V', b'S', 0x01];
    bytes.extend([0xFF; 9]);
    bytes.push(0x01);
    bytes.extend([0x01, 0x01, 0x00, 0x01, 0x01, 0x05]);
    shared.restore_vsync(&bytes).unwrap();
    assert_eq!(shared.lock().vsync().tick(), u64::MAX);
    tracing::subscriber::with_default(levels.clone(), || {
        shared.tick_vsync(&mut host, HostNanos(1));
    });
    assert_eq!(levels.take(), [Level::ERROR]);
    assert_eq!(shared.vsync_deadline(), None);
}
