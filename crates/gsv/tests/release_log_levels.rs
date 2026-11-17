//! The level at which the drain logs the kernel's refusal of a device-share release
//! (`starquake_gsv::shared` module docs, "Lock order and the drain"): debug for
//! `DeviceShareError::ProcessEnded`, and error for each other refusal.
//!
//! # Why this file holds exactly one test
//!
//! The logs are observed with a subscriber installed for the test's own thread
//! (`tracing::subscriber::with_default`). That is only reliable when no other thread in the
//! process logs from the same places: `tracing-core` decides whether a log statement is enabled
//! by asking the default subscriber of whichever thread reaches it first, and caches the answer
//! (`crates/services/tests/unknown_command_logs.rs` explains it in full). The crate's unit tests
//! drain refused releases with no subscriber of their own, so this check lives in its own test
//! binary, in a single test function. Do not add a second `#[test]` here; add to the one below.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use starquake_gpu::interface::Completions;
use starquake_gpu::syncpoint::Syncpoints;
use starquake_gsv::completion::{Completion, CompletionQueue};
use starquake_gsv::shared::{Shared, TimerHost};
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::id::{ObjectId, ProcessId};
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::memory::testing::device_share_token;
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

/// A [`TimerHost`] that answers each unlock with the next of its answers, and records nothing.
struct Answers(VecDeque<Result<(), DeviceShareError>>);

impl TimerHost for Answers {
    fn record(&mut self, _kind: EventKind) {}

    fn unlock_device_share(&mut self, _token: DeviceShareToken) -> Result<(), DeviceShareError> {
        self.0.pop_front().unwrap_or(Ok(()))
    }

    fn signal_event(&mut self, _event: ObjectId) -> SvcResult<()> {
        Ok(())
    }
}

/// The release of the `n`th page-sized range of a process.
fn released(n: u64) -> Completion {
    let page = 0x1000;
    Completion::DeviceShareReleased {
        token: device_share_token(ProcessId::from_parts(2, 3), n * page, page),
    }
}

#[test]
fn a_refused_release_is_logged_at_debug_when_its_process_ended_and_at_error_otherwise() {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    let shared = Shared::new(
        Arc::new(Syncpoints::new(completions)),
        queue,
        ServiceId(0x0657),
    );
    // Each refusal the kernel can answer, then an accepted release, which logs nothing. The
    // ended process comes between two other refusals, so a level given to the wrong arm moves.
    let answers = [
        Err(DeviceShareError::NotLocked {
            address: 0,
            size: 0x1000,
        }),
        Err(DeviceShareError::ProcessEnded),
        Err(DeviceShareError::WrongProcess),
        Err(DeviceShareError::Refused(
            KernelError::InvalidCurrentMemory.into(),
        )),
        Ok(()),
    ];
    for n in 0..answers.len() {
        shared.queue().push(released(u64::try_from(n).unwrap()));
    }
    let count = answers.len();
    let mut host = Answers(answers.into());
    let levels = Levels::default();
    let drained = tracing::subscriber::with_default(levels.clone(), || shared.drain(&mut host));
    assert_eq!(drained.len(), count);
    assert_eq!(
        levels.seen(),
        [Level::ERROR, Level::DEBUG, Level::ERROR, Level::ERROR]
    );
}
