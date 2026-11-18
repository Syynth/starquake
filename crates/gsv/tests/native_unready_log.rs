//! The warning a native command logs before the bootstrap is complete (`starquake_gsv::native`
//! module docs, "Where this differs from the references": accepted and logged, decision 7): one
//! warning naming the command for a native command in `Phase::Created`, and none once the
//! phase is `Phase::Ready`.
//!
//! # Why this file holds exactly one test
//!
//! The logs are observed with a subscriber installed for the test's own thread
//! (`tracing::subscriber::with_default`), which is only reliable when no other thread in the
//! process logs from the same places (`release_log_levels.rs` says why). The crate's unit tests
//! send native commands before the bootstrap with no subscriber of their own, so this check
//! lives in its own test binary, in a single test function. Do not add a second `#[test]` here;
//! add to the one below.

use std::fmt;
use std::sync::{Arc, Mutex};

use starquake_gpu::interface::Completions;
use starquake_gpu::syncpoint::Syncpoints;
use starquake_gsv::completion::CompletionQueue;
use starquake_gsv::protocol::bootstrap;
use starquake_gsv::protocol::command::{
    BIND_PROCESS, COMPLETE_BOOTSTRAP, CREATE_NATIVE_MEMORY, INITIALIZE,
};
use starquake_gsv::root::{Phase, RootFamily, RootState, root_server_with_page};
use starquake_gsv::shared::Shared;
use starquake_ipc::{BUFFER_SIZE, TIPC_COMMAND_OFFSET};
use starquake_kernel::id::ObjectId;
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::{Handled, Kernel, Server, UnknownPolicy};
use starquake_trace::ServiceId;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};

/// The number the test's root server is known by.
const ROOT: ServiceId = ServiceId(0x0658);
/// The number gsv's notes are recorded under.
const GSV: ServiceId = ServiceId(0x0659);
/// The TIPC header's "a special header follows" bit.
const SPECIAL: u32 = 0x8000_0000;
/// A special header that counts one copied handle.
const ONE_COPY: u32 = 2;

/// An event's level and its `command` field, if it has one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Seen {
    level: Level,
    command: Option<u64>,
}

/// Reads an event's `command` field.
#[derive(Default)]
struct CommandField(Option<u64>);

impl Visit for CommandField {
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "command" {
            self.0 = Some(value);
        }
    }

    fn record_debug(&mut self, _: &Field, _: &dyn fmt::Debug) {}
}

/// Records every event logged while it is the thread's default subscriber.
#[derive(Clone, Default)]
struct Events(Arc<Mutex<Vec<Seen>>>);

impl Events {
    /// The warnings and errors seen, then forgets every event.
    fn take_warnings(&self) -> Vec<Seen> {
        self.0.lock().map_or_default(|mut seen| {
            let warnings = seen
                .iter()
                .copied()
                .filter(|event| event.level <= Level::WARN)
                .collect();
            seen.clear();
            warnings
        })
    }
}

impl Subscriber for Events {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut command = CommandField::default();
        event.record(&mut command);
        if let Ok(mut seen) = self.0.lock() {
            seen.push(Seen {
                level: *event.metadata().level(),
                command: command.0,
            });
        }
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

/// A root server with gsv's shared state, a fake kernel and one root session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

/// In `cfg(test)`, so it may unwrap like the test does.
#[cfg(test)]
impl Rig {
    fn new() -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_current_service(ROOT);
        let session = kernel.new_object();
        let queue = Arc::new(CompletionQueue::new());
        let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
        let shared = Arc::new(Shared::new(
            Arc::new(Syncpoints::new(completions)),
            queue,
            GSV,
        ));
        let mut server = root_server_with_page(RootState::new(), UnknownPolicy::Strict, shared)
            .with_service_id(ROOT);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
        }
    }

    /// Sends the message `words` lays out; the reply's result word.
    fn send(&mut self, words: &[u32]) -> u32 {
        let mut buffer = vec![0; BUFFER_SIZE];
        for (at, word) in words.iter().enumerate() {
            buffer[at * 4..at * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let handled = self
            .server
            .handle(&mut self.kernel, self.session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        let header = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
        let at = if header & SPECIAL == 0 { 8 } else { 12 };
        u32::from_le_bytes(buffer[at..at + 4].try_into().unwrap())
    }

    /// A handle in the caller's table to a fresh object: what stands for the process.
    fn process(&mut self) -> u32 {
        let object = self.kernel.new_object();
        self.kernel.insert_handle(object).unwrap().raw()
    }

    fn phase(&self) -> Phase {
        self.server.family::<RootFamily>().unwrap().state().phase()
    }

    /// 111 with mode 0 and a page's size: its result word.
    fn create_native_memory(&mut self) -> u32 {
        self.send(&[tipc_type(CREATE_NATIVE_MEMORY), 3, 0, 0x1000, 0])
    }

    /// The bootstrap in PR's order: `Initialize` (local), `BindProcess`, `CompleteBootstrap`.
    fn bootstrap(&mut self) {
        let process = self.process();
        // PR `DecodeBootstrap`: `packet[1] != 0x8000000c` (a special header, twelve data words).
        let mut initialize = vec![tipc_type(INITIALIZE), SPECIAL | 0xC, ONE_COPY, process];
        initialize.extend([0xA1, 0x51]);
        initialize.extend(halves(bootstrap::LOCAL_FLAGS[0]));
        initialize.extend([0; 2 * bootstrap::PARAMETERS]);
        assert_eq!(self.send(&initialize), 0);
        let process = self.process();
        assert_eq!(
            self.send(&[tipc_type(BIND_PROCESS), SPECIAL, ONE_COPY, process]),
            0
        );
        assert_eq!(self.send(&[tipc_type(COMPLETE_BOOTSTRAP), 0]), 0);
    }
}

/// The low and high words of a `u64`.
#[cfg(test)]
fn halves(value: u64) -> [u32; 2] {
    let bytes = value.to_le_bytes();
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    [word(0), word(4)]
}

/// A TIPC message's type word for command `id`.
fn tipc_type(id: u32) -> u32 {
    u32::from(TIPC_COMMAND_OFFSET) + id
}

#[test]
fn a_native_command_before_the_bootstrap_warns_once_and_none_after() {
    let events = Events::default();
    tracing::subscriber::with_default(events.clone(), || {
        let mut rig = Rig::new();
        assert_eq!(rig.phase(), Phase::Created);
        assert_eq!(rig.create_native_memory(), 0);
        assert_eq!(
            events.take_warnings(),
            [Seen {
                level: Level::WARN,
                command: Some(u64::from(CREATE_NATIVE_MEMORY)),
            }]
        );
        rig.bootstrap();
        assert_eq!(rig.phase(), Phase::Ready);
        events.take_warnings();
        assert_eq!(rig.create_native_memory(), 0);
        assert_eq!(events.take_warnings(), []);
    });
}
