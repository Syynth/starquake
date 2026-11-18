//! The levels at which the vi2 session logs (`starquake_gsv::vi2` module docs, "Vsync" and
//! "Commands"): the vsync fallback at info, when a group's event starts listening with no
//! registration (31) and when an unregister leaves it with none; a kind-0 or other registration
//! at nothing above debug; a vsync clock with no room at error; the display state commands
//! 104 to 107 and 354, which a title calls about 75 times a second (row 119), at trace only;
//! 201's refusal of a bit at or past `CLIENT_NOTIFICATIONS` and 351's of an encoding at warning,
//! each before the framework's own warning (a 201 or 351 refused for another reason logs only
//! the framework's); and the root's 305 with flags PR refuses at warning, with PR's flags at
//! nothing above info.
//!
//! # Why this file holds exactly one test
//!
//! The logs are observed with a subscriber installed for the test's own thread
//! (`tracing::subscriber::with_default`), which is only reliable when no other thread in the
//! process logs from the same places (`release_log_levels.rs` says why). The crate's unit tests
//! drive the vi2 session with no subscriber of their own, so this check lives in its own test
//! binary, in a single test function. Do not add a second `#[test]` here; add to the one below.

use std::sync::{Arc, Mutex};

use starquake_gpu::interface::Completions;
use starquake_gpu::syncpoint::Syncpoints;
use starquake_gsv::completion::CompletionQueue;
use starquake_gsv::root::{DISPLAY_SESSION_FLAGS, RootState, root_server_for};
use starquake_gsv::shared::Shared;
use starquake_gsv::shared::vsync::MAX_LISTENERS;
use starquake_gsv::vi2::command::{
    CREATE_REQUEST_GROUP_EVENT, GET_BUFFER_MAX_RESOLUTION, GET_DISPLAY_STATE_105,
    GET_DISPLAY_STATE_107, GET_DISPLAY_STATE_354, GET_HDR_STATUS, OPEN_DISPLAY,
    REGISTER_CLIENT_STATE, REGISTER_DISPLAY_EVENT_MONITOR, REGISTER_HDR_STATUS_MONITOR,
    SET_CLIENT_OUTPUT_ENCODING, UNREGISTER_HDR_STATUS_MONITOR,
};
use starquake_gsv::vi2::{
    CLIENT_NOTIFICATIONS, DEFAULT_DISPLAY, OPEN_DISPLAY_FLAGS, OUTPUT_ENCODING, vi2_server,
};
use starquake_ipc::{BUFFER_SIZE, TIPC_COMMAND_OFFSET};
use starquake_kernel::id::ObjectId;
use starquake_kernel::slab::SlabId;
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::{Halt, Handled, Server, Stop, UnknownPolicy};
use starquake_trace::{Env, FixedClock, HostNanos, LiveEnv, ServiceId};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};

/// The number the test's vi2 server is known by.
const VI2: ServiceId = ServiceId(0x0305);
/// The number the test's root server is known by.
const ROOT: ServiceId = ServiceId(0x0658);

/// Records the level of every event logged while it is the thread's default subscriber.
#[derive(Clone, Default)]
struct Levels(Arc<Mutex<Vec<Level>>>);

impl Levels {
    fn take(&self) -> Vec<Level> {
        self.0
            .lock()
            .map_or_default(|mut seen| std::mem::take(&mut *seen))
    }

    /// The levels seen at or above `least`, then forgets every event.
    fn take_from(&self, least: Level) -> Vec<Level> {
        self.take().into_iter().filter(|l| *l <= least).collect()
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

/// A halt nothing here reaches.
struct NoHalt;

impl Halt for NoHalt {
    fn halt(&self, stop: Stop) {
        unreachable!("the environment failed: {stop:?}");
    }
}

/// A TIPC message's type word for command `id`.
fn tipc_type(id: u32) -> u32 {
    u32::from(TIPC_COMMAND_OFFSET) + id
}

/// The low and high words of a `u64`.
#[cfg(test)]
fn halves(value: u64) -> [u32; 2] {
    let bytes = value.to_le_bytes();
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    [word(0), word(4)]
}

/// A server, the fake kernel, and one session on it.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

#[cfg(test)]
impl Rig {
    fn over(server: Server, service: ServiceId) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_current_service(service);
        kernel.add_service(VI2);
        let session = kernel.new_object();
        let mut server = server;
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
        let at = if header & 0x8000_0000 == 0 { 8 } else { 16 };
        u32::from_le_bytes(buffer[at..at + 4].try_into().unwrap())
    }
}

fn shared() -> Arc<Shared> {
    let queue = Arc::new(CompletionQueue::new());
    let completions: Arc<dyn Completions> = Arc::<CompletionQueue>::clone(&queue);
    Arc::new(Shared::new(
        Arc::new(Syncpoints::new(completions)),
        queue,
        ServiceId(0x0657),
    ))
}

/// A request of a display name and a group: 261 to 272.
fn monitor(command: u32, group: u32) -> Vec<u32> {
    let [lo, hi] = halves(DEFAULT_DISPLAY);
    vec![tipc_type(command), 3, lo, hi, group]
}

#[test]
fn vi2_logs_the_fallback_at_info_and_the_display_state_not_at_all() {
    let levels = Levels::default();
    tracing::subscriber::with_default(levels.clone(), || {
        let shared = shared();
        let env: Arc<dyn Env> = Arc::new(LiveEnv::new(FixedClock::new(0, 1), 0));
        let server = vi2_server(UnknownPolicy::Strict, Arc::clone(&shared))
            .with_service_id(VI2)
            .with_env(env, Arc::new(NoHalt));
        let mut rig = Rig::over(server, VI2);

        // 31: the group listens through the fallback, at info.
        assert_eq!(rig.send(&[tipc_type(CREATE_REQUEST_GROUP_EVENT), 1, 0]), 0);
        assert_eq!(levels.take_from(Level::INFO), [Level::INFO]);
        // 261 (kind 0): nothing above debug.
        assert_eq!(rig.send(&monitor(REGISTER_DISPLAY_EVENT_MONITOR, 0)), 0);
        assert_eq!(levels.take_from(Level::INFO), Vec::<Level>::new());
        // A kind-1 registration and its unregister: nothing, then the fallback at info.
        assert_eq!(rig.send(&[tipc_type(CREATE_REQUEST_GROUP_EVENT), 1, 1]), 0);
        levels.take();
        assert_eq!(rig.send(&monitor(REGISTER_HDR_STATUS_MONITOR, 1)), 0);
        assert_eq!(levels.take_from(Level::INFO), Vec::<Level>::new());
        assert_eq!(rig.send(&monitor(UNREGISTER_HDR_STATUS_MONITOR, 1)), 0);
        assert_eq!(levels.take_from(Level::INFO), [Level::INFO]);

        // The display state commands answered with success log at trace only: 105, 107 and
        // 354 with the display closed, and all five with it open. 104 and 106 refused while it
        // is closed log only the framework's own warning of a failed command.
        let [lo, hi] = halves(DEFAULT_DISPLAY);
        for command in [GET_HDR_STATUS, GET_BUFFER_MAX_RESOLUTION] {
            assert_ne!(rig.send(&[tipc_type(command), 2, lo, hi]), 0);
            assert_eq!(levels.take_from(Level::DEBUG), [Level::WARN], "{command}");
        }
        let all = [
            GET_HDR_STATUS,
            GET_DISPLAY_STATE_105,
            GET_BUFFER_MAX_RESOLUTION,
            GET_DISPLAY_STATE_107,
            GET_DISPLAY_STATE_354,
        ];
        for open in [false, true] {
            if open {
                let words = [tipc_type(OPEN_DISPLAY), 3, lo, hi, OPEN_DISPLAY_FLAGS];
                assert_eq!(rig.send(&words), 0);
                levels.take();
            }
            for command in all {
                if !open && [GET_HDR_STATUS, GET_BUFFER_MAX_RESOLUTION].contains(&command) {
                    continue;
                }
                assert_eq!(rig.send(&[tipc_type(command), 2, lo, hi]), 0);
                assert_eq!(
                    levels.take_from(Level::DEBUG),
                    Vec::<Level>::new(),
                    "{command}"
                );
            }
        }

        // 201: a bit at or past CLIENT_NOTIFICATIONS logs its own warning before the
        // framework's; a zero token with good bits, only the framework's.
        let register = |token: u64, retire: u32| {
            let [lo, hi] = halves(token);
            vec![tipc_type(REGISTER_CLIENT_STATE), 4, lo, hi, retire, 0]
        };
        assert_ne!(rig.send(&register(0, 1)), 0);
        assert_eq!(levels.take_from(Level::WARN), [Level::WARN]);
        assert_ne!(rig.send(&register(9, CLIENT_NOTIFICATIONS)), 0);
        assert_eq!(levels.take_from(Level::WARN), [Level::WARN, Level::WARN]);
        // 351: an encoding other than OUTPUT_ENCODING logs its own warning before the
        // framework's; a client not registered, only the framework's.
        assert_eq!(rig.send(&register(9, 1)), 0);
        levels.take();
        let encode = |token: u64, encoding: u64| {
            let [lo, hi] = halves(token);
            let [elo, ehi] = halves(encoding);
            vec![tipc_type(SET_CLIENT_OUTPUT_ENCODING), 4, lo, hi, elo, ehi]
        };
        assert_ne!(rig.send(&encode(10, OUTPUT_ENCODING)), 0);
        assert_eq!(levels.take_from(Level::WARN), [Level::WARN]);
        assert_ne!(rig.send(&encode(9, OUTPUT_ENCODING + 1)), 0);
        assert_eq!(levels.take_from(Level::WARN), [Level::WARN, Level::WARN]);

        // A full clock: 31, and an unregister (266) that would put a group back on it, each log
        // one error.
        assert_eq!(rig.send(&[tipc_type(CREATE_REQUEST_GROUP_EVENT), 1, 2]), 0);
        assert_eq!(rig.send(&monitor(REGISTER_HDR_STATUS_MONITOR, 2)), 0);
        let listening = shared.lock().vsync().listeners().count();
        for index in listening..MAX_LISTENERS {
            let filler = ObjectId::from_parts(u32::try_from(0x1000 + index).unwrap(), 9);
            shared.lock().listen_vsync(filler, HostNanos(0)).unwrap();
        }
        levels.take();
        // 31's error, then the framework's warning of the failed command.
        assert_ne!(rig.send(&[tipc_type(CREATE_REQUEST_GROUP_EVENT), 1, 3]), 0);
        assert_eq!(levels.take_from(Level::WARN), [Level::ERROR, Level::WARN]);
        assert_eq!(rig.send(&monitor(UNREGISTER_HDR_STATUS_MONITOR, 2)), 0);
        assert_eq!(levels.take_from(Level::WARN), [Level::ERROR]);

        // The root's 305: PR's flags log nothing above info, other flags one warning.
        let state = RootState::new().with_vi2_server(VI2);
        let root = root_server_for(state, UnknownPolicy::Strict).with_service_id(ROOT);
        let mut rig = Rig::over(root, ROOT);
        let open = |flags: u64| {
            let [lo, hi] = halves(flags);
            vec![0x141, 2, lo, hi]
        };
        // The bootstrap is not complete: each 305 warns of that (decision 7) too.
        assert_eq!(rig.send(&open(DISPLAY_SESSION_FLAGS[0])), 0);
        assert_eq!(levels.take_from(Level::WARN), [Level::WARN]);
        assert_eq!(rig.send(&open(DISPLAY_SESSION_FLAGS[0] + 1)), 0);
        assert_eq!(levels.take_from(Level::WARN), [Level::WARN, Level::WARN]);
    });
}
