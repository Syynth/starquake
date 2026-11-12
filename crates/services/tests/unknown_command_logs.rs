//! What the framework logs for unknown commands: once per command under auto-stub, every time
//! under strict (`docs/specs/services.md`, "Unknown commands").
//!
//! # Why this file holds exactly one test
//!
//! The logs are observed with a subscriber installed for the test's own thread
//! (`tracing::subscriber::with_default`). That is only reliable when no other thread in the
//! process logs from the same places. While at most one subscriber exists, `tracing-core`
//! (0.1.36, `callsite.rs`, `Dispatchers::rebuilder` and `Rebuilder::for_each`) decides whether a
//! log statement is enabled by asking the default subscriber of whichever thread reaches the
//! statement first, and caches the answer. Under plain `cargo test`, tests run as threads of one
//! process: another test reaching the framework's `warn!` first, with no subscriber of its own,
//! caches "disabled", and this test then counts fewer warnings than were logged. (It showed up
//! as 6 failing runs in 40 when these checks lived among the crate's unit tests; `cargo nextest`
//! never showed it, because it runs each test in its own process.)
//!
//! So the checks live in their own test binary, in a single test function that runs them one
//! after another: no other thread ever reaches those statements. Do not add a second `#[test]`
//! here; add to the one below.

use std::error::Error;
use std::sync::{Arc, Mutex};

use starquake_ipc::{BUFFER_SIZE, Body, CommandType, Message, RequestBuilder, SessionKind};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;
use starquake_services::commands;
use starquake_services::framework::fake::FakeKernel;
use starquake_services::framework::{
    Context, Handled, MAX_STUBBED_COMMANDS, Outcome, Reply, Request, Server, UNKNOWN_COMMAND_ID,
    UnknownPolicy,
};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};

/// What a check returns: nothing, or whatever stopped it.
type Check<T = ()> = Result<T, Box<dyn Error>>;

/// A service with one command.
#[derive(serde::Serialize)]
struct Echo;

commands! {
    impl Echo {
        0 => ping,
    }
    tipc {
        0 => ping,
    }
}

impl Echo {
    fn ping(_: &mut Self, _cx: &mut Context<'_>, _req: &Request<'_>, reply: &mut Reply) -> Outcome {
        reply.ok()
    }
}

/// The command `Echo` has, in both protocols.
const PING: u16 = 0;
/// A command `Echo` does not have, in either protocol.
const MISSING: u16 = 5;

/// Records the level of every event logged while it is the thread's default subscriber.
#[derive(Clone, Default)]
struct Levels(Arc<Mutex<Vec<Level>>>);

impl Levels {
    fn seen(&self) -> Vec<Level> {
        self.0.lock().map(|seen| seen.clone()).unwrap_or_default()
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

/// Runs `f` and returns how many warnings it logged. Fails if it logged anything else.
fn warnings_in(f: impl FnOnce() -> Check) -> Check<usize> {
    let levels = Levels::default();
    tracing::subscriber::with_default(levels.clone(), f)?;
    let seen = levels.seen();
    assert!(seen.iter().all(|level| *level == Level::WARN), "{seen:?}");
    Ok(seen.len())
}

struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

impl Rig {
    fn new(policy: UnknownPolicy) -> Check<Self> {
        let mut kernel = FakeKernel::new(0)?;
        let session = kernel.new_object();
        let mut server = Server::new("echo", policy);
        server.open(session, Box::new(Echo))?;
        Ok(Self {
            kernel,
            server,
            session,
        })
    }

    /// Sends a request and checks the reply's result.
    fn expect(&mut self, request: &RequestBuilder, expected: ResultCode) -> Check {
        let mut buffer = vec![0; BUFFER_SIZE];
        request.build(&mut buffer)?;
        let handled = self
            .server
            .handle(&mut self.kernel, self.session, &mut buffer)?;
        assert_eq!(handled, Handled::Replied);
        let reply = Message::parse_reply(&buffer, SessionKind::Session)?;
        let result = match reply.body {
            Body::CmifReply(reply) => reply.header.result,
            Body::TipcReply { result, .. } => result,
            other => unreachable!("not a reply: {other:?}"),
        };
        assert_eq!(ResultCode::from_raw(result), expected);
        Ok(())
    }
}

fn cmif(id: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, id)
}

fn auto_stub_logs_each_command_once() -> Check {
    let mut rig = Rig::new(UnknownPolicy::AutoStub)?;
    let missing = u32::from(MISSING);
    let warnings = warnings_in(|| {
        for _ in 0..3 {
            rig.expect(&cmif(missing), ResultCode::SUCCESS)?;
        }
        // Another id, and the same id in the other protocol, are other commands.
        for _ in 0..2 {
            rig.expect(&cmif(missing + 1), ResultCode::SUCCESS)?;
            rig.expect(&RequestBuilder::tipc(MISSING), ResultCode::SUCCESS)?;
        }
        // Commands the service has are not logged at all.
        rig.expect(&cmif(u32::from(PING)), ResultCode::SUCCESS)?;
        rig.expect(&RequestBuilder::tipc(PING), ResultCode::SUCCESS)
    })?;
    assert_eq!(warnings, 3);
    assert_eq!(rig.server.stubbed().count(), 3);
    Ok(())
}

fn strict_logs_every_time() -> Check {
    let mut rig = Rig::new(UnknownPolicy::Strict)?;
    let warnings = warnings_in(|| {
        for _ in 0..3 {
            rig.expect(&cmif(u32::from(MISSING)), UNKNOWN_COMMAND_ID)?;
        }
        rig.expect(&RequestBuilder::tipc(MISSING), UNKNOWN_COMMAND_ID)?;
        rig.expect(&cmif(u32::from(PING)), ResultCode::SUCCESS)
    })?;
    assert_eq!(warnings, 4);
    assert_eq!(rig.server.stubbed().count(), 0);
    Ok(())
}

fn a_full_stub_record_is_reported_once() -> Check {
    let mut rig = Rig::new(UnknownPolicy::AutoStub)?;
    let limit = u32::try_from(MAX_STUBBED_COMMANDS)?;
    // Exactly as many commands as the record holds: one warning each.
    let first = u32::from(MISSING);
    let warnings = warnings_in(|| {
        (first..first + limit).try_for_each(|id| rig.expect(&cmif(id), ResultCode::SUCCESS))
    })?;
    assert_eq!(warnings, MAX_STUBBED_COMMANDS);
    // Three more: still stubbed, and together they log the one "record is full" warning. A
    // command already recorded stays quiet.
    let warnings = warnings_in(|| {
        for id in first + limit..first + limit + 3 {
            rig.expect(&cmif(id), ResultCode::SUCCESS)?;
        }
        rig.expect(&cmif(first), ResultCode::SUCCESS)
    })?;
    assert_eq!(warnings, 1);
    assert_eq!(rig.server.stubbed().count(), MAX_STUBBED_COMMANDS);
    Ok(())
}

#[test]
fn unknown_commands_are_logged_as_the_policy_says() -> Check {
    auto_stub_logs_each_command_once()?;
    strict_logs_every_time()?;
    a_full_stub_record_is_reported_once()
}
