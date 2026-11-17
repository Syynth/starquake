//! The framework's one path for an environment failure, driven through a toy service whose
//! methods read [`Context::env`] the way `fsp-srv`'s and `time:u`'s do, over a fake kernel.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use starquake_ipc::{BUFFER_SIZE, Body, CommandType, Message, RequestBuilder, SessionKind};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;
use starquake_trace::{EnvCall, EnvError, HostIoReply, HostIoRequest, HostNanos};

use super::*;
use crate::commands;
use crate::framework::fake::FakeKernel;
use crate::framework::result::code;
use crate::framework::{Context, Handled, Outcome, Reply, Request, Server, UnknownPolicy};
use crate::fsp::worker::ReplayError;

/// The toy's command ids, named after its methods.
mod probe {
    pub(super) const NOW: u32 = 0;
    pub(super) const RANDOM: u32 = 1;
    pub(super) const HOST_IO: u32 = 2;
    pub(super) const IGNORE: u32 = 3;
    pub(super) const COUNT_TOLD: u32 = 4;
}

/// The host I/O request the toy sends: distinct non-zero values.
fn toy_request() -> HostIoRequest {
    HostIoRequest {
        channel: 0x0C4A,
        payload: vec![0x11, 0x22, 0x33],
    }
}

/// The toy's own module, for the result codes it names for a host failure.
const TOY_MODULE: u32 = 0x1F;
/// What the toy answers when the host fails its request (`HostIoReply::Failed`).
const TOY_HOST_FAILED: ResultCode = code(TOY_MODULE, 0x2A);
/// What the toy answers when nothing handles host I/O (`HostIoReply::Unavailable`).
const TOY_NO_HOST: ResultCode = code(TOY_MODULE, 0x2B);

thread_local! {
    /// How many stops a [`Counting`] halt has been told on this thread: what [`Probe`]'s
    /// `count_told` reads. One per test thread, so tests do not see each other's.
    static TOLD: Cell<u32> = const { Cell::new(0) };
}

/// A halt that counts the stops it is told on the telling thread, and keeps them.
#[derive(Default)]
struct Counting(Halts);

impl Halt for Counting {
    fn halt(&self, stop: Stop) {
        TOLD.set(TOLD.get() + 1);
        self.0.halt(stop);
    }
}

/// A service that reads the environment through [`Context::env`].
#[derive(Default, Serialize, Deserialize)]
struct Probe;

commands! {
    impl Probe {
        probe::NOW => now,
        probe::RANDOM => random,
        probe::HOST_IO => host_io,
        probe::IGNORE => ignore,
        probe::COUNT_TOLD => count_told,
    }
}

impl Probe {
    /// out: `u64` the clock's reading.
    fn now(_: &mut Self, cx: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        let reading = cx.env()?.now()?;
        reply.payload().write_u64(reading.0);
        reply.ok()
    }

    /// out: `u64` a random number.
    fn random(_: &mut Self, cx: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        let value = cx.env()?.random_u64()?;
        reply.payload().write_u64(value);
        reply.ok()
    }

    /// out: the host's bytes; a host failure is the toy's own result code.
    fn host_io(_: &mut Self, cx: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        match cx.env()?.host_io(&toy_request())? {
            HostIoReply::Data(bytes) => {
                reply.payload().write_bytes(&bytes);
                reply.ok()
            }
            HostIoReply::Failed(_) => Err(TOY_HOST_FAILED.into()),
            HostIoReply::Unavailable => Err(TOY_NO_HOST.into()),
        }
    }

    /// Reads a random number, drops whatever came back, and replies success.
    fn ignore(_: &mut Self, cx: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        drop(cx.env()?.random_u64());
        reply.ok()
    }

    /// Reads a random number; on a failure, replies success with `u32` the number of stops this
    /// thread's halt had been told when the method saw the failure.
    fn count_told(
        _: &mut Self,
        cx: &mut Context<'_>,
        _: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        if cx.env()?.random_u64().is_err() {
            reply.payload().write_u32(TOLD.get());
        }
        reply.ok()
    }
}

/// An environment that gives each call the answer it holds for it.
struct Fixed {
    now: Result<HostNanos, EnvError>,
    random: Result<u64, EnvError>,
    host_io: Result<HostIoReply, EnvError>,
}

impl Env for Fixed {
    fn now(&self) -> Result<HostNanos, EnvError> {
        self.now.clone()
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        self.random.clone()
    }

    fn host_io(&self, request: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        assert_eq!(request, &toy_request());
        self.host_io.clone()
    }
}

/// An error for each call, each distinct.
fn now_failed() -> EnvError {
    EnvError::Exhausted {
        index: 3,
        requested: EnvCall::Clock,
    }
}

fn random_failed() -> EnvError {
    EnvError::Mismatch {
        index: 5,
        requested: EnvCall::Random,
        recorded: EnvCall::HostIo,
    }
}

fn host_io_failed() -> EnvError {
    EnvError::HostIoRequestMismatch { index: 7 }
}

/// An environment that fails every call, each with its own error.
fn failing() -> Fixed {
    Fixed {
        now: Err(now_failed()),
        random: Err(random_failed()),
        host_io: Err(host_io_failed()),
    }
}

/// An environment that answers every call, each with a distinct value.
fn answering(host_io: HostIoReply) -> Fixed {
    Fixed {
        now: Ok(HostNanos(0x1234_5678_9ABC)),
        random: Ok(0x0FED_CBA9_8765_4321),
        host_io: Ok(host_io),
    }
}

/// A probe server named `name` over `env`, a fake kernel, one open session, and the halt.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
    halt: Arc<Counting>,
}

impl Rig {
    fn new(name: &str, env: Fixed) -> Self {
        TOLD.set(0);
        let halt = Arc::new(Counting::default());
        let told: Arc<dyn Halt> = Arc::<Counting>::clone(&halt);
        let mut server = Server::new(name, UnknownPolicy::Strict)
            .with_root(|| Box::new(Probe))
            .with_env(Arc::new(env), told);
        let mut kernel = FakeKernel::new(0).unwrap();
        let session = kernel.new_object();
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
            halt,
        }
    }

    /// Sends `command` with no payload; the reply's result and payload.
    fn send(&mut self, command: u32) -> (ResultCode, Vec<u8>) {
        let mut buffer = vec![0xA5; BUFFER_SIZE];
        RequestBuilder::cmif(CommandType::Request, command)
            .build(&mut buffer)
            .unwrap();
        let handled = self
            .server
            .handle(&mut self.kernel, self.session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        let message = Message::parse_reply(&buffer, SessionKind::Session).unwrap();
        match &message.body {
            Body::CmifReply(reply) => (
                ResultCode::from_raw(reply.header.result),
                reply.data.to_vec(),
            ),
            other => unreachable!("not a reply: {other:?}"),
        }
    }

    fn stops(&self) -> Vec<Stop> {
        self.halt.0.stops()
    }
}

/// The stop a server named `service` tells for `error`.
fn handler(service: &str, error: EnvError) -> Stop {
    Stop::Handler {
        service: service.to_owned(),
        error,
    }
}

#[test]
fn every_call_that_fails_tells_the_halt_once() {
    // Each of the three calls, on two servers with different names: the stop names the server
    // that read the environment, and carries exactly the error the call got. The request is
    // answered as one the server gave up on.
    let cases = [
        (probe::NOW, now_failed()),
        (probe::RANDOM, random_failed()),
        (probe::HOST_IO, host_io_failed()),
    ];
    for name in ["halt:a", "clock:b"] {
        for (command, error) in cases.clone() {
            let mut rig = Rig::new(name, failing());
            let (result, data) = rig.send(command);
            assert_eq!((result, data), (STOPPED, Vec::new()), "{name} {command}");
            assert_eq!(rig.stops(), [handler(name, error)], "{name} {command}");
        }
    }
    assert_eq!(STOPPED, DEFERRED_REPLY_DROPPED);
}

#[test]
fn answers_pass_through_and_tell_nothing() {
    let mut rig = Rig::new("halt:a", answering(HostIoReply::Data(vec![9, 8, 7])));
    let (result, data) = rig.send(probe::NOW);
    assert_eq!(result, ResultCode::SUCCESS);
    assert_eq!(data, 0x1234_5678_9ABC_u64.to_le_bytes());
    let (result, data) = rig.send(probe::RANDOM);
    assert_eq!(result, ResultCode::SUCCESS);
    assert_eq!(data, 0x0FED_CBA9_8765_4321_u64.to_le_bytes());
    let (result, data) = rig.send(probe::HOST_IO);
    assert_eq!(result, ResultCode::SUCCESS);
    assert_eq!(data[..3], [9, 8, 7]);
    assert_eq!(rig.stops(), []);
}

#[test]
fn a_host_failure_is_the_handler_s_and_stops_nothing() {
    // The host failing is an answer, not an environment failure: the handler names its result.
    for (answer, expected) in [
        (HostIoReply::Failed(4), TOY_HOST_FAILED),
        (HostIoReply::Unavailable, TOY_NO_HOST),
    ] {
        let mut rig = Rig::new("halt:a", answering(answer));
        assert_eq!(rig.send(probe::HOST_IO), (expected, Vec::new()));
        assert_eq!(rig.stops(), []);
    }
}

#[test]
fn a_failure_is_told_before_the_method_sees_it() {
    // The method reports how many stops had been told when it saw the error: one.
    let mut rig = Rig::new("halt:a", failing());
    let (result, data) = rig.send(probe::COUNT_TOLD);
    assert_eq!(result, ResultCode::SUCCESS);
    assert_eq!(data[..4], 1_u32.to_le_bytes());
    assert_eq!(rig.stops(), [handler("halt:a", random_failed())]);
}

#[test]
fn a_method_that_ignores_the_failure_still_stops_the_run() {
    let mut rig = Rig::new("halt:a", failing());
    assert_eq!(rig.send(probe::IGNORE), (ResultCode::SUCCESS, Vec::new()));
    assert_eq!(rig.stops(), [handler("halt:a", random_failed())]);
}

#[test]
fn every_environment_error_answers_as_a_request_given_up_on() {
    let errors = [
        now_failed(),
        random_failed(),
        host_io_failed(),
        EnvError::Unconsumed {
            index: 2,
            remaining: 4,
        },
    ];
    for error in errors {
        assert_eq!(Fault::from(error), Fault::Result(DEFERRED_REPLY_DROPPED));
    }
}

#[test]
fn each_stop_says_why_in_plain_words() {
    assert_eq!(
        handler("time:u", now_failed()).to_string(),
        "the environment failed time:u while it handled a request: recording exhausted at \
         record 3: the replay asked for Clock"
    );
    assert_eq!(
        Stop::Job {
            job: 6,
            error: host_io_failed(),
        }
        .to_string(),
        "the environment failed file-system job 6: record 7: the host I/O request differs from \
         the recorded one"
    );
    assert_eq!(
        Stop::Replay(ReplayError::Unrecorded { job: 9 }).to_string(),
        "the replay of fsp-srv's reads left its record: the replay ran file-system job 9, which \
         the recorded run never finished"
    );
}

#[test]
fn a_recording_halt_keeps_every_stop_in_order() {
    let halts = Arc::new(Halts::default());
    let shared = halts.shared();
    assert_eq!(halts.stops(), []);
    shared.halt(handler("a", now_failed()));
    shared.halt(handler("b", random_failed()));
    assert_eq!(
        halts.stops(),
        [handler("a", now_failed()), handler("b", random_failed())]
    );
}

// ----- who reads the environment -----------------------------------------------------------------

/// The non-test files of this crate that may name an environment type (`Env`, or a name ending
/// in it: `LiveEnv`, `ReplayEnv`), relative to `src/`, each with why. The module docs, "Who
/// reads the environment", say who reads one; this is where it is checked.
const NAMES_THE_ENVIRONMENT: [(&str, &str); 8] = [
    (
        "framework/context.rs",
        "Context::env hands a method the server's wrapped environment",
    ),
    ("framework/halt.rs", "the wrapper itself"),
    (
        "framework/server.rs",
        "Server::with_env, which wraps what it is given",
    ),
    (
        "framework/work.rs",
        "a job reads the environment its executor gives it",
    ),
    ("fsp.rs", "fsp::server hands its environment to with_env"),
    (
        "fsp/host.rs",
        "host::ask reads the environment it is given: a handler's or a job's",
    ),
    (
        "fsp/worker.rs",
        "fsp-srv's executors, which watch their jobs' environments themselves",
    ),
    (
        "time.rs",
        "time::server hands its environment to with_env, and the set-up reads Context::env",
    ),
];

/// Whether a file is test code: every test module of this crate is a `tests.rs` file or a file
/// in a `tests` folder.
fn is_test_code(relative: &Path) -> bool {
    relative.file_name().is_some_and(|name| name == "tests.rs")
        || relative
            .components()
            .any(|part| part.as_os_str() == "tests")
}

/// Whether `code` names an environment type: a word that is `Env` or ends with it.
fn names_an_environment(code: &str) -> bool {
    code.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|word| word.ends_with("Env"))
}

/// Every `.rs` file below `folder`, sorted.
fn sources(folder: &Path, found: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(folder)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            sources(&path, found);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            found.push(path);
        }
    }
}

/// The non-test files below `src` whose code (comment lines left out) names an environment type,
/// relative to `src`, with `/` between folders, sorted as strings.
fn files_naming_the_environment(src: &Path) -> Vec<String> {
    let mut files = Vec::new();
    sources(src, &mut files);
    let mut named: Vec<String> = files
        .into_iter()
        .filter_map(|path| {
            let relative = path.strip_prefix(src).unwrap().to_path_buf();
            if is_test_code(&relative) {
                return None;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let mentions = text
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .any(names_an_environment);
            let parts: Vec<String> = relative
                .components()
                .map(|part| part.as_os_str().to_string_lossy().into_owned())
                .collect();
            mentions.then(|| parts.join("/"))
        })
        .collect();
    named.sort();
    named
}

#[test]
fn only_the_listed_files_name_the_environment() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut listed: Vec<String> = NAMES_THE_ENVIRONMENT
        .iter()
        .map(|(file, _)| (*file).to_owned())
        .collect();
    listed.sort();
    assert_eq!(
        files_naming_the_environment(&src),
        listed,
        "a file names an environment type that `framework::halt` does not list (a new reader of \
         the environment, which must go through `Server::with_env`), or a listed one no longer \
         does: update the list and the module docs, \"Who reads the environment\""
    );
}

#[test]
fn the_search_sees_each_way_of_naming_an_environment() {
    // What the search counts, and what it leaves out: the words around it, a comment line, and a
    // name that only starts with `Env`.
    for code in [
        "fn f(env: &dyn Env) {}",
        "let e: Arc<dyn Env> = x;",
        "use starquake_trace::{Env, HostNanos};",
        "let live = LiveEnv::new(clock, 0);",
        "ReplayEnv::new(records)",
        "Env",
    ] {
        assert!(names_an_environment(code), "{code}");
    }
    for code in [
        "fn f() -> Result<(), EnvError> {}",
        "EnvCall::Clock",
        "EnvRecord::Random(1)",
        "let env = cx.env()?;",
        "Environment",
    ] {
        assert!(!names_an_environment(code), "{code}");
    }
    // Test code, by path.
    assert!(is_test_code(Path::new("time/tests.rs")));
    assert!(is_test_code(Path::new("fsp/tests/worker.rs")));
    assert!(is_test_code(Path::new("framework/halt/tests.rs")));
    assert!(!is_test_code(Path::new("time.rs")));
    assert!(!is_test_code(Path::new("fsp/worker.rs")));
    assert!(!is_test_code(Path::new("contests.rs")));
}

#[test]
fn the_search_finds_a_new_reader_and_skips_comments_and_tests() {
    // A scratch `src` folder for this test alone, removed at the end.
    let src = std::env::temp_dir().join(format!(
        "starquake-halt-scan-{}-{}",
        std::process::id(),
        line!()
    ));
    let write = |relative: &str, text: &str| {
        let path = src.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    write("a.rs", "fn f(env: &dyn Env) {}\n");
    write("b/c.rs", "use starquake_trace::LiveEnv;\n");
    write("b/d.rs", "//! Reads the `Env`.\nfn g() {}\n");
    write("b/tests.rs", "fn h(env: &dyn Env) {}\n");
    write("e/tests/f.rs", "fn i(env: &dyn Env) {}\n");
    write("g.txt", "Env\n");
    let found = files_naming_the_environment(&src);
    std::fs::remove_dir_all(&src).unwrap();
    assert_eq!(found, ["a.rs", "b/c.rs"]);
}
