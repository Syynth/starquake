//! `fsp-srv` driven through raw messages laid out as libnx lays them out
//! (`libnx/nx/source/services/fs.c`), over a fake kernel, a fake host behind the environment, and
//! an executor whose jobs the test runs itself.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use starquake_ipc::{
    BUFFER_SIZE, Body, BufferDescriptor, CommandType, Message, RequestBuilder, SessionKind,
    StaticDescriptor,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_trace::{EnvCall, EnvError, HostIoReply, HostIoRequest, HostNanos};

use super::host::{CHANNEL, HostFailure, HostRequest, Mount};
use super::*;
use crate::framework::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
use crate::framework::fake::FakeKernel;
use crate::framework::result::{INVALID_CMIF_REQUEST, NOT_SUPPORTED, UNKNOWN_COMMAND_ID};
use crate::framework::{
    Executor, Handled, Job, Kernel, Notify, SNAPSHOT_MAGIC, SNAPSHOT_VERSION, Session,
    SnapshotError,
};

use super::command::file::{GET_SIZE, READ};
use super::command::file_system::OPEN_FILE;
use super::command::fsp_srv::{OPEN_SD_CARD_FILE_SYSTEM, SET_CURRENT_PROCESS};

/// `IFile::Write`, which this subset leaves out, so it has no shipping constant
/// (`eden/src/core/hle/service/filesystem/fsp/fs_i_file.cpp`: `{1, D<&IFile::Write>, "Write"}`).
const WRITE: u32 = 1;

/// The process the caller is in.
const PROCESS_ID: u64 = 0x51;
/// Where the caller keeps its path, and its read buffer.
const PATH_AT: u64 = 0x0000_0010_0000_0000;
const DATA_AT: u64 = 0x0000_0020_0000_0000;
/// What fills the caller's memory and message buffer beforehand.
const STALE: u8 = 0xA5;

/// The file the tests read: every byte distinct, so a read from the wrong offset shows.
const HELLO: &[u8] = b"0123456789abcdefghijklmnopqrstuv";
const HELLO_PATH: &[u8] = b"/hello.txt";

/// A host entry.
#[derive(Debug, Clone)]
enum Entry {
    File(Vec<u8>),
    Folder,
}

/// How the fake host misbehaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mischief {
    None,
    /// Reads fail with this.
    FailReads(HostFailure),
    /// Reads return one byte more than asked for.
    Greedy,
}

/// A kind of host request, for [`FakeHost::forced`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Asked {
    Probe,
    Size,
    Read,
}

impl Asked {
    fn of(request: &HostRequest) -> Self {
        match request {
            HostRequest::Probe { .. } => Self::Probe,
            HostRequest::Size { .. } => Self::Size,
            HostRequest::Read { .. } => Self::Read,
        }
    }
}

/// The host behind the environment: an SD card in memory, answering the host file protocol.
struct FakeHost {
    card: Option<BTreeMap<Vec<u8>, Entry>>,
    mischief: Mischief,
    /// What the environment returns, in place of the card's answer, for every request of a kind.
    forced: Option<(Asked, Result<HostIoReply, EnvError>)>,
    asked: Vec<HostRequest>,
}

/// The environment the server is given: [`FakeHost`] on [`CHANNEL`].
struct FakeEnv(Mutex<FakeHost>);

impl FakeEnv {
    fn new(card: bool) -> Arc<Self> {
        let card = card.then(|| {
            BTreeMap::from([
                (HELLO_PATH.to_vec(), Entry::File(HELLO.to_vec())),
                (b"/dir".to_vec(), Entry::Folder),
            ])
        });
        Arc::new(Self(Mutex::new(FakeHost {
            card,
            mischief: Mischief::None,
            forced: None,
            asked: Vec::new(),
        })))
    }

    fn host(&self) -> std::sync::MutexGuard<'_, FakeHost> {
        self.0.lock().unwrap()
    }

    fn asked(&self) -> Vec<HostRequest> {
        self.host().asked.clone()
    }
}

impl FakeHost {
    fn answer(&mut self, request: &HostRequest) -> HostIoReply {
        let Some(card) = &self.card else {
            return HostFailure::NoDevice.reply();
        };
        match request {
            HostRequest::Probe { .. } => HostIoReply::Data(Vec::new()),
            HostRequest::Size { path, .. } => match card.get(path) {
                Some(Entry::File(bytes)) => {
                    HostIoReply::Data(u64::try_from(bytes.len()).unwrap().to_le_bytes().to_vec())
                }
                Some(Entry::Folder) => HostFailure::NotAFile.reply(),
                None => HostFailure::NotFound.reply(),
            },
            HostRequest::Read {
                path, offset, len, ..
            } => {
                if let Mischief::FailReads(failure) = self.mischief {
                    return failure.reply();
                }
                let Some(Entry::File(bytes)) = card.get(path) else {
                    return HostFailure::NotFound.reply();
                };
                let start = usize::try_from(*offset).unwrap().min(bytes.len());
                let end = start
                    .saturating_add(usize::try_from(*len).unwrap())
                    .min(bytes.len());
                let mut data = bytes[start..end].to_vec();
                if self.mischief == Mischief::Greedy {
                    data.push(0xEE);
                }
                HostIoReply::Data(data)
            }
        }
    }
}

impl Env for FakeEnv {
    fn now(&self) -> Result<HostNanos, EnvError> {
        Ok(HostNanos(0))
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        Ok(0)
    }

    fn host_io(&self, request: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        assert_eq!(request.channel, CHANNEL);
        let decoded = HostRequest::decode(&request.payload).unwrap();
        let mut host = self.host();
        host.asked.push(decoded.clone());
        match &host.forced {
            Some((kind, answer)) if *kind == Asked::of(&decoded) => answer.clone(),
            _ => Ok(host.answer(&decoded)),
        }
    }
}

/// An executor that keeps its jobs until the test runs them.
#[derive(Default)]
struct Jobs(Mutex<Vec<Job>>);

impl Executor for Jobs {
    fn submit(&self, job: Job) {
        self.0.lock().unwrap().push(job);
    }
}

impl Jobs {
    fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }

    /// Runs every job, oldest first, on this thread.
    fn run(&self) {
        let jobs = std::mem::take(&mut *self.0.lock().unwrap());
        for job in jobs {
            job();
        }
    }
}

/// Counts completions posted.
#[derive(Default)]
struct Wakes(AtomicUsize);

impl Notify for Wakes {
    fn completion_posted(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// An `fsp-srv` server, a kernel, the host, and one session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
    env: Arc<FakeEnv>,
    jobs: Arc<Jobs>,
    wakes: Arc<Wakes>,
}

/// A parsed reply.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Answered {
    result: ResultCode,
    data: Vec<u8>,
    moves: Vec<u32>,
    objects: Vec<u32>,
}

fn parse(buffer: &[u8], kind: SessionKind) -> Answered {
    let message = Message::parse_reply(buffer, kind).unwrap();
    let Body::CmifReply(reply) = &message.body else {
        unreachable!("not a CMIF reply: {:?}", message.body);
    };
    Answered {
        result: ResultCode::from_raw(reply.header.result),
        data: reply.data.to_vec(),
        moves: message.move_handles.clone(),
        objects: reply.object_ids.clone(),
    }
}

/// Where requests go: a session of its own, or an object of the rig's domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum To {
    Session(ObjectId),
    Object(u32),
}

impl Rig {
    fn with(card: bool, executor: bool) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.set_process_id(PROCESS_ID);
        let session = kernel.new_object();
        let env = FakeEnv::new(card);
        let jobs = Arc::new(Jobs::default());
        let wakes = Arc::new(Wakes::default());
        let env_handle: Arc<dyn Env> = Arc::<FakeEnv>::clone(&env);
        let notify: Arc<dyn Notify> = Arc::<Wakes>::clone(&wakes);
        let mut server = server(UnknownPolicy::Strict, env_handle).with_notify(notify);
        if executor {
            let executor: Arc<dyn Executor> = Arc::<Jobs>::clone(&jobs);
            server = server.with_executor(executor);
        }
        server.accept(session).unwrap();
        kernel.map(DATA_AT, vec![STALE; 0x40]);
        Self {
            kernel,
            server,
            session,
            env,
            jobs,
            wakes,
        }
    }

    fn new() -> Self {
        Self::with(true, true)
    }

    /// Sends a request and returns what the server did and the buffer.
    fn try_send(&mut self, session: ObjectId, request: &RequestBuilder) -> (Handled, Vec<u8>) {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        (handled.unwrap(), buffer)
    }

    /// Sends a request to `to` and returns its reply, which must come at once.
    fn send(
        &mut self,
        to: To,
        command: u32,
        build: impl FnOnce(RequestBuilder) -> RequestBuilder,
    ) -> Answered {
        let (session, request, kind) = self.frame(to, command, build);
        let (handled, buffer) = self.try_send(session, &request);
        assert_eq!(handled, Handled::Replied);
        parse(&buffer, kind)
    }

    fn frame(
        &self,
        to: To,
        command: u32,
        build: impl FnOnce(RequestBuilder) -> RequestBuilder,
    ) -> (ObjectId, RequestBuilder, SessionKind) {
        match to {
            To::Session(session) => (
                session,
                build(RequestBuilder::cmif(CommandType::Request, command)),
                SessionKind::Session,
            ),
            To::Object(object) => (
                self.session,
                build(RequestBuilder::domain(
                    CommandType::Request,
                    object,
                    command,
                )),
                SessionKind::Domain,
            ),
        }
    }

    /// Sends a request that must be deferred; the buffer comes back untouched.
    fn defer(
        &mut self,
        to: To,
        command: u32,
        build: impl FnOnce(RequestBuilder) -> RequestBuilder,
    ) {
        let (session, request, _) = self.frame(to, command, build);
        let mut expected = vec![STALE; BUFFER_SIZE];
        request.build(&mut expected).unwrap();
        let (handled, buffer) = self.try_send(session, &request);
        assert!(matches!(handled, Handled::Deferred(_)), "{handled:?}");
        assert_eq!(buffer, expected);
    }

    /// Delivers the oldest completed request and returns its reply.
    fn deliver(&mut self, kind: SessionKind) -> Answered {
        let completed = self.server.take_completed().unwrap();
        let mut buffer = vec![STALE; BUFFER_SIZE];
        self.server
            .deliver(&mut self.kernel, completed, &mut buffer)
            .unwrap();
        parse(&buffer, kind)
    }

    fn object_of(&self, handle: u32) -> ObjectId {
        self.kernel
            .resolve_handle(Handle::from_raw(handle))
            .unwrap()
    }

    /// Maps `path` as the caller's path buffer: `FS_MAX_PATH` bytes, zero after the path.
    fn map_path(&mut self, path: &[u8]) -> StaticDescriptor {
        let mut bytes = path.to_vec();
        bytes.resize(PATH_BUFFER_SIZE.max(path.len()), 0);
        let size = u16::try_from(bytes.len()).unwrap();
        self.kernel.map(PATH_AT, bytes);
        StaticDescriptor {
            counter: 0,
            address: PATH_AT,
            size,
        }
    }

    /// Converts the rig's session to a domain, and returns the root's object id.
    fn convert(&mut self) -> u32 {
        let convert = RequestBuilder::cmif(CommandType::Control, CONVERT_CURRENT_OBJECT_TO_DOMAIN);
        let (handled, buffer) = self.try_send(self.session, &convert);
        assert_eq!(handled, Handled::Replied);
        let reply = parse(&buffer, SessionKind::Session);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        u32::from_le_bytes(reply.data.try_into().unwrap())
    }

    /// `OpenSdCardFileSystem` on a plain session: the file system's session.
    fn open_sd(&mut self) -> ObjectId {
        let reply = self.send(To::Session(self.session), OPEN_SD_CARD_FILE_SYSTEM, |r| r);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        self.object_of(reply.moves[0])
    }

    /// `OpenFile` on a plain file-system session, read-only.
    fn open_file(&mut self, fs: ObjectId, path: &[u8]) -> Answered {
        self.open_file_with(fs, path, open_mode::READ)
    }

    fn open_file_with(&mut self, fs: ObjectId, path: &[u8], mode: u32) -> Answered {
        let descriptor = self.map_path(path);
        self.send(To::Session(fs), OPEN_FILE, |r| {
            r.data(&mode.to_le_bytes()).static_descriptor(descriptor)
        })
    }

    /// The session of [`HELLO`], opened on a fresh file system.
    fn hello(&mut self) -> ObjectId {
        let fs = self.open_sd();
        let reply = self.open_file(fs, HELLO_PATH);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        self.object_of(reply.moves[0])
    }
}

/// `fsFileRead`'s payload: `u32 option; u32 pad; s64 offset; u64 read_size;`.
fn read_payload(offset: i64, size: i64) -> Vec<u8> {
    [
        &0u32.to_le_bytes()[..],
        &[0; 4],
        &offset.to_le_bytes(),
        &size.to_le_bytes(),
    ]
    .concat()
}

/// A read request's buffer: `len` bytes at [`DATA_AT`] as a map-alias (B) descriptor.
fn room(len: u64) -> BufferDescriptor {
    BufferDescriptor {
        address: DATA_AT,
        size: len,
        flags: 0,
    }
}

fn read(offset: i64, size: i64, len: u64) -> impl FnOnce(RequestBuilder) -> RequestBuilder {
    move |r| {
        r.data(&read_payload(offset, size))
            .receive_buffer(room(len))
    }
}

/// A reply holding a result and nothing else.
fn only(result: ResultCode) -> Answered {
    Answered {
        result,
        data: Vec::new(),
        moves: Vec::new(),
        objects: Vec::new(),
    }
}

/// A successful reply whose payload is one `s64`.
fn s64(value: i64) -> Answered {
    Answered {
        data: value.to_le_bytes().to_vec(),
        ..only(ResultCode::SUCCESS)
    }
}

#[test]
fn a_file_is_opened_read_and_closed_on_plain_sessions() {
    let mut rig = Rig::new();
    let root = To::Session(rig.session);
    let set = rig.send(root, SET_CURRENT_PROCESS, |r| {
        r.send_pid(0).data(&0u64.to_le_bytes())
    });
    assert_eq!(set, only(ResultCode::SUCCESS));

    let fs = rig.open_sd();
    let opened = rig.open_file(fs, HELLO_PATH);
    assert_eq!(
        (opened.result, opened.moves.len()),
        (ResultCode::SUCCESS, 1)
    );
    let file = rig.object_of(opened.moves[0]);
    assert_eq!(rig.server.session_count(), 3);

    let size = rig.send(To::Session(file), GET_SIZE, |r| r);
    assert_eq!(size, s64(i64::try_from(HELLO.len()).unwrap()));

    // Five bytes from offset 7 into a 16-byte buffer: deferred, and the job not run yet.
    rig.defer(To::Session(file), READ, read(7, 5, 16));
    assert_eq!(rig.jobs.len(), 1);
    assert!(rig.server.take_completed().is_none());
    assert_eq!(rig.kernel.memory(DATA_AT, 16).unwrap(), [STALE; 16]);
    rig.jobs.run();
    assert_eq!(rig.wakes.0.load(Ordering::SeqCst), 1);
    // Nothing reaches the caller's memory until the reply is delivered.
    assert_eq!(rig.kernel.memory(DATA_AT, 16).unwrap(), [STALE; 16]);
    assert_eq!(rig.deliver(SessionKind::Session), s64(5));
    let mut expected = HELLO[7..12].to_vec();
    expected.resize(16, STALE);
    assert_eq!(rig.kernel.memory(DATA_AT, 16).unwrap(), expected);

    // What the host was asked, in order.
    let path = HELLO_PATH.to_vec();
    let mount = Mount::SdCard;
    assert_eq!(
        rig.env.asked(),
        [
            HostRequest::Probe { mount },
            HostRequest::Size {
                mount,
                path: path.clone()
            },
            HostRequest::Size {
                mount,
                path: path.clone()
            },
            HostRequest::Read {
                mount,
                path,
                offset: 7,
                len: 5
            },
        ]
    );

    // Closing the file drops its interface.
    let interfaces = rig.server.interface_count();
    let (handled, _) = rig.try_send(file, &RequestBuilder::close());
    assert_eq!(handled, Handled::Closed);
    assert_eq!(rig.server.session_count(), 2);
    assert_eq!(rig.server.interface_count(), interfaces - 1);
}

#[test]
fn a_file_is_opened_read_and_closed_in_a_domain() {
    let mut rig = Rig::new();
    let root = To::Object(rig.convert());
    let set = rig.send(root, SET_CURRENT_PROCESS, |r| {
        r.send_pid(0).data(&0u64.to_le_bytes())
    });
    assert_eq!(set, only(ResultCode::SUCCESS));

    let opened = rig.send(root, OPEN_SD_CARD_FILE_SYSTEM, |r| r);
    assert_eq!(
        (opened.result, opened.moves.len()),
        (ResultCode::SUCCESS, 0)
    );
    let [fs] = opened.objects[..] else {
        unreachable!("one object: {opened:?}")
    };
    let descriptor = rig.map_path(HELLO_PATH);
    let opened = rig.send(To::Object(fs), OPEN_FILE, |r| {
        r.data(&open_mode::READ.to_le_bytes())
            .static_descriptor(descriptor)
    });
    assert_eq!(opened.result, ResultCode::SUCCESS);
    let [file] = opened.objects[..] else {
        unreachable!("one object: {opened:?}")
    };
    assert_ne!(fs, file);
    // Every interface is an object of the one session.
    assert_eq!(rig.server.session_count(), 1);
    assert_eq!(rig.kernel.sessions(), []);

    let size = rig.send(To::Object(file), GET_SIZE, |r| r);
    assert_eq!(size, s64(i64::try_from(HELLO.len()).unwrap()));

    rig.defer(To::Object(file), READ, read(0, 4, 4));
    rig.jobs.run();
    assert_eq!(rig.deliver(SessionKind::Domain), s64(4));
    assert_eq!(rig.kernel.memory(DATA_AT, 4).unwrap(), &HELLO[..4]);

    let interfaces = rig.server.interface_count();
    let (handled, _) = rig.try_send(rig.session, &RequestBuilder::domain_close(file));
    assert_eq!(handled, Handled::Replied);
    assert_eq!(rig.server.interface_count(), interfaces - 1);
    // The closed file answers nothing any more.
    let gone = rig.send(To::Object(file), GET_SIZE, |r| r);
    assert_eq!(gone.result, crate::framework::TARGET_NOT_FOUND);
}

#[test]
fn set_current_process_accepts_the_callers_process() {
    let mut rig = Rig::new();
    // The kernel attaches the caller's id; the placeholder in the payload is not read, so a
    // request without one works too.
    for payload in [&7u64.to_le_bytes()[..], &[]] {
        let reply = rig.send(To::Session(rig.session), SET_CURRENT_PROCESS, |r| {
            r.send_pid(0).data(payload)
        });
        assert_eq!(reply, only(ResultCode::SUCCESS));
    }
    assert_eq!(rig.env.asked(), []);
}

#[test]
fn set_current_process_without_a_process_id_is_target_not_found() {
    let mut rig = Rig::new();
    let reply = rig.send(To::Session(rig.session), SET_CURRENT_PROCESS, |r| {
        r.data(&0u64.to_le_bytes())
    });
    assert_eq!(reply, only(result::TARGET_NOT_FOUND));
}

#[test]
fn without_an_sd_card_opening_it_fails_with_no_device() {
    let mut rig = Rig::with(false, true);
    let reply = rig.send(To::Session(rig.session), OPEN_SD_CARD_FILE_SYSTEM, |r| r);
    assert_eq!(reply, only(result::PORT_SD_CARD_NO_DEVICE));
    assert_eq!(rig.server.session_count(), 1);
    assert_eq!(
        rig.env.asked(),
        [HostRequest::Probe {
            mount: Mount::SdCard
        }]
    );
}

/// An environment with no host I/O handler: what a run with no folder configured has.
struct NoHost;

impl Env for NoHost {
    fn now(&self) -> Result<HostNanos, EnvError> {
        Ok(HostNanos(0))
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        Ok(0)
    }

    fn host_io(&self, _: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        Ok(HostIoReply::Unavailable)
    }
}

#[test]
fn a_run_with_no_host_folder_has_no_sd_card() {
    let mut kernel = FakeKernel::new(0).unwrap();
    let session = kernel.new_object();
    let mut server = server(UnknownPolicy::Strict, Arc::new(NoHost));
    server.accept(session).unwrap();
    let mut buffer = vec![STALE; BUFFER_SIZE];
    let request = RequestBuilder::cmif(CommandType::Request, OPEN_SD_CARD_FILE_SYSTEM);
    request.build(&mut buffer).unwrap();
    let handled = server.handle(&mut kernel, session, &mut buffer);
    assert_eq!(handled, Ok(Handled::Replied));
    let reply = parse(&buffer, SessionKind::Session);
    assert_eq!(reply, only(result::PORT_SD_CARD_NO_DEVICE));
}

#[test]
fn a_server_without_an_environment_fails_what_needs_the_host() {
    let mut kernel = FakeKernel::new(0).unwrap();
    let session = kernel.new_object();
    let mut server = Server::new(NAME, UnknownPolicy::Strict)
        .with_root(|| Box::new(FspSrv))
        .with_interfaces(interfaces());
    server.accept(session).unwrap();
    let mut buffer = vec![STALE; BUFFER_SIZE];
    let request = RequestBuilder::cmif(CommandType::Request, OPEN_SD_CARD_FILE_SYSTEM);
    request.build(&mut buffer).unwrap();
    server.handle(&mut kernel, session, &mut buffer).unwrap();
    assert_eq!(parse(&buffer, SessionKind::Session), only(NOT_SUPPORTED));
}

#[test]
fn open_file_normalises_the_path_before_it_asks_the_host() {
    let mut rig = Rig::new();
    let fs = rig.open_sd();
    for path in [&b"//hello.txt"[..], b"/dir/../hello.txt", b"/./hello.txt/"] {
        let reply = rig.open_file(fs, path);
        assert_eq!(
            reply.result,
            ResultCode::SUCCESS,
            "{:?}",
            String::from_utf8_lossy(path)
        );
    }
    let asked = rig.env.asked();
    let sizes = asked.iter().filter(|request| {
        **request
            == HostRequest::Size {
                mount: Mount::SdCard,
                path: HELLO_PATH.to_vec(),
            }
    });
    assert_eq!(sizes.count(), 3);
}

#[test]
fn open_file_of_nothing_or_a_folder_is_path_not_found() {
    let mut rig = Rig::new();
    let fs = rig.open_sd();
    for path in [&b"/missing.txt"[..], b"/dir"] {
        assert_eq!(rig.open_file(fs, path), only(result::PATH_NOT_FOUND));
    }
}

/// Opens `path` on a fresh file system and returns the reply and whether the host was asked
/// about the path.
fn open_and_ask(path: &[u8]) -> (Answered, bool) {
    let mut rig = Rig::new();
    let fs = rig.open_sd();
    let before = rig.env.asked().len();
    let reply = rig.open_file(fs, path);
    (reply, rig.env.asked().len() > before)
}

#[test]
fn a_relative_path_is_refused() {
    // Eden's `SanitizePath` would find the file.
    assert_eq!(
        open_and_ask(b"hello.txt"),
        (only(result::INVALID_PATH_FORMAT), false)
    );
}

#[test]
fn a_path_above_the_root_is_unobtainable() {
    assert_eq!(
        open_and_ask(b"/../hello.txt"),
        (only(result::DIRECTORY_UNOBTAINABLE), false)
    );
}

#[test]
fn a_path_with_an_invalid_character_is_refused() {
    assert_eq!(
        open_and_ask(b"/hello.txt:"),
        (only(result::INVALID_CHARACTER), false)
    );
}

#[test]
fn a_path_with_no_terminator_ends_at_its_struct() {
    // The longest path that fits, with its terminator as the struct's last byte: it is asked
    // of the host whole.
    let mut longest = vec![b'/'];
    longest.resize(path::ENTRY_NAME_LENGTH_MAX, b'a');
    let mut rig = Rig::new();
    let fs = rig.open_sd();
    assert_eq!(rig.open_file(fs, &longest), only(result::PATH_NOT_FOUND));
    let last = rig.env.asked().pop();
    let asked = HostRequest::Size {
        mount: Mount::SdCard,
        path: longest.clone(),
    };
    assert_eq!(last, Some(asked));

    // One byte more fills the struct with no zero byte: the path ends there and is too long,
    // whatever follows in a larger buffer.
    longest.push(b'a');
    assert_eq!(longest.len(), PATH_BUFFER_SIZE);
    assert_eq!(open_and_ask(&longest), (only(result::TOO_LONG_PATH), false));
    let mut beyond = longest.clone();
    beyond.extend_from_slice(b"\0\0\0\0");
    assert_eq!(open_and_ask(&beyond), (only(result::TOO_LONG_PATH), false));

    // A struct full of a path that names the file, then more in a larger buffer that would name
    // another: only the struct is read.
    let mut full = HELLO_PATH.to_vec();
    full.resize(PATH_BUFFER_SIZE, b'/');
    let mut rig = Rig::new();
    let fs = rig.open_sd();
    let mut more = full.clone();
    more.extend_from_slice(b"x\0");
    assert_eq!(rig.open_file(fs, &more).result, ResultCode::SUCCESS);
    let asked = HostRequest::Size {
        mount: Mount::SdCard,
        path: HELLO_PATH.to_vec(),
    };
    assert_eq!(rig.env.asked().pop(), Some(asked));
}

#[test]
fn a_short_path_buffer_is_read_to_its_end() {
    // A buffer of exactly the path, with no terminator in it.
    let mut rig = Rig::new();
    let fs = rig.open_sd();
    rig.kernel.map(PATH_AT, HELLO_PATH.to_vec());
    let descriptor = StaticDescriptor {
        counter: 0,
        address: PATH_AT,
        size: u16::try_from(HELLO_PATH.len()).unwrap(),
    };
    let reply = rig.send(To::Session(fs), OPEN_FILE, |r| {
        r.data(&open_mode::READ.to_le_bytes())
            .static_descriptor(descriptor)
    });
    assert_eq!(reply.result, ResultCode::SUCCESS);
}

#[test]
fn open_file_without_a_path_is_an_invalid_request() {
    let mut rig = Rig::new();
    let fs = rig.open_sd();
    let reply = rig.send(To::Session(fs), OPEN_FILE, |r| {
        r.data(&open_mode::READ.to_le_bytes())
    });
    assert_eq!(reply, only(INVALID_CMIF_REQUEST));
    // And without its mode.
    let descriptor = rig.map_path(HELLO_PATH);
    let reply = rig.send(To::Session(fs), OPEN_FILE, |r| {
        r.static_descriptor(descriptor)
    });
    assert_eq!(reply, only(INVALID_CMIF_REQUEST));
}

/// `OpenFile` declares its path a pointer (X) buffer, as Eden's signature does. The same path as
/// a map-alias (A) descriptor, in place of the X descriptor or as well as it, is refused before
/// the command runs, and the host is not asked.
#[test]
fn open_file_refuses_a_map_alias_path() {
    let mut rig = Rig::new();
    let fs = rig.open_sd();
    let pointer = rig.map_path(HELLO_PATH);
    let alias = BufferDescriptor {
        address: pointer.address,
        size: u64::from(pointer.size),
        flags: 0,
    };
    let before = rig.env.asked().len();
    let mode = open_mode::READ.to_le_bytes();
    let instead = rig.send(To::Session(fs), OPEN_FILE, |r| {
        r.data(&mode).send_buffer(alias)
    });
    assert_eq!(instead, only(INVALID_CMIF_REQUEST));
    let as_well = rig.send(To::Session(fs), OPEN_FILE, |r| {
        r.data(&mode).static_descriptor(pointer).send_buffer(alias)
    });
    assert_eq!(as_well, only(INVALID_CMIF_REQUEST));
    assert_eq!(rig.env.asked().len(), before);
    // The X descriptor alone opens the file.
    assert_eq!(rig.open_file(fs, HELLO_PATH).result, ResultCode::SUCCESS);
}

/// The framework's check of the declared buffers comes before the command's own checks, as on
/// the console: a bad mode and no path at once is the missing path's error.
#[test]
fn the_declared_buffers_are_checked_before_the_open_mode() {
    let mut rig = Rig::new();
    let fs = rig.open_sd();
    let bad_mode = open_mode::WRITE.to_le_bytes();
    let reply = rig.send(To::Session(fs), OPEN_FILE, |r| r.data(&bad_mode));
    assert_eq!(reply, only(INVALID_CMIF_REQUEST));
    // With the path, the same mode is the mode's error.
    assert_eq!(
        rig.open_file_with(fs, HELLO_PATH, open_mode::WRITE),
        only(result::INVALID_OPEN_MODE)
    );
}

/// `Read` declares its buffer a map-alias (B) buffer, as Eden's signature does. A pointer (C)
/// buffer in its place is refused before the command runs: nothing is deferred or written.
#[test]
fn read_refuses_a_pointer_buffer() {
    use starquake_ipc::ReceiveDescriptor;
    let mut rig = Rig::new();
    let file = rig.hello();
    let len = HELLO.len();
    rig.kernel.map(DATA_AT, vec![STALE; len]);
    let pointer = ReceiveDescriptor {
        address: DATA_AT,
        size: u16::try_from(len).unwrap(),
    };
    let size = i64::try_from(len).unwrap();
    let reply = rig.send(To::Session(file), READ, |r| {
        r.data(&read_payload(0, size)).receive_descriptor(pointer)
    });
    assert_eq!(reply, only(INVALID_CMIF_REQUEST));
    assert_eq!(rig.kernel.memory(DATA_AT, len).unwrap(), vec![STALE; len]);
    // A B descriptor beside the C descriptor is what `Read` declares, and it is read into.
    let len_u64 = u64::try_from(len).unwrap();
    rig.defer(To::Session(file), READ, |r| {
        read(0, size, len_u64)(r).receive_descriptor(pointer)
    });
}

#[test]
fn open_file_checks_eden_s_modes_then_read_only() {
    use open_mode::{ALL, ALLOW_APPEND, READ, READ_WRITE, WRITE};
    // Eden's checks: no read or write bit, and a bit outside `All`.
    let beyond = (ALL + 1) | READ;
    // The read-only file system's: anything but exactly read.
    let refused = [
        0,
        ALLOW_APPEND,
        beyond,
        u32::MAX,
        WRITE,
        READ_WRITE,
        READ | ALLOW_APPEND,
        ALL,
    ];
    for mode in refused {
        let mut rig = Rig::new();
        let fs = rig.open_sd();
        let before = rig.env.asked().len();
        assert_eq!(
            rig.open_file_with(fs, HELLO_PATH, mode),
            only(result::INVALID_OPEN_MODE),
            "{mode:#x}"
        );
        assert_eq!(rig.env.asked().len(), before, "{mode:#x}");
    }
    // The mode is checked before the path.
    let mut rig = Rig::new();
    let fs = rig.open_sd();
    let reply = rig.open_file_with(fs, b"relative", WRITE);
    assert_eq!(reply, only(result::INVALID_OPEN_MODE));
    let reply = rig.open_file_with(fs, b"relative", READ);
    assert_eq!(reply, only(result::INVALID_PATH_FORMAT));
}

#[test]
fn get_size_asks_the_host_each_time() {
    let mut rig = Rig::new();
    let file = rig.hello();
    let longer = [HELLO, b"!!"].concat();
    let longer_len = i64::try_from(longer.len()).unwrap();
    rig.env
        .host()
        .card
        .as_mut()
        .unwrap()
        .insert(HELLO_PATH.to_vec(), Entry::File(longer));
    assert_eq!(
        rig.send(To::Session(file), GET_SIZE, |r| r),
        s64(longer_len)
    );
    // Gone from the host: not found.
    rig.env.host().card.as_mut().unwrap().clear();
    let reply = rig.send(To::Session(file), GET_SIZE, |r| r);
    assert_eq!(reply, only(result::PATH_NOT_FOUND));
}

#[test]
fn read_checks_the_offset_before_the_size() {
    let mut rig = Rig::new();
    let file = To::Session(rig.hello());
    assert_eq!(
        rig.send(file, READ, read(-1, -1, 16)),
        only(result::INVALID_OFFSET)
    );
    assert_eq!(
        rig.send(file, READ, read(i64::MIN, 1, 16)),
        only(result::INVALID_OFFSET)
    );
    assert_eq!(
        rig.send(file, READ, read(0, -1, 16)),
        only(result::INVALID_SIZE)
    );
    assert_eq!(rig.jobs.len(), 0);
}

#[test]
fn read_checks_size_against_the_buffer() {
    let mut rig = Rig::new();
    let file = To::Session(rig.hello());
    // One byte more than the buffer holds; then exactly what it holds.
    assert_eq!(
        rig.send(file, READ, read(0, 9, 8)),
        only(result::INVALID_SIZE)
    );
    assert_eq!(rig.jobs.len(), 0);
    rig.defer(file, READ, read(0, 8, 8));
    rig.jobs.run();
    assert_eq!(rig.deliver(SessionKind::Session), s64(8));
    assert_eq!(
        rig.kernel.memory(DATA_AT, 9).unwrap(),
        [&HELLO[..8], &[STALE]].concat()
    );
}

#[test]
fn a_read_of_nothing_answers_at_once() {
    let mut rig = Rig::new();
    let file = To::Session(rig.hello());
    assert_eq!(rig.send(file, READ, read(3, 0, 16)), s64(0));
    // An empty buffer takes a read of nothing too; and the offset is not compared with the file.
    assert_eq!(rig.send(file, READ, read(i64::MAX, 0, 0)), s64(0));
    assert_eq!(rig.jobs.len(), 0);
    // Eden answers a negative offset with nothing read; the console's check comes first here.
    assert_eq!(
        rig.send(file, READ, read(-1, 0, 16)),
        only(result::INVALID_OFFSET)
    );
}

#[test]
fn a_read_whose_end_overflows_is_out_of_range() {
    let mut rig = Rig::new();
    let file = To::Session(rig.hello());
    // offset + size is one past `i64::MAX`.
    let reply = rig.send(file, READ, read(i64::MAX - 2, 3, 16));
    assert_eq!(reply, only(result::OUT_OF_RANGE));
    assert_eq!(rig.jobs.len(), 0);
    // Ending exactly at `i64::MAX` is a read past the end of the file: nothing.
    rig.defer(file, READ, read(i64::MAX - 3, 3, 16));
    rig.jobs.run();
    assert_eq!(rig.deliver(SessionKind::Session), s64(0));
}

#[test]
fn a_read_past_the_end_returns_no_bytes_and_one_across_it_is_short() {
    let mut rig = Rig::new();
    let file = To::Session(rig.hello());
    let len = i64::try_from(HELLO.len()).unwrap();
    rig.defer(file, READ, read(len + 10, 8, 16));
    rig.jobs.run();
    assert_eq!(rig.deliver(SessionKind::Session), s64(0));
    assert_eq!(rig.kernel.memory(DATA_AT, 16).unwrap(), [STALE; 16]);

    rig.defer(file, READ, read(len - 3, 8, 16));
    rig.jobs.run();
    assert_eq!(rig.deliver(SessionKind::Session), s64(3));
    let mut expected = HELLO[HELLO.len() - 3..].to_vec();
    expected.resize(16, STALE);
    assert_eq!(rig.kernel.memory(DATA_AT, 16).unwrap(), expected);
}

#[test]
fn a_read_without_its_buffer_or_payload_is_an_invalid_request() {
    let mut rig = Rig::new();
    let file = To::Session(rig.hello());
    let no_buffer = rig.send(file, READ, |r| r.data(&read_payload(0, 1)));
    assert_eq!(no_buffer, only(INVALID_CMIF_REQUEST));
    // The size is the last field: a payload that stops before it.
    let payload = read_payload(0, 1);
    let short = rig.send(file, READ, |r| {
        r.data(&payload[..payload.len() - 8])
            .receive_buffer(room(16))
    });
    assert_eq!(short, only(INVALID_CMIF_REQUEST));
}

#[test]
fn a_host_failure_during_a_read_fails_the_read() {
    let cases = [
        (HostFailure::NotFound, result::PATH_NOT_FOUND),
        (HostFailure::NotAFile, result::PATH_NOT_FOUND),
        (HostFailure::NoDevice, result::PORT_SD_CARD_NO_DEVICE),
        (HostFailure::Io, result::HOST_IO_FAILED),
        (HostFailure::BadRequest, result::HOST_IO_FAILED),
    ];
    for (failure, code) in cases {
        let mut rig = Rig::new();
        let file = To::Session(rig.hello());
        rig.env.host().mischief = Mischief::FailReads(failure);
        rig.defer(file, READ, read(0, 4, 16));
        rig.jobs.run();
        assert_eq!(rig.deliver(SessionKind::Session), only(code), "{failure:?}");
        assert_eq!(rig.kernel.memory(DATA_AT, 16).unwrap(), [STALE; 16]);
    }
}

#[test]
fn a_host_that_answers_more_than_asked_fails_the_read() {
    let mut rig = Rig::new();
    let file = To::Session(rig.hello());
    rig.env.host().mischief = Mischief::Greedy;
    rig.defer(file, READ, read(0, 4, 16));
    rig.jobs.run();
    assert_eq!(
        rig.deliver(SessionKind::Session),
        only(result::HOST_IO_FAILED)
    );
    assert_eq!(rig.kernel.memory(DATA_AT, 16).unwrap(), [STALE; 16]);
}

#[test]
fn a_read_into_memory_the_caller_does_not_have_is_the_kernel_s_error() {
    let mut rig = Rig::new();
    let file = To::Session(rig.hello());
    // The buffer runs past the 0x40 mapped bytes.
    rig.defer(file, READ, |r| {
        r.data(&read_payload(0, 0x20))
            .receive_buffer(BufferDescriptor {
                address: DATA_AT + 0x30,
                size: 0x20,
                flags: 0,
            })
    });
    rig.jobs.run();
    let reply = rig.deliver(SessionKind::Session);
    assert_eq!(reply, only(KernelError::InvalidCurrentMemory.code()));
    assert_eq!(rig.kernel.memory(DATA_AT, 0x40).unwrap(), [STALE; 0x40]);
}

#[test]
fn a_server_without_an_executor_cannot_read() {
    let mut rig = Rig::with(true, false);
    let file = To::Session(rig.hello());
    let reply = rig.send(file, READ, read(0, 4, 16));
    assert_eq!(reply, only(NOT_SUPPORTED));
    // The token the read would have taken was never taken.
    assert!(rig.server.take_completed().is_none());
    assert_eq!(rig.server.pending().count(), 0);
}

#[test]
fn a_dropped_read_job_answers_the_caller() {
    let mut rig = Rig::new();
    let file = To::Session(rig.hello());
    rig.defer(file, READ, read(0, 4, 16));
    drop(std::mem::take(&mut *rig.jobs.0.lock().unwrap()));
    let reply = rig.deliver(SessionKind::Session);
    assert_eq!(reply, only(crate::framework::DEFERRED_REPLY_DROPPED));
}

#[test]
fn commands_this_subset_leaves_out_are_unknown() {
    let mut rig = Rig::new();
    let file = To::Session(rig.hello());
    assert_eq!(rig.send(file, WRITE, |r| r), only(UNKNOWN_COMMAND_ID));
    // `OpenFileSystem` (0), which Eden leaves out too.
    let root = To::Session(rig.session);
    assert_eq!(rig.send(root, 0, |r| r), only(UNKNOWN_COMMAND_ID));
}

#[test]
fn the_command_tables_name_eden_s_commands() {
    use crate::framework::Protocol::Cmif;
    let root = FspSrv;
    let names =
        [SET_CURRENT_PROCESS, OPEN_SD_CARD_FILE_SYSTEM].map(|id| root.commands().name(Cmif, id));
    assert_eq!(
        names,
        [
            Some("set_current_process"),
            Some("open_sd_card_file_system")
        ]
    );
    let fs = IFileSystem {
        mount: Mount::SdCard,
    };
    assert_eq!(fs.commands().name(Cmif, OPEN_FILE), Some("open_file"));
    let file = IFile {
        mount: Mount::SdCard,
        path: Vec::new(),
    };
    let names = [READ, GET_SIZE].map(|id| file.commands().name(Cmif, id));
    assert_eq!(names, [Some("read"), Some("get_size")]);
    assert_eq!(
        (root.name(), fs.name(), file.name()),
        ("FspSrv", "IFileSystem", "IFile")
    );
}

#[test]
fn an_open_file_survives_a_snapshot() {
    let mut rig = Rig::new();
    let file = rig.hello();
    let saved = rig.server.save().unwrap();

    let env: Arc<dyn Env> = Arc::<FakeEnv>::clone(&rig.env);
    let mut restored = server(UnknownPolicy::Strict, env);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.session_count(), rig.server.session_count());
    rig.server = restored;
    let size = rig.send(To::Session(file), GET_SIZE, |r| r);
    assert_eq!(size, s64(i64::try_from(HELLO.len()).unwrap()));
}

impl Rig {
    /// Makes the environment answer every request of `kind` with `answer`.
    fn force(&self, kind: Asked, answer: Result<HostIoReply, EnvError>) {
        self.env.host().forced = Some((kind, answer));
    }
}

/// `OpenFile` of [`HELLO`] on a fresh file system, and `GetSize` of [`HELLO`] opened before, when
/// the environment answers every size request with `answer`. A failed `OpenFile` must leave the
/// server's sessions as they were.
fn sized_by(answer: &Result<HostIoReply, EnvError>) -> (Answered, Answered) {
    let mut rig = Rig::new();
    let file = rig.hello();
    let fs = rig.open_sd();
    let sessions = rig.server.session_count();
    rig.force(Asked::Size, answer.clone());
    let opened = rig.open_file(fs, HELLO_PATH);
    if opened.result != ResultCode::SUCCESS {
        assert_eq!(rig.server.session_count(), sessions);
    }
    let size = rig.send(To::Session(file), GET_SIZE, |r| r);
    (opened, size)
}

/// A size answer as the host writes one.
fn size_answer(size: u64) -> HostIoReply {
    HostIoReply::Data(size.to_le_bytes().to_vec())
}

#[test]
fn a_size_answer_of_other_than_eight_bytes_fails() {
    let width = size_of::<u64>();
    // A size of 4, one byte short and one byte long: padded or cut to 8 bytes, either would read
    // as 4.
    let mut short = 4u64.to_le_bytes().to_vec();
    short.truncate(width - 1);
    let mut long = 4u64.to_le_bytes().to_vec();
    long.push(0);
    for bytes in [short, long] {
        let failed = only(result::HOST_IO_FAILED);
        let answer = Ok(HostIoReply::Data(bytes.clone()));
        assert_eq!(sized_by(&answer), (failed.clone(), failed), "{bytes:?}");
    }
    // Exactly 8 bytes is a size.
    let (opened, size) = sized_by(&Ok(size_answer(4)));
    assert_eq!(opened.result, ResultCode::SUCCESS);
    assert_eq!(size, s64(4));
}

#[test]
fn a_size_over_s64_max_fails() {
    let max = u64::try_from(i64::MAX).unwrap();
    for over in [max + 1, u64::MAX] {
        let failed = only(result::HOST_IO_FAILED);
        assert_eq!(
            sized_by(&Ok(size_answer(over))),
            (failed.clone(), failed),
            "{over:#x}"
        );
    }
    // Exactly `i64::MAX` is a size.
    let (opened, size) = sized_by(&Ok(size_answer(max)));
    assert_eq!(opened.result, ResultCode::SUCCESS);
    assert_eq!(size, s64(i64::MAX));
}

#[test]
fn a_probe_answered_with_bytes_opens_nothing() {
    let mut rig = Rig::new();
    rig.force(Asked::Probe, Ok(HostIoReply::Data(vec![1])));
    let reply = rig.send(To::Session(rig.session), OPEN_SD_CARD_FILE_SYSTEM, |r| r);
    assert_eq!(reply, only(result::HOST_IO_FAILED));
    assert_eq!(rig.server.session_count(), 1);
    // An empty answer opens the file system.
    rig.force(Asked::Probe, Ok(HostIoReply::Data(Vec::new())));
    let reply = rig.send(To::Session(rig.session), OPEN_SD_CARD_FILE_SYSTEM, |r| r);
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(rig.server.session_count(), 2);
}

/// What a broken host answers: an environment that fails, and a failure code with no name.
fn broken() -> [Result<HostIoReply, EnvError>; 2] {
    let unnamed = 99;
    assert_eq!(HostFailure::from_code(unnamed), None);
    let exhausted = EnvError::Exhausted {
        index: 0,
        requested: EnvCall::HostIo,
    };
    [Err(exhausted), Ok(HostIoReply::Failed(unnamed))]
}

#[test]
fn a_broken_host_fails_open_file_and_read() {
    for answer in broken() {
        let mut rig = Rig::new();
        let fs = rig.open_sd();
        let sessions = rig.server.session_count();
        rig.force(Asked::Size, answer.clone());
        let opened = rig.open_file(fs, HELLO_PATH);
        assert_eq!(opened, only(result::HOST_IO_FAILED), "{answer:?}");
        assert_eq!(rig.server.session_count(), sessions);

        let mut rig = Rig::new();
        let file = To::Session(rig.hello());
        rig.force(Asked::Read, answer.clone());
        rig.defer(file, READ, read(0, 4, 16));
        rig.jobs.run();
        let reply = rig.deliver(SessionKind::Session);
        assert_eq!(reply, only(result::HOST_IO_FAILED), "{answer:?}");
        assert_eq!(rig.kernel.memory(DATA_AT, 16).unwrap(), [STALE; 16]);
    }
}

#[test]
fn a_snapshot_with_a_read_in_flight_restores_a_read_nothing_completes() {
    let mut rig = Rig::new();
    let file = rig.hello();
    rig.defer(To::Session(file), READ, read(0, 4, 16));
    let saved = rig.server.save().unwrap();

    let env: Arc<dyn Env> = Arc::<FakeEnv>::clone(&rig.env);
    let executor: Arc<dyn Executor> = Arc::<Jobs>::clone(&rig.jobs);
    let mut restored = server(UnknownPolicy::Strict, env).with_executor(executor);
    restored.restore(&saved).unwrap();
    let pending: Vec<ObjectId> = restored.pending().map(|(_, session)| session).collect();
    assert_eq!(pending, [file]);

    // The one job there is completes the token of the server it was taken on.
    assert_eq!(rig.jobs.len(), 1);
    rig.jobs.run();
    assert!(restored.take_completed().is_none());
    assert_eq!(restored.pending().count(), 1);
    assert!(rig.server.take_completed().is_some());
}

/// The snapshot of a server with [`HELLO`] open on a plain session (`postcard`: integers are
/// varints, a sequence or a string starts with its length, an enum with its variant's index, an
/// `Option` with 0 or 1; ids are an index and then a generation). The framework's part is
/// described in `framework/tests/snapshot.rs`; the states are this module's.
///
/// The kernel objects, by index: 0 the `fsp-srv` session, 1 the file system's, 2 the file's. The
/// targets, by slot, are the same three interfaces.
#[rustfmt::skip]
const SNAPSHOT: &[u8] = &[
    // Header: magic, version.
    b'S', b'Q', b'S', b'V', 1,
    // The server's name.
    7, b'f', b's', b'p', b'-', b's', b'r', b'v',
    // Sessions: three, each a session object and the target it leads to.
    3,
    0, 1,  0, 1,
    1, 1,  1, 1,
    2, 1,  2, 1,
    // Targets: three slots, each occupied (1) with its generation, then the row: its reference
    // count, the kind of target (0 an interface), the interface's name and its state's bytes.
    3,
    1, 1,  1, 0, 6, b'F', b's', b'p', b'S', b'r', b'v',
    0, // `FspSrv`'s state: no bytes.
    1, 1,  1, 0, 11, b'I', b'F', b'i', b'l', b'e', b'S', b'y', b's', b't', b'e', b'm',
    1, 0, // `IFileSystem`'s state: one byte, the mount (`SdCard`).
    1, 1,  1, 0, 5, b'I', b'F', b'i', b'l', b'e',
    // `IFile`'s state: 12 bytes, the mount (`SdCard`), then the path's length and bytes.
    12, 0, 10, b'/', b'h', b'e', b'l', b'l', b'o', b'.', b't', b'x', b't',
    // The slab's free list (empty), its slot limit (`u32::MAX`) and its count of rows.
    0,  0xFF, 0xFF, 0xFF, 0xFF, 0x0F,  3,
    // The queue: one worker, nothing running, nobody ready, no sessions with a request.
    1, 0, 0, 0,
    // No deferred requests; the next token is 0; no family state.
    0, 0, 0,
];

#[test]
fn an_open_file_s_snapshot_bytes_are_pinned() {
    let mount = Mount::SdCard;
    // `FspSrv` holds nothing; `Mount` has one variant, so its index is 0.
    assert_eq!(FspSrv.save().unwrap(), []);
    assert_eq!(IFileSystem { mount }.save().unwrap(), [0]);
    let file = IFile {
        mount,
        path: HELLO_PATH.to_vec(),
    };
    let len = u8::try_from(HELLO_PATH.len()).unwrap();
    assert_eq!(file.save().unwrap(), [&[0, len][..], HELLO_PATH].concat());

    let mut rig = Rig::new();
    rig.hello();
    assert_eq!(rig.server.save().unwrap(), SNAPSHOT);
    // The constants in the bytes are the shipping ones.
    assert_eq!(SNAPSHOT[..4], SNAPSHOT_MAGIC);
    assert_eq!(u32::from(SNAPSHOT[4]), SNAPSHOT_VERSION);
    assert_eq!(SNAPSHOT[6..13], *NAME.as_bytes());
}

/// [`SNAPSHOT`] with [`HELLO_PATH`] in it replaced by `path`, of the same length.
fn snapshot_with(path: &[u8]) -> Vec<u8> {
    assert_eq!(path.len(), HELLO_PATH.len());
    let at = SNAPSHOT
        .windows(HELLO_PATH.len())
        .position(|window| window == HELLO_PATH)
        .unwrap();
    let mut bytes = SNAPSHOT.to_vec();
    bytes[at..at + path.len()].copy_from_slice(path);
    bytes
}

#[test]
fn a_snapshot_with_a_path_that_is_not_normalised_is_refused() {
    let refused: [&[u8]; 5] = [
        b"hello.txt/",
        b"//ello.txt",
        b"/hello.tx/",
        b"/hello\0txt",
        b"/./llo.txt",
    ];
    for path in refused {
        let env: Arc<dyn Env> = FakeEnv::new(true);
        let mut restored = server(UnknownPolicy::Strict, env);
        let error = SnapshotError::State {
            interface: "IFile".to_owned(),
            error: postcard::Error::SerdeDeCustom,
        };
        let bytes = snapshot_with(path);
        assert_eq!(restored.restore(&bytes), Err(error), "{path:?}");
        assert_eq!(restored.session_count(), 0);
    }
    let env: Arc<dyn Env> = FakeEnv::new(true);
    let mut restored = server(UnknownPolicy::Strict, env);
    restored.restore(&snapshot_with(HELLO_PATH)).unwrap();
    assert_eq!(restored.session_count(), 3);
}

#[test]
fn a_saved_path_may_be_as_long_as_a_path_and_no_longer() {
    let mut longest = vec![path::DIRECTORY_SEPARATOR];
    longest.resize(path::ENTRY_NAME_LENGTH_MAX, b'a');
    let bytes = postcard::to_allocvec(&(Mount::SdCard, &longest)).unwrap();
    let file: IFile = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(file.path, longest);

    longest.push(b'a');
    let bytes = postcard::to_allocvec(&(Mount::SdCard, &longest)).unwrap();
    let error = postcard::from_bytes::<IFile>(&bytes).unwrap_err();
    assert_eq!(error, postcard::Error::SerdeDeCustom);
}
