//! `fsp-srv`'s reads in a real run (`run`): the core threads, the timer thread and `fsp-srv`'s
//! workers, with a scripted guest that talks to the services as libnx lays its requests out
//! (`libnx/nx/source/services/sm.c` and `fs.c`), over a title RomFS built from a synthetic NUP
//! and an SD card folder made for the test.
//!
//! Every message, path and buffer is in the main thread's thread-local page, which the kernel
//! lets the guest read and write. Each run has a timeout, so a step that fails (a panic on core
//! 0) ends the run instead of leaving the other threads waiting.

use std::cell::RefCell;
use std::io;
use std::sync::Condvar;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, Sender};

use starquake_ipc::{
    BUFFER_SIZE, Body, BufferDescriptor, CommandType, Message, RequestBuilder, RequestLayout,
    SessionKind, StaticDescriptor,
};
use starquake_loader::nup::build::{self, Content, PackageSpec, ROMFS_DIR_NAME, Tree};
use starquake_loader::nup::storage::{CACHE_CAP, RomFsStorage};
use starquake_loader::nup::{Nup, PackageKind};
use starquake_loader::read_at::{Positional, ReadAt};
use starquake_services::framework::{Executor, Finish, Job};
use starquake_services::fsp::host::{CHANNEL, HostRequest, Mount};
use starquake_services::fsp::{self, command};
use starquake_services::sm::{self, MAX_NAME_LEN, ServiceName};
use starquake_services::time;
use starquake_trace::{EnvCall, EnvError, HostIoReply, HostIoRequest, HostNanos};

use super::*;
use crate::hle::tests::{Scratch, Unexpected};
use crate::title::RomFs;

/// Where in the thread-local page the port's name, a path and the read buffer are.
const NAME_AT: u64 = 0x300;
const PATH_AT: u64 = 0x400;
const DATA_AT: u64 = 0x800;
/// The read buffer's size.
const DATA_LEN: u64 = 0x40;

/// The longest a run here may take: its timeout.
const RUN_LIMIT: Duration = Duration::from_secs(30);

/// Where the RomFS read reads from, inside the file's data.
const IMAGE_OFFSET: u64 = 0x200;
const IMAGE_LEN: u64 = 8;
/// The SD card file, and where its read reads from.
const CARD_TEXT: &[u8] = b"read from the card by a worker";
const CARD_OFFSET: u64 = 5;
const CARD_LEN: u64 = 9;

/// A source over a container's bytes whose reads can be held until a gate opens, for at most
/// [`HELD_FOR`].
#[derive(Default)]
struct Source {
    bytes: Vec<u8>,
    holding: AtomicBool,
    gate: Mutex<Held>,
    /// Rung when [`Held`] changes.
    changed: Condvar,
}

/// Where a [`Source`]'s held read is.
#[derive(Default)]
struct Held {
    /// Set when a read starts being held.
    entered: bool,
    /// Set by the test to let a held read go before [`HELD_FOR`].
    opened: bool,
    /// Set when a held read has returned.
    released: bool,
}

/// The longest [`Source`] holds a read.
const HELD_FOR: Duration = Duration::from_secs(3);

impl Source {
    /// Whether a held read has returned.
    fn released(&self) -> bool {
        lock(&self.gate).released
    }

    /// Waits, for at most [`RUN_LIMIT`], until a read is being held.
    fn await_entered(&self) {
        let gate = lock(&self.gate);
        let (gate, timeout) = self
            .changed
            .wait_timeout_while(gate, RUN_LIMIT, |gate| !gate.entered)
            .unwrap();
        drop(gate);
        assert!(!timeout.timed_out(), "no read reached the title file");
    }

    /// Lets a held read go.
    fn open(&self) {
        lock(&self.gate).opened = true;
        self.changed.notify_all();
    }
}

impl ReadAt for Source {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        if self.holding.load(Ordering::SeqCst) {
            let mut gate = lock(&self.gate);
            gate.entered = true;
            self.changed.notify_all();
            let (mut gate, _) = self
                .changed
                .wait_timeout_while(gate, HELD_FOR, |gate| !gate.opened)
                .unwrap_or_else(PoisonError::into_inner);
            gate.released = true;
            self.changed.notify_all();
            return Err(io::Error::other("the read was held"));
        }
        self.bytes.read_at(offset, buf)
    }

    fn size(&self) -> io::Result<u64> {
        self.bytes.size()
    }
}

/// A title RomFS holding one file, from a synthetic NUP, and the source it reads from.
fn romfs() -> (RomFs, Arc<Source>) {
    let tree = Tree::new()
        .with_file(
            &format!("{ROMFS_DIR_NAME}/data.bin"),
            Content::raw((0u8..=250).rev().collect::<Vec<u8>>()),
        )
        .unwrap();
    let base = PackageSpec::new(PackageKind::Base, 0x0100_0000_0000_9300, 0, tree);
    let source = Arc::new(Source {
        bytes: build::build(&[base]).unwrap().bytes,
        ..Source::default()
    });
    let nup = Nup::open(Positional::new(Arc::clone(&source))).unwrap();
    let dir = nup.packages()[0].romfs.unwrap();
    let storage = RomFsStorage::new(nup, dir, CACHE_CAP).unwrap();
    (RomFs::new(storage), source)
}

/// The guest's image, and its main thread's thread-local page, which the script's steps write
/// and read too.
pub(super) struct WithTls<'a> {
    pub(super) image: &'a FlatMemory,
    pub(super) tls: Arc<FlatMemory>,
}

impl GuestMemory for WithTls<'_> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.image.read(va, buf).or_else(|_| self.tls.read(va, buf))
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        self.image
            .write(va, data)
            .or_else(|_| self.tls.write(va, data))
    }
}

impl<V> GuestSpace<V> for WithTls<'_> {
    fn touch(&self, _: u64, _: Access) -> Touch {
        Touch::Refused
    }

    fn settle(&self, _: &mut V) -> Result<(), starquake_cpu::Error> {
        Ok(())
    }
}

/// What the scripted guest saw: each reply's result word and payload, the handles it was
/// given, and the bytes of each read.
#[derive(Default)]
pub(super) struct Seen {
    pub(super) handles: Vec<u32>,
    pub(super) results: Vec<u32>,
    pub(super) reads: Vec<Vec<u8>>,
}

/// The scripted guest's view of its thread-local page.
#[derive(Clone)]
pub(super) struct Script {
    pub(super) tls: Arc<FlatMemory>,
    pub(super) base: u64,
    pub(super) seen: Rc<RefCell<Seen>>,
}

impl Script {
    /// Writes `request` as the thread's message.
    pub(super) fn write(&self, request: &RequestBuilder) {
        let mut buffer = [0; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        self.tls.write(self.base, &buffer).unwrap();
    }

    /// Reads the reply in the thread's message, with `size` bytes of payload, keeping its
    /// result word and any handle moved with it.
    pub(super) fn read_reply(&self, size: usize) -> Vec<u8> {
        let mut buffer = [0; BUFFER_SIZE];
        self.tls.read(self.base, &mut buffer).unwrap();
        let message = Message::parse_reply_sized(&buffer, SessionKind::Session, size).unwrap();
        let Body::CmifReply(body) = message.body else {
            unreachable!("not a CMIF reply");
        };
        let mut seen = self.seen.borrow_mut();
        seen.results.push(body.header.result);
        seen.handles.extend(message.move_handles);
        body.data.to_vec()
    }

    /// The `len` bytes the read buffer holds.
    fn read_buffer(&self, len: u64) {
        let mut bytes = vec![0; usize::try_from(len).unwrap()];
        self.tls.read(self.base + DATA_AT, &mut bytes).unwrap();
        self.seen.borrow_mut().reads.push(bytes);
    }

    /// The handle the last reply moved.
    pub(super) fn last_handle(&self) -> u64 {
        u64::from(*self.seen.borrow().handles.last().unwrap())
    }
}

pub(super) fn libnx(id: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, id).layout(RequestLayout::Libnx)
}

/// The read buffer as a map-alias (B) descriptor of `len` bytes.
fn room(guest: &Script, len: u64) -> BufferDescriptor {
    BufferDescriptor {
        address: guest.base + DATA_AT,
        size: len,
        flags: 0,
    }
}

/// A step that sends what `write` writes to the handle `handle` gives, after the previous call
/// succeeded. `handle` runs first, so it may read the last reply before `write` replaces it.
pub(super) fn send(
    guest: &Script,
    handle: impl FnOnce(&Script) -> u64 + 'static,
    write: impl FnOnce(&Script) + 'static,
) -> impl FnOnce(&mut GuestRegs) + 'static {
    let guest = guest.clone();
    move |regs| {
        assert_eq!(regs.x[0], 0, "the call before failed");
        regs.x[0] = handle(&guest);
        write(&guest);
    }
}

/// The scripted conversation: `sm:`'s `Initialize` and `GetService("fsp-srv")`, then
/// `OpenDataStorageByCurrentProcess` and an `IStorage::Read`, then [`reading_the_card`].
fn reading_guest(guest: &Script) -> ScriptedVcpu {
    let name = guest.base + NAME_AT;
    guest.tls.write(name, b"sm:\0").unwrap();
    let mut path = b"/card.txt".to_vec();
    path.resize(fsp::PATH_BUFFER_SIZE, 0);
    guest.tls.write(guest.base + PATH_AT, &path).unwrap();
    let sm_handle = Rc::new(RefCell::new(0u64));
    let keep = Rc::clone(&sm_handle);
    let take = Rc::clone(&sm_handle);
    let vcpu = ScriptedVcpu::new()
        .then(move |regs| regs.x[1] = name, svc(SvcId::ConnectToNamedPort))
        .then(
            {
                let guest = guest.clone();
                move |regs| {
                    assert_eq!(regs.x[0], 0, "ConnectToNamedPort failed");
                    *keep.borrow_mut() = regs.x[1];
                    guest.write(
                        &libnx(sm::command::INITIALIZE)
                            .send_pid(0)
                            .data(&0_u64.to_le_bytes()),
                    );
                    regs.x[0] = regs.x[1];
                }
            },
            svc(SvcId::SendSyncRequest),
        )
        .then(
            send(
                guest,
                move |_| *take.borrow(),
                |guest| {
                    guest.read_reply(0);
                    let name: [u8; MAX_NAME_LEN] = ServiceName::new(fsp::NAME).unwrap().to_field();
                    guest.write(&libnx(sm::command::GET_SERVICE).data(&name));
                },
            ),
            svc(SvcId::SendSyncRequest),
        )
        .then(
            send(
                guest,
                |guest| {
                    guest.read_reply(0);
                    guest.last_handle()
                },
                |guest| {
                    let open = command::fsp_srv::OPEN_DATA_STORAGE_BY_CURRENT_PROCESS;
                    guest.write(&libnx(open));
                },
            ),
            svc(SvcId::SendSyncRequest),
        )
        .then(
            send(
                guest,
                |guest| {
                    guest.read_reply(0);
                    guest.last_handle()
                },
                |guest| {
                    let offset = i64::try_from(IMAGE_OFFSET).unwrap();
                    let size = i64::try_from(IMAGE_LEN).unwrap();
                    let payload = [offset.to_le_bytes(), size.to_le_bytes()].concat();
                    let read = libnx(command::storage::READ)
                        .data(&payload)
                        .receive_buffer(room(guest, IMAGE_LEN));
                    guest.write(&read);
                },
            ),
            svc(SvcId::SendSyncRequest),
        );
    reading_the_card(guest, vcpu)
}

/// The conversation's second half, after the storage read: `OpenSdCardFileSystem`, `OpenFile`
/// and an `IFile::Read`, then `ExitProcess`.
fn reading_the_card(guest: &Script, vcpu: ScriptedVcpu) -> ScriptedVcpu {
    vcpu.then(
        send(
            guest,
            |guest| {
                guest.read_reply(0);
                guest.read_buffer(IMAGE_LEN);
                // The `fsp-srv` session: the first handle `GetService` moved.
                u64::from(guest.seen.borrow().handles[0])
            },
            |guest| guest.write(&libnx(command::fsp_srv::OPEN_SD_CARD_FILE_SYSTEM)),
        ),
        svc(SvcId::SendSyncRequest),
    )
    .then(
        send(
            guest,
            |guest| {
                guest.read_reply(0);
                guest.last_handle()
            },
            |guest| {
                let descriptor = StaticDescriptor {
                    counter: 0,
                    address: guest.base + PATH_AT,
                    size: u16::try_from(fsp::PATH_BUFFER_SIZE).unwrap(),
                };
                let open = libnx(command::file_system::OPEN_FILE)
                    .data(&fsp::open_mode::READ.to_le_bytes())
                    .static_descriptor(descriptor);
                guest.write(&open);
            },
        ),
        svc(SvcId::SendSyncRequest),
    )
    .then(
        send(
            guest,
            |guest| {
                guest.read_reply(0);
                guest.last_handle()
            },
            |guest| {
                let offset = i64::try_from(CARD_OFFSET).unwrap();
                let size = i64::try_from(CARD_LEN).unwrap();
                let payload = [
                    &0_u32.to_le_bytes()[..],
                    &[0; 4],
                    &offset.to_le_bytes(),
                    &size.to_le_bytes(),
                ]
                .concat();
                let read = libnx(command::file::READ)
                    .data(&payload)
                    .receive_buffer(room(guest, DATA_LEN));
                guest.write(&read);
            },
        ),
        svc(SvcId::SendSyncRequest),
    )
    .then(
        {
            let guest = guest.clone();
            move |regs| {
                assert_eq!(regs.x[0], 0, "the file read failed");
                let count = guest.read_reply(size_of::<i64>());
                guest.seen.borrow_mut().reads.push(count);
                guest.read_buffer(CARD_LEN);
            }
        },
        svc(SvcId::ExitProcess),
    )
}

/// Runs the scripted conversation with `romfs` as the data storage and a card holding
/// [`CARD_TEXT`] in a scratch folder named for `test` (each test its own: one test's folder
/// removed under another's run would fail its `OpenFile`); returns the report and what the
/// guest saw.
fn run_reading(test: &str, romfs: RomFs, timeout: Duration) -> (RunReport, Seen) {
    let guest = super::Guest::new(&programs::hello().unwrap());
    let page = usize::try_from(crate::image::PAGE).unwrap();
    let tls = Arc::new(FlatMemory::new(guest.tls, page));
    let space = WithTls {
        image: &guest.memory,
        tls: Arc::clone(&tls),
    };
    let seen = Rc::new(RefCell::new(Seen::default()));
    let view = Script {
        tls,
        base: guest.tls,
        seen: Rc::clone(&seen),
    };
    let mut main = reading_guest(&view);
    let scratch = Scratch::new(test);
    scratch.file("card.txt", CARD_TEXT);
    let options = RunOptions {
        timeout: Some(timeout),
        command_line: Vec::new(),
        sd_card: Some(scratch.path()),
        romfs: Some(romfs),
    };
    let report = run(
        vcpus(&mut main),
        guest.program(&space),
        &options,
        Instant::now(),
        timebase(),
    );
    // The steps not played still hold the guest's view.
    drop((view, main));
    let seen = Rc::try_unwrap(seen).ok().unwrap().into_inner();
    (report, seen)
}

#[test]
fn a_guest_reads_the_titles_romfs_and_a_card_file_in_a_real_run() {
    let (romfs, _) = romfs();
    let mut expected = vec![0; usize::try_from(IMAGE_LEN).unwrap()];
    assert_eq!(
        romfs.read_at(IMAGE_OFFSET, &mut expected).unwrap(),
        expected.len()
    );
    let (report, seen) = run_reading("machine-reads-romfs-and-card", romfs, RUN_LIMIT);
    assert_eq!(report.exit, ExitReason::ProcessExited, "{}", report.trace);
    // Every reply was a success.
    assert_eq!(seen.results, [0; 7]);
    let start = usize::try_from(CARD_OFFSET).unwrap();
    let end = start + usize::try_from(CARD_LEN).unwrap();
    let count = i64::try_from(CARD_LEN).unwrap().to_le_bytes().to_vec();
    assert_eq!(
        seen.reads,
        [expected, count, CARD_TEXT[start..end].to_vec()]
    );
    // Both reads were deferred, and answered by a delivery.
    let deferred = report
        .trace
        .lines()
        .filter(|line| line.contains("Deferred"))
        .count();
    assert_eq!(deferred, 2, "{}", report.trace);
}

#[test]
fn a_run_that_ends_with_a_read_in_flight_waits_for_it_and_leaves_no_thread() {
    let (romfs, source) = romfs();
    source.holding.store(true, Ordering::SeqCst);
    // The run times out while the read waits on the title file; the read returns later.
    let timeout = Duration::from_secs(1);
    let started = Instant::now();
    let (report, seen) = run_reading("machine-reads-in-flight", romfs, timeout);
    assert_eq!(report.exit, ExitReason::Timeout, "{}", report.trace);
    // The run did not return before the worker's read did, so no worker thread outlived it.
    assert!(source.released(), "the run returned with a read in flight");
    assert!(started.elapsed() < RUN_LIMIT);
    // The guest never saw the read's reply.
    assert_eq!(seen.reads, Vec::<Vec<u8>>::new());
}

/// A run's session over `options`, with no guest behind it: what a run builds its workers'
/// environments from.
fn session_over<'a>(memory: &'a FlatMemory, options: &'a RunOptions) -> Session<'a, SystemClock> {
    Session {
        memory,
        options,
        clock: RunClock::new(timebase()),
        recorder: Recorder::new(),
    }
}

/// The host request of a read of `len` bytes from `offset` of the file at `path` on `mount`.
fn host_read(mount: Mount, path: &[u8], offset: u64, len: u64) -> HostIoRequest {
    HostRequest::Read {
        mount,
        path: path.to_vec(),
        offset,
        len,
    }
    .to_host_io()
}

/// A job that asks the host for `request` and whose finish sends `name` and the answer on
/// `answers`.
fn asking(
    name: &'static str,
    request: HostIoRequest,
    answers: &Sender<(&'static str, HostIoReply)>,
) -> Job {
    let answers = answers.clone();
    Box::new(move |env: &dyn Env| {
        let reply = env.host_io(&request).unwrap();
        let finish: Finish = Box::new(move || answers.send((name, reply)).unwrap());
        finish
    })
}

#[test]
fn a_read_held_at_the_title_does_not_hold_up_a_card_read() {
    // A real run's workers on a real run's environments: a live environment holds its lock for
    // the whole of a host request, so workers that shared one would wait for each other.
    let (romfs, source) = romfs();
    source.holding.store(true, Ordering::SeqCst);
    let scratch = Scratch::new("machine-reads-held-title");
    scratch.file("card.txt", CARD_TEXT);
    let options = RunOptions {
        timeout: None,
        command_line: Vec::new(),
        sd_card: Some(scratch.path()),
        romfs: Some(romfs),
    };
    let memory = FlatMemory::new(RAM_BASE, RAM_SIZE);
    let session = session_over(&memory, &options);
    let workers = reads_workers(&session.reads_environments(), Arc::new(Unexpected)).unwrap();
    let (send, answers) = mpsc::channel();
    let title = host_read(Mount::DataStorage, b"", IMAGE_OFFSET, IMAGE_LEN);
    workers.submit(asking("title", title, &send));
    source.await_entered();
    let card = host_read(Mount::SdCard, b"/card.txt", CARD_OFFSET, CARD_LEN);
    workers.submit(asking("card", card, &send));

    // The card read completes while the title read is still held.
    let (name, reply) = answers.recv_timeout(RUN_LIMIT).unwrap();
    assert!(
        !source.released(),
        "the card read waited for the title read"
    );
    let start = usize::try_from(CARD_OFFSET).unwrap();
    let end = start + usize::try_from(CARD_LEN).unwrap();
    assert_eq!(
        (name, reply),
        ("card", HostIoReply::Data(CARD_TEXT[start..end].to_vec()))
    );
    source.open();
    let (name, _) = answers.recv_timeout(RUN_LIMIT).unwrap();
    assert_eq!(name, "title");
}

#[test]
fn a_real_runs_workers_keep_no_record() {
    let options = RunOptions::default();
    let memory = FlatMemory::new(RAM_BASE, RAM_SIZE);
    let session = session_over(&memory, &options);
    let workers = reads_workers(&session.reads_environments(), Arc::new(Unexpected)).unwrap();
    let (send, answers) = mpsc::channel();
    let probe = HostRequest::Probe {
        mount: Mount::SdCard,
    }
    .to_host_io();
    workers.submit(asking("probe", probe, &send));
    // The record is kept, if it is, under the lock the finish runs in.
    assert_eq!(
        answers.recv_timeout(RUN_LIMIT).unwrap(),
        ("probe", HostIoReply::Unavailable)
    );
    assert_eq!(workers.take_record(), []);
}

#[test]
fn a_runs_workers_read_through_environments_of_their_own() {
    // The services are given one environment per worker, none of them the kernel's: what the
    // workers read is in their own recordings, not in the kernel's.
    let guest = Guest::new(&programs::hello().unwrap());
    let options = RunOptions::default();
    let session = session_over(&guest.memory, &options);
    let env = session.environment();
    let mut vcpu = ScriptedVcpu::new().then(|_| {}, svc(SvcId::ExitProcess));
    let checked = Arc::new(AtomicBool::new(false));
    let kernels = Arc::clone(&env);
    let checking = Arc::clone(&checked);
    let install = move |dispatcher: &Dispatcher, given: Given| {
        assert_eq!(given.reads.len(), worker::COUNT.get());
        let before = kernels.recording();
        for (number, reads) in given.reads.iter().enumerate() {
            let request = HostIoRequest {
                channel: CHANNEL,
                payload: vec![u8::try_from(number).unwrap(); 3],
            };
            let reply = reads.host_io(&request).unwrap();
            for (other, theirs) in given.reads.iter().enumerate() {
                let expected = if other <= number {
                    vec![EnvRecord::HostIo {
                        request: HostIoRequest {
                            channel: CHANNEL,
                            payload: vec![u8::try_from(other).unwrap(); 3],
                        },
                        reply: reply.clone(),
                    }]
                } else {
                    Vec::new()
                };
                assert_eq!(theirs.recording(), expected, "worker {other}'s environment");
            }
        }
        assert_eq!(
            kernels.recording(),
            before,
            "a worker read through the kernel's environment"
        );
        checking.store(true, Ordering::SeqCst);
        real_services(dispatcher, given)
    };
    let (dispatcher, _) = boot(guest.address_space(), &guest.placed, &guest.memory).unwrap();
    let (ending, ..) = drive(
        vcpus(&mut vcpu),
        dispatcher,
        &guest.memory,
        &session,
        &env,
        Limits::DEFAULT,
        &install,
    );
    assert!(matches!(ending, Ending::Exited));
    assert!(checked.load(Ordering::SeqCst));
}

/// A step of a scripted guest that writes its next request.
type Writes = Box<dyn FnOnce(&Script)>;

/// What [`FailsHostIo`] answers every host request with.
const HOST_IO_FAILURE: EnvError = EnvError::Exhausted {
    index: 9,
    requested: EnvCall::HostIo,
};

/// What `fsp-srv`'s handlers tell the halt when [`FailsHostIo`] fails them.
fn fsp_handler_failed() -> Stop {
    Stop::Handler {
        service: fsp::NAME.to_owned(),
        error: HOST_IO_FAILURE,
    }
}

/// The run's environment, but with every host request failed as a replay that ran out would
/// fail it.
struct FailsHostIo(Arc<dyn Env>);

impl Env for FailsHostIo {
    fn now(&self) -> Result<HostNanos, EnvError> {
        self.0.now()
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        self.0.random_u64()
    }

    fn host_io(&self, _: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        Err(HOST_IO_FAILURE)
    }
}

/// `sm:`'s `Initialize` and `GetService(service)`, then the requests `requests` makes, each sent
/// after the call before succeeded; then a step that says the guest ran again, and
/// `ExitProcess`.
fn guest_failed_by_its_environment(
    guest: &Script,
    service: &'static str,
    requests: Vec<Writes>,
    ran_again: &Rc<std::cell::Cell<bool>>,
) -> ScriptedVcpu {
    let name = guest.base + NAME_AT;
    guest.tls.write(name, b"sm:\0").unwrap();
    let sm_handle = Rc::new(RefCell::new(0u64));
    let keep = Rc::clone(&sm_handle);
    let mut vcpu = ScriptedVcpu::new()
        .then(move |regs| regs.x[1] = name, svc(SvcId::ConnectToNamedPort))
        .then(
            {
                let guest = guest.clone();
                move |regs| {
                    assert_eq!(regs.x[0], 0, "ConnectToNamedPort failed");
                    *keep.borrow_mut() = regs.x[1];
                    guest.write(
                        &libnx(sm::command::INITIALIZE)
                            .send_pid(0)
                            .data(&0_u64.to_le_bytes()),
                    );
                    regs.x[0] = regs.x[1];
                }
            },
            svc(SvcId::SendSyncRequest),
        )
        .then(
            send(
                guest,
                move |_| *sm_handle.borrow(),
                |guest| {
                    guest.read_reply(0);
                    let name: [u8; MAX_NAME_LEN] = ServiceName::new(service).unwrap().to_field();
                    guest.write(&libnx(sm::command::GET_SERVICE).data(&name));
                },
            ),
            svc(SvcId::SendSyncRequest),
        );
    for write in requests {
        vcpu = vcpu.then(
            send(
                guest,
                |guest| {
                    guest.read_reply(0);
                    guest.last_handle()
                },
                write,
            ),
            svc(SvcId::SendSyncRequest),
        );
    }
    let ran_again = Rc::clone(ran_again);
    vcpu.then(move |_| ran_again.set(true), svc(SvcId::ExitProcess))
}

/// Runs `requests` to `service` (see [`guest_failed_by_its_environment`]) with the title RomFS
/// as the data storage, and the services `install` sets up; returns the report and whether the
/// guest ran again after its last request.
fn run_failed(
    service: &'static str,
    requests: Vec<Writes>,
    install: &Install,
) -> (RunReport, bool) {
    let guest = super::Guest::new(&programs::hello().unwrap());
    let page = usize::try_from(crate::image::PAGE).unwrap();
    let tls = Arc::new(FlatMemory::new(guest.tls, page));
    let space = WithTls {
        image: &guest.memory,
        tls: Arc::clone(&tls),
    };
    let view = Script {
        tls,
        base: guest.tls,
        seen: Rc::new(RefCell::new(Seen::default())),
    };
    let ran_again = Rc::new(std::cell::Cell::new(false));
    let mut main = guest_failed_by_its_environment(&view, service, requests, &ran_again);
    let options = RunOptions {
        timeout: Some(RUN_LIMIT),
        command_line: Vec::new(),
        sd_card: None,
        romfs: Some(romfs().0),
    };
    let report = run_serving(
        vcpus(&mut main),
        guest.program(&space),
        &options,
        (Instant::now(), timebase()),
        Limits::DEFAULT,
        install,
    );
    drop((view, main));
    (report, ran_again.get())
}

#[test]
fn an_environment_failure_in_a_handler_stops_the_run() {
    // `fsp-srv`'s handlers read an environment whose host requests fail: the probe of
    // `OpenDataStorageByCurrentProcess` stops the run.
    let install = |dispatcher: &Dispatcher, given: Given| {
        let env: Arc<dyn Env> = Arc::new(FailsHostIo(Arc::clone(&given.env)));
        real_services(dispatcher, Given { env, ..given })
    };
    let open: Writes = Box::new(|guest| {
        guest.write(&libnx(
            command::fsp_srv::OPEN_DATA_STORAGE_BY_CURRENT_PROCESS,
        ));
    });
    let (report, ran_again) = run_failed(fsp::NAME, vec![open], &install);
    let reason = "the environment failed fsp-srv while it handled a request: recording exhausted \
                  at record 9: the replay asked for HostIo";
    assert_eq!(
        report.exit,
        ExitReason::EmulatorError(reason.to_owned()),
        "{}",
        report.trace
    );
    assert!(!ran_again, "the guest ran after the environment failed");
}

/// What [`FailsRandom`] answers every random draw with: a replay whose next record answers
/// another kind of call.
const RANDOM_FAILURE: EnvError = EnvError::Mismatch {
    index: 4,
    requested: EnvCall::Random,
    recorded: EnvCall::Clock,
};

/// The run's environment, but with every random draw failed as a replay that disagrees would
/// fail it.
struct FailsRandom(Arc<dyn Env>);

impl Env for FailsRandom {
    fn now(&self) -> Result<HostNanos, EnvError> {
        self.0.now()
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        Err(RANDOM_FAILURE)
    }

    fn host_io(&self, request: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        self.0.host_io(request)
    }
}

#[test]
fn an_environment_failure_in_time_u_stops_the_run() {
    // `time:u`'s set-up draws its clock source from an environment whose random draws fail:
    // `GetSharedMemoryNativeHandle` stops the run, with the reason naming `time:u`, through the
    // halt `crate::hle::install` gives it.
    let install = |dispatcher: &Dispatcher, given: Given| {
        let env: Arc<dyn Env> = Arc::new(FailsRandom(Arc::clone(&given.env)));
        real_services(dispatcher, Given { env, ..given })
    };
    let get: Writes = Box::new(|guest| {
        guest.write(&libnx(
            time::command::static_service::GET_SHARED_MEMORY_NATIVE_HANDLE,
        ));
    });
    let (report, ran_again) = run_failed(time::NAME, vec![get], &install);
    let reason = "the environment failed time:u while it handled a request: record 4 answers \
                  Clock but the replay asked for Random";
    assert_eq!(
        report.exit,
        ExitReason::EmulatorError(reason.to_owned()),
        "{}",
        report.trace
    );
    assert!(!ran_again, "the guest ran after the environment failed");
}

#[test]
fn an_environment_failure_in_a_read_stops_the_run() {
    // `fsp-srv`'s workers read an environment whose host requests fail: the storage read stops
    // the run, on a worker thread, while the guest waits for the reply.
    let install = |dispatcher: &Dispatcher, given: Given| {
        let failing: Arc<dyn Env> = Arc::new(FailsHostIo(Arc::clone(&given.env)));
        let envs = vec![failing; worker::COUNT.get()];
        let workers = Workers::new(envs, Recording::Off, Arc::clone(&given.halt))
            .map_err(|error| error.to_string())?;
        let reads = Reads {
            executor: Arc::new(workers),
            ready: Arc::new(given.ready),
            halt: given.halt,
        };
        crate::hle::install(dispatcher, given.env, reads).map_err(|error| error.to_string())
    };
    let open: Writes = Box::new(|guest| {
        guest.write(&libnx(
            command::fsp_srv::OPEN_DATA_STORAGE_BY_CURRENT_PROCESS,
        ));
    });
    let read: Writes = Box::new(|guest| {
        let payload = [0_i64.to_le_bytes(), 4_i64.to_le_bytes()].concat();
        let read = libnx(command::storage::READ)
            .data(&payload)
            .receive_buffer(room(guest, 4));
        guest.write(&read);
    });
    let (report, ran_again) = run_failed(fsp::NAME, vec![open, read], &install);
    let reason = "the environment failed file-system job 0: recording exhausted at record 9: \
                  the replay asked for HostIo";
    assert_eq!(
        report.exit,
        ExitReason::EmulatorError(reason.to_owned()),
        "{}",
        report.trace
    );
    assert!(!ran_again, "the guest ran after the environment failed");
}

#[test]
fn a_halt_keeps_its_first_reason_and_rings_the_timer_each_time() {
    // The ring is what wakes a timer thread asleep until its next deadline; without it the run
    // would still stop, at the timer's next turn (at most one preemption interval later), so
    // only this test can see it.
    let bell = Arc::new(timer::Bell::default());
    let halted = Halted::new(Arc::clone(&bell));
    assert!(!bell.take_ring());
    assert_eq!(halted.reason(), None);
    let first = fsp_handler_failed();
    halted.halt(first.clone());
    assert!(bell.take_ring());
    let reason = first.to_string();
    assert_eq!(halted.reason(), Some(reason.as_str()));
    // A later stop rings again, and the reason stays the first.
    halted.halt(Stop::Job {
        job: 3,
        error: HOST_IO_FAILURE,
    });
    assert!(bell.take_ring());
    assert_eq!(halted.reason(), Some(reason.as_str()));
}

/// How long [`HaltsLater`] waits before it tells the run's halt: long enough that the guest is
/// waiting for its reply by then. (If it were not, the test would still pass: the halt ends the
/// run either way.)
const HALT_AFTER: Duration = Duration::from_millis(100);

/// An executor that keeps every job it is given, never run, until it is dropped, and for each
/// tells the run's halt from a thread of its own [`HALT_AFTER`] later: a halt that no reply
/// delivery follows.
struct HaltsLater {
    halt: Arc<dyn Halt>,
    jobs: Mutex<Vec<Job>>,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Executor for HaltsLater {
    fn submit(&self, job: Job) {
        lock(&self.jobs).push(job);
        let halt = Arc::clone(&self.halt);
        let thread = std::thread::spawn(move || {
            std::thread::sleep(HALT_AFTER);
            halt.halt(Stop::Job {
                job: 0,
                error: HOST_IO_FAILURE,
            });
        });
        lock(&self.threads).push(thread);
    }
}

impl Drop for HaltsLater {
    fn drop(&mut self) {
        for thread in std::mem::take(&mut *lock(&self.threads)) {
            thread.join().unwrap();
        }
    }
}

#[test]
fn a_halt_no_delivery_follows_stops_a_guest_waiting_for_its_reply() {
    // The storage read is handed to an executor that never runs it, so the guest waits for its
    // reply with no deadline; another thread then tells the run's halt.
    let install = |dispatcher: &Dispatcher, given: Given| {
        let reads = Reads {
            executor: Arc::new(HaltsLater {
                halt: Arc::clone(&given.halt),
                jobs: Mutex::default(),
                threads: Mutex::default(),
            }),
            ready: Arc::new(given.ready),
            halt: given.halt,
        };
        crate::hle::install(dispatcher, given.env, reads).map_err(|error| error.to_string())
    };
    let open: Writes = Box::new(|guest| {
        guest.write(&libnx(
            command::fsp_srv::OPEN_DATA_STORAGE_BY_CURRENT_PROCESS,
        ));
    });
    let read: Writes = Box::new(|guest| {
        let payload = [0_i64.to_le_bytes(), 4_i64.to_le_bytes()].concat();
        let read = libnx(command::storage::READ)
            .data(&payload)
            .receive_buffer(room(guest, 4));
        guest.write(&read);
    });
    let started = Instant::now();
    let (report, ran_again) = run_failed(fsp::NAME, vec![open, read], &install);
    let reason = "the environment failed file-system job 0: recording exhausted at record 9: \
                  the replay asked for HostIo";
    assert_eq!(
        report.exit,
        ExitReason::EmulatorError(reason.to_owned()),
        "{}",
        report.trace
    );
    assert!(!ran_again, "the guest ran after the halt");
    assert!(started.elapsed() < RUN_LIMIT);
}

/// A guest's flat memory as its address space, which tells the run's halt at the settle
/// `countdown` says: armed by the guest, it counts the settles down, and the one that takes it
/// to zero tells the halt.
struct HaltsAtSettle<'a> {
    memory: &'a FlatMemory,
    halt: Arc<OnceLock<Arc<dyn Halt>>>,
    countdown: Arc<AtomicUsize>,
}

impl GuestMemory for HaltsAtSettle<'_> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.memory.read(va, buf)
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        self.memory.write(va, data)
    }
}

impl<V> GuestSpace<V> for HaltsAtSettle<'_> {
    fn touch(&self, _: u64, _: Access) -> Touch {
        Touch::Refused
    }

    fn settle(&self, _: &mut V) -> Result<(), starquake_cpu::Error> {
        let counted = self
            .countdown
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(1)
            });
        if counted == Ok(1) {
            let halt = self.halt.get().expect("the services were installed");
            halt.halt(fsp_handler_failed());
        }
        Ok(())
    }
}

/// How long the guest of `a_halt_told_as_a_thread_is_loaded_keeps_the_guest_out` sleeps.
const SLEEP_NS: u64 = 1_000_000;

#[test]
fn a_halt_told_as_a_thread_is_loaded_keeps_the_guest_out() {
    // The halt lands after the core has chosen the thread it runs and before it enters the
    // guest: at the settle of the switch that loads the thread its sleep's end woke. The core
    // loop's own check at the top of its turn has passed by then, so only the check before
    // entering the guest keeps the guest's next step from running.
    let guest = Guest::new(&programs::hello().unwrap());
    let halt = Arc::new(OnceLock::new());
    let countdown = Arc::new(AtomicUsize::new(0));
    let space = HaltsAtSettle {
        memory: &guest.memory,
        halt: Arc::clone(&halt),
        countdown: Arc::clone(&countdown),
    };
    let install = move |_: &Dispatcher, given: Given| {
        halt.set(given.halt)
            .map_err(|_| "installed twice".to_owned())
    };
    let ran_again = Rc::new(Cell::new(false));
    let ran = Rc::clone(&ran_again);
    let mut main = ScriptedVcpu::new()
        .then(
            move |regs| {
                // The settles to come: the one after this call, which began a wait, then the
                // one of the switch that loads the thread again once the sleep ends.
                countdown.store(2, Ordering::SeqCst);
                regs.x[0] = SLEEP_NS;
            },
            svc(SvcId::SleepThread),
        )
        .then(move |_| ran.set(true), svc(SvcId::ExitProcess));
    let options = RunOptions {
        timeout: Some(RUN_LIMIT),
        ..RunOptions::default()
    };
    let report = run_serving(
        vcpus(&mut main),
        guest.program(&space),
        &options,
        (Instant::now(), timebase()),
        Limits::DEFAULT,
        &install,
    );
    let reason = fsp_handler_failed().to_string();
    assert_eq!(
        report.exit,
        ExitReason::EmulatorError(reason),
        "{}",
        report.trace
    );
    assert!(!ran_again.get(), "the guest ran after the halt");
}
