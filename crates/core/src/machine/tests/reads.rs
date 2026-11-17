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
use std::sync::atomic::AtomicBool;

use starquake_ipc::{
    BUFFER_SIZE, Body, BufferDescriptor, CommandType, Message, RequestBuilder, RequestLayout,
    SessionKind, StaticDescriptor,
};
use starquake_loader::nup::build::{self, Content, PackageSpec, ROMFS_DIR_NAME, Tree};
use starquake_loader::nup::storage::{CACHE_CAP, RomFsStorage};
use starquake_loader::nup::{Nup, PackageKind};
use starquake_loader::read_at::{Positional, ReadAt};
use starquake_services::fsp::{self, command};
use starquake_services::sm::{self, MAX_NAME_LEN, ServiceName};

use super::*;
use crate::hle::tests::Scratch;
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
    /// Set when a held read has returned.
    released: Mutex<bool>,
    opened: Condvar,
}

/// The longest [`Source`] holds a read.
const HELD_FOR: Duration = Duration::from_secs(3);

impl ReadAt for Source {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        if self.holding.load(Ordering::SeqCst) {
            let released = lock(&self.released);
            let (mut released, _) = self
                .opened
                .wait_timeout(released, HELD_FOR)
                .unwrap_or_else(PoisonError::into_inner);
            *released = true;
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
struct WithTls<'a> {
    image: &'a FlatMemory,
    tls: Arc<FlatMemory>,
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
struct Seen {
    handles: Vec<u32>,
    results: Vec<u32>,
    reads: Vec<Vec<u8>>,
}

/// The scripted guest's view of its thread-local page.
#[derive(Clone)]
struct Script {
    tls: Arc<FlatMemory>,
    base: u64,
    seen: Rc<RefCell<Seen>>,
}

impl Script {
    /// Writes `request` as the thread's message.
    fn write(&self, request: &RequestBuilder) {
        let mut buffer = [0; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        self.tls.write(self.base, &buffer).unwrap();
    }

    /// Reads the reply in the thread's message, with `size` bytes of payload, keeping its
    /// result word and any handle moved with it.
    fn read_reply(&self, size: usize) -> Vec<u8> {
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
    fn last_handle(&self) -> u64 {
        u64::from(*self.seen.borrow().handles.last().unwrap())
    }
}

fn libnx(id: u32) -> RequestBuilder {
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
fn send(
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
/// `OpenDataStorageByCurrentProcess` and an `IStorage::Read`, then `OpenSdCardFileSystem`,
/// `OpenFile` and an `IFile::Read`, then `ExitProcess`.
fn reading_guest(guest: &Script) -> ScriptedVcpu {
    let name = guest.base + NAME_AT;
    guest.tls.write(name, b"sm:\0").unwrap();
    let mut path = b"/card.txt".to_vec();
    path.resize(fsp::PATH_BUFFER_SIZE, 0);
    guest.tls.write(guest.base + PATH_AT, &path).unwrap();
    let sm_handle = Rc::new(RefCell::new(0u64));
    let keep = Rc::clone(&sm_handle);
    let take = Rc::clone(&sm_handle);
    ScriptedVcpu::new()
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
        )
        .then(
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
    assert_eq!(romfs.read_at(IMAGE_OFFSET, &mut expected).unwrap(), expected.len());
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
    assert!(*lock(&source.released), "the run returned with a read in flight");
    assert!(started.elapsed() < RUN_LIMIT);
    // The guest never saw the read's reply.
    assert!(seen.reads.is_empty());
}
