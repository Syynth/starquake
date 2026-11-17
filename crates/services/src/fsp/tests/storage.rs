//! The data storage: `OpenDataStorageByCurrentProcess` and `IStorage`, over the fake host's
//! storage, laid out as libnx lays its requests out (`fsOpenDataStorageByCurrentProcess`,
//! `fsStorageRead`, `fsStorageGetSize`).

use super::*;
use crate::fsp::command::fsp_srv::OPEN_DATA_STORAGE_BY_CURRENT_PROCESS;
use crate::fsp::command::storage::{GET_SIZE as STORAGE_GET_SIZE, READ as STORAGE_READ};

/// The data storage the tests read: every byte distinct from its neighbours.
fn image() -> Vec<u8> {
    (0u8..100).map(|b| b.wrapping_mul(3) ^ 0x5A).collect()
}

/// Its size, as `GetSize` answers it.
fn image_len() -> i64 {
    i64::try_from(image().len()).unwrap()
}

/// Whether a request is for the data storage.
pub(super) fn is_storage(request: &HostRequest) -> bool {
    request.mount() == Mount::DataStorage
}

impl FakeHost {
    /// The answer for the data storage: one file, named by the empty path.
    pub(super) fn answer_storage(&self, request: &HostRequest) -> HostIoReply {
        let Some(storage) = &self.storage else {
            return HostFailure::NoDevice.reply();
        };
        match request {
            HostRequest::Probe { .. } => HostIoReply::Data(Vec::new()),
            HostRequest::Size { path, .. } if path.is_empty() => {
                HostIoReply::Data(u64::try_from(storage.len()).unwrap().to_le_bytes().to_vec())
            }
            HostRequest::Read {
                path, offset, len, ..
            } if path.is_empty() => {
                if let Mischief::FailReads(failure) = self.mischief {
                    return failure.reply();
                }
                let start = usize::try_from(*offset).unwrap().min(storage.len());
                let end = start
                    .saturating_add(usize::try_from(*len).unwrap())
                    .min(storage.len());
                let mut data = storage[start..end].to_vec();
                if self.mischief == Mischief::Greedy {
                    data.push(0xEE);
                }
                HostIoReply::Data(data)
            }
            _ => HostFailure::NotFound.reply(),
        }
    }
}

/// `fsStorageRead`'s payload: `s64 offset; u64 read_size;`.
fn storage_read_payload(offset: i64, size: i64) -> Vec<u8> {
    [offset.to_le_bytes(), size.to_le_bytes()].concat()
}

/// A storage read of `size` bytes from `offset` into a `len`-byte buffer at [`DATA_AT`].
fn storage_read(offset: i64, size: i64, len: u64) -> impl FnOnce(RequestBuilder) -> RequestBuilder {
    move |r| {
        r.data(&storage_read_payload(offset, size))
            .receive_buffer(room(len))
    }
}

impl Rig {
    /// A rig whose run has the data storage [`image`] (and the SD card).
    fn with_storage() -> Self {
        let rig = Self::new();
        rig.env.host().storage = Some(image());
        rig
    }

    /// `OpenDataStorageByCurrentProcess` on the rig's session: the storage's session.
    pub(super) fn open_storage(&mut self) -> ObjectId {
        let reply = self.send(
            To::Session(self.session),
            OPEN_DATA_STORAGE_BY_CURRENT_PROCESS,
            |r| r,
        );
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.moves.len(), 1);
        self.object_of(reply.moves[0])
    }
}

#[test]
fn a_titles_data_storage_is_opened_sized_read_and_closed() {
    let mut rig = Rig::with_storage();
    let storage = rig.open_storage();
    assert_eq!(rig.server.session_count(), 2);

    let size = rig.send(To::Session(storage), STORAGE_GET_SIZE, |r| r);
    assert_eq!(size, s64(image_len()));

    // Five bytes from offset 7 into a 16-byte buffer: deferred until the job runs.
    rig.defer(To::Session(storage), STORAGE_READ, storage_read(7, 5, 16));
    assert_eq!(rig.jobs.len(), 1);
    assert_eq!(rig.kernel.memory(DATA_AT, 16).unwrap(), [STALE; 16]);
    rig.jobs.run();
    // The reply carries nothing but the bytes, which reach memory when it is delivered.
    assert_eq!(rig.deliver(SessionKind::Session), only(ResultCode::SUCCESS));
    let mut expected = image()[7..12].to_vec();
    expected.resize(16, STALE);
    assert_eq!(rig.kernel.memory(DATA_AT, 16).unwrap(), expected);

    let mount = Mount::DataStorage;
    assert_eq!(
        rig.env.asked(),
        [
            HostRequest::Probe { mount },
            HostRequest::Size {
                mount,
                path: Vec::new()
            },
            HostRequest::Read {
                mount,
                path: Vec::new(),
                offset: 7,
                len: 5
            },
        ]
    );

    let (handled, _) = rig.try_send(storage, &RequestBuilder::close());
    assert_eq!(handled, Handled::Closed);
    assert_eq!(rig.server.session_count(), 1);
}

#[test]
fn a_storage_read_at_the_end_returns_what_is_there_and_past_it_nothing() {
    let len = image_len();
    // The last two bytes of a five-byte read; exactly at the end, and one past it: none.
    for (offset, expected) in [(len - 2, 2usize), (len, 0), (len + 1, 0)] {
        let mut rig = Rig::with_storage();
        let storage = To::Session(rig.open_storage());
        rig.defer(storage, STORAGE_READ, storage_read(offset, 5, 16));
        rig.jobs.run();
        assert_eq!(rig.deliver(SessionKind::Session), only(ResultCode::SUCCESS));
        let start = usize::try_from(offset).unwrap().min(image().len());
        let mut memory = image()[start..start + expected].to_vec();
        memory.resize(16, STALE);
        assert_eq!(
            rig.kernel.memory(DATA_AT, 16).unwrap(),
            memory,
            "at {offset}"
        );
    }
}

#[test]
fn a_storage_works_as_a_domain_object() {
    let mut rig = Rig::with_storage();
    let root = rig.convert();
    let reply = rig.send(
        To::Object(root),
        OPEN_DATA_STORAGE_BY_CURRENT_PROCESS,
        |r| r,
    );
    assert_eq!(reply.result, ResultCode::SUCCESS);
    let storage = To::Object(reply.objects[0]);
    let size = rig.send(storage, STORAGE_GET_SIZE, |r| r);
    assert_eq!(size, s64(image_len()));
    rig.defer(storage, STORAGE_READ, storage_read(0, 4, 4));
    rig.jobs.run();
    assert_eq!(rig.deliver(SessionKind::Domain), only(ResultCode::SUCCESS));
    assert_eq!(rig.kernel.memory(DATA_AT, 4).unwrap(), &image()[..4]);
}

/// What [`Broken`] answers every host request with.
pub(super) fn broken_error() -> EnvError {
    EnvError::Exhausted {
        index: 0,
        requested: EnvCall::HostIo,
    }
}

/// An environment whose every host request fails.
pub(super) struct Broken;

impl Env for Broken {
    fn now(&self) -> Result<HostNanos, EnvError> {
        Ok(HostNanos(0))
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        Ok(0)
    }

    fn host_io(&self, _: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        Err(broken_error())
    }
}

/// `OpenDataStorageByCurrentProcess` on a fresh server over `env`.
fn open_over(env: Arc<dyn Env>) -> Answered {
    open_with(env, OPEN_DATA_STORAGE_BY_CURRENT_PROCESS, unwatched())
}

/// The open `command` (one of the `RomFS` opens) on a fresh server over `env`, telling `halt`.
pub(super) fn open_with(env: Arc<dyn Env>, command: u32, halt: Arc<dyn Halt>) -> Answered {
    let mut kernel = FakeKernel::new(0).unwrap();
    let session = kernel.new_object();
    let mut server = server(UnknownPolicy::Strict, env, halt);
    server.accept(session).unwrap();
    let mut buffer = vec![STALE; BUFFER_SIZE];
    RequestBuilder::cmif(CommandType::Request, command)
        .build(&mut buffer)
        .unwrap();
    let handled = server.handle(&mut kernel, session, &mut buffer);
    assert_eq!(handled, Ok(Handled::Replied));
    assert_eq!(server.session_count(), 1);
    parse(&buffer, SessionKind::Session)
}

#[test]
fn without_a_romfs_opening_the_data_storage_is_eden_s_unknown() {
    // The run's title has no RomFS: the host has no storage.
    let mut rig = Rig::new();
    let reply = rig.send(
        To::Session(rig.session),
        OPEN_DATA_STORAGE_BY_CURRENT_PROCESS,
        |r| r,
    );
    assert_eq!(reply, only(result::UNKNOWN));
    assert_eq!(result::UNKNOWN.raw(), u32::MAX);
    assert_eq!(rig.server.session_count(), 1);
    // No handler at all answers the same.
    assert_eq!(open_over(Arc::new(NoHost)), only(result::UNKNOWN));
    // An environment that fails is no answer of the host's: it stops the run (decision log,
    // 2026-11-17, "Env failures"), and the request is answered as one the server gave up on.
    let halts = Arc::new(Halts::default());
    let halt: Arc<dyn Halt> = Arc::<Halts>::clone(&halts);
    let command = OPEN_DATA_STORAGE_BY_CURRENT_PROCESS;
    let reply = open_with(Arc::new(Broken), command, halt);
    assert_eq!(reply, only(DEFERRED_REPLY_DROPPED));
    assert_eq!(halts.stops(), [handler_failed(broken_error())]);
    // A probe answered with bytes is not a storage either.
    let rig = Rig::with_storage();
    rig.env.host().forced = Some((Asked::Probe, Ok(HostIoReply::Data(vec![0]))));
    let env: Arc<dyn Env> = Arc::<FakeEnv>::clone(&rig.env);
    assert_eq!(open_over(env), only(result::UNKNOWN));
}

#[test]
fn a_storage_read_checks_the_offset_before_the_size_and_the_buffer() {
    let cases = [
        // Both a negative offset and a negative size: the offset is named.
        ((-1, -1, 16), result::INVALID_OFFSET),
        ((-1, 17, 16), result::INVALID_OFFSET),
        ((0, -1, 16), result::INVALID_SIZE),
        // One byte more than the buffer holds.
        ((0, 17, 16), result::INVALID_SIZE),
    ];
    for ((offset, size, len), code) in cases {
        let mut rig = Rig::with_storage();
        let storage = To::Session(rig.open_storage());
        let reply = rig.send(storage, STORAGE_READ, storage_read(offset, size, len));
        assert_eq!(reply, only(code), "{offset} {size} {len}");
        assert_eq!(rig.jobs.len(), 0);
    }
    // Exactly the buffer's size is read.
    let mut rig = Rig::with_storage();
    let storage = To::Session(rig.open_storage());
    rig.defer(storage, STORAGE_READ, storage_read(0, 16, 16));
    assert_eq!(rig.jobs.len(), 1);
}

#[test]
fn a_storage_read_of_nothing_answers_at_once_and_one_that_overflows_is_out_of_range() {
    let mut rig = Rig::with_storage();
    let storage = To::Session(rig.open_storage());
    let asked = rig.env.asked().len();
    let reply = rig.send(storage, STORAGE_READ, storage_read(5, 0, 16));
    assert_eq!(reply, only(ResultCode::SUCCESS));
    assert_eq!(rig.env.asked().len(), asked);
    // The end overflows an `s64` by one byte; one byte less does not.
    let reply = rig.send(storage, STORAGE_READ, storage_read(i64::MAX, 1, 16));
    assert_eq!(reply, only(result::OUT_OF_RANGE));
    rig.defer(storage, STORAGE_READ, storage_read(i64::MAX - 1, 1, 16));
    rig.jobs.run();
    assert_eq!(rig.deliver(SessionKind::Session), only(ResultCode::SUCCESS));
}

#[test]
fn a_storage_read_the_host_fails_fails() {
    for failure in [HostFailure::Io, HostFailure::NotFound] {
        let mut rig = Rig::with_storage();
        let storage = To::Session(rig.open_storage());
        rig.env.host().mischief = Mischief::FailReads(failure);
        rig.defer(storage, STORAGE_READ, storage_read(0, 4, 16));
        rig.jobs.run();
        let code = match failure {
            HostFailure::NotFound => result::PATH_NOT_FOUND,
            _ => result::HOST_IO_FAILED,
        };
        assert_eq!(rig.deliver(SessionKind::Session), only(code));
        assert_eq!(rig.kernel.memory(DATA_AT, 16).unwrap(), [STALE; 16]);
    }
    // More bytes than asked for.
    let mut rig = Rig::with_storage();
    let storage = To::Session(rig.open_storage());
    rig.env.host().mischief = Mischief::Greedy;
    rig.defer(storage, STORAGE_READ, storage_read(0, 4, 16));
    rig.jobs.run();
    assert_eq!(
        rig.deliver(SessionKind::Session),
        only(result::HOST_IO_FAILED)
    );
    // A size answer that is not eight bytes.
    rig.env.host().forced = Some((Asked::Size, Ok(HostIoReply::Data(vec![1, 2, 3, 4]))));
    let reply = rig.send(storage, STORAGE_GET_SIZE, |r| r);
    assert_eq!(reply, only(result::HOST_IO_FAILED));
}

#[test]
fn a_storage_read_refuses_a_pointer_buffer() {
    use starquake_ipc::ReceiveDescriptor;
    let mut rig = Rig::with_storage();
    let storage = To::Session(rig.open_storage());
    let pointer = ReceiveDescriptor {
        address: DATA_AT,
        size: 16,
    };
    let reply = rig.send(storage, STORAGE_READ, |r| {
        r.data(&storage_read_payload(0, 4))
            .receive_descriptor(pointer)
    });
    assert_eq!(reply, only(INVALID_CMIF_REQUEST));
    assert_eq!(rig.jobs.len(), 0);
}

#[test]
fn an_open_storage_is_saved_and_a_restored_one_answers() {
    // `IStorage` keeps nothing.
    assert_eq!(IStorage.save().unwrap(), []);
    let mut rig = Rig::with_storage();
    let storage = rig.open_storage();
    let saved = rig.server.save().unwrap();
    let env: Arc<dyn Env> = Arc::<FakeEnv>::clone(&rig.env);
    let executor: Arc<dyn Executor> = Arc::<Jobs>::clone(&rig.jobs);
    let mut restored = server(UnknownPolicy::Strict, env, unwatched()).with_executor(executor);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.session_count(), 2);
    rig.server = restored;
    // The restored storage is used: it answers its size and a read.
    let size = rig.send(To::Session(storage), STORAGE_GET_SIZE, |r| r);
    assert_eq!(size, s64(image_len()));
    rig.defer(To::Session(storage), STORAGE_READ, storage_read(1, 2, 2));
    rig.jobs.run();
    assert_eq!(rig.deliver(SessionKind::Session), only(ResultCode::SUCCESS));
    assert_eq!(rig.kernel.memory(DATA_AT, 2).unwrap(), &image()[1..3]);
}

#[test]
fn a_file_system_on_the_data_storage_is_never_saved_or_restored() {
    // Only a file system is a file system's mount; the storage's mount is not saved, and a
    // snapshot naming a variant past the saved ones (2, the storage's place) is refused.
    let file_system = IFileSystem {
        mount: Mount::DataStorage,
    };
    assert!(file_system.save().is_err());
    assert_eq!(
        postcard::from_bytes::<IFileSystem>(&[2]).unwrap_err(),
        postcard::Error::SerdeDeCustom
    );
    let bytes = postcard::to_allocvec(&(2u8, b"/a")).unwrap();
    assert!(postcard::from_bytes::<IFile>(&bytes).is_err());
    // The SD card's and the data file system's still are.
    assert_eq!(
        postcard::from_bytes::<IFileSystem>(&[0]).unwrap().mount,
        Mount::SdCard
    );
    assert_eq!(
        postcard::from_bytes::<IFileSystem>(&[1]).unwrap().mount,
        Mount::DataFileSystem
    );
}
