//! The data file system: `OpenDataFileSystemByCurrentProcess` and the `IFileSystem` it hands
//! out, over the fake host's `RomFS` files, laid out as libnx lays its requests out
//! (`fsOpenDataFileSystemByCurrentProcess`, then `fsFsOpenFile`, `fsFileGetSize` and
//! `fsFileRead`).

use super::storage::{Broken, broken_error, open_with};
use super::*;
use crate::fsp::command::fsp_srv::OPEN_DATA_FILE_SYSTEM_BY_CURRENT_PROCESS;
use crate::fsp::command::storage::GET_SIZE as STORAGE_GET_SIZE;

/// The `RomFS` file the tests read: every byte distinct from its neighbours, and from
/// [`HELLO`]'s, so a read of the SD card's file instead shows.
fn romfs_text() -> Vec<u8> {
    (0u8..48).map(|b| b.wrapping_mul(9) ^ 0xC3).collect()
}

const ROMFS_PATH: &[u8] = b"/data/level.bin";

impl Rig {
    /// A rig whose run has a `RomFS` holding [`romfs_text`] at [`ROMFS_PATH`] and a folder at
    /// `/data` (and the SD card).
    fn with_romfs() -> Self {
        let rig = Self::new();
        rig.env.host().files = Some(BTreeMap::from([
            (ROMFS_PATH.to_vec(), Entry::File(romfs_text())),
            (b"/data".to_vec(), Entry::Folder),
        ]));
        rig
    }

    /// `OpenDataFileSystemByCurrentProcess` on the rig's session: the file system's session.
    fn open_data_files(&mut self) -> ObjectId {
        let reply = self.send(
            To::Session(self.session),
            OPEN_DATA_FILE_SYSTEM_BY_CURRENT_PROCESS,
            |r| r,
        );
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.moves.len(), 1);
        self.object_of(reply.moves[0])
    }
}

#[test]
fn the_data_file_system_is_command_2() {
    // Eden's `FSP_SRV` table: `{2, D<&FSP_SRV::OpenDataFileSystemByCurrentProcess>, ..}`; libnx
    // `fsOpenDataFileSystemByCurrentProcess`: `_fsCmdGetSession(&g_fsSrv, &out->s, 2)`.
    assert_eq!(OPEN_DATA_FILE_SYSTEM_BY_CURRENT_PROCESS, 2);
    assert_eq!(
        FspSrv.commands().name(
            crate::framework::Protocol::Cmif,
            OPEN_DATA_FILE_SYSTEM_BY_CURRENT_PROCESS
        ),
        Some("open_data_file_system_by_current_process")
    );
}

#[test]
fn a_file_of_the_data_file_system_opens_sizes_and_reads() {
    let mut rig = Rig::with_romfs();
    let fs = rig.open_data_files();
    // It probed the data file system, not the SD card.
    let probe = HostRequest::Probe {
        mount: Mount::DataFileSystem,
    };
    assert_eq!(rig.env.asked().last(), Some(&probe));
    let reply = rig.open_file(fs, ROMFS_PATH);
    assert_eq!(reply.result, ResultCode::SUCCESS);
    let file = rig.object_of(reply.moves[0]);
    let text = romfs_text();
    let size = rig.send(To::Session(file), GET_SIZE, |r| r);
    assert_eq!(size, s64(i64::try_from(text.len()).unwrap()));
    // A read is deferred, asks the host for the file on the data file system, and delivers it.
    rig.defer(To::Session(file), READ, read(3, 5, 16));
    rig.jobs.run();
    assert_eq!(rig.deliver(SessionKind::Session), s64(5));
    assert_eq!(rig.kernel.memory(DATA_AT, 5).unwrap(), &text[3..8]);
    let asked = HostRequest::Read {
        mount: Mount::DataFileSystem,
        path: ROMFS_PATH.to_vec(),
        offset: 3,
        len: 5,
    };
    assert_eq!(rig.env.asked().last(), Some(&asked));
}

/// `OpenFile` of `path` with `mode` on the data file system of a fresh [`Rig::with_romfs`] (the
/// fake kernel keeps the first path a rig maps).
fn open_on_data_files(path: &[u8], mode: u32) -> Answered {
    let mut rig = Rig::with_romfs();
    let fs = rig.open_data_files();
    rig.open_file_with(fs, path, mode)
}

#[test]
fn the_data_file_system_takes_the_sd_card_s_checks() {
    let read = open_mode::READ;
    assert_eq!(
        open_on_data_files(ROMFS_PATH, read).result,
        ResultCode::SUCCESS
    );
    // What is not a file there, and the card's file, which is not in the RomFS.
    for path in [&b"/data"[..], b"/missing.bin", HELLO_PATH] {
        assert_eq!(
            open_on_data_files(path, read),
            only(result::PATH_NOT_FOUND),
            "{path:?}"
        );
    }
    // A mode that is not exactly read, and a path that is not one.
    let writing = open_mode::READ_WRITE;
    let reply = open_on_data_files(ROMFS_PATH, writing);
    assert_eq!(reply, only(result::INVALID_OPEN_MODE));
    assert_eq!(
        open_on_data_files(b"level.bin", read),
        only(result::INVALID_PATH_FORMAT)
    );
}

#[test]
fn without_a_romfs_opening_the_data_file_system_is_eden_s_unknown() {
    // The run's title has no RomFS: the host has no data file system.
    let mut rig = Rig::new();
    let reply = rig.send(
        To::Session(rig.session),
        OPEN_DATA_FILE_SYSTEM_BY_CURRENT_PROCESS,
        |r| r,
    );
    assert_eq!(reply, only(result::UNKNOWN));
    assert_eq!(rig.server.session_count(), 1);
    // No handler at all answers the same, and so does a probe answered with bytes.
    let command = OPEN_DATA_FILE_SYSTEM_BY_CURRENT_PROCESS;
    assert_eq!(
        open_with(Arc::new(NoHost), command, unwatched()),
        only(result::UNKNOWN)
    );
    let rig = Rig::with_romfs();
    rig.env.host().forced = Some((Asked::Probe, Ok(HostIoReply::Data(vec![0]))));
    let env: Arc<dyn Env> = Arc::<FakeEnv>::clone(&rig.env);
    assert_eq!(open_with(env, command, unwatched()), only(result::UNKNOWN));
    // An environment that fails stops the run instead.
    let halts = Arc::new(Halts::default());
    let halt: Arc<dyn Halt> = Arc::<Halts>::clone(&halts);
    let reply = open_with(Arc::new(Broken), command, halt);
    assert_eq!(reply, only(DEFERRED_REPLY_DROPPED));
    assert_eq!(halts.stops(), [Stop::Handler(broken_error())]);
}

#[test]
fn a_data_file_system_s_snapshot_bytes_are_pinned() {
    // `DataFileSystem` is `Mount`'s second variant, so its index is 1.
    let mount = Mount::DataFileSystem;
    assert_eq!(IFileSystem { mount }.save().unwrap(), [1]);
    let file = IFile {
        mount,
        path: ROMFS_PATH.to_vec(),
    };
    let len = u8::try_from(ROMFS_PATH.len()).unwrap();
    assert_eq!(file.save().unwrap(), [&[1, len][..], ROMFS_PATH].concat());

    // A server saved with the data file system and a file on it open restores them, and the
    // restored file reads from the data file system.
    let mut rig = Rig::with_romfs();
    let fs = rig.open_data_files();
    let reply = rig.open_file(fs, ROMFS_PATH);
    let file = rig.object_of(reply.moves[0]);
    let saved = rig.server.save().unwrap();
    let env: Arc<dyn Env> = Arc::<FakeEnv>::clone(&rig.env);
    let executor: Arc<dyn Executor> = Arc::<Jobs>::clone(&rig.jobs);
    let mut restored = server(UnknownPolicy::Strict, env, unwatched()).with_executor(executor);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.session_count(), 3);
    rig.server = restored;
    rig.defer(To::Session(file), READ, read(0, 4, 16));
    rig.jobs.run();
    assert_eq!(rig.deliver(SessionKind::Session), s64(4));
    assert_eq!(rig.kernel.memory(DATA_AT, 4).unwrap(), &romfs_text()[..4]);
    // And a file opened on the restored file system is the data file system's too.
    let reply = rig.open_file(fs, ROMFS_PATH);
    assert_eq!(reply.result, ResultCode::SUCCESS);
}

#[test]
fn every_open_of_the_data_file_system_asks_the_host_again() {
    // Eden keeps the RomFS it opened (`FSP_SRV::romfs`); here each open probes the host.
    let mut rig = Rig::with_romfs();
    rig.open_data_files();
    rig.open_data_files();
    let probe = HostRequest::Probe {
        mount: Mount::DataFileSystem,
    };
    assert_eq!(rig.env.asked(), [probe.clone(), probe.clone()]);
    // So a third open the host refuses is refused, though two before it opened.
    rig.env.host().files = None;
    let reply = rig.send(
        To::Session(rig.session),
        OPEN_DATA_FILE_SYSTEM_BY_CURRENT_PROCESS,
        |r| r,
    );
    assert_eq!(reply, only(result::UNKNOWN));
    assert_eq!(rig.env.asked(), [probe.clone(), probe.clone(), probe]);
}

#[test]
fn a_romfs_session_restored_without_a_romfs_answers_eden_s_unknown() {
    // A run with a RomFS saves a data file system, a file on it and the data storage, open.
    let mut rig = Rig::with_romfs();
    rig.env.host().storage = Some(romfs_text());
    let fs = rig.open_data_files();
    let reply = rig.open_file(fs, ROMFS_PATH);
    let file = rig.object_of(reply.moves[0]);
    let storage = rig.open_storage();
    let saved = rig.server.save().unwrap();
    // They are restored into a run that has none.
    {
        let mut host = rig.env.host();
        host.files = None;
        host.storage = None;
    }
    let env: Arc<dyn Env> = Arc::<FakeEnv>::clone(&rig.env);
    let executor: Arc<dyn Executor> = Arc::<Jobs>::clone(&rig.jobs);
    let mut restored = server(UnknownPolicy::Strict, env, unwatched()).with_executor(executor);
    restored.restore(&saved).unwrap();
    rig.server = restored;
    // Each command on them answers Eden's code for a process with no RomFS, not the SD card's.
    assert_eq!(rig.open_file(fs, ROMFS_PATH), only(result::UNKNOWN));
    assert_eq!(
        rig.send(To::Session(file), GET_SIZE, |r| r),
        only(result::UNKNOWN)
    );
    rig.defer(To::Session(file), READ, read(0, 4, 16));
    rig.jobs.run();
    assert_eq!(rig.deliver(SessionKind::Session), only(result::UNKNOWN));
    assert_eq!(rig.kernel.memory(DATA_AT, 16).unwrap(), [STALE; 16]);
    assert_eq!(
        rig.send(To::Session(storage), STORAGE_GET_SIZE, |r| r),
        only(result::UNKNOWN)
    );
    // The SD card beside them keeps its own code when it is not there.
    rig.env.host().card = None;
    let reply = rig.send(
        To::Session(rig.session),
        crate::fsp::command::fsp_srv::OPEN_SD_CARD_FILE_SYSTEM,
        |r| r,
    );
    assert_eq!(reply, only(result::PORT_SD_CARD_NO_DEVICE));
}
