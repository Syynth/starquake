//! `fsp-srv`'s data file system through the real dispatcher: the files of the title's RomFS,
//! from a synthetic NUP, opened as libnx opens it (`fsOpenDataFileSystemByCurrentProcess`), a
//! file in it opened (`fsFsOpenFile`), sized (`fsFileGetSize`) and read (`fsFileRead`, on
//! `fsp-srv`'s workers), served by the run's [`FspHost`] behind the environment.

use starquake_loader::nup::build::{self, Content, PackageSpec, ROMFS_DIR_NAME, Tree};
use starquake_loader::nup::storage::{CACHE_CAP, RomFsStorage};
use starquake_loader::nup::{Nup, PackageKind};
use starquake_loader::read_at::Positional;

use super::*;
use crate::title::RomFs;

/// The file the guest reads: every byte distinct from its neighbours.
fn text() -> Vec<u8> {
    (0u8..=120).map(|b| b.wrapping_mul(11) ^ 0x6B).collect()
}

/// Where the file is in the RomFS, and as the guest names it.
const NAME: &str = "levels/one.dat";
const GUEST_PATH: &[u8] = b"/levels/one.dat";

/// A title RomFS, from a synthetic NUP, holding [`text`] at [`NAME`] (compressed) and another
/// file beside it.
fn romfs() -> RomFs {
    let tree = Tree::new()
        .with_file(&format!("{ROMFS_DIR_NAME}/{NAME}"), Content::zstd(text()))
        .unwrap()
        .with_file(
            &format!("{ROMFS_DIR_NAME}/other.dat"),
            Content::raw(vec![0xEE; 64]),
        )
        .unwrap();
    let base = PackageSpec::new(PackageKind::Base, 0x0100_0000_0000_9500, 0, tree);
    let nup = Nup::open(Positional::new(build::build(&[base]).unwrap().bytes)).unwrap();
    let dir = nup.packages()[0].romfs.unwrap();
    RomFs::new(RomFsStorage::new(nup, dir, CACHE_CAP).unwrap())
}

/// A rig whose environment serves `romfs` as the title's.
fn rig_with(romfs: Option<RomFs>) -> Rig {
    let host = FspHost::new(None, romfs);
    Rig::with_env(LiveEnv::new(FixedClock::new(0, 1), 0).with_host_io(host))
}

/// `OpenFile` of `path`, for reading, on the file system `fs`.
fn open_file(rig: &Rig, fs: u32, path: &[u8]) -> RequestBuilder {
    let mut buffer = path.to_vec();
    buffer.resize(fsp::PATH_BUFFER_SIZE, 0);
    rig.memory.write(PATH, &buffer).unwrap();
    let descriptor = StaticDescriptor {
        counter: 0,
        address: PATH,
        size: u16::try_from(buffer.len()).unwrap(),
    };
    domain(fs, &IFileSystem::default(), "open_file")
        .data(&fsp::open_mode::READ.to_le_bytes())
        .static_descriptor(descriptor)
}

#[test]
fn a_guest_opens_the_titles_romfs_as_a_file_system_and_reads_a_file() {
    let text = text();
    let rig = rig_with(Some(romfs()));
    let fsp = rig.service(fsp::NAME);
    let root = rig.convert(fsp);

    let open = domain(root, &FspSrv, "open_data_file_system_by_current_process");
    let (code, buffer) = rig.send(fsp, &open);
    let (word, _, objects) = domain_reply(&buffer, 0);
    assert_eq!((code, word, objects.len()), (ResultCode::SUCCESS, 0, 1));
    let fs = objects[0];

    let (code, buffer) = rig.send(fsp, &open_file(&rig, fs, GUEST_PATH));
    let (word, _, objects) = domain_reply(&buffer, 0);
    assert_eq!((code, word, objects.len()), (ResultCode::SUCCESS, 0, 1));
    let file = objects[0];

    let get_size = domain(file, &IFile::default(), "get_size");
    let (code, buffer) = rig.send(fsp, &get_size);
    let (word, data, _) = domain_reply(&buffer, size_of::<i64>());
    assert_eq!((code, word), (ResultCode::SUCCESS, 0));
    assert_eq!(data, u64::try_from(text.len()).unwrap().to_le_bytes());

    // A read is deferred to `fsp-srv`'s workers, and the file's bytes arrive with the reply:
    // from offset 7, more than the file has left, so the count is what there was.
    let offset = 7;
    let capacity = 0x80;
    let buffer_room = BufferDescriptor {
        address: FIRMWARE,
        size: capacity,
        flags: 0,
    };
    let payload = [
        &0_u32.to_le_bytes()[..],
        &[0; 4],
        &i64::try_from(offset).unwrap().to_le_bytes(),
        &i64::try_from(capacity).unwrap().to_le_bytes(),
    ]
    .concat();
    let read = domain(file, &IFile::default(), "read")
        .data(&payload)
        .receive_buffer(buffer_room);
    let (code, buffer) = rig.send_deferred(fsp, &read);
    let expected = &text[offset..];
    let (word, data, _) = domain_reply(&buffer, size_of::<i64>());
    assert_eq!((code, word), (ResultCode::SUCCESS, 0));
    assert_eq!(data, i64::try_from(expected.len()).unwrap().to_le_bytes());
    let mut read_bytes = vec![0; expected.len()];
    rig.memory.read(FIRMWARE, &mut read_bytes).unwrap();
    assert_eq!(read_bytes, expected);

    // A path the RomFS does not hold is Eden's `ResultPathNotFound`.
    let (code, buffer) = rig.send(fsp, &open_file(&rig, fs, b"/levels/two.dat"));
    let (word, _, objects) = domain_reply(&buffer, 0);
    assert_eq!(code, ResultCode::SUCCESS);
    assert_eq!(
        (word, objects.len()),
        (fsp::result::PATH_NOT_FOUND.raw(), 0)
    );
}

#[test]
fn a_run_without_a_title_romfs_has_no_data_file_system() {
    let rig = rig_with(None);
    let fsp = rig.service(fsp::NAME);
    let root = rig.convert(fsp);
    let open = domain(root, &FspSrv, "open_data_file_system_by_current_process");
    let (code, buffer) = rig.send(fsp, &open);
    assert_eq!(code, ResultCode::SUCCESS);
    let (word, _, objects) = domain_reply(&buffer, 0);
    assert_eq!((word, objects.len()), (fsp::result::UNKNOWN.raw(), 0));
}
