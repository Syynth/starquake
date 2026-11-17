//! `fsp-srv`'s data storage through the real dispatcher: the title's RomFS, opened as libnx opens
//! it (`fsOpenDataStorageByCurrentProcess`), sized (`fsStorageGetSize`) and read
//! (`fsStorageRead`, on `fsp-srv`'s workers), served by the run's [`FspHost`] behind the
//! environment.

use std::sync::atomic::Ordering;

use starquake_loader::nup::build::{self, Content, PackageSpec, ROMFS_DIR_NAME, Tree};
use starquake_loader::nup::storage::{CACHE_CAP, RomFsStorage};
use starquake_loader::nup::{Nup, PackageKind};
use starquake_loader::read_at::Positional;
use starquake_services::fsp::IStorage;

use super::*;
use crate::title::RomFs;

/// A title RomFS with one file.
fn romfs() -> RomFs {
    let tree = Tree::new()
        .with_file(
            &format!("{ROMFS_DIR_NAME}/hello.txt"),
            Content::zstd(b"hello from the RomFS".to_vec()),
        )
        .unwrap();
    let base = PackageSpec::new(PackageKind::Base, 0x0100_0000_0000_9100, 0, tree);
    let nup = Nup::open(Positional::new(build::build(&[base]).unwrap().bytes)).unwrap();
    let dir = nup.packages()[0].romfs.unwrap();
    RomFs::new(RomFsStorage::new(nup, dir, CACHE_CAP).unwrap())
}

/// A rig whose environment serves `romfs` as the data storage.
fn rig_with(romfs: Option<RomFs>) -> Rig {
    let host = FspHost::new(None, romfs);
    Rig::with_env(LiveEnv::new(FixedClock::new(0, 1), 0).with_host_io(host))
}

#[test]
fn fsp_srv_opens_the_titles_romfs_as_its_data_storage() {
    let romfs = romfs();
    let size = romfs.size();
    let romfs_copy = romfs.clone();
    let rig = rig_with(Some(romfs));
    let fsp = rig.service(fsp::NAME);
    let root = rig.convert(fsp);

    let open = domain(root, &FspSrv, "open_data_storage_by_current_process");
    let (code, buffer) = rig.send(fsp, &open);
    let (word, _, objects) = domain_reply(&buffer, 0);
    assert_eq!((code, word, objects.len()), (ResultCode::SUCCESS, 0, 1));
    let storage = objects[0];

    let get_size = domain(storage, &IStorage, "get_size");
    let (code, buffer) = rig.send(fsp, &get_size);
    let (word, data, _) = domain_reply(&buffer, size_of::<i64>());
    assert_eq!((code, word), (ResultCode::SUCCESS, 0));
    assert_eq!(data, size.to_le_bytes());

    // A read is deferred to `fsp-srv`'s workers, and the image's bytes arrive with the reply.
    let (code, buffer) = rig.send_deferred(fsp, &storage_read(storage, 0x200, 4));
    assert_eq!(code, ResultCode::SUCCESS);
    assert_eq!(domain_reply(&buffer, 0).0, 0);
    let mut read_bytes = [0; 4];
    rig.memory.read(FIRMWARE, &mut read_bytes).unwrap();
    assert_eq!(read_bytes, image_at(&romfs_copy, 0x200, 4)[..]);
}

/// `IStorage::Read` of `len` bytes of the storage `storage` from `offset`, into [`FIRMWARE`].
fn storage_read(storage: u32, offset: i64, len: u64) -> RequestBuilder {
    let buffer_room = BufferDescriptor {
        address: FIRMWARE,
        size: len,
        flags: 0,
    };
    let size = i64::try_from(len).unwrap();
    let payload = [offset.to_le_bytes(), size.to_le_bytes()].concat();
    domain(storage, &IStorage, "read")
        .data(&payload)
        .receive_buffer(buffer_room)
}

/// `len` bytes of `romfs`'s image from `offset`, read directly.
fn image_at(romfs: &RomFs, offset: u64, len: usize) -> Vec<u8> {
    let mut bytes = vec![0; len];
    assert_eq!(romfs.read_at(offset, &mut bytes).unwrap(), len);
    bytes
}

/// Opens the data storage on a fresh session of `rig`'s `fsp-srv`: the session and the storage's
/// object.
fn open_storage(rig: &Rig) -> (Handle, u32) {
    let fsp = rig.service(fsp::NAME);
    let root = rig.convert(fsp);
    let open = domain(root, &FspSrv, "open_data_storage_by_current_process");
    let (code, buffer) = rig.send(fsp, &open);
    let (word, _, objects) = domain_reply(&buffer, 0);
    assert_eq!((code, word, objects.len()), (ResultCode::SUCCESS, 0, 1));
    (fsp, objects[0])
}

#[test]
fn a_read_the_title_file_fails_fails_the_same_way_every_time() {
    let (romfs, source) = crate::hle::fsp_host::tests::romfs();
    let rig = rig_with(Some(romfs.clone()));
    let (fsp, storage) = open_storage(&rig);
    source.failing.store(true, Ordering::SeqCst);
    let failed = fsp::result::HOST_IO_FAILED.raw();
    // The file's data needs the title file, which fails; a sticky failure, once it is back.
    let (code, buffer) = rig.send_deferred(fsp, &storage_read(storage, 0x200, 4));
    assert_eq!(
        (code, domain_reply(&buffer, 0).0),
        (ResultCode::SUCCESS, failed)
    );
    source.failing.store(false, Ordering::SeqCst);
    let (code, buffer) = rig.send_deferred(fsp, &storage_read(storage, 0x200, 4));
    assert_eq!(
        (code, domain_reply(&buffer, 0).0),
        (ResultCode::SUCCESS, failed)
    );
    // The image's header needs no read of the title file, so it still reads.
    let (code, buffer) = rig.send_deferred(fsp, &storage_read(storage, 0, 4));
    assert_eq!((code, domain_reply(&buffer, 0).0), (ResultCode::SUCCESS, 0));
    let mut read_bytes = [0; 4];
    rig.memory.read(FIRMWARE, &mut read_bytes).unwrap();
    assert_eq!(read_bytes, image_at(&romfs, 0, 4)[..]);
}

#[test]
fn a_run_without_a_title_romfs_has_no_data_storage() {
    let rig = rig_with(None);
    let fsp = rig.service(fsp::NAME);
    let root = rig.convert(fsp);
    let open = domain(root, &FspSrv, "open_data_storage_by_current_process");
    let (code, buffer) = rig.send(fsp, &open);
    assert_eq!(code, ResultCode::SUCCESS);
    let (word, _, objects) = domain_reply(&buffer, 0);
    assert_eq!((word, objects.len()), (fsp::result::UNKNOWN.raw(), 0));
}
