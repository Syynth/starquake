//! `fsp-srv`'s data storage through the real dispatcher: the title's RomFS, opened as libnx opens
//! it (`fsOpenDataStorageByCurrentProcess`) and sized (`fsStorageGetSize`), served by the run's
//! [`FspHost`] behind the environment.

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

    // PROVISIONAL (module docs): a run gives `fsp-srv` no executor, so a read is refused.
    let buffer_room = BufferDescriptor {
        address: FIRMWARE,
        size: 4,
        flags: 0,
    };
    let payload = [0_i64.to_le_bytes(), 4_i64.to_le_bytes()].concat();
    let read = domain(storage, &IStorage, "read")
        .data(&payload)
        .receive_buffer(buffer_room);
    let (code, buffer) = rig.send(fsp, &read);
    assert_eq!(code, ResultCode::SUCCESS);
    assert_eq!(domain_reply(&buffer, 0).0, NOT_SUPPORTED.raw());
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
