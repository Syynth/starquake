//! The run's `fsp-srv` host: the SD card and the data storage, told apart by mount, on synthetic
//! containers built in the test.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use starquake_loader::nup::build::{self, Content, PackageSpec, ROMFS_DIR_NAME, Tree};
use starquake_loader::nup::storage::{CACHE_CAP, RomFsStorage};
use starquake_loader::nup::{Nup, PackageKind};
use starquake_loader::read_at::{Positional, ReadAt};
use starquake_services::fsp::host::CHANNEL;

use super::*;
use crate::hle::tests::Scratch;

/// A source over a container's bytes that can be made to fail every read.
pub(crate) struct Source {
    bytes: Vec<u8>,
    pub(crate) failing: AtomicBool,
}

impl ReadAt for Source {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        if self.failing.load(Ordering::SeqCst) {
            return Err(io::Error::other("the disk is gone"));
        }
        self.bytes.read_at(offset, buf)
    }

    fn size(&self) -> io::Result<u64> {
        self.bytes.size()
    }
}

/// A title RomFS holding one file, and the source it reads from.
pub(crate) fn romfs() -> (RomFs, Arc<Source>) {
    let tree = Tree::new()
        .with_file(
            &format!("{ROMFS_DIR_NAME}/data.bin"),
            Content::raw((0u8..=200).collect::<Vec<u8>>()),
        )
        .unwrap();
    let base = PackageSpec::new(PackageKind::Base, 0x0100_0000_0000_9000, 0, tree);
    let built = build::build(&[base]).unwrap();
    let source = Arc::new(Source {
        bytes: built.bytes,
        failing: AtomicBool::new(false),
    });
    let nup = Nup::open(Positional::new(Arc::clone(&source))).unwrap();
    let dir = nup.packages()[0].romfs.unwrap();
    let storage = RomFsStorage::new(nup, dir, CACHE_CAP).unwrap();
    (RomFs::new(storage), source)
}

/// The whole image, read directly.
fn image(romfs: &RomFs) -> Vec<u8> {
    let mut out = vec![0; usize::try_from(romfs.size()).unwrap()];
    assert_eq!(romfs.read_at(0, &mut out).unwrap(), out.len());
    out
}

fn ask(host: &mut FspHost, request: &HostRequest) -> HostIoReply {
    host.handle(&request.to_host_io())
}

const STORAGE_PROBE: HostRequest = HostRequest::Probe {
    mount: Mount::DataStorage,
};

fn storage_size(path: &[u8]) -> HostRequest {
    HostRequest::Size {
        mount: Mount::DataStorage,
        path: path.to_vec(),
    }
}

fn storage_read(offset: u64, len: u64) -> HostRequest {
    HostRequest::Read {
        mount: Mount::DataStorage,
        path: Vec::new(),
        offset,
        len,
    }
}

#[test]
fn the_data_storage_is_the_titles_romfs_image() {
    let (romfs, _) = romfs();
    let whole = image(&romfs);
    let size = romfs.size();
    let mut host = FspHost::new(None, Some(romfs));
    assert!(!host.is_empty());
    assert_eq!(
        ask(&mut host, &STORAGE_PROBE),
        HostIoReply::Data(Vec::new())
    );
    assert_eq!(
        ask(&mut host, &storage_size(b"")),
        HostIoReply::Data(size.to_le_bytes().to_vec())
    );
    // Reads anywhere, up to and past the end.
    let len = whole.len();
    let cases = [
        (0, 16, &whole[..16]),
        (0x200, 0x40, &whole[0x200..0x240]),
        (size - 3, 10, &whole[len - 3..]),
        (size, 10, &[][..]),
        (size + 1, 10, &[][..]),
        (u64::MAX, u64::MAX, &[][..]),
    ];
    for (offset, count, expected) in cases {
        assert_eq!(
            ask(&mut host, &storage_read(offset, count)),
            HostIoReply::Data(expected.to_vec()),
            "{count} at {offset}"
        );
    }
    // A length far past the image holds only what is there.
    assert_eq!(
        ask(&mut host, &storage_read(0, u64::MAX)),
        HostIoReply::Data(whole)
    );
}

#[test]
fn the_data_storage_has_no_files_of_its_own() {
    let (romfs, _) = romfs();
    let mut host = FspHost::new(None, Some(romfs));
    assert_eq!(
        ask(&mut host, &storage_size(b"/data.bin")),
        HostFailure::NotFound.reply()
    );
    let read = HostRequest::Read {
        mount: Mount::DataStorage,
        path: b"/data.bin".to_vec(),
        offset: 0,
        len: 4,
    };
    assert_eq!(ask(&mut host, &read), HostFailure::NotFound.reply());
}

#[test]
fn without_a_romfs_or_a_folder_each_mount_is_no_device() {
    let mut host = FspHost::default();
    assert!(host.is_empty());
    let requests = [
        STORAGE_PROBE,
        storage_size(b""),
        storage_read(0, 4),
        HostRequest::Probe {
            mount: Mount::SdCard,
        },
    ];
    for request in requests {
        assert_eq!(
            ask(&mut host, &request),
            HostFailure::NoDevice.reply(),
            "{request:?}"
        );
    }
}

#[test]
fn each_request_goes_to_its_own_mount() {
    let scratch = Scratch::new("fsp-host");
    scratch.file("card.txt", b"on the card");
    let (romfs, _) = romfs();
    let size = romfs.size();
    let mut host = FspHost::new(Some(HostFolder::new(scratch.path())), Some(romfs));
    let card = HostRequest::Size {
        mount: Mount::SdCard,
        path: b"/card.txt".to_vec(),
    };
    assert_eq!(
        ask(&mut host, &card),
        HostIoReply::Data(11u64.to_le_bytes().to_vec())
    );
    assert_eq!(
        ask(&mut host, &storage_size(b"")),
        HostIoReply::Data(size.to_le_bytes().to_vec())
    );
    // The card alone: the storage is not there, the card still is.
    let mut card_only = FspHost::new(Some(HostFolder::new(scratch.path())), None);
    assert_eq!(
        ask(&mut card_only, &STORAGE_PROBE),
        HostFailure::NoDevice.reply()
    );
    assert_eq!(
        ask(&mut card_only, &card),
        HostIoReply::Data(11u64.to_le_bytes().to_vec())
    );
}

#[test]
fn a_read_the_image_fails_is_an_io_failure() {
    let (romfs, source) = romfs();
    let mut host = FspHost::new(None, Some(romfs.clone()));
    source.failing.store(true, Ordering::SeqCst);
    // The header and tables need no read of the title file; the file's data does.
    assert!(matches!(
        ask(&mut host, &storage_read(0, 0x50)),
        HostIoReply::Data(_)
    ));
    assert_eq!(
        ask(&mut host, &storage_read(0x200, 4)),
        HostFailure::Io.reply()
    );
    // Sticky, as the loader's storage keeps it, once the disk is back.
    source.failing.store(false, Ordering::SeqCst);
    assert_eq!(
        ask(&mut host, &storage_read(0x200, 4)),
        HostFailure::Io.reply()
    );
}

/// What the data file system's tests read: a file at the root, raw, and one in a folder,
/// compressed, every byte distinct from its neighbours.
fn top_bytes() -> Vec<u8> {
    (0u8..=200).map(|b| b.wrapping_mul(7) ^ 0x3C).collect()
}

fn inner_bytes() -> Vec<u8> {
    (0u8..=90).map(|b| b.wrapping_mul(5) ^ 0xA1).collect()
}

/// A title RomFS holding `/top.bin` and `/dir/inner.bin`, and the source it reads from.
fn files_romfs() -> (RomFs, Arc<Source>) {
    let tree = Tree::new()
        .with_file(
            &format!("{ROMFS_DIR_NAME}/top.bin"),
            Content::raw(top_bytes()),
        )
        .unwrap()
        .with_file(
            &format!("{ROMFS_DIR_NAME}/dir/inner.bin"),
            Content::zstd(inner_bytes()),
        )
        .unwrap();
    let base = PackageSpec::new(PackageKind::Base, 0x0100_0000_0000_9400, 0, tree);
    let source = Arc::new(Source {
        bytes: build::build(&[base]).unwrap().bytes,
        failing: AtomicBool::new(false),
    });
    let nup = Nup::open(Positional::new(Arc::clone(&source))).unwrap();
    let dir = nup.packages()[0].romfs.unwrap();
    let storage = RomFsStorage::new(nup, dir, CACHE_CAP).unwrap();
    (RomFs::new(storage), source)
}

const FILES_PROBE: HostRequest = HostRequest::Probe {
    mount: Mount::DataFileSystem,
};

fn files_size(path: &[u8]) -> HostRequest {
    HostRequest::Size {
        mount: Mount::DataFileSystem,
        path: path.to_vec(),
    }
}

fn files_read(path: &[u8], offset: u64, len: u64) -> HostRequest {
    HostRequest::Read {
        mount: Mount::DataFileSystem,
        path: path.to_vec(),
        offset,
        len,
    }
}

#[test]
fn the_data_file_system_is_the_files_of_the_titles_romfs() {
    let (romfs, _) = files_romfs();
    let mut host = FspHost::new(None, Some(romfs));
    assert_eq!(ask(&mut host, &FILES_PROBE), HostIoReply::Data(Vec::new()));
    for (path, bytes) in [
        (&b"/top.bin"[..], top_bytes()),
        (b"/dir/inner.bin", inner_bytes()),
    ] {
        let size = u64::try_from(bytes.len()).unwrap();
        assert_eq!(
            ask(&mut host, &files_size(path)),
            HostIoReply::Data(size.to_le_bytes().to_vec())
        );
        // Reads anywhere in the file, up to and past its end.
        let len = bytes.len();
        let cases = [
            (0, 16, &bytes[..16]),
            (5, 3, &bytes[5..8]),
            (size - 3, 10, &bytes[len - 3..]),
            (size - 1, 1, &bytes[len - 1..]),
            (size, 10, &[][..]),
            (size + 1, 10, &[][..]),
            (u64::MAX, u64::MAX, &[][..]),
        ];
        for (offset, count, expected) in cases {
            assert_eq!(
                ask(&mut host, &files_read(path, offset, count)),
                HostIoReply::Data(expected.to_vec()),
                "{count} at {offset} of {path:?}"
            );
        }
        // A length far past the file holds only what is there.
        assert_eq!(
            ask(&mut host, &files_read(path, 0, u64::MAX)),
            HostIoReply::Data(bytes.clone())
        );
    }
}

#[test]
fn a_file_of_the_data_file_system_is_the_storage_s_bytes_at_its_data() {
    // The storage and the file system serve one image.
    let (romfs, _) = files_romfs();
    let whole = image(&romfs);
    let mut host = FspHost::new(None, Some(romfs));
    let HostIoReply::Data(file) = ask(&mut host, &files_read(b"/top.bin", 0, u64::MAX)) else {
        unreachable!("the file did not read");
    };
    let at = whole
        .windows(file.len())
        .position(|window| window == file)
        .expect("the file's bytes are not in the image");
    let offset = u64::try_from(at).unwrap();
    let len = u64::try_from(file.len()).unwrap();
    assert_eq!(
        ask(&mut host, &storage_read(offset, len)),
        HostIoReply::Data(file)
    );
}

#[test]
fn a_path_that_names_no_file_is_not_found() {
    let (romfs, _) = files_romfs();
    let mut host = FspHost::new(None, Some(romfs));
    let paths: [&[u8]; 8] = [
        b"/",
        b"/dir",
        b"/missing.bin",
        b"/dir/missing.bin",
        b"/top.bin/x",
        b"/inner.bin",
        b"top.bin",
        b"/\xFF.bin",
    ];
    for path in paths {
        assert_eq!(
            ask(&mut host, &files_size(path)),
            HostFailure::NotFound.reply(),
            "{path:?}"
        );
        assert_eq!(
            ask(&mut host, &files_read(path, 0, 4)),
            HostFailure::NotFound.reply(),
            "{path:?}"
        );
    }
    // The paths beside them still are.
    assert!(matches!(
        ask(&mut host, &files_size(b"/dir/inner.bin")),
        HostIoReply::Data(_)
    ));
}

#[test]
fn without_a_romfs_the_data_file_system_is_no_device() {
    let mut host = FspHost::default();
    for request in [
        FILES_PROBE,
        files_size(b"/top.bin"),
        files_read(b"/top.bin", 0, 4),
    ] {
        assert_eq!(
            ask(&mut host, &request),
            HostFailure::NoDevice.reply(),
            "{request:?}"
        );
    }
}

#[test]
fn a_file_the_title_file_fails_to_read_is_an_io_failure() {
    let (romfs, source) = files_romfs();
    let mut host = FspHost::new(None, Some(romfs));
    source.failing.store(true, Ordering::SeqCst);
    // The tables need no read of the title file: the probe and the size still answer.
    assert_eq!(ask(&mut host, &FILES_PROBE), HostIoReply::Data(Vec::new()));
    let size = u64::try_from(top_bytes().len()).unwrap();
    assert_eq!(
        ask(&mut host, &files_size(b"/top.bin")),
        HostIoReply::Data(size.to_le_bytes().to_vec())
    );
    // The file's data does.
    let read = files_read(b"/dir/inner.bin", 0, 4);
    assert_eq!(ask(&mut host, &read), HostFailure::Io.reply());
    // Sticky, as the loader's storage keeps it, once the disk is back.
    source.failing.store(false, Ordering::SeqCst);
    assert_eq!(ask(&mut host, &read), HostFailure::Io.reply());
}

#[test]
fn a_copy_of_the_host_is_the_same_host() {
    let (romfs, _) = files_romfs();
    let mut host = FspHost::new(None, Some(romfs.clone()));
    // Reading the tables does not make it another host, and its copy reads them again.
    assert_eq!(ask(&mut host, &FILES_PROBE), HostIoReply::Data(Vec::new()));
    let mut copy = host.clone();
    assert_eq!(copy, host);
    assert_eq!(copy, FspHost::new(None, Some(romfs)));
    assert_eq!(
        ask(&mut copy, &files_read(b"/top.bin", 0, 4)),
        HostIoReply::Data(top_bytes()[..4].to_vec())
    );
    let (other, _) = files_romfs();
    assert_ne!(host, FspHost::new(None, Some(other)));
}

#[test]
fn other_channels_and_bad_requests_are_turned_away() {
    let (romfs, _) = romfs();
    let mut host = FspHost::new(None, Some(romfs));
    let mut request = STORAGE_PROBE.to_host_io();
    request.channel = CHANNEL + 1;
    assert_eq!(host.handle(&request), HostIoReply::Unavailable);
    let garbage = HostIoRequest {
        channel: CHANNEL,
        payload: vec![9; 4],
    };
    assert_eq!(host.handle(&garbage), HostFailure::BadRequest.reply());
}
