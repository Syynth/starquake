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
struct Source {
    bytes: Vec<u8>,
    failing: AtomicBool,
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
fn romfs() -> (RomFs, Arc<Source>) {
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
