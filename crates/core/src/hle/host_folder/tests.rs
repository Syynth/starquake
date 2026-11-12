//! A real host folder answering the host file protocol.

use starquake_services::fsp::host::HostRequest;

use super::*;
use crate::hle::tests::Scratch;

const TEXT: &[u8] = b"0123456789";

fn folder(test: &str) -> (Scratch, HostFolder) {
    let scratch = Scratch::new(test);
    scratch.file("a.txt", TEXT);
    scratch.file("dir/inner.bin", &[7; 3]);
    let folder = HostFolder::new(scratch.path());
    (scratch, folder)
}

fn ask(folder: &mut HostFolder, request: &HostRequest) -> HostIoReply {
    folder.handle(&request.to_host_io())
}

fn size(path: &[u8]) -> HostRequest {
    HostRequest::Size {
        mount: Mount::SdCard,
        path: path.to_vec(),
    }
}

fn read(path: &[u8], offset: u64, len: u64) -> HostRequest {
    HostRequest::Read {
        mount: Mount::SdCard,
        path: path.to_vec(),
        offset,
        len,
    }
}

const PROBE: HostRequest = HostRequest::Probe {
    mount: Mount::SdCard,
};

#[test]
fn a_folder_that_is_there_is_the_sd_card() {
    let (scratch, mut folder) = folder("probe");
    assert_eq!(ask(&mut folder, &PROBE), HostIoReply::Data(Vec::new()));
    // Not there, and a file in its place.
    let mut missing = HostFolder::new(scratch.path().join("missing"));
    assert_eq!(ask(&mut missing, &PROBE), HostFailure::NoDevice.reply());
    let mut file = HostFolder::new(scratch.path().join("a.txt"));
    assert_eq!(ask(&mut file, &PROBE), HostFailure::NoDevice.reply());
}

#[test]
fn a_file_s_size_and_what_is_not_a_file() {
    let (_scratch, mut folder) = folder("size");
    let len = u64::try_from(TEXT.len()).unwrap();
    let data = |len: u64| HostIoReply::Data(len.to_le_bytes().to_vec());
    assert_eq!(ask(&mut folder, &size(b"/a.txt")), data(len));
    assert_eq!(ask(&mut folder, &size(b"/dir/inner.bin")), data(3));
    assert_eq!(
        ask(&mut folder, &size(b"/dir")),
        HostFailure::NotAFile.reply()
    );
    assert_eq!(ask(&mut folder, &size(b"/")), HostFailure::NotAFile.reply());
    assert_eq!(
        ask(&mut folder, &size(b"/b.txt")),
        HostFailure::NotFound.reply()
    );
    // A file where a folder should be.
    let under_a_file = size(b"/a.txt/x");
    assert_eq!(
        ask(&mut folder, &under_a_file),
        HostFailure::NotFound.reply()
    );
}

#[test]
fn reads_stop_at_the_end_of_the_file() {
    let (_scratch, mut folder) = folder("read");
    let data = |bytes: &[u8]| HostIoReply::Data(bytes.to_vec());
    assert_eq!(ask(&mut folder, &read(b"/a.txt", 2, 3)), data(b"234"));
    assert_eq!(ask(&mut folder, &read(b"/a.txt", 8, 5)), data(b"89"));
    // At the end, past it, and a length larger than any file: only what is there.
    assert_eq!(ask(&mut folder, &read(b"/a.txt", 10, 1)), data(b""));
    assert_eq!(ask(&mut folder, &read(b"/a.txt", u64::MAX, 1)), data(b""));
    assert_eq!(ask(&mut folder, &read(b"/a.txt", 0, u64::MAX)), data(TEXT));
    assert_eq!(ask(&mut folder, &read(b"/a.txt", 0, 0)), data(b""));
    assert_eq!(
        ask(&mut folder, &read(b"/dir", 0, 1)),
        HostFailure::NotAFile.reply()
    );
    assert_eq!(
        ask(&mut folder, &read(b"/no", 0, 1)),
        HostFailure::NotFound.reply()
    );
}

#[test]
fn only_plain_paths_inside_the_folder_are_followed() {
    let (scratch, mut folder) = folder("paths");
    scratch.file("../outside.txt", b"secret");
    let refused: [&[u8]; 10] = [
        b"a.txt",
        b"",
        b"//a.txt",
        b"/dir//inner.bin",
        b"/./a.txt",
        b"/dir/../a.txt",
        b"/../outside.txt",
        b"/dir\\inner.bin",
        b"/a.txt/",
        b"/a.txt\0x",
    ];
    for path in refused {
        assert_eq!(
            ask(&mut folder, &size(path)),
            HostFailure::NotFound.reply(),
            "{:?}",
            String::from_utf8_lossy(path)
        );
    }
    let not_utf8 = [b'/', 0xFF];
    assert_eq!(
        ask(&mut folder, &size(&not_utf8)),
        HostFailure::NotFound.reply()
    );
}

#[cfg(unix)]
#[test]
fn a_link_out_of_the_folder_is_not_followed() {
    let (scratch, mut folder) = folder("link");
    scratch.file("../outside.txt", b"secret");
    let outside = scratch.path().join("../outside.txt");
    std::os::unix::fs::symlink(&outside, scratch.path().join("out")).unwrap();
    std::os::unix::fs::symlink(scratch.path().join("a.txt"), scratch.path().join("in")).unwrap();
    assert_eq!(
        ask(&mut folder, &size(b"/out")),
        HostFailure::NotFound.reply()
    );
    assert_eq!(
        ask(&mut folder, &read(b"/out", 0, 6)),
        HostFailure::NotFound.reply()
    );
    // A link that stays inside is followed.
    let len = u64::try_from(TEXT.len()).unwrap();
    let data = HostIoReply::Data(len.to_le_bytes().to_vec());
    assert_eq!(ask(&mut folder, &size(b"/in")), data);
}

#[test]
fn other_channels_and_bad_requests() {
    let (_scratch, mut folder) = folder("channels");
    let mut request = PROBE.to_host_io();
    request.channel = CHANNEL + 1;
    assert_eq!(folder.handle(&request), HostIoReply::Unavailable);
    let garbage = HostIoRequest {
        channel: CHANNEL,
        payload: vec![9; 4],
    };
    assert_eq!(folder.handle(&garbage), HostFailure::BadRequest.reply());
}
