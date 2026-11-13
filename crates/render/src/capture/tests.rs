//! The capture file: golden bytes, round trips, and every way a file is refused.

use super::{Capture, CaptureError, CapturedList, MAGIC, VERSION};
use crate::command::tests::{GOLDEN, every_command, every_command_from, hex};
use crate::command::{Command, CommandList, Marker};
use crate::id::FenceValue;
use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

/// The bytes a hex string stands for.
fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}

fn marker(submit: u64, draw: u64) -> CommandList {
    CommandList {
        commands: vec![Command::Marker(Marker { submit, draw })],
    }
}

fn captured(fence: u64, list: CommandList) -> CapturedList {
    CapturedList {
        fence: FenceValue::new(fence),
        list,
    }
}

/// A capture of two lists: every command and every variant inside one (the list whose own
/// bytes `command::tests::golden_bytes` pins), then a second list, so that the order of lists
/// and the place of each fence value show. The fence values take two and three bytes.
fn golden_capture() -> Capture {
    Capture {
        lists: vec![
            captured(300, every_command()),
            captured(70_000, marker(7, 11)),
        ],
    }
}

/// The file of [`golden_capture`], written out by hand from the layout in the module docs.
fn golden_file() -> Vec<u8> {
    let mut file = b"SQRC".to_vec();
    // The version.
    file.push(0x01);
    // A list follows; its fence value, 300, as a varint; the list.
    file.extend([0x01, 0xAC, 0x02]);
    file.extend(unhex(GOLDEN));
    // A list follows; its fence value, 70 000; one command, the tenth kind (a marker), with
    // its two numbers.
    file.extend([0x01, 0xF0, 0xA2, 0x04]);
    file.extend([0x01, 0x09, 0x07, 0x0B]);
    // The end.
    file.push(0x00);
    file
}

#[test]
fn golden_bytes() {
    let bytes = golden_capture().to_bytes().expect("a valid capture");
    assert_eq!(hex(&bytes), hex(&golden_file()));
    // The header is the shipping constants.
    assert_eq!(bytes[..4], MAGIC);
    assert_eq!(u32::from(bytes[4]), VERSION);
}

#[test]
fn golden_bytes_decode_to_the_capture() {
    assert_eq!(Capture::from_bytes(&golden_file()), Ok(golden_capture()));
}

/// Files written by hand, one for each thing the layout can hold: no list, an empty list, and
/// several lists.
#[test]
fn hand_built_files() {
    let cases: [(&[u8], Capture); 3] = [
        (b"SQRC\x01\x00", Capture::default()),
        (
            b"SQRC\x01\x01\x05\x00\x00",
            Capture {
                lists: vec![captured(5, CommandList::new())],
            },
        ),
        (
            // Fence values 1, 2 and 200 (a two-byte varint); a marker, an empty list, a marker.
            b"SQRC\x01\x01\x01\x01\x09\x03\x04\x01\x02\x00\x01\xC8\x01\x01\x09\x05\x06\x00",
            Capture {
                lists: vec![
                    captured(1, marker(3, 4)),
                    captured(2, CommandList::new()),
                    captured(200, marker(5, 6)),
                ],
            },
        ),
    ];
    for (file, capture) in cases {
        assert_eq!(
            Capture::from_bytes(file),
            Ok(capture.clone()),
            "{file:02x?}"
        );
        assert_eq!(capture.to_bytes().as_deref(), Ok(file), "{file:02x?}");
    }
}

#[test]
fn new_keeps_the_lists_in_order() {
    let capture = Capture::new([
        (FenceValue::new(3), marker(1, 2)),
        (FenceValue::new(9), every_command()),
    ]);
    assert_eq!(
        capture,
        Capture {
            lists: vec![captured(3, marker(1, 2)), captured(9, every_command())],
        }
    );
    assert_eq!(Capture::new([]), Capture::default());
}

proptest! {
    #![proptest_config(Config {
        rng_seed: RngSeed::Fixed(0x5ee_d164),
        cases: 64,
        failure_persistence: None,
        ..Config::default()
    })]

    /// A capture that is written and read back is the same capture, and writes the same bytes
    /// again. Each list has its own values and its own fence value, so a reader that took two
    /// fields in the wrong order, or lists in the wrong order, would not give it back.
    #[test]
    fn a_written_capture_reads_back(
        lists in proptest::collection::vec((1..5_000_u64, 0..9_000_u32, any::<bool>()), 0..5),
    ) {
        let mut fence = 0;
        let capture = Capture {
            lists: lists
                .into_iter()
                .map(|(gap, start, empty)| {
                    fence += gap;
                    let list = if empty { CommandList::new() } else { every_command_from(start) };
                    captured(fence, list)
                })
                .collect(),
        };
        let bytes = capture.to_bytes().expect("a valid capture");
        let read = Capture::from_bytes(&bytes).expect("the bytes just written");
        prop_assert_eq!(&read, &capture);
        prop_assert_eq!(read.to_bytes().expect("a valid capture"), bytes);
    }
}

/// Every file that stops short of its end is truncated, wherever it stops: inside the header,
/// inside a list, between two lists, or just before the end mark.
#[test]
fn every_truncated_file_is_refused() {
    let file = golden_file();
    assert!(Capture::from_bytes(&file).is_ok());
    for len in 0..file.len() {
        assert_eq!(
            Capture::from_bytes(&file[..len]),
            Err(CaptureError::Truncated),
            "{len} of {} bytes",
            file.len()
        );
    }
    // A command count that promises more than the file holds is a truncation too, and is not
    // believed: this returns at once.
    let mut promising = b"SQRC\x01\x01\x05".to_vec();
    promising.extend([0xFF; 9]);
    promising.push(0x01);
    assert_eq!(
        Capture::from_bytes(&promising),
        Err(CaptureError::Truncated)
    );
    assert_eq!(
        CaptureError::Truncated.to_string(),
        "the capture ends early"
    );
}

/// A file that does not start with the magic is not a capture, whatever else is wrong with it.
#[test]
fn another_kind_of_file_is_refused() {
    let file = golden_file();
    for at in 0..MAGIC.len() {
        let mut other = file.clone();
        other[at] ^= 0x20;
        assert_eq!(
            Capture::from_bytes(&other),
            Err(CaptureError::BadMagic),
            "{at}"
        );
    }
    // The magic is checked first: a wrong version and a missing body do not hide it.
    assert_eq!(
        Capture::from_bytes(b"SQRX\x02"),
        Err(CaptureError::BadMagic)
    );
    // A flight-recorder dump and a PNG.
    assert_eq!(
        Capture::from_bytes(b"SQFR\x01\x00"),
        Err(CaptureError::BadMagic)
    );
    assert_eq!(
        Capture::from_bytes(b"\x89PNG\r\n\x1a\n"),
        Err(CaptureError::BadMagic)
    );
    // A file too short to hold the magic is another kind of file too, unless it is the start
    // of the magic (`every_truncated_file_is_refused` has those).
    let short: [&[u8]; 9] = [
        b"P", b"\0", b"s", b"PN", b"SX", b"XQ", b"abc", b"SQX", b"SXR",
    ];
    for file in short {
        assert_eq!(
            Capture::from_bytes(file),
            Err(CaptureError::BadMagic),
            "{file:?}"
        );
    }
    assert_eq!(
        CaptureError::BadMagic.to_string(),
        "not a render capture (bad magic)"
    );
}

/// What reading accepts beyond what writing produces: `postcard` reads a number written with
/// more bytes than it needs, so more than one file holds the same capture, and reading such a
/// file and writing it again gives the shortest (module docs, "What reading checks").
#[test]
fn longer_numbers_are_read_and_written_short() {
    // The version, 1, in two bytes and in five (the most a 32-bit number takes).
    for file in [&b"SQRC\x81\x00\x00"[..], b"SQRC\x81\x80\x80\x80\x00\x00"] {
        let capture = Capture::from_bytes(file).expect("a capture with no list");
        assert_eq!(capture, Capture::default());
        assert_eq!(capture.to_bytes().as_deref(), Ok(&b"SQRC\x01\x00"[..]));
    }
    // A fence value, 5, and a command count, 0, in two bytes each.
    let capture = Capture::from_bytes(b"SQRC\x01\x01\x85\x00\x80\x00\x00").expect("one list");
    assert_eq!(
        capture,
        Capture {
            lists: vec![captured(5, CommandList::new())]
        }
    );
    assert_eq!(
        capture.to_bytes().as_deref(),
        Ok(&b"SQRC\x01\x01\x05\x00\x00"[..])
    );
    // A longer version is still the version it says: 2 in two bytes is refused as 2.
    assert_eq!(
        Capture::from_bytes(b"SQRC\x82\x00\x00"),
        Err(CaptureError::UnsupportedVersion { found: 2 })
    );
    // More bytes than a 32-bit number takes is not a version at all.
    assert_eq!(
        Capture::from_bytes(b"SQRC\x81\x80\x80\x80\x80\x00\x00"),
        Err(CaptureError::Decode(postcard::Error::DeserializeBadVarint))
    );
}

/// The memory a capture takes when read is bounded by the file's size, and the module docs
/// say by how much: a decoded command takes `size_of::<Command>()` bytes however few bytes
/// encode it. The shortest command is a marker of two small numbers, three bytes, and the
/// docs' "about seventy times" holds while a command is between 180 and 240 bytes.
#[test]
fn a_decoded_command_is_a_bounded_multiple_of_its_bytes() {
    let shortest = Command::Marker(Marker { submit: 0, draw: 0 });
    assert_eq!(
        postcard::to_allocvec(&shortest).as_deref(),
        Ok(&[0x09, 0x00, 0x00][..])
    );
    let decoded = size_of::<Command>();
    assert!(
        decoded <= 240,
        "{decoded} bytes: the docs' multiple is stale"
    );
    assert!(
        decoded >= 180,
        "{decoded} bytes: the docs' multiple is stale"
    );
}

/// A capture of another version is refused by its version, not misread: whatever follows the
/// header is not looked at.
#[test]
fn another_version_is_refused() {
    let with_version = |version: &[u8], body: &[u8]| [&MAGIC[..], version, body].concat();
    let body = &golden_file()[5..];
    // This version is read; the ones either side of it are not.
    assert!(Capture::from_bytes(&with_version(&[0x01], body)).is_ok());
    for (version, found) in [
        (&[0x00][..], VERSION - 1),
        (&[0x02], VERSION + 1),
        (&[0x7F], 127),
        (&[0x80, 0x01], 128),
        (&[0xFF, 0xFF, 0xFF, 0xFF, 0x0F], u32::MAX),
    ] {
        let refused = Err(CaptureError::UnsupportedVersion { found });
        // With a body this build could read, with none, and with one that is not a capture.
        assert_eq!(Capture::from_bytes(&with_version(version, body)), refused);
        assert_eq!(Capture::from_bytes(&with_version(version, &[])), refused);
        assert_eq!(
            Capture::from_bytes(&with_version(version, &[0xEE; 7])),
            refused
        );
    }
    assert_eq!(
        CaptureError::UnsupportedVersion { found: 2 }.to_string(),
        "capture layout version 2 is not supported (this build reads 1)"
    );
    // A version that is not a 32-bit number at all.
    assert_eq!(
        Capture::from_bytes(&with_version(&[0xFF, 0xFF, 0xFF, 0xFF, 0x1F], body)),
        Err(CaptureError::Decode(postcard::Error::DeserializeBadVarint))
    );
}

/// Bytes after the end mark are an error, even when they look like more of a capture.
#[test]
fn trailing_bytes_are_refused() {
    let file = golden_file();
    for extra in [&[0x00][..], &[0xEE, 0xEE, 0xEE], b"\x01\x05\x00\x00"] {
        let long = [&file[..], extra].concat();
        assert_eq!(
            Capture::from_bytes(&long),
            Err(CaptureError::TrailingBytes { count: extra.len() }),
            "{extra:02x?}"
        );
    }
    assert_eq!(
        CaptureError::TrailingBytes { count: 3 }.to_string(),
        "3 trailing bytes after the capture"
    );
}

/// Bytes that are all there but are not an encoding of a capture.
#[test]
fn malformed_files_are_refused() {
    // Neither "a list follows" nor "the end".
    assert_eq!(
        Capture::from_bytes(b"SQRC\x01\x02"),
        Err(CaptureError::Decode(postcard::Error::DeserializeBadOption))
    );
    // A command of a kind that does not exist (there are ten, numbered from zero).
    assert_eq!(
        Capture::from_bytes(b"SQRC\x01\x01\x05\x01\x0A\x00\x00\x00"),
        Err(CaptureError::Decode(postcard::Error::SerdeDeCustom))
    );
    // A fence value that does not fit in 64 bits.
    let mut wide = b"SQRC\x01\x01".to_vec();
    wide.extend([0xFF; 9]);
    wide.extend([0x7F, 0x00, 0x00]);
    assert_eq!(
        Capture::from_bytes(&wide),
        Err(CaptureError::Decode(postcard::Error::DeserializeBadVarint))
    );
    assert_eq!(
        CaptureError::Decode(postcard::Error::DeserializeBadOption).to_string(),
        "could not decode the capture: Found an Option discriminant that wasn't 0 or 1"
    );
}

/// A file with one empty list for each fence value given.
fn file_with_fences(fences: &[u8]) -> Vec<u8> {
    let mut file = b"SQRC\x01".to_vec();
    for fence in fences {
        file.extend([0x01, *fence, 0x00]);
    }
    file.push(0x00);
    file
}

/// Fence values rise from list to list and start above zero, as a backend's do. Both writing
/// and reading check it, at the edge on both sides.
#[test]
fn fence_values_must_rise() {
    let capture = |fences: &[u8]| Capture {
        lists: fences
            .iter()
            .map(|fence| captured((*fence).into(), CommandList::new()))
            .collect(),
    };
    for (fences, refused) in [
        (&[1][..], None),
        (&[1, 2], None),
        (&[1, 2, 3], None),
        (&[4, 9, 100], None),
        (&[0], Some((0, 0))),
        (&[5, 5], Some((1, 5))),
        (&[5, 4], Some((1, 4))),
        (&[5, 6, 6], Some((2, 6))),
        (&[5, 7, 6, 8], Some((2, 6))),
        (&[5, 0], Some((1, 0))),
        // Two faults: the first is reported.
        (&[0, 0], Some((0, 0))),
    ] {
        let expected = refused.map(|(list, fence)| CaptureError::FenceOrder { list, fence });
        let written = capture(fences).to_bytes();
        assert_eq!(written.as_ref().err(), expected.as_ref(), "{fences:?}");
        let file = file_with_fences(fences);
        assert_eq!(
            Capture::from_bytes(&file).err(),
            expected,
            "reading {fences:?}"
        );
        if expected.is_none() {
            assert_eq!(written, Ok(file), "{fences:?}");
        }
    }
    assert_eq!(
        CaptureError::FenceOrder { list: 2, fence: 6 }.to_string(),
        "list 2 of the capture has fence value 6, not above the one before it"
    );
}

/// The checks run in the order the module docs give. Each file here has two faults.
#[test]
fn the_first_fault_is_the_one_reported() {
    let bad_fences = file_with_fences(&[5, 5]);
    // Fence values that do not rise, and trailing bytes: the trailing bytes.
    let long = [&bad_fences[..], &[0xEE]].concat();
    assert_eq!(
        Capture::from_bytes(&long),
        Err(CaptureError::TrailingBytes { count: 1 })
    );
    // Fence values that do not rise, and no end mark: truncated.
    assert_eq!(
        Capture::from_bytes(&bad_fences[..bad_fences.len() - 1]),
        Err(CaptureError::Truncated)
    );
    // Fence values that do not rise, then something that is not a list: the decoding.
    let mut odd = bad_fences.clone();
    *odd.last_mut().expect("not empty") = 0x02;
    assert_eq!(
        Capture::from_bytes(&odd),
        Err(CaptureError::Decode(postcard::Error::DeserializeBadOption))
    );
    // Another version with any of these: the version.
    let mut other = long;
    other[4] = 0x02;
    assert_eq!(
        Capture::from_bytes(&other),
        Err(CaptureError::UnsupportedVersion { found: 2 })
    );
}
