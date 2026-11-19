//! The reader: the header's checks in order, each refusal as its exact error, the bounds, and a
//! generated round trip.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use super::*;
use crate::back::fence::Waiter;
use crate::capture::MAX_CAPTURE_MEMORY;
use crate::capture::format::Limits;
use crate::capture::format::one;
use crate::capture::{
    Access, BackRecord, Blobs, Capture, CaptureError, CapturedChannel, ContentHash, FrontRecord,
    Header, MAX_BUILD_ID, MAX_CAPTURE_BYTES, MAX_CHANNELS, MAX_SPACES, Output, StateRecord,
    VERSION,
};
use crate::capture::{EngineStateKind, ShaderAnswer};
use crate::interface::{
    ChannelConfig, ImageDesc, Kind, LayerId, RegionId, SYNCPOINT_COUNT, Ticket, VmUpdate,
};
use crate::memory::GpuFault;
use starquake_shader::lookup::ProgramRequest;

/// The bytes of `minimal()`'s outputs: the count, then one render record of a tag and a hash.
const OUTPUTS_LEN: usize = 4 + 1 + 16;

/// A capture with one space, one channel, the caps blob, a render output that names it (a
/// capture holds exactly one), and nothing else.
fn minimal() -> Capture {
    let mut blobs = Blobs::new();
    let caps = blobs.insert(&postcard::to_allocvec(&config().caps).unwrap());
    Capture {
        header: Header {
            build_id: "b".to_owned(),
            caps,
            memory_allowed: 1 << 20,
            setting: WaitWhileCompiling::Wait,
            mode: Mode::Fenced,
            spaces: vec![SPACE],
            channels: vec![CapturedChannel {
                id: ChannelId(0),
                config: ChannelConfig {
                    space: SPACE,
                    syncpoint: syncpoint(),
                },
            }],
        },
        state: vec![],
        front: vec![],
        back: vec![],
        blobs,
        outputs: vec![Output::Render { text: caps }],
    }
}

/// Where the header ends in `capture`'s file.
fn header_len(capture: &Capture) -> usize {
    one::header(&capture.header).unwrap().len()
}

#[test]
fn a_minimal_capture_reads_back() {
    let capture = minimal();
    assert_eq!(
        Capture::parse(&capture.to_bytes().unwrap()).unwrap(),
        capture
    );
}

#[test]
fn the_magic_is_checked_before_the_version() {
    let mut file = minimal().to_bytes().unwrap();
    file[0] = b'X';
    file[4..8].copy_from_slice(&(VERSION + 1).to_le_bytes());
    let error = Capture::parse(&file).unwrap_err();
    assert_eq!(
        error,
        CaptureError::BadMagic {
            found: crate::capture::format::Found(*b"XQGC")
        }
    );
    assert_eq!(
        error.to_string(),
        "the file is not a GPU capture: it starts with 58 51 47 43, not SQGC"
    );
}

#[test]
fn the_version_is_checked_before_the_body() {
    let mut file = minimal().to_bytes().unwrap();
    file[4..8].copy_from_slice(&(VERSION + 1).to_le_bytes());
    // A body that would not read either: its build id is past the limit.
    file[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    let error = Capture::parse(&file).unwrap_err();
    assert_eq!(
        error,
        CaptureError::UnsupportedVersion { found: VERSION + 1 }
    );
    assert_eq!(
        error.to_string(),
        "the capture is format version 2; this build reads version 1 only"
    );
    // Version 0 is refused too.
    file[4..8].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(
        Capture::parse(&file).unwrap_err(),
        CaptureError::UnsupportedVersion { found: 0 }
    );
}

#[test]
fn a_file_too_short_for_the_magic_or_the_version_is_truncated() {
    assert_eq!(
        Capture::parse(b"SQG").unwrap_err(),
        CaptureError::Truncated {
            offset: 3,
            what: "the magic"
        }
    );
    assert_eq!(
        Capture::parse(b"").unwrap_err().to_string(),
        "the capture ends at byte 0, inside the magic"
    );
    assert_eq!(
        Capture::parse(b"SQGC\x01\x00").unwrap_err(),
        CaptureError::Truncated {
            offset: 6,
            what: "the version"
        }
    );
}

#[test]
fn every_cut_of_a_capture_is_an_error_and_one_byte_short_is_truncated() {
    let mut builder = builder();
    let channel = channel(&mut builder);
    let submission = submission_at(&builder, CMD, &releases(SEM, 3));
    builder.submit(channel, submission).unwrap();
    builder.present(present_request()).unwrap();
    builder.run().unwrap();
    let file = builder.finish().unwrap().to_bytes().unwrap();
    for len in 0..file.len() {
        assert!(Capture::parse(&file[..len]).is_err(), "cut at {len}");
    }
    let short = file.len() - 1;
    assert!(matches!(
        Capture::parse(&file[..short]).unwrap_err(),
        CaptureError::Truncated { offset, .. } if offset == short
    ));
}

#[test]
fn a_byte_after_the_last_section_is_refused() {
    let mut file = minimal().to_bytes().unwrap();
    let end = file.len();
    file.push(0);
    let error = Capture::parse(&file).unwrap_err();
    assert_eq!(
        error,
        CaptureError::TrailingBytes {
            offset: end,
            count: 1
        }
    );
    assert_eq!(
        error.to_string(),
        format!("the capture has 1 bytes after its last section, from byte {end}")
    );
}

#[test]
fn a_capture_past_the_limit_is_refused_before_it_is_read() {
    // Zeroed and never touched past the first bytes, so the allocation costs no memory.
    let at_limit = vec![0u8; MAX_CAPTURE_BYTES];
    // At the limit, the reader reads: its magic is wrong.
    assert_eq!(
        Capture::parse(&at_limit).unwrap_err(),
        CaptureError::BadMagic {
            found: crate::capture::format::Found([0; 4])
        }
    );
    drop(at_limit);
    let past = vec![0u8; MAX_CAPTURE_BYTES + 1];
    assert_eq!(
        Capture::parse(&past).unwrap_err(),
        CaptureError::TooLarge {
            len: MAX_CAPTURE_BYTES + 1,
            limit: MAX_CAPTURE_BYTES,
        }
    );
}

#[test]
fn a_count_past_the_bytes_is_truncated_without_allocating_for_it() {
    let capture = minimal();
    let mut file = capture.to_bytes().unwrap();
    let state = header_len(&capture);
    file.truncate(state);
    file.extend_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        Capture::parse(&file).unwrap_err(),
        CaptureError::Truncated {
            offset: file.len(),
            what: "starting state"
        }
    );
}

#[test]
fn the_build_id_is_bounded_at_its_limit() {
    let mut capture = minimal();
    capture.header.build_id = "x".repeat(MAX_BUILD_ID);
    let file = capture.to_bytes().unwrap();
    assert_eq!(Capture::parse(&file).unwrap(), capture);
    capture.header.build_id.push('x');
    assert_eq!(
        capture.to_bytes().unwrap_err(),
        CaptureError::BuildIdTooLong {
            len: MAX_BUILD_ID + 1
        }
    );
    // A file that says one more byte is refused before it is read.
    let mut file = file;
    let past = u32::try_from(MAX_BUILD_ID + 1).unwrap();
    file[8..12].copy_from_slice(&past.to_le_bytes());
    let error = Capture::parse(&file).unwrap_err();
    assert_eq!(
        error,
        CaptureError::BuildIdTooLong {
            len: MAX_BUILD_ID + 1
        }
    );
    assert_eq!(
        error.to_string(),
        "the build id is 257 bytes; at most 256 are allowed"
    );
}

#[test]
fn a_build_id_that_is_not_utf8_is_refused() {
    let mut file = minimal().to_bytes().unwrap();
    // The one-byte build id "b" at byte 12.
    file[12] = 0xff;
    assert_eq!(
        Capture::parse(&file).unwrap_err(),
        CaptureError::BadValue {
            offset: 8,
            what: "the build id is not UTF-8"
        }
    );
}

/// The header's file offsets of `minimal()`: setting, mode, then the lists.
const SETTING_AT: usize = 4 + 4 + 4 + 1 + 16 + 8;

#[test]
fn an_unknown_setting_or_mode_is_refused() {
    let mut file = minimal().to_bytes().unwrap();
    file[SETTING_AT] = 2;
    assert_eq!(
        Capture::parse(&file).unwrap_err(),
        CaptureError::BadValue {
            offset: SETTING_AT,
            what: "the wait-while-compiling setting is unknown"
        }
    );
    let mut file = minimal().to_bytes().unwrap();
    file[SETTING_AT + 1] = 2;
    assert_eq!(
        Capture::parse(&file).unwrap_err(),
        CaptureError::BadValue {
            offset: SETTING_AT + 1,
            what: "the fence mode is unknown"
        }
    );
}

/// `minimal()`'s file with its header replaced by `header`'s bytes, written without the
/// writer's checks.
fn with_header(header: &Header) -> Vec<u8> {
    let capture = minimal();
    let file = capture.to_bytes().unwrap();
    let mut out = one::header(header).unwrap();
    out.extend_from_slice(&file[header_len(&capture)..]);
    out
}

/// Each header list check, read and written: the reader refuses the file, and the writer
/// refuses the capture with the same error.
#[test]
fn the_header_lists_are_checked() {
    let check = |capture: &Capture, expected: CaptureError, text: &str| {
        assert_eq!(
            Capture::parse(&with_header(&capture.header)).unwrap_err(),
            expected
        );
        assert_eq!(capture.to_bytes().unwrap_err(), expected);
        assert_eq!(expected.to_string(), text);
    };
    let mut capture = minimal();
    capture.header.spaces.push(SPACE);
    check(
        &capture,
        CaptureError::DuplicateSpace { space: SPACE },
        "address space 33 is listed twice",
    );

    let mut capture = minimal();
    let first = capture.header.channels[0];
    capture.header.channels.push(first);
    check(
        &capture,
        CaptureError::DuplicateChannel {
            channel: ChannelId(0),
        },
        "channel 0 is listed twice",
    );

    let mut capture = minimal();
    capture.header.channels[0].config.space = AsId(0x99);
    check(
        &capture,
        CaptureError::UnknownSpace {
            channel: ChannelId(0),
            space: AsId(0x99),
        },
        "channel 0 is in address space 153, which the header does not list",
    );

    // A second channel that repeats the first's id and is in an unlisted space: the repeat is
    // reported before the space.
    let mut capture = minimal();
    let mut second = capture.header.channels[0];
    second.config.space = AsId(0x99);
    capture.header.channels.push(second);
    check(
        &capture,
        CaptureError::DuplicateChannel {
            channel: ChannelId(0),
        },
        "channel 0 is listed twice",
    );
}

/// A header of `spaces` address spaces and `channels` channels, each channel in the first space.
fn header_of(spaces: usize, channels: usize) -> Header {
    let mut header = minimal().header;
    header.spaces = (0..spaces)
        .map(|n| AsId(u32::try_from(n).unwrap()))
        .collect();
    header.channels = (0..channels)
        .map(|n| CapturedChannel {
            id: ChannelId(u32::try_from(n).unwrap()),
            config: ChannelConfig {
                space: AsId(0),
                syncpoint: syncpoint(),
            },
        })
        .collect();
    header
}

/// The header's lists are capped, read and written: at the cap a list is accepted, one past it
/// is refused from its count, before any item is read.
#[test]
fn the_header_lists_are_capped_at_their_limits() {
    let mut capture = minimal();
    capture.header = header_of(MAX_SPACES, MAX_CHANNELS);
    let file = capture.to_bytes().unwrap();
    assert_eq!(Capture::parse(&file).unwrap(), capture);

    let mut capture = minimal();
    capture.header = header_of(MAX_SPACES + 1, 1);
    let expected = CaptureError::TooManySpaces {
        count: MAX_SPACES + 1,
    };
    assert_eq!(capture.to_bytes().unwrap_err(), expected);
    assert_eq!(
        Capture::parse(&with_header(&capture.header)).unwrap_err(),
        expected
    );
    assert_eq!(
        expected.to_string(),
        "the header lists 1025 address spaces; at most 1024 are allowed"
    );

    let mut capture = minimal();
    capture.header = header_of(1, MAX_CHANNELS + 1);
    let expected = CaptureError::TooManyChannels {
        count: MAX_CHANNELS + 1,
    };
    assert_eq!(capture.to_bytes().unwrap_err(), expected);
    assert_eq!(
        Capture::parse(&with_header(&capture.header)).unwrap_err(),
        expected
    );
    assert_eq!(
        expected.to_string(),
        "the header lists 4097 channels; at most 4096 are allowed"
    );

    // A count past the cap is refused before its items are read: here there are none.
    let mut file = one::header(&minimal().header).unwrap();
    let spaces_at = SETTING_AT + 2;
    file.truncate(spaces_at);
    file.extend_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        Capture::parse(&file).unwrap_err(),
        CaptureError::TooManySpaces {
            count: usize::try_from(u32::MAX).unwrap()
        }
    );
}

/// A header of many distinct spaces reads in time linear-ish in its length: the checks are set
/// lookups, not a scan of the list so far. At the cap the lists are read quickly; a hostile
/// count past the cap is refused at once, whatever the file holds after it.
#[test]
fn a_large_header_list_reads_quickly() {
    let mut capture = minimal();
    capture.header = header_of(MAX_SPACES, MAX_CHANNELS);
    let file = capture.to_bytes().unwrap();
    let started = std::time::Instant::now();
    for _ in 0..16 {
        Capture::parse(&file).unwrap();
    }
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "16 reads of a header at the caps took {:?}",
        started.elapsed()
    );
    // 160 000 distinct spaces: refused from the count, without reading them.
    let mut big = one::header(&header_of(1, 0)).unwrap();
    big.truncate(SETTING_AT + 2);
    big.extend_from_slice(&160_000u32.to_le_bytes());
    for space in 0..160_000u32 {
        big.extend_from_slice(&space.to_le_bytes());
    }
    let started = std::time::Instant::now();
    assert_eq!(
        Capture::parse(&big).unwrap_err(),
        CaptureError::TooManySpaces { count: 160_000 }
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(1));
}

#[test]
fn a_syncpoint_is_below_the_count() {
    let capture = minimal();
    let mut file = capture.to_bytes().unwrap();
    let at = header_len(&capture) - 1;
    let last = u8::try_from(SYNCPOINT_COUNT - 1).unwrap();
    file[at] = last;
    assert_eq!(
        Capture::parse(&file).unwrap().header.channels[0]
            .config
            .syncpoint
            .get(),
        last
    );
    file[at] = last + 1;
    assert_eq!(
        Capture::parse(&file).unwrap_err(),
        CaptureError::BadValue {
            offset: at,
            what: "a syncpoint number is past the last syncpoint"
        }
    );
}

/// `minimal()` with one front record whose bytes are `record`, as a file.
fn with_front_bytes(record: &[u8]) -> Vec<u8> {
    let capture = minimal();
    let mut file = capture.to_bytes().unwrap();
    let front_count = header_len(&capture) + 4;
    file[front_count..front_count + 4].copy_from_slice(&1u32.to_le_bytes());
    let at = front_count + 4;
    let mut out = file[..at].to_vec();
    out.extend_from_slice(record);
    out.extend_from_slice(&file[at..]);
    out
}

#[test]
fn a_record_in_the_wrong_section_is_an_unknown_tag() {
    let capture = minimal();
    let file = with_front_bytes(&[0x20]);
    let at = header_len(&capture) + 8;
    let error = Capture::parse(&file).unwrap_err();
    assert_eq!(
        error,
        CaptureError::UnknownTag {
            offset: at,
            tag: 0x20,
            section: "front log"
        }
    );
    assert_eq!(
        error.to_string(),
        format!("byte {at}: 0x20 is not a record of the front log")
    );
}

#[test]
fn each_bad_field_value_is_refused_where_it_is() {
    let base = header_len(&minimal()) + 8;
    let cases: [(Vec<u8>, usize, &str); 8] = [
        // A report's released flag of 2.
        (vec![0x16, 2], 1, "a flag is neither 0 nor 1"),
        // A syncpoint record past the count.
        (
            vec![0x17, 192, 0, 0, 0, 0],
            1,
            "a syncpoint number is past the last syncpoint",
        ),
        // An unmap of half a page.
        (
            [
                vec![0x12, 1, 0, 0, 0, 0],
                0x1000u64.to_le_bytes().to_vec(),
                0x800u64.to_le_bytes().to_vec(),
            ]
            .concat(),
            6,
            "a page range is not whole pages",
        ),
        // An update of kind 4.
        (vec![0x12, 1, 0, 0, 0, 4], 5, "an update's kind is unknown"),
        // A submission whose wait flag is 2.
        (
            vec![0x11, 1, 0, 0, 0, 2],
            5,
            "a presence flag is neither 0 nor 1",
        ),
        // A read whose outcome is 2.
        (
            [vec![0x30, 1, 0, 0, 0], vec![0; 16], vec![2]].concat(),
            21,
            "an access's outcome is neither 0 nor 1",
        ),
        // A read fault of kind 4.
        (
            [vec![0x30, 1, 0, 0, 0], vec![0; 16], vec![1, 4]].concat(),
            22,
            "a fault's kind is not one this build knows",
        ),
        // A GPU-only answer of 2.
        (
            [vec![0x32, 1, 0, 0, 0], vec![0; 16], vec![2]].concat(),
            21,
            "a flag is neither 0 nor 1",
        ),
    ];
    for (record, offset, what) in cases {
        assert_eq!(
            Capture::parse(&with_front_bytes(&record)).unwrap_err(),
            CaptureError::BadValue {
                offset: base + offset,
                what
            },
            "{record:02x?}"
        );
    }
}

/// `minimal()` with one back record whose bytes are `record`, as a file.
fn with_back_bytes(record: &[u8]) -> Vec<u8> {
    let capture = minimal();
    let mut file = capture.to_bytes().unwrap();
    let back_count = header_len(&capture) + 8;
    file[back_count..back_count + 4].copy_from_slice(&1u32.to_le_bytes());
    let at = back_count + 4;
    let mut out = file[..at].to_vec();
    out.extend_from_slice(record);
    out.extend_from_slice(&file[at..]);
    out
}

#[test]
fn each_bad_back_field_is_refused_where_it_is() {
    let base = header_len(&minimal()) + 12;
    let cases: [(Vec<u8>, usize, &str); 3] = [
        // A shader request of stage 7.
        // (The cache's accesses, 2, then one request.)
        (
            vec![0x27, 2, 0, 0, 0, 1, 0, 0, 0, 7],
            9,
            "a shader stage is unknown",
        ),
        // A wait answer of kind 3.
        (
            [vec![0x28], vec![0; 8], vec![3]].concat(),
            9,
            "a shader answer's kind is not one this build knows",
        ),
        // A waiter whose operation is 5.
        (
            [vec![0x25, 1, 0, 0, 0], vec![0; 16], vec![5]].concat(),
            21,
            "an acquire operation is unknown",
        ),
    ];
    for (record, offset, what) in cases {
        assert_eq!(
            Capture::parse(&with_back_bytes(&record)).unwrap_err(),
            CaptureError::BadValue {
                offset: base + offset,
                what
            },
            "{record:02x?}"
        );
    }
}

#[test]
fn an_engine_state_of_an_unknown_kind_is_refused() {
    let capture = minimal();
    let mut file = capture.to_bytes().unwrap();
    let count = header_len(&capture);
    file[count..count + 4].copy_from_slice(&1u32.to_le_bytes());
    let at = count + 4;
    let record = [vec![0x03, 8], vec![0; 4 + 16]].concat();
    let mut out = file[..at].to_vec();
    out.extend_from_slice(&record);
    out.extend_from_slice(&file[at..]);
    assert_eq!(
        Capture::parse(&out).unwrap_err(),
        CaptureError::BadValue {
            offset: at + 1,
            what: "an engine state's kind is unknown"
        }
    );
}

/// The file offset of `minimal()`'s blob table, and its one blob's end.
fn blob_table(capture: &Capture) -> usize {
    header_len(capture) + 12
}

#[test]
fn the_blob_table_is_in_order_hashes_match_and_every_named_blob_is_there() {
    // Two blobs, swapped in the file.
    let mut capture = minimal();
    capture.blobs.insert(b"second blob");
    let file = capture.to_bytes().unwrap();
    let table = blob_table(&capture);
    let first_len = 16
        + 8
        + usize::try_from(u64::from_le_bytes(
            file[table + 4 + 16..table + 4 + 24].try_into().unwrap(),
        ))
        .unwrap();
    let rest_start = table + 4;
    let first = file[rest_start..rest_start + first_len].to_vec();
    let second_end = file.len() - OUTPUTS_LEN;
    let second = file[rest_start + first_len..second_end].to_vec();
    let mut swapped = file[..rest_start].to_vec();
    swapped.extend_from_slice(&second);
    swapped.extend_from_slice(&first);
    swapped.extend_from_slice(&file[second_end..]);
    assert_eq!(
        Capture::parse(&swapped).unwrap_err(),
        CaptureError::BlobOrder {
            offset: rest_start + second.len()
        }
    );
    // The same blob twice.
    let mut twice = file[..rest_start].to_vec();
    twice[table..table + 4].copy_from_slice(&3u32.to_le_bytes());
    twice.extend_from_slice(&first);
    twice.extend_from_slice(&first);
    twice.extend_from_slice(&second);
    twice.extend_from_slice(&file[second_end..]);
    assert_eq!(
        Capture::parse(&twice).unwrap_err(),
        CaptureError::BlobOrder {
            offset: rest_start + first.len()
        }
    );
    // A blob whose last byte changed: the file's last blob, the one whose hash is larger.
    let mut corrupt = file.clone();
    corrupt[second_end - 1] ^= 1;
    let hash = ContentHash::of(b"second blob").max(capture.header.caps);
    assert_eq!(
        Capture::parse(&corrupt).unwrap_err(),
        CaptureError::CorruptBlob { hash }
    );
    assert_eq!(
        CaptureError::CorruptBlob { hash }.to_string(),
        format!("blob {hash} does not hash to its name")
    );
}

#[test]
fn a_named_blob_must_exist_and_a_read_must_be_its_length() {
    let mut capture = minimal();
    let missing = ContentHash::of(b"not stored");
    capture.front.push(FrontRecord::Access(Access::Write {
        space: SPACE,
        va: 0x1000,
        data: missing,
        result: Ok(()),
    }));
    assert_eq!(
        capture.to_bytes().unwrap_err(),
        CaptureError::MissingBlob { hash: missing }
    );
    let mut capture = minimal();
    let stored = capture.blobs.insert(b"four");
    capture.back.push(BackRecord::Access(Access::Read {
        space: SPACE,
        va: 0x1000,
        len: 5,
        result: Ok(stored),
    }));
    let error = capture.to_bytes().unwrap_err();
    assert_eq!(
        error,
        CaptureError::ReadLength {
            len: 5,
            hash: stored,
            found: 4
        }
    );
    assert_eq!(
        error.to_string(),
        format!("a read of 5 bytes names blob {stored}, which holds 4 bytes")
    );
    // A file whose record names a blob it lacks: the caps blob, renamed.
    let capture = minimal();
    let mut file = capture.to_bytes().unwrap();
    let caps_at = 4 + 4 + 4 + 1;
    file[caps_at] ^= 1;
    let mut renamed = capture.header.caps.value().to_le_bytes();
    renamed[0] ^= 1;
    assert_eq!(
        Capture::parse(&file).unwrap_err(),
        CaptureError::MissingBlob {
            hash: ContentHash::new(u128::from_le_bytes(renamed))
        }
    );
}

#[test]
fn an_unknown_output_is_refused() {
    let mut file = minimal().to_bytes().unwrap();
    let count = file.len() - OUTPUTS_LEN;
    file.truncate(count);
    file.extend_from_slice(&1u32.to_le_bytes());
    file.push(0x42);
    assert_eq!(
        Capture::parse(&file).unwrap_err(),
        CaptureError::UnknownTag {
            offset: count + 4,
            tag: 0x42,
            section: "outputs"
        }
    );
}

/// A capture with exactly `count` render outputs, each naming the caps blob.
fn with_renders(count: usize) -> Capture {
    let mut capture = minimal();
    let text = capture.header.caps;
    capture.outputs = vec![Output::Render { text }; count];
    capture
}

/// `capture`'s file, written without the writer's body checks: the header and every section as
/// the writer lays them out.
fn unchecked_bytes(capture: &Capture) -> Vec<u8> {
    let mut checked = capture.clone();
    // A capture the writer accepts with the same sections up to the outputs and waiters...
    checked.outputs = vec![Output::Render {
        text: capture.header.caps,
    }];
    checked
        .back
        .retain(|record| !matches!(record, BackRecord::Waiters { .. }));
    let mut file = one::header(&capture.header).unwrap();
    file.extend_from_slice(&u32::try_from(capture.state.len()).unwrap().to_le_bytes());
    for record in &capture.state {
        file.extend(one::state(record));
    }
    file.extend_from_slice(&u32::try_from(capture.front.len()).unwrap().to_le_bytes());
    for record in &capture.front {
        file.extend(one::front(record).unwrap());
    }
    file.extend_from_slice(&u32::try_from(capture.back.len()).unwrap().to_le_bytes());
    for record in &capture.back {
        file.extend(one::back(record).unwrap());
    }
    // ...whose blob table is the same, copied from its file.
    let checked_file = checked.to_bytes().unwrap();
    let table = blob_table_at(&checked);
    let outputs = checked_file.len() - OUTPUTS_LEN;
    file.extend_from_slice(&checked_file[table..outputs]);
    file.extend_from_slice(&u32::try_from(capture.outputs.len()).unwrap().to_le_bytes());
    for output in &capture.outputs {
        file.extend(one::output(output));
    }
    file
}

/// Where `capture`'s blob table starts in its file.
fn blob_table_at(capture: &Capture) -> usize {
    let mut at = header_len(capture) + 4;
    at += capture
        .state
        .iter()
        .map(|r| one::state(r).len())
        .sum::<usize>()
        + 4;
    at += capture
        .front
        .iter()
        .map(|r| one::front(r).unwrap().len())
        .sum::<usize>()
        + 4;
    at += capture
        .back
        .iter()
        .map(|r| one::back(r).unwrap().len())
        .sum::<usize>();
    at
}

/// A capture holds exactly one render stream: none and two are refused, read and written, so a
/// whole replay always has a stream to compare.
#[test]
fn a_capture_holds_exactly_one_render_stream() {
    let one_stream = with_renders(1);
    assert_eq!(
        Capture::parse(&one_stream.to_bytes().unwrap()).unwrap(),
        one_stream
    );
    for count in [0, 2] {
        let capture = with_renders(count);
        let expected = CaptureError::RenderOutputs { count };
        assert_eq!(capture.to_bytes().unwrap_err(), expected);
        assert_eq!(
            Capture::parse(&unchecked_bytes(&capture)).unwrap_err(),
            expected
        );
    }
    assert_eq!(
        CaptureError::RenderOutputs { count: 0 }.to_string(),
        "the capture's outputs hold 0 render command streams; a capture holds exactly one"
    );
}

/// `minimal()` with a second channel, 1, in a second space, and one `waiters` record of
/// `waiters` in its back log.
fn with_waiters(waiters: &[(u32, AsId)]) -> Capture {
    let mut capture = minimal();
    capture.header.spaces.push(AsId(0x22));
    capture.header.channels.push(CapturedChannel {
        id: ChannelId(1),
        config: ChannelConfig {
            space: AsId(0x22),
            syncpoint: syncpoint(),
        },
    });
    capture.back.push(BackRecord::Apply);
    capture.back.push(BackRecord::Waiters {
        waiters: waiters
            .iter()
            .map(|&(channel, space)| {
                (
                    Waiter {
                        channel: ChannelId(channel),
                        space,
                        address: 0x1000,
                        op: AcquireOp::Equal,
                        payload: 1,
                    },
                    false,
                )
            })
            .collect(),
    });
    capture
}

/// Each `waiters` check, read and written, with its exact error: a channel twice, a channel the
/// header does not list, a waiter in another space than its channel's. One per channel of the
/// header, each in its channel's space, is accepted.
#[test]
fn a_waiter_table_lists_each_header_channel_once_in_its_space() {
    let good = with_waiters(&[(1, AsId(0x22)), (0, SPACE)]);
    assert_eq!(Capture::parse(&good.to_bytes().unwrap()).unwrap(), good);
    let cases = [
        (
            with_waiters(&[(0, SPACE), (1, AsId(0x22)), (0, SPACE)]),
            CaptureError::WaiterRepeated {
                record: 1,
                channel: ChannelId(0),
            },
            "back log record 1 lists channel 0 in the waiter table twice",
        ),
        (
            with_waiters(&[(0, SPACE), (2, SPACE)]),
            CaptureError::WaiterChannel {
                record: 1,
                channel: ChannelId(2),
            },
            "back log record 1 lists a waiter of channel 2, which the header does not list",
        ),
        (
            with_waiters(&[(1, SPACE)]),
            CaptureError::WaiterSpace {
                record: 1,
                channel: ChannelId(1),
                space: SPACE,
                expected: AsId(0x22),
            },
            "back log record 1 puts channel 1's waiter in address space 33; the channel is in \
             address space 34",
        ),
    ];
    for (capture, expected, text) in cases {
        assert_eq!(capture.to_bytes().unwrap_err(), expected);
        assert_eq!(
            Capture::parse(&unchecked_bytes(&capture)).unwrap_err(),
            expected
        );
        assert_eq!(expected.to_string(), text);
    }
    // A repeat whose second entry is in the wrong space too: the repeat is reported.
    let capture = with_waiters(&[(1, AsId(0x22)), (1, SPACE)]);
    assert_eq!(
        capture.to_bytes().unwrap_err(),
        CaptureError::WaiterRepeated {
            record: 1,
            channel: ChannelId(1),
        }
    );
}

/// The checks after the file is read run in their documented order: the waiters, the render
/// stream, the named blobs. Each pair has both faults present.
#[test]
fn the_last_checks_run_in_order() {
    let bad_waiters = |capture: &mut Capture| {
        capture.back.push(BackRecord::Waiters {
            waiters: vec![(
                Waiter {
                    channel: ChannelId(9),
                    space: SPACE,
                    address: 0,
                    op: AcquireOp::Equal,
                    payload: 0,
                },
                false,
            )],
        });
    };
    let no_render = |capture: &mut Capture| capture.outputs.clear();
    let missing = ContentHash::of(b"not stored");
    let missing_blob = |capture: &mut Capture| {
        capture.front.push(FrontRecord::Access(Access::Write {
            space: SPACE,
            va: 0,
            data: missing,
            result: Ok(()),
        }));
    };
    let mut capture = minimal();
    bad_waiters(&mut capture);
    no_render(&mut capture);
    let waiters_first = CaptureError::WaiterChannel {
        record: 0,
        channel: ChannelId(9),
    };
    assert_eq!(capture.to_bytes().unwrap_err(), waiters_first);
    let mut capture = minimal();
    no_render(&mut capture);
    missing_blob(&mut capture);
    let render_first = CaptureError::RenderOutputs { count: 0 };
    assert_eq!(capture.to_bytes().unwrap_err(), render_first);
    let mut capture = minimal();
    missing_blob(&mut capture);
    assert_eq!(
        capture.to_bytes().unwrap_err(),
        CaptureError::MissingBlob { hash: missing }
    );
    // The header's checks come before the body's, written as read.
    let mut capture = minimal();
    capture.header.spaces.push(SPACE);
    no_render(&mut capture);
    assert_eq!(
        capture.to_bytes().unwrap_err(),
        CaptureError::DuplicateSpace { space: SPACE }
    );
}

/// A record or list item takes at least as many bytes in memory as in the file, so a file
/// longer than the memory budget cannot fit it ([`MAX_CAPTURE_BYTES`] is the budget). The file
/// sizes are written out from the module docs' layout: the smallest record of each kind.
#[test]
fn each_item_takes_at_least_its_file_bytes_in_memory() {
    use std::mem::size_of;
    // The largest record of each list item's file layout (module docs): a front or back record
    // is at most a read with a blob (1 + 4 + 8 + 8 + 1 + 16) among the records that hold no
    // list (a present is 1 + 8 + 20 + 16 + 16), each counted with its list, below.
    let cases: [(&str, usize, usize); 12] = [
        (
            "front record",
            size_of::<FrontRecord>(),
            1 + 8 + 20 + 16 + 16,
        ),
        (
            "back record",
            size_of::<BackRecord>(),
            1 + 4 + 8 + 8 + 1 + 16,
        ),
        ("state record", size_of::<StateRecord>(), 1 + 1 + 4 + 16),
        ("output", size_of::<Output>(), 1 + 16),
        ("waiter", size_of::<(Waiter, bool)>(), 4 + 4 + 8 + 1 + 4 + 1),
        ("shader request", size_of::<ProgramRequest>(), 1 + 8),
        ("page range", size_of::<PageRange>(), 8 + 8),
        ("GP entry", size_of::<GpfifoEntry>(), 8),
        ("marker", size_of::<crate::ring::MarkerToken>(), 8),
        ("address space", size_of::<AsId>(), 4),
        ("channel", size_of::<CapturedChannel>(), 4 + 4 + 1),
        // The magic, version, fixed header fields and the six counts.
        (
            "capture",
            size_of::<Capture>(),
            4 + 4 + 4 + 16 + 8 + 1 + 1 + 4 + 4 + 5 * 4,
        ),
    ];
    for (what, memory, file) in cases {
        assert!(
            memory >= file,
            "a {what} takes {memory} bytes in memory, {file} in the file"
        );
    }
    assert_eq!(MAX_CAPTURE_BYTES, MAX_CAPTURE_MEMORY);
}

/// The memory budget's edge: a capture whose charge is exactly the budget reads and writes, one
/// byte less refuses both, with the exact error.
#[test]
fn the_memory_budget_is_charged_to_the_byte() {
    let capture = Capture::parse(&with_waiters(&[(0, SPACE)]).to_bytes().unwrap()).unwrap();
    let file = capture.to_bytes().unwrap();
    let (_, charged) = Capture::parse_within(&file, Limits::DEFAULT).unwrap();
    assert_eq!(charged, capture.memory_charge());
    let at = Limits {
        bytes: MAX_CAPTURE_BYTES,
        memory: charged,
    };
    let under = Limits {
        memory: charged - 1,
        ..at
    };
    assert_eq!(Capture::parse_within(&file, at).unwrap().0, capture);
    assert_eq!(capture.to_bytes_within(at).unwrap(), file);
    let expected = CaptureError::OverBudget { limit: charged - 1 };
    assert_eq!(Capture::parse_within(&file, under).unwrap_err(), expected);
    assert_eq!(capture.to_bytes_within(under).unwrap_err(), expected);
    assert_eq!(
        expected.to_string(),
        format!(
            "the capture needs more than {} bytes of memory once read",
            charged - 1
        )
    );
}

/// One-byte records cost their in-memory size, not their file size: a file of turn records is
/// refused once their list passes the budget, long before its bytes would.
#[test]
fn a_file_of_one_byte_records_is_refused_by_its_memory() {
    let capture = minimal();
    let file = capture.to_bytes().unwrap();
    let front_count = header_len(&capture) + 4;
    let turns: usize = 1 << 20;
    let mut hostile = file[..front_count].to_vec();
    hostile.extend_from_slice(&u32::try_from(turns).unwrap().to_le_bytes());
    hostile.extend(std::iter::repeat_n(0x10u8, turns));
    hostile.extend_from_slice(&file[front_count + 4..]);
    // The turns' list alone needs `turns` records of memory; the budget is a sixteenth of it.
    let budget = turns * size_of::<FrontRecord>() / 16;
    assert!(hostile.len() < budget, "the file fits the byte bound");
    assert_eq!(
        Capture::parse_within(
            &hostile,
            Limits {
                bytes: MAX_CAPTURE_BYTES,
                memory: budget,
            }
        )
        .unwrap_err(),
        CaptureError::OverBudget { limit: budget }
    );
    // Under the default budget it reads, charged its in-memory size.
    let (read, charged) = Capture::parse_within(&hostile, Limits::DEFAULT).unwrap();
    assert_eq!(read.front.len(), turns);
    assert!(charged >= turns * size_of::<FrontRecord>());
}

/// The file's byte bound, written: a file exactly at the bound is written, one byte over is
/// refused with the exact error, as the reader refuses it.
#[test]
fn the_writer_refuses_a_file_past_the_byte_bound() {
    let capture = minimal();
    let file = capture.to_bytes().unwrap();
    let at = Limits {
        bytes: file.len(),
        memory: MAX_CAPTURE_MEMORY,
    };
    assert_eq!(capture.to_bytes_within(at).unwrap(), file);
    let under = Limits {
        bytes: file.len() - 1,
        ..at
    };
    let expected = CaptureError::TooLarge {
        len: file.len(),
        limit: file.len() - 1,
    };
    assert_eq!(capture.to_bytes_within(under).unwrap_err(), expected);
    assert_eq!(Capture::parse_within(&file, under).unwrap_err(), expected);
    assert_eq!(Capture::parse_within(&file, at).unwrap().0, capture);
}

fn proptest_config() -> Config {
    Config {
        cases: 256,
        rng_seed: RngSeed::Fixed(0x5351_4743),
        failure_persistence: None,
        ..Config::default()
    }
}

/// A generated record and the blobs it names, which the capture's table must hold.
type WithBlobs<T> = (T, Vec<Vec<u8>>);

fn bytes_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..8)
}

fn page_range() -> impl Strategy<Value = PageRange> {
    (0u64..1 << 20, 1u64..64)
        .prop_map(|(start, pages)| PageRange::new(start * PAGE_SIZE, pages * PAGE_SIZE).unwrap())
}

fn fault() -> impl Strategy<Value = GpuFault> {
    prop_oneof![
        any::<u64>().prop_map(|va| GpuFault::Unmapped { va }),
        (any::<u64>(), any::<u64>()).prop_map(|(va, len)| GpuFault::OutOfRange { va, len }),
        (any::<u64>(), any::<u64>()).prop_map(|(va, address)| GpuFault::Guest {
            va,
            fault: starquake_memory::Fault { address },
        }),
        any::<u64>().prop_map(|va| GpuFault::ShimWrite { va }),
    ]
}

fn shader_answer() -> impl Strategy<Value = ShaderAnswer> {
    prop_oneof![
        Just(ShaderAnswer::Ready),
        any::<u64>().prop_map(|ticket| ShaderAnswer::Pending { ticket }),
        Just(ShaderAnswer::Rejected),
    ]
}

/// Every access kind, a read's outcome either bytes or a fault and a write's either.
fn access() -> impl Strategy<Value = WithBlobs<Access>> {
    prop_oneof![
        (
            any::<u32>(),
            any::<u64>(),
            prop_oneof![bytes_strategy().prop_map(Ok), fault().prop_map(Err)]
        )
            .prop_map(|(space, va, result)| match result {
                Ok(bytes) => (
                    Access::Read {
                        space: AsId(space),
                        va,
                        len: u64::try_from(bytes.len()).unwrap(),
                        result: Ok(ContentHash::of(&bytes)),
                    },
                    vec![bytes],
                ),
                Err(fault) => (
                    Access::Read {
                        space: AsId(space),
                        va,
                        len: va.rotate_left(7),
                        result: Err(fault),
                    },
                    vec![],
                ),
            }),
        (
            any::<u32>(),
            any::<u64>(),
            bytes_strategy(),
            prop::option::of(fault())
        )
            .prop_map(|(space, va, bytes, fault)| (
                Access::Write {
                    space: AsId(space),
                    va,
                    data: ContentHash::of(&bytes),
                    result: fault.map_or(Ok(()), Err),
                },
                vec![bytes],
            )),
        (any::<u32>(), any::<u64>(), any::<u64>(), any::<bool>()).prop_map(
            |(space, va, len, answer)| (
                Access::GpuOnly {
                    space: AsId(space),
                    va,
                    len,
                    answer,
                },
                vec![],
            )
        ),
        (any::<u32>(), any::<u64>()).prop_map(|(space, value)| (
            Access::Generation {
                space: AsId(space),
                value,
            },
            vec![],
        )),
    ]
}

fn state_record() -> impl Strategy<Value = WithBlobs<StateRecord>> {
    prop_oneof![
        (any::<u32>(), bytes_strategy()).prop_map(|(space, bytes)| (
            StateRecord::AddressTable {
                space: AsId(space),
                blob: ContentHash::of(&bytes),
            },
            vec![bytes],
        )),
        bytes_strategy().prop_map(|bytes| (
            StateRecord::Syncpoints {
                blob: ContentHash::of(&bytes),
            },
            vec![bytes],
        )),
        (
            0..EngineStateKind::ALL.len(),
            any::<u32>(),
            bytes_strategy()
        )
            .prop_map(|(kind, owner, bytes)| (
                StateRecord::Engine {
                    kind: EngineStateKind::ALL[kind],
                    owner,
                    blob: ContentHash::of(&bytes),
                },
                vec![bytes],
            )),
    ]
}

fn syncpoint_id() -> impl Strategy<Value = SyncpointId> {
    (0..SYNCPOINT_COUNT).prop_map(|id| SyncpointId::new(u8::try_from(id).unwrap()).unwrap())
}

fn vm_update() -> impl Strategy<Value = VmUpdate> {
    prop_oneof![
        page_range().prop_map(|range| VmUpdate::Unmap { range }),
        (page_range(), any::<u8>()).prop_map(|(range, kind)| VmUpdate::ModifyKind {
            range,
            kind: Kind(kind),
        }),
        page_range().prop_map(|range| VmUpdate::DestroyMemory { range }),
        (page_range(), any::<u32>()).prop_map(|(range, region)| {
            VmUpdate::DestroySparseRegion {
                range,
                region: RegionId(region),
            }
        }),
    ]
}

fn present() -> impl Strategy<Value = PresentRequest> {
    (
        (any::<u64>(), any::<[u32; 5]>()),
        any::<[u32; 4]>(),
        any::<u64>(),
        any::<u64>(),
    )
        .prop_map(
            |((address, [width, height, pitch, block, format]), [x, y, w, h], layer, display)| {
                PresentRequest {
                    image: ImageDesc {
                        address,
                        width,
                        height,
                        pitch,
                        block_height_log2: block,
                        format,
                    },
                    shown: Rect {
                        x,
                        y,
                        width: w,
                        height: h,
                    },
                    layer: LayerId(layer),
                    display: DisplayId(display),
                }
            },
        )
}

/// One generated front record of every kind, with every field drawn independently.
fn front_record() -> impl Strategy<Value = WithBlobs<FrontRecord>> {
    let plain = prop_oneof![
        Just(FrontRecord::Turn),
        (
            any::<u32>(),
            prop::collection::vec(any::<u64>(), 0..4),
            prop::option::of((syncpoint_id(), any::<u32>())),
            any::<u32>(),
            any::<bool>()
        )
            .prop_map(|(channel, entries, wait, increments, wfi)| {
                FrontRecord::Submit {
                    channel: ChannelId(channel),
                    submission: Submission {
                        entries: entries.into_iter().map(GpfifoEntry).collect(),
                        wait: wait.map(|(id, value)| crate::interface::Fence { id, value }),
                        increments,
                        wfi,
                    },
                }
            }),
        (any::<u32>(), vm_update(), prop::option::of(any::<u64>())).prop_map(
            |(space, update, ticket)| FrontRecord::Update {
                space: AsId(space),
                update,
                ticket: ticket.map(Ticket),
            }
        ),
        any::<u64>().prop_map(|ticket| FrontRecord::Barrier {
            ticket: Ticket(ticket)
        }),
        present().prop_map(|request| FrontRecord::Present { request }),
        any::<u32>().prop_map(|channel| FrontRecord::Destroy {
            channel: ChannelId(channel)
        }),
        (
            prop::collection::vec(any::<u64>(), 0..4),
            any::<bool>(),
            any::<bool>()
        )
            .prop_map(|(markers, released, incremented)| FrontRecord::Report {
                markers: markers.into_iter().map(crate::ring::MarkerToken).collect(),
                released,
                incremented,
            }),
        (syncpoint_id(), any::<u32>()).prop_map(|(id, value)| FrontRecord::Syncpoint { id, value }),
        any::<u32>().prop_map(|channel| FrontRecord::Latch {
            channel: ChannelId(channel)
        }),
        any::<u64>().prop_map(|nanos| FrontRecord::Clock {
            now: std::time::Duration::from_nanos(nanos)
        }),
    ];
    prop_oneof![
        plain.prop_map(|record| (record, vec![])),
        access().prop_map(|(access, blobs)| (FrontRecord::Access(access), blobs)),
    ]
}

/// The channels of [`wide`]'s header and their spaces, for generated waiter tables.
const WIDE_CHANNELS: [(u32, AsId); 3] = [(0, SPACE), (1, AsId(0x22)), (2, SPACE)];

/// `minimal()` with the channels of [`WIDE_CHANNELS`] in two spaces.
fn wide() -> Capture {
    let mut capture = minimal();
    capture.header.spaces.push(AsId(0x22));
    capture.header.channels = WIDE_CHANNELS
        .iter()
        .map(|&(id, space)| CapturedChannel {
            id: ChannelId(id),
            config: ChannelConfig {
                space,
                syncpoint: syncpoint(),
            },
        })
        .collect();
    capture
}

/// One generated back record of every kind; a waiter table lists some of [`wide`]'s channels,
/// each once and in its space, as the reader requires.
fn back_record() -> impl Strategy<Value = WithBlobs<BackRecord>> {
    let waiter = (
        any::<bool>(),
        any::<u64>(),
        0..5u8,
        any::<u32>(),
        any::<bool>(),
    );
    let plain = prop_oneof![
        Just(BackRecord::Apply),
        Just(BackRecord::Close),
        any::<u64>().prop_map(|completed| BackRecord::Complete { completed }),
        any::<u64>().prop_map(|nanos| BackRecord::Elapsed {
            elapsed: std::time::Duration::from_nanos(nanos)
        }),
        any::<u64>().prop_map(|value| BackRecord::Timestamp { value }),
        prop::array::uniform3(waiter).prop_map(|slots| BackRecord::Waiters {
            waiters: WIDE_CHANNELS
                .iter()
                .zip(slots)
                .filter(|(_, (listed, ..))| *listed)
                .map(|(&(channel, space), (_, address, op, payload, latched))| {
                    (
                        Waiter {
                            channel: ChannelId(channel),
                            space,
                            address,
                            op: crate::capture::format::acquire_from(op).unwrap(),
                            payload,
                        },
                        latched,
                    )
                })
                .collect(),
        }),
        (any::<u32>(), prop::collection::vec(page_range(), 0..3)).prop_map(|(space, ranges)| {
            BackRecord::Drain {
                space: AsId(space),
                ranges,
            }
        }),
        (
            any::<u32>(),
            prop::collection::vec((0..7u8, any::<u64>()), 0..3),
            shader_answer()
        )
            .prop_map(
                |(cache_accesses, requests, answer)| BackRecord::ShaderLookup {
                    cache_accesses,
                    requests: requests
                        .into_iter()
                        .map(|(stage, address)| ProgramRequest {
                            stage: crate::capture::format::stage_from(stage).unwrap(),
                            address,
                        })
                        .collect(),
                    answer,
                }
            ),
        (any::<u64>(), shader_answer())
            .prop_map(|(ticket, answer)| BackRecord::ShaderWait { ticket, answer }),
    ];
    prop_oneof![
        plain.prop_map(|record| (record, vec![])),
        access().prop_map(|(access, blobs)| (BackRecord::Access(access), blobs)),
    ]
}

proptest! {
    #![proptest_config(proptest_config())]

    /// Every record kind, every sub-kind and every optional field, written and read back; the
    /// reader's memory charge is the writer's count of it, and at least the file's length.
    #[test]
    fn generated_captures_read_back_as_written(
        state in prop::collection::vec(state_record(), 0..4),
        front in prop::collection::vec(front_record(), 0..12),
        back in prop::collection::vec(back_record(), 0..12),
        rings in prop::collection::vec(any::<u128>(), 0..4),
        build_id in "[a-z ]{0,12}",
    ) {
        let mut capture = wide();
        capture.header.build_id = build_id;
        for (record, blobs) in state {
            for blob in blobs {
                capture.blobs.insert(&blob);
            }
            capture.state.push(record);
        }
        for (record, blobs) in front {
            for blob in blobs {
                capture.blobs.insert(&blob);
            }
            capture.front.push(record);
        }
        for (record, blobs) in back {
            for blob in blobs {
                capture.blobs.insert(&blob);
            }
            capture.back.push(record);
        }
        let text = capture.blobs.insert(b"render");
        capture.outputs = rings
            .into_iter()
            .map(|hash| Output::Ring { hash: ContentHash::new(hash) })
            .chain([Output::Render { text }])
            .collect();
        let file = capture.to_bytes().unwrap();
        let (read, charged) = Capture::parse_within(&file, Limits::DEFAULT).unwrap();
        prop_assert_eq!(&read, &capture);
        prop_assert_eq!(charged, capture.memory_charge());
        prop_assert!(charged >= file.len(), "{} charged for {} bytes", charged, file.len());
    }

    #[test]
    fn any_bytes_after_a_valid_header_read_or_fail_without_panicking(
        tail in prop::collection::vec(any::<u8>(), 0..256),
    ) {
        let capture = minimal();
        let mut file = one::header(&capture.header).unwrap();
        file.extend_from_slice(&tail);
        let _ = Capture::parse(&file);
    }
}
