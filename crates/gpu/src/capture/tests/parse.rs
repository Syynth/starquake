//! The reader: the header's checks in order, each refusal as its exact error, the bounds, and a
//! generated round trip.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use super::*;
use crate::capture::format::one;
use crate::capture::{
    Access, BackRecord, Blobs, Capture, CaptureError, CapturedChannel, ContentHash, FrontRecord,
    Header, MAX_BUILD_ID, MAX_CAPTURE_BYTES, Output, StateRecord, VERSION,
};
use crate::interface::{ChannelConfig, SYNCPOINT_COUNT};

/// A capture with one space, one channel, the caps blob and nothing else.
fn minimal() -> Capture {
    let mut blobs = Blobs::new();
    let caps = blobs.insert(&postcard::to_allocvec(&caps()).unwrap());
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
        outputs: vec![],
    }
}

/// Where the header ends in `capture`'s file.
fn header_len(capture: &Capture) -> usize {
    one::header(&capture.header).unwrap().len()
}

#[test]
fn a_minimal_capture_reads_back() {
    let capture = minimal();
    assert_eq!(Capture::parse(&capture.to_bytes().unwrap()).unwrap(), capture);
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
    assert_eq!(error, CaptureError::UnsupportedVersion { found: VERSION + 1 });
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
            len: MAX_CAPTURE_BYTES + 1
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

#[test]
fn the_header_lists_are_checked() {
    let mut capture = minimal();
    capture.header.spaces.push(SPACE);
    let error = Capture::parse(&capture.to_bytes().unwrap()).unwrap_err();
    assert_eq!(error, CaptureError::DuplicateSpace { space: SPACE });
    assert_eq!(error.to_string(), "address space 33 is listed twice");

    let mut capture = minimal();
    let first = capture.header.channels[0];
    capture.header.channels.push(first);
    assert_eq!(
        Capture::parse(&capture.to_bytes().unwrap()).unwrap_err(),
        CaptureError::DuplicateChannel {
            channel: ChannelId(0)
        }
    );

    let mut capture = minimal();
    capture.header.channels[0].config.space = AsId(0x99);
    let error = Capture::parse(&capture.to_bytes().unwrap()).unwrap_err();
    assert_eq!(
        error,
        CaptureError::UnknownSpace {
            channel: ChannelId(0),
            space: AsId(0x99)
        }
    );
    assert_eq!(
        error.to_string(),
        "channel 0 is in address space 153, which the header does not list"
    );
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
            [vec![0x12, 1, 0, 0, 0, 0], 0x1000u64.to_le_bytes().to_vec(), 0x800u64.to_le_bytes().to_vec()].concat(),
            6,
            "a page range is not whole pages",
        ),
        // An update of kind 4.
        (vec![0x12, 1, 0, 0, 0, 4], 5, "an update's kind is unknown"),
        // A submission whose wait flag is 2.
        (vec![0x11, 1, 0, 0, 0, 2], 5, "a presence flag is neither 0 nor 1"),
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
        (vec![0x27, 1, 0, 0, 0, 7], 5, "a shader stage is unknown"),
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
    let first_len = 16 + 8 + usize::try_from(u64::from_le_bytes(
        file[table + 4 + 16..table + 4 + 24].try_into().unwrap(),
    ))
    .unwrap();
    let rest_start = table + 4;
    let first = file[rest_start..rest_start + first_len].to_vec();
    let second_end = file.len() - 4;
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
    let count = file.len() - 4;
    file[count..].copy_from_slice(&1u32.to_le_bytes());
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

fn config() -> Config {
    Config {
        cases: 256,
        rng_seed: RngSeed::Fixed(0x5351_4743),
        failure_persistence: None,
        ..Config::default()
    }
}

/// One generated record of each log, with every field drawn independently.
fn front_record() -> impl Strategy<Value = FrontRecord> {
    prop_oneof![
        Just(FrontRecord::Turn),
        (any::<u32>(), prop::collection::vec(any::<u64>(), 0..4), any::<u32>(), any::<bool>())
            .prop_map(|(channel, entries, increments, wfi)| FrontRecord::Submit {
                channel: ChannelId(channel),
                submission: Submission {
                    entries: entries.into_iter().map(GpfifoEntry).collect(),
                    wait: None,
                    increments,
                    wfi,
                },
            }),
        (prop::collection::vec(any::<u64>(), 0..4), any::<bool>(), any::<bool>()).prop_map(
            |(markers, released, incremented)| FrontRecord::Report {
                markers: markers.into_iter().map(crate::ring::MarkerToken).collect(),
                released,
                incremented,
            }
        ),
        (0..SYNCPOINT_COUNT, any::<u32>()).prop_map(|(id, value)| FrontRecord::Syncpoint {
            id: SyncpointId::new(u8::try_from(id).unwrap()).unwrap(),
            value,
        }),
        any::<u64>().prop_map(|nanos| FrontRecord::Clock {
            now: std::time::Duration::from_nanos(nanos)
        }),
        (any::<u32>(), any::<u64>(), any::<u64>(), any::<bool>()).prop_map(
            |(space, va, len, answer)| FrontRecord::Access(Access::GpuOnly {
                space: AsId(space),
                va,
                len,
                answer,
            })
        ),
    ]
}

fn back_record() -> impl Strategy<Value = BackRecord> {
    prop_oneof![
        Just(BackRecord::Apply),
        Just(BackRecord::Close),
        any::<u64>().prop_map(|completed| BackRecord::Complete { completed }),
        any::<u64>().prop_map(|value| BackRecord::Timestamp { value }),
        (any::<u32>(), any::<u64>()).prop_map(|(space, value)| BackRecord::Access(
            Access::Generation {
                space: AsId(space),
                value,
            }
        )),
    ]
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn generated_captures_read_back_as_written(
        front in prop::collection::vec(front_record(), 0..8),
        back in prop::collection::vec(back_record(), 0..8),
        rings in prop::collection::vec(any::<u128>(), 0..4),
        build_id in "[a-z ]{0,12}",
    ) {
        let mut capture = minimal();
        capture.header.build_id = build_id;
        capture.front = front;
        capture.back = back;
        let text = capture.blobs.insert(b"render");
        capture.state.push(StateRecord::Syncpoints { blob: text });
        capture.outputs = rings
            .into_iter()
            .map(|hash| Output::Ring { hash: ContentHash::new(hash) })
            .chain([Output::Render { text }])
            .collect();
        let file = capture.to_bytes().unwrap();
        prop_assert_eq!(Capture::parse(&file).unwrap(), capture);
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
