//! The host file protocol: its bytes, its decoder's errors, and how replies are sorted.

use std::sync::Mutex;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_trace::{EnvError, HostNanos};

use super::*;

/// A request of every kind with distinct, non-zero values, and its bytes laid out by hand.
fn golden() -> [(HostRequest, Vec<u8>); 3] {
    let probe = HostRequest::Probe {
        mount: Mount::SdCard,
    };
    let probe_bytes = [[1, 1, 1, 0], [0; 4]]
        .concat()
        .into_iter()
        .chain([0; 16])
        .collect();
    let size = HostRequest::Size {
        mount: Mount::SdCard,
        path: b"/ab".to_vec(),
    };
    let size_bytes = [&[1, 2, 1, 0, 3, 0, 0, 0][..], &[0; 16], b"/ab"].concat();
    let read = HostRequest::Read {
        mount: Mount::SdCard,
        path: b"/x".to_vec(),
        offset: 0x0102_0304_0506_0708,
        len: 0x1112_1314_1516_1718,
    };
    let read_bytes = [
        &[1, 3, 1, 0, 2, 0, 0, 0][..],
        &[8, 7, 6, 5, 4, 3, 2, 1],
        &[0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11],
        b"/x",
    ]
    .concat();
    [(probe, probe_bytes), (size, size_bytes), (read, read_bytes)]
}

#[test]
fn every_request_has_these_bytes() {
    for (request, bytes) in golden() {
        assert_eq!(request.encode(), bytes, "{request:?}");
        assert_eq!(HostRequest::decode(&bytes), Ok(request.clone()));
        let io = request.to_host_io();
        assert_eq!((io.channel, io.payload), (CHANNEL, bytes));
    }
    // The header's fields are where the layout says.
    let bytes = &golden()[2].1;
    assert_eq!(bytes[layout::VERSION], VERSION);
    assert_eq!(bytes[layout::OPERATION], 3);
    assert_eq!(bytes[layout::MOUNT], Mount::SdCard.number());
    assert_eq!(bytes[layout::RESERVED], 0);
    assert_eq!(bytes[layout::PATH_LEN], 2);
    assert_eq!(bytes[layout::OFFSET], 8);
    assert_eq!(bytes[layout::LENGTH], 0x18);
    assert_eq!(&bytes[HEADER_SIZE..], b"/x");
}

/// Probe bytes with one header byte changed.
fn probe_with(offset: usize, value: u8) -> Vec<u8> {
    let mut bytes = golden()[0].1.clone();
    bytes[offset] = value;
    bytes
}

#[test]
fn a_request_shorter_than_its_header_is_truncated() {
    let bytes = golden()[0].1.clone();
    assert_eq!(bytes.len(), HEADER_SIZE);
    assert!(HostRequest::decode(&bytes).is_ok());
    let short = &bytes[..HEADER_SIZE - 1];
    let error = HostRequest::decode(short).unwrap_err();
    assert_eq!(
        error,
        RequestError::Truncated {
            len: HEADER_SIZE - 1
        }
    );
    assert_eq!(
        error.to_string(),
        "the request is 23 bytes, shorter than its 24-byte header"
    );
    assert_eq!(
        HostRequest::decode(&[]),
        Err(RequestError::Truncated { len: 0 })
    );
}

#[test]
fn each_header_field_is_checked() {
    let cases = [
        (
            layout::VERSION,
            0,
            RequestError::Version(0),
            "the request is version 0, not 1",
        ),
        (
            layout::VERSION,
            2,
            RequestError::Version(2),
            "the request is version 2, not 1",
        ),
        (
            layout::OPERATION,
            0,
            RequestError::Operation(0),
            "operation 0 is not a host file request",
        ),
        (
            layout::OPERATION,
            4,
            RequestError::Operation(4),
            "operation 4 is not a host file request",
        ),
        (
            layout::MOUNT,
            0,
            RequestError::Mount(0),
            "mount 0 is not a host file system",
        ),
        (
            layout::MOUNT,
            2,
            RequestError::Mount(2),
            "mount 2 is not a host file system",
        ),
        (
            layout::RESERVED,
            1,
            RequestError::Reserved(1),
            "the reserved byte is 0x1, not zero",
        ),
    ];
    for (offset, value, error, message) in cases {
        let decoded = HostRequest::decode(&probe_with(offset, value));
        assert_eq!(decoded, Err(error.clone()));
        assert_eq!(error.to_string(), message);
    }
}

#[test]
fn the_header_is_checked_in_the_order_of_its_fields() {
    // Every field wrong at once: the version is reported.
    let mut bytes = probe_with(layout::VERSION, 9);
    bytes[layout::OPERATION] = 9;
    bytes[layout::MOUNT] = 9;
    bytes[layout::RESERVED] = 9;
    assert_eq!(HostRequest::decode(&bytes), Err(RequestError::Version(9)));
    bytes[layout::VERSION] = VERSION;
    assert_eq!(HostRequest::decode(&bytes), Err(RequestError::Operation(9)));
    bytes[layout::OPERATION] = 1;
    assert_eq!(HostRequest::decode(&bytes), Err(RequestError::Mount(9)));
    bytes[layout::MOUNT] = 1;
    assert_eq!(HostRequest::decode(&bytes), Err(RequestError::Reserved(9)));
    // A wrong reserved byte before a path length that does not match.
    bytes[layout::PATH_LEN] = 5;
    assert_eq!(HostRequest::decode(&bytes), Err(RequestError::Reserved(9)));
    bytes[layout::RESERVED] = 0;
    let mismatch = RequestError::PathLength {
        stated: 5,
        follows: 0,
    };
    assert_eq!(HostRequest::decode(&bytes), Err(mismatch));
}

fn size_request(path_len: usize) -> HostRequest {
    HostRequest::Size {
        mount: Mount::SdCard,
        path: vec![b'a'; path_len],
    }
}

#[test]
fn the_path_length_must_match_and_fit() {
    // The longest path a request may carry, and one byte more.
    let longest = size_request(ENTRY_NAME_LENGTH_MAX);
    assert_eq!(HostRequest::decode(&longest.encode()), Ok(longest));
    let too_long = size_request(ENTRY_NAME_LENGTH_MAX + 1).encode();
    let stated = u32::try_from(ENTRY_NAME_LENGTH_MAX + 1).unwrap();
    let error = RequestError::PathLength {
        stated,
        follows: ENTRY_NAME_LENGTH_MAX + 1,
    };
    assert_eq!(HostRequest::decode(&too_long), Err(error.clone()));
    assert_eq!(
        error.to_string(),
        "the path length 769 does not match the 769 bytes after the header, or passes the \
         768-byte limit"
    );
    // A stated length one more, and one less, than the bytes that follow.
    let mut bytes = size_request(3).encode();
    for stated in [2u32, 4, u32::MAX] {
        bytes[layout::PATH_LEN..layout::PATH_LEN + 4].copy_from_slice(&stated.to_le_bytes());
        let error = RequestError::PathLength { stated, follows: 3 };
        assert_eq!(HostRequest::decode(&bytes), Err(error));
    }
}

#[test]
fn fields_an_operation_does_not_use_must_be_zero() {
    // A probe with a path, with an offset, with a length; a size with an offset or a length.
    let mut probe_path = probe_with(layout::PATH_LEN, 1);
    probe_path.push(b'/');
    let probe_offset = probe_with(layout::OFFSET, 1);
    let probe_length = probe_with(layout::LENGTH + 7, 1);
    let mut size_offset = size_request(1).encode();
    size_offset[layout::OFFSET + 3] = 1;
    let mut size_length = size_request(1).encode();
    size_length[layout::LENGTH] = 1;
    for (bytes, op) in [
        (probe_path, 1),
        (probe_offset, 1),
        (probe_length, 1),
        (size_offset, 2),
        (size_length, 2),
    ] {
        let error = RequestError::UnusedField(op);
        assert_eq!(HostRequest::decode(&bytes), Err(error.clone()));
        assert_eq!(
            error.to_string(),
            format!("a field operation {op} does not use is not zero")
        );
    }
    // A read uses both, and an empty path is still a path.
    let read = HostRequest::Read {
        mount: Mount::SdCard,
        path: Vec::new(),
        offset: 0,
        len: 0,
    };
    assert_eq!(HostRequest::decode(&read.encode()), Ok(read));
}

#[test]
fn failure_codes_name_their_failures() {
    let codes: Vec<u32> = HostFailure::ALL.iter().map(|f| f.code()).collect();
    assert_eq!(codes, [1, 2, 3, 4, 5]);
    for failure in HostFailure::ALL {
        assert_eq!(HostFailure::from_code(failure.code()), Some(failure));
        assert_eq!(failure.reply(), HostIoReply::Failed(failure.code()));
    }
    assert_eq!(HostFailure::from_code(0), None);
    assert_eq!(HostFailure::from_code(6), None);
    assert_eq!(
        Mount::from_number(Mount::SdCard.number()),
        Some(Mount::SdCard)
    );
}

/// An environment that answers host I/O with one scripted reply and keeps what it was asked.
struct Scripted {
    reply: Result<HostIoReply, EnvError>,
    asked: Mutex<Vec<HostIoRequest>>,
}

impl Env for Scripted {
    fn now(&self) -> Result<HostNanos, EnvError> {
        Ok(HostNanos(0))
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        Ok(0)
    }

    fn host_io(&self, request: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        self.asked.lock().unwrap().push(request.clone());
        self.reply.clone()
    }
}

#[test]
fn replies_are_sorted_by_what_they_say() {
    let exhausted = EnvError::Exhausted {
        index: 0,
        requested: starquake_trace::EnvCall::HostIo,
    };
    let cases = [
        (Ok(HostIoReply::Data(vec![1, 2])), Answer::Data(vec![1, 2])),
        (
            Ok(HostIoReply::Failed(HostFailure::NotAFile.code())),
            Answer::Failed(HostFailure::NotAFile),
        ),
        (Ok(HostIoReply::Failed(99)), Answer::Broken),
        (Ok(HostIoReply::Unavailable), Answer::Unavailable),
        (Err(exhausted), Answer::Broken),
    ];
    let request = &golden()[1].0;
    for (reply, answer) in cases {
        let env = Scripted {
            reply,
            asked: Mutex::default(),
        };
        assert_eq!(ask(&env, request), answer);
        assert_eq!(*env.asked.lock().unwrap(), [request.to_host_io()]);
    }
}

fn config() -> Config {
    Config {
        cases: 1024,
        failure_persistence: None,
        rng_seed: RngSeed::Fixed(0x484F_5354_494F_0001),
        rng_algorithm: RngAlgorithm::ChaCha,
        ..Config::default()
    }
}

fn requests() -> impl Strategy<Value = HostRequest> {
    let path = || proptest::collection::vec(any::<u8>(), 0..=ENTRY_NAME_LENGTH_MAX);
    prop_oneof![
        Just(HostRequest::Probe {
            mount: Mount::SdCard
        }),
        path().prop_map(|path| HostRequest::Size {
            mount: Mount::SdCard,
            path
        }),
        (path(), any::<u64>(), any::<u64>()).prop_map(|(path, offset, len)| HostRequest::Read {
            mount: Mount::SdCard,
            path,
            offset,
            len
        }),
    ]
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn a_request_decodes_to_itself(request in requests()) {
        let bytes = request.encode();
        prop_assert_eq!(HostRequest::decode(&bytes), Ok(request));
    }
}
