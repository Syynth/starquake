//! `set:sys` driven through raw messages laid out the way libnx lays them out
//! (`libnx/nx/source/services/set.c`, `_setsysGetFirmwareVersionImpl`).

use starquake_ipc::{
    BUFFER_SIZE, BufferDescriptor, CommandType, ReceiveDescriptor, RequestBuilder, SFCO_MAGIC,
    TIPC_COMMAND_OFFSET,
};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;

use super::*;
use crate::framework::fake::FakeKernel;
use crate::framework::result::{INVALID_CMIF_REQUEST, UNKNOWN_COMMAND_ID};
use crate::framework::{Family, Handled, Session};

/// `set:sys` command ids (`ISystemSettingsServer::ISystemSettingsServer`).
const GET_FIRMWARE_VERSION: u32 = 3;
const GET_FIRMWARE_VERSION2: u32 = 4;
const GET_FIRMWARE_VERSION_DIGEST: u32 = 5;

/// Where the caller's buffer is, and what fills it and the message buffer beforehand.
const AT: u64 = 0x0000_0012_3456_7000;
const STALE: u8 = 0xA5;

/// A version whose every field is distinct and not zero, so a field written in another's place
/// shows.
fn version() -> FirmwareVersion {
    FirmwareVersion::new(FirmwareVersionFields {
        major: 0x11,
        minor: 0x12,
        micro: 0x13,
        revision_major: 0x14,
        revision_minor: 0x15,
        platform: "PLATFORM".to_owned(),
        version_hash: "hash-0123456789abcdef".to_owned(),
        display_version: "17.18.19".to_owned(),
        display_title: "A title for the firmware".to_owned(),
    })
    .unwrap()
}

/// The bytes of [`version`], laid out by hand from `FirmwareVersionFormat`
/// (`eden/src/core/hle/service/set/settings_types.h`).
fn version_bytes() -> Vec<u8> {
    let mut bytes = vec![0x11, 0x12, 0x13, 0, 0x14, 0x15, 0, 0];
    for (text, size) in [
        ("PLATFORM", 0x20),
        ("hash-0123456789abcdef", 0x40),
        ("17.18.19", 0x18),
        ("A title for the firmware", 0x80),
    ] {
        let mut field = text.as_bytes().to_vec();
        field.resize(size, 0);
        bytes.extend(field);
    }
    bytes
}

/// A `set:sys` server reporting [`version`], a kernel, and one session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

fn rig() -> Rig {
    let mut kernel = FakeKernel::new(0).unwrap();
    let session = kernel.new_object();
    let settings = Settings {
        firmware: version(),
    };
    let mut server = sys_server(settings, UnknownPolicy::Strict);
    server.accept(session).unwrap();
    Rig {
        kernel,
        server,
        session,
    }
}

impl Rig {
    fn send(&mut self, request: &RequestBuilder) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self
            .server
            .handle(&mut self.kernel, self.session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    /// Maps `len` stale bytes at [`AT`] as the caller's buffer.
    fn map(&mut self, len: usize) {
        self.kernel.map(AT, vec![STALE; len]);
    }

    fn written(&self, len: usize) -> &[u8] {
        self.kernel.memory(AT, len).unwrap()
    }
}

/// The request as libnx builds it: no payload, and one fixed-size pointer buffer for the
/// result, which is a receive-list (C) descriptor.
fn get_firmware_version(command: u32, size: usize) -> RequestBuilder {
    let room = ReceiveDescriptor {
        address: AT,
        size: u16::try_from(size).unwrap(),
    };
    RequestBuilder::cmif(CommandType::Request, command).receive_descriptor(room)
}

/// A CMIF reply that holds only a result: the header (6 raw words), padding to 16 bytes,
/// `SFCO`, version, result, token, and zeros to the end of the buffer.
fn only(result: ResultCode) -> Vec<u8> {
    let words = [0, 6, 0, 0, SFCO_MAGIC, 0, result.raw(), 0];
    let mut buffer: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    buffer.resize(BUFFER_SIZE, 0);
    buffer
}

#[test]
fn the_layout_is_the_reference_s() {
    let bytes = version().to_bytes();
    assert_eq!(bytes.len(), FIRMWARE_VERSION_SIZE);
    assert_eq!(bytes.to_vec(), version_bytes());
    // Each field is where `FirmwareVersionFormat` puts it.
    assert_eq!(bytes[layout::MAJOR], 0x11);
    assert_eq!(bytes[layout::MINOR], 0x12);
    assert_eq!(bytes[layout::MICRO], 0x13);
    assert_eq!(bytes[layout::REVISION_MAJOR], 0x14);
    assert_eq!(bytes[layout::REVISION_MINOR], 0x15);
    let text = |offset: usize, len: usize| &bytes[offset..offset + len];
    assert_eq!(text(layout::PLATFORM, 8), b"PLATFORM");
    assert_eq!(text(layout::VERSION_HASH, 5), b"hash-");
    assert_eq!(text(layout::DISPLAY_VERSION, 8), b"17.18.19");
    assert_eq!(text(layout::DISPLAY_TITLE, 7), b"A title");
    // The fields tile the struct: each starts where the one before ends.
    assert_eq!(
        layout::PLATFORM + layout::PLATFORM_SIZE,
        layout::VERSION_HASH
    );
    assert_eq!(
        layout::VERSION_HASH + layout::VERSION_HASH_SIZE,
        layout::DISPLAY_VERSION
    );
    assert_eq!(
        layout::DISPLAY_VERSION + layout::DISPLAY_VERSION_SIZE,
        layout::DISPLAY_TITLE
    );
    assert_eq!(
        layout::DISPLAY_TITLE + layout::DISPLAY_TITLE_SIZE,
        FIRMWARE_VERSION_SIZE
    );
}

#[test]
fn get_firmware_version2_writes_the_version_and_replies_these_bytes() {
    let mut rig = rig();
    rig.map(FIRMWARE_VERSION_SIZE);
    let reply = rig.send(&get_firmware_version(
        GET_FIRMWARE_VERSION2,
        FIRMWARE_VERSION_SIZE,
    ));
    // The reply has no payload; the version is in the caller's buffer.
    assert_eq!(reply, only(ResultCode::SUCCESS));
    assert_eq!(rig.written(FIRMWARE_VERSION_SIZE), version_bytes());
}

#[test]
fn get_firmware_version_zeroes_the_minor_revision() {
    let mut rig = rig();
    rig.map(FIRMWARE_VERSION_SIZE);
    let reply = rig.send(&get_firmware_version(
        GET_FIRMWARE_VERSION,
        FIRMWARE_VERSION_SIZE,
    ));
    assert_eq!(reply, only(ResultCode::SUCCESS));
    let mut expected = version_bytes();
    assert_eq!(expected[layout::REVISION_MINOR], 0x15);
    expected[layout::REVISION_MINOR] = 0;
    assert_eq!(rig.written(FIRMWARE_VERSION_SIZE), expected);
    // The settings themselves are untouched: command 4 still reports it.
    rig.send(&get_firmware_version(
        GET_FIRMWARE_VERSION2,
        FIRMWARE_VERSION_SIZE,
    ));
    assert_eq!(rig.written(FIRMWARE_VERSION_SIZE), version_bytes());
}

#[test]
fn a_larger_buffer_gets_the_version_and_nothing_after_it() {
    let mut rig = rig();
    let size = FIRMWARE_VERSION_SIZE + 1;
    rig.map(size);
    let reply = rig.send(&get_firmware_version(GET_FIRMWARE_VERSION2, size));
    assert_eq!(reply, only(ResultCode::SUCCESS));
    let written = rig.written(size);
    assert_eq!(written[..FIRMWARE_VERSION_SIZE], version_bytes());
    assert_eq!(written[FIRMWARE_VERSION_SIZE], STALE);
}

/// Divergence from Eden, whose `WriteBufferC` writes as much as fits and replies success.
#[test]
fn a_buffer_too_small_gets_nothing_and_an_error() {
    for command in [GET_FIRMWARE_VERSION, GET_FIRMWARE_VERSION2] {
        let mut rig = rig();
        // One byte short.
        let size = FIRMWARE_VERSION_SIZE - 1;
        rig.map(size);
        let reply = rig.send(&get_firmware_version(command, size));
        assert_eq!(reply, only(INVALID_CMIF_REQUEST));
        assert!(rig.written(size).iter().all(|byte| *byte == STALE));
    }
}

/// Divergence from Eden, which logs a failed assertion, writes nothing and replies success.
#[test]
fn a_request_without_a_buffer_fails() {
    let mut rig = rig();
    for command in [GET_FIRMWARE_VERSION, GET_FIRMWARE_VERSION2] {
        let bare = RequestBuilder::cmif(CommandType::Request, command);
        assert_eq!(rig.send(&bare), only(INVALID_CMIF_REQUEST));
    }
}

/// Divergence from Eden, whose `WriteBufferC` looks at C descriptors only and so writes nothing
/// here: a caller that offers a map-alias (B) buffer gets the version in it.
#[test]
fn a_map_alias_buffer_is_written_too() {
    let mut rig = rig();
    rig.map(FIRMWARE_VERSION_SIZE);
    let room = BufferDescriptor {
        address: AT,
        size: u64::try_from(FIRMWARE_VERSION_SIZE).unwrap(),
        flags: 0,
    };
    let ask = RequestBuilder::cmif(CommandType::Request, GET_FIRMWARE_VERSION2);
    assert_eq!(
        rig.send(&ask.receive_buffer(room)),
        only(ResultCode::SUCCESS)
    );
    assert_eq!(rig.written(FIRMWARE_VERSION_SIZE), version_bytes());
}

#[test]
fn a_buffer_the_caller_has_not_mapped_is_the_kernels_error() {
    use starquake_kernel::result::KernelError;
    let mut rig = rig();
    // Mapped one byte short of what the descriptor claims.
    rig.map(FIRMWARE_VERSION_SIZE - 1);
    let reply = rig.send(&get_firmware_version(
        GET_FIRMWARE_VERSION2,
        FIRMWARE_VERSION_SIZE,
    ));
    assert_eq!(reply, only(KernelError::InvalidCurrentMemory.code()));
    assert!(
        rig.written(FIRMWARE_VERSION_SIZE - 1)
            .iter()
            .all(|byte| *byte == STALE)
    );
}

#[test]
fn commands_that_are_left_out_go_to_the_unknown_command_policy() {
    let mut rig = rig();
    let digest = RequestBuilder::cmif(CommandType::Request, GET_FIRMWARE_VERSION_DIGEST);
    assert_eq!(rig.send(&digest), only(UNKNOWN_COMMAND_ID));
    // `set:sys` has no TIPC commands.
    let id = u16::try_from(GET_FIRMWARE_VERSION).unwrap();
    let reply = rig.send(&RequestBuilder::tipc(id));
    // A TIPC reply: the request's type, one raw word, the result.
    let words = [
        u32::from(TIPC_COMMAND_OFFSET + id),
        1,
        UNKNOWN_COMMAND_ID.raw(),
    ];
    let mut expected: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    expected.resize(BUFFER_SIZE, 0);
    assert_eq!(reply, expected);
}

/// Sets one text field of a version's fields.
type Set = fn(&mut FirmwareVersionFields, String);

#[test]
fn a_version_s_text_must_fit_its_field_with_a_terminator() {
    let fields = version().fields().clone();
    // Exactly the most that fits: one byte less than the field.
    let longest = FirmwareVersionFields {
        platform: "p".repeat(layout::PLATFORM_SIZE - 1),
        version_hash: "h".repeat(layout::VERSION_HASH_SIZE - 1),
        display_version: "v".repeat(layout::DISPLAY_VERSION_SIZE - 1),
        display_title: "t".repeat(layout::DISPLAY_TITLE_SIZE - 1),
        ..fields.clone()
    };
    let bytes = FirmwareVersion::new(longest).unwrap().to_bytes();
    // Every field ends in its one zero byte, and the next starts right after.
    for (offset, size, fill) in [
        (layout::PLATFORM, layout::PLATFORM_SIZE, b'p'),
        (layout::VERSION_HASH, layout::VERSION_HASH_SIZE, b'h'),
        (layout::DISPLAY_VERSION, layout::DISPLAY_VERSION_SIZE, b'v'),
        (layout::DISPLAY_TITLE, layout::DISPLAY_TITLE_SIZE, b't'),
    ] {
        assert!(bytes[offset..offset + size - 1].iter().all(|b| *b == fill));
        assert_eq!(bytes[offset + size - 1], 0);
    }

    // One byte more does not fit, in any field.
    let cases: [(&str, usize, Set); 4] = [
        ("platform", layout::PLATFORM_SIZE, |f, s| f.platform = s),
        ("version_hash", layout::VERSION_HASH_SIZE, |f, s| {
            f.version_hash = s;
        }),
        ("display_version", layout::DISPLAY_VERSION_SIZE, |f, s| {
            f.display_version = s;
        }),
        ("display_title", layout::DISPLAY_TITLE_SIZE, |f, s| {
            f.display_title = s;
        }),
    ];
    for (field, size, set) in cases {
        let mut too_long = fields.clone();
        set(&mut too_long, "x".repeat(size));
        let error = FirmwareVersionError::TooLong {
            field,
            len: size,
            max: size - 1,
        };
        assert_eq!(FirmwareVersion::new(too_long), Err(error));
        let mut with_zero = fields.clone();
        set(&mut with_zero, "a\0b".to_owned());
        assert_eq!(
            FirmwareVersion::new(with_zero),
            Err(FirmwareVersionError::EmbeddedZero { field })
        );
    }
    assert_eq!(
        FirmwareVersionError::TooLong {
            field: "platform",
            len: 32,
            max: 31
        }
        .to_string(),
        "platform is 32 bytes, more than the 31 that fit its field with a terminator"
    );
}

#[test]
fn the_default_is_the_version_eden_synthesises() {
    let bytes = FirmwareVersion::eden_default().to_bytes();
    // `eden/src/core/hle/api_version.h`: 23.0.0, SDK revision 1.0, "NX".
    assert_eq!(bytes[..8], [23, 0, 0, 0, 1, 0, 0, 0]);
    let text = |offset: usize, size: usize| {
        let field = &bytes[offset..offset + size];
        let end = field.iter().position(|b| *b == 0).unwrap();
        String::from_utf8(field[..end].to_vec()).unwrap()
    };
    assert_eq!(text(layout::PLATFORM, layout::PLATFORM_SIZE), "NX");
    assert_eq!(
        text(layout::VERSION_HASH, layout::VERSION_HASH_SIZE),
        "c2663accb7490a6ccfe9bc4dfd120ca215490525"
    );
    assert_eq!(
        text(layout::DISPLAY_VERSION, layout::DISPLAY_VERSION_SIZE),
        "23.0.0"
    );
    assert_eq!(
        text(layout::DISPLAY_TITLE, layout::DISPLAY_TITLE_SIZE),
        "NintendoSDK Firmware for NX 23.0.0-4.0"
    );
    // It passes the checks every other version passes.
    let fields = FirmwareVersion::eden_default().fields().clone();
    assert_eq!(
        FirmwareVersion::new(fields),
        Ok(FirmwareVersion::eden_default())
    );
}

#[test]
fn the_state_a_snapshot_holds_is_these_bytes() {
    let settings = Settings {
        firmware: FirmwareVersion::new(FirmwareVersionFields {
            major: 1,
            minor: 2,
            micro: 3,
            revision_major: 4,
            revision_minor: 5,
            platform: "P".to_owned(),
            version_hash: "HH".to_owned(),
            display_version: "V.V".to_owned(),
            display_title: "TTTT".to_owned(),
        })
        .unwrap(),
    };
    // `postcard`: the five numbers, then four length-prefixed strings, in the struct's order.
    #[rustfmt::skip]
    let golden = [
        1, 2, 3, 4, 5,
        1, b'P',
        2, b'H', b'H',
        3, b'V', b'.', b'V',
        4, b'T', b'T', b'T', b'T',
    ];
    assert_eq!(Family::save_state(&settings).unwrap(), golden);
    let mut loaded = Settings {
        firmware: FirmwareVersion::eden_default(),
    };
    loaded.load_state(&golden).unwrap();
    assert_eq!(loaded, settings);
    // A session has no state of its own.
    assert_eq!(SetSys.save().unwrap(), []);

    // Saved fields are checked when they are read back: a platform of 32 bytes does not fit.
    let mut too_long = vec![1, 2, 3, 4, 5, 32];
    too_long.extend([b'p'; 32]);
    too_long.extend([0, 0, 0]);
    assert_eq!(
        loaded.load_state(&too_long),
        Err(crate::framework::SnapshotError::Decode(
            postcard::Error::SerdeDeCustom
        ))
    );
    assert_eq!(loaded, settings);
    // One byte less does.
    let mut fits = vec![1, 2, 3, 4, 5, 31];
    fits.extend([b'p'; 31]);
    fits.extend([0, 0, 0]);
    assert_eq!(loaded.load_state(&fits), Ok(()));
}

#[test]
fn a_restored_set_sys_reports_the_saved_version() {
    let mut rig = rig();
    rig.map(FIRMWARE_VERSION_SIZE);
    let saved = rig.server.save().unwrap();
    // A server made with another version takes the saved one's state.
    let other = Settings {
        firmware: FirmwareVersion::eden_default(),
    };
    rig.server = sys_server(other, UnknownPolicy::Strict);
    rig.server.restore(&saved).unwrap();
    rig.send(&get_firmware_version(
        GET_FIRMWARE_VERSION2,
        FIRMWARE_VERSION_SIZE,
    ));
    assert_eq!(rig.written(FIRMWARE_VERSION_SIZE), version_bytes());
}
