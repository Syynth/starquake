//! Hand-assembled buffers: every word below was written out from the documented layout
//! (`eden/src/core/hle/ipc.h`, `hle_ipc.cpp`, scoping doc row 22), not produced by this crate's
//! builder. Each test then parses the bytes, and the reply tests also check that the builder
//! produces exactly the same bytes.

use starquake_ipc::{
    BUFFER_SIZE, Body, BufferDescriptor, CmifReplyHeader, CmifRequest, CmifRequestHeader,
    CommandType, DomainCommand, DomainReplyHeader, DomainRequestHeader, IpcError, Message,
    ReplyBuilder, SFCI_MAGIC, SFCO_MAGIC, SessionKind,
};

/// Serialises words little-endian and zero-fills to a whole IPC buffer.
fn buffer(words: &[u32]) -> Vec<u8> {
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert!(bytes.len() <= BUFFER_SIZE);
    bytes.resize(BUFFER_SIZE, 0);
    bytes
}

/// The CMIF request in a parsed message.
fn cmif_request<'m, 'a>(message: &'m Message<'a>) -> Option<&'m CmifRequest<'a>> {
    match &message.body {
        Body::CmifRequest(request) => Some(request),
        _ => None,
    }
}

fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Request: type 4, one A descriptor, special header with PID and one copied handle.
///
/// ```text
/// word  0  0x0010_0004  type 4 (Request), bits 20..24 = 1 A descriptor
/// word  1  0x8000_0009  9 raw words, bit 31: special header follows
/// word  2  0x0000_0003  special header: bit 0 send pid, bits 1..5 = 1 copy handle
/// word  3  0x0000_0042  pid low   (the kernel fills this slot)
/// word  4  0x0000_0000  pid high
/// word  5  0x0000_00AB  copied handle
/// word  6  0x0000_1234  A: size bits 0..32
/// word  7  0x2345_6000  A: address bits 0..32
/// word  8  0x1200_0015  A: flags 1, address 36..39 = 5, size 32..36 = 2, address 32..36 = 1
/// words 9..12           padding to the 16-byte boundary at word 12
/// word 12..16           SFCI, version 0, command 11, token 0
/// words 16, 17          arguments
/// ```
fn cmif_request_with_pid_handle_and_a_descriptor() -> Vec<u32> {
    vec![
        0x0010_0004,
        0x8000_0009,
        0x0000_0003,
        0x0000_0042,
        0x0000_0000,
        0x0000_00AB,
        0x0000_1234,
        0x2345_6000,
        0x1200_0015,
        0,
        0,
        0,
        SFCI_MAGIC,
        0,
        11,
        0,
        0x1111_2222,
        0x3333_4444,
    ]
}

#[test]
fn cmif_request_with_pid_copy_handle_and_a_descriptor() {
    let bytes = buffer(&cmif_request_with_pid_handle_and_a_descriptor());
    let message = Message::parse_request(&bytes, SessionKind::Session).unwrap();

    assert_eq!(message.command_type, CommandType::Request);
    assert_eq!(message.pid, Some(0x42));
    assert_eq!(message.copy_handles, [0xAB]);
    assert_eq!(message.move_handles, [0_u32; 0]);
    assert_eq!(message.static_descriptors, []);
    assert_eq!(
        message.send_buffers,
        [BufferDescriptor {
            address: 0x51_2345_6000,
            size: 0x2_0000_1234,
            flags: 1,
        }]
    );
    let request = cmif_request(&message).unwrap();
    assert_eq!(request.domain, None);
    assert_eq!(
        request.header,
        CmifRequestHeader {
            version: 0,
            command_id: 11,
            token: 0
        }
    );
    assert_eq!(request.data, le(&[0x1111_2222, 0x3333_4444]));
    assert_eq!(request.object_ids, [0_u32; 0]);
}

#[test]
fn cmif_request_tolerates_the_reserved_padding_word_count() {
    // Clients that count four reserved padding words in the raw data size (13 instead of 9) still
    // parse; the arguments are followed by the slack.
    let mut words = cmif_request_with_pid_handle_and_a_descriptor();
    words[1] = 0x8000_000D;
    let bytes = buffer(&words);
    let message = Message::parse_request(&bytes, SessionKind::Session).unwrap();
    let request = cmif_request(&message).unwrap();
    assert!(request.data.starts_with(&le(&[0x1111_2222, 0x3333_4444])));
    assert!(request.data[8..].iter().all(|b| *b == 0));
}

/// Domain request with one input object.
///
/// ```text
/// word  0  0x0000_0004  type 4
/// word  1  0x0000_000C  12 raw words
/// words 2, 3            padding to word 4
/// word  4  0x0014_0101  domain header: command 1 (send), 1 object, size 20 bytes (SFCI + args)
/// word  5  0x0000_0003  object id 3
/// word  6  0x0000_0000  padding
/// word  7  0x0000_0000  token
/// words 8..12           SFCI, version 0, command 5, token 0
/// word 12  0xCAFE_F00D  argument
/// word 13  0x0000_0007  input object id
/// ```
#[test]
fn domain_request_with_input_object() {
    let bytes = buffer(&[
        0x0000_0004,
        0x0000_000C,
        0,
        0,
        0x0014_0101,
        0x0000_0003,
        0,
        0,
        SFCI_MAGIC,
        0,
        5,
        0,
        0xCAFE_F00D,
        0x0000_0007,
    ]);
    let message = Message::parse_request(&bytes, SessionKind::Domain).unwrap();

    assert_eq!(message.command_type, CommandType::Request);
    let request = cmif_request(&message).unwrap();
    assert_eq!(
        request.domain,
        Some(DomainRequestHeader {
            command: DomainCommand::SendMessage,
            input_object_count: 1,
            size: 20,
            object_id: 3,
            token: 0,
        })
    );
    assert_eq!(request.header.command_id, 5);
    assert_eq!(request.data, le(&[0xCAFE_F00D]));
    assert_eq!(request.object_ids, [7]);

    // The same bytes on a plain session carry no domain header, so word 4 is where the SFCI
    // magic should be.
    assert!(matches!(
        Message::parse_request(&bytes, SessionKind::Session),
        Err(IpcError::BadMagic { .. })
    ));
}

/// Domain request closing a virtual handle: command 2, no SFCI header, no data.
#[test]
fn domain_close_virtual_handle_has_no_sfci() {
    let bytes = buffer(&[
        0x0000_0004,
        0x0000_0006,
        0,
        0,
        0x0000_0002,
        0x0000_0003,
        0,
        0,
    ]);
    let message = Message::parse_request(&bytes, SessionKind::Domain).unwrap();
    assert_eq!(
        message.body,
        Body::DomainClose(DomainRequestHeader {
            command: DomainCommand::CloseVirtualHandle,
            input_object_count: 0,
            size: 0,
            object_id: 3,
            token: 0,
        })
    );
}

/// TIPC request, command id 3: the type is `16 + 3`; the raw data is the arguments, with no SFCI
/// header and no alignment padding.
///
/// ```text
/// word 0  0x0000_0013  type 19 = 16 + 3
/// word 1  0x0000_0002  2 raw words
/// word 2  0xAAAA_0001  argument
/// word 3  0xBBBB_0002  argument
/// ```
#[test]
fn tipc_request() {
    let bytes = buffer(&[0x0000_0013, 0x0000_0002, 0xAAAA_0001, 0xBBBB_0002]);
    let message = Message::parse_request(&bytes, SessionKind::Session).unwrap();
    assert_eq!(message.command_type, CommandType::Tipc(3));
    assert_eq!(
        message.body,
        Body::TipcRequest {
            data: &le(&[0xAAAA_0001, 0xBBBB_0002])
        }
    );
}

/// TIPC reply carrying data. The result is ONE word, and the data starts at the very next word.
/// Eden wrote a two-word result here, which put the data one word late (scoping doc row 22).
///
/// ```text
/// word 0  0x0000_0013  type 19, echoing the request
/// word 1  0x0000_0003  3 raw words: result + 2 data words
/// word 2  0x0000_0000  result (success)
/// word 3  0x1111_1111  data word 0   <- directly after the result word
/// word 4  0x2222_2222  data word 1
/// ```
#[test]
fn tipc_reply_data_follows_a_one_word_result() {
    let words = [
        0x0000_0013,
        0x0000_0003,
        0x0000_0000,
        0x1111_1111,
        0x2222_2222,
    ];
    let bytes = buffer(&words);
    let message = Message::parse_reply(&bytes, SessionKind::Session).unwrap();
    assert_eq!(message.command_type, CommandType::Tipc(3));
    assert_eq!(
        message.body,
        Body::TipcReply {
            result: 0,
            data: &le(&[0x1111_1111, 0x2222_2222])
        }
    );

    // The builder lays out exactly the same words: data in word 3, not word 4.
    let mut built = [0xFF_u8; BUFFER_SIZE];
    let used = ReplyBuilder::tipc(3, 0)
        .data(&le(&[0x1111_1111, 0x2222_2222]))
        .build(&mut built)
        .unwrap();
    assert_eq!(used, 20);
    assert_eq!(built.as_slice(), bytes.as_slice());
}

/// TIPC reply that also copies a handle (the shape of a session-returning gsv reply): special
/// header, handle, then the result word.
#[test]
fn tipc_reply_with_copied_handle_and_error_result() {
    let words = [
        0x0000_0013,
        0x8000_0001,
        0x0000_0002,
        0x0000_1234,
        0x0000_0E01,
    ];
    let bytes = buffer(&words);
    let message = Message::parse_reply(&bytes, SessionKind::Session).unwrap();
    assert_eq!(message.copy_handles, [0x1234]);
    assert_eq!(
        message.body,
        Body::TipcReply {
            result: 0x0E01,
            data: &[]
        }
    );

    let mut built = [0_u8; BUFFER_SIZE];
    ReplyBuilder::tipc(3, 0x0E01)
        .copy_handle(0x1234)
        .build(&mut built)
        .unwrap();
    assert_eq!(built.as_slice(), bytes.as_slice());
}

/// CMIF reply on a plain session: type zero, padding to word 4, `SFCO`, result, one data word.
#[test]
fn cmif_reply_layout() {
    let words = [
        0x0000_0000,
        0x0000_0007,
        0,
        0,
        SFCO_MAGIC,
        0,
        0x0000_0E01,
        0,
        0xDEAD_BEEF,
    ];
    let bytes = buffer(&words);
    let message = Message::parse_reply(&bytes, SessionKind::Session).unwrap();
    assert_eq!(message.command_type, CommandType::Invalid);
    assert_eq!(
        message.body,
        Body::CmifReply(starquake_ipc::CmifReply {
            domain: None,
            header: CmifReplyHeader {
                version: 0,
                result: 0x0E01,
                token: 0
            },
            data: &le(&[0xDEAD_BEEF]),
            object_ids: Vec::new(),
        })
    );

    let mut built = [0x55_u8; BUFFER_SIZE];
    ReplyBuilder::cmif(0x0E01)
        .data(&le(&[0xDEAD_BEEF]))
        .build(&mut built)
        .unwrap();
    assert_eq!(built.as_slice(), bytes.as_slice());
}

/// CMIF reply from a domain: domain reply header (object count), `SFCO`, data, then object ids.
#[test]
fn domain_reply_layout() {
    let words = [
        0x0000_0000,
        0x0000_000C,
        0,
        0,
        0x0000_0001,
        0,
        0,
        0,
        SFCO_MAGIC,
        0,
        0,
        0,
        0xDEAD_BEEF,
        0x0000_0009,
    ];
    let bytes = buffer(&words);
    let message = Message::parse_reply(&bytes, SessionKind::Domain).unwrap();
    assert_eq!(
        message.body,
        Body::CmifReply(starquake_ipc::CmifReply {
            domain: Some(DomainReplyHeader { num_objects: 1 }),
            header: CmifReplyHeader {
                version: 0,
                result: 0,
                token: 0
            },
            data: &le(&[0xDEAD_BEEF]),
            object_ids: vec![9],
        })
    );

    let mut built = [0x55_u8; BUFFER_SIZE];
    ReplyBuilder::domain(0)
        .data(&le(&[0xDEAD_BEEF]))
        .object_id(9)
        .build(&mut built)
        .unwrap();
    assert_eq!(built.as_slice(), bytes.as_slice());
}

/// A reply with moved and copied handles puts copies before moves, then pads to 16 bytes.
#[test]
fn cmif_reply_with_handles() {
    // header(2) + special(1) + copy(1) + move(1) = 5 words, padded to 8: pad 3, SFCO 4, no data.
    let words = [
        0x0000_0000,
        0x8000_0007,
        0x0000_0022,
        0x0000_00C0,
        0x0000_00D0,
        0,
        0,
        0,
        SFCO_MAGIC,
        0,
        0,
        0,
    ];
    let mut built = [0_u8; BUFFER_SIZE];
    ReplyBuilder::cmif(0)
        .copy_handle(0xC0)
        .move_handle(0xD0)
        .build(&mut built)
        .unwrap();
    assert_eq!(built.as_slice(), buffer(&words).as_slice());
    let message = Message::parse_reply(&built, SessionKind::Session).unwrap();
    assert_eq!(message.copy_handles, [0xC0]);
    assert_eq!(message.move_handles, [0xD0]);
}

#[test]
fn close_commands_only_need_the_header() {
    for ty in [2_u32, 15] {
        let bytes = buffer(&[ty, 0]);
        let message = Message::parse_request(&bytes[..8], SessionKind::Domain).unwrap();
        assert_eq!(message.body, Body::Close);
        assert!(message.command_type.is_close());
    }
}

#[test]
fn counts_that_overrun_the_buffer_are_errors() {
    // 15 A descriptors announced, nothing after the header.
    let bytes = buffer(&[0x00F0_0004, 0])[..8].to_vec();
    assert!(matches!(
        Message::parse_request(&bytes, SessionKind::Session),
        Err(IpcError::Truncated { .. })
    ));
    // Raw data size of 0x3FF words cannot fit the 0x100-byte buffer.
    let bytes = buffer(&[0x0000_0004, 0x0000_03FF]);
    assert!(matches!(
        Message::parse_request(&bytes, SessionKind::Session),
        Err(IpcError::Truncated { .. })
    ));
    // The special header word is announced but the buffer ends after the two header words.
    let bytes = [4_u8, 0, 0, 0, 0, 0, 0, 0x80];
    assert!(matches!(
        Message::parse_request(&bytes, SessionKind::Session),
        Err(IpcError::Truncated { .. })
    ));
    // Empty and tiny buffers.
    for len in 0..8 {
        assert!(Message::parse_request(&vec![0; len], SessionKind::Session).is_err());
    }
}

#[test]
fn bad_magic_and_bad_types_are_errors() {
    let mut words = cmif_request_with_pid_handle_and_a_descriptor();
    words[12] = SFCO_MAGIC;
    assert_eq!(
        Message::parse_request(&buffer(&words), SessionKind::Session),
        Err(IpcError::BadMagic {
            found: SFCO_MAGIC,
            expected: SFCI_MAGIC
        })
    );

    // A reply whose payload opens with SFCI is not a reply.
    let reply = [0, 7, 0, 0, SFCI_MAGIC, 0, 0, 0, 0];
    assert!(matches!(
        Message::parse_reply(&buffer(&reply), SessionKind::Session),
        Err(IpcError::BadMagic { .. })
    ));

    // Types 8..=14 are undefined.
    for ty in 8_u32..=14 {
        assert_eq!(
            Message::parse_request(&buffer(&[ty, 0]), SessionKind::Session),
            Err(IpcError::UnknownCommandType(u16::try_from(ty).unwrap()))
        );
    }
    // Zero is a reply type, not a request type; a request type is not a CMIF reply type.
    assert_eq!(
        Message::parse_request(&buffer(&[0, 0]), SessionKind::Session),
        Err(IpcError::UnexpectedCommandType(CommandType::Invalid))
    );
    assert_eq!(
        Message::parse_reply(&buffer(&[4, 0]), SessionKind::Session),
        Err(IpcError::UnexpectedCommandType(CommandType::Request))
    );
}

#[test]
fn inconsistent_domain_sizes_are_errors() {
    // Domain size 8 is smaller than the SFCI header it must contain.
    let small = buffer(&[4, 10, 0, 0, 0x0008_0001, 3, 0, 0, SFCI_MAGIC, 0, 0, 0]);
    assert!(matches!(
        Message::parse_request(&small, SessionKind::Domain),
        Err(IpcError::Inconsistent(_))
    ));
    // Size 20 but the section only holds the SFCI header.
    let overrun = buffer(&[4, 10, 0, 0, 0x0014_0001, 3, 0, 0, SFCI_MAGIC, 0, 0, 0]);
    assert!(matches!(
        Message::parse_request(&overrun, SessionKind::Domain),
        Err(IpcError::Inconsistent(_))
    ));
    // 255 input objects cannot fit.
    let ids = buffer(&[4, 10, 0, 0, 0x0010_FF01, 3, 0, 0, SFCI_MAGIC, 0, 0, 0]);
    assert!(matches!(
        Message::parse_request(&ids, SessionKind::Domain),
        Err(IpcError::Inconsistent(_))
    ));
    // Unknown domain command.
    let unknown = buffer(&[4, 6, 0, 0, 0x0000_0009, 3, 0, 0]);
    assert_eq!(
        Message::parse_request(&unknown, SessionKind::Domain),
        Err(IpcError::UnknownDomainCommand(9))
    );
    // A domain reply that claims more objects than the data section holds.
    let reply = buffer(&[0, 10, 0, 0, 0x0000_0020, 0, 0, 0, SFCO_MAGIC, 0, 0, 0, 0]);
    assert!(matches!(
        Message::parse_reply(&reply, SessionKind::Domain),
        Err(IpcError::Inconsistent(_))
    ));
    // A TIPC reply must hold at least the result word.
    assert!(matches!(
        Message::parse_reply(&buffer(&[0x13, 0]), SessionKind::Session),
        Err(IpcError::Truncated { .. })
    ));
}

#[test]
fn receive_descriptors_follow_the_raw_data() {
    // Type 4 with receive-list flag 3 (bits 10..14 of word 1: flag - 2 = one descriptor) and 6 raw
    // words (2 padding + a bare SFCI header); the C descriptor sits right after them, at word 8.
    let words = [
        0x0000_0004,
        0x0000_0C06,
        0,
        0,
        SFCI_MAGIC,
        0,
        1,
        0,
        0x4000_0000,
        0x0040_0001,
    ];
    let bytes = buffer(&words);
    let message = Message::parse_request(&bytes, SessionKind::Session).unwrap();
    assert_eq!(message.receive_list, starquake_ipc::ReceiveList::Many(1));
    assert_eq!(
        message.receive_descriptors,
        [starquake_ipc::ReceiveDescriptor {
            address: 0x1_4000_0000,
            size: 0x40
        }]
    );
}
