//! Golden bytes: requests and replies byte for byte as libnx writes and reads them.
//!
//! Every expected buffer below was derived by hand from the inline functions in
//! `libnx/nx/include/switch/sf/hipc.h`, `cmif.h` and `tipc.h` (libnx `b156b4774fb2`, ISC, pinned
//! in `docs/references.md`). Each word cites the lines that write it (requests) or read it
//! (replies). The words are literals on purpose: they are the reference's numbers, so a wrong
//! shipping constant (a magic, a bit position, a shift) fails here instead of moving both sides
//! of the comparison together.
//!
//! libnx never clears its TLS buffer: the alignment padding, the slack inside its raw-size reserve
//! and the pid slot keep whatever was there before. The expected request buffers are what libnx
//! leaves in a zeroed buffer, which is also what [`RequestBuilder::build`] leaves: it zeroes the
//! whole usable area, and the pid slot holds the value passed to `send_pid` (zero here).
//!
//! libnx reads no reply's command type (`hipcParseResponse`, `hipc.h:316-363`, uses only the
//! counts, the size and the special header). The reply type words below are starquake's own
//! convention, documented in `reply.rs`: zero for CMIF, the request's type echoed for TIPC.
//!
//! After the golden buffers come the listed divergences from libnx, the edges of the new bounds,
//! and generated build-then-parse round trips that restate libnx's arithmetic independently.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_ipc::{
    BUFFER_SIZE, Body, BufferDescriptor, CMIF_HEADER_BYTES, CmifReply, CmifReplyHeader,
    CmifRequest, CommandHeader, CommandType, DOMAIN_HEADER_BYTES, DomainReplyHeader, IpcError,
    MAX_RAW_WORDS, Message, RAW_ALIGNMENT_WORDS, ReceiveDescriptor, ReplyBuilder, RequestBuilder,
    RequestLayout, SFCI_MAGIC, SFCO_MAGIC, SessionKind, StaticDescriptor,
};

const WORD: usize = size_of::<u32>();

/// `words` as little-endian bytes, zero-filled to a whole buffer.
fn golden(words: &[u32]) -> Vec<u8> {
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.resize(BUFFER_SIZE, 0);
    bytes
}

/// What a builder leaves in a buffer that was full of stale bytes.
fn request_bytes(builder: &RequestBuilder) -> Vec<u8> {
    let mut buf = vec![0xA5_u8; BUFFER_SIZE];
    let result = builder.build(&mut buf);
    assert!(result.is_ok(), "{result:?}");
    buf
}

fn reply_bytes(builder: &ReplyBuilder) -> Vec<u8> {
    let mut buf = vec![0xA5_u8; BUFFER_SIZE];
    let result = builder.build(&mut buf);
    assert!(result.is_ok(), "{result:?}");
    buf
}

/// Byte `index` of the golden words, for checking a parsed slice against the place libnx reads.
fn golden_slice(words: &[u32], start: usize, len: usize) -> Vec<u8> {
    golden(words)[start..start + len].to_vec()
}

fn cmif_request(message: Message<'_>) -> Option<CmifRequest<'_>> {
    match message.body {
        Body::CmifRequest(request) => Some(request),
        _ => None,
    }
}

fn cmif_reply(message: Message<'_>) -> Option<CmifReply<'_>> {
    match message.body {
        Body::CmifReply(reply) => Some(reply),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------------------------

const PLAIN_ARGS: [u8; 6] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66];

/// `cmifMakeRequest` (`cmif.h:100-157`) with `object_id = 0`, `request_id = 0x51`,
/// `context = 0x7A11`, `data_size = 6`, `send_pid = 1`, `num_handles = 1`; then
/// `cmifRequestHandle` (`cmif.h:255-258`) with handle `0x0001_D00D` and the six argument bytes
/// copied to `req.data`.
const PLAIN_REQUEST: [u32; 16] = [
    // hipc.h:234-245 (bit-fields hipc.h:29-40): type = RequestWithContext (cmif.h:115, cmif.h:21).
    0x0000_0006,
    // num_data_words: cmif.h:103 16; cmif.h:106 + 16 + 6 = 38; cmif.h:107 38; cmif.h:111
    // (38 + 3) / 4 = 10. has_special_header (hipc.h:231) is bit 31 (hipc.h:39).
    0x8000_000A,
    // hipc.h:251-255 (hipc.h:42-47): send_pid bit 0, one copy handle in bits 1..5.
    0x0000_0003,
    // hipc.h:256-257: the pid slot is skipped, not written.
    0x0000_0000,
    0x0000_0000,
    // hipc.h:164-166: copy handles come first; cmif.h:257 writes the handle.
    0x0001_D00D,
    // cmif.h:94-98: data_words is byte 24, so the content starts at (24 + 15) & ~15 = 32.
    0x0000_0000,
    0x0000_0000,
    // cmif.h:145-150: magic (cmif.h:11), version 1 (a context), command id, token = context
    // (object_id is zero).
    0x4943_4653,
    0x0000_0001,
    0x0000_0051,
    0x0000_7A11,
    // cmif.h:152: the arguments follow the header.
    0x4433_2211,
    0x0000_6655,
    // The raw data section is words 6..16; the 16-byte reserve leaves these two unused.
    0x0000_0000,
    0x0000_0000,
];

#[test]
fn plain_request_is_byte_exact() {
    let builder = RequestBuilder::cmif(CommandType::RequestWithContext, 0x51)
        .layout(RequestLayout::Libnx)
        .version(1)
        .token(0x7A11)
        .send_pid(0)
        .copy_handle(0x0001_D00D)
        .data(&PLAIN_ARGS);
    let bytes = request_bytes(&builder);
    assert_eq!(bytes, golden(&PLAIN_REQUEST));

    let message = Message::parse_request(&bytes, SessionKind::Session).unwrap();
    assert_eq!(message.copy_handles, [0x0001_D00D]);
    let request = cmif_request(message).unwrap();
    // A plain request's data runs to the end of the raw data section: the arguments, then the
    // reserve's slack.
    assert!(request.data.starts_with(&PLAIN_ARGS));
    assert_eq!(
        request.data,
        golden_slice(&PLAIN_REQUEST, 12 * WORD, 4 * WORD)
    );
}

const DOMAIN_ARGS: [u8; 5] = [0xA1, 0xA2, 0xA3, 0xA4, 0xA5];
const DOMAIN_IN_OBJECTS: [u32; 2] = [0xFEED_0001, 0xFEED_0002];

/// `cmifMakeRequest` (`cmif.h:100-157`) with `object_id = 0x2A`, `request_id = 0x0E`,
/// `context = 0xC7C7`, `data_size = 5`, `num_objects = 2`; then `cmifRequestObject`
/// (`cmif.h:250-253`) twice and the five argument bytes.
const DOMAIN_REQUEST: [u32; 18] = [
    // cmif.h:115: a context makes the type RequestWithContext.
    0x0000_0006,
    // num_data_words: cmif.h:103 16; cmif.h:105 + 16 + 2 * 4 = 40; cmif.h:106 + 16 + 5 = 61;
    // cmif.h:107 62; cmif.h:111 (62 + 3) / 4 = 16. No special header.
    0x0000_0010,
    // cmif.h:94-98: data_words is byte 8; the content starts at byte 16.
    0x0000_0000,
    0x0000_0000,
    // cmif.h:132-139 (cmif.h:45-52): type SendMessage (cmif.h:27), num_in_objects 2,
    // data_size = payload_size = 16 + 5 = 21 (cmif.h:131).
    0x0015_0201,
    0x0000_002A,
    // padding
    0x0000_0000,
    // token = context
    0x0000_C7C7,
    // cmif.h:140, cmif.h:145-150: version 1, token 0 because object_id is set (cmif.h:149).
    0x4943_4653,
    0x0000_0001,
    0x0000_000E,
    0x0000_0000,
    // Arguments A1..A4.
    0xA4A3_A2A1,
    // Argument A5, then the objects at hdr + payload_size (cmif.h:141), byte 53, unaligned:
    // 0xFEED_0001 is 01 00 ED FE, 0xFEED_0002 is 02 00 ED FE.
    0xED00_01A5,
    0xED00_02FE,
    0x0000_00FE,
    // The raw data section is words 2..18: two words of slack.
    0x0000_0000,
    0x0000_0000,
];

#[test]
fn domain_request_is_byte_exact() {
    let builder = RequestBuilder::domain(CommandType::RequestWithContext, 0x2A, 0x0E)
        .layout(RequestLayout::Libnx)
        .version(1)
        .domain_token(0xC7C7)
        .data(&DOMAIN_ARGS)
        .object_id(DOMAIN_IN_OBJECTS[0])
        .object_id(DOMAIN_IN_OBJECTS[1]);
    let bytes = request_bytes(&builder);
    assert_eq!(bytes, golden(&DOMAIN_REQUEST));

    let message = Message::parse_request(&bytes, SessionKind::Domain).unwrap();
    let request = cmif_request(message).unwrap();
    assert_eq!(request.data, DOMAIN_ARGS);
    assert_eq!(request.object_ids, DOMAIN_IN_OBJECTS);
    assert_eq!(
        request.domain.map(|d| (d.object_id, d.token)),
        Some((0x2A, 0xC7C7))
    );
}

const POINTER_ARGS: u32 = 0x1357_9BDF;
const IN_POINTERS: [StaticDescriptor; 2] = [
    StaticDescriptor {
        counter: 0,
        address: 0x12_3456_7890,
        size: 0x0040,
    },
    StaticDescriptor {
        counter: 1,
        address: 0x23_4567_8900,
        size: 0x0080,
    },
];
const IN_BUFFER: BufferDescriptor = BufferDescriptor {
    address: 0x45_6789_ABC0,
    size: 0x3_0000_1000,
    flags: 1,
};
const OUT_POINTER: ReceiveDescriptor = ReceiveDescriptor {
    address: 0x67_89AB_CDE0,
    size: 0x0100,
};
const OUT_FIXED_POINTER: ReceiveDescriptor = ReceiveDescriptor {
    address: 0x78_9ABC_DEF0,
    size: 0x0200,
};

/// `cmifMakeRequest` (`cmif.h:100-157`) with `request_id = 0x77`, no context, `data_size = 4`,
/// `num_in_pointers = 2`, `num_in_buffers = 1`, `num_out_pointers = 1`,
/// `num_out_fixed_pointers = 1`; then, in this order, `cmifRequestInPointer` twice
/// (`cmif.h:210-214`), `cmifRequestInBuffer` with `HipcBufferMode_NonSecure` (`cmif.h:195-198`),
/// `cmifRequestOutPointer` (`cmif.h:222-226`) and `cmifRequestOutFixedPointer`
/// (`cmif.h:216-220`).
const POINTER_REQUEST: [u32; 23] = [
    // hipc.h:234-245: type Request (cmif.h:115), two X (cmif.h:116), one A (cmif.h:117).
    0x0012_0004,
    // num_data_words: cmif.h:103 16; cmif.h:106 + 16 + 4 = 36; cmif.h:107 36, the table offset
    // (cmif.h:108); cmif.h:109-110 + 2 * 1 = 38; cmif.h:111 (38 + 3) / 4 = 10.
    // recv_static_mode: num_recv_statics = 1 + 1 (cmif.h:121), so 2 + 2 = 4 (hipc.h:241), bits
    // 10..14.
    0x0000_100A,
    // hipc.h:108-117, index cur_in_ptr_id = 0 (cmif.h:212): index bits 0..6,
    // address >> 36 = 1 in bits 6..12, address >> 32 = 2 in bits 12..16, size 0x40 in 16..32.
    0x0040_2040,
    0x3456_7890,
    // The second in pointer: index 1, address >> 36 = 2, address >> 32 = 3, size 0x80.
    0x0080_3081,
    0x4567_8900,
    // hipc.h:119-129: size low, address low, then mode 1 (hipc.h:103) in bits 0..2,
    // address >> 36 = 4 in 2..24, size >> 32 = 3 in 24..28, address >> 32 = 5 in 28..32.
    0x0000_1000,
    0x6789_ABC0,
    0x5300_0011,
    // cmif.h:94-98: data_words is byte 36; the content starts at (36 + 15) & ~15 = 48.
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    // cmif.h:145-150: version 0 and token 0 without a context.
    0x4943_4653,
    0x0000_0000,
    0x0000_0077,
    0x0000_0000,
    // The arguments.
    0x1357_9BDF,
    // Between the arguments and the table: the table sits 36 bytes into the raw data section
    // (cmif.h:108, cmif.h:153), that is byte 72, word 18.
    0x0000_0000,
    // cmif.h:225: the out pointer's size, a u16, in the table's first entry.
    0x0000_0100,
    // hipc.h:211-214: the receive list follows the raw data section. hipc.h:131-138:
    // address low, then address >> 32 in bits 0..16 and the size in bits 16..32.
    0x89AB_CDE0,
    0x0100_0067,
    // The fixed pointer: a receive list entry and no table entry (cmif.h:216-220).
    0x9ABC_DEF0,
    0x0200_0078,
];

fn pointer_request() -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, 0x77)
        .layout(RequestLayout::Libnx)
        .static_descriptor(IN_POINTERS[0])
        .static_descriptor(IN_POINTERS[1])
        .send_buffer(IN_BUFFER)
        .out_pointer(OUT_POINTER)
        .receive_descriptor(OUT_FIXED_POINTER)
        .data(&POINTER_ARGS.to_le_bytes())
}

#[test]
fn pointer_request_is_byte_exact() {
    let bytes = request_bytes(&pointer_request());
    assert_eq!(bytes, golden(&POINTER_REQUEST));

    let message = Message::parse_request(&bytes, SessionKind::Session).unwrap();
    assert_eq!(message.static_descriptors, IN_POINTERS);
    assert_eq!(message.send_buffers, [IN_BUFFER]);
    assert_eq!(
        message.receive_descriptors,
        [OUT_POINTER, OUT_FIXED_POINTER]
    );
    let request = cmif_request(message).unwrap();
    assert!(request.data.starts_with(&POINTER_ARGS.to_le_bytes()));
}

const TIPC_ARGS: [u8; 6] = [0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6];
const TIPC_IN_BUFFER: BufferDescriptor = BufferDescriptor {
    address: 0x34_0000_3000,
    size: 0x20,
    flags: 0,
};
const TIPC_OUT_BUFFER: BufferDescriptor = BufferDescriptor {
    address: 0x56_0000_4000,
    size: 0x30,
    flags: 3,
};

/// `tipcMakeRequest` (`tipc.h:127-172`) with `request_id = 3`, `data_size = 6`, `send_pid`,
/// one handle, an in buffer (`SfBufferAttr_HipcMapAlias | In`, mode Normal) and an out buffer
/// (`HipcMapAlias | Out | HipcMapTransferAllowsNonDevice`, mode `NonDevice`), then the arguments
/// copied by `tipcDispatchImpl` (`tipc.h:242-243`).
const TIPC_REQUEST: [u32; 14] = [
    // tipc.h:132, tipc.h:147: type 3 + 16; tipc.h:149-150: one A and one B.
    0x0110_0013,
    // tipc.h:152: (6 + 3) / 4 = 2 words; a special header for the pid and the handle.
    0x8000_0002,
    // hipc.h:251-255: send_pid, one copy handle.
    0x0000_0003,
    // hipc.h:256-257: the unwritten pid slot.
    0x0000_0000,
    0x0000_0000,
    // tipc.h:159-160, tipc.h:84-87.
    0x0002_F00D,
    // hipc.h:119-129: size, address low, mode 0 | (address >> 36 = 3) << 2 |
    // (address >> 32 = 4) << 28.
    0x0000_0020,
    0x0000_3000,
    0x4000_000C,
    // The out buffer: mode 3 (hipc.h:105, tipc.h:115-116), address >> 36 = 5,
    // address >> 32 = 6.
    0x0000_0030,
    0x0000_4000,
    0x6000_0017,
    // tipc.h:171 returns data_words: no padding and no header before the arguments.
    0xB4B3_B2B1,
    0x0000_B6B5,
];

#[test]
fn tipc_request_is_byte_exact_in_both_layouts() {
    for layout in [RequestLayout::Exact, RequestLayout::Libnx] {
        let builder = RequestBuilder::tipc(3)
            .layout(layout)
            .send_pid(0)
            .copy_handle(0x0002_F00D)
            .send_buffer(TIPC_IN_BUFFER)
            .receive_buffer(TIPC_OUT_BUFFER)
            .data(&TIPC_ARGS);
        assert_eq!(request_bytes(&builder), golden(&TIPC_REQUEST), "{layout:?}");
    }
}

/// `cmifMakeControlRequest(base, 3, 0)` (`cmif.h:159-174`), as `cmifQueryPointerBufferSize`
/// sends it (`cmif.h:347`).
const CONTROL_REQUEST: [u32; 10] = [
    // cmif.h:163: type Control (cmif.h:20).
    0x0000_0005,
    // cmif.h:161: 16 + 16 + 0 = 32; cmif.h:164: (32 + 3) / 4 = 8.
    0x0000_0008,
    // cmif.h:166: content at byte 16.
    0x0000_0000,
    0x0000_0000,
    // cmif.h:167-172.
    0x4943_4653,
    0x0000_0000,
    0x0000_0003,
    0x0000_0000,
    // Slack: the raw data section is words 2..10.
    0x0000_0000,
    0x0000_0000,
];

#[test]
fn control_request_is_byte_exact() {
    let builder = RequestBuilder::cmif(CommandType::Control, 3).layout(RequestLayout::Libnx);
    assert_eq!(request_bytes(&builder), golden(&CONTROL_REQUEST));
}

/// `cmifMakeCloseRequest(base, 0x2A)` (`cmif.h:176-193`).
const DOMAIN_CLOSE_REQUEST: [u32; 10] = [
    // cmif.h:180: type Request.
    0x0000_0004,
    // cmif.h:181: (16 + 16) / 4 = 8.
    0x0000_0008,
    // cmif.h:183: content at byte 16.
    0x0000_0000,
    0x0000_0000,
    // cmif.h:184-187: type Close (cmif.h:28), object id; every other field zero.
    0x0000_0002,
    0x0000_002A,
    0x0000_0000,
    0x0000_0000,
    // Slack.
    0x0000_0000,
    0x0000_0000,
];

#[test]
fn domain_close_request_is_byte_exact() {
    let builder = RequestBuilder::domain_close(0x2A).layout(RequestLayout::Libnx);
    assert_eq!(request_bytes(&builder), golden(&DOMAIN_CLOSE_REQUEST));
}

/// The literal magics above are libnx's (`cmif.h:11-12`); the shipping constants must agree.
#[test]
fn golden_magics_are_the_shipping_constants() {
    assert_eq!(SFCI_MAGIC, PLAIN_REQUEST[8]);
    assert_eq!(SFCO_MAGIC, PLAIN_REPLY[8]);
}

// ---------------------------------------------------------------------------------------------
// Replies
// ---------------------------------------------------------------------------------------------

const PLAIN_REPLY_DATA: [u8; 6] = [0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6];

/// A successful reply to [`PLAIN_REQUEST`] with six data bytes, a copied and a moved handle.
const PLAIN_REPLY: [u32; 14] = [
    0x0000_0000,
    // Raw data: words 5..14, 9 words. hipc.h:330: a special header follows.
    0x8000_0009,
    // hipc.h:337-338: one copy and one move handle (hipc.h:44-45).
    0x0000_0022,
    // hipc.h:347-353: copies, then moves.
    0x0003_C0DE,
    0x0004_D0D0,
    // hipc.h:360, cmif.h:263: data_words is byte 20, the content starts at byte 32.
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    // cmif.h:273, cmif.h:275-278: magic (cmif.h:12) and a zero result.
    0x4F43_4653,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    // cmif.h:281: the data is hdr + 1, byte 48.
    0xD4D3_D2D1,
    0x0000_D6D5,
];

#[test]
fn plain_reply_is_byte_exact() {
    let builder = ReplyBuilder::cmif(0)
        .copy_handle(0x0003_C0DE)
        .move_handle(0x0004_D0D0)
        .data(&PLAIN_REPLY_DATA);
    let bytes = reply_bytes(&builder);
    assert_eq!(bytes, golden(&PLAIN_REPLY));

    let message =
        Message::parse_reply_sized(&bytes, SessionKind::Session, PLAIN_REPLY_DATA.len()).unwrap();
    assert_eq!(message.copy_handles, [0x0003_C0DE]);
    assert_eq!(message.move_handles, [0x0004_D0D0]);
    let reply = cmif_reply(message).unwrap();
    assert_eq!(reply.data, PLAIN_REPLY_DATA);
    assert_eq!(
        reply.data,
        golden_slice(&PLAIN_REPLY, 48, PLAIN_REPLY_DATA.len())
    );
    assert_eq!(reply.object_ids, []);
}

const DOMAIN_REPLY_DATA: [u8; 2] = [0xE1, 0xE2];
const DOMAIN_OUT_OBJECTS: [u32; 2] = [0x1111_0003, 0x2222_0004];

/// A successful domain reply with two data bytes (a `u16`, as `cmifQueryPointerBufferSize` reads
/// one, `cmif.h:351-353`) and two output objects.
const DOMAIN_REPLY: [u32; 15] = [
    0x0000_0000,
    // Raw data: words 2..15, 13 words.
    0x0000_000D,
    // cmif.h:263: data_words is byte 8, the content starts at byte 16.
    0x0000_0000,
    0x0000_0000,
    // cmif.h:268 (cmif.h:54-57): num_out_objects, three padding words.
    0x0000_0002,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    // cmif.h:269: the SFCO header follows the domain header.
    0x4F43_4653,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    // cmif.h:281: data at byte 48. cmif.h:270: the objects at hdr + 16 + size = byte 50,
    // unaligned: 0x1111_0003 is 03 00 11 11, 0x2222_0004 is 04 00 22 22.
    0x0003_E2E1,
    0x0004_1111,
    0x0000_2222,
];

#[test]
fn domain_reply_puts_objects_where_libnx_reads_them() {
    let builder = ReplyBuilder::domain(0)
        .data(&DOMAIN_REPLY_DATA)
        .object_id(DOMAIN_OUT_OBJECTS[0])
        .object_id(DOMAIN_OUT_OBJECTS[1]);
    let bytes = reply_bytes(&builder);
    assert_eq!(bytes, golden(&DOMAIN_REPLY));

    let message =
        Message::parse_reply_sized(&bytes, SessionKind::Domain, DOMAIN_REPLY_DATA.len()).unwrap();
    let reply = cmif_reply(message).unwrap();
    assert_eq!(reply.domain, Some(DomainReplyHeader { num_objects: 2 }));
    assert_eq!(reply.data, DOMAIN_REPLY_DATA);
    assert_eq!(reply.object_ids, DOMAIN_OUT_OBJECTS);
    // cmif.h:270, cmif.h:292: the first object is the word at byte 50.
    let first = golden_slice(&DOMAIN_REPLY, 50, WORD);
    assert_eq!(first, DOMAIN_OUT_OBJECTS[0].to_le_bytes());
}

/// A successful reply to [`POINTER_REQUEST`] with one data word.
const POINTER_REPLY: [u32; 9] = [
    0x0000_0000,
    // Raw data: words 2..9, 7 words.
    0x0000_0007,
    0x0000_0000,
    0x0000_0000,
    0x4F43_4653,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    // cmif.h:281: data at byte 32.
    0x0000_00C8,
];

#[test]
fn pointer_reply_is_byte_exact() {
    let data = 0x0000_00C8_u32.to_le_bytes();
    let bytes = reply_bytes(&ReplyBuilder::cmif(0).data(&data));
    assert_eq!(bytes, golden(&POINTER_REPLY));
    let message = Message::parse_reply_sized(&bytes, SessionKind::Session, WORD).unwrap();
    assert_eq!(cmif_reply(message).unwrap().data, data);
}

/// A failed reply: libnx returns the result before it looks at the data or the objects
/// (`cmif.h:277-278`; `R_FAILED` is `res != 0`, `result.h:12`), so the parser returns no data and
/// no objects whatever size the caller expected.
const FAILED_DOMAIN_REPLY: [u32; 12] = [
    0x0000_0000,
    0x0000_000A,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x4F43_4653,
    0x0000_0000,
    // The result.
    0x0000_D401,
    0x0000_0000,
];

#[test]
fn failed_replies_carry_no_data_or_objects() {
    let bytes = reply_bytes(&ReplyBuilder::domain(0xD401));
    assert_eq!(bytes, golden(&FAILED_DOMAIN_REPLY));
    for expected in [0, WORD, 64] {
        let message = Message::parse_reply_sized(&bytes, SessionKind::Domain, expected).unwrap();
        let reply = cmif_reply(message).unwrap();
        assert_eq!(
            reply.header,
            CmifReplyHeader {
                version: 0,
                result: 0xD401,
                token: 0
            }
        );
        assert_eq!(
            (reply.data, reply.object_ids.as_slice()),
            (&[][..], &[][..])
        );
    }
}

const TIPC_REPLY_DATA: [u8; 6] = [0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6];

/// A successful reply to [`TIPC_REQUEST`]: six data bytes, a copied handle and a moved handle
/// (an output object is a moved handle, `tipc.h:214-216`).
const TIPC_REPLY: [u32; 8] = [
    // The request's type echoed (reply.rs).
    0x0000_0013,
    // Raw data: the result and two data words.
    0x8000_0003,
    0x0000_0022,
    0x0006_1234,
    0x0005_ABCD,
    // tipc.h:207: the result is the first data word.
    0x0000_0000,
    // tipc.h:211-212: the data follows it at once.
    0xC4C3_C2C1,
    0x0000_C6C5,
];

#[test]
fn tipc_reply_is_byte_exact() {
    let builder = ReplyBuilder::tipc(3, 0)
        .copy_handle(0x0006_1234)
        .move_handle(0x0005_ABCD)
        .data(&TIPC_REPLY_DATA);
    let bytes = reply_bytes(&builder);
    assert_eq!(bytes, golden(&TIPC_REPLY));

    let message =
        Message::parse_reply_sized(&bytes, SessionKind::Session, TIPC_REPLY_DATA.len()).unwrap();
    assert_eq!(message.command_type, CommandType::Tipc(3));
    assert_eq!(message.move_handles, [0x0005_ABCD]);
    assert_eq!(
        message.body,
        Body::TipcReply {
            result: 0,
            data: &TIPC_REPLY_DATA,
        }
    );
}

/// tipc.h:208-209: a failed TIPC reply returns before the data is located.
#[test]
fn failed_tipc_replies_carry_no_data() {
    let bytes = reply_bytes(&ReplyBuilder::tipc(3, 0xD401));
    let message = Message::parse_reply_sized(&bytes, SessionKind::Session, WORD).unwrap();
    assert_eq!(
        message.body,
        Body::TipcReply {
            result: 0xD401,
            data: &[],
        }
    );
}

/// The sized parser needs the expected data and the object ids inside the raw data section;
/// libnx reads past it without a check (`cmif.h:270`, `cmif.h:281`, `tipc.h:211-212`). This is a
/// listed divergence: the edge is exact on both sides.
#[test]
fn sized_parse_refuses_data_or_objects_past_the_raw_data() {
    let domain = golden(&DOMAIN_REPLY);
    // The raw data section ends at byte 60: data at 48, two bytes of data and eight of objects
    // leave two bytes of slack.
    let room = 60 - 48 - DOMAIN_OUT_OBJECTS.len() * WORD;
    assert!(Message::parse_reply_sized(&domain, SessionKind::Domain, room).is_ok());
    assert_eq!(
        Message::parse_reply_sized(&domain, SessionKind::Domain, room + 1),
        Err(IpcError::Inconsistent(
            "domain object ids overrun the raw data"
        ))
    );
    assert_eq!(
        Message::parse_reply_sized(&domain, SessionKind::Domain, usize::MAX),
        Err(IpcError::Inconsistent("reply data overruns the raw data"))
    );
    // Data alone: the section ends at byte 36 and the data starts at byte 32, so four bytes fit
    // and five do not.
    let plain = golden(&POINTER_REPLY);
    assert!(Message::parse_reply_sized(&plain, SessionKind::Session, WORD).is_ok());
    for size in [WORD + 1, usize::MAX] {
        assert_eq!(
            Message::parse_reply_sized(&plain, SessionKind::Session, size),
            Err(IpcError::Inconsistent("reply data overruns the raw data")),
            "{size}"
        );
    }
    let tipc = golden(&TIPC_REPLY);
    let tipc_room = 2 * WORD;
    assert!(Message::parse_reply_sized(&tipc, SessionKind::Session, tipc_room).is_ok());
    for size in [tipc_room + 1, usize::MAX] {
        assert_eq!(
            Message::parse_reply_sized(&tipc, SessionKind::Session, size),
            Err(IpcError::Inconsistent("reply data overruns the raw data")),
            "{size}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Divergences and edges
// ---------------------------------------------------------------------------------------------

/// libnx never reads `num_out_objects`; it takes as many ids as the caller asks for
/// (`cmifResponseGetObject`, `cmif.h:290-293`). Here the header's count decides: a header that
/// counts one id yields one, although the section holds two. A listed divergence.
#[test]
fn domain_reply_object_count_comes_from_its_header() {
    let mut words = DOMAIN_REPLY;
    // Word 4 is `num_out_objects` (see `DOMAIN_REPLY`).
    words[4] = 1;
    let bytes = golden(&words);
    let message =
        Message::parse_reply_sized(&bytes, SessionKind::Domain, DOMAIN_REPLY_DATA.len()).unwrap();
    let reply = cmif_reply(message).unwrap();
    assert_eq!(reply.object_ids, [DOMAIN_OUT_OBJECTS[0]]);
    assert_eq!(reply.data, DOMAIN_REPLY_DATA);
}

/// The unsized parser takes the object ids from the end of the raw data section. After whole
/// words of data it splits a built reply as the sized parser does; after two bytes of data the
/// section ends in two bytes of padding and its split is off by those two bytes.
#[test]
fn unsized_parse_splits_domain_replies_only_after_whole_words() {
    let data = 0x0BAD_F00D_u32.to_le_bytes();
    let whole = reply_bytes(
        &ReplyBuilder::domain(0)
            .data(&data)
            .object_id(DOMAIN_OUT_OBJECTS[0]),
    );
    let unsized_split = cmif_reply(Message::parse_reply(&whole, SessionKind::Domain).unwrap());
    let sized =
        cmif_reply(Message::parse_reply_sized(&whole, SessionKind::Domain, data.len()).unwrap());
    assert_eq!(unsized_split, sized);

    let bytes = golden(&DOMAIN_REPLY);
    let unsized_split =
        cmif_reply(Message::parse_reply(&bytes, SessionKind::Domain).unwrap()).unwrap();
    // Words 12..15 of `DOMAIN_REPLY` hold the data, the ids and two bytes of padding.
    assert_eq!(
        unsized_split.data,
        golden_slice(&DOMAIN_REPLY, 12 * WORD, WORD)
    );
    assert_eq!(
        unsized_split.object_ids,
        [DOMAIN_REPLY[13], DOMAIN_REPLY[14]]
    );
}

/// The out-pointer size table exists only in the libnx layout and only in a request with an
/// `SFCI` header; anything else is refused with the exact reason, and the buffer is untouched.
#[test]
fn out_pointers_need_the_libnx_layout_and_an_sfci_header() {
    let exact_only = "the out-pointer size table is written only in the libnx layout";
    let no_sfci = "only a CMIF request with an SFCI header has an out-pointer size table";
    let cases = [
        (
            RequestBuilder::cmif(CommandType::Request, 1).out_pointer(OUT_POINTER),
            exact_only,
        ),
        (
            RequestBuilder::domain(CommandType::Request, 1, 1).out_pointer(OUT_POINTER),
            exact_only,
        ),
        (
            RequestBuilder::tipc(1)
                .layout(RequestLayout::Libnx)
                .out_pointer(OUT_POINTER),
            no_sfci,
        ),
        (
            RequestBuilder::domain_close(1)
                .layout(RequestLayout::Libnx)
                .out_pointer(OUT_POINTER),
            no_sfci,
        ),
        (
            RequestBuilder::close()
                .layout(RequestLayout::Libnx)
                .out_pointer(OUT_POINTER),
            "a close message carries nothing but its header",
        ),
    ];
    for (builder, reason) in cases {
        let mut buf = [0x5A_u8; BUFFER_SIZE];
        assert_eq!(
            builder.build(&mut buf),
            Err(IpcError::Inconsistent(reason)),
            "{builder:?}"
        );
        assert!(buf.iter().all(|b| *b == 0x5A), "{builder:?}");
    }
}

/// libnx's raw size (`cmif.h:103-111`) goes into the 10-bit field: arguments that make it
/// exactly [`MAX_RAW_WORDS`] words encode (and then do not fit the buffer), one byte more is
/// [`IpcError::FieldOutOfRange`].
#[test]
fn libnx_raw_size_is_checked_at_the_field_limit() {
    let max_words = usize::try_from(MAX_RAW_WORDS).unwrap();
    let reserve = RAW_ALIGNMENT_WORDS * WORD;
    let at_limit = max_words * WORD - reserve - CMIF_HEADER_BYTES;
    let request = |len: usize| {
        RequestBuilder::cmif(CommandType::Request, 1)
            .layout(RequestLayout::Libnx)
            .data(&vec![0x5A; len])
    };
    let mut buf = [0_u8; BUFFER_SIZE];
    assert_eq!(
        request(at_limit).build(&mut buf),
        Err(IpcError::DoesNotFit {
            needed: (CommandHeader::WORDS + max_words) * WORD,
            capacity: BUFFER_SIZE,
        })
    );
    assert_eq!(
        request(at_limit + 1).build(&mut buf),
        Err(IpcError::FieldOutOfRange {
            field: "raw data words",
            value: u64::from(MAX_RAW_WORDS) + 1,
        })
    );
}

/// libnx's X descriptor (`hipc.h:49-55`): index bits 0..6, address bits 36..42 in bits 6..12,
/// address bits 32..36 in bits 12..16. Eden's counter has bits 9..12 too, which libnx reads as
/// address bits 39..42. Below a counter of 64 and an address of `1 << 39` the two agree; a
/// counter bit 9 is libnx's address bit 39. A listed divergence.
#[test]
fn static_counter_high_bits_are_libnx_address_bits() {
    let libnx_view = |[w0, w1]: [u32; 2]| {
        let index = w0 & 0x3F;
        let address =
            u64::from(w1) | u64::from((w0 >> 12) & 0xF) << 32 | u64::from((w0 >> 6) & 0x3F) << 36;
        (index, address)
    };
    let top = (1_u64 << 39) - 1;
    let agrees = StaticDescriptor {
        counter: 0x3F,
        address: top,
        size: 0x10,
    };
    assert_eq!(libnx_view(agrees.encode().unwrap()), (0x3F, top));
    let eden_only = StaticDescriptor {
        counter: 1 << 9,
        address: 0,
        size: 0x10,
    };
    assert_eq!(libnx_view(eden_only.encode().unwrap()), (0, 1 << 39));
}

// ---------------------------------------------------------------------------------------------
// Generated round trips
// ---------------------------------------------------------------------------------------------

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 512,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x4C49_424E_5847_4F4C),
        failure_persistence: None,
        ..Config::default()
    }
}

fn receive_descriptor() -> impl Strategy<Value = ReceiveDescriptor> {
    (0_u64..1 << 48, any::<u16>()).prop_map(|(address, size)| ReceiveDescriptor { address, size })
}

/// `cmifMakeRequest`'s arithmetic (`cmif.h:103-111`), restated: returns the out-pointer table's
/// offset into the raw data section and the raw size in words.
fn libnx_sizes(domain_objects: Option<usize>, data: usize, out_pointers: usize) -> (usize, usize) {
    let mut size = RAW_ALIGNMENT_WORDS * WORD;
    if let Some(objects) = domain_objects {
        size += DOMAIN_HEADER_BYTES + objects * WORD;
    }
    size += CMIF_HEADER_BYTES + data;
    let table_offset = size.next_multiple_of(2);
    let words = (table_offset + 2 * out_pointers).div_ceil(WORD);
    (table_offset, words)
}

/// Where the raw data section starts: after the two header words and, with handles, the special
/// header and the handles.
fn raw_start(handles: usize) -> usize {
    let special = if handles == 0 { 0 } else { 1 + handles };
    (CommandHeader::WORDS + special) * WORD
}

fn le_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

proptest! {
    #![proptest_config(config())]

    /// Plain and domain requests in the libnx layout: the raw size, the domain `size`, the
    /// out-pointer table and the receive list are where `cmifMakeRequest` puts them, and the
    /// parser gives back every part in order.
    #[test]
    fn libnx_layout_matches_the_libnx_arithmetic(
        domain in any::<bool>(),
        object_id in 1_u32..,
        command_id in any::<u32>(),
        args in prop::collection::vec(any::<u8>(), 0..=24),
        ids in prop::collection::vec(any::<u32>(), 0..=4),
        handles in prop::collection::vec(any::<u32>(), 0..=3),
        outs in prop::collection::vec(receive_descriptor(), 0..=3),
        fixed in prop::collection::vec(receive_descriptor(), 0..=2),
    ) {
        let ids = if domain { ids } else { Vec::new() };
        let mut builder = if domain {
            RequestBuilder::domain(CommandType::Request, object_id, command_id)
        } else {
            RequestBuilder::cmif(CommandType::Request, command_id)
        }
        .layout(RequestLayout::Libnx)
        .data(&args);
        for id in &ids { builder = builder.object_id(*id); }
        for handle in &handles { builder = builder.copy_handle(*handle); }
        for out in &outs { builder = builder.out_pointer(*out); }
        for c in &fixed { builder = builder.receive_descriptor(*c); }
        let bytes = request_bytes(&builder);

        let (table_offset, raw_words) =
            libnx_sizes(domain.then_some(ids.len()), args.len(), outs.len());
        let start = raw_start(handles.len());
        for (i, out) in outs.iter().enumerate() {
            prop_assert_eq!(le_u16(&bytes, start + table_offset + 2 * i), out.size);
        }
        let w1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let header = CommandHeader::decode([0, w1]).unwrap();
        prop_assert_eq!(usize::try_from(header.raw_data_words).unwrap(), raw_words);

        let kind = if domain { SessionKind::Domain } else { SessionKind::Session };
        let message = Message::parse_request(&bytes, kind).unwrap();
        prop_assert_eq!(&message.copy_handles, &handles);
        let receive: Vec<_> = outs.iter().chain(&fixed).copied().collect();
        prop_assert_eq!(&message.receive_descriptors, &receive);
        let request = cmif_request(message).unwrap();
        prop_assert_eq!(request.header.command_id, command_id);
        if domain {
            prop_assert_eq!(request.data, args.as_slice());
            prop_assert_eq!(&request.object_ids, &ids);
            let size = request.domain.map(|d| usize::try_from(d.size).unwrap());
            prop_assert_eq!(size, Some(CMIF_HEADER_BYTES + args.len()));
            prop_assert_eq!(request.domain.map(|d| d.object_id), Some(object_id));
        } else {
            prop_assert!(request.data.starts_with(&args));
        }
    }

    /// Domain replies with any data length: the sized parser finds the data and the object ids
    /// at libnx's offsets (`cmif.h:270`, `cmif.h:281`), and gives back every part in order.
    #[test]
    fn domain_replies_round_trip_at_libnx_offsets(
        data in prop::collection::vec(any::<u8>(), 0..=24),
        ids in prop::collection::vec(any::<u32>(), 0..=4),
        copies in prop::collection::vec(any::<u32>(), 0..=2),
        moves in prop::collection::vec(any::<u32>(), 0..=2),
    ) {
        let mut builder = ReplyBuilder::domain(0).data(&data);
        for handle in &copies { builder = builder.copy_handle(*handle); }
        for handle in &moves { builder = builder.move_handle(*handle); }
        for id in &ids { builder = builder.object_id(*id); }
        let bytes = reply_bytes(&builder);

        // cmif.h:263 (cmif.h:94-98): the content starts at the next 16-byte boundary.
        let start = raw_start(copies.len() + moves.len())
            .next_multiple_of(RAW_ALIGNMENT_WORDS * WORD);
        let objects_at = start + DOMAIN_HEADER_BYTES + CMIF_HEADER_BYTES + data.len();
        for (i, id) in ids.iter().enumerate() {
            let at = objects_at + i * WORD;
            prop_assert_eq!(bytes[at..at + WORD].to_vec(), id.to_le_bytes().to_vec());
        }

        let message = Message::parse_reply_sized(&bytes, SessionKind::Domain, data.len()).unwrap();
        prop_assert_eq!(&message.copy_handles, &copies);
        prop_assert_eq!(&message.move_handles, &moves);
        let reply = cmif_reply(message).unwrap();
        prop_assert_eq!(reply.data, data.as_slice());
        prop_assert_eq!(&reply.object_ids, &ids);
        let count = u32::try_from(ids.len()).unwrap();
        prop_assert_eq!(reply.domain, Some(DomainReplyHeader { num_objects: count }));
    }

    /// TIPC replies with any data length: the sized parser returns exactly the expected data.
    #[test]
    fn tipc_replies_round_trip_sized(
        command_id in 0_u16..0x100,
        data in prop::collection::vec(any::<u8>(), 0..=24),
        moves in prop::collection::vec(any::<u32>(), 0..=2),
    ) {
        let mut builder = ReplyBuilder::tipc(command_id, 0).data(&data);
        for handle in &moves { builder = builder.move_handle(*handle); }
        let bytes = reply_bytes(&builder);
        let message = Message::parse_reply_sized(&bytes, SessionKind::Session, data.len()).unwrap();
        prop_assert_eq!(&message.move_handles, &moves);
        prop_assert_eq!(message.body, Body::TipcReply { result: 0, data: data.as_slice() });
    }
}
