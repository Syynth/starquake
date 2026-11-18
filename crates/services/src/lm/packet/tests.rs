//! The packet parser, the message assembler and the chunk parser, against hand-built bytes and
//! against bytes this file's encoder builds.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use serde::Serialize;

use super::*;

/// The header fields in the order the layout gives them, each with the offset it starts at and
/// its width (module docs, "The layout").
const HEADER_FIELDS: [(Field, usize, usize); 7] = [
    (Field::ProcessId, 0x00, 8),
    (Field::ThreadId, 0x08, 8),
    (Field::Flags, 0x10, 1),
    (Field::Padding, 0x11, 1),
    (Field::Severity, 0x12, 1),
    (Field::Verbosity, 0x13, 1),
    (Field::PayloadSize, 0x14, 4),
];

/// A header whose every field, and every byte of its multi-byte fields, differs from the
/// others.
const DISTINCT: Header = Header {
    process_id: 0x1122_3344_5566_7788,
    thread_id: 0x99AA_BBCC_DDEE_F0F1,
    flags: FLAG_HEAD | FLAG_TAIL,
    severity: 3,
    verbosity: 0x44,
    payload_size: 0x0102_0304,
};

/// The padding byte the encoder writes, distinct from every field's bytes.
const PADDING: u8 = 0x5A;

/// The bytes of `header`, in the layout's order.
fn encode_header(header: &Header) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(header.process_id.to_le_bytes());
    bytes.extend(header.thread_id.to_le_bytes());
    bytes.push(header.flags);
    bytes.push(PADDING);
    bytes.push(header.severity);
    bytes.push(header.verbosity);
    bytes.extend(header.payload_size.to_le_bytes());
    bytes
}

/// A packet: `header` with its payload size set to `payload`'s length, then `payload`.
fn packet(header: &Header, payload: &[u8]) -> Vec<u8> {
    let header = Header {
        payload_size: u32::try_from(payload.len()).unwrap(),
        ..*header
    };
    let mut bytes = encode_header(&header);
    bytes.extend(payload);
    bytes
}

/// `value` as unsigned LEB128.
fn leb128(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let byte = u8::try_from(value & u64::from(LEB128_VALUE_MASK)).unwrap();
        value >>= LEB128_SHIFT;
        if value == 0 {
            bytes.push(byte);
            return bytes;
        }
        bytes.push(byte | LEB128_CONTINUE);
    }
}

/// One chunk: its key, its size and its data.
fn chunk(key: ChunkKey, data: &[u8]) -> Vec<u8> {
    let mut bytes = leb128(key.raw());
    bytes.extend(leb128(u64::try_from(data.len()).unwrap()));
    bytes.extend(data);
    bytes
}

fn header_with(flags: u8) -> Header {
    Header { flags, ..DISTINCT }
}

fn truncated(field: Field, offset: usize, needed: u64, len: usize) -> PacketError {
    PacketError::Truncated {
        field,
        offset,
        needed,
        len,
    }
}

// The header.

#[test]
fn the_field_table_covers_the_header_exactly() {
    let mut next = 0;
    for (_, start, width) in HEADER_FIELDS {
        assert_eq!(start, next);
        next += width;
    }
    assert_eq!(next, HEADER_SIZE);
}

#[test]
fn a_hand_built_header_reads_every_field_from_its_place() {
    // Every field by hand, little-endian, each byte distinct; the padding byte is 0xEE, which
    // is read past and kept nowhere.
    let mut bytes = vec![
        0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // process id
        0xF1, 0xF0, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, // thread id
        0x07, // flags: head, tail and little-endian
        0xEE, // padding
        0x04, // severity: fatal
        0x09, // verbosity
        0x03, 0x00, 0x00, 0x00, // payload size: 3
    ];
    bytes.extend(b"abcdef");
    let (header, payload) = Header::parse(&bytes).unwrap();
    assert_eq!(
        header,
        Header {
            process_id: 0x1122_3344_5566_7788,
            thread_id: 0x99AA_BBCC_DDEE_F0F1,
            flags: FLAG_HEAD | FLAG_TAIL | FLAG_LITTLE_ENDIAN,
            severity: 4,
            verbosity: 9,
            payload_size: 3,
        }
    );
    assert_eq!(payload, b"abc");
    assert!(header.is_head() && header.is_tail());
    assert_eq!(Severity::from_raw(header.severity), Severity::Fatal);
}

#[test]
fn the_payload_is_the_size_the_header_gives() {
    let mut bytes = packet(&DISTINCT, b"exact");
    let (_, payload) = Header::parse(&bytes).unwrap();
    assert_eq!(payload, b"exact");
    // Bytes after the payload are ignored, where Eden would take them as more payload.
    bytes.extend(b"trailing");
    let (header, payload) = Header::parse(&bytes).unwrap();
    assert_eq!(payload, b"exact");
    assert_eq!(header.payload_size, 5);
}

#[test]
fn a_payload_shorter_than_its_size_is_refused() {
    let mut bytes = packet(&DISTINCT, b"exact");
    bytes.pop();
    assert_eq!(
        Header::parse(&bytes),
        Err(truncated(Field::Payload, HEADER_SIZE, 5, HEADER_SIZE + 4))
    );
    // Exactly at the size it is read.
    bytes.push(b't');
    assert!(Header::parse(&bytes).is_ok());
}

#[test]
fn a_packet_cut_inside_each_header_field_is_refused_at_that_field() {
    let whole = encode_header(&DISTINCT);
    for (field, start, width) in HEADER_FIELDS {
        for len in start..start + width {
            assert_eq!(
                Header::parse(&whole[..len]),
                Err(truncated(field, start, u64::try_from(width).unwrap(), len)),
                "cut at {len}"
            );
        }
    }
    // The whole header with no payload reaches the payload, which is too short.
    assert_eq!(
        Header::parse(&whole),
        Err(truncated(
            Field::Payload,
            HEADER_SIZE,
            u64::from(DISTINCT.payload_size),
            HEADER_SIZE
        ))
    );
}

#[test]
fn multi_byte_fields_are_read_whole() {
    // Values that differ only above their low byte read as different values.
    for (low, high) in [(0x01, 0x0100), (0x01, 0x0100_0000_0000_0001)] {
        let a = Header::parse(&packet(
            &Header {
                process_id: low,
                ..DISTINCT
            },
            b"",
        ))
        .unwrap()
        .0;
        let b = Header::parse(&packet(
            &Header {
                process_id: high,
                ..DISTINCT
            },
            b"",
        ))
        .unwrap()
        .0;
        assert_ne!(a.process_id, b.process_id);
        let a = Header::parse(&packet(
            &Header {
                thread_id: low,
                ..DISTINCT
            },
            b"",
        ))
        .unwrap()
        .0;
        let b = Header::parse(&packet(
            &Header {
                thread_id: high,
                ..DISTINCT
            },
            b"",
        ))
        .unwrap()
        .0;
        assert_ne!(a.thread_id, b.thread_id);
    }
    // A payload size of 0x100 is not a payload size of 0.
    let payload = vec![0; 0x100];
    let bytes = packet(&DISTINCT, &payload);
    let (header, read) = Header::parse(&bytes).unwrap();
    assert_eq!((header.payload_size, read.len()), (0x100, 0x100));
}

#[test]
fn the_flags_say_head_and_tail() {
    for (flags, head, tail) in [
        (0, false, false),
        (FLAG_HEAD, true, false),
        (FLAG_TAIL, false, true),
        (FLAG_HEAD | FLAG_TAIL, true, true),
        (FLAG_LITTLE_ENDIAN, false, false),
        (!(FLAG_HEAD | FLAG_TAIL), false, false),
    ] {
        let header = header_with(flags);
        assert_eq!(
            (header.is_head(), header.is_tail()),
            (head, tail),
            "{flags:#x}"
        );
    }
}

#[test]
fn the_message_key_is_every_field_but_the_flags_and_size() {
    let key = DISTINCT.key();
    assert_eq!(
        key,
        MessageKey {
            process_id: DISTINCT.process_id,
            thread_id: DISTINCT.thread_id,
            severity: DISTINCT.severity,
            verbosity: DISTINCT.verbosity,
        }
    );
    let other = Header {
        flags: 0,
        payload_size: 1,
        ..DISTINCT
    };
    assert_eq!(other.key(), key);
}

#[test]
fn severities_are_edens_and_any_other_byte_is_unknown() {
    let named = [
        (0, Severity::Trace, "TRACE"),
        (1, Severity::Info, "INFO"),
        (2, Severity::Warning, "WARNING"),
        (3, Severity::Error, "ERROR"),
        (4, Severity::Fatal, "FATAL"),
        (5, Severity::Other(5), "UNKNOWN"),
        (0xFF, Severity::Other(0xFF), "UNKNOWN"),
    ];
    for (raw, severity, name) in named {
        assert_eq!(Severity::from_raw(raw), severity);
        assert_eq!(severity.raw(), raw);
        assert_eq!(severity.name(), name);
    }
}

// Messages from packets.

#[test]
fn a_head_that_is_also_a_tail_is_a_whole_message() {
    let mut pending = Pending::new();
    let message = pending.push(&header_with(FLAG_HEAD | FLAG_TAIL), b"whole");
    assert_eq!(message, Ok(Some(b"whole".to_vec())));
    assert!(pending.is_empty());
}

#[test]
fn packets_build_a_message_in_order_until_the_tail() {
    let mut pending = Pending::new();
    assert_eq!(pending.push(&header_with(FLAG_HEAD), b"one "), Ok(None));
    assert_eq!(pending.push(&header_with(0), b"two "), Ok(None));
    assert_eq!(pending.message(&DISTINCT.key()), Some(&b"one two "[..]));
    assert_eq!(
        pending.push(&header_with(FLAG_TAIL), b"three"),
        Ok(Some(b"one two three".to_vec()))
    );
    assert!(pending.is_empty());
}

#[test]
fn messages_of_different_keys_are_kept_apart() {
    // Each key differs from the first in one field only, so a key that left one out would
    // mix two messages.
    let first = header_with(FLAG_HEAD);
    let others = [
        Header {
            process_id: 1,
            ..first
        },
        Header {
            thread_id: 2,
            ..first
        },
        Header {
            severity: 0,
            ..first
        },
        Header {
            verbosity: 0,
            ..first
        },
    ];
    let mut pending = Pending::new();
    pending.push(&first, b"first").unwrap();
    for (index, other) in others.iter().enumerate() {
        pending
            .push(other, format!("other {index}").as_bytes())
            .unwrap();
    }
    assert_eq!(pending.len(), 1 + others.len());
    let tail = |header: &Header| Header {
        flags: FLAG_TAIL,
        ..*header
    };
    for (index, other) in others.iter().enumerate().rev() {
        assert_eq!(
            pending.push(&tail(other), b"!"),
            Ok(Some(format!("other {index}!").into_bytes()))
        );
    }
    assert_eq!(
        pending.push(&tail(&first), b"!"),
        Ok(Some(b"first!".to_vec()))
    );
}

#[test]
fn a_packet_that_is_not_a_head_needs_a_message_in_progress() {
    let mut pending = Pending::new();
    assert_eq!(
        pending.push(&header_with(0), b"x"),
        Err(PacketError::NoHead)
    );
    assert_eq!(
        pending.push(&header_with(FLAG_TAIL), b"x"),
        Err(PacketError::NoHead)
    );
    assert!(pending.is_empty());
    // A message for another key is not this key's.
    pending
        .push(
            &Header {
                process_id: 1,
                ..header_with(FLAG_HEAD)
            },
            b"x",
        )
        .unwrap();
    assert_eq!(
        pending.push(&header_with(0), b"x"),
        Err(PacketError::NoHead)
    );
}

#[test]
fn a_head_begins_its_message_afresh() {
    // Eden's `insert_or_assign`.
    let mut pending = Pending::new();
    pending.push(&header_with(FLAG_HEAD), b"old").unwrap();
    pending.push(&header_with(FLAG_HEAD), b"new").unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(
        pending.push(&header_with(FLAG_TAIL), b"!"),
        Ok(Some(b"new!".to_vec()))
    );
    // A head that is also a tail drops the message in progress too.
    pending.push(&header_with(FLAG_HEAD), b"old").unwrap();
    assert_eq!(
        pending.push(&header_with(FLAG_HEAD | FLAG_TAIL), b"whole"),
        Ok(Some(b"whole".to_vec()))
    );
    assert!(pending.is_empty());
}

/// A head packet of the `index`th distinct key.
fn head_of(index: usize) -> Header {
    Header {
        process_id: u64::try_from(index).unwrap(),
        ..header_with(FLAG_HEAD)
    }
}

#[test]
fn messages_in_progress_are_capped() {
    let mut pending = Pending::new();
    for index in 0..MAX_IN_PROGRESS {
        assert_eq!(pending.push(&head_of(index), b"x"), Ok(None), "{index}");
    }
    assert_eq!(pending.len(), MAX_IN_PROGRESS);
    let before = pending.clone();
    // One more key is refused, and nothing changes.
    assert_eq!(
        pending.push(&head_of(MAX_IN_PROGRESS), b"x"),
        Err(PacketError::TooManyInProgress)
    );
    assert_eq!(pending, before);
    // A head for a key already in progress still begins it afresh.
    assert_eq!(pending.push(&head_of(0), b"again"), Ok(None));
    assert_eq!(pending.message(&head_of(0).key()), Some(&b"again"[..]));
    // A whole message of a new key needs no place, and is returned.
    let whole = Header {
        flags: FLAG_HEAD | FLAG_TAIL,
        ..head_of(MAX_IN_PROGRESS)
    };
    assert_eq!(pending.push(&whole, b"w"), Ok(Some(b"w".to_vec())));
    assert_eq!(pending.len(), MAX_IN_PROGRESS);
}

#[test]
fn a_message_is_capped_at_its_limit() {
    // A head of exactly the limit is kept; one byte more is refused and drops the message in
    // progress under its key.
    let mut pending = Pending::new();
    let full = vec![b'x'; MAX_MESSAGE_BYTES];
    assert_eq!(pending.push(&header_with(FLAG_HEAD), &full), Ok(None));
    let over = vec![b'x'; MAX_MESSAGE_BYTES + 1];
    assert_eq!(
        pending.push(&header_with(FLAG_HEAD), &over),
        Err(PacketError::MessageTooLarge {
            len: MAX_MESSAGE_BYTES + 1
        })
    );
    assert!(pending.is_empty());
    // So is a whole message one byte over.
    assert_eq!(
        pending.push(&header_with(FLAG_HEAD | FLAG_TAIL), &over),
        Err(PacketError::MessageTooLarge {
            len: MAX_MESSAGE_BYTES + 1
        })
    );

    // Packets that add up to exactly the limit make a message; one byte more drops it, and a
    // later packet of it has no message to add to.
    let half = MAX_MESSAGE_BYTES / 2;
    pending
        .push(&header_with(FLAG_HEAD), &full[..half])
        .unwrap();
    let message = pending
        .push(&header_with(FLAG_TAIL), &full[half..])
        .unwrap()
        .unwrap();
    assert_eq!(message.len(), MAX_MESSAGE_BYTES);
    pending
        .push(&header_with(FLAG_HEAD), &full[..half])
        .unwrap();
    assert_eq!(
        pending.push(&header_with(0), &full[half - 1..]),
        Err(PacketError::MessageTooLarge {
            len: MAX_MESSAGE_BYTES + 1
        })
    );
    assert!(pending.is_empty());
    assert_eq!(
        pending.push(&header_with(FLAG_TAIL), b""),
        Err(PacketError::NoHead)
    );
}

/// [`Pending`]'s saved shape, without its limits, to write states it refuses.
#[derive(Serialize)]
struct RawPending {
    messages: BTreeMap<MessageKey, Vec<u8>>,
}

fn restored(raw: &RawPending) -> Result<Pending, postcard::Error> {
    postcard::from_bytes(&postcard::to_allocvec(raw).unwrap())
}

#[test]
fn a_saved_pending_is_read_back_through_its_limits() {
    // What live packets leave is read back as it was.
    let mut pending = Pending::new();
    for index in 0..MAX_IN_PROGRESS {
        pending.push(&head_of(index), b"part").unwrap();
    }
    let bytes = postcard::to_allocvec(&pending).unwrap();
    let back: Pending = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(back, pending);

    // At the limits it is accepted; past either it is refused.
    let full = |count: usize, len: usize| RawPending {
        messages: (0..count)
            .map(|index| (head_of(index).key(), vec![0; len]))
            .collect(),
    };
    assert!(restored(&full(MAX_IN_PROGRESS, MAX_MESSAGE_BYTES)).is_ok());
    assert!(restored(&full(MAX_IN_PROGRESS + 1, 1)).is_err());
    assert!(restored(&full(1, MAX_MESSAGE_BYTES + 1)).is_err());
    // The checks themselves, by name.
    let state = |count: usize, len: usize| PendingState {
        messages: full(count, len).messages,
    };
    assert_eq!(
        Pending::try_from(state(MAX_IN_PROGRESS + 1, 1)),
        Err(PacketError::TooManyInProgress)
    );
    assert_eq!(
        Pending::try_from(state(1, MAX_MESSAGE_BYTES + 1)),
        Err(PacketError::MessageTooLarge {
            len: MAX_MESSAGE_BYTES + 1
        })
    );
    // Both faults at once: the count is checked first.
    assert_eq!(
        Pending::try_from(state(MAX_IN_PROGRESS + 1, MAX_MESSAGE_BYTES + 1)),
        Err(PacketError::TooManyInProgress)
    );
}

// Chunks.

#[test]
fn chunk_keys_are_edens() {
    let keys = [
        ChunkKey::SessionBegin,
        ChunkKey::SessionEnd,
        ChunkKey::TextLog,
        ChunkKey::LineNumber,
        ChunkKey::FileName,
        ChunkKey::FunctionName,
        ChunkKey::ModuleName,
        ChunkKey::ThreadName,
        ChunkKey::PacketDropCount,
        ChunkKey::UserSystemClock,
        ChunkKey::ProcessName,
    ];
    for (raw, key) in keys.into_iter().enumerate() {
        let raw = u64::try_from(raw).unwrap();
        assert_eq!(ChunkKey::from_raw(raw), key);
        assert_eq!(key.raw(), raw);
    }
    assert_eq!(ChunkKey::from_raw(11), ChunkKey::Other(11));
    assert_eq!(ChunkKey::Other(u64::MAX).raw(), u64::MAX);
}

#[test]
fn each_chunk_is_read_into_its_field_from_hand_built_bytes() {
    let bytes = [
        0x02, 0x02, b'h', b'i', // text
        0x03, 0x04, 0x78, 0x56, 0x34, 0x12, // line number
        0x04, 0x01, b'f', // file name
        0x05, 0x01, b'g', // function name
        0x06, 0x01, b'm', // module name
        0x07, 0x01, b't', // thread name
        0x08, 0x08, 1, 2, 3, 4, 5, 6, 7, 8, // packet drop count
        0x09, 0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, // user system clock
        0x0A, 0x01, b'p', // process name
    ];
    assert_eq!(
        LogRecord::parse(&bytes),
        Ok(LogRecord {
            text: Some(b"hi"),
            line_number: Some(0x1234_5678),
            file_name: Some(b"f"),
            function_name: Some(b"g"),
            module_name: Some(b"m"),
            thread_name: Some(b"t"),
            packet_drop_count: Some(0x0807_0605_0403_0201),
            user_system_clock: Some(i64::from_le_bytes([
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE
            ])),
            process_name: Some(b"p"),
        })
    );
}

#[test]
fn an_empty_message_holds_nothing() {
    assert_eq!(LogRecord::parse(&[]), Ok(LogRecord::default()));
}

#[test]
fn session_and_unknown_chunks_are_skipped_by_their_size() {
    // Each skipped chunk's data is a whole text chunk; read as chunks (Eden's reading for
    // these keys), it would set the text.
    let hidden = [0x02, 0x01, b'X'];
    for key in [0x00, 0x01, 0x0B, 0x7F] {
        let mut bytes = vec![key, 0x03];
        bytes.extend(hidden);
        bytes.extend([0x02, 0x01, b'Y']);
        let record = LogRecord::parse(&bytes).unwrap();
        assert_eq!(record.text, Some(&b"Y"[..]), "key {key:#x}");
        let mut alone = vec![key, 0x03];
        alone.extend(hidden);
        assert_eq!(LogRecord::parse(&alone), Ok(LogRecord::default()));
    }
}

#[test]
fn keys_and_sizes_of_several_bytes_are_read() {
    // A key of two bytes (2, written long), and a size of two bytes (0x80).
    let mut bytes = vec![0x82, 0x00, 0x80, 0x01];
    bytes.extend([b'z'; 0x80]);
    let record = LogRecord::parse(&bytes).unwrap();
    assert_eq!(record.text.map(<[u8]>::len), Some(0x80));
    // The longest number allowed: eight bytes.
    let mut longest = vec![0x82; MAX_LEB128_BYTES - 1];
    longest.push(0x00);
    assert_eq!(longest.len(), MAX_LEB128_BYTES);
    longest.extend([0x00]);
    // Key 2 | (2 << 7) | ..., an unknown key, with an empty chunk.
    assert_eq!(LogRecord::parse(&longest), Ok(LogRecord::default()));
}

#[test]
fn a_number_longer_than_eight_bytes_is_refused() {
    let too_long = vec![LEB128_CONTINUE; MAX_LEB128_BYTES];
    assert_eq!(
        LogRecord::parse(&[too_long.clone(), vec![0, 0]].concat()),
        Err(PacketError::NumberTooLong {
            field: Field::ChunkKey,
            offset: 0,
        })
    );
    let size = [vec![0x02], too_long, vec![0]].concat();
    assert_eq!(
        LogRecord::parse(&size),
        Err(PacketError::NumberTooLong {
            field: Field::ChunkSize,
            offset: 1,
        })
    );
}

/// `value`, which must fit in one byte's value bits, padded to `width` bytes: continuation
/// bytes whose value is 0 after the first (module docs, "Where this differs from Eden").
fn padded(value: u64, width: usize) -> Vec<u8> {
    let low = u8::try_from(value).unwrap();
    assert_eq!(low & LEB128_VALUE_MASK, low);
    let mut bytes = vec![low | LEB128_CONTINUE];
    bytes.extend(vec![LEB128_CONTINUE; width - 2]);
    bytes.push(0);
    assert_eq!(bytes.len(), width);
    bytes
}

#[test]
fn a_padded_number_is_read_up_to_eight_bytes() {
    let text = ChunkKey::TextLog.raw();
    // A key and a size each padded to the longest allowed: read as their values.
    let mut longest = padded(text, MAX_LEB128_BYTES);
    longest.extend(padded(3, MAX_LEB128_BYTES));
    longest.extend(b"abc");
    assert_eq!(LogRecord::parse(&longest).unwrap().text, Some(&b"abc"[..]));
    // One byte more of padding is refused, though the value is still small: the key first,
    // then the size after a key of the longest allowed width.
    let mut key = padded(text, MAX_LEB128_BYTES + 1);
    key.extend([3, b'a', b'b', b'c']);
    assert_eq!(
        LogRecord::parse(&key),
        Err(PacketError::NumberTooLong {
            field: Field::ChunkKey,
            offset: 0,
        })
    );
    let mut size = padded(text, MAX_LEB128_BYTES);
    size.extend(padded(3, MAX_LEB128_BYTES + 1));
    size.extend(b"abc");
    assert_eq!(
        LogRecord::parse(&size),
        Err(PacketError::NumberTooLong {
            field: Field::ChunkSize,
            offset: MAX_LEB128_BYTES,
        })
    );
}

#[test]
fn a_message_that_ends_inside_a_number_is_refused() {
    assert_eq!(
        LogRecord::parse(&[0x82]),
        Err(truncated(Field::ChunkKey, 1, 1, 1))
    );
    assert_eq!(
        LogRecord::parse(&[0x02]),
        Err(truncated(Field::ChunkSize, 1, 1, 1))
    );
    assert_eq!(
        LogRecord::parse(&[0x02, 0x81]),
        Err(truncated(Field::ChunkSize, 2, 1, 2))
    );
    // After a whole chunk, a key with nothing after it.
    assert_eq!(
        LogRecord::parse(&[0x02, 0x00, 0x02]),
        Err(truncated(Field::ChunkSize, 3, 1, 3))
    );
}

#[test]
fn a_chunk_past_the_end_is_refused() {
    assert_eq!(
        LogRecord::parse(&[0x02, 0x03, b'a', b'b']),
        Err(truncated(Field::ChunkData, 2, 3, 4))
    );
    // Exactly at the end it is read.
    assert_eq!(
        LogRecord::parse(&[0x02, 0x03, b'a', b'b', b'c'])
            .unwrap()
            .text,
        Some(&b"abc"[..])
    );
    // A size no message could hold.
    let huge = [vec![0x02], leb128(u64::MAX >> 8), vec![b'a']].concat();
    assert_eq!(
        LogRecord::parse(&huge),
        Err(truncated(
            Field::ChunkData,
            huge.len() - 1,
            u64::MAX >> 8,
            huge.len()
        ))
    );
}

#[test]
fn a_number_chunk_of_the_wrong_size_is_refused() {
    for (key, expected) in [
        (ChunkKey::LineNumber, U32_CHUNK_SIZE),
        (ChunkKey::PacketDropCount, U64_CHUNK_SIZE),
        (ChunkKey::UserSystemClock, U64_CHUNK_SIZE),
    ] {
        for size in [expected - 1, expected + 1, 0] {
            assert_eq!(
                LogRecord::parse(&chunk(key, &vec![1; size])),
                Err(PacketError::ChunkSize {
                    key,
                    size: u64::try_from(size).unwrap(),
                    expected,
                }),
                "{key:?} of {size}"
            );
        }
        assert!(LogRecord::parse(&chunk(key, &vec![1; expected])).is_ok());
    }
}

#[test]
fn a_later_chunk_replaces_an_earlier_one_and_an_empty_string_clears_it() {
    let bytes = [
        chunk(ChunkKey::TextLog, b"first"),
        chunk(ChunkKey::TextLog, b"second"),
        chunk(ChunkKey::FileName, b"file"),
        chunk(ChunkKey::FileName, b""),
        chunk(ChunkKey::LineNumber, &1u32.to_le_bytes()),
        chunk(ChunkKey::LineNumber, &2u32.to_le_bytes()),
    ]
    .concat();
    let record = LogRecord::parse(&bytes).unwrap();
    assert_eq!(record.text, Some(&b"second"[..]));
    assert_eq!(record.file_name, None);
    assert_eq!(record.line_number, Some(2));
}

#[test]
fn errors_read_as_plain_words() {
    let cases = [
        (
            truncated(Field::Severity, 18, 1, 18),
            "the log packet ends inside its severity: 1 bytes are needed at offset 18, and there \
             are 18"
                .to_owned(),
        ),
        (
            PacketError::NumberTooLong {
                field: Field::ChunkSize,
                offset: 3,
            },
            format!(
                "the log packet's chunk size at offset 3 is longer than {MAX_LEB128_BYTES} bytes"
            ),
        ),
        (
            PacketError::ChunkSize {
                key: ChunkKey::LineNumber,
                size: 3,
                expected: 4,
            },
            "a line number chunk of 3 bytes, where it takes 4".to_owned(),
        ),
        (
            PacketError::NoHead,
            "a log packet continues a message that nothing began".to_owned(),
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
    assert_eq!(
        PacketError::MessageTooLarge {
            len: MAX_MESSAGE_BYTES + 1
        }
        .to_string(),
        format!(
            "a log message of {} bytes is more than the {MAX_MESSAGE_BYTES} kept",
            MAX_MESSAGE_BYTES + 1
        )
    );
    assert_eq!(
        PacketError::TooManyInProgress.to_string(),
        format!("a log message begins while {MAX_IN_PROGRESS} are already in progress")
    );
    let names = [
        (Field::ProcessId, "process id"),
        (Field::ThreadId, "thread id"),
        (Field::Flags, "flags"),
        (Field::Padding, "padding"),
        (Field::Severity, "severity"),
        (Field::Verbosity, "verbosity"),
        (Field::PayloadSize, "payload size"),
        (Field::Payload, "payload"),
        (Field::ChunkKey, "chunk key"),
        (Field::ChunkSize, "chunk size"),
        (Field::ChunkData, "chunk data"),
    ];
    for (field, name) in names {
        assert_eq!(field.name(), name);
    }
}

// Generated round trips.

fn config() -> Config {
    Config {
        cases: 256,
        failure_persistence: None,
        rng_seed: RngSeed::Fixed(0x4C4D_5041_434B_4554),
        rng_algorithm: RngAlgorithm::ChaCha,
        ..Config::default()
    }
}

fn any_header() -> impl Strategy<Value = Header> {
    (
        any::<u64>(),
        any::<u64>(),
        any::<u8>(),
        any::<u8>(),
        any::<u8>(),
    )
        .prop_map(
            |(process_id, thread_id, flags, severity, verbosity)| Header {
                process_id,
                thread_id,
                flags,
                severity,
                verbosity,
                payload_size: 0,
            },
        )
}

/// A chunk key: the ones Eden names, and some it does not.
fn any_key() -> impl Strategy<Value = ChunkKey> {
    prop_oneof![
        (0u64..=10).prop_map(ChunkKey::from_raw),
        (11u64..=u64::MAX >> 8).prop_map(ChunkKey::Other),
    ]
}

/// The data a chunk of `key` may hold.
fn data_for(key: ChunkKey) -> BoxedStrategy<Vec<u8>> {
    match key {
        ChunkKey::LineNumber => any::<u32>().prop_map(|n| n.to_le_bytes().to_vec()).boxed(),
        ChunkKey::PacketDropCount | ChunkKey::UserSystemClock => {
            any::<u64>().prop_map(|n| n.to_le_bytes().to_vec()).boxed()
        }
        _ => proptest::collection::vec(any::<u8>(), 0..200).boxed(),
    }
}

fn any_chunks() -> impl Strategy<Value = Vec<(ChunkKey, Vec<u8>)>> {
    proptest::collection::vec(
        any_key().prop_flat_map(|key| data_for(key).prop_map(move |data| (key, data))),
        0..12,
    )
}

/// What [`LogRecord::parse`] should make of `chunks`: each key's last chunk, a string of no
/// bytes being none.
fn model(chunks: &[(ChunkKey, Vec<u8>)]) -> LogRecord<'_> {
    let mut record = LogRecord::default();
    for (key, data) in chunks {
        let string = (!data.is_empty()).then_some(data.as_slice());
        match key {
            ChunkKey::TextLog => record.text = string,
            ChunkKey::LineNumber => {
                record.line_number = Some(u32::from_le_bytes(data[..].try_into().unwrap()));
            }
            ChunkKey::FileName => record.file_name = string,
            ChunkKey::FunctionName => record.function_name = string,
            ChunkKey::ModuleName => record.module_name = string,
            ChunkKey::ThreadName => record.thread_name = string,
            ChunkKey::PacketDropCount => {
                record.packet_drop_count = Some(u64::from_le_bytes(data[..].try_into().unwrap()));
            }
            ChunkKey::UserSystemClock => {
                record.user_system_clock = Some(i64::from_le_bytes(data[..].try_into().unwrap()));
            }
            ChunkKey::ProcessName => record.process_name = string,
            ChunkKey::SessionBegin | ChunkKey::SessionEnd | ChunkKey::Other(_) => {}
        }
    }
    record
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn a_built_packet_parses_back(header in any_header(), payload in proptest::collection::vec(any::<u8>(), 0..64)) {
        let bytes = packet(&header, &payload);
        let (read, read_payload) = Header::parse(&bytes).unwrap();
        let expected = Header {
            payload_size: u32::try_from(payload.len()).unwrap(),
            ..header
        };
        prop_assert_eq!(read, expected);
        prop_assert_eq!(read_payload, &payload[..]);
    }

    #[test]
    fn built_chunks_parse_back(chunks in any_chunks()) {
        let bytes: Vec<u8> = chunks.iter().flat_map(|(key, data)| chunk(*key, data)).collect();
        prop_assert_eq!(LogRecord::parse(&bytes), Ok(model(&chunks)));
    }

    #[test]
    fn a_message_split_anywhere_is_put_back_together(
        message in proptest::collection::vec(any::<u8>(), 0..=256),
        cuts in proptest::collection::vec(any::<prop::sample::Index>(), 0..6),
    ) {
        let mut cuts: Vec<usize> = cuts.iter().map(|cut| cut.index(message.len() + 1)).collect();
        cuts.sort_unstable();
        let mut bounds = vec![0];
        bounds.extend(cuts);
        bounds.push(message.len());
        let parts: Vec<&[u8]> = bounds.windows(2).map(|w| &message[w[0]..w[1]]).collect();
        let mut pending = Pending::new();
        let last = parts.len() - 1;
        let mut out = None;
        for (index, part) in parts.iter().enumerate() {
            let mut flags = 0;
            if index == 0 {
                flags |= FLAG_HEAD;
            }
            if index == last {
                flags |= FLAG_TAIL;
            }
            out = pending.push(&header_with(flags), part).unwrap();
            prop_assert_eq!(out.is_some(), index == last);
        }
        prop_assert_eq!(out, Some(message));
        prop_assert!(pending.is_empty());
    }
}
