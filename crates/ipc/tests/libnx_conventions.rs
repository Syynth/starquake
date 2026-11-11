//! `RequestBuilder` against libnx, the client-side reference
//! (`libnx/nx/include/switch/sf/{hipc,cmif,tipc}.h`, pinned in `docs/references.md`).
//!
//! libnx is what a homebrew program uses to build requests, so these tests pin where the builder
//! agrees with it and where it deliberately differs. Each helper below restates the arithmetic of
//! one libnx function; the tests then check that the parser accepts the layout libnx writes and
//! that the builder's differences are the ones listed in the `request` module docs.

use starquake_ipc::{
    BUFFER_SIZE, Body, CMIF_HEADER_BYTES, CmifRequest, CommandHeader, CommandType,
    DOMAIN_HEADER_BYTES, DomainCommand, DomainRequestHeader, Message, RAW_ALIGNMENT_WORDS,
    ReceiveDescriptor, ReceiveList, RequestBuilder, SFCI_MAGIC, SessionKind, TIPC_COMMAND_OFFSET,
};

const WORD: usize = size_of::<u32>();

fn built(builder: &RequestBuilder) -> Vec<u8> {
    let mut buf = vec![0xA5_u8; BUFFER_SIZE];
    let result = builder.build(&mut buf);
    assert!(result.is_ok(), "{result:?}");
    buf
}

/// The CMIF request in a parsed message.
fn cmif_request(message: Message<'_>) -> Option<CmifRequest<'_>> {
    match message.body {
        Body::CmifRequest(request) => Some(request),
        _ => None,
    }
}

fn header_of(bytes: &[u8]) -> Option<CommandHeader> {
    let (first, rest) = bytes.split_first_chunk::<WORD>()?;
    let second = rest.first_chunk::<WORD>()?;
    CommandHeader::decode([u32::from_le_bytes(*first), u32::from_le_bytes(*second)]).ok()
}

/// `bytes` with the raw data size field of its header replaced by `raw_data_words`.
fn with_raw_words(bytes: &[u8], raw_data_words: usize) -> Option<Vec<u8>> {
    let mut header = header_of(bytes)?;
    header.raw_data_words = u32::try_from(raw_data_words).ok()?;
    let [w0, w1] = header.encode().ok()?;
    let mut out = bytes.to_vec();
    out.get_mut(..WORD)?.copy_from_slice(&w0.to_le_bytes());
    out.get_mut(WORD..2 * WORD)?
        .copy_from_slice(&w1.to_le_bytes());
    Some(out)
}

/// The raw data size of a built message, in words.
fn raw_words(bytes: &[u8]) -> Option<usize> {
    usize::try_from(header_of(bytes)?.raw_data_words).ok()
}

/// The raw data size in words that `cmifMakeRequest` writes (`cmif.h`): it starts from a fixed
/// 16-byte reserve for the alignment padding, adds the domain header and object ids if there is
/// an object id, the `SFCI` header and the data, rounds up to a half word, and rounds up to words.
/// This leaves out the out-pointer size table, which the builder does not model.
fn libnx_cmif_raw_words(data_size: usize, domain_objects: Option<usize>) -> usize {
    let mut size = RAW_ALIGNMENT_WORDS * WORD;
    if let Some(objects) = domain_objects {
        size += DOMAIN_HEADER_BYTES + objects * WORD;
    }
    size += CMIF_HEADER_BYTES + data_size;
    size.next_multiple_of(size_of::<u16>()).div_ceil(WORD)
}

/// Words the builder pads with when the raw data section would start at the end of the two header
/// words.
fn builder_padding_words() -> usize {
    CommandHeader::WORDS.next_multiple_of(RAW_ALIGNMENT_WORDS) - CommandHeader::WORDS
}

/// libnx always reserves a full alignment unit (`actual_size = 16`), whatever the real padding
/// is; the builder writes the exact size. The parser takes both.
#[test]
fn cmif_raw_size_differs_from_libnx_by_the_padding_reserve() {
    for len in 0..=24_usize {
        let data = vec![0x5A_u8; len];
        let bytes = built(&RequestBuilder::cmif(CommandType::Request, 7).data(&data));
        let exact = raw_words(&bytes).unwrap();
        let libnx = libnx_cmif_raw_words(len, None);
        assert_eq!(
            exact + RAW_ALIGNMENT_WORDS - builder_padding_words(),
            libnx,
            "data length {len}"
        );

        // The same message with libnx's raw size still parses; the extra words are slack.
        let padded = with_raw_words(&bytes, libnx).unwrap();
        let message = Message::parse_request(&padded, SessionKind::Session).unwrap();
        let request = cmif_request(message).unwrap();
        assert!(request.data.starts_with(&data), "data length {len}");
        assert!(
            request.data[data.len()..].iter().all(|b| *b == 0),
            "data length {len}"
        );
    }
}

/// libnx counts `(size + 3) / 4` words for a TIPC request and nothing else, with the type
/// `request_id + 16` (`tipc.h`, `tipcMakeRequest`): the builder agrees exactly.
#[test]
fn tipc_requests_match_libnx() {
    for len in 0..=24_usize {
        for command_id in [0_u16, 3, 0x1234] {
            let data = vec![0x3C_u8; len];
            let bytes = built(&RequestBuilder::tipc(command_id).data(&data));
            let header = header_of(&bytes).unwrap();
            assert_eq!(
                header.command_type.to_raw(),
                Ok(command_id + TIPC_COMMAND_OFFSET)
            );
            assert_eq!(
                usize::try_from(header.raw_data_words).unwrap(),
                len.div_ceil(WORD)
            );
        }
    }
}

/// `tipcClose` sends a bare type 15 message; `cmifMakeCloseRequest` sends a bare type 2 message
/// for an object that is not in a domain.
#[test]
fn close_messages_match_libnx() {
    assert_eq!(
        header_of(&built(&RequestBuilder::tipc_close()))
            .unwrap()
            .command_type,
        CommandType::TipcClose
    );
    assert_eq!(
        header_of(&built(&RequestBuilder::close()))
            .unwrap()
            .command_type,
        CommandType::Close
    );
}

/// `cmifMakeCloseRequest` for a domain object: type 4, the domain header with type 2, the object
/// id, a zero data size and a zero token, and no `SFCI` header. libnx's size reserves 16 bytes
/// for the alignment padding (`(16 + sizeof(CmifDomainInHeader)) / 4` words), so its raw data is
/// longer than the builder's; the parser stops after the domain header.
#[test]
fn domain_close_matches_libnx() {
    let object_id = 0x0B1D;
    let bytes = built(&RequestBuilder::domain_close(object_id));
    assert_eq!(
        header_of(&bytes).unwrap().command_type,
        CommandType::Request
    );
    let exact = raw_words(&bytes).unwrap();
    let libnx = (RAW_ALIGNMENT_WORDS * WORD + DOMAIN_HEADER_BYTES) / WORD;
    assert_eq!(exact + RAW_ALIGNMENT_WORDS - builder_padding_words(), libnx);

    let libnx_bytes = with_raw_words(&bytes, libnx).unwrap();
    let message = Message::parse_request(&libnx_bytes, SessionKind::Domain).unwrap();
    assert_eq!(
        message.body,
        Body::DomainClose(DomainRequestHeader {
            command: DomainCommand::CloseVirtualHandle,
            input_object_count: 0,
            size: 0,
            object_id,
            token: 0,
        })
    );
}

/// What libnx writes for a domain request whose arguments are not a whole number of words: the
/// `size` field is the `SFCI` header plus the exact argument bytes (`payload_size` in
/// `cmifMakeRequest`) and the object ids follow the arguments at once, unaligned.
#[test]
fn parser_accepts_the_domain_layout_libnx_writes() {
    let args = [1_u8, 2, 3, 4, 5];
    let object_ids = [0xFEED_0001_u32, 0xFEED_0002];
    let object_id = 9;
    let command_id = 0xC0DE;

    let mut raw = Vec::new();
    // 16-byte-aligned start: the type and size words, then two words of padding (the reserve).
    raw.extend_from_slice(&[0_u8; 2 * WORD]);
    let size = CMIF_HEADER_BYTES + args.len();
    let domain_word0 = DomainCommand::SendMessage.to_raw()
        | u32::try_from(object_ids.len()).unwrap() << 8
        | u32::try_from(size).unwrap() << 16;
    for word in [domain_word0, object_id, 0, 0] {
        raw.extend_from_slice(&word.to_le_bytes());
    }
    for word in [SFCI_MAGIC, 0, command_id, 0] {
        raw.extend_from_slice(&word.to_le_bytes());
    }
    raw.extend_from_slice(&args);
    for id in object_ids {
        raw.extend_from_slice(&id.to_le_bytes());
    }
    let raw_words = libnx_cmif_raw_words(args.len(), Some(object_ids.len()));
    assert!(
        raw.len() <= raw_words * WORD,
        "the reserve covers the content"
    );

    // The two header words of a plain request; the raw data size is set to libnx's below.
    let template = built(&RequestBuilder::cmif(CommandType::Request, command_id));
    let mut bytes = template[..CommandHeader::WORDS * WORD].to_vec();
    bytes.extend_from_slice(&raw);
    bytes.resize(BUFFER_SIZE, 0);
    let bytes = with_raw_words(&bytes, raw_words).unwrap();

    let message = Message::parse_request(&bytes, SessionKind::Domain).unwrap();
    let request = cmif_request(message).unwrap();
    assert_eq!(request.data, args);
    assert_eq!(request.object_ids, object_ids);
    assert_eq!(
        request.domain.map(|d| usize::try_from(d.size).unwrap()),
        Some(size)
    );
}

/// The builder rounds the arguments up to a whole word before the object ids and counts that
/// padding in `size`, where libnx counts the exact bytes; the two agree when the arguments are a
/// whole number of words. This is the listed divergence.
#[test]
fn builder_pads_domain_arguments_where_libnx_does_not() {
    for len in 0..=12_usize {
        let args = vec![0x77_u8; len];
        let bytes = built(
            &RequestBuilder::domain(CommandType::Request, 9, 1)
                .data(&args)
                .object_id(0xFEED),
        );
        let message = Message::parse_request(&bytes, SessionKind::Domain).unwrap();
        let request = cmif_request(message).unwrap();
        let libnx_size = CMIF_HEADER_BYTES + len;
        let builder_size = CMIF_HEADER_BYTES + len.next_multiple_of(WORD);
        assert_eq!(
            request.domain.map(|d| usize::try_from(d.size).unwrap()),
            Some(builder_size)
        );
        assert_eq!(builder_size == libnx_size, len % WORD == 0, "length {len}");
        assert_eq!(request.object_ids, [0xFEED]);
    }
}

/// libnx writes the receive list flag `2 + n` for `n` explicit C descriptors, so a single one is
/// flag 3, and flag 2 only for `HIPC_AUTO_RECV_STATIC` (`hipc.h`, `hipcMakeRequest`).
#[test]
fn one_receive_descriptor_is_flag_three_unless_set() {
    let c = ReceiveDescriptor {
        address: 0x4000,
        size: 0x40,
    };
    let flag =
        |builder: RequestBuilder| header_of(&built(&builder)).unwrap().receive_list.to_flag();
    let base = || RequestBuilder::cmif(CommandType::Request, 1);
    assert_eq!(flag(base().receive_descriptor(c)), Ok(3));
    assert_eq!(
        flag(base().receive_descriptor(c).receive_descriptor(c)),
        Ok(4)
    );
    assert_eq!(
        flag(base().receive_list(ReceiveList::One).receive_descriptor(c)),
        Ok(2)
    );
}

/// `cmifMakeControlRequest` writes type 5 with version 0 and token 0; with a context, libnx
/// writes type 6, version 1, and the context in the `SFCI` token (or, in a domain, in the domain
/// header's token with an `SFCI` token of zero).
#[test]
fn control_and_context_conventions_are_expressible() {
    let bytes = built(&RequestBuilder::cmif(CommandType::Control, 3));
    let message = Message::parse_request(&bytes, SessionKind::Domain).unwrap();
    assert_eq!(message.command_type, CommandType::Control);
    let request = cmif_request(message).unwrap();
    assert_eq!((request.header.version, request.header.token), (0, 0));
    assert_eq!(request.domain, None);

    let context = 0xC0DE_0001;
    let bytes = built(
        &RequestBuilder::cmif(CommandType::RequestWithContext, 3)
            .version(1)
            .token(context),
    );
    let message = Message::parse_request(&bytes, SessionKind::Session).unwrap();
    let request = cmif_request(message).unwrap();
    assert_eq!((request.header.version, request.header.token), (1, context));

    let bytes = built(
        &RequestBuilder::domain(CommandType::RequestWithContext, 5, 3)
            .version(1)
            .domain_token(context),
    );
    let message = Message::parse_request(&bytes, SessionKind::Domain).unwrap();
    let request = cmif_request(message).unwrap();
    assert_eq!((request.header.version, request.header.token), (1, 0));
    assert_eq!(request.domain.map(|d| d.token), Some(context));
}
