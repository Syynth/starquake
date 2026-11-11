//! Property tests: parsing arbitrary bytes never panics, descriptors and headers survive
//! encode/decode, and built replies parse back to what was put in.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_ipc::{
    BUFFER_SIZE, Body, BufferDescriptor, CMIF_HEADER_BYTES, CmifRequest, CmifRequestHeader,
    CommandHeader, CommandType, DomainCommand, DomainReplyHeader, DomainRequestHeader, IpcError,
    MAX_DESCRIPTORS, MAX_HANDLES, MAX_RAW_WORDS, MAX_TIPC_COMMAND_ID, Message, ReceiveDescriptor,
    ReceiveList, ReplyBuilder, SessionKind, SpecialHeader, StaticDescriptor, TIPC_COMMAND_OFFSET,
};

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 512,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x5741_5251_5541_4B45),
        failure_persistence: None,
        ..Config::default()
    }
}

fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn session_kind() -> impl Strategy<Value = SessionKind> {
    prop_oneof![Just(SessionKind::Session), Just(SessionKind::Domain)]
}

/// Words whose first two look like a plausible header, so the parser gets past the first checks.
fn structured_buffer() -> impl Strategy<Value = Vec<u8>> {
    // Type 0 and the CMIF types, the TIPC close type just below the TIPC offset, and a span of
    // TIPC command types starting at it.
    let tipc_offset = u32::from(TIPC_COMMAND_OFFSET);
    (
        prop_oneof![
            Just(0_u32),
            1..8_u32,
            Just(tipc_offset - 1),
            tipc_offset..tipc_offset + 24
        ],
        0..16_u32,
        0..MAX_RAW_WORDS,
        0..16_u32,
        any::<bool>(),
        prop::collection::vec(any::<u32>(), 0..70),
    )
        .prop_map(|(ty, counts, raw, flag, special, tail)| {
            let w0 = ty | (counts << 16) | (counts.rotate_left(4) << 20);
            let w1 = (raw % 80) | (flag << 10) | (u32::from(special) << 31);
            let mut words = vec![w0, w1];
            words.extend(tail);
            le(&words)
        })
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn parsing_arbitrary_bytes_never_panics(
        bytes in prop::collection::vec(any::<u8>(), 0..300),
        kind in session_kind(),
    ) {
        let _ = Message::parse_request(&bytes, kind);
        let _ = Message::parse_reply(&bytes, kind);
    }

    #[test]
    fn parsing_plausible_headers_never_panics(bytes in structured_buffer(), kind in session_kind()) {
        let _ = Message::parse_request(&bytes, kind);
        let _ = Message::parse_reply(&bytes, kind);
    }

    #[test]
    fn command_header_decode_encode_is_stable(words in any::<[u32; 2]>()) {
        if let Ok(header) = CommandHeader::decode(words) {
            let encoded = header.encode().unwrap();
            prop_assert_eq!(CommandHeader::decode(encoded).unwrap(), header);
        }
    }

    #[test]
    fn special_header_round_trips(word in any::<u32>()) {
        let header = SpecialHeader::decode(word);
        prop_assert!(header.num_copy_handles <= MAX_HANDLES);
        prop_assert!(header.num_move_handles <= MAX_HANDLES);
        prop_assert_eq!(SpecialHeader::decode(header.encode().unwrap()), header);
    }

    #[test]
    fn static_descriptor_round_trips(words in any::<[u32; 2]>()) {
        let descriptor = StaticDescriptor::decode(words);
        prop_assert_eq!(StaticDescriptor::decode(descriptor.encode().unwrap()), descriptor);
    }

    #[test]
    fn buffer_descriptor_round_trips(words in any::<[u32; 3]>()) {
        let descriptor = BufferDescriptor::decode(words);
        prop_assert_eq!(BufferDescriptor::decode(descriptor.encode().unwrap()), descriptor);
    }

    #[test]
    fn receive_descriptor_round_trips(words in any::<[u32; 2]>()) {
        let descriptor = ReceiveDescriptor::decode(words);
        // Every address and size bit of the layout is represented, so the words survive too.
        prop_assert_eq!(descriptor.encode().unwrap(), words);
    }

    #[test]
    fn descriptors_reject_values_wider_than_their_fields(address in (1_u64 << 39)..u64::MAX) {
        let x = StaticDescriptor { counter: 0, address, size: 0 };
        assert!(matches!(x.encode(), Err(IpcError::FieldOutOfRange { .. })));
        let abw = BufferDescriptor { address, size: 0, flags: 0 };
        assert!(matches!(abw.encode(), Err(IpcError::FieldOutOfRange { .. })));
    }

    #[test]
    fn domain_request_header_round_trips(words in any::<[u32; 4]>()) {
        if let Ok(header) = DomainRequestHeader::decode(words) {
            prop_assert_eq!(DomainRequestHeader::decode(header.encode().unwrap()).unwrap(), header);
        }
    }

    #[test]
    fn domain_reply_header_round_trips(count in any::<u32>()) {
        let header = DomainReplyHeader { num_objects: count };
        prop_assert_eq!(DomainReplyHeader::decode(header.encode()), header);
    }

    #[test]
    fn receive_list_flag_round_trips(flag in 0_u32..16) {
        prop_assert_eq!(ReceiveList::from_flag(flag).to_flag().unwrap(), flag);
    }

    #[test]
    fn command_type_round_trips(raw in any::<u16>()) {
        if let Ok(ty) = CommandType::from_raw(raw) {
            prop_assert_eq!(ty.to_raw().unwrap(), raw);
        }
    }

    #[test]
    fn built_cmif_replies_parse_back(
        result in any::<u32>(),
        domain in any::<bool>(),
        copy in prop::collection::vec(any::<u32>(), 0..=3),
        moved in prop::collection::vec(any::<u32>(), 0..=3),
        data in prop::collection::vec(any::<u32>(), 0..=6),
        ids in prop::collection::vec(any::<u32>(), 0..=4),
    ) {
        let mut builder = if domain { ReplyBuilder::domain(result) } else { ReplyBuilder::cmif(result) };
        for handle in &copy { builder = builder.copy_handle(*handle); }
        for handle in &moved { builder = builder.move_handle(*handle); }
        builder = builder.data(&le(&data));
        let ids = if domain { ids } else { Vec::new() };
        for id in &ids { builder = builder.object_id(*id); }

        let mut buf = [0xA5_u8; BUFFER_SIZE];
        let used = builder.build(&mut buf).unwrap();
        prop_assert_eq!(used % 4, 0);
        prop_assert!(buf[used..].iter().all(|b| *b == 0));

        let kind = if domain { SessionKind::Domain } else { SessionKind::Session };
        let message = Message::parse_reply(&buf, kind).unwrap();
        prop_assert_eq!(message.command_type, CommandType::Invalid);
        prop_assert_eq!(&message.copy_handles, &copy);
        prop_assert_eq!(&message.move_handles, &moved);
        let Body::CmifReply(reply) = message.body else {
            return Err(TestCaseError::fail("not a CMIF reply"));
        };
        prop_assert_eq!(reply.header.result, result);
        prop_assert_eq!(reply.data, le(&data));
        let count = u32::try_from(ids.len()).unwrap();
        prop_assert_eq!(reply.domain.map(|d| d.num_objects), domain.then_some(count));
        prop_assert_eq!(reply.object_ids, ids);
    }

    #[test]
    fn built_tipc_replies_parse_back(
        command_id in 0..=MAX_TIPC_COMMAND_ID,
        result in any::<u32>(),
        copy in prop::collection::vec(any::<u32>(), 0..=3),
        data in prop::collection::vec(any::<u32>(), 0..=8),
        kind in session_kind(),
    ) {
        let mut builder = ReplyBuilder::tipc(command_id, result).data(&le(&data));
        for handle in &copy { builder = builder.copy_handle(*handle); }

        let mut buf = [0xA5_u8; BUFFER_SIZE];
        let used = builder.build(&mut buf).unwrap();
        // 2 header words, optional special header + handles, 1 result word, then the data.
        let special = if copy.is_empty() { 0 } else { 1 + copy.len() };
        prop_assert_eq!(used, 4 * (2 + special + 1 + data.len()));

        let message = Message::parse_reply(&buf, kind).unwrap();
        prop_assert_eq!(message.command_type, CommandType::Tipc(command_id));
        prop_assert_eq!(&message.copy_handles, &copy);
        prop_assert_eq!(message.body, Body::TipcReply { result, data: &le(&data) });
    }

    #[test]
    fn building_into_any_capacity_never_panics(
        capacity in 0_usize..400,
        data in prop::collection::vec(any::<u8>(), 0..400),
        ids in prop::collection::vec(any::<u32>(), 0..8),
        handles in prop::collection::vec(any::<u32>(), 0..20),
        command_id in any::<u16>(),
        which in 0..3_u8,
    ) {
        let mut builder = match which {
            0 => ReplyBuilder::cmif(0),
            1 => ReplyBuilder::domain(0),
            _ => ReplyBuilder::tipc(command_id, 0),
        }
        .data(&data);
        for id in &ids { builder = builder.object_id(*id); }
        for handle in &handles { builder = builder.move_handle(*handle); }

        let mut buf = vec![0x77_u8; capacity];
        let before = buf.clone();
        if let Ok(used) = builder.build(&mut buf) {
            prop_assert!(used <= capacity.min(BUFFER_SIZE));
        } else {
            // On failure the buffer is untouched.
            prop_assert_eq!(buf, before);
        }
    }
}

/// What a generated request carries after its descriptors.
#[derive(Debug, Clone)]
enum Payload {
    Cmif {
        command_type: CommandType,
        command_id: u32,
        version: u32,
        token: u32,
        object_id: u32,
        data: Vec<u32>,
        input_objects: Vec<u32>,
    },
    Tipc {
        command_id: u16,
        data: Vec<u32>,
    },
}

/// A request described field by field, so the test can assemble it with the crate's encoders and
/// then check that the parser hands every list back in order.
#[derive(Debug, Clone)]
struct GeneratedRequest {
    pid: Option<u64>,
    force_special_header: bool,
    copy_handles: Vec<u32>,
    move_handles: Vec<u32>,
    statics: Vec<StaticDescriptor>,
    sends: Vec<BufferDescriptor>,
    receives: Vec<BufferDescriptor>,
    exchanges: Vec<BufferDescriptor>,
    receive_list: ReceiveList,
    receive_descriptors: Vec<ReceiveDescriptor>,
    payload: Payload,
}

/// A length as a header field. Saturates, so an impossible value fails the encoder's range check
/// instead of panicking in a helper.
fn count(len: usize) -> u32 {
    u32::try_from(len).unwrap_or(u32::MAX)
}

fn count_u64(value: u64) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

impl GeneratedRequest {
    /// Whether the parser should find a domain header for this request on a session of `kind`.
    fn expects_domain_header(&self, kind: SessionKind) -> bool {
        matches!(
            self.payload,
            Payload::Cmif {
                command_type: CommandType::Request | CommandType::RequestWithContext,
                ..
            }
        ) && kind == SessionKind::Domain
    }

    /// Lays the request out word by word with the crate's encode functions.
    fn assemble(&self, kind: SessionKind) -> Result<Vec<u8>, IpcError> {
        let has_special_header = self.force_special_header
            || self.pid.is_some()
            || !self.copy_handles.is_empty()
            || !self.move_handles.is_empty();
        // The two header words are filled in last.
        let mut words = vec![0_u32, 0];
        if has_special_header {
            let special = SpecialHeader {
                send_pid: self.pid.is_some(),
                num_copy_handles: count(self.copy_handles.len()),
                num_move_handles: count(self.move_handles.len()),
            };
            words.push(special.encode()?);
            if let Some(pid) = self.pid {
                words.push(count_u64(pid & u64::from(u32::MAX)));
                words.push(count_u64(pid >> 32));
            }
            words.extend_from_slice(&self.copy_handles);
            words.extend_from_slice(&self.move_handles);
        }
        for descriptor in &self.statics {
            words.extend(descriptor.encode()?);
        }
        for descriptor in self
            .sends
            .iter()
            .chain(&self.receives)
            .chain(&self.exchanges)
        {
            words.extend(descriptor.encode()?);
        }

        let raw_start = words.len();
        let command_type = match &self.payload {
            Payload::Cmif {
                command_type,
                command_id,
                version,
                token,
                object_id,
                data,
                input_objects,
            } => {
                // CMIF content is aligned to 16 bytes, measured from the start of the buffer.
                words.resize(words.len().next_multiple_of(4), 0);
                if self.expects_domain_header(kind) {
                    words.extend(
                        DomainRequestHeader {
                            command: DomainCommand::SendMessage,
                            input_object_count: count(input_objects.len()),
                            size: count(CMIF_HEADER_BYTES + data.len() * 4),
                            object_id: *object_id,
                            token: *token,
                        }
                        .encode()?,
                    );
                }
                words.extend(
                    CmifRequestHeader {
                        version: *version,
                        command_id: *command_id,
                        token: *token,
                    }
                    .encode(),
                );
                words.extend_from_slice(data);
                if self.expects_domain_header(kind) {
                    words.extend_from_slice(input_objects);
                }
                *command_type
            }
            Payload::Tipc { command_id, data } => {
                words.extend_from_slice(data);
                CommandType::Tipc(*command_id)
            }
        };
        let raw_data_words = count(words.len() - raw_start);
        for descriptor in &self.receive_descriptors {
            words.extend(descriptor.encode()?);
        }

        let header = CommandHeader {
            command_type,
            num_x: count(self.statics.len()),
            num_a: count(self.sends.len()),
            num_b: count(self.receives.len()),
            num_w: count(self.exchanges.len()),
            raw_data_words,
            receive_list: self.receive_list,
            has_special_header,
        };
        let [w0, w1] = header.encode()?;
        words[0] = w0;
        words[1] = w1;

        let mut bytes = le(&words);
        assert!(
            bytes.len() <= BUFFER_SIZE,
            "generated request overflows the IPC buffer"
        );
        bytes.resize(BUFFER_SIZE, 0);
        Ok(bytes)
    }
}

fn static_descriptor() -> impl Strategy<Value = StaticDescriptor> {
    any::<[u32; 2]>().prop_map(StaticDescriptor::decode)
}

fn buffer_descriptor() -> impl Strategy<Value = BufferDescriptor> {
    any::<[u32; 3]>().prop_map(BufferDescriptor::decode)
}

fn receive_descriptor() -> impl Strategy<Value = ReceiveDescriptor> {
    any::<[u32; 2]>().prop_map(ReceiveDescriptor::decode)
}

/// A receive list and exactly as many C descriptors as it announces.
fn receive_list_and_descriptors() -> impl Strategy<Value = (ReceiveList, Vec<ReceiveDescriptor>)> {
    let receive_list = prop_oneof![
        Just(ReceiveList::Disabled),
        Just(ReceiveList::Inline),
        Just(ReceiveList::One),
        (1..=3_u32).prop_map(ReceiveList::Many),
    ];
    (receive_list, prop::collection::vec(receive_descriptor(), 3)).prop_map(
        |(list, mut descriptors)| {
            descriptors.truncate(usize::try_from(list.descriptor_count()).unwrap_or(usize::MAX));
            (list, descriptors)
        },
    )
}

fn payload() -> impl Strategy<Value = Payload> {
    let cmif = (
        prop_oneof![
            Just(CommandType::Request),
            Just(CommandType::Control),
            Just(CommandType::RequestWithContext),
            Just(CommandType::ControlWithContext),
        ],
        any::<[u32; 4]>(),
        prop::collection::vec(any::<u32>(), 0..=4),
        prop::collection::vec(any::<u32>(), 0..=2),
    )
        .prop_map(
            |(command_type, [command_id, version, token, object_id], data, input_objects)| {
                Payload::Cmif {
                    command_type,
                    command_id,
                    version,
                    token,
                    object_id,
                    data,
                    input_objects,
                }
            },
        );
    let tipc = (
        0..=MAX_TIPC_COMMAND_ID,
        prop::collection::vec(any::<u32>(), 0..=8),
    )
        .prop_map(|(command_id, data)| Payload::Tipc { command_id, data });
    prop_oneof![cmif, tipc]
}

prop_compose! {
    /// Counts stay small enough that the largest request fits the 64-word IPC buffer.
    fn generated_request()(
        pid in prop::option::of(any::<u64>()),
        force_special_header in any::<bool>(),
        copy_handles in prop::collection::vec(any::<u32>(), 0..=3),
        move_handles in prop::collection::vec(any::<u32>(), 0..=3),
        statics in prop::collection::vec(static_descriptor(), 0..=3),
        sends in prop::collection::vec(buffer_descriptor(), 0..=2),
        receives in prop::collection::vec(buffer_descriptor(), 0..=2),
        exchanges in prop::collection::vec(buffer_descriptor(), 0..=2),
        (receive_list, receive_descriptors) in receive_list_and_descriptors(),
        payload in payload(),
    ) -> GeneratedRequest {
        GeneratedRequest {
            pid,
            force_special_header,
            copy_handles,
            move_handles,
            statics,
            sends,
            receives,
            exchanges,
            receive_list,
            receive_descriptors,
            payload,
        }
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn built_requests_parse_back_with_every_list_in_order(
        request in generated_request(),
        kind in session_kind(),
    ) {
        let bytes = request.assemble(kind).unwrap();
        let message = Message::parse_request(&bytes, kind).unwrap();

        match &request.payload {
            Payload::Cmif { command_type, .. } => prop_assert_eq!(message.command_type, *command_type),
            Payload::Tipc { command_id, .. } => {
                prop_assert_eq!(message.command_type, CommandType::Tipc(*command_id));
            }
        }
        prop_assert_eq!(message.pid, request.pid);
        prop_assert_eq!(&message.copy_handles, &request.copy_handles);
        prop_assert_eq!(&message.move_handles, &request.move_handles);
        prop_assert_eq!(&message.static_descriptors, &request.statics);
        prop_assert_eq!(&message.send_buffers, &request.sends);
        prop_assert_eq!(&message.receive_buffers, &request.receives);
        prop_assert_eq!(&message.exchange_buffers, &request.exchanges);
        prop_assert_eq!(message.receive_list, request.receive_list);
        prop_assert_eq!(&message.receive_descriptors, &request.receive_descriptors);

        let has_domain = request.expects_domain_header(kind);
        match request.payload {
            Payload::Cmif {
                command_id, version, token, object_id, data, input_objects, ..
            } => {
                let domain = has_domain.then(|| DomainRequestHeader {
                    command: DomainCommand::SendMessage,
                    input_object_count: u32::try_from(input_objects.len()).unwrap(),
                    size: u32::try_from(CMIF_HEADER_BYTES + data.len() * 4).unwrap(),
                    object_id,
                    token,
                });
                let object_ids = if domain.is_some() { input_objects } else { Vec::new() };
                prop_assert_eq!(
                    message.body,
                    Body::CmifRequest(CmifRequest {
                        domain,
                        header: CmifRequestHeader { version, command_id, token },
                        data: &le(&data),
                        object_ids,
                    })
                );
            }
            Payload::Tipc { data, .. } => {
                prop_assert_eq!(message.body, Body::TipcRequest { data: &le(&data) });
            }
        }
    }
}

#[test]
fn builder_failures_are_clean_errors() {
    let mut small = [0_u8; 16];
    assert!(matches!(
        ReplyBuilder::cmif(0).build(&mut small),
        Err(IpcError::DoesNotFit { .. })
    ));
    let mut buf = [0_u8; BUFFER_SIZE];
    assert!(matches!(
        ReplyBuilder::cmif(0)
            .data(&[0; BUFFER_SIZE])
            .build(&mut buf),
        Err(IpcError::DoesNotFit { .. })
    ));
    assert_eq!(
        ReplyBuilder::cmif(0).object_id(1).build(&mut buf),
        Err(IpcError::ObjectIdsOutsideDomain)
    );
    assert_eq!(
        ReplyBuilder::tipc(MAX_TIPC_COMMAND_ID + 1, 0).build(&mut buf),
        Err(IpcError::TipcCommandOutOfRange(MAX_TIPC_COMMAND_ID + 1))
    );
    let mut too_many = ReplyBuilder::cmif(0);
    for handle in 0..=MAX_HANDLES {
        too_many = too_many.copy_handle(handle);
    }
    assert!(matches!(
        too_many.build(&mut buf),
        Err(IpcError::FieldOutOfRange { .. })
    ));
}

#[test]
fn header_encoding_rejects_overwide_fields() {
    let header = CommandHeader {
        command_type: CommandType::Request,
        num_x: MAX_DESCRIPTORS + 1,
        num_a: 0,
        num_b: 0,
        num_w: 0,
        raw_data_words: 0,
        receive_list: ReceiveList::Disabled,
        has_special_header: false,
    };
    assert!(matches!(
        header.encode(),
        Err(IpcError::FieldOutOfRange { .. })
    ));
    assert!(matches!(
        ReceiveList::Many(14).to_flag(),
        Err(IpcError::FieldOutOfRange { .. })
    ));
    assert_eq!(
        CommandType::Tipc(u16::MAX).to_raw(),
        Err(IpcError::TipcCommandOutOfRange(u16::MAX))
    );
}
