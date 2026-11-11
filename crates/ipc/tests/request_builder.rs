//! `RequestBuilder`: build-then-parse round trips, list-order checks and error cases.
//!
//! Byte-for-byte checks against the hand-assembled buffers live in `fixtures.rs`, next to the
//! words they reproduce.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_ipc::{
    BUFFER_SIZE, Body, BufferDescriptor, CMIF_HEADER_BYTES, CmifRequest, CmifRequestHeader,
    CommandType, DOMAIN_HEADER_BYTES, DomainCommand, DomainRequestHeader, IpcError,
    MAX_DESCRIPTORS, MAX_HANDLES, MAX_RAW_WORDS, MAX_TIPC_COMMAND_ID, Message, ReceiveDescriptor,
    ReceiveList, RequestBuilder, SessionKind, StaticDescriptor,
};

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 512,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x5251_4255_494C_4452),
        failure_persistence: None,
        ..Config::default()
    }
}

fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The CMIF request in a parsed message.
fn cmif_request<'m, 'a>(message: &'m Message<'a>) -> Option<&'m CmifRequest<'a>> {
    match &message.body {
        Body::CmifRequest(request) => Some(request),
        _ => None,
    }
}

fn built(builder: &RequestBuilder) -> Vec<u8> {
    let mut buf = vec![0xA5_u8; BUFFER_SIZE];
    let result = builder.build(&mut buf);
    assert!(result.is_ok(), "{result:?}");
    buf
}

/// Distinct values in every list, so a builder that writes two lists in the wrong order, or a
/// parser that reads them in the wrong order, hands the wrong values back.
#[test]
fn every_list_lands_in_its_own_place() {
    let statics = [
        StaticDescriptor {
            counter: 1,
            address: 0x11_1111_1000,
            size: 0x11,
        },
        StaticDescriptor {
            counter: 2,
            address: 0x12_2222_2000,
            size: 0x12,
        },
    ];
    let abw = |base: u64| {
        [
            BufferDescriptor {
                address: base << 12,
                size: base << 4,
                flags: 1,
            },
            BufferDescriptor {
                address: (base + 1) << 12,
                size: (base + 1) << 4,
                flags: 2,
            },
        ]
    };
    let (sends, receives, exchanges) = (abw(0x21), abw(0x31), abw(0x41));
    let receive_descriptors = [
        ReceiveDescriptor {
            address: 0x5000,
            size: 0x51,
        },
        ReceiveDescriptor {
            address: 0x6000,
            size: 0x61,
        },
    ];
    let mut builder = RequestBuilder::domain(CommandType::Request, 0x0B1D, 0xC0DE)
        .version(0x7E57)
        .token(0x70C1)
        .domain_token(0x70D0)
        .send_pid(0x9999_8888_7777)
        .copy_handle(0xC1)
        .copy_handle(0xC2)
        .move_handle(0xD1)
        .move_handle(0xD2)
        .data(&le(&[0xDA7A_0001, 0xDA7A_0002]))
        .object_id(0x0B1)
        .object_id(0x0B2)
        .receive_list(ReceiveList::Many(2));
    for d in statics {
        builder = builder.static_descriptor(d);
    }
    for d in sends {
        builder = builder.send_buffer(d);
    }
    for d in receives {
        builder = builder.receive_buffer(d);
    }
    for d in exchanges {
        builder = builder.exchange_buffer(d);
    }
    for d in receive_descriptors {
        builder = builder.receive_descriptor(d);
    }

    let bytes = built(&builder);
    let message = Message::parse_request(&bytes, SessionKind::Domain).unwrap();
    assert_eq!(message.pid, Some(0x9999_8888_7777));
    assert_eq!(message.copy_handles, [0xC1, 0xC2]);
    assert_eq!(message.move_handles, [0xD1, 0xD2]);
    assert_eq!(message.static_descriptors, statics);
    assert_eq!(message.send_buffers, sends);
    assert_eq!(message.receive_buffers, receives);
    assert_eq!(message.exchange_buffers, exchanges);
    assert_eq!(message.receive_list, ReceiveList::Many(2));
    assert_eq!(message.receive_descriptors, receive_descriptors);
    assert_eq!(
        message.body,
        Body::CmifRequest(CmifRequest {
            domain: Some(DomainRequestHeader {
                command: DomainCommand::SendMessage,
                input_object_count: 2,
                size: u32::try_from(CMIF_HEADER_BYTES + 8).unwrap(),
                object_id: 0x0B1D,
                token: 0x70D0,
            }),
            header: CmifRequestHeader {
                version: 0x7E57,
                command_id: 0xC0DE,
                token: 0x70C1,
            },
            data: &le(&[0xDA7A_0001, 0xDA7A_0002]),
            object_ids: vec![0x0B1, 0x0B2],
        })
    );
}

/// What a generated request carries after its descriptors.
#[derive(Debug, Clone)]
enum Payload {
    Cmif {
        command_type: CommandType,
        command_id: u32,
        version: u32,
        token: u32,
        domain_token: u32,
        object_id: u32,
        data: Vec<u32>,
        input_objects: Vec<u32>,
    },
    Tipc {
        command_id: u16,
        data: Vec<u32>,
    },
}

/// A request described field by field: the test builds it with [`RequestBuilder`] and checks the
/// parser hands every field back.
#[derive(Debug, Clone)]
struct Spec {
    pid: Option<u64>,
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

impl Spec {
    /// Whether the parser should find a domain header for this request on a session of `kind`.
    fn has_domain_header(&self, kind: SessionKind) -> bool {
        matches!(
            self.payload,
            Payload::Cmif {
                command_type: CommandType::Request | CommandType::RequestWithContext,
                ..
            }
        ) && kind == SessionKind::Domain
    }

    fn builder(&self, kind: SessionKind) -> RequestBuilder {
        let mut builder = match &self.payload {
            Payload::Cmif {
                command_type,
                command_id,
                version,
                token,
                domain_token,
                object_id,
                data,
                input_objects,
            } => {
                let base = if self.has_domain_header(kind) {
                    let mut domain = RequestBuilder::domain(*command_type, *object_id, *command_id)
                        .domain_token(*domain_token);
                    for id in input_objects {
                        domain = domain.object_id(*id);
                    }
                    domain
                } else {
                    RequestBuilder::cmif(*command_type, *command_id)
                };
                base.version(*version).token(*token).data(&le(data))
            }
            Payload::Tipc { command_id, data } => RequestBuilder::tipc(*command_id).data(&le(data)),
        };
        if let Some(pid) = self.pid {
            builder = builder.send_pid(pid);
        }
        for handle in &self.copy_handles {
            builder = builder.copy_handle(*handle);
        }
        for handle in &self.move_handles {
            builder = builder.move_handle(*handle);
        }
        for d in &self.statics {
            builder = builder.static_descriptor(*d);
        }
        for d in &self.sends {
            builder = builder.send_buffer(*d);
        }
        for d in &self.receives {
            builder = builder.receive_buffer(*d);
        }
        for d in &self.exchanges {
            builder = builder.exchange_buffer(*d);
        }
        builder = builder.receive_list(self.receive_list);
        for d in &self.receive_descriptors {
            builder = builder.receive_descriptor(*d);
        }
        builder
    }
}

fn session_kind() -> impl Strategy<Value = SessionKind> {
    prop_oneof![Just(SessionKind::Session), Just(SessionKind::Domain)]
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
            Just(CommandType::LegacyRequest),
            Just(CommandType::Request),
            Just(CommandType::LegacyControl),
            Just(CommandType::Control),
            Just(CommandType::RequestWithContext),
            Just(CommandType::ControlWithContext),
        ],
        any::<[u32; 5]>(),
        prop::collection::vec(any::<u32>(), 0..=4),
        prop::collection::vec(any::<u32>(), 0..=2),
    )
        .prop_map(
            |(command_type, [command_id, version, token, domain_token, object_id], data, ids)| {
                Payload::Cmif {
                    command_type,
                    command_id,
                    version,
                    token,
                    domain_token,
                    object_id,
                    data,
                    input_objects: ids,
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
    fn spec()(
        pid in prop::option::of(any::<u64>()),
        copy_handles in prop::collection::vec(any::<u32>(), 0..=3),
        move_handles in prop::collection::vec(any::<u32>(), 0..=3),
        statics in prop::collection::vec(static_descriptor(), 0..=3),
        sends in prop::collection::vec(buffer_descriptor(), 0..=2),
        receives in prop::collection::vec(buffer_descriptor(), 0..=2),
        exchanges in prop::collection::vec(buffer_descriptor(), 0..=2),
        (receive_list, receive_descriptors) in receive_list_and_descriptors(),
        payload in payload(),
    ) -> Spec {
        Spec {
            pid,
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
    fn built_requests_parse_back_with_every_list_in_order(spec in spec(), kind in session_kind()) {
        let mut buf = [0xA5_u8; BUFFER_SIZE];
        let used = spec.builder(kind).build(&mut buf).unwrap();
        prop_assert_eq!(used % 4, 0);
        prop_assert!(buf[used..].iter().all(|b| *b == 0));
        let message = Message::parse_request(&buf, kind).unwrap();

        match &spec.payload {
            Payload::Cmif { command_type, .. } => prop_assert_eq!(message.command_type, *command_type),
            Payload::Tipc { command_id, .. } => {
                prop_assert_eq!(message.command_type, CommandType::Tipc(*command_id));
            }
        }
        prop_assert_eq!(message.pid, spec.pid);
        prop_assert_eq!(&message.copy_handles, &spec.copy_handles);
        prop_assert_eq!(&message.move_handles, &spec.move_handles);
        prop_assert_eq!(&message.static_descriptors, &spec.statics);
        prop_assert_eq!(&message.send_buffers, &spec.sends);
        prop_assert_eq!(&message.receive_buffers, &spec.receives);
        prop_assert_eq!(&message.exchange_buffers, &spec.exchanges);
        prop_assert_eq!(message.receive_list, spec.receive_list);
        prop_assert_eq!(&message.receive_descriptors, &spec.receive_descriptors);

        let has_domain = spec.has_domain_header(kind);
        match spec.payload {
            Payload::Cmif {
                command_id, version, token, domain_token, object_id, data, input_objects, ..
            } => {
                let domain = has_domain.then(|| DomainRequestHeader {
                    command: DomainCommand::SendMessage,
                    input_object_count: u32::try_from(input_objects.len()).unwrap(),
                    size: u32::try_from(CMIF_HEADER_BYTES + data.len() * 4).unwrap(),
                    object_id,
                    token: domain_token,
                });
                let object_ids = if has_domain { input_objects } else { Vec::new() };
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
fn domain_close_round_trips() {
    let bytes = built(&RequestBuilder::domain_close(0x0B1D).domain_token(0x70D0));
    let message = Message::parse_request(&bytes, SessionKind::Domain).unwrap();
    assert_eq!(message.command_type, CommandType::Request);
    assert_eq!(
        message.body,
        Body::DomainClose(DomainRequestHeader {
            command: DomainCommand::CloseVirtualHandle,
            input_object_count: 0,
            size: 0,
            object_id: 0x0B1D,
            token: 0x70D0,
        })
    );
}

#[test]
fn domain_close_carries_no_data_or_object_ids() {
    let mut buf = [0_u8; BUFFER_SIZE];
    for builder in [
        RequestBuilder::domain_close(1).data(&[0; 4]),
        RequestBuilder::domain_close(1).object_id(2),
    ] {
        assert!(matches!(
            builder.build(&mut buf),
            Err(IpcError::Inconsistent(_))
        ));
    }
}

#[test]
fn close_messages_are_only_the_header() {
    for (builder, command_type) in [
        (RequestBuilder::close(), CommandType::Close),
        (RequestBuilder::tipc_close(), CommandType::TipcClose),
    ] {
        let mut buf = [0xA5_u8; BUFFER_SIZE];
        let used = builder.build(&mut buf).unwrap();
        assert_eq!(used, 8);
        let message = Message::parse_request(&buf, SessionKind::Session).unwrap();
        assert_eq!(message.command_type, command_type);
        assert_eq!(message.body, Body::Close);
    }
}

#[test]
fn close_messages_reject_anything_else() {
    let mut buf = [0_u8; BUFFER_SIZE];
    for builder in [
        RequestBuilder::close().copy_handle(1),
        RequestBuilder::close().data(&[0; 4]),
        RequestBuilder::tipc_close().send_pid(0),
        RequestBuilder::tipc_close().send_buffer(BufferDescriptor {
            address: 0,
            size: 0,
            flags: 0,
        }),
        RequestBuilder::close().receive_descriptor(ReceiveDescriptor {
            address: 0,
            size: 0,
        }),
    ] {
        assert!(matches!(
            builder.build(&mut buf),
            Err(IpcError::Inconsistent(_))
        ));
    }
}

/// Data that is not a whole number of words is zero-padded; in a domain the size field counts
/// the padding, so the object ids that follow are found.
#[test]
fn partial_trailing_word_is_zero_padded() {
    let args = [1_u8, 2, 3, 4, 5];
    let padded = [1_u8, 2, 3, 4, 5, 0, 0, 0];

    let bytes = built(&RequestBuilder::cmif(CommandType::Request, 1).data(&args));
    let message = Message::parse_request(&bytes, SessionKind::Session).unwrap();
    let request = cmif_request(&message).unwrap();
    assert!(request.data.starts_with(&padded));

    let bytes = built(
        &RequestBuilder::domain(CommandType::Request, 9, 1)
            .data(&args)
            .object_id(0xFEED),
    );
    let message = Message::parse_request(&bytes, SessionKind::Domain).unwrap();
    let request = cmif_request(&message).unwrap();
    assert_eq!(request.data, padded);
    assert_eq!(request.object_ids, [0xFEED]);
    assert_eq!(
        request.domain.map(|d| d.size),
        Some(u32::try_from(CMIF_HEADER_BYTES + padded.len()).unwrap())
    );
}

/// The domain header (`DOMAIN_HEADER_BYTES`) and the `SFCI` header both precede the arguments,
/// after the 16-byte alignment of the raw data section.
#[test]
fn domain_requests_are_aligned_to_sixteen_bytes() {
    // Two header words, a special header, a two-word pid and one copy handle: the raw data would
    // start at word 6, so the builder pads to word 8.
    let bytes = built(
        &RequestBuilder::domain(CommandType::Request, 1, 2)
            .send_pid(0)
            .copy_handle(7),
    );
    let message = Message::parse_request(&bytes, SessionKind::Domain).unwrap();
    assert!(matches!(message.body, Body::CmifRequest(_)));
    let raw_words =
        usize::try_from(u32::from_le_bytes([bytes[4], bytes[5], 0, 0]) & MAX_RAW_WORDS).unwrap();
    // 2 padding words, then both 16-byte headers.
    assert_eq!(raw_words, 2 + (DOMAIN_HEADER_BYTES + CMIF_HEADER_BYTES) / 4);
}

#[test]
fn receive_list_follows_the_descriptor_count_unless_set() {
    let c = ReceiveDescriptor {
        address: 0x4000,
        size: 0x40,
    };
    let lists = |builder: RequestBuilder| {
        let bytes = built(&builder);
        Message::parse_request(&bytes, SessionKind::Session)
            .unwrap()
            .receive_list
    };
    let base = || RequestBuilder::cmif(CommandType::Request, 1);
    assert_eq!(lists(base()), ReceiveList::Disabled);
    assert_eq!(lists(base().receive_descriptor(c)), ReceiveList::One);
    assert_eq!(
        lists(base().receive_descriptor(c).receive_descriptor(c)),
        ReceiveList::Many(2)
    );
    assert_eq!(
        lists(base().receive_list(ReceiveList::Inline)),
        ReceiveList::Inline
    );
    assert_eq!(
        lists(
            base()
                .receive_list(ReceiveList::Many(1))
                .receive_descriptor(c)
        ),
        ReceiveList::Many(1)
    );
}

#[test]
fn a_receive_list_that_disagrees_with_its_descriptors_is_an_error() {
    let c = ReceiveDescriptor {
        address: 0,
        size: 0,
    };
    let mut buf = [0_u8; BUFFER_SIZE];
    for builder in [
        RequestBuilder::cmif(CommandType::Request, 1).receive_list(ReceiveList::One),
        RequestBuilder::cmif(CommandType::Request, 1)
            .receive_list(ReceiveList::Disabled)
            .receive_descriptor(c),
        RequestBuilder::cmif(CommandType::Request, 1)
            .receive_list(ReceiveList::Inline)
            .receive_descriptor(c),
        RequestBuilder::cmif(CommandType::Request, 1)
            .receive_list(ReceiveList::Many(2))
            .receive_descriptor(c),
    ] {
        assert!(matches!(
            builder.build(&mut buf),
            Err(IpcError::Inconsistent(_))
        ));
    }
}

#[test]
fn command_types_that_do_not_fit_the_constructor_are_errors() {
    let mut buf = [0_u8; BUFFER_SIZE];
    for command_type in [
        CommandType::Invalid,
        CommandType::Close,
        CommandType::TipcClose,
        CommandType::Tipc(1),
    ] {
        assert_eq!(
            RequestBuilder::cmif(command_type, 0).build(&mut buf),
            Err(IpcError::UnexpectedCommandType(command_type))
        );
    }
    // Only request and request-with-context carry a domain header.
    for command_type in [
        CommandType::LegacyRequest,
        CommandType::LegacyControl,
        CommandType::Control,
        CommandType::ControlWithContext,
        CommandType::Invalid,
    ] {
        assert_eq!(
            RequestBuilder::domain(command_type, 1, 0).build(&mut buf),
            Err(IpcError::UnexpectedCommandType(command_type))
        );
    }
}

#[test]
fn object_ids_need_a_domain_request() {
    let mut buf = [0_u8; BUFFER_SIZE];
    for builder in [
        RequestBuilder::cmif(CommandType::Request, 1).object_id(1),
        RequestBuilder::tipc(1).object_id(1),
    ] {
        assert_eq!(
            builder.build(&mut buf),
            Err(IpcError::ObjectIdsOutsideDomain)
        );
    }
}

/// One failure case per new bound: counts, sizes and ids one past what their field holds.
#[test]
fn out_of_range_values_are_errors_and_leave_the_buffer_alone() {
    let zero_buffer = BufferDescriptor {
        address: 0,
        size: 0,
        flags: 0,
    };
    let past = |n: u32| usize::try_from(n + 1).unwrap();
    let base = || RequestBuilder::cmif(CommandType::Request, 1);

    let mut too_many_x = base();
    for _ in 0..past(MAX_DESCRIPTORS) {
        too_many_x = too_many_x.static_descriptor(StaticDescriptor {
            counter: 0,
            address: 0,
            size: 0,
        });
    }
    let mut too_many_a = base();
    let mut too_many_b = base();
    let mut too_many_w = base();
    for _ in 0..past(MAX_DESCRIPTORS) {
        too_many_a = too_many_a.send_buffer(zero_buffer);
        too_many_b = too_many_b.receive_buffer(zero_buffer);
        too_many_w = too_many_w.exchange_buffer(zero_buffer);
    }
    let mut too_many_copies = base();
    let mut too_many_moves = base();
    for handle in 0..=MAX_HANDLES {
        too_many_copies = too_many_copies.copy_handle(handle);
        too_many_moves = too_many_moves.move_handle(handle);
    }
    // 0x400 words of data is one word past the 10-bit raw size; as it also overflows the buffer
    // the field check must come first.
    let raw_overflow = base().data(&vec![0; past(MAX_RAW_WORDS) * 4]);
    // A domain message whose size does not fit 16 bits, and one with 256 input objects.
    let domain_size_overflow = RequestBuilder::domain(CommandType::Request, 1, 1)
        .data(&vec![0; usize::from(u16::MAX) + 1]);
    let mut domain_count_overflow = RequestBuilder::domain(CommandType::Request, 1, 1);
    for id in 0..0x100 {
        domain_count_overflow = domain_count_overflow.object_id(id);
    }
    let wide_address = base().send_buffer(BufferDescriptor {
        address: 1 << 39,
        size: 0,
        flags: 0,
    });
    let tipc_too_high = RequestBuilder::tipc(MAX_TIPC_COMMAND_ID + 1);

    let mut buf = [0x3C_u8; BUFFER_SIZE];
    for (name, builder) in [
        ("x", too_many_x),
        ("a", too_many_a),
        ("b", too_many_b),
        ("w", too_many_w),
        ("copy", too_many_copies),
        ("move", too_many_moves),
        ("raw", raw_overflow),
        ("domain size", domain_size_overflow),
        ("domain count", domain_count_overflow),
        ("address", wide_address),
    ] {
        assert!(
            matches!(
                builder.build(&mut buf),
                Err(IpcError::FieldOutOfRange { .. })
            ),
            "{name}"
        );
    }
    assert_eq!(
        tipc_too_high.build(&mut buf),
        Err(IpcError::TipcCommandOutOfRange(MAX_TIPC_COMMAND_ID + 1))
    );
    assert!(buf.iter().all(|b| *b == 0x3C));
}

#[test]
fn a_request_larger_than_the_buffer_does_not_fit() {
    let mut buf = [0x3C_u8; BUFFER_SIZE];
    // Within every field's range, but more words than the buffer holds.
    let long = RequestBuilder::cmif(CommandType::Request, 1).data(&vec![0; BUFFER_SIZE]);
    assert!(matches!(
        long.build(&mut buf),
        Err(IpcError::DoesNotFit { .. })
    ));
    // A smaller destination than the minimum message.
    let mut small = [0x3C_u8; 8];
    assert!(matches!(
        RequestBuilder::cmif(CommandType::Request, 1).build(&mut small),
        Err(IpcError::DoesNotFit { .. })
    ));
    assert!(small.iter().all(|b| *b == 0x3C));
    assert!(buf.iter().all(|b| *b == 0x3C));
}

proptest! {
    #![proptest_config(config())]

    /// Whatever is thrown at the builder, it returns `Err` or a buffer that parses.
    #[test]
    fn building_into_any_capacity_never_panics(
        capacity in 0_usize..400,
        data in prop::collection::vec(any::<u8>(), 0..400),
        ids in prop::collection::vec(any::<u32>(), 0..8),
        handles in prop::collection::vec(any::<u32>(), 0..20),
        command_id in any::<u16>(),
        which in 0..4_u8,
    ) {
        let mut builder = match which {
            0 => RequestBuilder::cmif(CommandType::Request, 0),
            1 => RequestBuilder::domain(CommandType::Request, 0, 0),
            2 => RequestBuilder::domain_close(0),
            _ => RequestBuilder::tipc(command_id),
        }
        .data(&data);
        for id in &ids { builder = builder.object_id(*id); }
        for handle in &handles { builder = builder.move_handle(*handle); }

        let mut buf = vec![0x77_u8; capacity];
        let before = buf.clone();
        if let Ok(used) = builder.build(&mut buf) {
            prop_assert!(used <= capacity.min(BUFFER_SIZE));
            let kind = if which == 0 || which == 3 { SessionKind::Session } else { SessionKind::Domain };
            prop_assert!(Message::parse_request(&buf, kind).is_ok());
        } else {
            prop_assert_eq!(buf, before);
        }
    }
}
