//! `sm:` driven through raw messages, laid out the way libnx lays them out
//! (`libnx/nx/source/services/sm.c`), on a host with a second service to hand out.

use starquake_ipc::{
    BUFFER_SIZE, Body, CommandHeader, CommandType, Message, ReceiveList, RequestBuilder,
    SFCO_MAGIC, SessionKind, SpecialHeader, TIPC_COMMAND_OFFSET,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_trace::{EventKind, IpcOutcome, ServiceId};

use super::*;
use crate::framework::fake::{FakeHost, FakeKernel};
use crate::framework::result::{INVALID_CMIF_REQUEST, UNKNOWN_COMMAND_ID};
use crate::framework::{Handled, Kernel, Protocol, StubbedCommand};
use crate::set;

/// The numbers the two services of the test host are known by.
const SM: ServiceId = ServiceId(1);
const SET_SYS: ServiceId = ServiceId(2);

/// `sm:` command ids (`eden/src/core/hle/service/sm/sm.cpp`, `SM::SM`).
const INITIALIZE: u16 = 0;
const GET_SERVICE: u16 = 1;
const REGISTER_SERVICE: u16 = 2;

/// What fills a message buffer before a request is built into it.
const STALE: u8 = 0xA5;

fn name(text: &str) -> ServiceName {
    ServiceName::new(text).unwrap()
}

/// A host with `sm:` and `set:sys`, and the caller's handle to an `sm:` session.
fn host_with(policy: UnknownPolicy, handle_table_size: i32) -> (FakeHost, Handle) {
    let mut registry = Registry::new();
    registry.register(name(set::SYS_NAME), SET_SYS).unwrap();
    let settings = set::Settings {
        firmware: set::FirmwareVersion::eden_default(),
    };
    let mut host = FakeHost::new(FakeKernel::new(handle_table_size).unwrap());
    host.add(server(registry, policy).with_service_id(SM));
    host.add(set::sys_server(settings, policy).with_service_id(SET_SYS));
    let sm = host.connect(SM).unwrap();
    (host, sm)
}

fn host(policy: UnknownPolicy) -> (FakeHost, Handle) {
    host_with(policy, 0)
}

/// Sends a request and returns the buffer, which must hold a reply.
fn send(host: &mut FakeHost, session: Handle, request: &RequestBuilder) -> Vec<u8> {
    let mut buffer = vec![STALE; BUFFER_SIZE];
    request.build(&mut buffer).unwrap();
    assert_eq!(host.send(session, &mut buffer), Ok(Handled::Replied));
    buffer
}

/// A whole message buffer that starts with `words` and is zero after them.
fn buffer_of(words: &[u32]) -> Vec<u8> {
    let mut buffer: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert!(buffer.len() <= BUFFER_SIZE);
    buffer.resize(BUFFER_SIZE, 0);
    buffer
}

/// A CMIF reply that holds only a result: the header (6 raw words), padding to 16 bytes,
/// `SFCO`, version, result, token.
fn cmif_result(result: ResultCode) -> Vec<u8> {
    buffer_of(&[0, 6, 0, 0, SFCO_MAGIC, 0, result.raw(), 0])
}

/// A TIPC reply to command `id` that holds only a result.
fn tipc_result(id: u16, result: ResultCode) -> Vec<u8> {
    buffer_of(&[u32::from(TIPC_COMMAND_OFFSET + id), 1, result.raw()])
}

/// The header words of a reply that moves one handle: the command header with the special
/// header announced, and the special header counting one moved handle.
fn moved_handle_header(command_type: CommandType, raw_data_words: u32) -> [u32; 3] {
    let header = CommandHeader {
        command_type,
        num_x: 0,
        num_a: 0,
        num_b: 0,
        num_w: 0,
        raw_data_words,
        receive_list: ReceiveList::Disabled,
        has_special_header: true,
    };
    let special = SpecialHeader {
        send_pid: false,
        num_copy_handles: 0,
        num_move_handles: 1,
    };
    let [w0, w1] = header.encode().unwrap();
    [w0, w1, special.encode().unwrap()]
}

/// The one handle a reply moved.
fn moved_handle(buffer: &[u8]) -> Handle {
    let message = Message::parse_reply(buffer, SessionKind::Session).unwrap();
    assert_eq!(message.move_handles.len(), 1);
    assert_eq!(message.copy_handles, Vec::<u32>::new());
    Handle::from_raw(message.move_handles[0])
}

/// `Initialize` as libnx sends it over CMIF (`_smCmifCmdInPid`): the process id in the header
/// and a `u64` placeholder in the payload.
fn cmif_initialize() -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, u32::from(INITIALIZE))
        .send_pid(0)
        .data(&0_u64.to_le_bytes())
}

/// `GetService` as libnx sends it (`smGetServiceOriginal`): the name field.
fn cmif_get_service(field: [u8; MAX_NAME_LEN]) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, u32::from(GET_SERVICE)).data(&field)
}

fn initialize(host: &mut FakeHost, sm: Handle) {
    assert_eq!(
        send(host, sm, &cmif_initialize()),
        cmif_result(ResultCode::SUCCESS)
    );
}

/// The firmware version `set:sys` reports to a session, read through command 4.
fn firmware_version(host: &mut FakeHost, set_sys: Handle) -> Vec<u8> {
    const AT: u64 = 0x7000;
    host.kernel.map(AT, vec![STALE; set::FIRMWARE_VERSION_SIZE]);
    let room = starquake_ipc::ReceiveDescriptor {
        address: AT,
        size: u16::try_from(set::FIRMWARE_VERSION_SIZE).unwrap(),
    };
    let ask = RequestBuilder::cmif(CommandType::Request, 4).receive_descriptor(room);
    assert_eq!(send(host, set_sys, &ask), cmif_result(ResultCode::SUCCESS));
    host.kernel
        .memory(AT, set::FIRMWARE_VERSION_SIZE)
        .unwrap()
        .to_vec()
}

#[test]
fn get_service_hands_out_a_session_on_the_named_service() {
    let (mut host, sm) = host(UnknownPolicy::Strict);
    initialize(&mut host, sm);
    let reply = send(
        &mut host,
        sm,
        &cmif_get_service(name(set::SYS_NAME).to_field()),
    );
    let handle = moved_handle(&reply);
    // Byte for byte: the header with one moved handle, the handle, then `SFCO`, version,
    // result, token, already on a 16-byte boundary. No payload.
    let [w0, w1, special] = moved_handle_header(CommandType::Invalid, 4);
    let success = ResultCode::SUCCESS.raw();
    let expected = [w0, w1, special, handle.raw(), SFCO_MAGIC, 0, success, 0];
    assert_eq!(reply, buffer_of(&expected));

    // The handle is a session on the `set:sys` server, opened with its root interface.
    let session = host.kernel.resolve_handle(handle).unwrap();
    assert_eq!(host.kernel.owner(session), Some(SET_SYS));
    assert_eq!(host.server(SET_SYS).unwrap().session_count(), 1);
    assert_eq!(host.server(SM).unwrap().session_count(), 1);
    // And it answers as `set:sys`.
    assert_eq!(
        firmware_version(&mut host, handle),
        set::FirmwareVersion::eden_default().to_bytes()
    );

    // Asking again gives another session on the same server.
    let again = send(
        &mut host,
        sm,
        &cmif_get_service(name(set::SYS_NAME).to_field()),
    );
    assert_ne!(moved_handle(&again), handle);
    assert_eq!(host.server(SET_SYS).unwrap().session_count(), 2);
}

#[test]
fn the_tipc_forms_do_the_same() {
    let (mut host, sm) = host(UnknownPolicy::Strict);
    // `_smTipcCmdInPid`: the process id in the header, no payload.
    let init = RequestBuilder::tipc(INITIALIZE).send_pid(0);
    assert_eq!(
        send(&mut host, sm, &init),
        tipc_result(INITIALIZE, ResultCode::SUCCESS)
    );
    let ask = RequestBuilder::tipc(GET_SERVICE).data(&name(set::SYS_NAME).to_field());
    let reply = send(&mut host, sm, &ask);
    let handle = moved_handle(&reply);
    // The reply echoes the request's type; its raw data is the result word alone.
    let reply_type = CommandType::Tipc(GET_SERVICE);
    let [w0, w1, special] = moved_handle_header(reply_type, 1);
    let expected = [w0, w1, special, handle.raw(), ResultCode::SUCCESS.raw()];
    assert_eq!(reply, buffer_of(&expected));
    assert_eq!(
        firmware_version(&mut host, handle),
        set::FirmwareVersion::eden_default().to_bytes()
    );
}

#[test]
fn initialize_over_either_protocol_initialises_the_session_for_both() {
    for tipc_first in [false, true] {
        let (mut host, sm) = host(UnknownPolicy::Strict);
        let field = name(set::SYS_NAME).to_field();
        let cmif = cmif_get_service(field);
        let tipc = RequestBuilder::tipc(GET_SERVICE).data(&field);
        if tipc_first {
            send(&mut host, sm, &RequestBuilder::tipc(INITIALIZE).send_pid(0));
        } else {
            initialize(&mut host, sm);
        }
        moved_handle(&send(&mut host, sm, &cmif));
        moved_handle(&send(&mut host, sm, &tipc));
    }
}

#[test]
fn get_service_before_initialize_is_an_invalid_client() {
    let (mut host, sm) = host(UnknownPolicy::Strict);
    let ask = cmif_get_service(name(set::SYS_NAME).to_field());
    assert_eq!(send(&mut host, sm, &ask), cmif_result(INVALID_CLIENT));
    // Checked before the name is read: a request with no name fails the same way.
    let nameless = RequestBuilder::tipc(GET_SERVICE);
    assert_eq!(
        send(&mut host, sm, &nameless),
        tipc_result(GET_SERVICE, INVALID_CLIENT)
    );
    assert_eq!(host.server(SET_SYS).unwrap().session_count(), 0);

    // Another session's `Initialize` does not count for this one.
    let other = host.connect(SM).unwrap();
    initialize(&mut host, other);
    assert_eq!(send(&mut host, sm, &ask), cmif_result(INVALID_CLIENT));
    moved_handle(&send(&mut host, other, &ask));
}

#[test]
fn a_cloned_session_shares_its_initialised_state() {
    use crate::framework::control::CLONE_CURRENT_OBJECT;
    let (mut host, sm) = host(UnknownPolicy::Strict);
    let clone = RequestBuilder::cmif(CommandType::Control, CLONE_CURRENT_OBJECT);
    let cloned = moved_handle(&send(&mut host, sm, &clone));
    // Initialised through the clone, used through the original.
    initialize(&mut host, cloned);
    let ask = cmif_get_service(name(set::SYS_NAME).to_field());
    moved_handle(&send(&mut host, sm, &ask));
}

#[test]
fn a_name_is_its_printable_bytes_wherever_they_are() {
    let (mut host, sm) = host(UnknownPolicy::Strict);
    initialize(&mut host, sm);
    // Eden's `PopServiceName` drops every byte outside ' '..='~' and keeps the rest in order:
    // a zero byte in the middle of the field does not end the name.
    let scattered = [b's', b'e', 0, b't', b':', b's', b'y', b's'];
    let reply = send(&mut host, sm, &cmif_get_service(scattered));
    let session = host.kernel.resolve_handle(moved_handle(&reply)).unwrap();
    assert_eq!(host.kernel.owner(session), Some(SET_SYS));
    // "set:sy" is not "set:sys".
    let short = name("set:sy").to_field();
    assert_eq!(
        send(&mut host, sm, &cmif_get_service(short)),
        cmif_result(NOT_REGISTERED)
    );

    // Nothing printable at all is an invalid name, not an unknown one.
    for empty in [
        [0; MAX_NAME_LEN],
        [0x7F; MAX_NAME_LEN],
        [0x1F; MAX_NAME_LEN],
    ] {
        assert_eq!(
            send(&mut host, sm, &cmif_get_service(empty)),
            cmif_result(INVALID_SERVICE_NAME)
        );
    }
    // A name field that is cut short is a malformed request. (A TIPC payload is whole words,
    // so the longest payload that is too short is one word less than the field.)
    let cut = RequestBuilder::tipc(GET_SERVICE).data(&[b'a'; MAX_NAME_LEN - size_of::<u32>()]);
    assert_eq!(
        send(&mut host, sm, &cut),
        tipc_result(GET_SERVICE, INVALID_CMIF_REQUEST)
    );
    let registry = host.server(SM).unwrap().family::<Registry>().unwrap();
    assert_eq!(
        registry.unknown_names().collect::<Vec<_>>(),
        [&name("set:sy")]
    );
}

/// Divergence from Eden, which defers the request until the name is registered.
#[test]
fn an_unknown_name_is_an_error_under_strict() {
    let (mut host, sm) = host(UnknownPolicy::Strict);
    initialize(&mut host, sm);
    let ask = cmif_get_service(name("fsp-srv").to_field());
    assert_eq!(send(&mut host, sm, &ask), cmif_result(NOT_REGISTERED));
    assert_eq!(send(&mut host, sm, &ask), cmif_result(NOT_REGISTERED));
    let sm_server = host.server(SM).unwrap();
    // No session and no interface was made for it, and the name is on record once.
    assert_eq!(
        (sm_server.session_count(), sm_server.interface_count()),
        (1, 1)
    );
    let registry = sm_server.family::<Registry>().unwrap();
    assert_eq!(
        registry.unknown_names().collect::<Vec<_>>(),
        [&name("fsp-srv")]
    );
    assert_eq!(host.kernel.handle_count(), 1);
}

/// Divergence from Eden, as above: the service spec's rule for bringing up a title.
#[test]
fn an_unknown_name_gets_a_stub_under_auto_stub() {
    let (mut host, sm) = host(UnknownPolicy::AutoStub);
    initialize(&mut host, sm);
    let mut stubs = Vec::new();
    for unknown in ["hid", "fsp-srv"] {
        let reply = send(&mut host, sm, &cmif_get_service(name(unknown).to_field()));
        stubs.push(moved_handle(&reply));
    }
    // Each is a session of the `sm:` server itself.
    let session = host.kernel.resolve_handle(stubs[0]).unwrap();
    assert_eq!(host.kernel.owner(session), Some(SM));
    assert_eq!(host.server(SM).unwrap().session_count(), 3);
    let registry = host.server(SM).unwrap().family::<Registry>().unwrap();
    assert_eq!(
        registry.unknown_names().collect::<Vec<_>>(),
        [&name("fsp-srv"), &name("hid")]
    );

    // Whatever is sent to a stub is answered with success and nothing else, in either protocol.
    let cmif = RequestBuilder::cmif(CommandType::Request, 301).data(&[0xEE; 12]);
    assert_eq!(
        send(&mut host, stubs[0], &cmif),
        cmif_result(ResultCode::SUCCESS)
    );
    assert_eq!(
        send(&mut host, stubs[1], &RequestBuilder::tipc(9)),
        tipc_result(9, ResultCode::SUCCESS)
    );
    let stubbed: Vec<_> = host.server(SM).unwrap().stubbed().copied().collect();
    let stub = |protocol, id| StubbedCommand {
        interface: "Stub",
        protocol,
        id,
    };
    assert_eq!(
        stubbed,
        [stub(Protocol::Cmif, 301), stub(Protocol::Tipc, 9)]
    );
    // The trace marks the two requests as stubbed.
    let outcomes: Vec<_> = (host.kernel.recorded().iter())
        .filter_map(|event| match event.kind {
            EventKind::Ipc { outcome, .. } => Some(outcome),
            _ => None,
        })
        .collect();
    assert_eq!(
        outcomes[outcomes.len() - 2..],
        [IpcOutcome::Stubbed, IpcOutcome::Stubbed]
    );

    // A registered name still gets the real service.
    let real = send(
        &mut host,
        sm,
        &cmif_get_service(name(set::SYS_NAME).to_field()),
    );
    let real = host.kernel.resolve_handle(moved_handle(&real)).unwrap();
    assert_eq!(host.kernel.owner(real), Some(SET_SYS));
}

/// Divergence from Eden, whose TIPC reply carries a moved handle slot holding zero.
#[test]
fn a_failed_tipc_get_service_carries_no_handle() {
    let (mut host, sm) = host(UnknownPolicy::Strict);
    send(&mut host, sm, &RequestBuilder::tipc(INITIALIZE).send_pid(0));
    let ask = RequestBuilder::tipc(GET_SERVICE).data(&name("nope").to_field());
    let reply = send(&mut host, sm, &ask);
    assert_eq!(reply, tipc_result(GET_SERVICE, NOT_REGISTERED));
    let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
    assert_eq!(message.move_handles, Vec::<u32>::new());
    assert!(matches!(message.body, Body::TipcReply { .. }));
}

/// Divergence from Eden, which creates the session and then fails to add its handle.
#[test]
fn a_full_handle_table_gets_no_session() {
    // Room for the `sm:` session's handle only.
    let (mut host, sm) = host_with(UnknownPolicy::Strict, 1);
    initialize(&mut host, sm);
    let ask = cmif_get_service(name(set::SYS_NAME).to_field());
    assert_eq!(
        send(&mut host, sm, &ask),
        cmif_result(KernelError::OutOfHandles.code())
    );
    assert_eq!(host.server(SET_SYS).unwrap().session_count(), 0);
    assert_eq!(host.kernel.handle_count(), 1);
}

#[test]
fn commands_that_are_left_out_go_to_the_unknown_command_policy() {
    let register = RequestBuilder::cmif(CommandType::Request, u32::from(REGISTER_SERVICE));
    let (mut host, sm) = host(UnknownPolicy::Strict);
    assert_eq!(
        send(&mut host, sm, &register),
        cmif_result(UNKNOWN_COMMAND_ID)
    );
    let (mut host, sm) = host_with(UnknownPolicy::AutoStub, 0);
    assert_eq!(
        send(&mut host, sm, &register),
        cmif_result(ResultCode::SUCCESS)
    );
    let stubbed: Vec<_> = host.server(SM).unwrap().stubbed().copied().collect();
    assert_eq!(stubbed.len(), 1);
    assert_eq!(
        (stubbed[0].interface, stubbed[0].id),
        ("Sm", u32::from(REGISTER_SERVICE))
    );
}

#[test]
fn a_restored_sm_carries_on() {
    let (mut host, sm) = host(UnknownPolicy::AutoStub);
    initialize(&mut host, sm);
    let stub = moved_handle(&send(
        &mut host,
        sm,
        &cmif_get_service(name("hid").to_field()),
    ));
    let saved = host.server(SM).unwrap().save().unwrap();

    // A new `sm:` server with an empty registry takes the saved one's place.
    let mut fresh = server(Registry::new(), UnknownPolicy::AutoStub).with_service_id(SM);
    fresh.restore(&saved).unwrap();
    host.add(fresh);
    // The session is still initialised, the registry is back, and the stub still answers.
    let reply = send(
        &mut host,
        sm,
        &cmif_get_service(name(set::SYS_NAME).to_field()),
    );
    let set_sys = host.kernel.resolve_handle(moved_handle(&reply)).unwrap();
    assert_eq!(host.kernel.owner(set_sys), Some(SET_SYS));
    assert_eq!(
        send(&mut host, stub, &RequestBuilder::tipc(3)),
        tipc_result(3, ResultCode::SUCCESS)
    );
    let registry = host.server(SM).unwrap().family::<Registry>().unwrap();
    assert_eq!(registry.unknown_names().collect::<Vec<_>>(), [&name("hid")]);
    assert_eq!(registry.names().collect::<Vec<_>>(), [&name(set::SYS_NAME)]);
}

#[test]
fn the_state_a_snapshot_holds_is_these_bytes() {
    use crate::framework::{Family, Session};
    let mut registry = Registry::new();
    registry.register(name("set:sys"), ServiceId(0x21)).unwrap();
    registry.register(name("a"), ServiceId(0x22)).unwrap();
    assert!(registry.note_unknown(&name("hid")));
    // `postcard`: two services in name order, each a length-prefixed name, the provider's
    // variant (0, an HLE server) and its number; then the unknown names.
    #[rustfmt::skip]
    let golden = [
        2,
        1, b'a',                                      0, 0x22,
        7, b's', b'e', b't', b':', b's', b'y', b's',  0, 0x21,
        1,
        3, b'h', b'i', b'd',
    ];
    assert_eq!(Family::save_state(&registry).unwrap(), golden);
    let mut loaded = Registry::new();
    loaded.load_state(&golden).unwrap();
    assert_eq!(loaded, registry);

    // A session's state: whether it is initialised.
    assert_eq!(Sm { initialized: false }.save().unwrap(), [0]);
    assert_eq!(Sm { initialized: true }.save().unwrap(), [1]);
    // A stub's: the name it stands in for.
    let stub = Stub { name: name("hid") };
    assert_eq!(stub.save().unwrap(), [3, b'h', b'i', b'd']);
}

#[test]
fn a_saved_name_is_checked_when_it_is_read_back() {
    let read = |bytes: &[u8]| postcard::from_bytes::<ServiceName>(bytes);
    assert_eq!(read(&[3, b'h', b'i', b'd']), Ok(name("hid")));
    // Empty, nine bytes, and a byte that is not printable.
    for bad in [
        &[0][..],
        &[9, b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i'],
        &[1, 0x7F],
    ] {
        assert_eq!(read(bad), Err(postcard::Error::SerdeDeCustom));
    }
}

#[test]
fn a_name_is_one_to_eight_printable_bytes() {
    assert_eq!(name("sm:").as_str(), "sm:");
    assert_eq!(name("sm:").to_string(), "sm:");
    // Exactly at the limit, and one past it.
    let longest = "a".repeat(MAX_NAME_LEN);
    assert_eq!(ServiceName::new(&longest).unwrap().as_str(), longest);
    let too_long = "a".repeat(MAX_NAME_LEN + 1);
    assert_eq!(
        ServiceName::new(&too_long),
        Err(NameError::TooLong(too_long.clone()))
    );
    assert_eq!(ServiceName::new(""), Err(NameError::Empty));
    // The edges of the printable range are names; the bytes just outside are not.
    assert!(ServiceName::new(" ~").is_ok());
    for outside in ["\u{1F}", "\u{7F}", "é"] {
        assert_eq!(
            ServiceName::new(outside),
            Err(NameError::NotPrintable(outside.to_owned()))
        );
    }
    assert_eq!(
        NameError::TooLong(too_long.clone()).to_string(),
        format!("service name {too_long:?} is longer than {MAX_NAME_LEN} bytes")
    );

    // The field form is the bytes and then zeros, and reads back as the name.
    assert_eq!(name("hid").to_field(), [b'h', b'i', b'd', 0, 0, 0, 0, 0]);
    assert_eq!(name(&longest).to_field(), [b'a'; MAX_NAME_LEN]);
    assert_eq!(
        ServiceName::from_field(name("hid").to_field()),
        Some(name("hid"))
    );
    assert_eq!(
        ServiceName::from_field([b' '; MAX_NAME_LEN]),
        Some(name("        "))
    );
    assert_eq!(ServiceName::from_field([0; MAX_NAME_LEN]), None);
    // Bytes just outside the printable range, a zero and a high byte are dropped where they are.
    let scattered = [0x7F, b'h', 0x1F, b'i', 0, b'd', 0xFF, b'~'];
    assert_eq!(ServiceName::from_field(scattered), Some(name("hid~")));
}

#[test]
fn a_name_is_registered_once() {
    let mut registry = Registry::new();
    registry.register(name("hid"), ServiceId(5)).unwrap();
    assert_eq!(
        registry.register(name("hid"), ServiceId(6)),
        Err(RegistryError::AlreadyRegistered(name("hid")))
    );
    assert_eq!(
        registry.provider(&name("hid")),
        Some(Provider::Hle(ServiceId(5)))
    );
    assert_eq!(registry.provider(&name("hid:")), None);
    assert_eq!(
        RegistryError::AlreadyRegistered(name("hid")).to_string(),
        "a service is already registered as hid"
    );
}

#[test]
fn the_record_of_unknown_names_stops_exactly_at_its_limit() {
    let mut registry = Registry::new();
    let unknown = |n: usize| name(&format!("u{n}"));
    for n in 0..MAX_UNKNOWN_NAMES {
        assert!(registry.note_unknown(&unknown(n)), "{n}");
    }
    // A name already on record is not new, and one past the limit is not recorded.
    assert!(!registry.note_unknown(&unknown(0)));
    assert!(!registry.note_unknown(&unknown(MAX_UNKNOWN_NAMES)));
    assert_eq!(registry.unknown_names().count(), MAX_UNKNOWN_NAMES);
}

#[test]
fn result_codes_are_edens_words() {
    // Golden words: `ErrorModule::SM` is 21 (0x15) in bits 0..9, the description from bit 9.
    // Horizon prints them 2021-0002, 2021-0006 and 2021-0007.
    assert_eq!(INVALID_CLIENT.raw(), 0x415);
    assert_eq!(INVALID_SERVICE_NAME.raw(), 0xC15);
    assert_eq!(NOT_REGISTERED.raw(), 0xE15);
    for code in [INVALID_CLIENT, INVALID_SERVICE_NAME, NOT_REGISTERED] {
        assert_eq!(code.module(), SM_MODULE);
    }
}
