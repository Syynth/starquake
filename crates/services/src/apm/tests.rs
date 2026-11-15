//! `apm` driven through raw messages laid out as libnx lays them out
//! (`libnx/nx/source/services/apm.c`, `_apmCmdGetSession`).

use starquake_ipc::{
    BUFFER_SIZE, CommandHeader, CommandType, Message, ReceiveList, RequestBuilder, RequestLayout,
    SFCO_MAGIC, SessionKind, SpecialHeader,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;

use super::*;
use crate::framework::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
use crate::framework::fake::FakeKernel;
use crate::framework::result::UNKNOWN_COMMAND_ID;
use crate::framework::{Handled, Kernel, Session};

use super::command::apm::OPEN_SESSION;

/// `apm`'s `GetPerformanceMode`, which this port leaves out, so it has no shipping constant
/// (`eden/src/core/hle/service/apm/apm_interface.cpp`: `{1, &APM::GetPerformanceMode, ..}`).
const GET_PERFORMANCE_MODE: u32 = 1;
/// `ISession::SetPerformanceConfiguration`, likewise left out (`{0,
/// &ISession::SetPerformanceConfiguration, ..}`).
const SET_PERFORMANCE_CONFIGURATION: u32 = 0;

/// What fills the message buffer beforehand.
const STALE: u8 = 0xA5;

/// An `apm` server, a kernel, and one session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

impl Rig {
    fn new() -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        let session = kernel.new_object();
        let mut server = server(UnknownPolicy::Strict);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
        }
    }

    fn send_to(&mut self, session: ObjectId, request: &RequestBuilder) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    fn send(&mut self, request: &RequestBuilder) -> Vec<u8> {
        self.send_to(self.session, request)
    }
}

/// A whole message buffer that starts with `words` and is zero after them.
fn buffer_of(words: &[u32]) -> Vec<u8> {
    let mut buffer: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    buffer.resize(BUFFER_SIZE, 0);
    buffer
}

/// A CMIF reply that holds only a result.
fn only(result: ResultCode) -> Vec<u8> {
    buffer_of(&[0, 6, 0, 0, SFCO_MAGIC, 0, result.raw(), 0])
}

/// A request laid out as libnx lays one out (`cmifMakeRequest`), with no payload.
fn libnx_request(command: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, command).layout(RequestLayout::Libnx)
}

#[test]
fn open_session_hands_out_a_session_and_the_reply_is_these_bytes() {
    let mut rig = Rig::new();
    let reply = rig.send(&libnx_request(OPEN_SESSION));
    let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
    let handle = Handle::from_raw(message.move_handles[0]);
    // Byte for byte: the header with one moved handle, the handle, then `SFCO`, version,
    // result, token, already on a 16-byte boundary. No payload.
    let header = CommandHeader {
        command_type: CommandType::Invalid,
        num_x: 0,
        num_a: 0,
        num_b: 0,
        num_w: 0,
        raw_data_words: 4,
        receive_list: ReceiveList::Disabled,
        has_special_header: true,
    };
    let special = SpecialHeader {
        send_pid: false,
        num_copy_handles: 0,
        num_move_handles: 1,
    };
    let [w0, w1] = header.encode().unwrap();
    let success = ResultCode::SUCCESS.raw();
    let expected = [
        w0,
        w1,
        special.encode().unwrap(),
        handle.raw(),
        SFCO_MAGIC,
        0,
        success,
        0,
    ];
    assert_eq!(reply, buffer_of(&expected));

    // The handle is a new session on this server, and it is an `ISession`: its commands are
    // `ISession`'s (none ported, so a strict server refuses them), not `apm`'s.
    let session = rig.kernel.resolve_handle(handle).unwrap();
    assert_eq!(rig.kernel.sessions(), [session]);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 2)
    );
    let again = rig.send_to(session, &libnx_request(OPEN_SESSION));
    assert_eq!(again, only(UNKNOWN_COMMAND_ID));
}

#[test]
fn open_session_in_a_domain_returns_an_object() {
    let mut rig = Rig::new();
    let convert = RequestBuilder::cmif(CommandType::Control, CONVERT_CURRENT_OBJECT_TO_DOMAIN);
    let converted = rig.send(&convert);
    let root = u32::from_le_bytes(converted[32..36].try_into().unwrap());
    let reply = rig.send(&RequestBuilder::domain(
        CommandType::Request,
        root,
        OPEN_SESSION,
    ));
    // Header (11 raw words), padding, the domain header counting one object, `SFCO`, then the
    // object id after the (empty) data.
    let success = ResultCode::SUCCESS.raw();
    let object = root + 1;
    let expected = [0, 11, 0, 0, 1, 0, 0, 0, SFCO_MAGIC, 0, success, 0, object];
    assert_eq!(reply, buffer_of(&expected));
    assert_eq!(rig.kernel.sessions(), []);
    assert_eq!(rig.server.interface_count(), 2);
}

#[test]
fn commands_that_are_left_out_go_to_the_unknown_command_policy() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.send(&libnx_request(GET_PERFORMANCE_MODE)),
        only(UNKNOWN_COMMAND_ID)
    );
    let reply = rig.send(&libnx_request(OPEN_SESSION));
    let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
    let session = rig
        .kernel
        .resolve_handle(Handle::from_raw(message.move_handles[0]))
        .unwrap();
    let configure = libnx_request(SET_PERFORMANCE_CONFIGURATION).data(&[0; 8]);
    assert_eq!(rig.send_to(session, &configure), only(UNKNOWN_COMMAND_ID));
}

#[test]
fn the_tables_name_the_commands_as_eden_does() {
    let open = Apm.commands().name(crate::framework::Protocol::Cmif, OPEN_SESSION);
    assert_eq!(open, Some("open_session"));
    assert_eq!(Apm::interface(), "Apm");
    assert_eq!(ISession::interface(), "ISession");
    assert!(ISession.commands().cmif.is_empty());
}

#[test]
fn a_server_with_a_session_open_saves_and_restores() {
    let mut rig = Rig::new();
    rig.send(&libnx_request(OPEN_SESSION));
    let saved = rig.server.save().unwrap();
    let mut restored = server(UnknownPolicy::Strict);
    restored.restore(&saved).unwrap();
    assert_eq!(
        (restored.session_count(), restored.interface_count()),
        (2, 2)
    );
    assert_eq!(restored.save().unwrap(), saved);
}

#[test]
fn the_performance_mode_follows_docking_with_edens_numbers() {
    // `enum class PerformanceMode : s32 { Invalid = -1, Normal = 0, Boost = 1 }`.
    assert_eq!(PerformanceMode::for_docked(true), PerformanceMode::Boost);
    assert_eq!(PerformanceMode::for_docked(false), PerformanceMode::Normal);
    assert_eq!(PerformanceMode::Normal.raw(), 0);
    assert_eq!(PerformanceMode::Boost.raw(), 1);
}
