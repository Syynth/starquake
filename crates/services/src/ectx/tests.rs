//! `ectx:aw` driven through raw messages over a fake kernel.

use starquake_ipc::{
    BUFFER_SIZE, Body, CommandType, Message, RequestBuilder, RequestLayout, SessionKind,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;

use super::command::aw::CREATE_CONTEXT_REGISTRAR;
use super::*;
use crate::framework::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
use crate::framework::fake::FakeKernel;
use crate::framework::result::UNKNOWN_COMMAND_ID;
use crate::framework::{Handled, Kernel, Protocol, Session, StubbedCommand};

/// `IContextRegistrar::Complete`, which this port leaves out, so it has no shipping constant
/// (`eden/src/core/hle/service/glue/ectx.cpp`: `{0, &IContextRegistrar::Complete,
/// "Complete"}`).
const COMPLETE: u32 = 0;
/// `ECTX_AW::CommitContext`, which Eden lists without a handler (`{1, nullptr,
/// "CommitContext"}`).
const COMMIT_CONTEXT: u32 = 1;

/// What fills the message buffer beforehand.
const STALE: u8 = 0xA5;

/// An `ectx:aw` server, a kernel, and one session.
struct Rig {
    kernel: FakeKernel,
    server: Server,
    session: ObjectId,
}

impl Rig {
    fn new(policy: UnknownPolicy) -> Self {
        let mut kernel = FakeKernel::new(0).unwrap();
        let session = kernel.new_object();
        let mut server = aw_server(policy);
        server.accept(session).unwrap();
        Self {
            kernel,
            server,
            session,
        }
    }

    fn send_raw(&mut self, session: ObjectId, request: &RequestBuilder) -> Vec<u8> {
        let mut buffer = vec![STALE; BUFFER_SIZE];
        request.build(&mut buffer).unwrap();
        let handled = self.server.handle(&mut self.kernel, session, &mut buffer);
        assert_eq!(handled, Ok(Handled::Replied));
        buffer
    }

    /// The result, data and moved handles of the reply to `request`.
    fn send(
        &mut self,
        session: ObjectId,
        request: &RequestBuilder,
    ) -> (ResultCode, Vec<u8>, Vec<u32>) {
        let reply = self.send_raw(session, request);
        let message = Message::parse_reply(&reply, SessionKind::Session).unwrap();
        let Body::CmifReply(body) = &message.body else {
            unreachable!("not a CMIF reply: {:?}", message.body);
        };
        (
            ResultCode::from_raw(body.header.result),
            body.data.to_vec(),
            message.move_handles.clone(),
        )
    }

    /// Sends `CreateContextRegistrar` and returns the registrar's session.
    fn registrar(&mut self) -> ObjectId {
        let (result, _, moves) = self.send(self.session, &request(CREATE_CONTEXT_REGISTRAR));
        assert_eq!(result, ResultCode::SUCCESS);
        self.kernel
            .resolve_handle(Handle::from_raw(moves[0]))
            .unwrap()
    }
}

fn request(command: u32) -> RequestBuilder {
    RequestBuilder::cmif(CommandType::Request, command).layout(RequestLayout::Libnx)
}

/// `Complete` as Eden reads it: a `u32` in the payload.
fn complete() -> RequestBuilder {
    request(COMPLETE).data(&7u32.to_le_bytes())
}

#[test]
fn create_context_registrar_hands_out_a_registrar_that_answers() {
    // Under the auto-stub policy, the registrar answers `Complete` as Eden does (success and a
    // zero word after the result), and the stub is recorded against the registrar, not the
    // factory.
    let mut rig = Rig::new(UnknownPolicy::AutoStub);
    let registrar = rig.registrar();
    assert_eq!(rig.kernel.sessions(), [registrar]);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 2)
    );
    let (result, data, _) = rig.send(registrar, &complete());
    assert_eq!(result, ResultCode::SUCCESS);
    assert!(data.iter().all(|byte| *byte == 0), "{data:?}");
    let stubbed: Vec<_> = rig.server.stubbed().copied().collect();
    assert_eq!(
        stubbed,
        [StubbedCommand {
            interface: "IContextRegistrar",
            protocol: Protocol::Cmif,
            id: COMPLETE,
        }]
    );
}

#[test]
fn the_registrar_is_not_the_factory() {
    // Under the strict policy, the factory's command 0 sent to the registrar is unknown there.
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let registrar = rig.registrar();
    let (result, _, moves) = rig.send(registrar, &request(CREATE_CONTEXT_REGISTRAR));
    assert_eq!((result, moves.len()), (UNKNOWN_COMMAND_ID, 0));
    assert_eq!(rig.server.session_count(), 2);
}

#[test]
fn create_context_registrar_in_a_domain_returns_an_object() {
    let mut rig = Rig::new(UnknownPolicy::AutoStub);
    let convert = RequestBuilder::cmif(CommandType::Control, CONVERT_CURRENT_OBJECT_TO_DOMAIN);
    let converted = rig.send_raw(rig.session, &convert);
    let root = u32::from_le_bytes(converted[32..36].try_into().unwrap());
    let create = RequestBuilder::domain(CommandType::Request, root, CREATE_CONTEXT_REGISTRAR);
    let reply = rig.send_raw(rig.session, &create);
    let message = Message::parse_reply(&reply, SessionKind::Domain).unwrap();
    let Body::CmifReply(body) = &message.body else {
        unreachable!("not a CMIF reply");
    };
    assert_eq!(body.header.result, ResultCode::SUCCESS.raw());
    let object = body.object_ids[0];
    assert_eq!(object, root + 1);
    assert_eq!(rig.kernel.sessions(), []);
    // The object answers, as a registrar.
    let complete =
        RequestBuilder::domain(CommandType::Request, object, COMPLETE).data(&7u32.to_le_bytes());
    let reply = rig.send_raw(rig.session, &complete);
    let message = Message::parse_reply(&reply, SessionKind::Domain).unwrap();
    let Body::CmifReply(body) = &message.body else {
        unreachable!("not a CMIF reply");
    };
    assert_eq!(body.header.result, ResultCode::SUCCESS.raw());
    let interfaces: Vec<_> = rig.server.stubbed().map(|stub| stub.interface).collect();
    assert_eq!(interfaces, ["IContextRegistrar"]);
}

#[test]
fn commands_that_are_left_out_go_to_the_unknown_command_policy() {
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let (result, _, _) = rig.send(rig.session, &request(COMMIT_CONTEXT));
    assert_eq!(result, UNKNOWN_COMMAND_ID);
    let registrar = rig.registrar();
    assert_eq!(rig.send(registrar, &complete()).0, UNKNOWN_COMMAND_ID);
}

#[test]
fn a_server_with_a_registrar_open_saves_and_restores() {
    let mut rig = Rig::new(UnknownPolicy::Strict);
    let registrar = rig.registrar();
    let saved = rig.server.save().unwrap();
    let mut restored = aw_server(UnknownPolicy::Strict);
    restored.restore(&saved).unwrap();
    assert_eq!(restored.save().unwrap(), saved);
    // The restored server's registrar is a registrar, and its root still makes them.
    rig.server = restored;
    assert_eq!(rig.send(registrar, &complete()).0, UNKNOWN_COMMAND_ID);
    let another = rig.registrar();
    assert_ne!(another, registrar);
    assert_eq!(rig.server.session_count(), 3);
}

#[test]
fn the_tables_name_the_commands() {
    assert_eq!(
        EctxAw
            .commands()
            .name(Protocol::Cmif, CREATE_CONTEXT_REGISTRAR),
        Some("create_context_registrar")
    );
    assert_eq!(IContextRegistrar.commands().cmif, []);
    assert_eq!(EctxAw::interface(), "EctxAw");
    assert_eq!(IContextRegistrar::interface(), "IContextRegistrar");
    assert_eq!(AW_NAME, "ectx:aw");
}
