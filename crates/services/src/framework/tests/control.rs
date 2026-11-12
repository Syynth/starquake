//! Control requests: convert to domain, copy from domain, clone, and query pointer buffer size.

use starquake_ipc::{CommandType, RequestBuilder, SFCO_MAGIC};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::{KernelError, ResultCode};

use super::super::control::{
    CLONE_CURRENT_OBJECT, CLONE_CURRENT_OBJECT_EX, CONTROL_COMMANDS, CONTROL_INTERFACE,
    CONVERT_CURRENT_OBJECT_TO_DOMAIN, COPY_FROM_CURRENT_DOMAIN, QUERY_POINTER_BUFFER_SIZE,
};
use super::super::result::{
    DOMAIN_OBJECT_NOT_FOUND, INVALID_CMIF_REQUEST, NOT_SUPPORTED, TARGET_NOT_DOMAIN,
    UNKNOWN_COMMAND_ID,
};
use super::super::targets::Domain;
use super::super::{DEFAULT_POINTER_BUFFER_SIZE, Protocol, Server, StubbedCommand, UnknownPolicy};
use super::{
    Answer, Rig, Toy, buffer_of, child, control_request, domain_request, in_domain, only, payload,
    plain, request, toy,
};

const ROOT: u32 = Domain::ROOT_OBJECT_ID;

/// Sends a control request that returns a session, and returns that session.
fn new_session(rig: &mut Rig, on: ObjectId, build: &RequestBuilder) -> ObjectId {
    let reply = plain(&rig.send_on(on, build));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!((reply.copies.len(), reply.moves.len()), (0, 1));
    assert_eq!(reply.data, [0_u8; 0]);
    rig.object_of(reply.moves[0])
}

fn set_tag(rig: &mut Rig, child: ObjectId, tag: u32) {
    let set = request(child::SET_TAG).data(&tag.to_le_bytes());
    assert_eq!(plain(&rig.send_on(child, &set)), only(ResultCode::SUCCESS));
}

#[test]
fn convert_replies_the_root_object_id_in_these_bytes() {
    let mut rig = Rig::strict();
    let buffer = rig.send(&control_request(CONVERT_CURRENT_OBJECT_TO_DOMAIN));
    // An ordinary CMIF reply (no domain header) whose data is the u32 object id.
    let success = ResultCode::SUCCESS.raw();
    assert_eq!(
        buffer,
        buffer_of(&[0, 7, 0, 0, SFCO_MAGIC, 0, success, 0, ROOT])
    );

    // From now on requests carry a domain header, and object 1 is the toy.
    let numbers = payload(|w| {
        w.write_u32(2).align(8).write_u64(3);
    });
    let reply = in_domain(&rig.send(&domain_request(ROOT, toy::ADD).data(&numbers)));
    assert_eq!(reply.data, 5_u64.to_le_bytes());
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (1, 1)
    );
}

/// Divergence from Eden, which replies success and resets the domain to its first handler.
#[test]
fn converting_a_domain_again_is_not_supported_and_changes_nothing() {
    let mut rig = Rig::domain();
    let again = plain(&rig.send(&control_request(CONVERT_CURRENT_OBJECT_TO_DOMAIN)));
    assert_eq!(again, only(NOT_SUPPORTED));
    let reply = in_domain(&rig.send(&domain_request(ROOT, toy::INSPECT)));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (1, 1)
    );
}

#[test]
fn query_pointer_buffer_size_replies_a_u16_in_these_bytes() {
    const SIZE: u16 = 0x1234;
    let mut rig = Rig::strict();
    let buffer = rig.send(&control_request(QUERY_POINTER_BUFFER_SIZE));
    let success = ResultCode::SUCCESS.raw();
    let size = u32::from(DEFAULT_POINTER_BUFFER_SIZE);
    assert_eq!(
        buffer,
        buffer_of(&[0, 7, 0, 0, SFCO_MAGIC, 0, success, 0, size])
    );

    // A server can be told another size; on a domain the reply still has no domain header.
    let mut rig = Rig::strict();
    let session = rig.kernel.new_object();
    let mut server = Server::new("sized", UnknownPolicy::Strict).with_pointer_buffer_size(SIZE);
    server.open(session, Box::new(Toy::default())).unwrap();
    rig.server = server;
    rig.session = session;
    rig.convert();
    let buffer = rig.send(&control_request(QUERY_POINTER_BUFFER_SIZE));
    assert_eq!(
        buffer,
        buffer_of(&[0, 7, 0, 0, SFCO_MAGIC, 0, success, 0, u32::from(SIZE)])
    );
}

#[test]
fn control_with_context_is_served_like_control() {
    let mut rig = Rig::strict();
    let query = RequestBuilder::cmif(CommandType::ControlWithContext, QUERY_POINTER_BUFFER_SIZE);
    let reply = plain(&rig.send(&query));
    assert_eq!(reply.data[..2], DEFAULT_POINTER_BUFFER_SIZE.to_le_bytes());
}

#[test]
fn a_clone_leads_to_the_same_interface_and_outlives_the_original() {
    let mut rig = Rig::strict();
    let child = rig.open_child(1);
    for (clone_command, tag) in [
        (CLONE_CURRENT_OBJECT, 0x51),
        (CLONE_CURRENT_OBJECT_EX, 0x52),
    ] {
        // Command 4 carries a tag word, which is ignored.
        let build = control_request(clone_command).data(&0xFFFF_FFFF_u32.to_le_bytes());
        let clone = new_session(&mut rig, child, &build);
        assert_ne!(clone, child);
        // One child, reached through two sessions.
        assert_eq!(rig.server.interface_count(), 2);
        set_tag(&mut rig, clone, tag);
        assert_eq!(rig.child_tag(child), tag);
        rig.server.close(clone).unwrap();
        assert_eq!(rig.child_tag(child), tag);
    }

    let clone = new_session(&mut rig, child, &control_request(CLONE_CURRENT_OBJECT));
    assert_eq!(rig.server.session_count(), 3);
    rig.server.close(child).unwrap();
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 2)
    );
    assert_eq!(rig.child_tag(clone), 0x52);
    rig.server.close(clone).unwrap();
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (1, 1)
    );
}

#[test]
fn a_clone_of_a_domain_shares_the_domain() {
    let mut rig = Rig::domain();
    let root = rig.session;
    let clone = new_session(&mut rig, root, &control_request(CLONE_CURRENT_OBJECT));
    // An object opened through the clone is visible through the original, under the same id.
    let open = domain_request(ROOT, toy::OPEN_CHILD).data(&0x77_u32.to_le_bytes());
    let object = in_domain(&rig.send_on(clone, &open)).objects[0];
    let reply = in_domain(&rig.send(&domain_request(object, child::TAG)));
    assert_eq!(reply.data, 0x77_u32.to_le_bytes());
    // The domain lives until both sessions are closed.
    rig.server.close(rig.session).unwrap();
    assert_eq!(rig.server.interface_count(), 2);
    let reply = in_domain(&rig.send_on(clone, &domain_request(object, child::TAG)));
    assert_eq!(reply.data, 0x77_u32.to_le_bytes());
    rig.server.close(clone).unwrap();
    assert_eq!(rig.server.interface_count(), 0);
}

/// Divergence from Eden, where a session and its clones share one domain flag.
#[test]
fn converting_a_session_leaves_its_plain_clone_plain() {
    let mut rig = Rig::strict();
    let root = rig.session;
    let clone = new_session(&mut rig, root, &control_request(CLONE_CURRENT_OBJECT));
    rig.convert();
    // The original now needs domain headers; the clone still takes ordinary requests, and both
    // reach the one toy.
    let through_domain = in_domain(&rig.send(&domain_request(ROOT, toy::INSPECT)));
    assert_eq!(through_domain.result, ResultCode::SUCCESS.raw());
    let through_clone = plain(&rig.send_on(clone, &request(toy::INSPECT)));
    assert_eq!(through_clone.result, ResultCode::SUCCESS.raw());
    assert_eq!(through_clone.data, through_domain.data);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 1)
    );
}

#[test]
fn copy_from_domain_opens_a_session_on_one_object() {
    let mut rig = Rig::domain();
    let open = domain_request(ROOT, toy::OPEN_CHILD).data(&0x21_u32.to_le_bytes());
    let object = in_domain(&rig.send(&open)).objects[0];

    let copy = control_request(COPY_FROM_CURRENT_DOMAIN).data(&object.to_le_bytes());
    let root = rig.session;
    let session = new_session(&mut rig, root, &copy);
    // The new session is an ordinary one, and leads to the very object in the domain.
    assert_eq!(rig.child_tag(session), 0x21);
    set_tag(&mut rig, session, 0x22);
    let through_domain = in_domain(&rig.send(&domain_request(object, child::TAG)));
    assert_eq!(through_domain.data, 0x22_u32.to_le_bytes());
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 2)
    );

    // Closing the domain's id leaves the session's object alive; closing the session drops it.
    let closed = in_domain(&rig.send(&RequestBuilder::domain_close(object)));
    assert_eq!(closed, only(ResultCode::SUCCESS));
    assert_eq!(rig.child_tag(session), 0x22);
    rig.server.close(session).unwrap();
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (1, 1)
    );
}

#[test]
fn copy_from_domain_needs_a_domain_an_object_and_its_argument() {
    let mut rig = Rig::strict();
    let copy = |id: u32| control_request(COPY_FROM_CURRENT_DOMAIN).data(&id.to_le_bytes());
    assert_eq!(plain(&rig.send(&copy(ROOT))), only(TARGET_NOT_DOMAIN));
    rig.convert();
    for id in [0, ROOT + 1, u32::MAX] {
        let reply = plain(&rig.send(&copy(id)));
        assert_eq!(reply, only(DOMAIN_OBJECT_NOT_FOUND), "object id {id}");
    }
    // No object id in the payload.
    let reply = plain(&rig.send(&control_request(COPY_FROM_CURRENT_DOMAIN)));
    assert_eq!(reply, only(INVALID_CMIF_REQUEST));
    assert_eq!(rig.server.session_count(), 1);
    assert_eq!(rig.kernel.sessions(), []);
}

#[test]
fn a_clone_that_gets_no_handle_is_not_opened() {
    let mut rig = Rig::with_handle_table(UnknownPolicy::Strict, 1);
    let root = rig.session;
    new_session(&mut rig, root, &control_request(CLONE_CURRENT_OBJECT));
    let reply = plain(&rig.send(&control_request(CLONE_CURRENT_OBJECT)));
    assert_eq!(reply, only(KernelError::OutOfHandles.code()));
    assert_eq!(rig.server.session_count(), 2);
    assert_eq!(rig.kernel.sessions().len(), 1);
}

fn control_result(rig: &mut Rig, id: u32) -> Answer {
    // Enough payload for the one command that reads an argument.
    plain(&rig.send(&control_request(id).data(&ROOT.to_le_bytes())))
}

#[test]
fn the_listed_control_commands_are_answered_and_any_other_id_is_unknown() {
    assert!(CONTROL_COMMANDS.is_distinct());
    assert_eq!(CONTROL_COMMANDS.tipc, []);
    let listed: Vec<u32> = CONTROL_COMMANDS.cmif.iter().map(|c| c.id).collect();
    for id in &listed {
        // A fresh domain each time, so every command (copy from domain too) can succeed.
        let mut rig = Rig::domain();
        let reply = control_result(&mut rig, *id);
        let expected = if *id == CONVERT_CURRENT_OBJECT_TO_DOMAIN {
            NOT_SUPPORTED
        } else {
            ResultCode::SUCCESS
        };
        assert_eq!(reply.result, expected.raw(), "control command {id}");
    }
    // The first id past the table (Eden's set pointer buffer size) and a far one.
    let first_unlisted = listed.iter().max().unwrap() + 1;
    for id in [first_unlisted, u32::MAX] {
        assert!(CONTROL_COMMANDS.name(Protocol::Cmif, id).is_none());
        let mut rig = Rig::strict();
        assert_eq!(control_result(&mut rig, id), only(UNKNOWN_COMMAND_ID));
        assert_eq!(rig.server.stubbed().count(), 0);

        let mut rig = Rig::new(UnknownPolicy::AutoStub);
        assert_eq!(control_result(&mut rig, id), only(ResultCode::SUCCESS));
        let stubbed: Vec<_> = rig.server.stubbed().copied().collect();
        let command = StubbedCommand {
            interface: CONTROL_INTERFACE,
            protocol: Protocol::Cmif,
            id,
        };
        assert_eq!(stubbed, [command]);
    }
}
