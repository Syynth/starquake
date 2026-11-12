//! The server's own surface: opening and closing sessions, close messages, messages it cannot
//! parse, and message types it does not serve.

use starquake_ipc::{
    BUFFER_SIZE, CommandHeader, CommandType, IpcError, RequestBuilder, SFCI_MAGIC,
};
use starquake_kernel::result::ResultCode;

use starquake_kernel::result::KernelError;
use starquake_trace::ServiceId;

use super::super::result::NOT_SUPPORTED;
use super::super::targets::Domain;
use super::super::{Handled, Kernel, Server, ServerError, UnknownPolicy};
use super::{
    Child, Rig, SERVICE, Tally, Toy, built, domain_request, in_domain, only, plain, request, toy,
    toy_interfaces,
};

#[test]
fn a_server_reports_what_it_was_made_with() {
    let server = Server::new("set:sys", UnknownPolicy::AutoStub);
    assert_eq!(server.name(), "set:sys");
    assert_eq!(server.policy(), UnknownPolicy::AutoStub);
    assert_eq!((server.session_count(), server.interface_count()), (0, 0));
    assert_eq!(server.stubbed().count(), 0);
}

#[test]
fn a_server_is_known_by_the_number_it_was_given() {
    assert_eq!(
        Server::new("toy", UnknownPolicy::Strict).service_id(),
        ServiceId(0)
    );
    assert_eq!(Rig::strict().server.service_id(), SERVICE);
}

#[test]
fn an_accepted_session_gets_a_root_interface_of_its_own() {
    let mut rig = Rig::strict();
    let (first, second) = (rig.session, rig.second_session());
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 2)
    );
    // Each has a toy of its own: closing one session leaves the other's working.
    rig.server.close(first).unwrap();
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (1, 1)
    );
    assert_eq!(
        plain(&rig.send_on(second, &request(toy::INSPECT))).result,
        ResultCode::SUCCESS.raw()
    );
    // Accepting the same session twice is opening it twice.
    assert_eq!(
        rig.server.accept(second),
        Err(ServerError::SessionExists(second))
    );

    let mut rootless = Server::new("rootless", UnknownPolicy::Strict);
    assert_eq!(rootless.accept(second), Err(ServerError::NoRoot));
    assert_eq!(rootless.session_count(), 0);
    assert_eq!(
        ServerError::NoRoot.to_string(),
        "the server has no root interface to open for a new session"
    );
}

#[test]
fn a_family_s_state_is_shared_by_its_sessions() {
    let mut rig = Rig::strict();
    let (first, second) = (rig.session, rig.second_session());
    let bump = request(toy::BUMP);
    let count = |buffer: &[u8]| u32::from_le_bytes(plain(buffer).data.try_into().unwrap());
    assert_eq!(count(&rig.send_on(first, &bump)), 1);
    assert_eq!(count(&rig.send_on(second, &bump)), 2);
    assert_eq!(count(&rig.send_on(first, &bump)), 3);
    assert_eq!(rig.server.family::<Tally>(), Some(&Tally { bumps: 3 }));
    // State of another type is not there.
    assert_eq!(rig.server.family::<u32>(), None);
}

/// starquake's own choice: a method that asks for family state its server was not given fails
/// with "not supported".
#[test]
fn a_method_that_needs_family_state_fails_on_a_server_without_it() {
    let mut rig = Rig::strict();
    let bare = Server::new("toy", UnknownPolicy::Strict).with_interfaces(toy_interfaces());
    rig.server = bare;
    rig.server
        .open(rig.session, Box::new(Toy::default()))
        .unwrap();
    assert_eq!(plain(&rig.send(&request(toy::BUMP))), only(NOT_SUPPORTED));
    assert_eq!(rig.server.family::<Tally>(), None);

    // The same on a server whose family state is of another type.
    let other = Server::new("toy", UnknownPolicy::Strict).with_family(7_u32);
    rig.server = other;
    rig.server
        .open(rig.session, Box::new(Toy::default()))
        .unwrap();
    assert_eq!(plain(&rig.send(&request(toy::BUMP))), only(NOT_SUPPORTED));
    assert_eq!(rig.server.family::<u32>(), Some(&7));
}

#[test]
fn a_pushed_service_is_a_moved_handle_to_a_session_on_that_service() {
    const OTHER: ServiceId = ServiceId(0x99);
    let lend = request(toy::LEND).data(&OTHER.0.to_le_bytes());
    let mut rig = Rig::strict();
    // A service the kernel does not know: the kernel's error, and nothing is handed out.
    assert_eq!(plain(&rig.send(&lend)), only(KernelError::NotFound.code()));
    assert_eq!(rig.kernel.take_connections(), []);

    rig.kernel.add_service(OTHER);
    let reply = plain(&rig.send(&lend));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    // After the moved event, in the order the method pushed them.
    assert_eq!(reply.moves.len(), 2);
    let session = rig.object_of(reply.moves[1]);
    assert_eq!(rig.kernel.take_connections(), [(OTHER, session)]);
    assert_eq!(rig.kernel.owner(session), Some(OTHER));
    // The session is on the other service's server, not on this one.
    assert_eq!(rig.server.session_count(), 1);
    assert_eq!(rig.kernel.sessions(), []);
}

#[test]
fn a_pushed_service_is_a_handle_in_a_domain_too() {
    const OTHER: ServiceId = ServiceId(0x99);
    let mut rig = Rig::domain();
    rig.kernel.add_service(OTHER);
    let lend = domain_request(Domain::ROOT_OBJECT_ID, toy::LEND).data(&OTHER.0.to_le_bytes());
    let reply = in_domain(&rig.send(&lend));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!((reply.moves.len(), reply.objects.len()), (2, 0));
    let session = rig.object_of(reply.moves[1]);
    assert_eq!(rig.kernel.take_connections(), [(OTHER, session)]);
}

#[test]
fn a_pushed_service_needs_room_for_its_handle() {
    const OTHER: ServiceId = ServiceId(0x99);
    // Room for the moved event and not for the session.
    let mut rig = Rig::with_handle_table(UnknownPolicy::Strict, 1);
    rig.kernel.add_service(OTHER);
    let lend = request(toy::LEND).data(&OTHER.0.to_le_bytes());
    assert_eq!(
        plain(&rig.send(&lend)),
        only(KernelError::OutOfHandles.code())
    );
    // No session was made for a handle that did not fit.
    assert_eq!(rig.kernel.take_connections(), []);
    assert_eq!(rig.kernel.handle_count(), 1);
    // With room for both, the session is made.
    let mut rig = Rig::with_handle_table(UnknownPolicy::Strict, 2);
    rig.kernel.add_service(OTHER);
    assert_eq!(plain(&rig.send(&lend)).moves.len(), 2);
    assert_eq!(rig.kernel.take_connections().len(), 1);
    let filler = rig.kernel.new_object();
    assert_eq!(
        rig.kernel.insert_handle(filler),
        Err(KernelError::OutOfHandles)
    );
}

#[test]
fn a_server_can_move_to_another_core_thread() {
    fn assert_send<T: Send>() {}
    assert_send::<Server>();
}

#[test]
fn a_session_is_opened_once_and_closed_once() {
    let mut rig = Rig::strict();
    let session = rig.session;
    let again = rig.server.open(session, Box::new(Toy::default()));
    assert_eq!(again, Err(ServerError::SessionExists(session)));
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (1, 1)
    );

    let other = rig.kernel.new_object();
    rig.server.open(other, Box::new(Child { tag: 5 })).unwrap();
    assert_eq!(rig.child_tag(other), 5);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 2)
    );

    rig.server.close(other).unwrap();
    assert_eq!(
        rig.server.close(other),
        Err(ServerError::UnknownSession(other))
    );
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (1, 1)
    );
}

#[test]
fn a_message_on_a_session_the_server_does_not_have_is_an_error() {
    let mut rig = Rig::strict();
    let stranger = rig.kernel.new_object();
    let ask = request(toy::INSPECT);
    let (handled, buffer) = rig.try_send_on(stranger, &ask);
    assert_eq!(handled, Err(ServerError::UnknownSession(stranger)));
    assert_eq!(buffer, built(&ask));
}

/// Divergence from Eden, which writes a success reply: a close message closes the session and
/// writes nothing.
#[test]
fn a_close_message_closes_the_session_and_writes_nothing() {
    for close in [RequestBuilder::close(), RequestBuilder::tipc_close()] {
        let mut rig = Rig::strict();
        let session = rig.session;
        let child = rig.open_child(1);
        let (handled, buffer) = rig.try_send_on(session, &close);
        assert_eq!(handled, Ok(Handled::Closed));
        assert_eq!(buffer, built(&close));
        // The toy went with its session; the child has a session of its own.
        assert_eq!(
            (rig.server.session_count(), rig.server.interface_count()),
            (1, 1)
        );
        assert_eq!(rig.child_tag(child), 1);

        let (handled, _) = rig.try_send_on(session, &request(toy::INSPECT));
        assert_eq!(handled, Err(ServerError::UnknownSession(session)));
    }
}

#[test]
fn a_message_that_cannot_be_parsed_is_an_error_and_the_buffer_is_untouched() {
    let mut rig = Rig::strict();
    let mut buffer = built(&request(toy::INSPECT));
    // Spoil the `SFCI` magic, which sits at the first 16-byte boundary after the header.
    let magic_at = CommandHeader::WORDS.next_multiple_of(4) * size_of::<u32>();
    assert_eq!(buffer[magic_at..magic_at + 4], SFCI_MAGIC.to_le_bytes());
    buffer[magic_at] ^= 0xFF;
    let spoiled = buffer.clone();
    let handled = rig.server.handle(&mut rig.kernel, rig.session, &mut buffer);
    let found = SFCI_MAGIC ^ 0xFF;
    let bad_magic = IpcError::BadMagic {
        found,
        expected: SFCI_MAGIC,
    };
    assert_eq!(handled, Err(ServerError::Malformed(bad_magic.clone())));
    assert_eq!(buffer, spoiled);
    assert_eq!(
        ServerError::Malformed(bad_magic).to_string(),
        format!("malformed request: bad magic {found:#010x}, expected {SFCI_MAGIC:#010x}")
    );

    // The session is still there and still works.
    assert_eq!(
        plain(&rig.send(&request(toy::INSPECT))).result,
        ResultCode::SUCCESS.raw()
    );
}

#[test]
fn a_buffer_shorter_than_a_message_buffer_is_refused_before_anything_happens() {
    let mut rig = Rig::strict();
    // The request itself is only a few words long and would parse; it would open a child.
    let ask = RequestBuilder::tipc(toy::TIPC_OPEN_CHILD).data(&1_u32.to_le_bytes());
    let mut buffer = built(&ask)[..BUFFER_SIZE - 1].to_vec();
    let sent = buffer.clone();
    let handled = rig.server.handle(&mut rig.kernel, rig.session, &mut buffer);
    let too_short = ServerError::BufferTooShort {
        len: BUFFER_SIZE - 1,
    };
    assert_eq!(handled, Err(too_short.clone()));
    assert_eq!(
        too_short.to_string(),
        format!(
            "message buffer holds {} bytes, fewer than the {BUFFER_SIZE} a message needs",
            BUFFER_SIZE - 1
        )
    );
    assert_eq!(buffer, sent);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (1, 1)
    );
    assert_eq!(rig.kernel.sessions(), []);
}

#[test]
fn only_the_first_buffer_size_bytes_are_the_message() {
    let mut rig = Rig::strict();
    let mut buffer = built(&request(toy::INSPECT));
    buffer.extend_from_slice(&[0x5A; 64]);
    let handled = rig.server.handle(&mut rig.kernel, rig.session, &mut buffer);
    assert_eq!(handled, Ok(Handled::Replied));
    assert_eq!(plain(&buffer).result, ResultCode::SUCCESS.raw());
    assert_eq!(buffer[BUFFER_SIZE..], [0x5A; 64]);
}

/// Divergence from Eden, which logs a failed assertion and writes no reply.
#[test]
fn legacy_message_types_are_not_supported() {
    let mut rig = Rig::strict();
    for legacy in [CommandType::LegacyRequest, CommandType::LegacyControl] {
        let reply = plain(&rig.send(&RequestBuilder::cmif(legacy, toy::INSPECT)));
        assert_eq!(reply, only(NOT_SUPPORTED), "{legacy:?}");
    }
}
