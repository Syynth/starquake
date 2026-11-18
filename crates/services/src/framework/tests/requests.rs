//! What a server counts of the requests it is sent ([`Server::requests`](super::super::Server::requests)),
//! and what the trace event of a request names: the interface it went to and its protocol.

use starquake_ipc::{CommandType, RequestBuilder};
use starquake_kernel::result::ResultCode;
use starquake_trace::{EventKind, InterfaceName, IpcOutcome, IpcProtocol};

use super::super::control::{CONTROL_COMMANDS, CONTROL_INTERFACE, QUERY_POINTER_BUFFER_SIZE};
use super::super::result::TARGET_NOT_FOUND;
use super::super::targets::Domain;
use super::super::{
    CountedCommand, Handled, MAX_COUNTED_COMMANDS, Protocol, Reply, RequestCount, ServerError,
    Session, UnknownPolicy,
};
use super::{Child, Rig, Toy, child, control_request, domain_request, parked_one, request, toy};

/// The name of the interface `T`.
fn name_of<T: Session>() -> InterfaceName {
    InterfaceName::new(T::interface()).unwrap()
}

/// What the toy's table calls a command.
fn toy_name(protocol: Protocol, id: u32) -> Option<&'static str> {
    Toy::default().commands().name(protocol, id)
}

/// `count` requests for `id` on `interface` in `protocol`, named `name`.
fn counted(
    interface: InterfaceName,
    protocol: IpcProtocol,
    id: u32,
    name: Option<&'static str>,
    count: u64,
) -> RequestCount {
    RequestCount {
        command: CountedCommand {
            interface,
            protocol,
            id,
        },
        name,
        count,
    }
}

/// The interface, protocol, command and result of every event the rig's kernel recorded.
fn events(rig: &Rig) -> Vec<(InterfaceName, IpcProtocol, u32, u32, IpcOutcome)> {
    (rig.kernel.recorded().iter())
        .filter_map(|event| match event.kind {
            EventKind::Ipc {
                interface,
                protocol,
                command,
                result,
                outcome,
                ..
            } => Some((interface, protocol, command, result.0, outcome)),
            _ => None,
        })
        .collect()
}

#[test]
fn each_request_is_counted_under_its_interface_protocol_and_command() {
    let mut rig = Rig::strict();
    rig.send(&request(toy::ADD).data(&[0; 16]));
    rig.send(&request(toy::INSPECT));
    rig.send(&request(toy::ADD).data(&[0; 16]));
    rig.send(&RequestBuilder::tipc(toy::TIPC_INSPECT));
    rig.send(&control_request(QUERY_POINTER_BUFFER_SIZE));
    // A command the toy does not have: answered (by the strict policy) and counted, unnamed.
    rig.send(&request(u32::from(toy::MISSING)));

    let toy = name_of::<Toy>();
    let control = InterfaceName::new(CONTROL_INTERFACE).unwrap();
    let missing = u32::from(toy::MISSING);
    let tipc_inspect = u32::from(toy::TIPC_INSPECT);
    // In order of interface, protocol (CMIF, TIPC, control) and id.
    assert_eq!(
        rig.server.requests().collect::<Vec<_>>(),
        [
            counted(
                control,
                IpcProtocol::Control,
                QUERY_POINTER_BUFFER_SIZE,
                CONTROL_COMMANDS.name(Protocol::Cmif, QUERY_POINTER_BUFFER_SIZE),
                1
            ),
            counted(
                toy,
                IpcProtocol::Cmif,
                toy::ADD,
                toy_name(Protocol::Cmif, toy::ADD),
                2
            ),
            counted(
                toy,
                IpcProtocol::Cmif,
                toy::INSPECT,
                toy_name(Protocol::Cmif, toy::INSPECT),
                1
            ),
            counted(toy, IpcProtocol::Cmif, missing, None, 1),
            counted(
                toy,
                IpcProtocol::Tipc,
                tipc_inspect,
                toy_name(Protocol::Tipc, tipc_inspect),
                1
            ),
        ]
    );
    // The names are the tables' own: the TIPC one is not the CMIF command with that id.
    assert!(toy_name(Protocol::Tipc, tipc_inspect).is_some());
    assert_ne!(
        toy_name(Protocol::Tipc, tipc_inspect),
        toy_name(Protocol::Cmif, tipc_inspect)
    );
    assert_eq!(rig.server.uncounted_requests(), 0);
    // Each counted request is one event, under the same interface and protocol.
    let success = ResultCode::SUCCESS.raw();
    let done = IpcOutcome::Done;
    assert_eq!(
        events(&rig),
        [
            (toy, IpcProtocol::Cmif, toy::ADD, success, done),
            (toy, IpcProtocol::Cmif, toy::INSPECT, success, done),
            (toy, IpcProtocol::Cmif, toy::ADD, success, done),
            (toy, IpcProtocol::Tipc, tipc_inspect, success, done),
            (
                control,
                IpcProtocol::Control,
                QUERY_POINTER_BUFFER_SIZE,
                success,
                done
            ),
            (
                toy,
                IpcProtocol::Cmif,
                missing,
                super::super::UNKNOWN_COMMAND_ID.raw(),
                done
            ),
        ]
    );
}

#[test]
fn a_domain_request_names_the_object_it_went_to() {
    let mut rig = Rig::domain();
    let open = domain_request(Domain::ROOT_OBJECT_ID, toy::OPEN_CHILD).data(&5_u32.to_le_bytes());
    rig.send(&open);
    let child_id = Domain::ROOT_OBJECT_ID + 1;
    rig.send(&domain_request(child_id, child::TAG));
    // An id that names nothing: refused, and named after the session's own interface (the
    // domain's root object).
    let empty = child_id + 1;
    rig.send(&domain_request(empty, child::TAG));
    // A close-object request names the object it closes; it has no command id.
    rig.send(&RequestBuilder::domain_close(child_id));

    let (toy, child) = (name_of::<Toy>(), name_of::<Child>());
    let control = InterfaceName::new(CONTROL_INTERFACE).unwrap();
    let (success, done) = (ResultCode::SUCCESS.raw(), IpcOutcome::Done);
    let convert = super::super::control::CONVERT_CURRENT_OBJECT_TO_DOMAIN;
    assert_eq!(
        events(&rig),
        [
            (control, IpcProtocol::Control, convert, success, done),
            (toy, IpcProtocol::Cmif, toy::OPEN_CHILD, success, done),
            (child, IpcProtocol::Cmif, child::TAG, success, done),
            (
                toy,
                IpcProtocol::Cmif,
                child::TAG,
                TARGET_NOT_FOUND.raw(),
                done
            ),
            (
                child,
                IpcProtocol::Cmif,
                super::super::NO_COMMAND_ID,
                success,
                done
            ),
        ]
    );
    let child_tag = Child { tag: 0 }.commands().name(Protocol::Cmif, child::TAG);
    let counts: Vec<_> = rig.server.requests().collect();
    assert!(counts.contains(&counted(child, IpcProtocol::Cmif, child::TAG, child_tag, 1)));
    // The refused request is counted under the root, as its event is, but with no name: the
    // root's table names that id (the toy's own command with the same id), and the request did
    // not go to the root.
    assert!(toy_name(Protocol::Cmif, child::TAG).is_some());
    assert!(counts.contains(&counted(toy, IpcProtocol::Cmif, child::TAG, None, 1)));
}

/// A `ControlWithContext` request (type 7, which nnsdk titles send) is a control request like
/// type 5: named [`CONTROL_INTERFACE`], counted under the control table's name.
#[test]
fn a_control_request_with_context_is_a_control_request() {
    let mut rig = Rig::strict();
    rig.send(&RequestBuilder::cmif(
        CommandType::ControlWithContext,
        QUERY_POINTER_BUFFER_SIZE,
    ));
    let control = InterfaceName::new(CONTROL_INTERFACE).unwrap();
    assert_eq!(
        events(&rig),
        [(
            control,
            IpcProtocol::Control,
            QUERY_POINTER_BUFFER_SIZE,
            ResultCode::SUCCESS.raw(),
            IpcOutcome::Done
        )]
    );
    assert_eq!(
        rig.server.requests().collect::<Vec<_>>(),
        [counted(
            control,
            IpcProtocol::Control,
            QUERY_POINTER_BUFFER_SIZE,
            CONTROL_COMMANDS.name(Protocol::Cmif, QUERY_POINTER_BUFFER_SIZE),
            1
        )]
    );
    // The control table's name, which the toy's own table does not share for that id.
    assert!(
        CONTROL_COMMANDS
            .name(Protocol::Cmif, QUERY_POINTER_BUFFER_SIZE)
            .is_some()
    );
    assert_ne!(
        CONTROL_COMMANDS.name(Protocol::Cmif, QUERY_POINTER_BUFFER_SIZE),
        toy_name(Protocol::Cmif, QUERY_POINTER_BUFFER_SIZE)
    );
}

#[test]
fn a_domain_whose_root_is_closed_names_nothing() {
    let mut rig = Rig::domain();
    rig.send(&RequestBuilder::domain_close(Domain::ROOT_OBJECT_ID));
    // Neither the object named nor the root is there.
    rig.send(&domain_request(Domain::ROOT_OBJECT_ID, toy::INSPECT));
    let nothing = InterfaceName::lossy("");
    assert_eq!(nothing.as_str(), "?");
    let last = *events(&rig).last().unwrap();
    assert_eq!(
        last,
        (
            nothing,
            IpcProtocol::Cmif,
            toy::INSPECT,
            TARGET_NOT_FOUND.raw(),
            IpcOutcome::Done
        )
    );
}

#[test]
fn a_deferred_request_is_counted_once_when_it_is_deferred() {
    let mut rig = Rig::strict();
    rig.defer_on(rig.session, &RequestBuilder::tipc(toy::TIPC_WAIT));
    let tipc_wait = u32::from(toy::TIPC_WAIT);
    let expected = [counted(
        name_of::<Toy>(),
        IpcProtocol::Tipc,
        tipc_wait,
        toy_name(Protocol::Tipc, tipc_wait),
        1,
    )];
    // Counted before it has an event.
    assert_eq!(rig.server.requests().collect::<Vec<_>>(), expected);
    assert_eq!(events(&rig), []);
    let mut reply = Reply::default();
    reply.payload().write_u32(1);
    parked_one().complete(reply);
    rig.deliver();
    // Not counted again when its reply is written; its event names it as it was sent.
    assert_eq!(rig.server.requests().collect::<Vec<_>>(), expected);
    assert_eq!(
        events(&rig),
        [(
            name_of::<Toy>(),
            IpcProtocol::Tipc,
            tipc_wait,
            ResultCode::SUCCESS.raw(),
            IpcOutcome::Deferred
        )]
    );
}

#[test]
fn a_deferred_domain_request_keeps_the_object_it_went_to() {
    let mut rig = Rig::domain();
    let open = domain_request(Domain::ROOT_OBJECT_ID, toy::OPEN_CHILD).data(&5_u32.to_le_bytes());
    rig.send(&open);
    let child_id = Domain::ROOT_OBJECT_ID + 1;
    rig.defer_on(rig.session, &domain_request(child_id, child::WAIT));
    drop(parked_one());
    rig.deliver();
    let last = *events(&rig).last().unwrap();
    assert_eq!(
        last,
        (
            name_of::<Child>(),
            IpcProtocol::Cmif,
            child::WAIT,
            super::super::DEFERRED_REPLY_DROPPED.raw(),
            IpcOutcome::Deferred
        )
    );
}

#[test]
fn a_close_and_a_failed_message_are_not_counted() {
    let mut rig = Rig::strict();
    let session = rig.session;
    let (closed, _) = rig.try_send_on(session, &RequestBuilder::close());
    assert_eq!(closed, Ok(Handled::Closed));
    let (failed, _) = rig.try_send_on(session, &request(toy::INSPECT));
    assert_eq!(failed, Err(ServerError::UnknownSession(session)));
    assert_eq!(rig.server.requests().count(), 0);
    assert_eq!(rig.server.uncounted_requests(), 0);
}

#[test]
fn the_counts_stop_exactly_at_their_bound() {
    let mut rig = Rig::new(UnknownPolicy::AutoStub);
    let first_unknown = 0x1000_u32;
    let bound = u32::try_from(MAX_COUNTED_COMMANDS).unwrap();
    // Distinct commands up to the bound, each counted.
    for id in first_unknown..first_unknown + bound {
        rig.send(&request(id));
    }
    assert_eq!(rig.server.requests().count(), MAX_COUNTED_COMMANDS);
    assert_eq!(rig.server.uncounted_requests(), 0);
    // One distinct command past it: answered, not listed, counted as uncounted.
    rig.send(&request(first_unknown + bound));
    assert_eq!(rig.server.requests().count(), MAX_COUNTED_COMMANDS);
    assert_eq!(rig.server.uncounted_requests(), 1);
    // A command already listed is still counted in place.
    rig.send(&request(first_unknown));
    let first = rig
        .server
        .requests()
        .find(|count| count.command.id == first_unknown)
        .unwrap();
    assert_eq!(first.count, 2);
    assert_eq!(rig.server.uncounted_requests(), 1);
    assert!(
        rig.server
            .requests()
            .all(|count| count.command.id != first_unknown + bound)
    );
}
