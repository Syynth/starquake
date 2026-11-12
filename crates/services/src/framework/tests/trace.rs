//! The trace sink: every answered request is one event in the golden-trace format, marked when
//! it was stubbed or deferred (`docs/specs/services.md`, "Tracing").

use starquake_ipc::{Body, Message, RequestBuilder, SessionKind};
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;
use starquake_trace::{EventKind, GuestTicks, IpcOutcome, ThreadId};

use super::super::control::{CONVERT_CURRENT_OBJECT_TO_DOMAIN, QUERY_POINTER_BUFFER_SIZE};
use super::super::result::{INVALID_CMIF_REQUEST, UNKNOWN_COMMAND_ID};
use super::super::targets::Domain;
use super::super::trace::session_id;
use super::super::{NO_COMMAND_ID, Reply, UnknownPolicy};
use super::{
    Rig, SERVICE, built, control_request, domain_request, parked_one, payload, request, toy,
};

/// The guest thread and core the requests of these tests come from.
const THREAD: ThreadId = ThreadId(0x77);
const CORE: u8 = 4;

/// The bytes of arguments a request carries, as the IPC crate parses them back out of the
/// message: what a method's reader is given.
fn argument_bytes(request: &RequestBuilder, kind: SessionKind) -> u32 {
    let buffer = built(request);
    let message = Message::parse_request(&buffer, kind).unwrap();
    let data = match &message.body {
        Body::CmifRequest(body) => body.data,
        Body::TipcRequest { data } => *data,
        other => unreachable!("not a request with arguments: {other:?}"),
    };
    u32::try_from(data.len()).unwrap()
}

/// The event for a request on `session`.
fn ipc(
    session: ObjectId,
    command: u32,
    request_size: u32,
    reply_size: u32,
    result: ResultCode,
    outcome: IpcOutcome,
) -> EventKind {
    EventKind::Ipc {
        session: session_id(session),
        service: SERVICE,
        command,
        request_size,
        reply_size,
        result: starquake_trace::ResultCode(result.raw()),
        outcome,
    }
}

fn kinds(rig: &Rig) -> Vec<EventKind> {
    rig.kernel.recorded().iter().map(|e| e.kind).collect()
}

#[test]
fn a_request_is_one_line_of_the_golden_trace() {
    let mut rig = Rig::strict();
    rig.kernel.set_caller(THREAD, CORE, GuestTicks(1234));
    let numbers = payload(|w| {
        w.write_u32(1).align(8).write_u64(2);
    });
    let add = request(toy::ADD).data(&numbers);
    rig.send(&add);

    let recorded = rig.kernel.recorded();
    assert_eq!(recorded.len(), 1);
    let event = recorded[0];
    // Stamped by the kernel with the caller it was speaking for.
    assert_eq!(
        (event.thread, event.core, event.time),
        (THREAD, CORE, GuestTicks(1234))
    );
    let asked = argument_bytes(&add, SessionKind::Session);
    let replied = u32::try_from(size_of::<u64>()).unwrap();
    let success = ResultCode::SUCCESS;
    assert_eq!(
        event.kind,
        ipc(
            rig.session,
            toy::ADD,
            asked,
            replied,
            success,
            IpcOutcome::Done
        )
    );
    // The text is the trace crate's; this pins what a service request looks like in it.
    let session = session_id(rig.session).0;
    assert_eq!(
        rig.kernel.trace(),
        format!(
            "thread {}\n  ipc session={session} service={} command={} req={asked} rep={replied} \
             -> 0x0 Done\n",
            THREAD.0,
            SERVICE.0,
            toy::ADD
        )
    );
}

#[test]
fn a_stubbed_command_is_marked_stubbed() {
    let mut rig = Rig::new(UnknownPolicy::AutoStub);
    let missing = request(u32::from(toy::MISSING)).data(&[1, 2, 3, 4]);
    rig.send(&missing);
    rig.send(&RequestBuilder::tipc(toy::MISSING));
    let asked = argument_bytes(&missing, SessionKind::Session);
    let (command, success) = (u32::from(toy::MISSING), ResultCode::SUCCESS);
    assert_eq!(
        kinds(&rig),
        [
            ipc(rig.session, command, asked, 0, success, IpcOutcome::Stubbed),
            ipc(rig.session, command, 0, 0, success, IpcOutcome::Stubbed),
        ]
    );
    assert!(rig.kernel.trace().ends_with("-> 0x0 Stubbed\n"));
}

#[test]
fn a_command_that_fails_is_done_with_its_result_and_no_payload() {
    let mut rig = Rig::strict();
    // Unknown under the strict policy, and a command whose arguments are missing.
    rig.send(&request(u32::from(toy::MISSING)));
    rig.send(&request(toy::ADD));
    assert_eq!(
        kinds(&rig),
        [
            ipc(
                rig.session,
                u32::from(toy::MISSING),
                0,
                0,
                UNKNOWN_COMMAND_ID,
                IpcOutcome::Done
            ),
            ipc(
                rig.session,
                toy::ADD,
                0,
                0,
                INVALID_CMIF_REQUEST,
                IpcOutcome::Done
            ),
        ]
    );
}

#[test]
fn control_requests_and_domain_requests_are_traced_too() {
    let mut rig = Rig::strict();
    rig.send(&control_request(QUERY_POINTER_BUFFER_SIZE));
    rig.send(&control_request(CONVERT_CURRENT_OBJECT_TO_DOMAIN));
    let tag = domain_request(Domain::ROOT_OBJECT_ID, toy::OPEN_CHILD).data(&7_u32.to_le_bytes());
    rig.send(&tag);
    rig.send(&RequestBuilder::domain_close(Domain::ROOT_OBJECT_ID + 1));
    let (session, success) = (rig.session, ResultCode::SUCCESS);
    let done = IpcOutcome::Done;
    let asked = argument_bytes(&tag, SessionKind::Domain);
    let [u16_size, u32_size] =
        [size_of::<u16>(), size_of::<u32>()].map(|n| u32::try_from(n).unwrap());
    assert_eq!(
        kinds(&rig),
        [
            ipc(
                session,
                QUERY_POINTER_BUFFER_SIZE,
                0,
                u16_size,
                success,
                done
            ),
            ipc(
                session,
                CONVERT_CURRENT_OBJECT_TO_DOMAIN,
                0,
                u32_size,
                success,
                done
            ),
            // The new object's id is not payload.
            ipc(session, toy::OPEN_CHILD, asked, 0, success, done),
            // Closing an object has no command id.
            ipc(session, NO_COMMAND_ID, 0, 0, success, done),
        ]
    );
}

#[test]
fn a_deferred_request_is_one_event_when_it_is_answered_marked_deferred() {
    let mut rig = Rig::strict();
    let (first, second) = (rig.session, rig.second_session());
    rig.kernel.set_caller(ThreadId(1), 0, GuestTicks(10));
    let wait = request(toy::WAIT).data(&[9; 8]);
    rig.defer_on(first, &wait);
    // Nothing yet: the request has no result.
    assert_eq!(kinds(&rig), []);

    rig.kernel.set_caller(ThreadId(2), 1, GuestTicks(20));
    rig.send_on(second, &request(toy::INSPECT));

    let mut reply = Reply::default();
    reply.payload().write_u32(5);
    parked_one().complete(reply);
    // The kernel delivers on behalf of the thread that sent the request.
    rig.kernel.set_caller(ThreadId(1), 3, GuestTicks(30));
    rig.deliver();

    let recorded = rig.kernel.recorded();
    let asked = argument_bytes(&wait, SessionKind::Session);
    let replied = u32::try_from(size_of::<u32>()).unwrap();
    let success = ResultCode::SUCCESS;
    assert_eq!(recorded.len(), 2);
    assert_eq!(recorded[0].thread, ThreadId(2));
    assert_eq!(
        recorded[1].kind,
        ipc(
            first,
            toy::WAIT,
            asked,
            replied,
            success,
            IpcOutcome::Deferred
        )
    );
    assert_eq!(
        (recorded[1].thread, recorded[1].core, recorded[1].time),
        (ThreadId(1), 3, GuestTicks(30))
    );
    // The order of the events is the order the replies were written in.
    assert!(recorded[0].seq < recorded[1].seq);
}

#[test]
fn a_dropped_token_is_traced_with_its_error() {
    let mut rig = Rig::strict();
    rig.defer_on(rig.session, &request(toy::WAIT));
    drop(parked_one());
    rig.deliver();
    let dropped = super::super::DEFERRED_REPLY_DROPPED;
    assert_eq!(
        kinds(&rig),
        [ipc(
            rig.session,
            toy::WAIT,
            0,
            0,
            dropped,
            IpcOutcome::Deferred
        )]
    );
}

#[test]
fn a_message_that_gets_no_reply_gets_no_event() {
    let mut rig = Rig::strict();
    let (first, second) = (rig.session, rig.second_session());
    // A close message, and a message on a session the server does not have.
    let (closed, _) = rig.try_send_on(first, &RequestBuilder::close());
    assert_eq!(closed, Ok(super::super::Handled::Closed));
    let (handled, _) = rig.try_send_on(first, &request(toy::INSPECT));
    assert_eq!(
        handled,
        Err(super::super::ServerError::UnknownSession(first))
    );
    assert_eq!(kinds(&rig), []);
    // Every reply is exactly one event.
    for _ in 0..3 {
        rig.send_on(second, &request(toy::INSPECT));
    }
    assert_eq!(kinds(&rig).len(), 3);
}
