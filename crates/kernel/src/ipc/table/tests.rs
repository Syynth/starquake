//! The IPC table driven directly, without a dispatcher: sessions, the queue between their ends,
//! and what closing either end does.

use starquake_ipc::{BUFFER_SIZE, CommandType};
use starquake_trace::{ServiceId, ThreadId};

use super::*;
use crate::handle::{Closed, HandleTable};
use crate::id::SlabId;
use crate::ipc::DEFAULT_WORKERS;

const SERVICE: ServiceId = ServiceId(7);

/// A request from thread `thread` whose message is all `fill`, holding nothing.
fn request(thread: u64, fill: u8) -> Request {
    Request {
        sender: Sender {
            thread: ThreadId(thread),
            waiter: None,
            core: 2,
            message_address: 0x1000 * thread,
            session_handle: Handle::from_raw(0x100 + u32::try_from(thread).unwrap()),
            process: crate::id::ProcessId::from_parts(0, 0),
        },
        message: Box::new([fill; BUFFER_SIZE]),
        translated: Translated::default(),
    }
}

/// An IPC table with one server, an object table, and a session on the server whose only
/// reference is a handle in `handles`.
struct World {
    ipc: Ipc,
    objects: Objects,
    handles: HandleTable,
    server: ServerId,
}

impl World {
    fn new() -> Self {
        let mut ipc = Ipc::new();
        let server = ipc.add_server("test", SERVICE).unwrap();
        Self {
            ipc,
            objects: Objects::new(),
            handles: HandleTable::new(8).unwrap(),
            server,
        }
    }

    /// A session as `ConnectToNamedPort` leaves it: one reference, a handle's.
    fn connect(&mut self) -> (ObjectId, Handle) {
        let session = self
            .ipc
            .open_session(&mut self.objects, self.server, true)
            .unwrap();
        let handle = self.handles.add(&mut self.objects, session).unwrap();
        self.ipc.release(&mut self.objects, session).unwrap();
        (session, handle)
    }

    fn send(&mut self, session: ObjectId, request: Request) -> Result<Admission, Refused> {
        self.ipc.send(&mut self.objects, session, request)
    }

    /// Ends `done` as the dispatcher does: releases what each held.
    fn release(&mut self, session: ObjectId, done: &[Done]) {
        for done in done {
            self.ipc
                .release_request(&mut self.objects, session, &done.request)
                .unwrap();
        }
    }

    /// Closes `handle` as `CloseHandle` does.
    fn close(&mut self, handle: Handle) {
        let removed = self.handles.remove(&mut self.objects, handle).unwrap();
        if removed.closed == Closed::Last {
            self.ipc.forget(removed.object).unwrap();
        }
    }
}

fn threads(done: &[Done]) -> Vec<(u64, Result<(), KernelError>)> {
    done.iter()
        .map(|done| (done.request.sender.thread.0, done.result))
        .collect()
}

#[test]
fn a_connection_is_a_session_object_and_its_server_is_told_once() {
    let mut world = World::new();
    let (session, handle) = world.connect();
    assert_eq!(world.objects.get(session), Ok(Object::Session));
    assert_eq!(world.objects.refs(session), 1);
    assert_eq!(world.handles.get(handle), Ok(session));
    assert_eq!(world.ipc.server_of(session), Ok(world.server));
    assert!(world.ipc.has_any_notices());
    assert_eq!(
        world.ipc.take_notices(world.server),
        Ok(vec![Notice::Accept(session)])
    );
    assert_eq!(world.ipc.take_notices(world.server), Ok(vec![]));
    assert!(!world.ipc.has_any_notices());
    // A session the server makes on itself needs no notice.
    world
        .ipc
        .open_session(&mut world.objects, world.server, false)
        .unwrap();
    assert!(!world.ipc.has_any_notices());
}

#[test]
fn a_request_starts_when_nothing_is_in_flight_and_holds_its_session() {
    let mut world = World::new();
    let (session, _) = world.connect();
    assert_eq!(world.send(session, request(1, 0xA1)), Ok(Admission::Start));
    assert_eq!(world.objects.refs(session), 2);
    assert_eq!(
        world
            .ipc
            .in_flight(session)
            .unwrap()
            .map(|r| r.sender.thread),
        Some(ThreadId(1))
    );
    let reply = Box::new([0xEE; BUFFER_SIZE]);
    let done = world
        .ipc
        .finish(session, Finish::Replied(reply.clone()))
        .unwrap();
    assert_eq!(threads(&done), [(1, Ok(()))]);
    assert_eq!(done[0].request.message, reply);
    world.release(session, &done);
    assert_eq!(world.objects.refs(session), 1);
    assert_eq!(world.ipc.in_flight(session), Ok(None));
    assert_eq!(world.ipc.queue(world.server).unwrap().running(), 0);
}

#[test]
fn a_second_sender_waits_behind_a_deferred_request_and_starts_when_it_completes() {
    let mut world = World::new();
    let (session, _) = world.connect();
    assert_eq!(world.send(session, request(1, 1)), Ok(Admission::Start));
    // Deferred: the worker is free, but the session's request is still in flight.
    assert_eq!(world.ipc.finish(session, Finish::Deferred), Ok(vec![]));
    let queue = world.ipc.queue(world.server).unwrap();
    assert_eq!((queue.running(), queue.is_deferred(session)), (0, true));
    assert_eq!(world.send(session, request(2, 2)), Ok(Admission::Queued));
    assert_eq!(world.ipc.waiting(session), Ok(1));
    assert_eq!(world.objects.refs(session), 3);
    // Nothing may start while the first is in flight.
    assert_eq!(world.ipc.start_next(world.server), Ok(None));

    let done = world
        .ipc
        .complete(session, Box::new([9; BUFFER_SIZE]))
        .unwrap();
    assert_eq!(done.request.sender.thread, ThreadId(1));
    world.release(session, &[done]);
    assert_eq!(world.ipc.start_next(world.server), Ok(Some(session)));
    assert_eq!(world.ipc.waiting(session), Ok(0));
    assert_eq!(
        world
            .ipc
            .in_flight(session)
            .unwrap()
            .map(|r| r.sender.thread),
        Some(ThreadId(2))
    );
    assert_eq!(world.ipc.start_next(world.server), Ok(None));
}

#[test]
fn two_sessions_of_one_server_do_not_run_at_once() {
    let mut world = World::new();
    let (a, _) = world.connect();
    let (b, _) = world.connect();
    assert_eq!(DEFAULT_WORKERS, 1);
    assert_eq!(world.send(a, request(1, 1)), Ok(Admission::Start));
    assert_eq!(world.send(b, request(2, 2)), Ok(Admission::Queued));
    assert_eq!(world.ipc.start_next(world.server), Ok(None));
    let done = world
        .ipc
        .finish(a, Finish::Replied(Box::new([0; BUFFER_SIZE])))
        .unwrap();
    world.release(a, &done);
    assert_eq!(world.ipc.start_next(world.server), Ok(Some(b)));
}

#[test]
fn closing_the_server_end_ends_the_request_in_flight_and_every_waiting_one() {
    let mut world = World::new();
    let (session, _) = world.connect();
    world.send(session, request(1, 1)).unwrap();
    world.ipc.finish(session, Finish::Deferred).unwrap();
    world.send(session, request(2, 2)).unwrap();
    world.send(session, request(3, 3)).unwrap();
    let done = world.ipc.finish(session, Finish::ServerClosed).unwrap();
    let closed = Err(KernelError::SessionClosed);
    assert_eq!(threads(&done), [(1, closed), (2, closed), (3, closed)]);
    world.release(session, &done);
    assert_eq!(world.objects.refs(session), 1);
    assert_eq!(world.ipc.is_server_open(session), Ok(false));
    assert!(
        world
            .ipc
            .queue(world.server)
            .unwrap()
            .sessions()
            .next()
            .is_none()
    );
    // A new request is refused at once and handed back.
    let refused = world.send(session, request(4, 4)).unwrap_err();
    assert_eq!(refused.error, IpcError::Kernel(KernelError::SessionClosed));
    assert_eq!(refused.request.sender.thread, ThreadId(4));
    assert_eq!(world.objects.refs(session), 1);
}

#[test]
fn a_running_request_ended_by_the_server_closing_frees_its_worker() {
    let mut world = World::new();
    let (a, _) = world.connect();
    let (b, _) = world.connect();
    world.send(a, request(1, 1)).unwrap();
    world.send(b, request(2, 2)).unwrap();
    let done = world.ipc.finish(a, Finish::ServerClosed).unwrap();
    assert_eq!(threads(&done), [(1, Err(KernelError::SessionClosed))]);
    world.release(a, &done);
    assert_eq!(world.ipc.start_next(world.server), Ok(Some(b)));
}

#[test]
fn closing_the_last_handle_destroys_the_session_and_tells_the_server() {
    let mut world = World::new();
    let (session, handle) = world.connect();
    world.ipc.take_notices(world.server).unwrap();
    world.close(handle);
    assert!(!world.ipc.is_session(session));
    assert!(world.objects.is_empty());
    assert_eq!(
        world.ipc.take_notices(world.server),
        Ok(vec![Notice::Closed(session)])
    );
}

#[test]
fn a_session_whose_server_end_closed_is_destroyed_without_telling_the_server() {
    let mut world = World::new();
    let (session, handle) = world.connect();
    world.ipc.take_notices(world.server).unwrap();
    world.send(session, request(1, 1)).unwrap();
    let done = world.ipc.finish(session, Finish::ServerClosed).unwrap();
    world.release(session, &done);
    world.close(handle);
    assert!(!world.ipc.is_session(session));
    // The server closed it itself, so it is not told again.
    assert_eq!(world.ipc.take_notices(world.server), Ok(vec![]));
}

#[test]
fn closing_the_client_end_with_a_request_in_flight_waits_for_the_request() {
    let mut world = World::new();
    let (session, handle) = world.connect();
    world.ipc.take_notices(world.server).unwrap();
    world.send(session, request(1, 1)).unwrap();
    world.ipc.finish(session, Finish::Deferred).unwrap();
    // The handle goes; the request still holds the session.
    world.close(handle);
    assert!(world.ipc.is_session(session));
    assert_eq!(world.ipc.take_notices(world.server), Ok(vec![]));
    let done = world
        .ipc
        .complete(session, Box::new([0; BUFFER_SIZE]))
        .unwrap();
    world.release(session, &[done]);
    assert!(!world.ipc.is_session(session));
    assert_eq!(
        world.ipc.take_notices(world.server),
        Ok(vec![Notice::Closed(session)])
    );
}

#[test]
fn a_finished_request_releases_the_objects_it_held() {
    let mut world = World::new();
    let (session, _) = world.connect();
    let (held, held_handle) = world.connect();
    world.ipc.take_notices(world.server).unwrap();
    // As translating a moved handle leaves it: the reference is the request's.
    world.objects.open(held).unwrap();
    world.close(held_handle);
    let mut sent = request(1, 1);
    sent.translated.held.push(held);
    sent.translated.handles.push((held_handle, held));
    world.send(session, sent).unwrap();
    assert!(world.ipc.is_session(held));
    let done = world
        .ipc
        .finish(session, Finish::Failed(KernelError::InvalidHandle))
        .unwrap();
    assert_eq!(threads(&done), [(1, Err(KernelError::InvalidHandle))]);
    world.release(session, &done);
    // That was the moved session's last reference.
    assert!(!world.ipc.is_session(held));
    assert_eq!(
        world.ipc.take_notices(world.server),
        Ok(vec![Notice::Closed(held)])
    );
}

#[test]
fn a_discarded_session_leaves_nothing_behind() {
    let mut world = World::new();
    let session = world
        .ipc
        .open_session(&mut world.objects, world.server, true)
        .unwrap();
    world
        .ipc
        .discard_session(&mut world.objects, session)
        .unwrap();
    assert!(!world.ipc.is_session(session));
    assert!(world.objects.is_empty());
    assert!(!world.ipc.has_any_notices());
}

#[test]
fn ports_and_services_name_one_server_each() {
    let mut ipc = Ipc::new();
    let first = ipc.add_server("first", SERVICE).unwrap();
    assert_eq!(
        ipc.add_server("again", SERVICE),
        Err(IpcError::ServiceTaken(SERVICE))
    );
    let second = ipc.add_server("second", ServiceId(8)).unwrap();
    let name = PortName::new("first:").unwrap();
    ipc.name_port(name, first).unwrap();
    assert_eq!(ipc.name_port(name, second), Err(IpcError::PortTaken(name)));
    assert_eq!(ipc.port(&name), Some(first));
    assert_eq!(ipc.port(&PortName::new("second:").unwrap()), None);
    assert_eq!(ipc.service(ServiceId(8)), Some(second));
    assert_eq!(ipc.service(ServiceId(9)), None);
    assert_eq!(ipc.server_name(second), Ok("second"));
    assert_eq!(ipc.server_service(first), Ok(SERVICE));
    assert_eq!(ipc.servers().collect::<Vec<_>>(), [first, second]);
}

#[test]
fn ids_that_name_nothing_are_errors_not_panics() {
    let mut world = World::new();
    let stale = ObjectId::from_parts(40, 1);
    let unknown = IpcError::UnknownSession(stale);
    assert_eq!(world.ipc.server_of(stale), Err(unknown.clone()));
    assert_eq!(
        world.ipc.finish(stale, Finish::Deferred),
        Err(unknown.clone())
    );
    assert_eq!(
        world.ipc.complete(stale, Box::new([0; BUFFER_SIZE])),
        Err(unknown.clone())
    );
    assert_eq!(world.send(stale, request(1, 1)).unwrap_err().error, unknown);
    let nowhere = ServerId::from_parts(9, 9);
    assert_eq!(
        world.ipc.open_session(&mut world.objects, nowhere, true),
        Err(IpcError::UnknownServer(nowhere))
    );
    assert_eq!(
        world.ipc.take_notices(nowhere),
        Err(IpcError::UnknownServer(nowhere))
    );
    // Finishing a session with nothing in flight is the caller's bookkeeping gone wrong.
    let (session, _) = world.connect();
    assert!(world.ipc.finish(session, Finish::Deferred).is_err());
    assert!(
        world
            .ipc
            .complete(session, Box::new([0; BUFFER_SIZE]))
            .is_err()
    );
}

#[test]
fn an_ipc_failure_is_a_kernel_result_only_when_it_is_one() {
    assert_eq!(
        SvcError::from(IpcError::Kernel(KernelError::SessionClosed)),
        SvcError::Kernel(KernelError::SessionClosed)
    );
    let stale = IpcError::UnknownSession(ObjectId::from_parts(1, 2));
    assert_eq!(
        SvcError::from(stale.clone()),
        SvcError::Internal(stale.to_string())
    );
}

#[test]
fn a_server_event_is_one_its_own_server_noted() {
    let mut world = World::new();
    let other = world.ipc.add_server("other", ServiceId(8)).unwrap();
    let event = ObjectId::from_parts(3, 1);
    // The same slot in another generation: another object.
    let stale = ObjectId::from_parts(3, 2);
    assert!(!world.ipc.is_server_event(world.server, event));
    world.ipc.add_server_event(world.server, event).unwrap();
    assert!(world.ipc.is_server_event(world.server, event));
    assert!(!world.ipc.is_server_event(world.server, stale));
    assert!(!world.ipc.is_server_event(other, event));

    let nowhere = ServerId::from_parts(9, 9);
    assert_eq!(
        world.ipc.add_server_event(nowhere, event),
        Err(IpcError::UnknownServer(nowhere))
    );
    assert!(!world.ipc.is_server_event(nowhere, event));
}

// ----- Kept references and open sessions ------------------------------------------------------

#[test]
fn open_sessions_counts_sessions_whose_server_end_is_open_on_that_server_only() {
    let mut world = World::new();
    let other = world.ipc.add_server("other", ServiceId(8)).unwrap();
    assert_eq!(world.ipc.open_sessions(world.server), 0);
    let (first, _) = world.connect();
    let (_, second) = world.connect();
    world
        .ipc
        .open_session(&mut world.objects, other, true)
        .unwrap();
    assert_eq!(world.ipc.open_sessions(world.server), 2);
    assert_eq!(world.ipc.open_sessions(other), 1);
    // The server closes its end of one: it is no longer counted, though its client end is
    // still there.
    world.send(first, request(1, 1)).unwrap();
    let done = world.ipc.finish(first, Finish::ServerClosed).unwrap();
    world.release(first, &done);
    assert_eq!(world.ipc.is_server_open(first), Ok(false));
    assert_eq!(world.ipc.open_sessions(world.server), 1);
    // The client of the other goes.
    world.close(second);
    assert_eq!(world.ipc.open_sessions(world.server), 0);
    assert_eq!(world.ipc.open_sessions(other), 1);
    // A server the table does not know has none.
    assert_eq!(world.ipc.open_sessions(SlabId::from_parts(0x7F, 3)), 0);
}

#[test]
fn kept_notes_count_up_and_down_per_server_and_object() {
    let mut world = World::new();
    let other = world.ipc.add_server("other", ServiceId(8)).unwrap();
    let object = world.objects.create(Object::Session).unwrap();
    let neighbour = world.objects.create(Object::Session).unwrap();
    assert_eq!(world.ipc.kept(world.server, object), 0);
    assert!(!world.ipc.take_kept(world.server, object));
    world.ipc.add_kept(world.server, object).unwrap();
    world.ipc.add_kept(world.server, object).unwrap();
    world.ipc.add_kept(other, neighbour).unwrap();
    assert_eq!(world.ipc.kept(world.server, object), 2);
    // Neither the other server nor the other object shares the count.
    assert_eq!(world.ipc.kept(other, object), 0);
    assert_eq!(world.ipc.kept(world.server, neighbour), 0);
    assert!(!world.ipc.take_kept(other, object));
    assert!(world.ipc.take_kept(world.server, object));
    assert_eq!(world.ipc.kept(world.server, object), 1);
    assert!(world.ipc.take_kept(world.server, object));
    assert_eq!(world.ipc.kept(world.server, object), 0);
    assert!(!world.ipc.take_kept(world.server, object));
    assert_eq!(world.ipc.kept(other, neighbour), 1);
    // A server the table does not know keeps nothing and can note nothing.
    let unknown = SlabId::from_parts(0x7F, 3);
    assert_eq!(
        world.ipc.add_kept(unknown, object),
        Err(IpcError::UnknownServer(unknown))
    );
    assert!(!world.ipc.take_kept(unknown, object));
    assert_eq!(world.ipc.kept(unknown, object), 0);
}

/// A request whose message is all `fill` but for its first header word, `word0`: the command
/// type in bits 0..16 (`starquake_ipc::CommandHeader`).
fn request_typed(thread: u64, fill: u8, word0: u32) -> Request {
    let mut request = request(thread, fill);
    request.message[..size_of::<u32>()].copy_from_slice(&word0.to_le_bytes());
    request
}

#[test]
fn a_session_whose_request_in_flight_is_a_close_message_is_not_counted_open() {
    // The server's handler closes such a session before the kernel ends the request, so a
    // question asked in between (gsv's drain) must not count it: the services server has
    // already forgotten it. Each type, raw as the header carries it, with the descriptor counts
    // above bit 16 set, which the close test ignores (`CommandHeader::decode`).
    let closes = [
        (CommandType::Close.to_raw().unwrap(), true),
        (CommandType::TipcClose.to_raw().unwrap(), true),
        (CommandType::Request.to_raw().unwrap(), false),
        (CommandType::Control.to_raw().unwrap(), false),
        (CommandType::LegacyRequest.to_raw().unwrap(), false),
        (CommandType::Invalid.to_raw().unwrap(), false),
        (CommandType::Tipc(1).to_raw().unwrap(), false),
        // An undefined type, which the server's parse refuses: not a close.
        (8, false),
        (14, false),
    ];
    for (raw, close) in closes {
        let mut world = World::new();
        let (session, _) = world.connect();
        let (_, _neighbour) = world.connect();
        let word0 = 0x00F0_0000 | u32::from(raw);
        assert_eq!(
            world.send(session, request_typed(1, 1, word0)),
            Ok(Admission::Start)
        );
        let open = if close { 1 } else { 2 };
        assert_eq!(world.ipc.open_sessions(world.server), open, "type {raw}");
        // Once the request ends with the server's end closed, it is not counted either way.
        let done = world.ipc.finish(session, Finish::ServerClosed).unwrap();
        world.release(session, &done);
        assert_eq!(world.ipc.open_sessions(world.server), 1, "type {raw}");
    }
}

#[test]
fn a_close_message_waiting_behind_another_request_is_not_left_out() {
    // Only the request in flight is the one the handler is running: a close waiting behind it
    // has not closed anything yet.
    let mut world = World::new();
    let (session, _) = world.connect();
    let close = u32::from(CommandType::Close.to_raw().unwrap());
    let other = u32::from(CommandType::Request.to_raw().unwrap());
    assert_eq!(
        world.send(session, request_typed(1, 1, other)),
        Ok(Admission::Start)
    );
    assert_eq!(
        world.send(session, request_typed(2, 2, close)),
        Ok(Admission::Queued)
    );
    assert_eq!(world.ipc.open_sessions(world.server), 1);
    // The close starts once the first request is over: now it is left out.
    let done = world
        .ipc
        .finish(session, Finish::Replied(Box::new([0; BUFFER_SIZE])))
        .unwrap();
    world.release(session, &done);
    assert_eq!(world.ipc.start_next(world.server), Ok(Some(session)));
    assert_eq!(world.ipc.open_sessions(world.server), 0);
}
