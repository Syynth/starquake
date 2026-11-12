//! The IPC table driven directly, without a dispatcher: sessions, the queue between their ends,
//! and what closing either end does.

use starquake_ipc::BUFFER_SIZE;
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
