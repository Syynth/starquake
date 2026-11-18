//! Taking back the handles a handler placed in its sender's table that the sender does not
//! receive (`dispatch/ipc/placed.rs`, issue #209): a failure at each position, a full table, a
//! write that faults, a handler that fails or closes the session, a deferred reply, a sender
//! that ended, and a reply with no failure, which is unchanged. Each asserts the sender's table
//! and every object's reference count afterwards.

use starquake_memory::{Fault, GuestMemory};

use super::*;
use crate::handle::MAX_LINEAR_ID;

/// A server's events, made by the server so that it keeps the creator's reference to each
/// (`IpcContext::create_event`): what an HLE server copies or moves to a client.
fn server_events(world: &World, count: usize) -> Vec<ObjectId> {
    let handle = world.session();
    let made = Arc::new(Mutex::new(Vec::new()));
    let keep = Arc::clone(&made);
    world.second.script(Box::new(move |cx, _, message| {
        for _ in 0..count {
            lock(&keep).push(cx.create_event().unwrap());
        }
        name_in_reply(message, &[]);
        Ok(Handled::Replied)
    }));
    // The events are the second server's; a session on it is the way to its handler.
    let second = connect_second(world, handle);
    let (_, regs) = world.send(A, second, &request());
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    let made = lock(&made).clone();
    assert_eq!(made.len(), count);
    made
}

/// Has the first server give thread A a session on the second, through `handle`.
fn connect_second(world: &World, handle: Handle) -> Handle {
    let connected = Arc::new(Mutex::new(None));
    let keep = Arc::clone(&connected);
    world.first.script(Box::new(move |cx, _, message| {
        let (_, handle) = cx.connect(SECOND).unwrap();
        *lock(&keep) = Some(handle);
        name_in_reply(message, &[handle]);
        Ok(Handled::Replied)
    }));
    let (_, regs) = world.send(A, handle, &request());
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    lock(&connected).unwrap()
}

/// An object id that names nothing any more: an event the guest made and closed both ends of.
fn stale_object(world: &World) -> ObjectId {
    let (_, regs) = world.call(A, SvcId::CreateEvent, 0, 0);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    let ends = [regs.x[1], regs.x[2]].map(Handle::from_register);
    let object = world.object(ends[0]);
    for end in ends {
        assert_eq!(result(&world.close(end)), ResultCode::SUCCESS);
    }
    world.with_tables(|_, scheduler| assert!(scheduler.objects().get(object).is_err()));
    object
}

/// What the sender's table holds and what each object's count is: compared before and after.
fn snapshot(world: &World, objects: &[ObjectId]) -> (usize, Vec<u32>, usize) {
    world.with_tables(|handles, scheduler| {
        let refs = objects
            .iter()
            .map(|object| scheduler.objects().refs(*object))
            .collect();
        (handles.len(), refs, scheduler.objects_besides_processes())
    })
}

/// The result word of the reply in `thread`'s message buffer.
fn reply_result(world: &World, thread: ThreadId) -> ResultCode {
    let tls = world.tls(thread);
    let reply =
        starquake_ipc::Message::parse_reply(&tls, starquake_ipc::SessionKind::Session).unwrap();
    match reply.body {
        starquake_ipc::Body::CmifReply(reply) => ResultCode::from_raw(reply.header.result),
        other => panic!("not a CMIF reply: {other:?}"),
    }
}

/// Places a handle to each of `objects` in turn, as the service framework's reply builder
/// does; at the first that cannot be placed, writes an error reply naming none and stops, as the
/// builder's `?` does. With none failing, the reply names every handle. The placed handles and
/// the failure go to `seen`.
fn place_each(objects: Vec<ObjectId>, seen: Arc<Mutex<Placing>>) -> Script {
    Box::new(move |cx, _, message| {
        let mut placed = Vec::new();
        for object in &objects {
            match cx.insert_handle(*object) {
                Ok(handle) => placed.push(handle),
                Err(error) => {
                    ReplyBuilder::cmif(error.code().raw())
                        .build(message)
                        .unwrap();
                    *lock(&seen) = Placing {
                        placed,
                        failed: Some(error),
                    };
                    return Ok(Handled::Replied);
                }
            }
        }
        name_in_reply(message, &placed);
        *lock(&seen) = Placing {
            placed,
            failed: None,
        };
        Ok(Handled::Replied)
    })
}

/// What [`place_each`] saw.
#[derive(Debug, Default)]
struct Placing {
    placed: Vec<Handle>,
    failed: Option<KernelError>,
}

/// Sends a request whose handler places a handle to each of `objects`, one of which (at `bad`)
/// names nothing, and checks that the sender is left as it was: the handles placed before the
/// bad one are taken back, those after it never placed, and the reply carries the failure.
fn fail_at(bad: usize) {
    let world = World::new();
    let mut objects = server_events(&world, 3);
    objects[bad] = stale_object(&world);
    let handle = world.session();
    let good: Vec<ObjectId> = (0..3).filter(|n| *n != bad).map(|n| objects[n]).collect();
    let before = snapshot(&world, &good);
    let seen = Arc::new(Mutex::new(Placing::default()));
    world
        .first
        .script(place_each(objects.clone(), Arc::clone(&seen)));
    let (_, regs) = world.send(A, handle, &request());
    // The reply is delivered; it carries the failure, and names no handle.
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    let seen = std::mem::take(&mut *lock(&seen));
    assert_eq!(seen.placed.len(), bad, "placed before the bad one");
    let failure = seen.failed.expect("the bad object is refused");
    assert_eq!(reply_result(&world, A), failure.code());
    assert_eq!(snapshot(&world, &good), before);
    world.with_tables(|handles, scheduler| {
        for placed in &seen.placed {
            assert!(handles.get(*placed).is_err(), "{placed:?} was taken back");
        }
        // Every good object is still the server's, alive with its creator's reference.
        for object in &good {
            assert!(matches!(
                scheduler.objects().get(*object),
                Ok(Object::ReadableEvent(_))
            ));
            assert_eq!(scheduler.objects().refs(*object), 1);
        }
    });
}

#[test]
fn a_failure_at_the_first_handle_places_none_and_changes_nothing() {
    fail_at(0);
}

#[test]
fn a_failure_at_the_middle_handle_takes_back_the_first() {
    fail_at(1);
}

#[test]
fn a_failure_at_the_last_handle_takes_back_the_two_before_it() {
    fail_at(2);
}

#[test]
fn a_reply_with_no_failure_keeps_every_handle_it_names() {
    let world = World::new();
    let objects = server_events(&world, 3);
    let handle = world.session();
    let (table, _, live) = snapshot(&world, &objects);
    let seen = Arc::new(Mutex::new(Placing::default()));
    world
        .first
        .script(place_each(objects.clone(), Arc::clone(&seen)));
    let (_, regs) = world.send(A, handle, &request());
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    let seen = std::mem::take(&mut *lock(&seen));
    assert_eq!(seen.failed, None);
    assert_eq!(reply_result(&world, A), ResultCode::SUCCESS);
    assert_eq!(
        crate::ipc::message::reply_handles(&world.tls(A)),
        seen.placed
    );
    // Each handle is the sender's: one more entry, and one more reference, per object.
    assert_eq!(snapshot(&world, &objects), (table + 3, vec![2; 3], live));
    world.with_tables(|handles, _| {
        for (placed, object) in seen.placed.iter().zip(&objects) {
            assert_eq!(handles.get(*placed), Ok(*object));
        }
    });
}

#[test]
fn a_reply_naming_some_of_the_placed_handles_keeps_exactly_those() {
    let world = World::new();
    let objects = server_events(&world, 3);
    let handle = world.session();
    let (table, _, live) = snapshot(&world, &objects);
    let placed = Arc::new(Mutex::new(Vec::new()));
    let keep = Arc::clone(&placed);
    let to_place = objects.clone();
    world.first.script(Box::new(move |cx, _, message| {
        let handles: Vec<Handle> = to_place
            .iter()
            .map(|object| cx.insert_handle(*object).unwrap())
            .collect();
        // The middle one as a move, the last not named at all.
        ReplyBuilder::cmif(0)
            .copy_handle(handles[0].raw())
            .move_handle(handles[1].raw())
            .build(message)
            .unwrap();
        lock(&keep).extend(handles);
        Ok(Handled::Replied)
    }));
    world.send(A, handle, &request());
    let placed = lock(&placed).clone();
    assert_eq!(snapshot(&world, &objects), (table + 2, vec![2, 2, 1], live));
    world.with_tables(|handles, _| {
        assert_eq!(handles.get(placed[0]), Ok(objects[0]));
        assert_eq!(handles.get(placed[1]), Ok(objects[1]));
        assert!(handles.get(placed[2]).is_err());
    });
}

#[test]
fn a_full_sender_table_takes_back_what_was_placed() {
    // Room for the three sessions the set-up leaves and two more handles: the third object
    // does not fit.
    let sessions = 3;
    let world = World::with_handle_table(sessions + 2);
    let objects = server_events(&world, 3);
    let handle = world.session();
    let before = snapshot(&world, &objects);
    assert_eq!(
        before.0,
        usize::try_from(sessions).unwrap(),
        "the two sessions `server_events` made, and this one"
    );
    let watch = Arc::new(Mutex::new(Placing::default()));
    world
        .first
        .script(place_each(objects.clone(), Arc::clone(&watch)));
    let (_, regs) = world.send(A, handle, &request());
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    let seen = std::mem::take(&mut *lock(&watch));
    assert_eq!(seen.placed.len(), 2);
    assert_eq!(seen.failed, Some(KernelError::OutOfHandles));
    assert_eq!(reply_result(&world, A), KernelError::OutOfHandles.code());
    assert_eq!(snapshot(&world, &objects), before);
    // The table has its room back: a reply that names its handles places two again.
    world
        .first
        .script(place_each(objects[..2].to_vec(), Arc::clone(&watch)));
    world.send(A, handle, &request());
    assert_eq!(lock(&watch).failed, None);
    assert_eq!(snapshot(&world, &objects).0, before.0 + 2);
}

#[test]
fn taken_back_sessions_are_torn_down_and_their_servers_told_in_the_order_placed() {
    let world = World::new();
    let handle = world.session();
    let own_session = world.object(handle);
    let made = Arc::new(Mutex::new(Vec::new()));
    let keep = Arc::clone(&made);
    world.first.script(Box::new(move |cx, _, message| {
        // Two sessions on this server, one on the other between them, then a failure.
        let first = cx.create_session().unwrap();
        let other = cx.connect(SECOND).unwrap();
        let second = cx.create_session().unwrap();
        lock(&keep).extend([first, other, second]);
        ReplyBuilder::cmif(KernelError::OutOfResource.code().raw())
            .build(message)
            .unwrap();
        Ok(Handled::Replied)
    }));
    world.send(A, handle, &request());
    let made = lock(&made).clone();
    let [(first, _), (other, _), (second, _)] = [made[0], made[1], made[2]];
    // Each session went with its only handle, and its server was told, in the order placed.
    assert_eq!(world.first.log().closed, [first, second]);
    assert_eq!(world.second.log().accepted, [other]);
    assert_eq!(world.second.log().closed, [other]);
    world.with_tables(|handles, scheduler| {
        assert_eq!(handles.len(), 1);
        for session in [first, other, second] {
            assert!(!scheduler.ipc().is_session(session));
        }
        assert_eq!(scheduler.objects_besides_processes(), 1);
        assert_eq!(scheduler.objects().refs(own_session), 1);
    });
}

#[test]
fn a_write_to_the_senders_memory_that_faults_takes_back_what_was_placed() {
    let world = World::new();
    let objects = server_events(&world, 2);
    let handle = world.session();
    let before = snapshot(&world, &objects);
    let to_place = objects.clone();
    world.first.script(Box::new(move |cx, _, message| {
        for object in &to_place {
            cx.insert_handle(*object).unwrap();
        }
        // Text is not writable: the write an out buffer needs faults, and the reply is its
        // error, naming no handle.
        let error = cx.write_memory(CODE, &[0; 4]).unwrap_err();
        assert_eq!(error, KernelError::InvalidCurrentMemory);
        ReplyBuilder::cmif(error.code().raw())
            .build(message)
            .unwrap();
        Ok(Handled::Replied)
    }));
    let (_, regs) = world.send(A, handle, &request());
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    assert_eq!(
        reply_result(&world, A),
        KernelError::InvalidCurrentMemory.code()
    );
    assert_eq!(snapshot(&world, &objects), before);
}

/// Guest memory that refuses every write at one address: a sender's message buffer the reply
/// cannot be written to, though its request could be read.
struct NoWriteAt<'a> {
    inner: &'a FlatMemory,
    at: u64,
}

impl GuestMemory for NoWriteAt<'_> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.inner.read(va, buf)
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        if va == self.at {
            return Err(Fault { address: va });
        }
        self.inner.write(va, data)
    }
}

#[test]
fn a_reply_that_cannot_be_written_takes_back_every_handle_and_fails() {
    let world = World::new();
    let objects = server_events(&world, 2);
    let handle = world.session();
    let before = snapshot(&world, &objects);
    let to_place = objects.clone();
    world.first.script(Box::new(move |cx, _, message| {
        let placed: Vec<Handle> = to_place
            .iter()
            .map(|object| cx.insert_handle(*object).unwrap())
            .collect();
        // A good reply, naming both: only its write fails.
        name_in_reply(message, &placed);
        Ok(Handled::Replied)
    }));
    let mut buffer = [0; BUFFER_SIZE];
    request().build(&mut buffer).unwrap();
    world.memory.write(TLS_A, &buffer).unwrap();
    let memory = NoWriteAt {
        inner: &world.memory,
        at: TLS_A,
    };
    let mut debug = NoDebug;
    let mut caller = Caller {
        core: core_of(A),
        thread: A,
        env: &world.env,
        memory: &memory,
        debug: &mut debug,
        recorder: &world.recorder,
    };
    let mut regs = GuestRegs::default();
    regs.x[0] = u64::from(handle.raw());
    regs.tpidrro = TLS_A;
    let record = world
        .dispatcher
        .dispatch(SvcId::SendSyncRequest.number(), &mut regs, &mut caller);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(result(&regs), KernelError::InvalidCurrentMemory.code());
    // Nothing was written, and the sender is as it was.
    assert_eq!(world.tls(A), buffer);
    assert_eq!(snapshot(&world, &objects), before);
    // The system call's event carries the failure.
    let events = world.events();
    assert_eq!(
        threads_and_kinds(&events[events.len() - 1..]),
        [send_event(
            A,
            handle,
            KernelError::InvalidCurrentMemory.code()
        )]
    );
}

#[test]
fn a_handler_that_fails_or_closes_the_session_leaves_none_of_its_handles() {
    // How each ends, and whether the kernel tells the server it lost the session: a handler
    // that closes the session has forgotten it already (`Handled::Closed`).
    let scripts: [(fn() -> Result<Handled, HandlerError>, bool); 2] = [
        (|| Err(HandlerError("gave up".to_owned())), true),
        (|| Ok(Handled::Closed), false),
    ];
    for (finish, told) in scripts {
        let world = World::new();
        let objects = server_events(&world, 2);
        let handle = world.session();
        let session = world.object(handle);
        let before = snapshot(&world, &objects);
        let to_place = objects.clone();
        world.first.script(Box::new(move |cx, _, message| {
            let placed: Vec<Handle> = to_place
                .iter()
                .map(|object| cx.insert_handle(*object).unwrap())
                .collect();
            // Even a reply that names them is no reply when the request fails.
            name_in_reply(message, &placed);
            finish()
        }));
        let (_, regs) = world.send(A, handle, &request());
        assert_eq!(result(&regs), KernelError::SessionClosed.code());
        assert_eq!(snapshot(&world, &objects), before);
        let closed = if told { vec![session] } else { Vec::new() };
        assert_eq!(world.first.log().closed, closed);
    }
}

/// A third server's handler, which defers every request, placing a handle to `early` first
/// when there is one, and answers its deferred request by placing a handle to `object` and
/// naming it in the reply. ([`Scripted`]'s `take_reply` fills its reply with one byte, which
/// names no handle.)
struct NamingReply {
    early: Option<ObjectId>,
    object: ObjectId,
    deferred: Option<ObjectId>,
    placed: Arc<Mutex<Vec<Handle>>>,
}

impl SessionHandler for NamingReply {
    fn accept(&mut self, _: ObjectId) -> Result<(), HandlerError> {
        Ok(())
    }

    fn closed(&mut self, _: ObjectId) {}

    fn handle(
        &mut self,
        cx: &mut IpcContext<'_>,
        session: ObjectId,
        _: &mut [u8],
    ) -> Result<Handled, HandlerError> {
        if let Some(early) = self.early {
            let handle = cx.insert_handle(early).unwrap();
            lock(&self.placed).push(handle);
        }
        self.deferred = Some(session);
        Ok(Handled::Deferred)
    }

    fn take_reply(
        &mut self,
        cx: &mut IpcContext<'_>,
        message: &mut [u8],
    ) -> Option<(ObjectId, Result<(), HandlerError>)> {
        let session = self.deferred.take()?;
        let handle = cx.insert_handle(self.object).unwrap();
        lock(&self.placed).push(handle);
        name_in_reply(message, &[handle]);
        Some((session, Ok(())))
    }
}

/// The service number and port of the third server.
const THIRD: ServiceId = ServiceId(0x53);
const THIRD_PORT: &[u8] = b"third:\0";

/// Adds a third server behind a [`NamingReply`], connects thread A to it, and returns the
/// server, the session handle and the handles the handler places, in order.
fn naming_server(
    world: &World,
    early: Option<ObjectId>,
    object: ObjectId,
) -> (ServerId, Handle, Arc<Mutex<Vec<Handle>>>) {
    let placed = Arc::new(Mutex::new(Vec::new()));
    let handler = NamingReply {
        early,
        object,
        deferred: None,
        placed: Arc::clone(&placed),
    };
    let server = world
        .dispatcher
        .add_server(
            "third",
            THIRD,
            Some(PortName::new("third:").unwrap()),
            Box::new(handler),
        )
        .unwrap();
    let (record, regs) = world.connect(THIRD_PORT);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(result(&regs), ResultCode::SUCCESS);
    (server, Handle::from_register(regs.x[1]), placed)
}

/// Delivers the third server's ready replies, from thread A's core.
fn deliver_third(world: &World, server: ServerId) {
    world
        .dispatcher
        .deliver_replies(server, &mut world.caller(A))
        .unwrap();
}

#[test]
fn handles_placed_by_a_handler_that_defers_are_taken_back_when_it_returns() {
    let world = World::new();
    let objects = server_events(&world, 2);
    let (early, late) = (objects[0], objects[1]);
    let (server, handle, placed) = naming_server(&world, Some(early), late);
    let before = snapshot(&world, &objects);
    let (record, _) = world.send(A, handle, &request());
    assert_eq!(record.outcome, Outcome::Waiting);
    // Taken back as the deferring call returned, before any reply.
    let early_handle = lock(&placed)[0];
    assert_eq!(snapshot(&world, &objects), before);
    world.with_tables(|handles, _| assert!(handles.get(early_handle).is_err()));

    // The deferred reply places its own handle when it is written, and names it: kept.
    deliver_third(&world, server);
    assert_eq!(
        world.dispatcher.take_woken(),
        [Woken {
            thread: A,
            core: core_of(A),
            result: ResultCode::SUCCESS
        }]
    );
    let late_handle = lock(&placed)[1];
    assert_eq!(
        crate::ipc::message::reply_handles(&world.tls(A)),
        [late_handle]
    );
    assert_eq!(
        snapshot(&world, &objects),
        (before.0 + 1, vec![1, 2], before.2)
    );
    world.with_tables(|handles, _| assert_eq!(handles.get(late_handle), Ok(late)));
}

#[test]
fn a_sender_that_ended_while_its_request_was_in_flight_keeps_none_of_its_reply_handles() {
    let world = World::new();
    let objects = server_events(&world, 1);
    let (server, handle, placed) = naming_server(&world, None, objects[0]);
    let a = run_thread_for(&world, A);
    world.send(A, handle, &request());
    world
        .dispatcher
        .with::<(&mut Scheduler,), _>(|(scheduler,)| scheduler.terminate_thread(a))
        .unwrap();
    let before = snapshot(&world, &objects);
    let a_buffer = world.tls(A);
    deliver_third(&world, server);
    // The reply placed and named its handle, but nothing is written for the ended thread, and
    // the handle is taken back.
    let placed = lock(&placed).clone();
    assert_eq!(placed.len(), 1);
    assert_eq!(world.tls(A), a_buffer);
    assert_eq!(world.dispatcher.take_woken(), Vec::<Woken>::new());
    assert_eq!(snapshot(&world, &objects), before);
    world.with_tables(|handles, _| assert!(handles.get(placed[0]).is_err()));
}

#[test]
fn a_moved_object_not_yet_placed_stays_with_the_server() {
    // The server moves two objects; the handle between them is bad, so the second is never
    // placed. Eden closes a moved handle it did not place; an HLE server's object stays its
    // own, with the creator's reference (`dispatch/ipc/placed.rs`, "Divergences from Eden").
    let world = World::new();
    let events = server_events(&world, 2);
    let objects = vec![events[0], stale_object(&world), events[1]];
    let handle = world.session();
    let seen = Arc::new(Mutex::new(Placing::default()));
    world.first.script(place_each(objects, Arc::clone(&seen)));
    world.send(A, handle, &request());
    assert_eq!(lock(&seen).placed.len(), 1);
    world.with_tables(|_, scheduler| {
        for event in &events {
            assert!(matches!(
                scheduler.objects().get(*event),
                Ok(Object::ReadableEvent(_))
            ));
            assert_eq!(scheduler.objects().refs(*event), 1);
        }
    });
}

#[test]
fn a_placed_handle_the_guest_closed_meanwhile_is_left_alone() {
    // While the handler runs without the lock, the guest closes the placed handle and makes
    // handles until the table hands out the very same value for another object (the linear id
    // wraps after `MAX_LINEAR_ID`). Taking back by value alone would close the guest's own.
    let world = World::new();
    let objects = server_events(&world, 1);
    let handle = world.session();
    let own_session = world.object(handle);
    let (gate, keeper) = gate();
    let placed = Arc::new(Mutex::new(None));
    let keep = Arc::clone(&placed);
    let object = objects[0];
    world.first.script(Box::new(move |cx, _, message| {
        *lock(&keep) = Some(cx.insert_handle(object).unwrap());
        gate.pass();
        ReplyBuilder::cmif(KernelError::OutOfResource.code().raw())
            .build(message)
            .unwrap();
        Ok(Handled::Replied)
    }));
    let reused = std::thread::scope(|scope| {
        let sending = scope.spawn(|| world.send(A, handle, &request()));
        keeper.wait();
        let placed = lock(&placed).unwrap();
        let reused = world
            .dispatcher
            .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
                handles.remove(scheduler.objects_mut(), placed).unwrap();
                // Bounded: the linear id comes back round after `MAX_LINEAR_ID` adds.
                for _ in 0..=MAX_LINEAR_ID {
                    let added = handles.add(scheduler.objects_mut(), own_session).unwrap();
                    if added == placed {
                        return Some(added);
                    }
                    handles.remove(scheduler.objects_mut(), added).unwrap();
                }
                None
            });
        keeper.open();
        assert_eq!(sending.join().unwrap().0.outcome, Outcome::Completed);
        reused.expect("the table hands the value out again")
    });
    world.with_tables(|handles, scheduler| {
        // The guest's handle with the reused value still names its session.
        assert_eq!(handles.get(reused), Ok(own_session));
        assert_eq!(scheduler.objects().refs(own_session), 2);
        assert_eq!(scheduler.objects().refs(object), 1);
    });
}
