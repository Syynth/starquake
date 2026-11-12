//! The server's request queue, driven as the kernel's session code will drive it: senders are
//! submitted, the one that may start is handled, and the rest wait their turn
//! (`docs/specs/services.md`, "Concurrency").

use starquake_ipc::BUFFER_SIZE;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::ResultCode;

use super::super::{Admission, DEFAULT_WORKERS, QueueError, ServerError};
use super::{Rig, parked_one, plain, request, toy};

/// Handles a request on `session` that the queue said may start, and checks it was answered.
fn run(rig: &mut Rig, session: ObjectId) {
    let reply = plain(&rig.send_on(session, &request(toy::INSPECT)));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
}

#[test]
fn two_sessions_of_one_server_do_not_run_handlers_at_once() {
    let mut rig = Rig::strict();
    let (first, second) = (rig.session, rig.second_session());
    assert_eq!(rig.server.queue().workers(), DEFAULT_WORKERS);

    assert_eq!(rig.server.submit(first), Ok(Admission::Start));
    // The first request has the server's one worker until it is handled.
    assert_eq!(rig.server.submit(second), Ok(Admission::Queued));
    assert_eq!(rig.server.start_next(), None);
    // Handling the second out of turn is refused, and its sender still waits.
    let (handled, _) = rig.try_send_on(second, &request(toy::INSPECT));
    assert_eq!(handled, Err(ServerError::Busy(second)));
    assert_eq!(rig.server.queue().waiting(second), 1);

    run(&mut rig, first);
    assert_eq!(rig.server.start_next(), Some(second));
    assert_eq!(rig.server.start_next(), None);
    run(&mut rig, second);
    assert_eq!(rig.server.queue().sessions().count(), 0);
}

#[test]
fn a_second_sender_on_a_session_queues_behind_a_deferred_request() {
    let mut rig = Rig::strict();
    let session = rig.session;
    assert_eq!(rig.server.submit(session), Ok(Admission::Start));
    rig.defer_on(session, &request(toy::WAIT));

    // The worker is free, but the session has a request in flight.
    assert_eq!(rig.server.submit(session), Ok(Admission::Queued));
    assert_eq!(rig.server.start_next(), None);

    parked_one().fail(ResultCode::SUCCESS);
    rig.deliver();
    // The deferred request is answered: the waiting sender's turn.
    assert_eq!(rig.server.start_next(), Some(session));
    run(&mut rig, session);
    assert_eq!(rig.server.start_next(), None);
}

#[test]
fn waiting_senders_start_in_the_order_their_sessions_became_ready() {
    let mut rig = Rig::strict();
    let first = rig.session;
    let (second, third) = (rig.second_session(), rig.second_session());
    assert_eq!(rig.server.submit(first), Ok(Admission::Start));
    assert_eq!(rig.server.submit(third), Ok(Admission::Queued));
    assert_eq!(rig.server.submit(second), Ok(Admission::Queued));
    assert_eq!(rig.server.submit(first), Ok(Admission::Queued));
    run(&mut rig, first);
    for expected in [third, second, first] {
        assert_eq!(rig.server.start_next(), Some(expected));
        run(&mut rig, expected);
    }
    assert_eq!(rig.server.start_next(), None);
}

#[test]
fn a_sender_that_gives_up_leaves_the_queue() {
    let mut rig = Rig::strict();
    let (first, second) = (rig.session, rig.second_session());
    rig.server.submit(first).unwrap();
    rig.server.submit(second).unwrap();
    assert_eq!(rig.server.withdraw(second), Ok(()));
    assert_eq!(
        rig.server.withdraw(second),
        Err(ServerError::Queue(QueueError::NotWaiting(second)))
    );
    run(&mut rig, first);
    assert_eq!(rig.server.start_next(), None);
}

#[test]
fn only_open_sessions_are_queued() {
    let mut rig = Rig::strict();
    let stranger = rig.kernel.new_object();
    assert_eq!(
        rig.server.submit(stranger),
        Err(ServerError::UnknownSession(stranger))
    );
    assert_eq!(rig.server.queue().sessions().count(), 0);
}

#[test]
fn closing_a_session_drops_its_waiting_senders_and_frees_the_worker() {
    let mut rig = Rig::strict();
    let (first, second) = (rig.session, rig.second_session());
    rig.server.submit(first).unwrap();
    rig.server.submit(first).unwrap();
    rig.server.submit(second).unwrap();
    // The session is closed while its request is running (its client end was closed).
    rig.server.close(first).unwrap();
    assert_eq!(rig.server.start_next(), Some(second));
    run(&mut rig, second);
    assert_eq!(rig.server.start_next(), None);
}

#[test]
fn a_request_the_server_fails_on_still_frees_the_worker() {
    // A buffer one byte short, which the server cannot read a message from, and a whole buffer
    // that does not parse.
    let short = vec![0; BUFFER_SIZE - 1];
    let garbage = vec![0xFF; BUFFER_SIZE];
    for mut bad in [short, garbage] {
        let mut rig = Rig::strict();
        let (first, second) = (rig.session, rig.second_session());
        assert_eq!(rig.server.submit(first), Ok(Admission::Start));
        assert_eq!(rig.server.submit(second), Ok(Admission::Queued));
        let handled = rig.server.handle(&mut rig.kernel, first, &mut bad);
        if bad.len() < BUFFER_SIZE {
            assert_eq!(handled, Err(ServerError::BufferTooShort { len: bad.len() }));
        } else {
            assert!(matches!(handled, Err(ServerError::Malformed(_))));
        }
        // The request is over, on the error path too: the worker it held is free, and the
        // session that waited starts.
        assert!(!rig.server.queue().is_running(first));
        assert_eq!(rig.server.queue().running(), 0);
        assert_eq!(rig.server.start_next(), Some(second));
        assert_eq!(rig.server.start_next(), None);
        run(&mut rig, second);
        assert_eq!(rig.server.queue().sessions().count(), 0);
    }
}

#[test]
fn a_short_buffer_does_not_let_a_request_start_out_of_turn() {
    let mut rig = Rig::strict();
    let (first, second) = (rig.session, rig.second_session());
    assert_eq!(rig.server.submit(first), Ok(Admission::Start));
    assert_eq!(rig.server.submit(second), Ok(Admission::Queued));
    // The second session's request has not started: it is refused as busy before the buffer is
    // looked at, its sender still waits, and the first is still the running one.
    let mut short = vec![0; BUFFER_SIZE - 1];
    let handled = rig.server.handle(&mut rig.kernel, second, &mut short);
    assert_eq!(handled, Err(ServerError::Busy(second)));
    assert_eq!(rig.server.queue().waiting(second), 1);
    assert!(rig.server.queue().is_running(first));
    run(&mut rig, first);
    assert_eq!(rig.server.start_next(), Some(second));
}

#[test]
fn a_request_does_not_start_ahead_of_a_session_that_is_ready() {
    let mut rig = Rig::strict();
    let first = rig.session;
    let (second, third, fourth) = (
        rig.second_session(),
        rig.second_session(),
        rig.second_session(),
    );
    assert_eq!(rig.server.submit(first), Ok(Admission::Start));
    assert_eq!(rig.server.submit(second), Ok(Admission::Queued));
    run(&mut rig, first);
    // The worker is free, and the second session is ready and has not been started yet
    // (`start_next` was not called). A new sender waits behind it,
    assert_eq!(rig.server.queue().running(), 0);
    assert_eq!(rig.server.submit(third), Ok(Admission::Queued));
    // and a request given to `handle` without `submit` is refused and leaves nobody waiting.
    let (handled, _) = rig.try_send_on(fourth, &request(toy::INSPECT));
    assert_eq!(handled, Err(ServerError::Busy(fourth)));
    assert_eq!(rig.server.queue().waiting(fourth), 0);

    for expected in [second, third] {
        assert_eq!(rig.server.start_next(), Some(expected));
        assert_eq!(rig.server.start_next(), None);
        run(&mut rig, expected);
    }
    assert_eq!(rig.server.start_next(), None);
    // Nobody is ready now: the same request starts by itself.
    run(&mut rig, fourth);
    assert_eq!(rig.server.queue().sessions().count(), 0);
}
