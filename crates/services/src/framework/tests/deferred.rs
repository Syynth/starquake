//! Deferred replies: a method keeps a reply token, the request stays in flight, and the reply is
//! written when the token completes (`docs/specs/services.md`, port plan slice 3).

use starquake_ipc::{RequestBuilder, SFCO_MAGIC, TIPC_COMMAND_OFFSET};
use starquake_kernel::result::{KernelError, ResultCode};

use super::super::result::{DEFERRED_REPLY_DROPPED, NOT_SUPPORTED};
use super::super::targets::Domain;
use super::super::{Handled, Reply, ServerError};
use super::{
    Child, FAIL_PAYLOAD, HASTY, Rig, STALE, buffer_of, built, child, domain_request, in_domain,
    only, parked, parked_one, plain, request, toy,
};

/// What a completed token answers with in these tests.
const ANSWER: u64 = 0x0A0B_0C0D_0102_0304;

fn answer() -> Reply {
    let mut reply = Reply::default();
    reply.payload().write_u64(ANSWER);
    reply
}

/// The two halves of [`ANSWER`] as the words of a reply.
fn answer_words() -> [u32; 2] {
    [ANSWER & 0xFFFF_FFFF, ANSWER >> 32].map(|half| u32::try_from(half).unwrap())
}

#[test]
fn a_completed_token_wakes_the_caller_and_the_reply_is_these_bytes() {
    let mut rig = Rig::strict();
    let session = rig.session;
    let token = rig.defer_on(session, &request(toy::WAIT));
    // The request is in flight and nobody has been woken.
    assert_eq!(rig.server.pending().collect::<Vec<_>>(), [(token, session)]);
    assert!(rig.server.queue().is_deferred(session));
    assert!(rig.server.take_completed().is_none());
    assert_eq!(rig.wakes.count(), 0);

    // Another thread finishes the work.
    let parked = parked_one();
    assert_eq!(parked.id(), token);
    std::thread::spawn(move || parked.complete(answer()))
        .join()
        .unwrap();
    assert_eq!(rig.wakes.count(), 1);

    let (to, buffer) = rig.deliver();
    assert_eq!(to, session);
    // The reply a method would have sent for the same payload: header (8 raw words), padding,
    // `SFCO`, version, result, token, the u64. Nothing of what `wait` wrote before deferring.
    let [lo, hi] = answer_words();
    let success = ResultCode::SUCCESS.raw();
    let expected = [0, 8, 0, 0, SFCO_MAGIC, 0, success, 0, lo, hi];
    assert_eq!(buffer, buffer_of(&expected));
    assert_ne!(ANSWER, FAIL_PAYLOAD);

    // Nothing is in flight any more, and the session takes its next request.
    assert_eq!(rig.server.pending().count(), 0);
    assert!(!rig.server.queue().is_deferred(session));
    assert!(rig.server.take_completed().is_none());
    assert_eq!(
        plain(&rig.send(&request(toy::INSPECT))).result,
        ResultCode::SUCCESS.raw()
    );
}

#[test]
fn a_dropped_token_replies_with_an_error() {
    let mut rig = Rig::strict();
    rig.defer_on(rig.session, &request(toy::WAIT));
    drop(parked_one());
    // Dropping tells the server too: the caller must not wait for ever.
    assert_eq!(rig.wakes.count(), 1);
    let (_, buffer) = rig.deliver();
    assert_eq!(plain(&buffer), only(DEFERRED_REPLY_DROPPED));
    assert_eq!(rig.server.pending().count(), 0);
}

#[test]
fn a_failed_token_replies_its_code_and_nothing_else() {
    let mut rig = Rig::strict();
    rig.defer_on(rig.session, &request(toy::WAIT));
    let code = KernelError::TimedOut.code();
    parked_one().fail(code);
    let (_, buffer) = rig.deliver();
    assert_eq!(plain(&buffer), only(code));
    // The whole buffer was rewritten: none of its stale bytes is left.
    assert!(!buffer.contains(&STALE));
}

#[test]
fn a_deferred_tipc_request_is_answered_in_tipc() {
    let mut rig = Rig::strict();
    rig.defer_on(rig.session, &RequestBuilder::tipc(toy::TIPC_WAIT));
    parked_one().complete(answer());
    let (_, buffer) = rig.deliver();
    // The reply echoes the request's type; its raw data is the result word and the u64.
    let message_type = u32::from(TIPC_COMMAND_OFFSET + toy::TIPC_WAIT);
    let [lo, hi] = answer_words();
    let expected = [message_type, 3, ResultCode::SUCCESS.raw(), lo, hi];
    assert_eq!(buffer, buffer_of(&expected));
}

#[test]
fn a_deferred_domain_request_is_answered_in_its_domain() {
    const TAG: u32 = 0x7A6;
    let mut rig = Rig::domain();
    let wait = domain_request(Domain::ROOT_OBJECT_ID, toy::WAIT);
    rig.defer_on(rig.session, &wait);
    // The completion returns a new interface: in a domain it becomes an object id.
    let mut reply = answer();
    reply.push_session(Box::new(Child { tag: TAG }));
    parked_one().complete(reply);
    let (_, buffer) = rig.deliver();
    let reply = in_domain(&buffer);
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!(reply.data, ANSWER.to_le_bytes());
    assert_eq!(reply.objects, [Domain::ROOT_OBJECT_ID + 1]);
    assert_eq!(rig.kernel.sessions(), []);
    // The object is there, and is the child the completion made.
    let tag = in_domain(&rig.send(&domain_request(reply.objects[0], child::TAG)));
    assert_eq!(tag.data, TAG.to_le_bytes());
}

#[test]
fn a_completion_can_hand_out_handles_and_sessions() {
    const TAG: u32 = 0x61;
    let mut rig = Rig::strict();
    rig.defer_on(rig.session, &request(toy::WAIT));
    let event = rig.kernel.new_object();
    let mut reply = Reply::default();
    reply.copy_handle(event);
    reply.push_session(Box::new(Child { tag: TAG }));
    parked_one().complete(reply);
    let (_, buffer) = rig.deliver();
    let reply = plain(&buffer);
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    // The handles are in the table of the caller the kernel spoke for at delivery.
    assert_eq!(rig.object_of(reply.copies[0]), event);
    let child = rig.object_of(reply.moves[0]);
    assert_eq!(rig.child_tag(child), TAG);
}

#[test]
fn a_completion_whose_handles_do_not_fit_replies_the_kernels_error() {
    // Room for no handle at all.
    let mut rig = Rig::with_handle_table(super::UnknownPolicy::Strict, 1);
    let filler = rig.kernel.new_object();
    super::Kernel::insert_handle(&mut rig.kernel, filler).unwrap();
    rig.defer_on(rig.session, &request(toy::WAIT));
    let mut reply = Reply::default();
    reply.copy_handle(filler);
    parked_one().complete(reply);
    let (_, buffer) = rig.deliver();
    assert_eq!(plain(&buffer), only(KernelError::OutOfHandles.code()));
}

#[test]
fn a_deferred_request_does_not_hold_the_server_but_holds_its_session() {
    let mut rig = Rig::strict();
    let (first, second) = (rig.session, rig.second_session());
    rig.defer_on(first, &request(toy::WAIT));

    // The server takes a request from another session meanwhile.
    let other = plain(&rig.send_on(second, &request(toy::INSPECT)));
    assert_eq!(other.result, ResultCode::SUCCESS.raw());

    // The deferred session starts no second request: the message is not even read.
    let again = request(toy::INSPECT);
    let (handled, buffer) = rig.try_send_on(first, &again);
    assert_eq!(handled, Err(ServerError::Busy(first)));
    assert_eq!(buffer, built(&again));
    assert_eq!(
        ServerError::Busy(first).to_string(),
        format!("session {first:?} cannot start a request now")
    );
    // The refusal left no sender waiting.
    assert_eq!(rig.server.queue().waiting(first), 0);

    parked_one().complete(answer());
    rig.deliver();
    let after = plain(&rig.send_on(first, &again));
    assert_eq!(after.result, ResultCode::SUCCESS.raw());
}

#[test]
fn completions_are_delivered_in_the_order_they_were_posted() {
    let mut rig = Rig::strict();
    let (first, second) = (rig.session, rig.second_session());
    let first_token = rig.defer_on(first, &request(toy::WAIT));
    let second_token = rig.defer_on(second, &request(toy::WAIT));
    assert!(first_token < second_token);
    let mut tokens = parked();
    // The later request is answered first.
    tokens.pop().unwrap().fail(NOT_SUPPORTED);
    tokens.pop().unwrap().complete(answer());
    assert_eq!(rig.wakes.count(), 2);

    let completed = rig.server.take_completed().unwrap();
    assert_eq!(
        (completed.token(), completed.session()),
        (second_token, second)
    );
    let mut buffer = built(&request(toy::WAIT));
    let delivered = rig.server.deliver(&mut rig.kernel, completed, &mut buffer);
    assert_eq!(delivered, Ok(()));
    assert_eq!(plain(&buffer), only(NOT_SUPPORTED));
    let (session, buffer) = rig.deliver();
    assert_eq!(session, first);
    assert_eq!(plain(&buffer).data, ANSWER.to_le_bytes());
}

#[test]
fn deferring_without_a_token_is_answered_at_once_with_an_error() {
    let mut rig = Rig::strict();
    let reply = plain(&rig.send(&request(toy::FORGET)));
    assert_eq!(reply, only(DEFERRED_REPLY_DROPPED));
    assert_eq!(rig.server.pending().count(), 0);
    assert!(!rig.server.queue().is_deferred(rig.session));
}

#[test]
fn a_request_has_one_token() {
    let mut rig = Rig::strict();
    let reply = plain(&rig.send(&request(toy::WAIT_TWICE)));
    assert_eq!(reply, only(NOT_SUPPORTED));
    // The token it did get answers nothing: the request was answered already.
    assert_eq!(rig.server.pending().count(), 0);
    parked_one().complete(answer());
    assert!(rig.server.take_completed().is_none());
}

#[test]
fn a_token_taken_by_a_method_that_answers_at_once_is_void() {
    let mut rig = Rig::strict();
    let reply = plain(&rig.send(&request(toy::HASTY)));
    assert_eq!(reply.data, HASTY.to_le_bytes());
    assert_eq!(rig.server.pending().count(), 0);
    // Completing or dropping it later changes nothing, and nothing is left to deliver.
    parked_one().complete(answer());
    assert!(rig.server.take_completed().is_none());
    assert_eq!(
        plain(&rig.send(&request(toy::INSPECT))).result,
        ResultCode::SUCCESS.raw()
    );
}

#[test]
fn each_deferred_request_gets_the_next_token_id() {
    let mut rig = Rig::strict();
    let (first, second) = (rig.session, rig.second_session());
    let a = rig.defer_on(first, &request(toy::WAIT));
    // A token a method wasted still used up an id.
    rig.send_on(second, &request(toy::HASTY));
    let b = rig.defer_on(second, &request(toy::WAIT));
    assert_eq!((a.raw(), b.raw()), (0, 2));
    drop(parked());
}

#[test]
fn closing_a_session_abandons_its_deferred_request() {
    let mut rig = Rig::strict();
    let (first, second) = (rig.session, rig.second_session());
    rig.defer_on(first, &request(toy::WAIT));
    rig.defer_on(second, &request(toy::WAIT));
    rig.server.close(first).unwrap();
    assert_eq!(rig.server.pending().count(), 1);
    assert_eq!(rig.server.queue().sessions().collect::<Vec<_>>(), [second]);

    // The closed session's token completes into nothing; the other one is delivered.
    let mut tokens = parked();
    tokens.remove(0).complete(answer());
    tokens.remove(0).fail(NOT_SUPPORTED);
    let (session, buffer) = rig.deliver();
    assert_eq!(session, second);
    assert_eq!(plain(&buffer), only(NOT_SUPPORTED));
    assert!(rig.server.take_completed().is_none());
}

#[test]
fn a_session_closed_before_its_reply_is_delivered_is_an_error() {
    let mut rig = Rig::strict();
    let session = rig.session;
    rig.defer_on(session, &request(toy::WAIT));
    parked_one().complete(answer());
    let completed = rig.server.take_completed().unwrap();
    rig.server.close(session).unwrap();
    let mut buffer = vec![STALE; starquake_ipc::BUFFER_SIZE];
    let delivered = rig.server.deliver(&mut rig.kernel, completed, &mut buffer);
    assert_eq!(delivered, Err(ServerError::UnknownSession(session)));
    assert!(buffer.iter().all(|byte| *byte == STALE));
}

#[test]
fn a_reply_that_cannot_be_written_still_ends_the_request() {
    let mut rig = Rig::strict();
    let session = rig.session;
    rig.defer_on(session, &request(toy::WAIT));
    parked_one().complete(answer());
    let completed = rig.server.take_completed().unwrap();
    // One byte short of a message buffer.
    let mut buffer = vec![STALE; starquake_ipc::BUFFER_SIZE - 1];
    let delivered = rig.server.deliver(&mut rig.kernel, completed, &mut buffer);
    let too_short = ServerError::BufferTooShort { len: buffer.len() };
    assert_eq!(delivered, Err(too_short));
    assert!(buffer.iter().all(|byte| *byte == STALE));
    // The session is not stuck behind a request nobody can answer.
    assert!(!rig.server.queue().is_deferred(session));
    assert_eq!(
        plain(&rig.send(&request(toy::INSPECT))).result,
        ResultCode::SUCCESS.raw()
    );
}

#[test]
fn a_close_message_behind_nothing_closes_at_once() {
    // A close is a request like any other to the queue: it leaves nothing in flight.
    let mut rig = Rig::strict();
    let session = rig.session;
    let (handled, _) = rig.try_send_on(session, &RequestBuilder::close());
    assert_eq!(handled, Ok(Handled::Closed));
    assert_eq!(rig.server.queue().sessions().count(), 0);
    assert_eq!(rig.server.queue().running(), 0);
}
