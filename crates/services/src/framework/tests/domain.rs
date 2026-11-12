//! Domains: dispatch by object id, objects in and out, closing objects, and the table's limit.

use starquake_ipc::{RequestBuilder, SFCO_MAGIC};
use starquake_kernel::result::{KernelError, ResultCode};

use super::super::result::{
    INVALID_CMIF_REQUEST, NOT_SUPPORTED, OUT_OF_DOMAIN_ENTRIES, TARGET_NOT_FOUND,
};
use super::super::targets::Domain;
use super::super::{Handled, MAX_DOMAIN_OBJECTS, UnknownPolicy};
use super::{
    Answer, INTERLEAVED_TAG, Rig, buffer_of, child, domain_request, in_domain, only, payload,
    plain, toy,
};

/// The object id of the session's own interface after conversion.
const ROOT: u32 = Domain::ROOT_OBJECT_ID;
/// The id the first object added to a fresh domain gets.
const FIRST: u32 = ROOT + 1;

fn send(rig: &mut Rig, build: &RequestBuilder) -> Answer {
    in_domain(&rig.send(build))
}

/// Opens a child in the domain and returns its object id.
fn open_child(rig: &mut Rig, tag: u32) -> u32 {
    let reply = send(
        rig,
        &domain_request(ROOT, toy::OPEN_CHILD).data(&tag.to_le_bytes()),
    );
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!(reply.objects.len(), 1);
    reply.objects[0]
}

fn tag_of(rig: &mut Rig, object_id: u32) -> u32 {
    let reply = send(rig, &domain_request(object_id, child::TAG));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    u32::from_le_bytes(reply.data.try_into().unwrap())
}

#[test]
fn a_domain_request_reaches_the_object_it_names_and_the_reply_is_these_bytes() {
    let mut rig = Rig::domain();
    let numbers = payload(|w| {
        w.write_u32(5).align(8).write_u64(6);
    });
    let buffer = rig.send(&domain_request(ROOT, toy::ADD).data(&numbers));
    // Header (12 raw words), padding, the domain header (no objects), `SFCO`, then the u64.
    let success = ResultCode::SUCCESS.raw();
    let expected = [0, 12, 0, 0, 0, 0, 0, 0, SFCO_MAGIC, 0, success, 0, 11, 0];
    assert_eq!(buffer, buffer_of(&expected));
}

#[test]
fn a_pushed_interface_becomes_an_object_id_and_no_session() {
    const TAG: u32 = 0xD0D0;
    let mut rig = Rig::domain();
    let buffer = rig.send(&domain_request(ROOT, toy::OPEN_CHILD).data(&TAG.to_le_bytes()));
    // The domain header counts one object; its id trails the (empty) data.
    let success = ResultCode::SUCCESS.raw();
    let expected = [0, 11, 0, 0, 1, 0, 0, 0, SFCO_MAGIC, 0, success, 0, FIRST];
    assert_eq!(buffer, buffer_of(&expected));

    // No kernel session and no handle: the object lives in the domain.
    assert_eq!(rig.kernel.sessions(), []);
    assert_eq!(rig.kernel.handle_count(), 0);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (1, 2)
    );
    assert_eq!(tag_of(&mut rig, FIRST), TAG);
}

#[test]
fn kernel_objects_stay_handles_in_a_domain_while_interfaces_become_ids() {
    let mut rig = Rig::domain();
    let reply = send(&mut rig, &domain_request(ROOT, toy::INTERLEAVE));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!((reply.copies.len(), reply.moves.len()), (1, 2));
    assert_eq!(reply.objects, [FIRST]);
    assert_eq!(rig.kernel.sessions(), []);
    assert_eq!(tag_of(&mut rig, FIRST), INTERLEAVED_TAG);
}

#[test]
fn an_input_object_reaches_the_method_as_its_interface() {
    const TAG: u32 = 0x0BEC;
    let mut rig = Rig::domain();
    let other = open_child(&mut rig, 1);
    let object = open_child(&mut rig, TAG);
    assert_ne!(other, object);
    // `child_tag` reads input object 0 and bumps its tag: the method had the object itself.
    let ask = domain_request(ROOT, toy::CHILD_TAG).object_id(object);
    assert_eq!(send(&mut rig, &ask).data, TAG.to_le_bytes());
    assert_eq!(send(&mut rig, &ask).data, (TAG + 1).to_le_bytes());
    assert_eq!(tag_of(&mut rig, object), TAG + 2);
    assert_eq!(tag_of(&mut rig, other), 1);
}

#[test]
fn an_input_object_that_is_missing_unknown_or_the_wrong_interface_is_an_invalid_request() {
    let mut rig = Rig::domain();
    let invalid = only(INVALID_CMIF_REQUEST);
    // No input object at all.
    assert_eq!(
        send(&mut rig, &domain_request(ROOT, toy::CHILD_TAG)),
        invalid
    );
    // An id that names nothing, and id zero.
    for id in [FIRST, 0] {
        let ask = domain_request(ROOT, toy::CHILD_TAG).object_id(id);
        assert_eq!(send(&mut rig, &ask), invalid, "input object id {id}");
    }
    // A different interface.
    let mute = send(&mut rig, &domain_request(ROOT, toy::OPEN_MUTE)).objects[0];
    let ask = domain_request(ROOT, toy::CHILD_TAG).object_id(mute);
    assert_eq!(send(&mut rig, &ask), invalid);
    // The very object the command was sent to.
    let ask = domain_request(ROOT, toy::CHILD_TAG).object_id(ROOT);
    assert_eq!(send(&mut rig, &ask), invalid);
    // And the root still works afterwards.
    assert_eq!(open_child(&mut rig, 3), mute + 1);
}

/// Divergence from Eden, which writes no reply for an id past the table or closed, and indexes
/// out of range for id zero.
#[test]
fn a_request_to_an_object_id_that_names_nothing_is_target_not_found() {
    let mut rig = Rig::domain();
    for id in [0, FIRST, u32::MAX] {
        let reply = send(&mut rig, &domain_request(id, toy::ADD));
        assert_eq!(reply, only(TARGET_NOT_FOUND), "object id {id}");
    }
}

#[test]
fn closing_an_object_frees_it_and_its_id() {
    let mut rig = Rig::domain();
    let first = open_child(&mut rig, 1);
    let second = open_child(&mut rig, 2);
    assert_eq!((first, second), (FIRST, FIRST + 1));
    assert_eq!(rig.server.interface_count(), 3);

    let buffer = rig.send(&RequestBuilder::domain_close(first));
    // A domain reply with no objects and no data.
    let success = ResultCode::SUCCESS.raw();
    let expected = [0, 10, 0, 0, 0, 0, 0, 0, SFCO_MAGIC, 0, success, 0];
    assert_eq!(buffer, buffer_of(&expected));
    assert_eq!(rig.server.interface_count(), 2);
    assert_eq!(
        send(&mut rig, &domain_request(first, child::TAG)),
        only(TARGET_NOT_FOUND)
    );
    assert_eq!(tag_of(&mut rig, second), 2);

    // The id closed last is the next one handed out.
    assert_eq!(open_child(&mut rig, 9), first);
    assert_eq!(tag_of(&mut rig, first), 9);
}

/// Closing an id that names nothing still replies success, and changes nothing. (Eden builds a
/// success reply for a close too, but never writes it; see the framework's module docs.)
#[test]
fn closing_an_id_that_names_nothing_replies_success() {
    let mut rig = Rig::domain();
    let object = open_child(&mut rig, 1);
    for id in [0, object + 1, u32::MAX] {
        let reply = send(&mut rig, &RequestBuilder::domain_close(id));
        assert_eq!(reply, only(ResultCode::SUCCESS), "object id {id}");
    }
    assert_eq!(rig.server.interface_count(), 2);
    assert_eq!(
        send(&mut rig, &RequestBuilder::domain_close(object)),
        only(ResultCode::SUCCESS)
    );
    assert_eq!(
        send(&mut rig, &RequestBuilder::domain_close(object)),
        only(ResultCode::SUCCESS)
    );
    assert_eq!(rig.server.interface_count(), 1);
}

#[test]
fn the_root_object_can_be_closed_and_the_session_stays_open() {
    let mut rig = Rig::domain();
    let object = open_child(&mut rig, 4);
    assert_eq!(
        send(&mut rig, &RequestBuilder::domain_close(ROOT)),
        only(ResultCode::SUCCESS)
    );
    assert_eq!(
        send(&mut rig, &domain_request(ROOT, toy::ADD)),
        only(TARGET_NOT_FOUND)
    );
    assert_eq!(tag_of(&mut rig, object), 4);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (1, 1)
    );
}

#[test]
fn closing_the_session_drops_every_object_in_its_domain() {
    let mut rig = Rig::domain();
    open_child(&mut rig, 1);
    open_child(&mut rig, 2);
    assert_eq!(rig.server.interface_count(), 3);
    let (handled, _) = rig.try_send_on(rig.session, &RequestBuilder::close());
    assert_eq!(handled, Ok(Handled::Closed));
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (0, 0)
    );
}

/// A handle the kernel refuses stops the reply before any object is added to the domain.
#[test]
fn a_reply_that_fails_on_a_handle_adds_no_object_to_the_domain() {
    // Room for two handles: `interleave` adds its copy and its first move, comes to the child,
    // and then fails on its second move.
    let mut rig = Rig::with_handle_table(UnknownPolicy::Strict, 2);
    rig.convert();
    let reply = send(&mut rig, &domain_request(ROOT, toy::INTERLEAVE));
    assert_eq!(reply, only(KernelError::OutOfHandles.code()));
    assert_eq!(rig.kernel.handle_count(), 2);
    assert_eq!(rig.server.interface_count(), 1);
    // The id the child would have had is still free.
    assert_eq!(open_child(&mut rig, 6), FIRST);
}

/// Divergence from Eden, which sends a TIPC request on a domain to the session's first handler.
#[test]
fn a_tipc_request_on_a_domain_is_not_supported() {
    let mut rig = Rig::domain();
    let numbers = payload(|w| {
        w.write_u32(1).write_u32(2);
    });
    let reply = plain(&rig.send(&RequestBuilder::tipc(toy::TIPC_SUM).data(&numbers)));
    assert_eq!(reply, only(NOT_SUPPORTED));
}

/// How many children one `spawn` can return in a domain: the object ids have to fit the reply.
const BATCH: u32 = 32;

fn spawn(rig: &mut Rig, count: u32) -> Answer {
    let numbers = payload(|w| {
        w.write_u32(count).write_u32(0);
    });
    send(rig, &domain_request(ROOT, toy::SPAWN).data(&numbers))
}

#[test]
fn a_domain_is_full_exactly_at_its_limit() {
    let mut rig = Rig::domain();
    let limit = u32::try_from(MAX_DOMAIN_OBJECTS).unwrap();
    // The root is object 1; fill the rest.
    let mut live = 1;
    while live < limit {
        let count = BATCH.min(limit - live);
        let reply = spawn(&mut rig, count);
        assert_eq!(reply.result, ResultCode::SUCCESS.raw());
        assert_eq!(reply.objects.len(), usize::try_from(count).unwrap());
        live += count;
    }
    assert_eq!(rig.server.interface_count(), MAX_DOMAIN_OBJECTS);
    // The last id handed out is the limit itself.
    assert_eq!(
        send(&mut rig, &domain_request(limit, child::TAG)).result,
        ResultCode::SUCCESS.raw()
    );

    // One more does not fit, and is not kept.
    assert_eq!(spawn(&mut rig, 1), only(OUT_OF_DOMAIN_ENTRIES));
    assert_eq!(rig.server.interface_count(), MAX_DOMAIN_OBJECTS);

    // With one id free, a reply that needs two is refused whole; one fits.
    assert_eq!(
        send(&mut rig, &RequestBuilder::domain_close(limit)),
        only(ResultCode::SUCCESS)
    );
    assert_eq!(spawn(&mut rig, 2), only(OUT_OF_DOMAIN_ENTRIES));
    assert_eq!(rig.server.interface_count(), MAX_DOMAIN_OBJECTS - 1);
    assert_eq!(spawn(&mut rig, 1).objects, [limit]);
    assert_eq!(rig.server.interface_count(), MAX_DOMAIN_OBJECTS);
}

#[test]
fn several_new_objects_get_ids_in_the_order_they_were_pushed() {
    let mut rig = Rig::domain();
    let numbers = payload(|w| {
        w.write_u32(3).write_u32(0x40);
    });
    let reply = send(&mut rig, &domain_request(ROOT, toy::SPAWN).data(&numbers));
    assert_eq!(reply.objects, [FIRST, FIRST + 1, FIRST + 2]);
    for (id, tag) in reply.objects.iter().zip(0x40..) {
        assert_eq!(tag_of(&mut rig, *id), tag);
    }
}

/// The limit listed under "Not here yet" in the framework's module docs: the IPC crate cannot
/// yet put object ids straight after a payload that is not whole words, so the reply is refused.
#[test]
fn a_domain_reply_with_objects_after_an_unaligned_payload_is_refused_whole() {
    let mut rig = Rig::domain();
    let reply = send(&mut rig, &domain_request(ROOT, toy::ODD));
    assert_eq!(reply, only(NOT_SUPPORTED));
    assert_eq!(rig.server.interface_count(), 1);
    // The id the child would have had is still free.
    assert_eq!(open_child(&mut rig, 6), FIRST);
}
