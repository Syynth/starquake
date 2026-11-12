//! Ordinary sessions: CMIF and TIPC requests, faults, replies that carry handles and
//! interfaces, and replies that do not fit.

use starquake_ipc::{
    BUFFER_SIZE, CMIF_HEADER_BYTES, CommandHeader, RAW_ALIGNMENT_WORDS, RequestBuilder, SFCO_MAGIC,
    TIPC_COMMAND_OFFSET,
};
use starquake_kernel::handle::Handle;
use starquake_kernel::id::SlabId;
use starquake_kernel::result::{KernelError, ResultCode};

use super::super::result::{INVALID_CMIF_REQUEST, NOT_SUPPORTED};
use super::{
    FAIL_PAYLOAD, INTERLEAVED_TAG, ODD_BYTE, Rig, STALE, buffer_of, child, only, payload, plain,
    request, toy,
};

const A: u32 = 0x1111_1111;
const B: u64 = 0x2222_2222_0000_0003;

fn add_payload() -> Vec<u8> {
    payload(|w| {
        w.write_u32(A).align(8).write_u64(B);
    })
}

#[test]
fn a_cmif_request_reaches_its_method_and_the_reply_is_these_bytes() {
    let mut rig = Rig::strict();
    let buffer = rig.send(&request(toy::ADD).data(&add_payload()));
    let sum = u64::from(A) + B;
    let [lo, hi] = [sum & 0xFFFF_FFFF, sum >> 32].map(|half| u32::try_from(half).unwrap());
    // Header (type 0, 8 raw words), padding to 16 bytes, `SFCO`, version, result, token, then
    // the u64. Everything after is zero: nothing of the request or of the stale buffer is left.
    let expected = buffer_of(&[
        0,
        8,
        0,
        0,
        SFCO_MAGIC,
        0,
        ResultCode::SUCCESS.raw(),
        0,
        lo,
        hi,
    ]);
    assert_eq!(buffer, expected);
}

#[test]
fn request_with_context_is_served_like_request() {
    let mut rig = Rig::strict();
    let with_context =
        RequestBuilder::cmif(starquake_ipc::CommandType::RequestWithContext, toy::ADD);
    let reply = plain(&rig.send(&with_context.data(&add_payload())));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!(reply.data, (u64::from(A) + B).to_le_bytes());
}

#[test]
fn a_tipc_request_uses_the_tipc_block_and_the_reply_is_these_bytes() {
    let mut rig = Rig::strict();
    let numbers = payload(|w| {
        w.write_u32(A).write_u32(7);
    });
    let buffer = rig.send(&RequestBuilder::tipc(toy::TIPC_SUM).data(&numbers));
    // The reply echoes the request's type; its raw data is one result word and then the u32.
    let message_type = u32::from(TIPC_COMMAND_OFFSET + toy::TIPC_SUM);
    assert_eq!(
        buffer,
        buffer_of(&[message_type, 2, ResultCode::SUCCESS.raw(), A + 7])
    );

    // The same id over CMIF is a different command (`add`), which needs sixteen bytes.
    assert_eq!(u32::from(toy::TIPC_SUM), toy::ADD);
    let reply = plain(&rig.send(&request(toy::ADD).data(&numbers)));
    assert_eq!(reply, only(INVALID_CMIF_REQUEST));
}

#[test]
fn a_method_listed_in_both_blocks_is_told_how_it_was_called() {
    let mut rig = Rig::strict();
    let fields = |buffer: &[u8]| {
        let data = plain(buffer).data;
        let word = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
        (word(24), word(28))
    };
    let cmif = rig.send(&request(toy::INSPECT));
    assert_eq!(fields(&cmif), (toy::INSPECT, 0));
    let tipc = rig.send(&RequestBuilder::tipc(toy::TIPC_INSPECT));
    assert_eq!(fields(&tipc), (u32::from(toy::TIPC_INSPECT), 1));
}

#[test]
fn a_payload_one_field_short_is_an_invalid_request() {
    let mut rig = Rig::strict();
    let full = add_payload();
    // The u64 needs bytes 8..16: twelve bytes is one word short, sixteen is exactly enough.
    let short = plain(&rig.send(&request(toy::ADD).data(&full[..12])));
    assert_eq!(short, only(INVALID_CMIF_REQUEST));
    let exact = plain(&rig.send(&request(toy::ADD).data(&full)));
    assert_eq!(exact.result, ResultCode::SUCCESS.raw());
}

#[test]
fn a_pushed_interface_becomes_a_new_session_and_a_moved_handle() {
    const TAG: u32 = 0xC0FE;
    let mut rig = Rig::strict();
    let reply = plain(&rig.send(&request(toy::OPEN_CHILD).data(&TAG.to_le_bytes())));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!((reply.copies.len(), reply.moves.len()), (0, 1));
    assert_eq!((reply.data.len(), reply.objects.len()), (0, 0));

    // The handle names a session the kernel created on this server, and requests on it reach
    // the child.
    let session = rig.object_of(reply.moves[0]);
    assert_eq!(rig.kernel.sessions(), [session]);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 2)
    );
    assert_eq!(rig.child_tag(session), TAG);
    let set = request(child::SET_TAG).data(&(TAG + 1).to_le_bytes());
    assert_eq!(
        plain(&rig.send_on(session, &set)),
        only(ResultCode::SUCCESS)
    );
    assert_eq!(rig.child_tag(session), TAG + 1);
}

#[test]
fn a_pushed_interface_over_tipc_is_a_moved_handle_too() {
    const TAG: u32 = 0x71BC;
    let mut rig = Rig::strict();
    let open = RequestBuilder::tipc(toy::TIPC_OPEN_CHILD).data(&TAG.to_le_bytes());
    let reply = plain(&rig.send(&open));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!(reply.moves.len(), 1);
    let session = rig.object_of(reply.moves[0]);
    assert_eq!(rig.child_tag(session), TAG);
}

#[test]
fn a_fault_replies_only_its_result_and_keeps_nothing_of_the_reply() {
    let mut rig = Rig::strict();
    let code = KernelError::Busy.code();
    let buffer = rig.send(&request(toy::FAIL).data(&code.raw().to_le_bytes()));
    // The payload, the copied handle and the pushed child the method had prepared are gone.
    assert_eq!(
        buffer,
        buffer_of(&[0, 6, 0, 0, SFCO_MAGIC, 0, code.raw(), 0])
    );
    assert!(!buffer.windows(8).any(|w| w == FAIL_PAYLOAD.to_le_bytes()));
    assert_eq!(rig.kernel.handle_count(), 0);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (1, 1)
    );
    assert_eq!(rig.kernel.sessions(), []);
}

#[test]
fn copied_handles_are_added_before_moved_ones_and_moves_keep_their_order() {
    let mut rig = Rig::strict();
    let reply = plain(&rig.send(&request(toy::INTERLEAVE)));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!((reply.copies.len(), reply.moves.len()), (1, 3));

    // The toy created its events in the order first, second, copied; the child's session came
    // after them. Object indices count up from the rig's own session.
    let base = rig.session.index();
    let index = |handle: u32| rig.object_of(handle).index();
    assert_eq!(index(reply.moves[0]), base + 1);
    assert_eq!(index(reply.moves[2]), base + 2);
    assert_eq!(index(reply.copies[0]), base + 3);
    let child = rig.object_of(reply.moves[1]);
    assert_eq!(rig.kernel.sessions(), [child]);

    // A handle table hands out its highest free index first, so the order the handles were added
    // in shows in their indices: the copy, then the moves as the method listed them.
    let added = [
        reply.copies[0],
        reply.moves[0],
        reply.moves[1],
        reply.moves[2],
    ];
    let slots = added.map(|handle| Handle::from_raw(handle).index());
    assert!(slots.is_sorted_by(|a, b| a > b), "{slots:?}");
    assert_eq!(rig.child_tag(child), INTERLEAVED_TAG);
}

#[test]
fn an_event_made_by_a_method_reaches_the_caller_as_a_copied_handle() {
    let mut rig = Rig::strict();
    let reply = plain(&rig.send(&request(toy::MAKE_EVENT)));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!((reply.copies.len(), reply.moves.len()), (1, 0));
    let event = rig.object_of(reply.copies[0]);
    assert_eq!(rig.kernel.signals(event), Some(1));
}

#[test]
fn a_method_copies_guest_memory_in_and_out_through_its_context() {
    const FROM: u64 = 0x10_0000;
    const TO: u64 = 0x20_0000;
    let mut rig = Rig::strict();
    rig.kernel.map(FROM, vec![1, 2, 3, 4]);
    rig.kernel.map(TO, vec![STALE; 6]);
    let addresses = |from: u64, to: u64| {
        payload(|w| {
            w.write_u64(from).write_u64(to);
        })
    };
    let reply = plain(&rig.send(&request(toy::POKE).data(&addresses(FROM, TO + 1))));
    assert_eq!(reply, only(ResultCode::SUCCESS));
    assert_eq!(
        rig.kernel.memory(TO, 6).unwrap(),
        [STALE, 4, 3, 2, 1, STALE]
    );

    // Memory the caller has not mapped is the kernel's error, for the read and for the write.
    let fault = only(KernelError::InvalidCurrentMemory.code());
    let unread = plain(&rig.send(&request(toy::POKE).data(&addresses(FROM + 1, TO))));
    assert_eq!(unread, fault);
    let unwritten = plain(&rig.send(&request(toy::POKE).data(&addresses(FROM, TO + 3))));
    assert_eq!(unwritten, fault);
    assert_eq!(
        rig.kernel.memory(TO, 6).unwrap(),
        [STALE, 4, 3, 2, 1, STALE]
    );
}

/// The most payload a CMIF reply with one copied and one moved handle can carry: what is left of
/// the buffer after the two header words, the special header, the two handles, the padding and
/// the `SFCO` header.
fn flood_room() -> usize {
    let before_raw = (CommandHeader::WORDS + 1 + 2).next_multiple_of(RAW_ALIGNMENT_WORDS);
    BUFFER_SIZE - before_raw * size_of::<u32>() - CMIF_HEADER_BYTES
}

#[test]
fn a_reply_that_exactly_fills_the_buffer_is_sent() {
    let mut rig = Rig::strict();
    let room = u32::try_from(flood_room()).unwrap();
    let buffer = rig.send(&request(toy::FLOOD).data(&room.to_le_bytes()));
    let reply = plain(&buffer);
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    assert_eq!(reply.data, vec![0xEE; flood_room()]);
    assert_eq!((reply.copies.len(), reply.moves.len()), (1, 1));
    assert_eq!(buffer.last(), Some(&0xEE));
}

#[test]
fn a_reply_one_byte_too_large_is_refused_and_nothing_of_it_is_kept() {
    let mut rig = Rig::strict();
    for too_many in [flood_room() + 1, BUFFER_SIZE, 2 * BUFFER_SIZE] {
        let n = u32::try_from(too_many).unwrap();
        let reply = plain(&rig.send(&request(toy::FLOOD).data(&n.to_le_bytes())));
        assert_eq!(reply, only(NOT_SUPPORTED), "{too_many} bytes");
        // No handle was added, no session opened, and the pushed child is gone.
        assert_eq!(rig.kernel.handle_count(), 0);
        assert_eq!(rig.kernel.sessions(), []);
        assert_eq!(
            (rig.server.session_count(), rig.server.interface_count()),
            (1, 1)
        );
    }
}

/// The same reply that a domain refuses (see the domain tests) is fine here: the new interface
/// travels as a handle, in front of the payload.
#[test]
fn a_one_byte_payload_and_a_new_interface_fit_an_ordinary_reply() {
    let mut rig = Rig::strict();
    let reply = plain(&rig.send(&request(toy::ODD)));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    // The payload is padded with zeros to a whole word.
    assert_eq!(reply.data, [ODD_BYTE, 0, 0, 0]);
    assert_eq!(reply.moves.len(), 1);
    let child = rig.object_of(reply.moves[0]);
    assert_eq!(rig.child_tag(child), u32::from(ODD_BYTE));
}
