//! Handles and the process id a request carries, and what happens when the caller's handle table
//! is full.

use starquake_ipc::RequestBuilder;
use starquake_kernel::handle::Handle;
use starquake_kernel::id::{ObjectId, SlabId};
use starquake_kernel::result::{KernelError, ResultCode};

use super::super::result::INVALID_CMIF_REQUEST;
use super::super::{Kernel, UnknownPolicy};
use super::{ABSENT, Rig, only, payload, plain, request, toy};

const PROCESS_ID: u64 = 0x0000_0051_0000_0003;

/// A handle in the caller's table to a fresh object.
fn handle_to_new_object(rig: &mut Rig) -> (u32, ObjectId) {
    let object = rig.kernel.new_object();
    (rig.kernel.insert_handle(object).unwrap().raw(), object)
}

/// What `inspect` replies for a request with these parts.
fn inspected(process_id: Option<u64>, copies: &[ObjectId], moves: &[ObjectId]) -> Vec<u8> {
    let index = |list: &[ObjectId], at: usize| list.get(at).map_or(ABSENT, |o| o.index());
    payload(|w| {
        w.write_u64(process_id.unwrap_or(u64::from(ABSENT)));
        w.write_u32(index(copies, 0)).write_u32(index(copies, 1));
        w.write_u32(index(moves, 0)).write_u32(index(moves, 1));
        w.write_u32(toy::INSPECT).write_u32(0);
    })
}

/// Every mix of zero to two copied handles, zero to two moved handles, and a process id sent or
/// not. Each handle names a different object, so a framework that read the two lists in the
/// wrong order, or one entry for another, would report the wrong objects.
#[test]
fn each_handle_list_and_the_process_id_arrive_as_sent() {
    for copy_count in 0..=2 {
        for move_count in 0..=2 {
            for send_pid in [false, true] {
                let mut rig = Rig::strict();
                rig.kernel.set_process_id(PROCESS_ID);
                let mut build = request(toy::INSPECT);
                if send_pid {
                    // The guest's own word is ignored: the kernel's id is what arrives.
                    build = build.send_pid(u64::from(ABSENT - 1));
                }
                let mut copies = Vec::new();
                for _ in 0..copy_count {
                    let (handle, object) = handle_to_new_object(&mut rig);
                    build = build.copy_handle(handle);
                    copies.push(object);
                }
                let mut moves = Vec::new();
                for _ in 0..move_count {
                    let (handle, object) = handle_to_new_object(&mut rig);
                    build = build.move_handle(handle);
                    moves.push(object);
                }
                let reply = plain(&rig.send(&build));
                let case = (copy_count, move_count, send_pid);
                assert_eq!(reply.result, ResultCode::SUCCESS.raw(), "{case:?}");
                let expected = inspected(send_pid.then_some(PROCESS_ID), &copies, &moves);
                assert_eq!(reply.data, expected, "{case:?}");
            }
        }
    }
}

#[test]
fn a_handle_that_names_nothing_fails_only_the_method_that_asks_for_it() {
    let mut rig = Rig::strict();
    let (good, object) = handle_to_new_object(&mut rig);
    // `inspect` looks at every handle but does not insist: the bad one reads as absent and the
    // good one after it is still in its place.
    let build = request(toy::INSPECT)
        .copy_handle(Handle::INVALID.raw())
        .copy_handle(good);
    let reply = plain(&rig.send(&build));
    assert_eq!(reply.result, ResultCode::SUCCESS.raw());
    let index = |at: usize| u32::from_le_bytes(reply.data[at..at + 4].try_into().unwrap());
    assert_eq!((index(8), index(12)), (ABSENT, object.index()));

    // `signal` insists on copy handle 0.
    let bad = request(toy::SIGNAL).copy_handle(Handle::INVALID.raw());
    assert_eq!(
        plain(&rig.send(&bad)),
        only(KernelError::InvalidHandle.code())
    );
}

#[test]
fn a_method_uses_the_object_behind_a_copied_handle() {
    let mut rig = Rig::strict();
    let made = plain(&rig.send(&request(toy::MAKE_EVENT)));
    let event = rig.object_of(made.copies[0]);
    assert_eq!(rig.kernel.signals(event), Some(1));

    let signal = request(toy::SIGNAL).copy_handle(made.copies[0]);
    assert_eq!(plain(&rig.send(&signal)), only(ResultCode::SUCCESS));
    assert_eq!(rig.kernel.signals(event), Some(2));

    // No handle at that index at all is the request's fault, not the kernel's.
    assert_eq!(
        plain(&rig.send(&request(toy::SIGNAL))),
        only(INVALID_CMIF_REQUEST)
    );
    // The handle list asked is the copy list: a moved handle does not stand in.
    let moved = request(toy::SIGNAL).move_handle(made.copies[0]);
    assert_eq!(plain(&rig.send(&moved)), only(INVALID_CMIF_REQUEST));
    assert_eq!(rig.kernel.signals(event), Some(2));
}

#[test]
fn a_full_handle_table_is_the_kernels_error() {
    // Room for exactly one handle.
    let mut rig = Rig::with_handle_table(UnknownPolicy::Strict, 1);
    let first = plain(&rig.send(&request(toy::MAKE_EVENT)));
    assert_eq!(first.result, ResultCode::SUCCESS.raw());
    let second = plain(&rig.send(&request(toy::MAKE_EVENT)));
    assert_eq!(second, only(KernelError::OutOfHandles.code()));
    assert_eq!(rig.kernel.handle_count(), 1);
}

#[test]
fn a_new_session_that_gets_no_handle_is_not_opened() {
    let mut rig = Rig::with_handle_table(UnknownPolicy::Strict, 1);
    let open = request(toy::OPEN_CHILD).data(&7_u32.to_le_bytes());
    let first = plain(&rig.send(&open));
    assert_eq!(first.result, ResultCode::SUCCESS.raw());
    let second = plain(&rig.send(&open));
    assert_eq!(second, only(KernelError::OutOfHandles.code()));
    // One child session, one child: the second child was dropped.
    assert_eq!(rig.kernel.sessions().len(), 1);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 2)
    );

    let over_tipc = RequestBuilder::tipc(toy::TIPC_OPEN_CHILD).data(&7_u32.to_le_bytes());
    assert_eq!(
        plain(&rig.send(&over_tipc)),
        only(KernelError::OutOfHandles.code())
    );
}

/// The documented limit of [`Server`](super::super::Server)'s reply building: handles added
/// before the table filled up stay, as in Eden.
#[test]
fn handles_added_before_the_table_filled_stay_in_it() {
    // `interleave` adds a copy, a move, a session and a move; the table holds two.
    let mut rig = Rig::with_handle_table(UnknownPolicy::Strict, 2);
    let reply = plain(&rig.send(&request(toy::INTERLEAVE)));
    assert_eq!(reply, only(KernelError::OutOfHandles.code()));
    assert_eq!(rig.kernel.handle_count(), 2);
    // The session was not opened and its child is gone.
    assert_eq!(rig.kernel.sessions(), []);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (1, 1)
    );
}

/// The same limit one handle later: a session opened before the table filled up stays open.
#[test]
fn a_session_opened_before_the_table_filled_stays_open() {
    // Room for the copy, the first move and the session; the second move does not fit.
    let mut rig = Rig::with_handle_table(UnknownPolicy::Strict, 3);
    let reply = plain(&rig.send(&request(toy::INTERLEAVE)));
    assert_eq!(reply, only(KernelError::OutOfHandles.code()));
    assert_eq!(rig.kernel.handle_count(), 3);
    assert_eq!(rig.kernel.sessions().len(), 1);
    assert_eq!(
        (rig.server.session_count(), rig.server.interface_count()),
        (2, 2)
    );
    // It is a working session, though the caller was never told its handle.
    let child = rig.kernel.sessions()[0];
    assert_eq!(rig.child_tag(child), super::INTERLEAVED_TAG);
}
