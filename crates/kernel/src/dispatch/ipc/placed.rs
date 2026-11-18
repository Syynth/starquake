//! The handles a server's handler placed in its request's sender's handle table, and taking
//! back the ones the sender does not receive (issue #209).
//!
//! In Horizon a reply's handles reach the client's table only as the kernel translates the
//! reply (`ProcessMessageSpecialData<true>` on the reply path). An HLE handler has no table of
//! its own, so it places each handle in the sender's table while it runs
//! ([`IpcContext::insert_handle`](super::IpcContext::insert_handle),
//! [`IpcContext::create_session`](super::IpcContext::create_session),
//! [`IpcContext::connect`](super::IpcContext::connect)) and writes the handle values into its
//! reply. [`Placed`] records each one, with the object it names, in the order it was placed.
//!
//! When the request is settled, the sender keeps exactly the placed handles its delivered
//! reply names in its special header ([`crate::ipc::message::reply_handles`]); every other one
//! is taken back. A reply that is not delivered names none, so all of them are taken back.
//! That covers these cases, each pinned by a test in `dispatch/ipc/tests/rollback.rs`:
//!
//! - the handler gave up part-way (a later handle it could not place, a full table, a write
//!   to the sender's memory that faulted) and wrote a reply that names none of them, as the
//!   service framework's reply builder does with its error reply;
//! - the handler failed ([`HandlerError`](crate::ipc::HandlerError)) or closed the session
//!   ([`Handled::Closed`](crate::ipc::Handled::Closed)), so the sender gets
//!   [`KernelError::SessionClosed`](crate::result::KernelError::SessionClosed) and no reply;
//! - the reply could not be written into the sender's message buffer: the sender's result is
//!   then [`KernelError::InvalidCurrentMemory`](crate::result::KernelError::InvalidCurrentMemory)
//!   instead of success;
//! - the sender ended while its request was in flight, so nothing is written for it.
//!
//! Handles placed during a [`SessionHandler::handle`](crate::ipc::SessionHandler::handle) call
//! that defers ([`Handled::Deferred`](crate::ipc::Handled::Deferred)) are taken back when that
//! call returns: a deferred reply's handles are placed when its reply is written
//! ([`SessionHandler::take_reply`](crate::ipc::SessionHandler::take_reply)), with that call's
//! context, as the service framework places them. Test:
//! `rollback::handles_placed_by_a_handler_that_defers_are_taken_back_when_it_returns`.
//!
//! # The order, from Eden's reply path
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_server_session.cpp` @ cf654052
//! (GPL-3.0-or-later). In `ProcessSendMessage` (the reply direction of `SendMessage`), a
//! failure once the special data has been processed runs `CleanupSpecialData` from the
//! `ON_RESULT_FAILURE` block: it walks the copy and then the move handle words of the message
//! written into the client, in order, and `Remove`s each one that is not `Svc::InvalidHandle`
//! from the client process's table, which closes the handle's reference to its object. That
//! happens inside `SendMessage`, before `SendReply` hands the client its result
//! (`client_thread->EndWait(kernel, client_result)`, the failure's own result) and before the
//! request's reference goes (`SCOPE_EXIT { request->Close(kernel); }`). Here, likewise, the
//! handles are taken back in the order they were placed (the framework places copies and then
//! moves, `eden/src/core/hle/service/hle_ipc.cpp`, `WriteToOutgoingCommandBuffer`), before the
//! sender is woken with its result, its request's references are released and its events are
//! recorded (`Dispatcher::settle`). Those steps happen under one hold of the kernel lock, so no
//! other thread sees between them: their order follows Eden's, and only the order in which the
//! handles are taken back is observable (a taken-back session's server is told in that order).
//! Test: `rollback::taken_back_sessions_are_torn_down_and_their_servers_told_in_the_order_placed`.
//!
//! A taken-back handle is removed as `CloseHandle` removes one
//! ([`crate::svc::handle::remove_handle`]): when it held its object's last reference, the object
//! is torn down, so a session opened for the reply is forgotten and its server told.
//!
//! # Divergences from Eden
//!
//! - **Eden's HLE path takes nothing back.** Its `WriteToOutgoingCommandBuffer` returns part-way
//!   on a failed `handle_table.Add` (`R_TRY`) and leaves the handles it added. Starquake follows
//!   Eden's kernel path above instead, which is what a console does. Tests:
//!   `rollback::a_full_sender_table_takes_back_what_was_placed` and the three
//!   `rollback::a_failure_at_the_*_handle_*` tests.
//! - **Moved handles not yet placed stay with the server.** After a failure, Eden's
//!   `ProcessMessageSpecialData` still `Remove`s every later moved handle from the server's
//!   table, closing it. An HLE server has no table: what it moves is an object it keeps its own
//!   reference to (the creator's, as for [`IpcContext::create_event`](super::IpcContext::create_event)),
//!   so an object it did not place yet stays its own, as if handed back. Test:
//!   `rollback::a_moved_object_not_yet_placed_stays_with_the_server`.
//! - **A reply that cannot be written fails.** Eden copies a reply between two message buffers
//!   in thread-local memory with a `memcpy` that cannot fail. Here the guest memory write can;
//!   the sender's result is then [`KernelError::InvalidCurrentMemory`](crate::result::KernelError::InvalidCurrentMemory),
//!   the result Eden's `KPageTableBase::CopyMemoryFromKernelToLinear`
//!   (`eden/src/core/hle/kernel/k_page_table_base.cpp` @ cf654052) gives for a destination it
//!   cannot write (`R_UNLESS(this->Contains(dst_addr, size), ResultInvalidCurrentMemory)`), the
//!   copy `ProcessSendMessage` makes of a reply's raw data into a user buffer. Test:
//!   `rollback::a_reply_that_cannot_be_written_takes_back_every_handle_and_fails`.
//! - **A handle that no longer names what was placed is left alone.** The guest's other threads
//!   run while the handler does, and may close a placed handle, or move it in a request of their
//!   own, before it is taken back; a process that ended has no handles left. A placed handle is
//!   taken back only while it still names the object placed. Eden needs no such check: the
//!   client's table cannot change between the translation and its cleanup. Test:
//!   `rollback::a_placed_handle_the_guest_closed_meanwhile_is_left_alone`.

use crate::handle::{Handle, HandleTable};
use crate::id::ObjectId;
use crate::svc::handle::remove_handle;
use crate::tables::Scheduler;

/// The handles a handler placed in its request's sender's table, each with the object it
/// named, in the order they were placed. See the module docs.
#[derive(Debug, Default)]
pub(super) struct Placed(Vec<(Handle, ObjectId)>);

impl Placed {
    /// Records that `handle`, naming `object`, was placed in the sender's table.
    pub(super) fn push(&mut self, handle: Handle, object: ObjectId) {
        self.0.push((handle, object));
    }

    /// Takes back every placed handle that `kept` does not name, in the order they were
    /// placed, from `handles`, the sender's table: each is removed as `CloseHandle` removes it
    /// ([`remove_handle`]), when it still names the object placed. A removal the table refuses
    /// is the kernel's own bookkeeping, and is logged; the rest are still taken back.
    pub(super) fn take_back(
        self,
        handles: &mut HandleTable,
        scheduler: &mut Scheduler,
        kept: &[Handle],
    ) {
        for (handle, object) in self.0 {
            if kept.contains(&handle) {
                continue;
            }
            if handles.get(handle).ok() != Some(object) {
                tracing::debug!(
                    ?handle,
                    ?object,
                    "a placed handle no longer names what was placed; left alone"
                );
                continue;
            }
            if let Err(error) = remove_handle(handles, scheduler, handle) {
                tracing::error!(%error, ?handle, ?object, "a placed handle could not be taken back");
            }
        }
    }
}
