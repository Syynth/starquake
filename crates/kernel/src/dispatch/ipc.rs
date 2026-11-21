//! The dispatcher's IPC arms, `ConnectToNamedPort` (0x1f) and `SendSyncRequest` (0x21), the
//! handlers it keeps outside the tables, and [`IpcContext`], what a handler reaches the kernel
//! through.
//!
//! Kernel spec, "IPC". `SendSyncRequest` goes in steps, each of which takes the kernel lock
//! for itself and none of which holds it while a handler runs:
//!
//! 1. Under the lock: read the clock, look the session up, copy the message out of the caller's
//!    TLS block, translate its special header ([`crate::ipc::message`]) and queue the request
//!    ([`Ipc::send`]). A request that cannot start leaves its sender waiting
//!    ([`Outcome::Waiting`]). A failed translation can destroy a session whose last handle the
//!    message moved; that session is torn down as `CloseHandle` tears it down
//!    ([`crate::ipc::Ipc::forget`]).
//! 2. Take the server's handler out of its slot (under the lock), run it **without the lock**
//!    (the calls it makes through [`IpcContext`] take the lock one at a time), and end the
//!    request ([`Ipc::finish`]) and put the handler back under the lock.
//! 3. Start the server's next ready request, if one may start, on this core, and run it as in
//!    step 2; repeat until none may.
//! 4. Settle every request that is over, in the order they ended: its reply is written into its
//!    sender's TLS block, the handles its handler placed in the sender's table that the reply
//!    does not name are taken back (all of them when no reply is written; `dispatch/ipc/placed.rs`),
//!    and under the lock its references are released and its events recorded, the IPC events
//!    its handler made and then its `SendSyncRequest` event. The caller's own result goes into
//!    its registers; every other sender that was waiting is woken ([`Dispatcher::take_woken`]).
//!
//! The handler leaves its slot at two other times. Its notices (accept a session, forget a
//! closed one) are given to it under the lock: [`SessionHandler::accept`] and
//! [`SessionHandler::closed`] take no [`IpcContext`], so they cannot call back into the kernel.
//! [`Dispatcher::deliver_replies`] takes it out without the lock and without a worker, because
//! `take_reply` gets an [`IpcContext`]; a request the queue starts meanwhile (it has a free
//! worker) finds the slot empty. It stays in flight, holding its worker, and is parked: its
//! sender waits ([`Outcome::Waiting`]) and the core that puts the handler back runs it and
//! reports the sender woken ([`Dispatcher::take_woken`]). The other way round, a delivery that
//! finds the handler out of its slot (a core runs a request through it) is owed
//! (`State::owed`): the thread that puts the handler back makes it before its own call
//! returns, so replies said to be ready are delivered once the handler is free. Test:
//! `replies_said_to_be_ready_while_another_core_runs_the_handler_are_delivered`. Whenever a
//! handler goes back in its slot it is first given the notices that arrived while it was out.
//!
//! `ConnectToNamedPort` is ported from Eden `eden/src/core/hle/kernel/svc/svc_port.cpp` @
//! cf654052 (GPL-3.0-or-later), `ConnectToNamedPort`, with the name read as
//! [`PortName::read`] says: a port nothing is registered under is [`KernelError::NotFound`],
//! and a full handle table [`KernelError::OutOfHandles`], in that order. `SendSyncRequest` is
//! `SendSyncRequestImpl` in `eden/src/core/hle/kernel/svc/svc_ipc.cpp` @ cf654052: a handle that
//! names no client session is [`KernelError::InvalidHandle`].
//!
//! Where starquake differs from Eden, pinned by the tests named (in `dispatch/ipc/tests.rs`):
//!
//! - A handler's own failure ([`crate::ipc::HandlerError`]) closes the server end of the
//!   session, and its senders see [`KernelError::SessionClosed`], as a console's server
//!   framework closes a session it fails to process (see [`crate::ipc::HandlerError`]). Eden
//!   sends whatever reply the request's context holds, asserts on the failure (an assertion
//!   logs and carries on, `eden/src/common/assert.h`) and keeps the session
//!   (`eden/src/core/hle/service/server_manager.cpp`, `ServerManager::CompleteSyncRequest`:
//!   `SendReplyHLE`, then `R_ASSERT(service_res)`). Test:
//!   `a_handler_that_fails_closes_the_session`.
//! - A server's handler can reply later ([`Handled::Deferred`]); Eden's HLE servers reply on
//!   their own host threads instead. Test: `a_second_sender_waits_behind_a_deferred_request`.
//! - Sessions have no limit, so `ConnectToNamedPort` never fails for want of one. Eden's
//!   `KClientPort::CreateSession` (`eden/src/core/hle/kernel/k_client_port.cpp` @ cf654052),
//!   which runs after the handle is reserved, refuses with `ResultLimitReached` when the
//!   process's `LimitableResource::SessionCountMax` reservation fails and with
//!   `ResultOutOfSessions` when the port already has `m_max_sessions` sessions; neither error
//!   can happen here. PROVISIONAL: the process's resource limit counts no sessions yet
//!   ([`crate::resource_limit`], "Divergences from Eden"), and ports have no maximum. Test:
//!   `connecting_many_times_never_runs_out_of_sessions`.
//!
//! # Senders wait as scheduler threads (kernel slice 10)
//!
//! When the calling core runs a scheduler thread ([`Sender::waiter`]), its registers are saved
//! as its context when the request is queued or started (step 1), and it waits for its reply
//! ([`crate::sync::Wait::Reply`]) when the request waits behind another (step 1), is deferred
//! (step 2), or is parked because its handler is out of its slot (step 2). The wait begins
//! under the lock that queues, defers or parks the request, before it goes, so that a reply
//! delivered at once finds the sender waiting. This follows Eden, where every synchronous
//! request begins its sender's wait (`KServerSession::OnRequest`,
//! `eden/src/core/hle/kernel/k_server_session.cpp` @ cf654052, `BeginWait` after the request
//! is listed), and its reply ends it. Settling a request writes the reply into the sender's
//! message buffer and then, under the same lock, writes the result into its saved `x0` and makes
//! it runnable ([`Scheduler::end_reply_wait`]); the core that schedules it next loads that
//! context. A sender is still reported woken too ([`Dispatcher::take_woken`]).
//!
//! A dispatch that ends in [`Outcome::Waiting`] writes no register: the core reloads the
//! context saved in step 1, so a sender that is not in its reply wait would read its own
//! session handle back as the result. `SendSyncRequest` ends in [`Outcome::Waiting`] in exactly
//! two places, and the sender waits on each path through them except those listed after:
//!
//! - **The request is queued** behind another (`begin`, [`Admission::Queued`]): the sender
//!   waits, and one that cannot is an internal fault, its request withdrawn first. Test:
//!   `senders_that_wait_are_blocked_threads_and_the_reply_makes_them_runnable` (B).
//! - **The caller's own request is not among those this dispatch settled**, which is so when
//!   it was deferred or parked (both waits begin in `run_request`). Tests:
//!   `senders_that_wait_are_blocked_threads_and_the_reply_makes_them_runnable` (A, deferred),
//!   `a_request_parked_while_a_delivery_has_the_handler_leaves_its_sender_waiting_for_the_reply`
//!   and `a_parked_request_its_handler_then_defers_keeps_its_sender_waiting_until_the_reply`.
//!
//! The exceptions, where the dispatch ends in [`Outcome::Waiting`] with no reply wait: a
//! sender with no scheduler thread (only a caller that drives the dispatcher without a core
//! loop; it is only reported woken), a sender that ended while its handler ran (below), and
//! two failures of the kernel's own bookkeeping, each logged as an error: `wait_for_reply`
//! refusing on the deferred or parked path ("a sender could not wait for its reply"), and
//! [`crate::ipc::Ipc::finish`] failing for the caller's request ("a request could not be
//! ended"), which leaves nothing to settle. A sender that did wait may be out of its wait by
//! the time its dispatch returns, when another core or the timer thread settled its request
//! meanwhile; its result is then in the saved context already, which is what the core loads.
//!
//! A sender that ends while it waits leaves its session's queue if its request has not
//! started (`Scheduler::forget_waits`); a request already in flight stays with its handler,
//! and when it ends nothing is written to the sender's message buffer and no event is recorded
//! for its call, though the IPC events its handler made are. This diverges from Eden, which
//! keeps a terminating thread's request in the session's list, so the server still receives
//! it, and in `KServerSession::SendReply` (`eden/src/core/hle/kernel/k_server_session.cpp` @
//! cf654052) writes the reply but skips `EndWait` for a client whose termination was requested.
//! Here a thread ends at once (`sched/scheduler/lifecycle.rs`, "Termination is immediate") and
//! its thread-local page goes back to its process, so there is nowhere safe to write a reply.
//! Tests: `a_thread_that_ends_while_its_request_waits_leaves_the_queue` and
//! `a_thread_that_ends_while_its_request_is_in_flight_gets_no_reply_and_no_event`.
//!
//! # A handler ends the sender's process
//!
//! The services spec ("Writing a service", approved 2026-11-16) lets a service end the calling
//! process (`ISelfController::Exit`): "the request is recorded, the reply is sent, and then the
//! kernel tears the process down through its normal termination path". A handler asks with
//! [`IpcContext::end_process`], and the request that is over carries the process to end: its
//! sender's ([`Sender::process`]), whichever thread settles it. Whoever settles the request (a
//! `SendSyncRequest` dispatch, step 4, or a [`Dispatcher::deliver_replies`]) writes its reply
//! into the sender's message buffer and its result into the sender's registers, records its
//! events, settles every other request that is over with it, and only then ends the process
//! with [`terminate_process`], the function `ExitProcess` ends it with: every thread ends and
//! every handle closes, so the servers of its sessions are told. The replies and the end are
//! made under one hold of the kernel lock (`Dispatcher::settle_and_end`), so no core can run a
//! sender of the ending process between its reply and the end. Each process asked for ends
//! once, in the order first asked; a second ask for the same process in one settling is not
//! tried again (log-only: trying it would only log that the process has ended already). A run
//! whose process has ended is never stalled ([`Dispatcher::is_stalled`]): a core that parks
//! after a delivery ended the process, and before the timer thread stops the run, must not
//! call it a deadlock.
//!
//! - **A `SendSyncRequest` dispatch** ends in [`Outcome::ProcessExited`] when its caller's own
//!   process is one it ended, which its caller treats as it treats `ExitProcess`'s; otherwise
//!   it ends with its own result. Tests:
//!   `a_handler_that_ends_the_process_replies_first_and_then_releases_everything`, and
//!   `a_dispatch_that_ends_another_process_returns_its_own_result` for a request of another
//!   process that the dispatch ran.
//! - **A delivery** (the timer thread's, `deliver_replies`) returns the processes it ended
//!   ([`Ended`]), with the sender of the request that asked, for the run to stop and record
//!   the end, as a core does for [`Outcome::ProcessExited`]. That covers a deferred reply
//!   whose `take_reply` asked, and a request the delivery started once the deferred one freed
//!   the server. Tests: `an_end_asked_while_a_delivery_takes_a_deferred_reply_is_honoured_after_it`,
//!   `an_end_asked_by_a_request_a_delivery_runs_is_honoured_after_the_replies`,
//!   `a_delivery_ends_the_process_of_the_request_that_asked_and_no_other` (which also checks
//!   that the sender of the process that goes on is reported woken), and
//!   `a_deferred_reply_that_asks_to_end_ends_its_senders_process_and_no_other` (a `take_reply`
//!   that asks for a sender of another process than the delivering core's). A `take_reply` that
//!   asks and then fails closes the session, and the end is that of the request it answered,
//!   the first that is over: `a_deferred_reply_that_fails_while_asking_to_end_ends_the_process_of_the_request_it_answered`.
//!
//! The sender's process is that of the scheduler's thread on its core, or, for a core that
//! runs no thread of a process (the tests that drive the dispatcher without a core loop), the
//! process these tables belong to ([`crate::tables::Processes::own`]). PROVISIONAL (one process
//! per set of tables): the guest-visible process id and the `CURRENT_PROCESS` pseudo-handle a
//! request is given are still those of the tables' process, and the process an end names
//! shares the one handle table, which [`terminate_process`] closes whichever process it ends
//! (as `ExitThread` ending the last thread of any process closes it). A core loop runs only the
//! tables' process's threads, so in a run the two always agree. A dispatch whose caller's
//! process did not end does not report another process ending: nothing in a run makes one.
//!
//! Neither this nor [`IpcContext::clear_event`] records an event of its own: what is recorded
//! is the request that asked (its handler's IPC event, then its `SendSyncRequest` event), and
//! the end itself is the run's to record, as for `ExitProcess`.
//!
//! Two asks are not honoured; each is logged as an error:
//!
//! - **A handler that deferred its reply** ([`Handled::Deferred`]): the end follows the reply,
//!   and there is none yet, so the process goes on. The service framework refuses the
//!   combination before it reaches the kernel. Test:
//!   `an_end_asked_by_a_handler_that_defers_is_not_honoured`.
//! - **An end for a process that has ended already** (an `ExitProcess` while the request was
//!   in flight): there is nothing left to end, and no end is reported. Tests:
//!   `a_delivery_does_not_report_an_end_asked_for_a_process_that_ended_meanwhile` and
//!   `ending_a_process_that_has_ended_already_fails_and_reports_nothing`.
//!
//! A delivery that is owed because a guest's `SendSyncRequest` dispatch has the handler out of
//! its slot is that dispatch's to make (`deliver_owed`, when it puts the handler back), and an
//! end asked by a request it then runs is honoured there, after settling that request with
//! the others. Test:
//! `an_end_asked_by_a_request_the_dispatch_putting_the_handler_back_runs_is_honoured`. An end
//! asked by a handler that then fails, or that closes the session ([`Handled::Closed`]), is
//! honoured too, once the request has ended with [`KernelError::SessionClosed`]. Tests:
//! `an_end_asked_by_a_handler_that_then_fails_is_honoured` and
//! `an_end_asked_by_a_handler_that_closes_the_session_is_honoured`.
//!
//! Eden's `ISelfController::Exit` calls `KProcess::Terminate` inside the handler, before the
//! reply is sent (`eden/src/core/hle/service/am/service/self_controller.cpp` @ cf654052); the
//! order here is the services spec's.
//!
//! PROVISIONAL (walking skeleton, #74): the dispatcher is told its caller (`Caller`), and
//! there is one process. A caller the scheduler runs no thread for (the tests that drive the
//! dispatcher without a core loop) is only reported woken, as before slice 10.

use std::collections::VecDeque;
use std::sync::Arc;

use starquake_cpu::GuestRegs;
use starquake_ipc::BUFFER_SIZE;
use starquake_memory::GuestMemory;
use starquake_trace::{
    EnvError, EventKind, GuestTicks, Recorder, SVC_ARG_COUNT, ServiceId, ThreadId,
};

use super::hooks;
use super::sync::drain_wakes;
use super::{
    Call, Caller, Dispatcher, Locked, Outcome, State, Tables, complete, fault, mismatch,
    record_point, recorded_args, result_word,
};
use crate::handle::{Handle, HandleError, HandleTable};
use crate::id::{ObjectId, ProcessId, ServerId};
use crate::ipc::message::{
    Current, MessageBuffer, Translated, Translation, reply_handles, translate,
};
use crate::ipc::{
    Admission, Done, EventHook, Finish, Handled, IpcError, Notice, PortName, Request, Sender,
    SessionHandler,
};
use crate::memory::{
    AddressSpace, DeviceShareToken, SharedMemory, SharedPages, SvcPermission, TransferMemory,
    is_shared_memory_permission, shared_memory_size,
};
use crate::object::Object;
use crate::result::{CallResult, KernelError, ResultCode, SvcError, SvcResult};
use crate::svc::process::terminate_process;
use crate::svc::sync::read_user_memory;
use crate::svc::{Convention, SvcId};
use crate::sync::WaitOrigin;
use crate::tables::Scheduler;

mod placed;
#[cfg(test)]
mod tests;

use placed::Placed;

/// A guest thread whose blocking system call ended while it waited, and its result: a
/// `SendSyncRequest`, whose reply, when the result is success, is already in its message
/// buffer, or a `WaitSynchronization`, whose result and index are already in its thread's saved
/// registers (`dispatch/sync.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Woken {
    /// The thread, as the event stream names it.
    pub thread: ThreadId,
    /// The core it called from.
    pub core: u8,
    /// What its call returns in `x0`.
    pub result: ResultCode,
}

/// A process a handler's request ended during [`Dispatcher::deliver_replies`] (module docs, "A
/// handler ends the sender's process"): the process, and the sender of the request that asked,
/// which the run records the end for, as a core records it for the thread whose
/// `SendSyncRequest` ended in [`Outcome::ProcessExited`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ended {
    /// The process that ended: the sender's ([`Sender::process`]).
    pub process: ProcessId,
    /// The sending thread, as the event stream names it.
    pub thread: ThreadId,
    /// The core it called from.
    pub core: u8,
}

/// A request that is over, with the events its handler recorded, waiting to be settled.
struct Finished {
    session: ObjectId,
    done: Done,
    events: Vec<EventKind>,
    /// The sender's process ([`Sender::process`]), when its handler asked to end it
    /// ([`IpcContext::end_process`]): that process ends once the request is settled.
    end: Option<ProcessId>,
    /// The handles its handler placed in the sender's table: those its reply names are kept
    /// when the reply is written, and the rest taken back (`dispatch/ipc/placed.rs`).
    placed: Placed,
}

/// The kernel error a guest sees for a handle-table failure, logging the ones that are the
/// kernel's own bookkeeping.
fn handle_error(error: &HandleError) -> KernelError {
    error.kernel_error().unwrap_or_else(|| {
        tracing::error!(%error, "handle table bookkeeping failed during a request");
        KernelError::InvalidState
    })
}

/// The kernel error a handler sees for a failed table operation, logging the ones that are the
/// kernel's own bookkeeping.
fn call_error(error: SvcError) -> KernelError {
    match error {
        SvcError::Kernel(error) => error,
        SvcError::Internal(what) => {
            tracing::error!(what, "kernel bookkeeping failed during a request");
            KernelError::InvalidState
        }
    }
}

/// The kernel error a handler sees for an IPC table failure, logging the ones that are the
/// kernel's own bookkeeping.
fn ipc_error(error: IpcError) -> KernelError {
    match error {
        IpcError::Kernel(error) => error,
        error => {
            tracing::error!(%error, "IPC bookkeeping failed during a request");
            KernelError::InvalidState
        }
    }
}

/// What a server's handler reaches the kernel through while it handles one request: the
/// request's sender's process (its handle table and memory), the server it runs on, and the
/// event stream. Each call takes the kernel lock for itself; none is held between them.
pub struct IpcContext<'a> {
    dispatcher: &'a Dispatcher,
    memory: &'a dyn GuestMemory,
    server: ServerId,
    process_id: u64,
    translated: Translated,
    events: Vec<EventKind>,
    /// Whether the handler asked to end the sender's process ([`IpcContext::end_process`]).
    end: bool,
    /// The handles the handler placed in the sender's table, through this context.
    placed: Placed,
}

impl IpcContext<'_> {
    /// The sender's process id as the guest knows it.
    #[must_use]
    pub fn process_id(&self) -> u64 {
        self.process_id
    }

    /// The object a handle names: one of the request's own handles as it was when the request
    /// was sent (a moved one is no longer in the sender's table), or else whatever the handle
    /// names in the sender's table now.
    pub fn resolve_handle(&self, handle: Handle) -> SvcResult<ObjectId> {
        if let Some(object) = self.translated.object(handle) {
            return Ok(object);
        }
        self.dispatcher
            .with::<(&HandleTable,), _>(|(handles,)| handles.get(handle))
            .map_err(|error| handle_error(&error))
    }

    /// Adds a handle to `object` to the sender's table, for the handler's reply to name.
    ///
    /// The sender keeps it only if the reply that is written into its message buffer names it
    /// in its special header; otherwise it is taken back when the request is settled
    /// (`dispatch/ipc/placed.rs`). The same holds for the handles
    /// [`IpcContext::create_session`] and [`IpcContext::connect`] add.
    pub fn insert_handle(&mut self, object: ObjectId) -> SvcResult<Handle> {
        let handle = self
            .dispatcher
            .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
                handles.add(scheduler.objects_mut(), object)
            })
            .map_err(|error| handle_error(&error))?;
        self.placed.push(handle, object);
        Ok(handle)
    }

    /// Creates an event for the server and returns its readable end
    /// ([`Scheduler::create_readable_event`]), which the server gives the sender a copy handle
    /// to and signals with [`IpcContext::signal_event`]. The server keeps the creator's
    /// reference for as long as it lives; a service's event has no writable end (see the module
    /// docs of [`crate::sync`]). [`KernelError::OutOfResource`] when the object table is full.
    pub fn create_event(&mut self) -> SvcResult<ObjectId> {
        let server = self.server;
        self.dispatcher
            .with::<(&mut Scheduler,), _>(|(scheduler,)| {
                let readable = scheduler.create_readable_event()?;
                note_server_events(scheduler, server, &[readable])?;
                Ok(readable)
            })
            .map_err(call_error)
    }

    /// Clears the event `event` is an end of ([`Scheduler::clear_object`]): a signalled event
    /// stops being signalled, and one that is not stays as it is. Only an event the server
    /// made through this interface ([`IpcContext::create_event`],
    /// [`IpcContext::create_hooked_event`]) may be cleared; any other id, an end of another
    /// server's event or of a guest's included, is [`KernelError::InvalidHandle`], the answer
    /// [`IpcContext::signal_event`] gives an id that names no event.
    pub fn clear_event(&mut self, event: ObjectId) -> SvcResult<()> {
        let server = self.server;
        self.dispatcher
            .with::<(&mut Scheduler,), _>(|(scheduler,)| {
                if !scheduler.ipc().is_server_event(server, event) {
                    return Err(KernelError::InvalidHandle.into());
                }
                scheduler.clear_object(event)
            })
            .map_err(call_error)
    }

    /// Asks to end the sender's process ([`Sender::process`]) once this request is over
    /// (services spec, "Writing a service": a service "end\[s\] the calling process"): its reply
    /// is written into the sender's message buffer and its result into the sender's registers,
    /// its events are recorded, and then the process is ended through the path `ExitProcess`
    /// takes ([`crate::svc::process::terminate_process`]). The sender's own dispatch then ends
    /// in [`Outcome::ProcessExited`], and a delivery reports the end ([`Ended`]). Asking twice
    /// is asking once. The module docs say when the ask is not honoured.
    pub fn end_process(&mut self) {
        self.end = true;
    }

    /// Creates an event with both ends, for the server to give the sender its writable end
    /// (with a copy handle) and to hear, through `hook`, when the sender signals it
    /// ([`EventHook`]; `crate::ipc::hook` says exactly when the hook runs). Returns the
    /// writable end and the readable end, in that order ([`Scheduler::create_event`]). The
    /// server keeps the creator's reference to each for as long as it lives, as it does an
    /// event's from [`IpcContext::create_event`]. [`KernelError::OutOfResource`] when the object
    /// table is full; no event or hook is left behind then.
    pub fn create_hooked_event(
        &mut self,
        hook: Arc<dyn EventHook>,
    ) -> SvcResult<(ObjectId, ObjectId)> {
        let (dispatcher, server) = (self.dispatcher, self.server);
        dispatcher
            .with::<(&mut Scheduler,), _>(|(scheduler,)| {
                let (writable, readable) = scheduler.create_event()?;
                note_server_events(scheduler, server, &[writable, readable])?;
                dispatcher.hooks.attach(scheduler.objects(), writable, hook);
                Ok((writable, readable))
            })
            .map_err(call_error)
    }

    /// Whether every page of the sender's `size` bytes at `address` has the GPU-accessible
    /// attribute `svc 0x80` sets ([`AddressSpace::is_gpu_accessible`]): what the graphics
    /// server asks before it registers memory for the GPU (kernel spec, "Address space").
    #[must_use]
    pub fn is_gpu_accessible(&self, address: u64, size: u64) -> bool {
        self.dispatcher
            .with::<(&AddressSpace,), _>(|(space,)| space.is_gpu_accessible(address, size))
    }

    /// Takes a device-share lock on the sender's `size` bytes at `address`
    /// ([`AddressSpace::lock_device_share`], whose errors these are, and
    /// [`KernelError::TerminationRequested`], with nothing locked, once the process has ended):
    /// while it is held, the CPU
    /// mapping under the range cannot change. Returns the token that names the process and the
    /// exact range; each lock is released by one unlock of its token, on the timer thread
    /// ([`Dispatcher::unlock_device_share`]) or during a request
    /// ([`IpcContext::unlock_device_share`]). See `dispatch/device.rs`.
    pub fn lock_device_share(&mut self, address: u64, size: u64) -> SvcResult<DeviceShareToken> {
        self.dispatcher
            .lock_device_share(address, size)
            .map_err(call_error)
    }

    /// Releases the device-share lock `token` names, as [`Dispatcher::unlock_device_share`]
    /// does: what a server does when a request that took a lock fails after taking it. A
    /// refusal of the token itself is [`KernelError::TerminationRequested`] when its process has
    /// ended and [`KernelError::InvalidCurrentMemory`] otherwise (`DeviceShareError`'s
    /// mapping); the address space's own refusals are its results.
    pub fn unlock_device_share(&mut self, token: DeviceShareToken) -> SvcResult<()> {
        self.dispatcher
            .unlock_device_share(token)
            .map_err(|error| call_error(error.into_svc_error()))
    }

    /// Signals the event `event` is an end of ([`Scheduler::signal_object`]), waking the guest
    /// threads waiting on it. Their events are recorded, and they are reported woken, when the
    /// request ends (`dispatch/sync.rs`). [`KernelError::InvalidHandle`] for an id that names
    /// no live event.
    pub fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        self.dispatcher
            .with::<(&mut Scheduler,), _>(|(scheduler,)| scheduler.signal_object(event))
            .map_err(call_error)
    }

    /// Opens a session on the server running this request, and adds a handle to it to the
    /// sender's table. The server is not told: it records the session itself. All or nothing.
    pub fn create_session(&mut self) -> SvcResult<(ObjectId, Handle)> {
        let server = self.server;
        self.open_session(server, false)
    }

    /// Opens a session on the server known as `service`, and adds a handle to it to the
    /// sender's table; the server is told to accept it before it is given a request on it.
    /// [`KernelError::NotFound`] for a service no server is known as. All or nothing.
    pub fn connect(&mut self, service: ServiceId) -> SvcResult<(ObjectId, Handle)> {
        let server = self
            .dispatcher
            .with::<(&Scheduler,), _>(|(scheduler,)| scheduler.ipc().service(service));
        let server = server.ok_or(KernelError::NotFound)?;
        self.open_session(server, true)
    }

    fn open_session(&mut self, server: ServerId, notice: bool) -> SvcResult<(ObjectId, Handle)> {
        let (session, handle) = self
            .dispatcher
            .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
                open_session_with_handle(handles, scheduler, server, notice)
            })?;
        self.placed.push(handle, session);
        Ok((session, handle))
    }

    /// Copies the sender's memory at `address` into `out`, if the guest may read all of it;
    /// [`KernelError::InvalidCurrentMemory`] otherwise, with nothing copied.
    pub fn read_memory(&self, address: u64, out: &mut [u8]) -> SvcResult<()> {
        let memory = self.memory;
        self.dispatcher.with::<(&AddressSpace,), _>(|(space,)| {
            if read_user_memory(space, memory, address, out) {
                Ok(())
            } else {
                Err(KernelError::InvalidCurrentMemory)
            }
        })
    }

    /// Copies `data` into the sender's memory at `address`, if the guest may write all of it;
    /// [`KernelError::InvalidCurrentMemory`] otherwise, with nothing copied.
    pub fn write_memory(&mut self, address: u64, data: &[u8]) -> SvcResult<()> {
        let len = u64::try_from(data.len()).map_err(|_| KernelError::InvalidCurrentMemory)?;
        let memory = self.memory;
        self.dispatcher.with::<(&AddressSpace,), _>(|(space,)| {
            if !space.user_can_write(address, len) {
                return Err(KernelError::InvalidCurrentMemory);
            }
            memory
                .write(address, data)
                .map_err(|_| KernelError::InvalidCurrentMemory)
        })
    }

    /// Creates a shared memory for the server (`KSharedMemory::Create` and `Initialize`, as
    /// Eden's HLE services make theirs: no owner process, the owner permission `None`) of
    /// `size` bytes rounded up to whole pages, which processes map with `permission`. Returns
    /// the object, which the server gives the sender a copy handle to, and its pages, which the
    /// server reads and writes by offset ([`SharedPages`]). The server keeps the creator's
    /// reference for as long as it lives, as it does an event's.
    ///
    /// Refused, in this order, with nothing taken: [`KernelError::InvalidSize`] for a size of
    /// zero or one that rounds past the 64-bit space (a guess: see [`shared_memory_size`]);
    /// [`KernelError::InvalidNewMemoryPermission`] unless `permission` is read or read-write;
    /// [`KernelError::OutOfResource`] when the object table is full (`KSharedMemory::Create`
    /// comes before `Initialize`); then [`KernelError::OutOfMemory`] when the translation has
    /// no room for the pages (`KSharedMemory::Initialize`'s `ResultOutOfMemory`,
    /// [`AddressSpace::allocate_shared`]). The pages are taken last, after every check that can
    /// refuse, so a refused create takes no guest RAM. A translation with no backing to share,
    /// or that refuses for another reason, is the kernel's own failure, logged, and
    /// [`KernelError::InvalidState`] here.
    ///
    /// Eden's `Initialize` reserves the size from the system's resource limit before it
    /// allocates (`ResultLimitReached`); starquake has no system resource limit, and charges
    /// neither the process's pool nor its resource limit, which Eden does not charge either
    /// ([`crate::memory::shared`], "Not built").
    pub fn create_shared_memory(
        &mut self,
        size: u64,
        permission: SvcPermission,
    ) -> SvcResult<(ObjectId, SharedPages)> {
        let size = shared_memory_size(size).ok_or(KernelError::InvalidSize)?;
        if !is_shared_memory_permission(permission) {
            return Err(KernelError::InvalidNewMemoryPermission);
        }
        self.dispatcher
            .with::<(&mut AddressSpace, &mut Scheduler), _>(|(space, scheduler)| {
                create_shared_memory_in(space, scheduler, size, permission)
            })
            .map_err(call_error)
    }

    /// The transfer memory `object` names: one of the request's own handles resolved
    /// ([`IpcContext::resolve_handle`]), as a server is given a transfer memory.
    /// [`KernelError::InvalidHandle`] when it names no live transfer memory, or one whose pages
    /// are not locked (one being made, which no handle names yet).
    pub fn transfer_memory(&self, object: ObjectId) -> SvcResult<TransferMemory> {
        self.dispatcher.transfer_memory(object)
    }

    /// Copies `out.len()` bytes at `offset` into the transfer memory `object` names into
    /// `out`: what an HLE server does instead of mapping it (see
    /// [`crate::svc::transfer_memory`]). The bytes are read where the owner lent them, whatever
    /// permission the owner kept. [`KernelError::InvalidHandle`] as for
    /// [`IpcContext::transfer_memory`], and [`KernelError::InvalidCurrentMemory`] unless the
    /// bytes lie wholly inside the transfer memory; nothing is copied then.
    ///
    /// PROVISIONAL: the owner is taken to be the sender, whose memory this context reaches
    /// (one process per set of tables, `crate::tables`).
    pub fn read_transfer_memory(
        &self,
        object: ObjectId,
        offset: u64,
        out: &mut [u8],
    ) -> SvcResult<()> {
        self.dispatcher
            .read_transfer_memory(self.memory, object, offset, out)
    }

    /// Copies `data` into the transfer memory `object` names, at `offset`. The same rules as
    /// [`IpcContext::read_transfer_memory`].
    pub fn write_transfer_memory(
        &mut self,
        object: ObjectId,
        offset: u64,
        data: &[u8],
    ) -> SvcResult<()> {
        self.dispatcher
            .write_transfer_memory(self.memory, object, offset, data)
    }

    /// Keeps a reference of the server's own to the transfer memory `object` names, one of the
    /// request's own handles resolved, so it stays after the request and after the guest closes
    /// its handles, until the server drops it ([`IpcContext::release_kept_transfer_memory`],
    /// [`Dispatcher::release_kept_transfer_memory`]): Eden PR #4588's `KTransferMemory::Open` in
    /// `BindSyncpointMemory` (`dispatch/hooks.rs`, "A server's own reference"). Returns the
    /// transfer memory. [`KernelError::InvalidHandle`], with nothing kept, as for
    /// [`IpcContext::transfer_memory`]. Keeping it twice keeps two references, each dropped by
    /// one release.
    pub fn keep_transfer_memory(&mut self, object: ObjectId) -> SvcResult<TransferMemory> {
        let server = self.server;
        self.dispatcher
            .with::<(&mut Scheduler,), _>(|(scheduler,)| {
                hooks::keep_transfer_memory_in(scheduler, server, object)
            })
            .map_err(call_error)
    }

    /// Drops one reference the server keeps to the transfer memory `object` names
    /// ([`IpcContext::keep_transfer_memory`]), during a request: what a handler does when a
    /// request that kept one fails afterwards. [`KernelError::InvalidHandle`], with nothing
    /// changed, when the server keeps none; when it was the last reference the transfer memory
    /// goes, as [`Dispatcher::release_kept_transfer_memory`] says.
    pub fn release_kept_transfer_memory(&mut self, object: ObjectId) -> SvcResult<()> {
        let server = self.server;
        self.dispatcher
            .with::<(&mut Scheduler,), _>(|(scheduler,)| {
                hooks::release_kept_transfer_memory_in(scheduler, server, object)
            })
            .map_err(call_error)
    }

    /// Records an event for the request. It is recorded, stamped with the request's sender and
    /// the dispatch's time, when the request is settled, before the system call's own event.
    pub fn record(&mut self, kind: EventKind) {
        self.events.push(kind);
    }
}

/// [`IpcContext::create_shared_memory`] once its arguments are checked: room in the object
/// table, then the pages, then the object. `size` is whole pages and not zero, and
/// `permission` a shared memory's. When the object cannot be made after all (a table that was
/// not full refusing, which only broken bookkeeping does), the pages go back to the
/// translation at once ([`AddressSpace::release_shared`]); a failure to give them back is
/// logged, and the object's error returned either way (tests:
/// `an_object_not_made_after_its_pages_gives_them_back_at_once` and
/// `a_refused_give_back_of_an_object_not_made_returns_the_objects_error`). The log line is
/// log-only: no test captures it.
pub(super) fn create_shared_memory_in(
    space: &mut AddressSpace,
    scheduler: &mut Scheduler,
    size: u64,
    permission: SvcPermission,
) -> CallResult<(ObjectId, SharedPages)> {
    if scheduler.objects().is_full() {
        // `KSharedMemory::Create` returning null.
        return Err(KernelError::OutOfResource.into());
    }
    let pages = space.allocate_shared(size)?;
    let made = SharedMemory::new(pages.backing(), permission)
        .map_err(SvcError::from)
        .and_then(|memory| {
            Ok(scheduler
                .objects_mut()
                .create(Object::SharedMemory(memory))?)
        });
    match made {
        Ok(object) => Ok((object, pages)),
        Err(error) => {
            if let Err(release) = space.release_shared(pages.backing()) {
                tracing::error!(%release, "the pages of a shared memory not made stay used");
            }
            Err(error)
        }
    }
}

/// Notes that `server` made `events`, the ends of an event it just created
/// ([`crate::ipc::Ipc::add_server_event`]), so it may clear them. All or nothing: when the server
/// is not one the table knows (the kernel's own bookkeeping), the creator's references go again
/// and the error is returned, so no event is left that nobody holds.
fn note_server_events(
    scheduler: &mut Scheduler,
    server: ServerId,
    events: &[ObjectId],
) -> CallResult<()> {
    let ipc = scheduler.ipc_mut().1;
    let noted = events
        .iter()
        .try_for_each(|event| ipc.add_server_event(server, *event));
    if let Err(error) = noted {
        scheduler.release_objects(events)?;
        return Err(error.into());
    }
    Ok(())
}

/// Opens a session on `server` and gives the caller a handle to it: the session's object then
/// has one reference, the handle's. All or nothing: when the handle cannot be added, the
/// session is taken back.
fn open_session_with_handle(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
    server: ServerId,
    notice: bool,
) -> SvcResult<(ObjectId, Handle)> {
    let (objects, ipc) = scheduler.ipc_mut();
    let session = ipc
        .open_session(objects, server, notice)
        .map_err(ipc_error)?;
    match handles.add(objects, session) {
        Ok(handle) => {
            // The creator's reference; the handle's keeps the session.
            ipc.release(objects, session).map_err(ipc_error)?;
            Ok((session, handle))
        }
        Err(error) => {
            ipc.discard_session(objects, session).map_err(ipc_error)?;
            Err(handle_error(&error))
        }
    }
}

/// How the first step of `SendSyncRequest` ended.
enum Begun {
    /// The request is in flight: run its server's handler.
    Start { server: ServerId, session: ObjectId },
    /// The sender waits behind another request.
    Queued,
}

impl Dispatcher {
    /// Adds an HLE server known as `service` (the number another server's
    /// [`IpcContext::connect`] names it by), reachable by `ConnectToNamedPort` when it has a
    /// `port`, with `handler` behind it. `name` is for logs.
    pub fn add_server(
        &self,
        name: &str,
        service: ServiceId,
        port: Option<PortName>,
        handler: Box<dyn SessionHandler>,
    ) -> Result<ServerId, IpcError> {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        let (scheduler,) = locked.tables();
        let (_, ipc) = scheduler.ipc_mut();
        if let Some(port) = port
            && ipc.port(&port).is_some()
        {
            return Err(IpcError::PortTaken(port));
        }
        let server = ipc.add_server(name, service)?;
        if let Some(port) = port {
            ipc.name_port(port, server)?;
        }
        locked.state.handlers.insert(server, Some(handler));
        Ok(server)
    }

    /// Shows `inspect` the handler of every server, with the number the server is known by, in
    /// the order of the kernel's server ids: what the run's services leave behind (their counts,
    /// their records of what they stubbed) is read this way once the run is over. A handler out
    /// of its slot (a core is running a request through it, or delivering its replies) is not
    /// shown; the number of such handlers is returned.
    ///
    /// `inspect` runs with the kernel lock held: it must not call the dispatcher, and it is
    /// shown each handler only to read it.
    pub fn inspect_handlers(
        &self,
        mut inspect: impl FnMut(ServiceId, &dyn SessionHandler),
    ) -> usize {
        let locked = self.lock::<(&Scheduler,)>();
        let state = &*locked.state;
        let ipc = state.tables.scheduler.ipc();
        let mut out = 0;
        for (server, slot) in &state.handlers {
            let Ok(service) = ipc.server_service(*server) else {
                continue;
            };
            match slot {
                Some(handler) => inspect(service, &**handler),
                None => out += 1,
            }
        }
        out
    }

    /// The guest threads woken since the last call, oldest first: senders whose request ended
    /// while another thread's dispatch ran. Each one's result and reply are written; the caller
    /// resumes it.
    pub fn take_woken(&self) -> Vec<Woken> {
        let mut locked = self.lock::<()>();
        std::mem::take(&mut locked.state.woken)
    }

    /// Writes the deferred replies of `server` that are ready, wakes their senders, and starts
    /// the server's next requests. `caller` is the core doing it: its memory is the process's,
    /// and its clock reading stamps the events. The senders are reported by
    /// [`Dispatcher::take_woken`].
    ///
    /// Returns the processes ended because a handler asked to ([`IpcContext::end_process`]),
    /// each once, in the order first asked, after every request that is over is settled (module
    /// docs, "A handler ends the sender's process"). Each is its request's sender's process,
    /// whoever the `caller` is. Empty when no handler asked, and without a process that could
    /// not end (logged).
    ///
    /// The delivery is owed until a thread with the handler in hand makes it: this one, or,
    /// when the handler is out of its slot (a core is running a request through it), the one
    /// that puts it back, which delivers before its own call returns (`deliver_owed`). Either
    /// way the replies are delivered once the handler is free; this call does not wait for it.
    pub fn deliver_replies(
        &self,
        server: ServerId,
        caller: &mut Caller<'_>,
    ) -> Result<Vec<Ended>, EnvError> {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        let time = locked.guest_time(caller.env)?;
        let process_id = locked.state.tables.processes.process_id();
        locked.state.owed.insert(server);
        drop(locked);

        let finished = self.deliver_owed(server, caller.memory, process_id);
        // No sender is the delivering thread's own: every one is reported woken.
        let (_, ended) = self.settle_and_end(finished, None, caller.memory, caller.recorder, time);
        self.settle_notices();
        Ok(ended)
    }

    /// Makes the deliveries `server` owes ([`Dispatcher::deliver_replies`]) while one is owed
    /// and its handler is in its slot: takes the handler out, writes the replies it has ready,
    /// puts it back and runs the requests that may start then. Returns what is over, for the
    /// caller to settle. A handler out of its slot leaves the delivery owed: the thread that
    /// has it calls this after putting it back.
    fn deliver_owed(
        &self,
        server: ServerId,
        memory: &dyn GuestMemory,
        process_id: u64,
    ) -> Vec<Finished> {
        let mut finished = Vec::new();
        // Bounded by the deliveries owed: each turn takes one owed mark, and only
        // `deliver_replies` makes one.
        loop {
            let mut locked = self.lock::<(&mut Scheduler,)>();
            let state = &mut *locked.state;
            if !state.owed.contains(&server) {
                return finished;
            }
            let Some(slot) = state.handlers.get_mut(&server) else {
                tracing::error!(
                    ?server,
                    "deferred replies are owed by a server with no handler"
                );
                state.owed.remove(&server);
                return finished;
            };
            let Some(mut handler) = slot.take() else {
                // Whoever has the handler delivers when it puts it back.
                tracing::debug!(
                    ?server,
                    "deferred replies wait for their handler to come back"
                );
                return finished;
            };
            state.owed.remove(&server);
            // Bounded by the requests deferred now: each turn answers one of them.
            let deferred = state
                .tables
                .scheduler
                .ipc()
                .queue(server)
                .map_or(0, |queue| queue.deferred().count());
            drop(locked);

            finished.extend(self.take_replies(server, &mut *handler, memory, process_id, deferred));
            let parked = self.put_back(server, handler);
            finished.extend(self.run_ready(server, memory, process_id, parked));
        }
    }

    /// Asks `handler`, out of its slot, for at most `deferred` ready replies, and ends the
    /// requests they answer. Returns what is over.
    fn take_replies(
        &self,
        server: ServerId,
        handler: &mut dyn SessionHandler,
        memory: &dyn GuestMemory,
        process_id: u64,
        deferred: usize,
    ) -> Vec<Finished> {
        let mut finished = Vec::new();
        for _ in 0..deferred {
            let mut cx = self.context(memory, server, process_id, Translated::default());
            let mut message = [0; BUFFER_SIZE];
            let Some((session, written)) = handler.take_reply(&mut cx, &mut message) else {
                break;
            };
            let mut locked = self.lock::<(&mut HandleTable, &mut Scheduler)>();
            let (table, scheduler) = locked.tables();
            let (_, ipc) = scheduler.ipc_mut();
            let over = match written {
                Ok(()) => match ipc.complete(session, Box::new(message)) {
                    Ok(done) => {
                        let end = cx.end.then_some(done.request.sender.process);
                        vec![Finished {
                            session,
                            done,
                            events: cx.events,
                            end,
                            placed: Placed::default(),
                        }]
                    }
                    Err(error) => {
                        tracing::error!(%error, ?session, "a deferred reply has no request");
                        Vec::new()
                    }
                },
                Err(error) => {
                    tracing::error!(%error, ?server, ?session, "closing the session");
                    let closed = close_failed(ipc, server, session, cx.events);
                    ending(closed, cx.end)
                }
            };
            finished.extend(carry(over, cx.placed, table, scheduler));
        }
        finished
    }

    /// An [`IpcContext`] for a request on `server`.
    fn context<'a>(
        &'a self,
        memory: &'a dyn GuestMemory,
        server: ServerId,
        process_id: u64,
        translated: Translated,
    ) -> IpcContext<'a> {
        IpcContext {
            dispatcher: self,
            memory,
            server,
            process_id,
            translated,
            events: Vec::new(),
            end: false,
            placed: Placed::default(),
        }
    }

    /// Puts a handler back in its server's slot (see [`put_back`]), and returns the requests
    /// parked for it, which the caller runs.
    fn put_back(&self, server: ServerId, handler: Box<dyn SessionHandler>) -> Vec<ObjectId> {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        put_back(&mut locked.state, server, handler)
    }

    /// Gives every server whose handler is in its slot the notices waiting for it, under the
    /// lock. A handler out of its slot is given its notices when it is put back.
    pub(super) fn settle_notices(&self) {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        let state = &mut *locked.state;
        let servers: Vec<ServerId> = {
            let ipc = state.tables.scheduler.ipc();
            ipc.servers()
                .filter(|server| ipc.has_notices(*server).unwrap_or(false))
                .collect()
        };
        for server in servers {
            if let Some(Some(handler)) = state.handlers.get_mut(&server) {
                give_notices(&mut state.tables, server, &mut **handler);
            }
        }
    }

    /// `ConnectToNamedPort`: the name, the port, a session on its server and a handle to it.
    pub(super) fn connect_to_named_port(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        let mut notices = false;
        let outcome = self.call::<(&mut HandleTable, &AddressSpace, &mut Scheduler)>(
            call,
            |(handles, space, scheduler), call, _| {
                let memory = call.caller.memory;
                let outcome = super::returning(id, convention, call.regs, |[name]| {
                    let name = PortName::read(space, memory, name)?;
                    let server = scheduler.ipc().port(&name).ok_or(KernelError::NotFound)?;
                    let (_, handle) = open_session_with_handle(handles, scheduler, server, true)?;
                    Ok([u64::from(handle.raw())])
                });
                notices = scheduler.ipc().has_any_notices();
                outcome
            },
        );
        // The server accepts the session now, if it is free, rather than at its first request.
        if notices {
            self.settle_notices();
        }
        outcome
    }

    /// `SendSyncRequest`: see the module docs.
    pub(super) fn send_sync_request(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        let mut locked = self.lock::<(&mut HandleTable, &mut Scheduler)>();
        let time = match locked.guest_time(call.caller.env) {
            Ok(time) => time,
            Err(error) => {
                tracing::error!(number = call.number, %error, "environment failed");
                return Outcome::EnvFailed(error);
            }
        };
        let [handle] = match convention.read::<1>(call.regs) {
            Ok(inputs) => inputs,
            Err(error) => {
                let outcome = mismatch(id, error, call.regs);
                record_point(call, time, &outcome);
                return outcome;
            }
        };
        let handle = Handle::from_register(handle);
        let process_id = locked.state.tables.processes.process_id();
        let process = locked.state.tables.processes.own();
        let (handles, scheduler) = locked.tables();
        let sender = Sender {
            thread: call.caller.thread,
            waiter: calling_waiter(scheduler, call.caller.core),
            core: call.caller.core,
            message_address: call.regs.tpidrro,
            session_handle: handle,
            process: sending_process(scheduler, call.caller.core).unwrap_or(process),
        };
        let caller_process = sender.process;
        let begun = begin(
            handles,
            scheduler,
            call.caller.memory,
            (process, process_id),
            (sender, call.regs),
        );
        let (server, session) = match begun {
            Ok(Begun::Start { server, session }) => (server, session),
            Ok(Begun::Queued) => return Outcome::Waiting,
            Err(error) => {
                let outcome = match error {
                    SvcError::Kernel(error) => answer(id, convention, call.regs, error.code()),
                    SvcError::Internal(what) => fault(id, &what, call.regs),
                };
                record_point(call, time, &outcome);
                drop(locked);
                // A session the failed message destroyed: its server forgets it now, if free.
                self.settle_notices();
                return outcome;
            }
        };
        drop(locked);

        let (mut finished, parked) =
            self.run_request(server, session, call.caller.memory, process_id);
        finished.extend(self.run_ready(server, call.caller.memory, process_id, parked));
        // Replies said to be ready while this core had the handler out are this core's to
        // deliver now that it is back.
        finished.extend(self.deliver_owed(server, call.caller.memory, process_id));
        let (mine, ended) = self.settle_and_end(
            finished,
            Some(call.caller.thread),
            call.caller.memory,
            call.caller.recorder,
            time,
        );
        let outcome = if ended.iter().any(|end| end.process == caller_process) {
            // The caller's own process ended (module docs): its result is written, and the
            // call ends as `ExitProcess` does.
            if let Some(mine) = mine {
                answer(id, convention, call.regs, mine.result);
            }
            Outcome::ProcessExited
        } else {
            match mine {
                Some(mine) => answer(id, convention, call.regs, mine.result),
                // Deferred: the caller waits for its reply.
                None => Outcome::Waiting,
            }
        };
        self.settle_notices();
        outcome
    }

    /// Settles the requests that are over ([`settle_one`], in order), reports every sender but
    /// `own` woken ([`Dispatcher::take_woken`]), records and reports the waits the handlers'
    /// signals ended (`dispatch/sync.rs`), and then ends each process a handler asked to end
    /// ([`asked`], [`terminate_asked`]). Returns `own`'s result, when its request is one of
    /// them, and the processes that ended. The woken are reported in that order: the senders,
    /// then the signals' waits, then the waits the ends ended. Test:
    /// `a_delivery_reports_its_sender_and_its_signals_wakes_before_the_wakes_of_the_end` (the
    /// events are recorded in the same order, but the ends' teardown would record a signal's
    /// pending wait first anyway, so only the woken order tells a late report apart).
    ///
    /// When a process is to end, all of this happens under one hold of the lock, so that no
    /// core can run a sender of that process between its reply and the end: a sender the timer
    /// thread's delivery answers, or one a dispatch answers for another thread, is left as a
    /// dispatch's own caller is, answered and ended before it could run again. Otherwise each
    /// request is settled under a lock of its own, and the woken are reported under one more.
    /// Test: `replies_and_the_end_they_ask_for_are_settled_under_one_hold_of_the_lock`, which
    /// counts the acquisitions ([`Dispatcher::lock_waits`]). Without the one hold, a sender the
    /// timer thread answered could run in between (seen in repeated runs of
    /// `the_timer_thread_ends_the_run_when_a_delivery_ends_the_process`, `starquake-core`).
    fn settle_and_end(
        &self,
        finished: Vec<Finished>,
        own: Option<ThreadId>,
        memory: &dyn GuestMemory,
        recorder: &Recorder,
        time: GuestTicks,
    ) -> (Option<Woken>, Vec<Ended>) {
        let asked = asked(&finished);
        let mut woken = Vec::new();
        if asked.is_empty() {
            for finished in finished {
                let mut locked = self.lock::<(&mut HandleTable, &mut Scheduler)>();
                woken.extend(settle_one(&mut locked, finished, memory, recorder, time));
            }
            let mine = take_own(&mut woken, own);
            let mut locked = self.lock::<()>();
            report_woken(&mut locked.state, woken, recorder, time);
            return (mine, Vec::new());
        }
        let mut locked = self.lock::<(&mut HandleTable, &mut Scheduler)>();
        for finished in finished {
            woken.extend(settle_one(&mut locked, finished, memory, recorder, time));
        }
        let mine = take_own(&mut woken, own);
        report_woken(&mut locked.state, woken, recorder, time);
        // Only now that every reply is written and recorded (services spec: "the reply is
        // sent, and then the kernel tears the process down").
        let mut ended = Vec::new();
        for end in asked {
            if terminate_asked(&mut locked, end.process, recorder, time) {
                ended.push(end);
            }
        }
        (mine, ended)
    }

    /// Runs the request in flight on `session`, a session of `server`, through the server's
    /// handler without the lock, and ends it. Returns the requests that are over, and the
    /// requests parked for the handler while it ran, which the caller runs.
    ///
    /// When the handler is out of its slot (another core delivers its deferred replies), the
    /// request is parked instead, still in flight, and its sender waits for its reply (module
    /// docs, "Senders wait as scheduler threads"): nothing is over yet.
    fn run_request(
        &self,
        server: ServerId,
        session: ObjectId,
        memory: &dyn GuestMemory,
        process_id: u64,
    ) -> (Vec<Finished>, Vec<ObjectId>) {
        let mut locked = self.lock::<(&mut Scheduler,)>();
        let state = &mut *locked.state;
        let ipc = state.tables.scheduler.ipc_mut().1;
        // A started request is in flight, on a server that has a handler slot; anything else
        // means the kernel's bookkeeping is broken.
        let (Some(request), Some(slot)) = (
            ipc.in_flight(session).ok().flatten().cloned(),
            state.handlers.get_mut(&server),
        ) else {
            tracing::error!(
                ?server,
                ?session,
                "a started request is not in flight, or its server has no handler"
            );
            let failed = Finish::Failed(KernelError::NotImplemented);
            return (finish(ipc, session, failed, Vec::new()), Vec::new());
        };
        let Some(mut handler) = slot.take() else {
            // Out of its slot without a worker: whoever puts it back runs this request. The
            // sender waits for its reply now, as on the queued and deferred paths, before the
            // lock goes: the thread that puts the handler back may run and answer the request
            // at once, and its answer must find the sender waiting. Eden: every synchronous
            // request begins its sender's wait (`KServerSession::OnRequest`, module docs).
            state.parked.entry(server).or_default().push(session);
            await_reply(&mut state.tables.scheduler, request.sender, session);
            return (Vec::new(), Vec::new());
        };
        give_notices(&mut state.tables, server, &mut *handler);
        drop(locked);

        let sender = request.sender;
        let mut cx = self.context(memory, server, process_id, request.translated);
        let mut message: MessageBuffer = *request.message;
        let answered = handler.handle(&mut cx, session, &mut message);

        let mut locked = self.lock::<(&mut HandleTable, &mut Scheduler)>();
        let (table, scheduler) = locked.tables();
        if matches!(answered, Ok(Handled::Deferred)) {
            // The sender waits for the reply now, before the lock goes, so that a reply
            // delivered as soon as it goes finds it waiting.
            await_reply(scheduler, sender, session);
        }
        let ipc = scheduler.ipc_mut().1;
        let finished = match answered {
            Ok(Handled::Replied) => {
                finish(ipc, session, Finish::Replied(Box::new(message)), cx.events)
            }
            Ok(Handled::Deferred) => finish(ipc, session, Finish::Deferred, cx.events),
            Ok(Handled::Closed) => finish(ipc, session, Finish::ServerClosed, cx.events),
            Err(error) => {
                tracing::error!(%error, ?server, ?session, "closing the session");
                close_failed(ipc, server, session, cx.events)
            }
        };
        let finished = carry(finished, cx.placed, table, scheduler);
        let finished = ending(finished, cx.end);
        let parked = put_back(&mut locked.state, server, handler);
        (finished, parked)
    }

    /// Runs the requests `parked` for the server's handler, oldest first, then starts and runs
    /// the ready requests of `server`, one at a time, until none may start.
    fn run_ready(
        &self,
        server: ServerId,
        memory: &dyn GuestMemory,
        process_id: u64,
        parked: Vec<ObjectId>,
    ) -> Vec<Finished> {
        let mut finished = Vec::new();
        let mut parked = VecDeque::from(parked);
        // Bounded by the requests in flight and waiting: every turn runs a parked one, which
        // is then over or deferred, or starts a waiting one.
        loop {
            let next = if let Some(session) = parked.pop_front() {
                Ok(Some(session))
            } else {
                let mut locked = self.lock::<(&mut Scheduler,)>();
                let (scheduler,) = locked.tables();
                scheduler.ipc_mut().1.start_next(server)
            };
            match next {
                Ok(Some(session)) => {
                    let (done, more) = self.run_request(server, session, memory, process_id);
                    finished.extend(done);
                    parked.extend(more);
                }
                Ok(None) => return finished,
                Err(error) => {
                    tracing::error!(%error, ?server, "the next request could not start");
                    return finished;
                }
            }
        }
    }
}

/// Settles one request that is over, under `locked`: writes its reply into its sender's
/// message buffer, releases what the request held, and records its events and then its system
/// call's event, all at `time`. Returns its sender's result, unless the sender has ended.
fn settle_one(
    locked: &mut Locked<'_, (&mut HandleTable, &mut Scheduler)>,
    finished: Finished,
    memory: &dyn GuestMemory,
    recorder: &Recorder,
    time: GuestTicks,
) -> Option<Woken> {
    let Finished {
        session,
        done,
        events,
        end: _,
        placed,
    } = finished;
    let Done { request, result } = done;
    let sender = request.sender;
    let (handles, scheduler) = locked.tables();
    // A sender that ended while it waited gets no reply: its message buffer may be another
    // thread's by now.
    let alive = is_alive(scheduler, sender);
    let mut written = false;
    let mut result = result;
    if alive && result.is_ok() {
        match memory.write(sender.message_address, &request.message[..]) {
            Ok(()) => written = true,
            Err(error) => {
                // `dispatch/ipc/placed.rs`, "Divergences from Eden".
                tracing::error!(
                    ?error,
                    "a reply could not be written to its sender's TLS block"
                );
                result = Err(KernelError::InvalidCurrentMemory);
            }
        }
    }
    // The sender keeps the placed handles the reply it now holds names, and no others; they go
    // before it is woken with its result (`dispatch/ipc/placed.rs`).
    let kept = if written {
        reply_handles(&request.message)
    } else {
        Vec::new()
    };
    placed.take_back(handles, scheduler, &kept);
    let code = result.err().map_or(ResultCode::SUCCESS, KernelError::code);
    // The reply is in place before the sender can run again: both happen under the lock.
    if alive
        && let Some(waiter) = sender.waiter
        && let Err(error) = scheduler.end_reply_wait(waiter, session, code)
    {
        tracing::error!(%error, ?session, "a sender could not be woken");
    }
    let (objects, ipc) = scheduler.ipc_mut();
    if let Err(error) = ipc.release_request(objects, session, &request) {
        tracing::error!(%error, ?session, "a finished request could not let go");
    }
    for kind in events {
        recorder.record(time, sender.core, sender.thread, kind);
    }
    if alive {
        record_send(recorder, time, sender, code);
    }
    alive.then_some(Woken {
        thread: sender.thread,
        core: sender.core,
        result: code,
    })
}

/// Takes the result of `own`, the calling thread, out of `woken`: the first one that names it.
fn take_own(woken: &mut Vec<Woken>, own: Option<ThreadId>) -> Option<Woken> {
    let index = woken.iter().position(|woken| Some(woken.thread) == own)?;
    Some(woken.remove(index))
}

/// Reports the senders `woken` ([`Dispatcher::take_woken`]), and then records and reports the
/// waits the handlers' signals ended (`dispatch/sync.rs`).
fn report_woken(state: &mut State, woken: Vec<Woken>, recorder: &Recorder, time: GuestTicks) {
    state.woken.extend(woken);
    drain_wakes(state, recorder, time);
}

/// Ends `process` because a handler asked to, under `locked`, through
/// [`crate::svc::process::terminate_process`], and records and reports the waits its ending
/// threads ended. Returns whether it ended; when it could not (only the kernel's own
/// bookkeeping can make that happen: a process that has ended already), the failure is logged.
/// Test: `ending_a_process_that_has_ended_already_fails_and_reports_nothing`.
fn terminate_asked(
    locked: &mut Locked<'_, (&mut HandleTable, &mut Scheduler)>,
    process: ProcessId,
    recorder: &Recorder,
    time: GuestTicks,
) -> bool {
    let (handles, scheduler) = locked.tables();
    tracing::info!(?process, "a service ended the process after its reply");
    let ended = terminate_process(handles, scheduler, process);
    // Waits the ending threads ended (a thread's waiters wake as it terminates), recorded and
    // reported now, as `ExitProcess` does. Test:
    // `a_handler_ending_the_process_wakes_its_threads_waiters_as_exit_process_does`.
    drain_wakes(&mut locked.state, recorder, time);
    match ended {
        Ok(()) => true,
        Err(error) => {
            tracing::error!(
                %error,
                ?process,
                "a service asked to end the process, and it could not end"
            );
            false
        }
    }
}

/// The first step of `SendSyncRequest`, under the lock: the session, the message, its special
/// header, and the queue. `process` is the sender's process, whose guest-visible id is
/// `process_id`.
fn begin(
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
    memory: &dyn GuestMemory,
    (process, process_id): (ProcessId, u64),
    (sender, regs): (Sender, &GuestRegs),
) -> CallResult<Begun> {
    let session = handles
        .get(sender.session_handle)
        .map_err(|_| KernelError::InvalidHandle)?;
    if scheduler.objects().get(session)? != Object::Session {
        return Err(KernelError::InvalidHandle.into());
    }
    // A closed server end refuses before anything of the message is taken.
    if !scheduler.ipc().is_server_open(session)? {
        return Err(KernelError::SessionClosed.into());
    }
    let mut message: MessageBuffer = [0; BUFFER_SIZE];
    memory
        .read(sender.message_address, &mut message)
        .map_err(|error| {
            SvcError::Internal(format!(
                "the caller's message buffer at {:#x} cannot be read: {error:?}",
                sender.message_address
            ))
        })?;
    // What the two pseudo-handles name for this sender (`GetObjectForIpc`'s `cur_thread` and
    // its owner process). A process the scheduler has no row for (only tables built by hand,
    // not by `process::create`) has no object, and a copied `CURRENT_PROCESS` is then an
    // invalid handle; the rest of the request is unaffected.
    let current = Current {
        thread: calling_thread_object(scheduler, sender.core),
        process: scheduler
            .process(process)
            .ok()
            .map(crate::process::ProcessRecord::object),
    };
    let (objects, ipc) = scheduler.ipc_mut();
    let Translation {
        translated,
        result,
        destroyed,
    } = translate(&mut message, process_id, current, handles, objects);
    // What a moved handle's removal destroyed (only once the request has failed) is torn down
    // as closing its last handle tears it down (`svc::handle::close_handle`).
    let forgotten = destroyed
        .into_iter()
        .try_for_each(|object| ipc.forget(object));
    if let Err(error) = result {
        ipc.release_translated(objects, &translated)?;
        forgotten?;
        return Err(error.into());
    }
    forgotten?;
    let request = Request {
        sender,
        message: Box::new(message),
        translated,
    };
    let admission = match ipc.send(objects, session, request) {
        Ok(admission) => admission,
        Err(refused) => {
            ipc.release_translated(objects, &refused.request.translated)?;
            return Err(refused.error.into());
        }
    };
    let server = ipc.server_of(session)?;
    if let Some(waiter) = sender.waiter {
        // The sender's registers are its context from here on, before the lock goes: if it
        // waits, the reply's result is written there and a switch loads it (kernel slice 10).
        *scheduler.context_mut(waiter)? = regs.clone();
    }
    match admission {
        Admission::Start => Ok(Begun::Start { server, session }),
        Admission::Queued => {
            if sender.waiter.is_some() && !await_reply(scheduler, sender, session) {
                // A sender that cannot wait must not leave a request nobody waits for.
                let (objects, ipc) = scheduler.ipc_mut();
                if let Some(waiter) = sender.waiter
                    && let Some(request) = ipc.withdraw(session, waiter)?
                {
                    ipc.release_request(objects, session, &request)?;
                }
                return Err(SvcError::Internal(
                    "a sender whose request waits could not wait for it".to_owned(),
                ));
            }
            Ok(Begun::Queued)
        }
    }
}

/// The scheduler's thread on `core`, if it runs one that may begin a wait: what a request's
/// sender waits as ([`Sender::waiter`]). A core the scheduler does not name, or whose thread is
/// not runnable or is in a wait already (only a caller that names its thread by hand can be),
/// has none, and its sender is only reported woken.
fn calling_waiter(scheduler: &Scheduler, core: u8) -> Option<crate::id::ThreadId> {
    let thread = calling_scheduler_thread(scheduler, core)?;
    let runnable = scheduler
        .thread(thread)
        .is_ok_and(|row| row.state() == crate::sched::ThreadState::Runnable);
    (runnable && scheduler.sync().wait(thread).is_none()).then_some(thread)
}

/// The object of the thread the scheduler says `core` runs: what a copied
/// [`Handle::CURRENT_THREAD`] names. `None` when the core runs no scheduler thread, or one with
/// no object (only a thread of no process, which the core loop never runs for a guest).
fn calling_thread_object(scheduler: &Scheduler, core: u8) -> Option<ObjectId> {
    let thread = calling_scheduler_thread(scheduler, core)?;
    scheduler.thread(thread).ok()?.object()
}

/// The scheduler's thread on the sender's `core`, if the scheduler names that core and it runs
/// a thread: what [`calling_waiter`] and [`calling_thread_object`] start from.
fn calling_scheduler_thread(scheduler: &Scheduler, core: u8) -> Option<crate::id::ThreadId> {
    let core = crate::sched::CoreId::new(core)?;
    scheduler.current(core).ok().flatten()
}

/// The process of the scheduler's thread on the sender's `core`, when the scheduler names that
/// core and it runs a thread of a process: what the request is sent for ([`Sender::process`]).
/// `None` for a core that runs no scheduler thread, or a thread of no process.
fn sending_process(scheduler: &Scheduler, core: u8) -> Option<ProcessId> {
    let thread = calling_scheduler_thread(scheduler, core)?;
    scheduler.thread(thread).ok()?.process()
}

/// Makes the scheduler's thread behind `sender` wait for the reply to its request on
/// `session`, unless it waits for that reply already (a queued request that was started and
/// then deferred). Returns whether the sender now waits; one with no scheduler thread does
/// not, and is only reported woken. A sender that ended while its handler ran (its request is
/// in flight, `Scheduler::forget_waits`) does not wait either: it stays ended, and its reply
/// finds it so (`settle_one`). A failure is the kernel's own bookkeeping, and is
/// logged.
fn await_reply(scheduler: &mut Scheduler, sender: Sender, session: ObjectId) -> bool {
    let Some(waiter) = sender.waiter else {
        return false;
    };
    if !is_alive(scheduler, sender) {
        return false;
    }
    if matches!(
        scheduler.sync().wait(waiter),
        Some(crate::sync::Wait::Reply { session: on, .. }) if *on == session
    ) {
        return true;
    }
    let origin = WaitOrigin {
        thread: sender.thread,
        core: sender.core,
        args: send_args(sender),
    };
    match scheduler.wait_for_reply(waiter, session, origin) {
        Ok(()) => true,
        Err(error) => {
            tracing::error!(%error, ?session, "a sender could not wait for its reply");
            false
        }
    }
}

/// Whether a request's sender can still be answered: one the scheduler does not know is (it
/// is only reported), and one it knows must not have ended.
fn is_alive(scheduler: &Scheduler, sender: Sender) -> bool {
    sender.waiter.is_none_or(|waiter| {
        scheduler
            .thread(waiter)
            .is_ok_and(|row| row.state() != crate::sched::ThreadState::Terminated)
    })
}

/// Ends a started request and returns what is over, each with `events` for the first.
fn finish(
    ipc: &mut crate::ipc::Ipc,
    session: ObjectId,
    ending: Finish,
    events: Vec<EventKind>,
) -> Vec<Finished> {
    let done = match ipc.finish(session, ending) {
        Ok(done) => done,
        Err(error) => {
            tracing::error!(%error, ?session, "a request could not be ended");
            return Vec::new();
        }
    };
    let mut events = Some(events);
    done.into_iter()
        .map(|done| Finished {
            session,
            done,
            events: events.take().unwrap_or_default(),
            end: None,
            placed: Placed::default(),
        })
        .collect()
}

/// Gives the first of `finished`, the request a handler ran, the handles the handler `placed`
/// in its sender's table, to be kept or taken back when it is settled. When nothing is over,
/// they are taken back now, from `handles`: the handler deferred its reply (a deferred reply's
/// handles are placed when it is written; `dispatch/ipc/placed.rs`), or ending the request
/// failed on the kernel's own bookkeeping ([`finish`] logs that failure and returns nothing).
fn carry(
    mut finished: Vec<Finished>,
    placed: Placed,
    handles: &mut HandleTable,
    scheduler: &mut Scheduler,
) -> Vec<Finished> {
    match finished.first_mut() {
        Some(first) => first.placed = placed,
        None => placed.take_back(handles, scheduler, &[]),
    }
    finished
}

/// Marks the first of `finished`, the request a handler ran, as asking to end its sender's
/// process ([`Sender::process`]) when `end` says the handler asked ([`IpcContext::end_process`]).
/// When `finished` is empty there is nothing to mark, and the ask is not honoured, and logged.
/// That happens in two cases: the handler deferred its reply (the end follows the reply, and there is none
/// yet; module docs), or ending the request failed on the kernel's own bookkeeping
/// ([`finish`] logs that failure and returns nothing).
fn ending(mut finished: Vec<Finished>, end: bool) -> Vec<Finished> {
    if end {
        if let Some(first) = finished.first_mut() {
            first.end = Some(first.done.request.sender.process);
        } else {
            tracing::error!(
                "a handler asked to end the process, and its request is not over (its reply \
                 is deferred, or ending the request failed); the process goes on"
            );
        }
    }
    finished
}

/// The processes the handlers of `finished` asked to end, each once, in the order first asked,
/// each with the sender of the first request that asked for it. Bounded by `finished`.
fn asked(finished: &[Finished]) -> Vec<Ended> {
    let mut asked: Vec<Ended> = Vec::new();
    for finished in finished {
        let Some(process) = finished.end else {
            continue;
        };
        if asked.iter().any(|end| end.process == process) {
            continue;
        }
        let sender = finished.done.request.sender;
        asked.push(Ended {
            process,
            thread: sender.thread,
            core: sender.core,
        });
    }
    asked
}

/// Closes the server end of a session whose handler failed (see [`HandlerError`](crate::ipc::HandlerError)): its requests
/// end with [`KernelError::SessionClosed`], and the handler, which still has the session, is
/// told to forget it.
fn close_failed(
    ipc: &mut crate::ipc::Ipc,
    server: ServerId,
    session: ObjectId,
    events: Vec<EventKind>,
) -> Vec<Finished> {
    if let Err(error) = ipc.notify(server, Notice::Closed(session)) {
        tracing::error!(%error, ?server, "the server cannot be told of a session it lost");
    }
    finish(ipc, session, Finish::ServerClosed, events)
}

/// Puts a handler back in its server's slot, under the lock, after giving it the notices that
/// arrived while it was out, and returns the requests parked for it meanwhile.
fn put_back(
    state: &mut State,
    server: ServerId,
    mut handler: Box<dyn SessionHandler>,
) -> Vec<ObjectId> {
    give_notices(&mut state.tables, server, &mut *handler);
    state.handlers.insert(server, Some(handler));
    state.parked.remove(&server).unwrap_or_default()
}

/// Gives the handler of `server` the notices waiting for it, under the lock.
fn give_notices(tables: &mut Tables, server: ServerId, handler: &mut dyn SessionHandler) {
    match tables.scheduler.ipc_mut().1.take_notices(server) {
        Ok(notices) => apply_notices(handler, server, notices),
        Err(error) => tracing::error!(%error, ?server, "a server's notices cannot be taken"),
    }
}

/// Gives a handler its notices, in order.
fn apply_notices(handler: &mut dyn SessionHandler, server: ServerId, notices: Vec<Notice>) {
    for notice in notices {
        match notice {
            Notice::Accept(session) => {
                if let Err(error) = handler.accept(session) {
                    tracing::error!(%error, ?server, ?session, "the server could not accept a session");
                }
            }
            Notice::Closed(session) => handler.closed(session),
        }
    }
}

/// Writes a result into `x0` as the call's only output.
fn answer(id: SvcId, convention: &Convention, regs: &mut GuestRegs, result: ResultCode) -> Outcome {
    complete(id, convention, regs, &[result_word(result)])
}

/// The decoded arguments a sender's `SendSyncRequest` event carries: its session handle.
fn send_args(sender: Sender) -> [u64; SVC_ARG_COUNT] {
    let mut passed = [0; SVC_ARG_COUNT];
    passed[0] = u64::from(sender.session_handle.raw());
    recorded_args(Some(SvcId::SendSyncRequest), &passed)
}

/// Records the `SendSyncRequest` event of a sender woken by someone else's dispatch.
fn record_send(recorder: &Recorder, time: GuestTicks, sender: Sender, result: ResultCode) {
    recorder.record(
        time,
        sender.core,
        sender.thread,
        EventKind::Svc {
            id: u32::from(SvcId::SendSyncRequest.number()),
            args: send_args(sender),
            result: starquake_trace::ResultCode(result.raw()),
        },
    );
}
