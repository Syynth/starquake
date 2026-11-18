//! The kernel as the service framework sees it: the [`Kernel`] trait.
//!
//! The kernel's `IpcContext` implements it (`framework/host.rs`, kernel spec port plan slice 9);
//! `FakeKernel` (in `framework::fake`, behind the `test-support` feature) stands in for tests.
//! Every call is about the request being handled: "the caller" is the guest thread that sent
//! it, and its process's handle table and memory are the ones meant.
//! That holds for a deferred request too: when its reply is delivered, the kernel given to
//! [`Server::deliver`](super::Server::deliver) speaks for the thread that sent it.
//!
//! Failures are the kernel's own result codes ([`KernelError`]).
//! When a service method lets one through with `?`, the caller sees that code.
//!
//! # The graphics server's calls
//!
//! Four calls are for the graphics server (`docs/specs/services.md`, "Writing a service": the
//! kernel interface "also offers" them; kernel spec, "Waiting and waking" and "Address space"):
//! [`Kernel::create_hooked_event`], [`Kernel::is_gpu_accessible`],
//! [`Kernel::lock_device_share`] and [`Kernel::unlock_device_share`]. Each has a default that
//! answers [`KernelError::NotImplemented`], so a kernel double written for another service need
//! not offer them; the kernel's `IpcContext` and `FakeKernel` both do.
//!
//! Two more, with the same kind of default, give a server the transfer memories a request
//! carries (graphics-server spec, "What this spec needs from the kernel": "Transfer memory
//! (204, video 200)"): [`Kernel::transfer_memory`] and [`Kernel::write_transfer_memory`].
//!
//! # Clearing an event, and ending the caller's process
//!
//! Two more calls the services spec adds (approved 2026-11-16), with the same kind of default:
//! [`Kernel::clear_event`], for an event the service made (Eden clears the applet message event
//! at `SetFocusHandlingMode(0, 0, 1)`), and [`Kernel::end_process`], for
//! `ISelfController::Exit`: the request is recorded, the reply is sent, and then the kernel
//! ends the process through its normal termination path.

use std::sync::Arc;

use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::ipc::EventHook;
use starquake_kernel::memory::{DeviceShareToken, SharedPages, SvcPermission, TransferMemory};
use starquake_kernel::result::{KernelError, SvcResult};
use starquake_trace::{EventKind, ServiceId};

/// What the framework and its services ask of the kernel while a request is handled.
pub trait Kernel {
    /// The caller's process id as the guest knows it (what `svcGetProcessId` returns), not a
    /// table id. A request that asks for the sender's id gets this, as in Eden, which ignores the
    /// two id words in the message (`eden/src/core/hle/service/hle_ipc.cpp`,
    /// `ParseCommandBuffer`: `pid = thread->GetOwnerProcess()->GetProcessId()`).
    fn process_id(&self) -> u64;

    /// The object a handle in the caller's table names.
    fn resolve_handle(&self, handle: Handle) -> SvcResult<ObjectId>;

    /// Adds a handle to `object` to the caller's table.
    fn insert_handle(&mut self, object: ObjectId) -> SvcResult<Handle>;

    /// Creates an event and returns it. The service keeps the id to signal it later and gives the
    /// caller a handle with [`Reply::copy_handle`](super::Reply::copy_handle).
    fn create_event(&mut self) -> SvcResult<ObjectId>;

    /// Signals an event, waking the guest threads that wait on it.
    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()>;

    /// Clears an event this server made ([`Kernel::create_event`], either end of one
    /// [`Kernel::create_hooked_event`] made): a signalled event stops being signalled, and one
    /// that is not stays as it is. Any other id, an event another server or the guest made
    /// included, is [`KernelError::InvalidHandle`], and nothing changes.
    ///
    /// The default answers [`KernelError::NotImplemented`].
    fn clear_event(&mut self, _event: ObjectId) -> SvcResult<()> {
        Err(KernelError::NotImplemented)
    }

    /// Ends the caller's process once the request being handled is answered: its reply is
    /// written and the request recorded first, and then the kernel ends the process as
    /// `ExitProcess` does (every thread ends, every handle closes). Asking twice is asking once.
    /// The request must be answered now: the kernel does not honour the ask for a deferred one.
    ///
    /// The default answers [`KernelError::NotImplemented`].
    fn end_process(&mut self) -> SvcResult<()> {
        Err(KernelError::NotImplemented)
    }

    /// Creates a shared memory of `size` bytes, rounded up to whole pages, that the caller maps
    /// with `permission` (read, or read and write), and returns it with its pages. The service
    /// keeps the id and the pages: it writes the pages by offset, and gives the caller a handle
    /// with [`Reply::copy_handle`](super::Reply::copy_handle), which the caller maps with
    /// `MapSharedMemory`. A size of zero is the kernel's "invalid size" error, and any other
    /// permission its "invalid new memory permission" error.
    fn create_shared_memory(
        &mut self,
        size: u64,
        permission: SvcPermission,
    ) -> SvcResult<(ObjectId, SharedPages)>;

    /// Creates a new session on the server that is handling this request, and adds a handle to
    /// its client end to the caller's table. Returns the session and that handle. All or nothing:
    /// on an error no session exists and no handle was added.
    fn create_session(&mut self) -> SvcResult<(ObjectId, Handle)>;

    /// Creates a new session on the server of another service, and adds a handle to its client
    /// end to the caller's table. Returns the session and that handle. All or nothing, as
    /// [`Kernel::create_session`].
    ///
    /// The kernel tells that server of the session ([`Server::accept`](super::Server::accept))
    /// before it gives it a request on it. `service` is the number the service was registered
    /// under ([`Server::with_service_id`](super::Server::with_service_id)); one the kernel does
    /// not know is its "not found" error.
    fn connect(&mut self, service: ServiceId) -> SvcResult<(ObjectId, Handle)>;

    /// Creates an event with both ends, whose writable end the service gives the caller (with
    /// [`Reply::copy_handle`](super::Reply::copy_handle)) to signal, and attaches `hook` to it:
    /// the kernel runs the hook, after releasing its lock, each time the caller signals that
    /// end with `SignalEvent` (`starquake_kernel::ipc::hook` says exactly when). Returns the
    /// writable end and the readable end, in that order.
    ///
    /// The default answers [`KernelError::NotImplemented`].
    fn create_hooked_event(
        &mut self,
        _hook: Arc<dyn EventHook>,
    ) -> SvcResult<(ObjectId, ObjectId)> {
        Err(KernelError::NotImplemented)
    }

    /// Whether every page the caller's `size` bytes at `address` touch has the GPU-accessible
    /// attribute that `svc 0x80` sets. False for an empty range and one that is not wholly
    /// inside the caller's address space.
    ///
    /// The default answers [`KernelError::NotImplemented`].
    fn is_gpu_accessible(&self, _address: u64, _size: u64) -> SvcResult<bool> {
        Err(KernelError::NotImplemented)
    }

    /// Takes a device-share lock on the caller's `size` bytes at `address`: while it is held,
    /// the CPU mapping under the range cannot change. Returns the token naming the process and
    /// the exact range (graphics-server spec, "The model"). Locks nest; each is released by one
    /// unlock of its token: on the timer thread through gsv's completion queue
    /// (`Dispatcher::unlock_device_share`), or during a request with
    /// [`Kernel::unlock_device_share`]. The kernel's errors are its address space's
    /// (`AddressSpace::lock_device_share`), and [`KernelError::TerminationRequested`], with
    /// nothing locked, once the process has ended.
    ///
    /// The default answers [`KernelError::NotImplemented`].
    fn lock_device_share(&mut self, _address: u64, _size: u64) -> SvcResult<DeviceShareToken> {
        Err(KernelError::NotImplemented)
    }

    /// Releases the device-share lock `token` names, during a request: what a service does when
    /// a request that took a lock fails after taking it. A token the kernel refuses (another
    /// process's, or one whose lock is gone) is [`KernelError::InvalidCurrentMemory`], and one
    /// whose process has ended is [`KernelError::TerminationRequested`] (`DeviceShareError`'s
    /// mapping).
    ///
    /// The default answers [`KernelError::NotImplemented`].
    fn unlock_device_share(&mut self, _token: DeviceShareToken) -> SvcResult<()> {
        Err(KernelError::NotImplemented)
    }

    /// The transfer memory `object` names: one of the request's own copied handles, resolved
    /// ([`Request::copy_handle`](super::Request::copy_handle)), as the graphics server is given
    /// the two of its shared syncpoint page. [`KernelError::InvalidHandle`] when it names no
    /// live transfer memory (the kernel's `IpcContext::transfer_memory`).
    ///
    /// The default answers [`KernelError::NotImplemented`].
    fn transfer_memory(&self, _object: ObjectId) -> SvcResult<TransferMemory> {
        Err(KernelError::NotImplemented)
    }

    /// Copies `data` into the transfer memory `object` names, at `offset` from its start:
    /// what a server does instead of mapping it. [`KernelError::InvalidHandle`] as for
    /// [`Kernel::transfer_memory`], and [`KernelError::InvalidCurrentMemory`] unless the whole
    /// of `data` lies inside it; nothing is copied then (the kernel's
    /// `IpcContext::write_transfer_memory`).
    ///
    /// The default answers [`KernelError::NotImplemented`].
    fn write_transfer_memory(
        &mut self,
        _object: ObjectId,
        _offset: u64,
        _data: &[u8],
    ) -> SvcResult<()> {
        Err(KernelError::NotImplemented)
    }

    /// Copies `out.len()` bytes of the caller's memory at `address` into `out`. Fails without
    /// copying anything unless the whole range is readable.
    fn read_memory(&self, address: u64, out: &mut [u8]) -> SvcResult<()>;

    /// Copies `data` into the caller's memory at `address`. Fails without copying anything unless
    /// the whole range is writable.
    fn write_memory(&mut self, address: u64, data: &[u8]) -> SvcResult<()>;

    /// Records one event in the run's event stream, as happening now on the caller's thread and
    /// core (`docs/specs/observability.md`, "The event stream"). The framework calls it once
    /// for every request it answers, when the reply is written; services do not call it. The
    /// kernel stamps the event and gives it its sequence number under its own lock, which is
    /// what puts service events in the kernel's order.
    fn record(&mut self, kind: EventKind);
}
