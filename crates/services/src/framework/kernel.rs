//! The kernel as the service framework sees it: the [`Kernel`] trait.
//!
//! Defined here for now (issue #77). The kernel's session slice (kernel spec, port plan slice 9)
//! will implement it over its tables; until then `FakeKernel` (in `framework::fake`, behind
//! the `fakes` feature) stands in. Every call is about the request being handled: "the caller" is
//! the guest thread that sent it, and its process's handle table and memory are the ones meant.
//!
//! Failures are the kernel's own result codes ([`KernelError`](starquake_kernel::result::KernelError)).
//! When a service method lets one through with `?`, the caller sees that code.

use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::SvcResult;

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

    /// Creates a new session on the server that is handling this request, and adds a handle to
    /// its client end to the caller's table. Returns the session and that handle. All or nothing:
    /// on an error no session exists and no handle was added.
    fn create_session(&mut self) -> SvcResult<(ObjectId, Handle)>;

    /// Copies `out.len()` bytes of the caller's memory at `address` into `out`. Fails without
    /// copying anything unless the whole range is readable.
    fn read_memory(&self, address: u64, out: &mut [u8]) -> SvcResult<()>;

    /// Copies `data` into the caller's memory at `address`. Fails without copying anything unless
    /// the whole range is writable.
    fn write_memory(&mut self, address: u64, data: &[u8]) -> SvcResult<()>;
}
