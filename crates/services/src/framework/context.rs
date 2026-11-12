//! [`Context`]: what a service method may touch outside its own state.

use starquake_kernel::id::ObjectId;

use super::kernel::Kernel;
use super::result::Fault;
use super::targets::Targets;

/// What a service method may touch while it handles a request: the caller's memory, by copying in
/// and out, and a narrow part of the kernel.
///
/// Buffers a request describes are reached through the request
/// ([`Request::in_buffer`](super::Request::in_buffer),
/// [`Request::out_buffer`](super::Request::out_buffer)); the memory calls here are for addresses a
/// command carries in its payload. The clock and `defer()` of the service spec arrive with the
/// slices that need them.
pub struct Context<'a> {
    pub(super) kernel: &'a mut dyn Kernel,
    pub(super) targets: &'a mut Targets,
}

impl Context<'_> {
    /// Creates an event. Keep the id to signal it later; give the caller a handle with
    /// [`Reply::copy_handle`](super::Reply::copy_handle).
    pub fn create_event(&mut self) -> Result<ObjectId, Fault> {
        Ok(self.kernel.create_event()?)
    }

    /// Signals an event.
    pub fn signal_event(&mut self, event: ObjectId) -> Result<(), Fault> {
        Ok(self.kernel.signal_event(event)?)
    }

    /// Copies `out.len()` bytes of the caller's memory at `address` into `out`.
    pub fn read_memory(&self, address: u64, out: &mut [u8]) -> Result<(), Fault> {
        Ok(self.kernel.read_memory(address, out)?)
    }

    /// Copies `data` into the caller's memory at `address`.
    pub fn write_memory(&mut self, address: u64, data: &[u8]) -> Result<(), Fault> {
        Ok(self.kernel.write_memory(address, data)?)
    }
}
