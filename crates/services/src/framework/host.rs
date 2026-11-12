//! The framework on the real kernel: [`Kernel`] for the kernel's
//! [`IpcContext`], and the kernel's [`SessionHandler`] for [`Server`] (kernel spec, "IPC": the
//! handler interface "is defined here and implemented by the service framework").
//!
//! The kernel drives a [`Server`] through [`SessionHandler`] and keeps the request queue that
//! decides when a request starts (`starquake_kernel::ipc::queue`, the same type the server keeps
//! its own with). The kernel starts a request only when its rules allow it, and the server's
//! own queue follows the same rules, so the server admits every request the kernel gives it; a
//! request the server refuses ([`ServerError::Busy`]) is a disagreement between the two and
//! closes the session, as every [`ServerError`] does ([`HandlerError`]).
//!
//! What each call of the kernel's handler interface becomes:
//!
//! | `SessionHandler` | `Server` |
//! |---|---|
//! | `accept` | [`Server::accept`] |
//! | `closed` | [`Server::close`]; a session the server already forgot (it closed it on a close request) is not an error |
//! | `handle` | [`Server::handle`], with [`Handled`](super::Handled) mapped one for one |
//! | `take_reply` | [`Server::take_completed`], then [`Server::deliver`] into the kernel's buffer |

use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::ipc::{Handled, HandlerError, IpcContext, SessionHandler};
use starquake_kernel::result::SvcResult;
use starquake_trace::{EventKind, ServiceId};

use super::kernel::Kernel;
use super::server::{self, Server, ServerError};

impl Kernel for IpcContext<'_> {
    fn process_id(&self) -> u64 {
        IpcContext::process_id(self)
    }

    fn resolve_handle(&self, handle: Handle) -> SvcResult<ObjectId> {
        IpcContext::resolve_handle(self, handle)
    }

    fn insert_handle(&mut self, object: ObjectId) -> SvcResult<Handle> {
        IpcContext::insert_handle(self, object)
    }

    fn create_event(&mut self) -> SvcResult<ObjectId> {
        IpcContext::create_event(self)
    }

    fn signal_event(&mut self, event: ObjectId) -> SvcResult<()> {
        IpcContext::signal_event(self, event)
    }

    fn create_session(&mut self) -> SvcResult<(ObjectId, Handle)> {
        IpcContext::create_session(self)
    }

    fn connect(&mut self, service: ServiceId) -> SvcResult<(ObjectId, Handle)> {
        IpcContext::connect(self, service)
    }

    fn read_memory(&self, address: u64, out: &mut [u8]) -> SvcResult<()> {
        IpcContext::read_memory(self, address, out)
    }

    fn write_memory(&mut self, address: u64, data: &[u8]) -> SvcResult<()> {
        IpcContext::write_memory(self, address, data)
    }

    fn record(&mut self, kind: EventKind) {
        IpcContext::record(self, kind);
    }
}

/// A server failure as the kernel's handler interface reports it.
fn failed(error: &ServerError) -> HandlerError {
    HandlerError(error.to_string())
}

impl SessionHandler for Server {
    fn accept(&mut self, session: ObjectId) -> Result<(), HandlerError> {
        Server::accept(self, session).map_err(|error| failed(&error))
    }

    fn closed(&mut self, session: ObjectId) {
        match Server::close(self, session) {
            Ok(()) | Err(ServerError::UnknownSession(_)) => {}
            Err(error) => {
                tracing::error!(service = %self.name(), %error, ?session, "closing a session failed");
            }
        }
    }

    fn handle(
        &mut self,
        cx: &mut IpcContext<'_>,
        session: ObjectId,
        message: &mut [u8],
    ) -> Result<Handled, HandlerError> {
        match Server::handle(self, cx, session, message) {
            Ok(server::Handled::Replied) => Ok(Handled::Replied),
            Ok(server::Handled::Closed) => Ok(Handled::Closed),
            Ok(server::Handled::Deferred(_)) => Ok(Handled::Deferred),
            Err(error) => Err(failed(&error)),
        }
    }

    fn take_reply(
        &mut self,
        cx: &mut IpcContext<'_>,
        message: &mut [u8],
    ) -> Option<(ObjectId, Result<(), HandlerError>)> {
        let completed = self.take_completed()?;
        let session = completed.session();
        let delivered = self.deliver(cx, completed, message);
        Some((session, delivered.map_err(|error| failed(&error))))
    }
}
