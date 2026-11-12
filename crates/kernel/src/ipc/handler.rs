//! [`SessionHandler`]: the code behind an HLE server, as the kernel drives it (kernel spec, "IPC":
//! "The handler interface is defined here ... and implemented by the service framework").
//!
//! A server's handler is given one request at a time, without the kernel lock held, and with
//! an [`IpcContext`] through which it reaches the kernel for the request's sender. It writes its
//! reply over the request, closes the session, or defers. Before each request, and as soon as it
//! is free otherwise, it is told of sessions clients opened and closed on it.

use crate::dispatch::IpcContext;
use crate::id::ObjectId;

/// What a handler did with a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handled {
    /// The reply is in the message buffer.
    Replied,
    /// The request closed the session: the server has forgotten it, and nothing was written.
    /// The kernel closes the server end, so the sender sees
    /// [`KernelError::SessionClosed`](crate::result::KernelError::SessionClosed).
    Closed,
    /// The handler will reply later ([`SessionHandler::take_reply`]): the request stays in
    /// flight, its sender waits, and the server takes its next request.
    Deferred,
}

/// A handler's own failure: it has no reply for the request. The kernel logs it and closes the
/// server end of the session, so the sender sees
/// [`KernelError::SessionClosed`](crate::result::KernelError::SessionClosed); a console's server
/// framework does the same with a request it fails to process (fact only:
/// `atmosphere/libraries/libstratosphere/source/sf/hipc/sf_hipc_server_session_manager.cpp`,
/// `ProcessRequest`: "All other results indicate something went very wrong", then
/// `CloseSessionImpl`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the session handler failed: {0}")]
pub struct HandlerError(pub String);

/// An HLE server's code. See the module docs.
pub trait SessionHandler: Send {
    /// A client opened `session` on this server by connecting to it (a named port, or a session
    /// handed out for this server by another one): open it with the server's root interface.
    fn accept(&mut self, session: ObjectId) -> Result<(), HandlerError>;

    /// The client end of `session` is gone: forget it. A session the server already forgot (it
    /// closed it itself) is not an error.
    fn closed(&mut self, session: ObjectId);

    /// Handles the request in `message`, sent on `session`, writing the reply over it.
    fn handle(
        &mut self,
        cx: &mut IpcContext<'_>,
        session: ObjectId,
        message: &mut [u8],
    ) -> Result<Handled, HandlerError>;

    /// Writes the reply of the oldest deferred request whose answer is ready into `message`,
    /// and returns its session with whether the reply could be written; `None` when no
    /// deferred reply is ready. A reply that could not be written closes the session, as a
    /// failed [`SessionHandler::handle`] does. `cx` speaks for the request's sender: handles go
    /// into its process's table.
    fn take_reply(
        &mut self,
        cx: &mut IpcContext<'_>,
        message: &mut [u8],
    ) -> Option<(ObjectId, Result<(), HandlerError>)>;
}
