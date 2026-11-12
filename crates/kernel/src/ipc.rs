//! Sessions, servers and named ports: the client side of IPC (kernel spec, "IPC"; port plan
//! slice 9).
//!
//! - [`Ipc`] is the table: servers, sessions with their requests, and the names ports and
//!   services are known by. It lives in the scheduler's table, under the kernel lock.
//! - [`queue`] decides which of a server's requests starts when: one in flight per session,
//!   one worker per server, and a deferred request gives its worker up.
//! - [`message`] is the kernel's half of a request message (the process id and the handles).
//! - [`port`] is a port's name, as Horizon keeps it and as a guest passes it.
//! - [`SessionHandler`] is the code behind a server, which the dispatcher keeps outside the
//!   tables and calls without the kernel lock ([`IpcContext`] is what it reaches the kernel
//!   through).
//!
//! The system calls are dispatcher arms (`crate::dispatch`): `ConnectToNamedPort` (0x1f),
//! `SendSyncRequest` (0x21), and `CloseHandle` (0x16) when it closes a session's last handle.
//!
//! Deferred, as the kernel spec lists: server-side calls (`ReplyAndReceive`, `AcceptSession`,
//! `CreateSession`, ports a guest manages), light sessions, asynchronous requests and mapping
//! IPC buffers into a server. A guest server would be another kind of server taking requests
//! from the same session rows; nothing in the client path here knows which kind it talks to.

pub mod handler;
pub mod message;
pub mod port;
pub mod queue;
mod table;

pub use crate::dispatch::IpcContext;
pub use handler::{Handled, HandlerError, SessionHandler};
pub use message::{MessageBuffer, Translated};
pub use port::{NAME_LENGTH_MAX, PortName, PortNameError};
pub use queue::{Admission, DEFAULT_WORKERS, QueueError, RequestQueue};
pub use table::{Done, Finish, Ipc, IpcError, Notice, Refused, Request, Sender};
