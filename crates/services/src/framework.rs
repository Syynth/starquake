//! The service framework: what every HLE service is written against
//! (`docs/specs/services.md`, port plan slices 1 to 3, and the snapshot hook of
//! `docs/specs/observability.md`).
//!
//! # Writing a service
//!
//! A service is a struct per session, a command table and one method per command:
//!
//! ```
//! use starquake_services::commands;
//! use starquake_services::framework::{Context, Outcome, Reply, Request};
//!
//! #[derive(serde::Serialize, serde::Deserialize)]
//! pub struct Mixer {
//!     volume: u32,
//! }
//!
//! commands! {
//!     impl Mixer {
//!         0 => set_volume,
//!         1 => get_volume,
//!         2 => open_channel,
//!     }
//! }
//!
//! impl Mixer {
//!     // in: u32 volume
//!     fn set_volume(&mut self, _cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
//!         self.volume = req.payload().read_u32()?;
//!         reply.ok()
//!     }
//!
//!     // out: u32 volume
//!     fn get_volume(&mut self, _cx: &mut Context<'_>, _req: &Request<'_>, reply: &mut Reply) -> Outcome {
//!         reply.payload().write_u32(self.volume);
//!         reply.ok()
//!     }
//!
//!     // out: a new interface
//!     fn open_channel(&mut self, _cx: &mut Context<'_>, _req: &Request<'_>, reply: &mut Reply) -> Outcome {
//!         reply.push_session(Box::new(Mixer { volume: self.volume }));
//!         reply.ok()
//!     }
//! }
//! ```
//!
//! The session struct is plain serialisable data: numbers, strings, collections, and ids for
//! anything shared (an [`ObjectId`](starquake_kernel::id::ObjectId) or a `TargetId`, not a
//! reference). State that several sessions share is kept once, outside them, and each session
//! holds its id. A session holds no host handle, closure or lock, because none of those can be
//! saved. This is the standing rule of the observability spec (`docs/specs/observability.md`,
//! "Snapshots", decision 3): it is cheap only if it exists before the first service is written.
//! [`commands!`](crate::commands) enforces it: it implements [`Session::save`] with `serde`, so a
//! struct that holds something that cannot be saved does not compile as a service. See
//! [`snapshot`].
//!
//! - [`commands!`](crate::commands) implements [`Session`] for the struct: the dispatch `match`
//!   and the table of ids and names.
//! - [`Request`] gives the command id, the raw payload through a [`Reader`], buffers by index
//!   ([`InBuffer`], [`OutBuffer`]), the caller's process id when it was sent, handles resolved to
//!   kernel objects, and a domain request's input objects.
//! - [`Reply`] takes a raw payload through a [`Writer`], handles to copy or move, new
//!   interfaces, and sessions on other services. A method ends with `reply.ok()`, or with
//!   `reply.deferred()` after taking a reply token, or fails with a [`Fault`]: `?` on any
//!   framework call, or `Err(code.into())` with the service's own result code.
//! - [`Context`] is the rest of what a method may touch: guest memory by copy, events, the
//!   state its family shares ([`Context::family`]), and a reply token for a request it cannot
//!   answer yet ([`Context::defer`], see [`defer`]).
//!
//! # Running one
//!
//! A [`Server`] holds one service family's sessions, the state they share ([`Family`]) and its
//! request queue. The kernel's session code (kernel spec, port plan slice 9) drives it:
//!
//! - A session a client made by connecting to the service is opened with [`Server::accept`],
//!   which gives it the service's root interface.
//! - A request is submitted ([`Server::submit`]). One that may start is given to
//!   [`Server::handle`] with the caller's message buffer, and the reply is written over the
//!   request. One that may not waits, as a guest thread the kernel keeps, until
//!   [`Server::start_next`] names its session. The rules are the service spec's: one request in
//!   flight per session, one worker per server, and a deferred request gives its worker up
//!   ([`queue`]).
//! - A method that deferred makes [`Server::handle`] return [`Handled::Deferred`]. When its
//!   token completes, on whatever thread, the server's [`Notify`] is told, and the kernel has
//!   the reply written with [`Server::take_completed`] and [`Server::deliver`].
//! - [`Server::save`] and [`Server::restore`] write and read the server's state ([`snapshot`]).
//!
//! The kernel is reached through the small [`Kernel`] trait, defined here until that slice
//! implements it; `fake::FakeKernel` (behind the `test-support` feature) stands in for tests.
//!
//! The message is parsed here and not by the kernel, because how a request is framed depends on
//! whether its session is a domain, which only the server knows.
//!
//! The kernel spec's sketch of a server row keeps the list of ready sessions, the running count
//! and the worker count in the kernel. Here they are in the server's [`RequestQueue`], which is
//! plain data and is saved with the server; the kernel keeps what only it has, the waiting
//! threads and their message buffers. Which of the two holds the queue once slice 9 exists is
//! that slice's to settle: the type is public so that it can be either.
//!
//! # What the framework answers itself
//!
//! - **Control requests** (convert to domain, copy from domain, clone, query pointer buffer
//!   size): see [`control`].
//! - **Domains.** A converted session holds a table of objects. A request names its object in
//!   the domain header; input object ids reach the method through [`Request::in_object`]; a new
//!   interface pushed into a reply becomes an object id; and a close-object request empties the
//!   id. Outside a domain, a pushed interface becomes a new session and a moved handle.
//! - **Unknown commands**, by the server's [`UnknownPolicy`]. Strict logs and replies
//!   [`UNKNOWN_COMMAND_ID`]. Auto-stub logs each command once and replies success with nothing
//!   after the result word but zeros: the message buffer is zeroed before a reply is written
//!   into it, so no byte of the request survives (`eden-nx2/NX2_NOTES.md`, row 96, where a stub
//!   that kept the request's words made a title abort). [`Server::stubbed`] lists what was
//!   stubbed.
//! - **Close** messages: the session is dropped from the server and nothing is written.
//! - **Tracing.** Every request the framework answers is one event in the run's event stream,
//!   marked when it was stubbed or deferred: see [`trace`].
//!
//! A method's [`Fault`], and the cases in the table below, are answered with a reply that holds
//! only a result code. The codes are in [`result`]; most are guesses, and say so there.
//!
//! # Where this differs from Eden
//!
//! Dispatch is ported from Eden's `ServiceFrameworkBase::HandleSyncRequest`
//! (`eden/src/core/hle/service/service.cpp`), `SessionRequestManager` and `HLERequestContext`
//! (`eden/src/core/hle/service/hle_ipc.cpp`, `hle_ipc.h`), `ResponseBuilder` and `RequestParser`
//! (`eden/src/core/hle/service/ipc_helpers.h`) and `Controller`
//! (`eden/src/core/hle/service/sm/sm_controller.cpp`), all GPL-3.0-or-later, at the commit pinned
//! in `docs/references.md` (`cf654052`). What a client expects of control requests and of closing
//! is checked against libnx (`libnx/nx/include/switch/sf/cmif.h` and `service.h`, ISC).
//!
//! Eden's assertions log and carry on (`eden/src/common/assert.h`, `ASSERT_MSG`). Where the
//! table says Eden builds or writes no reply, the caller's buffer still holds its own request
//! when the caller wakes, and that is what it reads as the reply. Here each case is an answer.
//! Every row is pinned by a test under `framework/tests/`:
//!
//! | Case | Eden | Here |
//! |---|---|---|
//! | Unknown command, auto-stub off | `ReportUnimplementedFunction` logs a failed assertion and builds no reply | reply [`UNKNOWN_COMMAND_ID`] |
//! | Domain request to an object id past the table, or to a closed one | `SessionRequestManager::CompleteSyncRequest` logs "Session handler is invalid"; nothing is written to the caller's buffer | reply [`TARGET_NOT_FOUND`] |
//! | Domain request to object id zero | `object_id - 1` wraps and indexes the handler vector out of range | reply [`TARGET_NOT_FOUND`] |
//! | Closing a domain object | `HandleDomainSyncRequest` clears the handler and builds a success reply that is never written to the caller's buffer (only `ServiceFrameworkBase::HandleSyncRequest` writes replies, and a close does not pass through it); an id that names nothing takes the "Session handler is invalid" path | the object is dropped if the id names one, and a success reply is written either way |
//! | Message type legacy request or legacy control | `HandleSyncRequest` logs a failed assertion and builds no reply | reply [`NOT_SUPPORTED`] |
//! | TIPC request on a session that is a domain | dispatched to the handler the session had before it was converted | reply [`NOT_SUPPORTED`] |
//! | Close and TIPC close | writes a success reply, then reports the session closed | writes nothing; [`Handled::Closed`] |
//! | A reply too large for the message buffer | `ResponseBuilder` pushes words into its 64-word buffer without a bounds check | reply [`NOT_SUPPORTED`], and nothing the method returned is kept |
//! | A full domain | the handler vector grows without limit | reply [`OUT_OF_DOMAIN_ENTRIES`] at [`MAX_DOMAIN_OBJECTS`] |
//!
//! Eden has no reply tokens and no family state, so three more answers are starquake's own and
//! not divergences. Each is pinned by the test named, in `framework/tests/`:
//!
//! | Case | Here |
//! |---|---|
//! | A reply token is dropped without being completed, or a method returns "deferred" without taking one | reply [`DEFERRED_REPLY_DROPPED`] (`deferred.rs`: `a_dropped_token_replies_with_an_error`, `deferring_without_a_token_is_answered_at_once_with_an_error`) |
//! | A method asks for a second reply token | reply [`NOT_SUPPORTED`] (`deferred.rs`: `a_request_has_one_token`) |
//! | A method asks for family state its server was not given | reply [`NOT_SUPPORTED`] (`server.rs`: `a_method_that_needs_family_state_fails_on_a_server_without_it`) |
//!
//! More are listed where they belong: payload layout in [`payload`], buffers in [`request`],
//! control requests in [`control`], and the reuse of domain object ids on `Domain` in
//! `framework/targets.rs`.
//!
//! Two of the rows rest on guesses about a console, which no pinned reference confirms and a
//! trace from one would settle:
//!
//! - That a close message gets no reply (the server closes its end, and the caller's request
//!   ends with the kernel's session-closed result).
//! - That closing a domain object replies success whether or not the id named an object.
//!
//! Neither can upset a libnx client: `serviceClose` sends either kind of close and looks at
//! neither the system call's result nor the buffer afterwards
//! (`libnx/nx/include/switch/sf/service.h`).
//!
//! # Not here yet
//!
//! - Somewhere for a method to send a reply token. A session and its family are plain data, so
//!   neither can hold a worker or a channel; the service spec's answer is the injectable
//!   environment (`docs/specs/observability.md`, rule 1), which [`Context`] does not carry
//!   yet. The first service that defers (`fsp-srv`, slice 6) brings it. The tests here park
//!   their tokens in a thread-local.
//! - A deferred reply that writes into the caller's buffers: see [`defer`].
//! - Servers with more than one worker. [`RequestQueue`] counts workers, but [`Server::handle`]
//!   takes the server exclusively, so a `Server` always has [`DEFAULT_WORKERS`]. Opting in
//!   (service spec, "Opting in to parallelism") needs handlers that can run side by side,
//!   which is a design of its own.
//! - State shared between servers (the service spec's `Arc<Settings>`). A family's state
//!   belongs to one server and is saved with it; nothing yet saves one piece of state once for
//!   two servers and re-links it.
//! - Moved handles a caller sends are resolved but not removed from its table, as in Eden's HLE;
//!   removing them is handle transfer, which the kernel spec gives to the kernel.
//! - A domain reply that returns objects after a payload that is not a whole number of words.
//!   A client looks for the object ids straight after the payload, at its exact size (libnx,
//!   `cmifParseResponse`: `objects = hdr + sizeof(CmifOutHeader) + size`); `starquake-ipc`'s reply
//!   builder pads the payload to a whole word first. Until it can place the ids where the client
//!   looks, such a reply is refused with [`NOT_SUPPORTED`] and an error in the log, and nothing
//!   of it is kept, rather than sent with its ids one to three bytes late.

mod commands;
mod context;
pub mod control;
pub mod defer;
#[cfg(any(test, feature = "test-support"))]
pub mod fake;
mod host;
mod kernel;
pub mod payload;
pub use starquake_kernel::ipc::queue;
mod reply;
pub mod request;
pub mod result;
mod server;
pub mod snapshot;
mod targets;
pub mod trace;

#[cfg(test)]
mod tests;

pub use commands::{Command, CommandTable, Protocol, Session};
pub use context::Context;
pub use defer::{Completed, Notify, ReplyToken, TokenId};
pub use kernel::Kernel;
pub use payload::{Reader, Writer};
pub use queue::{Admission, DEFAULT_WORKERS, QueueError, RequestQueue};
pub use reply::Reply;
pub use request::{InBuffer, OutBuffer, Request};
pub use result::{
    Completion, DEFERRED_REPLY_DROPPED, Direction, Fault, HandleKind, NOT_SUPPORTED,
    OUT_OF_DOMAIN_ENTRIES, Outcome, TARGET_NOT_FOUND, UNKNOWN_COMMAND_ID,
};
pub use server::{
    DEFAULT_POINTER_BUFFER_SIZE, Handled, MAX_STUBBED_COMMANDS, Server, ServerError,
    StubbedCommand, UnknownPolicy,
};
pub use snapshot::{Family, Interfaces, SNAPSHOT_MAGIC, SNAPSHOT_VERSION, SnapshotError};
pub use targets::{MAX_DOMAIN_OBJECTS, TargetError};
pub use trace::NO_COMMAND_ID;
