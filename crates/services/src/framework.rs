//! The service framework: what every HLE service is written against
//! (`docs/specs/services.md`, port plan slices 1 and 2).
//!
//! # Writing a service
//!
//! A service is a struct per session, a command table and one method per command:
//!
//! ```
//! use starquake_services::commands;
//! use starquake_services::framework::{Context, Outcome, Reply, Request};
//!
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
//! Nothing here enforces it yet (see "Not here yet").
//!
//! - [`commands!`](crate::commands) implements [`Session`] for the struct: the dispatch `match`
//!   and the table of ids and names.
//! - [`Request`] gives the command id, the raw payload through a [`Reader`], buffers by index
//!   ([`InBuffer`], [`OutBuffer`]), the caller's process id when it was sent, handles resolved to
//!   kernel objects, and a domain request's input objects.
//! - [`Reply`] takes a raw payload through a [`Writer`], handles to copy or move, and new
//!   interfaces. A method ends with `reply.ok()`, or fails with a [`Fault`]: `?` on any framework
//!   call, or `Err(code.into())` with the service's own result code.
//! - [`Context`] is the rest of what a method may touch: guest memory by copy, and events.
//!
//! # Running one
//!
//! A [`Server`] holds one service family's sessions. The kernel's session code (kernel spec, port
//! plan slice 9) owns queueing and calls [`Server::handle`] with the caller's message buffer, one
//! request at a time per server; the reply is written over the request. The kernel is reached
//! through the small [`Kernel`] trait, defined here until that slice implements it;
//! `fake::FakeKernel` (behind the `test-support` feature) stands in for tests.
//!
//! The message is parsed here and not by the kernel, because how a request is framed depends on
//! whether its session is a domain, which only the server knows.
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
//! - Snapshots (`docs/specs/observability.md`, "Snapshots"). A [`Server`] already refers to its
//!   interfaces by `TargetId` and its sessions by [`ObjectId`](starquake_kernel::id::ObjectId),
//!   so its tables can be saved and re-linked by id. What is missing is a save and restore pair
//!   for the `Box<dyn Session>` behind each table row: [`Session`] gives a snapshot no way to
//!   reach a session's state. [`commands!`](crate::commands) is the natural place to require
//!   that pair, since every session goes through it; that is a guess about where it fits, to be
//!   confirmed when the snapshot format is designed.
//! - Deferred replies and the per-request trace line are slice 3. Until then a method cannot
//!   return "deferred", and requests are logged through `tracing` only where something went
//!   wrong.
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
#[cfg(any(test, feature = "test-support"))]
pub mod fake;
mod kernel;
pub mod payload;
mod reply;
pub mod request;
pub mod result;
mod server;
mod targets;

#[cfg(test)]
mod tests;

pub use commands::{Command, CommandTable, Protocol, Session};
pub use context::Context;
pub use kernel::Kernel;
pub use payload::{Reader, Writer};
pub use reply::Reply;
pub use request::{InBuffer, OutBuffer, Request};
pub use result::{
    Completion, Direction, Fault, HandleKind, NOT_SUPPORTED, OUT_OF_DOMAIN_ENTRIES, Outcome,
    TARGET_NOT_FOUND, UNKNOWN_COMMAND_ID,
};
pub use server::{
    DEFAULT_POINTER_BUFFER_SIZE, Handled, MAX_STUBBED_COMMANDS, Server, ServerError,
    StubbedCommand, UnknownPolicy,
};
pub use targets::{MAX_DOMAIN_OBJECTS, TargetError};
