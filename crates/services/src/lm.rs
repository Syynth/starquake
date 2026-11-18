//! `lm`, the log manager: a logger whose messages go to the run's guest log.
//!
//! Ported from `eden/src/core/hle/service/lm/lm.cpp` @ cf654052 (GPL-3.0-or-later): `LM`
//! (`OpenLogger`), `ILogger` (`Log`, `SetDestination`), `LogDestination` and `LoopProcess`.
//! Eden-NX2 @ 50fa86bb and Eden PR #4588 @ e5fb3b97 have the same file, apart from
//! `std::format` spelled `fmt::format`, so Switch 2 changes nothing here; the Switch 2 scoping
//! document (`docs/switch2-eden-diff-scoping.md`) has no `lm` row. The request layouts are
//! checked against the console's interface as Atmosphère declares it
//! (`atmosphere/libraries/libstratosphere/source/lm/sf/lm_i_log_service.hpp`, GPL-2.0-only, fact
//! only): `Log` takes `sf::InAutoSelectBuffer`, `SetDestination` a `u32`, and `OpenLogger` a
//! `sf::ClientProcessId` (a `u64` in the payload, with the process id sent).
//!
//! # Commands
//!
//! | Interface | Id | Command | In | Out |
//! |---|---|---|---|---|
//! | `lm` ([`Lm`]) | 0 | `OpenLogger` | the client process id (not read) | an [`ILogger`] |
//! | [`ILogger`] | 0 | `Log` | in buffer 0 (auto-select: an A or an X descriptor): one log packet | nothing |
//! | [`ILogger`] | 1 | `SetDestination` | `u32` destination | nothing |
//!
//! `ILogger`'s `TransmitHashedLog` (2) and `DevNotify` (3), which Eden lists without a handler,
//! go to the server's unknown-command policy, and so does every command of `lm:get`, which is
//! not registered here.
//!
//! # Where the messages go
//!
//! `Log` hands its packet to the logger's [`Pending`] messages ([`packet`]); when a tail packet
//! finishes a message, its text chunk ([`LogRecord::text`]) is one line of the run's guest log,
//! through the [`GuestLog`] the server was given ([`server`]). In a run that is the guest log
//! `OutputDebugString` writes to, under the same limit (`guest_log.txt`), so the two kinds of
//! line are in the order the guest wrote them. A message with no text chunk adds no line. Eden
//! instead prints every chunk it knows, with the process and thread ids, to its own debug log
//! (`ParseLog`, `LOG_DEBUG`); that is the emulator's log, not the guest's output, so the other
//! chunks are read and checked here but not written anywhere.
//!
//! A line is not a guest-log event in the trace, unlike an `OutputDebugString` line: the `Log`
//! request is already one IPC event there, and the service has no core or thread to stamp a
//! second event with. So a replay compares that a `Log` request was made, not its text.
//!
//! The destination `SetDestination` sets is kept ([`ILogger::destination`]) and changes
//! nothing, as in Eden, where it only names the destination in the debug log. A new logger's is
//! [`DESTINATION_ALL`] (`LogDestination destination{LogDestination::All}`).
//!
//! # Where this differs from Eden
//!
//! - **A `Log` request without both auto-select descriptors is refused.** The framework holds a
//!   request's descriptors against the row before `Log` runs ([`crate::framework::buffers`],
//!   "The check"): `Log` declares one auto-select in buffer, so a request must carry exactly one
//!   A and one X descriptor (one of them may be empty), or it is answered
//!   [`INVALID_CMIF_REQUEST`](crate::framework::result::INVALID_CMIF_REQUEST) and `Log` is not
//!   called. Eden checks no descriptor count and answers such a request with success, reading
//!   whichever descriptor is there or an empty buffer (`ctx.ReadBuffer()`). libnx's client
//!   sends both descriptors for an auto-select buffer, so only an unusual client sees this
//!   (`lm/tests.rs`: `a_log_request_without_both_descriptors_is_refused`).
//! - **A packet that cannot be read is dropped, and logged here.** Eden answers `Log` with
//!   success before it reads the packet ("This function only succeeds"), and so does this port
//!   for every request that passes the descriptor check above: one whose buffer cannot be read,
//!   or is more than [`MAX_PACKET_BYTES`], and one [`packet`] refuses. The difference is in what is read: [`packet`]'s module docs
//!   list where its checks are stricter than Eden's (`lm/tests.rs`:
//!   `a_malformed_packet_is_answered_with_success_and_adds_no_line`,
//!   `a_packet_larger_than_the_limit_is_dropped`). Eden reads a buffer of any size;
//!   [`MAX_PACKET_BYTES`] is starquake's own bound, a header and the largest message.
//! - **The messages in progress are the logger's own**, as Eden's `entries` are per
//!   `ILogger`, and are saved with its session.
//!
//! # Locks
//!
//! [`GuestLog::line`] is called while `Log`'s handler runs: in a run, on the guest thread that
//! sent the request, with the kernel's lock held, as `OutputDebugString`'s sink is. It must not
//! call into the kernel or a service.

pub mod packet;

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::commands;
use crate::framework::{
    Context, Family, Interfaces, Outcome, Reply, Request, Server, SnapshotError, UnknownPolicy,
};
use packet::{HEADER_SIZE, Header, LogRecord, MAX_MESSAGE_BYTES, PacketError, Pending};

#[cfg(test)]
mod tests;

/// The name `lm` is registered under (`eden/src/core/hle/service/lm/lm.cpp`, `LoopProcess`:
/// `RegisterNamedService("lm", ..)`).
pub const NAME: &str = "lm";

/// `LogDestination::All` (`All = 0xffff`): a new logger's destination.
pub const DESTINATION_ALL: u32 = 0xffff;

/// The most bytes `Log` reads from its buffer: a header and the largest message
/// ([`MAX_MESSAGE_BYTES`]). A guess of starquake's own (module docs, "Where this differs from
/// Eden").
pub const MAX_PACKET_BYTES: usize = HEADER_SIZE + MAX_MESSAGE_BYTES;

/// Where finished messages' text goes: the run's guest log, in a run (module docs, "Where the
/// messages go"). See the module docs, "Locks", for when it is called.
pub trait GuestLog: Send + Sync {
    /// Takes one line the guest logged: the bytes of a message's text chunk, not necessarily
    /// UTF-8.
    fn line(&self, text: &[u8]);
}

/// The `lm` server's family state: the guest log its loggers write to. It is the run's wiring,
/// not the guest's state, so a snapshot saves nothing of it and a restore keeps it.
pub struct Loggers {
    guest_log: Arc<dyn GuestLog>,
}

impl Family for Loggers {
    fn save_state(&self) -> Result<Vec<u8>, SnapshotError> {
        Ok(Vec::new())
    }

    fn load_state(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        if bytes.is_empty() {
            Ok(())
        } else {
            Err(SnapshotError::TrailingBytes { count: bytes.len() })
        }
    }
}

/// The root interface of an `lm` session (`LM`). It keeps nothing.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Lm;

/// A logger (`ILogger`): what `OpenLogger` hands out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ILogger {
    /// What `SetDestination` last set (module docs, "Where the messages go").
    destination: u32,
    /// The messages in progress.
    pending: Pending,
}

impl Default for ILogger {
    fn default() -> Self {
        Self::new()
    }
}

impl ILogger {
    /// A new logger: destination [`DESTINATION_ALL`], no message in progress.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            destination: DESTINATION_ALL,
            pending: Pending::new(),
        }
    }

    /// The destination `SetDestination` last set, or [`DESTINATION_ALL`].
    #[must_use]
    pub const fn destination(&self) -> u32 {
        self.destination
    }

    /// The messages in progress.
    #[must_use]
    pub const fn pending(&self) -> &Pending {
        &self.pending
    }
}

/// The ids of the commands ported here, per interface (Eden's tables).
pub mod command {
    /// `lm`'s commands (`LM::LM`).
    pub mod lm {
        /// `OpenLogger` (`{0, &LM::OpenLogger, "OpenLogger"}`).
        pub const OPEN_LOGGER: u32 = 0;
    }

    /// `ILogger`'s commands (`ILogger::ILogger`).
    pub mod logger {
        /// `Log` (`{0, &ILogger::Log, "Log"}`).
        pub const LOG: u32 = 0;
        /// `SetDestination` (`{1, &ILogger::SetDestination, "SetDestination"}`).
        pub const SET_DESTINATION: u32 = 1;
    }
}

commands! {
    impl Lm {
        command::lm::OPEN_LOGGER => open_logger,
    }
}

commands! {
    impl ILogger {
        command::logger::LOG => log [In(AutoSelect)],
        command::logger::SET_DESTINATION => set_destination,
    }
}

impl Lm {
    /// `OpenLogger`. in: the client process id, not read (Eden reads nothing). out: an
    /// [`ILogger`].
    fn open_logger(
        _: &mut Self,
        _cx: &mut Context<'_>,
        _req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        reply.push_session(Box::new(ILogger::new()));
        reply.ok()
    }
}

impl ILogger {
    /// `Log`. in buffer 0: one packet. Success for every request that reaches it; a request
    /// without both auto-select descriptors is refused before it runs (module docs, "Where this
    /// differs from Eden").
    fn log(&mut self, cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
        let packet = match req
            .in_buffer(cx, 0)
            .and_then(|mut buffer| buffer.read_all(MAX_PACKET_BYTES))
        {
            Ok(packet) => packet,
            Err(fault) => {
                tracing::warn!(%fault, "lm: a log packet could not be read, and is dropped");
                return reply.ok();
            }
        };
        match self.take(&packet) {
            Ok(Some(text)) => cx.family::<Loggers>()?.guest_log.line(&text),
            Ok(None) => {}
            Err(error) => {
                tracing::warn!(%error, "lm: a log packet was refused, and is dropped");
            }
        }
        reply.ok()
    }

    /// Adds `packet` to its message, and returns the message's text when the packet finished
    /// one that has text.
    fn take(&mut self, packet: &[u8]) -> Result<Option<Vec<u8>>, PacketError> {
        let (header, payload) = Header::parse(packet)?;
        let Some(message) = self.pending.push(&header, payload)? else {
            return Ok(None);
        };
        Ok(LogRecord::parse(&message)?.text.map(<[u8]>::to_vec))
    }

    /// `SetDestination`. in: `u32` destination.
    fn set_destination(
        &mut self,
        _cx: &mut Context<'_>,
        req: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        self.destination = req.payload().read_u32()?;
        reply.ok()
    }
}

/// The interfaces an `lm` server's sessions can be.
#[must_use]
pub fn interfaces() -> Interfaces {
    Interfaces::new().with::<Lm>().with::<ILogger>()
}

/// The `lm` server, whose loggers write finished messages' text to `guest_log`.
#[must_use]
pub fn server(policy: UnknownPolicy, guest_log: Arc<dyn GuestLog>) -> Server {
    Server::new(NAME, policy)
        .with_root(|| Box::new(Lm))
        .with_interfaces(interfaces())
        .with_family(Loggers { guest_log })
}
