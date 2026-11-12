//! The trace sink: one event in the run's event stream for every request the framework answers
//! (`docs/specs/services.md`, "Tracing"; `docs/specs/observability.md`, "The event stream": "the
//! service framework's dispatch ... calls the recorder once").
//!
//! The event is [`EventKind::Ipc`], whose text form is the golden-trace line
//! (`docs/specs/testing.md`, "Golden traces"). It is recorded when the reply is written, through
//! [`Kernel::record`](super::Kernel::record), so the kernel stamps it with the caller's thread
//! and core and numbers it in its own order. What each field holds:
//!
//! | Field | Holds |
//! |---|---|
//! | `session` | the session the request came on ([`session_id`]) |
//! | `service` | the server's number ([`Server::with_service_id`](super::Server::with_service_id)) |
//! | `command` | the command id; [`NO_COMMAND_ID`] for a domain's close-object request, which has none |
//! | `request_size` | bytes of the command's arguments: the raw payload a method reads |
//! | `reply_size` | bytes of payload in the reply: what a method wrote, none on a failure |
//! | `result` | the reply's result code |
//! | `outcome` | `Stubbed` for an unknown command answered by auto-stub, `Deferred` for a request answered through a reply token, otherwise `Done` |
//!
//! Exactly which messages produce an event: every message [`Server::handle`](super::Server::handle)
//! answers with [`Handled::Replied`](super::Handled::Replied), when it answers; and every
//! deferred request, when [`Server::deliver`](super::Server::deliver) writes its reply. A
//! deferred request therefore has no event until it is answered, and the order of those events
//! is the order deferred replies completed in. A close message and a message the server fails
//! on ([`ServerError`](super::ServerError)) get no reply and no event.
//!
//! The sizes are payload sizes, as the service spec says ("payload sizes"); the event type's own
//! docs call them message sizes. The event has no field for the command's name, the protocol, or
//! whether the request was a control request, so a CMIF command, the TIPC command and the
//! control command with one id read the same in a trace. Names are in the command tables
//! ([`Session::commands`](super::Session::commands),
//! [`CONTROL_COMMANDS`](super::control::CONTROL_COMMANDS)). Both are gaps in the event type,
//! which belongs to `starquake-trace`.

use starquake_ipc::{Body, CommandType, Message};
use starquake_kernel::id::{ObjectId, SlabId};
use starquake_kernel::result::ResultCode;
use starquake_trace::{EventKind, IpcOutcome, ServiceId, SessionId};

/// The `command` of an event for a message that has no command id: a domain's close-object
/// request.
pub const NO_COMMAND_ID: u32 = u32::MAX;

/// The event stream's name for a session: the session object's slot index in the low half and
/// its generation in the high half.
///
/// This is the one place a kernel object id becomes a trace [`SessionId`]. The packing is
/// starquake's own (the event type calls the id "a plain `u64`" and leaves its contents open);
/// the kernel's own events must name a session the same way, by calling this.
#[must_use]
pub fn session_id(session: ObjectId) -> SessionId {
    SessionId((u64::from(session.generation()) << u32::BITS) | u64::from(session.index()))
}

/// What a request's event says of the request itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Asked {
    pub(super) command: u32,
    pub(super) request_size: u32,
}

impl Asked {
    /// The command id and argument size of a parsed request.
    pub(super) fn of(message: &Message<'_>) -> Self {
        let (command, data) = match (&message.body, message.command_type) {
            (Body::CmifRequest(body), _) => (body.header.command_id, body.data),
            (Body::TipcRequest { data }, CommandType::Tipc(id)) => (u32::from(id), *data),
            _ => (NO_COMMAND_ID, &[][..]),
        };
        Self {
            command,
            request_size: size(data.len()),
        }
    }
}

/// A byte count as an event holds it. No message is anywhere near `u32::MAX` bytes.
pub(super) fn size(len: usize) -> u32 {
    u32::try_from(len).unwrap_or(u32::MAX)
}

/// The event for one answered request.
pub(super) fn event(
    service: ServiceId,
    session: ObjectId,
    asked: Asked,
    reply_size: u32,
    result: ResultCode,
    outcome: IpcOutcome,
) -> EventKind {
    EventKind::Ipc {
        session: session_id(session),
        service,
        command: asked.command,
        request_size: asked.request_size,
        reply_size,
        // The one place a service result becomes the event stream's result word.
        result: starquake_trace::ResultCode(result.raw()),
        outcome,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_is_named_by_its_index_and_generation() {
        let session = ObjectId::from_parts(0x1234_5678, 0x9ABC_DEF0);
        assert_eq!(session_id(session), SessionId(0x9ABC_DEF0_1234_5678));
        assert_eq!(
            session_id(ObjectId::from_parts(3, 1)),
            SessionId(0x1_0000_0003)
        );
    }

    #[test]
    fn a_size_past_a_word_is_recorded_as_the_largest_word() {
        assert_eq!(size(0), 0);
        assert_eq!(size(0x100), 0x100);
        #[cfg(target_pointer_width = "64")]
        {
            let word = usize::try_from(u32::MAX).unwrap();
            assert_eq!(size(word), u32::MAX);
            assert_eq!(size(word + 1), u32::MAX);
        }
    }
}
