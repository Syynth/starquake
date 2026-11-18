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
//! | `interface` | where the request went (below) |
//! | `protocol` | `Control` for a CMIF control request, `Tipc` for a TIPC request, otherwise `Cmif` (a domain's close-object request included) |
//! | `command` | the command id; [`NO_COMMAND_ID`] for a domain's close-object request, which has none |
//! | `request_size` | bytes of the command's arguments: the raw payload a method reads |
//! | `reply_size` | bytes of payload in the reply: what a method wrote, none on a failure |
//! | `result` | the reply's result code |
//! | `outcome` | `Stubbed` for an unknown command answered by auto-stub, `Deferred` for a request answered through a reply token, otherwise `Done` |
//!
//! The `interface` is the [`Session::trace_name`] of the interface the request is addressed to,
//! as [`InterfaceName::lossy`] keeps it:
//!
//! - a control request: [`CONTROL_INTERFACE`], the name the reference gives the control
//!   commands, whatever the session;
//! - a domain request (a close-object request too): the object its domain header names;
//! - any other request: the session's own interface.
//!
//! When that names no interface (an object id that is empty, or a request with no domain header
//! on a domain), it is the session's own interface: for a domain, object
//! `Domain::ROOT_OBJECT_ID`, the interface the session had before it became a domain. When that
//! is gone too, it is [`InterfaceName::lossy`]'s name for nothing, one
//! [`REPLACEMENT`](starquake_trace::REPLACEMENT). A deferred request keeps the name it had
//! when it was deferred, and its protocol follows from how its reply is framed (control
//! requests are never deferred).

//! Exactly which messages produce an event: every message [`Server::handle`](super::Server::handle)
//! answers with [`Handled::Replied`](super::Handled::Replied), when it answers; and every
//! deferred request, when [`Server::deliver`](super::Server::deliver) writes its reply. A
//! deferred request therefore has no event until it is answered, and the order of those events
//! is the order deferred replies completed in. A close message and a message the server fails
//! on ([`ServerError`](super::ServerError)) get no reply and no event.
//!
//! The sizes are payload sizes, as the service spec says ("payload sizes"); the event type's own
//! docs call them message sizes. The event has no field for the command's name: names are in
//! the command tables ([`Session::commands`](super::Session::commands),
//! [`CONTROL_COMMANDS`]), and the server counts each request under its name
//! ([`Server::requests`](super::Server::requests)).

use starquake_ipc::{Body, CmifRequest, CommandType, Message};
use starquake_kernel::id::{ObjectId, SlabId};
use starquake_kernel::result::ResultCode;
use starquake_trace::{EventKind, InterfaceName, IpcOutcome, IpcProtocol, ServiceId, SessionId};

use super::commands::{Protocol, Session};
use super::control::{CONTROL_COMMANDS, CONTROL_INTERFACE};
use super::defer::Framing;
use super::targets::{Domain, TargetId, Targets};

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

/// What a request's event says of the request itself, and the name it is counted under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Asked {
    pub(super) interface: InterfaceName,
    pub(super) protocol: IpcProtocol,
    pub(super) command: u32,
    pub(super) request_size: u32,
    /// The command's name in its table, if the table has it: what
    /// [`Server::requests`](super::Server::requests) counts the request under. Not in the event.
    pub(super) name: Option<&'static str>,
}

impl Asked {
    /// What the event of a parsed request on a session that leads to `target` says: see the
    /// [module docs](self) for the interface and the protocol.
    pub(super) fn of(message: &Message<'_>, targets: &Targets, target: TargetId) -> Self {
        let (command, data) = match (&message.body, message.command_type) {
            (Body::CmifRequest(body), _) => (body.header.command_id, body.data),
            (Body::TipcRequest { data }, CommandType::Tipc(id)) => (u32::from(id), *data),
            _ => (NO_COMMAND_ID, &[][..]),
        };
        let protocol = match message.command_type {
            CommandType::Control | CommandType::ControlWithContext => IpcProtocol::Control,
            CommandType::Tipc(_) => IpcProtocol::Tipc,
            _ => IpcProtocol::Cmif,
        };
        let (interface, name) = match protocol {
            IpcProtocol::Control => (
                InterfaceName::lossy(CONTROL_INTERFACE),
                CONTROL_COMMANDS.name(Protocol::Cmif, command),
            ),
            IpcProtocol::Cmif | IpcProtocol::Tipc => {
                let addressed =
                    addressee(message, targets, target).or_else(|| own_interface(targets, target));
                let table = if protocol == IpcProtocol::Tipc {
                    Protocol::Tipc
                } else {
                    Protocol::Cmif
                };
                match addressed {
                    Some(session) => (
                        InterfaceName::lossy(session.trace_name()),
                        session.commands().name(table, command),
                    ),
                    None => (InterfaceName::lossy(""), None),
                }
            }
        };
        Self {
            interface,
            protocol,
            command,
            request_size: size(data.len()),
            name,
        }
    }
}

/// The interface a CMIF or TIPC request is addressed to: the object a domain header (or a
/// close-object request) names, or else the session's own; `None` if that is no interface.
fn addressee<'t>(
    message: &Message<'_>,
    targets: &'t Targets,
    target: TargetId,
) -> Option<&'t dyn Session> {
    let object_id = match &message.body {
        Body::CmifRequest(CmifRequest {
            domain: Some(header),
            ..
        })
        | Body::DomainClose(header) => Some(header.object_id),
        _ => None,
    };
    let object = match object_id {
        Some(object_id) => targets.domain(target).ok()?.get(object_id)?,
        None => target,
    };
    targets.interface(object)
}

/// The session's own interface: the one it leads to, or for a domain its object
/// [`Domain::ROOT_OBJECT_ID`].
fn own_interface(targets: &Targets, target: TargetId) -> Option<&dyn Session> {
    let own = match targets.domain(target) {
        Ok(domain) => domain.get(Domain::ROOT_OBJECT_ID)?,
        Err(_) => target,
    };
    targets.interface(own)
}

/// The protocol of a deferred request, from how its reply is framed: control requests are
/// answered at once, so a deferred one is CMIF or TIPC.
pub(super) fn deferred_protocol(framing: Framing) -> IpcProtocol {
    match framing {
        Framing::Cmif | Framing::Domain => IpcProtocol::Cmif,
        Framing::Tipc(_) => IpcProtocol::Tipc,
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
        interface: asked.interface,
        protocol: asked.protocol,
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
