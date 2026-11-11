//! The event type: what one crossing of a boundary we own looks like.
//!
//! Source: `docs/specs/observability.md`, "The event stream". The spec names the fields
//! (`seq`, `time`, `core`, `thread`, `kind`) and the shape of each [`EventKind`] variant but not
//! their types, so every field type below is **a guess** made for this crate, chosen so that an
//! [`Event`] is `Copy` and fixed-size (the flight recorder must not allocate while recording).
//! Payloads that are not fixed-size (service-note text, guest log text, IPC bytes) belong to the
//! full binary log, which is out of scope here; the flight recorder keeps "only sizes and hashes"
//! (spec, "Replaying the kernel and services", rule 2), which is what [`EventKind::GuestLog`] and
//! [`EventKind::Ipc`] carry.
//!
//! No reference source is ported here: the layout is this project's own design.

use core::fmt;

use serde::{Deserialize, Serialize};

/// Number of system-call argument registers an [`EventKind::Svc`] event carries.
///
/// Guess: eight, matching the AArch64 argument registers `x0`..`x7`.
pub const SVC_ARG_COUNT: usize = 8;

/// A guest thread, as numbered by the kernel (guess: a plain `u64`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ThreadId(pub u64);

/// An IPC session, as numbered by the kernel (guess: a plain `u64`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SessionId(pub u64);

/// A service, as numbered by the service framework (guess: a plain `u32` index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ServiceId(pub u32);

/// A raw guest result code, recorded as the guest saw it (not decoded).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResultCode(pub u32);

/// A point in guest time. The unit is whatever the producing clock uses (guess: the recorder does
/// not interpret it); it is carried for the flight recorder and removed from the golden-trace
/// view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GuestTicks(pub u64);

/// How the service framework disposed of an IPC request (spec: "stubbed / deferred / done").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IpcOutcome {
    /// The command was auto-stubbed because it is not implemented.
    Stubbed,
    /// The reply is deferred and will complete later.
    Deferred,
    /// The command ran and replied.
    Done,
}

/// Why a thread was woken (guess: the spec only names the field).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WakeReason {
    /// The object or address it waited on was signalled.
    Signalled,
    /// Its wait timed out.
    TimedOut,
    /// The wait was cancelled.
    Cancelled,
    /// The thread was terminated while waiting.
    Terminated,
}

/// What an address-space mutation did (guess: the spec only names the field `kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MapKind {
    /// A range was mapped.
    Map,
    /// A range was unmapped.
    Unmap,
    /// A range's permissions changed.
    Protect,
}

/// What was done about a guest memory fault (guess: the spec only names the field `action`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FaultAction {
    /// The fault was resolved and the guest resumed.
    Handled,
    /// The fault was delivered to the guest as an exception.
    Delivered,
    /// The fault ended the process.
    Fatal,
}

/// A lifecycle step (guess: the spec names the variant but not its contents).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LifecyclePhase {
    /// A process started.
    ProcessStart,
    /// A process exited.
    ProcessExit,
    /// A thread was created.
    ThreadCreate,
    /// A thread exited.
    ThreadExit,
    /// The emulator hit an internal error.
    EmulatorError,
}

/// What happened at the boundary (spec: `EventKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    /// A system call, with its decoded arguments.
    Svc {
        /// System-call number.
        id: u32,
        /// Argument registers `x0`..`x7` as the guest passed them.
        args: [u64; SVC_ARG_COUNT],
        /// The result code returned to the guest.
        result: ResultCode,
    },
    /// An IPC request handled by the service framework.
    Ipc {
        /// The session the request arrived on.
        session: SessionId,
        /// The service that owns the session.
        service: ServiceId,
        /// The command id.
        command: u32,
        /// Size in bytes of the request message.
        request_size: u32,
        /// Size in bytes of the reply message.
        reply_size: u32,
        /// The result code in the reply.
        result: ResultCode,
        /// How the request was disposed of.
        outcome: IpcOutcome,
    },
    /// A thread was woken.
    Wake {
        /// The thread that woke.
        thread: ThreadId,
        /// Why.
        reason: WakeReason,
    },
    /// The scheduler switched a core from one thread to another (`None` is the idle state).
    Schedule {
        /// The core that switched.
        core: u8,
        /// The thread that stopped running.
        from: Option<ThreadId>,
        /// The thread that started running.
        to: Option<ThreadId>,
    },
    /// An address-space mutation.
    Map {
        /// Start virtual address.
        va: u64,
        /// Size in bytes.
        size: u64,
        /// The guest's raw permission word, not decoded.
        perms: u32,
        /// What happened to the range.
        kind: MapKind,
    },
    /// A guest memory fault.
    Fault {
        /// The faulting address.
        address: u64,
        /// Whether the access was a write.
        write: bool,
        /// What was done.
        action: FaultAction,
    },
    /// A note from a service. Fixed-size by design; text lives in the full log.
    ServiceNote {
        /// The service that wrote it.
        service: ServiceId,
        /// A service-defined code.
        code: u32,
        /// A service-defined value.
        value: u64,
    },
    /// A line the guest logged. The flight recorder keeps the size and a hash, not the text.
    GuestLog {
        /// Length of the text in bytes.
        len: u32,
        /// A hash of the text, computed by the producer (this crate does not interpret it).
        hash: u64,
    },
    /// A lifecycle step.
    Lifecycle {
        /// Which step.
        phase: LifecyclePhase,
        /// The id of the process or thread the step applies to.
        subject: u64,
    },
}

impl EventKind {
    /// Whether the golden-trace view shows this event.
    ///
    /// Every kind except [`EventKind::Schedule`]: which core a thread ran on is a property of the
    /// run, not of what the guest did, and differs between runs with the same guest behaviour.
    /// This narrows the spec's "per-thread view of this stream" and is pinned by a test.
    #[must_use]
    pub fn in_golden_trace(&self) -> bool {
        !matches!(self, Self::Schedule { .. })
    }
}

impl fmt::Display for EventKind {
    /// The golden-trace rendering of one event body: one line, no timing, no sequence number.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Svc { id, args, result } => {
                // Trailing zero arguments are not shown: the stream does not record how many
                // arguments a call takes, so this is a display convention only.
                let shown = args.iter().rposition(|a| *a != 0).map_or(0, |i| i + 1);
                write!(f, "svc {id:#x}(")?;
                for (i, arg) in args.iter().take(shown).enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{arg:#x}")?;
                }
                write!(f, ") -> {:#x}", result.0)
            }
            Self::Ipc {
                session,
                service,
                command,
                request_size,
                reply_size,
                result,
                outcome,
            } => write!(
                f,
                "ipc session={} service={} command={command} req={request_size} rep={reply_size} \
                 -> {:#x} {outcome:?}",
                session.0, service.0, result.0
            ),
            Self::Wake { thread, reason } => write!(f, "wake thread={} {reason:?}", thread.0),
            Self::Schedule { core, from, to } => {
                let name =
                    |t: &Option<ThreadId>| t.map_or_else(|| "idle".into(), |t| t.0.to_string());
                write!(f, "schedule core={core} {} -> {}", name(from), name(to))
            }
            Self::Map {
                va,
                size,
                perms,
                kind,
            } => write!(f, "map {kind:?} va={va:#x} size={size:#x} perms={perms:#x}"),
            Self::Fault {
                address,
                write,
                action,
            } => write!(
                f,
                "fault address={address:#x} {} {action:?}",
                if *write { "write" } else { "read" }
            ),
            Self::ServiceNote {
                service,
                code,
                value,
            } => write!(f, "note service={} code={code} value={value:#x}", service.0),
            Self::GuestLog { len, hash } => write!(f, "log len={len} hash={hash:#x}"),
            Self::Lifecycle { phase, subject } => write!(f, "lifecycle {phase:?} {subject}"),
        }
    }
}

/// One structured event (spec: `Event`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    /// Position in the global order, assigned by the recorder; strictly increasing until the
    /// recorder's counter saturates (see [`Recorder::record`](crate::Recorder::record)).
    pub seq: u64,
    /// When it happened.
    pub time: GuestTicks,
    /// The core it happened on.
    pub core: u8,
    /// The guest thread it happened on.
    pub thread: ThreadId,
    /// What happened.
    pub kind: EventKind,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svc_display_trims_trailing_zero_args() {
        let kind = EventKind::Svc {
            id: 0x1f,
            args: [1, 0, 0x20, 0, 0, 0, 0, 0],
            result: ResultCode(0),
        };
        assert_eq!(kind.to_string(), "svc 0x1f(0x1, 0x0, 0x20) -> 0x0");
        let none = EventKind::Svc {
            id: 1,
            args: [0; SVC_ARG_COUNT],
            result: ResultCode(0xe401),
        };
        assert_eq!(none.to_string(), "svc 0x1() -> 0xe401");
    }

    #[test]
    fn only_schedule_is_hidden_from_the_golden_trace() {
        let schedule = EventKind::Schedule {
            core: 1,
            from: None,
            to: Some(ThreadId(2)),
        };
        assert!(!schedule.in_golden_trace());
        assert_eq!(schedule.to_string(), "schedule core=1 idle -> 2");
        let log = EventKind::GuestLog { len: 3, hash: 9 };
        assert!(log.in_golden_trace());
    }
}
