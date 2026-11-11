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
    /// Build it with [`EventKind::guest_log`].
    GuestLog {
        /// Length of the text in bytes.
        len: u32,
        /// The FNV-1a 64-bit hash of the text.
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

/// FNV-1a 64-bit offset basis (Fowler, Noll, Vo, "FNV Hash", the 64-bit parameters).
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

impl EventKind {
    /// Whether the golden-trace view shows this event.
    ///
    /// System calls, IPC requests, guest log lines, lifecycle steps and service notes are shown.
    /// [`EventKind::Wake`], [`EventKind::Map`], [`EventKind::Fault`] and [`EventKind::Schedule`]
    /// are left out: they depend on timing, so they differ between runs in which the guest did
    /// the same thing (observability spec, "The event stream"). The spec's list of what the view
    /// holds does not mention service notes either way; showing them is this crate's reading.
    #[must_use]
    pub fn in_golden_trace(&self) -> bool {
        !matches!(
            self,
            Self::Wake { .. } | Self::Map { .. } | Self::Fault { .. } | Self::Schedule { .. }
        )
    }

    /// The [`EventKind::GuestLog`] event for a line of `text` the guest logged: its length and
    /// its FNV-1a 64-bit hash. Every producer builds the event here, so one text always gives
    /// one trace line. A length above `u32::MAX` is recorded as `u32::MAX`.
    #[must_use]
    pub fn guest_log(text: &[u8]) -> Self {
        let hash = text.iter().fold(FNV_OFFSET_BASIS, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
        });
        Self::GuestLog {
            len: u32::try_from(text.len()).unwrap_or(u32::MAX),
            hash,
        }
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
    fn the_golden_trace_holds_calls_requests_logs_and_lifecycle_only() {
        // Observability spec, "The event stream": "It holds system calls, IPC requests, guest
        // log lines and lifecycle events; wake, map, fault and schedule events are left out
        // because they depend on timing."
        let shown = [
            EventKind::Svc {
                id: 1,
                args: [0; SVC_ARG_COUNT],
                result: ResultCode(0),
            },
            EventKind::Ipc {
                session: SessionId(1),
                service: ServiceId(2),
                command: 3,
                request_size: 4,
                reply_size: 5,
                result: ResultCode(0),
                outcome: IpcOutcome::Done,
            },
            EventKind::GuestLog { len: 3, hash: 9 },
            EventKind::Lifecycle {
                phase: LifecyclePhase::ProcessStart,
                subject: 1,
            },
        ];
        for kind in shown {
            assert!(kind.in_golden_trace(), "{kind}");
        }
        let schedule = EventKind::Schedule {
            core: 1,
            from: None,
            to: Some(ThreadId(2)),
        };
        assert_eq!(schedule.to_string(), "schedule core=1 idle -> 2");
        let hidden = [
            schedule,
            EventKind::Wake {
                thread: ThreadId(1),
                reason: WakeReason::Signalled,
            },
            EventKind::Map {
                va: 0x1000,
                size: 0x1000,
                perms: 3,
                kind: MapKind::Map,
            },
            EventKind::Fault {
                address: 0x10,
                write: true,
                action: FaultAction::Fatal,
            },
        ];
        for kind in hidden {
            assert!(!kind.in_golden_trace(), "{kind}");
        }
    }

    #[test]
    fn a_guest_log_event_carries_the_length_and_the_fnv_1a_hash_of_the_text() {
        // FNV-1a 64-bit test vectors from the reference implementation's test suite
        // (Fowler, Noll, Vo; `test_fnv.c`): "", "a" and "foobar".
        for (text, hash) in [
            (&b""[..], 0xcbf2_9ce4_8422_2325_u64),
            (b"a", 0xaf63_dc4c_8601_ec8c),
            (b"foobar", 0x8594_4171_f739_67e8),
        ] {
            assert_eq!(
                EventKind::guest_log(text),
                EventKind::GuestLog {
                    len: u32::try_from(text.len()).unwrap(),
                    hash
                },
                "{text:?}"
            );
        }
        // The text of a line is its bytes: one changed byte changes the hash.
        assert_ne!(
            EventKind::guest_log(b"hello"),
            EventKind::guest_log(b"hellp")
        );
    }
}
