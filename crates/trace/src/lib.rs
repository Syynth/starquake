//! The event stream (`docs/specs/observability.md`).
//!
//! One structured event per crossing of a boundary we own, recorded at a single point per
//! boundary; an always-on flight recorder; and the injectable environment through which the
//! kernel and services read the clock, random numbers and host I/O, so their runs can be replayed.
//!
//! * [`event`]: the [`Event`] and [`EventKind`] types and the small id newtypes (this crate is a
//!   leaf, so the ids are defined here).
//! * [`recorder`]: the [`Recorder`] that numbers events and fans them out to [`Sink`]s.
//! * [`flight`] and [`dump`]: the [`FlightRecorder`] ring per core, and its `postcard` [`Dump`]
//!   with a reader.
//! * [`text`]: the golden-trace text view ([`TextSink`], [`golden_trace`]).
//! * [`env`]: the [`Env`] trait with [`LiveEnv`] and [`ReplayEnv`].

pub mod dump;
pub mod env;
pub mod event;
pub mod flight;
pub mod recorder;
pub mod text;

pub use dump::{CoreDump, DUMP_MAGIC, DUMP_VERSION, Dump, DumpError};
pub use env::{
    Env, EnvCall, EnvError, EnvRecord, FixedClock, HostClock, HostIoHandler, HostIoReply,
    HostIoRequest, LiveEnv, ReplayEnv, SystemClock,
};
pub use event::{
    Event, EventKind, FaultAction, GuestTicks, IpcOutcome, LifecyclePhase, MapKind, ResultCode,
    SVC_ARG_COUNT, ServiceId, SessionId, ThreadId, WakeReason,
};
pub use flight::FlightRecorder;
pub use recorder::{Recorder, Sink, VecSink};
pub use text::{TextSink, golden_trace};
