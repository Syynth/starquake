//! The injectable environment: the one door nondeterminism comes in through.
//!
//! Source: `docs/specs/observability.md`, "Replaying the kernel and services", rule 1 ("All
//! nondeterminism enters through one injectable environment. The clock, random numbers, host file
//! and network I/O, and input state are read through an `Env` trait. Live runs record what it
//! returned; replays feed the recording back."). `docs/specs/kernel.md` shows the call shape
//! `fn get_system_tick(env: &dyn Env) -> u64`, so [`Env`] is object safe and its methods take
//! `&self`; implementations keep their state behind a lock. The lint rule that bans direct host
//! clock and file calls in the kernel and service crates is a separate piece of work (spec,
//! "Order of work" step 2) and is not part of this crate.
//!
//! * [`LiveEnv`] reads a [`HostClock`], a seeded random generator and an optional
//!   [`HostIoHandler`], and records every answer as an [`EnvRecord`].
//! * [`ReplayEnv`] is built from such a recording, hands the answers back in order, and reports
//!   the first call that does not match.
//!
//! Replay is only meaningful when the calls arrive in the same order as in the live run. The
//! kernel guarantees that by making its calls under its single lock; this crate does not add a
//! lock of its own across calls.
//!
//! **Extension point: host I/O.** The spec lists host file and network I/O and input state. None
//! of those has a design yet, so [`Env::host_io`] is a deliberately generic channel: a numbered
//! channel, request bytes in, reply bytes out. Each concrete kind of host I/O is expected to be
//! layered on it (or to replace it with typed methods) when its owning service is written; until
//! then a [`LiveEnv`] with no handler answers [`HostIoReply::Unavailable`], and that answer is
//! recorded and replayed like any other.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::event::GuestTicks;

mod live;
mod replay;

pub use live::{FixedClock, HostClock, HostIoHandler, LiveEnv, SystemClock};
pub use replay::ReplayEnv;

/// A request on the host I/O extension point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostIoRequest {
    /// Which kind of host I/O is being asked for. Meaning is assigned by whoever layers on this
    /// extension point; this crate does not interpret it.
    pub channel: u32,
    /// The request bytes.
    pub payload: Vec<u8>,
}

/// The answer to a [`HostIoRequest`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostIoReply {
    /// The host produced these bytes.
    Data(Vec<u8>),
    /// The host failed the request with this host-defined code.
    Failed(u32),
    /// Nothing handles host I/O in this environment.
    Unavailable,
}

/// Which kind of call was made on an [`Env`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvCall {
    /// [`Env::now`].
    Clock,
    /// [`Env::random_u64`].
    Random,
    /// [`Env::host_io`].
    HostIo,
}

/// One answer an [`Env`] gave, in the order given.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvRecord {
    /// A clock reading, in the unit of the clock that produced it (see [`Env::now`]).
    Clock(GuestTicks),
    /// A random number.
    Random(u64),
    /// A host I/O exchange.
    HostIo {
        /// What was asked.
        request: HostIoRequest,
        /// What came back.
        reply: HostIoReply,
    },
}

impl EnvRecord {
    /// The kind of call this record answers.
    #[must_use]
    pub fn call(&self) -> EnvCall {
        match self {
            Self::Clock(_) => EnvCall::Clock,
            Self::Random(_) => EnvCall::Random,
            Self::HostIo { .. } => EnvCall::HostIo,
        }
    }
}

/// Why a [`ReplayEnv`] call failed. A replay error is a finding, not a bug in the replayer: it
/// means the code under replay behaved differently from the recorded run.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum EnvError {
    /// The recording ran out: the code asked for more than the live run did.
    #[error("recording exhausted at record {index}: the replay asked for {requested:?}")]
    Exhausted {
        /// Index of the record that would have been consumed.
        index: usize,
        /// The call that found nothing.
        requested: EnvCall,
    },
    /// The next record answers a different kind of call than the one made.
    #[error("record {index} answers {recorded:?} but the replay asked for {requested:?}")]
    Mismatch {
        /// Index of the record that was next.
        index: usize,
        /// The call the replay made.
        requested: EnvCall,
        /// The call the record answers.
        recorded: EnvCall,
    },
    /// A host I/O call asked something different from what the recording asked.
    #[error("record {index}: the host I/O request differs from the recorded one")]
    HostIoRequestMismatch {
        /// Index of the record.
        index: usize,
    },
    /// The replay ended with records still unconsumed.
    #[error("replay finished with {remaining} records unconsumed (next is record {index})")]
    Unconsumed {
        /// Index of the first unconsumed record.
        index: usize,
        /// How many records were left.
        remaining: usize,
    },
}

/// The environment kernel and service code reads nondeterministic inputs through.
///
/// Object safe, with `&self` methods, so code takes `&dyn Env`; implementations are `Send + Sync`
/// because services run on several host threads.
pub trait Env: Send + Sync {
    /// Read the clock.
    ///
    /// The unit is the unit of whatever [`HostClock`] the environment reads, not guest ticks:
    /// [`SystemClock`] reports host nanoseconds and [`FixedClock`] reports whatever its caller
    /// chose. [`GuestTicks`] is only the carrier type here. Converting to the guest counter's
    /// frequency is the consumer's job (guess: the kernel's tick conversion; the spec does not
    /// say where it happens).
    fn now(&self) -> Result<GuestTicks, EnvError>;

    /// Draw a random number.
    fn random_u64(&self) -> Result<u64, EnvError>;

    /// The host I/O extension point (see the module docs): send a request to the host and get its
    /// reply.
    fn host_io(&self, request: &HostIoRequest) -> Result<HostIoReply, EnvError>;
}
