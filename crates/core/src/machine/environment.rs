//! The environment the kernel and the services read during a run ([`RunEnv`]), and the part of
//! its recording the run keeps ([`Kept`]).
//!
//! A run's environment is live or a replay, as its options say ([`EnvSource`]). A live one reads
//! the host and records every answer (`starquake_trace::LiveEnv`); the loop moves what it
//! recorded into the run's kept recording ([`RunEnv::keep`]) after each step that can read it,
//! and the run's report holds what was kept ([`crate::RunReport::environment`]). Fed back as a
//! replay ([`EnvSource::Replay`]), that recording gives the run the same answers in the same
//! order (`docs/specs/observability.md`, rule 1). A replay records nothing: what it answers is
//! the recording it was given, so its run keeps an empty one.
//!
//! What is kept is bounded by bytes ([`super::Limits::env_recording_bytes`]): whole records,
//! from the first, while their total stays at or below the limit; from the first record that
//! does not fit, records are counted and not kept, so what is kept is always a prefix of what
//! was answered, and a replay of it runs out where the prefix ends (which stops the replayed run
//! as an emulator error: decision log, 2026-11-17, "Env failures"). A record's size is the size
//! of an [`EnvRecord`] plus the bytes of a host I/O request and reply, which a guest can make
//! as large as its reads.

use std::sync::Mutex;

use starquake_trace::{
    Env, EnvError, EnvRecord, HostIoReply, HostIoRequest, HostNanos, LiveEnv, ReplayEnv,
    UnixSeconds,
};

use super::lock;
use crate::report::EnvSource;

/// The bytes a run's kept recording may hold unless its limits say otherwise: 16 MiB.
/// Starquake's own number, like the trace's and the guest log's (the observability spec asks
/// for a size cap per run without fixing one).
pub(crate) const ENV_RECORDING_BYTES: usize = 16 << 20;

/// The environment of a run (module docs).
pub(crate) enum RunEnv {
    /// Reads the host and records; what the loop moves out of its recording is kept up to the
    /// limit.
    Live {
        /// The environment.
        env: LiveEnv,
        /// What has been kept of its recording.
        kept: Mutex<Kept>,
    },
    /// Feeds a recording back.
    Replay(ReplayEnv),
}

impl RunEnv {
    /// A run's environment for `source`: `live` when the run reads the host, a replay of the
    /// records `source` holds otherwise. A live one keeps up to `limit` bytes of its recording.
    pub(crate) fn new(source: &EnvSource, live: impl FnOnce() -> LiveEnv, limit: usize) -> Self {
        match source {
            EnvSource::Live => Self::Live {
                env: live(),
                kept: Mutex::new(Kept::new(limit)),
            },
            EnvSource::Replay(records) => Self::Replay(ReplayEnv::new(records.clone())),
        }
    }

    /// Moves what a live environment has recorded since the last call into the kept
    /// recording, as far as its limit allows; a replay has nothing to move.
    ///
    /// Called from every core and the timer thread. The kept recording's lock is held while the
    /// environment's recording is taken, so takes and appends happen in one order: each take
    /// holds everything recorded since the one before it, and the kept recording is the
    /// environment's answers in the order it gave them (test:
    /// `keeping_from_several_threads_keeps_the_answers_in_the_order_given`).
    pub(crate) fn keep(&self) {
        if let Self::Live { env, kept } = self {
            let mut kept = lock(kept);
            let records = env.take_recording();
            kept.push(records);
        }
    }

    /// What the run kept of its recording, in order, and how many later records were not kept.
    /// Empty for a replay.
    pub(crate) fn kept(&self) -> (Vec<EnvRecord>, u64) {
        match self {
            Self::Live { kept, .. } => {
                let kept = lock(kept);
                (kept.records.clone(), kept.dropped)
            }
            Self::Replay(_) => (Vec::new(), 0),
        }
    }

    /// What a live environment has recorded and not yet moved out ([`RunEnv::keep`]); nothing
    /// for a replay.
    #[cfg(test)]
    pub(crate) fn recording(&self) -> Vec<EnvRecord> {
        match self {
            Self::Live { env, .. } => env.recording(),
            Self::Replay(_) => Vec::new(),
        }
    }
}

impl Env for RunEnv {
    fn now(&self) -> Result<HostNanos, EnvError> {
        self.inner().now()
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        self.inner().random_u64()
    }

    fn host_io(&self, request: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        self.inner().host_io(request)
    }

    fn wall_clock(&self) -> Result<UnixSeconds, EnvError> {
        self.inner().wall_clock()
    }
}

impl RunEnv {
    /// The environment that answers.
    fn inner(&self) -> &dyn Env {
        match self {
            Self::Live { env, .. } => env,
            Self::Replay(env) => env,
        }
    }
}

/// The part of a live environment's recording a run keeps (module docs): a prefix, bounded by
/// bytes.
pub(crate) struct Kept {
    /// The records kept, from the first.
    records: Vec<EnvRecord>,
    /// Bytes of `records`, as [`record_bytes`] counts them.
    bytes: usize,
    /// The most bytes `records` may hold.
    limit: usize,
    /// Records not kept: the first that did not fit, and every one after it.
    dropped: u64,
}

impl Kept {
    /// Nothing kept yet, with room for `limit` bytes.
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            records: Vec::new(),
            bytes: 0,
            limit,
            dropped: 0,
        }
    }

    /// Keeps `records`, in order, while they fit; counts the rest.
    pub(crate) fn push(&mut self, records: Vec<EnvRecord>) {
        for record in records {
            let fits = self.dropped == 0
                && self
                    .bytes
                    .checked_add(record_bytes(&record))
                    .is_some_and(|total| total <= self.limit);
            if fits {
                self.bytes = self.bytes.saturating_add(record_bytes(&record));
                self.records.push(record);
            } else {
                self.dropped = self.dropped.saturating_add(1);
            }
        }
    }

    /// The records kept, in order.
    #[cfg(test)]
    pub(crate) fn records(&self) -> &[EnvRecord] {
        &self.records
    }

    /// How many records were not kept.
    #[cfg(test)]
    pub(crate) const fn dropped(&self) -> u64 {
        self.dropped
    }
}

/// The bytes a record counts for against the limit: the record itself, and the bytes of a host
/// I/O request and reply it carries, which live outside it.
pub(crate) fn record_bytes(record: &EnvRecord) -> usize {
    let carried = match record {
        EnvRecord::HostIo { request, reply } => {
            let reply = match reply {
                HostIoReply::Data(bytes) => bytes.len(),
                HostIoReply::Failed(_) | HostIoReply::Unavailable => 0,
            };
            request.payload.len().saturating_add(reply)
        }
        EnvRecord::Clock(_) | EnvRecord::Random(_) | EnvRecord::WallClock(_) => 0,
    };
    size_of::<EnvRecord>().saturating_add(carried)
}

#[cfg(test)]
mod tests;
