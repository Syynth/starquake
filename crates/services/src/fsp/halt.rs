//! How `fsp-srv` stops the run when the injectable environment fails it: [`Halt`], told why with
//! a [`Stop`].
//!
//! The owner's ruling (decision log, 2026-11-17, "Env failures: replay failures stop the run, host
//! I/O failures are guest results"; `docs/specs/observability.md`, "Replaying the kernel and
//! services", rule 1) splits what can go wrong behind a file request in two:
//!
//! - **The environment fails** (an [`EnvError`]: a replay that ran out of answers or disagreed
//!   with what was asked). The run is no longer the recorded one, so it stops as an emulator
//!   error, never as a guest-visible result. This module is that path.
//! - **The host fails** (a [`HostIoReply::Failed`] or [`HostIoReply::Unavailable`] answer: a
//!   missing or unreadable file, no SD card). That is a Horizon result code, as on a console:
//!   the module docs of [`fsp`](super), "Host failures", list each one.
//!
//! # Where an environment failure is seen
//!
//! There are three places, and each tells the run's [`Halt`] on the thread that saw the failure,
//! at the moment it saw it, before the request it belongs to is answered:
//!
//! - **A handler** (`OpenSdCardFileSystem`, `OpenDataStorageByCurrentProcess`,
//!   `OpenDataFileSystemByCurrentProcess`, `OpenFile`, and both `GetSize`s) reads the
//!   environment the server was given, which [`server`](super::server) wraps so that each
//!   failure is told as [`Stop::Handler`].
//! - **A read on the workers** ([`Workers`](super::worker::Workers)): the environment a job reads
//!   through tells [`Stop::Job`] at its first failure.
//! - **A replay** ([`Replay`](super::worker::Replay)): a job the record does not hold, or one
//!   that asks otherwise than the record, is [`Stop::Replay`]; a replayed job that reaches the
//!   failure the record holds for it is [`Stop::Job`], as the live job's was.
//!
//! # What the request gets
//!
//! The request whose environment failed is answered as one the server gave up on,
//! [`DEFERRED_REPLY_DROPPED`](crate::framework::DEFERRED_REPLY_DROPPED): a handler fails with it,
//! and an executor drops the job's finish without running it, which drops its reply token. The
//! guest never runs to see it: the run's [`Halt`] is told first, and a run stops before it
//! enters the guest again (`starquake-core`'s core loop checks for a halt after it has chosen
//! the thread it runs and before it runs it).

use std::sync::Arc;

use starquake_trace::{Env, EnvError, HostIoReply, HostIoRequest, HostNanos};

use super::worker::ReplayError;

/// Why `fsp-srv` stops the run (module docs).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Stop {
    /// The environment failed a handler, which reads the environment the server was given.
    #[error("the environment failed fsp-srv while it handled a request: {0}")]
    Handler(EnvError),
    /// The environment a read on the workers was given failed it, live or, as recorded, in a
    /// replay.
    #[error("the environment failed file-system job {job}: {error}")]
    Job {
        /// The job's number ([`JobIndex`](super::worker::JobIndex)).
        job: u64,
        /// What the environment answered.
        error: EnvError,
    },
    /// A replay of `fsp-srv`'s reads left its record.
    #[error("the replay of fsp-srv's reads left its record: {0}")]
    Replay(ReplayError),
}

/// What `fsp-srv` tells when the environment fails it: in a run, the core loop's, which ends the
/// run as an emulator error with the [`Stop`]'s words.
///
/// It is called on the thread that saw the failure: a core thread or the timer thread while a
/// handler runs, a worker thread, or, in a replay, the thread that submitted the job, with the
/// replay's lock held. It must not block, and must not call into `fsp-srv`'s server or submit
/// to its executor. It may be called more than once in a run; the first call is the one that
/// says why the run stopped.
pub trait Halt: Send + Sync {
    /// The environment failed `fsp-srv`: the run must stop, for `stop`.
    fn halt(&self, stop: Stop);
}

/// The environment a server's handlers read: the one the server was given, telling `halt` of
/// every failure as [`Stop::Handler`] before it returns it.
pub(crate) struct Watched {
    env: Arc<dyn Env>,
    halt: Arc<dyn Halt>,
}

impl Watched {
    pub(crate) fn new(env: Arc<dyn Env>, halt: Arc<dyn Halt>) -> Self {
        Self { env, halt }
    }

    /// `answer`, after telling the halt if it is a failure.
    fn told<T>(&self, answer: Result<T, EnvError>) -> Result<T, EnvError> {
        if let Err(error) = &answer {
            self.halt.halt(Stop::Handler(error.clone()));
        }
        answer
    }
}

impl Env for Watched {
    fn now(&self) -> Result<HostNanos, EnvError> {
        self.told(self.env.now())
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        self.told(self.env.random_u64())
    }

    fn host_io(&self, request: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        self.told(self.env.host_io(request))
    }
}
