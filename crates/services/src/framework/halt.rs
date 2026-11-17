//! How a service stops the run when the injectable environment fails it: [`Halt`], told why with
//! a [`Stop`], and the one path a handler reads the environment through
//! ([`Server::with_env`](super::Server::with_env)).
//!
//! The owner's ruling (decision log, 2026-11-17, "Env failures: replay failures stop the run, host
//! I/O failures are guest results"; `docs/specs/observability.md`, "Replaying the kernel and
//! services", rule 1) splits what can go wrong when a service asks the host in two:
//!
//! - **The environment fails** (an [`EnvError`]: a replay that ran out of answers or disagreed
//!   with what was asked). The run is no longer the recorded one, so it stops as an emulator
//!   error, never as a guest-visible result. This module is that path.
//! - **The host fails** (an `Ok` answer of [`HostIoReply::Failed`] or
//!   [`HostIoReply::Unavailable`]: a missing or unreadable file, no SD card). That is a Horizon
//!   result code, as on a console, and the handler that asked names it: this module lets such an
//!   answer through untouched (`a_host_failure_is_the_handler_s_and_stops_nothing`). Of the
//!   services below only `fsp-srv` sends host I/O; its module docs, "Host failures", list each
//!   code.
//!
//! # The path
//!
//! A server is given its environment with [`Server::with_env`](super::Server::with_env), which
//! takes the run's [`Halt`] with it. The server keeps the environment only wrapped: every call a
//! method makes through [`Context::env`](super::Context::env) that fails tells the halt
//! [`Stop::Handler`], naming the server ([`Server::name`](super::Server::name)) and the error, on
//! the thread that saw the failure, before the method gets the error back
//! (`a_failure_is_told_before_the_method_sees_it`). Each of the three calls
//! ([`Env::now`], [`Env::random_u64`], [`Env::host_io`]) is wrapped
//! (`every_call_that_fails_tells_the_halt_once`). An answer that is not an error passes through
//! unchanged and tells nothing (`answers_pass_through_and_tell_nothing`).
//!
//! The method then ends the request with `?`: an [`EnvError`] converts to the [`Fault`] that
//! answers it as one the server gave up on, [`DEFERRED_REPLY_DROPPED`]. The guest never runs to
//! see it: the run's [`Halt`] was told first, and a run stops before it enters the guest again
//! (`starquake-core`'s core loop checks for a halt after it has chosen the thread it runs and
//! before it runs it). A method that drops the error and replies anyway still stops the run,
//! because the halt was told before the method saw the error
//! (`a_method_that_ignores_the_failure_still_stops_the_run`).
//!
//! # Who reads the environment
//!
//! Exactly these, in this crate:
//!
//! - **`fsp-srv`'s handlers**, through [`Context::env`](super::Context::env):
//!   `OpenSdCardFileSystem`, `OpenDataStorageByCurrentProcess`,
//!   `OpenDataFileSystemByCurrentProcess`, `IFileSystem::OpenFile`, `IFile::GetSize` and
//!   `IStorage::GetSize` (`an_environment_failure_in_a_handler_stops_the_run` in
//!   `fsp/tests.rs`, for each, with a replay that runs out and one that disagrees).
//! - **`time:u`'s `GetSharedMemoryNativeHandle`**, the first time it is asked, when it sets the
//!   clocks up, through [`Context::env`](super::Context::env)
//!   (`an_environment_failure_stops_the_run_and_keeps_nothing` in `time/tests.rs`).
//! - **`fsp-srv`'s reads** (`IFile::Read`, `IStorage::Read`), which run as jobs on an executor
//!   and read the environment the executor gives them, not the server's
//!   ([`work`](super::work)). That is the one path besides this one: `fsp-srv`'s executors watch
//!   their own environments and tell the halt [`Stop::Job`] or [`Stop::Replay`]
//!   (`fsp/worker.rs`, "When the environment fails a job").
//!
//! `sm:`, `set:sys`, `apm`, `appletOE` and `hid` are given no environment and read none; a method
//! of theirs that asked would be answered `NOT_SUPPORTED`
//! ([`Context::env`](super::Context::env)).
//!
//! # What is checked
//!
//! - **By the compiler**: a server keeps no environment but the wrapped one, and the wrapper is
//!   private to this module, so [`Context::env`](super::Context::env) cannot hand a method an
//!   environment that does not tell the halt.
//! - **By a test**: a method could still reach an environment some other way: a hand-written
//!   [`Family`](super::Family) (which `serde` does not force to be plain data) holding one, a
//!   job capturing one, or a free function given one. The test
//!   `only_the_listed_files_name_the_environment` (`framework/halt/tests.rs`) reads this crate's
//!   non-test sources and fails when a file outside an exact list names `Env`, so a new reader of
//!   the environment has to be added there, with its path, and to the list above. The check is a
//!   text search for the word `Env` in code (not comments); it cannot see an environment reached
//!   through a type alias or a re-export under another name, which is for review to catch.

use std::sync::Arc;

use starquake_kernel::result::ResultCode;
use starquake_trace::{Env, EnvError, HostIoReply, HostIoRequest, HostNanos};

use super::result::{DEFERRED_REPLY_DROPPED, Fault};
use crate::fsp::worker::ReplayError;

#[cfg(test)]
mod tests;

/// Why a service stops the run (module docs).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Stop {
    /// The environment failed a handler of the server named `service`, which reads the
    /// environment the server was given ([`Server::with_env`](super::Server::with_env)).
    #[error("the environment failed {service} while it handled a request: {error}")]
    Handler {
        /// The server's name ([`Server::name`](super::Server::name)).
        service: String,
        /// What the environment answered.
        error: EnvError,
    },
    /// The environment a read on `fsp-srv`'s workers was given failed it, live or, as recorded,
    /// in a replay.
    #[error("the environment failed file-system job {job}: {error}")]
    Job {
        /// The job's number ([`JobIndex`](crate::fsp::worker::JobIndex)).
        job: u64,
        /// What the environment answered.
        error: EnvError,
    },
    /// A replay of `fsp-srv`'s reads left its record.
    #[error("the replay of fsp-srv's reads left its record: {0}")]
    Replay(ReplayError),
}

/// What a service tells when the environment fails it: in a run, the core loop's, which ends the
/// run as an emulator error with the [`Stop`]'s words.
///
/// It is called on the thread that saw the failure: a core thread or the timer thread while a
/// handler runs, a worker thread, or, in a replay of `fsp-srv`'s reads, the thread that submitted
/// the job, with the replay's lock held. It must not block, and must not call into a service's
/// server or submit to its executor. It may be called more than once in a run; the first call is
/// the one that says why the run stopped.
pub trait Halt: Send + Sync {
    /// The environment failed a service: the run must stop, for `stop`.
    fn halt(&self, stop: Stop);
}

/// What a request whose handler the environment failed is answered with: one the server gave up
/// on. The guest never runs to see it (module docs, "The path"). `fsp-srv`'s handlers, which sort
/// a host answer of their own before they fail, use it too.
pub(crate) const STOPPED: ResultCode = DEFERRED_REPLY_DROPPED;

impl From<EnvError> for Fault {
    /// The fault a method ends its request with when the environment failed it: [`STOPPED`]. The
    /// run's halt has already been told (module docs, "The path").
    fn from(_: EnvError) -> Self {
        Self::Result(STOPPED)
    }
}

/// The environment a server's handlers read: the one the server was given, telling `halt` of
/// every failure as [`Stop::Handler`], naming `service`, before it returns it.
pub(super) struct Watched {
    service: String,
    env: Arc<dyn Env>,
    halt: Arc<dyn Halt>,
}

impl Watched {
    pub(super) fn new(service: String, env: Arc<dyn Env>, halt: Arc<dyn Halt>) -> Self {
        Self { service, env, halt }
    }

    /// `answer`, after telling the halt if it is a failure.
    fn told<T>(&self, answer: Result<T, EnvError>) -> Result<T, EnvError> {
        if let Err(error) = &answer {
            self.halt.halt(Stop::Handler {
                service: self.service.clone(),
                error: error.clone(),
            });
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

/// A [`Halt`] for tests that keeps every stop it is told, in the order told, where a run's would
/// stop. Like a run's, it may be told more than once, from any thread, and never blocks for
/// longer than its own lock.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Default)]
pub struct Halts(std::sync::Mutex<Vec<Stop>>);

#[cfg(any(test, feature = "test-support"))]
impl Halt for Halts {
    fn halt(&self, stop: Stop) {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(stop);
    }
}

#[cfg(any(test, feature = "test-support"))]
impl Halts {
    /// This halt, as a server or an executor takes one.
    #[must_use]
    pub fn shared(self: &Arc<Self>) -> Arc<dyn Halt> {
        Arc::<Self>::clone(self)
    }

    /// Every stop told so far, oldest first.
    #[must_use]
    pub fn stops(&self) -> Vec<Stop> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}
