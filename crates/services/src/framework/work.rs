//! Where a method sends work that takes time: an [`Executor`].
//!
//! A method never blocks (`docs/specs/services.md`, "Concurrency": "Work that takes time goes to a
//! worker and completes a deferred reply"). A method that has such work takes a reply token
//! ([`Context::defer`](super::Context::defer)), wraps the work and the token in a [`Job`], and
//! gives it to its server's executor ([`Context::executor`](super::Context::executor)).
//!
//! A job runs in two steps, so that the executor decides both what the work reads and when its
//! result becomes visible:
//!
//! 1. The job itself runs wherever the executor runs it, with an environment the executor gives
//!    it, and reads the host through that environment only. It returns its [`Finish`].
//! 2. The [`Finish`] completes the reply token. It does no I/O; the executor calls it when it
//!    chooses, which is how an executor fixes the order in which completions reach the server's
//!    mailbox, and so the order in which the deferred replies are delivered (`fsp-srv`'s workers
//!    record that order and a replay reproduces it: `crate::fsp::worker`).
//!
//! The executor gives the job its environment, rather than the job taking the server's
//! ([`Context::env`](super::Context::env)), so that work running on other threads does not put
//! its host reads into the stream the core threads read the environment through, where their
//! place would depend on host timing (`docs/specs/observability.md`, "Replaying the kernel and
//! services": a replay feeds the recorded answers back in order).
//!
//! An executor is not one of a server's workers (the request queue's count of handlers that may
//! run at once): a deferred request has given its worker back by the time its job runs.
//!
//! Neither the executor nor the environment is part of a server's snapshot: both are how a run
//! reaches its host, not state. A session or a family therefore never holds one; a method gets
//! them from its [`Context`](super::Context) while it runs. A job still queued when a snapshot is
//! taken is lost with the run it belonged to, as its token is
//! ([`Server::reissue`](super::Server::reissue) says how such a request is resumed).
//!
//! No reference source is ported: Eden's services do their work on their own host threads.

use starquake_trace::Env;

/// The last step of a [`Job`]: completes the job's reply token with what the job read. It does
/// no I/O, so an executor may hold it back to fix the order completions are posted in.
pub type Finish = Box<dyn FnOnce() + Send + 'static>;

/// A piece of work a method hands to its server's [`Executor`]: it owns everything it needs,
/// including the reply token its [`Finish`] completes. It reads the host only through the
/// environment it is called with.
pub type Job = Box<dyn FnOnce(&dyn Env) -> Finish + Send + 'static>;

/// Runs the [`Job`]s a server's methods hand over, on whatever thread it chooses: a pool of I/O
/// threads in a run, or the test itself, one job at a time, in a test. It calls each job with the
/// environment the job is to read the host through, and then the job's [`Finish`].
///
/// `submit` must not block: a method calls it while it handles a request. A job that is dropped
/// without being run, or whose [`Finish`] is dropped without being called, drops its reply
/// token, and the caller gets [`DEFERRED_REPLY_DROPPED`](super::DEFERRED_REPLY_DROPPED).
pub trait Executor: Send + Sync {
    /// Takes `job`, to run later.
    fn submit(&self, job: Job);
}
