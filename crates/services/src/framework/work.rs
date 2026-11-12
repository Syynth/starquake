//! Where a method sends work that takes time: an [`Executor`].
//!
//! A method never blocks (`docs/specs/services.md`, "Concurrency": "Work that takes time goes to a
//! worker and completes a deferred reply"). A method that has such work takes a reply token
//! ([`Context::defer`](super::Context::defer)), wraps the work and the token in a [`Job`], and
//! gives it to its server's executor ([`Context::executor`](super::Context::executor)). The job
//! runs wherever the executor runs it, reads the host through the injectable environment it was
//! given ([`Context::env`](super::Context::env)), and completes the token. `fsp-srv`'s read is
//! the first user (`crate::fsp`).
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

/// A piece of work a method hands to its server's [`Executor`]: it owns everything it needs,
/// including the reply token it completes.
pub type Job = Box<dyn FnOnce() + Send + 'static>;

/// Runs the [`Job`]s a server's methods hand over, on whatever thread it chooses: a pool of I/O
/// threads in a run, or the test itself, one job at a time, in a test.
///
/// `submit` must not block: a method calls it while it handles a request. A job that is dropped
/// without being run drops its reply token, and the caller gets
/// [`DEFERRED_REPLY_DROPPED`](super::DEFERRED_REPLY_DROPPED).
pub trait Executor: Send + Sync {
    /// Takes `job`, to run later.
    fn submit(&self, job: Job);
}
