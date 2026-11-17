//! `fsp-srv`'s workers: the [`Executor`] its reads run on in a run ([`Workers`]), what they
//! record, and the executor that replays a recording ([`Replay`]).
//!
//! `docs/specs/services.md`, "Concurrency": methods never block, reading file data is always
//! deferred, and work that takes time goes to a worker that completes a deferred reply.
//! `IFile::Read` and `IStorage::Read` hand their read to the server's executor as a [`Job`]
//! (`crate::fsp`, "The host behind it"); a run gives `fsp-srv` a [`Workers`].
//!
//! No reference source is ported: Eden's file system answers on the calling thread, and the
//! design is the service and observability specs'.
//!
//! # The workers
//!
//! [`Workers`] is one host thread per environment it is given ([`COUNT`] in a run), and the
//! threads take jobs from one queue, oldest first. Each thread runs its jobs with its own
//! environment, and a job reads the host through it alone and returns its [`Finish`]. The
//! thread then calls the [`Finish`], which completes the job's reply token: the server's mailbox
//! has the completion, and the server's [`Notify`](crate::framework::Notify) is told (a run's
//! tells the core loop, which delivers the reply).
//!
//! Each thread has an environment of its own because an environment may answer one request at
//! a time: a `LiveEnv` (`starquake-trace`) holds its lock for the whole of a host request. Two
//! threads sharing one would wait for each other's host reads. With one each, two reads in
//! flight complete independently: a job that waits on the host holds up its own thread and its
//! own environment only, and a read that fails fails only its own request
//! (`two_reads_in_flight_complete_independently`, `a_read_that_fails_fails_only_its_own_request`
//! in `fsp/tests/worker.rs`; on a run's own environments,
//! `a_read_held_at_the_title_does_not_hold_up_a_card_read` in `starquake-core`'s
//! `machine/tests/reads.rs`).
//!
//! [`COUNT`] is starquake's own choice, not the console's (which no pinned reference states): two,
//! so that one long read does not hold up every other.
//!
//! A job or a finish that panics does not take its thread with it: the thread goes on to the next
//! job. The panicking job's reply token is dropped as the panic unwinds, so its caller gets
//! `DEFERRED_REPLY_DROPPED`, and a job that panics before it returns its [`Finish`] is not
//! recorded (`a_job_that_panics_leaves_its_thread_to_run_the_next`).
//!
//! Dropping [`Workers`] stops them and waits for every thread to end: a job running then runs to
//! its end and is finished, and the jobs still queued are dropped without running, so each of
//! their callers gets `DEFERRED_REPLY_DROPPED`. No thread is left behind
//! (`dropping_the_workers_with_reads_in_flight_leaves_no_thread_behind`).
//!
//! # What is recorded
//!
//! The observability spec asks that the order in which deferred replies complete be recorded
//! ("Replaying the kernel and services", rule 3), and that host I/O come through the environment
//! and be recorded there (rule 1). For a job run here, that is one [`Completed`] entry, made by
//! the thread that finished the job:
//!
//! - [`Completed::job`]: the job's [`JobIndex`], its place in the order the workers were given
//!   jobs (from zero, counting every job submitted). Jobs are submitted by the server's
//!   methods, one at a time, in the order the server handles requests, so a replay that handles
//!   the same requests submits the same jobs with the same numbers.
//! - [`Completed::answers`]: every answer the environment gave the job, in order, as
//!   [`EnvRecord`]s (for a read, the one host I/O exchange with its bytes).
//! - [`Completed::failure`]: the error the environment answered the job with, if one did, with
//!   the question that got it ([`Failure`]); the job's later calls get the same error without
//!   reaching the environment.
//!
//! Entries are kept in the order the jobs were finished. The thread finishes the job and appends
//! its entry under one lock, so the order of the entries is the order the completions reached the
//! server's mailbox, which is the order the server's deferred replies are delivered in, and so
//! the order the reads become visible to the guest
//! (`the_record_holds_completions_in_the_order_they_reached_the_mailbox`). Each delivery also
//! records the request's trace event (`framework::trace`), so the event stream says where in the
//! run's order each read became visible.
//!
//! A job's answers are kept with the job, not in the order of the environment's own recording,
//! because the job runs on a host thread of its own: where its reads fall among other threads'
//! calls is host timing, which a replay cannot reproduce.
//!
//! [`Workers::take_record`] takes what has been recorded. With [`Recording::Off`] nothing is kept
//! (`with_recording_off_nothing_is_kept`): the record holds every byte every read returned, so a
//! run that does not write it out must not keep it. A job's answers are not copied then either:
//! only its first failure is held while it runs, so that its later calls get the same error.
//!
//! # Replaying
//!
//! [`Replay`] is an executor made from a record. It runs no threads and never reaches the host:
//! a job submitted to it runs when every job recorded as finished before it has finished, on the
//! thread whose submission made that so (the job's own, or a later one's), with its state's lock
//! held, so a server's notification must not submit to it; with an environment that hands back
//! its own recorded answers
//! (`ReplayEnv`, `starquake-trace`), and is finished at once. So the completions reach the
//! server's mailbox in the recorded order, whatever order the replay submits them in, and the
//! replies are delivered in the recorded order with the recorded bytes
//! (`a_replay_delivers_the_reads_in_the_recorded_order`).
//!
//! A replay that does not follow the record is a finding, not a guest-visible result (decision
//! log, 2026-11-17, "Env failures"): [`Replay::check`] says how it first diverged
//! ([`ReplayError`]). A replayed job that reaches its recorded failure must ask what the live
//! job asked when it failed (the same kind of call, and for host I/O the same request), or it
//! has diverged too. A record that numbers one job twice is refused when the replay is made
//! ([`RecordError`]).
//! What becomes of the job's request then is not the guest's concern, since the run is no longer
//! the recorded one: a job the record does not hold is dropped, and one whose reads differ from
//! the record is finished with what the environment answered (`ReplayEnv`'s error).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::num::NonZeroUsize;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use starquake_trace::{
    Env, EnvCall, EnvError, EnvRecord, HostIoReply, HostIoRequest, HostNanos, ReplayEnv,
};

use crate::framework::{Executor, Job};

/// How many threads `fsp-srv`'s [`Workers`] run (module docs, "The workers").
pub const COUNT: NonZeroUsize = NonZeroUsize::MIN.saturating_add(1);

/// A job's place in the order an executor was given jobs, from zero (module docs, "What is
/// recorded").
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobIndex(u64);

impl JobIndex {
    /// The job submitted after `number` others.
    #[must_use]
    pub const fn new(number: u64) -> Self {
        Self(number)
    }

    /// The index as a number.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// One finished job, as the workers record it (module docs, "What is recorded").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completed {
    /// Which job.
    pub job: JobIndex,
    /// Every answer the environment gave it, in order.
    pub answers: Vec<EnvRecord>,
    /// The error the environment answered it with, if one did: after `answers`, every call the
    /// job made got it.
    pub failure: Option<Failure>,
}

/// What a job asked the environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Question {
    /// The time ([`Env::now`]).
    Clock,
    /// A random number ([`Env::random_u64`]).
    Random,
    /// This host I/O request ([`Env::host_io`]).
    HostIo(HostIoRequest),
}

impl Question {
    /// The kind of call it is.
    #[must_use]
    pub fn call(&self) -> EnvCall {
        match self {
            Self::Clock => EnvCall::Clock,
            Self::Random => EnvCall::Random,
            Self::HostIo(_) => EnvCall::HostIo,
        }
    }
}

/// The error the environment answered a job with, and the question that got it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    /// What the job asked.
    pub asked: Question,
    /// What the environment answered.
    pub error: EnvError,
}

/// Whether [`Workers`] keep a record of the jobs they finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recording {
    /// Keep one [`Completed`] entry per finished job, until [`Workers::take_record`] takes them.
    Kept,
    /// Keep nothing.
    Off,
}

/// Why [`Workers`] could not be started.
#[derive(Debug, thiserror::Error)]
pub enum StartError {
    /// The workers were given no environment, so they would have no thread.
    #[error("the file-system workers were given no environment to run a thread with")]
    NoEnvironment,
    /// The host could not start a thread.
    #[error("a file-system worker thread could not be started: {0}")]
    Thread(#[source] std::io::Error),
}

/// A mutex's value, whatever a panicking holder left: every value here is valid between the
/// steps that change it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The next [`JobIndex`] from `next`, which moves on. `None` once `next` is `u64::MAX`, which is
/// never handed out: a run would have to submit 2^64 - 1 jobs first.
fn take_index(next: &mut u64) -> Option<JobIndex> {
    let index = JobIndex(*next);
    *next = next.checked_add(1)?;
    Some(index)
}

/// The jobs waiting for a thread.
struct Queue {
    jobs: VecDeque<(JobIndex, Job)>,
    next: u64,
    /// Set when the workers are dropped: no thread takes another job.
    closed: bool,
}

/// What the workers' threads share.
struct Shared {
    queue: Mutex<Queue>,
    /// Rung when a job is queued or the queue closes.
    queued: Condvar,
    recording: Recording,
    /// The record. Its lock is also held while a job is finished, so the entries are in the
    /// order completions reached the mailbox.
    record: Mutex<Vec<Completed>>,
}

impl Shared {
    /// The next job to run, oldest first; `None` once the queue is closed.
    fn next_job(&self) -> Option<(JobIndex, Job)> {
        let mut queue = lock(&self.queue);
        // Bounded by the wake-ups: each turn either returns or waits for a ring.
        loop {
            if queue.closed {
                return None;
            }
            if let Some(job) = queue.jobs.pop_front() {
                return Some(job);
            }
            queue = self
                .queued
                .wait(queue)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// A worker thread's loop: run, finish and record jobs, each with the thread's own `env`,
    /// until the queue closes. A job or a finish that panics is logged, and the thread goes on
    /// (module docs, "The workers").
    fn work(&self, env: &dyn Env) {
        while let Some((job, run)) = self.next_job() {
            let recorded = Recorded::new(env, self.recording);
            let Ok(finish) = catch_unwind(AssertUnwindSafe(|| run(&recorded))) else {
                tracing::error!(
                    job = job.raw(),
                    "a file-system job panicked: its request is dropped"
                );
                continue;
            };
            let (answers, failure) = recorded.into_record();
            let mut record = lock(&self.record);
            if catch_unwind(AssertUnwindSafe(finish)).is_err() {
                tracing::error!(
                    job = job.raw(),
                    "a file-system job's finish panicked: its request is dropped"
                );
            }
            if self.recording == Recording::Kept {
                record.push(Completed {
                    job,
                    answers,
                    failure,
                });
            }
        }
    }
}

/// `fsp-srv`'s workers in a run (module docs, "The workers").
pub struct Workers {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    /// One thread per environment in `envs`, whose jobs read the host through that thread's
    /// environment alone, keeping a record of what they finish or not (module docs, "The
    /// workers").
    ///
    /// # Errors
    ///
    /// [`StartError::NoEnvironment`] when `envs` is empty, and [`StartError::Thread`] when the
    /// host cannot start a thread. The threads started before it are stopped and waited for.
    pub fn new(envs: Vec<Arc<dyn Env>>, recording: Recording) -> Result<Self, StartError> {
        if envs.is_empty() {
            return Err(StartError::NoEnvironment);
        }
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                next: 0,
                closed: false,
            }),
            queued: Condvar::new(),
            recording,
            record: Mutex::new(Vec::new()),
        });
        let mut workers = Self {
            shared,
            threads: Vec::with_capacity(envs.len()),
        };
        for (number, env) in envs.into_iter().enumerate() {
            let shared = Arc::clone(&workers.shared);
            let thread = std::thread::Builder::new()
                .name(format!("starquake-fsp-{number}"))
                .spawn(move || shared.work(&*env))
                .map_err(StartError::Thread)?;
            workers.threads.push(thread);
        }
        Ok(workers)
    }

    /// Takes what has been recorded so far, oldest first, leaving the record empty (module docs,
    /// "What is recorded"). Always empty with [`Recording::Off`].
    #[must_use]
    pub fn take_record(&self) -> Vec<Completed> {
        std::mem::take(&mut *lock(&self.shared.record))
    }

    /// Makes the next job submitted the one numbered `next`.
    #[cfg(test)]
    pub(crate) fn number_from(&self, next: u64) {
        lock(&self.shared.queue).next = next;
    }
}

impl Executor for Workers {
    fn submit(&self, job: Job) {
        let mut queue = lock(&self.shared.queue);
        let Some(index) = take_index(&mut queue.next) else {
            tracing::error!("the file-system workers have numbered every job they can: dropped");
            return;
        };
        queue.jobs.push_back((index, job));
        drop(queue);
        self.shared.queued.notify_one();
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        // The queued jobs are dropped here, outside the lock: dropping one drops its token,
        // which posts to the server's mailbox.
        let dropped = {
            let mut queue = lock(&self.shared.queue);
            queue.closed = true;
            std::mem::take(&mut queue.jobs)
        };
        self.shared.queued.notify_all();
        drop(dropped);
        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                tracing::error!("a file-system worker thread panicked");
            }
        }
    }
}

/// The environment one job runs with on a worker: its thread's own, with every answer kept for
/// the job's record when the workers keep one. The first error ends the job's use of the
/// environment: it and every later call get that error, which is held whether or not answers
/// are kept.
struct Recorded<'a> {
    env: &'a dyn Env,
    recording: Recording,
    kept: Mutex<(Vec<EnvRecord>, Option<Failure>)>,
}

impl<'a> Recorded<'a> {
    fn new(env: &'a dyn Env, recording: Recording) -> Self {
        Self {
            env,
            recording,
            kept: Mutex::new((Vec::new(), None)),
        }
    }

    /// Asks the environment with `ask`, unless it has already failed, and keeps the answer when
    /// the workers keep a record. `asked` says what was asked, for a failure.
    fn ask<T>(
        &self,
        asked: impl FnOnce() -> Question,
        ask: impl FnOnce(&dyn Env) -> Result<T, EnvError>,
        keep: impl FnOnce(&T) -> EnvRecord,
    ) -> Result<T, EnvError> {
        let mut kept = lock(&self.kept);
        if let Some(failure) = &kept.1 {
            return Err(failure.error.clone());
        }
        match ask(self.env) {
            Ok(answer) => {
                if self.recording == Recording::Kept {
                    kept.0.push(keep(&answer));
                }
                Ok(answer)
            }
            Err(error) => {
                kept.1 = Some(Failure {
                    asked: asked(),
                    error: error.clone(),
                });
                Err(error)
            }
        }
    }

    fn into_record(self) -> (Vec<EnvRecord>, Option<Failure>) {
        self.kept
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl Env for Recorded<'_> {
    fn now(&self) -> Result<HostNanos, EnvError> {
        self.ask(
            || Question::Clock,
            |env| env.now(),
            |now| EnvRecord::Clock(*now),
        )
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        self.ask(
            || Question::Random,
            |env| env.random_u64(),
            |value| EnvRecord::Random(*value),
        )
    }

    fn host_io(&self, request: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        self.ask(
            || Question::HostIo(request.clone()),
            |env| env.host_io(request),
            |reply| EnvRecord::HostIo {
                request: request.clone(),
                reply: reply.clone(),
            },
        )
    }
}

/// Why a record cannot be replayed ([`Replay::new`]).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecordError {
    /// The record holds two completions of one job.
    #[error("file-system job {job} is recorded twice, the second time at entry {entry}")]
    Duplicate {
        /// The job's number.
        job: u64,
        /// Where in the record the second completion is, from zero.
        entry: usize,
    },
}

/// How a replay left its record ([`Replay::check`]).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReplayError {
    /// The replay submitted a job the record holds no completion for.
    #[error("the replay ran file-system job {job}, which the recorded run never finished")]
    Unrecorded {
        /// The job's number.
        job: u64,
    },
    /// A job asked the environment for something other than what the recorded job asked.
    #[error("file-system job {job} read differently from the recorded run: {error}")]
    Diverged {
        /// The job's number.
        job: u64,
        /// What the replay environment said.
        error: EnvError,
    },
    /// The replay has not submitted every job the record holds.
    #[error(
        "the replay has not reached file-system job {job}, the next of the {remaining} jobs the \
         recorded run finished and the replay has not"
    )]
    Unfinished {
        /// The number of the next job recorded.
        job: u64,
        /// How many recorded jobs the replay has not finished.
        remaining: usize,
    },
}

/// What a [`Replay`] holds.
struct Replaying {
    /// What is left of the record, in the order it was finished.
    record: VecDeque<Completed>,
    /// Jobs submitted and waiting for their turn.
    waiting: BTreeMap<JobIndex, Job>,
    next: u64,
    /// The first divergence.
    error: Option<ReplayError>,
}

impl Replaying {
    /// Runs and finishes each job whose turn has come, in the record's order, until the next
    /// one recorded has not been submitted yet.
    fn run_due(&mut self) {
        // Bounded by the record: each turn takes one entry from it.
        while let Some(due) = self.record.front().map(|entry| entry.job) {
            let Some(job) = self.waiting.remove(&due) else {
                return;
            };
            let Some(entry) = self.record.pop_front() else {
                return;
            };
            let env = Replayed::new(entry);
            let finish = job(&env);
            if let Err(error) = env.check() {
                self.error.get_or_insert(ReplayError::Diverged {
                    job: due.raw(),
                    error,
                });
            }
            finish();
        }
    }
}

/// An executor that replays a record of finished jobs (module docs, "Replaying").
pub struct Replay {
    state: Mutex<Replaying>,
}

impl Replay {
    /// A replay of `record`, as [`Workers::take_record`] gave it.
    ///
    /// # Errors
    ///
    /// [`RecordError::Duplicate`] at the first entry whose job an earlier entry holds: a job
    /// is finished once, and a second entry would wait in front of the jobs after it forever.
    pub fn new(record: Vec<Completed>) -> Result<Self, RecordError> {
        let mut seen = BTreeSet::new();
        for (entry, completed) in record.iter().enumerate() {
            if !seen.insert(completed.job) {
                return Err(RecordError::Duplicate {
                    job: completed.job.raw(),
                    entry,
                });
            }
        }
        Ok(Self {
            state: Mutex::new(Replaying {
                record: record.into(),
                waiting: BTreeMap::new(),
                next: 0,
                error: None,
            }),
        })
    }

    /// Whether the replay has followed the record so far and finished all of it: the first
    /// divergence, or else [`ReplayError::Unfinished`] while recorded jobs remain.
    ///
    /// # Errors
    ///
    /// The [`ReplayError`] that says how the replay left the record.
    pub fn check(&self) -> Result<(), ReplayError> {
        let state = lock(&self.state);
        if let Some(error) = &state.error {
            return Err(error.clone());
        }
        match state.record.front() {
            Some(next) => Err(ReplayError::Unfinished {
                job: next.job.raw(),
                remaining: state.record.len(),
            }),
            None => Ok(()),
        }
    }
}

impl Executor for Replay {
    fn submit(&self, job: Job) {
        let mut state = lock(&self.state);
        let Some(index) = take_index(&mut state.next) else {
            tracing::error!("the file-system replay has numbered every job it can: dropped");
            return;
        };
        if !state.record.iter().any(|entry| entry.job == index) {
            tracing::error!(
                job = index.raw(),
                "the replay ran a file-system job the recorded run never finished"
            );
            state
                .error
                .get_or_insert(ReplayError::Unrecorded { job: index.raw() });
            return;
        }
        state.waiting.insert(index, job);
        state.run_due();
    }
}

/// The environment a replayed job runs with: its recorded answers, then its recorded failure.
struct Replayed {
    answers: ReplayEnv,
    count: usize,
    failure: Option<Failure>,
    /// Whether the first call past the recorded answers asked what the recorded failure
    /// answered: `Ok` if it did, else how it differed. Every later call gets the same.
    reached: Mutex<Option<Result<(), EnvError>>>,
}

impl Replayed {
    fn new(entry: Completed) -> Self {
        Self {
            count: entry.answers.len(),
            answers: ReplayEnv::new(entry.answers),
            failure: entry.failure,
            reached: Mutex::new(None),
        }
    }

    /// `answer`, unless the recorded answers have run out where the recorded job failed: then
    /// the recorded failure, as the live job got it, if the job asked what the live job asked
    /// then (`call`, and for host I/O `request`), and otherwise how its question differs.
    fn answer<T>(
        &self,
        call: EnvCall,
        request: Option<&HostIoRequest>,
        answer: Result<T, EnvError>,
    ) -> Result<T, EnvError> {
        match (answer, &self.failure) {
            (Err(EnvError::Exhausted { index, .. }), Some(failure)) if index == self.count => {
                let mut reached = lock(&self.reached);
                let verdict = reached
                    .get_or_insert_with(|| self.compare(failure, call, request))
                    .clone();
                Err(verdict.err().unwrap_or_else(|| failure.error.clone()))
            }
            (answer, _) => answer,
        }
    }

    /// Whether `call` (with `request`, for host I/O) is the question `failure` answered.
    fn compare(
        &self,
        failure: &Failure,
        call: EnvCall,
        request: Option<&HostIoRequest>,
    ) -> Result<(), EnvError> {
        let recorded = failure.asked.call();
        if recorded != call {
            return Err(EnvError::Mismatch {
                index: self.count,
                requested: call,
                recorded,
            });
        }
        match (&failure.asked, request) {
            (Question::HostIo(asked), Some(request)) if asked != request => {
                Err(EnvError::HostIoRequestMismatch { index: self.count })
            }
            _ => Ok(()),
        }
    }

    /// Whether the job asked what the recorded one asked: every recorded answer used, and no
    /// question that differs, up to and including the one the recorded failure answered. A
    /// recorded failure is the end of the recording, not a divergence.
    fn check(&self) -> Result<(), EnvError> {
        match (self.answers.finish(), &self.failure) {
            (Err(EnvError::Exhausted { index, .. }), Some(_)) if index == self.count => {
                lock(&self.reached).clone().unwrap_or(Ok(()))
            }
            // The recorded failure is one more answer, which the job never asked for.
            (Ok(()), Some(_)) => Err(EnvError::Unconsumed {
                index: self.count,
                remaining: 1,
            }),
            (result, _) => result,
        }
    }
}

impl Env for Replayed {
    fn now(&self) -> Result<HostNanos, EnvError> {
        self.answer(EnvCall::Clock, None, self.answers.now())
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        self.answer(EnvCall::Random, None, self.answers.random_u64())
    }

    fn host_io(&self, request: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        self.answer(
            EnvCall::HostIo,
            Some(request),
            self.answers.host_io(request),
        )
    }
}
