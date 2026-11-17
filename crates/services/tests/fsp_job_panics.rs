//! What `fsp-srv`'s executors log when a job or its finish panics: an error naming the job, on
//! the workers' threads and on a replay's submitting thread (`fsp/worker.rs`, "The workers" and
//! "Replaying"). The pool keeping its size and the next read completing are pinned by the unit
//! tests beside the workers (`a_panicking_job_on_every_thread_leaves_the_pool_whole`,
//! `a_job_that_panics_in_a_replay_does_not_unwind_into_its_submitter`); this file pins the logs.
//!
//! # Why this file holds exactly one test
//!
//! The workers log on threads of their own, so the logs are observed with a global subscriber
//! (`tracing::subscriber::set_global_default`), which a process can install once. The checks
//! therefore live in their own test binary, in a single test function that installs it and runs
//! them one after another (see `unknown_command_logs.rs` for why a subscriber shared with other
//! tests in one process miscounts). Do not add a second `#[test]` here; add to the one below.
//!
//! Every wait is bounded by [`WAIT`].

use std::error::Error;
use std::fmt;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use starquake_services::framework::{Executor, Finish};
use starquake_services::fsp::halt::{Halt, Stop};
use starquake_services::fsp::worker::{
    COUNT, Completed, FINISH_PANICKED, JOB_PANICKED, JobIndex, Recording, Replay, Workers,
};
use starquake_trace::{Env, EnvError, HostIoReply, HostIoRequest, HostNanos};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};

/// The longest the test waits for something that should happen.
const WAIT: Duration = Duration::from_secs(10);

/// One event: its level, its message, and its `job` field.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Logged {
    level: Level,
    message: String,
    job: Option<u64>,
}

/// Keeps every event, from every thread.
#[derive(Clone, Default)]
struct Logs(Arc<(Mutex<Vec<Logged>>, Condvar)>);

/// What a check returns: nothing, or whatever stopped it.
type Check<T = ()> = Result<T, Box<dyn Error>>;

impl Logs {
    /// Waits, for at most [`WAIT`], until an event like `wanted` has been logged.
    fn await_logged(&self, wanted: &Logged) -> Check {
        let (logged, changed) = &*self.0;
        let logged = logged.lock().map_err(|_| "the log was poisoned")?;
        let (logged, timeout) = changed
            .wait_timeout_while(logged, WAIT, |logged| !logged.contains(wanted))
            .map_err(|_| "the log was poisoned")?;
        if timeout.timed_out() {
            return Err(format!("{wanted:?} was not logged; these were: {:?}", *logged).into());
        }
        Ok(())
    }
}

/// Reads an event's message and `job` field.
#[derive(Default)]
struct Fields {
    message: String,
    job: Option<u64>,
}

impl Visit for Fields {
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "job" {
            self.job = Some(value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        }
    }
}

impl Subscriber for Logs {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        let (logged, changed) = &*self.0;
        if let Ok(mut logged) = logged.lock() {
            logged.push(Logged {
                level: *event.metadata().level(),
                message: fields.message,
                job: fields.job,
            });
        }
        changed.notify_all();
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

/// An environment no job here reads.
struct Unread;

impl Env for Unread {
    fn now(&self) -> Result<HostNanos, EnvError> {
        Ok(HostNanos(0))
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        Ok(0)
    }

    fn host_io(&self, _: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        Ok(HostIoReply::Unavailable)
    }
}

/// A halt no job here tells: no environment fails.
struct Unexpected;

impl Halt for Unexpected {
    fn halt(&self, stop: Stop) {
        unreachable!("the run was told to stop: {stop}");
    }
}

/// The error a worker or a replay logs for `job` with the message `what`.
fn panicked(what: &str, job: u64) -> Logged {
    Logged {
        level: Level::ERROR,
        message: format!("{what}: its request is dropped"),
        job: Some(job),
    }
}

/// A job that panics before it returns its finish.
fn job_panicking() -> starquake_services::framework::Job {
    Box::new(|_: &dyn Env| std::panic::resume_unwind(Box::new("a job panicked")))
}

/// A job whose finish panics.
fn finish_panicking() -> starquake_services::framework::Job {
    Box::new(|_: &dyn Env| {
        let finish: Finish = Box::new(|| std::panic::resume_unwind(Box::new("a finish panicked")));
        finish
    })
}

/// A job whose finish says it ran. A finish whose test has stopped listening has nobody to tell.
fn job_finishing(finished: &mpsc::Sender<u64>, job: u64) -> starquake_services::framework::Job {
    let finished = finished.clone();
    Box::new(move |_: &dyn Env| {
        let finish: Finish = Box::new(move || {
            if finished.send(job).is_err() {
                // The test stopped listening: nobody to tell.
            }
        });
        finish
    })
}

fn a_panic_on_a_worker_is_logged_with_its_job(logs: &Logs) -> Check {
    let env: Arc<dyn Env> = Arc::new(Unread);
    let envs = vec![env; COUNT.get()];
    let workers = Workers::new(envs, Recording::Off, Arc::new(Unexpected))?;
    let (finished, done) = mpsc::channel();
    workers.submit(job_panicking());
    workers.submit(finish_panicking());
    workers.submit(job_finishing(&finished, 2));
    assert_eq!(done.recv_timeout(WAIT), Ok(2));
    logs.await_logged(&panicked(JOB_PANICKED, 0))?;
    logs.await_logged(&panicked(FINISH_PANICKED, 1))
}

fn a_panic_in_a_replay_is_logged_with_its_job(logs: &Logs) -> Check {
    let record = (0..3)
        .map(|job| Completed {
            job: JobIndex::new(job),
            answers: Vec::new(),
            failure: None,
        })
        .collect();
    let replay = Replay::new(record, Arc::new(Unexpected))?;
    let (finished, done) = mpsc::channel();
    // Each runs on this thread as it is submitted; none unwinds into the submission.
    replay.submit(job_panicking());
    logs.await_logged(&panicked(JOB_PANICKED, 0))?;
    replay.submit(finish_panicking());
    logs.await_logged(&panicked(FINISH_PANICKED, 1))?;
    replay.submit(job_finishing(&finished, 2));
    assert_eq!(done.recv_timeout(WAIT), Ok(2));
    assert_eq!(replay.check(), Ok(()));
    Ok(())
}

#[test]
fn a_job_that_panics_is_logged_as_an_error_naming_it() -> Check {
    let logs = Logs::default();
    tracing::subscriber::set_global_default(logs.clone())?;
    a_panic_on_a_worker_is_logged_with_its_job(&logs)?;
    a_panic_in_a_replay_is_logged_with_its_job(&logs)
}
