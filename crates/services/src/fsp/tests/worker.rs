//! `fsp-srv`'s workers and their replay (`fsp/worker.rs`): reads on real host threads, what they
//! record, and a replay of the record. Every wait is bounded by [`WAIT`], so a broken property
//! fails a test instead of hanging the gate.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Condvar, MutexGuard};
use std::time::Duration;

use starquake_trace::EnvRecord;

use super::*;
use crate::framework::{DEFERRED_REPLY_DROPPED, Finish};
use crate::fsp::worker::{
    COUNT, Completed, JobIndex, Recording, Replay, ReplayError, StartError, Workers,
};

/// The longest a test waits for something that should happen.
const WAIT: Duration = Duration::from_secs(10);
/// How long the first finish of the record-order test holds its window open.
const HOLD: Duration = Duration::from_millis(300);

/// The offset of [`HELLO`] whose reads wait at the [`Gate`], and the one whose reads the host
/// fails, and the one whose reads the environment fails.
const GATED: i64 = 8;
const FAILING: i64 = 16;
const ENV_FAILING: i64 = 24;

/// What the environment answers a read at [`ENV_FAILING`] with.
const ENV_FAILURE: EnvError = EnvError::Exhausted {
    index: 7,
    requested: EnvCall::HostIo,
};

/// Holds back reads until it is opened.
#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    opened: Condvar,
    /// How many reads are waiting at it.
    waiting: Mutex<usize>,
    arrived: Condvar,
}

impl Gate {
    fn pass(&self) {
        *self.waiting.lock().unwrap() += 1;
        self.arrived.notify_all();
        let open = self.open.lock().unwrap();
        let (open, timeout) = self.opened.wait_timeout_while(open, WAIT, |open| !*open).unwrap();
        assert!(*open && !timeout.timed_out(), "the gate was never opened");
    }

    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.opened.notify_all();
    }

    /// Waits until `count` reads are waiting at the gate.
    fn await_waiting(&self, count: usize) {
        let waiting = self.waiting.lock().unwrap();
        let (waiting, timeout) = self
            .arrived
            .wait_timeout_while(waiting, WAIT, |waiting| *waiting < count)
            .unwrap();
        drop(waiting);
        assert!(!timeout.timed_out(), "the reads never reached the gate");
    }
}

/// The workers' environment: the rig's fake host, with reads at [`GATED`] held at a gate, reads
/// at [`FAILING`] failed by the host, and reads at [`ENV_FAILING`] failed by the environment.
struct Gated {
    host: Arc<FakeEnv>,
    gate: Arc<Gate>,
}

impl Env for Gated {
    fn now(&self) -> Result<HostNanos, EnvError> {
        self.host.now()
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        self.host.random_u64()
    }

    fn host_io(&self, request: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        let offset = match HostRequest::decode(&request.payload).unwrap() {
            HostRequest::Read { offset, .. } => i64::try_from(offset).unwrap(),
            _ => -1,
        };
        match offset {
            GATED => self.gate.pass(),
            FAILING => return Ok(HostFailure::Io.reply()),
            ENV_FAILING => return Err(ENV_FAILURE),
            _ => {}
        }
        self.host.host_io(request)
    }
}

/// Sends a message for every completion posted.
struct Posts(Mutex<Sender<()>>);

impl Notify for Posts {
    fn completion_posted(&self) {
        self.0.lock().unwrap().send(()).unwrap();
    }
}

fn posts() -> (Arc<dyn Notify>, Receiver<()>) {
    let (send, receive) = mpsc::channel();
    (Arc::new(Posts(Mutex::new(send))), receive)
}

/// Waits for `count` completions to be posted.
fn await_posts(posted: &Receiver<()>, count: usize) {
    for _ in 0..count {
        posted
            .recv_timeout(WAIT)
            .expect("a completion was not posted");
    }
}

/// The host request a read of `len` bytes of [`HELLO`] at `offset` makes.
fn hello_read(offset: u64, len: u64) -> HostIoRequest {
    HostRequest::Read {
        mount: Mount::SdCard,
        path: HELLO_PATH.to_vec(),
        offset,
        len,
    }
    .to_host_io()
}

/// What a read of `len` bytes of [`HELLO`] at `offset` records.
fn hello_record(job: u64, offset: u64, len: u64) -> Completed {
    let start = usize::try_from(offset).unwrap();
    let end = start + usize::try_from(len).unwrap();
    Completed {
        job: JobIndex::new(job),
        answers: vec![EnvRecord::HostIo {
            request: hello_read(offset, len),
            reply: HostIoReply::Data(HELLO[start..end].to_vec()),
        }],
        failure: None,
    }
}

/// A rig whose server's reads run on `COUNT` workers over [`Gated`], keeping their record.
fn rig_on_workers(recording: Recording) -> (Rig, Arc<Workers>, Arc<Gate>, Receiver<()>) {
    let host = FakeEnv::new(true);
    let gate = Arc::new(Gate::default());
    let env = Arc::new(Gated {
        host: Arc::clone(&host),
        gate: Arc::clone(&gate),
    });
    let workers = Arc::new(Workers::new(COUNT, env, recording).unwrap());
    let (notify, posted) = posts();
    let executor: Arc<dyn Executor> = Arc::<Workers>::clone(&workers);
    let rig = Rig::over(host, executor, notify);
    (rig, workers, gate, posted)
}

/// Delivers the oldest completion and returns its reply and the first `len` bytes of the
/// caller's buffer.
fn delivered(rig: &mut Rig, len: usize) -> (Answered, Vec<u8>) {
    let reply = rig.deliver(SessionKind::Session);
    (reply, rig.kernel.memory(DATA_AT, len).unwrap().to_vec())
}

/// [`HELLO`]'s four bytes from `offset`.
fn four_at(offset: i64) -> Vec<u8> {
    let start = usize::try_from(offset).unwrap();
    HELLO[start..start + 4].to_vec()
}

#[test]
fn the_workers_are_two() {
    assert_eq!(COUNT.get(), 2);
}

#[test]
fn two_reads_in_flight_complete_independently() {
    let (mut rig, workers, gate, posted) = rig_on_workers(Recording::Kept);
    let first = rig.hello();
    let second = rig.hello();
    rig.defer(To::Session(first), READ, read(GATED, 4, 16));
    gate.await_waiting(1);
    rig.defer(To::Session(second), READ, read(0, 4, 16));
    // The second completes while the first waits on the host.
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut rig, 4), (s64(4), four_at(0)));
    assert!(rig.server.take_completed().is_none());

    gate.open();
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut rig, 4), (s64(4), four_at(GATED)));
    assert_eq!(
        workers.take_record(),
        [hello_record(1, 0, 4), hello_record(0, 8, 4)]
    );
}

#[test]
fn a_read_that_fails_fails_only_its_own_request() {
    let (mut rig, _workers, gate, posted) = rig_on_workers(Recording::Kept);
    gate.open();
    let first = rig.hello();
    let second = rig.hello();
    rig.defer(To::Session(first), READ, read(FAILING, 4, 16));
    await_posts(&posted, 1);
    rig.defer(To::Session(second), READ, read(0, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(
        delivered(&mut rig, 4),
        (only(result::HOST_IO_FAILED), vec![STALE; 4])
    );
    assert_eq!(delivered(&mut rig, 4), (s64(4), four_at(0)));
    // The failing file reads again on its session, and fails again.
    rig.defer(To::Session(first), READ, read(FAILING, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut rig, 4).0, only(result::HOST_IO_FAILED));
}

#[test]
fn with_recording_off_nothing_is_kept() {
    let (mut rig, workers, gate, posted) = rig_on_workers(Recording::Off);
    gate.open();
    let file = rig.hello();
    rig.defer(To::Session(file), READ, read(0, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut rig, 4), (s64(4), four_at(0)));
    assert!(workers.take_record().is_empty());
}

/// What a test's finishes did, in order: each pushes its job's number.
#[derive(Default)]
struct Finished {
    order: Mutex<Vec<u64>>,
    pushed: Condvar,
}

impl Finished {
    /// Pushes `job`; how many have been pushed since.
    fn push(&self, job: u64) -> usize {
        let mut order = self.order.lock().unwrap();
        order.push(job);
        self.pushed.notify_all();
        order.len()
    }

    fn order(&self) -> MutexGuard<'_, Vec<u64>> {
        self.order.lock().unwrap()
    }

    /// Waits up to `limit` for `count` finishes; whether they came.
    fn await_count(&self, count: usize, limit: Duration) -> bool {
        let order = self.order();
        let (order, timeout) = self
            .pushed
            .wait_timeout_while(order, limit, |order| order.len() < count)
            .unwrap();
        drop(order);
        !timeout.timed_out()
    }
}

/// A job that reads [`HELLO`] at `offset` through its environment, and whose finish pushes `job`
/// to `finished`. With `hold`, the first finish to push then waits [`HOLD`] before
/// it returns: the window in which a second finish could overtake its record.
fn job_reading(offset: u64, job: u64, hold: bool, finished: &Arc<Finished>) -> Job {
    let finished = Arc::clone(finished);
    Box::new(move |env: &dyn Env| {
        env.host_io(&hello_read(offset, 4)).unwrap();
        let finish: Finish = Box::new(move || {
            if finished.push(job) == 1 && hold {
                // The whole window, so a second finish that runs in it is recorded within it.
                std::thread::sleep(HOLD);
            }
        });
        finish
    })
}

#[test]
fn the_record_holds_completions_in_the_order_they_reached_the_mailbox() {
    // Both jobs wait at the gate, then race to finish. The first finish to post waits for the
    // other before it returns, and the other must not post and be recorded in between: a job is
    // finished and recorded under one lock.
    let host = FakeEnv::new(true);
    let gate = Arc::new(Gate::default());
    let env = Arc::new(Gated {
        host,
        gate: Arc::clone(&gate),
    });
    let workers = Workers::new(COUNT, env, Recording::Kept).unwrap();
    let finished = Arc::new(Finished::default());
    let offset = u64::try_from(GATED).unwrap();
    workers.submit(job_reading(offset, 0, true, &finished));
    workers.submit(job_reading(offset, 1, true, &finished));
    gate.await_waiting(2);
    gate.open();
    assert!(finished.await_count(2, WAIT), "the jobs were not finished");
    let order = finished.order().clone();
    // The last record follows its finish: wait for it, within the bound.
    let mut recorded = Vec::new();
    let started = std::time::Instant::now();
    while recorded.len() < 2 && started.elapsed() < WAIT {
        recorded.extend(workers.take_record().iter().map(|entry| entry.job.raw()));
        std::thread::yield_now();
    }
    assert_eq!(recorded, order);
}

#[test]
fn a_replay_delivers_the_reads_in_the_recorded_order() {
    // The recorded run: the second read completes first.
    let (mut live, workers, gate, posted) = rig_on_workers(Recording::Kept);
    let first = live.hello();
    let second = live.hello();
    live.defer(To::Session(first), READ, read(GATED, 4, 16));
    gate.await_waiting(1);
    live.defer(To::Session(second), READ, read(0, 4, 16));
    await_posts(&posted, 1);
    gate.open();
    await_posts(&posted, 1);
    let live_replies = [delivered(&mut live, 4), delivered(&mut live, 4)];
    assert_eq!(live_replies[0].1, four_at(0));
    let record = workers.take_record();

    // The replay: the same requests, in the same order, with no host behind the reads.
    let replay = Arc::new(Replay::new(record));
    let (notify, posted) = posts();
    let executor: Arc<dyn Executor> = Arc::<Replay>::clone(&replay);
    let mut again = Rig::over(FakeEnv::new(true), executor, notify);
    let first = again.hello();
    let second = again.hello();
    again.defer(To::Session(first), READ, read(GATED, 4, 16));
    // The first read waits for the one recorded before it.
    assert!(again.server.take_completed().is_none());
    assert_eq!(replay.check(), Err(ReplayError::Unfinished { job: 1, remaining: 2 }));
    again.defer(To::Session(second), READ, read(0, 4, 16));
    await_posts(&posted, 2);
    let replayed = [delivered(&mut again, 4), delivered(&mut again, 4)];
    assert_eq!(replayed, live_replies);
    assert_eq!(replay.check(), Ok(()));
    let asked = again.env.asked();
    assert!(
        !asked
            .iter()
            .any(|request| matches!(request, HostRequest::Read { .. })),
        "the replay read the host: {asked:?}"
    );
}

#[test]
fn a_job_the_record_does_not_hold_is_a_divergence_and_its_request_is_dropped() {
    let replay = Arc::new(Replay::new(vec![hello_record(0, 0, 4)]));
    let (notify, posted) = posts();
    let executor: Arc<dyn Executor> = Arc::<Replay>::clone(&replay);
    let mut rig = Rig::over(FakeEnv::new(true), executor, notify);
    let file = rig.hello();
    rig.defer(To::Session(file), READ, read(0, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut rig, 4), (s64(4), four_at(0)));
    assert_eq!(replay.check(), Ok(()));
    rig.defer(To::Session(file), READ, read(0, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut rig, 4).0, only(DEFERRED_REPLY_DROPPED));
    assert_eq!(replay.check(), Err(ReplayError::Unrecorded { job: 1 }));
}

#[test]
fn a_read_that_asks_otherwise_than_the_record_is_a_divergence() {
    let replay = Arc::new(Replay::new(vec![hello_record(0, 0, 4)]));
    let (notify, posted) = posts();
    let executor: Arc<dyn Executor> = Arc::<Replay>::clone(&replay);
    let mut rig = Rig::over(FakeEnv::new(true), executor, notify);
    let file = rig.hello();
    rig.defer(To::Session(file), READ, read(4, 4, 16));
    await_posts(&posted, 1);
    let error = ReplayError::Diverged {
        job: 0,
        error: EnvError::HostIoRequestMismatch { index: 0 },
    };
    assert_eq!(replay.check(), Err(error));
    assert_eq!(
        replay.check().unwrap_err().to_string(),
        "file-system job 0 read differently from the recorded run: record 0: the host I/O \
         request differs from the recorded one"
    );
    // Not the guest's concern, but answered: the environment's error is a broken host.
    assert_eq!(delivered(&mut rig, 4).0, only(result::HOST_IO_FAILED));
}

#[test]
fn a_failure_the_environment_gave_is_recorded_and_replayed() {
    let (mut live, workers, gate, posted) = rig_on_workers(Recording::Kept);
    gate.open();
    let file = live.hello();
    live.defer(To::Session(file), READ, read(ENV_FAILING, 4, 16));
    await_posts(&posted, 1);
    let reply = delivered(&mut live, 4);
    assert_eq!(reply.0, only(result::HOST_IO_FAILED));
    let record = workers.take_record();
    let expected = Completed {
        job: JobIndex::new(0),
        answers: Vec::new(),
        failure: Some(ENV_FAILURE),
    };
    assert_eq!(record, [expected]);

    let replay = Arc::new(Replay::new(record));
    let (notify, posted) = posts();
    let executor: Arc<dyn Executor> = Arc::<Replay>::clone(&replay);
    let mut again = Rig::over(FakeEnv::new(true), executor, notify);
    let file = again.hello();
    again.defer(To::Session(file), READ, read(ENV_FAILING, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut again, 4), reply);
    assert_eq!(replay.check(), Ok(()));
}

/// A job that asks the environment for the time `asks` times, keeping the answers in `seen`;
/// its finish does nothing.
fn job_timing(asks: usize, seen: &Arc<Mutex<Vec<Result<HostNanos, EnvError>>>>) -> Job {
    let seen = Arc::clone(seen);
    Box::new(move |env: &dyn Env| {
        for _ in 0..asks {
            seen.lock().unwrap().push(env.now());
        }
        let finish: Finish = Box::new(|| {});
        finish
    })
}

#[test]
fn a_recorded_failure_ends_the_jobs_answers_live_and_in_replay() {
    let failed = Completed {
        job: JobIndex::new(0),
        answers: vec![EnvRecord::Clock(HostNanos(5))],
        failure: Some(ENV_FAILURE),
    };
    let expected = [Ok(HostNanos(5)), Err(ENV_FAILURE.clone()), Err(ENV_FAILURE)];
    // Replayed: the answer, then the failure for every later call.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let replay = Replay::new(vec![failed.clone()]);
    replay.submit(job_timing(3, &seen));
    assert_eq!(*seen.lock().unwrap(), expected);
    assert_eq!(replay.check(), Ok(()));

    // A job that stops before the recorded failure has not followed the record.
    let replay = Replay::new(vec![failed.clone()]);
    replay.submit(job_timing(1, &Arc::new(Mutex::new(Vec::new()))));
    let error = ReplayError::Diverged {
        job: 0,
        error: EnvError::Unconsumed {
            index: 1,
            remaining: 1,
        },
    };
    assert_eq!(replay.check(), Err(error));

    // A job that asks for more than the record holds, with no failure recorded, has not either.
    let mut answered = failed;
    answered.failure = None;
    let replay = Replay::new(vec![answered]);
    replay.submit(job_timing(2, &Arc::new(Mutex::new(Vec::new()))));
    let error = ReplayError::Diverged {
        job: 0,
        error: EnvError::Exhausted {
            index: 1,
            requested: EnvCall::Clock,
        },
    };
    assert_eq!(replay.check(), Err(error));
}

/// An environment that answers the time once, then fails, and counts the calls that reach it.
struct FailsSecond(AtomicUsize);

impl Env for FailsSecond {
    fn now(&self) -> Result<HostNanos, EnvError> {
        match self.0.fetch_add(1, Ordering::SeqCst) {
            0 => Ok(HostNanos(5)),
            _ => Err(ENV_FAILURE),
        }
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        unreachable!("the test asks for the time only")
    }

    fn host_io(&self, _: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        unreachable!("the test asks for the time only")
    }
}

#[test]
fn a_live_failure_ends_the_jobs_use_of_the_environment() {
    let env = Arc::new(FailsSecond(AtomicUsize::new(0)));
    let env_handle: Arc<dyn Env> = Arc::<FailsSecond>::clone(&env);
    let workers = Workers::new(COUNT, env_handle, Recording::Kept).unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let finished = Arc::new(Finished::default());
    let pushed = Arc::clone(&finished);
    let timing = job_timing(3, &seen);
    workers.submit(Box::new(move |env: &dyn Env| {
        drop(timing(env));
        let finish: Finish = Box::new(move || {
            pushed.push(0);
        });
        finish
    }));
    assert!(finished.await_count(1, WAIT));
    let expected = [Ok(HostNanos(5)), Err(ENV_FAILURE.clone()), Err(ENV_FAILURE.clone())];
    assert_eq!(*seen.lock().unwrap(), expected);
    // The third call got the failure without reaching the environment.
    assert_eq!(env.0.load(Ordering::SeqCst), 2);
    let entry = Completed {
        job: JobIndex::new(0),
        answers: vec![EnvRecord::Clock(HostNanos(5))],
        failure: Some(ENV_FAILURE),
    };
    assert_eq!(workers.take_record(), [entry]);
}

#[test]
fn a_replay_that_has_not_reached_a_recorded_job_is_unfinished() {
    let record = vec![hello_record(1, 0, 4), hello_record(0, 8, 4)];
    let replay = Replay::new(record);
    let finished = Arc::new(Finished::default());
    assert_eq!(
        replay.check(),
        Err(ReplayError::Unfinished { job: 1, remaining: 2 })
    );
    replay.submit(job_reading(8, 0, false, &finished));
    assert!(finished.order().is_empty());
    assert_eq!(
        replay.check().unwrap_err().to_string(),
        "the replay has not reached file-system job 1, the next of the 2 jobs the recorded run \
         finished and the replay has not"
    );
    replay.submit(job_reading(0, 1, false, &finished));
    assert_eq!(*finished.order(), [1, 0]);
    assert_eq!(replay.check(), Ok(()));
    assert_eq!(
        ReplayError::Unrecorded { job: 3 }.to_string(),
        "the replay ran file-system job 3, which the recorded run never finished"
    );
}

/// Says whether the job it was moved into was dropped without running.
struct Unrun(Option<Sender<u64>>, u64);

impl Unrun {
    fn ran(mut self) {
        self.0 = None;
    }
}

impl Drop for Unrun {
    fn drop(&mut self) {
        if let Some(dropped) = self.0.take() {
            dropped.send(self.1).unwrap();
        }
    }
}

#[test]
fn dropping_the_workers_with_reads_in_flight_leaves_no_thread_behind() {
    let gate = Arc::new(Gate::default());
    let env = Arc::new(Gated {
        host: FakeEnv::new(true),
        gate: Arc::clone(&gate),
    });
    let env_handle: Arc<dyn Env> = Arc::<Gated>::clone(&env);
    let workers = Workers::new(COUNT, env_handle, Recording::Kept).unwrap();
    let finished = Arc::new(Finished::default());
    let (unrun, dropped) = mpsc::channel();
    // One read in flight per worker, and one more queued behind them.
    let count = u64::try_from(COUNT.get()).unwrap();
    let offset = u64::try_from(GATED).unwrap();
    for job in 0..=count {
        let guard = Unrun(Some(unrun.clone()), job);
        let reading = job_reading(offset, job, false, &finished);
        workers.submit(Box::new(move |env: &dyn Env| {
            guard.ran();
            reading(env)
        }));
    }
    gate.await_waiting(COUNT.get());
    let (done, stopped) = mpsc::channel();
    let dropping = std::thread::spawn(move || {
        drop(workers);
        done.send(()).unwrap();
    });
    // The queued job is dropped without running; the drop waits for the ones in flight.
    assert_eq!(dropped.recv_timeout(WAIT), Ok(count));
    assert!(stopped.try_recv().is_err(), "the drop did not wait");
    gate.open();
    stopped.recv_timeout(WAIT).expect("the workers did not stop");
    dropping.join().unwrap();
    // The jobs in flight ran to their end and were finished, and nothing else ran.
    let mut order = finished.order().clone();
    order.sort_unstable();
    assert_eq!(order, (0..count).collect::<Vec<_>>());
    assert!(dropped.try_recv().is_err());
    // Every thread is gone: none holds the environment any more.
    assert_eq!(Arc::strong_count(&env), 1, "a worker thread is left");
}

#[test]
fn a_job_numbered_past_the_last_index_is_dropped() {
    let (mut rig, workers, gate, posted) = rig_on_workers(Recording::Kept);
    gate.open();
    workers.number_from(u64::MAX - 1);
    let file = rig.hello();
    rig.defer(To::Session(file), READ, read(0, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut rig, 4), (s64(4), four_at(0)));
    rig.defer(To::Session(file), READ, read(0, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut rig, 4).0, only(DEFERRED_REPLY_DROPPED));
    let recorded: Vec<u64> = workers
        .take_record()
        .iter()
        .map(|entry| entry.job.raw())
        .collect();
    assert_eq!(recorded, [u64::MAX - 1]);
}

#[test]
fn a_start_error_says_so_in_plain_words() {
    let error = StartError::from_io(std::io::Error::other("no threads left"));
    assert_eq!(
        error.to_string(),
        "a file-system worker thread could not be started: no threads left"
    );
}
