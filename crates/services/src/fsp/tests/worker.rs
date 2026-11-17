//! `fsp-srv`'s workers and their replay (`fsp/worker.rs`): reads on real host threads, what they
//! record, and a replay of the record. Every wait is bounded by [`WAIT`], so a broken property
//! fails a test instead of hanging the gate.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Condvar, MutexGuard};
use std::time::Duration;

use starquake_trace::EnvRecord;

use super::*;
use crate::framework::{DEFERRED_REPLY_DROPPED, Finish};
use crate::fsp::halt::{Halt, Stop};
use crate::fsp::worker::{
    COUNT, Completed, Failure, JobIndex, Question, RecordError, Recording, Replay, ReplayError,
    StartError, Workers,
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

/// `env` for every one of [`COUNT`] workers, as a test's workers share it.
fn each(env: Arc<dyn Env>) -> Vec<Arc<dyn Env>> {
    vec![env; COUNT.get()]
}

/// A replay of `record`, which numbers no job twice, whose test expects it never to stop the
/// run.
fn replay_of(record: Vec<Completed>) -> Replay {
    Replay::new(record, unwatched()).unwrap()
}

/// A replay of `record`, which numbers no job twice, and what it tells the run.
fn replay_halting(record: Vec<Completed>) -> (Replay, Arc<Halts>) {
    let (halt, halts) = halts();
    (Replay::new(record, halt).unwrap(), halts)
}

/// The failure the environment answers `asked` with in a test: [`ENV_FAILURE`].
fn env_failure(asked: Question) -> Failure {
    Failure {
        asked,
        error: ENV_FAILURE,
    }
}

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
        let (open, timeout) = self
            .opened
            .wait_timeout_while(open, WAIT, |open| !*open)
            .unwrap();
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

/// A rig whose server's reads run on `COUNT` workers over [`Gated`], keeping their record, whose
/// test expects them never to stop the run.
fn rig_on_workers(recording: Recording) -> (Rig, Arc<Workers>, Arc<Gate>, Receiver<()>) {
    rig_on_workers_telling(recording, unwatched())
}

/// [`rig_on_workers`], whose workers tell `halt` when a read's environment fails it.
fn rig_on_workers_telling(
    recording: Recording,
    halt: Arc<dyn Halt>,
) -> (Rig, Arc<Workers>, Arc<Gate>, Receiver<()>) {
    let host = FakeEnv::new(true);
    let gate = Arc::new(Gate::default());
    let env = Arc::new(Gated {
        host: Arc::clone(&host),
        gate: Arc::clone(&gate),
    });
    let workers = Arc::new(Workers::new(each(env), recording, halt).unwrap());
    let (notify, posted) = posts();
    let executor: Arc<dyn Executor> = Arc::<Workers>::clone(&workers);
    let rig = Rig::over(host, executor, notify);
    (rig, workers, gate, posted)
}

/// A halt to give, and the record of what it is told.
fn halts() -> (Arc<dyn Halt>, Arc<Halts>) {
    let halts = Arc::new(Halts::default());
    let halt: Arc<dyn Halt> = Arc::<Halts>::clone(&halts);
    (halt, halts)
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
    assert_eq!(workers.take_record(), []);
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
    let workers = Workers::new(each(env), Recording::Kept, unwatched()).unwrap();
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
    let replay = Arc::new(replay_of(record));
    let (notify, posted) = posts();
    let executor: Arc<dyn Executor> = Arc::<Replay>::clone(&replay);
    let mut again = Rig::over(FakeEnv::new(true), executor, notify);
    let first = again.hello();
    let second = again.hello();
    again.defer(To::Session(first), READ, read(GATED, 4, 16));
    // The first read waits for the one recorded before it.
    assert!(again.server.take_completed().is_none());
    assert_eq!(
        replay.check(),
        Err(ReplayError::Unfinished {
            job: 1,
            remaining: 2
        })
    );
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
    let (replay, halts) = replay_halting(vec![hello_record(0, 0, 4)]);
    let replay = Arc::new(replay);
    let (notify, posted) = posts();
    let executor: Arc<dyn Executor> = Arc::<Replay>::clone(&replay);
    let mut rig = Rig::over(FakeEnv::new(true), executor, notify);
    let file = rig.hello();
    rig.defer(To::Session(file), READ, read(0, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut rig, 4), (s64(4), four_at(0)));
    assert_eq!(replay.check(), Ok(()));
    assert_eq!(halts.stops(), []);
    rig.defer(To::Session(file), READ, read(0, 4, 16));
    // The run is told to stop when the job is submitted, before its request is dropped.
    let unrecorded = ReplayError::Unrecorded { job: 1 };
    assert_eq!(halts.stops(), [Stop::Replay(unrecorded.clone())]);
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut rig, 4).0, only(DEFERRED_REPLY_DROPPED));
    assert_eq!(replay.check(), Err(unrecorded));
    assert_eq!(
        halts.stops()[0].to_string(),
        "the replay of fsp-srv's reads left its record: the replay ran file-system job 1, which \
         the recorded run never finished"
    );
}

#[test]
fn a_read_that_asks_otherwise_than_the_record_is_a_divergence() {
    let (replay, halts) = replay_halting(vec![hello_record(0, 0, 4)]);
    let replay = Arc::new(replay);
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
    assert_eq!(replay.check(), Err(error.clone()));
    assert_eq!(
        replay.check().unwrap_err().to_string(),
        "file-system job 0 read differently from the recorded run: record 0: the host I/O \
         request differs from the recorded one"
    );
    // The run stops; the request is answered as one the server gave up on, unfinished.
    assert_eq!(halts.stops(), [Stop::Replay(error)]);
    assert_eq!(delivered(&mut rig, 4).0, only(DEFERRED_REPLY_DROPPED));
}

#[test]
fn an_environment_failure_in_a_read_stops_the_run() {
    // Live, on the workers: the run is told to stop with the job and the environment's error,
    // the job's finish is dropped unrun, and the job is recorded with its failure.
    let (halt, live_halts) = halts();
    let (mut live, workers, gate, posted) = rig_on_workers_telling(Recording::Kept, halt);
    gate.open();
    let file = live.hello();
    live.defer(To::Session(file), READ, read(ENV_FAILING, 4, 16));
    await_posts(&posted, 1);
    let answered = delivered(&mut live, 4);
    assert_eq!(answered, (only(DEFERRED_REPLY_DROPPED), vec![STALE; 4]));
    let stop = Stop::Job {
        job: 0,
        error: ENV_FAILURE,
    };
    assert_eq!(live_halts.stops(), std::slice::from_ref(&stop));
    assert_eq!(
        stop.to_string(),
        "the environment failed file-system job 0: recording exhausted at record 7: the replay \
         asked for HostIo"
    );
    // The server's own handlers were not failed.
    assert_eq!(live.halts.stops(), []);
    let record = workers.take_record();
    let offset = u64::try_from(ENV_FAILING).unwrap();
    let expected = Completed {
        job: JobIndex::new(0),
        answers: Vec::new(),
        failure: Some(env_failure(Question::HostIo(hello_read(offset, 4)))),
    };
    assert_eq!(record, [expected]);

    // Replayed: the job reaches its recorded failure and stops the run as the live one did.
    let (replay, halts) = replay_halting(record);
    let replay = Arc::new(replay);
    let (notify, posted) = posts();
    let executor: Arc<dyn Executor> = Arc::<Replay>::clone(&replay);
    let mut again = Rig::over(FakeEnv::new(true), executor, notify);
    let file = again.hello();
    again.defer(To::Session(file), READ, read(ENV_FAILING, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut again, 4), answered);
    assert_eq!(halts.stops(), [stop]);
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
        failure: Some(env_failure(Question::Clock)),
    };
    let expected = [Ok(HostNanos(5)), Err(ENV_FAILURE.clone()), Err(ENV_FAILURE)];
    // Replayed: the answer, then the failure for every later call; the run is told to stop
    // once, as the live run was.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (replay, halts) = replay_halting(vec![failed.clone()]);
    replay.submit(job_timing(3, &seen));
    assert_eq!(*seen.lock().unwrap(), expected);
    assert_eq!(replay.check(), Ok(()));
    let reached = Stop::Job {
        job: 0,
        error: ENV_FAILURE,
    };
    assert_eq!(halts.stops(), [reached]);

    // A job that stops before the recorded failure has not followed the record: the run is
    // told when the job returns.
    let (replay, halts) = replay_halting(vec![failed.clone()]);
    replay.submit(job_timing(1, &Arc::new(Mutex::new(Vec::new()))));
    let error = ReplayError::Diverged {
        job: 0,
        error: EnvError::Unconsumed {
            index: 1,
            remaining: 1,
        },
    };
    assert_eq!(replay.check(), Err(error.clone()));
    assert_eq!(halts.stops(), [Stop::Replay(error)]);

    // A job that asks for more than the record holds, with no failure recorded, has not either:
    // the run is told when the job is answered with the error, once.
    let mut answered = failed;
    answered.failure = None;
    let (replay, halts) = replay_halting(vec![answered]);
    replay.submit(job_timing(2, &Arc::new(Mutex::new(Vec::new()))));
    let error = ReplayError::Diverged {
        job: 0,
        error: EnvError::Exhausted {
            index: 1,
            requested: EnvCall::Clock,
        },
    };
    assert_eq!(replay.check(), Err(error.clone()));
    assert_eq!(halts.stops(), [Stop::Replay(error)]);
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
    // Whether or not the workers keep a record, the failure is held for the job's later calls.
    for recording in [Recording::Kept, Recording::Off] {
        a_live_failure_ends_the_jobs_use_of_the_environment_with(recording);
    }
}

fn a_live_failure_ends_the_jobs_use_of_the_environment_with(recording: Recording) {
    let env = Arc::new(FailsSecond(AtomicUsize::new(0)));
    let env_handle: Arc<dyn Env> = Arc::<FailsSecond>::clone(&env);
    let (halt, halts) = halts();
    let workers = Workers::new(each(env_handle), recording, halt).unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let finished = Arc::new(Finished::default());
    let pushed = Arc::clone(&finished);
    let timing = job_timing(3, &seen);
    let (unrun, dropped) = mpsc::channel();
    let guard = Unrun(Some(unrun), 0);
    workers.submit(Box::new(move |env: &dyn Env| {
        drop(timing(env));
        let finish: Finish = Box::new(move || {
            guard.ran();
            pushed.push(0);
        });
        finish
    }));
    // The finish is dropped without being called.
    assert_eq!(dropped.recv_timeout(WAIT), Ok(0));
    assert!(finished.order().is_empty());
    // The run was told once, at the first failure; the later call did not tell it again.
    let stop = Stop::Job {
        job: 0,
        error: ENV_FAILURE,
    };
    assert_eq!(halts.stops(), [stop], "{recording:?}");
    let expected = [
        Ok(HostNanos(5)),
        Err(ENV_FAILURE.clone()),
        Err(ENV_FAILURE.clone()),
    ];
    assert_eq!(*seen.lock().unwrap(), expected);
    // The third call got the failure without reaching the environment.
    assert_eq!(env.0.load(Ordering::SeqCst), 2);
    let entry = Completed {
        job: JobIndex::new(0),
        answers: vec![EnvRecord::Clock(HostNanos(5))],
        failure: Some(env_failure(Question::Clock)),
    };
    let expected = match recording {
        Recording::Kept => vec![entry],
        Recording::Off => Vec::new(),
    };
    // The entry follows the dropped finish: wait for it, within the bound.
    let mut recorded = Vec::new();
    let started = std::time::Instant::now();
    while recorded.len() < expected.len() && started.elapsed() < WAIT {
        recorded.extend(workers.take_record());
        std::thread::yield_now();
    }
    drop(workers);
    assert_eq!(recorded, expected, "{recording:?}");
}

#[test]
fn a_replay_that_has_not_reached_a_recorded_job_is_unfinished() {
    let record = vec![hello_record(1, 0, 4), hello_record(0, 8, 4)];
    let replay = replay_of(record);
    let finished = Arc::new(Finished::default());
    assert_eq!(
        replay.check(),
        Err(ReplayError::Unfinished {
            job: 1,
            remaining: 2
        })
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
    let workers = Workers::new(each(env_handle), Recording::Kept, unwatched()).unwrap();
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
    stopped
        .recv_timeout(WAIT)
        .expect("the workers did not stop");
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
    let error = StartError::Thread(std::io::Error::other("no threads left"));
    assert_eq!(
        error.to_string(),
        "a file-system worker thread could not be started: no threads left"
    );
    assert_eq!(
        StartError::NoEnvironment.to_string(),
        "the file-system workers were given no environment to run a thread with"
    );
}

#[test]
fn workers_given_no_environment_are_refused_and_one_is_enough() {
    let refused = Workers::new(Vec::new(), Recording::Kept, unwatched());
    assert!(matches!(refused, Err(StartError::NoEnvironment)));
    let env: Arc<dyn Env> = FakeEnv::new(true);
    let one = Workers::new(vec![env], Recording::Kept, unwatched()).unwrap();
    let finished = Arc::new(Finished::default());
    one.submit(job_reading(0, 0, false, &finished));
    assert!(finished.await_count(1, WAIT));
}

/// A job that asks the host for `request` and keeps what it was answered in `seen`; its finish
/// does nothing.
fn job_asking(
    request: HostIoRequest,
    seen: &Arc<Mutex<Vec<Result<HostIoReply, EnvError>>>>,
) -> Job {
    let seen = Arc::clone(seen);
    Box::new(move |env: &dyn Env| {
        seen.lock().unwrap().push(env.host_io(&request));
        let finish: Finish = Box::new(|| {});
        finish
    })
}

/// A job that asks the host for `request` and ignores the answer.
fn job_ignoring(request: HostIoRequest) -> Job {
    job_asking(request, &Arc::new(Mutex::new(Vec::new())))
}

#[test]
fn the_first_of_two_divergences_is_the_one_reported() {
    let mismatch = |job| ReplayError::Diverged {
        job,
        error: EnvError::HostIoRequestMismatch { index: 0 },
    };
    // A read that differs from the record, then a job the record does not hold. The run is
    // told of both, in the order they were found; the first is the one that stops it.
    let (replay, halts) = replay_halting(vec![hello_record(0, 0, 4)]);
    replay.submit(job_ignoring(hello_read(4, 4)));
    replay.submit(job_ignoring(hello_read(0, 4)));
    assert_eq!(replay.check(), Err(mismatch(0)));
    let unrecorded = ReplayError::Unrecorded { job: 1 };
    let told = [Stop::Replay(mismatch(0)), Stop::Replay(unrecorded)];
    assert_eq!(halts.stops(), told);

    // A job the record does not hold, then a read that differs from the record.
    let (replay, halts) = replay_halting(vec![hello_record(1, 0, 4)]);
    replay.submit(job_ignoring(hello_read(0, 4)));
    replay.submit(job_ignoring(hello_read(4, 4)));
    let unrecorded = ReplayError::Unrecorded { job: 0 };
    assert_eq!(replay.check(), Err(unrecorded.clone()));
    let told = [Stop::Replay(unrecorded), Stop::Replay(mismatch(1))];
    assert_eq!(halts.stops(), told);
}

#[test]
fn a_divergence_is_reported_before_the_recorded_jobs_left_unfinished() {
    // Job 0 diverges, and job 1, which the record holds too, is never submitted.
    let (replay, _halts) = replay_halting(vec![hello_record(0, 0, 4), hello_record(1, 0, 4)]);
    replay.submit(job_ignoring(hello_read(4, 4)));
    assert_eq!(
        replay.check(),
        Err(ReplayError::Diverged {
            job: 0,
            error: EnvError::HostIoRequestMismatch { index: 0 },
        })
    );
}

/// A replayed job that asks for `request`, notes how many stops `halts` had been told once it
/// had its answer, and whose finish, if it is ever called, sends its number on `guard`.
fn job_noting_stops(
    request: HostIoRequest,
    halts: &Arc<Halts>,
    guard: Unrun,
) -> (Job, Arc<Mutex<Vec<usize>>>) {
    let noted = Arc::new(Mutex::new(Vec::new()));
    let noting = Arc::clone(&noted);
    let halts = Arc::clone(halts);
    let job: Job = Box::new(move |env: &dyn Env| {
        drop(env.host_io(&request));
        noting.lock().unwrap().push(halts.stops().len());
        let finish: Finish = Box::new(move || guard.ran());
        finish
    });
    (job, noted)
}

#[test]
fn a_replayed_job_stops_the_run_at_its_answer_and_is_left_unfinished() {
    // A divergence, and a recorded failure reached: each is told while the job runs, before it
    // returns, and the job's finish is dropped without being called.
    let failed = Completed {
        job: JobIndex::new(0),
        answers: Vec::new(),
        failure: Some(env_failure(Question::HostIo(hello_read(4, 4)))),
    };
    for (record, request) in [
        (hello_record(0, 0, 4), hello_read(8, 4)),
        (failed, hello_read(4, 4)),
    ] {
        let (replay, halts) = replay_halting(vec![record]);
        let (unrun, dropped) = mpsc::channel();
        let (job, noted) = job_noting_stops(request, &halts, Unrun(Some(unrun), 0));
        replay.submit(job);
        assert_eq!(*noted.lock().unwrap(), [1]);
        assert_eq!(halts.stops().len(), 1);
        assert_eq!(dropped.recv_timeout(WAIT), Ok(0), "the finish was called");
    }
    // A job that follows the record is finished, and nothing is told.
    let (replay, halts) = replay_halting(vec![hello_record(0, 0, 4)]);
    let (unrun, dropped) = mpsc::channel();
    let (job, noted) = job_noting_stops(hello_read(0, 4), &halts, Unrun(Some(unrun), 0));
    replay.submit(job);
    assert_eq!(*noted.lock().unwrap(), [0]);
    assert_eq!(halts.stops(), []);
    assert!(dropped.try_recv().is_err(), "the finish was not called");
}

#[test]
fn a_mismatch_before_a_recorded_failure_is_answered_as_the_mismatch() {
    // The live job read once, then failed on its second read.
    let failed = Completed {
        job: JobIndex::new(0),
        answers: hello_record(0, 0, 4).answers,
        failure: Some(env_failure(Question::HostIo(hello_read(4, 4)))),
    };
    let (replay, halts) = replay_halting(vec![failed]);
    let seen = Arc::new(Mutex::new(Vec::new()));
    // The replayed job's first read differs from the recorded first read.
    replay.submit(job_asking(hello_read(8, 4), &seen));
    let mismatch = EnvError::HostIoRequestMismatch { index: 0 };
    assert_eq!(*seen.lock().unwrap(), [Err(mismatch.clone())]);
    let diverged = ReplayError::Diverged {
        job: 0,
        error: mismatch,
    };
    assert_eq!(replay.check(), Err(diverged.clone()));
    assert_eq!(halts.stops(), [Stop::Replay(diverged)]);
}

#[test]
fn a_replay_that_asks_otherwise_where_the_live_job_failed_is_a_divergence() {
    // The recorded run: the environment failed the read at `ENV_FAILING`.
    let (halt, _live_halts) = halts();
    let (mut live, workers, gate, posted) = rig_on_workers_telling(Recording::Kept, halt);
    gate.open();
    let file = live.hello();
    live.defer(To::Session(file), READ, read(ENV_FAILING, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut live, 4).0, only(DEFERRED_REPLY_DROPPED));
    let record = workers.take_record();

    // The replay reads at another offset where the live job failed.
    let (replay, halts) = replay_halting(record.clone());
    let replay = Arc::new(replay);
    let (notify, posted) = posts();
    let executor: Arc<dyn Executor> = Arc::<Replay>::clone(&replay);
    let mut again = Rig::over(FakeEnv::new(true), executor, notify);
    let file = again.hello();
    again.defer(To::Session(file), READ, read(0, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut again, 4).0, only(DEFERRED_REPLY_DROPPED));
    let mismatch = EnvError::HostIoRequestMismatch { index: 0 };
    let diverged = ReplayError::Diverged {
        job: 0,
        error: mismatch.clone(),
    };
    assert_eq!(replay.check(), Err(diverged.clone()));
    assert_eq!(halts.stops(), [Stop::Replay(diverged)]);

    // Directly: a different request, and then a different kind of call, where the live job
    // failed; each answered with how it differs, every later call alike.
    let (replay, _halts) = replay_halting(record.clone());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let asked = Arc::clone(&seen);
    replay.submit(Box::new(move |env: &dyn Env| {
        let mut asked = asked.lock().unwrap();
        asked.push(env.host_io(&hello_read(0, 4)).map(drop));
        asked.push(env.now().map(drop));
        let finish: Finish = Box::new(|| {});
        finish
    }));
    assert_eq!(
        *seen.lock().unwrap(),
        [Err(mismatch.clone()), Err(mismatch.clone())]
    );
    assert_eq!(
        replay.check(),
        Err(ReplayError::Diverged {
            job: 0,
            error: mismatch,
        })
    );
    let (replay, _halts) = replay_halting(record);
    let seen = Arc::new(Mutex::new(Vec::new()));
    replay.submit(job_timing(1, &seen));
    let kind = EnvError::Mismatch {
        index: 0,
        requested: EnvCall::Clock,
        recorded: EnvCall::HostIo,
    };
    assert_eq!(*seen.lock().unwrap(), [Err(kind.clone())]);
    assert_eq!(
        replay.check(),
        Err(ReplayError::Diverged {
            job: 0,
            error: kind
        })
    );
}

/// An environment whose random number is always the one it holds; it is asked for nothing else.
struct Rolls(u64);

impl Env for Rolls {
    fn now(&self) -> Result<HostNanos, EnvError> {
        unreachable!("the test asks for random numbers only")
    }

    fn random_u64(&self) -> Result<u64, EnvError> {
        Ok(self.0)
    }

    fn host_io(&self, _: &HostIoRequest) -> Result<HostIoReply, EnvError> {
        unreachable!("the test asks for random numbers only")
    }
}

/// A job that draws a random number and whose finish sends it on `drawn`.
fn job_drawing(drawn: &Sender<Result<u64, EnvError>>) -> Job {
    let drawn = drawn.clone();
    Box::new(move |env: &dyn Env| {
        let value = env.random_u64();
        let finish: Finish = Box::new(move || drawn.send(value).unwrap());
        finish
    })
}

#[test]
fn a_random_number_a_worker_drew_is_recorded_and_replayed() {
    const VALUE: u64 = 0x5EED_0123_4567_89AB;
    let workers = Workers::new(each(Arc::new(Rolls(VALUE))), Recording::Kept, unwatched()).unwrap();
    let (send, drawn) = mpsc::channel();
    workers.submit(job_drawing(&send));
    assert_eq!(drawn.recv_timeout(WAIT), Ok(Ok(VALUE)));
    let record = workers.take_record();
    let expected = Completed {
        job: JobIndex::new(0),
        answers: vec![EnvRecord::Random(VALUE)],
        failure: None,
    };
    assert_eq!(record, [expected]);

    let replay = replay_of(record);
    replay.submit(job_drawing(&send));
    assert_eq!(drawn.recv_timeout(WAIT), Ok(Ok(VALUE)));
    assert_eq!(replay.check(), Ok(()));
}

#[test]
fn a_job_that_panics_leaves_its_thread_to_run_the_next() {
    let env: Arc<dyn Env> = FakeEnv::new(true);
    let workers = Workers::new(each(env), Recording::Kept, unwatched()).unwrap();
    let (unrun, dropped) = mpsc::channel();
    let count = u64::try_from(COUNT.get()).unwrap();
    // One job that panics per thread: unfixed, each takes its thread with it.
    for job in 0..count {
        let guard = Unrun(Some(unrun.clone()), job);
        workers.submit(Box::new(move |_: &dyn Env| {
            let _guard = guard;
            std::panic::resume_unwind(Box::new("a job panicked"))
        }));
    }
    // One whose finish panics.
    let guard = Unrun(Some(unrun.clone()), count);
    workers.submit(Box::new(move |_: &dyn Env| {
        let finish: Finish = Box::new(move || {
            let _guard = guard;
            std::panic::resume_unwind(Box::new("a finish panicked"))
        });
        finish
    }));
    // And one after them, which a thread must still be there to run.
    let finished = Arc::new(Finished::default());
    workers.submit(job_reading(0, count + 1, false, &finished));
    assert!(
        finished.await_count(1, WAIT),
        "no thread was left to run the job after the panics"
    );
    let mut unwound: Vec<u64> = (0..=count)
        .map(|_| dropped.recv_timeout(WAIT).unwrap())
        .collect();
    unwound.sort_unstable();
    assert_eq!(unwound, (0..=count).collect::<Vec<_>>());
    // The jobs that panicked are not recorded; the one whose finish panicked is.
    let mut recorded: Vec<u64> = workers
        .take_record()
        .iter()
        .map(|entry| entry.job.raw())
        .collect();
    recorded.sort_unstable();
    assert_eq!(recorded, [count, count + 1]);
}

#[test]
fn a_record_that_holds_a_job_twice_is_refused_at_the_second_entry() {
    let distinct = vec![
        hello_record(1, 0, 4),
        hello_record(0, 8, 4),
        hello_record(2, 0, 4),
    ];
    assert!(Replay::new(distinct.clone(), unwatched()).is_ok());
    let mut twice = distinct.clone();
    twice.push(hello_record(0, 0, 4));
    let error = Replay::new(twice, unwatched()).err().unwrap();
    assert_eq!(error, RecordError::Duplicate { job: 0, entry: 3 });
    assert_eq!(
        error.to_string(),
        "file-system job 0 is recorded twice, the second time at entry 3"
    );
    // At once after the first.
    let next = vec![hello_record(5, 0, 4), hello_record(5, 8, 4)];
    assert_eq!(
        Replay::new(next, unwatched()).err(),
        Some(RecordError::Duplicate { job: 5, entry: 1 })
    );
}

/// Holds every job that arrives until `count` have, within [`WAIT`]: a job held here keeps its
/// thread, so `count` jobs held at once are on `count` threads.
#[derive(Default)]
struct Meeting {
    arrived: Mutex<usize>,
    all: Condvar,
}

impl Meeting {
    fn arrive(&self, count: usize) {
        let mut arrived = self.arrived.lock().unwrap();
        *arrived += 1;
        self.all.notify_all();
        let (arrived, timeout) = self
            .all
            .wait_timeout_while(arrived, WAIT, |arrived| *arrived < count)
            .unwrap();
        drop(arrived);
        assert!(!timeout.timed_out(), "the jobs never met");
    }
}

#[test]
fn a_panicking_job_on_every_thread_leaves_the_pool_whole() {
    let (mut rig, workers, gate, posted) = rig_on_workers(Recording::Kept);
    // One job per thread, held until every thread has one, each then panicking: unfixed, the
    // pool loses every thread.
    let meeting = Arc::new(Meeting::default());
    for _ in 0..COUNT.get() {
        let meeting = Arc::clone(&meeting);
        workers.submit(Box::new(move |_: &dyn Env| {
            meeting.arrive(COUNT.get());
            std::panic::resume_unwind(Box::new("a job panicked"))
        }));
    }
    // The pool keeps its size: as many reads as it has threads are in flight at once.
    let first = rig.hello();
    let second = rig.hello();
    rig.defer(To::Session(first), READ, read(GATED, 4, 16));
    rig.defer(To::Session(second), READ, read(GATED, 4, 16));
    gate.await_waiting(COUNT.get());
    gate.open();
    // And the next reads complete.
    await_posts(&posted, 2);
    assert_eq!(delivered(&mut rig, 4), (s64(4), four_at(GATED)));
    assert_eq!(delivered(&mut rig, 4), (s64(4), four_at(GATED)));
    let count = u64::try_from(COUNT.get()).unwrap();
    let mut recorded: Vec<u64> = workers
        .take_record()
        .iter()
        .map(|entry| entry.job.raw())
        .collect();
    recorded.sort_unstable();
    assert_eq!(recorded, [count, count + 1]);
}

#[test]
fn a_job_that_panics_in_a_replay_does_not_unwind_into_its_submitter() {
    // Three recorded jobs: the replayed first panics in its job, the second in its finish, and
    // the third must still be run and finished.
    let record = vec![
        hello_record(0, 0, 4),
        hello_record(1, 4, 4),
        hello_record(2, 8, 4),
    ];
    let replay = Arc::new(replay_of(record));
    let (notify, posted) = posts();
    let executor: Arc<dyn Executor> = Arc::<Replay>::clone(&replay);
    let mut rig = Rig::over(FakeEnv::new(true), executor, notify);
    let (unrun, dropped) = mpsc::channel();
    let guard = Unrun(Some(unrun.clone()), 0);
    replay.submit(Box::new(move |env: &dyn Env| {
        let _guard = guard;
        env.host_io(&hello_read(0, 4)).unwrap();
        std::panic::resume_unwind(Box::new("a replayed job panicked"))
    }));
    let guard = Unrun(Some(unrun), 1);
    replay.submit(Box::new(move |env: &dyn Env| {
        env.host_io(&hello_read(4, 4)).unwrap();
        let finish: Finish = Box::new(move || {
            let _guard = guard;
            std::panic::resume_unwind(Box::new("a replayed finish panicked"))
        });
        finish
    }));
    // Both ran on this thread, in the record's order, and unwound no further than the replay.
    let unwound = [dropped.recv_timeout(WAIT), dropped.recv_timeout(WAIT)];
    assert_eq!(unwound, [Ok(0), Ok(1)]);
    // The replay's lock is not left poisoned: the next read, submitted as a request, completes.
    let file = rig.hello();
    rig.defer(To::Session(file), READ, read(8, 4, 16));
    await_posts(&posted, 1);
    assert_eq!(delivered(&mut rig, 4), (s64(4), four_at(8)));
    assert_eq!(replay.check(), Ok(()));
}
