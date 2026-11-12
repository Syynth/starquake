//! [`RequestQueue`]: which of a server's requests may start, and in what order.
//!
//! The rules are the service spec's (`docs/specs/services.md`, "Concurrency") and the kernel
//! spec's (`docs/specs/kernel.md`, "IPC": "a request starts when its session has nothing in
//! flight and its server has a free worker"):
//!
//! - **One request in flight per session.** A session with a request running or deferred starts
//!   no second one; further senders wait.
//! - **A server has a fixed number of workers**, [`DEFAULT_WORKERS`] unless it says otherwise, and
//!   that many requests run at a time.
//! - **A deferred request does not hold a worker.** It stays in flight on its session until it
//!   completes, and the worker goes to the next session.
//! - **Sessions take turns in the order they became ready**: a session joins the back of the
//!   line when its first sender starts waiting, or when its request in flight ends with senders
//!   still waiting. A sender that arrives while the line is not empty waits behind it, free
//!   worker or not.
//!
//! The queue holds sessions and counts, not requests: a sender that has to wait waits as a guest
//! thread, and the thread and its message buffer are the kernel's to keep. When [`RequestQueue::start_next`]
//! names a session, the kernel starts that session's oldest waiting sender.
//!
//! No reference source is ported. Horizon's server ends keep a request list per session
//! (`eden/src/core/hle/kernel/k_server_session.h`), and Eden's HLE servers take requests on host
//! threads (`eden/src/core/hle/service/server_manager.cpp`); the turn-taking here is starquake's
//! own design from the two specs above, and nothing in it is visible to a guest except the order
//! in which waiting senders are answered. That order is a guess at what a console's single
//! server loop does; a trace of two sessions queueing on one console service would confirm it.
//!
//! The whole structure is plain data and derives `serde`, because requests in flight are part of
//! a server's snapshot. Deserialising validates every invariant ([`QueueLayoutError`]).

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};
use starquake_kernel::id::ObjectId;

/// The number of workers a server has unless it declares more: one, so one handler of the
/// server runs at a time (`docs/specs/services.md`, decision 4).
pub const DEFAULT_WORKERS: u8 = 1;

/// What becomes of a request given to [`RequestQueue::submit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// The request starts now: it is running, and holds a worker until it finishes or defers.
    Start,
    /// The sender waits. [`RequestQueue::start_next`] names its session when its turn comes.
    Queued,
}

/// Why a queue operation was refused. None of these is reachable from a guest request; they mean
/// the caller's own bookkeeping is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum QueueError {
    /// A queue was asked for with no workers.
    #[error("a server needs at least one worker")]
    NoWorkers,
    /// The session has no request running.
    #[error("session {0:?} has no request running")]
    NotRunning(ObjectId),
    /// The session has no deferred request.
    #[error("session {0:?} has no deferred request")]
    NotDeferred(ObjectId),
    /// The session has no sender waiting.
    #[error("session {0:?} has no sender waiting")]
    NotWaiting(ObjectId),
    /// The count of senders waiting on the session would pass `u32::MAX`.
    #[error("session {0:?} has too many senders waiting")]
    TooManyWaiting(ObjectId),
}

/// Why serialised queue data was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum QueueLayoutError {
    /// The queue has no workers.
    #[error("the queue has no workers")]
    NoWorkers,
    /// The running count is not the number of sessions marked running.
    #[error("{recorded} requests recorded as running, {counted} sessions marked running")]
    RunningCount {
        /// The count in the data.
        recorded: u8,
        /// Sessions whose request in flight is marked running.
        counted: usize,
    },
    /// More requests are running than the queue has workers.
    #[error("{running} requests running on {workers} workers")]
    TooManyRunning {
        /// Requests running.
        running: u8,
        /// The queue's workers.
        workers: u8,
    },
    /// A session is listed with nothing in flight and nobody waiting.
    #[error("session {0:?} is listed with nothing in flight and nobody waiting")]
    IdleSession(ObjectId),
    /// The line of ready sessions and the sessions disagree: a session in the line is not
    /// ready, is there twice, or a ready session is missing from it.
    #[error("the line of ready sessions disagrees with session {0:?}")]
    Ready(ObjectId),
}

/// A session's request in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Flight {
    /// A handler is running it, on one of the server's workers.
    Running,
    /// The handler returned without a reply; the request waits for its reply token.
    Deferred,
}

/// What the queue knows of one session. A session with nothing in flight and nobody waiting has
/// no entry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Line {
    flight: Option<Flight>,
    /// Senders waiting behind the request in flight, or for a worker.
    waiting: u32,
}

impl Line {
    /// Whether the session waits only for a worker.
    fn is_ready(self) -> bool {
        self.flight.is_none() && self.waiting > 0
    }
}

/// A server's request queue. See the [module docs](self).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "QueueData")]
pub struct RequestQueue {
    workers: u8,
    /// Requests running now. Never more than `workers`.
    running: u8,
    /// Sessions that wait only for a worker, the one that has waited longest first.
    ready: VecDeque<ObjectId>,
    lines: BTreeMap<ObjectId, Line>,
}

/// The serialised form of a [`RequestQueue`], and what deserialisation validates.
#[derive(Deserialize)]
struct QueueData {
    workers: u8,
    running: u8,
    ready: VecDeque<ObjectId>,
    lines: BTreeMap<ObjectId, Line>,
}

impl TryFrom<QueueData> for RequestQueue {
    type Error = QueueLayoutError;

    fn try_from(data: QueueData) -> Result<Self, Self::Error> {
        let queue = Self {
            workers: data.workers,
            running: data.running,
            ready: data.ready,
            lines: data.lines,
        };
        queue.validate()?;
        Ok(queue)
    }
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestQueue {
    /// An empty queue with [`DEFAULT_WORKERS`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            workers: DEFAULT_WORKERS,
            running: 0,
            ready: VecDeque::new(),
            lines: BTreeMap::new(),
        }
    }

    /// An empty queue for a server that runs `workers` requests at a time.
    pub fn with_workers(workers: u8) -> Result<Self, QueueError> {
        if workers == 0 {
            return Err(QueueError::NoWorkers);
        }
        Ok(Self {
            workers,
            ..Self::new()
        })
    }

    /// How many requests the server runs at a time.
    #[must_use]
    pub fn workers(&self) -> u8 {
        self.workers
    }

    /// How many requests are running now.
    #[must_use]
    pub fn running(&self) -> u8 {
        self.running
    }

    /// Whether the session's request in flight is running.
    #[must_use]
    pub fn is_running(&self, session: ObjectId) -> bool {
        self.flight(session) == Some(Flight::Running)
    }

    /// Whether the session's request in flight is deferred.
    #[must_use]
    pub fn is_deferred(&self, session: ObjectId) -> bool {
        self.flight(session) == Some(Flight::Deferred)
    }

    /// How many senders wait on the session.
    #[must_use]
    pub fn waiting(&self, session: ObjectId) -> u32 {
        self.lines.get(&session).map_or(0, |line| line.waiting)
    }

    /// The sessions with a request in flight or a sender waiting, in id order.
    pub fn sessions(&self) -> impl Iterator<Item = ObjectId> {
        self.lines.keys().copied()
    }

    /// The sessions whose request in flight is deferred, in id order.
    pub fn deferred(&self) -> impl Iterator<Item = ObjectId> {
        let deferred = |(_, line): &(&ObjectId, &Line)| line.flight == Some(Flight::Deferred);
        self.lines.iter().filter(deferred).map(|(id, _)| *id)
    }

    fn flight(&self, session: ObjectId) -> Option<Flight> {
        self.lines.get(&session).and_then(|line| line.flight)
    }

    /// A guest thread sent a request on `session`. [`Admission::Start`] if the session has
    /// nothing in flight and no sender waiting, the server has a free worker, and no other
    /// session is ready (a free worker is for the session at the front of the line, which
    /// [`RequestQueue::start_next`] has yet to name); otherwise the sender is counted as
    /// waiting.
    pub fn submit(&mut self, session: ObjectId) -> Result<Admission, QueueError> {
        let free = self.running < self.workers && self.ready.is_empty();
        let line = self.lines.entry(session).or_default();
        if line.flight.is_none() && line.waiting == 0 && free {
            line.flight = Some(Flight::Running);
            self.running = self.running.saturating_add(1);
            return Ok(Admission::Start);
        }
        let waiting = line.waiting.checked_add(1);
        line.waiting = waiting.ok_or(QueueError::TooManyWaiting(session))?;
        if line.is_ready() && line.waiting == 1 {
            self.ready.push_back(session);
        }
        Ok(Admission::Queued)
    }

    /// A sender waiting on `session` gave up (its thread was cancelled or terminated): one
    /// fewer waits.
    pub fn withdraw(&mut self, session: ObjectId) -> Result<(), QueueError> {
        let line = self.lines.get_mut(&session);
        let line = line.filter(|line| line.waiting > 0);
        let line = line.ok_or(QueueError::NotWaiting(session))?;
        line.waiting = line.waiting.saturating_sub(1);
        let (waiting, in_flight) = (line.waiting, line.flight.is_some());
        if waiting == 0 {
            self.ready.retain(|ready| *ready != session);
            if !in_flight {
                self.lines.remove(&session);
            }
        }
        Ok(())
    }

    /// The session whose oldest waiting sender starts now, if a worker is free and a session is
    /// ready. Its request is then running, as after an [`Admission::Start`]. Call it after
    /// [`RequestQueue::finish`], [`RequestQueue::defer`], [`RequestQueue::complete`] and
    /// [`RequestQueue::close`], until it returns `None`.
    pub fn start_next(&mut self) -> Option<ObjectId> {
        if self.running >= self.workers {
            return None;
        }
        let session = self.ready.pop_front()?;
        let line = self.lines.get_mut(&session)?;
        line.waiting = line.waiting.saturating_sub(1);
        line.flight = Some(Flight::Running);
        self.running = self.running.saturating_add(1);
        Some(session)
    }

    /// The running request on `session` was answered: its worker is free and the session has
    /// nothing in flight.
    pub fn finish(&mut self, session: ObjectId) -> Result<(), QueueError> {
        self.stop_running(session)?;
        self.landed(session);
        Ok(())
    }

    /// The running request on `session` was deferred: its worker is free, and the request stays
    /// in flight until [`RequestQueue::complete`].
    pub fn defer(&mut self, session: ObjectId) -> Result<(), QueueError> {
        self.stop_running(session)?;
        if let Some(line) = self.lines.get_mut(&session) {
            line.flight = Some(Flight::Deferred);
        }
        Ok(())
    }

    /// The deferred request on `session` was answered: the session has nothing in flight.
    pub fn complete(&mut self, session: ObjectId) -> Result<(), QueueError> {
        if !self.is_deferred(session) {
            return Err(QueueError::NotDeferred(session));
        }
        self.landed(session);
        Ok(())
    }

    /// Forgets `session`, which was closed. Returns how many senders were waiting on it; what
    /// they and a request in flight are told is the kernel's business (kernel spec: "waiting
    /// and in-flight requests complete with the session-closed result").
    pub fn close(&mut self, session: ObjectId) -> u32 {
        let Some(line) = self.lines.remove(&session) else {
            return 0;
        };
        if line.flight == Some(Flight::Running) {
            self.running = self.running.saturating_sub(1);
        }
        self.ready.retain(|ready| *ready != session);
        line.waiting
    }

    /// Frees the worker of the request running on `session`.
    fn stop_running(&mut self, session: ObjectId) -> Result<(), QueueError> {
        if !self.is_running(session) {
            return Err(QueueError::NotRunning(session));
        }
        self.running = self.running.saturating_sub(1);
        Ok(())
    }

    /// The session's request in flight is over: its senders, if any wait, join the back of the
    /// line.
    fn landed(&mut self, session: ObjectId) {
        let Some(line) = self.lines.get_mut(&session) else {
            return;
        };
        line.flight = None;
        if line.waiting > 0 {
            self.ready.push_back(session);
        } else {
            self.lines.remove(&session);
        }
    }

    /// Checks every invariant the operations keep. Deserialising runs it, so a damaged snapshot
    /// is an error and not a later wrong turn.
    pub fn validate(&self) -> Result<(), QueueLayoutError> {
        if self.workers == 0 {
            return Err(QueueLayoutError::NoWorkers);
        }
        let running = |line: &&Line| line.flight == Some(Flight::Running);
        let counted = self.lines.values().filter(running).count();
        if usize::from(self.running) != counted {
            return Err(QueueLayoutError::RunningCount {
                recorded: self.running,
                counted,
            });
        }
        if self.running > self.workers {
            return Err(QueueLayoutError::TooManyRunning {
                running: self.running,
                workers: self.workers,
            });
        }
        // Every session in the line is ready and there once; the walk is bounded by the line.
        let mut seen = std::collections::BTreeSet::new();
        for session in &self.ready {
            let ready = self.lines.get(session).is_some_and(|line| line.is_ready());
            if !ready || !seen.insert(*session) {
                return Err(QueueLayoutError::Ready(*session));
            }
        }
        for (session, line) in &self.lines {
            if line.flight.is_none() && line.waiting == 0 {
                return Err(QueueLayoutError::IdleSession(*session));
            }
            if line.is_ready() && !seen.contains(session) {
                return Err(QueueLayoutError::Ready(*session));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
    use starquake_kernel::id::SlabId;

    use super::*;

    /// Session `n`. The generation is distinct from the index so that a swap of the two shows in
    /// the golden bytes.
    fn session(n: u32) -> ObjectId {
        ObjectId::from_parts(n, n + 0x10)
    }

    #[test]
    fn the_first_request_starts_and_another_session_waits_its_turn() {
        let (a, b, c) = (session(1), session(2), session(3));
        let mut queue = RequestQueue::new();
        assert_eq!((queue.workers(), queue.running()), (DEFAULT_WORKERS, 0));
        assert_eq!(queue.submit(a), Ok(Admission::Start));
        assert!(queue.is_running(a));
        // Two sessions of one server do not run handlers at once.
        assert_eq!(queue.submit(c), Ok(Admission::Queued));
        assert_eq!(queue.submit(b), Ok(Admission::Queued));
        assert_eq!(queue.running(), DEFAULT_WORKERS);
        assert_eq!(queue.start_next(), None);

        queue.finish(a).unwrap();
        // In the order they started waiting, not in id order.
        assert_eq!(queue.start_next(), Some(c));
        assert_eq!(queue.start_next(), None);
        queue.finish(c).unwrap();
        assert_eq!(queue.start_next(), Some(b));
        queue.finish(b).unwrap();
        assert_eq!(queue.start_next(), None);
        assert_eq!(queue, RequestQueue::new());
    }

    #[test]
    fn a_deferred_request_frees_the_worker_but_not_its_session() {
        let (a, b) = (session(1), session(2));
        let mut queue = RequestQueue::new();
        assert_eq!(queue.submit(a), Ok(Admission::Start));
        assert_eq!(queue.submit(b), Ok(Admission::Queued));
        queue.defer(a).unwrap();
        assert!(queue.is_deferred(a) && !queue.is_running(a));
        assert_eq!(queue.deferred().collect::<Vec<_>>(), [a]);
        // The server takes the next request from another session.
        assert_eq!(queue.running(), 0);
        assert_eq!(queue.start_next(), Some(b));
        queue.finish(b).unwrap();

        // A second sender on the deferred session waits, though the worker is free.
        assert_eq!(queue.submit(a), Ok(Admission::Queued));
        assert_eq!(queue.waiting(a), 1);
        assert_eq!(queue.start_next(), None);
        // It starts once the deferred request completes.
        queue.complete(a).unwrap();
        assert_eq!(queue.start_next(), Some(a));
        assert_eq!(queue.waiting(a), 0);
        queue.finish(a).unwrap();
        assert_eq!(queue, RequestQueue::new());
    }

    #[test]
    fn a_session_whose_request_ends_goes_to_the_back_of_the_line() {
        let (a, b) = (session(1), session(2));
        let mut queue = RequestQueue::new();
        assert_eq!(queue.submit(a), Ok(Admission::Start));
        assert_eq!(queue.submit(a), Ok(Admission::Queued));
        assert_eq!(queue.submit(b), Ok(Admission::Queued));
        queue.finish(a).unwrap();
        // `b` became ready first: `a`'s second sender only became ready when the first ended.
        assert_eq!(queue.start_next(), Some(b));
        queue.finish(b).unwrap();
        assert_eq!(queue.start_next(), Some(a));
    }

    #[test]
    fn a_new_sender_does_not_start_ahead_of_a_session_that_is_ready() {
        let (a, b, c) = (session(1), session(2), session(3));
        let mut queue = RequestQueue::new();
        assert_eq!(queue.submit(a), Ok(Admission::Start));
        assert_eq!(queue.submit(b), Ok(Admission::Queued));
        queue.finish(a).unwrap();
        // The worker is free and `c` has nothing in flight, but `b` was ready first and has
        // not been started yet: `c` joins the line behind it.
        assert_eq!(queue.running(), 0);
        assert_eq!(queue.submit(c), Ok(Admission::Queued));
        assert_eq!(queue.waiting(c), 1);
        assert_eq!(queue.start_next(), Some(b));
        assert_eq!(queue.start_next(), None);
        queue.finish(b).unwrap();
        assert_eq!(queue.start_next(), Some(c));
        queue.finish(c).unwrap();
        // With nobody ready, the same sender starts at once.
        assert_eq!(queue.submit(c), Ok(Admission::Start));
        queue.finish(c).unwrap();
        assert_eq!(queue, RequestQueue::new());

        // The same with a second worker: a free worker is for the session at the front.
        let mut queue = RequestQueue::with_workers(2).unwrap();
        assert_eq!(queue.submit(a), Ok(Admission::Start));
        assert_eq!(queue.submit(b), Ok(Admission::Start));
        assert_eq!(queue.submit(c), Ok(Admission::Queued));
        queue.finish(a).unwrap();
        assert_eq!(queue.submit(a), Ok(Admission::Queued));
        assert_eq!(queue.start_next(), Some(c));
        assert_eq!(queue.start_next(), None);
    }

    #[test]
    fn a_server_runs_exactly_as_many_requests_as_it_has_workers() {
        assert_eq!(RequestQueue::with_workers(0), Err(QueueError::NoWorkers));
        let (a, b, c) = (session(1), session(2), session(3));
        let mut queue = RequestQueue::with_workers(2).unwrap();
        assert_eq!(queue.submit(a), Ok(Admission::Start));
        // One below the limit starts; at the limit the sender waits.
        assert_eq!(queue.submit(b), Ok(Admission::Start));
        assert_eq!(queue.submit(c), Ok(Admission::Queued));
        assert_eq!((queue.running(), queue.start_next()), (2, None));
        // A worker does not let a session have two requests in flight.
        queue.finish(b).unwrap();
        assert_eq!(queue.start_next(), Some(c));
        assert_eq!(queue.submit(a), Ok(Admission::Queued));
        queue.finish(c).unwrap();
        assert_eq!((queue.running(), queue.start_next()), (1, None));
    }

    #[test]
    fn operations_on_the_wrong_state_are_refused_and_change_nothing() {
        let (a, b) = (session(1), session(2));
        let mut queue = RequestQueue::new();
        queue.submit(a).unwrap();
        queue.submit(b).unwrap();
        let before = queue.clone();
        assert_eq!(queue.finish(b), Err(QueueError::NotRunning(b)));
        assert_eq!(queue.defer(b), Err(QueueError::NotRunning(b)));
        assert_eq!(queue.complete(a), Err(QueueError::NotDeferred(a)));
        assert_eq!(queue.complete(b), Err(QueueError::NotDeferred(b)));
        assert_eq!(queue.withdraw(a), Err(QueueError::NotWaiting(a)));
        assert_eq!(
            queue.withdraw(session(9)),
            Err(QueueError::NotWaiting(session(9)))
        );
        assert_eq!(queue, before);
        queue.defer(a).unwrap();
        assert_eq!(queue.finish(a), Err(QueueError::NotRunning(a)));
        assert_eq!(
            QueueError::NotDeferred(a).to_string(),
            format!("session {a:?} has no deferred request")
        );
    }

    #[test]
    fn a_sender_that_gives_up_is_no_longer_counted() {
        let (a, b) = (session(1), session(2));
        let mut queue = RequestQueue::new();
        queue.submit(a).unwrap();
        queue.submit(b).unwrap();
        queue.submit(b).unwrap();
        queue.withdraw(b).unwrap();
        assert_eq!(queue.waiting(b), 1);
        queue.withdraw(b).unwrap();
        // Nobody waits on `b` now: it is out of the line and forgotten.
        assert_eq!(queue.sessions().collect::<Vec<_>>(), [a]);
        queue.finish(a).unwrap();
        assert_eq!(queue.start_next(), None);
        assert_eq!(queue, RequestQueue::new());

        // A sender behind a deferred request can give up too; the request stays in flight.
        queue.submit(a).unwrap();
        queue.defer(a).unwrap();
        queue.submit(a).unwrap();
        queue.withdraw(a).unwrap();
        assert!(queue.is_deferred(a));
        queue.complete(a).unwrap();
        assert_eq!(queue, RequestQueue::new());
    }

    #[test]
    fn closing_a_session_forgets_it_and_frees_its_worker() {
        let (a, b) = (session(1), session(2));
        let mut queue = RequestQueue::new();
        queue.submit(a).unwrap();
        queue.submit(b).unwrap();
        queue.submit(b).unwrap();
        // Closing a waiting session reports its senders and takes it out of the line.
        assert_eq!(queue.close(b), 2);
        assert_eq!(queue.close(b), 0);
        // Closing the running one frees the worker.
        queue.submit(a).unwrap();
        assert_eq!(queue.close(a), 1);
        assert_eq!(queue, RequestQueue::new());
        // A deferred request holds no worker to free.
        queue.submit(a).unwrap();
        queue.defer(a).unwrap();
        queue.submit(b).unwrap();
        assert_eq!(queue.close(a), 0);
        assert_eq!(queue.running(), 1);
        assert!(queue.is_running(b));
    }

    #[test]
    fn the_count_of_waiting_senders_stops_at_its_limit() {
        let a = session(1);
        let mut queue = RequestQueue::new();
        queue.submit(a).unwrap();
        queue.lines.get_mut(&a).unwrap().waiting = u32::MAX - 1;
        // One below the limit: one more sender fits, the next does not.
        assert_eq!(queue.submit(a), Ok(Admission::Queued));
        assert_eq!(queue.submit(a), Err(QueueError::TooManyWaiting(a)));
        assert_eq!(queue.waiting(a), u32::MAX);
    }

    /// A queue with every state in it: `a` running, `b` deferred with one sender waiting, `c`
    /// ready with two senders, on three workers.
    fn busy() -> RequestQueue {
        let (a, b, c) = (session(1), session(2), session(3));
        let mut queue = RequestQueue::with_workers(3).unwrap();
        queue.submit(b).unwrap();
        queue.defer(b).unwrap();
        queue.submit(b).unwrap();
        queue.submit(a).unwrap();
        // `c` cannot start by itself while it has nothing in flight and a worker is free, so
        // its two waiting senders are put there by hand, as a restore would.
        queue.lines.insert(
            c,
            Line {
                flight: None,
                waiting: 2,
            },
        );
        queue.ready.push_back(c);
        queue.validate().unwrap();
        queue
    }

    #[test]
    fn the_byte_layout_is_pinned() {
        let bytes = postcard::to_allocvec(&busy()).unwrap();
        #[rustfmt::skip]
        let golden = [
            3,                 // workers
            1,                 // running
            1, 3, 0x13,        // ready: one session (index 3, generation 0x13)
            3,                 // lines: three sessions
            1, 0x11, 1, 0, 0,  // session 1: in flight (1), running (variant 0), nobody waiting
            2, 0x12, 1, 1, 1,  // session 2: in flight, deferred (variant 1), one waiting
            3, 0x13, 0, 2,     // session 3: nothing in flight, two waiting
        ];
        assert_eq!(bytes, golden);
        assert_eq!(postcard::from_bytes::<RequestQueue>(&bytes), Ok(busy()));
    }

    /// Encodes queue data that never went through [`RequestQueue`]'s operations.
    fn encoded(change: impl FnOnce(&mut RequestQueue)) -> Vec<u8> {
        let mut queue = busy();
        change(&mut queue);
        postcard::to_allocvec(&queue).unwrap()
    }

    #[test]
    fn damaged_queue_data_is_refused_with_the_invariant_it_breaks() {
        let (b, c) = (session(2), session(3));
        let cases: [(Vec<u8>, QueueLayoutError); 8] = [
            (encoded(|q| q.workers = 0), QueueLayoutError::NoWorkers),
            (
                encoded(|q| q.running = 2),
                QueueLayoutError::RunningCount {
                    recorded: 2,
                    counted: 1,
                },
            ),
            (
                encoded(|q| {
                    q.workers = 1;
                    q.running = 2;
                    q.lines.get_mut(&c).unwrap().flight = Some(Flight::Running);
                    q.ready.clear();
                }),
                QueueLayoutError::TooManyRunning {
                    running: 2,
                    workers: 1,
                },
            ),
            (
                encoded(|q| {
                    q.lines.insert(session(4), Line::default());
                }),
                QueueLayoutError::IdleSession(session(4)),
            ),
            // In the line twice.
            (
                encoded(|q| q.ready.push_back(c)),
                QueueLayoutError::Ready(c),
            ),
            // In the line with a request in flight.
            (
                encoded(|q| q.ready.push_back(b)),
                QueueLayoutError::Ready(b),
            ),
            // In the line and unknown.
            (
                encoded(|q| q.ready.push_back(session(9))),
                QueueLayoutError::Ready(session(9)),
            ),
            // Ready and not in the line.
            (encoded(|q| q.ready.clear()), QueueLayoutError::Ready(c)),
        ];
        for (bytes, error) in cases {
            // postcard reports a failed `try_from` as a custom error and drops its text, so
            // the invariant is checked through `validate` on the same data.
            assert_eq!(
                postcard::from_bytes::<RequestQueue>(&bytes),
                Err(postcard::Error::SerdeDeCustom),
                "{error}"
            );
            let data: QueueData = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(RequestQueue::try_from(data), Err(error));
        }
        // One running request is still within one worker.
        let at_limit = encoded(|q| q.workers = 1);
        assert!(postcard::from_bytes::<RequestQueue>(&at_limit).is_ok());
        assert_eq!(
            QueueLayoutError::Ready(c).to_string(),
            format!("the line of ready sessions disagrees with session {c:?}")
        );
    }

    /// One step of a random run.
    #[derive(Debug, Clone, Copy)]
    enum Step {
        Submit(u32),
        Settle(u32, bool),
        Complete(u32),
        Withdraw(u32),
        Close(u32),
        Next,
    }

    fn step() -> impl Strategy<Value = Step> {
        let id = 0..4_u32;
        prop_oneof![
            4 => id.clone().prop_map(Step::Submit),
            3 => (id.clone(), any::<bool>()).prop_map(|(id, defer)| Step::Settle(id, defer)),
            2 => id.clone().prop_map(Step::Complete),
            1 => id.clone().prop_map(Step::Withdraw),
            1 => id.prop_map(Step::Close),
            3 => Just(Step::Next),
        ]
    }

    fn config() -> ProptestConfig {
        ProptestConfig {
            cases: 512,
            failure_persistence: None,
            rng_seed: RngSeed::Fixed(0x5153_5155_4555_4531),
            rng_algorithm: RngAlgorithm::ChaCha,
            ..Config::default()
        }
    }

    proptest! {
        #![proptest_config(config())]

        /// Under any run of operations the invariants hold after every step, the queue
        /// survives a round trip through its bytes, and every sender that was admitted or
        /// queued is started exactly once, withdrawn, or closed with its session.
        #[test]
        fn invariants_hold_and_no_sender_is_lost(
            workers in 1..=3_u8,
            steps in proptest::collection::vec(step(), 0..80),
        ) {
            let mut queue = RequestQueue::with_workers(workers).unwrap();
            // Senders in: submitted. Senders out: started, withdrawn or closed.
            let (mut sent, mut started, mut dropped) = (0_u64, 0_u64, 0_u64);
            for step in steps {
                match step {
                    Step::Submit(n) => {
                        sent += 1;
                        if queue.submit(session(n)).unwrap() == Admission::Start {
                            started += 1;
                        }
                    }
                    Step::Settle(n, defer) => {
                        let was_running = queue.is_running(session(n));
                        let settled = if defer {
                            queue.defer(session(n))
                        } else {
                            queue.finish(session(n))
                        };
                        prop_assert_eq!(settled.is_ok(), was_running);
                    }
                    Step::Complete(n) => {
                        let was_deferred = queue.is_deferred(session(n));
                        prop_assert_eq!(queue.complete(session(n)).is_ok(), was_deferred);
                    }
                    Step::Withdraw(n) => {
                        if queue.withdraw(session(n)).is_ok() {
                            dropped += 1;
                        }
                    }
                    Step::Close(n) => dropped += u64::from(queue.close(session(n))),
                    Step::Next => {
                        let free = queue.running() < queue.workers();
                        let ready = queue.ready.front().copied();
                        let next = queue.start_next();
                        // The session that has waited longest starts, if a worker is free.
                        prop_assert_eq!(next, ready.filter(|_| free));
                        started += u64::from(next.is_some());
                    }
                }
                prop_assert_eq!(queue.validate(), Ok(()));
                prop_assert!(queue.running() <= queue.workers());
                let bytes = postcard::to_allocvec(&queue).unwrap();
                prop_assert_eq!(postcard::from_bytes::<RequestQueue>(&bytes), Ok(queue.clone()));
            }
            let waiting: u64 = queue.sessions().map(|s| u64::from(queue.waiting(s))).sum();
            prop_assert_eq!(sent, started + dropped + waiting);
        }
    }
}
