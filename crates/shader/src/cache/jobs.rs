//! Translations on the shared compile pool: the executor seam `core` implements, and the inbox
//! through which a finished translation reaches the back thread (spec, "Compilation, threads and
//! stutter"; review guide, M16).
//!
//! A worker posts each result into the [`Inbox`] and wakes any waiter; the back thread takes
//! the results out when it next looks something up or waits. The only lock is the inbox's, and
//! the back thread takes it only when [`Inbox::posted`] says something was posted since it last
//! took, or when it waits for a ticket.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use super::CacheRefusal;
use super::stamp::ShaderStamp;
use crate::program::Rejection;

/// One unit of work for the compile pool: one shader's translation. It runs at most once.
pub type TranslateJob = Box<dyn FnOnce() + Send + 'static>;

/// The shader crate's view of the shared compile pool, which `core` implements over the one
/// pool the render backend's `CompilePool` also runs on (spec, "Compilation, threads and
/// stutter": "The crate takes the pool through its own small executor trait, which `core`
/// implements").
///
/// The contract the cache relies on, the same as the render crate's `CompilePool`:
///
/// - [`Executor::spawn`] returns at once, without running the job on the calling thread first
///   (the caller is the GPU back thread, whose lookups never translate);
/// - the executor runs each job at most once, on any of its workers, in any order;
/// - an executor that drops a job without running it (because it is shutting down) drops the
///   job's captured values, and the cache settles the translation as abandoned from their
///   `Drop` ([`CacheRefusal::Abandoned`]), so a back thread waiting for it is released; an
///   executor never keeps a job it will not run.
pub trait Executor: Send + Sync {
    /// Queues `job` to run once on a worker and returns at once.
    fn spawn(&self, job: TranslateJob);
}

/// A finished translation, as a worker posts it.
pub(super) struct Finished<S> {
    /// The shader entry it settles.
    pub(super) stamp: ShaderStamp,
    /// What the translator gave, or why it did not finish.
    pub(super) result: Result<S, Rejection>,
}

/// Where workers post finished translations for the back thread (module docs).
pub(super) struct Inbox<S> {
    results: Mutex<Vec<Finished<S>>>,
    /// Results posted and not yet taken; read without the lock.
    posted: AtomicUsize,
    /// Signalled after every post.
    wake: Condvar,
}

impl<S> Inbox<S> {
    /// An empty inbox.
    pub(super) fn new() -> Self {
        Self {
            results: Mutex::new(Vec::new()),
            posted: AtomicUsize::new(0),
            wake: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Vec<Finished<S>>> {
        // A worker cannot panic while holding the lock (it only pushes), so a poisoned lock
        // still holds whole results.
        self.results.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Posts one result and wakes the waiters.
    fn post(&self, finished: Finished<S>) {
        let mut results = self.lock();
        results.push(finished);
        self.posted.fetch_add(1, Ordering::Release);
        drop(results);
        self.wake.notify_all();
    }

    /// Whether a result was posted since the last [`Inbox::take`].
    pub(super) fn posted(&self) -> bool {
        self.posted.load(Ordering::Acquire) > 0
    }

    /// Every result posted so far, in posting order; takes the lock only when
    /// [`Inbox::posted`] says there is something.
    pub(super) fn take(&self) -> Vec<Finished<S>> {
        if !self.posted() {
            return Vec::new();
        }
        let mut results = self.lock();
        self.posted.store(0, Ordering::Release);
        std::mem::take(&mut *results)
    }

    /// Blocks until a result is posted that has not been taken, or until `deadline` passes;
    /// returns whether one has been. Returns at once when one is already waiting.
    pub(super) fn wait(&self, deadline: Option<Instant>) -> bool {
        let mut results = self.lock();
        while results.is_empty() {
            match deadline {
                None => {
                    results = self
                        .wake
                        .wait(results)
                        .unwrap_or_else(PoisonError::into_inner);
                }
                Some(deadline) => {
                    let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                        return false;
                    };
                    results = self
                        .wake
                        .wait_timeout(results, left)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
            }
        }
        true
    }
}

/// The one way a translation job settles: posting its result, or, when dropped without having
/// posted (the executor dropped the job, or the translator panicked), posting
/// [`CacheRefusal::Abandoned`].
pub(super) struct Settle<S> {
    inbox: Arc<Inbox<S>>,
    stamp: ShaderStamp,
    posted: bool,
}

impl<S> Settle<S> {
    /// The settler of the translation of `stamp`.
    pub(super) const fn new(inbox: Arc<Inbox<S>>, stamp: ShaderStamp) -> Self {
        Self {
            inbox,
            stamp,
            posted: false,
        }
    }

    /// Posts `result`.
    pub(super) fn post(mut self, result: Result<S, Rejection>) {
        self.posted = true;
        self.inbox.post(Finished {
            stamp: self.stamp,
            result,
        });
    }
}

impl<S> Drop for Settle<S> {
    fn drop(&mut self) {
        if !self.posted {
            self.posted = true;
            self.inbox.post(Finished {
                stamp: self.stamp,
                result: Err(Rejection::Cache(CacheRefusal::Abandoned)),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn taking_with_nothing_posted_does_not_take_the_lock() {
        let inbox = Arc::new(Inbox::<u32>::new());
        let held = inbox.lock();
        let (sender, receiver) = mpsc::channel();
        let worker = {
            let inbox = Arc::clone(&inbox);
            thread::spawn(move || {
                let _sent = sender.send(inbox.take().len());
            })
        };
        let taken = receiver.recv_timeout(Duration::from_secs(10));
        drop(held);
        worker.join().expect("the taker ends");
        assert_eq!(taken, Ok(0));
    }

    #[test]
    fn a_post_is_taken_once_and_wakes_a_waiter() {
        let inbox = Arc::new(Inbox::<u32>::new());
        assert!(!inbox.posted());
        let poster = {
            let inbox = Arc::clone(&inbox);
            thread::spawn(move || {
                Settle::new(inbox, ShaderStamp(7)).post(Ok(3));
            })
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        assert!(inbox.wait(Some(deadline)));
        poster.join().expect("the poster ends");
        assert!(inbox.posted());
        let taken = inbox.take();
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].stamp, ShaderStamp(7));
        assert!(matches!(taken[0].result, Ok(3)));
        assert!(!inbox.posted());
        assert!(inbox.take().is_empty());
        // Nothing waiting: a wait with a passed deadline gives up at once.
        assert!(!inbox.wait(Some(Instant::now())));
    }

    #[test]
    fn a_settler_dropped_unposted_posts_abandoned_and_a_posted_one_does_not() {
        let inbox = Arc::new(Inbox::<u32>::new());
        drop(Settle::new(Arc::clone(&inbox), ShaderStamp(1)));
        Settle::new(Arc::clone(&inbox), ShaderStamp(2)).post(Ok(5));
        let taken = inbox.take();
        assert_eq!(taken.len(), 2);
        assert_eq!(taken[0].stamp, ShaderStamp(1));
        assert!(matches!(
            taken[0].result,
            Err(Rejection::Cache(CacheRefusal::Abandoned))
        ));
        assert_eq!(taken[1].stamp, ShaderStamp(2));
        assert!(matches!(taken[1].result, Ok(5)));
    }
}
