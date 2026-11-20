//! The compile pool: its size, that both crates' jobs share its workers, and its shutdown. Every
//! wait is bounded by [`BOUND`], so a deadlock fails a test instead of hanging the gate.

use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use super::*;

/// The longest any wait here takes.
const BOUND: Duration = Duration::from_secs(20);

/// Runs `f` on another thread and fails if it takes longer than [`BOUND`].
fn bounded<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || tx.send(f()));
    rx.recv_timeout(BOUND).expect("finished in time")
}

/// The worker a job ran on: its name and id.
type Ran = (Option<String>, ThreadId);

/// Where the job is running.
fn here() -> Ran {
    let thread = std::thread::current();
    (thread.name().map(str::to_owned), thread.id())
}

/// A pool of `count` workers.
fn pool(count: usize) -> CompileWorkers {
    CompileWorkers::new(NonZeroUsize::new(count).unwrap()).unwrap()
}

/// Sets its flag when dropped: a job's captured value, whose `Drop` is how a backend or the
/// shader cache learns that a job ended without finishing.
struct Guard(Sender<&'static str>);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.send("dropped");
    }
}

#[test]
fn the_pool_is_sized_as_eden_sizes_its_pipeline_workers() {
    // eden/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp @ cf654052,
    // `GetTotalPipelineWorkers`: max(hardware_concurrency, 2) - 1; an unknown concurrency is 0
    // there.
    let some = |n| NonZeroUsize::new(n);
    for (parallelism, workers) in [
        (None, 1),
        (some(1), 1),
        (some(2), 1),
        (some(3), 2),
        (some(10), 9),
        (some(usize::MAX), usize::MAX - 1),
    ] {
        assert_eq!(pool_size(parallelism).get(), workers, "{parallelism:?}");
    }
}

#[test]
fn a_host_s_pool_has_the_size_it_is_given() {
    let pool = CompileWorkers::for_host().unwrap();
    assert_eq!(
        pool.len(),
        pool_size(std::thread::available_parallelism().ok()).get()
    );
    let sized = self::pool(3);
    assert_eq!(sized.len(), 3);
}

#[test]
fn a_pool_dropped_on_its_own_worker_does_not_join_it() {
    let worker = std::thread::spawn(|| ()).thread().id();
    assert!(!should_join(worker, worker));
    assert!(should_join(worker, std::thread::current().id()));
}

#[test]
fn pipeline_builds_and_translations_run_on_the_same_workers() {
    let pool = pool(1);
    let (tx, rx) = mpsc::channel::<(&str, Ran)>();
    let build = tx.clone();
    CompilePool::spawn(
        &pool,
        Box::new(move || build.send(("build", here())).unwrap()),
    );
    Executor::spawn(
        &pool,
        Box::new(move || tx.send(("translate", here())).unwrap()),
    );
    let first = rx.recv_timeout(BOUND).unwrap();
    let second = rx.recv_timeout(BOUND).unwrap();
    assert_eq!((first.0, second.0), ("build", "translate"));
    let name = Some(format!("{WORKER_NAME}0"));
    assert_eq!(first.1.0, name);
    assert_eq!(second.1, first.1, "the same worker thread");
    assert_ne!(first.1.1, std::thread::current().id());
}

#[test]
fn jobs_of_both_kinds_run_in_the_order_they_were_spawned() {
    let pool = pool(1);
    let (release, released) = mpsc::channel::<()>();
    let (tx, rx) = mpsc::channel::<u32>();
    // The one worker waits until every job is queued, so the order is the queue's alone.
    CompilePool::spawn(
        &pool,
        Box::new(move || released.recv_timeout(BOUND).unwrap()),
    );
    for index in 0..6 {
        let tx = tx.clone();
        let job: Job = Box::new(move || tx.send(index).unwrap());
        if index % 2 == 0 {
            CompilePool::spawn(&pool, job);
        } else {
            Executor::spawn(&pool, job);
        }
    }
    release.send(()).unwrap();
    let order: Vec<u32> = (0..6).map(|_| rx.recv_timeout(BOUND).unwrap()).collect();
    assert_eq!(order, [0, 1, 2, 3, 4, 5]);
    assert_eq!(pool.counters().panicked, 0);
}

#[test]
fn spawning_returns_before_the_job_runs() {
    let pool = pool(2);
    let (release, released) = mpsc::channel::<()>();
    let (tx, rx) = mpsc::channel::<Ran>();
    let job_tx = tx.clone();
    // Neither spawn waits for its job: both jobs block until released, after both returned.
    let released = Arc::new(Mutex::new(released));
    let second = Arc::clone(&released);
    CompilePool::spawn(
        &pool,
        Box::new(move || {
            released.lock().unwrap().recv_timeout(BOUND).unwrap();
            job_tx.send(here()).unwrap();
        }),
    );
    Executor::spawn(
        &pool,
        Box::new(move || {
            second.lock().unwrap().recv_timeout(BOUND).unwrap();
            tx.send(here()).unwrap();
        }),
    );
    assert!(rx.try_recv().is_err(), "no job has run yet");
    release.send(()).unwrap();
    release.send(()).unwrap();
    for _ in 0..2 {
        let (name, id) = rx.recv_timeout(BOUND).unwrap();
        assert_ne!(id, std::thread::current().id());
        assert!(name.unwrap().starts_with(WORKER_NAME));
    }
}

#[test]
fn a_panicking_job_is_counted_and_its_worker_goes_on() {
    let pool = pool(1);
    let (guards, dropped) = mpsc::channel();
    let guard = Guard(guards);
    CompilePool::spawn(
        &pool,
        Box::new(move || {
            let _guard = guard;
            unreachable!("a build that panics, as a job may");
        }),
    );
    // Its captured value is dropped as it unwinds.
    assert_eq!(dropped.recv_timeout(BOUND), Ok("dropped"));
    let (tx, rx) = mpsc::channel();
    Executor::spawn(&pool, Box::new(move || tx.send(here()).unwrap()));
    let (name, _) = rx.recv_timeout(BOUND).unwrap();
    assert_eq!(
        name,
        Some(format!("{WORKER_NAME}0")),
        "the one worker lives on"
    );
    let counters = bounded(move || {
        // The second job's count lands after its send: wait for it.
        loop {
            let counters = pool.counters();
            if counters.finished == 1 {
                return counters;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    });
    assert_eq!(
        counters,
        PoolCounters {
            finished: 1,
            panicked: 1
        }
    );
}

/// Starts a one-worker pool whose worker runs a job that waits for `release`, and queues behind
/// it a job holding a [`Guard`] that reports `"ran"` if it runs.
fn busy_pool_with_a_queued_job() -> (CompileWorkers, Sender<()>, Receiver<&'static str>) {
    let pool = pool(1);
    let (release, released) = mpsc::channel::<()>();
    let (started_tx, started) = mpsc::channel::<()>();
    CompilePool::spawn(
        &pool,
        Box::new(move || {
            started_tx.send(()).unwrap();
            released.recv_timeout(BOUND).unwrap();
        }),
    );
    started.recv_timeout(BOUND).unwrap();
    let (events_tx, events) = mpsc::channel();
    let guard = Guard(events_tx.clone());
    Executor::spawn(
        &pool,
        Box::new(move || {
            let _guard = guard;
            events_tx.send("ran").unwrap();
        }),
    );
    (pool, release, events)
}

#[test]
fn shutdown_drops_the_queued_jobs_unrun_and_waits_for_the_running_one() {
    let (pool, release, events) = busy_pool_with_a_queued_job();
    let (dropped_tx, dropped) = mpsc::channel::<()>();
    std::thread::spawn(move || {
        drop(pool);
        dropped_tx.send(()).unwrap();
    });
    // The queued job is dropped unrun before the running one ends.
    assert_eq!(events.recv_timeout(BOUND), Ok("dropped"));
    // The drop waits for the running job.
    assert!(dropped.recv_timeout(Duration::from_millis(50)).is_err());
    release.send(()).unwrap();
    assert_eq!(dropped.recv_timeout(BOUND), Ok(()));
    assert!(events.try_recv().is_err(), "the queued job never ran");
}

#[test]
fn a_queued_job_runs_when_the_pool_is_not_dropped() {
    // The same rig as the shutdown test, with the pool kept: the queued job runs once the
    // running one ends, so the shutdown test's "dropped" is the drop's doing.
    let (pool, release, events) = busy_pool_with_a_queued_job();
    release.send(()).unwrap();
    assert_eq!(events.recv_timeout(BOUND), Ok("ran"));
    assert_eq!(events.recv_timeout(BOUND), Ok("dropped"));
    drop(pool);
}

#[test]
fn a_pool_dropped_by_its_own_job_ends_without_a_hang() {
    let pool = pool(2);
    let slot = Arc::new(Mutex::new(None::<CompileWorkers>));
    let (tx, rx) = mpsc::channel();
    let in_job = Arc::clone(&slot);
    let job_tx = tx.clone();
    // The lock is held until the pool is in the slot, so the job, which waits for the lock,
    // always finds it there.
    let mut held = slot.lock().unwrap();
    CompilePool::spawn(
        &pool,
        Box::new(move || {
            // The pool's last handle, dropped on one of its own workers.
            let pool = in_job.lock().unwrap().take();
            assert!(pool.is_some());
            drop(pool);
            job_tx.send(here()).unwrap();
        }),
    );
    *held = Some(pool);
    drop(held);
    let (name, _) = rx.recv_timeout(BOUND).unwrap();
    assert!(name.unwrap().starts_with(WORKER_NAME));
    assert!(slot.lock().unwrap().is_none());
    drop(tx);
}
