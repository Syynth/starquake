//! The one compile pool of a run (docs/specs/render-backend.md, "Where the backend sits": "One
//! compile pool, created by `core` and handed to both the backend and the shader cache, sizes
//! the compile threads in one place"; review guide, M16; shader-pipeline, "Compilation, threads
//! and stutter"; slice R12).
//!
//! [`CompileWorkers`] is a fixed set of worker threads over one queue. It implements both
//! crates' executor traits, the render crate's [`CompilePool`] (pipeline builds, slice R6) and
//! the shader crate's [`Executor`] (translations, slice 19), so pipeline builds and translations
//! share the same threads: a job from either goes into the same queue and runs on whichever
//! worker takes it first.
//!
//! # Sizing
//!
//! [`pool_size`] is the one place the number of workers is decided, from the host's available
//! parallelism, as Eden sizes its pipeline workers (`eden/src/video_core/renderer_vulkan/
//! vk_pipeline_cache.cpp` @ cf654052, `GetTotalPipelineWorkers`: `std::max<size_t>(
//! hardware_concurrency(), 2ULL) - 1ULL` off Android). Eden answers 0 for an unknown concurrency,
//! which that formula turns into one worker; [`pool_size`] does the same for `None`. Eden's
//! Android clamp and its single worker for drivers with broken parallel compiling are not
//! followed: neither applies to the hosts starquake runs on (macOS and Linux, through wgpu).
//!
//! Each worker's stack is [`WORKER_STACK`] bytes, a **guess**: the macOS main thread's 8 MiB,
//! four times Rust's default for spawned threads. Building a pipeline walks the shader's naga
//! module recursively (the wgpu backend refuses modules nested deeper than 256 for that reason,
//! and one nested 1,536 deep overflowed a 2 MiB thread when that bound was measured; see
//! `docs/proposals.md`, "Nesting depth across crates"), and a translation's structurizer
//! recurses too. Whether 8 MiB covers the deepest module the shader crate can produce is
//! unmeasured.
//!
//! # The contract both traits ask for
//!
//! - [`CompilePool::spawn`] and [`Executor::spawn`] return at once: they push the job and wake
//!   one worker, and never run it on the calling thread.
//! - Each job runs at most once, on one worker, in queue order as workers free up.
//! - A job that panics is caught on its worker: the worker logs it at error level, counts it
//!   ([`PoolCounters::panicked`]) and takes the next job, so one bad build does not cost the pool
//!   a thread. The job's captured values are dropped during the unwind, which is how the render
//!   backend and the shader cache settle a build or translation that did not finish
//!   (`CompilePool`'s and `Executor`'s docs).
//! - At shutdown (when the pool is dropped) the jobs still queued are dropped unrun, outside the
//!   queue's lock, so their `Drop`s settle what waits on them. No job can be spawned after that:
//!   both `spawn`s borrow the pool, which the drop holds exclusively. A pool never keeps a job it
//!   will not run.
//!
//! # Shutdown
//!
//! Dropping the [`CompileWorkers`] sets the shutdown flag, drops the queued jobs, wakes every
//! worker and joins them. A worker finishes the job it is running first, so the drop waits for
//! the longest job then running. When the drop happens on one of the pool's own workers (a job
//! held the last handle to a backend that held the last handle to the pool), that worker is not
//! joined, since a thread cannot join itself: it sees the flag when the job returns and ends by
//! itself ([`should_join`], the wgpu backend's rule for its render thread).
//!
//! # Bounds
//!
//! The queue holds what callers have spawned and no worker has taken. Neither caller spawns
//! without a bound of its own: the GPU crate records at most `MAX_PIPELINES` pipelines
//! (65,536, a guess; decision log, 2026-11-19, ruling 6), each built once, and the shader cache
//! caps its shader entries (`starquake_shader::cache`, "Limits"), each translated once. The pool
//! adds no cap of its own.

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{JoinHandle, ThreadId};

use starquake_render::{CompileJob, CompilePool};
use starquake_shader::cache::{Executor, TranslateJob};

/// The bytes of stack each worker gets (module docs, "Sizing"): a guess.
pub(crate) const WORKER_STACK: usize = 8 << 20;

/// The name each worker thread has, followed by its index.
pub(crate) const WORKER_NAME: &str = "compile-";

/// One job of either kind: a pipeline build or a translation.
type Job = Box<dyn FnOnce() + Send + 'static>;

/// How many workers a pool has on a host with `parallelism` hardware threads (module docs,
/// "Sizing"): Eden's `max(hardware_concurrency, 2) - 1`, and one when it is unknown.
#[must_use]
pub(crate) fn pool_size(parallelism: Option<NonZeroUsize>) -> NonZeroUsize {
    let threads = parallelism.map_or(0, NonZeroUsize::get);
    NonZeroUsize::new(threads.max(2) - 1).unwrap_or(NonZeroUsize::MIN)
}

/// Whether a pool dropped on thread `current` joins its worker `worker`: not when they are the
/// same thread (module docs, "Shutdown").
#[must_use]
pub(crate) fn should_join(worker: ThreadId, current: ThreadId) -> bool {
    worker != current
}

/// What a pool has done with the jobs it was given.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PoolCounters {
    /// Jobs that ran to their end.
    pub(crate) finished: u64,
    /// Jobs that panicked; their worker went on (module docs).
    pub(crate) panicked: u64,
}

/// The queue and its state, shared by the pool and its workers.
struct Shared {
    queue: Mutex<Queue>,
    /// Signalled when a job is queued or the pool shuts down.
    work: Condvar,
    finished: AtomicU64,
    panicked: AtomicU64,
}

/// The jobs waiting, and whether the pool is shutting down.
#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    shutdown: bool,
}

impl Shared {
    fn queue(&self) -> MutexGuard<'_, Queue> {
        // Nothing panics while holding the lock (jobs run outside it), so a poisoned lock still
        // holds a whole queue.
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Queues `job` and wakes one worker.
    fn push(&self, job: Job) {
        let mut queue = self.queue();
        queue.jobs.push_back(job);
        drop(queue);
        self.work.notify_one();
    }

    /// The next job for a worker, waiting for one; `None` once the pool is shutting down.
    fn take(&self) -> Option<Job> {
        let mut queue = self.queue();
        loop {
            if queue.shutdown {
                return None;
            }
            if let Some(job) = queue.jobs.pop_front() {
                return Some(job);
            }
            queue = self
                .work
                .wait(queue)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// A worker's life: runs jobs until shutdown, catching each one's panic (module docs).
    fn work(&self) {
        while let Some(job) = self.take() {
            if catch_unwind(AssertUnwindSafe(job)).is_ok() {
                self.finished.fetch_add(1, Ordering::Relaxed);
            } else {
                self.panicked.fetch_add(1, Ordering::Relaxed);
                tracing::error!("a compile job panicked; its worker goes on");
            }
        }
    }
}

/// The run's compile pool (module docs).
pub(crate) struct CompileWorkers {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl core::fmt::Debug for CompileWorkers {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CompileWorkers")
            .field("workers", &self.workers.len())
            .field("counters", &self.counters())
            .finish_non_exhaustive()
    }
}

impl CompileWorkers {
    /// A pool of `count` workers, named [`WORKER_NAME`] and their index, each with
    /// [`WORKER_STACK`] bytes of stack. The error is the host's, when a thread cannot be
    /// started; the workers already started are stopped and joined first.
    pub(crate) fn new(count: NonZeroUsize) -> std::io::Result<Self> {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue::default()),
            work: Condvar::new(),
            finished: AtomicU64::new(0),
            panicked: AtomicU64::new(0),
        });
        let mut pool = Self {
            shared,
            workers: Vec::with_capacity(count.get()),
        };
        for index in 0..count.get() {
            let shared = Arc::clone(&pool.shared);
            // On an error, dropping `pool` stops and joins the workers started so far.
            let worker = std::thread::Builder::new()
                .name(format!("{WORKER_NAME}{index}"))
                .stack_size(WORKER_STACK)
                .spawn(move || shared.work())?;
            pool.workers.push(worker);
        }
        Ok(pool)
    }

    /// A pool sized for this host ([`pool_size`] of `std::thread::available_parallelism`).
    pub(crate) fn for_host() -> std::io::Result<Self> {
        Self::new(pool_size(std::thread::available_parallelism().ok()))
    }

    /// How many workers the pool has.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.workers.len()
    }

    /// What the pool has done so far.
    pub(crate) fn counters(&self) -> PoolCounters {
        PoolCounters {
            finished: self.shared.finished.load(Ordering::Relaxed),
            panicked: self.shared.panicked.load(Ordering::Relaxed),
        }
    }
}

impl CompilePool for CompileWorkers {
    /// Queues a pipeline build (module docs, "The contract both traits ask for").
    fn spawn(&self, job: CompileJob) {
        self.shared.push(job);
    }
}

impl Executor for CompileWorkers {
    /// Queues a translation, on the same queue as the pipeline builds (module docs).
    fn spawn(&self, job: TranslateJob) {
        self.shared.push(job);
    }
}

impl Drop for CompileWorkers {
    fn drop(&mut self) {
        let unrun = {
            let mut queue = self.shared.queue();
            queue.shutdown = true;
            std::mem::take(&mut queue.jobs)
        };
        self.shared.work.notify_all();
        // Outside the lock: each job's `Drop` settles what waits on it (module docs).
        drop(unrun);
        let current = std::thread::current().id();
        for worker in self.workers.drain(..) {
            if !should_join(worker.thread().id(), current) {
                continue;
            }
            if worker.join().is_err() {
                // A worker catches its jobs' panics, so only its own code could get here.
                tracing::error!("a compile worker ended by a panic");
            }
        }
    }
}

#[cfg(test)]
mod tests;
