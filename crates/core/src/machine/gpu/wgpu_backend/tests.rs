//! Opening the wgpu backend: the wait setting's conversion, the call on the frontend's main
//! thread, and the run's frame sink. None of these needs a GPU; the tests that open a real
//! backend are in `hle/tests/graphics/present.rs`.

use std::sync::Mutex;
use std::thread::ThreadId;

use super::*;
use crate::renderer::{MainJob, MainThreadGone};

/// How long a test here waits for a main thread that answers.
const ANSWERS: Duration = Duration::from_secs(20);

/// How long a test here waits for a main thread that never answers.
const SILENT: Duration = Duration::from_millis(50);

#[test]
fn the_gpu_crate_s_wait_setting_becomes_the_render_crate_s_variant_for_variant() {
    assert_eq!(
        while_compiling(WaitWhileCompiling::Wait),
        WhileCompiling::Wait
    );
    assert_eq!(
        while_compiling(WaitWhileCompiling::Skip),
        WhileCompiling::Skip
    );
    // The default carries over: both crates wait by default (decision 5).
    assert_eq!(
        while_compiling(WaitWhileCompiling::default()),
        WhileCompiling::default()
    );
}

/// A main thread: a thread of the test's that runs the jobs posted to it, in order, and says
/// which thread it is.
struct Runner {
    jobs: Mutex<mpsc::Sender<MainJob>>,
    thread: ThreadId,
}

impl Runner {
    fn new() -> Self {
        let (jobs, posted) = mpsc::channel::<MainJob>();
        let thread = std::thread::spawn(move || {
            while let Ok(job) = posted.recv() {
                job();
            }
        })
        .thread()
        .id();
        Self {
            jobs: Mutex::new(jobs),
            thread,
        }
    }
}

impl MainThread for Runner {
    fn post(&self, job: MainJob) -> Result<(), MainThreadGone> {
        self.jobs
            .lock()
            .unwrap()
            .send(job)
            .map_err(|_| MainThreadGone)
    }
}

/// A main thread that takes jobs and keeps them, never running them.
#[derive(Default)]
struct Keeps(Mutex<Vec<MainJob>>);

impl MainThread for Keeps {
    fn post(&self, job: MainJob) -> Result<(), MainThreadGone> {
        self.0.lock().unwrap().push(job);
        Ok(())
    }
}

/// A main thread that takes jobs and drops them unrun.
struct Drops;

impl MainThread for Drops {
    fn post(&self, job: MainJob) -> Result<(), MainThreadGone> {
        drop(job);
        Ok(())
    }
}

/// A main thread that no longer takes jobs.
struct Gone;

impl MainThread for Gone {
    fn post(&self, _job: MainJob) -> Result<(), MainThreadGone> {
        Err(MainThreadGone)
    }
}

#[test]
fn the_present_target_is_set_on_the_main_thread_and_its_answer_waited_for() {
    let main = Runner::new();
    let (ran_on, on) = mpsc::channel();
    let answered = present_on_main(&main, ANSWERS, move || {
        // Late, so a caller that did not wait would return before the answer.
        std::thread::sleep(Duration::from_millis(20));
        ran_on.send(std::thread::current().id()).unwrap();
        Ok(())
    });
    assert_eq!(answered, Ok(()));
    assert_eq!(
        on.try_recv(),
        Ok(main.thread),
        "set on the main thread, before the answer"
    );
}

#[test]
fn a_refused_present_target_ends_the_run_with_the_backend_s_reason() {
    let answered = present_on_main(&Runner::new(), ANSWERS, || {
        Err(RenderError::Unsupported("a test's refusal"))
    });
    assert_eq!(
        answered,
        Err("the window could not be shown: a test's refusal".to_owned())
    );
}

#[test]
fn a_main_thread_that_takes_no_jobs_ends_the_run() {
    let answered = present_on_main(&Gone, ANSWERS, || unreachable!("never run"));
    assert_eq!(
        answered,
        Err(
            "the window could not be shown: the frontend's main thread no longer takes jobs"
                .to_owned()
        )
    );
}

#[test]
fn a_main_thread_that_does_not_answer_in_time_ends_the_run() {
    let main = Keeps::default();
    let started = std::time::Instant::now();
    let answered = present_on_main(&main, SILENT, || Ok(()));
    assert!(started.elapsed() >= SILENT);
    assert_eq!(
        answered,
        Err(
            "the window could not be shown: the frontend's main thread did not answer within 50 \
             ms"
            .to_owned()
        )
    );
    // The job is still there, and running it late harms nothing: nobody reads its answer.
    let late = main.0.lock().unwrap().pop().unwrap();
    late();
}

#[test]
fn a_main_thread_that_drops_the_job_ends_the_run_at_once() {
    let started = std::time::Instant::now();
    let answered = present_on_main(&Drops, ANSWERS, || unreachable!("never run"));
    assert!(started.elapsed() < ANSWERS / 2, "not a timeout");
    assert_eq!(
        answered,
        Err(
            "the window could not be shown: the frontend's main thread dropped the job without \
             running it"
                .to_owned()
        )
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn the_run_s_sink_keeps_each_frame_with_or_without_its_backend() {
    use starquake_render::command::PresentTransform;
    use starquake_render::frame::RGBA8_BYTES;
    use starquake_render::{Frame, FramePixels, FrameSink};

    let keeper = Arc::new(FrameKeeper::default());
    // A backend already gone: the sink still keeps the frame, and has nothing to drain.
    let sink = RunSink {
        keeper: Arc::clone(&keeper),
        backend: std::sync::Weak::new(),
    };
    for number in [3, 4] {
        sink.frame(Frame {
            frame: number,
            width: 1,
            height: 1,
            transform: PresentTransform::Identity,
            pixels: FramePixels::Rgba8(vec![0x5a; RGBA8_BYTES]),
        });
    }
    let kept = keeper.take();
    assert_eq!(
        kept.files.iter().map(|file| file.frame).collect::<Vec<_>>(),
        [3, 4]
    );
}
