//! The front on its own thread, with the host clock: the production arrangement. Every wait is
//! bounded.

use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use super::*;

/// How long a test waits for the front thread before it fails.
const PATIENCE: Duration = Duration::from_secs(10);

/// Runs `body` on a thread and returns its result, or fails after `PATIENCE`.
fn within<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    let (send, receive) = mpsc::channel();
    thread::spawn(move || {
        let _ = send.send(body());
    });
    receive.recv_timeout(PATIENCE).unwrap()
}

/// The rig's front with the host clock in place of the fake.
fn host_clocked(r: Rig) -> (Front, Back, Rig) {
    let Rig {
        front,
        shared,
        back,
        mem,
        vm,
        syncpoints,
        waiters,
        ordered,
        pushed,
        clock,
    } = r;
    let mut front = front;
    front.clock = Arc::new(HostClock::new());
    let rest = Rig {
        front: Front::new(FrontDeps {
            shared: Arc::clone(&shared),
            ring: ring(2).unwrap().0,
            syncpoints: Arc::clone(&syncpoints),
            completions: Arc::<Pushed>::clone(&pushed),
            waiters: Arc::clone(&waiters),
            ordered: Arc::clone(&ordered),
            clock: Arc::<Clock>::clone(&clock),
        }),
        shared,
        back: Back {
            consumer: ring(2).unwrap().1,
            fences: Fences::new(
                Arc::new(RecordingBackend::new(caps(), 1 << 30)),
                FenceEnv {
                    syncpoints: Arc::clone(&syncpoints),
                    waiters: Arc::clone(&waiters),
                    completions: Arc::<Pushed>::clone(&pushed),
                    clock: Arc::<Clock>::clone(&clock),
                    front: Arc::new(FrontShared::new()),
                },
                Mode::Fenced,
            ),
            ordered: Arc::clone(&ordered),
            memory: Arc::clone(&mem),
            log: Vec::new(),
        },
        mem,
        vm,
        syncpoints,
        waiters,
        ordered,
        pushed,
        clock,
    };
    (front, back, rest)
}

#[test]
fn the_front_thread_runs_a_submission_and_wakes_for_a_back_report() {
    // A acquires SEM == 1 and then releases SEM2 = 0x5a; B, submitted once A waits, releases
    // SEM = 1. The front thread runs both on the host clock; the test thread is the back.
    let (front, mut back, r) = host_clocked(rig());
    let a = r.channel(5);
    let b = r.channel(6);
    let thread = spawn(front).unwrap();
    let mut pb = PushBuffer::new();
    acquire(&mut pb, SEM, AcquireOp::Equal, 1);
    release(&mut pb, SEM2, 0x5a);
    r.submit(a, vec![r.entry(CMD_A, pb.words())]);
    let mut pb = PushBuffer::new();
    release(&mut pb, SEM, 1);
    let started = Instant::now();
    let mut submitted_b = false;
    while r.read_word(SEM2) != 0x5a {
        assert!(started.elapsed() < PATIENCE, "the release never came");
        if !submitted_b && r.waiters.has_waiter(SPACE, SEM) {
            r.submit(b, vec![r.entry(CMD_B, pb.words())]);
            submitted_b = true;
        }
        if let Ok(Some(entry)) = back.consumer.pop_timeout(Duration::from_millis(5)) {
            back.handle(entry);
        }
    }
    assert!(submitted_b);
    assert!(!thread.is_finished());
    assert!(Arc::ptr_eq(thread.shared(), &r.shared));
    assert_eq!(within(move || thread.stop()), Ok(()));
}

#[test]
fn a_front_thread_that_failed_reports_its_error_when_stopped() {
    let mut r = rig_with(4, |mem| mem);
    let a = r.channel(5);
    let mut pb = PushBuffer::new();
    bind_3d(&mut pb);
    let load = m(clc797::LOAD_CONSTANT_BUFFER.method_at(&[0]).unwrap());
    pb.methods(MethodMode::NonIncrementing, 0, load, &[7; 12])
        .unwrap();
    r.submit(a, vec![r.entry(CMD_A, pb.words())]);
    // The back drains what fits, so the front reaches the run that does not.
    let front = std::mem::replace(
        &mut r.front,
        Front::new(FrontDeps {
            shared: Arc::new(FrontShared::new()),
            ring: ring(2).unwrap().0,
            syncpoints: Arc::clone(&r.syncpoints),
            completions: Arc::<Pushed>::clone(&r.pushed),
            waiters: Arc::clone(&r.waiters),
            ordered: Arc::clone(&r.ordered),
            clock: Arc::<Clock>::clone(&r.clock),
        }),
    );
    let thread = spawn(front).unwrap();
    let started = Instant::now();
    while !thread.is_finished() {
        assert!(
            started.elapsed() < PATIENCE,
            "the front thread did not stop"
        );
        if let Ok(Some(entry)) = r.back.consumer.pop_timeout(Duration::from_millis(5)) {
            r.back.handle(entry);
        }
    }
    assert_eq!(
        within(move || thread.stop()),
        Err(FrontError::Ring(crate::ring::PushError::TooLarge {
            needed: 8,
            capacity: 4
        }))
    );
}

#[test]
fn a_park_ends_at_once_for_a_wake_that_came_before_it() {
    // The submission comes before the park: the park does not sleep (an hour would fail the
    // bounded wait).
    let shared = Arc::new(FrontShared::new());
    shared.create_address_space(
        SPACE,
        Arc::new(GpuVm::new()),
        Arc::new(TestGpuMemory::new(FlatMemory::new(CPU_BASE, 0x1000))),
    );
    let parker = Arc::clone(&shared);
    assert!(within(move || parker.park(Duration::from_secs(3600))));
    // Taken, the wake is spent: a park with nothing new sleeps out its timeout.
    shared.take();
    let parker = Arc::clone(&shared);
    assert!(!within(move || parker.park(Duration::from_millis(1))));
}

#[test]
fn a_park_ends_when_another_thread_submits() {
    let shared = Arc::new(FrontShared::new());
    shared.create_address_space(
        SPACE,
        Arc::new(GpuVm::new()),
        Arc::new(TestGpuMemory::new(FlatMemory::new(CPU_BASE, 0x1000))),
    );
    let channel = shared
        .create_channel(ChannelConfig {
            space: SPACE,
            syncpoint: syncpoint(1),
        })
        .unwrap();
    shared.take();
    let (parked_send, parked) = mpsc::channel();
    let sleeper = Arc::clone(&shared);
    let (woke_send, woke) = mpsc::channel();
    thread::spawn(move || {
        parked_send.send(()).unwrap();
        let _ = woke_send.send(sleeper.park(Duration::from_secs(3600)));
    });
    parked.recv_timeout(PATIENCE).unwrap();
    // Whether the submission lands before or during the park, the park ends.
    assert_eq!(shared.submit(channel, plain(vec![])), Submitted::Queued);
    assert_eq!(woke.recv_timeout(PATIENCE), Ok(true));
}

#[test]
fn a_back_report_ends_a_park() {
    let shared = Arc::new(FrontShared::new());
    let (woke_send, woke) = mpsc::channel();
    let parker = Arc::clone(&shared);
    thread::spawn(move || {
        let _ = woke_send.send(parker.park(Duration::from_secs(3600)));
    });
    crate::back::fence::FrontWake::wake(
        &*shared,
        &crate::back::fence::Report {
            incremented: 1,
            ..crate::back::fence::Report::default()
        },
    );
    assert_eq!(woke.recv_timeout(PATIENCE), Ok(true));
    let taken = shared.take();
    assert!(taken.reports.incremented);
    assert!(!taken.reports.released);
}
