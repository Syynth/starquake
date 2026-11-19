//! The counters and idleness: where each counter comes from, the idleness decision for each of
//! its inputs, and the back's report bumping the activity counter around the front's wake. The
//! counters' JSON, as `summary.json` writes it, is pinned in `core`
//! (`hle/tests/graphics/gpu.rs`), which has the JSON writer.

use crate::back::exec::ExecCounters;
use crate::back::fence::{ClosedCounts, FenceCounters};

use super::*;

/// An idle look: every input at its idle value, the counts and the activity not zero.
const IDLE: Looked = Looked {
    quiet: true,
    sent: 7,
    taken: 7,
    in_flight: 0,
    before: 3,
    after: 3,
};

#[test]
fn idleness_needs_every_input_idle() {
    assert!(idle(IDLE));
    // Each input made busy alone.
    let busy = [
        Looked {
            quiet: false,
            ..IDLE
        },
        Looked { taken: 6, ..IDLE },
        Looked { sent: 8, ..IDLE },
        Looked {
            in_flight: 1,
            ..IDLE
        },
        Looked { after: 4, ..IDLE },
        Looked { before: 2, ..IDLE },
    ];
    for looked in busy {
        assert!(!idle(looked), "{looked:?}");
    }
}

/// A front double that reads the activity counter when it is woken.
struct SeesActivity {
    status: Arc<Status>,
    seen: Mutex<Vec<u64>>,
}

impl FrontWake for SeesActivity {
    fn wake(&self, _report: &Report) {
        let now = self
            .status
            .activity
            .load(std::sync::atomic::Ordering::SeqCst);
        self.seen.lock().unwrap().push(now);
    }
}

#[test]
fn a_back_report_bumps_the_activity_before_and_after_it_reaches_the_front() {
    let status = Arc::new(Status::default());
    let front = Arc::new(SeesActivity {
        status: Arc::clone(&status),
        seen: Mutex::new(Vec::new()),
    });
    let recorder = Arc::new(FlightRecorder::new(RECORDER_EVENTS));
    let report = ReportToFront {
        front: Arc::<SeesActivity>::clone(&front),
        status: Arc::clone(&status),
        recorder: Arc::clone(&recorder),
    };
    report.wake(&Report {
        incremented: 1,
        ..Report::default()
    });
    assert_eq!(*front.seen.lock().unwrap(), [1]);
    assert_eq!(status.activity.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(
        recorder.events(),
        [GpuEvent::Performed {
            released: 0,
            incremented: 1,
            latched: false,
            markers: 0,
            failed: Vec::new(),
        }]
    );
}

#[test]
fn each_counter_comes_from_its_source() {
    // Every source field a distinct value, so a counter read from the wrong one fails.
    let front = FrontCounters {
        words: 101,
        submissions: 102,
        increments: 103,
        parks: 104,
        fence_waits: 105,
        sync_waits: 106,
        catch_ups: 107,
        polls: 108,
        ordered: 109,
        macro_failures: 110,
        stopped_channels: 111,
        presents: 112,
    };
    let back = BackStatus {
        exec: ExecCounters {
            applied: 201,
            failed: 202,
            unhandled_words: 203,
            captures: 204,
            presents: 205,
            dispatches: 206,
            dispatches_skipped: 207,
            write_backs: 208,
            write_back_failures: 209,
            catch_ups: 210,
            catch_up_failures: 211,
        },
        fences: FenceCounters {
            fenced: 301,
            skipped_empty: 302,
            closed: ClosedCounts {
                ring_empty: 1,
                present: 2,
                wait_for_idle: 4,
                waiter: 8,
                elapsed: 16,
                set_reference: 32,
                catch_up: 64,
                channel_removed: 128,
            },
            lost_wakes: 303,
            failed_releases: 304,
        },
        ..BackStatus::default()
    };
    let counters = GpuCounters::new(&front, &back, 401, 402);
    let expected = GpuCounters {
        words: 101,
        submissions: 102,
        submissions_full: 401,
        submissions_dropped: 402,
        increments: 103,
        acquires_parked: 104,
        fence_waits: 105,
        sync_waits: 106,
        macro_catch_ups: 107,
        polls: 108,
        ordered: 109,
        macro_failures: 110,
        channels_stopped: 111,
        presents_pushed: 112,
        entries_applied: 201,
        entries_refused: 202,
        unhandled_words: 203,
        presents: 205,
        dispatches: 206,
        dispatches_skipped: 207,
        write_backs: 208,
        write_back_failures: 209,
        render_catch_ups: 210,
        render_catch_up_failures: 211,
        fenced_actions: 301,
        fenced_actions_skipped_empty: 302,
        // Every close reason counted once: 1 + 2 + ... + 128.
        lists_submitted: 255,
        lost_wakes: 303,
        failed_releases: 304,
    };
    assert_eq!(counters, expected);
}
