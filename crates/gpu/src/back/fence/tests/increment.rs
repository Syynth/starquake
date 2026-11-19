//! A syncpoint increment with a count (the module docs' "Completion" and "Waiters and a run of
//! increments"): one wrapping addition and at most one push, whatever the count.

use super::*;
use crate::interface::Fence;
use crate::syncpoint::{CompareMode, SIGN_BIT, wait_satisfied};

/// Sets syncpoint `id` to `start` (unwatched, so nothing is pushed), watches it, records one
/// command, attaches an increment of `count`, closes the list and completes it; the report.
fn run(r: &mut Rig, id: SyncpointId, start: u32, count: u32) -> Report {
    r.syncpoints.increment_by(id, start);
    r.syncpoints.watch(id);
    r.fences.record(command(1)).unwrap();
    assert_eq!(
        r.fences.attach(FencedAction::Increment { id, count }),
        Ok(Attached::Open)
    );
    let fence = r.fences.close(CloseReason::Present).unwrap().unwrap();
    // Nothing is added before the list completes.
    assert_eq!(r.syncpoints.read(id), start);
    r.backend.finish(fence);
    r.fences.complete()
}

#[test]
fn u32_max_increments_take_one_addition() {
    // From a non-zero start, `u32::MAX` increments wrap to one below it, with one push of the
    // final value: an increment at a time would push `u32::MAX` times, or not finish in time.
    let start = 0x10;
    let (report, value, pushed) = within(move || {
        let mut r = rig();
        let id = syncpoint(5);
        let report = run(&mut r, id, start, u32::MAX);
        (report, r.syncpoints.read(id), r.pushed.all())
    });
    let last = start.wrapping_add(u32::MAX);
    assert_eq!(last, start - 1);
    assert_eq!(value, last);
    assert_eq!(
        report,
        Report {
            incremented: u64::from(u32::MAX),
            ..Report::default()
        }
    );
    assert_eq!(
        pushed,
        [GpuCompletion::SyncpointChanged {
            id: syncpoint(5),
            value: last
        }]
    );
}

#[test]
fn counts_at_zero_one_and_the_wrap_edge() {
    // (start, count, the value after it). Each start differs, so a count read as the start, or
    // a start read as 0, fails.
    let cases = [
        (7, 0, 7),
        (9, 1, 10),
        (u32::MAX - 1, 1, u32::MAX),
        (u32::MAX - 1, 2, 0),
        (u32::MAX, 1, 0),
        (u32::MAX, 3, 2),
    ];
    for (start, count, after) in cases {
        let mut r = rig();
        let id = syncpoint(0x2a);
        let report = run(&mut r, id, start, count);
        assert_eq!(r.syncpoints.read(id), after, "{start} + {count}");
        let expected = Report {
            incremented: u64::from(count),
            ..Report::default()
        };
        assert_eq!(report, expected, "{start} + {count}");
        if count == 0 {
            // Nothing changed: nothing is pushed and the front is not woken.
            assert_eq!(r.pushed.all(), [], "{start} + {count}");
            assert_eq!(r.front.all(), [], "{start} + {count}");
        } else {
            let changed = GpuCompletion::SyncpointChanged { id, value: after };
            assert_eq!(r.pushed.all(), [changed], "{start} + {count}");
            assert_eq!(r.front.all(), [expected], "{start} + {count}");
        }
    }
}

#[test]
fn an_increment_with_nothing_behind_it_adds_its_count_at_once() {
    // The empty-fence rule holds for a count as for one increment.
    let mut r = rig();
    let id = syncpoint(3);
    let performed = Report {
        incremented: 0x8000_0003,
        ..Report::default()
    };
    assert_eq!(
        r.fences.attach(FencedAction::Increment {
            id,
            count: 0x8000_0003
        }),
        Ok(Attached::Performed(performed))
    );
    assert_eq!(r.syncpoints.read(id), 0x8000_0003);
}

#[test]
fn a_run_meets_the_thresholds_it_passes_up_to_half_the_range() {
    // A threshold just past the start is the one a run leaves furthest behind. A count of 2^31
    // leaves it 2^31 - 1 behind the final value, which the wrapping comparison still meets, as
    // Eden would have met it on the way; one more leaves it 2^31 behind, which it does not (the
    // module docs' divergence). The final value itself is met either way.
    let start = 0x0100;
    let passed = start + 1;
    for (count, met) in [(SIGN_BIT, true), (SIGN_BIT + 1, false)] {
        let mut r = rig();
        let id = syncpoint(7);
        run(&mut r, id, start, count);
        let last = start.wrapping_add(count);
        assert_eq!(last.wrapping_sub(passed), count - 1);
        // The front's fence waits read the counter.
        let fence = Fence { id, value: passed };
        assert_eq!(r.syncpoints.reached(fence), met, "{count:#x}");
        assert!(
            r.syncpoints.reached(Fence { id, value: last }),
            "{count:#x}"
        );
        // gsv's waiters compare the pushed value.
        let pushed = r.pushed.all();
        assert_eq!(
            pushed,
            [GpuCompletion::SyncpointChanged { id, value: last }]
        );
        let satisfied = wait_satisfied(last, u64::from(passed), CompareMode::Wrapping32);
        assert_eq!(satisfied, met, "{count:#x}");
    }
}
