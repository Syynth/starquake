//! The counters: where each comes from. Their JSON, as `summary.json` writes it, is pinned in
//! `core` (`starquake_core::hle::graphics`'s tests), which writes it.

use crate::back::exec::ExecCounters;
use crate::back::fence::{ClosedCounts, FenceCounters};

use super::*;

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
