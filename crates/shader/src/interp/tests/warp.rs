//! The warp operations on a one-lane warp (`exec`'s `warp` module): votes, ballots, the four
//! shuffles with their in-range flags, the quad operations, and the lane-id check. Each
//! expected lane is worked out by hand from Eden's formulas (cited in that module), written
//! as literals.

use super::{run, u1_of, u32_of};
use crate::interp::{INACTIVE_LANE_VALUE, InterpError, Machine, Val};
use crate::ir::WARP_SIZE;

/// The value shuffled in these tests: distinct from [`INACTIVE_LANE_VALUE`] and from zero.
const OWN: u32 = 0x1234_5678;

/// A machine whose one active lane is `lane`.
fn on_lane(lane: u32) -> Machine {
    let mut machine = Machine::new();
    machine.system.lane_id = lane;
    machine
}

/// Runs `body` (one block, which this ends with a return) on lane `lane` and gives the values
/// `%0` and `%1`.
fn two(body: &str, lane: u32) -> (Val, Val) {
    let text = format!("block0:\n{body}\n    return\n");
    let finished = run(&text, &mut on_lane(lane)).expect("runs");
    (
        finished.value(0).expect("%0"),
        finished.value(1).expect("%1"),
    )
}

/// `shuffle` of [`OWN`] with `index`, `clamp` and `segment` on lane `lane`: its value and its
/// in-range flag.
fn shuffle(shuffle: &str, lane: u32, index: u32, clamp: u32, segment: u32) -> (u32, bool) {
    let body = format!(
        "    %0 = {shuffle} u32 {OWN:#x}, u32 {index:#x}, u32 {clamp:#x}, u32 {segment:#x}\n\
         \x20   %1 = GetInBoundsFromOp %0"
    );
    let (value, in_range) = two(&body, lane);
    (u32_of(value), u1_of(in_range))
}

/// The error, as words, of running `op` as `%0` on lane `lane`.
fn error_on_lane(op: &str, lane: u32) -> String {
    let text = format!("block0:\n    %0 = {op}\n    return\n");
    run(&text, &mut on_lane(lane))
        .err()
        .map_or_else(String::new, |error| error.to_string())
}

#[test]
fn votes_of_one_lane() {
    for vote in [false, true] {
        let body = format!(
            "    %0 = VoteAll u1 {vote}\n    %1 = VoteAny u1 {vote}\n    %2 = VoteEqual u1 {vote}"
        );
        let text = format!("block0:\n{body}\n    return\n");
        let finished = run(&text, &mut on_lane(3)).expect("runs");
        assert_eq!(finished.value(0), Some(Val::U1(vote)), "all of {vote}");
        assert_eq!(finished.value(1), Some(Val::U1(vote)), "any of {vote}");
        assert_eq!(finished.value(2), Some(Val::U1(true)), "equal of {vote}");
    }
}

#[test]
fn a_ballot_is_this_lanes_bit() {
    for (lane, bit) in [(0, 0x1), (5, 0x20), (31, 0x8000_0000)] {
        let (yes, no) = two(
            "    %0 = SubgroupBallot u1 true\n    %1 = SubgroupBallot u1 false",
            lane,
        );
        assert_eq!(u32_of(yes), bit, "lane {lane}");
        assert_eq!(u32_of(no), 0, "lane {lane}");
    }
}

#[test]
fn the_lane_id_must_be_inside_the_warp() {
    let last = WARP_SIZE - 1;
    let ops = [
        "SubgroupBallot u1 true",
        "ShuffleIndex u32 0x1, u32 0x0, u32 0x1f, u32 0x0",
        "ShuffleUp u32 0x1, u32 0x0, u32 0x1f, u32 0x0",
        "ShuffleDown u32 0x1, u32 0x0, u32 0x1f, u32 0x0",
        "ShuffleButterfly u32 0x1, u32 0x0, u32 0x1f, u32 0x0",
        "QuadBroadcast u32 0x1, u32 0x0",
        "QuadSwap u32 0x1, u32 0x0",
    ];
    for op in ops {
        assert_eq!(error_on_lane(op, last), "", "{op} on the last lane");
        assert_eq!(
            error_on_lane(op, WARP_SIZE),
            "%0: lane 32 is outside the warp of 32 lanes",
            "{op}"
        );
    }
    // Votes do not read the lane.
    assert_eq!(error_on_lane("VoteAll u1 true", WARP_SIZE), "");
    let mut machine = on_lane(u32::MAX);
    let text = "block0:\n    %0 = SubgroupBallot u1 false\n    return\n";
    let error = run(text, &mut machine).err();
    assert!(
        matches!(
            error,
            Some(InterpError::LaneOutsideWarp { lane: u32::MAX, at }) if at.index() == 0
        ),
        "{error:?}"
    );
}

#[test]
fn an_index_shuffle_reads_its_lane_in_range() {
    // Lane 5, clamp 31, no segment: the source is the index, always in range.
    assert_eq!(shuffle("ShuffleIndex", 5, 5, 0x1f, 0), (OWN, true));
    assert_eq!(
        shuffle("ShuffleIndex", 5, 6, 0x1f, 0),
        (INACTIVE_LANE_VALUE, true)
    );
    // Only the low five bits of the index count (`GuestLane`): 37 is lane 5.
    assert_eq!(shuffle("ShuffleIndex", 5, 37, 0x1f, 0), (OWN, true));
    // Clamp 6: lane 6 is the last in range, 7 the first out (its own value).
    assert_eq!(shuffle("ShuffleIndex", 6, 6, 6, 0), (OWN, true));
    assert_eq!(shuffle("ShuffleIndex", 6, 7, 6, 0), (OWN, false));
    assert_eq!(
        shuffle("ShuffleIndex", 5, 6, 6, 0),
        (INACTIVE_LANE_VALUE, true)
    );
    // Segment 0x18 on lane 13: lanes 8 to 15, the index's low three bits added to 8.
    assert_eq!(shuffle("ShuffleIndex", 13, 5, 0x1f, 0x18), (OWN, true));
    assert_eq!(
        shuffle("ShuffleIndex", 13, 7, 0x1f, 0x18),
        (INACTIVE_LANE_VALUE, true)
    );
    // The comparison is signed: a clamp of all ones is -1, below every lane.
    assert_eq!(shuffle("ShuffleIndex", 5, 5, 0xffff_ffff, 0), (OWN, false));
}

#[test]
fn an_up_shuffle_counts_down_from_its_lane() {
    // Lane 5, clamp 0: in range while the source is at least lane 0.
    assert_eq!(shuffle("ShuffleUp", 5, 0, 0, 0), (OWN, true));
    assert_eq!(
        shuffle("ShuffleUp", 5, 5, 0, 0),
        (INACTIVE_LANE_VALUE, true)
    );
    assert_eq!(shuffle("ShuffleUp", 5, 6, 0, 0), (OWN, false));
    // Clamp 3: lane 3 is the lowest in range.
    assert_eq!(
        shuffle("ShuffleUp", 5, 2, 3, 0),
        (INACTIVE_LANE_VALUE, true)
    );
    assert_eq!(shuffle("ShuffleUp", 5, 3, 3, 0), (OWN, false));
    // The delta is taken modulo 32.
    assert_eq!(shuffle("ShuffleUp", 5, 32, 0, 0), (OWN, true));
}

#[test]
fn a_down_shuffle_counts_up_from_its_lane() {
    assert_eq!(shuffle("ShuffleDown", 5, 0, 0x1f, 0), (OWN, true));
    assert_eq!(
        shuffle("ShuffleDown", 5, 26, 0x1f, 0),
        (INACTIVE_LANE_VALUE, true)
    );
    assert_eq!(shuffle("ShuffleDown", 5, 27, 0x1f, 0), (OWN, false));
    // Clamp 7: lane 7 is the highest in range.
    assert_eq!(
        shuffle("ShuffleDown", 5, 2, 7, 0),
        (INACTIVE_LANE_VALUE, true)
    );
    assert_eq!(shuffle("ShuffleDown", 5, 3, 7, 0), (OWN, false));
}

#[test]
fn a_butterfly_shuffle_flips_lane_bits() {
    assert_eq!(shuffle("ShuffleButterfly", 5, 0, 0x1f, 0), (OWN, true));
    assert_eq!(
        shuffle("ShuffleButterfly", 5, 1, 0x1f, 0),
        (INACTIVE_LANE_VALUE, true)
    );
    // Clamp 7: lane 5 ^ 2 = 7 is in range, 5 ^ 8 = 13 is not.
    assert_eq!(
        shuffle("ShuffleButterfly", 5, 2, 7, 0),
        (INACTIVE_LANE_VALUE, true)
    );
    assert_eq!(shuffle("ShuffleButterfly", 5, 8, 7, 0), (OWN, false));
}

#[test]
fn quad_operations_read_within_the_quad() {
    // Lane 6 is the third lane of the quad at 4.
    let broadcast = |index: u32| {
        let (value, _) = two(
            &format!(
                "    %0 = QuadBroadcast u32 {OWN:#x}, u32 {index:#x}\n    %1 = VoteAll u1 true"
            ),
            6,
        );
        u32_of(value)
    };
    assert_eq!(broadcast(2), OWN);
    assert_eq!(
        broadcast(10),
        OWN,
        "only the low two bits of the index count: 10 is lane 2 of the quad, not lane 14"
    );
    assert_eq!(broadcast(1), INACTIVE_LANE_VALUE);
    let swap = |direction: u32| {
        let (value, _) = two(
            &format!(
                "    %0 = QuadSwap u32 {OWN:#x}, u32 {direction:#x}\n    %1 = VoteAll u1 true"
            ),
            6,
        );
        u32_of(value)
    };
    for direction in 0..3 {
        assert_eq!(
            swap(direction),
            INACTIVE_LANE_VALUE,
            "direction {direction}"
        );
    }
    // Eden's fallback flips `direction + 1`, which wraps to 0 here: this lane.
    assert_eq!(swap(u32::MAX), OWN);
}

#[test]
fn in_bounds_is_the_latest_run_of_its_shuffle() {
    // The shuffle runs twice: first with index 0 (in range), then with index 7 (out of range
    // against clamp 6); the flag read after the second run is false.
    let text = "block0:
    %0 = SetRegister r0, u32 0
    jump block1
block1:
    %1 = GetRegister r0
    %2 = ShuffleIndex u32 0x9, %1, u32 0x6, u32 0x0
    %3 = GetInBoundsFromOp %2
    %4 = SetRegister r0, u32 7
    %5 = IEqual %1, u32 0
    branch %5, block1, block2
block2:
    return
";
    let finished = run(text, &mut on_lane(0)).expect("runs");
    assert_eq!(finished.value(3), Some(Val::U1(false)));
    assert_eq!(finished.value(2), Some(Val::U32(0x9)));
}

#[test]
fn in_bounds_of_anything_but_a_shuffle_is_unsupported() {
    let text =
        "block0:\n    %0 = IAdd32 u32 0x1, u32 0x2\n    %1 = GetInBoundsFromOp %0\n    return\n";
    assert_eq!(
        run(text, &mut on_lane(0))
            .err()
            .map(|error| error.to_string()),
        Some("%1: the interpreter does not support GetInBoundsFromOp".to_owned())
    );
}
