//! Measurement of the cost of recording one event with only the flight recorder attached
//! (`docs/specs/observability.md`: "well under 100 ns per event").
//!
//! This is `#[ignore]`d because a wall-clock number depends on the machine and the build profile,
//! and a timing assertion in the normal gate would be flaky. Run it where the number matters:
//!
//! ```text
//! cargo test --release -p starquake-trace --test recording_cost -- --ignored --nocapture
//! ```
//!
//! It prints the mean cost per event over several rounds and asserts only that the rings are full
//! and that the newest event kept is the last one the recorder numbered, so it also cannot pass by
//! measuring a no-op.

use std::hint::black_box;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use starquake_trace::{
    EventKind, FlightRecorder, GuestTicks, Recorder, ResultCode, SVC_ARG_COUNT, ThreadId,
};

const CORES: usize = 6;
const RING: usize = 4096;
const EVENTS_PER_ROUND: u32 = 1_000_000;
const ROUNDS: usize = 5;

#[test]
#[ignore = "wall-clock measurement; run with --release --ignored --nocapture"]
fn cost_of_recording_one_event_with_only_the_flight_recorder() {
    let recorder = Recorder::new();
    let flight = Arc::new(FlightRecorder::new(CORES, RING));
    recorder.add_sink(Arc::clone(&flight));

    let kind = EventKind::Svc {
        id: 0x1f,
        args: core::array::from_fn::<u64, SVC_ARG_COUNT, _>(|i| u64::try_from(i + 1).unwrap()),
        result: ResultCode(0),
    };

    let mut per_event_ns = Vec::new();
    let mut last_seq = None;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        for n in 0..EVENTS_PER_ROUND {
            let core = u8::try_from(usize::try_from(n).unwrap() % CORES).unwrap();
            last_seq = Some(black_box(recorder.record(
                GuestTicks(u64::from(n)),
                core,
                ThreadId(0),
                black_box(kind),
            )));
        }
        let elapsed = start.elapsed();
        per_event_ns.push(elapsed.as_secs_f64() * 1e9 / f64::from(EVENTS_PER_ROUND));
    }

    let best = per_event_ns.iter().copied().fold(f64::INFINITY, f64::min);
    let mean =
        per_event_ns.iter().sum::<f64>() / f64::from(u32::try_from(per_event_ns.len()).unwrap());
    writeln!(
        std::io::stderr(),
        "flight-recorder-only recording: best {best:.1} ns/event, mean {mean:.1} ns/event \
         ({ROUNDS} rounds of {EVENTS_PER_ROUND} events, {CORES} cores x {RING} slots)"
    )
    .unwrap();

    // Every ring is full of the newest events, and the newest one kept is the last one the
    // recorder numbered.
    let dump = flight.snapshot();
    assert_eq!(flight.unrouted(), 0);
    for core in &dump.cores {
        assert_eq!(core.events.len(), RING);
    }
    let newest = dump.merged().last().map(|e| e.seq);
    assert_eq!(newest, last_seq);
}
