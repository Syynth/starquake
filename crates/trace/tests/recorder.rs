//! Cross-module behaviour through the public API: concurrent recording, and one event stream
//! feeding the flight recorder, the text view and a full log at once.

use std::sync::Arc;
use std::thread;

use starquake_trace::{
    Dump, Event, EventKind, FlightRecorder, GuestTicks, LifecyclePhase, Recorder, ResultCode,
    SVC_ARG_COUNT, TextSink, ThreadId, VecSink, golden_trace,
};

const WRITERS: u64 = 6;
const EVENTS_PER_WRITER: u64 = 2_000;
const RING: usize = 64;

fn svc(id: u32) -> EventKind {
    EventKind::Svc {
        id,
        args: [0; SVC_ARG_COUNT],
        result: ResultCode(0),
    }
}

#[test]
fn sequence_numbers_are_strictly_increasing_under_concurrent_recording() {
    let recorder = Arc::new(Recorder::new());
    let log = Arc::new(VecSink::new());
    let flight = Arc::new(FlightRecorder::new(usize::try_from(WRITERS).unwrap(), RING));
    recorder.add_sink(Arc::clone(&log));
    recorder.add_sink(Arc::clone(&flight));

    let handles: Vec<_> = (0..WRITERS)
        .map(|writer| {
            let recorder = Arc::clone(&recorder);
            thread::spawn(move || {
                (0..EVENTS_PER_WRITER)
                    .map(|n| {
                        let id = u32::try_from(n).unwrap();
                        recorder.record(
                            GuestTicks(n),
                            u8::try_from(writer).unwrap(),
                            ThreadId(writer),
                            svc(id),
                        )
                    })
                    .collect::<Vec<u64>>()
            })
        })
        .collect();
    let returned: Vec<Vec<u64>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Each writer saw its own numbers increase.
    for seqs in &returned {
        assert!(seqs.windows(2).all(|w| w[0] < w[1]));
    }
    // The sink saw every event exactly once, in strictly increasing order, with no gaps.
    let total = WRITERS * EVENTS_PER_WRITER;
    let delivered: Vec<u64> = log.events().iter().map(|e| e.seq).collect();
    assert_eq!(delivered, (0..total).collect::<Vec<_>>());
    // The numbers handed back to the writers are the same set.
    let mut all: Vec<u64> = returned.into_iter().flatten().collect();
    all.sort_unstable();
    assert_eq!(all, delivered);

    // The flight recorder kept the newest RING events of each writer's core, in order.
    let dump = flight.snapshot();
    assert_eq!(flight.unrouted(), 0);
    for core in &dump.cores {
        assert_eq!(core.events.len(), RING);
        assert!(core.events.windows(2).all(|w| w[0].seq < w[1].seq));
        assert!(core.events.iter().all(|e| e.core == core.core));
        let ids: Vec<u32> = core
            .events
            .iter()
            .filter_map(|e| match e.kind {
                EventKind::Svc { id, .. } => Some(id),
                _ => None,
            })
            .collect();
        let first = u32::try_from(EVENTS_PER_WRITER).unwrap() - u32::try_from(RING).unwrap();
        let newest: Vec<u32> = (first..u32::try_from(EVENTS_PER_WRITER).unwrap()).collect();
        assert_eq!(ids, newest);
    }
}

#[test]
fn one_stream_feeds_every_sink_and_the_dump_renders_the_same_trace() {
    let recorder = Recorder::new();
    let log = Arc::new(VecSink::new());
    let text = Arc::new(TextSink::new());
    // Big enough that nothing is evicted, so the dump holds the whole stream.
    let flight = Arc::new(FlightRecorder::new(2, 16));
    recorder.add_sink(Arc::clone(&log));
    recorder.add_sink(Arc::clone(&text));
    recorder.add_sink(Arc::clone(&flight));
    for n in 0..10u32 {
        recorder.record(
            GuestTicks(u64::from(n)),
            u8::try_from(n % 2).unwrap(),
            ThreadId(u64::from(n % 3)),
            if n % 4 == 0 {
                EventKind::Lifecycle {
                    phase: LifecyclePhase::ThreadCreate,
                    subject: u64::from(n),
                }
            } else {
                svc(n)
            },
        );
    }
    let from_dump = Dump::from_bytes(&flight.dump().unwrap()).unwrap();
    let merged: Vec<Event> = from_dump.merged();
    assert_eq!(merged, log.events());
    assert_eq!(golden_trace(&merged), text.render());
    assert_ne!(text.render(), "");
}
