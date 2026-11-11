//! The flight recorder: a fixed-capacity ring of recent events per core.
//!
//! Source: `docs/specs/observability.md`, "Flight recorder" ("A fixed-size ring per core holding
//! the most recent events (a few thousand each). It costs no allocation and no I/O while
//! running"). All memory is allocated in [`FlightRecorder::new`]; [`Sink::record`] only copies an
//! [`Event`] into a slot. Dumping allocates, which is fine: it runs after something went wrong,
//! not on the hot path.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::dump::{CoreDump, Dump, DumpError};
use crate::event::{Event, EventKind, GuestTicks, LifecyclePhase, ThreadId};
use crate::recorder::{Sink, lock};

/// Placeholder written into unused slots; never observable, because only the newest `len` slots
/// are ever read back.
const FILLER: Event = Event {
    seq: 0,
    time: GuestTicks(0),
    core: 0,
    thread: ThreadId(0),
    kind: EventKind::Lifecycle {
        phase: LifecyclePhase::ProcessStart,
        subject: 0,
    },
};

struct Ring {
    slots: Box<[Event]>,
    /// Index the next event is written to.
    next: usize,
    /// How many slots hold real events (at most `slots.len()`).
    len: usize,
}

impl Ring {
    fn new(capacity: usize) -> Self {
        Self {
            slots: vec![FILLER; capacity].into_boxed_slice(),
            next: 0,
            len: 0,
        }
    }

    fn push(&mut self, event: Event) {
        let capacity = self.slots.len();
        if let Some(slot) = self.slots.get_mut(self.next) {
            *slot = event;
            self.next = (self.next + 1) % capacity;
            self.len = (self.len + 1).min(capacity);
        }
        // A zero-capacity ring has no slots, so nothing is kept.
    }

    /// The retained events, oldest first.
    fn events(&self) -> Vec<Event> {
        let capacity = self.slots.len();
        // While the ring has not wrapped, the oldest event is slot 0; once full it is the slot
        // the next write will overwrite.
        let start = if self.len < capacity { 0 } else { self.next };
        (0..self.len)
            .filter_map(|i| self.slots.get((start + i) % capacity))
            .copied()
            .collect()
    }
}

/// A fixed-capacity ring per core holding the newest events, attachable to a
/// [`Recorder`](crate::Recorder) as a [`Sink`].
pub struct FlightRecorder {
    rings: Vec<Mutex<Ring>>,
    capacity: usize,
    unrouted: AtomicU64,
}

impl FlightRecorder {
    /// A flight recorder with `cores` rings of `capacity_per_core` events each. All memory is
    /// allocated here.
    #[must_use]
    pub fn new(cores: usize, capacity_per_core: usize) -> Self {
        Self {
            rings: (0..cores)
                .map(|_| Mutex::new(Ring::new(capacity_per_core)))
                .collect(),
            capacity: capacity_per_core,
            unrouted: AtomicU64::new(0),
        }
    }

    /// How many cores (rings) there are.
    #[must_use]
    pub fn cores(&self) -> usize {
        self.rings.len()
    }

    /// How many events each ring keeps.
    #[must_use]
    pub fn capacity_per_core(&self) -> usize {
        self.capacity
    }

    /// How many events named a core this recorder has no ring for and were therefore not kept.
    #[must_use]
    pub fn unrouted(&self) -> u64 {
        self.unrouted.load(Ordering::Relaxed)
    }

    /// Copy out what every ring holds right now, oldest first within each core.
    #[must_use]
    pub fn snapshot(&self) -> Dump {
        let cores = self
            .rings
            .iter()
            .enumerate()
            .map(|(index, ring)| CoreDump {
                // `new` takes any `usize` core count; a ring beyond u8 range cannot be named by
                // an event (`Event::core` is a `u8`), so it is always empty and numbered by
                // saturation.
                core: u8::try_from(index).unwrap_or(u8::MAX),
                events: lock(ring).events(),
            })
            .collect();
        Dump::new(cores)
    }

    /// Write the rings out as bytes (a [`Dump`] encoded with `postcard`).
    pub fn dump(&self) -> Result<Vec<u8>, DumpError> {
        self.snapshot().to_bytes()
    }
}

impl Sink for FlightRecorder {
    fn record(&self, event: &Event) {
        match self.rings.get(usize::from(event.core)) {
            Some(ring) => lock(ring).push(*event),
            None => {
                self.unrouted.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

    use super::*;
    use crate::Recorder;

    fn config() -> ProptestConfig {
        ProptestConfig {
            cases: 128,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(0x464C_4947_4854_5243),
            failure_persistence: None,
            ..Config::default()
        }
    }

    fn lifecycle(n: u64) -> EventKind {
        EventKind::Lifecycle {
            phase: LifecyclePhase::ThreadCreate,
            subject: n,
        }
    }

    fn seqs(dump: &Dump, core: u8) -> Vec<u64> {
        dump.cores
            .iter()
            .find(|c| c.core == core)
            .unwrap()
            .events
            .iter()
            .map(|e| e.seq)
            .collect()
    }

    #[test]
    fn keeps_exactly_the_newest_n_events_in_order() {
        const CAPACITY: usize = 4;
        let flight = Arc::new(FlightRecorder::new(1, CAPACITY));
        let recorder = Recorder::new();
        recorder.add_sink(Arc::clone(&flight));
        let total = CAPACITY * 2 + 1;
        for n in 0..total {
            recorder.record(GuestTicks(0), 0, ThreadId(0), lifecycle(n as u64));
        }
        let expected: Vec<u64> = ((total - CAPACITY) as u64..total as u64).collect();
        assert_eq!(seqs(&flight.snapshot(), 0), expected);
    }

    #[test]
    fn a_ring_that_has_not_wrapped_returns_everything() {
        let flight = Arc::new(FlightRecorder::new(1, 8));
        let recorder = Recorder::new();
        recorder.add_sink(Arc::clone(&flight));
        for n in 0..3 {
            recorder.record(GuestTicks(0), 0, ThreadId(0), lifecycle(n));
        }
        assert_eq!(seqs(&flight.snapshot(), 0), [0, 1, 2]);
    }

    #[test]
    fn rings_are_per_core() {
        let flight = Arc::new(FlightRecorder::new(2, 2));
        let recorder = Recorder::new();
        recorder.add_sink(Arc::clone(&flight));
        // core 0 gets seqs 0, 2, 4; core 1 gets seqs 1, 3.
        for n in 0..5u64 {
            let core = u8::from(n % 2 == 1);
            recorder.record(GuestTicks(0), core, ThreadId(0), lifecycle(n));
        }
        let dump = flight.snapshot();
        assert_eq!(seqs(&dump, 0), [2, 4]);
        assert_eq!(seqs(&dump, 1), [1, 3]);
    }

    #[test]
    fn an_event_for_an_unknown_core_is_counted_not_kept_and_does_not_panic() {
        let flight = Arc::new(FlightRecorder::new(2, 2));
        let recorder = Recorder::new();
        recorder.add_sink(Arc::clone(&flight));
        recorder.record(GuestTicks(0), 2, ThreadId(0), lifecycle(0));
        recorder.record(GuestTicks(0), u8::MAX, ThreadId(0), lifecycle(1));
        assert_eq!(flight.unrouted(), 2);
        assert_eq!(flight.snapshot().merged().len(), 0);
    }

    #[test]
    fn zero_capacity_keeps_nothing_and_does_not_panic() {
        let flight = Arc::new(FlightRecorder::new(1, 0));
        let recorder = Recorder::new();
        recorder.add_sink(Arc::clone(&flight));
        recorder.record(GuestTicks(0), 0, ThreadId(0), lifecycle(0));
        assert_eq!(flight.snapshot().merged().len(), 0);
        assert_eq!(flight.capacity_per_core(), 0);
        assert_eq!(flight.cores(), 1);
    }

    #[test]
    fn dump_then_read_round_trips() {
        let flight = Arc::new(FlightRecorder::new(2, 3));
        let recorder = Recorder::new();
        recorder.add_sink(Arc::clone(&flight));
        for n in 0..10u64 {
            recorder.record(
                GuestTicks(n),
                u8::from(n % 3 == 0),
                ThreadId(n),
                lifecycle(n),
            );
        }
        let bytes = flight.dump().unwrap();
        assert_eq!(Dump::from_bytes(&bytes).unwrap(), flight.snapshot());
    }

    proptest! {
        #![proptest_config(config())]

        /// For any capacity, core count and event count, each ring holds the newest
        /// `min(capacity, count)` of its own core's events in order, and a dump round-trips.
        #[test]
        fn rings_hold_the_newest_per_core_and_dumps_round_trip(
            capacity in 0usize..9,
            cores in 1usize..4,
            route in proptest::collection::vec(0u8..4, 0..60),
        ) {
            let flight = Arc::new(FlightRecorder::new(cores, capacity));
            let recorder = Recorder::new();
            recorder.add_sink(Arc::clone(&flight));
            let mut per_core: Vec<Vec<u64>> = vec![Vec::new(); cores];
            let mut unrouted = 0u64;
            for (n, core) in route.iter().enumerate() {
                let seq = recorder.record(GuestTicks(0), *core, ThreadId(0), lifecycle(n as u64));
                match per_core.get_mut(usize::from(*core)) {
                    Some(list) => list.push(seq),
                    None => unrouted += 1,
                }
            }
            let dump = flight.snapshot();
            for (core, all) in per_core.iter().enumerate() {
                let keep = all.len().min(capacity);
                let expected = all[all.len() - keep..].to_vec();
                prop_assert_eq!(seqs(&dump, u8::try_from(core).unwrap()), expected);
            }
            prop_assert_eq!(flight.unrouted(), unrouted);
            let bytes = flight.dump().unwrap();
            prop_assert_eq!(Dump::from_bytes(&bytes).unwrap(), dump);
        }

        /// Arbitrary bytes are never a panic when read as a dump.
        #[test]
        fn arbitrary_bytes_never_panic_as_a_dump(bytes in proptest::collection::vec(any::<u8>(), 0..256)) {
            let _ = Dump::from_bytes(&bytes);
        }
    }
}
