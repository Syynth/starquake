//! The recorder: assigns sequence numbers and fans events out to sinks.
//!
//! Source: `docs/specs/observability.md`, "The event stream" ("`seq` is assigned under the kernel
//! lock, so sorting by it reproduces the kernel's order"; "Sinks consume the same events").
//!
//! The kernel lock is not part of this crate, so the [`Recorder`] has its own: sequence
//! assignment and delivery to every sink happen under one mutex. That makes two things hold even
//! when several host threads record at once, without the caller holding any lock: sequence
//! numbers are unique and strictly increasing (up to the saturation point described on
//! [`Recorder::record`]), and every sink sees events in sequence order.
//! The cost is one uncontended-mutex acquisition per event; the measured total is in
//! `tests/recording_cost.rs`.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::event::{Event, EventKind, GuestTicks, ThreadId};

/// A consumer of the event stream.
///
/// [`Recorder::record`] calls [`Sink::record`] while holding the recorder's lock, so a sink sees
/// events in sequence order. A sink must not call back into the recorder that is feeding it (that
/// would deadlock) and should be quick.
pub trait Sink: Send + Sync {
    /// Consume one event.
    fn record(&self, event: &Event);
}

/// A sink that keeps every event, in arrival order. Unbounded; meant for tests and for building a
/// full log in memory.
#[derive(Debug, Default)]
pub struct VecSink {
    events: Mutex<Vec<Event>>,
}

impl VecSink {
    /// An empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A copy of every event received so far, in arrival order.
    #[must_use]
    pub fn events(&self) -> Vec<Event> {
        lock(&self.events).clone()
    }
}

impl Sink for VecSink {
    fn record(&self, event: &Event) {
        lock(&self.events).push(*event);
    }
}

/// Lock a mutex, ignoring poison: every critical section in this crate leaves its data valid at
/// each step, so a panic elsewhere (a test failure, say) must not make recording panic too.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Default)]
struct State {
    next_seq: u64,
    sinks: Vec<Arc<dyn Sink>>,
}

/// Assigns sequence numbers and fans events out to sinks.
///
/// One recorder serves a whole run; every boundary reports to it once (spec: "one recording point
/// per boundary").
#[derive(Default)]
pub struct Recorder {
    state: Mutex<State>,
}

impl Recorder {
    /// A recorder with no sinks; its first event gets sequence number 0.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a sink. It receives every event recorded from now on. The caller keeps its own
    /// `Arc` to read the sink back.
    pub fn add_sink<S: Sink + 'static>(&self, sink: Arc<S>) {
        lock(&self.state).sinks.push(sink);
    }

    /// Record one event: assign it the next sequence number, deliver it to every sink in the
    /// order they were attached, and return the number.
    ///
    /// The counter saturates at `u64::MAX` rather than wrapping. The first 2^64 - 1 events get
    /// the numbers 0 to `u64::MAX` - 1, each once; from the 2^64th event on, every event gets
    /// `u64::MAX`, so from then on `seq` is no longer strictly increasing and sorting by it no
    /// longer reproduces the recording order. (Reaching that point takes 2^64 events.)
    pub fn record(&self, time: GuestTicks, core: u8, thread: ThreadId, kind: EventKind) -> u64 {
        let mut state = lock(&self.state);
        let seq = state.next_seq;
        state.next_seq = seq.saturating_add(1);
        let event = Event {
            seq,
            time,
            core,
            thread,
            kind,
        };
        for sink in &state.sinks {
            sink.record(&event);
        }
        seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{LifecyclePhase, ResultCode};

    fn note(n: u64) -> EventKind {
        EventKind::Lifecycle {
            phase: LifecyclePhase::ThreadCreate,
            subject: n,
        }
    }

    #[test]
    fn numbers_events_from_zero_and_fans_out_to_every_sink() {
        let recorder = Recorder::new();
        let a = Arc::new(VecSink::new());
        let b = Arc::new(VecSink::new());
        recorder.add_sink(Arc::clone(&a));
        recorder.add_sink(Arc::clone(&b));
        let first = recorder.record(GuestTicks(5), 2, ThreadId(7), note(1));
        let second = recorder.record(GuestTicks(6), 3, ThreadId(8), note(2));
        assert_eq!((first, second), (0, 1));
        let expected = vec![
            Event {
                seq: 0,
                time: GuestTicks(5),
                core: 2,
                thread: ThreadId(7),
                kind: note(1),
            },
            Event {
                seq: 1,
                time: GuestTicks(6),
                core: 3,
                thread: ThreadId(8),
                kind: note(2),
            },
        ];
        assert_eq!(a.events(), expected);
        assert_eq!(b.events(), expected);
    }

    #[test]
    fn a_sink_attached_late_sees_only_later_events() {
        let recorder = Recorder::new();
        recorder.record(GuestTicks(0), 0, ThreadId(0), note(0));
        let late = Arc::new(VecSink::new());
        recorder.add_sink(Arc::clone(&late));
        recorder.record(GuestTicks(0), 0, ThreadId(0), note(1));
        let seen = late.events();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].seq, 1);
    }

    #[test]
    fn the_counter_saturates_and_repeats_the_last_number() {
        let recorder = Recorder::new();
        lock(&recorder.state).next_seq = u64::MAX - 1;
        let seqs: Vec<u64> = (0..4)
            .map(|n| recorder.record(GuestTicks(0), 0, ThreadId(0), note(n)))
            .collect();
        assert_eq!(seqs, [u64::MAX - 1, u64::MAX, u64::MAX, u64::MAX]);
    }

    #[test]
    fn recording_with_no_sink_still_numbers_events() {
        let recorder = Recorder::new();
        let kind = EventKind::Svc {
            id: 1,
            args: [0; crate::event::SVC_ARG_COUNT],
            result: ResultCode(0),
        };
        assert_eq!(recorder.record(GuestTicks(0), 0, ThreadId(0), kind), 0);
        assert_eq!(recorder.record(GuestTicks(0), 0, ThreadId(0), kind), 1);
    }
}
