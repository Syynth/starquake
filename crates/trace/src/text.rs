//! The golden-trace text view of the event stream.
//!
//! Source: `docs/specs/observability.md` ("the golden-trace text format from the test strategy,
//! which is a per-thread view of this stream with timing removed") and
//! `docs/specs/testing.md`, "Golden traces" ("grouped per guest thread so that cross-thread timing
//! doesn't change the file").
//!
//! The view is a function of each thread's own event sequence only. It leaves out `seq`, `time`
//! and `core` (all differ between runs that did the same thing), orders threads by id, and keeps
//! each thread's events in the order they were recorded. Two recordings that interleave the same
//! per-thread sequences differently therefore render to identical text. Wake, map, fault and
//! schedule events are left out, because they depend on timing (see
//! [`EventKind::in_golden_trace`](crate::EventKind::in_golden_trace)).
//!
//! The line format itself is this crate's own design (the spec gives none); it is the
//! `Display` of [`EventKind`](crate::EventKind), pinned by tests here and in `event.rs`.

use std::collections::BTreeMap;

use std::sync::Mutex;

use crate::event::{Event, ThreadId};
use crate::recorder::{Sink, lock};

/// Render events as the golden-trace view.
///
/// `events` may be in any order across threads; they are first sorted by `seq`, so each thread's
/// lines follow the order the recorder assigned. Each thread gets a `thread N` header, then one
/// two-space-indented line per event; threads appear in increasing id order. The result ends with
/// a newline unless there is nothing to show, in which case it is empty.
#[must_use]
pub fn golden_trace<'a>(events: impl IntoIterator<Item = &'a Event>) -> String {
    let mut ordered: Vec<&Event> = events.into_iter().collect();
    ordered.sort_by_key(|e| e.seq);
    let mut threads: BTreeMap<ThreadId, Vec<String>> = BTreeMap::new();
    for event in ordered.into_iter().filter(|e| e.kind.in_golden_trace()) {
        threads
            .entry(event.thread)
            .or_default()
            .push(format!("  {}", event.kind));
    }
    let mut lines = Vec::new();
    for (thread, thread_lines) in threads {
        lines.push(format!("thread {}", thread.0));
        lines.extend(thread_lines);
    }
    lines.join("\n") + if lines.is_empty() { "" } else { "\n" }
}

/// A sink that keeps events and renders the golden-trace view on request.
///
/// [`TextSink::new`] keeps every event, so it is meant for tests. A run of unknown length uses
/// [`TextSink::with_limit`], which keeps the first events up to a limit and counts the rest, so
/// the sink's memory is bounded however long the guest runs. The first events are the ones kept
/// because they are the reproducible part of a run; the newest are in the flight recorder.
/// Events the view does not show ([`EventKind::in_golden_trace`](crate::EventKind::in_golden_trace))
/// are not stored and do not count towards the limit.
#[derive(Debug, Default)]
pub struct TextSink {
    state: Mutex<State>,
    limit: Option<usize>,
}

#[derive(Debug, Default)]
struct State {
    events: Vec<Event>,
    dropped: u64,
}

impl TextSink {
    /// An empty sink that keeps every event.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty sink that keeps the first `limit` events the view shows and counts the rest.
    #[must_use]
    pub fn with_limit(limit: usize) -> Self {
        Self {
            state: Mutex::default(),
            limit: Some(limit),
        }
    }

    /// How many events arrived after the limit was reached and are not in the view.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        lock(&self.state).dropped
    }

    /// The golden-trace view of the events kept (see [`golden_trace`]). When events were dropped,
    /// one more line follows the view and says how many:
    /// `truncated: <n> later events are not shown (the trace keeps its first <limit>)`.
    #[must_use]
    pub fn render(&self) -> String {
        let state = lock(&self.state);
        let mut text = golden_trace(state.events.iter());
        if let (Some(limit), dropped @ 1..) = (self.limit, state.dropped) {
            let marker = format!(
                "truncated: {dropped} later events are not shown (the trace keeps its first \
                 {limit})\n"
            );
            text.push_str(&marker);
        }
        text
    }
}

impl Sink for TextSink {
    fn record(&self, event: &Event) {
        if !event.kind.in_golden_trace() {
            return;
        }
        let mut state = lock(&self.state);
        if self.limit.is_some_and(|limit| state.events.len() >= limit) {
            state.dropped = state.dropped.saturating_add(1);
        } else {
            state.events.push(*event);
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
    use crate::event::{
        EventKind, FaultAction, GuestTicks, LifecyclePhase, MapKind, ResultCode, SVC_ARG_COUNT,
        WakeReason,
    };

    fn config() -> ProptestConfig {
        ProptestConfig {
            cases: 256,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(0x474F_4C44_454E_5452),
            failure_persistence: None,
            ..Config::default()
        }
    }

    fn svc(id: u32) -> EventKind {
        EventKind::Svc {
            id,
            args: [u64::from(id); SVC_ARG_COUNT],
            result: ResultCode(0),
        }
    }

    /// Record `plan` (thread, kind) pairs in order, with times and cores that depend on the
    /// position in the plan, and render the sink's view.
    fn render(plan: &[(u64, EventKind)]) -> String {
        let recorder = Recorder::new();
        let sink = Arc::new(TextSink::new());
        recorder.add_sink(Arc::clone(&sink));
        for (n, (thread, kind)) in plan.iter().enumerate() {
            recorder.record(
                GuestTicks(1000 + (n as u64) * 7),
                u8::try_from(n % 6).unwrap(),
                ThreadId(*thread),
                *kind,
            );
        }
        sink.render()
    }

    #[test]
    fn two_interleavings_of_the_same_threads_render_identically() {
        let a = [svc(1), svc(2), svc(3)];
        let b = [
            EventKind::Lifecycle {
                phase: LifecyclePhase::ThreadCreate,
                subject: 2,
            },
            svc(9),
        ];
        let tag = |thread: u64, kinds: &[EventKind]| -> Vec<(u64, EventKind)> {
            kinds.iter().map(|k| (thread, *k)).collect()
        };
        let (ta, tb) = (tag(1, &a), tag(2, &b));
        // Interleaving one: a b a b a. Interleaving two: b b a a a.
        let one = [ta[0], tb[0], ta[1], tb[1], ta[2]];
        let two = [tb[0], tb[1], ta[0], ta[1], ta[2]];
        let text = render(&one);
        assert_eq!(text, render(&two));
        assert_eq!(
            text,
            "thread 1\n  svc 0x1(0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1) -> 0x0\n  \
             svc 0x2(0x2, 0x2, 0x2, 0x2, 0x2, 0x2, 0x2, 0x2) -> 0x0\n  \
             svc 0x3(0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3) -> 0x0\n\
             thread 2\n  lifecycle ThreadCreate 2\n  \
             svc 0x9(0x9, 0x9, 0x9, 0x9, 0x9, 0x9, 0x9, 0x9) -> 0x0\n"
        );
    }

    /// Records `kinds` in order on thread 1 into `sink`.
    fn feed(sink: &Arc<TextSink>, kinds: &[EventKind]) {
        let recorder = Recorder::new();
        recorder.add_sink(Arc::clone(sink));
        for kind in kinds {
            recorder.record(GuestTicks(0), 0, ThreadId(1), *kind);
        }
    }

    #[test]
    fn a_limited_sink_keeps_its_first_events_and_says_how_many_it_dropped() {
        const LIMIT: usize = 3;
        let sink = Arc::new(TextSink::with_limit(LIMIT));
        // A fault is not shown in the view, so it is neither kept nor counted.
        let fault = EventKind::Fault {
            address: 0x10,
            write: true,
            action: FaultAction::Fatal,
        };
        feed(
            &sink,
            &[svc(1), fault, svc(2), svc(3), svc(4), fault, svc(5)],
        );
        assert_eq!(sink.dropped(), 2);
        assert_eq!(
            sink.render(),
            render(&[(1, svc(1)), (1, svc(2)), (1, svc(3))])
                + "truncated: 2 later events are not shown (the trace keeps its first 3)\n"
        );
    }

    #[test]
    fn a_sink_exactly_at_its_limit_is_not_truncated() {
        const LIMIT: usize = 3;
        let sink = Arc::new(TextSink::with_limit(LIMIT));
        let kinds = [svc(1), svc(2), svc(3)];
        assert_eq!(kinds.len(), LIMIT);
        feed(&sink, &kinds);
        assert_eq!(sink.dropped(), 0);
        assert_eq!(
            sink.render(),
            render(&[(1, svc(1)), (1, svc(2)), (1, svc(3))])
        );
        // One more is the first to be dropped.
        feed(&sink, &[svc(4)]);
        assert_eq!(sink.dropped(), 1);
    }

    #[test]
    fn a_limit_of_zero_keeps_nothing_but_the_count_and_no_limit_keeps_everything() {
        let none = Arc::new(TextSink::with_limit(0));
        feed(&none, &[svc(1), svc(2)]);
        assert_eq!(
            none.render(),
            "truncated: 2 later events are not shown (the trace keeps its first 0)\n"
        );
        let all = Arc::new(TextSink::new());
        let many: Vec<EventKind> = (0..1000).map(svc).collect();
        feed(&all, &many);
        assert_eq!(all.dropped(), 0);
        assert_eq!(all.render().lines().count(), 1001);
    }

    #[test]
    fn within_a_thread_order_still_matters() {
        let swapped = render(&[(1, svc(1)), (1, svc(2))]);
        assert_ne!(swapped, render(&[(1, svc(2)), (1, svc(1))]));
    }

    #[test]
    fn timing_dependent_events_are_not_shown_and_an_empty_view_is_empty() {
        let hidden = [
            EventKind::Schedule {
                core: 0,
                from: None,
                to: Some(ThreadId(1)),
            },
            EventKind::Wake {
                thread: ThreadId(1),
                reason: WakeReason::Signalled,
            },
            EventKind::Map {
                va: 0x1000,
                size: 0x2000,
                perms: 3,
                kind: MapKind::Map,
            },
            EventKind::Fault {
                address: 0x10,
                write: true,
                action: FaultAction::Fatal,
            },
        ];
        for kind in hidden {
            assert_eq!(render(&[(1, kind)]), "", "{kind}");
            // Hidden events leave the lines around them as they are.
            assert_eq!(
                render(&[(1, svc(1)), (1, kind), (1, svc(2))]),
                render(&[(1, svc(1)), (1, svc(2))]),
                "{kind}"
            );
        }
        assert_eq!(render(&[]), "");
    }

    #[test]
    fn events_are_ordered_by_seq_not_by_the_order_they_are_handed_in() {
        let recorder = Recorder::new();
        let sink = Arc::new(crate::VecSink::new());
        recorder.add_sink(Arc::clone(&sink));
        for id in 1..=3 {
            recorder.record(GuestTicks(0), 0, ThreadId(1), svc(id));
        }
        let events = sink.events();
        let reversed: Vec<Event> = events.iter().rev().copied().collect();
        assert_eq!(golden_trace(&events), golden_trace(&reversed));
    }

    fn thread_plan() -> impl Strategy<Value = Vec<Vec<EventKind>>> {
        let kind = prop_oneof![
            (0u32..100).prop_map(svc),
            (0u64..50).prop_map(|n| EventKind::Lifecycle {
                phase: LifecyclePhase::ThreadCreate,
                subject: n,
            }),
            (0u32..50, any::<u64>()).prop_map(|(len, hash)| EventKind::GuestLog { len, hash }),
        ];
        proptest::collection::vec(proptest::collection::vec(kind, 0..6), 1..5)
    }

    proptest! {
        #![proptest_config(config())]

        /// Any two interleavings of the same per-thread sequences (with different timing and
        /// cores) give the same text; the interleaving is chosen by a generated schedule.
        #[test]
        fn any_interleaving_gives_the_same_text(
            threads in thread_plan(),
            choices_a in proptest::collection::vec(any::<usize>(), 0..40),
            choices_b in proptest::collection::vec(any::<usize>(), 0..40),
        ) {
            let interleave = |choices: &[usize]| -> Vec<(u64, EventKind)> {
                let mut cursors = vec![0usize; threads.len()];
                let mut out = Vec::new();
                let mut picks = choices.iter().copied().cycle();
                let total: usize = threads.iter().map(Vec::len).sum();
                while out.len() < total {
                    let pick = picks.next().unwrap_or(0);
                    // Choose the pick-th thread that still has events left.
                    let live: Vec<usize> = (0..threads.len())
                        .filter(|t| cursors[*t] < threads[*t].len())
                        .collect();
                    let t = live[pick % live.len()];
                    out.push((t as u64, threads[t][cursors[t]]));
                    cursors[t] += 1;
                }
                out
            };
            prop_assert_eq!(render(&interleave(&choices_a)), render(&interleave(&choices_b)));
        }
    }
}
