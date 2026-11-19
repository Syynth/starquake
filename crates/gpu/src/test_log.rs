//! Log capture for tests: a `tracing` subscriber that keeps every event's level and fields, so
//! a test can assert a documented log level or log-once policy.
//!
//! # One subscriber for the whole test binary
//!
//! A subscriber installed for one thread (`tracing::subscriber::with_default`) can miss events
//! under plain `cargo test`, where the tests run as threads of one process. While at most one
//! subscriber is registered, `tracing-core` (0.1.36, `callsite.rs`, `Dispatchers::rebuilder`
//! and `Rebuilder::for_each`) decides whether a log statement is enabled by asking the default
//! subscriber of whichever thread reaches the statement first, and caches the answer. A test
//! with no subscriber of its own that reaches a `warn!` first, while a capturing test's
//! subscriber is the only one registered, caches "never", and the capturing test then misses
//! that warning (`crates/services/tests/unknown_command_logs.rs` has the same account).
//! `cargo nextest` never shows it, since it runs each test in its own process.
//!
//! So no thread's default is ever set here. [`captured`] installs one subscriber for the whole
//! process, the first time it is called, that is enabled for every event and hands each to the
//! capture of the thread that logged it, if that thread is inside [`captured`]. Every thread
//! that asks then gets that subscriber's answer, "always". A statement that a thread reached
//! before the subscriber was installed cached "never"; installing it, and every later call of
//! [`captured`], rebuilds every cached answer. (That covers a thread whose first visit to a
//! statement straddles the installation, a window too narrow for a test to hit on purpose.)
//!
//! `tests::an_event_is_captured_when_another_thread_reached_its_statement_first` sets that
//! order up on purpose: it failed with the thread-default helper this replaces, and passes with
//! this one.

use std::cell::RefCell;
use std::fmt::{self, Write as _};
use std::sync::Once;

/// Captured events: each event's level and its fields written `name=value;`.
type Events = Vec<(tracing::Level, String)>;

thread_local! {
    /// The events of this thread's innermost [`captured`] call, while one runs.
    static CAPTURE: RefCell<Option<Events>> = const { RefCell::new(None) };
}

/// The process-wide subscriber: every event goes to its thread's capture, if it has one.
struct Router;

struct Fields<'a>(&'a mut String);

impl tracing::field::Visit for Fields<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        write!(self.0, "{}={value:?};", field.name()).unwrap();
    }
}

impl tracing::Subscriber for Router {
    fn register_callsite(&self, _: &tracing::Metadata<'_>) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::always()
    }
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::TRACE)
    }
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        if !CAPTURE.with(|capture| capture.borrow().is_some()) {
            return;
        }
        // Written before the capture is borrowed, so a field that logs does not borrow it twice.
        let mut fields = String::new();
        event.record(&mut Fields(&mut fields));
        CAPTURE.with(|capture| {
            if let Some(events) = capture.borrow_mut().as_mut() {
                events.push((*event.metadata().level(), fields));
            }
        });
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// Puts back the capture that was in place before a [`captured`] call, even if `f` panics.
struct Restore(Option<Events>);

impl Drop for Restore {
    fn drop(&mut self) {
        let outer = self.0.take();
        CAPTURE.with(|capture| *capture.borrow_mut() = outer);
    }
}

/// Runs `f` with every event on this thread captured: its result and the events, each as its
/// level and its fields written `name=value;`.
pub(crate) fn captured<T>(f: impl FnOnce() -> T) -> (T, Events) {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        tracing::subscriber::set_global_default(Router)
            .expect("no other global subscriber in this test binary");
    });
    tracing::callsite::rebuild_interest_cache();
    let restore = Restore(CAPTURE.with(|capture| capture.borrow_mut().replace(Vec::new())));
    let result = f();
    let events = CAPTURE
        .with(|capture| capture.borrow_mut().take())
        .unwrap_or_default();
    drop(restore);
    (result, events)
}

/// The events at `level`, as captured.
pub(crate) fn at(events: &[(tracing::Level, String)], level: tracing::Level) -> Vec<String> {
    events
        .iter()
        .filter(|(l, _)| *l == level)
        .map(|(_, fields)| fields.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;

    use super::*;

    /// A log statement no other code reaches.
    fn probe() {
        tracing::warn!("the capture probe");
    }

    #[test]
    fn an_event_is_captured_when_another_thread_reached_its_statement_first() {
        // While this thread captures, a thread with no capture of its own reaches the statement
        // first; the event this thread then logs from it is still captured.
        let (inside, reached_inside) = mpsc::channel();
        let (first, reached_first) = mpsc::channel();
        let other = thread::spawn(move || {
            reached_inside.recv().unwrap();
            probe();
            first.send(()).unwrap();
        });
        let ((), events) = captured(|| {
            inside.send(()).unwrap();
            reached_first.recv().unwrap();
            probe();
        });
        other.join().unwrap();
        assert_eq!(
            at(&events, tracing::Level::WARN),
            ["message=the capture probe;"],
            "{events:?}"
        );
    }

    #[test]
    fn a_nested_capture_keeps_its_own_events_and_the_outer_one_goes_on() {
        let (((), inner), outer) = captured(|| {
            tracing::warn!("before");
            let inner = captured(|| tracing::warn!("inside"));
            tracing::warn!("after");
            inner
        });
        assert_eq!(at(&inner, tracing::Level::WARN), ["message=inside;"]);
        assert_eq!(
            at(&outer, tracing::Level::WARN),
            ["message=before;", "message=after;"]
        );
        // Outside every capture, nothing is kept.
        tracing::warn!("outside");
        let ((), events) = captured(|| {});
        assert_eq!(events, []);
    }
}
