//! Log capture for tests: a `tracing` subscriber that keeps every event's level and fields, so
//! a test can assert a documented log level or log-once policy.

use std::fmt::{self, Write as _};
use std::sync::{Arc, Mutex};

/// A subscriber that keeps every event's level and fields.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<(tracing::Level, String)>>>);

struct Fields<'a>(&'a mut String);

impl tracing::field::Visit for Fields<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        write!(self.0, "{}={value:?};", field.name()).unwrap();
    }
}

impl tracing::Subscriber for Capture {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut fields = String::new();
        event.record(&mut Fields(&mut fields));
        self.0
            .lock()
            .unwrap()
            .push((*event.metadata().level(), fields));
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// Runs `f` with every event on this thread captured: its result and the events, each as its
/// level and its fields written `name=value;`.
pub(crate) fn captured<T>(f: impl FnOnce() -> T) -> (T, Vec<(tracing::Level, String)>) {
    let capture = Capture::default();
    let result = tracing::subscriber::with_default(capture.clone(), f);
    let events = capture.0.lock().unwrap().clone();
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
