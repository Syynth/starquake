//! The emulator's own warnings during a run, kept for the run's `log.txt`
//! ([`LOG_FILE`](crate::artifacts::LOG_FILE)).
//!
//! The kernel and the services report what a title does that starquake does not handle with
//! `tracing` (`sm:`'s name nothing provides, an auto-stubbed command, a refused request). A
//! [`HostLog`] is a `tracing` subscriber that keeps those events as lines of text, so a run's
//! artifacts say what happened without a scratch build (the owner's ruling of 2026-11-17 on
//! issue #207). The command-line tool installs one for the whole process
//! ([`HostLog::install`]) and puts what it kept into the run's report
//! ([`RunReport::host_log`](crate::RunReport::host_log)).
//!
//! # What it keeps
//!
//! - Events at [`KEPT_LEVEL`] and more severe: warnings and errors. Spans are not kept.
//! - Each as one line: the level, the event's target (its module), a colon, its message, then
//!   each other field as ` name=value` in the order the event gives them
//!   (`WARN starquake_services::sm: a service was asked for that nothing provides name=lm
//!   policy=AutoStub`). A value is written with its `Display` when the event gave it with `%`,
//!   and its `Debug` when with `?`; text is written as it is. A character that is a control
//!   character (a line break, say) is written escaped, so one event is one line.
//! - Bounded like the guest log ([`GUEST_LOG_BYTES`] of text, line breaks not counted): whole
//!   lines from the first while their total fits; from the first line that does not fit, every
//!   later line is counted and not kept ([`HostLog::take`]).

use std::fmt::{self, Write as _};
use std::sync::{Arc, Mutex, PoisonError};

use tracing::field::{Field, Visit};
use tracing::level_filters::LevelFilter;
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};

use crate::report::GUEST_LOG_BYTES;

/// The least severe level a [`HostLog`] keeps: warnings, and errors with them.
pub const KEPT_LEVEL: Level = Level::WARN;

/// Why a [`HostLog`] could not be installed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HostLogError {
    /// The process already has a subscriber for every thread.
    #[error("the process already has a log subscriber")]
    AlreadyInstalled,
}

/// What a [`HostLog`] keeps.
#[derive(Debug, Default)]
struct Kept {
    lines: Vec<String>,
    /// Bytes of text in `lines`.
    bytes: usize,
    /// Lines not kept: the first that did not fit, and every line after it.
    dropped: u64,
    /// Bytes the lines may hold.
    limit: usize,
}

impl Kept {
    /// Keeps `line` if it fits and nothing was dropped before it; counts it otherwise.
    fn keep(&mut self, line: String) {
        let fits = (self.bytes.checked_add(line.len())).filter(|total| *total <= self.limit);
        match fits {
            Some(total) if self.dropped == 0 => {
                self.bytes = total;
                self.lines.push(line);
            }
            _ => self.dropped = self.dropped.saturating_add(1),
        }
    }
}

/// A `tracing` subscriber that keeps warnings and errors as lines of text ([module docs](self)).
/// Clones share what they keep.
#[derive(Debug, Clone)]
pub struct HostLog {
    kept: Arc<Mutex<Kept>>,
}

impl Default for HostLog {
    fn default() -> Self {
        Self::new()
    }
}

impl HostLog {
    /// A log that keeps up to [`GUEST_LOG_BYTES`] of text.
    #[must_use]
    pub fn new() -> Self {
        Self::with_limit(GUEST_LOG_BYTES)
    }

    /// A log that keeps up to `limit` bytes of text.
    #[must_use]
    pub fn with_limit(limit: usize) -> Self {
        Self {
            kept: Arc::new(Mutex::new(Kept {
                limit,
                ..Kept::default()
            })),
        }
    }

    /// Makes this log the subscriber of every thread of the process, for the rest of its life.
    /// It can be done once per process.
    pub fn install(&self) -> Result<(), HostLogError> {
        tracing::subscriber::set_global_default(self.clone())
            .map_err(|_| HostLogError::AlreadyInstalled)
    }

    /// The lines kept so far, oldest first, and how many lines were not kept because the limit
    /// was reached. The log is empty afterwards and keeps from the start again: its limit
    /// counts from here.
    #[must_use]
    pub fn take(&self) -> (Vec<String>, u64) {
        let mut kept = self.kept.lock().unwrap_or_else(PoisonError::into_inner);
        let lines = std::mem::take(&mut kept.lines);
        let dropped = std::mem::take(&mut kept.dropped);
        kept.bytes = 0;
        (lines, dropped)
    }

    /// The line an event is written as.
    fn line(event: &Event<'_>) -> String {
        let metadata = event.metadata();
        let mut fields = Fields::default();
        event.record(&mut fields);
        let mut line = format!(
            "{} {}: {}",
            metadata.level(),
            metadata.target(),
            fields.message
        );
        line.push_str(&fields.rest);
        line
    }
}

/// An event's message and its other fields, as text.
#[derive(Default)]
struct Fields {
    message: String,
    /// ` name=value` for each other field.
    rest: String,
}

impl Fields {
    fn put(&mut self, field: &Field, value: fmt::Arguments<'_>) {
        let text = escaped(&value.to_string());
        if field.name() == "message" {
            self.message = text;
        } else {
            // Writing to a `String` cannot fail.
            let _ = write!(self.rest, " {}={text}", field.name());
        }
    }
}

impl Visit for Fields {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.put(field, format_args!("{value}"));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.put(field, format_args!("{value:?}"));
    }
}

/// `text` with each control character escaped (`\n`, `\u{7f}`), so it stays on one line.
fn escaped(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_control() {
                c.escape_default().to_string()
            } else {
                c.to_string()
            }
        })
        .collect()
}

impl Subscriber for HostLog {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.is_event() && *metadata.level() <= KEPT_LEVEL
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(LevelFilter::from_level(KEPT_LEVEL))
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        // Spans are not kept; every span gets the same id.
        Id::from_u64(1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        if !self.enabled(event.metadata()) {
            return;
        }
        // Written before the lock is taken: a field's `Display` that logged would otherwise
        // wait for this thread's own lock.
        let line = Self::line(event);
        let mut kept = self.kept.lock().unwrap_or_else(PoisonError::into_inner);
        kept.keep(line);
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

#[cfg(test)]
mod tests;
