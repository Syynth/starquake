//! A [`HostLog`] as the subscriber of the test's own thread. Each test logs from its own
//! statements, which no other test reaches, so `tracing`'s cache of which statements are
//! enabled is decided by these subscribers alone.

use super::*;

/// What `log` keeps of what `f` logs on this thread.
fn kept_by(log: &HostLog, f: impl FnOnce()) -> (Vec<String>, u64) {
    tracing::subscriber::with_default(log.clone(), f);
    log.take()
}

/// This module's target, as a line names it.
const TARGET: &str = module_path!();

#[test]
fn warnings_and_errors_are_kept_and_nothing_less_severe() {
    let log = HostLog::new();
    let (lines, dropped) = kept_by(&log, || {
        tracing::trace!("a trace");
        tracing::debug!("a debug line");
        tracing::info!("some information");
        tracing::warn!("a warning");
        tracing::error!("an error");
        tracing::info_span!("a span").in_scope(|| tracing::warn!("in a span"));
    });
    assert_eq!(
        lines,
        [
            format!("WARN {TARGET}: a warning"),
            format!("ERROR {TARGET}: an error"),
            format!("WARN {TARGET}: in a span"),
        ]
    );
    assert_eq!(dropped, 0);
    assert_eq!(KEPT_LEVEL, Level::WARN);
}

#[test]
fn a_line_writes_the_message_then_each_field_in_order() {
    /// A value whose `Display` and `Debug` differ.
    struct Both;
    impl fmt::Display for Both {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("shown")
        }
    }
    impl fmt::Debug for Both {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("Debugged")
        }
    }
    let log = HostLog::new();
    let name = "lm";
    let (lines, _) = kept_by(&log, || {
        tracing::warn!(
            service = "sm:",
            interface = %name,
            shown = %Both,
            debugged = ?Both,
            command = 5_u32,
            "unknown command auto-stubbed"
        );
    });
    assert_eq!(
        lines,
        [format!(
            "WARN {TARGET}: unknown command auto-stubbed service=sm: interface=lm shown=shown \
             debugged=Debugged command=5"
        )]
    );
}

#[test]
fn a_control_character_is_escaped_so_an_event_is_one_line() {
    let log = HostLog::new();
    let (lines, _) = kept_by(&log, || {
        tracing::warn!(name = "two\nlines\u{7f}", "a\rb");
    });
    assert_eq!(
        lines,
        [format!("WARN {TARGET}: a\\rb name=two\\nlines\\u{{7f}}")]
    );
}

#[test]
fn lines_are_kept_while_they_fit_and_none_after_the_first_that_does_not() {
    let line = format!("WARN {TARGET}: x");
    // Room for exactly two lines.
    let log = HostLog::with_limit(2 * line.len());
    let (lines, dropped) = kept_by(&log, || {
        for _ in 0..2 {
            tracing::warn!("x");
        }
    });
    assert_eq!(lines, [line.clone(), line.clone()]);
    assert_eq!(dropped, 0);

    // One byte short of the second line: it is dropped, and so is every line after it, even
    // one that would fit.
    let log = HostLog::with_limit(2 * line.len() - 1);
    let (lines, dropped) = kept_by(&log, || {
        tracing::warn!("x");
        tracing::warn!("x");
        tracing::warn!("");
        tracing::error!("y");
    });
    assert_eq!(lines, std::slice::from_ref(&line));
    assert_eq!(dropped, 3);

    // What was taken is gone, and the limit counts from the take.
    let (lines, dropped) = kept_by(&log, || tracing::warn!("x"));
    assert_eq!((lines, dropped), (vec![line], 0));
    assert_eq!(log.take(), (Vec::new(), 0));
}

#[test]
fn a_new_log_keeps_as_much_as_the_guest_log() {
    let log = HostLog::new();
    let kept = log.kept.lock().unwrap();
    assert_eq!(kept.limit, GUEST_LOG_BYTES);
    assert_eq!((kept.lines.len(), kept.bytes, kept.dropped), (0, 0, 0));
}

#[test]
fn the_error_is_plain_words() {
    assert_eq!(
        HostLogError::AlreadyInstalled.to_string(),
        "the process already has a log subscriber"
    );
}
