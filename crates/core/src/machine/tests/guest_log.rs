//! The services' way into the guest log ([`ServicesLog`]) and how the report takes the log
//! ([`take_log`]).

use super::*;

/// `lm`'s lines and `OutputDebugString`'s go into one log, in the order they were written, under
/// one limit: once a line does not fit, it and every later line, of either kind, are dropped.
#[test]
fn services_lines_share_the_debug_log_and_its_limit() {
    let log = Arc::new(Mutex::new(GuestLog::new(10)));
    let services = ServicesLog(Arc::clone(&log));
    // `LogSink::output` keeps a debug string this way.
    lock(&log).keep(b"abc");
    lm::GuestLog::line(&services, b"defg");
    lock(&log).keep(b"hij");
    assert_eq!(
        lock(&log).lines,
        [b"abc".to_vec(), b"defg".to_vec(), b"hij".to_vec()]
    );
    assert_eq!(lock(&log).kept_bytes, 10);
    // Exactly at the limit; the next byte is dropped, then everything after it.
    lm::GuestLog::line(&services, b"k");
    lock(&log).keep(b"");
    lm::GuestLog::line(&services, b"");
    let log = take_log(&log);
    assert_eq!(log.lines.len(), 3);
    assert_eq!(log.dropped, 3);
}

/// Taking the log leaves an empty one with the same limit behind, for anything that still
/// holds it.
#[test]
fn taking_the_log_leaves_an_empty_one_with_the_same_limit() {
    let log = Arc::new(Mutex::new(GuestLog::new(4)));
    let services = ServicesLog(Arc::clone(&log));
    lm::GuestLog::line(&services, b"four");
    lm::GuestLog::line(&services, b"x");
    let taken = take_log(&log);
    assert_eq!((taken.lines, taken.dropped), (vec![b"four".to_vec()], 1));
    let left = lock(&log);
    assert_eq!(
        (left.lines.len(), left.kept_bytes, left.limit, left.dropped),
        (0, 0, 4, 0)
    );
}
