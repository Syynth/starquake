//! The front's end of the ring, as the register files and the macro host see it
//! ([`EntrySink`]), with what the front knows of the back's progress ([`RingSink`]).
//!
//! # A push that fails
//!
//! [`crate::regs::EntrySink::send`] cannot fail, but [`Producer::push`] can. A full ring is not a
//! failure: `push` waits for space, which waits only the front thread (spec, "Back-pressure").
//! What remains are [`PushError::Disconnected`] (the back thread is gone),
//! [`PushError::TooLarge`] (an entry longer than the whole ring: a [`RingEntry::Run`] is at most
//! [`crate::ring::MAX_RUN_WORDS`] words, so only a ring smaller than one maximal run can refuse
//! one) and [`PushError::Encode`] (a run past `MAX_RUN_WORDS`, which the register files never
//! send). The first such failure is kept, every later entry is dropped without a push, and the
//! front loop reports the kept failure from that turn on ([`super::FrontError::Ring`]), sticky:
//! a ring that lost an entry cannot be trusted with the ones after it
//! (`a_failed_push_is_sticky_and_keeps_the_first_failure`).
//!
//! # What the back has performed
//!
//! The macro host asks whether the back has work it has not performed ([`BackWork`]; spec,
//! "Macros"), and a GP entry with `SYNC_WAIT` waits for the same answer. It is answered here
//! from what the front pushed and the markers the back reported, not from the ring's consumed
//! position: the back pops a release or an increment long before it performs it (it waits for
//! host completion, decision 2), so an empty ring does not mean the work is done. Every entry
//! but a marker counts as work. A marker records how many had been pushed before it; when the
//! back reports it ([`RingSink::reported`]), everything before it is performed (spec, "Ordering
//! guarantees" 5). [`RingSink::has_unperformed`] is then whether anything was pushed after the
//! newest marker reported.

use std::collections::VecDeque;

use crate::macros::BackWork;
use crate::regs::EntrySink;
use crate::ring::{MarkerToken, Producer, PushError, RingEntry};

/// The front's end of the ring (module docs).
pub(crate) struct RingSink {
    producer: Producer,
    /// Entries other than markers pushed so far.
    pushed: u64,
    /// `pushed` as it stood at the newest marker the back reported.
    performed: u64,
    /// Markers pushed and not yet reported, oldest first, with `pushed` as it stood at each.
    markers: VecDeque<(MarkerToken, u64)>,
    /// The token the next marker takes.
    next_token: u64,
    /// The newest token the back reported.
    reported: Option<MarkerToken>,
    /// The first push that failed (module docs).
    error: Option<PushError>,
    /// Entries of every kind pushed so far, markers included: the ring position the back
    /// counts its entries against ([`RingSink::sent`]).
    sent: u64,
}

impl RingSink {
    /// The sink over `producer`, with nothing pushed.
    pub(crate) fn new(producer: Producer) -> Self {
        Self {
            producer,
            sent: 0,
            pushed: 0,
            performed: 0,
            markers: VecDeque::new(),
            next_token: 0,
            reported: None,
            error: None,
        }
    }

    /// The first failed push, if any (module docs).
    pub(crate) fn error(&self) -> Option<PushError> {
        self.error
    }

    /// Entries of every kind pushed so far, markers included. The back takes entries in push
    /// order, so the entry the back takes `n`-th (from 0) is the one pushed when this was `n`.
    pub(crate) fn sent(&self) -> u64 {
        self.sent
    }

    /// Makes everything pushed so far visible to the back.
    pub(crate) fn publish(&mut self) {
        self.producer.publish();
    }

    /// Takes the next marker token without pushing a marker, so the caller can file what the
    /// marker carries before the back can see it.
    #[must_use]
    pub(crate) fn reserve_token(&mut self) -> MarkerToken {
        let token = MarkerToken(self.next_token);
        self.next_token = self.next_token.wrapping_add(1);
        token
    }

    /// Pushes a marker with `token` (from [`RingSink::reserve_token`]).
    pub(crate) fn push_marker(&mut self, token: MarkerToken) {
        if self.push(&RingEntry::Marker { token }) {
            self.markers.push_back((token, self.pushed));
        }
    }

    /// Pushes a marker with a fresh token and returns the token.
    #[must_use]
    pub(crate) fn marker(&mut self) -> MarkerToken {
        let token = self.reserve_token();
        self.push_marker(token);
        token
    }

    /// The back reported `token`: everything pushed before it is performed, and so is
    /// everything before every earlier marker. A token the front never pushed, or one already
    /// reported, changes nothing.
    pub(crate) fn reported(&mut self, token: MarkerToken) {
        if !self.markers.iter().any(|&(pending, _)| pending == token) {
            return;
        }
        while let Some(&(pending, pushed)) = self.markers.front() {
            self.markers.pop_front();
            self.performed = self.performed.max(pushed);
            self.reported = Some(pending);
            if pending == token {
                break;
            }
        }
    }

    /// Whether the back has reported `token`, or a marker pushed after it.
    pub(crate) fn is_reported(&self, token: MarkerToken) -> bool {
        // Tokens are handed out in push order (a `u64` that would take centuries to wrap) and
        // markers are reported in ring order.
        self.reported.is_some_and(|newest| newest >= token)
    }

    /// Whether anything was pushed after the newest marker the back reported (module docs).
    pub(crate) fn has_unperformed(&self) -> bool {
        self.pushed > self.performed
    }

    /// Pushes one entry unless an earlier push failed; whether it was pushed.
    fn push(&mut self, entry: &RingEntry<'_>) -> bool {
        if self.error.is_some() {
            return false;
        }
        match self.producer.push(entry) {
            Ok(()) => {
                self.sent = self.sent.saturating_add(1);
                if !matches!(entry, RingEntry::Marker { .. }) {
                    self.pushed = self.pushed.saturating_add(1);
                }
                true
            }
            Err(error) => {
                tracing::error!(%error, "the GPU front could not push to the ring; it stops");
                self.error = Some(error);
                false
            }
        }
    }
}

impl EntrySink for RingSink {
    fn send(&mut self, entry: RingEntry<'_>) {
        self.push(&entry);
    }
}

/// [`BackWork`]'s answer, taken from a [`RingSink`] before a word goes to the macro host: a
/// run's writes are held until it finishes (`crate::macros`, "A run"), so nothing is pushed
/// while it runs and the answer cannot change under it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Unperformed(pub(crate) bool);

impl BackWork for Unperformed {
    fn has_unperformed_work(&self) -> bool {
        self.0
    }
}
