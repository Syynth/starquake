//! The vsync clock: the 60 Hz deadline from the `Env` clock, ticks that do not bank lost time,
//! and the request-group ticks (`docs/specs/graphics-server.md`, "Display: vi2", "The vsync
//! clock"; port slice 12).
//!
//! Ported from (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project), cited below by these
//! short forms, as in [`crate::protocol`]:
//!
//! - **NX2:** Eden-NX2, `eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb:
//!   `INx2Vi2Session::PaceToVblank` (when the next vblank is, and not banking lost time) and
//!   `INx2Vi2Session::GetVsyncEvent` (its `VblankInterval`, and a timer that signals every vsync
//!   event it handed out).
//! - **PR:** Eden PR #4588, `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp`
//!   @ e5fb3b97: `IVi2Session::StartGroupTick` (what a tick signals: every request group
//!   registered for display event kind 0, the vsync) and `RegisterRequestGroupMonitor` (the
//!   first kind-0 registration starts it).
//!
//! # The clock
//!
//! The clock runs while something listens to it. A listener is a kernel event the tick signals:
//! a vi2 request group's event once 261 registers the group for kind 0, or, under the spec's
//! fallback, a group event 31 or 32 handed out that nothing registered (spec, "Vsync events:
//! conflict"). Registering and unregistering them is vi2's (slice 13), through
//! [`Locked::listen_vsync`] and [`Locked::unlisten_vsync`]; this slice builds the clock they
//! drive.
//!
//! - **Start.** The first listener starts the clock: its deadline is one [`INTERVAL_NS`] after
//!   the time the caller gives, which is the `Env` clock's reading on the caller's thread (a
//!   handler reads it with `Context::env`), as PR's first kind-0 registration schedules its
//!   tick one interval after the registration
//!   (`the_first_listener_starts_the_clock_one_interval_out`). A listener added to a running
//!   clock does not move its deadline (`a_second_listener_joins_the_running_clock`).
//! - **Tick.** The timer thread's vsync turn ([`Shared::tick_vsync`], called with the turn's time
//!   after each drain) ticks when the time has reached the deadline, never before
//!   (`a_tick_comes_at_exactly_the_deadline_and_not_before`): it counts the tick, records it
//!   ("What is recorded"), and signals every listener, in ascending event order
//!   (`a_tick_records_its_number_then_signals_every_listener_in_event_order`). The slice-14
//!   latch will run on the same tick.
//! - **The next deadline: no banking.** NX2 `PaceToVblank`, step for step: `if (now >
//!   m_next_vblank + interval) m_next_vblank = now;` then `m_next_vblank += interval;`. So a turn
//!   at most one interval late keeps the clock on its grid (the next deadline is one interval
//!   after the one just met), and a turn more than one interval late ticks once and restarts the
//!   grid one interval after the turn: the lost time is not banked, and one turn never delivers
//!   more than one tick (`a_late_tick_does_not_double_up`). On a fixed clock whose turns come on
//!   time the ticks fall at exact multiples of [`INTERVAL_NS`] from the start
//!   (`on_a_fixed_clock_ticks_fall_at_exact_multiples`). At exactly one interval late NX2's `>`
//!   does not restart: the next deadline is the turn's own time, so the next turn ticks again
//!   (`a_tick_late_by_exactly_one_interval_keeps_the_grid`, beside
//!   `a_tick_late_by_one_interval_and_a_nanosecond_restarts_the_grid`).
//! - **Stop.** When the last listener leaves ([`Locked::unlisten_vsync`], or a refused signal
//!   below) the clock stops: no deadline, so the timer thread neither wakes for it nor reads the
//!   clock for it. The tick count is kept, and the next listener starts the clock again with a
//!   new phase (`the_last_listener_leaving_stops_the_clock_and_the_next_restarts_it`).
//!
//! **A refused signal.** A listener whose signal the kernel refuses (its event is gone: vi2
//! closed it without unlistening it, or a restored snapshot named one the kernel does not hold)
//! no longer listens: it is logged as an error and taken out as [`Locked::unlisten_vsync`] would,
//! after the tick has signalled the rest (`a_refused_listener_is_dropped_and_the_rest_are_signalled`;
//! the level is pinned by `tests/vsync_log_levels.rs`, a test binary of its own).
//!
//! **The end of the counts.** The deadline is a `u64` of host nanoseconds and the tick count a
//! `u64`. Neither runs out in a run (2^64 ns is 584 years), but a restored snapshot can hold any
//! value, so neither wraps or saturates: a deadline one interval past `u64::MAX` cannot be
//! written, so the tick that would need it is delivered and the clock stops there
//! (`a_deadline_past_the_end_of_time_stops_the_clock`); a tick count at `u64::MAX` delivers no
//! further tick and stops the clock (`a_tick_count_at_its_maximum_stops_the_clock`). Both are
//! logged as errors (`tests/vsync_log_levels.rs`). A stopped clock with listeners starts again
//! only when its listeners all leave and one comes back.
//!
//! **The interval: conflict.** NX2 ticks every `1'000'000'000LL / hz` nanoseconds with `hz` 60
//! (`VblankInterval`), and `PaceToVblank`'s `100'000'000'000LL / 60 / SpeedLimit()` is the same
//! 16,666,666 ns at the default speed limit of 100 %; PR ticks every
//! `std::chrono::microseconds{16667}` (16,667,000 ns). Neither answer contains the other and
//! neither cites the system module, so decision 8's house rule takes the one a title ran on:
//! NX2's, which reaches gameplay (`docs/switch2-eden-diff-scoping.md`, "Status as of the tips";
//! PR #4588 is a draft there). [`INTERVAL_NS`] is NX2's
//! (`the_interval_and_the_note_code_are_the_references_values`). The spec's "What to verify
//! before building", item 1 (which reference wins where they disagree), is the open question: a
//! trace of the vsync event's signals on a console, or of the system module's display worker,
//! settles it.
//!
//! # Replay
//!
//! In a replay the clock does not tick on its own: a recorded tick is fed back with
//! [`Shared::replay_vsync`], which takes only the next number (the count delivered and fed back
//! so far, plus one: [`VsyncReplayError::OutOfOrder`]), and the next replay turn
//! ([`Shared::tick_replayed_vsync`]) delivers every tick fed back since the last one, in order,
//! as a live tick is delivered: counted, recorded with the same number, and signalled
//! (`a_replayed_tick_is_delivered_as_the_live_one_was`). Which of the two turns a run takes is
//! the wiring's (`starquake-core`'s `hle::graphics`, by its front's mode).
//!
//! # What is recorded
//!
//! Each tick is recorded as a `VsyncTick` note ([`crate::note`], code 4) with its number, the
//! count of ticks delivered so far, this one included, before it signals any listener. A tick
//! with no listener left to signal is recorded too. The number is the clock's only input from
//! outside the guest: the times of the turns are host timing, which no note keeps, so a replay
//! follows the recorded ticks, not the clock (above).
//!
//! # Where this differs from the references
//!
//! Each row is pinned by the test named, in `shared/vsync/tests.rs`:
//!
//! | Case | PR | NX2 | Here |
//! |---|---|---|---|
//! | The interval | 16,667,000 ns | 16,666,666 ns | **conflict**, NX2's (above; `the_interval_and_the_note_code_are_the_references_values`) |
//! | A tick a turn meets late | every missed tick, each in turn (Eden's `CoreTiming::Advance` fires a looping event again while its time is still behind) | one, and lost time is not banked (`PaceToVblank`) | NX2's, as the spec says (`a_late_tick_does_not_double_up`) |
//! | How many clocks | one per vi2 session (`m_group_tick`) | one per vi2 session (`m_vsync_timer`) | one for the graphics server: every listener ticks together (`a_tick_records_its_number_then_signals_every_listener_in_event_order`) |
//! | When it stops | when the session is destroyed | when the session is destroyed | when the last listener leaves, the next start taking a new phase (`the_last_listener_leaving_stops_the_clock_and_the_next_restarts_it`) |
//! | Speed limit | none | `PaceToVblank` paces nothing when the speed limit is off, and scales the interval with it | none: always [`INTERVAL_NS`] (there is no speed-limit setting) |
//! | A listener the kernel no longer holds | cannot happen (PR holds the events) | cannot happen (NX2 holds the events) | dropped and logged (`a_refused_listener_is_dropped_and_the_rest_are_signalled`) |
//! | How many listeners | 8 request groups per session (`std::array<Kernel::KEvent*, 8> m_request_group_events`), sessions not counted | one per 31 or 32 call, without bound | at most [`MAX_LISTENERS`] in all (`listeners_are_refused_past_the_cap`) |
//!
//! [`MAX_LISTENERS`] is starquake's own bound, a **guess**: room for eight vi2 sessions of PR's
//! eight groups each. It bounds what a title can make the timer thread signal on every tick.
//!
//! # What is saved
//!
//! [`Shared::save_vsync`] writes the clock: `postcard`, after a header of [`SNAPSHOT_MAGIC`] and
//! [`SNAPSHOT_VERSION`] that is decoded and checked before the body. The body is, in order, the
//! tick count, the next deadline (`None` while stopped), the ticks fed back for replay and not
//! yet delivered, and the listeners in ascending order (the spec's "the vsync tick and next
//! deadline"; the listeners are the request groups' half of it that the clock holds). The
//! golden-bytes tests `the_vsync_snapshot_layout_is_pinned` and
//! `a_stopped_clock_s_snapshot_layout_is_pinned` pin it; changing it means bumping
//! [`SNAPSHOT_VERSION`].
//!
//! A snapshot is a file and is not trusted. [`Shared::restore_vsync`] checks, in this order, and
//! returns the first failure: the magic, the version, that the body decodes, that no bytes follow
//! it, then the clock's own relations ([`VsyncSnapshotError`], in the order of its variants):
//! at most [`MAX_LISTENERS`] listeners, the listeners strictly ascending (so none twice), a
//! deadline only with a listener, and a tick count and fed-back ticks whose sum fits a `u64`.
//! Only then is the clock replaced. [`Shared::save_vsync`] runs the same relation checks on what
//! it is about to write.
//!
//! Not checked here: that each listener is a live event a gsv server made, which the kernel's
//! object-table snapshot would show (its consequence is bounded: a listener the kernel does not
//! hold is refused at the first tick and dropped, above); that the deadline is near the restored
//! run's clock (a deadline in the past is met by the first turn, which ticks once and does not
//! bank, and one far in the future delays the first tick to it); and that listeners and no
//! deadline is the state the end of the counts leaves (it is also what a snapshot of a clock
//! stopped there holds, so it is accepted).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use starquake_kernel::id::ObjectId;
use starquake_trace::{HostNanos, ServiceId};
use thiserror::Error;

use super::{Locked, Shared, TimerHost};
use crate::note::Note;

#[cfg(test)]
mod tests;

/// The vsync interval in host nanoseconds: a sixtieth of a second, as NX2 computes it
/// (`VblankInterval{1'000'000'000LL / hz}` with `hz` 60; module docs, "The interval: conflict").
pub const INTERVAL_NS: u64 = 1_000_000_000 / 60;

/// The most listeners the clock holds at once (module docs: starquake's own bound, a guess).
pub const MAX_LISTENERS: usize = 64;

/// First bytes of every vsync clock snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SQVS";

/// Layout version written into every vsync clock snapshot and required when reading one.
pub const SNAPSHOT_VERSION: u32 = 1;

/// The vsync clock's state (module docs): the ticks delivered, the next deadline, the ticks fed
/// back for replay, and the listeners. Built fresh only by gsv's shared state, and changed only
/// through [`Locked`] and [`Shared`], which keep a deadline only with a listener.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vsync {
    /// The ticks delivered so far; the last one's number.
    tick: u64,
    /// The next deadline, while the clock runs.
    next: Option<HostNanos>,
    /// The ticks fed back for replay and not yet delivered.
    replayed: u64,
    /// The events each tick signals.
    listeners: BTreeSet<ObjectId>,
}

impl Vsync {
    /// A clock with no listener, stopped, that has delivered no tick.
    pub(super) const fn new() -> Self {
        Self {
            tick: 0,
            next: None,
            replayed: 0,
            listeners: BTreeSet::new(),
        }
    }

    /// The ticks delivered so far, which is the last one's number (0 before the first).
    #[must_use]
    pub const fn tick(&self) -> u64 {
        self.tick
    }

    /// The next deadline, while the clock runs.
    #[must_use]
    pub const fn deadline(&self) -> Option<HostNanos> {
        self.next
    }

    /// The ticks fed back for replay ([`Shared::replay_vsync`]) and not yet delivered.
    #[must_use]
    pub const fn replayed(&self) -> u64 {
        self.replayed
    }

    /// The listeners, in ascending order.
    pub fn listeners(&self) -> impl Iterator<Item = ObjectId> + '_ {
        self.listeners.iter().copied()
    }

    /// Whether a live turn at `now` would tick: the clock runs and `now` has reached its
    /// deadline.
    #[must_use]
    pub fn is_due(&self, now: HostNanos) -> bool {
        self.next.is_some_and(|next| now >= next)
    }

    /// Takes `event` out of the listeners, stopping the clock if it was the last; whether it
    /// listened.
    fn remove(&mut self, event: ObjectId) -> bool {
        let removed = self.listeners.remove(&event);
        if self.listeners.is_empty() {
            self.next = None;
        }
        removed
    }

    /// The live turn's decision at `now` (module docs, "The next deadline: no banking"): whether
    /// the clock ticks, and if it does, its next deadline set as NX2 `PaceToVblank` sets
    /// `m_next_vblank`.
    fn advance(&mut self, now: HostNanos) -> bool {
        let Some(next) = self.next else {
            return false;
        };
        if now < next {
            return false;
        }
        // NX2: `if (now > m_next_vblank + interval) m_next_vblank = now;`. A deadline whose edge
        // does not fit a `u64` cannot be passed by more than an interval.
        let late = next
            .0
            .checked_add(INTERVAL_NS)
            .is_some_and(|edge| now.0 > edge);
        let from = if late { now } else { next };
        // NX2: `m_next_vblank += interval;`.
        self.next = from.0.checked_add(INTERVAL_NS).map(HostNanos);
        if self.next.is_none() {
            tracing::error!(
                deadline = from.0,
                "gsv: the vsync clock's next deadline is past the end of time; it stops"
            );
        }
        true
    }
}

/// Why a listener could not be added ([`Locked::listen_vsync`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum VsyncError {
    /// The clock already holds [`MAX_LISTENERS`] listeners.
    #[error("the vsync clock already has the most listeners it takes ({MAX_LISTENERS})")]
    TooManyListeners,
}

/// Why a recorded tick could not be fed back ([`Shared::replay_vsync`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum VsyncReplayError {
    /// The tick is not the next one: the replay has diverged from the run.
    #[error("vsync tick {found} was fed back where tick {expected} comes next")]
    OutOfOrder {
        /// The number the next tick has.
        expected: u64,
        /// The number fed back.
        found: u64,
    },
    /// The ticks delivered and fed back already reach `u64::MAX`: no tick has a number after
    /// them.
    #[error("no vsync tick comes after tick {}", u64::MAX)]
    CountExhausted,
}

/// Why a vsync clock snapshot could not be written or read (module docs, "What is saved").
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum VsyncSnapshotError {
    /// Encoding failed.
    #[error("could not encode the vsync clock snapshot: {0}")]
    Encode(postcard::Error),
    /// The header or the body is not a well-formed encoding.
    #[error("could not decode the vsync clock snapshot: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`SNAPSHOT_MAGIC`].
    #[error("not a vsync clock snapshot (bad magic)")]
    BadMagic,
    /// The snapshot was written with another layout version.
    #[error(
        "vsync clock snapshot layout version {found} is not supported (this build reads {SNAPSHOT_VERSION})"
    )]
    UnsupportedVersion {
        /// The version found.
        found: u32,
    },
    /// Bytes follow the body.
    #[error("{count} unexpected bytes after the vsync clock snapshot")]
    TrailingBytes {
        /// How many.
        count: usize,
    },
    /// More listeners than [`MAX_LISTENERS`].
    #[error("the vsync clock snapshot holds {count} listeners, more than {MAX_LISTENERS}")]
    TooManyListeners {
        /// How many it holds.
        count: usize,
    },
    /// A listener that is not above the one before it: listed twice, or out of order.
    #[error(
        "the vsync clock snapshot's listener {index} is not above the one before it (listed twice or out of order)"
    )]
    ListenersNotAscending {
        /// Its index in the list.
        index: usize,
    },
    /// A deadline with no listener: a clock runs only while something listens.
    #[error("the vsync clock snapshot has a deadline but no listeners")]
    DeadlineWithoutListeners,
    /// The tick count and the ticks fed back add up past `u64::MAX`.
    #[error(
        "the vsync clock snapshot's {tick} ticks and {replayed} ticks fed back add up past the largest tick number"
    )]
    TicksOverflow {
        /// The ticks delivered.
        tick: u64,
        /// The ticks fed back.
        replayed: u64,
    },
}

/// The snapshot header.
#[derive(Serialize, Deserialize)]
struct Header {
    /// [`SNAPSHOT_MAGIC`].
    magic: [u8; 4],
    /// [`SNAPSHOT_VERSION`].
    version: u32,
}

/// The snapshot body (module docs, "What is saved"): the listeners as a list, so a restore sees
/// a duplicate or a misordered one rather than a set that drops it.
#[derive(Serialize, Deserialize)]
struct Saved {
    tick: u64,
    next: Option<HostNanos>,
    replayed: u64,
    listeners: Vec<ObjectId>,
}

impl Saved {
    /// The relation checks [`Shared::save_vsync`] and [`Shared::restore_vsync`] share, in the
    /// documented order; the clock the snapshot holds.
    fn check(self) -> Result<Vsync, VsyncSnapshotError> {
        let count = self.listeners.len();
        if count > MAX_LISTENERS {
            return Err(VsyncSnapshotError::TooManyListeners { count });
        }
        if let Some(index) = self
            .listeners
            .windows(2)
            .position(|pair| pair[0] >= pair[1])
        {
            return Err(VsyncSnapshotError::ListenersNotAscending { index: index + 1 });
        }
        if self.next.is_some() && self.listeners.is_empty() {
            return Err(VsyncSnapshotError::DeadlineWithoutListeners);
        }
        if self.tick.checked_add(self.replayed).is_none() {
            return Err(VsyncSnapshotError::TicksOverflow {
                tick: self.tick,
                replayed: self.replayed,
            });
        }
        Ok(Vsync {
            tick: self.tick,
            next: self.next,
            replayed: self.replayed,
            listeners: self.listeners.into_iter().collect(),
        })
    }
}

impl super::GsvShared {
    /// The vsync clock.
    #[must_use]
    pub const fn vsync(&self) -> &Vsync {
        &self.vsync
    }
}

impl Locked<'_> {
    /// Adds `event` to the clock's listeners, starting the clock one [`INTERVAL_NS`] after `now`
    /// if it had none (module docs, "The clock"); whether `event` was not listening already. It
    /// checks, in this order: an event already listening is accepted and changes nothing, then
    /// a clock at [`MAX_LISTENERS`] refuses a new one ([`VsyncError::TooManyListeners`]),
    /// changing nothing.
    pub fn listen_vsync(&mut self, event: ObjectId, now: HostNanos) -> Result<bool, VsyncError> {
        let vsync = &mut self.state.vsync;
        if vsync.listeners.contains(&event) {
            return Ok(false);
        }
        if vsync.listeners.len() >= MAX_LISTENERS {
            return Err(VsyncError::TooManyListeners);
        }
        if vsync.listeners.is_empty() {
            vsync.next = now.0.checked_add(INTERVAL_NS).map(HostNanos);
        }
        vsync.listeners.insert(event);
        Ok(true)
    }

    /// Takes `event` out of the clock's listeners, stopping the clock if it was the last (module
    /// docs, "The clock"); whether it was listening.
    pub fn unlisten_vsync(&mut self, event: ObjectId) -> bool {
        self.state.vsync.remove(event)
    }

    /// Delivers the next tick through `host`, recording for `service` (module docs, "Tick"):
    /// counts it, records it, signals every listener in ascending order, then drops each one the
    /// kernel refused. A count at `u64::MAX` delivers none and stops the clock (module docs, "The
    /// end of the counts").
    fn deliver_tick(&mut self, host: &mut dyn TimerHost, service: ServiceId) {
        let vsync = &mut self.state.vsync;
        let Some(tick) = vsync.tick.checked_add(1) else {
            tracing::error!("gsv: the vsync clock has counted every tick a u64 holds; it stops");
            vsync.next = None;
            return;
        };
        vsync.tick = tick;
        host.record(Note::VsyncTick { tick }.event(service));
        let refused: Vec<ObjectId> = vsync
            .listeners
            .iter()
            .copied()
            .filter(|&event| match host.signal_event(event) {
                Ok(()) => false,
                Err(error) => {
                    tracing::error!(
                        %error,
                        ?event,
                        "gsv: the kernel refused a vsync listener's signal; it no longer listens"
                    );
                    true
                }
            })
            .collect();
        for event in refused {
            vsync.remove(event);
        }
    }
}

impl Shared {
    /// The timer thread's live vsync turn at `now`: call it after each [`Shared::drain`], with
    /// the time of that turn (module docs, "The clock"). It takes gsv's shared lock and ticks at
    /// most once, signalling the listeners and recording through `host`.
    pub fn tick_vsync(&self, host: &mut dyn TimerHost, now: HostNanos) {
        let mut state = self.lock();
        if state.state.vsync.advance(now) {
            state.deliver_tick(host, self.service);
        }
    }

    /// The clock's next deadline, while it runs: the latest the timer thread may run its next
    /// live vsync turn ([`Shared::tick_vsync`]).
    #[must_use]
    pub fn vsync_deadline(&self) -> Option<HostNanos> {
        self.lock().state.vsync.next
    }

    /// Feeds the recorded tick numbered `tick` back for the next replay turn (module docs,
    /// "Replay"). It must be the next number: the ticks delivered and fed back so far, plus one.
    /// On an error nothing changes.
    pub fn replay_vsync(&self, tick: u64) -> Result<(), VsyncReplayError> {
        let mut state = self.lock();
        let vsync = &mut state.state.vsync;
        let expected = vsync
            .tick
            .checked_add(vsync.replayed)
            .and_then(|fed| fed.checked_add(1))
            .ok_or(VsyncReplayError::CountExhausted)?;
        if tick != expected {
            return Err(VsyncReplayError::OutOfOrder {
                expected,
                found: tick,
            });
        }
        vsync.replayed = vsync.replayed.saturating_add(1);
        Ok(())
    }

    /// The timer thread's replay vsync turn: delivers every tick fed back since the last one, in
    /// order, as a live tick is delivered (module docs, "Replay"). It reads no clock. Each
    /// delivered tick has a number: [`Shared::replay_vsync`] and a restore
    /// ([`VsyncSnapshotError::TicksOverflow`]) both keep the count and the ticks fed back within
    /// a `u64`.
    pub fn tick_replayed_vsync(&self, host: &mut dyn TimerHost) {
        let mut state = self.lock();
        while state.state.vsync.replayed > 0 {
            state.state.vsync.replayed -= 1;
            state.deliver_tick(host, self.service);
        }
    }

    /// Writes the vsync clock (module docs, "What is saved").
    pub fn save_vsync(&self) -> Result<Vec<u8>, VsyncSnapshotError> {
        let vsync = self.lock().state.vsync.clone();
        let saved = Saved {
            tick: vsync.tick,
            next: vsync.next,
            replayed: vsync.replayed,
            listeners: vsync.listeners.into_iter().collect(),
        };
        let mut bytes = postcard::to_allocvec(&Header {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
        })
        .map_err(VsyncSnapshotError::Encode)?;
        bytes.extend(postcard::to_allocvec(&saved).map_err(VsyncSnapshotError::Encode)?);
        saved.check()?;
        Ok(bytes)
    }

    /// Replaces the vsync clock with what [`Shared::save_vsync`] wrote, after the checks of the
    /// module docs ("What is saved"). On an error the clock is as it was.
    pub fn restore_vsync(&self, bytes: &[u8]) -> Result<(), VsyncSnapshotError> {
        let (header, rest) =
            postcard::take_from_bytes::<Header>(bytes).map_err(VsyncSnapshotError::Decode)?;
        if header.magic != SNAPSHOT_MAGIC {
            return Err(VsyncSnapshotError::BadMagic);
        }
        if header.version != SNAPSHOT_VERSION {
            return Err(VsyncSnapshotError::UnsupportedVersion {
                found: header.version,
            });
        }
        let (saved, rest) =
            postcard::take_from_bytes::<Saved>(rest).map_err(VsyncSnapshotError::Decode)?;
        if !rest.is_empty() {
            return Err(VsyncSnapshotError::TrailingBytes { count: rest.len() });
        }
        let vsync = saved.check()?;
        self.lock().state.vsync = vsync;
        Ok(())
    }
}
