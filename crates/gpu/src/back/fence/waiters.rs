//! The acquire waiter table: the channels parked on a semaphore acquire, which releases latch
//! (gpu-protocol spec, "Semaphore acquires", step 1; "Threads and data flow": "the acquire
//! waiter table (a small mutex-guarded list)"; row 70 of `eden-nx2/NX2_NOTES.md`).
//!
//! The spec lists the table under slice G14, whose front registers and removes waiters; slice
//! G15's completion path latches them, so the type is defined here for G14 to use, and the front
//! and the back share one table behind an `Arc`.
//!
//! Ported from `Nx2SemaphoreWaiters` in `eden-nx2/src/video_core/nx2.h` @ 50fa86bb (Copyright
//! 2026 Eden Emulator Project, GPL-3.0-or-later): one waiter per channel, a register that
//! replaces the channel's earlier waiter, a remove, a latched query, and a write that latches
//! every unlatched waiter on its address whose condition the written value meets. A latched
//! waiter stays latched, whatever is written later, until it is removed or replaced, so an
//! acquire-equal whose value a later release moves past still resumes (row 70: "releases latch
//! the waiters they meet").
//!
//! The table holds at most one waiter per channel ([`WaiterTable::register`] replaces), so its
//! size is bounded by the channels gsv has created; the front removes a channel's waiter when the
//! channel resumes or is destroyed (G14).
//!
//! # Divergences from Eden-NX2
//!
//! - **Every acquire condition latches.** Eden-NX2's `Matches` knows the three `SEMAPHORED`
//!   acquires (equal, greater-or-equal, and) and never latches any other operation; here a
//!   waiter carries an [`AcquireOp`], all five of them latch by [`AcquireOp::is_met`], because
//!   the spec suspends on `SEM_EXECUTE` acquires too ("Semaphore acquires"). Pinned by
//!   `tests::waiters::every_acquire_condition_latches`.
//! - **Keyed by GPU address, not device address.** Eden-NX2 keys a waiter by the CPU address its
//!   semaphore translates to, so channels of different address spaces that share memory meet.
//!   Neither a release nor a channel names its address space yet (the spec's `AsId` has no type
//!   until G14 and G21), so a waiter here is keyed by the GPU address the acquire read. Two
//!   address spaces that map one semaphore at different GPU addresses do not meet, and two that
//!   use one GPU address for different memory latch each other's waiters. **Open** until the
//!   address-space id exists; pinned by `tests::waiters::waiters_meet_by_gpu_address_alone`.
//! - **[`WaiterTable::has_waiter`]** is new: the back closes its open command list when a fenced
//!   release's address has an unlatched waiter (the spec's close trigger; Eden-NX2 has none).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::interface::ChannelId;
use crate::sema::AcquireOp;

/// A channel parked on an acquire: the semaphore it reads and the condition it waits for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Waiter {
    /// The parked channel.
    pub channel: ChannelId,
    /// The GPU address of the 32-bit semaphore word the acquire reads.
    pub address: u64,
    /// The condition.
    pub op: AcquireOp,
    /// The acquire's payload.
    pub payload: u32,
}

#[derive(Debug)]
struct Entry {
    waiter: Waiter,
    latched: bool,
}

/// The channels parked on an acquire, shared by the front (which registers and removes them)
/// and the back (which latches them when it performs a release). Every method takes `&self`; the
/// list is behind a mutex held only for the method's own walk, never across a call out.
#[derive(Debug, Default)]
pub struct WaiterTable {
    entries: Mutex<Vec<Entry>>,
    /// How many entries there are, so a write with no waiters costs one load (as Eden-NX2's
    /// `count`).
    count: AtomicUsize,
}

impl WaiterTable {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, Vec<Entry>> {
        // Every method leaves the list whole before it can panic, so a poisoned one is usable.
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Registers `waiter`, unlatched, replacing any earlier waiter of its channel.
    pub fn register(&self, waiter: Waiter) {
        let mut entries = self.lock();
        entries.retain(|e| e.waiter.channel != waiter.channel);
        entries.push(Entry {
            waiter,
            latched: false,
        });
        self.count.store(entries.len(), Ordering::SeqCst);
    }

    /// Removes `channel`'s waiter; whether it had one.
    pub fn remove(&self, channel: ChannelId) -> bool {
        if self.count.load(Ordering::SeqCst) == 0 {
            return false;
        }
        let mut entries = self.lock();
        let before = entries.len();
        entries.retain(|e| e.waiter.channel != channel);
        self.count.store(entries.len(), Ordering::SeqCst);
        entries.len() != before
    }

    /// Whether `channel` has a waiter that a release has latched.
    pub fn is_latched(&self, channel: ChannelId) -> bool {
        if self.count.load(Ordering::SeqCst) == 0 {
            return false;
        }
        self.lock()
            .iter()
            .any(|e| e.waiter.channel == channel && e.latched)
    }

    /// Whether an unlatched waiter reads the semaphore at `address`: a channel is parked on it
    /// and a release there would resume it.
    pub fn has_waiter(&self, address: u64) -> bool {
        if self.count.load(Ordering::SeqCst) == 0 {
            return false;
        }
        self.lock()
            .iter()
            .any(|e| e.waiter.address == address && !e.latched)
    }

    /// A GPU write left `value` in the 32-bit semaphore word at `address`: latches every
    /// unlatched waiter on that address whose condition `value` meets, and returns whether it
    /// latched any.
    pub fn on_write(&self, address: u64, value: u32) -> bool {
        if self.count.load(Ordering::SeqCst) == 0 {
            return false;
        }
        let mut latched = false;
        for entry in self.lock().iter_mut() {
            let w = entry.waiter;
            if w.address == address && !entry.latched && w.op.is_met(value, w.payload) {
                entry.latched = true;
                latched = true;
            }
        }
        latched
    }

    /// How many waiters are registered.
    pub fn len(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }

    /// Whether no waiter is registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
