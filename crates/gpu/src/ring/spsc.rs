//! The bounded single-producer, single-consumer ring between the front and back threads
//! (gpu-protocol spec, "The ring", "Back-pressure").
//!
//! Structure from `eden-nx2/src/video_core/gpu_split.{h,cpp}` @ 50fa86bb (GPL-3.0-or-later): a
//! power-of-two array of 64-bit words, a write position the producer *publishes* in batches,
//! and a consumed position the consumer hands back; the consumer spins briefly and then sleeps
//! when the ring is empty, and the producer waits when it is full. The differences, none of
//! which changes what the consumer receives or in what order:
//!
//! - The words are atomics written and read with relaxed ordering, published by a sequentially
//!   consistent store of the write position that the consumer loads before reading them, so the
//!   ring needs no `unsafe`.
//! - An entry may wrap around the end of the array (each word's index is taken modulo the
//!   capacity), so there is no wrap marker; Eden-NX2 writes a `Wrap` entry and skips to the
//!   start.
//! - There is no `Drain` (the front never waits for the back to empty the ring; the spec's
//!   "Semaphore acquires"), so the batch size is fixed at [`PUBLISH_EVERY`]; Eden-NX2 drops it to
//!   8 words after a drain and doubles it back.
//! - A full producer spins, then sleeps until the consumer frees space; Eden-NX2's yields in a
//!   loop.
//! - The consumer hands space back after every entry: it stores the consumed position
//!   (sequentially consistent) and checks whether the producer is waiting each time. Eden-NX2's
//!   back stores it every 4,096 words within a batch and at the batch's end
//!   (`read_position - last_store >= 4096` in `GpuSplit`'s back loop), and its front only polls
//!   it. Handing space back sooner lets a full producer go on sooner; it costs one store and one
//!   load per entry.
//! - Both sides spin [`SPINS_BEFORE_SLEEP`] (4,096) times before they sleep. Eden-NX2's front
//!   pauses 4,096 times in `WaitForSpace` before it yields, but its back spins 8,192 times
//!   (`spins < 8192`) before it sleeps, so an idle consumer here goes to sleep after half the
//!   spins.
//! - A sleeping side looks again after at most 100 ms even if no wake-up came; Eden-NX2's back
//!   sleeps on its wake counter with no time limit. The bound is only a safety net: the tests
//!   make it far longer than their patience, so a lost wake-up fails them.
//!
//! The ring delivers entries in exactly the order they were pushed, which is ordering guarantees
//! 1 and 2 of the spec (within a channel in method order; across channels in the order the front
//! emitted them). The other guarantees (releases after host completion, markers, presents, the
//! order of `GpuFront` calls) are kept by the front and back on top of it (slices G14 and G15).
//!
//! # Back-pressure
//!
//! [`Producer::push`] waits for space when the ring is full, and only the front, which owns the
//! producer, ever waits there. Before it waits it publishes what it has written, so the consumer
//! always has something to drain. [`Producer::try_push`] reports [`PushError::Full`] instead.
//!
//! # Guesses
//!
//! The spin counts ([`SPINS_BEFORE_SLEEP`]) are Eden-NX2's order of magnitude, not measured here.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use thiserror::Error;

use super::encode::{DecodeError, EncodeError, decode, encode, entry_words};
use super::entry::RingEntry;

/// The ring's size in words: 1 Mi words, 8 MiB (Eden-NX2's `RingWords`; a **guess** that it is
/// enough, to be measured as peak occupancy).
pub const RING_WORDS: usize = 1 << 20;

/// The producer publishes on its own once this many words are unpublished (Eden-NX2's
/// `PublishEvery`).
pub const PUBLISH_EVERY: u64 = 256;

/// Times either side polls before it sleeps. Eden-NX2's front (`WaitForSpace`) pauses 4,096
/// times before yielding; its back spins 8,192 times before sleeping, twice this (a listed
/// difference in the module docs).
pub const SPINS_BEFORE_SLEEP: u32 = 4096;

/// The longest one sleep of either side lasts before it looks again (it is woken sooner by a
/// publish or a pop; this only bounds a sleep against a lost wake-up).
#[cfg(not(test))]
const LONGEST_SLEEP: Duration = Duration::from_millis(100);

/// Under test, far longer than any test waits, so that a lost wake-up fails a bounded test
/// instead of being rescued by the next look.
#[cfg(test)]
const LONGEST_SLEEP: Duration = Duration::from_secs(3600);

/// A capacity the ring cannot have.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("a ring of {words} words is not a power of two")]
pub struct CapacityError {
    /// The capacity asked for.
    pub words: usize,
}

/// Why an entry was not pushed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PushError {
    /// The ring has no room for the entry now ([`Producer::try_push`] only).
    #[error("the ring has {free} free words and the entry needs {needed}")]
    Full {
        /// Words free.
        free: usize,
        /// Words the entry takes.
        needed: usize,
    },
    /// The entry is longer than the whole ring.
    #[error("the entry takes {needed} words and the ring holds {capacity}")]
    TooLarge {
        /// Words the entry takes.
        needed: usize,
        /// The ring's capacity.
        capacity: usize,
    },
    /// The entry has no encoding.
    #[error(transparent)]
    Encode(#[from] EncodeError),
    /// The consumer is gone, so nothing would read the entry.
    #[error("the ring's consumer is gone")]
    Disconnected,
}

/// Why no entry was popped.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PopError {
    /// The ring is empty and its producer is gone.
    #[error("the ring is empty and its producer is gone")]
    Disconnected,
    /// The words at the read position are not an entry. Nothing past them is consumed, so every
    /// later pop returns this again (pinned by `a_corrupt_entry_stops_the_ring_for_good` and
    /// `an_entry_longer_than_what_was_published_is_corrupt_for_good`).
    #[error("the ring is corrupt: {0}")]
    Corrupt(DecodeError),
}

/// What the two sides share.
struct Shared {
    words: Box<[AtomicU64]>,
    /// `capacity - 1`.
    mask: u64,
    /// Words the producer has made visible (a running count; the index is it modulo capacity).
    published: AtomicU64,
    /// Words the consumer has finished with.
    consumed: AtomicU64,
    producer_gone: AtomicBool,
    consumer_gone: AtomicBool,
    /// The consumer is about to sleep, or sleeping, on `data`.
    consumer_waiting: AtomicBool,
    /// The producer is about to sleep, or sleeping, on `space`.
    producer_waiting: AtomicBool,
    lock: Mutex<()>,
    data: Condvar,
    space: Condvar,
}

impl Shared {
    fn capacity(&self) -> u64 {
        self.mask + 1
    }

    fn lock(&self) -> MutexGuard<'_, ()> {
        // The mutex guards no data, so a poisoned one is as good as any.
        self.lock.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn slot(&self, position: u64) -> Option<&AtomicU64> {
        self.words.get(usize::try_from(position & self.mask).ok()?)
    }

    /// Wakes the consumer if it is waiting for data.
    fn wake_consumer(&self) {
        if self.consumer_waiting.load(Ordering::SeqCst) {
            let _guard = self.lock();
            self.data.notify_one();
        }
    }

    /// Wakes the producer if it is waiting for space.
    fn wake_producer(&self) {
        if self.producer_waiting.load(Ordering::SeqCst) {
            let _guard = self.lock();
            self.space.notify_one();
        }
    }
}

/// Creates a ring of `capacity` words (a power of two; [`RING_WORDS`] in production).
pub fn ring(capacity: usize) -> Result<(Producer, Consumer), CapacityError> {
    let mask = capacity
        .is_power_of_two()
        .then(|| u64::try_from(capacity - 1).ok())
        .flatten()
        .ok_or(CapacityError { words: capacity })?;
    let shared = Arc::new(Shared {
        words: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
        mask,
        published: AtomicU64::new(0),
        consumed: AtomicU64::new(0),
        producer_gone: AtomicBool::new(false),
        consumer_gone: AtomicBool::new(false),
        consumer_waiting: AtomicBool::new(false),
        producer_waiting: AtomicBool::new(false),
        lock: Mutex::new(()),
        data: Condvar::new(),
        space: Condvar::new(),
    });
    let producer = Producer {
        shared: Arc::clone(&shared),
        written: 0,
        published: 0,
        consumed: 0,
        scratch: Vec::new(),
    };
    let consumer = Consumer {
        shared,
        read: 0,
        available: 0,
        scratch: Vec::new(),
        #[cfg(test)]
        between_looks: None,
    };
    Ok((producer, consumer))
}

/// The front's end of the ring.
pub struct Producer {
    shared: Arc<Shared>,
    /// Words written so far.
    written: u64,
    /// Words published so far.
    published: u64,
    /// The consumed position as last read.
    consumed: u64,
    scratch: Vec<u64>,
}

impl Producer {
    /// The ring's capacity in words.
    pub fn capacity(&self) -> usize {
        self.shared.words.len()
    }

    /// Words written but not yet published.
    pub fn unpublished(&self) -> u64 {
        self.written - self.published
    }

    /// Encodes `entry` into `scratch` and checks it can ever fit.
    fn prepare(&mut self, entry: &RingEntry<'_>) -> Result<u64, PushError> {
        if self.shared.consumer_gone.load(Ordering::SeqCst) {
            return Err(PushError::Disconnected);
        }
        self.scratch.clear();
        encode(entry, &mut self.scratch)?;
        let needed = self.scratch.len();
        u64::try_from(needed)
            .ok()
            .filter(|words| *words <= self.shared.capacity())
            .ok_or(PushError::TooLarge {
                needed,
                capacity: self.capacity(),
            })
    }

    /// Whether `needed` words fit now, after reading the consumed position again.
    fn fits(&mut self, needed: u64) -> bool {
        if self.written + needed - self.consumed <= self.shared.capacity() {
            return true;
        }
        self.consumed = self.shared.consumed.load(Ordering::SeqCst);
        self.written + needed - self.consumed <= self.shared.capacity()
    }

    /// Copies the encoded entry in and publishes if [`PUBLISH_EVERY`] words are waiting.
    fn commit(&mut self) {
        for (offset, &word) in (0..).zip(&self.scratch) {
            if let Some(slot) = self.shared.slot(self.written + offset) {
                slot.store(word, Ordering::Relaxed);
            }
        }
        self.written += u64::try_from(self.scratch.len()).unwrap_or_default();
        if self.unpublished() >= PUBLISH_EVERY {
            self.publish();
        }
    }

    /// Writes `entry` if it fits now.
    pub fn try_push(&mut self, entry: &RingEntry<'_>) -> Result<(), PushError> {
        let needed = self.prepare(entry)?;
        if !self.fits(needed) {
            return Err(PushError::Full {
                free: usize::try_from(self.shared.capacity() - (self.written - self.consumed))
                    .unwrap_or_default(),
                needed: self.scratch.len(),
            });
        }
        self.commit();
        Ok(())
    }

    /// Writes `entry`, waiting for space while the ring is full.
    ///
    /// # Errors
    ///
    /// [`PushError::TooLarge`] and [`PushError::Encode`] at once; [`PushError::Disconnected`]
    /// if the consumer is gone, including while waiting. Never [`PushError::Full`].
    pub fn push(&mut self, entry: &RingEntry<'_>) -> Result<(), PushError> {
        let needed = self.prepare(entry)?;
        let mut spins = 0;
        while !self.fits(needed) {
            if self.shared.consumer_gone.load(Ordering::SeqCst) {
                return Err(PushError::Disconnected);
            }
            if spins < SPINS_BEFORE_SLEEP {
                if spins == 0 {
                    // The consumer may be waiting for what we have not published.
                    self.publish();
                }
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let shared = Arc::clone(&self.shared);
            let guard = shared.lock();
            shared.producer_waiting.store(true, Ordering::SeqCst);
            if !self.fits(needed) && !shared.consumer_gone.load(Ordering::SeqCst) {
                let _guard = shared
                    .space
                    .wait_timeout(guard, LONGEST_SLEEP)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            shared.producer_waiting.store(false, Ordering::SeqCst);
        }
        self.commit();
        Ok(())
    }

    /// Makes every entry written so far visible to the consumer.
    pub fn publish(&mut self) {
        if self.published == self.written {
            return;
        }
        self.published = self.written;
        self.shared.published.store(self.written, Ordering::SeqCst);
        self.shared.wake_consumer();
    }

    /// Writes raw words, for tests of a corrupt ring.
    #[cfg(test)]
    fn push_raw(&mut self, words: &[u64]) {
        self.scratch.clear();
        self.scratch.extend_from_slice(words);
        self.commit();
    }
}

impl Drop for Producer {
    fn drop(&mut self) {
        self.publish();
        self.shared.producer_gone.store(true, Ordering::SeqCst);
        let _guard = self.shared.lock();
        self.shared.data.notify_all();
    }
}

/// The back's end of the ring.
pub struct Consumer {
    shared: Arc<Shared>,
    /// Words read so far.
    read: u64,
    /// The published position as last read.
    available: u64,
    scratch: Vec<u64>,
    /// Runs in [`Consumer::try_pop`] between its first look for an entry and its look at whether
    /// the producer is gone, so a test can drop the producer exactly there.
    #[cfg(test)]
    between_looks: Option<Box<dyn FnMut() + Send>>,
}

impl Consumer {
    /// Whether an entry is visible, reading the published position again if needed.
    fn has_entry(&mut self) -> bool {
        if self.available == self.read {
            self.available = self.shared.published.load(Ordering::SeqCst);
        }
        self.available != self.read
    }

    /// The entry at the read position, which must be visible.
    ///
    /// A bad entry is not consumed and the published position is not read again while one is
    /// visible, so a failure repeats exactly on every later call: the words it reads cannot
    /// change, since the producer never overwrites words that are not consumed.
    fn take(&mut self) -> Result<RingEntry<'static>, PopError> {
        self.decode_next().map_err(PopError::Corrupt)
    }

    fn decode_next(&mut self) -> Result<RingEntry<'static>, DecodeError> {
        let word = |position: u64| {
            self.shared
                .slot(position)
                .map_or(0, |slot| slot.load(Ordering::Relaxed))
        };
        let (kind, len) = entry_words(word(self.read))?;
        let available = self.available - self.read;
        let needed = u64::try_from(len).unwrap_or(u64::MAX);
        if needed > available {
            return Err(DecodeError::Truncated {
                kind,
                needed: len,
                available: usize::try_from(available).unwrap_or(usize::MAX),
            });
        }
        let start = self.read;
        self.scratch.clear();
        self.scratch.extend((start..start + needed).map(word));
        let (entry, _) = decode(&self.scratch)?;
        self.read += needed;
        self.shared.consumed.store(self.read, Ordering::SeqCst);
        self.shared.wake_producer();
        Ok(entry)
    }

    /// The next entry if one is visible: `Ok(None)` when the ring is empty and the producer is
    /// still there.
    pub fn try_pop(&mut self) -> Result<Option<RingEntry<'static>>, PopError> {
        if self.has_entry() {
            return self.take().map(Some);
        }
        #[cfg(test)]
        if let Some(hook) = self.between_looks.as_mut() {
            hook();
        }
        if self.shared.producer_gone.load(Ordering::SeqCst) {
            // The producer publishes everything before it goes: look once more.
            if self.has_entry() {
                return self.take().map(Some);
            }
            return Err(PopError::Disconnected);
        }
        Ok(None)
    }

    /// The next entry, waiting while the ring is empty.
    pub fn pop(&mut self) -> Result<RingEntry<'static>, PopError> {
        loop {
            if let Some(entry) = self.pop_timeout(Duration::MAX)? {
                return Ok(entry);
            }
        }
    }

    /// The next entry, waiting at most `timeout` while the ring is empty: `Ok(None)` if none
    /// arrived in time.
    pub fn pop_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<RingEntry<'static>>, PopError> {
        let deadline = Instant::now().checked_add(timeout);
        let mut spins = 0;
        loop {
            if let Some(entry) = self.try_pop()? {
                return Ok(Some(entry));
            }
            if spins < SPINS_BEFORE_SLEEP {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let now = Instant::now();
            let wait = match deadline {
                Some(deadline) if deadline <= now => return Ok(None),
                Some(deadline) => deadline.saturating_duration_since(now),
                None => Duration::MAX,
            };
            let shared = Arc::clone(&self.shared);
            let guard = shared.lock();
            shared.consumer_waiting.store(true, Ordering::SeqCst);
            if !self.has_entry() && !shared.producer_gone.load(Ordering::SeqCst) {
                let _guard = shared
                    .data
                    .wait_timeout(guard, wait.min(LONGEST_SLEEP))
                    .unwrap_or_else(PoisonError::into_inner);
            }
            shared.consumer_waiting.store(false, Ordering::SeqCst);
        }
    }
}

impl Drop for Consumer {
    fn drop(&mut self) {
        self.shared.consumer_gone.store(true, Ordering::SeqCst);
        let _guard = self.shared.lock();
        self.shared.space.notify_all();
    }
}

#[cfg(test)]
mod tests;
