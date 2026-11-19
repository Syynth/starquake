//! What the capture builder and the replay both use to run the real front and back on one thread
//! of their own: the [`Collector`] thread that empties the ring as the front fills it, the hash
//! of a ring entry, and the queue of channels the back forgets.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
#[cfg(any(test, feature = "test-support"))]
use std::time::{Duration, Instant};

use super::format::ContentHash;
use crate::front::ChannelRetired;
use crate::interface::{ChannelId, Completions, GpuCompletion};
use crate::ring::{Consumer, PopError, RingEntry, encode, encoded_words};

/// The most ring words a collector keeps at once, summed over its entries
/// ([`crate::ring::encoded_words`]): the bound on what a front run on a capture can make the
/// replay hold. 2^24 words (128 MiB), a **guess** at a size no synthetic capture reaches.
pub const MAX_COLLECTED_WORDS: usize = 1 << 24;

/// The hash of a ring entry: its encoding's words, little-endian. `None` when the entry does
/// not encode (an entry no front pushes: a run past the ring's limit).
pub fn entry_hash(entry: &RingEntry<'_>) -> Option<ContentHash> {
    let mut words = Vec::with_capacity(encoded_words(entry));
    encode(entry, &mut words).ok()?;
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    Some(ContentHash::of(&bytes))
}

#[derive(Default)]
struct Collected {
    entries: VecDeque<RingEntry<'static>>,
    /// Entries popped so far, kept or not.
    taken: u64,
    /// Words of the entries kept.
    words: usize,
    /// Whether an entry was dropped past [`MAX_COLLECTED_WORDS`].
    overflowed: bool,
    /// Whether the ring ended (its producer was dropped), or failed.
    ended: Option<Result<(), PopError>>,
}

/// A thread that pops every entry the front publishes and keeps it, so the front never waits on
/// a full ring while nothing on its thread takes entries.
pub(crate) struct Collector {
    shared: Arc<(Mutex<Collected>, Condvar)>,
    handle: Option<JoinHandle<()>>,
}

fn lock(mutex: &Mutex<Collected>) -> MutexGuard<'_, Collected> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Collector {
    /// Starts the thread on `consumer`, keeping at most `limit` words of entries at once.
    pub(crate) fn start(mut consumer: Consumer, limit: usize) -> std::io::Result<Self> {
        let shared = Arc::new((Mutex::new(Collected::default()), Condvar::new()));
        let theirs = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name("gpu-capture-ring".to_owned())
            .spawn(move || {
                let (mutex, changed) = &*theirs;
                loop {
                    let popped = consumer.pop();
                    let mut state = lock(mutex);
                    match popped {
                        Ok(entry) => {
                            state.taken = state.taken.saturating_add(1);
                            let words = encoded_words(&entry);
                            match state.words.checked_add(words).filter(|&w| w <= limit) {
                                Some(total) => {
                                    state.words = total;
                                    state.entries.push_back(entry);
                                }
                                None => state.overflowed = true,
                            }
                        }
                        Err(PopError::Disconnected) => {
                            state.ended = Some(Ok(()));
                            changed.notify_all();
                            return;
                        }
                        Err(error) => {
                            state.ended = Some(Err(error));
                            changed.notify_all();
                            return;
                        }
                    }
                    changed.notify_all();
                }
            })?;
        Ok(Self {
            shared,
            handle: Some(handle),
        })
    }

    /// Waits until `count` entries have been popped, or the ring ended, at most `timeout`;
    /// whether `count` were.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn wait_taken(&self, count: u64, timeout: Duration) -> bool {
        let (mutex, changed) = &*self.shared;
        let deadline = Instant::now().checked_add(timeout);
        let mut state = lock(mutex);
        while state.taken < count && state.ended.is_none() {
            let left = deadline.map_or(timeout, |d| d.saturating_duration_since(Instant::now()));
            if left.is_zero() {
                break;
            }
            state = changed
                .wait_timeout(state, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        state.taken >= count
    }

    /// The entries popped and kept since the last call, in ring order.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn drain(&self) -> Vec<RingEntry<'static>> {
        let mut state = lock(&self.shared.0);
        state.words = 0;
        state.entries.drain(..).collect()
    }

    /// Whether an entry was dropped past the limit.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn overflowed(&self) -> bool {
        lock(&self.shared.0).overflowed
    }

    /// Waits for the thread to end, which it does once the ring's producer is dropped and every
    /// entry is popped; then the entries kept and not drained, and how the ring ended.
    pub(crate) fn finish(mut self) -> (Vec<RingEntry<'static>>, Result<(), CollectError>) {
        let joined = self.handle.take().map_or(Ok(()), JoinHandle::join);
        let mut state = lock(&self.shared.0);
        let entries = state.entries.drain(..).collect();
        let ended = match (joined, state.ended.take(), state.overflowed) {
            (Err(_), _, _) => Err(CollectError::Panicked),
            (Ok(()), _, true) => Err(CollectError::Overflow),
            (Ok(()), Some(Err(error)), false) => Err(CollectError::Ring(error)),
            (Ok(()), Some(Ok(())) | None, false) => Ok(()),
        };
        (entries, ended)
    }
}

/// Why the collector lost entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CollectError {
    /// The front pushed more than [`MAX_COLLECTED_WORDS`] words the back had not taken.
    #[error("the front pushed more than {MAX_COLLECTED_WORDS} ring words the back had not taken")]
    Overflow,
    /// The ring failed.
    #[error("the ring failed: {0}")]
    Ring(PopError),
    /// The collector's thread panicked.
    #[error("the thread that empties the ring panicked")]
    Panicked,
}

/// The channels the front dropped, with how many entries it had pushed then: the back forgets
/// each once it has taken that many (as the wiring's back thread does, `crate::wiring`).
#[derive(Default)]
pub(crate) struct Retired {
    due: Mutex<VecDeque<(ChannelId, u64)>>,
}

impl ChannelRetired for Retired {
    fn retired(&self, channel: ChannelId, after: u64) {
        self.due
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push_back((channel, after));
    }
}

impl Retired {
    /// The channels due once `taken` entries are taken, in the order the front dropped them.
    pub(crate) fn due(&self, taken: u64) -> Vec<ChannelId> {
        let mut due = self.due.lock().unwrap_or_else(PoisonError::into_inner);
        let mut out = Vec::new();
        while let Some(&(channel, after)) = due.front() {
            if after > taken {
                break;
            }
            out.push(channel);
            due.pop_front();
        }
        out
    }
}

/// gsv's completion queue where nothing reads it: a capture records the GPU's inputs, and the
/// completions it pushes are outputs to gsv that the capture does not compare.
pub(crate) struct Discard;

impl Completions for Discard {
    fn push(&self, _completion: GpuCompletion) {}
}
