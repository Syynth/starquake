//! The ring between the GPU's front and back threads (gpu-protocol spec, "The ring" and slice
//! G4).
//!
//! The front sends the back everything that survives it, as [`RingEntry`]s: class binds,
//! resolved register writes, runs of data words, releases, syncpoint increments, wait-for-idle,
//! `SET_REFERENCE`, memory operations, markers, presents and capture control. Entries are plain data, so a stream
//! of them can be printed, captured and replayed.
//!
//! - `entry` owns [`RingEntry`] and the ring-only data it carries; the types it shares with gsv
//!   ([`crate::interface`]: channels, syncpoints, presents) are imported, not defined, here.
//! - `encode` owns the word encoding: [`encode`], [`decode`], [`encoded_words`] and
//!   [`entry_words`] (a header's kind and length), with the layout table in its docs.
//! - `text` owns the text form: `Display` on [`RingEntry`], [`parse_entry`], [`parse_text`] and
//!   [`write_text`].
//! - `spsc` owns the bounded single-producer, single-consumer ring itself: [`ring`] makes a
//!   [`Producer`] for the front and a [`Consumer`] for the back, with back-pressure and batched
//!   publishing, and a [`ConsumerWake`] that wakes the back without an entry (slice G15).
//!
//! The front loop (G14, [`crate::front`]) pushes into it; the back's executor (G19,
//! [`crate::back::exec`]) takes from it.

mod encode;
mod entry;
mod spsc;
mod text;

#[cfg(test)]
mod tests;

pub use encode::{
    DecodeError, EncodeError, Kind, MAX_RUN_WORDS, decode, encode, encoded_words, entry_words,
};
pub use entry::{
    CaptureCommand, ClassId, Invalidation, MEM_OP_D_OPERATION, MarkerToken, Release, ReleaseOp,
    RingEntry, SUBCHANNEL_COUNT, Subchannel,
};
pub use spsc::{
    CapacityError, Consumer, ConsumerWake, PUBLISH_EVERY, PopError, Popped, Producer, PushError,
    RING_WORDS, SPINS_BEFORE_SLEEP, ring,
};
pub use text::{ParseError, TextError, parse_entry, parse_text, write_text};
