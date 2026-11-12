//! The ring between the GPU's front and back threads (gpu-protocol spec, "The ring" and slice
//! G4).
//!
//! The front sends the back everything that survives it, as [`RingEntry`]s: class binds,
//! resolved register writes, runs of data words, releases, syncpoint increments, wait-for-idle,
//! memory operations, markers, presents and capture control. Entries are plain data, so a stream
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
//!   publishing.
//!
//! What uses the ring (the front loop, G14; the back's executor, G19) is not here yet.

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
    CapacityError, Consumer, PUBLISH_EVERY, PopError, Producer, PushError, RING_WORDS,
    SPINS_BEFORE_SLEEP, ring,
};
pub use text::{ParseError, TextError, parse_entry, parse_text, write_text};
