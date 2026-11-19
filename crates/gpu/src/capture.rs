//! GPU capture and replay (gpu-protocol spec, slice G20: "Capture and replay", "Observability";
//! observability spec, "Headless runs" and "Storage and what may be committed").
//!
//! A capture records every input each GPU thread consumed, in the order it consumed it, so that
//! replaying it runs the real front and back on any host with no game and no GPU (owner's
//! decision of 2026-11-12, gpu decision 8 (a): "record every byte each GPU thread read, in
//! order"). No reference is ported here: the format and both drivers are new.
//!
//! - [`format`]: the `SQGC` file, [`Capture::to_bytes`] and [`Capture::parse`].
//! - [`record`]: the seams that record as the threads consume ([`RecordingMemory`], the two
//!   clocks, [`RecordingShaders`]) into a [`Recorder`].
//! - [`replay()`]: the seams that answer from the logs and the replay driver.
//! - `builder` (the `test-support` feature only): the capture builder, which records a capture by
//!   running the real GPU on one thread over hand-written method streams.
//!
//! # The file
//!
//! Little-endian throughout. In order:
//!
//! 1. **Header**: magic `SQGC` ([`MAGIC`]), version (`u32`, [`VERSION`]), build id (`u32`
//!    length, at most [`MAX_BUILD_ID`] bytes of UTF-8), the render backend's `Caps` (a blob
//!    hash, below; the blob is `postcard`'s encoding of `starquake_render::caps::Caps`), its
//!    memory budget (`u64`), the wait-while-compiling setting (`u8`: 0 wait, 1 skip), the fence
//!    mode (`u8`: 0 fenced, 1 at record), the address spaces (`u32` count, `u32` each) and the
//!    channels (`u32` count; each `u32` id, `u32` address space, `u8` syncpoint).
//! 2. **Starting state**: `u32` count, then records.
//! 3. **Front log**: `u32` count, then records.
//! 4. **Back log**: `u32` count, then records.
//! 5. **Blob table**: `u32` count; each a 16-byte hash, a `u64` length and the bytes, in strictly
//!    ascending hash order.
//! 6. **Outputs**: `u32` count, then records. Nothing may follow.
//!
//! A record is a tag byte ([`TAGS`]; the tags are pairwise distinct, so a record in the wrong
//! section is refused) and its fields. A hash is [`ContentHash`]: XXH3-128, 16 bytes. Fields
//! of a sub-kind are a byte, then that kind's fields:
//!
//! | Tag | Record | Fields after the tag |
//! |---|---|---|
//! | `01` | address table | `u32` space, hash of [`crate::vm::GpuVm::save`]'s bytes |
//! | `02` | syncpoints | hash of [`crate::syncpoint::Syncpoints::save`]'s bytes |
//! | `03` | engine state | `u8` kind ([`EngineStateKind`], 1–7), `u32` owner, hash |
//! | `10` | turn | none |
//! | `11` | submit | `u32` channel; `u8` 0, or 1 then `u8` syncpoint and `u32` value (the wait fence); `u32` increments; `u8` wait-for-idle; `u32` count of `u64` GP entries |
//! | `12` | update | `u32` space; `u8` kind (0 unmap, 1 modify kind, 2 destroy memory, 3 destroy sparse region), `u64` start, `u64` size, then `u8` kind for 1 or `u32` region for 3; `u8` 0, or 1 then `u64` ticket |
//! | `13` | barrier | `u64` ticket |
//! | `14` | present | `u64` address, `u32` width, height, pitch, block height (log2), format; `u32` x, y, width, height shown; `u64` layer, `u64` display |
//! | `15` | destroy | `u32` channel |
//! | `16` | report | `u8` released, `u8` incremented, `u32` count of `u64` marker tokens |
//! | `17` | syncpoint | `u8` syncpoint, `u32` value |
//! | `18` | latch | `u32` channel |
//! | `19` | clock | `u64` nanoseconds |
//! | `20` | apply | none |
//! | `21` | close | none |
//! | `22` | complete | `u64` completed fence value |
//! | `23` | elapsed | `u64` nanoseconds |
//! | `24` | timestamp | `u64` |
//! | `25` | waiters | `u32` count; each `u32` channel, `u32` space, `u64` address, `u8` operation (0 equal, 1 greater or equal, 2 circular greater or equal, 3 and, 4 nor), `u32` payload, `u8` latched |
//! | `26` | drain | `u32` space, `u32` count of (`u64` start, `u64` size) |
//! | `27` | shader lookup | `u32` count of (`u8` stage, 0–6 in `GuestStage` order, `u64` address); answer |
//! | `28` | shader wait | `u64` ticket; answer |
//! | `30` | read | `u32` space, `u64` address, `u64` length; `u8` 0 then the bytes' hash, or 1 then a fault |
//! | `31` | write | `u32` space, `u64` address, the bytes' hash; `u8` 0, or 1 then a fault |
//! | `32` | GPU-only query | `u32` space, `u64` address, `u64` length, `u8` answer |
//! | `33` | generation | `u32` space, `u64` value |
//! | `40` | ring | the hash of one ring entry's encoding |
//! | `41` | render | the hash of the render command stream's text |
//!
//! A fault is `u8` kind then: 0 unmapped, `u64` address; 1 out of range, `u64` address and
//! `u64` length; 2 guest, `u64` GPU address and `u64` guest address; 3 shim write, `u64`
//! address. A shader answer is `u8` 0 ready, 1 pending then `u64` ticket, or 2 rejected.
//!
//! The reader checks the magic, then the version, before anything else; then every field's
//! range (a syncpoint below 192, whole pages, flags 0 or 1, known kinds), the header's lists
//! (spaces distinct, channel ids distinct, each channel's space listed), the blob table's order
//! and each blob's hash, that every blob a record names is in the table, and that a read's
//! length is its blob's. Golden bytes for every record are pinned in `tests::golden`.
//!
//! **Bounds.** [`Capture::parse`] reads at most [`MAX_CAPTURE_BYTES`]; every list it reads is
//! read item by item (no allocation from a count), so what it holds is at most a small multiple
//! of the file's length. A replay's ring is held by a collector bounded at
//! [`MAX_COLLECTED_WORDS`]; past it the replay stops ([`CollectError::Overflow`]).
//!
//! # What the logs hold
//!
//! The spec's list, and where each item is:
//!
//! - front: submissions, ordered updates, barriers, presents and destroys taken (each before
//!   the turn that took it); memory reads, with the bytes by hash (acquires read their
//!   semaphores through memory, so the value an acquire saw is the read's bytes); the
//!   syncpoint values fence waits read, as a `syncpoint` record whenever one changed since the
//!   front's last turn; back reports received; the acquire poll's ticks, as the readings of the
//!   front's clock; and the waiters the back latched since the front's last turn (`latch`).
//! - back: memory reads; completions, as a `complete` record with the backend's completed
//!   value; the steps the back took (`apply`, `close`); its clock readings; the waiter table
//!   when the front changed it (releases close the open list on a waiter, and latch them); and
//!   shader lookups and waits.
//! - both: writes (with their bytes and outcome: an outcome is an input, the bytes an output),
//!   and the two queries a memory answers.
//!
//! # Replay
//!
//! [`replay()`] restores the starting state and builds a fresh front with a [`ReplayMemory`]
//! per address space answering from the front log, runs one turn per `turn` record after
//! queueing the requests and reports recorded before it, then stops it: the ring it pushed is
//! compared with the capture's ring hashes, first difference first
//! ([`ReplayError::RingDiffers`], then [`ReplayError::RingLength`]). Then a fresh back (the
//! recording backend with the capture's `Caps`, the executor, the fence path) takes its steps
//! from the back log: each `apply` applies the next entry the front pushed. When the back ran
//! the whole log, the recording backend's snapshot text is compared with the capture's render
//! stream ([`ReplayError::RenderDiffers`]). `--until present=N` stops the back once it has
//! applied `N` presents.
//!
//! A call that is not the log's next record, a top-level record the driver did not expect, or
//! a `complete` whose fence the backend has not reached, stops the replay with the thread and
//! the record's index ([`Divergence`]). Single-threaded on each thread's side, so deterministic
//! on any host: the only other thread empties the ring and changes nothing the GPU reads.
//!
//! # Not built, and departures
//!
//! - **Starting state of a running GPU.** The spec's capture starts at a present boundary with
//!   every channel's register files, shadow file, pusher state, bindings, macro code and data
//!   RAM and the QMD table. The front has no snapshot of its state yet (`crate::front`, "Not
//!   here"), so the only captures made start from a fresh GPU (the builder's): their starting
//!   state is the address tables and the syncpoints. The `engine state` record names each kind
//!   the spec lists; their blob layouts are not defined, and a replay refuses them
//!   ([`ReplayError::UnsupportedState`]).
//! - **Shader lookups.** The record keeps the requests and the kind of answer, not the variant
//!   key, the questions answered or the shaders: those go in the shader capture format of
//!   shader-pipeline slice 20, which is not built. A replay feeds back a pending answer; a ready
//!   or rejected one stops it ([`ReplayError::Unsupported`]). The live cache's reads while it
//!   answered are in the back log before the lookup record, and a replayed lookup steps past
//!   them.
//! - **Write-tracking drains.** Only the builder records captures, and its watches are never
//!   armed, so no capture has a `drain` record; a replay refuses one
//!   ([`ReplayError::Unsupported`]).
//! - **No live capture.** Nothing records a running title yet: the observability spec's
//!   contract has no switch for a GPU capture today ("neither capture has a switch"), and the
//!   threaded wiring (`crate::wiring`) would need the recording seams on its two threads and a
//!   starting-state snapshot.
//! - **A faulted read's bytes.** A read that faulted may have filled part of its buffer; the
//!   capture keeps only the fault, and a replay gives zeros.
//! - **Unrecorded outputs.** Completions pushed to gsv and shader-cache invalidations are not
//!   recorded or compared.

#[cfg(any(test, feature = "test-support"))]
pub mod builder;
mod drive;
pub mod format;
pub mod record;
pub mod replay;
pub mod text;

#[cfg(test)]
mod tests;

pub use drive::{CollectError, MAX_COLLECTED_WORDS, entry_hash};
pub use format::{
    Access, BackRecord, Blobs, Capture, CaptureError, CapturedChannel, ContentHash,
    EngineStateKind, FrontRecord, Header, MAGIC, MAX_BUILD_ID, MAX_CAPTURE_BYTES, Output,
    ShaderAnswer, StateRecord, TAGS, VERSION,
};
pub use record::{
    RecordError, Recorded, Recorder, RecordingFenceClock, RecordingFrontClock, RecordingMemory,
    RecordingShaders, Thread,
};
pub use replay::{
    Divergence, ReplayError, ReplayMemory, ReplayOptions, ReplayOutput, ReplayedWrite, replay,
};
pub use text::{WRITE_BYTES_SHOWN, writes_text};
