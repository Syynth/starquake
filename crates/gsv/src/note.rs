//! The notes the graphics server records for replay: every input replay cannot reproduce, written
//! by the timer thread as it drains the completion queue (`docs/specs/graphics-server.md`, "What
//! is recorded"; port slice 3 writes the first three).
//!
//! Each note is one [`EventKind::ServiceNote`]: the service's number, a code, and one `u64`
//! value. The codes and the packing of each value are starquake's own (no reference records
//! these), fixed here and pinned by `every_note_s_code_and_value_are_pinned`. Replaying reads
//! them back with [`Note::decode`].
//!
//! | Code | Note (spec row) | Value | Written by |
//! |---|---|---|---|
//! | 1 | `SyncpointObserved` | the id in bits 32..40 ([`SYNCPOINT_ID_SHIFT`]), the value in bits 0..32 | the drain, for each `SyncpointChanged` it takes |
//! | 2 | `SyncptPageAnswered` | reserved for the shared page (slice 5) | |
//! | 3 | `GpuOrderedDone` | the ticket | the drain, for each `GpuOrdered` it takes |
//! | 4 | `VsyncTick` | reserved for the vsync clock (slice 12) | |
//! | 5 | `FrameLatched` | reserved for present (slice 14) | |
//! | 6 | `BufferReleased` | reserved for present (slice 14) | |
//! | 7 | `VideoFrameDone` | reserved for the video worker | |
//! | 8 | `QueueRoom` | the channel | the drain, for each `GpuQueueRoom` it takes |
//!
//! The reserved codes keep the spec's order, so the slices that write them need not renumber
//! anything. Some of those notes hold more than one `u64` of fields (the spec gives
//! `FrameLatched` a client, a slot and an image); how they are split is their slices' to decide.

use starquake_gpu::interface::{ChannelId, SYNCPOINT_COUNT, SyncpointId, Ticket};
use starquake_trace::{EventKind, ServiceId};
use thiserror::Error;

#[cfg(test)]
mod tests;

/// The code of a `SyncpointObserved` note.
pub const SYNCPOINT_OBSERVED: u32 = 1;

/// The code of a `GpuOrderedDone` note.
pub const GPU_ORDERED_DONE: u32 = 3;

/// The code of a `QueueRoom` note.
pub const QUEUE_ROOM: u32 = 8;

/// Where a `SyncpointObserved` note's value holds the syncpoint id: the bits from here up; the
/// counter value is the `u32` below it.
pub const SYNCPOINT_ID_SHIFT: u32 = u32::BITS;

/// One recorded note (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Note {
    /// gsv acted on syncpoint `id` having reached `value`.
    SyncpointObserved {
        /// The syncpoint.
        id: SyncpointId,
        /// Its value.
        value: u32,
    },
    /// The GPU finished the ordered update or barrier queued with `ticket`.
    GpuOrderedDone {
        /// The ticket.
        ticket: Ticket,
    },
    /// The full queue of `channel` has room again.
    QueueRoom {
        /// The channel.
        channel: ChannelId,
    },
}

/// Why a recorded note could not be read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum NoteError {
    /// The code is not one gsv writes (yet).
    #[error("gsv writes no note with code {code}")]
    UnknownCode {
        /// The code.
        code: u32,
    },
    /// A `SyncpointObserved` value whose id bits name no syncpoint.
    #[error("syncpoint note value {value:#x} names no syncpoint (there are {SYNCPOINT_COUNT})")]
    NoSuchSyncpoint {
        /// The value.
        value: u64,
    },
    /// A `QueueRoom` value that is wider than a channel number.
    #[error("queue room note value {value:#x} is not a channel number")]
    NoSuchChannel {
        /// The value.
        value: u64,
    },
}

impl Note {
    /// The note's code (module docs).
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::SyncpointObserved { .. } => SYNCPOINT_OBSERVED,
            Self::GpuOrderedDone { .. } => GPU_ORDERED_DONE,
            Self::QueueRoom { .. } => QUEUE_ROOM,
        }
    }

    /// The note's value (module docs).
    #[must_use]
    pub fn value(self) -> u64 {
        match self {
            Self::SyncpointObserved { id, value } => {
                (u64::from(id.get()) << SYNCPOINT_ID_SHIFT) | u64::from(value)
            }
            Self::GpuOrderedDone { ticket } => ticket.0,
            Self::QueueRoom { channel } => u64::from(channel.0),
        }
    }

    /// The event that records the note, for the service numbered `service`.
    #[must_use]
    pub fn event(self, service: ServiceId) -> EventKind {
        EventKind::ServiceNote {
            service,
            code: self.code(),
            value: self.value(),
        }
    }

    /// The note a recorded `code` and `value` hold.
    pub fn decode(code: u32, value: u64) -> Result<Self, NoteError> {
        match code {
            SYNCPOINT_OBSERVED => {
                let id = u8::try_from(value >> SYNCPOINT_ID_SHIFT)
                    .ok()
                    .and_then(SyncpointId::new)
                    .ok_or(NoteError::NoSuchSyncpoint { value })?;
                let low = u32::try_from(value & !(u64::MAX << SYNCPOINT_ID_SHIFT))
                    .map_err(|_| NoteError::NoSuchSyncpoint { value })?;
                Ok(Self::SyncpointObserved { id, value: low })
            }
            GPU_ORDERED_DONE => Ok(Self::GpuOrderedDone {
                ticket: Ticket(value),
            }),
            QUEUE_ROOM => Ok(Self::QueueRoom {
                channel: ChannelId(
                    u32::try_from(value).map_err(|_| NoteError::NoSuchChannel { value })?,
                ),
            }),
            code => Err(NoteError::UnknownCode { code }),
        }
    }
}
