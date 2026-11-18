//! The notes the graphics server records for replay: every input replay cannot reproduce, written
//! by the timer thread as it drains the completion queue or answers the shared syncpoint page
//! (`docs/specs/graphics-server.md`, "What is recorded"; port slice 3 writes codes 1, 3 and 8,
//! slice 5 codes 2 and 9, slice 12 code 4, slice 14 codes 5 and 6).
//!
//! Each note is one [`EventKind::ServiceNote`]: the service's number, a code, and one `u64`
//! value. The codes and the packing of each value are starquake's own (no reference records
//! these), fixed here and pinned by `every_note_s_code_and_value_are_pinned`. Replaying reads
//! them back with [`Note::decode`].
//!
//! | Code | Note (spec row) | Value | Written by |
//! |---|---|---|---|
//! | 1 | `SyncpointObserved` | the id in bits 32..40 ([`SYNCPOINT_ID_SHIFT`]), the value in bits 0..32 | the drain, for each `SyncpointChanged` it takes |
//! | 2 | `SyncptPageAnswered` | packed as code 1: the id in bits 32..40, the counter value the page's answer read in bits 0..32 | the timer thread's page turn, once for each value it writes into the page ([`crate::shared::page`], "What is recorded") |
//! | 3 | `GpuOrderedDone` | the ticket | the drain, for each `GpuOrdered` it takes |
//! | 4 | `VsyncTick` | the tick's number: the count of ticks the clock has delivered, this one included, so the first is 1 (0 is never written) | the timer thread's vsync turn, once for each tick, before the tick signals its listeners ([`crate::shared::vsync`], "What is recorded") |
//! | 5 | `FrameLatched` | the image id in bits 0..32, the buffer slot in bits 32..40 ([`SLOT_SHIFT`]; below [`SLOTS`]), the client's number in the present table in bits 40..48 ([`CLIENT_SHIFT`]; below [`MAX_PRESENT_CLIENTS`]), bits 48..64 zero | the vsync tick's latch, for each frame it latches, after the tick's `VsyncTick` ([`crate::shared::present`], "What is recorded") |
//! | 6 | `BufferReleased` | packed as code 5: the released frame's client, slot and image | the vsync tick's latch, for each frame a latched one replaces, after that one's `FrameLatched` and before the release's signal |
//! | 7 | `VideoFrameDone` | reserved for the video worker | |
//! | 8 | `QueueRoom` | the channel | the drain, for each `GpuQueueRoom` it takes |
//! | 9 | `SyncptPageAnswered`, its request bits (`SyncptPageAcknowledged`) | bit `g` set for each group `g` whose acknowledge word the answer wrote (bits below [`GROUP_COUNT`]; at least one) | the timer thread's page turn, once for each answer whose acknowledge summary write lands, after that write and before the answer event's signal, whether or not the answer wrote a value ([`crate::shared::page`], "What is recorded") |
//!
//! The reserved codes keep the spec's order, so the slices that write them need not renumber
//! anything.
//!
//! **The client of `FrameLatched` and `BufferReleased`.** The spec gives both "client, slot,
//! image". A vi2 client is named by a `u64` token, which leaves no room beside the slot and the
//! image, and two sessions may register the same token; so the notes name the client by its
//! number in gsv's present table ([`crate::shared::present`], "Why the queues are shared"),
//! which 201 gives it, is below [`MAX_PRESENT_CLIENTS`] and stays the client's until 202.
//! Decoding refuses, in this order, a slot at or past [`SLOTS`] ([`NoteError::NoSuchSlot`]) and
//! a client number at or past [`MAX_PRESENT_CLIENTS`], which covers any bit from 48 up
//! ([`NoteError::NoSuchClient`]).
//! The spec's `SyncptPageAnswered` ("request bits answered, values written") is split so: code 2
//! for each value written, code 9 for the request bits each answer acknowledged.

use starquake_gpu::interface::{ChannelId, SYNCPOINT_COUNT, SyncpointId, Ticket};
use starquake_trace::{EventKind, ServiceId};
use thiserror::Error;

use crate::shared::page::{GROUP_COUNT, GROUP_MASK};
use crate::shared::present::MAX_PRESENT_CLIENTS;
use crate::vi2::SLOTS;

#[cfg(test)]
mod tests;

/// The code of a `SyncpointObserved` note.
pub const SYNCPOINT_OBSERVED: u32 = 1;

/// The code of a `SyncptPageAnswered` note.
pub const SYNCPT_PAGE_ANSWERED: u32 = 2;

/// The code of a `GpuOrderedDone` note.
pub const GPU_ORDERED_DONE: u32 = 3;

/// The code of a `VsyncTick` note.
pub const VSYNC_TICK: u32 = 4;

/// The code of a `FrameLatched` note.
pub const FRAME_LATCHED: u32 = 5;
/// The code of a `BufferReleased` note.
pub const BUFFER_RELEASED: u32 = 6;
/// The code of a `QueueRoom` note.
pub const QUEUE_ROOM: u32 = 8;

/// The code of a `SyncptPageAcknowledged` note: the request bits half of the spec's
/// `SyncptPageAnswered` (module docs).
pub const SYNCPT_PAGE_ACKNOWLEDGED: u32 = 9;

/// Where a `SyncpointObserved` or `SyncptPageAnswered` note's value holds the syncpoint id: the
/// bits from here up; the counter value is the `u32` below it.
pub const SYNCPOINT_ID_SHIFT: u32 = u32::BITS;

/// Where a `FrameLatched` or `BufferReleased` note's value holds the buffer slot, a byte; the
/// image id is the `u32` below it.
pub const SLOT_SHIFT: u32 = u32::BITS;

/// Where a `FrameLatched` or `BufferReleased` note's value holds the client's number in the
/// present table: the bits from here up, the byte above the slot's.
pub const CLIENT_SHIFT: u32 = SLOT_SHIFT + u8::BITS;

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
    /// The shared syncpoint page's answer wrote syncpoint `id`, whose counter it read as `value`
    /// (the 64-bit value it wrote extends that: `crate::shared::page`).
    SyncptPageAnswered {
        /// The syncpoint.
        id: SyncpointId,
        /// Its counter, as the answer read it.
        value: u32,
    },
    /// The GPU finished the ordered update or barrier queued with `ticket`.
    GpuOrderedDone {
        /// The ticket.
        ticket: Ticket,
    },
    /// The vsync clock delivered its tick numbered `tick` ([`crate::shared::vsync`]).
    VsyncTick {
        /// The tick's number, counted from 1.
        tick: u64,
    },
    /// The vsync tick latched client `client`'s frame of buffer `slot`, native image `image`
    /// ([`crate::shared::present`]).
    FrameLatched {
        /// The client's number in the present table, below [`MAX_PRESENT_CLIENTS`].
        client: u8,
        /// The buffer slot, below [`SLOTS`].
        slot: u8,
        /// The native image.
        image: u32,
    },
    /// The vsync tick released client `client`'s frame of buffer `slot`, native image `image`,
    /// which a latched frame replaced ([`crate::shared::present`]).
    BufferReleased {
        /// The client's number in the present table, below [`MAX_PRESENT_CLIENTS`].
        client: u8,
        /// The buffer slot, below [`SLOTS`].
        slot: u8,
        /// The native image.
        image: u32,
    },
    /// The full queue of `channel` has room again.
    QueueRoom {
        /// The channel.
        channel: ChannelId,
    },
    /// The shared syncpoint page's answer wrote the acknowledge word of each group whose bit is
    /// set in `groups`, then the acknowledge summary (`crate::shared::page`, "What is recorded").
    SyncptPageAcknowledged {
        /// Bit `g` for group `g`: below [`GROUP_COUNT`], and at least one.
        groups: u64,
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
    /// A `SyncpointObserved` or `SyncptPageAnswered` value whose id bits name no syncpoint.
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
    /// A `FrameLatched` or `BufferReleased` value whose slot bits name no buffer slot.
    #[error("frame note value {value:#x} names no buffer slot (there are {SLOTS})")]
    NoSuchSlot {
        /// The value.
        value: u64,
    },
    /// A `FrameLatched` or `BufferReleased` value whose bits from [`CLIENT_SHIFT`] up name no
    /// client of the present table.
    #[error(
        "frame note value {value:#x} names no present table client (there are {MAX_PRESENT_CLIENTS})"
    )]
    NoSuchClient {
        /// The value.
        value: u64,
    },
    /// A `VsyncTick` value of 0: ticks are counted from 1.
    #[error("vsync tick note value 0 is not a tick number (ticks are counted from 1)")]
    NoSuchTick,
    /// A `SyncptPageAcknowledged` value that names no group, or a bit past the last group.
    #[error(
        "page acknowledge note value {value:#x} is not a non-empty set of the page's {GROUP_COUNT} groups"
    )]
    NoSuchGroups {
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
            Self::SyncptPageAnswered { .. } => SYNCPT_PAGE_ANSWERED,
            Self::GpuOrderedDone { .. } => GPU_ORDERED_DONE,
            Self::VsyncTick { .. } => VSYNC_TICK,
            Self::FrameLatched { .. } => FRAME_LATCHED,
            Self::BufferReleased { .. } => BUFFER_RELEASED,
            Self::QueueRoom { .. } => QUEUE_ROOM,
            Self::SyncptPageAcknowledged { .. } => SYNCPT_PAGE_ACKNOWLEDGED,
        }
    }

    /// The note's value (module docs).
    #[must_use]
    pub fn value(self) -> u64 {
        match self {
            Self::SyncpointObserved { id, value } | Self::SyncptPageAnswered { id, value } => {
                (u64::from(id.get()) << SYNCPOINT_ID_SHIFT) | u64::from(value)
            }
            Self::GpuOrderedDone { ticket } => ticket.0,
            Self::VsyncTick { tick } => tick,
            Self::FrameLatched {
                client,
                slot,
                image,
            }
            | Self::BufferReleased {
                client,
                slot,
                image,
            } => {
                (u64::from(client) << CLIENT_SHIFT)
                    | (u64::from(slot) << SLOT_SHIFT)
                    | u64::from(image)
            }
            Self::QueueRoom { channel } => u64::from(channel.0),
            Self::SyncptPageAcknowledged { groups } => groups,
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
                let (id, value) = unpack_syncpoint(value)?;
                Ok(Self::SyncpointObserved { id, value })
            }
            SYNCPT_PAGE_ANSWERED => {
                let (id, value) = unpack_syncpoint(value)?;
                Ok(Self::SyncptPageAnswered { id, value })
            }
            GPU_ORDERED_DONE => Ok(Self::GpuOrderedDone {
                ticket: Ticket(value),
            }),
            VSYNC_TICK => match value {
                0 => Err(NoteError::NoSuchTick),
                tick => Ok(Self::VsyncTick { tick }),
            },
            FRAME_LATCHED => {
                let (client, slot, image) = unpack_frame(value)?;
                Ok(Self::FrameLatched {
                    client,
                    slot,
                    image,
                })
            }
            BUFFER_RELEASED => {
                let (client, slot, image) = unpack_frame(value)?;
                Ok(Self::BufferReleased {
                    client,
                    slot,
                    image,
                })
            }
            QUEUE_ROOM => Ok(Self::QueueRoom {
                channel: ChannelId(
                    u32::try_from(value).map_err(|_| NoteError::NoSuchChannel { value })?,
                ),
            }),
            SYNCPT_PAGE_ACKNOWLEDGED => {
                if value == 0 || value & !GROUP_MASK != 0 {
                    return Err(NoteError::NoSuchGroups { value });
                }
                Ok(Self::SyncptPageAcknowledged { groups: value })
            }
            code => Err(NoteError::UnknownCode { code }),
        }
    }
}

/// The syncpoint id and the counter value a `SyncpointObserved` or `SyncptPageAnswered` note's
/// `value` packs (module docs).
fn unpack_syncpoint(value: u64) -> Result<(SyncpointId, u32), NoteError> {
    let id = u8::try_from(value >> SYNCPOINT_ID_SHIFT)
        .ok()
        .and_then(SyncpointId::new)
        .ok_or(NoteError::NoSuchSyncpoint { value })?;
    let low = u32::try_from(value & !(u64::MAX << SYNCPOINT_ID_SHIFT))
        .map_err(|_| NoteError::NoSuchSyncpoint { value })?;
    Ok((id, low))
}

/// The client's number, the buffer slot and the image id a `FrameLatched` or `BufferReleased`
/// note's `value` packs, checking the slot before the client (module docs).
fn unpack_frame(value: u64) -> Result<(u8, u8, u32), NoteError> {
    let slot = u8::try_from((value >> SLOT_SHIFT) & u64::from(u8::MAX))
        .ok()
        .filter(|&slot| slot < SLOTS)
        .ok_or(NoteError::NoSuchSlot { value })?;
    let client = u8::try_from(value >> CLIENT_SHIFT)
        .ok()
        .filter(|&client| usize::from(client) < MAX_PRESENT_CLIENTS)
        .ok_or(NoteError::NoSuchClient { value })?;
    let image = u32::try_from(value & !(u64::MAX << SLOT_SHIFT))
        .map_err(|_| NoteError::NoSuchSlot { value })?;
    Ok((client, slot, image))
}
