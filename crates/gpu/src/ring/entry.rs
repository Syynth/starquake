//! What crosses the ring: [`RingEntry`] and the plain data it carries (gpu-protocol spec, "The
//! ring", "Entries").
//!
//! The entry kinds are the spec's sketch. Where a payload type is shared with a neighbouring
//! design it comes from [`crate::interface`] ([`ChannelId`], [`SyncpointId`], [`PresentRequest`]),
//! and [`Invalidation`] is a host `MEM_OP_D` operation from `classes/clc76f.h`
//! (`mesa-nouveau/src/nouveau/headers/nvidia/classes/` @ 52238762, MIT).

use std::borrow::Cow;

use crate::field::Field;
use crate::fifo::METHOD_SUBCHANNEL;
use crate::interface::{ChannelId, PresentRequest, SyncpointId};
use crate::sema::{Reduction, ReleaseForm};

/// An engine class number, as bound with the host's bind-object method (`0xC797` for the 3D
/// class, `0` to unbind).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClassId(pub u32);

/// The number of subchannels: every value of a method header's 3-bit subchannel field
/// (`NVC76F_DMA_METHOD_SUBCHANNEL 15:13`).
pub const SUBCHANNEL_COUNT: u32 = METHOD_SUBCHANNEL.max() + 1;

/// A subchannel number, below [`SUBCHANNEL_COUNT`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Subchannel(u8);

impl Subchannel {
    /// The subchannel `value`, or `None` from [`SUBCHANNEL_COUNT`] up.
    pub fn new(value: u8) -> Option<Self> {
        (u32::from(value) < SUBCHANNEL_COUNT).then_some(Self(value))
    }

    /// The number.
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// What the back reports when it has performed everything before a [`RingEntry::Marker`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MarkerToken(pub u64);

/// What a release does at its address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReleaseOp {
    /// Writes the payload.
    Write,
    /// Applies a reduction with the payload.
    Reduce(Reduction),
}

/// A semaphore release or reduction, from any engine (the spec's `Release`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Release {
    /// The GPU address of the semaphore.
    pub address: u64,
    /// The payload (only the low 32 bits for a [`crate::sema::PayloadSize::Four`] form).
    pub payload: u64,
    /// The payload's width and whether a timestamp follows it.
    pub form: ReleaseForm,
    /// Write or reduce.
    pub op: ReleaseOp,
}

/// A host memory operation: the `OPERATION` field (31:27) of `NVC76F_MEM_OP_D`, with the values
/// `clc76f.h` defines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Invalidation {
    /// `MEMBAR` (5).
    Membar,
    /// `MMU_TLB_INVALIDATE` (9).
    MmuTlbInvalidate,
    /// `MMU_TLB_INVALIDATE_TARGETED` (0xA).
    MmuTlbInvalidateTargeted,
    /// `MMU_OPERATION` (0xB).
    MmuOperation,
    /// `L2_PEERMEM_INVALIDATE` (0xD).
    L2PeermemInvalidate,
    /// `L2_SYSMEM_INVALIDATE` (0xE; `L2_INVALIDATE_CLEAN_LINES` is the same value).
    L2SysmemInvalidate,
    /// `L2_CLEAN_COMPTAGS` (0xF).
    L2CleanComptags,
    /// `L2_FLUSH_DIRTY` (0x10).
    L2FlushDirty,
    /// `L2_WAIT_FOR_SYS_PENDING_READS` (0x15).
    L2WaitForSysPendingReads,
    /// `ACCESS_COUNTER_CLR` (0x16).
    AccessCounterClear,
}

/// `NVC76F_MEM_OP_D_OPERATION`.
pub const MEM_OP_D_OPERATION: Field = Field::new(31, 27);

/// Every [`Invalidation`] with its `MEM_OP_D_OPERATION` value from `clc76f.h`.
pub(super) const INVALIDATIONS: [(Invalidation, u32); 10] = [
    (Invalidation::Membar, 0x5),
    (Invalidation::MmuTlbInvalidate, 0x9),
    (Invalidation::MmuTlbInvalidateTargeted, 0xa),
    (Invalidation::MmuOperation, 0xb),
    (Invalidation::L2PeermemInvalidate, 0xd),
    (Invalidation::L2SysmemInvalidate, 0xe),
    (Invalidation::L2CleanComptags, 0xf),
    (Invalidation::L2FlushDirty, 0x10),
    (Invalidation::L2WaitForSysPendingReads, 0x15),
    (Invalidation::AccessCounterClear, 0x16),
];

impl Invalidation {
    /// The `MEM_OP_D_OPERATION` value.
    pub fn operation(self) -> u32 {
        INVALIDATIONS
            .iter()
            .find(|(what, _)| *what == self)
            .map_or(0, |&(_, value)| value)
    }

    /// The operation with `MEM_OP_D_OPERATION` value `value`, or `None` for a value the header
    /// does not define.
    pub fn from_operation(value: u32) -> Option<Self> {
        INVALIDATIONS
            .iter()
            .find(|&&(_, operation)| operation == value)
            .map(|&(what, _)| what)
    }

    /// The operation a `MEM_OP_D` word requests, or `None` for one the header does not define.
    pub fn from_mem_op_d(word: u32) -> Option<Self> {
        Self::from_operation(MEM_OP_D_OPERATION.get(word))
    }
}

/// Starts or stops a GPU capture at this point in the stream. **Guess** at the commands: the
/// spec names the entry (`CaptureControl(CaptureCommand)`) and leaves its content to the capture
/// slice (G20).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CaptureCommand {
    /// Start recording.
    Start,
    /// Stop recording.
    Stop,
}

/// One entry of the ring. The words of a [`RingEntry::Run`] are borrowed when the front pushes
/// and owned when the back pops.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RingEntry<'a> {
    /// A class bound (or, with class 0, unbound) on a subchannel.
    Bind {
        /// The channel.
        channel: ChannelId,
        /// The subchannel.
        subchannel: Subchannel,
        /// The class.
        class: ClassId,
    },
    /// One register write the front resolved.
    Write {
        /// The channel.
        channel: ChannelId,
        /// The class whose register it is.
        class: ClassId,
        /// The method (dword index).
        method: u16,
        /// The value.
        value: u32,
    },
    /// Data words to one trigger method (constant buffer loads, inline data), in order.
    Run {
        /// The channel.
        channel: ChannelId,
        /// The class.
        class: ClassId,
        /// The method (dword index).
        method: u16,
        /// The words, at most [`super::MAX_RUN_WORDS`].
        words: Cow<'a, [u32]>,
    },
    /// A semaphore release or reduction.
    Release(Release),
    /// A syncpoint increment.
    Syncpoint {
        /// The syncpoint.
        id: SyncpointId,
    },
    /// Wait for idle.
    WaitForIdle {
        /// The channel.
        channel: ChannelId,
    },
    /// A host memory operation.
    Invalidate {
        /// The channel.
        channel: ChannelId,
        /// The operation.
        what: Invalidation,
    },
    /// Report the token once everything before it is performed.
    Marker {
        /// The token.
        token: MarkerToken,
    },
    /// Show a frame.
    Present(PresentRequest),
    /// Start or stop a capture.
    CaptureControl(CaptureCommand),
}

impl RingEntry<'_> {
    /// The entry with its run words owned.
    pub fn into_owned(self) -> RingEntry<'static> {
        match self {
            Self::Bind {
                channel,
                subchannel,
                class,
            } => RingEntry::Bind {
                channel,
                subchannel,
                class,
            },
            Self::Write {
                channel,
                class,
                method,
                value,
            } => RingEntry::Write {
                channel,
                class,
                method,
                value,
            },
            Self::Run {
                channel,
                class,
                method,
                words,
            } => RingEntry::Run {
                channel,
                class,
                method,
                words: Cow::Owned(words.into_owned()),
            },
            Self::Release(release) => RingEntry::Release(release),
            Self::Syncpoint { id } => RingEntry::Syncpoint { id },
            Self::WaitForIdle { channel } => RingEntry::WaitForIdle { channel },
            Self::Invalidate { channel, what } => RingEntry::Invalidate { channel, what },
            Self::Marker { token } => RingEntry::Marker { token },
            Self::Present(present) => RingEntry::Present(present),
            Self::CaptureControl(command) => RingEntry::CaptureControl(command),
        }
    }

    /// The channel the entry belongs to, for the kinds that name one.
    pub fn channel(&self) -> Option<ChannelId> {
        match self {
            Self::Bind { channel, .. }
            | Self::Write { channel, .. }
            | Self::Run { channel, .. }
            | Self::WaitForIdle { channel }
            | Self::Invalidate { channel, .. } => Some(*channel),
            Self::Release(_)
            | Self::Syncpoint { .. }
            | Self::Marker { .. }
            | Self::Present(_)
            | Self::CaptureControl(_) => None,
        }
    }
}
