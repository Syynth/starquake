//! What crosses the ring: [`RingEntry`] and the plain data it carries (gpu-protocol spec, "The
//! ring", "Entries").
//!
//! The entry kinds are the spec's sketch. Where a payload type is shared with a neighbouring
//! design its fields follow that design: [`PresentRequest`] is graphics-server's ("Interfaces to
//! neighbouring designs"), with the image fields its section "What vi2 needs from the image"
//! lists (row 48 of Eden-NX2's notes), and [`Invalidation`] is a host `MEM_OP_D` operation from
//! `classes/clc76f.h` (`mesa-nouveau/src/nouveau/headers/nvidia/classes/` @ 52238762, MIT).

use std::borrow::Cow;

use starquake_render::command::Rect;

use crate::field::Field;
use crate::fifo::METHOD_SUBCHANNEL;
use crate::sema::{Reduction, ReleaseForm};

/// A GPU channel, as gsv created it (`GpuFront::create_channel`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChannelId(pub u32);

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

/// The number of syncpoints: the size of Eden's host1x syncpoint table, which graphics-server
/// keeps ("Syncpoints"; its verification item 10 checks the console's count).
pub const SYNCPOINT_COUNT: u32 = 192;

/// A syncpoint, below [`SYNCPOINT_COUNT`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyncpointId(u8);

impl SyncpointId {
    /// The syncpoint `value`, or `None` from [`SYNCPOINT_COUNT`] up.
    pub fn new(value: u8) -> Option<Self> {
        (u32::from(value) < SYNCPOINT_COUNT).then_some(Self(value))
    }

    /// The number.
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// What the back reports when it has performed everything before a [`RingEntry::Marker`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MarkerToken(pub u64);

/// A vi2 layer. A **guess** at the width: `nn::vi` layer ids are 64-bit; graphics-server names
/// the type only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayerId(pub u64);

/// A vi2 display. A **guess** at the width, as for [`LayerId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DisplayId(pub u64);

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
    /// The payload (only the low 32 bits for a [`PayloadSize::Four`] form).
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

/// The guest image a present shows, as vi2 learns it from a native image's preference words
/// (graphics-server, "What vi2 needs from the image": word 2 the size, word 4 the block height,
/// word 5 the pitch, word 6 the format).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageDesc {
    /// The image's GPU address.
    pub address: u64,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row pitch in 64-byte GOBs.
    pub pitch: u32,
    /// Block height, as log2 of GOB rows.
    pub block_height_log2: u32,
    /// The format word as the guest gave it (1 RGBA8, 4 RGB10A2).
    pub format: u32,
}

/// A frame gsv latched after its fence was reached (graphics-server's `PresentRequest`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PresentRequest {
    /// The image.
    pub image: ImageDesc,
    /// The area of it shown (word 7 of vi2 request 229).
    pub shown: Rect,
    /// The layer.
    pub layer: LayerId,
    /// The display.
    pub display: DisplayId,
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
