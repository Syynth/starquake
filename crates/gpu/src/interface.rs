//! The types gsv and the GPU share: the block that graphics-server's "Interfaces to neighbouring
//! designs" and gpu-protocol's "Graphics server sessions (#11)" give in the same words. They live
//! in this crate because gsv depends on it (graphics-server, decision 3), and in this module, not
//! in the module that first used them, so every slice that needs one (the ring, G4; the
//! syncpoints, graphics-server slice 2; the front loop, G14; the wiring, G21) imports the same
//! definition.
//!
//! This module holds the ones that exist so far: [`ChannelId`], [`SyncpointId`] and
//! [`SYNCPOINT_COUNT`], the present's [`PresentRequest`] with its [`ImageDesc`], [`LayerId`]
//! and [`DisplayId`], graphics-server slice 2's [`Fence`], [`Ticket`], [`GpuCompletion`] and
//! [`Completions`], and slice 6's [`Target`], [`RegionId`], [`Kind`] and [`VmUpdate`]. The
//! block's `Syncpoints` and `GpuVm` have behaviour, not only a shape, so they live in
//! [`crate::syncpoint`] and [`crate::vm`]. The rest of the block (`Submission`, `GpuFront`, ...)
//! belongs here too when its slice lands.

use starquake_memory::PageRange;
/// The area of an image a present shows ([`PresentRequest::shown`]): the render backend's
/// rectangle, named here so that gsv, which builds a [`PresentRequest`] and does not depend on
/// the render crate, can name it too.
pub use starquake_render::command::Rect;

/// A GPU channel, as gsv created it (`GpuFront::create_channel`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChannelId(pub u32);

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

/// A fence: done once syncpoint `id` reaches `value` (graphics-server, "Syncpoints and the syncpt
/// session"), by the 32-bit wrapping comparison of
/// [`Syncpoints::reached`](crate::syncpoint::Syncpoints::reached).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fence {
    /// The syncpoint.
    pub id: SyncpointId,
    /// The value it has to reach.
    pub value: u32,
}

/// The ticket of an ordered update or a barrier (`GpuFront::update_vm`, `GpuFront::barrier`),
/// reported back in [`GpuCompletion::GpuOrdered`]. Not a protocol fact: gsv hands tickets out
/// and only compares them, so the width is this crate's choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ticket(pub u64);

/// What a host thread reports to gsv's completion queue (graphics-server, "Who signals what: the
/// completion queue"; the same block in gpu-protocol, "Graphics server sessions (#11)").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuCompletion {
    /// A watched syncpoint changed. Pushed by
    /// [`Syncpoints::increment`](crate::syncpoint::Syncpoints::increment), which the back thread
    /// calls after host completion and gsv's video worker calls per finished frame.
    SyncpointChanged {
        /// The syncpoint.
        id: SyncpointId,
        /// Its value after the change.
        value: u32,
    },
    /// An ordered update or a barrier is done.
    GpuOrdered {
        /// The ticket it was queued with.
        ticket: Ticket,
    },
    /// A full channel queue has room again.
    GpuQueueRoom {
        /// The channel.
        channel: ChannelId,
    },
}

/// Where host threads push [`GpuCompletion`]s: gsv's completion queue, which gsv implements
/// (graphics-server slice 3) and drains on its timer thread, so guest waiters wake in one
/// recorded order. `push` never blocks and is the only call from the GPU into gsv
/// (graphics-server, "What gsv assumes").
pub trait Completions: Send + Sync {
    /// Queues `completion`.
    fn push(&self, completion: GpuCompletion);
}

/// A sparse region (gsv 214, `CreateSparseRegion`), as gsv names it. Not a protocol fact: the
/// guest names a region by its range, and gsv hands these ids out and only compares them, so the
/// width is this crate's choice (as for [`Ticket`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId(pub u32);

/// A GPU page-table kind (the PTE kind) as the GPU reads it: one byte, the kind a mapping's pages
/// are translated with.
///
/// PR's `SelectNativeMemoryKind`
/// (`eden-pr4588/src/core/hle/service/nvdrv/devices/native_gpu_kind.h` @ e5fb3b97) turns the
/// guest's requested and fallback kinds (gsv 211, words 12 and 13; gsv 212, words 7 and 8) into
/// two bytes, `NativeMemoryKind { backend, selected }`: `backend` is the translated PTE kind PR
/// passes to `gmmu->Map` (`MapNative4K` and `ModifyKind` in
/// `eden-pr4588/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.cpp`, `NormalMapGpuMemory4K`
/// in `eden-pr4588/src/core/hle/service/am/service/gpu_session.cpp`, both @ e5fb3b97), and
/// `selected` is the requested byte sent back in the 211 and 212 reply. **A `Kind` is the
/// `backend` byte**: it is what [`crate::vm::GpuVm`] stores and what a translation gives the GPU.
/// The `selected` byte is the reply's, which gsv computes and sends (slice 7) and never reaches
/// this crate. NX2's 212 handler takes `w[7] & 0xff`
/// (`eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb, `SubModifyKind`) and keeps no kind.
/// The address table only stores the byte; what each value means is the GPU's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Kind(pub u8);

/// Where a mapped GPU range leads (graphics-server, "The model"). In a [`crate::vm::GpuVm`] each
/// mapping has one target, which names its *first* page; a page further in leads as far further
/// in (see [`crate::vm::Translation`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// Memory the title owns, read through `GuestMemory` at this CPU address (a 101
    /// registration at its own address, or a 211 map of part of one).
    Guest {
        /// The CPU address.
        cpu_va: u64,
    },
    /// GPU-only memory: `offset` bytes into sparse region `region`, backed on the host by this
    /// crate (graphics-server decision 4).
    GpuOnly {
        /// The sparse region.
        region: RegionId,
        /// The byte offset into it.
        offset: u64,
    },
    /// The read-only page through which GPU reads return syncpoint values (row 46 of
    /// `eden-nx2/NX2_NOTES.md`; nvhost-as-gpu ioctl 0xB reports where it is).
    SyncpointShim,
}

/// A change to a [`crate::vm::GpuVm`] that the GPU applies only once no work queued before it can
/// still use the old mapping (`GpuFront::update_vm`; graphics-server, "How mappings reach the
/// GPU"). Each carries its GPU range.
///
/// The three removals do the same to the table; they are told apart because the GPU does more for
/// some (a sparse-region destroy frees that region's host storage, gpu-protocol slice G22).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VmUpdate {
    /// gsv 213, `NormalUnmapGpuMemory4K`.
    Unmap {
        /// The GPU range.
        range: PageRange,
    },
    /// gsv 212, `NormalModifyMemoryKind4K`: the mapped pages of `range` take `kind`.
    ModifyKind {
        /// The GPU range.
        range: PageRange,
        /// The kind applied.
        kind: Kind,
    },
    /// gsv 102, `DestroyGpuAccessibleMemory`: the registration's range at its own address.
    DestroyMemory {
        /// The GPU range.
        range: PageRange,
    },
    /// gsv 215, `DestroySparseRegion`.
    DestroySparseRegion {
        /// The GPU range.
        range: PageRange,
        /// The region destroyed.
        region: RegionId,
    },
}

impl VmUpdate {
    /// The GPU range the update applies to.
    pub const fn range(&self) -> PageRange {
        match *self {
            Self::Unmap { range }
            | Self::ModifyKind { range, .. }
            | Self::DestroyMemory { range }
            | Self::DestroySparseRegion { range, .. } => range,
        }
    }
}

/// A vi2 layer. A **guess** at the width: `nn::vi` layer ids are 64-bit; graphics-server names
/// the type only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayerId(pub u64);

/// A vi2 display. A **guess** at the width, as for [`LayerId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DisplayId(pub u64);

/// The guest image a present shows, as vi2 learns it from a native image's preference words
/// (graphics-server, "What vi2 needs from the image": word 2 the size, word 4 the block height,
/// word 5 the pitch, word 6 the format; row 48 of Eden-NX2's notes).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_syncpoint_id_is_checked_at_its_bound() {
        // 192 syncpoints: graphics-server's `SyncpointId` (`< SYNCPOINT_COUNT (192)`).
        assert_eq!(SyncpointId::new(191).map(SyncpointId::get), Some(191));
        assert_eq!(SyncpointId::new(192), None);
        let last = u8::try_from(SYNCPOINT_COUNT - 1).unwrap();
        assert_eq!(SyncpointId::new(last).map(SyncpointId::get), Some(last));
        let first_out = u8::try_from(SYNCPOINT_COUNT).unwrap();
        assert_eq!(SyncpointId::new(first_out), None);
    }
}
