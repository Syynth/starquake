//! The types gsv and the GPU share: the block that graphics-server's "Interfaces to neighbouring
//! designs" and gpu-protocol's "Graphics server sessions (#11)" give in the same words. They live
//! in this crate because gsv depends on it (graphics-server, decision 3), and in this module, not
//! in the module that first used them, so every slice that needs one (the ring, G4; the
//! syncpoints, graphics-server slice 2; the front loop, G14; the wiring, G21) imports the same
//! definition.
//!
//! This module holds the ones that exist so far: [`ChannelId`], [`SyncpointId`] and
//! [`SYNCPOINT_COUNT`], and the present's [`PresentRequest`] with its [`ImageDesc`], [`LayerId`]
//! and [`DisplayId`]. The rest of the block (`Fence`, `Syncpoints`, `Submission`, `GpuFront`,
//! ...) belongs here too when its slice lands.

use starquake_render::command::Rect;

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
