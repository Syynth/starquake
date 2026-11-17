//! Test helpers shared by the queue's tests and the drain's: device-share tokens and a kernel
//! that takes them back.

use std::collections::BTreeMap;

use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::id::{ProcessId, SlabId};
use starquake_kernel::memory::DeviceShareToken;
use starquake_kernel::memory::testing::device_share_token;

use super::{Completion, Reclaim};

/// The process the tests' tokens name: not the first slot or generation, so a hard-coded one
/// fails.
pub(crate) fn process() -> ProcessId {
    ProcessId::from_parts(2, 3)
}

/// The size of each test token's range.
const RANGE: u64 = 0x1000;

/// The token for lock `n`: [`process`]'s range `n` (`n * RANGE`, `RANGE` bytes).
pub(crate) fn token(n: u64) -> DeviceShareToken {
    device_share_token(process(), n * RANGE, RANGE)
}

/// The release of lock `n`.
pub(crate) fn released(n: u64) -> Completion {
    Completion::DeviceShareReleased { token: token(n) }
}

/// How many locks [`locks`] holds: `token(0)` to `token(LOCKS - 1)`.
pub(crate) const LOCKS: u64 = 8;

/// A [`Reclaim`] standing in for the kernel's `Dispatcher::reclaim_device_share`: it holds a
/// count of live locks on exact ranges of one process and answers as the kernel does for a live
/// process, another process being [`DeviceShareError::WrongProcess`] and a range on which it
/// holds no more locks than `earlier` being [`DeviceShareError::NotLocked`]. It has no process
/// that ends, so it never answers [`DeviceShareError::ProcessEnded`]. Parity with the kernel:
/// `the_fake_kernel_reclaims_as_the_real_one_refuses`.
pub(crate) struct Locks {
    /// The process whose memory it holds.
    pub(crate) process: ProcessId,
    /// How many live locks it holds, by `(address, size)`.
    pub(crate) live: BTreeMap<(u64, u64), u32>,
}

impl Reclaim for Locks {
    fn reclaim(
        &self,
        process: ProcessId,
        address: u64,
        size: u64,
        earlier: u32,
    ) -> Result<DeviceShareToken, DeviceShareError> {
        if process != self.process {
            return Err(DeviceShareError::WrongProcess);
        }
        if self.live.get(&(address, size)).copied().unwrap_or(0) <= earlier {
            return Err(DeviceShareError::NotLocked { address, size });
        }
        Ok(device_share_token(process, address, size))
    }
}

/// A kernel holding one lock on each of `0..LOCKS` of [`process`].
pub(crate) fn locks() -> Locks {
    Locks {
        process: process(),
        live: (0..LOCKS)
            .map(|n| ((token(n).address(), token(n).size()), 1))
            .collect(),
    }
}

/// A dispatcher over a fresh process, which holds no device-share lock, and that process.
pub(crate) fn real_kernel() -> (starquake_kernel::dispatch::Dispatcher, ProcessId) {
    use starquake_kernel::memory::testing::RecordingMapper;
    use starquake_kernel::memory::{AddressSpace, MemoryArrangement, ProcessMemory, RegionOffsets};
    use starquake_kernel::process::{self, Console, ProcessDescription};

    let (mapper, _log) = RecordingMapper::new();
    let memory = ProcessMemory {
        code_address: 0x8000_0000,
        code_size: 0x1000,
        arrangement: MemoryArrangement::FourGb,
        alias_region_extra: false,
        offsets: RegionOffsets::default(),
    };
    let space = AddressSpace::new(&memory, Box::new(mapper)).unwrap();
    let tables = process::create(&ProcessDescription::homebrew(Console::Switch2), space).unwrap();
    let own = tables.processes.own();
    (starquake_kernel::dispatch::Dispatcher::new(tables), own)
}
