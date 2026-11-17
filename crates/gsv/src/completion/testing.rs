//! Test helpers shared by the queue's tests and the drain's: device-share tokens and a kernel
//! that takes them back.

use std::collections::BTreeSet;

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

/// A [`Reclaim`] standing in for the kernel's `Dispatcher::reclaim_device_share`: it holds live
/// locks on exact ranges of one process and answers as the kernel does for a live process,
/// another process being [`DeviceShareError::WrongProcess`] and a range it holds no lock on
/// [`DeviceShareError::NotLocked`]. It has no process that ends, so it never answers
/// [`DeviceShareError::ProcessEnded`]. Parity with the kernel:
/// `the_fake_kernel_reclaims_as_the_real_one_refuses`.
pub(crate) struct Locks {
    /// The process whose memory it holds.
    pub(crate) process: ProcessId,
    /// The live locks, by `(address, size)`.
    pub(crate) live: BTreeSet<(u64, u64)>,
}

impl Reclaim for Locks {
    fn reclaim(
        &self,
        process: ProcessId,
        address: u64,
        size: u64,
    ) -> Result<DeviceShareToken, DeviceShareError> {
        if process != self.process {
            return Err(DeviceShareError::WrongProcess);
        }
        if !self.live.contains(&(address, size)) {
            return Err(DeviceShareError::NotLocked { address, size });
        }
        Ok(device_share_token(process, address, size))
    }
}

/// A kernel holding locks `0..LOCKS` of [`process`].
pub(crate) fn locks() -> Locks {
    Locks {
        process: process(),
        live: (0..LOCKS)
            .map(|n| (token(n).address(), token(n).size()))
            .collect(),
    }
}
