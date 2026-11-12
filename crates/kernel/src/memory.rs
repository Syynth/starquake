//! A process's address space: Horizon's page-table bookkeeping without the hardware part.
//!
//! Kernel spec, "Address space" (port slice 6, and the part of slice 7 that `GetInfo`'s memory
//! ids need):
//!
//! - [`state`]: memory states, permissions, attributes and merge barriers.
//! - [`block`]: one memory block, and what `QueryMemory` reports of it.
//! - [`block_map`]: the sorted map of blocks, with merge and split.
//! - [`layout`]: where the regions of a 39-bit address space go. Placement is deterministic by
//!   default.
//! - [`pool`]: the console's memory arrangement and a process's memory count.
//! - [`space`]: [`AddressSpace`], the table a system call declares, and its operations.
//! - [`mapper`]: [`Mapper`], through which a change becomes real for guest code.
//!
//! The system calls that use it are in [`crate::svc::memory`] and [`crate::svc::info`].

pub mod block;
pub mod block_map;
#[cfg(test)]
pub(crate) mod fixture;
pub mod layout;
pub mod mapper;
pub mod pool;
pub mod space;
pub mod state;
#[cfg(any(test, feature = "test-support"))]
pub mod testing;

pub use block::{MemoryBlock, SvcMemoryInfo};
pub use layout::{Region, RegionOffsets, Regions};
pub use mapper::{Mapper, MapperError};
pub use pool::{MemoryArrangement, MemoryLimit};
pub use space::{AddressSpace, CreateError, GUARD_PAGES, ProcessMemory};
pub use state::{MemoryAttribute, MemoryPermission, MemoryState, SvcPermission};
