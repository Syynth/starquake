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
//! - [`space::snapshot`]: the address space saved and restored ([`AddressSpace::save`],
//!   [`AddressSpace::restore`]): the blocks and the counts, with the [`Mapper`] rebuilt, since a
//!   `Box<dyn Mapper>` cannot be serialised (observability spec, "Order of work", step 6,
//!   "kernel and memory first").
//! - [`mapper`]: [`Mapper`], through which a change becomes real for guest code.
//! - [`shared`]: shared and transfer memory as kernel objects, and the pages of a shared
//!   memory as host code reaches them.
//!
//! The system calls that use it are in [`crate::svc::memory`], [`crate::svc::info`],
//! [`crate::svc::shared_memory`] and [`crate::svc::transfer_memory`].
//!
//! # Not built in this slice
//!
//! One thing the specs ask of the address space is not here:
//!
//! - **The `Map` recording point.** `docs/specs/observability.md` has each of the address
//!   space's mutators call the recorder once, and `docs/specs/kernel.md` ("Recording and
//!   replay") lists the address space as a recording point. [`AddressSpace`]'s mutators record
//!   nothing, and no `EventKind::Map` is produced anywhere. Owner: a follow-up to slice 6 (not
//!   yet scheduled), which gives the mutators the recorder.

pub mod block;
pub mod block_map;
#[cfg(test)]
pub(crate) mod fixture;
pub mod layout;
pub mod mapper;
pub mod pool;
pub mod shared;
pub mod space;
pub mod state;
#[cfg(any(test, feature = "test-support"))]
pub mod testing;

pub use block::{MemoryBlock, SvcMemoryInfo};
pub use layout::{Region, RegionOffsets, Regions};
pub use mapper::{Mapper, MapperError};
pub use pool::{MemoryArrangement, MemoryLimit};
pub use shared::{
    SharedMemory, SharedPages, TransferMemory, is_shared_memory_permission,
    is_transfer_memory_permission, shared_memory_size,
};
pub use space::snapshot::{
    ADDRESS_SPACE_MAGIC, ADDRESS_SPACE_VERSION, ARRANGEMENT_TAGS, AliasFault, BlockFault,
    LockFault, SharedFault, SnapshotError,
};
pub use space::{AddressSpace, CreateError, DeviceShareToken, GUARD_PAGES, ProcessMemory};
pub use state::{MemoryAttribute, MemoryPermission, MemoryState, SvcPermission};
