//! HLE Horizon kernel.
//!
//! Guest threads and their scheduling onto vCPUs, SVC dispatch, handles and kernel objects,
//! synchronisation (`WaitForAddress`/`SignalToAddress` are hot: ~1,400 per frame), memory
//! regions, and `svcGetInfo` including the Switch 2 ids 29-33.
//!
//! Structure (`docs/specs/kernel.md`): kernel state lives in a few tables, objects refer to each
//! other by typed ids, system calls declare the tables they touch, and only the dispatcher locks.

pub mod dispatch;
pub mod handle;
pub mod id;
pub mod result;
pub mod sched;
pub mod slab;
pub mod svc;
pub mod tables;
