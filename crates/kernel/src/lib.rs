//! HLE Horizon kernel.
//!
//! Guest threads and their scheduling onto vCPUs, SVC dispatch, handles and kernel objects,
//! synchronisation (`WaitForAddress`/`SignalToAddress` are hot: ~1,400 per frame), a process's
//! address space ([`memory`]), and `svcGetInfo` (its memory ids so far; the Switch 2 ids 29-33
//! are to come).
//!
//! Structure (`docs/specs/kernel.md`): kernel state lives in a few tables, objects refer to each
//! other by typed ids, system calls declare the tables they touch, and only the dispatcher locks.

pub mod dispatch;
pub mod handle;
pub mod id;
pub mod memory;
pub mod object;
pub mod result;
pub mod sched;
pub mod slab;
pub mod svc;
pub mod tables;
pub mod time;
