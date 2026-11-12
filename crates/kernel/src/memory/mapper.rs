//! [`Mapper`]: how a change to an address space's bookkeeping becomes real for guest code.
//!
//! Kernel spec, "Address space": each operation "validates against the block map and then calls
//! the execution model's `Translation` trait to make the change real". This trait is the
//! kernel's side of that call. It is Eden's seam too: `KPageTableBase::Operate`
//! (`eden/src/core/hle/kernel/k_page_table_base.cpp` @ cf654052) ends in
//! `Core::Memory::Memory::MapMemoryRegion`, `UnmapRegion` and `ProtectRegion`.
//!
//! It differs from `starquake_memory::Translation` in one way: `map` names no backing. Horizon's
//! kernel picks physical pages from its page heaps; starquake's kernel only counts memory
//! (kernel spec, "Not planned": page heaps are "replaced by pool accounting"), and the
//! execution model backs pages when they are first touched ("Sparse and lazy memory"). So
//! whoever implements this trait owns the backing, and maps it through `Translation` when it
//! chooses to. `starquake-core` does.

use starquake_memory::Perms;

/// Why a [`Mapper`] could not make a change real.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct MapperError(pub String);

/// Makes the changes of one address space real for its guest code. Every range is whole pages
/// inside the address space: the address space checks that before it calls.
///
/// A call that fails must leave the translation as it was. After a failure the address space
/// leaves its own bookkeeping as it was too, and the system call fails inside the kernel.
pub trait Mapper: Send {
    /// Guest code may now use the `size` bytes at `address` as `perms` allows. Nothing was
    /// mapped there, and the memory reads as zero until something is written to it.
    fn map(&mut self, address: u64, size: u64, perms: Perms) -> Result<(), MapperError>;
    /// Guest code may no longer use the `size` bytes at `address`. All of it was mapped. What
    /// it held is gone: mapped again, it reads as zero.
    fn unmap(&mut self, address: u64, size: u64) -> Result<(), MapperError>;
    /// Guest code may now use the `size` bytes at `address` as `perms` allows, and no
    /// otherwise. All of it was mapped, and keeps what it holds.
    fn protect(&mut self, address: u64, size: u64, perms: Perms) -> Result<(), MapperError>;
}
