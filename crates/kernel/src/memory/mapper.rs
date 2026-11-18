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
//!
//! Shared memory is the exception: its pages must be the same for every mapping of it and for
//! the service that writes them, so they are named. [`Mapper::allocate_shared`] hands out a
//! range of backing as [`SharedPages`], and [`Mapper::map_shared`] maps that range. Both have a
//! default that refuses, for a translation with no backing to share (the kernel then fails the
//! call inside itself). `starquake-core`'s translation hands out guest RAM for both
//! (`SpaceMapper` in `crates/core/src/space.rs`).
//!
//! `MapMemory`'s alias is the other exception: its pages must be the source's, so
//! [`Mapper::map_alias`] names the source by its address. It has a default that refuses too
//! ([`NO_ALIAS`]); `starquake-core`'s translation maps the source's backing a second time.

use starquake_memory::{BackingRange, Perms};

use super::shared::SharedPages;

/// What the default [`Mapper::allocate_shared`] and [`Mapper::map_shared`] refuse with.
pub const NO_SHARED_BACKING: &str = "this translation has no backing to share";

/// What the default [`Mapper::map_alias`] refuses with.
pub const NO_ALIAS: &str = "this translation cannot map the same pages at two addresses";

/// Why a [`Mapper`] could not make a change real, in the translation's own words.
///
/// The kinds are told apart because one of them has a Horizon result of its own:
/// [`MapperError::OutOfMemory`] from [`Mapper::allocate_shared`] is `ResultOutOfMemory`
/// (`R_UNLESS(m_physical_address != 0, ResultOutOfMemory)` in `KSharedMemory::Initialize`,
/// `eden/src/core/hle/kernel/k_shared_memory.cpp` @ cf654052;
/// [`crate::memory::AddressSpace::allocate_shared`]). Every other refusal, of either kind from
/// any other call, is the kernel's own failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MapperError {
    /// The translation refused the change: it cannot make it, or was asked for something it
    /// does not hold.
    #[error("{0}")]
    Refused(String),
    /// The translation has no room left for the backing asked for.
    #[error("{0}")]
    OutOfMemory(String),
}

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
    /// it held is gone: mapped again, it reads as zero. Pages mapped with
    /// [`Mapper::map_shared`] or [`Mapper::map_alias`] are the exception: they belong to their
    /// shared memory, or to the alias's source, and keep what they hold there.
    fn unmap(&mut self, address: u64, size: u64) -> Result<(), MapperError>;
    /// Guest code may now use the `size` bytes at `address` as `perms` allows, and no
    /// otherwise. All of it was mapped, and keeps what it holds.
    fn protect(&mut self, address: u64, size: u64, perms: Perms) -> Result<(), MapperError>;

    /// Backing for a shared memory of `size` bytes (whole pages, not zero): pages that read as
    /// zero, that host code reaches through the returned [`SharedPages`], and that
    /// [`Mapper::map_shared`] maps for guest code. The pages must be `size` bytes; the address
    /// space refuses any others. [`MapperError::OutOfMemory`] when there is no room for them,
    /// with nothing handed out.
    ///
    /// The default has no backing to share and refuses with [`NO_SHARED_BACKING`].
    fn allocate_shared(&mut self, _size: u64) -> Result<SharedPages, MapperError> {
        Err(MapperError::Refused(NO_SHARED_BACKING.into()))
    }

    /// Guest code may now use the `backing.size()` bytes at `address` as `perms` allows, and
    /// they are the pages of `backing`, which [`Mapper::allocate_shared`] handed out: what host
    /// code writes through its [`SharedPages`] guest code reads there, and the reverse. Nothing
    /// was mapped there.
    ///
    /// The default refuses with [`NO_SHARED_BACKING`].
    fn map_shared(
        &mut self,
        _address: u64,
        _backing: BackingRange,
        _perms: Perms,
    ) -> Result<(), MapperError> {
        Err(MapperError::Refused(NO_SHARED_BACKING.into()))
    }

    /// The last reference to the shared memory whose pages are `backing` (which
    /// [`Mapper::allocate_shared`] handed out) has gone (`KSharedMemory::Finalize`: the page
    /// group is closed): no mapping of them is left, and nothing will map them again. The
    /// translation may hand them out again, but only once every flush owed by the unmappings
    /// of them has run, and zeroed (CLAUDE.md: a freed resource is not reusable until every
    /// pending follow-up on it has run). A refusal leaves the pages as they were.
    ///
    /// The default refuses with [`NO_SHARED_BACKING`]: it hands out no shared pages, so it was
    /// given pages it does not hold.
    fn release_shared(&mut self, _backing: BackingRange) -> Result<(), MapperError> {
        Err(MapperError::Refused(NO_SHARED_BACKING.into()))
    }

    /// Guest code may now use the `size` bytes at `address` as `perms` allows, and they are the
    /// pages of the `size` bytes at `source`: what is written through one is read through the
    /// other (`MapPageGroupImpl` with the source's page group, in `KPageTableBase::MapMemory`).
    /// Nothing was mapped at `address`; all of `source` was mapped, and keeps its own
    /// permission. [`Mapper::unmap`] of the alias leaves the source's pages and what they hold.
    ///
    /// The default refuses with [`NO_ALIAS`].
    fn map_alias(
        &mut self,
        _address: u64,
        _source: u64,
        _size: u64,
        _perms: Perms,
    ) -> Result<(), MapperError> {
        Err(MapperError::Refused(NO_ALIAS.into()))
    }
}
