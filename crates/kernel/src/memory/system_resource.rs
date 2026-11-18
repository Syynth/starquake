//! A process's system resource (Horizon's `KSecureSystemResource`): the memory the kernel sets
//! aside for its own structures for the process (page tables, memory blocks), sized by the
//! program's metadata (`META` 0x14), and what it charges to the process's memory count.
//!
//! Ported from Eden (GPL-3.0-or-later), paths relative to the reference folder, all @ cf654052;
//! Eden PR #4588's copies of `KSecureSystemResource::Initialize`, the application overload of
//! `KProcess::Initialize` and the two `KSystemControl` functions are the same, line for line
//! (`eden-pr4588/...` @ e5fb3b97):
//!
//! - `eden/src/core/hle/kernel/k_process.cpp`, `KProcess::LoadFromMetadata`:
//!   `.system_resource_num_pages = metadata.GetSystemResourceSize() / PageSize`, so a recorded
//!   size counts in whole pages ([`SystemResource::new`]).
//! - The same file, `KProcess::Initialize` (the overload `LoadFromMetadata` calls), in its order:
//!   1. `KScopedResourceReservation memory_reservation(kernel, res_limit, PhysicalMemoryMax,
//!      code_size)`, `R_UNLESS(memory_reservation.Succeeded(), ResultLimitReached)`: the code.
//!   2. `if (system_resource_num_pages != 0)`: `secure_resource->Initialize(kernel,
//!      system_resource_size, res_limit, m_memory_pool)`, with `system_resource_size =
//!      system_resource_num_pages * PageSize`; otherwise the process uses the kernel's shared
//!      default resource, which charges nothing.
//!   3. The page table, then `MapPages` of the code.
//!
//!   [`crate::memory::AddressSpace::with_system_resource`] keeps that order.
//! - `eden/src/core/hle/kernel/k_system_resource.cpp`, `KSecureSystemResource::Initialize`, in
//!   its order ([`SystemResource::initialize`]):
//!   1. `secure_size = CalculateRequiredSecureMemorySize()`; `KScopedResourceReservation
//!      memory_reservation(kernel, m_resource_limit, PhysicalMemoryMax, secure_size)`,
//!      `R_UNLESS(memory_reservation.Succeeded(), ResultLimitReached)`.
//!   2. `R_TRY(KSystemControl::AllocateSecureMemory(...))`.
//!   3. `rc_size = AlignUp(KPageTableSlabHeap::CalculateReferenceCountSize(m_resource_size),
//!      PageSize)`, `R_UNLESS(m_resource_size > rc_size, ResultOutOfMemory)`.
//!   4. `memory_reservation.Commit()`: the charge stays until the resource is finalized
//!      (`Finalize`, `Release(PhysicalMemoryMax, CalculateRequiredSecureMemorySize())`).
//! - `eden/src/core/hle/kernel/board/nintendo/nx/k_system_control.cpp`:
//!   `CalculateRequiredSecureMemorySize` is zero for the applet pool and the size for any other
//!   ([`SystemResource::required_secure_memory_size`]); `AllocateSecureMemory` checks
//!   `R_UNLESS(Common::IsAligned(size, alignment), ResultInvalidSize)`, `alignment` being
//!   `PageSize` for the system pool and `SecureAlignment` (128 KiB, [`SECURE_ALIGNMENT`]) for
//!   any other ([`SystemResource::alignment`]).
//! - `eden/src/core/hle/kernel/k_page_table_slab_heap.h`: `CalculateReferenceCountSize(size)` is
//!   `(size / PageSize) * sizeof(RefCount)`, `RefCount` a `u16` ([`REFERENCE_COUNT_SIZE`]).
//!
//! Who charges it (#242, `crate::resource_limit`): the charge goes to the address space's
//! [`MemoryLimit`], the one count behind the process's resource limit's `PhysicalMemoryMax`
//! value, so `GetResourceLimitCurrentValue` and `GetInfo` 6, 7 and 21 see it with no count of
//! their own.
//!
//! # Divergences from Eden
//!
//! - **No secure memory is allocated.** Eden takes the resource's pages from the pool's page
//!   heap (`AllocateAndOpenContinuous`) and fails with `ResultOutOfMemory` when the heap has no
//!   run of them; here the pool is a count (kernel spec, "Not planned": page heaps), so a
//!   resource that passes the checks always gets its memory, and nothing reads or writes it
//!   (the translation tables are the backend's own). The checks on the size are Eden's.
//! - **A system resource in the applet pool** charges nothing, as in Eden, and is checked for
//!   [`SECURE_ALIGNMENT`] like any pool but the system pool. Eden's `AllocateSecureMemory`
//!   asserts first (`UNIMPLEMENTED_IF(pool == Applet)`, "Applet secure memory is handled
//!   separately") and then makes the same check. What Horizon checks for it is a guess here;
//!   a trace of an applet's creation would settle it. Test:
//!   `an_applet_resource_charges_nothing_and_is_checked_like_an_applications`.

use starquake_memory::PAGE_SIZE;

use super::pool::{MemoryLimit, Pool};
use crate::result::{KernelError, SvcResult};

/// The alignment a secure system resource of any pool but the system pool must have:
/// `SecureAlignment = 128_KiB` (`eden/src/core/hle/kernel/board/nintendo/nx/k_system_control.cpp`
/// @ cf654052).
pub const SECURE_ALIGNMENT: u64 = 128 << 10;

/// Bytes of one page-table reference count: `sizeof(KPageTableSlabHeap::RefCount)`, a `u16`
/// (`eden/src/core/hle/kernel/k_page_table_slab_heap.h` @ cf654052).
pub const REFERENCE_COUNT_SIZE: u64 = 2;

/// A process's system resource: its size in whole pages and the pool it is in. A size of zero
/// is no secure resource (the process uses the kernel's shared default one, which charges
/// nothing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemResource {
    size: u64,
    pool: Pool,
}

impl SystemResource {
    /// No secure resource, in the application pool: what a program with no metadata has
    /// (`ProgramMetadata::GetDefault`, `system_resource_size` zero and a zero `ACID`).
    pub const NONE: Self = Self {
        size: 0,
        pool: Pool::Application,
    };

    /// The resource of a process whose metadata records `recorded_size` bytes (`META` 0x14) and
    /// whose `ACID` names `pool`: the size in whole pages, rounded down
    /// (`metadata.GetSystemResourceSize() / PageSize`, then `* PageSize`).
    #[must_use]
    pub const fn new(recorded_size: u32, pool: Pool) -> Self {
        let size = recorded_size as u64;
        Self {
            size: size - size % PAGE_SIZE,
            pool,
        }
    }

    /// Its size in bytes, whole pages (`KSecureSystemResource::GetSize`, what `GetInfo` 16
    /// reports): zero for no secure resource.
    #[must_use]
    pub const fn size(self) -> u64 {
        self.size
    }

    /// The pool it is in: the process's own.
    #[must_use]
    pub const fn pool(self) -> Pool {
        self.pool
    }

    /// Whether it is a secure resource of its own: a size of a page or more
    /// (`system_resource_num_pages != 0`).
    #[must_use]
    pub const fn is_secure(self) -> bool {
        self.size != 0
    }

    /// What it charges to the process's memory count (`CalculateRequiredSecureMemorySize`, and
    /// so `GetRequiredSecureMemorySize`): zero in the applet pool, its size in any other.
    #[must_use]
    pub const fn required_secure_memory_size(self) -> u64 {
        match self.pool {
            Pool::Applet => 0,
            Pool::Application | Pool::System | Pool::SystemNonSecure => self.size,
        }
    }

    /// The alignment its size must have (`AllocateSecureMemory`): the page size in the system
    /// pool, [`SECURE_ALIGNMENT`] in any other.
    #[must_use]
    pub const fn alignment(self) -> u64 {
        match self.pool {
            Pool::System => PAGE_SIZE,
            Pool::Application | Pool::Applet | Pool::SystemNonSecure => SECURE_ALIGNMENT,
        }
    }

    /// The bytes its page-table reference counts take, in whole pages:
    /// `AlignUp(CalculateReferenceCountSize(size), PageSize)`. The size is at most a `u32`'s
    /// worth of pages, so this cannot overflow.
    #[must_use]
    pub const fn reference_count_size(self) -> u64 {
        (self.size / PAGE_SIZE * REFERENCE_COUNT_SIZE).next_multiple_of(PAGE_SIZE)
    }

    /// The checks `KSecureSystemResource::Initialize` makes after its reservation, in its order:
    /// the size aligned to [`Self::alignment`], else [`KernelError::InvalidSize`]
    /// (`AllocateSecureMemory`); then larger than [`Self::reference_count_size`], else
    /// [`KernelError::OutOfMemory`]. No secure resource passes both (Eden does not initialize
    /// one).
    pub const fn check_allocation(self) -> SvcResult<()> {
        if !self.is_secure() {
            return Ok(());
        }
        if !self.size.is_multiple_of(self.alignment()) {
            return Err(KernelError::InvalidSize);
        }
        if self.size <= self.reference_count_size() {
            return Err(KernelError::OutOfMemory);
        }
        Ok(())
    }

    /// Sets the resource up against `limit` (`KSecureSystemResource::Initialize`, module docs):
    /// reserves [`Self::required_secure_memory_size`] ([`KernelError::LimitReached`] when it
    /// does not fit), then [`Self::check_allocation`]. On any failure `limit` is unchanged; on
    /// success it holds the charge. No secure resource changes nothing and succeeds.
    pub fn initialize(self, limit: &mut MemoryLimit) -> SvcResult<()> {
        if !self.is_secure() {
            return Ok(());
        }
        // The reservation is made on a copy and committed last, as Eden commits its scoped
        // reservation only once every check has passed.
        let mut charged = *limit;
        charged.reserve(self.required_secure_memory_size())?;
        self.check_allocation()?;
        *limit = charged;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
