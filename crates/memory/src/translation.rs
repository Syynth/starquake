//! The translation layer of guest memory: the [`Translation`] trait that makes the address-space
//! bookkeeping real for guest code, and the values it takes.
//!
//! The trait follows `docs/specs/execution-model.md` ("Guest memory", "Three layers"):
//! `map`, `unmap` and `protect` over guest virtual ranges, with permissions as EL0 (guest user
//! code) sees them. [`crate::stage1::Stage1Tables`] implements it for the native backends; a
//! JIT backend would implement it with a software MMU.

use crate::Fault;

/// The guest page size: the stage-1 translation granule. Titles are told 0x1000 by `svcGetInfo`
/// 33 (`docs/specs/execution-model.md`, "Native backends").
pub const PAGE_SIZE: u64 = 0x1000;

/// What guest user code (EL0) may do with a mapped page.
///
/// Write access always comes with read access: AArch64 stage-1 descriptors can't express a
/// write-only page (Arm DDI 0487, data access permissions `AP[2:1]`), so the type has no
/// variant for one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Perms {
    /// Mapped, but EL0 may neither access nor execute it.
    None,
    /// Read only.
    Read,
    /// Read and write.
    ReadWrite,
    /// Execute only: instruction fetches succeed, loads and stores fault.
    Execute,
    /// Read and execute (code).
    ReadExecute,
    /// Read, write and execute.
    ReadWriteExecute,
}

impl Perms {
    /// Whether EL0 loads from the page succeed.
    pub const fn readable(self) -> bool {
        matches!(
            self,
            Self::Read | Self::ReadWrite | Self::ReadExecute | Self::ReadWriteExecute
        )
    }

    /// Whether EL0 stores to the page succeed.
    pub const fn writable(self) -> bool {
        matches!(self, Self::ReadWrite | Self::ReadWriteExecute)
    }

    /// Whether EL0 may execute instructions from the page.
    pub const fn executable(self) -> bool {
        matches!(
            self,
            Self::Execute | Self::ReadExecute | Self::ReadWriteExecute
        )
    }

    /// Whether these permissions allow `access`.
    pub const fn allows(self, access: Access) -> bool {
        match access {
            Access::Read => self.readable(),
            Access::Write => self.writable(),
            Access::Execute => self.executable(),
        }
    }
}

/// One kind of access EL0 makes to a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    /// A load.
    Read,
    /// A store.
    Write,
    /// An instruction fetch.
    Execute,
}

/// A page-aligned, non-empty range of guest-physical memory that backs a mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackingRange {
    base: u64,
    size: u64,
}

impl BackingRange {
    /// Describes `size` bytes of guest-physical memory starting at `base`.
    ///
    /// Both must be multiples of [`PAGE_SIZE`], `size` must be non-zero, and `base + size` must
    /// not overflow.
    pub fn new(base: u64, size: u64) -> Result<Self, MapError> {
        check_page_range(base, size)?;
        if base.checked_add(size).is_none() {
            return Err(MapError::PhysicalRange { pa: base, size });
        }
        Ok(Self { base, size })
    }

    /// The first guest-physical address of the range.
    pub const fn base(self) -> u64 {
        self.base
    }

    /// The length of the range in bytes.
    pub const fn size(self) -> u64 {
        self.size
    }

    /// One past the last guest-physical address of the range. Never overflows: [`Self::new`]
    /// checks it.
    pub const fn end(self) -> u64 {
        self.base.wrapping_add(self.size)
    }

    /// Whether this range and `other` share at least one byte.
    pub const fn overlaps(self, other: Self) -> bool {
        self.base < other.end() && other.base < self.end()
    }
}

/// Checks that `address` and `size` are page multiples and `size` is non-zero, in that order.
pub(crate) fn check_page_range(address: u64, size: u64) -> Result<(), MapError> {
    if !address.is_multiple_of(PAGE_SIZE) {
        return Err(MapError::MisalignedAddress { address });
    }
    if !size.is_multiple_of(PAGE_SIZE) {
        return Err(MapError::MisalignedSize { size });
    }
    if size == 0 {
        return Err(MapError::Empty);
    }
    Ok(())
}

/// Why a [`Translation`] operation (or building the structure behind one) failed.
///
/// A failed `map`, `unmap` or `protect` changes nothing, with one exception: a [`Self::Memory`]
/// fault while writing the tables, which [`crate::stage1::Stage1Tables`] can only hit if its
/// memory stops accepting writes it accepted when the tables were created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MapError {
    /// An address is not a multiple of [`PAGE_SIZE`].
    #[error("address {address:#x} is not page aligned")]
    MisalignedAddress {
        /// The misaligned address.
        address: u64,
    },
    /// A size is not a multiple of [`PAGE_SIZE`].
    #[error("size {size:#x} is not a multiple of the page size")]
    MisalignedSize {
        /// The misaligned size.
        size: u64,
    },
    /// A range of zero bytes.
    #[error("empty range")]
    Empty,
    /// A virtual range that overflows or ends past the translated address space.
    #[error("virtual range {va:#x}+{size:#x} is outside the address space")]
    VirtualRange {
        /// The start of the range.
        va: u64,
        /// Its size.
        size: u64,
    },
    /// A physical range that overflows or ends past what a descriptor can address.
    #[error("physical range {pa:#x}+{size:#x} is outside the output address space")]
    PhysicalRange {
        /// The start of the range.
        pa: u64,
        /// Its size.
        size: u64,
    },
    /// The backing range overlaps the memory that holds the translation tables, which would let
    /// the guest rewrite its own tables.
    #[error("backing {pa:#x}+{size:#x} overlaps the translation table pool")]
    OverlapsTables {
        /// The start of the backing range.
        pa: u64,
        /// Its size.
        size: u64,
    },
    /// `map` over a page that is already mapped.
    #[error("{va:#x} is already mapped")]
    AlreadyMapped {
        /// The lowest page of the range that is already mapped.
        va: u64,
    },
    /// `unmap` or `protect` over a page that is not mapped.
    #[error("{va:#x} is not mapped")]
    NotMapped {
        /// The lowest page of the range that is not mapped.
        va: u64,
    },
    /// No free page is left for a new translation table.
    #[error("out of translation table pages")]
    OutOfTablePages,
    /// The table memory holds a descriptor the builder never writes.
    #[error("unexpected descriptor {descriptor:#x} at level {level} for {va:#x}")]
    UnexpectedDescriptor {
        /// The first virtual address the descriptor covers within the operation's range.
        va: u64,
        /// The table level it was found at.
        level: u8,
        /// The raw descriptor.
        descriptor: u64,
    },
    /// The memory holding the tables refused an access.
    #[error(transparent)]
    Memory(#[from] Fault),
}

/// Maps guest virtual ranges onto guest-physical backing with EL0 permissions.
///
/// Every operation works on whole pages and is all or nothing (see [`MapError`] for the one
/// exception).
pub trait Translation {
    /// Maps `backing.size()` bytes at `va` onto `backing`. Every page of the range must be
    /// unmapped.
    fn map(&self, va: u64, backing: BackingRange, perms: Perms) -> Result<(), MapError>;
    /// Unmaps `size` bytes at `va`. Every page of the range must be mapped.
    fn unmap(&self, va: u64, size: u64) -> Result<(), MapError>;
    /// Changes the permissions of `size` bytes at `va`, keeping their backing. Every page of the
    /// range must be mapped.
    fn protect(&self, va: u64, size: u64, perms: Perms) -> Result<(), MapError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_PERMS: [Perms; 6] = [
        Perms::None,
        Perms::Read,
        Perms::ReadWrite,
        Perms::Execute,
        Perms::ReadExecute,
        Perms::ReadWriteExecute,
    ];

    #[test]
    fn perms_report_exactly_their_accesses() {
        let expected = [
            (false, false, false),
            (true, false, false),
            (true, true, false),
            (false, false, true),
            (true, false, true),
            (true, true, true),
        ];
        for (perms, (read, write, execute)) in ALL_PERMS.into_iter().zip(expected) {
            assert_eq!(perms.readable(), read, "{perms:?}");
            assert_eq!(perms.writable(), write, "{perms:?}");
            assert_eq!(perms.executable(), execute, "{perms:?}");
            assert_eq!(perms.allows(Access::Read), read, "{perms:?}");
            assert_eq!(perms.allows(Access::Write), write, "{perms:?}");
            assert_eq!(perms.allows(Access::Execute), execute, "{perms:?}");
        }
    }

    #[test]
    fn backing_range_validates_in_order() {
        assert_eq!(
            BackingRange::new(PAGE_SIZE + 1, PAGE_SIZE),
            Err(MapError::MisalignedAddress {
                address: PAGE_SIZE + 1
            })
        );
        assert_eq!(
            BackingRange::new(PAGE_SIZE, PAGE_SIZE - 1),
            Err(MapError::MisalignedSize {
                size: PAGE_SIZE - 1
            })
        );
        assert_eq!(BackingRange::new(PAGE_SIZE, 0), Err(MapError::Empty));
        // The last page of the 64-bit space ends exactly at 2^64, which overflows.
        let last = u64::MAX - (PAGE_SIZE - 1);
        assert_eq!(
            BackingRange::new(last, PAGE_SIZE),
            Err(MapError::PhysicalRange {
                pa: last,
                size: PAGE_SIZE
            })
        );
        let below = last - PAGE_SIZE;
        let range = BackingRange::new(below, PAGE_SIZE).unwrap();
        assert_eq!(
            (range.base(), range.size(), range.end()),
            (below, PAGE_SIZE, last)
        );
    }

    #[test]
    fn overlap_is_exact_at_the_edges() {
        let a = BackingRange::new(4 * PAGE_SIZE, 2 * PAGE_SIZE).unwrap();
        let touching_below = BackingRange::new(2 * PAGE_SIZE, 2 * PAGE_SIZE).unwrap();
        let touching_above = BackingRange::new(6 * PAGE_SIZE, PAGE_SIZE).unwrap();
        let first_page = BackingRange::new(3 * PAGE_SIZE, 2 * PAGE_SIZE).unwrap();
        let last_page = BackingRange::new(5 * PAGE_SIZE, 4 * PAGE_SIZE).unwrap();
        assert!(!a.overlaps(touching_below) && !touching_below.overlaps(a));
        assert!(!a.overlaps(touching_above) && !touching_above.overlaps(a));
        assert!(a.overlaps(first_page) && first_page.overlaps(a));
        assert!(a.overlaps(last_page) && last_page.overlaps(a));
        assert!(a.overlaps(a));
    }

    #[test]
    fn errors_name_the_offending_values() {
        assert_eq!(
            MapError::AlreadyMapped { va: 0x4000 }.to_string(),
            "0x4000 is already mapped"
        );
        assert_eq!(
            MapError::NotMapped { va: 0x5000 }.to_string(),
            "0x5000 is not mapped"
        );
        assert_eq!(
            MapError::Memory(Fault { address: 0x6000 }).to_string(),
            "guest memory fault at 0x6000"
        );
    }
}
