//! AArch64 stage-1 translation tables that the host builds for the guest: [`Stage1Tables`]
//! writes them, [`walk`] reads them back.
//!
//! Design: `docs/specs/execution-model.md` ("Guest memory", "Native backends: stage 1 is ours,
//! stage 2 is coarse"). The tables use a 4 KiB granule and a 39-bit address space translated
//! through `TTBR0_EL1` (`TCR_EL1.T0SZ` = 25, walk starting at level 1), as
//! `tools/hvprobe/src/probe.rs` verified on hardware. Descriptor bits are in [`descriptor`],
//! with their Arm DDI 0487 citations.
//!
//! `Stage1Tables` is plain data over a [`GuestMemory`] whose addresses are guest-physical: it
//! takes its table pages from a pool inside that memory and writes descriptors there. The
//! backing that pages map onto may lie anywhere below [`OUTPUT_ADDRESS_END`] except inside the
//! pool. It is shared by the Hypervisor.framework and KVM backends; neither uses it yet.
//! Each descriptor is written with one eight-byte [`GuestMemory::write`], and a new table is
//! linked only after it is zeroed; a backend whose tables are live under a running vCPU must
//! make that write single-copy atomic so the hardware walker never sees half a descriptor.
//!
//! # What the builder writes
//!
//! - Only table descriptors at levels 1 and 2 and page descriptors at level 3: no blocks, no
//!   contiguous hint, no hierarchical table attributes.
//! - Every page descriptor has the access flag set, is inner shareable with `AttrIndx` 0, is
//!   never executable at EL1 (`PXN`), and is non-global (`nG`), so its TLB entries are tagged
//!   with the process's ASID. hvprobe wrote global entries and EL1-executable code pages; `nG`
//!   and `PXN` on code pages are therefore not yet verified on hardware, only taken from the
//!   manual.
//! - EL0 permissions map to `AP[2:1]` and `UXN` as [`descriptor::page`] shows. An EL0-
//!   inaccessible page ([`Perms::None`], [`Perms::Execute`]) gets `AP[2:1]` = 0b00, read-write
//!   at EL1, as hvprobe's marker pages did. Execute-only ([`Perms::Execute`]) is from the
//!   manual and not verified on hardware.
//!
//! # Rules this builder adds
//!
//! - `map` needs every page of its range unmapped, `unmap` and `protect` need every page mapped;
//!   otherwise they fail without writing anything. Before writing, `map` also counts the table
//!   pages it will need and fails with [`MapError::OutOfTablePages`] if the pool is short.
//! - A table page whose last valid descriptor is removed is unlinked and returned to the pool at
//!   once, so the pages in use are always the root, one level-2 table per level-1 slot that
//!   holds a mapping, and one level-3 table per level-2 slot that does.
//! - Backing that overlaps the pool is refused ([`MapError::OverlapsTables`]): the spec keeps
//!   the tables in their own backing region, and mapping them would let the guest rewrite them.
//!
//! # TLB invalidation
//!
//! The host can't invalidate the guest's TLBs; the EL1 shim does (spec, "TLB invalidation").
//! The builder records what needs invalidating as [`FlushRange`]s, collected with
//! [`Stage1Tables::take_pending_flushes`]. Every successful `unmap` and `protect` records
//! exactly one range, its own. `map` records none: it only replaces invalid descriptors, and
//! Arm DDI 0487 (TLB maintenance requirements) does not let a TLB hold a translation table
//! entry that generates a Translation fault. Valid descriptors are never re-pointed to a
//! different output address, so no break-before-make sequence is needed.

pub(crate) mod descriptor;
mod walk;

pub use walk::{Resolved, WalkError, translate, walk};

use std::sync::{Mutex, PoisonError};

use crate::translation::{BackingRange, MapError, PAGE_SIZE, Perms, Translation, check_page_range};
use crate::{Fault, GuestMemory};
use descriptor::{DESCRIPTOR_LEN, LEAF_LEVEL, OUTPUT_ADDRESS_MASK, START_LEVEL, VALID};

/// [`PAGE_SIZE`] as a buffer length.
const PAGE_LEN: usize = 1 << descriptor::PAGE_SHIFT;

/// One past the highest translated virtual address: the space is 39 bits wide.
pub const VA_SPACE_END: u64 = 1 << 39;

/// One past the highest output address a descriptor holds: bits 47:12 (Arm DDI 0487, without
/// the 52-bit output address extensions).
pub const OUTPUT_ADDRESS_END: u64 = OUTPUT_ADDRESS_MASK + PAGE_SIZE;

/// The guest-physical address of a level-1 table: the base address the backend loads into
/// `TTBR0_EL1` (`BADDR`), next to the process's ASID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableRoot(u64);

impl TableRoot {
    /// A root at guest-physical `pa`, which must be page aligned and below
    /// [`OUTPUT_ADDRESS_END`].
    pub fn new(pa: u64) -> Result<Self, MapError> {
        if !pa.is_multiple_of(PAGE_SIZE) {
            return Err(MapError::MisalignedAddress { address: pa });
        }
        if pa >= OUTPUT_ADDRESS_END {
            return Err(MapError::PhysicalRange {
                pa,
                size: PAGE_SIZE,
            });
        }
        Ok(Self(pa))
    }

    /// The guest-physical address of the table.
    pub const fn pa(self) -> u64 {
        self.0
    }
}

/// A virtual range whose stale TLB entries the guest must invalidate before relying on a change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlushRange {
    /// The first virtual address of the range.
    pub va: u64,
    /// Its size in bytes, a multiple of [`PAGE_SIZE`].
    pub size: u64,
    /// Only page descriptors changed, so last-level invalidation (`TLBI VALE1IS`) is enough.
    /// `false` when table pages were also unlinked, which needs invalidation of cached walk
    /// entries too (`TLBI VAE1IS`).
    pub leaf_only: bool,
}

/// The table page pool and the pending invalidations, behind the lock.
#[derive(Debug)]
struct Pool {
    /// Freed table pages, reused last-freed first.
    free: Vec<u64>,
    /// The next never-used pool page.
    next: u64,
    /// Table pages in use, the root included.
    in_use: u64,
    pending: Vec<FlushRange>,
}

/// AArch64 stage-1 translation tables (4 KiB granule, 39-bit address space) written into
/// guest-physical memory.
#[derive(Debug)]
pub struct Stage1Tables<M> {
    memory: M,
    root: TableRoot,
    pool: BackingRange,
    state: Mutex<Pool>,
}

impl<M: GuestMemory> Stage1Tables<M> {
    /// Creates empty tables whose pages come from `pool`, a range of `memory`'s guest-physical
    /// addresses. The first pool page becomes the root.
    ///
    /// Zeroes the whole pool, so a pool `memory` doesn't cover fails here, with the
    /// [`MapError::Memory`] fault of the first page it refuses.
    pub fn new(memory: M, pool: BackingRange) -> Result<Self, MapError> {
        if pool.end() > OUTPUT_ADDRESS_END {
            return Err(MapError::PhysicalRange {
                pa: pool.base(),
                size: pool.size(),
            });
        }
        let tables = Self {
            memory,
            root: TableRoot::new(pool.base())?,
            pool,
            state: Mutex::new(Pool {
                free: Vec::new(),
                next: pool.base().saturating_add(PAGE_SIZE),
                in_use: 1,
                pending: Vec::new(),
            }),
        };
        let mut page = pool.base();
        while page < pool.end() {
            tables.zero(page)?;
            page = page.saturating_add(PAGE_SIZE);
        }
        Ok(tables)
    }

    /// The root table, for `TTBR0_EL1`.
    pub fn root(&self) -> TableRoot {
        self.root
    }

    /// The memory the tables live in.
    pub fn memory(&self) -> &M {
        &self.memory
    }

    /// How many pages the pool holds.
    pub fn pool_pages(&self) -> u64 {
        self.pool.size() / PAGE_SIZE
    }

    /// How many pool pages hold tables now, the root included.
    pub fn table_pages_in_use(&self) -> u64 {
        self.lock().in_use
    }

    /// Returns the ranges recorded since the last call, oldest first, and forgets them.
    pub fn take_pending_flushes(&self) -> Vec<FlushRange> {
        std::mem::take(&mut self.lock().pending)
    }

    /// Resolves `va` through these tables with [`walk`].
    pub fn walk(&self, va: u64) -> Result<Resolved, WalkError> {
        walk(&self.memory, self.root, va)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Pool> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn zero(&self, page: u64) -> Result<(), Fault> {
        const ZEROES: [u8; PAGE_LEN] = [0; PAGE_LEN];
        self.memory.write(page, &ZEROES)
    }

    /// Takes a pool page for a new table. It needs no zeroing: [`Self::new`] zeroed every pool
    /// page, the builder invalidates a descriptor by writing 0, and a table is freed only when
    /// none of its descriptors is valid, so a freed page holds only zeros.
    fn allocate(&self, pool: &mut Pool) -> Result<u64, MapError> {
        let page = if let Some(page) = pool.free.pop() {
            page
        } else if pool.next < self.pool.end() {
            let page = pool.next;
            pool.next = page.saturating_add(PAGE_SIZE);
            page
        } else {
            return Err(MapError::OutOfTablePages);
        };
        pool.in_use = pool.in_use.saturating_add(1);
        Ok(page)
    }

    /// The table a level-1 or level-2 descriptor points to, or `None` if it is invalid. A block,
    /// or a table outside the pool, is [`MapError::UnexpectedDescriptor`].
    fn child(&self, entry: u64, level: u8, va: u64) -> Result<Option<u64>, MapError> {
        if entry & VALID == 0 {
            return Ok(None);
        }
        let table = entry & OUTPUT_ADDRESS_MASK;
        if descriptor::is_table_or_page(entry)
            && entry & descriptor::TABLE_ATTRIBUTES == 0
            && table >= self.pool.base()
            && table < self.pool.end()
        {
            return Ok(Some(table));
        }
        Err(MapError::UnexpectedDescriptor {
            va,
            level,
            descriptor: entry,
        })
    }

    /// Whether a level-3 descriptor is a page (`Some(true)`), invalid (`Some(false)`), or
    /// something the builder never writes (`None`).
    fn leaf_is_page(entry: u64) -> Option<bool> {
        if entry & VALID == 0 {
            Some(false)
        } else if descriptor::is_table_or_page(entry) {
            Some(true)
        } else {
            None
        }
    }

    /// Counts the table pages mapping `start..end` would create under `table` at `level`, or
    /// fails with the lowest page of the range that is already mapped.
    fn tables_needed(&self, table: u64, level: u8, start: u64, end: u64) -> Result<u64, MapError> {
        let mut needed = 0u64;
        for (s, e) in entries(start, end, level) {
            let entry = descriptor::read(&self.memory, table, descriptor::index(s, level))?;
            if level == LEAF_LEVEL {
                match Self::leaf_is_page(entry) {
                    Some(false) => {}
                    Some(true) => return Err(MapError::AlreadyMapped { va: s }),
                    None => return Err(unexpected(s, level, entry)),
                }
                continue;
            }
            let below = match self.child(entry, level, s)? {
                Some(child) => self.tables_needed(child, level + 1, s, e)?,
                None => new_tables_below(level, s, e),
            };
            needed = needed.saturating_add(below);
        }
        Ok(needed)
    }

    /// Fails with the lowest page of `start..end` under `table` that is not mapped.
    fn check_mapped(&self, table: u64, level: u8, start: u64, end: u64) -> Result<(), MapError> {
        for (s, e) in entries(start, end, level) {
            let entry = descriptor::read(&self.memory, table, descriptor::index(s, level))?;
            if level == LEAF_LEVEL {
                match Self::leaf_is_page(entry) {
                    Some(true) => {}
                    Some(false) => return Err(MapError::NotMapped { va: s }),
                    None => return Err(unexpected(s, level, entry)),
                }
                continue;
            }
            match self.child(entry, level, s)? {
                Some(child) => self.check_mapped(child, level + 1, s, e)?,
                None => return Err(MapError::NotMapped { va: s }),
            }
        }
        Ok(())
    }

    /// Writes page descriptors for `start..end` under `table`, creating tables as needed.
    /// `va` and `backing` give the virtual-to-physical offset.
    fn write_map(
        &self,
        pool: &mut Pool,
        table: u64,
        level: u8,
        (start, end): (u64, u64),
        (va, backing): (u64, BackingRange),
        perms: Perms,
    ) -> Result<(), MapError> {
        for (s, e) in entries(start, end, level) {
            let index = descriptor::index(s, level);
            if level == LEAF_LEVEL {
                let pa = s
                    .checked_sub(va)
                    .and_then(|offset| backing.base().checked_add(offset))
                    .ok_or(MapError::PhysicalRange {
                        pa: backing.base(),
                        size: backing.size(),
                    })?;
                descriptor::write(&self.memory, table, index, descriptor::page(pa, perms))?;
                continue;
            }
            let entry = descriptor::read(&self.memory, table, index)?;
            let child = if let Some(child) = self.child(entry, level, s)? {
                child
            } else {
                let child = self.allocate(pool)?;
                descriptor::write(&self.memory, table, index, descriptor::table(child))?;
                child
            };
            self.write_map(pool, child, level + 1, (s, e), (va, backing), perms)?;
        }
        Ok(())
    }

    /// Invalidates the page descriptors for `start..end` under `table`, unlinking and freeing
    /// tables left empty. Returns whether any table was freed.
    fn write_unmap(
        &self,
        pool: &mut Pool,
        table: u64,
        level: u8,
        start: u64,
        end: u64,
    ) -> Result<bool, MapError> {
        let mut freed = false;
        for (s, e) in entries(start, end, level) {
            let index = descriptor::index(s, level);
            if level == LEAF_LEVEL {
                descriptor::write(&self.memory, table, index, 0)?;
                continue;
            }
            let entry = descriptor::read(&self.memory, table, index)?;
            let child = self
                .child(entry, level, s)?
                .ok_or(MapError::NotMapped { va: s })?;
            freed |= self.write_unmap(pool, child, level + 1, s, e)?;
            if self.is_empty(child)? {
                descriptor::write(&self.memory, table, index, 0)?;
                pool.free.push(child);
                pool.in_use = pool.in_use.saturating_sub(1);
                freed = true;
            }
        }
        Ok(freed)
    }

    /// Rewrites the page descriptors for `start..end` under `table` with `perms`.
    fn write_protect(
        &self,
        table: u64,
        level: u8,
        start: u64,
        end: u64,
        perms: Perms,
    ) -> Result<(), MapError> {
        for (s, e) in entries(start, end, level) {
            let index = descriptor::index(s, level);
            let entry = descriptor::read(&self.memory, table, index)?;
            if level == LEAF_LEVEL {
                let page = descriptor::page(entry & OUTPUT_ADDRESS_MASK, perms);
                descriptor::write(&self.memory, table, index, page)?;
                continue;
            }
            let child = self
                .child(entry, level, s)?
                .ok_or(MapError::NotMapped { va: s })?;
            self.write_protect(child, level + 1, s, e, perms)?;
        }
        Ok(())
    }

    /// Whether no descriptor in the table at `table` is valid.
    fn is_empty(&self, table: u64) -> Result<bool, Fault> {
        let mut bytes = [0; PAGE_LEN];
        self.memory.read(table, &mut bytes)?;
        let (words, _) = bytes.as_chunks::<DESCRIPTOR_LEN>();
        Ok(words
            .iter()
            .all(|&word| u64::from_le_bytes(word) & VALID == 0))
    }
}

impl<M: GuestMemory> Translation for Stage1Tables<M> {
    fn map(&self, va: u64, backing: BackingRange, perms: Perms) -> Result<(), MapError> {
        let end = virtual_end(va, backing.size())?;
        if backing.end() > OUTPUT_ADDRESS_END {
            return Err(MapError::PhysicalRange {
                pa: backing.base(),
                size: backing.size(),
            });
        }
        if backing.overlaps(self.pool) {
            return Err(MapError::OverlapsTables {
                pa: backing.base(),
                size: backing.size(),
            });
        }
        let mut pool = self.lock();
        let root = self.root.pa();
        let needed = self.tables_needed(root, START_LEVEL, va, end)?;
        if needed > self.pool_pages().saturating_sub(pool.in_use) {
            return Err(MapError::OutOfTablePages);
        }
        self.write_map(
            &mut pool,
            root,
            START_LEVEL,
            (va, end),
            (va, backing),
            perms,
        )
    }

    fn unmap(&self, va: u64, size: u64) -> Result<(), MapError> {
        let end = virtual_end(va, size)?;
        let mut pool = self.lock();
        let root = self.root.pa();
        self.check_mapped(root, START_LEVEL, va, end)?;
        let freed = self.write_unmap(&mut pool, root, START_LEVEL, va, end)?;
        pool.pending.push(FlushRange {
            va,
            size,
            leaf_only: !freed,
        });
        Ok(())
    }

    fn protect(&self, va: u64, size: u64, perms: Perms) -> Result<(), MapError> {
        let end = virtual_end(va, size)?;
        let mut pool = self.lock();
        let root = self.root.pa();
        self.check_mapped(root, START_LEVEL, va, end)?;
        self.write_protect(root, START_LEVEL, va, end, perms)?;
        pool.pending.push(FlushRange {
            va,
            size,
            leaf_only: true,
        });
        Ok(())
    }
}

/// Checks a virtual range (alignment, non-empty, inside the 39-bit space) and returns its end.
fn virtual_end(va: u64, size: u64) -> Result<u64, MapError> {
    check_page_range(va, size)?;
    va.checked_add(size)
        .filter(|&end| end <= VA_SPACE_END)
        .ok_or(MapError::VirtualRange { va, size })
}

fn unexpected(va: u64, level: u8, descriptor: u64) -> MapError {
    MapError::UnexpectedDescriptor {
        va,
        level,
        descriptor,
    }
}

/// Splits `start..end` into the pieces that fall under one descriptor each at `level`.
fn entries(start: u64, end: u64, level: u8) -> impl Iterator<Item = (u64, u64)> {
    let span = 1u64 << descriptor::shift(level);
    let mut at = start;
    std::iter::from_fn(move || {
        if at >= end {
            return None;
        }
        let next = (at & !(span - 1)).saturating_add(span).min(end);
        let piece = (at, next);
        at = next;
        Some(piece)
    })
}

/// How many tables mapping `start..end` creates beneath an invalid descriptor at `level`.
fn new_tables_below(level: u8, start: u64, end: u64) -> u64 {
    let child = level + 1;
    if child == LEAF_LEVEL {
        return 1;
    }
    entries(start, end, child).fold(1, |count, (s, e)| {
        count.saturating_add(new_tables_below(child, s, e))
    })
}

#[cfg(test)]
mod tests;
