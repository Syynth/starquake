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
//!   pages it will need and fails with [`MapError::OutOfTablePages`] if the pool lacks that many
//!   reusable pages (see "TLB invalidation" for which freed pages are not yet reusable).
//! - A table page whose last valid descriptor is removed is unlinked at once, so the pages in use
//!   ([`Stage1Tables::table_pages_in_use`]) are always the root, one level-2 table per level-1
//!   slot that holds a mapping, and one level-3 table per level-2 slot that does. The page itself
//!   returns to the pool for reuse when [`Stage1Tables::take_pending_flushes`] hands out the
//!   flush that covers it.
//! - A table descriptor is followed only if it is a plain table descriptor (no hierarchical
//!   attribute bits) for a page this builder allocated as a table of the next level. A block, a
//!   reserved level-3 encoding, or a table descriptor aimed at anything else (the root, a free
//!   page, a page outside the pool, a table of another level) is
//!   [`MapError::UnexpectedDescriptor`]; the tables are corrupt and the builder will not follow
//!   or free what it did not allocate.
//! - Backing that overlaps the pool is refused ([`MapError::OverlapsTables`]): the spec keeps
//!   the tables in their own backing region, and mapping them would let the guest rewrite them.
//!
//! # TLB invalidation
//!
//! The host can't invalidate the guest's TLBs; the EL1 shim does (spec, "TLB invalidation").
//! The builder records what needs invalidating as [`FlushRange`]s, collected with
//! [`Stage1Tables::take_pending_flushes`]. Every successful `unmap` and `protect` records
//! exactly one range, its own; so does an `unmap` or `protect` that fails with a
//! [`MapError::Memory`] fault after it began writing (see [`MapError`]), conservatively
//! `leaf_only: false` for `unmap`. `map` records none. That relies on two things:
//!
//! - Arm DDI 0487 (TLB maintenance requirements) does not let a TLB hold a translation table
//!   entry that generates a Translation fault, and `map` only replaces invalid descriptors.
//!   Valid descriptors are never re-pointed to a different output address, so no
//!   break-before-make sequence is needed.
//! - A table page that `unmap` unlinked may still be named by a cached intermediate (walk)
//!   entry on another vCPU until that unmap's flush has run, so it is quarantined: it is not
//!   reused for a new table until [`Stage1Tables::take_pending_flushes`] has handed out the
//!   flushes recorded so far.
//!
//! The contract on the caller: flushes returned by `take_pending_flushes` must be executed
//! (`TLBI` and the barriers, on every vCPU that may use the tables) before the next `map`,
//! `unmap` or `protect`. Taking a flush is what lets the builder reuse the pages it freed.

pub(crate) mod descriptor;
mod walk;

pub use walk::{Resolved, WalkError, translate, walk};

use std::collections::BTreeMap;
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
    /// entries too (`TLBI VAE1IS`). A range with `leaf_only` `false` must be executed before any
    /// later `map`, because the builder reuses the unlinked table pages once the range has been
    /// taken with [`Stage1Tables::take_pending_flushes`].
    pub leaf_only: bool,
}

/// The table page pool and the pending invalidations, behind the lock.
#[derive(Debug)]
struct Pool {
    /// Freed table pages that may be reused, last-freed first.
    free: Vec<u64>,
    /// Table pages unlinked since the last [`Stage1Tables::take_pending_flushes`]: not reusable
    /// until their flush has been handed out.
    quarantine: Vec<u64>,
    /// The next never-used pool page.
    next: u64,
    /// The table pages in use (linked, plus the root) and the level of each; the root is at
    /// [`START_LEVEL`]. The builder follows a table descriptor only to a page listed here.
    tables: BTreeMap<u64, u8>,
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
                quarantine: Vec::new(),
                next: pool.base().saturating_add(PAGE_SIZE),
                tables: BTreeMap::from([(pool.base(), START_LEVEL)]),
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
        u64::try_from(self.lock().tables.len()).unwrap_or(u64::MAX)
    }

    /// Returns the ranges recorded since the last call, oldest first, and forgets them.
    ///
    /// The caller must execute the returned flushes before the next `map`, `unmap` or `protect`:
    /// this call releases the table pages that `unmap` unlinked for reuse, and a vCPU that has
    /// not yet invalidated may still reach them through a cached walk entry.
    pub fn take_pending_flushes(&self) -> Vec<FlushRange> {
        let mut pool = self.lock();
        let pool = &mut *pool;
        pool.free.append(&mut pool.quarantine);
        std::mem::take(&mut pool.pending)
    }

    /// Offers the ranges recorded since the last flush to `execute`, oldest first, and, only if
    /// it succeeds, forgets them and releases the table pages `unmap` unlinked for reuse.
    ///
    /// This is how a backend whose tables are live runs a flush: `execute` performs the
    /// invalidations (on Hypervisor.framework, the EL1 shim's flush entry point). The tables
    /// stay locked while it runs, so no `map`, `unmap` or `protect` from any thread comes
    /// between handing out the ranges and executing them, and a freed table page is never
    /// reused before the flush that covers it has run. If `execute` fails, its error is
    /// returned and nothing is forgotten or released: the next call offers the same ranges
    /// again.
    ///
    /// `execute` is not called when nothing is pending. It must not call back into these
    /// tables: they are locked, and it would wait for itself.
    pub fn flush_pending_with<E>(
        &self,
        execute: impl FnOnce(&[FlushRange]) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut pool = self.lock();
        // Table pages are only unlinked by an `unmap`, which always records a range, so nothing
        // pending means nothing quarantined either.
        if pool.pending.is_empty() {
            return Ok(());
        }
        execute(&pool.pending)?;
        let pool = &mut *pool;
        pool.pending.clear();
        pool.free.append(&mut pool.quarantine);
        Ok(())
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

    /// Takes a pool page for a new table at `level`. It needs no zeroing: [`Self::new`] zeroed
    /// every pool page, the builder invalidates a descriptor by writing 0, and a table is freed
    /// only when none of its descriptors is valid, so a freed page holds only zeros.
    fn allocate(&self, pool: &mut Pool, level: u8) -> Result<u64, MapError> {
        let page = if let Some(page) = pool.free.pop() {
            page
        } else if pool.next < self.pool.end() {
            let page = pool.next;
            pool.next = page.saturating_add(PAGE_SIZE);
            page
        } else {
            return Err(MapError::OutOfTablePages);
        };
        pool.tables.insert(page, level);
        Ok(page)
    }

    /// How many pages a new table could take now: the free list plus the never-used pages.
    /// Pages unlinked since the last [`Self::take_pending_flushes`] are not counted.
    fn reusable_pages(&self, pool: &Pool) -> u64 {
        let never_used = self.pool.end().saturating_sub(pool.next) / PAGE_SIZE;
        never_used.saturating_add(u64::try_from(pool.free.len()).unwrap_or(u64::MAX))
    }

    /// The table a level-1 or level-2 descriptor points to, or `None` if it is invalid.
    ///
    /// [`MapError::UnexpectedDescriptor`] for a block; a table descriptor with a hierarchical
    /// attribute bit (63:59) set; and a table descriptor whose target is not a page this builder
    /// allocated as a table of the next level, which covers a page below or above the pool, the
    /// root, a free or never-allocated pool page, and a table of another level.
    fn child(pool: &Pool, entry: u64, level: u8, va: u64) -> Result<Option<u64>, MapError> {
        if entry & VALID == 0 {
            return Ok(None);
        }
        let table = entry & OUTPUT_ADDRESS_MASK;
        if descriptor::is_table_or_page(entry)
            && entry & descriptor::TABLE_ATTRIBUTES == 0
            && pool.tables.get(&table) == Some(&(level + 1))
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
    fn tables_needed(
        &self,
        pool: &Pool,
        table: u64,
        level: u8,
        start: u64,
        end: u64,
    ) -> Result<u64, MapError> {
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
            let below = match Self::child(pool, entry, level, s)? {
                Some(child) => self.tables_needed(pool, child, level + 1, s, e)?,
                None => new_tables_below(level, s, e),
            };
            needed = needed.saturating_add(below);
        }
        Ok(needed)
    }

    /// Fails with the lowest page of `start..end` under `table` that is not mapped.
    fn check_mapped(
        &self,
        pool: &Pool,
        table: u64,
        level: u8,
        start: u64,
        end: u64,
    ) -> Result<(), MapError> {
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
            match Self::child(pool, entry, level, s)? {
                Some(child) => self.check_mapped(pool, child, level + 1, s, e)?,
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
            let child = if let Some(child) = Self::child(pool, entry, level, s)? {
                child
            } else {
                let child = self.allocate(pool, level + 1)?;
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
            let child = Self::child(pool, entry, level, s)?.ok_or(MapError::NotMapped { va: s })?;
            freed |= self.write_unmap(pool, child, level + 1, s, e)?;
            if self.is_empty(child)? {
                descriptor::write(&self.memory, table, index, 0)?;
                pool.tables.remove(&child);
                pool.quarantine.push(child);
                freed = true;
            }
        }
        Ok(freed)
    }

    /// Rewrites the page descriptors for `start..end` under `table` with `perms`.
    fn write_protect(
        &self,
        pool: &Pool,
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
            let child = Self::child(pool, entry, level, s)?.ok_or(MapError::NotMapped { va: s })?;
            self.write_protect(pool, child, level + 1, s, e, perms)?;
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
        let needed = self.tables_needed(&pool, root, START_LEVEL, va, end)?;
        if needed > self.reusable_pages(&pool) {
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
        self.check_mapped(&pool, root, START_LEVEL, va, end)?;
        let written = self.write_unmap(&mut pool, root, START_LEVEL, va, end);
        // A flush is owed once the write phase began, also when a write fault ended it early:
        // descriptors may have changed and tables may have been unlinked. After a fault that
        // is not known, so it is conservatively not leaf-only.
        pool.pending.push(FlushRange {
            va,
            size,
            leaf_only: matches!(written, Ok(false)),
        });
        written.map(|_| ())
    }

    fn protect(&self, va: u64, size: u64, perms: Perms) -> Result<(), MapError> {
        let end = virtual_end(va, size)?;
        let mut pool = self.lock();
        let root = self.root.pa();
        self.check_mapped(&pool, root, START_LEVEL, va, end)?;
        let written = self.write_protect(&pool, root, START_LEVEL, va, end, perms);
        // As in `unmap`: a flush is owed once the write phase began, even if a fault ended it.
        pool.pending.push(FlushRange {
            va,
            size,
            leaf_only: true,
        });
        written
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
