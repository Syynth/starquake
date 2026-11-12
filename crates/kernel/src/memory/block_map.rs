//! The sorted map of memory blocks that covers an address space, with Horizon's splitting and
//! merging.
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_memory_block_manager.cpp` @ cf654052
//! (GPL-3.0-or-later): `Initialize`, `FindFreeArea`, `Update`, `UpdateAttribute`,
//! `CoalesceForUpdate` and `CheckState`, and `FindBlock` from `k_memory_block_manager.h`.
//!
//! Not ported yet, because nothing in this slice calls them: `UpdateLock` (IPC and device
//! locks) and `UpdateIfMatch` (`UnmapPhysicalMemory`).
//!
//! Divergences from Eden:
//!
//! - Eden keeps blocks in an intrusive red-black tree and takes new ones from a slab, so an
//!   update can run out of blocks (`ResultOutOfResource`). Here the blocks are in a
//!   [`BTreeMap`] keyed by their first address and the host's allocator supplies them, so that
//!   failure does not exist (kernel spec: Horizon's slab heaps are replaced, not ported).
//! - Eden asserts its preconditions. Here a range that is not whole pages inside the map, or an
//!   update of a block that is locked for IPC, is a [`BlockMapError`] and changes nothing
//!   (tests: `a_range_outside_the_map_or_off_a_page_is_refused_and_changes_nothing`,
//!   `an_update_over_a_block_locked_for_ipc_is_refused_and_changes_nothing`). A caller sees the
//!   error before it has touched the translation by asking [`BlockMap::check_update`] first.

use std::collections::BTreeMap;

use starquake_memory::PAGE_SIZE;

use super::block::MemoryBlock;
use super::state::{DisableMerge, MemoryAttribute, MemoryPermission, MemoryState};

/// Why a block map operation was refused. No guest input reaches the map without passing the
/// address space's own checks first, so every one of these is a bug in the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BlockMapError {
    /// The map's bounds are not page aligned, or it would be empty.
    #[error("a block map cannot cover {start:#x}..{end:#x}")]
    BadBounds {
        /// The first address.
        start: u64,
        /// One past the last address.
        end: u64,
    },
    /// A range that is empty, not page aligned, or not inside the map.
    #[error("{pages:#x} pages at {address:#x} are not a run of pages inside the block map")]
    BadRange {
        /// The first address of the range.
        address: u64,
        /// Its length in pages.
        pages: u64,
    },
    /// An update would change a block that is locked for IPC.
    #[error("the block at {address:#x} is locked for IPC and cannot be updated")]
    Locked {
        /// The first address of the block.
        address: u64,
    },
    /// An update was given a lock or share attribute, which only counts can set.
    #[error("an update cannot set the IPC-locked or device-shared attribute")]
    CountedAttribute,
    /// The map has a hole or an overlap: its blocks do not tile it.
    #[error("the block map is broken at {address:#x}")]
    Broken {
        /// Where a block was expected and not found.
        address: u64,
    },
}

/// The memory blocks of one address space (`KMemoryBlockManager`): they tile it from its first
/// address to its last without gaps, and no two neighbours could be merged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockMap {
    start: u64,
    end: u64,
    /// The blocks by their first address.
    blocks: BTreeMap<u64, MemoryBlock>,
}

impl BlockMap {
    /// A map of `start..end` that is one free block (`Initialize`). Both bounds must be page
    /// aligned, with `start` below `end`.
    pub fn new(start: u64, end: u64) -> Result<Self, BlockMapError> {
        let bad = BlockMapError::BadBounds { start, end };
        if start >= end || !end.is_multiple_of(PAGE_SIZE) {
            return Err(bad);
        }
        let block = MemoryBlock::new(
            start,
            (end - start) / PAGE_SIZE,
            MemoryState::Free,
            MemoryPermission::NONE,
            MemoryAttribute::NONE,
        )
        .ok_or(bad)?;
        Ok(Self {
            start,
            end,
            blocks: BTreeMap::from([(start, block)]),
        })
    }

    /// The first address the map covers.
    #[must_use]
    pub const fn start(&self) -> u64 {
        self.start
    }

    /// One past the last address the map covers.
    #[must_use]
    pub const fn end(&self) -> u64 {
        self.end
    }

    /// The block that holds `address` (`FindBlock`), or `None` outside the map.
    #[must_use]
    pub fn find(&self, address: u64) -> Option<&MemoryBlock> {
        self.blocks
            .range(..=address)
            .next_back()
            .map(|(_, block)| block)
            .filter(|block| block.contains(address))
    }

    /// The blocks from the one that holds `address` upwards, in address order.
    pub fn blocks_from(&self, address: u64) -> impl Iterator<Item = &MemoryBlock> {
        let first = self.find(address).map_or(self.end, MemoryBlock::address);
        self.blocks.range(first..).map(|(_, block)| block)
    }

    /// Every block, in address order.
    pub fn blocks(&self) -> impl Iterator<Item = &MemoryBlock> {
        self.blocks.values()
    }

    /// The end of the range of `pages` pages at `address`, if it is a non-empty run of whole
    /// pages inside the map.
    fn range_end(&self, address: u64, pages: u64) -> Result<u64, BlockMapError> {
        let bad = BlockMapError::BadRange { address, pages };
        if pages == 0 || !address.is_multiple_of(PAGE_SIZE) || address < self.start {
            return Err(bad);
        }
        pages
            .checked_mul(PAGE_SIZE)
            .and_then(|size| address.checked_add(size))
            .filter(|end| *end <= self.end)
            .ok_or(bad)
    }

    fn block_at(&self, address: u64) -> Result<MemoryBlock, BlockMapError> {
        self.find(address)
            .copied()
            .ok_or(BlockMapError::Broken { address })
    }

    /// Whether [`Self::update`] of this range to these properties would be refused: the checks
    /// it makes before it changes anything, for a caller that has a side effect of its own to
    /// make first.
    pub fn check_update(
        &self,
        address: u64,
        pages: u64,
        state: MemoryState,
        permission: MemoryPermission,
        attribute: MemoryAttribute,
    ) -> Result<(), BlockMapError> {
        let end = self.range_end(address, pages)?;
        if !attribute.excludes(MemoryAttribute::COUNTED) {
            return Err(BlockMapError::CountedAttribute);
        }
        for block in self.blocks_from(address) {
            if block.address() >= end {
                break;
            }
            if !block.has_properties(state, permission, attribute) && !block.can_update() {
                return Err(BlockMapError::Locked {
                    address: block.address(),
                });
            }
        }
        Ok(())
    }

    /// Makes the block that holds `cursor` start there and end at or before `end`, splitting
    /// it as needed, and returns its first address (`cursor`).
    fn isolate(&mut self, cursor: u64, end: u64) -> Result<u64, BlockMapError> {
        let broken = BlockMapError::Broken { address: cursor };
        let mut block = self.block_at(cursor)?;
        // If we need to, create a new block before and insert it.
        if block.address() != cursor {
            let lower = block.split(cursor).ok_or(broken)?;
            self.blocks.insert(lower.address(), lower);
        }
        // If we need to, create a new block after and insert it.
        if block.last() >= end {
            let upper_start = end;
            let mut upper = block;
            block = upper.split(upper_start).ok_or(broken)?;
            self.blocks.insert(upper.address(), upper);
        }
        self.blocks.insert(block.address(), block);
        Ok(block.address())
    }

    /// Gives `pages` pages at `address` a state, a permission and attributes (`Update`),
    /// splitting the blocks at the ends of the range and merging what can be merged afterwards.
    ///
    /// Blocks that already have the three are left alone. `set` is added to the merge barriers
    /// of the block that starts at `address`, if that block is changed; `clear` is removed from
    /// every block that is changed.
    ///
    /// Refused, with nothing changed, for a range that is not whole pages inside the map, for
    /// an attribute that only counts set, and when a block that would change is locked for IPC.
    #[expect(
        clippy::too_many_arguments,
        reason = "the seven arguments are KMemoryBlockManager::Update's, in its order"
    )]
    pub fn update(
        &mut self,
        address: u64,
        pages: u64,
        state: MemoryState,
        permission: MemoryPermission,
        attribute: MemoryAttribute,
        set: DisableMerge,
        clear: DisableMerge,
    ) -> Result<(), BlockMapError> {
        self.check_update(address, pages, state, permission, attribute)?;
        let end = self.range_end(address, pages)?;
        let mut cursor = address;
        while cursor < end {
            let block = self.block_at(cursor)?;
            if block.has_properties(state, permission, attribute) {
                // If we already have the right properties, just advance.
                cursor = block.end().map_or(end, |block_end| block_end.min(end));
                continue;
            }
            let at = self.isolate(cursor, end)?;
            let block = self
                .blocks
                .get_mut(&at)
                .ok_or(BlockMapError::Broken { address: at })?;
            block.update(state, permission, attribute, at == address, set, clear);
            cursor = block.end().unwrap_or(end);
        }
        self.coalesce(address, end)
    }

    /// Replaces the attributes in `mask` with `attribute` on `pages` pages at `address`
    /// (`UpdateAttribute`), splitting and merging as [`Self::update`] does. Blocks whose masked
    /// attributes already equal `attribute` are left alone.
    ///
    /// Refused, with nothing changed, for a range that is not whole pages inside the map, and
    /// for a mask or attribute with a bit that only counts set.
    pub fn update_attribute(
        &mut self,
        address: u64,
        pages: u64,
        mask: MemoryAttribute,
        attribute: MemoryAttribute,
    ) -> Result<(), BlockMapError> {
        let end = self.range_end(address, pages)?;
        if !(mask | attribute).excludes(MemoryAttribute::COUNTED) {
            return Err(BlockMapError::CountedAttribute);
        }
        let mut cursor = address;
        while cursor < end {
            let block = self.block_at(cursor)?;
            if block.attribute() & mask == attribute {
                // If we already have the right attributes, just advance.
                cursor = block.end().map_or(end, |block_end| block_end.min(end));
                continue;
            }
            let at = self.isolate(cursor, end)?;
            let block = self
                .blocks
                .get_mut(&at)
                .ok_or(BlockMapError::Broken { address: at })?;
            block.update_attribute(mask, attribute);
            cursor = block.end().unwrap_or(end);
        }
        self.coalesce(address, end)
    }

    /// Merges the blocks around `address..end` that can be merged (`CoalesceForUpdate`): from
    /// the block below the range to the first block that ends above it.
    fn coalesce(&mut self, address: u64, end: u64) -> Result<(), BlockMapError> {
        // Find the block the range starts in, and start one block before it if there is one.
        let mut current = self.block_at(address)?.address();
        if address != self.start
            && let Some((below, _)) = self.blocks.range(..current).next_back()
        {
            current = *below;
        }
        // Coalesce blocks that we can. Each round removes a block or moves up one, so the loop
        // ends at the top of the map at the latest.
        loop {
            let below = self.block_at(current)?;
            let Some(above) = below.end().and_then(|at| self.blocks.get(&at)).copied() else {
                break;
            };
            if below.can_merge_with(&above) {
                self.blocks.remove(&above.address());
                let merged = self
                    .blocks
                    .get_mut(&current)
                    .ok_or(BlockMapError::Broken { address: current })?;
                merged.add(&above);
            } else {
                current = above.address();
            }
            if self.block_at(current)?.end().is_none_or(|at| end < at) {
                break;
            }
        }
        Ok(())
    }

    /// The lowest address in the region of `region_pages` pages at `region_start` where
    /// `pages` free pages fit with `guard_pages` free pages on either side, at `offset` past a
    /// multiple of `alignment` (`FindFreeArea`). `None` when there is no such place, or when an
    /// address in the search overflows.
    ///
    /// `alignment` must be a power of two.
    #[must_use]
    pub fn find_free_area(
        &self,
        region_start: u64,
        region_pages: u64,
        pages: u64,
        alignment: u64,
        offset: u64,
        guard_pages: u64,
    ) -> Option<u64> {
        if pages == 0 || !alignment.is_power_of_two() {
            return None;
        }
        let region_last = region_pages
            .checked_mul(PAGE_SIZE)
            .and_then(|size| region_start.checked_add(size))?
            .checked_sub(1)?;
        let guard = guard_pages.checked_mul(PAGE_SIZE)?;
        let size = pages.checked_mul(PAGE_SIZE)?;
        for block in self.blocks_from(region_start) {
            if region_last < block.address() {
                break;
            }
            if block.state() != MemoryState::Free {
                continue;
            }
            let fits = || {
                let area = block.address().max(region_start).checked_add(guard)?;
                let offset_area = (area & !(alignment - 1)).checked_add(offset)?;
                let area = if area <= offset_area {
                    offset_area
                } else {
                    offset_area.checked_add(alignment)?
                };
                let area_last = area.checked_add(size)?.checked_add(guard)?.checked_sub(1)?;
                (block.address() <= area
                    && area < area_last
                    && area_last <= region_last
                    && area_last <= block.last())
                .then_some(area)
            };
            if let Some(area) = fits() {
                return Some(area);
            }
        }
        None
    }

    /// Whether the map is whole (`CheckState`, plus the tiling Eden's tree keeps by
    /// construction): the blocks tile `start..end` in order, no two neighbours could be merged,
    /// and a block with a lock or share attribute has the count behind it.
    #[must_use]
    pub fn check_state(&self) -> bool {
        let mut expected = Some(self.start);
        let mut below: Option<&MemoryBlock> = None;
        for (key, block) in &self.blocks {
            if Some(*key) != expected || block.address() != *key {
                return false;
            }
            if below.is_some_and(|below| below.can_merge_with(block)) {
                return false;
            }
            let attribute = block.attribute();
            if attribute.contains(MemoryAttribute::IPC_LOCKED) && block.ipc_lock_count() == 0 {
                return false;
            }
            if attribute.contains(MemoryAttribute::DEVICE_SHARED) && block.device_use_count() == 0 {
                return false;
            }
            expected = block.end();
            below = Some(block);
        }
        expected == Some(self.end)
    }
}

#[cfg(test)]
mod tests;
