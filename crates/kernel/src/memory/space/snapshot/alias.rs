//! The `MapMemory` aliases in an address space snapshot: each run of alias pages the address
//! space records (`memory::space::alias`), saved as where it starts, the first source page it
//! maps again and its size ([`SavedAlias`]); checked against what `MapMemory` and `UnmapMemory`
//! enforce (check 8 of the parent module's docs, [`check`]); and mapped again through
//! [`Mapper::map_alias`](crate::memory::Mapper::map_alias) after every other mapping (check 10).
//!
//! The parent module's docs say where the aliases sit in the format, why each relation holds
//! live, and what is not checked.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use starquake_memory::PAGE_SIZE;

use super::super::alias::{ALIAS_DESTINATION_CHECK, Aliases, LOCKED_SOURCE_CHECK};
use super::super::{Check, can_contain};
use super::{SnapshotError, blocks_touching};
use crate::memory::block::MemoryBlock;
use crate::memory::layout::{Region, Regions};
use crate::memory::state::MemoryState;

/// One run of alias pages as saved: where it starts (the destination), the first source page
/// it maps again, and its size in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct SavedAlias {
    pub(super) address: u64,
    pub(super) source: u64,
    pub(super) size: u64,
}

/// What went wrong with one alias ([`SnapshotError::Alias`]), in the order [`check`] tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AliasFault {
    /// Its destination, source or size is not page aligned, its size is zero, or its
    /// destination or source is not inside the address space (what `MapMemory`'s page-table
    /// half refuses with `InvalidCurrentMemory` before it looks at a block).
    #[error("is not whole pages inside the address space")]
    Shape,
    /// Its destination starts before the end of the one before it: the record keeps its runs
    /// in address order, and never two over one page.
    #[error("is out of order or overlaps the one before it")]
    Order,
    /// Its destination is not where the stack region can hold `Stack` memory (what `MapMemory`
    /// refuses with `InvalidMemoryRegion`).
    #[error("is outside the stack region")]
    Region,
    /// Its source and its destination share a page.
    #[error("has a source that overlaps its destination")]
    Overlap,
    /// A block of its source is not what `MapMemory` leaves a source: a state that can be
    /// aliased, no access for the guest, and locked, every block alike (`UnmapMemory`'s check
    /// of the source).
    #[error("has a source that is not memory MapMemory locked")]
    Source,
    /// A block of its destination is not what `MapMemory` leaves a destination: `Stack`
    /// memory, read-write, with no attribute but `DeviceShared`.
    #[error("has a destination that is not the stack memory MapMemory makes")]
    Destination,
    /// Its source shares a page with the source of an alias before it in the list.
    #[error("has a source that an alias before it maps too")]
    SourceTaken,
}

impl SavedAlias {
    /// One past the destination's last byte. Saturates; [`check`] refuses a range that wraps
    /// before anything uses it.
    const fn end(&self) -> u64 {
        self.address.saturating_add(self.size)
    }

    /// One past the source's last byte, likewise.
    const fn source_end(&self) -> u64 {
        self.source.saturating_add(self.size)
    }
}

/// The aliases `aliases` records, as saved: in address order.
pub(super) fn saved(aliases: &Aliases) -> Vec<SavedAlias> {
    aliases
        .iter()
        .map(|(address, source, size)| SavedAlias {
            address,
            source,
            size,
        })
        .collect()
}

/// The record a restore keeps for `saved`, which [`check`] passed.
pub(super) fn restored(saved: &[SavedAlias]) -> Aliases {
    let mut aliases = Aliases::default();
    for alias in saved {
        aliases.insert(alias.address, alias.source, alias.size);
    }
    aliases
}

/// Check 8 of the parent module's docs: each alias in turn, and the first failing check of the
/// first failing alias ([`AliasFault`], in its order). `blocks` passed check 4: they tile the
/// address space in address order.
pub(super) fn check(
    aliases: &[SavedAlias],
    regions: &Regions,
    blocks: &[MemoryBlock],
) -> Result<(), SnapshotError> {
    let mut previous_end = None;
    // The sources of the aliases checked so far, start to end. They are disjoint, so one
    // neighbour on each side decides whether a new source overlaps any of them.
    let mut sources: BTreeMap<u64, u64> = BTreeMap::new();
    for (index, alias) in aliases.iter().enumerate() {
        check_one(alias, previous_end, regions, blocks, &sources)
            .map_err(|fault| SnapshotError::Alias { index, fault })?;
        previous_end = Some(alias.end());
        sources.insert(alias.source, alias.source_end());
    }
    Ok(())
}

/// The checks of one alias, in [`AliasFault`]'s order.
fn check_one(
    alias: &SavedAlias,
    previous_end: Option<u64>,
    regions: &Regions,
    blocks: &[MemoryBlock],
    sources: &BTreeMap<u64, u64>,
) -> Result<(), AliasFault> {
    let &SavedAlias {
        address,
        source,
        size,
    } = alias;
    // `Contains` of both ranges, as `MapMemory`'s page-table half checks them
    // (`AddressSpace::alias_pages`): `holds` refuses no bytes and a range that wraps.
    let whole = address.is_multiple_of(PAGE_SIZE)
        && source.is_multiple_of(PAGE_SIZE)
        && size.is_multiple_of(PAGE_SIZE)
        && regions.contains(address, size)
        && regions.contains(source, size);
    if !whole {
        return Err(AliasFault::Shape);
    }
    if previous_end.is_some_and(|end| address < end) {
        return Err(AliasFault::Order);
    }
    if !can_contain(regions, address, size, MemoryState::Stack) {
        return Err(AliasFault::Region);
    }
    let destination = Region {
        start: address,
        end: alias.end(),
    };
    let source_range = Region {
        start: source,
        end: alias.source_end(),
    };
    if destination.overlaps(source_range) {
        return Err(AliasFault::Overlap);
    }
    if !every_block(blocks, source_range, &LOCKED_SOURCE_CHECK) {
        return Err(AliasFault::Source);
    }
    if !every_block(blocks, destination, &ALIAS_DESTINATION_CHECK) {
        return Err(AliasFault::Destination);
    }
    let below = sources
        .range(..=source)
        .next_back()
        .is_some_and(|(_, &end)| end > source);
    let above = sources
        .range(source..)
        .next()
        .is_some_and(|(&start, _)| start < source_range.end);
    if below || above {
        return Err(AliasFault::SourceTaken);
    }
    Ok(())
}

/// Whether every block that `range` touches passes `check`, alike with the first
/// (`CheckMemoryState`). The range is whole pages inside the address space, which the blocks
/// tile, so the blocks it touches cover it.
fn every_block(blocks: &[MemoryBlock], range: Region, check: &Check) -> bool {
    let mut touched = blocks_touching(blocks, range);
    let Some(first) = touched.next() else {
        return false;
    };
    check.admits(first, first, true) && touched.all(|block| check.admits(first, block, true))
}

/// The parts of `block` that no alias of `aliases` covers, in address order, as `(address,
/// size)`: what the translation maps as fresh memory (check 10). `aliases` are in address order
/// and do not overlap (check 8).
pub(super) fn uncovered(block: &MemoryBlock, aliases: &[SavedAlias]) -> Vec<(u64, u64)> {
    let start = block.address();
    let end = block.end().unwrap_or(u64::MAX);
    // The aliases that end past the block's start, from the first of them. (Taking one that
    // ends exactly at the start too would change nothing: it covers none of the block.)
    let first = aliases.partition_point(|alias| alias.end() <= start);
    let mut parts = Vec::new();
    let mut cursor = start;
    for alias in aliases.get(first..).unwrap_or_default() {
        if alias.address >= end {
            break;
        }
        if alias.address > cursor {
            parts.push((cursor, alias.address - cursor));
        }
        cursor = cursor.max(alias.end().min(end));
    }
    if cursor < end {
        parts.push((cursor, end - cursor));
    }
    parts
}
