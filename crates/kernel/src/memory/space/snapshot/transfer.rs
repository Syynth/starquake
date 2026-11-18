//! The address space's locked memory checked against the transfer memories that lent it
//! ([`AddressSpace::check_lends`]): check 11 of the address space's restore (module docs of
//! [`super`], "Checks"), made once the object table is restored, since the transfer memory
//! objects are the object table's (`crate::object::snapshot`).
//!
//! # What a lend leaves, live
//!
//! `KTransferMemory::Initialize` locks its owner's pages through `LockForTransferMemory`
//! (`eden/src/core/hle/kernel/k_transfer_memory.cpp` and `k_page_table_base.cpp` @ cf654052;
//! here [`AddressSpace::lock_for_transfer_memory`]): every block of the range has a state with
//! [`TRANSFER_LOCK_STATE`], one state throughout, is read-write for the user, and has no
//! attribute at all (the attribute test's mask is `KMemoryAttribute::All` with `None`
//! expected; the counted attributes are ignored only when the blocks are compared with the
//! first); afterwards each has exactly the `Locked` attribute and the permission
//! `ConvertToKMemoryPermission` makes of what the owner keeps. While the range is
//! lent nothing changes those blocks' state, permission or attributes: every call that would
//! (`SetMemoryPermission`, `SetProcessMemoryPermission`, `SetMemoryAttribute` and `svc 0x80`,
//! `MapMemory`, a heap shrink, `UnmapPhysicalMemory`, a device-share lock and a second lend)
//! tests the attributes with `Locked` in the mask and none expected, so refuses a locked block;
//! the one that does not, `UnlockForTransferMemory`, is called only by the transfer memory's
//! own `Finalize` with its own range. So two lends never share a page, and a lend never shares
//! a page with a `MapMemory` alias's source, whose pages `MapMemory` leaves locked too
//! (`a_transfer_memory_lend_and_an_alias_refuse_each_others_source_but_the_take_back_does_not`,
//! in `memory/space/alias_tests.rs`). The only `Locked` memory is a lend's or an alias source's:
//! no other call sets the attribute.
//!
//! # Checks
//!
//! [`AddressSpace::check_lends`] takes the transfer memories the space's process owns, sorts
//! them by address (then size), and checks, in this order, returning the first failure:
//!
//! 1. Each lend in turn ([`LendError::Lend`], first failing check of the first failing lend,
//!    in [`LendFault`]'s order): its range is whole pages, not empty, and inside the address
//!    space ([`LendFault::Shape`]); it starts at or after the end of the one before
//!    ([`LendFault::Overlap`]); it shares no page with an alias's source
//!    ([`LendFault::AliasSource`]); and every block it touches, in address order, has exactly
//!    the `Locked` attribute ([`LendFault::NotLocked`]), a state with
//!    [`TRANSFER_LOCK_STATE`] that is the first block's ([`LendFault::State`]), and the
//!    permission its owner keeps ([`LendFault::Permission`]).
//! 2. Every page of a `Locked` block, in address order, is a lend's or an alias source's
//!    ([`LendError::Unlent`], at the first page that is neither).
//!
//! # Not checked here
//!
//! - Where the `Locked` merge barrier sits: a lend's first block gets it, and a restore takes
//!   the barriers as they are (module docs of [`super`], "Divergences and what is not
//!   checked").
//! - That the lends' owner is this space's process: the caller hands over the transfer memories
//!   of one owner (`crate::object::snapshot` groups them by owner).

use super::super::TRANSFER_LOCK_STATE;
use super::{AddressSpace, blocks_touching};
use crate::memory::block::MemoryBlock;
use crate::memory::layout::Region;
use crate::memory::shared::TransferMemory;
use crate::memory::state::{MemoryAttribute, MemoryPermission};
use starquake_memory::PAGE_SIZE;

/// What went wrong with one transfer memory's range ([`LendError::Lend`]), in the order the
/// checks are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LendFault {
    /// It is not whole pages, is empty, or does not lie inside the address space.
    #[error("is not whole pages inside the address space")]
    Shape,
    /// It starts before the end of the lend before it, by address.
    #[error("overlaps the transfer memory before it")]
    Overlap,
    /// It shares a page with a `MapMemory` alias's source.
    #[error("shares a page with a MapMemory alias's source")]
    AliasSource,
    /// One of its blocks does not have exactly the `Locked` attribute: it lacks it, or has
    /// another besides it.
    #[error("has a block that is not locked for it")]
    NotLocked,
    /// One of its blocks has a state that cannot be lent, or a state other than its first
    /// block's.
    #[error("has a block whose memory state cannot be lent or differs from its first block's")]
    State,
    /// One of its blocks has a permission other than what the owner keeps.
    #[error("has a block whose permission is not what its owner keeps")]
    Permission,
}

/// Why an address space's locked memory disagrees with its process's transfer memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LendError {
    /// A transfer memory's range fails a check.
    #[error("the transfer memory of {size:#x} bytes at {address:#x} {fault}")]
    Lend {
        /// Where the range starts.
        address: u64,
        /// Its size in bytes.
        size: u64,
        /// The first check it fails.
        fault: LendFault,
    },
    /// A locked page that no transfer memory lent and that is no alias's source.
    #[error(
        "the locked page at {address:#x} is lent to no transfer memory and is no alias's source"
    )]
    Unlent {
        /// The page.
        address: u64,
    },
}

/// The `[start, end)` range of `size` bytes at `address`, if the bytes lie inside the address
/// space as whole pages.
fn lend_range(space: &AddressSpace, address: u64, size: u64) -> Option<Region> {
    let whole = address.is_multiple_of(PAGE_SIZE) && size.is_multiple_of(PAGE_SIZE);
    if !whole || !space.regions.contains(address, size) {
        return None;
    }
    let end = address.checked_add(size)?;
    Some(Region {
        start: address,
        end,
    })
}

/// The first check of [`LendFault`]'s block checks `block` fails, if any. `first` is the
/// lend's first block, `permission` what its owner keeps.
pub(super) fn block_fault(
    block: &MemoryBlock,
    first: &MemoryBlock,
    permission: MemoryPermission,
) -> Option<LendFault> {
    if block.attribute() != MemoryAttribute::LOCKED {
        return Some(LendFault::NotLocked);
    }
    let lendable = block.state().word() & TRANSFER_LOCK_STATE == TRANSFER_LOCK_STATE;
    if !lendable || block.state() != first.state() {
        return Some(LendFault::State);
    }
    (block.permission() != permission).then_some(LendFault::Permission)
}

/// Whether `range` shares a page with one of `sources`, which are sorted by start and do not
/// overlap each other.
fn touches_any(sources: &[Region], range: Region) -> bool {
    let after = sources.partition_point(|source| source.end <= range.start);
    sources
        .get(after)
        .is_some_and(|source| source.overlaps(range))
}

/// The first page of `block` that none of `covered` holds, or `None` when every page is held.
/// `covered` is sorted by start and its ranges do not overlap, so each step moves past one range
/// and the walk ends within their number.
fn unheld(block: &MemoryBlock, covered: &[Region]) -> Option<u64> {
    let end = block.end().unwrap_or(u64::MAX);
    let mut cursor = block.address();
    while cursor < end {
        match next_end(covered, cursor) {
            Some(range_end) => cursor = range_end,
            None => return Some(cursor),
        }
    }
    None
}

/// The end of the range of `covered` that holds `address`, if one does.
fn next_end(covered: &[Region], address: u64) -> Option<u64> {
    covered
        .partition_point(|range| range.start <= address)
        .checked_sub(1)
        .and_then(|at| covered.get(at))
        .filter(|range| address < range.end)
        .map(|range| range.end)
}

impl AddressSpace {
    /// Checks the space's locked memory against `lends`, the transfer memories its process owns
    /// (module docs of `memory::space::snapshot::transfer` for the checks and their order): what
    /// a restore of the object table runs for each process's address space, and what its save
    /// runs on what it is about to write. A live space and its process's live transfer memories
    /// pass: a failure is a corrupt snapshot, or a bug in the kernel.
    pub fn check_lends(&self, lends: &[TransferMemory]) -> Result<(), LendError> {
        let blocks: Vec<MemoryBlock> = self.blocks().copied().collect();
        let mut sources: Vec<Region> = self
            .aliases
            .iter()
            .map(|(_, source, size)| Region {
                start: source,
                end: source.saturating_add(size),
            })
            .collect();
        sources.sort_unstable_by_key(|source| source.start);
        let mut sorted: Vec<&TransferMemory> = lends.iter().collect();
        sorted.sort_unstable_by_key(|lend| (lend.address(), lend.size()));
        let mut covered = sources.clone();
        let mut previous_end = None;
        for lend in sorted {
            let (address, size) = (lend.address(), lend.size());
            let fault = |fault| LendError::Lend {
                address,
                size,
                fault,
            };
            let range = lend_range(self, address, size).ok_or(fault(LendFault::Shape))?;
            if previous_end.is_some_and(|end| range.start < end) {
                return Err(fault(LendFault::Overlap));
            }
            if touches_any(&sources, range) {
                return Err(fault(LendFault::AliasSource));
            }
            let permission = MemoryPermission::from_svc(lend.owner_permission());
            let mut touched = blocks_touching(&blocks, range);
            if let Some(first) = touched.next() {
                for block in core::iter::once(first).chain(touched) {
                    if let Some(found) = block_fault(block, first, permission) {
                        return Err(fault(found));
                    }
                }
            }
            covered.push(range);
            previous_end = Some(range.end);
        }
        covered.sort_unstable_by_key(|range| range.start);
        let locked = blocks
            .iter()
            .filter(|block| block.attribute().contains(MemoryAttribute::LOCKED));
        for block in locked {
            if let Some(address) = unheld(block, &covered) {
                return Err(LendError::Unlent { address });
            }
        }
        Ok(())
    }
}
