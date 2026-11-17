//! `MapMemory` and `UnmapMemory` on the address space ([`AddressSpace::map_memory`],
//! [`AddressSpace::unmap_memory`]): a second mapping of a process's own memory, its source
//! locked while the alias exists. A libnx program uses it to give a thread a stack in the stack
//! region backed by memory it took from its heap.
//!
//! Ported from Eden (GPL-3.0-or-later), paths relative to the reference folder:
//! `eden/src/core/hle/kernel/k_page_table_base.cpp` @ cf654052, `KPageTableBase::MapMemory`
//! and `KPageTableBase::UnmapMemory`, with `IsValidPageGroup` for the check that an alias maps
//! exactly its source's pages. Eden-NX2's and PR #4588's copies of the two functions are
//! Eden's, line for line (`eden-nx2/src/core/hle/kernel/k_page_table_base.cpp` @ 50fa86bb,
//! `eden-pr4588/src/core/hle/kernel/k_page_table_base.cpp` @ e5fb3b97).
//!
//! # `MapMemory`, in Eden's order
//!
//! 1. The source's blocks have one state, permission and attributes throughout
//!    (`CheckMemoryState`), a state with [`flag::CAN_ALIAS`], the permission
//!    [`MemoryPermission::USER_READ_WRITE`] exactly (`KMemoryPermission::All` masked), and no
//!    attributes (`KMemoryAttribute::All` masked, `None` expected; the counted ones,
//!    `IpcLocked | DeviceShared`, may differ between blocks but are tested too):
//!    [`KernelError::InvalidCurrentMemory`].
//! 2. The destination is free ([`MemoryState::Free`], no permission, no attributes):
//!    [`KernelError::InvalidCurrentMemory`].
//! 3. The source is reprotected to `KernelRead | NotMapped` (no access for the guest), then the
//!    destination is mapped as the source's pages, read-write ([`Mapper::protect`], then
//!    [`Mapper::map_alias`]). If the second fails, the source is reprotected back to
//!    read-write (Eden's `ON_RESULT_FAILURE`).
//! 4. The blocks: the source keeps its state and gets `KernelRead | NotMapped` and
//!    [`MemoryAttribute::LOCKED`], with the `Locked` merge barrier at its start; the
//!    destination becomes [`MemoryState::Stack`], read-write, with the `Normal` barrier.
//!
//! # `UnmapMemory`, in Eden's order
//!
//! 1. The source's blocks are alike, of a state with [`flag::CAN_ALIAS`], with the permission
//!    `KernelRead | NotMapped` exactly and the attribute [`MemoryAttribute::LOCKED`] alone:
//!    [`KernelError::InvalidCurrentMemory`].
//! 2. The destination's blocks are alike, [`MemoryState::Stack`] with no attributes:
//!    [`KernelError::InvalidCurrentMemory`].
//! 3. Each destination page is an alias of the source page at the same offset
//!    (`IsValidPageGroup(pg, src_address, num_pages)`): [`KernelError::InvalidMemoryRegion`].
//! 4. The destination is unmapped, then the source reprotected to read-write
//!    ([`Mapper::unmap`], then [`Mapper::protect`]). If the second fails, the destination is
//!    mapped again as the source's pages (Eden's `ON_RESULT_FAILURE`, `RemapPageGroup`).
//! 5. The blocks: the source gets read-write and no attributes, its `Locked` barrier cleared;
//!    the destination becomes free, its `Normal` barrier cleared.
//!
//! Every check is made, and the block map asked whether it would take both updates, before
//! the [`Mapper`] is asked anything; the block updates come after the [`Mapper`] and cannot
//! fail then. A TLB flush owed by a reprotect or unmap that happened before a failure is run
//! as after any system call: the translation records it, and the core loop runs it after
//! every call, whatever the call returned (`crates/core/src/machine/core_loop.rs`; the
//! translation's side is pinned by
//! `a_reprotect_before_a_refused_alias_still_owes_its_flush` in
//! `crates/core/src/space/tests/alias.rs`).
//!
//! # Divergences from Eden
//!
//! - **The page-table half checks its ranges itself.** Eden's two functions trust the system
//!   call to have checked that both ranges are whole pages inside the address space and that
//!   the destination may hold `Stack` memory. Here they check again: a range that is not whole
//!   pages inside the address space is [`KernelError::InvalidCurrentMemory`] (as every
//!   operation of the address space), then a destination the stack region cannot hold
//!   ([`AddressSpace::can_contain`] of [`MemoryState::Stack`]) is
//!   [`KernelError::InvalidMemoryRegion`], the system call's result for it. Test:
//!   `the_page_table_half_checks_pages_then_the_stack_region`.
//! - **No page groups.** Eden compares the physical pages the destination maps with the
//!   source's (`IsValidPageGroup`). With no physical pages, the address space records, for
//!   each run of alias pages, the source page its first page maps again ([`Aliases`]), and
//!   checks each destination page against that record. The two agree: a source page is
//!   aliased at most once at a time (step 1 of `MapMemory` refuses a `Locked` source), an
//!   alias cannot be a source (`Stack` has no [`flag::CAN_ALIAS`]), and the states that have
//!   it (heap and physical memory, `Normal`; module data, `CodeData`; their `AliasCodeData`
//!   twin) are memory whose pages no other virtual page maps, so destination page `i` holds
//!   the source's page `i` exactly when the record says so. A range that covers parts
//!   of two aliases passes when their pages line up with the source, as Eden's walk does.
//!   Test: `unmapping_needs_each_page_to_alias_the_source_page_at_its_offset`.
//! - **A [`Mapper`] that refuses** fails the call as an internal failure. Eden's `Operate`
//!   does not fail on these paths, and it asserts its undo (`R_ASSERT`). Here, when the undo
//!   fails too, the bookkeeping says what the translation then holds: in `MapMemory`, the
//!   source stays without access for the guest, as `KernelRead | NotMapped` memory with no
//!   attributes and no alias (so `SetMemoryPermission` can give it back); in `UnmapMemory`,
//!   the destination is free and the alias forgotten, and the source stays locked. Tests:
//!   `a_refused_alias_reprotects_the_source_back_and_changes_nothing`,
//!   `a_refused_undo_of_the_source_leaves_it_without_access_and_unlocked`,
//!   `a_refused_reprotect_maps_the_alias_again_and_changes_nothing`,
//!   `a_refused_remap_leaves_the_destination_free_and_the_source_locked`.

use std::collections::BTreeMap;

use starquake_memory::PAGE_SIZE;

use super::{AddressSpace, Change, Check, Effect, broken};
use crate::memory::mapper::MapperError;
use crate::memory::state::{
    DisableMerge, MemoryAttribute, MemoryPermission, MemoryState, STATE_ALL, flag,
};
use crate::result::{CallResult, KernelError, SvcError};

/// What `MapMemory` makes of the source: `KernelRead | NotMapped`, no access for the guest
/// (`new_src_perm` in `KPageTableBase::MapMemory`).
const LOCKED_SOURCE: MemoryPermission = MemoryPermission::from_bits(
    MemoryPermission::KERNEL_READ.bits() | MemoryPermission::NOT_MAPPED.bits(),
);

/// One run of alias pages: the first source page it maps again, and its size in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Run {
    source: u64,
    size: u64,
}

/// The `MapMemory` aliases of an address space: runs of alias pages by where each starts, each
/// mapping the source pages from `source` on in order. Runs never overlap (an alias is made
/// only over free memory) and none is empty.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Aliases(BTreeMap<u64, Run>);

impl Aliases {
    /// The run with the lowest address, as `(address, size)`.
    pub(super) fn first(&self) -> Option<(u64, u64)> {
        self.0
            .first_key_value()
            .map(|(address, run)| (*address, run.size))
    }

    /// Every run, in address order, as `(address, source, size)`.
    #[cfg(test)]
    pub(super) fn runs(&self) -> Vec<(u64, u64, u64)> {
        self.0
            .iter()
            .map(|(address, run)| (*address, run.source, run.size))
            .collect()
    }

    /// Records that the `size` bytes at `address` map the source pages from `source` on.
    pub(super) fn insert(&mut self, address: u64, source: u64, size: u64) {
        self.0.insert(address, Run { source, size });
    }

    /// Whether every page of the `size` bytes at `address` is an alias of the page at the
    /// same offset from `source` (`IsValidPageGroup`). False for no bytes: Eden's empty group
    /// is invalid. The walk takes one run per step and each step moves past the run, so it
    /// ends within the number of runs.
    pub(super) fn matches(&self, address: u64, source: u64, size: u64) -> bool {
        let Some(end) = address.checked_add(size).filter(|_| size != 0) else {
            return false;
        };
        let mut cursor = address;
        while cursor < end {
            let Some((&start, run)) = self.0.range(..=cursor).next_back() else {
                return false;
            };
            let Some(run_end) = start.checked_add(run.size).filter(|end| cursor < *end) else {
                return false;
            };
            // The run maps `start + k` to `run.source + k`; this page must map
            // `source + (cursor - address)`.
            let mapped = run.source.checked_add(cursor - start);
            let wanted = source.checked_add(cursor - address);
            if mapped.is_none() || mapped != wanted {
                return false;
            }
            cursor = run_end;
        }
        true
    }

    /// Forgets the `size` bytes at `address`, cutting the runs at both ends. The bytes past
    /// the range of a run that starts before it or ends after it stay recorded.
    pub(super) fn remove(&mut self, address: u64, size: u64) {
        let end = address.saturating_add(size);
        let before = self
            .0
            .range(..address)
            .next_back()
            .map(|(start, run)| (*start, *run));
        let inside: Vec<(u64, Run)> = self
            .0
            .range(address..end)
            .map(|(start, run)| (*start, *run))
            .collect();
        for (start, run) in before.into_iter().chain(inside) {
            let run_end = start.saturating_add(run.size);
            if run_end <= address {
                continue;
            }
            self.0.remove(&start);
            if start < address {
                self.insert(start, run.source, address - start);
            }
            if run_end > end {
                let offset = end - start;
                self.insert(end, run.source.saturating_add(offset), run_end - end);
            }
        }
    }
}

impl AddressSpace {
    /// The range checks both calls' page-table halves make first (module docs, "Divergences"):
    /// both ranges whole pages inside the address space, then a destination the stack region
    /// can hold. Returns the page count.
    fn alias_pages(&self, dst: u64, src: u64, size: u64) -> CallResult<u64> {
        let pages = self.pages_of(src, size)?;
        self.pages_of(dst, size)?;
        if !self.can_contain(dst, size, MemoryState::Stack) {
            return Err(KernelError::InvalidMemoryRegion.into());
        }
        Ok(pages)
    }

    /// Maps the `size` bytes at `src` a second time at `dst`, as [`MemoryState::Stack`] memory
    /// of the same pages, and takes the source from the guest until
    /// [`Self::unmap_memory`] (`KPageTableBase::MapMemory`). See the module docs for the checks
    /// and their order; the system call checks the arguments first
    /// ([`crate::svc::memory::map_memory`]).
    pub fn map_memory(&mut self, dst: u64, src: u64, size: u64) -> CallResult<()> {
        let pages = self.alias_pages(dst, src, size)?;
        // Validate that the source address's state is valid.
        let source_check = Check {
            state_mask: flag::CAN_ALIAS,
            state: flag::CAN_ALIAS,
            permission_mask: MemoryPermission::ALL,
            permission: MemoryPermission::USER_READ_WRITE,
            attribute_mask: MemoryAttribute::ALL,
            attribute: MemoryAttribute::NONE,
            ignore: MemoryAttribute::COUNTED,
        };
        let source = self.check_memory_state(src, size, &source_check)?;
        // Validate that the dst address's state is valid.
        self.check_memory_state(dst, size, &Check::FREE)?;
        let source_change = Change {
            state: source.state,
            permission: LOCKED_SOURCE,
            attribute: MemoryAttribute::LOCKED,
            set: DisableMerge::LOCKED,
            clear: DisableMerge::NONE,
        };
        let alias_change = Change {
            state: MemoryState::Stack,
            permission: MemoryPermission::USER_READ_WRITE,
            attribute: MemoryAttribute::NONE,
            set: DisableMerge::NORMAL,
            clear: DisableMerge::NONE,
        };
        self.check_updates(src, dst, pages, &source_change, &alias_change)?;
        let locked = guest_perms(LOCKED_SOURCE)?;
        let read_write = guest_perms(MemoryPermission::USER_READ_WRITE)?;
        // Reprotect the source as kernel-read/not mapped.
        self.mapper
            .protect(src, size, locked)
            .map_err(|error| SvcError::Internal(refused(Effect::Protect, src, size, &error)))?;
        // Map the alias pages; on failure, unprotect the source pages.
        if let Err(error) = self.mapper.map_alias(dst, src, size, read_write) {
            let failed = refused(Effect::Map, dst, size, &error);
            return Err(match self.mapper.protect(src, size, read_write) {
                Ok(()) => SvcError::Internal(failed),
                Err(undo) => self.keep_source_inaccessible(src, pages, source.state, failed, &undo),
            });
        }
        // Apply the memory block updates.
        self.update(src, pages, &source_change)?;
        self.update(dst, pages, &alias_change)?;
        self.aliases.insert(dst, src, size);
        Ok(())
    }

    /// Unmaps the alias of `size` bytes at `dst` that [`Self::map_memory`] made of `src`, and
    /// gives the source back to the guest (`KPageTableBase::UnmapMemory`). See the module docs
    /// for the checks and their order; the system call checks the arguments first
    /// ([`crate::svc::memory::unmap_memory`]).
    pub fn unmap_memory(&mut self, dst: u64, src: u64, size: u64) -> CallResult<()> {
        let pages = self.alias_pages(dst, src, size)?;
        // Validate that the source address's state is valid.
        let source_check = Check {
            state_mask: flag::CAN_ALIAS,
            state: flag::CAN_ALIAS,
            permission_mask: MemoryPermission::ALL,
            permission: LOCKED_SOURCE,
            attribute_mask: MemoryAttribute::ALL,
            attribute: MemoryAttribute::LOCKED,
            ignore: MemoryAttribute::COUNTED,
        };
        let source = self.check_memory_state(src, size, &source_check)?;
        // Validate that the dst address's state is valid.
        let alias_check = Check {
            state_mask: STATE_ALL,
            state: MemoryState::Stack.word(),
            permission_mask: MemoryPermission::NONE,
            permission: MemoryPermission::NONE,
            attribute_mask: MemoryAttribute::ALL,
            attribute: MemoryAttribute::NONE,
            ignore: MemoryAttribute::COUNTED,
        };
        self.check_memory_state(dst, size, &alias_check)?;
        // Ensure the page group is the valid for the source.
        if !self.aliases.matches(dst, src, size) {
            return Err(KernelError::InvalidMemoryRegion.into());
        }
        let source_change = Change {
            state: source.state,
            permission: MemoryPermission::USER_READ_WRITE,
            attribute: MemoryAttribute::NONE,
            set: DisableMerge::NONE,
            clear: DisableMerge::LOCKED,
        };
        let free_change = Change {
            state: MemoryState::Free,
            permission: MemoryPermission::NONE,
            attribute: MemoryAttribute::NONE,
            set: DisableMerge::NONE,
            clear: DisableMerge::NORMAL,
        };
        self.check_updates(src, dst, pages, &source_change, &free_change)?;
        let read_write = guest_perms(MemoryPermission::USER_READ_WRITE)?;
        // Unmap the aliased copy of the pages.
        self.mapper
            .unmap(dst, size)
            .map_err(|error| SvcError::Internal(refused(Effect::Unmap, dst, size, &error)))?;
        // Try to set the permissions for the source pages back to what they should be; on
        // failure, re-map the aliased pages.
        if let Err(error) = self.mapper.protect(src, size, read_write) {
            let failed = refused(Effect::Protect, src, size, &error);
            return Err(match self.mapper.map_alias(dst, src, size, read_write) {
                Ok(()) => SvcError::Internal(failed),
                Err(undo) => self.forget_alias(dst, pages, &free_change, failed, &undo),
            });
        }
        // Apply the memory block updates.
        self.update(src, pages, &source_change)?;
        self.update(dst, pages, &free_change)?;
        self.aliases.remove(dst, size);
        Ok(())
    }

    /// Asks the block map whether it would take both changes, before anything changes.
    fn check_updates(
        &self,
        src: u64,
        dst: u64,
        pages: u64,
        source: &Change,
        alias: &Change,
    ) -> CallResult<()> {
        for (address, change) in [(src, source), (dst, alias)] {
            self.blocks
                .check_update(
                    address,
                    pages,
                    change.state,
                    change.permission,
                    change.attribute,
                )
                .map_err(broken)?;
        }
        Ok(())
    }

    /// Applies `change` to the block map, which [`Self::check_updates`] said it would take.
    fn update(&mut self, address: u64, pages: u64, change: &Change) -> CallResult<()> {
        self.blocks
            .update(
                address,
                pages,
                change.state,
                change.permission,
                change.attribute,
                change.set,
                change.clear,
            )
            .map_err(broken)
    }

    /// `MapMemory`'s undo failed (module docs, "Divergences"): the source stays without access
    /// for the guest in the translation, so the blocks say so, unlocked. Returns the error the
    /// call fails with.
    fn keep_source_inaccessible(
        &mut self,
        src: u64,
        pages: u64,
        state: MemoryState,
        failed: String,
        undo: &MapperError,
    ) -> SvcError {
        let kept = Change {
            state,
            permission: LOCKED_SOURCE,
            attribute: MemoryAttribute::NONE,
            set: DisableMerge::NONE,
            clear: DisableMerge::NONE,
        };
        match self.update(src, pages, &kept) {
            Ok(()) => SvcError::Internal(format!(
                "{failed}; reprotecting the source back failed too, so the guest cannot reach \
                 it: {undo}"
            )),
            Err(error) => error,
        }
    }

    /// `UnmapMemory`'s undo failed (module docs, "Divergences"): the destination is no longer
    /// mapped in the translation, so the blocks say it is free and the alias is forgotten; the
    /// source stays locked. Returns the error the call fails with.
    fn forget_alias(
        &mut self,
        dst: u64,
        pages: u64,
        free: &Change,
        failed: String,
        undo: &MapperError,
    ) -> SvcError {
        if let Err(error) = self.update(dst, pages, free) {
            return error;
        }
        self.aliases.remove(dst, pages.saturating_mul(PAGE_SIZE));
        SvcError::Internal(format!(
            "{failed}; mapping the alias again failed too, so it is gone: {undo}"
        ))
    }
}

/// The translation's permission for `permission`.
fn guest_perms(permission: MemoryPermission) -> CallResult<starquake_memory::Perms> {
    permission.guest().ok_or_else(|| {
        SvcError::Internal(format!(
            "permission {:#x} has no translation",
            permission.bits()
        ))
    })
}

/// What a refused [`crate::memory::Mapper`] call fails the call with, in words.
fn refused(effect: Effect, address: u64, size: u64, error: &MapperError) -> String {
    format!(
        "the translation refused to {} {size:#x} bytes at {address:#x}: {error}",
        effect.verb()
    )
}
