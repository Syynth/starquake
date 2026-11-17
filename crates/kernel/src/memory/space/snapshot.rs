//! A process's address space saved and restored ([`AddressSpace::save`],
//! [`AddressSpace::restore`]): the observability spec's snapshots, step 6 ("kernel and memory
//! first"), with the kernel's shared header ([`crate::snapshot`]).
//!
//! # What is saved
//!
//! What the address space was created from (the code and the layout's inputs), the regions,
//! every memory block with every count it keeps (the counts behind the merge barriers too), the
//! heap's size, the largest heap, the main thread's stack size, the bytes reserved from the
//! pool, the mapped physical memory size (`m_mapped_physical_memory_size`), the shared memory
//! mappings, the device-share locks by exact range with how many are live on each, and the
//! `MapMemory` aliases: each run of alias pages the address space records
//! (`memory::space::alias`), as where it starts, the first source page it maps again, and its
//! size.
//!
//! Not saved, because a restore derives it: the largest process memory (the heap region's size)
//! and the pool's limit (the arrangement's application pool). Not saved because it is not the
//! address space's: the [`Mapper`] (a restore is handed a new one and rebuilds what it maps,
//! below), what guest memory holds (the snapshot's guest memory part, not built yet), and the
//! shared and transfer memory objects (the object table's part).
//!
//! # Format
//!
//! [`crate::snapshot`]'s header, with [`ADDRESS_SPACE_MAGIC`] and [`ADDRESS_SPACE_VERSION`], then
//! the body, in this order: the creation request (code address, code size, arrangement tag,
//! whether the alias region gets its extra size, the alias, heap, stack and kernel-map offsets);
//! the regions (address space, code, alias, heap, stack, kernel map, each start and end, then
//! the alias region's extra size); the heap size, the largest heap, the bytes reserved, the main
//! thread's stack size and the mapped physical memory size; the blocks (each address, pages,
//! state word, permission, original permission, attributes, IPC lock count, device use count,
//! IPC merge count, device left and right merge counts, merge barriers); the shared memory
//! mappings (each address, size, object); the device-share locks (each address, size, count);
//! the aliases (each address, source, size; version 2 added them, and a version 1 snapshot is
//! refused with the version error: `a_version_1_snapshot_is_refused_with_the_version_error`).
//! The arrangement tag is starquake's own ([`ARRANGEMENT_TAGS`]); the state word is the
//! reference's (`KMemoryState`, [`MemoryState::word`]), and the bit sets are the reference's
//! bits ([`super::super::state`]). The golden-bytes tests `the_body_layout_is_pinned` and
//! `the_arrangement_tags_are_pinned` (in `snapshot/tests.rs`) pin it; changing it means bumping
//! [`ADDRESS_SPACE_VERSION`].
//!
//! # Checks
//!
//! A snapshot is a file and is not trusted. [`AddressSpace::restore`] checks, in this order, and
//! returns the first failure ([`SnapshotError`]):
//!
//! 1. The header and the body's encoding ([`crate::snapshot::decode`]: header, magic, version,
//!    body, trailing bytes).
//! 2. The arrangement tag ([`SnapshotError::Arrangement`]), then that the code is whole pages and
//!    the request can be laid out ([`SnapshotError::Layout`]), then that the saved regions are
//!    exactly where the layout puts them ([`SnapshotError::Regions`]).
//! 3. The counts, each against what fixed it live: the stack is whole pages and, if there is
//!    one, it and the code fit the largest process memory ([`SnapshotError::Stack`]); the
//!    largest heap is zero or what the stack left (`KProcess::Run`), and exactly that when
//!    there is a stack ([`SnapshotError::MaxHeap`]); the heap is whole pages and at most the
//!    largest heap, which is at most the heap region ([`SnapshotError::Heap`]); the reservation
//!    is at most the pool ([`SnapshotError::OverLimit`]) and exactly the code, stack, heap and
//!    mapped physical memory ([`SnapshotError::Reservation`]).
//! 4. Each block in turn ([`SnapshotError::Block`], first failing check of the first failing
//!    block): its shape, that it starts where the one before ended (the first at the address
//!    space's start: [`BlockFault::Overlap`] below that, [`BlockFault::Gap`] above), that it
//!    ends inside the address space, its state word, that its permission and attributes have
//!    no bit starquake does not name ([`BlockFault::Bits`]), that it could not be merged with
//!    the one before, that a free block has nothing, that it has no IPC lock state, that its
//!    device counts agree with its attribute and barriers, that its permission has a
//!    translation, that a block that is not free has a permission the kernel gives its state
//!    ([`BlockFault::Permission`]: one of the four converted system call permissions, and not
//!    writable if its state is code), and that it lies in the region of its state. Then that
//!    the blocks reach the address space's end ([`SnapshotError::Short`]).
//! 5. The code's pages are `Code` or `CodeData` ([`SnapshotError::CodeRange`]); the heap's
//!    pages are `Normal` and the heap region holds no other `Normal` memory
//!    ([`SnapshotError::HeapRange`]); the mapped physical memory size is the bytes of `Normal`
//!    memory inside the alias region ([`SnapshotError::Physical`]).
//! 6. Each shared memory mapping in turn ([`SnapshotError::Shared`]): after the one before, and
//!    exactly one `Shared` block. Then that every `Shared` block is a mapping
//!    ([`SnapshotError::Unmapped`]).
//! 7. Each device-share lock in turn ([`SnapshotError::Lock`]): a count that is not zero, after
//!    the one before by address then size, whole pages inside the address space, starting and
//!    ending where blocks start and end, only over device-shared blocks
//!    ([`LockFault::NotShared`]), and only over blocks whose state a lock can be taken on and
//!    released from ([`LockFault::State`]: `CanAlignedDeviceMap`, `ReferenceCounted` and
//!    `CanDeviceMap`). Then that each
//!    block's device use count and its left and right merge counts are what the locks over,
//!    starting at and ending at it add up to ([`SnapshotError::DeviceCounts`]).
//! 8. Each `MapMemory` alias in turn ([`SnapshotError::Alias`], first failing check of the
//!    first failing alias, in [`AliasFault`]'s order): its destination, source and size are
//!    whole pages and both ranges inside the address space ([`AliasFault::Shape`]); it starts
//!    at or after the end of the one before ([`AliasFault::Order`]); the stack region can hold
//!    its destination ([`AliasFault::Region`], `CanContain` of `Stack`); its source and
//!    destination share no page ([`AliasFault::Overlap`]); every block of its source passes
//!    `UnmapMemory`'s check of a source, alike with the first (a state with `CanAlias`,
//!    `KernelRead | NotMapped`, `Locked` alone: [`AliasFault::Source`]); every block of its
//!    destination is what `MapMemory` leaves there (`Stack`, read-write, no attribute but
//!    `DeviceShared`: [`AliasFault::Destination`]); and its source shares no page with an
//!    earlier alias's ([`AliasFault::SourceTaken`]).
//! 9. Each shared memory mapping's object, from the caller: it exists, is the mapping's size,
//!    and its user permission is the permission of the mapping's block ([`SharedFault::Missing`],
//!    [`SharedFault::Size`], [`SharedFault::Permission`]).
//! 10. The translation: every block that is not free is mapped through the new [`Mapper`], in
//!     address order, a `Shared` one with its shared memory's pages ([`Mapper::map_shared`]),
//!     any other as fresh memory ([`Mapper::map`]), with the guest permission of its
//!     permission, leaving out the pages an alias covers (one call for each part left). Then
//!     each alias, in address order, as its source's pages, read-write
//!     ([`Mapper::map_alias`]). A refusal unmaps what was mapped, the aliases first (each maps
//!     pages of a source mapped before it), then the rest, each in address order, and is
//!     [`SnapshotError::Translation`], which counts the bytes whose unmap was refused too. The
//!     pool's count, the block map and the alias record are built before it (the first two's
//!     own errors, [`SnapshotError::OverLimit`] and [`SnapshotError::BlockMap`], would be bugs
//!     in checks 3 and 4), so this is the last step that can fail: whatever a restore returns
//!     as an error, nothing it mapped is left mapped, except the bytes `stuck` counts.
//!
//! [`AddressSpace::save`] runs checks 2 to 8 on what it is about to write: a live address space
//! that fails one is a bug in the kernel, which it reports rather than writes.
//!
//! # Why each relation holds live
//!
//! - **The reservation.** [`AddressSpace::new`] reserves the code, the main thread's stack is
//!   reserved when it is made, heap growth reserves what it grows and a shrink releases it, and
//!   `MapPhysicalMemory` reserves what it maps (and keeps what an undo could not unmap, counted
//!   as mapped physical memory); nothing else reserves.
//! - **The code and the heap.** The code is mapped as `Code` when the address space is made,
//!   and only `SetProcessMemoryPermission` changes its state, to `CodeData` when it becomes
//!   writable. Only heap growth maps `Normal` memory inside the heap region (nothing in the kernel calls
//!   [`AddressSpace::map_pages`] with `Normal`; a caller that did would make a space that fails
//!   this check), and only a heap shrink frees it.
//! - **The mapped physical memory.** Only `MapPhysicalMemory` makes `Normal` memory in the
//!   alias region (`CanContain` keeps `MapPages` of `Normal` out of it), and only
//!   `UnmapPhysicalMemory` frees it. The count is of the `Normal` bytes inside the alias
//!   region, and `Normal` memory may lie only in the heap and alias regions (the region check
//!   of check 4 takes each block's bytes in the two together).
//! - **Shared memory.** Each mapping is one block: `MapPageGroup` sets the `Normal` barrier at
//!   its start, and no operation here can change part of a `Shared` block (its state has none of
//!   the flags the reprotecting, attribute, transfer and device calls test for).
//! - **The permissions.** Every call that maps or reprotects memory gives it
//!   `ConvertToKMemoryPermission` of one of the four system call permissions
//!   ([`MemoryPermission::from_svc`]), or a constant equal to one (`KernelRead | NotMapped` for
//!   the code as it is created, `UserReadWrite` for heap, physical memory, stacks and
//!   thread-local pages); [`AddressSpace::map_pages`] takes any permission, and a caller that
//!   passed another would make a space that fails this check. `SetProcessMemoryPermission` makes
//!   writable code code data, and code cannot be reprotected otherwise (its state has no
//!   `CanReprotect`). A `Shared` block keeps the permission it was mapped with, which
//!   `MapSharedMemory` requires to be the shared memory's user permission
//!   (`svc::shared_memory`), since no `Shared` state can be reprotected.
//! - **The device-share locks' states.** [`AddressSpace::lock_device_share`] takes a lock only
//!   on blocks with `CanAlignedDeviceMap | ReferenceCounted`, every state with those has
//!   `CanDeviceMap` too (what [`AddressSpace::unlock_device_share`] tests; so check 7's test
//!   of `CanDeviceMap` refuses no block the other two pass, and is there in case a state is
//!   added: `every_state_a_lock_can_be_taken_on_can_be_unlocked`), and the only state
//!   change a device-shared block can go through here, `SetProcessMemoryPermission`'s `Code` to
//!   `CodeData` (or `AliasCode` to `AliasCodeData`), keeps all three.
//! - **The device counts.** A lock raises the use count of every block of its range and the
//!   left and right merge counts of the blocks it starts and ends at (`ShareToDevice`); an
//!   unlock of the same exact range lowers the same counts (`UnshareToDevice`), a split gives the
//!   upper part no left count and the lower no right count, and a merge happens only between
//!   blocks with no barrier between them, which is no count between them. So each count is the
//!   sum over the live locks, and its barrier is set exactly when it is not zero. Eden's
//!   `UpdateDeviceDisableMergeStateForUnshareLeft` also caps the left count at the use count;
//!   with exact-range unlocks the cap never bites, since a block's left count never passes the
//!   locks over it.
//! - **The aliases** (`KPageTableBase::MapMemory` and `UnmapMemory` in
//!   `eden/src/core/hle/kernel/k_page_table_base.cpp` @ cf654052; `memory::space::alias`).
//!   `MapMemory` checks both ranges are whole pages inside the address space and that the
//!   stack region can hold the destination, takes a source whose blocks are alike, of a state
//!   with `CanAlias`, read-write and with no attribute (`KMemoryAttribute::All` masked, so not
//!   device shared either), and a free destination, and leaves the source `KernelRead |
//!   NotMapped` and `Locked` and the destination `Stack`, read-write, with no attribute. Since
//!   the source is not free and the destination is, they share no page; since a `Locked`
//!   source is refused, no source page is aliased twice. After that, `UnmapMemory` changes
//!   both ranges (cutting a run where it unmaps part of it). Of the other operations on a
//!   source's blocks, [`AddressSpace::set_process_memory_permission`],
//!   [`AddressSpace::set_memory_permission`],
//!   [`AddressSpace::set_memory_attribute`], [`AddressSpace::set_heap_size`]'s shrink,
//!   [`AddressSpace::lock_for_transfer_memory`] and the device-share lock test the attributes
//!   with `Locked` in the mask and `None` expected, so they refuse it, and
//!   [`AddressSpace::unmap_physical_memory`] takes only `Normal` memory with no attribute. The
//!   exception is [`AddressSpace::unlock_for_transfer_memory`], which expects `Locked` and does
//!   not test the permission, so it would take a source; the kernel calls it only on the
//!   range a transfer memory locked, and that is never a source, since each of the transfer
//!   memory lock and `MapMemory` refuses the other's `Locked` range. Of the operations on a
//!   destination's blocks, `Stack` has neither `CanReprotect` nor `CanChangeAttribute`, and a
//!   device-share lock is the one change, adding `DeviceShared`. Pinned by
//!   `a_locked_source_cannot_be_device_share_locked_and_a_free_destination_neither` and
//!   `map_memory_refuses_a_source_a_device_holds_on_any_of_its_pages`
//!   (`memory/space/alias_tests.rs`).
//!
//! The generated test `a_space_built_by_random_operations_saves_and_restores_as_it_was` runs
//! random sequences of heap size changes, physical memory maps and unmaps, device-share locks and
//! unlocks (of heap, physical memory and stack pages), attribute and permission changes,
//! transfer memory lends and returns, and `MapMemory` and `UnmapMemory` (of whole runs and parts
//! of them) on a live address space, and saves and restores after each, so a relation those
//! operations can break fails it (`the_random_operations_reach_split_and_merged_aliases` shows
//! they reach two runs in one block and a run cut by an unmap). Shared memory, the stack,
//! thread-local pages and generated code are in the fixed round trip
//! `a_rich_space_saves_and_restores_as_it_was`, with an alias a device holds.
//!
//! # Divergences and what is not checked
//!
//! - **Stricter: no IPC lock state.** A block with an IPC lock count, an IPC merge count, an
//!   original permission, the `IpcLocked` attribute or the `IpcLeft` barrier is refused
//!   ([`BlockFault::Ipc`]): nothing here takes IPC locks yet (`memory::block_map`, "Not ported
//!   yet"), so no live address space has any. The snapshot version changes when they land.
//!   Test: `a_block_with_ipc_lock_state_is_refused`.
//! - **Not checked here:** which transfer memory holds a `Locked` range (the transfer memory
//!   objects are the object table's), that a shared mapping's object is the one the guest mapped
//!   beyond its size (the caller resolves it), where the main thread's stack is (only its size
//!   is saved, and other memory may have the `Stack` state), where the `Normal` and
//!   `Locked` merge barriers sit, and which merge barrier bits a block has beyond the IPC and
//!   device ones it checks (an unknown barrier bit restores as it is).
//! - **Not checked here, for the aliases:** that every block shaped like an alias's source is
//!   one. A transfer memory lent with no permission leaves its owner's memory the same way
//!   (`Normal`, `KernelRead | NotMapped`, `Locked`), so such a block that no alias names
//!   restores as it is, and `UnmapMemory` of it is refused with `InvalidMemoryRegion`, as
//!   Eden's page group check refuses pages that are not the source's
//!   (`a_restored_alias_shape_is_not_an_alias`, in `memory/space/alias_tests.rs`). Likewise
//!   not checked: that an alias's source is not instead such a lend (the transfer memory
//!   objects are the object table's); that every `Stack` block is an alias (the main thread's
//!   stack is `Stack` too, and a block may hold an alias and plain stack pages both:
//!   `a_block_holding_two_aliases_and_a_block_partly_aliased_restore`); and that two runs
//!   that could be one (side by side in both ranges) were saved as one: the record matches
//!   page by page either way.
//! - **Not checked here: which permission and attributes each state may carry**, beyond the two
//!   permission rules of check 4 and the attribute rules above (no unknown bit, no IPC lock,
//!   `DeviceShared` exactly with a use count). So a restore accepts, for example, `Normal` or
//!   `CodeData` memory that is read and execute (the kernel's
//!   [`AddressSpace::set_memory_permission`] allows it; only the system call refuses it:
//!   `IsValidSetMemoryPermission` in `eden/src/core/hle/kernel/svc/svc_memory.cpp`), memory
//!   whose state has no `CanChangeAttribute` that is `Uncached` or `GpuAccessible`, memory whose
//!   state has no `CanPermissionLock` that is `PermissionLocked`, and `Locked` memory outside a
//!   transfer memory lend. A live address space has none of these; a snapshot that has one
//!   restores it as it is.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use starquake_memory::PAGE_SIZE;

use super::{
    AddressSpace, DEVICE_SHARE_LOCK_STATE, DEVICE_SHARE_UNLOCK_STATE, MemoryLimit, ProcessMemory,
    SharedMapping, can_contain, layout, max_process_memory,
};
use crate::id::ObjectId;
use crate::memory::block::{BlockCounts, MemoryBlock};
use crate::memory::block_map::{BlockMap, BlockMapError};
use crate::memory::layout::{LayoutError, Region, RegionOffsets, Regions};
use crate::memory::mapper::{Mapper, MapperError};
use crate::memory::pool::MemoryArrangement;
use crate::memory::shared::SharedMemory;
use crate::memory::state::{
    DisableMerge, MemoryAttribute, MemoryPermission, MemoryState, SvcPermission, flag,
};
use crate::snapshot::{Format, FormatError, decode, encode};

mod alias;

pub use alias::AliasFault;
use alias::SavedAlias;

#[cfg(test)]
mod tests;

/// First bytes of every address space snapshot.
pub const ADDRESS_SPACE_MAGIC: [u8; 4] = *b"SQAS";

/// Layout version written into every address space snapshot and required when reading one.
///
/// Version 1 had no aliases; version 2 appends them to the body (module docs, "Format").
pub const ADDRESS_SPACE_VERSION: u32 = 2;

/// The address space snapshot's header.
const FORMAT: Format = Format {
    name: "address space",
    magic: ADDRESS_SPACE_MAGIC,
    version: ADDRESS_SPACE_VERSION,
};

/// The tag each memory arrangement is saved as: starquake's own numbers, not the reference's
/// `Smc::MemoryArrangement` values. Neither is zero, so the golden-bytes test sees each.
pub const ARRANGEMENT_TAGS: [(MemoryArrangement, u8); 2] = [
    (MemoryArrangement::FourGb, 1),
    (MemoryArrangement::TwelveGb, 2),
];

/// Every permission bit starquake names ([`MemoryPermission`]): the user's read, write and
/// execute, the kernel's read and write, and `NotMapped`. Eden's `KernelExecute` (bit 5) is not
/// among them: nothing in starquake or in `ConvertToKMemoryPermission` sets it, so a block that
/// has it is refused with the other unknown bits ([`BlockFault::Bits`]).
const KNOWN_PERMISSION_BITS: u8 = MemoryPermission::USER_MASK.bits()
    | MemoryPermission::KERNEL_READ.bits()
    | MemoryPermission::KERNEL_WRITE.bits()
    | MemoryPermission::NOT_MAPPED.bits();

/// Every attribute bit starquake names ([`MemoryAttribute`]): Eden's five (`KMemoryAttribute`)
/// and the Switch 2's `GpuAccessible`.
const KNOWN_ATTRIBUTE_BITS: u8 = MemoryAttribute::LOCKED.bits()
    | MemoryAttribute::IPC_LOCKED.bits()
    | MemoryAttribute::DEVICE_SHARED.bits()
    | MemoryAttribute::UNCACHED.bits()
    | MemoryAttribute::PERMISSION_LOCKED.bits()
    | MemoryAttribute::GPU_ACCESSIBLE.bits();

/// What went wrong with one block ([`SnapshotError::Block`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BlockFault {
    /// Not page aligned, no pages, or its end overflows.
    #[error("is not a run of whole pages")]
    Shape,
    /// It starts before the end of the one before (or the first before the address space).
    #[error("overlaps the one before it")]
    Overlap,
    /// It starts after the end of the one before (or the first after the address space's start).
    #[error("leaves a gap after the one before it")]
    Gap,
    /// It ends past the end of the address space.
    #[error("runs past the end of the address space")]
    PastEnd,
    /// Its state word is no memory state.
    #[error("has the unknown state word {word:#x}")]
    State {
        /// The word.
        word: u32,
    },
    /// Its permission or attributes have a bit starquake does not name
    /// ([`KNOWN_PERMISSION_BITS`], [`KNOWN_ATTRIBUTE_BITS`]).
    #[error("has a permission or attribute bit no memory block has")]
    Bits,
    /// It could be merged with the one before, which the block map never leaves.
    #[error("could be merged with the one before it")]
    Mergeable,
    /// It is free and has a permission, attributes or counts.
    #[error("is free but has a permission, attributes or counts")]
    Free,
    /// It has IPC lock state (see "Divergences" in the module docs).
    #[error("has IPC lock state, which this build never makes")]
    Ipc,
    /// Its device counts disagree with its attribute, its barriers or each other.
    #[error("has device counts that disagree with its attribute or barriers")]
    DeviceCounts,
    /// Its permission has no translation (write without read).
    #[error("has a permission with no translation")]
    Translation,
    /// It is not free and its permission is not one the kernel gives its state: not what
    /// `ConvertToKMemoryPermission` makes of one of the four [`SvcPermission`]s, or writable
    /// code (a state with [`flag::CODE`]).
    #[error("has a permission the kernel never gives its state")]
    Permission,
    /// It is not in the region its state belongs in.
    #[error("is outside the region of its state")]
    Region,
}

/// What went wrong with one shared memory mapping ([`SnapshotError::Shared`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SharedFault {
    /// It does not start after the one before.
    #[error("is out of order or repeats the one before it")]
    Order,
    /// No `Shared` block is exactly its range.
    #[error("is not exactly one shared memory block")]
    NotABlock,
    /// The caller has no shared memory for its object.
    #[error("names a shared memory that does not exist")]
    Missing,
    /// Its object's shared memory is not its size.
    #[error("names a shared memory of another size")]
    Size,
    /// Its block's permission is not its object's user permission.
    #[error("is mapped with a permission other than its shared memory's")]
    Permission,
}

/// What went wrong with one device-share lock ([`SnapshotError::Lock`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LockFault {
    /// Its count is zero: a range with no live lock is not saved.
    #[error("has a count of zero")]
    Count,
    /// It is not after the one before, by address then size.
    #[error("is out of order or repeats the one before it")]
    Order,
    /// It is not whole pages inside the address space.
    #[error("is not whole pages inside the address space")]
    Range,
    /// It does not start where a block starts, or end where one ends.
    #[error("does not start and end where blocks do")]
    Unaligned,
    /// A block of its range is not device shared.
    #[error("covers a block that is not device shared")]
    NotShared,
    /// A block of its range has a state that cannot be device-share locked and unlocked
    /// ([`DEVICE_SHARE_LOCK_STATE`], [`DEVICE_SHARE_UNLOCK_STATE`]).
    #[error("covers a block whose memory state cannot be shared with a device")]
    State,
}

/// Why an address space could not be saved or restored.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SnapshotError {
    /// The header or the body's encoding ([`crate::snapshot::decode`]).
    #[error(transparent)]
    Format(#[from] FormatError),
    /// The arrangement tag is not one of [`ARRANGEMENT_TAGS`].
    #[error("the address space snapshot has the unknown memory arrangement tag {tag}")]
    Arrangement {
        /// The tag found.
        tag: u8,
    },
    /// The saved program cannot be laid out.
    #[error("the address space snapshot's program cannot be laid out: {0}")]
    Layout(LayoutError),
    /// The saved regions are not where the layout puts them.
    #[error("the address space snapshot's regions are not where its program's layout puts them")]
    Regions,
    /// The main thread's stack is not whole pages, or it and the code do not fit.
    #[error("the address space snapshot's main thread stack of {size:#x} bytes does not fit")]
    Stack {
        /// The saved stack size.
        size: u64,
    },
    /// The largest heap is not what the stack left.
    #[error(
        "the address space snapshot's largest heap of {size:#x} bytes is not what the stack left"
    )]
    MaxHeap {
        /// The saved largest heap.
        size: u64,
    },
    /// The heap is not whole pages, or is larger than the largest heap.
    #[error("the address space snapshot's heap of {size:#x} bytes does not fit")]
    Heap {
        /// The saved heap size.
        size: u64,
    },
    /// More is reserved than the pool holds.
    #[error("the address space snapshot reserves {reserved:#x} bytes of a pool of {limit:#x}")]
    OverLimit {
        /// The saved reservation.
        reserved: u64,
        /// The arrangement's pool.
        limit: u64,
    },
    /// The reservation is not what the code, stack, heap and mapped physical memory use.
    #[error("the address space snapshot reserves {reserved:#x} bytes but uses {used:#x}")]
    Reservation {
        /// The saved reservation.
        reserved: u64,
        /// What the saved sizes add up to (at most `u64::MAX`).
        used: u64,
    },
    /// A block is wrong.
    #[error("block {index} of the address space snapshot {fault}")]
    Block {
        /// Its position in the list.
        index: usize,
        /// What is wrong with it.
        fault: BlockFault,
    },
    /// The blocks end before the address space does.
    #[error(
        "the address space snapshot's blocks stop at {end:#x}, short of the address space's end"
    )]
    Short {
        /// Where the last block ends (the address space's start when there are none).
        end: u64,
    },
    /// A page of the code is not code.
    #[error("the address space snapshot's code is not all code memory")]
    CodeRange,
    /// A page of the heap is not heap, or the heap region holds heap memory past the heap.
    #[error("the address space snapshot's heap is not exactly its heap region's heap memory")]
    HeapRange,
    /// The mapped physical memory size is not what the alias region's `Normal` blocks hold.
    #[error(
        "the address space snapshot counts {saved:#x} bytes of mapped physical memory, but its blocks hold {mapped:#x}"
    )]
    Physical {
        /// The saved size.
        saved: u64,
        /// The bytes of `Normal` memory inside the alias region.
        mapped: u64,
    },
    /// A shared memory mapping is wrong.
    #[error("shared memory mapping {index} of the address space snapshot {fault}")]
    Shared {
        /// Its position in the list.
        index: usize,
        /// What is wrong with it.
        fault: SharedFault,
    },
    /// A `Shared` block no mapping names.
    #[error("block {block} of the address space snapshot is shared memory no mapping names")]
    Unmapped {
        /// The block's position in the list.
        block: usize,
    },
    /// A device-share lock is wrong.
    #[error("device-share lock {index} of the address space snapshot {fault}")]
    Lock {
        /// Its position in the list.
        index: usize,
        /// What is wrong with it.
        fault: LockFault,
    },
    /// A block's device counts are not what the locks add up to.
    #[error(
        "block {block} of the address space snapshot has device counts its locks do not add up to"
    )]
    DeviceCounts {
        /// The block's position in the list.
        block: usize,
    },
    /// The new translation refused a mapping.
    #[error(
        "the new translation refused to map {size:#x} bytes at {address:#x} ({error}); {stuck:#x} bytes it had mapped could not be unmapped"
    )]
    Translation {
        /// Where the refused mapping starts.
        address: u64,
        /// Its size.
        size: u64,
        /// The refusal.
        error: MapperError,
        /// Bytes mapped before it whose unmap was refused too.
        stuck: u64,
    },
    /// The checked blocks did not make a block map: a bug in the checks.
    #[error("the address space snapshot's blocks do not make a block map: {0}")]
    BlockMap(BlockMapError),
    /// A `MapMemory` alias is wrong.
    #[error("alias {index} of the address space snapshot {fault}")]
    Alias {
        /// Its position in the list.
        index: usize,
        /// What is wrong with it.
        fault: AliasFault,
    },
}

/// The creation request as saved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SavedRequest {
    code_address: u64,
    code_size: u64,
    /// One of [`ARRANGEMENT_TAGS`].
    arrangement: u8,
    alias_region_extra: bool,
    /// Alias, heap, stack, kernel map.
    offsets: [u64; 4],
}

/// A region as saved: start, end.
type SavedRegion = (u64, u64);

/// The regions as saved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SavedRegions {
    address_space: SavedRegion,
    code: SavedRegion,
    alias: SavedRegion,
    heap: SavedRegion,
    stack: SavedRegion,
    kernel_map: SavedRegion,
    alias_region_extra_size: u64,
}

/// A block as saved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct SavedBlock {
    address: u64,
    pages: u64,
    /// [`MemoryState::word`].
    state: u32,
    permission: u8,
    original_permission: u8,
    attribute: u8,
    ipc_lock_count: u16,
    device_use_count: u16,
    ipc_disable_merge_count: u16,
    device_disable_merge_left_count: u16,
    device_disable_merge_right_count: u16,
    disable_merge: u8,
}

/// A shared memory mapping as saved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct SavedShared {
    address: u64,
    size: u64,
    object: ObjectId,
}

/// The live device-share locks of one exact range, as saved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct SavedLock {
    address: u64,
    size: u64,
    count: u32,
}

/// The snapshot body (module docs, "Format").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Body {
    request: SavedRequest,
    regions: SavedRegions,
    heap_size: u64,
    max_heap_size: u64,
    reserved: u64,
    main_thread_stack_size: u64,
    mapped_physical_memory_size: u64,
    blocks: Vec<SavedBlock>,
    shared: Vec<SavedShared>,
    locks: Vec<SavedLock>,
    aliases: Vec<SavedAlias>,
}

/// What checks 2 to 8 found: what a restore builds from.
struct Checked {
    memory: ProcessMemory,
    regions: Regions,
    blocks: Vec<MemoryBlock>,
}

fn saved_region(region: Region) -> SavedRegion {
    (region.start, region.end)
}

const fn region((start, end): SavedRegion) -> Region {
    Region { start, end }
}

impl SavedBlock {
    fn of(block: &MemoryBlock) -> Self {
        let counts = block.counts();
        Self {
            address: block.address(),
            pages: block.pages(),
            state: block.state().word(),
            permission: block.permission().bits(),
            original_permission: block.original_permission().bits(),
            attribute: block.attribute().bits(),
            ipc_lock_count: counts.ipc_lock,
            device_use_count: counts.device_use,
            ipc_disable_merge_count: counts.ipc_disable_merge,
            device_disable_merge_left_count: counts.device_disable_merge_left,
            device_disable_merge_right_count: counts.device_disable_merge_right,
            disable_merge: block.disable_merge().bits(),
        }
    }
}

impl Body {
    /// The address space, as saved.
    fn of(space: &AddressSpace) -> Self {
        let memory = &space.created_from;
        let offsets = memory.offsets;
        let regions = &space.regions;
        Self {
            request: SavedRequest {
                code_address: memory.code_address,
                code_size: memory.code_size,
                arrangement: arrangement_tag(memory.arrangement),
                alias_region_extra: memory.alias_region_extra,
                offsets: [
                    offsets.alias,
                    offsets.heap,
                    offsets.stack,
                    offsets.kernel_map,
                ],
            },
            regions: SavedRegions {
                address_space: saved_region(regions.address_space),
                code: saved_region(regions.code),
                alias: saved_region(regions.alias),
                heap: saved_region(regions.heap),
                stack: saved_region(regions.stack),
                kernel_map: saved_region(regions.kernel_map),
                alias_region_extra_size: regions.alias_region_extra_size,
            },
            heap_size: space.heap_size(),
            max_heap_size: space.max_heap_size,
            reserved: space.limit.reserved(),
            main_thread_stack_size: space.main_thread_stack_size,
            mapped_physical_memory_size: space.mapped_physical_memory_size,
            blocks: space.blocks.blocks().map(SavedBlock::of).collect(),
            shared: space
                .shared
                .iter()
                .map(|(&address, mapping)| SavedShared {
                    address,
                    size: mapping.size,
                    object: mapping.object,
                })
                .collect(),
            locks: space
                .device_locks
                .iter()
                .map(|(&(address, size), &count)| SavedLock {
                    address,
                    size,
                    count,
                })
                .collect(),
            aliases: alias::saved(&space.aliases),
        }
    }

    /// Checks 2 to 8 of the module docs, in order.
    fn check(&self) -> Result<Checked, SnapshotError> {
        let (memory, regions) = self.check_layout()?;
        self.check_counts(&memory, &regions)?;
        let blocks = self.check_blocks(&regions)?;
        self.check_ranges(&memory, &regions, &blocks)?;
        self.check_shared(&blocks)?;
        self.check_locks(&regions, &blocks)?;
        alias::check(&self.aliases, &regions, &blocks)?;
        Ok(Checked {
            memory,
            regions,
            blocks,
        })
    }

    /// Check 2: the arrangement, the layout and the regions.
    fn check_layout(&self) -> Result<(ProcessMemory, Regions), SnapshotError> {
        let request = &self.request;
        let arrangement = arrangement_of(request.arrangement)?;
        let [alias, heap, stack, kernel_map] = request.offsets;
        let memory = ProcessMemory {
            code_address: request.code_address,
            code_size: request.code_size,
            arrangement,
            alias_region_extra: request.alias_region_extra,
            offsets: RegionOffsets {
                alias,
                heap,
                stack,
                kernel_map,
            },
        };
        let regions = layout(&memory).map_err(SnapshotError::Layout)?;
        let saved = &self.regions;
        let saved = Regions {
            address_space: region(saved.address_space),
            code: region(saved.code),
            alias: region(saved.alias),
            heap: region(saved.heap),
            stack: region(saved.stack),
            kernel_map: region(saved.kernel_map),
            alias_region_extra_size: saved.alias_region_extra_size,
        };
        if saved != regions {
            return Err(SnapshotError::Regions);
        }
        Ok((memory, regions))
    }

    /// Check 3: the stack, the largest heap, the heap and the reservation.
    fn check_counts(&self, memory: &ProcessMemory, regions: &Regions) -> Result<(), SnapshotError> {
        let max = max_process_memory(regions);
        let stack = self.main_thread_stack_size;
        // `KProcess::Run`: the stack and the code together within the largest process memory.
        let fixed = memory.code_size.checked_add(stack);
        let stack_fits = fixed.is_some_and(|fixed| fixed <= max);
        if !stack.is_multiple_of(PAGE_SIZE) || (stack != 0 && !stack_fits) {
            return Err(SnapshotError::Stack { size: stack });
        }
        // `m_max_heap_size = m_max_process_memory - (stack + code)`, set once with the stack; zero
        // before. A stack of no bytes leaves no size behind, so either is then possible.
        let left = fixed.and_then(|fixed| max.checked_sub(fixed));
        let max_heap = self.max_heap_size;
        let max_heap_fits = Some(max_heap) == left || (stack == 0 && max_heap == 0);
        if !max_heap_fits {
            return Err(SnapshotError::MaxHeap { size: max_heap });
        }
        let heap = self.heap_size;
        if !heap.is_multiple_of(PAGE_SIZE) || heap > max_heap {
            return Err(SnapshotError::Heap { size: heap });
        }
        let limit = memory.arrangement.application_pool_size();
        let reserved = self.reserved;
        if reserved > limit {
            return Err(SnapshotError::OverLimit { reserved, limit });
        }
        let used = memory
            .code_size
            .saturating_add(stack)
            .saturating_add(heap)
            .saturating_add(self.mapped_physical_memory_size);
        if reserved != used {
            return Err(SnapshotError::Reservation { reserved, used });
        }
        Ok(())
    }

    /// Check 4: each block, then that they reach the end.
    fn check_blocks(&self, regions: &Regions) -> Result<Vec<MemoryBlock>, SnapshotError> {
        let space = regions.address_space;
        let mut expected = space.start;
        let mut blocks: Vec<MemoryBlock> = Vec::with_capacity(self.blocks.len());
        for (index, saved) in self.blocks.iter().enumerate() {
            let fault = |fault| SnapshotError::Block { index, fault };
            let block = check_block(saved, expected, space.end, blocks.last()).map_err(fault)?;
            let in_region = block.state() == MemoryState::Free || in_region(regions, &block);
            if !in_region {
                return Err(fault(BlockFault::Region));
            }
            expected = block.end().unwrap_or(u64::MAX);
            blocks.push(block);
        }
        if expected != space.end {
            return Err(SnapshotError::Short { end: expected });
        }
        Ok(blocks)
    }

    /// Check 5: the code, the heap and the mapped physical memory against the blocks.
    fn check_ranges(
        &self,
        memory: &ProcessMemory,
        regions: &Regions,
        blocks: &[MemoryBlock],
    ) -> Result<(), SnapshotError> {
        let code = Region {
            start: memory.code_address,
            end: memory.code_address.saturating_add(memory.code_size),
        };
        let is_code = |state| matches!(state, MemoryState::Code | MemoryState::CodeData);
        if !every_block(blocks, code, is_code) {
            return Err(SnapshotError::CodeRange);
        }
        // The heap's pages are all heap memory, and the heap region holds no more.
        let heap = Region {
            start: regions.heap.start,
            end: regions.heap.start.saturating_add(self.heap_size),
        };
        let size = self.heap_size;
        if normal_bytes(blocks, heap) != size || normal_bytes(blocks, regions.heap) != size {
            return Err(SnapshotError::HeapRange);
        }
        let mapped = normal_bytes(blocks, regions.alias);
        let saved = self.mapped_physical_memory_size;
        if mapped != saved {
            return Err(SnapshotError::Physical { saved, mapped });
        }
        Ok(())
    }

    /// Check 6: each shared memory mapping, then that every `Shared` block is one.
    fn check_shared(&self, blocks: &[MemoryBlock]) -> Result<(), SnapshotError> {
        let mut named = BTreeSet::new();
        let mut previous = None;
        for (index, mapping) in self.shared.iter().enumerate() {
            let fault = |fault| SnapshotError::Shared { index, fault };
            if previous.is_some_and(|previous| mapping.address <= previous) {
                return Err(fault(SharedFault::Order));
            }
            previous = Some(mapping.address);
            let block = block_starting_at(blocks, mapping.address)
                .filter(|&at| {
                    blocks.get(at).is_some_and(|block| {
                        block.state() == MemoryState::Shared && block.size() == mapping.size
                    })
                })
                .ok_or_else(|| fault(SharedFault::NotABlock))?;
            named.insert(block);
        }
        let unmapped = blocks
            .iter()
            .enumerate()
            .find(|(at, block)| block.state() == MemoryState::Shared && !named.contains(at));
        if let Some((block, _)) = unmapped {
            return Err(SnapshotError::Unmapped { block });
        }
        Ok(())
    }

    /// Check 7: each device-share lock, then each block's device counts against them.
    fn check_locks(&self, regions: &Regions, blocks: &[MemoryBlock]) -> Result<(), SnapshotError> {
        let len = blocks.len();
        // How many blocks below each index are not device shared, so that a lock's range is
        // checked in two lookups, whatever its length.
        let mut unshared = Vec::with_capacity(len.saturating_add(1));
        unshared.push(0_usize);
        // Likewise, how many blocks below each index have a state a lock cannot be taken on or
        // released from.
        let mut unlockable = Vec::with_capacity(len.saturating_add(1));
        unlockable.push(0_usize);
        let lockable_state = DEVICE_SHARE_LOCK_STATE | DEVICE_SHARE_UNLOCK_STATE;
        for block in blocks {
            let below = unshared.last().copied().unwrap_or(0);
            let shared = block.attribute().contains(MemoryAttribute::DEVICE_SHARED);
            unshared.push(below + usize::from(!shared));
            let below = unlockable.last().copied().unwrap_or(0);
            let lockable = block.state().word() & lockable_state == lockable_state;
            unlockable.push(below + usize::from(!lockable));
        }
        // What the locks add up to at each block: those starting there, ending there, and (as a
        // running sum of starts minus ends) over it.
        let mut starting = vec![0_u64; len];
        let mut ending = vec![0_u64; len];
        let mut previous = None;
        for (index, lock) in self.locks.iter().enumerate() {
            let fault = |fault| SnapshotError::Lock { index, fault };
            if lock.count == 0 {
                return Err(fault(LockFault::Count));
            }
            let key = (lock.address, lock.size);
            if previous.is_some_and(|previous| key <= previous) {
                return Err(fault(LockFault::Order));
            }
            previous = Some(key);
            let whole = lock.address.is_multiple_of(PAGE_SIZE)
                && lock.size.is_multiple_of(PAGE_SIZE)
                && regions.contains(lock.address, lock.size);
            if !whole {
                return Err(fault(LockFault::Range));
            }
            // Inside the address space, so the end does not overflow.
            let end = lock.address.saturating_add(lock.size);
            let first = block_starting_at(blocks, lock.address);
            let last = block_ending_at(blocks, end);
            let (Some(first), Some(last)) = (first, last) else {
                return Err(fault(LockFault::Unaligned));
            };
            let after = last.saturating_add(1);
            let not_shared = unshared.get(after).copied().unwrap_or(0)
                > unshared.get(first).copied().unwrap_or(0);
            if not_shared {
                return Err(fault(LockFault::NotShared));
            }
            let not_lockable = unlockable.get(after).copied().unwrap_or(0)
                > unlockable.get(first).copied().unwrap_or(0);
            if not_lockable {
                return Err(fault(LockFault::State));
            }
            let count = u64::from(lock.count);
            if let Some(at) = starting.get_mut(first) {
                *at = at.saturating_add(count);
            }
            if let Some(at) = ending.get_mut(last) {
                *at = at.saturating_add(count);
            }
        }
        let mut over = 0_u64;
        for (block, ((saved, &starts), &ends)) in
            blocks.iter().zip(&starting).zip(&ending).enumerate()
        {
            over = over.saturating_add(starts);
            let counts = saved.counts();
            let agree = over == u64::from(counts.device_use)
                && starts == u64::from(counts.device_disable_merge_left)
                && ends == u64::from(counts.device_disable_merge_right);
            if !agree {
                return Err(SnapshotError::DeviceCounts { block });
            }
            over = over.saturating_sub(ends);
        }
        Ok(())
    }
}

/// The tag `arrangement` is saved as.
fn arrangement_tag(arrangement: MemoryArrangement) -> u8 {
    ARRANGEMENT_TAGS
        .iter()
        .find(|(known, _)| *known == arrangement)
        .map_or(0, |&(_, tag)| tag)
}

/// The arrangement saved as `tag`.
fn arrangement_of(tag: u8) -> Result<MemoryArrangement, SnapshotError> {
    ARRANGEMENT_TAGS
        .iter()
        .find(|(_, known)| *known == tag)
        .map(|&(arrangement, _)| arrangement)
        .ok_or(SnapshotError::Arrangement { tag })
}

/// The block a saved one describes, after the per-block checks of check 4 but the region (in
/// their order): `expected` is where it must start, `end` the address space's end, and `below`
/// the block before it.
fn check_block(
    saved: &SavedBlock,
    expected: u64,
    end: u64,
    below: Option<&MemoryBlock>,
) -> Result<MemoryBlock, BlockFault> {
    let block_end = saved
        .pages
        .checked_mul(PAGE_SIZE)
        .and_then(|size| saved.address.checked_add(size));
    let Some(block_end) =
        block_end.filter(|_| saved.address.is_multiple_of(PAGE_SIZE) && saved.pages != 0)
    else {
        return Err(BlockFault::Shape);
    };
    if saved.address < expected {
        return Err(BlockFault::Overlap);
    }
    if saved.address > expected {
        return Err(BlockFault::Gap);
    }
    if block_end > end {
        return Err(BlockFault::PastEnd);
    }
    let state = MemoryState::ALL
        .iter()
        .copied()
        .find(|state| state.word() == saved.state)
        .ok_or(BlockFault::State { word: saved.state })?;
    let unknown_bits = saved.permission & !KNOWN_PERMISSION_BITS != 0
        || saved.attribute & !KNOWN_ATTRIBUTE_BITS != 0;
    if unknown_bits {
        return Err(BlockFault::Bits);
    }
    let permission = MemoryPermission::from_bits(saved.permission);
    let original_permission = MemoryPermission::from_bits(saved.original_permission);
    let attribute = MemoryAttribute::from_bits(saved.attribute);
    let disable_merge = DisableMerge::from_bits(saved.disable_merge);
    let counts = BlockCounts {
        ipc_lock: saved.ipc_lock_count,
        device_use: saved.device_use_count,
        ipc_disable_merge: saved.ipc_disable_merge_count,
        device_disable_merge_left: saved.device_disable_merge_left_count,
        device_disable_merge_right: saved.device_disable_merge_right_count,
    };
    let block = MemoryBlock::restored(
        saved.address,
        saved.pages,
        (state, permission, attribute),
        original_permission,
        counts,
        disable_merge,
    )
    .ok_or(BlockFault::Shape)?;
    if below.is_some_and(|below| below.can_merge_with(&block)) {
        return Err(BlockFault::Mergeable);
    }
    let nothing = permission == MemoryPermission::NONE
        && attribute == MemoryAttribute::NONE
        && original_permission == MemoryPermission::NONE
        && counts == BlockCounts::default();
    if state == MemoryState::Free && !nothing {
        return Err(BlockFault::Free);
    }
    let ipc = counts.ipc_lock != 0
        || counts.ipc_disable_merge != 0
        || original_permission != MemoryPermission::NONE
        || attribute.contains(MemoryAttribute::IPC_LOCKED)
        || disable_merge.contains(DisableMerge::IPC_LEFT);
    if ipc {
        return Err(BlockFault::Ipc);
    }
    let shared = attribute.contains(MemoryAttribute::DEVICE_SHARED);
    let device_agrees = shared == (counts.device_use != 0)
        && disable_merge.contains(DisableMerge::DEVICE_LEFT)
            == (counts.device_disable_merge_left != 0)
        && disable_merge.contains(DisableMerge::DEVICE_RIGHT)
            == (counts.device_disable_merge_right != 0)
        && counts.device_disable_merge_left <= counts.device_use
        && counts.device_disable_merge_right <= counts.device_use;
    if !device_agrees {
        return Err(BlockFault::DeviceCounts);
    }
    if permission.guest().is_none() {
        return Err(BlockFault::Translation);
    }
    if state != MemoryState::Free && !is_live_permission(state, permission) {
        return Err(BlockFault::Permission);
    }
    Ok(block)
}

/// Whether the kernel gives memory in `state`, which is not free, `permission`: every mapping
/// call converts one of the four [`SvcPermission`]s (`ConvertToKMemoryPermission`, in
/// `eden/src/core/hle/kernel/k_memory_block.h`), and code memory never stays writable
/// (`SetProcessMemoryPermission` makes writable `Code` `CodeData` and writable `AliasCode`
/// `AliasCodeData`: `is_w` in `KPageTableBase::SetProcessMemoryPermission`,
/// `eden/src/core/hle/kernel/k_page_table_base.cpp`; and no state with [`flag::CODE`] can be
/// reprotected). See "Not checked here" in the module docs for what this leaves out.
fn is_live_permission(state: MemoryState, permission: MemoryPermission) -> bool {
    let converted = SvcPermission::ALL
        .into_iter()
        .any(|svc| MemoryPermission::from_svc(svc) == permission);
    let code = state.word() & flag::CODE != 0;
    converted && !(code && permission.contains(MemoryPermission::USER_WRITE))
}

/// Whether a block that is not free lies in the region of its state: [`can_contain`]'s rule,
/// except that `Normal` memory may also be in the alias region, where `MapPhysicalMemory` puts
/// it: each of its bytes is in one of the two.
fn in_region(regions: &Regions, block: &MemoryBlock) -> bool {
    if block.state() == MemoryState::Normal {
        let inside = overlap(block, regions.heap).saturating_add(overlap(block, regions.alias));
        return inside == block.size();
    }
    can_contain(regions, block.address(), block.size(), block.state())
}

/// How many bytes of `block` lie inside `region`.
fn overlap(block: &MemoryBlock, region: Region) -> u64 {
    let start = block.address().max(region.start);
    let end = block.end().unwrap_or(u64::MAX).min(region.end);
    end.saturating_sub(start)
}

/// Whether every block that `range` touches has a state `test` accepts. The range is not
/// empty (the layout refuses code of no bytes: `the_code_must_have_pages`). The blocks tile the
/// address space in order, so the walk starts at the block holding the range's start.
fn every_block(blocks: &[MemoryBlock], range: Region, test: impl Fn(MemoryState) -> bool) -> bool {
    blocks_touching(blocks, range).all(|block| test(block.state()))
}

/// The blocks that `range` touches, in address order. The blocks tile the address space in
/// order, so the walk starts at the block holding the range's start.
fn blocks_touching(blocks: &[MemoryBlock], range: Region) -> impl Iterator<Item = &MemoryBlock> {
    let first = blocks.partition_point(|block| block.end().is_some_and(|end| end <= range.start));
    blocks
        .get(first..)
        .unwrap_or_default()
        .iter()
        .take_while(move |block| block.address() < range.end)
}

/// How many bytes of `Normal` memory lie inside `region`.
fn normal_bytes(blocks: &[MemoryBlock], region: Region) -> u64 {
    blocks
        .iter()
        .filter(|block| block.state() == MemoryState::Normal)
        .map(|block| overlap(block, region))
        .fold(0_u64, u64::saturating_add)
}

/// The index of the block that starts at `address`.
fn block_starting_at(blocks: &[MemoryBlock], address: u64) -> Option<usize> {
    blocks
        .binary_search_by_key(&address, MemoryBlock::address)
        .ok()
}

/// The index of the block that ends at `end`.
fn block_ending_at(blocks: &[MemoryBlock], end: u64) -> Option<usize> {
    blocks
        .binary_search_by_key(&end, |block| block.end().unwrap_or(u64::MAX))
        .ok()
}

impl AddressSpace {
    /// The address space as a snapshot (module docs, "Format"), its `MapMemory` aliases with
    /// it, after checks 2 to 8 of the module docs on what it is about to write: a failure is a
    /// bug in the kernel.
    pub fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let body = Body::of(self);
        body.check()?;
        Ok(encode(FORMAT, &body)?)
    }

    /// The address space a snapshot holds, with `mapper` making its mappings real (module docs,
    /// "Checks", for what is checked and in what order). `shared` gives the shared memory an
    /// object id names, for each shared memory mapping, from the restored object table.
    ///
    /// `mapper` must be new: nothing mapped. On a refusal before check 10 it has been asked
    /// nothing; on a refusal in check 10 it has been asked to unmap what it mapped.
    pub fn restore(
        bytes: &[u8],
        mapper: Box<dyn Mapper>,
        mut shared: impl FnMut(ObjectId) -> Option<SharedMemory>,
    ) -> Result<Self, SnapshotError> {
        let body: Body = decode(FORMAT, bytes)?;
        let Checked {
            memory,
            regions,
            blocks,
        } = body.check()?;
        // Check 9: the shared memory each mapping names.
        let mut objects = BTreeMap::new();
        for (index, mapping) in body.shared.iter().enumerate() {
            let fault = |fault| SnapshotError::Shared { index, fault };
            let found = shared(mapping.object).ok_or_else(|| fault(SharedFault::Missing))?;
            if found.size() != mapping.size {
                return Err(fault(SharedFault::Size));
            }
            // Check 6 found the mapping's one `Shared` block. `KSharedMemory::Map` maps it with
            // exactly the user permission, and nothing reprotects a `Shared` block.
            let block = block_starting_at(&blocks, mapping.address).and_then(|at| blocks.get(at));
            let permission = MemoryPermission::from_svc(found.user_permission());
            if block.is_none_or(|block| block.permission() != permission) {
                return Err(fault(SharedFault::Permission));
            }
            objects.insert(mapping.address, found);
        }
        // What the address space keeps, built before the translation so that the translation
        // is the last step that can fail, and a refusal of any step leaves nothing mapped.
        let mut limit = MemoryLimit::new(memory.arrangement.application_pool_size());
        // Check 3 held the reservation within the pool.
        limit
            .reserve(body.reserved)
            .map_err(|_| SnapshotError::OverLimit {
                reserved: body.reserved,
                limit: limit.limit(),
            })?;
        let blocks = BlockMap::restored(
            regions.address_space.start,
            regions.address_space.end,
            blocks,
        )
        .map_err(SnapshotError::BlockMap)?;
        // Check 10: the translation.
        let mut mapper = mapper;
        rebuild(mapper.as_mut(), blocks.blocks(), &objects, &body.aliases)?;
        Ok(Self {
            created_from: memory,
            regions,
            blocks,
            heap_end: regions.heap.start.saturating_add(body.heap_size),
            max_heap_size: body.max_heap_size,
            max_process_memory: max_process_memory(&regions),
            limit,
            code_size: memory.code_size,
            main_thread_stack_size: body.main_thread_stack_size,
            shared: body
                .shared
                .iter()
                .map(|mapping| {
                    let saved = SharedMapping {
                        object: mapping.object,
                        size: mapping.size,
                    };
                    (mapping.address, saved)
                })
                .collect(),
            device_locks: body
                .locks
                .iter()
                .map(|lock| ((lock.address, lock.size), lock.count))
                .collect(),
            mapped_physical_memory_size: body.mapped_physical_memory_size,
            aliases: alias::restored(&body.aliases),
            mapper,
        })
    }
}

/// Check 10: maps every block that is not free through `mapper`, in address order, a `Shared`
/// one with the pages of the shared memory `objects` holds for its address, any other as fresh
/// memory without the pages `aliases` covers (one call for each part left, in address order);
/// then each of `aliases`, in address order, as its source's pages ([`Mapper::map_alias`]),
/// read-write (check 8 found every page of each destination read-write). A refusal unmaps
/// what was mapped: the aliases first, since each maps pages of a source mapped before it,
/// then the rest, each in address order.
fn rebuild<'a>(
    mapper: &mut dyn Mapper,
    blocks: impl IntoIterator<Item = &'a MemoryBlock>,
    objects: &BTreeMap<u64, SharedMemory>,
    aliases: &[SavedAlias],
) -> Result<(), SnapshotError> {
    let mut made: Vec<(u64, u64)> = Vec::new();
    for block in blocks {
        let state = block.state();
        if state == MemoryState::Free {
            continue;
        }
        let parts = if state == MemoryState::Shared {
            vec![(block.address(), block.size())]
        } else {
            alias::uncovered(block, aliases)
        };
        for (address, size) in parts {
            let result = translation(block.permission()).and_then(|perms| {
                if state != MemoryState::Shared {
                    return mapper.map(address, size, perms);
                }
                match objects.get(&address) {
                    Some(memory) => mapper.map_shared(address, memory.backing(), perms),
                    None => Err(MapperError(
                        "no shared memory was resolved for this mapping".into(),
                    )),
                }
            });
            if let Err(error) = result {
                return Err(SnapshotError::Translation {
                    address,
                    size,
                    error,
                    stuck: undo(mapper, &[], &made),
                });
            }
            made.push((address, size));
        }
    }
    let mut aliased: Vec<(u64, u64)> = Vec::new();
    for alias in aliases {
        let (address, size) = (alias.address, alias.size);
        let result = translation(MemoryPermission::USER_READ_WRITE)
            .and_then(|perms| mapper.map_alias(address, alias.source, size, perms));
        if let Err(error) = result {
            return Err(SnapshotError::Translation {
                address,
                size,
                error,
                stuck: undo(mapper, &aliased, &made),
            });
        }
        aliased.push((address, size));
    }
    Ok(())
}

/// The translation's permission for `permission`, or the refusal [`rebuild`] reports.
fn translation(permission: MemoryPermission) -> Result<starquake_memory::Perms, MapperError> {
    permission.guest().ok_or_else(|| {
        MapperError(format!(
            "permission {:#x} has no translation",
            permission.bits()
        ))
    })
}

/// Unmaps what [`rebuild`] mapped before a refusal: `aliased`, then `made`, each in the order
/// given (address order). Returns the bytes whose unmap was refused.
fn undo(mapper: &mut dyn Mapper, aliased: &[(u64, u64)], made: &[(u64, u64)]) -> u64 {
    let mut stuck = 0_u64;
    for &(at, bytes) in aliased.iter().chain(made) {
        if mapper.unmap(at, bytes).is_err() {
            stuck = stuck.saturating_add(bytes);
        }
    }
    stuck
}
