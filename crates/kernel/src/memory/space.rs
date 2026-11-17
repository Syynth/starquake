//! [`AddressSpace`]: one process's memory as the kernel keeps it, and the operations system
//! calls and process creation perform on it.
//!
//! Ported from Eden (GPL-3.0-or-later), paths relative to the reference folder:
//!
//! - `eden/src/core/hle/kernel/k_page_table_base.cpp` and `.h` @ cf654052: `GetRegionAddress`,
//!   `GetRegionSize`, `CanContain`, `Contains`, `CheckMemoryState`, `MapPages` (both forms),
//!   `SetMemoryPermission`, `SetProcessMemoryPermission`, `SetMemoryAttribute`, `SetHeapSize`,
//!   `SetMaxHeapSize`, `QueryInfo`, `GetNormalMemorySize`, `GetNumGuardPages`.
//! - `eden/src/core/hle/kernel/k_process.cpp` @ cf654052: the memory side of
//!   `KProcess::Initialize` (reserve and map the code), of `KProcess::Run` (reserve and map the
//!   main thread's stack, set the largest heap), and `GetUsedUserPhysicalMemorySize` and
//!   `GetTotalUserPhysicalMemorySize`.
//! - `eden/src/core/hle/kernel/k_page_table_base.cpp` @ cf654052, for the graphics server's
//!   device-share lock (kernel spec, "Address space"): `LockForMapDeviceAddressSpace` (as Eden
//!   PR #4588's graphics server calls it: no permission, aligned, heap checked;
//!   `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97,
//!   `BindNativeMemoryBacking`), `UnlockForDeviceAddressSpace` and
//!   `CheckMemoryStateContiguous`. `SetMemoryAttribute`'s state test for the Switch 2 attribute
//!   is Eden-NX2's and PR #4588's (`k_page_table_base.cpp` @ 50fa86bb and @ e5fb3b97): the mask
//!   `Uncached | GpuAccessible` needs `FlagCanChangeAttribute`. The GPU-accessible range query
//!   is PR #4588's walk in `BindNativeMemoryBacking` and `IGpuWorker::CreateGpuAccessibleMemory`
//!   (`eden-pr4588/src/core/hle/service/am/service/gpu_session.cpp` @ e5fb3b97).
//! - `eden/src/core/hle/kernel/k_page_table_base.cpp` @ cf654052, for shared and transfer
//!   memory: `MapPageGroup` (the form with an address) and `UnmapPageGroup`, which
//!   `KSharedMemory::Map` and `Unmap` call with `KMemoryState::Shared`;
//!   `LockForTransferMemory` and `UnlockForTransferMemory`, through `LockMemoryAndOpen` and
//!   `UnlockMemory`.
//! - `eden/src/core/hle/kernel/k_page_table_base.cpp` and `.h` @ cf654052, for the alias
//!   region's own memory: `MapPhysicalMemory`, `UnmapPhysicalMemory`, `IsInAliasRegion` and
//!   `m_mapped_physical_memory_size`, which `GetNormalMemorySize` adds. Eden-NX2's and PR
//!   #4588's copies of the three functions are Eden's, line for line
//!   (`eden-nx2/src/core/hle/kernel/k_page_table_base.cpp` @ 50fa86bb,
//!   `eden-pr4588/src/core/hle/kernel/k_page_table_base.cpp` @ e5fb3b97). What Switch 2 changes
//!   is the alias region's size, which the layout already scales (`memory::layout`; scoping
//!   doc, "Process, memory, kernel": the alias region is "up to 8x" the heap region, and a
//!   title "puts a 6.33 GiB VAMM-backed root heap in the alias region", row 80).
//!
//! Each operation checks the block map, makes the change real through the [`Mapper`], and
//! then updates the block map, in Eden's order (`Operate`, then `m_memory_block_manager.Update`).
//!
//! # What is here that Eden keeps elsewhere
//!
//! Eden keeps the code size, the main thread's stack size and the largest process memory on
//! `KProcess`, and the reservation count on its `KResourceLimit`. Starquake's process table
//! holds only facts fixed at creation so far (kernel slice 8 builds the rest), and the kernel
//! spec puts "the memory pools" with the address space, so they are here.
//!
//! # Divergences from Eden
//!
//! - **A range that is not whole pages inside the address space** is
//!   [`KernelError::InvalidCurrentMemory`] from every operation here. Eden's system calls check
//!   that before they reach the page table, with their own results
//!   ([`crate::svc::memory`] does the same), and the page table then asserts it. Test:
//!   `a_range_that_is_not_whole_pages_inside_the_address_space_is_invalid_current_memory`.
//! - **No physical pages.** Eden takes pages from the pool's page heap and can fail with
//!   `ResultOutOfMemory` when the heap has none (`MapPhysicalMemory` among others, through
//!   `AllocateForProcess`). Here the pool is a count ([`MemoryLimit`]): a reservation that fits
//!   always gets its memory, lazily backed by the execution model.
//! - **`SetMemoryAttribute` with `Uncached` does not call the [`Mapper`].** Eden re-applies
//!   the block's unchanged permission with a cache flag (`Operate` with
//!   `ChangePermissionsAndRefreshAndFlush`), which its emulated page table ignores. Test:
//!   `setting_uncached_changes_the_block_and_asks_nothing_of_the_translation`.
//! - **`SetProcessMemoryPermission` does no cache maintenance** for pages that become
//!   executable. The execution model does it on every host write to guest memory.
//! - **The largest process memory is the heap region's size**, Eden's rule for a 64-bit
//!   address space. Eden-NX2 adds the alias region's size for a Switch 2 program
//!   (`eden-nx2/src/core/hle/kernel/k_process.cpp` @ 50fa86bb, `LoadFromMetadata`); that
//!   lands with process creation (kernel slice 8), which knows what a program was built for.
//! - **A failed [`Mapper`]** ends the operation as an internal failure with nothing changed.
//!   Eden's `Operate` cannot fail on the paths ported here. Test:
//!   `a_refusing_translation_fails_the_call_and_changes_nothing`. The two physical memory
//!   calls make one [`Mapper`] call per run of pages they change, so one can fail after
//!   others succeeded; they leave the bookkeeping saying what the translation then holds.
//!   [`AddressSpace::map_physical_memory`] unmaps what it mapped and releases the memory, as
//!   Eden's `ON_RESULT_FAILURE` does, and keeps as mapped physical memory any piece whose
//!   unmap fails too (Eden asserts that unmap, `R_ASSERT`). Tests (in
//!   `memory/space/physical_tests.rs`):
//!   `a_map_refused_partway_unmaps_what_it_mapped_and_releases_the_memory` and
//!   `a_piece_whose_undo_fails_too_stays_mapped_and_counted`.
//!   [`AddressSpace::unmap_physical_memory`] frees and releases the pieces it unmapped before
//!   the refusal (Eden asserts every unmap). Test:
//!   `an_unmap_refused_partway_frees_what_it_unmapped`.
//! - **The physical memory calls' page-table half checks the range itself.** Eden's
//!   `KPageTableBase::MapPhysicalMemory` and `UnmapPhysicalMemory` trust the system call to
//!   have checked that the range lies inside the alias region. Here they check it again, with
//!   the system call's result (`ResultInvalidMemoryRegion`), after the whole-pages check above,
//!   so a range outside the alias region cannot reach the block map from any caller. Test:
//!   `the_page_table_half_checks_pages_then_the_alias_region`.
//! - **No page groups.** `UnmapPageGroup` checks that the range holds the shared memory's own
//!   pages (`IsValidPageGroup`, `ResultInvalidCurrentMemory` otherwise). With no physical pages
//!   to compare, the address space records which shared memory each mapping it made holds and
//!   where it starts, and refuses with the same result unless the range is exactly such a
//!   mapping of the same shared memory. Eden's check passes in the same cases: a mapping is the
//!   shared memory's pages in order from its first byte, so a range that starts anywhere else,
//!   or holds another shared memory, fails it. Test:
//!   `unmapping_shared_memory_needs_the_mapping_of_that_shared_memory_at_its_start`.
//!   `UnlockForTransferMemory`'s page-group check (`IsValidPageGroup`,
//!   `ResultInvalidMemoryRegion` otherwise) is not made at all: nothing but the transfer memory
//!   that locked a range can unlock it (it is the only caller), and a locked range cannot change
//!   underneath it. The operations here that change a range already in use refuse a block with
//!   [`MemoryAttribute::LOCKED`]: [`AddressSpace::set_memory_permission`],
//!   [`AddressSpace::set_process_memory_permission`], the heap shrink of
//!   [`AddressSpace::set_heap_size`], [`AddressSpace::lock_for_transfer_memory`] and
//!   [`AddressSpace::unmap_shared`] check for no attributes at all, and
//!   [`AddressSpace::set_memory_attribute`] for none outside the settable ones and
//!   `DeviceShared`; [`AddressSpace::map_shared`] and [`AddressSpace::map_pages`] (heap growth
//!   too) need the range free, which a lent range never is. Test:
//!   `svc::transfer_memory::tests::while_lent_the_owner_cannot_change_the_pages_and_closing_the_handle_gives_them_back`
//!   (`SetMemoryPermission`, `SetMemoryAttribute`, the heap shrink and a second lend, each
//!   refused while the pages are lent).
//! - **A device-share lock that would take a block's device count past `u16::MAX`** is
//!   [`KernelError::OutOfResource`] with nothing changed; Eden asserts (`ShareToDevice`:
//!   `ASSERT(new_count > 0)`). The result is starquake's own: a guess, which no reference
//!   settles because Horizon aborts. Test:
//!   `a_device_lock_past_the_largest_count_is_out_of_resource`.
//! - **A device-share unlock needs a lock taken on exactly its range.** The address space keeps
//!   a count of the live locks of each exact range it locked (the token's range; see "The
//!   device-share token" below), and an unlock of a range no live lock was taken on exactly is
//!   [`KernelError::InvalidCurrentMemory`] with nothing changed. Eden keeps no such record: it
//!   unlocks any range whose blocks are device shared, and asserts when the range ends where
//!   no lock ends (`UnshareToDevice`: `ASSERT(old_device_disable_merge_right_count > 0)`). So
//!   starquake refuses two kinds of range Eden does not unlock as asked: one that ends inside
//!   a locked block (Eden asserts), and one that starts inside a lock and ends where it ends
//!   (Eden unlocks that part). The result is the one Eden's state check gives a range that is
//!   not device shared, a guess for these cases, which Horizon either aborts on or takes.
//!   Tests: `an_unlock_that_ends_where_no_lock_ends_is_refused_and_changes_nothing` and
//!   `an_unlock_that_starts_inside_a_lock_is_refused_though_it_ends_where_the_lock_ends`.
//! - **The device-share lock does not report I/O memory.** Eden's `LockForMapDeviceAddressSpace`
//!   says whether the locked memory was `Io`; no application process here can have I/O memory
//!   (`MapIoRegion` is not planned, `crate::svc::ids`), so there is nothing to report.
//!
//! # What a device-share lock stops
//!
//! A locked range's blocks have [`MemoryAttribute::DEVICE_SHARED`] (kernel spec, rule 4). The
//! operations here that check a range for no attributes at all refuse it, so the CPU mapping
//! under memory the GPU may use cannot change: [`AddressSpace::set_memory_permission`],
//! [`AddressSpace::set_process_memory_permission`], the heap shrink of
//! [`AddressSpace::set_heap_size`] and [`AddressSpace::lock_for_transfer_memory`] (as in Eden,
//! whose checks for these pass `KMemoryAttribute::All` and expect `None`). Shared memory cannot
//! be locked at all (its state has no [`flag::CAN_ALIGNED_DEVICE_MAP`]), so
//! [`AddressSpace::unmap_shared`] never meets a lock. [`AddressSpace::set_memory_attribute`]
//! ignores it, as Eden's does (`AttributeTestMask` leaves `DeviceShared` out), and so does
//! another device-share lock, which counts. Tests (in `memory/space/device_tests.rs`):
//! `a_device_locked_range_refuses_what_would_change_its_mapping_until_unlocked` and
//! `locked_code_cannot_be_reprotected`.
//!
//! # The device-share token
//!
//! Taking a device-share lock gives a [`DeviceShareToken`] naming the process and the exact
//! range, and releasing one takes the token back (`docs/specs/graphics-server.md`, "The model":
//! the owner's ruling of 2026-11-17). The address space does not know which process it belongs
//! to, so it neither makes nor checks tokens: it locks and unlocks exact ranges
//! ([`AddressSpace::lock_device_share`] and [`AddressSpace::unlock_device_share`], both
//! crate-private, so code outside the kernel reaches them only with a token), and the
//! dispatcher makes each token and checks its process (`dispatch/device.rs`). A token's fields
//! are private and only the kernel builds one. It is `Copy`, so a holder can present it twice;
//! the second unlock is refused, because the lock it names is gone.

use core::fmt;
use std::collections::BTreeMap;

use starquake_memory::{BackingRange, PAGE_SIZE};

use super::block::MemoryBlock;
use super::block_map::{BlockMap, BlockMapError, DeviceLock, Properties};
use super::layout::{LayoutError, LayoutRequest, Region, RegionOffsets, Regions, pages_to_bytes};
use super::mapper::{Mapper, MapperError};
use super::pool::{MemoryArrangement, MemoryLimit};
use super::shared::{SharedMemory, SharedPages};
use super::state::{
    DisableMerge, MemoryAttribute, MemoryPermission, MemoryState, STATE_ALL, SvcPermission, flag,
};
use crate::id::{ObjectId, ProcessId};
use crate::result::{CallResult, KernelError, SvcError, SvcResult};

/// The state flags every block of a device-share lock's range has: what
/// [`AddressSpace::lock_device_share`] tests (`CanAlignedDeviceMap | ReferenceCounted`), and
/// what [`AddressSpace::unlock_device_share`] tests (`CanDeviceMap`). A restore refuses a lock
/// over a block without them (`snapshot`, check 7).
pub(super) const DEVICE_SHARE_LOCK_STATE: u32 =
    flag::CAN_ALIGNED_DEVICE_MAP | flag::REFERENCE_COUNTED;

/// The state flag [`AddressSpace::unlock_device_share`] tests (`CanDeviceMap`).
pub(super) const DEVICE_SHARE_UNLOCK_STATE: u32 = flag::CAN_DEVICE_MAP;

/// Free pages kept on each side of a mapping the kernel places for a process
/// (`KPageTableBase::GetNumGuardPages`, for a page table that is not the kernel's).
pub const GUARD_PAGES: u64 = 4;

/// A device-share lock as the kernel hands it out and takes it back: the process whose memory
/// is locked and the exact range (module docs, "The device-share token").
///
/// Only the kernel makes one: by taking a lock
/// ([`IpcContext::lock_device_share`](crate::dispatch::IpcContext::lock_device_share)), or by
/// checking what a snapshot saved
/// ([`Dispatcher::reclaim_device_share`](crate::dispatch::Dispatcher::reclaim_device_share)).
/// The reclaim is `pub` and takes raw numbers (a process, a range and how many releases of it
/// were reclaimed before), so it is the one way outside tests to get a token without taking a
/// lock: it is meant only for a snapshot restore, and it makes a token only for a range on
/// which the kernel holds a live lock (its module docs, "Making a token", say what it does
/// not prevent). The `test-support` feature adds an unchecked constructor for fakes
/// (`memory::testing::device_share_token`). Its fields can be read and never set. Releasing it
/// ([`Dispatcher::unlock_device_share`](crate::dispatch::Dispatcher::unlock_device_share))
/// checks it again, whichever way it was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceShareToken {
    process: ProcessId,
    address: u64,
    size: u64,
}

impl DeviceShareToken {
    /// The token for a lock on `size` bytes at `address` of `process`. Kernel-only (and the
    /// `test-support` fakes', `memory::testing::device_share_token`).
    pub(crate) const fn new(process: ProcessId, address: u64, size: u64) -> Self {
        Self {
            process,
            address,
            size,
        }
    }

    /// The process whose memory is locked.
    #[must_use]
    pub const fn process(self) -> ProcessId {
        self.process
    }

    /// Where the locked range starts.
    #[must_use]
    pub const fn address(self) -> u64 {
        self.address
    }

    /// How many bytes are locked.
    #[must_use]
    pub const fn size(self) -> u64 {
        self.size
    }
}

/// What a process's address space is created from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessMemory {
    /// Where the program's code goes.
    pub code_address: u64,
    /// How many bytes of code there are: a multiple of the page size, not zero.
    pub code_size: u64,
    /// The console's memory arrangement: it sizes the heap and alias regions, and the pool
    /// the process draws from.
    pub arrangement: MemoryArrangement,
    /// Whether the alias region gets its extra size (see
    /// [`LayoutRequest::alias_region_extra`]).
    pub alias_region_extra: bool,
    /// Where in their slack the regions go; the default is the lowest place for each.
    pub offsets: RegionOffsets,
}

/// Why an address space could not be created.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CreateError {
    /// The regions could not be laid out around the code.
    #[error(transparent)]
    Layout(#[from] LayoutError),
    /// The code could not be reserved or mapped.
    #[error("the program's code could not be mapped: {0}")]
    Code(#[from] SvcError),
}

/// One test of `CheckMemoryState`: what a range's blocks must be.
#[derive(Debug, Clone, Copy)]
struct Check {
    state_mask: u32,
    state: u32,
    permission_mask: MemoryPermission,
    permission: MemoryPermission,
    attribute_mask: MemoryAttribute,
    attribute: MemoryAttribute,
    /// Attributes in which the blocks of the range may differ from the first one.
    ignore: MemoryAttribute,
}

impl Check {
    /// The range must be free.
    const FREE: Self = Self {
        state_mask: STATE_ALL,
        state: MemoryState::Free.word(),
        permission_mask: MemoryPermission::NONE,
        permission: MemoryPermission::NONE,
        attribute_mask: MemoryAttribute::NONE,
        attribute: MemoryAttribute::NONE,
        ignore: MemoryAttribute::COUNTED,
    };

    /// The state must have `flags`, and no attribute may be set.
    const fn flagged(flags: u32) -> Self {
        Self {
            state_mask: flags,
            state: flags,
            permission_mask: MemoryPermission::NONE,
            permission: MemoryPermission::NONE,
            attribute_mask: MemoryAttribute::ALL,
            attribute: MemoryAttribute::NONE,
            ignore: MemoryAttribute::COUNTED,
        }
    }
}

/// What a passed check found: the state and permission every block of the range has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Checked {
    state: MemoryState,
    permission: MemoryPermission,
}

/// A change to the block map: the properties a range gets, and the merge barriers to set at
/// its head and to clear.
#[derive(Debug, Clone, Copy)]
struct Change {
    state: MemoryState,
    permission: MemoryPermission,
    attribute: MemoryAttribute,
    set: DisableMerge,
    clear: DisableMerge,
}

/// What the [`Mapper`] is asked to do for a change.
#[derive(Debug, Clone, Copy)]
enum Effect {
    Map,
    /// Map the pages of a shared memory ([`Mapper::map_shared`]).
    MapShared(BackingRange),
    Unmap,
    Protect,
    /// Nothing: the change is to the bookkeeping alone (`Operate` is not called).
    Keep,
}

impl Effect {
    /// The effect as a verb, for an error message.
    const fn verb(self) -> &'static str {
        match self {
            Self::Map => "map",
            Self::MapShared(_) => "map shared memory over",
            Self::Unmap => "unmap",
            Self::Protect => "reprotect",
            Self::Keep => "keep",
        }
    }
}

/// One shared memory mapping the address space made: which shared memory it holds, and how
/// many bytes. Keyed by where it starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SharedMapping {
    object: ObjectId,
    size: u64,
}

/// One process's memory blocks, regions, heap and memory accounting (kernel spec, "Tables
/// behind a dispatcher": the `AddressSpace` table).
pub struct AddressSpace {
    /// What the address space was created from: the program's code and the layout's inputs,
    /// which a snapshot saves so that a restore can lay the regions out again and compare.
    created_from: ProcessMemory,
    regions: Regions,
    blocks: BlockMap,
    /// One past the last byte of the heap (`m_current_heap_end`).
    heap_end: u64,
    /// The largest heap `SetHeapSize` allows (`m_max_heap_size`): zero until the main thread's
    /// stack exists.
    max_heap_size: u64,
    /// The most memory the process may use (`KProcess::m_max_process_memory`).
    max_process_memory: u64,
    limit: MemoryLimit,
    /// Bytes of code (`KProcess::m_code_size`).
    code_size: u64,
    /// Bytes of the main thread's stack (`KProcess::m_main_thread_stack_size`).
    main_thread_stack_size: u64,
    /// The shared memory mapped here, by where each mapping starts (see "No page groups" in
    /// the module docs).
    shared: BTreeMap<u64, SharedMapping>,
    /// The live device-share locks by exact range, `(address, size)`: how many were taken on
    /// the range and not yet released. A count is never zero; a range with none is absent.
    device_locks: BTreeMap<(u64, u64), u32>,
    /// Bytes `MapPhysicalMemory` has mapped and `UnmapPhysicalMemory` has not unmapped
    /// (`m_mapped_physical_memory_size`).
    mapped_physical_memory_size: u64,
    /// The `MapMemory` aliases, by where each run of alias pages starts (see [`alias`]).
    aliases: alias::Aliases,
    mapper: Box<dyn Mapper>,
}

/// The part of one block that lies inside a range: what Eden's physical memory calls look at
/// as they walk the blocks of their range.
#[derive(Debug, Clone, Copy)]
struct Piece {
    /// The first address of the piece: the range's start or the block's, whichever is higher.
    address: u64,
    /// The bytes of the piece.
    size: u64,
    /// The block it lies in.
    block: MemoryBlock,
}

impl Piece {
    const fn pages(&self) -> u64 {
        self.size / PAGE_SIZE
    }

    fn is_free(&self) -> bool {
        self.block.state() == MemoryState::Free
    }
}

/// Free memory: what `MapPhysicalMemory` maps, and what `UnmapPhysicalMemory` leaves.
const FREE: Properties = Properties {
    state: MemoryState::Free,
    permission: MemoryPermission::NONE,
    attribute: MemoryAttribute::NONE,
};

/// What `MapPhysicalMemory` maps: `Normal`, read-write for the user, no attributes.
const PHYSICAL: Properties = Properties {
    state: MemoryState::Normal,
    permission: MemoryPermission::USER_READ_WRITE,
    attribute: MemoryAttribute::NONE,
};

impl fmt::Debug for AddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AddressSpace")
            .field("regions", &self.regions)
            .field("blocks", &self.blocks)
            .field("heap_end", &self.heap_end)
            .field("max_heap_size", &self.max_heap_size)
            .field("max_process_memory", &self.max_process_memory)
            .field("limit", &self.limit)
            .field("code_size", &self.code_size)
            .field("main_thread_stack_size", &self.main_thread_stack_size)
            .field("shared", &self.shared)
            .field("device_locks", &self.device_locks)
            .field(
                "mapped_physical_memory_size",
                &self.mapped_physical_memory_size,
            )
            .field("aliases", &self.aliases)
            .field("created_from", &self.created_from)
            .finish_non_exhaustive()
    }
}

/// A failure of the block map, which no guest input explains.
fn broken(error: BlockMapError) -> SvcError {
    SvcError::internal(&error)
}

/// Where the regions of an address space created from `memory` go: the code checked for whole
/// pages ([`LayoutError::BadCode`]), then laid out ([`Regions::new`]). What
/// [`AddressSpace::new`] does first, and what a snapshot's restore compares the saved regions
/// with.
fn layout(memory: &ProcessMemory) -> Result<Regions, LayoutError> {
    if !memory.code_address.is_multiple_of(PAGE_SIZE) || !memory.code_size.is_multiple_of(PAGE_SIZE)
    {
        return Err(LayoutError::BadCode {
            address: memory.code_address,
            size: memory.code_size,
        });
    }
    Regions::new(&LayoutRequest {
        code_address: memory.code_address,
        code_size: memory.code_size,
        physical_memory_size: memory.arrangement.physical_memory_size(),
        alias_region_extra: memory.alias_region_extra,
        offsets: memory.offsets,
    })
}

/// The most memory a process with these regions may use: the heap region's size (see "The
/// largest process memory" in the module docs' divergences).
const fn max_process_memory(regions: &Regions) -> u64 {
    regions.heap.size()
}

/// The region memory of `state` goes in (`GetRegionAddress` and `GetRegionSize`), or `None`
/// for [`MemoryState::Inaccessible`], which has none (Eden: `UNREACHABLE`).
const fn region_for(regions: &Regions, state: MemoryState) -> Option<Region> {
    use MemoryState as S;
    Some(match state {
        S::Free | S::Kernel => regions.address_space,
        S::Normal => regions.heap,
        S::Ipc | S::NonSecureIpc | S::NonDeviceIpc => regions.alias,
        S::Stack => regions.stack,
        S::Static | S::ThreadLocal => regions.kernel_map,
        // The alias-code region, which for a 39-bit space is the code region.
        S::IoMemory
        | S::IoRegister
        | S::Shared
        | S::AliasCode
        | S::AliasCodeData
        | S::Transferred
        | S::SharedTransferred
        | S::SharedCode
        | S::GeneratedCode
        | S::CodeOut
        | S::Coverage
        | S::Insecure
        | S::Code
        | S::CodeData => regions.code,
        S::Inaccessible => return None,
    })
}

/// [`AddressSpace::can_contain`] for an address space with these regions (`CanContain`).
fn can_contain(regions: &Regions, address: u64, size: u64, state: MemoryState) -> bool {
    use MemoryState as S;
    let Some(region) = region_for(regions, state) else {
        return false;
    };
    let range = Region {
        start: address,
        end: address.saturating_add(size),
    };
    let in_region = region.holds(address, size);
    let in_heap = regions.heap.overlaps(range);
    let in_alias = regions.alias.overlaps(range);
    match state {
        S::Free | S::Kernel => in_region,
        S::Normal => in_region && !in_alias,
        S::Ipc | S::NonSecureIpc | S::NonDeviceIpc => in_region && !in_heap,
        _ => in_region && !in_heap && !in_alias,
    }
}

impl AddressSpace {
    /// The address space of a new process (`KProcess::Initialize`): the regions laid out
    /// around the code, the code's memory reserved from the application pool, and the code
    /// mapped with no access for the guest (`KMemoryState::Code`,
    /// `KernelRead | NotMapped`), ready for the loader to fill and reprotect
    /// ([`Self::set_process_memory_permission`]).
    ///
    /// Code that is not whole pages inside the code region is [`LayoutError::BadCode`], and
    /// code the pool has no room for [`KernelError::LimitReached`] inside
    /// [`CreateError::Code`].
    pub fn new(memory: &ProcessMemory, mapper: Box<dyn Mapper>) -> Result<Self, CreateError> {
        let regions = layout(memory)?;
        let blocks = BlockMap::new(regions.address_space.start, regions.address_space.end)
            .map_err(broken)?;
        let mut space = Self {
            created_from: *memory,
            regions,
            blocks,
            heap_end: regions.heap.start,
            max_heap_size: 0,
            max_process_memory: max_process_memory(&regions),
            limit: MemoryLimit::new(memory.arrangement.application_pool_size()),
            code_size: memory.code_size,
            main_thread_stack_size: 0,
            shared: BTreeMap::new(),
            device_locks: BTreeMap::new(),
            mapped_physical_memory_size: 0,
            aliases: alias::Aliases::default(),
            mapper,
        };
        // Reserve memory for our code resource.
        space
            .limit
            .reserve(memory.code_size)
            .map_err(SvcError::from)?;
        // Map the code region. (Eden first checks `CanContain(.., KMemoryState::Code)`, for
        // `ResultInvalidMemoryRegion`; the layout has already refused code outside the code
        // region, and the regions it places never overlap the code.)
        space.map_pages(
            memory.code_address,
            memory.code_size / PAGE_SIZE,
            MemoryState::Code,
            MemoryPermission::KERNEL_READ | MemoryPermission::NOT_MAPPED,
        )?;
        Ok(space)
    }

    /// Where the regions are.
    #[must_use]
    pub const fn regions(&self) -> &Regions {
        &self.regions
    }

    /// Every memory block, in address order.
    pub fn blocks(&self) -> impl Iterator<Item = &MemoryBlock> {
        self.blocks.blocks()
    }

    /// The process's memory count: what it may reserve, and what it has.
    #[must_use]
    pub const fn limit(&self) -> &MemoryLimit {
        &self.limit
    }

    /// The size of the heap in bytes.
    #[must_use]
    pub const fn heap_size(&self) -> u64 {
        self.heap_end - self.regions.heap.start
    }

    /// The largest heap `SetHeapSize` allows.
    #[must_use]
    pub const fn max_heap_size(&self) -> u64 {
        self.max_heap_size
    }

    /// Bytes of heap and of memory [`Self::map_physical_memory`] mapped (`GetNormalMemorySize`:
    /// `(m_current_heap_end - m_heap_region_start) + m_mapped_physical_memory_size`).
    #[must_use]
    pub const fn normal_memory_size(&self) -> u64 {
        self.heap_size()
            .saturating_add(self.mapped_physical_memory_size)
    }

    /// Bytes [`Self::map_physical_memory`] has mapped and [`Self::unmap_physical_memory`] has
    /// not unmapped (`m_mapped_physical_memory_size`).
    #[must_use]
    pub const fn mapped_physical_memory_size(&self) -> u64 {
        self.mapped_physical_memory_size
    }

    /// The memory the process uses (`KProcess::GetUsedUserPhysicalMemorySize`, with no secure
    /// system resource): its normal memory ([`Self::normal_memory_size`]: the heap and the
    /// memory `MapPhysicalMemory` mapped), its code and its main thread's stack.
    #[must_use]
    pub const fn used_user_physical_memory_size(&self) -> u64 {
        self.normal_memory_size()
            .saturating_add(self.code_size)
            .saturating_add(self.main_thread_stack_size)
    }

    /// The memory the process can use in all (`KProcess::GetTotalUserPhysicalMemorySize`, with
    /// no secure system resource): what it uses plus what it can still reserve, or the largest
    /// process memory if that is less.
    #[must_use]
    pub const fn total_user_physical_memory_size(&self) -> u64 {
        let used = self.used_user_physical_memory_size();
        let available = used.saturating_add(self.limit.free());
        if available > self.max_process_memory {
            self.max_process_memory
        } else {
            available
        }
    }

    /// The most memory the process may use (`KProcess::m_max_process_memory`), which
    /// [`Self::total_user_physical_memory_size`] is capped at.
    #[must_use]
    pub const fn max_process_memory(&self) -> u64 {
        self.max_process_memory
    }

    /// The region memory of `state` goes in (`GetRegionAddress` and `GetRegionSize`), or
    /// `None` for [`MemoryState::Inaccessible`], which has none (Eden: `UNREACHABLE`).
    const fn region_for(&self, state: MemoryState) -> Option<Region> {
        region_for(&self.regions, state)
    }

    /// Whether the `size` bytes at `address` may hold memory of `state` (`CanContain`): inside
    /// the state's region, and outside the heap and alias regions unless the state belongs
    /// there.
    #[must_use]
    pub fn can_contain(&self, address: u64, size: u64, state: MemoryState) -> bool {
        can_contain(&self.regions, address, size, state)
    }

    /// The page count of a range that is whole pages inside the address space, or
    /// [`KernelError::InvalidCurrentMemory`] (see "Divergences" in the module docs).
    fn pages_of(&self, address: u64, size: u64) -> SvcResult<u64> {
        if address.is_multiple_of(PAGE_SIZE)
            && size.is_multiple_of(PAGE_SIZE)
            && self.regions.contains(address, size)
        {
            Ok(size / PAGE_SIZE)
        } else {
            Err(KernelError::InvalidCurrentMemory)
        }
    }

    /// `CheckMemoryState` over a range: every block that the `size` bytes at `address` touch
    /// has the state, permission and attributes of the first one (attributes in
    /// `check.ignore` aside) and passes the masked tests. The range must lie inside the
    /// address space. A failed test is [`KernelError::InvalidCurrentMemory`].
    fn check_memory_state(&self, address: u64, size: u64, check: &Check) -> SvcResult<Checked> {
        self.walk_check(address, size, check, true)
    }

    /// `CheckMemoryStateContiguous`: every block that the `size` bytes at `address` touch
    /// passes the masked tests, each on its own (`check.ignore` is not used). The range must
    /// lie inside the address space. A failed test is [`KernelError::InvalidCurrentMemory`].
    fn check_memory_state_contiguous(
        &self,
        address: u64,
        size: u64,
        check: &Check,
    ) -> SvcResult<Checked> {
        self.walk_check(address, size, check, false)
    }

    /// The walk both checks make; `alike` adds `CheckMemoryState`'s comparison of every block
    /// with the first.
    fn walk_check(
        &self,
        address: u64,
        size: u64,
        check: &Check,
        alike: bool,
    ) -> SvcResult<Checked> {
        let invalid = KernelError::InvalidCurrentMemory;
        let last = size
            .checked_sub(1)
            .and_then(|offset| address.checked_add(offset))
            .ok_or(invalid)?;
        let mut blocks = self.blocks.blocks_from(address);
        let first = blocks.next().ok_or(invalid)?;
        let ignore = check.ignore;
        let mut block = first;
        loop {
            // Validate the current block against the first, then against the provided masks.
            let uniform = !alike
                || block.state() == first.state()
                    && block.permission() == first.permission()
                    && (block.attribute() | ignore) == (first.attribute() | ignore);
            let passes = block.state().word() & check.state_mask == check.state
                && block.permission() & check.permission_mask == check.permission
                && block.attribute() & check.attribute_mask == check.attribute;
            if !uniform || !passes {
                return Err(invalid);
            }
            if last <= block.last() {
                break;
            }
            block = blocks.next().ok_or(invalid)?;
        }
        Ok(Checked {
            state: first.state(),
            permission: first.permission(),
        })
    }

    /// Makes a change real and then records it: asks the block map whether it would take the
    /// update, calls the [`Mapper`], and updates the block map. Nothing has changed if this
    /// fails before or in the [`Mapper`].
    fn apply(
        &mut self,
        address: u64,
        pages: u64,
        change: Change,
        effect: Effect,
    ) -> CallResult<()> {
        let Change {
            state,
            permission,
            attribute,
            set,
            clear,
        } = change;
        self.blocks
            .check_update(address, pages, state, permission, attribute)
            .map_err(broken)?;
        let size = pages_to_bytes(pages).ok_or(KernelError::InvalidCurrentMemory)?;
        let perms = permission.guest().ok_or_else(|| {
            SvcError::Internal(format!(
                "permission {:#x} has no translation",
                permission.bits()
            ))
        });
        let refused = |error: MapperError| {
            SvcError::Internal(format!(
                "the translation refused to {} {size:#x} bytes at {address:#x}: {error}",
                effect.verb()
            ))
        };
        match effect {
            Effect::Map => self.mapper.map(address, size, perms?),
            Effect::MapShared(backing) => self.mapper.map_shared(address, backing, perms?),
            Effect::Unmap => self.mapper.unmap(address, size),
            Effect::Protect => self.mapper.protect(address, size, perms?),
            Effect::Keep => Ok(()),
        }
        .map_err(refused)?;
        // `check_update` passed and nothing has touched the map since, so this cannot fail.
        self.blocks
            .update(address, pages, state, permission, attribute, set, clear)
            .map_err(broken)
    }

    /// Maps `pages` fresh pages at `address` with `state` and `permission`
    /// (`MapPages(address, num_pages, state, perm)`).
    ///
    /// [`KernelError::InvalidCurrentMemory`] unless the range may hold memory of that state
    /// ([`Self::can_contain`]) and is free.
    pub fn map_pages(
        &mut self,
        address: u64,
        pages: u64,
        state: MemoryState,
        permission: MemoryPermission,
    ) -> CallResult<()> {
        let invalid = KernelError::InvalidCurrentMemory;
        let size = pages_to_bytes(pages).ok_or(invalid)?;
        self.pages_of(address, size)?;
        // Check that the map is in range.
        if !self.can_contain(address, size, state) {
            return Err(invalid.into());
        }
        self.check_memory_state(address, size, &Check::FREE)?;
        let change = Change {
            state,
            permission,
            attribute: MemoryAttribute::NONE,
            set: DisableMerge::NORMAL,
            clear: DisableMerge::NONE,
        };
        self.apply(address, pages, change, Effect::Map)
    }

    /// Maps `pages` fresh pages with `state` and `permission` at the lowest free place of the
    /// state's region that leaves [`GUARD_PAGES`] free pages on each side, and returns where
    /// (`MapPages(out_addr, num_pages, state, perm)`, without address-space randomisation).
    ///
    /// [`KernelError::InvalidCurrentMemory`] for a state with no region of its own,
    /// [`KernelError::OutOfMemory`] when `pages` is not less than the region's pages or no
    /// place is free.
    pub fn map_pages_in_region(
        &mut self,
        pages: u64,
        state: MemoryState,
        permission: MemoryPermission,
    ) -> CallResult<u64> {
        let invalid = KernelError::InvalidCurrentMemory;
        let region = self.region_for(state).ok_or(invalid)?;
        // Ensure this is a valid map request.
        if !self.can_contain(region.start, region.size(), state) {
            return Err(invalid.into());
        }
        let region_pages = region.size() / PAGE_SIZE;
        if pages >= region_pages {
            return Err(KernelError::OutOfMemory.into());
        }
        let address = self
            .blocks
            .find_free_area(region.start, region_pages, pages, PAGE_SIZE, 0, GUARD_PAGES)
            .ok_or(KernelError::OutOfMemory)?;
        self.map_pages(address, pages, state, permission)?;
        Ok(address)
    }

    /// Reserves and maps the main thread's stack and sets the largest heap (`KProcess::Run`,
    /// its memory part). `stack_size` is rounded up to whole pages. Returns the stack's top:
    /// one past its last byte, or zero for a stack of no bytes.
    ///
    /// [`KernelError::OutOfMemory`] when the stack and the code together exceed the largest
    /// process memory, [`KernelError::LimitReached`] when the pool has no room for the stack.
    /// A second call is an internal failure: Eden asserts there is no stack yet.
    pub fn create_main_thread_stack(&mut self, stack_size: u64) -> CallResult<u64> {
        if self.main_thread_stack_size != 0 {
            return Err(SvcError::Internal(
                "the main thread's stack exists already".into(),
            ));
        }
        // Ensure that we're allocating a valid stack.
        let out_of_memory = KernelError::OutOfMemory;
        let stack_size = stack_size
            .checked_next_multiple_of(PAGE_SIZE)
            .ok_or(out_of_memory)?;
        let fixed = stack_size
            .checked_add(self.code_size)
            .filter(|fixed| *fixed <= self.max_process_memory)
            .ok_or(out_of_memory)?;
        // Place a reservation of memory for our new stack; allocate and map it.
        self.limit.reserve(stack_size)?;
        let mut stack_top = 0;
        if stack_size != 0 {
            let mapped = self.map_pages_in_region(
                stack_size / PAGE_SIZE,
                MemoryState::Stack,
                MemoryPermission::USER_READ_WRITE,
            );
            match mapped {
                Ok(bottom) => stack_top = bottom + stack_size,
                Err(error) => {
                    self.release(stack_size)?;
                    return Err(error);
                }
            }
            self.main_thread_stack_size = stack_size;
        }
        // Set our maximum heap size.
        self.max_heap_size = self.max_process_memory - fixed;
        Ok(stack_top)
    }

    /// Gives `bytes` back to the pool. More than was reserved is the kernel's own miscount.
    fn release(&mut self, bytes: u64) -> CallResult<()> {
        self.limit.release(bytes).ok_or_else(|| {
            SvcError::Internal(format!(
                "releasing {bytes:#x} bytes of which only {:#x} are reserved",
                self.limit.reserved()
            ))
        })
    }

    /// Changes what the guest may do with code memory (`SetProcessMemoryPermission`): what the
    /// loader does to each segment of a module after writing it. Code that becomes writable
    /// becomes code data.
    ///
    /// [`KernelError::InvalidCurrentMemory`] unless the range is code memory (a state with
    /// [`flag::CODE`]) with one state and permission throughout and no attributes.
    pub fn set_process_memory_permission(
        &mut self,
        address: u64,
        size: u64,
        permission: SvcPermission,
    ) -> CallResult<()> {
        let pages = self.pages_of(address, size)?;
        // Verify we can change the memory permission.
        let old = self.check_memory_state(address, size, &Check::flagged(flag::CODE))?;
        // Determine new perm/state.
        let new_permission = MemoryPermission::from_svc(permission);
        let writable = new_permission.contains(MemoryPermission::USER_WRITE);
        let new_state = match (writable, old.state) {
            (true, MemoryState::Code) => MemoryState::CodeData,
            (true, MemoryState::AliasCode) => MemoryState::AliasCodeData,
            (_, state) => state,
        };
        // Succeed if there's nothing to do.
        if old.permission == new_permission && old.state == new_state {
            return Ok(());
        }
        let change = Change {
            state: new_state,
            permission: new_permission,
            attribute: MemoryAttribute::NONE,
            set: DisableMerge::NONE,
            clear: DisableMerge::NONE,
        };
        self.apply(address, pages, change, Effect::Protect)
    }

    /// Changes what the guest may do with its own data (`SetMemoryPermission`).
    ///
    /// [`KernelError::InvalidCurrentMemory`] unless the range is memory that may be
    /// reprotected (a state with [`flag::CAN_REPROTECT`]) with one state and permission
    /// throughout and no attributes.
    pub fn set_memory_permission(
        &mut self,
        address: u64,
        size: u64,
        permission: SvcPermission,
    ) -> CallResult<()> {
        let pages = self.pages_of(address, size)?;
        // Verify we can change the memory permission.
        let old = self.check_memory_state(address, size, &Check::flagged(flag::CAN_REPROTECT))?;
        // Determine new perm.
        let new_permission = MemoryPermission::from_svc(permission);
        if old.permission == new_permission {
            return Ok(());
        }
        let change = Change {
            state: old.state,
            permission: new_permission,
            attribute: MemoryAttribute::NONE,
            set: DisableMerge::NONE,
            clear: DisableMerge::NONE,
        };
        self.apply(address, pages, change, Effect::Protect)
    }

    /// Replaces the attributes in `mask` with `attribute` (`SetMemoryAttribute`). `mask` must
    /// lie inside [`MemoryAttribute::SET_MASK`] and `attribute` inside `mask`: the system
    /// call checks both, and anything else is an internal failure here.
    ///
    /// [`KernelError::InvalidCurrentMemory`] unless the range has one state and permission
    /// throughout, the state allows each attribute in the mask to change
    /// ([`flag::CAN_CHANGE_ATTRIBUTE`] for `Uncached` and `GpuAccessible`,
    /// [`flag::CAN_PERMISSION_LOCK`] for `PermissionLocked`), and no attribute other than those
    /// three and `DeviceShared` is set.
    pub fn set_memory_attribute(
        &mut self,
        address: u64,
        size: u64,
        mask: MemoryAttribute,
        attribute: MemoryAttribute,
    ) -> CallResult<()> {
        if !MemoryAttribute::SET_MASK.contains(mask) || !mask.contains(attribute) {
            return Err(SvcError::Internal(format!(
                "attribute {:#x} under mask {:#x} is not settable",
                attribute.bits(),
                mask.bits()
            )));
        }
        let pages = self.pages_of(address, size)?;
        // Verify we can change the memory attribute.
        let test_mask = !(MemoryAttribute::SET_MASK | MemoryAttribute::DEVICE_SHARED);
        let state_test = [
            (
                MemoryAttribute::UNCACHED | MemoryAttribute::GPU_ACCESSIBLE,
                flag::CAN_CHANGE_ATTRIBUTE,
            ),
            (
                MemoryAttribute::PERMISSION_LOCKED,
                flag::CAN_PERMISSION_LOCK,
            ),
        ]
        .into_iter()
        .filter(|(bits, _)| !mask.excludes(*bits))
        .fold(0, |flags, (_, flag)| flags | flag);
        let check = Check {
            state_mask: state_test,
            state: state_test,
            permission_mask: MemoryPermission::NONE,
            permission: MemoryPermission::NONE,
            attribute_mask: test_mask,
            attribute: MemoryAttribute::NONE,
            ignore: !test_mask,
        };
        self.check_memory_state(address, size, &check)?;
        // Update the blocks. (See "Divergences" in the module docs for `Uncached`.)
        self.blocks
            .update_attribute(address, pages, mask, attribute)
            .map_err(broken)
    }

    /// Sets the heap's size and returns where the heap starts (`SetHeapSize`). The system
    /// call checks the size's alignment first.
    ///
    /// [`KernelError::OutOfMemory`] when `size` exceeds the heap region or the largest heap,
    /// [`KernelError::LimitReached`] when the pool has no room for the growth, and
    /// [`KernelError::InvalidCurrentMemory`] when the memory to give up is not plain heap or
    /// the memory to take is not free.
    pub fn set_heap_size(&mut self, size: u64) -> CallResult<u64> {
        let heap_start = self.regions.heap.start;
        // Validate that setting heap size is possible at all: it must fit the heap region and
        // the largest heap. (Eden makes the two tests one after the other.)
        if size > self.max_heap_size.min(self.regions.heap.size()) {
            return Err(KernelError::OutOfMemory.into());
        }
        let current = self.heap_size();
        let new_end = heap_start + size;
        if size < current {
            // The size being requested is less than the current size, so we need to free the
            // end of the heap.
            let freed = current - size;
            let pages = self.pages_of(new_end, freed)?;
            let check = Check {
                state_mask: STATE_ALL,
                state: MemoryState::Normal.word(),
                permission_mask: MemoryPermission::ALL,
                permission: MemoryPermission::USER_READ_WRITE,
                attribute_mask: MemoryAttribute::ALL,
                attribute: MemoryAttribute::NONE,
                ignore: MemoryAttribute::COUNTED,
            };
            self.check_memory_state(new_end, freed, &check)?;
            // Unmap the end of the heap, then release the memory from the resource limit.
            let change = Change {
                state: MemoryState::Free,
                permission: MemoryPermission::NONE,
                attribute: MemoryAttribute::NONE,
                set: DisableMerge::NONE,
                clear: if size == 0 {
                    DisableMerge::NORMAL
                } else {
                    DisableMerge::NONE
                },
            };
            self.apply(new_end, pages, change, Effect::Unmap)?;
            self.release(freed)?;
        } else if size > current {
            // We have to allocate memory. Reserve it for the heap extension.
            let grown = size - current;
            self.limit.reserve(grown)?;
            let mapped = self.grow_heap(grown);
            if let Err(error) = mapped {
                self.release(grown)?;
                return Err(error);
            }
        }
        self.heap_end = new_end;
        Ok(heap_start)
    }

    /// Maps `grown` more bytes at the end of the heap.
    fn grow_heap(&mut self, grown: u64) -> CallResult<()> {
        let pages = self.pages_of(self.heap_end, grown)?;
        self.check_memory_state(self.heap_end, grown, &Check::FREE)?;
        let first = self.heap_end == self.regions.heap.start;
        let change = Change {
            state: MemoryState::Normal,
            permission: MemoryPermission::USER_READ_WRITE,
            attribute: MemoryAttribute::NONE,
            set: if first {
                DisableMerge::NORMAL
            } else {
                DisableMerge::NONE
            },
            clear: DisableMerge::NONE,
        };
        self.apply(self.heap_end, pages, change, Effect::Map)
    }

    /// Backing for a new shared memory of `size` bytes, from the [`Mapper`]
    /// ([`Mapper::allocate_shared`]). `size` is whole pages and not zero: the caller rounds it
    /// ([`super::shared_memory_size`]). Pages of any other size than asked for are refused.
    ///
    /// An internal failure when the translation has no backing to share, or refuses.
    pub fn allocate_shared(&mut self, size: u64) -> CallResult<SharedPages> {
        if !size.is_multiple_of(PAGE_SIZE) || size == 0 {
            return Err(SvcError::Internal(format!(
                "shared memory of {size:#x} bytes is not whole pages"
            )));
        }
        let pages = self.mapper.allocate_shared(size).map_err(|error| {
            SvcError::Internal(format!(
                "the translation gave no backing for {size:#x} bytes of shared memory: {error}"
            ))
        })?;
        if pages.size() != size {
            return Err(SvcError::Internal(format!(
                "the translation gave {:#x} bytes of shared memory for {size:#x}",
                pages.size()
            )));
        }
        Ok(pages)
    }

    /// Maps the shared memory `memory`, which `object` names, at `address` with `permission`
    /// (`MapPageGroup(addr, pg, KMemoryState::Shared, perm)`), and records the mapping for
    /// [`Self::unmap_shared`]. The range is the shared memory's size.
    ///
    /// [`KernelError::InvalidCurrentMemory`] unless the range may hold shared memory
    /// ([`Self::can_contain`]) and is free.
    pub fn map_shared(
        &mut self,
        address: u64,
        object: ObjectId,
        memory: SharedMemory,
        permission: SvcPermission,
    ) -> CallResult<()> {
        let size = memory.size();
        // Ensure this is a valid map request.
        if !self.can_contain(address, size, MemoryState::Shared) {
            return Err(KernelError::InvalidCurrentMemory.into());
        }
        let pages = self.pages_of(address, size)?;
        // Check if state allows us to map.
        self.check_memory_state(address, size, &Check::FREE)?;
        let change = Change {
            state: MemoryState::Shared,
            permission: MemoryPermission::from_svc(permission),
            attribute: MemoryAttribute::NONE,
            set: DisableMerge::NORMAL,
            clear: DisableMerge::NONE,
        };
        self.apply(address, pages, change, Effect::MapShared(memory.backing()))?;
        self.shared.insert(address, SharedMapping { object, size });
        Ok(())
    }

    /// Unmaps the `size` bytes of shared memory at `address`, which must be a mapping of the
    /// shared memory `object` names that [`Self::map_shared`] made there
    /// (`UnmapPageGroup(address, pg, KMemoryState::Shared)`).
    ///
    /// [`KernelError::InvalidCurrentMemory`] unless the range may hold shared memory, is shared
    /// memory with no attributes throughout, and is that mapping (see "No page groups" in the
    /// module docs).
    pub fn unmap_shared(&mut self, address: u64, size: u64, object: ObjectId) -> CallResult<()> {
        let invalid = KernelError::InvalidCurrentMemory;
        // Ensure this is a valid unmap request.
        if !self.can_contain(address, size, MemoryState::Shared) {
            return Err(invalid.into());
        }
        let pages = self.pages_of(address, size)?;
        // Check if state allows us to unmap.
        let check = Check {
            state_mask: STATE_ALL,
            state: MemoryState::Shared.word(),
            permission_mask: MemoryPermission::NONE,
            permission: MemoryPermission::NONE,
            attribute_mask: MemoryAttribute::ALL,
            attribute: MemoryAttribute::NONE,
            ignore: MemoryAttribute::COUNTED,
        };
        self.check_memory_state(address, size, &check)?;
        // Check that the page group is valid.
        if self.shared.get(&address) != Some(&SharedMapping { object, size }) {
            return Err(invalid.into());
        }
        let change = Change {
            state: MemoryState::Free,
            permission: MemoryPermission::NONE,
            attribute: MemoryAttribute::NONE,
            set: DisableMerge::NONE,
            clear: DisableMerge::NORMAL,
        };
        self.apply(address, pages, change, Effect::Unmap)?;
        self.shared.remove(&address);
        Ok(())
    }

    /// Lends `size` bytes at `address` to a transfer memory (`LockForTransferMemory`): the
    /// blocks get [`MemoryAttribute::LOCKED`], and the owner keeps `permission` of them.
    ///
    /// [`KernelError::InvalidCurrentMemory`] unless the range lies inside the address space
    /// (`Contains`), is memory that can be transferred (a state with [`flag::CAN_TRANSFER`])
    /// with one state throughout, is read-write for the user, and has no attributes but the
    /// counted ones.
    pub fn lock_for_transfer_memory(
        &mut self,
        address: u64,
        size: u64,
        permission: SvcPermission,
    ) -> CallResult<()> {
        let pages = self.pages_of(address, size)?;
        // Check the state.
        let lockable = flag::CAN_TRANSFER | flag::REFERENCE_COUNTED;
        let check = Check {
            state_mask: lockable,
            state: lockable,
            permission_mask: MemoryPermission::ALL,
            permission: MemoryPermission::USER_READ_WRITE,
            attribute_mask: MemoryAttribute::ALL,
            attribute: MemoryAttribute::NONE,
            ignore: MemoryAttribute::COUNTED,
        };
        let old = self.check_memory_state(address, size, &check)?;
        // Decide on new perm and attr. (`ConvertToKMemoryPermission` always sets the kernel's
        // read bit, so the new permission is never `None` and always replaces the old one.)
        let new_permission = MemoryPermission::from_svc(permission);
        let change = Change {
            state: old.state,
            permission: new_permission,
            attribute: MemoryAttribute::LOCKED,
            set: DisableMerge::LOCKED,
            clear: DisableMerge::NONE,
        };
        // Update permission, if we need to.
        let effect = if new_permission == old.permission {
            Effect::Keep
        } else {
            Effect::Protect
        };
        self.apply(address, pages, change, effect)
    }

    /// Takes back `size` bytes at `address` from the transfer memory they were lent to
    /// (`UnlockForTransferMemory`): [`MemoryAttribute::LOCKED`] goes, and the owner may read
    /// and write them again.
    ///
    /// [`KernelError::InvalidCurrentMemory`] unless the range lies inside the address space, is
    /// memory that can be transferred with one state and permission throughout, and has
    /// [`MemoryAttribute::LOCKED`] and no other attribute but the counted ones.
    pub fn unlock_for_transfer_memory(&mut self, address: u64, size: u64) -> CallResult<()> {
        let pages = self.pages_of(address, size)?;
        // Check the state.
        let lockable = flag::CAN_TRANSFER | flag::REFERENCE_COUNTED;
        let check = Check {
            state_mask: lockable,
            state: lockable,
            permission_mask: MemoryPermission::NONE,
            permission: MemoryPermission::NONE,
            attribute_mask: MemoryAttribute::ALL,
            attribute: MemoryAttribute::LOCKED,
            ignore: MemoryAttribute::COUNTED,
        };
        let old = self.check_memory_state(address, size, &check)?;
        // Decide on new perm and attr.
        let new_permission = MemoryPermission::USER_READ_WRITE;
        let change = Change {
            state: old.state,
            permission: new_permission,
            attribute: MemoryAttribute::NONE,
            set: DisableMerge::NONE,
            clear: DisableMerge::LOCKED,
        };
        // Update permission, if we need to.
        let effect = if new_permission == old.permission {
            Effect::Keep
        } else {
            Effect::Protect
        };
        self.apply(address, pages, change, effect)
    }

    /// Whether the `size` bytes at `address` lie inside the alias region (`IsInAliasRegion`:
    /// inside the address space, `Contains`, and from the alias region's start to its last
    /// byte). False for an empty range and for one that wraps.
    #[must_use]
    pub fn is_in_alias_region(&self, address: u64, size: u64) -> bool {
        self.regions.contains(address, size) && self.regions.alias.holds(address, size)
    }

    /// The pieces of the `size` bytes at `address`, in address order: one for each block the
    /// range touches, cut to the range. The range is not empty and lies inside the address
    /// space (the callers check [`Self::pages_of`] first), so the walk ends at the block that
    /// holds its last byte; the block map running out before that is an internal failure.
    fn pieces(&self, address: u64, size: u64) -> CallResult<Vec<Piece>> {
        let last = size
            .checked_sub(1)
            .and_then(|offset| address.checked_add(offset))
            .ok_or(KernelError::InvalidCurrentMemory)?;
        let mut pieces = Vec::new();
        let mut cursor = address;
        for block in self.blocks.blocks_from(address) {
            // `block` holds `cursor` in a sound map: the first holds `address`, each next one
            // starts where the one before ended.
            if !block.contains(cursor) {
                return Err(broken(BlockMapError::Broken { address: cursor }));
            }
            let piece_last = block.last().min(last);
            pieces.push(Piece {
                address: cursor,
                size: piece_last - cursor + 1,
                block: *block,
            });
            // Check if we're done.
            if last <= block.last() {
                return Ok(pieces);
            }
            // Advance. (`block.last() < last`, so this does not overflow.)
            cursor = block.last() + 1;
        }
        Err(broken(BlockMapError::Broken { address: cursor }))
    }

    /// Maps fresh read-write memory over every free page of the `size` bytes at `address`
    /// (`KPageTableBase::MapPhysicalMemory`): what the guest uses as memory of its own in the
    /// alias region. Pages that are not free, whatever their state, are left as they are and
    /// count as mapped already. The new pages are [`MemoryState::Normal`], read-write for the
    /// user, and count in [`Self::normal_memory_size`].
    ///
    /// In this order: [`KernelError::InvalidCurrentMemory`] unless the range is whole pages
    /// inside the address space (see "Divergences" in the module docs), then
    /// [`KernelError::InvalidMemoryRegion`] unless it lies inside the alias region
    /// ([`Self::is_in_alias_region`]; the system call checks both first, with
    /// `ResultInvalidMemoryRegion`), then [`KernelError::LimitReached`] when the pool has no
    /// room for the free pages. Nothing changes on any of these, nor when every page is mapped
    /// already, which succeeds.
    ///
    /// A failed [`Mapper`] is an internal failure. The pages mapped before it are unmapped and
    /// their memory released, as Eden's `ON_RESULT_FAILURE` does; a page whose unmap fails as
    /// well stays mapped, and is recorded and counted as mapped physical memory, so the
    /// bookkeeping says what the translation holds (module docs, "Divergences").
    pub fn map_physical_memory(&mut self, address: u64, size: u64) -> CallResult<()> {
        let pages = self.pages_of(address, size)?;
        if !self.is_in_alias_region(address, size) {
            return Err(KernelError::InvalidMemoryRegion.into());
        }
        // Check if the memory is already mapped: every piece that is not free counts.
        let pieces = self.pieces(address, size)?;
        let mapped_size: u64 = pieces
            .iter()
            .filter(|piece| !piece.is_free())
            .map(|piece| piece.size)
            .sum();
        // If the size mapped is the size requested, we've nothing to do.
        if size == mapped_size {
            return Ok(());
        }
        // Reserve the memory from the process resource limit. (The pieces are the range, so
        // `mapped_size < size`.)
        let unmapped = size - mapped_size;
        let perms = PHYSICAL
            .permission
            .guest()
            .ok_or_else(|| SvcError::Internal("read-write memory has no translation".into()))?;
        self.limit.reserve(unmapped)?;
        // Map the free pieces.
        let free: Vec<Piece> = pieces.into_iter().filter(Piece::is_free).collect();
        for (index, piece) in free.iter().enumerate() {
            if let Err(error) = self.mapper.map(piece.address, piece.size, perms) {
                let mapped = free.get(..index).unwrap_or_default();
                return Err(self.undo_physical_map(mapped, unmapped, piece, &error));
            }
        }
        // We succeeded, so commit the memory reservation, and increase our tracked mapped size.
        self.add_physical(unmapped)?;
        // Update the relevant memory blocks. (Eden passes a barrier to set when the range
        // starts at the alias region's start, which `UpdateIfMatch` never sets; see
        // `BlockMap::update_if_match`.)
        self.blocks
            .update_if_match(address, pages, FREE, PHYSICAL, DisableMerge::NONE)
            .map_err(broken)
    }

    /// Counts `bytes` more of mapped physical memory.
    fn add_physical(&mut self, bytes: u64) -> CallResult<()> {
        self.mapped_physical_memory_size = self
            .mapped_physical_memory_size
            .checked_add(bytes)
            .ok_or_else(|| SvcError::Internal("the mapped physical memory overflowed".into()))?;
        Ok(())
    }

    /// Counts `bytes` less of mapped physical memory. More than is counted is the kernel's own
    /// miscount.
    fn remove_physical(&mut self, bytes: u64) -> CallResult<()> {
        self.mapped_physical_memory_size = self
            .mapped_physical_memory_size
            .checked_sub(bytes)
            .ok_or_else(|| {
                SvcError::Internal(format!(
                    "unmapping {bytes:#x} bytes of physical memory of which only {:#x} are mapped",
                    self.mapped_physical_memory_size
                ))
            })?;
        Ok(())
    }

    /// What [`Self::map_physical_memory`] does when the [`Mapper`] refuses to map `failed`
    /// after it mapped the pieces `mapped` (Eden's `ON_RESULT_FAILURE`): unmaps them and
    /// releases the `reserved` bytes. A piece whose unmap fails too is still mapped: it becomes
    /// mapped physical memory in the blocks and the count, and keeps its part of the
    /// reservation. Returns the error the call fails with.
    fn undo_physical_map(
        &mut self,
        mapped: &[Piece],
        reserved: u64,
        failed: &Piece,
        error: &MapperError,
    ) -> SvcError {
        let mut messages = vec![format!(
            "the translation refused to map {:#x} bytes at {:#x}: {error}",
            failed.size, failed.address
        )];
        let mut kept = 0_u64;
        for piece in mapped {
            if let Err(error) = self.mapper.unmap(piece.address, piece.size) {
                messages.push(format!(
                    "undoing the map of {:#x} bytes at {:#x} failed too, so they stay mapped: \
                     {error}",
                    piece.size, piece.address
                ));
                kept = kept.saturating_add(piece.size);
                if let Err(error) = self.blocks.update_if_match(
                    piece.address,
                    piece.pages(),
                    FREE,
                    PHYSICAL,
                    DisableMerge::NONE,
                ) {
                    return broken(error);
                }
            }
        }
        let recorded = self
            .add_physical(kept)
            .and_then(|()| self.release(reserved.saturating_sub(kept)));
        match recorded {
            Ok(()) => SvcError::Internal(messages.join("; ")),
            Err(error) => error,
        }
    }

    /// Unmaps every page of the `size` bytes at `address` that [`Self::map_physical_memory`]
    /// mapped and gives its memory back (`KPageTableBase::UnmapPhysicalMemory`). Free pages in
    /// the range are left free.
    ///
    /// In this order: [`KernelError::InvalidCurrentMemory`] unless the range is whole pages
    /// inside the address space (module docs, "Divergences"), then
    /// [`KernelError::InvalidMemoryRegion`] unless it lies inside the alias region (the system
    /// call checks both first), then [`KernelError::InvalidCurrentMemory`] unless every block
    /// the range touches is free, or is [`MemoryState::Normal`] with no attributes at all.
    /// Nothing changes on any of these, nor when nothing in the range is mapped, which
    /// succeeds.
    ///
    /// A failed [`Mapper`] is an internal failure. The pieces it unmapped before the one it
    /// refused are freed and their memory released, so the bookkeeping says what the
    /// translation holds (module docs, "Divergences"); Eden asserts the unmap succeeds.
    pub fn unmap_physical_memory(&mut self, address: u64, size: u64) -> CallResult<()> {
        let pages = self.pages_of(address, size)?;
        if !self.is_in_alias_region(address, size) {
            return Err(KernelError::InvalidMemoryRegion.into());
        }
        // Check if the memory is mapped, and verify the memory's state.
        let pieces = self.pieces(address, size)?;
        let mut mapped_size = 0_u64;
        for piece in &pieces {
            let is_normal = piece.block.state() == MemoryState::Normal
                && piece.block.attribute() == MemoryAttribute::NONE;
            if is_normal {
                // The pieces are the range, so the sum is at most `size`.
                mapped_size += piece.size;
            } else if !piece.is_free() {
                return Err(KernelError::InvalidCurrentMemory.into());
            }
        }
        // If there's nothing mapped, we've nothing to do.
        if mapped_size == 0 {
            return Ok(());
        }
        // The first block's barrier goes if it is mapped physical memory that starts both the
        // range and the alias region.
        let clear = match pieces.first() {
            Some(first)
                if first.block.state() == MemoryState::Normal
                    && first.block.address() == self.regions.alias.start
                    && first.block.address() == address =>
            {
                DisableMerge::NORMAL
            }
            _ => DisableMerge::NONE,
        };
        // Unmap the mapped pieces.
        let normal: Vec<Piece> = pieces
            .into_iter()
            .filter(|piece| !piece.is_free())
            .collect();
        for (index, piece) in normal.iter().enumerate() {
            if let Err(error) = self.mapper.unmap(piece.address, piece.size) {
                let unmapped = normal.get(..index).unwrap_or_default();
                let message = format!(
                    "the translation refused to unmap {:#x} bytes at {:#x}: {error}",
                    piece.size, piece.address
                );
                self.free_physical(unmapped, clear)?;
                return Err(SvcError::Internal(message));
            }
        }
        // Release the memory resource.
        self.remove_physical(mapped_size)?;
        self.release(mapped_size)?;
        // Update memory blocks.
        self.blocks
            .update(
                address,
                pages,
                FREE.state,
                FREE.permission,
                FREE.attribute,
                DisableMerge::NONE,
                clear,
            )
            .map_err(broken)
    }

    /// Records the `unmapped` pieces of mapped physical memory as free, removing `clear` from
    /// their barriers, and releases their memory: what [`Self::unmap_physical_memory`] keeps of
    /// an unmap the [`Mapper`] refused partway.
    fn free_physical(&mut self, unmapped: &[Piece], clear: DisableMerge) -> CallResult<()> {
        for piece in unmapped {
            self.blocks
                .update(
                    piece.address,
                    piece.pages(),
                    FREE.state,
                    FREE.permission,
                    FREE.attribute,
                    DisableMerge::NONE,
                    clear,
                )
                .map_err(broken)?;
            self.remove_physical(piece.size)?;
            self.release(piece.size)?;
        }
        Ok(())
    }

    /// Whether every page the `size` bytes at `address` touch has
    /// [`MemoryAttribute::GPU_ACCESSIBLE`], the attribute `svc 0x80` sets: the graphics
    /// server's range query (kernel spec, "Address space"), PR #4588's walk of `QueryInfo`
    /// blocks in `BindNativeMemoryBacking`.
    ///
    /// False for an empty range and for one that wraps or does not lie wholly inside the
    /// address space: the walk reaches the end of the block map before the range's last byte
    /// (PR #4588's callers check `Contains` first, with `ResultInvalidCurrentMemory`, the result
    /// they give a range without the attribute too). The empty range is starquake's choice (a
    /// guess): the reference's callers refuse a size of zero before they walk.
    #[must_use]
    pub fn is_gpu_accessible(&self, address: u64, size: u64) -> bool {
        let Some(last) = size
            .checked_sub(1)
            .and_then(|offset| address.checked_add(offset))
        else {
            return false;
        };
        // The walk ends at the block holding `last`, or at the end of the map (false).
        for block in self.blocks.blocks_from(address) {
            if !block.attribute().contains(MemoryAttribute::GPU_ACCESSIBLE) {
                return false;
            }
            if last <= block.last() {
                return true;
            }
        }
        false
    }

    /// Takes a device-share lock on `size` bytes at `address`
    /// (`LockForMapDeviceAddressSpace(address, size, KMemoryPermission::None, is_aligned: true,
    /// check_heap: true)`, as PR #4588's graphics server calls it): every block of the range
    /// counts one more device use and has [`MemoryAttribute::DEVICE_SHARED`] (see "What a
    /// device-share lock stops" in the module docs). Locks nest: each is undone by one
    /// [`Self::unlock_device_share`] of the same range.
    ///
    /// In this order: [`KernelError::InvalidCurrentMemory`] unless the range is whole pages
    /// inside the address space (`Contains`; see "Divergences" for the pages); then unless it
    /// has one state, permission and attributes throughout (`DeviceShared` aside), its state
    /// can be device mapped aligned and is reference counted
    /// ([`flag::CAN_ALIGNED_DEVICE_MAP`], [`flag::REFERENCE_COUNTED`]), and it is neither
    /// [`MemoryAttribute::LOCKED`] nor [`MemoryAttribute::IPC_LOCKED`]; then
    /// [`KernelError::OutOfResource`] when a block's device count is at its largest.
    ///
    /// A lock taken is counted for its exact range ([`Self::device_locks`]); crate-private, so
    /// that outside the kernel a lock comes with its token (module docs, "The device-share
    /// token").
    pub(crate) fn lock_device_share(&mut self, address: u64, size: u64) -> CallResult<()> {
        let pages = self.pages_of(address, size)?;
        // Check the memory state.
        let test_state = DEVICE_SHARE_LOCK_STATE;
        let check = Check {
            state_mask: test_state,
            state: test_state,
            permission_mask: MemoryPermission::NONE,
            permission: MemoryPermission::NONE,
            attribute_mask: MemoryAttribute::IPC_LOCKED | MemoryAttribute::LOCKED,
            attribute: MemoryAttribute::NONE,
            ignore: MemoryAttribute::DEVICE_SHARED,
        };
        self.check_memory_state(address, size, &check)?;
        // A range's count never passes a block's (each lock of the range counts in every
        // block of it), so this cannot overflow while the block map is sound.
        let count = self
            .device_locks(address, size)
            .checked_add(1)
            .ok_or_else(|| SvcError::Internal("a device-share lock count overflowed".into()))?;
        // Update the memory blocks.
        self.blocks
            .update_lock(address, pages, DeviceLock::Share)
            .map_err(|error| match error {
                BlockMapError::DeviceCount { .. } => SvcError::from(KernelError::OutOfResource),
                error => broken(error),
            })?;
        self.device_locks.insert((address, size), count);
        Ok(())
    }

    /// How many live device-share locks were taken on exactly `size` bytes at `address`: taken
    /// by [`Self::lock_device_share`] and not yet released. Zero for any other range, a range
    /// inside or around a locked one included.
    #[must_use]
    pub fn device_locks(&self, address: u64, size: u64) -> u32 {
        self.device_locks
            .get(&(address, size))
            .copied()
            .unwrap_or(0)
    }

    /// Releases a device-share lock on `size` bytes at `address` that
    /// [`Self::lock_device_share`] took (`UnlockForDeviceAddressSpace`): every block of the range
    /// counts one device use fewer, and loses [`MemoryAttribute::DEVICE_SHARED`] with its last.
    ///
    /// [`KernelError::InvalidCurrentMemory`] unless the range is whole pages inside the address
    /// space; then unless each of its blocks can be device mapped ([`flag::CAN_DEVICE_MAP`]),
    /// is [`MemoryAttribute::DEVICE_SHARED`] and is not [`MemoryAttribute::LOCKED`]
    /// (`CheckMemoryStateContiguous`: the blocks need not be alike); then unless a live lock
    /// was taken on exactly this range ([`Self::device_locks`]; see "Divergences"). Nothing
    /// changes on a refusal.
    ///
    /// Crate-private, like the lock: outside the kernel, a lock is released with its token.
    pub(crate) fn unlock_device_share(&mut self, address: u64, size: u64) -> CallResult<()> {
        let pages = self.pages_of(address, size)?;
        // Check the memory state.
        let check = Check {
            state_mask: DEVICE_SHARE_UNLOCK_STATE,
            state: DEVICE_SHARE_UNLOCK_STATE,
            permission_mask: MemoryPermission::NONE,
            permission: MemoryPermission::NONE,
            attribute_mask: MemoryAttribute::DEVICE_SHARED | MemoryAttribute::LOCKED,
            attribute: MemoryAttribute::DEVICE_SHARED,
            ignore: MemoryAttribute::NONE,
        };
        self.check_memory_state_contiguous(address, size, &check)?;
        // Only a range a live lock was taken on exactly (module docs, "Divergences").
        let Some(left) = self.device_locks(address, size).checked_sub(1) else {
            return Err(KernelError::InvalidCurrentMemory.into());
        };
        // Update the memory blocks. A range a lock was taken on ends where that lock ends, so
        // the block map's own refusal is left only to unsound bookkeeping.
        self.blocks
            .update_lock(address, pages, DeviceLock::Unshare)
            .map_err(|error| match error {
                BlockMapError::DeviceCount { .. } => {
                    SvcError::from(KernelError::InvalidCurrentMemory)
                }
                error => broken(error),
            })?;
        if left == 0 {
            self.device_locks.remove(&(address, size));
        } else {
            self.device_locks.insert((address, size), left);
        }
        Ok(())
    }

    /// The block that holds `address` (`QueryInfo`). Above the address space it is one
    /// inaccessible block from the address space's end to the end of the 64-bit space.
    pub fn query_info(&self, address: u64) -> CallResult<MemoryBlock> {
        if !self.regions.contains(address, 1) {
            // If the address is invalid, create a fake block.
            let end = self.regions.address_space.end;
            let pages = (u64::MAX - end) / PAGE_SIZE + 1;
            return MemoryBlock::new(
                end,
                pages,
                MemoryState::Inaccessible,
                MemoryPermission::NONE,
                MemoryAttribute::NONE,
            )
            .ok_or_else(|| SvcError::Internal("no block above the address space".into()));
        }
        self.blocks
            .find(address)
            .copied()
            .ok_or_else(|| broken(BlockMapError::Broken { address }))
    }

    /// Whether the guest may write all `size` bytes at `address`: every block the range
    /// touches has [`MemoryPermission::USER_WRITE`]. False for a range that wraps or reaches
    /// above the address space; true for `size == 0`, which writes nothing.
    ///
    /// Unlike the state check an operation makes of its range (`CheckMemoryState`), the blocks
    /// need not be alike: a buffer may span two neighbouring blocks of different states, both
    /// writable.
    #[must_use]
    pub fn user_can_write(&self, address: u64, size: u64) -> bool {
        self.user_has(MemoryPermission::USER_WRITE, address, size)
    }

    /// Whether the guest may read all `size` bytes at `address`: every block the range touches
    /// has [`MemoryPermission::USER_READ`]. The same rules as [`AddressSpace::user_can_write`]
    /// otherwise.
    #[must_use]
    pub fn user_can_read(&self, address: u64, size: u64) -> bool {
        self.user_has(MemoryPermission::USER_READ, address, size)
    }

    /// Whether every block the range touches has `permission`.
    fn user_has(&self, permission: MemoryPermission, address: u64, size: u64) -> bool {
        let Some(offset) = size.checked_sub(1) else {
            return true;
        };
        let Some(last) = address.checked_add(offset) else {
            return false;
        };
        // The walk ends at the block holding `last`, or at the end of the map (false).
        for block in self.blocks.blocks_from(address) {
            if !block.permission().contains(permission) {
                return false;
            }
            if last <= block.last() {
                return true;
            }
        }
        false
    }
}

mod alias;

pub mod snapshot;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod alias_tests;

#[cfg(test)]
mod device_tests;

#[cfg(test)]
mod physical_tests;
