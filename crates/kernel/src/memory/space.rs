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
//!   `ResultOutOfMemory` when the heap has none. Here the pool is a count ([`MemoryLimit`]):
//!   a reservation that fits always gets its memory, lazily backed by the execution model.
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
//! - **`MapPhysicalMemory` is not built**, so [`AddressSpace::normal_memory_size`] is the
//!   heap alone; Eden adds `m_mapped_physical_memory_size`.
//! - **A failed [`Mapper`]** ends the operation as an internal failure with nothing changed.
//!   Eden's `Operate` cannot fail on the paths ported here. Test:
//!   `a_refusing_translation_fails_the_call_and_changes_nothing`.

use core::fmt;

use starquake_memory::PAGE_SIZE;

use super::block::MemoryBlock;
use super::block_map::{BlockMap, BlockMapError};
use super::layout::{LayoutError, LayoutRequest, Region, RegionOffsets, Regions, pages_to_bytes};
use super::mapper::{Mapper, MapperError};
use super::pool::{MemoryArrangement, MemoryLimit};
use super::state::{
    DisableMerge, MemoryAttribute, MemoryPermission, MemoryState, STATE_ALL, SvcPermission, flag,
};
use crate::result::{CallResult, KernelError, SvcError, SvcResult};

/// Free pages kept on each side of a mapping the kernel places for a process
/// (`KPageTableBase::GetNumGuardPages`, for a page table that is not the kernel's).
pub const GUARD_PAGES: u64 = 4;

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
    set: DisableMerge,
    clear: DisableMerge,
}

/// What the [`Mapper`] is asked to do for a change.
#[derive(Debug, Clone, Copy)]
enum Effect {
    Map,
    Unmap,
    Protect,
}

impl Effect {
    /// The effect as a verb, for an error message.
    const fn verb(self) -> &'static str {
        match self {
            Self::Map => "map",
            Self::Unmap => "unmap",
            Self::Protect => "reprotect",
        }
    }
}

/// One process's memory blocks, regions, heap and memory accounting (kernel spec, "Tables
/// behind a dispatcher": the `AddressSpace` table).
pub struct AddressSpace {
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
    mapper: Box<dyn Mapper>,
}

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
            .finish_non_exhaustive()
    }
}

/// A failure of the block map, which no guest input explains.
fn broken(error: BlockMapError) -> SvcError {
    SvcError::internal(&error)
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
        if !memory.code_address.is_multiple_of(PAGE_SIZE)
            || !memory.code_size.is_multiple_of(PAGE_SIZE)
        {
            return Err(LayoutError::BadCode {
                address: memory.code_address,
                size: memory.code_size,
            }
            .into());
        }
        let regions = Regions::new(&LayoutRequest {
            code_address: memory.code_address,
            code_size: memory.code_size,
            physical_memory_size: memory.arrangement.physical_memory_size(),
            alias_region_extra: memory.alias_region_extra,
            offsets: memory.offsets,
        })?;
        let blocks = BlockMap::new(regions.address_space.start, regions.address_space.end)
            .map_err(broken)?;
        let mut space = Self {
            regions,
            blocks,
            heap_end: regions.heap.start,
            max_heap_size: 0,
            max_process_memory: regions.heap.size(),
            limit: MemoryLimit::new(memory.arrangement.application_pool_size()),
            code_size: memory.code_size,
            main_thread_stack_size: 0,
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

    /// Bytes of heap (`GetNormalMemorySize`; see "Divergences" in the module docs).
    #[must_use]
    pub const fn normal_memory_size(&self) -> u64 {
        self.heap_size()
    }

    /// The memory the process uses (`KProcess::GetUsedUserPhysicalMemorySize`, with no secure
    /// system resource): its heap, its code and its main thread's stack.
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

    /// The region memory of `state` goes in (`GetRegionAddress` and `GetRegionSize`), or
    /// `None` for [`MemoryState::Inaccessible`], which has none (Eden: `UNREACHABLE`).
    fn region_for(&self, state: MemoryState) -> Option<Region> {
        use MemoryState as S;
        Some(match state {
            S::Free | S::Kernel => self.regions.address_space,
            S::Normal => self.regions.heap,
            S::Ipc | S::NonSecureIpc | S::NonDeviceIpc => self.regions.alias,
            S::Stack => self.regions.stack,
            S::Static | S::ThreadLocal => self.regions.kernel_map,
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
            | S::CodeData => self.regions.code,
            S::Inaccessible => return None,
        })
    }

    /// Whether the `size` bytes at `address` may hold memory of `state` (`CanContain`): inside
    /// the state's region, and outside the heap and alias regions unless the state belongs
    /// there.
    #[must_use]
    pub fn can_contain(&self, address: u64, size: u64, state: MemoryState) -> bool {
        use MemoryState as S;
        let Some(region) = self.region_for(state) else {
            return false;
        };
        let range = Region {
            start: address,
            end: address.saturating_add(size),
        };
        let in_region = region.holds(address, size);
        let in_heap = self.regions.heap.overlaps(range);
        let in_alias = self.regions.alias.overlaps(range);
        match state {
            S::Free | S::Kernel => in_region,
            S::Normal => in_region && !in_alias,
            S::Ipc | S::NonSecureIpc | S::NonDeviceIpc => in_region && !in_heap,
            _ => in_region && !in_heap && !in_alias,
        }
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
            let uniform = block.state() == first.state()
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
            set,
            clear,
        } = change;
        let none = MemoryAttribute::NONE;
        self.blocks
            .check_update(address, pages, state, permission, none)
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
            Effect::Unmap => self.mapper.unmap(address, size),
            Effect::Protect => self.mapper.protect(address, size, perms?),
        }
        .map_err(refused)?;
        // `check_update` passed and nothing has touched the map since, so this cannot fail.
        self.blocks
            .update(address, pages, state, permission, none, set, clear)
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
    /// ([`flag::CAN_CHANGE_ATTRIBUTE`] for `Uncached`, [`flag::CAN_PERMISSION_LOCK`] for
    /// `PermissionLocked`), and no attribute other than those two and `DeviceShared` is set.
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
            (MemoryAttribute::UNCACHED, flag::CAN_CHANGE_ATTRIBUTE),
            (
                MemoryAttribute::PERMISSION_LOCKED,
                flag::CAN_PERMISSION_LOCK,
            ),
        ]
        .into_iter()
        .filter(|(bit, _)| mask.contains(*bit))
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
            set: if first {
                DisableMerge::NORMAL
            } else {
                DisableMerge::NONE
            },
            clear: DisableMerge::NONE,
        };
        self.apply(self.heap_end, pages, change, Effect::Map)
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
}

#[cfg(test)]
mod tests;
