//! The kernel's shared-memory calls (`starquake_kernel::svc::shared_memory`) over the real
//! mapper: the scenarios of the kernel's own tests (`crates/kernel/src/svc/shared_memory/tests.rs`,
//! which run on a recording mapper), checked here in the translation guest code runs on, read
//! back with the software table walker. `space/tests/hvf.rs` runs them on a vCPU.

use starquake_kernel::handle::{Handle, HandleTable};
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::{
    AddressSpace as KernelSpace, MemoryArrangement, MemoryBlock, MemoryState, ProcessMemory,
    RegionOffsets, SharedMemory, SharedPages, SvcPermission,
};
use starquake_kernel::object::Object;
use starquake_kernel::result::{CallResult, KernelError, SvcError};
use starquake_kernel::svc::shared_memory::{map_shared_memory, unmap_shared_memory};
use starquake_kernel::tables::Scheduler;

use super::*;

/// Where the code goes: the kernel fixture's address (`crates/kernel/src/memory/fixture.rs`,
/// Eden's address for 39-bit code without randomisation).
pub(super) const CODE: u64 = 0x8000_0000;
/// Seven pages of code, as the kernel fixture has.
const CODE_SIZE: u64 = 7 * PAGE_SIZE;
/// A free place where shared memory may go: the kernel fixture's `SHARED`, inside the
/// alias-code region (checked by [`World::over`]).
pub(super) const SHARED: u64 = 0x4000_0000;
/// Pages in the tests' usual shared memory, and its size.
pub(super) const PAGES: u64 = 2;
pub(super) const SIZE: u64 = PAGES * PAGE_SIZE;

pub(super) fn raw(permission: SvcPermission) -> u32 {
    u32::from(permission.bits())
}

/// The permission the translation gives guest code for a shared memory's permission.
fn perms_of(permission: SvcPermission) -> Perms {
    match permission {
        SvcPermission::Read => Perms::Read,
        SvcPermission::ReadWrite => Perms::ReadWrite,
        SvcPermission::None | SvcPermission::ReadExecute => {
            unreachable!("not a shared memory permission")
        }
    }
}

/// The kernel's address space over the real one, with a handle table and a scheduler beside it.
pub(super) struct World<M> {
    handles: HandleTable,
    pub(super) kernel: KernelSpace,
    scheduler: Scheduler,
    pub(super) space: Arc<AddressSpace<M>>,
}

/// A shared memory a test made: its object (the creator's reference is still held), the
/// guest's handle to it, and its pages.
pub(super) struct Shared {
    object: ObjectId,
    pub(super) handle: Handle,
    pub(super) pages: SharedPages,
}

impl World<FlatMemory> {
    fn new() -> Self {
        Self::over(space())
    }
}

impl<M: GuestMemory + 'static> World<M> {
    /// The kernel's address space for the fixture's program over `space`.
    pub(super) fn over(space: AddressSpace<M>) -> Self {
        let space = Arc::new(space);
        let memory = ProcessMemory {
            code_address: CODE,
            code_size: CODE_SIZE,
            arrangement: MemoryArrangement::FourGb,
            alias_region_extra: false,
            offsets: RegionOffsets::default(),
        };
        let kernel = KernelSpace::new(&memory, Box::new(SpaceMapper(Arc::clone(&space)))).unwrap();
        assert!(kernel.can_contain(SHARED, 4 * SIZE, MemoryState::Shared));
        Self {
            handles: HandleTable::new(8).unwrap(),
            kernel,
            scheduler: Scheduler::default(),
            space,
        }
    }

    /// Makes a shared memory as a server does (`IpcContext::create_shared_memory`), and gives
    /// the guest a handle to it.
    pub(super) fn shared(&mut self, pages: u64, permission: SvcPermission) -> Shared {
        let pages = self.kernel.allocate_shared(pages * PAGE_SIZE).unwrap();
        let memory = SharedMemory::new(pages.backing(), permission).unwrap();
        let object = self
            .scheduler
            .objects_mut()
            .create(Object::SharedMemory(memory))
            .unwrap();
        let handle = self
            .handles
            .add(self.scheduler.objects_mut(), object)
            .unwrap();
        Shared {
            object,
            handle,
            pages,
        }
    }

    pub(super) fn map(
        &mut self,
        handle: Handle,
        address: u64,
        size: u64,
        permission: u32,
    ) -> CallResult<()> {
        map_shared_memory(
            &self.handles,
            &mut self.kernel,
            &mut self.scheduler,
            handle,
            address,
            size,
            permission,
        )
    }

    pub(super) fn unmap(&mut self, handle: Handle, address: u64, size: u64) -> CallResult<()> {
        unmap_shared_memory(
            &self.handles,
            &mut self.kernel,
            &mut self.scheduler,
            handle,
            address,
            size,
        )
    }

    fn refs(&self, object: ObjectId) -> u32 {
        self.scheduler.objects().refs(object)
    }

    fn state(&self, address: u64) -> MemoryState {
        self.kernel.query_info(address).unwrap().state()
    }

    /// The kernel's blocks and the real address space, for "nothing changed".
    fn everything(&self) -> (Vec<MemoryBlock>, Snapshot) {
        (
            self.kernel.blocks().copied().collect(),
            snapshot(&self.space),
        )
    }

    /// Whether every page of `pages` is what the table walker finds at `address` on, with
    /// `perms`.
    fn mapped_as(&self, address: u64, pages: &SharedPages, perms: Perms) -> bool {
        let count = pages.size() / PAGE_SIZE;
        (0..count).all(|index| {
            self.space.tables.walk(address + index * PAGE_SIZE)
                == Ok(Resolved {
                    pa: pages.backing().base() + index * PAGE_SIZE,
                    perms,
                    level: 3,
                })
        })
    }
}

#[test]
fn mapping_shared_memory_maps_its_pages_as_asked_and_unmapping_takes_them_away() {
    for (pages, permission) in [
        (1, SvcPermission::ReadWrite),
        (PAGES, SvcPermission::Read),
        (PAGES, SvcPermission::ReadWrite),
    ] {
        let mut world = World::new();
        let shared = world.shared(pages, permission);
        let size = pages * PAGE_SIZE;
        let tag = format!("{pages} pages, {permission:?}");
        assert_eq!(world.refs(shared.object), 2, "{tag}");
        assert_eq!(
            world.map(shared.handle, SHARED, size, raw(permission)),
            Ok(()),
            "{tag}"
        );
        assert_eq!(world.state(SHARED), MemoryState::Shared, "{tag}");
        assert!(
            world.mapped_as(SHARED, &shared.pages, perms_of(permission)),
            "{tag}"
        );
        assert!(unmapped_at(&world.space, SHARED + size), "{tag}");
        assert_eq!(world.refs(shared.object), 3, "{tag}");
        assert_eq!(world.space.tables().take_pending_flushes(), [], "{tag}");

        assert_eq!(world.unmap(shared.handle, SHARED, size), Ok(()), "{tag}");
        assert_eq!(world.state(SHARED), MemoryState::Free, "{tag}");
        assert!(unmapped_at(&world.space, SHARED), "{tag}");
        let flushes = world.space.tables().take_pending_flushes();
        assert_eq!(
            flushes.iter().map(|f| (f.va, f.size)).collect::<Vec<_>>(),
            [(SHARED, size)],
            "{tag}"
        );
        assert_eq!(world.refs(shared.object), 2, "{tag}");
    }
}

#[test]
fn a_host_write_is_visible_through_the_guest_mapping_and_the_reverse() {
    let mut world = World::new();
    let shared = world.shared(PAGES, SvcPermission::ReadWrite);
    let read_write = raw(SvcPermission::ReadWrite);
    world.map(shared.handle, SHARED, SIZE, read_write).unwrap();
    // The service writes; the guest's address space reads at the mapping.
    shared.pages.write(PAGE_SIZE + 8, &[0xA1, 0xB2]).unwrap();
    let mut seen = [0; 2];
    world.space.read(SHARED + PAGE_SIZE + 8, &mut seen).unwrap();
    assert_eq!(seen, [0xA1, 0xB2]);
    // The guest's address space writes; the service reads.
    world.space.write(SHARED + 3, &[0xC3]).unwrap();
    let mut byte = [0];
    shared.pages.read(3, &mut byte).unwrap();
    assert_eq!(byte, [0xC3]);
    // Unmapped, the guest no longer reaches the pages, which keep what they hold.
    world.unmap(shared.handle, SHARED, SIZE).unwrap();
    assert_eq!(
        world.space.read(SHARED + 3, &mut byte),
        Err(Fault {
            address: SHARED + 3
        })
    );
    shared.pages.read(3, &mut byte).unwrap();
    assert_eq!(byte, [0xC3]);
}

#[test]
fn each_refused_mapping_changes_neither_the_blocks_nor_the_translation() {
    let mut world = World::new();
    let read_only = world.shared(PAGES, SvcPermission::Read);
    let read = raw(SvcPermission::Read);
    world.map(read_only.handle, SHARED, SIZE, read).unwrap();
    let other = world.shared(PAGES, SvcPermission::Read);
    let before = world.everything();
    let regions = *world.kernel.regions();
    let refusals = [
        // Over the mapping already there.
        (SHARED, SIZE, read, KernelError::InvalidCurrentMemory),
        // Not the shared memory's size.
        (SHARED + SIZE, PAGE_SIZE, read, KernelError::InvalidSize),
        // Not its permission.
        (
            SHARED + SIZE,
            SIZE,
            raw(SvcPermission::ReadWrite),
            KernelError::InvalidNewMemoryPermission,
        ),
        // Past the end of the region.
        (
            regions.code.end - PAGE_SIZE,
            SIZE,
            read,
            KernelError::InvalidMemoryRegion,
        ),
    ];
    for (address, size, permission, error) in refusals {
        assert_eq!(
            world.map(other.handle, address, size, permission),
            Err(SvcError::Kernel(error)),
            "{address:#x}"
        );
        assert_eq!(world.everything(), before, "{address:#x}");
        assert_eq!(world.refs(other.object), 2, "{address:#x}");
    }
    // An unmapping that names the other shared memory is refused too.
    assert_eq!(
        world.unmap(other.handle, SHARED, SIZE),
        Err(SvcError::Kernel(KernelError::InvalidCurrentMemory))
    );
    assert_eq!(world.everything(), before);
}

#[test]
fn a_mapping_that_ends_where_its_neighbour_starts_and_one_that_ends_at_the_regions_end() {
    let mut world = World::new();
    let read_write = raw(SvcPermission::ReadWrite);
    let end = world.kernel.regions().code.end;
    // The neighbour ends exactly at the end of the region; the other ends where it starts.
    let neighbour = world.shared(PAGES, SvcPermission::ReadWrite);
    let last = end - SIZE;
    assert_eq!(world.map(neighbour.handle, last, SIZE, read_write), Ok(()));
    neighbour.pages.write(0, b"next").unwrap();
    let below = world.shared(PAGES, SvcPermission::ReadWrite);
    assert_eq!(
        world.map(below.handle, last - SIZE, SIZE, read_write),
        Ok(())
    );
    assert!(world.mapped_as(last - SIZE, &below.pages, Perms::ReadWrite));
    assert!(world.mapped_as(last, &neighbour.pages, Perms::ReadWrite));
    // Unmapping the lower one leaves the neighbour's mapping and bytes as they were.
    assert_eq!(world.unmap(below.handle, last - SIZE, SIZE), Ok(()));
    assert!(unmapped_at(&world.space, last - SIZE));
    assert!(world.mapped_as(last, &neighbour.pages, Perms::ReadWrite));
    let mut buf = [0; 4];
    world.space.read(last, &mut buf).unwrap();
    assert_eq!(&buf, b"next");
    assert_eq!(world.state(last), MemoryState::Shared);
}

#[test]
fn one_shared_memory_mapped_at_two_places_shares_its_pages() {
    let mut world = World::new();
    let shared = world.shared(PAGES, SvcPermission::ReadWrite);
    let read_write = raw(SvcPermission::ReadWrite);
    world.map(shared.handle, SHARED, SIZE, read_write).unwrap();
    world
        .map(shared.handle, SHARED + 2 * SIZE, SIZE, read_write)
        .unwrap();
    world.space.write(SHARED + 2 * SIZE + 1, b"twice").unwrap();
    let mut seen = [0; 5];
    world.space.read(SHARED + 1, &mut seen).unwrap();
    assert_eq!(&seen, b"twice");
    assert_eq!(world.refs(shared.object), 4);
}

#[test]
fn running_out_of_ram_by_exactly_one_page_is_the_kernels_own_failure_and_not_a_panic() {
    let mut world = World::new();
    let all = RAM_PAGES * PAGE_SIZE;
    let over = all + PAGE_SIZE;
    assert_eq!(
        world.kernel.allocate_shared(over).map(|pages| pages.size()),
        Err(SvcError::Internal(format!(
            "the translation gave no backing for {over:#x} bytes of shared memory: mapping \
             {over:#x} bytes needs more guest RAM than the {all:#x} bytes left"
        )))
    );
    assert_eq!(
        world.kernel.allocate_shared(all).map(|pages| pages.size()),
        Ok(all)
    );
}
