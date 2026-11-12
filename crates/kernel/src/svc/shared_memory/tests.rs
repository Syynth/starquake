//! `MapSharedMemory` and `UnmapSharedMemory` over the fixture's address space: what a mapping
//! does to the blocks, the translation and the references, the pages it shares with the host,
//! and each refusal with its result, in Eden's order.

use starquake_memory::{GuestMemory, PAGE_SIZE, Perms};

use super::*;
use crate::handle::Closed;
use crate::id::{ProcessId, SlabId};
use crate::memory::fixture::{self, SHARED};
use crate::memory::testing::{MapperCall, MapperLog};
use crate::memory::{MemoryAttribute, MemoryBlock, MemoryPermission, SharedPages};

/// Pages in the tests' usual shared memory.
const PAGES: u64 = 2;
/// Its size in bytes.
const SIZE: u64 = PAGES * PAGE_SIZE;

fn raw(permission: SvcPermission) -> u32 {
    u32::from(permission.bits())
}

fn refused(error: KernelError) -> CallResult<()> {
    Err(SvcError::Kernel(error))
}

/// A loaded fixture with a handle table and a scheduler beside it.
struct World {
    handles: HandleTable,
    space: AddressSpace,
    scheduler: Scheduler,
    log: MapperLog,
}

/// A shared memory a test made: its object (the server's creator reference is still held), a
/// handle the guest has to it, and its pages.
struct Shared {
    object: ObjectId,
    handle: Handle,
    pages: SharedPages,
}

impl World {
    fn new() -> Self {
        let (space, log, _) = fixture::loaded();
        Self {
            handles: HandleTable::new(8).unwrap(),
            space,
            scheduler: Scheduler::default(),
            log,
        }
    }

    fn shared(&mut self, pages: u64, permission: SvcPermission) -> Shared {
        let (object, pages) = fixture::shared_memory(
            &mut self.space,
            self.scheduler.objects_mut(),
            pages,
            permission,
        );
        let handle = self
            .handles
            .add(self.scheduler.objects_mut(), object)
            .unwrap();
        drop(self.log.take());
        Shared {
            object,
            handle,
            pages,
        }
    }

    fn map(&mut self, handle: Handle, address: u64, size: u64, permission: u32) -> CallResult<()> {
        map_shared_memory(
            &self.handles,
            &mut self.space,
            &mut self.scheduler,
            handle,
            address,
            size,
            permission,
        )
    }

    fn unmap(&mut self, handle: Handle, address: u64, size: u64) -> CallResult<()> {
        unmap_shared_memory(
            &self.handles,
            &mut self.space,
            &mut self.scheduler,
            handle,
            address,
            size,
        )
    }

    fn refs(&self, object: ObjectId) -> u32 {
        self.scheduler.objects().refs(object)
    }

    fn blocks(&self) -> Vec<MemoryBlock> {
        self.space.blocks().copied().collect()
    }

    /// The block at `address`: its first address, pages, state, permission and attributes.
    fn block(&self, address: u64) -> (u64, u64, MemoryState, MemoryPermission, MemoryAttribute) {
        let block = self.space.query_info(address).unwrap();
        (
            block.address(),
            block.pages(),
            block.state(),
            block.permission(),
            block.attribute(),
        )
    }
}

#[test]
fn mapping_shared_memory_maps_its_pages_and_holds_a_reference_until_it_is_unmapped() {
    let mut world = World::new();
    let shared = world.shared(PAGES, SvcPermission::Read);
    assert_eq!(world.refs(shared.object), 2);
    let read = raw(SvcPermission::Read);
    assert_eq!(world.map(shared.handle, SHARED, SIZE, read), Ok(()));
    // `MapPageGroup(.., KMemoryState::Shared, ConvertToKMemoryPermission(map_perm))`.
    assert_eq!(
        world.block(SHARED),
        (
            SHARED,
            PAGES,
            MemoryState::Shared,
            MemoryPermission::USER_READ,
            MemoryAttribute::NONE
        )
    );
    assert_eq!(
        world.log.take(),
        [MapperCall::MapShared {
            address: SHARED,
            backing: shared.pages.backing(),
            perms: Perms::Read,
        }]
    );
    // `AddSharedMemory`: the mapping holds a reference of its own.
    assert_eq!(world.refs(shared.object), 3);

    assert_eq!(world.unmap(shared.handle, SHARED, SIZE), Ok(()));
    assert_eq!(world.block(SHARED).2, MemoryState::Free);
    assert_eq!(
        world.log.take(),
        [MapperCall::Unmap {
            address: SHARED,
            size: SIZE,
        }]
    );
    assert_eq!(world.refs(shared.object), 2);
}

#[test]
fn unmapping_leaves_the_blocks_as_they_were_before_the_mapping() {
    let mut world = World::new();
    let shared = world.shared(PAGES, SvcPermission::ReadWrite);
    let before = world.blocks();
    let read_write = raw(SvcPermission::ReadWrite);
    assert_eq!(world.map(shared.handle, SHARED, SIZE, read_write), Ok(()));
    assert_ne!(world.blocks(), before);
    assert_eq!(world.unmap(shared.handle, SHARED, SIZE), Ok(()));
    // The freed range joins the free memory around it again: no barrier is left at its head.
    assert_eq!(world.blocks(), before);
}

#[test]
fn a_host_write_is_visible_through_the_guest_mapping_and_the_reverse() {
    let mut world = World::new();
    let shared = world.shared(PAGES, SvcPermission::ReadWrite);
    let guest = world.log.shared_view();
    let read_write = raw(SvcPermission::ReadWrite);
    world.map(shared.handle, SHARED, SIZE, read_write).unwrap();
    // The service writes; the guest reads at the mapping.
    shared.pages.write(PAGE_SIZE + 8, &[0xA1, 0xB2]).unwrap();
    let mut seen = [0; 2];
    guest.read(SHARED + PAGE_SIZE + 8, &mut seen).unwrap();
    assert_eq!(seen, [0xA1, 0xB2]);
    // The guest writes; the service reads.
    guest.write(SHARED + 3, &[0xC3]).unwrap();
    let mut byte = [0];
    shared.pages.read(3, &mut byte).unwrap();
    assert_eq!(byte, [0xC3]);
    // Unmapped, the guest no longer reaches the pages, which keep what they hold.
    world.unmap(shared.handle, SHARED, SIZE).unwrap();
    assert!(guest.read(SHARED + 3, &mut byte).is_err());
    shared.pages.read(3, &mut byte).unwrap();
    assert_eq!(byte, [0xC3]);
}

#[test]
fn the_permission_must_be_the_user_permission() {
    let mut world = World::new();
    let read_only = world.shared(1, SvcPermission::Read);
    let writable = world.shared(1, SvcPermission::ReadWrite);
    let (read, read_write) = (raw(SvcPermission::Read), raw(SvcPermission::ReadWrite));
    let wrong = refused(KernelError::InvalidNewMemoryPermission);
    assert_eq!(
        world.map(read_only.handle, SHARED, PAGE_SIZE, read_write),
        wrong
    );
    assert_eq!(world.map(writable.handle, SHARED, PAGE_SIZE, read), wrong);
    // A refused mapping gives its reference back.
    assert_eq!(
        (world.refs(read_only.object), world.refs(writable.object)),
        (2, 2)
    );
    assert_eq!(
        world.map(writable.handle, SHARED, PAGE_SIZE, read_write),
        Ok(())
    );
    assert_eq!(world.block(SHARED).3, MemoryPermission::USER_READ_WRITE);
}

#[test]
fn a_permission_that_is_neither_read_nor_read_write_is_refused_before_the_handle_is_read() {
    let mut world = World::new();
    world.shared(1, SvcPermission::Read);
    // Each with a handle that names nothing, which a later check would refuse.
    for permission in [
        raw(SvcPermission::None),
        raw(SvcPermission::ReadExecute),
        u32::MAX,
    ] {
        assert_eq!(
            world.map(Handle::INVALID, SHARED, PAGE_SIZE, permission),
            refused(KernelError::InvalidNewMemoryPermission)
        );
    }
}

#[test]
fn each_refusal_of_a_mapping_has_its_result_in_eden_order_and_changes_nothing() {
    let mut world = World::new();
    let shared = world.shared(PAGES, SvcPermission::Read);
    // A handle to something that is not a shared memory.
    let process = world
        .scheduler
        .objects_mut()
        .create(Object::Process(ProcessId::from_parts(0, 1)))
        .unwrap();
    let other = world
        .handles
        .add(world.scheduler.objects_mut(), process)
        .unwrap();
    let (read, read_write) = (raw(SvcPermission::Read), raw(SvcPermission::ReadWrite));
    // Read, write and execute: no shared memory permission.
    let execute = read_write | raw(SvcPermission::ReadExecute);
    let (heap, wraps) = (world.space.regions().heap.start, u64::MAX - PAGE_SIZE + 1);
    let (good, busy) = (shared.handle, fixture::DATA);
    // Each case breaks one check and, where there is one, the check after it too, so the
    // first one wins.
    let cases = [
        (good, SHARED + 1, SIZE, execute, KernelError::InvalidAddress),
        (good, wraps, SIZE + 1, read, KernelError::InvalidSize),
        (good, SHARED, 0, read, KernelError::InvalidSize),
        (
            good,
            wraps,
            SIZE,
            execute,
            KernelError::InvalidCurrentMemory,
        ),
        (
            Handle::INVALID,
            SHARED,
            SIZE,
            execute,
            KernelError::InvalidNewMemoryPermission,
        ),
        (other, heap, SIZE, read, KernelError::InvalidHandle),
        (
            Handle::CURRENT_PROCESS,
            SHARED,
            SIZE,
            read,
            KernelError::InvalidHandle,
        ),
        (
            good,
            heap,
            PAGE_SIZE,
            read,
            KernelError::InvalidMemoryRegion,
        ),
        (
            good,
            SHARED,
            PAGE_SIZE,
            read_write,
            KernelError::InvalidSize,
        ),
        (
            good,
            busy,
            SIZE,
            read_write,
            KernelError::InvalidNewMemoryPermission,
        ),
        (good, busy, SIZE, read, KernelError::InvalidCurrentMemory),
    ];
    let before = world.blocks();
    for (handle, address, size, permission, error) in cases {
        let at = (handle, address, size, permission);
        assert_eq!(
            world.map(handle, address, size, permission),
            refused(error),
            "{at:x?}"
        );
        assert_eq!(world.blocks(), before, "{at:x?}");
        assert_eq!(world.refs(shared.object), 2, "{at:x?}");
        assert_eq!(world.log.take(), [], "{at:x?}");
    }
}

#[test]
fn a_mapping_must_lie_in_the_alias_code_region_up_to_its_last_byte() {
    let mut world = World::new();
    let shared = world.shared(PAGES, SvcPermission::Read);
    let read = raw(SvcPermission::Read);
    let regions = *world.space.regions();
    let region = refused(KernelError::InvalidMemoryRegion);
    // Ending exactly at the region's end is inside; one page higher is not, and neither is a
    // range that starts one page below the region or ends exactly where it starts.
    let last = regions.code.end - SIZE;
    assert_eq!(
        world.map(shared.handle, last + PAGE_SIZE, SIZE, read),
        region
    );
    assert_eq!(
        world.map(shared.handle, regions.code.start - PAGE_SIZE, SIZE, read),
        region
    );
    assert_eq!(
        world.map(shared.handle, regions.code.start - SIZE, SIZE, read),
        region
    );
    // Overlapping the alias region or the heap by one page.
    assert_eq!(
        world.map(shared.handle, regions.alias.start - PAGE_SIZE, SIZE, read),
        region
    );
    assert_eq!(
        world.map(shared.handle, regions.heap.end - PAGE_SIZE, SIZE, read),
        region
    );
    assert_eq!(world.map(shared.handle, last, SIZE, read), Ok(()));
    assert_eq!(world.block(last).2, MemoryState::Shared);
}

#[test]
fn mapping_one_shared_memory_twice_at_one_place_is_refused_and_at_two_places_is_not() {
    let mut world = World::new();
    let shared = world.shared(PAGES, SvcPermission::Read);
    let read = raw(SvcPermission::Read);
    world.map(shared.handle, SHARED, SIZE, read).unwrap();
    drop(world.log.take());
    let before = world.blocks();
    // `MapPageGroup` finds the range not free; `RemoveSharedMemory` gives the reference back.
    assert_eq!(
        world.map(shared.handle, SHARED, SIZE, read),
        refused(KernelError::InvalidCurrentMemory)
    );
    // Overlapping the mapping by one page.
    assert_eq!(
        world.map(shared.handle, SHARED + SIZE - PAGE_SIZE, SIZE, read),
        refused(KernelError::InvalidCurrentMemory)
    );
    assert_eq!((world.blocks(), world.refs(shared.object)), (before, 3));
    // Right after the first mapping: the same pages, mapped again, with a reference each.
    let second = SHARED + SIZE;
    assert_eq!(world.map(shared.handle, second, SIZE, read), Ok(()));
    assert_eq!(world.refs(shared.object), 4);
    assert_eq!(
        world.log.take(),
        [MapperCall::MapShared {
            address: second,
            backing: shared.pages.backing(),
            perms: Perms::Read,
        }]
    );
    // The two mappings stay two blocks (each starts a mapping), and each unmaps alone.
    assert_eq!(world.block(SHARED).0, SHARED);
    assert_eq!(world.block(second).0, second);
    assert_eq!(world.unmap(shared.handle, second, SIZE), Ok(()));
    assert_eq!(world.block(SHARED).2, MemoryState::Shared);
    assert_eq!(world.unmap(shared.handle, SHARED, SIZE), Ok(()));
    assert_eq!(world.refs(shared.object), 2);
}

#[test]
fn unmapping_shared_memory_needs_the_mapping_of_that_shared_memory_at_its_start() {
    let mut world = World::new();
    let first = world.shared(PAGES, SvcPermission::Read);
    let second = world.shared(PAGES, SvcPermission::Read);
    let read = raw(SvcPermission::Read);
    // Two mappings of two shared memories, the second starting where the first ends.
    let next = SHARED + SIZE;
    world.map(first.handle, SHARED, SIZE, read).unwrap();
    world.map(second.handle, next, SIZE, read).unwrap();
    drop(world.log.take());
    let before = world.blocks();
    let invalid = refused(KernelError::InvalidCurrentMemory);
    // The other shared memory's mapping.
    assert_eq!(world.unmap(first.handle, next, SIZE), invalid);
    // A range of the right size and state throughout, starting inside the first mapping and
    // ending inside the second: the state check passes, the page check does not.
    assert_eq!(world.unmap(first.handle, SHARED + PAGE_SIZE, SIZE), invalid);
    assert_eq!(
        world.unmap(second.handle, SHARED + PAGE_SIZE, SIZE),
        invalid
    );
    assert_eq!(world.blocks(), before);
    assert_eq!(
        (world.refs(first.object), world.refs(second.object)),
        (3, 3)
    );
    assert_eq!(world.log.take(), []);
    // The neighbour that starts exactly where the unmapped range ends is left as it was.
    let neighbour = world.block(next);
    assert_eq!(world.unmap(first.handle, SHARED, SIZE), Ok(()));
    assert_eq!(world.block(next), neighbour);
    assert_eq!(world.block(SHARED).2, MemoryState::Free);
    assert_eq!(world.refs(second.object), 3);
}

#[test]
fn each_refusal_of_an_unmapping_has_its_result_in_eden_order_and_changes_nothing() {
    let mut world = World::new();
    let shared = world.shared(PAGES, SvcPermission::Read);
    let read = raw(SvcPermission::Read);
    world.map(shared.handle, SHARED, SIZE, read).unwrap();
    drop(world.log.take());
    let heap = world.space.regions().heap.start;
    let before = world.blocks();
    // Each case breaks one check and, where there is one, the check after it too.
    let cases = [
        (SHARED + 1, 0, shared.handle, KernelError::InvalidAddress),
        (SHARED, SIZE + 1, Handle::INVALID, KernelError::InvalidSize),
        (SHARED, 0, shared.handle, KernelError::InvalidSize),
        (
            u64::MAX - PAGE_SIZE + 1,
            SIZE,
            Handle::INVALID,
            KernelError::InvalidCurrentMemory,
        ),
        (heap, SIZE, Handle::INVALID, KernelError::InvalidHandle),
        (
            heap,
            PAGE_SIZE,
            shared.handle,
            KernelError::InvalidMemoryRegion,
        ),
        (SHARED, PAGE_SIZE, shared.handle, KernelError::InvalidSize),
        // Code memory, not shared memory.
        (
            fixture::DATA,
            SIZE,
            shared.handle,
            KernelError::InvalidCurrentMemory,
        ),
        // Free memory.
        (
            SHARED + SIZE,
            SIZE,
            shared.handle,
            KernelError::InvalidCurrentMemory,
        ),
    ];
    for (address, size, handle, error) in cases {
        let at = (address, size, handle);
        assert_eq!(
            world.unmap(handle, address, size),
            refused(error),
            "{at:x?}"
        );
        assert_eq!(world.blocks(), before, "{at:x?}");
        assert_eq!(world.refs(shared.object), 3, "{at:x?}");
        assert_eq!(world.log.take(), [], "{at:x?}");
    }
}

#[test]
fn a_mapping_keeps_its_shared_memory_alive_after_every_handle_and_the_server_let_go() {
    let mut world = World::new();
    let shared = world.shared(1, SvcPermission::Read);
    let read = raw(SvcPermission::Read);
    world.map(shared.handle, SHARED, PAGE_SIZE, read).unwrap();
    // The server's reference, then the guest's handle.
    world.scheduler.close_object(shared.object).unwrap();
    let removed = world
        .handles
        .remove(world.scheduler.objects_mut(), shared.handle)
        .unwrap();
    assert_ne!(removed.closed, Closed::Last);
    assert_eq!(world.refs(shared.object), 1);
    // With no handle the guest cannot name it to unmap it: the mapping stays, and so does the
    // object.
    assert_eq!(
        world.unmap(shared.handle, SHARED, PAGE_SIZE),
        refused(KernelError::InvalidHandle)
    );
    assert_eq!(world.block(SHARED).2, MemoryState::Shared);
}

#[test]
fn unmapping_the_last_mapping_of_a_shared_memory_with_no_server_leaves_it_to_its_handle() {
    let mut world = World::new();
    let shared = world.shared(1, SvcPermission::Read);
    let read = raw(SvcPermission::Read);
    world.map(shared.handle, SHARED, PAGE_SIZE, read).unwrap();
    // The server lets go; the guest's handle and the mapping are left.
    world.scheduler.close_object(shared.object).unwrap();
    assert_eq!(world.unmap(shared.handle, SHARED, PAGE_SIZE), Ok(()));
    assert_eq!(world.refs(shared.object), 1);
    let removed = world
        .handles
        .remove(world.scheduler.objects_mut(), shared.handle)
        .unwrap();
    assert_eq!(removed.closed, Closed::Last);
    assert_eq!(world.refs(shared.object), 0);
}

#[test]
fn the_size_checked_is_the_objects_not_the_one_asked_for() {
    let mut world = World::new();
    let small = world.shared(1, SvcPermission::Read);
    let large = world.shared(PAGES, SvcPermission::Read);
    let read = raw(SvcPermission::Read);
    let size = refused(KernelError::InvalidSize);
    assert_eq!(world.map(small.handle, SHARED, SIZE, read), size);
    assert_eq!(world.map(large.handle, SHARED, PAGE_SIZE, read), size);
    assert_eq!(world.map(small.handle, SHARED, PAGE_SIZE, read), Ok(()));
    assert_eq!(world.unmap(small.handle, SHARED, SIZE), size);
    assert_eq!(world.unmap(small.handle, SHARED, PAGE_SIZE), Ok(()));
}

/// A translation that hands out one page more than it is asked for.
struct Generous;

impl crate::memory::Mapper for Generous {
    fn map(&mut self, _: u64, _: u64, _: Perms) -> Result<(), crate::memory::MapperError> {
        Ok(())
    }

    fn unmap(&mut self, _: u64, _: u64) -> Result<(), crate::memory::MapperError> {
        Ok(())
    }

    fn protect(&mut self, _: u64, _: u64, _: Perms) -> Result<(), crate::memory::MapperError> {
        Ok(())
    }

    fn allocate_shared(&mut self, size: u64) -> Result<SharedPages, crate::memory::MapperError> {
        Ok(crate::memory::testing::standalone_shared_pages(size + PAGE_SIZE).unwrap())
    }
}

#[test]
fn the_address_space_takes_only_whole_pages_of_the_size_it_asked_for() {
    let memory = fixture::memory(crate::memory::MemoryArrangement::FourGb);
    let mut generous = AddressSpace::new(&memory, Box::new(Generous)).unwrap();
    assert_eq!(
        generous
            .allocate_shared(PAGE_SIZE)
            .map(|pages| pages.size()),
        Err(SvcError::Internal(format!(
            "the translation gave {:#x} bytes of shared memory for {PAGE_SIZE:#x}",
            2 * PAGE_SIZE
        )))
    );
    let (mut space, _, _) = fixture::loaded();
    for size in [0, PAGE_SIZE + 1] {
        assert_eq!(
            space.allocate_shared(size).map(|pages| pages.size()),
            Err(SvcError::Internal(format!(
                "shared memory of {size:#x} bytes is not whole pages"
            )))
        );
    }
    assert_eq!(
        space.allocate_shared(PAGE_SIZE).map(|pages| pages.size()),
        Ok(PAGE_SIZE)
    );
}

#[test]
fn a_translation_with_the_default_has_no_shared_backing() {
    struct Plain;
    impl crate::memory::Mapper for Plain {
        fn map(&mut self, _: u64, _: u64, _: Perms) -> Result<(), crate::memory::MapperError> {
            Ok(())
        }

        fn unmap(&mut self, _: u64, _: u64) -> Result<(), crate::memory::MapperError> {
            Ok(())
        }

        fn protect(&mut self, _: u64, _: u64, _: Perms) -> Result<(), crate::memory::MapperError> {
            Ok(())
        }
    }
    let memory = fixture::memory(crate::memory::MemoryArrangement::FourGb);
    let mut plain = Plain;
    let refused = crate::memory::MapperError(crate::memory::mapper::NO_SHARED_BACKING.into());
    assert_eq!(
        crate::memory::Mapper::allocate_shared(&mut plain, PAGE_SIZE).map(|pages| pages.size()),
        Err(refused.clone())
    );
    let backing = starquake_memory::BackingRange::new(0, PAGE_SIZE).unwrap();
    assert_eq!(
        crate::memory::Mapper::map_shared(&mut plain, SHARED, backing, Perms::Read),
        Err(refused.clone())
    );
    let mut space = AddressSpace::new(&memory, Box::new(Plain)).unwrap();
    assert_eq!(
        space.allocate_shared(PAGE_SIZE).map(|pages| pages.size()),
        Err(SvcError::Internal(format!(
            "the translation gave no backing for {PAGE_SIZE:#x} bytes of shared memory: {refused}"
        )))
    );
}
