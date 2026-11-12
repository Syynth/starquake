//! `CreateTransferMemory` over the fixture's address space with a heap: what lending does to
//! the owner's pages, what the owner can no longer do while they are lent, how they come back
//! when the transfer memory goes, and each refusal with its result, in Eden's order.

use starquake_memory::{PAGE_SIZE, Perms};

use super::*;
use crate::handle::{Closed, ObjectRefs};
use crate::id::SlabId;
use crate::memory::fixture;
use crate::memory::testing::{MapperCall, MapperLog};
use crate::memory::{MemoryAttribute, MemoryBlock, MemoryPermission, MemoryState};
use crate::object::{ObjectError, Objects};
use crate::svc::handle::close_handle;
use crate::svc::memory::{
    HEAP_SIZE_ALIGNMENT, set_heap_size, set_memory_attribute, set_memory_permission,
};

/// Pages the tests usually lend.
const PAGES: u64 = 2;
/// Their size in bytes.
const SIZE: u64 = PAGES * PAGE_SIZE;

fn raw(permission: SvcPermission) -> u32 {
    u32::from(permission.bits())
}

fn refused<T>(error: KernelError) -> CallResult<T> {
    Err(SvcError::Kernel(error))
}

fn owner() -> ProcessId {
    ProcessId::from_parts(0, 1)
}

/// A loaded fixture with one heap's worth of heap, a handle table and a scheduler beside it.
struct World {
    handles: HandleTable,
    space: AddressSpace,
    scheduler: Scheduler,
    log: MapperLog,
    heap: u64,
}

impl World {
    fn new() -> Self {
        let (mut space, log, _) = fixture::loaded();
        let heap = space.set_heap_size(HEAP_SIZE_ALIGNMENT).unwrap();
        drop(log.take());
        Self {
            handles: HandleTable::new(8).unwrap(),
            space,
            scheduler: Scheduler::default(),
            log,
            heap,
        }
    }

    fn create(&mut self, address: u64, size: u64, permission: u32) -> CallResult<Handle> {
        create_transfer_memory(
            &mut self.handles,
            &mut self.space,
            &mut self.scheduler,
            owner(),
            address,
            size,
            permission,
        )
    }

    /// Closes `handle` as `CloseHandle` does, then does what the dispatcher does before it
    /// lets the kernel lock go.
    fn close(&mut self, handle: Handle) {
        close_handle(&mut self.handles, &mut self.scheduler, handle).unwrap();
        unlock_released(&mut self.scheduler, &mut self.space, owner()).unwrap();
    }

    fn transfer(&self, handle: Handle) -> TransferMemory {
        let object = self.handles.get(handle).unwrap();
        match self.scheduler.objects().get(object).unwrap() {
            Object::TransferMemory(transfer) => transfer,
            other => unreachable!("{other:?} is not a transfer memory"),
        }
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
fn lending_locks_the_pages_and_leaves_the_owner_the_permission_it_kept() {
    for (kept, translated) in [
        (SvcPermission::None, Some(Perms::None)),
        (SvcPermission::Read, Some(Perms::Read)),
        // Read-write is what the pages had: the translation is not asked.
        (SvcPermission::ReadWrite, None),
    ] {
        let mut world = World::new();
        let heap = world.heap;
        let handle = world.create(heap, SIZE, raw(kept)).unwrap();
        assert_eq!(
            world.block(heap),
            (
                heap,
                PAGES,
                MemoryState::Normal,
                MemoryPermission::from_svc(kept),
                MemoryAttribute::LOCKED
            ),
            "{kept:?}"
        );
        let protect = translated.map(|perms| MapperCall::Protect {
            address: heap,
            size: SIZE,
            perms,
        });
        assert_eq!(world.log.take(), Vec::from_iter(protect), "{kept:?}");
        // The rest of the heap, from the page right after the lent range, is as it was.
        assert_eq!(
            world.block(heap + SIZE),
            (
                heap + SIZE,
                HEAP_SIZE_ALIGNMENT / PAGE_SIZE - PAGES,
                MemoryState::Normal,
                MemoryPermission::USER_READ_WRITE,
                MemoryAttribute::NONE
            ),
            "{kept:?}"
        );
        // The handle holds the only reference, to a transfer memory with its pages locked.
        let transfer = world.transfer(handle);
        assert_eq!(
            transfer,
            TransferMemory::new(owner(), heap, SIZE, kept).with_pages_locked()
        );
        let object = world.handles.get(handle).unwrap();
        assert_eq!(world.scheduler.objects().refs(object), 1);
        // Given back, the pages are read-write again; the translation is asked only when
        // they were not.
        world.close(handle);
        let restore = translated.map(|_| MapperCall::Protect {
            address: heap,
            size: SIZE,
            perms: Perms::ReadWrite,
        });
        assert_eq!(world.log.take(), Vec::from_iter(restore), "{kept:?}");
        assert_eq!(
            world.block(heap),
            (
                heap,
                HEAP_SIZE_ALIGNMENT / PAGE_SIZE,
                MemoryState::Normal,
                MemoryPermission::USER_READ_WRITE,
                MemoryAttribute::NONE
            ),
            "{kept:?}"
        );
    }
}

#[test]
fn while_lent_the_owner_cannot_change_the_pages_and_closing_the_handle_gives_them_back() {
    let mut world = World::new();
    let heap = world.heap;
    let handle = world.create(heap, SIZE, raw(SvcPermission::None)).unwrap();
    drop(world.log.take());
    let lent = world.blocks();
    let invalid = refused::<()>(KernelError::InvalidCurrentMemory);
    // The owner's own calls on the lent pages.
    assert_eq!(
        set_memory_permission(&mut world.space, heap, PAGE_SIZE, raw(SvcPermission::Read)),
        invalid
    );
    let uncached = MemoryAttribute::UNCACHED.svc();
    assert_eq!(
        set_memory_attribute(&mut world.space, heap, PAGE_SIZE, uncached, uncached),
        invalid
    );
    // Shrinking the heap would free them.
    assert_eq!(set_heap_size(&mut world.space, 0).map(|_| ()), invalid);
    // Lending them, or a range that overlaps them by one page, a second time.
    assert_eq!(
        world.create(heap, SIZE, raw(SvcPermission::None)),
        refused(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(
        world.create(heap + SIZE - PAGE_SIZE, SIZE, raw(SvcPermission::None)),
        refused(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(world.blocks(), lent);
    assert_eq!(world.log.take(), []);
    // The pages right after them can still be lent.
    let next = world
        .create(heap + SIZE, SIZE, raw(SvcPermission::Read))
        .unwrap();
    drop(world.log.take());
    // The transfer memory goes with its handle; its pages come back read-write and unlocked,
    // and the neighbour lent after it is left as it is.
    let neighbour = world.block(heap + SIZE);
    world.close(handle);
    assert_eq!(
        world.block(heap),
        (
            heap,
            PAGES,
            MemoryState::Normal,
            MemoryPermission::USER_READ_WRITE,
            MemoryAttribute::NONE
        )
    );
    assert_eq!(world.block(heap + SIZE), neighbour);
    assert_eq!(
        world.log.take(),
        [MapperCall::Protect {
            address: heap,
            size: SIZE,
            perms: Perms::ReadWrite,
        }]
    );
    world.close(next);
    // Everything back as it was: the heap is one block again and can go.
    assert_eq!(
        world.block(heap),
        (
            heap,
            HEAP_SIZE_ALIGNMENT / PAGE_SIZE,
            MemoryState::Normal,
            MemoryPermission::USER_READ_WRITE,
            MemoryAttribute::NONE
        )
    );
    assert_eq!(set_heap_size(&mut world.space, 0), Ok(heap));
}

#[test]
fn a_transfer_memory_with_another_handle_keeps_its_pages_lent() {
    let mut world = World::new();
    let heap = world.heap;
    let handle = world.create(heap, SIZE, raw(SvcPermission::Read)).unwrap();
    let object = world.handles.get(handle).unwrap();
    let second = world
        .handles
        .add(world.scheduler.objects_mut(), object)
        .unwrap();
    world.close(handle);
    assert_eq!(world.block(heap).4, MemoryAttribute::LOCKED);
    world.close(second);
    assert_eq!(world.block(heap).4, MemoryAttribute::NONE);
}

#[test]
fn only_read_write_memory_that_can_be_transferred_with_one_state_throughout_is_lent() {
    let mut world = World::new();
    let heap = world.heap;
    let none = raw(SvcPermission::None);
    let invalid = refused(KernelError::InvalidCurrentMemory);
    // Text and read-only data are code: no `FlagCanTransfer`.
    assert_eq!(world.create(fixture::CODE, PAGE_SIZE, none), invalid);
    // Free memory.
    assert_eq!(
        world.create(heap + HEAP_SIZE_ALIGNMENT, PAGE_SIZE, none),
        invalid
    );
    // Heap made read-only.
    set_memory_permission(&mut world.space, heap, PAGE_SIZE, raw(SvcPermission::Read)).unwrap();
    assert_eq!(world.create(heap, PAGE_SIZE, none), invalid);
    // Two pages whose permissions differ.
    assert_eq!(world.create(heap, SIZE, none), invalid);
    // The data segment can be lent, as the heap can.
    assert!(world.create(fixture::DATA, PAGE_SIZE, none).is_ok());
    assert!(world.create(heap + PAGE_SIZE, PAGE_SIZE, none).is_ok());
}

#[test]
fn each_refusal_has_its_result_in_eden_order_and_changes_nothing() {
    let mut world = World::new();
    let heap = world.heap;
    let end = world.space.regions().address_space.end;
    let (none, execute) = (raw(SvcPermission::None), raw(SvcPermission::ReadExecute));
    let wraps = u64::MAX - PAGE_SIZE + 1;
    // Each case breaks one check and, where there is one, the check after it too.
    let cases = [
        (heap + 1, SIZE, execute, KernelError::InvalidAddress),
        (wraps, SIZE + 1, execute, KernelError::InvalidSize),
        (heap, 0, none, KernelError::InvalidSize),
        (wraps, SIZE, execute, KernelError::InvalidCurrentMemory),
        (end, SIZE, execute, KernelError::InvalidNewMemoryPermission),
        (
            heap,
            SIZE,
            u32::MAX,
            KernelError::InvalidNewMemoryPermission,
        ),
        // Above the address space (`Contains`), and ending one page past it.
        (end, SIZE, none, KernelError::InvalidCurrentMemory),
        (
            end - PAGE_SIZE,
            SIZE,
            none,
            KernelError::InvalidCurrentMemory,
        ),
    ];
    let before = world.blocks();
    for (address, size, permission, error) in cases {
        let at = (address, size, permission);
        assert_eq!(
            world.create(address, size, permission),
            refused(error),
            "{at:x?}"
        );
        assert_eq!(world.blocks(), before, "{at:x?}");
        assert!(world.scheduler.objects().is_empty(), "{at:x?}");
        assert!(world.handles.is_empty(), "{at:x?}");
        assert_eq!(world.log.take(), [], "{at:x?}");
    }
}

#[test]
fn a_full_object_table_is_reported_before_the_range_is_looked_at() {
    let mut world = World::new();
    *world.scheduler.objects_mut() = Objects::with_max_slots(0);
    let end = world.space.regions().address_space.end;
    let none = raw(SvcPermission::None);
    let before = world.blocks();
    assert_eq!(
        world.create(end, SIZE, none),
        refused(KernelError::OutOfResource)
    );
    assert_eq!(
        world.create(world.heap, SIZE, none),
        refused(KernelError::OutOfResource)
    );
    assert_eq!(world.blocks(), before);
}

#[test]
fn a_full_handle_table_gives_the_pages_back_before_the_call_returns() {
    let mut world = World::new();
    world.handles = HandleTable::new(1).unwrap();
    let heap = world.heap;
    let first = world
        .create(heap, PAGE_SIZE, raw(SvcPermission::None))
        .unwrap();
    drop(world.log.take());
    let before = world.blocks();
    assert_eq!(
        world.create(heap + PAGE_SIZE, PAGE_SIZE, raw(SvcPermission::None)),
        refused(KernelError::OutOfHandles)
    );
    // Locked and then unlocked: the translation was told both, and the blocks are as before.
    assert_eq!(world.blocks(), before);
    assert_eq!(
        world.log.take(),
        [
            MapperCall::Protect {
                address: heap + PAGE_SIZE,
                size: PAGE_SIZE,
                perms: Perms::None,
            },
            MapperCall::Protect {
                address: heap + PAGE_SIZE,
                size: PAGE_SIZE,
                perms: Perms::ReadWrite,
            },
        ]
    );
    // Only the first transfer memory is left, and nothing waits to be unlocked.
    assert_eq!(world.scheduler.objects().len(), 1);
    assert!(!world.scheduler.objects().has_unlocks());
    assert_eq!(world.transfer(first).address(), heap);
}

#[test]
fn unlocking_goes_on_past_a_failure_and_reports_the_first() {
    let mut world = World::new();
    let heap = world.heap;
    let first = world
        .create(heap, PAGE_SIZE, raw(SvcPermission::None))
        .unwrap();
    let second = world
        .create(heap + PAGE_SIZE, PAGE_SIZE, raw(SvcPermission::None))
        .unwrap();
    // Transfer memories of two other processes went first, one after the other; then both of
    // these.
    let stranger = ProcessId::from_parts(7, 1);
    let later_stranger = ProcessId::from_parts(8, 1);
    for owner in [stranger, later_stranger] {
        let foreign =
            TransferMemory::new(owner, heap, PAGE_SIZE, SvcPermission::None).with_pages_locked();
        let stray = world
            .scheduler
            .objects_mut()
            .create(Object::TransferMemory(foreign))
            .unwrap();
        assert_eq!(world.scheduler.objects_mut().close(stray), Ok(Closed::Last));
    }
    close_handle(&mut world.handles, &mut world.scheduler, first).unwrap();
    close_handle(&mut world.handles, &mut world.scheduler, second).unwrap();
    let unlocked = unlock_released(&mut world.scheduler, &mut world.space, owner());
    assert_eq!(
        unlocked,
        Err(SvcError::Internal(format!(
            "a transfer memory of {stranger:?} went in the tables of {:?}",
            owner()
        )))
    );
    // Both of the process's own were unlocked all the same.
    assert_eq!(world.block(heap).4, MemoryAttribute::NONE);
    assert_eq!(world.block(heap).1, HEAP_SIZE_ALIGNMENT / PAGE_SIZE);
    assert!(!world.scheduler.objects().has_unlocks());
}

#[test]
fn adjacent_lends_stay_two_blocks() {
    let mut world = World::new();
    let heap = world.heap;
    let kept = raw(SvcPermission::Read);
    // The same state, permission and attributes on both sides: only the merge barrier that
    // locking sets at each lent range's head keeps them apart.
    let first = world.create(heap, SIZE, kept).unwrap();
    let second = world.create(heap + SIZE, SIZE, kept).unwrap();
    assert_eq!(world.block(heap).1, PAGES);
    assert_eq!(world.block(heap + SIZE).0, heap + SIZE);
    assert_eq!(world.block(heap + SIZE).1, PAGES);
    world.close(first);
    world.close(second);
    assert_eq!(world.block(heap).1, HEAP_SIZE_ALIGNMENT / PAGE_SIZE);
}

#[test]
fn a_failure_after_the_pages_are_locked_unlocks_them_again() {
    let mut world = World::new();
    let heap = world.heap;
    // An object id that names nothing any more, so recording the lock in it fails.
    let unlocked = TransferMemory::new(owner(), heap, SIZE, SvcPermission::None);
    let stale = world
        .scheduler
        .objects_mut()
        .create(Object::TransferMemory(unlocked))
        .unwrap();
    assert_eq!(world.scheduler.objects_mut().close(stale), Ok(Closed::Last));
    let before = world.blocks();
    let added = lock_and_add(
        &mut world.handles,
        &mut world.space,
        &mut world.scheduler,
        stale,
        unlocked,
    );
    assert_eq!(added, Err(SvcError::from(ObjectError::Stale { id: stale })));
    // Locked, then unlocked again: the translation was told both, and the blocks are as before.
    assert_eq!(world.blocks(), before);
    assert_eq!(
        world.log.take(),
        [
            MapperCall::Protect {
                address: heap,
                size: SIZE,
                perms: Perms::None,
            },
            MapperCall::Protect {
                address: heap,
                size: SIZE,
                perms: Perms::ReadWrite,
            },
        ]
    );
    assert!(world.handles.is_empty());
    assert!(!world.scheduler.objects().has_unlocks());
    // The pages can be lent again.
    assert!(world.create(heap, SIZE, raw(SvcPermission::None)).is_ok());
}

/// Eden's limit on a process's transfer memories (`TransferMemoryCountMax`, set to 200 in
/// `CreateResourceLimitForProcess`, `eden/src/core/hle/kernel/k_resource_limit.cpp` @
/// cf654052). Starquake has no such limit yet.
const EDEN_TRANSFER_MEMORY_COUNT_MAX: u64 = 200;

#[test]
fn more_transfer_memories_than_edens_limit_are_all_lent() {
    let mut world = World::new();
    world.handles = HandleTable::new(0).unwrap();
    let heap = world.heap;
    let count = EDEN_TRANSFER_MEMORY_COUNT_MAX + 1;
    assert!(count * PAGE_SIZE <= HEAP_SIZE_ALIGNMENT);
    for page in 0..count {
        let address = heap + page * PAGE_SIZE;
        assert!(
            world
                .create(address, PAGE_SIZE, raw(SvcPermission::Read))
                .is_ok(),
            "{page}"
        );
    }
    assert_eq!(world.handles.len(), usize::try_from(count).unwrap());
}
