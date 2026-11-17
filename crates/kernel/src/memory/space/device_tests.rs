//! The graphics server's two address-space calls (kernel spec, "Address space"): the range
//! query of the GPU-accessible attribute, and the device-share lock and unlock.
//!
//! "Eden" below is `eden/src/core/hle/kernel/k_page_table_base.cpp` @ cf654052; "PR" is
//! `eden-pr4588/src/core/hle/service/am/service/graphics_supervisor.cpp` @ e5fb3b97.

use super::*;
use crate::memory::fixture::{CODE, DATA, SHARED, loaded, shared_memory};
use crate::object::Objects;

const INVALID: CallResult<()> = Err(SvcError::Kernel(KernelError::InvalidCurrentMemory));
const GPU: MemoryAttribute = MemoryAttribute::GPU_ACCESSIBLE;

/// The fixture's loaded address space with `pages` pages of heap. Returns the heap's start.
fn with_heap(pages: u64) -> (AddressSpace, u64) {
    let (mut space, _log, _) = loaded();
    space.set_heap_size(pages * PAGE_SIZE).unwrap();
    let heap = space.regions().heap.start;
    (space, heap)
}

/// Every block, as `QueryMemory` would report it.
fn blocks(space: &AddressSpace) -> Vec<SvcMemoryInfoRow> {
    space
        .blocks()
        .map(|block| {
            let info = block.svc_info();
            (
                info.base_address,
                info.size,
                info.attribute,
                info.device_count,
            )
        })
        .collect()
}

/// Where a block is, its attributes and its device count.
type SvcMemoryInfoRow = (u64, u64, u32, u32);

// ----- The GPU-accessible range query ------------------------------------------------------

#[test]
fn the_gpu_accessible_query_is_true_only_when_every_page_touched_has_the_attribute() {
    // Heap pages 0-5. Pages 1-2 get the attribute; page 2 is also uncached, so 1 and 2 are two
    // blocks that both have it; page 3 does not.
    let (mut space, heap) = with_heap(6);
    let page = |index: u64| heap + index * PAGE_SIZE;
    space
        .set_memory_attribute(page(1), 2 * PAGE_SIZE, GPU, GPU)
        .unwrap();
    let uncached = MemoryAttribute::UNCACHED;
    space
        .set_memory_attribute(page(2), PAGE_SIZE, uncached, uncached)
        .unwrap();
    assert_ne!(
        space.query_info(page(1)).unwrap().address(),
        space.query_info(page(2)).unwrap().address()
    );
    let query = |address: u64, size: u64| space.is_gpu_accessible(address, size);
    // The marked pages, whole, across their two blocks.
    assert!(query(page(1), 2 * PAGE_SIZE));
    // Byte edges inside them: the first byte, the last byte, one byte anywhere between.
    assert!(query(page(1), 1));
    assert!(query(page(3) - 1, 1));
    assert!(query(page(1) + 0x123, 0x456));
    // One byte past either end touches an unmarked page.
    assert!(!query(page(1) - 1, 2 * PAGE_SIZE + 1));
    assert!(!query(page(1), 2 * PAGE_SIZE + 1));
    assert!(!query(page(3), 1));
    assert!(!query(page(1) - 1, 1));
    // An empty range, and one that wraps.
    assert!(!query(page(1), 0));
    assert!(!query(page(1), u64::MAX));
    assert!(!query(u64::MAX, 2));
}

#[test]
fn the_gpu_accessible_query_is_false_past_the_end_of_the_address_space() {
    // The attribute on the last page of the address space can't be had (nothing is mapped
    // there), so this checks the walk stops at the end of the map: a range starting at the
    // end, and one ending one byte past it, are false, as is the free last page.
    let (space, _) = with_heap(1);
    let end = space.regions().address_space.end;
    assert!(!space.is_gpu_accessible(end, PAGE_SIZE));
    assert!(!space.is_gpu_accessible(end - PAGE_SIZE, PAGE_SIZE + 1));
    assert!(!space.is_gpu_accessible(end - PAGE_SIZE, PAGE_SIZE));
}

#[test]
fn the_gpu_accessible_attribute_changes_only_where_the_state_can_change_attributes() {
    // Eden-NX2 and PR: a mask with `GpuAccessible` needs `FlagCanChangeAttribute`, which heap
    // and data have and text and the stack do not.
    let (mut space, _log, stack_top) = loaded();
    space.set_heap_size(PAGE_SIZE).unwrap();
    let heap = space.regions().heap.start;
    for address in [heap, DATA] {
        assert_eq!(
            space.set_memory_attribute(address, PAGE_SIZE, GPU, GPU),
            Ok(()),
            "{address:#x}"
        );
        assert!(space.is_gpu_accessible(address, PAGE_SIZE));
        assert_eq!(
            space.set_memory_attribute(address, PAGE_SIZE, GPU, MemoryAttribute::NONE),
            Ok(())
        );
        assert!(!space.is_gpu_accessible(address, PAGE_SIZE));
    }
    for address in [CODE, stack_top - PAGE_SIZE] {
        assert_eq!(
            space.set_memory_attribute(address, PAGE_SIZE, GPU, GPU),
            INVALID,
            "{address:#x}"
        );
    }
    // A block with the attribute may have its other attributes changed: `SetMask` holds it,
    // so `AttributeTestMask` leaves it out.
    space
        .set_memory_attribute(heap, PAGE_SIZE, GPU, GPU)
        .unwrap();
    let uncached = MemoryAttribute::UNCACHED;
    assert_eq!(
        space.set_memory_attribute(heap, PAGE_SIZE, uncached, uncached),
        Ok(())
    );
    assert_eq!(space.query_info(heap).unwrap().attribute(), GPU | uncached);
}

// ----- The device-share lock ---------------------------------------------------------------

#[test]
fn a_device_lock_marks_every_block_of_the_range_and_unlocking_restores_them() {
    let (mut space, heap) = with_heap(4);
    let before = blocks(&space);
    let shared = u32::from(MemoryAttribute::DEVICE_SHARED.bits());
    assert_eq!(
        space.lock_device_share(heap + PAGE_SIZE, 2 * PAGE_SIZE),
        Ok(())
    );
    let at = |space: &AddressSpace, address| {
        let info = space.query_info(address).unwrap().svc_info();
        (
            info.base_address,
            info.size,
            info.attribute,
            info.device_count,
        )
    };
    assert_eq!(at(&space, heap), (heap, PAGE_SIZE, 0, 0));
    assert_eq!(
        at(&space, heap + PAGE_SIZE),
        (heap + PAGE_SIZE, 2 * PAGE_SIZE, shared, 1)
    );
    // The neighbour that starts exactly at the end of the range is unchanged.
    assert_eq!(
        at(&space, heap + 3 * PAGE_SIZE),
        (heap + 3 * PAGE_SIZE, PAGE_SIZE, 0, 0)
    );
    // Locks nest.
    assert_eq!(
        space.lock_device_share(heap + PAGE_SIZE, 2 * PAGE_SIZE),
        Ok(())
    );
    assert_eq!(at(&space, heap + PAGE_SIZE).3, 2);
    assert_eq!(
        space.unlock_device_share(heap + PAGE_SIZE, 2 * PAGE_SIZE),
        Ok(())
    );
    assert_eq!(at(&space, heap + PAGE_SIZE).3, 1);
    assert_eq!(
        space.unlock_device_share(heap + PAGE_SIZE, 2 * PAGE_SIZE),
        Ok(())
    );
    assert_eq!(blocks(&space), before);
}

#[test]
fn a_device_lock_is_refused_unless_the_range_is_alike_lockable_and_unlocked() {
    let (mut space, heap) = with_heap(4);
    let mut objects = Objects::new();
    let (object, memory) = {
        let (object, pages) = shared_memory(&mut space, &mut objects, 1, SvcPermission::ReadWrite);
        let memory = SharedMemory::new(pages.backing(), SvcPermission::ReadWrite).unwrap();
        (object, memory)
    };
    space
        .map_shared(SHARED, object, memory, SvcPermission::ReadWrite)
        .unwrap();
    // Page 2 of the heap is read-only: unlike its neighbours.
    space
        .set_memory_permission(heap + 2 * PAGE_SIZE, PAGE_SIZE, SvcPermission::Read)
        .unwrap();
    // Page 3 is lent to a transfer memory: `Locked`.
    space
        .lock_for_transfer_memory(heap + 3 * PAGE_SIZE, PAGE_SIZE, SvcPermission::None)
        .unwrap();
    let before = blocks(&space);
    let end = space.regions().address_space.end;
    for (address, size) in [
        // Not whole pages inside the address space (Eden's `Contains`, then pages).
        (heap + 1, PAGE_SIZE),
        (heap, PAGE_SIZE + 1),
        (heap, 0),
        (end, PAGE_SIZE),
        (end - PAGE_SIZE, 2 * PAGE_SIZE),
        (u64::MAX - PAGE_SIZE + 1, PAGE_SIZE),
        // Free memory, and shared memory: no `FlagCanAlignedDeviceMap`.
        (heap + 4 * PAGE_SIZE, PAGE_SIZE),
        (SHARED, PAGE_SIZE),
        // Two permissions in one range.
        (heap + PAGE_SIZE, 2 * PAGE_SIZE),
        // `Locked`.
        (heap + 3 * PAGE_SIZE, PAGE_SIZE),
    ] {
        assert_eq!(
            space.lock_device_share(address, size),
            INVALID,
            "{address:#x}+{size:#x}"
        );
    }
    assert_eq!(blocks(&space), before);
    // Each of the pages alone is fine, except the lent one.
    for page in 0..3 {
        assert_eq!(
            space.lock_device_share(heap + page * PAGE_SIZE, PAGE_SIZE),
            Ok(()),
            "page {page}"
        );
    }
}

#[test]
fn a_device_locked_range_refuses_what_would_change_its_mapping_until_unlocked() {
    let (mut space, heap) = with_heap(2);
    let page = heap + PAGE_SIZE;
    space.lock_device_share(page, PAGE_SIZE).unwrap();
    space.lock_device_share(page, PAGE_SIZE).unwrap();
    let refused = |space: &mut AddressSpace| {
        let before = blocks(space);
        assert_eq!(
            space.set_memory_permission(page, PAGE_SIZE, SvcPermission::Read),
            INVALID
        );
        // Shrinking the heap over the locked page.
        assert_eq!(space.set_heap_size(PAGE_SIZE).map(|_| ()), INVALID);
        assert_eq!(space.heap_size(), 2 * PAGE_SIZE);
        assert_eq!(
            space.lock_for_transfer_memory(page, PAGE_SIZE, SvcPermission::None),
            INVALID
        );
        assert_eq!(blocks(space), before);
    };
    refused(&mut space);
    // The attribute calls still work on it, and so does another lock.
    assert_eq!(
        space.set_memory_attribute(page, PAGE_SIZE, GPU, GPU),
        Ok(())
    );
    assert_eq!(
        space.set_memory_attribute(page, PAGE_SIZE, GPU, MemoryAttribute::NONE),
        Ok(())
    );
    // The first unlock of two leaves it locked.
    space.unlock_device_share(page, PAGE_SIZE).unwrap();
    refused(&mut space);
    // The last unlock frees it, and each refused operation then goes through.
    space.unlock_device_share(page, PAGE_SIZE).unwrap();
    assert_eq!(
        space.set_memory_permission(page, PAGE_SIZE, SvcPermission::Read),
        Ok(())
    );
    space
        .set_memory_permission(page, PAGE_SIZE, SvcPermission::ReadWrite)
        .unwrap();
    assert_eq!(
        space.lock_for_transfer_memory(page, PAGE_SIZE, SvcPermission::None),
        Ok(())
    );
    space.unlock_for_transfer_memory(page, PAGE_SIZE).unwrap();
    assert_eq!(space.set_heap_size(PAGE_SIZE), Ok(heap));
}

#[test]
fn locked_code_cannot_be_reprotected() {
    let (mut space, _log, _) = loaded();
    space.lock_device_share(CODE, PAGE_SIZE).unwrap();
    assert_eq!(
        space.set_process_memory_permission(CODE, PAGE_SIZE, SvcPermission::Read),
        INVALID
    );
    space.unlock_device_share(CODE, PAGE_SIZE).unwrap();
    assert_eq!(
        space.set_process_memory_permission(CODE, PAGE_SIZE, SvcPermission::Read),
        Ok(())
    );
}

#[test]
fn a_device_lock_past_the_largest_count_is_out_of_resource() {
    let (mut space, heap) = with_heap(1);
    // `u16::MAX` locks fit: the count reaches exactly its largest value.
    for _ in 0..u16::MAX {
        space.lock_device_share(heap, PAGE_SIZE).unwrap();
    }
    assert_eq!(space.query_info(heap).unwrap().device_use_count(), u16::MAX);
    let before = blocks(&space);
    assert_eq!(
        space.lock_device_share(heap, PAGE_SIZE),
        Err(SvcError::Kernel(KernelError::OutOfResource))
    );
    assert_eq!(blocks(&space), before);
    // Unlocking is still possible.
    assert_eq!(space.unlock_device_share(heap, PAGE_SIZE), Ok(()));
    assert_eq!(
        space.query_info(heap).unwrap().device_use_count(),
        u16::MAX - 1
    );
}

// ----- The device-share unlock -------------------------------------------------------------

#[test]
fn an_unlock_needs_every_block_shared_and_none_locked() {
    let (mut space, heap) = with_heap(4);
    space.lock_device_share(heap, 2 * PAGE_SIZE).unwrap();
    let before = blocks(&space);
    let end = space.regions().address_space.end;
    for (address, size) in [
        // Not whole pages inside the address space.
        (heap + 1, PAGE_SIZE),
        (heap, 0),
        (end, PAGE_SIZE),
        // A range that reaches an unshared page: one past the lock.
        (heap, 3 * PAGE_SIZE),
        // Never locked.
        (heap + 2 * PAGE_SIZE, PAGE_SIZE),
        // Free memory: no `FlagCanDeviceMap`.
        (heap + 4 * PAGE_SIZE, PAGE_SIZE),
    ] {
        assert_eq!(
            space.unlock_device_share(address, size),
            INVALID,
            "{address:#x}+{size:#x}"
        );
    }
    assert_eq!(blocks(&space), before);
    // The lock, unlocked exactly, goes.
    assert_eq!(space.unlock_device_share(heap, 2 * PAGE_SIZE), Ok(()));
    assert!(blocks(&space).iter().all(|row| row.3 == 0));
}

#[test]
fn an_unlock_takes_blocks_that_differ_where_a_lock_needs_them_alike() {
    // `UnlockForDeviceAddressSpace` checks each block on its own
    // (`CheckMemoryStateContiguous`); `LockForMapDeviceAddressSpace` wants them alike.
    let (mut space, heap) = with_heap(2);
    space.lock_device_share(heap, 2 * PAGE_SIZE).unwrap();
    let uncached = MemoryAttribute::UNCACHED;
    space
        .set_memory_attribute(heap + PAGE_SIZE, PAGE_SIZE, uncached, uncached)
        .unwrap();
    // A second lock over both now fails: the attributes differ (`Uncached` is not ignored).
    assert_eq!(space.lock_device_share(heap, 2 * PAGE_SIZE), INVALID);
    // The unlock still goes through.
    assert_eq!(space.unlock_device_share(heap, 2 * PAGE_SIZE), Ok(()));
    assert!(blocks(&space).iter().all(|row| row.3 == 0));
}

#[test]
fn an_unlock_that_ends_where_no_lock_ends_is_refused_and_changes_nothing() {
    let (mut space, heap) = with_heap(4);
    space.lock_device_share(heap, 3 * PAGE_SIZE).unwrap();
    let before = blocks(&space);
    // Less than the lock: it ends inside the locked block. Eden asserts here.
    assert_eq!(space.unlock_device_share(heap, 2 * PAGE_SIZE), INVALID);
    assert_eq!(space.unlock_device_share(heap, PAGE_SIZE), INVALID);
    assert_eq!(blocks(&space), before);
    // Its last page alone ends where the lock does, and so may be unlocked; the rest then
    // ends where no lock ends either.
    assert_eq!(
        space.unlock_device_share(heap + 2 * PAGE_SIZE, PAGE_SIZE),
        Ok(())
    );
    assert_eq!(space.unlock_device_share(heap, 2 * PAGE_SIZE), INVALID);
}
