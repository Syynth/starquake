//! Each memory call's argument checks, in Horizon's order, and what a passing call does.
//!
//! "Eden" below is `eden/src/core/hle/kernel/svc/svc_memory.cpp` @ cf654052 unless a test names
//! another file.

use starquake_memory::{FlatMemory, Perms};

use super::*;
use crate::memory::fixture::{CODE, DATA, DATA_PAGES, loaded, memory};
use crate::memory::testing::{MapperCall, RecordingMapper};
use crate::memory::{MemoryArrangement, MemoryPermission, MemoryState, SvcMemoryInfo};
use crate::result::SvcError;

fn kernel<T>(error: KernelError) -> CallResult<T> {
    Err(SvcError::Kernel(error))
}

/// The raw values a guest passes for a permission (`Svc::MemoryPermission` in `svc_types.h`):
/// read is bit 0, write bit 1, execute bit 2.
const PERM_NONE: u32 = 0;
const PERM_READ: u32 = 1;
const PERM_READ_WRITE: u32 = 3;
const PERM_READ_EXECUTE: u32 = 5;

/// The raw values a guest passes for the two settable attributes (`Svc::MemoryAttribute`):
/// `Uncached` is bit 3, `PermissionLocked` bit 4.
const ATTR_UNCACHED: u32 = 1 << 3;
const ATTR_PERMISSION_LOCKED: u32 = 1 << 4;

// ----- SetHeapSize ----------------------------------------------------------------------------

#[test]
fn set_heap_size_maps_the_heap_and_returns_the_heap_regions_start() {
    let (mut space, log, _) = loaded();
    let heap = space.regions().heap.start;
    // The size the fixture asks for.
    assert_eq!(set_heap_size(&mut space, 0xcd00_0000), Ok(heap));
    assert_eq!(
        log.take(),
        [MapperCall::Map {
            address: heap,
            size: 0xcd00_0000,
            perms: Perms::ReadWrite
        }]
    );
    assert_eq!(space.heap_size(), 0xcd00_0000);
    assert_eq!(set_heap_size(&mut space, 0), Ok(heap));
    assert_eq!(space.heap_size(), 0);
}

#[test]
fn a_heap_size_must_be_a_multiple_of_two_mebibytes() {
    let (mut space, log, _) = loaded();
    // `R_UNLESS(Common::IsAligned(size, HeapSizeAlignment), ResultInvalidSize)`
    // (`svc_physical_memory.cpp`).
    for size in [
        1,
        PAGE_SIZE,
        HEAP_SIZE_ALIGNMENT - 1,
        HEAP_SIZE_ALIGNMENT + PAGE_SIZE,
    ] {
        assert_eq!(
            set_heap_size(&mut space, size),
            kernel(KernelError::InvalidSize),
            "{size:#x}"
        );
    }
    assert_eq!(log.take(), []);
    assert_eq!(
        set_heap_size(&mut space, HEAP_SIZE_ALIGNMENT),
        Ok(space.regions().heap.start)
    );
}

#[test]
fn the_heap_size_bound_is_the_larger_of_eight_gib_and_the_heap_region() {
    // A 4 GiB console: the heap region is 8 GiB, which is `MainMemorySizeMax`, so Eden's bound
    // and PR #4588's are the same. One alignment unit below it is a valid size, which the
    // page table then refuses (with a stack of two units, it is more than the largest heap:
    // `R_UNLESS(size <= m_max_heap_size, ResultOutOfMemory)`); the bound itself is an invalid
    // size.
    let (mapper, _log) = RecordingMapper::new();
    let mut space =
        AddressSpace::new(&memory(MemoryArrangement::FourGb), Box::new(mapper)).unwrap();
    space
        .create_main_thread_stack(2 * HEAP_SIZE_ALIGNMENT)
        .unwrap();
    assert_eq!(space.regions().heap.size(), MAIN_MEMORY_SIZE_MAX);
    assert_eq!(
        set_heap_size(&mut space, MAIN_MEMORY_SIZE_MAX - HEAP_SIZE_ALIGNMENT),
        kernel(KernelError::OutOfMemory)
    );
    assert_eq!(
        set_heap_size(&mut space, MAIN_MEMORY_SIZE_MAX),
        kernel(KernelError::InvalidSize)
    );
    assert_eq!(
        set_heap_size(&mut space, u64::MAX - HEAP_SIZE_ALIGNMENT + 1),
        kernel(KernelError::InvalidSize)
    );

    // A 12 GiB console: the heap region is 12 GiB, and a heap of 8 GiB or more is a valid
    // size (PR #4588); Eden would refuse it as an invalid size.
    let (mapper, _log) = RecordingMapper::new();
    let mut large =
        AddressSpace::new(&memory(MemoryArrangement::TwelveGb), Box::new(mapper)).unwrap();
    large.create_main_thread_stack(PAGE_SIZE).unwrap();
    let region = large.regions().heap.size();
    assert!(region > MAIN_MEMORY_SIZE_MAX);
    assert_eq!(
        set_heap_size(&mut large, MAIN_MEMORY_SIZE_MAX),
        Ok(large.regions().heap.start)
    );
    assert_eq!(
        set_heap_size(&mut large, region - HEAP_SIZE_ALIGNMENT),
        kernel(KernelError::LimitReached)
    );
    assert_eq!(
        set_heap_size(&mut large, region),
        kernel(KernelError::InvalidSize)
    );
}

#[test]
fn a_heap_larger_than_the_pool_has_room_for_is_limit_reached() {
    let (mut space, log, _) = loaded();
    // What is left of the pool, rounded down to the alignment, fits; one unit more does not.
    let room = space.limit().free();
    let fits = room - room % HEAP_SIZE_ALIGNMENT;
    assert_eq!(
        set_heap_size(&mut space, fits + HEAP_SIZE_ALIGNMENT),
        kernel(KernelError::LimitReached)
    );
    assert_eq!(log.take(), []);
    assert_eq!(space.heap_size(), 0);
    assert_eq!(
        set_heap_size(&mut space, fits),
        Ok(space.regions().heap.start)
    );
}

// ----- SetMemoryPermission and SetMemoryAttribute: the shared checks ----------------------------

/// The argument checks the two calls share, each with the one after it passing: Eden's
/// `R_UNLESS` lines, in order.
#[test]
fn address_and_size_are_checked_in_horizons_order() {
    let (mut space, log, _) = loaded();
    let end = space.regions().address_space.end;
    let cases = [
        // `R_UNLESS(Common::IsAligned(address, PageSize), ResultInvalidAddress)`, before the
        // size is looked at.
        (DATA + 1, 0, KernelError::InvalidAddress),
        (
            DATA + PAGE_SIZE - 1,
            PAGE_SIZE + 1,
            KernelError::InvalidAddress,
        ),
        // `R_UNLESS(Common::IsAligned(size, PageSize), ResultInvalidSize)`, also when the
        // range wraps or starts above the address space as well.
        (DATA, PAGE_SIZE + 1, KernelError::InvalidSize),
        (DATA, PAGE_SIZE - 1, KernelError::InvalidSize),
        (
            u64::MAX - PAGE_SIZE + 1,
            PAGE_SIZE + 1,
            KernelError::InvalidSize,
        ),
        (end, PAGE_SIZE + 1, KernelError::InvalidSize),
        // `R_UNLESS(size > 0, ResultInvalidSize)`, also above the address space.
        (end, 0, KernelError::InvalidSize),
        // `R_UNLESS(size > 0, ResultInvalidSize)`.
        (DATA, 0, KernelError::InvalidSize),
        // `R_UNLESS((address < address + size), ResultInvalidCurrentMemory)`: the sum wraps,
        // to zero and past it.
        (
            u64::MAX - PAGE_SIZE + 1,
            PAGE_SIZE,
            KernelError::InvalidCurrentMemory,
        ),
        (
            u64::MAX - PAGE_SIZE + 1,
            2 * PAGE_SIZE,
            KernelError::InvalidCurrentMemory,
        ),
        (
            DATA,
            u64::MAX - PAGE_SIZE + 1,
            KernelError::InvalidCurrentMemory,
        ),
        // `R_UNLESS(page_table.Contains(address, size), ResultInvalidCurrentMemory)`: starting
        // at the end of the address space, and ending one page above it.
        (end, PAGE_SIZE, KernelError::InvalidCurrentMemory),
        (
            end - PAGE_SIZE,
            2 * PAGE_SIZE,
            KernelError::InvalidCurrentMemory,
        ),
    ];
    for (address, size, error) in cases {
        assert_eq!(
            set_memory_permission(&mut space, address, size, PERM_READ),
            kernel(error),
            "{address:#x}+{size:#x}"
        );
        assert_eq!(
            set_memory_attribute(
                &mut space,
                address,
                size,
                ATTR_PERMISSION_LOCKED,
                ATTR_PERMISSION_LOCKED
            ),
            kernel(error),
            "{address:#x}+{size:#x}"
        );
    }
    assert_eq!(log.take(), []);
    // The last page of the address space is in range: the call gets as far as the page
    // table, which refuses free memory.
    assert_eq!(
        set_memory_permission(&mut space, end - PAGE_SIZE, PAGE_SIZE, PERM_READ),
        kernel(KernelError::InvalidCurrentMemory)
    );
}

// ----- SetMemoryPermission --------------------------------------------------------------------

#[test]
fn set_memory_permission_takes_none_read_and_read_write() {
    let (mut space, log, _) = loaded();
    for (raw, permission, perms) in [
        (PERM_READ, MemoryPermission::USER_READ, Perms::Read),
        (
            PERM_NONE,
            MemoryPermission::KERNEL_READ | MemoryPermission::NOT_MAPPED,
            Perms::None,
        ),
        (
            PERM_READ_WRITE,
            MemoryPermission::USER_READ_WRITE,
            Perms::ReadWrite,
        ),
    ] {
        assert_eq!(
            set_memory_permission(&mut space, DATA, DATA_PAGES * PAGE_SIZE, raw),
            Ok(())
        );
        let block = space.query_info(DATA).unwrap();
        assert_eq!(
            (block.state(), block.permission()),
            (MemoryState::CodeData, permission)
        );
        assert_eq!(
            log.take(),
            [MapperCall::Protect {
                address: DATA,
                size: DATA_PAGES * PAGE_SIZE,
                perms
            }]
        );
    }
}

#[test]
fn any_other_permission_is_an_invalid_new_memory_permission() {
    let (mut space, log, _) = loaded();
    // `IsValidSetMemoryPermission`: read-execute, write alone, execute alone, write-execute,
    // all three, `DontCare` (1 << 28), and bits a permission does not have.
    for raw in [PERM_READ_EXECUTE, 2, 4, 6, 7, 1 << 28, 8, u32::MAX] {
        assert_eq!(
            set_memory_permission(&mut space, DATA, PAGE_SIZE, raw),
            kernel(KernelError::InvalidNewMemoryPermission),
            "{raw:#x}"
        );
    }
    // The permission is checked before the range is looked at: free memory, and a range
    // above the address space, give the same result.
    let end = space.regions().address_space.end;
    for address in [CODE - PAGE_SIZE, end] {
        assert_eq!(
            set_memory_permission(&mut space, address, PAGE_SIZE, PERM_READ_EXECUTE),
            kernel(KernelError::InvalidNewMemoryPermission),
            "{address:#x}"
        );
    }
    // And after the address, the size and the wrap.
    assert_eq!(
        set_memory_permission(&mut space, DATA + 1, PAGE_SIZE, PERM_READ_EXECUTE),
        kernel(KernelError::InvalidAddress)
    );
    assert_eq!(
        set_memory_permission(&mut space, u64::MAX - PAGE_SIZE + 1, PAGE_SIZE, 7),
        kernel(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(log.take(), []);
}

#[test]
fn set_memory_permission_on_memory_that_cannot_be_reprotected_is_invalid_current_memory() {
    let (mut space, log, stack_top) = loaded();
    // `KPageTableBase::SetMemoryPermission`: text, the stack, free memory.
    for address in [CODE, stack_top - PAGE_SIZE, CODE - PAGE_SIZE] {
        assert_eq!(
            set_memory_permission(&mut space, address, PAGE_SIZE, PERM_READ),
            kernel(KernelError::InvalidCurrentMemory),
            "{address:#x}"
        );
    }
    assert_eq!(log.take(), []);
}

// ----- SetMemoryAttribute ---------------------------------------------------------------------

#[test]
fn set_memory_attribute_locks_a_data_pages_permission() {
    let (mut space, log, _) = loaded();
    // The fixture's first two calls.
    assert_eq!(
        set_memory_permission(&mut space, DATA, PAGE_SIZE, PERM_READ),
        Ok(())
    );
    assert_eq!(
        set_memory_attribute(
            &mut space,
            DATA,
            PAGE_SIZE,
            ATTR_PERMISSION_LOCKED,
            ATTR_PERMISSION_LOCKED
        ),
        Ok(())
    );
    let block = space.query_info(DATA).unwrap();
    assert_eq!(block.svc_info().attribute, ATTR_PERMISSION_LOCKED);
    assert_eq!(block.svc_info().permission, PERM_READ);
    assert_eq!(block.pages(), 1);
    // Only the reprotection reached the translation.
    assert_eq!(log.take().len(), 1);
    // The lock holds.
    assert_eq!(
        set_memory_permission(&mut space, DATA, PAGE_SIZE, PERM_READ_WRITE),
        kernel(KernelError::InvalidCurrentMemory)
    );
}

#[test]
fn a_mask_and_attribute_that_do_not_go_together_are_an_invalid_combination() {
    let (mut space, log, _) = loaded();
    let before = space.query_info(DATA).unwrap();
    let cases = [
        // `R_UNLESS((mask | attr) == mask, ResultInvalidCombination)`: an attribute outside
        // the mask.
        (0, ATTR_UNCACHED),
        (ATTR_UNCACHED, ATTR_PERMISSION_LOCKED | ATTR_UNCACHED),
        // `R_UNLESS((mask | attr | SupportedMask) == SupportedMask, ..)`: `Locked` (bit 0),
        // `IpcLocked` (1), `DeviceShared` (2), the Switch 2 bit (5), and a bit above the
        // eight a block keeps.
        (1, 1),
        (1 << 1, 0),
        (1 << 2, 1 << 2),
        (1 << 5, 1 << 5),
        (ATTR_UNCACHED | 1 << 8, ATTR_UNCACHED),
        (u32::MAX, u32::MAX),
        // "Check that permission locked is either being set or not masked": masked and
        // cleared.
        (ATTR_PERMISSION_LOCKED, 0),
        (ATTR_PERMISSION_LOCKED | ATTR_UNCACHED, ATTR_UNCACHED),
    ];
    for (mask, attribute) in cases {
        assert_eq!(
            set_memory_attribute(&mut space, DATA, PAGE_SIZE, mask, attribute),
            kernel(KernelError::InvalidCombination),
            "mask {mask:#x} attribute {attribute:#x}"
        );
        // The combination is checked before the range is looked at (free memory, and a
        // range above the address space), and after the address.
        for address in [CODE - PAGE_SIZE, space.regions().address_space.end] {
            assert_eq!(
                set_memory_attribute(&mut space, address, PAGE_SIZE, mask, attribute),
                kernel(KernelError::InvalidCombination),
                "{address:#x}"
            );
        }
        assert_eq!(
            set_memory_attribute(&mut space, DATA + 1, PAGE_SIZE, mask, attribute),
            kernel(KernelError::InvalidAddress)
        );
    }
    assert_eq!(space.query_info(DATA).unwrap(), before);
    assert_eq!(log.take(), []);
    // An empty mask changes nothing and is allowed, as in Eden.
    assert_eq!(
        set_memory_attribute(&mut space, DATA, PAGE_SIZE, 0, 0),
        Ok(())
    );
    // `Uncached` can be set and cleared; setting both at once is allowed on data.
    for (mask, attribute) in [
        (ATTR_UNCACHED, ATTR_UNCACHED),
        (ATTR_UNCACHED, 0),
        (
            ATTR_UNCACHED | ATTR_PERMISSION_LOCKED,
            ATTR_UNCACHED | ATTR_PERMISSION_LOCKED,
        ),
    ] {
        assert_eq!(
            set_memory_attribute(&mut space, DATA, PAGE_SIZE, mask, attribute),
            Ok(()),
            "mask {mask:#x} attribute {attribute:#x}"
        );
        assert_eq!(
            space.query_info(DATA).unwrap().svc_info().attribute,
            attribute
        );
    }
}

#[test]
fn set_memory_attribute_on_memory_whose_state_forbids_it_is_invalid_current_memory() {
    let (mut space, _log, stack_top) = loaded();
    // `KPageTableBase::SetMemoryAttribute`: text and the stack can neither be locked nor
    // made uncached; free memory neither.
    for address in [CODE, stack_top - PAGE_SIZE, CODE - PAGE_SIZE] {
        for attribute in [ATTR_PERMISSION_LOCKED, ATTR_UNCACHED] {
            assert_eq!(
                set_memory_attribute(&mut space, address, PAGE_SIZE, attribute, attribute),
                kernel(KernelError::InvalidCurrentMemory),
                "{address:#x} {attribute:#x}"
            );
        }
    }
}

// ----- QueryMemory ----------------------------------------------------------------------------

/// Where the tests' guest memory is, and how large: room for one result with space around it.
const OUT_BASE: u64 = 0x10_0000;
const OUT_SIZE: usize = 0x100;

/// The bytes of the result `query_memory` wrote at `at`.
fn written(memory: &FlatMemory, at: u64) -> [u8; SvcMemoryInfo::SIZE] {
    let mut bytes = [0; SvcMemoryInfo::SIZE];
    memory.read(at, &mut bytes).unwrap();
    bytes
}

#[test]
fn query_memory_writes_the_blocks_report_and_returns_no_page_flags() {
    let (space, _log, stack_top) = loaded();
    let memory = FlatMemory::new(OUT_BASE, OUT_SIZE);
    memory.write(OUT_BASE, &[0xee; OUT_SIZE]).unwrap();
    let out = OUT_BASE + 0x20;
    // The fixture's first query: an address on its stack.
    assert_eq!(
        query_memory(&space, &memory, out, stack_top - 0x38),
        Ok(PAGE_INFO_FLAGS)
    );
    let stack = space.query_info(stack_top - PAGE_SIZE).unwrap();
    assert_eq!(written(&memory, out), stack.svc_info().to_bytes());
    assert_eq!(stack.svc_info().state, MemoryState::Stack.svc());
    assert_eq!(stack.svc_info().permission, PERM_READ_WRITE);
    // Nothing around the result was touched.
    let mut around = [0; OUT_SIZE];
    memory.read(OUT_BASE, &mut around).unwrap();
    assert!(around[..0x20].iter().all(|byte| *byte == 0xee));
    assert!(
        around[0x20 + SvcMemoryInfo::SIZE..]
            .iter()
            .all(|byte| *byte == 0xee)
    );

    // Above the address space: one inaccessible block to the end of the 64-bit space.
    assert_eq!(query_memory(&space, &memory, out, u64::MAX), Ok(0));
    let end = space.regions().address_space.end;
    let fake = SvcMemoryInfo {
        base_address: end,
        size: end.wrapping_neg(),
        state: MemoryState::Inaccessible.svc(),
        attribute: 0,
        permission: 0,
        ipc_count: 0,
        device_count: 0,
    };
    assert_eq!(written(&memory, out), fake.to_bytes());
}

#[test]
fn query_memory_to_a_pointer_that_cannot_be_written_is_invalid_pointer() {
    let (space, _log, _) = loaded();
    let memory = FlatMemory::new(OUT_BASE, OUT_SIZE);
    let end = OUT_BASE + OUT_SIZE as u64;
    let fits = end - SvcMemoryInfo::SIZE as u64;
    // The result ends exactly at the end of the memory: written.
    assert_eq!(query_memory(&space, &memory, fits, CODE), Ok(0));
    // One byte further, below the memory, where the address wraps: nothing is written.
    let before = written(&memory, fits);
    for out in [fits + 1, OUT_BASE - 1, 0, u64::MAX] {
        assert_eq!(
            query_memory(&space, &memory, out, DATA),
            kernel(KernelError::InvalidPointer),
            "{out:#x}"
        );
    }
    assert_eq!(written(&memory, fits), before);
}

#[test]
fn the_fixtures_address_space_is_what_the_tests_assume() {
    // The tests above use the data segment as "memory that may be reprotected and locked".
    let (space, _log, _) = loaded();
    let data = space.query_info(DATA).unwrap();
    assert_eq!(data.state(), MemoryState::CodeData);
    assert_eq!(data.permission(), MemoryPermission::USER_READ_WRITE);
}
