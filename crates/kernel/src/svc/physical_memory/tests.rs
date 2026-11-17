//! Both calls' argument checks, each at its exact edge and in Eden's order, and what a passing
//! call does.
//!
//! "Eden" below is `eden/src/core/hle/kernel/svc/svc_physical_memory.cpp` @ cf654052.

use starquake_memory::{PAGE_SIZE, Perms};

use super::*;
use crate::memory::fixture::loaded;
use crate::memory::testing::MapperCall;
use crate::memory::{MemoryAttribute, MemoryPermission, MemoryState};
use crate::result::SvcError;

/// A system resource of one page: the smallest a process can have (`GetInfo` 16 reports the
/// size in whole pages, [`crate::svc::info::secure_resource_size`]).
const RESOURCE: u64 = PAGE_SIZE;

fn kernel(error: KernelError) -> CallResult<()> {
    Err(SvcError::Kernel(error))
}

type Call = fn(&mut AddressSpace, u64, u64, u64) -> CallResult<()>;

/// Both calls, by name.
const CALLS: [(&str, Call); 2] = [
    ("MapPhysicalMemory", map_physical_memory),
    ("UnmapPhysicalMemory", unmap_physical_memory),
];

#[test]
fn map_then_unmap_maps_and_frees_the_range() {
    let (mut space, log, _) = loaded();
    let alias = space.regions().alias.start;
    let reserved = space.limit().reserved();
    assert_eq!(
        map_physical_memory(&mut space, RESOURCE, alias, 2 * PAGE_SIZE),
        Ok(())
    );
    assert_eq!(
        log.take(),
        [MapperCall::Map {
            address: alias,
            size: 2 * PAGE_SIZE,
            perms: Perms::ReadWrite
        }]
    );
    let block = space.query_info(alias).unwrap();
    assert_eq!(
        (
            block.address(),
            block.pages(),
            block.state(),
            block.permission(),
            block.attribute()
        ),
        (
            alias,
            2,
            MemoryState::Normal,
            MemoryPermission::USER_READ_WRITE,
            MemoryAttribute::NONE
        )
    );
    assert_eq!(space.limit().reserved(), reserved + 2 * PAGE_SIZE);
    assert_eq!(
        unmap_physical_memory(&mut space, RESOURCE, alias, 2 * PAGE_SIZE),
        Ok(())
    );
    assert_eq!(
        log.take(),
        [MapperCall::Unmap {
            address: alias,
            size: 2 * PAGE_SIZE
        }]
    );
    assert_eq!(space.query_info(alias).unwrap().state(), MemoryState::Free);
    assert_eq!(space.limit().reserved(), reserved);
}

#[test]
fn each_check_refuses_at_its_edge_and_passes_just_inside_it() {
    let (mut space, log, _) = loaded();
    let alias = space.regions().alias;
    let address_space = space.regions().address_space;
    let page = |index: u64| alias.start + index * PAGE_SIZE;
    // (address, size, system resource, result), for each check in Eden's order.
    let cases = [
        // `IsAligned(addr, YUZU_PAGESIZE)`: `ResultInvalidAddress`.
        (page(0) + 1, PAGE_SIZE, RESOURCE, Err(KernelError::InvalidAddress)),
        (page(1) - 1, PAGE_SIZE, RESOURCE, Err(KernelError::InvalidAddress)),
        // `IsAligned(size, YUZU_PAGESIZE)`: `ResultInvalidSize`.
        (page(0), PAGE_SIZE + 1, RESOURCE, Err(KernelError::InvalidSize)),
        (page(0), PAGE_SIZE - 1, RESOURCE, Err(KernelError::InvalidSize)),
        // `size == 0`: `ResultInvalidSize`.
        (page(0), 0, RESOURCE, Err(KernelError::InvalidSize)),
        // `addr < addr + size`: `ResultInvalidMemoryRegion`, also for a sum of exactly 2^64.
        (u64::MAX - PAGE_SIZE + 1, PAGE_SIZE, RESOURCE, Err(KernelError::InvalidMemoryRegion)),
        (u64::MAX - PAGE_SIZE + 1, 2 * PAGE_SIZE, RESOURCE, Err(KernelError::InvalidMemoryRegion)),
        // `GetTotalSystemResourceSize() == 0`: `ResultInvalidState`.
        (page(0), PAGE_SIZE, 0, Err(KernelError::InvalidState)),
        // `Contains`: `ResultInvalidMemoryRegion`; the last page of the address space is
        // inside it, the page past it is not.
        (address_space.end, PAGE_SIZE, RESOURCE, Err(KernelError::InvalidMemoryRegion)),
        (address_space.end - PAGE_SIZE, 2 * PAGE_SIZE, RESOURCE, Err(KernelError::InvalidMemoryRegion)),
        // `IsInAliasRegion`: `ResultInvalidMemoryRegion`, one page below the alias region or
        // one past its end.
        (alias.start - PAGE_SIZE, PAGE_SIZE, RESOURCE, Err(KernelError::InvalidMemoryRegion)),
        (alias.start - PAGE_SIZE, 2 * PAGE_SIZE, RESOURCE, Err(KernelError::InvalidMemoryRegion)),
        (alias.end, PAGE_SIZE, RESOURCE, Err(KernelError::InvalidMemoryRegion)),
        (alias.end - PAGE_SIZE, 2 * PAGE_SIZE, RESOURCE, Err(KernelError::InvalidMemoryRegion)),
        (address_space.end - PAGE_SIZE, PAGE_SIZE, RESOURCE, Err(KernelError::InvalidMemoryRegion)),
        // Just inside every edge.
        (page(0), PAGE_SIZE, RESOURCE, Ok(())),
        (alias.end - PAGE_SIZE, PAGE_SIZE, RESOURCE, Ok(())),
        (page(0), alias.size(), RESOURCE, Ok(())),
    ];
    for (name, call) in CALLS {
        for (address, size, resource, expected) in cases {
            // A whole alias region is more than the pool holds.
            let expected = if size == alias.size() && name == "MapPhysicalMemory" {
                Err(KernelError::LimitReached)
            } else {
                expected
            };
            assert_eq!(
                call(&mut space, resource, address, size),
                expected.map_err(SvcError::Kernel),
                "{name} {address:#x} {size:#x} {resource:#x}"
            );
        }
    }
    // The two that passed mapped a page each and unmapped it again.
    assert_eq!(space.mapped_physical_memory_size(), 0);
    assert_eq!(log.take().len(), 4);
}

#[test]
fn with_two_faults_the_one_checked_first_is_reported() {
    let (mut space, log, _) = loaded();
    let alias = space.regions().alias;
    let heap = space.regions().heap.start;
    // Each case holds two faults, and the result is the one checked first.
    let cases = [
        // An unaligned address and a size of zero: the address first.
        (alias.start + 1, 0, RESOURCE, KernelError::InvalidAddress),
        // An unaligned size and no system resource: the size first.
        (alias.start, 1, 0, KernelError::InvalidSize),
        // A range that wraps and no system resource: the wrap first.
        (u64::MAX - PAGE_SIZE + 1, PAGE_SIZE, 0, KernelError::InvalidMemoryRegion),
        // No system resource and a range outside the alias region: the resource first.
        (heap, PAGE_SIZE, 0, KernelError::InvalidState),
    ];
    for (name, call) in CALLS {
        for (address, size, resource, first) in cases {
            assert_eq!(
                call(&mut space, resource, address, size),
                kernel(first),
                "{name} {address:#x} {size:#x} {resource:#x}"
            );
        }
    }
    // The page-table half's own fault comes after every one of the system call's: a range of
    // IPC memory (refused by `UnmapPhysicalMemory`) with no system resource is `InvalidState`.
    space
        .map_pages(
            alias.start,
            1,
            MemoryState::Ipc,
            MemoryPermission::USER_READ_WRITE,
        )
        .unwrap();
    assert_eq!(
        unmap_physical_memory(&mut space, RESOURCE, alias.start, PAGE_SIZE),
        kernel(KernelError::InvalidCurrentMemory)
    );
    assert_eq!(
        unmap_physical_memory(&mut space, 0, alias.start, PAGE_SIZE),
        kernel(KernelError::InvalidState)
    );
    // And a map the pool has no room for, with no system resource, is `InvalidState` too.
    let room = space.limit().free() + PAGE_SIZE;
    let past = alias.start + PAGE_SIZE;
    assert_eq!(
        map_physical_memory(&mut space, RESOURCE, past, room),
        kernel(KernelError::LimitReached)
    );
    assert_eq!(
        map_physical_memory(&mut space, 0, past, room),
        kernel(KernelError::InvalidState)
    );
    drop(log.take());
    assert_eq!(space.mapped_physical_memory_size(), 0);
}
