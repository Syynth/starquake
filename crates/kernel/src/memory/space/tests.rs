//! The address space's operations: each call's effect on the blocks, the translation and the
//! memory count, Horizon's result for each request it refuses, and a property test that the
//! three stay in step under random operations.
//!
//! "Eden" below is `eden/src/core/hle/kernel/k_page_table_base.cpp` @ cf654052 unless a test
//! names another file.

use std::collections::BTreeMap;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_memory::Perms;

use super::*;
use crate::memory::fixture::{
    CODE, CODE_PAGES, CODE_SIZE, DATA, DATA_PAGES, RODATA, RODATA_PAGES, STACK_SIZE, TEXT_PAGES,
    created, loaded, memory,
};
use crate::memory::testing::{MapperCall, MapperLog, RecordingMapper};

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
const PROPTEST_SEED: u64 = 0x5350_4143_455F_3031;

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 256,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(PROPTEST_SEED),
        failure_persistence: None,
        ..Config::default()
    }
}

const INVALID: Result<(), SvcError> = Err(SvcError::Kernel(KernelError::InvalidCurrentMemory));

/// What the tests read of a block.
type Row = (u64, u64, MemoryState, MemoryPermission, MemoryAttribute);

/// The blocks that are not free.
fn mapped(space: &AddressSpace) -> Vec<Row> {
    space
        .blocks()
        .filter(|block| block.state() != MemoryState::Free)
        .map(|block| {
            (
                block.address(),
                block.pages(),
                block.state(),
                block.permission(),
                block.attribute(),
            )
        })
        .collect()
}

fn row(address: u64, pages: u64, state: MemoryState, permission: MemoryPermission) -> Row {
    (address, pages, state, permission, MemoryAttribute::NONE)
}

/// Everything about an address space that an operation could change, for "nothing changed".
fn snapshot(space: &AddressSpace) -> (Vec<MemoryBlock>, MemoryLimit, u64, u64) {
    (
        space.blocks().copied().collect(),
        *space.limit(),
        space.heap_size(),
        space.max_heap_size(),
    )
}

/// The rows of a loaded fixture's code and stack.
fn loaded_rows(space: &AddressSpace) -> Vec<Row> {
    let stack = space.regions().stack.start + GUARD_PAGES * PAGE_SIZE;
    vec![
        row(
            CODE,
            TEXT_PAGES,
            MemoryState::Code,
            MemoryPermission::USER_READ_EXECUTE,
        ),
        row(
            RODATA,
            RODATA_PAGES,
            MemoryState::Code,
            MemoryPermission::USER_READ,
        ),
        row(
            DATA,
            DATA_PAGES,
            MemoryState::CodeData,
            MemoryPermission::USER_READ_WRITE,
        ),
        row(
            stack,
            STACK_SIZE / PAGE_SIZE,
            MemoryState::Stack,
            MemoryPermission::USER_READ_WRITE,
        ),
    ]
}

// ----- Creation -------------------------------------------------------------------------------

#[test]
fn a_new_address_space_holds_its_code_with_no_guest_access_and_has_reserved_it() {
    let (space, log) = created();
    // `KProcess::Initialize`: the code is mapped as `Code` with `KernelRead | NotMapped`.
    let code = row(
        CODE,
        CODE_PAGES,
        MemoryState::Code,
        MemoryPermission::KERNEL_READ | MemoryPermission::NOT_MAPPED,
    );
    assert_eq!(mapped(&space), [code]);
    assert_eq!(
        log.take(),
        [MapperCall::Map {
            address: CODE,
            size: CODE_SIZE,
            perms: Perms::None
        }]
    );
    // The pool is the application pool of the arrangement, and the code is counted.
    let pool = MemoryArrangement::FourGb.application_pool_size();
    assert_eq!(
        (space.limit().limit(), space.limit().reserved()),
        (pool, CODE_SIZE)
    );
    assert_eq!(space.used_user_physical_memory_size(), CODE_SIZE);
    // No stack yet, so no heap may be set, and none exists.
    assert_eq!((space.heap_size(), space.max_heap_size()), (0, 0));
    assert_eq!(space.normal_memory_size(), 0);
    // The whole address space is tiled: free below and above the code.
    let blocks: Vec<_> = space.blocks().collect();
    assert_eq!(blocks.len(), 3);
    assert_eq!(blocks[0].address(), 0);
    assert_eq!(blocks[2].end(), Some(space.regions().address_space.end));
    assert!(format!("{space:?}").starts_with("AddressSpace { regions: "));
}

#[test]
fn code_that_is_not_pages_in_the_code_region_or_exceeds_the_pool_is_refused() {
    let new = |code_address: u64, code_size: u64| {
        let (mapper, _log) = RecordingMapper::new();
        let memory = ProcessMemory {
            code_address,
            code_size,
            ..memory(MemoryArrangement::FourGb)
        };
        AddressSpace::new(&memory, Box::new(mapper)).map(|_| ())
    };
    let bad =
        |address: u64, size: u64| Err(CreateError::Layout(LayoutError::BadCode { address, size }));
    assert_eq!(new(CODE + 1, PAGE_SIZE), bad(CODE + 1, PAGE_SIZE));
    assert_eq!(new(CODE, PAGE_SIZE + 1), bad(CODE, PAGE_SIZE + 1));
    assert_eq!(new(CODE, 0), bad(CODE, 0));
    assert_eq!(new(0, PAGE_SIZE), bad(0, PAGE_SIZE));
    // Exactly the pool fits; one page more does not (`KProcess::Initialize`,
    // `R_UNLESS(memory_reservation.Succeeded(), ResultLimitReached)`).
    let pool = MemoryArrangement::FourGb.application_pool_size();
    assert_eq!(new(CODE, pool), Ok(()));
    assert_eq!(
        new(CODE, pool + PAGE_SIZE),
        Err(CreateError::Code(SvcError::Kernel(
            KernelError::LimitReached
        )))
    );
    assert_eq!(
        CreateError::Code(SvcError::Kernel(KernelError::LimitReached)).to_string(),
        "the program's code could not be mapped: limit reached"
    );
}

#[test]
fn the_pool_and_the_regions_follow_the_arrangement() {
    let (mapper, _log) = RecordingMapper::new();
    let space = AddressSpace::new(&memory(MemoryArrangement::TwelveGb), Box::new(mapper)).unwrap();
    assert_eq!(
        space.limit().limit(),
        MemoryArrangement::TwelveGb.application_pool_size()
    );
    assert_eq!(
        space.regions().heap.size(),
        MemoryArrangement::TwelveGb.physical_memory_size()
    );
}

// ----- Loading: SetProcessMemoryPermission ----------------------------------------------------

#[test]
fn reprotecting_segments_gives_text_rodata_and_data_their_states_and_permissions() {
    let (mut space, log) = created();
    drop(log.take());
    // `KProcess::LoadModule` (`eden/src/core/hle/kernel/k_process.cpp`): text read-execute,
    // rodata read, data read-write.
    space
        .set_process_memory_permission(CODE, TEXT_PAGES * PAGE_SIZE, SvcPermission::ReadExecute)
        .unwrap();
    space
        .set_process_memory_permission(RODATA, RODATA_PAGES * PAGE_SIZE, SvcPermission::Read)
        .unwrap();
    space
        .set_process_memory_permission(DATA, DATA_PAGES * PAGE_SIZE, SvcPermission::ReadWrite)
        .unwrap();
    // Writable code becomes code data; the rest stays code.
    assert_eq!(mapped(&space), loaded_rows(&loaded().0)[..3]);
    let protect = |address: u64, pages: u64, perms| MapperCall::Protect {
        address,
        size: pages * PAGE_SIZE,
        perms,
    };
    assert_eq!(
        log.take(),
        [
            protect(CODE, TEXT_PAGES, Perms::ReadExecute),
            protect(RODATA, RODATA_PAGES, Perms::Read),
            protect(DATA, DATA_PAGES, Perms::ReadWrite),
        ]
    );
    // The same permission again is nothing to do (`R_SUCCEED_IF`): no call to the
    // translation.
    space
        .set_process_memory_permission(RODATA, RODATA_PAGES * PAGE_SIZE, SvcPermission::Read)
        .unwrap();
    assert_eq!(log.take(), []);
    // No memory was reserved by any of it.
    assert_eq!(space.limit().reserved(), CODE_SIZE);
}

#[test]
fn only_code_memory_with_one_state_and_permission_can_be_reprotected_as_code() {
    let (mut space, log, stack_top) = loaded();
    let before = snapshot(&space);
    let mut refuse = |address: u64, size: u64| {
        // `CheckMemoryState(.., FlagCode, FlagCode, .., KMemoryAttribute::All, None)`.
        assert_eq!(
            space.set_process_memory_permission(address, size, SvcPermission::Read),
            INVALID,
            "{address:#x}+{size:#x}"
        );
        assert_eq!(snapshot(&space), before);
    };
    // Code data (the data segment) no longer has `FlagCode`; nor do the stack and free memory.
    refuse(DATA, PAGE_SIZE);
    refuse(stack_top - PAGE_SIZE, PAGE_SIZE);
    refuse(CODE - PAGE_SIZE, PAGE_SIZE);
    // Text and rodata are both code, with different permissions: not one block's worth.
    refuse(RODATA - PAGE_SIZE, 2 * PAGE_SIZE);
    assert_eq!(log.take(), []);
}

#[test]
fn alias_code_made_writable_becomes_alias_code_data() {
    // `KPageTableBase::SetProcessMemoryPermission`: `if (old_state == KMemoryState::AliasCode)
    // new_state = KMemoryState::AliasCodeData` when the new permission is writable; any other
    // permission keeps the state.
    let (mut space, _log, _) = loaded();
    let alias = CODE + CODE_SIZE;
    space
        .map_pages(
            alias,
            2,
            MemoryState::AliasCode,
            MemoryPermission::KERNEL_READ | MemoryPermission::NOT_MAPPED,
        )
        .unwrap();
    let state_at = |space: &AddressSpace, address: u64| space.query_info(address).unwrap().state();
    // Not writable: still alias code.
    space
        .set_process_memory_permission(alias, PAGE_SIZE, SvcPermission::ReadExecute)
        .unwrap();
    assert_eq!(state_at(&space, alias), MemoryState::AliasCode);
    assert_eq!(
        space.query_info(alias).unwrap().permission(),
        MemoryPermission::USER_READ_EXECUTE
    );
    space
        .set_process_memory_permission(alias + PAGE_SIZE, PAGE_SIZE, SvcPermission::Read)
        .unwrap();
    assert_eq!(state_at(&space, alias + PAGE_SIZE), MemoryState::AliasCode);
    // Writable: alias code data.
    space
        .set_process_memory_permission(alias + PAGE_SIZE, PAGE_SIZE, SvcPermission::ReadWrite)
        .unwrap();
    let block = space.query_info(alias + PAGE_SIZE).unwrap();
    assert_eq!(
        (block.state(), block.permission()),
        (
            MemoryState::AliasCodeData,
            MemoryPermission::USER_READ_WRITE
        )
    );
    // The page below it is untouched.
    assert_eq!(state_at(&space, alias), MemoryState::AliasCode);
}

// ----- What the guest may write ---------------------------------------------------------------

#[test]
fn the_user_can_write_a_range_only_if_every_block_it_touches_is_user_writable() {
    let (mut space, _log, stack_top) = loaded();
    // The data's middle page read-only, its last page no access: three blocks.
    space
        .set_memory_permission(DATA + PAGE_SIZE, PAGE_SIZE, SvcPermission::Read)
        .unwrap();
    space
        .set_memory_permission(DATA + 2 * PAGE_SIZE, PAGE_SIZE, SvcPermission::None)
        .unwrap();
    let writable = |address: u64, size: u64| space.user_can_write(address, size);
    // The writable data page, whole, and ending exactly at its end.
    assert!(writable(DATA, PAGE_SIZE));
    assert!(writable(DATA + PAGE_SIZE - 1, 1));
    // One byte further reaches the read-only page.
    assert!(!writable(DATA, PAGE_SIZE + 1));
    assert!(!writable(DATA + PAGE_SIZE - 1, 2));
    // Read-only, no access, text, rodata, free memory.
    for address in [
        DATA + PAGE_SIZE,
        DATA + 2 * PAGE_SIZE,
        CODE,
        RODATA,
        CODE - PAGE_SIZE,
    ] {
        assert!(!writable(address, 1), "{address:#x}");
    }
    // The stack, up to its top exactly; one byte past it is free memory.
    assert!(writable(stack_top - STACK_SIZE, STACK_SIZE));
    assert!(!writable(stack_top - 1, 2));
    // Above the address space, at its last byte and past it, and a range that wraps.
    let end = space.regions().address_space.end;
    assert!(!writable(end, 1));
    assert!(!writable(end - 1, 2));
    assert!(!writable(u64::MAX, 1));
    assert!(!writable(DATA, u64::MAX));
    assert!(!writable(u64::MAX, 2));
    // Nothing to write is always allowed.
    assert!(writable(CODE, 0));
    assert!(writable(u64::MAX, 0));
}

// ----- The main thread's stack ----------------------------------------------------------------

#[test]
fn the_main_threads_stack_goes_four_guard_pages_into_the_stack_region() {
    let (space, _log, stack_top) = loaded();
    // `KProcess::Run`: `MapPages(&stack_bottom, .., KMemoryState::Stack, UserReadWrite)`, at
    // the first free place of the region with `GetNumGuardPages()` (4) pages before it. The
    // reference's stack pointer agrees: its stack region started at 0x1080e00000 and its
    // `GetInfo` buffers were just below 0x1080f04000, the region's start plus four pages
    // plus the 1 MiB stack.
    let bottom = space.regions().stack.start + GUARD_PAGES * PAGE_SIZE;
    assert_eq!(stack_top, bottom + STACK_SIZE);
    assert_eq!(mapped(&space), loaded_rows(&space));
    // The stack is counted, and the largest heap is what the heap region leaves after the
    // code and the stack (`SetMaxHeapSize(m_max_process_memory - (stack + code))`).
    assert_eq!(space.limit().reserved(), CODE_SIZE + STACK_SIZE);
    assert_eq!(
        space.max_heap_size(),
        space.regions().heap.size() - CODE_SIZE - STACK_SIZE
    );
    // What the reference told the fixture: used memory 0x107000, total memory 0xcd500000.
    assert_eq!(space.used_user_physical_memory_size(), 0x10_7000);
    assert_eq!(space.total_user_physical_memory_size(), 0xcd50_0000);
}

#[test]
fn a_stack_size_is_rounded_up_to_pages_and_the_translation_maps_it_read_write() {
    let (mut space, log) = created();
    drop(log.take());
    let top = space.create_main_thread_stack(PAGE_SIZE + 1).unwrap();
    let bottom = space.regions().stack.start + GUARD_PAGES * PAGE_SIZE;
    assert_eq!(top, bottom + 2 * PAGE_SIZE);
    assert_eq!(
        log.take(),
        [MapperCall::Map {
            address: bottom,
            size: 2 * PAGE_SIZE,
            perms: Perms::ReadWrite
        }]
    );
    assert_eq!(space.limit().reserved(), CODE_SIZE + 2 * PAGE_SIZE);
    // A second stack: Eden asserts there is none yet.
    assert_eq!(
        space.create_main_thread_stack(PAGE_SIZE),
        Err(SvcError::Internal(
            "the main thread's stack exists already".into()
        ))
    );
}

#[test]
fn a_stack_of_no_bytes_maps_nothing_and_still_sets_the_largest_heap() {
    let (mut space, log) = created();
    drop(log.take());
    assert_eq!(space.create_main_thread_stack(0), Ok(0));
    assert_eq!(log.take(), []);
    assert_eq!(space.limit().reserved(), CODE_SIZE);
    assert_eq!(
        space.max_heap_size(),
        space.regions().heap.size() - CODE_SIZE
    );
}

#[test]
fn a_stack_is_refused_at_each_of_its_three_limits_and_reserves_nothing() {
    let pool = MemoryArrangement::FourGb.application_pool_size();
    let attempt = |stack_size: u64| {
        let (mut space, log) = created();
        drop(log.take());
        let result = space.create_main_thread_stack(stack_size);
        if result.is_err() {
            assert_eq!(space.limit().reserved(), CODE_SIZE, "{stack_size:#x}");
            assert_eq!(mapped(&space).len(), 1);
            assert_eq!(log.take(), []);
            assert_eq!(space.max_heap_size(), 0);
        }
        (result, space.regions().stack)
    };
    let kernel = |error| Err(SvcError::Kernel(error));

    // 1. `R_UNLESS(stack_size + m_code_size <= m_max_process_memory, ResultOutOfMemory)`: the
    //    largest process memory is the heap region's size.
    let (_, stack_region) = attempt(PAGE_SIZE);
    let (space, _) = created();
    let largest = space.regions().heap.size() - CODE_SIZE;
    assert_eq!(
        attempt(largest + PAGE_SIZE).0,
        kernel(KernelError::OutOfMemory)
    );
    // Rounding up overflows, and so does adding the code.
    assert_eq!(attempt(u64::MAX).0, kernel(KernelError::OutOfMemory));
    assert_eq!(
        attempt(u64::MAX - CODE_SIZE - PAGE_SIZE + 2).0,
        kernel(KernelError::OutOfMemory)
    );
    // 2. Exactly the largest passes that test and fails the next: the pool has no room
    //    (`R_UNLESS(mem_reservation.Succeeded(), ResultLimitReached)`).
    assert_eq!(attempt(largest).0, kernel(KernelError::LimitReached));
    assert_eq!(
        attempt(pool - CODE_SIZE + PAGE_SIZE).0,
        kernel(KernelError::LimitReached)
    );
    // 3. The pool has room but the stack region does not: `MapPages` needs fewer pages than
    //    the region has (`R_UNLESS(num_pages < region_num_pages, ResultOutOfMemory)`) and
    //    room for the guard pages on both sides (`R_UNLESS(addr != 0, ResultOutOfMemory)`).
    let region = stack_region.size();
    assert!(region < pool - CODE_SIZE);
    assert_eq!(attempt(region).0, kernel(KernelError::OutOfMemory));
    let guards = 2 * GUARD_PAGES * PAGE_SIZE;
    assert_eq!(
        attempt(region - guards + PAGE_SIZE).0,
        kernel(KernelError::OutOfMemory)
    );
    // With exactly the guards to spare it fits, ending four pages below the region's end.
    assert_eq!(
        attempt(region - guards).0,
        Ok(stack_region.end - GUARD_PAGES * PAGE_SIZE)
    );
}

// ----- MapPages -------------------------------------------------------------------------------

#[test]
fn thread_local_pages_go_into_the_kernel_map_region_with_guards_between_them() {
    let (mut space, log, _) = loaded();
    // `KThreadLocalPage::Initialize` (`eden/src/core/hle/kernel/k_thread_local_page.cpp`):
    // one page of `KMemoryState::ThreadLocal`, `UserReadWrite`, in the region
    // `GetRegionAddress` gives that state: the kernel map region.
    let map = |space: &mut AddressSpace| {
        space.map_pages_in_region(
            1,
            MemoryState::ThreadLocal,
            MemoryPermission::USER_READ_WRITE,
        )
    };
    let start = space.regions().kernel_map.start;
    let first = start + GUARD_PAGES * PAGE_SIZE;
    assert_eq!(map(&mut space), Ok(first));
    // The next free place keeps four pages clear of the first page too.
    let second = first + (1 + GUARD_PAGES) * PAGE_SIZE;
    assert_eq!(map(&mut space), Ok(second));
    let page = |address| MapperCall::Map {
        address,
        size: PAGE_SIZE,
        perms: Perms::ReadWrite,
    };
    assert_eq!(log.take(), [page(first), page(second)]);
    // Mapping pages reserves nothing by itself: the caller does (`KProcess` for the stack and
    // the code; thread-local pages come from the system resource).
    assert_eq!(space.limit().reserved(), CODE_SIZE + STACK_SIZE);
    // A state with no region of its own.
    assert_eq!(
        space.map_pages_in_region(1, MemoryState::Inaccessible, MemoryPermission::NONE),
        Err(SvcError::Kernel(KernelError::InvalidCurrentMemory))
    );
}

#[test]
fn pages_map_only_where_their_state_may_go_and_only_onto_free_memory() {
    let (mut space, log, stack_top) = loaded();
    let before = snapshot(&space);
    let regions = *space.regions();
    let read_write = MemoryPermission::USER_READ_WRITE;
    let mut refuse = |address: u64, pages: u64, state| {
        assert_eq!(
            space.map_pages(address, pages, state, read_write),
            INVALID,
            "{state:?} at {address:#x}"
        );
        assert_eq!(snapshot(&space), before);
    };
    // `R_UNLESS(this->CanContain(address, size, state), ResultInvalidCurrentMemory)`: a stack
    // outside the stack region, heap outside the heap region, and code in the heap region
    // (which lies inside the code region, and `CanContain` excludes).
    refuse(regions.heap.start, 1, MemoryState::Stack);
    refuse(regions.stack.start, 1, MemoryState::Normal);
    refuse(regions.heap.start, 1, MemoryState::Code);
    refuse(regions.alias.start, 1, MemoryState::Code);
    // The last page of the stack region is inside; one page more leaves it.
    refuse(regions.stack.end - PAGE_SIZE, 2, MemoryState::Stack);
    // `CheckMemoryState(.., KMemoryState::All, KMemoryState::Free, ..)`: onto the stack, and
    // onto a range that starts free and runs into it.
    refuse(stack_top - PAGE_SIZE, 1, MemoryState::Stack);
    refuse(regions.stack.start, GUARD_PAGES + 1, MemoryState::Stack);
    // Not whole pages, no pages, a size that overflows.
    refuse(regions.stack.start + 1, 1, MemoryState::Stack);
    refuse(regions.stack.start, 0, MemoryState::Stack);
    refuse(regions.stack.start, u64::MAX, MemoryState::Stack);
    assert_eq!(log.take(), []);
    // Exactly the free pages below the stack map.
    assert_eq!(
        space.map_pages(
            regions.stack.start,
            GUARD_PAGES,
            MemoryState::Stack,
            read_write
        ),
        Ok(())
    );
}

#[test]
fn can_contain_keeps_other_states_out_of_the_heap_and_alias_regions() {
    let (space, _log) = created();
    let regions = *space.regions();
    let at = |address: u64, state| space.can_contain(address, PAGE_SIZE, state);
    // Free and kernel memory may be anywhere in the address space.
    for address in [0, regions.heap.start, regions.address_space.end - PAGE_SIZE] {
        assert!(at(address, MemoryState::Free), "{address:#x}");
        assert!(at(address, MemoryState::Kernel), "{address:#x}");
    }
    assert!(!at(regions.address_space.end, MemoryState::Free));
    // Heap in the heap region, IPC buffers in the alias region, each at both edges.
    assert!(at(regions.heap.start, MemoryState::Normal));
    assert!(at(regions.heap.end - PAGE_SIZE, MemoryState::Normal));
    assert!(!at(regions.heap.end, MemoryState::Normal));
    assert!(!at(regions.heap.start - PAGE_SIZE, MemoryState::Normal));
    assert!(at(regions.alias.start, MemoryState::Ipc));
    assert!(at(regions.alias.end - PAGE_SIZE, MemoryState::NonSecureIpc));
    assert!(!at(regions.alias.end, MemoryState::NonDeviceIpc));
    // Shared memory goes in the alias-code region, which contains the heap and alias regions
    // and excludes them.
    assert!(at(CODE, MemoryState::Shared));
    assert!(!at(regions.heap.start, MemoryState::Shared));
    assert!(!at(regions.alias.end - PAGE_SIZE, MemoryState::Shared));
    assert!(at(regions.heap.end, MemoryState::Shared));
    assert!(at(regions.kernel_map.start, MemoryState::ThreadLocal));
    assert!(!at(regions.stack.start, MemoryState::ThreadLocal));
    // No region, an empty range, a range that wraps.
    assert!(!at(CODE, MemoryState::Inaccessible));
    assert!(!space.can_contain(CODE, 0, MemoryState::Code));
    assert!(!space.can_contain(u64::MAX - PAGE_SIZE + 1, PAGE_SIZE, MemoryState::Free));
}

// ----- SetMemoryPermission --------------------------------------------------------------------

#[test]
fn set_memory_permission_reprotects_data_and_keeps_its_state() {
    let (mut space, log, _) = loaded();
    // What the fixture does first: its first data page becomes read-only.
    assert_eq!(
        space.set_memory_permission(DATA, PAGE_SIZE, SvcPermission::Read),
        Ok(())
    );
    assert_eq!(
        log.take(),
        [MapperCall::Protect {
            address: DATA,
            size: PAGE_SIZE,
            perms: Perms::Read
        }]
    );
    let mut expected = loaded_rows(&space);
    expected[2] = row(DATA, 1, MemoryState::CodeData, MemoryPermission::USER_READ);
    expected.insert(
        3,
        row(
            DATA + PAGE_SIZE,
            DATA_PAGES - 1,
            MemoryState::CodeData,
            MemoryPermission::USER_READ_WRITE,
        ),
    );
    assert_eq!(mapped(&space), expected);
    // The same permission again changes nothing and asks nothing (`R_SUCCEED_IF`).
    assert_eq!(
        space.set_memory_permission(DATA, PAGE_SIZE, SvcPermission::Read),
        Ok(())
    );
    assert_eq!(log.take(), []);
    // No access at all, and back to read-write, which merges the data segment again.
    assert_eq!(
        space.set_memory_permission(DATA, PAGE_SIZE, SvcPermission::None),
        Ok(())
    );
    assert_eq!(
        mapped(&space)[2],
        row(
            DATA,
            1,
            MemoryState::CodeData,
            MemoryPermission::KERNEL_READ | MemoryPermission::NOT_MAPPED
        )
    );
    assert_eq!(
        space.set_memory_permission(DATA, PAGE_SIZE, SvcPermission::ReadWrite),
        Ok(())
    );
    assert_eq!(mapped(&space), loaded_rows(&space));
    assert_eq!(
        log.take(),
        [
            MapperCall::Protect {
                address: DATA,
                size: PAGE_SIZE,
                perms: Perms::None
            },
            MapperCall::Protect {
                address: DATA,
                size: PAGE_SIZE,
                perms: Perms::ReadWrite
            }
        ]
    );
}

#[test]
fn set_memory_permission_refuses_memory_that_cannot_be_reprotected() {
    let (mut space, log, stack_top) = loaded();
    space.set_heap_size(4 * PAGE_SIZE).unwrap();
    let heap = space.regions().heap.start;
    space
        .set_memory_attribute(
            heap,
            PAGE_SIZE,
            MemoryAttribute::UNCACHED,
            MemoryAttribute::UNCACHED,
        )
        .unwrap();
    space
        .set_memory_permission(heap + 2 * PAGE_SIZE, PAGE_SIZE, SvcPermission::Read)
        .unwrap();
    drop(log.take());
    let before = snapshot(&space);
    let mut refuse = |address: u64, size: u64| {
        assert_eq!(
            space.set_memory_permission(address, size, SvcPermission::Read),
            INVALID,
            "{address:#x}+{size:#x}"
        );
        assert_eq!(snapshot(&space), before);
    };
    // `CheckMemoryState(.., FlagCanReprotect, FlagCanReprotect, ..)`: text and rodata (code),
    // the stack and free memory have no such flag.
    refuse(CODE, PAGE_SIZE);
    refuse(RODATA, PAGE_SIZE);
    refuse(stack_top - PAGE_SIZE, PAGE_SIZE);
    refuse(heap + 4 * PAGE_SIZE, PAGE_SIZE);
    // `.., KMemoryAttribute::All, KMemoryAttribute::None`: a page with an attribute.
    refuse(heap, PAGE_SIZE);
    // One permission throughout: a read-write page and the read-only page after it.
    refuse(heap + PAGE_SIZE, 2 * PAGE_SIZE);
    // The data segment and the free page after it: not one state.
    refuse(DATA, (DATA_PAGES + 1) * PAGE_SIZE);
    assert_eq!(log.take(), []);
    // The heap page between them can be reprotected.
    assert_eq!(
        space.set_memory_permission(heap + PAGE_SIZE, PAGE_SIZE, SvcPermission::Read),
        Ok(())
    );
}

// ----- SetMemoryAttribute ---------------------------------------------------------------------

#[test]
fn locking_a_data_pages_permission_sets_the_attribute_and_stops_reprotection() {
    let (mut space, log, _) = loaded();
    // What the fixture does second, after making the page read-only.
    space
        .set_memory_permission(DATA, PAGE_SIZE, SvcPermission::Read)
        .unwrap();
    drop(log.take());
    let lock = MemoryAttribute::PERMISSION_LOCKED;
    assert_eq!(
        space.set_memory_attribute(DATA, PAGE_SIZE, lock, lock),
        Ok(())
    );
    assert_eq!(
        mapped(&space)[2],
        (
            DATA,
            1,
            MemoryState::CodeData,
            MemoryPermission::USER_READ,
            lock
        )
    );
    assert_eq!(log.take(), []);
    // The permission can no longer change: `SetMemoryPermission` needs no attribute at all.
    assert_eq!(
        space.set_memory_permission(DATA, PAGE_SIZE, SvcPermission::ReadWrite),
        INVALID
    );
    // Locking it again is allowed (the attribute test ignores the settable bits).
    assert_eq!(
        space.set_memory_attribute(DATA, PAGE_SIZE, lock, lock),
        Ok(())
    );
}

#[test]
fn setting_uncached_changes_the_block_and_asks_nothing_of_the_translation() {
    let (mut space, log, _) = loaded();
    space.set_heap_size(3 * PAGE_SIZE).unwrap();
    drop(log.take());
    let heap = space.regions().heap.start;
    let uncached = MemoryAttribute::UNCACHED;
    assert_eq!(
        space.set_memory_attribute(heap + PAGE_SIZE, PAGE_SIZE, uncached, uncached),
        Ok(())
    );
    let heap_rows = |space: &AddressSpace| -> Vec<Row> {
        mapped(space)
            .into_iter()
            .filter(|row| row.2 == MemoryState::Normal)
            .collect()
    };
    let read_write = MemoryPermission::USER_READ_WRITE;
    assert_eq!(
        heap_rows(&space),
        [
            row(heap, 1, MemoryState::Normal, read_write),
            (
                heap + PAGE_SIZE,
                1,
                MemoryState::Normal,
                read_write,
                uncached
            ),
            row(heap + 2 * PAGE_SIZE, 1, MemoryState::Normal, read_write),
        ]
    );
    // Clearing it over the whole heap merges the three blocks again.
    assert_eq!(
        space.set_memory_attribute(heap, 3 * PAGE_SIZE, uncached, MemoryAttribute::NONE),
        Ok(())
    );
    assert_eq!(
        heap_rows(&space),
        [row(heap, 3, MemoryState::Normal, read_write)]
    );
    // Divergence from Eden, which re-applies the unchanged permission here: no call at all.
    assert_eq!(log.take(), []);
}

#[test]
fn an_attribute_changes_only_where_the_state_allows_that_attribute() {
    let (mut space, log, stack_top) = loaded();
    space.set_heap_size(2 * PAGE_SIZE).unwrap();
    drop(log.take());
    let heap = space.regions().heap.start;
    let before = snapshot(&space);
    let (uncached, lock) = (
        MemoryAttribute::UNCACHED,
        MemoryAttribute::PERMISSION_LOCKED,
    );
    let mut refuse = |address: u64, size: u64, mask| {
        assert_eq!(
            space.set_memory_attribute(address, size, mask, mask),
            INVALID,
            "{mask:?} at {address:#x}"
        );
        assert_eq!(snapshot(&space), before);
    };
    // `PermissionLocked` needs `FlagCanPermissionLock`: heap, text and the stack lack it.
    refuse(heap, PAGE_SIZE, lock);
    refuse(CODE, PAGE_SIZE, lock);
    refuse(stack_top - PAGE_SIZE, PAGE_SIZE, lock);
    // `Uncached` needs `FlagCanChangeAttribute`: text, the stack and free memory lack it.
    refuse(CODE, PAGE_SIZE, uncached);
    refuse(stack_top - PAGE_SIZE, PAGE_SIZE, uncached);
    refuse(heap + 2 * PAGE_SIZE, PAGE_SIZE, uncached);
    // Both at once need both flags: the heap has only one of them, the data segment both.
    refuse(heap, PAGE_SIZE, MemoryAttribute::SET_MASK);
    // One state throughout: the data segment and the free page after it.
    refuse(DATA, (DATA_PAGES + 1) * PAGE_SIZE, lock);
    assert_eq!(
        space.set_memory_attribute(
            DATA,
            PAGE_SIZE,
            MemoryAttribute::SET_MASK,
            MemoryAttribute::SET_MASK
        ),
        Ok(())
    );
    assert_eq!(log.take(), []);
}

#[test]
fn a_mask_or_attribute_the_system_call_would_have_refused_is_an_internal_failure() {
    let (mut space, _log, _) = loaded();
    let before = snapshot(&space);
    let cases = [
        // A bit outside the settable ones.
        (MemoryAttribute::LOCKED, MemoryAttribute::NONE, "0x0", "0x1"),
        (
            MemoryAttribute::IPC_LOCKED,
            MemoryAttribute::IPC_LOCKED,
            "0x2",
            "0x2",
        ),
        // An attribute outside its mask.
        (
            MemoryAttribute::UNCACHED,
            MemoryAttribute::PERMISSION_LOCKED,
            "0x10",
            "0x8",
        ),
    ];
    for (mask, attribute, attribute_text, mask_text) in cases {
        assert_eq!(
            space.set_memory_attribute(DATA, PAGE_SIZE, mask, attribute),
            Err(SvcError::Internal(format!(
                "attribute {attribute_text} under mask {mask_text} is not settable"
            )))
        );
        assert_eq!(snapshot(&space), before);
    }
}

// ----- SetHeapSize ----------------------------------------------------------------------------

#[test]
fn the_heap_grows_shrinks_and_goes_away_and_the_pool_follows() {
    let (mut space, log, _) = loaded();
    let heap = space.regions().heap.start;
    let fixed = CODE_SIZE + STACK_SIZE;
    let heap_row = |pages: u64| {
        row(
            heap,
            pages,
            MemoryState::Normal,
            MemoryPermission::USER_READ_WRITE,
        )
    };
    let heap_blocks = |space: &AddressSpace| -> Vec<Row> {
        mapped(space)
            .into_iter()
            .filter(|row| row.2 == MemoryState::Normal)
            .collect()
    };

    // Grow from nothing: mapped read-write at the heap region's start, which is the answer.
    assert_eq!(space.set_heap_size(3 * PAGE_SIZE), Ok(heap));
    assert_eq!(heap_blocks(&space), [heap_row(3)]);
    assert_eq!(space.limit().reserved(), fixed + 3 * PAGE_SIZE);
    assert_eq!(space.heap_size(), 3 * PAGE_SIZE);
    // Grow again: only the new part is mapped, and it joins the block.
    assert_eq!(space.set_heap_size(5 * PAGE_SIZE), Ok(heap));
    assert_eq!(heap_blocks(&space), [heap_row(5)]);
    // The same size: nothing to do.
    assert_eq!(space.set_heap_size(5 * PAGE_SIZE), Ok(heap));
    // Shrink: the end is unmapped and its memory given back.
    assert_eq!(space.set_heap_size(PAGE_SIZE), Ok(heap));
    assert_eq!(heap_blocks(&space), [heap_row(1)]);
    assert_eq!(space.limit().reserved(), fixed + PAGE_SIZE);
    assert_eq!(space.used_user_physical_memory_size(), fixed + PAGE_SIZE);
    // To nothing: the heap region is one free block again (the barrier at its head is
    // cleared, `size == 0 ? DisableMergeAttribute::Normal : None`).
    assert_eq!(space.set_heap_size(0), Ok(heap));
    assert_eq!(heap_blocks(&space), []);
    assert_eq!(mapped(&space), loaded_rows(&space));
    // Free, text, rodata, data, free, the stack, free.
    assert_eq!(space.blocks().count(), 7);
    assert_eq!(space.limit().reserved(), fixed);

    let map = |address: u64, pages: u64| MapperCall::Map {
        address,
        size: pages * PAGE_SIZE,
        perms: Perms::ReadWrite,
    };
    let unmap = |address: u64, pages: u64| MapperCall::Unmap {
        address,
        size: pages * PAGE_SIZE,
    };
    assert_eq!(
        log.take(),
        [
            map(heap, 3),
            map(heap + 3 * PAGE_SIZE, 2),
            unmap(heap + PAGE_SIZE, 4),
            unmap(heap, 1),
        ]
    );
}

#[test]
fn no_heap_can_be_set_before_the_main_threads_stack_exists() {
    let (mut space, _log) = created();
    // `m_max_heap_size` is zero until `KProcess::Run` sets it.
    assert_eq!(
        space.set_heap_size(PAGE_SIZE),
        Err(SvcError::Kernel(KernelError::OutOfMemory))
    );
    assert_eq!(space.set_heap_size(0), Ok(space.regions().heap.start));
}

#[test]
fn the_heap_is_bounded_by_the_largest_heap_and_then_by_the_pool_exactly() {
    let (mut space, log, _) = loaded();
    let before = snapshot(&space);
    let kernel = |error| Err(SvcError::Kernel(error));
    // `R_UNLESS(size <= m_max_heap_size, ResultOutOfMemory)`.
    let largest = space.max_heap_size();
    assert_eq!(
        space.set_heap_size(largest + PAGE_SIZE),
        kernel(KernelError::OutOfMemory)
    );
    assert_eq!(
        space.set_heap_size(u64::MAX),
        kernel(KernelError::OutOfMemory)
    );
    // Exactly the largest heap passes that test; the pool then has no room for it
    // (`R_UNLESS(memory_reservation.Succeeded(), ResultLimitReached)`).
    assert_eq!(
        space.set_heap_size(largest),
        kernel(KernelError::LimitReached)
    );
    let room = space.limit().free();
    assert_eq!(
        space.set_heap_size(room + PAGE_SIZE),
        kernel(KernelError::LimitReached)
    );
    assert_eq!(snapshot(&space), before);
    assert_eq!(log.take(), []);
    // Exactly what the pool has left fits, and uses all of it.
    assert_eq!(space.set_heap_size(room), Ok(space.regions().heap.start));
    assert_eq!(space.limit().free(), 0);
    assert_eq!(
        space.used_user_physical_memory_size(),
        space.limit().limit()
    );
    // The heap the fixture asks for is below that.
    assert!(0xcd00_0000 < room);
}

#[test]
fn a_heap_that_is_not_plain_read_write_cannot_shrink_past_the_odd_page() {
    let (mut space, log, _) = loaded();
    let heap = space.regions().heap.start;
    space.set_heap_size(4 * PAGE_SIZE).unwrap();
    space
        .set_memory_permission(heap + 2 * PAGE_SIZE, PAGE_SIZE, SvcPermission::Read)
        .unwrap();
    drop(log.take());
    let before = snapshot(&space);
    // `CheckMemoryState(.., KMemoryState::All, KMemoryState::Normal, KMemoryPermission::All,
    // KMemoryPermission::UserReadWrite, KMemoryAttribute::All, KMemoryAttribute::None)` over
    // what would be freed: pages 2 and 3, of which page 2 is read-only.
    assert_eq!(space.set_heap_size(2 * PAGE_SIZE).map(|_| ()), INVALID);
    assert_eq!(space.set_heap_size(0).map(|_| ()), INVALID);
    assert_eq!(snapshot(&space), before);
    assert_eq!(log.take(), []);
    // Freeing only the last page, which is read-write, is fine.
    assert_eq!(space.set_heap_size(3 * PAGE_SIZE), Ok(heap));
    // A size off a page is not a run of pages.
    assert_eq!(space.set_heap_size(PAGE_SIZE + 1).map(|_| ()), INVALID);
    assert_eq!(space.set_heap_size(3 * PAGE_SIZE + 1).map(|_| ()), INVALID);
    assert_eq!(
        space.limit().reserved(),
        CODE_SIZE + STACK_SIZE + 3 * PAGE_SIZE
    );
}

// ----- QueryInfo ------------------------------------------------------------------------------

#[test]
fn a_query_finds_the_block_at_any_address_and_one_inaccessible_block_above_the_space() {
    let (space, _log, stack_top) = loaded();
    let at = |address: u64| {
        let block = space.query_info(address).unwrap();
        (block.address(), block.size(), block.state())
    };
    // The two queries the fixture makes: an address on its stack, and the page after its
    // image.
    let stack_bottom = stack_top - STACK_SIZE;
    assert_eq!(
        at(stack_top - 0x38),
        (stack_bottom, STACK_SIZE, MemoryState::Stack)
    );
    let image_end = CODE + CODE_SIZE;
    assert_eq!(
        at(image_end),
        (image_end, stack_bottom - image_end, MemoryState::Free)
    );
    // Inside a block, at its first and last byte.
    assert_eq!(
        at(DATA + 1),
        (DATA, DATA_PAGES * PAGE_SIZE, MemoryState::CodeData)
    );
    assert_eq!(at(DATA - 1).0, RODATA);
    assert_eq!(at(0), (0, CODE, MemoryState::Free));
    // The last byte of the address space is in the free block that ends it; one above is
    // the fake block (`QueryInfo`, "If the address is invalid, create a fake block"): from
    // the end of the address space to the end of the 64-bit space, inaccessible.
    let end = space.regions().address_space.end;
    assert_eq!(at(end - 1).2, MemoryState::Free);
    for address in [end, end + 1, u64::MAX] {
        assert_eq!(
            at(address),
            (end, end.wrapping_neg(), MemoryState::Inaccessible),
            "{address:#x}"
        );
    }
    let fake = space.query_info(u64::MAX).unwrap();
    assert_eq!(fake.last(), u64::MAX);
    assert_eq!(fake.permission(), MemoryPermission::NONE);
    assert_eq!(fake.attribute(), MemoryAttribute::NONE);
}

// ----- Shared ---------------------------------------------------------------------------------

#[test]
fn a_range_that_is_not_whole_pages_inside_the_address_space_is_invalid_current_memory() {
    let (mut space, log, _) = loaded();
    let before = snapshot(&space);
    let end = space.regions().address_space.end;
    let ranges = [
        // Off a page at either end, empty.
        (DATA + 1, PAGE_SIZE),
        (DATA, PAGE_SIZE + 1),
        (DATA, 0),
        // Ending one page above the address space, starting at its end, wrapping.
        (end - PAGE_SIZE, 2 * PAGE_SIZE),
        (end, PAGE_SIZE),
        (u64::MAX - PAGE_SIZE + 1, PAGE_SIZE),
        (DATA, u64::MAX - PAGE_SIZE + 1),
    ];
    for (address, size) in ranges {
        assert_eq!(
            space.set_memory_permission(address, size, SvcPermission::Read),
            INVALID,
            "{address:#x}+{size:#x}"
        );
        assert_eq!(
            space.set_process_memory_permission(address, size, SvcPermission::Read),
            INVALID
        );
        assert_eq!(
            space.set_memory_attribute(
                address,
                size,
                MemoryAttribute::UNCACHED,
                MemoryAttribute::UNCACHED
            ),
            INVALID
        );
    }
    assert_eq!(snapshot(&space), before);
    assert_eq!(log.take(), []);
}

#[test]
fn a_refusing_translation_fails_the_call_and_changes_nothing() {
    let (mut space, log, _) = loaded();
    space.set_heap_size(2 * PAGE_SIZE).unwrap();
    drop(log.take());
    let before = snapshot(&space);
    let heap = space.regions().heap.start;
    log.refuse(true);

    let internal = |result: CallResult<()>, what: &str| {
        let message = match result {
            Err(SvcError::Internal(message)) => message,
            other => format!("not an internal failure: {other:?}"),
        };
        assert!(message.starts_with(what), "{message}");
    };
    // Growing: the reservation is given back.
    internal(
        space.set_heap_size(4 * PAGE_SIZE).map(|_| ()),
        "the translation refused to map 0x2000 bytes at",
    );
    // Shrinking, reprotecting data and code, and mapping a page.
    internal(
        space.set_heap_size(0).map(|_| ()),
        "the translation refused to unmap 0x2000 bytes at",
    );
    internal(
        space.set_memory_permission(heap, PAGE_SIZE, SvcPermission::Read),
        "the translation refused to reprotect 0x1000 bytes at",
    );
    internal(
        space.set_process_memory_permission(CODE, PAGE_SIZE, SvcPermission::Read),
        "the translation refused to reprotect 0x1000 bytes at",
    );
    internal(
        space
            .map_pages_in_region(
                1,
                MemoryState::ThreadLocal,
                MemoryPermission::USER_READ_WRITE,
            )
            .map(|_| ()),
        "the translation refused to map 0x1000 bytes at",
    );
    assert_eq!(snapshot(&space), before);
    assert_eq!(log.take(), []);

    // A new address space whose code cannot be mapped is not created, and a stack that
    // cannot be mapped is not counted.
    let (mapper, refusing) = RecordingMapper::new();
    refusing.refuse(true);
    let result = AddressSpace::new(&memory(MemoryArrangement::FourGb), Box::new(mapper));
    assert!(matches!(
        result,
        Err(CreateError::Code(SvcError::Internal(_)))
    ));
    let (mut fresh, fresh_log) = created();
    fresh_log.refuse(true);
    internal(
        fresh.create_main_thread_stack(STACK_SIZE).map(|_| ()),
        "the translation refused to map 0x100000 bytes at",
    );
    assert_eq!(fresh.limit().reserved(), CODE_SIZE);
    // Once the translation works again, so does the call.
    log.refuse(false);
    assert_eq!(space.set_heap_size(4 * PAGE_SIZE), Ok(heap));
}

#[test]
fn a_permission_with_write_and_no_read_has_no_translation() {
    let (mut space, log, _) = loaded();
    let before = snapshot(&space);
    let address = space.regions().stack.start;
    assert_eq!(
        space.map_pages(address, 1, MemoryState::Stack, MemoryPermission::USER_WRITE),
        Err(SvcError::Internal(
            "permission 0x12 has no translation".into()
        ))
    );
    assert_eq!(snapshot(&space), before);
    assert_eq!(log.take(), []);
}

#[test]
fn total_memory_is_capped_by_the_largest_process_memory() {
    let (mut space, _log, _) = loaded();
    let pool = space.limit().limit();
    // With the pool smaller than the heap region (both arrangements), the total is the pool,
    // whatever is in use.
    assert_eq!(space.total_user_physical_memory_size(), pool);
    space.set_heap_size(0x20_0000).unwrap();
    assert_eq!(space.total_user_physical_memory_size(), pool);
    // `if (used_size + free_size > max_size) return max_size`: a process whose largest
    // memory is below the pool is told that, exactly from one byte below.
    space.max_process_memory = pool;
    assert_eq!(space.total_user_physical_memory_size(), pool);
    space.max_process_memory = pool - 1;
    assert_eq!(space.total_user_physical_memory_size(), pool - 1);
}

// ----- The three views stay in step ------------------------------------------------------------

/// One step of a generated sequence, in pages of heap.
#[derive(Debug, Clone, Copy)]
enum Op {
    Heap(u64),
    Permission {
        first: u64,
        pages: u64,
        permission: usize,
    },
    Attribute {
        first: u64,
        pages: u64,
        set: bool,
    },
}

/// The most heap pages a sequence asks for.
const HEAP_PAGES: u64 = 24;
const PERMISSIONS: [SvcPermission; 3] = [
    SvcPermission::None,
    SvcPermission::Read,
    SvcPermission::ReadWrite,
];

fn op() -> impl Strategy<Value = Op> {
    let range = (0..HEAP_PAGES).prop_flat_map(|first| (Just(first), 1..=HEAP_PAGES - first));
    prop_oneof![
        (0..=HEAP_PAGES).prop_map(Op::Heap),
        (range.clone(), 0..PERMISSIONS.len()).prop_map(|((first, pages), permission)| {
            Op::Permission {
                first,
                pages,
                permission,
            }
        }),
        (range, any::<bool>()).prop_map(|((first, pages), set)| Op::Attribute {
            first,
            pages,
            set
        }),
    ]
}

/// What the translation holds, rebuilt page by page from the calls it received.
#[derive(Default)]
struct Translated(BTreeMap<u64, Perms>);

/// The addresses of the pages of a range.
fn pages_of(address: u64, size: u64) -> impl Iterator<Item = u64> {
    (0..size / PAGE_SIZE).map(move |index| address + index * PAGE_SIZE)
}

impl Translated {
    fn replay(&mut self, log: &MapperLog) {
        for call in log.take() {
            match call {
                MapperCall::Map {
                    address,
                    size,
                    perms,
                } => {
                    for page in pages_of(address, size) {
                        assert_eq!(self.0.insert(page, perms), None, "mapped twice: {page:#x}");
                    }
                }
                MapperCall::Unmap { address, size } => {
                    for page in pages_of(address, size) {
                        assert!(self.0.remove(&page).is_some(), "not mapped: {page:#x}");
                    }
                }
                MapperCall::Protect {
                    address,
                    size,
                    perms,
                } => {
                    for page in pages_of(address, size) {
                        *self.0.get_mut(&page).expect("a reprotected page is mapped") = perms;
                    }
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(config())]

    /// Random heap sizes, reprotections and attribute changes, each of which may be refused.
    /// After every step:
    ///
    /// - the block map is whole (`CheckState`);
    /// - the translation, replayed page by page from the calls it received, maps exactly the
    ///   pages the block map says are not free, each with the guest's part of its block's
    ///   permission, and every call it received was for pages in the state the call needs
    ///   (never mapped twice, never unmapped or reprotected while unmapped);
    /// - the accounting adds up: what is reserved is the code, the stack and the heap; the
    ///   heap is the sum of the heap blocks; used plus free is the pool.
    #[test]
    fn blocks_translation_and_accounting_stay_in_step(ops in proptest::collection::vec(op(), 1..40)) {
        let (mapper, log) = RecordingMapper::new();
        let mut space =
            AddressSpace::new(&memory(MemoryArrangement::FourGb), Box::new(mapper)).unwrap();
        let mut translated = Translated::default();
        space.set_process_memory_permission(DATA, DATA_PAGES * PAGE_SIZE, SvcPermission::ReadWrite).unwrap();
        space.create_main_thread_stack(STACK_SIZE).unwrap();
        let heap = space.regions().heap.start;
        let page = |index: u64| heap + index * PAGE_SIZE;
        let lock = MemoryAttribute::UNCACHED;

        for op in ops {
            let before = snapshot(&space);
            let result = match op {
                Op::Heap(pages) => space.set_heap_size(pages * PAGE_SIZE).map(|_| ()),
                Op::Permission { first, pages, permission } => space.set_memory_permission(
                    page(first), pages * PAGE_SIZE, PERMISSIONS[permission],
                ),
                Op::Attribute { first, pages, set } => space.set_memory_attribute(
                    page(first), pages * PAGE_SIZE, lock,
                    if set { lock } else { MemoryAttribute::NONE },
                ),
            };
            // A refusal is always the guest's result for a range in the wrong state, and
            // changes nothing.
            if let Err(error) = &result {
                prop_assert_eq!(error, &SvcError::Kernel(KernelError::InvalidCurrentMemory), "{:?}", op);
                prop_assert_eq!(snapshot(&space), before.clone(), "{:?}", op);
            }
            translated.replay(&log);

            prop_assert!(space.blocks.check_state(), "after {:?}", op);
            let mut expected = BTreeMap::new();
            let mut heap_bytes = 0;
            for block in space.blocks().filter(|block| block.state() != MemoryState::Free) {
                if block.state() == MemoryState::Normal {
                    heap_bytes += block.size();
                }
                let perms = block.permission().guest().unwrap();
                for index in 0..block.pages() {
                    expected.insert(block.address() + index * PAGE_SIZE, perms);
                }
            }
            prop_assert_eq!(&translated.0, &expected, "after {:?}", op);
            prop_assert_eq!(heap_bytes, space.heap_size());
            prop_assert_eq!(space.normal_memory_size(), space.heap_size());
            let fixed = CODE_SIZE + STACK_SIZE;
            prop_assert_eq!(space.limit().reserved(), fixed + space.heap_size());
            prop_assert_eq!(space.used_user_physical_memory_size(), space.limit().reserved());
            prop_assert_eq!(
                space.used_user_physical_memory_size() + space.limit().free(),
                space.limit().limit()
            );
            prop_assert_eq!(space.total_user_physical_memory_size(), space.limit().limit());
        }
    }
}
