//! `GetInfo`: the values a libnx program is told at start-up, against what the reference
//! emulator told the homebrew fixture, and Horizon's result for each request it refuses.

use super::*;
use crate::handle::ObjectRefs;
use crate::id::{ProcessId, SlabId};
use crate::memory::fixture::{CODE_SIZE, STACK_SIZE, loaded, memory};
use crate::memory::testing::RecordingMapper;
use crate::memory::{MemoryArrangement, ProcessMemory};
use crate::result::SvcError;
use crate::sched::{CoreId, Priority};

fn kernel(error: KernelError) -> CallResult<u64> {
    Err(SvcError::Kernel(error))
}

/// A handle table with nothing in it, and a scheduler with nothing to run.
fn tables() -> (HandleTable, Scheduler) {
    (HandleTable::new(8).unwrap(), Scheduler::default())
}

/// Asks the loaded fixture's address space for `id` with the current-process pseudo-handle.
fn ask(space: &AddressSpace, id: u32) -> CallResult<u64> {
    let (handles, scheduler) = tables();
    get_info(&handles, space, &scheduler, id, Handle::CURRENT_PROCESS, 0)
}

/// The numbers of the ids this slice answers, by Eden's `InfoType` (`svc_types.h`).
const ALIAS_REGION_ADDRESS: u32 = 2;
const ALIAS_REGION_SIZE: u32 = 3;
const HEAP_REGION_ADDRESS: u32 = 4;
const HEAP_REGION_SIZE: u32 = 5;
const TOTAL_MEMORY_SIZE: u32 = 6;
const USED_MEMORY_SIZE: u32 = 7;
const ASLR_REGION_ADDRESS: u32 = 12;
const ASLR_REGION_SIZE: u32 = 13;
const STACK_REGION_ADDRESS: u32 = 14;
const STACK_REGION_SIZE: u32 = 15;
const ALIAS_REGION_EXTRA_SIZE: u32 = 28;

/// Every id this slice answers.
const ANSWERED: [u32; 11] = [
    ALIAS_REGION_ADDRESS,
    ALIAS_REGION_SIZE,
    HEAP_REGION_ADDRESS,
    HEAP_REGION_SIZE,
    TOTAL_MEMORY_SIZE,
    USED_MEMORY_SIZE,
    ASLR_REGION_ADDRESS,
    ASLR_REGION_SIZE,
    STACK_REGION_ADDRESS,
    STACK_REGION_SIZE,
    ALIAS_REGION_EXTRA_SIZE,
];

/// What Eden-NX2 @ 50fa86bb answered the homebrew fixture, in the order the fixture asked
/// (`tests/homebrew/hello/reference/startup.trace`): the sizes as it reported them, which do
/// not depend on where it put things. The addresses it reported are checked against the layout
/// in `memory/layout.rs`.
#[test]
fn the_fixture_is_told_the_sizes_the_reference_told_it() {
    let (space, _log, _) = loaded();
    let reported = [
        (ALIAS_REGION_SIZE, 0x10_0000_0000),
        (ALIAS_REGION_EXTRA_SIZE, 0),
        (HEAP_REGION_SIZE, 0x2_0000_0000),
        (ASLR_REGION_ADDRESS, 0x800_0000),
        (ASLR_REGION_SIZE, 0x7f_f800_0000),
        (STACK_REGION_SIZE, 0x8000_0000),
        (TOTAL_MEMORY_SIZE, 0xcd50_0000),
        (USED_MEMORY_SIZE, 0x10_7000),
    ];
    for (id, value) in reported {
        assert_eq!(ask(&space, id), Ok(value), "id {id}");
    }
    // From the last two libnx computes the heap it asks for
    // (`libnx/nx/source/runtime/init.c` @ b156b477, `__libnx_initheap`): total minus used
    // minus 2 MiB, rounded down to 2 MiB. The reference's trace has that size.
    let (total, used) = (0xcd50_0000_u64, 0x10_7000_u64);
    assert_eq!((total - used - 0x20_0000) & !0x1f_ffff, 0xcd00_0000);
}

#[test]
fn each_region_id_reads_its_region() {
    let (space, _log, _) = loaded();
    let regions = *space.regions();
    let expected = [
        (ALIAS_REGION_ADDRESS, regions.alias.start),
        (ALIAS_REGION_SIZE, regions.alias.size()),
        (HEAP_REGION_ADDRESS, regions.heap.start),
        (HEAP_REGION_SIZE, regions.heap.size()),
        (ASLR_REGION_ADDRESS, regions.code.start),
        (ASLR_REGION_SIZE, regions.code.size()),
        (STACK_REGION_ADDRESS, regions.stack.start),
        (STACK_REGION_SIZE, regions.stack.size()),
    ];
    // The eight values are all different, so a swapped pair shows.
    let mut values: Vec<_> = expected.iter().map(|(_, value)| *value).collect();
    values.sort_unstable();
    values.dedup();
    assert_eq!(values.len(), expected.len());
    for (id, value) in expected {
        assert_eq!(ask(&space, id), Ok(value), "id {id}");
    }
}

#[test]
fn total_and_used_memory_follow_the_heap() {
    let (mut space, _log, _) = loaded();
    let fixed = CODE_SIZE + STACK_SIZE;
    assert_eq!(ask(&space, USED_MEMORY_SIZE), Ok(fixed));
    space.set_heap_size(0x40_0000).unwrap();
    // The heap is used memory; the total does not move (used plus free is the pool).
    assert_eq!(ask(&space, USED_MEMORY_SIZE), Ok(fixed + 0x40_0000));
    assert_eq!(
        ask(&space, TOTAL_MEMORY_SIZE),
        Ok(MemoryArrangement::FourGb.application_pool_size())
    );
}

#[test]
fn the_extra_alias_size_is_reported_for_a_process_that_has_it() {
    let (mapper, _log) = RecordingMapper::new();
    let extra = ProcessMemory {
        alias_region_extra: true,
        ..memory(MemoryArrangement::FourGb)
    };
    let space = AddressSpace::new(&extra, Box::new(mapper)).unwrap();
    let size = space.regions().alias_region_extra_size;
    assert_ne!(size, 0);
    assert_eq!(ask(&space, ALIAS_REGION_EXTRA_SIZE), Ok(size));
    // libnx subtracts it from the alias region's end (`virtmemSetup`): the size reported for
    // the region includes it.
    assert_eq!(
        ask(&space, ALIAS_REGION_SIZE),
        Ok(space.regions().alias.size())
    );
    assert!(space.regions().alias.size() > size);
}

#[test]
fn a_sub_id_other_than_zero_is_an_invalid_enum_value() {
    let (space, _log, _) = loaded();
    let (handles, scheduler) = tables();
    // `R_UNLESS(info_sub_id == 0, ResultInvalidEnumValue)`, before the handle is looked at.
    for id in ANSWERED {
        for sub_id in [1, u64::MAX] {
            for handle in [Handle::CURRENT_PROCESS, Handle::INVALID] {
                assert_eq!(
                    get_info(&handles, &space, &scheduler, id, handle, sub_id),
                    kernel(KernelError::InvalidEnumValue),
                    "id {id} sub id {sub_id:#x}"
                );
            }
        }
    }
}

#[test]
fn a_handle_must_name_a_process() {
    let (space, _log, _) = loaded();
    let (mut handles, mut scheduler) = tables();
    // A handle to a thread, and one to a process object.
    let thread = scheduler
        .create_thread(Priority::new(44).unwrap(), CoreId::new(0).unwrap())
        .unwrap();
    let mut add = |object: Object| {
        let id = scheduler.objects_mut().create(object).unwrap();
        let handle = handles.add(scheduler.objects_mut(), id).unwrap();
        scheduler.objects_mut().close(id).unwrap();
        handle
    };
    let thread_handle = add(Object::Thread(thread));
    let process_handle = add(Object::Process(ProcessId::from_parts(0, 1)));

    for id in ANSWERED {
        let ask = |handle| get_info(&handles, &space, &scheduler, id, handle, 0);
        // `R_UNLESS(process.IsNotNull(), ResultInvalidHandle)`: the invalid handle, the
        // current-thread pseudo-handle, a handle no entry has, and a handle to a thread.
        for handle in [
            Handle::INVALID,
            Handle::CURRENT_THREAD,
            Handle::from_raw(0x0001_0001),
            thread_handle,
        ] {
            assert_eq!(
                ask(handle),
                kernel(KernelError::InvalidHandle),
                "id {id} handle {:#x}",
                handle.raw()
            );
        }
        // The pseudo-handle and a real handle to a process name the same, only, process.
        assert_eq!(ask(process_handle), ask(Handle::CURRENT_PROCESS));
        assert!(ask(process_handle).is_ok());
    }
}

#[test]
fn an_id_this_slice_does_not_answer_is_not_implemented_whatever_else_is_passed() {
    let (space, _log, _) = loaded();
    let (handles, scheduler) = tables();
    let unanswered: Vec<u32> = InfoType::ALL
        .iter()
        .map(|id| id.number())
        .filter(|number| !ANSWERED.contains(number))
        .collect();
    // Every id Eden answers besides the eleven, and the five Switch 2 ids.
    assert_eq!(unanswered.len(), InfoType::ALL.len() - ANSWERED.len());
    assert_eq!(
        unanswered,
        [
            0, 1, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 29, 30, 31, 32, 33, 37, 38,
            65000, 65001
        ]
    );
    for id in unanswered {
        for (handle, sub_id) in [
            (Handle::CURRENT_PROCESS, 0),
            (Handle::INVALID, 0),
            (Handle::CURRENT_PROCESS, 7),
        ] {
            assert_eq!(
                get_info(&handles, &space, &scheduler, id, handle, sub_id),
                kernel(KernelError::NotImplemented),
                "id {id}"
            );
        }
    }
}

#[test]
fn an_id_eden_does_not_list_is_an_invalid_enum_value() {
    let (space, _log, _) = loaded();
    // `IsSvcPermitted` (26), `IoRegionHint` (27) and `TransferMemoryHint` (34) are in the enum
    // and not in `GetInfo`'s switch; the rest are not ids at all. Eden's `default`.
    for id in [26, 27, 34, 35, 36, 39, 64999, 65002, u32::MAX] {
        assert_eq!(
            ask(&space, id),
            kernel(KernelError::InvalidEnumValue),
            "id {id}"
        );
        assert_eq!(InfoType::from_raw(id), None);
    }
}

#[test]
fn every_listed_id_has_its_own_number_in_order() {
    let numbers: Vec<_> = InfoType::ALL.iter().map(|id| id.number()).collect();
    assert!(numbers.is_sorted_by(|a, b| a < b), "{numbers:?}");
    for &id in InfoType::ALL {
        assert_eq!(InfoType::from_raw(id.number()), Some(id));
    }
}
