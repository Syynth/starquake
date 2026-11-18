//! The resource limit's values, numbers and thread check, without the tables.

use super::*;
use crate::id::SlabId;

#[test]
fn the_numbers_and_limits_are_edens() {
    // `enum class LimitableResource` (`svc_types.h`) and `CreateResourceLimitForProcess`'s
    // `SetLimitValue` calls (`k_resource_limit.cpp` @ cf654052), written out.
    let numbers: Vec<u32> = LimitableResource::ALL
        .iter()
        .map(|resource| resource.number())
        .collect();
    assert_eq!(numbers, [0, 1, 2, 3, 4]);
    assert_eq!(LimitableResource::PhysicalMemory.number(), 0);
    assert_eq!(LimitableResource::ThreadCount.number(), 1);
    assert_eq!(LimitableResource::EventCount.number(), 2);
    assert_eq!(LimitableResource::TransferMemoryCount.number(), 3);
    assert_eq!(LimitableResource::SessionCount.number(), 4);
    assert_eq!(THREAD_COUNT_LIMIT, 800);
    assert_eq!(EVENT_COUNT_LIMIT, 900);
    assert_eq!(TRANSFER_MEMORY_COUNT_LIMIT, 200);
    assert_eq!(SESSION_COUNT_LIMIT, 1133);
    assert_eq!(UNCOUNTED_CURRENT_VALUE, 0);
}

#[test]
fn a_resource_number_is_valid_below_the_count_only() {
    for resource in LimitableResource::ALL {
        assert_eq!(
            LimitableResource::from_raw(resource.number()),
            Some(resource)
        );
    }
    // `IsValidResourceType`: `type < Count`, `Count` being one past the last.
    let count = u32::try_from(LimitableResource::ALL.len()).unwrap();
    assert_eq!(
        LimitableResource::from_raw(count - 1),
        Some(LimitableResource::SessionCount)
    );
    assert_eq!(LimitableResource::from_raw(count), None);
    assert_eq!(LimitableResource::from_raw(count + 1), None);
    assert_eq!(LimitableResource::from_raw(u32::MAX), None);
}

#[test]
fn each_value_is_read_from_what_the_owner_holds() {
    let mut memory = MemoryLimit::new(0x9000);
    memory.reserve(0x3000).unwrap();
    let usage = Usage {
        memory: &memory,
        threads: 7,
    };
    let values = |resource| Values::of(resource, usage);
    assert_eq!(
        values(LimitableResource::PhysicalMemory),
        Values {
            limit: 0x9000,
            current: 0x3000
        }
    );
    assert_eq!(
        values(LimitableResource::ThreadCount),
        Values {
            limit: THREAD_COUNT_LIMIT,
            current: 7
        }
    );
    assert_eq!(values(LimitableResource::PhysicalMemory).free(), 0x6000);
    assert_eq!(
        values(LimitableResource::ThreadCount).free(),
        THREAD_COUNT_LIMIT - 7
    );
}

#[test]
fn events_transfer_memories_and_sessions_are_not_counted() {
    let memory = MemoryLimit::new(0x9000);
    let usage = Usage {
        memory: &memory,
        threads: 3,
    };
    for (resource, limit) in [
        (LimitableResource::EventCount, EVENT_COUNT_LIMIT),
        (
            LimitableResource::TransferMemoryCount,
            TRANSFER_MEMORY_COUNT_LIMIT,
        ),
        (LimitableResource::SessionCount, SESSION_COUNT_LIMIT),
    ] {
        let values = Values::of(resource, usage);
        assert_eq!(
            values,
            Values {
                limit,
                current: UNCOUNTED_CURRENT_VALUE
            },
            "{resource:?}"
        );
        assert_eq!(values.free(), limit, "{resource:?}");
    }
}

#[test]
fn one_more_fits_below_the_limit_and_not_at_it() {
    let at = |current| Values {
        limit: THREAD_COUNT_LIMIT,
        current,
    };
    assert_eq!(at(0).reserve_one(), Ok(()));
    assert_eq!(at(THREAD_COUNT_LIMIT - 1).reserve_one(), Ok(()));
    assert_eq!(
        at(THREAD_COUNT_LIMIT).reserve_one(),
        Err(KernelError::LimitReached)
    );
    assert_eq!(
        at(THREAD_COUNT_LIMIT + 1).reserve_one(),
        Err(KernelError::LimitReached)
    );
    // One more would overflow: refused, whatever the limit.
    let full = Values {
        limit: u64::MAX,
        current: u64::MAX,
    };
    assert_eq!(full.reserve_one(), Err(KernelError::LimitReached));
    let below = Values {
        limit: u64::MAX,
        current: u64::MAX - 1,
    };
    assert_eq!(below.reserve_one(), Ok(()));
}

#[test]
fn the_free_value_stops_at_zero_past_the_limit() {
    let at = |current| Values { limit: 5, current }.free();
    assert_eq!((at(4), at(5), at(6), at(u64::MAX)), (1, 0, 0, 0));
}

#[test]
fn a_thread_count_at_the_top_of_its_range_is_refused() {
    let memory = MemoryLimit::new(0);
    let usage = Usage {
        memory: &memory,
        threads: usize::MAX,
    };
    let values = Values::of(LimitableResource::ThreadCount, usage);
    assert_eq!(values.reserve_one(), Err(KernelError::LimitReached));
    assert_eq!(values.free(), 0);
}

#[test]
fn a_limit_names_its_owner() {
    let owner = ProcessId::from_parts(3, 9);
    let limit = ResourceLimit::of(owner);
    assert_eq!(limit.owner(), owner);
    assert_ne!(limit, ResourceLimit::of(ProcessId::from_parts(3, 8)));
}
