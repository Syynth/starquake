//! The `MapPhysicalMemory` and `UnmapPhysicalMemory` arms: the registers they read and write,
//! the system resource size they take from the process facts, and what `GetInfo` reports of
//! the memory they map.

use super::*;
use crate::svc::info::InfoType;

/// One page: the smallest system resource a process can have.
const PAGE: u64 = starquake_memory::PAGE_SIZE;

/// Fresh tables for a process whose metadata asks for a system resource of `size` bytes.
fn tables_with_resource(size: u64) -> Tables {
    let mut description = ProcessDescription::homebrew(CONSOLE);
    description.handle_table_size = i32::from(MAX_TABLE_SIZE);
    description.system_resource_size = u32::try_from(size).unwrap();
    crate::process::create(&description, fixture::loaded().0).expect("the fixture's process")
}

/// The alias region's first address.
fn alias_start(dispatcher: &Dispatcher) -> u64 {
    dispatcher.with::<(&AddressSpace,), _>(|(space,)| space.regions().alias.start)
}

/// What `GetInfo(id, current process, 0)` answers; it must succeed.
fn info(dispatcher: &Dispatcher, id: InfoType) -> u64 {
    let mut regs = regs();
    regs.x[1] = u64::from(id.number());
    regs.x[2] = u64::from(Handle::CURRENT_PROCESS.raw());
    regs.x[3] = 0;
    dispatch_on(dispatcher, SvcId::GetInfo.number(), &mut regs, 0);
    assert_eq!(regs.x[0], 0, "{id:?}");
    regs.x[1]
}

/// The four memory sizes `GetInfo` reports: total (6), used (7), non-system total (21) and
/// non-system used (22).
fn memory_info(dispatcher: &Dispatcher) -> [u64; 4] {
    [
        InfoType::TotalMemorySize,
        InfoType::UsedMemorySize,
        InfoType::TotalNonSystemMemorySize,
        InfoType::UsedNonSystemMemorySize,
    ]
    .map(|id| info(dispatcher, id))
}

/// Dispatches `id` with `address` in x0 and `size` in x1, checks that only x0 changed and what
/// was recorded, and returns x0.
fn call(dispatcher: &Dispatcher, id: SvcId, address: u64, size: u64) -> u64 {
    let mut regs = regs();
    regs.x[0] = address;
    regs.x[1] = size;
    let before = regs.clone();
    let (record, events) = dispatch_on(dispatcher, id.number(), &mut regs, 0);
    assert_eq!(record.outcome, Outcome::Completed, "{}", id.name());
    assert_untouched_from(&regs, &before, 1);
    let mut args = [0; SVC_ARG_COUNT];
    args[..2].copy_from_slice(&[address, size]);
    assert_eq!(
        kinds(&events),
        [EventKind::Svc {
            id: u32::from(id.number()),
            args,
            result: starquake_trace::ResultCode(u32::try_from(regs.x[0]).unwrap()),
        }]
    );
    regs.x[0]
}

#[test]
fn mapped_physical_memory_is_used_memory_that_get_info_reports_and_unmapping_gives_it_back() {
    let dispatcher = Dispatcher::new(tables_with_resource(PAGE));
    let alias = alias_start(&dispatcher);
    let size = 3 * PAGE;
    let [total, used, non_system_total, non_system_used] = memory_info(&dispatcher);

    assert_eq!(call(&dispatcher, SvcId::MapPhysicalMemory, alias, size), 0);
    dispatcher.with::<(&AddressSpace,), _>(|(space,)| {
        assert_eq!(
            space.query_info(alias).unwrap().state(),
            MemoryState::Normal
        );
        assert_eq!(space.mapped_physical_memory_size(), size);
    });
    // The used sizes (7, 22) grow by the memory mapped; the totals (6, 21) do not move, as
    // the free memory they add shrinks by as much.
    assert_eq!(
        memory_info(&dispatcher),
        [total, used + size, non_system_total, non_system_used + size]
    );

    assert_eq!(
        call(&dispatcher, SvcId::UnmapPhysicalMemory, alias, size),
        0
    );
    dispatcher.with::<(&AddressSpace,), _>(|(space,)| {
        assert_eq!(space.query_info(alias).unwrap().state(), MemoryState::Free);
        assert_eq!(space.mapped_physical_memory_size(), 0);
    });
    assert_eq!(
        memory_info(&dispatcher),
        [total, used, non_system_total, non_system_used]
    );
}

#[test]
fn a_process_with_less_than_a_page_of_system_resource_is_refused_with_invalid_state() {
    // `GetTotalSystemResourceSize() == 0`: the metadata's size in whole pages, so a size one
    // byte short of a page is none, and exactly a page is one.
    for (resource, expected) in [
        (0, word(KernelError::InvalidState)),
        (PAGE - 1, word(KernelError::InvalidState)),
        (PAGE, 0),
    ] {
        let dispatcher = Dispatcher::new(tables_with_resource(resource));
        let alias = alias_start(&dispatcher);
        for id in [SvcId::MapPhysicalMemory, SvcId::UnmapPhysicalMemory] {
            assert_eq!(
                call(&dispatcher, id, alias, PAGE),
                expected,
                "{} {resource:#x}",
                id.name()
            );
        }
    }
}

#[test]
fn the_physical_memory_calls_declare_the_address_space_exclusively() {
    let dispatcher = Dispatcher::new(tables_with_resource(PAGE));
    for id in [SvcId::MapPhysicalMemory, SvcId::UnmapPhysicalMemory] {
        dispatch_on(&dispatcher, id.number(), &mut GuestRegs::default(), 0);
    }
    assert_eq!(
        dispatcher.lock_waits(),
        LockWaits {
            timed: false,
            lock: waits(2, 2, 0, Duration::ZERO),
            handles: waits(0, 0, 0, Duration::ZERO),
            address_space: waits(2, 2, 0, Duration::ZERO),
            scheduler: waits(0, 0, 0, Duration::ZERO),
        }
    );
}
