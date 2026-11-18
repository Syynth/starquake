//! `GetResourceLimitLimitValue` and `GetResourceLimitCurrentValue` against a created and
//! started process, beside a second process the tables also hold.

use starquake_memory::{FlatMemory, PAGE_SIZE};

use super::*;
use crate::dispatch::Tables;
use crate::id::{ObjectId, ProcessId};
use crate::memory::GUARD_PAGES;
use crate::memory::fixture::{CODE_SIZE, STACK_SIZE, loaded};
use crate::process::{Console, ProcessDescription};
use crate::resource_limit::{
    EVENT_COUNT_LIMIT, SESSION_COUNT_LIMIT, THREAD_COUNT_LIMIT, TRANSFER_MEMORY_COUNT_LIMIT,
    UNCOUNTED_CURRENT_VALUE,
};
use crate::sched::{Priority, UserThread};
use crate::svc::info::{InfoTables, InfoType, get_info};
use crate::svc::memory::set_heap_size;

const ENTRY: u64 = 0x8000_0040;
/// Pages of guest memory the tests give the thread-local pages.
const TLS_PAGES: u64 = 4;

/// A started process, a handle in its table to its own resource limit, a second process and a
/// handle to the second's limit (which no guest call gives out, but a forged table could hold).
struct Rig {
    tables: Tables,
    limit: Handle,
    second: ProcessId,
    second_limit: Handle,
}

impl Rig {
    fn new() -> Self {
        let (space, _, stack_top) = loaded();
        let mut tables =
            crate::process::create(&ProcessDescription::homebrew(Console::Switch2), space).unwrap();
        let first = tables.address_space.regions().kernel_map.start + GUARD_PAGES * PAGE_SIZE;
        let memory = FlatMemory::new(first, usize::try_from(TLS_PAGES * PAGE_SIZE).unwrap());
        crate::process::run(&mut tables, &memory, ENTRY, stack_top).unwrap();
        let own = tables.processes.own();
        let limit = tables.scheduler.process(own).unwrap().resource_limit();
        let limit = tables
            .handles
            .add(tables.scheduler.objects_mut(), limit)
            .unwrap();
        let mut second_description = ProcessDescription::homebrew(Console::Switch2);
        second_description.program_id = 0x0100_dddd_eeee_f000;
        let second = tables.processes.create(&second_description).unwrap();
        tables.scheduler.add_process(second).unwrap();
        let second_limit = tables.scheduler.process(second).unwrap().resource_limit();
        let second_limit = tables
            .handles
            .add(tables.scheduler.objects_mut(), second_limit)
            .unwrap();
        Self {
            tables,
            limit,
            second,
            second_limit,
        }
    }

    fn view(&self) -> LimitTables<'_> {
        LimitTables {
            handles: &self.tables.handles,
            space: &self.tables.address_space,
            scheduler: &self.tables.scheduler,
            processes: &self.tables.processes,
        }
    }

    /// Both calls, for `which` in the limit `handle` names: (limit value, current value).
    fn ask(&self, handle: Handle, which: u32) -> (CallResult<u64>, CallResult<u64>) {
        (
            get_resource_limit_limit_value(self.view(), handle, which),
            get_resource_limit_current_value(self.view(), handle, which),
        )
    }

    /// `GetInfo(id)` for the calling process.
    fn info(&mut self, id: InfoType) -> u64 {
        let Tables {
            handles,
            address_space,
            scheduler,
            processes,
        } = &mut self.tables;
        let tables = InfoTables {
            handles,
            space: address_space,
            scheduler,
            processes,
        };
        get_info(
            tables,
            crate::sched::CoreId::new(0).unwrap(),
            id.number(),
            Handle::CURRENT_PROCESS,
            0,
        )
        .unwrap()
    }

    /// Creates a thread of `process`, not started, as `CreateThread` does before its handle.
    fn create_thread(&mut self, process: ProcessId) -> ObjectId {
        let spec = UserThread {
            entry: ENTRY,
            argument: 0,
            stack_top: 0,
            priority: Priority::new(44).unwrap(),
            virtual_core: 0,
        };
        self.tables
            .scheduler
            .create_user_thread(&mut self.tables.address_space, process, &spec)
            .unwrap()
            .1
    }
}

fn kernel(error: KernelError) -> CallResult<u64> {
    Err(SvcError::Kernel(error))
}

#[test]
fn an_invalid_resource_is_refused_before_the_handle() {
    let rig = Rig::new();
    let count = u32::try_from(LimitableResource::ALL.len()).unwrap();
    // `IsValidResourceType`: the first number past the last resource, and the top, are
    // refused, with a good handle and with a bad one: the resource is checked first.
    for which in [count, count + 1, u32::MAX] {
        for handle in [rig.limit, Handle::INVALID, Handle::CURRENT_PROCESS] {
            let refused = kernel(KernelError::InvalidEnumValue);
            assert_eq!(
                rig.ask(handle, which),
                (refused.clone(), refused),
                "{which}"
            );
        }
    }
    // The last resource is valid: the handle is checked next.
    let last = count - 1;
    let refused = kernel(KernelError::InvalidHandle);
    assert_eq!(rig.ask(Handle::INVALID, last), (refused.clone(), refused));
    assert_eq!(
        rig.ask(rig.limit, last),
        (Ok(SESSION_COUNT_LIMIT), Ok(UNCOUNTED_CURRENT_VALUE))
    );
}

#[test]
fn only_a_handle_to_a_resource_limit_is_answered() {
    let mut rig = Rig::new();
    let own = rig.tables.processes.own();
    let process_object = rig.tables.scheduler.process(own).unwrap().object();
    let to_process = rig
        .tables
        .handles
        .add(rig.tables.scheduler.objects_mut(), process_object)
        .unwrap();
    let (event, readable) =
        crate::svc::event::create_event(&mut rig.tables.handles, &mut rig.tables.scheduler)
            .unwrap();
    let thread = rig.create_thread(own);
    let to_thread = rig
        .tables
        .handles
        .add(rig.tables.scheduler.objects_mut(), thread)
        .unwrap();
    // A handle to the limit that was closed names nothing.
    let limit = rig.tables.scheduler.process(own).unwrap().resource_limit();
    let closed = rig
        .tables
        .handles
        .add(rig.tables.scheduler.objects_mut(), limit)
        .unwrap();
    crate::svc::handle::close_handle(&mut rig.tables.handles, &mut rig.tables.scheduler, closed)
        .unwrap();
    let which = LimitableResource::ThreadCount.number();
    for handle in [
        Handle::INVALID,
        Handle::CURRENT_PROCESS,
        Handle::CURRENT_THREAD,
        to_process,
        to_thread,
        event,
        readable,
        closed,
    ] {
        let refused = kernel(KernelError::InvalidHandle);
        assert_eq!(
            rig.ask(handle, which),
            (refused.clone(), refused),
            "handle {:#x}",
            handle.raw()
        );
    }
    assert!(rig.ask(rig.limit, which).0.is_ok());
}

#[test]
fn each_resource_is_answered_from_the_limit() {
    let mut rig = Rig::new();
    let ask = |rig: &Rig, resource: LimitableResource| {
        let (limit, current) = rig.ask(rig.limit, resource.number());
        (limit.unwrap(), current.unwrap())
    };
    let pool = rig.tables.address_space.limit().limit();
    // The code and the main thread's stack are reserved at start-up.
    assert_eq!(
        ask(&rig, LimitableResource::PhysicalMemory),
        (pool, CODE_SIZE + STACK_SIZE)
    );
    assert_eq!(
        ask(&rig, LimitableResource::ThreadCount),
        (THREAD_COUNT_LIMIT, 1)
    );
    for (resource, limit) in [
        (LimitableResource::EventCount, EVENT_COUNT_LIMIT),
        (
            LimitableResource::TransferMemoryCount,
            TRANSFER_MEMORY_COUNT_LIMIT,
        ),
        (LimitableResource::SessionCount, SESSION_COUNT_LIMIT),
    ] {
        assert_eq!(ask(&rig, resource), (limit, UNCOUNTED_CURRENT_VALUE));
    }
    // A heap and a thread later, each moves its own value only.
    let heap = 0x20_0000;
    set_heap_size(&mut rig.tables.address_space, heap).unwrap();
    let own = rig.tables.processes.own();
    rig.create_thread(own);
    assert_eq!(
        ask(&rig, LimitableResource::PhysicalMemory),
        (pool, CODE_SIZE + STACK_SIZE + heap)
    );
    assert_eq!(
        ask(&rig, LimitableResource::ThreadCount),
        (THREAD_COUNT_LIMIT, 2)
    );
}

#[test]
fn the_physical_memory_values_agree_with_get_info_6_and_7() {
    let mut rig = Rig::new();
    for heap in [0, 0x20_0000, 0x60_0000, 0x20_0000] {
        set_heap_size(&mut rig.tables.address_space, heap).unwrap();
        let (limit, current) = rig.ask(rig.limit, LimitableResource::PhysicalMemory.number());
        let (limit, current) = (limit.unwrap(), current.unwrap());
        let used = rig.info(InfoType::UsedMemorySize);
        let total = rig.info(InfoType::TotalMemorySize);
        // 7 is what is reserved; 6 is what is used plus what is free, capped at the largest
        // process memory.
        assert_eq!(current, used, "heap {heap:#x}");
        let max = rig.tables.address_space.max_process_memory();
        assert_eq!(total, (used + (limit - current)).min(max), "heap {heap:#x}");
    }
}

#[test]
fn another_process_s_limit_answers_its_threads_but_not_its_memory() {
    let mut rig = Rig::new();
    let second = rig.second;
    rig.create_thread(second);
    rig.create_thread(second);
    let threads = LimitableResource::ThreadCount.number();
    assert_eq!(
        rig.ask(rig.second_limit, threads),
        (Ok(THREAD_COUNT_LIMIT), Ok(2))
    );
    // The calling process's count does not move.
    assert_eq!(rig.ask(rig.limit, threads), (Ok(THREAD_COUNT_LIMIT), Ok(1)));
    // Its physical memory is in an address space these tables do not hold: an internal
    // failure, after the guest's checks.
    let memory = LimitableResource::PhysicalMemory.number();
    let (limit, current) = rig.ask(rig.second_limit, memory);
    assert!(matches!(limit, Err(SvcError::Internal(_))), "{limit:?}");
    assert!(matches!(current, Err(SvcError::Internal(_))), "{current:?}");
    assert_eq!(
        rig.ask(rig.second_limit, u32::MAX),
        (
            kernel(KernelError::InvalidEnumValue),
            kernel(KernelError::InvalidEnumValue)
        )
    );
}
