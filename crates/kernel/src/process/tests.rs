//! Process creation and the main thread, driven against the fixture's address space.

use starquake_memory::{FlatMemory, PAGE_SIZE};

use super::*;
use crate::handle::MAX_TABLE_SIZE;
use crate::memory::{GUARD_PAGES, MemoryState, fixture};
use crate::object::Object;
use crate::sched::{AffinityMask, CoreId, FIRST_THREAD_ID, SchedError, ThreadState};
use crate::tables::FIRST_USER_PROCESS_ID;

/// Where the main thread starts and its stack's top: distinct, non-zero values.
const ENTRY: u64 = 0x8000_0040;
const STACK_TOP: u64 = 0x10_8000_0000;

fn description() -> ProcessDescription {
    let mut description = ProcessDescription::homebrew(Console::Switch1);
    description.handle_table_size = 16;
    description
}

/// The address of the first thread-local page the fixture's address space hands out: the
/// kernel-map region, past its guard pages.
fn first_tls_page(space: &AddressSpace) -> u64 {
    space.regions().kernel_map.start + GUARD_PAGES * PAGE_SIZE
}

/// Guest memory over the first thread-local page.
fn tls_memory(space: &AddressSpace) -> FlatMemory {
    FlatMemory::new(first_tls_page(space), usize::try_from(PAGE_SIZE).unwrap())
}

#[test]
fn creating_a_process_builds_its_tables() {
    let tables = create(&description(), fixture::loaded().0).unwrap();
    assert_eq!(tables.processes.process_id(), FIRST_USER_PROCESS_ID);
    assert_eq!(
        tables.processes.counter_rate(),
        Console::Switch1.counter_rate()
    );
    assert_eq!(tables.handles.capacity(), 16);
    assert!(tables.handles.is_empty());
    let record = tables.scheduler.process(tables.processes.own()).unwrap();
    assert_eq!(record.state(), ProcessState::Created);
    assert_eq!(record.threads().count(), 0);
    assert_eq!(record.running_threads(), 0);
}

#[test]
fn a_description_that_is_refused_builds_nothing() {
    let mut cores = description();
    cores.cores = Range { first: 1, last: 0 };
    assert_eq!(
        create(&cores, fixture::loaded().0).map(|_| ()),
        Err(CreateError::Process(ProcessError::Capability(
            KernelError::InvalidCombination
        )))
    );
    let mut handles = description();
    handles.handle_table_size = i32::from(MAX_TABLE_SIZE) + 1;
    assert_eq!(
        create(&handles, fixture::loaded().0).map(|_| ()),
        Err(CreateError::Handles(HandleError::TableTooLarge {
            requested: i32::from(MAX_TABLE_SIZE) + 1
        }))
    );
    handles.handle_table_size = i32::from(MAX_TABLE_SIZE);
    assert!(create(&handles, fixture::loaded().0).is_ok());
}

#[test]
fn the_main_thread_starts_at_the_entry_with_its_own_handle() {
    let mut tables = create(&description(), fixture::loaded().0).unwrap();
    let memory = tls_memory(&tables.address_space);
    let tls = first_tls_page(&tables.address_space);
    let main = run(&mut tables, &memory, ENTRY, STACK_TOP).unwrap();

    let thread = tables.scheduler.thread(main.thread).unwrap();
    assert_eq!(thread.state(), ThreadState::Runnable);
    assert_eq!(thread.thread_id(), FIRST_THREAD_ID);
    assert_eq!(thread.priority().get(), description().main_thread_priority);
    let core = CoreId::new(description().main_thread_core).unwrap();
    assert_eq!(thread.affinity(), AffinityMask::single(core));
    assert_eq!(thread.tls_address(), tls);
    // x0 zero, x1 the handle, and the rest of the start state.
    let context = thread.context();
    assert_eq!(
        (context.pc, context.sp, context.tpidrro),
        (ENTRY, STACK_TOP, tls)
    );
    assert_eq!(context.x[0], 0);
    assert_eq!(context.x[1], u64::from(main.handle.raw()));
    assert!(context.x.iter().skip(2).all(|x| *x == 0));
    // The handle is written into the thread's region.
    let mut written = [0; 4];
    memory
        .read(tls + CURRENT_THREAD_HANDLE_OFFSET, &mut written)
        .unwrap();
    assert_eq!(u32::from_le_bytes(written), main.handle.raw());
    // The region's page is thread-local memory.
    let block = tables.address_space.query_info(tls).unwrap();
    assert_eq!(block.state(), MemoryState::ThreadLocal);
    // The handle names the thread; it and the running thread hold its object.
    let object = tables.handles.get(main.handle).unwrap();
    assert_eq!(
        tables.scheduler.objects().get(object),
        Ok(Object::Thread(main.thread))
    );
    assert_eq!(tables.scheduler.objects().refs(object), 2);
    let record = tables.scheduler.process(tables.processes.own()).unwrap();
    assert_eq!(record.state(), ProcessState::Running);
    assert_eq!(record.running_threads(), 1);
}

#[test]
fn a_process_runs_once() {
    let mut tables = create(&description(), fixture::loaded().0).unwrap();
    let memory = tls_memory(&tables.address_space);
    run(&mut tables, &memory, ENTRY, STACK_TOP).unwrap();
    let before = (tables.handles.len(), tables.scheduler.objects().len());
    assert_eq!(
        run(&mut tables, &memory, ENTRY, STACK_TOP),
        Err(SvcError::Kernel(KernelError::InvalidState))
    );
    assert_eq!(
        (tables.handles.len(), tables.scheduler.objects().len()),
        before
    );
    let own = tables.processes.own();
    assert_eq!(tables.scheduler.process(own).unwrap().threads().count(), 1);
}

#[test]
fn a_full_handle_table_leaves_no_main_thread() {
    let mut full = description();
    full.handle_table_size = 1;
    let mut tables = create(&full, fixture::loaded().0).unwrap();
    // Fill the one slot.
    let object = tables
        .scheduler
        .objects_mut()
        .create(Object::Process(tables.processes.own()))
        .unwrap();
    tables
        .handles
        .add(tables.scheduler.objects_mut(), object)
        .unwrap();
    let memory = tls_memory(&tables.address_space);
    assert_eq!(
        run(&mut tables, &memory, ENTRY, STACK_TOP),
        Err(SvcError::Kernel(KernelError::OutOfHandles))
    );
    let record = tables.scheduler.process(tables.processes.own()).unwrap();
    assert_eq!(record.threads().count(), 0);
    assert_eq!(record.tls().used(), 0);
    assert_eq!(record.state(), ProcessState::Created);
    assert_eq!(tables.scheduler.objects().len(), 1);
}

#[test]
fn a_process_the_scheduler_does_not_know_does_not_run() {
    let mut tables = create(&description(), fixture::loaded().0).unwrap();
    tables.scheduler = crate::sched::Scheduler::default();
    let own = tables.processes.own();
    let memory = tls_memory(&tables.address_space);
    assert_eq!(
        run(&mut tables, &memory, ENTRY, STACK_TOP),
        Err(SvcError::from(SchedError::UnknownProcess(own)))
    );
}

#[test]
fn a_region_memory_refuses_is_logged_and_the_thread_still_starts() {
    let mut tables = create(&description(), fixture::loaded().0).unwrap();
    // Memory that does not cover the region at all.
    let memory = FlatMemory::new(0x1000, 0x10);
    let main = run(&mut tables, &memory, ENTRY, STACK_TOP).unwrap();
    assert_eq!(
        tables.scheduler.thread(main.thread).unwrap().state(),
        ThreadState::Runnable
    );
}
