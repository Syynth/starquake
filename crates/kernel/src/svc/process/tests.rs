//! Ending a process, and `GetProcessId`.

use starquake_memory::{FlatMemory, PAGE_SIZE};

use super::*;
use crate::dispatch::Tables;
use crate::handle::ObjectRefs;
use crate::id::SlabId;
use crate::memory::{GUARD_PAGES, fixture};
use crate::process::{Console, MainThread, ProcessDescription};
use crate::sched::{Priority, UserThread};
use crate::tables::FIRST_USER_PROCESS_ID;

const ENTRY: u64 = 0x8000_0040;
const STACK_TOP: u64 = 0x10_8000_0000;

/// A created process, and memory over its first thread-local page.
fn created() -> (Tables, FlatMemory) {
    let tables = crate::process::create(
        &ProcessDescription::homebrew(Console::Switch2),
        fixture::loaded().0,
    )
    .unwrap();
    let first = tables.address_space.regions().kernel_map.start + GUARD_PAGES * PAGE_SIZE;
    let memory = FlatMemory::new(first, usize::try_from(PAGE_SIZE).unwrap());
    (tables, memory)
}

/// A process whose main thread was started.
fn running() -> (Tables, MainThread) {
    let (mut tables, memory) = created();
    let main = crate::process::run(&mut tables, &memory, ENTRY, STACK_TOP).unwrap();
    (tables, main)
}

/// Adds a thread of the own process and a handle to it, and starts it if `start`.
fn add_thread(tables: &mut Tables, start: bool) -> (crate::id::ThreadId, Handle) {
    let own = tables.processes.own();
    let (thread, object) = tables
        .scheduler
        .create_user_thread(
            &mut tables.address_space,
            own,
            &UserThread {
                entry: ENTRY,
                argument: 0,
                stack_top: STACK_TOP,
                priority: Priority::new(44).unwrap(),
                virtual_core: 1,
            },
        )
        .unwrap();
    let handle = tables
        .handles
        .add(tables.scheduler.objects_mut(), object)
        .unwrap();
    tables.scheduler.close_object(object).unwrap();
    if start {
        tables.scheduler.start(thread).unwrap();
    }
    (thread, handle)
}

fn state(tables: &Tables) -> ProcessState {
    let own = tables.processes.own();
    tables.scheduler.process(own).unwrap().state()
}

#[test]
fn exiting_the_process_ends_every_thread_and_closes_every_handle() {
    let (mut tables, main) = running();
    let (started, _) = add_thread(&mut tables, true);
    let (idle, _) = add_thread(&mut tables, false);
    // A second handle to the main thread.
    let object = tables.handles.get(main.handle).unwrap();
    tables
        .handles
        .add(tables.scheduler.objects_mut(), object)
        .unwrap();
    assert_eq!(state(&tables), ProcessState::Running);
    let Tables {
        handles,
        scheduler,
        processes,
        ..
    } = &mut tables;
    exit_process(handles, scheduler, processes).unwrap();
    assert!(tables.handles.is_empty());
    assert_eq!(state(&tables), ProcessState::Terminated);
    // With every handle closed and every thread ended, every thread is destroyed and every
    // region given back.
    for thread in [main.thread, started, idle] {
        assert!(tables.scheduler.thread(thread).is_err(), "{thread:?}");
    }
    let own = tables.processes.own();
    let record = tables.scheduler.process(own).unwrap();
    assert_eq!(record.tls().used(), 0);
    assert_eq!(record.running_threads(), 0);
    assert!(tables.scheduler.objects().is_empty());
}

#[test]
fn a_process_ends_once() {
    // A created process may end (Eden asserts it is running), and then not again.
    let (mut tables, _) = created();
    assert_eq!(state(&tables), ProcessState::Created);
    let Tables {
        handles,
        scheduler,
        processes,
        ..
    } = &mut tables;
    exit_process(handles, scheduler, processes).unwrap();
    assert_eq!(
        exit_process(handles, scheduler, processes),
        Err(SvcError::Internal(format!(
            "process {:?} has ended already",
            processes.own()
        )))
    );
    assert_eq!(state(&tables), ProcessState::Terminated);
}

#[test]
fn an_ended_thread_is_left_as_it_ended() {
    let (mut tables, main) = running();
    let (thread, handle) = add_thread(&mut tables, true);
    tables.scheduler.terminate_thread(thread).unwrap();
    let own = tables.processes.own();
    assert_eq!(tables.scheduler.process(own).unwrap().running_threads(), 1);
    terminate_process(&mut tables.handles, &mut tables.scheduler, own).unwrap();
    assert!(tables.scheduler.thread(thread).is_err());
    assert!(tables.scheduler.thread(main.thread).is_err());
    assert_eq!(tables.handles.get(handle).ok(), None);
}

#[test]
fn get_process_id_names_the_process_of_a_process_or_a_thread() {
    let (mut tables, main) = running();
    let own = tables.processes.own();
    let object = tables
        .scheduler
        .objects_mut()
        .create(Object::Process(own))
        .unwrap();
    let process_handle = tables
        .handles
        .add(tables.scheduler.objects_mut(), object)
        .unwrap();
    tables.scheduler.objects_mut().close(object).unwrap();
    let id = |tables: &Tables, handle| {
        get_process_id(
            &tables.handles,
            &tables.scheduler,
            &tables.processes,
            handle,
        )
    };
    for named in [Handle::CURRENT_PROCESS, main.handle, process_handle] {
        assert_eq!(id(&tables, named), Ok(FIRST_USER_PROCESS_ID), "{named:?}");
    }
    // For this lookup the thread pseudo-handle names nothing.
    for nothing in [Handle::CURRENT_THREAD, Handle::INVALID] {
        assert_eq!(
            id(&tables, nothing),
            Err(SvcError::Kernel(KernelError::InvalidHandle)),
            "{nothing:?}"
        );
    }
}

#[test]
fn get_process_id_refuses_a_session_and_a_thread_of_no_process() {
    let (mut tables, _) = running();
    let session = tables
        .scheduler
        .objects_mut()
        .create(Object::Session)
        .unwrap();
    let session = tables
        .handles
        .add(tables.scheduler.objects_mut(), session)
        .unwrap();
    let stray = tables
        .scheduler
        .create_thread(
            Priority::new(44).unwrap(),
            crate::sched::CoreId::new(0).unwrap(),
        )
        .unwrap();
    let stray = tables
        .scheduler
        .objects_mut()
        .create(Object::Thread(stray))
        .unwrap();
    let stray = tables
        .handles
        .add(tables.scheduler.objects_mut(), stray)
        .unwrap();
    for handle in [session, stray] {
        assert_eq!(
            get_process_id(
                &tables.handles,
                &tables.scheduler,
                &tables.processes,
                handle
            ),
            Err(SvcError::Kernel(KernelError::InvalidHandle)),
            "{handle:?}"
        );
    }
    // A process object naming a process the table does not have is broken bookkeeping.
    let missing = ProcessId::from_parts(7, 7);
    let object = tables
        .scheduler
        .objects_mut()
        .create(Object::Process(missing))
        .unwrap();
    let handle = tables
        .handles
        .add(tables.scheduler.objects_mut(), object)
        .unwrap();
    assert!(matches!(
        get_process_id(
            &tables.handles,
            &tables.scheduler,
            &tables.processes,
            handle
        ),
        Err(SvcError::Internal(_))
    ));
}
