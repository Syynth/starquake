//! `CreateThread` and `KProcess::Run` against the process's thread limit
//! ([`crate::resource_limit`]): a thread fits at exactly the limit and is refused one past it
//! with `ResultLimitReached`, after the core and priority checks and before anything is made.

use starquake_memory::{FlatMemory, PAGE_SIZE};
use starquake_trace::GuestTicks;

use super::*;
use crate::dispatch::Tables;
use crate::memory::{GUARD_PAGES, fixture};
use crate::process::{Console, ProcessDescription, ProcessState, REGIONS_PER_PAGE};
use crate::resource_limit::THREAD_COUNT_LIMIT;
use crate::sched::CoreSignals;
use crate::svc::handle::close_handle;
use crate::svc::info::{InfoTables, InfoType, get_info};

struct NoSignals;

impl CoreSignals for NoSignals {
    fn interrupt(&mut self, _: CoreId) {}

    fn wake(&mut self, _: CoreId) {}
}

const ENTRY: u64 = 0x8000_0040;
const STACK_TOP: u64 = 0x10_8000_0000;

/// The thread limit as a count of rows.
fn limit() -> usize {
    usize::try_from(THREAD_COUNT_LIMIT).unwrap()
}

/// Guest memory for the thread-local pages of [`THREAD_COUNT_LIMIT`] threads, and a page more.
fn tls_memory(tables: &Tables) -> FlatMemory {
    let first = tables.address_space.regions().kernel_map.start + GUARD_PAGES * PAGE_SIZE;
    let pages = limit().div_ceil(REGIONS_PER_PAGE) + 1;
    FlatMemory::new(first, pages * usize::try_from(PAGE_SIZE).unwrap())
}

fn core(index: u8) -> CoreId {
    CoreId::new(index).unwrap()
}

/// A started homebrew process whose main thread runs on core 0.
struct World {
    tables: Tables,
    memory: FlatMemory,
}

impl World {
    fn new() -> Self {
        let description = ProcessDescription::homebrew(Console::Switch1);
        let mut tables = crate::process::create(&description, fixture::loaded().0).unwrap();
        let memory = tls_memory(&tables);
        let main = crate::process::run(&mut tables, &memory, ENTRY, STACK_TOP).unwrap();
        let scheduler = &mut tables.scheduler;
        scheduler
            .update(GuestTicks(0), None, &mut NoSignals)
            .unwrap();
        assert_eq!(
            scheduler.schedule(core(0), &mut NoSignals),
            Ok(Some(main.thread))
        );
        Self { tables, memory }
    }

    fn create(&mut self, args: CreateThread) -> CallResult<Handle> {
        let Tables {
            handles,
            address_space,
            scheduler,
            processes,
        } = &mut self.tables;
        create_thread(
            (handles, address_space, scheduler),
            processes,
            &self.memory,
            core(0),
            args,
        )
    }

    fn close(&mut self, handle: Handle) {
        let Tables {
            handles, scheduler, ..
        } = &mut self.tables;
        close_handle(handles, scheduler, handle).unwrap();
    }

    /// The process's threads that still have a row.
    fn threads(&self) -> usize {
        let own = self.tables.processes.own();
        self.tables.scheduler.process(own).unwrap().thread_count()
    }

    /// Creates threads until the process has `count`, and returns their handles.
    fn fill_to(&mut self, count: usize) -> Vec<Handle> {
        let mut handles = Vec::new();
        while self.threads() < count {
            handles.push(self.create(args(0)).unwrap());
        }
        handles
    }

    /// `GetInfo`'s `FreeThreadCount` for the calling process.
    fn free(&mut self) -> u64 {
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
            core(0),
            InfoType::FreeThreadCount.number(),
            Handle::CURRENT_PROCESS,
            0,
        )
        .unwrap()
    }
}

/// A thread at priority 44 on virtual core `core_id`.
fn args(core_id: i32) -> CreateThread {
    CreateThread {
        entry: 0x8000_2000,
        argument: 0x1234_5678,
        stack_top: 0x10_9000_0000,
        priority: 44,
        core_id,
    }
}

fn kernel<T>(error: KernelError) -> CallResult<T> {
    Err(SvcError::Kernel(error))
}

#[test]
fn a_thread_fits_at_exactly_the_limit_and_one_past_it_is_limit_reached() {
    let mut world = World::new();
    // One below the limit: the next thread is the limit's last.
    world.fill_to(limit() - 1);
    assert_eq!(world.free(), 1);
    let last = world.create(args(0)).unwrap();
    assert_eq!(world.threads(), limit());
    assert_eq!(world.free(), 0);
    // One past it: refused, and nothing was made (no row, no region, no handle).
    let before = (
        world.tables.handles.clone(),
        world.tables.scheduler.clone(),
        world.tables.address_space.limit().reserved(),
    );
    assert_eq!(world.create(args(0)), kernel(KernelError::LimitReached));
    assert_eq!(
        (
            world.tables.handles.clone(),
            world.tables.scheduler.clone(),
            world.tables.address_space.limit().reserved(),
        ),
        before
    );
    // A thread destroyed makes room for exactly one.
    world.close(last);
    assert_eq!(world.free(), 1);
    assert!(world.create(args(0)).is_ok());
    assert_eq!(world.create(args(0)), kernel(KernelError::LimitReached));
}

#[test]
fn the_core_and_priority_are_checked_before_the_limit() {
    let mut world = World::new();
    world.fill_to(limit());
    // `CreateThread`'s order: the core, then the priority, then the reservation.
    let mut bad_core = args(0);
    bad_core.core_id = 9;
    assert_eq!(world.create(bad_core), kernel(KernelError::InvalidCoreId));
    let mut bad_priority = args(0);
    bad_priority.priority = 64;
    assert_eq!(
        world.create(bad_priority),
        kernel(KernelError::InvalidPriority)
    );
    assert_eq!(world.create(args(0)), kernel(KernelError::LimitReached));
}

#[test]
fn the_limit_is_checked_before_the_handle_table() {
    let mut world = World::new();
    let mut threads = world.fill_to(limit());
    let own = world.tables.processes.own();
    let object = world.tables.scheduler.process(own).unwrap().object();
    let fill = |world: &mut World| {
        while world
            .tables
            .handles
            .add(world.tables.scheduler.objects_mut(), object)
            .is_ok()
        {}
    };
    // At the limit with a full handle table: the limit is what refuses.
    fill(&mut world);
    assert_eq!(world.create(args(0)), kernel(KernelError::LimitReached));
    // Below the limit with a full handle table: the table is.
    world.close(threads.pop().unwrap());
    fill(&mut world);
    assert_eq!(world.threads(), limit() - 1);
    assert_eq!(world.create(args(0)), kernel(KernelError::OutOfHandles));
}

#[test]
fn an_ended_thread_with_an_open_handle_still_counts_and_nothing_waits() {
    // Divergence: Eden's `CreateThread` waits up to 100 ms for an ended thread's release;
    // here the call is refused at once, until the handle is closed.
    let mut world = World::new();
    let mut handles = world.fill_to(limit());
    let ended = handles.pop().unwrap();
    let thread = thread_from_handle(
        &world.tables.scheduler,
        &world.tables.handles,
        core(0),
        ended,
    )
    .unwrap();
    world.tables.scheduler.start(thread).unwrap();
    world.tables.scheduler.terminate_thread(thread).unwrap();
    assert_eq!(world.threads(), limit());
    assert_eq!(world.create(args(0)), kernel(KernelError::LimitReached));
    world.close(ended);
    assert_eq!(world.threads(), limit() - 1);
    assert!(world.create(args(0)).is_ok());
}

#[test]
fn starting_a_process_at_its_thread_limit_is_limit_reached_after_its_state() {
    // `KProcess::Run`: the state, then the thread reservation.
    let description = ProcessDescription::homebrew(Console::Switch1);
    let mut tables = crate::process::create(&description, fixture::loaded().0).unwrap();
    let memory = tls_memory(&tables);
    let own = tables.processes.own();
    let spec = UserThread {
        entry: ENTRY,
        argument: 0,
        stack_top: STACK_TOP,
        priority: crate::sched::Priority::new(44).unwrap(),
        virtual_core: 0,
    };
    // Threads made before the process runs (no guest call can, but the scheduler can), up to
    // the limit.
    let mut objects = Vec::new();
    for _ in 0..limit() {
        let (_, object) = tables
            .scheduler
            .create_user_thread(&mut tables.address_space, own, &spec)
            .unwrap();
        objects.push(object);
    }
    let before = tables.scheduler.clone();
    assert_eq!(
        crate::process::run(&mut tables, &memory, ENTRY, STACK_TOP),
        kernel(KernelError::LimitReached)
    );
    assert_eq!(tables.scheduler, before);
    assert_eq!(
        tables.scheduler.process(own).unwrap().state(),
        ProcessState::Created
    );
    // One fewer: it starts.
    tables
        .scheduler
        .close_object(objects.pop().unwrap())
        .unwrap();
    assert!(crate::process::run(&mut tables, &memory, ENTRY, STACK_TOP).is_ok());
    // Running, at the limit again: the state is checked first.
    assert_eq!(
        crate::process::run(&mut tables, &memory, ENTRY, STACK_TOP),
        kernel(KernelError::InvalidState)
    );
}
