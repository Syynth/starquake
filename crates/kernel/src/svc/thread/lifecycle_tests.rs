//! The thread lifecycle calls, driven against a process whose main thread runs on core 0.

use starquake_memory::{FlatMemory, PAGE_SIZE};
use starquake_trace::GuestTicks;

use super::*;
use crate::dispatch::Tables;
use crate::memory::{GUARD_PAGES, fixture};
use crate::process::{
    CURRENT_THREAD_HANDLE_OFFSET, Console, MainThread, ProcessDescription, Range,
    THREAD_LOCAL_REGION_SIZE,
};
use crate::sched::{CoreSignals, FIRST_THREAD_ID, ThreadState};
use crate::svc::handle::close_handle;

struct NoSignals;

impl CoreSignals for NoSignals {
    fn interrupt(&mut self, _: CoreId) {}

    fn wake(&mut self, _: CoreId) {}
}

const ENTRY: u64 = 0x8000_0040;
const STACK_TOP: u64 = 0x10_8000_0000;
/// Pages of guest memory the tests give the thread-local pages: more than they use.
const TLS_PAGES: u64 = 64;

fn core(index: u8) -> CoreId {
    CoreId::new(index).unwrap()
}

/// A process with `description`, its main thread running on core 0.
struct World {
    tables: Tables,
    memory: FlatMemory,
    main: MainThread,
}

impl World {
    fn new(description: &ProcessDescription) -> Self {
        let mut tables = crate::process::create(description, fixture::loaded().0).unwrap();
        let first = tables.address_space.regions().kernel_map.start + GUARD_PAGES * PAGE_SIZE;
        let memory = FlatMemory::new(first, usize::try_from(TLS_PAGES * PAGE_SIZE).unwrap());
        let main = crate::process::run(&mut tables, &memory, ENTRY, STACK_TOP).unwrap();
        let mut world = Self {
            tables,
            memory,
            main,
        };
        world.schedule(core(0), Some(main.thread));
        world
    }

    /// The homebrew process: virtual cores 0-3, priorities 16-63, built for Switch 1.
    fn homebrew() -> Self {
        Self::new(&ProcessDescription::homebrew(Console::Switch1))
    }

    /// A Switch 2 process that may use virtual cores 0-7.
    fn switch2() -> Self {
        let mut description = ProcessDescription::homebrew(Console::Switch2);
        description.cores = Range { first: 0, last: 7 };
        Self::new(&description)
    }

    fn scheduler(&mut self) -> &mut Scheduler {
        &mut self.tables.scheduler
    }

    /// Updates the scheduler and has `on` switch to what it chose, which must be `expected`.
    fn schedule(&mut self, on: CoreId, expected: Option<ThreadId>) {
        let scheduler = self.scheduler();
        scheduler
            .update(GuestTicks(0), None, &mut NoSignals)
            .unwrap();
        assert_eq!(scheduler.schedule(on, &mut NoSignals), Ok(expected));
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

    fn start(&mut self, handle: Handle) -> CallResult<()> {
        let Tables {
            handles, scheduler, ..
        } = &mut self.tables;
        start_thread(handles, scheduler, core(0), handle)
    }

    fn close(&mut self, handle: Handle) -> CallResult<()> {
        let Tables {
            handles, scheduler, ..
        } = &mut self.tables;
        close_handle(handles, scheduler, handle)
    }

    fn exit(&mut self, on: CoreId) -> CallResult<ThreadExit> {
        let Tables {
            handles, scheduler, ..
        } = &mut self.tables;
        exit_thread(handles, scheduler, on)
    }

    fn set_priority(&mut self, handle: Handle, priority: i32) -> CallResult<()> {
        let Tables {
            handles,
            scheduler,
            processes,
            ..
        } = &mut self.tables;
        set_thread_priority(handles, scheduler, processes, core(0), handle, priority)
    }

    fn set_core_mask(&mut self, handle: Handle, core_id: i32, mask: u64) -> CallResult<()> {
        let Tables {
            handles,
            scheduler,
            processes,
            ..
        } = &mut self.tables;
        set_thread_core_mask(
            handles,
            scheduler,
            processes,
            core(0),
            handle,
            (core_id, mask),
        )
    }

    fn core_mask(&self, handle: Handle) -> CallResult<(i32, u64)> {
        get_thread_core_mask(
            &self.tables.scheduler,
            &self.tables.handles,
            core(0),
            handle,
        )
    }

    fn thread_id(&self, handle: Handle) -> CallResult<u64> {
        get_thread_id(
            &self.tables.scheduler,
            &self.tables.handles,
            core(0),
            handle,
        )
    }

    /// The thread a handle names.
    fn thread(&self, handle: Handle) -> ThreadId {
        thread_from_handle(
            &self.tables.scheduler,
            &self.tables.handles,
            core(0),
            handle,
        )
        .unwrap()
    }

    fn tls_used(&self) -> usize {
        let own = self.tables.processes.own();
        self.tables.scheduler.process(own).unwrap().tls().used()
    }
}

/// A thread at priority 44 on virtual core `core_id`, with distinct start values.
fn args(core_id: i32) -> CreateThread {
    CreateThread {
        entry: 0x8000_2000,
        argument: 0x1234_5678,
        stack_top: 0x10_9000_0000,
        priority: 44,
        core_id,
    }
}

fn kernel(error: KernelError) -> CallResult<()> {
    Err(SvcError::Kernel(error))
}

#[test]
fn create_start_and_exit_a_thread() {
    let mut world = World::homebrew();
    let main_tls = world
        .tables
        .scheduler
        .thread(world.main.thread)
        .unwrap()
        .tls_address();
    let handle = world.create(args(1)).unwrap();
    let thread = world.thread(handle);
    let row = world.tables.scheduler.thread(thread).unwrap();
    assert_eq!(row.state(), ThreadState::Initialized);
    // The next region of the main thread's page, and the handle written into it.
    let tls = main_tls + THREAD_LOCAL_REGION_SIZE;
    assert_eq!(row.tls_address(), tls);
    let context = row.context();
    let expected = args(1);
    assert_eq!(
        (context.pc, context.sp, context.x[0], context.tpidrro),
        (expected.entry, expected.stack_top, expected.argument, tls)
    );
    let mut written = [0; 4];
    world
        .memory
        .read(tls + CURRENT_THREAD_HANDLE_OFFSET, &mut written)
        .unwrap();
    assert_eq!(u32::from_le_bytes(written), handle.raw());
    assert_eq!(world.thread_id(handle), Ok(FIRST_THREAD_ID + 1));

    world.start(handle).unwrap();
    assert_eq!(
        world.tables.scheduler.thread(thread).unwrap().state(),
        ThreadState::Runnable
    );
    // Starting it again: it is not newly created.
    assert_eq!(world.start(handle), kernel(KernelError::InvalidState));
    world.schedule(core(1), Some(thread));

    // It exits; the main thread still runs, so the process goes on.
    assert_eq!(world.exit(core(1)), Ok(ThreadExit::Others));
    assert_eq!(
        world.tables.scheduler.thread(thread).unwrap().state(),
        ThreadState::Terminated
    );
    // The handle still names it: the guest can still ask about it.
    assert_eq!(world.thread_id(handle), Ok(FIRST_THREAD_ID + 1));
    // WAIT SEAM (kernel slice 4, #111): waiting on the handle for the exit (a join) belongs to
    // `WaitSynchronization`; this checks only that the ended thread is what such a wait sees.
    let own = world.tables.processes.own();
    assert_eq!(
        world
            .tables
            .scheduler
            .process(own)
            .unwrap()
            .running_threads(),
        1
    );
}

#[test]
fn a_thread_that_exits_frees_its_tls_slot_and_the_slot_is_reused() {
    let mut world = World::homebrew();
    let handle = world.create(args(1)).unwrap();
    let thread = world.thread(handle);
    let tls = world.tables.scheduler.thread(thread).unwrap().tls_address();
    world.start(handle).unwrap();
    world.schedule(core(1), Some(thread));
    assert_eq!(world.tls_used(), 2);
    world.exit(core(1)).unwrap();
    // Its handle still holds it, so the slot is still taken.
    assert_eq!(world.tls_used(), 2);
    world.close(handle).unwrap();
    assert!(world.tables.scheduler.thread(thread).is_err());
    assert_eq!(world.tls_used(), 1);
    // The next thread gets the same region.
    let next = world.create(args(2)).unwrap();
    let next = world.thread(next);
    assert_eq!(
        world.tables.scheduler.thread(next).unwrap().tls_address(),
        tls
    );
}

#[test]
fn a_thread_closed_before_it_starts_is_destroyed() {
    let mut world = World::homebrew();
    let handle = world.create(args(0)).unwrap();
    let thread = world.thread(handle);
    assert_eq!(world.tls_used(), 2);
    world.close(handle).unwrap();
    assert!(world.tables.scheduler.thread(thread).is_err());
    assert_eq!(world.tls_used(), 1);
}

#[test]
fn the_last_thread_to_exit_ends_the_process() {
    let mut world = World::homebrew();
    let own = world.tables.processes.own();
    assert_eq!(world.exit(core(0)), Ok(ThreadExit::LastOf(own)));
    let record = world.tables.scheduler.process(own).unwrap();
    assert_eq!(record.state(), crate::process::ProcessState::Terminated);
    assert!(world.tables.handles.is_empty());
    // The main thread's handle was the last reference: it is gone, and its region with it.
    assert!(world.tables.scheduler.thread(world.main.thread).is_err());
    assert_eq!(world.tls_used(), 0);
}

#[test]
fn create_thread_checks_the_core_then_the_priority() {
    let mut world = World::homebrew();
    let before = world.tls_used();
    let with = |core_id, priority| CreateThread {
        priority,
        ..args(core_id)
    };
    let invalid_core = Err(SvcError::Kernel(KernelError::InvalidCoreId));
    let invalid_priority = Err(SvcError::Kernel(KernelError::InvalidPriority));
    // A Switch 1 process names virtual cores 0-3: 4 is not one, nor is "don't care".
    for core_id in [
        4,
        IDEAL_CORE_DONT_CARE,
        IDEAL_CORE_NO_UPDATE,
        i32::MIN,
        i32::MAX,
    ] {
        assert_eq!(world.create(with(core_id, 44)), invalid_core, "{core_id}");
    }
    // Both faults at once: the core is checked first.
    assert_eq!(world.create(with(4, 64)), invalid_core);
    // The priority: outside 0-63, or outside the process's 16-63.
    for priority in [-1, 15, 64, i32::MIN, i32::MAX] {
        assert_eq!(
            world.create(with(0, priority)),
            invalid_priority,
            "{priority}"
        );
    }
    assert_eq!(world.tls_used(), before, "a refused call takes no region");
    // The edges that pass: core 3, priorities 16 and 63, and the process's own core.
    for (core_id, priority) in [(3, 16), (0, 63), (IDEAL_CORE_USE_PROCESS_VALUE, 44)] {
        let handle = world.create(with(core_id, priority)).unwrap();
        let thread = world.thread(handle);
        let row = world.tables.scheduler.thread(thread).unwrap();
        let virtual_core = if core_id == IDEAL_CORE_USE_PROCESS_VALUE {
            0
        } else {
            core_id
        };
        assert_eq!(row.virtual_core_mask().0, virtual_core);
        assert_eq!(i32::from(row.priority().get()), priority);
    }
}

#[test]
fn a_core_must_be_valid_for_the_console_and_in_the_process_mask() {
    // Switch 1 with cores 0-5 in its mask: 4 is in the mask but not a core it may name.
    let mut description = ProcessDescription::homebrew(Console::Switch1);
    description.cores = Range { first: 0, last: 5 };
    let mut switch1 = World::new(&description);
    assert_eq!(
        switch1.create(args(4)),
        Err(SvcError::Kernel(KernelError::InvalidCoreId))
    );
    // Switch 2 may name 4, and 7 (the system core) when its mask has it, but not 8.
    let mut switch2 = World::switch2();
    for core_id in [4, 7] {
        let handle = switch2.create(args(core_id)).unwrap();
        assert_eq!(switch2.core_mask(handle).map(|mask| mask.0), Ok(core_id));
    }
    assert_eq!(
        switch2.create(args(8)),
        Err(SvcError::Kernel(KernelError::InvalidCoreId))
    );
}

#[test]
fn a_failed_create_leaves_nothing_behind() {
    let mut description = ProcessDescription::homebrew(Console::Switch1);
    // One handle: the main thread's.
    description.handle_table_size = 1;
    let mut world = World::new(&description);
    let objects = world.tables.scheduler.objects().len();
    assert_eq!(
        world.create(args(0)),
        Err(SvcError::Kernel(KernelError::OutOfHandles))
    );
    assert_eq!(world.tls_used(), 1);
    assert_eq!(world.tables.scheduler.objects().len(), objects);
    let own = world.tables.processes.own();
    assert_eq!(
        world
            .tables
            .scheduler
            .process(own)
            .unwrap()
            .threads()
            .count(),
        1
    );
}

#[test]
fn starting_checks_the_handle_then_the_state() {
    let mut world = World::homebrew();
    assert_eq!(
        world.start(Handle::INVALID),
        kernel(KernelError::InvalidHandle)
    );
    // The calling thread runs already.
    assert_eq!(
        world.start(Handle::CURRENT_THREAD),
        kernel(KernelError::InvalidState)
    );
}

#[test]
fn set_thread_priority_checks_the_priority_then_the_handle() {
    let mut world = World::homebrew();
    let invalid_priority = kernel(KernelError::InvalidPriority);
    // Both faults at once: the priority is checked first.
    assert_eq!(world.set_priority(Handle::INVALID, 64), invalid_priority);
    for priority in [-1, 15, 64] {
        assert_eq!(
            world.set_priority(Handle::CURRENT_THREAD, priority),
            invalid_priority
        );
    }
    assert_eq!(
        world.set_priority(Handle::INVALID, 30),
        kernel(KernelError::InvalidHandle)
    );
    assert_eq!(
        world.set_priority(Handle::CURRENT_PROCESS, 30),
        kernel(KernelError::InvalidHandle)
    );
    for priority in [16, 63] {
        world.set_priority(world.main.handle, priority).unwrap();
        let row = world.tables.scheduler.thread(world.main.thread).unwrap();
        assert_eq!(i32::from(row.base_priority().get()), priority);
        assert_eq!(i32::from(row.priority().get()), priority);
    }
}

#[test]
fn the_core_mask_reads_back_as_the_guest_set_it() {
    let mut world = World::homebrew();
    let main = world.main.handle;
    assert_eq!(world.core_mask(main), Ok((0, 1)));
    assert_eq!(world.core_mask(Handle::CURRENT_THREAD), Ok((0, 1)));
    world.set_core_mask(main, 1, 0b11).unwrap();
    assert_eq!(world.core_mask(main), Ok((1, 0b11)));
    let row = world.tables.scheduler.thread(world.main.thread).unwrap();
    assert_eq!(row.ideal_core(), Some(core(1)));
    assert_eq!(row.affinity().bits(), 0b11);
    // No update keeps core 1, which is in 0b10 and not in 0b01.
    world
        .set_core_mask(main, IDEAL_CORE_NO_UPDATE, 0b10)
        .unwrap();
    assert_eq!(world.core_mask(main), Ok((1, 0b10)));
    assert_eq!(
        world.set_core_mask(main, IDEAL_CORE_NO_UPDATE, 0b01),
        kernel(KernelError::InvalidCombination)
    );
    assert_eq!(world.core_mask(main), Ok((1, 0b10)));
    world
        .set_core_mask(main, IDEAL_CORE_DONT_CARE, 0b100)
        .unwrap();
    assert_eq!(world.core_mask(main), Ok((IDEAL_CORE_DONT_CARE, 0b100)));
    // The process's value: its ideal core alone, whatever mask is passed.
    world
        .set_core_mask(main, IDEAL_CORE_USE_PROCESS_VALUE, u64::MAX)
        .unwrap();
    assert_eq!(world.core_mask(main), Ok((0, 1)));
}

#[test]
fn set_thread_core_mask_checks_in_edens_order() {
    let mut world = World::homebrew();
    let main = world.main.handle;
    let bad = Handle::INVALID;
    // The process has cores 0-3. A mask past them, with every later fault too.
    assert_eq!(
        world.set_core_mask(bad, 5, 1 << 4),
        kernel(KernelError::InvalidCoreId)
    );
    // Then an empty mask.
    assert_eq!(
        world.set_core_mask(bad, 5, 0),
        kernel(KernelError::InvalidCombination)
    );
    // Then a valid core outside the mask.
    assert_eq!(
        world.set_core_mask(bad, 2, 0b11),
        kernel(KernelError::InvalidCombination)
    );
    // Then a core that is neither valid nor one of the two special values.
    for core_id in [4, IDEAL_CORE_NO_UPDATE - 1, i32::MIN] {
        assert_eq!(
            world.set_core_mask(bad, core_id, 0b11),
            kernel(KernelError::InvalidCoreId),
            "{core_id}"
        );
    }
    // Then the handle.
    assert_eq!(
        world.set_core_mask(bad, 1, 0b11),
        kernel(KernelError::InvalidHandle)
    );
    // The edges that pass: the process's whole mask, and its last core.
    world.set_core_mask(main, 3, 0b1111).unwrap();
    assert_eq!(world.core_mask(main), Ok((3, 0b1111)));
}

#[test]
fn a_switch_2_thread_on_a_system_core_runs_on_physical_core_3() {
    let mut world = World::switch2();
    let main = world.main.handle;
    world.set_core_mask(main, 6, (1 << 6) | (1 << 1)).unwrap();
    assert_eq!(world.core_mask(main), Ok((6, (1 << 6) | (1 << 1))));
    let row = world.tables.scheduler.thread(world.main.thread).unwrap();
    assert_eq!(row.ideal_core(), Some(core(3)));
    assert_eq!(row.affinity().bits(), core(1).bit() | core(3).bit());
}

#[test]
fn changing_the_mask_of_a_running_thread_returns_without_waiting() {
    let mut world = World::homebrew();
    let main = world.main.thread;
    // The main thread runs on core 0; its new mask leaves core 0 out.
    world.set_core_mask(world.main.handle, 1, 0b10).unwrap();
    // Still current on core 0 when the call returns (Eden would not have returned yet).
    assert_eq!(world.tables.scheduler.running_on(main), Some(core(0)));
    // Core 0 learns it must let the thread go; core 1 takes it.
    world
        .scheduler()
        .update(GuestTicks(1), None, &mut NoSignals)
        .unwrap();
    assert_eq!(world.tables.scheduler.needs_scheduling(core(0)), Ok(true));
    assert_eq!(world.tables.scheduler.highest(core(1)), Ok(Some(main)));
    // A mask that keeps its core changes nothing about where it runs.
    let mut other = World::homebrew();
    other.set_core_mask(other.main.handle, 0, 0b11).unwrap();
    other
        .scheduler()
        .update(GuestTicks(1), None, &mut NoSignals)
        .unwrap();
    assert_eq!(
        other.tables.scheduler.highest(core(0)),
        Ok(Some(other.main.thread))
    );
}

#[test]
fn sleep_thread_yields_by_its_three_negative_values() {
    for (ns, moves_to_the_back) in [
        (YIELD_WITHOUT_CORE_MIGRATION, true),
        (YIELD_WITH_CORE_MIGRATION, true),
        (YIELD_TO_ANY_THREAD, true),
        (YIELD_TO_ANY_THREAD - 1, false),
        (i64::MIN, false),
    ] {
        let mut world = World::homebrew();
        // A second thread at the main thread's priority on core 0, queued behind it.
        let handle = world.create(args(0)).unwrap();
        let second = world.thread(handle);
        world.start(handle).unwrap();
        world.schedule(core(0), Some(world.main.thread));
        sleep_thread(world.scheduler(), core(0), ns).unwrap();
        world
            .scheduler()
            .update(GuestTicks(1), None, &mut NoSignals)
            .unwrap();
        let expected = if moves_to_the_back {
            second
        } else {
            world.main.thread
        };
        assert_eq!(
            world.tables.scheduler.highest(core(0)),
            Ok(Some(expected)),
            "{ns}"
        );
    }
}

#[test]
fn a_positive_sleep_returns_at_once_and_changes_nothing() {
    for ns in [1, i64::MAX] {
        let mut world = World::homebrew();
        let handle = world.create(args(0)).unwrap();
        world.start(handle).unwrap();
        world.schedule(core(0), Some(world.main.thread));
        let before = world.tables.scheduler.clone();
        sleep_thread(world.scheduler(), core(0), ns).unwrap();
        assert_eq!(world.tables.scheduler, before, "{ns}");
    }
}

#[test]
fn get_thread_id_names_threads_by_handle() {
    let world = World::homebrew();
    assert_eq!(world.thread_id(world.main.handle), Ok(FIRST_THREAD_ID));
    assert_eq!(world.thread_id(Handle::CURRENT_THREAD), Ok(FIRST_THREAD_ID));
    for handle in [Handle::INVALID, Handle::CURRENT_PROCESS] {
        assert_eq!(
            world.thread_id(handle),
            Err(SvcError::Kernel(KernelError::InvalidHandle))
        );
    }
}

#[test]
fn a_call_from_a_thread_of_no_process_is_an_internal_failure() {
    let mut world = World::homebrew();
    // A scheduler-only thread, as the scheduler's own tests make them, running on core 2.
    let stray = world
        .scheduler()
        .create_thread(Priority::new(44).unwrap(), core(2))
        .unwrap();
    world.scheduler().start(stray).unwrap();
    world.schedule(core(2), Some(stray));
    let Tables {
        handles,
        scheduler,
        processes,
        ..
    } = &mut world.tables;
    assert_eq!(
        set_thread_priority(
            handles,
            scheduler,
            processes,
            core(2),
            Handle::CURRENT_THREAD,
            30
        ),
        Err(SvcError::Internal(format!(
            "thread {stray:?} belongs to no process"
        )))
    );
}
