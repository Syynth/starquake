//! The scheduler's half of the lifecycle: user threads, their references, ending them, and the
//! virtual core mask.

use starquake_cpu::GuestRegs;
use starquake_memory::PAGE_SIZE;
use starquake_trace::{GuestTicks, SVC_ARG_COUNT};

use super::*;
use crate::handle::ObjectRefs;
use crate::id::SlabId;
use crate::memory::{GUARD_PAGES, fixture};
use crate::process::{THREAD_LOCAL_REGION_SIZE, VIRTUAL_CORE_COUNT};
use crate::result::ResultCode;
use crate::sched::{CoreId, CoreSignals, FIRST_THREAD_ID};
use crate::sync::{Timeout, WaitBegun, WaitOrigin, Wake};

struct NoSignals;

impl CoreSignals for NoSignals {
    fn interrupt(&mut self, _: CoreId) {}

    fn wake(&mut self, _: CoreId) {}
}

/// The process every test's threads belong to.
fn process() -> ProcessId {
    ProcessId::from_parts(3, 1)
}

fn core(index: u8) -> CoreId {
    CoreId::new(index).unwrap()
}

/// A scheduler with [`process`]'s row, and the fixture's address space.
fn world() -> (Scheduler, AddressSpace) {
    let mut scheduler = Scheduler::default();
    scheduler.add_process(process()).unwrap();
    (scheduler, fixture::loaded().0)
}

/// A user thread on virtual core `virtual_core`, with distinct start values.
fn spec(virtual_core: u8) -> UserThread {
    UserThread {
        entry: 0x8000_1234,
        argument: 0xA5A5_0001,
        stack_top: 0x1_0000_8000,
        priority: Priority::new(44).unwrap(),
        virtual_core,
    }
}

/// Creates a user thread and leaves its object with one reference, the creator's.
fn create(scheduler: &mut Scheduler, space: &mut AddressSpace, core: u8) -> (ThreadId, ObjectId) {
    scheduler
        .create_user_thread(space, process(), &spec(core))
        .unwrap()
}

/// Makes `thread` the thread `on` runs.
fn run_on(scheduler: &mut Scheduler, thread: ThreadId, on: CoreId) {
    scheduler
        .update(GuestTicks(0), None, &mut NoSignals)
        .unwrap();
    assert_eq!(scheduler.schedule(on, &mut NoSignals), Ok(Some(thread)));
}

#[test]
fn a_user_thread_starts_where_its_creator_says() {
    let (mut scheduler, mut space) = world();
    let first_page = space.regions().kernel_map.start + GUARD_PAGES * PAGE_SIZE;
    let (thread, object) = create(&mut scheduler, &mut space, 2);
    let row = scheduler.thread(thread).unwrap();
    let expected = spec(2);
    assert_eq!(row.state(), ThreadState::Initialized);
    assert_eq!(row.process(), Some(process()));
    assert_eq!(row.object(), Some(object));
    assert_eq!(row.thread_id(), FIRST_THREAD_ID);
    assert_eq!(row.tls_address(), first_page);
    assert_eq!(row.priority(), expected.priority);
    assert_eq!(row.ideal_core(), Some(core(2)));
    assert_eq!(row.affinity().bits(), core(2).bit());
    assert_eq!(row.virtual_core_mask(), (2, 1 << 2));
    let context = row.context();
    assert_eq!(
        (context.pc, context.sp, context.x[0], context.tpidrro),
        (
            expected.entry,
            expected.stack_top,
            expected.argument,
            first_page
        )
    );
    // Divergence: Eden seeds x18 with a random odd number.
    assert_eq!(context.x[18], 0);
    assert!(context.x.iter().skip(1).all(|x| *x == 0));
    assert_eq!(scheduler.objects().refs(object), 1);
    let record = scheduler.process(process()).unwrap();
    assert_eq!(record.threads().collect::<Vec<_>>(), vec![thread]);
    assert_eq!(record.tls().used(), 1);
    // The next thread gets the next region and the next id.
    let (second, _) = create(&mut scheduler, &mut space, 2);
    let second = scheduler.thread(second).unwrap();
    assert_eq!(second.tls_address(), first_page + THREAD_LOCAL_REGION_SIZE);
    assert_eq!(second.thread_id(), FIRST_THREAD_ID + 1);
}

#[test]
fn a_virtual_core_runs_on_its_physical_core() {
    let (mut scheduler, mut space) = world();
    // Virtual 7 and 63 are the system core, 3.
    for virtual_core in [7, VIRTUAL_CORE_COUNT - 1] {
        let (thread, _) = create(&mut scheduler, &mut space, virtual_core);
        let row = scheduler.thread(thread).unwrap();
        assert_eq!(row.ideal_core(), Some(core(3)));
        assert_eq!(
            row.virtual_core_mask(),
            (i32::from(virtual_core), 1 << virtual_core)
        );
    }
    // A scheduler with two cores has no physical core 3: nothing is made, no region taken.
    let mut small = Scheduler::new(2).unwrap();
    small.add_process(process()).unwrap();
    assert_eq!(
        small.create_user_thread(&mut space, process(), &spec(3)),
        Err(SvcError::Kernel(KernelError::InvalidCoreId))
    );
    assert_eq!(small.process(process()).unwrap().tls().used(), 0);
}

#[test]
fn a_thread_of_an_unknown_process_is_not_made() {
    let (mut scheduler, mut space) = world();
    let stranger = ProcessId::from_parts(9, 9);
    assert_eq!(
        scheduler.create_user_thread(&mut space, stranger, &spec(0)),
        Err(SvcError::from(SchedError::UnknownProcess(stranger)))
    );
    assert_eq!(
        scheduler.add_process(process()),
        Err(SchedError::Corrupt),
        "a process is added once"
    );
}

#[test]
fn starting_counts_the_thread_and_it_holds_itself_until_it_ends() {
    let (mut scheduler, mut space) = world();
    let (thread, object) = create(&mut scheduler, &mut space, 0);
    scheduler.start(thread).unwrap();
    assert_eq!(scheduler.objects().refs(object), 2);
    assert_eq!(scheduler.process(process()).unwrap().running_threads(), 1);
    // The creator lets go: the thread's own reference keeps it.
    scheduler.close_object(object).unwrap();
    assert_eq!(scheduler.objects().refs(object), 1);
    // Ending it closes that reference too, which destroys it and frees its region.
    assert_eq!(
        scheduler.terminate_thread(thread),
        Ok(ThreadExit::LastOf(process()))
    );
    assert!(scheduler.thread(thread).is_err());
    let record = scheduler.process(process()).unwrap();
    assert_eq!(record.running_threads(), 0);
    assert_eq!(record.tls().used(), 0);
    assert_eq!(record.threads().count(), 0);
}

#[test]
fn an_ended_thread_with_an_open_handle_keeps_its_row_and_region() {
    let (mut scheduler, mut space) = world();
    let (thread, object) = create(&mut scheduler, &mut space, 0);
    let (other, _) = create(&mut scheduler, &mut space, 1);
    scheduler.start(thread).unwrap();
    scheduler.start(other).unwrap();
    // The creator's reference stands in for a handle here.
    assert_eq!(scheduler.terminate_thread(thread), Ok(ThreadExit::Others));
    let row = scheduler.thread(thread).unwrap();
    assert_eq!(row.state(), ThreadState::Terminated);
    assert_eq!(scheduler.objects().refs(object), 1);
    let record = scheduler.process(process()).unwrap();
    assert_eq!((record.running_threads(), record.tls().used()), (1, 2));
    // Ending it again changes nothing.
    assert_eq!(scheduler.terminate_thread(thread), Ok(ThreadExit::Others));
    assert_eq!(scheduler.process(process()).unwrap().running_threads(), 1);
    // The last reference goes: now it is destroyed.
    scheduler.close_object(object).unwrap();
    assert!(scheduler.thread(thread).is_err());
    assert_eq!(scheduler.process(process()).unwrap().tls().used(), 1);
}

#[test]
fn a_thread_that_never_started_ends_without_touching_the_running_count() {
    let (mut scheduler, mut space) = world();
    let (running, _) = create(&mut scheduler, &mut space, 0);
    scheduler.start(running).unwrap();
    let (idle, object) = create(&mut scheduler, &mut space, 0);
    assert_eq!(scheduler.terminate_thread(idle), Ok(ThreadExit::Others));
    assert_eq!(scheduler.process(process()).unwrap().running_threads(), 1);
    assert_eq!(scheduler.objects().refs(object), 1);
}

#[test]
fn losing_the_last_reference_while_running_is_an_internal_failure() {
    let (mut scheduler, mut space) = world();
    let (thread, object) = create(&mut scheduler, &mut space, 0);
    scheduler.start(thread).unwrap();
    scheduler.close_object(object).unwrap();
    // Only broken bookkeeping closes the thread's own reference behind its back.
    assert_eq!(
        scheduler.close_object(object),
        Err(SvcError::Internal(format!(
            "thread {thread:?} lost its last reference while it runs"
        )))
    );
}

#[test]
fn a_thread_destroyed_while_a_core_names_it_leaves_that_core() {
    let (mut scheduler, mut space) = world();
    let (thread, object) = create(&mut scheduler, &mut space, 0);
    scheduler.start(thread).unwrap();
    scheduler.close_object(object).unwrap();
    run_on(&mut scheduler, thread, core(0));
    // A neighbouring core keeps what it had.
    assert_eq!(scheduler.current(core(1)), Ok(None));
    assert_eq!(
        scheduler.exit_thread(core(0)),
        Ok(ThreadExit::LastOf(process()))
    );
    assert_eq!(scheduler.current(core(0)), Ok(None));
    assert_eq!(scheduler.highest(core(0)), Ok(None));
    assert_eq!(scheduler.needs_scheduling(core(0)), Ok(true));
    assert_eq!(scheduler.needs_scheduling(core(1)), Ok(true));
    // The next update and schedule find nothing stale.
    scheduler
        .update(GuestTicks(1), None, &mut NoSignals)
        .unwrap();
    assert_eq!(scheduler.schedule(core(0), &mut NoSignals), Ok(None));
}

#[test]
fn exiting_needs_a_thread_on_the_core() {
    let (mut scheduler, _) = world();
    assert_eq!(
        scheduler.exit_thread(core(4)),
        Err(SvcError::from(SchedError::NoCurrentThread { core: 4 }))
    );
}

#[test]
fn terminating_a_process_ends_each_of_its_threads() {
    let (mut scheduler, mut space) = world();
    let (started, _) = create(&mut scheduler, &mut space, 0);
    let (idle, _) = create(&mut scheduler, &mut space, 1);
    scheduler.start(started).unwrap();
    scheduler.terminate_threads(process()).unwrap();
    for thread in [started, idle] {
        assert_eq!(
            scheduler.thread(thread).unwrap().state(),
            ThreadState::Terminated
        );
    }
    assert_eq!(scheduler.process(process()).unwrap().running_threads(), 0);
}

#[test]
fn the_process_state_moves_from_created_to_running_once() {
    let (mut scheduler, _) = world();
    assert_eq!(
        scheduler.process(process()).unwrap().state(),
        ProcessState::Created
    );
    scheduler.mark_running(process()).unwrap();
    assert_eq!(
        scheduler.process(process()).unwrap().state(),
        ProcessState::Running
    );
    assert_eq!(
        scheduler.mark_running(process()),
        Err(SchedError::Kernel(KernelError::InvalidState))
    );
    scheduler.mark_terminated(process()).unwrap();
    assert_eq!(
        scheduler.process(process()).unwrap().state(),
        ProcessState::Terminated
    );
}

#[test]
fn the_virtual_mask_is_kept_and_its_translation_scheduled() {
    let (mut scheduler, mut space) = world();
    let (thread, _) = create(&mut scheduler, &mut space, 0);
    // Virtual 1 and 6: physical 1 and 3; the ideal core 6 is physical 3.
    let mask = (1 << 1) | (1 << 6);
    scheduler.set_virtual_core_mask(thread, 6, mask).unwrap();
    let row = scheduler.thread(thread).unwrap();
    assert_eq!(row.virtual_core_mask(), (6, mask));
    assert_eq!(row.ideal_core(), Some(core(3)));
    assert_eq!(row.affinity().bits(), core(1).bit() | core(3).bit());
    // No update keeps virtual 6, which must be in the new mask.
    assert_eq!(
        scheduler.set_virtual_core_mask(thread, IDEAL_CORE_NO_UPDATE, 1 << 1),
        Err(SchedError::Kernel(KernelError::InvalidCombination))
    );
    assert_eq!(
        scheduler.thread(thread).unwrap().virtual_core_mask(),
        (6, mask)
    );
    scheduler
        .set_virtual_core_mask(thread, IDEAL_CORE_NO_UPDATE, 1 << 6)
        .unwrap();
    assert_eq!(
        scheduler.thread(thread).unwrap().virtual_core_mask(),
        (6, 1 << 6)
    );
    // Don't care: no physical ideal core.
    scheduler
        .set_virtual_core_mask(thread, IDEAL_CORE_DONT_CARE, 1 << 2)
        .unwrap();
    let row = scheduler.thread(thread).unwrap();
    assert_eq!(row.virtual_core_mask(), (IDEAL_CORE_DONT_CARE, 1 << 2));
    assert_eq!(row.ideal_core(), None);
    // No update with a "don't care" ideal core tests bit 63 (the listed guess).
    assert_eq!(
        scheduler.set_virtual_core_mask(thread, IDEAL_CORE_NO_UPDATE, 1 << 2),
        Err(SchedError::Kernel(KernelError::InvalidCombination))
    );
    scheduler
        .set_virtual_core_mask(thread, IDEAL_CORE_NO_UPDATE, 1 << 63)
        .unwrap();
    // An ideal core below "don't care", other than "no update", is refused.
    assert_eq!(
        scheduler.set_virtual_core_mask(thread, IDEAL_CORE_NO_UPDATE - 1, 1),
        Err(SchedError::Kernel(KernelError::InvalidCoreId))
    );
}

#[test]
fn a_virtual_mask_this_scheduler_cannot_run_is_an_invalid_core() {
    let mut scheduler = Scheduler::new(2).unwrap();
    scheduler.add_process(process()).unwrap();
    let mut space = fixture::loaded().0;
    let (thread, _) = create(&mut scheduler, &mut space, 1);
    // Physical core 2 is the first this scheduler lacks; core 1 its last.
    assert_eq!(
        scheduler.set_virtual_core_mask(thread, 1, (1 << 1) | (1 << 2)),
        Err(SchedError::Kernel(KernelError::InvalidCoreId))
    );
    scheduler.set_virtual_core_mask(thread, 1, 1 << 1).unwrap();
    assert_eq!(
        scheduler.set_virtual_core_mask(thread, 2, 1 << 1),
        Err(SchedError::Kernel(KernelError::InvalidCoreId))
    );
}

#[test]
fn a_session_object_is_torn_down_by_the_ipc_table_and_a_process_object_by_nothing() {
    let (mut scheduler, _) = world();
    let object = scheduler
        .objects_mut()
        .create(Object::Process(process()))
        .unwrap();
    scheduler.objects_mut().open(object).unwrap();
    scheduler.close_object(object).unwrap();
    assert_eq!(scheduler.objects().refs(object), 1);
    scheduler.close_object(object).unwrap();
    assert_eq!(scheduler.objects().refs(object), 0);
    // The process's row outlives its object.
    assert!(scheduler.process(process()).is_ok());
}

/// Who a test wait is for, with distinct non-zero values.
fn origin() -> WaitOrigin {
    let mut args = [0; SVC_ARG_COUNT];
    args[1] = 0x5150;
    WaitOrigin {
        thread: starquake_trace::ThreadId(0x7001),
        core: 0,
        args,
    }
}

/// Has the thread core 0 runs wait on `objects` with no timeout, opening the reference each
/// object's wait holds, as `WaitSynchronization` does.
fn wait_on(scheduler: &mut Scheduler, objects: &[ObjectId]) -> WaitBegun {
    for object in objects {
        scheduler.objects_mut().open(*object).unwrap();
    }
    scheduler
        .wait_synchronization(
            core(0),
            objects,
            Timeout::Never,
            &GuestRegs::default(),
            origin(),
        )
        .unwrap()
}

#[test]
fn a_thread_id_that_cannot_be_given_frees_the_region_it_took() {
    let (mut scheduler, mut space) = world();
    create(&mut scheduler, &mut space, 0);
    scheduler.next_thread_id = u64::MAX;
    assert_eq!(
        scheduler.create_user_thread(&mut space, process(), &spec(0)),
        Err(SvcError::from(SchedError::Corrupt))
    );
    let record = scheduler.process(process()).unwrap();
    assert_eq!(record.tls().used(), 1);
    assert_eq!(record.threads().count(), 1);
}

#[test]
fn a_thread_that_terminates_wakes_its_waiters_and_the_last_wait_reference_destroys_it() {
    let (mut scheduler, mut space) = world();
    let (waiter, _) = create(&mut scheduler, &mut space, 0);
    let (joined, joined_object) = create(&mut scheduler, &mut space, 1);
    scheduler.start(waiter).unwrap();
    scheduler.start(joined).unwrap();
    run_on(&mut scheduler, waiter, core(0));
    assert_eq!(
        wait_on(&mut scheduler, &[joined_object]),
        WaitBegun::Waiting(waiter)
    );
    // The creator lets go: the joined thread's reference to itself and the wait's are left.
    scheduler.close_object(joined_object).unwrap();
    assert_eq!(scheduler.objects().refs(joined_object), 2);

    // The join: ending the thread signals it, and its waiter wakes with success, index 0.
    assert_eq!(scheduler.terminate_thread(joined), Ok(ThreadExit::Others));
    assert_eq!(
        scheduler.thread(waiter).unwrap().state(),
        ThreadState::Runnable
    );
    assert_eq!(scheduler.sync().wait(waiter), None);
    assert_eq!(
        scheduler.take_wakes(),
        vec![Wake {
            thread: waiter,
            origin: origin(),
            result: ResultCode::SUCCESS,
            index: 0,
        }]
    );
    let context = scheduler.thread(waiter).unwrap().context();
    assert_eq!(
        (context.x[0], context.x[1]),
        (u64::from(ResultCode::SUCCESS.raw()), 0)
    );
    // No reference is left, so the joined thread is gone and its region free.
    assert!(scheduler.thread(joined).is_err());
    assert_eq!(scheduler.process(process()).unwrap().tls().used(), 1);
}

#[test]
fn a_wait_that_holds_the_last_reference_to_a_thread_destroys_it_when_it_ends() {
    let (mut scheduler, mut space) = world();
    let (waiter, _) = create(&mut scheduler, &mut space, 0);
    // Never started, so it holds no reference to itself.
    let (idle, idle_object) = create(&mut scheduler, &mut space, 1);
    scheduler.start(waiter).unwrap();
    run_on(&mut scheduler, waiter, core(0));
    assert_eq!(
        wait_on(&mut scheduler, &[idle_object]),
        WaitBegun::Waiting(waiter)
    );
    scheduler.close_object(idle_object).unwrap();
    assert_eq!(scheduler.objects().refs(idle_object), 1);
    assert_eq!(scheduler.process(process()).unwrap().tls().used(), 2);
    // Cancelling the wait releases the last reference: the thread is destroyed through the one
    // teardown path, so its row and region go with the object.
    scheduler.cancel_synchronization(waiter).unwrap();
    assert!(scheduler.objects().get(idle_object).is_err());
    assert!(scheduler.thread(idle).is_err());
    let record = scheduler.process(process()).unwrap();
    assert_eq!(record.tls().used(), 1);
    assert_eq!(record.threads().collect::<Vec<_>>(), vec![waiter]);
}

#[test]
fn a_thread_that_terminates_while_waiting_leaves_every_wait_list() {
    let (mut scheduler, mut space) = world();
    let (waiter, _) = create(&mut scheduler, &mut space, 0);
    let (other, _) = create(&mut scheduler, &mut space, 1);
    scheduler.start(waiter).unwrap();
    scheduler.start(other).unwrap();
    run_on(&mut scheduler, waiter, core(0));
    let (writable, readable) = scheduler.create_event().unwrap();
    assert_eq!(
        wait_on(&mut scheduler, &[readable]),
        WaitBegun::Waiting(waiter)
    );
    assert_eq!(scheduler.objects().refs(readable), 2);

    assert_eq!(scheduler.terminate_thread(waiter), Ok(ThreadExit::Others));
    assert_eq!(scheduler.sync().wait(waiter), None);
    assert_eq!(scheduler.sync().waiters(readable).count(), 0);
    // The wait's reference is released; the creator's is left.
    assert_eq!(scheduler.objects().refs(readable), 1);
    // A later signal wakes nothing: the ended thread stays ended and nothing is reported.
    scheduler.signal_object(writable).unwrap();
    assert_eq!(
        scheduler.thread(waiter).unwrap().state(),
        ThreadState::Terminated
    );
    assert_eq!(scheduler.take_wakes(), vec![]);
    // The thread that was not waiting is untouched.
    assert_eq!(
        scheduler.thread(other).unwrap().state(),
        ThreadState::Runnable
    );
}

#[test]
fn a_pending_cancel_goes_with_its_thread() {
    let (mut scheduler, mut space) = world();
    let (ending, _) = create(&mut scheduler, &mut space, 0);
    let (staying, _) = create(&mut scheduler, &mut space, 1);
    scheduler.start(ending).unwrap();
    scheduler.start(staying).unwrap();
    scheduler.cancel_synchronization(ending).unwrap();
    scheduler.cancel_synchronization(staying).unwrap();
    assert!(scheduler.sync().is_cancel_pending(ending));
    scheduler.terminate_thread(ending).unwrap();
    assert!(!scheduler.sync().is_cancel_pending(ending));
    assert!(scheduler.sync().is_cancel_pending(staying));
}

#[test]
fn a_running_process_goes_back_to_created_once() {
    let (mut scheduler, _) = world();
    assert_eq!(
        scheduler.unmark_running(process()),
        Err(SchedError::from(KernelError::InvalidState))
    );
    scheduler.mark_running(process()).unwrap();
    scheduler.unmark_running(process()).unwrap();
    assert_eq!(
        scheduler.process(process()).unwrap().state(),
        ProcessState::Created
    );
    scheduler.mark_running(process()).unwrap();
    scheduler.mark_terminated(process()).unwrap();
    assert_eq!(
        scheduler.unmark_running(process()),
        Err(SchedError::from(KernelError::InvalidState))
    );
    let stranger = ProcessId::from_parts(9, 9);
    assert_eq!(
        scheduler.unmark_running(stranger),
        Err(SchedError::UnknownProcess(stranger))
    );
}
