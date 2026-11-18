//! Process creation and the main thread, driven against the fixture's address space.

use starquake_memory::{FlatMemory, PAGE_SIZE};
use starquake_trace::{EnvCall, EnvRecord, ReplayEnv};

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
    assert_eq!(tables.scheduler.objects_besides_processes(), 1);
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

#[test]
fn the_main_thread_takes_the_first_region_because_no_process_local_region_is_reserved() {
    // Divergence: Eden reserves the process-local region first, so its main thread's region is
    // the page's second (page + THREAD_LOCAL_REGION_SIZE) and two regions are in use.
    let mut tables = create(&description(), fixture::loaded().0).unwrap();
    let memory = tls_memory(&tables.address_space);
    let main = run(&mut tables, &memory, ENTRY, STACK_TOP).unwrap();
    let thread = tables.scheduler.thread(main.thread).unwrap();
    assert_eq!(thread.tls_address(), first_tls_page(&tables.address_space));
    let record = tables.scheduler.process(tables.processes.own()).unwrap();
    assert_eq!(record.tls().used(), 1);
}

#[test]
fn a_main_thread_that_fails_to_start_is_undone() {
    // The two states `run` can fail in after the thread and its handle exist: before the thread
    // is started (it holds no reference to itself) and after (it does).
    for started in [false, true] {
        let mut tables = create(&description(), fixture::loaded().0).unwrap();
        let memory = tls_memory(&tables.address_space);
        let own = tables.processes.own();
        let spec = UserThread {
            entry: ENTRY,
            argument: 0,
            stack_top: STACK_TOP,
            priority: crate::sched::Priority::new(description().main_thread_priority).unwrap(),
            virtual_core: description().main_thread_core,
        };
        let Tables {
            handles,
            address_space,
            scheduler,
            ..
        } = &mut tables;
        let (thread, handle) = create_thread_with_handle(
            (handles, address_space, scheduler),
            own,
            (&spec, 0),
            &memory,
        )
        .unwrap();
        scheduler.mark_running(own).unwrap();
        if started {
            scheduler.start(thread).unwrap();
        }
        undo_run(handles, scheduler, own, MainThread { thread, handle }).unwrap();

        assert!(tables.handles.is_empty(), "started: {started}");
        assert!(
            tables.scheduler.thread(thread).is_err(),
            "started: {started}"
        );
        assert_eq!(
            tables.scheduler.objects_besides_processes(),
            0,
            "started: {started}"
        );
        let record = tables.scheduler.process(own).unwrap();
        assert_eq!(record.state(), ProcessState::Created, "started: {started}");
        assert_eq!(record.threads().count(), 0, "started: {started}");
        assert_eq!(record.running_threads(), 0, "started: {started}");
        assert_eq!(record.tls().used(), 0, "started: {started}");
        // Nothing is left over: the process runs as a fresh one would.
        let main = run(&mut tables, &memory, ENTRY, STACK_TOP).unwrap();
        assert_eq!(
            tables.scheduler.thread(main.thread).unwrap().tls_address(),
            first_tls_page(&tables.address_space)
        );
    }
}

#[test]
fn a_region_whose_end_does_not_fit_is_refused_at_the_exact_edge() {
    let memory = FlatMemory::new(0x1000, 0x10);
    let handle = Handle::from_register(0x8001);
    // The last address whose region end (exclusive) still fits: the write is tried, and this
    // memory refuses it.
    let last = u64::MAX - THREAD_LOCAL_REGION_SIZE;
    assert!(matches!(
        initialise_region(&memory, last, handle),
        Err(RegionError::Refused { tls, .. }) if tls == last
    ));
    assert_eq!(
        initialise_region(&memory, last + 1, handle),
        Err(RegionError::TopOfMemory { tls: last + 1 })
    );
}

#[test]
fn a_space_made_for_another_system_resource_is_refused() {
    use crate::memory::system_resource::SECURE_ALIGNMENT;
    use crate::memory::testing::RecordingMapper;
    use crate::memory::{MemoryArrangement, Pool, SystemResource};

    let made_with = |resource| {
        let (mapper, _log) = RecordingMapper::new();
        let memory = fixture::memory(MemoryArrangement::FourGb);
        AddressSpace::with_system_resource(&memory, resource, Box::new(mapper)).unwrap()
    };
    let size = 2 * SECURE_ALIGNMENT;
    let mut described = description();
    // A few bytes past whole pages, which the resource leaves out.
    described.system_resource_size = u32::try_from(size + 5).unwrap();
    let resource = SystemResource::new(u32::try_from(size).unwrap(), Pool::Application);
    assert_eq!(described.system_resource(), resource);
    assert!(create(&described, made_with(resource)).is_ok());
    // No resource, another size, another pool: each refused, naming both.
    let others = [
        SystemResource::NONE,
        SystemResource::new(
            u32::try_from(3 * SECURE_ALIGNMENT).unwrap(),
            Pool::Application,
        ),
        SystemResource::new(u32::try_from(size).unwrap(), Pool::Applet),
    ];
    for other in others {
        assert_eq!(
            create(&described, made_with(other)).map(|_| ()),
            Err(CreateError::SystemResource {
                described: size,
                described_pool: Pool::Application,
                made: other.size(),
                made_pool: other.pool(),
            }),
            "{other:?}"
        );
    }
    assert_eq!(
        create(&described, made_with(SystemResource::NONE))
            .map(|_| ())
            .unwrap_err()
            .to_string(),
        "the process's address space was made for a system resource of 0x0 bytes in the \
         application pool, not the 0x40000 bytes in the application pool its description asks \
         for"
    );
    // The description's own checks come first, and the handle table after.
    let mut both = described.clone();
    both.cores = Range { first: 1, last: 0 };
    assert_eq!(
        create(&both, made_with(SystemResource::NONE)).map(|_| ()),
        Err(CreateError::Process(ProcessError::Capability(
            KernelError::InvalidCombination
        )))
    );
    let mut handles = described;
    handles.handle_table_size = i32::from(MAX_TABLE_SIZE) + 1;
    assert!(matches!(
        create(&handles, made_with(SystemResource::NONE)),
        Err(CreateError::SystemResource { .. })
    ));
    assert!(matches!(
        create(&handles, made_with(resource)),
        Err(CreateError::Handles(_))
    ));
}

// --- What a process draws from the environment ----------------------------------------------

/// The random numbers the replays below hand out, in order: the four entropy words, then the
/// main thread's `x18`. All differ, and the last is even, so the bit `x18` gets set shows.
const DRAWN: [u64; ENTROPY_COUNT + 1] = [
    0x1111_0000_0000_00a1,
    0x2222_0000_0000_00b2,
    0x3333_0000_0000_00c3,
    0x4444_0000_0000_00d4,
    0x5555_0000_0000_00e6,
];

/// The first `n` of [`DRAWN`], as a replay's records.
fn drawn(n: usize) -> Vec<EnvRecord> {
    DRAWN
        .iter()
        .take(n)
        .map(|&word| EnvRecord::Random(word))
        .collect()
}

/// The entropy words of [`DRAWN`].
fn entropy_words() -> [u64; ENTROPY_COUNT] {
    let mut words = [0; ENTROPY_COUNT];
    words.copy_from_slice(&DRAWN[..ENTROPY_COUNT]);
    words
}

#[test]
fn a_process_made_without_an_environment_has_no_entropy_and_a_zero_x18() {
    let mut tables = create(&description(), fixture::loaded().0).unwrap();
    let own = tables.processes.own();
    assert_eq!(tables.scheduler.process(own).unwrap().entropy(), None);
    let memory = tls_memory(&tables.address_space);
    let main = run(&mut tables, &memory, ENTRY, STACK_TOP).unwrap();
    let context = tables.scheduler.thread(main.thread).unwrap().context();
    assert_eq!(context.x[CFI_REGISTER], 0);
}

#[test]
fn the_entropy_and_x18_are_drawn_once_at_creation_in_order() {
    // One replay for both steps, as a run boots: creation takes the first four numbers as the
    // entropy, in order, and the start the fifth as the main thread's x18, with its low bit set.
    let env = ReplayEnv::new(drawn(DRAWN.len()));
    let mut tables = create_with_env(&description(), fixture::loaded().0, &env).unwrap();
    let own = tables.processes.own();
    assert_eq!(env.consumed(), ENTROPY_COUNT);
    assert_eq!(
        tables.scheduler.process(own).unwrap().entropy(),
        Some(entropy_words())
    );
    let memory = tls_memory(&tables.address_space);
    let main = run_with_env(&mut tables, &memory, (ENTRY, STACK_TOP), &env).unwrap();
    // Exactly the five numbers were drawn.
    assert_eq!(env.finish(), Ok(()));
    let context = tables.scheduler.thread(main.thread).unwrap().context();
    assert_eq!(
        context.x[CFI_REGISTER],
        DRAWN[ENTROPY_COUNT] | CFI_VALUE_SET_BITS
    );
    assert_eq!(context.x[CFI_REGISTER], 0x5555_0000_0000_00e7);
    // The rest of the start state is `run`'s: x0 zero, x1 the handle, every other register
    // zero.
    assert_eq!(context.x[0], 0);
    assert_eq!(context.x[1], u64::from(main.handle.raw()));
    for (n, value) in context.x.iter().enumerate().skip(2) {
        if n != CFI_REGISTER {
            assert_eq!(*value, 0, "x{n}");
        }
    }
    assert_eq!((context.pc, context.sp), (ENTRY, STACK_TOP));
    // The entropy did not change when the thread started.
    assert_eq!(
        tables.scheduler.process(own).unwrap().entropy(),
        Some(entropy_words())
    );
}

#[test]
fn an_x18_draw_that_is_already_odd_keeps_its_value() {
    // `| 1` sets the bit; it does not add one.
    let env = ReplayEnv::new(vec![EnvRecord::Random(0x0f0f_0000_0000_0001)]);
    let mut tables = create(&description(), fixture::loaded().0).unwrap();
    let memory = tls_memory(&tables.address_space);
    let main = run_with_env(&mut tables, &memory, (ENTRY, STACK_TOP), &env).unwrap();
    let context = tables.scheduler.thread(main.thread).unwrap().context();
    assert_eq!(context.x[CFI_REGISTER], 0x0f0f_0000_0000_0001);
}

#[test]
fn a_replay_missing_an_entropy_word_fails_creation_with_the_exact_error() {
    // Each count short of the four: the draw that finds nothing is the record after the last
    // one given.
    for given in 0..ENTROPY_COUNT {
        let env = ReplayEnv::new(drawn(given));
        let created = create_with_env(&description(), fixture::loaded().0, &env).map(|_| ());
        let error = EnvError::Exhausted {
            index: given,
            requested: EnvCall::Random,
        };
        assert_eq!(created, Err(CreateError::Env(error.clone())), "{given}");
        assert_eq!(
            CreateError::Env(error).to_string(),
            format!(
                "the environment failed while the process's entropy was drawn: recording \
                 exhausted at record {given}: the replay asked for Random"
            )
        );
    }
}

#[test]
fn a_refused_description_draws_nothing() {
    // The description's checks come first: with a replay that has nothing to give, the error
    // is the description's, not the environment's.
    let mut cores = description();
    cores.cores = Range { first: 1, last: 0 };
    let env = ReplayEnv::new(Vec::new());
    assert_eq!(
        create_with_env(&cores, fixture::loaded().0, &env).map(|_| ()),
        Err(CreateError::Process(ProcessError::Capability(
            KernelError::InvalidCombination
        )))
    );
    // A space made for another system resource: refused before the draw too.
    let mut resource = description();
    resource.system_resource_size =
        u32::try_from(2 * crate::memory::system_resource::SECURE_ALIGNMENT).unwrap();
    assert!(matches!(
        create_with_env(&resource, fixture::loaded().0, &env).map(|_| ()),
        Err(CreateError::SystemResource { .. })
    ));
    assert_eq!(env.consumed(), 0);
}

#[test]
fn a_replay_missing_the_x18_fails_the_start_with_nothing_made_and_the_next_start_succeeds() {
    let mut tables = create(&description(), fixture::loaded().0).unwrap();
    let own = tables.processes.own();
    let memory = tls_memory(&tables.address_space);
    let before = tables.scheduler.clone();
    let empty = ReplayEnv::new(Vec::new());
    let error = EnvError::Exhausted {
        index: 0,
        requested: EnvCall::Random,
    };
    assert_eq!(
        run_with_env(&mut tables, &memory, (ENTRY, STACK_TOP), &empty),
        Err(StartError::Env(error.clone()))
    );
    assert_eq!(
        StartError::Env(error).to_string(),
        "the environment failed while the main thread's x18 was drawn: recording exhausted at \
         record 0: the replay asked for Random"
    );
    assert_eq!(tables.scheduler, before);
    assert!(tables.handles.is_empty());
    assert_eq!(
        tables.scheduler.process(own).unwrap().state(),
        ProcessState::Created
    );
    let env = ReplayEnv::new(drawn(1));
    let main = run_with_env(&mut tables, &memory, (ENTRY, STACK_TOP), &env).unwrap();
    assert_eq!(
        tables.scheduler.thread(main.thread).unwrap().context().x[CFI_REGISTER],
        DRAWN[0] | CFI_VALUE_SET_BITS
    );
}

#[test]
fn a_process_that_is_not_created_is_refused_before_the_draw() {
    // Both faults at once: the process already runs, and the replay has nothing to give. The
    // state check comes first, so the error is Eden's InvalidState and nothing is drawn.
    let mut tables = create(&description(), fixture::loaded().0).unwrap();
    let memory = tls_memory(&tables.address_space);
    run(&mut tables, &memory, ENTRY, STACK_TOP).unwrap();
    let empty = ReplayEnv::new(Vec::new());
    assert_eq!(
        run_with_env(&mut tables, &memory, (ENTRY, STACK_TOP), &empty),
        Err(StartError::Call(SvcError::Kernel(
            KernelError::InvalidState
        )))
    );
    assert_eq!(empty.consumed(), 0);
}

#[test]
fn a_start_refused_after_the_draw_has_still_drawn_one_number() {
    // Divergence (module docs): at its thread limit the start is refused with Eden's
    // LimitReached, after the draw; Eden's draw comes after the reservation, so it would have
    // drawn nothing.
    let mut tables = create(&description(), fixture::loaded().0).unwrap();
    let own = tables.processes.own();
    let spec = UserThread {
        entry: ENTRY,
        argument: 0,
        stack_top: STACK_TOP,
        priority: crate::sched::Priority::new(44).unwrap(),
        virtual_core: 0,
    };
    let limit = crate::resource_limit::THREAD_COUNT_LIMIT;
    for _ in 0..limit {
        tables
            .scheduler
            .create_user_thread(&mut tables.address_space, own, &spec)
            .unwrap();
    }
    let memory = tls_memory(&tables.address_space);
    let env = ReplayEnv::new(drawn(1));
    assert_eq!(
        run_with_env(&mut tables, &memory, (ENTRY, STACK_TOP), &env),
        Err(StartError::Call(SvcError::Kernel(
            KernelError::LimitReached
        )))
    );
    assert_eq!(env.finish(), Ok(()));
}
