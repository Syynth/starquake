//! The `GetInfo` arm's entropy, which it reads from what creating the process drew and never
//! draws itself, its Switch 2 ids, and the `Break` arm: which registers each reads and writes,
//! how each dispatch ends and what it records.

use super::*;
use crate::handle::Resolved;
use crate::process::ENTROPY_COUNT;
use crate::svc::exception::{Break, BreakReason, NOTIFICATION_ONLY_FLAG, RESULT_ARGUMENT_SIZE};
use crate::svc::info::{IDLE_TICK_COUNT, InfoType, MEMORY_MAPPING_ALIGNMENT};
use crate::tables::Processes;

/// The entropy words the replays below hand out: all different, and none a clock reading.
const DRAWN: [u64; ENTROPY_COUNT] = [
    0xa1a1_0000_0000_0001,
    0xb2b2_0000_0000_0002,
    0xc3c3_0000_0000_0003,
    0xd4d4_0000_0000_0004,
];

/// A clock record, as each dispatch reads one.
fn clock(n: u64) -> EnvRecord {
    EnvRecord::Clock(HostNanos(CLOCK_START_NS + n * CLOCK_STEP_NS))
}

/// The four random records of [`DRAWN`].
fn drawn() -> Vec<EnvRecord> {
    DRAWN.iter().map(|&word| EnvRecord::Random(word)).collect()
}

/// [`tables`], made as a run makes them: with the process's entropy drawn from a replay of
/// [`DRAWN`] ([`crate::process::create_with_env`]), which is used up exactly.
fn drawn_tables() -> Tables {
    let mut description = ProcessDescription::homebrew(CONSOLE);
    description.handle_table_size = i32::from(MAX_TABLE_SIZE);
    let env = ReplayEnv::new(drawn());
    let tables = crate::process::create_with_env(&description, fixture::loaded().0, &env)
        .expect("the fixture's process");
    assert_eq!(env.finish(), Ok(()));
    tables
}

/// Runs one call on `dispatcher` from core 0 with `env`, `memory`, and a recorder that keeps the
/// events.
fn dispatch_with(
    dispatcher: &Dispatcher,
    env: &dyn Env,
    memory: &FlatMemory,
    number: u16,
    regs: &mut GuestRegs,
) -> (Record, Vec<Event>) {
    dispatch_from(0, dispatcher, env, memory, number, regs)
}

/// [`dispatch_with`], from core `core`.
fn dispatch_from(
    core: u8,
    dispatcher: &Dispatcher,
    env: &dyn Env,
    memory: &FlatMemory,
    number: u16,
    regs: &mut GuestRegs,
) -> (Record, Vec<Event>) {
    let mut sink = Collect::default();
    let (recorder, events) = recorder();
    let mut caller = Caller {
        core,
        thread: THREAD,
        env,
        memory,
        debug: &mut sink,
        recorder: &recorder,
    };
    let record = dispatcher.dispatch(number, regs, &mut caller);
    (record, events.events())
}

/// Registers for `GetInfo(info_type, handle, sub_id)` over [`regs`], so every register the
/// call does not take holds something.
fn get_info_regs(info_type: InfoType, handle: Handle, sub_id: u64) -> GuestRegs {
    let mut regs = regs();
    regs.x[1] = u64::from(info_type.number());
    regs.x[2] = u64::from(handle.raw());
    regs.x[3] = sub_id;
    regs
}

/// The entropy the dispatcher's process has.
fn entropy_of(dispatcher: &Dispatcher) -> Option<[u64; ENTROPY_COUNT]> {
    dispatcher.with::<(&Scheduler,), _>(|(scheduler,)| {
        scheduler.process(own_process(scheduler)).unwrap().entropy()
    })
}

/// The one process the scheduler has a row for: the one [`tables`] created.
fn own_process(scheduler: &Scheduler) -> ProcessId {
    let description = ProcessDescription::homebrew(CONSOLE);
    let own = Processes::new(&description).unwrap().own();
    assert!(scheduler.process(own).is_ok());
    own
}

/// `GetInfo(RandomEntropy, handle, sub_id)`'s result and value, with `env`.
fn ask_entropy(dispatcher: &Dispatcher, env: &dyn Env, handle: Handle, sub_id: u64) -> [u64; 2] {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut regs = get_info_regs(InfoType::RandomEntropy, handle, sub_id);
    let before = regs.clone();
    let (record, _) = dispatch_with(dispatcher, env, &memory, SvcId::GetInfo.number(), &mut regs);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_untouched_from(&regs, &before, 2);
    [regs.x[0], regs.x[1]]
}

#[test]
fn random_entropy_reads_what_creation_drew_and_draws_nothing() {
    // Creating the process drew the entropy; each ask reads one word of it, and the
    // environment gives each dispatch its clock reading and nothing else, so a replay of only
    // those readings is used up exactly.
    let dispatcher = Dispatcher::new(drawn_tables());
    assert_eq!(entropy_of(&dispatcher), Some(DRAWN));
    let count = u64::try_from(ENTROPY_COUNT).unwrap();
    let env = ReplayEnv::new((0..count).map(clock).collect());
    for (sub_id, word) in (0_u64..).zip(DRAWN) {
        assert_eq!(
            ask_entropy(&dispatcher, &env, Handle::INVALID, sub_id),
            [0, word],
            "sub id {sub_id}"
        );
    }
    assert_eq!(env.finish(), Ok(()));
    assert_eq!(entropy_of(&dispatcher), Some(DRAWN));
}

#[test]
fn a_refused_entropy_ask_draws_nothing() {
    // The guest's checks refuse the call before the entropy is read, and nothing is drawn:
    // the replay holds only the dispatches' clock readings.
    let dispatcher = Dispatcher::new(drawn_tables());
    let env = ReplayEnv::new(vec![clock(0), clock(1)]);
    assert_eq!(
        ask_entropy(&dispatcher, &env, Handle::CURRENT_PROCESS, 0),
        [word(KernelError::InvalidHandle), 0]
    );
    assert_eq!(
        ask_entropy(
            &dispatcher,
            &env,
            Handle::INVALID,
            u64::try_from(ENTROPY_COUNT).unwrap()
        ),
        [word(KernelError::InvalidCombination), 0]
    );
    assert_eq!(env.finish(), Ok(()));
}

#[test]
fn a_process_with_no_scheduler_row_draws_nothing_and_is_refused_by_the_call() {
    // The call reports the guest's checks before it looks for the process's row: here the
    // handle is not the invalid handle. No random number is drawn, so a replay holding only the dispatch's clock reading is used up exactly.
    let mut tables = tables();
    tables.scheduler = Scheduler::default();
    let dispatcher = Dispatcher::new(tables);
    let env = ReplayEnv::new(vec![clock(0)]);
    assert_eq!(
        ask_entropy(&dispatcher, &env, Handle::CURRENT_PROCESS, 0),
        [word(KernelError::InvalidHandle), 0]
    );
    assert_eq!(env.finish(), Ok(()));
}

#[test]
fn other_ids_draw_nothing() {
    // Each id, with the arguments it accepts, reads only the clock. DebuggerAttached's value
    // is zero; the memory id's comes from the address space.
    let dispatcher = Dispatcher::new(tables());
    let env = ReplayEnv::new(vec![clock(0), clock(1)]);
    let memory = FlatMemory::new(BASE, 0x10);
    for (n, (id, handle)) in [
        (InfoType::DebuggerAttached, Handle::INVALID),
        (InfoType::HeapRegionSize, Handle::CURRENT_PROCESS),
    ]
    .into_iter()
    .enumerate()
    {
        let mut regs = get_info_regs(id, handle, 0);
        let (record, events) = dispatch_with(
            &dispatcher,
            &env,
            &memory,
            SvcId::GetInfo.number(),
            &mut regs,
        );
        assert_eq!(record.outcome, Outcome::Completed, "{id:?}");
        assert_eq!(regs.x[0], 0, "{id:?}");
        if id == InfoType::DebuggerAttached {
            assert_eq!(regs.x[1], 0);
        }
        // The event carries the three inputs and success.
        let mut args = [0; SVC_ARG_COUNT];
        (args[1], args[2], args[3]) = (u64::from(id.number()), u64::from(handle.raw()), 0);
        assert_eq!(
            events,
            [svc_event(
                0,
                time_of(u64::try_from(n).unwrap()),
                0,
                SvcId::GetInfo.number(),
                args,
                ResultCode::SUCCESS
            )]
        );
    }
    assert_eq!(env.finish(), Ok(()));
    assert_eq!(entropy_of(&dispatcher), None);
}


#[test]
fn a_switch_2_id_is_answered_by_the_calling_process_s_console() {
    // The arm hands the call the process table, so the console of the calling process decides
    // the answer: a Switch 2 program is told the mapping alignment, a Switch 1 program that
    // the id is not one (`svc::info`). Neither draws a random number.
    let memory = FlatMemory::new(BASE, 0x10);
    let id = InfoType::MemoryMappingAlignment;
    for (console, expected) in [
        (Console::Switch2, [0, MEMORY_MAPPING_ALIGNMENT]),
        (Console::Switch1, [word(KernelError::InvalidEnumValue), 0]),
    ] {
        let dispatcher = Dispatcher::new(tables_at(console));
        let env = ReplayEnv::new(vec![clock(0)]);
        let mut regs = get_info_regs(id, Handle::CURRENT_PROCESS, 0);
        let before = regs.clone();
        let (record, _) = dispatch_with(
            &dispatcher,
            &env,
            &memory,
            SvcId::GetInfo.number(),
            &mut regs,
        );
        assert_eq!(record.outcome, Outcome::Completed, "{console:?}");
        assert_untouched_from(&regs, &before, 2);
        assert_eq!([regs.x[0], regs.x[1]], expected, "{console:?}");
        assert_eq!(env.finish(), Ok(()));
    }
}

#[test]
fn the_arm_passes_the_calling_core() {
    // `IdleTickCount` accepts the calling core's own index and no other: asked from core 3
    // with sub id 3 it answers, from core 0 it is refused.
    let memory = FlatMemory::new(BASE, 0x10);
    let dispatcher = Dispatcher::new(tables());
    let env = ReplayEnv::new(vec![clock(0), clock(1)]);
    let core = 3;
    for (from, expected) in [
        (core, [0, IDLE_TICK_COUNT]),
        (0, [word(KernelError::InvalidCombination), 0]),
    ] {
        let mut regs = get_info_regs(InfoType::IdleTickCount, Handle::INVALID, u64::from(core));
        let (record, _) = dispatch_from(
            from,
            &dispatcher,
            &env,
            &memory,
            SvcId::GetInfo.number(),
            &mut regs,
        );
        assert_eq!(record.outcome, Outcome::Completed, "core {from}");
        assert_eq!([regs.x[0], regs.x[1]], expected, "core {from}");
    }
    assert_eq!(env.finish(), Ok(()));
}

#[test]
fn the_current_process_id_adds_its_handle_to_the_process_s_table() {
    // The arm hands the call the handle table to write: the handle it returns names the
    // process's own object.
    let memory = FlatMemory::new(BASE, 0x10);
    let dispatcher = Dispatcher::new(tables());
    let env = ReplayEnv::new(vec![clock(0)]);
    let mut regs = get_info_regs(InfoType::MesosphereCurrentProcess, Handle::INVALID, 0);
    let (record, _) = dispatch_with(
        &dispatcher,
        &env,
        &memory,
        SvcId::GetInfo.number(),
        &mut regs,
    );
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(regs.x[0], 0);
    let added = Handle::from_raw(u32::try_from(regs.x[1]).unwrap());
    dispatcher.with::<(&HandleTable, &Scheduler), _>(|(handles, scheduler)| {
        let own = scheduler.process(own_process(scheduler)).unwrap().object();
        assert_eq!(handles.resolve(added), Ok(Resolved::Object(own)));
    });
    assert_eq!(env.finish(), Ok(()));
}

/// The registers for `Break(raw_reason, address, size)` over [`regs`].
fn break_regs(raw_reason: u64, address: u64, size: u64) -> GuestRegs {
    let mut regs = regs();
    (regs.x[0], regs.x[1], regs.x[2]) = (raw_reason, address, size);
    regs
}

/// The event a `Break(raw_reason, address, size)` is recorded as: the reason truncated to its
/// word, and success, the call having no `Result` output.
fn break_event(raw_reason: u32, address: u64, size: u64) -> Event {
    let mut args = [0; SVC_ARG_COUNT];
    (args[0], args[1], args[2]) = (u64::from(raw_reason), address, size);
    svc_event(
        0,
        time_of(0),
        0,
        SvcId::Break.number(),
        args,
        ResultCode::SUCCESS,
    )
}

/// The word the buffer holds in the `Break` tests.
const CODE: u32 = 0x0bad_c0de;

/// Memory at [`BASE`] whose first word is [`CODE`].
fn code_memory() -> FlatMemory {
    let memory = FlatMemory::new(BASE, 0x10);
    memory.write(BASE, &CODE.to_le_bytes()).unwrap();
    memory
}

#[test]
fn a_break_that_is_not_a_notification_ends_the_dispatch_in_broke() {
    // The reason's upper half is not part of its word: an `Assert` with junk above it.
    let raw = 0xdead_beef_0000_0000 | u64::from(BreakReason::Assert.raw());
    let dispatcher = Dispatcher::new(tables());
    let memory = code_memory();
    let mut regs = break_regs(raw, BASE, RESULT_ARGUMENT_SIZE);
    let before = regs.clone();
    let env = env();
    let (record, events) =
        dispatch_with(&dispatcher, &env, &memory, SvcId::Break.number(), &mut regs);
    assert_eq!(
        record.outcome,
        Outcome::Broke(Break {
            reason: BreakReason::Assert,
            notification_only: false,
            address: BASE,
            size: RESULT_ARGUMENT_SIZE,
            code: Some(CODE),
        })
    );
    // No register is written, x0 included.
    assert_untouched_from(&regs, &before, 0);
    assert_eq!(
        events,
        [break_event(
            BreakReason::Assert.raw(),
            BASE,
            RESULT_ARGUMENT_SIZE
        )]
    );
}

#[test]
fn a_notification_only_break_returns_and_writes_no_register() {
    // Each with the flag: a module notice, and a panic, which without the flag would stop the
    // program.
    for reason in [BreakReason::PreLoadDll, BreakReason::Panic] {
        let raw = reason.raw() | NOTIFICATION_ONLY_FLAG;
        let dispatcher = Dispatcher::new(tables());
        let memory = code_memory();
        let mut regs = break_regs(u64::from(raw), BASE, RESULT_ARGUMENT_SIZE);
        let before = regs.clone();
        let env = env();
        let (record, events) =
            dispatch_with(&dispatcher, &env, &memory, SvcId::Break.number(), &mut regs);
        assert_eq!(record.outcome, Outcome::Completed, "{reason:?}");
        assert_untouched_from(&regs, &before, 0);
        assert_eq!(
            events,
            [break_event(raw, BASE, RESULT_ARGUMENT_SIZE)],
            "{reason:?}"
        );
    }
    // The same panic without the flag stops the program.
    let dispatcher = Dispatcher::new(tables());
    let mut regs = break_regs(0, 0, 0);
    let (record, _) = dispatch_with(
        &dispatcher,
        &env(),
        &code_memory(),
        SvcId::Break.number(),
        &mut regs,
    );
    assert_eq!(record.outcome, Outcome::Broke(Break::decode(0, 0, 0)));
}
