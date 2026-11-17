//! The `GetInfo` arm's entropy, drawn from the environment at the first ask, and the `Break`
//! arm: which registers each reads and writes, how each dispatch ends and what it records.

use super::*;
use crate::process::ENTROPY_COUNT;
use crate::svc::exception::{Break, BreakReason, NOTIFICATION_ONLY_FLAG, RESULT_ARGUMENT_SIZE};
use crate::svc::info::InfoType;
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

/// Runs one call on `dispatcher` from core 0 with `env`, `memory`, and a recorder that keeps the
/// events.
fn dispatch_with(
    dispatcher: &Dispatcher,
    env: &dyn Env,
    memory: &FlatMemory,
    number: u16,
    regs: &mut GuestRegs,
) -> (Record, Vec<Event>) {
    let mut sink = Collect::default();
    let (recorder, events) = recorder();
    let mut caller = Caller {
        core: 0,
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

/// `GetInfo(RandomEntropy, invalid handle, sub_id)`'s result and value, with `env`.
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
fn the_entropy_is_drawn_once_at_the_first_ask_and_kept() {
    let dispatcher = Dispatcher::new(tables());
    assert_eq!(entropy_of(&dispatcher), None);
    // One clock reading per dispatch, and the four random numbers right after the first.
    let mut records = vec![clock(0)];
    records.extend(drawn());
    records.extend((1..u64::try_from(ENTROPY_COUNT).unwrap()).map(clock));
    let env = ReplayEnv::new(records);
    for (sub_id, word) in (0_u64..).zip(DRAWN) {
        assert_eq!(
            ask_entropy(&dispatcher, &env, Handle::INVALID, sub_id),
            [0, word],
            "sub id {sub_id}"
        );
    }
    // Exactly the records given: the three later asks drew nothing.
    assert_eq!(env.finish(), Ok(()));
    assert_eq!(entropy_of(&dispatcher), Some(DRAWN));
}

#[test]
fn a_first_ask_that_the_checks_refuse_still_draws_the_entropy() {
    // Divergence (`svc::info`): Horizon draws the entropy when the process is created, so a
    // refused call finds it there already. Here the first ask draws it, whatever its
    // arguments, and the call is then refused as Eden refuses it.
    let dispatcher = Dispatcher::new(tables());
    let mut records = vec![clock(0)];
    records.extend(drawn());
    records.push(clock(1));
    let env = ReplayEnv::new(records);
    assert_eq!(
        ask_entropy(&dispatcher, &env, Handle::CURRENT_PROCESS, 0),
        [word(KernelError::InvalidHandle), 0]
    );
    assert_eq!(entropy_of(&dispatcher), Some(DRAWN));
    assert_eq!(
        ask_entropy(&dispatcher, &env, Handle::INVALID, 3),
        [0, DRAWN[3]]
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
fn an_environment_that_fails_while_drawing_ends_the_call_without_writing_registers() {
    // The fourth random number is missing: the call does not run, writes nothing and records
    // nothing, and the process keeps no entropy, so the next ask draws all four again.
    let dispatcher = Dispatcher::new(tables());
    let memory = FlatMemory::new(BASE, 0x10);
    let mut records = vec![clock(0)];
    records.extend(drawn().into_iter().take(ENTROPY_COUNT - 1));
    let short = ReplayEnv::new(records);
    let mut regs = get_info_regs(InfoType::RandomEntropy, Handle::INVALID, 0);
    let before = regs.clone();
    let (record, events) = dispatch_with(
        &dispatcher,
        &short,
        &memory,
        SvcId::GetInfo.number(),
        &mut regs,
    );
    assert_eq!(
        record.outcome,
        Outcome::EnvFailed(EnvError::Exhausted {
            index: ENTROPY_COUNT,
            requested: EnvCall::Random
        })
    );
    assert_untouched_from(&regs, &before, 0);
    assert_eq!(events, []);
    assert_eq!(entropy_of(&dispatcher), None);

    let mut records = vec![clock(1)];
    records.extend(drawn());
    let env = ReplayEnv::new(records);
    assert_eq!(
        ask_entropy(&dispatcher, &env, Handle::INVALID, 1),
        [0, DRAWN[1]]
    );
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
