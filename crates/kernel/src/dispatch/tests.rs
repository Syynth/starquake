//! The dispatcher driven with a fake register file, environment, memory and debug sink.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_memory::FlatMemory;
use std::sync::Arc;

use starquake_trace::{
    EnvCall, Event, EventKind, FixedClock, GuestTicks, LiveEnv, Recorder, ReplayEnv, ThreadId,
    VecSink,
};

use super::*;
use crate::handle::MAX_TABLE_SIZE;
use crate::svc::tick::{CNTFREQ, NANOS_PER_SECOND};
use crate::svc::{Arg, Kind};

/// Above 4 GiB, so an address truncated to 32 bits would miss it.
const BASE: u64 = 0x80_0000_0000;
/// The first clock reading, in nanoseconds (what the `Env` reports): a whole number of seconds,
/// so the expected guest ticks are `CLOCK_SECONDS * CNTFREQ` with no rounding.
const CLOCK_SECONDS: u64 = 1_000_000_007;
const CLOCK_START_NS: u64 = CLOCK_SECONDS * NANOS_PER_SECOND;
/// What `GetSystemTick` returns for the first reading.
const FIRST_TICKS: u64 = CLOCK_SECONDS * CNTFREQ;

/// The thread and time every test call is stamped with: distinct from each other, from the core
/// numbers used and from zero, so a swapped or dropped field shows in the recorded event.
const THREAD: ThreadId = ThreadId(0x7EAD);
const TIME: GuestTicks = GuestTicks(0x71C5);

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
const PROPTEST_SEED: u64 = 0x4449_5350_4154_4348;

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 256,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(PROPTEST_SEED),
        failure_persistence: None,
        ..Config::default()
    }
}

/// A live environment whose clock reads `CLOCK_START_NS` first, then one nanosecond more.
fn env() -> LiveEnv {
    LiveEnv::new(FixedClock::new(CLOCK_START_NS, 1), 0)
}

/// Fresh placeholder tables. The one place the tests build them.
fn tables() -> Tables {
    Tables {
        handles: HandleTable::new(i32::from(MAX_TABLE_SIZE)).expect("a maximum-size handle table"),
        address_space: AddressSpace::default(),
        scheduler: Scheduler::default(),
        processes: Processes::default(),
    }
}

#[derive(Default)]
struct Collect(Vec<Vec<u8>>);

impl DebugSink for Collect {
    fn output(&mut self, message: &[u8]) {
        self.0.push(message.to_vec());
    }
}

/// Registers that all differ, so a stray write or a read from the wrong register shows up.
fn regs() -> GuestRegs {
    let mut regs = GuestRegs::default();
    for (i, x) in (0u64..).zip(regs.x.iter_mut()) {
        *x = 0x1111_1111_1111_1111_u64.wrapping_mul(i + 1);
    }
    regs
}

fn not_implemented_word() -> u64 {
    u64::from(KernelError::NotImplemented.code().raw())
}

/// Runs one call on core `core` against fresh tables and the given memory.
fn dispatch_one(
    number: u16,
    regs: &mut GuestRegs,
    core: u8,
    memory: &FlatMemory,
    sink: &mut Collect,
) -> Record {
    dispatch_recorded(number, regs, core, memory, sink, &Recorder::new())
}

/// As [`dispatch_one`], recording into `recorder`.
fn dispatch_recorded(
    number: u16,
    regs: &mut GuestRegs,
    core: u8,
    memory: &FlatMemory,
    sink: &mut Collect,
    recorder: &Recorder,
) -> Record {
    let dispatcher = Dispatcher::new(tables());
    let env = env();
    let mut caller = Caller {
        core,
        thread: THREAD,
        time: TIME,
        env: &env,
        memory,
        debug: sink,
        recorder,
    };
    dispatcher.dispatch(number, regs, &mut caller)
}

/// A recorder with one sink that keeps every event.
fn recorder() -> (Recorder, Arc<VecSink>) {
    let recorder = Recorder::new();
    let sink = Arc::new(VecSink::new());
    recorder.add_sink(Arc::clone(&sink));
    (recorder, sink)
}

/// The event a call with these arguments and this result must have been recorded as.
fn svc_event(seq: u64, core: u8, number: u16, args: &GuestRegs, result: ResultCode) -> Event {
    let mut passed = [0; SVC_ARG_COUNT];
    passed.copy_from_slice(&args.x[..SVC_ARG_COUNT]);
    Event {
        seq,
        time: TIME,
        core,
        thread: THREAD,
        kind: EventKind::Svc {
            id: u32::from(number),
            args: passed,
            result: starquake_trace::ResultCode(result.raw()),
        },
    }
}

fn assert_untouched_from(regs: &GuestRegs, before: &GuestRegs, first: usize) {
    assert_eq!(regs.x[first..], before.x[first..]);
    assert_eq!(
        (regs.sp, regs.pc, regs.pstate, regs.tpidr, regs.tpidrro),
        (
            before.sp,
            before.pc,
            before.pstate,
            before.tpidr,
            before.tpidrro
        )
    );
}

#[test]
fn get_system_tick_writes_the_environments_counter_to_x0() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let mut regs = regs();
    let before = regs.clone();
    let record = dispatch_one(
        SvcId::GetSystemTick.number(),
        &mut regs,
        0,
        &memory,
        &mut sink,
    );
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(record.id, Some(SvcId::GetSystemTick));
    // The clock reports nanoseconds; the guest gets counter ticks (CNTFREQ), not the raw reading.
    // A full 64-bit value (int64_t), not truncated to a 32-bit word.
    assert_ne!(FIRST_TICKS >> 32, 0);
    assert_eq!(regs.x[0], FIRST_TICKS);
    assert_untouched_from(&regs, &before, 1);
}

#[test]
fn get_current_processor_number_writes_the_core_zero_extended() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    for core in [0, 3, 5] {
        let mut regs = regs();
        regs.x[0] = u64::MAX;
        let before = regs.clone();
        let record = dispatch_one(
            SvcId::GetCurrentProcessorNumber.number(),
            &mut regs,
            core,
            &memory,
            &mut sink,
        );
        assert_eq!(record.outcome, Outcome::Completed);
        assert_eq!(record.core, core);
        assert_eq!(regs.x[0], u64::from(core));
        assert_untouched_from(&regs, &before, 1);
    }
}

#[test]
fn output_debug_string_decodes_x0_and_x1_and_writes_success() {
    let memory = FlatMemory::new(BASE, 0x100);
    memory.write(BASE + 0x10, b"hello from the guest").unwrap();
    let mut sink = Collect::default();
    let mut regs = regs();
    regs.x[0] = BASE + 0x10;
    regs.x[1] = 5;
    let before = regs.clone();
    let record = dispatch_one(
        SvcId::OutputDebugString.number(),
        &mut regs,
        0,
        &memory,
        &mut sink,
    );
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(sink.0, [b"hello".to_vec()]);
    assert_eq!(regs.x[0], u64::from(ResultCode::SUCCESS.raw()));
    // Only x0 is an output; x1 still holds the length.
    assert_untouched_from(&regs, &before, 1);
}

#[test]
fn output_debug_string_reports_its_error_in_x0() {
    let memory = FlatMemory::new(BASE, 0x100);
    let mut sink = Collect::default();
    let mut regs = regs();
    regs.x[0] = BASE + 0xff;
    regs.x[1] = 2; // one byte past the end of mapped memory
    let record = dispatch_one(
        SvcId::OutputDebugString.number(),
        &mut regs,
        0,
        &memory,
        &mut sink,
    );
    // The call ran; its error result is an output like any other.
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(
        regs.x[0],
        u64::from(KernelError::InvalidCurrentMemory.code().raw())
    );
    assert_eq!(sink.0, Vec::<Vec<u8>>::new());
}

#[test]
fn unknown_numbers_return_not_implemented_and_touch_only_x0() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let unlisted: Vec<u16> = (0..=u16::MAX)
        .filter(|&n| SvcId::from_number(n).is_none())
        .collect();
    // 0 (below the first call), the gap after GetResourceLimitPeakValue, the first number past
    // the last call, and the top.
    let gap = SvcId::GetResourceLimitPeakValue.number() + 1;
    let past_last = SvcId::ALL.last().unwrap().number() + 1;
    for number in [0, gap, past_last, u16::MAX] {
        assert!(unlisted.contains(&number), "{number:#x} is listed");
        let mut regs = regs();
        let before = regs.clone();
        let record = dispatch_one(number, &mut regs, 1, &memory, &mut sink);
        let mut args = [0; SVC_ARG_COUNT];
        args.copy_from_slice(&before.x[..SVC_ARG_COUNT]);
        assert_eq!(
            record,
            Record {
                core: 1,
                number,
                id: None,
                args,
                outcome: Outcome::NotImplemented,
            }
        );
        assert_eq!(regs.x[0], not_implemented_word());
        assert_untouched_from(&regs, &before, 1);
    }
}

#[test]
fn every_unlisted_number_is_handled_without_panicking() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let dispatcher = Dispatcher::new(tables());
    let env = env();
    let recorder = Recorder::new();
    let mut caller = Caller {
        core: 0,
        thread: THREAD,
        time: TIME,
        env: &env,
        memory: &memory,
        debug: &mut sink,
        recorder: &recorder,
    };
    for number in (0..=u16::MAX).filter(|&n| SvcId::from_number(n).is_none()) {
        let mut regs = regs();
        let record = dispatcher.dispatch(number, &mut regs, &mut caller);
        assert_eq!(record.outcome, Outcome::NotImplemented, "{number:#x}");
        assert_eq!(regs.x[0], not_implemented_word(), "{number:#x}");
    }
}

proptest! {
    #![proptest_config(config())]

    /// Any svc number with any register values, against an `Env`, memory and sink that don't
    /// panic themselves, completes the dispatch and returns a record for that number.
    #[test]
    fn any_number_and_registers_dispatch_without_panicking(
        number in any::<u16>(),
        core in any::<u8>(),
        x in prop::array::uniform8(any::<u64>()),
    ) {
        let memory = FlatMemory::new(BASE, 0x10);
        let mut sink = Collect::default();
        let mut regs = GuestRegs::default();
        regs.x[..x.len()].copy_from_slice(&x);
        let record = dispatch_one(number, &mut regs, core, &memory, &mut sink);
        prop_assert_eq!(record.number, number);
        prop_assert_eq!(record.core, core);
        prop_assert_eq!(record.args, x);
    }
}

#[test]
fn a_deferred_call_returns_not_implemented_and_leaves_its_outputs_alone() {
    // SetHeapSize's convention outputs x0 (result) and x1 (address). Divergence from Eden's
    // wrapper, which would write x1 too: only x0 changes.
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    for id in [
        SvcId::SetHeapSize,
        SvcId::SetGpuMemoryAttribute,
        SvcId::ExitThread,
    ] {
        assert_eq!(id.status(), Status::Deferred);
        let mut regs = regs();
        let before = regs.clone();
        let record = dispatch_one(id.number(), &mut regs, 0, &memory, &mut sink);
        assert_eq!(record.outcome, Outcome::NotImplemented, "{}", id.name());
        assert_eq!(record.id, Some(id));
        assert_eq!(regs.x[0], not_implemented_word(), "{}", id.name());
        assert_untouched_from(&regs, &before, 1);
    }
}

#[test]
fn a_not_planned_call_returns_not_implemented() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let id = SvcId::CallSecureMonitor;
    assert_eq!(id.status(), Status::NotPlanned);
    let mut regs = regs();
    let record = dispatch_one(id.number(), &mut regs, 0, &memory, &mut sink);
    assert_eq!(record.outcome, Outcome::NotImplemented);
    assert_eq!(regs.x[0], not_implemented_word());
}

#[test]
fn the_dispatcher_runs_exactly_the_calls_marked_implemented() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    for &id in SvcId::ALL {
        // Zeroed registers: OutputDebugString then has length 0 and succeeds.
        let mut regs = GuestRegs::default();
        let record = dispatch_one(id.number(), &mut regs, 0, &memory, &mut sink);
        let expected = match (id.status(), id) {
            (Status::Implemented, SvcId::ExitProcess) => Outcome::ProcessExited,
            (Status::Implemented, _) => Outcome::Completed,
            _ => Outcome::NotImplemented,
        };
        assert_eq!(
            record.outcome,
            expected,
            "{} ({:#x})",
            id.name(),
            id.number()
        );
    }
}

#[test]
fn one_dispatcher_serves_calls_in_turn_and_records_the_arguments_as_passed() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let dispatcher = Dispatcher::new(tables());
    let env = env();
    let (recorder, events) = recorder();
    let mut caller = Caller {
        core: 2,
        thread: THREAD,
        time: TIME,
        env: &env,
        memory: &memory,
        debug: &mut sink,
        recorder: &recorder,
    };
    for number in [
        SvcId::GetSystemTick.number(),
        u16::MAX,
        SvcId::SetHeapSize.number(),
        SvcId::GetSystemTick.number(),
    ] {
        let mut regs = regs();
        let before = regs.clone();
        let record = dispatcher.dispatch(number, &mut regs, &mut caller);
        // The record holds x0-x7 from before the call, not the outputs written over them.
        assert_eq!(record.args[..], before.x[..SVC_ARG_COUNT]);
        assert_ne!(regs.x[0], before.x[0]);
        assert_eq!((record.core, record.number), (2, number));
    }
    // One event per call, numbered in call order.
    let seqs: Vec<u64> = events.events().iter().map(|event| event.seq).collect();
    assert_eq!(seqs, [0, 1, 2, 3]);
    // The clock was read once per GetSystemTick, in order, and nothing else touched it: stamping
    // an event does not read the environment's clock.
    assert_eq!(
        env.recording(),
        [
            starquake_trace::EnvRecord::Clock(starquake_trace::GuestTicks(CLOCK_START_NS)),
            starquake_trace::EnvRecord::Clock(starquake_trace::GuestTicks(CLOCK_START_NS + 1)),
        ]
    );
    let _tables: Tables = dispatcher.into_tables();
}

#[test]
fn a_failing_environment_ends_the_call_without_writing_registers() {
    // A replay with no clock reading left: Env::now fails, so GetSystemTick has nothing to return.
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let dispatcher = Dispatcher::new(tables());
    let env = ReplayEnv::new(Vec::new());
    let (recorder, events) = recorder();
    let mut caller = Caller {
        core: 0,
        thread: THREAD,
        time: TIME,
        env: &env,
        memory: &memory,
        debug: &mut sink,
        recorder: &recorder,
    };
    let mut regs = regs();
    let before = regs.clone();
    let record = dispatcher.dispatch(SvcId::GetSystemTick.number(), &mut regs, &mut caller);
    assert_eq!(
        record.outcome,
        Outcome::EnvFailed(EnvError::Exhausted {
            index: 0,
            requested: EnvCall::Clock
        })
    );
    assert_untouched_from(&regs, &before, 0);
    // The call never returned to the guest, so it has no result and no event.
    assert_eq!(events.events(), []);
}

#[test]
fn exit_process_ends_the_process_and_writes_no_register() {
    // Eden's `SvcWrap_ExitProcess64` takes no argument and writes no output.
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let mut regs = regs();
    let before = regs.clone();
    let record = dispatch_one(
        SvcId::ExitProcess.number(),
        &mut regs,
        4,
        &memory,
        &mut sink,
    );
    assert_eq!(record.outcome, Outcome::ProcessExited);
    assert_eq!(record.id, Some(SvcId::ExitProcess));
    assert_untouched_from(&regs, &before, 0);
}

#[test]
fn every_call_that_returns_is_recorded_once_with_its_arguments_and_result() {
    let memory = FlatMemory::new(BASE, 0x100);
    memory.write(BASE, b"line").unwrap();
    let not_implemented = KernelError::NotImplemented.code();
    // (number, x0, x1, the result the event must carry)
    let calls = [
        // A `Result` output: success, then an error (one byte past the end of mapped memory).
        (
            SvcId::OutputDebugString.number(),
            BASE,
            4,
            ResultCode::SUCCESS,
        ),
        (
            SvcId::OutputDebugString.number(),
            BASE + 0xff,
            2,
            KernelError::InvalidCurrentMemory.code(),
        ),
        // Outputs that are not a `Result` (ticks, a core number) are not recorded as one.
        (SvcId::GetSystemTick.number(), 1, 2, ResultCode::SUCCESS),
        (
            SvcId::GetCurrentProcessorNumber.number(),
            3,
            4,
            ResultCode::SUCCESS,
        ),
        // A void call.
        (SvcId::ExitProcess.number(), 5, 6, ResultCode::SUCCESS),
        // A deferred call and an unlisted number.
        (SvcId::SetHeapSize.number(), 7, 8, not_implemented),
        (u16::MAX, 9, 10, not_implemented),
    ];
    for (number, x0, x1, result) in calls {
        let (recorder, events) = recorder();
        let mut sink = Collect::default();
        let mut regs = regs();
        regs.x[0] = x0;
        regs.x[1] = x1;
        let before = regs.clone();
        dispatch_recorded(number, &mut regs, 3, &memory, &mut sink, &recorder);
        assert_eq!(
            events.events(),
            [svc_event(0, 3, number, &before, result)],
            "svc {number:#x}"
        );
    }
}

#[test]
fn a_call_that_disagrees_with_its_convention_returns_not_implemented() {
    static ONE_OUTPUT: &[Arg] = &[Arg {
        name: "ret",
        reg: 0,
        kind: Kind::Result,
    }];
    let convention = Convention {
        inputs: &[],
        outputs: ONE_OUTPUT,
    };
    let mut regs = regs();
    let before = regs.clone();
    assert_eq!(
        complete(SvcId::GetSystemTick, &convention, &mut regs, &[1, 2]),
        Outcome::NotImplemented
    );
    assert_eq!(regs.x[0], not_implemented_word());
    assert_untouched_from(&regs, &before, 1);
}

#[test]
fn the_dispatcher_can_be_shared_between_core_threads() {
    fn shared<T: Send + Sync>() {}
    shared::<Dispatcher>();
}
