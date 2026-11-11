//! The dispatcher driven with a fake register file, environment, memory and debug sink.

use starquake_memory::FlatMemory;

use super::*;
use crate::svc::{Arg, Kind};

/// Above 4 GiB, so an address truncated to 32 bits would miss it.
const BASE: u64 = 0x80_0000_0000;
const TICK: u64 = 0xfedc_ba98_7654_3210;

struct FakeEnv;

impl Env for FakeEnv {
    fn system_tick(&self) -> u64 {
        TICK
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
    let dispatcher = Dispatcher::new(Tables::default());
    let mut caller = Caller {
        core,
        env: &FakeEnv,
        memory,
        debug: sink,
    };
    dispatcher.dispatch(number, regs, &mut caller)
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
    // A full 64-bit value (int64_t), not truncated to a 32-bit word.
    assert_eq!(regs.x[0], TICK);
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
        assert_eq!(
            record,
            Record {
                seq: 0,
                core: 1,
                number,
                id: None,
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
    let dispatcher = Dispatcher::new(Tables::default());
    let mut caller = Caller {
        core: 0,
        env: &FakeEnv,
        memory: &memory,
        debug: &mut sink,
    };
    for number in (0..=u16::MAX).filter(|&n| SvcId::from_number(n).is_none()) {
        let mut regs = regs();
        let record = dispatcher.dispatch(number, &mut regs, &mut caller);
        assert_eq!(record.outcome, Outcome::NotImplemented, "{number:#x}");
        assert_eq!(regs.x[0], not_implemented_word(), "{number:#x}");
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
        let expected = if id.status() == Status::Implemented {
            Outcome::Completed
        } else {
            Outcome::NotImplemented
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
fn sequence_numbers_count_every_dispatch_in_order() {
    let memory = FlatMemory::new(BASE, 0x10);
    let mut sink = Collect::default();
    let dispatcher = Dispatcher::new(Tables::default());
    let mut caller = Caller {
        core: 2,
        env: &FakeEnv,
        memory: &memory,
        debug: &mut sink,
    };
    let numbers = [
        SvcId::GetSystemTick.number(),
        u16::MAX,
        SvcId::SetHeapSize.number(),
        SvcId::GetCurrentProcessorNumber.number(),
    ];
    let seqs: Vec<u64> = numbers
        .iter()
        .map(|&n| dispatcher.dispatch(n, &mut regs(), &mut caller).seq)
        .collect();
    assert_eq!(seqs, [0, 1, 2, 3]);
    let _tables: Tables = dispatcher.into_tables();
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
