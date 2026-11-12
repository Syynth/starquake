//! Tests for the macro memo, with the interpreter as the oracle: every memoised call is checked
//! against a fresh interpreter run on a copy of the same state.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use super::*;
use crate::mme::testing::{
    Host, alu, branch, finish, inst0, mov, mov_reg, op3, r, store, with_imm, with_out,
};
use crate::mme::{AluOp, LOAD_BARRIER, OutOp, Reg, run};

/// A method the test programs write; any method works, the memo does not interpret them.
const METHOD: u16 = 0x45;

/// `R<dst> = data RAM [index]`.
fn dread(dst: u8, index: u16) -> Instruction {
    with_imm(inst0(AluOp::Dread, r(dst), Reg::Imm, Reg::Zero), index, 0)
}

/// Writes `R<src>` to `method` (method set from immediate 1, value from ALU 0).
fn emit(method: u16, src: u8) -> Instruction {
    with_imm(
        with_out(
            inst0(AluOp::Add, Reg::Zero, r(src), Reg::Zero),
            0,
            OutOp::Imm1,
            OutOp::Alu0,
        ),
        0,
        method,
    )
}

/// `R<dst> = register [method]`.
fn state(dst: u8, method: u16) -> Instruction {
    with_imm(inst0(AluOp::State, r(dst), Reg::Imm, Reg::Zero), method, 0)
}

/// The load barrier.
fn barrier() -> Instruction {
    let x = u16::try_from(LOAD_BARRIER).unwrap();
    with_imm(inst0(AluOp::Extended, Reg::Zero, Reg::Imm, Reg::Zero), x, 0)
}

/// `R1 = param; R2 = ram[0]; R3 = R1 + R2; ram[1] = R3; write R3 and R1`: a pure call that reads
/// data RAM word 0 and writes word 1.
fn pure_program() -> Vec<Instruction> {
    finish(vec![
        mov_reg(1, Reg::Load0),
        dread(2, 0),
        op3(AluOp::Add, 3, 1, 2),
        store(1, 3),
        emit(METHOD, 3),
        emit(METHOD + 1, 1),
    ])
}

/// The words of `pure_program`'s record with `parameters` parameters: one read, one data RAM
/// write and two method writes.
fn pure_record_words(parameters: usize) -> usize {
    HEADER_WORDS + parameters + 2 * (1 + 1 + 2)
}

/// The interpreter's result for one call, on copies of the state.
fn oracle(
    program: &[Instruction],
    parameters: &[u32],
    ram: &DataRam,
) -> (Result<(), MacroError>, DataRam, Vec<(u16, u32)>) {
    let mut ram = ram.clone();
    let mut host = Host::with_params(parameters);
    let result = run(program, &mut ram, &mut host, StepLimit::DEFAULT);
    (result, ram, host.calls)
}

/// Calls through the memo, checks the effects against [`oracle`], and returns the outcome.
fn checked_call(
    memo: &mut MacroMemo,
    program: &[Instruction],
    start: u32,
    parameters: &[u32],
    ram: &mut DataRam,
) -> Result<CallOutcome, MacroError> {
    let (expected, expected_ram, expected_calls) = oracle(program, parameters, ram);
    let mut host = Host::default();
    let outcome = memo.call(program, start, parameters, ram, &mut host);
    assert_eq!(outcome.map(|_| ()), expected);
    assert_eq!(*ram, expected_ram);
    assert_eq!(host.calls, expected_calls);
    outcome
}

#[test]
fn a_repeated_call_replays_the_interpreters_exact_result() {
    let program = pure_program();
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    *ram.get_mut(0).unwrap() = 0x10;

    assert_eq!(
        checked_call(&mut memo, &program, 7, &[5], &mut ram),
        Ok(CallOutcome::Stored)
    );
    assert_eq!(ram.get(1), Some(0x15));
    assert_eq!(
        checked_call(&mut memo, &program, 7, &[5], &mut ram),
        Ok(CallOutcome::Replayed)
    );
    assert_eq!(memo.len(), 1);
    let stats = memo.statistics();
    assert_eq!(
        (stats.hits, stats.slot_hits, stats.misses, stats.stores),
        (1, 1, 1, 1)
    );
}

#[test]
fn a_replay_writes_methods_in_recorded_order_and_does_not_interpret() {
    let program = pure_program();
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    let mut host = Host::default();
    memo.call(&program, 7, &[5], &mut ram, &mut host).unwrap();
    // A replay must not touch the host's load FIFO or registers.
    let mut host = Host::default();
    assert_eq!(
        memo.call(&program, 7, &[5], &mut ram, &mut host),
        Ok(CallOutcome::Replayed)
    );
    assert_eq!(host.calls, vec![(METHOD, 5), (METHOD + 1, 5)]);
    assert_eq!(host.loads, 0);
    assert_eq!(host.state_reads, []);
}

#[test]
fn changing_a_data_ram_word_the_call_read_fails_the_guard() {
    let program = pure_program();
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    checked_call(&mut memo, &program, 7, &[5], &mut ram).unwrap();
    *ram.get_mut(0).unwrap() = 0x20;
    assert_eq!(
        checked_call(&mut memo, &program, 7, &[5], &mut ram),
        Ok(CallOutcome::Stored)
    );
    assert_eq!(ram.get(1), Some(0x25));
    let stats = memo.statistics();
    assert_eq!(
        (
            stats.hits,
            stats.guard_failures,
            stats.stores,
            stats.overwrites
        ),
        (0, 1, 2, 1)
    );
    // The rewritten record holds the new guard.
    assert_eq!(
        checked_call(&mut memo, &program, 7, &[5], &mut ram),
        Ok(CallOutcome::Replayed)
    );
    assert_eq!(memo.len(), 1);
    assert_eq!(memo.arena_words(), pure_record_words(1));
}

#[test]
fn changing_a_data_ram_word_the_call_did_not_read_still_hits() {
    let program = pure_program();
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    checked_call(&mut memo, &program, 7, &[5], &mut ram).unwrap();
    // Word 1 is written by the call, not read; word 2 is untouched by it.
    *ram.get_mut(1).unwrap() = 0x99;
    *ram.get_mut(2).unwrap() = 0x77;
    assert_eq!(
        checked_call(&mut memo, &program, 7, &[5], &mut ram),
        Ok(CallOutcome::Replayed)
    );
    assert_eq!(ram.get(1), Some(5));
    assert_eq!(ram.get(2), Some(0x77));
}

#[test]
fn a_new_code_generation_forgets_every_call_and_the_same_one_keeps_them() {
    let program = pure_program();
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    checked_call(&mut memo, &program, 7, &[5], &mut ram).unwrap();

    memo.synchronize(0);
    assert_eq!(memo.len(), 1);
    assert_eq!(
        checked_call(&mut memo, &program, 7, &[5], &mut ram),
        Ok(CallOutcome::Replayed)
    );

    memo.synchronize(1);
    assert!(memo.is_empty());
    assert_eq!(memo.arena_words(), 0);
    // New code at the same start: the old record must not replay.
    let new_code = finish(vec![mov_reg(1, Reg::Load0), emit(METHOD + 2, 1)]);
    assert_eq!(
        checked_call(&mut memo, &new_code, 7, &[5], &mut ram),
        Ok(CallOutcome::Stored)
    );
    assert_eq!(memo.statistics().misses, 2);
}

#[test]
fn every_parameter_and_the_start_are_part_of_the_key() {
    // The program consumes only the first parameter, but a different unread one is another call.
    let program = pure_program();
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    checked_call(&mut memo, &program, 7, &[5, 1], &mut ram).unwrap();
    for (start, parameters) in [
        (7, &[5, 2][..]),
        (7, &[5][..]),
        (8, &[5, 1][..]),
        (7, &[6, 1][..]),
    ] {
        assert_eq!(
            checked_call(&mut memo, &program, start, parameters, &mut ram),
            Ok(CallOutcome::Stored),
            "start {start}, parameters {parameters:?}"
        );
    }
    assert_eq!(memo.len(), 5);
    assert_eq!(
        checked_call(&mut memo, &program, 7, &[5, 1], &mut ram),
        Ok(CallOutcome::Replayed)
    );
}

#[test]
fn a_hit_through_the_hash_moves_the_starts_slot() {
    // Starts 7 and 7 + 256 share a slot; alternating them finds each through the hash.
    let program = pure_program();
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    let other = 7 + u32::try_from(SLOTS).unwrap();
    checked_call(&mut memo, &program, 7, &[5], &mut ram).unwrap();
    checked_call(&mut memo, &program, other, &[5], &mut ram).unwrap();
    assert_eq!(
        checked_call(&mut memo, &program, 7, &[5], &mut ram),
        Ok(CallOutcome::Replayed)
    );
    assert_eq!(
        (memo.statistics().hits, memo.statistics().slot_hits),
        (1, 0)
    );
    // The hash hit put start 7 back in the slot.
    checked_call(&mut memo, &program, 7, &[5], &mut ram).unwrap();
    assert_eq!(
        (memo.statistics().hits, memo.statistics().slot_hits),
        (2, 1)
    );
}

#[test]
fn a_register_read_disqualifies_a_call() {
    let program = finish(vec![state(1, 0x20), emit(METHOD, 1)]);
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    for _ in 0..2 {
        assert_eq!(
            checked_call(&mut memo, &program, 3, &[], &mut ram),
            Ok(CallOutcome::NotStored(NotMemoised::RegisterRead))
        );
    }
    assert!(memo.is_empty());
}

#[test]
fn a_barrier_disqualifies_a_call_even_without_a_dma_read() {
    // Divergence: Eden-NX2 remembers this call, since it wrote no MME DMA read method.
    let program = finish(vec![barrier(), mov(1, 4), emit(METHOD, 1)]);
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    assert_eq!(
        checked_call(&mut memo, &program, 3, &[], &mut ram),
        Ok(CallOutcome::NotStored(NotMemoised::Barrier))
    );
    assert!(memo.is_empty());
}

#[test]
fn a_load_past_the_parameters_reaches_the_host_and_disqualifies() {
    let program = finish(vec![
        mov_reg(1, Reg::Load0),
        mov_reg(2, Reg::Load0),
        emit(METHOD, 2),
    ]);
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    let mut host = Host::with_params(&[0x33]);
    assert_eq!(
        memo.call(&program, 3, &[0x11], &mut ram, &mut host),
        Ok(CallOutcome::NotStored(NotMemoised::LoadPastParameters))
    );
    assert_eq!(host.calls, vec![(METHOD, 0x33)]);
    assert_eq!(host.loads, 1);
    // With both words as parameters the host is not asked, and the call is remembered.
    let mut host = Host::default();
    assert_eq!(
        memo.call(&program, 3, &[0x11, 0x33], &mut ram, &mut host),
        Ok(CallOutcome::Stored)
    );
    assert_eq!(host.loads, 0);
    assert_eq!(host.calls, vec![(METHOD, 0x33)]);
}

#[test]
fn reading_a_word_after_writing_it_disqualifies_but_writing_after_reading_does_not() {
    let reread = finish(vec![
        mov_reg(1, Reg::Load0),
        store(2, 1),
        dread(3, 2),
        emit(METHOD, 3),
    ]);
    let read_then_write = finish(vec![
        mov_reg(1, Reg::Load0),
        dread(3, 2),
        store(2, 1),
        emit(METHOD, 3),
    ]);
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    assert_eq!(
        checked_call(&mut memo, &reread, 3, &[9], &mut ram),
        Ok(CallOutcome::NotStored(NotMemoised::DataRamReread))
    );
    // A later call starts with nothing written.
    assert_eq!(
        checked_call(&mut memo, &read_then_write, 4, &[9], &mut ram),
        Ok(CallOutcome::Stored)
    );
}

#[test]
fn the_first_disqualifier_the_run_reaches_is_reported() {
    let state_first = finish(vec![state(1, 0x20), barrier(), emit(METHOD, 1)]);
    let barrier_first = finish(vec![barrier(), state(1, 0x20), emit(METHOD, 1)]);
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    assert_eq!(
        checked_call(&mut memo, &state_first, 3, &[], &mut ram),
        Ok(CallOutcome::NotStored(NotMemoised::RegisterRead))
    );
    assert_eq!(
        checked_call(&mut memo, &barrier_first, 3, &[], &mut ram),
        Ok(CallOutcome::NotStored(NotMemoised::Barrier))
    );
}

#[test]
fn a_failed_run_returns_the_interpreters_error_and_is_not_remembered() {
    let program = pure_program();
    let mut memo = MacroMemo::new();
    let mut ram = DataRam::new();
    assert_eq!(
        checked_call(&mut memo, &program, 7, &[], &mut ram),
        Err(MacroError::ParameterUnderflow { pc: 0 })
    );
    assert!(memo.is_empty());
    assert_eq!(memo.statistics().stores, 0);
}

#[test]
fn the_memo_interprets_with_its_step_limit() {
    let program = pure_program();
    let mut memo = MacroMemo::with_limits(MAX_ARENA_WORDS, StepLimit::new(3));
    let mut ram = DataRam::new();
    let mut host = Host::default();
    assert_eq!(
        memo.call(&program, 7, &[5], &mut ram, &mut host),
        Err(MacroError::StepLimitExceeded { limit: 3 })
    );
}

#[test]
fn a_record_longer_than_the_arena_is_not_stored() {
    // Divergence: Eden-NX2 stores it and lets the arena grow.
    let program = pure_program();
    let words = pure_record_words(1);
    let mut memo = MacroMemo::with_limits(words, StepLimit::DEFAULT);
    let mut ram = DataRam::new();
    assert_eq!(
        checked_call(&mut memo, &program, 7, &[5], &mut ram),
        Ok(CallOutcome::Stored)
    );
    assert_eq!(memo.arena_words(), words);

    let mut memo = MacroMemo::with_limits(words - 1, StepLimit::DEFAULT);
    assert_eq!(
        checked_call(&mut memo, &program, 7, &[5], &mut ram),
        Ok(CallOutcome::NotStored(NotMemoised::TooLarge))
    );
    assert!(memo.is_empty());
    assert_eq!(memo.statistics().oversized, 1);
    assert_eq!(memo.statistics().stores, 0);
}

#[test]
fn an_arena_limit_above_the_maximum_is_reduced_to_it() {
    assert_eq!(
        MacroMemo::with_limits(MAX_ARENA_WORDS + 1, StepLimit::DEFAULT).max_arena_words,
        MAX_ARENA_WORDS
    );
    assert_eq!(
        MacroMemo::with_limits(MAX_ARENA_WORDS, StepLimit::DEFAULT).max_arena_words,
        MAX_ARENA_WORDS
    );
}

#[test]
fn a_full_arena_of_live_records_is_compacted_then_cleared() {
    let program = finish(vec![mov_reg(1, Reg::Load0), emit(METHOD, 1)]);
    let record = HEADER_WORDS + 1 + 2;
    let mut memo = MacroMemo::with_limits(4 * record, StepLimit::DEFAULT);
    let mut ram = DataRam::new();
    for parameter in 0..4 {
        checked_call(&mut memo, &program, 1, &[parameter], &mut ram).unwrap();
    }
    assert_eq!(memo.arena_words(), 4 * record);
    assert_eq!(memo.statistics().compactions, 0);
    // The arena is full: the next store compacts first. All four records are live, which is
    // more than three quarters of it, so the compaction empties the memo.
    checked_call(&mut memo, &program, 1, &[4], &mut ram).unwrap();
    let stats = memo.statistics();
    assert_eq!((stats.compactions, stats.clears), (1, 1));
    assert_eq!(memo.len(), 1);
    assert_eq!(memo.arena_words(), record);
}

/// A program whose data RAM read decides how many methods it writes:
/// `R2 = ram[0]; write R2; if R2 != 0 write it again`.
fn guarded_shape_program() -> Vec<Instruction> {
    finish(vec![
        dread(2, 0),
        emit(METHOD, 2),
        branch(AluOp::Beq, r(2), Reg::Zero, true, 2),
        emit(METHOD + 1, 2),
    ])
}

#[test]
fn a_guard_failure_with_a_new_shape_appends_and_compaction_reclaims_the_old_record() {
    let program = guarded_shape_program();
    // One read, then one or two writes.
    let short = HEADER_WORDS + 2 * (1 + 1);
    let long = HEADER_WORDS + 2 * (1 + 2);
    let mut memo = MacroMemo::with_limits(short + long, StepLimit::DEFAULT);
    let mut ram = DataRam::new();

    assert_eq!(
        checked_call(&mut memo, &program, 1, &[], &mut ram),
        Ok(CallOutcome::Stored)
    );
    *ram.get_mut(0).unwrap() = 3;
    // The guard fails and the shape differs: a new record is appended, the old one is stale.
    assert_eq!(
        checked_call(&mut memo, &program, 1, &[], &mut ram),
        Ok(CallOutcome::Stored)
    );
    assert_eq!(memo.arena_words(), short + long);
    assert_eq!(memo.len(), 1);
    assert_eq!(memo.statistics().overwrites, 0);

    // A second macro's store finds the arena full and compacts: only the long record is live,
    // below three quarters of the arena, so nothing is cleared.
    let other = finish(vec![mov(1, 1), emit(METHOD, 1)]);
    let other_words = HEADER_WORDS + 2;
    assert_eq!(
        checked_call(&mut memo, &other, 2, &[], &mut ram),
        Ok(CallOutcome::Stored)
    );
    let stats = memo.statistics();
    assert_eq!((stats.compactions, stats.clears), (1, 0));
    assert_eq!(memo.arena_words(), long + other_words);
    assert_eq!(memo.len(), 2);
    // Both survivors replay from their new offsets.
    assert_eq!(
        checked_call(&mut memo, &program, 1, &[], &mut ram),
        Ok(CallOutcome::Replayed)
    );
    assert_eq!(
        checked_call(&mut memo, &other, 2, &[], &mut ram),
        Ok(CallOutcome::Replayed)
    );
}

// ---------------------------------------------------------------------------------------------
// The interpreter as oracle, on generated programs
// ---------------------------------------------------------------------------------------------

/// One generated operation on registers R1 to R4 and data RAM words 0 to 7.
#[derive(Clone, Copy, Debug)]
enum Op {
    Mov(u8, u16),
    Alu(AluOp, u8, u8, u8),
    Load(u8),
    Read(u8, u16),
    Write(u16, u8),
    Emit(u16, u8),
    State(u8, u16),
    Barrier,
}

impl Op {
    fn instruction(self) -> Instruction {
        match self {
            Self::Mov(dst, value) => mov(dst, value),
            Self::Alu(op, dst, a, b) => {
                let mut inst = Instruction::default();
                inst.alu[0] = alu(op, r(dst), r(a), r(b));
                inst
            }
            Self::Load(dst) => mov_reg(dst, Reg::Load0),
            Self::Read(dst, index) => dread(dst, index),
            Self::Write(index, src) => store(index, src),
            Self::Emit(method, src) => emit(method, src),
            Self::State(dst, method) => state(dst, method),
            Self::Barrier => barrier(),
        }
    }
}

fn reg() -> impl Strategy<Value = u8> {
    1..=4_u8
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        1 => (reg(), any::<u16>()).prop_map(|(d, v)| Op::Mov(d, v)),
        1 => (
            prop::sample::select(vec![AluOp::Add, AluOp::Sub, AluOp::Xor, AluOp::Or]),
            reg(),
            reg(),
            reg()
        )
            .prop_map(|(op, d, a, b)| Op::Alu(op, d, a, b)),
        3 => reg().prop_map(Op::Load),
        3 => (reg(), 0..8_u16).prop_map(|(d, i)| Op::Read(d, i)),
        3 => (0..8_u16, reg()).prop_map(|(i, s)| Op::Write(i, s)),
        3 => (0x40..0x48_u16, reg()).prop_map(|(m, s)| Op::Emit(m, s)),
        1 => (reg(), 0x40..0x48_u16).prop_map(|(d, m)| Op::State(d, m)),
        1 => Just(Op::Barrier),
    ]
}

/// A generated program: R1 to R4 initialised, then the operations, then Mesa's ending.
fn program() -> impl Strategy<Value = Vec<Instruction>> {
    (
        prop::array::uniform4(any::<u16>()),
        prop::collection::vec(op(), 0..12),
    )
        .prop_map(|(init, ops)| {
            let mut program: Vec<Instruction> = (1..=4)
                .zip(init)
                .map(|(dst, value)| mov(dst, value))
                .collect();
            program.extend(ops.into_iter().map(Op::instruction));
            finish(program)
        })
}

/// Calls: which of two parameter lists, and an optional data RAM word to change first.
fn calls() -> impl Strategy<Value = Vec<(bool, Option<(u32, u32)>)>> {
    prop::collection::vec(
        (any::<bool>(), prop::option::of((0..8_u32, 0..4_u32))),
        1..10,
    )
}

fn config() -> Config {
    Config {
        cases: 512,
        rng_seed: RngSeed::Fixed(0x6d65_6d6f),
        failure_persistence: None,
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(config())]

    /// Every call through the memo has the interpreter's exact effects, and a call the memo
    /// stored replays when repeated from the same data RAM.
    #[test]
    fn the_memo_matches_the_interpreter(
        program in program(),
        params in prop::array::uniform2(prop::collection::vec(0..4_u32, 0..3)),
        calls in calls(),
    ) {
        let mut memo = MacroMemo::new();
        let mut memo_ram = DataRam::new();
        let mut memo_host = Host::default();
        let mut oracle_ram = DataRam::new();
        let mut oracle_host = Host::default();
        // The registers macros read start out the same on both sides.
        for method in 0x40..0x48 {
            memo_host.regs.insert(method, u32::from(method) * 3);
            oracle_host.regs.insert(method, u32::from(method) * 3);
        }

        for (second, poke) in calls {
            if let Some((index, value)) = poke {
                *memo_ram.get_mut(index).unwrap() = value;
                *oracle_ram.get_mut(index).unwrap() = value;
            }
            let parameters = &params[usize::from(second)];
            let before = memo_ram.clone();
            for repeat in 0..2 {
                if repeat == 1 {
                    // Repeat the stored call from the same data RAM: its guard holds.
                    memo_ram.clone_from(&before);
                    oracle_ram.clone_from(&before);
                }
                let outcome = memo.call(&program, 0, parameters, &mut memo_ram, &mut memo_host);
                oracle_host.params = parameters.iter().copied().collect();
                let expected = run(&program, &mut oracle_ram, &mut oracle_host, StepLimit::DEFAULT);
                prop_assert_eq!(outcome.map(|_| ()), expected);
                prop_assert_eq!(&memo_ram, &oracle_ram);
                prop_assert_eq!(&memo_host.calls, &oracle_host.calls);
                prop_assert_eq!(&memo_host.regs, &oracle_host.regs);
                if repeat == 1 {
                    prop_assert_eq!(outcome, Ok(CallOutcome::Replayed));
                } else if outcome != Ok(CallOutcome::Stored) {
                    break;
                }
            }
        }
    }
}
