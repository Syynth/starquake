//! Fixtures ported from Mesa's Turing simulator tests
//! (`mesa-nouveau/src/nouveau/mme/tests/mme_tu104_sim_hw_test.cpp` @ 52238762, MIT), plus error
//! cases for malformed programs and property tests.
//!
//! Each Mesa test builds a macro with `mme_builder`, runs it on a GPU and in the simulator and
//! compares memory. The programs here are the instruction sequences those builder calls lower to
//! (the builder call is named in a comment), observed through data RAM instead of a report
//! semaphore, and the expected values are worked out here rather than read off a GPU.
//! Method numbers are arbitrary test ids.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use super::testing::{
    DMA_FIFOED_METHOD, Host, add_imm, alu, branch, finish, inst0, inst01, jal, loop_, merge, mov,
    mov_reg, op3, r, run_program, store, with_imm, with_out,
};
use super::{
    Alu, AluOp, DATA_RAM_WORDS, DataRam, DecodeError, EncodeError, GPR_COUNT, INSTRUCTION_WORDS,
    Instruction, LOAD_BARRIER, METHOD_MASK, MacroError, Out, OutOp, Pred, Reg, StepLimit,
    decode_program, encode_program, inst, run,
};

/// The branch offset that ends the macro, as the signed offset the `branch` helper takes.
const EXIT_OFFSET: i16 = inst::EXIT_OFFSET.cast_signed();

fn lo(value: u64) -> u32 {
    u32::try_from(value & 0xffff_ffff).unwrap()
}

fn hi(value: u64) -> u32 {
    u32::try_from(value >> 32).unwrap()
}

fn word(ram: &DataRam, index: u32) -> u32 {
    ram.get(index).unwrap()
}

/// Finishes `program` like Mesa's builder and runs it.
fn go(program: Vec<Instruction>, params: &[u32]) -> (DataRam, Host) {
    run_program(&finish(program), params).unwrap()
}

fn fails(program: Vec<Instruction>, params: &[u32]) -> MacroError {
    run_program(&finish(program), params).err().unwrap()
}

fn imm_u16(value: u32) -> u16 {
    u16::try_from(value).unwrap()
}

// ---------------------------------------------------------------------------------------------
// Outputs, parameters, predicates.

#[test]
fn outputs_take_their_sources_from_the_immediates() {
    // Mesa: out_imm0, out_imm1, out_immhigh0, out_immhigh1, out_imm32, in both output slots.
    let cases = [
        (OutOp::Imm0, 0x1234, 0x0000, 0x1234),
        (OutOp::Imm1, 0x0000, 0x8765, 0x8765),
        (OutOp::ImmHigh0, 0x8765, 0x0000, 0x8),
        (OutOp::ImmHigh1, 0x0000, 0x1234, 0x1),
        (OutOp::Imm32, 0x1234, 0x7654, 0x1234_7654),
    ];
    for slot in 0..2 {
        for (source, imm0, imm1, expected) in cases {
            let emit = with_out(Instruction::default(), slot, OutOp::None, source);
            let (_, host) = go(vec![with_imm(emit, imm0, imm1)], &[]);
            assert_eq!(host.calls, [(0, expected)], "{source:?} in slot {slot}");
        }
    }
}

#[test]
fn two_parameters_feed_registers_and_outputs() {
    // Mesa: multi_param. Load0 = 2581, Load1 = 3048.
    let first = Instruction {
        alu: [
            alu(AluOp::Add, r(0), Reg::Load1, Reg::Zero),
            alu(AluOp::Add, r(1), Reg::Load0, Reg::Zero),
        ],
        // Method index 12 and 35, each with increment 1.
        imm: [0x100c, 0x1023],
        out: [
            Out {
                mthd: OutOp::Imm0,
                emit: OutOp::Load0,
            },
            Out {
                mthd: OutOp::Imm1,
                emit: OutOp::Load1,
            },
        ],
        ..Instruction::default()
    };
    let read_12 = with_imm(inst0(AluOp::State, r(2), Reg::Imm, Reg::Zero), 12, 0);
    let read_35 = with_imm(inst0(AluOp::State, r(3), Reg::Imm, Reg::Zero), 35, 0);
    let (ram, host) = go(
        vec![
            first,
            read_12,
            read_35,
            store(0, 0),
            store(1, 1),
            store(2, 2),
            store(3, 3),
        ],
        &[2581, 3048],
    );
    assert_eq!(host.calls, [(12, 2581), (35, 3048)]);
    assert_eq!(host.state_reads, [12, 35]);
    assert_eq!(
        (word(&ram, 0), word(&ram, 1), word(&ram, 2), word(&ram, 3)),
        (3048, 2581, 2581, 3048)
    );
}

#[test]
fn method_increments_across_emits_and_instructions() {
    // mthd(0x10, incr 2) then three emits, the last in a later instruction.
    let select = with_imm(
        with_out(Instruction::default(), 0, OutOp::Imm0, OutOp::Imm1),
        (2 << 12) | 0x10,
        7,
    );
    let emit_pair = with_out(
        with_out(Instruction::default(), 0, OutOp::None, OutOp::Imm0),
        1,
        OutOp::None,
        OutOp::Imm1,
    );
    let (_, host) = go(vec![select, with_imm(emit_pair, 8, 9)], &[]);
    assert_eq!(host.calls, [(0x10, 7), (0x12, 8), (0x14, 9)]);
}

#[test]
fn method_index_wraps_at_fourteen_bits() {
    // Method 0xfff with increment 15, emitted often enough to pass METHOD_MASK.
    let emits = 821_u32;
    let select = with_imm(
        with_out(Instruction::default(), 0, OutOp::Imm0, OutOp::None),
        (15 << 12) | 0xfff,
        0,
    );
    let emit = with_out(Instruction::default(), 0, OutOp::None, OutOp::Imm0);
    let (_, host) = go(
        vec![
            select,
            mov(0, imm_u16(emits)),
            loop_(r(0), 2),
            emit,
            Instruction::default(),
        ],
        &[],
    );
    let expected = (0xfff + 15 * (emits - 1)) & u32::from(METHOD_MASK);
    assert_eq!(host.calls.len(), usize::try_from(emits).unwrap());
    assert_eq!(
        host.calls.last().map(|call| u32::from(call.0)),
        Some(expected)
    );
    assert!(host.calls.iter().all(|call| call.0 <= METHOD_MASK));
}

#[test]
fn state_reads_mask_the_method_index_to_fourteen_bits() {
    // STATE x + y; Mesa casts the sum to 16 bits and shifts by two into a 16-bit address.
    let state = inst0(AluOp::State, r(2), r(0), r(1));
    let (_, host) = go(
        vec![mov_reg(0, Reg::Load0), mov_reg(1, Reg::Load1), state],
        &[0x4000, 5],
    );
    assert_eq!(host.state_reads, [5]);
}

#[test]
fn both_alu_slots_may_be_state() {
    // Mesa's `mme_tu104_sim_core` asserts that ALU slots 0 and 1 are not both STATE ("No idea why
    // the HW has this rule but it does"); Eden-NX2 allows it, and so does this interpreter (see
    // the module docs of `interp`). Pin the permissive behaviour: both reads happen, in order.
    let pair = inst01(
        alu(AluOp::State, r(0), Reg::Imm, Reg::Zero),
        alu(AluOp::State, r(1), Reg::Imm, Reg::Zero),
    );
    let (_, host) = go(vec![with_imm(pair, 3, 9), store(0, 0)], &[]);
    assert_eq!(host.state_reads, [3, 9]);
}

#[test]
fn branch_offsets_sign_extend_thirteen_bits_and_exit_is_distinct() {
    assert_eq!(inst::branch_offset(0), 0);
    assert_eq!(inst::branch_offset(5), 5);
    assert_eq!(inst::branch_offset(0x1fff), -1);
    assert_eq!(inst::branch_offset(0x1ffe), -2);
    // Bits 14 and 15 are not part of the offset.
    assert_eq!(inst::branch_offset(0xe003), 3);
    assert!(inst::is_exit_offset(inst::EXIT_OFFSET));
    assert!(inst::is_exit_offset(0x8000 | inst::EXIT_OFFSET));
    assert!(!inst::is_exit_offset(inst::EXIT_OFFSET + 1));
}

#[test]
fn load1_alone_consumes_one_parameter_and_both_load_in_order() {
    let (ram, host) = go(vec![mov_reg(0, Reg::Load1), store(0, 0)], &[7, 8]);
    assert_eq!((word(&ram, 0), host.loads), (7, 1));

    let both = inst01(
        alu(AluOp::Add, r(0), Reg::Load0, Reg::Zero),
        alu(AluOp::Add, r(1), Reg::Load1, Reg::Zero),
    );
    let (ram, host) = go(vec![both, store(0, 0), store(1, 1)], &[1, 2]);
    assert_eq!((word(&ram, 0), word(&ram, 1), host.loads), (1, 2, 2));
}

#[test]
fn load_fires_for_the_predicate_register_and_for_disabled_lanes() {
    // The predicate register reads LOAD0 even in mode UUUU.
    let pred_load = Instruction {
        pred: Reg::Load0,
        ..Instruction::default()
    };
    let (_, host) = go(vec![pred_load], &[9]);
    assert_eq!(host.loads, 1);

    // A lane that is switched off still consumes the parameter it names.
    let disabled = Instruction {
        pred_mode: Pred::Tttt,
        pred: r(0),
        ..inst0(AluOp::Add, r(1), Reg::Load0, Reg::Zero)
    };
    let (_, host) = go(vec![mov(0, 1), mov(1, 0), disabled], &[5]);
    assert_eq!(host.loads, 1);
}

#[test]
fn predicate_modes_enable_the_documented_lanes() {
    // Mesa: pred_alu and pred_out, all sixteen modes with a zero and a non-zero predicate.
    // `lane_mask` bit 0/1 are the ALU slots, bit 2/3 the output slots.
    for (mode, (nonzero, zero)) in Pred::ALL.into_iter().zip(super::testing::PRED_LANES) {
        for (value, expected) in [(0_u16, zero), (25894, nonzero)] {
            let predicated = Instruction {
                pred_mode: mode,
                pred: r(0),
                alu: [
                    alu(AluOp::Add, r(1), Reg::Imm, Reg::Zero),
                    alu(AluOp::Add, r(2), Reg::Imm, Reg::Zero),
                ],
                imm: [111, 222],
                out: [
                    Out {
                        mthd: OutOp::None,
                        emit: OutOp::Imm0,
                    },
                    Out {
                        mthd: OutOp::None,
                        emit: OutOp::Imm1,
                    },
                ],
                ..Instruction::default()
            };
            let (ram, host) = go(
                vec![
                    mov(0, value),
                    mov(1, 0),
                    mov(2, 0),
                    predicated,
                    store(0, 1),
                    store(1, 2),
                ],
                &[],
            );
            let ran = |lane: u8| expected & (1 << lane) != 0;
            let context = format!("{mode} with predicate {value}");
            assert_eq!(
                word(&ram, 0),
                if ran(0) { 111 } else { 0 },
                "alu 0, {context}"
            );
            assert_eq!(
                word(&ram, 1),
                if ran(1) { 222 } else { 0 },
                "alu 1, {context}"
            );
            assert_eq!(host.calls.contains(&(0, 111)), ran(2), "out 0, {context}");
            assert_eq!(host.calls.contains(&(0, 222)), ran(3), "out 1, {context}");
            assert_eq!(mode.lane_mask(value != 0), expected, "lane_mask, {context}");
        }
    }
}

#[test]
fn mode_uuuu_does_not_read_the_predicate_register() {
    let unset = Instruction {
        pred: r(5),
        ..Instruction::default()
    };
    go(vec![unset], &[]);

    let predicated = Instruction {
        pred_mode: Pred::Tttt,
        ..unset
    };
    assert_eq!(
        fails(vec![predicated], &[]),
        MacroError::UninitializedRegister { pc: 0, reg: 5 }
    );
}

#[test]
fn the_second_alu_slot_sees_the_first_slots_register_write() {
    // Mesa evaluates slot 0 completely before slot 1 (Eden-NX2 reads both operands first).
    let pair = inst01(
        alu(AluOp::Add, r(0), Reg::Imm, Reg::Zero),
        alu(AluOp::Add, r(1), r(0), Reg::Zero),
    );
    let (ram, _) = go(vec![with_imm(pair, 5, 0), store(0, 1)], &[]);
    assert_eq!(word(&ram, 0), 5);
}

// ---------------------------------------------------------------------------------------------
// Arithmetic and logic.

/// Loads four parameters into `r0..r3` (x lo, x hi, y lo, y hi) like `mme_load` four times.
fn load_pairs() -> Vec<Instruction> {
    vec![
        inst01(
            alu(AluOp::Add, r(0), Reg::Load0, Reg::Zero),
            alu(AluOp::Add, r(1), Reg::Load1, Reg::Zero),
        ),
        inst01(
            alu(AluOp::Add, r(2), Reg::Load0, Reg::Zero),
            alu(AluOp::Add, r(3), Reg::Load1, Reg::Zero),
        ),
    ]
}

const X_LO: u32 = 0x8000_8650;
const X_HI: u32 = 0x596;
const Y_LO: u32 = 0x8000_a8f6;
const Y_HI: u32 = 0x836;

fn pair(low: u32, high: u32) -> u64 {
    (u64::from(high) << 32) | u64::from(low)
}

#[test]
fn add_and_addc_make_a_64_bit_sum() {
    // Mesa: add, addc (mme_add64).
    let mut program = load_pairs();
    program.extend([
        inst01(
            alu(AluOp::Add, r(4), r(0), r(2)),
            alu(AluOp::Addc, r(5), r(1), r(3)),
        ),
        store(0, 4),
        store(1, 5),
    ]);
    let (ram, _) = go(program, &[X_LO, X_HI, Y_LO, Y_HI]);
    let sum = pair(X_LO, X_HI).wrapping_add(pair(Y_LO, Y_HI));
    assert_eq!((word(&ram, 0), word(&ram, 1)), (lo(sum), hi(sum)));
    // The low halves overflow, so the carry mattered.
    assert!(u64::from(X_LO) + u64::from(Y_LO) > u64::from(u32::MAX));
}

#[test]
fn sub_and_subb_make_a_64_bit_difference() {
    // Mesa: sub, subb (mme_sub64).
    let mut program = load_pairs();
    program.extend([
        inst01(
            alu(AluOp::Sub, r(4), r(0), r(2)),
            alu(AluOp::Subb, r(5), r(1), r(3)),
        ),
        store(0, 4),
        store(1, 5),
    ]);
    let (ram, _) = go(program, &[X_LO, X_HI, Y_LO, Y_HI]);
    let diff = pair(X_LO, X_HI).wrapping_sub(pair(Y_LO, Y_HI));
    assert_eq!((word(&ram, 0), word(&ram, 1)), (lo(diff), hi(diff)));
    // The low halves borrow.
    const { assert!(X_LO < Y_LO) };
}

#[test]
fn an_idle_alu_slot_clears_the_carry_so_addc_must_share_the_adds_instruction() {
    // An unused slot is `ADD $zero $zero`, which sets the carry to 0. Mesa therefore only ever
    // puts ADDC, SUBB and MULH in slot 1 of the instruction that produced the carry.
    let same_instruction = inst01(
        alu(AluOp::Add, r(2), r(0), r(1)),
        alu(AluOp::Addc, r(3), Reg::Zero, Reg::Zero),
    );
    let split = vec![
        op3(AluOp::Add, 2, 0, 1),
        inst0(AluOp::Addc, r(3), Reg::Zero, Reg::Zero),
    ];
    for (instructions, expected) in [(vec![same_instruction], 1), (split, 0)] {
        let mut program = vec![mov_reg(0, Reg::Load0), mov_reg(1, Reg::Load1)];
        program.extend(instructions);
        program.push(store(0, 3));
        let (ram, _) = go(program, &[u32::MAX, 2]);
        assert_eq!(word(&ram, 0), expected);
    }

    // SUBB takes the borrow of a SUB in the same instruction: (1 - 2) - 1.
    let sub_pair = inst01(
        alu(AluOp::Sub, r(2), r(0), r(1)),
        alu(AluOp::Subb, r(3), Reg::Zero, Reg::Zero),
    );
    let (ram, _) = go(
        vec![
            mov_reg(0, Reg::Load0),
            mov_reg(1, Reg::Load1),
            sub_pair,
            store(0, 3),
        ],
        &[1, 2],
    );
    assert_eq!(word(&ram, 0), u32::MAX);

    // Equal operands do not borrow.
    let (ram, _) = go(
        vec![
            mov_reg(0, Reg::Load0),
            mov_reg(1, Reg::Load1),
            sub_pair,
            store(0, 3),
        ],
        &[5, 5],
    );
    assert_eq!(word(&ram, 0), 0);
}

#[test]
fn immediates_are_sign_extended() {
    // Mesa: add_imm. Imm and ImmPair are 16-bit and sign-extended; Imm32 joins both slots.
    for x in [0x0000_ffff_u32, 0x0000_8000, 0x0001_ffff, 0xffff_ffff] {
        let imm_32 = with_imm(inst0(AluOp::Add, r(5), r(0), Reg::Imm32), 0x1234, 0x5678);
        let from_pair = with_imm(inst0(AluOp::Add, r(4), r(0), Reg::ImmPair), 0, 0x8000);
        let program = vec![
            mov_reg(0, Reg::Load0),
            with_imm(inst0(AluOp::Add, r(1), r(0), Reg::Imm), 1, 0),
            with_imm(inst0(AluOp::Add, r(2), r(0), Reg::Imm), 0xffff, 0),
            with_imm(inst0(AluOp::Add, r(3), r(0), Reg::Imm), 0x8000, 0),
            from_pair,
            imm_32,
            store(0, 1),
            store(1, 2),
            store(2, 3),
            store(3, 4),
            store(4, 5),
        ];
        let (ram, _) = go(program, &[x]);
        let got: Vec<u32> = (0..5).map(|index| word(&ram, index)).collect();
        assert_eq!(
            got,
            [
                x.wrapping_add(1),
                x.wrapping_sub(1),
                x.wrapping_sub(0x8000),
                x.wrapping_sub(0x8000),
                x.wrapping_add(0x1234_5678),
            ],
            "x = {x:#x}"
        );
    }
}

#[test]
fn mul_and_mulh_make_a_signed_64_bit_product() {
    // Mesa: mul, mul_mulh (mme_imul_32x32_64). MULH reads the high half in the same instruction.
    let mut program = load_pairs();
    program.extend([
        inst01(
            alu(AluOp::Mul, r(4), r(0), r(1)),
            alu(AluOp::Mulh, r(5), Reg::Zero, Reg::Zero),
        ),
        store(0, 4),
        store(1, 5),
    ]);
    let (ram, _) = go(program, &[X_LO, X_HI, 0, 0]);
    let product = (i64::from(X_LO.cast_signed()) * i64::from(X_HI.cast_signed())).cast_unsigned();
    assert_eq!((word(&ram, 0), word(&ram, 1)), (lo(product), hi(product)));

    let (ram, _) = go(
        vec![
            mov(0, 25),
            mov(1, 138),
            op3(AluOp::Mul, 2, 0, 1),
            store(0, 2),
        ],
        &[],
    );
    assert_eq!(word(&ram, 0), 25 * 138);
}

#[test]
fn mulu_makes_an_unsigned_64_bit_product() {
    // Mesa: mulu_imm, mulu_mulh (mme_umul_32x32_64).
    let mut program = load_pairs();
    program.extend([
        inst01(
            alu(AluOp::Mulu, r(4), r(0), r(1)),
            alu(AluOp::Mulh, r(5), Reg::Zero, Reg::Zero),
        ),
        store(0, 4),
        store(1, 5),
    ]);
    let (ram, _) = go(program, &[X_LO, X_HI, 0, 0]);
    let product = u64::from(X_LO) * u64::from(X_HI);
    assert_eq!((word(&ram, 0), word(&ram, 1)), (lo(product), hi(product)));

    // A negative immediate is huge when multiplied unsigned: 5 * 0xffff_ffff.
    let (ram, _) = go(
        vec![
            mov_reg(0, Reg::Load0),
            with_imm(
                inst01(
                    alu(AluOp::Mulu, r(1), r(0), Reg::Imm),
                    alu(AluOp::Mulh, r(2), Reg::Zero, Reg::Zero),
                ),
                0xffff,
                0,
            ),
            store(0, 1),
            store(1, 2),
        ],
        &[5],
    );
    let product = 5_u64 * u64::from(u32::MAX);
    assert_eq!((word(&ram, 0), word(&ram, 1)), (lo(product), hi(product)));
}

#[test]
fn clz_counts_leading_zeros() {
    // Mesa: clz.
    for (value, expected) in [
        (0, 32),
        (1, 31),
        (0x00ff_0000, 8),
        (0x8000_0000, 0),
        (u32::MAX, 0),
    ] {
        let (ram, _) = go(
            vec![
                mov_reg(0, Reg::Load0),
                inst0(AluOp::Clz, r(1), r(0), Reg::Zero),
                store(0, 1),
            ],
            &[value],
        );
        assert_eq!(word(&ram, 0), expected, "clz({value:#x})");
    }
}

#[test]
fn shifts_use_the_low_five_bits_of_the_amount() {
    // Mesa: the shift and bfe tests. x = 0x8000_0010.
    let x = 0x8000_0010_u32;
    // (amount, SLL, SRL, SRA)
    let cases = [
        (0, x, x, x),
        (4, 0x0000_0100, 0x0800_0001, 0xf800_0001),
        (33, 0x0000_0020, 0x4000_0008, 0xc000_0008),
        (32, x, x, x),
    ];
    for (amount, sll, srl, sra) in cases {
        let program = vec![
            mov_reg(0, Reg::Load0),
            mov_reg(1, Reg::Load1),
            op3(AluOp::Sll, 2, 0, 1),
            op3(AluOp::Srl, 3, 0, 1),
            op3(AluOp::Sra, 4, 0, 1),
            store(0, 2),
            store(1, 3),
            store(2, 4),
        ];
        let (ram, _) = go(program, &[x, amount]);
        let got = [word(&ram, 0), word(&ram, 1), word(&ram, 2)];
        assert_eq!(got, [sll, srl, sra], "shift by {amount}");
    }
}

#[test]
fn bitwise_operations() {
    // Mesa: the and/nand/or/xor tests and `not`.
    let program = vec![
        mov_reg(0, Reg::Load0),
        mov_reg(1, Reg::Load1),
        op3(AluOp::And, 2, 0, 1),
        op3(AluOp::Nand, 3, 0, 1),
        op3(AluOp::Or, 4, 0, 1),
        op3(AluOp::Xor, 5, 0, 1),
        store(0, 2),
        store(1, 3),
        store(2, 4),
        store(3, 5),
    ];
    let (ram, _) = go(program, &[0x0ff0_f00f, 0x00ff_ff00]);
    let got: Vec<u32> = (0..4).map(|index| word(&ram, index)).collect();
    assert_eq!(got, [0x00f0_f000, 0xff0f_0fff, 0x0fff_ff0f, 0x0f0f_0f0f]);
}

/// Insert `bits` bits of `y` from `src_pos` into `x` at `dst_pos`, widened to 64 bits so no shift
/// is undefined, then truncated like a 32-bit register.
fn merge_reference(x: u32, y: u32, dst_pos: u32, bits: u32, src_pos: u32) -> u32 {
    let mask = (1_u64 << bits) - 1;
    let kept = u64::from(x) & !(mask << dst_pos);
    let inserted = ((u64::from(y) >> src_pos) & mask) << dst_pos;
    lo(kept | inserted)
}

#[test]
fn merge_inserts_a_bitfield() {
    // Mesa: merge, with mme_merge(dst_pos, bits, src_pos) for these five shapes.
    let (x, y) = (0x0c40_6fe0_u32, 0x7654_3210_u32);
    let shapes = [
        (12, 12, 20),
        (12, 8, 20),
        (8, 12, 20),
        (12, 16, 8),
        (24, 12, 8),
    ];
    let mut program = vec![mov_reg(0, Reg::Load0), mov_reg(1, Reg::Load1)];
    for (index, (dst_pos, bits, src_pos)) in shapes.into_iter().enumerate() {
        program.push(merge(2, 0, 1, dst_pos, bits, src_pos));
        program.push(store(u16::try_from(index).unwrap(), 2));
    }
    let (ram, _) = go(program, &[x, y]);
    for (index, (dst_pos, bits, src_pos)) in shapes.into_iter().enumerate() {
        assert_eq!(
            word(&ram, u32::try_from(index).unwrap()),
            merge_reference(x, y, dst_pos.into(), bits.into(), src_pos.into()),
            "merge({dst_pos}, {bits}, {src_pos})"
        );
    }
}

#[test]
fn merge_hand_checked_cases() {
    let cases = [
        // (x, y, dst_pos, bits, src_pos, expected)
        (0xffff_ffff, 0x0000_0000, 4, 8, 0, 0xffff_f00f),
        (0x1234_5678, 0xffff_ffff, 4, 0, 0, 0x1234_5678),
        (0x0000_0000, 0x0000_abcd, 0, 16, 0, 0x0000_abcd),
        (0x0000_0000, 0x8000_0000, 0, 1, 31, 0x0000_0001),
        (0xffff_ffff, 0x0000_0000, 28, 8, 0, 0x0fff_ffff),
    ];
    for (x, y, dst_pos, bits, src_pos, expected) in cases {
        let program = vec![
            mov_reg(0, Reg::Load0),
            mov_reg(1, Reg::Load1),
            merge(2, 0, 1, dst_pos, bits, src_pos),
            store(0, 2),
        ];
        let (ram, _) = go(program, &[x, y]);
        assert_eq!(
            word(&ram, 0),
            expected,
            "merge({dst_pos}, {bits}, {src_pos})"
        );
    }
}

#[test]
fn comparisons_produce_all_ones_or_zero() {
    // Mesa: the slt/sltu/sle/sleu/seq tests.
    let pairs = [(u32::MAX, 1), (1, u32::MAX), (5, 5), (0, 0x8000_0000)];
    for (x, y) in pairs {
        let program = vec![
            mov_reg(0, Reg::Load0),
            mov_reg(1, Reg::Load1),
            op3(AluOp::Slt, 2, 0, 1),
            op3(AluOp::Sltu, 3, 0, 1),
            op3(AluOp::Sle, 4, 0, 1),
            op3(AluOp::Sleu, 5, 0, 1),
            op3(AluOp::Seq, 6, 0, 1),
            store(0, 2),
            store(1, 3),
            store(2, 4),
            store(3, 5),
            store(4, 6),
        ];
        let (ram, _) = go(program, &[x, y]);
        let mask = |condition: bool| if condition { u32::MAX } else { 0 };
        let (sx, sy) = (x.cast_signed(), y.cast_signed());
        let expected = [
            mask(sx < sy),
            mask(x < y),
            mask(sx <= sy),
            mask(x <= y),
            mask(x == y),
        ];
        let got: Vec<u32> = (0..5).map(|index| word(&ram, index)).collect();
        assert_eq!(got, expected, "x = {x:#x}, y = {y:#x}");
    }
}

// ---------------------------------------------------------------------------------------------
// Control flow.

#[test]
fn loop_repeats_its_body() {
    // Mesa: loop. x += count, count times; y counts the code after the loop.
    for count in [1_u32, 5, 9] {
        let program = vec![
            mov_reg(0, Reg::Load0),
            inst01(
                alu(AluOp::Add, r(1), Reg::Zero, Reg::Zero),
                alu(AluOp::Add, r(2), Reg::Zero, Reg::Zero),
            ),
            loop_(r(0), 2),
            Instruction::default(),
            op3(AluOp::Add, 1, 1, 0),
            add_imm(2, 1),
            store(0, 0),
            store(1, 1),
            store(2, 2),
        ];
        let (ram, _) = go(program, &[count]);
        assert_eq!(
            (word(&ram, 0), word(&ram, 1), word(&ram, 2)),
            (count, count * count, 1)
        );
    }
}

#[test]
fn loop_with_a_zero_count_runs_the_body_once() {
    // Mesa: MAX2(1, x) - 1 extra passes. Eden-NX2 skips the body; see the interpreter docs.
    let program = vec![
        mov_reg(0, Reg::Load0),
        mov(1, 0),
        loop_(r(0), 1),
        add_imm(1, 1),
        Instruction::default(),
        store(0, 1),
    ];
    let (ram, _) = go(program, &[0]);
    assert_eq!(word(&ram, 0), 1);
}

#[test]
fn a_loop_inside_a_loop_is_an_error() {
    let program = vec![
        mov(0, 3),
        loop_(r(0), 3),
        loop_(r(0), 1),
        Instruction::default(),
        Instruction::default(),
    ];
    assert_eq!(fails(program, &[]), MacroError::NestedLoop { pc: 2 });
}

#[test]
fn a_loop_needs_a_body_of_at_least_one_instruction() {
    for length in [0, 1] {
        let program = vec![
            mov(0, 3),
            with_imm(inst0(AluOp::Loop, Reg::Zero, r(0), Reg::Zero), length, 0),
        ];
        assert_eq!(
            fails(program, &[]),
            MacroError::InvalidLoop { pc: 1, length }
        );
    }
}

#[test]
fn jal_skips_forward() {
    // Mesa: jal. JAL +6 over five of ten increments of x.
    let mut program = vec![mov(0, 0), jal(6)];
    program.extend((0..10).map(|_| add_imm(0, 1)));
    program.push(store(0, 0));
    let (ram, _) = go(program, &[]);
    assert_eq!(word(&ram, 0), 5);
}

#[test]
fn beq_skips_forward_when_its_outcome_matches() {
    // Mesa: bxx_fwd.
    let mut program = vec![mov(0, 0), branch(AluOp::Beq, Reg::Zero, Reg::Zero, true, 6)];
    program.extend((0..10).map(|_| add_imm(0, 1)));
    program.push(store(0, 0));
    let (ram, _) = go(program, &[]);
    assert_eq!(word(&ram, 0), 5);
}

#[test]
fn branches_go_backwards() {
    // Mesa: bxx_bwd. JAL over the first ten increments, JAL again over a BEQ that jumps back
    // into the middle of them, then five more increments.
    let mut program: Vec<Instruction> = (0..15).map(|reg| mov(reg, 0)).collect();
    program.push(jal(12));
    program.extend((0..10).map(|reg| add_imm(reg, 1)));
    program.push(jal(2));
    program.push(branch(AluOp::Beq, Reg::Zero, Reg::Zero, true, -8));
    program.extend((10..15).map(|reg| add_imm(reg, 1)));
    program.extend((0..15).map(|reg| store(u16::from(reg), reg)));
    let (ram, _) = go(program, &[]);
    let got: Vec<u32> = (0..15).map(|index| word(&ram, index)).collect();
    let expected: Vec<u32> = (0..15).map(|index| u32::from(index >= 3)).collect();
    assert_eq!(got, expected);
}

#[test]
fn every_branch_condition_takes_the_branch_when_the_outcome_matches() {
    // Mesa: the if_* tests. A branch over `r2 = 1`.
    type Holds = fn(u32, u32) -> bool;
    let ops: [(AluOp, Holds); 5] = [
        (AluOp::Blt, |x, y| x.cast_signed() < y.cast_signed()),
        (AluOp::Bltu, |x, y| x < y),
        (AluOp::Ble, |x, y| x.cast_signed() <= y.cast_signed()),
        (AluOp::Bleu, |x, y| x <= y),
        (AluOp::Beq, |x, y| x == y),
    ];
    let pairs = [(u32::MAX, 1), (1, u32::MAX), (5, 5), (0, 0x8000_0000)];
    for (op, holds) in ops {
        for expect in [false, true] {
            for (x, y) in pairs {
                let program = vec![
                    inst01(
                        alu(AluOp::Add, r(0), Reg::Load0, Reg::Zero),
                        alu(AluOp::Add, r(1), Reg::Load1, Reg::Zero),
                    ),
                    mov(2, 0),
                    branch(op, r(0), r(1), expect, 2),
                    mov(2, 1),
                    store(0, 2),
                ];
                let (ram, _) = go(program, &[x, y]);
                let taken = holds(x, y) == expect;
                assert_eq!(
                    word(&ram, 0),
                    u32::from(!taken),
                    "{op:?} expect {expect} on {x:#x}, {y:#x}"
                );
            }
        }
    }
}

#[test]
fn a_backward_branch_makes_a_do_while() {
    // Mesa: do_ble / while_*. Sum 5 + 4 + 3 + 2 + 1 while 0 < counter.
    let program = vec![
        mov(0, 5),
        mov(1, 0),
        op3(AluOp::Add, 1, 1, 0),
        with_imm(inst0(AluOp::Sub, r(0), r(0), Reg::Imm), 1, 0),
        branch(AluOp::Blt, Reg::Zero, r(0), true, -2),
        store(0, 1),
        store(1, 0),
    ];
    let (ram, _) = go(program, &[]);
    assert_eq!((word(&ram, 0), word(&ram, 1)), (15, 0));
}

#[test]
fn a_branch_with_the_exit_offset_ends_the_macro() {
    // Mesa: bxx_exit and mme_exit (mme_exit_if). Stores after the exit are not visible.
    let program = vec![
        mov(0, 7),
        store(0, 0),
        branch(AluOp::Beq, Reg::Zero, Reg::Zero, true, EXIT_OFFSET),
        store(1, 0),
        store(2, 0),
    ];
    let (ram, _) = go(program, &[]);
    assert_eq!((word(&ram, 0), word(&ram, 1), word(&ram, 2)), (7, 0, 0));
}

#[test]
fn an_exit_branch_that_is_not_taken_falls_through() {
    // mme_exit_if with a condition that does not hold.
    let program = vec![
        mov(0, 7),
        branch(AluOp::Beq, r(0), Reg::Zero, true, EXIT_OFFSET),
        store(0, 0),
    ];
    let (ram, _) = go(program, &[]);
    assert_eq!(word(&ram, 0), 7);
}

#[test]
fn end_next_lets_one_more_instruction_run() {
    // The marked instruction and the one after it run; nothing later does.
    let mut marked = store(0, 0);
    marked.end_next = true;
    let program = vec![mov(0, 1), marked, store(1, 0), store(2, 0)];
    let (ram, _) = run_program(&program, &[]).unwrap();
    assert_eq!((word(&ram, 0), word(&ram, 1), word(&ram, 2)), (1, 1, 0));
}

// ---------------------------------------------------------------------------------------------
// Data RAM.

#[test]
fn dwrite_and_dread_round_trip() {
    // Mesa: dread_dwrite. Params -10 and 5 go to words 5 and 8, then are read back in reverse.
    let program = vec![
        inst01(
            alu(AluOp::Add, r(0), Reg::Load0, Reg::Zero),
            alu(AluOp::Add, r(1), Reg::Load1, Reg::Zero),
        ),
        store(5, 0),
        store(8, 1),
        with_imm(inst0(AluOp::Dread, r(2), Reg::Imm, Reg::Zero), 8, 0),
        with_imm(inst0(AluOp::Dread, r(3), Reg::Imm, Reg::Zero), 5, 0),
        store(20, 2),
        store(21, 3),
    ];
    let (ram, _) = go(program, &[(-10_i32).cast_unsigned(), 5]);
    assert_eq!(
        (word(&ram, 5), word(&ram, 8), word(&ram, 20), word(&ram, 21)),
        ((-10_i32).cast_unsigned(), 5, 5, (-10_i32).cast_unsigned())
    );
}

#[test]
fn every_data_ram_word_is_reachable_and_the_next_is_not() {
    // Mesa: dram_limit. A loop writes `i` to word `i` for `count` words starting at `start`.
    let program = vec![
        inst01(
            alu(AluOp::Add, r(0), Reg::Load0, Reg::Zero),
            alu(AluOp::Add, r(1), Reg::Load1, Reg::Zero),
        ),
        mov_reg(2, r(0)),
        loop_(r(1), 2),
        inst0(AluOp::Dwrite, Reg::Zero, r(2), r(2)),
        add_imm(2, 1),
    ];
    let words = u32::try_from(DATA_RAM_WORDS).unwrap();
    let (ram, _) = go(program.clone(), &[0, words]);
    assert!((0..words).all(|index| word(&ram, index) == index));

    let error = fails(program, &[0, words + 1]);
    assert_eq!(
        error,
        MacroError::DataRamOutOfBounds {
            pc: 3,
            index: words
        }
    );
}

#[test]
fn dread_outside_data_ram_is_an_error() {
    let program = vec![
        mov_reg(0, Reg::Load0),
        inst0(AluOp::Dread, r(1), r(0), Reg::Zero),
    ];
    assert_eq!(
        fails(program, &[u32::MAX]),
        MacroError::DataRamOutOfBounds {
            pc: 1,
            index: u32::MAX
        }
    );
}

#[test]
fn data_ram_belongs_to_the_caller_and_persists_across_runs() {
    // Eden-NX2 notes, row 84: each channel keeps its own data RAM; macros share code, not state.
    let writer = finish(vec![mov(0, 42), store(7, 0)]);
    let reader = finish(vec![
        with_imm(inst0(AluOp::Dread, r(1), Reg::Imm, Reg::Zero), 7, 0),
        store(8, 1),
    ]);
    let mut channel_a = DataRam::new();
    let mut channel_b = DataRam::new();
    let mut host = Host::default();
    run(&writer, &mut channel_a, &mut host, StepLimit::DEFAULT).unwrap();
    run(&reader, &mut channel_a, &mut host, StepLimit::DEFAULT).unwrap();
    run(&reader, &mut channel_b, &mut host, StepLimit::DEFAULT).unwrap();
    assert_eq!(word(&channel_a, 8), 42);
    assert_eq!(word(&channel_b, 8), 0);
    assert_ne!(channel_a, channel_b);
}

#[test]
fn data_ram_accessors_are_bounds_checked() {
    let mut ram = DataRam::new();
    assert_eq!(ram.as_slice().len(), DATA_RAM_WORDS);
    let last = u32::try_from(DATA_RAM_WORDS - 1).unwrap();
    *ram.get_mut(last).unwrap() = 9;
    assert_eq!(ram.get(last), Some(9));
    assert_eq!(ram.get(last + 1), None);
    assert!(ram.get_mut(u32::MAX).is_none());
    ram.as_mut_slice()[0] = 3;
    assert_eq!(ram.get(0), Some(3));
    assert_eq!(DataRam::default(), DataRam::new());
}

// ---------------------------------------------------------------------------------------------
// The DMA-read hook.

#[test]
fn the_load_barrier_lets_the_host_refill_the_load_fifo() {
    // Mesa: dma_read_fifoed. mthd(dma, 0) emit(2); barrier; load x; load y. The test host
    // appends the requested words from its memory when the barrier runs.
    let select = with_imm(
        with_out(Instruction::default(), 0, OutOp::Imm0, OutOp::None),
        DMA_FIFOED_METHOD,
        2,
    );
    let request = with_out(Instruction::default(), 0, OutOp::None, OutOp::Imm1);
    let barrier = with_imm(
        inst0(AluOp::Extended, Reg::Zero, Reg::Imm, Reg::ImmPair),
        imm_u16(LOAD_BARRIER),
        1,
    );
    let program = vec![
        select,
        with_imm(request, 0, 2),
        barrier,
        inst01(
            alu(AluOp::Add, r(0), Reg::Load0, Reg::Zero),
            alu(AluOp::Add, r(1), Reg::Load1, Reg::Zero),
        ),
        store(0, 0),
        store(1, 1),
    ];
    let mut host = Host {
        memory: (0..64).map(|index| 1000 + index).collect(),
        ..Host::default()
    };
    let mut ram = DataRam::new();
    run(&finish(program), &mut ram, &mut host, StepLimit::DEFAULT).unwrap();
    assert_eq!((word(&ram, 0), word(&ram, 1)), (1000, 1001));
    assert_eq!(host.barriers, 1);
}

#[test]
fn only_the_load_barrier_is_a_known_extended_operation() {
    let other = with_imm(inst0(AluOp::Extended, Reg::Zero, Reg::Imm, Reg::Zero), 5, 0);
    assert_eq!(
        fails(vec![other], &[]),
        MacroError::UnsupportedExtended { pc: 0, x: 5 }
    );
}

// ---------------------------------------------------------------------------------------------
// Malformed programs: errors, never panics or hangs.

#[test]
fn running_out_of_parameters_is_an_error() {
    assert_eq!(
        fails(vec![mov_reg(0, Reg::Load0)], &[]),
        MacroError::ParameterUnderflow { pc: 0 }
    );
    let two = inst01(
        alu(AluOp::Add, r(0), Reg::Load0, Reg::Zero),
        alu(AluOp::Add, r(1), Reg::Load1, Reg::Zero),
    );
    assert_eq!(
        fails(vec![two], &[1]),
        MacroError::ParameterUnderflow { pc: 0 }
    );
}

#[test]
fn reading_an_unwritten_register_is_an_error() {
    assert_eq!(
        fails(vec![op3(AluOp::Add, 1, 0, 0)], &[]),
        MacroError::UninitializedRegister { pc: 0, reg: 0 }
    );
}

#[test]
fn registers_do_not_survive_between_runs() {
    let program = finish(vec![mov(0, 1)]);
    let reads = finish(vec![op3(AluOp::Add, 1, 0, 0)]);
    let mut ram = DataRam::new();
    let mut host = Host::default();
    run(&program, &mut ram, &mut host, StepLimit::DEFAULT).unwrap();
    assert_eq!(
        run(&reads, &mut ram, &mut host, StepLimit::DEFAULT),
        Err(MacroError::UninitializedRegister { pc: 0, reg: 0 })
    );
}

#[test]
fn unusable_register_selectors_are_errors() {
    assert_eq!(
        fails(
            vec![inst0(AluOp::Add, r(0), Reg::Invalid(30), Reg::Zero)],
            &[]
        ),
        MacroError::InvalidRegister { pc: 0, bits: 30 }
    );
    assert_eq!(
        fails(
            vec![inst0(AluOp::Add, Reg::Invalid(31), Reg::Zero, Reg::Zero)],
            &[]
        ),
        MacroError::InvalidRegister { pc: 0, bits: 31 }
    );
    assert_eq!(
        fails(vec![inst0(AluOp::Add, Reg::Imm, Reg::Zero, Reg::Zero)], &[]),
        MacroError::InvalidRegister { pc: 0, bits: 25 }
    );
    assert_eq!(
        fails(
            vec![inst0(AluOp::Add, Reg::Gpr(GPR_COUNT), Reg::Zero, Reg::Zero)],
            &[]
        ),
        MacroError::InvalidRegister {
            pc: 0,
            bits: GPR_COUNT
        }
    );
    // The predicate register has no slot to take an immediate from.
    let predicated = Instruction {
        pred_mode: Pred::Tttt,
        pred: Reg::Imm,
        ..Instruction::default()
    };
    assert_eq!(
        fails(vec![predicated], &[]),
        MacroError::InvalidRegister { pc: 0, bits: 25 }
    );
}

#[test]
fn unusable_output_sources_are_errors() {
    for (source, bits) in [(OutOp::Reserved, 7), (OutOp::Invalid(12), 12)] {
        let program = vec![with_out(Instruction::default(), 1, OutOp::None, source)];
        assert_eq!(
            fails(program, &[]),
            MacroError::InvalidOutput { pc: 0, bits }
        );
        let program = vec![with_out(Instruction::default(), 0, source, OutOp::None)];
        assert_eq!(
            fails(program, &[]),
            MacroError::InvalidOutput { pc: 0, bits }
        );
    }
}

#[test]
fn a_merge_past_bit_31_is_an_error() {
    let program = vec![mov(0, 1), mov(1, 2), merge(2, 0, 1, 32, 4, 0)];
    assert_eq!(
        fails(program, &[]),
        MacroError::InvalidMerge { pc: 2, dst_pos: 32 }
    );
}

#[test]
fn leaving_the_program_is_an_error() {
    // Falling off the end, and an empty program.
    assert_eq!(
        run_program(&[Instruction::default()], &[]).err(),
        Some(MacroError::PcOutOfRange { pc: 1 })
    );
    assert_eq!(
        run_program(&[], &[]).err(),
        Some(MacroError::PcOutOfRange { pc: 0 })
    );

    // Jumps and branches whose target is outside it.
    assert_eq!(
        fails(vec![jal(100)], &[]),
        MacroError::BranchOutOfRange { pc: 0, offset: 100 }
    );
    assert_eq!(
        fails(
            vec![branch(AluOp::Beq, Reg::Zero, Reg::Zero, true, -1)],
            &[]
        ),
        MacroError::BranchOutOfRange { pc: 0, offset: -1 }
    );
    // A branch that is not taken is not checked.
    go(
        vec![branch(AluOp::Beq, Reg::Zero, Reg::Zero, false, -1)],
        &[],
    );
}

#[test]
fn a_program_that_never_ends_hits_the_step_limit() {
    // JAL +0 spins in place; each pass loads one parameter, so loads count the steps.
    let spin = with_out(jal(0), 0, OutOp::None, OutOp::Load0);
    let mut host = Host {
        endless: true,
        ..Host::default()
    };
    let mut ram = DataRam::new();
    let result = run(&[spin], &mut ram, &mut host, StepLimit::new(1000));
    assert_eq!(result, Err(MacroError::StepLimitExceeded { limit: 1000 }));
    assert_eq!(host.loads, 1000);
}

#[test]
fn the_step_limit_is_exact() {
    // Three instructions run: a nop, the marked one, and the one after it.
    let program = finish(vec![Instruction::default()]);
    let mut ram = DataRam::new();
    let mut host = Host::default();
    assert_eq!(
        run(&program, &mut ram, &mut host, StepLimit::new(3)),
        Ok(())
    );
    assert_eq!(
        run(&program, &mut ram, &mut host, StepLimit::new(2)),
        Err(MacroError::StepLimitExceeded { limit: 2 })
    );
    assert_eq!(
        run(&program, &mut ram, &mut host, StepLimit::new(0)),
        Err(MacroError::StepLimitExceeded { limit: 0 })
    );
    assert_eq!(StepLimit::default(), StepLimit::DEFAULT);
    assert_eq!(StepLimit::new(7).get(), 7);
}

#[test]
fn a_huge_loop_count_is_bounded_by_the_step_limit() {
    let program = finish(vec![
        mov_reg(0, Reg::Load0),
        loop_(r(0), 2),
        Instruction::default(),
        Instruction::default(),
    ]);
    let mut ram = DataRam::new();
    let mut host = Host::with_params(&[u32::MAX]);
    assert_eq!(
        run(&program, &mut ram, &mut host, StepLimit::new(10_000)),
        Err(MacroError::StepLimitExceeded { limit: 10_000 })
    );
}

// ---------------------------------------------------------------------------------------------
// Instruction format.

#[test]
fn the_predicate_table_matches_the_letters() {
    assert_eq!(Pred::ALL.len(), super::testing::PRED_LANES.len());
    for (index, (mode, (nonzero, zero))) in Pred::ALL
        .into_iter()
        .zip(super::testing::PRED_LANES)
        .enumerate()
    {
        assert_eq!(usize::from(mode.bits()), index);
        assert_eq!(Pred::from_bits(mode.bits()), mode);
        assert_eq!(mode.lane_mask(true), nonzero, "{mode} non-zero");
        assert_eq!(mode.lane_mask(false), zero, "{mode} zero");
    }
    assert_eq!(Pred::Tfuu.letters(), "TFUU");
}

#[test]
fn field_positions_match_mesas_encoder() {
    // mme_tu104_encode packs end_next at bit 0 and the output selectors at the top, with the
    // three words reversed in the stream. One field at a time, so a misplaced field shows up.
    let encode = |inst: Instruction| inst.encode().unwrap();
    assert_eq!(encode(Instruction::default())[2] & 1, 0);
    let end = Instruction {
        end_next: true,
        ..Instruction::default()
    };
    assert_eq!(encode(end)[2] & 1, 1);

    let pred = Instruction {
        pred_mode: Pred::Uuuf,
        pred: r(3),
        ..Instruction::default()
    };
    assert_eq!(encode(pred)[2] & 0x3ff, (15 << 1) | (3 << 5));

    // ALU 0 occupies bits 10..30 and imm 0 bits 30..46, so imm 0 straddles words 2 and 1.
    let alu0 = Instruction {
        alu: [alu(AluOp::Dwrite, r(2), r(3), r(4)), Alu::default()],
        imm: [0xabcd, 0],
        ..Instruction::default()
    };
    let words = encode(alu0);
    assert_eq!((words[2] >> 10) & 0x1f, 31);
    assert_eq!((words[2] >> 15) & 0x1f, 2);
    assert_eq!((words[2] >> 20) & 0x1f, 3);
    assert_eq!((words[2] >> 25) & 0x1f, 4);
    assert_eq!(words[2] >> 30, 0xabcd & 3);
    assert_eq!(words[1] & 0x3fff, 0xabcd >> 2);

    // Output 1 sits in the top of word 0.
    let out1 = Instruction {
        out: [
            Out::default(),
            Out {
                mthd: OutOp::Reserved,
                emit: OutOp::Imm32,
            },
        ],
        ..Instruction::default()
    };
    let words = encode(out1);
    assert_eq!((words[0] >> 25) & 0x7, 7);
    assert_eq!((words[0] >> 28) & 0xf, 10);
}

#[test]
fn encoding_rejects_values_the_fields_cannot_hold() {
    let unencodable = [
        inst0(AluOp::Add, Reg::Gpr(GPR_COUNT), Reg::Zero, Reg::Zero),
        inst0(AluOp::Add, Reg::Gpr(40), Reg::Zero, Reg::Zero),
        inst0(AluOp::Add, Reg::Zero, Reg::Invalid(3), Reg::Zero),
        // The method selector is 3 bits wide.
        with_out(Instruction::default(), 0, OutOp::Imm32, OutOp::None),
        with_out(Instruction::default(), 0, OutOp::None, OutOp::Invalid(3)),
        with_out(Instruction::default(), 1, OutOp::None, OutOp::Invalid(20)),
    ];
    for inst in unencodable {
        assert!(
            matches!(inst.encode(), Err(EncodeError::FieldOverflow { .. })),
            "{inst:?}"
        );
    }
    assert!(encode_program(&unencodable).is_err());
}

#[test]
fn programs_decode_and_encode_as_a_whole() {
    let program = finish(vec![mov(0, 5), store(1, 0)]);
    let words = encode_program(&program).unwrap();
    assert_eq!(words.len(), program.len() * INSTRUCTION_WORDS);
    assert_eq!(decode_program(&words).unwrap(), program);

    assert_eq!(
        decode_program(&words[..words.len() - 1]),
        Err(DecodeError::TruncatedProgram {
            words: words.len() - 1
        })
    );
    assert_eq!(decode_program(&[]), Ok(Vec::new()));
}

#[test]
fn a_decoded_program_runs() {
    // Words as a guest would upload them, through the decoder into the interpreter.
    let program = finish(vec![mov_reg(0, Reg::Load0), store(3, 0)]);
    let words = encode_program(&program).unwrap();
    let decoded = decode_program(&words).unwrap();
    let (ram, _) = run_program(&decoded, &[77]).unwrap();
    assert_eq!(word(&ram, 3), 77);
}

#[test]
fn selectors_round_trip_through_their_encodings() {
    for bits in 0..32 {
        assert_eq!(Reg::from_bits(bits).bits(), bits);
        assert_eq!(AluOp::from_bits(bits).bits(), bits);
    }
    for bits in 0..16 {
        assert_eq!(OutOp::from_bits(bits).bits(), bits);
    }
    assert_eq!(AluOp::from_bits(7), AluOp::Extended);
    assert_eq!(AluOp::ALL.len(), 32);
    assert!(AluOp::Loop.is_control_flow() && !AluOp::Dread.is_control_flow());
    assert!(AluOp::Dwrite.has_side_effects() && !AluOp::Add.has_side_effects());
}

// ---------------------------------------------------------------------------------------------
// Disassembly.

#[test]
fn instructions_disassemble_like_mesas_printer() {
    assert_eq!(Instruction::default().to_string(), "nop");
    assert_eq!(mov(3, 5).to_string(), "$r3 = ADD 5 /* 0x0005 */ $zero");
    assert_eq!(
        mov(3, 0xffff).to_string(),
        "$r3 = ADD -1 /* 0xffff */ $zero"
    );
    assert_eq!(
        inst01(
            alu(AluOp::Add, r(4), r(0), r(2)),
            alu(AluOp::Addc, r(5), r(1), r(3)),
        )
        .to_string(),
        "$r4 = ADD $r0 $r2\n$r5 = ADDC $r1 $r3 $carry"
    );
    assert_eq!(
        merge(0, 1, 2, 12, 12, 20).to_string(),
        "$r0 = MERGE $r1 $r2 (12, 12, 20)"
    );
    assert_eq!(store(5, 1).to_string(), "DWRITE 5 /* 0x0005 */ $r1");
    assert_eq!(loop_(r(0), 2).to_string(), "LOOP $r0 (3)");
    assert_eq!(jal(6).to_string(), "JAL (0x0006)");
    assert_eq!(
        branch(AluOp::Beq, r(0), r(1), true, -8).to_string(),
        "BEQ $r0 $r1 (-8, true)"
    );
    assert_eq!(
        branch(AluOp::Blt, r(0), r(1), false, EXIT_OFFSET).to_string(),
        "BLT $r0 $r1 (exit, false)"
    );
}

#[test]
fn outputs_and_predicates_disassemble() {
    let multi = Instruction {
        alu: [
            alu(AluOp::Add, r(0), Reg::Load1, Reg::Zero),
            alu(AluOp::Add, Reg::Zero, Reg::Zero, Reg::Zero),
        ],
        // Method index 12 with increment 1.
        imm: [0x100c, 0x8765],
        out: [
            Out {
                mthd: OutOp::Imm0,
                emit: OutOp::Load0,
            },
            Out {
                mthd: OutOp::None,
                emit: OutOp::ImmHigh1,
            },
        ],
        ..Instruction::default()
    };
    assert_eq!(
        multi.to_string(),
        "$r0 = ADD $load1 $zero\nmthd(0x000c, +1)\nemit($load0)\nemit(8)"
    );

    // An unused ALU slot with a discarded result is hidden unless an output reads it.
    let read_by_out = with_out(Instruction::default(), 1, OutOp::None, OutOp::Alu1);
    assert_eq!(
        read_by_out.to_string(),
        "$alu1 = ADD $zero $zero\nemit($alu1)"
    );

    let predicated = Instruction {
        pred_mode: Pred::Tfuu,
        pred: r(5),
        end_next: true,
        ..mov(0, 1)
    };
    assert_eq!(
        predicated.to_string(),
        "pred TFUU $r5 {\n    $r0 = ADD 1 /* 0x0001 */ $zero\n    end_next\n}"
    );
    assert_eq!(Reg::Imm32.to_string(), "$imm32");
    assert_eq!(Reg::Invalid(30).to_string(), "$invalid30");
    assert_eq!(AluOp::Dwrite.to_string(), "DWRITE");
}

// ---------------------------------------------------------------------------------------------
// Property tests. A fixed seed keeps them deterministic.

fn config() -> Config {
    Config {
        cases: 256,
        rng_seed: RngSeed::Fixed(0x5741_5251),
        failure_persistence: None,
        ..Config::default()
    }
}

fn any_reg() -> impl Strategy<Value = Reg> {
    (0_u8..32).prop_map(Reg::from_bits)
}

fn any_alu() -> impl Strategy<Value = Alu> {
    (0_u8..32, any_reg(), any_reg(), any_reg()).prop_map(|(op, dst, src0, src1)| Alu {
        op: AluOp::from_bits(op),
        dst,
        src: [src0, src1],
    })
}

/// Output slots as the encoding can hold them: a 3-bit method selector and a 4-bit emit source.
fn any_out() -> impl Strategy<Value = Out> {
    (0_u8..8, 0_u8..16).prop_map(|(mthd, emit)| Out {
        mthd: OutOp::from_bits(mthd),
        emit: OutOp::from_bits(emit),
    })
}

fn any_instruction() -> impl Strategy<Value = Instruction> {
    (
        any::<bool>(),
        0_u8..16,
        any_reg(),
        any::<[u16; 2]>(),
        [any_alu(), any_alu()],
        [any_out(), any_out()],
    )
        .prop_map(|(end_next, mode, pred, imm, alu, out)| Instruction {
            end_next,
            pred_mode: Pred::from_bits(mode),
            pred,
            imm,
            alu,
            out,
        })
}

/// Selectors a fuzzed program can run with: a handful of initialised registers, immediates and
/// loads, so most programs get past their first instruction.
fn runnable_reg() -> impl Strategy<Value = Reg> {
    prop_oneof![
        3 => (0_u8..4).prop_map(Reg::Gpr),
        1 => Just(Reg::Zero),
        1 => Just(Reg::Imm),
        1 => Just(Reg::ImmPair),
        1 => Just(Reg::Imm32),
        1 => Just(Reg::Load0),
        1 => Just(Reg::Load1),
    ]
}

fn runnable_alu() -> impl Strategy<Value = Alu> {
    let dst = prop_oneof![(0_u8..4).prop_map(Reg::Gpr), Just(Reg::Zero)];
    (0_u8..32, dst, runnable_reg(), runnable_reg()).prop_map(|(op, dst, src0, src1)| Alu {
        op: AluOp::from_bits(op),
        dst,
        src: [src0, src1],
    })
}

fn runnable_instruction() -> impl Strategy<Value = Instruction> {
    (
        any::<bool>(),
        0_u8..16,
        runnable_reg(),
        any::<[u16; 2]>(),
        [runnable_alu(), runnable_alu()],
        [any_out(), any_out()],
    )
        .prop_map(|(end_next, mode, pred, imm, alu, out)| Instruction {
            end_next,
            pred_mode: Pred::from_bits(mode),
            pred,
            imm,
            alu,
            out,
        })
}

/// Everything a run leaves behind, for comparing two runs.
/// Runs `program` against an endless load FIFO, so that parameter underflow never ends the run
/// early and the step limit is what stops a program that loops.
fn outcome(program: &[Instruction], limit: u32) -> (Result<(), MacroError>, Host, DataRam) {
    let mut host = Host {
        endless: true,
        ..Host::default()
    };
    let mut ram = DataRam::new();
    let result = run(program, &mut ram, &mut host, StepLimit::new(limit));
    (result, host, ram)
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn encode_inverts_decode_for_generated_instructions(inst in any_instruction()) {
        let words = inst.encode().unwrap();
        prop_assert_eq!(Instruction::decode(words), inst);
    }

    #[test]
    fn decode_inverts_encode_for_every_bit_pattern(words in any::<[u32; 3]>()) {
        // Decoding is total and loses nothing, so a guest program is never rejected up front.
        let inst = Instruction::decode(words);
        prop_assert_eq!(inst.encode(), Ok(words));
    }

    #[test]
    fn programs_round_trip_through_words(program in prop::collection::vec(any_instruction(), 0..32)) {
        let words = encode_program(&program).unwrap();
        prop_assert_eq!(decode_program(&words).unwrap(), program);
    }

    #[test]
    fn arbitrary_words_never_panic_and_stop_within_the_limit(
        words in prop::collection::vec(any::<u32>(), 0..96),
    ) {
        let words = &words[..words.len() / INSTRUCTION_WORDS * INSTRUCTION_WORDS];
        let program = decode_program(words).unwrap();
        let limit = 200;
        let (result, host, ram) = outcome(&program, limit);
        // The FIFO never runs dry and each step loads at most two parameters, so this bounds the
        // steps that ran.
        prop_assert!(host.loads <= 2 * limit);
        prop_assert_eq!(ram.as_slice().len(), DATA_RAM_WORDS);
        // Same program, same inputs, same outcome.
        let (again, host_again, ram_again) = outcome(&program, limit);
        prop_assert_eq!(result, again);
        prop_assert_eq!(host.calls, host_again.calls);
        prop_assert_eq!(ram, ram_again);
    }

    #[test]
    fn runnable_programs_never_panic_and_stop_within_the_limit(
        body in prop::collection::vec(runnable_instruction(), 1..24),
        limit in 1_u32..400,
    ) {
        // Initialise r0..r3 first so the random body gets to run.
        let mut program: Vec<Instruction> = (0..4).map(|reg| mov(reg, u16::from(reg) + 1)).collect();
        program.extend(body);
        let mut host = Host { endless: true, ..Host::default() };
        let mut ram = DataRam::new();
        let result = run(&program, &mut ram, &mut host, StepLimit::new(limit));
        prop_assert!(host.loads <= 2 * limit);
        if let Err(MacroError::StepLimitExceeded { limit: reported }) = result {
            prop_assert_eq!(reported, limit);
        }
    }
}
