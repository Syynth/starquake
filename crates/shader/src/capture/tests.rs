//! Program capture: the extent rules at and past each edge, the cap, the header, the reader
//! seam, the jump-table extension and the hash.

use std::cell::Cell;

use proptest::prelude::*;

use super::layout::{
    BRA_DIV_LOW, BRA_DIV_MODE, BRA_OPCODE, BRANCH_LOW, BRANCH_MODE, BRANCH_OFFSET, BSSY_OFFSET,
    BSSY_OPCODE, CALL_OPCODE, OFFSET_UNIT, RET_OPCODE, branch_displacement,
};
use super::scan::{Scan, Step, branch_target};
use super::{
    Capture, CaptureError, ContentHash, Extension, JumpTables, MAX_INSTRUCTIONS, ProgramReader,
    Unreadable, capture, header_bytes, table_entry,
};
use crate::decode::{
    EXIT_KEEP_REF_COUNT, EXIT_NO_AT_EXIT, EXIT_OPCODE, EXIT_PRED, EXIT_PRED_NOT, Field, GUARD,
    GUARD_NOT, Instruction, OPCODE, Pred,
};
use crate::sph::HEADER_BYTES;
use crate::testing::config;
use crate::translate::GuestStage;

// ---------------------------------------------------------------------------------------------
// The tests' encoder: words built from the shared layout constants.

/// The four-bit guard value `PT`, not negated.
const PT: u8 = Pred::PT.index();
/// An opcode the capture does not read: not one of the control instructions it looks at.
const FILLER_OPCODE: u16 = 0x202;
/// Every graphics stage.
const GRAPHICS: [GuestStage; 6] = [
    GuestStage::VertexCullBeforeFetch,
    GuestStage::Vertex,
    GuestStage::TessellationInit,
    GuestStage::Tessellation,
    GuestStage::Geometry,
    GuestStage::Pixel,
];
/// [`Instruction::SIZE`] as a number of bytes for offsets.
const SIZE: u64 = Instruction::SIZE as u64;

/// A word with `opcode` and the four-bit guard `guard` (bit 3 negates).
fn guarded(opcode: u16, guard: u8) -> Instruction {
    Instruction::default()
        .with_field(OPCODE, u64::from(opcode))
        .with_field(GUARD, u64::from(guard & 7))
        .with_flag(GUARD_NOT, guard & 8 != 0)
}

/// An instruction the capture does not read.
fn filler() -> Instruction {
    guarded(FILLER_OPCODE, PT)
}

/// An `EXIT` that ends the program where the rules allow.
fn exit() -> Instruction {
    guarded(EXIT_OPCODE, PT).with_field(EXIT_PRED, u64::from(PT))
}

/// A `RET` whose guard is `PT`.
fn ret() -> Instruction {
    guarded(RET_OPCODE, PT)
}

/// A relative branch with `opcode` and `guard`, displaced by `bytes` from its end.
fn branch_by(opcode: u16, guard: u8, bytes: i64) -> Instruction {
    let offset = if opcode == BSSY_OPCODE {
        BSSY_OFFSET
    } else {
        BRANCH_OFFSET
    };
    guarded(opcode, guard)
        .with_field(offset, bytes.div_euclid(OFFSET_UNIT).cast_unsigned())
        .with_field(BRANCH_LOW, bytes.rem_euclid(OFFSET_UNIT).cast_unsigned())
}

/// The displacement from the end of instruction `from` to instruction `to`.
fn displacement(from: u32, to: u32) -> i64 {
    (i64::from(to) - i64::from(from) - 1) * 16
}

/// A branch at `from` to instruction `to`.
fn branch(opcode: u16, guard: u8, from: u32, to: u32) -> Instruction {
    branch_by(opcode, guard, displacement(from, to))
}

/// An unconditional `CALL` at `from` to `to`.
fn call(from: u32, to: u32) -> Instruction {
    branch(CALL_OPCODE, PT, from, to)
}

/// A `BRA` at `from` to `to`.
fn bra(from: u32, to: u32) -> Instruction {
    branch(BRA_OPCODE, PT, from, to)
}

/// A word from its four 32-bit words, low first, as PR #4588's tests write them.
fn native(words: [u32; 4]) -> Instruction {
    let [a, b, c, d] = words.map(u128::from);
    Instruction::from_bits(a | (b << 32) | (c << 64) | (d << 96))
}

/// The header test programs carry: distinct non-zero bytes.
fn header() -> Vec<u8> {
    (1..=HEADER_BYTES)
        .map(|byte| u8::try_from(byte % 251).unwrap() + 1)
        .collect()
}

/// A program of `stage`: its header, then `words`.
fn program(stage: GuestStage, words: &[Instruction]) -> Vec<u8> {
    let mut bytes = header();
    bytes.truncate(header_bytes(stage));
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Captures `words` as a compute program.
fn capture_words(words: &[Instruction]) -> Result<Capture, CaptureError> {
    capture(
        &program(GuestStage::Compute, words)[..],
        GuestStage::Compute,
    )
}

/// How many instructions capturing `words` keeps.
fn captured(words: &[Instruction]) -> u32 {
    capture_words(words).map(|c| c.instruction_count()).unwrap()
}

/// A reader that counts its reads and the bytes they ask for.
struct Counting {
    bytes: Vec<u8>,
    reads: Cell<u64>,
    read_bytes: Cell<u64>,
}

impl Counting {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            reads: Cell::new(0),
            read_bytes: Cell::new(0),
        }
    }
}

impl ProgramReader for Counting {
    fn read(&self, offset: u64, out: &mut [u8]) -> Result<(), Unreadable> {
        self.reads.set(self.reads.get() + 1);
        self.read_bytes
            .set(self.read_bytes.get() + out.len() as u64);
        self.bytes[..].read(offset, out)
    }
}

// ---------------------------------------------------------------------------------------------
// Layout facts and constants, against the reference's literals.

/// Every constant the capture reads, written out as PR #4588's literals: the opcodes of
/// `Observe` and `ampere_isa_table.h`, the bit positions of `Observe` and `instruction.h`, the
/// cap and the header sizes.
#[test]
fn layout_matches_the_reference() {
    assert_eq!(CALL_OPCODE, 0x944);
    assert_eq!(BSSY_OPCODE, 0x945);
    assert_eq!(BRA_OPCODE, 0x947);
    assert_eq!(RET_OPCODE, 0x950);
    assert_eq!(EXIT_OPCODE, 0x94d);
    assert_eq!(OPCODE, Field::new(0, 12));
    // `Bits(12, 4) == 7`: three register bits and the negation.
    assert_eq!(GUARD, Field::new(12, 3));
    assert_eq!(GUARD_NOT, Field::new(15, 1));
    assert_eq!(PT, 7);
    // `Bits(87, 4) == 7` and `!Bits(84, 2)`.
    assert_eq!(EXIT_PRED, Field::new(87, 3));
    assert_eq!(EXIT_PRED_NOT, Field::new(90, 1));
    assert_eq!(EXIT_KEEP_REF_COUNT, Field::new(84, 1));
    assert_eq!(EXIT_NO_AT_EXIT, Field::new(85, 1));
    // `words[1] | (words[2] & 0x3ffff) << 32`: bits 32..82, the low two below the offset.
    assert_eq!(BRANCH_LOW, Field::new(32, 2));
    assert_eq!(BRANCH_OFFSET, Field::new(34, 48));
    assert_eq!(BSSY_OFFSET, Field::new(34, 30));
    assert_eq!(BRANCH_MODE, Field::new(82, 14));
    let div = Instruction::default().with_field(BRANCH_MODE, BRA_DIV_MODE);
    assert_eq!(div.bits() >> 64, 0x0b80_0000);
    assert_eq!(BRA_DIV_LOW, 3);
    assert_eq!(OFFSET_UNIT, 4);
    assert_eq!(MAX_INSTRUCTIONS, 65_536);
    assert_eq!(header_bytes(GuestStage::Compute), 0);
    for stage in GRAPHICS {
        assert_eq!(header_bytes(stage), 128, "{stage:?}");
    }
}

#[test]
fn only_call_bssy_and_bra_have_a_displacement() {
    for opcode in [EXIT_OPCODE, RET_OPCODE, FILLER_OPCODE, 0x949, 0x946, 0x948] {
        let word = guarded(opcode, PT).with_field(BRANCH_OFFSET, 5);
        assert_eq!(branch_displacement(word), None, "{opcode:#x}");
    }
    for opcode in [CALL_OPCODE, BSSY_OPCODE, BRA_OPCODE] {
        assert_eq!(branch_displacement(branch_by(opcode, PT, 48)), Some(48));
    }
}

/// The offset field counts words, the low two bits are bytes, and each opcode reads its own
/// width: bits 64..82 belong to `CALL`'s and `BRA`'s offset and not to `BSSY`'s.
#[test]
fn each_branch_reads_its_own_offset_width() {
    let high = Field::new(64, 18);
    for opcode in [CALL_OPCODE, BRA_OPCODE] {
        // Bit 64 is bit 30 of the word offset: 2^30 words.
        let word = branch_by(opcode, PT, 32).with_field(high, 1);
        assert_eq!(branch_displacement(word), Some(32 + (1 << 30) * 4));
        // Bit 81 is the sign of the 48-bit offset.
        let word = branch_by(opcode, PT, 32).with_field(high, 1 << 17);
        assert_eq!(branch_displacement(word), Some(32 - (1 << 49)));
    }
    let bssy = branch_by(BSSY_OPCODE, PT, 32).with_field(high, 0x3_ffff);
    assert_eq!(branch_displacement(bssy), Some(32));
    // Bit 63 is BSSY's sign.
    let bssy = branch_by(BSSY_OPCODE, PT, -32);
    assert!(bssy.flag(Field::bit(63)));
    assert_eq!(branch_displacement(bssy), Some(-32));
    // The low two bits add bytes.
    assert_eq!(branch_displacement(branch_by(BRA_OPCODE, PT, 35)), Some(35));
    assert_eq!(
        branch_displacement(branch_by(BSSY_OPCODE, PT, 34)),
        Some(34)
    );
}

/// `BRA.DIV` (the exact mode bits and both low bits set) drops its two mode bits; one bit off in
/// either, or another opcode, keeps them as bytes.
#[test]
fn bra_div_drops_its_mode_bits() {
    let div = |opcode, mode, low| {
        branch_by(opcode, PT, 64)
            .with_field(BRANCH_MODE, mode)
            .with_field(BRANCH_LOW, low)
    };
    assert_eq!(
        branch_displacement(div(BRA_OPCODE, BRA_DIV_MODE, 3)),
        Some(64)
    );
    assert_eq!(
        branch_displacement(div(BRA_OPCODE, BRA_DIV_MODE, 2)),
        Some(66)
    );
    assert_eq!(
        branch_displacement(div(BRA_OPCODE, BRA_DIV_MODE, 1)),
        Some(65)
    );
    for bit in 0..BRANCH_MODE.width {
        let mode = BRA_DIV_MODE ^ (1 << bit);
        assert_eq!(
            branch_displacement(div(BRA_OPCODE, mode, 3)),
            Some(67),
            "{bit}"
        );
    }
    assert_eq!(
        branch_displacement(div(CALL_OPCODE, BRA_DIV_MODE, 3)),
        Some(67)
    );
    // A plain BRA with `PT` as its predicate (`0x03800000`) keeps them.
    let plain = 0x0380_0000 >> (BRANCH_MODE.start - 64);
    assert_eq!(branch_displacement(div(BRA_OPCODE, plain, 3)), Some(67));
}

proptest! {
    #![proptest_config(config())]

    /// Build then read: every displacement each branch can encode reads back, so a swapped or
    /// misplaced field fails.
    #[test]
    fn displacement_round_trips(
        bytes in -(1_i64 << 49)..(1_i64 << 49),
        small in i64::from(i32::MIN)..=i64::from(i32::MAX),
        guard in 0_u8..16,
    ) {
        for opcode in [CALL_OPCODE, BRA_OPCODE] {
            let word = branch_by(opcode, guard, bytes);
            // A BRA.DIV pattern needs the mode bits, which the encoder leaves clear.
            prop_assert_eq!(branch_displacement(word), Some(bytes));
            prop_assert_eq!(word.field(GUARD), u64::from(guard & 7));
            prop_assert_eq!(word.flag(GUARD_NOT), guard & 8 != 0);
        }
        prop_assert_eq!(branch_displacement(branch_by(BSSY_OPCODE, guard, small)), Some(small));
    }
}

// ---------------------------------------------------------------------------------------------
// PR #4588's own capture cases, word for word (`src/tests/video_core/native_shader.cpp`).

/// The final `EXIT` of the reference's `VertexTransform` program.
fn reference_exit() -> Instruction {
    native([0x0000_794d, 0, 0x0380_0000, 0x000f_ea00])
}

/// "Native capture distinguishes forward branches from call targets beyond main EXIT".
#[test]
fn reference_forward_branches_and_call_targets() {
    let mut calls = Scan::default();
    assert_eq!(
        calls.observe(native([0x7944, 48, 0x03c0_0000, 0]), 1),
        Ok(Step::Continue)
    );
    assert_eq!(calls.observe(reference_exit(), 2), Ok(Step::Continue));
    assert_eq!(
        calls.observe(native([0x0800_7950, 0, 0, 0]), 6),
        Ok(Step::Complete)
    );
    let mut branches = Scan::default();
    assert_eq!(
        branches.observe(native([0x7947, 48, 0x0380_0000, 0]), 0),
        Ok(Step::Continue)
    );
    assert_eq!(branches.observe(reference_exit(), 1), Ok(Step::Continue));
    assert_eq!(branches.observe(reference_exit(), 4), Ok(Step::Complete));
    let mut multiple = Scan::default();
    assert_eq!(
        multiple.observe(native([0x7944, 96, 0x03c0_0000, 0]), 1),
        Ok(Step::Continue)
    );
    assert_eq!(
        multiple.observe(native([0x7944, 128, 0x03c0_0000, 0]), 3),
        Ok(Step::Continue)
    );
    assert_eq!(multiple.observe(reference_exit(), 4), Ok(Step::Continue));
    assert_eq!(
        multiple.observe(native([0x0800_7950, 0, 0, 0]), 9),
        Ok(Step::Continue)
    );
    assert_eq!(
        multiple.observe(native([0x0800_7950, 0, 0, 0]), 13),
        Ok(Step::Complete)
    );
    let mut invalid = Scan::default();
    assert_eq!(
        invalid.observe(native([0x7947, 1, 0x0380_0000, 0]), 0),
        Err(CaptureError::MisalignedTarget {
            index: 0,
            target: 17
        })
    );
}

/// "Native capture treats predicated NOINC forward exits as branches".
#[test]
fn reference_predicated_noinc_calls_are_forward_branches() {
    let mut looped = Scan::default();
    assert_eq!(
        looped.observe(native([0x3944, 16, 0x03c0_0000, 0]), 0),
        Ok(Step::Continue)
    );
    assert_eq!(
        looped.observe(native([0x7947, 0xffff_ffe0, 0x0383_ffff, 0]), 1),
        Ok(Step::Continue)
    );
    assert_eq!(looped.observe(reference_exit(), 2), Ok(Step::Complete));
    let mut forward = Scan::default();
    assert_eq!(
        forward.observe(native([0x3944, 48, 0x03c0_0000, 0]), 0),
        Ok(Step::Continue)
    );
    assert_eq!(forward.observe(reference_exit(), 1), Ok(Step::Continue));
    assert_eq!(forward.observe(reference_exit(), 4), Ok(Step::Complete));
}

/// "Native EXIT retains its execution and branch predicates": an `EXIT` whose own predicate is
/// `P0` does not end the program, whatever its guard.
#[test]
fn reference_exit_keeps_its_predicates() {
    for execution in [7_u32, 9] {
        let words = [
            native([0x94d | (execution << 12), 0, 0, 0]),
            native([0x27802, 1, 0xf00, 0]),
            reference_exit(),
        ];
        assert_eq!(captured(&words), 3, "{execution}");
    }
}

/// "Native predicated loop exit is a forward transfer with no pending return".
#[test]
fn reference_predicated_loop_exit() {
    for predicate in [3_u32, 11] {
        let words = [
            native([0x0000_7802, 0, 0xf00, 0]),
            native([0x0000_7810, 1, 0x07ff_e0ff, 0]),
            native([0x0000_780c, 4, 0x03f6_6270, 0]),
            native([(predicate << 12) | 0x944, 16, 0x03c0_0000, 0]),
            native([0x7947, 0xffff_ffc0, 0x0383_ffff, 0]),
            reference_exit(),
        ];
        let mut scan = Scan::default();
        for (index, word) in (0..).zip(words) {
            let expected = if index == 5 {
                Step::Complete
            } else {
                Step::Continue
            };
            assert_eq!(
                scan.observe(word, index),
                Ok(expected),
                "{predicate} {index}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The EXIT rule.

#[test]
fn exit_ends_the_program_at_the_forward_target_not_before() {
    // The branch reaches instruction 3: an EXIT at 2 does not end it, one at 3 does.
    assert_eq!(captured(&[bra(0, 3), filler(), exit(), exit(), exit()]), 4);
    // An EXIT exactly at the target ends it.
    assert_eq!(captured(&[bra(0, 2), filler(), exit(), exit()]), 3);
    // A backward branch does not hold it.
    assert_eq!(captured(&[filler(), bra(1, 0), exit(), exit()]), 3);
    // The furthest target counts, not the last one seen.
    assert_eq!(
        captured(&[bra(0, 4), bra(1, 2), exit(), exit(), exit(), exit()]),
        5
    );
}

#[test]
fn exit_guard_must_be_pt() {
    for guard in 0..16 {
        let expected = if guard == PT { 1 } else { 2 };
        let word = exit()
            .with_field(GUARD, u64::from(guard & 7))
            .with_flag(GUARD_NOT, guard & 8 != 0);
        assert_eq!(captured(&[word, exit()]), expected, "guard {guard}");
    }
}

#[test]
fn exit_predicate_must_be_pt() {
    for pred in 0..16_u8 {
        let expected = if pred == PT { 1 } else { 2 };
        let word = exit()
            .with_field(EXIT_PRED, u64::from(pred & 7))
            .with_flag(EXIT_PRED_NOT, pred & 8 != 0);
        assert_eq!(captured(&[word, exit()]), expected, "predicate {pred}");
    }
}

#[test]
fn exit_mode_bits_must_be_clear() {
    for field in [EXIT_KEEP_REF_COUNT, EXIT_NO_AT_EXIT] {
        assert_eq!(
            captured(&[exit().with_flag(field, true), exit()]),
            2,
            "{field:?}"
        );
    }
    // Bits the rule does not read do not matter: the scheduling bits and bit 86.
    let other = exit()
        .with_field(Field::new(105, 23), 0x7f_ffff)
        .with_flag(Field::bit(86), true);
    assert_eq!(captured(&[other, exit()]), 1);
}

// ---------------------------------------------------------------------------------------------
// Calls, RET and library code.

#[test]
fn an_unconditional_call_defers_the_end_to_a_ret() {
    assert_eq!(captured(&[call(0, 3), exit(), filler(), ret()]), 4);
}

#[test]
fn ret_ends_only_at_or_past_every_call_target() {
    // The call reaches 4: a RET at 3 does not end the program, one at 4 does.
    assert_eq!(
        captured(&[call(0, 4), exit(), filler(), ret(), ret(), ret()]),
        5
    );
    // The furthest call target counts.
    assert_eq!(
        captured(&[call(0, 5), call(1, 3), exit(), ret(), ret(), ret(), ret()]),
        6
    );
}

#[test]
fn ret_ends_only_at_or_past_every_forward_target() {
    // A forward branch in the library reaches 5: a RET at 4 is past the call target but not the
    // branch target.
    assert_eq!(
        captured(&[call(0, 2), exit(), bra(2, 5), filler(), ret(), ret(), ret()]),
        6
    );
}

#[test]
fn ret_before_the_main_exit_does_not_end_the_program() {
    assert_eq!(captured(&[call(0, 2), filler(), ret(), exit(), ret()]), 5);
}

#[test]
fn ret_guard_must_be_pt() {
    for guard in 0..16 {
        let expected = if guard == PT { 3 } else { 4 };
        let word = guarded(RET_OPCODE, guard);
        assert_eq!(
            captured(&[call(0, 2), exit(), word, ret()]),
            expected,
            "guard {guard}"
        );
    }
}

/// Row 88: the register a `RET` returns through is not read; PR #4588's `RET R8` ends the
/// program.
#[test]
fn ret_register_is_not_read() {
    let ret_r8 = native([0x0800_7950, 0, 0, 0]);
    assert_eq!(captured(&[call(0, 2), exit(), ret_r8, ret()]), 3);
}

/// Row 88: library functions may EXIT; with a call pending, an EXIT does not end the program.
#[test]
fn a_library_exit_does_not_end_the_program() {
    assert_eq!(
        captured(&[call(0, 3), exit(), filler(), exit(), exit(), ret()]),
        6
    );
}

/// A call to instruction 0 reads as no call (`if (!last_call_target)`), so the next EXIT ends
/// the program.
#[test]
fn a_call_to_instruction_zero_is_not_counted() {
    assert_eq!(captured(&[filler(), call(1, 0), exit(), ret()]), 3);
    // A call to instruction 1 is counted, for contrast.
    assert_eq!(captured(&[filler(), call(1, 1), exit(), ret()]), 4);
}

/// Row 67: a `CALL` with any guard but `PT` (a predicated `CALL.REL.NOINC`) is a forward
/// branch: the EXIT at its target ends the program with no RET.
#[test]
fn a_predicated_call_is_a_forward_branch() {
    for guard in 0..16 {
        if guard == PT {
            continue;
        }
        let words = [
            branch(CALL_OPCODE, guard, 0, 3),
            exit(),
            filler(),
            exit(),
            ret(),
        ];
        assert_eq!(captured(&words), 4, "guard {guard}");
    }
    let words = [call(0, 3), exit(), filler(), exit(), ret()];
    assert_eq!(captured(&words), 5);
}

#[test]
fn bssy_and_predicated_bra_are_forward_branches() {
    assert_eq!(
        captured(&[branch(BSSY_OPCODE, PT, 0, 2), exit(), exit()]),
        3
    );
    assert_eq!(captured(&[branch(BRA_OPCODE, 0, 0, 2), exit(), exit()]), 3);
}

// ---------------------------------------------------------------------------------------------
// Branch targets.

#[test]
fn branch_targets_at_each_edge() {
    let last = MAX_INSTRUCTIONS - 1;
    // Instruction 0 and the last instruction under the cap are targets.
    assert_eq!(branch_target(5, displacement(5, 0)), Ok(0));
    assert_eq!(branch_target(5, displacement(5, last)), Ok(last));
    // One instruction past either side is not.
    assert_eq!(
        branch_target(5, displacement(5, 0) - 16),
        Err(CaptureError::TargetOutside {
            index: 5,
            target: -16
        })
    );
    assert_eq!(
        branch_target(5, displacement(5, MAX_INSTRUCTIONS)),
        Err(CaptureError::TargetOutside {
            index: 5,
            target: i64::from(MAX_INSTRUCTIONS) * 16,
        })
    );
    // A byte past an instruction's start is not, on either side of the cap.
    for byte in [1, 4, 8, 15] {
        assert_eq!(
            branch_target(5, displacement(5, last) + byte),
            Err(CaptureError::MisalignedTarget {
                index: 5,
                target: i64::from(last) * 16 + byte,
            })
        );
    }
    // The largest displacements a branch encodes are refused, not overflowed.
    assert_eq!(
        branch_target(last, (1 << 49) - 16),
        Err(CaptureError::TargetOutside {
            index: last,
            target: i64::from(MAX_INSTRUCTIONS) * 16 + (1 << 49) - 16,
        })
    );
    assert_eq!(
        branch_target(last, -(1 << 49)),
        Err(CaptureError::TargetOutside {
            index: last,
            target: i64::from(MAX_INSTRUCTIONS) * 16 - (1 << 49),
        })
    );
}

/// The reference's order: below zero, then not a whole instruction, then past the cap.
#[test]
fn target_checks_run_in_the_reference_order() {
    // Below zero and misaligned: below zero wins.
    assert_eq!(
        branch_target(0, -24),
        Err(CaptureError::TargetOutside {
            index: 0,
            target: -8
        })
    );
    // Past the cap and misaligned: misaligned wins.
    let past = displacement(0, MAX_INSTRUCTIONS) + 4;
    assert_eq!(
        branch_target(0, past),
        Err(CaptureError::MisalignedTarget {
            index: 0,
            target: i64::from(MAX_INSTRUCTIONS) * 16 + 4,
        })
    );
}

#[test]
fn a_refused_target_names_its_branch() {
    let words = [
        filler(),
        filler(),
        filler(),
        branch_by(BSSY_OPCODE, 2, -80),
        exit(),
    ];
    assert_eq!(
        capture_words(&words),
        Err(CaptureError::TargetOutside {
            index: 3,
            target: -16
        })
    );
    let words = [filler(), filler(), branch_by(CALL_OPCODE, PT, 2), exit()];
    assert_eq!(
        capture_words(&words),
        Err(CaptureError::MisalignedTarget {
            index: 2,
            target: 50
        })
    );
}

// ---------------------------------------------------------------------------------------------
// The cap.

/// `count - 1` fillers and a final EXIT, as a graphics program.
fn long_program(count: u32) -> Vec<u8> {
    let mut words = vec![filler(); count as usize - 1];
    words.push(exit());
    program(GuestStage::Pixel, &words)
}

#[test]
fn a_program_ending_on_the_last_instruction_under_the_cap_is_captured() {
    let bytes = long_program(MAX_INSTRUCTIONS);
    let captured = capture(&bytes[..], GuestStage::Pixel).unwrap();
    assert_eq!(captured.instruction_count(), MAX_INSTRUCTIONS);
    assert_eq!(captured.bytes(), &bytes[..]);
}

/// The reference keeps the first 65,536 instructions; starquake rejects (a listed divergence).
#[test]
fn a_program_with_no_end_within_the_cap_is_rejected() {
    let reader = Counting::new(long_program(MAX_INSTRUCTIONS + 1));
    assert_eq!(
        capture(&reader, GuestStage::Pixel),
        Err(CaptureError::NoEnd)
    );
    // It never reads the instruction past the cap: the header and the cap's instructions.
    assert_eq!(reader.reads.get(), u64::from(MAX_INSTRUCTIONS) + 1);
    assert_eq!(
        CaptureError::NoEnd.to_string(),
        "no instruction within the first 65536 ends the program"
    );
}

#[test]
fn a_branch_to_the_last_instruction_under_the_cap_is_captured() {
    let mut words = vec![bra(0, MAX_INSTRUCTIONS - 1)];
    words.resize(MAX_INSTRUCTIONS as usize - 1, exit());
    words.push(exit());
    assert_eq!(captured(&words), MAX_INSTRUCTIONS);
}

/// Row 121: the scan reads each instruction once, so its work is linear in the program's
/// length; on a program near the cap with a forward branch in every other instruction, it reads
/// the header and each instruction exactly once.
#[test]
fn the_scan_is_linear_near_the_cap() {
    let count = MAX_INSTRUCTIONS - 1;
    let words: Vec<_> = (0..count)
        .map(|index| match index {
            _ if index == count - 1 => exit(),
            _ if index % 2 == 0 => bra(index, count - 1),
            _ => exit(),
        })
        .collect();
    let reader = Counting::new(program(GuestStage::Vertex, &words));
    let captured = capture(&reader, GuestStage::Vertex).unwrap();
    assert_eq!(captured.instruction_count(), count);
    assert_eq!(reader.reads.get(), u64::from(count) + 1);
    assert_eq!(
        reader.read_bytes.get(),
        u64::from(count) * SIZE + HEADER_BYTES as u64
    );
}

// ---------------------------------------------------------------------------------------------
// The header and the reader seam.

#[test]
fn the_header_is_split_off_by_stage() {
    let words = [filler(), exit()];
    for stage in GRAPHICS.into_iter().chain([GuestStage::Compute]) {
        let bytes = program(stage, &words);
        let captured = capture(&bytes[..], stage).unwrap();
        assert_eq!(captured.stage(), stage);
        assert_eq!(
            captured.header(),
            &header()[..header_bytes(stage)],
            "{stage:?}"
        );
        assert_eq!(captured.code(), &bytes[header_bytes(stage)..], "{stage:?}");
        assert_eq!(captured.bytes(), &bytes[..]);
        assert_eq!(captured.instructions().collect::<Vec<_>>(), words);
    }
    // Read as compute, the same bytes start with the header's words.
    let bytes = program(GuestStage::Pixel, &words);
    let compute = capture(&bytes[..], GuestStage::Compute).unwrap();
    assert_eq!(compute.header(), []);
    assert_eq!(compute.instruction_count(), 10);
}

#[test]
fn an_unreadable_header_is_rejected() {
    let bytes = header();
    let short = &bytes[..HEADER_BYTES - 1];
    assert_eq!(
        capture(short, GuestStage::Vertex),
        Err(CaptureError::HeaderUnreadable { offset: 127 })
    );
    assert_eq!(
        capture(short, GuestStage::Vertex).unwrap_err().to_string(),
        "the program header cannot be read at byte 0x7f"
    );
}

#[test]
fn an_unreadable_instruction_is_rejected() {
    for stage in [GuestStage::Geometry, GuestStage::Compute] {
        let mut bytes = program(stage, &[filler(), filler(), exit()]);
        bytes.pop();
        let end = bytes.len() as u64;
        assert_eq!(
            capture(&bytes[..], stage),
            Err(CaptureError::InstructionUnreadable {
                index: 2,
                offset: end
            })
        );
        // Cut at the instruction's start, the first byte read is the one missing.
        bytes.truncate(bytes.len() - 15);
        assert_eq!(
            capture(&bytes[..], stage),
            Err(CaptureError::InstructionUnreadable {
                index: 2,
                offset: end - 15
            })
        );
    }
    assert_eq!(
        CaptureError::InstructionUnreadable {
            index: 2,
            offset: 0xaf
        }
        .to_string(),
        "instruction 2 cannot be read at byte 0xaf"
    );
}

#[test]
fn a_byte_slice_reads_up_to_its_end() {
    let bytes: &[u8] = &[1, 2, 3, 4];
    let mut out = [0; 4];
    assert_eq!(bytes.read(0, &mut out), Ok(()));
    assert_eq!(out, [1, 2, 3, 4]);
    let mut two = [0; 2];
    assert_eq!(bytes.read(2, &mut two), Ok(()));
    assert_eq!(two, [3, 4]);
    assert_eq!(bytes.read(4, &mut []), Ok(()));
    assert_eq!(bytes.read(3, &mut two), Err(Unreadable { offset: 4 }));
    assert_eq!(bytes.read(5, &mut two), Err(Unreadable { offset: 5 }));
    assert_eq!(
        bytes.read(u64::MAX, &mut two),
        Err(Unreadable { offset: u64::MAX })
    );
    assert_eq!(
        Unreadable { offset: 5 }.to_string(),
        "program byte 0x5 cannot be read"
    );
}

#[test]
fn error_messages_are_plain_words() {
    assert_eq!(
        CaptureError::TargetOutside {
            index: 3,
            target: -16
        }
        .to_string(),
        "instruction 3 branches to code byte -16, outside the first 65536 instructions"
    );
    assert_eq!(
        CaptureError::MisalignedTarget {
            index: 2,
            target: 50
        }
        .to_string(),
        "instruction 2 branches to code byte 50, which does not start an instruction"
    );
}

// ---------------------------------------------------------------------------------------------
// The hash.

/// The function is XXH3-128: its value for empty input with seed 0 is the one xxHash publishes
/// (`xxhsum -H2` of an empty file: `99aa06d3014798d86001c324468d497f`).
#[test]
fn the_hash_is_xxh3_128() {
    assert_eq!(
        ContentHash::of(0, &[]).value(),
        0x99aa_06d3_0147_98d8_6001_c324_468d_497f
    );
}

/// The golden values: a fixed program, as compute and as graphics. A change of hash function or
/// seed fails here, and would orphan every stored cache entry.
#[test]
fn the_hash_of_a_fixed_program_is_pinned() {
    let words = [filler(), call(1, 3), exit(), ret()];
    let compute = capture_words(&words).unwrap();
    assert_eq!(
        compute.hash(),
        ContentHash::new(0xf4ff_da79_cef2_67fd_9dea_3eb2_6fc2_e97d)
    );
    let bytes = program(GuestStage::Pixel, &words);
    let pixel = capture(&bytes[..], GuestStage::Pixel).unwrap();
    assert_eq!(
        pixel.hash(),
        ContentHash::new(0xd911_6eba_8a72_2fdb_461b_4a38_a79f_23cf)
    );
}

/// A capture's hash is the hash of all its bytes, header included, and a change to any one
/// byte moves it.
#[test]
fn the_hash_covers_every_byte() {
    let words = [filler(), bra(1, 3), exit(), exit()];
    let bytes = program(GuestStage::Vertex, &words);
    let captured = capture(&bytes[..], GuestStage::Vertex).unwrap();
    assert_eq!(captured.hash(), ContentHash::of(HEADER_BYTES, &bytes));
    for offset in 0..bytes.len() {
        let mut changed = bytes.clone();
        changed[offset] ^= 0x80;
        assert_ne!(
            ContentHash::of(HEADER_BYTES, &changed),
            captured.hash(),
            "byte {offset}"
        );
    }
}

/// The same bytes captured as compute and as graphics hash differently.
#[test]
fn the_hash_depends_on_the_header_length() {
    let bytes = program(GuestStage::Pixel, &[exit()]);
    let compute_words: Vec<_> = bytes
        .as_chunks::<{ Instruction::SIZE }>()
        .0
        .iter()
        .map(|word| Instruction::from_le_bytes(*word))
        .collect();
    // The header's words do not end a program, so compute captures the same bytes.
    let compute = capture(&bytes[..], GuestStage::Compute).unwrap();
    let pixel = capture(&bytes[..], GuestStage::Pixel).unwrap();
    assert_eq!(compute.bytes(), pixel.bytes());
    assert_eq!(compute.instructions().collect::<Vec<_>>(), compute_words);
    assert_ne!(compute.hash(), pixel.hash());
}

#[test]
fn a_hash_prints_as_hex_and_keeps_its_value() {
    let hash = ContentHash::new(0x0123_4567_89ab_cdef_0011_2233_4455_6677);
    assert_eq!(hash.value(), 0x0123_4567_89ab_cdef_0011_2233_4455_6677);
    assert_eq!(hash.to_string(), "0123456789abcdef0011223344556677");
    assert_eq!(ContentHash::new(1).to_string(), format!("{:0>32}", "1"));
}

// ---------------------------------------------------------------------------------------------
// Jump tables.

/// A graphics program that the linear scan ends at instruction 2, with a tail through `tail`.
fn table_program(tail: &[Instruction]) -> Vec<u8> {
    let mut words = vec![filler(), filler(), exit()];
    words.extend_from_slice(tail);
    program(GuestStage::Pixel, &words)
}

/// The table value naming instruction `index` with the graphics base.
fn value(index: u32) -> u32 {
    u32::try_from(HEADER_BYTES).unwrap() + index * 16
}

#[test]
fn table_entries_at_each_edge() {
    for base in [0, 128] {
        assert_eq!(table_entry(0, base), None);
        assert_eq!(table_entry(base + 16, base), Some(1));
        assert_eq!(table_entry(base + 17, base), None);
        assert_eq!(table_entry(base + 8, base), None);
        let last = base + (MAX_INSTRUCTIONS - 1) * 16;
        assert_eq!(table_entry(last, base), Some(MAX_INSTRUCTIONS - 1));
        assert_eq!(table_entry(last + 16, base), None);
        assert_eq!(table_entry(u32::MAX, base), None);
    }
    // Below the base names nothing; at the base, instruction 0.
    assert_eq!(table_entry(112, 128), None);
    assert_eq!(table_entry(128, 128), Some(0));
    // Value 0 names nothing even when the base is 0.
    assert_eq!(table_entry(0, 0), None);
}

#[test]
fn no_table_changes_nothing() {
    let bytes = table_program(&[filler(), exit()]);
    let mut captured = capture(&bytes[..], GuestStage::Pixel).unwrap();
    let before = captured.clone();
    let tables = captured.resolve_jump_tables(&bytes[..], &[]);
    assert_eq!(
        tables,
        JumpTables {
            base: 0,
            extension: Extension::NotNeeded,
            targets: vec![],
        }
    );
    assert_eq!(captured, before);
}

#[test]
fn the_base_is_the_header_unless_a_value_lies_below_it() {
    let bytes = table_program(&[]);
    let resolve = |tables: &[&[u32]]| {
        let mut captured = capture(&bytes[..], GuestStage::Pixel).unwrap();
        captured.resolve_jump_tables(&bytes[..], tables).base
    };
    assert_eq!(resolve(&[&[value(1), value(2)]]), 128);
    assert_eq!(resolve(&[&[value(1)], &[127]]), 0);
    assert_eq!(resolve(&[&[value(1)], &[128]]), 128);
    // Zero does not count as below the header.
    assert_eq!(resolve(&[&[0, value(1)]]), 128);
    assert_eq!(resolve(&[&[]]), 128);
    // A compute program has no header, so its base is 0.
    let compute = program(GuestStage::Compute, &[exit()]);
    let mut captured = capture(&compute[..], GuestStage::Compute).unwrap();
    assert_eq!(captured.resolve_jump_tables(&compute[..], &[&[16]]).base, 0);
}

#[test]
fn targets_inside_the_capture_need_no_extension() {
    let bytes = table_program(&[filler(), exit()]);
    let mut captured = capture(&bytes[..], GuestStage::Pixel).unwrap();
    let before = captured.clone();
    let tables = captured.resolve_jump_tables(&bytes[..], &[&[value(2), value(0), value(2)]]);
    assert_eq!(tables.extension, Extension::NotNeeded);
    assert_eq!(tables.targets, vec![vec![0, 2]]);
    assert_eq!(captured, before);
}

#[test]
fn a_target_at_the_capture_end_extends_it() {
    // The capture holds instructions 0..3; a target of 3 is past it.
    let bytes = table_program(&[filler(), exit(), exit()]);
    let mut captured = capture(&bytes[..], GuestStage::Pixel).unwrap();
    let tables = captured.resolve_jump_tables(&bytes[..], &[&[value(1)], &[value(3)]]);
    assert_eq!(tables.extension, Extension::Extended { from: 3 });
    assert_eq!(captured.instruction_count(), 5);
    assert_eq!(captured.bytes(), &bytes[..bytes.len() - 16]);
    assert_eq!(tables.targets, vec![vec![1], vec![3]]);
}

#[test]
fn an_extension_ends_only_past_the_furthest_target() {
    // Instructions 4 and 7 end the program; with the furthest target at 5 the capture runs to 7.
    let tail = [filler(), exit(), filler(), filler(), exit(), exit()];
    let bytes = table_program(&tail);
    let mut captured = capture(&bytes[..], GuestStage::Pixel).unwrap();
    let tables = captured.resolve_jump_tables(&bytes[..], &[&[value(5), value(3)]]);
    assert_eq!(tables.extension, Extension::Extended { from: 3 });
    assert_eq!(captured.instruction_count(), 8);
    assert_eq!(tables.targets, vec![vec![3, 5]]);
    // With the furthest target exactly on the ending instruction, it ends there.
    let mut captured = capture(&bytes[..], GuestStage::Pixel).unwrap();
    let tables = captured.resolve_jump_tables(&bytes[..], &[&[value(4)]]);
    assert_eq!(tables.extension, Extension::Extended { from: 3 });
    assert_eq!(captured.instruction_count(), 5);
}

#[test]
fn targets_are_distinct_ascending_and_inside() {
    let bytes = table_program(&[filler(), exit(), filler()]);
    let mut captured = capture(&bytes[..], GuestStage::Pixel).unwrap();
    let tables = captured.resolve_jump_tables(
        &bytes[..],
        &[
            &[value(4), value(1), 0, value(4), value(3) + 4],
            &[value(0)],
            &[],
        ],
    );
    assert_eq!(tables.extension, Extension::Extended { from: 3 });
    assert_eq!(tables.targets, vec![vec![1, 4], vec![0], vec![]]);
    // Values past the capture and past the cap name nothing.
    assert_eq!(
        captured.jump_targets(128, &[value(5), value(MAX_INSTRUCTIONS), value(2)]),
        [2]
    );
    // The base given is the one used.
    assert_eq!(captured.jump_targets(0, &[16, 64]), [1, 4]);
}

/// An extension that cannot end leaves the capture as it was, for each reason.
#[test]
fn an_abandoned_extension_changes_nothing() {
    let unreadable = table_program(&[filler(), filler()]);
    let outside = table_program(&[filler(), branch_by(BRA_OPCODE, PT, -128), exit()]);
    // Fillers through the cap and one past it.
    let capped = table_program(&vec![filler(); MAX_INSTRUCTIONS as usize - 2]);
    let cases = [
        (
            &unreadable,
            Extension::Abandoned(CaptureError::InstructionUnreadable {
                index: 5,
                offset: 128 + 5 * 16,
            }),
        ),
        (
            &outside,
            Extension::Abandoned(CaptureError::TargetOutside {
                index: 4,
                target: -48,
            }),
        ),
        (&capped, Extension::Abandoned(CaptureError::NoEnd)),
    ];
    for (bytes, expected) in cases {
        let mut captured = capture(&bytes[..], GuestStage::Pixel).unwrap();
        let before = captured.clone();
        let tables = captured.resolve_jump_tables(&bytes[..], &[&[value(3), value(1)]]);
        assert_eq!(tables.extension, expected);
        assert_eq!(captured, before);
        assert_eq!(captured.hash(), before.hash());
        assert_eq!(tables.targets, vec![vec![1]]);
    }
}

/// The scan's state is restored too: after an abandoned tail with a call in it, an extension
/// over another tail behaves as on a fresh capture.
#[test]
fn an_abandoned_extension_restores_the_scan() {
    let first = table_program(&[call(3, 20)]);
    let second = table_program(&[filler(), exit()]);
    let mut captured = capture(&first[..], GuestStage::Pixel).unwrap();
    assert!(matches!(
        captured
            .resolve_jump_tables(&first[..], &[&[value(3)]])
            .extension,
        Extension::Abandoned(CaptureError::InstructionUnreadable { index: 4, .. })
    ));
    let tables = captured.resolve_jump_tables(&second[..], &[&[value(3)]]);
    assert_eq!(tables.extension, Extension::Extended { from: 3 });
    assert_eq!(captured.instruction_count(), 5);
}

/// The extension reads only the tail, once.
#[test]
fn an_extension_reads_only_the_tail() {
    let reader = Counting::new(table_program(&[filler(), filler(), exit()]));
    let mut captured = capture(&reader, GuestStage::Pixel).unwrap();
    assert_eq!(reader.reads.get(), 4);
    let tables = captured.resolve_jump_tables(&reader, &[&[value(4)]]);
    assert_eq!(tables.extension, Extension::Extended { from: 3 });
    assert_eq!(reader.reads.get(), 7);
}

// ---------------------------------------------------------------------------------------------
// Arbitrary input.

/// Words biased toward the instructions the scan reads, with small displacements, so that many
/// generated programs end.
fn any_word() -> impl Strategy<Value = Instruction> {
    let control = (
        prop::sample::select(vec![
            CALL_OPCODE,
            BSSY_OPCODE,
            BRA_OPCODE,
            EXIT_OPCODE,
            RET_OPCODE,
            FILLER_OPCODE,
        ]),
        prop::sample::select(vec![PT, PT, PT, 0, 3, 15]),
        -64_i64..64,
    )
        .prop_map(|(opcode, guard, words)| match opcode {
            EXIT_OPCODE => exit().with_field(GUARD, u64::from(guard & 7)),
            RET_OPCODE => guarded(RET_OPCODE, guard),
            FILLER_OPCODE => filler(),
            _ => branch_by(opcode, guard, words * 16),
        });
    prop_oneof![3 => control, 1 => any::<u128>().prop_map(Instruction::from_bits)]
}

proptest! {
    #![proptest_config(config())]

    /// Arbitrary bytes never panic, as either kind of program, nor do arbitrary table words.
    #[test]
    fn arbitrary_bytes_never_panic(
        bytes in prop::collection::vec(any::<u8>(), 0..1024),
        values in prop::collection::vec(any::<u32>(), 0..8),
    ) {
        for stage in [GuestStage::Vertex, GuestStage::Compute] {
            if let Ok(mut captured) = capture(&bytes[..], stage) {
                let _ = captured.hash();
                let _ = captured.resolve_jump_tables(&bytes[..], &[&values]);
            }
        }
    }

    /// When a capture ends, every relative branch in it reaches an instruction inside it, and
    /// the last instruction is one that can end a program.
    #[test]
    fn every_target_lies_inside_a_finished_capture(
        words in prop::collection::vec(any_word(), 1..64),
    ) {
        if let Ok(captured) = capture_words(&words) {
            let count = captured.instruction_count();
            prop_assert!(count as usize <= words.len());
            for (index, word) in (0..).zip(captured.instructions()) {
                if let Some(displacement) = branch_displacement(word) {
                    let target = branch_target(index, displacement);
                    prop_assert!(matches!(target, Ok(target) if target < count));
                }
            }
            let last = captured.instructions().last().unwrap();
            let opcode = last.field(OPCODE);
            prop_assert!(opcode == u64::from(EXIT_OPCODE) || opcode == u64::from(RET_OPCODE));
        }
    }
}
