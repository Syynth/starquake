//! Constant propagation: each rule's fold and the case one step off it that must stay, the
//! departures from Eden, and the folds that do not keep the interpreter's result.

use super::{parse, ran};
use crate::interp::Machine;
use crate::ir::{Attribute, Function};
use crate::opt::constant::{
    INTEGER_ATTRIBUTES, QUAD_CLAMP, QUAD_SEGMENTATION, SWIZZLE_DX, SWIZZLE_DY,
};
use crate::opt::{Pass, run};

/// The straight-line `body` (instruction lines) with a `return`, as a function.
fn function(body: &str) -> Function {
    parse(&format!("block0:\n{body}    return\n"))
}

/// `body` after constant propagation, as text without the block header and the return.
fn fold(body: &str) -> String {
    let mut function = function(body);
    run(&mut function, Pass::ConstantPropagation).expect("runs");
    let text = function.to_string();
    text.strip_prefix("block0:\n")
        .and_then(|rest| rest.strip_suffix("    return\n"))
        .map(str::to_owned)
        .unwrap_or(text)
}

/// Checks that constant propagation leaves `body` as it is (in canonical form).
fn unchanged(body: &str) {
    let canonical = {
        let text = function(body).to_string();
        text.strip_prefix("block0:\n")
            .and_then(|rest| rest.strip_suffix("    return\n"))
            .map(str::to_owned)
            .unwrap_or(text)
    };
    assert_eq!(fold(body), canonical);
}

/// A machine whose constant buffer 0 holds `words`.
fn machine(words: &[u32]) -> Machine {
    let mut machine = Machine::new();
    machine
        .cbufs
        .insert(0, words.iter().flat_map(|w| w.to_le_bytes()).collect());
    machine
}

/// The machines a run of `body` leaves before and after constant propagation.
fn before_and_after(body: &str, start: &Machine) -> (Machine, Machine) {
    let before = function(body);
    let mut after = before.clone();
    run(&mut after, Pass::ConstantPropagation).expect("runs");
    (ran(&before, start), ran(&after, start))
}

#[test]
fn reserved_reads_fold() {
    assert_eq!(
        fold(
            "    %0: u32 = GetRegister rz
    %1: u1 = GetPred pt
    %2: u32 = GetUniformRegister urz
    %3: u1 = GetUniformPred upt
    %4: u32 = GetRegister r0
    %5: u1 = GetPred p0
    %6: u32 = GetUniformRegister ur0
    %7: u1 = GetUniformPred up0
"
        ),
        "    %0: opaque = Identity u32 0x0
    %1: opaque = Identity u1 true
    %2: opaque = Identity u32 0x0
    %3: opaque = Identity u1 true
    %4: u32 = GetRegister r0
    %5: u1 = GetPred p0
    %6: u32 = GetUniformRegister ur0
    %7: u1 = GetUniformPred up0
"
    );
}

#[test]
fn integer_adds_fold_normalise_and_combine() {
    assert_eq!(
        fold(
            "    %0: u32 = IAdd32 u32 0xffffffff, u32 3
    %1: u32 = GetCbufU32 u32 0, u32 0
    %2: u32 = IAdd32 u32 4, %1
    %3: u32 = IAdd32 %2, u32 5
    %4: u32 = IAdd32 u32 6, %3
    %5: u32 = IAdd32 %1, u32 0
    %6: u32 = IAdd32 %1, %1
    %7: u32 = IAdd32 u32 0, %1
"
        ),
        "    %0: opaque = Identity u32 0x2
    %1: u32 = GetCbufU32 u32 0x0, u32 0x0
    %2: u32 = IAdd32 %1, u32 0x4
    %3: u32 = IAdd32 %1, u32 0x9
    %4: u32 = IAdd32 %1, u32 0xf
    %5: opaque = Identity %1
    %6: u32 = IAdd32 %1, %1
    %7: opaque = Identity %1
"
    );
    assert_eq!(
        fold(
            "    %0: u64 = IAdd64 u64 0xffffffffffffffff, u64 2
    %1: u32x2 = GetCbufU32x2 u32 0, u32 0
    %2: u64 = PackUint2x32 %1
    %3: u64 = IAdd64 u64 1, %2
    %4: u64 = IAdd64 %3, u64 2
    %5: u64 = IAdd64 %2, u64 0
"
        ),
        "    %0: opaque = Identity u64 0x1
    %1: u32x2 = GetCbufU32x2 u32 0x0, u32 0x0
    %2: u64 = PackUint2x32 %1
    %3: u64 = IAdd64 %2, u64 0x1
    %4: u64 = IAdd64 %2, u64 0x3
    %5: opaque = Identity %2
"
    );
}

#[test]
fn a_combine_needs_the_same_opcode_and_an_immediate_inside() {
    // An inner ISub32, and an inner IAdd32 of two values, do not combine; the immediate still
    // moves to the right.
    assert_eq!(
        fold(
            "    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: u32 = ISub32 %0, u32 1
    %2: u32 = IAdd32 u32 2, %1
    %3: u32 = IAdd32 %0, %0
    %4: u32 = IAdd32 %3, u32 2
"
        ),
        "    %0: u32 = GetCbufU32 u32 0x0, u32 0x0
    %1: u32 = ISub32 %0, u32 0x1
    %2: u32 = IAdd32 %1, u32 0x2
    %3: u32 = IAdd32 %0, %0
    %4: u32 = IAdd32 %3, u32 0x2
"
    );
}

#[test]
fn pseudo_operations_stop_folds() {
    // A carry read of the add keeps it, immediates and all (Eden's
    // `HasAssociatedPseudoOperation`); the same for an all-immediate fold.
    unchanged(
        "    %0: u32 = IAdd32 u32 1, u32 2
    %1: u1 = GetCarryFromOp %0
    %2: u32 = IMul32 u32 3, u32 4
    %3: u1 = GetZeroFromOp %2
    %4: u32 = GetCbufU32 u32 0, u32 0
    %5: u32 = IAdd32 %4, u32 0
    %6: u1 = GetOverflowFromOp %5
",
    );
}

/// Eden's `FoldXmadMultiply` pattern, with `{n}` for the immediates in order: the shift, the
/// high extract's offset and count, the low extract's offset and count.
const XMAD_MULTIPLY: &str = "    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: u32 = GetCbufU32 u32 0, u32 4
    %2: u32 = BitFieldUExtract %0, u32 {1}, u32 {2}
    %3: u32 = IMul32 %2, %1
    %4: u32 = ShiftLeftLogical32 %3, u32 {0}
    %5: u32 = BitFieldUExtract %0, u32 {3}, u32 {4}
    %6: u32 = IMul32 %5, %1
    %7: u32 = IAdd32 %4, %6
    %8 = SetRegister r0, %7
";

/// `pattern` with `{index}` replaced by `values[index]`.
fn fill(pattern: &str, values: &[u32]) -> String {
    let mut text = pattern.to_owned();
    for (index, value) in values.iter().enumerate() {
        text = text.replace(&format!("{{{index}}}"), &value.to_string());
    }
    text
}

#[test]
fn the_xmad_multiply_folds_to_one_multiply() {
    let text = fill(XMAD_MULTIPLY, &[16, 16, 16, 0, 16]);
    let out = fold(&text);
    assert!(
        out.contains("    %9: u32 = IMul32 %0, %1\n    %7: opaque = Identity %9\n"),
        "{out}"
    );
    for (a, b) in [
        (0x1234_5678, 0x9abc_def0),
        (0xffff_ffff, 3),
        (7, 0x0001_0001),
    ] {
        let (before, after) = before_and_after(&text, &machine(&[a, b]));
        assert_eq!(after, before);
        assert_eq!(before.regs[0], u32::wrapping_mul(a, b));
    }
}

#[test]
fn the_xmad_multiply_needs_every_immediate() {
    let good = [16, 16, 16, 0, 16];
    for index in 0..good.len() {
        let mut values = good;
        values[index] ^= 1;
        let text = fill(XMAD_MULTIPLY, &values);
        assert!(!fold(&text).contains("%9"), "folded with {values:?}");
    }
    // The factors must agree: a different b in the low half, or a different a.
    let text = fill(XMAD_MULTIPLY, &good);
    for (from, to) in [
        ("%6: u32 = IMul32 %5, %1", "%6: u32 = IMul32 %5, %0"),
        (
            "%5: u32 = BitFieldUExtract %0",
            "%5: u32 = BitFieldUExtract %1",
        ),
    ] {
        assert!(!fold(&text.replace(from, to)).contains("%9"), "{to}");
    }
}

/// Eden's `FoldXmadMultiplyAdd` pattern, named as Eden's comment names it, with `{n}` for its
/// fifteen immediates in order of appearance.
const XMAD_MULTIPLY_ADD: &str = "    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: u32 = GetCbufU32 u32 0, u32 4
    %2: u32 = GetCbufU32 u32 0, u32 8
    %6: u32 = BitFieldUExtract %1, u32 {0}, u32 {1}
    %7: u32 = BitFieldUExtract %0, u32 {2}, u32 {3}
    %8: u32 = IMul32 %6, %7
    %10: u32 = BitFieldUExtract %0, u32 {4}, u32 {5}
    %11: u32 = BitFieldInsert %8, %10, u32 {6}, u32 {7}
    %15: u32 = BitFieldUExtract %1, u32 {8}, u32 {9}
    %16: u32 = BitFieldUExtract %0, u32 {10}, u32 {11}
    %17: u32 = IMul32 %15, %16
    %18: u32 = IAdd32 %17, %2
    %22: u32 = BitFieldUExtract %1, u32 {12}, u32 {13}
    %23: u32 = BitFieldUExtract %11, u32 {14}, u32 {15}
    %24: u32 = IMul32 %22, %23
    %25: u32 = ShiftLeftLogical32 %24, u32 {16}
    %26: u32 = ShiftLeftLogical32 %11, u32 {17}
    %27: u32 = IAdd32 %26, %18
    %28: u32 = IAdd32 %25, %27
    %29 = SetRegister r0, %28
";

/// The immediates of [`XMAD_MULTIPLY_ADD`] as Eden's pattern has them.
const XMAD_ADD_IMMEDIATES: [u32; 18] = [
    0, 16, 16, 16, 0, 16, 16, 16, 0, 16, 0, 16, 16, 16, 16, 16, 16, 16,
];

#[test]
fn the_xmad_multiply_add_folds_to_a_multiply_and_an_add() {
    let text = fill(XMAD_MULTIPLY_ADD, &XMAD_ADD_IMMEDIATES);
    let out = fold(&text);
    assert!(
        out.contains(
            "    %30: u32 = IMul32 %0, %1\n    %31: u32 = IAdd32 %30, %2\n    %28: opaque = Identity %31\n"
        ),
        "{out}"
    );
    for (a, b, c) in [
        (0x1234_5678, 0x9abc_def0, 0x0f0f_0f0f),
        (0xffff_ffff, 0xffff_ffff, 1),
        (0x8000, 0x8000, 0),
    ] {
        let (before, after) = before_and_after(&text, &machine(&[a, b, c]));
        assert_eq!(after, before);
        assert_eq!(before.regs[0], a.wrapping_mul(b).wrapping_add(c));
    }
}

#[test]
fn the_xmad_multiply_add_needs_every_immediate() {
    for index in 0..XMAD_ADD_IMMEDIATES.len() {
        let mut values = XMAD_ADD_IMMEDIATES;
        values[index] ^= 1;
        let text = fill(XMAD_MULTIPLY_ADD, &values);
        assert!(
            !fold(&text).contains("%31"),
            "folded with immediate {index} changed"
        );
    }
}

#[test]
fn the_xmad_multiply_add_needs_matching_operands() {
    let text = fill(XMAD_MULTIPLY_ADD, &XMAD_ADD_IMMEDIATES);
    // Each of the three reads of a, the three of b, and the link from %26 to %11, in turn.
    for (from, to) in [
        (
            "%7: u32 = BitFieldUExtract %0",
            "%7: u32 = BitFieldUExtract %2",
        ),
        (
            "%10: u32 = BitFieldUExtract %0",
            "%10: u32 = BitFieldUExtract %2",
        ),
        (
            "%16: u32 = BitFieldUExtract %0",
            "%16: u32 = BitFieldUExtract %2",
        ),
        (
            "%6: u32 = BitFieldUExtract %1",
            "%6: u32 = BitFieldUExtract %2",
        ),
        (
            "%15: u32 = BitFieldUExtract %1",
            "%15: u32 = BitFieldUExtract %2",
        ),
        (
            "%22: u32 = BitFieldUExtract %1",
            "%22: u32 = BitFieldUExtract %2",
        ),
        (
            "%26: u32 = ShiftLeftLogical32 %11",
            "%26: u32 = ShiftLeftLogical32 %8",
        ),
        (
            "%11: u32 = BitFieldInsert %8",
            "%11: u32 = BitFieldInsert %7",
        ),
    ] {
        assert!(!fold(&text.replace(from, to)).contains("%31"), "{to}");
    }
    // Reads of the same constant buffer word count as the same operand (Eden's `AreEqual`).
    let copy = text
        .replace(
            "%16: u32 = BitFieldUExtract %0",
            "%16: u32 = BitFieldUExtract %3",
        )
        .replace(
            "    %2: u32 = GetCbufU32 u32 0, u32 8\n",
            "    %2: u32 = GetCbufU32 u32 0, u32 8\n    %3: u32 = GetCbufU32 u32 0, u32 0\n",
        );
    assert!(
        fold(&copy).contains("%30: u32 = IMul32 %0, %1"),
        "{}",
        fold(&copy)
    );
}

#[test]
fn isub_folds_immediates_and_equal_constant_buffer_reads() {
    assert_eq!(
        fold(
            "    %0: u32 = ISub32 u32 1, u32 2
    %1: u32 = GetCbufU32 u32 0, u32 4
    %2: u32 = GetCbufU32 u32 0, u32 4
    %3: u32 = GetCbufU32 u32 0, u32 8
    %4: u32 = ISub32 %1, %2
    %5: u32 = ISub32 %1, %3
    %6: u32 = GetCbufU32 u32 1, u32 4
    %7: u32 = ISub32 %1, %6
    %8: u32 = ISub32 %1, u32 3
"
        ),
        "    %0: opaque = Identity u32 0xffffffff
    %1: u32 = GetCbufU32 u32 0x0, u32 0x4
    %2: u32 = GetCbufU32 u32 0x0, u32 0x4
    %3: u32 = GetCbufU32 u32 0x0, u32 0x8
    %4: opaque = Identity u32 0x0
    %5: u32 = ISub32 %1, %3
    %6: u32 = GetCbufU32 u32 0x1, u32 0x4
    %7: u32 = ISub32 %1, %6
    %8: u32 = ISub32 %1, u32 0x3
"
    );
}

#[test]
fn isub_folds_only_the_add_on_the_left() {
    // (x + c) - c is x, and (c + 5) - c is 5; c - (x + c) is -x and stays (Eden folds it to x);
    // (x + d) - c with d another word stays.
    let body = "    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: u32 = GetCbufU32 u32 0, u32 4
    %2: u32 = GetCbufU32 u32 0, u32 4
    %3: u32 = IAdd32 %0, %1
    %4: u32 = ISub32 %3, %2
    %5: u32 = IAdd32 %1, u32 5
    %6: u32 = ISub32 %5, %2
    %7: u32 = ISub32 %2, %3
    %8: u32 = GetCbufU32 u32 0, u32 8
    %9: u32 = IAdd32 %0, %8
    %10: u32 = ISub32 %9, %2
    %11 = SetRegister r0, %4
    %12 = SetRegister r1, %6
    %13 = SetRegister r2, %7
    %14 = SetRegister r3, %10
";
    let out = fold(body);
    assert!(out.contains("    %4: opaque = Identity %0\n"), "{out}");
    assert!(out.contains("    %6: opaque = Identity u32 0x5\n"), "{out}");
    assert!(out.contains("    %7: u32 = ISub32 %2, %3\n"), "{out}");
    assert!(out.contains("    %10: u32 = ISub32 %9, %2\n"), "{out}");
    let (before, after) = before_and_after(body, &machine(&[10, 3, 100]));
    assert_eq!(after, before);
    assert_eq!(before.regs[..4], [10, 5, 10_u32.wrapping_neg(), 107]);
}

#[test]
fn selects_with_an_immediate_condition_fold() {
    assert_eq!(
        fold(
            "    %0: u32 = SelectU32 u1 true, u32 1, u32 2
    %1: f32 = SelectF32 u1 false, f32 0x3f800000, f32 0x40000000
    %2: u1 = GetPred p0
    %3: u32 = SelectU32 %2, u32 1, u32 2
"
        ),
        "    %0: opaque = Identity u32 0x1
    %1: opaque = Identity f32 0x40000000
    %2: u1 = GetPred p0
    %3: u32 = SelectU32 %2, u32 0x1, u32 0x2
"
    );
}

#[test]
fn float_folds_honour_controls() {
    // 1.0 + 1e-8: round to nearest gives 1.0, round up the next float (Eden folds both to
    // 1.0); 1.5 * 2.0 needs no rounding.
    assert_eq!(
        fold(
            "    %0: f32 = FPAdd32 f32 0x3f800000, f32 0x322bcc77
    %1: f32 = FPAdd32 f32 0x3f800000, f32 0x322bcc77 {rp}
    %2: f32 = FPMul32 f32 0x3fc00000, f32 0x40000000
    %3: f32 = FPNeg32 f32 0x3f800000
"
        ),
        "    %0: opaque = Identity f32 0x3f800000
    %1: opaque = Identity f32 0x3f800001
    %2: opaque = Identity f32 0x40400000
    %3: opaque = Identity f32 0xbf800000
"
    );
}

#[test]
fn float_adds_of_zero_fold() {
    assert_eq!(
        fold(
            "    %0: f32 = GetCbufF32 u32 0, u32 0
    %1: f32 = FPAdd32 %0, f32 0x0
    %2: f32 = FPAdd32 %0, f32 0x80000000
    %3: f32 = FPAdd32 f32 0x0, %0
    %4: f32 = FPAdd32 %0, f32 0x1
    %5: f32 = FPAdd32 %0, %0
"
        ),
        "    %0: f32 = GetCbufF32 u32 0x0, u32 0x0
    %1: opaque = Identity %0
    %2: opaque = Identity %0
    %3: opaque = Identity %0
    %4: f32 = FPAdd32 %0, f32 0x1
    %5: f32 = FPAdd32 %0, %0
"
    );
}

#[test]
fn float_add_of_zero_is_not_exact() {
    let body = "    %0: f32 = GetCbufF32 u32 0, u32 0
    %1: f32 = FPAdd32 %0, f32 0x0
    %2: u32 = BitCastU32F32 %1
    %3 = SetRegister r0, %2
";
    // -0.0 + 0.0 is +0.0; the fold gives -0.0.
    let (before, after) = before_and_after(body, &machine(&[0x8000_0000]));
    assert_eq!((before.regs[0], after.regs[0]), (0, 0x8000_0000));
    // A denormal under flush to zero: flushed before, kept after.
    let ftz = body.replace("f32 0x0", "f32 0x0 {ftz}");
    let (before, after) = before_and_after(&ftz, &machine(&[1]));
    assert_eq!((before.regs[0], after.regs[0]), (0, 1));
}

#[test]
fn float_multiplies_fold_derivatives_and_interpolation() {
    assert_eq!(
        fold(
            "    %0: f32 = GetAttribute a0x80, u32 0
    %1: f32 = YDirection
    %2: f32 = DPdyFine %0
    %3: f32 = FPMul32 %1, %2
    %4: f32 = FPMul32 %2, %1
    %5: f32 = FPMul32 %1, %0
    %6: f32 = GetAttribute a0x84, u32 0
    %7: f32 = FPMul32 %0, %6
    %8: f32 = FPRecip32 %6
    %9: f32 = FPMul32 %7, %8
    %10: f32 = FPMul32 %7, %8 {no_contraction}
    %11: f32 = GetAttribute a0x88, u32 0
    %12: f32 = FPRecip32 %11
    %13: f32 = FPMul32 %7, %12
    %14: f32 = FPMul32 %1, %2 {no_contraction}
"
        ),
        "    %0: f32 = GetAttribute a0x80, u32 0x0
    %1: f32 = YDirection
    %2: f32 = DPdyFine %0
    %3: opaque = Identity %2
    %4: opaque = Identity %2
    %5: f32 = FPMul32 %1, %0
    %6: f32 = GetAttribute a0x84, u32 0x0
    %7: f32 = FPMul32 %0, %6
    %8: f32 = FPRecip32 %6
    %9: opaque = Identity %0
    %10: f32 = FPMul32 %7, %8 {no_contraction}
    %11: f32 = GetAttribute a0x88, u32 0x0
    %12: f32 = FPRecip32 %11
    %13: f32 = FPMul32 %7, %12
    %14: f32 = FPMul32 %1, %2 {no_contraction}
"
    );
}

#[test]
fn interpolation_fold_is_not_exact() {
    // (1.0 * 41.0) * (1 / 41.0) rounds to the float below 1.0; the fold gives 1.0.
    let body = "    %0: f32 = GetAttribute a0x80, u32 0
    %1: f32 = GetAttribute a0x84, u32 0
    %2: f32 = FPMul32 %0, %1
    %3: f32 = FPRecip32 %1
    %4: f32 = FPMul32 %2, %3
    %5: u32 = BitCastU32F32 %4
    %6 = SetRegister r0, %5
";
    let mut start = Machine::new();
    start.attributes.insert((Attribute(0x20), 0), 0x3f80_0000);
    start.attributes.insert((Attribute(0x21), 0), 0x4224_0000);
    let (before, after) = before_and_after(body, &start);
    assert_eq!((before.regs[0], after.regs[0]), (0x3f7f_ffff, 0x3f80_0000));
}

#[test]
fn logical_operations_fold() {
    assert_eq!(
        fold(
            "    %0: u1 = GetPred p0
    %1: u1 = LogicalAnd %0, u1 true
    %2: u1 = LogicalAnd u1 false, %0
    %3: u1 = LogicalOr %0, u1 true
    %4: u1 = LogicalOr u1 false, %0
    %5: u1 = LogicalAnd u1 true, u1 false
    %6: u1 = LogicalOr u1 false, u1 false
    %7: u1 = LogicalNot u1 false
    %8: u1 = LogicalNot %0
    %9: u1 = LogicalNot %8
    %10: u1 = LogicalAnd %0, %8
    %11: u1 = LogicalOr %0, %8
"
        ),
        "    %0: u1 = GetPred p0
    %1: opaque = Identity %0
    %2: opaque = Identity u1 false
    %3: opaque = Identity u1 true
    %4: opaque = Identity %0
    %5: opaque = Identity u1 false
    %6: opaque = Identity u1 false
    %7: opaque = Identity u1 true
    %8: u1 = LogicalNot %0
    %9: opaque = Identity %0
    %10: u1 = LogicalAnd %0, %8
    %11: u1 = LogicalOr %0, %8
"
    );
}

#[test]
fn logical_and_or_combine_nested_immediates() {
    // (p && true) folds first; ((p || q) || false) normalises and folds to (p || q).
    assert_eq!(
        fold(
            "    %0: u1 = GetPred p0
    %1: u1 = GetPred p1
    %2: u1 = LogicalOr %0, %1
    %3: u1 = LogicalOr u1 false, %2
    %4: u1 = LogicalAnd %0, u1 true
    %5: u1 = LogicalAnd u1 true, %4
"
        ),
        "    %0: u1 = GetPred p0
    %1: u1 = GetPred p1
    %2: u1 = LogicalOr %0, %1
    %3: opaque = Identity %2
    %4: opaque = Identity %0
    %5: opaque = Identity %4
"
    );
}

#[test]
fn integer_operations_fold_when_every_argument_is_an_immediate() {
    // Each against the reference's definition, the expected value written out.
    let cases: [(&str, &str); 17] = [
        ("u32 = IMul32 u32 0x10000, u32 0x10001", "u32 0x10000"),
        (
            "u32 = ShiftRightArithmetic32 u32 0x80000000, u32 31",
            "u32 0xffffffff",
        ),
        (
            "u32 = ShiftRightArithmetic32 u32 0x40000000, u32 30",
            "u32 0x1",
        ),
        ("u1 = SLessThan u32 0xffffffff, u32 0", "u1 true"),
        ("u1 = ULessThan u32 0xffffffff, u32 0", "u1 false"),
        ("u1 = SLessThanEqual u32 5, u32 5", "u1 true"),
        ("u1 = ULessThanEqual u32 6, u32 5", "u1 false"),
        ("u1 = SGreaterThan u32 0, u32 0xffffffff", "u1 true"),
        ("u1 = UGreaterThan u32 0, u32 0xffffffff", "u1 false"),
        ("u1 = SGreaterThanEqual u32 0x80000000, u32 0", "u1 false"),
        ("u1 = UGreaterThanEqual u32 0x80000000, u32 0", "u1 true"),
        ("u1 = IEqual u32 3, u32 3", "u1 true"),
        ("u1 = INotEqual u32 3, u32 3", "u1 false"),
        ("u32 = BitwiseAnd32 u32 0xff00, u32 0x0ff0", "u32 0xf00"),
        ("u32 = BitwiseOr32 u32 0xff00, u32 0x0ff0", "u32 0xfff0"),
        ("u32 = BitwiseXor32 u32 0xff00, u32 0x0ff0", "u32 0xf0f0"),
        (
            "u32 = BitFieldSExtract u32 0x00f0, u32 4, u32 4",
            "u32 0xffffffff",
        ),
    ];
    for (op, result) in cases {
        assert_eq!(
            fold(&format!("    %0: {op}\n")),
            format!("    %0: opaque = Identity {result}\n"),
            "{op}"
        );
    }
    assert_eq!(
        fold("    %0: u32 = BitFieldUExtract u32 0xabcd, u32 4, u32 8\n"),
        "    %0: opaque = Identity u32 0xbc\n"
    );
    // One argument a value: nothing folds.
    unchanged(
        "    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: u32 = IMul32 %0, u32 2
    %2: u1 = IEqual %0, u32 2
    %3: u32 = BitFieldUExtract %0, u32 0, u32 4
    %4: u32 = ShiftRightArithmetic32 %0, u32 1
",
    );
}

#[test]
fn undefined_folds_are_left_alone() {
    // Each edge: the last defined case folds, the first undefined one stays.
    assert_eq!(
        fold("    %0: u32 = BitFieldUExtract u32 0xffffffff, u32 1, u32 31\n"),
        "    %0: opaque = Identity u32 0x7fffffff\n"
    );
    unchanged("    %0: u32 = BitFieldUExtract u32 0xffffffff, u32 2, u32 31\n");
    unchanged("    %0: u32 = BitFieldUExtract u32 0xffffffff, u32 0, u32 32\n");
    assert_eq!(
        fold("    %0: u32 = BitFieldSExtract u32 0x80000000, u32 31, u32 1\n"),
        "    %0: opaque = Identity u32 0xffffffff\n"
    );
    unchanged("    %0: u32 = BitFieldSExtract u32 0x80000000, u32 31, u32 2\n");
    unchanged("    %0: u32 = BitFieldSExtract u32 0x80000000, u32 4, u32 0\n");
    assert_eq!(
        fold("    %0: u32 = BitFieldInsert u32 0, u32 0x7fffffff, u32 31, u32 31\n"),
        "    %0: opaque = Identity u32 0x80000000\n"
    );
    unchanged("    %0: u32 = BitFieldInsert u32 0, u32 1, u32 0, u32 32\n");
    unchanged("    %0: u32 = BitFieldInsert u32 0, u32 1, u32 32, u32 1\n");
    unchanged("    %0: u32 = ShiftRightArithmetic32 u32 0x80000000, u32 32\n");
}

#[test]
fn bit_field_insert_masks_the_inserted_value() {
    // Eden's fold gives 0xff here.
    assert_eq!(
        fold("    %0: u32 = BitFieldInsert u32 0, u32 0xff, u32 0, u32 4\n"),
        "    %0: opaque = Identity u32 0xf\n"
    );
}

#[test]
fn bit_casts_fold() {
    assert_eq!(
        fold(
            "    %0: f32 = BitCastF32U32 u32 0x3f800000
    %1: u32 = BitCastU32F32 f32 0x40000000
    %2: u32 = GetCbufU32 u32 1, u32 8
    %3: f32 = BitCastF32U32 %2
    %4: f32 = GetAttribute a0x80, u32 0
    %5: u32 = BitCastU32F32 %4
    %6: f32 = BitCastF32U32 %5
    %7: f32 = GetAttribute a0x2f8, u32 1
    %8: u32 = BitCastU32F32 %7
    %9: f32 = BitCastF32U32 %8
"
        ),
        "    %0: opaque = Identity f32 0x3f800000
    %1: opaque = Identity u32 0x40000000
    %2: u32 = GetCbufU32 u32 0x1, u32 0x8
    %3: f32 = GetCbufF32 u32 0x1, u32 0x8
    %4: f32 = GetAttribute a0x80, u32 0x0
    %5: u32 = BitCastU32F32 %4
    %6: opaque = Identity %4
    %7: f32 = GetAttribute a0x2f8, u32 0x1
    %8: u32 = GetAttributeU32 a0x2f8, u32 0x1
    %9: f32 = BitCastF32U32 %8
"
    );
}

#[test]
fn the_integer_attributes_are_edens() {
    // attribute.h: PrimitiveId 24, InstanceId 190, VertexId 191, BaseInstance 256,
    // BaseVertex 257, DrawID 258.
    assert_eq!(
        INTEGER_ATTRIBUTES,
        [24, 190, 191, 256, 257, 258].map(Attribute)
    );
    for attribute in [23, 25, 189, 192, 255, 259]
        .map(Attribute)
        .into_iter()
        .chain(INTEGER_ATTRIBUTES)
    {
        let body = format!(
            "    %0: f32 = GetAttribute {attribute}, u32 0\n    %1: u32 = BitCastU32F32 %0\n"
        );
        let integer = INTEGER_ATTRIBUTES.contains(&attribute);
        assert_eq!(
            fold(&body).contains("GetAttributeU32"),
            integer,
            "{attribute}"
        );
    }
}

#[test]
fn inverse_packings_fold() {
    assert_eq!(
        fold(
            "    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: f32x2 = UnpackHalf2x16 %0
    %2: u32 = PackHalf2x16 %1
    %3: f16x2 = UnpackFloat2x16 %0
    %4: u32 = PackFloat2x16 %3
    %5: f32x2 = UnpackHalf2x16 %2
    %6: f16x2 = UnpackFloat2x16 %4
    %7: u32 = PackHalf2x16 %5
    %8: f32x2 = UnpackHalf2x16 u32 0x3c00
"
        ),
        "    %0: u32 = GetCbufU32 u32 0x0, u32 0x0
    %1: f32x2 = UnpackHalf2x16 %0
    %2: opaque = Identity %0
    %3: f16x2 = UnpackFloat2x16 %0
    %4: opaque = Identity %0
    %5: f32x2 = UnpackHalf2x16 %2
    %6: f16x2 = UnpackFloat2x16 %4
    %7: opaque = Identity %2
    %8: f32x2 = UnpackHalf2x16 u32 0x3c00
"
    );
}

#[test]
fn the_unpack_of_a_pack_folds_too() {
    // UnpackFloat2x16(PackFloat2x16 v) and UnpackHalf2x16(PackHalf2x16 v) are v; an argument
    // that resolves to an immediate is not folded (Eden's `FoldInverseFunc` returns on
    // `IsImmediate`, which looks through identities).
    assert_eq!(
        fold(
            "    %0: f32x2 = CompositeConstructF32x2 f32 0x3f800000, f32 0x40000000
    %1: u32 = PackHalf2x16 %0
    %2: f32x2 = UnpackHalf2x16 %1
    %9: f16 = ConvertF16F32 f32 0x3f800000
    %3: f16x2 = CompositeConstructF16x2 %9, %9
    %4: u32 = PackFloat2x16 %3
    %5: f16x2 = UnpackFloat2x16 %4
    %6: opaque = Identity u32 0x3c00
    %7: f16x2 = UnpackFloat2x16 %6
"
        ),
        "    %0: f32x2 = CompositeConstructF32x2 f32 0x3f800000, f32 0x40000000
    %1: u32 = PackHalf2x16 %0
    %2: opaque = Identity %0
    %9: f16 = ConvertF16F32 f32 0x3f800000
    %3: f16x2 = CompositeConstructF16x2 %9, %9
    %4: u32 = PackFloat2x16 %3
    %5: opaque = Identity %3
    %6: opaque = Identity u32 0x3c00
    %7: f16x2 = UnpackFloat2x16 %6
"
    );
}

#[test]
fn half_round_trip_fold_is_not_exact() {
    // 0.1 does not survive half precision; the fold keeps it.
    let body = "    %0: f32 = GetCbufF32 u32 0, u32 0
    %1: f32x2 = CompositeConstructF32x2 %0, %0
    %2: u32 = PackHalf2x16 %1
    %3: f32x2 = UnpackHalf2x16 %2
    %4: f32 = CompositeExtractF32x2 %3, u32 0
    %5: u32 = BitCastU32F32 %4
    %6 = SetRegister r0, %5
";
    let tenth = 0.1_f32.to_bits();
    let (before, after) = before_and_after(body, &machine(&[tenth]));
    assert_ne!(before.regs[0], tenth);
    assert_eq!(after.regs[0], tenth);
}

#[test]
fn composite_extracts_fold_through_inserts_and_constructs() {
    assert_eq!(
        fold(
            "    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: u32 = GetCbufU32 u32 0, u32 4
    %2: u32x3 = CompositeConstructU32x3 %0, %1, u32 7
    %3: u32x3 = CompositeInsertU32x3 %2, u32 9, u32 1
    %4: u32x3 = CompositeInsertU32x3 %3, %1, u32 2
    %5: u32 = CompositeExtractU32x3 %4, u32 0
    %6: u32 = CompositeExtractU32x3 %4, u32 1
    %7: u32 = CompositeExtractU32x3 %4, u32 2
    %8: u32 = CompositeExtractU32x3 %2, u32 2
    %9: u32 = CompositeExtractU32x3 %4, u32 3
    %10: u32 = CompositeExtractU32x3 %4, %0
    %11: u32x3 = CompositeInsertU32x3 %2, u32 9, %0
    %12: u32 = CompositeExtractU32x3 %11, u32 0
"
        ),
        "    %0: u32 = GetCbufU32 u32 0x0, u32 0x0
    %1: u32 = GetCbufU32 u32 0x0, u32 0x4
    %2: u32x3 = CompositeConstructU32x3 %0, %1, u32 0x7
    %3: u32x3 = CompositeInsertU32x3 %2, u32 0x9, u32 0x1
    %4: u32x3 = CompositeInsertU32x3 %3, %1, u32 0x2
    %5: opaque = Identity %0
    %6: opaque = Identity u32 0x9
    %7: opaque = Identity %1
    %8: opaque = Identity u32 0x7
    %9: u32 = CompositeExtractU32x3 %4, u32 0x3
    %10: u32 = CompositeExtractU32x3 %4, %0
    %11: u32x3 = CompositeInsertU32x3 %2, u32 0x9, %0
    %12: u32 = CompositeExtractU32x3 %11, u32 0x0
"
    );
}

#[test]
fn every_composite_extract_pairs_with_its_own_construct() {
    for (shape, scalar, value) in [
        ("U32x2", "u32", "u32 0x5"),
        ("U32x3", "u32", "u32 0x5"),
        ("U32x4", "u32", "u32 0x5"),
        ("F32x2", "f32", "f32 0x5"),
        ("F32x3", "f32", "f32 0x5"),
        ("F32x4", "f32", "f32 0x5"),
    ] {
        let count: usize = shape[4..].parse().expect("a count");
        let args = vec![value; count].join(", ");
        let lower = shape.to_lowercase();
        let body = format!(
            "    %0: {lower} = CompositeConstruct{shape} {args}
    %1: {scalar} = CompositeExtract{shape} %0, u32 1
    %2: {lower} = CompositeInsert{shape} %0, {value}, u32 0
    %3: {scalar} = CompositeExtract{shape} %2, u32 0
"
        );
        let out = fold(&body);
        assert!(
            out.contains(&format!("%1: opaque = Identity {value}")),
            "{out}"
        );
        assert!(
            out.contains(&format!("%3: opaque = Identity {value}")),
            "{out}"
        );
    }
    // The half-precision extracts, whose components no immediate has, through a value.
    for count in 2..=4 {
        let parts = vec!["%0"; count].join(", ");
        let body = format!(
            "    %9: f16x2 = UnpackFloat2x16 u32 0
    %0: f16 = CompositeExtractF16x2 %9, u32 0
    %1: f16x{count} = CompositeConstructF16x{count} {parts}
    %2: f16 = CompositeExtractF16x{count} %1, u32 1
"
        );
        assert!(
            fold(&body).contains("%2: opaque = Identity %0"),
            "f16x{count}"
        );
    }
}

#[test]
fn a_composite_extract_of_another_shape_does_not_fold() {
    // A U32x2 extract's operand is never a U32x3 construct in verified IR; through a phi, a
    // value that is no operation, nothing folds.
    let text = "block0:
    %0: u32x2 = CompositeConstructU32x2 u32 1, u32 2
    jump block1
block1:
    %1: u32x2 = Phi [block0: %0]
    %2: u32 = CompositeExtractU32x2 %1, u32 0
    return
";
    let mut function = parse(text);
    run(&mut function, Pass::ConstantPropagation).expect("runs");
    assert!(
        function
            .to_string()
            .contains("%2: u32 = CompositeExtractU32x2 %1, u32 0x0")
    );
}

/// A derivative pattern: `FSwizzleAdd(bitcast(shuffle(bitcast(%0))), %0, swizzle)`.
fn swizzle(shuffle: &str, swizzle: u32) -> String {
    format!(
        "    %0: f32 = GetAttribute a0x80, u32 0
    %1: u32 = BitCastU32F32 %0
    %2: u32 = {shuffle}
    %3: f32 = BitCastF32U32 %2
    %4: f32 = FSwizzleAdd %3, %0, u32 {swizzle}
"
    )
}

#[test]
fn swizzled_adds_of_quad_butterflies_become_derivatives() {
    let dx = fold(&swizzle("QuadSwap %1, u32 0", SWIZZLE_DX));
    assert!(
        dx.ends_with("    %5: f32 = DPdxFine %0\n    %4: opaque = Identity %5\n"),
        "{dx}"
    );
    let dy = fold(&swizzle("QuadSwap %1, u32 1", SWIZZLE_DY));
    assert!(
        dy.ends_with("    %5: f32 = DPdyFine %0\n    %4: opaque = Identity %5\n"),
        "{dy}"
    );
    let butterfly =
        format!("ShuffleButterfly %1, u32 1, u32 {QUAD_CLAMP}, u32 {QUAD_SEGMENTATION}");
    assert!(fold(&swizzle(&butterfly, SWIZZLE_DX)).contains("DPdxFine %0"));
}

#[test]
fn swizzled_adds_need_the_matching_lane_and_swizzle() {
    for (shuffle, sw) in [
        // The lane does not match the swizzle.
        ("QuadSwap %1, u32 1".to_owned(), SWIZZLE_DX),
        ("QuadSwap %1, u32 0".to_owned(), SWIZZLE_DY),
        // Another swizzle.
        ("QuadSwap %1, u32 0".to_owned(), SWIZZLE_DX + 1),
        // A butterfly that is not a quad's.
        (
            format!(
                "ShuffleButterfly %1, u32 1, u32 {}, u32 {QUAD_SEGMENTATION}",
                QUAD_CLAMP + 1
            ),
            SWIZZLE_DX,
        ),
        (
            format!(
                "ShuffleButterfly %1, u32 1, u32 {QUAD_CLAMP}, u32 {}",
                QUAD_SEGMENTATION + 1
            ),
            SWIZZLE_DX,
        ),
        // QuadSwap direction 0xffffffff wraps to lane mask 0.
        ("QuadSwap %1, u32 0xffffffff".to_owned(), SWIZZLE_DX),
    ] {
        let out = fold(&swizzle(&shuffle, sw));
        assert!(!out.contains("DPd"), "{shuffle} {sw:#x}: {out}");
    }
    // The shuffled value must be the second argument.
    let other = swizzle("QuadSwap %1, u32 0", SWIZZLE_DX)
        .replace("FSwizzleAdd %3, %0", "FSwizzleAdd %3, %5")
        .replace("    %4", "    %5: f32 = GetAttribute a0x84, u32 0\n    %4");
    assert!(!fold(&other).contains("DPd"));
}

#[test]
fn swizzled_adds_compare_immediates_by_bits() {
    // The second argument and the shuffled value may be the same bits as f32 and u32.
    let body = format!(
        "    %2: u32 = QuadSwap u32 0x3f800000, u32 0
    %3: f32 = BitCastF32U32 %2
    %4: f32 = FSwizzleAdd %3, f32 0x3f800000, u32 {SWIZZLE_DX}
"
    );
    assert!(
        fold(&body).contains("DPdxFine f32 0x3f800000"),
        "{}",
        fold(&body)
    );
    let off = body.replace(
        "FSwizzleAdd %3, f32 0x3f800000",
        "FSwizzleAdd %3, f32 0x3f800001",
    );
    assert!(!fold(&off).contains("DPd"));
}

#[test]
fn folds_see_through_identities() {
    // Constant propagation runs after identity removal, but its own folds leave identities:
    // a later fold reads through them, as Eden's `IsImmediate` does.
    assert_eq!(
        fold(
            "    %0: u32 = IAdd32 u32 1, u32 2
    %1: u32 = IAdd32 %0, u32 4
    %2: u1 = IEqual %1, u32 7
    %3: u1 = LogicalNot %2
    %4: u32 = SelectU32 %3, u32 1, u32 2
"
        ),
        "    %0: opaque = Identity u32 0x3
    %1: opaque = Identity u32 0x7
    %2: opaque = Identity u1 true
    %3: opaque = Identity u1 false
    %4: opaque = Identity u32 0x2
"
    );
}

#[test]
fn a_chain_of_identities_is_followed_when_one_folds_later() {
    // %2 adds 0 to %1, which is an Identity of %0; %3 adds 0 to %2. Each resolves to %0.
    assert_eq!(
        fold(
            "    %0: u32 = GetCbufU32 u32 0, u32 0
    %1: opaque = Identity %0
    %2: u32 = IAdd32 %1, u32 0
    %3: u32 = IAdd32 %2, u32 0
    %4: u32 = IAdd32 %3, u32 1
    %5: u32 = IAdd32 %4, u32 1
"
        ),
        "    %0: u32 = GetCbufU32 u32 0x0, u32 0x0
    %1: opaque = Identity %0
    %2: opaque = Identity %1
    %3: opaque = Identity %2
    %4: u32 = IAdd32 %3, u32 0x1
    %5: u32 = IAdd32 %3, u32 0x2
"
    );
}

#[test]
fn blocks_are_folded_in_reverse_post_order() {
    // block2 comes before block1 in reverse post order, after it in the arena: block1's add
    // of block2's sum folds only if block2 is visited first.
    let text = "block0:
    jump block2
block1:
    %2: u32 = IAdd32 %1, u32 4
    %3 = SetRegister r0, %2
    return
block2:
    %1: u32 = IAdd32 u32 1, u32 2
    jump block1
";
    let mut function = parse(text);
    run(&mut function, Pass::ConstantPropagation).expect("runs");
    let out = function.to_string();
    assert!(out.contains("%2: opaque = Identity u32 0x7"), "{out}");
}

#[test]
fn a_remembered_end_that_folds_later_is_followed() {
    // Unreachable blocks are visited in arena order: block1's add resolves %4 to %3 before
    // block2 folds %3 to 3; block2's add must then follow %4 through %3 to the 3.
    let text = "block0:
    return
block1:
    %5: u32 = IAdd32 %4, u32 0
    return
block2:
    %3: u32 = IAdd32 u32 1, u32 2
    %4: opaque = Identity %3
    %6: u32 = IAdd32 %4, u32 4
    return
";
    let mut function = parse(text);
    run(&mut function, Pass::ConstantPropagation).expect("runs");
    let out = function.to_string();
    assert!(out.contains("%6: opaque = Identity u32 0x7"), "{out}");
}
