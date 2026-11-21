//! The double-precision instructions: register, uniform, constant-buffer and immediate pairs,
//! their edges, the arithmetic and its controls, and `DSETP`'s comparison table.

use super::half::COMPARISONS;
use super::*;

/// Writes `value` to the register pair at `reg`, the low word first.
pub(in super::super) fn set_double(machine: &mut Machine, reg: u8, value: f64) {
    let bits = value.to_bits();
    let index = usize::from(reg);
    machine.regs[index] = (bits & 0xffff_ffff) as u32;
    machine.regs[index + 1] = (bits >> 32) as u32;
}

/// The double in the register pair at `reg`, as its bits.
pub(in super::super) fn get_double(machine: &Machine, reg: u8) -> u64 {
    let index = usize::from(reg);
    u64::from(machine.regs[index + 1]) << 32 | u64::from(machine.regs[index])
}

/// A double's words, low first.
fn words(value: f64) -> [u32; 2] {
    let bits = value.to_bits();
    [(bits & 0xffff_ffff) as u32, (bits >> 32) as u32]
}

/// A double source with modifiers.
fn dreg(index: u8, abs: bool, neg: bool) -> AluSrc {
    reg_mod(index, abs, neg)
}

/// A machine with 1.5 in `R2`, 2.25 in `R4`, -1.5 in `R6`, 2.0 in `UR4`, and in constant
/// buffer 1 4.0 at byte 8 and 0.25 at byte 16 (each double at a multiple of 8, as NAK emits
/// them, its high word four bytes after its low word, with other words before).
fn machine() -> Machine {
    let mut machine = Machine::new();
    set_double(&mut machine, 2, 1.5);
    set_double(&mut machine, 4, 2.25);
    set_double(&mut machine, 6, -1.5);
    let [low, high] = words(2.0);
    machine.uniform_regs[4] = low;
    machine.uniform_regs[5] = high;
    let [four_low, four_high] = words(4.0);
    let [quarter_low, quarter_high] = words(0.25);
    let bytes: Vec<u8> = [
        0xdead_beef,
        0x1234_5678,
        four_low,
        four_high,
        quarter_low,
        quarter_high,
    ]
    .iter()
    .flat_map(|word| word.to_le_bytes())
    .collect();
    machine.cbufs.insert(1, bytes);
    machine
}

/// Every double source kind: register pairs (low word first), `RZ` as a zero, a uniform pair,
/// `URZ` as a zero, a constant-buffer double at an offset that is a multiple of 8 (its high
/// word four bytes on), and an immediate as the high word over a zero low word; each with its absolute value
/// and negation, the absolute value first.
#[test]
fn dadd_reads_every_source_kind() {
    let even = |index: u8| AluSrc::reg(r(index));
    let cases = [
        (dadd(r(0), even(2), even(4)), 3.75),
        (dadd(r(0), even(2), AluSrc::reg(Gpr::RZ)), 1.5),
        (dadd(r(0), dreg(255, false, true), even(4)), 2.25),
        (dadd(r(0), even(2), ureg_src(4)), 3.5),
        (dadd(r(0), even(2), ureg_src(63)), 1.5),
        (dadd(r(0), even(2), cbuf_src(1, 8)), 5.5),
        (dadd(r(0), even(2), cbuf_src(1, 16)), 1.75),
        // 0x40000000 is 2.0's high word; 0x3ff80000 is 1.5's.
        (dadd(r(0), even(2), AluSrc::Imm(0x4000_0000)), 3.5),
        (dadd(r(0), even(2), AluSrc::Imm(0x3ff8_0000)), 3.0),
        // |-1.5| + 2.25 = 3.75; -|-1.5| + 2.25 = 0.75, where negating first would give 3.75;
        // -(-1.5) + -(1.5) = 0.
        (dadd(r(0), dreg(6, true, false), even(4)), 3.75),
        (dadd(r(0), dreg(6, true, true), even(4)), 0.75),
        (dadd(r(0), dreg(6, false, true), dreg(2, false, true)), 0.0),
    ];
    let base = machine();
    for (op, expected) in cases {
        let op = Op::DAdd(op);
        let mut run = base.clone();
        run_on(&[Instr::new(op)], &mut run);
        assert_eq!(get_double(&run, 0), f64::to_bits(expected), "{op}");
    }
}

/// The edges of a pair: `R254` and `UR62` are even and read their high word from `RZ` and
/// `URZ` (zero, whatever the slot behind them holds), so 1 in the low word is the smallest
/// denormal; `RZ` as the destination writes nothing; `R254` as the destination writes its low
/// word and drops the high. The odd neighbours `R253`, `UR61` and the destination `R253` are
/// refused, and `R252`, `UR60` accepted.
#[test]
fn double_pairs_at_each_edge() {
    use HalfDoubleUnsupported as U;
    let mut machine = Machine::new();
    machine.regs[254] = 1;
    machine.regs[255] = 0x4000_0000;
    machine.uniform_regs[62] = 1;
    machine.uniform_regs[63] = 0x4000_0000;
    let zero = AluSrc::reg(Gpr::RZ);
    // `a` is always a register; `b` may be a uniform register (form 7).
    for (a, b) in [(AluSrc::reg(r(254)), zero), (zero, ureg_src(62))] {
        let mut run = machine.clone();
        run_on(&[Instr::new(Op::DAdd(dadd(r(0), a, b)))], &mut run);
        assert_eq!(get_double(&run, 0), 1, "{a:?} {b:?}");
    }
    for (a, b) in [(AluSrc::reg(r(252)), zero), (zero, ureg_src(60))] {
        translated(&[Instr::new(Op::DAdd(dadd(r(0), a, b)))]);
    }
    assert_refused(
        Op::DAdd(dadd(r(0), AluSrc::reg(r(253)), zero)),
        U::DoubleRegister(r(253)),
    );
    assert_refused(
        Op::DAdd(dadd(r(0), zero, ureg_src(61))),
        U::DoubleUniformRegister(ur(61)),
    );
    assert_refused(
        Op::DAdd(dadd(r(253), zero, zero)),
        U::DoubleDestination(r(253)),
    );
    // RZ as the destination: nothing is written.
    let op = Op::DAdd(dadd(Gpr::RZ, AluSrc::reg(r(2)), AluSrc::reg(r(2))));
    let function = translated(&[Instr::new(op)]);
    assert!(!opcodes(&function).contains(&Opcode::SetRegister));
    // R254: the low word written, the high word's write to RZ dropped.
    let mut run = Machine::new();
    set_double(&mut run, 2, 1.5);
    let op = Op::DAdd(dadd(r(254), AluSrc::reg(r(2)), zero));
    let function = translated(&[Instr::new(op)]);
    assert_eq!(
        opcodes(&function)
            .iter()
            .filter(|&&opcode| opcode == Opcode::SetRegister)
            .count(),
        1
    );
    run.run(&function).expect("runs");
    assert_eq!(run.regs[254], words(1.5)[0]);
}

/// A constant-buffer double at an offset that is 4 mod 8, which NAK never emits
/// (`prop_to_64_src` keeps a pair only at `lo_cb.offset % 8 == 0`), reads the word at the offset
/// and the word four bytes on, NAK's layout carried over: a guess, listed as a divergence from
/// Eden's `GetDoubleCbuf` (`impl.cpp`), which reads the high word at `(offset & ~7) | 4`, here
/// the offset itself, so its low and high words would be one word. Confirmed by an `nvdisasm`
/// check of such an encoding or a hardware trace. At `0xfffc` the high word's read is at byte
/// `0x10000`, past what a 16-bit offset names.
#[test]
fn a_double_constant_off_eight_byte_alignment_is_a_guess() {
    let [low, high] = words(1.5);
    let bytes: Vec<u8> = [
        0xdead_beef,
        0x1234_5678,
        0x0bad_f00d,
        low,
        high,
        0x5555_aaaa,
    ]
    .iter()
    .flat_map(|word| word.to_le_bytes())
    .collect();
    let mut machine = Machine::new();
    machine.cbufs.insert(1, bytes);
    let op = Op::DAdd(dadd(r(0), AluSrc::reg(Gpr::RZ), cbuf_src(1, 12)));
    run_on(&[Instr::new(op)], &mut machine);
    assert_eq!(get_double(&machine, 0), f64::to_bits(1.5));
    // Eden's reading at 12 would pack the word at 12 twice: neither 1.5 nor this machine's result.
    assert_ne!(u64::from(low) << 32 | u64::from(low), f64::to_bits(1.5));
    let text = translated(&[Instr::new(Op::DAdd(dadd(
        r(0),
        AluSrc::reg(Gpr::RZ),
        cbuf_src(1, 0xfffc),
    )))])
    .to_string();
    assert!(text.contains("GetCbufU32 u32 0x1, u32 0xfffc\n"), "{text}");
    assert!(text.contains("GetCbufU32 u32 0x1, u32 0x10000\n"), "{text}");
}

/// `DMUL` and `DFMA` compute in 64 bits, in every form: `b` a register, an immediate (form
/// 4), a constant buffer (form 5) or a uniform pair (form 6); `DFMA`'s `c` likewise in forms
/// 2, 3 and 7; with modifiers on every source.
#[test]
fn dmul_and_dfma_compute_in_64_bits() {
    let even = |index: u8| AluSrc::reg(r(index));
    let cases = [
        (Op::DMul(dmul(r(0), even(2), even(4))), 3.375),
        (Op::DMul(dmul(r(0), even(2), AluSrc::Imm(0x4000_0000))), 3.0),
        (Op::DMul(dmul(r(0), even(2), cbuf_src(1, 8))), 6.0),
        (Op::DMul(dmul(r(0), even(2), ureg_src(4))), 3.0),
        (
            Op::DMul(dmul(r(0), dreg(6, true, false), dreg(4, false, true))),
            -3.375,
        ),
        // 1.5 * 2.25 + -1.5 = 1.875.
        (Op::DFma(dfma(r(0), even(2), even(4), even(6))), 1.875),
        (
            Op::DFma(dfma(r(0), even(2), even(4), AluSrc::Imm(0x4000_0000))),
            5.375,
        ),
        (
            Op::DFma(dfma(r(0), even(2), AluSrc::Imm(0x4000_0000), even(6))),
            1.5,
        ),
        (
            Op::DFma(dfma(r(0), even(2), even(4), cbuf_src(1, 16))),
            3.625,
        ),
        (Op::DFma(dfma(r(0), even(2), cbuf_src(1, 8), even(6))), 4.5),
        (Op::DFma(dfma(r(0), even(2), ureg_src(4), even(6))), 1.5),
        (Op::DFma(dfma(r(0), even(2), even(4), ureg_src(4))), 5.375),
        // 1.5 * -2.25 + |-1.5| = -1.875.
        (
            Op::DFma(dfma(
                r(0),
                even(2),
                dreg(4, false, true),
                dreg(6, true, false),
            )),
            -1.875,
        ),
    ];
    let base = machine();
    for (op, expected) in cases {
        let mut run = base.clone();
        run_on(&[Instr::new(op)], &mut run);
        assert_eq!(get_double(&run, 0), f64::to_bits(expected), "{op}");
    }
}

/// The arithmetic's controls, Eden's: contraction off, the rounding field (`RoundingOf`), no
/// flushing; each rounding mode carried.
#[test]
fn double_arithmetic_carries_its_rounding() {
    let even = |index: u8| AluSrc::reg(r(index));
    for (rnd, rounding) in [
        (RoundMode::NearestEven, FpRounding::Rn),
        (RoundMode::NegInf, FpRounding::Rm),
        (RoundMode::PosInf, FpRounding::Rp),
        (RoundMode::Zero, FpRounding::Rz),
    ] {
        let expected = [Flags::Fp(control(true, rounding, FmzMode::None))];
        let ops = [
            (
                Op::DAdd(OpDAdd {
                    rnd,
                    ..dadd(r(0), even(2), even(4))
                }),
                Opcode::FPAdd64,
            ),
            (
                Op::DMul(OpDMul {
                    rnd,
                    ..dmul(r(0), even(2), even(4))
                }),
                Opcode::FPMul64,
            ),
            (
                Op::DFma(OpDFma {
                    rnd,
                    ..dfma(r(0), even(2), even(4), even(6))
                }),
                Opcode::FPFma64,
            ),
        ];
        for (op, opcode) in ops {
            let function = translated(&[Instr::new(op)]);
            assert_eq!(flags_of(&function, opcode), expected, "{op}");
        }
    }
}

/// `DSETP` compares in 64 bits, every comparison, writing the comparison combined with the
/// accumulator to the first predicate and its negation combined likewise to the second.
#[test]
fn dsetp_compares_and_writes_both_predicates() {
    let pairs = [
        (1.0, 2.0),
        (f64::NAN, 1.0),
        (2.0, 2.0),
        (3.0, 2.0),
        (1.0, f64::NAN),
        (f64::NAN, f64::NAN),
    ];
    for (cmp, results) in COMPARISONS {
        let op = Op::DSetP(dsetp(p(1), p(2), cmp, AluSrc::reg(r(2)), AluSrc::reg(r(4))));
        for ((a, b), expected) in pairs.into_iter().zip(results) {
            let mut machine = Machine::new();
            set_double(&mut machine, 2, a);
            set_double(&mut machine, 4, b);
            machine.preds[1] = !expected;
            machine.preds[2] = expected;
            run_on(&[Instr::new(op)], &mut machine);
            assert_eq!(
                (machine.preds[1], machine.preds[2]),
                (expected, !expected),
                "{op}"
            );
        }
    }
    let function = translated(&[Instr::new(Op::DSetP(dsetp(
        p(1),
        p(2),
        FloatCmpOp::Lt,
        AluSrc::reg(r(2)),
        AluSrc::reg(r(4)),
    )))]);
    assert_eq!(
        flags_of(&function, Opcode::FPOrdLessThan64),
        [Flags::Fp(FpControl::default())]
    );
}

/// `DSETP`'s accumulator and combine: with `Or` and a true accumulator both predicates are
/// true; with `Xor` the comparison and its negation flip; a `PT` destination is not written
/// (the other predicate is, with `Or` and a true accumulator).
#[test]
fn dsetp_combines_with_the_accumulator() {
    let run = |op: OpDSetP, accum: bool| {
        let mut machine = Machine::new();
        set_double(&mut machine, 2, 1.0);
        set_double(&mut machine, 4, 2.0);
        machine.preds[3] = accum;
        machine.preds[1] = false;
        machine.preds[2] = false;
        run_on(&[Instr::new(Op::DSetP(op))], &mut machine);
        (machine.preds[1], machine.preds[2])
    };
    // 1.0 < 2.0.
    let base = dsetp(
        p(1),
        p(2),
        FloatCmpOp::Lt,
        AluSrc::reg(r(2)),
        AluSrc::reg(r(4)),
    );
    let with = |set_op, accum| OpDSetP {
        set_op,
        accum,
        ..base
    };
    assert_eq!(
        run(with(PredSetOp::And, pred(3, false)), false),
        (false, false)
    );
    assert_eq!(
        run(with(PredSetOp::And, pred(3, true)), false),
        (true, false)
    );
    assert_eq!(run(with(PredSetOp::Or, pred(3, false)), true), (true, true));
    assert_eq!(
        run(with(PredSetOp::Xor, pred(3, false)), true),
        (false, true)
    );
    assert_eq!(
        run(
            OpDSetP {
                dst: Pred::PT,
                ..with(PredSetOp::Or, pred(3, false))
            },
            true
        ),
        (false, true)
    );
    assert_eq!(
        run(
            OpDSetP {
                dst1: Pred::PT,
                ..with(PredSetOp::Or, pred(3, false))
            },
            true
        ),
        (true, false)
    );
}
