//! Half and double translator tests: every instruction and modifier run through the IR
//! interpreter (slice 7) with expected values worked out from the instruction's documented
//! behaviour and written as literals, PR #4588's synthetic cases for this group decoded from
//! their words, every refusal with its exact error and order, the f16 and f64 requirements, and
//! a fixed-seed property test that no decoded instruction panics.
//!
//! Every instruction a test runs first goes through the test assembler and the decoder and must
//! come back unchanged ([`translated`]), so each case is a word the decoder reads; the
//! hand-made operations no word encodes (a constant buffer past bank 31) are translated
//! directly.

mod double;
mod half;

use proptest::prelude::*;

use super::*;
use crate::asm::assemble;
use crate::decode::{GROUP_HOOKS, GROUPS, Instruction, OPCODE, OpFAdd, decode};
use crate::interp::Machine;
use crate::ir::{Function, Inst, Terminator};
use crate::testing::{any_pred_src, config};
use crate::translate::{TranslateError, Unsupported};

pub(super) fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

pub(super) fn p(index: u8) -> Pred {
    Pred::new(index).expect("a predicate")
}

pub(super) fn pred(index: u8, negated: bool) -> PredSrc {
    PredSrc {
        reg: p(index),
        negated,
    }
}

pub(super) const NOT_PT: PredSrc = PredSrc {
    reg: Pred::PT,
    negated: true,
};

pub(super) fn ur(index: u8) -> UniformReg {
    UniformReg::new(index).expect("a uniform register")
}

/// A register as an ALU source with modifiers.
pub(super) fn reg_mod(index: u8, abs: bool, neg: bool) -> AluSrc {
    AluSrc::Reg {
        reg: r(index),
        abs,
        neg,
    }
}

pub(super) fn cbuf_src(bank: u8, offset: u16) -> AluSrc {
    AluSrc::CBuf {
        cbuf: CBufRef {
            buf: CBuf::Binding(bank),
            offset,
        },
        abs: false,
        neg: false,
    }
}

pub(super) fn bindless_src() -> AluSrc {
    AluSrc::CBuf {
        cbuf: CBufRef {
            buf: CBuf::BindlessUniform(ur(3)),
            offset: 8,
        },
        abs: false,
        neg: false,
    }
}

pub(super) fn ureg_src(index: u8) -> AluSrc {
    AluSrc::UniformReg {
        reg: ur(index),
        abs: false,
        neg: false,
    }
}

/// A packed-half register source reading both halves as stored.
pub(super) fn h(index: u8) -> HalfSrc {
    HalfSrc::new(AluSrc::reg(r(index)))
}

/// A packed-half register source with a swizzle and modifiers.
pub(super) fn hs(index: u8, swizzle: HalfSwizzle, abs: bool, neg: bool) -> HalfSrc {
    HalfSrc {
        src: reg_mod(index, abs, neg),
        swizzle,
    }
}

/// A packed-half immediate.
pub(super) fn himm(bits: u32) -> HalfSrc {
    HalfSrc::new(AluSrc::Imm(bits))
}

/// Two halves in a word, `low` in the low half.
pub(super) const fn pack(low: u32, high: u32) -> u32 {
    high << 16 | low
}

pub(super) const H_NZERO: u32 = 0x8000;
pub(super) const H_HALF: u32 = 0x3800;
pub(super) const H_ONE: u32 = 0x3c00;
pub(super) const H_ONE_HALF: u32 = 0x3e00;
pub(super) const H_TWO: u32 = 0x4000;
pub(super) const H_TWO_HALF: u32 = 0x4100;
pub(super) const H_THREE: u32 = 0x4200;
pub(super) const H_FOUR: u32 = 0x4400;
pub(super) const H_FIVE: u32 = 0x4500;
pub(super) const H_SIX: u32 = 0x4600;
pub(super) const H_NEG_ONE: u32 = 0xbc00;
pub(super) const H_NEG_TWO: u32 = 0xc000;
pub(super) const H_NEG_THREE: u32 = 0xc200;
pub(super) const H_NEG_FOUR: u32 = 0xc400;
pub(super) const H_INF: u32 = 0x7c00;
pub(super) const H_NEG_INF: u32 = 0xfc00;
pub(super) const H_QNAN: u32 = 0x7e00;
/// The IR's canonical half NaN, as the interpreter gives a NaN result of an arithmetic
/// operation or a conversion (`interp::float`'s module docs).
pub(super) const H_CANONICAL_NAN: u32 = 0x7fff;
pub(super) const H_DENORM: u32 = 0x0001;

pub(super) fn hadd2(dst: Gpr, a: HalfSrc, b: HalfSrc) -> OpHAdd2 {
    OpHAdd2 {
        dst,
        srcs: [a, b],
        saturate: false,
        f32: false,
        ftz: false,
    }
}

pub(super) fn hmul2(dst: Gpr, a: HalfSrc, b: HalfSrc) -> OpHMul2 {
    OpHMul2 {
        dst,
        srcs: [a, b],
        saturate: false,
        ftz: false,
        dnz: false,
        relu: false,
    }
}

pub(super) fn hfma2(dst: Gpr, a: HalfSrc, b: HalfSrc, c: HalfSrc) -> OpHFma2 {
    OpHFma2 {
        dst,
        srcs: [a, b, c],
        saturate: false,
        f32: false,
        ftz: false,
        dnz: false,
        relu: None,
        bf16: false,
        neg_high_b: false,
    }
}

pub(super) fn hmnmx2(dst: Gpr, a: HalfSrc, b: HalfSrc, min: PredSrc) -> OpHMnMx2 {
    OpHMnMx2 {
        dst,
        srcs: [a, b],
        min,
        ftz: false,
        f32: false,
        nan: false,
        xorsign: false,
        bf16: false,
    }
}

pub(super) fn hsetp2(dst: Pred, dst1: Pred, cmp: FloatCmpOp, a: HalfSrc, b: HalfSrc) -> OpHSetP2 {
    OpHSetP2 {
        dst,
        dst1,
        cmp,
        set_op: PredSetOp::And,
        srcs: [a, b],
        accum: PredSrc::TRUE,
        ftz: false,
        horizontal: false,
    }
}

pub(super) fn hset2(dst: Gpr, cmp: FloatCmpOp, a: HalfSrc, b: HalfSrc) -> OpHSet2 {
    OpHSet2 {
        dst,
        cmp,
        srcs: [a, b],
        set_op: PredSetOp::And,
        accum: PredSrc::TRUE,
        ftz: false,
        bf: false,
    }
}

pub(super) fn dadd(dst: Gpr, a: AluSrc, b: AluSrc) -> OpDAdd {
    OpDAdd {
        dst,
        srcs: [a, b],
        rnd: RoundMode::NearestEven,
    }
}

pub(super) fn dmul(dst: Gpr, a: AluSrc, b: AluSrc) -> OpDMul {
    OpDMul {
        dst,
        srcs: [a, b],
        rnd: RoundMode::NearestEven,
    }
}

pub(super) fn dfma(dst: Gpr, a: AluSrc, b: AluSrc, c: AluSrc) -> OpDFma {
    OpDFma {
        dst,
        srcs: [a, b, c],
        rnd: RoundMode::NearestEven,
    }
}

pub(super) fn dsetp(dst: Pred, dst1: Pred, cmp: FloatCmpOp, a: AluSrc, b: AluSrc) -> OpDSetP {
    OpDSetP {
        dst,
        dst1,
        cmp,
        set_op: PredSetOp::And,
        srcs: [a, b],
        accum: PredSrc::TRUE,
    }
}

/// `instr`, after checking that the test assembler encodes it and the decoder reads the word
/// back as the same instruction.
pub(super) fn encodable(instr: Instr) -> Instr {
    let word = assemble(&instr).expect("assembles");
    assert_eq!(decode(word), Ok(instr), "{instr}");
    instr
}

/// The rejection the translation driver reports for `error`, instruction `index`.
fn reported(index: usize, instr: Instr, error: HalfDoubleError) -> TranslateError {
    match error {
        HalfDoubleError::Unsupported(what) => TranslateError::Unsupported {
            index,
            instr,
            what: Unsupported::HalfDouble(what),
        },
        HalfDoubleError::Build(error) => TranslateError::Build(error),
    }
}

/// `program`, this group's instructions, translated one after another by [`translate_instr`]
/// into a function's entry block, which then returns, without the encoding check. A refusal is
/// reported as the translation driver reports it.
pub(super) fn translate_raw(program: &[Instr]) -> Result<Function, TranslateError> {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    for (index, instr) in program.iter().enumerate() {
        translate_instr(&mut builder, *instr, false)
            .map_err(|error| reported(index, *instr, error))?;
    }
    builder.terminate(Terminator::Return)?;
    Ok(function)
}

/// `program`, each instruction checked by [`encodable`], translated; the result verifies.
pub(super) fn translated(program: &[Instr]) -> Function {
    for instr in program {
        encodable(*instr);
    }
    let function = translate_raw(program).expect("translates");
    assert_eq!(function.verify(), Ok(()));
    function
}

/// Runs `program` on `machine`.
pub(super) fn run_on(program: &[Instr], machine: &mut Machine) {
    let function = translated(program);
    machine.run(&function).expect("runs");
}

/// Runs `op` on a machine whose registers are set from `regs` and returns the machine.
pub(super) fn run_op(op: Op, regs: &[(u8, u32)]) -> Machine {
    let mut machine = Machine::new();
    for &(index, bits) in regs {
        machine.regs[usize::from(index)] = bits;
    }
    run_on(&[Instr::new(op)], &mut machine);
    machine
}

/// The rejection translating `instr` as the only instruction gives.
pub(super) fn refusal(
    instr: Instr,
    what: HalfDoubleUnsupported,
) -> Result<Function, TranslateError> {
    Err(TranslateError::Unsupported {
        index: 0,
        instr,
        what: Unsupported::HalfDouble(what),
    })
}

/// Asserts that `op` is refused with `what`, after checking it encodes.
pub(super) fn assert_refused(op: Op, what: HalfDoubleUnsupported) {
    let instr = encodable(Instr::new(op));
    assert_eq!(translate_raw(&[instr]), refusal(instr, what), "{instr}");
}

/// Asserts that `op`, which no word encodes, is refused with `what`.
pub(super) fn assert_refused_raw(op: Op, what: HalfDoubleUnsupported) {
    let instr = Instr::new(op);
    assert!(assemble(&instr).is_err(), "{instr}");
    assert_eq!(translate_raw(&[instr]), refusal(instr, what), "{instr}");
}

/// The flags of every operation with `opcode` in `function`, in order.
pub(super) fn flags_of(function: &Function, opcode: Opcode) -> Vec<Flags> {
    function
        .listed()
        .filter_map(|(_, inst)| match inst {
            Inst::Op(op) if op.opcode == opcode => Some(op.flags),
            _ => None,
        })
        .collect()
}

/// Every opcode `function` lists, in order.
pub(super) fn opcodes(function: &Function) -> Vec<Opcode> {
    function
        .listed()
        .filter_map(|(_, inst)| match inst {
            Inst::Op(op) => Some(op.opcode),
            Inst::Phi(_) => None,
        })
        .collect()
}

/// The constants, against the literals of the references they cite.
#[test]
fn constants_match_the_reference() {
    // `ops_half.cpp`, `HFMA2`: `std::bit_cast<f32>(0x7fc00000U)`.
    assert_eq!(RELU_NAN, 0x7fc0_0000);
    // Eden's `HSET2`: `bf ? 0x3c00 : 0xffff`.
    assert_eq!(HSET2_TRUE_HALF, 0x3c00);
    assert_eq!(HSET2_TRUE_MASK, 0xffff);
    // `Maxwell::Extract`: `CompositeExtract(vector, 0)` and `(vector, 1)`.
    assert_eq!(LOW_LANE, 0);
    assert_eq!(HIGH_LANE, 1);
    // NAK's `prop_to_64_src`: `cb.offset(4)`; an `Imm32` is the high word of 64 bits.
    assert_eq!(DOUBLE_HIGH_WORD, 4);
    assert_eq!(DOUBLE_IMM_SHIFT, 32);
    // `visitor.cpp`, `Visitor::Cbuf`: `cbuf_index >= 32`.
    assert_eq!(CBUF_BANKS, 32);
    // `lower_to_maxwell.h` case `0x231`: `b.reg >= 250` and `a.absolute && a.reg >= 248`.
    assert_eq!(FMA_MIXED_SIGN_FIRST_SCRATCH, 250);
    assert_eq!(FMA_ABSOLUTE_FIRST_SCRATCH, 248);
}

/// Every operation of the group is in it, and no operation of another group (the float
/// group's `FADD`, the integer group's `IADD3` as samples).
#[test]
fn the_group_is_its_ten_operations() {
    let ops = [
        Op::HAdd2(hadd2(r(0), h(1), h(2))),
        Op::HFma2(hfma2(r(0), h(1), h(2), h(3))),
        Op::HMul2(hmul2(r(0), h(1), h(2))),
        Op::HSet2(hset2(r(0), FloatCmpOp::Lt, h(1), h(2))),
        Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, h(1), h(2))),
        Op::HMnMx2(hmnmx2(r(0), h(1), h(2), PredSrc::TRUE)),
        Op::DAdd(dadd(r(0), AluSrc::reg(r(2)), AluSrc::reg(r(4)))),
        Op::DMul(dmul(r(0), AluSrc::reg(r(2)), AluSrc::reg(r(4)))),
        Op::DFma(dfma(
            r(0),
            AluSrc::reg(r(2)),
            AluSrc::reg(r(4)),
            AluSrc::reg(r(6)),
        )),
        Op::DSetP(dsetp(
            p(0),
            Pred::PT,
            FloatCmpOp::Lt,
            AluSrc::reg(r(2)),
            AluSrc::reg(r(4)),
        )),
    ];
    let mnemonics: std::collections::BTreeSet<&str> = ops.iter().map(Op::mnemonic).collect();
    assert_eq!(mnemonics.len(), 10);
    for op in ops {
        assert!(in_group(&op), "{op}");
    }
    let fadd = Op::FAdd(OpFAdd {
        dst: r(0),
        srcs: [AluSrc::reg(r(1)), AluSrc::reg(r(2))],
        ignored: Gpr::RZ,
        saturate: false,
        rnd: RoundMode::NearestEven,
        ftz: false,
    });
    assert!(!in_group(&fadd));
    let mut function = Function::new();
    let entry = function.entry();
    assert_eq!(
        translate_instr(
            &mut Builder::new(&mut function, entry),
            Instr::new(fadd),
            false
        ),
        Err(HalfDoubleError::Unsupported(
            HalfDoubleUnsupported::OtherGroup
        ))
    );
}

/// The f16 and f64 requirements are recorded in the IR, as collect info reads them
/// (`info::opcode_requirements`): every packed-half instruction needs 16-bit floats and no
/// 64-bit ones, every double one 64-bit floats and no 16-bit ones.
#[test]
fn requirements_are_recorded() {
    let half = [
        Op::HAdd2(hadd2(r(0), h(1), h(2))),
        Op::HAdd2(OpHAdd2 {
            f32: true,
            ..hadd2(
                r(0),
                hs(255, HalfSwizzle::H1H0, false, true),
                hs(2, HalfSwizzle::H0H0, false, false),
            )
        }),
        Op::HFma2(hfma2(r(0), h(1), h(2), h(3))),
        Op::HMul2(hmul2(r(0), h(1), h(2))),
        Op::HSet2(hset2(r(0), FloatCmpOp::Lt, h(1), h(2))),
        Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, h(1), h(2))),
        Op::HMnMx2(hmnmx2(r(0), h(1), h(2), PredSrc::TRUE)),
    ];
    let double = [
        Op::DAdd(dadd(r(0), AluSrc::reg(r(2)), AluSrc::reg(r(4)))),
        Op::DMul(dmul(r(0), AluSrc::reg(r(2)), AluSrc::reg(r(4)))),
        Op::DFma(dfma(
            r(0),
            AluSrc::reg(r(2)),
            AluSrc::reg(r(4)),
            AluSrc::reg(r(6)),
        )),
        Op::DSetP(dsetp(
            p(0),
            Pred::PT,
            FloatCmpOp::Lt,
            AluSrc::reg(r(2)),
            AluSrc::reg(r(4)),
        )),
    ];
    let needs = |op: Op| {
        let function = translated(&[Instr::new(op)]);
        opcodes(&function)
            .into_iter()
            .map(crate::info::opcode_requirements)
            .fold((false, false), |(f16, f64), req| {
                (f16 || req.f16, f64 || req.f64)
            })
    };
    for op in half {
        assert_eq!(needs(op), (true, false), "{op}");
    }
    for op in double {
        assert_eq!(needs(op), (false, true), "{op}");
    }
}

/// A guarded instruction becomes a conditional block: it runs where the guard holds and leaves
/// its destination as it was elsewhere, for a plain and a negated guard, for a register
/// destination (`HADD2`), a register pair (`DADD`) and predicates (`HSETP2`).
#[test]
fn guarded_instructions_become_conditional_blocks() {
    let add = Op::HAdd2(hadd2(r(0), h(1), h(2)));
    let dadd = Op::DAdd(dadd(r(4), AluSrc::reg(r(6)), AluSrc::reg(r(8))));
    let setp = Op::HSetP2(hsetp2(p(1), p(2), FloatCmpOp::Lt, h(1), h(2)));
    for negated in [false, true] {
        for p3 in [false, true] {
            let mut machine = Machine::new();
            machine.regs[0] = 0xdead_beef;
            machine.regs[1] = pack(H_ONE, H_TWO);
            machine.regs[2] = pack(H_HALF, H_FOUR);
            machine.regs[4] = 0x1111_1111;
            machine.regs[5] = 0x2222_2222;
            double::set_double(&mut machine, 6, 1.0);
            double::set_double(&mut machine, 8, 2.0);
            machine.preds[1] = false;
            machine.preds[2] = false;
            machine.preds[3] = p3;
            let program: Vec<Instr> = [add, dadd, setp]
                .into_iter()
                .map(|op| Instr {
                    guard: pred(3, negated),
                    op,
                })
                .collect();
            run_on(&program, &mut machine);
            let runs = p3 != negated;
            let context = format!("negated {negated}, p3 {p3}");
            if runs {
                assert_eq!(machine.regs[0], pack(H_ONE_HALF, H_SIX), "{context}");
                assert_eq!(
                    double::get_double(&machine, 4),
                    3.0_f64.to_bits(),
                    "{context}"
                );
                // 1.0 < 0.5 is false, 2.0 < 4.0 true.
                assert_eq!(
                    (machine.preds[1], machine.preds[2]),
                    (false, true),
                    "{context}"
                );
            } else {
                assert_eq!(machine.regs[0], 0xdead_beef, "{context}");
                assert_eq!(
                    (machine.regs[4], machine.regs[5]),
                    (0x1111_1111, 0x2222_2222)
                );
                assert_eq!(
                    (machine.preds[1], machine.preds[2]),
                    (false, false),
                    "{context}"
                );
            }
        }
    }
}

/// A block flow analysis already made conditional on the guard (`guard_tested`) holds the
/// instruction unconditionally, and so does one under `PT`: no blocks are added and no guard
/// is read. Under `@!PT` the instruction goes in a block behind a constant `false`.
#[test]
fn tested_and_constant_guards() {
    let op = Op::HMul2(hmul2(r(0), h(1), h(2)));
    for (guard, tested) in [(pred(2, false), true), (PredSrc::TRUE, false)] {
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        translate_instr(&mut builder, Instr { guard, op }, tested).expect("translates");
        assert_eq!(builder.block(), entry);
        builder.terminate(Terminator::Return).expect("terminates");
        assert_eq!(function.blocks().count(), 1);
        assert!(!opcodes(&function).contains(&Opcode::GetPred));
    }
    let function = translated(&[Instr { guard: NOT_PT, op }]);
    assert!(
        function
            .to_string()
            .starts_with("block0:\n    branch u1 false, block1, block2\n"),
        "{function}"
    );
    let mut machine = Machine::new();
    machine.regs[0] = 7;
    machine.run(&function).expect("runs");
    assert_eq!(machine.regs[0], 7);
}

/// The translation driver takes the group (slice 12i): a program of half and double
/// instructions assembled, run through flow analysis and the driver, and interpreted.
#[test]
fn the_driver_translates_the_group() {
    use crate::decode::OpExit;
    use crate::translate::driver::translate_instrs;
    use crate::translate::{COLOR_TARGETS, ColorOutputs, GuestStage, Stage};
    let stage = Stage {
        stage: GuestStage::Vertex,
        color_outputs: ColorOutputs {
            targets: [0; COLOR_TARGETS],
        },
    };
    let program = [
        Instr {
            guard: pred(0, false),
            op: Op::HAdd2(hadd2(r(0), h(1), h(2))),
        },
        Instr::new(Op::DMul(dmul(r(4), AluSrc::reg(r(6)), AluSrc::reg(r(6))))),
        Instr::new(Op::Exit(OpExit {
            keep_ref_count: false,
            no_at_exit: false,
            pred: PredSrc::TRUE,
        })),
    ];
    let function = translate_instrs(stage, None, &program).expect("translates");
    for p0 in [false, true] {
        let mut machine = Machine::new();
        machine.regs[1] = pack(H_ONE, H_TWO);
        machine.regs[2] = pack(H_TWO, H_TWO);
        double::set_double(&mut machine, 6, 1.5);
        machine.preds[0] = p0;
        machine.run(&function).expect("runs");
        let expected = if p0 { pack(H_THREE, H_FOUR) } else { 0 };
        assert_eq!(machine.regs[0], expected);
        assert_eq!(double::get_double(&machine, 4), 2.25_f64.to_bits());
    }
}

/// Error messages in plain words, as the driver prints them.
#[test]
fn refusals_read_in_plain_words() {
    let cases = [
        (
            HalfDoubleUnsupported::HalfAsFloat(2),
            "packed-half operand 2 read as a 32-bit float",
        ),
        (
            HalfDoubleUnsupported::HalfAddFloatSource(HalfSwizzle::H1H0),
            "an hadd2.f32 operand of swizzle 0",
        ),
        (
            HalfDoubleUnsupported::HalfFmaMixedSign,
            "an hfma2.h0_nh1 with a 32-bit float operand and its b not a plain register",
        ),
        (
            HalfDoubleUnsupported::HalfFmaAbsolute,
            "an hfma2 with a 32-bit float operand and an absolute value the lowering refuses",
        ),
        (
            HalfDoubleUnsupported::HalfFmaMultiplyZero,
            "hfma2's multiply-by-zero mode on halves without saturation",
        ),
        (
            HalfDoubleUnsupported::DoubleRegister(r(3)),
            "a double operand in the register pair at r3",
        ),
        (
            HalfDoubleUnsupported::DoubleUniformRegister(ur(5)),
            "a double operand in the uniform register pair at ur5",
        ),
        (
            HalfDoubleUnsupported::DoubleDestination(r(7)),
            "a double destination in the register pair at r7",
        ),
        (
            HalfDoubleUnsupported::ConstantBufferBank(32),
            "constant buffer 32",
        ),
    ];
    for (what, text) in cases {
        assert_eq!(what.to_string(), text);
    }
    let instr = Instr::new(Op::HMul2(OpHMul2 {
        dnz: true,
        ..hmul2(r(0), h(1), h(2))
    }));
    assert_eq!(
        translate_raw(&[instr])
            .map(|_| ())
            .map_err(|e| e.to_string()),
        Err(
            "instruction 0 (hmul2.fmz r0, r1, r2;): hmul2's multiply-by-zero mode on halves \
             without saturation is not supported"
                .to_owned()
        )
    );
}

/// One encodable operation per refusal that a decoded operation can reach, with its refusal.
pub(super) fn refusal_cases() -> Vec<(Op, HalfDoubleUnsupported)> {
    let mut cases = source_refusal_cases();
    cases.extend(mode_refusal_cases());
    cases.extend(fma_refusal_cases());
    cases.extend(min_max_refusal_cases());
    cases.extend(pair_refusal_cases());
    cases
}

/// The refusals of a source: its form, a bindless constant buffer, swizzle 1 where neither
/// reference translates it (`HMNMX2`, `HSETP2`).
fn source_refusal_cases() -> Vec<(Op, HalfDoubleUnsupported)> {
    use HalfDoubleUnsupported as U;
    let uniform = HalfSrc::new(ureg_src(4));
    let bindless = HalfSrc::new(bindless_src());
    let f32_swizzle = hs(1, HalfSwizzle::F32, false, false);
    let even = AluSrc::reg(r(2));
    vec![
        // The forms.
        (Op::HAdd2(hadd2(r(0), h(1), uniform)), U::HalfAddSourceForm),
        (
            Op::HMul2(hmul2(r(0), h(1), uniform)),
            U::HalfMultiplySourceForm,
        ),
        (
            Op::HFma2(hfma2(r(0), h(1), uniform, h(3))),
            U::HalfSourceForm,
        ),
        (
            Op::HFma2(hfma2(r(0), h(1), h(2), uniform)),
            U::HalfSourceForm,
        ),
        (
            Op::HMnMx2(hmnmx2(r(0), h(1), uniform, PredSrc::TRUE)),
            U::HalfSourceForm,
        ),
        (
            Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, h(1), uniform)),
            U::HalfSourceForm,
        ),
        (
            Op::HSet2(hset2(r(0), FloatCmpOp::Lt, h(1), uniform)),
            U::HSet2UniformForm,
        ),
        // Bindless constant buffers, in every instruction.
        (
            Op::HAdd2(hadd2(r(0), h(1), bindless)),
            U::BindlessConstantBuffer,
        ),
        (
            Op::HMul2(hmul2(r(0), h(1), bindless)),
            U::BindlessConstantBuffer,
        ),
        (
            Op::HFma2(hfma2(r(0), h(1), bindless, h(3))),
            U::BindlessConstantBuffer,
        ),
        (
            Op::HMnMx2(hmnmx2(r(0), h(1), bindless, PredSrc::TRUE)),
            U::BindlessConstantBuffer,
        ),
        (
            Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, h(1), bindless)),
            U::BindlessConstantBuffer,
        ),
        (
            Op::HSet2(hset2(r(0), FloatCmpOp::Lt, h(1), bindless)),
            U::BindlessConstantBuffer,
        ),
        (
            Op::DAdd(dadd(r(0), even, bindless_src())),
            U::BindlessConstantBuffer,
        ),
        (
            Op::DMul(dmul(r(0), even, bindless_src())),
            U::BindlessConstantBuffer,
        ),
        (
            Op::DFma(dfma(r(0), even, bindless_src(), even)),
            U::BindlessConstantBuffer,
        ),
        (
            Op::DSetP(dsetp(p(0), Pred::PT, FloatCmpOp::Lt, even, bindless_src())),
            U::BindlessConstantBuffer,
        ),
        // Swizzle 1, in `HMNMX2` and `HSETP2`, both operand positions.
        (
            Op::HMnMx2(hmnmx2(r(0), f32_swizzle, h(2), PredSrc::TRUE)),
            U::HalfAsFloat(0),
        ),
        (
            Op::HMnMx2(hmnmx2(r(0), h(2), f32_swizzle, PredSrc::TRUE)),
            U::HalfAsFloat(1),
        ),
        (
            Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, f32_swizzle, h(1))),
            U::HalfAsFloat(0),
        ),
        (
            Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, h(1), f32_swizzle)),
            U::HalfAsFloat(1),
        ),
    ]
}

/// The refusals of an instruction's modes.
fn mode_refusal_cases() -> Vec<(Op, HalfDoubleUnsupported)> {
    use HalfDoubleUnsupported as U;
    let neg_rz = hs(255, HalfSwizzle::H1H0, false, true);
    let low = hs(2, HalfSwizzle::H0H0, false, false);
    let hadd2_f32 = |a: HalfSrc, b: HalfSrc| OpHAdd2 {
        f32: true,
        ..hadd2(r(0), a, b)
    };
    vec![
        // `HADD2.F32`.
        (
            Op::HAdd2(OpHAdd2 {
                ftz: true,
                ..hadd2_f32(neg_rz, low)
            }),
            U::HalfAddFloatFlags,
        ),
        (
            Op::HAdd2(OpHAdd2 {
                saturate: true,
                ..hadd2_f32(neg_rz, low)
            }),
            U::HalfAddFloatFlags,
        ),
        (
            Op::HAdd2(hadd2_f32(neg_rz, h(2))),
            U::HalfAddFloatSource(HalfSwizzle::H1H0),
        ),
        (
            Op::HAdd2(hadd2_f32(neg_rz, hs(2, HalfSwizzle::F32, false, false))),
            U::HalfAddFloatSource(HalfSwizzle::F32),
        ),
        (
            Op::HAdd2(hadd2_f32(hs(1, HalfSwizzle::H1H0, false, true), low)),
            U::HalfAddFloatOperand,
        ),
        (Op::HAdd2(hadd2_f32(h(255), low)), U::HalfAddFloatOperand),
        (
            Op::HAdd2(hadd2_f32(hs(255, HalfSwizzle::H1H0, true, true), low)),
            U::HalfAddFloatOperand,
        ),
        (
            Op::HAdd2(hadd2_f32(hs(255, HalfSwizzle::H0H0, false, true), low)),
            U::HalfAddFloatOperand,
        ),
        // `HMUL2`.
        (
            Op::HMul2(OpHMul2 {
                relu: true,
                ..hmul2(r(0), h(1), h(2))
            }),
            U::HalfMultiplyMode,
        ),
        (
            Op::HMul2(OpHMul2 {
                dnz: true,
                ..hmul2(r(0), h(1), h(2))
            }),
            U::HalfMultiplyZero,
        ),
    ]
}

/// The refusals of `HFMA2`'s modes, and of a promoted `HFMA2`'s sources.
fn fma_refusal_cases() -> Vec<(Op, HalfDoubleUnsupported)> {
    use HalfDoubleUnsupported as U;
    let f32_at = |index: u8| hs(index, HalfSwizzle::F32, false, false);
    let neg_high = |op: OpHFma2| OpHFma2 {
        neg_high_b: true,
        ..op
    };
    vec![
        (
            Op::HFma2(OpHFma2 {
                dnz: true,
                ..hfma2(r(0), h(1), h(2), h(3))
            }),
            U::HalfFmaMultiplyZero,
        ),
        (
            Op::HFma2(OpHFma2 {
                f32: true,
                ..hfma2(r(0), h(1), h(2), h(3))
            }),
            U::HalfFmaFloatMode,
        ),
        (
            Op::HFma2(OpHFma2 {
                relu: Some(PredSrc::TRUE),
                saturate: true,
                ..hfma2(r(0), h(1), h(2), h(3))
            }),
            U::HalfFmaReluSaturate,
        ),
        (
            Op::HFma2(OpHFma2 {
                bf16: true,
                ..hfma2(r(0), h(1), h(2), h(3))
            }),
            U::HalfFmaBf16,
        ),
        (
            Op::HFma2(neg_high(hfma2(
                r(0),
                f32_at(1),
                hs(2, HalfSwizzle::H0H0, false, false),
                h(3),
            ))),
            U::HalfFmaMixedSign,
        ),
        (
            Op::HFma2(hfma2(
                r(0),
                h(1),
                hs(2, HalfSwizzle::H1H0, true, false),
                f32_at(3),
            )),
            U::HalfFmaAbsolute,
        ),
    ]
}

/// The refusals of `HMNMX2`'s modes, each on its own.
fn min_max_refusal_cases() -> Vec<(Op, HalfDoubleUnsupported)> {
    use HalfDoubleUnsupported as U;
    vec![
        (
            Op::HMnMx2(OpHMnMx2 {
                f32: true,
                ..hmnmx2(r(0), h(1), h(2), PredSrc::TRUE)
            }),
            U::HalfMinMaxMode,
        ),
        (
            Op::HMnMx2(OpHMnMx2 {
                ftz: true,
                ..hmnmx2(r(0), h(1), h(2), PredSrc::TRUE)
            }),
            U::HalfMinMaxMode,
        ),
        (
            Op::HMnMx2(OpHMnMx2 {
                nan: true,
                ..hmnmx2(r(0), h(1), h(2), PredSrc::TRUE)
            }),
            U::HalfMinMaxMode,
        ),
        (
            Op::HMnMx2(OpHMnMx2 {
                xorsign: true,
                ..hmnmx2(r(0), h(1), h(2), PredSrc::TRUE)
            }),
            U::HalfMinMaxMode,
        ),
        (
            Op::HMnMx2(OpHMnMx2 {
                bf16: true,
                ..hmnmx2(r(0), h(1), h(2), PredSrc::TRUE)
            }),
            U::HalfMinMaxMode,
        ),
    ]
}

/// The refusals of a register pair.
fn pair_refusal_cases() -> Vec<(Op, HalfDoubleUnsupported)> {
    use HalfDoubleUnsupported as U;
    let even = AluSrc::reg(r(2));
    vec![
        (
            Op::DAdd(dadd(r(0), AluSrc::reg(r(1)), even)),
            U::DoubleRegister(r(1)),
        ),
        (
            Op::DMul(dmul(r(0), even, AluSrc::reg(r(3)))),
            U::DoubleRegister(r(3)),
        ),
        (
            Op::DFma(dfma(r(0), even, even, AluSrc::reg(r(5)))),
            U::DoubleRegister(r(5)),
        ),
        (
            Op::DSetP(dsetp(
                p(0),
                Pred::PT,
                FloatCmpOp::Lt,
                even,
                AluSrc::reg(r(253)),
            )),
            U::DoubleRegister(r(253)),
        ),
        (
            Op::DAdd(dadd(r(0), even, ureg_src(3))),
            U::DoubleUniformRegister(ur(3)),
        ),
        (Op::DAdd(dadd(r(1), even, even)), U::DoubleDestination(r(1))),
        (Op::DMul(dmul(r(7), even, even)), U::DoubleDestination(r(7))),
        (
            Op::DFma(dfma(r(253), even, even, even)),
            U::DoubleDestination(r(253)),
        ),
    ]
}

/// Every refusal a decoded operation can reach, with its exact error; a refusal leaves the
/// function as it was and the builder in its block, whatever the guard.
#[test]
fn refusals_append_nothing() {
    for (op, what) in refusal_cases() {
        assert_refused(op, what);
        for guard in [PredSrc::TRUE, pred(1, false), NOT_PT] {
            let mut function = Function::new();
            let entry = function.entry();
            let mut builder = Builder::new(&mut function, entry);
            builder
                .op(Opcode::GetRegister, &[Operand::Imm(Imm::Reg(r(9)))])
                .expect("appends");
            let before = builder.function().clone();
            let result = translate_instr(&mut builder, Instr { guard, op }, false);
            assert_eq!(result, Err(HalfDoubleError::Unsupported(what)), "{op}");
            assert_eq!(builder.block(), entry);
            assert_eq!(builder.function(), &before, "{op}");
        }
    }
}

/// Each variant a decoded operation can reach is among [`refusal_cases`]; the bank past the
/// last, which no decoded operation reaches, has its own test
/// ([`a_bank_past_the_last_is_refused`]).
#[test]
fn every_reachable_refusal_has_a_case() {
    use HalfDoubleUnsupported as U;
    let names: std::collections::BTreeSet<String> = refusal_cases()
        .iter()
        .map(|(_, what)| {
            let debug = format!("{what:?}");
            debug
                .split(['(', ' '])
                .next()
                .unwrap_or_default()
                .to_owned()
        })
        .collect();
    let all = [
        U::HalfAddSourceForm,
        U::HalfMultiplySourceForm,
        U::HalfSourceForm,
        U::HSet2UniformForm,
        U::BindlessConstantBuffer,
        U::HalfAsFloat(0),
        U::HalfAddFloatFlags,
        U::HalfAddFloatSource(HalfSwizzle::H1H0),
        U::HalfAddFloatOperand,
        U::HalfMultiplyMode,
        U::HalfMultiplyZero,
        U::HalfFmaMultiplyZero,
        U::HalfFmaFloatMode,
        U::HalfFmaReluSaturate,
        U::HalfFmaBf16,
        U::HalfFmaMixedSign,
        U::HalfFmaAbsolute,
        U::HalfMinMaxMode,
        U::DoubleRegister(r(1)),
        U::DoubleUniformRegister(ur(1)),
        U::DoubleDestination(r(1)),
    ];
    for what in all {
        // A match with no wildcard: a new variant fails to compile until it is listed here
        // (`OtherGroup` and `ConstantBufferBank` have tests of their own).
        match what {
            U::OtherGroup
            | U::ConstantBufferBank(_)
            | U::HalfAddSourceForm
            | U::HalfMultiplySourceForm
            | U::HalfSourceForm
            | U::HSet2UniformForm
            | U::BindlessConstantBuffer
            | U::HalfAsFloat(_)
            | U::HalfAddFloatFlags
            | U::HalfAddFloatSource(_)
            | U::HalfAddFloatOperand
            | U::HalfMultiplyMode
            | U::HalfMultiplyZero
            | U::HalfFmaMultiplyZero
            | U::HalfFmaFloatMode
            | U::HalfFmaReluSaturate
            | U::HalfFmaBf16
            | U::HalfFmaMixedSign
            | U::HalfFmaAbsolute
            | U::HalfMinMaxMode
            | U::DoubleRegister(_)
            | U::DoubleUniformRegister(_)
            | U::DoubleDestination(_) => {}
        }
        let debug = format!("{what:?}");
        let name = debug.split(['(', ' ']).next().unwrap_or_default();
        assert!(names.contains(name), "{name}");
    }
}

/// A constant buffer past bank 31, which no word encodes, is refused in every instruction that
/// reads one, and bank 31 is accepted: the check is `bank < 32`.
#[test]
fn a_bank_past_the_last_is_refused() {
    use HalfDoubleUnsupported as U;
    let even = AluSrc::reg(r(2));
    let cases = |bank: u8| {
        let half = HalfSrc::new(cbuf_src(bank, 8));
        let wide = cbuf_src(bank, 8);
        [
            Op::HAdd2(hadd2(r(0), h(1), half)),
            Op::HMul2(hmul2(r(0), h(1), half)),
            Op::HFma2(hfma2(r(0), h(1), half, h(3))),
            Op::HMnMx2(hmnmx2(r(0), h(1), half, PredSrc::TRUE)),
            Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, h(1), half)),
            Op::HSet2(hset2(r(0), FloatCmpOp::Lt, h(1), half)),
            Op::DAdd(dadd(r(0), even, wide)),
            Op::DMul(dmul(r(0), even, wide)),
            Op::DFma(dfma(r(0), even, wide, even)),
            Op::DSetP(dsetp(p(0), Pred::PT, FloatCmpOp::Lt, even, wide)),
        ]
    };
    for op in cases(CBUF_BANKS) {
        assert_refused_raw(op, U::ConstantBufferBank(CBUF_BANKS));
    }
    for op in cases(CBUF_BANKS - 1) {
        translated(&[Instr::new(op)]);
    }
}

/// With two faults at once in `HADD2` and `HMUL2`, the one the PR checks first is reported:
/// the instruction's form before its modes, a source's constant buffer before the modes, each
/// mode check before the next.
#[test]
fn add_and_multiply_refusals_report_in_the_prs_order() {
    use HalfDoubleUnsupported as U;
    let uniform = HalfSrc::new(ureg_src(4));
    let bindless = HalfSrc::new(bindless_src());
    let f32_at = |index: u8| hs(index, HalfSwizzle::F32, false, false);
    let neg_rz = hs(255, HalfSwizzle::H1H0, false, true);
    let cases = [
        // `HADD2`: the form, then `.F32`'s flags; the constant buffer, then the swizzles.
        (
            Op::HAdd2(OpHAdd2 {
                f32: true,
                ftz: true,
                ..hadd2(r(0), neg_rz, uniform)
            }),
            U::HalfAddSourceForm,
        ),
        (
            Op::HAdd2(hadd2(r(0), f32_at(1), bindless)),
            U::BindlessConstantBuffer,
        ),
        // `HADD2.F32`: the flags, then `b`'s swizzle, then `a`.
        (
            Op::HAdd2(OpHAdd2 {
                f32: true,
                saturate: true,
                ..hadd2(r(0), h(1), h(2))
            }),
            U::HalfAddFloatFlags,
        ),
        (
            Op::HAdd2(OpHAdd2 {
                f32: true,
                ..hadd2(r(0), h(1), h(2))
            }),
            U::HalfAddFloatSource(HalfSwizzle::H1H0),
        ),
        // `HMUL2`: the form, the constant buffer, `.RELU`, `.FMZ`, the swizzles.
        (
            Op::HMul2(OpHMul2 {
                relu: true,
                ..hmul2(r(0), h(1), uniform)
            }),
            U::HalfMultiplySourceForm,
        ),
        (
            Op::HMul2(OpHMul2 {
                relu: true,
                ..hmul2(r(0), h(1), bindless)
            }),
            U::BindlessConstantBuffer,
        ),
        (
            Op::HMul2(OpHMul2 {
                relu: true,
                dnz: true,
                ..hmul2(r(0), h(1), h(2))
            }),
            U::HalfMultiplyMode,
        ),
        (
            Op::HMul2(OpHMul2 {
                dnz: true,
                ..hmul2(r(0), h(1), bindless)
            }),
            U::BindlessConstantBuffer,
        ),
    ];
    for (op, what) in cases {
        assert_refused(op, what);
    }
}

/// With two faults at once in `HFMA2`, the one the PR checks first is reported: the form or
/// the constant buffer, then `.FMZ`, `.F32`, `.RELU` with `.SAT`, `.BF16_V2`, and on a
/// promoted instruction the lowering's `.H0_NH1` check before its absolute-value check.
#[test]
fn fma_refusals_report_in_the_prs_order() {
    use HalfDoubleUnsupported as U;
    let uniform = HalfSrc::new(ureg_src(4));
    let bindless = HalfSrc::new(bindless_src());
    let f32_at = |index: u8| hs(index, HalfSwizzle::F32, false, false);
    let cases = [
        (
            Op::HFma2(OpHFma2 {
                dnz: true,
                ..hfma2(r(0), h(1), uniform, h(3))
            }),
            U::HalfSourceForm,
        ),
        (
            Op::HFma2(OpHFma2 {
                dnz: true,
                ..hfma2(r(0), h(1), bindless, h(3))
            }),
            U::BindlessConstantBuffer,
        ),
        (
            Op::HFma2(OpHFma2 {
                dnz: true,
                f32: true,
                ..hfma2(r(0), h(1), h(2), h(3))
            }),
            U::HalfFmaMultiplyZero,
        ),
        (
            Op::HFma2(OpHFma2 {
                f32: true,
                relu: Some(pred(1, false)),
                ..hfma2(r(0), h(1), h(2), h(3))
            }),
            U::HalfFmaFloatMode,
        ),
        (
            Op::HFma2(OpHFma2 {
                relu: Some(pred(1, false)),
                saturate: true,
                bf16: true,
                ..hfma2(r(0), h(1), h(2), h(3))
            }),
            U::HalfFmaReluSaturate,
        ),
        (
            Op::HFma2(OpHFma2 {
                bf16: true,
                neg_high_b: true,
                ..hfma2(
                    r(0),
                    f32_at(1),
                    hs(2, HalfSwizzle::H1H1, false, false),
                    h(3),
                )
            }),
            U::HalfFmaBf16,
        ),
        // `.H0_NH1` with `b`'s swizzle `H1_H1`, and an absolute value on `c`.
        (
            Op::HFma2(OpHFma2 {
                neg_high_b: true,
                ..hfma2(
                    r(0),
                    f32_at(1),
                    hs(2, HalfSwizzle::H1H1, false, false),
                    hs(3, HalfSwizzle::H1H0, true, false),
                )
            }),
            U::HalfFmaMixedSign,
        ),
    ];
    for (op, what) in cases {
        assert_refused(op, what);
    }
}

/// With two faults at once in `HMNMX2`, `HSETP2` and `HSET2`, the one the PR checks first is
/// reported: the form or the constant buffer, the modes, the swizzles in operand order.
#[test]
fn compare_refusals_report_in_the_prs_order() {
    use HalfDoubleUnsupported as U;
    let uniform = HalfSrc::new(ureg_src(4));
    let bindless = HalfSrc::new(bindless_src());
    let f32_at = |index: u8| hs(index, HalfSwizzle::F32, false, false);
    let cases = [
        (
            Op::HMnMx2(OpHMnMx2 {
                ftz: true,
                ..hmnmx2(r(0), h(1), uniform, PredSrc::TRUE)
            }),
            U::HalfSourceForm,
        ),
        (
            Op::HMnMx2(OpHMnMx2 {
                ftz: true,
                ..hmnmx2(r(0), h(1), bindless, PredSrc::TRUE)
            }),
            U::BindlessConstantBuffer,
        ),
        (
            Op::HMnMx2(OpHMnMx2 {
                nan: true,
                ..hmnmx2(r(0), f32_at(1), h(2), PredSrc::TRUE)
            }),
            U::HalfMinMaxMode,
        ),
        (
            Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, f32_at(1), bindless)),
            U::BindlessConstantBuffer,
        ),
        (
            Op::HSet2(hset2(r(0), FloatCmpOp::Lt, f32_at(1), bindless)),
            U::BindlessConstantBuffer,
        ),
        (
            Op::HMnMx2(hmnmx2(r(0), f32_at(1), f32_at(2), PredSrc::TRUE)),
            U::HalfAsFloat(0),
        ),
        (
            Op::HSetP2(hsetp2(p(0), p(1), FloatCmpOp::Lt, f32_at(1), f32_at(2))),
            U::HalfAsFloat(0),
        ),
    ];
    for (op, what) in cases {
        assert_refused(op, what);
    }
}

/// The double group's register pairs are checked after every constant-buffer check, in operand
/// order and then the destination (module docs, "Register pairs").
#[test]
fn pairs_report_in_operand_order() {
    use HalfDoubleUnsupported as U;
    let odd = |index: u8| AluSrc::reg(r(index));
    let cases = [
        (
            Op::DAdd(dadd(r(1), odd(3), bindless_src())),
            U::BindlessConstantBuffer,
        ),
        (
            Op::DFma(dfma(r(1), odd(3), bindless_src(), odd(5))),
            U::BindlessConstantBuffer,
        ),
        (
            Op::DAdd(dadd(r(1), odd(3), odd(5))),
            U::DoubleRegister(r(3)),
        ),
        (
            Op::DFma(dfma(r(1), AluSrc::reg(r(2)), odd(5), odd(7))),
            U::DoubleRegister(r(5)),
        ),
        (
            Op::DMul(dmul(r(1), AluSrc::reg(r(2)), odd(5))),
            U::DoubleRegister(r(5)),
        ),
        (
            Op::DSetP(dsetp(p(0), Pred::PT, FloatCmpOp::Lt, odd(3), odd(5))),
            U::DoubleRegister(r(3)),
        ),
    ];
    for (op, what) in cases {
        assert_refused(op, what);
    }
    // A constant-buffer bank past the last comes before a pair too (a hand-made operation).
    assert_refused_raw(
        Op::DAdd(dadd(r(1), odd(3), cbuf_src(CBUF_BANKS, 0))),
        U::ConstantBufferBank(CBUF_BANKS),
    );
}

/// The half and double group's generators (`decode`'s per-group hooks), found by the group
/// whose forms hold `HADD2`'s opcode.
fn half_double_ops() -> Vec<BoxedStrategy<Op>> {
    let word = assemble(&Instr::new(Op::HAdd2(hadd2(r(0), h(1), h(2))))).expect("assembles");
    let opcode = word.field(OPCODE);
    let index = GROUPS
        .iter()
        .position(|forms| forms.iter().any(|form| u64::from(form.opcode) == opcode))
        .expect("a group decodes hadd2");
    (GROUP_HOOKS[index].any_ops)()
}

proptest! {
    #![proptest_config(config())]

    /// Every decoded half or double instruction under every guard translates to IR that
    /// verifies, or is refused with nothing appended; none panics.
    #[test]
    fn decoded_instructions_never_panic(
        op in proptest::strategy::Union::new(half_double_ops()),
        guard in any_pred_src(),
        tested in any::<bool>(),
    ) {
        let instr = Instr { guard, op };
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        let before = builder.function().clone();
        match translate_instr(&mut builder, instr, tested) {
            Ok(()) => {
                builder.terminate(Terminator::Return).expect("terminates");
                prop_assert_eq!(function.verify(), Ok(()));
            }
            Err(HalfDoubleError::Unsupported(_)) => prop_assert_eq!(builder.function(), &before),
            Err(HalfDoubleError::Build(error)) => prop_assert!(false, "{}", error),
        }
    }
}

/// A word of PR #4588's test file, decoded: four 32-bit words, low first.
pub(super) fn pr_word(words: [u32; 4]) -> Instr {
    let bits = words
        .iter()
        .rev()
        .fold(0_u128, |acc, &word| acc << 32 | u128::from(word));
    decode(Instruction::from_bits(bits)).expect("decodes")
}
