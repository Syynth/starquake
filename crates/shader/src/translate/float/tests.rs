//! Float translator tests: every instruction run through the IR interpreter (slice 7) with
//! expected values worked out from the instruction's documented behaviour and written as
//! literals, golden IR, PR #4588's cases for this group rebuilt from decoded operations, every
//! refusal with its exact error and order, and a fixed-seed property test that no decoded
//! instruction panics.
//!
//! Every instruction a test runs first goes through the test assembler and the decoder and must
//! come back unchanged ([`translated`]), so each case is a word the decoder reads; the few
//! hand-made operations no word encodes (a constant buffer past bank 31, an `I2F` select the
//! source does not have) are translated directly.

mod arith;
mod convert;

use proptest::prelude::*;

use super::*;
use crate::asm::assemble;
use crate::decode::{
    CBufRef, GROUP_HOOKS, GROUPS, MovSrc, OPCODE, OpExit, OpMov, UniformReg, decode,
};
use crate::interp::Machine;
use crate::ir::{Terminator, Value};
use crate::testing::{any_pred_src, config};
use crate::translate::{
    COLOR_TARGETS, ColorOutputs, GuestStage, Stage, TranslateError, Unsupported, translate,
};

pub(super) const VERTEX: Stage = Stage {
    stage: GuestStage::Vertex,
    color_outputs: ColorOutputs {
        targets: [0; COLOR_TARGETS],
    },
};

pub(super) fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

pub(super) fn reg(index: u8) -> AluSrc {
    AluSrc::reg(r(index))
}

/// A register source with modifiers.
pub(super) fn reg_mod(index: u8, abs: bool, neg: bool) -> AluSrc {
    AluSrc::Reg {
        reg: r(index),
        abs,
        neg,
    }
}

pub(super) fn imm(value: f32) -> AluSrc {
    AluSrc::Imm(value.to_bits())
}

pub(super) fn cbuf(bank: u8, offset: u16) -> AluSrc {
    AluSrc::CBuf {
        cbuf: CBufRef {
            buf: CBuf::Binding(bank),
            offset,
        },
        abs: false,
        neg: false,
    }
}

pub(super) fn ureg(index: u8) -> AluSrc {
    AluSrc::UniformReg {
        reg: UniformReg::new(index).expect("a uniform register"),
        abs: false,
        neg: false,
    }
}

pub(super) fn pred(index: u8, negated: bool) -> PredSrc {
    PredSrc {
        reg: Pred::new(index).expect("a predicate"),
        negated,
    }
}

pub(super) const NOT_PT: PredSrc = PredSrc {
    reg: Pred::PT,
    negated: true,
};

pub(super) fn int(size: u8, signed: bool) -> IntType {
    IntType::new(size, signed).expect("an integer type")
}

pub(super) fn exit() -> Instr {
    Instr::new(Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    }))
}

/// `instr`, after checking that the test assembler encodes it and the decoder reads the word
/// back as the same instruction.
pub(super) fn encodable(instr: Instr) -> Instr {
    let word = assemble(&instr).expect("assembles");
    assert_eq!(decode(word), Ok(instr), "{instr}");
    instr
}

/// `program` followed by `EXIT`, translated as a vertex program, without the encoding check.
pub(super) fn translate_raw(program: &[Instr]) -> Result<Function, TranslateError> {
    let mut program = program.to_vec();
    program.push(exit());
    translate(VERTEX, &program)
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
pub(super) fn refusal(instr: Instr, what: FloatUnsupported) -> Result<Function, TranslateError> {
    Err(TranslateError::Unsupported {
        index: 0,
        instr,
        what: Unsupported::Float(what),
    })
}

/// Asserts that `op` is refused with `what`, after checking it encodes.
pub(super) fn assert_refused(op: Op, what: FloatUnsupported) {
    let instr = encodable(Instr::new(op));
    assert_eq!(translate_raw(&[instr]), refusal(instr, what), "{instr}");
}

/// Asserts that `op`, which no word encodes, is refused with `what`.
pub(super) fn assert_refused_raw(op: Op, what: FloatUnsupported) {
    let instr = Instr::new(op);
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

/// `function` as a backend that ignores rounding modes and flushing sees it: every
/// floating-point control's rounding unspecified (to nearest) and its denormals kept.
pub(super) fn without_controls(mut function: Function) -> Function {
    let values: Vec<Value> = function.insts().map(|(value, _)| value).collect();
    for value in values {
        if let Some(Inst::Op(op)) = function.inst_mut(value)
            && let Flags::Fp(control) = &mut op.flags
        {
            control.rounding = FpRounding::DontCare;
            control.fmz_mode = FmzMode::None;
        }
    }
    function
}

pub(super) fn fadd(dst: Gpr, a: AluSrc, b: AluSrc) -> OpFAdd {
    OpFAdd {
        dst,
        srcs: [a, b],
        ignored: Gpr::RZ,
        saturate: false,
        rnd: RoundMode::NearestEven,
        ftz: false,
    }
}

pub(super) fn fmul(dst: Gpr, a: AluSrc, b: AluSrc) -> OpFMul {
    OpFMul {
        dst,
        srcs: [a, b],
        ignored: Gpr::RZ,
        saturate: false,
        rnd: RoundMode::NearestEven,
        ftz: false,
        dnz: false,
        scale: FMulScale::None,
    }
}

pub(super) fn ffma(dst: Gpr, a: AluSrc, b: AluSrc, c: AluSrc) -> OpFFma {
    OpFFma {
        dst,
        srcs: [a, b, c],
        saturate: false,
        rnd: RoundMode::NearestEven,
        ftz: false,
        dnz: false,
    }
}

pub(super) fn fmnmx(dst: Gpr, a: AluSrc, b: AluSrc, min: PredSrc) -> OpFMnMx {
    OpFMnMx {
        dst,
        srcs: [a, b],
        min,
        ftz: false,
        nan: false,
    }
}

pub(super) fn fset(dst: Gpr, cmp: FloatCmpOp, a: AluSrc, b: AluSrc) -> OpFSet {
    OpFSet {
        dst,
        cmp,
        srcs: [a, b],
        set_op: PredSetOp::And,
        accum: PredSrc::TRUE,
        ftz: false,
    }
}

pub(super) fn fsetp(dst: Pred, dst1: Pred, cmp: FloatCmpOp, a: AluSrc, b: AluSrc) -> OpFSetP {
    OpFSetP {
        dst,
        dst1,
        cmp,
        set_op: PredSetOp::And,
        srcs: [a, b],
        accum: PredSrc::TRUE,
        ftz: false,
    }
}

pub(super) fn mufu(dst: Gpr, op: MuFuOp, src: AluSrc) -> OpMuFu {
    OpMuFu {
        dst,
        op,
        src,
        f16: false,
        high: false,
    }
}

pub(super) fn fswzadd(dst: Gpr, a: Gpr, b: Gpr, ops: [FSwzAddOp; 4]) -> OpFSwzAdd {
    OpFSwzAdd {
        dst,
        srcs: [a, b],
        ops,
        rnd: RoundMode::NearestEven,
        ftz: false,
        ndv: false,
    }
}

pub(super) fn frnd(dst: Gpr, src: AluSrc, rnd: RoundMode) -> OpFRnd {
    OpFRnd {
        dst,
        src,
        dst_type: FloatType::F32,
        src_type: FloatType::F32,
        rnd,
        ftz: false,
        high: false,
        wide: false,
    }
}

pub(super) fn f2f(dst: Gpr, src: AluSrc, rnd: RoundMode) -> OpF2F {
    OpF2F {
        dst,
        src,
        dst_type: FloatType::F16,
        src_type: FloatType::F32,
        rnd,
        ftz: false,
        high: false,
        wide: false,
    }
}

pub(super) fn f2i(dst: Gpr, src: AluSrc, dst_type: IntType, rnd: RoundMode) -> OpF2I {
    OpF2I {
        dst,
        src,
        dst_type,
        src_type: FloatType::F32,
        rnd,
        ftz: false,
        ntz: false,
        high: false,
        wide: false,
    }
}

pub(super) fn i2f(
    dst: Gpr,
    src: AluSrc,
    src_type: IntType,
    rnd: RoundMode,
) -> crate::decode::OpI2F {
    crate::decode::OpI2F {
        dst,
        src,
        dst_type: FloatType::F32,
        src_type,
        rnd,
        select: 0,
        wide: false,
    }
}

pub(super) fn i2fp(dst: Gpr, src: AluSrc, src_type: IntType, rnd: RoundMode) -> OpI2FP {
    OpI2FP {
        dst,
        src,
        dst_type: FloatType::F32,
        src_type,
        rnd,
    }
}

pub(super) fn f2fp(dst: Gpr, a: AluSrc, b: AluSrc) -> OpF2FP {
    OpF2FP {
        dst,
        srcs: [a, b],
        c: Gpr::RZ,
        merge_c: false,
        h1: false,
        rnd: RoundMode::NearestEven,
    }
}

pub(super) const ONE: u32 = 0x3f80_0000;
pub(super) const TWO: u32 = 0x4000_0000;
pub(super) const INF: u32 = 0x7f80_0000;
pub(super) const NEG_INF: u32 = 0xff80_0000;
pub(super) const QNAN: u32 = 0x7fc0_0000;

/// The constants, against the literals of the references they cite.
#[test]
fn constants_match_the_reference() {
    // `visitor.cpp`, `Visitor::Cbuf`: `cbuf_index >= 32`, `byte_offset % 4`.
    assert_eq!(CBUF_BANKS, 32);
    assert_eq!(CBUF_WORD_BYTES, 4);
    // `ops_alu.cpp`: `FSET` writes `0x3f800000U`, `FMNMX.NAN` writes `0x7fffffffU`.
    assert_eq!(FSET_TRUE, 0x3f80_0000);
    assert_eq!(FMNMX_NAN, 0x7fff_ffff);
    // `ops_convert.cpp`, `F2FP`: `ShiftLeftLogical(high, Imm32(16U))`.
    assert_eq!(HALF_SHIFT, 16);
    // `ops_float.cpp`, `FSWZADD`: `subop == 0x5a || subop == 0x66`, passed as `~subop & 0xffU`.
    let bytes: Vec<u32> = FSWZADD_OPERATIONS.iter().map(|&(_, byte)| byte).collect();
    assert_eq!(bytes, [0x5a, 0x66]);
    assert_eq!(FSWZADD_BYTE, 0xff);
    // 2 pi to binary32 is 6.2831855 (0x40c90fdb).
    assert_eq!(TURN, 0x40c9_0fdb);
    // IEEE 754 binary32 and binary16 fields.
    assert_eq!(F32_SIGN, 0x8000_0000);
    assert_eq!(F16_SIGN, 0x8000);
    assert_eq!(F16_EXPONENT, 0x7c00);
    // `static_cast<f32>` of 2^32 - 1, 2^31 - 1 and -2^31 (`ops_convert.cpp`, `F2I`).
    assert_eq!(U32_MAX_AS_F32, 4_294_967_296.0_f32.to_bits());
    assert_eq!(I32_MAX_AS_F32, 2_147_483_648.0_f32.to_bits());
    assert_eq!(I32_MIN_AS_F32, (-2_147_483_648.0_f32).to_bits());
}

/// The table's lanes are the operation bytes the PR names, as the test assembler encodes them
/// (bits 32..40, NAK's `set_field(32..40, subop)`).
#[test]
fn swizzled_add_operations_match_the_assembler() {
    const SUBOP: crate::decode::Field = crate::decode::Field::new(32, 8);
    for (ops, byte) in FSWZADD_OPERATIONS {
        let word =
            assemble(&Instr::new(Op::FSwzAdd(fswzadd(r(0), r(1), r(2), ops)))).expect("assembles");
        assert_eq!(word.field(SUBOP), u64::from(byte));
    }
}

/// Every source kind reads the bits the PR reads (`Visitor::Value`), seen through `FSEL`,
/// which copies the bits of the source it picks, and `FMUL` by 1 for the uniform registers: registers (`RZ` reads 0), uniform registers
/// (`URZ` reads 0), immediates, and constant-buffer words at every byte offset in a word.
#[test]
fn sources_of_every_kind() {
    let mut machine = Machine::new();
    machine.regs[1] = 0x1111_1111;
    machine.uniform_regs[5] = 0x2222_2222;
    machine
        .cbufs
        .insert(3, vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]);
    machine.cbufs.insert(31, vec![0x44, 0x33, 0x22, 0x11]);
    let cases = [
        (reg(1), 0x1111_1111),
        (AluSrc::reg(Gpr::RZ), 0),
        (AluSrc::Imm(0x1234_5678), 0x1234_5678),
        (cbuf(3, 0), 0x7654_3210),
        (cbuf(3, 1), 0x9876_5432),
        (cbuf(3, 2), 0xba98_7654),
        (cbuf(3, 3), 0xdcba_9876),
        (cbuf(3, 4), 0xfedc_ba98),
        // The last bank a source may name.
        (cbuf(CBUF_BANKS - 1, 0), 0x1122_3344),
    ];
    for (src, expected) in cases {
        // `FSEL dst, RZ, src, !PT` picks `src`.
        let op = OpFSel {
            dst: r(0),
            srcs: [AluSrc::reg(Gpr::RZ), src],
            cond: NOT_PT,
        };
        run_on(&[Instr::new(Op::FSel(op))], &mut machine);
        assert_eq!(machine.regs[0], expected, "{src:?}");
    }
    // `FSEL` has no uniform-register form: `FMUL dst, R2, src` with `R2 = 1.0` keeps the bits of
    // a normal value.
    machine.regs[2] = ONE;
    let urz = AluSrc::UniformReg {
        reg: UniformReg::URZ,
        abs: false,
        neg: false,
    };
    for (src, expected) in [(ureg(5), 0x2222_2222), (urz, 0)] {
        let op = fmul(r(0), reg(2), src);
        run_on(&[Instr::new(Op::FMul(op))], &mut machine);
        assert_eq!(machine.regs[0], expected, "{src:?}");
    }
    // A bank past the last is refused (a decoded bank has five bits, so this one is hand-made).
    let op = OpFSel {
        dst: r(0),
        srcs: [AluSrc::reg(Gpr::RZ), cbuf(CBUF_BANKS, 0)],
        cond: NOT_PT,
    };
    assert_refused_raw(
        Op::FSel(op),
        FloatUnsupported::ConstantBufferBank(CBUF_BANKS),
    );
}

/// The unaligned constant-buffer read joins two aligned words, as `Visitor::Cbuf` does.
#[test]
fn golden_unaligned_constant_buffer_read() {
    let op = fadd(r(0), reg(1), cbuf(2, 0x13));
    let function = translated(&[Instr::new(Op::FAdd(op))]);
    assert_eq!(
        function.to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r1\n\
         \x20   %1: f32 = BitCastF32U32 %0\n\
         \x20   %2: u32 = GetCbufU32 u32 0x2, u32 0x10\n\
         \x20   %3: u32 = GetCbufU32 u32 0x2, u32 0x14\n\
         \x20   %4: u32 = ShiftRightLogical32 %2, u32 0x18\n\
         \x20   %5: u32 = ShiftLeftLogical32 %3, u32 0x8\n\
         \x20   %6: u32 = BitwiseOr32 %4, %5\n\
         \x20   %7: f32 = BitCastF32U32 %6\n\
         \x20   %8: f32 = FPAdd32 %1, %7 {no_contraction, rn, fmz_none}\n\
         \x20   %9: u32 = BitCastU32F32 %8\n\
         \x20   %10 = SetRegister r0, %9\n\
         \x20   return\n"
    );
}

/// A guarded instruction becomes a conditional block: it runs where the guard holds and leaves
/// its destination as it was elsewhere, for a plain and a negated guard.
#[test]
fn guarded_instructions_become_conditional_blocks() {
    let op = Op::FAdd(fadd(r(0), reg(1), reg(2)));
    for negated in [false, true] {
        let instr = Instr {
            guard: pred(3, negated),
            op,
        };
        for p3 in [false, true] {
            let mut machine = Machine::new();
            machine.regs[0] = 0xdead_beef;
            machine.regs[1] = ONE;
            machine.regs[2] = TWO;
            machine.preds[3] = p3;
            run_on(&[instr], &mut machine);
            let runs = p3 != negated;
            let expected = if runs { 3.0_f32.to_bits() } else { 0xdead_beef };
            assert_eq!(machine.regs[0], expected, "negated {negated}, p3 {p3}");
        }
    }
    let instr = Instr {
        guard: pred(3, true),
        op,
    };
    assert_eq!(
        translated(&[instr]).to_string(),
        "block0:\n\
         \x20   %0: u1 = GetPred p3\n\
         \x20   %1: u1 = LogicalNot %0\n\
         \x20   branch %1, block1, block2\n\
         block1:\n\
         \x20   %2: u32 = GetRegister r1\n\
         \x20   %3: f32 = BitCastF32U32 %2\n\
         \x20   %4: u32 = GetRegister r2\n\
         \x20   %5: f32 = BitCastF32U32 %4\n\
         \x20   %6: f32 = FPAdd32 %3, %5 {no_contraction, rn, fmz_none}\n\
         \x20   %7: u32 = BitCastU32F32 %6\n\
         \x20   %8 = SetRegister r0, %7\n\
         \x20   jump block2\n\
         block2:\n\
         \x20   return\n"
    );
}

/// A guard that writes predicates: `FSETP` under `@P0` leaves its destinations as they were
/// when the guard is false.
#[test]
fn guarded_predicate_writes() {
    let op = fsetp(
        Pred::new(1).expect("p1"),
        Pred::new(2).expect("p2"),
        FloatCmpOp::Lt,
        reg(0),
        imm(1.0),
    );
    let instr = Instr {
        guard: pred(0, false),
        op: Op::FSetP(op),
    };
    for p0 in [false, true] {
        let mut machine = Machine::new();
        machine.preds[0] = p0;
        machine.preds[1] = false;
        machine.preds[2] = true;
        // 0.0 < 1.0: P1 true, P2 false when the guard holds.
        run_on(&[instr], &mut machine);
        assert_eq!((machine.preds[1], machine.preds[2]), (p0, !p0));
    }
}

/// `@!PT` never runs: the instruction is still checked (the PR skips it unchecked, a listed
/// divergence), and a valid one goes in a block behind a constant `false`.
#[test]
fn never_guarded_instructions_are_checked() {
    let refused = Op::FMul(OpFMul {
        dnz: true,
        ..fmul(r(0), reg(1), reg(2))
    });
    let instr = encodable(Instr {
        guard: NOT_PT,
        op: refused,
    });
    assert_eq!(
        translate_raw(&[instr]),
        refusal(instr, FloatUnsupported::FmulMultiplyZero)
    );
    let instr = Instr {
        guard: NOT_PT,
        op: Op::FAdd(fadd(r(0), reg(1), reg(2))),
    };
    let function = translated(&[instr]);
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

/// A block flow analysis already made conditional on the guard (`guard_tested`) holds the
/// instruction unconditionally: no blocks are added and the guard is not read.
#[test]
fn a_tested_guard_is_ignored() {
    let instr = Instr {
        guard: pred(2, false),
        op: Op::FAdd(fadd(r(0), reg(1), reg(1))),
    };
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    translate_instr(&mut builder, instr, true).expect("translates");
    assert_eq!(builder.block(), entry);
    builder.terminate(Terminator::Return).expect("terminates");
    assert_eq!(function.block_count(), 1);
    assert!(!opcodes(&function).contains(&Opcode::GetPred));
    let mut machine = Machine::new();
    machine.regs[1] = ONE;
    machine.run(&function).expect("runs");
    assert_eq!(machine.regs[0], TWO);
    // The same instruction with the guard untested adds its conditional block.
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    translate_instr(&mut builder, instr, false).expect("translates");
    assert_ne!(builder.block(), entry);
}

/// An instruction of another group is refused by its mnemonic, and nothing is appended.
#[test]
fn other_groups_are_refused() {
    let instr = Instr::new(Op::Mov(OpMov {
        dst: r(0),
        src: MovSrc::Imm(1),
        quad_lanes: OpMov::ALL_LANES,
    }));
    assert!(!in_group(&instr.op));
    let mut function = Function::new();
    let entry = function.entry();
    let before = function.clone();
    let mut builder = Builder::new(&mut function, entry);
    let error = translate_instr(&mut builder, instr, false);
    assert_eq!(
        error,
        Err(FloatError::Unsupported(FloatUnsupported::OtherGroup("mov")))
    );
    assert_eq!(
        error.map_err(|error| error.to_string()),
        Err("mov, which is not a float or conversion instruction, is not supported".into())
    );
    assert_eq!(function, before);
}

/// `in_group` names exactly the fifteen instructions of slice 6b.
#[test]
fn the_group_is_the_decoder_slice() {
    let members = [
        Op::FAdd(fadd(r(0), reg(1), reg(2))),
        Op::FMul(fmul(r(0), reg(1), reg(2))),
        Op::FFma(ffma(r(0), reg(1), reg(2), reg(3))),
        Op::FMnMx(fmnmx(r(0), reg(1), reg(2), PredSrc::TRUE)),
        Op::FSet(fset(r(0), FloatCmpOp::Lt, reg(1), reg(2))),
        Op::FSetP(fsetp(Pred::PT, Pred::PT, FloatCmpOp::Lt, reg(1), reg(2))),
        Op::FSel(OpFSel {
            dst: r(0),
            srcs: [reg(1), reg(2)],
            cond: PredSrc::TRUE,
        }),
        Op::MuFu(mufu(r(0), MuFuOp::Rcp, reg(1))),
        Op::FSwzAdd(fswzadd(r(0), r(1), r(2), FSWZADD_OPERATIONS[0].0)),
        Op::FRnd(frnd(r(0), reg(1), RoundMode::Zero)),
        Op::F2F(f2f(r(0), reg(1), RoundMode::NearestEven)),
        Op::F2I(f2i(r(0), reg(1), IntType::S32, RoundMode::Zero)),
        Op::I2F(i2f(r(0), reg(1), IntType::S32, RoundMode::NearestEven)),
        Op::I2FP(i2fp(r(0), reg(1), IntType::S32, RoundMode::NearestEven)),
        Op::F2FP(f2fp(r(0), reg(1), reg(2))),
    ];
    let mnemonics: Vec<&str> = members.iter().map(Op::mnemonic).collect();
    assert_eq!(
        mnemonics,
        [
            "fadd", "fmul", "ffma", "fmnmx", "fset", "fsetp", "fsel", "mufu", "fswzadd", "frnd",
            "f2f", "f2i", "i2f", "i2fp", "f2fp"
        ]
    );
    for op in members {
        assert!(in_group(&op), "{op}");
        let function = translated(&[Instr::new(op)]);
        assert_eq!(function.verify(), Ok(()));
    }
}

/// `FSWZADD`'s lane id makes the program need 32-wide subgroups; the other instructions do not.
#[test]
fn swizzled_add_needs_32_wide_subgroups() {
    let swizzle = translated(&[Instr::new(Op::FSwzAdd(fswzadd(
        r(0),
        r(1),
        r(2),
        FSWZADD_OPERATIONS[1].0,
    )))]);
    assert!(needs_subgroup_size_32(&swizzle));
    let add = translated(&[Instr::new(Op::FAdd(fadd(r(0), reg(1), reg(2))))]);
    assert!(!needs_subgroup_size_32(&add));
}

/// Each refusal and its exact words, including the full message through `translate`.
#[test]
fn refusals_read_as_words() {
    use FloatUnsupported as U;
    let cases = [
        (
            U::OtherGroup("mov"),
            "mov, which is not a float or conversion instruction,",
        ),
        (
            U::BindlessConstantBuffer,
            "a bindless constant-buffer source",
        ),
        (U::ConstantBufferBank(32), "constant buffer 32"),
        (U::WideConversion("f2i"), "the 64-bit form of f2i"),
        (U::MufuImmediate, "an immediate mufu operand"),
        (U::FfmaMultiplyZero, "ffma's multiply-by-zero mode"),
        (U::FmulMultiplyZero, "fmul's multiply-by-zero mode"),
        (
            U::FmulScaleModifiers,
            "a scaled fmul without ftz or with a rounding mode",
        ),
        (U::MufuFunction(MuFuOp::Tanh), "mufu function .tanh"),
        (
            U::HalfMufuSource,
            "a 16-bit mufu operand that is not a register",
        ),
        (
            U::SwizzledAddMode,
            "a swizzled add with ndv, a rounding mode or ftz",
        ),
        (
            U::SwizzledAddOperation([
                FSwzAddOp::Add,
                FSwzAddOp::SubLeft,
                FSwzAddOp::SubRight,
                FSwzAddOp::MoveLeft,
            ]),
            "swizzled add operations add sub subr mov2",
        ),
        (
            U::F2iDestinationSize(int(3, true)),
            "an f2i destination of 64 bits",
        ),
        (U::F2iSourceSize(FloatType::F64), "an f2i source of 64 bits"),
        (
            U::HalfF2iSource,
            "a 16-bit f2i operand that is not a register",
        ),
        (
            U::I2fDestinationSize(FloatType::F16),
            "an i2f destination of 16 bits",
        ),
        (U::I2fSourceSize(int(3, false)), "an i2f source of 64 bits"),
        (
            U::I2fSelect {
                select: 4,
                width: 8,
            },
            "select 4 of an i2f source of 8 bits",
        ),
        (
            U::FrndDestinationSize(FloatType::F64),
            "an frnd destination of 64 bits",
        ),
        (
            U::FrndSourceSize(FloatType::F16),
            "an frnd source of 16 bits",
        ),
        (
            U::F2fDestinationSize(FloatType::F32),
            "an f2f destination of 32 bits",
        ),
        (U::F2fSourceSize(FloatType::F64), "an f2f source of 64 bits"),
        (U::F2fpSelector, "f2fp's high-half selector"),
        (
            U::F2fpRounding(RoundMode::Zero),
            "f2fp rounding toward zero",
        ),
        (U::F2fpRounding(RoundMode::NegInf), "f2fp rounding down"),
        (U::F2fpRounding(RoundMode::PosInf), "f2fp rounding up"),
        (
            U::F2fpRounding(RoundMode::NearestEven),
            "f2fp rounding to nearest",
        ),
    ];
    for (what, words) in cases {
        assert_eq!(what.to_string(), words);
        assert_eq!(
            FloatError::Unsupported(what).to_string(),
            format!("{words} is not supported")
        );
    }
}

/// A refusal's full message through `translate`.
#[test]
fn a_refusal_names_the_instruction() {
    let instr = Instr::new(Op::FMul(OpFMul {
        dnz: true,
        ..fmul(r(0), reg(1), reg(2))
    }));
    assert_eq!(
        translate_raw(&[instr]).map_err(|error| error.to_string()),
        Err(format!(
            "instruction 0 ({instr}): fmul's multiply-by-zero mode is not supported"
        ))
    );
}

/// Golden IR for the emulated conversions and the swizzled add.
#[test]
fn golden_emulated_conversions() {
    let to_half = OpF2F {
        ftz: true,
        ..f2f(r(0), reg(1), RoundMode::Zero)
    };
    let to_int = OpF2I {
        ftz: true,
        ..f2i(r(2), reg(3), int(1, true), RoundMode::NegInf)
    };
    let swizzle = fswzadd(r(4), r(5), r(6), FSWZADD_OPERATIONS[0].0);
    for (op, golden) in [
        (
            Op::F2F(to_half),
            "block0:\n\
         \x20   %0: u32 = GetRegister r1\n\
         \x20   %1: f32 = BitCastF32U32 %0\n\
         \x20   %2: f32 = FPAbs32 %1\n\
         \x20   %3: u1 = FPOrdLessThan32 %2, f32 0x800000 {fmz_none}\n\
         \x20   %4: u32 = BitCastU32F32 %1\n\
         \x20   %5: u32 = BitwiseAnd32 %4, u32 0x80000000\n\
         \x20   %6: f32 = BitCastF32U32 %5\n\
         \x20   %7: f32 = SelectF32 %3, %6, %1\n\
         \x20   %8: f16 = ConvertF16F32 %7 {rz, ftz}\n\
         \x20   %9: f16 = ConvertF16F32 f32 0x0\n\
         \x20   %10: f16x2 = CompositeConstructF16x2 %8, %9\n\
         \x20   %11: u32 = PackFloat2x16 %10\n\
         \x20   %12: f32 = ConvertF32F16 %8 {rn, fmz_none}\n\
         \x20   %13: u32 = ISub32 %11, u32 0x1\n\
         \x20   %14: f32 = FPAbs32 %12\n\
         \x20   %15: f32 = FPAbs32 %7\n\
         \x20   %16: u1 = FPOrdGreaterThan32 %14, %15 {fmz_none}\n\
         \x20   %17: u32 = SelectU32 %16, %13, %11\n\
         \x20   %18: u32 = BitwiseAnd32 %17, u32 0x7c00\n\
         \x20   %19: u1 = IEqual %18, u32 0x0\n\
         \x20   %20: u32 = BitwiseAnd32 %17, u32 0x8000\n\
         \x20   %21: u32 = SelectU32 %19, %20, %17\n\
         \x20   %22 = SetRegister r0, %21\n\
         \x20   return\n",
        ),
        (
            Op::F2I(to_int),
            "block0:\n\
         \x20   %0: u32 = GetRegister r3\n\
         \x20   %1: f32 = BitCastF32U32 %0\n\
         \x20   %2: f32 = FPAbs32 %1\n\
         \x20   %3: u1 = FPOrdLessThan32 %2, f32 0x800000 {fmz_none}\n\
         \x20   %4: u32 = BitCastU32F32 %1\n\
         \x20   %5: u32 = BitwiseAnd32 %4, u32 0x80000000\n\
         \x20   %6: f32 = BitCastF32U32 %5\n\
         \x20   %7: f32 = SelectF32 %3, %6, %1\n\
         \x20   %8: f32 = FPFloor32 %7 {no_contraction, ftz}\n\
         \x20   %9: f32 = FPClamp32 %8, f32 0xc7000000, f32 0x46fffe00\n\
         \x20   %10: u32 = ConvertS32F32 %9\n\
         \x20   %11: u1 = FPIsNan32 %1\n\
         \x20   %12: u32 = SelectU32 %11, u32 0x0, %10\n\
         \x20   %13 = SetRegister r2, %12\n\
         \x20   return\n",
        ),
        (
            Op::FSwzAdd(swizzle),
            "block0:\n\
         \x20   %0: u32 = GetRegister r5\n\
         \x20   %1: f32 = BitCastF32U32 %0\n\
         \x20   %2: u32 = GetRegister r6\n\
         \x20   %3: f32 = BitCastF32U32 %2\n\
         \x20   %4: f32 = FSwizzleAdd %1, %3, u32 0xa5 {rn, fmz_none}\n\
         \x20   %5: u32 = BitCastU32F32 %4\n\
         \x20   %6 = SetRegister r4, %5\n\
         \x20   return\n",
        ),
    ] {
        assert_eq!(translated(&[Instr::new(op)]).to_string(), golden, "{op}");
    }
}

/// Every refusal is decided before anything is appended: the function is unchanged, and the
/// builder stays in its block. One case per refusal reachable from a decoded operation.
#[test]
fn refusals_append_nothing() {
    for (op, what) in refusal_cases() {
        for guard in [PredSrc::TRUE, pred(1, false), NOT_PT] {
            let mut function = Function::new();
            let entry = function.entry();
            let mut builder = Builder::new(&mut function, entry);
            builder
                .op(Opcode::GetRegister, &[Operand::Imm(Imm::Reg(r(9)))])
                .expect("appends");
            let before = builder.function().clone();
            let result = translate_instr(&mut builder, Instr { guard, op }, false);
            assert_eq!(result, Err(FloatError::Unsupported(what)), "{op}");
            assert_eq!(builder.block(), entry);
            assert_eq!(builder.function(), &before, "{op}");
        }
    }
}

/// One encodable operation per refusal that a decoded operation can reach, with its refusal.
fn refusal_cases() -> Vec<(Op, FloatUnsupported)> {
    use FloatUnsupported as U;
    let bindless = AluSrc::CBuf {
        cbuf: CBufRef {
            buf: CBuf::BindlessUniform(UniformReg::new(3).expect("ur3")),
            offset: 8,
        },
        abs: false,
        neg: false,
    };
    let mut cases = vec![
        (
            Op::FAdd(fadd(r(0), reg(1), bindless)),
            U::BindlessConstantBuffer,
        ),
        (
            Op::F2I(OpF2I {
                wide: true,
                ..f2i(r(0), reg(1), IntType::S32, RoundMode::Zero)
            }),
            U::WideConversion("f2i"),
        ),
        (
            Op::MuFu(mufu(r(0), MuFuOp::Rcp, imm(2.0))),
            U::MufuImmediate,
        ),
        (
            Op::FFma(OpFFma {
                dnz: true,
                ..ffma(r(0), reg(1), reg(2), reg(3))
            }),
            U::FfmaMultiplyZero,
        ),
        (
            Op::FMul(OpFMul {
                dnz: true,
                ..fmul(r(0), reg(1), reg(2))
            }),
            U::FmulMultiplyZero,
        ),
        (
            Op::FMul(OpFMul {
                scale: FMulScale::M2,
                ..fmul(r(0), reg(1), reg(2))
            }),
            U::FmulScaleModifiers,
        ),
        (
            Op::MuFu(mufu(r(0), MuFuOp::Tanh, reg(1))),
            U::MufuFunction(MuFuOp::Tanh),
        ),
        (
            Op::MuFu(OpMuFu {
                f16: true,
                ..mufu(r(0), MuFuOp::Rcp, cbuf(1, 0))
            }),
            U::HalfMufuSource,
        ),
        (
            Op::FSwzAdd(OpFSwzAdd {
                ndv: true,
                ..fswzadd(r(0), r(1), r(2), FSWZADD_OPERATIONS[0].0)
            }),
            U::SwizzledAddMode,
        ),
        (
            Op::FSwzAdd(fswzadd(r(0), r(1), r(2), [FSwzAddOp::Add; 4])),
            U::SwizzledAddOperation([FSwzAddOp::Add; 4]),
        ),
    ];
    cases.extend(conversion_refusal_cases());
    cases
}

/// [`refusal_cases`]'s conversions.
fn conversion_refusal_cases() -> Vec<(Op, FloatUnsupported)> {
    use FloatUnsupported as U;
    vec![
        (
            Op::F2I(f2i(r(0), reg(1), int(3, true), RoundMode::Zero)),
            U::F2iDestinationSize(int(3, true)),
        ),
        (
            Op::F2I(OpF2I {
                src_type: FloatType::F64,
                ..f2i(r(0), reg(1), IntType::S32, RoundMode::Zero)
            }),
            U::F2iSourceSize(FloatType::F64),
        ),
        (
            Op::F2I(OpF2I {
                src_type: FloatType::F16,
                ..f2i(r(0), cbuf(1, 0), IntType::S32, RoundMode::Zero)
            }),
            U::HalfF2iSource,
        ),
        (
            Op::I2F(crate::decode::OpI2F {
                dst_type: FloatType::F64,
                ..i2f(r(0), reg(1), IntType::S32, RoundMode::NearestEven)
            }),
            U::I2fDestinationSize(FloatType::F64),
        ),
        (
            Op::I2F(i2f(r(0), reg(1), int(3, true), RoundMode::NearestEven)),
            U::I2fSourceSize(int(3, true)),
        ),
        (
            Op::FRnd(OpFRnd {
                dst_type: FloatType::F16,
                ..frnd(r(0), reg(1), RoundMode::Zero)
            }),
            U::FrndDestinationSize(FloatType::F16),
        ),
        (
            Op::FRnd(OpFRnd {
                src_type: FloatType::F64,
                ..frnd(r(0), reg(1), RoundMode::Zero)
            }),
            U::FrndSourceSize(FloatType::F64),
        ),
        (
            Op::F2F(OpF2F {
                dst_type: FloatType::F32,
                ..f2f(r(0), reg(1), RoundMode::Zero)
            }),
            U::F2fDestinationSize(FloatType::F32),
        ),
        (
            Op::F2F(OpF2F {
                src_type: FloatType::F16,
                ..f2f(r(0), reg(1), RoundMode::Zero)
            }),
            U::F2fSourceSize(FloatType::F16),
        ),
        (
            Op::F2FP(OpF2FP {
                merge_c: true,
                h1: true,
                ..f2fp(r(0), reg(1), reg(2))
            }),
            U::F2fpSelector,
        ),
        (
            Op::F2FP(OpF2FP {
                rnd: RoundMode::Zero,
                ..f2fp(r(0), reg(1), reg(2))
            }),
            U::F2fpRounding(RoundMode::Zero),
        ),
    ]
}

/// Every case of [`refusal_cases`] encodes and is refused with its error through `translate`.
#[test]
fn each_refusal_through_translate() {
    for (op, what) in refusal_cases() {
        assert_refused(op, what);
    }
}

/// The forms the decoder reads and the PR refuses are refused here, as listed choices: every
/// 64-bit float size, the 64-bit opcodes of `FRND`, `F2F`, `F2I` and `I2F`, `MUFU`'s immediate
/// form and its three functions the PR has no case for, and the `FSWZADD` operation bytes other
/// than the PR's two. `FMUL`'s scale 0 and a float size of 0, which the PR also refuses, the
/// decoder refuses (slice 4b), so every decoded scale is translated.
#[test]
fn decoded_forms_the_pr_refuses_are_refused() {
    use FloatUnsupported as U;
    // Every scale is translated, with FTZ and to-nearest where it scales.
    for scale in FMulScale::ALL {
        translated(&[Instr::new(Op::FMul(OpFMul {
            scale,
            ftz: true,
            ..fmul(r(0), reg(1), reg(2))
        }))]);
    }
    let f64 = FloatType::F64;
    assert_refused(
        Op::F2I(OpF2I {
            src_type: f64,
            ..f2i(r(0), reg(1), IntType::S32, RoundMode::Zero)
        }),
        U::F2iSourceSize(f64),
    );
    assert_refused(
        Op::I2F(crate::decode::OpI2F {
            dst_type: f64,
            ..i2f(r(0), reg(1), IntType::S32, RoundMode::Zero)
        }),
        U::I2fDestinationSize(f64),
    );
    assert_refused(
        Op::I2FP(OpI2FP {
            dst_type: f64,
            ..i2fp(r(0), reg(1), IntType::S32, RoundMode::Zero)
        }),
        U::I2fDestinationSize(f64),
    );
    for (dst_type, src_type) in [(f64, FloatType::F32), (FloatType::F32, f64)] {
        let what = if dst_type == f64 {
            U::FrndDestinationSize(f64)
        } else {
            U::FrndSourceSize(f64)
        };
        assert_refused(
            Op::FRnd(OpFRnd {
                dst_type,
                src_type,
                ..frnd(r(0), reg(1), RoundMode::Zero)
            }),
            what,
        );
    }
    for (dst_type, src_type, what) in [
        (f64, FloatType::F32, U::F2fDestinationSize(f64)),
        (FloatType::F16, f64, U::F2fSourceSize(f64)),
    ] {
        assert_refused(
            Op::F2F(OpF2F {
                dst_type,
                src_type,
                ..f2f(r(0), reg(1), RoundMode::Zero)
            }),
            what,
        );
    }
}

/// The 64-bit opcodes the decoder reads and `Visitor::Run` sends nowhere, with 32-bit types so
/// nothing else is wrong.
#[test]
fn wide_conversion_opcodes_are_refused() {
    use FloatUnsupported as U;
    assert_refused(
        Op::FRnd(OpFRnd {
            wide: true,
            ..frnd(r(0), reg(1), RoundMode::Zero)
        }),
        U::WideConversion("frnd"),
    );
    assert_refused(
        Op::F2F(OpF2F {
            wide: true,
            ..f2f(r(0), reg(1), RoundMode::Zero)
        }),
        U::WideConversion("f2f"),
    );
    assert_refused(
        Op::F2I(OpF2I {
            wide: true,
            ..f2i(r(0), reg(1), IntType::S32, RoundMode::Zero)
        }),
        U::WideConversion("f2i"),
    );
    assert_refused(
        Op::I2F(crate::decode::OpI2F {
            wide: true,
            ..i2f(r(0), reg(1), IntType::S32, RoundMode::Zero)
        }),
        U::WideConversion("i2f"),
    );
    // PR #4588's test "Native unsigned 64-bit float conversion preserves both output words"
    // (an `F2I.U64`, `0x311`) runs on the PR's Maxwell lowering; its native translator has no
    // case for the opcode, so it is refused here.
    assert_refused(
        Op::F2I(OpF2I {
            wide: true,
            ..f2i(r(4), reg(12), int(3, false), RoundMode::Zero)
        }),
        U::WideConversion("f2i"),
    );
}

/// `MUFU`'s immediate form and its functions the PR has no case for, and the `FSWZADD`
/// operation bytes other than the PR's two.
#[test]
fn mufu_and_fswzadd_forms_the_pr_refuses_are_refused() {
    use FloatUnsupported as U;
    assert_refused(
        Op::MuFu(mufu(r(0), MuFuOp::Sqrt, imm(4.0))),
        U::MufuImmediate,
    );
    // PR #4588's test "Native hyperbolic tangent retains its function modifiers and execution
    // guard" runs `MUFU.TANH` on the Maxwell lowering; the native translator refuses it.
    for op in [MuFuOp::Rcp64H, MuFuOp::Rsq64H, MuFuOp::Tanh] {
        assert_refused(
            Op::MuFu(mufu(r(0), op, reg_mod(0, true, true))),
            U::MufuFunction(op),
        );
    }
    // Every lane list but the PR's two.
    let mut refused = 0;
    for code in 0..=u8::MAX {
        let ops = [6, 4, 2, 0].map(|shift| FSwzAddOp::from_bits(u64::from(code >> shift)));
        let op = Op::FSwzAdd(fswzadd(r(0), r(1), r(2), ops));
        if FSWZADD_OPERATIONS.iter().any(|&(lanes, _)| lanes == ops) {
            translated(&[Instr::new(op)]);
        } else {
            assert_refused(op, U::SwizzledAddOperation(ops));
            refused += 1;
        }
    }
    assert_eq!(refused, 256 - FSWZADD_OPERATIONS.len());
}

/// When an instruction has two faults, the one the PR checks first is reported.
#[test]
fn refusals_report_in_the_prs_order() {
    use FloatUnsupported as U;
    let bindless = AluSrc::CBuf {
        cbuf: CBufRef {
            buf: CBuf::BindlessUniform(UniformReg::new(3).expect("ur3")),
            offset: 0,
        },
        abs: false,
        neg: false,
    };
    let encoded = [
        // A source slot before the instruction's own checks.
        (
            Op::FMul(OpFMul {
                dnz: true,
                ..fmul(r(0), reg(1), bindless)
            }),
            U::BindlessConstantBuffer,
        ),
        (
            Op::FFma(OpFFma {
                dnz: true,
                ..ffma(r(0), reg(1), bindless, reg(2))
            }),
            U::BindlessConstantBuffer,
        ),
        // FMUL: multiply-by-zero, then the scale's modifiers.
        (
            Op::FMul(OpFMul {
                dnz: true,
                scale: FMulScale::M2,
                ..fmul(r(0), reg(1), reg(2))
            }),
            U::FmulMultiplyZero,
        ),
    ];
    for (op, what) in encoded {
        assert_refused(op, what);
    }
}

/// [`refusals_report_in_the_prs_order`] for `MUFU`, `FSWZADD` and the conversions.
#[test]
fn conversion_refusals_report_in_the_prs_order() {
    use FloatUnsupported as U;
    let bindless = AluSrc::CBuf {
        cbuf: CBufRef {
            buf: CBuf::BindlessUniform(UniformReg::new(3).expect("ur3")),
            offset: 0,
        },
        abs: false,
        neg: false,
    };
    let encoded = [
        // MUFU: the form, the slot, the 16-bit operand, the function.
        (
            Op::MuFu(mufu(r(0), MuFuOp::Tanh, imm(1.0))),
            U::MufuImmediate,
        ),
        (
            Op::MuFu(OpMuFu {
                f16: true,
                ..mufu(r(0), MuFuOp::Tanh, bindless)
            }),
            U::BindlessConstantBuffer,
        ),
        (
            Op::MuFu(OpMuFu {
                f16: true,
                ..mufu(r(0), MuFuOp::Tanh, cbuf(1, 0))
            }),
            U::HalfMufuSource,
        ),
        // FSWZADD: the mode, then the operations.
        (
            Op::FSwzAdd(OpFSwzAdd {
                ftz: true,
                ..fswzadd(r(0), r(1), r(2), [FSwzAddOp::Add; 4])
            }),
            U::SwizzledAddMode,
        ),
        // F2I: the opcode, the slot, the destination, the source, the 16-bit operand.
        (
            Op::F2I(OpF2I {
                wide: true,
                src_type: FloatType::F64,
                ..f2i(r(0), bindless, int(3, true), RoundMode::Zero)
            }),
            U::WideConversion("f2i"),
        ),
        (
            Op::F2I(OpF2I {
                src_type: FloatType::F64,
                ..f2i(r(0), bindless, int(3, true), RoundMode::Zero)
            }),
            U::BindlessConstantBuffer,
        ),
        (
            Op::F2I(OpF2I {
                src_type: FloatType::F64,
                ..f2i(r(0), reg(1), int(3, true), RoundMode::Zero)
            }),
            U::F2iDestinationSize(int(3, true)),
        ),
        (
            Op::F2I(OpF2I {
                src_type: FloatType::F64,
                ..f2i(r(0), cbuf(1, 0), IntType::S32, RoundMode::Zero)
            }),
            U::F2iSourceSize(FloatType::F64),
        ),
    ];
    for (op, what) in encoded {
        assert_refused(op, what);
    }
}

/// [`refusals_report_in_the_prs_order`] for `I2F`, `FRND`, `F2F` and `F2FP`.
#[test]
fn more_conversion_refusals_report_in_the_prs_order() {
    use FloatUnsupported as U;
    let bindless = AluSrc::CBuf {
        cbuf: CBufRef {
            buf: CBuf::BindlessUniform(UniformReg::new(3).expect("ur3")),
            offset: 0,
        },
        abs: false,
        neg: false,
    };
    let encoded = [
        // I2F: the opcode, the destination, the source.
        (
            Op::I2F(crate::decode::OpI2F {
                wide: true,
                dst_type: FloatType::F16,
                ..i2f(r(0), reg(1), int(3, false), RoundMode::Zero)
            }),
            U::WideConversion("i2f"),
        ),
        (
            Op::I2F(crate::decode::OpI2F {
                dst_type: FloatType::F16,
                ..i2f(r(0), reg(1), int(3, false), RoundMode::Zero)
            }),
            U::I2fDestinationSize(FloatType::F16),
        ),
        // FRND and F2F: the opcode, the destination, then the source.
        (
            Op::FRnd(OpFRnd {
                wide: true,
                dst_type: FloatType::F16,
                src_type: FloatType::F16,
                ..frnd(r(0), reg(1), RoundMode::Zero)
            }),
            U::WideConversion("frnd"),
        ),
        (
            Op::FRnd(OpFRnd {
                dst_type: FloatType::F16,
                src_type: FloatType::F16,
                ..frnd(r(0), reg(1), RoundMode::Zero)
            }),
            U::FrndDestinationSize(FloatType::F16),
        ),
        (
            Op::F2F(OpF2F {
                wide: true,
                dst_type: FloatType::F32,
                src_type: FloatType::F16,
                ..f2f(r(0), reg(1), RoundMode::Zero)
            }),
            U::WideConversion("f2f"),
        ),
        (
            Op::F2F(OpF2F {
                dst_type: FloatType::F32,
                src_type: FloatType::F16,
                ..f2f(r(0), reg(1), RoundMode::Zero)
            }),
            U::F2fDestinationSize(FloatType::F32),
        ),
        // F2FP: the slot, the selector, the rounding.
        (
            Op::F2FP(OpF2FP {
                merge_c: true,
                h1: true,
                rnd: RoundMode::Zero,
                ..f2fp(r(0), reg(1), bindless)
            }),
            U::BindlessConstantBuffer,
        ),
        (
            Op::F2FP(OpF2FP {
                merge_c: true,
                h1: true,
                rnd: RoundMode::Zero,
                ..f2fp(r(0), reg(1), reg(2))
            }),
            U::F2fpSelector,
        ),
    ];
    for (op, what) in encoded {
        assert_refused(op, what);
    }
}

/// The bank is read with the value, after the instruction's own checks (hand-made: no word
/// names bank 32).
#[test]
fn the_bank_is_checked_last() {
    use FloatUnsupported as U;
    let past = cbuf(CBUF_BANKS, 0);
    let raw = [
        (
            Op::FMul(OpFMul {
                scale: FMulScale::M2,
                ..fmul(r(0), reg(1), past)
            }),
            U::FmulScaleModifiers,
        ),
        (
            Op::FFma(OpFFma {
                dnz: true,
                ..ffma(r(0), reg(1), past, reg(2))
            }),
            U::FfmaMultiplyZero,
        ),
        (
            Op::MuFu(mufu(r(0), MuFuOp::Tanh, past)),
            U::ConstantBufferBank(CBUF_BANKS),
        ),
        (
            Op::F2I(OpF2I {
                src_type: FloatType::F64,
                ..f2i(r(0), past, IntType::S32, RoundMode::Zero)
            }),
            U::F2iSourceSize(FloatType::F64),
        ),
        (
            Op::I2F(crate::decode::OpI2F {
                select: 1,
                ..i2f(r(0), past, IntType::S32, RoundMode::Zero)
            }),
            U::I2fSelect {
                select: 1,
                width: 32,
            },
        ),
        (
            Op::I2F(i2f(r(0), past, int(3, false), RoundMode::Zero)),
            U::I2fSourceSize(int(3, false)),
        ),
        (
            Op::FRnd(OpFRnd {
                src_type: FloatType::F16,
                ..frnd(r(0), past, RoundMode::Zero)
            }),
            U::FrndSourceSize(FloatType::F16),
        ),
        (
            Op::F2F(OpF2F {
                src_type: FloatType::F16,
                ..f2f(r(0), past, RoundMode::Zero)
            }),
            U::F2fSourceSize(FloatType::F16),
        ),
        (
            Op::F2FP(OpF2FP {
                rnd: RoundMode::Zero,
                ..f2fp(r(0), reg(1), past)
            }),
            U::F2fpRounding(RoundMode::Zero),
        ),
        // A 16-bit MUFU operand is a register or refused: its bank is never read.
        (
            Op::MuFu(OpMuFu {
                f16: true,
                ..mufu(r(0), MuFuOp::Rcp, past)
            }),
            U::HalfMufuSource,
        ),
        // Two-source instructions check the first source's bank before the second's.
        (
            Op::FSel(OpFSel {
                dst: r(0),
                srcs: [cbuf(CBUF_BANKS + 1, 0), past],
                cond: PredSrc::TRUE,
            }),
            U::ConstantBufferBank(CBUF_BANKS + 1),
        ),
    ];
    for (op, what) in raw {
        assert_refused_raw(op, what);
    }
}

/// An instruction built around one source.
type OpWith = fn(AluSrc) -> Op;

/// Each instruction's bank check refuses on its own: a source in the first bank past the last
/// ([`CBUF_BANKS`]) is the only fault, and the refusal names that bank; the same instruction
/// with the last bank translates (hand-made: no word names bank 32). `FSEL`'s and `MUFU`'s are
/// in [`the_bank_is_checked_last`].
#[test]
fn each_bank_check_refuses_on_its_own() {
    let last = CBUF_BANKS - 1;
    let cases: [(&str, OpWith); 11] = [
        ("fadd", |src| Op::FAdd(fadd(r(0), reg(1), src))),
        ("fmul", |src| Op::FMul(fmul(r(0), reg(1), src))),
        ("ffma b", |src| Op::FFma(ffma(r(0), reg(1), src, reg(2)))),
        ("ffma c", |src| Op::FFma(ffma(r(0), reg(1), reg(2), src))),
        ("frnd", |src| Op::FRnd(frnd(r(0), src, RoundMode::Zero))),
        ("f2f", |src| Op::F2F(f2f(r(0), src, RoundMode::Zero))),
        ("f2i.f32", |src| {
            Op::F2I(f2i(r(0), src, IntType::S32, RoundMode::Zero))
        }),
        ("i2f", |src| {
            Op::I2F(i2f(r(0), src, IntType::S32, RoundMode::Zero))
        }),
        ("i2f.u8", |src| {
            Op::I2F(crate::decode::OpI2F {
                select: 3,
                ..i2f(r(0), src, int(0, false), RoundMode::Zero)
            })
        }),
        ("i2fp", |src| {
            Op::I2FP(i2fp(r(0), src, IntType::S32, RoundMode::Zero))
        }),
        ("f2fp", |src| Op::F2FP(f2fp(r(0), reg(1), src))),
    ];
    for (name, case) in cases {
        let past = Instr::new(case(cbuf(CBUF_BANKS, 0)));
        assert_eq!(
            translate_raw(&[past]),
            refusal(past, FloatUnsupported::ConstantBufferBank(CBUF_BANKS)),
            "{name}"
        );
        let accepted = translate_raw(&[Instr::new(case(cbuf(last, 0)))]);
        assert_eq!(
            accepted.as_ref().map(Function::verify),
            Ok(Ok(())),
            "{name}"
        );
    }
}

/// `I2F` checks its source size before its select (`ops_convert.cpp`, `I2F`: "native
/// integer-to-float source size", then "source select"): a 64-bit source with select 1 has both
/// faults and is refused for its size; the select alone, on a 32-bit source, is refused for the
/// select (hand-made: the decoder gives a 64-bit or 32-bit source no select).
#[test]
fn i2f_source_size_comes_before_the_select() {
    let both = crate::decode::OpI2F {
        select: 1,
        ..i2f(r(0), reg(1), int(3, false), RoundMode::Zero)
    };
    assert_refused_raw(
        Op::I2F(both),
        FloatUnsupported::I2fSourceSize(int(3, false)),
    );
    let select_only = crate::decode::OpI2F {
        src_type: IntType::U32,
        ..both
    };
    assert_refused_raw(
        Op::I2F(select_only),
        FloatUnsupported::I2fSelect {
            select: 1,
            width: u32::BITS,
        },
    );
}

/// The 16-bit forms' operand read refuses anything but a register with the calling
/// instruction's own refusal, and appends nothing; [`check`] refuses those operands before
/// emission, so this is driven on the emitter directly.
#[test]
fn a_half_operand_is_refused_with_the_callers_name() {
    let cases = [
        (
            Op::MuFu(OpMuFu {
                f16: true,
                ..mufu(r(0), MuFuOp::Rcp, imm(1.0))
            }),
            FloatUnsupported::HalfMufuSource,
        ),
        (
            Op::F2I(OpF2I {
                src_type: FloatType::F16,
                ..f2i(r(0), cbuf(1, 0), IntType::S32, RoundMode::Zero)
            }),
            FloatUnsupported::HalfF2iSource,
        ),
    ];
    for (op, what) in cases {
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        assert_eq!(
            Emit { b: &mut builder }.op(&op),
            Err(FloatError::Unsupported(what)),
            "{op}"
        );
        assert_eq!(function.listed().count(), 0, "{op}");
    }
}

/// The float group's generators (`decode`'s per-group hooks), found by the group whose forms
/// hold `FADD`'s opcode.
fn float_ops() -> Vec<BoxedStrategy<Op>> {
    let word = assemble(&Instr::new(Op::FAdd(fadd(r(0), reg(1), reg(2))))).expect("assembles");
    let opcode = word.field(OPCODE);
    let index = GROUPS
        .iter()
        .position(|forms| forms.iter().any(|form| u64::from(form.opcode) == opcode))
        .expect("a group decodes fadd");
    (GROUP_HOOKS[index].any_ops)()
}

proptest! {
    #![proptest_config(config())]

    /// Every decoded float instruction under every guard translates to IR that verifies, or is
    /// refused with nothing appended; none panics.
    #[test]
    fn decoded_float_instructions_never_panic(
        op in proptest::strategy::Union::new(float_ops()),
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
            Err(FloatError::Unsupported(_)) => prop_assert_eq!(builder.function(), &before),
            Err(FloatError::Build(error)) => prop_assert!(false, "{}", error),
        }
        if let Ok(function) = translate_raw(&[instr]) {
            prop_assert_eq!(function.verify(), Ok(()));
        }
    }
}
