//! Translator tests: golden IR, each rejection with its exact error, and the limits at their
//! edges.

use proptest::prelude::*;

use super::*;
use crate::asm::assemble;
use crate::decode::{
    AluSrc, CBufRef, OPCODE, OpAL2P, OpALd, OpASt, OpCS2R, OpExit, OpFFma, OpIpa, OpPixLd, OpS2R,
    PixVal, Pred, PredSrc, RoundMode, S2RDst, UniformReg,
};
use crate::interp::Machine;
use crate::testing::{any_instr, config};

fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

const VERTEX: Stage = Stage {
    stage: GuestStage::Vertex,
    color_outputs: ColorOutputs {
        targets: [0; COLOR_TARGETS],
    },
};

fn pixel(targets: [u8; COLOR_TARGETS]) -> Stage {
    Stage {
        stage: GuestStage::Pixel,
        color_outputs: ColorOutputs { targets },
    }
}

fn exit() -> Instr {
    Instr::new(Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    }))
}

fn mov(dst: Gpr, src: MovSrc) -> Instr {
    Instr::new(Op::Mov(OpMov {
        dst,
        src,
        quad_lanes: OpMov::ALL_LANES,
    }))
}

fn ald(dst: Gpr, addr: u16, comps: u8) -> OpALd {
    OpALd {
        dst,
        vtx: Gpr::RZ,
        offset: Gpr::RZ,
        addr,
        comps,
        patch: false,
        phys: false,
        output: false,
    }
}

fn ast(data: Gpr, addr: u16, comps: u8) -> OpASt {
    OpASt {
        data,
        vtx: Gpr::RZ,
        offset: Gpr::RZ,
        addr,
        comps,
        patch: false,
        phys: false,
    }
}

/// Translates `instr` followed by `EXIT` in a vertex program.
fn one(instr: Instr) -> Result<Function, TranslateError> {
    translate(VERTEX, &[instr, exit()])
}

/// The rejection `one(instr)` should give.
fn unsupported(instr: Instr, what: Unsupported) -> Result<Function, TranslateError> {
    Err(TranslateError::Unsupported {
        index: 0,
        instr,
        what,
    })
}

#[test]
fn vertex_program_golden_ir() {
    let program = [
        Instr::new(Op::ALd(ald(r(0), 0x80, 2))),
        Instr::new(Op::FFma(OpFFma {
            dst: r(2),
            srcs: [
                AluSrc::reg(r(0)),
                AluSrc::Imm(0.5_f32.to_bits()),
                AluSrc::Reg {
                    reg: r(1),
                    abs: true,
                    neg: true,
                },
            ],
            saturate: true,
            rnd: RoundMode::Zero,
            ftz: true,
            dnz: false,
        })),
        mov(r(3), MovSrc::Reg(Gpr::RZ)),
        mov(Gpr::RZ, MovSrc::Imm(7)),
        Instr::new(Op::ASt(ast(r(2), 0x70, 2))),
        Instr::new(Op::ASt(ast(Gpr::RZ, 0x78, 1))),
        exit(),
    ];
    let ir = translate(VERTEX, &program).expect("translates");
    assert_eq!(ir.verify(), Ok(()));
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: f32 = GetAttribute a0x80, u32 0x0\n\
         \x20   %1: u32 = BitCastU32F32 %0\n\
         \x20   %2 = SetRegister r0, %1\n\
         \x20   %3: f32 = GetAttribute a0x84, u32 0x0\n\
         \x20   %4: u32 = BitCastU32F32 %3\n\
         \x20   %5 = SetRegister r1, %4\n\
         \x20   %6: u32 = GetRegister r0\n\
         \x20   %7: f32 = BitCastF32U32 %6\n\
         \x20   %8: f32 = BitCastF32U32 u32 0x3f000000\n\
         \x20   %9: u32 = GetRegister r1\n\
         \x20   %10: f32 = BitCastF32U32 %9\n\
         \x20   %11: f32 = FPAbs32 %10\n\
         \x20   %12: f32 = FPNeg32 %11\n\
         \x20   %13: f32 = FPFma32 %7, %8, %12 {no_contraction, rz, ftz}\n\
         \x20   %14: f32 = FPSaturate32 %13\n\
         \x20   %15: u32 = BitCastU32F32 %14\n\
         \x20   %16 = SetRegister r2, %15\n\
         \x20   %17 = SetRegister r3, u32 0x0\n\
         \x20   %18: u32 = GetRegister r2\n\
         \x20   %19: f32 = BitCastF32U32 %18\n\
         \x20   %20 = SetAttribute a0x70, %19, u32 0x0\n\
         \x20   %21: u32 = GetRegister r3\n\
         \x20   %22: f32 = BitCastF32U32 %21\n\
         \x20   %23 = SetAttribute a0x74, %22, u32 0x0\n\
         \x20   %24: f32 = BitCastF32U32 u32 0x0\n\
         \x20   %25 = SetAttribute a0x78, %24, u32 0x0\n\
         \x20   return\n"
    );
}

/// The colour registers are packed four per written target, skipping targets with no
/// component, a register per component whether written or not.
#[test]
fn pixel_exit_reads_colour_registers_in_target_order() {
    let mut targets = [0; COLOR_TARGETS];
    targets[0] = 0b1010;
    targets[2] = 0b0001;
    targets[7] = 0xf0; // Only bits above the component mask: no components.
    let ir = translate(pixel(targets), &[exit()]).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u32 = GetRegister r1\n\
         \x20   %1: f32 = BitCastF32U32 %0\n\
         \x20   %2 = SetFragColor u32 0x0, u32 0x1, %1\n\
         \x20   %3: u32 = GetRegister r3\n\
         \x20   %4: f32 = BitCastF32U32 %3\n\
         \x20   %5 = SetFragColor u32 0x0, u32 0x3, %4\n\
         \x20   %6: u32 = GetRegister r4\n\
         \x20   %7: f32 = BitCastF32U32 %6\n\
         \x20   %8 = SetFragColor u32 0x2, u32 0x0, %7\n\
         \x20   return\n"
    );
    // Every component of every target reads r0 to r31.
    let all = translate(pixel([0xf; COLOR_TARGETS]), &[exit()]).expect("translates");
    let last = all.to_string();
    assert!(last.contains("GetRegister r31\n"), "{last}");
    assert!(
        last.ends_with("%95 = SetFragColor u32 0x7, u32 0x3, %94\n    return\n"),
        "{last}"
    );
}

#[test]
fn a_vertex_exit_writes_nothing() {
    let ir = translate(VERTEX, &[exit()]).expect("translates");
    assert_eq!(ir.to_string(), "block0:\n    return\n");
}

#[test]
fn s2r_lane_id() {
    let s2r = |sr| {
        Instr::new(Op::S2R(OpS2R {
            dst: S2RDst::Reg(r(5)),
            sr,
        }))
    };
    let ir = translate(
        pixel([0; COLOR_TARGETS]),
        &[s2r(SpecialReg::LANE_ID), exit()],
    )
    .expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n    %0: u32 = LaneId\n    %1 = SetRegister r5, %0\n    return\n"
    );
    let clock = s2r(SpecialReg::CLOCK_LO);
    assert_eq!(
        one(clock),
        unsupported(clock, Unsupported::SpecialRegister(SpecialReg::CLOCK_LO))
    );
}

/// `S2R` to `RZ` reads nothing, so it adds no lane id (and no subgroup requirement); the special
/// register is still checked.
#[test]
fn s2r_to_rz_reads_nothing() {
    let s2r = |sr| {
        Instr::new(Op::S2R(OpS2R {
            dst: S2RDst::Reg(Gpr::RZ),
            sr,
        }))
    };
    for stage in [VERTEX, pixel([0; COLOR_TARGETS])] {
        let ir = translate(stage, &[s2r(SpecialReg::LANE_ID), exit()]).expect("translates");
        assert_eq!(ir.to_string(), "block0:\n    return\n");
    }
    let clock = s2r(SpecialReg::CLOCK_LO);
    assert_eq!(
        one(clock),
        unsupported(clock, Unsupported::SpecialRegister(SpecialReg::CLOCK_LO))
    );
}

/// `S2UR`, `S2R` to a uniform register, writes the uniform register class, as PR #4588 writes
/// its uniform register through `Visitor::S2R` (`case 0x919: case 0x9c3:`, `WriteDst`).
#[test]
fn s2ur_writes_the_uniform_register() {
    let ur1 = UniformReg::new(1).expect("a uniform register");
    let s2ur = Instr::new(Op::S2R(OpS2R {
        dst: S2RDst::Uniform(ur1),
        sr: SpecialReg::LANE_ID,
    }));
    let ir = one(s2ur).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n    %0: u32 = LaneId\n    %1 = SetUniformRegister ur1, %0\n    return\n"
    );
}

/// `instr` under `guard`.
fn under(guard: PredSrc, instr: Instr) -> Instr {
    Instr { guard, ..instr }
}

fn p2(negated: bool) -> PredSrc {
    PredSrc {
        reg: Pred::new(2).expect("p2"),
        negated,
    }
}

/// A guard on a pure instruction, plain or negated, is kept as a select of the new value
/// against the register's old one (PR #4588's `guard_select`, `Visitor::X`), and the next,
/// unguarded instruction writes plainly. Run on the interpreter, the register takes the new
/// value exactly when the guard holds.
#[test]
fn guarded_pure_instructions_become_selects() {
    let guarded = |negated| under(p2(negated), mov(r(0), MovSrc::Imm(7)));
    let ir = translate(VERTEX, &[guarded(false), mov(r(1), MovSrc::Imm(9)), exit()])
        .expect("translates");
    assert_eq!(ir.verify(), Ok(()));
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u1 = GetPred p2\n\
         \x20   %1: u32 = GetRegister r0\n\
         \x20   %2: u32 = SelectU32 %0, u32 0x7, %1\n\
         \x20   %3 = SetRegister r0, %2\n\
         \x20   %4 = SetRegister r1, u32 0x9\n\
         \x20   return\n"
    );
    let negated = translate(VERTEX, &[guarded(true), exit()]).expect("translates");
    assert_eq!(
        negated.to_string(),
        "block0:\n\
         \x20   %0: u1 = GetPred p2\n\
         \x20   %1: u1 = LogicalNot %0\n\
         \x20   %2: u32 = GetRegister r0\n\
         \x20   %3: u32 = SelectU32 %1, u32 0x7, %2\n\
         \x20   %4 = SetRegister r0, %3\n\
         \x20   return\n"
    );
    for negated in [false, true] {
        let ir = translate(VERTEX, &[guarded(negated), exit()]).expect("translates");
        for p in [false, true] {
            let mut machine = Machine::default();
            machine.regs[0] = 3;
            machine.preds[2] = p;
            machine.run(&ir).expect("runs");
            let runs = p != negated;
            assert_eq!(
                machine.regs[0],
                if runs { 7 } else { 3 },
                "p2 {p}, negated {negated}"
            );
        }
    }
    // Each register of a multi-word write selects on its own old value.
    let load = under(p2(false), Instr::new(Op::ALd(ald(r(4), 0x80, 2))));
    let ir = one(load).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u1 = GetPred p2\n\
         \x20   %1: f32 = GetAttribute a0x80, u32 0x0\n\
         \x20   %2: u32 = BitCastU32F32 %1\n\
         \x20   %3: u32 = GetRegister r4\n\
         \x20   %4: u32 = SelectU32 %0, %2, %3\n\
         \x20   %5 = SetRegister r4, %4\n\
         \x20   %6: f32 = GetAttribute a0x84, u32 0x0\n\
         \x20   %7: u32 = BitCastU32F32 %6\n\
         \x20   %8: u32 = GetRegister r5\n\
         \x20   %9: u32 = SelectU32 %0, %7, %8\n\
         \x20   %10 = SetRegister r5, %9\n\
         \x20   return\n"
    );
    for p in [false, true] {
        let mut machine = Machine::default();
        machine.regs[4] = 1;
        machine.regs[5] = 2;
        machine.preds[2] = p;
        machine
            .attributes
            .insert((crate::ir::Attribute(0x20), 0), 10.0_f32.to_bits());
        machine
            .attributes
            .insert((crate::ir::Attribute(0x21), 0), 20.0_f32.to_bits());
        machine.run(&ir).expect("runs");
        let wanted = if p {
            [10.0_f32.to_bits(), 20.0_f32.to_bits()]
        } else {
            [1, 2]
        };
        assert_eq!([machine.regs[4], machine.regs[5]], wanted, "p2 {p}");
    }
}

/// A guard on an instruction that is not pure (`AST` here) makes it a conditional block: the
/// predicate is read, the instruction runs in a block of its own, and translation goes on in
/// the block after it. Run on the interpreter, the attribute is written exactly when the guard
/// holds.
#[test]
fn guarded_side_effects_become_conditional_blocks() {
    let store = |negated| under(p2(negated), Instr::new(Op::ASt(ast(r(0), 0x80, 1))));
    let ir =
        translate(VERTEX, &[store(false), mov(r(1), MovSrc::Imm(9)), exit()]).expect("translates");
    assert_eq!(ir.verify(), Ok(()));
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u1 = GetPred p2\n\
         \x20   branch %0, block1, block2\n\
         block1:\n\
         \x20   %1: u32 = GetRegister r0\n\
         \x20   %2: f32 = BitCastF32U32 %1\n\
         \x20   %3 = SetAttribute a0x80, %2, u32 0x0\n\
         \x20   jump block2\n\
         block2:\n\
         \x20   %4 = SetRegister r1, u32 0x9\n\
         \x20   return\n"
    );
    for negated in [false, true] {
        let ir = translate(VERTEX, &[store(negated), exit()]).expect("translates");
        for p in [false, true] {
            let mut machine = Machine::default();
            machine.preds[2] = p;
            machine.regs[0] = 1.5_f32.to_bits();
            machine.run(&ir).expect("runs");
            let written = machine
                .outputs
                .get(&(crate::ir::Attribute(0x20), 0))
                .copied();
            assert_eq!(
                written,
                (p != negated).then_some(1.5_f32.to_bits()),
                "p2 {p}, negated {negated}"
            );
        }
    }
}

/// One instruction of each operation whose guard [`translate`] reads itself (every one it takes
/// but the float group's, which [`float`] guards), each in a form it accepts in a vertex
/// program, with the class PR #4588's `classify.h` gives its opcode written out: `(instr, pure,
/// uniform)`.
fn one_of_each_translated_op() -> Vec<(Instr, bool, bool)> {
    let ipa = OpIpa {
        dst: r(0),
        addr: 0x70,
        freq: crate::decode::InterpFreq::Pass,
        loc: crate::decode::InterpLoc::Default,
        offset: Gpr::RZ,
        pred: Pred::PT,
    };
    let s2r = |dst| OpS2R {
        dst,
        sr: SpecialReg::LANE_ID,
    };
    let ur1 = UniformReg::new(1).expect("ur1");
    vec![
        // `classify.h`: `MOV` by its base, in every form.
        (mov(r(0), MovSrc::Imm(7)), true, false),
        // `case 0x321: // ALD`, `case 0x326: // IPA`, `case 0x805: // CS2R`, `case 0x919:
        // // S2R`.
        (Instr::new(Op::ALd(ald(r(0), 0x80, 1))), true, false),
        (Instr::new(Op::Ipa(ipa)), true, false),
        (
            Instr::new(Op::CS2R(OpCS2R {
                dst: r(0),
                sr: SpecialReg::SRZ,
                wide: false,
            })),
            true,
            false,
        ),
        (Instr::new(Op::S2R(s2r(S2RDst::Reg(r(0))))), true, false),
        // `case 0x9c3: // S2UR`, with the uniform datapath instructions.
        (Instr::new(Op::S2R(s2r(S2RDst::Uniform(ur1)))), true, true),
        // No row: `AST` (`0x322`), `AL2P` (`0x920`) and `PIXLD` (`0x925`) are plain.
        (Instr::new(Op::ASt(ast(r(0), 0x80, 1))), false, false),
        (
            Instr::new(Op::AL2P(OpAL2P {
                dst: r(0),
                offset: r(1),
                addr: 0x80,
                comps: 1,
                output: false,
            })),
            false,
            false,
        ),
        (
            Instr::new(Op::PixLd(OpPixLd {
                dst: r(0),
                val: PixVal::MyIndex,
                pred: Pred::PT,
            })),
            false,
            false,
        ),
    ]
}

/// The 12-bit opcode of `instr` as the test assembler encodes it.
fn encoded_opcode(instr: Instr) -> u16 {
    let word = assemble(&instr).expect("encodable");
    u16::try_from(word.field(OPCODE)).expect("12 bits")
}

/// The translator classifies a guard by flow analysis's table, looked up by the opcode the
/// decoder reads the operation from: for each operation it takes, its `opcode` is the one the
/// test assembler encodes (`MOV`: the same class in every operand form), the class is
/// `classify.h`'s, and the guard becomes a select exactly for the pure ones. The float group,
/// `FFMA` included, takes its own guard, so `opcode` does not name it.
#[test]
fn guards_classify_by_the_flow_table() {
    for (instr, pure, uniform) in one_of_each_translated_op() {
        let encoded = encoded_opcode(instr);
        let info = classify(opcode(instr.op).expect("named"));
        assert_eq!(info, classify(encoded), "{instr}");
        if !matches!(instr.op, Op::Mov(_)) {
            assert_eq!(opcode(instr.op), Some(encoded), "{instr}");
        }
        assert_eq!((info.pure, info.uniform), (pure, uniform), "{instr}");
        let wanted = if pure {
            Guarding::Select { uniform }
        } else {
            Guarding::Block { uniform }
        };
        assert_eq!(guarding(under(p2(false), instr)), wanted, "{instr}");
        assert_eq!(guarding(instr), Guarding::Always, "{instr}");
    }
    // Every source form of `MOV` classifies as its register form does.
    let cbuf = CBufRef {
        buf: crate::decode::CBuf::Binding(1),
        offset: 0x10,
    };
    let ur1 = UniformReg::new(1).expect("ur1");
    for src in [
        MovSrc::Reg(r(1)),
        MovSrc::Imm(7),
        MovSrc::CBuf(cbuf),
        MovSrc::UniformReg(ur1),
    ] {
        let instr = mov(r(0), src);
        assert_eq!(
            classify(encoded_opcode(instr)),
            classify(opcode(instr.op).expect("named")),
            "{instr}"
        );
    }
    // `FFMA`, in every source form, is the float group's: `opcode` does not name it, since
    // `translate` hands it to `float` before it reads a guard.
    let ffma = |srcs| {
        Instr::new(Op::FFma(OpFFma {
            dst: r(0),
            srcs,
            saturate: false,
            rnd: RoundMode::NearestEven,
            ftz: false,
            dnz: false,
        }))
    };
    let reg = AluSrc::reg(r(1));
    for other in [reg, AluSrc::Imm(0x3f80_0000)] {
        let instr = ffma([reg, other, reg]);
        assert!(float::in_group(&instr.op), "{instr}");
        assert_eq!(opcode(instr.op), None, "{instr}");
    }
    // The guard rules before the class: `@!PT` never runs, whatever the operation.
    let never = PredSrc {
        reg: Pred::PT,
        negated: true,
    };
    assert_eq!(
        guarding(under(never, mov(r(0), MovSrc::Imm(7)))),
        Guarding::Never
    );
    // An operation `opcode` does not name gets a conditional block.
    let nop = Instr::new(Op::Nop(crate::decode::OpNop {}));
    assert_eq!(opcode(nop.op), None);
    assert_eq!(
        guarding(under(p2(false), nop)),
        Guarding::Block { uniform: false }
    );
}

/// The guard of a pure uniform datapath instruction reads a uniform predicate,
/// `GetUniformPred` (PR #4588's `classify.h` and `UPred`), and every other guard a predicate,
/// for each operation the translator takes; of these, only `S2UR` is uniform. `S2UR`'s select
/// reads the uniform register's old value.
#[test]
fn a_guard_on_each_pure_uniform_op_reads_a_uniform_predicate() {
    let guard = PredSrc {
        reg: Pred::new(3).expect("p3"),
        negated: true,
    };
    let mut uniform_ops = Vec::new();
    for (instr, _, uniform) in one_of_each_translated_op() {
        let text = one(under(guard, instr)).expect("translates").to_string();
        let first = text.lines().nth(1).unwrap_or_default().to_owned();
        if uniform {
            uniform_ops.push(instr.op.mnemonic());
            assert_eq!(first, "    %0: u1 = GetUniformPred up3", "{instr}");
        } else {
            assert_eq!(first, "    %0: u1 = GetPred p3", "{instr}");
        }
    }
    assert_eq!(uniform_ops, ["s2r"]);
    let ur1 = UniformReg::new(1).expect("ur1");
    let s2ur = under(
        guard,
        Instr::new(Op::S2R(OpS2R {
            dst: S2RDst::Uniform(ur1),
            sr: SpecialReg::LANE_ID,
        })),
    );
    let ir = one(s2ur).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u1 = GetUniformPred up3\n\
         \x20   %1: u1 = LogicalNot %0\n\
         \x20   %2: u32 = LaneId\n\
         \x20   %3: u32 = GetUniformRegister ur1\n\
         \x20   %4: u32 = SelectU32 %1, %2, %3\n\
         \x20   %5 = SetUniformRegister ur1, %4\n\
         \x20   return\n"
    );
    for up in [false, true] {
        let mut machine = Machine::default();
        machine.uniform_preds[3] = up;
        machine.uniform_regs[1] = 2;
        machine.system.lane_id = 6;
        machine.run(&ir).expect("runs");
        assert_eq!(machine.uniform_regs[1], if up { 2 } else { 6 }, "up3 {up}");
    }
}

/// A guarded `EXIT` is a conditional exit, which the straight-line translator does not build.
#[test]
fn a_guarded_exit_is_rejected() {
    let guarded = Instr {
        guard: PredSrc {
            reg: Pred::new(0).expect("p0"),
            negated: false,
        },
        ..exit()
    };
    assert_eq!(
        translate(VERTEX, &[guarded]),
        Err(TranslateError::Unsupported {
            index: 0,
            instr: guarded,
            what: Unsupported::Guard
        })
    );
    assert_eq!(
        translate(VERTEX, &[guarded]).map_err(|error| error.to_string()),
        Err("instruction 0 (@p0 exit;): a guarded exit is not supported".into())
    );
}

/// `@!PT` never runs: it translates to nothing, as the PR skips it (`Visitor::Run`,
/// `guard.Never()`), and an `EXIT` under it does not end the program.
#[test]
fn never_guarded_instructions_are_skipped() {
    let never = PredSrc {
        reg: Pred::PT,
        negated: true,
    };
    let skipped = |instr: Instr| Instr {
        guard: never,
        ..instr
    };
    let program = [
        skipped(mov(r(0), MovSrc::Imm(1))),
        skipped(Instr::new(Op::ASt(ast(r(0), 0x80, 1)))),
        skipped(Instr::new(Op::Nop(crate::decode::OpNop {}))),
        skipped(exit()),
        exit(),
    ];
    let ir = translate(VERTEX, &program).expect("translates");
    assert_eq!(ir.to_string(), "block0:\n    return\n");
    // The same instruction under `PT` translates to something.
    let ir = one(mov(r(0), MovSrc::Imm(1))).expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n    %0 = SetRegister r0, u32 0x1\n    return\n"
    );
}

#[test]
fn mov_lane_mask_must_be_full() {
    let partial = Instr::new(Op::Mov(OpMov {
        dst: r(0),
        src: MovSrc::Imm(1),
        quad_lanes: 0xe,
    }));
    assert_eq!(one(partial), unsupported(partial, Unsupported::MovLaneMask));
    assert!(one(mov(r(0), MovSrc::Imm(1))).is_ok());
}

/// The decoder reads uniform-register and constant-buffer sources of `MOV`; the translator
/// rejects them and keeps translating register and immediate sources. (`FFMA` reads them since
/// slice 12b: `float::tests::sources_of_every_kind`.)
#[test]
fn uniform_and_constant_buffer_sources_are_rejected() {
    use crate::decode::{CBuf, CBufRef, UniformReg};
    let cbuf = CBufRef {
        buf: CBuf::Binding(1),
        offset: 0x10,
    };
    let ureg = UniformReg::new(4).expect("ur4");
    for src in [MovSrc::CBuf(cbuf), MovSrc::UniformReg(ureg)] {
        let instr = mov(r(0), src);
        assert_eq!(one(instr), unsupported(instr, Unsupported::Source));
    }
    let instr = mov(r(0), MovSrc::UniformReg(ureg));
    assert_eq!(
        one(instr).map_err(|error| error.to_string()),
        Err(
            "instruction 0 (mov r0, ur4;): a uniform-register or constant-buffer source is not \
             supported"
                .into()
        )
    );
    assert!(one(mov(r(0), MovSrc::Reg(r(1)))).is_ok());
}

/// Every operation but the six of slice 1 is rejected by its mnemonic.
#[test]
fn other_instructions_are_rejected_by_mnemonic() {
    use crate::decode::{
        ChannelMask, MemEvictionPriority, OpIAbs, OpNop, OpTex, Pred, TexDim, TexLodMode, TexRef,
    };
    let iabs = Op::IAbs(OpIAbs {
        dst: r(0),
        src: AluSrc::reg(r(1)),
    });
    let tex = Op::Tex(OpTex {
        dsts: [r(0), Gpr::RZ],
        fault: Pred::PT,
        tex: TexRef::Bindless,
        srcs: [r(1), r(2)],
        dim: TexDim::D2,
        lod_mode: TexLodMode::Auto,
        ndv: false,
        z_cmpr: false,
        offset: false,
        f16: false,
        eviction: MemEvictionPriority::Normal,
        nodep: false,
        channel_mask: ChannelMask::RGBA,
        scalar: false,
    });
    for op in [iabs, Op::Nop(OpNop {}), tex] {
        let instr = Instr::new(op);
        assert_eq!(
            one(instr),
            unsupported(instr, Unsupported::Instruction(op.mnemonic()))
        );
    }
    let instr = Instr::new(Op::Nop(OpNop {}));
    assert_eq!(
        one(instr).map_err(|error| error.to_string()),
        Err("instruction 0 (nop;): nop is not supported".into())
    );
}

#[test]
fn ffma_multiply_zero_mode_is_rejected() {
    let op = OpFFma {
        dst: r(0),
        srcs: [AluSrc::reg(r(1)); 3],
        saturate: false,
        rnd: RoundMode::NearestEven,
        ftz: false,
        dnz: true,
    };
    let instr = Instr::new(Op::FFma(op));
    assert_eq!(
        one(instr),
        unsupported(
            instr,
            Unsupported::Float(float::FloatUnsupported::FfmaMultiplyZero)
        )
    );
    assert!(one(Instr::new(Op::FFma(OpFFma { dnz: false, ..op }))).is_ok());
}

/// `FFMA`'s controls follow PR #4588 (`ops_alu.cpp`, `FFMA`): no contraction, each rounding
/// field through `RoundingOf`, and `FTZ` or no flushing.
#[test]
fn ffma_controls_follow_the_reference() {
    use crate::ir::{Flags, FmzMode, FpControl, FpRounding, Inst, Opcode};
    let rounding = [
        (RoundMode::NearestEven, FpRounding::Rn),
        (RoundMode::NegInf, FpRounding::Rm),
        (RoundMode::PosInf, FpRounding::Rp),
        (RoundMode::Zero, FpRounding::Rz),
    ];
    for (rnd, wanted) in rounding {
        for (ftz, fmz_mode) in [(false, FmzMode::None), (true, FmzMode::Ftz)] {
            let op = OpFFma {
                dst: r(0),
                srcs: [AluSrc::reg(r(1)); 3],
                saturate: false,
                rnd,
                ftz,
                dnz: false,
            };
            let ir = one(Instr::new(Op::FFma(op))).expect("translates");
            let flags: Vec<Flags> = ir
                .listed()
                .filter_map(|(_, inst)| match inst {
                    Inst::Op(op) if op.opcode == Opcode::FPFma32 => Some(op.flags),
                    _ => None,
                })
                .collect();
            assert_eq!(
                flags,
                [Flags::Fp(FpControl {
                    no_contraction: true,
                    rounding: wanted,
                    fmz_mode,
                })],
                "{rnd:?} {ftz}"
            );
        }
    }
}

#[test]
fn attribute_access_checks() {
    let cases = [
        (
            OpALd {
                offset: r(1),
                ..ald(r(0), 0x80, 1)
            },
            Unsupported::IndexedAttribute,
        ),
        (
            OpALd {
                patch: true,
                ..ald(r(0), 0x80, 1)
            },
            Unsupported::PatchOrPhysical,
        ),
        (
            OpALd {
                phys: true,
                ..ald(r(0), 0x80, 1)
            },
            Unsupported::PatchOrPhysical,
        ),
        (
            OpALd {
                output: true,
                ..ald(r(0), 0x80, 1)
            },
            Unsupported::OutputRead,
        ),
        // Both the offset register and the patch bit: the offset is reported, first in the
        // PR's order.
        (
            OpALd {
                offset: r(1),
                patch: true,
                ..ald(r(0), 0x80, 1)
            },
            Unsupported::IndexedAttribute,
        ),
    ];
    for (op, what) in cases {
        let instr = Instr::new(Op::ALd(op));
        assert_eq!(one(instr), unsupported(instr, what));
    }
    let ast_cases = [
        (
            OpASt {
                offset: r(1),
                ..ast(r(0), 0x80, 1)
            },
            Unsupported::IndexedAttribute,
        ),
        (
            OpASt {
                patch: true,
                ..ast(r(0), 0x80, 1)
            },
            Unsupported::PatchOrPhysical,
        ),
    ];
    for (op, what) in ast_cases {
        let instr = Instr::new(Op::ASt(op));
        assert_eq!(one(instr), unsupported(instr, what));
    }
}

/// Instructions with two faults at once report the one the PR checks first (offset register,
/// patch or physical, output, address, registers). A vertex register is no fault: it is read
/// (the attribute translator's `tests::ald_and_ast_pass_the_vertex_register`).
#[test]
fn attribute_checks_report_in_the_prs_order() {
    for (op, what) in [
        // Offset register and patch bit: the offset is reported, first in the PR's order.
        (
            OpASt {
                offset: r(1),
                patch: true,
                ..ast(r(0), 0x80, 1)
            },
            Unsupported::IndexedAttribute,
        ),
        // Bad address and registers past RZ: the address is reported first.
        (
            ast(r(254), 0x81, 2),
            Unsupported::AttributeAddress {
                addr: 0x81,
                comps: 2,
            },
        ),
        // Bad registers with a vertex register: the registers are reported.
        (
            OpASt {
                vtx: r(2),
                ..ast(r(254), 0x80, 2)
            },
            Unsupported::AttributeRegisters {
                first: r(254),
                comps: 2,
            },
        ),
    ] {
        let instr = Instr::new(Op::ASt(op));
        assert_eq!(one(instr), unsupported(instr, what));
    }
    let ald_order = [
        (
            ald(r(254), 0x81, 2),
            Unsupported::AttributeAddress {
                addr: 0x81,
                comps: 2,
            },
        ),
        (
            OpALd {
                output: true,
                ..ald(r(0), 0x81, 1)
            },
            Unsupported::OutputRead,
        ),
        (
            OpALd {
                vtx: r(2),
                ..ald(r(254), 0x80, 2)
            },
            Unsupported::AttributeRegisters {
                first: r(254),
                comps: 2,
            },
        ),
    ];
    for (op, what) in ald_order {
        let instr = Instr::new(Op::ALd(op));
        assert_eq!(one(instr), unsupported(instr, what));
    }
}

/// `addr % 4 == 0 && addr + comps * 4 <= 0x400`, at the edge on both sides.
#[test]
fn attribute_addresses_end_at_the_attribute_space() {
    let last_word = u16::try_from(ATTRIBUTE_SPACE_END - 4).expect("fits");
    let last_quad = u16::try_from(ATTRIBUTE_SPACE_END - 16).expect("fits");
    for (addr, comps, ok) in [
        (last_word, 1, true),
        (last_word, 2, false),
        (last_quad, 4, true),
        (last_quad + 4, 4, false),
        (0x81, 1, false),
        (0x82, 1, false),
        (0x3ff, 1, false),
    ] {
        for instr in [
            Instr::new(Op::ALd(ald(r(0), addr, comps))),
            Instr::new(Op::ASt(ast(r(0), addr, comps))),
        ] {
            let result = one(instr);
            if ok {
                assert!(result.is_ok(), "{instr}: {result:?}");
            } else {
                assert_eq!(
                    result,
                    unsupported(instr, Unsupported::AttributeAddress { addr, comps })
                );
            }
        }
    }
}

/// `dest + comps <= 255`, at the edge on both sides; `AST` from `RZ` stores zeros.
#[test]
fn attribute_registers_stop_before_rz() {
    let limit = u8::try_from(REGISTER_LIMIT).expect("fits");
    for comps in 1_u8..=4 {
        let fits = r(limit - comps);
        let past = r(limit - comps + 1);
        assert!(one(Instr::new(Op::ALd(ald(fits, 0x80, comps)))).is_ok());
        assert!(one(Instr::new(Op::ASt(ast(fits, 0x80, comps)))).is_ok());
        let mut past_cases = vec![Instr::new(Op::ALd(ald(past, 0x80, comps)))];
        // `AST` from `RZ` (the one-word case's `past`) is the zero store, tested below.
        if !past.is_zero() {
            past_cases.push(Instr::new(Op::ASt(ast(past, 0x80, comps))));
        }
        for instr in past_cases {
            let first = past;
            assert_eq!(
                one(instr),
                unsupported(instr, Unsupported::AttributeRegisters { first, comps })
            );
        }
    }
    // ALD into RZ is rejected (RZ + 1 > 255); AST from RZ is allowed.
    let into_rz = Instr::new(Op::ALd(ald(Gpr::RZ, 0x80, 1)));
    assert_eq!(
        one(into_rz),
        unsupported(
            into_rz,
            Unsupported::AttributeRegisters {
                first: Gpr::RZ,
                comps: 1
            }
        )
    );
    assert!(one(Instr::new(Op::ASt(ast(Gpr::RZ, 0x80, 4)))).is_ok());
}

#[test]
fn exit_modes_and_predicates_are_rejected() {
    let cases = [
        (
            OpExit {
                keep_ref_count: true,
                no_at_exit: false,
                pred: PredSrc::TRUE,
            },
            Unsupported::ExitMode,
        ),
        (
            OpExit {
                keep_ref_count: false,
                no_at_exit: true,
                pred: PredSrc::TRUE,
            },
            Unsupported::ExitMode,
        ),
        (
            OpExit {
                keep_ref_count: false,
                no_at_exit: false,
                pred: PredSrc {
                    reg: Pred::new(3).expect("p3"),
                    negated: false,
                },
            },
            Unsupported::ConditionalExit,
        ),
        (
            OpExit {
                keep_ref_count: false,
                no_at_exit: false,
                pred: PredSrc {
                    reg: Pred::PT,
                    negated: true,
                },
            },
            Unsupported::ConditionalExit,
        ),
    ];
    for (op, what) in cases {
        let instr = Instr::new(Op::Exit(op));
        assert_eq!(translate(VERTEX, &[instr]), unsupported(instr, what));
    }
}

#[test]
fn the_program_ends_at_its_only_exit() {
    assert_eq!(translate(VERTEX, &[]), Err(TranslateError::NoExit));
    assert_eq!(
        translate(VERTEX, &[mov(r(0), MovSrc::Imm(1))]),
        Err(TranslateError::NoExit)
    );
    assert_eq!(
        translate(VERTEX, &[exit(), mov(r(0), MovSrc::Imm(1)), exit()]),
        Err(TranslateError::AfterExit { index: 1, exit: 0 })
    );
    assert!(translate(VERTEX, &[mov(r(0), MovSrc::Imm(1)), exit()]).is_ok());
}

#[test]
fn only_vertex_and_pixel_stages_translate() {
    for stage in [
        GuestStage::VertexCullBeforeFetch,
        GuestStage::TessellationInit,
        GuestStage::Tessellation,
        GuestStage::Geometry,
        GuestStage::Compute,
    ] {
        let stage = Stage { stage, ..VERTEX };
        assert_eq!(
            translate(stage, &[exit()]),
            Err(TranslateError::Stage(Unsupported::Stage(stage.stage)))
        );
    }
}

#[test]
fn rejections_read_as_words() {
    let instr = Instr::new(Op::ALd(ald(r(0), 0x81, 2)));
    assert_eq!(
        one(instr).map_err(|error| error.to_string()),
        Err(
            "instruction 0 (ald.64 r0, a[0x81];): attribute address 0x81 for a 2-word access \
             is not supported"
                .into()
        )
    );
    assert_eq!(
        TranslateError::AfterExit { index: 3, exit: 2 }.to_string(),
        "instruction 3 follows the exit at 2"
    );
    assert_eq!(
        TranslateError::Stage(Unsupported::Stage(GuestStage::Geometry)).to_string(),
        "geometry programs are not supported yet"
    );
}

/// Every guest stage has a plain-words name.
#[test]
fn guest_stages_read_as_words() {
    let names = [
        (
            GuestStage::VertexCullBeforeFetch,
            "vertex (cull before fetch)",
        ),
        (GuestStage::Vertex, "vertex"),
        (GuestStage::TessellationInit, "tessellation control"),
        (GuestStage::Tessellation, "tessellation evaluation"),
        (GuestStage::Geometry, "geometry"),
        (GuestStage::Pixel, "pixel"),
        (GuestStage::Compute, "compute"),
    ];
    for (stage, name) in names {
        assert_eq!(stage.to_string(), name);
        assert_eq!(
            Unsupported::Stage(stage).to_string(),
            format!("{name} programs are not supported yet")
        );
    }
}

proptest! {
    #![proptest_config(config())]

    /// Arbitrary instruction sequences never panic the translator, and what it accepts
    /// verifies.
    #[test]
    fn arbitrary_programs_never_panic(
        program in proptest::collection::vec(any_instr(), 0..12),
        pixel_stage in any::<bool>(),
        targets in any::<[u8; COLOR_TARGETS]>(),
    ) {
        let stage = if pixel_stage { pixel(targets) } else { VERTEX };
        if let Ok(ir) = translate(stage, &program) {
            prop_assert_eq!(ir.verify(), Ok(()));
        }
    }

    /// A generated operation that `opcode` names classifies as the opcode the test assembler
    /// encodes for it (the fixed list is `guards_classify_by_the_flow_table`).
    #[test]
    fn named_opcodes_classify_as_encoded(instr in any_instr()) {
        if let Some(named) = opcode(instr.op) {
            prop_assert_eq!(classify(named), classify(encoded_opcode(instr)));
        }
    }
}
