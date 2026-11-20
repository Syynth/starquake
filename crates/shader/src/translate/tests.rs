//! Translator tests: golden IR of straight-line programs through the translation driver, `EXIT`,
//! the shared guard helper, each rejection with its exact error, and the limits at their edges.
//! Whole programs with branches and guards are the driver's (`driver::tests`).

use proptest::prelude::*;

use super::driver::translate_instrs;
use super::*;
use crate::asm::assemble;
use crate::decode::{
    AluSrc, CBuf, CBufRef, MovSrc, Op, OpALd, OpASt, OpExit, OpFFma, OpMov, OpS2R, Pred, RoundMode,
    S2RDst,
};
use crate::flow::BranchTables;
use crate::interp::Machine;
use crate::ir::Function;
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

/// `program` through the translation driver, with no header.
fn translate(stage: Stage, program: &[Instr]) -> Result<Function, TranslateError> {
    translate_instrs(stage, None, program)
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

/// `MOV` is the integer group's (slice 12a), slice 1's restricted `MOV` path being retired: a
/// uniform-register or constant-buffer source, which slice 1 refused, translates, and a partial
/// lane mask is the integer group's refusal.
#[test]
fn mov_is_the_integer_groups() {
    let ur4 = UniformReg::new(4).expect("ur4");
    let cbuf = CBufRef {
        buf: CBuf::Binding(1),
        offset: 0x10,
    };
    let ir = translate(
        VERTEX,
        &[
            mov(r(0), MovSrc::UniformReg(ur4)),
            mov(r(1), MovSrc::CBuf(cbuf)),
            exit(),
        ],
    )
    .expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n\
         \x20   %0: u32 = GetUniformRegister ur4\n\
         \x20   %1 = SetRegister r0, %0\n\
         \x20   %2: u32 = GetCbufU32 u32 0x1, u32 0x10\n\
         \x20   %3 = SetRegister r1, %2\n\
         \x20   return\n"
    );
    let mut machine = Machine::default();
    machine.uniform_regs[4] = 0x55;
    machine.cbufs.insert(1, (0_u8..0x20).collect());
    machine.run(&ir).expect("runs");
    assert_eq!(machine.regs[0], 0x55);
    assert_eq!(
        machine.regs[1],
        u32::from_le_bytes([0x10, 0x11, 0x12, 0x13])
    );
    let partial = Instr::new(Op::Mov(OpMov {
        dst: r(0),
        src: MovSrc::Imm(1),
        quad_lanes: 0xe,
    }));
    assert_eq!(
        one(partial),
        Err(TranslateError::Integer {
            index: 0,
            instr: partial,
            error: integer::IntegerError::Unsupported(integer::IntegerUnsupported::MovLaneMask),
        })
    );
    assert_eq!(
        one(partial).map_err(|error| error.to_string()),
        Err(
            "instruction 0 (mov r0, 0x1, 0xe;): a mov with a partial lane mask is not supported"
                .into()
        )
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

/// `EXIT`'s mode bits are refused by flow analysis first ([`FlowError::SpecialExit`]); the
/// `EXIT` translator's own check, kept for the PR's order, refuses them too when it is reached
/// directly. An `EXIT` with a predicate of its own is no longer refused: flow analysis gives it a
/// block whose test reads the predicate (`driver::tests::a_conditional_exit_writes_its_outputs_only_where_it_exits`).
#[test]
fn exit_mode_bits_are_refused() {
    let modes = [
        OpExit {
            keep_ref_count: true,
            no_at_exit: false,
            pred: PredSrc::TRUE,
        },
        OpExit {
            keep_ref_count: false,
            no_at_exit: true,
            pred: PredSrc::TRUE,
        },
    ];
    for op in modes {
        let instr = Instr::new(Op::Exit(op));
        assert_eq!(
            translate(VERTEX, &[instr]),
            Err(TranslateError::Flow(FlowError::SpecialExit { index: 0 }))
        );
        let mut function = Function::new();
        let entry = function.entry();
        let mut t = Translator {
            b: Builder::new(&mut function, entry),
            stage: VERTEX,
            header: None,
        };
        assert!(matches!(
            t.exit(op),
            Err(Failure::Unsupported(Unsupported::ExitMode))
        ));
    }
    let conditional = Instr::new(Op::Exit(OpExit {
        pred: PredSrc {
            reg: Pred::new(3).expect("p3"),
            negated: false,
        },
        ..modes[0]
    }));
    let conditional = Instr {
        op: match conditional.op {
            Op::Exit(op) => Op::Exit(OpExit {
                keep_ref_count: false,
                ..op
            }),
            other => other,
        },
        ..conditional
    };
    assert!(translate(VERTEX, &[conditional, exit()]).is_ok());
}

/// A program must exit: an empty one and one that runs off its end are flow analysis's
/// refusals; instructions after the program's only `EXIT` are not reached, so they are neither
/// decoded nor translated.
#[test]
fn the_program_runs_to_an_exit() {
    assert_eq!(
        translate(VERTEX, &[]),
        Err(TranslateError::Flow(FlowError::ProgramSize { len: 0 }))
    );
    assert_eq!(
        translate(VERTEX, &[mov(r(0), MovSrc::Imm(1))]),
        Err(TranslateError::Flow(FlowError::RunsPastEnd { last: 0 }))
    );
    let mut code = vec![assemble(&exit()).expect("encodable")];
    // A word that does not decode, after the exit: never reached.
    code.push(crate::decode::Instruction::default());
    let blocks = translate_flow(VERTEX, None, &code, &BranchTables::new(0)).expect("translates");
    assert_eq!(blocks.function.to_string(), "block0:\n    return\n");
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
        TranslateError::Stage(Unsupported::Stage(GuestStage::Geometry)).to_string(),
        "geometry programs are not supported yet"
    );
    assert_eq!(
        TranslateError::Decode {
            index: 2,
            error: crate::decode::DecodeError::UnknownOpcode { opcode: 0 }
        }
        .to_string(),
        "instruction 2: unknown opcode 0x000"
    );
    assert_eq!(
        TranslateError::Flow(FlowError::RunsPastEnd { last: 4 }).to_string(),
        "control runs past instruction 4, the last of the program"
    );
    assert_eq!(
        TranslateError::TooManyInstructions.to_string(),
        format!(
            "the program, its calls inlined, has more than {MAX_TRANSLATED_INSTRUCTIONS} \
             instructions to translate"
        )
    );
    assert_eq!(
        Unsupported::MemoryGroup("ld").to_string(),
        "ld, of the memory group, which is not translated yet,"
    );
}

/// The shared guard helper: the split block branches on the condition to the instruction's
/// block (made first) or the block after (made second); the instruction's block jumps to the
/// block after, which takes over the split block's terminator; the builder is left in the
/// instruction's block. In a block flow analysis has terminated, the old terminator moves; in an
/// open block, the open terminator does.
#[test]
fn the_guard_helper_moves_the_terminator_to_the_block_after() {
    for old in [Terminator::Return, Terminator::None] {
        let mut function = Function::new();
        let entry = function.entry();
        let mut b = Builder::new(&mut function, entry);
        b.terminate(old).expect("terminates");
        let cond = b
            .op(
                Opcode::GetPred,
                &[Operand::Imm(Imm::Pred(Pred::new(1).expect("p1")))],
            )
            .expect("reads")
            .into();
        let join = split_for_guard(&mut b, cond).expect("splits");
        let then = b.block();
        assert_eq!((then.index(), join.index()), (1, 2), "{old:?}");
        let block = |id| function.block(id).map(|block| block.terminator);
        assert_eq!(
            block(entry),
            Some(Terminator::Branch {
                cond,
                then,
                otherwise: join
            })
        );
        assert_eq!(block(then), Some(Terminator::Jump(join)));
        assert_eq!(block(join), Some(old));
    }
}

/// `conditional_block`: no split when the block's guard is tested or the guard is `PT`; else
/// the guard's read, the split, the body in the instruction's block and the builder left in the
/// block after.
#[test]
fn conditional_block_splits_only_for_an_untested_guard() {
    let p1 = PredSrc {
        reg: Pred::new(1).expect("p1"),
        negated: false,
    };
    for (guard, tested, splits) in [
        (p1, false, true),
        (p1, true, false),
        (PredSrc::TRUE, false, false),
    ] {
        let mut function = Function::new();
        let entry = function.entry();
        let mut b = Builder::new(&mut function, entry);
        b.terminate(Terminator::Return).expect("terminates");
        let mut body_block = None;
        conditional_block::<BuildError>(&mut b, guard, tested, |b| {
            body_block = Some(b.block());
            b.op(Opcode::Barrier, &[])?;
            Ok(())
        })
        .expect("builds");
        let after = b.block();
        if splits {
            assert_eq!(body_block.map(BlockId::index), Some(1));
            assert_eq!(after.index(), 2);
            assert_eq!(
                function.to_string(),
                "block0:\n\
                 \x20   %0: u1 = GetPred p1\n\
                 \x20   branch %0, block1, block2\n\
                 block1:\n\
                 \x20   %1 = Barrier\n\
                 \x20   jump block2\n\
                 block2:\n\
                 \x20   return\n"
            );
        } else {
            assert_eq!((body_block, after), (Some(entry), entry));
            assert_eq!(
                function.to_string(),
                "block0:\n    %0 = Barrier\n    return\n"
            );
        }
    }
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

    /// Arbitrary instruction sequences, ended by an `EXIT`, never panic the driver, and what it
    /// accepts verifies.
    #[test]
    fn arbitrary_programs_never_panic(
        program in proptest::collection::vec(any_instr(), 0..12),
        pixel_stage in any::<bool>(),
        targets in any::<[u8; COLOR_TARGETS]>(),
    ) {
        let stage = if pixel_stage { pixel(targets) } else { VERTEX };
        let mut program = program;
        program.push(exit());
        if let Ok(ir) = translate(stage, &program) {
            prop_assert_eq!(ir.verify(), Ok(()));
        }
    }
}
