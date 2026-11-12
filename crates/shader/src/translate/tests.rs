//! Translator tests: golden IR, each rejection with its exact error, and the limits at their
//! edges.

use proptest::prelude::*;

use super::*;
use crate::decode::{OpExit, Pred, PredSrc, RoundMode};
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
                FloatSrc::reg(r(0)),
                FloatSrc::Imm(0.5_f32.to_bits()),
                FloatSrc::Reg {
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
         \x20   %0: f32 = get_attribute a0x80\n\
         \x20   %1: u32 = bitcast_u32 %0\n\
         \x20   set_reg r0, %1\n\
         \x20   %3: f32 = get_attribute a0x84\n\
         \x20   %4: u32 = bitcast_u32 %3\n\
         \x20   set_reg r1, %4\n\
         \x20   %6: u32 = get_reg r0\n\
         \x20   %7: f32 = bitcast_f32 %6\n\
         \x20   %8: u32 = imm32 0x3f000000\n\
         \x20   %9: f32 = bitcast_f32 %8\n\
         \x20   %10: u32 = get_reg r1\n\
         \x20   %11: f32 = bitcast_f32 %10\n\
         \x20   %12: f32 = fabs %11\n\
         \x20   %13: f32 = fneg %12\n\
         \x20   %14: f32 = ffma.rz.ftz %7, %9, %13\n\
         \x20   %15: f32 = fsaturate %14\n\
         \x20   %16: u32 = bitcast_u32 %15\n\
         \x20   set_reg r2, %16\n\
         \x20   %18: u32 = imm32 0x0\n\
         \x20   set_reg r3, %18\n\
         \x20   %20: u32 = imm32 0x7\n\
         \x20   %21: u32 = get_reg r2\n\
         \x20   %22: f32 = bitcast_f32 %21\n\
         \x20   set_attribute a0x70, %22\n\
         \x20   %24: u32 = get_reg r3\n\
         \x20   %25: f32 = bitcast_f32 %24\n\
         \x20   set_attribute a0x74, %25\n\
         \x20   %27: u32 = imm32 0x0\n\
         \x20   %28: f32 = bitcast_f32 %27\n\
         \x20   set_attribute a0x78, %28\n"
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
         \x20   %0: u32 = get_reg r1\n\
         \x20   %1: f32 = bitcast_f32 %0\n\
         \x20   set_frag_color 0.1, %1\n\
         \x20   %3: u32 = get_reg r3\n\
         \x20   %4: f32 = bitcast_f32 %3\n\
         \x20   set_frag_color 0.3, %4\n\
         \x20   %6: u32 = get_reg r4\n\
         \x20   %7: f32 = bitcast_f32 %6\n\
         \x20   set_frag_color 2.0, %7\n"
    );
    // Every component of every target reads r0 to r31.
    let all = translate(pixel([0xf; COLOR_TARGETS]), &[exit()]).expect("translates");
    let last = all.to_string();
    assert!(last.contains("get_reg r31\n"), "{last}");
    assert!(last.ends_with("set_frag_color 7.3, %94\n"), "{last}");
}

#[test]
fn a_vertex_exit_writes_nothing() {
    let ir = translate(VERTEX, &[exit()]).expect("translates");
    assert_eq!(ir.to_string(), "block0:\n");
}

#[test]
fn s2r_lane_id() {
    let s2r = |sr| Instr::new(Op::S2R(OpS2R { dst: r(5), sr }));
    let ir = translate(
        pixel([0; COLOR_TARGETS]),
        &[s2r(SpecialReg::LANE_ID), exit()],
    )
    .expect("translates");
    assert_eq!(
        ir.to_string(),
        "block0:\n    %0: u32 = lane_id\n    set_reg r5, %0\n"
    );
    let tid = s2r(SpecialReg::TID_X);
    assert_eq!(
        one(tid),
        unsupported(tid, Unsupported::SpecialRegister(SpecialReg::TID_X))
    );
}

/// `S2R` to `RZ` reads nothing, so it adds no lane id (and no subgroup requirement); the special
/// register is still checked.
#[test]
fn s2r_to_rz_reads_nothing() {
    let s2r = |sr| Instr::new(Op::S2R(OpS2R { dst: Gpr::RZ, sr }));
    for stage in [VERTEX, pixel([0; COLOR_TARGETS])] {
        let ir = translate(stage, &[s2r(SpecialReg::LANE_ID), exit()]).expect("translates");
        assert_eq!(ir.to_string(), "block0:\n");
    }
    let tid = s2r(SpecialReg::TID_X);
    assert_eq!(
        one(tid),
        unsupported(tid, Unsupported::SpecialRegister(SpecialReg::TID_X))
    );
}

/// A guard on a real predicate, plain or negated, needs flow analysis (a listed divergence: the
/// PR selects a pure instruction's result or makes a side effect its own block).
#[test]
fn guarded_instructions_are_rejected() {
    for negated in [false, true] {
        let guard = PredSrc {
            reg: Pred::new(0).expect("p0"),
            negated,
        };
        for instr in [
            mov(r(0), MovSrc::Imm(1)),
            Instr::new(Op::ASt(ast(r(0), 0x80, 1))),
        ] {
            let instr = Instr { guard, ..instr };
            assert_eq!(one(instr), unsupported(instr, Unsupported::Guard));
        }
    }
}

/// `@!PT` never runs: the PR skips the instruction (`Visitor::Run`, `guard.Never()`), starquake
/// rejects it (a listed divergence).
#[test]
fn never_guarded_instructions_are_rejected() {
    let never = PredSrc {
        reg: Pred::PT,
        negated: true,
    };
    for instr in [
        mov(r(0), MovSrc::Imm(1)),
        Instr::new(Op::ASt(ast(r(0), 0x80, 1))),
    ] {
        let instr = Instr {
            guard: never,
            ..instr
        };
        assert_eq!(one(instr), unsupported(instr, Unsupported::Guard));
    }
    // The same instruction under `PT` translates.
    assert!(one(mov(r(0), MovSrc::Imm(1))).is_ok());
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

#[test]
fn ffma_multiply_zero_mode_is_rejected() {
    let op = OpFFma {
        dst: r(0),
        srcs: [FloatSrc::reg(r(1)); 3],
        saturate: false,
        rnd: RoundMode::NearestEven,
        ftz: false,
        dnz: true,
    };
    let instr = Instr::new(Op::FFma(op));
    assert_eq!(
        one(instr),
        unsupported(instr, Unsupported::FfmaMultiplyZero)
    );
    assert!(one(Instr::new(Op::FFma(OpFFma { dnz: false, ..op }))).is_ok());
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
        (
            OpALd {
                vtx: r(2),
                ..ald(r(0), 0x80, 1)
            },
            Unsupported::AttributeVertex,
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
        (
            OpASt {
                vtx: r(2),
                ..ast(r(0), 0x80, 1)
            },
            Unsupported::AttributeVertex,
        ),
    ];
    for (op, what) in ast_cases {
        let instr = Instr::new(Op::ASt(op));
        assert_eq!(one(instr), unsupported(instr, what));
    }
}

/// Instructions with two faults at once report the one the PR checks first (offset register,
/// patch or physical, output, address, registers), then starquake's vertex register check.
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
        // Bad registers and a vertex register: the registers are reported first.
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
            "instruction 0 (ald.64 r0, a[0x81];): attribute address 0x81 with 2 words is not \
             supported"
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
}
