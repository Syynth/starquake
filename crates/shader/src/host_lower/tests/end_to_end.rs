//! The production translator end to end ([`translate_shader`], step 6 of its module docs):
//! compute programs of `DADD`, `HADD2` and `VOTE` assembled, translated, put through the IR
//! half, the host lowering passes the target calls for, and the lowering; the target's caps
//! decide which pass runs and which shaders are refused, with the exact reasons.

use super::super::lower_for_host;
use super::{count, first};
use crate::asm::assemble;
use crate::cache::{CodeForm, HostTarget, ShaderInput};
use crate::capture::capture;
use crate::decode::{
    AluSrc, Gpr, HalfSrc, Instr, Instruction, MemAddr, MemSpace, MemType, Op, OpDAdd, OpExit,
    OpHAdd2, OpLd, OpSt, OpVote, OffsetStride, Pred, PredSrc, RoundMode, UniformAddr, UniformReg,
    VoteDst, VoteOp,
};
use crate::exchange::Answers;
use crate::flow::BranchTables;
use crate::host::{Feature, HostCaps, Requirements};
use crate::interp::Machine;
use crate::ir::Opcode;
use crate::lower::LowerError;
use crate::pipeline::{PipelineProgram, host_caps, translate_shader};
use crate::program::{Rejection, pipeline_ir};
use crate::translate::{ColorOutputs, GuestStage, Stage};
use crate::variant::VariantKey;

fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

/// `[RZ + offset]` in shared memory, with the uniform register field off.
fn shared(offset: i32) -> MemAddr {
    MemAddr {
        reg: Gpr::RZ,
        reg_64: false,
        uniform: Some(UniformAddr {
            reg: UniformReg::new(0).expect("a uniform register"),
            wide: false,
            enabled: false,
        }),
        offset,
    }
}

/// `LDS dst, [RZ + offset]`, one word.
fn lds(dst: u8, offset: i32) -> Instr {
    Instr::new(Op::Ld(OpLd {
        dst: r(dst),
        addr: shared(offset),
        mem_type: MemType::B32,
        space: MemSpace::Shared(OffsetStride::X1),
        pred: PredSrc::TRUE,
        pred_dst: Pred::PT,
        zd: false,
        bit76: false,
    }))
}

/// `STS [RZ + offset], data`, one word.
fn sts(data: u8, offset: i32) -> Instr {
    Instr::new(Op::St(OpSt {
        data: r(data),
        addr: MemAddr {
            uniform: None,
            ..shared(offset)
        },
        mem_type: MemType::B32,
        space: MemSpace::Shared(OffsetStride::X1),
        bit76: false,
    }))
}

fn exit() -> Instr {
    Instr::new(Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    }))
}

fn words(instrs: &[Instr]) -> Vec<Instruction> {
    instrs
        .iter()
        .map(|instr| assemble(instr).expect("encodable"))
        .collect()
}

/// `R0:R1 + R2:R3` (read from shared words 0 to 3) stored to shared words 4 and 5.
fn double_program() -> Vec<Instruction> {
    words(&[
        lds(0, 0),
        lds(1, 4),
        lds(2, 8),
        lds(3, 12),
        Instr::new(Op::DAdd(OpDAdd {
            dst: r(4),
            srcs: [AluSrc::reg(r(0)), AluSrc::reg(r(2))],
            rnd: RoundMode::NearestEven,
        })),
        sts(4, 16),
        sts(5, 20),
        exit(),
    ])
}

/// The packed halves of shared words 0 and 1 added lane by lane into shared word 2.
fn half_program() -> Vec<Instruction> {
    words(&[
        lds(0, 0),
        lds(1, 4),
        Instr::new(Op::HAdd2(OpHAdd2 {
            dst: r(2),
            srcs: [
                HalfSrc::new(AluSrc::reg(r(0))),
                HalfSrc::new(AluSrc::reg(r(1))),
            ],
            saturate: false,
            f32: false,
            ftz: false,
        })),
        sts(2, 8),
        exit(),
    ])
}

/// `VOTE.ANY R0, PT, PT`: the ballot of the warp, stored to shared word 0.
fn vote_program() -> Vec<Instruction> {
    words(&[
        Instr::new(Op::Vote(OpVote {
            op: VoteOp::Any,
            dst: VoteDst::Reg {
                ballot: r(0),
                vote: Pred::PT,
            },
            pred: PredSrc::TRUE,
        })),
        sts(0, 0),
        exit(),
    ])
}

/// A compute variant with 32 bytes of shared memory.
fn variant() -> VariantKey {
    VariantKey {
        workgroup_size: [32, 1, 1],
        shared_memory_size: 32,
        ..VariantKey::default()
    }
}

/// A target with exactly the features `caps` names.
fn target(caps: HostCaps) -> HostTarget {
    HostTarget {
        form: CodeForm::NagaModule,
        caps: Requirements {
            subgroup: caps.subgroup,
            subgroup_size_32: caps.subgroup_size_32,
            f16: caps.f16,
            f64: caps.f64,
            i64: caps.i64,
            i64_atomics: false,
        },
    }
}

/// The production translator's shader for the compute program `code` on a target with `caps`.
fn translated(
    code: &[Instruction],
    caps: HostCaps,
) -> Result<crate::host::HostShader, Rejection> {
    let bytes: Vec<u8> = code.iter().flat_map(|word| word.to_le_bytes()).collect();
    let captured = capture(bytes.as_slice(), GuestStage::Compute).expect("captures");
    let entry = PipelineProgram::new(&captured).expect("an entry");
    let answers = Answers::new();
    let variant = variant();
    let target = target(caps);
    translate_shader(&ShaderInput {
        stage: GuestStage::Compute,
        program: &entry,
        answers: &answers,
        variant: &variant,
        target: &target,
    })
}

/// The IR half's output for the compute program `code`, before the host lowering passes.
fn ir(code: &[Instruction]) -> crate::ir::Function {
    let stage = Stage {
        stage: GuestStage::Compute,
        color_outputs: ColorOutputs::default(),
    };
    pipeline_ir(stage, None, code, &BranchTables::new(0)).expect("the IR half runs")
}

/// A target without 64-bit floats gets a shader from a `DADD` program: the IR half leaves the
/// 64-bit add, the f64 pass turns it into a 32-bit one, the lowering accepts it and the shader
/// needs no 64-bit floats; and the IR the passes leave gives the interpreter's sum of two
/// binary32 operands, as before them.
#[test]
fn the_production_translator_lowers_a_double_program() {
    let caps = host_caps(target(HostCaps::NONE).caps);
    assert!(!caps.f64);
    let shader = translated(&double_program(), caps).expect("translates");
    assert_eq!(shader.requirements, Requirements::default());
    let before = ir(&double_program());
    assert_eq!(count(&before, Opcode::FPAdd64), 1, "{before}");
    let mut after = before.clone();
    lower_for_host(&mut after, caps).expect("lowers for the host");
    assert_eq!(count(&after, Opcode::FPAdd64), 0);
    assert_eq!(count(&after, Opcode::FPAdd32), 1);
    let mut machine = Machine::default();
    machine.shared = [1.5_f64, -0.375]
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .chain([0; 8])
        .collect();
    let read = |m: &Machine| f64::from_le_bytes(m.shared[16..24].try_into().expect("8 bytes"));
    let mut ran_before = machine.clone();
    ran_before.run(&before).expect("runs");
    let mut ran_after = machine;
    ran_after.run(&after).expect("runs");
    assert_eq!(read(&ran_before), 1.125);
    assert_eq!(read(&ran_after), 1.125);
}

/// A target with 64-bit floats keeps the 64-bit add, which the lowering refuses by name: no
/// slice lowers 64-bit floats to naga yet (module docs, "Which passes run").
#[test]
fn a_double_program_is_refused_where_the_target_has_f64() {
    let caps = HostCaps {
        f64: true,
        ..HostCaps::NONE
    };
    let before = ir(&double_program());
    assert_eq!(
        translated(&double_program(), caps).err(),
        Some(Rejection::Lower(LowerError::Operation {
            at: first(&before, Opcode::PackDouble2x32),
            opcode: Opcode::PackDouble2x32
        }))
    );
}

/// An `HADD2` program lowers on a target without 16-bit floats through the f16 pass (the
/// shader needs none), and natively on one with them (the shader needs them): the target's
/// `f16` is the only thing that varies.
#[test]
fn the_production_translator_lowers_a_half_program() {
    let without = translated(&half_program(), HostCaps::NONE).expect("translates");
    assert_eq!(without.requirements, Requirements::default());
    let with = HostCaps {
        f16: true,
        ..HostCaps::NONE
    };
    let native = translated(&half_program(), with).expect("translates");
    assert_eq!(
        native.requirements,
        Requirements {
            f16: true,
            ..Requirements::default()
        }
    );
}

/// A warp vote records "needs subgroup size 32", and the target decides: refused for lacking
/// subgroups, refused for lacking a size of 32 with subgroups, and translated, with both
/// requirements set, where the target has both.
#[test]
fn a_warp_program_needs_a_subgroup_size_of_32() {
    let subgroups = HostCaps {
        subgroup: true,
        ..HostCaps::NONE
    };
    let both = HostCaps {
        subgroup_size_32: true,
        ..subgroups
    };
    assert_eq!(
        translated(&vote_program(), HostCaps::NONE).err(),
        Some(Rejection::Lower(LowerError::Missing(Feature::Subgroup)))
    );
    assert_eq!(
        translated(&vote_program(), subgroups).err(),
        Some(Rejection::Lower(LowerError::Missing(Feature::SubgroupSize32)))
    );
    let shader = translated(&vote_program(), both).expect("translates");
    assert_eq!(
        shader.requirements,
        Requirements {
            subgroup: true,
            subgroup_size_32: true,
            ..Requirements::default()
        }
    );
}
