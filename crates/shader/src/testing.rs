//! Shared test helpers: the property-test configuration and generators of instructions.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use crate::decode::{
    ALD_KNOWN, ALD_OPCODE, AST_KNOWN, AST_OPCODE, EXIT_KNOWN, EXIT_OPCODE, FFMA_IMM_SRC1,
    FFMA_IMM_SRC1_KNOWN, FFMA_IMM_SRC2, FFMA_IMM_SRC2_KNOWN, FFMA_REG, FFMA_REG_KNOWN, FloatSrc,
    Gpr, Instr, Instruction, MOV_IMM, MOV_IMM_KNOWN, MOV_REG, MOV_REG_KNOWN, MovSrc, OPCODE, Op,
    OpALd, OpASt, OpExit, OpFFma, OpMov, OpS2R, Pred, PredSrc, RoundMode, S2R_KNOWN, S2R_OPCODE,
    SpecialReg,
};

/// A fixed seed keeps every property test deterministic.
pub(crate) fn config() -> Config {
    Config {
        cases: 512,
        rng_seed: RngSeed::Fixed(0x5348_4144),
        failure_persistence: None,
        ..Config::default()
    }
}

/// Every decodable form: its 12-bit opcode and the bits it explains.
pub(crate) const FORMS: [(u64, u128); 9] = [
    (MOV_REG, MOV_REG_KNOWN),
    (MOV_IMM, MOV_IMM_KNOWN),
    (FFMA_REG, FFMA_REG_KNOWN),
    (FFMA_IMM_SRC2, FFMA_IMM_SRC2_KNOWN),
    (FFMA_IMM_SRC1, FFMA_IMM_SRC1_KNOWN),
    (ALD_OPCODE, ALD_KNOWN),
    (AST_OPCODE, AST_KNOWN),
    (EXIT_OPCODE, EXIT_KNOWN),
    (S2R_OPCODE, S2R_KNOWN),
];

/// A word of one of the [`FORMS`]: its opcode, random bits wherever the form has fields.
pub(crate) fn any_form_word() -> impl Strategy<Value = Instruction> {
    (0..FORMS.len(), any::<u128>()).prop_map(|(index, noise)| {
        let (opcode, known) = FORMS[index];
        Instruction::from_bits(noise & known).with_field(OPCODE, opcode)
    })
}

fn any_gpr() -> impl Strategy<Value = Gpr> {
    any::<u8>().prop_map(Gpr::new)
}

fn any_pred_src() -> impl Strategy<Value = PredSrc> {
    (0_u8..=7, any::<bool>()).prop_map(|(reg, negated)| PredSrc {
        reg: Pred::new(reg).unwrap_or(Pred::PT),
        negated,
    })
}

fn any_float_reg() -> impl Strategy<Value = FloatSrc> {
    (any_gpr(), any::<bool>(), any::<bool>()).prop_map(|(reg, abs, neg)| FloatSrc::Reg {
        reg,
        abs,
        neg,
    })
}

fn any_ffma() -> impl Strategy<Value = Op> {
    let srcs = prop_oneof![
        (any_float_reg(), any_float_reg(), any_float_reg()).prop_map(|(a, b, c)| [a, b, c]),
        (any_float_reg(), any::<u32>(), any_float_reg()).prop_map(|(a, b, c)| [
            a,
            FloatSrc::Imm(b),
            c
        ]),
        (any_float_reg(), any_float_reg(), any::<u32>()).prop_map(|(a, b, c)| [
            a,
            b,
            FloatSrc::Imm(c)
        ]),
    ];
    (any_gpr(), srcs, any::<[bool; 3]>(), 0_u64..4).prop_map(|(dst, srcs, flags, rnd)| {
        let [saturate, ftz, dnz] = flags;
        Op::FFma(OpFFma {
            dst,
            srcs,
            saturate,
            rnd: RoundMode::from_bits(rnd),
            ftz,
            dnz,
        })
    })
}

/// Any encodable instruction of the six forms.
pub(crate) fn any_instr() -> impl Strategy<Value = Instr> {
    let mov = (
        any_gpr(),
        prop_oneof![
            any_gpr().prop_map(MovSrc::Reg),
            any::<u32>().prop_map(MovSrc::Imm)
        ],
        0_u8..16,
    )
        .prop_map(|(dst, src, quad_lanes)| {
            Op::Mov(OpMov {
                dst,
                src,
                quad_lanes,
            })
        });
    let attr = || {
        (
            any_gpr(),
            any_gpr(),
            any_gpr(),
            0_u16..1024,
            1_u8..=4,
            any::<[bool; 3]>(),
        )
    };
    let ald = attr().prop_map(|(dst, vtx, offset, addr, comps, [patch, phys, output])| {
        Op::ALd(OpALd {
            dst,
            vtx,
            offset,
            addr,
            comps,
            patch,
            phys,
            output,
        })
    });
    let ast = attr().prop_map(|(data, vtx, offset, addr, comps, [patch, phys, _])| {
        Op::ASt(OpASt {
            data,
            vtx,
            offset,
            addr,
            comps,
            patch,
            phys,
        })
    });
    let exit = (any::<bool>(), any::<bool>(), any_pred_src()).prop_map(
        |(keep_ref_count, no_at_exit, pred)| {
            Op::Exit(OpExit {
                keep_ref_count,
                no_at_exit,
                pred,
            })
        },
    );
    let s2r = (any_gpr(), any::<u8>()).prop_map(|(dst, sr)| {
        Op::S2R(OpS2R {
            dst,
            sr: SpecialReg(sr),
        })
    });
    let op = prop_oneof![mov, any_ffma(), ald, ast, exit, s2r];
    (any_pred_src(), op).prop_map(|(guard, op)| Instr { guard, op })
}
