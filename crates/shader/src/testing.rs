//! Shared test helpers: the property-test configuration and generators of instructions and
//! words.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use crate::decode::{
    AluSrc, CBuf, CBufRef, GROUPS, Gpr, Instr, Instruction, MovSrc, OPCODE, Op, OpALd, OpASt,
    OpExit, OpFFma, OpMov, OpS2R, Pred, PredSrc, RoundMode, SpecialReg, UniformReg,
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

/// Every decodable form, as its opcode, its known-bit mask and its fixed bits, in table order.
pub(crate) fn forms() -> Vec<(u16, u128, u128)> {
    GROUPS
        .iter()
        .flat_map(|group| group.iter())
        .map(|form| (form.opcode, form.known, form.fixed))
        .collect()
}

/// A word of one of the [`forms`]: its opcode and fixed bits, random bits wherever the form
/// has fields. The decoder may still reject it in a field's other layout (see `decode`, step 5).
pub(crate) fn any_form_word() -> impl Strategy<Value = Instruction> {
    let forms = forms();
    (0..forms.len(), any::<u128>()).prop_map(move |(index, noise)| {
        let (opcode, known, fixed) = forms[index];
        Instruction::from_bits((noise & known) | fixed).with_field(OPCODE, u64::from(opcode))
    })
}

pub(crate) fn any_gpr() -> impl Strategy<Value = Gpr> {
    any::<u8>().prop_map(Gpr::new)
}

pub(crate) fn any_uniform_reg() -> impl Strategy<Value = UniformReg> {
    (0..=UniformReg::URZ.index())
        .prop_map(|index| UniformReg::new(index).unwrap_or(UniformReg::URZ))
}

fn any_pred_src() -> impl Strategy<Value = PredSrc> {
    (0..=Pred::PT.index(), any::<bool>()).prop_map(|(reg, negated)| PredSrc {
        reg: Pred::new(reg).unwrap_or(Pred::PT),
        negated,
    })
}

/// Any encodable constant-buffer reference: a bound buffer 0 to 31 or a bindless one.
pub(crate) fn any_cbuf() -> impl Strategy<Value = CBufRef> {
    let buf = prop_oneof![
        (0_u8..32).prop_map(CBuf::Binding),
        any_uniform_reg().prop_map(CBuf::BindlessUniform),
    ];
    (buf, any::<u16>()).prop_map(|(buf, offset)| CBufRef { buf, offset })
}

/// A register source with any modifiers.
fn any_reg_src() -> impl Strategy<Value = AluSrc> {
    (any_gpr(), any::<bool>(), any::<bool>()).prop_map(|(reg, abs, neg)| AluSrc::Reg {
        reg,
        abs,
        neg,
    })
}

/// A source that takes the wide slot, with any modifiers its kind has.
fn any_wide_src() -> impl Strategy<Value = AluSrc> {
    prop_oneof![
        any::<u32>().prop_map(AluSrc::Imm),
        (any_cbuf(), any::<bool>(), any::<bool>()).prop_map(|(cbuf, abs, neg)| AluSrc::CBuf {
            cbuf,
            abs,
            neg
        }),
        (any_uniform_reg(), any::<bool>(), any::<bool>())
            .prop_map(|(reg, abs, neg)| AluSrc::UniformReg { reg, abs, neg }),
    ]
}

fn any_ffma() -> impl Strategy<Value = Op> {
    let srcs = prop_oneof![
        (any_reg_src(), any_reg_src(), any_reg_src()).prop_map(|(a, b, c)| [a, b, c]),
        (any_reg_src(), any_wide_src(), any_reg_src()).prop_map(|(a, b, c)| [a, b, c]),
        (any_reg_src(), any_reg_src(), any_wide_src()).prop_map(|(a, b, c)| [a, b, c]),
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

fn any_mov() -> impl Strategy<Value = Op> {
    let src = prop_oneof![
        any_gpr().prop_map(MovSrc::Reg),
        any::<u32>().prop_map(MovSrc::Imm),
        any_cbuf().prop_map(MovSrc::CBuf),
        any_uniform_reg().prop_map(MovSrc::UniformReg),
    ];
    (any_gpr(), src, 0_u8..16).prop_map(|(dst, src, quad_lanes)| {
        Op::Mov(OpMov {
            dst,
            src,
            quad_lanes,
        })
    })
}

/// Any encodable instruction of the decoded forms.
pub(crate) fn any_instr() -> impl Strategy<Value = Instr> {
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
    let op = prop_oneof![any_mov(), any_ffma(), ald, ast, exit, s2r];
    (any_pred_src(), op).prop_map(|(guard, op)| Instr { guard, op })
}
