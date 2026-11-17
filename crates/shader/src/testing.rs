//! Shared test helpers: the property-test configuration and generators of instructions and
//! words.

use proptest::prelude::*;
use proptest::strategy::Union;
use proptest::test_runner::{Config, RngSeed};

use crate::decode::{
    AluSrc, CBuf, CBufRef, GROUP_HOOKS, GROUPS, Gpr, Instr, Instruction, OPCODE, Pred, PredSrc,
    UniformReg,
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

pub(crate) fn any_pred_src() -> impl Strategy<Value = PredSrc> {
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
pub(crate) fn any_reg_src() -> impl Strategy<Value = AluSrc> {
    (any_gpr(), any::<bool>(), any::<bool>()).prop_map(|(reg, abs, neg)| AluSrc::Reg {
        reg,
        abs,
        neg,
    })
}

/// A source that takes the wide slot, with any modifiers its kind has.
pub(crate) fn any_wide_src() -> impl Strategy<Value = AluSrc> {
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

/// Any encodable instruction of the decoded forms: a guard and an operation from one of every
/// group's generators (`any_ops` in each group's module), each operation equally likely.
pub(crate) fn any_instr() -> impl Strategy<Value = Instr> {
    let ops = GROUP_HOOKS.iter().flat_map(|hooks| (hooks.any_ops)());
    (any_pred_src(), Union::new(ops)).prop_map(|(guard, op)| Instr { guard, op })
}
