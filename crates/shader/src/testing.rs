//! Shared test helpers: the property-test configuration, generators of instructions and words,
//! and the words of a jump table's selector pattern.

use proptest::prelude::*;
use proptest::strategy::Union;
use proptest::test_runner::{Config, RngSeed};

use crate::decode::{
    AluSrc, CBuf, CBufRef, Form, GROUP_HOOKS, GROUPS, Gpr, Instr, Instruction, OPCODE, Pred,
    PredSrc, ReferenceRow, UniformReg,
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

/// The form of `opcode` and PR #4588's row for it, from the group that lists the form (each
/// group's rows are in its `FORMS` order: `forms_have_distinct_opcodes_and_lookup_finds_each`).
pub(crate) fn form_and_row(opcode: u16) -> (&'static Form, &'static ReferenceRow) {
    GROUPS
        .iter()
        .zip(&GROUP_HOOKS)
        .find_map(|(forms, hooks)| {
            let index = forms.iter().position(|form| form.opcode == opcode)?;
            Some((forms.get(index)?, hooks.reference_rows.get(index)?))
        })
        .expect("a decoded form's opcode")
}

/// The canonical word of the form of `opcode`: the opcode, the form's fixed bits, and the
/// fields its row's [`ReferenceRow::canonical`] sets. Every form decodes its canonical word
/// (`decode::tests::framework::canonical_words_decode`).
pub(crate) fn canonical_word(opcode: u16) -> Instruction {
    let (form, row) = form_and_row(opcode);
    Instruction::from_bits(form.fixed | row.canonical).with_field(OPCODE, u64::from(opcode))
}

/// A word of one of the [`forms`]: its opcode and fixed bits, random bits wherever the form
/// has fields. The decoder may still reject it in a field's other layout (see `decode`, step 5).
pub(crate) fn any_form_word() -> impl Strategy<Value = Instruction> {
    any_word_of(GROUPS.concat())
}

/// A word of one of `forms`, made as [`any_form_word`] makes one: for one group's tests.
pub(crate) fn any_word_of(forms: impl Into<Vec<Form>>) -> impl Strategy<Value = Instruction> {
    let forms: Vec<Form> = forms.into();
    (0..forms.len(), any::<u128>()).prop_map(move |(index, noise)| {
        let form = forms[index];
        Instruction::from_bits((noise & form.known) | form.fixed)
            .with_field(OPCODE, u64::from(form.opcode))
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

/// Any predicate register, `PT` included.
pub(crate) fn any_pred() -> impl Strategy<Value = Pred> {
    (0..=Pred::PT.index()).prop_map(|index| Pred::new(index).unwrap_or(Pred::PT))
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

/// An unmodified source of the ALU forms that put source 1 in the wide slot (1, 4, 5 and 6):
/// a register, an immediate, a constant buffer or a uniform register.
pub(crate) fn any_unmodified_src() -> impl Strategy<Value = AluSrc> {
    prop_oneof![
        any_gpr().prop_map(AluSrc::reg),
        any::<u32>().prop_map(AluSrc::Imm),
        any_cbuf().prop_map(|cbuf| AluSrc::CBuf {
            cbuf,
            abs: false,
            neg: false
        }),
        any_uniform_reg().prop_map(|reg| AluSrc::UniformReg {
            reg,
            abs: false,
            neg: false
        }),
    ]
}

/// Any encodable instruction of the decoded forms: a guard and an operation from one of every
/// group's generators (`any_ops` in each group's module), each operation equally likely.
pub(crate) fn any_instr() -> impl Strategy<Value = Instr> {
    let ops = GROUP_HOOKS.iter().flat_map(|hooks| (hooks.any_ops)());
    (any_pred_src(), Union::new(ops)).prop_map(|(guard, op)| Instr { guard, op })
}

// The reference's literals (`program_capture.h`, `TrackCaptureIndirectBranches` @ e5fb3b97),
// written out to build words independently of the shipping constants.
const REF_LDC: u128 = 0xb82;
const REF_SHF: u128 = 0x819;
const REF_IMNMX: u128 = 0x817;
const REF_ALWAYS: u128 = 7 << 12;

/// `IMNMX.U32 dst, src, clamp, PT` as the reference matches it.
pub(crate) fn imnmx(dst: u8, src: u8, clamp: u32) -> Instruction {
    Instruction::from_bits(
        REF_IMNMX
            | REF_ALWAYS
            | u128::from(dst) << 16
            | u128::from(src) << 24
            | u128::from(clamp) << 32
            | 0x0380_0000 << 64,
    )
}

/// `SHF.L.U32 dst, src, 0x2, RZ`.
pub(crate) fn shf(dst: u8, src: u8) -> Instruction {
    Instruction::from_bits(
        REF_SHF
            | REF_ALWAYS
            | u128::from(dst) << 16
            | u128::from(src) << 24
            | 2 << 32
            | 0x6ff << 64,
    )
}

/// `LDC.32 dst, c[bank][src + offset]`.
pub(crate) fn ldc(dst: u8, src: u8, bank: u8, offset: u32) -> Instruction {
    Instruction::from_bits(
        REF_LDC
            | REF_ALWAYS
            | u128::from(dst) << 16
            | u128::from(src) << 24
            | (u128::from(bank) << 22 | u128::from(offset) << 6) << 32
            | 0x800 << 64,
    )
}
