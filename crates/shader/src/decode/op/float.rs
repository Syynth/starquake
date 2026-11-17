//! Slice 6b, 32-bit float and conversions: `FFMA` is decoded in all seven operand forms; the
//! other instructions are shells.
//!
//! `FFMA`'s shape follows NAK's `OpFFma` and its encoding NAK's `impl SM70Op for OpFFma`
//! (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` and `sm70_encode.rs` @ 52238762, MIT,
//! Copyright 2022 Collabora, Ltd.): `encode_alu(0x023, dst, src0, src1, src2)` with source
//! modifiers, then the bits below. PR #4588's table has a row for each of the seven forms
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97, `0x223`
//! to `0xe23`).

use std::fmt::{self, Display, Formatter};

use super::super::DecodeError;
use super::super::alu::{self, AluForm, Source, Sources};
use super::super::fields::{DST, mask_of};
use super::super::form::{Form, Reader};
use super::super::instruction::Field;
use super::super::operand::{AluSrc, Gpr, RoundMode};
use super::super::print::{FloatOperand, flag};
use super::{Named, Op, shells};
#[cfg(test)]
use {super::super::form::ReferenceRow, proptest::strategy::BoxedStrategy};

/// `FFMA`'s 9-bit opcode (`impl SM70Op for OpFFma`: `encode_alu(0x023, ..)`).
pub(crate) const FFMA_OPCODE: u16 = 0x023;
/// `FFMA`'s "denorm/zero multiply" bit (`set_bit(76, dnz)`).
pub(crate) const FFMA_DNZ: Field = Field::bit(76);
/// `FFMA`'s saturate bit (`set_bit(77, saturate)`).
pub(crate) const FFMA_SAT: Field = Field::bit(77);
/// `FFMA`'s rounding mode (`set_rnd_mode(78..80, ..)`).
pub(crate) const FFMA_RND: Field = Field::new(78, RoundMode::BITS);
/// `FFMA`'s flush-to-zero bit (`set_bit(80, ftz)`).
pub(crate) const FFMA_FTZ: Field = Field::bit(80);
/// `FFMA`'s sources: all three.
pub(crate) const FFMA_SOURCES: Sources = Sources {
    src0: true,
    src1: true,
    src2: true,
};

/// `FFMA dst, a, b, c`: `a * b + c` in 32-bit float, fused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpFFma {
    /// The destination.
    pub dst: Gpr,
    /// `a`, `b`, `c`. `a` is always a register; at most one of `b` and `c` is in the wide slot
    /// (an immediate, a constant buffer or a uniform register).
    pub srcs: [AluSrc; 3],
    /// Clamp the result to [0, 1].
    pub saturate: bool,
    /// The rounding mode.
    pub rnd: RoundMode,
    /// Flush denormal inputs and results to zero.
    pub ftz: bool,
    /// NAK's `dnz` bit; PR #4588 calls it the multiply-by-zero mode
    /// (`ops_alu.cpp`, `FFMA`: "native FFMA multiply-zero mode").
    pub dnz: bool,
}

impl Named for OpFFma {
    const MNEMONIC: &'static str = "ffma";
}

impl Display for OpFFma {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("ffma")?;
        flag(f, self.ftz, ".ftz")?;
        flag(f, self.dnz, ".fmz")?;
        write!(f, "{}", self.rnd)?;
        flag(f, self.saturate, ".sat")?;
        let [a, b, c] = self.srcs.map(FloatOperand);
        write!(f, " {}, {a}, {b}, {c}", self.dst)
    }
}

fn decode_ffma(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::FFma(OpFFma {
        dst: reader.gpr(DST),
        srcs: [
            alu::src0(reader, true),
            alu::src(reader, form, Source::One, true),
            alu::src(reader, form, Source::Two, true),
        ],
        saturate: reader.flag(FFMA_SAT),
        rnd: RoundMode::from_bits(reader.field(FFMA_RND)),
        ftz: reader.flag(FFMA_FTZ),
        dnz: reader.flag(FFMA_DNZ),
    }))
}

/// `FFMA` in `form`.
const fn ffma(form: AluForm) -> Form {
    Form::new(
        form.opcode(FFMA_OPCODE),
        mask_of(&[DST, FFMA_DNZ, FFMA_SAT, FFMA_RND, FFMA_FTZ])
            | alu::known(form, FFMA_SOURCES, true),
        decode_ffma,
    )
    .with_fixed(alu::fixed(form))
}

/// The 32-bit float group's forms.
pub(crate) const FORMS: &[Form] = &[
    ffma(AluForm::ALL[0]),
    ffma(AluForm::ALL[1]),
    ffma(AluForm::ALL[2]),
    ffma(AluForm::ALL[3]),
    ffma(AluForm::ALL[4]),
    ffma(AluForm::ALL[5]),
    ffma(AluForm::ALL[6]),
];

/// PR #4588's rows for the group's forms, in [`FORMS`] order
/// (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97; each
/// row's four words, low word first).
#[cfg(test)]
pub(crate) const REFERENCE_ROWS: &[ReferenceRow] = &[
    ReferenceRow::new(0x223, [0xffff_ffff, 0xc000_00ff, 0x0001_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x423, [0xffff_ffff, 0xffff_ffff, 0x0001_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x623, [0xffff_ffff, 0xc7ff_ffff, 0x0801_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x823, [0xffff_ffff, 0xffff_ffff, 0x0001_ffff, 0xffff_fe00]),
    ReferenceRow::new(0xa23, [0xffff_ffff, 0xc7ff_ffff, 0x0801_ffff, 0xffff_fe00]),
    ReferenceRow::new(0xc23, [0xffff_ffff, 0xc000_003f, 0x0801_ffff, 0xffff_fe00]),
    ReferenceRow::new(0xe23, [0xffff_ffff, 0xc000_003f, 0x0801_ffff, 0xffff_fe00]),
];

/// Generators of the group's decoded operations, one per operation.
#[cfg(test)]
pub(crate) fn any_ops() -> Vec<BoxedStrategy<Op>> {
    use proptest::prelude::*;

    use crate::testing::{any_gpr, any_reg_src, any_wide_src};

    let srcs = prop_oneof![
        (any_reg_src(), any_reg_src(), any_reg_src()).prop_map(|(a, b, c)| [a, b, c]),
        (any_reg_src(), any_wide_src(), any_reg_src()).prop_map(|(a, b, c)| [a, b, c]),
        (any_reg_src(), any_reg_src(), any_wide_src()).prop_map(|(a, b, c)| [a, b, c]),
    ];
    let rnd = proptest::sample::select(RoundMode::ALL.to_vec());
    let ffma = (any_gpr(), srcs, any::<[bool; 3]>(), rnd).prop_map(|(dst, srcs, flags, rnd)| {
        let [saturate, ftz, dnz] = flags;
        Op::FFma(OpFFma {
            dst,
            srcs,
            saturate,
            rnd,
            ftz,
            dnz,
        })
    });
    vec![ffma.boxed()]
}

shells! {
    FAdd(OpFAdd) = "fadd";
    FMul(OpFMul) = "fmul";
    FMnMx(OpFMnMx) = "fmnmx";
    FSet(OpFSet) = "fset";
    FSetP(OpFSetP) = "fsetp";
    FSel(OpFSel) = "fsel";
    MuFu(OpMuFu) = "mufu";
    FSwzAdd(OpFSwzAdd) = "fswzadd";
    FRnd(OpFRnd) = "frnd";
    F2F(OpF2F) = "f2f";
    F2I(OpF2I) = "f2i";
    I2F(OpI2F) = "i2f";
    I2FP(OpI2FP) = "i2fp";
    F2FP(OpF2FP) = "f2fp";
}
