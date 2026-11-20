//! Slice 6c, half and double precision: every instruction of the group is decoded, in the
//! vector datapath's forms.
//!
//! # Sources
//!
//! Shapes follow NAK's `Op*` structs and encodings NAK's `impl SM70Op` blocks for `OpHAdd2`,
//! `OpHFma2`, `OpHMul2`, `OpHSet2`, `OpHSetP2`, `OpHMnMx2`, `OpDAdd`, `OpDMul`, `OpDFma` and
//! `OpDSetP` (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` and `sm70_encode.rs` @ 52238762,
//! MIT, Copyright 2022 Collabora, Ltd.), as each constant says. Where NAK writes a field as a
//! constant (`HFMA2`'s and `HMUL2`'s `.RELU`, `HFMA2`'s and `HMNMX2`'s `.BF16_V2`, `HMNMX2`'s
//! `.F32`, `.NAN` and `.XORSIGN`, `HSET2`'s `.BF`, the second destination of `DSETP`), the op
//! carries it as a field so the decoder can read any value of it. Masks are PR #4588's rows
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97), and
//! the fields NAK does not write are read where PR #4588's translator reads them
//! (`ops_half.cpp` @ e5fb3b97). PR #4588's translator has no double-precision instruction, so
//! `DADD`, `DMUL`, `DFMA` and `DSETP` come from NAK alone.
//!
//! The packed-half sources (`HalfSrc`) are NAK's `F16v2` sources (`encode_fp16v2_alu`): each
//! has a two-bit swizzle beside its modifier bits, and in the register slot those bits move to
//! 81..85 (`decode::alu`'s module docs). PR #4588's rows already mark every swizzle bit NAK
//! writes in these forms, so no form adds a bit to its row (slice 4b's known bits are not
//! needed here).
//!
//! From PR #4588 alone, and so **each needs an `nvdisasm` check** (spec, verification E2):
//! `HFMA2`'s bit 86, which negates only the high half of `b` (`Visitor::HFMA2`:
//! `negate_high_b = Take(86, 1)`; `lower_to_maxwell.h`: "H0_NH1: negate only the upper packed
//! component"), and its bits 87..89, which hold 7 exactly when `.RELU` is set
//! (`Require(Take(87, 4) == (relu ? 7U : 0U), "native half fused multiply-add mode")`: "The relu
//! form carries a constant 7 in bits 87..90 (observed on every captured program)").
//!
//! # Forms
//!
//! Each instruction is decoded in the operand forms NAK's encoder produces for it. `HADD2`,
//! `HSET2`, `HSETP2` and `DSETP` put their second operand in the wide slot in every form: as
//! source 1 when it is a register (form 1), as source 2 otherwise (forms 2, 3 and 7; NAK's
//! `src_is_zero_or_gpr` choice, `alu::WIDE_OPERAND_FORMS`). `HMUL2`, `HMNMX2` and `DMUL` put it
//! in source 1: forms 1, 4, 5 and 6. `DADD` puts it in source 2 always (`encode_alu(0x029, ..,
//! None, srcs[1])`): the register slot in form 1, the wide slot in forms 2, 3 and 7. `HFMA2`
//! and `DFMA` have all seven forms. The uniform datapath has none of these instructions.
//!
//! # Decisions taken here
//!
//! - **Swizzle 1 is refused** (`HalfSwizzle`): NAK writes it only for a single `F16` source,
//!   PR #4588's translator refuses it for every packed-half source.
//! - **`HFMA2`'s bits 87..89** are read as PR #4588 requires them: 7 with `.RELU`, 0 without;
//!   any other value is [`DecodeError::UnknownValue`]. The op has no field for them (the
//!   assembler writes 7 with `.RELU`), and bit 90, which PR #4588's `Take(87, 4)` also covers
//!   but its row leaves out, stays unknown, so it is refused at step 2 as PR #4588 refuses it
//!   (its required values 0 and 7 have bit 90 clear).
//! - **The printer** is a guess but for `test_hfma2`'s cases: immediates of both kinds print in
//!   hexadecimal (a packed pair of halves, or the 32 bits of a double's immediate), and the
//!   modifiers' order and names (`.f32`, `.bf16_v2`, `.ftz`, `.fmz` for the `dnz` bit as in the
//!   float group, `.sat`, `.relu`, `.bf`, `.h_and`, `.nan`, `.xorsign`, `.h0_nh1` after `b`)
//!   follow `nvdisasm`'s usual style.
//!
//! # Divergences from PR #4588's table and translator
//!
//! | Rows | What this decoder does | Pinned by |
//! |---|---|---|
//! | `HADD2` forms 4, 5, 6 (`0x830`, `0xa30`, `0xc30`), `HSET2` and `HSETP2` forms 4, 5, 6 (`0x833`, `0xa33`, `0xc33`, `0x834`, `0xa34`, `0xc34`), `DSETP` and `DADD` forms 4, 5, 6 (`0x82a`, `0xa2a`, `0xc2a`, `0x829`, `0xa29`, `0xc29`) | Unknown opcodes: NAK never puts their second operand in source 1 when it is not a register. PR #4588's translator accepts `HSETP2` forms 4 and 5 (`Visitor::HalfSource` reads source 1 from the wide slot there) | `decode::tests::half_double::half_double_rows_left_undecoded_are_unknown_opcodes` |
//! | `HMUL2`, `HMNMX2` and `DMUL` forms 2, 3, 7 (`0x432`, `0x632`, `0xe32`, `0x440`, `0x640`, `0xe40`, `0x428`, `0x628`, `0xe28`) | Unknown opcodes: NAK puts their second operand in source 1, which only forms 1, 4, 5 and 6 hold | as above |
//! | Bits 64..71 and 81..84 of every `HADD2` form | Rejected: the register slot, which `HADD2` does not use (NAK passes `None` for it, so writes nothing; PR #4588's `Visitor::HADD2` does not take the bits, and `lower_to_maxwell.h` requires 81..90 clear) | `decode::tests::half_double::half_double_rows_drop_exactly_the_listed_bits` |
//! | Bits 64, 66..68 and 81..84 of every `HSET2` form, bits 64 and 66..68 of every `HSETP2` form | Rejected: what is left of the unused register slot beside the fields (NAK writes nothing there; PR #4588's `Visitor::HSETP2` does not take them, and `lower_to_maxwell.h` requires `HSET2`'s 64..68 and 81..86 clear) | as above |
//! | Bit 65 of every `HSET2` and `HSETP2` form, bit 85 of every `HADD2` and `HMUL2` form | Rejected: NAK's `.BF16_V2`, written false and marked SM 9.0 and later ("SM90+"), which SM86 predates (as the float group's `FMNMX` `.IS_A`); PR #4588 refuses or does not take them | as above |
//! | Bit 78 of every `HMUL2` form | Rejected: NAK's `.F32`, written false and marked "SM70-SM75"; PR #4588 refuses it (`Require(Take(78, 2) == 0, ..)`) | as above |
//! | Bits 64..71 of every `DSETP` form | Rejected: the register slot, which `DSETP` does not use (NAK passes `None`); PR #4588 has no `DSETP` | as above |
//! | The uniform-register forms of the packed-half instructions (`0xe30`, `0xc31`, `0xe31`, `0xc32`, `0xe33`, `0xe34`, `0xc40`) and a bindless constant buffer in their constant-buffer forms | Decoded (NAK encodes them: `encode_alu_ureg`, `set_src_cb`); PR #4588's translator refuses them (`Visitor::HalfSource`: `Require(form >= 1 && form <= 5, ..)`, `Require(Take(91, 1) == 0, "native bindless constant buffer")`) | `decode::tests::half_double::hand_built_words_of_each_form_decode` |
//! | `HFMA2`'s `.fmz` (76), `.f32` (78) and `.bf16_v2` (85), `HMUL2`'s `.fmz` (76) and `.relu` (79), `HMNMX2`'s `.f32` (78), `.ftz` (80), `.nan` (81), `.xorsign` (82) and `.bf16_v2` (85) | Decoded (NAK names each); PR #4588's translator refuses them set | `decode::tests::half_double::hand_built_words_pin_each_modifier_bit` |
//! | `HFMA2` with `.relu` and `.sat`; `HADD2.F32` with `.sat`, `.ftz` or a swizzle below 2 on `b` | Decoded; PR #4588's translator refuses them (`Require(!relu \|\| !saturate, ..)`, `Require(!ftz && !saturate, ..)`, `Require(b.swizzle >= 2, ..)`) | `hand_built_words_pin_each_modifier_bit` |
//!
//! Every other row of a decoded form is matched exactly (`decode::tests::masks_match_the_reference_table`).
//!
//! # Values the decoder refuses
//!
//! A field whose value names nothing is [`DecodeError::UnknownValue`]: swizzle 1 in any of the
//! three swizzle fields (60..62, 74..76, 81..83) of a packed-half source; the combine operation
//! 3 of `HSET2`, `HSETP2` (bits 69..71) and `DSETP` (74..76), as in the float group; and
//! `HFMA2`'s bits 87..89 other than 7 with `.RELU` and 0 without. A decode function reads its
//! fields in this order, so of two such values the earlier is reported: the combine operation,
//! then source 0's swizzle, then the second and third sources', then `HFMA2`'s bits 87..89
//! (`decode::tests::half_double::half_double_checks_run_in_the_documented_order`).

use std::fmt::{self, Display, Formatter};

use super::super::DecodeError;
use super::super::alu::{self, AluForm, Source, Sources};
use super::super::fields::{DST, mask_of};
use super::super::form::{Form, Reader};
use super::super::instruction::Field;
use super::super::operand::{AluSrc, Gpr, HalfSrc, Pred, PredSrc, RoundMode};
use super::super::print::{IntOperand, flag};
use super::control::{COND, COND_NOT, PRED_DST, SOURCE_1_FORMS};
use super::float::{
    CMP_OP, DNZ, FSETP_DST1, FTZ, FloatCmpOp, PredSetOp, RND, SAT, SET_OP, read_pred_set_op,
};
use super::{Named, Op, shells};
#[cfg(test)]
use {
    super::super::fields::{SRC2, SRC2_HALF_ABS, SRC2_HALF_NEG, SRC2_HALF_SWIZZLE, SLOT_UNIFORM},
    super::super::form::ReferenceRow,
    proptest::strategy::BoxedStrategy,
};

// ---------------------------------------------------------------------------------------------
// Opcodes and fields.

/// `HADD2`'s 9-bit opcode (`impl SM70Op for OpHAdd2`: `encode_fp16v2_alu(0x030, ..)`).
pub(crate) const HADD2_OPCODE: u16 = 0x030;
/// `HFMA2`'s 9-bit opcode (`impl SM70Op for OpHFma2`: `encode_fp16v2_alu(0x031, ..)`).
pub(crate) const HFMA2_OPCODE: u16 = 0x031;
/// `HMUL2`'s 9-bit opcode (`impl SM70Op for OpHMul2`: `encode_fp16v2_alu(0x032, ..)`).
pub(crate) const HMUL2_OPCODE: u16 = 0x032;
/// `HSET2`'s 9-bit opcode (`impl SM70Op for OpHSet2`: `encode_fp16v2_alu(0x033, ..)`).
pub(crate) const HSET2_OPCODE: u16 = 0x033;
/// `HSETP2`'s 9-bit opcode (`impl SM70Op for OpHSetP2`: `encode_fp16v2_alu(0x034, ..)`).
pub(crate) const HSETP2_OPCODE: u16 = 0x034;
/// `HMNMX2`'s 9-bit opcode (`impl SM70Op for OpHMnMx2`: `encode_fp16v2_alu(0x040, ..)`).
pub(crate) const HMNMX2_OPCODE: u16 = 0x040;
/// `DMUL`'s 9-bit opcode (`impl SM70Op for OpDMul`: `encode_alu(0x028, ..)`).
pub(crate) const DMUL_OPCODE: u16 = 0x028;
/// `DADD`'s 9-bit opcode (`impl SM70Op for OpDAdd`: `encode_alu(0x029, ..)`).
pub(crate) const DADD_OPCODE: u16 = 0x029;
/// `DSETP`'s 9-bit opcode (`impl SM70Op for OpDSetP`: `encode_alu(0x02a, ..)`).
pub(crate) const DSETP_OPCODE: u16 = 0x02a;
/// `DFMA`'s 9-bit opcode (`impl SM70Op for OpDFma`: `encode_alu(0x02b, ..)`).
pub(crate) const DFMA_OPCODE: u16 = 0x02b;

/// `.F32`: the result as a 32-bit float (NAK: `set_bit(78, self.f32)` in `OpHAdd2` and
/// `OpHFma2`, `set_bit(78, false); // .F32 (SM86)` in `OpHMnMx2`).
pub(crate) const HALF_F32: Field = Field::bit(78);
/// `.RELU`: negative results become zero (NAK: `set_bit(79, false); // .RELU (SM86+)` in
/// `OpHFma2` and `OpHMul2`).
pub(crate) const RELU: Field = Field::bit(79);
/// `.BF16_V2`: the halves are bfloat16 (NAK: `set_bit(85, false); // .BF16_V2 (SM86+)` in
/// `OpHFma2`, `set_bit(85, false); // .BF16_V2` in `OpHMnMx2`).
pub(crate) const BF16_V2: Field = Field::bit(85);
/// `HFMA2`'s negation of `b`'s high half only (PR #4588 alone: `negate_high_b = Take(86, 1)`).
pub(crate) const HFMA2_NEG_HIGH_B: Field = Field::bit(86);
/// `HFMA2`'s bits 87..89, which hold [`HFMA2_RELU_MODE_VALUE`] exactly with `.RELU` (PR #4588
/// alone: `Take(87, 4)`, of which its row knows these three bits; module docs).
pub(crate) const HFMA2_RELU_MODE: Field = Field::new(87, 3);
/// The value of [`HFMA2_RELU_MODE`] with `.RELU` (PR #4588: `relu ? 7U : 0U`).
pub(crate) const HFMA2_RELU_MODE_VALUE: u64 = 7;
/// The combine operation of `HSET2` and `HSETP2` (NAK: `set_pred_set_op(69..71, ..)`; PR #4588
/// `HSETP2`: `Take(69, 2)`).
pub(crate) const HALF_SET_OP: Field = Field::new(69, PredSetOp::BITS);
/// `HSET2`'s `.BF`: 1.0 for true where clear gives an integer mask (NAK: `set_bit(71, true); //
/// .BF`, "This differentiate between integer and fp16 output").
pub(crate) const HSET2_BF: Field = Field::bit(71);
/// `HSETP2`'s `.H_AND`: both halves' results combined (NAK: `set_bit(71, self.horizontal); //
/// .H_AND`; PR #4588: `horizontal = Take(71, 1)`).
pub(crate) const HSETP2_H_AND: Field = Field::bit(71);
/// `HMNMX2`'s `.NAN` (NAK: `set_bit(81, false); // .NAN`).
pub(crate) const HMNMX2_NAN: Field = Field::bit(81);
/// `HMNMX2`'s `.XORSIGN` (NAK: `set_bit(82, false); // .XORSIGN`).
pub(crate) const HMNMX2_XORSIGN: Field = Field::bit(82);

/// The sources of the two-source instructions in forms 1, 4, 5 and 6: 0 and 1.
const SOURCES_0_1: Sources = Sources {
    src0: true,
    src1: true,
    src2: false,
};
/// All three sources.
const SOURCES_ALL: Sources = Sources {
    src0: true,
    src1: true,
    src2: true,
};

/// The sources of an instruction of `alu::WIDE_OPERAND_FORMS` in `form`: 0, and 1 or 2.
const fn wide_operand_sources(form: AluForm) -> Sources {
    let one = matches!(alu::wide_operand_source(form), Source::One);
    Sources {
        src0: true,
        src1: one,
        src2: !one,
    }
}

/// The sources of `DADD` in `form`: 0 and 2.
const SOURCES_0_2: Sources = Sources {
    src0: true,
    src1: false,
    src2: true,
};

// ---------------------------------------------------------------------------------------------
// HADD2.

/// `HADD2 dst, a, b`: `a + b` on each half.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpHAdd2 {
    /// The destination.
    pub dst: Gpr,
    /// `a`, a register, and `b`, in the wide slot.
    pub srcs: [HalfSrc; 2],
    /// Clamp each result to [0, 1].
    pub saturate: bool,
    /// `.F32`: the result as a 32-bit float.
    pub f32: bool,
    /// Flush denormal inputs and results to zero.
    pub ftz: bool,
}

impl Named for OpHAdd2 {
    const MNEMONIC: &'static str = "hadd2";
}

impl Display for OpHAdd2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("hadd2")?;
        flag(f, self.f32, ".f32")?;
        flag(f, self.ftz, ".ftz")?;
        flag(f, self.saturate, ".sat")?;
        let [a, b] = self.srcs;
        write!(f, " {}, {a}, {b}", self.dst)
    }
}

fn decode_hadd2(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let a = alu::half_src0(reader)?;
    let b = alu::half_src(reader, form, alu::wide_operand_source(form))?;
    Ok(Op::HAdd2(OpHAdd2 {
        dst: reader.gpr(DST),
        srcs: [a, b],
        saturate: reader.flag(SAT),
        f32: reader.flag(HALF_F32),
        ftz: reader.flag(FTZ),
    }))
}

/// `HADD2` in `form`.
const fn hadd2(form: AluForm) -> Form {
    Form::new(
        form.opcode(HADD2_OPCODE),
        mask_of(&[DST, SAT, HALF_F32, FTZ]) | alu::half_known(form, wide_operand_sources(form)),
        decode_hadd2,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// HFMA2.

/// `HFMA2 dst, a, b, c`: `a * b + c` on each half, fused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpHFma2 {
    /// The destination.
    pub dst: Gpr,
    /// `a`, `b`, `c`. `a` is always a register; at most one of `b` and `c` is in the wide slot.
    pub srcs: [HalfSrc; 3],
    /// Clamp each result to [0, 1].
    pub saturate: bool,
    /// `.F32`: the result as a 32-bit float.
    pub f32: bool,
    /// Flush denormal inputs and results to zero.
    pub ftz: bool,
    /// NAK's `dnz` bit, printed `.fmz` as for `FFMA`.
    pub dnz: bool,
    /// `.RELU` (NAK writes it false).
    pub relu: bool,
    /// `.BF16_V2` (NAK writes it false).
    pub bf16: bool,
    /// Negate only `b`'s high half (PR #4588 alone; module docs).
    pub neg_high_b: bool,
}

impl Named for OpHFma2 {
    const MNEMONIC: &'static str = "hfma2";
}

impl Display for OpHFma2 {
    /// NAK's `test_hfma2`: `hfma2` and three register operands with their modifiers; the
    /// modifier names and order are guesses.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("hfma2")?;
        flag(f, self.f32, ".f32")?;
        flag(f, self.bf16, ".bf16_v2")?;
        flag(f, self.ftz, ".ftz")?;
        flag(f, self.dnz, ".fmz")?;
        flag(f, self.saturate, ".sat")?;
        flag(f, self.relu, ".relu")?;
        let [a, b, c] = self.srcs;
        write!(f, " {}, {a}, {b}", self.dst)?;
        flag(f, self.neg_high_b, ".h0_nh1")?;
        write!(f, ", {c}")
    }
}

fn decode_hfma2(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let a = alu::half_src0(reader)?;
    let b = alu::half_src(reader, form, Source::One)?;
    let c = alu::half_src(reader, form, Source::Two)?;
    let relu = reader.flag(RELU);
    let mode = reader.field(HFMA2_RELU_MODE);
    let expected = if relu { HFMA2_RELU_MODE_VALUE } else { 0 };
    if mode != expected {
        return Err(reader.unknown_value(HFMA2_RELU_MODE, mode));
    }
    Ok(Op::HFma2(OpHFma2 {
        dst: reader.gpr(DST),
        srcs: [a, b, c],
        saturate: reader.flag(SAT),
        f32: reader.flag(HALF_F32),
        ftz: reader.flag(FTZ),
        dnz: reader.flag(DNZ),
        relu,
        bf16: reader.flag(BF16_V2),
        neg_high_b: reader.flag(HFMA2_NEG_HIGH_B),
    }))
}

/// `HFMA2` in `form`.
const fn hfma2(form: AluForm) -> Form {
    Form::new(
        form.opcode(HFMA2_OPCODE),
        mask_of(&[
            DST,
            DNZ,
            SAT,
            HALF_F32,
            RELU,
            FTZ,
            BF16_V2,
            HFMA2_NEG_HIGH_B,
            HFMA2_RELU_MODE,
        ]) | alu::half_known(form, SOURCES_ALL),
        decode_hfma2,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// HMUL2.

/// `HMUL2 dst, a, b`: `a * b` on each half.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpHMul2 {
    /// The destination.
    pub dst: Gpr,
    /// `a`, a register, and `b`, source 1.
    pub srcs: [HalfSrc; 2],
    /// Clamp each result to [0, 1].
    pub saturate: bool,
    /// Flush denormal inputs and results to zero.
    pub ftz: bool,
    /// NAK's `dnz` bit, printed `.fmz`.
    pub dnz: bool,
    /// `.RELU` (NAK writes it false).
    pub relu: bool,
}

impl Named for OpHMul2 {
    const MNEMONIC: &'static str = "hmul2";
}

impl Display for OpHMul2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("hmul2")?;
        flag(f, self.ftz, ".ftz")?;
        flag(f, self.dnz, ".fmz")?;
        flag(f, self.saturate, ".sat")?;
        flag(f, self.relu, ".relu")?;
        let [a, b] = self.srcs;
        write!(f, " {}, {a}, {b}", self.dst)
    }
}

fn decode_hmul2(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let a = alu::half_src0(reader)?;
    let b = alu::half_src(reader, form, Source::One)?;
    Ok(Op::HMul2(OpHMul2 {
        dst: reader.gpr(DST),
        srcs: [a, b],
        saturate: reader.flag(SAT),
        ftz: reader.flag(FTZ),
        dnz: reader.flag(DNZ),
        relu: reader.flag(RELU),
    }))
}

/// `HMUL2` in `form`.
const fn hmul2(form: AluForm) -> Form {
    Form::new(
        form.opcode(HMUL2_OPCODE),
        mask_of(&[DST, DNZ, SAT, RELU, FTZ]) | alu::half_known(form, SOURCES_0_1),
        decode_hmul2,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// HSET2 and HSETP2.

/// `HSET2 dst, a, b, accum`: per half, 1.0 (with `.BF`) or an all-ones mask where the
/// comparison, combined with `accum`, is true, else zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpHSet2 {
    /// The destination.
    pub dst: Gpr,
    /// The comparison.
    pub cmp: FloatCmpOp,
    /// `a`, a register, and `b`, in the wide slot.
    pub srcs: [HalfSrc; 2],
    /// How the comparison combines with `accum`.
    pub set_op: PredSetOp,
    /// The accumulator (NAK: `set_pred_src(87..90, 90, &self.accum)`).
    pub accum: PredSrc,
    /// Flush denormal inputs to zero.
    pub ftz: bool,
    /// `.BF`: 1.0 for true (NAK always writes it).
    pub bf: bool,
}

impl Named for OpHSet2 {
    const MNEMONIC: &'static str = "hset2";
}

impl Display for OpHSet2 {
    /// As the float group's `fset.bf`: `.bf`, the comparison, `.ftz`, the combine operation,
    /// the operands and the accumulator.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("hset2")?;
        flag(f, self.bf, ".bf")?;
        write!(f, "{}", self.cmp)?;
        flag(f, self.ftz, ".ftz")?;
        let [a, b] = self.srcs;
        write!(f, "{} {}, {a}, {b}, {}", self.set_op, self.dst, self.accum)
    }
}

fn decode_hset2(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let set_op = read_pred_set_op(reader, HALF_SET_OP)?;
    let a = alu::half_src0(reader)?;
    let b = alu::half_src(reader, form, alu::wide_operand_source(form))?;
    Ok(Op::HSet2(OpHSet2 {
        dst: reader.gpr(DST),
        cmp: FloatCmpOp::from_bits(reader.field(CMP_OP)),
        srcs: [a, b],
        set_op,
        accum: reader.pred_src(COND, COND_NOT),
        ftz: reader.flag(FTZ),
        bf: reader.flag(HSET2_BF),
    }))
}

/// `HSET2` in `form`.
const fn hset2(form: AluForm) -> Form {
    Form::new(
        form.opcode(HSET2_OPCODE),
        mask_of(&[DST, HALF_SET_OP, HSET2_BF, CMP_OP, FTZ, COND, COND_NOT])
            | alu::half_known(form, wide_operand_sources(form)),
        decode_hset2,
    )
    .with_fixed(alu::fixed(form))
}

/// `HSETP2 dst, dst1, a, b, accum`: the comparison of the low halves combined with `accum` into
/// `dst` and of the high halves into `dst1`; with `.H_AND`, both combined into `dst` and its
/// negation into `dst1` (PR #4588's `Visitor::HSETP2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpHSetP2 {
    /// The first destination (NAK: `set_pred_dst(81..84, &self.dsts[0])`).
    pub dst: Pred,
    /// The second destination (NAK: `set_pred_dst(84..87, &self.dsts[1])`).
    pub dst1: Pred,
    /// The comparison.
    pub cmp: FloatCmpOp,
    /// How the comparison combines with `accum`.
    pub set_op: PredSetOp,
    /// `a`, a register, and `b`, in the wide slot.
    pub srcs: [HalfSrc; 2],
    /// The accumulator.
    pub accum: PredSrc,
    /// Flush denormal inputs to zero.
    pub ftz: bool,
    /// `.H_AND`: both halves' results combined.
    pub horizontal: bool,
}

impl Named for OpHSetP2 {
    const MNEMONIC: &'static str = "hsetp2";
}

impl Display for OpHSetP2 {
    /// As the float group's `fsetp`, with `.h_and` after the comparison.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "hsetp2{}", self.cmp)?;
        flag(f, self.horizontal, ".h_and")?;
        flag(f, self.ftz, ".ftz")?;
        let [a, b] = self.srcs;
        write!(
            f,
            "{} {}, {}, {a}, {b}, {}",
            self.set_op, self.dst, self.dst1, self.accum
        )
    }
}

fn decode_hsetp2(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let set_op = read_pred_set_op(reader, HALF_SET_OP)?;
    let a = alu::half_src0(reader)?;
    let b = alu::half_src(reader, form, alu::wide_operand_source(form))?;
    Ok(Op::HSetP2(OpHSetP2 {
        // Cannot fail for a three-bit field.
        dst: Pred::new(reader.byte(PRED_DST)).unwrap_or(Pred::PT),
        dst1: Pred::new(reader.byte(FSETP_DST1)).unwrap_or(Pred::PT),
        cmp: FloatCmpOp::from_bits(reader.field(CMP_OP)),
        set_op,
        srcs: [a, b],
        accum: reader.pred_src(COND, COND_NOT),
        ftz: reader.flag(FTZ),
        horizontal: reader.flag(HSETP2_H_AND),
    }))
}

/// `HSETP2` in `form`.
const fn hsetp2(form: AluForm) -> Form {
    Form::new(
        form.opcode(HSETP2_OPCODE),
        mask_of(&[
            HALF_SET_OP,
            HSETP2_H_AND,
            CMP_OP,
            FTZ,
            PRED_DST,
            FSETP_DST1,
            COND,
            COND_NOT,
        ]) | alu::half_known(form, wide_operand_sources(form)),
        decode_hsetp2,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// HMNMX2.

/// `HMNMX2 dst, a, b, min`: per half, the smaller of `a` and `b` where `min` is true, the
/// larger where it is false.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpHMnMx2 {
    /// The destination.
    pub dst: Gpr,
    /// `a`, a register, and `b`, source 1.
    pub srcs: [HalfSrc; 2],
    /// Picks the minimum (NAK: `set_pred_src(87..90, 90, &self.min)`).
    pub min: PredSrc,
    /// Flush denormal inputs to zero.
    pub ftz: bool,
    /// `.F32` (NAK writes it false).
    pub f32: bool,
    /// `.NAN` (NAK writes it false).
    pub nan: bool,
    /// `.XORSIGN` (NAK writes it false).
    pub xorsign: bool,
    /// `.BF16_V2` (NAK writes it false).
    pub bf16: bool,
}

impl Named for OpHMnMx2 {
    const MNEMONIC: &'static str = "hmnmx2";
}

impl Display for OpHMnMx2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("hmnmx2")?;
        flag(f, self.f32, ".f32")?;
        flag(f, self.bf16, ".bf16_v2")?;
        flag(f, self.ftz, ".ftz")?;
        flag(f, self.nan, ".nan")?;
        flag(f, self.xorsign, ".xorsign")?;
        let [a, b] = self.srcs;
        write!(f, " {}, {a}, {b}, {}", self.dst, self.min)
    }
}

fn decode_hmnmx2(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let a = alu::half_src0(reader)?;
    let b = alu::half_src(reader, form, Source::One)?;
    Ok(Op::HMnMx2(OpHMnMx2 {
        dst: reader.gpr(DST),
        srcs: [a, b],
        min: reader.pred_src(COND, COND_NOT),
        ftz: reader.flag(FTZ),
        f32: reader.flag(HALF_F32),
        nan: reader.flag(HMNMX2_NAN),
        xorsign: reader.flag(HMNMX2_XORSIGN),
        bf16: reader.flag(BF16_V2),
    }))
}

/// `HMNMX2` in `form`.
const fn hmnmx2(form: AluForm) -> Form {
    Form::new(
        form.opcode(HMNMX2_OPCODE),
        mask_of(&[
            DST,
            HALF_F32,
            FTZ,
            HMNMX2_NAN,
            HMNMX2_XORSIGN,
            BF16_V2,
            COND,
            COND_NOT,
        ]) | alu::half_known(form, SOURCES_0_1),
        decode_hmnmx2,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// DADD, DMUL and DFMA.

/// `DADD dst, a, b`: `a + b` in 64-bit float, each a register pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpDAdd {
    /// The destination.
    pub dst: Gpr,
    /// `a`, a register, and `b`, source 2: the register slot or the wide slot.
    pub srcs: [AluSrc; 2],
    /// The rounding mode.
    pub rnd: RoundMode,
}

impl Named for OpDAdd {
    const MNEMONIC: &'static str = "dadd";
}

impl Display for OpDAdd {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let [a, b] = self.srcs.map(IntOperand);
        write!(f, "dadd{} {}, {a}, {b}", self.rnd, self.dst)
    }
}

/// `DADD`'s operand forms: its second operand in source 2 (module docs).
pub(crate) const DADD_FORMS: [AluForm; 4] = [
    AluForm::Reg,
    AluForm::ImmSrc2,
    AluForm::CBufSrc2,
    AluForm::UniformSrc2,
];

fn decode_dadd(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::DAdd(OpDAdd {
        dst: reader.gpr(DST),
        srcs: [
            alu::src0(reader, true),
            alu::src(reader, form, Source::Two, true),
        ],
        rnd: RoundMode::from_bits(reader.field(RND)),
    }))
}

/// `DADD` in `form`.
const fn dadd(form: AluForm) -> Form {
    Form::new(
        form.opcode(DADD_OPCODE),
        mask_of(&[DST, RND]) | alu::known(form, SOURCES_0_2, true),
        decode_dadd,
    )
    .with_fixed(alu::fixed(form))
}

/// `DMUL dst, a, b`: `a * b` in 64-bit float.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpDMul {
    /// The destination.
    pub dst: Gpr,
    /// `a`, a register, and `b`, source 1.
    pub srcs: [AluSrc; 2],
    /// The rounding mode.
    pub rnd: RoundMode,
}

impl Named for OpDMul {
    const MNEMONIC: &'static str = "dmul";
}

impl Display for OpDMul {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let [a, b] = self.srcs.map(IntOperand);
        write!(f, "dmul{} {}, {a}, {b}", self.rnd, self.dst)
    }
}

fn decode_dmul(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::DMul(OpDMul {
        dst: reader.gpr(DST),
        srcs: [
            alu::src0(reader, true),
            alu::src(reader, form, Source::One, true),
        ],
        rnd: RoundMode::from_bits(reader.field(RND)),
    }))
}

/// `DMUL` in `form`.
const fn dmul(form: AluForm) -> Form {
    Form::new(
        form.opcode(DMUL_OPCODE),
        mask_of(&[DST, RND]) | alu::known(form, SOURCES_0_1, true),
        decode_dmul,
    )
    .with_fixed(alu::fixed(form))
}

/// `DFMA dst, a, b, c`: `a * b + c` in 64-bit float, fused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpDFma {
    /// The destination.
    pub dst: Gpr,
    /// `a`, `b`, `c`. `a` is always a register; at most one of `b` and `c` is in the wide slot.
    pub srcs: [AluSrc; 3],
    /// The rounding mode.
    pub rnd: RoundMode,
}

impl Named for OpDFma {
    const MNEMONIC: &'static str = "dfma";
}

impl Display for OpDFma {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let [a, b, c] = self.srcs.map(IntOperand);
        write!(f, "dfma{} {}, {a}, {b}, {c}", self.rnd, self.dst)
    }
}

fn decode_dfma(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::DFma(OpDFma {
        dst: reader.gpr(DST),
        srcs: [
            alu::src0(reader, true),
            alu::src(reader, form, Source::One, true),
            alu::src(reader, form, Source::Two, true),
        ],
        rnd: RoundMode::from_bits(reader.field(RND)),
    }))
}

/// `DFMA` in `form`.
const fn dfma(form: AluForm) -> Form {
    Form::new(
        form.opcode(DFMA_OPCODE),
        mask_of(&[DST, RND]) | alu::known(form, SOURCES_ALL, true),
        decode_dfma,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// DSETP.

/// `DSETP dst, dst1, a, b, accum`: the 64-bit comparison combined with `accum` into `dst`, and
/// its negation combined with `accum` into `dst1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpDSetP {
    /// The destination (NAK: `set_pred_dst(81..84, &self.dst)`); `PT` writes nothing.
    pub dst: Pred,
    /// The second destination (NAK: `set_pred_dst(84..87, &Dst::None)`, so `PT`).
    pub dst1: Pred,
    /// The comparison.
    pub cmp: FloatCmpOp,
    /// How the comparison combines with `accum`.
    pub set_op: PredSetOp,
    /// `a`, a register, and `b`, in the wide slot.
    pub srcs: [AluSrc; 2],
    /// The accumulator.
    pub accum: PredSrc,
}

impl Named for OpDSetP {
    const MNEMONIC: &'static str = "dsetp";
}

impl Display for OpDSetP {
    /// As the float group's `fsetp`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let [a, b] = self.srcs.map(IntOperand);
        write!(
            f,
            "dsetp{}{} {}, {}, {a}, {b}, {}",
            self.cmp, self.set_op, self.dst, self.dst1, self.accum
        )
    }
}

fn decode_dsetp(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let set_op = read_pred_set_op(reader, SET_OP)?;
    Ok(Op::DSetP(OpDSetP {
        // Cannot fail for a three-bit field.
        dst: Pred::new(reader.byte(PRED_DST)).unwrap_or(Pred::PT),
        dst1: Pred::new(reader.byte(FSETP_DST1)).unwrap_or(Pred::PT),
        cmp: FloatCmpOp::from_bits(reader.field(CMP_OP)),
        set_op,
        srcs: [
            alu::src0(reader, true),
            alu::src(reader, form, alu::wide_operand_source(form), true),
        ],
        accum: reader.pred_src(COND, COND_NOT),
    }))
}

/// `DSETP` in `form`.
const fn dsetp(form: AluForm) -> Form {
    Form::new(
        form.opcode(DSETP_OPCODE),
        mask_of(&[SET_OP, CMP_OP, PRED_DST, FSETP_DST1, COND, COND_NOT])
            | alu::known(form, wide_operand_sources(form), true),
        decode_dsetp,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// The form table.

/// The second-operand-in-the-wide-slot forms, by index.
const WIDE_1: AluForm = alu::WIDE_OPERAND_FORMS[0];
const WIDE_2: AluForm = alu::WIDE_OPERAND_FORMS[1];
const WIDE_3: AluForm = alu::WIDE_OPERAND_FORMS[2];
const WIDE_7: AluForm = alu::WIDE_OPERAND_FORMS[3];
/// The source-1 forms, by index.
const FORM_1: AluForm = SOURCE_1_FORMS[0];
const FORM_4: AluForm = SOURCE_1_FORMS[1];
const FORM_5: AluForm = SOURCE_1_FORMS[2];
const FORM_6: AluForm = SOURCE_1_FORMS[3];

/// The half and double group's forms.
pub(crate) const FORMS: &[Form] = &[
    hadd2(WIDE_1),
    hadd2(WIDE_2),
    hadd2(WIDE_3),
    hadd2(WIDE_7),
    hfma2(AluForm::ALL[0]),
    hfma2(AluForm::ALL[1]),
    hfma2(AluForm::ALL[2]),
    hfma2(AluForm::ALL[3]),
    hfma2(AluForm::ALL[4]),
    hfma2(AluForm::ALL[5]),
    hfma2(AluForm::ALL[6]),
    hmul2(FORM_1),
    hmul2(FORM_4),
    hmul2(FORM_5),
    hmul2(FORM_6),
    hset2(WIDE_1),
    hset2(WIDE_2),
    hset2(WIDE_3),
    hset2(WIDE_7),
    hsetp2(WIDE_1),
    hsetp2(WIDE_2),
    hsetp2(WIDE_3),
    hsetp2(WIDE_7),
    hmnmx2(FORM_1),
    hmnmx2(FORM_4),
    hmnmx2(FORM_5),
    hmnmx2(FORM_6),
    dadd(DADD_FORMS[0]),
    dadd(DADD_FORMS[1]),
    dadd(DADD_FORMS[2]),
    dadd(DADD_FORMS[3]),
    dmul(FORM_1),
    dmul(FORM_4),
    dmul(FORM_5),
    dmul(FORM_6),
    dfma(AluForm::ALL[0]),
    dfma(AluForm::ALL[1]),
    dfma(AluForm::ALL[2]),
    dfma(AluForm::ALL[3]),
    dfma(AluForm::ALL[4]),
    dfma(AluForm::ALL[5]),
    dfma(AluForm::ALL[6]),
    dsetp(WIDE_1),
    dsetp(WIDE_2),
    dsetp(WIDE_3),
    dsetp(WIDE_7),
];

// ---------------------------------------------------------------------------------------------
// Test hooks.

/// The packed-half register slot: the register and its swizzle and modifier bits, which the
/// two-source instructions that keep their second operand in the wide slot do not use.
#[cfg(test)]
const HALF_REGISTER_SLOT: u128 = mask_of(&[SRC2, SRC2_HALF_SWIZZLE, SRC2_HALF_ABS, SRC2_HALF_NEG]);
/// NAK's `.BF16_V2` of `HSET2` and `HSETP2`, marked "SM90+" (`set_bit(65, false)`).
#[cfg(test)]
const SET_BF16_SM90: Field = Field::bit(65);
/// The bits PR #4588's `HADD2` rows mark known that the forms reject: the unused register slot
/// and NAK's SM 9.0 `.BF16_V2` (bit 85).
#[cfg(test)]
pub(crate) const HADD2_DROPPED: u128 = HALF_REGISTER_SLOT | BF16_V2.mask();
/// The bits PR #4588's `HMUL2` rows mark known that the forms reject: NAK's SM 7.x `.F32`
/// (`set_bit(78, false); // .F32 (SM70-SM75)`) and its SM 9.0 `.BF16_V2` (85).
#[cfg(test)]
pub(crate) const HMUL2_DROPPED: u128 = mask_of(&[HALF_F32, BF16_V2]);
/// The bits PR #4588's `HSETP2` rows mark known that the forms reject: the part of the unused
/// register slot (64..72) that holds no field of `HSETP2` (69..72 hold the combine operation and
/// `.H_AND`), bit 65 included.
#[cfg(test)]
pub(crate) const HSETP2_DROPPED: u128 = SRC2.mask() & !mask_of(&[HALF_SET_OP, HSETP2_H_AND]);
/// The bits PR #4588's `HSET2` rows mark known that the forms reject: as for `HSETP2`, and the
/// register slot's swizzle and modifier bits (81..85), which `HSET2` has no field in.
#[cfg(test)]
pub(crate) const HSET2_DROPPED: u128 =
    HSETP2_DROPPED | mask_of(&[SRC2_HALF_SWIZZLE, SRC2_HALF_ABS, SRC2_HALF_NEG]);
/// The bits PR #4588's `DSETP` rows mark known that the forms reject: the unused register slot.
#[cfg(test)]
pub(crate) const DSETP_DROPPED: u128 = SRC2.mask();

// Bit 65 is inside the dropped register slot of `HSET2` and `HSETP2`.
#[cfg(test)]
const _: () = assert!(HSETP2_DROPPED & SET_BF16_SM90.mask() != 0);

/// PR #4588's rows for the group's forms, in [`FORMS`] order
/// (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97; each
/// row's four words, low word first), with the bits each form drops or requires (module docs).
#[cfg(test)]
pub(crate) const REFERENCE_ROWS: &[ReferenceRow] = &[
    // HADD2.
    ReferenceRow::new(0x230, [0xffff_ffff, 0xf000_00ff, 0x003f_6fff, 0xffff_fe00])
        .dropping(HADD2_DROPPED),
    ReferenceRow::new(0x430, [0xffff_ffff, 0xffff_ffff, 0x003f_6fff, 0xffff_fe00])
        .dropping(HADD2_DROPPED),
    ReferenceRow::new(0x630, [0xffff_ffff, 0xf7ff_ffff, 0x083f_6fff, 0xffff_fe00])
        .dropping(HADD2_DROPPED),
    ReferenceRow::new(0xe30, [0xffff_ffff, 0xf000_003f, 0x083f_6fff, 0xffff_fe00])
        .dropping(HADD2_DROPPED)
        .fixing(SLOT_UNIFORM.mask()),
    // HFMA2.
    ReferenceRow::new(0x231, [0xffff_ffff, 0xf000_00ff, 0x03ff_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x431, [0xffff_ffff, 0xffff_ffff, 0x03ff_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x631, [0xffff_ffff, 0xf7ff_ffff, 0x0bff_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x831, [0xffff_ffff, 0xffff_ffff, 0x03ff_ffff, 0xffff_fe00]),
    ReferenceRow::new(0xa31, [0xffff_ffff, 0xf7ff_ffff, 0x0bff_ffff, 0xffff_fe00]),
    ReferenceRow::new(0xc31, [0xffff_ffff, 0xf000_003f, 0x0bff_ffff, 0xffff_fe00])
        .fixing(SLOT_UNIFORM.mask()),
    ReferenceRow::new(0xe31, [0xffff_ffff, 0xf000_003f, 0x0bff_ffff, 0xffff_fe00])
        .fixing(SLOT_UNIFORM.mask()),
    // HMUL2.
    ReferenceRow::new(0x232, [0xffff_ffff, 0xf000_00ff, 0x0021_ff00, 0xffff_fe00])
        .dropping(HMUL2_DROPPED),
    ReferenceRow::new(0x832, [0xffff_ffff, 0xffff_ffff, 0x0021_ff00, 0xffff_fe00])
        .dropping(HMUL2_DROPPED),
    ReferenceRow::new(0xa32, [0xffff_ffff, 0xf7ff_ffff, 0x0821_ff00, 0xffff_fe00])
        .dropping(HMUL2_DROPPED),
    ReferenceRow::new(0xc32, [0xffff_ffff, 0xf000_003f, 0x0821_ff00, 0xffff_fe00])
        .dropping(HMUL2_DROPPED)
        .fixing(SLOT_UNIFORM.mask()),
    // HSET2.
    ReferenceRow::new(0x233, [0xffff_ffff, 0xf000_00ff, 0x079f_ffff, 0xffff_fe00])
        .dropping(HSET2_DROPPED),
    ReferenceRow::new(0x433, [0xffff_ffff, 0xffff_ffff, 0x079f_ffff, 0xffff_fe00])
        .dropping(HSET2_DROPPED),
    ReferenceRow::new(0x633, [0xffff_ffff, 0xf7ff_ffff, 0x0f9f_ffff, 0xffff_fe00])
        .dropping(HSET2_DROPPED),
    ReferenceRow::new(0xe33, [0xffff_ffff, 0xf000_003f, 0x0f9f_ffff, 0xffff_fe00])
        .dropping(HSET2_DROPPED)
        .fixing(SLOT_UNIFORM.mask()),
    // HSETP2.
    ReferenceRow::new(0x234, [0xff00_ffff, 0xf000_00ff, 0x07ff_ffff, 0xffff_fe00])
        .dropping(HSETP2_DROPPED),
    ReferenceRow::new(0x434, [0xff00_ffff, 0xffff_ffff, 0x07ff_ffff, 0xffff_fe00])
        .dropping(HSETP2_DROPPED),
    ReferenceRow::new(0x634, [0xff00_ffff, 0xf7ff_ffff, 0x0fff_ffff, 0xffff_fe00])
        .dropping(HSETP2_DROPPED),
    ReferenceRow::new(0xe34, [0xff00_ffff, 0xf000_003f, 0x0fff_ffff, 0xffff_fe00])
        .dropping(HSETP2_DROPPED)
        .fixing(SLOT_UNIFORM.mask()),
    // HMNMX2.
    ReferenceRow::new(0x240, [0xffff_ffff, 0xf000_00ff, 0x07a7_4f00, 0xffff_fe00]),
    ReferenceRow::new(0x840, [0xffff_ffff, 0xffff_ffff, 0x07a7_4f00, 0xffff_fe00]),
    ReferenceRow::new(0xa40, [0xffff_ffff, 0xf7ff_ffff, 0x0fa7_4f00, 0xffff_fe00]),
    ReferenceRow::new(0xc40, [0xffff_ffff, 0xf000_003f, 0x0fa7_4f00, 0xffff_fe00])
        .fixing(SLOT_UNIFORM.mask()),
    // DADD.
    ReferenceRow::new(0x229, [0xffff_ffff, 0x0000_0000, 0x0000_cfff, 0xffff_fe00]),
    ReferenceRow::new(0x429, [0xffff_ffff, 0xffff_ffff, 0x0000_c300, 0xffff_fe00]),
    ReferenceRow::new(0x629, [0xffff_ffff, 0xc7ff_ffff, 0x0800_c300, 0xffff_fe00]),
    ReferenceRow::new(0xe29, [0xffff_ffff, 0xc000_003f, 0x0800_c300, 0xffff_fe00])
        .fixing(SLOT_UNIFORM.mask()),
    // DMUL.
    ReferenceRow::new(0x228, [0xffff_ffff, 0xc000_00ff, 0x0000_c300, 0xffff_fe00]),
    ReferenceRow::new(0x828, [0xffff_ffff, 0xffff_ffff, 0x0000_c300, 0xffff_fe00]),
    ReferenceRow::new(0xa28, [0xffff_ffff, 0xc7ff_ffff, 0x0800_c300, 0xffff_fe00]),
    ReferenceRow::new(0xc28, [0xffff_ffff, 0xc000_003f, 0x0800_c300, 0xffff_fe00])
        .fixing(SLOT_UNIFORM.mask()),
    // DFMA.
    ReferenceRow::new(0x22b, [0xffff_ffff, 0xc000_00ff, 0x0000_cfff, 0xffff_fe00]),
    ReferenceRow::new(0x42b, [0xffff_ffff, 0xffff_ffff, 0x0000_cfff, 0xffff_fe00]),
    ReferenceRow::new(0x62b, [0xffff_ffff, 0xc7ff_ffff, 0x0800_cfff, 0xffff_fe00]),
    ReferenceRow::new(0x82b, [0xffff_ffff, 0xffff_ffff, 0x0000_cfff, 0xffff_fe00]),
    ReferenceRow::new(0xa2b, [0xffff_ffff, 0xc7ff_ffff, 0x0800_cfff, 0xffff_fe00]),
    ReferenceRow::new(0xc2b, [0xffff_ffff, 0xc000_003f, 0x0800_cfff, 0xffff_fe00])
        .fixing(SLOT_UNIFORM.mask()),
    ReferenceRow::new(0xe2b, [0xffff_ffff, 0xc000_003f, 0x0800_cfff, 0xffff_fe00])
        .fixing(SLOT_UNIFORM.mask()),
    // DSETP.
    ReferenceRow::new(0x22a, [0xff00_ffff, 0xc000_00ff, 0x07fe_ffff, 0xffff_fe00])
        .dropping(DSETP_DROPPED),
    ReferenceRow::new(0x42a, [0xff00_ffff, 0xffff_ffff, 0x07fe_ffff, 0xffff_fe00])
        .dropping(DSETP_DROPPED),
    ReferenceRow::new(0x62a, [0xff00_ffff, 0xc7ff_ffff, 0x0ffe_ffff, 0xffff_fe00])
        .dropping(DSETP_DROPPED),
    ReferenceRow::new(0xe2a, [0xff00_ffff, 0xc000_003f, 0x0ffe_ffff, 0xffff_fe00])
        .dropping(DSETP_DROPPED)
        .fixing(SLOT_UNIFORM.mask()),
];

/// Generators of the group's decoded operations, one per operation (`generate`).
#[cfg(test)]
pub(crate) fn any_ops() -> Vec<BoxedStrategy<Op>> {
    generate::all()
}

#[cfg(test)]
mod generate;

shells! {}
