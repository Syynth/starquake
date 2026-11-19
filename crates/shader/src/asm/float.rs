//! Slice 6b's encoders: NAK's `impl SM70Op` for `OpFAdd`, `OpFFma`, `OpFMnMx`, `OpFMul`,
//! `OpFSet`, `OpFSetP`, `OpMuFu`, `OpFSwzAdd`, `OpFRnd`, `OpF2F`, `OpF2I`, `OpI2F` and `OpF2FP`
//! (`mesa-nouveau/src/nouveau/compiler/nak/sm70_encode.rs` @ 52238762, MIT, Copyright 2022
//! Collabora, Ltd.), with the fields NAK writes as constants taken from the op, and the inverse
//! of the decoder for the instructions and fields that come from PR #4588 alone (`FSEL`,
//! `I2FP`; see `decode::op::float`).
//!
//! A field the operation's form has no bits for is refused ([`EncodeError::NoBits`]) rather
//! than dropped, so that every value the assembler accepts decodes back to itself.

use super::{Encode, EncodeError, Writer};
use crate::decode::control::{COND, COND_NOT, PRED_DST};
use crate::decode::float::{
    CMP_OP, CVT_DST_SIZE, CVT_SRC_SIZE, DNZ, F2F_OPCODE, F2F_WIDE_OPCODE, F2FP_C, F2FP_H1,
    F2FP_MERGE_C, F2FP_OPCODE, F2FP_RND, F2I_NTZ, F2I_OPCODE, F2I_SIGNED, F2I_WIDE_OPCODE,
    FADD_OPCODE, FFMA_OPCODE, FMNMX_NAN, FMNMX_OPCODE, FMUL_OPCODE, FMUL_SCALE, FRND_OPCODE,
    FRND_WIDE_OPCODE, FSEL_OPCODE, FSET_OPCODE, FSETP_DST1, FSETP_OPCODE, FSWZADD_NDV,
    FSWZADD_OPCODE, FSWZADD_SRC1, FTZ, HIGH_HALF, I2F_OPCODE, I2F_SIGNED, I2F_WIDE_OPCODE,
    I2FP_OPCODE, MUFU_F16, MUFU_OP, MUFU_OPCODE, RND, SAT, SET_OP, fswzadd_op, i2f_select,
};
use crate::decode::{
    AluSrc, DST, Field, FloatType, Gpr, IntType, Named, OpF2F, OpF2FP, OpF2I, OpFAdd, OpFFma,
    OpFMnMx, OpFMul, OpFRnd, OpFSel, OpFSet, OpFSetP, OpFSwzAdd, OpI2F, OpI2FP, OpMuFu, PredSetOp,
    RoundMode, SRC0, SRC2,
};

/// Writes a rounding mode at [`RND`].
fn rnd(writer: &mut Writer, rnd: RoundMode) {
    writer.set(RND, rnd.bits());
}

/// The 9-bit opcode of a conversion: `wide` picks the 64-bit one.
const fn conversion_opcode(narrow: u16, wide_opcode: u16, wide: bool) -> u16 {
    if wide { wide_opcode } else { narrow }
}

/// The refusal of `what`, which the operation's form has no bits for.
const fn no_bits(mnemonic: &'static str, what: &'static str) -> EncodeError {
    EncodeError::NoBits { mnemonic, what }
}

/// Writes a float type into a size field.
fn float_type(writer: &mut Writer, field: Field, ty: FloatType) {
    writer.set(field, ty.bits());
}

/// Writes an integer type into a size field and a signed bit.
fn int_type(writer: &mut Writer, size: Field, signed: Field, ty: IntType) {
    writer.set(size, ty.size_bits());
    writer.flag(signed, ty.is_signed());
}

/// Sets the high-half select of a 16-bit source `src`, refusing it where the form has no bit
/// for it: a source that is not 16-bit (`half` false) or an immediate, whose form 4 has bit 60
/// in the immediate (NAK writes the swizzle from `encode_alu_reg`, `encode_alu_cb` and
/// `encode_alu_ureg` only). A clear select writes nothing.
fn high_half(
    writer: &mut Writer,
    mnemonic: &'static str,
    high: bool,
    half: bool,
    src: AluSrc,
) -> Result<(), EncodeError> {
    if high && !(half && !matches!(src, AluSrc::Imm(_))) {
        return Err(no_bits(
            mnemonic,
            "the high half of a source other than a 16-bit register, constant buffer or \
             uniform register",
        ));
    }
    if high {
        writer.flag(HIGH_HALF, true);
    }
    Ok(())
}

impl Encode for OpFFma {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.alu(Self::MNEMONIC, FFMA_OPCODE, self.srcs.map(Some), true)?;
        writer.gpr(DST, self.dst);
        writer.flag(DNZ, self.dnz);
        writer.flag(SAT, self.saturate);
        rnd(writer, self.rnd);
        writer.flag(FTZ, self.ftz);
        Ok(())
    }
}

impl Encode for OpFAdd {
    /// `encode_alu(0x021, dst, a, b, None)` for a register `b`, else `encode_alu(0x021, dst, a,
    /// ZERO, b)`; the register slot `b` leaves is [`OpFAdd::ignored`].
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        let sources = if matches!(b, AluSrc::Reg { .. }) {
            [Some(a), Some(b), None]
        } else {
            [Some(a), None, Some(b)]
        };
        writer.alu(Self::MNEMONIC, FADD_OPCODE, sources, true)?;
        writer.gpr(DST, self.dst);
        writer.gpr(SRC2, self.ignored);
        writer.flag(SAT, self.saturate);
        rnd(writer, self.rnd);
        writer.flag(FTZ, self.ftz);
        Ok(())
    }
}

impl Encode for OpFMul {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        writer.alu(Self::MNEMONIC, FMUL_OPCODE, [Some(a), Some(b), None], true)?;
        writer.gpr(DST, self.dst);
        writer.gpr(SRC2, self.ignored);
        writer.flag(DNZ, self.dnz);
        writer.flag(SAT, self.saturate);
        rnd(writer, self.rnd);
        writer.flag(FTZ, self.ftz);
        writer.set(FMUL_SCALE, self.scale.bits());
        Ok(())
    }
}

impl Encode for OpFMnMx {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        if self.nan && matches!(b, AluSrc::UniformReg { .. }) {
            return Err(no_bits(
                Self::MNEMONIC,
                ".nan with a uniform-register source",
            ));
        }
        writer.alu(Self::MNEMONIC, FMNMX_OPCODE, [Some(a), Some(b), None], true)?;
        writer.gpr(DST, self.dst);
        writer.pred_src(COND, COND_NOT, self.min);
        writer.flag(FTZ, self.ftz);
        writer.flag(FMNMX_NAN, self.nan);
        Ok(())
    }
}

impl Encode for OpFSet {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        if self.set_op != PredSetOp::And && matches!(b, AluSrc::UniformReg { .. }) {
            return Err(no_bits(
                Self::MNEMONIC,
                "a combine operation with a uniform-register source",
            ));
        }
        writer.alu(Self::MNEMONIC, FSET_OPCODE, [Some(a), Some(b), None], true)?;
        writer.gpr(DST, self.dst);
        writer.set(CMP_OP, self.cmp.bits());
        writer.set(SET_OP, self.set_op.bits());
        writer.pred_src(COND, COND_NOT, self.accum);
        writer.flag(FTZ, self.ftz);
        Ok(())
    }
}

impl Encode for OpFSetP {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        writer.alu(Self::MNEMONIC, FSETP_OPCODE, [Some(a), Some(b), None], true)?;
        writer.set(PRED_DST, u64::from(self.dst.index()));
        writer.set(FSETP_DST1, u64::from(self.dst1.index()));
        writer.set(SET_OP, self.set_op.bits());
        writer.set(CMP_OP, self.cmp.bits());
        writer.pred_src(COND, COND_NOT, self.accum);
        writer.flag(FTZ, self.ftz);
        Ok(())
    }
}

impl Encode for OpFSel {
    /// The inverse of PR #4588's `Visitor::SEL(true)` in the forms its table has rows for.
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        if matches!(b, AluSrc::UniformReg { .. }) {
            return Err(no_bits(Self::MNEMONIC, "a uniform-register source"));
        }
        writer.alu(Self::MNEMONIC, FSEL_OPCODE, [Some(a), Some(b), None], true)?;
        writer.gpr(DST, self.dst);
        writer.pred_src(COND, COND_NOT, self.cond);
        Ok(())
    }
}

impl Encode for OpMuFu {
    /// `encode_alu_base(0x108, dst, None, src, None, ..)`, then `set_bit(73, f16)` and
    /// `set_field(74..80, op)`; the swizzle of a 16-bit register source at bit 60.
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.alu(
            Self::MNEMONIC,
            MUFU_OPCODE,
            [None, Some(self.src), None],
            true,
        )?;
        high_half(writer, Self::MNEMONIC, self.high, self.f16, self.src)?;
        writer.gpr(DST, self.dst);
        writer.flag(MUFU_F16, self.f16);
        writer.set(MUFU_OP, self.op.bits());
        Ok(())
    }
}

impl Encode for OpFSwzAdd {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(FSWZADD_OPCODE);
        writer.gpr(DST, self.dst);
        let [a, b]: [Gpr; 2] = self.srcs;
        writer.gpr(SRC0, a);
        writer.gpr(FSWZADD_SRC1, b);
        for (lane, op) in (0..).zip(self.ops) {
            writer.set(fswzadd_op(lane), op.bits());
        }
        writer.flag(FSWZADD_NDV, self.ndv);
        rnd(writer, self.rnd);
        writer.flag(FTZ, self.ftz);
        Ok(())
    }
}

impl Encode for OpFRnd {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let opcode = conversion_opcode(FRND_OPCODE, FRND_WIDE_OPCODE, self.wide);
        writer.alu(Self::MNEMONIC, opcode, [None, Some(self.src), None], true)?;
        let half = self.src_type == FloatType::F16;
        high_half(writer, Self::MNEMONIC, self.high, half, self.src)?;
        writer.gpr(DST, self.dst);
        float_type(writer, CVT_SRC_SIZE, self.src_type);
        writer.flag(FTZ, self.ftz);
        rnd(writer, self.rnd);
        float_type(writer, CVT_DST_SIZE, self.dst_type);
        Ok(())
    }
}

impl Encode for OpF2F {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let opcode = conversion_opcode(F2F_OPCODE, F2F_WIDE_OPCODE, self.wide);
        writer.alu(Self::MNEMONIC, opcode, [None, Some(self.src), None], true)?;
        let half = self.src_type == FloatType::F16;
        high_half(writer, Self::MNEMONIC, self.high, half, self.src)?;
        writer.gpr(DST, self.dst);
        float_type(writer, CVT_DST_SIZE, self.dst_type);
        rnd(writer, self.rnd);
        writer.flag(FTZ, self.ftz);
        float_type(writer, CVT_SRC_SIZE, self.src_type);
        Ok(())
    }
}

impl Encode for OpF2I {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let opcode = conversion_opcode(F2I_OPCODE, F2I_WIDE_OPCODE, self.wide);
        writer.alu(Self::MNEMONIC, opcode, [None, Some(self.src), None], true)?;
        let half = self.src_type == FloatType::F16;
        high_half(writer, Self::MNEMONIC, self.high, half, self.src)?;
        writer.gpr(DST, self.dst);
        int_type(writer, CVT_DST_SIZE, F2I_SIGNED, self.dst_type);
        writer.flag(F2I_NTZ, self.ntz);
        rnd(writer, self.rnd);
        writer.flag(FTZ, self.ftz);
        float_type(writer, CVT_SRC_SIZE, self.src_type);
        Ok(())
    }
}

impl Encode for OpI2F {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let opcode = conversion_opcode(I2F_OPCODE, I2F_WIDE_OPCODE, self.wide);
        writer.alu(Self::MNEMONIC, opcode, [None, Some(self.src), None], false)?;
        if matches!(self.src, AluSrc::Imm(_)) {
            if self.select != 0 {
                return Err(no_bits(Self::MNEMONIC, "a byte select of an immediate"));
            }
        } else {
            writer.checked(
                i2f_select(self.src_type),
                "byte select",
                u64::from(self.select),
            )?;
        }
        writer.gpr(DST, self.dst);
        int_type(writer, CVT_SRC_SIZE, I2F_SIGNED, self.src_type);
        float_type(writer, CVT_DST_SIZE, self.dst_type);
        rnd(writer, self.rnd);
        Ok(())
    }
}

impl Encode for OpI2FP {
    /// The inverse of PR #4588's `Visitor::I2F` on `0x045`.
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.alu(
            Self::MNEMONIC,
            I2FP_OPCODE,
            [None, Some(self.src), None],
            false,
        )?;
        writer.gpr(DST, self.dst);
        int_type(writer, CVT_SRC_SIZE, I2F_SIGNED, self.src_type);
        float_type(writer, CVT_DST_SIZE, self.dst_type);
        rnd(writer, self.rnd);
        Ok(())
    }
}

impl Encode for OpF2FP {
    /// `encode_alu(0x03e, dst, a, b, ZERO)` with [`OpF2FP::c`] as source 2, then `.MERGE_C`
    /// (78), `.H1` (72, with `.MERGE_C` only) and the rounding mode at 79..81.
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        if self.h1 && !self.merge_c {
            return Err(no_bits(Self::MNEMONIC, ".h1 without .merge_c"));
        }
        let [a, b] = self.srcs;
        writer.alu(Self::MNEMONIC, F2FP_OPCODE, [Some(a), Some(b), None], false)?;
        writer.gpr(DST, self.dst);
        writer.gpr(F2FP_C, self.c);
        writer.flag(F2FP_MERGE_C, self.merge_c);
        writer.flag(F2FP_H1, self.h1);
        writer.set(F2FP_RND, self.rnd.bits());
        Ok(())
    }
}
