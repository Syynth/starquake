//! Slice 6a's encoders: each operation's `impl SM70Op` in NAK's `sm70_encode.rs` @ 52238762
//! (MIT, Copyright 2022 Collabora, Ltd.), and PR #4588's field reads for `IMAD.HI`, `P2R`,
//! `R2P` and the fields only it reads (`decode::op::integer`'s module docs). Every ALU-form
//! operation goes through [`int_alu`], which checks the sources against the decoder's own
//! description of the instruction ([`Alu`]: its forms and which sources have a negation bit)
//! before `encode_alu` writes them; the operation's fields are written after, as NAK's encoders
//! do.

use super::{Encode, EncodeError, Writer, choose_form};
use crate::decode::control::{COND, COND_NOT, PRED_DST};
use crate::decode::integer::{
    Alu, BMSK_WRAP, FLO_SH, FLO_SIGNED, IADD3_X, IDP_2A, IDP_SIGNED_A, IDP_SIGNED_B, IMAD_SIGNED,
    IMAD_X, IMNMX_SIGNED, ISETP_CMP, ISETP_EX, ISETP_SET_OP, ISETP_SIGNED, LEA_HI,
    LEA_INTERMEDIATE, LEA_SHIFT, LEA_SX32, LEA_X, LOP3_LUT, MOV_QUAD_LANES, NOT_PT_INPUT,
    P2R_OPCODE, PLOP3_LUT0_HIGH, PLOP3_LUT0_LOW, PLOP3_LUT1, PLOP3_OPCODE, PLOP3_UNIFORM, PR_MASK,
    PRED_B, PRED_B_NOT, PRED_C, PRED_C_NOT, PRED_DST1, PRMT_MODE, R2P_BYTE, R2P_OPCODE,
    SGXT_SIGNED, SGXT_WRAP, SHF_HI, SHF_RIGHT, SHF_TYPE, SHF_WRAP, neg_field,
};
use crate::decode::{
    AluForm, AluSrc, DST, Field, IMadMode, Named, OPCODE, OpBMsk, OpBRev, OpFlo, OpIAbs, OpIAdd3,
    OpIAdd3X, OpIDp4, OpIMad, OpIMad64, OpIMnMx, OpISetP, OpLea, OpLeaX, OpLop3, OpMov, OpP2R,
    OpPLop3, OpPopC, OpPrmt, OpR2P, OpSel, OpSgxt, OpShf, PLop3Src, Pred, SRC0,
};

/// Encodes the sources of `kind` with `encode_alu` and sets the opcode: refuses a source with
/// an absolute value, or negated where `kind` has no negation bit for it, and a form `kind`
/// does not have. Returns the form.
fn int_alu(
    writer: &mut Writer,
    mnemonic: &'static str,
    kind: Alu,
    srcs: [Option<AluSrc>; 3],
) -> Result<AluForm, EncodeError> {
    let [_, one, two] = srcs;
    let form = choose_form(one, two).ok_or(EncodeError::TwoWideSources { mnemonic })?;
    for ((index, src), spec) in (0_u8..).zip(srcs).zip(kind.srcs()) {
        let Some(src) = src else { continue };
        let (abs, neg) = src.modifiers();
        if abs || (neg && neg_field(form, usize::from(index), spec).is_none()) {
            return Err(EncodeError::Modifier { mnemonic, index });
        }
    }
    if !kind.forms().contains(&form) {
        return Err(EncodeError::NoForm {
            mnemonic,
            form: form.bits(),
        });
    }
    writer.alu(mnemonic, kind.base(), srcs, true)?;
    Ok(form)
}

/// Sets a predicate destination.
fn pred(writer: &mut Writer, field: Field, pred: Pred) {
    writer.set(field, u64::from(pred.index()));
}

/// Sets every bit of `fixed`, a field every word has set.
fn fixed(writer: &mut Writer, fixed: u128) {
    writer.set(
        Field::new(fixed.trailing_zeros(), fixed.count_ones()),
        u64::MAX,
    );
}

impl Encode for OpMov {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        int_alu(
            writer,
            Self::MNEMONIC,
            Alu::Mov,
            [None, Some(self.src.to_alu()), None],
        )?;
        writer.gpr(DST, self.dst);
        writer.checked(MOV_QUAD_LANES, "quad lane mask", u64::from(self.quad_lanes))
    }
}

/// `IADD3` and `IADD3.X` (`OpIAdd3`: the carries `!PT` and no `.X`; `OpIAdd3X`: the carries and
/// `.X`).
fn encode_iadd3(
    writer: &mut Writer,
    mnemonic: &'static str,
    op: &OpIAdd3,
    x: bool,
) -> Result<(), EncodeError> {
    int_alu(writer, mnemonic, Alu::IAdd3, op.srcs.map(Some))?;
    writer.gpr(DST, op.dst);
    writer.pred_src(COND, COND_NOT, op.carry[0]);
    writer.pred_src(PRED_B, PRED_B_NOT, op.carry[1]);
    pred(writer, PRED_DST, op.overflow[0]);
    pred(writer, PRED_DST1, op.overflow[1]);
    writer.flag(IADD3_X, x);
    Ok(())
}

impl Encode for OpIAdd3 {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        encode_iadd3(writer, Self::MNEMONIC, self, false)
    }
}

impl Encode for OpIAdd3X {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        encode_iadd3(writer, Self::MNEMONIC, &self.add, true)
    }
}

/// `LEA` and `LEA.X` (`OpLea` and `OpLeaX`: the same fields but bit 74).
fn encode_lea(
    writer: &mut Writer,
    mnemonic: &'static str,
    op: &OpLea,
    x: bool,
) -> Result<(), EncodeError> {
    int_alu(
        writer,
        mnemonic,
        Alu::Lea,
        [Some(op.a), Some(op.b), Some(op.a_high)],
    )?;
    writer.gpr(DST, op.dst);
    writer.flag(LEA_INTERMEDIATE, op.intermediate);
    writer.checked(LEA_SHIFT, "shift", u64::from(op.shift))?;
    writer.flag(LEA_HI, op.dst_high);
    pred(writer, PRED_DST, op.overflow);
    writer.flag(LEA_X, x);
    writer.flag(LEA_SX32, op.sx32);
    writer.pred_src(COND, COND_NOT, op.carry);
    Ok(())
}

impl Encode for OpLea {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        encode_lea(writer, Self::MNEMONIC, self, false)
    }
}

impl Encode for OpLeaX {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        encode_lea(writer, Self::MNEMONIC, &self.lea, true)
    }
}

impl Encode for OpLop3 {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        int_alu(writer, Self::MNEMONIC, Alu::Lop3, self.srcs.map(Some))?;
        writer.gpr(DST, self.dst);
        fixed(writer, NOT_PT_INPUT);
        writer.set(LOP3_LUT, u64::from(self.lut));
        pred(writer, PRED_DST, self.pred);
        Ok(())
    }
}

impl Encode for OpIAbs {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        int_alu(
            writer,
            Self::MNEMONIC,
            Alu::IAbs,
            [None, Some(self.src), None],
        )?;
        writer.gpr(DST, self.dst);
        Ok(())
    }
}

impl Encode for OpBRev {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        int_alu(
            writer,
            Self::MNEMONIC,
            Alu::BRev,
            [None, Some(self.src), None],
        )?;
        writer.gpr(DST, self.dst);
        Ok(())
    }
}

impl Encode for OpPopC {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        int_alu(
            writer,
            Self::MNEMONIC,
            Alu::PopC,
            [None, Some(self.src), None],
        )?;
        writer.gpr(DST, self.dst);
        Ok(())
    }
}

impl Encode for OpFlo {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        int_alu(
            writer,
            Self::MNEMONIC,
            Alu::Flo,
            [None, Some(self.src), None],
        )?;
        writer.gpr(DST, self.dst);
        fixed(writer, PRED_DST.mask());
        writer.flag(FLO_SH, self.return_shift_amount);
        writer.flag(FLO_SIGNED, self.signed);
        Ok(())
    }
}

impl Encode for OpIMnMx {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        int_alu(writer, Self::MNEMONIC, Alu::IMnMx, [Some(a), Some(b), None])?;
        writer.gpr(DST, self.dst);
        writer.pred_src(COND, COND_NOT, self.min);
        writer.flag(IMNMX_SIGNED, self.signed);
        Ok(())
    }
}

impl Encode for OpSel {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        int_alu(writer, Self::MNEMONIC, Alu::Sel, [Some(a), Some(b), None])?;
        writer.gpr(DST, self.dst);
        writer.pred_src(COND, COND_NOT, self.cond);
        Ok(())
    }
}

impl Encode for OpSgxt {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        int_alu(
            writer,
            Self::MNEMONIC,
            Alu::Sgxt,
            [Some(self.a), Some(self.bits), None],
        )?;
        writer.gpr(DST, self.dst);
        writer.flag(SGXT_SIGNED, self.signed);
        writer.flag(SGXT_WRAP, self.wrap);
        Ok(())
    }
}

impl Encode for OpBMsk {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        int_alu(
            writer,
            Self::MNEMONIC,
            Alu::BMsk,
            [Some(self.pos), Some(self.width), None],
        )?;
        writer.gpr(DST, self.dst);
        writer.flag(BMSK_WRAP, self.wrap);
        Ok(())
    }
}

impl Encode for OpPrmt {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        int_alu(
            writer,
            Self::MNEMONIC,
            Alu::Prmt,
            [Some(a), Some(self.sel), Some(b)],
        )?;
        writer.gpr(DST, self.dst);
        writer.set(PRMT_MODE, self.mode.bits());
        Ok(())
    }
}

impl Encode for OpShf {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        int_alu(
            writer,
            Self::MNEMONIC,
            Alu::Shf,
            [Some(self.low), Some(self.shift), Some(self.high)],
        )?;
        writer.gpr(DST, self.dst);
        writer.set(SHF_TYPE, self.data_type.bits());
        writer.flag(SHF_WRAP, self.wrap);
        writer.flag(SHF_RIGHT, self.right);
        writer.flag(SHF_HI, self.dst_high);
        Ok(())
    }
}

impl Encode for OpIMad {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let kind = match self.mode {
            IMadMode::Lo { .. } => Alu::IMad,
            IMadMode::Hi => Alu::IMadHi,
        };
        int_alu(writer, Self::MNEMONIC, kind, self.srcs.map(Some))?;
        writer.gpr(DST, self.dst);
        fixed(writer, PRED_DST.mask());
        writer.flag(IMAD_SIGNED, self.signed);
        match self.mode {
            IMadMode::Lo { x, carry } => {
                writer.flag(IMAD_X, x);
                writer.pred_src(COND, COND_NOT, carry);
            }
            IMadMode::Hi => fixed(writer, NOT_PT_INPUT),
        }
        Ok(())
    }
}

impl Encode for OpIMad64 {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        int_alu(writer, Self::MNEMONIC, Alu::IMadWide, self.srcs.map(Some))?;
        writer.gpr(DST, self.dst);
        fixed(writer, PRED_DST.mask());
        writer.flag(IMAD_SIGNED, self.signed);
        writer.pred_src(COND, COND_NOT, self.carry);
        Ok(())
    }
}

impl Encode for OpIDp4 {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let form = int_alu(writer, Self::MNEMONIC, Alu::IDp, self.srcs.map(Some))?;
        if self.two_a && form != AluForm::Reg {
            return Err(EncodeError::NoForm {
                mnemonic: Self::MNEMONIC,
                form: form.bits(),
            });
        }
        writer.gpr(DST, self.dst);
        writer.flag(IDP_SIGNED_A, self.signed[0]);
        writer.flag(IDP_SIGNED_B, self.signed[1]);
        writer.flag(IDP_2A, self.two_a);
        Ok(())
    }
}

impl Encode for OpISetP {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        int_alu(writer, Self::MNEMONIC, Alu::ISetP, [Some(a), Some(b), None])?;
        writer.pred_src(PRED_C, PRED_C_NOT, self.low_cmp);
        writer.pred_src(COND, COND_NOT, self.accum);
        writer.flag(ISETP_EX, self.ex);
        writer.flag(ISETP_SIGNED, self.signed);
        writer.set(ISETP_SET_OP, self.set_op.bits());
        writer.set(ISETP_CMP, self.cmp_op.bits());
        pred(writer, PRED_DST, self.dsts[0]);
        pred(writer, PRED_DST1, self.dsts[1]);
        Ok(())
    }
}

impl Encode for OpPLop3 {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.set(OPCODE, u64::from(PLOP3_OPCODE));
        match self.c {
            PLop3Src::Pred(src) => writer.pred_src(PRED_C, PRED_C_NOT, src),
            PLop3Src::Uniform(src) => {
                writer.uniform_pred_src(PRED_C, PRED_C_NOT, src);
                writer.flag(PLOP3_UNIFORM, true);
            }
        }
        writer.pred_src(PRED_B, PRED_B_NOT, self.srcs[1]);
        writer.pred_src(COND, COND_NOT, self.srcs[0]);
        let [lut0, lut1] = self.luts;
        writer.set(PLOP3_LUT1, u64::from(lut1));
        writer.set(PLOP3_LUT0_LOW, u64::from(lut0));
        writer.set(PLOP3_LUT0_HIGH, u64::from(lut0 >> PLOP3_LUT0_LOW.width));
        pred(writer, PRED_DST, self.dsts[0]);
        pred(writer, PRED_DST1, self.dsts[1]);
        Ok(())
    }
}

impl Encode for OpP2R {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.set(OPCODE, u64::from(P2R_OPCODE));
        writer.gpr(DST, self.dst);
        writer.gpr(SRC0, self.src);
        writer.checked(PR_MASK, "predicate mask", u64::from(self.mask))
    }
}

impl Encode for OpR2P {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.set(OPCODE, u64::from(R2P_OPCODE));
        writer.gpr(SRC0, self.src);
        writer.checked(R2P_BYTE, "byte", u64::from(self.byte))?;
        writer.checked(PR_MASK, "predicate mask", u64::from(self.mask))
    }
}
