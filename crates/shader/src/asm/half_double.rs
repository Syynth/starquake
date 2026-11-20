//! Slice 6c's encoders: NAK's `impl SM70Op` for `OpHAdd2`, `OpHFma2`, `OpHMul2`, `OpHSet2`,
//! `OpHSetP2`, `OpHMnMx2`, `OpDAdd`, `OpDMul`, `OpDFma` and `OpDSetP`
//! (`mesa-nouveau/src/nouveau/compiler/nak/sm70_encode.rs` @ 52238762, MIT, Copyright 2022
//! Collabora, Ltd.), with the fields NAK writes as constants taken from the op, and the inverse
//! of the decoder for `HFMA2`'s bits from PR #4588 alone (`decode::op::half_double`).

use super::{Encode, EncodeError, Writer};
use crate::decode::control::{COND, COND_NOT, PRED_DST};
use crate::decode::float::{CMP_OP, DNZ, FSETP_DST1, FTZ, RND, SAT, SET_OP};
use crate::decode::half_double::{
    BF16_V2, DADD_OPCODE, DFMA_OPCODE, DMUL_OPCODE, DSETP_OPCODE, HADD2_OPCODE, HALF_F32,
    HALF_SET_OP, HFMA2_NEG_HIGH_B, HFMA2_OPCODE, HFMA2_RELU_MODE, HFMA2_RELU_MODE_VALUE,
    HMNMX2_NAN, HMNMX2_OPCODE, HMNMX2_XORSIGN, HMUL2_OPCODE, HSET2_BF, HSET2_OPCODE,
    HSETP2_H_AND, HSETP2_OPCODE, RELU,
};
use crate::decode::{
    AluSrc, DST, HalfSrc, Named, OpDAdd, OpDFma, OpDMul, OpDSetP, OpHAdd2, OpHFma2, OpHMnMx2,
    OpHMul2, OpHSet2, OpHSetP2,
};

/// The sources of an instruction that keeps its second operand `b` in the wide slot: as source
/// 1 when it is a register, else as source 2 (NAK's `src_is_zero_or_gpr` choice; the register
/// slot `b` leaves is not written).
fn wide_operand<T>(a: T, b: T, b_is_reg: bool) -> [Option<T>; 3] {
    if b_is_reg {
        [Some(a), Some(b), None]
    } else {
        [Some(a), None, Some(b)]
    }
}

/// [`wide_operand`] for packed-half sources.
fn half_wide_operand(srcs: [HalfSrc; 2]) -> [Option<HalfSrc>; 3] {
    let [a, b] = srcs;
    wide_operand(a, b, matches!(b.src, AluSrc::Reg { .. }))
}

impl Encode for OpHAdd2 {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.half_alu(Self::MNEMONIC, HADD2_OPCODE, half_wide_operand(self.srcs))?;
        writer.gpr(DST, self.dst);
        writer.flag(SAT, self.saturate);
        writer.flag(HALF_F32, self.f32);
        writer.flag(FTZ, self.ftz);
        Ok(())
    }
}

impl Encode for OpHFma2 {
    /// NAK's encoder, with `.RELU` and `.BF16_V2` from the op, and PR #4588's bit 86 and its
    /// bits 87..89, 7 with `.RELU`.
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.half_alu(Self::MNEMONIC, HFMA2_OPCODE, self.srcs.map(Some))?;
        writer.gpr(DST, self.dst);
        writer.flag(DNZ, self.dnz);
        writer.flag(SAT, self.saturate);
        writer.flag(HALF_F32, self.f32);
        writer.flag(RELU, self.relu);
        writer.flag(FTZ, self.ftz);
        writer.flag(BF16_V2, self.bf16);
        writer.flag(HFMA2_NEG_HIGH_B, self.neg_high_b);
        if self.relu {
            writer.set(HFMA2_RELU_MODE, HFMA2_RELU_MODE_VALUE);
        }
        Ok(())
    }
}

impl Encode for OpHMul2 {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        writer.half_alu(Self::MNEMONIC, HMUL2_OPCODE, [Some(a), Some(b), None])?;
        writer.gpr(DST, self.dst);
        writer.flag(DNZ, self.dnz);
        writer.flag(SAT, self.saturate);
        writer.flag(RELU, self.relu);
        writer.flag(FTZ, self.ftz);
        Ok(())
    }
}

impl Encode for OpHSet2 {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.half_alu(Self::MNEMONIC, HSET2_OPCODE, half_wide_operand(self.srcs))?;
        writer.gpr(DST, self.dst);
        writer.set(HALF_SET_OP, self.set_op.bits());
        writer.flag(HSET2_BF, self.bf);
        writer.set(CMP_OP, self.cmp.bits());
        writer.flag(FTZ, self.ftz);
        writer.pred_src(COND, COND_NOT, self.accum);
        Ok(())
    }
}

impl Encode for OpHSetP2 {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.half_alu(Self::MNEMONIC, HSETP2_OPCODE, half_wide_operand(self.srcs))?;
        writer.set(HALF_SET_OP, self.set_op.bits());
        writer.flag(HSETP2_H_AND, self.horizontal);
        writer.set(CMP_OP, self.cmp.bits());
        writer.flag(FTZ, self.ftz);
        writer.set(PRED_DST, u64::from(self.dst.index()));
        writer.set(FSETP_DST1, u64::from(self.dst1.index()));
        writer.pred_src(COND, COND_NOT, self.accum);
        Ok(())
    }
}

impl Encode for OpHMnMx2 {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        writer.half_alu(Self::MNEMONIC, HMNMX2_OPCODE, [Some(a), Some(b), None])?;
        writer.gpr(DST, self.dst);
        writer.flag(HALF_F32, self.f32);
        writer.flag(FTZ, self.ftz);
        writer.flag(HMNMX2_NAN, self.nan);
        writer.flag(HMNMX2_XORSIGN, self.xorsign);
        writer.flag(BF16_V2, self.bf16);
        writer.pred_src(COND, COND_NOT, self.min);
        Ok(())
    }
}

impl Encode for OpDAdd {
    /// `encode_alu(0x029, dst, a, None, b)`: `b` is always source 2.
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        writer.alu(Self::MNEMONIC, DADD_OPCODE, [Some(a), None, Some(b)], true)?;
        writer.gpr(DST, self.dst);
        writer.set(RND, self.rnd.bits());
        Ok(())
    }
}

impl Encode for OpDMul {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        writer.alu(Self::MNEMONIC, DMUL_OPCODE, [Some(a), Some(b), None], true)?;
        writer.gpr(DST, self.dst);
        writer.set(RND, self.rnd.bits());
        Ok(())
    }
}

impl Encode for OpDFma {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.alu(Self::MNEMONIC, DFMA_OPCODE, self.srcs.map(Some), true)?;
        writer.gpr(DST, self.dst);
        writer.set(RND, self.rnd.bits());
        Ok(())
    }
}

impl Encode for OpDSetP {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let [a, b] = self.srcs;
        let sources = wide_operand(a, b, matches!(b, AluSrc::Reg { .. }));
        writer.alu(Self::MNEMONIC, DSETP_OPCODE, sources, true)?;
        writer.set(SET_OP, self.set_op.bits());
        writer.set(CMP_OP, self.cmp.bits());
        writer.set(PRED_DST, u64::from(self.dst.index()));
        writer.set(FSETP_DST1, u64::from(self.dst1.index()));
        writer.pred_src(COND, COND_NOT, self.accum);
        Ok(())
    }
}
