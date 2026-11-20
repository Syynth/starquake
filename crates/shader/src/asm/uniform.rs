//! Slice 6d's encoders: NAK's `sm70_encode.rs` @ 52238762 (MIT, Copyright 2022 Collabora,
//! Ltd.) `impl SM70Op` for `OpR2UR`, `OpMov` (`0x882`, `0xc82`), `OpLdc` (`0xab9`), `OpLdcg`
//! (`0x8b8`) and `OpPLop3` (`0x89c`) with uniform operands, and `encode_ualu_base` for the
//! twins; PR #4588's field reads for the indexed `ULDC` (`0xabb`). A twin is written by its
//! vector op's encoder, then narrowed: the twin's opcode, bit 91, and each register field's
//! `RZ` (255) written as `URZ` (63) (`decode::op::uniform`'s module docs).

use super::{Encode, EncodeError, Writer, no_encoding};
use crate::decode::control::{COND, COND_NOT, PRED_DST, UNIFORM_BIT, UNIFORM_DST};
use crate::decode::integer::{
    PLOP3_LUT0_HIGH, PLOP3_LUT0_LOW, PLOP3_LUT1, PRED_B, PRED_B_NOT, PRED_C, PRED_C_NOT,
    PRED_DST1,
};
use crate::decode::memory::{LDC_HANDLE, MEM_TYPE};
use crate::decode::uniform::{
    LDCG_ADDR, LDCG_OFFSET, LDCG_OPCODE, R2UR_OPCODE, TwinOperands, ULDC_INDEX, ULDC_OPCODE,
    ULDC64_OPCODE, UMOV_IMM_OPCODE, UMOV_UR_OPCODE, UPLOP3_OPCODE, Twin,
};
use crate::decode::{
    ALU_FORM, AluForm, CBUF_INDEX, CBUF_OFFSET, CBuf, Gpr, Named, OPCODE, OpR2UR, OpULdc, OpUMov,
    OpUP2UR, OpUPLop3, SLOT_IMM, SLOT_UNIFORM, SLOT_UNIFORM_REG, SRC0, ULdcSrc, UMovSrc,
    Uniform, UniformReg,
};

impl Encode for OpR2UR {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(R2UR_OPCODE);
        writer.uniform_reg(UNIFORM_DST, self.dst);
        writer.gpr(SRC0, self.src);
        // `set_pred_dst(81..84, &Dst::None)`: `PT`.
        writer.set(PRED_DST, u64::MAX);
        Ok(())
    }
}

impl Encode for OpUMov {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.uniform_reg(UNIFORM_DST, self.dst);
        match self.src {
            UMovSrc::Imm(imm) => {
                writer.opcode(UMOV_IMM_OPCODE);
                writer.set(SLOT_IMM, u64::from(imm));
            }
            UMovSrc::Reg(reg) => {
                writer.opcode(UMOV_UR_OPCODE);
                writer.uniform_reg(SLOT_UNIFORM_REG, reg);
                writer.flag(SLOT_UNIFORM, true);
            }
        }
        Ok(())
    }
}

impl Encode for OpULdc {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.uniform_reg(UNIFORM_DST, self.dst);
        writer.set(MEM_TYPE, self.mem_type.bits());
        match self.src {
            ULdcSrc::CBuf(cbuf) => {
                writer.opcode(ULDC_OPCODE);
                match cbuf.buf {
                    CBuf::Binding(index) => {
                        writer.checked(CBUF_INDEX, "constant buffer index", u64::from(index))?;
                    }
                    CBuf::BindlessUniform(handle) => {
                        writer.uniform_reg(LDC_HANDLE, handle);
                        writer.flag(UNIFORM_BIT, true);
                    }
                }
                writer.set(CBUF_OFFSET, u64::from(cbuf.offset));
            }
            ULdcSrc::Indexed { index, reg, offset } => {
                writer.opcode(ULDC64_OPCODE);
                writer.checked(CBUF_INDEX, "constant buffer index", u64::from(index))?;
                writer.uniform_reg(ULDC_INDEX, reg);
                writer.set(CBUF_OFFSET, u64::from(offset));
                writer.flag(UNIFORM_BIT, true);
            }
            ULdcSrc::Global { addr, offset, pred } => {
                writer.opcode(LDCG_OPCODE);
                writer.uniform_reg(LDCG_ADDR, addr);
                writer.signed(LDCG_OFFSET, "global offset", i64::from(offset))?;
                writer.uniform_pred_src(COND, COND_NOT, pred);
                writer.flag(UNIFORM_BIT, true);
            }
        }
        Ok(())
    }
}

impl Encode for OpUPLop3 {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(UPLOP3_OPCODE);
        let [a, b, c] = self.srcs;
        writer.uniform_pred_src(COND, COND_NOT, a);
        writer.uniform_pred_src(PRED_B, PRED_B_NOT, b);
        writer.uniform_pred_src(PRED_C, PRED_C_NOT, c);
        let [lut0, lut1] = self.luts;
        writer.set(PRED_DST, u64::from(self.dsts[0].index()));
        writer.set(PRED_DST1, u64::from(self.dsts[1].index()));
        writer.set(PLOP3_LUT1, u64::from(lut1));
        writer.set(PLOP3_LUT0_LOW, u64::from(lut0));
        writer.set(PLOP3_LUT0_HIGH, u64::from(lut0 >> PLOP3_LUT0_LOW.width));
        Ok(())
    }
}

/// A twin: its vector op's encoding, then the twin's opcode, bit 91 and six-bit register
/// fields. Refuses a vector encoding in a form the twin does not have, and a register field
/// past `UR62` other than `RZ`; [`Uniform::new`] rules both out, so neither is reachable from
/// a value this crate made.
impl<T> Encode for Uniform<T>
where
    T: Encode + TwinOperands,
    Self: Named,
{
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        self.op().encode(writer)?;
        let word = writer.word();
        let opcode = word.field(OPCODE);
        let mnemonic = Self::MNEMONIC;
        let unencodable = EncodeError::NoBits {
            mnemonic,
            what: "its vector encoding",
        };
        let twin =
            Twin::of_vector_base(opcode & ((1 << ALU_FORM.start) - 1)).ok_or(unencodable)?;
        let form = AluForm::from_bits(opcode >> ALU_FORM.start).ok_or(unencodable)?;
        if !twin.forms().contains(&form) {
            return Err(EncodeError::NoForm {
                mnemonic,
                form: form.bits(),
            });
        }
        for field in twin.register_fields(form).into_iter().flatten() {
            let value = word.field(field);
            let narrowed = if value == u64::from(Gpr::RZ.index()) {
                u64::from(UniformReg::URZ.index())
            } else if value < u64::from(UniformReg::URZ.index()) {
                value
            } else {
                return Err(EncodeError::NoBits {
                    mnemonic,
                    what: "a register past ur62",
                });
            };
            // The whole eight-bit field: its top bits held RZ's.
            writer.set(field, narrowed);
        }
        writer.opcode(form.opcode(twin.base));
        writer.flag(SLOT_UNIFORM, true);
        Ok(())
    }
}

no_encoding!(OpUP2UR);
