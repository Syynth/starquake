//! Slice 6g's encoders: NAK's `impl SM70Op` for `OpALd`, `OpASt`, `OpAL2P`, `OpIpa`,
//! `OpLdTram`, `OpS2R` (both forms), `OpCS2R`, `OpPixLd`, `OpOut`, `OpOutFinal`, `OpIsberd` and
//! `OpIsbewr` (`decode::op::attribute`). Where NAK writes a constant, the encoder writes the
//! op's field instead (the decoder's module docs list them).

use super::{Encode, EncodeError, Writer};
use crate::decode::attribute::{
    AL2P_OPCODE, ALD_OPCODE, ALD_VTX, AST_OPCODE, AST_VTX, ATTR_ADDR, ATTR_COMPS, ATTR_DATA,
    ATTR_OFFSET, ATTR_OUTPUT, ATTR_PATCH, ATTR_PHYS, ATTR_WORD, ATTR_WORD_BYTES, CS2R_64,
    CS2R_OPCODE, IPA_FREQ, IPA_LOC, IPA_OFFSET, IPA_OPCODE, ISBE_ACCESS, ISBE_IMM, ISBE_MEM,
    ISBE_SKEW, ISBERD_OPCODE, ISBEWR_OPCODE, LDTRAM_C, LDTRAM_FIXED, LDTRAM_OPCODE, LDTRAM_UREG,
    OUT_OPCODE, OUT_TYPE, PIXLD_OPCODE, PIXLD_VAL, S2R_OPCODE, S2UR_OPCODE, SR_INDEX,
};
use crate::decode::control::{PRED_DST, UNIFORM_DST};
use crate::decode::{
    AluSrc, DST, Gpr, IsbeAccess, IsbeMemType, Named, OpAL2P, OpALd, OpASt, OpCS2R, OpIpa,
    OpIsberd, OpIsbewr, OpLdTram, OpOut, OpOutFinal, OpPixLd, OpS2R, Pred, S2RDst, SpecialReg,
};

/// The address and width fields `ALD`, `AST` and `AL2P` share.
fn attr(writer: &mut Writer, offset: Gpr, addr: u16, comps: u8) -> Result<(), EncodeError> {
    if !(1..=4).contains(&comps) {
        return Err(EncodeError::Comps(comps));
    }
    writer.checked(ATTR_ADDR, "attribute address", u64::from(addr))?;
    writer.gpr(ATTR_OFFSET, offset);
    writer.set(ATTR_COMPS, u64::from(comps - 1));
    Ok(())
}

/// Writes a byte address into [`ATTR_WORD`] (`set_field(64..72, addr >> 2)` after
/// `assert!(self.addr % 4 == 0)`).
fn attr_word(writer: &mut Writer, addr: u16) -> Result<(), EncodeError> {
    if !addr.is_multiple_of(ATTR_WORD_BYTES) {
        return Err(EncodeError::AttrAlign {
            addr,
            align: ATTR_WORD_BYTES,
        });
    }
    writer.checked(
        ATTR_WORD,
        "attribute word",
        u64::from(addr / ATTR_WORD_BYTES),
    )
}

/// Writes a predicate destination at bits 81..84 (`set_pred_dst(81..84, ..)`).
fn pred_dst(writer: &mut Writer, pred: Pred) {
    writer.set(PRED_DST, u64::from(pred.index()));
}

/// Writes a special register index (`set_field(72..80, idx)`).
fn special_reg(writer: &mut Writer, sr: SpecialReg) {
    writer.set(SR_INDEX, u64::from(sr.0));
}

impl Encode for OpALd {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        attr(writer, self.offset, self.addr, self.comps)?;
        writer.opcode(ALD_OPCODE);
        writer.gpr(DST, self.dst);
        writer.gpr(ALD_VTX, self.vtx);
        writer.flag(ATTR_PATCH, self.patch);
        writer.flag(ATTR_PHYS, self.phys);
        writer.flag(ATTR_OUTPUT, self.output);
        Ok(())
    }
}

impl Encode for OpASt {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        attr(writer, self.offset, self.addr, self.comps)?;
        writer.opcode(AST_OPCODE);
        writer.gpr(ATTR_DATA, self.data);
        writer.gpr(AST_VTX, self.vtx);
        writer.flag(ATTR_PATCH, self.patch);
        writer.flag(ATTR_PHYS, self.phys);
        Ok(())
    }
}

impl Encode for OpAL2P {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        attr(writer, self.offset, self.addr, self.comps)?;
        writer.opcode(AL2P_OPCODE);
        writer.gpr(DST, self.dst);
        writer.flag(ATTR_OUTPUT, self.output);
        Ok(())
    }
}

impl Encode for OpIpa {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        attr_word(writer, self.addr)?;
        writer.opcode(IPA_OPCODE);
        writer.gpr(DST, self.dst);
        writer.set(IPA_LOC, self.loc.bits());
        writer.set(IPA_FREQ, self.freq.bits());
        writer.gpr(IPA_OFFSET, self.offset);
        pred_dst(writer, self.pred);
        Ok(())
    }
}

impl Encode for OpLdTram {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        attr_word(writer, self.addr)?;
        writer.opcode(LDTRAM_OPCODE);
        writer.gpr(DST, self.dst);
        writer.uniform_reg(LDTRAM_UREG, self.ureg);
        writer.flag(LDTRAM_C, self.use_c);
        writer.flag(LDTRAM_FIXED, true);
        Ok(())
    }
}

impl Encode for OpS2R {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        match self.dst {
            S2RDst::Reg(dst) => {
                writer.opcode(S2R_OPCODE);
                writer.gpr(DST, dst);
            }
            S2RDst::Uniform(dst) => {
                writer.opcode(S2UR_OPCODE);
                writer.uniform_reg(UNIFORM_DST, dst);
            }
        }
        special_reg(writer, self.sr);
        Ok(())
    }
}

impl Encode for OpCS2R {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(CS2R_OPCODE);
        writer.gpr(DST, self.dst);
        special_reg(writer, self.sr);
        writer.flag(CS2R_64, self.wide);
        Ok(())
    }
}

impl Encode for OpPixLd {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(PIXLD_OPCODE);
        writer.gpr(DST, self.dst);
        writer.set(PIXLD_VAL, self.val.bits());
        pred_dst(writer, self.pred);
        Ok(())
    }
}

/// The fields `OUT` and `OUT.FINAL` share (`encode_alu(0x124, dst, handle, stream, None)`).
fn out(
    writer: &mut Writer,
    mnemonic: &'static str,
    dst: Gpr,
    handle: Gpr,
    stream: AluSrc,
) -> Result<(), EncodeError> {
    writer.alu(
        mnemonic,
        OUT_OPCODE,
        [Some(AluSrc::reg(handle)), Some(stream), None],
        false,
    )?;
    writer.gpr(DST, dst);
    Ok(())
}

impl Encode for OpOut {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        out(writer, Self::MNEMONIC, self.dst, self.handle, self.stream)?;
        writer.set(OUT_TYPE, self.out_type.bits());
        Ok(())
    }
}

impl Encode for OpOutFinal {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        // The type field stays clear: `OUT.FINAL` is type 0.
        out(writer, Self::MNEMONIC, self.dst, self.handle, self.stream)
    }
}

/// The fields `ISBERD` and `ISBEWR` share.
fn isbe(
    writer: &mut Writer,
    offset: Gpr,
    imm_offset: u16,
    mem_type: IsbeMemType,
    access: IsbeAccess,
    output: bool,
    skew: bool,
) {
    writer.gpr(ATTR_OFFSET, offset);
    writer.set(ISBE_IMM, u64::from(imm_offset));
    writer.set(ISBE_MEM, mem_type.bits());
    writer.set(ISBE_ACCESS, access.bits());
    writer.flag(ISBE_SKEW, skew);
    writer.flag(ATTR_OUTPUT, output);
}

impl Encode for OpIsberd {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(ISBERD_OPCODE);
        writer.gpr(DST, self.dst);
        isbe(
            writer,
            self.offset,
            self.imm_offset,
            self.mem_type,
            self.access,
            self.output,
            self.skew,
        );
        Ok(())
    }
}

impl Encode for OpIsbewr {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        if !self.access.writable() {
            return Err(EncodeError::IsbewrAccess(self.access.bits()));
        }
        writer.opcode(ISBEWR_OPCODE);
        writer.gpr(ATTR_DATA, self.data);
        isbe(
            writer,
            self.offset,
            self.imm_offset,
            self.mem_type,
            self.access,
            self.output,
            self.skew,
        );
        Ok(())
    }
}
