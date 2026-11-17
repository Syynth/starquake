//! Slice 6g's encoders: `ALD`, `AST` and `S2R` (`impl SM70Op for OpALd`, `OpASt` and `OpS2R`).

use super::{Encode, EncodeError, Writer, no_encoding};
use crate::decode::attribute::{
    ALD_OPCODE, ALD_OUTPUT, ALD_VTX, AST_DATA, AST_OPCODE, AST_VTX, ATTR_ADDR, ATTR_COMPS,
    ATTR_OFFSET, ATTR_PATCH, ATTR_PHYS, S2R_OPCODE, S2R_SR,
};
use crate::decode::{
    DST, Gpr, OpAL2P, OpALd, OpASt, OpCS2R, OpIpa, OpIsberd, OpIsbewr, OpLdTram, OpOut, OpOutFinal,
    OpPixLd, OpS2R,
};

/// The address and width fields `ALD` and `AST` share.
fn attr(
    writer: &mut Writer,
    offset: Gpr,
    addr: u16,
    comps: u8,
    patch: bool,
    phys: bool,
) -> Result<(), EncodeError> {
    if !(1..=4).contains(&comps) {
        return Err(EncodeError::Comps(comps));
    }
    writer.checked(ATTR_ADDR, "attribute address", u64::from(addr))?;
    writer.gpr(ATTR_OFFSET, offset);
    writer.set(ATTR_COMPS, u64::from(comps - 1));
    writer.flag(ATTR_PATCH, patch);
    writer.flag(ATTR_PHYS, phys);
    Ok(())
}

impl Encode for OpALd {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        attr(
            writer,
            self.offset,
            self.addr,
            self.comps,
            self.patch,
            self.phys,
        )?;
        writer.opcode(ALD_OPCODE);
        writer.gpr(DST, self.dst);
        writer.gpr(ALD_VTX, self.vtx);
        writer.flag(ALD_OUTPUT, self.output);
        Ok(())
    }
}

impl Encode for OpASt {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        attr(
            writer,
            self.offset,
            self.addr,
            self.comps,
            self.patch,
            self.phys,
        )?;
        writer.opcode(AST_OPCODE);
        writer.gpr(AST_DATA, self.data);
        writer.gpr(AST_VTX, self.vtx);
        Ok(())
    }
}

impl Encode for OpS2R {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(S2R_OPCODE);
        writer.gpr(DST, self.dst);
        writer.set(S2R_SR, u64::from(self.sr.0));
        Ok(())
    }
}

no_encoding!(
    OpIpa, OpAL2P, OpCS2R, OpPixLd, OpOut, OpOutFinal, OpIsberd, OpIsbewr, OpLdTram,
);
