//! Slice 6a's encoders: `MOV` (`impl SM70Op for OpMov`: `encode_alu(0x002, dst, None, src,
//! None)`, then `set_field(72..76, quad_lanes)`).

use super::{Encode, EncodeError, Writer, no_encoding};
use crate::decode::integer::{MOV_OPCODE, MOV_QUAD_LANES};
use crate::decode::{
    DST, Named, OpBMsk, OpBRev, OpFlo, OpIAbs, OpIAdd3, OpIAdd3X, OpIDp4, OpIMad, OpIMad64,
    OpIMnMx, OpISetP, OpLea, OpLeaX, OpLop3, OpMov, OpP2R, OpPLop3, OpPopC, OpPrmt, OpR2P, OpSel,
    OpSgxt, OpShf,
};

impl Encode for OpMov {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.alu(
            Self::MNEMONIC,
            MOV_OPCODE,
            [None, Some(self.src.to_alu()), None],
            false,
        )?;
        writer.gpr(DST, self.dst);
        writer.checked(MOV_QUAD_LANES, "quad lane mask", u64::from(self.quad_lanes))
    }
}

no_encoding!(
    OpIAdd3, OpIAdd3X, OpIMad, OpIMad64, OpLop3, OpShf, OpLea, OpLeaX, OpISetP, OpIMnMx, OpPrmt,
    OpPopC, OpFlo, OpBRev, OpSgxt, OpBMsk, OpIAbs, OpIDp4, OpSel, OpPLop3, OpP2R, OpR2P,
);
