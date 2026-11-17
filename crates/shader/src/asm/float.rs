//! Slice 6b's encoders: `FFMA` (`impl SM70Op for OpFFma`: `encode_alu(0x023, dst, src0, src1,
//! src2)`, then `set_bit(76, dnz)`, `set_bit(77, saturate)`, `set_rnd_mode(78..80, ..)` and
//! `set_bit(80, ftz)`).

use super::{Encode, EncodeError, Writer, no_encoding};
use crate::decode::float::{FFMA_DNZ, FFMA_FTZ, FFMA_OPCODE, FFMA_RND, FFMA_SAT};
use crate::decode::{
    DST, Named, OpF2F, OpF2FP, OpF2I, OpFAdd, OpFFma, OpFMnMx, OpFMul, OpFRnd, OpFSel, OpFSet,
    OpFSetP, OpFSwzAdd, OpI2F, OpI2FP, OpMuFu,
};

impl Encode for OpFFma {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.alu(Self::MNEMONIC, FFMA_OPCODE, self.srcs.map(Some), true)?;
        writer.gpr(DST, self.dst);
        writer.flag(FFMA_DNZ, self.dnz);
        writer.flag(FFMA_SAT, self.saturate);
        writer.set(FFMA_RND, self.rnd.bits());
        writer.flag(FFMA_FTZ, self.ftz);
        Ok(())
    }
}

no_encoding!(
    OpFAdd, OpFMul, OpFMnMx, OpFSet, OpFSetP, OpFSel, OpMuFu, OpFSwzAdd, OpFRnd, OpF2F, OpF2I,
    OpI2F, OpI2FP, OpF2FP,
);
