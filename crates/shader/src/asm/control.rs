//! Slice 6h's encoders: `EXIT` (`impl SM70Op for OpExit`).

use super::{Encode, EncodeError, Writer, no_encoding};
use crate::decode::control::{
    EXIT_KEEP_REF_COUNT, EXIT_NO_AT_EXIT, EXIT_OPCODE, EXIT_PRED, EXIT_PRED_NOT,
};
use crate::decode::{
    OpBMov, OpBSSy, OpBSync, OpBar, OpBpt, OpBra, OpBreak, OpBrx, OpCall, OpDepBar, OpExit, OpJmx,
    OpKill, OpMatch, OpNanosleep, OpNop, OpRedux, OpRet, OpShfl, OpVote, OpWarpSync,
};

impl Encode for OpExit {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(EXIT_OPCODE);
        writer.flag(EXIT_KEEP_REF_COUNT, self.keep_ref_count);
        writer.flag(EXIT_NO_AT_EXIT, self.no_at_exit);
        writer.pred_src(EXIT_PRED, EXIT_PRED_NOT, self.pred);
        Ok(())
    }
}

no_encoding!(
    OpBra,
    OpBrx,
    OpJmx,
    OpCall,
    OpRet,
    OpKill,
    OpBSSy,
    OpBSync,
    OpBreak,
    OpBMov,
    OpWarpSync,
    OpBar,
    OpVote,
    OpShfl,
    OpMatch,
    OpRedux,
    OpNop,
    OpDepBar,
    OpNanosleep,
    OpBpt,
);
