//! Slice 6h's encoders: NAK's `impl SM70Op` for `OpBra`, `OpExit`, `OpKill`, `OpBSSy`,
//! `OpBSync`, `OpBreak`, `OpBMov` (and `OpBClear`, the same bits), `OpWarpSync`, `OpBar`,
//! `OpVote`, `OpShfl`, `OpMatch`, `OpRedux`, `OpNop` and `OpNanosleep`, and the inverse of the
//! decoder for the forms that come from PR #4588 alone (`BRX`, `CALL`, `RET`, `DEPBAR`; see
//! `decode::op::control`). `JMX` and `BPT` are shells.

use super::{Encode, EncodeError, Writer, no_encoding};
use crate::decode::control::{
    BAR_DEFER_BLOCKING, BAR_MODE, BAR_OPCODE, BAR_RED, BAR_SRC, BMOV_BARRIER, BMOV_CLEAR,
    BMOV_FROM_BARRIER_OPCODE, BMOV_SRC, BMOV_TO_BARRIER_OPCODE, BRA_OPCODE, BRA_U,
    BRA_UNIFORM_OPCODE, BRA_UPRED, BRA_UPRED_NOT, BRANCH_OFFSET, BREAK_OPCODE, BRX_OPCODE, BRX_REG,
    BRX_UNIFORM_REG, BRXU_OPCODE, BSSY_OFFSET, BSSY_OPCODE, BSYNC_OPCODE, CALL_NO_INC, CALL_OPCODE,
    COND, COND_NOT, CONV_BARRIER, DEPBAR_COUNT, DEPBAR_LE, DEPBAR_OPCODE, DEPBAR_SCOREBOARD,
    EXIT_KEEP_REF_COUNT, EXIT_NO_AT_EXIT, EXIT_OPCODE, KILL_OPCODE, MATCH_ANY, MATCH_OPCODE,
    MATCH_SRC, MATCH_U64, NANOSLEEP_CLEAR, NANOSLEEP_OPCODE, NANOSLEEP_RAND, NANOSLEEP_WARP,
    NOP_OPCODE, PRED_DST, REDUX_OP, REDUX_OPCODE, REDUX_SIGNED, REDUX_SRC, RET_NO_DEC, RET_OPCODE,
    RET_REG, SHFL_C_IMM, SHFL_C_REG, SHFL_IMM_BOTH_OPCODE, SHFL_IMM_C_OPCODE, SHFL_IMM_LANE_OPCODE,
    SHFL_LANE_IMM, SHFL_LANE_REG, SHFL_OP, SHFL_OPCODE, SHFL_SRC, UNIFORM_BIT, UNIFORM_DST,
    VOTE_OP, VOTE_OPCODE, VOTEU_OPCODE, WARPSYNC_OPCODE,
};
use crate::decode::{
    BMovDir, Barrier, BraCond, BranchOffset, BrxSrc, DST, Field, MatchOp, Named, OpBMov, OpBSSy,
    OpBSync, OpBar, OpBpt, OpBra, OpBreak, OpBrx, OpCall, OpDepBar, OpExit, OpJmx, OpKill, OpMatch,
    OpNanosleep, OpNop, OpRedux, OpRet, OpShfl, OpVote, OpWarpSync, Pred, PredSrc, ReduxOp,
    ShflSrc, VoteDst,
};

/// Writes a branch offset into `field`, which must hold it.
fn offset(writer: &mut Writer, field: Field, offset: BranchOffset) -> Result<(), EncodeError> {
    writer.signed(field, "branch offset", offset.words())
}

/// Writes a barrier register (`set_bar_reg`).
fn barrier(writer: &mut Writer, field: Field, bar: Barrier) {
    writer.set(field, u64::from(bar.index()));
}

/// Writes the predicate at bits 87..91 (`set_pred_src(87..90, 90, ..)`).
fn cond(writer: &mut Writer, cond: PredSrc) {
    writer.pred_src(COND, COND_NOT, cond);
}

impl Encode for OpBra {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        match self.cond {
            BraCond::Pred(pred) => {
                writer.opcode(BRA_OPCODE);
                cond(writer, pred);
            }
            BraCond::UniformPred(upred) => {
                writer.opcode(BRA_UNIFORM_OPCODE);
                writer.uniform_pred_src(BRA_UPRED, BRA_UPRED_NOT, upred);
                // The form's fixed bits: `set_field(87..90, 0x7)`, `set_bit(91, true)`.
                writer.set(COND, u64::from(Pred::PT.index()));
                writer.flag(UNIFORM_BIT, true);
            }
        }
        writer.flag(BRA_U, self.uniform);
        offset(writer, BRANCH_OFFSET, self.offset)
    }
}

impl Encode for OpBrx {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        match self.src {
            BrxSrc::Reg(reg) => {
                writer.opcode(BRX_OPCODE);
                writer.gpr(BRX_REG, reg);
            }
            BrxSrc::UniformReg(reg) => {
                writer.opcode(BRXU_OPCODE);
                writer.uniform_reg(BRX_UNIFORM_REG, reg);
                writer.flag(UNIFORM_BIT, true);
            }
        }
        cond(writer, self.cond);
        offset(writer, BRANCH_OFFSET, self.offset)
    }
}

impl Encode for OpCall {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(CALL_OPCODE);
        writer.flag(CALL_NO_INC, self.no_inc);
        cond(writer, self.cond);
        offset(writer, BRANCH_OFFSET, self.offset)
    }
}

impl Encode for OpRet {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(RET_OPCODE);
        writer.gpr(RET_REG, self.src);
        writer.flag(RET_NO_DEC, self.no_dec);
        cond(writer, self.cond);
        offset(writer, BRANCH_OFFSET, self.offset)
    }
}

impl Encode for OpExit {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(EXIT_OPCODE);
        writer.flag(EXIT_KEEP_REF_COUNT, self.keep_ref_count);
        writer.flag(EXIT_NO_AT_EXIT, self.no_at_exit);
        cond(writer, self.pred);
        Ok(())
    }
}

impl Encode for OpKill {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(KILL_OPCODE);
        cond(writer, self.pred);
        Ok(())
    }
}

impl Encode for OpBSSy {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(BSSY_OPCODE);
        barrier(writer, CONV_BARRIER, self.bar);
        cond(writer, self.cond);
        offset(writer, BSSY_OFFSET, self.offset)
    }
}

impl Encode for OpBSync {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(BSYNC_OPCODE);
        barrier(writer, CONV_BARRIER, self.bar);
        cond(writer, self.cond);
        Ok(())
    }
}

impl Encode for OpBreak {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(BREAK_OPCODE);
        barrier(writer, CONV_BARRIER, self.bar);
        cond(writer, self.cond);
        Ok(())
    }
}

impl Encode for OpBMov {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        match self.dir {
            BMovDir::ToBarrier { bar, src } => {
                writer.opcode(BMOV_TO_BARRIER_OPCODE);
                barrier(writer, BMOV_BARRIER, bar);
                writer.gpr(BMOV_SRC, src);
            }
            BMovDir::FromBarrier { dst, bar } => {
                writer.opcode(BMOV_FROM_BARRIER_OPCODE);
                writer.gpr(DST, dst);
                barrier(writer, BMOV_BARRIER, bar);
            }
        }
        writer.flag(BMOV_CLEAR, self.clear);
        Ok(())
    }
}

impl Encode for OpWarpSync {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.alu(
            Self::MNEMONIC,
            WARPSYNC_OPCODE,
            [None, Some(self.mask), None],
            false,
        )?;
        cond(writer, self.cond);
        Ok(())
    }
}

impl Encode for OpNanosleep {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.alu(
            Self::MNEMONIC,
            NANOSLEEP_OPCODE,
            [None, Some(self.time), None],
            false,
        )?;
        writer.flag(NANOSLEEP_CLEAR, self.clear);
        writer.flag(NANOSLEEP_WARP, self.warp);
        writer.flag(NANOSLEEP_RAND, self.rand);
        cond(writer, self.cond);
        Ok(())
    }
}

impl Encode for OpBar {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(BAR_OPCODE);
        writer.set(BAR_MODE, self.mode.bits());
        writer.set(BAR_RED, self.red.bits());
        writer.gpr(BAR_SRC, self.src);
        writer.flag(BAR_DEFER_BLOCKING, self.defer_blocking);
        cond(writer, self.pred);
        Ok(())
    }
}

impl Encode for OpVote {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        match self.dst {
            VoteDst::Reg { ballot, vote } => {
                writer.opcode(VOTE_OPCODE);
                writer.gpr(DST, ballot);
                writer.set(PRED_DST, u64::from(vote.index()));
            }
            VoteDst::Uniform { ballot, vote } => {
                writer.opcode(VOTEU_OPCODE);
                writer.uniform_reg(UNIFORM_DST, ballot);
                writer.set(PRED_DST, u64::from(vote.index()));
            }
        }
        writer.set(VOTE_OP, self.op.bits());
        cond(writer, self.pred);
        Ok(())
    }
}

impl Encode for OpShfl {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let opcode = match (self.lane, self.c) {
            (ShflSrc::Reg(lane), ShflSrc::Reg(c)) => {
                writer.gpr(SHFL_LANE_REG, lane);
                writer.gpr(SHFL_C_REG, c);
                SHFL_OPCODE
            }
            (ShflSrc::Reg(lane), ShflSrc::Imm(c)) => {
                writer.gpr(SHFL_LANE_REG, lane);
                writer.checked(SHFL_C_IMM, "shuffle clamp", u64::from(c))?;
                SHFL_IMM_C_OPCODE
            }
            (ShflSrc::Imm(lane), ShflSrc::Reg(c)) => {
                writer.checked(SHFL_LANE_IMM, "shuffle lane", u64::from(lane))?;
                writer.gpr(SHFL_C_REG, c);
                SHFL_IMM_LANE_OPCODE
            }
            (ShflSrc::Imm(lane), ShflSrc::Imm(c)) => {
                writer.checked(SHFL_C_IMM, "shuffle clamp", u64::from(c))?;
                writer.checked(SHFL_LANE_IMM, "shuffle lane", u64::from(lane))?;
                SHFL_IMM_BOTH_OPCODE
            }
        };
        writer.opcode(opcode);
        writer.gpr(DST, self.dst);
        writer.set(PRED_DST, u64::from(self.in_bounds.index()));
        writer.gpr(SHFL_SRC, self.src);
        writer.set(SHFL_OP, self.op.bits());
        Ok(())
    }
}

impl Encode for OpMatch {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(MATCH_OPCODE);
        writer.gpr(DST, self.mask);
        writer.gpr(MATCH_SRC, self.src);
        writer.flag(MATCH_U64, self.u64);
        writer.flag(MATCH_ANY, self.op == MatchOp::Any);
        writer.set(PRED_DST, u64::from(self.pred.index()));
        Ok(())
    }
}

impl Encode for OpRedux {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(REDUX_OPCODE);
        writer.uniform_reg(UNIFORM_DST, self.dst);
        writer.gpr(REDUX_SRC, self.src);
        let signed = match self.op {
            ReduxOp::Min { signed } | ReduxOp::Max { signed } => signed,
            ReduxOp::And | ReduxOp::Or | ReduxOp::Xor | ReduxOp::Sum => false,
        };
        writer.flag(REDUX_SIGNED, signed);
        writer.set(REDUX_OP, self.op.bits());
        Ok(())
    }
}

impl Encode for OpNop {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(NOP_OPCODE);
        Ok(())
    }
}

impl Encode for OpDepBar {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(DEPBAR_OPCODE);
        writer.checked(
            DEPBAR_SCOREBOARD,
            "dependency scoreboard",
            u64::from(self.scoreboard),
        )?;
        writer.flag(DEPBAR_LE, self.le);
        writer.checked(DEPBAR_COUNT, "dependency count", u64::from(self.count))
    }
}

no_encoding!(OpJmx, OpBpt);
