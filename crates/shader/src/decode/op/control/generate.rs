//! Test generators of the control group's decoded operations, one per operation, every field
//! over its whole encodable range (`decode::tests` joins them through `form`'s `GROUP_HOOKS`).

use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;

use super::*;
use crate::testing::{any_cbuf, any_gpr, any_pred_src, any_uniform_reg};

/// Any offset `field` can hold.
fn any_offset(field: Field) -> impl Strategy<Value = BranchOffset> {
    let half = 1_i64 << (field.width - 1);
    (-half..half).prop_map(BranchOffset::from_words)
}

fn any_barrier() -> impl Strategy<Value = Barrier> {
    (0..Barrier::COUNT).prop_map(|index| Barrier::new(index).unwrap_or(Barrier(0)))
}

fn any_pred() -> impl Strategy<Value = Pred> {
    (0..=Pred::PT.index()).prop_map(|index| Pred::new(index).unwrap_or(Pred::PT))
}

fn any_uniform_pred() -> impl Strategy<Value = UniformPred> {
    (0..=UniformPred::UPT.index())
        .prop_map(|index| UniformPred::new(index).unwrap_or(UniformPred::UPT))
}

/// An unmodified source of forms 1, 4, 5 and 6.
fn any_source() -> impl Strategy<Value = AluSrc> {
    prop_oneof![
        any_gpr().prop_map(AluSrc::reg),
        any::<u32>().prop_map(AluSrc::Imm),
        any_cbuf().prop_map(|cbuf| AluSrc::CBuf {
            cbuf,
            abs: false,
            neg: false
        }),
        any_uniform_reg().prop_map(|reg| AluSrc::UniformReg {
            reg,
            abs: false,
            neg: false
        }),
    ]
}

/// A `SHFL` operand: a register, or an immediate of `imm`'s width.
fn any_shfl_src(imm: Field) -> impl Strategy<Value = ShflSrc> {
    prop_oneof![
        any_gpr().prop_map(ShflSrc::Reg),
        (0..(1_u16 << imm.width)).prop_map(ShflSrc::Imm),
    ]
}

/// `BRA`, `BRX`, `CALL`, `RET`, `EXIT` and `KILL`.
pub(super) fn branches() -> Vec<BoxedStrategy<Op>> {
    let cond = prop_oneof![
        any_pred_src().prop_map(BraCond::Pred),
        (any_uniform_pred(), any::<bool>())
            .prop_map(|(reg, negated)| BraCond::UniformPred(UniformPredSrc { reg, negated })),
    ];
    let bra =
        (cond, any::<bool>(), any_offset(BRANCH_OFFSET)).prop_map(|(cond, uniform, offset)| {
            Op::Bra(OpBra {
                cond,
                uniform,
                offset,
            })
        });
    let brx_src = prop_oneof![
        any_gpr().prop_map(BrxSrc::Reg),
        any_uniform_reg().prop_map(BrxSrc::UniformReg),
    ];
    let brx = (brx_src, any_offset(BRANCH_OFFSET), any_pred_src())
        .prop_map(|(src, offset, cond)| Op::Brx(OpBrx { src, offset, cond }));
    let call = (any_offset(BRANCH_OFFSET), any::<bool>(), any_pred_src()).prop_map(
        |(offset, no_inc, cond)| {
            Op::Call(OpCall {
                offset,
                no_inc,
                cond,
            })
        },
    );
    let ret = (
        any_gpr(),
        any_offset(BRANCH_OFFSET),
        any::<bool>(),
        any_pred_src(),
    )
        .prop_map(|(src, offset, no_dec, cond)| {
            Op::Ret(OpRet {
                src,
                offset,
                no_dec,
                cond,
            })
        });
    let exit = (any::<bool>(), any::<bool>(), any_pred_src()).prop_map(
        |(keep_ref_count, no_at_exit, pred)| {
            Op::Exit(OpExit {
                keep_ref_count,
                no_at_exit,
                pred,
            })
        },
    );
    let kill = any_pred_src().prop_map(|pred| Op::Kill(OpKill { pred }));
    vec![
        bra.boxed(),
        brx.boxed(),
        call.boxed(),
        ret.boxed(),
        exit.boxed(),
        kill.boxed(),
    ]
}

/// `BSSY`, `BSYNC`, `BREAK`, `BMOV`, `BAR`, `NOP` and `DEPBAR`.
pub(super) fn barriers() -> Vec<BoxedStrategy<Op>> {
    let bssy = (any_barrier(), any_offset(BSSY_OFFSET), any_pred_src())
        .prop_map(|(bar, offset, cond)| Op::BSSy(OpBSSy { bar, offset, cond }));
    let bsync =
        (any_barrier(), any_pred_src()).prop_map(|(bar, cond)| Op::BSync(OpBSync { bar, cond }));
    let brk =
        (any_barrier(), any_pred_src()).prop_map(|(bar, cond)| Op::Break(OpBreak { bar, cond }));
    let dir = prop_oneof![
        (any_barrier(), any_gpr()).prop_map(|(bar, src)| BMovDir::ToBarrier { bar, src }),
        (any_gpr(), any_barrier()).prop_map(|(dst, bar)| BMovDir::FromBarrier { dst, bar }),
    ];
    let bmov = (dir, any::<bool>()).prop_map(|(dir, clear)| Op::BMov(OpBMov { dir, clear }));
    let bar = (
        0..BarMode::ALL.len(),
        0..BarRed::ALL.len(),
        any_gpr(),
        any::<bool>(),
        any_pred_src(),
    )
        .prop_map(|(mode, red, src, defer_blocking, pred)| {
            Op::Bar(OpBar {
                mode: BarMode::ALL[mode],
                red: BarRed::ALL[red],
                src,
                defer_blocking,
                pred,
            })
        });
    let nop = Just(Op::Nop(OpNop {}));
    let depbar = (
        0..(1_u8 << DEPBAR_SCOREBOARD.width),
        any::<bool>(),
        0..(1_u8 << DEPBAR_COUNT.width),
    )
        .prop_map(|(scoreboard, le, count)| {
            Op::DepBar(OpDepBar {
                scoreboard,
                le,
                count,
            })
        });
    vec![
        bssy.boxed(),
        bsync.boxed(),
        brk.boxed(),
        bmov.boxed(),
        bar.boxed(),
        nop.boxed(),
        depbar.boxed(),
    ]
}

/// `WARPSYNC`, `VOTE`, `SHFL`, `MATCH`, `REDUX` and `NANOSLEEP`.
pub(super) fn warp() -> Vec<BoxedStrategy<Op>> {
    let warpsync = (any_source(), any_pred_src())
        .prop_map(|(mask, cond)| Op::WarpSync(OpWarpSync { mask, cond }));
    let vote_dst = prop_oneof![
        (any_gpr(), any_pred()).prop_map(|(ballot, vote)| VoteDst::Reg { ballot, vote }),
        (any_uniform_reg(), any_uniform_pred())
            .prop_map(|(ballot, vote)| VoteDst::Uniform { ballot, vote }),
    ];
    let vote = (0..VoteOp::ALL.len(), vote_dst, any_pred_src()).prop_map(|(op, dst, pred)| {
        Op::Vote(OpVote {
            op: VoteOp::ALL[op],
            dst,
            pred,
        })
    });
    let shfl = (
        0..ShflOp::ALL.len(),
        any_gpr(),
        any_pred(),
        any_gpr(),
        any_shfl_src(SHFL_LANE_IMM),
        any_shfl_src(SHFL_C_IMM),
    )
        .prop_map(|(op, dst, in_bounds, src, lane, c)| {
            Op::Shfl(OpShfl {
                op: ShflOp::ALL[op],
                dst,
                in_bounds,
                src,
                lane,
                c,
            })
        });
    let matcher = (
        any::<bool>(),
        any::<bool>(),
        any_gpr(),
        any_gpr(),
        any_pred(),
    )
        .prop_map(|(any, u64, mask, src, pred)| {
            Op::Match(OpMatch {
                op: if any { MatchOp::Any } else { MatchOp::All },
                u64,
                mask,
                src,
                pred,
            })
        });
    let redux_op = prop_oneof![
        Just(ReduxOp::And),
        Just(ReduxOp::Or),
        Just(ReduxOp::Xor),
        Just(ReduxOp::Sum),
        any::<bool>().prop_map(|signed| ReduxOp::Min { signed }),
        any::<bool>().prop_map(|signed| ReduxOp::Max { signed }),
    ];
    let redux = (redux_op, any_uniform_reg(), any_gpr())
        .prop_map(|(op, dst, src)| Op::Redux(OpRedux { op, dst, src }));
    let nanosleep = (any_source(), any::<[bool; 3]>(), any_pred_src()).prop_map(
        |(time, [clear, warp, rand], cond)| {
            Op::Nanosleep(OpNanosleep {
                time,
                clear,
                warp,
                rand,
                cond,
            })
        },
    );
    vec![
        warpsync.boxed(),
        vote.boxed(),
        shfl.boxed(),
        matcher.boxed(),
        redux.boxed(),
        nanosleep.boxed(),
    ]
}
