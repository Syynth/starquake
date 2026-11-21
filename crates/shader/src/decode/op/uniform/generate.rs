//! Test generators of the uniform group's decoded operations, one per operation, every field
//! over its whole encodable range (`decode::tests` joins them through `form`'s `GROUP_HOOKS`).
//! A twin's generator is its vector operation's (`op::integer`'s), with each register mapped to
//! the six-bit range and each operation the uniform encoding cannot hold left out.

use proptest::prelude::*;
use proptest::sample::select;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

use super::super::integer;
use super::*;
use crate::testing::{any_cbuf, any_gpr, any_uniform_reg, config};

fn any_uniform_pred() -> impl Strategy<Value = UniformPred> {
    (0..=UniformPred::UPT.index())
        .prop_map(|index| UniformPred::new(index).unwrap_or(UniformPred::UPT))
}

fn any_uniform_pred_src() -> impl Strategy<Value = UniformPredSrc> {
    (any_uniform_pred(), any::<bool>()).prop_map(|(reg, negated)| UniformPredSrc { reg, negated })
}

/// `reg` as a twin's register: its number's low six bits, 63 (`URZ`) as [`Gpr::RZ`].
pub(crate) fn uniform_gpr(reg: Gpr) -> Gpr {
    let index = reg.index() % (1 << UniformReg::BITS);
    if index == UniformReg::URZ.index() {
        Gpr::RZ
    } else {
        Gpr::new(index)
    }
}

/// `src` with its register, if it is one, mapped by [`uniform_gpr`].
fn uniform_src(src: AluSrc) -> AluSrc {
    match src {
        AluSrc::Reg { reg, abs, neg } => AluSrc::Reg {
            reg: uniform_gpr(reg),
            abs,
            neg,
        },
        other => other,
    }
}

/// One generator per twin: each of the integer group's generators whose operation has a twin
/// (one value from each names its operation, as `each_group_generates_every_operation_it_decodes`
/// relies on), mapped to the twin.
fn twins() -> Vec<BoxedStrategy<Op>> {
    let mut runner = TestRunner::new(config());
    integer::any_ops()
        .into_iter()
        .filter_map(|strategy| {
            let sample = strategy.new_tree(&mut runner).ok()?.current();
            TWIN_MNEMONICS
                .iter()
                .any(|&(_, vector)| vector == sample.mnemonic())
                .then(|| {
                    strategy
                        .prop_filter_map("a vector operation with a twin", |op| {
                            twin_of(op, uniform_gpr, uniform_src)
                        })
                        .boxed()
                })
        })
        .collect()
}

fn any_r2ur() -> BoxedStrategy<Op> {
    (any_uniform_reg(), any_gpr())
        .prop_map(|(dst, src)| Op::R2UR(OpR2UR { dst, src }))
        .boxed()
}

fn any_umov() -> BoxedStrategy<Op> {
    let src = prop_oneof![
        any::<u32>().prop_map(UMovSrc::Imm),
        any_uniform_reg().prop_map(UMovSrc::Reg),
    ];
    (any_uniform_reg(), src)
        .prop_map(|(dst, src)| Op::UMov(OpUMov { dst, src }))
        .boxed()
}

fn any_uldc() -> BoxedStrategy<Op> {
    let src = prop_oneof![
        any_cbuf().prop_map(ULdcSrc::CBuf),
        (0..1_u8 << CBUF_INDEX.width, any_uniform_reg(), any::<u16>())
            .prop_map(|(index, reg, offset)| ULdcSrc::Indexed { index, reg, offset }),
        (any_uniform_reg(), any::<i32>(), any_uniform_pred_src())
            .prop_map(|(addr, offset, pred)| ULdcSrc::Global { addr, offset, pred }),
    ];
    (any_uniform_reg(), src, select(MemType::ALL))
        .prop_map(|(dst, src, mem_type)| Op::ULdc(OpULdc { dst, src, mem_type }))
        .boxed()
}

fn any_uplop3() -> BoxedStrategy<Op> {
    (
        [any_uniform_pred(), any_uniform_pred()],
        [
            any_uniform_pred_src(),
            any_uniform_pred_src(),
            any_uniform_pred_src(),
        ],
        any::<[u8; 2]>(),
    )
        .prop_map(|(dsts, srcs, luts)| Op::UPLop3(OpUPLop3 { dsts, srcs, luts }))
        .boxed()
}

/// One generator per operation the group decodes.
pub(super) fn ops() -> Vec<BoxedStrategy<Op>> {
    let mut ops = vec![any_r2ur(), any_umov(), any_uldc(), any_uplop3()];
    ops.extend(twins());
    ops
}
