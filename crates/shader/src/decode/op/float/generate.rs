//! Test generators of the 32-bit float group's decoded operations, one per operation, every
//! field over its whole encodable range (`decode::tests` joins them through `form`'s
//! `GROUP_HOOKS`). A field that only some layouts have (`FMNMX`'s `.NAN` and `FSET`'s combine
//! operation without a uniform register, a high half only for a 16-bit register, `I2F`'s select
//! by source width, `F2FP`'s `.H1` with `.MERGE_C`) is generated only where it can be encoded.

use proptest::prelude::*;
use proptest::sample::select;
use proptest::strategy::BoxedStrategy;

use super::*;
use crate::testing::{any_gpr, any_pred_src, any_reg_src, any_wide_src};

/// `src` without its modifier bits.
fn unmodified(src: AluSrc) -> AluSrc {
    match src {
        AluSrc::Reg { reg, .. } => AluSrc::reg(reg),
        AluSrc::UniformReg { reg, .. } => AluSrc::UniformReg {
            reg,
            abs: false,
            neg: false,
        },
        AluSrc::CBuf { cbuf, .. } => AluSrc::CBuf {
            cbuf,
            abs: false,
            neg: false,
        },
        AluSrc::Imm(imm) => AluSrc::Imm(imm),
    }
}

/// A source of any kind, with any modifiers its kind has.
fn any_src() -> impl Strategy<Value = AluSrc> {
    prop_oneof![any_reg_src(), any_wide_src()]
}

/// A source of any kind, unmodified.
fn any_plain_src() -> impl Strategy<Value = AluSrc> {
    any_src().prop_map(unmodified)
}

fn any_pred() -> impl Strategy<Value = Pred> {
    (0..=Pred::PT.index()).prop_map(|index| Pred::new(index).unwrap_or(Pred::PT))
}

fn any_rnd() -> impl Strategy<Value = RoundMode> {
    select(RoundMode::ALL.to_vec())
}

fn any_float_type() -> impl Strategy<Value = FloatType> {
    select(FloatType::ALL.to_vec())
}

fn any_int_type() -> impl Strategy<Value = IntType> {
    (0_u8..4, any::<bool>()).prop_map(|(size, signed)| IntType::new(size, signed))
}

const fn is_uniform(src: AluSrc) -> bool {
    matches!(src, AluSrc::UniformReg { .. })
}

const fn is_reg(src: AluSrc) -> bool {
    matches!(src, AluSrc::Reg { .. })
}

/// Every operation of the group, one generator each.
pub(super) fn all() -> Vec<BoxedStrategy<Op>> {
    vec![
        ffma(),
        fadd(),
        fmul(),
        mufu(),
        fswzadd(),
        fmnmx(),
        fset(),
        fsetp(),
        fsel(),
        frnd(),
        f2f(),
        f2i(),
        i2f(),
        i2fp(),
        f2fp(),
    ]
}

fn ffma() -> BoxedStrategy<Op> {
    let ffma_srcs = prop_oneof![
        (any_reg_src(), any_reg_src(), any_reg_src()).prop_map(|(a, b, c)| [a, b, c]),
        (any_reg_src(), any_wide_src(), any_reg_src()).prop_map(|(a, b, c)| [a, b, c]),
        (any_reg_src(), any_reg_src(), any_wide_src()).prop_map(|(a, b, c)| [a, b, c]),
    ];
    ((any_gpr(), ffma_srcs, any::<[bool; 3]>(), any_rnd()).prop_map(|(dst, srcs, flags, rnd)| {
        let [saturate, ftz, dnz] = flags;
        Op::FFma(OpFFma {
            dst,
            srcs,
            saturate,
            rnd,
            ftz,
            dnz,
        })
    }))
    .boxed()
}

fn fadd() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        any_reg_src(),
        any_src(),
        any_gpr(),
        any::<[bool; 2]>(),
        any_rnd(),
    )
        .prop_map(|(dst, a, b, ignored, [saturate, ftz], rnd)| {
            Op::FAdd(OpFAdd {
                dst,
                srcs: [a, b],
                ignored,
                saturate,
                rnd,
                ftz,
            })
        }))
    .boxed()
}

fn fmul() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        any_reg_src(),
        any_src(),
        any_gpr(),
        any::<[bool; 3]>(),
        any_rnd(),
        select(FMulScale::ALL.to_vec()),
    )
        .prop_map(|(dst, a, b, ignored, [saturate, ftz, dnz], rnd, scale)| {
            Op::FMul(OpFMul {
                dst,
                srcs: [a, b],
                ignored,
                saturate,
                rnd,
                ftz,
                dnz,
                scale,
            })
        }))
    .boxed()
}

fn mufu() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        select(MuFuOp::ALL.to_vec()),
        any_src(),
        any::<[bool; 2]>(),
    )
        .prop_map(|(dst, op, src, [f16, high])| {
            Op::MuFu(OpMuFu {
                dst,
                op,
                src,
                f16,
                high: high && f16 && is_reg(src),
            })
        }))
    .boxed()
}

fn fswzadd() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        [any_gpr(), any_gpr()],
        [
            select(FSwzAddOp::ALL.to_vec()),
            select(FSwzAddOp::ALL.to_vec()),
            select(FSwzAddOp::ALL.to_vec()),
            select(FSwzAddOp::ALL.to_vec()),
        ],
        any_rnd(),
        any::<[bool; 2]>(),
    )
        .prop_map(|(dst, srcs, ops, rnd, [ftz, ndv])| {
            Op::FSwzAdd(OpFSwzAdd {
                dst,
                srcs,
                ops,
                rnd,
                ftz,
                ndv,
            })
        }))
    .boxed()
}

fn fmnmx() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        any_reg_src(),
        any_src(),
        any_pred_src(),
        any::<[bool; 2]>(),
    )
        .prop_map(|(dst, a, b, min, [ftz, nan])| {
            Op::FMnMx(OpFMnMx {
                dst,
                srcs: [a, b],
                min,
                ftz,
                nan: nan && !is_uniform(b),
            })
        }))
    .boxed()
}

fn fset() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        select(FloatCmpOp::ALL.to_vec()),
        any_reg_src(),
        any_src(),
        select(PredSetOp::ALL.to_vec()),
        any_pred_src(),
        any::<bool>(),
    )
        .prop_map(|(dst, cmp, a, b, set_op, accum, ftz)| {
            Op::FSet(OpFSet {
                dst,
                cmp,
                srcs: [a, b],
                set_op: if is_uniform(b) {
                    PredSetOp::And
                } else {
                    set_op
                },
                accum,
                ftz,
            })
        }))
    .boxed()
}

fn fsetp() -> BoxedStrategy<Op> {
    ((
        [any_pred(), any_pred()],
        select(FloatCmpOp::ALL.to_vec()),
        select(PredSetOp::ALL.to_vec()),
        (any_reg_src(), any_src()),
        any_pred_src(),
        any::<bool>(),
    )
        .prop_map(|([dst, dst1], cmp, set_op, (a, b), accum, ftz)| {
            Op::FSetP(OpFSetP {
                dst,
                dst1,
                cmp,
                set_op,
                srcs: [a, b],
                accum,
                ftz,
            })
        }))
    .boxed()
}

fn fsel() -> BoxedStrategy<Op> {
    let fsel_b =
        any_src().prop_filter("FSEL has no uniform-register form", |src| !is_uniform(*src));
    ((any_gpr(), any_reg_src(), fsel_b, any_pred_src()).prop_map(|(dst, a, b, cond)| {
        Op::FSel(OpFSel {
            dst,
            srcs: [a, b],
            cond,
        })
    }))
    .boxed()
}

fn frnd() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        any_src(),
        (any_float_type(), any_float_type()),
        any_rnd(),
        any::<[bool; 2]>(),
    )
        .prop_map(|(dst, src, (dst_type, src_type), rnd, [ftz, wide])| {
            Op::FRnd(OpFRnd {
                dst,
                src,
                dst_type,
                src_type,
                rnd,
                ftz,
                wide,
            })
        }))
    .boxed()
}

fn f2f() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        any_src(),
        (any_float_type(), any_float_type()),
        any_rnd(),
        any::<[bool; 2]>(),
    )
        .prop_map(|(dst, src, (dst_type, src_type), rnd, [ftz, wide])| {
            Op::F2F(OpF2F {
                dst,
                src,
                dst_type,
                src_type,
                rnd,
                ftz,
                wide,
            })
        }))
    .boxed()
}

fn f2i() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        any_src(),
        (any_int_type(), any_float_type()),
        any_rnd(),
        any::<[bool; 4]>(),
    )
        .prop_map(
            |(dst, src, (dst_type, src_type), rnd, [ftz, ntz, high, wide])| {
                let half = src_type == FloatType::F16 && is_reg(src) && !wide;
                Op::F2I(OpF2I {
                    dst,
                    src,
                    dst_type,
                    src_type,
                    rnd,
                    ftz,
                    ntz,
                    high: high && half,
                    wide,
                })
            },
        ))
    .boxed()
}

fn i2f() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        any_plain_src(),
        (any_float_type(), any_int_type()),
        any_rnd(),
        any::<u8>(),
        any::<bool>(),
    )
        .prop_map(|(dst, src, (dst_type, src_type), rnd, select, wide)| {
            let width = if matches!(src, AluSrc::Imm(_)) {
                0
            } else {
                i2f_select(src_type).width
            };
            Op::I2F(OpI2F {
                dst,
                src,
                dst_type,
                src_type,
                rnd,
                select: select & ((1 << width) - 1),
                wide,
            })
        }))
    .boxed()
}

fn i2fp() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        any_plain_src(),
        (any_float_type(), any_int_type()),
        any_rnd(),
    )
        .prop_map(|(dst, src, (dst_type, src_type), rnd)| {
            Op::I2FP(OpI2FP {
                dst,
                src,
                dst_type,
                src_type,
                rnd,
            })
        }))
    .boxed()
}

fn f2fp() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        any_gpr(),
        any_plain_src(),
        any_gpr(),
        any::<[bool; 2]>(),
        any_rnd(),
    )
        .prop_map(|(dst, a, b, c, [merge_c, h1], rnd)| {
            Op::F2FP(OpF2FP {
                dst,
                srcs: [AluSrc::reg(a), b],
                c,
                merge_c,
                h1: h1 && merge_c,
                rnd,
            })
        }))
    .boxed()
}
