//! Test generators of the half and double group's decoded operations, one per operation, every
//! field over its whole encodable range (`decode::tests` joins them through `form`'s
//! `GROUP_HOOKS`). A packed-half source gets any swizzle but an immediate, which has no swizzle
//! bits and so reads both halves as stored.

use proptest::prelude::*;
use proptest::sample::select;
use proptest::strategy::BoxedStrategy;

use super::*;
use crate::decode::HalfSwizzle;
use crate::testing::{any_gpr, any_pred, any_pred_src, any_reg_src, any_wide_src};

fn any_swizzle() -> impl Strategy<Value = HalfSwizzle> {
    select(HalfSwizzle::ALL.to_vec())
}

/// `src` with `swizzle`, or both halves as stored for an immediate.
fn swizzled(src: AluSrc, swizzle: HalfSwizzle) -> HalfSrc {
    match src {
        AluSrc::Imm(_) => HalfSrc::new(src),
        _ => HalfSrc { src, swizzle },
    }
}

/// A packed-half register source with any modifiers and swizzle.
fn any_half_reg() -> impl Strategy<Value = HalfSrc> {
    (any_reg_src(), any_swizzle()).prop_map(|(src, swizzle)| swizzled(src, swizzle))
}

/// A packed-half source that takes the wide slot.
fn any_half_wide() -> impl Strategy<Value = HalfSrc> {
    (any_wide_src(), any_swizzle()).prop_map(|(src, swizzle)| swizzled(src, swizzle))
}

/// A packed-half source of any kind.
fn any_half_src() -> impl Strategy<Value = HalfSrc> {
    prop_oneof![any_half_reg(), any_half_wide()]
}

/// A 64-bit source of any kind, with any modifiers its kind has.
fn any_src() -> impl Strategy<Value = AluSrc> {
    prop_oneof![any_reg_src(), any_wide_src()]
}

fn any_rnd() -> impl Strategy<Value = RoundMode> {
    select(RoundMode::ALL.to_vec())
}

fn any_cmp() -> impl Strategy<Value = FloatCmpOp> {
    select(FloatCmpOp::ALL.to_vec())
}

fn any_set_op() -> impl Strategy<Value = PredSetOp> {
    select(PredSetOp::ALL.to_vec())
}

/// Every operation of the group, one generator each.
pub(super) fn all() -> Vec<BoxedStrategy<Op>> {
    vec![
        hadd2(),
        hfma2(),
        hmul2(),
        hset2(),
        hsetp2(),
        hmnmx2(),
        dadd(),
        dmul(),
        dfma(),
        dsetp(),
    ]
}

fn hadd2() -> BoxedStrategy<Op> {
    ((any_gpr(), any_half_reg(), any_half_src(), any::<[bool; 3]>()).prop_map(
        |(dst, a, b, [saturate, f32, ftz])| {
            Op::HAdd2(OpHAdd2 {
                dst,
                srcs: [a, b],
                saturate,
                f32,
                ftz,
            })
        },
    ))
    .boxed()
}

fn hfma2() -> BoxedStrategy<Op> {
    let srcs = prop_oneof![
        (any_half_reg(), any_half_reg(), any_half_reg()).prop_map(|(a, b, c)| [a, b, c]),
        (any_half_reg(), any_half_wide(), any_half_reg()).prop_map(|(a, b, c)| [a, b, c]),
        (any_half_reg(), any_half_reg(), any_half_wide()).prop_map(|(a, b, c)| [a, b, c]),
    ];
    ((any_gpr(), srcs, any::<[bool; 7]>()).prop_map(|(dst, srcs, flags)| {
        let [saturate, f32, ftz, dnz, relu, bf16, neg_high_b] = flags;
        Op::HFma2(OpHFma2 {
            dst,
            srcs,
            saturate,
            f32,
            ftz,
            dnz,
            relu,
            bf16,
            neg_high_b,
        })
    }))
    .boxed()
}

fn hmul2() -> BoxedStrategy<Op> {
    ((any_gpr(), any_half_reg(), any_half_src(), any::<[bool; 4]>()).prop_map(
        |(dst, a, b, [saturate, ftz, dnz, relu])| {
            Op::HMul2(OpHMul2 {
                dst,
                srcs: [a, b],
                saturate,
                ftz,
                dnz,
                relu,
            })
        },
    ))
    .boxed()
}

fn hset2() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        any_cmp(),
        (any_half_reg(), any_half_src()),
        any_set_op(),
        any_pred_src(),
        any::<[bool; 2]>(),
    )
        .prop_map(|(dst, cmp, (a, b), set_op, accum, [ftz, bf])| {
            Op::HSet2(OpHSet2 {
                dst,
                cmp,
                srcs: [a, b],
                set_op,
                accum,
                ftz,
                bf,
            })
        }))
    .boxed()
}

fn hsetp2() -> BoxedStrategy<Op> {
    ((
        [any_pred(), any_pred()],
        any_cmp(),
        any_set_op(),
        (any_half_reg(), any_half_src()),
        any_pred_src(),
        any::<[bool; 2]>(),
    )
        .prop_map(
            |([dst, dst1], cmp, set_op, (a, b), accum, [ftz, horizontal])| {
                Op::HSetP2(OpHSetP2 {
                    dst,
                    dst1,
                    cmp,
                    set_op,
                    srcs: [a, b],
                    accum,
                    ftz,
                    horizontal,
                })
            },
        ))
    .boxed()
}

fn hmnmx2() -> BoxedStrategy<Op> {
    ((
        any_gpr(),
        any_half_reg(),
        any_half_src(),
        any_pred_src(),
        any::<[bool; 5]>(),
    )
        .prop_map(|(dst, a, b, min, [ftz, f32, nan, xorsign, bf16])| {
            Op::HMnMx2(OpHMnMx2 {
                dst,
                srcs: [a, b],
                min,
                ftz,
                f32,
                nan,
                xorsign,
                bf16,
            })
        }))
    .boxed()
}

fn dadd() -> BoxedStrategy<Op> {
    ((any_gpr(), any_reg_src(), any_src(), any_rnd()))
        .prop_map(|(dst, a, b, rnd)| {
            Op::DAdd(OpDAdd {
                dst,
                srcs: [a, b],
                rnd,
            })
        })
        .boxed()
}

fn dmul() -> BoxedStrategy<Op> {
    ((any_gpr(), any_reg_src(), any_src(), any_rnd()))
        .prop_map(|(dst, a, b, rnd)| {
            Op::DMul(OpDMul {
                dst,
                srcs: [a, b],
                rnd,
            })
        })
        .boxed()
}

fn dfma() -> BoxedStrategy<Op> {
    let srcs = prop_oneof![
        (any_reg_src(), any_reg_src(), any_reg_src()).prop_map(|(a, b, c)| [a, b, c]),
        (any_reg_src(), any_wide_src(), any_reg_src()).prop_map(|(a, b, c)| [a, b, c]),
        (any_reg_src(), any_reg_src(), any_wide_src()).prop_map(|(a, b, c)| [a, b, c]),
    ];
    ((any_gpr(), srcs, any_rnd()))
        .prop_map(|(dst, srcs, rnd)| Op::DFma(OpDFma { dst, srcs, rnd }))
        .boxed()
}

fn dsetp() -> BoxedStrategy<Op> {
    ((
        [any_pred(), any_pred()],
        any_cmp(),
        any_set_op(),
        (any_reg_src(), any_src()),
        any_pred_src(),
    )
        .prop_map(|([dst, dst1], cmp, set_op, (a, b), accum)| {
            Op::DSetP(OpDSetP {
                dst,
                dst1,
                cmp,
                set_op,
                srcs: [a, b],
                accum,
            })
        }))
    .boxed()
}
