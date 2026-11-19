//! Test generators of the integer group's decoded operations, one per operation, every field
//! over its whole encodable range (`decode::tests` joins them through `form`'s `GROUP_HOOKS`).

use proptest::prelude::*;
use proptest::sample::select;

use super::*;
use crate::testing::{any_cbuf, any_gpr, any_pred_src, any_uniform_reg};

fn any_pred() -> impl Strategy<Value = Pred> {
    (0..=Pred::PT.index()).prop_map(|index| Pred::new(index).unwrap_or(Pred::PT))
}

fn any_uniform_pred_src() -> impl Strategy<Value = UniformPredSrc> {
    use super::super::super::operand::UniformPred;
    (0..=UniformPred::UPT.index(), any::<bool>()).prop_map(|(index, negated)| UniformPredSrc {
        reg: UniformPred::new(index).unwrap_or(UniformPred::UPT),
        negated,
    })
}

/// Any value of a field `width` bits wide.
fn any_bits(field: Field) -> std::ops::Range<u8> {
    0..u8::try_from(1_u32 << field.width).unwrap_or(u8::MAX)
}

/// Source `index` of `kind` in `form`: of the kind its slot holds, negated where it has a
/// negation bit; an absent source is `RZ`, which nothing reads.
fn any_src(form: AluForm, kind: Alu, index: usize) -> BoxedStrategy<AluSrc> {
    let src = kind.srcs()[index];
    if src == IntSrc::Absent {
        return Just(AluSrc::reg(Gpr::RZ)).boxed();
    }
    let slot = match index {
        0 => Slot::Reg,
        1 => form.slot(Source::One),
        _ => form.slot(Source::Two),
    };
    let base = match slot {
        Slot::WideReg | Slot::Reg => any_gpr().prop_map(AluSrc::reg).boxed(),
        Slot::Imm => any::<u32>().prop_map(AluSrc::Imm).boxed(),
        Slot::CBuf => any_cbuf()
            .prop_map(|cbuf| AluSrc::CBuf {
                cbuf,
                abs: false,
                neg: false,
            })
            .boxed(),
        Slot::UniformReg => any_uniform_reg()
            .prop_map(|reg| AluSrc::UniformReg {
                reg,
                abs: false,
                neg: false,
            })
            .boxed(),
    };
    if neg_field(form, index, src).is_some() {
        (base, any::<bool>())
            .prop_map(|(src, neg)| negated(src, neg))
            .boxed()
    } else {
        base
    }
}

/// A form of `kind` and its three sources.
fn any_srcs(kind: Alu) -> BoxedStrategy<(AluForm, [AluSrc; 3])> {
    select(kind.forms().to_vec())
        .prop_flat_map(move |form| {
            (
                Just(form),
                any_src(form, kind, 0),
                any_src(form, kind, 1),
                any_src(form, kind, 2),
            )
        })
        .prop_map(|(form, a, b, c)| (form, [a, b, c]))
        .boxed()
}

/// The sources of `kind`, without the form.
fn srcs_of(kind: Alu) -> BoxedStrategy<[AluSrc; 3]> {
    any_srcs(kind).prop_map(|(_, srcs)| srcs).boxed()
}

fn mov() -> BoxedStrategy<Op> {
    let src = prop_oneof![
        any_gpr().prop_map(MovSrc::Reg),
        any::<u32>().prop_map(MovSrc::Imm),
        any_cbuf().prop_map(MovSrc::CBuf),
        any_uniform_reg().prop_map(MovSrc::UniformReg),
    ];
    (any_gpr(), src, any_bits(MOV_QUAD_LANES))
        .prop_map(|(dst, src, quad_lanes)| {
            Op::Mov(OpMov {
                dst,
                src,
                quad_lanes,
            })
        })
        .boxed()
}

fn any_iadd3() -> impl Strategy<Value = OpIAdd3> {
    (
        any_gpr(),
        [any_pred(), any_pred()],
        srcs_of(Alu::IAdd3),
        [any_pred_src(), any_pred_src()],
    )
        .prop_map(|(dst, overflow, srcs, carry)| OpIAdd3 {
            dst,
            overflow,
            srcs,
            carry,
        })
}

fn any_lea() -> impl Strategy<Value = OpLea> {
    (
        any_gpr(),
        any_pred(),
        srcs_of(Alu::Lea),
        any_bits(LEA_SHIFT),
        any::<[bool; 3]>(),
        any_pred_src(),
    )
        .prop_map(|(dst, overflow, [a, b, a_high], shift, flags, carry)| {
            let [dst_high, sx32, intermediate] = flags;
            OpLea {
                dst,
                overflow,
                a,
                b,
                a_high,
                shift,
                dst_high,
                sx32,
                intermediate,
                carry,
            }
        })
}

/// `IADD3`, `IADD3.X`, `LEA`, `LEA.X`, `LOP3`, `SHF`, `PRMT` and `IDP`.
fn three_sources() -> Vec<BoxedStrategy<Op>> {
    let lop3 = (any_pred(), any_gpr(), srcs_of(Alu::Lop3), any::<u8>()).prop_map(
        |(pred, dst, srcs, lut)| {
            Op::Lop3(OpLop3 {
                pred,
                dst,
                srcs,
                lut,
            })
        },
    );
    let shf = (
        any_gpr(),
        srcs_of(Alu::Shf),
        select(ShfType::ALL.to_vec()),
        any::<[bool; 3]>(),
    )
        .prop_map(|(dst, [low, shift, high], data_type, flags)| {
            let [wrap, right, dst_high] = flags;
            Op::Shf(OpShf {
                dst,
                low,
                shift,
                high,
                data_type,
                wrap,
                right,
                dst_high,
            })
        });
    let prmt = (
        any_gpr(),
        srcs_of(Alu::Prmt),
        select(PrmtMode::ALL.to_vec()),
    )
        .prop_map(|(dst, [a, sel, b], mode)| {
            Op::Prmt(OpPrmt {
                dst,
                srcs: [a, b],
                sel,
                mode,
            })
        });
    let idp = (any_gpr(), any_srcs(Alu::IDp), any::<[bool; 3]>()).prop_map(
        |(dst, (form, srcs), [signed_a, signed_b, two_a])| {
            Op::IDp4(OpIDp4 {
                dst,
                srcs,
                signed: [signed_a, signed_b],
                two_a: two_a && form == AluForm::Reg,
            })
        },
    );
    vec![
        any_iadd3().prop_map(Op::IAdd3).boxed(),
        any_iadd3()
            .prop_map(|add| Op::IAdd3X(OpIAdd3X { add }))
            .boxed(),
        any_lea().prop_map(Op::Lea).boxed(),
        any_lea().prop_map(|lea| Op::LeaX(OpLeaX { lea })).boxed(),
        lop3.boxed(),
        shf.boxed(),
        prmt.boxed(),
        idp.boxed(),
    ]
}

/// `IMAD` (with `IMAD.HI`) and `IMAD.WIDE`.
fn multiplies() -> Vec<BoxedStrategy<Op>> {
    let lo = (srcs_of(Alu::IMad), any::<bool>(), any_pred_src())
        .prop_map(|(srcs, x, carry)| (srcs, IMadMode::Lo { x, carry }));
    let hi = srcs_of(Alu::IMadHi).prop_map(|srcs| (srcs, IMadMode::Hi));
    let imad =
        (any_gpr(), prop_oneof![lo, hi], any::<bool>()).prop_map(|(dst, (srcs, mode), signed)| {
            Op::IMad(OpIMad {
                dst,
                srcs,
                signed,
                mode,
            })
        });
    let wide = (
        any_gpr(),
        srcs_of(Alu::IMadWide),
        any::<bool>(),
        any_pred_src(),
    )
        .prop_map(|(dst, srcs, signed, carry)| {
            Op::IMad64(OpIMad64 {
                dst,
                srcs,
                signed,
                carry,
            })
        });
    vec![imad.boxed(), wide.boxed()]
}

/// `IMNMX`, `SEL`, `SGXT`, `BMSK` and `ISETP`.
fn two_sources() -> Vec<BoxedStrategy<Op>> {
    let imnmx = (
        any_gpr(),
        srcs_of(Alu::IMnMx),
        any::<bool>(),
        any_pred_src(),
    )
        .prop_map(|(dst, [a, b, _], signed, min)| {
            Op::IMnMx(OpIMnMx {
                dst,
                srcs: [a, b],
                signed,
                min,
            })
        });
    let sel = (any_gpr(), srcs_of(Alu::Sel), any_pred_src()).prop_map(|(dst, [a, b, _], cond)| {
        Op::Sel(OpSel {
            dst,
            srcs: [a, b],
            cond,
        })
    });
    let sgxt = (any_gpr(), srcs_of(Alu::Sgxt), any::<[bool; 2]>()).prop_map(
        |(dst, [a, bits, _], [signed, wrap])| {
            Op::Sgxt(OpSgxt {
                dst,
                a,
                bits,
                signed,
                wrap,
            })
        },
    );
    let bmsk =
        (any_gpr(), srcs_of(Alu::BMsk), any::<bool>()).prop_map(|(dst, [pos, width, _], wrap)| {
            Op::BMsk(OpBMsk {
                dst,
                pos,
                width,
                wrap,
            })
        });
    let isetp = (
        [any_pred(), any_pred()],
        srcs_of(Alu::ISetP),
        select(IntCmpOp::ALL.to_vec()),
        select(PredSetOp::ALL.to_vec()),
        any::<[bool; 2]>(),
        [any_pred_src(), any_pred_src()],
    )
        .prop_map(
            |(dsts, [a, b, _], cmp_op, set_op, [signed, ex], [accum, low_cmp])| {
                Op::ISetP(OpISetP {
                    dsts,
                    srcs: [a, b],
                    cmp_op,
                    signed,
                    set_op,
                    ex,
                    accum,
                    low_cmp,
                })
            },
        );
    vec![
        imnmx.boxed(),
        sel.boxed(),
        sgxt.boxed(),
        bmsk.boxed(),
        isetp.boxed(),
    ]
}

/// `MOV`, `IABS`, `BREV`, `POPC` and `FLO`.
fn one_source() -> Vec<BoxedStrategy<Op>> {
    let iabs = (any_gpr(), srcs_of(Alu::IAbs))
        .prop_map(|(dst, [_, src, _])| Op::IAbs(OpIAbs { dst, src }));
    let brev = (any_gpr(), srcs_of(Alu::BRev))
        .prop_map(|(dst, [_, src, _])| Op::BRev(OpBRev { dst, src }));
    let popc = (any_gpr(), srcs_of(Alu::PopC))
        .prop_map(|(dst, [_, src, _])| Op::PopC(OpPopC { dst, src }));
    let flo = (any_gpr(), srcs_of(Alu::Flo), any::<[bool; 2]>()).prop_map(
        |(dst, [_, src, _], [signed, return_shift_amount])| {
            Op::Flo(OpFlo {
                dst,
                src,
                signed,
                return_shift_amount,
            })
        },
    );
    vec![mov(), iabs.boxed(), brev.boxed(), popc.boxed(), flo.boxed()]
}

/// `PLOP3`, `P2R` and `R2P`.
fn predicates() -> Vec<BoxedStrategy<Op>> {
    let c = prop_oneof![
        any_pred_src().prop_map(PLop3Src::Pred),
        any_uniform_pred_src().prop_map(PLop3Src::Uniform),
    ];
    let plop3 = (
        [any_pred(), any_pred()],
        [any_pred_src(), any_pred_src()],
        c,
        any::<[u8; 2]>(),
    )
        .prop_map(|(dsts, srcs, c, luts)| {
            Op::PLop3(OpPLop3 {
                dsts,
                srcs,
                c,
                luts,
            })
        });
    let p2r = (any_gpr(), any_gpr(), any_bits(PR_MASK))
        .prop_map(|(dst, src, mask)| Op::P2R(OpP2R { dst, src, mask }));
    let r2p = (any_gpr(), any_bits(R2P_BYTE), any_bits(PR_MASK))
        .prop_map(|(src, byte, mask)| Op::R2P(OpR2P { src, byte, mask }));
    vec![plop3.boxed(), p2r.boxed(), r2p.boxed()]
}

/// One generator per operation of the group.
pub(super) fn ops() -> Vec<BoxedStrategy<Op>> {
    [
        one_source(),
        two_sources(),
        three_sources(),
        multiplies(),
        predicates(),
    ]
    .concat()
}
