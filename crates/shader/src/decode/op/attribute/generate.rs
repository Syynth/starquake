//! Test generators of the attribute group's decoded operations, one per operation, every field
//! over its whole encodable range (`decode::tests` joins them through `form`'s `GROUP_HOOKS`).

use proptest::prelude::*;
use proptest::sample::select;
use proptest::strategy::BoxedStrategy;

use super::*;
use crate::testing::{any_gpr, any_pred, any_uniform_reg, any_unmodified_src};

/// Any byte address [`ATTR_ADDR`] holds.
fn any_attr_addr() -> impl Strategy<Value = u16> {
    0_u16..(1 << ATTR_ADDR.width)
}

/// Any count [`ATTR_COMPS`] holds, 1 to 4.
fn any_comps() -> impl Strategy<Value = u8> {
    1_u8..=(1 << ATTR_COMPS.width)
}

/// Any byte address [`ATTR_WORD`] holds: a multiple of [`ATTR_WORD_BYTES`].
fn any_attr_word() -> impl Strategy<Value = u16> {
    (0_u16..(1 << ATTR_WORD.width)).prop_map(|word| word * ATTR_WORD_BYTES)
}

/// Every operation of the group.
pub(super) fn all() -> Vec<BoxedStrategy<Op>> {
    let ald = (
        (any_gpr(), any_gpr(), any_gpr()),
        any_attr_addr(),
        any_comps(),
        any::<[bool; 3]>(),
    )
        .prop_map(|((dst, vtx, offset), addr, comps, [patch, phys, output])| {
            Op::ALd(OpALd {
                dst,
                vtx,
                offset,
                addr,
                comps,
                patch,
                phys,
                output,
            })
        });
    let ast = (
        (any_gpr(), any_gpr(), any_gpr()),
        any_attr_addr(),
        any_comps(),
        any::<[bool; 2]>(),
    )
        .prop_map(|((data, vtx, offset), addr, comps, [patch, phys])| {
            Op::ASt(OpASt {
                data,
                vtx,
                offset,
                addr,
                comps,
                patch,
                phys,
            })
        });
    let al2p = (
        any_gpr(),
        any_gpr(),
        any_attr_addr(),
        any_comps(),
        any::<bool>(),
    )
        .prop_map(|(dst, offset, addr, comps, output)| {
            Op::AL2P(OpAL2P {
                dst,
                offset,
                addr,
                comps,
                output,
            })
        });
    let s2r_dst = prop_oneof![
        any_gpr().prop_map(S2RDst::Reg),
        any_uniform_reg().prop_map(S2RDst::Uniform),
    ];
    let s2r = (s2r_dst, any::<u8>()).prop_map(|(dst, sr)| {
        Op::S2R(OpS2R {
            dst,
            sr: SpecialReg(sr),
        })
    });
    let cs2r = (any_gpr(), any::<u8>(), any::<bool>()).prop_map(|(dst, sr, wide)| {
        Op::CS2R(OpCS2R {
            dst,
            sr: SpecialReg(sr),
            wide,
        })
    });
    let ipa = (
        (any_gpr(), any_gpr()),
        any_attr_word(),
        select(&InterpFreq::ALL[..]),
        select(&InterpLoc::ALL[..]),
        any_pred(),
    )
        .prop_map(|((dst, offset), addr, freq, loc, pred)| {
            Op::Ipa(OpIpa {
                dst,
                addr,
                freq,
                loc,
                offset,
                pred,
            })
        });
    let ldtram = (any_gpr(), any_attr_word(), any::<bool>(), any_uniform_reg()).prop_map(
        |(dst, addr, use_c, ureg)| {
            Op::LdTram(OpLdTram {
                dst,
                addr,
                use_c,
                ureg,
            })
        },
    );
    let pixld = (any_gpr(), select(&PixVal::ALL[..]), any_pred())
        .prop_map(|(dst, val, pred)| Op::PixLd(OpPixLd { dst, val, pred }));
    let out = (
        any_gpr(),
        any_gpr(),
        any_unmodified_src(),
        select(&OutType::ALL[..]),
    )
        .prop_map(|(dst, handle, stream, out_type)| {
            Op::Out(OpOut {
                dst,
                handle,
                stream,
                out_type,
            })
        });
    let out_final = (any_gpr(), any_gpr(), any_unmodified_src()).prop_map(|(dst, handle, stream)| {
        Op::OutFinal(OpOutFinal {
            dst,
            handle,
            stream,
        })
    });
    let isberd = (
        (any_gpr(), any_gpr()),
        any::<u16>(),
        select(&IsbeMemType::ALL[..]),
        select(&IsbeAccess::ALL[..]),
        any::<[bool; 2]>(),
    )
        .prop_map(|((dst, offset), imm_offset, mem_type, access, [output, skew])| {
            Op::Isberd(OpIsberd {
                dst,
                offset,
                imm_offset,
                mem_type,
                access,
                output,
                skew,
            })
        });
    let writable: Vec<IsbeAccess> = IsbeAccess::ALL
        .into_iter()
        .filter(|access| access.writable())
        .collect();
    let isbewr = (
        (any_gpr(), any_gpr()),
        any::<u16>(),
        select(&IsbeMemType::ALL[..]),
        select(writable),
        any::<[bool; 2]>(),
    )
        .prop_map(|((offset, data), imm_offset, mem_type, access, [output, skew])| {
            Op::Isbewr(OpIsbewr {
                offset,
                data,
                imm_offset,
                mem_type,
                access,
                output,
                skew,
            })
        });
    vec![
        ald.boxed(),
        ast.boxed(),
        al2p.boxed(),
        s2r.boxed(),
        cs2r.boxed(),
        ipa.boxed(),
        ldtram.boxed(),
        pixld.boxed(),
        out.boxed(),
        out_final.boxed(),
        isberd.boxed(),
        isbewr.boxed(),
    ]
}
