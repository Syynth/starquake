//! Test generators of the texture and surface group's decoded operations, one per operation,
//! every field over its whole encodable range (`decode::tests` joins them through `form`'s
//! `GROUP_HOOKS`). A bit a form's row leaves out (`.F16` with a `.B` handle, `.CL` without a
//! `.UR` one) is generated only where the form has it.

use proptest::prelude::*;
use proptest::sample::select;
use proptest::strategy::BoxedStrategy;

use super::*;
use crate::testing::{any_gpr, any_pred, any_uniform_reg};

/// Any constant-buffer handle the fields hold: a buffer [`TEX_CBUF_INDEX`] holds and a byte
/// offset of a word [`TEX_CBUF_OFFSET`] holds.
fn any_cbuf() -> impl Strategy<Value = TexCBufRef> {
    (
        0_u8..(1 << TEX_CBUF_INDEX.width),
        0_u16..(1 << TEX_CBUF_OFFSET.width),
    )
        .prop_map(|(index, word)| TexCBufRef {
            index,
            offset: word * TEX_CBUF_UNIT,
        })
}

/// Any handle of the forms in `opcodes`: a constant buffer, the sources, and a uniform register
/// where the instruction has a `.UR` form.
fn any_tex_ref(opcodes: TexOpcodes) -> BoxedStrategy<TexRef> {
    let cbuf = any_cbuf().prop_map(TexRef::CBuf);
    if opcodes.uniform.is_some() {
        prop_oneof![
            cbuf,
            Just(TexRef::Bindless),
            any_uniform_reg().prop_map(TexRef::Uniform),
        ]
        .boxed()
    } else {
        prop_oneof![cbuf, Just(TexRef::Bindless)].boxed()
    }
}

/// Any four-bit channel mask.
fn any_mask() -> impl Strategy<Value = ChannelMask> {
    (0_u8..(1 << CHANNEL_MASK.width))
        .prop_map(|bits| ChannelMask::new(bits).unwrap_or(ChannelMask::RGBA))
}

fn any_dsts() -> impl Strategy<Value = [Gpr; 2]> {
    (any_gpr(), any_gpr()).prop_map(|(a, b)| [a, b])
}

fn any_eviction() -> impl Strategy<Value = MemEvictionPriority> {
    select(MemEvictionPriority::ALL)
}

/// Every operation of the group.
pub(super) fn all() -> Vec<BoxedStrategy<Op>> {
    [textures(), queries(), surfaces()].concat()
}

/// `TEX`, `TLD` and `TLD4` ([`gather`]).
fn textures() -> Vec<BoxedStrategy<Op>> {
    let tex = (
        (any_dsts(), any_pred(), any_tex_ref(TEX_OPCODES), any_dsts()),
        (select(TexDim::ALL), select(TexLodMode::ALL), any_eviction()),
        any::<[bool; 6]>(),
        any_mask(),
    )
        .prop_map(
            |(
                (dsts, fault, tex, srcs),
                (dim, lod_mode, eviction),
                [ndv, z_cmpr, offset, f16, nodep, scalar],
                channel_mask,
            )| {
                Op::Tex(OpTex {
                    dsts,
                    fault,
                    tex,
                    srcs,
                    dim,
                    lod_mode,
                    ndv,
                    z_cmpr,
                    offset,
                    f16: f16 && has_f16(tex),
                    eviction,
                    nodep,
                    channel_mask,
                    scalar,
                })
            },
        );
    let tld = (
        (any_dsts(), any_pred(), any_tex_ref(TLD_OPCODES), any_dsts()),
        (select(TexDim::ALL), select(TexLodMode::ALL), any_eviction()),
        any::<[bool; 6]>(),
        any_mask(),
    )
        .prop_map(
            |(
                (dsts, fault, tex, srcs),
                (dim, lod_mode, eviction),
                [is_ms, clamp, offset, f16, nodep, scalar],
                channel_mask,
            )| {
                Op::Tld(OpTld {
                    dsts,
                    fault,
                    tex,
                    srcs,
                    dim,
                    is_ms,
                    clamp: clamp && has_clamp(tex),
                    lod_mode,
                    offset,
                    f16: f16 && has_f16(tex),
                    eviction,
                    nodep,
                    channel_mask,
                    scalar,
                })
            },
        );
    vec![tex.boxed(), tld.boxed(), gather()]
}

/// `TLD4`.
fn gather() -> BoxedStrategy<Op> {
    let tld4 = (
        (
            any_dsts(),
            any_pred(),
            any_tex_ref(TLD4_OPCODES),
            any_dsts(),
        ),
        (
            select(TexDim::ALL),
            select(TexChannel::ALL),
            select(TexOffsetMode::ALL),
            any_eviction(),
        ),
        any::<[bool; 3]>(),
        any_mask(),
    )
        .prop_map(
            |(
                (dsts, fault, tex, srcs),
                (dim, comp, offset_mode, eviction),
                [z_cmpr, nodep, scalar],
                channel_mask,
            )| {
                Op::Tld4(OpTld4 {
                    dsts,
                    fault,
                    tex,
                    srcs,
                    dim,
                    comp,
                    offset_mode,
                    z_cmpr,
                    eviction,
                    nodep,
                    channel_mask,
                    scalar,
                })
            },
        );
    tld4.boxed()
}

/// `TMML`, `TXD` and `TXQ`.
fn queries() -> Vec<BoxedStrategy<Op>> {
    let tmml = (
        (any_dsts(), any_tex_ref(TMML_OPCODES), any_dsts()),
        select(TexDim::ALL),
        any::<[bool; 2]>(),
        any_mask(),
    )
        .prop_map(|((dsts, tex, srcs), dim, [ndv, nodep], channel_mask)| {
            Op::Tmml(OpTmml {
                dsts,
                tex,
                srcs,
                dim,
                ndv,
                nodep,
                channel_mask,
            })
        });
    let txd = (
        (any_dsts(), any_pred(), any_tex_ref(TXD_OPCODES), any_dsts()),
        (select(TexDim::ALL), any_eviction()),
        any::<[bool; 2]>(),
        any_mask(),
    )
        .prop_map(
            |((dsts, fault, tex, srcs), (dim, eviction), [offset, nodep], channel_mask)| {
                Op::Txd(OpTxd {
                    dsts,
                    fault,
                    tex,
                    srcs,
                    dim,
                    offset,
                    eviction,
                    nodep,
                    channel_mask,
                })
            },
        );
    let txq = (
        (any_dsts(), any_tex_ref(TXQ_OPCODES), any_gpr()),
        select(TexQuery::ALL),
        any::<bool>(),
        any_mask(),
    )
        .prop_map(|((dsts, tex, src), query, nodep, channel_mask)| {
            Op::Txq(OpTxq {
                dsts,
                tex,
                src,
                query,
                nodep,
                channel_mask,
            })
        });
    vec![tmml.boxed(), txd.boxed(), txq.boxed()]
}

/// Any access: a memory type or a channel mask.
fn any_access() -> impl Strategy<Value = ImageAccess> {
    prop_oneof![
        select(MemType::ALL).prop_map(ImageAccess::Binary),
        any_mask().prop_map(ImageAccess::Formatted),
    ]
}

/// Any operation and form of `SUATOM`: a compare-and-swap (packed), any encoded operation with
/// a result, or a reduction (`Add` to `Xor`, which three bits hold).
fn any_atom_form() -> impl Strategy<Value = (AtomOp, bool)> {
    let reductions: Vec<AtomOp> = AtomOp::ENCODED
        .into_iter()
        .filter(|op| op.bits().is_some_and(|bits| bits < 1 << SURED_OP.width))
        .collect();
    prop_oneof![
        Just((AtomOp::CmpExch(AtomCmpSrc::Packed), false)),
        select(&AtomOp::ENCODED[..]).prop_map(|op| (op, false)),
        select(reductions).prop_map(|op| (op, true)),
    ]
}

/// `SULD`, `SUST` and `SUATOM` (with `SURED`).
fn surfaces() -> Vec<BoxedStrategy<Op>> {
    let suld = (
        (any_gpr(), any_pred(), any_gpr(), any_gpr()),
        any_access(),
        select(ImageDim::ALL),
        select(MemOrder::ALL),
        any_eviction(),
    )
        .prop_map(
            |((dst, fault, handle, coord), access, dim, order, eviction)| {
                Op::SuLd(OpSuLd {
                    dst,
                    fault,
                    access,
                    dim,
                    order,
                    eviction,
                    handle,
                    coord,
                })
            },
        );
    let sust = (
        (any_gpr(), any_gpr(), any_gpr()),
        any_access(),
        select(ImageDim::ALL),
        select(MemOrder::ALL),
        any_eviction(),
    )
        .prop_map(|((handle, coord, data), access, dim, order, eviction)| {
            Op::SuSt(OpSuSt {
                access,
                dim,
                order,
                eviction,
                handle,
                coord,
                data,
            })
        });
    let suatom = (
        (any_gpr(), any_pred(), any_gpr(), any_gpr(), any_gpr()),
        (
            select(ImageDim::ALL),
            any_atom_form(),
            select(AtomType::ALL),
        ),
        (select(MemOrder::ALL), any_eviction()),
        any::<bool>(),
    )
        .prop_map(
            |(
                (dst, fault, handle, coord, data),
                (dim, (atom_op, reduce), atom_type),
                (order, eviction),
                ba,
            )| {
                Op::SuAtom(OpSuAtom {
                    dst,
                    fault,
                    dim,
                    atom_op,
                    atom_type,
                    order,
                    eviction,
                    handle,
                    coord,
                    data,
                    ba,
                    reduce,
                })
            },
        );
    vec![suld.boxed(), sust.boxed(), suatom.boxed()]
}
