//! Test generators of the memory group's decoded operations, one per operation, every field
//! over its whole encodable range in every form (`decode::tests` joins them through `form`'s
//! `GROUP_HOOKS`).

use proptest::prelude::*;
use proptest::sample::select;
use proptest::strategy::BoxedStrategy;

use super::*;
use crate::testing::{any_gpr, any_pred, any_pred_src, any_uniform_reg};

/// Any offset [`MEM_OFFSET`] holds: a signed 24-bit number.
fn any_offset() -> impl Strategy<Value = i32> {
    let half = 1_i32 << (MEM_OFFSET.width - 1);
    -half..half
}

/// Any address in `layout`, with only the flags the layout has set.
fn any_addr(layout: AddrLayout) -> BoxedStrategy<MemAddr> {
    let uniform =
        (any_uniform_reg(), any::<bool>(), any::<bool>()).prop_map(move |(reg, wide, enabled)| {
            UniformAddr {
                reg,
                wide: wide && layout.e,
                enabled,
            }
        });
    let uniform = if layout.uniform.is_some() {
        uniform.prop_map(Some).boxed()
    } else {
        Just(None).boxed()
    };
    (any_gpr(), any::<bool>(), uniform, any_offset())
        .prop_map(move |(reg, reg_64, uniform, offset)| MemAddr {
            reg,
            reg_64: reg_64 && (layout.reg_64.is_some() || (layout.uniform.is_none() && layout.e)),
            uniform,
            offset,
        })
        .boxed()
}

fn any_mem_type() -> impl Strategy<Value = MemType> {
    select(MemType::ALL)
}

fn any_eviction() -> impl Strategy<Value = MemEvictionPriority> {
    select(MemEvictionPriority::ALL)
}

fn any_stride() -> impl Strategy<Value = OffsetStride> {
    select(OffsetStride::ALL)
}

fn any_global() -> impl Strategy<Value = MemSpace> {
    (select(MemOrder::ALL), any_eviction())
        .prop_map(|(order, eviction)| MemSpace::Global(GlobalAccess { order, eviction }))
}

/// Every operation of the group.
pub(super) fn all() -> Vec<BoxedStrategy<Op>> {
    vec![
        loads(),
        stores(),
        ldc(),
        atoms(),
        atoms_popc(),
        cctl(),
        membar(),
        Just(Op::ErrBar(OpErrBar {})).boxed(),
        Just(Op::LdgDepBar(OpLdgDepBar {})).boxed(),
        ldgsts(),
    ]
}

/// `LDG` in both forms, `LDL` and `LDS`.
fn loads() -> BoxedStrategy<Op> {
    let ldg = (
        any_addr(GLOBAL),
        any_global(),
        any::<bool>(),
        any_pred(),
        Just(PredSrc::TRUE),
    );
    let ldg_uniform = (
        any_addr(LDG_UNIFORM),
        any_global(),
        any::<bool>(),
        any_pred(),
        any_pred_src(),
    );
    let global = prop_oneof![ldg, ldg_uniform].prop_map(|(addr, space, bit76, pred_dst, pred)| {
        (addr, space, pred, pred_dst, false, bit76)
    });
    let local = (any_addr(LD_LOCAL_SHARED), any_eviction()).prop_map(|(addr, eviction)| {
        (
            addr,
            MemSpace::Local(eviction),
            PredSrc::TRUE,
            Pred::PT,
            false,
            false,
        )
    });
    let shared =
        (any_addr(LD_LOCAL_SHARED), any_stride(), any::<bool>()).prop_map(|(addr, stride, zd)| {
            (
                addr,
                MemSpace::Shared(stride),
                PredSrc::TRUE,
                Pred::PT,
                zd,
                false,
            )
        });
    (
        any_gpr(),
        any_mem_type(),
        prop_oneof![global, local, shared],
    )
        .prop_map(
            |(dst, mem_type, (addr, space, pred, pred_dst, zd, bit76))| {
                Op::Ld(OpLd {
                    dst,
                    addr,
                    mem_type,
                    space,
                    pred,
                    pred_dst,
                    zd,
                    bit76,
                })
            },
        )
        .boxed()
}

/// `STG`, `STL` and `STS`, each in both forms.
fn stores() -> BoxedStrategy<Op> {
    let global = prop_oneof![
        (any_addr(GLOBAL), any_global(), Just(false)),
        (any_addr(ST_GLOBAL_UNIFORM), any_global(), any::<bool>()),
    ];
    let local = (
        prop_oneof![any_addr(AddrLayout::PLAIN), any_addr(ST_UNIFORM_PLAIN)],
        any_eviction(),
    )
        .prop_map(|(addr, eviction)| (addr, MemSpace::Local(eviction), false));
    let shared = (
        prop_oneof![any_addr(AddrLayout::PLAIN), any_addr(ST_UNIFORM_PLAIN)],
        any_stride(),
    )
        .prop_map(|(addr, stride)| (addr, MemSpace::Shared(stride), false));
    (
        any_gpr(),
        any_mem_type(),
        prop_oneof![global, local, shared],
    )
        .prop_map(|(data, mem_type, (addr, space, bit76))| {
            Op::St(OpSt {
                data,
                addr,
                mem_type,
                space,
                bit76,
            })
        })
        .boxed()
}

/// `LDC` with a bound and a bindless buffer.
fn ldc() -> BoxedStrategy<Op> {
    let index_limit = 1_u8 << CBUF_INDEX.width;
    let bound = (
        (0..index_limit).prop_map(CBuf::Binding),
        select(LdcMode::ALL),
    );
    let bindless = (
        any_uniform_reg().prop_map(CBuf::BindlessUniform),
        Just(LdcMode::Indexed),
    );
    (
        any_gpr(),
        prop_oneof![bound, bindless],
        any::<u16>(),
        any_gpr(),
        any_mem_type(),
    )
        .prop_map(|(dst, (buf, mode), offset, offset_reg, mem_type)| {
            Op::Ldc(OpLdc {
                dst,
                cbuf: CBufRef { buf, offset },
                offset: offset_reg,
                mode,
                mem_type,
            })
        })
        .boxed()
}

/// Any operation of the four-bit operation field, or a compare-and-swap when `cas`.
fn any_atom_op(cas: bool) -> BoxedStrategy<AtomOp> {
    if cas {
        any_gpr().prop_map(|cmpr| AtomOp::CmpExch { cmpr }).boxed()
    } else {
        select(&AtomOp::ENCODED[..]).boxed()
    }
}

/// `RED`, `ATOMG` in both forms and its compare-and-swap, and `ATOMS` likewise.
fn atoms() -> BoxedStrategy<Op> {
    let reg_dst = || any_gpr().prop_map(AtomDst::Reg);
    let red = (
        any_gpr().prop_map(|ignored| AtomDst::Reduction { ignored }),
        any_addr(ST_GLOBAL_UNIFORM),
        // `RED`'s three-bit field holds every operation but the exchange.
        select(&AtomOp::ENCODED[..AtomOp::ENCODED.len() - 1]),
        any_global(),
        Just(Pred::PT),
    );
    let atomg = (
        reg_dst(),
        prop_oneof![any_addr(GLOBAL), any_addr(ATOMG_UNIFORM)],
        any_atom_op(false),
        any_global(),
        any_pred(),
    );
    let atomg_cas = (
        reg_dst(),
        any_addr(GLOBAL),
        any_atom_op(true),
        any_global(),
        any_pred(),
    );
    let shared = |addr: BoxedStrategy<MemAddr>, cas: bool| {
        (
            reg_dst(),
            addr,
            any_atom_op(cas),
            any_stride().prop_map(MemSpace::Shared),
            Just(Pred::PT),
        )
    };
    let atoms = shared(
        prop_oneof![any_addr(AddrLayout::PLAIN), any_addr(ST_UNIFORM_PLAIN)].boxed(),
        false,
    );
    let atoms_cas = shared(any_addr(AddrLayout::PLAIN), true);
    let forms = prop_oneof![
        red.prop_map(|(dst, addr, op, space, pred_dst)| (dst, addr, op, space, pred_dst)),
        atomg,
        atomg_cas,
        atoms,
        atoms_cas,
    ];
    (forms, any_gpr(), select(AtomType::ALL))
        .prop_map(|((dst, addr, op, space, pred_dst), data, atom_type)| {
            Op::Atom(OpAtom {
                dst,
                addr,
                data,
                op,
                atom_type,
                space,
                pred_dst,
            })
        })
        .boxed()
}

fn atoms_popc() -> BoxedStrategy<Op> {
    (any_gpr(), any_stride())
        .prop_map(|(addr, stride)| Op::AtomsPopc(OpAtomsPopc { addr, stride }))
        .boxed()
}

fn cctl() -> BoxedStrategy<Op> {
    (select(CCtlOp::ALL), any_gpr(), any::<i32>())
        .prop_map(|(op, addr, offset)| Op::CCtl(OpCCtl { op, addr, offset }))
        .boxed()
}

fn membar() -> BoxedStrategy<Op> {
    (select(MemBarScope::ALL), any::<bool>(), any::<bool>())
        .prop_map(|(scope, mmio, sc)| Op::MemBar(OpMemBar { scope, mmio, sc }))
        .boxed()
}

fn ldgsts() -> BoxedStrategy<Op> {
    let half = 1_i16 << (LDGSTS_GLOBAL_OFFSET.width - 1);
    let descriptors = 1_u8 << LDGSTS_DESCRIPTOR.width;
    (
        (any_gpr(), any::<i16>(), any_gpr()),
        -half..half,
        0..descriptors,
        any_pred_src(),
    )
        .prop_map(
            |((shared, shared_offset, global), global_offset, descriptor, enable)| {
                Op::LdGsts(OpLdGsts {
                    shared,
                    shared_offset,
                    global,
                    global_offset,
                    descriptor,
                    enable,
                })
            },
        )
        .boxed()
}
