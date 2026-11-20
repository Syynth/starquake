//! Every refusal with its exact error and message, the order of the checks with two faults
//! present at once, and each limit at its edge on both sides.

use super::*;
use crate::decode::{AtomCmpSrc, OpLdGsts};

/// What translating `op`, through its word, is refused with.
fn refused(op: Op) -> Option<MemoryUnsupported> {
    match build_op(op) {
        Err(MemoryError::Unsupported(what)) => Some(what),
        _ => None,
    }
}

/// What translating `op`, built by hand (no word), is refused with.
fn refused_by_hand(op: Op) -> Option<MemoryUnsupported> {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    match translate_memory(&mut builder, &Instr::new(op), GuardMode::Honour) {
        Err(MemoryError::Unsupported(what)) => Some(what),
        _ => None,
    }
}

/// Whether `op` translates.
fn accepted(op: Op) -> bool {
    build_op(op).is_ok()
}

/// The full message of `op`'s refusal.
fn message(op: Op) -> String {
    build_op(op).map_or_else(|error| error.to_string(), |_| "translated".to_owned())
}

const SHARED: MemSpace = MemSpace::Shared(OffsetStride::X1);

fn ldc(dst: u8, mem_type: MemType) -> OpLdc {
    OpLdc {
        dst: r(dst),
        cbuf: cbuf(1, 0x10),
        offset: Gpr::RZ,
        mode: LdcMode::Indexed,
        mem_type,
    }
}

fn membar(scope: MemBarScope, mmio: bool, sc: bool) -> Op {
    Op::MemBar(OpMemBar { scope, mmio, sc })
}

fn copy(shared: u8, global: u8, enable: PredSrc) -> Op {
    Op::LdGsts(OpLdGsts {
        shared: r(shared),
        shared_offset: 0,
        global: r(global),
        global_offset: 0,
        descriptor: 0,
        enable,
    })
}

/// A `RED` with every field PR #4588 refuses set or clear as given.
fn red_with(enabled: bool, uniform: u8, ignored: u8, reg_64: bool, order: MemOrder) -> Op {
    let base = red(AtomOp::Add, AtomType::U32, 2, true, 0, 4);
    Op::Atom(OpAtom {
        dst: AtomDst::Reduction {
            ignored: r(ignored),
        },
        addr: MemAddr {
            reg_64,
            ..with_uniform(2, false, uniform, true, enabled, 0)
        },
        space: global_with(order),
        ..base
    })
}

/// A compare-and-swap in `space` at `addr`.
fn cas(space: MemSpace, addr: MemAddr) -> Op {
    Op::Atom(atom(
        1,
        AtomOp::CmpExch(AtomCmpSrc::Separate { cmpr: r(3) }),
        AtomType::U32,
        space,
        addr,
        4,
    ))
}

/// The system-scope order, which the PR's translator does not read (bit 80).
const SYSTEM: MemOrder = MemOrder::Strong(MemScope::System);

/// Each refusal's exact message, one case per variant (part 1 of 3).
#[test]
fn every_refusal_has_its_message_1() {
    let cases: Vec<(Op, MemoryUnsupported, &str)> = vec![
        (
            cas(GLOBAL, plain(2, true, 0)),
            MemoryUnsupported::CompareAndSwap(Space::Global),
            "a global compare-and-swap atomic, which neither reference translates,",
        ),
        (
            cas(SHARED, plain(2, false, 0)),
            MemoryUnsupported::CompareAndSwap(Space::Shared),
            "a shared compare-and-swap atomic, which neither reference translates,",
        ),
        (
            Op::St(st(
                1,
                MemType::B32,
                LOCAL,
                with_uniform(2, false, 63, false, true, 0),
            )),
            MemoryUnsupported::UniformStore(Space::Local),
            "the uniform-address form of a local store",
        ),
        (
            Op::St(st(
                1,
                MemType::B32,
                SHARED,
                with_uniform(2, false, 63, false, false, 0),
            )),
            MemoryUnsupported::UniformStore(Space::Shared),
            "the uniform-address form of a shared store",
        ),
        (
            Op::Atom(atom(
                1,
                AtomOp::Add,
                AtomType::U32,
                SHARED,
                with_uniform(2, false, 63, false, false, 0),
                3,
            )),
            MemoryUnsupported::UniformSharedAtomic,
            "the uniform-address form of a shared atomic",
        ),
        (
            Op::Ldc(OpLdc {
                cbuf: CBufRef {
                    buf: CBuf::BindlessUniform(ur(4)),
                    offset: 0,
                },
                ..ldc(1, MemType::B32)
            }),
            MemoryUnsupported::BindlessConstantLoad,
            "a bindless ldc",
        ),
        (
            Op::Ldc(OpLdc {
                mode: LdcMode::IndexedLinear,
                ..ldc(1, MemType::B32)
            }),
            MemoryUnsupported::ConstantLoadMode(LdcMode::IndexedLinear),
            "an ldc with an indexing mode other than plain indexing",
        ),
        (
            Op::Ldc(ldc(3, MemType::B64)),
            MemoryUnsupported::ConstantLoadDestination {
                dst: r(3),
                words: 2,
            },
            "an ldc destination of 2 registers at r3",
        ),
        (
            Op::Ld(OpLd {
                pred_dst: p(2),
                ..ldg(1, MemType::B32, plain(2, true, 0))
            }),
            MemoryUnsupported::LoadResidency(p(2)),
            "an ldg predicate destination p2",
        ),
        (
            Op::Ld(ldg(
                1,
                MemType::B32,
                with_uniform(2, true, 63, false, true, 0),
            )),
            MemoryUnsupported::GlobalUniformAddress {
                reg: UniformReg::URZ,
                pair: false,
            },
            "a global address in uniform register urz",
        ),
    ];
    for (op, what, text) in cases {
        assert_eq!(refused(op), Some(what), "{op}");
        assert_eq!(message(op), format!("{text} is not supported"), "{op}");
    }
}

/// Each refusal's exact message, one case per variant (part 2 of 3).
#[test]
fn every_refusal_has_its_message_2() {
    let cases: Vec<(Op, MemoryUnsupported, &str)> = vec![
        (
            Op::Ld(ldg(
                1,
                MemType::B32,
                with_uniform(2, true, 61, true, true, 0),
            )),
            MemoryUnsupported::GlobalUniformAddress {
                reg: ur(61),
                pair: true,
            },
            "a global address in uniform register pair ur61",
        ),
        (
            Op::Ld(ldg(1, MemType::B32, plain(3, true, 0))),
            MemoryUnsupported::GlobalRegisterAddress(r(3)),
            "a global address register pair at r3",
        ),
        (
            Op::Ld(lds(5, MemType::B64, 2, OffsetStride::X1, 0)),
            MemoryUnsupported::LoadDestination {
                space: Space::Shared,
                dst: r(5),
                words: 2,
            },
            "a shared load destination of 2 registers at r5",
        ),
        (
            Op::St(st(6, MemType::B128, LOCAL, plain(2, false, 0))),
            MemoryUnsupported::StoreData {
                space: Space::Local,
                data: r(6),
                words: 4,
            },
            "local store data of 4 registers at r6",
        ),
        (
            Op::Ld(OpLd {
                space: SHARED,
                ..ldg(1, MemType::B32, with_uniform(2, false, 0, false, true, 0))
            }),
            MemoryUnsupported::UniformAddress(Space::Shared),
            "a uniform register in a shared address",
        ),
        (
            Op::Atom(atom(
                1,
                AtomOp::Add,
                AtomType::F32,
                GLOBAL,
                plain(2, true, 0),
                3,
            )),
            MemoryUnsupported::AtomicType {
                space: Space::Global,
                atom_type: AtomType::F32,
            },
            "a global atomic of type f32",
        ),
        (
            Op::Atom(OpAtom {
                pred_dst: p(1),
                ..atom(1, AtomOp::Add, AtomType::U32, GLOBAL, plain(2, true, 0), 3)
            }),
            MemoryUnsupported::AtomicPredicateOutput(p(1)),
            "an atomg predicate destination p1",
        ),
        (
            red_with(true, 0, 0, false, MemOrder::Weak),
            MemoryUnsupported::ReductionUniformAddress,
            "a red with a uniform address",
        ),
        (
            membar(MemBarScope::Sm, false, false),
            MemoryUnsupported::BarrierScope(MemBarScope::Sm),
            "a membar at the sm scope",
        ),
        (
            copy(6, 5, ps(0, false)),
            MemoryUnsupported::CopyRegisters {
                shared: r(6),
                global: r(5),
            },
            "ldgsts registers r6 (shared) and r5 (global pair)",
        ),
    ];
    for (op, what, text) in cases {
        assert_eq!(refused(op), Some(what), "{op}");
        assert_eq!(message(op), format!("{text} is not supported"), "{op}");
    }
}

/// Each refusal's exact message, one case per variant (part 3 of 3).
#[test]
fn every_refusal_has_its_message_3() {
    let cases: Vec<(Op, MemoryUnsupported, &str)> = vec![
        (
            Op::St(st(
                1,
                MemType::B32,
                global_with(MemOrder::Strong(MemScope::System)),
                plain(2, true, 0),
            )),
            MemoryUnsupported::SystemOrder,
            "a global access with the system-scope memory order",
        ),
        (
            Op::Ld(OpLd {
                space: SHARED,
                ..ldg(1, MemType::B32, with_uniform(2, false, 1, false, false, 0))
            }),
            MemoryUnsupported::SharedUniformField(ur(1)),
            "an lds uniform register field of ur1",
        ),
        (
            Op::Ld(OpLd {
                zd: true,
                ..lds(1, MemType::B32, 2, OffsetStride::X1, 0)
            }),
            MemoryUnsupported::SharedLoadZd,
            "an lds with bit 87 set",
        ),
        (
            red_with(false, 0, 1, false, MemOrder::Weak),
            MemoryUnsupported::ReductionDestination(r(1)),
            "a red destination field of r1",
        ),
        (
            red_with(false, 2, 0, false, MemOrder::Weak),
            MemoryUnsupported::ReductionUniformField(ur(2)),
            "a red uniform register field of ur2",
        ),
        (
            red_with(false, 0, 0, true, MemOrder::Weak),
            MemoryUnsupported::ReductionRegister64,
            "a red with bit 90 set",
        ),
        (
            membar(MemBarScope::Cta, true, false),
            MemoryUnsupported::BarrierMmio,
            "a membar.mmio",
        ),
        (
            membar(MemBarScope::Cta, false, true),
            MemoryUnsupported::BarrierSc,
            "a membar.sc",
        ),
    ];
    for (op, what, text) in cases {
        assert_eq!(refused(op), Some(what), "{op}");
        assert_eq!(message(op), format!("{text} is not supported"), "{op}");
    }
}

/// A local atomic has no encoding, so only a hand-built operation reaches its refusal.
#[test]
fn a_local_atomic_is_refused() {
    let op = Op::Atom(atom(
        1,
        AtomOp::Add,
        AtomType::U32,
        LOCAL,
        plain(2, false, 0),
        3,
    ));
    assert_eq!(refused_by_hand(op), Some(MemoryUnsupported::LocalAtomic));
    assert_eq!(
        MemoryError::from(MemoryUnsupported::LocalAtomic).to_string(),
        "a local atomic, which no encoding has, is not supported"
    );
}

/// `LDL` refuses the uniform address on, and a uniform register field other than 0 with it
/// off (`"native local uniform address"`, `Take(32, 6) == 0 && Take(91, 1) == 0`).
#[test]
fn ldl_refuses_any_uniform_field() {
    let local = |uniform, enabled| {
        Op::Ld(OpLd {
            space: LOCAL,
            ..ldg(
                1,
                MemType::B32,
                with_uniform(2, false, uniform, false, enabled, 0),
            )
        })
    };
    assert!(accepted(local(0, false)));
    for (uniform, enabled) in [(0, true), (1, false), (63, false), (63, true)] {
        assert_eq!(
            refused(local(uniform, enabled)),
            Some(MemoryUnsupported::UniformAddress(Space::Local)),
            "{uniform} {enabled}"
        );
    }
}

/// Every atomic type but `U32` and `S32` is refused, in shared and global memory and in
/// `RED`.
#[test]
fn atomic_types_past_s32_are_refused() {
    for &atom_type in AtomType::ALL {
        let ok = matches!(atom_type, AtomType::U32 | AtomType::I32);
        let shared = Op::Atom(atom(
            1,
            AtomOp::Add,
            atom_type,
            SHARED,
            plain(2, false, 0),
            3,
        ));
        let global = Op::Atom(atom(
            1,
            AtomOp::Add,
            atom_type,
            GLOBAL,
            plain(2, true, 0),
            3,
        ));
        let reduction = Op::Atom(red(AtomOp::Add, atom_type, 2, true, 0, 3));
        for (op, space) in [
            (shared, Space::Shared),
            (global, Space::Global),
            (reduction, Space::Global),
        ] {
            if ok {
                assert!(accepted(op), "{op}");
            } else {
                assert_eq!(
                    refused(op),
                    Some(MemoryUnsupported::AtomicType { space, atom_type }),
                    "{op}"
                );
            }
        }
    }
}

/// Every `LDC` mode but plain indexing is refused.
#[test]
fn ldc_modes_past_indexed_are_refused() {
    for &mode in LdcMode::ALL {
        let op = Op::Ldc(OpLdc {
            mode,
            ..ldc(1, MemType::B32)
        });
        if mode == LdcMode::Indexed {
            assert!(accepted(op));
        } else {
            assert_eq!(refused(op), Some(MemoryUnsupported::ConstantLoadMode(mode)));
        }
    }
}

/// A bindless `LDC` is refused whatever its handle: odd, even or `URZ`.
#[test]
fn bindless_ldc_is_refused_for_every_handle() {
    for handle in [0, 1, 4, 61, 63] {
        let op = Op::Ldc(OpLdc {
            cbuf: CBufRef {
                buf: CBuf::BindlessUniform(ur(handle)),
                offset: 4,
            },
            ..ldc(1, MemType::B32)
        });
        assert_eq!(
            refused(op),
            Some(MemoryUnsupported::BindlessConstantLoad),
            "ur{handle}"
        );
    }
}

/// `MEMBAR`'s scopes: only `SM` is refused.
#[test]
fn only_the_sm_barrier_scope_is_refused() {
    for &scope in MemBarScope::ALL {
        let op = membar(scope, false, false);
        if scope == MemBarScope::Sm {
            assert_eq!(refused(op), Some(MemoryUnsupported::BarrierScope(scope)));
        } else {
            assert!(accepted(op), "{op}");
        }
    }
}

/// Each global form refuses the system-scope order and takes the others.
#[test]
fn the_system_order_is_refused_in_every_global_form() {
    let system = global_with(MemOrder::Strong(MemScope::System));
    let gpu = global_with(MemOrder::Strong(MemScope::Gpu));
    let forms = |space| {
        [
            Op::Ld(OpLd {
                space,
                ..ldg(1, MemType::B32, plain(2, true, 0))
            }),
            Op::St(st(1, MemType::B32, space, plain(2, true, 0))),
            Op::Atom(atom(
                1,
                AtomOp::Add,
                AtomType::U32,
                space,
                plain(2, true, 0),
                3,
            )),
            red_with(false, 0, 0, false, MemOrder::Weak).with_order(space),
        ]
    };
    for op in forms(system) {
        assert_eq!(refused(op), Some(MemoryUnsupported::SystemOrder), "{op}");
    }
    for op in forms(gpu) {
        assert!(accepted(op), "{op}");
    }
}

trait WithOrder {
    fn with_order(self, space: MemSpace) -> Self;
}

impl WithOrder for Op {
    fn with_order(self, space: MemSpace) -> Self {
        match self {
            Self::Atom(atom) => Self::Atom(OpAtom { space, ..atom }),
            other => other,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Edges.

/// A wide load's destination at each edge, for each space: a pair at `R252` (ends at `R253`)
/// and at `RZ` is taken, at `R253` (odd) and `R254` (ends at `RZ`) refused; a quad at `R248`
/// and `RZ` is taken, at `R250` and `R252` (ends at `RZ`) refused.
#[test]
fn load_destinations_at_their_edges() {
    let cases = [
        (MemType::B64, 252, true),
        (MemType::B64, 253, false),
        (MemType::B64, 254, false),
        (MemType::B64, 255, true),
        (MemType::B128, 248, true),
        (MemType::B128, 250, false),
        (MemType::B128, 252, false),
        (MemType::B128, 255, true),
        (MemType::B32, 253, true),
    ];
    for (mem_type, dst, ok) in cases {
        let loads = [
            (ldg(dst, mem_type, plain(2, true, 0)), Space::Global),
            (lds(dst, mem_type, 2, OffsetStride::X1, 0), Space::Shared),
            (ldl(dst, mem_type, 2, 0), Space::Local),
        ];
        for (op, space) in loads {
            let op = Op::Ld(op);
            if ok {
                assert!(accepted(op), "{op}");
            } else {
                assert_eq!(
                    refused(op),
                    Some(MemoryUnsupported::LoadDestination {
                        space,
                        dst: r(dst),
                        words: words_of(mem_type),
                    }),
                    "{op}"
                );
            }
        }
    }
}

/// A wide store's data at each edge, for each space, as for loads.
#[test]
fn store_data_at_their_edges() {
    let cases = [
        (MemType::B64, 252, true),
        (MemType::B64, 253, false),
        (MemType::B64, 254, false),
        (MemType::B64, 255, true),
        (MemType::B128, 248, true),
        (MemType::B128, 250, false),
        (MemType::B128, 252, false),
        (MemType::B128, 255, true),
    ];
    for (mem_type, data, ok) in cases {
        for (space, addr, which) in [
            (GLOBAL, plain(2, true, 0), Space::Global),
            (SHARED, plain(2, false, 0), Space::Shared),
            (LOCAL, plain(2, false, 0), Space::Local),
        ] {
            let op = Op::St(st(data, mem_type, space, addr));
            if ok {
                assert!(accepted(op), "{op}");
            } else {
                assert_eq!(
                    refused(op),
                    Some(MemoryUnsupported::StoreData {
                        space: which,
                        data: r(data),
                        words: words_of(mem_type),
                    }),
                    "{op}"
                );
            }
        }
    }
}

/// `LDC`'s destination edges: a pair below 254 and even (`R252` taken, `R253` and `R254`
/// refused, `RZ` refused too), a quad below 252 and a multiple of four (`R248` taken, `R250` and
/// `R252` refused).
#[test]
fn ldc_destinations_at_their_edges() {
    for (mem_type, dst, ok) in [
        (MemType::B64, 252, true),
        (MemType::B64, 253, false),
        (MemType::B64, 254, false),
        (MemType::B64, 255, false),
        (MemType::B128, 248, true),
        (MemType::B128, 250, false),
        (MemType::B128, 252, false),
        (MemType::B32, 255, true),
    ] {
        let op = Op::Ldc(ldc(dst, mem_type));
        if ok {
            assert!(accepted(op), "{op}");
        } else {
            assert_eq!(
                refused(op),
                Some(MemoryUnsupported::ConstantLoadDestination {
                    dst: r(dst),
                    words: words_of(mem_type),
                }),
                "{op}"
            );
        }
    }
}

/// The global address register pair's edges: `R252` taken, `R254` (ends at `RZ`) and `R3`
/// (odd) refused, `RZ` and an odd single register taken.
#[test]
fn global_register_pairs_at_their_edges() {
    for (reg, pair, ok) in [
        (252, true, true),
        (254, true, false),
        (3, true, false),
        (255, true, true),
        (3, false, true),
    ] {
        let op = Op::Ld(ldg(1, MemType::B32, plain(reg, pair, 0)));
        if ok {
            assert!(accepted(op), "{op}");
        } else {
            assert_eq!(
                refused(op),
                Some(MemoryUnsupported::GlobalRegisterAddress(r(reg)))
            );
        }
    }
}

/// The uniform address's edges: a word at `UR62` taken and at `URZ` refused; a pair at `UR60`
/// taken, at `UR62` (ends at `URZ`) and `UR61` (odd) refused; with the uniform address off,
/// `URZ` is taken.
#[test]
fn global_uniform_registers_at_their_edges() {
    for (uniform, wide, enabled, ok) in [
        (62, false, true, true),
        (63, false, true, false),
        (60, true, true, true),
        (62, true, true, false),
        (61, true, true, false),
        (63, true, false, true),
    ] {
        let op = Op::Ld(ldg(
            1,
            MemType::B32,
            with_uniform(2, true, uniform, wide, enabled, 0),
        ));
        if ok {
            assert!(accepted(op), "{op}");
        } else {
            assert_eq!(
                refused(op),
                Some(MemoryUnsupported::GlobalUniformAddress {
                    reg: ur(uniform),
                    pair: wide,
                })
            );
        }
    }
}

/// `LDGSTS`'s register edges: a global pair at `R252` taken, at `R254` (ends at `RZ`), `R5`
/// (odd) and `RZ` refused; a shared register at `R254` taken and at `RZ` refused.
#[test]
fn copy_registers_at_their_edges() {
    for (shared, global, ok) in [
        (6, 252, true),
        (6, 254, false),
        (6, 5, false),
        (6, 255, false),
        (254, 4, true),
        (255, 4, false),
    ] {
        let op = copy(shared, global, ps(0, true));
        if ok {
            assert!(accepted(op), "{op}");
        } else {
            assert_eq!(
                refused(op),
                Some(MemoryUnsupported::CopyRegisters {
                    shared: r(shared),
                    global: r(global),
                })
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Order: two faults at once, the first in the documented order reported.

/// Each pair of faults in one instruction reports the one the module docs order first (part
/// 1 of 3).
#[test]
fn faults_report_in_the_documented_order_1() {
    let cases: Vec<(Op, MemoryUnsupported)> = vec![
        // LDG: residency, uniform address, register address, destination, order.
        (
            Op::Ld(OpLd {
                pred_dst: p(0),
                ..ldg(1, MemType::B32, with_uniform(2, false, 63, false, true, 0))
            }),
            MemoryUnsupported::LoadResidency(p(0)),
        ),
        (
            Op::Ld(ldg(
                1,
                MemType::B32,
                with_uniform(3, true, 63, false, true, 0),
            )),
            MemoryUnsupported::GlobalUniformAddress {
                reg: UniformReg::URZ,
                pair: false,
            },
        ),
        (
            Op::Ld(ldg(3, MemType::B64, plain(3, true, 0))),
            MemoryUnsupported::GlobalRegisterAddress(r(3)),
        ),
        (
            Op::Ld(OpLd {
                space: global_with(SYSTEM),
                ..ldg(3, MemType::B64, plain(2, true, 0))
            }),
            MemoryUnsupported::LoadDestination {
                space: Space::Global,
                dst: r(3),
                words: 2,
            },
        ),
        // STG: address, data, order.
        (
            Op::St(st(3, MemType::B64, GLOBAL, plain(3, true, 0))),
            MemoryUnsupported::GlobalRegisterAddress(r(3)),
        ),
        (
            Op::St(st(3, MemType::B64, global_with(SYSTEM), plain(2, true, 0))),
            MemoryUnsupported::StoreData {
                space: Space::Global,
                data: r(3),
                words: 2,
            },
        ),
        // LDS: uniform address, destination, uniform field, bit 87.
        (
            Op::Ld(OpLd {
                space: SHARED,
                ..ldg(3, MemType::B64, with_uniform(2, false, 1, false, true, 0))
            }),
            MemoryUnsupported::UniformAddress(Space::Shared),
        ),
        (
            Op::Ld(OpLd {
                space: SHARED,
                ..ldg(3, MemType::B64, with_uniform(2, false, 1, false, false, 0))
            }),
            MemoryUnsupported::LoadDestination {
                space: Space::Shared,
                dst: r(3),
                words: 2,
            },
        ),
        (
            Op::Ld(OpLd {
                space: SHARED,
                zd: true,
                ..ldg(1, MemType::B32, with_uniform(2, false, 1, false, false, 0))
            }),
            MemoryUnsupported::SharedUniformField(ur(1)),
        ),
    ];
    for (op, what) in cases {
        assert_eq!(refused(op), Some(what), "{op}");
    }
}

/// Each pair of faults in one instruction reports the one the module docs order first (part
/// 2 of 3).
#[test]
fn faults_report_in_the_documented_order_2() {
    let cases: Vec<(Op, MemoryUnsupported)> = vec![
        // LDL: uniform address, destination.
        (
            Op::Ld(OpLd {
                space: LOCAL,
                ..ldg(3, MemType::B64, with_uniform(2, false, 0, false, true, 0))
            }),
            MemoryUnsupported::UniformAddress(Space::Local),
        ),
        // STS, STL: the uniform form, data.
        (
            Op::St(st(
                3,
                MemType::B64,
                SHARED,
                with_uniform(2, false, 63, false, true, 0),
            )),
            MemoryUnsupported::UniformStore(Space::Shared),
        ),
        (
            Op::St(st(
                3,
                MemType::B64,
                LOCAL,
                with_uniform(2, false, 63, false, true, 0),
            )),
            MemoryUnsupported::UniformStore(Space::Local),
        ),
        // LDC: bindless, mode, destination.
        (
            Op::Ldc(OpLdc {
                cbuf: CBufRef {
                    buf: CBuf::BindlessUniform(ur(2)),
                    offset: 0,
                },
                ..ldc(3, MemType::B64)
            }),
            MemoryUnsupported::BindlessConstantLoad,
        ),
        // ATOMG: compare-and-swap, predicate output, type, address, order.
        (
            Op::Atom(OpAtom {
                op: AtomOp::CmpExch(AtomCmpSrc::Separate { cmpr: r(5) }),
                pred_dst: p(1),
                ..atom(1, AtomOp::Add, AtomType::U64, GLOBAL, plain(2, true, 0), 4)
            }),
            MemoryUnsupported::CompareAndSwap(Space::Global),
        ),
        (
            Op::Atom(OpAtom {
                pred_dst: p(1),
                ..atom(1, AtomOp::Add, AtomType::U64, GLOBAL, plain(2, true, 0), 4)
            }),
            MemoryUnsupported::AtomicPredicateOutput(p(1)),
        ),
        (
            Op::Atom(atom(
                1,
                AtomOp::Add,
                AtomType::U64,
                GLOBAL,
                plain(3, true, 0),
                4,
            )),
            MemoryUnsupported::AtomicType {
                space: Space::Global,
                atom_type: AtomType::U64,
            },
        ),
        (
            Op::Atom(atom(
                1,
                AtomOp::Add,
                AtomType::U32,
                global_with(SYSTEM),
                plain(3, true, 0),
                4,
            )),
            MemoryUnsupported::GlobalRegisterAddress(r(3)),
        ),
        // ATOMS: the uniform form, type.
        (
            Op::Atom(atom(
                1,
                AtomOp::Add,
                AtomType::F32,
                SHARED,
                with_uniform(2, false, 63, false, true, 0),
                4,
            )),
            MemoryUnsupported::UniformSharedAtomic,
        ),
    ];
    for (op, what) in cases {
        assert_eq!(refused(op), Some(what), "{op}");
    }
}

/// Each pair of faults in one instruction reports the one the module docs order first (part
/// 3 of 3).
#[test]
fn faults_report_in_the_documented_order_3() {
    let cases: Vec<(Op, MemoryUnsupported)> = vec![
        // RED: uniform address, type, register address, destination field, uniform field,
        // order, bit 90.
        (
            Op::Atom(OpAtom {
                atom_type: AtomType::U64,
                ..match red_with(true, 0, 0, false, MemOrder::Weak) {
                    Op::Atom(atom) => atom,
                    _ => unreachable!(),
                }
            }),
            MemoryUnsupported::ReductionUniformAddress,
        ),
        (
            Op::Atom(OpAtom {
                addr: with_uniform(3, false, 0, true, false, 0),
                atom_type: AtomType::U64,
                ..red(AtomOp::Add, AtomType::U32, 2, true, 0, 4)
            }),
            MemoryUnsupported::AtomicType {
                space: Space::Global,
                atom_type: AtomType::U64,
            },
        ),
        (
            Op::Atom(OpAtom {
                addr: with_uniform(3, false, 0, true, false, 0),
                dst: AtomDst::Reduction { ignored: r(1) },
                ..red(AtomOp::Add, AtomType::U32, 2, true, 0, 4)
            }),
            MemoryUnsupported::GlobalRegisterAddress(r(3)),
        ),
        (
            red_with(false, 2, 1, true, SYSTEM),
            MemoryUnsupported::ReductionDestination(r(1)),
        ),
        (
            red_with(false, 2, 0, true, SYSTEM),
            MemoryUnsupported::ReductionUniformField(ur(2)),
        ),
        (
            red_with(false, 0, 0, true, SYSTEM),
            MemoryUnsupported::SystemOrder,
        ),
        // MEMBAR: scope, .MMIO, .SC.
        (
            membar(MemBarScope::Sm, true, true),
            MemoryUnsupported::BarrierScope(MemBarScope::Sm),
        ),
        (
            membar(MemBarScope::Gpu, true, true),
            MemoryUnsupported::BarrierMmio,
        ),
    ];
    for (op, what) in cases {
        assert_eq!(refused(op), Some(what), "{op}");
    }
}

// ---------------------------------------------------------------------------------------------
// The descriptor word (bit 76) and the fields only the global forms read.

/// A descriptor `LDG` or `STG` (bit 76, the uniform address on) with each field as given: the
/// exact word PR #4588's lowering reads as a descriptor when `pred` is `PT`, `wide` and
/// `reg_64` are set, the type is one it takes, the order weak, the eviction priority normal
/// and `reg` even and not `RZ`.
#[derive(Clone, Copy)]
struct Descriptor {
    store: bool,
    pred: PredSrc,
    wide: bool,
    mem_type: MemType,
    order: MemOrder,
    eviction: MemEvictionPriority,
    reg_64: bool,
    reg: u8,
    uniform: u8,
}

impl Descriptor {
    /// The accepted word: `R2:R3` and `UR8:UR9`, 32 bits.
    const fn exact(store: bool) -> Self {
        Self {
            store,
            pred: PredSrc::TRUE,
            wide: true,
            mem_type: MemType::B32,
            order: MemOrder::Weak,
            eviction: MemEvictionPriority::Normal,
            reg_64: true,
            reg: 2,
            uniform: 8,
        }
    }

    fn op(self) -> Op {
        let addr = with_uniform(self.reg, self.reg_64, self.uniform, self.wide, true, 0x10);
        let space = MemSpace::Global(GlobalAccess {
            order: self.order,
            eviction: self.eviction,
        });
        if self.store {
            Op::St(OpSt {
                bit76: true,
                ..st(8, self.mem_type, space, addr)
            })
        } else {
            Op::Ld(OpLd {
                bit76: true,
                pred: self.pred,
                space,
                ..ldg(8, self.mem_type, addr)
            })
        }
    }
}

/// One clearing step of the order test: the fault reported, and how to clear it.
type Step = (DescriptorFault, fn(&mut Descriptor));

/// Each condition of the lowering's descriptor word at its edge, both sides, in `LDG` and
/// `STG`: the exact word translates, and a word one field off it is refused with that field's
/// [`DescriptorFault`]. With the uniform address off, bit 76 changes nothing, so the same
/// fields translate.
#[test]
fn descriptor_words_at_each_edge() {
    for store in [false, true] {
        let exact = Descriptor::exact(store);
        assert!(accepted(exact.op()), "store {store}");
        // Bit 72: the uniform register a pair, or a single (even or odd) register.
        for uniform in [8, 9] {
            let single = Descriptor {
                wide: false,
                uniform,
                ..exact
            };
            assert_eq!(
                refused(single.op()),
                Some(MemoryUnsupported::DescriptorAddress(
                    DescriptorFault::UniformSingle
                )),
                "store {store} UR{uniform}"
            );
        }
        // The uniform pair: UR60 translates; UR62, which the lowering takes, is refused by the
        // native translator's bound.
        assert!(accepted(
            Descriptor {
                uniform: 60,
                ..exact
            }
            .op()
        ));
        assert_eq!(
            refused(
                Descriptor {
                    uniform: 62,
                    ..exact
                }
                .op()
            ),
            Some(MemoryUnsupported::GlobalUniformAddress {
                reg: ur(62),
                pair: true,
            })
        );
        // Bit 90: the register a pair, or a single register.
        assert_eq!(
            refused(
                Descriptor {
                    reg_64: false,
                    ..exact
                }
                .op()
            ),
            Some(MemoryUnsupported::DescriptorAddress(
                DescriptorFault::RegisterSingle
            ))
        );
        // The register: R0 and R252 (the lowering's own bound, `address + 2 >
        // reserved_register`, is not mirrored) translate, RZ is refused.
        for reg in [0, 252] {
            assert!(accepted(Descriptor { reg, ..exact }.op()), "R{reg}");
        }
        assert_eq!(
            refused(Descriptor { reg: 255, ..exact }.op()),
            Some(MemoryUnsupported::DescriptorAddress(
                DescriptorFault::ZeroRegister
            ))
        );
    }
    // The load predicate: PT translates; a register or a negated PT is refused.
    let exact = Descriptor::exact(false);
    for pred in [ps(0, false), ps(6, true), ps(7, true)] {
        assert_eq!(
            refused(Descriptor { pred, ..exact }.op()),
            Some(MemoryUnsupported::DescriptorAddress(
                DescriptorFault::LoadPredicate
            )),
            "{pred:?}"
        );
    }
}

/// The order, eviction priority and type of the lowering's descriptor word at each value, in
/// `LDG` and `STG`: the weak order, the normal eviction priority and the types the lowering
/// takes translate, every other value is refused with its [`DescriptorFault`], and the system
/// order with [`MemoryUnsupported::SystemOrder`] first.
#[test]
fn descriptor_orders_evictions_and_types() {
    for store in [false, true] {
        let exact = Descriptor::exact(store);
        // Every order: weak translates, the system scope is refused first by its own check.
        for &order in MemOrder::ALL {
            let expected = match order {
                MemOrder::Weak => None,
                MemOrder::Strong(MemScope::System) => Some(MemoryUnsupported::SystemOrder),
                MemOrder::Constant | MemOrder::Strong(_) => {
                    Some(MemoryUnsupported::DescriptorAddress(DescriptorFault::Order))
                }
            };
            assert_eq!(
                refused(Descriptor { order, ..exact }.op()),
                expected,
                "{order:?}"
            );
        }
        // Every eviction priority: normal translates.
        for &eviction in MemEvictionPriority::ALL {
            let expected = (eviction != MemEvictionPriority::Normal).then_some(
                MemoryUnsupported::DescriptorAddress(DescriptorFault::Eviction),
            );
            assert_eq!(
                refused(Descriptor { eviction, ..exact }.op()),
                expected,
                "{eviction:?}"
            );
        }
        // Every type: the load takes .u16, 32, 64 and 128 bits, the store .u8, .u16, 32 and
        // 64 bits.
        for &mem_type in MemType::ALL {
            let taken = if store {
                [MemType::U8, MemType::U16, MemType::B32, MemType::B64]
            } else {
                [MemType::U16, MemType::B32, MemType::B64, MemType::B128]
            };
            let expected = (!taken.contains(&mem_type)).then_some(
                MemoryUnsupported::DescriptorAddress(DescriptorFault::Type(mem_type)),
            );
            assert_eq!(
                refused(Descriptor { mem_type, ..exact }.op()),
                expected,
                "store {store} {mem_type:?}"
            );
        }
    }
}

/// With the uniform address off, bit 76 changes nothing: a word with every descriptor field
/// off translates, as the uniform register is neither added nor a descriptor.
#[test]
fn bit_76_without_the_uniform_address_is_not_a_descriptor() {
    for store in [false, true] {
        let addr = with_uniform(2, false, 9, false, false, 0);
        let space = MemSpace::Global(GlobalAccess {
            order: MemOrder::Constant,
            eviction: MemEvictionPriority::First,
        });
        let op = if store {
            Op::St(OpSt {
                bit76: true,
                ..st(8, MemType::I8, space, addr)
            })
        } else {
            Op::Ld(OpLd {
                bit76: true,
                pred: ps(0, false),
                space,
                ..ldg(8, MemType::I8, addr)
            })
        };
        assert!(accepted(op), "store {store}");
    }
}

/// Every descriptor fault present at once reports them in [`DescriptorFault`]'s order: each
/// step clears the fault just reported and the next is reported, until the word translates.
/// The checks before it (the register pair, the system order) come first.
#[test]
fn descriptor_faults_report_in_the_documented_order() {
    for store in [false, true] {
        let exact = Descriptor::exact(store);
        let mut word = Descriptor {
            pred: if store { PredSrc::TRUE } else { ps(0, false) },
            wide: false,
            mem_type: MemType::I8,
            order: MemOrder::Constant,
            eviction: MemEvictionPriority::First,
            reg_64: false,
            reg: 255,
            uniform: 9,
            ..exact
        };
        // An odd register pair and the system order come before every descriptor fault.
        let odd = Descriptor {
            reg_64: true,
            reg: 3,
            order: MemOrder::Strong(MemScope::System),
            ..word
        };
        assert_eq!(
            refused(odd.op()),
            Some(MemoryUnsupported::GlobalRegisterAddress(r(3)))
        );
        let system = Descriptor {
            order: MemOrder::Strong(MemScope::System),
            ..word
        };
        assert_eq!(refused(system.op()), Some(MemoryUnsupported::SystemOrder));
        let mut steps: Vec<Step> = Vec::new();
        if !store {
            steps.push((DescriptorFault::LoadPredicate, |d| d.pred = PredSrc::TRUE));
        }
        let rest: [Step; 6] = [
            (DescriptorFault::UniformSingle, |d: &mut Descriptor| {
                d.wide = true;
                d.uniform = 8;
            }),
            (DescriptorFault::Type(MemType::I8), |d: &mut Descriptor| {
                d.mem_type = MemType::B32;
            }),
            (DescriptorFault::Order, |d: &mut Descriptor| {
                d.order = MemOrder::Weak;
            }),
            (DescriptorFault::Eviction, |d: &mut Descriptor| {
                d.eviction = MemEvictionPriority::Normal;
            }),
            (DescriptorFault::RegisterSingle, |d: &mut Descriptor| {
                d.reg_64 = true;
            }),
            (DescriptorFault::ZeroRegister, |d: &mut Descriptor| {
                d.reg = 2;
            }),
        ];
        steps.extend(rest);
        for (fault, clear) in steps {
            assert_eq!(
                refused(word.op()),
                Some(MemoryUnsupported::DescriptorAddress(fault)),
                "store {store}"
            );
            clear(&mut word);
        }
        assert!(accepted(word.op()), "store {store}");
    }
}

/// Each descriptor fault's exact message.
#[test]
fn descriptor_faults_have_their_messages() {
    let exact = Descriptor::exact(false);
    let cases = [
        (
            Descriptor {
                pred: ps(0, false),
                ..exact
            },
            "a load predicate",
        ),
        (
            Descriptor {
                wide: false,
                ..exact
            },
            "a single uniform register",
        ),
        (
            Descriptor {
                mem_type: MemType::I16,
                ..exact
            },
            "type s16",
        ),
        (
            Descriptor {
                store: true,
                mem_type: MemType::B128,
                ..exact
            },
            "type b128",
        ),
        (
            Descriptor {
                order: MemOrder::Strong(MemScope::Gpu),
                ..exact
            },
            "a memory order other than weak",
        ),
        (
            Descriptor {
                eviction: MemEvictionPriority::Last,
                ..exact
            },
            "an eviction priority other than normal",
        ),
        (
            Descriptor {
                reg_64: false,
                ..exact
            },
            "a single address register",
        ),
        (
            Descriptor { reg: 255, ..exact },
            "rz as the address register",
        ),
    ];
    for (descriptor, field) in cases {
        assert_eq!(
            message(descriptor.op()),
            format!("a descriptor-addressed global access (bit 76) with {field} is not supported")
        );
    }
}

/// The fields only `LDG` reads (its load predicate and predicate destination) and bit 76 of
/// `LDG` and `STG` are refused in a hand-built `LDS`, `LDL`, `STS` and `STL`, which no
/// encoding has, with the exact variant; with them `PT` and clear the same operations
/// translate. All three present report the load predicate, then the predicate destination,
/// then bit 76, before the uniform address.
#[test]
fn global_only_fields_are_refused_by_hand() {
    let loads = [
        (Space::Shared, lds(1, MemType::B32, 2, OffsetStride::X1, 0)),
        (Space::Local, ldl(1, MemType::B32, 2, 0)),
    ];
    for (space, load) in loads {
        let fault = |field| Some(MemoryUnsupported::GlobalOnlyField { space, field });
        assert_eq!(refused_by_hand(Op::Ld(load)), None, "{space}");
        assert!(build_op(Op::Ld(load)).is_ok(), "{space}");
        for pred in [ps(0, false), ps(7, true)] {
            assert_eq!(
                refused_by_hand(Op::Ld(OpLd { pred, ..load })),
                fault(GlobalOnlyField::LoadPredicate),
                "{space} {pred:?}"
            );
        }
        assert_eq!(
            refused_by_hand(Op::Ld(OpLd {
                pred_dst: p(0),
                ..load
            })),
            fault(GlobalOnlyField::PredicateDestination)
        );
        assert_eq!(
            refused_by_hand(Op::Ld(OpLd {
                bit76: true,
                ..load
            })),
            fault(GlobalOnlyField::Bit76)
        );
        // Order: all three, then the last two, with the uniform address on as well.
        let uniform = with_uniform(2, false, 1, false, true, 0);
        let all = OpLd {
            pred: ps(0, false),
            pred_dst: p(0),
            bit76: true,
            addr: uniform,
            ..load
        };
        assert_eq!(
            refused_by_hand(Op::Ld(all)),
            fault(GlobalOnlyField::LoadPredicate)
        );
        assert_eq!(
            refused_by_hand(Op::Ld(OpLd {
                pred: PredSrc::TRUE,
                ..all
            })),
            fault(GlobalOnlyField::PredicateDestination)
        );
        assert_eq!(
            refused_by_hand(Op::Ld(OpLd {
                pred: PredSrc::TRUE,
                pred_dst: Pred::PT,
                ..all
            })),
            fault(GlobalOnlyField::Bit76)
        );
        assert_eq!(
            refused_by_hand(Op::Ld(OpLd {
                pred: PredSrc::TRUE,
                pred_dst: Pred::PT,
                bit76: false,
                ..all
            })),
            Some(MemoryUnsupported::UniformAddress(space))
        );
    }
}

/// Bit 76 is refused in a hand-built `STS` and `STL`, with the exact variant, and the same
/// stores translate with it clear; each [`GlobalOnlyField`] has its exact message.
#[test]
fn global_only_fields_are_refused_in_hand_built_stores() {
    for (space, mem_space) in [(Space::Shared, SHARED), (Space::Local, LOCAL)] {
        let store = st(4, MemType::B32, mem_space, plain(2, false, 0));
        assert_eq!(refused_by_hand(Op::St(store)), None, "{space}");
        assert!(build_op(Op::St(store)).is_ok(), "{space}");
        assert_eq!(
            refused_by_hand(Op::St(OpSt {
                bit76: true,
                ..store
            })),
            Some(MemoryUnsupported::GlobalOnlyField {
                space,
                field: GlobalOnlyField::Bit76,
            })
        );
    }
    assert_eq!(
        MemoryError::from(MemoryUnsupported::GlobalOnlyField {
            space: Space::Shared,
            field: GlobalOnlyField::LoadPredicate,
        })
        .to_string(),
        "a shared access with a load predicate, which only the global forms read, is not supported"
    );
    assert_eq!(
        MemoryError::from(MemoryUnsupported::GlobalOnlyField {
            space: Space::Local,
            field: GlobalOnlyField::PredicateDestination,
        })
        .to_string(),
        "a local access with a predicate destination, which only the global forms read, is not \
         supported"
    );
    assert_eq!(
        MemoryError::from(MemoryUnsupported::GlobalOnlyField {
            space: Space::Shared,
            field: GlobalOnlyField::Bit76,
        })
        .to_string(),
        "a shared access with bit 76, which only the global forms read, is not supported"
    );
}

/// A hand-built shared `RED` (`AtomDst::Reduction` in shared memory), which no encoding has,
/// is refused, before the uniform address; the same atomic with a destination translates.
#[test]
fn a_shared_reduction_is_refused_by_hand() {
    let shared = atom(1, AtomOp::Add, AtomType::U32, SHARED, plain(2, false, 0), 3);
    assert_eq!(refused_by_hand(Op::Atom(shared)), None);
    let reduction = OpAtom {
        dst: AtomDst::Reduction { ignored: r(0) },
        ..shared
    };
    assert_eq!(
        refused_by_hand(Op::Atom(reduction)),
        Some(MemoryUnsupported::SharedReduction)
    );
    assert_eq!(
        refused_by_hand(Op::Atom(OpAtom {
            addr: with_uniform(2, false, 1, false, true, 0),
            ..reduction
        })),
        Some(MemoryUnsupported::SharedReduction)
    );
    assert_eq!(
        MemoryError::from(MemoryUnsupported::SharedReduction).to_string(),
        "a shared reduction, which no encoding has, is not supported"
    );
}
