//! Tests of slice 6e, the memory group (`op::memory`): NAK's expected text for `LDC` and for
//! loads, stores and atomics, one hand-built word per form and per modifier bit, every value of
//! every field at its edges, the values and bits the group refuses and the order it refuses them
//! in, the divergences from PR #4588's table, and generated round trips of the group alone.

use proptest::prelude::*;
use proptest::strategy::Union;

use super::super::op::control::UNIFORM_BIT;
use super::super::op::memory::{self, *};
use super::*;
use crate::asm::{EncodeError, assemble};

/// A hand-built word's opcode and fields, and its expected text.
type Case = (u128, &'static [(u32, u128)], &'static str);
/// [`Case`] with fields built at run time.
type BuiltCase = (u128, Vec<(u32, u128)>, &'static str);
/// A hand-built word's opcode and fields, and the error that refuses it.
type RefusalCase = (u128, Vec<(u32, u128)>, DecodeError);

/// The operation a word decodes to.
fn op_of(word: Instruction) -> Op {
    decode(word).expect("decodes").op
}

/// The load a word decodes to.
fn ld_of(word: Instruction) -> OpLd {
    match op_of(word) {
        Op::Ld(ld) => Some(ld),
        _ => None,
    }
    .expect("a load")
}

/// The store a word decodes to.
fn st_of(word: Instruction) -> OpSt {
    match op_of(word) {
        Op::St(st) => Some(st),
        _ => None,
    }
    .expect("a store")
}

/// The atomic a word decodes to.
fn atom_of(word: Instruction) -> OpAtom {
    match op_of(word) {
        Op::Atom(atom) => Some(atom),
        _ => None,
    }
    .expect("an atomic")
}

/// The uniform register `ur2` as NAK's tests pass it: a 64-bit pair (`.E` set where the form has
/// it), bit 91 set.
fn ur2(wide: bool) -> UniformAddr {
    UniformAddr {
        reg: ur(2),
        wide,
        enabled: true,
    }
}

fn global(order: MemOrder, eviction: MemEvictionPriority) -> MemSpace {
    MemSpace::Global(GlobalAccess { order, eviction })
}

/// The atomic types of NAK's `test_ld_st_atom`, their `nvdisasm` text, and whether each is a
/// float and 64 bits wide.
const ATOM_TYPES: [(AtomType, &str, bool, bool); 7] = [
    (AtomType::F16v2, ".f16x2.rn", true, false),
    (AtomType::U32, "", false, false),
    (AtomType::I32, ".s32", false, false),
    (AtomType::F32, ".f32.ftz.rn", true, false),
    (AtomType::U64, ".64", false, true),
    (AtomType::I64, ".s64", false, true),
    (AtomType::F64, ".f64.rn", true, true),
];

/// The atomic operations of NAK's `test_ld_st_atom` and their `nvdisasm` text.
const ATOM_OPS: [(AtomOp, &str); 9] = [
    (AtomOp::Add, ".add"),
    (AtomOp::Min, ".min"),
    (AtomOp::Max, ".max"),
    (AtomOp::Inc, ".inc"),
    (AtomOp::Dec, ".dec"),
    (AtomOp::And, ".and"),
    (AtomOp::Or, ".or"),
    (AtomOp::Xor, ".xor"),
    (AtomOp::Exch, ".exch"),
];

/// NAK's `test_ld_st_atom` atomics on `addr` (printed `at`) in `space`, global or shared, and
/// their expected text: every type and operation, with and without a destination (the shared
/// space skips float and 64-bit types; no `RED` exchange).
fn ld_st_atom_atomics(addr: MemAddr, space: MemSpace, at: &str) -> Vec<(Op, String)> {
    let is_global = matches!(space, MemSpace::Global(_));
    let mut cases = Vec::new();
    for (atom_type, type_str, float, wide) in ATOM_TYPES {
        if !is_global && (float || wide) {
            continue;
        }
        let ops = if float { &ATOM_OPS[..3] } else { &ATOM_OPS[..] };
        for &(op, op_str) in ops {
            for use_dst in [true, false] {
                if !use_dst && op == AtomOp::Exch {
                    continue;
                }
                let (dst, expected) = match (is_global, use_dst) {
                    (true, true) => (
                        AtomDst::Reg(r(0)),
                        format!("atomg.e{op_str}.ef{type_str}.strong.sm pt, r0, {at}, r2;"),
                    ),
                    (true, false) => (
                        AtomDst::Reduction { ignored: Gpr::RZ },
                        format!("red.e{op_str}.ef{type_str}.strong.sm {at}, r2;"),
                    ),
                    (false, true) => (
                        AtomDst::Reg(r(0)),
                        format!("atoms{op_str}{type_str} r0, {at}, r2;"),
                    ),
                    (false, false) => (
                        AtomDst::Reg(Gpr::RZ),
                        format!("atoms{op_str}{type_str} rz, {at}, r2;"),
                    ),
                };
                let atom = Op::Atom(OpAtom {
                    dst,
                    addr,
                    data: r(2),
                    op,
                    atom_type,
                    space,
                    pred_dst: Pred::PT,
                });
                cases.push((atom, expected));
            }
        }
    }
    cases
}

/// NAK's `test_ld_st_atom`, ported as a fixture table (`nvdisasm_tests.rs` @ 52238762), for
/// SM86: `r0` loaded from, `r2` stored to and atomics on `[r4+ur2+offset]` (`r4.64` in the
/// global space), with the offsets `0x12`, `-0x1` and `0x4`, the shared space's `.x8` stride,
/// every atomic type and operation with and without a destination, and the text NAK checks
/// against `nvdisasm` (`.strong.sm` for the workgroup on SM 8.0 and later; the shared space
/// skips float and 64-bit atomics and the local space atomics; no `RED` exchange). The expected
/// strings are written out here, independently of the printers.
#[test]
fn ld_st_atom_text_matches_nak_nvdisasm_cases() {
    let spaces = [
        (
            global(MemOrder::Strong(MemScope::Cta), MemEvictionPriority::First),
            "r4.64",
            ("ldg.e.ef.strong.sm", ", p4"),
            "stg.e.ef.strong.sm",
        ),
        (MemSpace::Shared(OffsetStride::X1), "r4", ("lds", ""), "sts"),
        (
            MemSpace::Shared(OffsetStride::X8),
            "r4.x8",
            ("lds", ""),
            "sts",
        ),
        (
            MemSpace::Local(MemEvictionPriority::Normal),
            "r4",
            ("ldl", ""),
            "stl",
        ),
    ];
    let mut checked = 0;
    for (space, r4, (ld_str, pred_str), st_str) in spaces {
        let is_global = matches!(space, MemSpace::Global(_));
        for (offset, offset_str) in [(0x12, "0x12"), (-1, "-0x1"), (0x4, "0x4")] {
            let addr = MemAddr {
                reg: r(4),
                reg_64: is_global,
                uniform: Some(ur2(is_global)),
                offset,
            };
            let at = format!("[{r4}+ur2+{offset_str}]");
            let pred = if is_global {
                PredSrc {
                    reg: p(4),
                    negated: false,
                }
            } else {
                PredSrc::TRUE
            };
            let ld = Op::Ld(OpLd {
                dst: r(0),
                addr,
                mem_type: MemType::B32,
                space,
                pred,
                pred_dst: Pred::PT,
                zd: false,
                bit76: false,
            });
            let st = Op::St(OpSt {
                data: r(2),
                addr,
                mem_type: MemType::B32,
                space,
                bit76: false,
            });
            let mut cases = vec![
                (ld, format!("{ld_str} r0, {at}{pred_str};")),
                (st, format!("{st_str} {at}, r2;")),
            ];
            if !matches!(space, MemSpace::Local(_)) {
                cases.extend(ld_st_atom_atomics(addr, space, &at));
            }
            for (op, expected) in cases {
                assert_eq!(round_text(Instr::new(op)), expected);
                checked += 1;
            }
        }
    }
    // Global: 3 offsets x (2 + 3 float types x 3 operations x 2 + 4 integer types x 17);
    // shared: 6 address cases x (2 + 2 types x 17); local: 3 x 2.
    assert_eq!(checked, 3 * (2 + 18 + 68) + 6 * (2 + 34) + 3 * 2);
}

/// NAK's `test_ldc`, ported (`nvdisasm_tests.rs` @ 52238762), for SM86's vector `LDC` (the
/// uniform `ULDC` cases are slice 6d's): a bound and a bindless buffer, every type NAK tests
/// for a register destination (not `.128`), with and without an offset register.
#[test]
fn ldc_text_matches_nak_nvdisasm_cases() {
    let cbufs = [
        (CBuf::Binding(5), "c[0x5]"),
        (CBuf::BindlessUniform(ur(2)), "cx[ur2]"),
    ];
    let mem_types = [
        (MemType::U8, ".u8"),
        (MemType::I8, ".s8"),
        (MemType::U16, ".u16"),
        (MemType::I16, ".s16"),
        (MemType::B32, ""),
        (MemType::B64, ".64"),
    ];
    let mut checked = 0;
    for (buf, buf_str) in cbufs {
        for (mem_type, type_str) in mem_types {
            for (offset, offset_str) in [(Gpr::RZ, ""), (r(6), "r6+")] {
                let ldc = Op::Ldc(OpLdc {
                    dst: r(4),
                    cbuf: CBufRef { buf, offset: 0x248 },
                    offset,
                    mode: LdcMode::Indexed,
                    mem_type,
                });
                assert_eq!(
                    round_text(Instr::new(ldc)),
                    format!("ldc{type_str} r4, {buf_str}[{offset_str}0x248];")
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 24);
}

/// Hand-built words of `LDG` in both forms, `LDL` and `LDS`: opcode, fields at distinct values, text.
fn load_words() -> Vec<Case> {
    vec![
        // LDG, no uniform register: dst 1, addr 2, offset 0x123456, .E (72), type 5 (73),
        // order 7 (77), predicate destination 3 (81), eviction 2 (84).
        (
            0x381,
            &[
                (16, 1),
                (24, 2),
                (40, 0x12_3456),
                (72, 1),
                (73, 5),
                (77, 7),
                (81, 3),
                (84, 2),
            ],
            "ldg.e.el.64.strong.gpu p3, r1, [r2+0x123456];",
        ),
        // LDG: dst 1, addr 2, ur3 (32), offset -0x10 (0xfffff0), load predicate field 2 (P5)
        // negated (67), .E (72), type 6, order 0xa, predicate destination PT, eviction 3, the
        // register's 64-bit flag (90), bit 91.
        (
            0x981,
            &[
                (16, 1),
                (24, 2),
                (32, 3),
                (40, 0xff_fff0),
                (64, 2),
                (67, 1),
                (72, 1),
                (73, 6),
                (77, 0xa),
                (81, 7),
                (84, 3),
                (90, 1),
                (91, 1),
            ],
            "ldg.e.lu.128.strong.sys r1, [r2.64+ur3+-0x10], !p5;",
        ),
        // LDL: dst 4, addr 5, ur6, offset 0x20, type 1, eviction 4, bit 91.
        (
            0x983,
            &[
                (16, 4),
                (24, 5),
                (32, 6),
                (40, 0x20),
                (73, 1),
                (84, 4),
                (91, 1),
            ],
            "ldl.eu.s8 r4, [r5+ur6+0x20];",
        ),
        // LDS: dst 7, addr 8, ur9, offset 0x30, type 3, stride 3 (78), bit 87, bit 91.
        (
            0x984,
            &[
                (16, 7),
                (24, 8),
                (32, 9),
                (40, 0x30),
                (73, 3),
                (78, 3),
                (87, 1),
                (91, 1),
            ],
            "lds.s16.zd r7, [r8.x16+ur9+0x30];",
        ),
    ]
}

/// Hand-built words of `STG`, `STL` and `STS`, each in both forms: opcode, fields at distinct values, text.
fn store_words() -> Vec<Case> {
    vec![
        // STG, no uniform register: addr 10, data 11 (32), offset 0x40, .E, type 2, order 4,
        // eviction 5.
        (
            0x386,
            &[
                (24, 10),
                (32, 11),
                (40, 0x40),
                (72, 1),
                (73, 2),
                (77, 4),
                (84, 5),
            ],
            "stg.e.na.u16.constant [r10+0x40], r11;",
        ),
        // STG: addr 12, data 13, offset 0x50, ur14 (64), .E, type 4, bit 76, order 5, eviction
        // 0, the register's 64-bit flag, bit 91.
        (
            0x986,
            &[
                (24, 12),
                (32, 13),
                (40, 0x50),
                (64, 14),
                (72, 1),
                (73, 4),
                (76, 1),
                (77, 5),
                (90, 1),
                (91, 1),
            ],
            "stg.e.ef.strong.sm [r12.64+ur14+0x50], r13;",
        ),
        // STL, no uniform register: addr 15, data 16, offset 0x60, type 0, eviction 1.
        (
            0x387,
            &[(24, 15), (32, 16), (40, 0x60), (84, 1)],
            "stl.u8 [r15+0x60], r16;",
        ),
        // STL: addr 17, data 18, offset 0x70, ur19, type 5, eviction 2, bit 91.
        (
            0x987,
            &[
                (24, 17),
                (32, 18),
                (40, 0x70),
                (64, 19),
                (73, 5),
                (84, 2),
                (91, 1),
            ],
            "stl.el.64 [r17+ur19+0x70], r18;",
        ),
        // STS, no uniform register: addr 20, data 21, offset 0x80, type 6, stride 1.
        (
            0x388,
            &[(24, 20), (32, 21), (40, 0x80), (73, 6), (78, 1)],
            "sts.128 [r20.x4+0x80], r21;",
        ),
        // STS: addr 22, data 23, offset 0x90, ur24, type 4, stride 2, bit 91.
        (
            0x988,
            &[
                (24, 22),
                (32, 23),
                (40, 0x90),
                (64, 24),
                (73, 4),
                (78, 2),
                (91, 1),
            ],
            "sts [r22.x8+ur24+0x90], r23;",
        ),
    ]
}

/// Hand-built words of `LDC` in both forms, `RED` and `ATOMG` in its three forms: opcode, fields at distinct values, text.
fn ldc_and_global_atomic_words() -> Vec<Case> {
    vec![
        // LDC, bound: dst 25, offset register 26 (24), offset 0x1234 (38), index 31 (54), type
        // 5, mode 2 (78).
        (
            0xb82,
            &[(16, 25), (24, 26), (38, 0x1234), (54, 31), (73, 5), (78, 2)],
            "ldc.is.64 r25, c[0x1f][r26+0x1234];",
        ),
        // LDC, bindless: dst 27, handle ur28 (24), offset 0x248, offset register 29 (64), type
        // 0, bit 91.
        (
            0x582,
            &[(16, 27), (24, 28), (38, 0x248), (64, 29), (91, 1)],
            "ldc.u8 r27, cx[ur28][r29+0x248];",
        ),
        // RED: destination field RZ, addr 30, data 31, offset 0x12, ur32 (64), .E, type 1,
        // order 5, eviction 0, operation 7 (87), the register's 64-bit flag, bit 91.
        (
            0x98e,
            &[
                (16, 0xff),
                (24, 30),
                (32, 31),
                (40, 0x12),
                (64, 32),
                (72, 1),
                (73, 1),
                (77, 5),
                (87, 7),
                (90, 1),
                (91, 1),
            ],
            "red.e.xor.ef.s32.strong.sm [r30.64+ur32+0x12], r31;",
        ),
        // ATOMG, no uniform register: dst 33, addr 34, data 35, offset 0x14, .E, type 3, order
        // 0, predicate destination 2, eviction 1, operation 2.
        (
            0x3a8,
            &[
                (16, 33),
                (24, 34),
                (32, 35),
                (40, 0x14),
                (72, 1),
                (73, 3),
                (81, 2),
                (84, 1),
                (87, 2),
            ],
            "atomg.e.max.f32.ftz.rn p2, r33, [r34+0x14], r35;",
        ),
        // ATOMG: dst 36, addr 37, data 38, offset 0x16, ur39, the register's 64-bit flag at 70,
        // .E, type 2, order 7, predicate destination PT, eviction 0, operation 8, bit 91.
        (
            0x9a8,
            &[
                (16, 36),
                (24, 37),
                (32, 38),
                (40, 0x16),
                (64, 39),
                (70, 1),
                (72, 1),
                (73, 2),
                (77, 7),
                (81, 7),
                (87, 8),
                (91, 1),
            ],
            "atomg.e.exch.ef.64.strong.gpu pt, r36, [r37.64+ur39+0x16], r38;",
        ),
        // ATOMG.CAS: dst 40, addr 41, comparison 42 (32), offset 0x18, data 43 (64), .E, type
        // 5, order 0xa, predicate destination 1, eviction 0.
        (
            0x3a9,
            &[
                (16, 40),
                (24, 41),
                (32, 42),
                (40, 0x18),
                (64, 43),
                (72, 1),
                (73, 5),
                (77, 0xa),
                (81, 1),
            ],
            "atomg.e.cas.ef.s64.strong.sys p1, r40, [r41+0x18], r42, r43;",
        ),
    ]
}

/// Hand-built words of `ATOMS` in its three forms, `ATOMS.POPC`, `CCTL`, the barriers and `LDGSTS`: opcode, fields at distinct values, text.
fn shared_atomic_and_barrier_words() -> Vec<Case> {
    vec![
        // ATOMS, no uniform register: dst 44, addr 45, data 46, offset 0x1a, type 0, stride 2,
        // operation 3.
        (
            0x38c,
            &[(16, 44), (24, 45), (32, 46), (40, 0x1a), (78, 2), (87, 3)],
            "atoms.inc r44, [r45.x8+0x1a], r46;",
        ),
        // ATOMS: dst 47, addr 48, data 49, offset 0x1c, ur50, type 1, stride 1, operation 4,
        // bit 91.
        (
            0x98c,
            &[
                (16, 47),
                (24, 48),
                (32, 49),
                (40, 0x1c),
                (64, 50),
                (73, 1),
                (78, 1),
                (87, 4),
                (91, 1),
            ],
            "atoms.dec.s32 r47, [r48.x4+ur50+0x1c], r49;",
        ),
        // ATOMS.CAS: dst 51, addr 52, comparison 53, offset 0x1e, data 54, type 2, stride 3.
        (
            0x38d,
            &[
                (16, 51),
                (24, 52),
                (32, 53),
                (40, 0x1e),
                (64, 54),
                (73, 2),
                (78, 3),
            ],
            "atoms.cas.64 r51, [r52.x16+0x1e], r53, r54;",
        ),
        // ATOMS.POPC: the fixed RZ, URZ, operation 10 and bit 91; addr 55, stride 2.
        (
            0xf8c,
            &[(16, 0xff), (24, 55), (64, 63), (78, 2), (87, 10), (91, 1)],
            "atoms.popc.inc.32 [r55.x8];",
        ),
        // CCTL.IV: addr 56, offset 0x76543210 (32), operation 3 (87).
        (
            0x98f,
            &[(24, 56), (32, 0x7654_3210), (87, 3)],
            "cctl.iv [r56+0x76543210];",
        ),
        // MEMBAR: .MMIO (72), scope 2 (76), .SC (80).
        (0x992, &[(72, 1), (76, 2), (80, 1)], "membar.sc.mmio.gpu;"),
        // ERRBAR.
        (0x9ab, &[], "errbar;"),
        // LDGSTS: shared 57, global 58, global offset 0x7ff (32), shared offset 3 units (48),
        // descriptor 0x2a (64), the enable predicate's negation (70) and register 4 (87), and
        // the fixed fields.
        (
            0xfae,
            &[
                (16, 57),
                (24, 58),
                (32, 0x7ff),
                (48, 3),
                (64, 0x2a),
                (70, 1),
                (74, 7),
                (84, 1),
                (87, 4),
                (90, 3),
            ],
            "ldgsts.e.bypass.128 [r57+0x30], [r58.64+0x7ff], 0x2a, !p4;",
        ),
        // LDGDEPBAR.
        (0x9af, &[], "ldgdepbar;"),
    ]
}

/// One hand-built word per form of the group, every field at a value no other field of the word
/// has, and its text (`nvdisasm`'s style; the parts NAK's tests do not cover are guesses).
#[test]
fn hand_built_words_of_each_form_decode() {
    let cases = [
        load_words(),
        store_words(),
        ldc_and_global_atomic_words(),
        shared_atomic_and_barrier_words(),
    ]
    .concat();
    let mut opcodes: Vec<u16> = cases
        .iter()
        .map(|&(opcode, ..)| u16::try_from(opcode).expect("12 bits"))
        .collect();
    for (opcode, fields, expected) in cases {
        assert_eq!(text(word(opcode, fields)), expected, "{opcode:#x}");
    }
    // One case per form of the group.
    let mut forms: Vec<u16> = memory::FORMS.iter().map(|form| form.opcode).collect();
    opcodes.sort_unstable();
    forms.sort_unstable();
    assert_eq!(opcodes, forms);
}

/// `LDG r1, [r2+ur3]`, every other field zero but the predicate destination, `PT`: bit 91 set,
/// the load predicate field 0 (`PT`).
const LDG_BASE: &[(u32, u128)] = &[(16, 1), (24, 2), (32, 3), (81, 7), (91, 1)];
/// `LDS r1, [r2+ur3]`.
const LDS_BASE: &[(u32, u128)] = &[(16, 1), (24, 2), (32, 3), (73, 4), (91, 1)];
/// `ATOMS.ADD r1, [r2], r4`.
const ATOMS_BASE: &[(u32, u128)] = &[(16, 1), (24, 2), (32, 4)];
/// `RED.ADD [r2], r4` with bit 91 clear.
const RED_BASE: &[(u32, u128)] = &[(24, 2), (32, 4)];

/// `base`'s fields and `extra`'s.
fn with(base: &[(u32, u128)], extra: &[(u32, u128)]) -> Vec<(u32, u128)> {
    base.iter().chain(extra).copied().collect()
}

/// `LDG`'s modifier bits and mode values, each alone.
fn ldg_modifier_cases() -> Vec<BuiltCase> {
    let ldg = LDG_BASE;
    vec![
        (0x981, with(ldg, &[]), "ldg.ef.u8 r1, [r2+ur3];"),
        // .E alone: the uniform register's flag in this form.
        (0x981, with(ldg, &[(72, 1)]), "ldg.e.ef.u8 r1, [r2+ur3];"),
        // The register's 64-bit flag alone.
        (0x981, with(ldg, &[(90, 1)]), "ldg.ef.u8 r1, [r2.64+ur3];"),
        // Bit 91 clear: the uniform register is left out.
        (
            0x981,
            vec![(16, 1), (24, 2), (32, 3), (81, 7)],
            "ldg.ef.u8 r1, [r2];",
        ),
        // The load predicate's negation alone: !PT.
        (0x981, with(ldg, &[(67, 1)]), "ldg.ef.u8 r1, [r2+ur3], !pt;"),
        // The load predicate field 7: P0.
        (0x981, with(ldg, &[(64, 7)]), "ldg.ef.u8 r1, [r2+ur3], p0;"),
        (0x981, with(ldg, &[(64, 1)]), "ldg.ef.u8 r1, [r2+ur3], p6;"),
        // The predicate destination: P0 when its field is clear, P4.
        (
            0x981,
            vec![(16, 1), (24, 2), (32, 3), (91, 1)],
            "ldg.ef.u8 p0, r1, [r2+ur3];",
        ),
        (
            0x981,
            vec![(16, 1), (24, 2), (32, 3), (81, 4), (91, 1)],
            "ldg.ef.u8 p4, r1, [r2+ur3];",
        ),
        (0x981, with(ldg, &[(84, 1)]), "ldg.u8 r1, [r2+ur3];"),
        (0x981, with(ldg, &[(84, 2)]), "ldg.el.u8 r1, [r2+ur3];"),
        (0x981, with(ldg, &[(84, 3)]), "ldg.lu.u8 r1, [r2+ur3];"),
        (0x981, with(ldg, &[(84, 4)]), "ldg.eu.u8 r1, [r2+ur3];"),
        (0x981, with(ldg, &[(84, 5)]), "ldg.na.u8 r1, [r2+ur3];"),
        (0x981, with(ldg, &[(73, 1)]), "ldg.ef.s8 r1, [r2+ur3];"),
        (0x981, with(ldg, &[(73, 2)]), "ldg.ef.u16 r1, [r2+ur3];"),
        (0x981, with(ldg, &[(73, 3)]), "ldg.ef.s16 r1, [r2+ur3];"),
        (0x981, with(ldg, &[(73, 4)]), "ldg.ef r1, [r2+ur3];"),
        (0x981, with(ldg, &[(73, 5)]), "ldg.ef.64 r1, [r2+ur3];"),
        (0x981, with(ldg, &[(73, 6)]), "ldg.ef.128 r1, [r2+ur3];"),
        (
            0x981,
            with(ldg, &[(77, 4)]),
            "ldg.ef.u8.constant r1, [r2+ur3];",
        ),
        (
            0x981,
            with(ldg, &[(77, 5)]),
            "ldg.ef.u8.strong.sm r1, [r2+ur3];",
        ),
        (
            0x981,
            with(ldg, &[(77, 7)]),
            "ldg.ef.u8.strong.gpu r1, [r2+ur3];",
        ),
        (
            0x981,
            with(ldg, &[(77, 0xa)]),
            "ldg.ef.u8.strong.sys r1, [r2+ur3];",
        ),
        // The offset's lowest and highest bits alone.
        (0x981, with(ldg, &[(40, 1)]), "ldg.ef.u8 r1, [r2+ur3+0x1];"),
        (
            0x981,
            with(ldg, &[(63, 1)]),
            "ldg.ef.u8 r1, [r2+ur3+-0x800000];",
        ),
        // LDG without a uniform register: .E is the register's flag, printed as `.e`.
        (
            0x381,
            vec![(16, 1), (24, 2), (72, 1), (81, 7)],
            "ldg.e.ef.u8 r1, [r2];",
        ),
    ]
}

/// The other instructions' modifier bits and mode values, each alone.
fn other_modifier_cases() -> Vec<BuiltCase> {
    let (lds, atoms, red) = (LDS_BASE, ATOMS_BASE, RED_BASE);
    vec![
        (0x984, with(lds, &[]), "lds r1, [r2+ur3];"),
        (0x984, with(lds, &[(87, 1)]), "lds.zd r1, [r2+ur3];"),
        (0x984, with(lds, &[(78, 1)]), "lds r1, [r2.x4+ur3];"),
        (0x984, with(lds, &[(78, 2)]), "lds r1, [r2.x8+ur3];"),
        (0x984, with(lds, &[(78, 3)]), "lds r1, [r2.x16+ur3];"),
        // URZ is left out.
        (
            0x984,
            vec![(16, 1), (24, 2), (32, 63), (73, 4), (91, 1)],
            "lds r1, [r2];",
        ),
        (0x38c, with(atoms, &[]), "atoms.add r1, [r2], r4;"),
        (0x38c, with(atoms, &[(87, 1)]), "atoms.min r1, [r2], r4;"),
        (0x38c, with(atoms, &[(87, 2)]), "atoms.max r1, [r2], r4;"),
        (0x38c, with(atoms, &[(87, 3)]), "atoms.inc r1, [r2], r4;"),
        (0x38c, with(atoms, &[(87, 4)]), "atoms.dec r1, [r2], r4;"),
        (0x38c, with(atoms, &[(87, 5)]), "atoms.and r1, [r2], r4;"),
        (0x38c, with(atoms, &[(87, 6)]), "atoms.or r1, [r2], r4;"),
        (0x38c, with(atoms, &[(87, 7)]), "atoms.xor r1, [r2], r4;"),
        (0x38c, with(atoms, &[(87, 8)]), "atoms.exch r1, [r2], r4;"),
        (
            0x38c,
            with(atoms, &[(73, 1)]),
            "atoms.add.s32 r1, [r2], r4;",
        ),
        (0x38c, with(atoms, &[(73, 2)]), "atoms.add.64 r1, [r2], r4;"),
        (
            0x38c,
            with(atoms, &[(73, 3)]),
            "atoms.add.f32.ftz.rn r1, [r2], r4;",
        ),
        (
            0x38c,
            with(atoms, &[(73, 4)]),
            "atoms.add.f16x2.rn r1, [r2], r4;",
        ),
        (
            0x38c,
            with(atoms, &[(73, 5)]),
            "atoms.add.s64 r1, [r2], r4;",
        ),
        (
            0x38c,
            with(atoms, &[(73, 6)]),
            "atoms.add.f64.rn r1, [r2], r4;",
        ),
        (0x98e, with(red, &[]), "red.add.ef [r2], r4;"),
        (0x98e, with(red, &[(87, 7)]), "red.xor.ef [r2], r4;"),
        (0x98f, vec![(24, 2)], "cctl.pf1 [r2];"),
        (0x98f, vec![(24, 2), (87, 1)], "cctl.pf2 [r2];"),
        (0x98f, vec![(24, 2), (87, 2)], "cctl.wb [r2];"),
        (0x98f, vec![(24, 2), (87, 5)], "cctl.rs [r2];"),
        // An operation on every line leaves out `[rz]`, not another address.
        (0x98f, vec![(24, 0xff), (87, 4)], "cctl.ivall;"),
        (0x98f, vec![(24, 0xff), (87, 6)], "cctl.ivallp;"),
        (0x98f, vec![(24, 0xff), (87, 7)], "cctl.wball;"),
        (0x98f, vec![(24, 0xff), (87, 8)], "cctl.wballp;"),
        (0x98f, vec![(24, 2), (87, 8)], "cctl.wballp [r2];"),
        (
            0x98f,
            vec![(24, 0xff), (32, 4), (87, 8)],
            "cctl.wballp [rz+0x4];",
        ),
        (0x98f, vec![(24, 0xff), (87, 3)], "cctl.iv [rz];"),
        (0x992, vec![], "membar.cta;"),
        (0x992, vec![(72, 1)], "membar.mmio.cta;"),
        (0x992, vec![(80, 1)], "membar.sc.cta;"),
        (0x992, vec![(76, 1)], "membar.sm;"),
        (0x992, vec![(76, 2)], "membar.gpu;"),
        (0x992, vec![(76, 3)], "membar.sys;"),
        (0x992, vec![(76, 5)], "membar.scope5;"),
        (0xb82, vec![(16, 1), (24, 0xff)], "ldc.u8 r1, c[0x0][0x0];"),
        (0xb82, vec![(16, 1), (24, 2)], "ldc.u8 r1, c[0x0][r2];"),
        (
            0xb82,
            vec![(16, 1), (24, 0xff), (78, 1)],
            "ldc.il.u8 r1, c[0x0][0x0];",
        ),
        (
            0xb82,
            vec![(16, 1), (24, 0xff), (78, 3)],
            "ldc.isl.u8 r1, c[0x0][0x0];",
        ),
        (
            0xb82,
            vec![(16, 1), (24, 0xff), (73, 6)],
            "ldc.128 r1, c[0x0][0x0];",
        ),
    ]
}

/// Each modifier bit and each value of each mode field, set alone in a hand-built word, changes
/// the text the way its field says.
#[test]
fn hand_built_words_pin_each_modifier_bit() {
    for (opcode, fields, expected) in [ldg_modifier_cases(), other_modifier_cases()].concat() {
        assert_eq!(
            text(word(opcode, &fields)),
            expected,
            "{opcode:#x} {fields:?}"
        );
    }
}

/// The fields that do not print: bit 76 of `LDG` and the uniform `STG`, and `RED`'s destination
/// field, each set alone, change the decoded instruction and nothing else.
#[test]
fn unprinted_fields_decode() {
    let with_76 = word(0x981, &[(16, 1), (24, 2), (76, 1)]);
    assert!(ld_of(with_76).bit76);
    assert_eq!(text(with_76), text(word(0x981, &[(16, 1), (24, 2)])));
    let ldg = decode(word(0x381, &[(76, 1)])).expect("decodes");
    assert!(matches!(ldg.op, Op::Ld(OpLd { bit76: true, .. })));
    let stg = decode(word(0x986, &[(76, 1)])).expect("decodes");
    assert!(matches!(stg.op, Op::St(OpSt { bit76: true, .. })));
    // The `STG` without a uniform register has no bit 76.
    assert_eq!(
        decode(word(0x386, &[(76, 1)])),
        Err(DecodeError::UnknownBit {
            opcode: 0x386,
            bit: 76
        })
    );
    for (field, ignored) in [(0, r(0)), (0x5a, r(0x5a)), (0xff, Gpr::RZ)] {
        let red = word(0x98e, &[(16, field)]);
        assert_eq!(atom_of(red).dst, AtomDst::Reduction { ignored });
        assert_eq!(text(red), "red.add.ef [r0], r0;");
    }
}

/// `LDG`'s load predicate is encoded reversed (NAK's `set_rev_pred_src`: the field holds 7 minus
/// the register), checked for every register against the literal field values.
#[test]
fn the_load_predicate_is_encoded_reversed() {
    for (field, register) in [
        (0, 7),
        (1, 6),
        (2, 5),
        (3, 4),
        (4, 3),
        (5, 2),
        (6, 1),
        (7, 0),
    ] {
        for negated in [false, true] {
            let ld = ld_of(word(0x981, &[(64, field), (67, u128::from(negated))]));
            assert_eq!(
                ld.pred,
                PredSrc {
                    reg: p(register),
                    negated
                }
            );
        }
    }
}

/// The address fields at both edges: the signed 24-bit offset (0, 1, the largest, -1 and the
/// smallest), a register in place of `RZ`, `URZ` and `UR62`, in a hand-built word decoded to
/// its fields.
#[test]
fn addresses_at_the_edges_of_their_fields() {
    let cases = [
        (0, 0),
        (1, 1),
        (0x7f_ffff, 0x7f_ffff),
        (0x80_0000, -0x80_0000),
        (0xff_ffff, -1),
        // Bytes above the low one: 0x123456 is not 0x56.
        (0x12_3456, 0x12_3456),
    ];
    for (field, offset) in cases {
        let ld = ld_of(word(0x984, &[(40, field)]));
        assert_eq!(ld.addr.offset, offset, "{field:#x}");
    }
    for (field, reg) in [
        (0, UniformReg::new(0)),
        (62, UniformReg::new(62)),
        (63, Some(UniformReg::URZ)),
    ] {
        let st = st_of(word(0x988, &[(64, field)]));
        assert_eq!(st.addr.uniform.map(|uniform| uniform.reg), reg);
    }
    for (field, reg) in [(0, r(0)), (0xfe, r(0xfe)), (0xff, Gpr::RZ)] {
        let st = st_of(word(0x387, &[(24, field), (32, 0x11)]));
        assert_eq!((st.addr.reg, st.data), (reg, r(0x11)));
    }
}

/// The other offsets at both edges: `CCTL`'s 32 bits, `LDGSTS`'s 12-bit global and 16-bit shared
/// offsets, `LDC`'s 16-bit offset and 5-bit index.
#[test]
fn other_offsets_at_the_edges_of_their_fields() {
    let cctl = |field: u128| match op_of(word(0x98f, &[(32, field)])) {
        Op::CCtl(cctl) => Some(cctl.offset),
        _ => None,
    };
    assert_eq!(cctl(0x7fff_ffff), Some(i32::MAX));
    assert_eq!(cctl(0x8000_0000), Some(i32::MIN));
    assert_eq!(cctl(0xffff_ffff), Some(-1));
    let ldgsts = |fields: &[(u32, u128)]| {
        let fixed = [(74, 7), (84, 1), (90, 3)];
        let all: Vec<(u32, u128)> = fields.iter().chain(&fixed).copied().collect();
        match op_of(word(0xfae, &all)) {
            Op::LdGsts(op) => Some((op.global_offset, op.shared_offset, op.descriptor)),
            _ => None,
        }
    };
    assert_eq!(ldgsts(&[(32, 0x7ff)]), Some((0x7ff, 0, 0)));
    assert_eq!(ldgsts(&[(32, 0x800)]), Some((-0x800, 0, 0)));
    assert_eq!(ldgsts(&[(48, 0x7fff)]), Some((0, 0x7fff, 0)));
    assert_eq!(ldgsts(&[(48, 0x8000)]), Some((0, -0x8000, 0)));
    assert_eq!(ldgsts(&[(64, 0x3f)]), Some((0, 0, 0x3f)));
    assert_eq!(
        text(word(0xfae, &[(48, 0x8000), (74, 7), (84, 1), (90, 3)])),
        "ldgsts.e.bypass.128 [r0+-0x80000], [r0.64], 0x0, p0;"
    );
    assert_eq!(
        text(word(
            0xb82,
            &[(16, 1), (24, 0xff), (38, 0xffff), (54, 0x1f)]
        )),
        "ldc.u8 r1, c[0x1f][0xffff];"
    );
}

/// The encoder at and one past the edge of each signed or narrow field it checks.
#[test]
fn encoder_refuses_values_past_their_fields() {
    let lds = |offset| {
        assemble(&Instr::new(Op::Ld(OpLd {
            dst: r(1),
            addr: MemAddr {
                reg: r(2),
                reg_64: false,
                uniform: Some(UniformAddr {
                    reg: UniformReg::URZ,
                    wide: false,
                    enabled: true,
                }),
                offset,
            },
            mem_type: MemType::B32,
            space: MemSpace::Shared(OffsetStride::X1),
            pred: PredSrc::TRUE,
            pred_dst: Pred::PT,
            zd: false,
            bit76: false,
        })))
    };
    assert!(lds(0x7f_ffff).is_ok());
    assert!(lds(-0x80_0000).is_ok());
    for offset in [0x80_0000, -0x80_0001] {
        assert_eq!(
            lds(offset),
            Err(EncodeError::OutOfRange {
                what: "address offset",
                value: i64::from(offset),
                bits: 24
            })
        );
    }
    let ldgsts = |global_offset, descriptor| {
        assemble(&Instr::new(Op::LdGsts(OpLdGsts {
            shared: r(1),
            shared_offset: i16::MIN,
            global: r(2),
            global_offset,
            descriptor,
            enable: PredSrc::TRUE,
        })))
    };
    assert!(ldgsts(0x7ff, 0x3f).is_ok());
    assert!(ldgsts(-0x800, 0).is_ok());
    for offset in [0x800, -0x801] {
        assert_eq!(
            ldgsts(offset, 0),
            Err(EncodeError::OutOfRange {
                what: "global offset",
                value: i64::from(offset),
                bits: 12
            })
        );
    }
    assert_eq!(
        ldgsts(0, 0x40).map_err(|error| error.to_string()),
        Err("memory descriptor 64 does not fit in 6 bits".into())
    );
    let ldc = |index| {
        assemble(&Instr::new(Op::Ldc(OpLdc {
            dst: r(1),
            cbuf: CBufRef {
                buf: CBuf::Binding(index),
                offset: 0,
            },
            offset: Gpr::RZ,
            mode: LdcMode::Indexed,
            mem_type: MemType::B32,
        })))
    };
    assert!(ldc(31).is_ok());
    assert_eq!(
        ldc(32),
        Err(EncodeError::TooWide {
            what: "constant buffer index",
            value: 32,
            bits: 5
        })
    );
    let cctl = |offset| {
        assemble(&Instr::new(Op::CCtl(OpCCtl {
            op: CCtlOp::Iv,
            addr: r(2),
            offset,
        })))
        .map(|word| decode(word).map(|instr| instr.op))
    };
    for offset in [i32::MIN, -1, i32::MAX] {
        assert_eq!(
            cctl(offset),
            Ok(Ok(Op::CCtl(OpCCtl {
                op: CCtlOp::Iv,
                addr: r(2),
                offset
            })))
        );
    }
}

/// Every value of each access field against the references' encodings, written out here:
/// NAK's `set_mem_type`, `set_mem_order` (SM 8.0), `set_eviction_priority` and `encode_sm75`.
#[test]
fn access_field_values_match_the_references() {
    use MemEvictionPriority as Evict;
    assert_eq!(
        MemType::ALL.iter().map(|t| t.bits()).collect::<Vec<_>>(),
        [0, 1, 2, 3, 4, 5, 6]
    );
    let types = [
        (MemType::U8, 0),
        (MemType::I8, 1),
        (MemType::U16, 2),
        (MemType::I16, 3),
        (MemType::B32, 4),
        (MemType::B64, 5),
        (MemType::B128, 6),
    ];
    for (value, bits) in types {
        assert_eq!(
            (value.bits(), MemType::from_bits(bits)),
            (bits, Some(value))
        );
    }
    let orders = [
        (MemOrder::Weak, 0x0),
        (MemOrder::Constant, 0x4),
        (MemOrder::Strong(MemScope::Cta), 0x5),
        (MemOrder::Strong(MemScope::Gpu), 0x7),
        (MemOrder::Strong(MemScope::System), 0xa),
    ];
    for (value, bits) in orders {
        assert_eq!(
            (value.bits(), MemOrder::from_bits(bits)),
            (bits, Some(value))
        );
    }
    assert_eq!(MemOrder::ALL, orders.map(|(value, _)| value));
    let evictions = [
        (Evict::First, 0),
        (Evict::Normal, 1),
        (Evict::Last, 2),
        (Evict::LastUse, 3),
        (Evict::Unchanged, 4),
        (Evict::NoAllocate, 5),
    ];
    for (value, bits) in evictions {
        assert_eq!((value.bits(), Evict::from_bits(bits)), (bits, Some(value)));
    }
    let strides = [
        (OffsetStride::X1, 0),
        (OffsetStride::X4, 1),
        (OffsetStride::X8, 2),
        (OffsetStride::X16, 3),
    ];
    for (value, bits) in strides {
        assert_eq!(
            (value.bits(), OffsetStride::from_bits(bits)),
            (bits, Some(value))
        );
    }
}

/// The atomic, cache, barrier and `LDC` field values against the references' encodings,
/// written out here: NAK's `set_atom_type` (below SM 9.0), `set_atom_op`, `OpCCtl`, `OpMemBar`
/// (with PR #4588's scope 5) and `OpLdc`'s mode.
#[test]
fn atomic_and_barrier_field_values_match_the_references() {
    let atom_types = [
        (AtomType::U32, 0),
        (AtomType::I32, 1),
        (AtomType::U64, 2),
        (AtomType::F32, 3),
        (AtomType::F16v2, 4),
        (AtomType::I64, 5),
        (AtomType::F64, 6),
    ];
    for (value, bits) in atom_types {
        assert_eq!(
            (value.bits(), AtomType::from_bits(bits)),
            (bits, Some(value))
        );
    }
    let atom_ops = [
        (AtomOp::Add, 0),
        (AtomOp::Min, 1),
        (AtomOp::Max, 2),
        (AtomOp::Inc, 3),
        (AtomOp::Dec, 4),
        (AtomOp::And, 5),
        (AtomOp::Or, 6),
        (AtomOp::Xor, 7),
        (AtomOp::Exch, 8),
    ];
    for (value, bits) in atom_ops {
        assert_eq!(
            (value.bits(), AtomOp::from_bits(bits)),
            (Some(bits), Some(value))
        );
    }
    assert_eq!(AtomOp::ENCODED, atom_ops.map(|(value, _)| value));
    assert_eq!(AtomOp::CmpExch(AtomCmpSrc::Separate { cmpr: r(1) }).bits(), None);
    let cctl_ops = [
        (CCtlOp::Pf1, 0, false),
        (CCtlOp::Pf2, 1, false),
        (CCtlOp::Wb, 2, false),
        (CCtlOp::Iv, 3, false),
        (CCtlOp::IvAll, 4, true),
        (CCtlOp::Rs, 5, false),
        (CCtlOp::IvAllP, 6, true),
        (CCtlOp::WbAll, 7, true),
        (CCtlOp::WbAllP, 8, true),
    ];
    for (value, bits, all) in cctl_ops {
        assert_eq!(
            (value.bits(), CCtlOp::from_bits(bits), value.is_all()),
            (bits, Some(value), all)
        );
    }
    let scopes = [
        (MemBarScope::Cta, 0),
        (MemBarScope::Sm, 1),
        (MemBarScope::Gpu, 2),
        (MemBarScope::System, 3),
        (MemBarScope::Scope5, 5),
    ];
    for (value, bits) in scopes {
        assert_eq!(
            (value.bits(), MemBarScope::from_bits(bits)),
            (bits, Some(value))
        );
    }
    let modes = [
        (LdcMode::Indexed, 0),
        (LdcMode::IndexedLinear, 1),
        (LdcMode::IndexedSegmented, 2),
        (LdcMode::IndexedSegmentedLinear, 3),
    ];
    for (value, bits) in modes {
        assert_eq!(
            (value.bits(), LdcMode::from_bits(bits)),
            (bits, Some(value))
        );
    }
}

/// The group's opcodes and field positions against the references' values, written out here:
/// NAK's `set_opcode` and field ranges (`sm70_encode.rs` @ 52238762), and PR #4588's rows and
/// `Take` calls for the forms it alone has (`ampere_isa_table.h`, `ops_memory.cpp` @ e5fb3b97).
#[test]
fn layout_constants_match_the_references() {
    let opcodes = [
        (LDG_OPCODE, 0x381),
        (LDG_UNIFORM_OPCODE, 0x981),
        (LDL_OPCODE, 0x983),
        (LDS_OPCODE, 0x984),
        (STG_OPCODE, 0x386),
        (STG_UNIFORM_OPCODE, 0x986),
        (STL_OPCODE, 0x387),
        (STL_UNIFORM_OPCODE, 0x987),
        (STS_OPCODE, 0x388),
        (STS_UNIFORM_OPCODE, 0x988),
        (LDC_OPCODE, 0xb82),
        (LDC_BINDLESS_OPCODE, 0x582),
        (RED_OPCODE, 0x98e),
        (ATOMG_OPCODE, 0x3a8),
        (ATOMG_UNIFORM_OPCODE, 0x9a8),
        (ATOMG_CAS_OPCODE, 0x3a9),
        (ATOMS_OPCODE, 0x38c),
        (ATOMS_UNIFORM_OPCODE, 0x98c),
        (ATOMS_CAS_OPCODE, 0x38d),
        (ATOMS_POPC_OPCODE, 0xf8c),
        (CCTL_OPCODE, 0x98f),
        (MEMBAR_OPCODE, 0x992),
        (ERRBAR_OPCODE, 0x9ab),
        (LDGDEPBAR_OPCODE, 0x9af),
        (LDGSTS_OPCODE, 0xfae),
    ];
    for (constant, literal) in opcodes {
        assert_eq!(constant, literal);
    }
    let fields = [
        (MEM_ADDR, (24, 8)),
        (MEM_OFFSET, (40, 24)),
        (MEM_DATA, (32, 8)),
        (LD_UNIFORM, (32, 6)),
        (ST_UNIFORM, (64, 6)),
        (MEM_E, (72, 1)),
        (MEM_REG_64, (90, 1)),
        (ATOMG_REG_64, (70, 1)),
        (MEM_TYPE, (73, 3)),
        (MEM_BIT_76, (76, 1)),
        (MEM_ORDER, (77, 4)),
        (MEM_STRIDE, (78, 2)),
        (MEM_EVICTION, (84, 3)),
        (LDG_PRED, (64, 3)),
        (LDG_PRED_NOT, (67, 1)),
        (LDS_ZD, (87, 1)),
        (LDC_OFFSET, (24, 8)),
        (LDC_BINDLESS_OFFSET, (64, 8)),
        (LDC_HANDLE, (24, 6)),
        (LDC_MODE, (78, 2)),
        (ATOM_OP, (87, 4)),
        (RED_OP, (87, 3)),
        (ATOM_TYPE, (73, 3)),
        (CAS_CMPR, (32, 8)),
        (CAS_DATA, (64, 8)),
        (CCTL_OFFSET, (32, 32)),
        (CCTL_OP, (87, 4)),
        (MEMBAR_MMIO, (72, 1)),
        (MEMBAR_SCOPE, (76, 3)),
        (MEMBAR_SC, (80, 1)),
        (LDGSTS_SHARED, (16, 8)),
        (LDGSTS_GLOBAL, (24, 8)),
        (LDGSTS_GLOBAL_OFFSET, (32, 12)),
        (LDGSTS_SHARED_OFFSET, (48, 16)),
        (LDGSTS_DESCRIPTOR, (64, 6)),
        (LDGSTS_ENABLE_NOT, (70, 1)),
        (UNIFORM_BIT, (91, 1)),
    ];
    for (field, (start, width)) in fields {
        assert_eq!((field.start, field.width), (start, width));
    }
    assert_eq!(LDGSTS_SHARED_UNIT, 16);
    // Bits 74..77, 84, 90 and 91.
    assert_eq!(LDGSTS_FIXED, (0b111 << 74) | (1 << 84) | (0b11 << 90));
    // RZ at 16..24, URZ at 64..70, the operation 10 at 87..91, bit 91.
    assert_eq!(ATOMS_POPC_OP, 10);
    assert_eq!(
        ATOMS_POPC_FIXED,
        (0xff << 16) | (0x3f << 64) | (10 << 87) | (1 << 91)
    );
    assert_eq!(
        ATOMS_POPC_FIXED & ATOM_OP.mask(),
        u128::from(ATOMS_POPC_OP) << ATOM_OP.start
    );
}

/// The values the group refuses, each alone, with the exact error; and, for each pair of
/// refusable fields of one instruction, both refused at once: the error names the field read
/// first, the lowest.
#[test]
fn values_the_group_refuses() {
    let unknown = |opcode: u16, bit: u32, width: u32, value: u64| DecodeError::UnknownValue {
        opcode,
        bit,
        width,
        value,
    };
    let cases: Vec<RefusalCase> = vec![
        (0x981, vec![(73, 7)], unknown(0x981, 73, 3, 7)),
        (0x983, vec![(73, 7)], unknown(0x983, 73, 3, 7)),
        (0x984, vec![(73, 7)], unknown(0x984, 73, 3, 7)),
        (0x987, vec![(73, 7)], unknown(0x987, 73, 3, 7)),
        (0xb82, vec![(73, 7)], unknown(0xb82, 73, 3, 7)),
        (0x582, vec![(73, 7), (91, 1)], unknown(0x582, 73, 3, 7)),
        (0x381, vec![(84, 6)], unknown(0x381, 84, 3, 6)),
        (0x986, vec![(84, 7)], unknown(0x986, 84, 3, 7)),
        (0x983, vec![(84, 6)], unknown(0x983, 84, 3, 6)),
        (0x387, vec![(84, 7)], unknown(0x387, 84, 3, 7)),
        (0x3a9, vec![(73, 7)], unknown(0x3a9, 73, 3, 7)),
        (0x38d, vec![(73, 7)], unknown(0x38d, 73, 3, 7)),
        (0x98e, vec![(84, 6)], unknown(0x98e, 84, 3, 6)),
        (0x9a8, vec![(87, 9)], unknown(0x9a8, 87, 4, 9)),
        (0x3a8, vec![(87, 15)], unknown(0x3a8, 87, 4, 15)),
        (0x98c, vec![(87, 9)], unknown(0x98c, 87, 4, 9)),
        (0x38c, vec![(87, 15)], unknown(0x38c, 87, 4, 15)),
        (0x98f, vec![(87, 9)], unknown(0x98f, 87, 4, 9)),
        (0x98f, vec![(87, 15)], unknown(0x98f, 87, 4, 15)),
        (0x992, vec![(76, 4)], unknown(0x992, 76, 3, 4)),
        (0x992, vec![(76, 6)], unknown(0x992, 76, 3, 6)),
        (0x992, vec![(76, 7)], unknown(0x992, 76, 3, 7)),
        // Two refused values at once: the lower field is named.
        (0x981, vec![(73, 7), (84, 7)], unknown(0x981, 73, 3, 7)),
        (0x981, vec![(77, 1), (84, 6)], unknown(0x981, 77, 4, 1)),
        (0x983, vec![(73, 7), (84, 7)], unknown(0x983, 73, 3, 7)),
        (0x9a8, vec![(73, 7), (87, 9)], unknown(0x9a8, 73, 3, 7)),
        (0x9a8, vec![(77, 3), (87, 9)], unknown(0x9a8, 77, 4, 3)),
        (0x9a8, vec![(84, 6), (87, 9)], unknown(0x9a8, 84, 3, 6)),
        (0x98c, vec![(73, 7), (87, 9)], unknown(0x98c, 73, 3, 7)),
    ];
    for (opcode, fields, error) in cases {
        assert_eq!(
            decode(word(opcode, &fields)),
            Err(error),
            "{opcode:#x} {fields:?}"
        );
    }
    assert_eq!(
        decode(word(0x981, &[(73, 7)])).map_err(|error| error.to_string()),
        Err(
            "opcode 0x981 has the value 7 in its 3-bit field at bit 73, which its form does not \
             define"
                .into()
        )
    );
    // Every four-bit memory order: the five NAK names decode, the others are refused.
    for value in 0..16_u64 {
        let result = decode(word(0x986, &[(77, u128::from(value))]));
        if [0x0, 0x4, 0x5, 0x7, 0xa].contains(&value) {
            assert!(result.is_ok(), "{value:#x}");
        } else {
            assert_eq!(result, Err(unknown(0x986, 77, 4, value)), "{value:#x}");
        }
    }
    // `RED`'s three-bit operation names an operation in every value.
    for value in 0..8 {
        assert!(decode(word(0x98e, &[(87, value)])).is_ok());
    }
}

/// The bits each row of the group marks known and the form rejects, written out: bit 91 of the
/// forms without a uniform register, bits 64..71 and 91 of `ATOMG` `0x3a8`, and the bits PR
/// #4588 requires clear in `LDGSTS` and `ATOMS.POPC` (module docs). Each, set alone in the form's
/// bare word, is an unknown bit.
#[test]
fn memory_rows_drop_exactly_the_listed_bits() {
    let range = |start: u32, end: u32| (start..end).fold(0_u128, |mask, bit| mask | (1 << bit));
    let expected = |opcode: u16| -> u128 {
        match opcode {
            0x381 | 0x386 | 0x387 | 0x388 | 0xb82 => 1 << 91,
            0x3a8 => range(64, 71) | (1 << 91),
            0xfae => range(44, 48) | (1 << 71) | range(72, 74) | range(77, 84) | range(85, 87),
            0xf8c => range(32, 64) | range(73, 76) | (1 << 87) | (1 << 89),
            _ => 0,
        }
    };
    for (form, row) in memory::FORMS.iter().zip(memory::REFERENCE_ROWS) {
        let opcode = form.opcode;
        assert_eq!(row.dropped, expected(opcode), "{opcode:#x}");
        let base = Instruction::from_bits(form.fixed).with_field(OPCODE, u64::from(opcode));
        for bit in 0..128 {
            if row.dropped & (1 << bit) != 0 {
                assert_eq!(
                    decode(Instruction::from_bits(base.bits() | (1 << bit))),
                    Err(DecodeError::UnknownBit { opcode, bit }),
                    "{opcode:#x} bit {bit}"
                );
            }
        }
    }
}

/// `LDGSTS`'s fixed bits, the values PR #4588 requires (`Take(74, 3) == 7`, `Take(84, 1) == 1`,
/// `Take(90, 2) == 3`): each cleared alone is a missing bit, and with two cleared the lower one
/// is named.
#[test]
fn ldgsts_requires_the_bits_pr_4588_requires() {
    let fixed = [(74, 7), (84, 1), (90, 3)];
    assert_eq!(
        text(word(0xfae, &fixed)),
        "ldgsts.e.bypass.128 [r0], [r0.64], 0x0, p0;"
    );
    for bit in [74, 75, 76, 84, 90, 91] {
        let bits = word(0xfae, &fixed).bits() & !(1 << bit);
        assert_eq!(
            decode(Instruction::from_bits(bits)),
            Err(DecodeError::MissingBit { opcode: 0xfae, bit })
        );
    }
    let both = word(0xfae, &fixed).bits() & !(1 << 84) & !(1 << 90);
    assert_eq!(
        decode(Instruction::from_bits(both)),
        Err(DecodeError::MissingBit {
            opcode: 0xfae,
            bit: 84
        })
    );
    // A dropped bit and a missing bit at once: the dropped bit is reported first (`decode`'s
    // step 2 before step 3).
    let stray_and_missing = word(0xfae, &fixed).bits() & !(1 << 74) | (1 << 85);
    assert_eq!(
        decode(Instruction::from_bits(stray_and_missing)),
        Err(DecodeError::UnknownBit {
            opcode: 0xfae,
            bit: 85
        })
    );
}

/// `ATOMS.POPC`'s fixed bits (`RZ`, `URZ`, the operation 10 and bit 91): each cleared alone is
/// a missing bit; the operation's clear bits set are unknown bits.
#[test]
fn atoms_popc_requires_the_bits_pr_4588_requires() {
    let fixed = [(16, 0xff), (64, 0x3f), (87, 10), (91, 1)];
    assert_eq!(text(word(0xf8c, &fixed)), "atoms.popc.inc.32 [r0];");
    for bit in (16..24).chain(64..70).chain([88, 90, 91]) {
        let bits = word(0xf8c, &fixed).bits() & !(1 << bit);
        assert_eq!(
            decode(Instruction::from_bits(bits)),
            Err(DecodeError::MissingBit { opcode: 0xf8c, bit }),
            "bit {bit}"
        );
    }
    for bit in [87, 89] {
        let bits = word(0xf8c, &fixed).bits() | (1 << bit);
        assert_eq!(
            decode(Instruction::from_bits(bits)),
            Err(DecodeError::UnknownBit { opcode: 0xf8c, bit })
        );
    }
}

/// The bindless `LDC`'s handle is any uniform register the field holds, odd ones and `URZ`
/// included (module docs), and its bit 91 is required.
#[test]
fn bindless_ldc_handles_decode_every_register() {
    for handle in [0_u8, 1, 3, 62, 63] {
        let op = op_of(word(0x582, &[(24, u128::from(handle)), (91, 1)]));
        assert!(
            matches!(op, Op::Ldc(ldc) if ldc.cbuf.buf == CBuf::BindlessUniform(ur(handle))),
            "{op:?}"
        );
    }
    assert_eq!(
        decode(word(0x582, &[])),
        Err(DecodeError::MissingBit {
            opcode: 0x582,
            bit: 91
        })
    );
}

/// The rows PR #4588's table has near the group that this slice leaves undecoded (module docs),
/// and neighbours of the group's opcodes that no reference names, are unknown opcodes.
#[test]
fn memory_rows_left_undecoded_are_unknown_opcodes() {
    // ULDC and ULDC.64 (slice 6d), LDCG, and opcodes with no row: generic LD and ST, SM 9.0's
    // float atomics, and a gap among the global forms.
    for opcode in [
        0xab9, 0xabb, 0x8b8, 0x980, 0x985, 0x9a3, 0x9a6, 0x982, 0x9ac,
    ] {
        let word = Instruction::default().with_field(OPCODE, u64::from(opcode));
        assert_eq!(
            decode(word),
            Err(DecodeError::UnknownOpcode { opcode }),
            "{opcode:#x}"
        );
    }
}

/// `[r2]` with `uniform` and the register's 64-bit flag `reg_64`.
fn addr(uniform: Option<UniformAddr>, reg_64: bool) -> MemAddr {
    MemAddr {
        reg: r(2),
        reg_64,
        uniform,
        offset: 0,
    }
}

/// `URZ`, enabled, with the 64-bit flag `wide`.
fn urz(wide: bool) -> UniformAddr {
    UniformAddr {
        reg: UniformReg::URZ,
        wide,
        enabled: true,
    }
}

/// The encoder's error message for `op`, if it refuses it.
fn refused(op: Op) -> Result<Instruction, String> {
    assemble(&Instr::new(op)).map_err(|error| error.to_string())
}

/// The encoder refuses a load or store field the form it picks has no bits for, with
/// plain-word messages.
#[test]
fn unencodable_loads_and_stores_are_errors() {
    let ld = OpLd {
        dst: r(1),
        addr: addr(Some(urz(false)), false),
        mem_type: MemType::B32,
        space: MemSpace::Shared(OffsetStride::X1),
        pred: PredSrc::TRUE,
        pred_dst: Pred::PT,
        zd: false,
        bit76: false,
    };
    assert!(refused(Op::Ld(ld)).is_ok());
    let ld_cases = [
        (
            OpLd {
                addr: addr(None, false),
                ..ld
            },
            "ld has no bits for an address without a uniform register",
        ),
        (
            OpLd {
                addr: addr(Some(urz(false)), true),
                ..ld
            },
            "ld has no bits for a 64-bit address register",
        ),
        (
            OpLd {
                addr: addr(Some(urz(true)), false),
                ..ld
            },
            "ld has no bits for a 64-bit uniform address",
        ),
        (
            OpLd {
                pred: PredSrc {
                    reg: p(1),
                    negated: false,
                },
                ..ld
            },
            "ld has no bits for a load predicate",
        ),
        (
            OpLd {
                pred_dst: p(1),
                ..ld
            },
            "ld has no bits for a predicate destination",
        ),
        (OpLd { bit76: true, ..ld }, "ld has no bits for bit 76"),
        (
            OpLd {
                space: MemSpace::Local(MemEvictionPriority::Normal),
                zd: true,
                ..ld
            },
            "ld has no bits for .zd",
        ),
        // The global load without a uniform register has no load predicate.
        (
            OpLd {
                space: global(MemOrder::Weak, MemEvictionPriority::Normal),
                addr: addr(None, true),
                pred: PredSrc {
                    reg: Pred::PT,
                    negated: true,
                },
                ..ld
            },
            "ld has no bits for a load predicate",
        ),
    ];
    for (op, message) in ld_cases {
        assert_eq!(refused(Op::Ld(op)), Err(message.into()));
    }
    let st = OpSt {
        data: r(1),
        addr: addr(None, true),
        mem_type: MemType::B32,
        space: global(MemOrder::Weak, MemEvictionPriority::Normal),
        bit76: false,
    };
    assert!(refused(Op::St(st)).is_ok());
    assert_eq!(
        refused(Op::St(OpSt { bit76: true, ..st })),
        Err("st has no bits for bit 76".into())
    );
}

/// The encoder refuses an `LDC` or atomic field the form it picks has no bits for, with
/// plain-word messages.
#[test]
fn unencodable_ldc_and_atomics_are_errors() {
    let ldc = OpLdc {
        dst: r(1),
        cbuf: CBufRef {
            buf: CBuf::BindlessUniform(ur(2)),
            offset: 0,
        },
        offset: Gpr::RZ,
        mode: LdcMode::IndexedLinear,
        mem_type: MemType::B32,
    };
    assert_eq!(
        refused(Op::Ldc(ldc)),
        Err("ldc has no bits for an indexing mode with a bindless buffer".into())
    );
    let atom = OpAtom {
        dst: AtomDst::Reg(r(1)),
        addr: addr(Some(urz(true)), true),
        data: r(3),
        op: AtomOp::Add,
        atom_type: AtomType::U32,
        space: global(MemOrder::Weak, MemEvictionPriority::Normal),
        pred_dst: Pred::PT,
    };
    assert!(refused(Op::Atom(atom)).is_ok());
    let red = OpAtom {
        dst: AtomDst::Reduction { ignored: Gpr::RZ },
        ..atom
    };
    assert!(refused(Op::Atom(red)).is_ok());
    let shared = OpAtom {
        space: MemSpace::Shared(OffsetStride::X1),
        addr: addr(None, false),
        ..atom
    };
    assert!(refused(Op::Atom(shared)).is_ok());
    let atom_cases = [
        (
            OpAtom {
                space: MemSpace::Local(MemEvictionPriority::Normal),
                ..atom
            },
            "atom has no bits for local memory".to_string(),
        ),
        (
            OpAtom {
                addr: addr(None, false),
                ..red
            },
            "atom has no bits for an address without a uniform register".into(),
        ),
        (
            OpAtom {
                op: AtomOp::Exch,
                ..red
            },
            "atomic operation 8 does not fit in 3 bits".into(),
        ),
        (
            OpAtom {
                pred_dst: p(1),
                ..red
            },
            "atom has no bits for a predicate destination".into(),
        ),
        (
            OpAtom {
                dst: AtomDst::Reduction { ignored: Gpr::RZ },
                ..shared
            },
            "atom has no bits for an atomic without a destination".into(),
        ),
        (
            OpAtom {
                pred_dst: p(1),
                ..shared
            },
            "atom has no bits for a predicate destination".into(),
        ),
        (
            OpAtom {
                op: AtomOp::CmpExch(AtomCmpSrc::Separate { cmpr: r(4) }),
                ..atom
            },
            "atom has no bits for a uniform address register".into(),
        ),
        // The surface atomics' packed compare source has no memory-atomic form; without the
        // refusal it would encode as an `ADD` (the operation field left clear).
        (
            OpAtom {
                op: AtomOp::CmpExch(AtomCmpSrc::Packed),
                ..atom
            },
            "atom has no bits for a packed compare source".into(),
        ),
        (
            OpAtom {
                op: AtomOp::CmpExch(AtomCmpSrc::Packed),
                ..shared
            },
            "atom has no bits for a packed compare source".into(),
        ),
    ];
    for (op, message) in atom_cases {
        assert_eq!(refused(Op::Atom(op)), Err(message));
    }
}

proptest! {
    #![proptest_config(config())]

    /// The group's operations alone, so each gets many more cases than in the crate-wide round
    /// trip: assembling then decoding gives the instruction back.
    #[test]
    fn memory_assemble_then_decode_round_trips(
        guard in crate::testing::any_pred_src(),
        op in Union::new(memory::any_ops()),
    ) {
        let instr = Instr { guard, op };
        let word = assemble(&instr).expect("the generator makes encodable instructions");
        prop_assert_eq!(decode(word), Ok(instr));
    }

    /// The group's words alone: a word decodes and re-assembles to itself (scheduling bits
    /// aside), or is refused for a value that names nothing, in one of the fields the module
    /// docs list. No form of the group has a layout that leaves a known bit unread.
    #[test]
    fn memory_decode_then_assemble_round_trips(word in any_word_of(memory::FORMS)) {
        match decode(word) {
            Ok(instr) => {
                let again = assemble(&instr).expect("a decoded instruction is encodable");
                prop_assert_eq!(again.bits(), word.bits() & !SCHEDULING.mask());
            }
            Err(DecodeError::UnknownValue { opcode, bit, width, .. }) => {
                let field = (bit, width);
                let refusable = match opcode {
                    0x981 | 0x381 | 0x986 | 0x386 | 0x98e | 0x3a9 => {
                        [(73, 3), (77, 4), (84, 3)].contains(&field)
                    }
                    0x983 | 0x987 | 0x387 => [(73, 3), (84, 3)].contains(&field),
                    0x984 | 0x988 | 0x388 | 0xb82 | 0x582 | 0x38d => field == (73, 3),
                    0x9a8 | 0x3a8 => [(73, 3), (77, 4), (84, 3), (87, 4)].contains(&field),
                    0x98c | 0x38c => [(73, 3), (87, 4)].contains(&field),
                    0x98f => field == (87, 4),
                    0x992 => field == (76, 3),
                    _ => false,
                };
                prop_assert!(refusable, "{opcode:#x} bit {bit}");
            }
            Err(error) => prop_assert!(false, "{error}"),
        }
    }
}
