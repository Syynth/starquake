//! PR #4588's cases whose titles name the uniform datapath
//! (`eden-pr4588/src/tests/video_core/native_shader.cpp` @ e5fb3b97), rebuilt with the test
//! assembler: the PR's words may come from games, so each
//! case is the same instruction built as an [`Instr`] and the same expectations, with the PR's
//! packed register numbers (`216 - (n / 16) * 16 + n % 16` for `URn`, bit `6 - n` of register
//! 239 for `UPn`) read as the uniform registers and predicates they stand for. The PR's
//! evaluators run one invocation; so does the interpreter (its warp has one active lane).
//!
//! # Which cases
//!
//! The set is every `TEST_CASE` of the file whose title contains "uniform" (ignoring case):
//! 53 cases. Ported here, by the PR's title (each test's doc names its case):
//!
//! - The cases whose instructions are all this group's: "uniform bit tests", "uniform carry
//!   chains", "uniform register effective addresses", "uniform permutation sign extends",
//!   "uniform bit reverse", "uniform comparisons and selects", "uniform immediate multiply",
//!   "immediate uniform comparison and select", "uniform signed right shift", "uniform
//!   predicate comparisons", "predicated uniform address arithmetic", "unsigned clamp",
//!   "uniform LUT preserves its bank", "uniform moves", "uniform logical right shift", "upper
//!   uniform registers", "uniform shifts honor their uniform execution predicate", "uniform
//!   right shifts preserve signedness", "uniform constant pairs", "uniform bitwise logic",
//!   "uniform loads retain the full five-bit constant buffer bank" and "merged high uniform bit
//!   reversal".
//! - The cases that mix this group's instructions with vector ones the integer group translates
//!   (`MOV`, `IADD3`, `IMAD`, `IMNMX`, `ISETP`, `LEA` and `PLOP3`, reading a uniform register or
//!   predicate), run through both translators: "warp uniform transfers" (its `R2UR` and `PLOP3`
//!   operations), "uniform high shifts and multiply operands", "histogram uniform leading bit",
//!   "uniform predicate LUT and ordinary copies", "extended uniform registers preserve
//!   independent carry state" and "uniform register mapping covers each reserved bank".
//! - The integer group's vector instructions reading a uniform register or predicate, run
//!   through the integer group's translator: "scalar uniform add", "scalar effective
//!   addresses", "mixed predicate AND", "three-input uniform addition", "uniform subtrahends"
//!   and "integer multiply preserves negated uniform operands".
//! - The uniform half of two cases whose other instructions are other groups' (an attribute
//!   load and store, `FMUL` and `FADD`): "uniform constants keep their indexed load" and
//!   "uniform 64-bit loads" (the uniform registers are checked where the PR checks the vertex
//!   outputs computed from them).
//!
//! Not ported, with the reason:
//!
//! - Only the lowering's register packing (its bank mapping, the extended bank and its
//!   shadow, its scratch registers), which this translator does not do (uniform registers keep
//!   their own class, the module docs): "extended uniform bank rejects overlapping general
//!   registers", "general registers below the used uniform slots remain available" and
//!   "merged uniform forms reserve high banks". The packing checks inside ported cases (the
//!   PR's registers 232 and 233 in "upper uniform registers" aside, which are kept as
//!   per-thread registers; the shadow registers 160 to 167 of "merged high uniform bit
//!   reversal"; the `0x12345600` above the bank in "extended uniform registers") are left out
//!   for the same reason.
//! - Other groups' instructions reading a uniform operand: the memory group's "scalar global
//!   stores", "global loads combine a uniform pointer", "uniform global reduction", "shared
//!   loads retain uniform offsets" and "uniform shared stores and loads"; the float group's
//!   "float predicate comparisons", "fused multiply add retains its uniform multiplier",
//!   "uniform reciprocal" and "fused multiply-add reads a uniform multiplier"; the attribute
//!   group's "indexed interpolation"; the texture group's "scalar texture samples" and
//!   "partial gathers"; and the control group's `VOTEU` (operations 2 and 3 of "warp uniform
//!   transfers").
//! - "Uniform native trap guards": the lowering's `VerifyTrapGuards`, not a translation.
//! - "Uniform predicate extraction selects UP0": its opcode `0x883` is not decoded (the
//!   decoder's `unknown opcode 0x883`), so it never reaches this group.
//! - The checks of the lowering's own output: the Maxwell opcode it picks ("unsigned clamp"'s
//!   `IMNMX_imm`, "uniform shifts"' `SHR_imm`) and the guard bits it emits ("predicated
//!   uniform address arithmetic", "uniform shifts"). The ports check the guard's effect
//!   instead: the instruction runs exactly where its uniform predicate holds.
//!
//! Where a port differs from the PR's case, its doc says how: "uniform 64-bit loads" and
//! "predicated uniform address arithmetic".

use super::alu::lea;
use super::*;
use crate::decode::{
    IMadMode, IntCmpOp, MovSrc, OpBRev, OpFlo, OpIAdd3X, OpIMad, OpIMnMx, OpISetP, OpLop3, OpMov,
    OpPrmt, OpSel, OpShf, PLop3Src, PredSetOp, PrmtMode, ShfType,
};

/// A uniform register as a vector instruction's source, negated or not.
fn usrc(index: u8, neg: bool) -> AluSrc {
    AluSrc::UniformReg {
        reg: ur(index),
        abs: false,
        neg,
    }
}

/// The function `instrs` translate to with guards honoured, through their words: this group's
/// by its translator, and the vector instructions some cases mix in by the integer group's.
fn mixed(instrs: &[Instr]) -> Function {
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    for instr in instrs {
        let word = word(instr);
        if in_group(&instr.op) {
            translate_uniform_word(&mut builder, word, GuardMode::Honour).expect("uniform");
        } else {
            integer::translate_integer_word(&mut builder, word, GuardMode::Honour)
                .expect("integer");
        }
    }
    builder.terminate(Terminator::Return).expect("terminates");
    assert_eq!(function.verify(), Ok(()));
    function
}

/// The uniform predicates as the PR packs them into register 239: `UPn` is bit `6 - n`.
fn set_bank(machine: &mut Machine, bank: u32) {
    for (n, slot) in (0_u32..7).zip(machine.uniform_preds.iter_mut()) {
        *slot = bank >> (6 - n) & 1 != 0;
    }
}

/// The PR's register 239 read back from the uniform predicates.
fn bank_of(machine: &Machine) -> u32 {
    (0_u32..7)
        .zip(machine.uniform_preds)
        .fold(0, |bank, (n, set)| bank | u32::from(set) << (6 - n))
}

/// The PR's alternating predicates, `(p & 1) != 0` for `P0` to `P6`.
fn alternating(machine: &mut Machine) {
    for (p, slot) in (0_u8..7).zip(machine.preds.iter_mut()) {
        *slot = p & 1 != 0;
    }
}

/// A shift right of the high source `high` by an immediate into a uniform register's high
/// form (`USHF.R.<type>.HI dst, URZ, shift, high`).
fn ushf_right(dst: u8, shift: u32, high: u8, signed: bool) -> Op {
    Op::UShf(twin(OpShf {
        dst: r(dst),
        low: RZ,
        shift: AluSrc::Imm(shift),
        high: reg(high),
        data_type: if signed { ShfType::I32 } else { ShfType::U32 },
        wrap: false,
        right: true,
        dst_high: true,
    }))
}

/// An unsigned `ISETP` of `a` against `b` into `dst` with the operation `set_op` and the
/// accumulator `accum` (a uniform one in a twin).
fn isetp(
    dst: u8,
    a: AluSrc,
    b: AluSrc,
    cmp_op: IntCmpOp,
    set_op: PredSetOp,
    accum: PredSrc,
) -> OpISetP {
    OpISetP {
        dsts: [p(dst), Pred::PT],
        srcs: [a, b],
        cmp_op,
        signed: false,
        set_op,
        ex: false,
        accum,
        low_cmp: PredSrc::TRUE,
    }
}

/// A signed `IMAD` with no carry.
fn imad(dst: u8, srcs: [AluSrc; 3]) -> OpIMad {
    OpIMad {
        dst: r(dst),
        srcs,
        signed: true,
        mode: IMadMode::Lo {
            x: false,
            carry: NOT_PT,
        },
    }
}

/// A `MOV` of a uniform register into a general register.
fn mov_from(dst: u8, src: u8) -> Instr {
    Instr::new(Op::Mov(OpMov {
        dst: r(dst),
        src: MovSrc::UniformReg(ur(src)),
        quad_lanes: OpMov::ALL_LANES,
    }))
}

/// A `PLOP3` of `a`, `b` and `c` into `dst` by `lut`.
fn plop3(dst: u8, a: PredSrc, b: PredSrc, c: PLop3Src, lut: u8) -> Instr {
    Instr::new(Op::PLop3(crate::decode::OpPLop3 {
        dsts: [p(dst), Pred::PT],
        srcs: [a, b],
        c,
        luts: [lut, 0],
    }))
}

/// "Native uniform bit tests update one uniform predicate and preserve scalar predicates":
/// `ULOP3.LUT URZ, UPn, UR10, mask, URZ, 0xc0` sets `UPn` to whether `UR10 & mask` is not
/// zero, for every `n` below 7, leaving the other uniform predicates, the per-thread
/// predicates and `UR10` as they were.
#[test]
fn uniform_bit_tests() {
    for mask in [0_u32, 1, 0x20000, 0x40000, 0xffff_ffff] {
        for output in 0..7_u8 {
            let instr = Instr::new(Op::ULop3(twin(OpLop3 {
                pred: p(output),
                dst: Gpr::RZ,
                srcs: [reg(10), AluSrc::Imm(mask), RZ],
                lut: 0xc0,
            })));
            let function = build(&[instr]).expect("translates");
            for value in [0_u32, 1, 2, 0x20000, 0x40000, 0x8000_0000, 0xffff_ffff] {
                for bank in [0_u8, 0x55, 0x7f] {
                    let mut machine = machine_with(&[]);
                    machine.uniform_regs[10] = value;
                    for (n, slot) in machine.uniform_preds.iter_mut().take(7).enumerate() {
                        *slot = bank >> n & 1 != 0;
                    }
                    machine.preds = [true, false, true, false, true, false, true, false];
                    let original = machine.clone();
                    machine.run(&function).expect("runs");
                    let mut expected = original.uniform_preds;
                    expected[usize::from(output)] = value & mask != 0;
                    assert_eq!(
                        machine.uniform_preds, expected,
                        "{mask:#x} {output} {value:#x}"
                    );
                    assert_eq!(machine.preds, original.preds);
                    assert_eq!(machine.uniform_regs[10], value);
                }
            }
        }
    }
}

/// "Native uniform carry chains preserve 64-bit overflow and both predicate banks":
/// `UIADD3 UR4, UPn, UR4, UR8, URZ` and `UIADD3.X UR5, URZ, UR9, URZ, UPn` give `UR4` the low
/// word of `UR4 + UR8`, `UR5` `UR9` plus its carry, and `UPn` the carry, for every `n` below
/// 7, leaving the per-thread predicates and `UR8` as they were.
#[test]
fn uniform_carry_chains() {
    for output in 0..7_u8 {
        let chain = [
            Instr::new(Op::UIAdd3(twin(OpIAdd3 {
                overflow: [p(output), Pred::PT],
                ..iadd3(4, [reg(4), reg(8), RZ])
            }))),
            Instr::new(Op::UIAdd3X(twin(OpIAdd3X {
                add: OpIAdd3 {
                    carry: [ps(output, false), NOT_PT],
                    ..iadd3(5, [RZ, reg(9), RZ])
                },
            }))),
        ];
        let function = build(&chain).expect("translates");
        for a in [0_u32, 1, 0xffff_fff0, 0xffff_ffff] {
            for b in [0_u32, 1, 32, 0xffff_ffff] {
                for high in [0_u32, 0xffff_ffff] {
                    let mut machine = machine_with(&[]);
                    machine.uniform_regs[4] = a;
                    machine.uniform_regs[8] = b;
                    machine.uniform_regs[9] = high;
                    machine.preds = [true, false, true, false, true, false, true, false];
                    let preds = machine.preds;
                    machine.run(&function).expect("runs");
                    let sum = u64::from(a) + u64::from(b);
                    let carry = u32::try_from(sum >> 32).expect("a carry");
                    let low = u32::try_from(sum & u64::from(u32::MAX)).expect("low");
                    assert_eq!(machine.uniform_regs[4], low, "{output} {a} {b}");
                    assert_eq!(machine.uniform_regs[5], high.wrapping_add(carry));
                    assert_eq!(machine.uniform_preds[usize::from(output)], carry != 0);
                    assert_eq!(machine.preds, preds);
                    assert_eq!(machine.uniform_regs[8], b);
                }
            }
        }
    }
}

/// "Native uniform register effective addresses preserve scaled addition and aliasing":
/// `ULEA UR4, UR5, UR8, shift` gives `(UR5 << shift) + UR8`, leaving both sources.
#[test]
fn uniform_effective_addresses() {
    for shift in [0_u8, 1, 2, 16, 31] {
        let function =
            build(&[Instr::new(Op::ULea(twin(lea(4, reg(5), reg(8), shift))))]).expect("ok");
        for value in [0_u32, 1, 0x8000_0000, 0xffff_ffff] {
            let mut machine = machine_with(&[]);
            machine.uniform_regs[5] = value;
            machine.uniform_regs[8] = 0xffff_fff0;
            machine.run(&function).expect("runs");
            assert_eq!(
                machine.uniform_regs[4],
                (value << shift).wrapping_add(0xffff_fff0),
                "{shift} {value:#x}"
            );
            assert_eq!(machine.uniform_regs[5], value);
            assert_eq!(machine.uniform_regs[8], 0xffff_fff0);
        }
    }
}

/// "Native uniform permutation sign extends its low half and supports aliasing":
/// `UPRMT URd, UR8, 0x9910, URZ` gives the low half-word of `UR8` sign-extended, into `UR6`
/// and into `UR8` itself.
#[test]
fn uniform_permutation_sign_extends() {
    for destination in [6_u8, 8] {
        let function = build(&[Instr::new(Op::UPrmt(twin(OpPrmt {
            dst: r(destination),
            srcs: [reg(8), RZ],
            sel: AluSrc::Imm(0x9910),
            mode: PrmtMode::Index,
        })))])
        .expect("translates");
        for value in [0_u32, 1, 0x7fff, 0x8000, 0xffff, 0xdead_8000, 0xabcd_7fff] {
            let mut machine = machine_with(&[]);
            machine.uniform_regs[8] = value;
            machine.run(&function).expect("runs");
            let half = u16::try_from(value & 0xffff).expect("a half-word");
            let expected = i32::from(half.cast_signed()).cast_unsigned();
            assert_eq!(machine.uniform_regs[usize::from(destination)], expected);
            if destination != 8 {
                assert_eq!(machine.uniform_regs[8], value);
            }
        }
    }
}

/// "Native uniform bit reverse preserves every bit and permits source destination aliases":
/// `UBREV URd, UR4` reverses every bit, into each destination the PR checks (`UR4` itself
/// among them); from `URZ` it gives zero.
#[test]
fn uniform_bit_reverse() {
    let mut inputs = vec![0_u32, !0, 0x0123_4567, 0x89ab_cdef];
    inputs.extend((0..32).map(|bit| 1_u32 << bit));
    for dest in [4_u8, 17, 18, 30, 32, 46, 48, 62] {
        let brev = |src: AluSrc| {
            build(&[Instr::new(Op::UBRev(twin(OpBRev { dst: r(dest), src })))]).expect("translates")
        };
        let function = brev(reg(4));
        for value in &inputs {
            let mut machine = machine_with(&[]);
            machine.uniform_regs[4] = *value;
            machine.run(&function).expect("runs");
            assert_eq!(
                machine.uniform_regs[usize::from(dest)],
                value.reverse_bits(),
                "{dest} {value:#x}"
            );
        }
        let mut machine = machine_with(&[]);
        machine.uniform_regs[usize::from(dest)] = 1;
        machine.run(&brev(RZ)).expect("runs");
        assert_eq!(machine.uniform_regs[usize::from(dest)], 0);
    }
}

/// "Native warp uniform transfers use the first selected active lane and preserve
/// predicates", operations 0 and 1 (`@P1 R2UR UR5, R2` and `@!P1 R2UR UR5, R2`) and 4 to 7
/// (`PLOP3`s reading `UP0` or per-thread predicates, through the integer group), on lanes 0, 6
/// and 31 of the interpreter's one-lane warp: the active mask is the running lane, so the
/// ballot is its guard. `UR5` takes the lane's `R2` (`100 + lane * 17`) where the guard holds
/// and keeps `0xcafebeef` where it does not (the PR's `ballot ? 100 + first * 17 :
/// 0xcafebeef`); every predicate keeps its value but a `PLOP3`'s destination.
#[test]
fn warp_uniform_transfers() {
    let r2ur = |negated| {
        under(
            ps(1, negated),
            Op::R2UR(OpR2UR {
                dst: ur(5),
                src: r(2),
            }),
        )
    };
    let pt = PredSrc::TRUE;
    let operations = [
        (0, r2ur(false)),
        (1, r2ur(true)),
        (4, plop3(1, pt, pt, PLop3Src::Uniform(ups(0, false)), 0x80)),
        (5, plop3(0, pt, pt, PLop3Src::Pred(pt), 0x80)),
        (6, plop3(0, pt, pt, PLop3Src::Pred(pt), 0x08)),
        (
            7,
            plop3(2, ps(1, false), ps(0, false), PLop3Src::Pred(pt), 0x80),
        ),
    ];
    for (operation, instr) in operations {
        let function = mixed(&[instr]);
        for lane in [0_u32, 6, 31] {
            for selected in [false, true] {
                let mut machine = Machine::new();
                machine.system.lane_id = lane;
                machine.regs[2] = 100 + lane * 17;
                machine.uniform_regs[5] = 0xcafe_beef;
                set_bank(&mut machine, if selected { 64 } else { 0 });
                for (p, slot) in (0_u32..7).zip(machine.preds.iter_mut()) {
                    *slot = (lane + p) & 1 != 0;
                }
                machine.preds[1] = selected != (operation == 4);
                let original = machine.preds;
                machine.run(&function).expect("runs");
                let mut expected = original;
                match operation {
                    7 => expected[2] = original[1] && original[0],
                    4 => expected[1] = selected,
                    5 | 6 => expected[0] = operation == 5,
                    _ => {}
                }
                let at = format!("{operation} {lane} {selected}");
                assert_eq!(machine.preds, expected, "{at}");
                if operation < 2 {
                    let ballot = selected != (operation == 1);
                    let value = if ballot { 100 + lane * 17 } else { 0xcafe_beef };
                    assert_eq!(machine.uniform_regs[5], value, "{at}");
                }
            }
        }
    }
}

/// "Native uniform constants keep their indexed load and arithmetic values", its uniform half:
/// `ULDC UR4, c[0x3][0x0]`, `USHF.L.U32 UR4, UR4, 0x2, URZ`, `ULDC UR5, c[0x3][UR4+0x0]` read the
/// word the first constant indexes. The PR's attribute load, `FMUL`, `FADD` and store compute
/// its outputs from `UR5` (`input * UR5 - UR5`); `UR5` is checked instead, for the PR's first
/// word 1 and for 2 and 3.
#[test]
fn uniform_constants_keep_their_indexed_load() {
    let function = mixed(&[
        Instr::new(Op::ULdc(OpULdc {
            dst: ur(4),
            src: ULdcSrc::CBuf(CBufRef {
                buf: CBuf::Binding(3),
                offset: 0,
            }),
            mem_type: MemType::B32,
        })),
        Instr::new(Op::UShf(twin(OpShf {
            dst: r(4),
            low: reg(4),
            shift: AluSrc::Imm(2),
            high: RZ,
            data_type: ShfType::U32,
            wrap: false,
            right: false,
            dst_high: false,
        }))),
        Instr::new(Op::ULdc(OpULdc {
            dst: ur(5),
            src: ULdcSrc::Indexed {
                index: 3,
                reg: ur(4),
                offset: 0,
            },
            mem_type: MemType::B32,
        })),
    ]);
    let constants = [3.0_f32, 7.0, 9.0].map(f32::to_bits);
    for first in [1_u32, 2, 3] {
        let mut machine = machine_with(&[]);
        let words = [first, constants[0], constants[1], constants[2]];
        machine.cbufs.insert(
            3,
            words.iter().flat_map(|word| word.to_le_bytes()).collect(),
        );
        machine.run(&function).expect("runs");
        assert_eq!(machine.uniform_regs[4], first * 4, "{first}");
        let index = usize::try_from(first).expect("an index");
        assert_eq!(machine.uniform_regs[5], words[index], "{first}");
    }
}

/// "Native uniform 64-bit loads populate consecutive registers", its uniform half and the
/// moves: `ULDC UR5, c[0x3][0x0]`, `ULDC.64 UR6, c[0x3][UR5+0x4]`, `MOV R0, UR6`, `MOV R1,
/// UR7` (the PR stores `R0` and `R1` as the vertex position). One difference: the first word
/// holds 4 where the PR's holds 0, so that the 64-bit read lands at byte 8. The interpreter
/// aligns a 64-bit constant read down to 8 bytes, as Eden's SPIR-V `GetCbufU32x2` does, where
/// the PR's evaluator reads at byte 4; the translated read itself is checked to be at
/// `UR5 + 4` as the PR's is.
#[test]
fn uniform_64_bit_loads() {
    let instrs = [
        Instr::new(Op::ULdc(OpULdc {
            dst: ur(5),
            src: ULdcSrc::CBuf(CBufRef {
                buf: CBuf::Binding(3),
                offset: 0,
            }),
            mem_type: MemType::B32,
        })),
        Instr::new(Op::ULdc(OpULdc {
            dst: ur(6),
            src: ULdcSrc::Indexed {
                index: 3,
                reg: ur(5),
                offset: 4,
            },
            mem_type: MemType::B64,
        })),
        mov_from(0, 6),
        mov_from(1, 7),
    ];
    let function = mixed(&instrs);
    let text = function.to_string();
    assert!(
        text.contains("IAdd32 %2, u32 0x4\n    %4: u32x2 = GetCbufU32x2 u32 0x3, %3\n"),
        "{text}"
    );
    let mut machine = machine_with(&[]);
    let words = [4, 3.0_f32.to_bits(), 7.0_f32.to_bits(), 9.0_f32.to_bits()];
    machine.cbufs.insert(
        3,
        words.iter().flat_map(|word| word.to_le_bytes()).collect(),
    );
    machine.run(&function).expect("runs");
    assert_eq!(machine.uniform_regs[6..8], words[2..4]);
    assert_eq!(machine.regs[0..2], words[2..4]);
}

/// "Native uniform comparisons and selects preserve independent predicate banks": a `ULEA`,
/// two `UISETP.LT.U32` and `USEL` pairs and a `UIADD3` compute `width - min(192, width)` into
/// `UR5` and `min(40, 32)` into `UR4`, with `UP1` whether `192 < width` and `UP0` false; the
/// per-thread predicates keep their values.
#[test]
fn uniform_comparisons_and_selects() {
    let lt = |dst, a, b| {
        Instr::new(Op::UISetP(twin(isetp(
            dst,
            reg(a),
            reg(b),
            IntCmpOp::Lt,
            PredSetOp::And,
            PredSrc::TRUE,
        ))))
    };
    let sel = |dst, b, cond| {
        Instr::new(Op::USel(twin(OpSel {
            dst: r(dst),
            srcs: [reg(dst), reg(b)],
            cond: ps(cond, false),
        })))
    };
    let function = mixed(&[
        Instr::new(Op::ULea(twin(lea(5, reg(9), AluSrc::Imm(0x40), 6)))),
        lt(1, 5, 7),
        sel(5, 7, 1),
        Instr::new(Op::UIAdd3(twin(iadd3(5, [neg(5), reg(7), RZ])))),
        lt(0, 4, 6),
        sel(4, 6, 0),
    ]);
    for width in [100_u32, 256] {
        let mut machine = machine_with(&[]);
        machine.uniform_regs[9] = 2;
        machine.uniform_regs[7] = width;
        machine.uniform_regs[4] = 40;
        machine.uniform_regs[6] = 32;
        alternating(&mut machine);
        let original = machine.preds;
        machine.run(&function).expect("runs");
        assert_eq!(machine.uniform_regs[5], width - width.min(192), "{width}");
        assert_eq!(machine.uniform_regs[4], 32);
        assert_eq!(machine.uniform_preds[1], 192 < width);
        assert!(!machine.uniform_preds[0]);
        assert_eq!(machine.preds, original);
    }
}

/// "Native uniform high shifts and multiply operands retain numeric values": `UMOV UR5,
/// value`, `USHF.L.U32.HI UR4, URZ, 0x5, UR5`, `USHF.L.U32 UR5, UR5, 0x5, URZ`, then the vector
/// `IMAD R2, R1, UR5, R2`, `IMNMX R0, R0, UR4, PT` and `IMAD R10, R0, R5, 0x4`.
#[test]
fn uniform_high_shifts_and_multiply_operands() {
    let shl = |dst, low, high, dst_high| {
        Instr::new(Op::UShf(twin(OpShf {
            dst: r(dst),
            low,
            shift: AluSrc::Imm(5),
            high,
            data_type: ShfType::U32,
            wrap: false,
            right: false,
            dst_high,
        })))
    };
    for value in [0_u32, 1, 123, 0x07ff_ffff, 0xffff_ffff] {
        let function = mixed(&[
            Instr::new(Op::UMov(OpUMov {
                dst: ur(5),
                src: UMovSrc::Imm(value),
            })),
            shl(4, RZ, reg(5), true),
            shl(5, reg(5), RZ, false),
            Instr::new(Op::IMad(imad(2, [reg(1), usrc(5, false), reg(2)]))),
            Instr::new(Op::IMnMx(OpIMnMx {
                dst: r(0),
                srcs: [reg(0), usrc(4, false)],
                signed: true,
                min: PredSrc::TRUE,
            })),
            Instr::new(Op::IMad(imad(10, [reg(0), reg(5), AluSrc::Imm(4)]))),
        ]);
        let mut machine = Machine::new();
        machine.regs[0] = 5000;
        machine.regs[1] = 11;
        machine.regs[2] = 7;
        machine.regs[5] = 3;
        machine.run(&function).expect("runs");
        let shifted = value << 5;
        let minimum = 5000_i32.min(shifted.cast_signed()).cast_unsigned();
        assert_eq!(machine.uniform_regs[4], shifted, "{value:#x}");
        assert_eq!(machine.uniform_regs[5], shifted, "{value:#x}");
        assert_eq!(
            machine.regs[2],
            11_u32.wrapping_mul(shifted).wrapping_add(7)
        );
        assert_eq!(machine.regs[0], minimum, "{value:#x}");
        assert_eq!(machine.regs[10], minimum.wrapping_mul(3).wrapping_add(4));
    }
}

/// "Native uniform immediate multiply preserves wraparound and aliased operands": `UIMAD UR4,
/// UR4, multiplier, URZ` wraps, leaving `UR5` and the predicates.
#[test]
fn uniform_immediate_multiply() {
    for multiplier in [0_u32, 1, 5, 48, 0xffff_ffff] {
        let function = mixed(&[Instr::new(Op::UIMad(twin(imad(
            4,
            [reg(4), AluSrc::Imm(multiplier), RZ],
        ))))]);
        for value in [0_u32, 1, 123, 0x8000_0000, 0xffff_ffff] {
            let mut machine = Machine::new();
            machine.uniform_regs[4] = value;
            machine.uniform_regs[5] = 0xcafe_beef;
            machine.preds = [true; 8];
            machine.run(&function).expect("runs");
            assert_eq!(machine.uniform_regs[4], value.wrapping_mul(multiplier));
            assert_eq!(machine.uniform_regs[5], 0xcafe_beef);
            assert_eq!(machine.preds, [true; 8]);
        }
    }
}

/// "Native histogram uniform leading bit and comparison select the highest active lane":
/// `UFLO.U32 UR4, UR4`, then the vector `ISETP.EQ.U32.AND P0, PT, R2, UR4, PT`.
#[test]
fn histogram_uniform_leading_bit() {
    let function = mixed(&[
        Instr::new(Op::UFlo(twin(OpFlo {
            dst: r(4),
            src: reg(4),
            signed: false,
            return_shift_amount: false,
        }))),
        Instr::new(Op::ISetP(isetp(
            0,
            reg(2),
            usrc(4, false),
            IntCmpOp::Eq,
            PredSetOp::And,
            PredSrc::TRUE,
        ))),
    ]);
    for mask in [1_u32, 0x8000_0000, 0x5555_5555, 0xffff_ffff] {
        for lane in 0..32_u32 {
            let mut machine = Machine::new();
            machine.uniform_regs[4] = mask;
            machine.regs[2] = lane;
            machine.run(&function).expect("runs");
            let highest = mask.ilog2();
            assert_eq!(machine.uniform_regs[4], highest, "{mask:#x}");
            assert_eq!(machine.preds[0], lane == highest, "{mask:#x} {lane}");
        }
    }
}

/// "Native immediate uniform comparison and select preserve predicates at unsigned
/// boundaries": `UISETP.LT.U32 UP0, UPT, UR4, 0x6, UPT` and `USEL UR5, UR4, 0x6, UP0`.
#[test]
fn immediate_uniform_comparison_and_select() {
    let function = mixed(&[
        Instr::new(Op::UISetP(twin(isetp(
            0,
            reg(4),
            AluSrc::Imm(6),
            IntCmpOp::Lt,
            PredSetOp::And,
            PredSrc::TRUE,
        )))),
        Instr::new(Op::USel(twin(OpSel {
            dst: r(5),
            srcs: [reg(4), AluSrc::Imm(6)],
            cond: ps(0, false),
        }))),
    ]);
    for value in [0_u32, 5, 6, 7, 0x8000_0000, 0xffff_ffff] {
        let mut machine = Machine::new();
        machine.uniform_regs[4] = value;
        alternating(&mut machine);
        let saved = machine.preds;
        machine.run(&function).expect("runs");
        assert_eq!(machine.uniform_preds[0], value < 6, "{value:#x}");
        assert_eq!(machine.uniform_regs[5], value.min(6), "{value:#x}");
        assert_eq!(machine.preds, saved);
    }
}

/// "Native uniform signed right shift sign extends the high word": `USHF.R.S32.HI UR5, URZ,
/// 0x1f, UR4`.
#[test]
fn uniform_signed_right_shift() {
    let function = mixed(&[Instr::new(ushf_right(5, 31, 4, true))]);
    for value in [0_u32, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff] {
        let mut machine = Machine::new();
        machine.uniform_regs[4] = value;
        machine.run(&function).expect("runs");
        let expected = if value & 0x8000_0000 != 0 {
            u32::MAX
        } else {
            0
        };
        assert_eq!(machine.uniform_regs[5], expected, "{value:#x}");
    }
}

/// "Native uniform predicate comparisons preserve all seven predicates and accumulation":
/// `UISETP.NE.U32.<op> UPn, UPT, UR7, 0x1, [!]UPn` for every `n` below 7, each operation and
/// each accumulator negation: `UPn` becomes `(UR7 != 1) op (UPn ^ negated)`, the other uniform
/// predicates, the per-thread ones and `UR7` unchanged.
#[test]
fn uniform_predicate_comparisons() {
    for output in 0..7_u8 {
        for set_op in [PredSetOp::And, PredSetOp::Or, PredSetOp::Xor] {
            for invert in [false, true] {
                let function = mixed(&[Instr::new(Op::UISetP(twin(isetp(
                    output,
                    reg(7),
                    AluSrc::Imm(1),
                    IntCmpOp::Ne,
                    set_op,
                    ps(output, invert),
                ))))]);
                for bank in [0_u32, 0x2a, 0x55, 0x7f] {
                    for value in [0_u32, 1, 2, 0xffff_ffff] {
                        let mut machine = Machine::new();
                        machine.uniform_regs[7] = value;
                        set_bank(&mut machine, bank);
                        alternating(&mut machine);
                        let original = machine.preds;
                        machine.run(&function).expect("runs");
                        let mask = 1 << (6 - output);
                        let a = value != 1;
                        let b = (bank & mask != 0) != invert;
                        let expected = match set_op {
                            PredSetOp::And => a && b,
                            PredSetOp::Or => a || b,
                            PredSetOp::Xor => a != b,
                        };
                        let bits = (bank & !mask) | if expected { mask } else { 0 };
                        let at = format!("{output} {set_op:?} {invert} {bank:#x} {value:#x}");
                        assert_eq!(bank_of(&machine), bits, "{at}");
                        assert_eq!(machine.preds, original, "{at}");
                        assert_eq!(machine.uniform_regs[7], value, "{at}");
                    }
                }
            }
        }
    }
}

/// "Native uniform predicate LUT and ordinary copies keep separate banks": `UPLOP3.LUT UPn, UPT,
/// UP1, UPT, UP2, lut` with `lut` `0x80` and `0x40` (`UP1 && (UP2 ^ (lut == 0x40))`) for every
/// `n` below 7 and every bank, then the vector `PLOP3.LUT P2, PT, PT, PT, UPn, lut` (`UPn`,
/// inverted for `0x40`), each leaving the other file as it was.
#[test]
fn uniform_predicate_lut_and_ordinary_copies() {
    for output in 0..7_u8 {
        for lut in [0x80_u8, 0x40] {
            let function = mixed(&[Instr::new(Op::UPLop3(OpUPLop3 {
                dsts: [up(output), UniformPred::UPT],
                srcs: [
                    ups(1, false),
                    UniformPredSrc {
                        reg: UniformPred::UPT,
                        negated: false,
                    },
                    ups(2, false),
                ],
                luts: [lut, 0],
            }))]);
            for bank in 0..128_u32 {
                let mut machine = Machine::new();
                set_bank(&mut machine, bank);
                alternating(&mut machine);
                let original = machine.preds;
                machine.run(&function).expect("runs");
                let expected = bank & 1 << 5 != 0 && ((bank & 1 << 4 != 0) != (lut == 0x40));
                let mask = 1 << (6 - output);
                let bits = (bank & !mask) | if expected { mask } else { 0 };
                assert_eq!(bank_of(&machine), bits, "{output} {lut:#x} {bank:#x}");
                assert_eq!(machine.preds, original);
            }
        }
    }
    let pt = PredSrc::TRUE;
    for source in 0..7_u8 {
        for invert in [false, true] {
            let lut = if invert { 0x40 } else { 0x80 };
            let function = mixed(&[plop3(2, pt, pt, PLop3Src::Uniform(ups(source, false)), lut)]);
            for bank in [0_u32, 0x2a, 0x55, 0x7f] {
                let mut machine = Machine::new();
                set_bank(&mut machine, bank);
                machine.preds = [true; 8];
                machine.run(&function).expect("runs");
                let held = bank & 1 << (6 - source) != 0;
                assert_eq!(
                    machine.preds[2],
                    held != invert,
                    "{source} {invert} {bank:#x}"
                );
                assert_eq!(bank_of(&machine), bank);
                for (p, value) in machine.preds.iter().enumerate().take(7) {
                    assert!(p == 2 || *value, "{source} {invert} {bank:#x} P{p}");
                }
            }
        }
    }
}

/// "Native predicated uniform address arithmetic uses the selected uniform predicate": `ULEA
/// UR5, UR4, 0x1, 0x2` under `UPn` and `!UPn` for every `n` below 7 runs exactly where its
/// uniform predicate (not the per-thread one, which holds everywhere) holds, and leaves `UR5`
/// otherwise; under `!UPT` it appends nothing, where the lowering refuses it ("native inverted
/// uniform true predicate", the module docs' "Divergences"). The PR checks the guard bits the
/// lowering emits; the port checks the guard's effect.
#[test]
fn predicated_uniform_address_arithmetic() {
    let ulea = Op::ULea(twin(lea(5, reg(4), AluSrc::Imm(1), 2)));
    let mut function = Function::new();
    let entry = function.entry();
    let mut builder = Builder::new(&mut function, entry);
    assert_eq!(
        translate_uniform_word(&mut builder, word(&under(NOT_PT, ulea)), GuardMode::Honour),
        Ok(())
    );
    assert_eq!(builder.function(), &Function::new());
    for condition in 0..7_u8 {
        for invert in [false, true] {
            let function = mixed(&[under(ps(condition, invert), ulea)]);
            for bank in [0_u32, 0x2a, 0x55, 0x7f] {
                let mut machine = Machine::new();
                set_bank(&mut machine, bank);
                machine.preds = [true; 8];
                machine.uniform_regs[4] = 0x10;
                machine.uniform_regs[5] = 0xcafe;
                machine.run(&function).expect("runs");
                let holds = (bank & 1 << (6 - condition) != 0) != invert;
                let expected = if holds { (0x10 << 2) + 1 } else { 0xcafe };
                let at = format!("{condition} {invert} {bank:#x}");
                assert_eq!(machine.uniform_regs[5], expected, "{at}");
                assert_eq!(bank_of(&machine), bank, "{at}");
                assert_eq!(machine.preds, [true; 8], "{at}");
            }
        }
    }
}

/// "Native unsigned clamp tracks the selected uniform predicate": `UISETP.LT.U32 UPn, UPT,
/// UR4, 0x2, UPT` and `USEL UR4, UR4, 0x2, UPn` clamp `UR4` to 2 for every `n` below 7; with
/// the select reading `UP(n ^ 1)` instead (the PR's mismatch, `n` other than 6), the select
/// follows that predicate of the bank `0x55`. The PR's check of the Maxwell opcode the lowering
/// picks (`IMNMX_imm`) is the lowering's own.
#[test]
fn unsigned_clamp() {
    let program = |compare: u8, select: u8| {
        mixed(&[
            Instr::new(Op::UISetP(twin(isetp(
                compare,
                reg(4),
                AluSrc::Imm(2),
                IntCmpOp::Lt,
                PredSetOp::And,
                PredSrc::TRUE,
            )))),
            Instr::new(Op::USel(twin(OpSel {
                dst: r(4),
                srcs: [reg(4), AluSrc::Imm(2)],
                cond: ps(select, false),
            }))),
        ])
    };
    for condition in 0..7_u8 {
        let mut cases = vec![(condition, program(condition, condition))];
        if condition != 6 {
            cases.push((condition ^ 1, program(condition, condition ^ 1)));
        }
        for (select, function) in cases {
            for value in [0_u32, 1, 2, 3, 0xffff_ffff] {
                let mut machine = Machine::new();
                machine.uniform_regs[4] = value;
                set_bank(&mut machine, 0x55);
                machine.preds = [true; 8];
                machine.run(&function).expect("runs");
                let chosen = if select == condition {
                    value < 2
                } else {
                    0x55 & 1 << (6 - select) != 0
                };
                let expected = if chosen { value } else { 2 };
                assert_eq!(
                    machine.uniform_regs[4], expected,
                    "{condition} {select} {value}"
                );
                assert_eq!(machine.preds, [true; 8]);
            }
        }
    }
}

/// "Native uniform LUT preserves its bank and aliases its source": `ULOP3.LUT UR4, UR4,
/// 0x3fffffff, URZ, 0xc0, !UPT` masks `UR4` in place, leaving both predicate files.
#[test]
fn uniform_lut_preserves_its_bank() {
    let function = mixed(&[Instr::new(Op::ULop3(twin(OpLop3 {
        pred: Pred::PT,
        dst: r(4),
        srcs: [reg(4), AluSrc::Imm(0x3fff_ffff), RZ],
        lut: 0xc0,
    })))]);
    for value in [0_u32, 1, 0x3fff_ffff, 0x8000_0000, 0xffff_ffff] {
        let mut machine = Machine::new();
        machine.uniform_regs[4] = value;
        set_bank(&mut machine, 0x55);
        machine.preds = [true; 8];
        machine.run(&function).expect("runs");
        assert_eq!(machine.uniform_regs[4], value & 0x3fff_ffff, "{value:#x}");
        assert_eq!(bank_of(&machine), 0x55);
        assert_eq!(machine.preds, [true; 8]);
    }
}

/// "Native scalar uniform add preserves wraparound and its source": the vector `IADD3 R20, R0,
/// UR4, RZ`.
#[test]
fn scalar_uniform_add() {
    let function = mixed(&[Instr::new(Op::IAdd3(iadd3(
        20,
        [reg(0), usrc(4, false), RZ],
    )))]);
    for a in [0_u32, 1, 0xffff_ffff] {
        for b in [0_u32, 1, 0x8000_0000, 0xffff_ffff] {
            let mut machine = Machine::new();
            machine.regs[0] = a;
            machine.uniform_regs[4] = b;
            machine.run(&function).expect("runs");
            assert_eq!(machine.regs[20], a.wrapping_add(b), "{a:#x} {b:#x}");
            assert_eq!(machine.regs[0], a);
            assert_eq!(machine.uniform_regs[4], b);
        }
    }
}

/// "Native uniform moves read the zero register without aliasing the bank": `UMOV UR4, URn`
/// for the PR's sources, `URZ` (63) reading zero, the predicate bank unchanged; sources 64 and
/// 255 do not decode (the PR's lowering refuses them).
#[test]
fn uniform_moves() {
    for source in [0_u8, 5, 15, 16, 17, 18, 30, 32, 46, 48, 62, 63] {
        let umov = Instr::new(Op::UMov(OpUMov {
            dst: ur(4),
            src: UMovSrc::Reg(UniformReg::new(source).expect("a uniform register")),
        }));
        let function = mixed(&[umov]);
        let mut machine = Machine::new();
        machine.uniform_regs = [0xa5a5_a5a5; crate::interp::UNIFORM_REG_SLOTS];
        set_bank(&mut machine, 0xa5a5_a5a5 & 0x7f);
        if source != 63 {
            machine.uniform_regs[usize::from(source)] = 0x1234_0000 | u32::from(source);
        }
        machine.run(&function).expect("runs");
        let expected = if source == 63 {
            0
        } else {
            0x1234_0000 | u32::from(source)
        };
        assert_eq!(machine.uniform_regs[4], expected, "{source}");
        assert_eq!(bank_of(&machine), 0xa5a5_a5a5 & 0x7f, "{source}");
    }
    let base = word(&Instr::new(Op::UMov(OpUMov {
        dst: ur(4),
        src: UMovSrc::Reg(ur(5)),
    })))
    .bits();
    for source in [64_u128, 255] {
        let bits = (base & !(0xff << 32)) | source << 32;
        let mut function = Function::new();
        let entry = function.entry();
        let mut builder = Builder::new(&mut function, entry);
        assert_eq!(
            translate_uniform_word(
                &mut builder,
                Instruction::from_bits(bits),
                GuardMode::Honour
            ),
            Err(UniformError::Decode(DecodeError::UnknownBit {
                opcode: 0xc82,
                bit: 38,
            })),
            "{source}"
        );
    }
}

/// "Native uniform logical right shift retains unsigned high bits": `USHF.R.U32.HI UR5, URZ,
/// shift, UR4`, zero for a shift of 32, leaving `UR4`.
#[test]
fn uniform_logical_right_shift() {
    for shift in [0_u32, 1, 2, 16, 31, 32] {
        let function = mixed(&[Instr::new(ushf_right(5, shift, 4, false))]);
        for value in [0_u32, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff] {
            let mut machine = Machine::new();
            machine.uniform_regs[4] = value;
            machine.run(&function).expect("runs");
            let expected = value.checked_shr(shift).unwrap_or(0);
            assert_eq!(machine.uniform_regs[5], expected, "{shift} {value:#x}");
            assert_eq!(machine.uniform_regs[4], value);
        }
    }
}

/// "Native upper uniform registers remain separate from predicate state": `UMOV UR16,
/// 0xffffffff`, `UMOV UR17, 0x12345678` and `UIADD3 UR16, UR16, 0x2, URZ` leave `UR16` 1 and
/// `UR17` `0x12345678`, with the predicate bank and the per-thread registers 232 and 233 (the
/// PR's registers of the same numbers) unchanged.
#[test]
fn upper_uniform_registers() {
    let function = mixed(&[
        Instr::new(Op::UMov(OpUMov {
            dst: ur(16),
            src: UMovSrc::Imm(0xffff_ffff),
        })),
        Instr::new(Op::UMov(OpUMov {
            dst: ur(17),
            src: UMovSrc::Imm(0x1234_5678),
        })),
        Instr::new(Op::UIAdd3(twin(iadd3(16, [reg(16), AluSrc::Imm(2), RZ])))),
    ]);
    let mut machine = Machine::new();
    set_bank(&mut machine, 0x55);
    machine.regs[232] = 0xabcd_ef12;
    machine.regs[233] = 0x7654_3210;
    machine.run(&function).expect("runs");
    assert_eq!(machine.uniform_regs[16], 1);
    assert_eq!(machine.uniform_regs[17], 0x1234_5678);
    assert_eq!(bank_of(&machine), 0x55);
    assert_eq!(machine.regs[232], 0xabcd_ef12);
    assert_eq!(machine.regs[233], 0x7654_3210);
}

/// "Native scalar effective addresses add the selected uniform base": the vector `LEA R28,
/// R53, UR5, 0x2`.
#[test]
fn scalar_effective_addresses() {
    let function = mixed(&[Instr::new(Op::Lea(lea(28, reg(53), usrc(5, false), 2)))]);
    for value in [0_u32, 1, 0x8000_0000, 0xffff_ffff] {
        let mut machine = Machine::new();
        machine.regs[53] = value;
        machine.uniform_regs[5] = 0xffff_fff0;
        machine.run(&function).expect("runs");
        assert_eq!(machine.regs[28], (value << 2).wrapping_add(0xffff_fff0));
        assert_eq!(machine.regs[53], value);
        assert_eq!(machine.uniform_regs[5], 0xffff_fff0);
    }
}

/// "Native mixed predicate AND retains its scalar input and uniform bank": the vector
/// `PLOP3.LUT P2, PT, P1, PT, UP4, 0x40` gives `P1 && !UP4`.
#[test]
fn mixed_predicate_and() {
    let function = mixed(&[plop3(
        2,
        ps(1, false),
        PredSrc::TRUE,
        PLop3Src::Uniform(ups(4, false)),
        0x40,
    )]);
    for bank in 0..128_u32 {
        for scalar in [false, true] {
            let mut machine = Machine::new();
            set_bank(&mut machine, bank);
            machine.preds = [true; 8];
            machine.preds[1] = scalar;
            let mut expected = machine.preds;
            expected[2] = scalar && bank & 1 << 2 == 0;
            machine.run(&function).expect("runs");
            assert_eq!(machine.preds, expected, "{bank:#x} {scalar}");
            assert_eq!(bank_of(&machine), bank);
        }
    }
}

/// "Native uniform shifts honor their uniform execution predicate": `USHF.R.U32.HI UR7, URZ,
/// 0x2, UR6` under `UPn` and `!UPn` for every `n` below 7 runs exactly where its uniform
/// predicate holds. The PR checks the guard bits and opcodes the lowering emits; the port
/// checks the guard's effect.
#[test]
fn uniform_shifts_honor_their_execution_predicate() {
    for condition in 0..7_u8 {
        for invert in [false, true] {
            let function = mixed(&[under(ps(condition, invert), ushf_right(7, 2, 6, false))]);
            for bank in [0_u32, 0x2a, 0x55, 0x7f] {
                let mut machine = Machine::new();
                set_bank(&mut machine, bank);
                machine.preds = [true; 8];
                machine.uniform_regs[6] = 0x8000_0040;
                machine.uniform_regs[7] = 0xcafe;
                machine.run(&function).expect("runs");
                let holds = (bank & 1 << (6 - condition) != 0) != invert;
                let expected = if holds { 0x2000_0010 } else { 0xcafe };
                assert_eq!(
                    machine.uniform_regs[7], expected,
                    "{condition} {invert} {bank:#x}"
                );
            }
        }
    }
}

/// "Native three-input uniform addition retains all operands and aliases": the vector `IADD3
/// Rd, [P0,] R0, UR3, R2` (or `RZ`) into `R0`, `R2` and `R4`, with or without the carry
/// output.
#[test]
fn three_input_uniform_addition() {
    for destination in [0_u8, 2, 4] {
        for zero_third in [false, true] {
            for carry_output in [false, true] {
                let third = if zero_third { RZ } else { reg(2) };
                let mut add = iadd3(destination, [reg(0), usrc(3, false), third]);
                if carry_output {
                    add.overflow[0] = p(0);
                }
                let function = mixed(&[Instr::new(Op::IAdd3(add))]);
                for a in [0_u32, 1, 0x8000_0000, 0xffff_ffff] {
                    for b in [0_u32, 1, 0x1234_5678, 0xffff_ffff] {
                        for c in [0_u32, 1, 0x8765_4321, 0xffff_ffff] {
                            let mut machine = Machine::new();
                            machine.regs[0] = a;
                            machine.uniform_regs[3] = b;
                            machine.regs[2] = c;
                            machine.preds = [true, false, true, false, true, false, true, false];
                            let mut expected = machine.preds;
                            let sum = u64::from(a)
                                + u64::from(b)
                                + if zero_third { 0 } else { u64::from(c) };
                            if carry_output {
                                expected[0] = sum >= 1 << 32;
                            }
                            machine.run(&function).expect("runs");
                            let low = u32::try_from(sum & u64::from(u32::MAX)).expect("low");
                            let at = format!(
                                "{destination} {zero_third} {carry_output} {a:#x} {b:#x} {c:#x}"
                            );
                            assert_eq!(machine.regs[usize::from(destination)], low, "{at}");
                            assert_eq!(machine.uniform_regs[3], b, "{at}");
                            assert_eq!(machine.preds, expected, "{at}");
                        }
                    }
                }
            }
        }
    }
}

/// "Native uniform right shifts preserve signedness bounds and aliases": `USHF.R.<S32|U32>.HI
/// URd, URZ, distance, UR4` into `UR4` itself and `UR5`, a signed shift by 32 filling with the
/// sign and an unsigned one giving zero.
#[test]
fn uniform_right_shifts_preserve_signedness() {
    for signed in [false, true] {
        for distance in [0_u32, 2, 31, 32] {
            for destination in [4_u8, 5] {
                let function = mixed(&[Instr::new(ushf_right(destination, distance, 4, signed))]);
                for value in [0_u32, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff] {
                    let mut machine = Machine::new();
                    machine.uniform_regs[4] = value;
                    machine.run(&function).expect("runs");
                    let expected = if signed {
                        (value.cast_signed() >> distance.min(31)).cast_unsigned()
                    } else {
                        value.checked_shr(distance).unwrap_or(0)
                    };
                    let at = format!("{signed} {distance} {destination} {value:#x}");
                    assert_eq!(
                        machine.uniform_regs[usize::from(destination)],
                        expected,
                        "{at}"
                    );
                }
            }
        }
    }
}

/// "Native extended uniform registers preserve independent carry state": `UIADD3 UR16, UP0,
/// UPT, UR4, UR6, URZ` and `UIADD3.X UR18, UR5, UR7, URZ, UP0, !UPT` add two 64-bit values into
/// `UR16` and `UR18`, moved to `R0` and `R1`; `UP0` is the low word's carry, the rest of the
/// bank and the per-thread predicates unchanged.
#[test]
fn extended_uniform_registers_carry_state() {
    let function = mixed(&[
        Instr::new(Op::UIAdd3(twin(OpIAdd3 {
            overflow: [p(0), Pred::PT],
            ..iadd3(16, [reg(4), reg(6), RZ])
        }))),
        Instr::new(Op::UIAdd3X(twin(OpIAdd3X {
            add: OpIAdd3 {
                carry: [ps(0, false), NOT_PT],
                ..iadd3(18, [reg(5), reg(7), RZ])
            },
        }))),
        mov_from(0, 16),
        mov_from(1, 18),
    ]);
    let values = [
        0_u64,
        1,
        0xffff_ffff,
        0x1_0000_0001,
        0x7fff_ffff_ffff_ffff,
        u64::MAX,
    ];
    let low = |value: u64| u32::try_from(value & u64::from(u32::MAX)).expect("low");
    let high = |value: u64| u32::try_from(value >> 32).expect("high");
    for a in values {
        for b in values {
            for pattern in [0_u32, 0x15, 0x7f] {
                let mut machine = Machine::new();
                machine.uniform_regs[4] = low(a);
                machine.uniform_regs[5] = high(a);
                machine.uniform_regs[6] = low(b);
                machine.uniform_regs[7] = high(b);
                set_bank(&mut machine, pattern);
                machine.preds = [true, false, true, false, false, true, false, false];
                let preds = machine.preds;
                machine.run(&function).expect("runs");
                let sum = a.wrapping_add(b);
                let at = format!("{a:#x} {b:#x} {pattern:#x}");
                assert_eq!(machine.regs[0], low(sum), "{at}");
                assert_eq!(machine.regs[1], high(sum), "{at}");
                assert_eq!(machine.uniform_regs[16], machine.regs[0], "{at}");
                assert_eq!(machine.uniform_regs[18], machine.regs[1], "{at}");
                let carry = u64::from(low(a)) + u64::from(low(b)) >= 1 << 32;
                let bank = (pattern & !0x40) | if carry { 0x40 } else { 0 };
                assert_eq!(bank_of(&machine), bank, "{at}");
                assert_eq!(machine.preds, preds, "{at}");
            }
        }
    }
}

/// "Native uniform register mapping covers each reserved bank": `UIMAD URt, UR4, 0x5, URZ`
/// and `MOV R10, URt` for each of the PR's targets.
#[test]
fn uniform_register_mapping() {
    for target in [0_u8, 14, 16, 18, 30, 32, 46, 48, 60, 62] {
        let function = mixed(&[
            Instr::new(Op::UIMad(twin(imad(target, [reg(4), AluSrc::Imm(5), RZ])))),
            mov_from(10, target),
        ]);
        let mut machine = Machine::new();
        machine.uniform_regs[4] = 0x1234_5678;
        machine.preds = [true, false, true, false, true, false, true, false];
        let preds = machine.preds;
        machine.run(&function).expect("runs");
        let product = 0x1234_5678_u32.wrapping_mul(5);
        assert_eq!(
            machine.uniform_regs[usize::from(target)],
            product,
            "{target}"
        );
        assert_eq!(machine.regs[10], product, "{target}");
        assert_eq!(machine.preds, preds, "{target}");
    }
}

/// "Native uniform constant pairs remain contiguous at bank boundaries": `ULDC.64 URt,
/// c[0x3][0x100]` writes `URt` and `URt+1` alone, for each of the PR's targets.
#[test]
fn uniform_constant_pairs() {
    for target in [14_u8, 16, 30, 32, 46, 48, 60] {
        let function = mixed(&[Instr::new(Op::ULdc(OpULdc {
            dst: ur(target),
            src: ULdcSrc::CBuf(CBufRef {
                buf: CBuf::Binding(3),
                offset: 0x100,
            }),
            mem_type: MemType::B64,
        }))]);
        let mut machine = Machine::new();
        for (index, slot) in (0_u32..63).zip(machine.uniform_regs.iter_mut()) {
            *slot = 0xc000_0000 + index;
        }
        set_bank(&mut machine, 0x15);
        machine.preds = [true, false, true, true, false, false, true, false];
        let preds = machine.preds;
        let mut bytes = vec![0; 0x108];
        bytes[0x100..0x104].copy_from_slice(&0x1234_5678_u32.to_le_bytes());
        bytes[0x104..0x108].copy_from_slice(&0x8765_4321_u32.to_le_bytes());
        machine.cbufs.insert(3, bytes);
        machine.run(&function).expect("runs");
        for index in 0..63_u8 {
            let expected = if index == target {
                0x1234_5678
            } else if index == target + 1 {
                0x8765_4321
            } else {
                0xc000_0000 + u32::from(index)
            };
            assert_eq!(
                machine.uniform_regs[usize::from(index)],
                expected,
                "{target} {index}"
            );
        }
        assert_eq!(bank_of(&machine), 0x15);
        assert_eq!(machine.preds, preds);
    }
}

/// "Native uniform subtrahends preserve modular results and execution guards": the vector
/// `IADD3 Rd, R0, -UR6, RZ` under `P0`, `!P0` and `PT`, into `R0` and `R4`.
#[test]
fn uniform_subtrahends() {
    for destination in [0_u8, 4] {
        for guard in [ps(0, false), ps(0, true), PredSrc::TRUE] {
            let function = mixed(&[under(
                guard,
                Op::IAdd3(iadd3(destination, [reg(0), usrc(6, true), RZ])),
            )]);
            for a in [0_u32, 1, 0x8000_0000, 0xffff_ffff] {
                for b in [0_u32, 1, 0x8000_0000, 0xffff_ffff] {
                    for enabled in [false, true] {
                        let mut machine = Machine::new();
                        machine.regs[0] = a;
                        machine.uniform_regs[6] = b;
                        machine.preds = [enabled, true, false, true, false, true, false, false];
                        let preds = machine.preds;
                        let before = machine.regs[usize::from(destination)];
                        let active = guard == PredSrc::TRUE || enabled != guard.negated;
                        machine.run(&function).expect("runs");
                        let expected = if active { a.wrapping_sub(b) } else { before };
                        let at = format!("{destination} {guard:?} {a:#x} {b:#x} {enabled}");
                        assert_eq!(machine.regs[usize::from(destination)], expected, "{at}");
                        assert_eq!(machine.uniform_regs[6], b, "{at}");
                        assert_eq!(machine.preds, preds, "{at}");
                    }
                }
            }
        }
    }
}

/// "Native uniform bitwise logic preserves separate predicate banks and discarded
/// destinations": `ULOP3.LUT UPn, URd, UR4, 0x1, URZ, 0xc0, !UPT` (form 4, `0x892`) sets `UPn`
/// to `UR4 & 1` for every `n` below 7, into `UR5` and into `URZ` (discarded), leaving the
/// per-thread predicates and `UR4`.
#[test]
fn uniform_bitwise_logic() {
    for output in 0..7_u8 {
        for destination in [5_u8, 63] {
            let dst = if destination == 63 {
                Gpr::RZ
            } else {
                r(destination)
            };
            let function = mixed(&[Instr::new(Op::ULop3(twin(OpLop3 {
                pred: p(output),
                dst,
                srcs: [reg(4), AluSrc::Imm(1), RZ],
                lut: 0xc0,
            })))]);
            for bank in [0_u32, 0x2a, 0x55, 0x7f] {
                for value in [0_u32, 1, 2, 3, 0xffff_ffff] {
                    let mut machine = Machine::new();
                    machine.uniform_regs[4] = value;
                    machine.uniform_regs[5] = 0xcafe;
                    set_bank(&mut machine, bank);
                    alternating(&mut machine);
                    let ordinary = machine.preds;
                    machine.run(&function).expect("runs");
                    let bit = 1 << (6 - output);
                    let bits = (bank & !bit) | if value & 1 != 0 { bit } else { 0 };
                    let at = format!("{output} {destination} {bank:#x} {value:#x}");
                    assert_eq!(bank_of(&machine), bits, "{at}");
                    assert_eq!(machine.preds, ordinary, "{at}");
                    assert_eq!(machine.uniform_regs[4], value, "{at}");
                    let five = if destination == 63 { 0xcafe } else { value & 1 };
                    assert_eq!(machine.uniform_regs[5], five, "{at}");
                }
            }
        }
    }
}

/// "Native uniform loads retain the full five-bit constant buffer bank": `ULDC.64 UR6,
/// c[bank][0x58]` for the PR's banks, leaving `UR8`.
#[test]
fn uniform_loads_retain_the_five_bit_bank() {
    for bank in [0_u8, 17, 24, 31] {
        let function = mixed(&[Instr::new(Op::ULdc(OpULdc {
            dst: ur(6),
            src: ULdcSrc::CBuf(CBufRef {
                buf: CBuf::Binding(bank),
                offset: 0x58,
            }),
            mem_type: MemType::B64,
        }))]);
        let mut machine = Machine::new();
        let mut bytes = vec![0; 0x60];
        bytes[0x58..0x5c].copy_from_slice(&0xffff_fff0_u32.to_le_bytes());
        bytes[0x5c..0x60].copy_from_slice(&0x29_u32.to_le_bytes());
        machine.cbufs.insert(u32::from(bank), bytes);
        machine.uniform_regs[8] = 0xcafe_beef;
        machine.run(&function).expect("runs");
        assert_eq!(machine.uniform_regs[6], 0xffff_fff0, "{bank}");
        assert_eq!(machine.uniform_regs[7], 0x29, "{bank}");
        assert_eq!(machine.uniform_regs[8], 0xcafe_beef, "{bank}");
    }
}

/// "Merged high uniform bit reversal preserves the actual guest register alias": `UBREV UR17,
/// UR4` and `UBREV UR4, UR17` reverse every bit, leaving the predicates.
#[test]
fn merged_high_uniform_bit_reversal() {
    for (source, destination) in [(4_u8, 17_u8), (17, 4)] {
        let function = mixed(&[Instr::new(Op::UBRev(twin(OpBRev {
            dst: r(destination),
            src: reg(source),
        })))]);
        for value in [0_u32, 1, 0x0123_4567, 0x8000_0000, 0xffff_ffff] {
            let mut machine = Machine::new();
            machine.uniform_regs[usize::from(source)] = value;
            machine.preds = [true, false, true, false, true, false, true, false];
            let preds = machine.preds;
            machine.run(&function).expect("runs");
            assert_eq!(
                machine.uniform_regs[usize::from(destination)],
                value.reverse_bits(),
                "{source} {value:#x}"
            );
            assert_eq!(machine.preds, preds);
        }
    }
}

/// "Native integer multiply preserves negated uniform operands and wrapping addends": the
/// vector `IMAD Rd, R0, -UR4, R1` and `IMAD Rd, R0, R1, -UR4` into `R0`, `R1` and `R5`, every
/// other register unchanged.
#[test]
fn integer_multiply_negated_uniform_operands() {
    let values = [
        0_u32,
        1,
        2,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_fffe,
        0xffff_ffff,
    ];
    for uniform_addend in [false, true] {
        for destination in [0_u8, 1, 5] {
            let srcs = if uniform_addend {
                [reg(0), reg(1), usrc(4, true)]
            } else {
                [reg(0), usrc(4, true), reg(1)]
            };
            let function = mixed(&[Instr::new(Op::IMad(imad(destination, srcs)))]);
            for uniform in values {
                for left in values {
                    for right in values {
                        let mut machine = Machine::new();
                        machine.regs = [0x1234_5678; crate::interp::REG_SLOTS];
                        machine.regs[0] = left;
                        machine.regs[1] = right;
                        machine.uniform_regs[4] = uniform;
                        machine.preds = [true, false, true, false, true, false, true, false];
                        let preds = machine.preds;
                        let mut expected = machine.regs;
                        expected[usize::from(destination)] = if uniform_addend {
                            left.wrapping_mul(right).wrapping_sub(uniform)
                        } else {
                            left.wrapping_mul(uniform.wrapping_neg())
                                .wrapping_add(right)
                        };
                        machine.run(&function).expect("runs");
                        assert_eq!(machine.regs, expected, "{uniform_addend} {destination}");
                        assert_eq!(machine.preds, preds);
                    }
                }
            }
        }
    }
}
