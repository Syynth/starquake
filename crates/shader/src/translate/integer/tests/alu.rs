//! Each instruction and modifier on the interpreter. Expected values come from the documented
//! behaviour, written out here as plain Rust (NAK's folds and PR #4588's comments, as each
//! test says), not from running the translator.

use super::*;
use crate::decode::{
    IMadMode, IntCmpOp, MovSrc, OpBRev, OpFlo, OpIAbs, OpIAdd3X, OpIDp4, OpIMad, OpIMad64, OpIMnMx,
    OpISetP, OpLea, OpLeaX, OpLop3, OpP2R, OpPLop3, OpPopC, OpPrmt, OpR2P, OpSel, OpSgxt, OpShf,
    PLop3Src, PredSetOp, PrmtMode, ShfType, UniformPredSrc,
};

fn mov(dst: u8, src: MovSrc) -> Op {
    Op::Mov(OpMov {
        dst: r(dst),
        src,
        quad_lanes: OpMov::ALL_LANES,
    })
}

/// `MOV` from each source form; a write to `RZ` is dropped and `RZ` reads zero.
#[test]
fn mov_reads_every_source_form() {
    let mut machine = machine_with(&[0, 0, 0x1111_2222]);
    with_cbuf(&mut machine, 3, &[0xaaaa_aaaa, 0x0403_0201, 0x0807_0605]);
    machine.uniform_regs[5] = 0x5555_0005;
    machine.regs[6] = 6;
    machine.regs[7] = 7;
    let program = [
        Instr::new(mov(1, MovSrc::Reg(r(2)))),
        Instr::new(mov(3, MovSrc::Imm(0xdead_beef))),
        Instr::new(mov(
            4,
            MovSrc::CBuf(CBufRef {
                buf: CBuf::Binding(3),
                offset: 4,
            }),
        )),
        Instr::new(mov(5, MovSrc::UniformReg(ur(5)))),
        Instr::new(mov(6, MovSrc::Reg(Gpr::RZ))),
        Instr::new(mov(7, MovSrc::UniformReg(UniformReg::URZ))),
        Instr::new(mov(255, MovSrc::Imm(9))),
    ];
    run(&program, &mut machine);
    assert_eq!(
        machine.regs[1..8],
        [
            0x1111_2222,
            0x1111_2222,
            0xdead_beef,
            0x0403_0201,
            0x5555_0005,
            0,
            0
        ]
    );
    assert_eq!(machine.regs[255], 0);
}

/// An unaligned constant-buffer offset reads the 32 bits starting there, from two words
/// (`Visitor::Cbuf`), at each of the three skews; an aligned one reads one word.
#[test]
fn unaligned_constant_buffer_reads_span_two_words() {
    for (offset, expected) in [
        (4, 0x0403_0201),
        (5, 0x0504_0302),
        (6, 0x0605_0403),
        (7, 0x0706_0504),
        (8, 0x0807_0605),
    ] {
        let mut machine = Machine::new();
        with_cbuf(
            &mut machine,
            2,
            &[0xaaaa_aaaa, 0x0403_0201, 0x0807_0605, 0x0c0b_0a09],
        );
        run_op(
            mov(
                1,
                MovSrc::CBuf(CBufRef {
                    buf: CBuf::Binding(2),
                    offset,
                }),
            ),
            &mut machine,
        );
        assert_eq!(machine.regs[1], expected, "offset {offset}");
    }
}

/// `IADD3` without carry outputs: a wrapping 32-bit sum, each source possibly negated.
#[test]
fn iadd3_sums_with_negations() {
    for mask in 0..8_u8 {
        for (a, b, c) in [
            (1, 2, 3),
            (0xffff_ffff, 1, 0),
            (0x8000_0000, 0x8000_0000, 5),
        ] {
            let srcs: [AluSrc; 3] = std::array::from_fn(|i| {
                let index = u8::try_from(i).unwrap_or(0) + 1;
                if mask & (1 << i) != 0 {
                    neg(index)
                } else {
                    reg(index)
                }
            });
            let term = |value: u32, i: u8| {
                if mask & (1 << i) != 0 {
                    value.wrapping_neg()
                } else {
                    value
                }
            };
            let expected = term(a, 0).wrapping_add(term(b, 1)).wrapping_add(term(c, 2));
            let mut machine = machine_with(&[0, a, b, c]);
            run_op(Op::IAdd3(iadd3(0, srcs)), &mut machine);
            assert_eq!(
                machine.regs[0], expected,
                "mask {mask} {a:#x} {b:#x} {c:#x}"
            );
        }
    }
}

/// A two-instruction carry chain (`IADD3` with a carry-out predicate, then `IADD3.X` reading
/// it) gives the 64-bit sum, at and around the 32-bit carry boundary.
#[test]
fn a_two_instruction_carry_chain_adds_64_bits() {
    let low = Op::IAdd3(OpIAdd3 {
        overflow: [p(0), Pred::PT],
        ..iadd3(0, [reg(2), reg(4), AluSrc::reg(Gpr::RZ)])
    });
    let high = Op::IAdd3X(OpIAdd3X {
        add: OpIAdd3 {
            carry: [ps(0, false), NOT_PT],
            ..iadd3(1, [reg(3), reg(5), AluSrc::reg(Gpr::RZ)])
        },
    });
    for (x, y) in [
        (0x0000_0000_ffff_ffff_u64, 1_u64),
        (0x0000_0000_ffff_fffe, 1),
        (0x0000_0001_ffff_ffff, 0x0000_0001_0000_0001),
        (u64::MAX, 1),
        (0x1234_5678_9abc_def0, 0x0fed_cba9_8765_4321),
    ] {
        let words = |v: u64| [(v & 0xffff_ffff) as u32, (v >> 32) as u32];
        let [xl, xh] = words(x);
        let [yl, yh] = words(y);
        let mut machine = machine_with(&[0, 0, xl, xh, yl, yh]);
        run(&[Instr::new(low), Instr::new(high)], &mut machine);
        let sum = x.wrapping_add(y);
        assert_eq!(machine.regs[0..2], words(sum), "{x:#x} + {y:#x}");
    }
}

/// `IADD3.X` in each operand form reads its carries: bit 74 is `.X` in every form (NAK), where
/// PR #4588 reads source 2's absolute-value bit.
#[test]
fn extended_adds_in_every_form() {
    let mut machine = machine_with(&[0, 10, 20, 30]);
    machine.preds[0] = true;
    machine.uniform_regs[2] = 20;
    with_cbuf(&mut machine, 1, &[20]);
    let forms = [
        [reg(1), reg(2), reg(3)],
        [reg(1), reg(3), AluSrc::Imm(20)],
        [reg(1), reg(3), cb(1, 0)],
        [reg(1), AluSrc::Imm(20), reg(3)],
        [reg(1), cb(1, 0), reg(3)],
        [reg(1), ureg(2), reg(3)],
        [reg(1), reg(3), ureg(2)],
    ];
    for srcs in forms {
        let op = Op::IAdd3X(OpIAdd3X {
            add: OpIAdd3 {
                carry: [ps(0, false), ps(1, true)],
                ..iadd3(4, srcs)
            },
        });
        let mut machine = machine.clone();
        run_op(op, &mut machine);
        // 10 + 20 + 30, plus P0 (true) and !P1 (true).
        assert_eq!(machine.regs[4], 62, "{srcs:?}");
    }
}

fn lea(dst: u8, a: AluSrc, b: AluSrc, shift: u8) -> OpLea {
    OpLea {
        dst: r(dst),
        overflow: Pred::PT,
        a,
        b,
        a_high: AluSrc::reg(Gpr::RZ),
        shift,
        dst_high: false,
        sx32: false,
        intermediate: false,
        carry: NOT_PT,
    }
}

/// `LEA`: `(a << shift) + b`, with `b` negated, with a carry output, with `.X`.
#[test]
fn lea_shifts_and_adds() {
    for shift in [0, 1, 4, 31] {
        for (a, b) in [(3_u32, 5_u32), (0xffff_ffff, 1), (0x8000_0001, 0x7fff_ffff)] {
            let shifted = a.wrapping_shl(u32::from(shift));
            let mut machine = machine_with(&[0, a, b]);
            run_op(Op::Lea(lea(0, reg(1), reg(2), shift)), &mut machine);
            assert_eq!(machine.regs[0], shifted.wrapping_add(b));

            let mut machine = machine_with(&[0, a, b]);
            run_op(Op::Lea(lea(0, reg(1), neg(2), shift)), &mut machine);
            assert_eq!(machine.regs[0], shifted.wrapping_sub(b));

            // The carry output: whether the 32-bit sum of the truncated shift and `b` carries.
            let mut machine = machine_with(&[0, a, b]);
            run_op(
                Op::Lea(OpLea {
                    overflow: p(3),
                    ..lea(0, reg(1), reg(2), shift)
                }),
                &mut machine,
            );
            let wide = u64::from(shifted) + u64::from(b);
            assert_eq!(machine.regs[0], wide as u32);
            assert_eq!(machine.preds[3], wide >> 32 != 0, "{a:#x} {b:#x} {shift}");

            // `.X`: the carry predicate adds one, and a negated `b` is a bitwise not.
            for carry in [false, true] {
                let mut machine = machine_with(&[0, a, b]);
                machine.preds[2] = carry;
                let op = OpLea {
                    overflow: p(4),
                    carry: ps(2, false),
                    ..lea(0, reg(1), neg(2), shift)
                };
                run_op(Op::LeaX(OpLeaX { lea: op }), &mut machine);
                let wide = u64::from(shifted) + u64::from(!b) + u64::from(carry);
                assert_eq!(machine.regs[0], wide as u32);
                assert_eq!(machine.preds[4], wide >> 32 != 0);
            }
        }
    }
}

/// `LEA.HI`: the high word of `(high:a) << shift` plus `b`; `.SX32` takes `a`'s sign as the
/// high word; `.X` adds the carry.
#[test]
fn lea_hi_takes_the_high_word() {
    for shift in [0_u8, 1, 4, 31] {
        for (a, high, b) in [
            (0x8000_0001_u32, 0x0000_0003_u32, 5_u32),
            (0xffff_ffff, 0, 1),
        ] {
            let wide = (u64::from(high) << 32) | u64::from(a);
            let top = ((wide << shift) >> 32) as u32;
            let mut machine = machine_with(&[0, a, b, high]);
            let op = OpLea {
                dst_high: true,
                a_high: reg(3),
                ..lea(0, reg(1), reg(2), shift)
            };
            run_op(Op::Lea(op), &mut machine);
            assert_eq!(
                machine.regs[0],
                top.wrapping_add(b),
                "{a:#x} {high:#x} {shift}"
            );

            let signed = i64::from(a as i32) as u64;
            let top = ((signed << shift) >> 32) as u32;
            let mut machine = machine_with(&[0, a, b]);
            let op = OpLea {
                dst_high: true,
                sx32: true,
                ..lea(0, reg(1), neg(2), shift)
            };
            run_op(Op::Lea(op), &mut machine);
            assert_eq!(machine.regs[0], top.wrapping_sub(b));

            let mut machine = machine_with(&[0, a, b, high]);
            machine.preds[1] = true;
            let op = OpLea {
                dst_high: true,
                a_high: reg(3),
                carry: ps(1, false),
                ..lea(0, reg(1), reg(2), shift)
            };
            run_op(Op::LeaX(OpLeaX { lea: op }), &mut machine);
            let top = ((wide << shift) >> 32) as u32;
            assert_eq!(machine.regs[0], top.wrapping_add(b).wrapping_add(1));
        }
    }
}

/// In forms 2, 3 and 7, `LEA.HI` takes its high word from source 2, the wide slot (NAK's
/// operand order), and `b` from the register slot; PR #4588 reads both from bits 64..72.
#[test]
fn lea_hi_reads_its_high_word_from_source_2() {
    let mut machine = machine_with(&[0, 0x8000_0000, 7]);
    machine.uniform_regs[4] = 2;
    with_cbuf(&mut machine, 0, &[2]);
    for high in [AluSrc::Imm(2), cb(0, 0), ureg(4)] {
        let mut machine = machine.clone();
        let op = OpLea {
            dst_high: true,
            a_high: high,
            ..lea(0, reg(1), reg(2), 1)
        };
        run_op(Op::Lea(op), &mut machine);
        // (2:0x80000000) << 1 has high word 5; plus b = 7.
        assert_eq!(machine.regs[0], 12, "{high:?}");
    }
}

/// `LOP3` gives every truth table's function of three values (bit `a << 2 | b << 1 | c` of
/// each table, Eden's `ApplyLUT`), and its predicate output says whether it is not zero.
#[test]
fn lop3_follows_every_truth_table() {
    let [a, b, c] = [0xf0f0_ff00_u32, 0xcccc_f0f0, 0xaaaa_cccc];
    for lut in 0..=u8::MAX {
        let expected = (0..32).fold(0_u32, |acc, bit| {
            let row = ((a >> bit) & 1) << 2 | ((b >> bit) & 1) << 1 | ((c >> bit) & 1);
            acc | (u32::from(lut) >> row & 1) << bit
        });
        let mut machine = machine_with(&[0, a, b, c]);
        machine.preds[2] = expected == 0;
        let op = OpLop3 {
            pred: p(2),
            dst: r(0),
            srcs: [reg(1), reg(2), reg(3)],
            lut,
        };
        run_op(Op::Lop3(op), &mut machine);
        assert_eq!(machine.regs[0], expected, "lut {lut:#x}");
        assert_eq!(machine.preds[2], expected != 0, "lut {lut:#x}");
    }
}

/// `IABS`, `BREV` and `POPC` (with and without its bitwise not).
#[test]
fn one_source_instructions() {
    for value in VALUES {
        let mut machine = machine_with(&[0, value]);
        run(
            &[
                Instr::new(Op::IAbs(OpIAbs {
                    dst: r(2),
                    src: reg(1),
                })),
                Instr::new(Op::BRev(OpBRev {
                    dst: r(3),
                    src: reg(1),
                })),
                Instr::new(Op::PopC(OpPopC {
                    dst: r(4),
                    src: reg(1),
                })),
                Instr::new(Op::PopC(OpPopC {
                    dst: r(5),
                    src: neg(1),
                })),
            ],
            &mut machine,
        );
        assert_eq!(machine.regs[2], (value as i32).wrapping_abs() as u32);
        assert_eq!(machine.regs[3], value.reverse_bits());
        assert_eq!(machine.regs[4], value.count_ones());
        assert_eq!(machine.regs[5], (!value).count_ones());
    }
}

/// `BREV` takes every source form NAK encodes for it (PR #4588 translates form 1 only).
#[test]
fn brev_takes_every_source_form() {
    let mut machine = Machine::new();
    machine.uniform_regs[1] = 1;
    with_cbuf(&mut machine, 0, &[1]);
    for src in [AluSrc::Imm(1), cb(0, 0), ureg(1)] {
        let mut machine = machine.clone();
        run_op(Op::BRev(OpBRev { dst: r(0), src }), &mut machine);
        assert_eq!(machine.regs[0], 0x8000_0000, "{src:?}");
    }
}

/// `FLO`: the highest set bit (unsigned) or the highest bit unlike the sign (signed), -1 for
/// none; `.SH` gives `31 - that` (32 for none); the bitwise not applies first.
#[test]
fn flo_finds_the_leading_bit() {
    let msb = |value: u32, signed: bool| -> u32 {
        let bits = if signed && (value as i32) < 0 {
            !value
        } else {
            value
        };
        if bits == 0 {
            u32::MAX
        } else {
            31 - bits.leading_zeros()
        }
    };
    for value in VALUES {
        for signed in [false, true] {
            for shift in [false, true] {
                for not in [false, true] {
                    let src = if not { neg(1) } else { reg(1) };
                    let mut machine = machine_with(&[0, value]);
                    let op = OpFlo {
                        dst: r(0),
                        src,
                        signed,
                        return_shift_amount: shift,
                    };
                    run_op(Op::Flo(op), &mut machine);
                    let found = msb(if not { !value } else { value }, signed);
                    let expected = if shift {
                        31_u32.wrapping_sub(found)
                    } else {
                        found
                    };
                    assert_eq!(
                        machine.regs[0], expected,
                        "{value:#x} {signed} {shift} {not}"
                    );
                }
            }
        }
    }
}

/// `IMNMX` and `SEL` pick by their predicate.
#[test]
fn imnmx_and_sel_pick_by_predicate() {
    for (a, b) in [(1_u32, 2_u32), (0xffff_ffff, 1), (0x8000_0000, 0x7fff_ffff)] {
        for signed in [false, true] {
            for (pred, negated) in [(0, false), (0, true), (7, false), (7, true)] {
                let mut machine = machine_with(&[0, a, b]);
                machine.preds[0] = true;
                let op = OpIMnMx {
                    dst: r(0),
                    srcs: [reg(1), reg(2)],
                    signed,
                    min: ps(pred, negated),
                };
                run_op(Op::IMnMx(op), &mut machine);
                let take_min = !negated;
                let (lo, hi) = if signed {
                    let (x, y) = (a as i32, b as i32);
                    (x.min(y) as u32, x.max(y) as u32)
                } else {
                    (a.min(b), a.max(b))
                };
                assert_eq!(machine.regs[0], if take_min { lo } else { hi });
            }
        }
        for cond in [false, true] {
            let mut machine = machine_with(&[0, a, b]);
            machine.preds[5] = cond;
            run_op(
                Op::Sel(OpSel {
                    dst: r(0),
                    srcs: [reg(1), reg(2)],
                    cond: ps(5, false),
                }),
                &mut machine,
            );
            assert_eq!(machine.regs[0], if cond { a } else { b });
        }
    }
}

/// `SGXT`: the low `bits` bits, sign- or zero-extended; 0 bits give 0 and 32 or more keep the
/// value (NAK's fold, as PR #4588's comment states).
#[test]
fn sgxt_extends_the_low_bits() {
    for value in [0x0000_0080_u32, 0xffff_ff7f, 0x8000_0000, 0x1234_5678] {
        for bits in [0_u32, 1, 7, 8, 31, 32, 33, 0xffff_ffff] {
            for signed in [false, true] {
                let expected = if bits == 0 {
                    0
                } else if bits >= 32 {
                    value
                } else {
                    let shift = 32 - bits;
                    if signed {
                        (((value << shift) as i32) >> shift) as u32
                    } else {
                        (value << shift) >> shift
                    }
                };
                let mut machine = machine_with(&[0, value, bits]);
                let op = OpSgxt {
                    dst: r(0),
                    a: reg(1),
                    bits: reg(2),
                    signed,
                    wrap: false,
                };
                run_op(Op::Sgxt(op), &mut machine);
                assert_eq!(machine.regs[0], expected, "{value:#x} {bits} {signed}");
            }
        }
    }
}

/// `PRMT`: each selector nibble picks a byte of `b:a`, or with bit 3 repeats its sign.
#[test]
fn prmt_picks_bytes() {
    let (a, b) = (0x8877_6655_u32, 0x4433_2211_u32);
    let bytes = [a.to_le_bytes(), b.to_le_bytes()].concat();
    for selector in [
        0x3210_u32,
        0x7654,
        0x0123,
        0x4444,
        0x8888,
        0xbfa9,
        0xffff_7531,
    ] {
        let expected = (0..4).fold(0_u32, |acc, byte| {
            let nibble = (selector >> (4 * byte)) & 0xf;
            let picked = bytes[(nibble & 7) as usize];
            let value = if nibble & 8 != 0 {
                if picked & 0x80 != 0 { 0xff } else { 0 }
            } else {
                u32::from(picked)
            };
            acc | value << (8 * byte)
        });
        let mut machine = machine_with(&[0, a, b]);
        let op = OpPrmt {
            dst: r(0),
            srcs: [reg(1), reg(2)],
            sel: AluSrc::Imm(selector),
            mode: PrmtMode::Index,
        };
        run_op(Op::Prmt(op), &mut machine);
        assert_eq!(machine.regs[0], expected, "selector {selector:#x}");
    }
}

/// NAK's `OpShf` fold for SM 7.0 and later (`ir.rs`, `impl Foldable for OpShf`).
fn shf_model(
    low: u32,
    high: u32,
    shift: u32,
    ty: ShfType,
    wrap: bool,
    right: bool,
    hi: bool,
) -> u32 {
    let bits = match ty {
        ShfType::I64 | ShfType::U64 => 64,
        ShfType::I32 | ShfType::U32 => 32,
    };
    let shift = if wrap {
        shift & (bits - 1)
    } else {
        shift.min(bits)
    };
    let x = u64::from(low) | (u64::from(high) << 32);
    let signed = matches!(ty, ShfType::I64 | ShfType::I32);
    let shifted = if right && signed {
        let x = x as i64;
        x.checked_shr(shift).unwrap_or(x >> 63) as u64
    } else if right {
        x.checked_shr(shift).unwrap_or(0)
    } else {
        x.checked_shl(shift).unwrap_or(0)
    };
    if hi {
        (shifted >> 32) as u32
    } else {
        shifted as u32
    }
}

/// `SHF` follows NAK's type encoding and shift amounts (the decoder reads the type field as NAK
/// encodes it; PR #4588 reads it differently and clamps every type to 32).
#[test]
fn shf_follows_naks_types_and_amounts() {
    let (low, high) = (0x8765_4321_u32, 0x8123_4567_u32);
    for ty in ShfType::ALL {
        for shift in [0_u32, 1, 31, 32, 33, 40, 63, 64, 65, 0xffff_ffff] {
            for (wrap, right, hi) in
                (0..8).map(|bits| (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0))
            {
                let mut machine = machine_with(&[0, low, shift, high]);
                let op = OpShf {
                    dst: r(0),
                    low: reg(1),
                    shift: reg(2),
                    high: reg(3),
                    data_type: ty,
                    wrap,
                    right,
                    dst_high: hi,
                };
                run_op(Op::Shf(op), &mut machine);
                assert_eq!(
                    machine.regs[0],
                    shf_model(low, high, shift, ty, wrap, right, hi),
                    "{ty:?} {shift} wrap {wrap} right {right} hi {hi}"
                );
            }
        }
    }
}

fn imad(dst: u8, srcs: [AluSrc; 3], signed: bool, mode: IMadMode) -> Op {
    Op::IMad(OpIMad {
        dst: r(dst),
        srcs,
        signed,
        mode,
    })
}

const LO: IMadMode = IMadMode::Lo {
    x: false,
    carry: NOT_PT,
};

/// `IMAD`: the low word of `a * b + c`, with negations; `.X` adds the carry and makes a negated
/// `c` a bitwise not; `IMAD.HI`: the high word of the product (signed or not) plus `c`.
#[test]
fn imad_multiplies_and_adds() {
    for a in VALUES {
        for (b, c) in [
            (3_u32, 5_u32),
            (0xffff_ffff, 0x8000_0000),
            (0x1234_5678, 0xffff_ffff),
        ] {
            let mut machine = machine_with(&[0, a, b, c]);
            run(
                &[
                    Instr::new(imad(4, [reg(1), reg(2), reg(3)], false, LO)),
                    Instr::new(imad(5, [neg(1), neg(2), neg(3)], true, LO)),
                    Instr::new(imad(
                        6,
                        [reg(1), reg(2), neg(3)],
                        false,
                        IMadMode::Lo {
                            x: true,
                            carry: ps(0, true),
                        },
                    )),
                    Instr::new(imad(7, [reg(1), reg(2), reg(3)], false, IMadMode::Hi)),
                    Instr::new(imad(8, [reg(1), reg(2), reg(3)], true, IMadMode::Hi)),
                ],
                &mut machine,
            );
            assert_eq!(machine.regs[4], a.wrapping_mul(b).wrapping_add(c));
            assert_eq!(
                machine.regs[5],
                a.wrapping_neg()
                    .wrapping_mul(b.wrapping_neg())
                    .wrapping_sub(c)
            );
            // P0 is clear, so `!P0` adds one.
            assert_eq!(
                machine.regs[6],
                a.wrapping_mul(b).wrapping_add(!c).wrapping_add(1)
            );
            let unsigned = (u64::from(a) * u64::from(b)) >> 32;
            assert_eq!(machine.regs[7], (unsigned as u32).wrapping_add(c));
            let signed = (i64::from(a as i32) * i64::from(b as i32)) >> 32;
            assert_eq!(
                machine.regs[8],
                (signed as u32).wrapping_add(c),
                "{a:#x} {b:#x}"
            );
        }
    }
}

fn wide(dst: u8, srcs: [AluSrc; 3], signed: bool) -> Op {
    Op::IMad64(OpIMad64 {
        dst: r(dst),
        srcs,
        signed,
        carry: NOT_PT,
    })
}

/// `IMAD.WIDE`: the 64-bit product (signed or not) plus a 64-bit addend from a register pair,
/// `RZ`, a constant buffer (two words, at an unaligned offset too) or a uniform pair.
#[test]
fn imad_wide_writes_a_pair() {
    for a in VALUES {
        for b in [3_u32, 0xffff_ffff, 0x8000_0000] {
            for signed in [false, true] {
                let product = if signed {
                    (i64::from(a as i32) * i64::from(b as i32)) as u64
                } else {
                    u64::from(a) * u64::from(b)
                };
                let addend = 0xffff_ffff_8000_0001_u64;
                let split = |v: u64| [v as u32, (v >> 32) as u32];
                let mut machine =
                    machine_with(&[0, a, b, 0, 0, 0, split(addend)[0], split(addend)[1]]);
                machine.uniform_regs[2] = split(addend)[0];
                machine.uniform_regs[3] = split(addend)[1];
                let mut bytes = vec![0_u8; 3];
                bytes.extend(addend.to_le_bytes());
                bytes.extend([0; 5]);
                machine.cbufs.insert(1, bytes);
                let cases = [
                    (reg(6), addend),
                    (AluSrc::reg(Gpr::RZ), 0),
                    (cb(1, 3), addend),
                    (ureg(2), addend),
                ];
                for (c, add) in cases {
                    let mut machine = machine.clone();
                    run_op(wide(8, [reg(1), reg(2), c], signed), &mut machine);
                    assert_eq!(
                        machine.regs[8..10],
                        split(product.wrapping_add(add)),
                        "{a:#x} {b:#x} {signed} {c:?}"
                    );
                }
            }
        }
    }
}

/// `IDP.2A.LO.U16.U8`: `c` plus the 16-bit halves of `a` times the low two bytes of `b`
/// (`Visitor::IDP`).
#[test]
fn idp_2a_dots_halves_with_bytes() {
    let (a, b, c) = (0x1234_ff01_u32, 0xaabb_80ff_u32, 0x0000_0010_u32);
    let mut machine = machine_with(&[0, a, b, c]);
    let op = OpIDp4 {
        dst: r(0),
        srcs: [reg(1), reg(2), reg(3)],
        signed: [false, false],
        two_a: true,
    };
    run_op(Op::IDp4(op), &mut machine);
    let expected = (a & 0xffff) * (b & 0xff) + (a >> 16) * ((b >> 8) & 0xff) + c;
    assert_eq!(machine.regs[0], expected);
}

/// Eden's `IntegerCompare`.
fn compare(cmp: IntCmpOp, signed: bool, a: u32, b: u32) -> bool {
    let ord = if signed {
        (a as i32).cmp(&(b as i32))
    } else {
        a.cmp(&b)
    };
    match cmp {
        IntCmpOp::False => false,
        IntCmpOp::Lt => ord.is_lt(),
        IntCmpOp::Eq => ord.is_eq(),
        IntCmpOp::Le => ord.is_le(),
        IntCmpOp::Gt => ord.is_gt(),
        IntCmpOp::Ne => ord.is_ne(),
        IntCmpOp::Ge => ord.is_ge(),
        IntCmpOp::True => true,
    }
}

/// `ISETP` with every comparison, signedness and combining operation: the first destination
/// takes the comparison combined with the accumulator, the second the inverse; the accumulator
/// is read before either is written, even when the first destination is the accumulator.
#[test]
fn isetp_compares_and_combines() {
    let pairs = [(1_u32, 2_u32), (2, 2), (0xffff_ffff, 1), (1, 0x8000_0000)];
    for cmp in IntCmpOp::ALL {
        for signed in [false, true] {
            for set_op in PredSetOp::ALL {
                for (a, b) in pairs {
                    for accum in [false, true] {
                        let mut machine = machine_with(&[0, a, b]);
                        machine.preds[1] = accum;
                        let op = OpISetP {
                            dsts: [p(1), p(2)],
                            srcs: [reg(1), reg(2)],
                            cmp_op: cmp,
                            signed,
                            set_op,
                            ex: false,
                            accum: ps(1, false),
                            low_cmp: PredSrc::TRUE,
                        };
                        run_op(Op::ISetP(op), &mut machine);
                        let combine = |x: bool| match set_op {
                            PredSetOp::And => x && accum,
                            PredSetOp::Or => x || accum,
                            PredSetOp::Xor => x ^ accum,
                        };
                        let result = compare(cmp, signed, a, b);
                        assert_eq!(
                            machine.preds[1..3],
                            [combine(result), combine(!result)],
                            "{cmp:?} {signed} {set_op:?} {a:#x} {b:#x} {accum}"
                        );
                    }
                }
            }
        }
    }
}

/// `PLOP3` gives each destination its own table's function of the three inputs, read once
/// before either destination is written, for every table and input combination; the third
/// input can be a uniform predicate.
#[test]
fn plop3_follows_every_truth_table() {
    for lut in 0..=u8::MAX {
        let other = lut.rotate_left(3) ^ 0x5a;
        for inputs in 0..8_u8 {
            let [a, b, c] = [inputs & 4 != 0, inputs & 2 != 0, inputs & 1 != 0];
            for uniform in [false, true] {
                let mut machine = Machine::new();
                machine.preds[0] = a;
                machine.preds[1] = b;
                machine.preds[2] = c;
                machine.uniform_preds[3] = c;
                let c_src = if uniform {
                    PLop3Src::Uniform(UniformPredSrc {
                        reg: upred(3),
                        negated: false,
                    })
                } else {
                    PLop3Src::Pred(ps(2, false))
                };
                // P0, an input, is also the first destination.
                let op = OpPLop3 {
                    dsts: [p(0), p(4)],
                    srcs: [ps(0, false), ps(1, false)],
                    c: c_src,
                    luts: [lut, other],
                };
                run_op(Op::PLop3(op), &mut machine);
                let row = (u8::from(a) << 2) | (u8::from(b) << 1) | u8::from(c);
                assert_eq!(
                    machine.preds[0],
                    lut >> row & 1 != 0,
                    "lut {lut:#x} inputs {inputs}"
                );
                assert_eq!(
                    machine.preds[4],
                    other >> row & 1 != 0,
                    "lut {other:#x} inputs {inputs}"
                );
            }
        }
    }
}

/// `PLOP3`'s inputs are negated where their sources say so.
#[test]
fn plop3_negates_its_inputs() {
    let mut machine = Machine::new();
    machine.preds[0] = true;
    machine.uniform_preds[1] = true;
    // A table that is `a & b & c` (row 7 only).
    let op = OpPLop3 {
        dsts: [p(3), Pred::PT],
        srcs: [ps(0, false), ps(1, true)],
        c: PLop3Src::Uniform(UniformPredSrc {
            reg: upred(2),
            negated: true,
        }),
        luts: [0x80, 0xff],
    };
    run_op(Op::PLop3(op), &mut machine);
    assert!(machine.preds[3]);
}

/// `P2R` and `R2P` move only the predicates their masks select.
#[test]
fn p2r_and_r2p_follow_their_masks() {
    for mask in [0_u8, 1, 0x40, 0x55, 0x7f] {
        for preds in [0_u8, 0x2a, 0x7f] {
            let mut machine = machine_with(&[0, 0xa5a5_a5a5]);
            for (index, slot) in machine.preds.iter_mut().take(7).enumerate() {
                *slot = preds & (1 << index) != 0;
            }
            run_op(
                Op::P2R(OpP2R {
                    dst: r(0),
                    src: r(1),
                    mask,
                }),
                &mut machine,
            );
            let mask = u32::from(mask);
            assert_eq!(
                machine.regs[0],
                (0xa5a5_a5a5 & !mask) | (u32::from(preds) & mask)
            );
        }
    }
    for byte in 0..4_u8 {
        for mask in [0_u8, 7, 0x40, 0x7f] {
            let value = 0x5a_u32 << (8 * byte) | 0x0101_0101 & !(0xff << (8 * byte));
            let mut machine = machine_with(&[value]);
            let before = [true, false, true, false, true, false, true];
            machine.preds[..7].copy_from_slice(&before);
            run_op(
                Op::R2P(OpR2P {
                    src: r(0),
                    byte,
                    mask,
                }),
                &mut machine,
            );
            for (index, was) in before.iter().enumerate() {
                let expected = if mask & (1 << index) != 0 {
                    0x5a & (1 << index) != 0
                } else {
                    *was
                };
                assert_eq!(
                    machine.preds[index], expected,
                    "byte {byte} mask {mask:#x} P{index}"
                );
            }
        }
    }
}
