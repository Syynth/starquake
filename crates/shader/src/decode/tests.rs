//! Decoder tests: expected text, round trips, strictness and robustness. The framework's own
//! tests (form table, operand types at their edges, shells, check order) are in `framework`.

mod attribute;
mod control;
mod float32;
mod framework;
mod half_double;
mod integer;
mod memory;
mod texture;
mod uniform;

use proptest::prelude::*;

use super::fields::ALU_FORM;
use super::print::FloatOperand;
use super::*;
use crate::asm::{EncodeError, assemble};
use crate::decode::control::EXIT_OPCODE;
use crate::decode::integer::MOV_OPCODE;
use crate::testing::{any_form_word, any_instr, any_word_of, canonical_word, config, forms};

/// The 9-bit opcode below the operand form of a 12-bit ALU `opcode` (its low [`ALU_FORM`]`.start`
/// bits: `encode_alu_base`'s `set_field(0..9, opcode)`).
fn base_of(opcode: u16) -> u16 {
    opcode & ((1 << ALU_FORM.start) - 1)
}

fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

fn p(index: u8) -> Pred {
    Pred::new(index).expect("a predicate register")
}

/// A word built by hand: `opcode`, the guard `PT` (7 at bit 12), and each `(start, value)` placed
/// at its bit, with positions written out here rather than taken from the decoder's constants.
fn word(opcode: u128, fields: &[(u32, u128)]) -> Instruction {
    let mut bits = opcode | (7 << 12);
    for &(start, value) in fields {
        bits |= value << start;
    }
    Instruction::from_bits(bits)
}

/// Decodes a hand-built word and prints it.
fn text(word: Instruction) -> String {
    match decode(word) {
        Ok(instr) => instr.to_string(),
        Err(error) => format!("error: {error}"),
    }
}

fn ur(index: u8) -> UniformReg {
    UniformReg::new(index).expect("a uniform register")
}

/// Assembles `instr`, decodes the word and prints it.
fn round_text(instr: Instr) -> String {
    let word = assemble(&instr).expect("encodable");
    let decoded = decode(word).expect("decodable");
    assert_eq!(decoded, instr, "decoding {word:x?}");
    decoded.to_string()
}

/// NAK's `test_float_ops` `FFMA` cases, ported (`nvdisasm_tests.rs` @ 52238762): the source
/// pairs `r2, r3, r4` and `r2, 2, r4`, each with `.ftz`, `.rz` (with `.re` printing nothing)
/// and `.sat` on and off, and the text NAK checks against `nvdisasm` for SM86.
#[test]
fn ffma_text_matches_nak_nvdisasm_cases() {
    let cases = [
        (
            [AluSrc::reg(r(2)), AluSrc::reg(r(3)), AluSrc::reg(r(4))],
            "r2, r3, r4",
        ),
        (
            [
                AluSrc::reg(r(2)),
                AluSrc::Imm(2.0_f32.to_bits()),
                AluSrc::reg(r(4)),
            ],
            "r2, 2, r4",
        ),
    ];
    let mut checked = 0;
    for (ftz, ftz_str) in [(false, ""), (true, ".ftz")] {
        for (rnd, rnd_str) in [(RoundMode::NearestEven, ""), (RoundMode::Zero, ".rz")] {
            for (saturate, saturate_str) in [(false, ""), (true, ".sat")] {
                for (srcs, srcs_str) in cases {
                    let instr = Instr::new(Op::FFma(OpFFma {
                        dst: r(1),
                        srcs,
                        saturate,
                        rnd,
                        ftz,
                        dnz: false,
                    }));
                    let expected = format!("ffma{ftz_str}{rnd_str}{saturate_str} r1, {srcs_str};");
                    assert_eq!(round_text(instr), expected);
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 16);
}

/// The printer's text for the forms NAK's `nvdisasm` tests do not cover. **Unconfirmed**: these
/// pin the printer's guesses (see `print`'s module docs) until a developer with `nvdisasm` runs
/// the cross-check.
#[test]
fn unconfirmed_text_of_the_other_forms() {
    let cases = [
        (
            Op::Mov(OpMov {
                dst: r(1),
                src: MovSrc::Reg(r(2)),
                quad_lanes: OpMov::ALL_LANES,
            }),
            "mov r1, r2;",
        ),
        (
            Op::Mov(OpMov {
                dst: r(5),
                src: MovSrc::Imm(0x3f80_0000),
                quad_lanes: 0x3,
            }),
            "mov r5, 0x3f800000, 0x3;",
        ),
        (
            Op::Mov(OpMov {
                dst: Gpr::RZ,
                src: MovSrc::Reg(Gpr::RZ),
                quad_lanes: OpMov::ALL_LANES,
            }),
            "mov rz, rz;",
        ),
        (
            Op::FFma(OpFFma {
                dst: r(0),
                srcs: [
                    AluSrc::Reg {
                        reg: r(1),
                        abs: true,
                        neg: true,
                    },
                    AluSrc::Reg {
                        reg: r(2),
                        abs: false,
                        neg: true,
                    },
                    AluSrc::Imm(0.5_f32.to_bits()),
                ],
                saturate: false,
                rnd: RoundMode::NegInf,
                ftz: false,
                dnz: true,
            }),
            "ffma.fmz.rm r0, -|r1|, -r2, 0.5;",
        ),
        (
            Op::FFma(OpFFma {
                dst: r(0),
                srcs: [
                    AluSrc::reg(r(1)),
                    AluSrc::Imm(f32::NEG_INFINITY.to_bits()),
                    AluSrc::Reg {
                        reg: r(3),
                        abs: true,
                        neg: false,
                    },
                ],
                saturate: false,
                rnd: RoundMode::PosInf,
                ftz: false,
                dnz: false,
            }),
            "ffma.rp r0, r1, -inf, |r3|;",
        ),
    ];
    for (op, expected) in cases {
        assert_eq!(round_text(Instr::new(op)), expected);
    }
}

/// More unconfirmed text: attributes, exit and special registers.
#[test]
fn unconfirmed_text_of_attribute_exit_and_s2r() {
    let cases = [
        (
            Op::ALd(OpALd {
                dst: r(4),
                vtx: r(7),
                offset: r(9),
                addr: 0x80,
                comps: 4,
                patch: true,
                phys: true,
                output: true,
            }),
            "ald.o.p.phys.128 r4, a[r7][0x80+r9];",
        ),
        (
            Op::ALd(OpALd {
                dst: r(4),
                vtx: Gpr::RZ,
                offset: Gpr::RZ,
                addr: 0x70,
                comps: 1,
                patch: false,
                phys: false,
                output: false,
            }),
            "ald r4, a[0x70];",
        ),
        (
            Op::ASt(OpASt {
                data: r(2),
                vtx: Gpr::RZ,
                offset: Gpr::RZ,
                addr: 0x90,
                comps: 3,
                patch: false,
                phys: false,
            }),
            "ast.96 a[0x90], r2;",
        ),
        (
            Op::ASt(OpASt {
                data: r(2),
                vtx: r(1),
                offset: Gpr::RZ,
                addr: 0x84,
                comps: 2,
                patch: true,
                phys: false,
            }),
            "ast.p.64 a[r1][0x84], r2;",
        ),
        (
            Op::Exit(OpExit {
                keep_ref_count: false,
                no_at_exit: false,
                pred: PredSrc::TRUE,
            }),
            "exit;",
        ),
        (
            Op::Exit(OpExit {
                keep_ref_count: true,
                no_at_exit: true,
                pred: PredSrc {
                    reg: p(2),
                    negated: true,
                },
            }),
            "exit.keeprefcount.no_atexit !p2;",
        ),
        (
            Op::S2R(OpS2R {
                dst: S2RDst::Reg(r(3)),
                sr: SpecialReg::LANE_ID,
            }),
            "s2r r3, sr_laneid;",
        ),
        (
            Op::S2R(OpS2R {
                dst: S2RDst::Reg(r(3)),
                sr: SpecialReg::TID_Y,
            }),
            "s2r r3, sr_tid.y;",
        ),
        (
            Op::S2R(OpS2R {
                dst: S2RDst::Reg(r(3)),
                sr: SpecialReg(12),
            }),
            "s2r r3, sr12;",
        ),
    ];
    for (op, expected) in cases {
        assert_eq!(round_text(Instr::new(op)), expected);
    }
}

/// Unconfirmed: the guard prefix.
#[test]
fn unconfirmed_text_of_guards() {
    let guarded = Instr {
        guard: PredSrc {
            reg: p(0),
            negated: false,
        },
        op: Op::Exit(OpExit {
            keep_ref_count: false,
            no_at_exit: false,
            pred: PredSrc::TRUE,
        }),
    };
    assert_eq!(round_text(guarded), "@p0 exit;");
    let never = Instr {
        guard: PredSrc {
            reg: Pred::PT,
            negated: true,
        },
        ..guarded
    };
    assert_eq!(round_text(never), "@!pt exit;");
}

#[test]
fn float_immediates_print_integers_plainly_and_specials_by_name() {
    let text = |bits: u32| FloatOperand(AluSrc::Imm(bits)).to_string();
    assert_eq!(text(2.0_f32.to_bits()), "2");
    assert_eq!(text((-0.0_f32).to_bits()), "-0");
    assert_eq!(text(16_777_215.0_f32.to_bits()), "16777215");
    // 2^24 is the first magnitude printed in Rust's shortest form instead.
    assert_eq!(text(16_777_216.0_f32.to_bits()), "16777216.0");
    assert_eq!(text(0.25_f32.to_bits()), "0.25");
    assert_eq!(text(f32::INFINITY.to_bits()), "+inf");
    assert_eq!(text(f32::NAN.to_bits()), "+qnan");
    assert_eq!(text((-f32::NAN).to_bits()), "-qnan");
}

/// Words built by hand from the field positions, not by the assembler: one per form.
#[test]
fn hand_built_words_decode() {
    // FFMA r1, r2, r3, r4: opcode 0x223, guard PT (7 << 12), dst 1 << 16, src0 2 << 24,
    // src1 3 << 32, src2 4 << 64.
    let ffma = Instruction::from_bits(0x0000_0000_0000_0004_0000_0003_0201_7223);
    assert_eq!(
        decode(ffma).map(|i| i.to_string()),
        Ok("ffma r1, r2, r3, r4;".into())
    );
    // FFMA r1, r2, 2.0, r4 (form 4): opcode 0x823, imm 0x4000_0000 at bit 32.
    let ffma_imm = Instruction::from_bits(0x0000_0000_0000_0004_4000_0000_0201_7823);
    assert_eq!(
        decode(ffma_imm).map(|i| i.to_string()),
        Ok("ffma r1, r2, 2, r4;".into())
    );
    // FFMA r1, r2, r4, 2.0 (form 2): opcode 0x423, imm at bit 32, source 1 at bit 64.
    let ffma_imm2 = Instruction::from_bits(0x0000_0000_0000_0004_4000_0000_0201_7423);
    assert_eq!(
        decode(ffma_imm2).map(|i| i.to_string()),
        Ok("ffma r1, r2, r4, 2;".into())
    );
    // MOV r5, r6: opcode 0x202, dst 5 << 16, src 6 << 32, lanes 0xf << 72.
    let mov = Instruction::from_bits(0x0000_0000_0000_0f00_0000_0006_0005_7202);
    assert_eq!(decode(mov).map(|i| i.to_string()), Ok("mov r5, r6;".into()));
    // MOV r5, 0x12345678: opcode 0x802.
    let mov_imm = Instruction::from_bits(0x0000_0000_0000_0f00_1234_5678_0005_7802);
    assert_eq!(
        decode(mov_imm).map(|i| i.to_string()),
        Ok("mov r5, 0x12345678;".into())
    );
    // ALD.64 r8, a[0x84]: opcode 0x321, dst 8, offset RZ (0xff << 24), vtx RZ (0xff << 32),
    // addr 0x84 << 40, comps - 1 = 1 << 74.
    let ald = Instruction::from_bits(0x0000_0000_0000_0400_0000_84ff_ff08_7321);
    assert_eq!(
        decode(ald).map(|i| i.to_string()),
        Ok("ald.64 r8, a[0x84];".into())
    );
    // AST a[0x70], r0: opcode 0x322, offset RZ, data r0 at bit 32, addr 0x70 << 40, vtx RZ at
    // bit 64.
    let ast = Instruction::from_bits(0x0000_0000_0000_00ff_0000_7000_ff00_7322);
    assert_eq!(
        decode(ast).map(|i| i.to_string()),
        Ok("ast a[0x70], r0;".into())
    );
    // EXIT: opcode 0x94d, exit predicate PT (7 << 87).
    let exit = Instruction::from_bits(0x0000_0000_0380_0000_0000_0000_0000_794d);
    assert_eq!(decode(exit).map(|i| i.to_string()), Ok("exit;".into()));
    // S2R r2, SR_TID.X: opcode 0x919, dst 2, index 33 << 72, guard P1 negated (0x9 << 12).
    let s2r = Instruction::from_bits(0x0000_0000_0000_2100_0000_0000_0002_9919);
    assert_eq!(
        decode(s2r).map(|i| i.to_string()),
        Ok("@!p1 s2r r2, sr_tid.x;".into())
    );
}

/// `FFMA r1, r2, r3, r4` built by hand with one more bit of word 2 (bits 64..96) set, so each
/// modifier, rounding-mode and flag position is pinned by a word the assembler did not make.
#[test]
fn hand_built_ffma_words_pin_each_modifier_and_mode_bit() {
    // Opcode 0x223, guard PT (7 << 12), dst 1 << 16, src0 2 << 24, src1 3 << 32, src2 4 << 64.
    let plain = 0x0000_0000_0000_0004_0000_0003_0201_7223_u128;
    let cases = [
        // Bit 72 (word 2 bit 8): source 0's negation.
        (0x100, "ffma r1, -r2, r3, r4;"),
        // Bit 73: source 0's absolute value.
        (0x200, "ffma r1, |r2|, r3, r4;"),
        // Bit 74: source 2's absolute value.
        (0x400, "ffma r1, r2, r3, |r4|;"),
        // Bit 75: source 2's negation.
        (0x800, "ffma r1, r2, r3, -r4;"),
        // Bit 76: the multiply-by-zero mode.
        (0x1000, "ffma.fmz r1, r2, r3, r4;"),
        // Bit 77: saturate.
        (0x2000, "ffma.sat r1, r2, r3, r4;"),
        // Bits 78..80 = 1: toward negative infinity.
        (0x4000, "ffma.rm r1, r2, r3, r4;"),
        // Bits 78..80 = 2: toward positive infinity.
        (0x8000, "ffma.rp r1, r2, r3, r4;"),
        // Bits 78..80 = 3: toward zero.
        (0xc000, "ffma.rz r1, r2, r3, r4;"),
    ];
    for (word2, expected) in cases {
        let word = Instruction::from_bits(plain | (word2 << 64));
        assert_eq!(
            decode(word).map(|i| i.to_string()),
            Ok(expected.into()),
            "{word2:#x}"
        );
    }
}

/// Hand-built words of the operand forms slice 4 added: constant buffers (bound and bindless)
/// and uniform registers, in source 1 and source 2.
#[test]
fn hand_built_words_of_the_wide_slot_forms_decode() {
    // MOV r5, c[0x3][0x10]: opcode 0xa02, dst 5 << 16, offset 0x10 << 38, index 3 << 54
    // (word 1 = 0x10 << 6 | 3 << 22 = 0x00c0_0400), lanes 0xf << 72.
    let mov_cbuf = Instruction::from_bits(0x0000_0000_0000_0f00_00c0_0400_0005_7a02);
    assert_eq!(
        decode(mov_cbuf).map(|i| i.to_string()),
        Ok("mov r5, c[0x3][0x10];".into())
    );
    // MOV r5, ur9: opcode 0xc02, ur 9 << 32, bit 91 (1 << 27 of word 2).
    let mov_ureg = Instruction::from_bits(0x0000_0000_0800_0f00_0000_0009_0005_7c02);
    assert_eq!(
        decode(mov_ureg).map(|i| i.to_string()),
        Ok("mov r5, ur9;".into())
    );
    // FFMA r1, r2, r4, c[0x1][0x8] (form 3): opcode 0x623, source 1 in the register slot at
    // bit 64, offset 8 << 38, index 1 << 54 (word 1 = 0x8 << 6 | 1 << 22 = 0x0040_0200).
    let ffma_cbuf2 = Instruction::from_bits(0x0000_0000_0000_0004_0040_0200_0201_7623);
    assert_eq!(
        decode(ffma_cbuf2).map(|i| i.to_string()),
        Ok("ffma r1, r2, r4, c[0x1][0x8];".into())
    );
    // FFMA r1, r2, -cx[ur6][0x20], r4 (form 5, bindless): ur 6 at bit 32, offset 0x20 << 38,
    // neg bit 63, bit 91 (word 1 = 6 | 0x20 << 6 | 1 << 31 = 0x8000_0806).
    let ffma_bindless = Instruction::from_bits(0x0000_0000_0800_0004_8000_0806_0201_7a23);
    assert_eq!(
        decode(ffma_bindless).map(|i| i.to_string()),
        Ok("ffma r1, r2, -cx[ur6][0x20], r4;".into())
    );
    // FFMA r1, r2, ur3, r4 (form 6): ur 3 at bit 32, bit 91.
    let ffma_ureg1 = Instruction::from_bits(0x0000_0000_0800_0004_0000_0003_0201_7c23);
    assert_eq!(
        decode(ffma_ureg1).map(|i| i.to_string()),
        Ok("ffma r1, r2, ur3, r4;".into())
    );
    // FFMA r1, r2, r4, |urz| (form 7): urz (63) at bit 32, abs bit 62, bit 91 (word 1 =
    // 0x3f | 1 << 30 = 0x4000_003f).
    let ffma_ureg2 = Instruction::from_bits(0x0000_0000_0800_0004_4000_003f_0201_7e23);
    assert_eq!(
        decode(ffma_ureg2).map(|i| i.to_string()),
        Ok("ffma r1, r2, r4, |urz|;".into())
    );
}

/// Unconfirmed text of the wide-slot forms (the operand syntax is NAK's, the whole line a
/// guess; see `print`).
#[test]
fn unconfirmed_text_of_the_wide_slot_forms() {
    let ffma = |b, c| {
        Instr::new(Op::FFma(OpFFma {
            dst: r(1),
            srcs: [AluSrc::reg(r(2)), b, c],
            saturate: false,
            rnd: RoundMode::NearestEven,
            ftz: false,
            dnz: false,
        }))
    };
    let cbuf = CBufRef {
        buf: CBuf::Binding(5),
        offset: 0x100,
    };
    let bindless = CBufRef {
        buf: CBuf::BindlessUniform(UniformReg::new(2).expect("ur2")),
        offset: 0x100,
    };
    let unmodified = |cbuf| AluSrc::CBuf {
        cbuf,
        abs: false,
        neg: false,
    };
    assert_eq!(
        round_text(ffma(unmodified(cbuf), AluSrc::reg(r(4)))),
        "ffma r1, r2, c[0x5][0x100], r4;"
    );
    assert_eq!(
        round_text(ffma(AluSrc::reg(r(4)), unmodified(bindless))),
        "ffma r1, r2, r4, cx[ur2][0x100];"
    );
    let ur3 = AluSrc::UniformReg {
        reg: UniformReg::new(3).expect("ur3"),
        abs: true,
        neg: true,
    };
    assert_eq!(
        round_text(ffma(ur3, AluSrc::reg(r(4)))),
        "ffma r1, r2, -|ur3|, r4;"
    );
    let mov = |src| {
        Instr::new(Op::Mov(OpMov {
            dst: r(1),
            src,
            quad_lanes: OpMov::ALL_LANES,
        }))
    };
    assert_eq!(
        round_text(mov(MovSrc::CBuf(bindless))),
        "mov r1, cx[ur2][0x100];"
    );
    assert_eq!(
        round_text(mov(MovSrc::UniformReg(UniformReg::URZ))),
        "mov r1, urz;"
    );
}

/// The known-bit mask of the form of `opcode`.
fn known_of(opcode: u16) -> u128 {
    lookup(opcode).expect("a form").known
}

/// Each group's known-bit masks against the rows of PR #4588's generated table that the group
/// lists (`REFERENCE_ROWS`, one per form, in form order): equal, except for the bits a row
/// drops, which the form rejects (bits 62 and 63 of `MOV`'s register, constant-buffer and
/// uniform-register forms, `mov_rejects_source_modifier_bits`, and of the same forms of
/// `WARPSYNC` and `NANOSLEEP`, `control::control_sources_reject_modifier_bits`), and the bits
/// a row adds, which the form knows though the row leaves them out
/// (`framework::added_bits_are_the_listed_cases`). A row drops only bits it marks known and
/// adds only bits it leaves out, so no bit is both. Each form's fixed bits are the ones its row
/// lists (`ReferenceRow::fixing`).
#[test]
fn masks_match_the_reference_table() {
    for (forms, hooks) in GROUPS.iter().zip(&GROUP_HOOKS) {
        let rows = hooks.reference_rows;
        assert_eq!(forms.len(), rows.len());
        for (form, row) in forms.iter().zip(rows) {
            let opcode = form.opcode;
            assert_eq!(opcode, row.opcode);
            assert_eq!(row.dropped & !row.mask(), 0, "{opcode:#x}");
            assert_eq!(row.added() & row.mask(), 0, "{opcode:#x}");
            assert_eq!(
                form.known,
                (row.mask() & !row.dropped) | row.added(),
                "{opcode:#x}"
            );
            assert_eq!(form.fixed, row.fixed, "{opcode:#x}");
        }
    }
}

/// `MOV` rejects the slot's modifier bits in each form whose slot has them in PR #4588's table.
#[test]
fn mov_rejects_source_modifier_bits() {
    let mov = |src| {
        assemble(&Instr::new(Op::Mov(OpMov {
            dst: r(1),
            src,
            quad_lanes: OpMov::ALL_LANES,
        })))
        .expect("encodable")
    };
    let cbuf = CBufRef {
        buf: CBuf::Binding(1),
        offset: 4,
    };
    let words = [
        (AluForm::Reg, mov(MovSrc::Reg(r(2)))),
        (AluForm::CBufSrc1, mov(MovSrc::CBuf(cbuf))),
        (
            AluForm::UniformSrc1,
            mov(MovSrc::UniformReg(UniformReg::URZ)),
        ),
    ];
    // These forms, and no other `MOV` form, drop exactly the two bits from PR #4588's rows.
    let forms: Vec<u16> = words
        .iter()
        .map(|(form, _)| form.opcode(MOV_OPCODE))
        .collect();
    let mov_rows = super::integer::REFERENCE_ROWS
        .iter()
        .filter(|row| base_of(row.opcode) == MOV_OPCODE);
    for row in mov_rows {
        let expected = if forms.contains(&row.opcode) {
            SLOT_ABS.mask() | SLOT_NEG.mask()
        } else {
            0
        };
        assert_eq!(row.dropped, expected, "{:#x}", row.opcode);
    }
    for (form, word) in words {
        assert!(decode(word).is_ok());
        for bit in [SLOT_ABS, SLOT_NEG] {
            assert_eq!(
                decode(word.with_flag(bit, true)),
                Err(DecodeError::UnknownBit {
                    opcode: form.opcode(MOV_OPCODE),
                    bit: bit.start
                })
            );
        }
    }
}

/// Flipping any bit outside a form's mask in its canonical word makes the word unknown, naming
/// that bit, whatever the form's fields hold (the mask is checked before anything is read).
#[test]
fn every_bit_outside_a_form_is_rejected() {
    for (opcode, known, _) in forms() {
        let word = canonical_word(opcode);
        assert!(decode(word).is_ok(), "{opcode:#x}");
        let mut flipped_bits = 0;
        for bit in 0..128 {
            if known & (1 << bit) != 0 {
                continue;
            }
            flipped_bits += 1;
            assert_eq!(
                decode(Instruction::from_bits(word.bits() ^ (1 << bit))),
                Err(DecodeError::UnknownBit { opcode, bit }),
                "{opcode:#x} bit {bit}"
            );
        }
        assert_eq!(flipped_bits, 128 - known.count_ones());
    }
}

#[test]
fn unknown_opcodes_name_the_opcode() {
    // Form 0 of MOV and FFMA, MOV's forms 2, 3 and 7 (rows in PR #4588's table, not decoded:
    // `op::integer`), and opcodes no decoded instruction has.
    let unknown = [
        MOV_OPCODE,
        AluForm::ImmSrc2.opcode(MOV_OPCODE),
        AluForm::CBufSrc2.opcode(MOV_OPCODE),
        AluForm::UniformSrc2.opcode(MOV_OPCODE),
        float::FFMA_OPCODE,
        0x000,
        0x9ff,
        0xfff,
    ];
    for opcode in unknown {
        let word = Instruction::default().with_field(OPCODE, u64::from(opcode));
        assert_eq!(decode(word), Err(DecodeError::UnknownOpcode { opcode }));
        assert_eq!(
            decode(word).map_err(|error| error.to_string()),
            Err(format!("unknown opcode {opcode:#05x}"))
        );
    }
    // The opcode is checked before the stray bits: an unknown opcode with stray bits too is
    // reported as the opcode.
    let both = Instruction::default()
        .with_field(OPCODE, 0x9ff)
        .with_flag(Field::bit(100), true);
    assert_eq!(
        decode(both),
        Err(DecodeError::UnknownOpcode { opcode: 0x9ff })
    );
    let stray = Instruction::default()
        .with_field(OPCODE, u64::from(EXIT_OPCODE))
        .with_flag(Field::bit(16), true);
    assert_eq!(
        decode(stray).map_err(|error| error.to_string()),
        Err("opcode 0x94d has bit 16 set, which its form does not use".into())
    );
}

#[test]
fn scheduling_bits_do_not_change_the_decoding() {
    let word = assemble(&Instr::new(Op::S2R(OpS2R {
        dst: S2RDst::Reg(r(9)),
        sr: SpecialReg::LANE_ID,
    })))
    .expect("encodable");
    let plain = decode(word);
    for bit in SCHEDULING.start..128 {
        assert_eq!(
            decode(Instruction::from_bits(word.bits() ^ (1 << bit))),
            plain
        );
    }
    assert_eq!(SCHEDULING.start + SCHEDULING.width, 128);
}

#[test]
fn program_bytes_split_into_whole_instructions() {
    assert_eq!(Instruction::from_program(&[]), Ok(vec![]));
    let one = [0xab_u8; Instruction::SIZE];
    assert_eq!(
        Instruction::from_program(&one),
        Ok(vec![Instruction::from_le_bytes(one)])
    );
    for len in [Instruction::SIZE - 1, Instruction::SIZE + 1] {
        let bytes = vec![0; len];
        assert_eq!(
            Instruction::from_program(&bytes),
            Err(LengthError {
                len,
                size: Instruction::SIZE
            })
        );
    }
    assert_eq!(
        LengthError { len: 17, size: 16 }.to_string(),
        "a program of 17 bytes is not a whole number of 16-byte instructions"
    );
}

#[test]
fn words_are_little_endian_with_bit_n_in_word_n_div_32() {
    let mut bytes = [0_u8; Instruction::SIZE];
    bytes[0] = 0x01;
    bytes[15] = 0x80;
    let word = Instruction::from_le_bytes(bytes);
    assert!(word.flag(Field::bit(0)));
    assert!(word.flag(Field::bit(127)));
    assert_eq!(word.to_le_bytes(), bytes);
}

#[test]
fn fields_past_bit_127_read_zero_and_wide_fields_keep_64_bits() {
    let ones = Instruction::from_bits(u128::MAX);
    assert_eq!(ones.field(Field::new(120, 16)), 0xff);
    assert_eq!(ones.field(Field::new(128, 8)), 0);
    assert_eq!(ones.field(Field::new(0, 128)), u64::MAX);
    assert_eq!(ones.field(Field::new(0, 0)), 0);
    assert_eq!(Field::new(120, 16).mask(), 0xff << 120);
    assert_eq!(Field::new(0, 128).mask(), u128::MAX);
    assert_eq!(Field::new(128, 1).mask(), 0);
    let set = Instruction::default().with_field(Field::new(124, 8), 0xff);
    assert_eq!(set.bits(), 0xf << 124);
    assert_eq!(
        Instruction::default()
            .with_field(Field::new(4, 4), 0x1f)
            .bits(),
        0xf0
    );
}

#[test]
fn unencodable_instructions_are_errors() {
    let ffma = |srcs| {
        assemble(&Instr::new(Op::FFma(OpFFma {
            dst: r(0),
            srcs,
            saturate: false,
            rnd: RoundMode::NearestEven,
            ftz: false,
            dnz: false,
        })))
    };
    let imm = AluSrc::Imm(0);
    let reg = AluSrc::reg(r(1));
    let ureg = AluSrc::UniformReg {
        reg: UniformReg::URZ,
        abs: false,
        neg: false,
    };
    let cbuf = AluSrc::CBuf {
        cbuf: CBufRef {
            buf: CBuf::Binding(0),
            offset: 0,
        },
        abs: false,
        neg: false,
    };
    for first in [imm, ureg, cbuf] {
        assert_eq!(
            ffma([first, reg, reg]),
            Err(EncodeError::FirstSourceNotRegister { mnemonic: "ffma" })
        );
    }
    for (b, c) in [(imm, imm), (cbuf, ureg), (ureg, imm), (imm, cbuf)] {
        assert_eq!(
            ffma([reg, b, c]),
            Err(EncodeError::TwoWideSources { mnemonic: "ffma" })
        );
    }
    assert_eq!(
        ffma([reg, imm, imm]).map_err(|error| error.to_string()),
        Err("ffma can take only one immediate, constant-buffer or uniform-register source".into())
    );
    assert_eq!(
        ffma([imm, reg, reg]).map_err(|error| error.to_string()),
        Err("the first source of ffma must be a register".into())
    );
    let mov = |src, quad_lanes| {
        assemble(&Instr::new(Op::Mov(OpMov {
            dst: r(0),
            src,
            quad_lanes,
        })))
    };
    assert_eq!(
        mov(MovSrc::Reg(r(1)), 0x10),
        Err(EncodeError::TooWide {
            what: "quad lane mask",
            value: 0x10,
            bits: 4
        })
    );
    let ald = |addr, comps| {
        assemble(&Instr::new(Op::ALd(OpALd {
            dst: r(0),
            vtx: Gpr::RZ,
            offset: Gpr::RZ,
            addr,
            comps,
            patch: false,
            phys: false,
            output: false,
        })))
    };
    assert!(ald(0x3ff, 1).is_ok());
    assert_eq!(
        ald(0x400, 1),
        Err(EncodeError::TooWide {
            what: "attribute address",
            value: 0x400,
            bits: 10
        })
    );
    assert!(ald(0, 4).is_ok());
    assert_eq!(ald(0, 0), Err(EncodeError::Comps(0)));
    assert_eq!(ald(0, 5), Err(EncodeError::Comps(5)));
}

proptest! {
    #![proptest_config(config())]

    /// Assembling then decoding gives the instruction back, so every field is read from where
    /// the encoder put it; distinct random values in every field catch two swapped reads.
    #[test]
    fn assemble_then_decode_round_trips(instr in any_instr()) {
        let word = assemble(&instr).expect("the generator makes encodable instructions");
        prop_assert_eq!(decode(word), Ok(instr));
    }

    /// A word of a form either is rejected for a bit its layout does not read, or for a field
    /// value that names nothing, or decodes and re-assembles to itself, scheduling bits aside.
    #[test]
    fn decode_then_assemble_round_trips(word in any_form_word(), sched in any::<u32>()) {
        let word = word.with_field(SCHEDULING, u64::from(sched));
        match decode(word) {
            Ok(instr) => {
                let again = assemble(&instr).expect("a decoded instruction is encodable");
                prop_assert_eq!(again.bits(), word.bits() & !SCHEDULING.mask());
            }
            Err(DecodeError::UnknownBit { bit, .. }) => {
                // Only a layout bit (step 5) can fail a word made of the form's own bits.
                let read = bits_read(word).expect("the form's function accepts the word");
                prop_assert_eq!(read & (1 << bit), 0);
                prop_assert!(word.bits() & (1 << bit) != 0);
            }
            Err(DecodeError::UnknownValue { bit, width, value, .. }) => {
                // Step 4: the value is the word's own, in a field inside the form's mask.
                let field = Field::new(bit, width);
                prop_assert_eq!(word.field(field), value);
                let opcode = u16::try_from(word.field(OPCODE)).expect("12 bits");
                prop_assert_eq!(field.mask() & !known_of(opcode), 0);
            }
            Err(error) => prop_assert!(false, "{error}"),
        }
    }

    /// A form's function reads no bit outside the form's mask (a word it refuses at step 4,
    /// for a field value, has nothing read to compare).
    #[test]
    fn forms_read_only_their_known_bits(word in any_form_word()) {
        let opcode = u16::try_from(word.field(OPCODE)).expect("12 bits");
        match bits_read(word) {
            Some(read) => prop_assert_eq!(read & !known_of(opcode), 0),
            None => prop_assert!(
                matches!(decode(word), Err(DecodeError::UnknownValue { .. })),
                "{opcode:#x}"
            ),
        }
    }

    /// Arbitrary words never panic the decoder.
    #[test]
    fn arbitrary_words_never_panic(bits in any::<u128>()) {
        let _ = decode(Instruction::from_bits(bits));
    }
}
