//! Decoder tests: expected text, round trips, strictness and robustness.

use proptest::prelude::*;

use super::*;
use crate::asm::{EncodeError, assemble};
use crate::testing::{FORMS, any_form_word, any_instr, config};

fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

fn p(index: u8) -> Pred {
    Pred::new(index).expect("a predicate register")
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
            [
                FloatSrc::reg(r(2)),
                FloatSrc::reg(r(3)),
                FloatSrc::reg(r(4)),
            ],
            "r2, r3, r4",
        ),
        (
            [
                FloatSrc::reg(r(2)),
                FloatSrc::Imm(2.0_f32.to_bits()),
                FloatSrc::reg(r(4)),
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
                    FloatSrc::Reg {
                        reg: r(1),
                        abs: true,
                        neg: true,
                    },
                    FloatSrc::Reg {
                        reg: r(2),
                        abs: false,
                        neg: true,
                    },
                    FloatSrc::Imm(0.5_f32.to_bits()),
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
                    FloatSrc::reg(r(1)),
                    FloatSrc::Imm(f32::NEG_INFINITY.to_bits()),
                    FloatSrc::Reg {
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
                dst: r(3),
                sr: SpecialReg::LANE_ID,
            }),
            "s2r r3, sr_laneid;",
        ),
        (
            Op::S2R(OpS2R {
                dst: r(3),
                sr: SpecialReg::TID_Y,
            }),
            "s2r r3, sr_tid.y;",
        ),
        (
            Op::S2R(OpS2R {
                dst: r(3),
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
    let text = |bits: u32| FloatSrc::Imm(bits).to_string();
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

/// The known-bit masks against PR #4588's generated table (`ampere_isa_table.h` @ e5fb3b97;
/// each row's four words, low word first). `MOV`'s register form differs by bits 62 and 63
/// only (the divergence in this module's docs).
#[test]
fn masks_match_the_reference_table() {
    fn row(words: [u32; 4]) -> u128 {
        words
            .iter()
            .rev()
            .fold(0, |bits, &word| (bits << 32) | u128::from(word))
    }
    let slot_modifiers = SLOT_ABS.mask() | SLOT_NEG.mask();
    let rows = [
        (
            MOV_REG_KNOWN | slot_modifiers,
            [0x00ff_ffff, 0xc000_00ff, 0x0000_0f00, 0xffff_fe00],
        ),
        (
            MOV_IMM_KNOWN,
            [0x00ff_ffff, 0xffff_ffff, 0x0000_0f00, 0xffff_fe00],
        ),
        (
            FFMA_REG_KNOWN,
            [0xffff_ffff, 0xc000_00ff, 0x0001_ffff, 0xffff_fe00],
        ),
        (
            FFMA_IMM_SRC2_KNOWN,
            [0xffff_ffff, 0xffff_ffff, 0x0001_ffff, 0xffff_fe00],
        ),
        (
            FFMA_IMM_SRC1_KNOWN,
            [0xffff_ffff, 0xffff_ffff, 0x0001_ffff, 0xffff_fe00],
        ),
        (
            ALD_KNOWN,
            [0xffff_ffff, 0x0003_ffff, 0x0000_bc00, 0xffff_fe00],
        ),
        (
            AST_KNOWN,
            [0xff00_ffff, 0x0003_ffff, 0x0000_3cff, 0xffff_fe00],
        ),
        (
            EXIT_KNOWN,
            [0x0000_ffff, 0x0000_0000, 0x07b0_0000, 0xffff_fe00],
        ),
        (
            S2R_KNOWN,
            [0x00ff_ffff, 0x0000_0000, 0x0000_ff00, 0xffff_fe00],
        ),
    ];
    for (mine, words) in rows {
        assert_eq!(mine, row(words), "{words:x?}");
    }
    assert_eq!(MOV_REG_KNOWN & slot_modifiers, 0);
}

#[test]
fn mov_register_form_rejects_source_modifier_bits() {
    let mov = assemble(&Instr::new(Op::Mov(OpMov {
        dst: r(1),
        src: MovSrc::Reg(r(2)),
        quad_lanes: OpMov::ALL_LANES,
    })))
    .expect("encodable");
    for bit in [SLOT_ABS, SLOT_NEG] {
        assert_eq!(
            decode(mov.with_flag(bit, true)),
            Err(DecodeError::UnknownBit {
                opcode: 0x202,
                bit: bit.start
            })
        );
    }
}

/// Flipping any bit outside a form's mask makes the word unknown, naming that bit.
#[test]
fn every_bit_outside_a_form_is_rejected() {
    for (opcode, known) in FORMS {
        let word = Instruction::default().with_field(OPCODE, opcode);
        assert!(decode(word).is_ok(), "{opcode:#x}");
        for bit in 0..128 {
            let flipped = word.bits() ^ (1 << bit);
            if known & (1 << bit) != 0 {
                continue;
            }
            assert_eq!(
                decode(Instruction::from_bits(flipped)),
                Err(DecodeError::UnknownBit {
                    opcode: u16::try_from(opcode).expect("12 bits"),
                    bit
                }),
                "{opcode:#x} bit {bit}"
            );
        }
    }
}

#[test]
fn unknown_opcodes_name_the_opcode() {
    // FFMA's constant-buffer form (5) and MOV's uniform-register form (6) are not decoded yet.
    for opcode in [0x000, 0xa23, 0xc02, 0x9c3, 0xfff] {
        let word = Instruction::default().with_field(OPCODE, opcode);
        assert_eq!(
            decode(word),
            Err(DecodeError::UnknownOpcode {
                opcode: u16::try_from(opcode).expect("12 bits")
            })
        );
        assert_eq!(
            decode(word).map_err(|error| error.to_string()),
            Err(format!("unknown opcode {opcode:#05x}"))
        );
    }
    // The opcode is checked before the stray bits: an unknown opcode with stray bits too is
    // reported as the opcode.
    let both = Instruction::default()
        .with_field(OPCODE, 0xa23)
        .with_flag(Field::bit(100), true);
    assert_eq!(
        decode(both),
        Err(DecodeError::UnknownOpcode { opcode: 0xa23 })
    );
    let stray = Instruction::default()
        .with_field(OPCODE, EXIT_OPCODE)
        .with_flag(Field::bit(16), true);
    assert_eq!(
        decode(stray).map_err(|error| error.to_string()),
        Err("opcode 0x94d has bit 16 set, which its form does not use".into())
    );
}

#[test]
fn scheduling_bits_do_not_change_the_decoding() {
    let word = assemble(&Instr::new(Op::S2R(OpS2R {
        dst: r(9),
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
    let imm = FloatSrc::Imm(0);
    let reg = FloatSrc::reg(r(1));
    assert_eq!(
        ffma([imm, reg, reg]),
        Err(EncodeError::ImmediateFirstSource)
    );
    assert_eq!(ffma([reg, imm, imm]), Err(EncodeError::TwoImmediates));
    let mov = assemble(&Instr::new(Op::Mov(OpMov {
        dst: r(0),
        src: MovSrc::Reg(r(1)),
        quad_lanes: 0x10,
    })));
    assert_eq!(
        mov,
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

    /// A word the decoder accepts re-assembles to itself, scheduling bits aside.
    #[test]
    fn decode_then_assemble_round_trips(word in any_form_word(), sched in any::<u32>()) {
        let word = word.with_field(SCHEDULING, u64::from(sched));
        let instr = decode(word).expect("a form's own fields are known");
        let again = assemble(&instr).expect("a decoded instruction is encodable");
        prop_assert_eq!(again.bits(), word.bits() & !SCHEDULING.mask());
    }

    /// Arbitrary words never panic the decoder.
    #[test]
    fn arbitrary_words_never_panic(bits in any::<u128>()) {
        let _ = decode(Instruction::from_bits(bits));
    }
}
