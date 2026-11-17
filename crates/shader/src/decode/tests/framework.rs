//! Tests of the decoder framework: the form table, the order of checks, the layouts of the wide
//! slot, every operand type at the edges of its encoding, and the shells.

use super::super::form::Reader;
use super::super::op::{attribute, control, float, half_double, integer, memory, tensor};
use super::super::op::{texture, uniform};
use super::*;
use crate::asm::Writer;

/// Every shell, from every group.
fn shells() -> Vec<Op> {
    [
        integer::SHELLS,
        float::SHELLS,
        half_double::SHELLS,
        uniform::SHELLS,
        memory::SHELLS,
        texture::SHELLS,
        attribute::SHELLS,
        control::SHELLS,
        tensor::SHELLS,
    ]
    .concat()
}

/// The operations decoded so far.
const DECODED: [&str; 6] = ["mov", "ffma", "ald", "ast", "s2r", "exit"];

#[test]
fn forms_have_distinct_opcodes_and_lookup_finds_each() {
    let forms = forms();
    let mut opcodes: Vec<u16> = forms.iter().map(|&(opcode, _, _)| opcode).collect();
    opcodes.sort_unstable();
    opcodes.dedup();
    assert_eq!(opcodes.len(), forms.len());
    for (opcode, known, fixed) in forms {
        assert!(u64::from(opcode) <= OPCODE.mask().try_into().expect("12 bits"));
        let form = lookup(opcode).expect("listed forms are found");
        assert_eq!(
            (form.opcode, form.known, form.fixed),
            (opcode, known, fixed)
        );
        // Fixed bits are known bits, and every form explains the common fields.
        assert_eq!(fixed & !known, 0);
        assert_eq!(COMMON & !known, 0);
    }
    // MOV in four forms, FFMA in seven, ALD, AST, S2R, EXIT.
    assert_eq!(super::forms().len(), 4 + 7 + 4);
}

#[test]
fn alu_forms_put_sources_where_the_table_says() {
    let expected = [
        (AluForm::Reg, 1, Slot::WideReg, Slot::Reg),
        (AluForm::ImmSrc2, 2, Slot::Reg, Slot::Imm),
        (AluForm::CBufSrc2, 3, Slot::Reg, Slot::CBuf),
        (AluForm::ImmSrc1, 4, Slot::Imm, Slot::Reg),
        (AluForm::CBufSrc1, 5, Slot::CBuf, Slot::Reg),
        (AluForm::UniformSrc1, 6, Slot::UniformReg, Slot::Reg),
        (AluForm::UniformSrc2, 7, Slot::Reg, Slot::UniformReg),
    ];
    assert_eq!(AluForm::ALL.len(), expected.len());
    for (form, bits, one, two) in expected {
        assert_eq!(form.bits(), bits);
        assert_eq!(AluForm::from_bits(u64::from(bits)), Some(form));
        assert_eq!(form.slot(Source::One), one);
        assert_eq!(form.slot(Source::Two), two);
    }
    assert_eq!(AluForm::from_bits(0), None);
    assert_eq!(AluForm::from_bits(8), None);
}

/// The uniform-register forms require bit 91 (PR #4588: "native uniform source flag"); the
/// others do not have it fixed.
#[test]
fn uniform_forms_require_bit_91() {
    let required = [
        AluForm::UniformSrc1.opcode(integer::MOV_OPCODE),
        AluForm::UniformSrc1.opcode(float::FFMA_OPCODE),
        AluForm::UniformSrc2.opcode(float::FFMA_OPCODE),
    ];
    for (opcode, _, fixed) in forms() {
        let word = Instruction::default().with_field(OPCODE, u64::from(opcode));
        if required.contains(&opcode) {
            assert_eq!(fixed, SLOT_UNIFORM.mask(), "{opcode:#x}");
            assert_eq!(
                decode(word),
                Err(DecodeError::MissingBit { opcode, bit: 91 })
            );
            assert!(decode(word.with_flag(SLOT_UNIFORM, true)).is_ok());
        } else {
            assert_eq!(fixed, 0, "{opcode:#x}");
            assert!(decode(word).is_ok(), "{opcode:#x}");
        }
    }
    assert_eq!(
        decode(Instruction::default().with_field(OPCODE, 0xc23)).map_err(|e| e.to_string()),
        Err("opcode 0xc23 has bit 91 clear, which its form requires set".into())
    );
}

/// A word of FFMA with a bound constant buffer `c[3][0x40]` in source 1.
fn ffma_bound_cbuf() -> Instruction {
    assemble(&Instr::new(Op::FFma(OpFFma {
        dst: r(1),
        srcs: [
            AluSrc::reg(r(2)),
            AluSrc::CBuf {
                cbuf: CBufRef {
                    buf: CBuf::Binding(3),
                    offset: 0x40,
                },
                abs: false,
                neg: false,
            },
            AluSrc::reg(r(4)),
        ],
        saturate: false,
        rnd: RoundMode::NearestEven,
        ftz: false,
        dnz: false,
    })))
    .expect("encodable")
}

/// A bound constant buffer does not read the bindless register's bits, and a bindless one does
/// not read the index bits: either set in the other layout is an unknown bit, at both ends of
/// the field; the neighbouring offset bits are read in both.
#[test]
fn constant_buffer_layouts_reject_the_other_layouts_bits() {
    let bound = ffma_bound_cbuf();
    let opcode = AluForm::CBufSrc1.opcode(float::FFMA_OPCODE);
    assert!(decode(bound).is_ok());
    let ur = SLOT_UNIFORM_REG;
    for bit in [ur.start, ur.start + ur.width - 1] {
        assert_eq!(
            decode(bound.with_flag(Field::bit(bit), true)),
            Err(DecodeError::UnknownBit { opcode, bit })
        );
    }
    assert!(decode(bound.with_flag(Field::bit(CBUF_OFFSET.start), true)).is_ok());
    let bindless = bound
        .with_field(CBUF_INDEX, 0)
        .with_flag(SLOT_UNIFORM, true)
        .with_field(SLOT_UNIFORM_REG, 5);
    assert!(decode(bindless).is_ok());
    for bit in [CBUF_INDEX.start, CBUF_INDEX.start + CBUF_INDEX.width - 1] {
        assert_eq!(
            decode(bindless.with_flag(Field::bit(bit), true)),
            Err(DecodeError::UnknownBit { opcode, bit })
        );
    }
    let offset_end = CBUF_OFFSET.start + CBUF_OFFSET.width - 1;
    assert!(decode(bindless.with_flag(Field::bit(offset_end), true)).is_ok());
}

/// The checks run in the documented order: with faults of two steps present at once, the
/// earlier step's is reported.
#[test]
fn checks_run_in_the_documented_order() {
    // Step 2 before step 3: a uniform-register form with bit 91 clear and a bit outside its
    // mask.
    let opcode = AluForm::UniformSrc1.opcode(float::FFMA_OPCODE);
    let outside = Field::bit(100);
    assert_eq!(known_of(opcode) & outside.mask(), 0);
    let both = Instruction::default()
        .with_field(OPCODE, u64::from(opcode))
        .with_flag(outside, true);
    assert_eq!(
        decode(both),
        Err(DecodeError::UnknownBit { opcode, bit: 100 })
    );
    // Step 2 before step 5: a bound constant buffer with a bindless register bit (bit 33, a
    // step 5 fault) and a bit outside the mask (bit 100); step 5 alone would report bit 33.
    let cbuf = AluForm::CBufSrc1.opcode(float::FFMA_OPCODE);
    let layout = ffma_bound_cbuf().with_flag(Field::bit(33), true);
    assert_eq!(
        decode(layout),
        Err(DecodeError::UnknownBit {
            opcode: cbuf,
            bit: 33
        })
    );
    assert_eq!(
        decode(layout.with_flag(outside, true)),
        Err(DecodeError::UnknownBit {
            opcode: cbuf,
            bit: 100
        })
    );
    // Step 5 names the lowest unread bit.
    assert_eq!(
        decode(
            ffma_bound_cbuf()
                .with_flag(Field::bit(37), true)
                .with_flag(Field::bit(34), true)
        ),
        Err(DecodeError::UnknownBit {
            opcode: cbuf,
            bit: 34
        })
    );
}

/// Assembles and decodes `op`, checking the round trip, and prints it.
fn op_text(op: Op) -> String {
    round_text(Instr::new(op))
}

fn mov(dst: Gpr, src: MovSrc) -> Op {
    Op::Mov(OpMov {
        dst,
        src,
        quad_lanes: OpMov::ALL_LANES,
    })
}

#[test]
fn registers_at_the_edges_of_their_encoding() {
    for (index, text) in [(0, "r0"), (254, "r254"), (255, "rz")] {
        let reg = Gpr::new(index);
        assert_eq!(
            op_text(mov(reg, MovSrc::Reg(reg))),
            format!("mov {text}, {text};")
        );
    }
    assert_eq!(Gpr::RZ.index(), u8::MAX);
    assert_eq!(Gpr::BITS, DST.width);
}

#[test]
fn uniform_registers_at_the_edges_of_their_encoding() {
    let last = UniformReg::URZ.index();
    assert_eq!(u64::from(last), (1 << UniformReg::BITS) - 1);
    for (index, text) in [(0, "ur0"), (last - 1, "ur62"), (last, "urz")] {
        let reg = UniformReg::new(index).expect("in range");
        assert_eq!(
            op_text(mov(r(1), MovSrc::UniformReg(reg))),
            format!("mov r1, {text};")
        );
    }
    assert_eq!(UniformReg::new(last + 1), None);
    assert_eq!(UniformReg::new(u8::MAX), None);
    assert!(UniformReg::URZ.is_zero());
    assert!(!UniformReg::new(last - 1).expect("ur62").is_zero());
}

#[test]
fn predicates_at_the_edges_of_their_encoding() {
    let last = Pred::PT.index();
    assert_eq!(u64::from(last), (1 << Pred::BITS) - 1);
    for (index, text) in [(0, "p0"), (last - 1, "p6")] {
        let guarded = Instr {
            guard: PredSrc {
                reg: p(index),
                negated: true,
            },
            op: Op::Exit(OpExit {
                keep_ref_count: false,
                no_at_exit: false,
                pred: PredSrc {
                    reg: p(index),
                    negated: false,
                },
            }),
        };
        assert_eq!(round_text(guarded), format!("@!{text} exit {text};"));
    }
    assert_eq!(Pred::new(last + 1), None);
    assert!(Pred::PT.is_true());
}

/// Uniform predicates have no decoded form yet (slice 6d), so their reader and writer are
/// checked directly.
#[test]
fn uniform_predicates_at_the_edges_of_their_encoding() {
    let reg_field = Field::new(87, 3);
    let not_field = Field::bit(90);
    let last = UniformPred::UPT.index();
    for (index, negated, text) in [
        (0, false, "up0"),
        (last - 1, true, "!up6"),
        (last, false, "upt"),
    ] {
        let src = UniformPredSrc {
            reg: UniformPred::new(index).expect("in range"),
            negated,
        };
        let mut writer = Writer::default();
        writer.uniform_pred_src(reg_field, not_field, src);
        let word = writer.word();
        assert_eq!(word.field(reg_field), u64::from(index));
        let mut reader = Reader::new(word);
        assert_eq!(reader.uniform_pred_src(reg_field, not_field), src);
        assert_eq!(reader.unread(), 0);
        assert_eq!(src.to_string(), text);
    }
    assert_eq!(UniformPred::new(last + 1), None);
    assert!(UniformPred::UPT.is_true());
}

#[test]
fn constant_buffers_at_the_edges_of_their_encoding() {
    let last_index = u8::try_from((1_u32 << CBUF_INDEX.width) - 1).expect("five bits");
    let cases = [
        (CBuf::Binding(0), 0, "c[0x0][0x0]"),
        (CBuf::Binding(last_index), u16::MAX, "c[0x1f][0xffff]"),
        (
            CBuf::BindlessUniform(UniformReg::new(0).expect("ur0")),
            0,
            "cx[ur0][0x0]",
        ),
        (
            CBuf::BindlessUniform(UniformReg::URZ),
            u16::MAX,
            "cx[urz][0xffff]",
        ),
    ];
    for (buf, offset, text) in cases {
        let cbuf = CBufRef { buf, offset };
        assert_eq!(
            op_text(mov(r(1), MovSrc::CBuf(cbuf))),
            format!("mov r1, {text};")
        );
    }
    let past = CBufRef {
        buf: CBuf::Binding(last_index + 1),
        offset: 0,
    };
    assert_eq!(
        assemble(&Instr::new(mov(r(1), MovSrc::CBuf(past)))),
        Err(EncodeError::TooWide {
            what: "constant buffer index",
            value: 32,
            bits: CBUF_INDEX.width
        })
    );
}

#[test]
fn immediates_at_the_edges_of_their_encoding() {
    for (imm, text) in [(0, "0x0"), (u32::MAX, "0xffffffff")] {
        assert_eq!(
            op_text(mov(r(1), MovSrc::Imm(imm))),
            format!("mov r1, {text};")
        );
    }
}

/// Two's-complement fields have no decoded form yet (slices 6e and 6h), so the reader and
/// writer are checked directly at each end of the range and one past it.
#[test]
fn signed_fields_at_the_edges_of_their_range() {
    for width in [1, 24, 63, 64] {
        let field = Field::new(128 - width.max(24), width);
        let min = i64::MIN >> (64 - width);
        let max = i64::MAX >> (64 - width);
        for value in [min, -1, 0, max] {
            let mut writer = Writer::default();
            writer.signed(field, "offset", value).expect("in range");
            let mut reader = Reader::new(writer.word());
            assert_eq!(reader.signed(field), value, "width {width}");
            assert_eq!(reader.unread() & !COMMON, 0);
        }
        if width < 64 {
            for value in [min - 1, max + 1] {
                assert_eq!(
                    Writer::default().signed(field, "offset", value),
                    Err(EncodeError::OutOfRange {
                        what: "offset",
                        value,
                        bits: width
                    })
                );
            }
        }
    }
    assert_eq!(
        Writer::default()
            .signed(Field::new(40, 24), "offset", 1 << 23)
            .map_err(|error| error.to_string()),
        Err("offset 8388608 does not fit in 24 signed bits".into())
    );
    assert_eq!(
        Writer::default().signed(Field::new(40, 0), "offset", 0),
        Err(EncodeError::OutOfRange {
            what: "offset",
            value: 0,
            bits: 0
        })
    );
    assert_eq!(
        Reader::new(Instruction::from_bits(u128::MAX)).signed(Field::new(40, 0)),
        0
    );
}

/// A source modifier on an instruction without modifier bits is refused, naming the source.
#[test]
fn modifiers_without_bits_are_refused() {
    let modified = [
        AluSrc::Reg {
            reg: r(1),
            abs: true,
            neg: false,
        },
        AluSrc::UniformReg {
            reg: UniformReg::URZ,
            abs: false,
            neg: true,
        },
    ];
    for src in modified {
        for (index, srcs) in [(0, [Some(src), None, None]), (1, [None, Some(src), None])] {
            assert_eq!(
                Writer::default().alu("mov", integer::MOV_OPCODE, srcs, false),
                Err(EncodeError::Modifier {
                    mnemonic: "mov",
                    index
                })
            );
        }
    }
    assert_eq!(
        Writer::default()
            .alu(
                "mov",
                integer::MOV_OPCODE,
                [None, None, Some(modified[0])],
                false
            )
            .map_err(|error| error.to_string()),
        Err("source 2 of mov has a modifier the instruction cannot encode".into())
    );
    // The same sources are accepted where the instruction has modifier bits.
    assert!(
        Writer::default()
            .alu(
                "ffma",
                float::FFMA_OPCODE,
                [Some(modified[0]), Some(modified[1]), None],
                true
            )
            .is_ok()
    );
}

#[test]
fn every_operation_has_a_distinct_lower_case_mnemonic() {
    // The 104 instructions of slices 6a to 6i (`op`'s module docs).
    assert_eq!(MNEMONICS.len(), 23 + 15 + 10 + 2 + 8 + 9 + 12 + 21 + 4);
    let mut sorted = MNEMONICS.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), MNEMONICS.len());
    for mnemonic in MNEMONICS {
        assert_eq!(mnemonic.to_lowercase(), *mnemonic);
    }
    for decoded in DECODED {
        assert!(MNEMONICS.contains(&decoded), "{decoded}");
    }
}

/// Every operation that is not decoded is a shell: it prints its mnemonic and the assembler
/// refuses it.
#[test]
fn shells_print_their_mnemonic_and_have_no_encoding() {
    let shells = shells();
    assert_eq!(shells.len(), MNEMONICS.len() - DECODED.len());
    for op in shells {
        let mnemonic = op.mnemonic();
        assert!(!DECODED.contains(&mnemonic));
        assert_eq!(Instr::new(op).to_string(), format!("{mnemonic};"));
        let refused = assemble(&Instr::new(op));
        assert_eq!(refused, Err(EncodeError::NoEncoding { mnemonic }));
        assert_eq!(
            refused.map_err(|error| error.to_string()),
            Err(format!("the test assembler cannot encode {mnemonic} yet"))
        );
    }
}

#[test]
fn decoded_operations_name_their_mnemonic() {
    let exit = Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    });
    let s2r = Op::S2R(OpS2R {
        dst: r(0),
        sr: SpecialReg::LANE_ID,
    });
    assert_eq!(mov(r(0), MovSrc::Imm(0)).mnemonic(), "mov");
    assert_eq!(exit.mnemonic(), "exit");
    assert_eq!(s2r.mnemonic(), "s2r");
}
